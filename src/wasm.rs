#![allow(dead_code)]
// Every type here is constructed from JavaScript through
// `#[wasm_bindgen(constructor)]`, which is what `new()` is for. A `Default`
// impl would be unreachable from that side and meaningless on this one, so the
// bindings are exempt rather than carrying two dozen impls nobody calls.
#![allow(clippy::new_without_default)]
// The JS side has no keyword arguments and no struct literals, so a binding
// that configures a dozen things takes a dozen positional parameters. Bundling
// them into a Rust struct would only move the problem: the shim would have to
// unpack it again on the way in.
#![allow(clippy::too_many_arguments)]
//! The [`web/threejs-shim.js`](../../web/threejs-shim.js) companion maps these
//! types onto three.js r165-style `THREE.*` symbols so existing examples can run
//! against wasm/WebGPU with minimal changes.
//!
//! Bindings cover the renderer, scene graph, geometries, materials, lights,
//! textures, controls, loaders, post-processing passes, PMREM, and helpers.
//! Gaps shrink over time — see `tests/parity/` for coverage tracking.

use std::sync::Arc;
use wasm_bindgen::prelude::*;

use crate::core::ObjectId;

/// Top-level WebGPU renderer bound to a canvas element. The constructor is
/// async because adapter/device acquisition is async on the web.
#[wasm_bindgen]
pub struct WebRenderer {
    renderer: crate::Renderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    width: u32,
    height: u32,
    // When non-null, render() draws into this render target instead of the surface.
    current_target_id: Option<u32>,
    /// Scratch RT for kind-0 canvas copies (RT→RT→canvas avoids Dawn swapchain sampling glitch).
    copy_scratch: Option<(u32, Arc<crate::renderer::RenderTarget>)>,
}

// Thread-local registry mapping render-target IDs to live Arcs. We use this
// to look up the target from setRenderTarget() because wasm-bindgen's
// `Option<&WebRenderTarget>` can't safely hold an Arc across the JS boundary.
thread_local! {
    static ACTIVE_TARGETS: std::cell::RefCell<std::collections::HashMap<u32, std::sync::Arc<crate::renderer::RenderTarget>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
    static ACTIVE_CUBE_TARGETS: std::cell::RefCell<std::collections::HashMap<u32, std::sync::Arc<crate::renderer::CubeRenderTarget>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// GPU readback of a render-target region into a tight byte vec.
async fn read_texture_region(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, String> {
    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba16Float => 8u32,
        _ => 4u32,
    };
    const ALIGN: u32 = 256;
    let unpadded = w * bytes_per_pixel;
    let padded = unpadded.div_ceil(ALIGN) * ALIGN;
    let buf_size = (padded * h) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threers readback buffer sync"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers readback encoder sync"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
    }
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await
        .map_err(|_| "readback channel dropped".to_string())?
        .map_err(|e| format!("buffer map failed: {e:?}"))?;
    let mapped = buffer_slice.get_mapped_range().expect("buffer range is mapped");
    let mut out = Vec::with_capacity((w * h * bytes_per_pixel) as usize);
    for row in 0..h {
        let start = (row * padded) as usize;
        let end = start + (unpadded as usize);
        out.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    buffer.unmap();
    Ok(out)
}

fn postfx_camera_from(
    near: f32,
    far: f32,
    kernel_radius: f32,
    kernel_size: u32,
    proj: &[f32],
    inv_proj: &[f32],
) -> crate::renderer::PostFxCamera {
    let mut id = [0f32; 16];
    id[0] = 1.0;
    id[5] = 1.0;
    id[10] = 1.0;
    id[15] = 1.0;
    let mut p = id;
    let mut ip = id;
    if proj.len() >= 16 {
        p.copy_from_slice(&proj[..16]);
    }
    if inv_proj.len() >= 16 {
        ip.copy_from_slice(&inv_proj[..16]);
    }
    crate::renderer::PostFxCamera {
        near,
        far,
        kernel_radius,
        kernel_size,
        proj: p,
        inv_proj: ip,
    }
}

// Owns an offscreen render target. Reference-counted via Arc so multiple
// places (renderer + materials sampling the texture) can share it.
#[wasm_bindgen]
pub struct WebRenderTarget {
    pub(crate) inner: std::sync::Arc<crate::renderer::RenderTarget>,
    pub width: u32,
    pub height: u32,
    pub id: u32,
}

// A 6-face cube render target. Used by CubeCamera + scene.environment.
#[wasm_bindgen]
pub struct WebCubeRenderTarget {
    pub(crate) inner: std::sync::Arc<crate::renderer::CubeRenderTarget>,
    pub side: u32,
    pub id: u32,
}

#[wasm_bindgen]
impl WebCubeRenderTarget {
    #[wasm_bindgen(constructor)]
    pub fn new(renderer: &WebRenderer, side: u32) -> WebCubeRenderTarget {
        let rt =
            crate::renderer::RenderTarget::new_cube(&renderer.device, side, renderer.config.format);
        let arc = std::sync::Arc::new(rt);
        let id = NEXT_RT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Register so scene.environment_cube_rt can find this RT at render time.
        ACTIVE_CUBE_TARGETS.with(|m| m.borrow_mut().insert(id, arc.clone()));
        WebCubeRenderTarget {
            inner: arc,
            side,
            id,
        }
    }
}

static NEXT_RT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

// Reuse one WebGPU device/queue across renderer instances in the same tab.
// `requestAdapter` + `requestDevice` dominate cold-start cost; surfaces stay
// per-canvas. Thread-local because wasm's main thread is single-threaded and
// `wgpu::Device` is not `Sync`.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED_BROWSER_GPU: std::cell::RefCell<Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
async fn acquire_browser_gpu(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), JsValue> {
    if let Some(pair) = SHARED_BROWSER_GPU.with(|g| g.borrow().clone()) {
        return Ok(pair);
    }

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("no adapter: {e}")))?;

    // Request what the adapter actually offers rather than a floor.
    //
    // `downlevel_webgl2_defaults` allows 16 sampled textures per shader stage,
    // and the standard fragment layout — material maps, shadow maps, the
    // environment cube, and the screen-space capture — needs more than that once
    // a scene uses shadows or glass. WebGPU rejects the pipeline for exceeding
    // the limit, and it does so *asynchronously*: nothing throws, no call fails,
    // the pipeline simply never draws. What that looks like from the outside is
    // a black canvas at a full frame rate, which is a great deal harder to
    // diagnose than an error would have been.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {e:?}")))?;

    let pair = (Arc::new(device), Arc::new(queue));
    SHARED_BROWSER_GPU.with(|g| *g.borrow_mut() = Some(pair.clone()));
    Ok(pair)
}

#[wasm_bindgen]
impl WebRenderTarget {
    #[wasm_bindgen(constructor)]
    pub fn new(renderer: &WebRenderer, width: u32, height: u32) -> WebRenderTarget {
        Self::alloc(renderer, width, height, renderer.config.format)
    }

    /// Half-float color RT for outline intermediates (matches three.js OutlinePass).
    #[wasm_bindgen(js_name = newHalfFloat)]
    pub fn new_half_float(renderer: &WebRenderer, width: u32, height: u32) -> WebRenderTarget {
        Self::alloc(renderer, width, height, wgpu::TextureFormat::Rgba16Float)
    }

    fn alloc(
        renderer: &WebRenderer,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> WebRenderTarget {
        let rt = crate::renderer::RenderTarget::new(&renderer.device, width, height, format);
        let arc = std::sync::Arc::new(rt);
        let id = NEXT_RT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ACTIVE_TARGETS.with(|m| m.borrow_mut().insert(id, arc.clone()));
        WebRenderTarget {
            inner: arc,
            width,
            height,
            id,
        }
    }

    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, _w: u32, _h: u32) {
        // Resize requires reallocating the wgpu texture. Users typically
        // create a new render target at the new size; we report the request
        // but keep the original allocation.
    }
}

/// Prefer alpha compositing modes that preserve a transparent canvas.
fn pick_surface_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> wgpu::CompositeAlphaMode {
    for preferred in [
        wgpu::CompositeAlphaMode::PreMultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied,
        wgpu::CompositeAlphaMode::Inherit,
        wgpu::CompositeAlphaMode::Auto,
    ] {
        if modes.contains(&preferred) {
            return preferred;
        }
    }
    modes[0]
}

#[wasm_bindgen]
impl WebRenderer {
    /// Async factory. Pass an HTMLCanvasElement and the renderer attaches to it.
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        console_error_panic_hook::set_once();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = {
            // wgpu 30 dropped `Default` here; the display handle is only
            // consulted by GLES/Wayland, not Vulkan, Metal or DX12.
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::BROWSER_WEBGPU;
            wgpu::Instance::new(d)
        };

        let target = wgpu::SurfaceTarget::Canvas(canvas);
        let surface = instance
            .create_surface(target)
            .map_err(|e| JsValue::from_str(&format!("surface: {e:?}")))?;

        let (device, queue) = acquire_browser_gpu(&instance, &surface).await?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("no adapter: {e}")))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: pick_surface_alpha_mode(&surface_caps.alpha_modes),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &config);

        let renderer =
            crate::Renderer::new(device.clone(), queue.clone(), surface_format, width, height);
        // Continue construction below — see existing return statement.

        Ok(WebRenderer {
            renderer,
            surface,
            config,
            device,
            width,
            height,
            current_target_id: None,
            copy_scratch: None,
        })
    }

    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, w: u32, h: u32) {
        self.width = w.max(1);
        self.height = h.max(1);
        self.config.width = self.width;
        self.config.height = self.height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(self.width, self.height);
        self.copy_scratch = None;
    }

    /// Enable hardware MSAA on the direct-to-canvas pass. `samples <= 1` = off,
    /// else 4×. Only the opaque forward pass is multisampled — render targets,
    /// shadows, and glass/OIT/refraction scenes stay single-sampled.
    /// three.js `renderer.toneMapping` / `renderer.toneMappingExposure`.
    ///
    /// Takes three.js's numeric constants: `0` NoToneMapping, `1` Linear,
    /// `4` ACESFilmic. Anything else falls back to linear, which is closer to
    /// the intent than clipping.
    #[wasm_bindgen(js_name = setToneMapping)]
    pub fn set_tone_mapping(&mut self, mode: u32, exposure: f32) {
        let mapping = match mode {
            0 => crate::ToneMapping::None,
            4 => crate::ToneMapping::AcesFilmic,
            _ => crate::ToneMapping::Linear,
        };
        self.renderer.set_tone_mapping(mapping, exposure);
    }

    #[wasm_bindgen(js_name = setMsaa)]
    pub fn set_msaa(&mut self, samples: u32) {
        self.renderer.set_msaa(samples);
    }

    fn ensure_copy_scratch(&mut self) -> (u32, Arc<crate::renderer::RenderTarget>) {
        let needs_alloc = match &self.copy_scratch {
            None => true,
            Some((_, rt)) => rt.width != self.width || rt.height != self.height,
        };
        if needs_alloc {
            let rt = Arc::new(crate::renderer::RenderTarget::new(
                &self.device,
                self.width,
                self.height,
                self.config.format,
            ));
            let id = NEXT_RT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ACTIVE_TARGETS.with(|m| m.borrow_mut().insert(id, rt.clone()));
            self.renderer.register_render_target(id, &rt);
            self.copy_scratch = Some((id, rt));
        }
        let (id, rt) = self.copy_scratch.as_ref().unwrap();
        (*id, rt.clone())
    }

    /// Set the active render target by id. When non-zero, `render()` draws
    /// into the target's offscreen texture; pass 0 to restore canvas rendering.
    /// We pass the id (not the WebRenderTarget instance) so the JS-side
    /// wrapper isn't moved into Rust ownership and remains usable for
    /// subsequent calls like `readRenderTargetPixels`.
    #[wasm_bindgen(js_name = setRenderTarget)]
    pub fn set_render_target(&mut self, target_id: u32) {
        if target_id == 0 {
            // Just clear the active-target pointer; leave the RT in ACTIVE_TARGETS
            // so subsequent calls like applyPostFx and readRenderTargetPixels
            // can still find it by id.
            self.current_target_id = None;
            return;
        }
        if let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&target_id).cloned()) {
            self.current_target_id = Some(target_id);
            self.renderer.register_render_target(target_id, &target);
        }
    }

    /// Build a Texture that samples from the given render target's color view.
    /// Used to chain post-processing passes: render scene into RT, then use
    /// that RT as a material map on a fullscreen quad mesh.
    #[wasm_bindgen(js_name = renderTargetTexture)]
    pub fn render_target_texture(&mut self, rt: &WebRenderTarget) -> WebTexture {
        // Make sure the renderer is aware of this RT's view (in case the user
        // calls renderTargetTexture before ever setRenderTarget).
        self.renderer.register_render_target(rt.id, &rt.inner);
        // Whatever the surface format is, the RT was created with the same
        // format; on the crate side we treat it as sRGB (matches WebGL default).
        let t = crate::Texture::from_render_target(
            rt.id,
            rt.width,
            rt.height,
            crate::textures::TextureFormat::Rgba8UnormSrgb,
        );
        WebTexture {
            inner: std::sync::Arc::new(t),
        }
    }

    /// Read pixels from a render target. Returns a Promise<Uint8Array> of RGBA
    /// bytes in row-major order (4 bytes per pixel). The copy is encoded into
    /// a staging buffer; the buffer is mapped asynchronously and the bytes
    /// returned to JS once the GPU has finished.
    #[wasm_bindgen(js_name = readRenderTargetPixels)]
    pub fn read_render_target_pixels(
        &self,
        target_id: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> js_sys::Promise {
        let device = self.renderer.device_arc();
        let queue = self.renderer.queue_arc();
        let target = match ACTIVE_TARGETS.with(|m| m.borrow().get(&target_id).cloned()) {
            Some(t) => t,
            None => return js_sys::Promise::reject(&JsValue::from_str("unknown render target id")),
        };
        // wgpu requires the row byte count to be a multiple of 256.
        const ALIGN: u32 = 256;
        let bytes_per_pixel = 4u32;
        let unpadded = w * bytes_per_pixel;
        let padded = unpadded.div_ceil(ALIGN) * ALIGN;
        let buf_size = (padded * h) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers readback buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers readback encoder"),
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target.color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(std::iter::once(encoder.finish()));
        }

        wasm_bindgen_futures::future_to_promise(async move {
            // Map the buffer for reading. wgpu uses a callback; bridge with a oneshot.
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = futures_channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            // Pump the device until the map completes. On the web backend this is
            // a no-op (mapping is satisfied by the browser when the queue drains).
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.await
                .map_err(|_| JsValue::from_str("readback channel dropped"))?
                .map_err(|e| JsValue::from_str(&format!("buffer map failed: {e:?}")))?;
            // Copy the mapped (possibly padded) range into a tight RGBA byte vec.
            let mapped = buffer_slice.get_mapped_range().expect("buffer range is mapped");
            let mut out = Vec::with_capacity((w * h * bytes_per_pixel) as usize);
            for row in 0..h {
                let start = (row * padded) as usize;
                let end = start + (unpadded as usize);
                out.extend_from_slice(&mapped[start..end]);
            }
            drop(mapped);
            buffer.unmap();
            let u8arr = js_sys::Uint8Array::new_with_length(out.len() as u32);
            u8arr.copy_from(&out);
            Ok(JsValue::from(u8arr))
        })
    }

    /// Apply a post-fx pass: read input render target, write to canvas.
    /// `kind`: 0=copy … 11=ssao, 12=ssr, 13=ssao-blur, 14=ssao-composite.
    /// `depth_rt_id` / `normal_rt_id`: optional prepass RTs (0 = fallback).
    /// For SSAO (kind 11), pass camera near/far, kernel size, and projection
    /// matrices via the trailing arguments.
    #[wasm_bindgen(js_name = applyPostFx)]
    pub fn apply_post_fx(
        &mut self,
        input_rt_id: u32,
        depth_rt_id: u32,
        normal_rt_id: u32,
        kind: u32,
        time: f32,
        p2x: f32,
        p2y: f32,
        p2z: f32,
        p2w: f32,
        p3x: f32,
        p3y: f32,
        p3z: f32,
        p3w: f32,
        additive: u32,
        cam_near: f32,
        cam_far: f32,
        kernel_radius: f32,
        kernel_size: u32,
        proj: Vec<f32>,
        inv_proj: Vec<f32>,
    ) {
        let input = match ACTIVE_TARGETS.with(|m| m.borrow().get(&input_rt_id).cloned()) {
            Some(t) => t,
            None => return,
        };
        self.renderer.register_render_target(input_rt_id, &input);
        if depth_rt_id != 0 {
            if let Some(depth) = ACTIVE_TARGETS.with(|m| m.borrow().get(&depth_rt_id).cloned()) {
                self.renderer.register_render_target(depth_rt_id, &depth);
            }
        }
        if normal_rt_id != 0 {
            if let Some(normal) = ACTIVE_TARGETS.with(|m| m.borrow().get(&normal_rt_id).cloned()) {
                self.renderer.register_render_target(normal_rt_id, &normal);
            }
        }
        let camera = postfx_camera_from(
            cam_near,
            cam_far,
            kernel_radius,
            kernel_size,
            &proj,
            &inv_proj,
        );
        // Dawn/WebGPU: sampling a render target in the same frame it was written,
        // then writing straight to the swapchain, misaligns at silhouettes. An
        // intermediate RT→RT copy matches the stable RT→RT→canvas path.
        let mut input_rt_id = input_rt_id;
        if kind == 0 && additive == 0 && p2x > 0.5 {
            let (scratch_id, scratch) = self.ensure_copy_scratch();
            self.renderer.register_render_target(scratch_id, &scratch);
            self.renderer.apply_postfx_by_id(
                input_rt_id,
                scratch_id,
                &scratch.color_view,
                depth_rt_id,
                normal_rt_id,
                kind,
                time,
                [p2x, p2y, p2z, p2w],
                [p3x, p3y, p3z, p3w],
                scratch.width,
                scratch.height,
                false,
                camera.clone(),
            );
            input_rt_id = scratch_id;
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                }
            }
        };
        let output_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.apply_postfx_by_id(
            input_rt_id,
            0,
            &output_view,
            depth_rt_id,
            normal_rt_id,
            kind,
            time,
            [p2x, p2y, p2z, p2w],
            [p3x, p3y, p3z, p3w],
            self.width,
            self.height,
            additive != 0,
            camera,
        );
        self.renderer.queue_arc().present(frame);
    }

    /// Apply a post-fx pass writing into a render target (not canvas).
    #[wasm_bindgen(js_name = applyPostFxToRT)]
    pub fn apply_post_fx_to_rt(
        &mut self,
        input_rt_id: u32,
        output_rt_id: u32,
        depth_rt_id: u32,
        normal_rt_id: u32,
        kind: u32,
        time: f32,
        p2x: f32,
        p2y: f32,
        p2z: f32,
        p2w: f32,
        p3x: f32,
        p3y: f32,
        p3z: f32,
        p3w: f32,
        cam_near: f32,
        cam_far: f32,
        kernel_radius: f32,
        kernel_size: u32,
        proj: Vec<f32>,
        inv_proj: Vec<f32>,
    ) {
        let input = match ACTIVE_TARGETS.with(|m| m.borrow().get(&input_rt_id).cloned()) {
            Some(t) => t,
            None => return,
        };
        let output = match ACTIVE_TARGETS.with(|m| m.borrow().get(&output_rt_id).cloned()) {
            Some(t) => t,
            None => return,
        };
        self.renderer.register_render_target(input_rt_id, &input);
        self.renderer.register_render_target(output_rt_id, &output);
        if depth_rt_id != 0 {
            if let Some(depth) = ACTIVE_TARGETS.with(|m| m.borrow().get(&depth_rt_id).cloned()) {
                self.renderer.register_render_target(depth_rt_id, &depth);
            }
        }
        if normal_rt_id != 0 {
            if let Some(normal) = ACTIVE_TARGETS.with(|m| m.borrow().get(&normal_rt_id).cloned()) {
                self.renderer.register_render_target(normal_rt_id, &normal);
            }
        }
        let camera = postfx_camera_from(
            cam_near,
            cam_far,
            kernel_radius,
            kernel_size,
            &proj,
            &inv_proj,
        );
        self.renderer.apply_postfx_by_id(
            input_rt_id,
            output_rt_id,
            &output.color_view,
            depth_rt_id,
            normal_rt_id,
            kind,
            time,
            [p2x, p2y, p2z, p2w],
            [p3x, p3y, p3z, p3w],
            output.width,
            output.height,
            false,
            camera,
        );
    }

    #[wasm_bindgen(js_name = setGlitchSnow)]
    pub fn set_glitch_snow(&mut self, data: Vec<f32>, width: u32, height: u32) {
        self.renderer.set_glitch_snow(&data, width, height);
    }

    /// Async readback of a half-float render target (raw RGBA16F bytes).
    #[wasm_bindgen(js_name = readRenderTargetF16)]
    pub fn read_render_target_f16(
        &self,
        target_id: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> js_sys::Promise {
        let device = self.device.clone();
        let queue = self.renderer.queue_arc();
        let target = ACTIVE_TARGETS.with(|m| m.borrow().get(&target_id).cloned());
        wasm_bindgen_futures::future_to_promise(async move {
            let target = target.ok_or_else(|| JsValue::from_str("unknown render target id"))?;
            let bytes = read_texture_region(
                &device,
                &queue,
                &target.color_texture,
                target.format,
                x,
                y,
                w,
                h,
            )
            .await
            .map_err(|e| JsValue::from_str(&e))?;
            let arr = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
            arr.copy_from(&bytes);
            Ok(JsValue::from(arr))
        })
    }

    /// Present tightly-packed RGBA8 bytes (top-first rows) to the canvas surface.
    #[wasm_bindgen(js_name = blitRgba8ToCanvas)]
    pub fn blit_rgba8_to_canvas(&mut self, data: Vec<u8>, width: u32, height: u32) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                }
            }
        };
        let output_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .blit_rgba8(&data, width, height, &output_view, self.config.format);
        self.renderer.queue_arc().present(frame);
    }

    #[wasm_bindgen(js_name = setDotscreenPattern)]
    pub fn set_dotscreen_pattern(&mut self, data: Vec<f32>, width: u32, height: u32) {
        self.renderer.set_dotscreen_pattern(&data, width, height);
    }

    #[wasm_bindgen(js_name = setGlitchDisp)]
    pub fn set_glitch_disp(&mut self, data: Vec<f32>, size: u32) {
        self.renderer.set_glitch_disp(&data, size);
    }

    /// Upload SSAO hemisphere kernel (96 floats: 32×vec3, padded to vec4 on GPU).
    #[wasm_bindgen(js_name = setSsaoKernel)]
    pub fn set_ssao_kernel(&mut self, kernel: &[f32]) {
        self.renderer.set_ssao_kernel(kernel);
    }

    /// Upload 4×4 SSAO rotation noise (16 floats, R32Float texture).
    #[wasm_bindgen(js_name = setSsaoNoise)]
    pub fn set_ssao_noise(&mut self, noise: &[f32]) {
        self.renderer.set_ssao_noise(noise);
    }

    /// Render one face of a cube render target. CubeCamera.update() calls
    /// this six times with face indices 0..6 and the matching face camera.
    #[wasm_bindgen(js_name = renderToCubeFace)]
    pub fn render_to_cube_face(
        &mut self,
        scene: &mut WebScene,
        camera: &WebCamera,
        target: &WebCubeRenderTarget,
        face: u32,
    ) {
        match &camera.inner {
            CameraInner::Perspective(c) => {
                self.renderer
                    .render_to_cube_face(&mut scene.inner, c, &target.inner, face as usize)
            }
            CameraInner::Orthographic(c) => {
                self.renderer
                    .render_to_cube_face(&mut scene.inner, c, &target.inner, face as usize)
            }
        }
    }

    /// Render, then blend `overlay`'s caption for `time` seconds over the frame.
    ///
    /// The caption is composited on the GPU before the frame is presented, so
    /// it lands on top of the 3D image without a second canvas. Sizing follows
    /// the canvas automatically.
    #[cfg(feature = "captions")]
    #[wasm_bindgen(js_name = renderWithCaptions)]
    pub fn render_with_captions(
        &mut self,
        scene: &mut WebScene,
        camera: &WebCamera,
        overlay: &mut WebCaptionOverlay,
        time: f64,
    ) {
        if let Some(id) = self.current_target_id {
            if let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned()) {
                match &camera.inner {
                    CameraInner::Perspective(c) => {
                        self.renderer.render_to(&mut scene.inner, c, &target)
                    }
                    CameraInner::Orthographic(c) => {
                        self.renderer.render_to(&mut scene.inner, c, &target)
                    }
                }
                self.renderer.draw_caption_overlay(
                    &mut overlay.inner,
                    time,
                    target.width,
                    target.height,
                    &target.color_view,
                    target.format,
                );
                return;
            }
        }
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                match &camera.inner {
                    CameraInner::Perspective(c) => {
                        self.renderer.render(&mut scene.inner, c, &view, false)
                    }
                    CameraInner::Orthographic(c) => {
                        self.renderer.render(&mut scene.inner, c, &view, false)
                    }
                }
                self.renderer.draw_caption_overlay(
                    &mut overlay.inner,
                    time,
                    self.width,
                    self.height,
                    &view,
                    self.config.format,
                );
                self.renderer.queue_arc().present(frame);
            }
            _ => {
                self.surface.configure(&self.device, &self.config);
            }
        }
    }

    pub fn render(&mut self, scene: &mut WebScene, camera: &WebCamera) {
        // If the scene references a cube render target as its environment map,
        // make sure the renderer's view cache has its cube view registered.
        if let Some(env_rt_id) = scene.inner.environment_cube_rt {
            if let Some(target) = ACTIVE_CUBE_TARGETS.with(|m| m.borrow().get(&env_rt_id).cloned())
            {
                self.renderer
                    .register_cube_render_target(env_rt_id, &target);
            }
        }
        // If a render target is set, render into it (no canvas presentation).
        if let Some(id) = self.current_target_id {
            if let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned()) {
                match &camera.inner {
                    CameraInner::Perspective(c) => {
                        self.renderer.render_to(&mut scene.inner, c, &target)
                    }
                    CameraInner::Orthographic(c) => {
                        self.renderer.render_to(&mut scene.inner, c, &target)
                    }
                }
                return;
            }
        }
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                match &camera.inner {
                    CameraInner::Perspective(c) => {
                        self.renderer.render(&mut scene.inner, c, &view, false)
                    }
                    CameraInner::Orthographic(c) => {
                        self.renderer.render(&mut scene.inner, c, &view, false)
                    }
                }
                self.renderer.queue_arc().present(frame);
            }
            _ => {
                self.surface.configure(&self.device, &self.config);
            }
        }
    }

    /// Device and queue for sharing with [`crate::wasm_pathtrace::WebPathTracer`].
    #[cfg(feature = "raytrace")]
    pub(crate) fn gpu_device_queue(&self) -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        (
            std::sync::Arc::clone(&self.device),
            self.renderer.queue_arc(),
        )
    }
}

/// JS-visible scene wrapper. Returns a `Mesh` handle from `add(mesh)` so JS
/// can keep a reference for subsequent transform updates.
#[wasm_bindgen]
pub struct WebScene {
    inner: crate::Scene,
}

#[wasm_bindgen]
impl WebScene {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebScene {
        WebScene {
            inner: crate::Scene::new(),
        }
    }

    /// Add a mesh to the scene. Returns its `ObjectId` (wrapped as `WebObjectHandle`).
    pub fn add(&mut self, mesh: &WebMesh) -> WebObjectHandle {
        let obj = crate::core::Object3D::mesh(crate::core::Mesh::from_arc(
            mesh.geometry.clone(),
            mesh.material.clone(),
        ));
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Point a scene mesh at the current `WebMaterial` Arc (after `setColor` etc.).
    #[wasm_bindgen(js_name = setMeshMaterial)]
    pub fn set_mesh_material(&mut self, handle: &WebObjectHandle, mat: &WebMaterial) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            if let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind {
                mesh.material = mat.inner.clone();
            }
        }
    }

    /// Query whether a LineSegments object's geometry has a named attribute.
    #[wasm_bindgen(js_name = lineGeometryHasAttr)]
    pub fn line_geometry_has_attr(&self, handle: &WebObjectHandle, name: &str) -> bool {
        self.inner
            .get(handle.id)
            .and_then(|obj| match &obj.kind {
                crate::core::ObjectKind::LineSegments(ls) => {
                    ls.geometry.get_attribute(name).map(|_| ())
                }
                _ => None,
            })
            .is_some()
    }

    #[wasm_bindgen(js_name = lineGeometryUvX)]
    pub fn line_geometry_uv_x(&self, handle: &WebObjectHandle, vert: u32) -> f32 {
        self.inner
            .get(handle.id)
            .and_then(|obj| match &obj.kind {
                crate::core::ObjectKind::LineSegments(ls) => {
                    ls.geometry.get_attribute("uv").and_then(|u| {
                        let i = vert as usize * 2;
                        u.array.get(i).copied()
                    })
                }
                _ => None,
            })
            .unwrap_or(0.0)
    }

    #[wasm_bindgen(js_name = lineMaterialDashSize)]
    pub fn line_material_dash_size(&self, handle: &WebObjectHandle) -> f32 {
        self.inner
            .get(handle.id)
            .and_then(|obj| match &obj.kind {
                crate::core::ObjectKind::LineSegments(ls) => match &*ls.material {
                    crate::Material::Line(m) => Some(m.dash_size),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or(0.0)
    }

    /// Replace a LineSegments object's geometry after JS-side attribute edits
    /// (e.g. `computeLineDistances()` after `scene.add()`).
    #[wasm_bindgen(js_name = syncLineGeometry)]
    pub fn sync_line_geometry(&mut self, handle: &WebObjectHandle, geom: &WebBufferGeometry) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            if let crate::core::ObjectKind::LineSegments(ls) = &mut obj.kind {
                ls.geometry = geom.inner.clone();
            }
        }
    }

    /// Add a LineSegments primitive (geometry interpreted as line-list).
    #[wasm_bindgen(js_name = addLineSegments)]
    pub fn add_line_segments(
        &mut self,
        geom: &WebBufferGeometry,
        mat: &WebMaterial,
    ) -> WebObjectHandle {
        let ls = crate::core::LineSegments::from_arc(geom.inner.clone(), mat.inner.clone());
        let obj = crate::core::Object3D::line_segments(ls);
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add a Sprite (billboard) at the scene root. The renderer expands the
    /// material into a camera-facing quad on the fly.
    #[wasm_bindgen(js_name = addSprite)]
    pub fn add_sprite(&mut self, mat: &WebMaterial) -> WebObjectHandle {
        let s = crate::core::Sprite::new((*mat.inner).clone());
        let obj = crate::core::Object3D::sprite(s);
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add a SkinnedMesh. We default the skeleton to `bone_count` identity
    /// transforms — the mesh renders as a regular Mesh until bone matrices
    /// get updated (a hook can be added later to drive bones each frame).
    #[wasm_bindgen(js_name = addSkinnedMesh)]
    pub fn add_skinned_mesh(
        &mut self,
        geom: &WebGeometry,
        mat: &WebMaterial,
        bone_count: usize,
    ) -> WebObjectHandle {
        let skeleton = crate::core::Skeleton {
            bones: Vec::new(),
            bone_matrices: vec![crate::math::Matrix4::identity(); bone_count.max(1)],
        };
        let sm = crate::core::SkinnedMesh::from_unweighted(
            (*geom.inner).clone(),
            (*mat.inner).clone(),
            skeleton,
        );
        let obj = crate::core::Object3D::skinned_mesh(sm);
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add an InstancedMesh — a geometry rendered N times with per-instance
    /// transforms, each entry is a column-major mat4 packed as 16 f32s.
    #[wasm_bindgen(js_name = addInstancedMesh)]
    pub fn add_instanced_mesh(
        &mut self,
        geom: &WebGeometry,
        mat: &WebMaterial,
        transforms: Vec<f32>,
    ) -> WebObjectHandle {
        let count = transforms.len() / 16;
        let mut im =
            crate::core::InstancedMesh::new((*geom.inner).clone(), (*mat.inner).clone(), count);
        for i in 0..count {
            let mut e = [0.0f32; 16];
            e.copy_from_slice(&transforms[i * 16..(i + 1) * 16]);
            im.set_matrix_at(i, crate::math::Matrix4 { elements: e });
        }
        let obj = crate::core::Object3D::instanced_mesh(im);
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add a Points primitive (geometry interpreted as point-list).
    #[wasm_bindgen(js_name = addPoints)]
    pub fn add_points(&mut self, geom: &WebBufferGeometry, mat: &WebMaterial) -> WebObjectHandle {
        let p = crate::core::Points {
            geometry: geom.inner.clone(),
            material: mat.inner.clone(),
        };
        let obj = crate::core::Object3D::points(p);
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add an empty Group node (Object3D with no kind). Returns its handle so
    /// children can be parented under it via `addMeshTo` / `addGroupTo`.
    #[wasm_bindgen(js_name = addGroup)]
    pub fn add_group(&mut self) -> WebObjectHandle {
        let obj = crate::core::Object3D::group();
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }

    /// Add a mesh under the given parent (a Group's handle). Returns the
    /// mesh's own handle for further transform updates.
    #[wasm_bindgen(js_name = addMeshTo)]
    pub fn add_mesh_to(&mut self, parent: &WebObjectHandle, mesh: &WebMesh) -> WebObjectHandle {
        let obj = crate::core::Object3D::mesh(crate::core::Mesh::from_arc(
            mesh.geometry.clone(),
            mesh.material.clone(),
        ));
        let id = self.inner.add_to(parent.id, obj);
        WebObjectHandle { id }
    }

    /// Add a light source.
    #[wasm_bindgen(js_name = addLight)]
    pub fn add_light(&mut self, light: &WebLight) -> WebObjectHandle {
        let id = match &light.inner {
            LightInner::Ambient(l) => self.inner.add_light(*l),
            LightInner::Directional(l) => self.inner.add_light(*l),
            LightInner::Point(l) => self.inner.add_light(*l),
            LightInner::Spot(l) => self.inner.add_light(*l),
            LightInner::Hemisphere(l) => self.inner.add_light(*l),
            LightInner::RectArea(l) => self.inner.add_light(*l),
        };
        WebObjectHandle { id }
    }

    /// Replace a light already in the scene with the wrapper's current state.
    ///
    /// `addLight` copies the light in, so every setter on the `WebLight`
    /// afterwards writes to a handle the renderer never reads again — moving a
    /// light, or changing its colour or intensity, silently did nothing once it
    /// had been added. This pushes the wrapper's state back over the scene's
    /// copy; the shim calls it from every light setter.
    #[wasm_bindgen(js_name = updateLight)]
    pub fn update_light(&mut self, handle: &WebObjectHandle, light: &WebLight) {
        let Some(obj) = self.inner.get_mut(handle.id) else {
            return;
        };
        let replacement: crate::lights::Light = match &light.inner {
            LightInner::Ambient(l) => (*l).into(),
            LightInner::Directional(l) => (*l).into(),
            LightInner::Point(l) => (*l).into(),
            LightInner::Spot(l) => (*l).into(),
            LightInner::Hemisphere(l) => (*l).into(),
            LightInner::RectArea(l) => (*l).into(),
        };
        if let crate::ObjectKind::Light(slot) = &mut obj.kind {
            *slot = replacement;
        }
    }

    /// Diagnostic — for each light in the scene, print its world position and
    /// (for spot/directional) direction, after update_world is called.
    #[wasm_bindgen(js_name = dumpLights)]
    pub fn dump_lights(&mut self) -> String {
        self.inner.update_world();
        let mut out = String::new();
        let root = self.inner.root;
        self.inner.arena.traverse_visible(root, &mut |_id, obj| {
            if let crate::ObjectKind::Light(l) = &obj.kind {
                let p = obj.world_position();
                let name = match l {
                    crate::Light::Ambient(_) => "Ambient",
                    crate::Light::Directional(_) => "Directional",
                    crate::Light::Point(_) => "Point",
                    crate::Light::Spot(_) => "Spot",
                    crate::Light::Hemisphere(_) => "Hemisphere",
                    crate::Light::RectArea(_) => "RectArea",
                };
                out.push_str(&format!(
                    "{name} world_pos=({:.3},{:.3},{:.3})\n",
                    p.x, p.y, p.z
                ));
            }
        });
        out
    }

    /// Diagnostic — count lights of each kind currently in this scene.
    #[wasm_bindgen(js_name = lightCounts)]
    pub fn light_counts(&self) -> String {
        let mut n_amb = 0;
        let mut n_dir = 0;
        let mut n_point = 0;
        let mut n_spot = 0;
        let mut n_hemi = 0;
        let root = self.inner.root;
        self.inner.arena.traverse_visible(root, &mut |_id, obj| {
            if let crate::ObjectKind::Light(l) = &obj.kind {
                match l {
                    crate::Light::Ambient(_) => n_amb += 1,
                    crate::Light::Directional(_) => n_dir += 1,
                    crate::Light::Point(_) => n_point += 1,
                    crate::Light::Spot(s) => {
                        n_spot += 1;
                        let _ = s; // intensity available for richer dumps later
                    }
                    crate::Light::Hemisphere(_) => n_hemi += 1,
                    crate::Light::RectArea(_) => {}
                }
            }
        });
        format!("amb={n_amb} dir={n_dir} pt={n_point} sp={n_spot} hemi={n_hemi}")
    }

    /// Configure fog. `mode`: 0 = off, 1 = linear, 2 = exp2. `near`/`far` are
    /// used for linear; `density` is used for exp2.
    #[wasm_bindgen(js_name = setFog)]
    pub fn set_fog(&mut self, color: &WebColor, near: f32, far: f32, density: f32, mode: u32) {
        self.inner.fog = crate::scene::FogParams {
            color: color.inner,
            near,
            far,
            density,
            mode,
        };
    }

    /// Bind a cube render target as this scene's environment map. Pass `0`
    /// to clear and fall back to the CPU-side `environment` cubemap.
    #[wasm_bindgen(js_name = setEnvironmentCube)]
    pub fn set_environment_cube(&mut self, rt_id: u32) {
        self.inner.environment_cube_rt = if rt_id == 0 { None } else { Some(rt_id) };
    }

    /// Bind a CPU-side cubemap (optionally PMREM-filtered) as the environment.
    #[wasm_bindgen(js_name = setEnvironmentMap)]
    pub fn set_environment_map(&mut self, cube: &WebCubeTexture) {
        self.inner.environment = Some(Arc::clone(&cube.inner));
        self.inner.environment_cube_rt = None;
    }

    /// Set the background color (clear color).
    #[wasm_bindgen(setter, js_name = background)]
    pub fn set_background(&mut self, color: &WebColor) {
        self.inner.background = color.inner;
    }

    /// Clear alpha for the framebuffer (0 = transparent canvas over HTML video).
    #[wasm_bindgen(js_name = setBackgroundAlpha)]
    pub fn set_background_alpha(&mut self, alpha: f32) {
        self.inner.background_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Update an object's transform (position + quaternion).
    #[wasm_bindgen(js_name = setTransform)]
    pub fn set_transform(&mut self, handle: &WebObjectHandle, pos: &WebVector3, rot: &WebEuler) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.position = pos.inner;
            obj.quaternion = rot.inner.to_quaternion();
        }
    }

    /// three.js `Object3D.castShadow`.
    #[wasm_bindgen(js_name = setObjectCastShadow)]
    pub fn set_object_cast_shadow(&mut self, handle: &WebObjectHandle, cast: bool) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.cast_shadow = cast;
        }
    }

    /// three.js `Object3D.receiveShadow`.
    #[wasm_bindgen(js_name = setObjectReceiveShadow)]
    pub fn set_object_receive_shadow(&mut self, handle: &WebObjectHandle, receive: bool) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.receive_shadow = receive;
        }
    }
}

#[wasm_bindgen]
pub struct WebObjectHandle {
    id: ObjectId,
}

/// JS-visible camera that wraps either a perspective or orthographic camera.
#[wasm_bindgen]
pub struct WebCamera {
    inner: CameraInner,
}

enum CameraInner {
    Perspective(crate::PerspectiveCamera),
    Orthographic(crate::OrthographicCamera),
}

#[wasm_bindgen]
impl WebCamera {
    #[wasm_bindgen(js_name = perspective)]
    pub fn perspective(fov_deg: f32, aspect: f32, near: f32, far: f32) -> WebCamera {
        WebCamera {
            inner: CameraInner::Perspective(crate::PerspectiveCamera::new(
                fov_deg, aspect, near, far,
            )),
        }
    }

    #[wasm_bindgen(js_name = orthographic)]
    pub fn orthographic(
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
    ) -> WebCamera {
        WebCamera {
            inner: CameraInner::Orthographic(crate::OrthographicCamera::new(
                left, right, top, bottom, near, far,
            )),
        }
    }

    #[wasm_bindgen(js_name = setPosition)]
    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        match &mut self.inner {
            CameraInner::Perspective(c) => c.position = crate::Vector3::new(x, y, z),
            CameraInner::Orthographic(c) => c.position = crate::Vector3::new(x, y, z),
        }
    }

    #[wasm_bindgen(js_name = readPosition)]
    pub fn read_position(&self) -> WebVector3 {
        let p = match &self.inner {
            CameraInner::Perspective(c) => c.position,
            CameraInner::Orthographic(c) => c.position,
        };
        WebVector3::new(p.x, p.y, p.z)
    }

    #[wasm_bindgen(js_name = readTarget)]
    pub fn read_target(&self) -> WebVector3 {
        let t = match &self.inner {
            CameraInner::Perspective(c) => c.target,
            CameraInner::Orthographic(c) => c.target,
        };
        WebVector3::new(t.x, t.y, t.z)
    }

    #[wasm_bindgen(js_name = lookAt)]
    pub fn look_at(&mut self, x: f32, y: f32, z: f32) {
        match &mut self.inner {
            CameraInner::Perspective(c) => {
                c.look_at(crate::Vector3::new(x, y, z));
            }
            CameraInner::Orthographic(c) => {
                c.target = crate::Vector3::new(x, y, z);
            }
        }
    }

    /// Atomic eye + look-at (+ optional FOV) update for cinematic animation.
    /// Pass `fov_deg < 0` to leave FOV unchanged. One wasm call per frame
    /// instead of separate `setPosition` / `lookAt` / `setFov`.
    #[wasm_bindgen(js_name = setView)]
    pub fn set_view(
        &mut self,
        px: f32,
        py: f32,
        pz: f32,
        tx: f32,
        ty: f32,
        tz: f32,
        fov_deg: f32,
    ) {
        let eye = crate::Vector3::new(px, py, pz);
        let target = crate::Vector3::new(tx, ty, tz);
        match &mut self.inner {
            CameraInner::Perspective(c) => {
                c.position = eye;
                c.look_at(target);
                if fov_deg.is_finite() && fov_deg >= 0.0 {
                    c.fov = fov_deg.to_radians();
                }
            }
            CameraInner::Orthographic(c) => {
                c.position = eye;
                c.target = target;
            }
        }
    }

    /// Set the camera's up vector. Needed by CubeCamera face cameras whose
    /// +Y/-Y views look straight up/down — with the default up of (0,1,0)
    /// those views become degenerate (lookAt parallel to up).
    #[wasm_bindgen(js_name = setUp)]
    pub fn set_up(&mut self, x: f32, y: f32, z: f32) {
        let v = crate::Vector3::new(x, y, z);
        match &mut self.inner {
            CameraInner::Perspective(c) => {
                c.up = v;
            }
            CameraInner::Orthographic(c) => {
                c.up = v;
            }
        }
    }

    /// three.js `PerspectiveCamera.fov`, in degrees. No-op on orthographic.
    #[wasm_bindgen(js_name = setFov)]
    pub fn set_fov(&mut self, fov_deg: f32) {
        if let CameraInner::Perspective(c) = &mut self.inner {
            c.fov = fov_deg.to_radians();
        }
    }

    /// Lens shift as a fraction of the sensor (three.js `filmOffset` / filmGauge).
    #[wasm_bindgen(js_name = setShift)]
    pub fn set_shift(&mut self, shift_x: f32, shift_y: f32) {
        if let CameraInner::Perspective(c) = &mut self.inner {
            c.shift_x = shift_x;
            c.shift_y = shift_y;
        }
    }

    /// Focus distance for DoF (three.js `camera.focus` is related but in different units;
    /// here we store world-space distance).
    #[wasm_bindgen(js_name = setFocusDistance)]
    pub fn set_focus_distance(&mut self, distance: f32) {
        if let CameraInner::Perspective(c) = &mut self.inner {
            c.focus_distance = distance.max(0.0);
        }
    }

    #[wasm_bindgen(js_name = setAspect)]
    pub fn set_aspect(&mut self, aspect: f32) {
        use crate::cameras::Camera;
        match &mut self.inner {
            CameraInner::Perspective(c) => c.set_aspect(aspect),
            CameraInner::Orthographic(c) => c.set_aspect(aspect),
        }
    }

    /// Copy fov/near/far/aspect from another camera. Used by Reflector to
    /// align its virtual camera with the main camera so the projective
    /// texture matrix lines up.
    #[wasm_bindgen(js_name = copyProjection)]
    pub fn copy_projection(&mut self, other: &WebCamera) {
        match (&mut self.inner, &other.inner) {
            (CameraInner::Perspective(a), CameraInner::Perspective(b)) => {
                a.fov = b.fov;
                a.aspect = b.aspect;
                a.near = b.near;
                a.far = b.far;
            }
            (CameraInner::Orthographic(a), CameraInner::Orthographic(b)) => {
                a.left = b.left;
                a.right = b.right;
                a.top = b.top;
                a.bottom = b.bottom;
                a.near = b.near;
                a.far = b.far;
            }
            _ => {}
        }
    }

    /// Read back the column-major 4×4 view matrix (camera.matrixWorldInverse
    /// in three.js terms). Needed by Reflector/Refractor to build the
    /// projective texture matrix for their reflected camera.
    #[wasm_bindgen(js_name = viewMatrix)]
    pub fn view_matrix(&self) -> Vec<f32> {
        use crate::cameras::Camera;
        match &self.inner {
            CameraInner::Perspective(c) => c.view_matrix().elements.to_vec(),
            CameraInner::Orthographic(c) => c.view_matrix().elements.to_vec(),
        }
    }

    /// Read back the column-major 4×4 projection matrix.
    #[wasm_bindgen(js_name = projectionMatrix)]
    pub fn projection_matrix(&self) -> Vec<f32> {
        use crate::cameras::Camera;
        match &self.inner {
            CameraInner::Perspective(c) => c.projection_matrix().elements.to_vec(),
            CameraInner::Orthographic(c) => c.projection_matrix().elements.to_vec(),
        }
    }

    /// Override the virtual camera projection (Reflector oblique near clip).
    #[wasm_bindgen(js_name = setProjectionOverride)]
    pub fn set_projection_override(&mut self, elements: Vec<f32>) {
        if elements.len() != 16 {
            return;
        }
        if let CameraInner::Perspective(c) = &mut self.inner {
            let mut m = [0f32; 16];
            m.copy_from_slice(&elements);
            c.projection_override = Some(m);
        }
    }

    #[wasm_bindgen(js_name = clearProjectionOverride)]
    pub fn clear_projection_override(&mut self) {
        if let CameraInner::Perspective(c) = &mut self.inner {
            c.projection_override = None;
        }
    }

    /// Flatten the scene for [`crate::raytrace::RaytraceRenderer`].
    #[cfg(feature = "raytrace")]
    pub(crate) fn prepare_path_tracer(
        &self,
        scene: &mut WebScene,
        renderer: &mut crate::raytrace::RaytraceRenderer,
    ) {
        match &self.inner {
            CameraInner::Perspective(c) => renderer.prepare(&mut scene.inner, c),
            CameraInner::Orthographic(c) => renderer.prepare(&mut scene.inner, c),
        }
    }

    /// As [`Self::prepare_path_tracer`], but skip BVH rebuild when unchanged.
    #[cfg(feature = "raytrace")]
    pub(crate) fn prepare_path_tracer_if_changed(
        &self,
        scene: &mut WebScene,
        renderer: &mut crate::raytrace::RaytraceRenderer,
    ) -> bool {
        match &self.inner {
            CameraInner::Perspective(c) => renderer.prepare_if_changed(&mut scene.inner, c),
            CameraInner::Orthographic(c) => renderer.prepare_if_changed(&mut scene.inner, c),
        }
    }
}

#[wasm_bindgen]
pub struct WebGeometry {
    inner: Arc<crate::BufferGeometry>,
}

#[wasm_bindgen]
impl WebGeometry {
    #[wasm_bindgen(js_name = box)]
    pub fn box_(width: f32, height: f32, depth: f32) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::BoxGeometry::new(width, height, depth)),
        }
    }

    #[wasm_bindgen(js_name = sphere)]
    pub fn sphere(radius: f32, w_segments: usize, h_segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::SphereGeometry::new(radius, w_segments, h_segments)),
        }
    }

    /// A sphere patch — three.js's `phiStart/phiLength/thetaStart/thetaLength`.
    ///
    /// UVs span 0..1 across the patch, not across the whole sphere, so each
    /// patch can carry its own texture. That is what makes a tiled globe
    /// possible without a single texture past the 8192 device limit.
    #[wasm_bindgen(js_name = sphereRange)]
    #[allow(clippy::too_many_arguments)]
    pub fn sphere_range(
        radius: f32,
        w_segments: usize,
        h_segments: usize,
        phi_start: f32,
        phi_length: f32,
        theta_start: f32,
        theta_length: f32,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::SphereGeometry::with_range(
                radius,
                w_segments,
                h_segments,
                phi_start,
                phi_length,
                theta_start,
                theta_length,
            )),
        }
    }

    #[wasm_bindgen(js_name = plane)]
    pub fn plane(width: f32, height: f32) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::PlaneGeometry::new(width, height)),
        }
    }

    #[wasm_bindgen(js_name = cylinder)]
    pub fn cylinder(
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: usize,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::CylinderGeometry::new(
                radius_top,
                radius_bottom,
                height,
                radial_segments,
                1,
                false,
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }

    #[wasm_bindgen(js_name = torus)]
    pub fn torus(
        radius: f32,
        tube: f32,
        radial_segments: u32,
        tubular_segments: u32,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::TorusGeometry::new(
                radius,
                tube,
                radial_segments.max(3) as usize,
                tubular_segments.max(3) as usize,
                std::f32::consts::PI * 2.0,
            )),
        }
    }

    /// Kirigami Expanded Miura plate lattice.
    ///
    /// `variant`: preset index with slot [`KIRIGAMI_NET_VARIANT`] = developed 2D net.
    /// See [`KirigamiPreset::from_variant`].
    #[wasm_bindgen(js_name = kirigami)]
    pub fn kirigami(variant: u32, nx: u32, ny: u32, thickness: f32) -> WebGeometry {
        let nx = (nx as usize).max(1);
        let ny = (ny as usize).max(2);
        let t = thickness.max(0.0) as f64;
        let geom = if variant == crate::KIRIGAMI_NET_VARIANT {
            crate::KirigamiPreset::Planar
                .evaluate(nx, ny, t)
                .develop_joined()
                .to_geometry(t.max(0.4))
        } else if let Some(preset) = crate::KirigamiPreset::from_variant(variant) {
            let mesh = preset.evaluate(nx, ny, t);
            preset.to_geometry(&mesh)
        } else {
            crate::KirigamiPreset::Planar.evaluate(nx, ny, t).to_geometry()
        };
        WebGeometry {
            inner: Arc::new(geom),
        }
    }

    /// Developed crease-pattern SVG (mm) for a preset or the planar net fallback.
    #[wasm_bindgen(js_name = kirigamiNetSvg)]
    pub fn kirigami_net_svg(variant: u32, nx: u32, ny: u32) -> String {
        let nx = (nx as usize).max(1);
        let ny = (ny as usize).max(2);
        let preset = crate::KirigamiPreset::from_variant(variant)
            .unwrap_or(crate::KirigamiPreset::Planar);
        preset.evaluate(nx, ny, 0.0).develop_joined().to_svg()
    }

    /// Face-connected cuboct continuum lattice (Jenett et al. Sci. Adv. 2020).
    ///
    /// `variant`: 0 rigid, 1 compliant, 2 auxetic, 3 chiral CW, 4 chiral CCW.
    #[wasm_bindgen(js_name = cuboctLattice)]
    pub fn cuboct_lattice(
        variant: u32,
        size: f32,
        cells: u32,
        resolution: u32,
        shape: f32,
    ) -> WebGeometry {
        let kind = match variant {
            1 => crate::Cuboct::Compliant,
            2 => crate::Cuboct::Auxetic,
            3 => crate::Cuboct::ChiralCw,
            4 => crate::Cuboct::ChiralCcw,
            _ => crate::Cuboct::Rigid,
        };
        let n = (cells as usize).clamp(1, 3);
        let res = resolution.clamp(10, 20) as usize;
        let geom = crate::Lattice::new(crate::LatticeKind::Cuboct(kind))
            .size(crate::Vector3::new(size, size, size))
            .cells([n, n, n])
            .shape(shape)
            .resolution(res)
            .max_samples(3_000_000)
            .fit_relative_density(0.18)
            .resolve_walls(2.5)
            .build();
        WebGeometry {
            inner: Arc::new(geom),
        }
    }

    /// Discrete cuboct assembly — exploded face parts, optional vertex colors.
    #[wasm_bindgen(js_name = cuboctAssembly)]
    pub fn cuboct_assembly(
        variant: u32,
        pitch: f32,
        cells: u32,
        explode: f32,
        colored: bool,
    ) -> WebGeometry {
        let kind = match variant {
            1 => crate::Cuboct::Compliant,
            2 => crate::Cuboct::Auxetic,
            3 => crate::Cuboct::ChiralCw,
            4 => crate::Cuboct::ChiralCcw,
            _ => crate::Cuboct::Rigid,
        };
        let n = (cells as usize).clamp(1, 2);
        let asm = crate::CuboctAssembly::new(kind)
            .pitch(pitch.max(0.1))
            .cells([n, n, n])
            .shape(kind.default_shape())
            .explode(explode.max(0.0))
            .resolution(16);
        let geom = if colored {
            asm.build_colored()
        } else {
            asm.build()
        };
        WebGeometry {
            inner: Arc::new(geom),
        }
    }

    /// 2D laser-cut profile for one cuboct face (mm).
    #[wasm_bindgen(js_name = cuboctFaceSvg)]
    pub fn cuboct_face_svg(variant: u32, pitch: f32, shape: f32) -> String {
        let kind = match variant {
            1 => crate::Cuboct::Compliant,
            2 => crate::Cuboct::Auxetic,
            3 => crate::Cuboct::ChiralCw,
            4 => crate::Cuboct::ChiralCcw,
            _ => crate::Cuboct::Rigid,
        };
        crate::CuboctAssembly::new(kind)
            .pitch(pitch.max(0.1))
            .shape(shape)
            .svg()
    }
}

#[wasm_bindgen]
pub struct WebMaterial {
    inner: Arc<crate::Material>,
}

#[wasm_bindgen]
impl WebMaterial {
    #[wasm_bindgen(js_name = basic)]
    pub fn basic(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Basic(crate::BasicMaterial::new(
                color.inner,
            ))),
        }
    }

    /// ShadowMaterial — black transparent surface that only shows shadow darkness.
    #[wasm_bindgen(js_name = shadow)]
    pub fn shadow(opacity: f32) -> WebMaterial {
        let mut m = crate::BasicMaterial::new(crate::math::Color::BLACK);
        m.transparent = true;
        m.opacity = opacity;
        m.shadow_only = true;
        WebMaterial {
            inner: Arc::new(crate::Material::Basic(m)),
        }
    }

    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, color: &WebColor) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Basic(m) => m.color = color.inner,
            crate::Material::Lambert(m) => m.color = color.inner,
            crate::Material::Phong(m) => m.color = color.inner,
            crate::Material::Standard(m) => m.color = color.inner,
            crate::Material::Physical(m) => m.color = color.inner,
            crate::Material::Toon(m) => m.color = color.inner,
            crate::Material::Matcap(m) => m.color = color.inner,
            crate::Material::Line(m) => m.color = color.inner,
            crate::Material::Points(m) => m.color = color.inner,
            crate::Material::Sprite(m) => m.color = color.inner,
            crate::Material::Mirror(m) => m.color = color.inner,
            _ => {}
        }
    }

    #[wasm_bindgen(js_name = setAlphaTest)]
    pub fn set_alpha_test(&mut self, value: f32) {
        if let crate::Material::Basic(m) = Arc::make_mut(&mut self.inner) {
            m.alpha_test = value.max(0.0);
        }
    }

    #[wasm_bindgen(js_name = lambert)]
    pub fn lambert(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Lambert(crate::LambertMaterial::new(
                color.inner,
            ))),
        }
    }

    #[wasm_bindgen(js_name = standard)]
    pub fn standard(color: &WebColor, roughness: f32, metalness: f32) -> WebMaterial {
        let m = crate::StandardMaterial::new(color.inner)
            .with_roughness(roughness)
            .with_metalness(metalness);
        WebMaterial {
            inner: Arc::new(crate::Material::Standard(m)),
        }
    }

    #[wasm_bindgen(js_name = matcap)]
    pub fn matcap(color: &WebColor) -> WebMaterial {
        let m = crate::MatcapMaterial {
            color: color.inner,
            ..Default::default()
        };
        WebMaterial {
            inner: Arc::new(crate::Material::Matcap(m)),
        }
    }

    /// SkyMaterial — Preetham atmospheric scattering shader. Defaults match
    /// three.js's `examples/jsm/objects/Sky.js`: turbidity 10, rayleigh 3,
    /// mieCoefficient 0.005, mieDirectionalG 0.7.
    #[wasm_bindgen(js_name = sky)]
    pub fn sky(
        sun_x: f32,
        sun_y: f32,
        sun_z: f32,
        turbidity: f32,
        rayleigh: f32,
        mie_coefficient: f32,
        mie_directional_g: f32,
    ) -> WebMaterial {
        let m = crate::materials::SkyMaterial {
            sun_position: crate::math::Vector3::new(sun_x, sun_y, sun_z),
            turbidity,
            rayleigh,
            mie_coefficient,
            mie_directional_g,
        };
        WebMaterial {
            inner: Arc::new(crate::Material::Sky(m)),
        }
    }

    /// Mirror material — drives the Reflector / Refractor / Water shader path.
    /// Color tints the reflection; `setMap` binds the render-target texture;
    /// `setTextureMatrix` pushes the per-frame projective UV transform.
    #[wasm_bindgen(js_name = mirror)]
    pub fn mirror(r: f32, g: f32, b: f32) -> WebMaterial {
        let m = crate::materials::MirrorMaterial {
            color: crate::math::Color::new(r, g, b),
            ..Default::default()
        };
        WebMaterial {
            inner: Arc::new(crate::Material::Mirror(m)),
        }
    }

    /// Push a per-frame texture matrix (column-major 4×4) used by the mirror
    /// fragment shader to projectively sample the RT. Computed JS-side as
    /// `(0.5*bias+0.5) * virtualCam.projection * virtualCam.matrixWorldInverse`.
    #[wasm_bindgen(js_name = setTextureMatrix)]
    pub fn set_texture_matrix(&mut self, elements: Vec<f32>) {
        if elements.len() < 16 {
            return;
        }
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Mirror(m) = inner {
            m.texture_matrix.copy_from_slice(&elements[..16]);
        }
    }

    /// MeshDistanceMaterial — used by point-light shadow depth passes.
    /// `ref_x/y/z` is the reference position (typically the light's world pos).
    #[wasm_bindgen(js_name = distance)]
    pub fn distance(ref_x: f32, ref_y: f32, ref_z: f32, near: f32, far: f32) -> WebMaterial {
        let m = crate::materials::DistanceMaterial::new(
            crate::math::Vector3::new(ref_x, ref_y, ref_z),
            near,
            far,
        );
        WebMaterial {
            inner: Arc::new(crate::Material::Distance(m)),
        }
    }

    #[wasm_bindgen(js_name = setMatcap)]
    pub fn set_matcap(&mut self, tex: &WebTexture) {
        // The matcap texture lives on the MatcapMaterial variant; for all
        // others this is a no-op.
        let new_inner = match &*self.inner {
            crate::Material::Matcap(m) => {
                let mut m2 = m.clone();
                m2.matcap = Some(tex.inner.clone());
                crate::Material::Matcap(m2)
            }
            other => other.clone(),
        };
        self.inner = Arc::new(new_inner);
    }

    #[wasm_bindgen(js_name = setMatcapData)]
    pub fn set_matcap_data(&mut self, tex: &WebDataTexture) {
        let new_inner = match &*self.inner {
            crate::Material::Matcap(m) => {
                let mut m2 = m.clone();
                m2.matcap = Some(tex.inner.clone());
                crate::Material::Matcap(m2)
            }
            other => other.clone(),
        };
        self.inner = Arc::new(new_inner);
    }

    /// Attach an albedo / color texture to this material. Mirrors three.js's
    /// `material.map = texture`. Only Basic / Standard / Physical / Sprite
    /// honor the slot; calling on other variants is a no-op.
    #[wasm_bindgen(js_name = setMap)]
    pub fn set_map(&mut self, tex: &WebTexture) {
        self.set_map_arc(tex.inner.clone());
    }

    /// `setMapData` for textures created from raw `Uint8Array` data (three.js
    /// `DataTexture` path). Same effect as `setMap` but typed for that handle.
    #[wasm_bindgen(js_name = setMapData)]
    pub fn set_map_data(&mut self, tex: &WebDataTexture) {
        self.set_map_arc(tex.inner.clone());
    }

    /// Attach a tangent-space normal map (three.js `material.normalMap`).
    #[wasm_bindgen(js_name = setNormalMap)]
    pub fn set_normal_map(&mut self, tex: &WebTexture) {
        self.set_normal_map_arc(tex.inner.clone());
    }

    /// `setNormalMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setNormalMapData)]
    pub fn set_normal_map_data(&mut self, tex: &WebDataTexture) {
        self.set_normal_map_arc(tex.inner.clone());
    }

    /// Attach a roughness map (three.js `material.roughnessMap`). Sampled from
    /// the green channel, and multiplied by `material.roughness`.
    /// An analytic planetary atmosphere — see
    /// [`AtmosphereMaterial`](crate::AtmosphereMaterial). Put it on a sphere a
    /// little larger than the planet, sharing its centre.
    pub fn atmosphere(planet_radius: f32, atmosphere_radius: f32) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Atmosphere(crate::AtmosphereMaterial::new(
                planet_radius,
                atmosphere_radius,
            ))),
        }
    }

    /// Tune the atmosphere: scattering tint, the colour it takes at the
    /// terminator, overall strength, and how fast density falls with altitude.
    #[wasm_bindgen(js_name = setAtmosphere)]
    pub fn set_atmosphere(
        &mut self,
        sunset: &WebColor,
        intensity: f32,
        falloff: f32,
        opacity: f32,
    ) {
        if let crate::Material::Atmosphere(m) = Arc::make_mut(&mut self.inner) {
            m.sunset_color = crate::Color::new(sunset.r(), sunset.g(), sunset.b());
            m.intensity = intensity.max(0.0);
            m.falloff = falloff.clamp(0.1, 32.0);
            m.opacity = opacity.clamp(0.0, 1.0);
        }
    }

    /// Airglow strength and colour on an `AtmosphereMaterial`.
    #[wasm_bindgen(js_name = setAirglow)]
    pub fn set_airglow(&mut self, strength: f32, color: &WebColor) {
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Atmosphere(m) = inner {
            m.airglow = strength.max(0.0);
            m.airglow_color = crate::Color::new(color.r(), color.g(), color.b());
        }
    }

    #[wasm_bindgen(js_name = setRoughnessMap)]
    pub fn set_roughness_map(&mut self, tex: &WebTexture) {
        self.set_roughness_map_arc(tex.inner.clone());
    }

    /// `setRoughnessMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setRoughnessMapData)]
    pub fn set_roughness_map_data(&mut self, tex: &WebDataTexture) {
        self.set_roughness_map_arc(tex.inner.clone());
    }

    /// Attach an emissive map (three.js `material.emissiveMap`), multiplied by
    /// `emissive` and `emissiveIntensity` — city lights on a night side, say.
    #[wasm_bindgen(js_name = setEmissiveMap)]
    pub fn set_emissive_map(&mut self, tex: &WebTexture) {
        self.set_emissive_map_arc(tex.inner.clone());
    }

    /// `setEmissiveMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setEmissiveMapData)]
    pub fn set_emissive_map_data(&mut self, tex: &WebDataTexture) {
        self.set_emissive_map_arc(tex.inner.clone());
    }

    /// Attach a height map (three.js `material.displacementMap`). Unlike a
    /// normal map this moves vertices, so the mesh needs enough of them.
    #[wasm_bindgen(js_name = setDisplacementMap)]
    pub fn set_displacement_map(&mut self, tex: &WebTexture) {
        self.set_displacement_map_arc(tex.inner.clone());
    }

    /// `setDisplacementMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setDisplacementMapData)]
    pub fn set_displacement_map_data(&mut self, tex: &WebDataTexture) {
        self.set_displacement_map_arc(tex.inner.clone());
    }

    /// three.js `material.displacementScale` / `displacementBias`.
    #[wasm_bindgen(js_name = setDisplacement)]
    pub fn set_displacement(&mut self, scale: f32, bias: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => {
                m.displacement_scale = scale;
                m.displacement_bias = bias;
            }
            crate::Material::Physical(m) => {
                m.displacement_scale = scale;
                m.displacement_bias = bias;
            }
            _ => {}
        }
    }

    /// A cloud deck that casts shadows onto this surface.
    #[wasm_bindgen(js_name = setCloudShadowMapData)]
    pub fn set_cloud_shadow_map_data(&mut self, tex: &WebDataTexture) {
        let inner = Arc::make_mut(&mut self.inner);
        let t = Some(tex.inner.clone());
        match inner {
            crate::Material::Standard(m) => m.cloud_shadow_map = t,
            crate::Material::Physical(m) => m.cloud_shadow_map = t,
            _ => {}
        }
    }

    /// `[shell height, shadow strength, cloud longitude offset in turns,
    /// twilight width]`.
    #[wasm_bindgen(js_name = setAtmosphereShading)]
    pub fn set_atmosphere_shading(
        &mut self,
        height: f32,
        shadow: f32,
        rotation: f32,
        twilight: f32,
    ) {
        let inner = Arc::make_mut(&mut self.inner);
        macro_rules! set {
            ($m:expr) => {{
                $m.cloud_height = height.max(0.0);
                $m.cloud_shadow = shadow.clamp(0.0, 1.0);
                $m.cloud_rotation = rotation;
                $m.twilight = twilight.max(0.0);
            }};
        }
        match inner {
            crate::Material::Standard(m) => set!(m),
            crate::Material::Physical(m) => set!(m),
            _ => {}
        }
    }

    /// The colour scattered light takes on near the terminator.
    #[wasm_bindgen(js_name = setTwilightColor)]
    pub fn set_twilight_color(&mut self, color: &WebColor) {
        let inner = Arc::make_mut(&mut self.inner);
        let c = crate::Color::new(color.r(), color.g(), color.b());
        match inner {
            crate::Material::Standard(m) => m.twilight_color = c,
            crate::Material::Physical(m) => m.twilight_color = c,
            _ => {}
        }
    }

    /// A sphere that can eclipse the sun for this surface: world-space centre
    /// and radius, plus the sun's angular radius in radians. A radius of 0
    /// clears it.
    #[wasm_bindgen(js_name = setEclipse)]
    pub fn set_eclipse(&mut self, x: f32, y: f32, z: f32, radius: f32, sun_angular_radius: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        let occ = if radius > 0.0 {
            Some([x, y, z, radius])
        } else {
            None
        };
        match inner {
            crate::Material::Standard(m) => {
                m.eclipse_occluder = occ;
                m.sun_angular_radius = sun_angular_radius;
            }
            crate::Material::Physical(m) => {
                m.eclipse_occluder = occ;
                m.sun_angular_radius = sun_angular_radius;
            }
            _ => {}
        }
    }

    /// three.js `material.normalScale`.
    #[wasm_bindgen(js_name = setNormalScale)]
    pub fn set_normal_scale(&mut self, x: f32, y: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        let v = crate::math::Vector2::new(x, y);
        match inner {
            crate::Material::Standard(m) => m.normal_scale = v,
            crate::Material::Physical(m) => m.normal_scale = v,
            _ => {}
        }
    }
}

/// Extended `MeshPhysicalMaterial` layers.
///
/// These are separate setters rather than a 20-argument constructor so the JS
/// shim can forward only the options the caller actually supplied, and so each
/// one mirrors the corresponding Rust builder method on `PhysicalMaterial`.
/// All are no-ops on non-Physical materials.
#[wasm_bindgen]
impl WebMaterial {
    /// three.js `transmission` / `ior` / `thickness` / `dispersion`.
    #[wasm_bindgen(js_name = setTransmission)]
    pub fn set_transmission(
        &mut self,
        transmission: f32,
        ior: f32,
        thickness: f32,
        dispersion: f32,
    ) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.transmission = transmission;
            m.ior = ior;
            m.thickness = thickness;
            m.dispersion = dispersion;
        }
    }

    /// three.js `material.roughness`.
    #[wasm_bindgen(js_name = setRoughness)]
    pub fn set_roughness(&mut self, roughness: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.roughness = roughness,
            crate::Material::Physical(m) => m.roughness = roughness,
            _ => {}
        }
    }

    /// three.js `material.metalness`.
    #[wasm_bindgen(js_name = setMetalness)]
    pub fn set_metalness(&mut self, metalness: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.metalness = metalness,
            crate::Material::Physical(m) => m.metalness = metalness,
            _ => {}
        }
    }

    /// three.js `anisotropy` / `anisotropyRotation` (radians).
    #[wasm_bindgen(js_name = setAnisotropy)]
    pub fn set_anisotropy(&mut self, strength: f32, rotation: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.anisotropy = strength;
            m.anisotropy_rotation = rotation;
        }
    }

    /// three.js `sheen` / `sheenColor` / `sheenRoughness`.
    #[wasm_bindgen(js_name = setSheen)]
    pub fn set_sheen(&mut self, strength: f32, color: &WebColor, roughness: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.sheen = strength;
            m.sheen_color = color.inner;
            m.sheen_roughness = roughness;
        }
    }

    /// three.js `iridescence` / `iridescenceIOR`, plus the film thickness in
    /// nanometres (three.js expresses this as `iridescenceThicknessRange`; a
    /// single thickness is used here since there is no thickness map).
    #[wasm_bindgen(js_name = setIridescence)]
    pub fn set_iridescence(&mut self, strength: f32, ior: f32, thickness_nm: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.iridescence = strength;
            m.iridescence_ior = ior;
            m.iridescence_thickness = thickness_nm;
        }
    }

    /// three.js `attenuationColor` / `attenuationDistance`. A non-finite or
    /// non-positive distance disables Beer-Lambert absorption.
    #[wasm_bindgen(js_name = setAttenuation)]
    pub fn set_attenuation(&mut self, color: &WebColor, distance: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.attenuation_color = color.inner;
            m.attenuation_distance = distance;
        }
    }

    /// Transparency compositing mode: 0 = Blend, 1 = Glass, 2 = OIT,
    /// 3 = Refract (screen-space refraction).
    #[wasm_bindgen(js_name = setTransparencyMode)]
    pub fn set_transparency_mode(&mut self, mode: u32) {
        use crate::TransparencyMode::*;
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.transparency = match mode {
                1 => Glass,
                2 => Oit,
                3 => Refract,
                _ => Blend,
            };
        }
    }

    /// A measured-reflectance preset from `threers::materials::presets`.
    ///
    /// `param` is only read by the presets that take one: `anodized_titanium`
    /// (film thickness, nm) and `brushed_aluminum` (streak rotation, radians).
    /// An unknown name falls back to a neutral dielectric rather than throwing,
    /// so a typo shows up as a visibly plain sphere instead of a dead page.
    #[wasm_bindgen(js_name = preset)]
    pub fn preset(name: &str, param: f32) -> WebMaterial {
        use crate::materials::presets as p;
        let m = match name {
            "gold_foil" => p::gold_foil(),
            "silver_foil" => p::silver_foil(),
            "aluminum" => p::aluminum(),
            "brushed_aluminum" => p::brushed_aluminum(param),
            "titanium" => p::titanium(),
            "anodized_titanium" => p::anodized_titanium(param),
            "solar_cell" => p::solar_cell(),
            "array_backing" => p::array_backing(),
            "optical_glass" => p::optical_glass(),
            "white_thermal_paint" => p::white_thermal_paint(),
            "black_kapton" => p::black_kapton(),
            _ => crate::PhysicalMaterial::new(crate::Color::from_hex(0x808080)),
        };
        WebMaterial {
            inner: Arc::new(crate::Material::Physical(m)),
        }
    }
}

#[wasm_bindgen]
impl WebMaterial {
    #[wasm_bindgen(js_name = debugString)]
    pub fn debug_string(&self) -> String {
        let slots = self.inner.texture_slots();
        format!(
            "Material kind={:?} map={} normal={} rough={} metal={} ao={} emissive={}",
            self.inner.kind(),
            slots.map.is_some(),
            slots.normal_map.is_some(),
            slots.roughness_map.is_some(),
            slots.metalness_map.is_some(),
            slots.ao_map.is_some(),
            slots.emissive_map.is_some(),
        )
    }
}

#[wasm_bindgen]
impl WebMaterial {
    #[wasm_bindgen(js_name = setDashed)]
    pub fn set_dashed(&mut self, scale: f32, dash_size: f32, gap_size: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Line(m) = inner {
            m.dashed = true;
            m.dash_scale = scale;
            m.dash_size = dash_size;
            m.gap_size = gap_size;
        }
    }

    /// Set per-material opacity (0..1). Sets the matching field on whichever
    /// concrete material this wraps.
    #[wasm_bindgen(js_name = setOpacity)]
    pub fn set_opacity(&mut self, opacity: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Basic(m) => m.opacity = opacity,
            crate::Material::Lambert(m) => m.opacity = opacity,
            crate::Material::Phong(m) => m.opacity = opacity,
            crate::Material::Standard(m) => m.opacity = opacity,
            crate::Material::Physical(m) => m.opacity = opacity,
            crate::Material::Toon(m) => m.opacity = opacity,
            _ => {}
        }
    }

    /// Mark a material as needing alpha blending. three.js uses an explicit
    /// `transparent` flag separate from `opacity` so semi-transparent textures
    /// can render correctly even at opacity 1.0.
    #[wasm_bindgen(js_name = setTransparent)]
    pub fn set_transparent(&mut self, transparent: bool) {
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Basic(m) = inner {
            m.transparent = transparent;
        }
    }

    /// Set the emissive color. Standard / Physical / Lambert / Phong / Toon
    /// honor this; other variants are no-ops.
    #[wasm_bindgen(js_name = setEmissive)]
    pub fn set_emissive(&mut self, color: &WebColor) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.emissive = color.inner,
            crate::Material::Physical(m) => m.emissive = color.inner,
            crate::Material::Lambert(m) => m.emissive = color.inner,
            crate::Material::Phong(m) => m.emissive = color.inner,
            crate::Material::Toon(m) => m.emissive = color.inner,
            _ => {}
        }
    }

    /// Multiplier on the emissive contribution (`material.emissiveIntensity`).
    #[wasm_bindgen(js_name = setEmissiveIntensity)]
    pub fn set_emissive_intensity(&mut self, intensity: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Standard(m) = inner {
            m.emissive_intensity = intensity;
        } else if let crate::Material::Physical(m) = inner {
            m.emissive_intensity = intensity;
        }
    }

    /// Confine the emissive map to the night side (0 = always on, 1 = terminator-gated).
    /// City-lights maps need 1; lamps need 0.
    #[wasm_bindgen(js_name = setEmissiveNightSide)]
    pub fn set_emissive_night_side(&mut self, amount: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        if let crate::Material::Standard(m) = inner {
            m.emissive_night_side = amount.clamp(0.0, 1.0);
        }
    }

    /// Toggle wireframe rendering (renderer picks the line-polygon pipeline).
    #[wasm_bindgen(js_name = setWireframe)]
    pub fn set_wireframe(&mut self, wireframe: bool) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Basic(m) => m.wireframe = wireframe,
            crate::Material::Lambert(m) => m.wireframe = wireframe,
            crate::Material::Phong(m) => m.wireframe = wireframe,
            crate::Material::Standard(m) => m.wireframe = wireframe,
            crate::Material::Physical(m) => m.wireframe = wireframe,
            crate::Material::Normal(m) => m.wireframe = wireframe,
            crate::Material::Depth(m) => m.wireframe = wireframe,
            crate::Material::Toon(m) => m.wireframe = wireframe,
            crate::Material::Matcap(m) => m.wireframe = wireframe,
            _ => {}
        }
    }
}

impl WebMaterial {
    fn set_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Basic(m) => m.map = Some(tex),
            crate::Material::Standard(m) => m.map = Some(tex),
            crate::Material::Physical(m) => m.map = Some(tex),
            crate::Material::Sprite(m) => m.map = Some(tex),
            crate::Material::Mirror(m) => m.map = Some(tex),
            _ => {}
        }
    }

    fn set_normal_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.normal_map = Some(tex),
            crate::Material::Physical(m) => m.normal_map = Some(tex),
            _ => {}
        }
    }

    fn set_roughness_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.roughness_map = Some(tex),
            crate::Material::Physical(m) => m.roughness_map = Some(tex),
            _ => {}
        }
    }

    fn set_displacement_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.displacement_map = Some(tex),
            crate::Material::Physical(m) => m.displacement_map = Some(tex),
            _ => {}
        }
    }

    fn set_emissive_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.emissive_map = Some(tex),
            crate::Material::Physical(m) => m.emissive_map = Some(tex),
            _ => {}
        }
    }
}

#[wasm_bindgen]
impl WebMaterial {
    /// Set the material's render side. `0` = FrontSide (default),
    /// `1` = BackSide, `2` = DoubleSide. Both 1 and 2 route to the no-cull
    /// triangle pipeline at draw time.
    #[wasm_bindgen(js_name = setSide)]
    pub fn set_side(&mut self, side: u32) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Basic(m) => m.side = side,
            crate::Material::Lambert(m) => m.side = side,
            crate::Material::Phong(m) => m.side = side,
            crate::Material::Standard(m) => m.side = side,
            crate::Material::Physical(m) => m.side = side,
            crate::Material::Toon(m) => m.side = side,
            // Sky always uses BackSide; setting from JS is a no-op (already 1).
            _ => {}
        }
    }
}

#[wasm_bindgen]
pub struct WebMesh {
    geometry: Arc<crate::BufferGeometry>,
    material: Arc<crate::Material>,
}

#[wasm_bindgen]
impl WebMesh {
    #[wasm_bindgen(constructor)]
    pub fn new(geom: &WebGeometry, mat: &WebMaterial) -> WebMesh {
        WebMesh {
            geometry: geom.inner.clone(),
            material: mat.inner.clone(),
        }
    }
}

#[wasm_bindgen]
pub struct WebLight {
    inner: LightInner,
}

enum LightInner {
    Ambient(crate::AmbientLight),
    Directional(crate::DirectionalLight),
    Point(crate::PointLight),
    Spot(crate::SpotLight),
    Hemisphere(crate::HemisphereLight),
    RectArea(crate::RectAreaLight),
}

#[wasm_bindgen]
impl WebLight {
    #[wasm_bindgen(js_name = ambient)]
    pub fn ambient(color: &WebColor, intensity: f32) -> WebLight {
        WebLight {
            inner: LightInner::Ambient(crate::AmbientLight::new(color.inner, intensity)),
        }
    }

    #[wasm_bindgen(js_name = directional)]
    pub fn directional(color: &WebColor, intensity: f32) -> WebLight {
        WebLight {
            inner: LightInner::Directional(crate::DirectionalLight::new(color.inner, intensity)),
        }
    }

    /// Set the direction the light shines toward (only meaningful for
    /// Directional / Spot lights). Vector does not need to be normalized.
    #[wasm_bindgen(js_name = setDirection)]
    pub fn set_direction(&mut self, x: f32, y: f32, z: f32) {
        let v = crate::Vector3::new(x, y, z);
        match &mut self.inner {
            LightInner::Directional(d) => d.direction = v,
            LightInner::Spot(s) => s.direction = v,
            _ => {}
        }
    }

    /// Diagnostic: dump kind + key state as a string. Lets parity tests assert
    /// that JS-side mutations actually landed on the wasm side.
    #[wasm_bindgen(js_name = debugString)]
    pub fn debug_string(&self) -> String {
        match &self.inner {
            LightInner::Spot(s) => format!(
                "Spot dir=({:.3},{:.3},{:.3}) intensity={} angle={} penumbra={} distance={} decay={}",
                s.direction.x, s.direction.y, s.direction.z,
                s.intensity, s.angle, s.penumbra, s.distance, s.decay,
            ),
            LightInner::Directional(d) => format!(
                "Directional dir=({:.3},{:.3},{:.3}) intensity={}",
                d.direction.x, d.direction.y, d.direction.z, d.intensity,
            ),
            LightInner::Hemisphere(h) => format!(
                "Hemisphere sky=({},{},{}) ground=({},{},{}) intensity={}",
                h.sky_color.r, h.sky_color.g, h.sky_color.b,
                h.ground_color.r, h.ground_color.g, h.ground_color.b,
                h.intensity,
            ),
            _ => "(other)".to_string(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebColor {
    inner: crate::Color,
}

#[wasm_bindgen]
impl WebColor {
    #[wasm_bindgen(constructor)]
    pub fn new(r: f32, g: f32, b: f32) -> WebColor {
        WebColor {
            inner: crate::Color::new(r, g, b),
        }
    }

    #[wasm_bindgen(js_name = fromHex)]
    pub fn from_hex(hex: u32) -> WebColor {
        WebColor {
            inner: crate::Color::from_hex(hex),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebVector3 {
    inner: crate::Vector3,
}

#[wasm_bindgen]
impl WebVector3 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32) -> WebVector3 {
        WebVector3 {
            inner: crate::Vector3::new(x, y, z),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.inner.x
    }
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.inner.y
    }
    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f32 {
        self.inner.z
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebEuler {
    inner: crate::Euler,
}

#[wasm_bindgen]
impl WebEuler {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32) -> WebEuler {
        WebEuler {
            inner: crate::Euler::new(x, y, z),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.inner.x
    }
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.inner.y
    }
    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f32 {
        self.inner.z
    }
}

// JS console logging convenience.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

// ======================================================================
//                              MATH
// ======================================================================

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebVector2 {
    pub x: f32,
    pub y: f32,
}
#[wasm_bindgen]
impl WebVector2 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32) -> WebVector2 {
        WebVector2 { x, y }
    }
    #[wasm_bindgen(js_name = lengthSq)]
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }
    pub fn dot(&self, o: &WebVector2) -> f32 {
        self.x * o.x + self.y * o.y
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebVector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[wasm_bindgen]
impl WebVector4 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> WebVector4 {
        WebVector4 { x, y, z, w }
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebMatrix3 {
    inner: crate::Matrix3,
}
#[wasm_bindgen]
impl WebMatrix3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMatrix3 {
        WebMatrix3 {
            inner: crate::Matrix3::identity(),
        }
    }
    pub fn identity() -> WebMatrix3 {
        WebMatrix3 {
            inner: crate::Matrix3::identity(),
        }
    }
    pub fn elements(&self) -> Vec<f32> {
        self.inner.elements.to_vec()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebMatrix4 {
    pub(crate) inner: crate::Matrix4,
}
#[wasm_bindgen]
impl WebMatrix4 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::identity(),
        }
    }
    pub fn identity() -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::identity(),
        }
    }
    pub fn elements(&self) -> Vec<f32> {
        self.inner.elements.to_vec()
    }
    #[wasm_bindgen(js_name = makePerspective)]
    pub fn make_perspective(fov: f32, aspect: f32, near: f32, far: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::perspective(fov, aspect, near, far),
        }
    }
    pub fn invert(&self) -> WebMatrix4 {
        WebMatrix4 {
            inner: self.inner.invert(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebQuaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[wasm_bindgen]
impl WebQuaternion {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> WebQuaternion {
        WebQuaternion { x, y, z, w }
    }
    pub fn identity() -> WebQuaternion {
        WebQuaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
    #[wasm_bindgen(js_name = setFromEuler)]
    pub fn set_from_euler(e: &WebEuler) -> WebQuaternion {
        let q = e.inner.to_quaternion();
        WebQuaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebBox2 {
    inner: crate::Box2,
}
#[wasm_bindgen]
impl WebBox2 {
    #[wasm_bindgen(constructor)]
    pub fn new(min: &WebVector2, max: &WebVector2) -> WebBox2 {
        WebBox2 {
            inner: crate::Box2::new(
                crate::Vector2::new(min.x, min.y),
                crate::Vector2::new(max.x, max.y),
            ),
        }
    }
    pub fn empty() -> WebBox2 {
        WebBox2 {
            inner: crate::Box2::empty(),
        }
    }
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebBox3 {
    pub(crate) inner: crate::Box3,
}
#[wasm_bindgen]
impl WebBox3 {
    #[wasm_bindgen(constructor)]
    pub fn new(min: &WebVector3, max: &WebVector3) -> WebBox3 {
        WebBox3 {
            inner: crate::Box3::new(min.inner, max.inner),
        }
    }
    pub fn empty() -> WebBox3 {
        WebBox3 {
            inner: crate::Box3::empty(),
        }
    }
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
    #[wasm_bindgen(js_name = intersectsBox)]
    pub fn intersects_box(&self, o: &WebBox3) -> bool {
        self.inner.intersects_box(&o.inner)
    }
    pub fn center(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.center(),
        }
    }
    pub fn size(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.size(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebSphere {
    inner: crate::Sphere,
}
#[wasm_bindgen]
impl WebSphere {
    #[wasm_bindgen(constructor)]
    pub fn new(center: &WebVector3, radius: f32) -> WebSphere {
        WebSphere {
            inner: crate::Sphere::new(center.inner, radius),
        }
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebRay {
    inner: crate::Ray,
}
#[wasm_bindgen]
impl WebRay {
    #[wasm_bindgen(constructor)]
    pub fn new(origin: &WebVector3, direction: &WebVector3) -> WebRay {
        WebRay {
            inner: crate::Ray::new(origin.inner, direction.inner),
        }
    }
    pub fn at(&self, t: f32) -> WebVector3 {
        WebVector3 {
            inner: self.inner.at(t),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebPlane {
    inner: crate::Plane,
}
#[wasm_bindgen]
impl WebPlane {
    #[wasm_bindgen(constructor)]
    pub fn new(normal: &WebVector3, constant: f32) -> WebPlane {
        WebPlane {
            inner: crate::Plane::new(normal.inner, constant),
        }
    }
    #[wasm_bindgen(js_name = distanceToPoint)]
    pub fn distance_to_point(&self, p: &WebVector3) -> f32 {
        self.inner.distance_to_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebTriangle {
    inner: crate::Triangle,
}
#[wasm_bindgen]
impl WebTriangle {
    #[wasm_bindgen(constructor)]
    pub fn new(a: &WebVector3, b: &WebVector3, c: &WebVector3) -> WebTriangle {
        WebTriangle {
            inner: crate::Triangle::new(a.inner, b.inner, c.inner),
        }
    }
    pub fn area(&self) -> f32 {
        self.inner.area()
    }
    pub fn normal(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.normal(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebFrustum {
    inner: crate::Frustum,
}
#[wasm_bindgen]
impl WebFrustum {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFrustum {
        WebFrustum {
            inner: crate::Frustum::default(),
        }
    }
    #[wasm_bindgen(js_name = setFromProjectionMatrix)]
    pub fn set_from_projection_matrix(m: &WebMatrix4) -> WebFrustum {
        WebFrustum {
            inner: crate::Frustum::from_projection_matrix(&m.inner),
        }
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebSpherical {
    inner: crate::Spherical,
}
#[wasm_bindgen]
impl WebSpherical {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, phi: f32, theta: f32) -> WebSpherical {
        WebSpherical {
            inner: crate::Spherical::new(radius, phi, theta),
        }
    }
    #[wasm_bindgen(js_name = setFromVector3)]
    pub fn set_from_vector3(v: &WebVector3) -> WebSpherical {
        WebSpherical {
            inner: crate::Spherical::from_vector3(v.inner),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebCylindrical {
    inner: crate::Cylindrical,
}
#[wasm_bindgen]
impl WebCylindrical {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, theta: f32, y: f32) -> WebCylindrical {
        WebCylindrical {
            inner: crate::Cylindrical::new(radius, theta, y),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebLine3 {
    inner: crate::Line3,
}
#[wasm_bindgen]
impl WebLine3 {
    #[wasm_bindgen(constructor)]
    pub fn new(start: &WebVector3, end: &WebVector3) -> WebLine3 {
        WebLine3 {
            inner: crate::Line3::new(start.inner, end.inner),
        }
    }
    pub fn distance(&self) -> f32 {
        self.inner.distance()
    }
    pub fn center(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.center(),
        }
    }
}

// ======================================================================
//                          GEOMETRIES (extras)
// ======================================================================

#[wasm_bindgen]
impl WebGeometry {
    #[wasm_bindgen(js_name = circle)]
    pub fn circle(radius: f32, segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::CircleGeometry::new(
                radius,
                segments.max(3),
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = ring)]
    pub fn ring(inner: f32, outer: f32, theta_segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::RingGeometry::new(
                inner,
                outer,
                theta_segments.max(3),
                1,
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = cone)]
    pub fn cone(radius: f32, height: f32, radial_segments: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::ConeGeometry::new(
                radius,
                height,
                radial_segments.max(3),
                1,
                false,
                0.0,
                std::f32::consts::PI * 2.0,
            )),
        }
    }
    #[wasm_bindgen(js_name = torusKnot)]
    pub fn torus_knot(
        radius: f32,
        tube: f32,
        tubular_segments: usize,
        radial_segments: usize,
        p: u32,
        q: u32,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::TorusKnotGeometry::new(
                radius,
                tube,
                tubular_segments.max(3),
                radial_segments.max(3),
                p,
                q,
            )),
        }
    }
    #[wasm_bindgen(js_name = capsule)]
    pub fn capsule(
        radius: f32,
        length: f32,
        cap_segments: usize,
        radial_segments: usize,
    ) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::CapsuleGeometry::new(
                radius,
                length,
                cap_segments.max(1),
                radial_segments.max(3),
            )),
        }
    }
    #[wasm_bindgen(js_name = tetrahedron)]
    pub fn tetrahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::TetrahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = octahedron)]
    pub fn octahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::OctahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = icosahedron)]
    pub fn icosahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::IcosahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = dodecahedron)]
    pub fn dodecahedron(radius: f32, detail: usize) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::DodecahedronGeometry::new(radius, detail)),
        }
    }
    #[wasm_bindgen(js_name = boxLine)]
    pub fn box_line(w: f32, h: f32, d: f32) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::BoxLineGeometry::new(w, h, d)),
        }
    }

    /// Clone into a `WebBufferGeometry` handle (same underlying data).
    #[wasm_bindgen(js_name = toBufferGeometry)]
    pub fn to_buffer_geometry(&self) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: self.inner.clone(),
        }
    }
}

// ======================================================================
//                            MATERIALS (extras)
// ======================================================================

#[wasm_bindgen]
impl WebMaterial {
    #[wasm_bindgen(js_name = phong)]
    pub fn phong(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Phong(crate::PhongMaterial::new(
                color.inner,
            ))),
        }
    }
    #[wasm_bindgen(js_name = physical)]
    pub fn physical(
        color: &WebColor,
        roughness: f32,
        metalness: f32,
        clearcoat: f32,
        clearcoat_roughness: f32,
    ) -> WebMaterial {
        let mut m = crate::PhysicalMaterial::new(color.inner);
        m.roughness = roughness;
        m.metalness = metalness;
        m.clearcoat = clearcoat;
        m.clearcoat_roughness = clearcoat_roughness;
        WebMaterial {
            inner: Arc::new(crate::Material::Physical(m)),
        }
    }
    #[wasm_bindgen(js_name = normalMat)]
    pub fn normal_mat() -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Normal(crate::NormalMaterial::new())),
        }
    }
    #[wasm_bindgen(js_name = depth)]
    pub fn depth() -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Depth(crate::DepthMaterial::new())),
        }
    }
    #[wasm_bindgen(js_name = toon)]
    pub fn toon(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Toon(crate::ToonMaterial::new(color.inner))),
        }
    }
    #[wasm_bindgen(js_name = lineDashed)]
    pub fn line_dashed(color: &WebColor, scale: f32, dash_size: f32, gap_size: f32) -> WebMaterial {
        let mut m = crate::LineBasicMaterial::new(color.inner);
        m.dashed = true;
        m.dash_scale = scale;
        m.dash_size = dash_size;
        m.gap_size = gap_size;
        WebMaterial {
            inner: Arc::new(crate::Material::Line(m)),
        }
    }
    #[wasm_bindgen(js_name = line)]
    pub fn line(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Line(crate::LineBasicMaterial::new(
                color.inner,
            ))),
        }
    }
    #[wasm_bindgen(js_name = points)]
    pub fn points(color: &WebColor, size: f32) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Points(crate::PointsMaterial::new(
                color.inner,
                size,
            ))),
        }
    }
    #[wasm_bindgen(js_name = sprite)]
    pub fn sprite(color: &WebColor) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Sprite(crate::SpriteMaterial::new(
                color.inner,
            ))),
        }
    }
}

// ======================================================================
//                              LIGHTS (extras)
// ======================================================================

#[wasm_bindgen]
impl WebLight {
    #[wasm_bindgen(js_name = point)]
    pub fn point(color: &WebColor, intensity: f32, distance: f32, decay: f32) -> WebLight {
        let mut p = crate::PointLight::new(color.inner, intensity);
        p.distance = distance;
        p.decay = decay;
        WebLight {
            inner: LightInner::Point(p),
        }
    }

    // Adjust intensity in-place (three.js .intensity = ...).
    #[wasm_bindgen(js_name = setIntensity)]
    pub fn set_intensity(&mut self, intensity: f32) {
        match &mut self.inner {
            LightInner::Ambient(l) => l.intensity = intensity,
            LightInner::Directional(l) => l.intensity = intensity,
            LightInner::Point(l) => l.intensity = intensity,
            LightInner::Spot(l) => l.intensity = intensity,
            LightInner::Hemisphere(l) => l.intensity = intensity,
            LightInner::RectArea(l) => l.intensity = intensity,
        }
    }

    /// Toggle whether this light casts shadows. Only Directional / Spot /
    /// Point variants honor it — the renderer picks the first cast_shadow=true
    /// light of each kind as that frame's shadow caster.
    #[wasm_bindgen(js_name = setCastShadow)]
    pub fn set_cast_shadow(&mut self, cast: bool) {
        match &mut self.inner {
            LightInner::Directional(l) => l.cast_shadow = cast,
            LightInner::Spot(l) => l.cast_shadow = cast,
            LightInner::Point(l) => l.cast_shadow = cast,
            _ => {}
        }
    }

    /// Orthographic shadow frustum (three.js `light.shadow.camera.*`).
    #[wasm_bindgen(js_name = setShadowCamera)]
    pub fn set_shadow_camera(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
    ) {
        let size = ((right - left).abs().max((top - bottom).abs())) * 0.5;
        let settings = crate::ShadowSettings {
            camera_size: size.max(0.1),
            camera_near: near,
            camera_far: far,
            ..match &self.inner {
                LightInner::Directional(l) => l.shadow,
                LightInner::Spot(l) => l.shadow,
                LightInner::Point(l) => l.shadow,
                _ => crate::ShadowSettings::default(),
            }
        };
        match &mut self.inner {
            LightInner::Directional(l) => l.shadow = settings,
            LightInner::Spot(l) => l.shadow = settings,
            LightInner::Point(l) => l.shadow = settings,
            _ => {}
        }
    }
    #[wasm_bindgen(js_name = spot)]
    pub fn spot(
        color: &WebColor,
        intensity: f32,
        distance: f32,
        angle: f32,
        penumbra: f32,
        decay: f32,
    ) -> WebLight {
        let mut s = crate::SpotLight::new(color.inner, intensity);
        s.distance = distance;
        s.angle = angle;
        s.penumbra = penumbra;
        s.decay = decay;
        WebLight {
            inner: LightInner::Spot(s),
        }
    }
    #[wasm_bindgen(js_name = hemisphere)]
    pub fn hemisphere(sky: &WebColor, ground: &WebColor, intensity: f32) -> WebLight {
        let h = crate::HemisphereLight::new(sky.inner, ground.inner, intensity);
        WebLight {
            inner: LightInner::Hemisphere(h),
        }
    }
    #[wasm_bindgen(js_name = rectArea)]
    pub fn rect_area(color: &WebColor, intensity: f32, width: f32, height: f32) -> WebLight {
        let r = crate::RectAreaLight::new(color.inner, intensity, width, height);
        WebLight {
            inner: LightInner::RectArea(r),
        }
    }
}

// ======================================================================
//                            TEXTURES
// ======================================================================

#[wasm_bindgen]
pub struct WebTexture {
    inner: Arc<crate::Texture>,
}
#[wasm_bindgen]
impl WebTexture {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> WebTexture {
        WebTexture {
            inner: Arc::new(crate::Texture::new(
                width,
                height,
                crate::TextureFormat::Rgba8UnormSrgb,
                data,
            )),
        }
    }
    #[wasm_bindgen(js_name = solid)]
    pub fn solid(r: u8, g: u8, b: u8, a: u8) -> WebTexture {
        WebTexture {
            inner: Arc::new(crate::Texture::solid(
                [r, g, b, a],
                crate::TextureFormat::Rgba8UnormSrgb,
            )),
        }
    }
    /// Set sampler filter/wrap. 0 = LinearFilter, 1 = NearestFilter.
    /// 0 = ClampToEdge, 1 = Repeat, 2 = MirroredRepeat.
    #[wasm_bindgen(js_name = setFilters)]
    pub fn set_filters(&mut self, mag: u32, _min: u32, wrap_s: u32, wrap_t: u32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.mag_filter = if mag == 1 {
            crate::textures::TextureFilter::Nearest
        } else {
            crate::textures::TextureFilter::Linear
        };
        inner.min_filter = inner.mag_filter;
        let conv = |w: u32| match w {
            1 => crate::textures::TextureWrap::Repeat,
            2 => crate::textures::TextureWrap::Repeat, // MirroredRepeat → Repeat (no mirror in pre-built samplers yet)
            _ => crate::textures::TextureWrap::ClampToEdge,
        };
        inner.wrap_s = conv(wrap_s);
        inner.wrap_t = conv(wrap_t);
    }

    /// Resample this equirectangular map onto the six faces of a cube.
    ///
    /// A cube has no poles. An equirectangular map converges every longitude
    /// onto one texel at each end, and anything with width near there fans out
    /// radially when it is wrapped on a sphere; a cube face is a plane, its
    /// texels are near enough uniform everywhere, and no point on it is
    /// special. Returns the faces in `+X, -X, +Y, -Y, +Z, -Z` order, which is
    /// the order `planet::CUBE_FACES` places them in.
    ///
    /// `face_size` of 0 picks a quarter of the map's width — four faces carry
    /// the 360 degrees the map spends its full width on, so that is lossless at
    /// the equator and a gain toward the poles.
    #[cfg(feature = "planet")]
    #[wasm_bindgen(js_name = cubeFaces)]
    pub fn cube_faces(&self, face_size: u32) -> js_sys::Array {
        let n = if face_size == 0 {
            (self.inner.width / 4).clamp(64, 4096)
        } else {
            face_size
        };
        let out = js_sys::Array::new();
        for face in crate::planet::equirect_to_cube_faces(&self.inner, n) {
            out.push(&JsValue::from(WebTexture {
                inner: Arc::new(face),
            }));
        }
        out
    }

    /// A second handle on the same pixels, carrying its own UV transform.
    ///
    /// `Texture`'s bytes live behind their own `Arc`, so this clones a handle
    /// and not an image — and the renderer keys its GPU cache on those bytes,
    /// so every view of one buffer shares a single upload between them.
    ///
    /// This is what lets a body split into patches for tile streaming still
    /// cost one texture per map at level 0: eight patches over one map, each
    /// addressing its own quarter through `offset`/`repeat`, instead of eight
    /// sliced copies uploaded separately.
    #[wasm_bindgen(js_name = view)]
    pub fn view(&self, ox: f32, oy: f32, rx: f32, ry: f32) -> WebDataTexture {
        let mut inner = (*self.inner).clone();
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        WebDataTexture {
            inner: Arc::new(inner),
        }
    }

    /// three.js `Texture.offset` / `Texture.repeat` — the UV sub-rectangle this
    /// texture covers. What lets several meshes share one atlas.
    #[wasm_bindgen(js_name = setUvTransform)]
    pub fn set_uv_transform(&mut self, ox: f32, oy: f32, rx: f32, ry: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
    }
    pub fn width(&self) -> u32 {
        self.inner.width
    }
    pub fn height(&self) -> u32 {
        self.inner.height
    }
}

#[wasm_bindgen]
pub struct WebCubeTexture {
    inner: Arc<crate::CubeTexture>,
}
#[wasm_bindgen]
impl WebCubeTexture {
    #[wasm_bindgen(constructor)]
    pub fn new(
        size: u32,
        px: Vec<u8>,
        nx: Vec<u8>,
        py: Vec<u8>,
        ny: Vec<u8>,
        pz: Vec<u8>,
        nz: Vec<u8>,
    ) -> WebCubeTexture {
        WebCubeTexture {
            inner: Arc::new(crate::CubeTexture::new(
                size,
                crate::TextureFormat::Rgba8UnormSrgb,
                [px, nx, py, ny, pz, nz],
            )),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> u32 {
        self.inner.size
    }

    #[wasm_bindgen(js_name = sampleCubeUvEnv)]
    pub fn sample_cube_uv_env(&self, dx: f32, dy: f32, dz: f32, roughness: f32) -> Vec<f32> {
        let Some(atlas) = self.inner.cube_uv_atlas.as_ref() else {
            return vec![0.0, 0.0, 0.0];
        };
        crate::extras::cube_uv::sample_cube_uv_env(atlas, [dx, dy, dz], roughness).to_vec()
    }
}

#[wasm_bindgen]
pub struct WebPmremGenerator;
#[wasm_bindgen]
impl WebPmremGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebPmremGenerator {
        WebPmremGenerator
    }

    /// Prefilter `cube` into a PMREM mip chain at `size` and return a CubeTexture wrapper.
    #[wasm_bindgen(js_name = fromCubemap)]
    pub fn from_cubemap(cube: &WebCubeTexture, size: u32) -> WebCubeTexture {
        let pmrem = crate::PmremGenerator::generate_pmrem(&cube.inner, size.max(1));
        WebCubeTexture {
            inner: Arc::new(pmrem),
        }
    }

    /// Convert an equirectangular RGBA8 texture into a PMREM cubemap.
    #[wasm_bindgen(js_name = fromEquirectangular)]
    pub fn from_equirectangular(
        data: Vec<u8>,
        src_w: u32,
        src_h: u32,
        cube_size: u32,
    ) -> WebCubeTexture {
        let cube = crate::PmremGenerator::from_equirect(&data, src_w, src_h, cube_size.max(1));
        let pmrem = crate::PmremGenerator::generate_pmrem(&cube, cube_size.max(1));
        WebCubeTexture {
            inner: Arc::new(pmrem),
        }
    }
}

#[wasm_bindgen]
pub struct WebDataTexture {
    inner: Arc<crate::Texture>,
}
#[wasm_bindgen]
impl WebDataTexture {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> WebDataTexture {
        WebDataTexture {
            inner: Arc::new(crate::DataTexture::new(
                width,
                height,
                crate::TextureFormat::Rgba8Unorm,
                data,
            )),
        }
    }
    /// A texture whose bytes are sRGB-encoded — what every JPEG and PNG colour
    /// map is.
    ///
    /// The default constructor uploads `Rgba8Unorm`, i.e. the sampler reads the
    /// bytes as linear. Feeding it an ordinary image makes the darks far too
    /// bright; this asks the hardware to decode instead, which is both correct
    /// and free. Use it for colour maps, never for normal/roughness data.
    #[wasm_bindgen(js_name = newSrgb)]
    pub fn new_srgb(width: u32, height: u32, data: Vec<u8>) -> WebDataTexture {
        WebDataTexture {
            inner: Arc::new(crate::DataTexture::new(
                width,
                height,
                crate::TextureFormat::Rgba8UnormSrgb,
                data,
            )),
        }
    }

    /// Replace RGBA8 level-0 bytes without allocating a new texture id.
    #[wasm_bindgen(js_name = replaceRgba)]
    pub fn replace_rgba(&mut self, data: Vec<u8>) -> bool {
        let inner = Arc::make_mut(&mut self.inner);
        inner.replace_rgba_bytes(data)
    }

    #[wasm_bindgen(js_name = setFilters)]
    pub fn set_filters(&mut self, mag: u32, _min: u32, wrap_s: u32, wrap_t: u32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.mag_filter = if mag == 1 {
            crate::textures::TextureFilter::Nearest
        } else {
            crate::textures::TextureFilter::Linear
        };
        inner.min_filter = inner.mag_filter;
        let conv = |w: u32| match w {
            1 => crate::textures::TextureWrap::Repeat,
            2 => crate::textures::TextureWrap::Repeat,
            _ => crate::textures::TextureWrap::ClampToEdge,
        };
        inner.wrap_s = conv(wrap_s);
        inner.wrap_t = conv(wrap_t);
    }

    /// Resample this equirectangular map onto the six faces of a cube.
    ///
    /// See [`WebTexture::cube_faces`]. Both sky paths need it — the generated
    /// starfield arrives as a `DataTexture`, and leaving it on a sphere left
    /// the pole exactly as it was.
    #[cfg(feature = "planet")]
    #[wasm_bindgen(js_name = cubeFaces)]
    pub fn cube_faces(&self, face_size: u32) -> js_sys::Array {
        let n = if face_size == 0 {
            (self.inner.width / 4).clamp(64, 4096)
        } else {
            face_size
        };
        let out = js_sys::Array::new();
        for face in crate::planet::equirect_to_cube_faces(&self.inner, n) {
            out.push(&JsValue::from(WebDataTexture {
                inner: Arc::new(face),
            }));
        }
        out
    }

    /// A second handle on the same pixels, carrying its own UV transform.
    ///
    /// `Texture`'s bytes live behind their own `Arc`, so this clones a handle
    /// and not an image — and the renderer keys its GPU cache on those bytes,
    /// so every view of one buffer shares a single upload between them.
    #[wasm_bindgen(js_name = view)]
    pub fn view(&self, ox: f32, oy: f32, rx: f32, ry: f32) -> WebDataTexture {
        let mut inner = (*self.inner).clone();
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
        WebDataTexture {
            inner: Arc::new(inner),
        }
    }

    /// three.js `Texture.offset` / `Texture.repeat` — the UV sub-rectangle this
    /// texture covers. What lets several meshes share one atlas.
    #[wasm_bindgen(js_name = setUvTransform)]
    pub fn set_uv_transform(&mut self, ox: f32, oy: f32, rx: f32, ry: f32) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.offset = crate::math::Vector2::new(ox, oy);
        inner.repeat = crate::math::Vector2::new(rx, ry);
    }
}

// ======================================================================
//                             CURVES
// ======================================================================

#[wasm_bindgen]
pub struct WebLineCurve {
    inner: crate::LineCurve,
}
#[wasm_bindgen]
impl WebLineCurve {
    #[wasm_bindgen(constructor)]
    pub fn new(v1: &WebVector2, v2: &WebVector2) -> WebLineCurve {
        WebLineCurve {
            inner: crate::LineCurve::new(
                crate::Vector2::new(v1.x, v1.y),
                crate::Vector2::new(v2.x, v2.y),
            ),
        }
    }
}

#[wasm_bindgen]
pub struct WebLineCurve3 {
    inner: crate::LineCurve3,
}
#[wasm_bindgen]
impl WebLineCurve3 {
    #[wasm_bindgen(constructor)]
    pub fn new(v1: &WebVector3, v2: &WebVector3) -> WebLineCurve3 {
        WebLineCurve3 {
            inner: crate::LineCurve3::new(v1.inner, v2.inner),
        }
    }
}

#[wasm_bindgen]
pub struct WebEllipseCurve {
    inner: crate::EllipseCurve,
}
#[wasm_bindgen]
impl WebEllipseCurve {
    #[wasm_bindgen(constructor)]
    pub fn new(
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        a0: f32,
        a1: f32,
        clockwise: bool,
        rot: f32,
    ) -> WebEllipseCurve {
        WebEllipseCurve {
            inner: crate::EllipseCurve::new(
                crate::Vector2::new(cx, cy),
                rx,
                ry,
                a0,
                a1,
                clockwise,
                rot,
            ),
        }
    }
}

#[wasm_bindgen]
pub struct WebCatmullRomCurve3 {
    inner: crate::CatmullRomCurve3,
}
#[wasm_bindgen]
impl WebCatmullRomCurve3 {
    #[wasm_bindgen(constructor)]
    pub fn new(points_flat: Vec<f32>) -> WebCatmullRomCurve3 {
        let points = points_flat
            .chunks_exact(3)
            .map(|c| crate::Vector3::new(c[0], c[1], c[2]))
            .collect();
        WebCatmullRomCurve3 {
            inner: crate::CatmullRomCurve3::new(points),
        }
    }
}

#[wasm_bindgen]
pub struct WebPath {
    inner: crate::Path,
}
#[wasm_bindgen]
impl WebPath {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebPath {
        WebPath {
            inner: crate::Path::new(),
        }
    }
    #[wasm_bindgen(js_name = moveTo)]
    pub fn move_to(&mut self, x: f32, y: f32) {
        self.inner.move_to(crate::Vector2::new(x, y));
    }
    #[wasm_bindgen(js_name = lineTo)]
    pub fn line_to(&mut self, x: f32, y: f32) {
        self.inner.line_to(crate::Vector2::new(x, y));
    }

    #[wasm_bindgen(js_name = quadraticCurveTo)]
    pub fn quadratic_curve_to(&mut self, cpx: f32, cpy: f32, x: f32, y: f32) {
        self.inner
            .quadratic_curve_to(crate::Vector2::new(cpx, cpy), crate::Vector2::new(x, y));
    }

    #[wasm_bindgen(js_name = bezierCurveTo)]
    pub fn bezier_curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        self.inner.bezier_curve_to(
            crate::Vector2::new(c1x, c1y),
            crate::Vector2::new(c2x, c2y),
            crate::Vector2::new(x, y),
        );
    }

    pub fn arc(&mut self, cx: f32, cy: f32, radius: f32, start: f32, end: f32, clockwise: bool) {
        self.inner
            .arc(crate::Vector2::new(cx, cy), radius, start, end, clockwise);
    }

    /// Sample the path as a flat polyline — `divisions` points per sub-curve.
    #[wasm_bindgen(js_name = getPoints)]
    pub fn get_points(&self, divisions: usize) -> Vec<f32> {
        self.inner
            .get_points(divisions.max(1))
            .into_iter()
            .flat_map(|p| [p.x, p.y])
            .collect()
    }

    /// The path as an SVG `d` string, with its curves intact rather than
    /// flattened. Note that SVG's y axis points down and these coordinates are
    /// y-up, so a document usually wants a `scale(1,-1)` around them.
    #[wasm_bindgen(js_name = toSvgPathData)]
    pub fn to_svg_path_data(&self, precision: usize) -> String {
        self.inner.to_svg_path_data(precision)
    }

    /// Read an SVG `d` string — the inbound half, for turning artwork into
    /// geometry.
    #[wasm_bindgen(js_name = fromSvgPathData)]
    pub fn from_svg_path_data(d: &str) -> Result<WebPath, JsValue> {
        crate::Path::from_svg_path_data(d)
            .map(|inner| WebPath { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[wasm_bindgen]
pub struct WebShape {
    pub(crate) inner: crate::Shape,
}
#[wasm_bindgen]
impl WebShape {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebShape {
        WebShape {
            inner: crate::Shape::new(),
        }
    }

    #[wasm_bindgen(js_name = fromPath)]
    pub fn from_path(outline: &WebPath) -> WebShape {
        WebShape {
            inner: crate::Shape::from_path(
                crate::Path::from_svg_path_data(&outline.inner.to_svg_path_data(6))
                    .unwrap_or_default(),
            ),
        }
    }

    #[wasm_bindgen(js_name = addHole)]
    pub fn add_hole(&mut self, hole: &WebPath) {
        if let Ok(p) = crate::Path::from_svg_path_data(&hole.inner.to_svg_path_data(6)) {
            self.inner.add_hole(p);
        }
    }

    /// Outline then holes, as one `d` string. Fill it with
    /// `fill-rule="evenodd"` — see the Rust-side docs for why.
    #[wasm_bindgen(js_name = toSvgPathData)]
    pub fn to_svg_path_data(&self, precision: usize) -> String {
        self.inner.to_svg_path_data(precision)
    }

    /// Read SVG path data as an outline plus holes: first subpath is the
    /// outline, the rest are holes.
    #[wasm_bindgen(js_name = fromSvgPathData)]
    pub fn from_svg_path_data(d: &str) -> Result<WebShape, JsValue> {
        crate::Shape::from_svg_path_data(d)
            .map(|inner| WebShape { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ======================================================================
//                            ANIMATION
// ======================================================================

#[wasm_bindgen]
pub struct WebAnimationClip {
    pub(crate) inner: crate::AnimationClip,
}
#[wasm_bindgen]
impl WebAnimationClip {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, _duration: f32) -> WebAnimationClip {
        WebAnimationClip {
            inner: crate::AnimationClip::empty(name),
        }
    }
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f32 {
        self.inner.duration
    }
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }
}

#[wasm_bindgen]
pub struct WebAnimationMixer {
    inner: crate::AnimationMixer,
}
#[wasm_bindgen]
impl WebAnimationMixer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAnimationMixer {
        WebAnimationMixer {
            inner: crate::AnimationMixer::new(),
        }
    }
    #[wasm_bindgen(js_name = clipAction)]
    pub fn clip_action(&mut self, clip: WebAnimationClip) -> usize {
        self.inner.clip_action(clip.inner)
    }
    pub fn update(&mut self, scene: &mut WebScene, delta: f32) {
        self.inner.update(&mut scene.inner, delta);
    }
}

// ======================================================================
//                            CONTROLS
// ======================================================================

#[wasm_bindgen]
pub struct WebOrbitControls {
    inner: crate::OrbitControls,
}
#[wasm_bindgen]
impl WebOrbitControls {
    #[wasm_bindgen(constructor)]
    pub fn new(camera: &WebCamera) -> WebOrbitControls {
        let c = match &camera.inner {
            CameraInner::Perspective(p) => p.clone(),
            _ => crate::PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
        };
        WebOrbitControls {
            inner: crate::OrbitControls::new(&c),
        }
    }
    #[wasm_bindgen(js_name = setLimits)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_limits(
        &mut self,
        min_distance: f32,
        max_distance: f32,
        min_polar_angle: f32,
        max_polar_angle: f32,
        rotate_speed: f32,
        zoom_speed: f32,
        pan_speed: f32,
        damping: f32,
    ) {
        self.inner.min_distance = min_distance.max(0.0);
        self.inner.max_distance = max_distance.max(self.inner.min_distance);
        self.inner.min_polar_angle = min_polar_angle;
        self.inner.max_polar_angle = max_polar_angle;
        self.inner.rotate_speed = rotate_speed;
        self.inner.zoom_speed = zoom_speed;
        self.inner.pan_speed = pan_speed;
        self.inner.damping = damping;
        self.inner.enable_damping = damping > 0.0;
    }

    /// Enable / tune inertia and auto-rotate (three.js parity).
    #[wasm_bindgen(js_name = setMotion)]
    pub fn set_motion(
        &mut self,
        enable_damping: bool,
        damping: f32,
        auto_rotate: bool,
        auto_rotate_speed: f32,
    ) {
        self.inner.enable_damping = enable_damping;
        self.inner.damping = damping.clamp(0.0, 1.0);
        self.inner.auto_rotate = auto_rotate;
        self.inner.auto_rotate_speed = auto_rotate_speed;
    }

    pub fn update(
        &mut self,
        camera: &mut WebCamera,
        dx: f32,
        dy: f32,
        wheel: f32,
        rotating: bool,
        panning: bool,
        w: f32,
        h: f32,
    ) {
        self.update_dt(camera, dx, dy, wheel, rotating, panning, w, h, 1.0 / 60.0);
    }

    #[wasm_bindgen(js_name = updateDt)]
    #[allow(clippy::too_many_arguments)]
    pub fn update_dt(
        &mut self,
        camera: &mut WebCamera,
        dx: f32,
        dy: f32,
        wheel: f32,
        rotating: bool,
        panning: bool,
        w: f32,
        h: f32,
        dt: f32,
    ) {
        let ev = crate::PointerEvent {
            dx,
            dy,
            wheel,
            rotating,
            panning,
        };
        if let CameraInner::Perspective(c) = &mut camera.inner {
            self.inner.update_dt(ev, c, (w, h), dt);
        }
    }

    #[wasm_bindgen(js_name = reseedFromCamera)]
    pub fn reseed_from_camera(&mut self, camera: &WebCamera) {
        if let CameraInner::Perspective(c) = &camera.inner {
            self.inner.reseed_from_camera(c);
        }
    }

    /// Orbit centre.
    #[wasm_bindgen(js_name = setTarget)]
    pub fn set_target(&mut self, x: f32, y: f32, z: f32) {
        self.inner.target = crate::Vector3::new(x, y, z);
    }
}

#[wasm_bindgen]
pub struct WebTrackballControls {
    inner: crate::TrackballControls,
}
#[wasm_bindgen]
impl WebTrackballControls {
    #[wasm_bindgen(constructor)]
    pub fn new(camera: &WebCamera) -> WebTrackballControls {
        let c = match &camera.inner {
            CameraInner::Perspective(p) => p.clone(),
            _ => crate::PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
        };
        WebTrackballControls {
            inner: crate::TrackballControls::new(&c),
        }
    }
    pub fn update(
        &mut self,
        camera: &mut WebCamera,
        dx: f32,
        dy: f32,
        wheel: f32,
        rotating: bool,
        panning: bool,
        w: f32,
        h: f32,
    ) {
        let ev = crate::PointerEvent {
            dx,
            dy,
            wheel,
            rotating,
            panning,
        };
        if let CameraInner::Perspective(c) = &mut camera.inner {
            self.inner.update(ev, c, (w, h));
        }
    }
}

#[wasm_bindgen]
pub struct WebArcballControls {
    inner: crate::ArcballControls,
}
#[wasm_bindgen]
impl WebArcballControls {
    #[wasm_bindgen(constructor)]
    pub fn new(camera: &WebCamera) -> WebArcballControls {
        let c = match &camera.inner {
            CameraInner::Perspective(p) => p.clone(),
            _ => crate::PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
        };
        WebArcballControls {
            inner: crate::ArcballControls::new(&c),
        }
    }
    #[wasm_bindgen(js_name = setTarget)]
    pub fn set_target(&mut self, x: f32, y: f32, z: f32) {
        self.inner.target = crate::Vector3::new(x, y, z);
    }
    pub fn update(
        &mut self,
        camera: &mut WebCamera,
        ndc_x: f32,
        ndc_y: f32,
        wheel: f32,
        rotating: bool,
    ) {
        let ev = crate::PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel,
            rotating,
            panning: false,
        };
        let ndc = crate::Vector2::new(ndc_x, ndc_y);
        if let CameraInner::Perspective(c) = &mut camera.inner {
            self.inner.update(ndc, ev, c);
        }
    }
}

#[wasm_bindgen]
pub struct WebFirstPersonControls {
    inner: crate::FirstPersonControls,
}
#[wasm_bindgen]
impl WebFirstPersonControls {
    #[wasm_bindgen(constructor)]
    pub fn new(camera: &WebCamera) -> WebFirstPersonControls {
        let c = match &camera.inner {
            CameraInner::Perspective(p) => p.clone(),
            _ => crate::PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
        };
        WebFirstPersonControls {
            inner: crate::FirstPersonControls::new(&c),
        }
    }
    pub fn update(&mut self, camera: &mut WebCamera, dx: f32, dy: f32, dt: f32, rotating: bool) {
        let ev = crate::PointerEvent {
            dx,
            dy,
            wheel: 0.0,
            rotating,
            panning: false,
        };
        if let CameraInner::Perspective(c) = &mut camera.inner {
            self.inner.update(ev, c, dt);
        }
    }
    #[wasm_bindgen(js_name = setMoveInput)]
    pub fn set_move_input(&mut self, forward: f32, right: f32, up: f32) {
        self.inner.move_input = crate::Vector3::new(forward, right, up);
    }
}

#[wasm_bindgen]
pub struct WebPointerLockControls {
    inner: crate::PointerLockControls,
}
#[wasm_bindgen]
impl WebPointerLockControls {
    #[wasm_bindgen(constructor)]
    pub fn new(camera: &WebCamera) -> WebPointerLockControls {
        let c = match &camera.inner {
            CameraInner::Perspective(p) => p.clone(),
            _ => crate::PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0),
        };
        WebPointerLockControls {
            inner: crate::PointerLockControls::new(&c),
        }
    }
}

// ======================================================================
//                              HELPERS
// ======================================================================

#[wasm_bindgen]
pub struct WebAxesHelper {
    obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebAxesHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(size: f32) -> WebAxesHelper {
        WebAxesHelper {
            obj: crate::AxesHelper::new(size),
        }
    }
}

#[wasm_bindgen]
pub struct WebGridHelper {
    obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebGridHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(size: f32, divisions: usize, color1: u32, color2: u32) -> WebGridHelper {
        WebGridHelper {
            obj: crate::GridHelper::new_from_hex(size, divisions, color1, color2),
        }
    }
}

#[wasm_bindgen]
pub struct WebBoxHelper {
    obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebBoxHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(bb: &WebBox3) -> WebBoxHelper {
        WebBoxHelper {
            obj: crate::BoxHelper::new(&bb.inner, crate::Color::WHITE),
        }
    }
}

#[wasm_bindgen]
pub struct WebPolarGridHelper {
    obj: crate::core::Object3D,
}
#[wasm_bindgen]
impl WebPolarGridHelper {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, segments: usize, circles: usize) -> WebPolarGridHelper {
        WebPolarGridHelper {
            obj: crate::PolarGridHelper::default_(radius, segments, circles),
        }
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Add a helper as a scene-graph child (returns its handle for later removal).
    #[wasm_bindgen(js_name = addHelper)]
    pub fn add_helper(
        &mut self,
        axes: Option<WebAxesHelper>,
        grid: Option<WebGridHelper>,
    ) -> WebObjectHandle {
        let obj = if let Some(a) = axes {
            a.obj
        } else if let Some(g) = grid {
            g.obj
        } else {
            crate::core::Object3D::group()
        };
        let id = self.inner.add(obj);
        WebObjectHandle { id }
    }
}

// ======================================================================
//                              LOADERS
// ======================================================================

#[wasm_bindgen]
pub struct WebObjLoader;
#[wasm_bindgen]
impl WebObjLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebObjLoader {
        WebObjLoader
    }
    pub fn parse(&self, src: &str) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::ObjLoader::parse(src)),
        }
    }
}

#[wasm_bindgen]
pub struct WebStlLoader;
#[wasm_bindgen]
impl WebStlLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebStlLoader {
        WebStlLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::StlLoader::parse(&bytes)),
        }
    }
}

#[wasm_bindgen]
pub struct WebPlyLoader;
#[wasm_bindgen]
impl WebPlyLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebPlyLoader {
        WebPlyLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> WebGeometry {
        WebGeometry {
            inner: Arc::new(crate::PlyLoader::parse(&bytes)),
        }
    }
}

#[wasm_bindgen]
pub struct WebHdrLoader;
#[wasm_bindgen]
impl WebHdrLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebHdrLoader {
        WebHdrLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::HdrLoader::parse(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

#[wasm_bindgen]
pub struct WebFbxLoader;
#[wasm_bindgen]
impl WebFbxLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFbxLoader {
        WebFbxLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebGeometry, JsValue> {
        crate::FbxLoader::parse(&bytes)
            .map(|g| WebGeometry { inner: Arc::new(g) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

#[wasm_bindgen]
pub struct WebColladaLoader;
#[wasm_bindgen]
impl WebColladaLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebColladaLoader {
        WebColladaLoader
    }
    pub fn parse(&self, src: &str) -> Result<WebGeometry, JsValue> {
        crate::ColladaLoader::parse(src)
            .map(|g| WebGeometry { inner: Arc::new(g) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

#[wasm_bindgen]
pub struct WebExrLoader;
#[wasm_bindgen]
impl WebExrLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebExrLoader {
        WebExrLoader
    }
    pub fn parse(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::ExrLoader::parse(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }

    /// Decode to half-float instead of clipping to 8 bits.
    ///
    /// EXR is scene-referred, and a star map is the extreme case: over half of
    /// NASA's Deep Star Maps sits below a hundredth of full scale, so the 8-bit
    /// path leaves the galaxy as a handful of grey dots.
    #[wasm_bindgen(js_name = parseHdr)]
    pub fn parse_hdr(&self, bytes: Vec<u8>) -> Result<WebTexture, JsValue> {
        crate::ExrLoader::parse_hdr(&bytes)
            .map(|t| WebTexture { inner: Arc::new(t) })
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))
    }
}

// ======================================================================
//                               AUDIO
// ======================================================================

#[wasm_bindgen]
pub struct WebAudioListener {
    inner: crate::AudioListener,
}
#[wasm_bindgen]
impl WebAudioListener {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAudioListener {
        WebAudioListener {
            inner: crate::AudioListener::default(),
        }
    }
    #[wasm_bindgen(js_name = setMasterVolume)]
    pub fn set_master_volume(&mut self, v: f32) {
        self.inner.master_volume = v;
    }
}

#[wasm_bindgen]
pub struct WebAudio {
    inner: crate::Audio,
}
#[wasm_bindgen]
impl WebAudio {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebAudio {
        WebAudio {
            inner: crate::Audio::default(),
        }
    }
    #[wasm_bindgen(js_name = setVolume)]
    pub fn set_volume(&mut self, v: f32) {
        self.inner.volume = v;
    }
    #[wasm_bindgen(js_name = setLoop)]
    pub fn set_loop(&mut self, l: bool) {
        self.inner.loop_ = l;
    }
    pub fn play(&mut self) {
        self.inner.playing = true;
    }
    pub fn stop(&mut self) {
        self.inner.playing = false;
    }
}

// ======================================================================
//                           POST-PROCESSING
// ======================================================================

#[wasm_bindgen]
pub struct WebEffectComposer;
#[wasm_bindgen]
impl WebEffectComposer {
    /// EffectComposer construction requires a wgpu device; the wasm path
    /// builds one internally with the renderer. Use `renderer.composer()`
    /// to obtain a composer bound to the same surface.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebEffectComposer {
        WebEffectComposer
    }
}

#[wasm_bindgen]
pub struct WebRenderPass;
#[wasm_bindgen]
impl WebRenderPass {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebRenderPass {
        WebRenderPass
    }
}

#[wasm_bindgen]
pub struct WebBloomPass;
#[wasm_bindgen]
impl WebBloomPass {
    #[wasm_bindgen(constructor)]
    pub fn new(_strength: f32, _radius: f32, _threshold: f32) -> WebBloomPass {
        WebBloomPass
    }
}

#[wasm_bindgen]
pub struct WebFxaaPass;
#[wasm_bindgen]
impl WebFxaaPass {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFxaaPass {
        WebFxaaPass
    }
}

// ======================================================================
//                              EXTRAS
// ======================================================================

#[wasm_bindgen]
pub struct WebOctree {
    inner: crate::Octree,
}
#[wasm_bindgen]
impl WebOctree {
    #[wasm_bindgen(constructor)]
    pub fn new(bb: &WebBox3, max_depth: u32, max_points: usize) -> WebOctree {
        WebOctree {
            inner: crate::Octree::new(bb.inner, max_depth, max_points),
        }
    }
    pub fn insert(&mut self, p: &WebVector3) {
        self.inner.insert(p.inner);
    }
}

#[wasm_bindgen]
pub struct WebSimplexNoise;
#[wasm_bindgen]
impl WebSimplexNoise {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebSimplexNoise {
        WebSimplexNoise
    }
    pub fn noise2(&self, x: f32, y: f32) -> f32 {
        crate::SimplexNoise::noise2(x, y)
    }
}

#[wasm_bindgen]
pub struct WebMarchingCubes;
#[wasm_bindgen]
impl WebMarchingCubes {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMarchingCubes {
        WebMarchingCubes
    }
}

// ======================================================================
//                            ALT RENDERERS
// ======================================================================

#[wasm_bindgen]
pub struct WebCss2dRenderer {
    inner: crate::Css2dRenderer,
}
#[wasm_bindgen]
impl WebCss2dRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(w: u32, h: u32) -> WebCss2dRenderer {
        WebCss2dRenderer {
            inner: crate::Css2dRenderer::new(w, h),
        }
    }
}

#[wasm_bindgen]
pub struct WebSvgRenderer {
    inner: crate::SvgRenderer,
}
#[wasm_bindgen]
impl WebSvgRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new(w: u32, h: u32) -> WebSvgRenderer {
        WebSvgRenderer {
            inner: crate::SvgRenderer::new(w, h),
        }
    }

    /// `0` lit (default), `1` flat material colour, `2` wireframe.
    #[wasm_bindgen(js_name = setShading)]
    pub fn set_shading(&mut self, mode: u32) {
        self.inner.options.shading = match mode {
            1 => crate::SvgShading::Flat,
            2 => crate::SvgShading::Wireframe,
            _ => crate::SvgShading::Lit,
        };
    }

    /// Same three.js constants as
    /// [`WebRenderer::setToneMapping`](WebRenderer::set_tone_mapping): `0`
    /// NoToneMapping, `1` Linear, `4` ACESFilmic.
    #[wasm_bindgen(js_name = setToneMapping)]
    pub fn set_tone_mapping(&mut self, mode: u32, exposure: f32) {
        let mapping = match mode {
            0 => crate::ToneMapping::None,
            4 => crate::ToneMapping::AcesFilmic,
            _ => crate::ToneMapping::Linear,
        };
        self.inner.set_tone_mapping(mapping, exposure);
    }

    /// Emit `scene.background` as a full-canvas rect. Off gives a transparent
    /// document that sits on whatever is behind it in the page.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, on: bool) {
        self.inner.options.background = on;
    }

    #[wasm_bindgen(js_name = setCullBackfaces)]
    pub fn set_cull_backfaces(&mut self, on: bool) {
        self.inner.options.cull_backfaces = on;
    }

    /// Hairline stroke per face, in pixels, that hides the light seams SVG
    /// leaves along shared edges. 0 disables it.
    #[wasm_bindgen(js_name = setSeamStroke)]
    pub fn set_seam_stroke(&mut self, px: f32) {
        self.inner.options.seam_stroke = px.max(0.0);
    }

    /// Decimal places kept on coordinates. Lower is a smaller document.
    #[wasm_bindgen(js_name = setPrecision)]
    pub fn set_precision(&mut self, places: u32) {
        self.inner.options.precision = places.min(8) as usize;
    }

    /// Draw a face's edges as Bézier curves when a straight one would miss the
    /// real surface by more than this many pixels. Pass a value of 0 or less to
    /// keep every edge straight.
    #[wasm_bindgen(js_name = setCurveTolerance)]
    pub fn set_curve_tolerance(&mut self, px: f32) {
        self.inner.options.curve_tolerance = (px > 0.0).then_some(px);
    }

    /// Split faces spanning more than this fraction of their own distance from
    /// the camera, so the depth sort has something meaningful to order. 0 or
    /// less disables splitting.
    #[wasm_bindgen(js_name = setDepthSplit)]
    pub fn set_depth_split(&mut self, tolerance: f32) {
        self.inner.options.depth_split = (tolerance > 0.0).then_some(tolerance);
    }

    /// Depth-sort faces back to front. Off only makes sense when the caller has
    /// already ordered the scene.
    #[wasm_bindgen(js_name = setSort)]
    pub fn set_sort(&mut self, on: bool) {
        self.inner.options.sort = on;
    }

    /// Resize the canvas, keeping every option set so far.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.inner.set_size(width, height);
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    /// Background colour as `0xRRGGBB`, overriding `scene.background`. Mirrors
    /// three.js's `setClearColor`.
    #[wasm_bindgen(js_name = setClearColor)]
    pub fn set_clear_color(&mut self, hex: u32, alpha: f32) {
        self.inner.set_clear_color(
            Some(crate::Color::from_hex(hex)),
            Some(alpha.clamp(0.0, 1.0)),
        );
    }

    /// Go back to taking the background from the scene.
    #[wasm_bindgen(js_name = clearClearColor)]
    pub fn clear_clear_color(&mut self) {
        self.inner.set_clear_color(None, None);
    }

    #[wasm_bindgen(js_name = renderToString)]
    pub fn render_to_string(&self, scene: &mut WebScene, camera: &WebCamera) -> String {
        match &camera.inner {
            CameraInner::Perspective(c) => self.inner.render_to_string(&mut scene.inner, c),
            CameraInner::Orthographic(c) => self.inner.render_to_string(&mut scene.inner, c),
        }
    }
}

/// Wrap an RGBA frame — typically read back off a canvas — in an SVG document
/// as an embedded PNG. The counterpart to [`WebSvgRenderer`] for when the scene
/// needs the real renderer's textures, shadows or post-fx and a per-face
/// painter's algorithm will not do.
#[wasm_bindgen(js_name = svgFromRgba)]
pub fn svg_from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<String, JsValue> {
    let want = width as usize * height as usize * 4;
    if rgba.len() != want {
        return Err(JsValue::from_str(&format!(
            "svgFromRgba: expected {want} bytes for {width}x{height}, got {}",
            rgba.len()
        )));
    }
    Ok(crate::svg_from_rgba(width, height, rgba))
}

// ======================================================================
//                               STATS
// ======================================================================

#[wasm_bindgen]
pub struct WebStats {
    inner: crate::Stats,
}
#[wasm_bindgen]
impl WebStats {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebStats {
        WebStats {
            inner: crate::Stats::new(),
        }
    }
    pub fn begin(&mut self) {
        self.inner.begin();
    }
    pub fn end(&mut self) {
        self.inner.end();
    }
    #[wasm_bindgen(getter)]
    pub fn fps(&self) -> f32 {
        self.inner.fps
    }
    #[wasm_bindgen(js_name = frameMs, getter)]
    pub fn frame_ms(&self) -> f32 {
        self.inner.frame_ms
    }
}

// ======================================================================
//                           RAYCASTER
// ======================================================================

#[wasm_bindgen]
pub struct WebRaycaster {
    inner: crate::Raycaster,
}
#[wasm_bindgen]
impl WebRaycaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebRaycaster {
        WebRaycaster {
            inner: crate::Raycaster::default(),
        }
    }
    #[wasm_bindgen(js_name = setFromCamera)]
    pub fn set_from_camera(&mut self, ndc_x: f32, ndc_y: f32, camera: &WebCamera) {
        match &camera.inner {
            CameraInner::Perspective(c) => self
                .inner
                .set_from_camera_perspective(crate::Vector2::new(ndc_x, ndc_y), c),
            CameraInner::Orthographic(c) => self
                .inner
                .set_from_camera_ortho(crate::Vector2::new(ndc_x, ndc_y), c),
        }
    }
}

// ======================================================================
//                             CLOCK
// ======================================================================

#[wasm_bindgen]
pub struct WebClock {
    inner: crate::Clock,
}
#[wasm_bindgen]
impl WebClock {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebClock {
        WebClock {
            inner: crate::Clock::new(true),
        }
    }
    #[wasm_bindgen(js_name = getDelta)]
    pub fn get_delta(&mut self) -> f64 {
        self.inner.get_delta()
    }
    #[wasm_bindgen(js_name = getElapsedTime)]
    pub fn get_elapsed_time(&mut self) -> f64 {
        self.inner.get_elapsed_time()
    }
}

// ======================================================================
//                       OBJECT3D + NESTED OBJECTS
// ======================================================================

/// Generic Object3D handle (Group or any scene-graph node).
#[wasm_bindgen]
pub struct WebObject3D {
    pub(crate) inner: Option<crate::core::Object3D>,
}

#[wasm_bindgen]
impl WebObject3D {
    /// Empty group — three.js `new Group()` / `new Object3D()`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebObject3D {
        WebObject3D {
            inner: Some(crate::core::Object3D::group()),
        }
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Add a generic Object3D (group, helper, etc.) to the scene.
    #[wasm_bindgen(js_name = addObject)]
    pub fn add_object(&mut self, obj: &mut WebObject3D) -> WebObjectHandle {
        let o = obj
            .inner
            .take()
            .unwrap_or_else(crate::core::Object3D::group);
        WebObjectHandle {
            id: self.inner.add(o),
        }
    }

    /// Remove an object from the scene (by handle).
    pub fn remove(&mut self, handle: &WebObjectHandle) {
        self.inner.arena.remove_child(handle.id);
    }

    /// Replace a mesh's geometry positions in place. Used for morph-target
    /// vertex blending and any dynamic vertex animation.
    #[wasm_bindgen(js_name = setMeshPositions)]
    pub fn set_mesh_positions(&mut self, handle: &WebObjectHandle, positions: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            match &mut obj.kind {
                crate::core::ObjectKind::Mesh(mesh) => {
                    let mut new_geom = (*mesh.geometry).clone();
                    new_geom
                        .set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
                    mesh.geometry = std::sync::Arc::new(new_geom);
                }
                crate::core::ObjectKind::SkinnedMesh(sm) => {
                    let mut new_geom = (*sm.geometry).clone();
                    new_geom
                        .set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
                    sm.geometry = std::sync::Arc::new(new_geom);
                }
                _ => {}
            }
        }
    }

    /// Replace a mesh's geometry normals in place.
    #[wasm_bindgen(js_name = setMeshNormals)]
    pub fn set_mesh_normals(&mut self, handle: &WebObjectHandle, normals: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            if let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind {
                let mut new_geom = (*mesh.geometry).clone();
                new_geom.set_attribute("normal", crate::core::BufferAttribute::new(normals, 3));
                mesh.geometry = std::sync::Arc::new(new_geom);
            }
        }
    }

    /// Update a skinned mesh's bone matrices. Each matrix is 16 floats
    /// (column-major); the caller passes the matrices concatenated.
    /// Re-uploaded to the storage buffer on the next render.
    #[wasm_bindgen(js_name = setBoneMatrices)]
    pub fn set_bone_matrices(&mut self, handle: &WebObjectHandle, matrices: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            if let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind {
                let mut new_skel = (*sm.skeleton).clone();
                let mat_count = matrices.len() / 16;
                new_skel.bone_matrices.clear();
                for i in 0..mat_count {
                    let mut e = [0.0_f32; 16];
                    e.copy_from_slice(&matrices[i * 16..(i + 1) * 16]);
                    new_skel
                        .bone_matrices
                        .push(crate::math::Matrix4 { elements: e });
                }
                sm.skeleton = std::sync::Arc::new(new_skel);
            }
        }
    }

    /// Update a skinned mesh's per-vertex joint indices.
    #[wasm_bindgen(js_name = setSkinJoints)]
    pub fn set_skin_joints(&mut self, handle: &WebObjectHandle, joints: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            if let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind {
                let mut new_geom = (*sm.geometry).clone();
                new_geom.set_attribute("joint", crate::core::BufferAttribute::new(joints, 4));
                sm.geometry = std::sync::Arc::new(new_geom);
            }
        }
    }

    /// Update a skinned mesh's per-vertex bone weights.
    #[wasm_bindgen(js_name = setSkinWeights)]
    pub fn set_skin_weights(&mut self, handle: &WebObjectHandle, weights: Vec<f32>) {
        if let Some(obj) = self.inner.arena.get_mut(handle.id) {
            if let crate::core::ObjectKind::SkinnedMesh(sm) = &mut obj.kind {
                let mut new_geom = (*sm.geometry).clone();
                new_geom.set_attribute("weight", crate::core::BufferAttribute::new(weights, 4));
                sm.geometry = std::sync::Arc::new(new_geom);
            }
        }
    }

    /// Find the first object whose name matches.
    #[wasm_bindgen(js_name = getObjectByName)]
    pub fn get_object_by_name(&self, name: &str) -> Option<WebObjectHandle> {
        let root = self.inner.root;
        self.inner
            .arena
            .get_object_by_name(root, name)
            .map(|id| WebObjectHandle { id })
    }

    /// Set a name on the object identified by `handle`.
    #[wasm_bindgen(js_name = setName)]
    pub fn set_name(&mut self, handle: &WebObjectHandle, name: &str) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.name = name.into();
        }
    }

    /// Apply a translation to the object's local position.
    pub fn translate(&mut self, handle: &WebObjectHandle, dx: f32, dy: f32, dz: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.position = obj.position + crate::Vector3::new(dx, dy, dz);
        }
    }

    /// Set scale.
    #[wasm_bindgen(js_name = setScale)]
    pub fn set_scale(&mut self, handle: &WebObjectHandle, sx: f32, sy: f32, sz: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.scale = crate::Vector3::new(sx, sy, sz);
        }
    }

    /// Set visibility.
    #[wasm_bindgen(js_name = setVisible)]
    pub fn set_visible(&mut self, handle: &WebObjectHandle, visible: bool) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.visible = visible;
        }
    }

    /// three.js Object3D.renderOrder — lower draws first within transparency class.
    #[wasm_bindgen(js_name = setRenderOrder)]
    pub fn set_render_order(&mut self, handle: &WebObjectHandle, order: i32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.render_order = order;
        }
    }

    /// Look-at: orient the object so its -Z axis points at target.
    #[wasm_bindgen(js_name = lookAt)]
    pub fn look_at(&mut self, handle: &WebObjectHandle, x: f32, y: f32, z: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.look_at(crate::Vector3::new(x, y, z));
        }
    }

    /// Get the world-space position (after parent transforms).
    #[wasm_bindgen(js_name = getWorldPosition)]
    pub fn get_world_position(&mut self, handle: &WebObjectHandle) -> WebVector3 {
        self.inner.update_world();
        let p = self
            .inner
            .get(handle.id)
            .map(|o| o.world_position())
            .unwrap_or(crate::Vector3::ZERO);
        WebVector3 { inner: p }
    }
}

// ======================================================================
//                    MATH METHOD COMPLETENESS
// ======================================================================

#[wasm_bindgen]
impl WebVector3 {
    pub fn add(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner + o.inner;
        WebVector3 { inner: self.inner }
    }
    pub fn sub(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner - o.inner;
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = multiplyScalar)]
    pub fn multiply_scalar(&mut self, s: f32) -> WebVector3 {
        self.inner = self.inner * s;
        WebVector3 { inner: self.inner }
    }
    pub fn length(&self) -> f32 {
        self.inner.length()
    }
    #[wasm_bindgen(js_name = lengthSq)]
    pub fn length_sq(&self) -> f32 {
        self.inner.length_sq()
    }
    pub fn normalize(&mut self) -> WebVector3 {
        self.inner = self.inner.normalize();
        WebVector3 { inner: self.inner }
    }
    pub fn dot(&self, o: &WebVector3) -> f32 {
        self.inner.dot(o.inner)
    }
    pub fn cross(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner.cross(o.inner);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = distanceTo)]
    pub fn distance_to(&self, o: &WebVector3) -> f32 {
        self.inner.distance_to(o.inner)
    }
    pub fn lerp(&mut self, o: &WebVector3, t: f32) -> WebVector3 {
        self.inner = self.inner.lerp(o.inner, t);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = applyMatrix4)]
    pub fn apply_matrix4(&mut self, m: &WebMatrix4) -> WebVector3 {
        self.inner = self.inner.apply_matrix4(&m.inner);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = applyQuaternion)]
    pub fn apply_quaternion(&mut self, q: &WebQuaternion) -> WebVector3 {
        let qq = crate::Quaternion::new(q.x, q.y, q.z, q.w);
        self.inner = self.inner.apply_quaternion(qq);
        WebVector3 { inner: self.inner }
    }
}

#[wasm_bindgen]
impl WebColor {
    pub fn r(&self) -> f32 {
        self.inner.r
    }
    pub fn g(&self) -> f32 {
        self.inner.g
    }
    pub fn b(&self) -> f32 {
        self.inner.b
    }
    #[wasm_bindgen(js_name = setRGB)]
    pub fn set_rgb(&mut self, r: f32, g: f32, b: f32) -> WebColor {
        self.inner = crate::Color::new(r, g, b);
        *self
    }
    #[wasm_bindgen(js_name = setHex)]
    pub fn set_hex(&mut self, h: u32) -> WebColor {
        self.inner = crate::Color::from_hex(h);
        *self
    }
    pub fn lerp(&mut self, other: &WebColor, t: f32) -> WebColor {
        let r = self.inner.r + (other.inner.r - self.inner.r) * t;
        let g = self.inner.g + (other.inner.g - self.inner.g) * t;
        let b = self.inner.b + (other.inner.b - self.inner.b) * t;
        self.inner = crate::Color::new(r, g, b);
        *self
    }
    #[wasm_bindgen(js_name = getHex)]
    pub fn get_hex(&self) -> u32 {
        let r = (self.inner.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.inner.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.inner.b.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}

#[wasm_bindgen]
impl WebMatrix4 {
    pub fn multiply(&mut self, m: &WebMatrix4) -> WebMatrix4 {
        self.inner = self.inner.multiply(&m.inner);
        WebMatrix4 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = makeTranslation)]
    pub fn make_translation(x: f32, y: f32, z: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::translation(crate::Vector3::new(x, y, z)),
        }
    }
    #[wasm_bindgen(js_name = makeScale)]
    pub fn make_scale(x: f32, y: f32, z: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::scale(crate::Vector3::new(x, y, z)),
        }
    }
    #[wasm_bindgen(js_name = lookAt)]
    pub fn look_at(eye: &WebVector3, target: &WebVector3, up: &WebVector3) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::look_at(eye.inner, target.inner, up.inner),
        }
    }
    pub fn determinant(&self) -> f32 {
        self.inner.determinant()
    }
}

#[wasm_bindgen]
impl WebQuaternion {
    pub fn normalize(&mut self) -> WebQuaternion {
        let q = crate::Quaternion::new(self.x, self.y, self.z, self.w).normalize();
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
    pub fn invert(&mut self) -> WebQuaternion {
        let q = crate::Quaternion::new(self.x, self.y, self.z, self.w).invert();
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
    pub fn multiply(&mut self, o: &WebQuaternion) -> WebQuaternion {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        let c = a.multiply(b);
        self.x = c.x;
        self.y = c.y;
        self.z = c.z;
        self.w = c.w;
        *self
    }
    pub fn slerp(&mut self, o: &WebQuaternion, t: f32) -> WebQuaternion {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        let c = a.slerp(b, t);
        self.x = c.x;
        self.y = c.y;
        self.z = c.z;
        self.w = c.w;
        *self
    }
    pub fn dot(&self, o: &WebQuaternion) -> f32 {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        a.dot(b)
    }
    #[wasm_bindgen(js_name = setFromAxisAngle)]
    pub fn set_from_axis_angle(&mut self, axis: &WebVector3, angle: f32) -> WebQuaternion {
        let q = crate::Quaternion::from_axis_angle(axis.inner, angle);
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
}

// ======================================================================
//                     MATH UTILITIES (THREE.MathUtils)
// ======================================================================

#[wasm_bindgen]
pub struct WebMathUtils;

#[wasm_bindgen]
impl WebMathUtils {
    pub fn clamp(v: f32, min: f32, max: f32) -> f32 {
        v.clamp(min, max)
    }
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    #[wasm_bindgen(js_name = degToRad)]
    pub fn deg_to_rad(d: f32) -> f32 {
        d.to_radians()
    }
    #[wasm_bindgen(js_name = radToDeg)]
    pub fn rad_to_deg(r: f32) -> f32 {
        r.to_degrees()
    }
    #[wasm_bindgen(js_name = mapLinear)]
    pub fn map_linear(x: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
        b1 + (x - a1) * (b2 - b1) / (a2 - a1)
    }
    #[wasm_bindgen(js_name = smoothstep)]
    pub fn smoothstep(x: f32, min: f32, max: f32) -> f32 {
        if x <= min {
            return 0.0;
        }
        if x >= max {
            return 1.0;
        }
        let t = (x - min) / (max - min);
        t * t * (3.0 - 2.0 * t)
    }
    #[wasm_bindgen(js_name = euclideanModulo)]
    pub fn euclidean_modulo(n: f32, m: f32) -> f32 {
        ((n % m) + m) % m
    }
    #[wasm_bindgen(js_name = isPowerOfTwo)]
    pub fn is_power_of_two(n: u32) -> bool {
        n != 0 && (n & (n - 1)) == 0
    }
}

// ======================================================================
//                  BUFFER GEOMETRY + BUFFER ATTRIBUTE
// ======================================================================

#[wasm_bindgen]
pub struct WebBufferAttribute {
    pub(crate) inner: crate::BufferAttribute,
}

#[wasm_bindgen]
impl WebBufferAttribute {
    #[wasm_bindgen(constructor)]
    pub fn new(array: Vec<f32>, item_size: usize) -> WebBufferAttribute {
        WebBufferAttribute {
            inner: crate::BufferAttribute::new(array, item_size),
        }
    }
    pub fn count(&self) -> usize {
        self.inner.count()
    }
    #[wasm_bindgen(js_name = itemSize, getter)]
    pub fn item_size(&self) -> usize {
        self.inner.item_size
    }
    pub fn array(&self) -> Vec<f32> {
        self.inner.array.clone()
    }
}

#[wasm_bindgen]
pub struct WebBufferGeometry {
    pub(crate) inner: Arc<crate::BufferGeometry>,
}

#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::BufferGeometry::new()),
        }
    }
    #[wasm_bindgen(js_name = setAttribute)]
    pub fn set_attribute(&mut self, name: &str, attr: WebBufferAttribute) {
        Arc::make_mut(&mut self.inner).set_attribute(name, attr.inner);
    }
    #[wasm_bindgen(js_name = setIndex)]
    pub fn set_index(&mut self, indices: Vec<u32>) {
        Arc::make_mut(&mut self.inner).set_index(indices);
    }
    #[wasm_bindgen(js_name = computeBoundingBox)]
    pub fn compute_bounding_box(&mut self) -> WebBox3 {
        WebBox3 {
            inner: Arc::make_mut(&mut self.inner).compute_bounding_box(),
        }
    }
    #[wasm_bindgen(js_name = computeBoundingSphere)]
    pub fn compute_bounding_sphere(&mut self) -> WebSphere {
        WebSphere {
            inner: Arc::make_mut(&mut self.inner).compute_bounding_sphere(),
        }
    }
    #[wasm_bindgen(js_name = computeVertexNormals)]
    pub fn compute_vertex_normals(&mut self) {
        crate::compute_vertex_normals(Arc::make_mut(&mut self.inner));
    }
    #[wasm_bindgen(js_name = hasAttribute)]
    pub fn has_attribute(&self, name: &str) -> bool {
        self.inner.get_attribute(name).is_some()
    }
    #[wasm_bindgen(js_name = drawCount)]
    pub fn draw_count(&self) -> usize {
        self.inner.draw_count()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(js_name = getAttributeArray)]
    pub fn get_attribute_array(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.get_attribute(name).map(|a| a.array.clone())
    }

    #[wasm_bindgen(js_name = getAttributeItemSize)]
    pub fn get_attribute_item_size(&self, name: &str) -> Option<usize> {
        self.inner.get_attribute(name).map(|a| a.item_size)
    }

    #[wasm_bindgen(js_name = getIndexArray)]
    pub fn get_index_array(&self) -> Option<Vec<u32>> {
        self.inner.index.clone()
    }
}

// Convert a user-built BufferGeometry into a WebGeometry handle.
#[wasm_bindgen]
impl WebGeometry {
    #[wasm_bindgen(js_name = fromBufferGeometry)]
    pub fn from_buffer_geometry(g: WebBufferGeometry) -> WebGeometry {
        WebGeometry { inner: g.inner }
    }
}

// ======================================================================
//                         MESH BVH (feature mesh-bvh)
// ======================================================================

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
pub struct WebMeshBvh {
    pub(crate) inner: Arc<crate::mesh_bvh::MeshBvh>,
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
impl WebMeshBvh {
    #[wasm_bindgen(constructor)]
    pub fn new(geometry: &WebBufferGeometry) -> Result<WebMeshBvh, JsValue> {
        Self::new_with_options(geometry, 40, 10)
    }

    #[wasm_bindgen(js_name = newWithOptions)]
    pub fn new_with_options(
        geometry: &WebBufferGeometry,
        max_depth: u32,
        max_leaf_tris: u32,
    ) -> Result<WebMeshBvh, JsValue> {
        Self::new_with_options_full(
            geometry,
            max_depth,
            max_leaf_tris,
            crate::mesh_bvh::CENTER,
            0,
            0,
            false,
        )
    }

    #[wasm_bindgen(js_name = newWithOptionsFull)]
    pub fn new_with_options_full(
        geometry: &WebBufferGeometry,
        max_depth: u32,
        max_leaf_tris: u32,
        strategy: u32,
        offset: u32,
        count: u32,
        indirect: bool,
    ) -> Result<WebMeshBvh, JsValue> {
        let options = crate::mesh_bvh::BuildOptions {
            strategy,
            max_depth,
            max_leaf_tris,
            offset,
            count,
            indirect,
        };
        let bvh = crate::mesh_bvh::MeshBvh::build(&geometry.inner, options)
            .ok_or_else(|| JsValue::from_str("failed to build MeshBVH"))?;
        Ok(WebMeshBvh {
            inner: Arc::new(bvh),
        })
    }

    /// Packed hits: per entry `[distance, px, py, pz, face_index, u, v]`.
    pub fn raycast(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        near: f32,
        far: f32,
    ) -> Vec<f32> {
        let ray = crate::Ray::new(
            crate::Vector3::new(ox, oy, oz),
            crate::Vector3::new(dx, dy, dz),
        );
        let hits = self.inner.raycast(&ray, near, far, true);
        pack_hits(&hits)
    }

    /// Closest hit packed as `[distance, px, py, pz, face_index, u, v]`, or empty vec.
    #[wasm_bindgen(js_name = raycastFirst)]
    pub fn raycast_first(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        near: f32,
        far: f32,
        side: u32,
    ) -> Vec<f32> {
        let ray = crate::Ray::new(
            crate::Vector3::new(ox, oy, oz),
            crate::Vector3::new(dx, dy, dz),
        );
        match self.inner.raycast_first_with_side(&ray, near, far, side) {
            Some(h) => pack_hits(&[h]),
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = resolveTriangleIndex)]
    pub fn resolve_triangle_index(&self, bvh_triangle_index: u32) -> i32 {
        match self
            .inner
            .resolve_triangle_index(bvh_triangle_index as usize)
        {
            Some(i) => i as i32,
            None => -1,
        }
    }

    /// Packed triangle pairs: `[ia, ib, ia, ib, ...]` (BVH-layout indices).
    #[wasm_bindgen(js_name = bvhcast)]
    pub fn bvhcast(&self, other: &WebMeshBvh, matrix: &[f32]) -> Vec<u32> {
        if matrix.len() < 16 {
            return Vec::new();
        }
        let mut elems = [0.0f32; 16];
        elems.copy_from_slice(&matrix[..16]);
        let m = crate::Matrix4 { elements: elems };
        let pairs = self.inner.bvhcast(&other.inner, &m);
        let mut out = Vec::with_capacity(pairs.len() * 2);
        for (a, b) in pairs {
            out.push(a as u32);
            out.push(b as u32);
        }
        out
    }

    #[wasm_bindgen(js_name = getBoundingBox)]
    pub fn get_bounding_box(&self) -> WebBox3 {
        WebBox3 {
            inner: self.inner.bounding_box(),
        }
    }

    #[wasm_bindgen(js_name = nodeBuffer)]
    pub fn node_buffer(&self) -> Vec<f32> {
        self.inner.node_buffer().to_vec()
    }

    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[wasm_bindgen(js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    #[wasm_bindgen(js_name = triangleIndices)]
    pub fn triangle_indices(&self) -> Vec<u32> {
        self.inner
            .triangle_indices()
            .iter()
            .flat_map(|(a, b, c)| [*a, *b, *c])
            .collect()
    }

    pub fn positions(&self) -> Vec<f32> {
        self.inner.positions().to_vec()
    }

    #[wasm_bindgen(js_name = triangleOrder)]
    pub fn triangle_order(&self) -> Vec<u32> {
        self.inner
            .triangle_order()
            .iter()
            .map(|&i| i as u32)
            .collect()
    }

    #[wasm_bindgen(js_name = refit)]
    pub fn refit(&mut self, positions: &[f32]) {
        Arc::make_mut(&mut self.inner).refit(positions);
    }

    #[wasm_bindgen(js_name = intersectsBox)]
    pub fn intersects_box(
        &self,
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
    ) -> bool {
        let b = crate::Box3::new(
            crate::Vector3::new(min_x, min_y, min_z),
            crate::Vector3::new(max_x, max_y, max_z),
        );
        self.inner.intersects_box(&b)
    }

    #[wasm_bindgen(js_name = intersectsSphere)]
    pub fn intersects_sphere(&self, cx: f32, cy: f32, cz: f32, radius: f32) -> bool {
        let s = crate::Sphere::new(crate::Vector3::new(cx, cy, cz), radius);
        self.inner.intersects_sphere(&s)
    }

    /// Closest point: `[px, py, pz, distance, face_index]`.
    #[wasm_bindgen(js_name = closestPointToPoint)]
    pub fn closest_point_to_point(&self, px: f32, py: f32, pz: f32) -> Vec<f32> {
        let (point, dist, tri) = self
            .inner
            .closest_point_to_point(crate::Vector3::new(px, py, pz));
        vec![point.x, point.y, point.z, dist, tri as f32]
    }

    /// Serialize to flat arrays for JS: version, nodeBuffer, triangleOrder, triangleIndices, positions.
    #[wasm_bindgen(js_name = serialize)]
    pub fn serialize(&self) -> js_sys::Array {
        let data = self.inner.serialize();
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_f64(data.version as f64));
        arr.push(&js_sys::Float32Array::from(data.node_buffer.as_slice()).into());
        arr.push(&js_sys::Uint32Array::from(data.triangle_order.as_slice()).into());
        arr.push(&js_sys::Uint32Array::from(data.triangle_indices.as_slice()).into());
        arr.push(&js_sys::Float32Array::from(data.positions.as_slice()).into());
        arr
    }

    #[wasm_bindgen(js_name = deserialize)]
    pub fn deserialize(
        version: u32,
        node_buffer: &[f32],
        triangle_order: &[u32],
        triangle_indices: &[u32],
        positions: &[f32],
    ) -> Result<WebMeshBvh, JsValue> {
        let data = crate::mesh_bvh::SerializedMeshBvh {
            version,
            node_buffer: node_buffer.to_vec(),
            triangle_order: triangle_order.to_vec(),
            triangle_indices: triangle_indices.to_vec(),
            positions: positions.to_vec(),
        };
        let bvh = crate::mesh_bvh::MeshBvh::deserialize(data)
            .ok_or_else(|| JsValue::from_str("failed to deserialize MeshBVH"))?;
        Ok(WebMeshBvh {
            inner: Arc::new(bvh),
        })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
fn pack_hits(hits: &[crate::mesh_bvh::BvhHit]) -> Vec<f32> {
    let mut out = Vec::with_capacity(hits.len() * 7);
    for h in hits {
        out.push(h.distance);
        out.push(h.point.x);
        out.push(h.point.y);
        out.push(h.point.z);
        out.push(h.face_index as f32);
        out.push(h.uv.x);
        out.push(h.uv.y);
    }
    out
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen]
impl WebBufferGeometry {
    #[wasm_bindgen(js_name = computeBoundsTree)]
    pub fn compute_bounds_tree(
        &mut self,
        max_depth: u32,
        max_leaf_tris: u32,
    ) -> Result<WebMeshBvh, JsValue> {
        self.compute_bounds_tree_full(
            max_depth,
            max_leaf_tris,
            crate::mesh_bvh::CENTER,
            0,
            0,
            false,
        )
    }

    #[wasm_bindgen(js_name = computeBoundsTreeFull)]
    pub fn compute_bounds_tree_full(
        &mut self,
        max_depth: u32,
        max_leaf_tris: u32,
        strategy: u32,
        offset: u32,
        count: u32,
        indirect: bool,
    ) -> Result<WebMeshBvh, JsValue> {
        let options = crate::mesh_bvh::BuildOptions {
            strategy,
            max_depth,
            max_leaf_tris,
            offset,
            count,
            indirect,
        };
        let bvh = Arc::make_mut(&mut self.inner)
            .compute_bounds_tree(options)
            .ok_or_else(|| JsValue::from_str("failed to build bounds tree"))?;
        Ok(WebMeshBvh { inner: bvh })
    }

    #[wasm_bindgen(js_name = disposeBoundsTree)]
    pub fn dispose_bounds_tree(&mut self) {
        Arc::make_mut(&mut self.inner).dispose_bounds_tree();
    }

    #[wasm_bindgen(js_name = hasBoundsTree)]
    pub fn has_bounds_tree(&self) -> bool {
        self.inner.bounds_tree.is_some()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "mesh-bvh"))]
#[wasm_bindgen(js_name = mergeGeometries)]
pub fn merge_geometries(geometries: Vec<WebBufferGeometry>) -> Result<WebBufferGeometry, JsValue> {
    if geometries.is_empty() {
        return Err(JsValue::from_str("mergeGeometries: empty input"));
    }
    let refs: Vec<crate::BufferGeometry> = geometries.iter().map(|g| (*g.inner).clone()).collect();
    let merged = crate::merge_geometries(&refs)
        .ok_or_else(|| JsValue::from_str("mergeGeometries: incompatible geometries"))?;
    Ok(WebBufferGeometry {
        inner: Arc::new(merged),
    })
}

// ---------------------------------------------------------------------------
// Browser animation export (native-codec): GIF / APNG / WebM / MP4 from RGBA frames
// ---------------------------------------------------------------------------

#[cfg(feature = "native-codec")]
fn js_rgba_frames(frames: &js_sys::Array) -> Result<Vec<Vec<u8>>, JsValue> {
    let n = frames.length() as usize;
    if n == 0 {
        return Err(JsValue::from_str("encode: need at least one frame"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..frames.length() {
        let v = frames.get(i);
        let u8a = js_sys::Uint8Array::new(&v);
        let mut buf = vec![0u8; u8a.length() as usize];
        u8a.copy_to(&mut buf);
        out.push(buf);
    }
    Ok(out)
}

#[cfg(feature = "native-codec")]
fn encode_browser_animation(
    width: u32,
    height: u32,
    fps: u32,
    codec: crate::BrowserCodec,
    transparent: bool,
    gif_colors: u16,
    frames: &js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    let collected = js_rgba_frames(frames)?;
    let opts = crate::AnimationEncodeOptions {
        width,
        height,
        fps: fps.max(1),
        codec,
        transparent,
        gif_colors: gif_colors.clamp(2, 256),
    };
    let bytes = crate::encode_animation_rgba(&opts, collected)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(js_sys::Uint8Array::from(bytes.as_slice()))
}

/// Encode RGBA frames to an animated GIF (`native-codec` feature).
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeGifRgba)]
pub fn encode_gif_rgba(
    width: u32,
    height: u32,
    fps: u32,
    colors: u16,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Gif,
        transparent,
        colors,
        &frames,
    )
}

/// Encode RGBA frames to an animated PNG (`native-codec` feature).
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeApngRgba)]
pub fn encode_apng_rgba(
    width: u32,
    height: u32,
    fps: u32,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Apng,
        transparent,
        256,
        &frames,
    )
}

/// Encode RGBA frames to a VP9 WebM (`native-codec` feature).
/// Width and height must be multiples of 8.
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeWebmRgba)]
pub fn encode_webm_rgba(
    width: u32,
    height: u32,
    fps: u32,
    transparent: bool,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Webm,
        transparent,
        256,
        &frames,
    )
}

/// Encode RGBA frames to an H.264 MP4 (`native-codec` feature).
/// Width and height must be even; transparency is not supported.
#[cfg(feature = "native-codec")]
#[wasm_bindgen(js_name = encodeMp4Rgba)]
pub fn encode_mp4_rgba(
    width: u32,
    height: u32,
    fps: u32,
    frames: js_sys::Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    encode_browser_animation(
        width,
        height,
        fps,
        crate::BrowserCodec::Mp4,
        false,
        256,
        &frames,
    )
}

// ---------------------------------------------------------------------------
// OpenSCAD front end → geometry, for the in-browser gallery. Gated on the
// `openscad` feature (which pulls in the exact CSG kernel + bvh-csg fallback).
// ---------------------------------------------------------------------------

/// A parsed OpenSCAD model as a flat, flat-shaded triangle soup: `positions` and
/// per-face `normals` (9 floats per triangle each), ready to drop straight into a
/// `BufferGeometry`. `error` is set (and the arrays empty) when parsing fails.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub struct ScadGeometry {
    positions: Vec<f32>,
    normals: Vec<f32>,
    triangles: u32,
    error: Option<String>,
}

#[cfg(feature = "openscad")]
#[wasm_bindgen]
impl ScadGeometry {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Vec<f32> {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn triangles(&self) -> u32 {
        self.triangles
    }
    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }
}

/// Register an in-memory file (text) so `surface`/`import`/`include`/`use` can
/// read it — the browser has no filesystem, so height-fields, meshes, etc. must
/// be supplied this way before calling `scad_geometry`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_register_file(name: &str, contents: &str) {
    crate::register_file(name, contents.as_bytes().to_vec());
}

/// Register an in-memory file from raw bytes (e.g. a binary STL to `import`).
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_register_file_bytes(name: &str, bytes: &[u8]) {
    crate::register_file(name, bytes.to_vec());
}

/// Drop every registered in-memory file.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_clear_files() {
    crate::clear_files();
}

/// Parse OpenSCAD source and build its solid with the exact CSG kernel (float
/// fallback behind the never-wrong gate), returning a flat-shaded triangle soup.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn scad_geometry(src: &str) -> ScadGeometry {
    let solid = match crate::parse_scad(src) {
        Ok(s) => s,
        Err(e) => {
            return ScadGeometry {
                positions: vec![],
                normals: vec![],
                triangles: 0,
                error: Some(e),
            }
        }
    };
    let (positions, normals) = flat_soup(&solid.to_geometry_exact());
    let tris = (positions.len() / 9) as u32;
    ScadGeometry {
        positions,
        normals,
        triangles: tris,
        error: None,
    }
}

/// Parse OpenSCAD source and encode its solid in a mesh format for download.
/// `format` is one of `stl` / `obj` / `off` / `3mf` / `glb` / `fcstd`; returns
/// the encoded bytes (empty on parse error or unknown format).
#[cfg(feature = "openscad")]
#[wasm_bindgen(js_name = scadExport)]
pub fn scad_export(src: &str, format: &str) -> Vec<u8> {
    let solid = match crate::parse_scad(src) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let g = solid.to_geometry_exact();
    match format {
        "stl" => crate::geometry_to_stl(&g),
        "obj" => crate::geometry_to_obj(&g).into_bytes(),
        "off" => crate::geometry_to_off(&g).into_bytes(),
        "3mf" => crate::geometry_to_3mf(&g),
        "glb" => crate::geometry_to_glb(&g),
        "fcstd" => crate::geometry_to_fcstd(&g, "Model"),
        _ => Vec::new(),
    }
}

/// A geometry as a flat-shaded triangle soup: `(positions, per-face normals)`,
/// 9 floats per triangle each. Expands the index buffer when present.
#[cfg(feature = "openscad")]
fn flat_soup(g: &crate::BufferGeometry) -> (Vec<f32>, Vec<f32>) {
    let verts: Vec<[f32; 3]> = match g.positions() {
        Some(it) => it.map(|v| [v.x, v.y, v.z]).collect(),
        None => Vec::new(),
    };
    let soup: Vec<[f32; 3]> = match &g.index {
        Some(idx) => idx.iter().map(|&i| verts[i as usize]).collect(),
        None => verts,
    };
    let (mut positions, mut normals) = (
        Vec::with_capacity(soup.len() * 3),
        Vec::with_capacity(soup.len() * 3),
    );
    for t in soup.chunks_exact(3) {
        let (a, b, c) = (t[0], t[1], t[2]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-20 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        }
        for vtx in t {
            positions.extend_from_slice(vtx);
            normals.extend_from_slice(&n);
        }
    }
    (positions, normals)
}

// ---------------------------------------------------------------------------
// Parametric 3D-printer assembly (see examples/printer_assembly.rs). Returned to
// JS as a list of flat-shaded, coloured parts — one mesh each, no cross-part CSG.
// ---------------------------------------------------------------------------

/// A parametric printer as a list of coloured parts. `positions(i)`/`normals(i)`
/// are flat-shaded soups; `color(i)` is an `[r,g,b]`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub struct PrinterScene {
    parts: Vec<(Vec<f32>, Vec<f32>, [f32; 3])>,
}

#[cfg(feature = "openscad")]
#[wasm_bindgen]
impl PrinterScene {
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.parts.len()
    }
    pub fn positions(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.0.clone()).unwrap_or_default()
    }
    pub fn normals(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.1.clone()).unwrap_or_default()
    }
    pub fn color(&self, i: usize) -> Vec<f32> {
        self.parts.get(i).map(|p| p.2.to_vec()).unwrap_or_default()
    }
}

/// Build the parametric printer (all lengths in mm). Mirrors the native
/// `examples/printer_assembly.rs`.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn printer_scene(
    bed_x: f32,
    bed_y: f32,
    z_travel: f32,
    ext: f32,
    gantry_z: f32,
    carriage: f32,
    bed_pos: f32,
) -> PrinterScene {
    let parts = crate::build_printer(bed_x, bed_y, z_travel, ext, gantry_z, carriage, bed_pos)
        .into_iter()
        .map(|(g, c)| {
            let (p, n) = flat_soup(&g);
            (p, n, c)
        })
        .collect();
    PrinterScene { parts }
}

/// Build the detailed NEMA 17 motor (see `examples/nema17.rs`) as coloured parts.
#[cfg(feature = "openscad")]
#[wasm_bindgen]
pub fn nema17_scene() -> PrinterScene {
    let parts = crate::build_nema17()
        .into_iter()
        .map(|(g, c)| {
            let (p, n) = flat_soup(&g);
            (p, n, c)
        })
        .collect();
    PrinterScene { parts }
}

// ---------------------------------------------------------------------------
// Subtitles and captions
// ---------------------------------------------------------------------------

/// A timed-text track — the JS face of [`crate::captions::CaptionTrack`].
///
/// ```js
/// const track = CaptionTrack.parse(await (await fetch('dialogue.vtt')).text());
/// const overlay = new CaptionOverlay(track);
/// overlay.setAutoScale(true);
/// // ...then each frame:
/// renderer.renderWithCaptions(scene, camera, overlay, video.currentTime);
/// ```
#[cfg(feature = "captions")]
#[wasm_bindgen]
pub struct WebCaptionTrack {
    inner: crate::captions::CaptionTrack,
}

#[cfg(feature = "captions")]
#[wasm_bindgen]
impl WebCaptionTrack {
    /// An empty track.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebCaptionTrack {
        WebCaptionTrack {
            inner: crate::captions::CaptionTrack::new(),
        }
    }

    /// Parse SubRip or WebVTT, sniffing the `WEBVTT` magic.
    pub fn parse(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Parse SubRip (`.srt`).
    #[wasm_bindgen(js_name = parseSrt)]
    pub fn parse_srt(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse_srt(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Parse WebVTT (`.vtt`) — what a `<track>` element takes.
    #[wasm_bindgen(js_name = parseVtt)]
    pub fn parse_vtt(text: &str) -> Result<WebCaptionTrack, JsValue> {
        crate::captions::CaptionTrack::parse_vtt(text)
            .map(|inner| WebCaptionTrack { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Append a cue, in seconds.
    #[wasm_bindgen(js_name = addCue)]
    pub fn add_cue(&mut self, start: f64, end: f64, text: &str) {
        self.inner.push(crate::captions::Cue::new(start, end, text));
    }

    /// BCP-47 language tag (`"en"`, `"pt-BR"`).
    #[wasm_bindgen(js_name = setLanguage)]
    pub fn set_language(&mut self, language: &str) {
        self.inner.language = language.to_string();
    }

    /// Human-readable name shown in player menus.
    #[wasm_bindgen(js_name = setLabel)]
    pub fn set_label(&mut self, label: &str) {
        self.inner.label = label.to_string();
    }

    /// Shift every cue by `seconds` (negative moves earlier).
    pub fn shift(&mut self, seconds: f64) {
        self.inner.shift(seconds);
    }

    /// Number of cues.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// End time of the last cue, in seconds.
    pub fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Text showing at `time` seconds, markup stripped. Empty when nothing is.
    #[wasm_bindgen(js_name = textAt)]
    pub fn text_at(&self, time: f64) -> String {
        self.inner.text_at(time)
    }

    /// Serialize to SubRip.
    #[wasm_bindgen(js_name = toSrt)]
    pub fn to_srt(&self) -> String {
        self.inner.to_srt()
    }

    /// Serialize to WebVTT — feed this to a `<track>` via a Blob URL.
    #[wasm_bindgen(js_name = toVtt)]
    pub fn to_vtt(&self) -> String {
        self.inner.to_vtt()
    }
}

#[cfg(feature = "captions")]
impl Default for WebCaptionTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// A frame-sized caption overlay — the JS face of
/// [`crate::captions::CaptionOverlay`].
///
/// Pass it to [`WebRenderer::render_with_captions`] to have the GPU blend it
/// over the 3D image, or read [`rgba`](Self::rgba) to composite it yourself
/// (e.g. into a 2D canvas `ImageData`).
#[cfg(feature = "captions")]
#[wasm_bindgen]
pub struct WebCaptionOverlay {
    inner: crate::captions::CaptionOverlay,
}

#[cfg(feature = "captions")]
#[wasm_bindgen]
impl WebCaptionOverlay {
    /// An overlay for `track`, using the built-in bitmap face.
    #[wasm_bindgen(constructor)]
    pub fn new(track: &WebCaptionTrack) -> WebCaptionOverlay {
        WebCaptionOverlay {
            inner: crate::captions::CaptionOverlay::new(track.inner.clone()),
        }
    }

    /// Replace the cues.
    #[wasm_bindgen(js_name = setTrack)]
    pub fn set_track(&mut self, track: &WebCaptionTrack) {
        self.inner.set_track(track.inner.clone());
    }

    /// Use a TrueType font for the text (`.ttf` bytes, `glyf` outlines).
    #[wasm_bindgen(js_name = setFont)]
    pub fn set_font(&mut self, ttf: &[u8]) -> Result<(), JsValue> {
        let font = crate::captions::CaptionFont::from_ttf_bytes(ttf)
            .map_err(|e| JsValue::from_str(&format!("font could not be parsed: {e:?}")))?;
        self.inner.painter.font = font;
        self.inner.invalidate();
        Ok(())
    }

    /// Rescale the style with the frame height, treating the current values as
    /// authored for 1080p. Off by default.
    #[wasm_bindgen(js_name = setAutoScale)]
    pub fn set_auto_scale(&mut self, enabled: bool) {
        self.inner.set_auto_scale(enabled);
    }

    /// Type size in pixels.
    #[wasm_bindgen(js_name = setFontSize)]
    pub fn set_font_size(&mut self, px: f32) {
        self.restyle(|s| s.font_size(px));
    }

    /// Text fill color, 0–255 per channel.
    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.restyle(|s| s.color([r, g, b, a]));
    }

    /// Outline color and half-width in pixels. Width `0` removes it.
    #[wasm_bindgen(js_name = setOutline)]
    pub fn set_outline(&mut self, r: u8, g: u8, b: u8, a: u8, width: f32) {
        self.restyle(|s| s.outline([r, g, b, a], width));
    }

    /// Background box color. Alpha `0` removes the box.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.restyle(|s| s.background([r, g, b, a]));
    }

    /// Drop shadow color and offset in pixels. Alpha `0` removes it.
    #[wasm_bindgen(js_name = setShadow)]
    pub fn set_shadow(&mut self, r: u8, g: u8, b: u8, a: u8, dx: f32, dy: f32) {
        self.restyle(|s| s.shadow([r, g, b, a], dx, dy));
    }

    /// Distance from the anchored frame edge, in pixels.
    #[wasm_bindgen(js_name = setMargin)]
    pub fn set_margin(&mut self, px: f32) {
        self.restyle(|s| s.margin(px));
    }

    /// Padding inside the background box, in pixels.
    #[wasm_bindgen(js_name = setPadding)]
    pub fn set_padding(&mut self, px: f32) {
        self.restyle(|s| s.padding(px));
    }

    /// Line advance as a multiple of the type size.
    #[wasm_bindgen(js_name = setLineHeight)]
    pub fn set_line_height(&mut self, factor: f32) {
        self.restyle(|s| s.line_height(factor));
    }

    /// Wrap width as a fraction (`0..=1`) of the frame width.
    #[wasm_bindgen(js_name = setMaxWidth)]
    pub fn set_max_width(&mut self, fraction: f32) {
        self.restyle(|s| s.max_width(fraction));
    }

    /// Horizontal alignment: `"left"`, `"center"`, or `"right"`.
    #[wasm_bindgen(js_name = setAlign)]
    pub fn set_align(&mut self, align: &str) {
        let align = match align {
            "left" | "start" => crate::captions::CaptionAlign::Left,
            "right" | "end" => crate::captions::CaptionAlign::Right,
            _ => crate::captions::CaptionAlign::Center,
        };
        self.restyle(|s| s.align(align));
    }

    /// Frame edge to anchor to: `"top"`, `"middle"`, or `"bottom"`.
    #[wasm_bindgen(js_name = setAnchor)]
    pub fn set_anchor(&mut self, anchor: &str) {
        let anchor = match anchor {
            "top" => crate::captions::CaptionAnchor::Top,
            "middle" | "center" => crate::captions::CaptionAnchor::Middle,
            _ => crate::captions::CaptionAnchor::Bottom,
        };
        self.restyle(|s| s.anchor(anchor));
    }

    /// Overlay resolution. [`WebRenderer::render_with_captions`] sets this for
    /// you; call it directly only when compositing by hand.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.inner.set_size(width, height);
    }

    /// Rasterize for `time` seconds if needed. Returns whether the buffer
    /// changed — the cue to skip re-uploading on unchanged frames.
    pub fn update(&mut self, time: f64) -> bool {
        self.inner.update(time)
    }

    /// Whether a cue is currently showing.
    #[wasm_bindgen(js_name = isVisible)]
    pub fn is_visible(&self) -> bool {
        self.inner.is_visible()
    }

    /// The RGBA8 overlay buffer, transparent where there is no caption.
    /// Copy it into an `ImageData` to composite in a 2D canvas.
    pub fn rgba(&self) -> Vec<u8> {
        self.inner.rgba().to_vec()
    }
}

#[cfg(feature = "captions")]
impl WebCaptionOverlay {
    /// Apply a style builder and invalidate the cached raster.
    fn restyle(
        &mut self,
        f: impl FnOnce(crate::captions::CaptionStyle) -> crate::captions::CaptionStyle,
    ) {
        // Style off the authored values, not the auto-scaled ones, so
        // repeated setter calls do not compound the rescale.
        let style = f(self.inner.style().clone());
        self.inner.set_style(style);
    }
}

// ---------------------------------------------------------------------------
// NURBS (feature = "nurbs")
// ---------------------------------------------------------------------------
//
// The three.js parity types in `curves::nurbs` were never reachable from JS at
// all — no binding, no shim export. These are the f64 kernel types, which is
// what a caller actually wants on the web side too: exact circles, analytic
// normals, and a tessellator that answers to a tolerance instead of a segment
// count.
//
// Flat `Float64Array`s rather than arrays-of-arrays throughout: one copy across
// the wasm boundary instead of an allocation per control point.

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
pub struct WebNurbsCurve {
    inner: crate::nurbs::NurbsCurve,
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
impl WebNurbsCurve {
    /// Build from Cartesian control points (flat `xyz`) and optional weights.
    #[wasm_bindgen(constructor)]
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        points: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<WebNurbsCurve, JsValue> {
        let pts: Vec<[f64; 3]> = points.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        crate::nurbs::NurbsCurve::new(degree, knots, &pts, weights.as_deref())
            .map(|inner| WebNurbsCurve { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Build from control points already in homogeneous form (flat `xyzw`) —
    /// the layout three.js's `NURBSCurve` and STEP both use.
    #[wasm_bindgen(js_name = fromHomogeneous)]
    pub fn from_homogeneous(
        degree: usize,
        knots: Vec<f64>,
        control: Vec<f64>,
    ) -> Result<WebNurbsCurve, JsValue> {
        let cw: Vec<[f64; 4]> = control
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        crate::nurbs::NurbsCurve::from_homogeneous(degree, knots, cw)
            .map(|inner| WebNurbsCurve { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    pub fn knots(&self) -> Vec<f64> {
        self.inner.knots().to_vec()
    }

    #[wasm_bindgen(js_name = controlCount)]
    pub fn control_count(&self) -> usize {
        self.inner.n_control()
    }

    /// `[u_min, u_max]`.
    pub fn domain(&self) -> Vec<f64> {
        let (a, b) = self.inner.domain();
        vec![a, b]
    }

    /// Point at parameter `u` (clamped to the domain) as `[x, y, z]`.
    pub fn point(&self, u: f64) -> Vec<f64> {
        self.inner.point(u).to_vec()
    }

    /// Point at normalized `t ∈ [0, 1]`.
    #[wasm_bindgen(js_name = pointAt)]
    pub fn point_at(&self, t: f64) -> Vec<f64> {
        self.inner.point(self.inner.param_at(t)).to_vec()
    }

    /// Derivatives 0..=`k`, flattened: `[C, C', C'', …]`.
    pub fn derivatives(&self, u: f64, k: usize) -> Vec<f64> {
        self.inner.derivatives(u, k).into_iter().flatten().collect()
    }

    /// Unit tangent, or an empty array at a cusp where none exists.
    pub fn tangent(&self, u: f64) -> Vec<f64> {
        self.inner
            .tangent(u)
            .map(|t| t.to_vec())
            .unwrap_or_default()
    }

    /// Adaptive polyline honouring `tolerance`, flattened `xyz`.
    pub fn tessellate(&self, tolerance: f64) -> Vec<f64> {
        let opts = crate::nurbs::TessellationOptions::with_tolerance(tolerance);
        crate::nurbs::tessellate_curve(&self.inner, &opts)
            .into_iter()
            .flatten()
            .collect()
    }
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
pub struct WebNurbsSurface {
    inner: crate::nurbs::NurbsSurface,
}

#[cfg(feature = "nurbs")]
#[wasm_bindgen]
impl WebNurbsSurface {
    /// Build from a Cartesian control grid (flat `xyz`, **u-major**: index
    /// `i * n_v + j`) and optional weights.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        n_u: usize,
        n_v: usize,
        points: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Result<WebNurbsSurface, JsValue> {
        let pts: Vec<[f64; 3]> = points.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        crate::nurbs::NurbsSurface::new(
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            n_u,
            n_v,
            &pts,
            weights.as_deref(),
        )
        .map(|inner| WebNurbsSurface { inner })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = degreeU)]
    pub fn degree_u(&self) -> usize {
        self.inner.degree_u()
    }

    #[wasm_bindgen(js_name = degreeV)]
    pub fn degree_v(&self) -> usize {
        self.inner.degree_v()
    }

    /// `[u_min, u_max, v_min, v_max]`.
    pub fn domain(&self) -> Vec<f64> {
        let (u0, u1) = self.inner.domain_u();
        let (v0, v1) = self.inner.domain_v();
        vec![u0, u1, v0, v1]
    }

    /// Point at `(u, v)` as `[x, y, z]`.
    pub fn point(&self, u: f64, v: f64) -> Vec<f64> {
        self.inner.point(u, v).to_vec()
    }

    /// Point at normalized `(s, t) ∈ [0, 1]²`.
    #[wasm_bindgen(js_name = pointAt)]
    pub fn point_at(&self, s: f64, t: f64) -> Vec<f64> {
        let (u, v) = self.inner.param_at(s, t);
        self.inner.point(u, v).to_vec()
    }

    /// Analytic unit normal. Empty only if the surface is degenerate over a
    /// whole neighbourhood — poles are handled and do return a normal.
    pub fn normal(&self, u: f64, v: f64) -> Vec<f64> {
        self.inner
            .normal(u, v)
            .map(|n| n.to_vec())
            .unwrap_or_default()
    }

    /// Tessellate to a `BufferGeometry` at the given chord tolerance.
    pub fn tessellate(&self, tolerance: f64) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::geometries::NurbsGeometry::with_tolerance(
                &self.inner,
                tolerance,
            )),
        }
    }

    /// Tessellate on a fixed `u_segments × v_segments` grid.
    #[wasm_bindgen(js_name = tessellateGrid)]
    pub fn tessellate_grid(&self, u_segments: usize, v_segments: usize) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(crate::geometries::NurbsGeometry::with_segments(
                &self.inner,
                u_segments,
                v_segments,
            )),
        }
    }

    /// Did the sampler actually reach `tolerance`, or did its sample ceiling
    /// bind first? A truncated grid otherwise looks identical to a converged one.
    #[wasm_bindgen(js_name = meetsTolerance)]
    pub fn meets_tolerance(&self, tolerance: f64) -> bool {
        let opts = crate::nurbs::TessellationOptions::with_tolerance(tolerance);
        let (pu, pv) = crate::nurbs::tessellate::sample_grid(&self.inner, &opts);
        crate::nurbs::tessellate::sample_grid_meets_tolerance(&self.inner, &pu, &pv, tolerance)
    }

    /// Sweep this surface's defining profile — see the free `nurbs*` builders
    /// for the constructors that produce exact quadrics.
    #[wasm_bindgen(js_name = isRational)]
    pub fn is_rational(&self) -> bool {
        self.inner.is_rational()
    }
}

/// An exact circle as a rational quadratic — not a polyline approximation.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCircle)]
pub fn nurbs_circle(cx: f64, cy: f64, cz: f64, radius: f64) -> WebNurbsCurve {
    WebNurbsCurve {
        inner: crate::nurbs::construct::circle(
            [cx, cy, cz],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            radius,
        ),
    }
}

/// An exact circular arc from `start` to `end` radians in the xy-plane.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsArc)]
pub fn nurbs_arc(cx: f64, cy: f64, cz: f64, radius: f64, start: f64, end: f64) -> WebNurbsCurve {
    WebNurbsCurve {
        inner: crate::nurbs::construct::arc(
            [cx, cy, cz],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            radius,
            start,
            end,
        ),
    }
}

/// An exact sphere. Sampling it reproduces the radius to ~1e-15 at every
/// parameter, unlike a `SphereGeometry` at any segment count.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsSphere)]
pub fn nurbs_sphere(cx: f64, cy: f64, cz: f64, radius: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::sphere([cx, cy, cz], radius),
    }
}

/// An exact cylindrical side surface about `+Z` (no caps).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCylinder)]
pub fn nurbs_cylinder(cx: f64, cy: f64, cz: f64, radius: f64, height: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::cylinder([cx, cy, cz], [0.0, 0.0, 1.0], radius, height),
    }
}

/// An exact conical side surface about `+Z` (no cap).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsCone)]
pub fn nurbs_cone(
    apex_x: f64,
    apex_y: f64,
    apex_z: f64,
    base_radius: f64,
    height: f64,
) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::cone(
            [apex_x, apex_y, apex_z],
            [0.0, 0.0, 1.0],
            base_radius,
            height,
        ),
    }
}

/// An exact torus about `+Z`.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsTorus)]
pub fn nurbs_torus(cx: f64, cy: f64, cz: f64, major: f64, minor: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::torus([cx, cy, cz], [0.0, 0.0, 1.0], major, minor),
    }
}

/// Revolve a profile curve about an axis through `(px, py, pz)` in direction
/// `(dx, dy, dz)`, sweeping `angle` radians (0 means a full turn).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsRevolve)]
#[allow(clippy::too_many_arguments)]
pub fn nurbs_revolve(
    profile: &WebNurbsCurve,
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    angle: f64,
) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::revolve(&profile.inner, [px, py, pz], [dx, dy, dz], angle),
    }
}

/// Linearly sweep a profile curve along `(dx, dy, dz)`.
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsExtrude)]
pub fn nurbs_extrude(profile: &WebNurbsCurve, dx: f64, dy: f64, dz: f64) -> WebNurbsSurface {
    WebNurbsSurface {
        inner: crate::nurbs::construct::extrude(&profile.inner, [dx, dy, dz]),
    }
}

/// The ruled surface between two curves, made compatible first (they may differ
/// in degree and knot vector).
#[cfg(feature = "nurbs")]
#[wasm_bindgen(js_name = nurbsRuled)]
pub fn nurbs_ruled(a: &WebNurbsCurve, b: &WebNurbsCurve) -> Result<WebNurbsSurface, JsValue> {
    crate::nurbs::construct::ruled(&a.inner, &b.inner)
        .map(|inner| WebNurbsSurface { inner })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── RLX bridge ──────────────────────────────────────────────────────────
//
// The browser half of `threers::rlx`. Everything the bridge does is pure data
// in and pure data out, so the bindings are plain typed arrays: JS hands over
// an `Uint8Array` of pixels or a `Float32Array` of positions and gets one
// back. `RLX=1 web/build.sh` turns them on; `RLX_GEO=1` adds the geometry set.
//
// Which devices a browser build finds is rlx's answer, not this crate's — ask
// `rlxDevices()` at startup rather than assuming.

/// The rlx backends this build found, comma-separated (e.g. `"WebGpu,Cpu"`).
/// Empty when rlx has none, in which case every call below will fail.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxDevices)]
pub fn rlx_devices() -> String {
    ::rlx::available_devices()
        .iter()
        .map(|d| format!("{d:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Convolve an RGBA8 frame with a named 3×3 kernel: `box`, `gaussian`,
/// `sharpen`, `laplacian`, `sobel-x`, `sobel-y` or `emboss`.
///
/// Alpha is carried through, and the edges are replicated rather than
/// zero-padded — see [`crate::rlx::ConvFilter`].
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxConvolve)]
pub fn rlx_convolve(
    rgba: &[u8],
    width: u32,
    height: u32,
    kernel: &str,
) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, ConvFilter, Kernel3x3};
    let kernel = match kernel {
        "box" => Kernel3x3::BOX_BLUR,
        "gaussian" => Kernel3x3::GAUSSIAN,
        "sharpen" => Kernel3x3::SHARPEN,
        "laplacian" => Kernel3x3::LAPLACIAN,
        "sobel-x" => Kernel3x3::SOBEL_X,
        "sobel-y" => Kernel3x3::SOBEL_Y,
        "emboss" => Kernel3x3::EMBOSS,
        other => return Err(JsValue::from_str(&format!("unknown kernel {other}"))),
    };
    ConvFilter::new(width, height, kernel, preferred_device())
        .apply(rgba)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Run `iterations` diffusion passes over a single-channel height grid.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxDiffuse)]
pub fn rlx_diffuse(
    field: &[f32],
    width: u32,
    height: u32,
    rate: f32,
    iterations: u32,
) -> Result<Vec<f32>, JsValue> {
    use crate::rlx::{preferred_device, Diffusion};
    Diffusion::new(width, height, rate, preferred_device())
        .run(field, iterations as usize)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Smooth a mesh, returning new positions (`[x, y, z, …]`).
///
/// A negative `mu` runs Taubin's λ|μ filter, which keeps the volume; pass 0 for
/// plain Laplacian smoothing, which does not.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxSmoothMesh)]
pub fn rlx_smooth_mesh(
    positions: &[f32],
    index: &[u32],
    iterations: u32,
    lambda: f32,
    mu: f32,
) -> Result<Vec<f32>, JsValue> {
    use crate::core::{BufferAttribute, BufferGeometry};
    use crate::rlx::{mesh, preferred_device};

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(positions.to_vec(), 3));
    geometry.set_index(index.to_vec());

    let device = preferred_device();
    let result = if mu < 0.0 {
        mesh::taubin_smooth(&mut geometry, iterations as usize, lambda, mu, device)
    } else {
        mesh::laplacian_smooth(&mut geometry, iterations as usize, lambda, device)
    };
    result.map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(geometry
        .get_attribute("position")
        .map(|a| a.array.clone())
        .unwrap_or_default())
}

/// Fit a colour grade carrying `source` onto `target`, both RGBA8 frames of the
/// same size. Returns twelve numbers: the 3×3 matrix row-major, then the bias.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxFitGrade)]
pub fn rlx_fit_grade(source: &[u8], target: &[u8]) -> Result<Vec<f32>, JsValue> {
    use crate::rlx::{preferred_device, ColorGrade, FitOptions};
    let report = ColorGrade::fit(source, target, &FitOptions::default(), preferred_device())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut out = Vec::with_capacity(12);
    for row in report.grade.matrix {
        out.extend_from_slice(&row);
    }
    out.extend_from_slice(&report.grade.bias);
    Ok(out)
}

/// Apply a grade in the twelve-number form [`rlx_fit_grade`] returns.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxApplyGrade)]
pub fn rlx_apply_grade(rgba: &[u8], grade: &[f32]) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::ColorGrade;
    if grade.len() != 12 {
        return Err(JsValue::from_str("a grade is 9 matrix values then 3 bias"));
    }
    let grade = ColorGrade {
        matrix: [
            [grade[0], grade[1], grade[2]],
            [grade[3], grade[4], grade[5]],
            [grade[6], grade[7], grade[8]],
        ],
        bias: [grade[9], grade[10], grade[11]],
    };
    Ok(grade.apply(rgba))
}

/// Cluster a frame into `count` colours, returned as sRGB `[r, g, b, …]`
/// bytes. May be shorter than asked for — see [`crate::rlx::Palette::extract`].
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxPalette)]
pub fn rlx_palette(rgba: &[u8], count: u32) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, Palette, PaletteOptions};
    let palette = Palette::extract(
        rgba,
        count as usize,
        &PaletteOptions::default(),
        preferred_device(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(palette
        .colors
        .iter()
        .flat_map(|c| {
            let e = |v: f32| {
                let s = if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            };
            [e(c.r), e(c.g), e(c.b)]
        })
        .collect())
}

/// Snap every pixel of a frame to its nearest entry of a `count`-colour
/// palette extracted from that same frame.
#[cfg(feature = "rlx")]
#[wasm_bindgen(js_name = rlxPosterize)]
pub fn rlx_posterize(rgba: &[u8], count: u32) -> Result<Vec<u8>, JsValue> {
    use crate::rlx::{preferred_device, Palette, PaletteOptions};
    let device = preferred_device();
    let palette = Palette::extract(rgba, count as usize, &PaletteOptions::default(), device)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    palette
        .posterize(rgba, device)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// A triangulated height field, ready to hand to `BufferGeometry`.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen]
pub struct GeoGeometry {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    index: Vec<u32>,
}

#[cfg(feature = "rlx-geo")]
#[wasm_bindgen]
impl GeoGeometry {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Vec<f32> {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn uvs(&self) -> Vec<f32> {
        self.uvs.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> Vec<u32> {
        self.index.clone()
    }
}

/// Exact Delaunay triangulation of `[x, y, …]` pairs, as triangle indices.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoDelaunay)]
pub fn geo_delaunay(points: &[f32]) -> Result<Vec<u32>, JsValue> {
    use crate::math::Vector2;
    let points: Vec<Vector2> = points
        .chunks_exact(2)
        .map(|p| Vector2::new(p[0], p[1]))
        .collect();
    crate::rlx::geo::delaunay_indices(&points).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scattered `[x, y, z, …]` samples triangulated in xz, with y as the height.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoHeightfield)]
pub fn geo_heightfield(points: &[f32]) -> Result<GeoGeometry, JsValue> {
    use crate::math::Vector3;
    let points: Vec<Vector3> = points
        .chunks_exact(3)
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect();
    let geometry = crate::rlx::geo::heightfield_geometry(&points)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let attribute = |name: &str| {
        geometry
            .get_attribute(name)
            .map(|a| a.array.clone())
            .unwrap_or_default()
    };
    Ok(GeoGeometry {
        positions: attribute("position"),
        normals: attribute("normal"),
        uvs: attribute("uv"),
        index: geometry.index.clone().unwrap_or_default(),
    })
}

/// Which site owns each pixel, for sites given as `[x, y, …]` in pixels.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoVoronoiLabels)]
pub fn geo_voronoi_labels(sites: &[f32], width: u32, height: u32) -> Vec<u32> {
    use crate::math::Vector2;
    let sites: Vec<Vector2> = sites
        .chunks_exact(2)
        .map(|s| Vector2::new(s[0], s[1]))
        .collect();
    crate::rlx::geo::voronoi_labels(&sites, width, height)
}

/// Distance from every pixel to the nearest cell wall — the field cell
/// textures want. Cost is pixels × sites; keep the site count in the hundreds.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoVoronoiWallDistance)]
pub fn geo_voronoi_wall_distance(sites: &[f32], width: u32, height: u32) -> Vec<f32> {
    use crate::math::Vector2;
    let sites: Vec<Vector2> = sites
        .chunks_exact(2)
        .map(|s| Vector2::new(s[0], s[1]))
        .collect();
    crate::rlx::geo::voronoi_wall_distance(&sites, width, height)
}

/// A height grid as a tangent-space normal map, RGBA8 and linear.
#[cfg(feature = "rlx-geo")]
#[wasm_bindgen(js_name = geoNormalMap)]
pub fn geo_normal_map(field: &[f32], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let texture = crate::rlx::geo::normal_map_from_height(field, width, height, strength);
    texture.data.as_ref().clone()
}

// ---------------------------------------------------------------------------
// IK + geared servo plant (planar 3R) — bodies, joints, contacts, materials.
// ---------------------------------------------------------------------------

/// One planar 3R arm with geometric IK, actuator plant, capsule solids,
/// joint reports, and world contacts (floor + obstacle).
///
/// `kind`: `"direct"` | `"qdd"` | `"high"` | `"hydraulic"` | `"tendon"`.
/// Lengths are millimetres.
#[wasm_bindgen]
pub struct WebIkServoPlant {
    world: crate::kinematics::ArmWorld,
    ik: crate::kinematics::Ik,
    q_ik: Vec<f64>,
    label: String,
}

fn ik_for_chain(chain: &crate::kinematics::SerialChain) -> crate::kinematics::Ik {
    crate::kinematics::Ik {
        limits: (-179.0, 179.0),
        max_iters: 120,
        joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
        ..Default::default()
    }
}

fn drive_for_kind(kind: &str) -> Result<crate::kinematics::Drive, JsValue> {
    use crate::kinematics::Drive;
    Ok(match kind {
        "direct" | "dd" => Drive::direct(),
        "qdd" | "qdd-15:1" => Drive::qdd(),
        "high" | "servo" | "servo-288:1" => Drive::high_ratio(),
        "hydraulic" | "hyd" => Drive::hydraulic(),
        "tendon" | "cable" | "string" => Drive::tendon(),
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown drive '{other}' (use direct|qdd|high|hydraulic|tendon)"
            )))
        }
    })
}

#[wasm_bindgen]
impl WebIkServoPlant {
    #[wasm_bindgen(constructor)]
    pub fn new(kind: &str) -> Result<WebIkServoPlant, JsValue> {
        let chain = crate::kinematics::SerialChain::planar_3r();
        let drive = drive_for_kind(kind)?;
        let label = drive.label().to_string();
        let ik = ik_for_chain(&chain);
        let mut plant = crate::kinematics::ArmPlant::from_drive(chain, drive);
        let seed = vec![-2.8, 88.9, 93.9];
        plant.seed(&seed);
        let mut world = crate::kinematics::ArmWorld::from_plant(plant);
        world.last_cmd = seed.clone();
        world.update_contacts();
        Ok(Self {
            world,
            ik,
            q_ik: seed,
            label,
        })
    }

    pub fn seed(&mut self, q_deg: &[f64]) -> Result<(), JsValue> {
        if q_deg.len() != self.world.plant.servos.len() {
            return Err(JsValue::from_str("seed length must match joint count"));
        }
        self.world.plant.seed(q_deg);
        self.q_ik = q_deg.to_vec();
        self.world.last_cmd = q_deg.to_vec();
        self.world.update_contacts();
        Ok(())
    }

    /// IK + plant step. Returns tip position error in millimetres.
    pub fn step(&mut self, tx: f64, ty: f64, tz: f64, ax: f64, ay: f64, az: f64, dt: f64) -> f64 {
        use crate::kinematics::Goal;
        let goal = Goal::PointAlong {
            at: [tx, ty, tz],
            along: [ax, ay, az],
        };
        let sol = self.ik.solve_chain(&self.q_ik, &goal, &self.world.plant.chain);
        self.q_ik = sol.joints.clone();
        self.world.last_cmd = sol.joints.clone();
        self.world.plant.step(&sol.joints, dt.max(1e-4));
        self.world.update_contacts();
        let q_act = self.world.plant.q_deg();
        let (p, _) = self.world.plant.chain.tool(&q_act);
        let dx = p[0] - tx;
        let dy = p[1] - ty;
        let dz = p[2] - tz;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[wasm_bindgen(getter, js_name = jointCount)]
    pub fn joint_count(&self) -> usize {
        self.world.plant.servos.len()
    }

    #[wasm_bindgen(getter, js_name = bodyCount)]
    pub fn body_count(&self) -> usize {
        self.world.plant.chain.n()
    }

    #[wasm_bindgen(getter, js_name = contactCount)]
    pub fn contact_count(&self) -> usize {
        self.world.contacts.len()
    }

    #[wasm_bindgen(js_name = qAct)]
    pub fn q_act(&self) -> Vec<f64> {
        self.world.plant.q_deg()
    }

    #[wasm_bindgen(js_name = qCmd)]
    pub fn q_cmd(&self) -> Vec<f64> {
        self.world.last_cmd.clone()
    }

    #[wasm_bindgen(js_name = skeletonAct)]
    pub fn skeleton_act(&self) -> Vec<f64> {
        flat_skeleton(&self.world.plant.chain, &self.world.plant.q_deg())
    }

    #[wasm_bindgen(js_name = skeletonCmd)]
    pub fn skeleton_cmd(&self) -> Vec<f64> {
        flat_skeleton(&self.world.plant.chain, &self.world.last_cmd)
    }

    #[wasm_bindgen(js_name = hingesAct)]
    pub fn hinges_act(&self) -> Vec<f64> {
        flat_hinges(&self.world.plant.chain, &self.world.plant.q_deg())
    }

    #[wasm_bindgen(js_name = tipAct)]
    pub fn tip_act(&self) -> Vec<f64> {
        let (p, _) = self.world.plant.chain.tool(&self.world.plant.q_deg());
        p.to_vec()
    }

    #[wasm_bindgen(js_name = tipCmd)]
    pub fn tip_cmd(&self) -> Vec<f64> {
        let (p, _) = self.world.plant.chain.tool(&self.world.last_cmd);
        p.to_vec()
    }

    #[wasm_bindgen(js_name = meanJointErr)]
    pub fn mean_joint_err(&self) -> f64 {
        let n = self.world.plant.servos.len().max(1) as f64;
        self.world
            .plant
            .servos
            .iter()
            .map(|s| s.tracking_error_deg())
            .sum::<f64>()
            / n
    }

    #[wasm_bindgen(js_name = reflectedInertia)]
    pub fn reflected_inertia(&self) -> f64 {
        self.world.plant.servos[0].drive.reflected_inertia()
    }

    /// Gear reduction N, or `1` for hydraulic / tendon.
    pub fn ratio(&self) -> f64 {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Gear(g) => g.ratio,
            _ => 1.0,
        }
    }

    #[wasm_bindgen(js_name = driveKind)]
    pub fn drive_kind(&self) -> String {
        self.world.plant.servos[0].drive.label().to_string()
    }

    /// Working fluid / cable / grease name.
    #[wasm_bindgen(js_name = materialName)]
    pub fn material_name(&self) -> String {
        self.world.plant.servos[0].drive.material_name().to_string()
    }

    #[wasm_bindgen(js_name = tempC)]
    pub fn temp_c(&self) -> f64 {
        self.world.plant.temp_c()
    }

    /// Operating temperature in °C (oil, grease, cable).
    #[wasm_bindgen(js_name = setTemp)]
    pub fn set_temp(&mut self, temp_c: f64) {
        self.world.plant.set_temp(temp_c.clamp(-20.0, 90.0));
    }

    /// Swap hydraulic fluid: `iso32|iso46|iso68|water-glycol|silicone`.
    #[wasm_bindgen(js_name = setFluid)]
    pub fn set_fluid(&mut self, name: &str) -> Result<(), JsValue> {
        use crate::kinematics::{Drive, HydraulicFluid};
        let fluid = HydraulicFluid::from_name(name).ok_or_else(|| {
            JsValue::from_str("unknown fluid (iso32|iso46|iso68|water-glycol|silicone)")
        })?;
        let temp = self.world.plant.temp_c();
        for s in &mut self.world.plant.servos {
            if matches!(s.drive, Drive::Hydraulic(_)) {
                let mut h = crate::kinematics::HydraulicDrive::with_fluid(fluid);
                h.env.temp_c = temp;
                s.drive = Drive::Hydraulic(h);
                s.tau_max = s.drive.torque_limit();
            }
        }
        Ok(())
    }

    /// Swap tendon material: `steel|uhmwpe|nylon|aramid`.
    #[wasm_bindgen(js_name = setTendon)]
    pub fn set_tendon(&mut self, name: &str) -> Result<(), JsValue> {
        use crate::kinematics::{Drive, TendonMaterial};
        let mat = TendonMaterial::from_name(name).ok_or_else(|| {
            JsValue::from_str("unknown tendon (steel|uhmwpe|nylon|aramid)")
        })?;
        let temp = self.world.plant.temp_c();
        for s in &mut self.world.plant.servos {
            if matches!(s.drive, Drive::Tendon(_)) {
                let mut t = crate::kinematics::TendonDrive::with_material(mat);
                t.env.temp_c = temp;
                s.drive = Drive::Tendon(t);
                s.tau_max = s.drive.torque_limit();
            }
        }
        Ok(())
    }

    /// Live agonist/antagonist tensions for joint 0, N: `[T+, T−, pretension]`.
    #[wasm_bindgen(js_name = tensions0)]
    pub fn tensions0(&self) -> Vec<f64> {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Tendon(t) => {
                vec![t.tension_plus, t.tension_minus, t.live_pretension()]
            }
            _ => vec![],
        }
    }

    /// Fluid kinematic viscosity at current temp, cSt (hydraulic only).
    #[wasm_bindgen(js_name = fluidNuCst)]
    pub fn fluid_nu_cst(&self) -> f64 {
        match &self.world.plant.servos[0].drive {
            crate::kinematics::Drive::Hydraulic(h) => h.fluid.nu_cst(h.env.temp_c),
            _ => 0.0,
        }
    }

    /// Mechanical design label (e.g. "harmonic + planetary 288:1").
    #[wasm_bindgen(js_name = designLabel)]
    pub fn design_label(&self) -> String {
        self.world.design_label()
    }

    /// Assembled mechanism parts in world mm.
    /// Stride 16: `[joint, role, ox,oy,oz, ax,ay,az, radius, length, r,g,b, metal, rough, shape]`.
    /// `role`: motor=0 … encoder=14, housing=15.
    /// `shape`: capsule=0, segment=1, disk=2.
    #[wasm_bindgen(js_name = designPartsFlat)]
    pub fn design_parts_flat(&self) -> Vec<f64> {
        crate::kinematics::parts_flat(&self.world.design_parts())
    }

    #[wasm_bindgen(js_name = designPartCount)]
    pub fn design_part_count(&self) -> usize {
        self.world.design_parts().len()
    }

    /// Flat link solids: per body
    /// `[ox,oy,oz, dx,dy,dz, radius, mass, volume, mat_id, rgb_r,rgb_g,rgb_b, metal, rough]`.
    /// `mat_id`: 0 aluminum, 1 steel, 2 plastic, 3 rubber.
    #[wasm_bindgen(js_name = bodiesFlat)]
    pub fn bodies_flat(&self) -> Vec<f64> {
        let bodies = self.world.bodies();
        let mut out = Vec::with_capacity(bodies.len() * 15);
        for b in bodies {
            let mat_id = match b.material {
                crate::kinematics::LinkMaterial::Aluminum => 0.0,
                crate::kinematics::LinkMaterial::Steel => 1.0,
                crate::kinematics::LinkMaterial::Plastic => 2.0,
                crate::kinematics::LinkMaterial::Rubber => 3.0,
            };
            let rgb = b.material.rgb();
            out.extend_from_slice(&b.origin);
            out.extend_from_slice(&b.distal);
            out.push(b.radius);
            out.push(b.mass);
            out.push(b.volume);
            out.push(mat_id);
            out.push(rgb[0] as f64);
            out.push(rgb[1] as f64);
            out.push(rgb[2] as f64);
            out.push(b.material.metalness() as f64);
            out.push(b.material.roughness() as f64);
        }
        out
    }

    /// Body names, newline-separated.
    #[wasm_bindgen(js_name = bodyNames)]
    pub fn body_names(&self) -> String {
        self.world
            .bodies()
            .into_iter()
            .map(|b| b.name)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Flat joints: per hinge
    /// `[ox,oy,oz, ax,ay,az, angle, cmd, omega, lo, hi, torque, engaged]`.
    #[wasm_bindgen(js_name = jointsFlat)]
    pub fn joints_flat(&self) -> Vec<f64> {
        let joints = self.world.joints();
        let mut out = Vec::with_capacity(joints.len() * 13);
        for j in joints {
            out.extend_from_slice(&j.origin);
            out.extend_from_slice(&j.axis);
            out.push(j.angle_deg);
            out.push(j.cmd_deg);
            out.push(j.omega);
            out.push(j.limits.0);
            out.push(j.limits.1);
            out.push(j.torque);
            out.push(if j.engaged { 1.0 } else { 0.0 });
        }
        out
    }

    /// Joint names, newline-separated.
    #[wasm_bindgen(js_name = jointNames)]
    pub fn joint_names(&self) -> String {
        self.world
            .joints()
            .into_iter()
            .map(|j| j.name)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Flat contacts: `[px,py,pz, nx,ny,nz, depth, …]` plus parallel name pairs
    /// via [`Self::contact_pairs`].
    #[wasm_bindgen(js_name = contactsFlat)]
    pub fn contacts_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.world.contacts.len() * 7);
        for c in &self.world.contacts {
            out.extend_from_slice(&c.point);
            out.extend_from_slice(&c.normal);
            out.push(c.depth);
        }
        out
    }

    /// `"a|b"` pairs, newline-separated, matching [`Self::contacts_flat`] order.
    #[wasm_bindgen(js_name = contactPairs)]
    pub fn contact_pairs(&self) -> String {
        self.world
            .contacts
            .iter()
            .map(|c| format!("{}|{}", c.a, c.b))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Obstacle AABB `[xmin,ymin,zmin, xmax,ymax,zmax]` and floor z as 7th value.
    #[wasm_bindgen(js_name = worldBounds)]
    pub fn world_bounds(&self) -> Vec<f64> {
        let o = &self.world.obstacles;
        let (mn, mx) = match (o.box_min, o.box_max) {
            (Some(a), Some(b)) => (a, b),
            _ => ([0.0; 3], [0.0; 3]),
        };
        vec![mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], o.floor_z]
    }
}

fn flat_skeleton(chain: &crate::kinematics::SerialChain, q: &[f64]) -> Vec<f64> {
    let sk = chain.skeleton(q);
    let mut out = Vec::with_capacity(sk.len() * 3);
    for p in sk {
        out.extend_from_slice(&p);
    }
    out
}

fn flat_hinges(chain: &crate::kinematics::SerialChain, q: &[f64]) -> Vec<f64> {
    let poses = chain.poses(q);
    let mut out = Vec::with_capacity(poses.len() * 6);
    for p in poses {
        out.extend_from_slice(&p.origin);
        out.extend_from_slice(&p.axis);
    }
    out
}
