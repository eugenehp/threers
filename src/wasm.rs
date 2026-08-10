#![allow(dead_code)]
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
    let padded = ((unpadded + ALIGN - 1) / ALIGN) * ALIGN;
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
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
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
    device.poll(wgpu::Maintain::Wait);
    rx.await
        .map_err(|_| "readback channel dropped".to_string())?
        .map_err(|e| format!("buffer map failed: {e:?}"))?;
    let mapped = buffer_slice.get_mapped_range();
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
        })
        .await
        .ok_or_else(|| JsValue::from_str("no adapter"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None,
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

#[wasm_bindgen]
impl WebRenderer {
    /// Async factory. Pass an HTMLCanvasElement and the renderer attaches to it.
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        console_error_panic_hook::set_once();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

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
            })
            .await
            .ok_or_else(|| JsValue::from_str("no adapter"))?;

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
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
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
        let padded = ((unpadded + ALIGN - 1) / ALIGN) * ALIGN;
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
                wgpu::ImageCopyTexture {
                    texture: &target.color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &buffer,
                    layout: wgpu::ImageDataLayout {
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
            device.poll(wgpu::Maintain::Wait);
            rx.await
                .map_err(|_| JsValue::from_str("readback channel dropped"))?
                .map_err(|e| JsValue::from_str(&format!("buffer map failed: {e:?}")))?;
            // Copy the mapped (possibly padded) range into a tight RGBA byte vec.
            let mapped = buffer_slice.get_mapped_range();
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
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
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
        frame.present();
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
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
                }
            }
        };
        let output_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .blit_rgba8(&data, width, height, &output_view, self.config.format);
        frame.present();
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
            Ok(frame) => {
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
                frame.present();
            }
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
            }
        }
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
        let mut m = crate::MatcapMaterial::default();
        m.color = color.inner;
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
            for i in 0..16 {
                m.texture_matrix[i] = elements[i];
            }
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
    #[wasm_bindgen(js_name = reseedFromCamera)]
    pub fn reseed_from_camera(&mut self, camera: &WebCamera) {
        if let CameraInner::Perspective(c) = &camera.inner {
            self.inner.reseed_from_camera(c);
        }
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
    #[wasm_bindgen(js_name = renderToString)]
    pub fn render_to_string(&self, scene: &mut WebScene, camera: &WebCamera) -> String {
        match &camera.inner {
            CameraInner::Perspective(c) => self.inner.render_to_string(&mut scene.inner, c),
            CameraInner::Orthographic(c) => self.inner.render_to_string(&mut scene.inner, c),
        }
    }
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
// Browser animation export (native-codec): GIF / APNG / WebM from RGBA frames
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
            return ScadGeometry { positions: vec![], normals: vec![], triangles: 0, error: Some(e) }
        }
    };
    let (positions, normals) = flat_soup(&solid.to_geometry_exact());
    let tris = (positions.len() / 9) as u32;
    ScadGeometry { positions, normals, triangles: tris, error: None }
}

/// Parse OpenSCAD source and encode its solid in a mesh format for download.
/// `format` is one of `stl` / `obj` / `off` / `3mf` / `glb`; returns the encoded
/// bytes (empty on parse error or unknown format).
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
    let (mut positions, mut normals) = (Vec::with_capacity(soup.len() * 3), Vec::with_capacity(soup.len() * 3));
    for t in soup.chunks_exact(3) {
        let (a, b, c) = (t[0], t[1], t[2]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let mut n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
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
