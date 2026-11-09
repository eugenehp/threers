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

// Bindings that live in their own files under `src/wasm/`: shapes, paths,
// extrusion and text; the other geometry generators; the exporters; and
// helpers, loaders and noise. Child modules, so they see these types' fields.
// TransformControls, FlyControls, MapControls' screen-space panning.
mod controls;
// The shim's Curve classes (three.js r165 curves, arc length, Frenet frames).
mod curves;
// AnimationClip, KeyframeTracks, Interpolants, AnimationMixer and its actions,
// AnimationObjectGroup: three.js r165's animation system, run in Rust.
mod animation;
mod exporters;
mod extras;
mod generators;
// GLTFLoader: the three.js r165 parser in `crate::loaders::gltf`, building the
// scene graph's nodes directly; the shim wraps them.
mod gltf;
// Scene data behind the shim's Box3 / Sphere / Frustum: geometry and vertex bounds.
mod math;
mod mesh_builder;
mod mesh_state;
// The shim's scene-graph mirror: nodes, structure, raycasting, DragControls.
mod scene_graph;
// Particles, snow and sand (`threers::particles`).
mod particles;
pub use particles::WebParticleSystem;
mod shapes;
// TimeSmear: the offline echo filter over a captured clip.
mod time_smear;
pub use time_smear::WebTimeSmear;
// Post-processing: the composer and its passes.
mod postfx;
pub use postfx::{WebEffectComposer, WebPass};
// Split out of this file by area; each re-exported so its paths stay `crate::wasm::*`.
pub use math::*;
mod objects;
pub use objects::*;
mod textures;
pub use animation::*;
pub use controls::*;
pub use curves::*;
pub use textures::*;
mod helpers;
pub use helpers::*;
mod loaders;
pub use loaders::*;
#[cfg(feature = "audio")]
mod audio;
#[cfg(feature = "audio")]
pub use audio::*;
pub use extras::*;
pub use postfx::*;
mod alt_renderers;
pub use alt_renderers::*;
mod stats;
pub use stats::*;
mod raycaster;
pub use raycaster::*;
mod mesh_bvh;
pub use mesh_bvh::*;

/// A lazy renderer's pipelines, to be built from a page's event loop. Hold it
/// past the renderer's own borrow: awaiting `run()` does not lock the renderer.
#[wasm_bindgen]
pub struct WebWarmup(crate::renderer::PipelineWarmup);

#[wasm_bindgen]
impl WebWarmup {
    /// How many pipelines the job builds.
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> u32 {
        self.0.len() as u32
    }

    /// How many are built so far.
    #[wasm_bindgen(getter)]
    pub fn built(&self) -> u32 {
        self.0.progress().0 as u32
    }

    /// Build the pipelines one after another, yielding to the browser between
    /// them. Resolves with the count built; rejects with the validation
    /// errors, one per line, if any pipeline was rejected.
    pub async fn run(&self) -> Result<u32, JsValue> {
        let mut built = 0;
        while self.0.step_async().await {
            built += 1;
            yield_to_event_loop().await;
        }
        let errors = self.0.errors();
        if errors.is_empty() {
            Ok(built)
        } else {
            let lines: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            Err(JsValue::from_str(&lines.join("\n")))
        }
    }
}

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
    /// True on wgpu's WebGL2 backend. See [`WebRenderer::gl_back_buffer`].
    is_gl: bool,
    /// Set once a page uses viewport / scissor / auto-clear off / `clear()` on
    /// the canvas; from then on canvas draws on WebGL2 go through `gl_back`.
    multi_view: bool,
    /// Canvas-sized offscreen colour buffer that WebGL2 canvas draws accumulate
    /// in, copied to the canvas after each draw.
    gl_back: Option<(u32, Arc<crate::renderer::RenderTarget>)>,
}

// Thread-local registry mapping render-target IDs to live Arcs. We use this
// to look up the target from setRenderTarget() because wasm-bindgen's
// `Option<&WebRenderTarget>` can't safely hold an Arc across the JS boundary.
thread_local! {
    static ACTIVE_TARGETS: std::cell::RefCell<std::collections::HashMap<u32, std::sync::Arc<crate::renderer::RenderTarget>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
    static ACTIVE_CUBE_TARGETS: std::cell::RefCell<std::collections::HashMap<u32, std::sync::Arc<crate::renderer::CubeRenderTarget>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
    // Targets reallocated by `setSize` since the last render: the renderer's
    // sample views of them are stale until registered again.
    static RESIZED_TARGETS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// GPU readback of a render-target region into a tight byte vec.
/// Hand control back to the browser for one turn of the event loop.
async fn yield_to_event_loop() {
    if let Some(window) = web_sys::window() {
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

/// Wait for a buffer mapping to complete without blocking the event loop.
///
/// `PollType::wait_indefinitely()` blocks the calling thread, and in a browser
/// that is the same thread which would deliver the mapping callback — so the
/// wait can never end. It went unnoticed because WebGPU's wgpu backend makes
/// `poll` a no-op and resolves the mapping through a JS promise; on the WebGL2
/// backend, which goes through wgpu-core, polling really does wait and a
/// read-back hung the page.
///
/// So: poll without blocking, yield, and let the browser deliver the callback.
/// The cap is a liveness bound rather than a deadline — it is set well past any
/// plausible read-back so that a loaded machine cannot turn a slow frame into a
/// spurious failure, while a mapping that genuinely never completes still ends.
async fn await_buffer_mapping(
    device: &wgpu::Device,
    mut rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
) -> Result<(), String> {
    for _ in 0..20_000 {
        let _ = device.poll(wgpu::PollType::Poll);
        if let Ok(Some(res)) = rx.try_recv() {
            return res.map_err(|e| format!("buffer map failed: {e:?}"));
        }
        yield_to_event_loop().await;
    }
    Err("buffer mapping never completed".to_string())
}

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
    await_buffer_mapping(device, rx).await?;
    let mapped = buffer_slice
        .get_mapped_range()
        .expect("buffer range is mapped");
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
    /// The instance every renderer in this tab shares.
    ///
    /// A `wgpu::Device` belongs to the instance that created it, so pairing a
    /// cached device with a surface from a *different* instance is invalid — it
    /// shows up as `Cannot get non-existent resource DeviceId`, which says
    /// nothing about the cause. The device is cached below and the backend
    /// discovery makes a fresh instance per attempt, so the instance that won
    /// has to be kept alongside it.
    static SHARED_BROWSER_INSTANCE: std::cell::RefCell<Option<wgpu::Instance>> =
        const { std::cell::RefCell::new(None) };
    /// Which backend the adapter turned out to be, for [`WebRenderer::backend`].
    ///
    /// Worth surfacing rather than inferring: on WebGL2 several effects switch
    /// themselves off, and without a way to ask, the only symptom is that bloom
    /// and ambient occlusion quietly do nothing.
    static SHARED_BROWSER_BACKEND: std::cell::RefCell<Option<wgpu::Backend>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
/// Open a GPU context on `canvas`, preferring WebGPU and falling back to WebGL2.
///
/// Two things here are forced by the browser rather than chosen.
///
/// **One instance per backend.** On wasm, `wgpu::Instance::new` commits to a
/// single context implementation: ask for `BROWSER_WEBGPU` and you get the
/// WebGPU one, which contains no GL whatever the Cargo features say. So
/// `BROWSER_WEBGPU | GL` does not mean "try both" — it means WebGPU, and on a
/// browser without it the adapter request fails claiming
/// `gl support not compiled in`, which is not what has gone wrong.
///
/// **The canvas is not touched until a backend is settled.** A canvas hands out
/// exactly one kind of context for its whole life: once `getContext("webgpu")`
/// has been called, `getContext("webgl2")` on that same canvas returns null
/// forever. Creating a surface to test WebGPU therefore destroys the fallback it
/// was testing for — the GL attempt then fails with `canvas already in use`.
/// WebGPU can enumerate adapters with no surface at all, so it is asked first
/// and the surface is created only once an adapter has answered.
pub async fn open_browser_gpu(
    canvas: &web_sys::HtmlCanvasElement,
) -> Result<(wgpu::Instance, wgpu::Surface<'static>, wgpu::Adapter), JsValue> {
    // A second canvas reuses the instance that already won — but only on
    // WebGPU, where an adapter is not tied to a canvas.
    //
    // On WebGL2 it cannot: a WebGL2 context *belongs* to its canvas, so an
    // instance that has already made a surface for one canvas cannot serve
    // another. Reusing it gets as far as `Surface::configure` and fails there
    // with `Surface does not support the adapter's queue family`, which does not
    // mention canvases at all. So on that backend every canvas gets its own
    // instance, surface and device, and the sharing below is skipped.
    let shareable = !matches!(
        SHARED_BROWSER_BACKEND.with(|b| *b.borrow()),
        Some(wgpu::Backend::Gl)
    );
    if let Some(instance) =
        SHARED_BROWSER_INSTANCE.with(|i| if shareable { i.borrow().clone() } else { None })
    {
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("surface: {e:?}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("no adapter: {e}")))?;
        return Ok((instance, surface, adapter));
    }

    let already = SHARED_BROWSER_BACKEND.with(|b| *b.borrow());
    let mut why = String::new();

    if !matches!(already, Some(wgpu::Backend::Gl)) {
        // wgpu 30 dropped `Default` here; the display handle is only consulted
        // by GLES/Wayland, not Vulkan, Metal or DX12.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(d);
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(adapter) => {
                let surface = instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                    .map_err(|e| JsValue::from_str(&format!("webgpu surface: {e:?}")))?;
                SHARED_BROWSER_BACKEND.with(|b| *b.borrow_mut() = Some(adapter.get_info().backend));
                SHARED_BROWSER_INSTANCE.with(|i| *i.borrow_mut() = Some(instance.clone()));
                return Ok((instance, surface, adapter));
            }
            Err(e) => why = format!("webgpu: {e}"),
        }
    }

    if cfg!(feature = "webgl") && !matches!(already, Some(wgpu::Backend::BrowserWebGpu)) {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::GL;
        let instance = wgpu::Instance::new(d);
        // GL is the other way round: a WebGL2 context *is* the canvas, so the
        // surface has to exist before there is an adapter to ask about.
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("{why}; webgl2 surface: {e:?}")))?;
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(adapter) => {
                // Deliberately not cached: see the note above about WebGL2
                // contexts belonging to their canvas.
                SHARED_BROWSER_BACKEND.with(|b| *b.borrow_mut() = Some(adapter.get_info().backend));
                return Ok((instance, surface, adapter));
            }
            Err(e) => why = format!("{why}; webgl2: {e}"),
        }
    }

    Err(JsValue::from_str(&format!("no adapter: {why}")))
}

#[cfg(target_arch = "wasm32")]
pub async fn acquire_browser_gpu(
    adapter: &wgpu::Adapter,
) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), JsValue> {
    // Shared across canvases on WebGPU, where `requestAdapter` + `requestDevice`
    // dominate cold start. Not on WebGL2: the device belongs to an instance that
    // belongs to a canvas, so each canvas pays for its own.
    let shareable = adapter.get_info().backend != wgpu::Backend::Gl;
    if shareable && let Some(pair) = SHARED_BROWSER_GPU.with(|g| g.borrow().clone()) {
        return Ok(pair);
    }

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
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("device: {e:?}")))?;
    crate::gpu::report_uncaptured_errors(&device, "threers");

    let pair = (Arc::new(device), Arc::new(queue));
    if shareable {
        SHARED_BROWSER_GPU.with(|g| *g.borrow_mut() = Some(pair.clone()));
    }
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
        // `readRenderTargetPixels` here returns what the target stores — the
        // encoded picture — unless told the target is linear: the three.js
        // layer sets each target's `texture.colorSpace` before reading.
        rt.set_srgb_color_space(true);
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

    /// three.js `renderTarget.setSize(width, height)`: reallocate at the new
    /// size, keeping the id — so every texture made from this target, and every
    /// material sampling it, carries on with the resized one — and its colour
    /// space. The contents start cleared, as three.js's do.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.inner = std::sync::Arc::new(self.inner.resized(width, height));
        (self.width, self.height) = (width, height);
        ACTIVE_TARGETS.with(|m| m.borrow_mut().insert(self.id, self.inner.clone()));
        RESIZED_TARGETS.with(|r| r.borrow_mut().push(self.id));
    }

    /// three.js `renderTarget.texture.colorSpace`: `true` for `SRGBColorSpace`,
    /// whose read-back is sRGB-encoded (this API's default, the stored
    /// bytes); `false` for three.js's default `NoColorSpace`, which reads back
    /// linear. See `RenderTarget::srgb_color_space`.
    #[wasm_bindgen(js_name = setSrgbColorSpace)]
    pub fn set_srgb_color_space(&self, srgb: bool) {
        self.inner.set_srgb_color_space(srgb);
    }

    /// three.js `renderTarget.samples`: above 1, drawn multisampled and
    /// resolved, colour and depth (see `RenderTarget::samples`). A half-float
    /// target stays single-sampled.
    #[wasm_bindgen(js_name = setSamples)]
    pub fn set_samples(&self, samples: u32) {
        self.inner.set_samples(samples);
    }

    #[wasm_bindgen(getter)]
    pub fn samples(&self) -> u32 {
        self.inner.samples()
    }
}

/// Prefer alpha compositing modes that preserve a transparent canvas.
pub fn pick_surface_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> wgpu::CompositeAlphaMode {
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

/// Forward `log` records to the browser console.
///
/// Without this, wgpu is silent in a browser: it reports a rejected pipeline
/// or a failed surface configuration through `log::error!`, and with no
/// logger installed those records go nowhere. The symptom is a canvas that
/// stays blank while the frame loop runs at sixty — which is indistinguishable
/// from a slow start until you can see the error.
///
/// Hand-rolled rather than `console_log`, which is a dependency for twenty
/// lines, and `web-sys`'s `console` is already here for the panic hook.
struct ConsoleLogger;

impl log::Log for ConsoleLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "[{}] {}: {}",
            record.level(),
            record.target(),
            record.args()
        );
        let msg = wasm_bindgen::JsValue::from_str(&line);
        match record.level() {
            log::Level::Error => web_sys::console::error_1(&msg),
            _ => web_sys::console::warn_1(&msg),
        }
    }

    fn flush(&self) {}
}

/// Install [`ConsoleLogger`], once. Safe to call from every entry point.
pub fn install_console_logger() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `set_logger` only fails if one is already installed, which is a
        // host that wanted its own — leave theirs alone.
        if log::set_logger(&ConsoleLogger).is_ok() {
            log::set_max_level(log::LevelFilter::Warn);
        }
    });
}

impl WebRenderer {
    /// The renderer's device and queue, for other bindings that compute on
    /// the same GPU (`WebSoftBody::useGpu`).
    pub(crate) fn device_queue(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        (self.renderer.device_arc(), self.renderer.queue_arc())
    }
}

#[wasm_bindgen]
impl WebRenderer {
    /// Async factory. Pass an HTMLCanvasElement and the renderer attaches to it.
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        Self::create(canvas, false).await
    }

    /// [`new`](Self::new) with the pipelines built on first use instead of in
    /// the constructor, which returns much sooner. Call
    /// [`warm_up`](Self::warm_up) and await its `run()` while a loading state
    /// shows, or the first frames stall as the browser compiles.
    #[wasm_bindgen(js_name = newLazy)]
    pub async fn new_lazy(canvas: web_sys::HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        Self::create(canvas, true).await
    }

    /// The pipelines a lazy renderer has not built yet, as a job. Awaiting
    /// its `run()` builds them and reports validation errors (a browser
    /// cannot block on them). Empty for an eager renderer.
    #[wasm_bindgen(js_name = warmUp)]
    pub fn warm_up(&self) -> WebWarmup {
        WebWarmup(self.renderer.warm_up())
    }

    async fn create(
        canvas: web_sys::HtmlCanvasElement,
        lazy: bool,
    ) -> Result<WebRenderer, JsValue> {
        console_error_panic_hook::set_once();
        install_console_logger();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        // The instance is not kept: `Surface` and `Device` hold their own handles
        // to the context, and the WebGPU instance that wins is cached in
        // `SHARED_BROWSER_INSTANCE` for the next canvas. Bound rather than
        // discarded so it outlives the device creation below.
        let (_instance, surface, adapter) = open_browser_gpu(&canvas).await?;
        let (device, queue) = acquire_browser_gpu(&adapter).await?;

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

        let renderer = if lazy {
            crate::Renderer::new_lazy(device.clone(), queue.clone(), surface_format, width, height)
        } else {
            crate::Renderer::new(device.clone(), queue.clone(), surface_format, width, height)
        };
        // Continue construction below — see existing return statement.

        Ok(WebRenderer {
            renderer,
            surface,
            config,
            device,
            width,
            height,
            current_target_id: None,
            is_gl: adapter.get_info().backend == wgpu::Backend::Gl,
            multi_view: false,
            gl_back: None,
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
        self.gl_back = None;
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
        // three.js's constants: 0 NoToneMapping, 1 Linear, 4 ACESFilmic,
        // 6 AgX, 7 Neutral. 2 (Reinhard) and 3 (Cineon) have no curve here and
        // fall through to linear, as they always have.
        let mapping = match mode {
            0 => crate::ToneMapping::None,
            4 => crate::ToneMapping::AcesFilmic,
            6 => crate::ToneMapping::AgX,
            7 => crate::ToneMapping::Neutral,
            _ => crate::ToneMapping::Linear,
        };
        self.renderer.set_tone_mapping(mapping, exposure);
    }

    /// Reversed depth for the view — see
    /// [`Renderer::set_reversed_z`](crate::Renderer::set_reversed_z). For a
    /// globe or anything spanning kilometres with a near plane of metres; off
    /// by default.
    #[wasm_bindgen(js_name = setReversedZ)]
    pub fn set_reversed_z(&mut self, on: bool) {
        self.renderer.set_reversed_z(on);
    }

    /// Geometric specular antialiasing on Standard and Physical materials, in
    /// place of three.js's `geometryRoughness` — see
    /// [`Renderer::set_specular_antialiasing`](crate::Renderer::set_specular_antialiasing).
    /// Off by default, as three.js renders.
    #[wasm_bindgen(js_name = setSpecularAntialiasing)]
    pub fn set_specular_antialiasing(&mut self, on: bool) {
        self.renderer.set_specular_antialiasing(on);
    }

    /// threers's reflective transparency on Standard and Physical — see
    /// [`Renderer::set_reflective_transparency`](crate::Renderer::set_reflective_transparency).
    /// Off by default, as three.js renders.
    #[wasm_bindgen(js_name = setReflectiveTransparency)]
    pub fn set_reflective_transparency(&mut self, on: bool) {
        self.renderer.set_reflective_transparency(on);
    }

    #[wasm_bindgen(js_name = setMsaa)]
    pub fn set_msaa(&mut self, samples: u32) {
        self.renderer.set_msaa(samples);
    }

    /// three.js `renderer.shadowMap.type`: 0 `BasicShadowMap`, 1
    /// `PCFShadowMap` (the default), 2 `PCFSoftShadowMap`, 3 `VSMShadowMap` —
    /// drawn here as percentage-closer soft shadows, whose edge softens with
    /// distance from the caster as a variance map's does.
    #[wasm_bindgen(js_name = setShadowType)]
    pub fn set_shadow_type(&mut self, kind: u32) {
        self.renderer.set_shadow_type(shadow_type_of(kind));
    }

    /// The filter `setShadowType` set, as three.js's number.
    #[wasm_bindgen(getter, js_name = shadowType)]
    pub fn shadow_type(&self) -> u32 {
        self.renderer.shadow_type().code()
    }

    /// How many layers overlapping refraction glass composites in
    /// (`Renderer::set_refraction_layers`).
    #[wasm_bindgen(js_name = setRefractionLayers)]
    pub fn set_refraction_layers(&mut self, layers: u32) {
        self.renderer.set_refraction_layers(layers as usize);
    }

    /// Whether the last render was multisampled — MSAA asked for is not
    /// always delivered (see `Renderer::last_frame_multisampled`).
    #[wasm_bindgen(getter, js_name = lastFrameMultisampled)]
    pub fn last_frame_multisampled(&self) -> bool {
        self.renderer.last_frame_multisampled()
    }

    /// three.js `renderer.shadowMap.autoUpdate`: on (default), every render
    /// redraws the shadow maps; off, only a render after
    /// `shadowMapNeedsUpdate = true` does. See `renderer::ShadowMap`.
    #[wasm_bindgen(getter, js_name = shadowMapAutoUpdate)]
    pub fn shadow_map_auto_update(&self) -> bool {
        self.renderer.shadow_map().auto_update
    }
    #[wasm_bindgen(setter, js_name = shadowMapAutoUpdate)]
    pub fn set_shadow_map_auto_update(&mut self, on: bool) {
        self.renderer.shadow_map_mut().auto_update = on;
    }

    /// three.js `renderer.shadowMap.needsUpdate`, consumed by the next render
    /// that draws the maps.
    #[wasm_bindgen(getter, js_name = shadowMapNeedsUpdate)]
    pub fn shadow_map_needs_update(&self) -> bool {
        self.renderer.shadow_map().needs_update
    }
    #[wasm_bindgen(setter, js_name = shadowMapNeedsUpdate)]
    pub fn set_shadow_map_needs_update(&mut self, on: bool) {
        self.renderer.shadow_map_mut().needs_update = on;
    }

    /// three.js `setViewport(x, y, width, height)`: bottom-left origin, in
    /// pixels scaled by `setPixelRatio`. See `Renderer::set_viewport`.
    #[wasm_bindgen(js_name = setViewport)]
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.multi_view = true;
        self.renderer.set_viewport(x, y, width, height);
    }

    /// Back to the whole canvas (or render target).
    #[wasm_bindgen(js_name = resetViewport)]
    pub fn reset_viewport(&mut self) {
        self.renderer.reset_viewport();
    }

    /// three.js `setScissor(x, y, width, height)`; same coordinates as
    /// `setViewport`, used while `setScissorTest(true)`.
    #[wasm_bindgen(js_name = setScissor)]
    pub fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.renderer.set_scissor(x, y, width, height);
    }

    /// three.js `setScissorTest`. On, each render clears and draws only inside
    /// the scissor — several renders per animation frame then share the canvas
    /// (on WebGPU the canvas texture is the same for the whole frame; on
    /// WebGL2 the draws accumulate in an offscreen buffer — see
    /// `gl_back_buffer`).
    #[wasm_bindgen(js_name = setScissorTest)]
    pub fn set_scissor_test(&mut self, enabled: bool) {
        self.multi_view |= enabled;
        self.renderer.set_scissor_test(enabled);
    }

    /// Scale for viewport/scissor rectangles only (three.js `setPixelRatio`);
    /// `setSize` still takes device pixels.
    #[wasm_bindgen(js_name = setPixelRatio)]
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.renderer.set_pixel_ratio(ratio);
    }

    /// three.js `autoClear`: false keeps the colour already on the canvas.
    #[wasm_bindgen(js_name = setAutoClear)]
    pub fn set_auto_clear(&mut self, enabled: bool) {
        self.multi_view |= !enabled;
        self.renderer.set_auto_clear(enabled);
    }

    /// three.js `renderer.clear()` for colour — the same colour handling as
    /// `scene.setBackground` — only inside the scissor while the scissor test
    /// is on. Clears the active render target if one is set.
    pub fn clear(&mut self, color: &WebColor, alpha: f32) {
        let color = color.inner;
        let alpha = alpha.clamp(0.0, 1.0);
        if let Some(id) = self.current_target_id
            && let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned())
        {
            self.renderer.clear_render_target(&target, color, alpha);
            return;
        }
        // A clear on the canvas is a frame being composed by hand.
        self.multi_view = true;
        if let Some((id, back)) = self.gl_back_buffer() {
            self.renderer.clear_render_target(&back, color, alpha);
            self.present_gl_back_buffer(id);
            return;
        }
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                self.renderer.clear(&view, false, color, alpha);
                self.renderer.queue_arc().present(frame);
            }
            _ => self.surface.configure(&self.device, &self.config),
        }
    }

    /// The offscreen buffer canvas draws go to on WebGL2 once a page composes
    /// frames from several draws (viewport, scissor test, auto-clear off, or
    /// `clear()` on the canvas); `None` otherwise, and always on WebGPU.
    ///
    /// Why: wgpu's GL backend hands out the canvas as a *new* texture on
    /// every acquire, tracked as uninitialised, so a render pass that loads
    /// it — a pane drawn after another pane — gets it zero-filled first and
    /// the earlier pane is lost. (Browser WebGPU has no such tracking: the
    /// canvas texture is the same object for the whole frame.) Drawing into
    /// a persistent target and copying it to the canvas after each draw keeps
    /// every pane. The copy is a fullscreen pass, sRGB in and out, so it is
    /// exact. With MSAA on, these draws go through the multisampled path
    /// (`render_back`), which resolves a pane through its own pane-sized
    /// target — so the panes keep their anti-aliasing here too.
    fn gl_back_buffer(&mut self) -> Option<(u32, Arc<crate::renderer::RenderTarget>)> {
        if !(self.is_gl && self.multi_view) {
            return None;
        }
        let stale = match &self.gl_back {
            None => true,
            Some((_, rt)) => rt.width != self.width || rt.height != self.height,
        };
        if stale {
            if let Some((old, _)) = self.gl_back.take() {
                ACTIVE_TARGETS.with(|m| m.borrow_mut().remove(&old));
            }
            let rt = Arc::new(crate::renderer::RenderTarget::new(
                &self.device,
                self.width,
                self.height,
                self.config.format,
            ));
            // It stands in for the canvas, so it is tone-mapped like one.
            rt.set_presents(true);
            let id = NEXT_RT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ACTIVE_TARGETS.with(|m| m.borrow_mut().insert(id, rt.clone()));
            self.gl_back = Some((id, rt));
        }
        let (id, rt) = self.gl_back.as_ref()?;
        self.renderer.register_render_target(*id, rt);
        Some((*id, rt.clone()))
    }

    /// Copy the WebGL2 back buffer over the whole canvas and present it.
    fn present_gl_back_buffer(&mut self, back_id: u32) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let camera = postfx_camera_from(0.1, 100.0, 0.5, 16, &identity, &identity);
        // Kind 0, the byte copy: the buffer and the canvas are both sRGB on
        // WebGL2, and the pass re-encodes what its raw view decodes and
        // decodes what the canvas encodes (`postfx_flag`), so the bytes
        // arrive as they were.
        self.renderer.apply_postfx_by_id(
            back_id,
            0,
            &view,
            0,
            0,
            0,
            0.0,
            [0.0; 4],
            [0.0; 4],
            self.width,
            self.height,
            false,
            camera,
        );
        self.renderer.queue_arc().present(frame);
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
        // On WebGPU a target takes the canvas's format, `bgra8unorm` in
        // Chrome; three.js's `readRenderTargetPixels` returns RGBA whatever
        // the target, so BGRA is swizzled on the way out.
        let bgra = matches!(
            target.color_texture.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        wasm_bindgen_futures::future_to_promise(async move {
            // Map the buffer for reading. wgpu uses a callback; bridge with a oneshot.
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = futures_channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            // Pump the device until the map completes, yielding between polls:
            // blocking here would starve the callback. See
            // `await_buffer_mapping`.
            await_buffer_mapping(&device, rx)
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            // Copy the mapped (possibly padded) range into a tight RGBA byte vec.
            let mapped = buffer_slice
                .get_mapped_range()
                .expect("buffer range is mapped");
            let mut out = Vec::with_capacity((w * h * bytes_per_pixel) as usize);
            for row in 0..h {
                let start = (row * padded) as usize;
                let end = start + (unpadded as usize);
                out.extend_from_slice(&mapped[start..end]);
            }
            drop(mapped);
            buffer.unmap();
            if bgra {
                for px in out.as_chunks_mut::<4>().0 {
                    px.swap(0, 2);
                }
            }
            // What three.js returns for the target's colour space.
            target.bytes_as_color_space(&mut out);
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
        if depth_rt_id != 0
            && let Some(depth) = ACTIVE_TARGETS.with(|m| m.borrow().get(&depth_rt_id).cloned())
        {
            self.renderer.register_render_target(depth_rt_id, &depth);
        }
        if normal_rt_id != 0
            && let Some(normal) = ACTIVE_TARGETS.with(|m| m.borrow().get(&normal_rt_id).cloned())
        {
            self.renderer.register_render_target(normal_rt_id, &normal);
        }
        let camera = postfx_camera_from(
            cam_near,
            cam_far,
            kernel_radius,
            kernel_size,
            &proj,
            &inv_proj,
        );
        // On WebGL2 every canvas write goes through the persistent buffer
        // (`gl_back_buffer`): wgpu's GL backend hands the canvas out as a
        // fresh, uninitialised texture each time, so a second pass onto it —
        // UnrealBloomPass's bloom added over its copy, OutlinePass's edges
        // over the scene — would start from a cleared surface and lose the
        // first. Drawing into the buffer and presenting it keeps both.
        if self.is_gl {
            self.multi_view = true;
            if let Some((back_id, back)) = self.gl_back_buffer() {
                self.renderer.apply_postfx_by_id(
                    input_rt_id,
                    back_id,
                    &back.color_view,
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
                self.present_gl_back_buffer(back_id);
                return;
            }
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
        if depth_rt_id != 0
            && let Some(depth) = ACTIVE_TARGETS.with(|m| m.borrow().get(&depth_rt_id).cloned())
        {
            self.renderer.register_render_target(depth_rt_id, &depth);
        }
        if normal_rt_id != 0
            && let Some(normal) = ACTIVE_TARGETS.with(|m| m.borrow().get(&normal_rt_id).cloned())
        {
            self.renderer.register_render_target(normal_rt_id, &normal);
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
        if let Some(id) = self.current_target_id
            && let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned())
        {
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

    /// Which backend this renderer got: `"webgpu"`, `"gl"`, or the backend's
    /// own name on other targets.
    ///
    /// Worth asking. On `"gl"` — WebGL2 — there is no compute stage, so bloom,
    /// ambient occlusion and the packed post chain switch themselves off. They
    /// do it quietly, because the alternative is refusing to draw at all, and
    /// without this there is no way to tell that from a scene that simply has
    /// no bloom in it.
    pub fn backend(&self) -> String {
        SHARED_BROWSER_BACKEND
            .with(|b| *b.borrow())
            .map(|b| match b {
                wgpu::Backend::BrowserWebGpu => "webgpu".to_string(),
                wgpu::Backend::Gl => "gl".to_string(),
                other => format!("{other:?}").to_lowercase(),
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Whether this device can run compute passes.
    ///
    /// False on WebGL2, and the reason the compute-backed effects are
    /// unavailable there.
    #[wasm_bindgen(js_name = supportsCompute)]
    pub fn supports_compute(&self) -> bool {
        crate::renderer::caps::supports_compute(&self.device)
    }

    pub fn render(&mut self, scene: &mut WebScene, camera: &WebCamera) {
        // Targets resized since the last frame: their sample views are new.
        for id in RESIZED_TARGETS.with(|r| std::mem::take(&mut *r.borrow_mut())) {
            if let Some(t) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned()) {
                self.renderer.register_render_target(id, &t);
            }
        }
        // If the scene references a cube render target as its environment map,
        // make sure the renderer's view cache has its cube view registered.
        if let Some(env_rt_id) = scene.inner.environment_cube_rt
            && let Some(target) = ACTIVE_CUBE_TARGETS.with(|m| m.borrow().get(&env_rt_id).cloned())
        {
            self.renderer
                .register_cube_render_target(env_rt_id, &target);
        }
        // If a render target is set, render into it (no canvas presentation).
        if let Some(id) = self.current_target_id
            && let Some(target) = ACTIVE_TARGETS.with(|m| m.borrow().get(&id).cloned())
        {
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
        // A WebGL2 canvas is sRGB, so its hardware blend mixes linear light
        // where three.js's mixes the encoded bytes. The renderer reaches the
        // bytes through a texture — it draws the blended surfaces into a
        // `Unorm` copy of one — so on WebGL2 a frame with anything blended
        // goes through the back buffer, as a multi-pass frame does. The copy
        // to the canvas is exact; from then on every frame takes that path.
        if self.is_gl && !self.multi_view && scene_blends(&scene.inner) {
            self.multi_view = true;
        }
        if let Some((id, back)) = self.gl_back_buffer() {
            match &camera.inner {
                CameraInner::Perspective(c) => {
                    render_back(&mut self.renderer, &mut scene.inner, c, &back)
                }
                CameraInner::Orthographic(c) => {
                    render_back(&mut self.renderer, &mut scene.inner, c, &back)
                }
            }
            self.present_gl_back_buffer(id);
            return;
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
    pub(crate) fn gpu_device_queue(
        &self,
    ) -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
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
    /// Mirrored three.js `uuid`s, for animation track names that use them
    /// (`animation::WebScene::find_animation_node`).
    node_uuids: std::collections::HashMap<ObjectId, String>,
    /// Fur in this scene, animated by `stepFur`.
    fur_animator: crate::fur::FurAnimator,
    /// Particles, snow and sand in this scene, run by `stepParticles`.
    particle_animator: crate::particles::ParticleAnimator,
}

#[wasm_bindgen]
impl WebScene {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebScene {
        WebScene {
            inner: crate::Scene::new(),
            node_uuids: std::collections::HashMap::new(),
            fur_animator: crate::fur::FurAnimator::new(),
            particle_animator: crate::particles::ParticleAnimator::new(),
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
        if let Some(obj) = self.inner.get_mut(handle.id)
            && let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind
        {
            mesh.material = mat.inner.clone();
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
        if let Some(obj) = self.inner.get_mut(handle.id)
            && let crate::core::ObjectKind::LineSegments(ls) = &mut obj.kind
        {
            ls.geometry = geom.inner.clone();
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

    /// Grow shell fur (`threers::fur`) on a geometry; returns its handle.
    /// The strand grid is fitted to the geometry's UVs.
    #[wasm_bindgen(js_name = addFur)]
    pub fn add_fur(&mut self, geom: &WebGeometry, fur: &WebFurMaterial) -> WebObjectHandle {
        let obj = crate::fur::fur_object((*geom.inner).clone(), &fur.inner);
        let id = self.inner.add(obj);
        self.fur_animator.track(id, &fur.inner);
        WebObjectHandle { id }
    }

    /// Grow strand hair (`threers::fur::strands`) on a geometry: `count`
    /// strands of `segments` pieces, ribbons `width` wide at the root, shaded
    /// and animated (by `stepFur`) as `fur` says. Returns its handle.
    /// `shadowStrands` (default 0.25) is the share of strands that cast
    /// shadows, 0 for none; the object casts them from shadow-casting lights.
    #[wasm_bindgen(js_name = addHair)]
    pub fn add_hair(
        &mut self,
        geom: &WebGeometry,
        fur: &WebFurMaterial,
        count: u32,
        segments: u32,
        width: f32,
        shadow_strands: Option<f32>,
    ) -> WebObjectHandle {
        let options = crate::fur::strands::StrandOptions {
            count: count as usize,
            segments: segments as usize,
            width,
            shadow_strands: shadow_strands
                .unwrap_or(crate::fur::strands::StrandOptions::default().shadow_strands),
        };
        let obj = crate::fur::strands::hair_object(&geom.inner, &fur.inner, &options);
        let id = self.inner.add(obj);
        self.fur_animator.track(id, &fur.inner);
        WebObjectHandle { id }
    }

    /// Grow strand hair on the skinned mesh `skinned` (a handle from
    /// `addSkinnedMesh` or `createSkinnedMesh`), as its child, carried
    /// through its animation by `stepFur`: each strand on its triangle,
    /// trailing its root on a spring. `undefined` if `skinned` is not a
    /// skinned mesh.
    /// `shadowStrands` as for `addHair`.
    #[wasm_bindgen(js_name = growHairOnSkinned)]
    pub fn grow_hair_on_skinned(
        &mut self,
        skinned: &WebObjectHandle,
        fur: &WebFurMaterial,
        count: u32,
        segments: u32,
        width: f32,
        shadow_strands: Option<f32>,
    ) -> Option<WebObjectHandle> {
        let options = crate::fur::strands::StrandOptions {
            count: count as usize,
            segments: segments as usize,
            width,
            shadow_strands: shadow_strands
                .unwrap_or(crate::fur::strands::StrandOptions::default().shadow_strands),
        };
        self.fur_animator
            .grow_hair_on_skinned(&mut self.inner, skinned.id, &fur.inner, &options)
            .map(|id| WebObjectHandle { id })
    }

    /// Advance every fur object in the scene — wind, flutter and inertia — to
    /// `time` seconds, `dt` after the last step. Once a frame, before
    /// rendering. (The three.js shim's renderer calls it for you.)
    #[wasm_bindgen(js_name = stepFur)]
    pub fn step_fur(&mut self, time: f32, dt: f32) {
        self.fur_animator.update(&mut self.inner, time, dt);
    }

    /// A hand in a fur or hair object's coat (`handle`): within `radius` of
    /// (`x`, `y`, `z`, world), strands are pushed by (`px`, `py`, `pz`) — a
    /// pull, per unit of their length, along the hand's motion — and parted
    /// away from it by `part`. Call it as the hand moves; let go and it
    /// fades, the strands springing back (`stepFur` runs that).
    #[wasm_bindgen(js_name = strokeFur)]
    #[allow(clippy::too_many_arguments)]
    pub fn stroke_fur(
        &mut self,
        handle: &WebObjectHandle,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        px: f32,
        py: f32,
        pz: f32,
        part: f32,
    ) {
        self.fur_animator.stroke(
            handle.id,
            crate::Vector3::new(x, y, z),
            radius,
            crate::Vector3::new(px, py, pz),
            part,
        );
    }

    /// Advance a fur object's wind to `seconds`.
    #[wasm_bindgen(js_name = setFurTime)]
    pub fn set_fur_time(&mut self, handle: &WebObjectHandle, seconds: f32) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            crate::fur::set_fur_time(obj, seconds);
        }
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

    /// three.js `scene.environmentIntensity` (default 1).
    #[wasm_bindgen(js_name = setEnvironmentIntensity)]
    pub fn set_environment_intensity(&mut self, v: f32) {
        self.inner.environment_intensity = v;
    }

    /// three.js `scene.environmentRotation`, an Euler in `order`.
    #[wasm_bindgen(js_name = setEnvironmentRotation)]
    pub fn set_environment_rotation(&mut self, x: f32, y: f32, z: f32, order: &str) {
        self.inner.environment_rotation = crate::math::Euler {
            x,
            y,
            z,
            order: crate::math::EulerOrder::parse(order).unwrap_or_default(),
        };
    }

    /// three.js `scene.environment = null`: no environment at all, cube map
    /// or cube render target.
    #[wasm_bindgen(js_name = clearEnvironment)]
    pub fn clear_environment(&mut self) {
        self.inner.environment = None;
        self.inner.environment_cube_rt = None;
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

    /// One world-space clipping plane (`Object3D::clip_plane`): fragments
    /// where `n·p + constant < 0` are discarded. The shim resolves three.js's
    /// `material.clippingPlanes` / `renderer.clippingPlanes` to this each
    /// frame. `on = false` removes it.
    #[wasm_bindgen(js_name = setObjectClipPlane)]
    pub fn set_object_clip_plane(
        &mut self,
        handle: &WebObjectHandle,
        nx: f32,
        ny: f32,
        nz: f32,
        constant: f32,
        on: bool,
    ) {
        if let Some(obj) = self.inner.get_mut(handle.id) {
            obj.clip_plane = on
                .then(|| crate::math::Plane::new(crate::math::Vector3::new(nx, ny, nz), constant));
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

#[wasm_bindgen]
impl WebObjectHandle {
    /// A number naming this object, stable while it lives — what
    /// `WebRaycaster.intersectObject(s)` take and report as `objectId`.
    /// `WebScene.handleOf` turns it back into a handle.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> f64 {
        object_id_to_f64(self.id)
    }
}

/// An `ObjectId` as a JS number: slotmap's `(version << 32) | index`, exact
/// below 2^53 (a slot reused two million times).
fn object_id_to_f64(id: ObjectId) -> f64 {
    use slotmap::Key;
    id.data().as_ffi() as f64
}

fn object_id_from_f64(id: f64) -> Option<ObjectId> {
    if !(id.is_finite() && id >= 0.0 && id.fract() == 0.0 && id < 9_007_199_254_740_992.0) {
        return None;
    }
    Some(ObjectId::from(slotmap::KeyData::from_ffi(id as u64)))
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
    pub fn set_view(&mut self, px: f32, py: f32, pz: f32, tx: f32, ty: f32, tz: f32, fov_deg: f32) {
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

    /// A quad from four corners in winding order, flattened as
    /// `[x0, y0, z0, …, x3, y3, z3]`; subdivided as a bilinear patch.
    #[wasm_bindgen(js_name = quad)]
    pub fn quad(
        corners: &[f32],
        width_segments: usize,
        height_segments: usize,
    ) -> Result<WebGeometry, JsValue> {
        if corners.len() != 12 {
            return Err(JsValue::from_str(&format!(
                "quad: expected 12 corner coordinates, got {}",
                corners.len()
            )));
        }
        let c =
            |i: usize| crate::Vector3::new(corners[i * 3], corners[i * 3 + 1], corners[i * 3 + 2]);
        Ok(WebGeometry {
            inner: Arc::new(crate::QuadGeometry::with_segments(
                [c(0), c(1), c(2), c(3)],
                width_segments,
                height_segments,
            )),
        })
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
            crate::KirigamiPreset::Planar
                .evaluate(nx, ny, t)
                .to_geometry()
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
        let preset =
            crate::KirigamiPreset::from_variant(variant).unwrap_or(crate::KirigamiPreset::Planar);
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

    /// three.js `toneMapped`: whether the renderer's tone mapping applies to
    /// this material (normal and depth materials never take it).
    #[wasm_bindgen(js_name = setToneMapped)]
    pub fn set_tone_mapped(&mut self, on: bool) {
        Arc::make_mut(&mut self.inner).set_tone_mapped(on);
    }

    /// three.js `depthTest`, `depthWrite`, `depthFunc` (`THREE.NeverDepth`
    /// … `THREE.NotEqualDepth`; anything else keeps the current one) and
    /// `colorWrite`, on any material.
    #[wasm_bindgen(js_name = setRenderState)]
    pub fn set_render_state(
        &mut self,
        depth_test: bool,
        depth_write: bool,
        depth_func: u32,
        color_write: bool,
    ) {
        let s = Arc::make_mut(&mut self.inner).render_state_mut();
        s.depth_test = depth_test;
        s.depth_write = Some(depth_write);
        if let Some(f) = crate::materials::DepthFunc::from_three(depth_func) {
            s.depth_func = f;
        }
        s.color_write = color_write;
    }

    /// three.js `blending` on any material: `THREE.NormalBlending` (1) or
    /// `THREE.AdditiveBlending` (2); anything else keeps the current mode.
    #[wasm_bindgen(js_name = setBlending)]
    pub fn set_blending(&mut self, mode: u32) {
        if let Some(b) = crate::materials::Blending::from_three(mode) {
            Arc::make_mut(&mut self.inner).render_state_mut().blending = b;
        }
    }

    /// three.js `polygonOffset` / `polygonOffsetFactor` /
    /// `polygonOffsetUnits`, on every mesh material that has them.
    #[wasm_bindgen(js_name = setPolygonOffset)]
    pub fn set_polygon_offset(&mut self, enabled: bool, factor: f32, units: f32) {
        let offset = enabled.then_some(crate::materials::PolygonOffset::new(factor, units));
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Basic(m) => m.polygon_offset = offset,
            crate::Material::Lambert(m) => m.polygon_offset = offset,
            crate::Material::Phong(m) => m.polygon_offset = offset,
            crate::Material::Standard(m) => m.polygon_offset = offset,
            crate::Material::Physical(m) => m.polygon_offset = offset,
            crate::Material::Toon(m) => m.polygon_offset = offset,
            crate::Material::Normal(m) => m.polygon_offset = offset,
            crate::Material::Matcap(m) => m.polygon_offset = offset,
            crate::Material::Depth(m) => m.polygon_offset = offset,
            _ => {}
        }
    }

    #[wasm_bindgen(js_name = setAlphaTest)]
    pub fn set_alpha_test(&mut self, value: f32) {
        Arc::make_mut(&mut self.inner).set_alpha_test(value);
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
            ..Default::default()
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
            crate::Material::Lambert(m) => {
                m.displacement_scale = scale;
                m.displacement_bias = bias;
            }
            crate::Material::Phong(m) => {
                m.displacement_scale = scale;
                m.displacement_bias = bias;
            }
            crate::Material::Toon(m) => {
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
            crate::Material::Lambert(m) => m.normal_scale = v,
            crate::Material::Phong(m) => m.normal_scale = v,
            crate::Material::Toon(m) => m.normal_scale = v,
            _ => {}
        }
    }

    /// three.js `material.aoMap`: ambient occlusion from the red channel,
    /// occluding the indirect light. Standard, Physical, Lambert, Phong, Toon.
    #[wasm_bindgen(js_name = setAoMap)]
    pub fn set_ao_map(&mut self, tex: &WebTexture) {
        self.set_ao_map_arc(tex.inner.clone());
    }

    /// `setAoMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setAoMapData)]
    pub fn set_ao_map_data(&mut self, tex: &WebDataTexture) {
        self.set_ao_map_arc(tex.inner.clone());
    }

    /// three.js `material.alphaMap`: its green channel scales the opacity.
    /// Basic, Lambert, Phong, Toon, Standard, Physical, Matcap and Sprite.
    #[wasm_bindgen(js_name = setAlphaMap)]
    pub fn set_alpha_map(&mut self, tex: &WebTexture) {
        self.set_slot("alphaMap", Some(tex.inner.clone()));
    }

    /// `setAlphaMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setAlphaMapData)]
    pub fn set_alpha_map_data(&mut self, tex: &WebDataTexture) {
        self.set_slot("alphaMap", Some(tex.inner.clone()));
    }

    /// three.js `material.metalnessMap`: its blue channel scales `metalness`.
    /// Standard and Physical.
    #[wasm_bindgen(js_name = setMetalnessMap)]
    pub fn set_metalness_map(&mut self, tex: &WebTexture) {
        self.set_slot("metalnessMap", Some(tex.inner.clone()));
    }

    /// `setMetalnessMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setMetalnessMapData)]
    pub fn set_metalness_map_data(&mut self, tex: &WebDataTexture) {
        self.set_slot("metalnessMap", Some(tex.inner.clone()));
    }

    /// three.js `material.lightMap`: baked irradiance added to the indirect
    /// light. Basic, Lambert, Phong, Toon, Standard and Physical.
    #[wasm_bindgen(js_name = setLightMap)]
    pub fn set_light_map(&mut self, tex: &WebTexture) {
        self.set_slot("lightMap", Some(tex.inner.clone()));
    }

    /// `setLightMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setLightMapData)]
    pub fn set_light_map_data(&mut self, tex: &WebDataTexture) {
        self.set_slot("lightMap", Some(tex.inner.clone()));
    }

    /// three.js `material.lightMapIntensity` (default 1).
    #[wasm_bindgen(js_name = setLightMapIntensity)]
    pub fn set_light_map_intensity(&mut self, v: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Basic(m) => m.light_map_intensity = v,
            crate::Material::Lambert(m) => m.light_map_intensity = v,
            crate::Material::Phong(m) => m.light_map_intensity = v,
            crate::Material::Toon(m) => m.light_map_intensity = v,
            crate::Material::Standard(m) => m.light_map_intensity = v,
            crate::Material::Physical(m) => m.light_map_intensity = v,
            _ => {}
        }
    }

    /// three.js `material.bumpMap`: a height field perturbing the normal.
    /// Lambert, Phong, Toon, Standard and Physical.
    #[wasm_bindgen(js_name = setBumpMap)]
    pub fn set_bump_map(&mut self, tex: &WebTexture) {
        self.set_slot("bumpMap", Some(tex.inner.clone()));
    }

    /// `setBumpMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setBumpMapData)]
    pub fn set_bump_map_data(&mut self, tex: &WebDataTexture) {
        self.set_slot("bumpMap", Some(tex.inner.clone()));
    }

    /// three.js `material.bumpScale` (default 1).
    #[wasm_bindgen(js_name = setBumpScale)]
    pub fn set_bump_scale(&mut self, v: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Lambert(m) => m.bump_scale = v,
            crate::Material::Phong(m) => m.bump_scale = v,
            crate::Material::Toon(m) => m.bump_scale = v,
            crate::Material::Standard(m) => m.bump_scale = v,
            crate::Material::Physical(m) => m.bump_scale = v,
            _ => {}
        }
    }

    /// three.js `material.specularMap`: its red channel scales the specular
    /// term. Basic, Lambert and Phong.
    #[wasm_bindgen(js_name = setSpecularMap)]
    pub fn set_specular_map(&mut self, tex: &WebTexture) {
        self.set_slot("specularMap", Some(tex.inner.clone()));
    }

    /// `setSpecularMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setSpecularMapData)]
    pub fn set_specular_map_data(&mut self, tex: &WebDataTexture) {
        self.set_slot("specularMap", Some(tex.inner.clone()));
    }

    /// `material[name] = texture` for any map slot three.js names — the
    /// Physical material's `specularIntensityMap`, `clearcoatNormalMap`,
    /// `sheenColorMap` and the rest included (see
    /// [`Material::map_slot_mut`](crate::Material::map_slot_mut)). Whether
    /// this material has that slot.
    #[wasm_bindgen(js_name = setNamedMap)]
    pub fn set_named_map(&mut self, name: &str, tex: &WebTexture) -> bool {
        self.set_slot(name, Some(tex.inner.clone()))
    }

    /// `setNamedMap` for the raw `Uint8Array` / `DataTexture` path.
    #[wasm_bindgen(js_name = setNamedMapData)]
    pub fn set_named_map_data(&mut self, name: &str, tex: &WebDataTexture) -> bool {
        self.set_slot(name, Some(tex.inner.clone()))
    }

    /// three.js `clearcoatNormalScale`.
    #[wasm_bindgen(js_name = setClearcoatNormalScale)]
    pub fn set_clearcoat_normal_scale(&mut self, x: f32, y: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.clearcoat_normal_scale = crate::math::Vector2::new(x, y);
        }
    }

    /// `material[name] = null`: empty the map slot three.js calls `name`
    /// (`"map"`, `"alphaMap"`, `"aoMap"`, `"envMap"`, …). Whether this
    /// material had one.
    #[wasm_bindgen(js_name = clearMap)]
    pub fn clear_map(&mut self, name: &str) -> bool {
        if name == "envMap" {
            if let Some(slot) = self.pbr_env_slot() {
                *slot = None;
                return true;
            }
            return self.env_mut(|m| *m.env_map = None);
        }
        self.set_slot(name, None)
    }

    /// three.js `material.envMap`. On Basic, Lambert and Phong a cube map
    /// reflected (or refracted, per the cube's `mapping`) and combined with
    /// the shaded colour; on Standard and Physical the material's own
    /// image-based light, in place of `scene.environment`. Whether this
    /// material takes one.
    #[wasm_bindgen(js_name = setEnvMap)]
    pub fn set_env_map(&mut self, cube: &WebCubeTexture) -> bool {
        let c = cube.inner.clone();
        if let Some(slot) = self.pbr_env_slot() {
            *slot = Some(c);
            return true;
        }
        self.env_mut(|m| *m.env_map = Some(c.clone()))
    }

    /// three.js `material.envMapIntensity` on Standard and Physical
    /// (default 1).
    #[wasm_bindgen(js_name = setEnvMapIntensity)]
    pub fn set_env_map_intensity(&mut self, v: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.env_map_intensity = v,
            crate::Material::Physical(m) => m.env_map_intensity = v,
            _ => {}
        }
    }

    /// three.js `material.combine`: 0 `MultiplyOperation`, 1 `MixOperation`,
    /// 2 `AddOperation`.
    #[wasm_bindgen(js_name = setCombine)]
    pub fn set_combine(&mut self, combine: u32) {
        let c = match combine {
            1 => crate::materials::Combine::Mix,
            2 => crate::materials::Combine::Add,
            _ => crate::materials::Combine::Multiply,
        };
        self.env_mut(|m| *m.combine = c);
    }

    /// three.js `material.reflectivity` (default 1).
    #[wasm_bindgen(js_name = setReflectivity)]
    pub fn set_reflectivity(&mut self, v: f32) {
        self.env_mut(|m| *m.reflectivity = v);
    }

    /// three.js `material.refractionRatio` (default 0.98).
    #[wasm_bindgen(js_name = setRefractionRatio)]
    pub fn set_refraction_ratio(&mut self, v: f32) {
        self.env_mut(|m| *m.refraction_ratio = v);
    }

    /// three.js `material.envMapRotation`, an Euler in `order` (`"XYZ"`, …).
    #[wasm_bindgen(js_name = setEnvMapRotation)]
    pub fn set_env_map_rotation(&mut self, x: f32, y: f32, z: f32, order: &str) {
        let e = crate::math::Euler {
            x,
            y,
            z,
            order: crate::math::EulerOrder::parse(order).unwrap_or_default(),
        };
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.env_map_rotation = e,
            crate::Material::Physical(m) => m.env_map_rotation = e,
            _ => {
                self.env_mut(|m| *m.env_map_rotation = e);
            }
        }
    }

    /// three.js `material.aoMapIntensity` (default 1).
    #[wasm_bindgen(js_name = setAoMapIntensity)]
    pub fn set_ao_map_intensity(&mut self, intensity: f32) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.ao_intensity = intensity,
            crate::Material::Physical(m) => m.ao_intensity = intensity,
            crate::Material::Lambert(m) => m.ao_intensity = intensity,
            crate::Material::Phong(m) => m.ao_intensity = intensity,
            crate::Material::Toon(m) => m.ao_intensity = intensity,
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

    /// three.js `iridescence` / `iridescenceIOR`, and the film thickness in
    /// nanometres — the top of `iridescenceThicknessRange`.
    #[wasm_bindgen(js_name = setIridescence)]
    pub fn set_iridescence(&mut self, strength: f32, ior: f32, thickness_nm: f32) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.iridescence = strength;
            m.iridescence_ior = ior;
            m.iridescence_thickness_range[1] = thickness_nm;
        }
    }

    /// Every scalar of three.js's `MeshPhysicalMaterial` at once, as the
    /// JavaScript layer holds them: `clearcoat`, `clearcoatRoughness`, `ior`,
    /// `specularIntensity`, `sheen`, `sheenRoughness`, `iridescence`,
    /// `iridescenceIOR`, `iridescenceThicknessRange` (min, max),
    /// `anisotropy`, `anisotropyRotation`, `transmission`, `thickness`,
    /// `attenuationDistance` and `dispersion`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = setPhysical)]
    pub fn set_physical(
        &mut self,
        clearcoat: f32,
        clearcoat_roughness: f32,
        ior: f32,
        specular_intensity: f32,
        sheen: f32,
        sheen_roughness: f32,
        iridescence: f32,
        iridescence_ior: f32,
        iridescence_thickness_min: f32,
        iridescence_thickness_max: f32,
        anisotropy: f32,
        anisotropy_rotation: f32,
        transmission: f32,
        thickness: f32,
        attenuation_distance: f32,
        dispersion: f32,
    ) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.clearcoat = clearcoat;
            m.clearcoat_roughness = clearcoat_roughness;
            m.ior = ior;
            m.specular_intensity = specular_intensity;
            m.sheen = sheen;
            m.sheen_roughness = sheen_roughness;
            m.iridescence = iridescence;
            m.iridescence_ior = iridescence_ior;
            m.iridescence_thickness_range = [iridescence_thickness_min, iridescence_thickness_max];
            m.anisotropy = anisotropy;
            m.anisotropy_rotation = anisotropy_rotation;
            m.transmission = transmission;
            m.thickness = thickness;
            m.attenuation_distance = attenuation_distance;
            m.dispersion = dispersion;
        }
    }

    /// three.js `sheenColor`.
    #[wasm_bindgen(js_name = setSheenColor)]
    pub fn set_sheen_color(&mut self, color: &WebColor) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.sheen_color = color.inner;
        }
    }

    /// three.js `specularColor`.
    #[wasm_bindgen(js_name = setSpecularColor)]
    pub fn set_specular_color(&mut self, color: &WebColor) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.specular_color = color.inner;
        }
    }

    /// three.js `attenuationColor`.
    #[wasm_bindgen(js_name = setAttenuationColor)]
    pub fn set_attenuation_color(&mut self, color: &WebColor) {
        if let crate::Material::Physical(m) = Arc::make_mut(&mut self.inner) {
            m.attenuation_color = color.inner;
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

    /// The transparency compositing mode, as `setTransparencyMode` numbers
    /// it (0 for anything but a `PhysicalMaterial`).
    #[wasm_bindgen(getter, js_name = transparencyMode)]
    pub fn transparency_mode(&self) -> u32 {
        use crate::TransparencyMode::*;
        match &*self.inner {
            crate::Material::Physical(m) => match m.transparency {
                Blend => 0,
                Glass => 1,
                Oit => 2,
                Refract => 3,
            },
            _ => 0,
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
    /// three.js `material.opacity`, on every material that has one.
    #[wasm_bindgen(js_name = setOpacity)]
    pub fn set_opacity(&mut self, opacity: f32) {
        Arc::make_mut(&mut self.inner).set_opacity(opacity);
    }

    /// three.js `PointsMaterial.size`, in pixels.
    #[wasm_bindgen(js_name = setPointSize)]
    pub fn set_point_size(&mut self, size: f32) {
        if let crate::Material::Points(m) = Arc::make_mut(&mut self.inner) {
            m.size = size;
        }
    }

    /// Mark a material as needing alpha blending. three.js uses an explicit
    /// `transparent` flag separate from `opacity` so semi-transparent textures
    /// can render correctly even at opacity 1.0.
    #[wasm_bindgen(js_name = setTransparent)]
    pub fn set_transparent(&mut self, transparent: bool) {
        Arc::make_mut(&mut self.inner).set_transparent(transparent);
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
            crate::Material::Lambert(m) => m.map = Some(tex),
            crate::Material::Phong(m) => m.map = Some(tex),
            crate::Material::Toon(m) => m.map = Some(tex),
            crate::Material::Matcap(m) => m.map = Some(tex),
            crate::Material::Points(m) => m.map = Some(tex),
            _ => {}
        }
    }

    /// A Standard or Physical material's own `envMap` slot.
    fn pbr_env_slot(&mut self) -> Option<&mut Option<Arc<crate::CubeTexture>>> {
        if !matches!(
            &*self.inner,
            crate::Material::Standard(_) | crate::Material::Physical(_)
        ) {
            return None;
        }
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => Some(&mut m.env_map),
            crate::Material::Physical(m) => Some(&mut m.env_map),
            _ => None,
        }
    }

    /// Apply `f` to the env-map state of a Basic, Lambert or Phong material;
    /// whether this is one.
    fn env_mut(&mut self, f: impl Fn(&mut EnvState<'_>)) -> bool {
        let inner = Arc::make_mut(&mut self.inner);
        let mut state = match inner {
            crate::Material::Basic(m) => EnvState::of(m),
            crate::Material::Lambert(m) => EnvState::of(m),
            crate::Material::Phong(m) => EnvState::of(m),
            _ => return false,
        };
        f(&mut state);
        true
    }

    fn set_slot(&mut self, name: &str, tex: Option<std::sync::Arc<crate::Texture>>) -> bool {
        match Arc::make_mut(&mut self.inner).map_slot_mut(name) {
            Some(slot) => {
                *slot = tex;
                true
            }
            None => false,
        }
    }

    fn set_ao_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        match Arc::make_mut(&mut self.inner) {
            crate::Material::Standard(m) => m.ao_map = Some(tex),
            crate::Material::Physical(m) => m.ao_map = Some(tex),
            crate::Material::Lambert(m) => m.ao_map = Some(tex),
            crate::Material::Phong(m) => m.ao_map = Some(tex),
            crate::Material::Toon(m) => m.ao_map = Some(tex),
            _ => {}
        }
    }

    fn set_normal_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.normal_map = Some(tex),
            crate::Material::Physical(m) => m.normal_map = Some(tex),
            crate::Material::Lambert(m) => m.normal_map = Some(tex),
            crate::Material::Phong(m) => m.normal_map = Some(tex),
            crate::Material::Toon(m) => m.normal_map = Some(tex),
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
            crate::Material::Lambert(m) => m.displacement_map = Some(tex),
            crate::Material::Phong(m) => m.displacement_map = Some(tex),
            crate::Material::Toon(m) => m.displacement_map = Some(tex),
            _ => {}
        }
    }

    fn set_emissive_map_arc(&mut self, tex: std::sync::Arc<crate::Texture>) {
        let inner = Arc::make_mut(&mut self.inner);
        match inner {
            crate::Material::Standard(m) => m.emissive_map = Some(tex),
            crate::Material::Physical(m) => m.emissive_map = Some(tex),
            crate::Material::Lambert(m) => m.emissive_map = Some(tex),
            crate::Material::Phong(m) => m.emissive_map = Some(tex),
            crate::Material::Toon(m) => m.emissive_map = Some(tex),
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

/// A custom-shader material, from JavaScript.
///
/// three.js's `ShaderMaterial` is one of the most-used classes in the library
/// and this had no counterpart at all, so every custom shader in the crate was
/// reachable from Rust and from nowhere else — which for a package that ships
/// to npm is most of the audience.
///
/// The WGSL contract is the one documented on
/// [`crate::ShaderMaterial`]: a fragment defining
/// `@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32>`, optionally a
/// vertex stage defining `@vertex fn vs_custom`, both compiled against the
/// renderer's preamble.
///
/// ```js
/// const m = new THREE.ShaderMaterial(`
///   @fragment
///   fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
///     let tint = u_data.data[0].rgb;
///     return vec4<f32>(framebuffer_encode(apply_tone_mapping(tint)), 1.0);
///   }
/// `);
/// m.setData([1.0, 0.3, 0.1, 1.0]);
/// scene.add(new THREE.Mesh(geometry, m.material()));
/// ```
///
/// On WebGL2 there are no storage buffers, so read the storage slots through
/// `s0(i)`/`s1(i)`/`s2(i)` rather than `u_s0[i]` — those are defined on every
/// backend and fall back to a texture where they have to.
#[wasm_bindgen]
pub struct WebShaderMaterial {
    inner: crate::ShaderMaterial,
    /// `u_vol0` / `u_vol1` as set, kept apart from `inner.volumes` (which is
    /// positional and has no holes — see `sync_volumes`).
    volumes: [Option<crate::marks::gpu::TextureView>; 2],
}

#[wasm_bindgen]
impl WebShaderMaterial {
    /// `fragment` is WGSL defining `@fragment fn fs_main`.
    #[wasm_bindgen(constructor)]
    pub fn new(fragment: &str) -> WebShaderMaterial {
        WebShaderMaterial {
            inner: crate::ShaderMaterial::new(fragment),
            volumes: [None, None],
        }
    }

    /// three.js `Data3DTexture` bound as `u_vol0` (`slot` 0) or `u_vol1` (1),
    /// sampled with `u_vol_samp` or read with `textureLoad`. 8-bit data, x
    /// fastest: `channels` 1 is `RedFormat` (`R8Unorm`), 4 is `RGBAFormat`.
    ///
    /// Uploaded to the GPU the WebGPU renderer opened, so a renderer must
    /// exist first; WebGL2 has no shared device and is refused.
    #[wasm_bindgen(js_name = setVolume)]
    pub fn set_volume(
        &mut self,
        slot: u32,
        data: &[u8],
        width: u32,
        height: u32,
        depth: u32,
        channels: u32,
    ) -> Result<(), JsValue> {
        use crate::volume::VolumeFormat;
        let format = if channels == 4 {
            VolumeFormat::Rgba8Unorm
        } else {
            VolumeFormat::R8Unorm
        };
        self.put_volume(slot, width, height, depth, format, data.to_vec())
    }

    /// A `FloatType` `Data3DTexture` (one channel). Stored as `R16Float`, the
    /// float format that can be filtered everywhere: the volume slots are
    /// declared filterable, which `R32Float` is not without an extension.
    #[wasm_bindgen(js_name = setVolumeF32)]
    pub fn set_volume_f32(
        &mut self,
        slot: u32,
        data: &[f32],
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<(), JsValue> {
        let bytes = data
            .iter()
            .flat_map(|v| crate::renderer::readback::f32_to_f16(*v).to_le_bytes())
            .collect();
        self.put_volume(
            slot,
            width,
            height,
            depth,
            crate::volume::VolumeFormat::R16Float,
            bytes,
        )
    }

    /// Unbind a volume slot (it reads as zero again).
    #[wasm_bindgen(js_name = clearVolume)]
    pub fn clear_volume(&mut self, slot: u32) {
        if let Some(v) = self.volumes.get_mut(slot as usize) {
            *v = None;
        }
        let device = SHARED_BROWSER_GPU.with(|g| g.borrow().as_ref().map(|(d, _)| d.clone()));
        self.sync_volumes(device.as_deref());
    }

    /// WGSL defining `@vertex fn vs_custom(in: VsIn, ...) -> VsOut`.
    ///
    /// For anything whose position has to be computed rather than read — a
    /// camera-facing ribbon, a strand, a billboard. `standard_vertex(in)` is
    /// the built-in transform, so the usual shape is to take its output and
    /// move it.
    #[wasm_bindgen(js_name = setVertex)]
    pub fn set_vertex(&mut self, vertex: &str) {
        self.inner.vertex = Some(vertex.to_string());
    }

    /// The uniform block, as a flat array of floats read four at a time into
    /// `u_data.data[0..16]`. Anything past sixty-four floats is ignored.
    #[wasm_bindgen(js_name = setData)]
    pub fn set_data(&mut self, data: Vec<f32>) {
        self.inner.data = data
            .chunks(4)
            .map(|c| {
                let mut v = [0.0f32; 4];
                v[..c.len()].copy_from_slice(c);
                v
            })
            .collect();
    }

    /// One of the three storage slots, read as `s0(i)`, `s1(i)` or `s2(i)`.
    ///
    /// Use those rather than `u_s0[i]`: on WebGL2 there are no storage buffers
    /// and the same data arrives as a texture instead, which the accessors
    /// know about and a raw index does not.
    #[wasm_bindgen(js_name = setStorage)]
    pub fn set_storage(&mut self, slot: u32, data: Vec<f32>) {
        match slot {
            0 => self.inner.storage0 = data,
            1 => self.inner.storage1 = data,
            _ => self.inner.storage2 = data,
        }
    }

    #[wasm_bindgen(js_name = setTransparent)]
    pub fn set_transparent(&mut self, transparent: bool) {
        self.inner.transparent = transparent;
    }

    #[wasm_bindgen(js_name = setOpacity)]
    pub fn set_opacity(&mut self, opacity: f32) {
        self.inner.opacity = opacity;
    }

    /// three.js `alphaToCoverage`: under MSAA the fragment's alpha is the
    /// share of the pixel's samples it covers (return
    /// `coverage_alpha(c)`, or write samples with `frag_out`).
    #[wasm_bindgen(js_name = setAlphaToCoverage)]
    pub fn set_alpha_to_coverage(&mut self, on: bool) {
        self.inner.alpha_to_coverage = on;
    }

    /// `0` FrontSide, `1` BackSide, `2` DoubleSide — three.js's `side`.
    #[wasm_bindgen(js_name = setSide)]
    pub fn set_side(&mut self, side: u32) {
        self.inner.side = side;
    }

    /// Override depth writing: by default opaque writes and blended does not.
    #[wasm_bindgen(js_name = setDepthWrite)]
    pub fn set_depth_write(&mut self, write: bool) {
        self.inner.render_state.depth_write = Some(write);
    }

    /// Draw in the screen-space pass, with the opaque scene available to
    /// sample — what screen-space refraction needs.
    #[wasm_bindgen(js_name = setScreenSpace)]
    pub fn set_screen_space(&mut self, on: bool) {
        self.inner.screen_space = on;
    }

    /// Compile-check the shader, returning the error or `undefined`.
    ///
    /// Always `undefined` in a browser: naga is not in wgpu's wasm build, so
    /// there is nothing to check against and the browser's own WGSL compiler
    /// reports to the console instead. `validationAvailable()` says so rather
    /// than letting an always-passing check look like a passing one.
    #[wasm_bindgen(js_name = validate)]
    pub fn validate(&self) -> Option<String> {
        self.inner.validate(true).err()
    }

    /// Whether [`validate`](Self::validate) can check anything here.
    #[wasm_bindgen(js_name = validationAvailable)]
    pub fn validation_available() -> bool {
        crate::ShaderMaterial::VALIDATION_AVAILABLE
    }

    /// The material to hand to a `Mesh`.
    #[wasm_bindgen(js_name = material)]
    pub fn material(&self) -> WebMaterial {
        WebMaterial {
            inner: Arc::new(crate::Material::Shader(self.inner.clone())),
        }
    }
}

impl WebShaderMaterial {
    fn put_volume(
        &mut self,
        slot: u32,
        width: u32,
        height: u32,
        depth: u32,
        format: crate::volume::VolumeFormat,
        bytes: Vec<u8>,
    ) -> Result<(), JsValue> {
        use crate::volume::{Data3DTexture, GpuVolume};
        if slot > 1 {
            return Err(JsValue::from_str(
                "a ShaderMaterial has two volume slots, 0 and 1",
            ));
        }
        let expected =
            width as usize * height as usize * depth as usize * format.bytes_per_texel() as usize;
        if bytes.len() != expected {
            return Err(JsValue::from_str(&format!(
                "a {width}x{height}x{depth} {format:?} volume needs {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let Some((device, queue)) = SHARED_BROWSER_GPU.with(|g| g.borrow().clone()) else {
            return Err(JsValue::from_str(
                "3D textures need the WebGPU renderer: create a WebGLRenderer before setting a volume",
            ));
        };
        let vol = GpuVolume::from_texture(
            &device,
            &queue,
            &Data3DTexture::new(width, height, depth, format, bytes),
        );
        self.volumes[slot as usize] = Some(vol.view);
        self.sync_volumes(Some(&device));
        Ok(())
    }

    /// `inner.volumes` from the slots. It is positional, so `u_vol1` alone
    /// needs something in slot 0: a 1x1x1 zero volume, as an empty slot reads.
    fn sync_volumes(&mut self, device: Option<&wgpu::Device>) {
        self.inner.volumes = match &self.volumes {
            [None, None] => Vec::new(),
            [Some(a), None] => vec![a.clone()],
            [a, Some(b)] => {
                let first = a.clone().or_else(|| {
                    let device = device?;
                    let v = crate::volume::GpuVolume::new(
                        device,
                        [1, 1, 1],
                        crate::volume::VolumeFormat::R8Unorm,
                    );
                    Some(v.view)
                });
                first.map_or_else(Vec::new, |a| vec![a, b.clone()])
            }
        };
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

    /// A CSS colour string — hex (`#rgb` to `#rrggbbaa`), the named colours,
    /// `rgb()`, `hsl()`, `hwb()`, `lab()`, `lch()`, `oklab()`, `oklch()`,
    /// `color()` — as linear RGB, or `undefined` if it is not one. The alpha
    /// is dropped; `cssAlpha` reads it.
    #[wasm_bindgen(js_name = fromCss)]
    pub fn from_css(css: &str) -> Option<WebColor> {
        crate::Color::from_css(css)
            .ok()
            .map(|inner| WebColor { inner })
    }

    /// The alpha of a CSS colour string (1 when it gives none), or
    /// `undefined` if it is not one.
    #[wasm_bindgen(js_name = cssAlpha)]
    pub fn css_alpha(css: &str) -> Option<f32> {
        crate::Color::parse_css(css).ok().map(|(_, a)| a)
    }
}

/// three.js's shadow-map type numbers as this renderer's filters.
fn shadow_type_of(kind: u32) -> crate::lights::ShadowType {
    use crate::lights::ShadowType::*;
    match kind {
        0 => Basic,
        2 => PcfSoft,
        3 => Pcss,
        _ => Pcf,
    }
}

/// Data to colour — `threers::math::ColorScale`, d3's sequential, diverging,
/// quantize, quantile and threshold scales over any named scheme.
#[wasm_bindgen]
pub struct WebColorScale {
    inner: crate::math::ColorScale,
}

fn named_ramp(name: &str) -> Result<crate::math::ColorRamp, JsValue> {
    crate::math::ColorRamp::named(name)
        .ok_or_else(|| JsValue::from_str(&format!("no colour scheme named {name}")))
}

#[wasm_bindgen]
impl WebColorScale {
    /// `lo..hi` along the named scheme (`"viridis"`, `"blues"`, `"RdBu"`,
    /// `"interpolateSpectral"`, …).
    pub fn sequential(scheme: &str, lo: f32, hi: f32) -> Result<WebColorScale, JsValue> {
        Ok(Self {
            inner: crate::math::ColorScale::sequential(named_ramp(scheme)?, lo, hi),
        })
    }

    /// Away from `mid` both ways.
    pub fn diverging(scheme: &str, lo: f32, mid: f32, hi: f32) -> Result<WebColorScale, JsValue> {
        Ok(Self {
            inner: crate::math::ColorScale::diverging(named_ramp(scheme)?, lo, mid, hi),
        })
    }

    /// `classes` equal-width bins.
    pub fn quantize(
        scheme: &str,
        lo: f32,
        hi: f32,
        classes: usize,
    ) -> Result<WebColorScale, JsValue> {
        Ok(Self {
            inner: crate::math::ColorScale::quantize(named_ramp(scheme)?, lo, hi, classes),
        })
    }

    /// `classes` bins of equal count of `data`.
    pub fn quantile(scheme: &str, data: &[f32], classes: usize) -> Result<WebColorScale, JsValue> {
        Ok(Self {
            inner: crate::math::ColorScale::quantile(named_ramp(scheme)?, data, classes),
        })
    }

    /// Explicit class breaks.
    pub fn threshold(scheme: &str, breaks: Vec<f32>) -> Result<WebColorScale, JsValue> {
        Ok(Self {
            inner: crate::math::ColorScale::threshold(named_ramp(scheme)?, breaks),
        })
    }

    /// Logarithmic in `base`.
    pub fn log(self, base: f32) -> WebColorScale {
        Self {
            inner: self.inner.log(base),
        }
    }

    /// By the square root.
    pub fn sqrt(self) -> WebColorScale {
        Self {
            inner: self.inner.sqrt(),
        }
    }

    /// By a power.
    pub fn pow(self, exponent: f32) -> WebColorScale {
        Self {
            inner: self.inner.pow(exponent),
        }
    }

    /// Symmetric log, linear within `constant` of zero.
    pub fn symlog(self, constant: f32) -> WebColorScale {
        Self {
            inner: self.inner.symlog(constant),
        }
    }

    /// The scheme reversed, end to end.
    pub fn reversed(self) -> WebColorScale {
        let ramp = self.inner.ramp().clone().reversed();
        let domain = self.inner.domain().to_vec();
        let mut inner = if domain.len() == 3 {
            crate::math::ColorScale::diverging(ramp, domain[0], domain[1], domain[2])
        } else {
            crate::math::ColorScale::sequential(ramp, domain[0], domain[1])
        };
        inner = match self.inner.kind().clone() {
            crate::math::ScaleKind::Log { base } => inner.log(base),
            crate::math::ScaleKind::Pow { exponent } => inner.pow(exponent),
            crate::math::ScaleKind::Symlog { constant } => inner.symlog(constant),
            _ => return Self { inner: self.inner },
        };
        Self { inner }
    }

    /// The colour for `value`, linear RGB.
    pub fn map(&self, value: f32) -> WebColor {
        WebColor {
            inner: self.inner.map(value),
        }
    }

    /// Colours for `values` as sRGB bytes, four to a value — straight into a
    /// colour attribute or a data texture.
    #[wasm_bindgen(js_name = mapRgba8)]
    pub fn map_rgba8(&self, values: &[f32]) -> Vec<u8> {
        self.inner.map_rgba8(values).into_iter().flatten().collect()
    }

    /// A legend's rows, `[from, to, r, g, b]` each (linear RGB), `n` steps
    /// for a continuous scale.
    pub fn legend(&self, n: usize) -> Vec<f32> {
        self.inner
            .legend(n)
            .into_iter()
            .flat_map(|(a, b, c)| [a, b, c.r, c.g, c.b])
            .collect()
    }

    /// Every scheme name the scales know.
    #[wasm_bindgen(js_name = schemeNames)]
    pub fn scheme_names() -> Vec<String> {
        crate::math::ColorRamp::scheme_names()
            .into_iter()
            .map(|(n, _)| n.to_string())
            .collect()
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
