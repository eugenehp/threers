//! A scene drawn into a winit window, on wgpu or on the native Metal backend.
//!
//! Every windowed program needs the same sixty lines: an instance, a surface
//! for the window, an adapter that can present to it, a device with limits
//! high enough for the renderer, a surface configuration, and a frame loop
//! that knows which acquire failures to retry and which to ignore.
//! [`WindowRenderer`] is those lines, once.
//!
//! On Apple platforms with the `metal` feature — macOS, iOS and iPadOS, tvOS,
//! visionOS — the same calls draw through
//! [`MetalRenderer`](crate::metal::MetalRenderer) into a `CAMetalLayer`
//! instead: a winit window's view under `THREERS_BACKEND=metal` or
//! [`WindowBackend::Metal`], or an app's own `UIView`, `NSView` or layer
//! ([`WindowRenderer::from_ui_view`] and friends — the way in on tvOS and
//! visionOS, where winit does not run). A program written against this runs on
//! either backend without a change.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use threers::{PerspectiveCamera, Scene};
//! # use threers::renderer::WindowRenderer;
//! # fn demo(window: Arc<winit::window::Window>, scene: &mut Scene, camera: &PerspectiveCamera) {
//! let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");
//! // On `WindowEvent::Resized(size)`:
//! let size = window.inner_size();
//! view.resize(size.width, size.height);
//! // On `WindowEvent::RedrawRequested`:
//! view.render(scene, camera);
//! # }
//! ```

#[cfg(winit_window)]
use std::sync::Arc;

#[cfg(winit_window)]
use winit::window::Window;

use super::Renderer;
use crate::cameras::{Camera, PerspectiveCamera};
use crate::helpers::QuadView;
use crate::overlay::Overlay;
use crate::scene::Scene;

/// What [`WindowRenderer::render_with`] hands its callback: the frame just
/// drawn, as [`marks`](crate::marks) draw over it, before it is presented.
pub struct MarksFrame<'a> {
    /// The GPU to draw the marks with — this window's.
    pub gpu: crate::marks::gpu::Gpu,
    /// The frame: its colour (multisampled, and what that resolves to) and
    /// its depth.
    pub target: crate::marks::MarkTarget<'a>,
    /// The frame's depth for a pick to test against, when it can be read —
    /// single-sampled; `None` under MSAA.
    pub occluder: Option<crate::marks::gpu::TextureView>,
    /// Its size, tone mapping and depth direction, for the
    /// [`MarkView`](crate::marks::MarkView).
    pub surface: crate::marks::MarkSurface,
}

/// Which backend [`WindowRenderer::with_backend`] draws with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowBackend {
    /// wgpu, unless `THREERS_BACKEND=metal` asks for Metal (falling back to
    /// wgpu, with a notice, if Metal cannot open).
    #[default]
    Auto,
    Wgpu,
    /// The native Metal backend, on any Apple platform with the `metal`
    /// feature.
    Metal,
}

/// The view or layer a Metal window draws into.
#[cfg(metal_backend)]
#[derive(Clone, Copy)]
enum ViewSource {
    #[cfg(target_os = "macos")]
    NsView(*mut std::ffi::c_void),
    #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
    UiView(*mut std::ffi::c_void),
    Layer(*mut std::ffi::c_void),
}

#[cfg(metal_backend)]
impl ViewSource {
    /// A surface on it.
    ///
    /// # Safety
    /// The view or layer is live.
    unsafe fn surface(
        self,
        device: &crate::metal::MetalDevice,
        (w, h): (u32, u32),
        samples: u32,
    ) -> Result<crate::metal::MetalSurface, crate::metal::MetalError> {
        use crate::metal::MetalSurface;
        match self {
            #[cfg(target_os = "macos")]
            ViewSource::NsView(v) => MetalSurface::from_ns_view(device, v, w, h, samples),
            #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
            ViewSource::UiView(v) => MetalSurface::from_ui_view(device, v, w, h, samples),
            ViewSource::Layer(l) => MetalSurface::from_layer(device, l, w, h, samples),
        }
    }
}

/// A window's renderer. See the [module docs](self).
pub struct WindowRenderer {
    backend: Backend,
    size: (u32, u32),
}

enum Backend {
    #[cfg(winit_window)]
    Wgpu(Box<WgpuWindow>),
    #[cfg(metal_backend)]
    Metal(Box<MetalWindow>),
}

#[cfg(winit_window)]
struct WgpuWindow {
    /// What marks draw with: this device and its queue.
    marks: crate::marks::gpu::Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The surface format's sRGB variant, for the overlay: fixed-function
    /// blending on a plain `Unorm` view mixes encodings rather than light.
    srgb_format: wgpu::TextureFormat,
    device: Arc<wgpu::Device>,
    renderer: Renderer,
}

#[cfg(metal_backend)]
struct MetalWindow {
    device: crate::metal::MetalDevice,
    /// What marks draw with: this device and its queue.
    marks: crate::marks::gpu::Gpu,
    /// What the layer is attached to, for rebuilding the surface at another
    /// sample count.
    source: ViewSource,
    sample_count: u32,
    surface: crate::metal::MetalSurface,
    renderer: crate::metal::MetalRenderer,
}

impl WindowRenderer {
    #[cfg(winit_window)]
    /// Open the GPU for `window` and size the drawing to it. Blocks while the
    /// device is acquired. Call on the thread running the event loop.
    ///
    /// wgpu, or the native Metal backend when `THREERS_BACKEND=metal` asks —
    /// [`with_backend`](Self::with_backend) to choose in code.
    pub fn new(window: Arc<Window>) -> Result<Self, String> {
        Self::with_backend(window, WindowBackend::Auto)
    }

    #[cfg(winit_window)]
    /// [`new`](Self::new) on the backend `backend` names. An explicit
    /// [`WindowBackend::Metal`] that cannot open is an error rather than a
    /// fallback to wgpu.
    pub fn with_backend(window: Arc<Window>, backend: WindowBackend) -> Result<Self, String> {
        let size = window.inner_size();
        let size = (size.width.max(1), size.height.max(1));
        #[cfg(metal_backend)]
        {
            let metal = match backend {
                WindowBackend::Metal => true,
                WindowBackend::Wgpu => false,
                WindowBackend::Auto => super::headless_metal::requested(),
            };
            if metal {
                match MetalWindow::from_winit(&window, size) {
                    Ok(m) => {
                        return Ok(Self {
                            backend: Backend::Metal(Box::new(m)),
                            size,
                        })
                    }
                    Err(e) if backend == WindowBackend::Metal => return Err(e),
                    Err(e) => super::headless_metal::warn_fallback(&format!("this window ({e})")),
                }
            }
        }
        #[cfg(not(metal_backend))]
        if backend == WindowBackend::Metal {
            return Err(
                "the native Metal backend needs the `metal` feature on an Apple platform".into(),
            );
        }
        let w = pollster::block_on(WgpuWindow::new(window, size))?;
        Ok(Self {
            backend: Backend::Wgpu(Box::new(w)),
            size,
        })
    }

    #[cfg(metal_backend)]
    /// A renderer drawing into the `CAMetalLayer` of a `UIView` — an iOS,
    /// iPadOS, tvOS or visionOS app's own view, no winit needed — on the
    /// native Metal backend. `width × height` in physical pixels. Call on
    /// the main thread.
    ///
    /// # Safety
    /// `ui_view` is a live `UIView` whose layer is a `CAMetalLayer`, alive as
    /// long as the renderer.
    #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
    pub unsafe fn from_ui_view(
        ui_view: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::from_source(ViewSource::UiView(ui_view), (width, height))
    }

    #[cfg(metal_backend)]
    /// A renderer drawing into an `NSView`, on the native Metal backend.
    ///
    /// # Safety
    /// `ns_view` is a live `NSView`, alive as long as the renderer.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_ns_view(
        ns_view: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::from_source(ViewSource::NsView(ns_view), (width, height))
    }

    #[cfg(metal_backend)]
    /// A renderer drawing into a `CAMetalLayer` you manage, on any Apple
    /// platform and any thread.
    ///
    /// # Safety
    /// `layer` is a live `CAMetalLayer`, alive as long as the renderer.
    pub unsafe fn from_layer(
        layer: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::from_source(ViewSource::Layer(layer), (width, height))
    }

    #[cfg(metal_backend)]
    unsafe fn from_source(source: ViewSource, size: (u32, u32)) -> Result<Self, String> {
        let size = (size.0.max(1), size.1.max(1));
        let m = MetalWindow::new(source, size)?;
        Ok(Self {
            backend: Backend::Metal(Box::new(m)),
            size,
        })
    }

    /// New drawable size in physical pixels (winit's `Resized`); no-op when
    /// unchanged.
    pub fn resize(&mut self, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size == self.size {
            return;
        }
        self.size = size;
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => {
                w.config.width = size.0;
                w.config.height = size.1;
                w.surface.configure(&w.device, &w.config);
                w.renderer.resize(size.0, size.1);
            }
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                if let Err(e) = m.surface.resize(size.0, size.1) {
                    log::error!("Metal surface resize: {e}");
                }
            }
        }
    }

    /// MSAA sample count: 1 (off) or 4.
    pub fn set_msaa(&mut self, samples: u32) {
        let samples = if samples <= 1 { 1 } else { 4 };
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.renderer.set_msaa(samples),
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                if samples != m.sample_count {
                    // SAFETY: the view is the window's, alive as long as it.
                    match unsafe { m.source.surface(&m.device, self.size, samples) } {
                        Ok(surface) => {
                            m.surface = surface;
                            m.sample_count = samples;
                        }
                        Err(e) => log::error!("Metal MSAA {samples}x: {e}"),
                    }
                }
            }
        }
    }

    /// Physical pixels per logical pixel, so viewport and scissor rectangles
    /// (a quad view's panes) can be given in logical pixels.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.renderer.set_pixel_ratio(ratio),
            #[cfg(metal_backend)]
            Backend::Metal(m) => m.renderer.set_pixel_ratio(ratio),
        }
    }

    /// Drawable size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Width over height, for the camera.
    pub fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }

    /// Draw and present a frame. `false` when there was nothing to draw into
    /// this time — the window is hidden, or every drawable is in flight —
    /// which is normal and needs nothing from the caller.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.render(scene, camera),
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                let Some(frame) = m.surface.next_frame() else {
                    return false;
                };
                match m.renderer.render(scene, camera, &frame.attachments()) {
                    Ok(_) => true,
                    Err(e) => {
                        log::error!("Metal render: {e}");
                        false
                    }
                }
            }
        }
    }

    /// One frame as [`sketch`](crate::sketch) draws it: the scene, through a
    /// [`QuadView`] when given one, then the 2D [`Overlay`] over the top when
    /// anything was drawn on it, then presented. `false` as for
    /// [`render`](Self::render).
    pub fn render_frame(
        &mut self,
        scene: &mut Scene,
        camera: &PerspectiveCamera,
        quad: Option<&mut QuadView>,
        overlay: Option<&mut Overlay>,
    ) -> bool {
        let overlay = overlay.filter(|o| o.touched());
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.render_frame(scene, camera, quad, overlay),
            #[cfg(metal_backend)]
            Backend::Metal(m) => m.render_frame(scene, camera, quad, overlay),
        }
    }

    /// The wgpu [`Renderer`], for settings beyond the scene — `None` on the
    /// native Metal backend.
    pub fn renderer(&mut self) -> Option<&mut Renderer> {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => Some(&mut w.renderer),
            #[cfg(metal_backend)]
            Backend::Metal(_) => None,
        }
    }

    /// The GPU marks draw with in this window — make a
    /// [`MarkRenderer`](crate::marks::MarkRenderer) with
    /// [`new_on`](crate::marks::MarkRenderer::new_on) it.
    pub fn marks_gpu(&self) -> crate::marks::gpu::Gpu {
        match &self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.marks.clone(),
            #[cfg(metal_backend)]
            Backend::Metal(m) => m.marks.clone(),
        }
    }

    /// The frames' size, tone mapping and depth direction, for a
    /// [`MarkView`](crate::marks::MarkView) or a
    /// [`MapView`](crate::marks::MapView) frame.
    pub fn mark_surface(&self) -> crate::marks::MarkSurface {
        match &self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => crate::marks::MarkSurface::of(&w.renderer),
            #[cfg(metal_backend)]
            Backend::Metal(m) => crate::marks::MarkSurface {
                size: self.size,
                tone_mapping: m.renderer.tone_mapping(),
                reversed_z: false,
            },
        }
    }

    /// Draw the scene, then `over` it — marks, a pick — then present. `false`
    /// as for [`render`](Self::render), and `over` is not called.
    pub fn render_with(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        over: impl FnOnce(&mut MarksFrame<'_>),
    ) -> bool {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => {
                let Some(frame) = w.acquire() else {
                    return false;
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                w.renderer.render(scene, camera, &view, false);
                let (msaa, depth, _) = w.renderer.overlay_attachments();
                let mut f = MarksFrame {
                    gpu: w.marks.clone(),
                    target: crate::marks::MarkTarget::over_frame(&w.renderer, &view),
                    occluder: msaa.is_none().then(|| depth.into()),
                    surface: crate::marks::MarkSurface::of(&w.renderer),
                };
                over(&mut f);
                drop(f);
                w.renderer.queue_arc().present(frame);
                true
            }
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                let Some(frame) = m.surface.next_frame() else {
                    return false;
                };
                let pass = frame.attachments();
                // Marks drawn over the frame load it: keep it whole, for this
                // frame only — it costs bandwidth nothing else reads.
                m.renderer.set_keep_attachments(true);
                let drawn = m.renderer.render(scene, camera, &pass.without_drawable());
                m.renderer.set_keep_attachments(false);
                if let Err(e) = drawn {
                    log::error!("Metal render: {e}");
                    return false;
                }
                let surface = crate::marks::MarkSurface {
                    size: self.size,
                    tone_mapping: m.renderer.tone_mapping(),
                    reversed_z: false,
                };
                // SAFETY: the frame's textures, alive until it is presented.
                let target = unsafe { crate::marks::MarkTarget::from_metal_pass(&pass) };
                let occluder = (target.samples == 1)
                    .then(|| target.depth.clone())
                    .flatten();
                let mut f = MarksFrame {
                    gpu: m.marks.clone(),
                    target,
                    occluder,
                    surface,
                };
                over(&mut f);
                drop(f);
                if let Err(e) = m.renderer.present(&pass) {
                    log::error!("Metal present: {e}");
                }
                true
            }
        }
    }

    /// Let finished GPU work report back — a pick in flight. Once a frame, as
    /// wgpu's `device.poll(PollType::Poll)`, on either backend.
    pub fn poll(&self) {
        self.marks_gpu().poll();
    }

    /// Tone mapping and exposure, as [`Renderer::set_tone_mapping`].
    pub fn set_tone_mapping(&mut self, mode: crate::renderer::ToneMapping, exposure: f32) {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.renderer.set_tone_mapping(mode, exposure),
            #[cfg(metal_backend)]
            Backend::Metal(m) => m.renderer.set_tone_mapping(mode, exposure),
        }
    }

    /// Upload `geometry` now rather than at its first draw, for a compute
    /// pass that writes its [`vertex_buffer`](Self::vertex_buffer).
    pub fn upload_geometry(&mut self, geometry: &std::sync::Arc<crate::core::BufferGeometry>) {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.renderer.upload_geometry(geometry),
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                if let Err(e) = m.renderer.upload_geometry(geometry) {
                    log::warn!("metal: geometry upload failed: {e}");
                }
            }
        }
    }

    /// The vertex buffer an uploaded geometry is drawn from, for a compute
    /// pass on [`marks_gpu`](Self::marks_gpu) to write in place;
    /// [`vertex_floats`](Self::vertex_floats) `f32` a vertex. As
    /// [`HeadlessRenderer::vertex_buffer`](crate::HeadlessRenderer::vertex_buffer).
    pub fn vertex_buffer(
        &mut self,
        geometry: &std::sync::Arc<crate::core::BufferGeometry>,
    ) -> Option<crate::marks::gpu::Buffer> {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w
                .renderer
                .vertex_buffer(geometry)
                .map(crate::marks::gpu::Buffer::from),
            #[cfg(metal_backend)]
            Backend::Metal(m) => {
                let (id, size) = m.renderer.vertex_buffer(geometry)?;
                // SAFETY: the renderer's live buffer of `size` bytes.
                Some(unsafe { crate::marks::gpu::Buffer::from_metal(id, size as u64) })
            }
        }
    }

    /// `f32`s a vertex in [`vertex_buffer`](Self::vertex_buffer): 16 on
    /// wgpu, 12 on the native Metal backend.
    pub fn vertex_floats(&self) -> u32 {
        if self.is_native_metal() {
            12
        } else {
            (crate::renderer::gpu_mesh::GpuMesh::VERTEX_STRIDE / 4) as u32
        }
    }

    /// Drop every cached GPU mesh, as [`Renderer::clear_geometry_cache`] —
    /// after replacing a scene's geometry wholesale.
    pub fn clear_geometry_cache(&mut self) {
        match &mut self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(w) => w.renderer.clear_geometry_cache(),
            #[cfg(metal_backend)]
            Backend::Metal(m) => m.renderer.clear_caches(),
        }
    }

    /// Whether this window draws through the native Metal backend.
    pub fn is_native_metal(&self) -> bool {
        match &self.backend {
            #[cfg(winit_window)]
            Backend::Wgpu(_) => false,
            #[cfg(metal_backend)]
            Backend::Metal(_) => true,
        }
    }
}

#[cfg(winit_window)]
impl WgpuWindow {
    async fn new(window: Arc<Window>, size: (u32, u32)) -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("no adapter for this window: {e}"))?;
        // The renderer's main pass binds more textures and samplers than the
        // downlevel limits allow, and allocates a 4096 shadow map; raise those
        // to what the adapter has, as `HeadlessRenderer` does.
        let adapter_limits = adapter.limits();
        let mut limits =
            wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
        limits.max_sampled_textures_per_shader_stage =
            adapter_limits.max_sampled_textures_per_shader_stage;
        limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;
        // One vertex buffer a mesh; the default cap of 256 MiB is below what a
        // large mesh needs and well below what the adapter allows.
        limits.max_buffer_size = limits.max_buffer_size.max(adapter_limits.max_buffer_size);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("threers window"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))?;
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let srgb_format = format.add_srgb_suffix();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: if srgb_format == format {
                vec![]
            } else {
                vec![srgb_format]
            },
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &config);
        let marks = crate::marks::gpu::Gpu::from((&*device, &*queue));
        let renderer = Renderer::new(device.clone(), queue, format, size.0, size.1);
        Ok(Self {
            marks,
            surface,
            config,
            srgb_format,
            device,
            renderer,
        })
    }

    fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        let Some(frame) = self.acquire() else {
            return false;
        };
        self.draw(scene, camera, frame)
    }

    fn render_frame(
        &mut self,
        scene: &mut Scene,
        camera: &PerspectiveCamera,
        quad: Option<&mut QuadView>,
        overlay: Option<&mut Overlay>,
    ) -> bool {
        let Some(frame) = self.acquire() else {
            return false;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        match quad {
            Some(quad) => quad.render(
                &mut crate::helpers::ViewSurface {
                    renderer: &mut self.renderer,
                    view: &view,
                    linear_framebuffer: false,
                },
                scene,
                camera,
            ),
            None => self.renderer.render(scene, camera, &view, false),
        }
        if let Some(overlay) = overlay {
            let srgb_view = (self.srgb_format != self.config.format).then(|| {
                frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.srgb_format),
                    ..Default::default()
                })
            });
            self.renderer.draw_overlay(
                overlay,
                srgb_view.as_ref().unwrap_or(&view),
                self.srgb_format,
            );
        }
        self.renderer.queue_arc().present(frame);
        true
    }

    /// The next drawable, or `None` for a frame to skip (reconfiguring first
    /// when the surface asks for it).
    fn acquire(&mut self) -> Option<wgpu::SurfaceTexture> {
        use wgpu::CurrentSurfaceTexture as Acquired;
        match self.surface.get_current_texture() {
            Acquired::Success(f) => Some(f),
            // Usable now; the window changed under it, so fit the next one.
            Acquired::Suboptimal(f) => {
                self.surface.configure(&self.device, &self.config);
                Some(f)
            }
            // Hidden, minimised, or every drawable in flight: try next frame.
            Acquired::Timeout | Acquired::Occluded => None,
            Acquired::Outdated | Acquired::Lost => {
                self.surface.configure(&self.device, &self.config);
                None
            }
            other => {
                log::error!("surface: {other:?}");
                None
            }
        }
    }

    fn draw(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        frame: wgpu::SurfaceTexture,
    ) -> bool {
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(scene, camera, &view, false);
        self.renderer.queue_arc().present(frame);
        true
    }
}

#[cfg(metal_backend)]
impl MetalWindow {
    /// The view under a winit window: AppKit's on macOS, UIKit's on iOS.
    #[cfg(winit_window)]
    fn from_winit(window: &Window, size: (u32, u32)) -> Result<Self, String> {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let source = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(h) => ViewSource::NsView(h.ns_view.as_ptr()),
            #[cfg(target_os = "ios")]
            RawWindowHandle::UiKit(h) => ViewSource::UiView(h.ui_view.as_ptr()),
            other => return Err(format!("no Metal surface for this window: {other:?}")),
        };
        // SAFETY: the view is the live window's, and winit calls us on the
        // main thread, which the surface constructors check.
        unsafe { Self::new(source, size) }
    }

    unsafe fn new(source: ViewSource, size: (u32, u32)) -> Result<Self, String> {
        let device = crate::metal::MetalDevice::new().map_err(|e| e.to_string())?;
        let renderer =
            crate::metal::MetalRenderer::with_device(device.clone()).map_err(|e| e.to_string())?;
        let marks = crate::marks::gpu::Gpu::from(&device);
        let surface = source
            .surface(&device, size, 4)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "threers: drawing the window on the native Metal backend ({})",
            device.name()
        );
        Ok(Self {
            device,
            marks,
            source,
            sample_count: 4,
            surface,
            renderer,
        })
    }

    fn render_frame(
        &mut self,
        scene: &mut Scene,
        camera: &PerspectiveCamera,
        quad: Option<&mut QuadView>,
        overlay: Option<&mut Overlay>,
    ) -> bool {
        let Some(frame) = self.surface.next_frame() else {
            return false;
        };
        let pass = frame.attachments();
        // Drawn without the drawable, so nothing reaches the screen until the
        // overlay is over it; whichever goes last presents.
        let drawn = match quad {
            Some(quad) => {
                let mut target = crate::helpers::MetalViewSurface::new(
                    &mut self.renderer,
                    pass.without_drawable(),
                );
                quad.render(&mut target, scene, camera);
                true
            }
            None => match self
                .renderer
                .render(scene, camera, &pass.without_drawable())
            {
                Ok(_) => true,
                Err(e) => {
                    log::error!("Metal render: {e}");
                    false
                }
            },
        };
        let presented = match overlay {
            Some(overlay) => self.renderer.draw_overlay(overlay, &pass),
            None => self.renderer.present(&pass),
        };
        if let Err(e) = presented {
            log::error!("Metal present: {e}");
        }
        drawn
    }
}
