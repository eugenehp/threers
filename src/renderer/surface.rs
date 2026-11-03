//! Rendering into a native window or child view someone else owns — a Tauri
//! or Wry child window beside a web view, a tao or winit window, an HWND or
//! an X11/Wayland surface handed over raw — on one device shared by every
//! viewport, from any thread.
//!
//! [`SurfaceGpu`] is the device: open it once, and every [`SurfaceRenderer`]
//! on it shares meshes, shaders and pipelines, so the second viewport costs
//! no compilation the first did not. Each renderer is `Send`: make it where
//! the window is, then move it to the thread that draws.
//!
//! ```ignore
//! // The cache goes anywhere: a directory (`PathBuf`), a `FileStore`, a
//! // `MemoryStore`, a `CallbackStore` into the app's own storage.
//! let gpu = SurfaceGpu::new()?.with_pipeline_cache(app_cache_dir.join("gpu"));
//! // On the main thread (macOS requires it): `window` is anything with
//! // raw-window-handle 0.6 window and display handles — a `tauri::Window`,
//! // a tao or winit window in an `Arc`.
//! let mut view = SurfaceRenderer::new(&gpu, window.clone(), w, h, SurfaceOptions::default().lazy(true))?;
//! view.renderer().warm_up().run();       // or on worker threads
//! gpu.save_pipeline_cache()?;
//! std::thread::spawn(move || loop { view.render(&mut scene, &camera); });
//! ```
//!
//! The device comes from [`GpuContext::headless`](crate::gpu::GpuContext),
//! with no display handle: right for Vulkan, Metal and DX12, which are what
//! desktop GPUs get. GL on Wayland needs one; open a
//! [`GpuContext`](crate::gpu::GpuContext) yourself and use
//! [`SurfaceGpu::from_context`].

use std::sync::{Arc, RwLock};

use super::pipeline_cache::{PipelineCache, PipelineCacheStore};
use super::{Renderer, RendererOptions};
use crate::cameras::Camera;
use crate::gpu::GpuContext;
use crate::scene::Scene;

/// The device every [`SurfaceRenderer`] draws with. `Send + Sync`.
pub struct SurfaceGpu {
    gpu: GpuContext,
    /// Frames hold it shared; a surface reconfigure holds it alone — wgpu's
    /// `Surface::configure` waits for the queue to empty and fails if another
    /// thread submits meanwhile (see `layer.rs`, which this follows).
    frames: Arc<RwLock<()>>,
    cache: Option<PipelineCache>,
}

impl SurfaceGpu {
    /// The system's high-performance adapter. Blocks briefly.
    pub fn new() -> Result<Self, String> {
        Ok(Self::from_context(GpuContext::headless_blocking()?))
    }

    /// A device you opened yourself — with a display handle, other limits or
    /// features. Ask for `Features::PIPELINE_CACHE` if you want
    /// [`with_pipeline_cache`](Self::with_pipeline_cache) to do anything
    /// (`GpuContext` asks for it wherever there is one).
    pub fn from_context(gpu: GpuContext) -> Self {
        Self { gpu, frames: Arc::new(RwLock::new(())), cache: None }
    }

    /// Build every viewport's pipelines through the cache `store` holds — a
    /// directory (`PathBuf`), [`FileStore`](super::FileStore),
    /// [`MemoryStore`](super::MemoryStore),
    /// [`CallbackStore`](super::CallbackStore) or your own
    /// [`PipelineCacheStore`]. Written on
    /// [`save_pipeline_cache`](Self::save_pipeline_cache). Nothing changes
    /// where the backend has no pipeline cache. Set it before making the
    /// viewports: each takes the cache when it is made.
    pub fn with_pipeline_cache(mut self, store: impl PipelineCacheStore + 'static) -> Self {
        self.cache = PipelineCache::open(&self.gpu.device, &self.gpu.adapter.get_info(), store);
        self
    }

    /// A [`PipelineCache`] opened yourself (it must be on this device).
    pub fn with_opened_pipeline_cache(mut self, cache: Option<PipelineCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Write the pipeline cache — after a warm-up, or on exit. `Ok(false)`
    /// with no cache, or nothing in it.
    pub fn save_pipeline_cache(&self) -> std::io::Result<bool> {
        match &self.cache {
            Some(cache) => cache.save(),
            None => Ok(false),
        }
    }

    /// The cache, when there is one: for its [`data`](PipelineCache::data),
    /// or to build pipelines of your own through it.
    pub fn pipeline_cache(&self) -> Option<&PipelineCache> {
        self.cache.as_ref()
    }

    pub fn context(&self) -> &GpuContext {
        &self.gpu
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.gpu.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.gpu.queue
    }
}

/// How a [`SurfaceRenderer`] draws.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SurfaceOptions {
    /// 1 (off) or 4.
    pub msaa: u32,
    /// Build pipelines on first use; see [`Renderer::new_lazy`] and
    /// [`Renderer::warm_up`].
    pub lazy: bool,
    /// `AutoVsync` by default: display-paced, so a render thread drawing
    /// several viewports blocks on each one's next image and never spins.
    pub present_mode: wgpu::PresentMode,
    /// Composite with what is behind the window (premultiplied alpha), where
    /// the platform offers it.
    pub transparent: bool,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self { msaa: 4, lazy: false, present_mode: wgpu::PresentMode::AutoVsync, transparent: false }
    }
}

impl SurfaceOptions {
    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = samples;
        self
    }

    pub fn lazy(mut self, lazy: bool) -> Self {
        self.lazy = lazy;
        self
    }

    pub fn present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    pub fn transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }
}

/// How to make the surface again when the platform reports it lost.
enum Target {
    Safe(Box<dyn Fn() -> wgpu::SurfaceTarget<'static> + Send>),
    Raw(RawHandles),
}

struct RawHandles {
    display: Option<wgpu::rwh::RawDisplayHandle>,
    window: wgpu::rwh::RawWindowHandle,
}

// SAFETY: the handles are only used to create a surface on the window
// `SurfaceRenderer::from_raw` was given, which its contract keeps alive; the
// raw pointers inside are what wgpu's own surface already holds and uses from
// whichever thread renders.
unsafe impl Send for RawHandles {}

/// A [`Renderer`] presenting to one window or view.
pub struct SurfaceRenderer {
    // Dropped first: the surface before the window it may own (`Target::Safe`).
    renderer: Renderer,
    surface: wgpu::Surface<'static>,
    target: Target,
    config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    frames: Arc<RwLock<()>>,
    /// The last configure failed; try again before the next frame.
    unconfigured: bool,
}

impl SurfaceRenderer {
    /// Draw into `window` at `width × height` physical pixels.
    ///
    /// `window` is anything with raw-window-handle 0.6 window and display
    /// handles that can be cloned — a `tauri::Window`, an `Arc` of a tao or
    /// winit window. The renderer keeps a clone, so the window outlives it.
    ///
    /// On macOS, call this on the main thread (wgpu reads the view's layer);
    /// the renderer can then move to any thread.
    pub fn new<W>(gpu: &SurfaceGpu, window: W, width: u32, height: u32, options: SurfaceOptions) -> Result<Self, String>
    where
        W: wgpu::WindowHandle + wgpu::rwh::HasDisplayHandle + Clone + 'static,
    {
        let make = move || wgpu::SurfaceTarget::from(window.clone());
        let surface = gpu.gpu.instance.create_surface(make()).map_err(|e| format!("window surface: {e}"))?;
        Self::finish(gpu, surface, Target::Safe(Box::new(make)), width, height, options)
    }

    /// [`new`](Self::new) from raw handles — what a framework that does not
    /// expose its window type, or a C host, hands over.
    ///
    /// # Safety
    /// The handles must be valid and the window they name must outlive the
    /// renderer. On macOS, call this on the main thread.
    pub unsafe fn from_raw(
        gpu: &SurfaceGpu,
        display: Option<wgpu::rwh::RawDisplayHandle>,
        window: wgpu::rwh::RawWindowHandle,
        width: u32,
        height: u32,
        options: SurfaceOptions,
    ) -> Result<Self, String> {
        let target = wgpu::SurfaceTargetUnsafe::RawHandle { raw_display_handle: display, raw_window_handle: window };
        // SAFETY: the caller's contract.
        let surface =
            unsafe { gpu.gpu.instance.create_surface_unsafe(target) }.map_err(|e| format!("window surface: {e}"))?;
        Self::finish(gpu, surface, Target::Raw(RawHandles { display, window }), width, height, options)
    }

    fn finish(
        gpu: &SurfaceGpu,
        surface: wgpu::Surface<'static>,
        target: Target,
        width: u32,
        height: u32,
        options: SurfaceOptions,
    ) -> Result<Self, String> {
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]
        let _pool = super::layer::Pool::new();
        let (width, height) = (width.max(1), height.max(1));
        let caps = surface.get_capabilities(&gpu.gpu.adapter);
        if caps.formats.is_empty() {
            return Err(format!(
                "the adapter ({}) cannot present to this window — open a GpuContext with a \
                 compatible surface and use SurfaceGpu::from_context",
                gpu.gpu.adapter.get_info().name
            ));
        }
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let alpha_preference: &[wgpu::CompositeAlphaMode] = if options.transparent {
            &[wgpu::CompositeAlphaMode::PreMultiplied, wgpu::CompositeAlphaMode::PostMultiplied]
        } else {
            &[wgpu::CompositeAlphaMode::Opaque]
        };
        let alpha_mode = alpha_preference
            .iter()
            .chain(&[wgpu::CompositeAlphaMode::Auto])
            .copied()
            .find(|m| caps.alpha_modes.contains(m))
            .unwrap_or(caps.alpha_modes[0]);
        let present_mode = if caps.present_modes.contains(&options.present_mode)
            || matches!(options.present_mode, wgpu::PresentMode::AutoVsync | wgpu::PresentMode::AutoNoVsync)
        {
            options.present_mode
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        let renderer_options = RendererOptions::default()
            .lazy(options.lazy)
            .pipeline_cache(&gpu.cache);
        let mut renderer = Renderer::with_options(
            Arc::clone(&gpu.gpu.device),
            Arc::clone(&gpu.gpu.queue),
            format,
            width,
            height,
            renderer_options,
        );
        renderer.set_msaa(options.msaa);
        let mut this = Self {
            renderer,
            surface,
            target,
            config,
            gpu: GpuContext {
                instance: gpu.gpu.instance.clone(),
                adapter: gpu.gpu.adapter.clone(),
                device: Arc::clone(&gpu.gpu.device),
                queue: Arc::clone(&gpu.gpu.queue),
            },
            frames: gpu.frames.clone(),
            unconfigured: true,
        };
        this.configure();
        Ok(this)
    }

    /// New size in physical pixels; no-op when unchanged.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]
        let _pool = super::layer::Pool::new();
        self.config.width = width;
        self.config.height = height;
        self.configure();
        self.renderer.resize(width, height);
    }

    /// Apply `config`, with every other viewport on the device between
    /// frames. A failure is caught rather than left to wgpu's panic, and
    /// retried before the next frame.
    fn configure(&mut self) -> bool {
        let _alone = self.frames.write().unwrap_or_else(|e| e.into_inner());
        // Error scopes are per thread in wgpu, so this catches this
        // configure's error and nobody else's.
        let scope = self.gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.surface.configure(&self.gpu.device, &self.config);
        self.unconfigured = pollster::block_on(scope.pop()).is_some();
        !self.unconfigured
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// The surface's format, which the renderer draws in.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Render and present one frame. `false`: no image this time (minimised,
    /// occluded, every image in flight, or the surface was just remade), and
    /// the next call draws.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        use wgpu::CurrentSurfaceTexture as Acquired;
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]
        let _pool = super::layer::Pool::new();
        if self.unconfigured && !self.configure() {
            return false;
        }
        match self.surface.get_current_texture() {
            Acquired::Success(frame) => {
                self.draw(scene, camera, frame);
                true
            }
            Acquired::Suboptimal(frame) => {
                self.draw(scene, camera, frame);
                self.unconfigured = true;
                true
            }
            Acquired::Timeout | Acquired::Occluded => false,
            Acquired::Outdated => {
                self.configure();
                false
            }
            Acquired::Lost => {
                self.remake();
                self.configure();
                false
            }
            _ => false,
        }
    }

    /// A new surface on the same window. Not on Apple platforms, where it
    /// has to happen on the main thread and this may be a render thread; a
    /// reconfigure is what Metal needs there.
    fn remake(&mut self) {
        if cfg!(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos")) {
            return;
        }
        let surface = match &self.target {
            Target::Safe(make) => self.gpu.instance.create_surface(make()),
            Target::Raw(raw) => {
                let target = wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: raw.display,
                    raw_window_handle: raw.window,
                };
                // SAFETY: the handles `from_raw` was given, which its
                // contract keeps valid for as long as this renderer.
                unsafe { self.gpu.instance.create_surface_unsafe(target) }
            }
        };
        if let Ok(surface) = surface {
            self.surface = surface;
        }
    }

    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera, frame: wgpu::SurfaceTexture) {
        // Shared: viewports draw in parallel, and only a reconfigure waits.
        let _frame = self.frames.read().unwrap_or_else(|e| e.into_inner());
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(scene, camera, &view, false);
        self.gpu.queue.present(frame);
    }

    /// The renderer, for its settings, [`warm_up`](Renderer::warm_up) and
    /// [`warm_scene`](Renderer::warm_scene).
    pub fn renderer(&mut self) -> &mut Renderer {
        &mut self.renderer
    }
}

// Compile-time: what this module promises about threads.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<SurfaceGpu>();
    send::<SurfaceRenderer>();
};
