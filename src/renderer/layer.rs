//! Rendering into a `CAMetalLayer` someone else owns: a layer added to a
//! UIKit or AppKit view (over a web view, in a toolbar, wherever), drawn from
//! any thread.
//!
//! The view and its layer are the app's — creating them, adding them to a
//! view hierarchy and moving them about is main-thread UIKit/AppKit work this
//! module doesn't do. What it does needs no main thread: wgpu takes the layer
//! as a surface, and acquiring a drawable, rendering and presenting are all
//! thread-safe on Metal, so a render thread can draw at display rate while the
//! main thread stays free.
//!
//! Several layers share one [`LayerGpu`] (one device, so meshes, pipelines
//! and shaders compiled for one view serve the next): open it once, then a
//! [`LayerRenderer`] per layer.
//!
//! macOS, iOS and iPadOS (Mac Catalyst included), tvOS and visionOS. The
//! [`LayerGpu`] is `Send + Sync` — share it, by reference or in an `Arc` — and
//! each [`LayerRenderer`] is `Send`: make it wherever, move it to the thread
//! that draws it, hand it on to another later. Layers on one [`LayerGpu`]
//! draw in parallel; one resizing pauses the others' submissions for the
//! moment wgpu needs the queue empty. wgpu drains an autorelease
//! pool around every Metal call it makes, so a bare `std::thread` does not
//! leak (`tests/layer_threads.rs` checks, as it does for the Metal backend).
//!
//! ```ignore
//! let gpu = LayerGpu::new()?;
//! // `layer`: a CAMetalLayer, made and placed on the main thread.
//! let mut view = unsafe { LayerRenderer::new(&gpu, layer, 600, 400, 4)? };
//! loop { view.render(&mut scene, &camera); } // on any thread
//! ```

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Arc, RwLock};

use super::Renderer;
use crate::cameras::Camera;
use crate::scene::Scene;

/// The device every [`LayerRenderer`] draws with.
pub struct LayerGpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    /// `ManuallyDrop`, with `queue`, so `Drop` releases them inside an
    /// autorelease pool: the last reference purges the device's resource
    /// pool, which autoreleases as it goes.
    device: ManuallyDrop<Arc<wgpu::Device>>,
    queue: ManuallyDrop<Arc<wgpu::Queue>>,
    /// Frames hold it shared; a surface reconfigure holds it alone. wgpu's
    /// `Surface::configure` waits for the device's queue to empty and fails
    /// — a panic, through the default error handler — if another thread
    /// submitted meanwhile, so a layer resizing while another draws needs
    /// the others to pause for it.
    frames: Arc<RwLock<()>>,
}

impl LayerGpu {
    /// The system's Metal device. Blocks briefly.
    pub fn new() -> Result<Self, String> {
        let _pool = Pool::new();
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no Metal adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("threers layer device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("wgpu request_device failed: {e:?}"))?;
        Ok(Self {
            instance,
            adapter,
            device: ManuallyDrop::new(Arc::new(device)),
            queue: ManuallyDrop::new(Arc::new(queue)),
            frames: Arc::new(RwLock::new(())),
        })
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

/// A [`Renderer`] presenting to one `CAMetalLayer`.
pub struct LayerRenderer {
    /// `ManuallyDrop`, with `renderer`, so `Drop` can release both inside an
    /// autorelease pool.
    surface: ManuallyDrop<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    /// Kept to rebuild the surface when Metal reports it lost.
    instance: wgpu::Instance,
    /// The app's layer, not retained (see [`LayerRenderer::new`]). An address,
    /// so the renderer stays `Send`: wgpu's Metal surface already holds and
    /// uses the same layer from whichever thread renders.
    layer: usize,
    frames: Arc<RwLock<()>>,
    /// The last configure failed; try again before the next frame.
    unconfigured: bool,
    renderer: ManuallyDrop<Renderer>,
}

impl Drop for LayerGpu {
    fn drop(&mut self) {
        let _pool = Pool::new();
        // SAFETY: dropped once, here, and never touched again.
        unsafe {
            ManuallyDrop::drop(&mut self.queue);
            ManuallyDrop::drop(&mut self.device);
        }
    }
}

impl Drop for LayerRenderer {
    fn drop(&mut self) {
        // Field drops run after this returns, outside any pool; release the
        // GPU objects here, inside one.
        let _pool = Pool::new();
        // SAFETY: dropped once, here, and never touched again.
        unsafe {
            ManuallyDrop::drop(&mut self.renderer);
            ManuallyDrop::drop(&mut self.surface);
        }
    }
}

// Compile-time: what this module promises about threads.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<LayerGpu>();
    send::<LayerRenderer>();
};

/// An Objective-C autorelease pool for the length of a call.
///
/// wgpu drains one around each Metal call it makes, but not around the
/// releases: a frame that frees its predecessor's textures and buffers lets
/// the driver autorelease the device on the way, and on a `std::thread` with
/// no pool that is an object leaked per resource, every frame.
pub(crate) struct Pool(*mut c_void);

impl Pool {
    pub(crate) fn new() -> Self {
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

/// Whether `layer` is a `CAMetalLayer` — wgpu asserts it, which in an app is a
/// crash; this turns it into an error first.
fn is_metal_layer(layer: *mut c_void) -> bool {
    use std::ffi::c_char;
    /// Laid out as `metal::objc::Object`, so the crate's declarations agree.
    #[repr(C)]
    struct Object {
        _private: [u8; 0],
    }
    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut Object;
        fn sel_registerName(name: *const c_char) -> *const c_void;
        fn objc_msgSend();
    }
    // QuartzCore registers CAMetalLayer; wgpu links it already.
    if layer.is_null() {
        return false;
    }
    unsafe {
        let cls = objc_getClass(c"CAMetalLayer".as_ptr());
        if cls.is_null() {
            return false;
        }
        let is_kind: unsafe extern "C" fn(*mut c_void, *const c_void, *mut Object) -> i8 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        is_kind(layer, sel_registerName(c"isKindOfClass:".as_ptr()), cls) != 0
    }
}

impl LayerRenderer {
    /// Draw into `layer` at `width × height` drawable pixels (the layer's
    /// bounds × its `contentsScale`), `msaa` 1 (off) or 4.
    ///
    /// Callable on any thread. A layer that is not a `CAMetalLayer` — a view's
    /// default `CALayer`, say, when its `layerClass` was not overridden — is
    /// an error rather than wgpu's assertion.
    ///
    /// # Safety
    /// `layer` must be null or a live `CALayer` that outlives the renderer.
    pub unsafe fn new(gpu: &LayerGpu, layer: *mut c_void, width: u32, height: u32, msaa: u32) -> Result<Self, String> {
        let _pool = Pool::new();
        if !is_metal_layer(layer) {
            return Err("not a CAMetalLayer — on iOS, override the view's `layerClass` \
                        to return `CAMetalLayer.self`"
                .into());
        }
        let (width, height) = (width.max(1), height.max(1));
        let surface = gpu
            .instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer))
            .map_err(|e| format!("layer surface: {e:?}"))?;
        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let alpha_mode = [wgpu::CompositeAlphaMode::Opaque, wgpu::CompositeAlphaMode::Auto]
            .into_iter()
            .find(|m| caps.alpha_modes.contains(m))
            .unwrap_or(caps.alpha_modes[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            // Display-paced: a render thread drawing several layers blocks
            // on each one's next drawable, never spins.
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        let mut renderer = Renderer::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), format, width, height);
        renderer.set_msaa(msaa);
        let mut this = Self {
            surface: ManuallyDrop::new(surface),
            config,
            device: Arc::clone(&gpu.device),
            instance: gpu.instance.clone(),
            layer: layer as usize,
            frames: gpu.frames.clone(),
            unconfigured: true,
            renderer: ManuallyDrop::new(renderer),
        };
        this.configure();
        Ok(this)
    }

    /// New drawable size (pixels); no-op when unchanged.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        let _pool = Pool::new();
        self.config.width = width;
        self.config.height = height;
        self.configure();
        self.renderer.resize(width, height);
    }

    /// Apply `config` to the surface, with every other layer on the device
    /// between frames. A failure is caught rather than left to wgpu's panic,
    /// and retried before the next frame.
    fn configure(&mut self) -> bool {
        let _alone = self.frames.write().unwrap_or_else(|e| e.into_inner());
        // Error scopes are per thread in wgpu, so this catches this
        // configure's error and nobody else's.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.surface.configure(&self.device, &self.config);
        self.unconfigured = pollster::block_on(scope.pop()).is_some();
        !self.unconfigured
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Render and present one frame. `false`: no drawable this time, and the
    /// next call draws.
    ///
    /// Nothing is drawn while the layer is off screen or every drawable is in
    /// flight (`Timeout`, `Occluded`: skip and try again); an outdated
    /// configuration is reapplied, and a surface Metal reports lost is rebuilt
    /// on the same layer.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        use wgpu::CurrentSurfaceTexture as Acquired;
        let _pool = Pool::new();
        if self.unconfigured && !self.configure() {
            return false;
        }
        match self.surface.get_current_texture() {
            Acquired::Success(frame) => {
                self.draw(scene, camera, frame);
                true
            }
            Acquired::Suboptimal(frame) => {
                // Draw this one; the reconfigure fits the next to the layer.
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
                let target = wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(self.layer as *mut c_void);
                // SAFETY: the layer is the one `new` was given, which its
                // contract keeps alive for as long as this renderer.
                if let Ok(surface) = unsafe { self.instance.create_surface_unsafe(target) } {
                    *self.surface = surface;
                }
                self.configure();
                false
            }
            _ => false,
        }
    }

    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera, frame: wgpu::SurfaceTexture) {
        // Shared: layers draw in parallel, and only a reconfigure waits.
        let _frame = self.frames.read().unwrap_or_else(|e| e.into_inner());
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(scene, camera, &view, false);
        self.renderer.queue_arc().present(frame);
    }

    pub fn renderer(&mut self) -> &mut Renderer {
        &mut self.renderer
    }
}
