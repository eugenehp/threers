//! Drawing the demo on the GPU in the browser: threers' UI pass, which draws
//! the whole UI in one instanced draw call (and boxes turned in 3D in true
//! perspective), on WebGPU where the browser has it and WebGL2 where not.
//!
//! The canvas is not touched until WebGPU has said whether it has an
//! adapter: a canvas gives out one kind of context for its life, so asking
//! for a WebGPU surface first would spoil it for WebGL2 (and for the 2D
//! canvas the page falls back to otherwise, on a fresh canvas).

use threers::renderer::UiPass;
use threers::xr::phone::PhoneView;
use threers::xr::web::XrSession;
use threers::xr::{XrEvent, XrEye, XrInput, XrSource};
use threers::wgpu;
use web_sys::HtmlCanvasElement;

/// The GPU, the canvas's surface, and the pass that draws the UI.
pub struct Gpu {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    canvas: HtmlCanvasElement,
    /// What the surface may do with alpha: an AR session draws see-through.
    alpha_modes: Vec<wgpu::CompositeAlphaMode>,
    pub pass: UiPass,
    /// threers' renderer for a headset's room, made when a session first
    /// needs it.
    xr: Option<threers::Renderer>,
    /// `WebGPU` or `WebGL2`, for the page to show.
    pub backend: &'static str,
}

impl Gpu {
    /// The GPU for `canvas`: WebGPU if the browser has an adapter (and
    /// `webgpu`), else WebGL2.
    pub async fn open(canvas: HtmlCanvasElement, webgpu: bool) -> Result<Gpu, String> {
        let adapter_options = |surface| wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        };
        // WebGPU: an adapter first, with no surface.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(d);
        let found = match webgpu {
            true => instance.request_adapter(&adapter_options(None)).await.map_err(|e| e.to_string()),
            false => Err("not asked for".to_string()),
        };
        let (instance, surface, adapter, backend) = match found {
            Ok(adapter) => {
                let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone())).map_err(|e| format!("webgpu surface: {e:?}"))?;
                (instance, surface, adapter, "WebGPU")
            }
            Err(e) => {
                // WebGL2: there the context is the canvas, so the surface
                // comes before the adapter.
                let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
                d.backends = wgpu::Backends::GL;
                let instance = wgpu::Instance::new(d);
                let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone())).map_err(|g| format!("webgpu: {e}; webgl2 surface: {g:?}"))?;
                let adapter = instance.request_adapter(&adapter_options(Some(&surface))).await.map_err(|g| format!("webgpu: {e}; webgl2: {g}"))?;
                (instance, surface, adapter, "WebGL2")
            }
        };
        let _ = instance;
        // What the adapter offers, not a floor: WebGPU rejects a pipeline
        // past a limit silently (a black canvas at full frame rate).
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("threers ui demo"), required_limits: adapter.limits(), ..Default::default() })
            .await
            .map_err(|e| format!("device: {e:?}"))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).or(caps.formats.first().copied()).ok_or("no surface format")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &config);
        let pass = UiPass::new(&device);
        // Shared with threers' renderer (the headset's room), which takes
        // `Arc`s; on the web nothing crosses threads.
        #[allow(clippy::arc_with_non_send_sync)]
        let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
        Ok(Gpu { device, queue, surface, config, canvas, alpha_modes: if caps.alpha_modes.is_empty() { vec![wgpu::CompositeAlphaMode::Auto] } else { caps.alpha_modes }, pass, xr: None, backend })
    }

    /// Draws `scene` (UI pixels, `scale` device pixels each) over `clear`
    /// (straight sRGB), the canvas resized to its backing size first.
    pub fn draw(&mut self, scene: &threers::ui::scene::Scene, scale: f32, clear: [f32; 4]) {
        let (w, h) = (self.canvas.width().max(1), self.canvas.height().max(1));
        if (w, h) != (self.config.width, self.config.height) {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Painted again only where the scene changed, into a picture kept
        // between frames, then copied out (a canvas's texture starts
        // undefined each frame, WebGL2's too).
        self.pass.draw_retained(&self.device, &self.queue, scene, scale, &view, (w, h), self.config.format, clear);
        self.queue.present(frame);
    }

    /// The canvas drawn on (a headset session copies it to the headset).
    pub fn canvas(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }

    /// One headset frame: the room, and the UI `ui_scene` on its panel
    /// (see [`crate::xr::UiPanel::draw`]), into the canvas, which the
    /// session has sized to the headset's framebuffer and copies to it after.
    pub fn draw_xr(&mut self, ui_scene: &threers::ui::scene::Scene, scale: f32, clear: [f32; 4], headset: &mut Headset) {
        let (w, h) = headset.session.fit_canvas();
        // See-through needs a surface whose alpha the page shows.
        // (wgpu lists only Opaque for a WebGPU canvas, but passes
        // PreMultiplied on: WebGPU's canvases take it.)
        let alpha = match headset.session.see_through() {
            true if self.backend == "WebGPU" => wgpu::CompositeAlphaMode::PreMultiplied,
            true => self.alpha_modes.iter().copied().find(|m| *m == wgpu::CompositeAlphaMode::PreMultiplied).unwrap_or(self.alpha_modes[0]),
            false => self.alpha_modes[0],
        };
        if (w, h, alpha) != (self.config.width, self.config.height, self.config.alpha_mode) {
            (self.config.width, self.config.height, self.config.alpha_mode) = (w, h, alpha);
            self.surface.configure(&self.device, &self.config);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let format = self.config.format;
        let renderer = self.xr.get_or_insert_with(|| threers::Renderer::new_lazy(self.device.clone(), self.queue.clone(), format, w, h));
        let mut gpu = threers::xr::Gpu { device: &self.device, queue: &self.queue, format, target: &view, renderer, source: &headset.session };
        headset.ui.draw(&mut gpu, &mut self.pass, ui_scene, scale, clear, &mut headset.scene, &headset.lens);
        self.queue.present(frame);
        headset.session.mark_drawn();
    }

    /// Back to the page after a session: an opaque surface again.
    pub fn leave_xr(&mut self) {
        if self.config.alpha_mode != self.alpha_modes[0] {
            self.config.alpha_mode = self.alpha_modes[0];
            self.surface.configure(&self.device, &self.config);
        }
    }
}

/// What the demo is shown through: a WebXR session (a headset, a phone in
/// AR), or a phone's look-around view where there is no immersive WebXR.
pub enum View {
    Xr(XrSession),
    Phone(PhoneView),
}

impl View {
    /// Sizes the canvas for a frame and returns its size.
    pub fn fit_canvas(&mut self) -> (u32, u32) {
        match self {
            View::Xr(s) => s.fit_canvas(),
            View::Phone(p) => p.fit_canvas(),
        }
    }
    pub fn mark_drawn(&self) {
        match self {
            View::Xr(s) => s.mark_drawn(),
            View::Phone(p) => p.mark_drawn(),
        }
    }
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        match self {
            View::Xr(s) => s.set_depth_range(near, far),
            View::Phone(p) => p.set_depth_range(near, far),
        }
    }
    pub fn end(&mut self) {
        match self {
            View::Xr(s) => drop(s.end()),
            View::Phone(p) => p.end(),
        }
    }
    pub fn on_end(&self, callback: Option<Box<dyn FnOnce()>>) {
        match self {
            View::Xr(s) => s.on_end(callback),
            View::Phone(p) => p.on_end(callback),
        }
    }
    /// Whether this page frame is the view's to draw: a WebXR session's
    /// own frame, or any page frame for the phone's.
    pub fn in_frame(&self) -> bool {
        match self {
            View::Xr(s) => s.in_frame(),
            View::Phone(p) => !p.ended(),
        }
    }
}

impl XrSource for View {
    fn head(&self) -> Option<[f32; 16]> {
        match self {
            View::Xr(s) => s.head(),
            View::Phone(p) => p.head(),
        }
    }
    fn eyes(&self, origin: &threers::Matrix4) -> Vec<XrEye> {
        match self {
            View::Xr(s) => s.eyes(origin),
            View::Phone(p) => p.eyes(origin),
        }
    }
    fn framebuffer_size(&self) -> (u32, u32) {
        match self {
            View::Xr(s) => s.framebuffer_size(),
            View::Phone(p) => p.framebuffer_size(),
        }
    }
    fn inputs(&self) -> Vec<XrInput> {
        match self {
            View::Xr(s) => s.inputs(),
            View::Phone(p) => p.inputs(),
        }
    }
    fn take_events(&mut self) -> Vec<XrEvent> {
        match self {
            View::Xr(s) => XrSource::take_events(s),
            View::Phone(p) => p.take_events(),
        }
    }
    fn see_through(&self) -> bool {
        match self {
            View::Xr(s) => XrSource::see_through(s),
            View::Phone(p) => p.see_through(),
        }
    }
}

/// The demo shown in a headset (or a phone): its room and the UI on a
/// panel in it.
pub struct Headset {
    pub session: View,
    pub scene: threers::Scene,
    pub room: crate::xr::Room,
    pub ui: crate::xr::UiPanel,
    pub lens: threers::PerspectiveCamera,
    /// The last frame's time, for the next one's `dt`.
    pub last: f64,
}

impl Headset {
    pub fn new(mut session: View) -> Self {
        let mut scene = threers::Scene::new();
        let room = crate::xr::Room::furnish(&mut scene);
        let lens = crate::xr::lens();
        session.set_depth_range(lens.near, lens.far);
        Headset { session, scene, room, ui: Default::default(), lens, last: 0.0 }
    }
}
