//! The browser shell: a canvas, a device, and `requestAnimationFrame`.
//!
//! The only web-specific part of the fluid. It does what `examples/fluid.rs`
//! does with winit — make a surface, size it, feed the pointer in, pump frames
//! — with the platform's own primitives, over the same [`FluidSim`].
//!
//! Backend selection goes through [`crate::wasm::open_browser_gpu`], which
//! prefers WebGPU and falls back to WebGL2 where the `webgl` feature is built
//! in. The fluid is one of the few things in this crate that is *complete* on
//! WebGL2: it is fullscreen fragment passes over float targets, with no compute
//! stage and no storage buffers, so nothing has to be left out there.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::{
    quality, Advection, FluidConfig, FluidFormats, FluidSim, PressureSolver, QUALITY_LEVELS,
};

/// Everything the frame callback needs. The browser gives no place to put
/// state, so it lives behind one handle the closures can share.
struct App {
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sim: FluidSim,
    /// Page time of the last frame, for the step.
    last: f64,
    /// Frames since `fps_since`, and the last rate computed from them.
    frames: u32,
    fps_since: f64,
    fps: f32,
}

/// A running simulation, for the page to steer.
#[wasm_bindgen]
pub struct Fluid {
    app: Rc<RefCell<App>>,
}

fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

fn dpr() -> f64 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .filter(|r| *r > 0.0)
        .unwrap_or(1.0)
}

/// Start the fluid on `canvas`. Resolves once the first frame is scheduled;
/// the loop then runs until the page goes away.
///
/// `quality_name` is one of [`QUALITY_LEVELS`]; `None` is `high`, the
/// original's settings.
#[wasm_bindgen(js_name = fluidStart)]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    quality_name: Option<String>,
) -> Result<Fluid, JsValue> {
    console_error_panic_hook::set_once();

    let level = quality_name
        .as_deref()
        .and_then(quality)
        .unwrap_or(&QUALITY_LEVELS[2]);

    let (_instance, surface, adapter) = crate::wasm::open_browser_gpu(&canvas).await?;
    let formats = FluidFormats::for_adapter(&adapter).ok_or_else(|| {
        JsValue::from_str(
            "this browser cannot render to a filterable half-float texture \
             (WebGL2 needs EXT_color_buffer_float)",
        )
    })?;

    // Downlevel WebGL2 defaults are enough — the simulation is fullscreen
    // draws over five textures — but the *texture size* has to come from the
    // adapter, because the dye grid at `ultra` is larger than the 2048 floor.
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers fluid"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("no device: {e}")))?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let (width, height) = canvas_size(&canvas);
    canvas.set_width(width);
    canvas.set_height(height);

    let caps = surface.get_capabilities(&adapter);
    // Prefer a non-sRGB surface: the display pass hands back display-referred
    // values, which is what the original wrote into a plain RGBA8 canvas. An
    // sRGB surface works too — the shader puts the inverse transfer in first.
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);

    let mut sim = FluidSim::with_formats(
        device.clone(),
        queue.clone(),
        width,
        height,
        FluidConfig {
            // A page that opens on an empty black canvas reads as a failure to
            // start, so it opens on a fluid that is already moving.
            seed: (now() * 1000.0) as u64 | 1,
            ..FluidConfig::default().with_quality(level)
        },
        formats,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    sim.multiple_splats(20);

    let app = Rc::new(RefCell::new(App {
        canvas: canvas.clone(),
        surface,
        config,
        device,
        queue,
        sim,
        last: now(),
        frames: 0,
        fps_since: now(),
        fps: 0.0,
    }));

    install_input(&canvas, &app)?;
    start_frames(app.clone());
    Ok(Fluid { app })
}

#[wasm_bindgen]
impl Fluid {
    /// Set one numeric or boolean knob by its Rust field name. Booleans take
    /// 0 or 1. Unknown names are an error rather than a silent no-op, which is
    /// the failure mode a stringly-typed setter otherwise has.
    pub fn set(&self, name: &str, value: f64) -> Result<(), JsValue> {
        let mut app = self.app.borrow_mut();
        let mut c = *app.sim.config();
        let on = value != 0.0;
        match name {
            "sim_resolution" => c.sim_resolution = value as u32,
            "dye_resolution" => c.dye_resolution = value as u32,
            "density_dissipation" => c.density_dissipation = value as f32,
            "velocity_dissipation" => c.velocity_dissipation = value as f32,
            "pressure" => c.pressure = value as f32,
            "pressure_iterations" => c.pressure_iterations = (value as u32).max(1),
            // 0 = the original's Jacobi sweeps, 1 = a multigrid V-cycle.
            // Second-order advection: sharper dye for an extra pass.
            "maccormack" => {
                c.advection = if on {
                    Advection::MacCormack
                } else {
                    Advection::SemiLagrangian
                }
            }
            "multigrid" => {
                c.pressure_solver = if on {
                    PressureSolver::Multigrid
                } else {
                    PressureSolver::Jacobi
                }
            }
            "multigrid_cycles" => c.multigrid_cycles = (value as u32).clamp(1, 8),
            "gradient_scale" => c.gradient_scale = value as f32,
            "curl" => c.curl = value as f32,
            "splat_radius" => c.splat_radius = value as f32,
            "splat_force" => c.splat_force = value as f32,
            "shading" => c.shading = on,
            "colorful" => c.colorful = on,
            "color_update_speed" => c.color_update_speed = value as f32,
            "transparent" => c.transparent = on,
            "bloom" => c.bloom = on,
            "bloom_iterations" => c.bloom_iterations = value as u32,
            "bloom_intensity" => c.bloom_intensity = value as f32,
            "bloom_threshold" => c.bloom_threshold = value as f32,
            "sunrays" => c.sunrays = on,
            "sunrays_weight" => c.sunrays_weight = value as f32,
            "paused" => {
                app.sim.set_paused(on);
                return Ok(());
            }
            other => return Err(JsValue::from_str(&format!("unknown setting '{other}'"))),
        }
        app.sim.set_config(c);
        Ok(())
    }

    /// Set the background colour, as three channels in 0..1.
    #[wasm_bindgen(js_name = setBackColor)]
    pub fn set_back_color(&self, r: f64, g: f64, b: f64) {
        let mut app = self.app.borrow_mut();
        let mut c = *app.sim.config();
        c.back_color = [r as f32, g as f32, b as f32];
        app.sim.set_config(c);
    }

    /// Switch quality level by name — `low`, `medium`, `high`, `ultra`,
    /// `extreme`. The fields are kept; only the grids are rebuilt.
    pub fn quality(&self, name: &str) -> Result<(), JsValue> {
        let level = quality(name).ok_or_else(|| {
            let names: Vec<_> = QUALITY_LEVELS.iter().map(|q| q.name).collect();
            JsValue::from_str(&format!(
                "unknown quality '{name}' (have {})",
                names.join(", ")
            ))
        })?;
        let mut app = self.app.borrow_mut();
        let c = app.sim.config().with_quality(level);
        app.sim.set_config(c);
        Ok(())
    }

    /// Every quality level's name, coarsest first, so a menu need not hardcode
    /// them.
    #[wasm_bindgen(js_name = qualityNames)]
    pub fn quality_names() -> Vec<String> {
        QUALITY_LEVELS.iter().map(|q| q.name.to_string()).collect()
    }

    /// A handful of splats, as the space bar throws.
    pub fn splats(&self, count: u32) {
        self.app.borrow_mut().sim.multiple_splats(count.min(200));
    }

    pub fn paused(&self) -> bool {
        self.app.borrow().sim.paused()
    }

    /// Frames per second, averaged over the last half second.
    pub fn fps(&self) -> f32 {
        self.app.borrow().fps
    }

    /// What the grids actually came out as, and what they cost — a quality
    /// level is a request, and the device has the last word.
    pub fn info(&self) -> String {
        let app = self.app.borrow();
        let ((dw, dh), (sw, sh)) = app.sim.grid_sizes();
        format!(
            "{{\"dye\":[{dw},{dh}],\"sim\":[{sw},{sh}],\"sweeps\":{},\"bytes\":{},\"formats\":\"{:?}\"}}",
            app.sim.config().pressure_iterations,
            app.sim.memory_bytes(),
            app.sim.formats(),
        )
    }
}

/// CSS pixels times the device ratio, which is what the backing store has to
/// be for the picture not to be soft on a retina display.
fn canvas_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    let ratio = dpr();
    let w = (canvas.client_width() as f64 * ratio).floor() as u32;
    let h = (canvas.client_height() as f64 * ratio).floor() as u32;
    (w.max(1), h.max(1))
}

/// Pointer events, which unify mouse and touch and hand out an id per finger —
/// exactly the shape [`FluidSim::pointer_down`] wants.
fn install_input(
    canvas: &web_sys::HtmlCanvasElement,
    app: &Rc<RefCell<App>>,
) -> Result<(), JsValue> {
    let target: &web_sys::EventTarget = canvas.as_ref();

    let listen =
        |name: &str, f: Closure<dyn FnMut(web_sys::PointerEvent)>| -> Result<(), JsValue> {
            target.add_event_listener_with_callback(name, f.as_ref().unchecked_ref())?;
            f.forget();
            Ok(())
        };

    let a = app.clone();
    listen(
        "pointerdown",
        Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            let (x, y) = pointer_at(&e);
            a.borrow_mut().sim.pointer_down(e.pointer_id(), x, y);
            e.prevent_default();
        }),
    )?;

    let a = app.clone();
    listen(
        "pointermove",
        Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            let (x, y) = pointer_at(&e);
            a.borrow_mut().sim.pointer_move(e.pointer_id(), x, y);
        }),
    )?;

    for name in ["pointerup", "pointercancel", "pointerleave"] {
        let a = app.clone();
        listen(
            name,
            Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
                a.borrow_mut().sim.pointer_up(e.pointer_id());
            }),
        )?;
    }

    // Space and `p`, as the original binds them.
    if let Some(window) = web_sys::window() {
        let a = app.clone();
        let keys =
            Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
                match e.key().as_str() {
                    " " => {
                        a.borrow_mut().sim.multiple_splats(15);
                        e.prevent_default();
                    }
                    "p" | "P" => {
                        let mut app = a.borrow_mut();
                        let paused = !app.sim.paused();
                        app.sim.set_paused(paused);
                    }
                    _ => {}
                }
            });
        let target: &web_sys::EventTarget = window.as_ref();
        target.add_event_listener_with_callback("keydown", keys.as_ref().unchecked_ref())?;
        keys.forget();
    }
    Ok(())
}

/// Where the pointer is, in backing-store pixels — `offsetX` is in CSS pixels
/// and the simulation is sized in device ones.
fn pointer_at(e: &web_sys::PointerEvent) -> (f32, f32) {
    let ratio = dpr();
    (
        (e.offset_x() as f64 * ratio) as f32,
        (e.offset_y() as f64 * ratio) as f32,
    )
}

/// The rAF callback, held somewhere it can reschedule itself.
type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

fn start_frames(app: Rc<RefCell<App>>) {
    let holder: FrameCallback = Rc::new(RefCell::new(None));
    let next = holder.clone();
    *holder.borrow_mut() = Some(Closure::<dyn FnMut()>::new(move || {
        frame(&app);
        request_frame(next.borrow().as_ref());
    }));
    request_frame(holder.borrow().as_ref());
    // Leaked on purpose: the loop outlives this call, and there is nothing to
    // cancel it with — the page going away is what stops it.
    std::mem::forget(holder);
}

fn request_frame(callback: Option<&Closure<dyn FnMut()>>) {
    if let (Some(window), Some(cb)) = (web_sys::window(), callback) {
        let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
    }
}

fn frame(app: &Rc<RefCell<App>>) {
    let mut app = app.borrow_mut();

    // The canvas can be resized by CSS at any time, and the backing store does
    // not follow on its own.
    let (w, h) = canvas_size(&app.canvas);
    if (w, h) != (app.config.width, app.config.height) {
        app.canvas.set_width(w);
        app.canvas.set_height(h);
        app.config.width = w;
        app.config.height = h;
        let App {
            surface,
            config,
            device,
            ..
        } = &*app;
        surface.configure(device, config);
        app.sim.resize(w, h);
    }

    let now = now();
    app.frames += 1;
    if now - app.fps_since >= 500.0 {
        app.fps = app.frames as f32 * 1000.0 / (now - app.fps_since) as f32;
        app.frames = 0;
        app.fps_since = now;
    }
    // Clamped as the original clamps it: a frame that took a quarter of a
    // second — a background tab waking up — must not advance the fluid a
    // quarter of a second, or it jumps.
    let dt = (((now - app.last) / 1000.0) as f32).clamp(0.0, 1.0 / 60.0);
    app.last = now;
    app.sim.update(dt);

    match app.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
            app.sim.render(&frame.texture);
            app.queue.present(frame);
        }
        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
            let App {
                surface,
                config,
                device,
                ..
            } = &*app;
            surface.configure(device, config);
        }
        e => log::error!("surface error: {e:?}"),
    }
}
