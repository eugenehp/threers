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
    quality, quality3d, Advection, FluidConfig, FluidConfig3d, FluidFormats, FluidSim, FluidSim3d,
    PressureSolver, QUALITY_LEVELS, QUALITY_LEVELS_3D,
};
use crate::cameras::Camera as _;
use crate::controls::{OrbitControls, PointerEvent as OrbitEvent};
use crate::{PerspectiveCamera, Vector3};

/// Everything the frame callback needs. The browser gives no place to put
/// state, so it lives behind one handle the closures can share.
/// Which fluid is running. One canvas, one device, one frame loop; switching
/// builds the other simulation and drops this one.
enum Which {
    Flat(Box<FluidSim>),
    Volume(Box<Volume3d>),
}

/// The 3D fluid and what it takes to look at it.
struct Volume3d {
    sim: FluidSim3d,
    camera: PerspectiveCamera,
    controls: OrbitControls,
    /// Seconds of simulated time, so the source can stop emitting.
    elapsed: f32,
}

struct App {
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    formats: FluidFormats,
    which: Which,
    /// Drag state for the 3D camera.
    dragging: bool,
    /// Whether this drag pushes the fluid rather than turning the camera.
    /// Held for the whole drag from whether ctrl was down when it started,
    /// so letting go of the key mid-drag does not hand the gesture over to
    /// the camera halfway through.
    pushing: bool,
    /// Pushes without a modifier, for touch — where there is no ctrl key to
    /// hold and a drag has to mean one thing or the other. The panel offers
    /// it as a switch; `pushing` is this or ctrl.
    push_mode: bool,
    /// Pushes so far, for the hue.
    pushes: u32,
    last_pointer: Option<(f32, f32)>,
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

impl App {
    fn flat(&mut self) -> Option<&mut FluidSim> {
        match &mut self.which {
            Which::Flat(s) => Some(s),
            Which::Volume(_) => None,
        }
    }

    fn volume(&mut self) -> Option<&mut Volume3d> {
        match &mut self.which {
            Which::Volume(v) => Some(v),
            Which::Flat(_) => None,
        }
    }

    /// Build the other simulation on the same device.
    fn set_mode(&mut self, three: bool) {
        let (w, h) = (self.config.width, self.config.height);
        match (&self.which, three) {
            (Which::Flat(_), false) | (Which::Volume(_), true) => return,
            _ => {}
        }
        if three {
            let Ok(mut sim) = FluidSim3d::with_formats(
                self.device.clone(),
                self.queue.clone(),
                FluidConfig3d {
                    // Well under the native default. A browser may be on an
                    // integrated GPU or, with the WebGL2 fallback, on a
                    // software rasteriser — and a volume costs the cube of
                    // its resolution times the march steps, so guessing high
                    // is the difference between smooth and a slideshow. The
                    // panel raises it for anyone whose machine can take it.
                    resolution: 40,
                    dye_resolution: 80,
                    march_steps: 64,
                    buoyancy: 46.0,
                    curl: 26.0,
                    density_dissipation: 0.2,
                    emission: 1.3,
                    absorption: 34.0,
                    seed: (now() * 1000.0) as u64 | 1,
                    ..FluidConfig3d::default()
                },
                self.formats,
            ) else {
                return;
            };
            sim.multiple_splats(4);
            let mut camera = PerspectiveCamera::new(40.0, w as f32 / h.max(1) as f32, 0.05, 20.0);
            camera.position = Vector3::new(1.1, 0.2, 1.25);
            camera.look_at(Vector3::new(0.0, -0.02, 0.0));
            let mut controls = OrbitControls::new(&camera);
            controls.min_distance = 0.7;
            controls.max_distance = 6.0;
            self.which = Which::Volume(Box::new(Volume3d {
                sim,
                camera,
                controls,
                elapsed: 0.0,
            }));
        } else {
            let Ok(mut sim) = FluidSim::with_formats(
                self.device.clone(),
                self.queue.clone(),
                w,
                h,
                FluidConfig {
                    seed: (now() * 1000.0) as u64 | 1,
                    ..FluidConfig::default()
                },
                self.formats,
            ) else {
                return;
            };
            sim.multiple_splats(20);
            self.which = Which::Flat(Box::new(sim));
        }
    }
}

fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// How many device pixels to a CSS pixel, capped at two.
///
/// A phone reports three, and the fluid is a per-pixel cost all the way
/// through — nine device pixels a CSS pixel for the ray march is most of a
/// frame for a sharpness nobody can resolve at arm's length. A Mac's Retina
/// display reports exactly two, so the cap costs desktop nothing.
fn dpr() -> f64 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .filter(|r| *r > 0.0)
        .unwrap_or(1.0)
        .min(2.0)
}

/// Start the fluid on `canvas`. Resolves once the first frame is scheduled;
/// the loop then runs until the page goes away.
///
/// `quality_name` is one of [`QUALITY_LEVELS`]; `None` is `high`, the
/// original's settings.
/// One query parameter off the page's own URL, for the knobs that have to be
/// set before anything exists to call a setter on.
fn query_param(name: &str) -> Option<String> {
    // Off `Document::url` rather than `Location::search`, which would be
    // another `web-sys` feature for the same string.
    let url = web_sys::window()?.document()?.url().ok()?;
    let query = url.split_once('?')?.1;
    let name = format!("{name}=");
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix(&name[..]).map(str::to_owned))
}

#[wasm_bindgen(js_name = fluidStart)]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    quality_name: Option<String>,
) -> Result<Fluid, JsValue> {
    console_error_panic_hook::set_once();
    crate::wasm::install_console_logger();
    // `?log=info` (or debug, trace) raises what reaches the console. Without
    // it wgpu is silent in a browser, which is how a rejected pipeline comes
    // to look like a slow start.
    if let Some(level) = query_param("log").and_then(|v| v.parse().ok()) {
        log::set_max_level(level);
    }

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
    crate::gpu::report_uncaptured_errors(&device, "threers fluid");
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
    // Which backend, which formats, which surface — the first question in
    // every "it is blank in my browser" report, and the one the page cannot
    // answer for itself. At `info`, so `?log=info` shows it and a normal
    // load stays quiet.
    {
        let i = adapter.get_info();
        log::info!(
            "fluid: {:?} adapter {:?}; surface {format:?} of {:?}, alpha {:?}; {formats:?}",
            i.backend,
            i.name,
            caps.formats,
            config.alpha_mode,
        );
    }
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
        formats,
        which: Which::Flat(Box::new(sim)),
        dragging: false,
        pushing: false,
        push_mode: false,
        pushes: 0,
        last_pointer: None,
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
    /// Switch between the two-dimensional fluid and the volume. The other
    /// simulation is built on the same device; this one is dropped.
    #[wasm_bindgen(js_name = setMode)]
    pub fn set_mode(&self, mode: &str) -> Result<(), JsValue> {
        let three = match mode {
            "3d" | "volume" => true,
            "2d" | "flat" => false,
            other => return Err(JsValue::from_str(&format!("unknown mode '{other}'"))),
        };
        self.app.borrow_mut().set_mode(three);
        Ok(())
    }

    /// `"2d"` or `"3d"`.
    pub fn mode(&self) -> String {
        match self.app.borrow().which {
            Which::Flat(_) => "2d".into(),
            Which::Volume(_) => "3d".into(),
        }
    }

    /// Set one numeric or boolean knob by its Rust field name. Booleans take
    /// 0 or 1. Unknown names are an error rather than a silent no-op, which
    /// is the failure mode a stringly-typed setter otherwise has — and the
    /// one this crate has just spent a round removing elsewhere.
    ///
    /// Each mode answers to its own names; a name the current mode does not
    /// have is accepted and ignored, so a panel can push its whole state at
    /// either without knowing which is running.
    pub fn set(&self, name: &str, value: f64) -> Result<(), JsValue> {
        let mut app = self.app.borrow_mut();
        let on = value != 0.0;
        // Not a simulation setting but a gesture one: what a plain drag
        // means, which is only a question in the volume and only on a
        // pointer with no ctrl key to hold.
        if name == "push" {
            app.push_mode = on;
            return Ok(());
        }
        match &mut app.which {
            Which::Flat(sim) => {
                let mut c = *sim.config();
                match name {
                    "sim_resolution" => c.sim_resolution = value as u32,
                    "dye_resolution" => c.dye_resolution = value as u32,
                    "density_dissipation" => c.density_dissipation = value as f32,
                    "velocity_dissipation" => c.velocity_dissipation = value as f32,
                    "pressure" => c.pressure = value as f32,
                    "pressure_iterations" => c.pressure_iterations = (value as u32).max(1),
                    "speed" => c.time_scale = (value as f32).clamp(0.0, 8.0),
                    "show_fps" => c.show_fps = on,
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
                    "paused" => {
                        sim.set_paused(on);
                        return Ok(());
                    }
                    _ if is_volume_knob(name) => return Ok(()),
                    other => return Err(JsValue::from_str(&format!("unknown setting '{other}'"))),
                }
                sim.set_config(c);
            }
            Which::Volume(v) => {
                let mut c = *v.sim.config();
                match name {
                    // Voxels a side. Both rebuild the volume, which throws
                    // the fluid away — see `FluidSim3d::set_resolution`.
                    "resolution" => c.resolution = (value as u32).clamp(8, 256),
                    "dye_resolution" => c.dye_resolution = (value as u32).clamp(8, 512),
                    "density_dissipation" => c.density_dissipation = value as f32,
                    "velocity_dissipation" => c.velocity_dissipation = value as f32,
                    "pressure" => c.pressure = value as f32,
                    "pressure_iterations" => c.pressure_iterations = (value as u32).max(1),
                    "speed" => c.time_scale = (value as f32).clamp(0.0, 8.0),
                    "gradient_scale" => c.gradient_scale = value as f32,
                    "curl" => c.curl = value as f32,
                    "splat_radius" => c.splat_radius = value as f32,
                    "splat_force" => c.splat_force = value as f32,
                    "buoyancy" => c.buoyancy = value as f32,
                    // Which way buoyancy pushes. Three knobs rather than a
                    // vector because `set` takes one number.
                    "buoyancy_x" => c.buoyancy_axis[0] = value as f32,
                    "buoyancy_y" => c.buoyancy_axis[1] = value as f32,
                    "buoyancy_z" => c.buoyancy_axis[2] = value as f32,
                    "march_steps" => c.march_steps = (value as u32).clamp(8, 512),
                    "absorption" => c.absorption = value as f32,
                    "emission" => c.emission = value as f32,
                    "exposure" => c.exposure = value as f32,
                    "transparent" => c.transparent = on,
                    // The optimisations, switchable while it runs so their
                    // cost and their effect can both be seen. See
                    // `FluidConfig3d` for what each one measured.
                    "accumulate" => c.temporal_samples = (value as u32).clamp(1, 32),
                    "reject" => c.temporal_clip = (value as f32).clamp(0.0, 8.0),
                    "show_fps" => c.show_fps = on,
                    "empty_space_skip" => c.empty_space_skip = on,
                    "cache_bindings" => c.cache_bindings = on,
                    "compact_dye" => c.compact_dye = on,
                    "multigrid" => {
                        c.pressure_solver = if on {
                            PressureSolver::Multigrid
                        } else {
                            PressureSolver::Jacobi
                        }
                    }
                    "multigrid_cycles" => c.multigrid_cycles = (value as u32).clamp(1, 8),
                    "paused" => {
                        v.sim.set_paused(on);
                        return Ok(());
                    }
                    _ if is_flat_knob(name) => return Ok(()),
                    other => return Err(JsValue::from_str(&format!("unknown setting '{other}'"))),
                }
                v.sim.set_config(c);
            }
        }
        Ok(())
    }

    /// Set the background, as three channels in 0..1.
    #[wasm_bindgen(js_name = setBackColor)]
    pub fn set_back_color(&self, r: f64, g: f64, b: f64) {
        let mut app = self.app.borrow_mut();
        let rgb = [r as f32, g as f32, b as f32];
        match &mut app.which {
            Which::Flat(sim) => {
                let mut c = *sim.config();
                c.back_color = rgb;
                sim.set_config(c);
            }
            Which::Volume(v) => {
                let mut c = *v.sim.config();
                c.background = rgb;
                v.sim.set_config(c);
            }
        }
    }

    /// Switch quality level — `low`, `medium`, `high`, `ultra`, `extreme`.
    ///
    /// The names are shared, the numbers behind them are not: a volume's
    /// level sets voxels a side and how many samples a ray takes, and costs
    /// the cube of the first. See `QUALITY_LEVELS_3D`. Either way this
    /// reallocates the grids, and in three dimensions that empties them.
    pub fn quality(&self, name: &str) -> Result<(), JsValue> {
        let mut app = self.app.borrow_mut();
        let known = |names: Vec<&str>| {
            JsValue::from_str(&format!(
                "unknown quality '{name}' (have {})",
                names.join(", ")
            ))
        };
        match &mut app.which {
            Which::Flat(sim) => {
                let level = quality(name)
                    .ok_or_else(|| known(QUALITY_LEVELS.iter().map(|q| q.name).collect()))?;
                let c = sim.config().with_quality(level);
                sim.set_config(c);
            }
            Which::Volume(v) => {
                let level = quality3d(name)
                    .ok_or_else(|| known(QUALITY_LEVELS_3D.iter().map(|q| q.name).collect()))?;
                let c = v.sim.config().with_quality(level);
                v.sim.set_config(c);
            }
        }
        Ok(())
    }

    /// Every quality level's name, coarsest first.
    #[wasm_bindgen(js_name = qualityNames)]
    pub fn quality_names() -> Vec<String> {
        QUALITY_LEVELS.iter().map(|q| q.name.to_string()).collect()
    }

    /// A handful of splats, as the space bar throws.
    pub fn splats(&self, count: u32) {
        match &mut self.app.borrow_mut().which {
            Which::Flat(sim) => sim.multiple_splats(count.min(200)),
            Which::Volume(v) => v.sim.multiple_splats(count.min(16)),
        }
    }

    pub fn paused(&self) -> bool {
        match &self.app.borrow().which {
            Which::Flat(sim) => sim.paused(),
            Which::Volume(v) => v.sim.paused(),
        }
    }

    /// Frames per second, averaged over the last half second.
    pub fn fps(&self) -> f32 {
        self.app.borrow().fps
    }

    /// What the grids actually came out as, and what they cost — a quality
    /// level is a request, and the device has the last word.
    pub fn info(&self) -> String {
        let app = self.app.borrow();
        match &app.which {
            Which::Flat(sim) => {
                let ((dw, dh), (sw, sh)) = sim.grid_sizes();
                format!(
                    "{{\"mode\":\"2d\",\"dye\":[{dw},{dh}],\"sim\":[{sw},{sh}],\
                     \"sweeps\":{},\"bytes\":{},\"formats\":\"{:?}\"}}",
                    sim.config().pressure_iterations,
                    sim.memory_bytes(),
                    sim.formats(),
                )
            }
            Which::Volume(v) => {
                let (solver, dye) = v.sim.grids();
                format!(
                    "{{\"mode\":\"3d\",\"dye\":[{dye},{dye},{dye}],\
                     \"sim\":[{solver},{solver},{solver}],\"sweeps\":{},\"bytes\":{},\
                     \"steps\":{},\"formats\":\"{:?}\"}}",
                    v.sim.config().pressure_iterations,
                    v.sim.memory_bytes(),
                    v.sim.config().march_steps,
                    v.sim.formats(),
                )
            }
        }
    }
}

/// Knobs only the volume has, so pushing a whole panel at the flat fluid is
/// not an error.
fn is_volume_knob(name: &str) -> bool {
    matches!(
        name,
        "resolution"
            | "buoyancy"
            | "buoyancy_x"
            | "buoyancy_y"
            | "buoyancy_z"
            | "march_steps"
            | "absorption"
            | "emission"
            | "exposure"
            | "empty_space_skip"
            | "cache_bindings"
            | "compact_dye"
            | "accumulate"
            | "reject"
    )
}

/// …and the other way round.
fn is_flat_knob(name: &str) -> bool {
    matches!(
        name,
        "sim_resolution"
            | "shading"
            | "colorful"
            | "color_update_speed"
            | "bloom"
            | "bloom_iterations"
            | "bloom_intensity"
            | "bloom_threshold"
            | "sunrays"
            | "sunrays_weight"
            | "maccormack"
    )
}

/// CSS pixels times the device ratio, which is what the backing store has to
/// be for the picture not to be soft on a retina display.
fn canvas_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    let ratio = dpr();
    let w = (canvas.client_width() as f64 * ratio).floor() as u32;
    let h = (canvas.client_height() as f64 * ratio).floor() as u32;
    (w.max(1), h.max(1))
}

/// Push the volume where the pointer is, dragging it the way the pointer
/// moved.
///
/// The colour walks the hue wheel so successive pushes are distinguishable,
/// the same way the plane's splats do.
fn push_into_volume(app: &mut App, at: (f32, f32), drag: (f32, f32), size: (f32, f32)) {
    let (w, h) = (size.0.max(1.0), size.1.max(1.0));
    let hue = app.pushes as f32 * 0.137;
    app.pushes = app.pushes.wrapping_add(1);
    let colour = super::hsv_to_rgb(hue.fract(), 1.0, 1.0);
    let force = app
        .volume()
        .map(|v| v.sim.config().splat_force)
        .unwrap_or(1.0);
    if let Some(v) = app.volume() {
        let camera = v.camera.clone();
        v.sim.impulse_at_screen(
            &camera,
            [at.0 / w, at.1 / h],
            [drag.0 / w, drag.1 / h],
            force,
            [colour[0] * 1.2, colour[1] * 1.2, colour[2] * 1.2],
        );
    }
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

    // On macOS a ctrl-click *is* a secondary click, so without this the
    // context menu opens over the canvas in the middle of the gesture.
    let menu = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
        e.prevent_default();
    });
    target.add_event_listener_with_callback("contextmenu", menu.as_ref().unchecked_ref())?;
    menu.forget();

    let a = app.clone();
    listen(
        "pointerdown",
        Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            let (x, y) = pointer_at(&e);
            let mut app = a.borrow_mut();
            app.dragging = true;
            // Ctrl pushes the fluid; everything else is the orbit control
            // doing what it does everywhere else in this library. On touch
            // there is no ctrl, so the panel's switch stands in for it.
            app.pushing = e.ctrl_key() || app.push_mode;
            app.last_pointer = Some((x, y));
            let (w, h) = (app.config.width as f32, app.config.height as f32);
            if let Some(sim) = app.flat() {
                sim.pointer_down(e.pointer_id(), x, y);
            } else if app.pushing {
                // A click with no drag is still a push: the volume gets a
                // puff where the pointer is looking, so ctrl-clicking does
                // something rather than waiting for movement.
                push_into_volume(&mut app, (x, y), (0.0, 0.0), (w, h));
            }
            e.prevent_default();
        }),
    )?;

    let a = app.clone();
    listen(
        "pointermove",
        Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            let (x, y) = pointer_at(&e);
            let mut app = a.borrow_mut();
            let (w, h) = (app.config.width as f32, app.config.height as f32);
            let last = app.last_pointer;
            let dragging = app.dragging;
            if app.flat().is_some() {
                app.flat().unwrap().pointer_move(e.pointer_id(), x, y);
            } else if dragging {
                let (dx, dy) = match last {
                    Some((px, py)) => (x - px, y - py),
                    None => (0.0, 0.0),
                };
                if app.pushing {
                    // A ctrl-drag pushes the fluid the way it was dragged.
                    // The depth it acts at is the volume's centre — see
                    // `FluidSim3d::impulse_at_screen` for why that is the
                    // only answer that does not need a readback.
                    push_into_volume(&mut app, (x, y), (dx, dy), (w, h));
                    app.last_pointer = Some((x, y));
                    return;
                }
                let ev = OrbitEvent {
                    dx,
                    dy,
                    wheel: 0.0,
                    rotating: true,
                    panning: false,
                    cursor: ndc((x, y), (w, h)),
                };
                if let Some(v) = app.volume() {
                    v.controls.update(ev, &mut v.camera, (w, h));
                }
            }
            app.last_pointer = Some((x, y));
        }),
    )?;

    for name in ["pointerup", "pointercancel", "pointerleave"] {
        let a = app.clone();
        listen(
            name,
            Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
                let mut app = a.borrow_mut();
                app.dragging = false;
                app.last_pointer = None;
                if let Some(sim) = app.flat() {
                    sim.pointer_up(e.pointer_id());
                }
            }),
        )?;
    }

    // Scroll zooms the 3D camera.
    {
        let a = app.clone();
        let wheel =
            Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
                let mut app = a.borrow_mut();
                let (w, h) = (app.config.width as f32, app.config.height as f32);
                if let Some(v) = app.volume() {
                    let ratio = dpr();
                    let at = ((e.offset_x() as f64 * ratio) as f32, (e.offset_y() as f64 * ratio) as f32);
                    let ev = OrbitEvent {
                        dx: 0.0,
                        dy: 0.0,
                        // Browsers report wheel deltas an order larger than winit.
                        wheel: -e.delta_y() as f32 * 0.4,
                        rotating: false,
                        panning: false,
                        cursor: ndc(at, (w, h)),
                    };
                    v.controls.update(ev, &mut v.camera, (w, h));
                    e.prevent_default();
                }
            });
        target.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
        wheel.forget();
    }

    // Space and `p`, as the original binds them.
    if let Some(window) = web_sys::window() {
        let a = app.clone();
        let keys =
            Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
                let mut app = a.borrow_mut();
                match e.key().as_str() {
                    " " => {
                        match &mut app.which {
                            Which::Flat(sim) => sim.multiple_splats(15),
                            Which::Volume(v) => v.sim.multiple_splats(4),
                        }
                        e.prevent_default();
                    }
                    "p" | "P" => match &mut app.which {
                        Which::Flat(sim) => {
                            let p = !sim.paused();
                            sim.set_paused(p);
                        }
                        Which::Volume(v) => {
                            let p = !v.sim.paused();
                            v.sim.set_paused(p);
                        }
                    },
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

/// Backing-store pixels to NDC (−1..1, +y up), as `PointerEvent::cursor` wants.
fn ndc((x, y): (f32, f32), (w, h): (f32, f32)) -> (f32, f32) {
    (x / w.max(1.0) * 2.0 - 1.0, 1.0 - y / h.max(1.0) * 2.0)
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
        match &mut app.which {
            Which::Flat(sim) => sim.resize(w, h),
            Which::Volume(v) => v.camera.set_aspect(w as f32 / h.max(1) as f32),
        }
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
    match &mut app.which {
        Which::Flat(sim) => sim.update(dt),
        Which::Volume(v) => {
            // A source near the floor that wanders, so the plume braids. It
            // stops after a few seconds and then restarts, which is what
            // keeps a page you leave open from filling the box and staying
            // full.
            v.elapsed += dt;
            let cycle = v.elapsed % 5.0;
            if cycle < 1.1 {
                let a = cycle * 9.0;
                let t = cycle / 1.1;
                v.sim.splat(
                    [0.5 + 0.06 * a.cos(), 0.11, 0.5 + 0.06 * a.sin()],
                    [45.0 * a.sin(), 300.0, 45.0 * a.cos()],
                    [1.15 - 0.8 * t, 0.26 + 0.34 * t, 0.2 + 0.9 * t],
                );
            }
            v.sim.update(dt);
        }
    }

    match app.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
            match &mut app.which {
                Which::Flat(sim) => sim.render(&frame.texture),
                Which::Volume(v) => {
                    let camera = &v.camera;
                    v.sim.render(camera, &frame.texture);
                }
            }
            app.queue.present(frame);
        }
        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
            // Worth a word: a surface that is lost every frame leaves the
            // loop looking alive and the canvas blank.
            log::warn!("fluid: surface lost or outdated, reconfiguring");
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
