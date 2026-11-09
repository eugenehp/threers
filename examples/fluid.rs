//! The fluid simulation, in a window, under your hand.
//!
//!     cargo run --release --features fluid --example fluid
//!     THREERS_BACKEND=metal cargo run --release --features fluid,metal --example fluid
//!
//! Drag to pour ink into it. Space throws a handful of splats, `p` pauses, and
//! the display stages toggle: `s` shading, `b` bloom, `r` sunrays, `c` the hue
//! cycling. `[` and `]` take the vorticity confinement down and up, which is
//! the knob that decides whether this looks like smoke or like spilled paint,
//! `m` switches the pressure solver between the original's twenty Jacobi
//! sweeps and a multigrid V-cycle, `a` switches the advection between
//! semi-Lagrangian and MacCormack, and `1`–`5` step through the quality
//! levels, from `low` to `extreme`.
//!
//! The two above `high` are not in the original, which was written for 2017
//! mobile browsers: `ultra` carries a 2048-texel dye field and a 256-texel
//! solver, `extreme` 4096 and 512. Both print what they cost — the dye field
//! alone is a few hundred megabytes at the top — and both fall back to
//! whatever the device will actually allocate.
//!
//! The window is the only thing here that is not in the library: the fluid is
//! [`threers::fluid`], and the same calls drive it in the browser.

use std::sync::Arc;

use threers::fluid::{
    quality, Advection, FluidConfig, FluidFormats, FluidSim, PressureSolver, QUALITY_LEVELS,
};
use threers::gpu::GpuContext;

use winit::{
    event::{ElementState, Event, MouseButton, Touch, TouchPhase, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
};

/// The mouse's pointer id. Touches use their own, which winit hands out as
/// `u64`s, so negative ids cannot collide with them.
const MOUSE: i32 = -1;

/// What the window's events go to, whichever backend draws.
type Handler = Box<dyn FnMut(Event<()>, &ActiveEventLoop)>;

fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — fluid")
        .with_inner_size(winit::dpi::LogicalSize::new(1100, 700));
    threers::sketch::run_window(attributes, start);
}

/// The fluid for a window that has just opened, and the handler that drives it.
fn start(window: Arc<winit::window::Window>) -> Handler {
    // A window opens on a fluid that is already moving; an empty black
    // canvas reads as a failure to start.
    let config = FluidConfig {
        seed: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1),
        ..FluidConfig::default()
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if std::env::var("THREERS_BACKEND").is_ok_and(|b| b.eq_ignore_ascii_case("metal")) {
        let canvas = metal::Canvas::new(&window, config);
        return Box::new(handler(window, canvas));
    }
    let canvas = pollster::block_on(WgpuCanvas::new(window.clone(), config));
    Box::new(handler(window, canvas))
}

/// A fluid in the window, on one backend or the other.
trait Canvas {
    fn pointer_down(&mut self, id: i32, x: f32, y: f32);
    fn pointer_move(&mut self, id: i32, x: f32, y: f32);
    fn pointer_up(&mut self, id: i32);
    fn resize(&mut self, width: u32, height: u32);
    fn key(&mut self, key: &Key);
    /// Step by `dt` seconds and draw a frame.
    fn frame(&mut self, dt: f32);
}

struct WgpuCanvas {
    sim: FluidSim,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuCanvas {
    async fn new(window: Arc<winit::window::Window>, fluid: FluidConfig) -> Self {
        let size = window.inner_size();
        // One call for the instance, adapter, device and queue, with the
        // limits a threers renderer needs already raised. See `threers::gpu`.
        let (gpu, surface) = GpuContext::with_surface(window.into()).await.expect("gpu");
        println!("adapter: {}", gpu.describe());

        // The narrowest float formats this adapter will both render to and
        // filter.
        let formats = FluidFormats::for_adapter(&gpu.adapter)
            .expect("this adapter cannot render to a half-float texture");
        println!("formats: {formats:?}");
        let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

        // `false`: prefer a *non*-sRGB surface. The display pass hands back
        // display-referred values, which is what the original wrote into a
        // plain RGBA8 canvas. An sRGB surface works too — the shader puts the
        // inverse transfer in first — but this is one less step.
        let config = gpu.surface_config(&surface, size.width, size.height, false);
        surface.configure(&device, &config);

        let mut sim = FluidSim::with_formats(
            device.clone(),
            queue.clone(),
            config.width,
            config.height,
            fluid,
            formats,
        )
        .expect("fluid");
        sim.multiple_splats(20);
        report(&sim);
        Self {
            sim,
            surface,
            config,
            device,
            queue,
        }
    }
}

impl Canvas for WgpuCanvas {
    fn pointer_down(&mut self, id: i32, x: f32, y: f32) {
        self.sim.pointer_down(id, x, y);
    }
    fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
        self.sim.pointer_move(id, x, y);
    }
    fn pointer_up(&mut self, id: i32) {
        self.sim.pointer_up(id);
    }
    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.sim.resize(self.config.width, self.config.height);
    }
    fn key(&mut self, k: &Key) {
        key(&mut self.sim, k);
    }
    fn frame(&mut self, dt: f32) {
        self.sim.update(dt);
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.sim.render(&frame.texture);
                self.queue.present(frame);
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
            }
            // Hidden, minimised, or every drawable in flight: try next frame.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {}
            e => log::error!("surface error: {e:?}"),
        }
    }
}

/// The same fluid on the native Metal backend (`THREERS_BACKEND=metal`):
/// `threers::metal::MetalFluid` drawing straight into the window's
/// `CAMetalLayer`, with no wgpu underneath.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::{action, Action, FluidConfig, Key};
    use threers::metal::{fluid_shader_source, MetalDevice, MetalFluid, MetalSurface};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    pub struct Canvas {
        device: MetalDevice,
        fluid: MetalFluid,
        surface: MetalSurface,
    }

    impl Canvas {
        pub fn new(window: &winit::window::Window, config: FluidConfig) -> Self {
            let size = window.inner_size();
            let ns_view = match window.window_handle().expect("window handle").as_raw() {
                RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr(),
                other => panic!("expected an AppKit window handle, got {other:?}"),
            };
            let device =
                MetalDevice::with_shader_source(&fluid_shader_source()).expect("Metal device");
            println!("device: {} (native Metal)", device.name());
            // Safety: the handle came from a live winit window, and winit runs
            // its event loop — and this code — on the main thread.
            let surface =
                unsafe { MetalSurface::from_ns_view(&device, ns_view, size.width, size.height, 1) }
                    .expect("CAMetalLayer");
            let mut fluid =
                MetalFluid::new(&device, size.width, size.height, config).expect("fluid");
            fluid.multiple_splats(&device, 20);
            let ((dw, dh), (sw, sh)) = fluid.grid_sizes();
            println!(
                "dye {dw}x{dh}  sim {sw}x{sh}  sweeps {}",
                config.pressure_iterations
            );
            Self {
                device,
                fluid,
                surface,
            }
        }
    }

    impl super::Canvas for Canvas {
        fn pointer_down(&mut self, id: i32, x: f32, y: f32) {
            self.fluid.pointer_down(id, x, y);
        }
        fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
            self.fluid.pointer_move(id, x, y);
        }
        fn pointer_up(&mut self, id: i32) {
            self.fluid.pointer_up(id);
        }
        fn resize(&mut self, width: u32, height: u32) {
            self.surface.resize(width, height).expect("resize");
            self.fluid
                .resize(&self.device, width, height)
                .expect("fluid resize");
        }
        fn key(&mut self, key: &Key) {
            match action(*self.fluid.config(), key) {
                Some(Action::Splats) => self.fluid.multiple_splats(&self.device, 15),
                Some(Action::Pause) => {
                    let paused = !self.fluid.paused();
                    self.fluid.set_paused(paused);
                    println!("{}", if paused { "paused" } else { "running" });
                }
                Some(Action::Set(c, _)) => {
                    self.fluid
                        .set_config(&self.device, c)
                        .expect("fluid config");
                }
                None => {}
            }
        }
        fn frame(&mut self, dt: f32) {
            self.fluid.update(&self.device, dt);
            // `None`: every drawable is in flight; skip the frame.
            if let Some(frame) = self.surface.next_frame()
                && let Err(e) = self.fluid.present(&self.device, &frame)
            {
                log::error!("present: {e}");
            }
        }
    }
}

fn handler(
    window: Arc<winit::window::Window>,
    mut canvas: impl Canvas + 'static,
) -> impl FnMut(Event<()>, &ActiveEventLoop) + 'static {
    let mut cursor = (0.0f32, 0.0f32);
    let mut last = std::time::Instant::now();
    move |event, target| match event {
        Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => target.exit(),
            WindowEvent::Resized(size) => canvas.resize(size.width, size.height),
            WindowEvent::CursorMoved { position, .. } => {
                cursor = (position.x as f32, position.y as f32);
                canvas.pointer_move(MOUSE, cursor.0, cursor.1);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => canvas.pointer_down(MOUSE, cursor.0, cursor.1),
                ElementState::Released => canvas.pointer_up(MOUSE),
            },
            WindowEvent::Touch(Touch {
                id,
                phase,
                location,
                ..
            }) => {
                // Touch ids are arbitrary `u64`s; narrowing keeps them
                // distinct from each other and from the mouse's -1.
                let id = (id as i32).abs().max(1);
                let (x, y) = (location.x as f32, location.y as f32);
                match phase {
                    TouchPhase::Started => canvas.pointer_down(id, x, y),
                    TouchPhase::Moved => canvas.pointer_move(id, x, y),
                    TouchPhase::Ended | TouchPhase::Cancelled => canvas.pointer_up(id),
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                canvas.key(&event.logical_key);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                // Clamped the way the original clamps it: a frame that
                // took a quarter of a second — a drag of the window, a
                // shader compile — must not advance the fluid a quarter
                // of a second, or it jumps.
                let dt = (now - last).as_secs_f32().min(1.0 / 60.0);
                last = now;
                canvas.frame(dt);
                window.request_redraw();
            }
            _ => {}
        },
        _ => {}
    }
}

/// What the grids came out as and what they cost — worth printing, because a
/// quality level is a request and the device has the last word.
fn report(sim: &FluidSim) {
    let ((dw, dh), (sw, sh)) = sim.grid_sizes();
    println!(
        "dye {dw}x{dh}  sim {sw}x{sh}  sweeps {}  fields {} MB",
        sim.config().pressure_iterations,
        sim.memory_bytes() / (1024 * 1024),
    );
}

/// What a key asks for, on either backend.
enum Action {
    /// A burst of splats.
    Splats,
    /// Pause or resume.
    Pause,
    /// A new configuration; `true` for a quality level, which reports the
    /// grids it came out as.
    Set(FluidConfig, bool),
}

fn action(mut c: FluidConfig, key: &Key) -> Option<Action> {
    match key.as_ref() {
        Key::Named(NamedKey::Space) => return Some(Action::Splats),
        Key::Character("p") => return Some(Action::Pause),
        Key::Character("s") => c.shading = !c.shading,
        Key::Character("b") => c.bloom = !c.bloom,
        Key::Character("r") => c.sunrays = !c.sunrays,
        Key::Character("c") => c.colorful = !c.colorful,
        Key::Character("t") => c.transparent = !c.transparent,
        Key::Character("[") => c.curl = (c.curl - 5.0).max(0.0),
        Key::Character("]") => c.curl = (c.curl + 5.0).min(50.0),
        Key::Character(d @ ("1" | "2" | "3" | "4" | "5")) => {
            let i = d.parse::<usize>().unwrap_or(3) - 1;
            let level = QUALITY_LEVELS
                .get(i)
                .unwrap_or_else(|| quality("high").expect("high"));
            println!("quality {}", level.name);
            return Some(Action::Set(c.with_quality(level), true));
        }
        Key::Character("a") => {
            // Second-order advection: the dye stays sharp instead of blurring
            // a little every step. Costs about an extra pass on the biggest
            // grid in the frame.
            c.advection = match c.advection {
                Advection::SemiLagrangian => Advection::MacCormack,
                Advection::MacCormack => Advection::SemiLagrangian,
            };
            println!("advection {:?}", c.advection);
        }
        Key::Character("m") => {
            // The pressure solver. Multigrid makes the fluid visibly more
            // incompressible — which is a *different* fluid, not just a
            // better-converged one, so it is not the default.
            c.pressure_solver = match c.pressure_solver {
                PressureSolver::Jacobi => PressureSolver::Multigrid,
                PressureSolver::Multigrid => PressureSolver::Jacobi,
            };
            println!("solver {:?}", c.pressure_solver);
        }
        Key::Character("g") => {
            // The projection coefficient, and the one toggle here that changes
            // the physics rather than the look. See `FluidConfig`.
            c.gradient_scale = if c.gradient_scale == 1.0 { 0.5 } else { 1.0 };
        }
        _ => return None,
    }
    println!(
        "shading {} bloom {} sunrays {} colorful {} transparent {} curl {} dye {} gradient {}",
        c.shading,
        c.bloom,
        c.sunrays,
        c.colorful,
        c.transparent,
        c.curl,
        c.dye_resolution,
        c.gradient_scale,
    );
    Some(Action::Set(c, false))
}

fn key(sim: &mut FluidSim, key: &Key) {
    match action(*sim.config(), key) {
        Some(Action::Splats) => sim.multiple_splats(15),
        Some(Action::Pause) => {
            let paused = !sim.paused();
            sim.set_paused(paused);
            println!("{}", if paused { "paused" } else { "running" });
        }
        Some(Action::Set(c, quality)) => {
            sim.set_config(c);
            if quality {
                report(sim);
            }
        }
        None => {}
    }
}
