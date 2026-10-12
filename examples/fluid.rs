//! The fluid simulation, in a window, under your hand.
//!
//!     cargo run --release --features fluid --example fluid
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
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

/// The mouse's pointer id. Touches use their own, which winit hands out as
/// `u64`s, so negative ids cannot collide with them.
const MOUSE: i32 = -1;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — fluid")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 700))
            .build(&event_loop)
            .expect("window"),
    );

    // One call for the instance, adapter, device and queue, with the limits
    // a threers renderer needs already raised. See `threers::gpu`.
    let (gpu, surface) = GpuContext::with_surface(window.clone().into())
        .await
        .expect("gpu");
    println!("adapter: {}", gpu.describe());

    // The narrowest float formats this adapter will both render to and filter.
    let formats = FluidFormats::for_adapter(&gpu.adapter)
        .expect("this adapter cannot render to a half-float texture");
    println!("formats: {formats:?}");
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

    let size = window.inner_size();
    // `false`: prefer a *non*-sRGB surface. The display pass hands back
    // display-referred values, which is what the original wrote into a plain
    // RGBA8 canvas. An sRGB surface works too — the shader puts the inverse
    // transfer in first — but this is one less step.
    let mut config = gpu.surface_config(&surface, size.width, size.height, false);
    surface.configure(&device, &config);

    let mut sim = FluidSim::with_formats(
        device.clone(),
        queue.clone(),
        config.width,
        config.height,
        FluidConfig {
            // A window opens on a fluid that is already moving; an empty black
            // canvas reads as a failure to start.
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1),
            ..FluidConfig::default()
        },
        formats,
    )
    .expect("fluid");
    sim.multiple_splats(20);
    report(&sim);

    let mut cursor = (0.0f32, 0.0f32);
    let mut last = std::time::Instant::now();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        sim.resize(config.width, config.height);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        cursor = (position.x as f32, position.y as f32);
                        sim.pointer_move(MOUSE, cursor.0, cursor.1);
                    }
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => match state {
                        ElementState::Pressed => sim.pointer_down(MOUSE, cursor.0, cursor.1),
                        ElementState::Released => sim.pointer_up(MOUSE),
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
                            TouchPhase::Started => sim.pointer_down(id, x, y),
                            TouchPhase::Moved => sim.pointer_move(id, x, y),
                            TouchPhase::Ended | TouchPhase::Cancelled => sim.pointer_up(id),
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. }
                        if event.state == ElementState::Pressed =>
                    {
                        key(&mut sim, &event.logical_key);
                    }
                    WindowEvent::RedrawRequested => {
                        let now = std::time::Instant::now();
                        // Clamped the way the original clamps it: a frame that
                        // took a quarter of a second — a drag of the window, a
                        // shader compile — must not advance the fluid a quarter
                        // of a second, or it jumps.
                        let dt = (now - last).as_secs_f32().min(1.0 / 60.0);
                        last = now;
                        sim.update(dt);

                        match surface.get_current_texture() {
                            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                                sim.render(&frame.texture);
                                queue.present(frame);
                            }
                            wgpu::CurrentSurfaceTexture::Lost
                            | wgpu::CurrentSurfaceTexture::Outdated => {
                                surface.configure(&device, &config);
                            }
                            e => log::error!("surface error: {e:?}"),
                        }
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .expect("event loop");
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

fn key(sim: &mut FluidSim, key: &Key) {
    let mut c = *sim.config();
    match key.as_ref() {
        Key::Named(NamedKey::Space) => {
            sim.multiple_splats(15);
            return;
        }
        Key::Character("p") => {
            let paused = !sim.paused();
            sim.set_paused(paused);
            println!("{}", if paused { "paused" } else { "running" });
            return;
        }
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
            sim.set_config(c.with_quality(level));
            report(sim);
            return;
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
        _ => return,
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
    sim.set_config(c);
}
