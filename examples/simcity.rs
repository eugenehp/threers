//! A procedurally generated city you can fly around, with a running clock:
//! the sun sweeps, shadows swing, and at dusk the windows and street lights
//! come on.
//!
//! ```text
//! cargo run --release --example simcity
//! cargo run --release --example simcity -- --seed 41 --blocks 10
//! cargo run --release --example simcity -- --layout waterfront
//! ```
//!
//! | Input | |
//! |-------|--|
//! | left-drag / right-drag / wheel | orbit · pan · zoom |
//! | `space` | pause the clock and the traffic |
//! | `[` `]` | slow down / speed up the day |
//! | `n` `d` | jump to midnight / midday |
//! | `v` `f` | street view / frame the whole city |
//! | `c` | ride along: in a car, then on foot, then off |
//! | `l` | next street layout |
//! | `r` | regenerate with a new seed |
//!
//! Run it with `--release`. The city is ~110k triangles in ~30 draw calls, which
//! is fine either way, but a debug build spends its whole frame budget
//! generating it.
//!
//! The city itself lives in `examples/simcity/`, shared with the headless
//! `simcity_render` and `simcity_metal` examples.
//! `cargo test --test simcity` checks its invariants.

use std::time::Instant;

use threers::cameras::Camera;
use threers::{OrbitControls, PerspectiveCamera, PointerEvent, Renderer, ToneMapping};

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

// The city itself lives in `examples/simcity/`, shared by all three
// entry points. `#[path]` because a directory example would otherwise
// need its own `main.rs`, and there are three of those.
#[path = "simcity/mod.rs"]
mod city;
use city::*;

/// Seconds of wall clock for one in-world day, before `[` / `]`.
const DAY_SECONDS: f32 = 100.0;

fn has_flag(name: &str) -> bool {
    std::env::args().skip(1).any(|a| a == name)
}

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

#[derive(Default)]
struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<PhysicalPosition<f64>>,
}

impl InputState {
    fn motion(&mut self, pos: PhysicalPosition<f64>) -> (f32, f32) {
        let d = match self.last {
            Some(l) => ((pos.x - l.x) as f32, (pos.y - l.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        d
    }
}

/// Frame the whole city from a three-quarter aerial view.
fn place_camera(camera: &mut PerspectiveCamera, extent: f32) {
    camera.position = Vector3::new(extent * 0.88, extent * 0.74, extent * 1.10);
    camera.look_at(Vector3::new(0.0, extent * 0.07, 0.0));
}

fn clock_label(t: f32) -> String {
    // t = 0 is sunrise, so 06:00.
    let hours = (t.rem_euclid(1.0) * 24.0 + 6.0) % 24.0;
    format!("{:02}:{:02}", hours as u32, ((hours.fract()) * 60.0) as u32)
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let defaults = CityParams::default();
    let mut params = CityParams {
        bake: !has_flag("--no-bake"),
        seed: arg("--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.seed),
        blocks: arg("--blocks")
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.blocks),
        cars: arg("--cars")
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.cars),
        layout: arg("--layout")
            .and_then(|s| Layout::from_name(&s))
            .unwrap_or(defaults.layout),
    };

    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — simcity")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
        // wgpu 30 dropped `Default` here; the display handle is only consulted
        // by GLES/Wayland, not Vulkan, Metal or DX12.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        wgpu::Instance::new(d)
    };
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("adapter");

    let adapter_limits = adapter.limits();
    // Two departures from `downlevel_defaults()`, both required:
    //
    // - `using_resolution` for the texture size, because the shadow map is
    //   4096 and downlevel caps it at 2048.
    // - the sampled-texture and sampler counts, because the main pass declares
    //   20 textures in the fragment stage (material maps, shadow atlases, the
    //   environment) and downlevel allows 16. `using_resolution` does not
    //   cover binding counts. `HeadlessRenderer` does the same thing.
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    limits.max_sampled_textures_per_shader_stage =
        adapter_limits.max_sampled_textures_per_shader_stage;
    limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers simcity"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        format,
        config.width,
        config.height,
    );
    // Lit windows and a low sun both go well above white; ACES rolls them off
    // without dragging the hue with them.
    renderer.set_tone_mapping(ToneMapping::AcesFilmic, 1.0);

    let mut scene = Scene::new();
    let mut city = generate_city(&mut scene, &params);
    let s = &city.stats;
    println!(
        "{} · seed {} · {} blocks/side · {} buildings · {} cars · {} on foot · {} draws · {} static triangles + up to {} in movers · {:.0} m across",
        params.layout.name(), params.seed, s.blocks, s.buildings, s.cars, s.walkers, s.draws, s.triangles, s.mover_triangles,
        s.extent * 2.0
    );
    println!("drag to orbit · right-drag to pan · wheel to zoom");
    println!("space pause · [ ] day speed · n night · d day · v street · f frame");
    println!("l next layout · c ride along · r new city");

    let mut camera = PerspectiveCamera::new(
        38.0,
        config.width as f32 / config.height as f32,
        1.0,
        city.stats.extent * 14.0,
    );
    place_camera(&mut camera, city.stats.extent);
    let mut controls = OrbitControls::new(&camera);
    controls.min_distance = 25.0;
    controls.max_distance = city.stats.extent * 5.0;
    // Stop just short of the horizon: below it the camera is under the ground
    // and the whole city turns into the back face of a terrain plane.
    controls.max_polar_angle = std::f32::consts::FRAC_PI_2 * 0.98;
    controls.enable_damping = true;

    // Raw wgpu: a compute pass rewrites the water mesh's vertex buffer in
    // place between frames, so the surface never crosses host memory.
    let mut water = city
        .water
        .as_ref()
        .map(|w| WaterSim::new(&device, w.vertices, w.level));

    let mut input = InputState::default();
    let mut time_of_day = 0.07f32;
    let mut day_speed = 1.0f32;
    let mut paused = false;
    // Ride-along: none, in a car, or on foot.
    let mut ride: Option<Gait> = None;
    let mut rebuild = false;
    let mut last = Instant::now();
    let mut elapsed = 0.0f32;
    let mut frames = 0u32;
    let mut fps_accum = 0.0f32;
    let window_loop = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_loop.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(new_size) => {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(&device, &config);
                    renderer.resize(config.width, config.height);
                    camera.set_aspect(config.width as f32 / config.height as f32);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let down = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => input.rotating = down,
                        MouseButton::Right => input.panning = down,
                        _ => {}
                    }
                    if !down {
                        input.last = None;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if !input.rotating && !input.panning {
                        input.last = Some(position);
                        return;
                    }
                    let (dx, dy) = input.motion(position);
                    controls.update(
                        PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                        },
                        &mut camera,
                        (config.width as f32, config.height as f32),
                    );
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let wheel = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    controls.update(
                        PointerEvent {
                            dx: 0.0,
                            dy: 0.0,
                            wheel,
                            rotating: false,
                            panning: false,
                        },
                        &mut camera,
                        (config.width as f32, config.height as f32),
                    );
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state != ElementState::Pressed {
                        return;
                    }
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Space) => paused = !paused,
                        Key::Named(NamedKey::Escape) => target.exit(),
                        Key::Character("[") => day_speed = (day_speed * 0.6).max(0.05),
                        Key::Character("]") => day_speed = (day_speed * 1.6).min(30.0),
                        Key::Character("n") | Key::Character("N") => time_of_day = 0.75,
                        Key::Character("d") | Key::Character("D") => time_of_day = 0.22,
                        Key::Character("c") | Key::Character("C") => {
                            ride = match ride {
                                None => Some(Gait::Drive),
                                Some(Gait::Drive) => Some(Gait::Walk),
                                _ => None,
                            };
                            if ride.is_none() {
                                place_camera(&mut camera, city.stats.extent);
                                controls.reseed_from_camera(&camera);
                            }
                        }
                        Key::Character("v") | Key::Character("V") => {
                            // Down in an avenue. The generator picked the spot,
                            // so it is on tarmac rather than inside a tower.
                            camera.position = city.vista.0;
                            camera.look_at(city.vista.1);
                            controls.reseed_from_camera(&camera);
                        }
                        Key::Character("f") | Key::Character("F") => {
                            place_camera(&mut camera, city.stats.extent);
                            controls.reseed_from_camera(&camera);
                        }
                        Key::Character("l") | Key::Character("L") => {
                            params.layout = params.layout.next();
                            rebuild = true;
                        }
                        Key::Character("r") | Key::Character("R") => {
                            params.seed = params
                                .seed
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1);
                            rebuild = true;
                        }
                        _ => {}
                    }
                    if rebuild {
                        rebuild = false;
                        scene = Scene::new();
                        // The old batches' GPU buffers are keyed on geometry
                        // identity and nothing else will drop them, so a
                        // rebuild leaks a city per press without this.
                        renderer.clear_geometry_cache();
                        city = generate_city(&mut scene, &params);
                        // The old water buffer went with the geometry cache,
                        // so the compute pass needs a new bind group.
                        water = city
                            .water
                            .as_ref()
                            .map(|w| WaterSim::new(&device, w.vertices, w.level));
                        controls.max_distance = city.stats.extent * 5.0;
                        let s = &city.stats;
                        println!(
                            "{} · seed {} · {} buildings · {} draws · {} triangles",
                            params.layout.name(),
                            params.seed,
                            s.buildings,
                            s.draws,
                            s.triangles
                        );
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - last).as_secs_f32().min(0.1);
                    last = now;
                    if !paused {
                        time_of_day = (time_of_day + dt * day_speed / DAY_SECONDS).rem_euclid(1.0);
                    }
                    // Runs even when paused: the camera still moves, and the
                    // level-of-detail split is measured from where it is.
                    let (near, movers) =
                        city.drive(&mut scene, if paused { 0.0 } else { dt }, camera.position, elapsed);
                    elapsed += dt;
                    if let (Some(sim), Some(w)) = (&mut water, &city.water) {
                        sim.step(&device, &queue, &mut renderer, &w.geometry, elapsed, 0.35);
                    }
                    let sky = city.apply_sky(&mut scene, time_of_day);
                    city.place_lights(&mut scene, camera.position, sky.lights);
                    city.animate_lights(&mut scene, elapsed);
                    // Riding takes the camera over; the orbit controls are
                    // reseeded from wherever it ends up when you get off.
                    if let Some(gait) = ride {
                        if let Some((pos, facing)) = city.mover_pose(gait, 7) {
                            let (back, up, ahead) = match gait {
                                Gait::Walk => (2.4, 1.68, 7.0),
                                _ => (8.5, 3.4, 14.0),
                            };
                            camera.position = pos - facing * back + Vector3::new(0.0, up, 0.0);
                            camera.look_at(pos + facing * ahead + Vector3::new(0.0, up * 0.7, 0.0));
                            controls.reseed_from_camera(&camera);
                        }
                    } else {
                        controls.update(
                            PointerEvent::default(),
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                    }
                    scene.update_world();

                    frames += 1;
                    fps_accum += dt;
                    if fps_accum > 0.5 {
                        window_loop.set_title(&format!(
                            "threers — simcity · {} · {} · lights {:.0}% · lod {}/{} · {:.0} fps · {}x day{}",
                            params.layout.name(),
                            clock_label(time_of_day),
                            sky.lights * 100.0,
                            near,
                            movers,
                            frames as f32 / fps_accum,
                            day_speed,
                            if paused { " · paused" } else { "" },
                        ));
                        frames = 0;
                        fps_accum = 0.0;
                    }
                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            renderer.render(&mut scene, &camera, &view, false);
                            queue.present(frame);
                        }
                        wgpu::CurrentSurfaceTexture::Lost
                        | wgpu::CurrentSurfaceTexture::Outdated => {
                            surface.configure(&device, &config);
                        }
                        // Window hidden or fully covered — normal, and not
                        // worth a line of stderr per frame.
                        wgpu::CurrentSurfaceTexture::Occluded => {}
                        e => log::error!("surface: {e:?}"),
                    }
                    window_loop.request_redraw();
                }
                _ => {}
            },
            _ => {}
        })
        .expect("event loop");
}
