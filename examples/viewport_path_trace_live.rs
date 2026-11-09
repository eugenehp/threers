//! Interactive path-traced viewport: orbit the camera and watch the image refine.
//!
//! ```text
//! cargo run --release --example viewport_path_trace_live --features raytrace
//! cargo run --release --example viewport_path_trace_live --features raytrace -- --gpu
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom · Release mouse to refine.
//! Keys: `R` reset · `D` denoise · `A` adaptive · `S` redistribute · `Tab` AOV
//! · `H` error · `E` PNG · `X` EXR · `C`/`L` checkpoint · `[`/`]` exposure
//! · `=`/`-` sample budget · `F` firefly clamp · `G` glass caustics · `Space` pause.

use threers::cameras::Camera;
use threers::overlay::Overlay;
use threers::raytrace::{gpu::GpuBackend, Aov, CpuBackend, RaytraceRenderer, RaytraceSettings};
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Material, Mesh, Object3D, OrbitControls,
    PerspectiveCamera, PhysicalMaterial, PlaneGeometry, PointerEvent, Scene, SphereGeometry,
    StandardMaterial, Vector3,
};

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<winit::dpi::PhysicalPosition<f64>>,
    camera_dirty: bool,
    paused: bool,
    view: ViewMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Beauty,
    Albedo,
    Normal,
    Depth,
    Error,
    Samples,
}

impl ViewMode {
    fn next(self) -> Self {
        match self {
            Self::Beauty => Self::Albedo,
            Self::Albedo => Self::Normal,
            Self::Normal => Self::Depth,
            Self::Depth => Self::Error,
            Self::Error => Self::Samples,
            Self::Samples => Self::Beauty,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Beauty => "beauty",
            Self::Albedo => "albedo",
            Self::Normal => "normal",
            Self::Depth => "depth",
            Self::Error => "error",
            Self::Samples => "samples",
        }
    }
}

impl InputState {
    fn new() -> Self {
        Self {
            rotating: false,
            panning: false,
            last: None,
            camera_dirty: false,
            paused: false,
            view: ViewMode::Beauty,
        }
    }

    fn motion(&mut self, pos: winit::dpi::PhysicalPosition<f64>) -> (f32, f32) {
        let (dx, dy) = match self.last {
            Some(last) => ((pos.x - last.x) as f32, (pos.y - last.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        (dx, dy)
    }
}

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.04, 0.05, 0.08);

    let floor = StandardMaterial::new(Color::new(0.55, 0.55, 0.58)).with_roughness(0.85);
    let mut floor_obj = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(14.0, 14.0),
        Material::Standard(floor),
    ));
    floor_obj.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor_obj);

    let glass = PhysicalMaterial::new(Color::new(0.92, 0.96, 1.0))
        .with_roughness(0.04)
        .with_metalness(0.0)
        .with_transmission(1.0, 1.45, 0.5);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 32, 24),
        Material::Physical(glass),
    )));

    let mut hero = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        Material::Standard(
            StandardMaterial::new(Color::new(0.78, 0.35, 0.18)).with_roughness(0.35),
        ),
    ));
    hero.position = Vector3::new(-1.2, 0.6, 0.0);
    scene.add(hero);

    scene.add_light(AmbientLight::new(Color::new(0.25, 0.28, 0.32), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::new(1.0, 0.96, 0.88), 2.2));
    key.position = Vector3::new(4.0, 6.0, 3.0);
    scene.add(key);
    scene
}

fn use_gpu() -> bool {
    std::env::args().any(|a| a == "--gpu")
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — live path trace")
        .with_inner_size(PhysicalSize::new(960, 720));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu by default; the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal`. The traced picture goes up as a 2D overlay
        // over an empty frame, which both backends composite.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        let size = window.inner_size();
        let mut config = (size.width.max(1), size.height.max(1));
        let mut overlay = Overlay::new(config.0, config.1);
        let mut blank = Scene::new();
        blank.background = Color::new(0.0, 0.0, 0.0);

        let mut scene = build_scene();
        let mut camera =
            PerspectiveCamera::new(42.0, config.0 as f32 / config.1 as f32, 0.1, 100.0);
        camera.position = Vector3::new(4.5, 2.8, 5.5);
        camera.look_at(Vector3::new(0.0, 0.6, 0.0));

        let settings = RaytraceSettings::preview()
            .with_samples(256)
            .with_denoise(true)
            .with_adaptive(0.02)
            .with_sample_redistribution(true);

        let mut path = if use_gpu() {
            // A device of its own: the picture comes back as bytes either way,
            // and the window's may be Metal's.
            let backend = GpuBackend::headless()
                .expect("a GPU for the path tracer")
                .with_samples_per_dispatch(4);
            let mut r = RaytraceRenderer::with_backend(config.0, config.1, Box::new(backend));
            r.set_settings(settings);
            r.prepare(&mut scene, &camera);
            eprintln!("backend: gpu");
            r
        } else {
            let mut r =
                RaytraceRenderer::with_backend(config.0, config.1, Box::new(CpuBackend::new()));
            r.set_settings(settings);
            r.prepare(&mut scene, &camera);
            eprintln!("backend: cpu");
            r
        };

        let mut controls = OrbitControls::new(&camera);
        let mut input = InputState::new();
        let mut title_timer = std::time::Instant::now();
        let window_for_loop = window.clone();

        move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        view.resize(new_size.width.max(1), new_size.height.max(1));
                        overlay.resize(new_size.width.max(1), new_size.height.max(1));
                        camera.set_aspect(
                            new_size.width.max(1) as f32 / new_size.height.max(1) as f32,
                        );
                        config = path.set_size(new_size.width.max(1), new_size.height.max(1));
                        path.prepare_if_changed(&mut scene, &camera);
                        input.camera_dirty = false;
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let pressed = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => input.rotating = pressed,
                            MouseButton::Right => input.panning = pressed,
                            _ => {}
                        }
                        if !pressed {
                            input.last = None;
                            if input.camera_dirty {
                                path.prepare_if_changed(&mut scene, &camera);
                                input.camera_dirty = false;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if !input.rotating && !input.panning {
                            return;
                        }
                        let (dx, dy) = input.motion(position);
                        let ev = PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                            ..Default::default()
                        };
                        controls.update(ev, &mut camera, (config.0 as f32, config.1 as f32));
                        input.camera_dirty = true;
                        // Clear + coarse preview while dragging so the film
                        // matches the new view instead of showing a stale frame.
                        path.prepare_if_changed(&mut scene, &camera);
                        let _ = path.accumulate_unconverged_tiles_budgeted(0, 1, 2, 8);
                        input.camera_dirty = false;
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                state: ElementState::Pressed,
                                repeat: false,
                                ..
                            },
                        ..
                    } => match code {
                        KeyCode::KeyR => {
                            path.reset_film();
                            path.prepare_if_changed(&mut scene, &camera);
                        }
                        KeyCode::KeyD => {
                            path.set_denoise(!path.settings().denoise);
                        }
                        KeyCode::KeyA => {
                            let t = path.settings().adaptive_threshold;
                            path.set_adaptive(if t > 0.0 { 0.0 } else { 0.02 });
                        }
                        KeyCode::KeyS => {
                            let on = !path.settings().sample_redistribution;
                            path.set_sample_redistribution(on);
                        }
                        KeyCode::KeyH => {
                            input.view = if input.view == ViewMode::Error {
                                ViewMode::Beauty
                            } else {
                                ViewMode::Error
                            };
                        }
                        KeyCode::Tab => {
                            input.view = input.view.next();
                        }
                        KeyCode::KeyE => {
                            let (w, h) = path.size();
                            let rgba = path.resolve_rgba();
                            let png = threers::encode_png(w, h, &rgba);
                            let out = "out/viewport_path_trace_live.png";
                            if let Some(dir) = std::path::Path::new(out).parent() {
                                let _ = std::fs::create_dir_all(dir);
                            }
                            match std::fs::write(out, png) {
                                Ok(()) => eprintln!("wrote {out} ({w}×{h})"),
                                Err(e) => eprintln!("save png failed: {e}"),
                            }
                        }
                        KeyCode::KeyX => match path.save_exr("out/viewport_path_trace_live.exr") {
                            Ok(()) => eprintln!(
                                "wrote viewport_path_trace_live.exr ({} spp)",
                                path.samples()
                            ),
                            Err(e) => eprintln!("save exr failed: {e}"),
                        },
                        KeyCode::KeyC => match path.save_film("viewport_path_trace_live.rtfc") {
                            Ok(()) => eprintln!(
                                "wrote viewport_path_trace_live.rtfc ({} spp)",
                                path.samples()
                            ),
                            Err(e) => eprintln!("save checkpoint failed: {e}"),
                        },
                        KeyCode::KeyL => match path.load_film("viewport_path_trace_live.rtfc") {
                            Ok(()) => eprintln!(
                                "loaded viewport_path_trace_live.rtfc ({} spp)",
                                path.samples()
                            ),
                            Err(e) => eprintln!("load checkpoint failed: {e}"),
                        },
                        KeyCode::BracketLeft => {
                            let e = (path.settings().exposure / 1.25).max(0.05);
                            path.set_exposure(e);
                        }
                        KeyCode::BracketRight => {
                            let e = (path.settings().exposure * 1.25).min(32.0);
                            path.set_exposure(e);
                        }
                        KeyCode::Equal => {
                            let spp = path.settings().samples_per_pixel.saturating_mul(2).max(1);
                            path.set_sample_budget(spp);
                        }
                        KeyCode::Minus => {
                            let spp = (path.settings().samples_per_pixel / 2).max(1);
                            path.set_sample_budget(spp);
                        }
                        KeyCode::KeyF => {
                            let c = path.settings().clamp_indirect;
                            path.set_clamp_indirect(if c > 0.0 { 0.0 } else { 10.0 });
                        }
                        KeyCode::KeyG => {
                            let on = !path.settings().caustic_glass_shadows;
                            path.set_caustic_glass_shadows(on);
                        }
                        KeyCode::Space => {
                            input.paused = !input.paused;
                        }
                        _ => {}
                    },
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
                                ..Default::default()
                            },
                            &mut camera,
                            (config.0 as f32, config.1 as f32),
                        );
                        input.camera_dirty = true;
                        path.prepare_if_changed(&mut scene, &camera);
                        input.camera_dirty = false;
                    }
                    WindowEvent::RedrawRequested => {
                        const MAX_TILES: u32 = 16;
                        const BUDGET_MS: u32 = 14;
                        if !input.rotating
                            && !input.panning
                            && !input.paused
                            && path.needs_more_samples()
                        {
                            let _ = path.accumulate_interactive(
                                &mut scene, &camera, 0, 1, MAX_TILES, BUDGET_MS,
                            );
                        }

                        let rgba = match input.view {
                            ViewMode::Beauty => path.resolve_rgba(),
                            ViewMode::Albedo => path.resolve_rgba_aov(Aov::Albedo),
                            ViewMode::Normal => path.resolve_rgba_aov(Aov::Normal),
                            ViewMode::Depth => path.resolve_rgba_aov(Aov::Depth),
                            ViewMode::Error => path.resolve_rgba_error_overlay(0.65),
                            ViewMode::Samples => path.resolve_rgba_samples(),
                        };
                        overlay.put_image_data(0, 0, config.0, config.1, &rgba);
                        view.render_frame(&mut blank, &camera, None, Some(&mut overlay));

                        if title_timer.elapsed().as_millis() > 250 {
                            title_timer = std::time::Instant::now();
                            let spp = path.samples();
                            let total = path.settings().samples_per_pixel;
                            let tile = path.recommended_tile_size();
                            let stats = path.render_stats_for_tile_size(tile);
                            let eff = stats.efficiency * 100.0;
                            let conv = stats.converged_fraction * 100.0;
                            let tiles = stats.unconverged_tile_count;
                            let ms = path.last_tile_ms();
                            let denoise = if path.settings().denoise {
                                "denoise"
                            } else {
                                "raw"
                            };
                            let adapt = if path.settings().adaptive_threshold > 0.0 {
                                "adapt"
                            } else {
                                "fixed"
                            };
                            let redist = if path.settings().sample_redistribution {
                                "+redist"
                            } else {
                                ""
                            };
                            let clamp = if path.settings().clamp_indirect > 0.0 {
                                " · clamp"
                            } else {
                                ""
                            };
                            let caustic = if path.settings().caustic_glass_shadows {
                                " · caustic"
                            } else {
                                ""
                            };
                            let pause = if input.paused { " · paused" } else { "" };
                            let exp = path.settings().exposure;
                            let view = input.view.label();
                            let max_side = path
                                .max_film_side()
                                .map(|n| format!("{n}px"))
                                .unwrap_or_else(|| "n/a".into());
                            window_for_loop.set_title(&format!(
                                "threers — path trace {spp}/{total} spp · {view} · eff {eff:.0}% · converged {conv:.0}% · tiles {tiles} · {ms:.1}ms/tile · exp {exp:.2} · max {max_side} · {denoise} · {adapt}{redist}{clamp}{caustic}{pause}"
                            ));
                        }

                        let keep_going = (!input.paused && path.needs_more_samples())
                            || input.rotating
                            || input.panning;
                        if keep_going {
                            window_for_loop.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait
                if ((!input.paused && path.needs_more_samples())
                    || input.rotating
                    || input.panning) =>
            {
                window_for_loop.request_redraw();
            }
            _ => {}
        }
    });
}
