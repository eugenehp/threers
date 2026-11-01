//! An orbiting spacecraft built entirely from the material presets: a gold-MLI
//! bus, silver-foil thruster module, two solar-array wings, a brushed-aluminium
//! truss, a heat-tinted titanium engine bell, and a glass sensor porthole.
//!
//! ```text
//! cargo run --example spacecraft_materials
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom
//!
//! Everything is procedural — no HDRI or texture downloads needed. See
//! `material_chart` for a side-by-side contact sheet of every preset.
//!
//! # The one setup step that matters
//!
//! `scene.environment` is not optional here. A `metalness = 1.0` surface has no
//! diffuse term, so with no environment every metal below would render
//! essentially black regardless of how many lights are in the scene. The
//! `PmremGenerator::generate_pmrem` call inside `orbital_environment` is what
//! makes roughness mean anything: it prefilters the cubemap into a mip chain
//! that rough metals sample at blurrier levels.
//!
//! # A note on exposure
//!
//! `Renderer` applies no tone mapping (that lives in `ToneMappingPass`), so
//! anything above 1.0 hard-clips to white and loses its hue. The light
//! intensities here are set to keep the metal highlights just under that.

use std::f32::consts::{PI, TAU};
use threers::cameras::Camera;
use threers::materials::presets;
use threers::{
    AmbientLight, BoxGeometry, CylinderGeometry, DirectionalLight, Material, Mesh, Object3D,
    OrbitControls, PerspectiveCamera, PhysicalMaterial, PointerEvent, Scene,
    SphereGeometry, TorusGeometry, Vector3,
};
use threers::renderer::WindowRenderer;

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// Brings in `Arc`-wrapped procedural textures, `orbital_environment`, `SUN_DIR`.
include!("spacecraft_common.inc");

struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<PhysicalPosition<f64>>,
}

impl InputState {
    fn new() -> Self {
        Self {
            rotating: false,
            panning: false,
            last: None,
        }
    }
    fn motion(&mut self, pos: PhysicalPosition<f64>) -> (f32, f32) {
        let (dx, dy) = match self.last {
            Some(last) => ((pos.x - last.x) as f32, (pos.y - last.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        (dx, dy)
    }
}

include!("spacecraft_build.inc");

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — spacecraft materials")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 750))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x03040a);

    // The load-bearing line: prefiltered orbital IBL. Remove it and every metal
    // below turns black.
    scene.environment = Some(orbital_environment(SUN_DIR));

    build_spacecraft(&mut scene);

    scene.add_light(AmbientLight::new(Color::from_hex(0x223044), 0.06));

    let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff6e8), 2.4));
    sun.position = Vector3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) * 12.0;
    sun.cast_shadow = true;
    scene.add(sun);

    let mut bounce = Object3D::light(DirectionalLight::new(earth_bounce(), 0.5));
    bounce.position = Vector3::new(-0.3, -1.0, 0.2) * 12.0;
    scene.add(bounce);

    let mut camera =
        PerspectiveCamera::new(42.0, view.aspect(), 0.1, 200.0);
    camera.position = Vector3::new(4.2, 2.4, 6.0);
    camera.look_at(Vector3::ZERO);

    let mut controls = OrbitControls::new(&camera);
    let mut input = InputState::new();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent { event, window_id } = event {
                if window_id != window_for_loop.id() {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        view.resize(new_size.width, new_size.height);
                        camera.set_aspect(view.aspect());
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
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if !input.rotating && !input.panning {
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
                                ..Default::default()
                            },
                            &mut camera,
                            { let (w, h) = view.size(); (w as f32, h as f32) },
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
                                ..Default::default()
                            },
                            &mut camera,
                            { let (w, h) = view.size(); (w as f32, h as f32) },
                        );
                    }
                    WindowEvent::RedrawRequested => {
                        scene.update_world();
                        view.render(&mut scene, &camera);
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            }
        })
        .expect("event loop");
}
