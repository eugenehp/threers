//! Orbit a cube with mouse — native Rust controls demo.
//!
//! ```text
//! cargo run --example controls_orbit
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom
//!
//! See `docs/controls.md` for three.js / threers shim / TypeScript equivalents.

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Mesh, Object3D, OrbitControls,
    PerspectiveCamera, PointerEvent, Scene, StandardMaterial, Vector3,
};
use threers::renderer::WindowRenderer;

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

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

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — OrbitControls (Rust)")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 700))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202030);

    let geom = BoxGeometry::new(1.0, 1.0, 1.0);
    let mat = StandardMaterial::new(Color::from_hex(0xff6633))
        .with_roughness(0.4)
        .with_metalness(0.1);
    scene.add(Object3D::mesh(Mesh::new(geom, mat.into())));

    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    key.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(key);

    let mut camera =
        PerspectiveCamera::new(60.0, view.aspect(), 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);

    let mut controls = OrbitControls::new(&camera);
    let mut input = InputState::new();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
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
                        let ev = PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                            ..Default::default()
                        };
                        controls.update(
                            ev,
                            &mut camera,
                            { let (w, h) = view.size(); (w as f32, h as f32) },
                        );
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let wheel = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32,
                        };
                        let ev = PointerEvent {
                            dx: 0.0,
                            dy: 0.0,
                            wheel,
                            rotating: false,
                            panning: false,
                            ..Default::default()
                        };
                        controls.update(
                            ev,
                            &mut camera,
                            { let (w, h) = view.size(); (w as f32, h as f32) },
                        );
                    }
                    WindowEvent::RedrawRequested => {
                        view.render(&mut scene, &camera);
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .expect("event loop");
}
