//! Parity scene `capsule` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-capsule.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example capsule
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, CapsuleGeometry, Color, DirectionalLight, Mesh, Object3D, PerspectiveCamera,
    Scene, StandardMaterial, Vector3,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — capsule")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    // --- scene (matches threers parity iframe) ---
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.5));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    scene.add(Object3D::mesh(Mesh::new(
        CapsuleGeometry::new(0.35, 0.9, 8, 16),
        StandardMaterial::new(Color::from_hex(0x88cc66))
            .with_roughness(0.4)
            .into(),
    )));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.5);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    scene.update_world();
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
                    WindowEvent::RedrawRequested => {
                        scene.update_world();
                        view.render(&mut scene, &camera);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => window_for_loop.request_redraw(),
            _ => {}
        })
        .expect("event loop");
}
