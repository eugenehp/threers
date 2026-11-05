//! Parity scene `colored-dir` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-colored-dir.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example colored_dir
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, Color, DirectionalLight, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, Vector3,
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
            .with_title("threers — colored-dir")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    // --- scene (matches threers parity iframe) ---
    // const canvas = document.getElementById('c');
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0a0e14);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.1));
    let mut dl1 = Object3D::light(DirectionalLight::new(Color::from_hex(0xff4488), 1.0));
    dl1.position = Vector3::new(3.0, 2.0, 1.0);
    scene.add(dl1);
    let mut dl2 = Object3D::light(DirectionalLight::new(Color::from_hex(0x44aaff), 1.0));
    dl2.position = Vector3::new(-3.0, 2.0, 1.0);
    scene.add(dl2);
    let sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.1, 32, 16),
        StandardMaterial::new(Color::from_hex(0xffffff))
            .with_roughness(0.5)
            .with_metalness(0.0)
            .into(),
    ));
    scene.add(sphere);
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
