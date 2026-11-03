//! Parity scene `shadow` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-shadow.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example shadow
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera,
    PlaneGeometry, Scene, StandardMaterial, Vector3,
};
use threers::renderer::WindowRenderer;

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
            .with_title("threers — shadow")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    // --- scene (matches threers parity iframe) ---
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101820);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut dl = Object3D::light({
        let mut l = DirectionalLight::new(Color::from_hex(0xffffff), 1.0);
        l.cast_shadow = true;
        l.shadow.camera_size = 3.0;
        l.shadow.camera_near = 0.1;
        l.shadow.camera_far = 20.0;
        l
    });
    dl.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(dl);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xcc4444))
            .with_roughness(0.5)
            .with_metalness(0.0)
            .into(),
    ));
    cube.cast_shadow = true;
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    scene.add(cube);
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::from_hex(0xaaaaaa))
            .with_roughness(0.7)
            .with_metalness(0.0)
            .into(),
    ));
    ground.receive_shadow = true;
    ground.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    scene.add(ground);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 3.0, 4.0);
    camera.look_at(Vector3::ZERO);

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
