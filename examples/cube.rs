//! Spinning cube — the three.js "hello world", in Rust.
//!
//! ```text
//! cargo run --example cube
//! ```

use std::sync::Arc;
use std::time::Instant;

use threers::cameras::Camera;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera,
    Scene, StandardMaterial, Vector3,
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
            .with_title("threers — spinning cube")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 700))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    // Build scene.
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202030);

    let geom = BoxGeometry::new(1.0, 1.0, 1.0);
    // PBR: warm-orange dielectric with moderate roughness.
    let mat = StandardMaterial::new(Color::from_hex(0xff6633))
        .with_roughness(0.4)
        .with_metalness(0.1);
    let cube_obj = Object3D::mesh(Mesh::new(geom, mat.into()));
    let cube_id = scene.add(cube_obj);

    // Lights: a soft ambient fill + one directional key light.
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.25));
    let mut key = Object3D::light(
        DirectionalLight::new(Color::from_hex(0xffffff), 1.0)
            .with_direction(Vector3::new(-0.5, -1.0, -0.4)),
    );
    key.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(key);

    let mut camera =
        PerspectiveCamera::new(60.0, view.aspect(), 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);

    let start = Instant::now();
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
                        let t = start.elapsed().as_secs_f32();
                        if let Some(obj) = scene.get_mut(cube_id) {
                            obj.quaternion = Euler::new(t * 0.7, t * 1.1, 0.0).to_quaternion();
                        }

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
