//! Parity scene `lathe` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-lathe.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example lathe
//! ```

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, LatheGeometry, Mesh, Object3D, PerspectiveCamera,
    Scene, StandardMaterial, Vector2, Vector3,
};

use winit::event::{Event, WindowEvent};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — lathe")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu by default; the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal`.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

        // --- scene (matches threers parity iframe) ---
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x1a1a22);
        scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
        let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
        dl.position = Vector3::new(2.0, 3.0, 4.0);
        scene.add(dl);
        let points = [
            Vector2::new(0.4, -1.0),
            Vector2::new(0.6, -0.5),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.3, 0.5),
            Vector2::new(0.5, 1.0),
        ];
        let mut lathe = Object3D::mesh(Mesh::new(
            LatheGeometry::new(&points, 16, 0.0, std::f32::consts::TAU),
            StandardMaterial::new(Color::from_hex(0xddaa66))
                .with_roughness(0.45)
                .with_metalness(0.0)
                .into(),
        ));
        lathe.quaternion = Euler::new(0.0, 0.4, 0.0).to_quaternion();
        scene.add(lathe);
        let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.3, 4.0);
        camera.look_at(Vector3::ZERO);

        scene.update_world();
        let window_for_loop = window.clone();

        move |event, target| match event {
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
        }
    });
}
