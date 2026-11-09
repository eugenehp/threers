//! Parity scene `physical` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-physical.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example physical
//! ```

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, Color, DirectionalLight, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, Scene, SphereGeometry, Vector3,
};

use winit::event::{Event, WindowEvent};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — physical")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu by default; the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal`.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

        // --- scene (matches threers parity iframe) ---
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x0c0c10);
        scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.1));
        let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.5));
        dl.position = Vector3::new(2.0, 3.0, 4.0);
        scene.add(dl);
        let mat = {
            let mut m = PhysicalMaterial::new(Color::from_hex(0xe05050));
            m.roughness = 0.25;
            m.metalness = 0.3;
            m.clearcoat = 0.7;
            m.clearcoat_roughness = 0.15;
            Material::Physical(m)
        };
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 48, 24),
            mat,
        )));
        let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.2, 3.5);
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
