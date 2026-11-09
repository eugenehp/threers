//! Parent/child scene-graph demo: a group rotates, and child meshes inherit
//! its transform — the same demonstration three.js's `Group` docs use.

use std::time::Instant;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    BasicMaterial, BoxGeometry, Color, Euler, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, Vector3,
};

use winit::event::{Event, WindowEvent};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — scene graph")
        .with_inner_size(winit::dpi::LogicalSize::new(900, 700));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu by default; the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal`.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x101018);

        // A rotating group at the origin holding two children.
        let group_id = scene.add(Object3D::group());

        let mut left = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.8, 0.8, 0.8),
            BasicMaterial::new(Color::from_hex(0x44aaff)).into(),
        ));
        left.position = Vector3::new(-1.5, 0.0, 0.0);
        scene.add_to(group_id, left);

        let mut right = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.5, 24, 16),
            BasicMaterial::new(Color::from_hex(0xff6688)).into(),
        ));
        right.position = Vector3::new(1.5, 0.0, 0.0);
        scene.add_to(group_id, right);

        let mut camera = PerspectiveCamera::new(60.0, view.aspect(), 0.1, 100.0);
        camera.position = Vector3::new(0.0, 1.5, 5.0);
        camera.look_at(Vector3::ZERO);

        let start = Instant::now();
        let win = window.clone();

        move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == win.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    view.resize(s.width, s.height);
                    camera.set_aspect(view.aspect());
                }
                WindowEvent::RedrawRequested => {
                    let t = start.elapsed().as_secs_f32();
                    if let Some(g) = scene.get_mut(group_id) {
                        g.quaternion = Euler::new(0.0, t * 0.8, 0.0).to_quaternion();
                    }

                    view.render(&mut scene, &camera);
                    win.request_redraw();
                }
                _ => {}
            },
            _ => {}
        }
    });
}
