//! Parity scene `normal-map` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-normal-map.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example normal_map
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, Color, DataTexture, DirectionalLight, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, TextureFormat, Vector3,
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
            .with_title("threers — normal-map")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    // wgpu by default; the native Metal backend with `--features metal` and
    // `THREERS_BACKEND=metal`.
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    // --- scene (matches threers parity iframe) ---
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0a0a0a);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    // Procedural 16×16 tangent-space normal map (matches threers-normal-map.html).
    let tex_size: u32 = 16;
    let mut data = vec![0u8; (tex_size * tex_size * 4) as usize];
    for y in 0..tex_size {
        for x in 0..tex_size {
            let u = (x % 4) as f32 / 3.0 - 0.5;
            let v = (y % 4) as f32 / 3.0 - 0.5;
            let z = (1.0 - u * u - v * v).max(0.0).sqrt();
            let i = ((y * tex_size + x) * 4) as usize;
            data[i] = ((u * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 1] = ((v * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 2] = ((z * 0.5 + 0.5) * 255.0).round() as u8;
            data[i + 3] = 255;
        }
    }
    let normal_tex = Arc::new(DataTexture::new(
        tex_size,
        tex_size,
        TextureFormat::Rgba8Unorm,
        data,
    ));
    let mat = StandardMaterial::new(Color::from_hex(0xaaaaaa))
        .with_roughness(0.4)
        .with_metalness(0.0)
        .with_normal_map(normal_tex)
        .into();
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.1, 48, 24),
        mat,
    )));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.5);
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
