//! Parity scene `lambert` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-lambert.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example lambert
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight,
    BoxGeometry,
    Color,
    DirectionalLight,
    Euler,
    LambertMaterial,
    Mesh,
    Object3D,
    PerspectiveCamera,
    Renderer,
    Scene,
    Vector3,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::ActiveEventLoop,
};


fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — lambert")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    threers::sketch::run_window(attributes, |window| pollster::block_on(setup(window)));
}

/// The scene for a window that has just opened, and the handler that draws it.
async fn setup(window: Arc<winit::window::Window>) -> impl FnMut(Event<()>, &ActiveEventLoop) {

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).expect("surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(device.clone(), queue.clone(), format, config.width, config.height);

    // --- scene (matches threers parity iframe) ---
    // const canvas = document.getElementById('c');
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x141420);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(2.0, 3.0, 4.0);
    scene.add(dl);
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.3, 1.3, 1.3), LambertMaterial::new(Color::from_hex(0x60ccaa)).into()));
    cube.quaternion = Euler::new(0.4, 0.6, 0.1).to_quaternion();
    scene.add(cube);
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 4.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    scene.update_world();
    let window_for_loop = window.clone();

    move |event, target| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                    match event {
                        WindowEvent::CloseRequested => target.exit(),
                        WindowEvent::Resized(new_size) => {
                            config.width = new_size.width.max(1);
                            config.height = new_size.height.max(1);
                            surface.configure(&device, &config);
                            renderer.resize(config.width, config.height);
                            camera.set_aspect(config.width as f32 / config.height as f32);
                        }
                        WindowEvent::RedrawRequested => {
                            scene.update_world();
                            match surface.get_current_texture() {
                                Ok(frame) => {
                                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                    renderer.render(&mut scene, &camera, &view, false);
                                    frame.present();
                                }
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    surface.configure(&device, &config);
                                }
                                Err(e) => log::error!("surface: {e:?}"),
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => window_for_loop.request_redraw(),
                _ => {}
            }
        }
}
