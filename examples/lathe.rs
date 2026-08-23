//! Parity scene `lathe` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-lathe.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example lathe
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, LatheGeometry, Mesh, Object3D, PerspectiveCamera,
    Renderer, Scene, StandardMaterial, Vector2, Vector3,
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
            .with_title("threers — lathe")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
        // wgpu 30 dropped `Default` here; the display handle is only
        // consulted by GLES/Wayland, not Vulkan, Metal or DX12.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        wgpu::Instance::new(d)
    };
    let surface = instance.create_surface(window.clone()).expect("surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )
        .await
        .expect("device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        format,
        config.width,
        config.height,
    );

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

    event_loop
        .run(move |event, target| match event {
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
                            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                                let view = frame
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());
                                renderer.render(&mut scene, &camera, &view, false);
                                queue.present(frame);
                            }
                            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                                surface.configure(&device, &config);
                            }
                            e => log::error!("surface: {e:?}"),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => window_for_loop.request_redraw(),
            _ => {}
        })
        .expect("event loop");
}
