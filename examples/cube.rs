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
    Renderer, Scene, StandardMaterial, Vector3,
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
            .with_title("threers — spinning cube")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 700))
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

    // Downlevel defaults allow 16 sampled textures and 16 samplers per stage,
    // and the main pass declares 20 of each in the fragment stage — material
    // maps, the shadow atlases and the environment all bind there. Without
    // raising these two, `Renderer::new` fails wgpu 30's pipeline-layout
    // validation outright. `HeadlessRenderer` raises the same pair.
    let adapter_limits = adapter.limits();
    // `using_resolution` for the texture size, too: the renderer allocates a
    // 4096 shadow map and downlevel caps 2D textures at 2048.
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    limits.max_sampled_textures_per_shader_stage =
        adapter_limits.max_sampled_textures_per_shader_stage;
    limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            },
        )
        .await
        .expect("device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
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
        PerspectiveCamera::new(60.0, config.width as f32 / config.height as f32, 0.1, 100.0);
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
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        renderer.resize(config.width, config.height);
                        camera.set_aspect(config.width as f32 / config.height as f32);
                    }
                    WindowEvent::RedrawRequested => {
                        let t = start.elapsed().as_secs_f32();
                        if let Some(obj) = scene.get_mut(cube_id) {
                            obj.quaternion = Euler::new(t * 0.7, t * 1.1, 0.0).to_quaternion();
                        }

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
                            e => log::error!("surface error: {e:?}"),
                        }

                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .expect("event loop");
}
