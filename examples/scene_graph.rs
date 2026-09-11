//! Parent/child scene-graph demo: a group rotates, and child meshes inherit
//! its transform — the same demonstration three.js's `Group` docs use.

use std::sync::Arc;
use std::time::Instant;

use threers::cameras::Camera;
use threers::{
    BasicMaterial, BoxGeometry, Color, Euler, Mesh, Object3D, PerspectiveCamera, Renderer, Scene,
    SphereGeometry, Vector3,
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
            .with_title("threers — scene graph")
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

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
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

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        },
    ))
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

    let mut camera =
        PerspectiveCamera::new(60.0, config.width as f32 / config.height as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.5, 5.0);
    camera.look_at(Vector3::ZERO);

    let start = Instant::now();
    let win = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == win.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    config.width = s.width.max(1);
                    config.height = s.height.max(1);
                    surface.configure(&device, &config);
                    renderer.resize(config.width, config.height);
                    camera.set_aspect(config.width as f32 / config.height as f32);
                }
                WindowEvent::RedrawRequested => {
                    let t = start.elapsed().as_secs_f32();
                    if let Some(g) = scene.get_mut(group_id) {
                        g.quaternion = Euler::new(0.0, t * 0.8, 0.0).to_quaternion();
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
                    win.request_redraw();
                }
                _ => {}
            },
            _ => {}
        })
        .expect("event loop");
}
