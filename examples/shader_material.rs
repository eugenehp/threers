//! Custom-shader material demo — the extensibility escape hatch.
//!
//! A cube drawn with a [`ShaderMaterial`]: the fragment is user WGSL that reads
//! a user uniform (base color) and a user storage array (stripe frequency),
//! shaded against the built-in `frame` lights. Compiled against threers'
//! standard preamble, so `VsOut`, `frame`, and `framebuffer_encode` are free.
//!
//! ```text
//! cargo run --example shader_material
//! ```

use std::sync::Arc;
use std::time::Instant;

use threers::cameras::Camera;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera,
    Renderer, Scene, ShaderMaterial, Vector3,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

const FRAGMENT: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = u_data.data[0].rgb;         // user uniform: base color
    let freq = u_s0[0];                     // user storage: stripe frequency
    let stripe = 0.5 + 0.5 * sin(in.world_pos.y * freq);
    let tint = mix(base, in.world_normal * 0.5 + 0.5, stripe);

    var light = frame.ambient.rgb;
    for (var i: u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let dir = normalize(frame.dir_lights[i].direction.xyz);
        light = light + frame.dir_lights[i].color.rgb * max(dot(normalize(in.world_normal), -dir), 0.0);
    }
    return vec4<f32>(framebuffer_encode(tint * light), 1.0);
}
"#;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — shader material")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 700))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
        // wgpu 30 has no `Default` here. The display handle only
        // matters for GLES/Wayland presentation; Vulkan, Metal and
        // DX12 ignore it, and those are the backends asked for.
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
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
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

    // A cube with a custom-shader material.
    let mut data = vec![[0.0f32; 4]; 16];
    data[0] = [1.0, 0.45, 0.2, 0.0]; // base color
    let material = ShaderMaterial::new(FRAGMENT)
        .with_data(data)
        .with_storage0(vec![8.0]);
    let cube_id = scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        material.into(),
    )));

    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.3));
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
                    if let Some(obj) = scene.get_mut(cube_id) {
                        obj.quaternion = Euler::new(t * 0.6, t * 0.9, 0.0).to_quaternion();
                    }
                    if let wgpu::CurrentSurfaceTexture::Success(frame)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) =
                        surface.get_current_texture() {
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        renderer.render(&mut scene, &camera, &view, false);
                        queue.present(frame);
                    }
                    win.request_redraw();
                }
                _ => {}
            },
            _ => {}
        })
        .expect("event loop");
}
