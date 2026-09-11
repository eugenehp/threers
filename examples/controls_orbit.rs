//! Orbit a cube with mouse — native Rust controls demo.
//!
//! ```text
//! cargo run --example controls_orbit
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom
//!
//! See `docs/controls.md` for three.js / threers shim / TypeScript equivalents.

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Mesh, Object3D, OrbitControls,
    PerspectiveCamera, PointerEvent, Renderer, Scene, StandardMaterial, Vector3,
};

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<PhysicalPosition<f64>>,
}

impl InputState {
    fn new() -> Self {
        Self {
            rotating: false,
            panning: false,
            last: None,
        }
    }

    fn motion(&mut self, pos: PhysicalPosition<f64>) -> (f32, f32) {
        let (dx, dy) = match self.last {
            Some(last) => ((pos.x - last.x) as f32, (pos.y - last.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        (dx, dy)
    }
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — OrbitControls (Rust)")
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

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202030);

    let geom = BoxGeometry::new(1.0, 1.0, 1.0);
    let mat = StandardMaterial::new(Color::from_hex(0xff6633))
        .with_roughness(0.4)
        .with_metalness(0.1);
    scene.add(Object3D::mesh(Mesh::new(geom, mat.into())));

    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    key.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(key);

    let mut camera =
        PerspectiveCamera::new(60.0, config.width as f32 / config.height as f32, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);

    let mut controls = OrbitControls::new(&camera);
    let mut input = InputState::new();
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
                    WindowEvent::MouseInput { state, button, .. } => {
                        let pressed = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => input.rotating = pressed,
                            MouseButton::Right => input.panning = pressed,
                            _ => {}
                        }
                        if !pressed {
                            input.last = None;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if !input.rotating && !input.panning {
                            return;
                        }
                        let (dx, dy) = input.motion(position);
                        let ev = PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                        };
                        controls.update(
                            ev,
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let wheel = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32,
                        };
                        let ev = PointerEvent {
                            dx: 0.0,
                            dy: 0.0,
                            wheel,
                            rotating: false,
                            panning: false,
                        };
                        controls.update(
                            ev,
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                    }
                    WindowEvent::RedrawRequested => {
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
