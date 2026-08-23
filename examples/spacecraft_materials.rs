//! An orbiting spacecraft built entirely from the material presets: a gold-MLI
//! bus, silver-foil thruster module, two solar-array wings, a brushed-aluminium
//! truss, a heat-tinted titanium engine bell, and a glass sensor porthole.
//!
//! ```text
//! cargo run --example spacecraft_materials
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom
//!
//! Everything is procedural — no HDRI or texture downloads needed. See
//! `material_chart` for a side-by-side contact sheet of every preset.
//!
//! # The one setup step that matters
//!
//! `scene.environment` is not optional here. A `metalness = 1.0` surface has no
//! diffuse term, so with no environment every metal below would render
//! essentially black regardless of how many lights are in the scene. The
//! `PmremGenerator::generate_pmrem` call inside `orbital_environment` is what
//! makes roughness mean anything: it prefilters the cubemap into a mip chain
//! that rough metals sample at blurrier levels.
//!
//! # A note on exposure
//!
//! `Renderer` applies no tone mapping (that lives in `ToneMappingPass`), so
//! anything above 1.0 hard-clips to white and loses its hue. The light
//! intensities here are set to keep the metal highlights just under that.

use std::f32::consts::{PI, TAU};
use threers::cameras::Camera;
use threers::materials::presets;
use threers::{
    AmbientLight, BoxGeometry, CylinderGeometry, DirectionalLight, Material, Mesh, Object3D,
    OrbitControls, PerspectiveCamera, PhysicalMaterial, PointerEvent, Renderer, Scene,
    SphereGeometry, TorusGeometry, Vector3,
};

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// Brings in `Arc`-wrapped procedural textures, `orbital_environment`, `SUN_DIR`.
include!("spacecraft_common.inc");

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

include!("spacecraft_build.inc");

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — spacecraft materials")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 750))
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
                label: Some("threers device"),
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

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x03040a);

    // The load-bearing line: prefiltered orbital IBL. Remove it and every metal
    // below turns black.
    scene.environment = Some(orbital_environment(SUN_DIR));

    build_spacecraft(&mut scene);

    scene.add_light(AmbientLight::new(Color::from_hex(0x223044), 0.06));

    let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff6e8), 2.4));
    sun.position = Vector3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) * 12.0;
    sun.cast_shadow = true;
    scene.add(sun);

    let mut bounce = Object3D::light(DirectionalLight::new(earth_bounce(), 0.5));
    bounce.position = Vector3::new(-0.3, -1.0, 0.2) * 12.0;
    scene.add(bounce);

    let mut camera =
        PerspectiveCamera::new(42.0, config.width as f32 / config.height as f32, 0.1, 200.0);
    camera.position = Vector3::new(4.2, 2.4, 6.0);
    camera.look_at(Vector3::ZERO);

    let mut controls = OrbitControls::new(&camera);
    let mut input = InputState::new();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent { event, window_id } = event {
                if window_id != window_for_loop.id() {
                    return;
                }
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
                        controls.update(
                            PointerEvent {
                                dx,
                                dy,
                                wheel: 0.0,
                                rotating: input.rotating,
                                panning: input.panning,
                            },
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let wheel = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32,
                        };
                        controls.update(
                            PointerEvent {
                                dx: 0.0,
                                dy: 0.0,
                                wheel,
                                rotating: false,
                                panning: false,
                            },
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
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
                            e => log::error!("surface error: {e:?}"),
                        }
                        window_for_loop.request_redraw();
                    }
                    _ => {}
                }
            }
        })
        .expect("event loop");
}
