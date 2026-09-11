//! Interactive Kirigami Expanded Miura viewer.
//!
//! Keys 1–9 / 0 jump to a variant; space or arrows cycle all 14 (13 presets + net).
//! Left-drag orbit · right-drag pan · scroll zoom.
//!
//! ```text
//! cargo run --release --example kirigami_orbit
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, KirigamiMesh, KirigamiPreset,
    KIRIGAMI_NET_VARIANT, LineBasicMaterial, LineSegments, Mesh, Object3D, OrbitControls,
    PerspectiveCamera, PointerEvent, Renderer, Scene, SphereGeometry, StandardMaterial, Vector3,
};

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

const NX: usize = 5;
const THICK: f64 = 1.2;
const VARIANT_COUNT: u32 = KirigamiPreset::count() as u32 + 1;

fn variant_label(v: u32) -> String {
    if v == KIRIGAMI_NET_VARIANT {
        "developed net".to_string()
    } else if let Some(p) = KirigamiPreset::from_variant(v) {
        p.name().to_string()
    } else {
        format!("variant {v}")
    }
}

struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<PhysicalPosition<f64>>,
}

impl InputState {
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
            .with_title("threers — kirigami (1–9 / space)")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 800))
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
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
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
                label: Some("kirigami"),
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

    let mut variant: u32 = 0;
    let mut scene = build_scene(variant);
    let mut camera = PerspectiveCamera::new(
        35.0,
        config.width as f32 / config.height as f32,
        1.0,
        4000.0,
    );
    camera.position = Vector3::new(0.0, 90.0, 420.0);
    camera.look_at(Vector3::ZERO);
    let mut controls = OrbitControls::new(&camera);
    controls.max_distance = 5000.0;
    controls.min_distance = 40.0;
    let mut input = InputState {
        rotating: false,
        panning: false,
        last: None,
    };
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
                WindowEvent::KeyboardInput { event: key, .. } => {
                    if key.state != ElementState::Pressed {
                        return;
                    }
                    let next = match key.logical_key.as_ref() {
                        Key::Character("1") => Some(0),
                        Key::Character("2") => Some(1),
                        Key::Character("3") => Some(2),
                        Key::Character("4") => Some(3),
                        Key::Character("5") => Some(4),
                        Key::Character("6") => Some(5),
                        Key::Character("7") => Some(6),
                        Key::Character("8") => Some(7),
                        Key::Character("9") => Some(8),
                        Key::Character("0") => Some(9),
                        Key::Named(NamedKey::Space) | Key::Named(NamedKey::ArrowRight) => {
                            Some((variant + 1) % VARIANT_COUNT)
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            Some((variant + VARIANT_COUNT - 1) % VARIANT_COUNT)
                        }
                        _ => None,
                    };
                    if let Some(v) = next {
                        variant = v;
                        scene = build_scene(variant);
                        win.set_title(&format!(
                            "threers — kirigami · {} ({}/{})",
                            variant_label(variant),
                            variant + 1,
                            VARIANT_COUNT
                        ));
                    }
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

fn build_scene(variant: u32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.06, 0.07, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.6)
            .with_direction(Vector3::new(-0.35, -0.75, -0.5).normalize()),
    );
    scene.add_light(
        DirectionalLight::new(Color::new(0.5, 0.62, 0.95), 0.9)
            .with_direction(Vector3::new(0.6, 0.2, 0.7).normalize()),
    );

    let mut root = Object3D::group();
    root.quaternion = Euler::new(-0.35, 0.4, 0.08).to_quaternion();
    let root_id = scene.add(root);

    if variant == KIRIGAMI_NET_VARIANT {
        let net = KirigamiPreset::Planar
            .evaluate(NX, 6, 0.0)
            .develop_joined();
        let geom = net.to_geometry(0.9);
        let mut mat = StandardMaterial::new(Color::new(0.78, 0.82, 0.76));
        mat.metalness = 0.15;
        mat.roughness = 0.48;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(geom, mat.into()));
        plate.position = Vector3::new(-170.0, -50.0, 0.0);
        scene.add_to(root_id, plate);
        let edges = Object3D::line_segments(LineSegments::new(
            net.crease_geometry(),
            LineBasicMaterial::new(Color::new(0.12, 0.13, 0.16)).into(),
        ));
        scene.add_to(root_id, edges);
        return scene;
    }

    let preset = KirigamiPreset::from_variant(variant).unwrap_or(KirigamiPreset::Planar);
    let mesh = preset.evaluate(preset.default_nx(), preset.default_ny(), THICK);
    let center = center_of(&mesh);

    if preset.uses_cell_colours() {
        let mut mat = StandardMaterial::new(Color::WHITE);
        mat.metalness = 0.22;
        mat.roughness = 0.48;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(mesh.to_geometry_cells(), mat.into()));
        plate.position = center;
        scene.add_to(root_id, plate);
        let mut edges = Object3D::line_segments(LineSegments::new(
            mesh.crease_geometry(),
            LineBasicMaterial::new(Color::new(0.1, 0.11, 0.13)).into(),
        ));
        edges.position = center;
        scene.add_to(root_id, edges);
        let mut rivet_mat = StandardMaterial::new(Color::new(0.85, 0.78, 0.35));
        rivet_mat.metalness = 0.7;
        rivet_mat.roughness = 0.28;
        let ball = SphereGeometry::new(1.6, 10, 8);
        for p in mesh.rivet_points(1) {
            let mut s = Object3D::mesh(Mesh::new(ball.clone(), rivet_mat.clone().into()));
            s.position = Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32) + center;
            scene.add_to(root_id, s);
        }
    } else if preset.uses_lattice_core() {
        let mut mat = StandardMaterial::new(Color::new(0.72, 0.78, 0.82));
        mat.metalness = 0.32;
        mat.roughness = 0.44;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(preset.to_geometry(&mesh), mat.into()));
        plate.position = center;
        scene.add_to(root_id, plate);
    } else {
        let (bottom, top, inclined) = mesh.to_geometry_by_kind();
        for (geom, color, metal, rough) in [
            (bottom, Color::new(0.82, 0.78, 0.70), 0.35, 0.45),
            (top, Color::new(0.75, 0.82, 0.88), 0.40, 0.40),
            (inclined, Color::new(0.55, 0.72, 0.58), 0.20, 0.55),
        ] {
            let mut mat = StandardMaterial::new(color);
            mat.metalness = metal;
            mat.roughness = rough;
            mat.side = 2;
            let mut plate = Object3D::mesh(Mesh::new(geom, mat.into()));
            plate.position = center;
            scene.add_to(root_id, plate);
        }
    }
    scene
}

fn center_of(mesh: &KirigamiMesh) -> Vector3 {
    let (min, max) = mesh.aabb();
    Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32,
        -0.5 * (min[2] + max[2]) as f32,
    )
}
