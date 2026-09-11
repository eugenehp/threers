//! Run one hierarchy CSG step in isolation (native wgpu).
//!
//! | Step | Command | Mode | JS target |
//! |------|---------|------|-----------|
//! | s1 shell cut | `cargo run --example bvh_csg_steps --features bvh-csg -- 1` | live | 72 |
//! | s2 + sphere | `... -- 2` | live | 53070 |
//! | s3 + winCut | `... -- 3` | ref-sphere (default) | 55914 |
//! | s3 + winCut | `... -- 3 --live` | live chain | 55914 |
//! | s4 + winFrame | `... -- 4` | ref-sphere (default) | 59454 |
//! | s4 + winFrame | `... -- 4 --live` | live chain | 59454 |

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    evaluate_live_through, evaluate_through, js_target_verts, AmbientLight, Color, CsgEvaluator,
    DirectionalLight, Mesh, Object3D, PerspectiveCamera, Renderer, Scene, StandardMaterial,
    Vector3,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

const AFTER_SPHERE_BIN: &[u8] =
    include_bytes!("../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
const AFTER_WINCUT_BIN: &[u8] =
    include_bytes!("../tests/parity/scenes/rust/bvh-csg-step3-wincut.geom.bin");

fn parse_step() -> (u8, bool, bool) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let step: u8 = args
        .first()
        .and_then(|s| s.parse().ok())
        .filter(|&n| (1..=4).contains(&n))
        .unwrap_or_else(|| {
            eprintln!("usage: bvh_csg_steps <1|2|3|4> [--live] [--ref-wincut]");
            eprintln!("  1 shell cut | 2 +sphere | 3 +winCut | 4 +winFrame");
            std::process::exit(1);
        });
    let live = args.iter().any(|a| a == "--live");
    let ref_wincut = args.iter().any(|a| a == "--ref-wincut");
    (step, live, ref_wincut)
}

fn build_geometry(step: u8, live: bool, ref_wincut: bool) -> threers::BufferGeometry {
    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = false;
    if step == 4 && ref_wincut {
        let mut after_cut = threers::load_positions_geometry_bin(AFTER_WINCUT_BIN);
        threers::utils::compute_vertex_normals(&mut after_cut);
        return threers::step4_win_frame(&mut evaluator, after_cut);
    }
    if live || step <= 2 {
        evaluate_live_through(step, &mut evaluator)
    } else {
        evaluate_through(step, &mut evaluator, Some(AFTER_SPHERE_BIN))
    }
}

fn main() {
    env_logger::init();
    let (step, live, ref_wincut) = parse_step();
    pollster::block_on(run(step, live, ref_wincut));
}

async fn run(step: u8, live: bool, ref_wincut: bool) {
    let geom = build_geometry(step, live, ref_wincut);
    let verts = geom
        .get_attribute("position")
        .map(|a| a.count())
        .unwrap_or(0);
    let target = js_target_verts(step);
    let mode = if ref_wincut {
        "ref-wincut"
    } else if live || step <= 2 {
        "live"
    } else {
        "ref-sphere"
    };

    let event_loop = EventLoop::new().expect("event loop");
    let title = format!("threers CSG step {step} ({mode}) — {verts} verts (JS {target})");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(&title)
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
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
                label: Some("threers"),
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

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.45));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(dl);

    let mat = StandardMaterial::new(Color::from_hex(0x4488cc))
        .with_roughness(0.45)
        .with_metalness(0.1);
    scene.add(Object3D::mesh(Mesh::new(geom, mat.into())));

    let mut camera = PerspectiveCamera::new(50.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(-3.8, 2.9, 1.25);
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
