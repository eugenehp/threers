//! The bvh-csg hierarchy parity scene, evaluated live and drawn with wgpu: a
//! shell (box minus a smaller box), a sphere added on, and a window cut and
//! framed through an operation group — three-bvh-csg's `evaluateHierarchy`.
//!
//! An optional step, 1 to 4, applies only that many of the root's
//! operations, to watch the hierarchy build up. Each prints its triangle
//! count beside three-bvh-csg 0.0.16's (`tests/csg_upstream.rs` holds them
//! equal).
//!
//! ```text
//! cargo run --example bvh_csg_hierarchy --features bvh-csg [-- 1|2|3|4]
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    evaluate_hierarchy, AmbientLight, BoxGeometry, BufferGeometry, Color, CsgEvaluator, CsgNode,
    CsgOperation, CsgOperationGroup, DirectionalLight, Mesh, Object3D, PerspectiveCamera, Renderer,
    Scene, SphereGeometry, StandardMaterial, Vector3, ADDITION, SUBTRACTION,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

/// three-bvh-csg's triangle count after each step.
const UPSTREAM_TRIANGLES: [usize; 4] = [24, 774, 804, 816];

/// The root with its first `steps` operations: the cut, the sphere, and the
/// window group's cut and frame.
fn build_hierarchy_geometry(steps: usize) -> BufferGeometry {
    let soup = |g: BufferGeometry| g.to_non_indexed();
    let mut root = CsgOperation::new(soup(BoxGeometry::new(4.0, 2.5, 2.5)), ADDITION);
    root.add_child(CsgNode::Operation(CsgOperation::new(soup(BoxGeometry::new(3.6, 2.1, 2.1)), SUBTRACTION)));
    if steps >= 2 {
        let mut sphere = CsgOperation::new(soup(SphereGeometry::new(0.55, 24, 12)), ADDITION);
        sphere.set_position(-1.1, 0.2, 1.35);
        root.add_child(CsgNode::Operation(sphere));
    }
    if steps >= 3 {
        let mut window = CsgOperationGroup::new();
        window.add_child(CsgNode::Operation(CsgOperation::new(soup(BoxGeometry::new(1.2, 1.0, 0.5)), SUBTRACTION)));
        if steps >= 4 {
            window.add_child(CsgNode::Operation(CsgOperation::new(soup(BoxGeometry::new(1.2, 1.0, 0.12)), ADDITION)));
        }
        window.set_position(0.8, 0.15, 1.35);
        root.add_child(CsgNode::Group(window));
    }
    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = false;
    let mut geometry = evaluate_hierarchy(&mut root, &mut evaluator);
    threers::utils::compute_vertex_normals(&mut geometry);
    geometry
}

fn main() {
    env_logger::init();
    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .filter(|n| (1..=4).contains(n))
        .unwrap_or(4);
    pollster::block_on(run(steps));
}

async fn run(steps: usize) {
    let geometry = build_hierarchy_geometry(steps);
    let triangles = geometry.get_attribute("position").map_or(0, |a| a.count() / 3);
    println!(
        "step {steps}: {triangles} triangles (three-bvh-csg {})",
        UPSTREAM_TRIANGLES[steps - 1]
    );
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(format!("threers — bvh-csg hierarchy, step {steps}: {triangles} triangles"))
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
    let mesh = Object3D::mesh(Mesh::new(geometry, mat.into()));
    scene.add(mesh);

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
