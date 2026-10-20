//! Parity scene `bvh-csg-hierarchy` — native Rust (winit + wgpu) with live CSG.
//!
//! ```text
//! cargo run --features bvh-csg --bin bvh-csg-hierarchy
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

/// The scene's hierarchy, as `threers-bvh-csg-hierarchy.html` builds it:
/// shell − cut + sphere, and a window cut and framed through a group.
fn build_hierarchy_geometry() -> BufferGeometry {
    let soup = |g: BufferGeometry| g.to_non_indexed();
    let mut root = CsgOperation::new(soup(BoxGeometry::new(4.0, 2.5, 2.5)), ADDITION);
    let cut = CsgOperation::new(soup(BoxGeometry::new(3.6, 2.1, 2.1)), SUBTRACTION);
    let mut sphere = CsgOperation::new(soup(SphereGeometry::new(0.55, 24, 12)), ADDITION);
    sphere.set_position(-1.1, 0.2, 1.35);
    let mut window = CsgOperationGroup::new();
    window.add_child(CsgNode::Operation(CsgOperation::new(soup(BoxGeometry::new(1.2, 1.0, 0.5)), SUBTRACTION)));
    window.add_child(CsgNode::Operation(CsgOperation::new(soup(BoxGeometry::new(1.2, 1.0, 0.12)), ADDITION)));
    window.set_position(0.8, 0.15, 1.35);
    root.add_child(CsgNode::Operation(cut));
    root.add_child(CsgNode::Operation(sphere));
    root.add_child(CsgNode::Group(window));
    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = false;
    let mut geometry = evaluate_hierarchy(&mut root, &mut evaluator);
    threers::utils::compute_vertex_normals(&mut geometry);
    geometry
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — bvh-csg-hierarchy (native CSG)")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("window"),
    );

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

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.45));
    let mut dl = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.0));
    dl.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(dl);

    let mat = StandardMaterial::new(Color::from_hex(0x4488cc))
        .with_roughness(0.45)
        .with_metalness(0.1);
    let mesh = Object3D::mesh(Mesh::new(build_hierarchy_geometry(), mat.into()));
    scene.add(mesh);

    let mut camera = PerspectiveCamera::new(50.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(-3.8, 2.9, 1.25);
    camera.look_at(Vector3::ZERO);

    scene.update_world();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| {
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
        })
        .expect("event loop");
}
