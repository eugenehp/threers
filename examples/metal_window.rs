//! A window driven by the Metal backend.
//!
//! ```sh
//! cargo run --release --features metal --example metal_window
//! cargo run --release --features metal --example metal_window -- --frames 120
//! ```
//!
//! winit opens the window and owns the event loop; everything inside it is
//! `src/metal`. `--frames N` renders N frames and exits, which is what makes
//! this runnable from a script.

use std::f32::consts::TAU;

use threers::cameras::Camera;
use threers::core::{Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry, TorusKnotGeometry};
use threers::lights::{AmbientLight, DirectionalLight, PointLight};
use threers::materials::{Material, StandardMaterial};
use threers::math::{Color, Quaternion, Vector3};
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface};
use threers::scene::Scene;
use threers::PerspectiveCamera;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::WindowBuilder;

fn main() {
    // `--frames N` exits after N frames; without it the window stays open.
    let mut args = std::env::args().skip(1);
    let mut frame_limit: Option<u32> = None;
    while let Some(arg) = args.next() {
        if arg == "--frames" {
            frame_limit = args.next().and_then(|n| n.parse().ok());
        }
    }

    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("threers — Metal")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)
        .expect("window");

    let scale = window.scale_factor();
    let physical = window.inner_size();
    let ns_view = match window.window_handle().expect("window handle").as_raw() {
        RawWindowHandle::AppKit(handle) => handle.ns_view.as_ptr(),
        other => panic!("expected an AppKit window handle, got {other:?}"),
    };

    let device = MetalDevice::new().expect("Metal device");
    println!("device: {}", device.name());
    let mut renderer = MetalRenderer::with_device(device.clone()).expect("renderer");
    // Safety: the handle came from a live winit window, and winit runs its
    // event loop — and this code — on the main thread.
    let mut surface =
        unsafe { MetalSurface::from_ns_view(&device, ns_view, physical.width, physical.height, 4) }
            .expect("CAMetalLayer");
    println!(
        "surface: {}x{} at {scale}x, vsync {}",
        surface.size().0,
        surface.size().1,
        surface.vsync()
    );

    let (mut scene, knot) = build_scene();
    let mut camera = PerspectiveCamera::new(
        50.0,
        physical.width as f32 / physical.height.max(1) as f32,
        0.1,
        100.0,
    );
    camera.position = Vector3::new(0.0, 1.6, 5.5);
    camera.look_at(Vector3::new(0.0, 0.2, 0.0));

    let start = std::time::Instant::now();
    let mut frames = 0u32;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    surface.resize(size.width, size.height).expect("resize");
                    camera.set_aspect(size.width as f32 / size.height.max(1) as f32);
                }
                WindowEvent::RedrawRequested => {
                    let t = start.elapsed().as_secs_f32();
                    if let Some(obj) = scene.get_mut(knot) {
                        obj.quaternion = Quaternion::from_axis_angle(
                            Vector3::new(0.3, 1.0, 0.1).normalize(),
                            t * 0.7,
                        );
                    }
                    camera.position = Vector3::new(
                        (t * 0.25 * TAU / 4.0).sin() * 5.5,
                        1.6,
                        (t * 0.25 * TAU / 4.0).cos() * 5.5,
                    );

                    // `None` means every drawable is in flight; skipping the
                    // frame is the correct response, not waiting on one.
                    if let Some(frame) = surface.next_frame() {
                        renderer
                            .render(&mut scene, &camera, &frame.attachments())
                            .expect("render");
                        frames += 1;
                    }
                    if let Some(limit) = frame_limit {
                        if frames >= limit {
                            let fps = frames as f32 / start.elapsed().as_secs_f32();
                            println!("{frames} frames in {:.2?} ({fps:.1} fps)", start.elapsed());
                            target.exit();
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .expect("event loop");
}

fn build_scene() -> (Scene, threers::core::ObjectId) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10141c);

    let knot = scene.add(Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(1.0, 0.32, 160, 24, 2, 3),
        Material::Standard(
            StandardMaterial::new(Color::from_hex(0xff6f3c))
                .with_roughness(0.28)
                .with_metalness(0.1),
        ),
    )));

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(30.0, 30.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0x1b2430)).with_roughness(0.9)),
    ));
    floor.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    floor.position = Vector3::new(0.0, -1.5, 0.0);
    scene.add(floor);

    for (i, x) in [-3.0f32, 3.0].into_iter().enumerate() {
        let mut cube = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.8, 0.8, 0.8),
            Material::Standard(
                StandardMaterial::new(if i == 0 {
                    Color::from_hex(0x4fc3f7)
                } else {
                    Color::from_hex(0x81c784)
                })
                .with_roughness(0.4),
            ),
        ));
        cube.position = Vector3::new(x, -1.1, 0.0);
        scene.add(cube);
    }

    scene.add_light(AmbientLight::new(Color::from_hex(0x24384f), 1.0));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff0dd), 2.4));
    key.position = Vector3::new(3.0, 5.0, 4.0);
    scene.add(key);
    let mut fill = Object3D::light(PointLight::new(Color::from_hex(0x5c9dff), 20.0));
    fill.position = Vector3::new(-3.5, 1.0, -2.0);
    scene.add(fill);

    (scene, knot)
}
