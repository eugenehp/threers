//! A viewport the way a Tauri or Wry app would host one: the window is the
//! app's, the renderer is made on the main thread and drawn on a thread of
//! its own, its pipelines are built lazily and warmed on worker threads
//! before the first frame, and they are kept in a pipeline cache for the next
//! launch (Vulkan).
//!
//! winit stands in for tao here; anything with raw-window-handle 0.6 window
//! and display handles does.
//!
//! ```text
//! cargo run --release --example surface_viewport
//! FRAMES=120 cargo run --release --example surface_viewport   # exit after 120 frames
//! ```
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use threers::cameras::Camera as _;
use threers::renderer::{SurfaceGpu, SurfaceOptions, SurfaceRenderer};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera, Scene,
    StandardMaterial, Vector3,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoopBuilder;
use winit::window::WindowBuilder;

fn main() {
    env_logger::init();
    let frames: Option<u64> = std::env::var("FRAMES").ok().and_then(|v| v.parse().ok());
    let event_loop = EventLoopBuilder::<()>::with_user_event().build().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — surface viewport")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 700))
            .build(&event_loop)
            .expect("window"),
    );

    let t = Instant::now();
    let cache_dir = std::env::temp_dir().join("threers-surface-viewport-cache");
    // Any `PipelineCacheStore`: a directory here; `MemoryStore`, `FileStore`
    // or a `CallbackStore` into the app's own storage work the same way.
    let gpu = SurfaceGpu::new().expect("a GPU").with_pipeline_cache(cache_dir);
    println!("device: {:.1} ms ({})", ms(t), gpu.context().describe());

    // On the main thread: macOS requires it for the surface.
    let size = window.inner_size();
    let t = Instant::now();
    let mut view = SurfaceRenderer::new(&gpu, window.clone(), size.width, size.height, SurfaceOptions::default().lazy(true))
        .expect("a surface on the window");
    println!("viewport: {:.1} ms ({:?})", ms(t), view.format());

    // Warm on worker threads; an app would show its loading state meanwhile.
    let t = Instant::now();
    let warmup = view.renderer().warm_up();
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let w = warmup.clone();
            std::thread::spawn(move || w.run())
        })
        .collect();
    warmup.wait();
    workers.into_iter().for_each(|w| drop(w.join()));
    println!("warm-up: {} pipelines in {:.1} ms, {} errors", warmup.len(), ms(t), warmup.errors().len());
    let (mut scene, mut camera, cube) = build_scene(size.width as f32 / size.height.max(1) as f32);
    let t = Instant::now();
    view.renderer().warm_scene(&mut scene, &camera);
    println!("scene warm-up: {:.1} ms", ms(t));
    match gpu.save_pipeline_cache() {
        Ok(true) => println!("pipeline cache saved ({})", gpu.pipeline_cache().unwrap().key()),
        Ok(false) => println!("no pipeline cache on this backend"),
        Err(e) => println!("pipeline cache not saved: {e}"),
    }

    // The renderer moves to its own thread; the event loop only forwards sizes.
    let pending_size = Arc::new(Mutex::new(None::<(u32, u32)>));
    let drawn = Arc::new(AtomicU64::new(0));
    let proxy = event_loop.create_proxy();
    {
        let pending_size = pending_size.clone();
        let drawn = drawn.clone();
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut first = true;
            loop {
                if let Some((w, h)) = pending_size.lock().unwrap().take() {
                    view.resize(w, h);
                    camera.set_aspect(w as f32 / h.max(1) as f32);
                }
                let s = start.elapsed().as_secs_f32();
                if let Some(obj) = scene.get_mut(cube) {
                    obj.quaternion = Euler::new(s * 0.7, s * 1.1, 0.0).to_quaternion();
                }
                let t = Instant::now();
                if view.render(&mut scene, &camera) {
                    if first {
                        println!("first frame: {:.1} ms", ms(t));
                        first = false;
                    }
                    let n = drawn.fetch_add(1, Ordering::Relaxed) + 1;
                    if frames == Some(n) {
                        println!("{n} frames in {:.2} s", start.elapsed().as_secs_f64());
                        let _ = proxy.send_event(());
                        return;
                    }
                }
            }
        });
    }

    let id = window.id();
    event_loop
        .run(move |event, target| match event {
            Event::UserEvent(()) => target.exit(),
            Event::WindowEvent { event, window_id } if window_id == id => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => *pending_size.lock().unwrap() = Some((s.width, s.height)),
                _ => {}
            },
            _ => {}
        })
        .expect("event loop");
}

fn build_scene(aspect: f32) -> (Scene, PerspectiveCamera, threers::ObjectId) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x202030);
    let mat = StandardMaterial::new(Color::from_hex(0xff6633)).with_roughness(0.4).with_metalness(0.1);
    let cube = scene.add(Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), mat.into())));
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.25));
    let mut key = Object3D::light(
        DirectionalLight::new(Color::from_hex(0xffffff), 1.0).with_direction(Vector3::new(-0.5, -1.0, -0.4)),
    );
    key.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(key);
    let mut camera = PerspectiveCamera::new(60.0, aspect, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);
    (scene, camera, cube)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
