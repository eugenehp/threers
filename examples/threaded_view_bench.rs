//! What drawing the 3D view on its own thread does for the UI's frames.
//!
//! A UI of a few hundred text rows (one a readout changing every frame) and a
//! 3D box, drawn offscreen at 60 fps
//! for a few seconds in three ways:
//!
//! - `serial`: each UI frame first renders the 3D view on the same thread.
//! - `threaded`: the 3D view renders continuously on a thread of its own, and
//!   the UI shows its newest frame ([`ViewProducer`], [`SurfaceView`]).
//! - `adaptive`: threaded, with the view's resolution chosen to keep its GPU
//!   frames within 6 ms ([`AdaptiveScale`]).
//!
//! For each it prints, in milliseconds, how far apart UI frames finished
//! (`interval`), the UI thread's time to record and submit a frame (`cpu`),
//! and from the start of a UI frame until the GPU finished it (`done`). The
//! last includes waiting behind 3D work already on the queue, which threads
//! do not remove: both share one GPU.
//!
//! ```text
//! cargo run --release --features ui --example threaded_view_bench [knots] [seconds]
//! ```

use std::time::{Duration, Instant};

use threers::cameras::Camera;
use threers::gpu::GpuContext;
use threers::renderer::{AdaptiveScale, FramePacer, RendererOptions, UiPass, ViewProducer};
use threers::style;
use threers::ui::surface_view::SurfaceView;
use threers::ui::Ui;
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera, Scene,
    StandardMaterial, TorusKnotGeometry, Vector3,
};

const W: u32 = 1600;
const H: u32 = 1000;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The 3D view's GPU budget in `adaptive`: tight, to leave the UI's frames
/// less 3D work to wait behind.
const ADAPTIVE_BUDGET_MS: f32 = 6.0;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Serial,
    Threaded,
    Adaptive,
}

/// The heavy part: `knots` high-detail torus knots, turning.
fn heavy_scene(knots: usize) -> (Scene, Vec<threers::core::ObjectId>, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.2));
    key.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(key);
    let side = (knots as f32).sqrt().ceil() as usize;
    let ids = (0..knots)
        .map(|i| {
            let mut o = Object3D::mesh(Mesh::new(
                TorusKnotGeometry::new(0.35, 0.12, 512, 64, 2, 3),
                StandardMaterial::new(Color::from_hex(0xd0763a))
                    .with_roughness(0.35)
                    .into(),
            ));
            let (x, y) = ((i % side) as f32, (i / side) as f32);
            let half = side as f32 / 2.0;
            o.position = Vector3::new(x - half + 0.5, y - half + 0.5, 0.0);
            scene.add(o)
        })
        .collect();
    let mut camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 200.0);
    camera.position = Vector3::new(0.0, 0.0, side as f32 * 1.3 + 2.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));
    (scene, ids, camera)
}

fn turn(scene: &mut Scene, ids: &[threers::core::ObjectId], t: f32) {
    for (i, id) in ids.iter().enumerate() {
        if let Some(o) = scene.get_mut(*id) {
            o.quaternion = Euler::new(t * 0.5 + i as f32, t * 0.8, 0.0).to_quaternion();
        }
    }
}

struct Stats {
    interval: Vec<f32>,
    cpu: Vec<f32>,
    done: Vec<f32>,
    view_frames: u64,
    final_scale: f32,
    /// The producer's own measure of a 3D frame's cost, at the end.
    view_ms: f32,
}

fn run(gpu: &GpuContext, mode: Mode, knots: usize, seconds: f32) -> Stats {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let (mut producer, frames) = ViewProducer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        RendererOptions::default(),
        None,
    );
    producer.renderer().set_msaa(4);
    if mode == Mode::Adaptive {
        producer.set_adaptive_scale(Some(AdaptiveScale::new(ADAPTIVE_BUDGET_MS)));
    }

    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; flex-direction: row; background: #12151b; color: #d6dae3; font-size: 13px; },
    );
    let side = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; gap: 2px; padding: 8px; width: 320px; overflow: hidden; },
    );
    let readout = ui.add_text(side, "");
    for i in 0..300 {
        ui.add_text(side, format!("row {i}"));
    }
    let mut view = SurfaceView::new(&mut ui, root, style! { flex-grow: 1; margin: 8px; }, frames);
    let mut pass = UiPass::new(&gpu.device);

    let start = Instant::now();
    let (render, mut serial) = if mode == Mode::Serial {
        (None, Some((producer, heavy_scene(knots))))
    } else {
        let render = std::thread::spawn(move || {
            let (mut scene, ids, mut camera) = heavy_scene(knots);
            while !producer.is_closed() {
                turn(&mut scene, &ids, start.elapsed().as_secs_f32());
                camera.set_aspect(producer.aspect());
                if !producer.render(&mut scene, &camera) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            (producer.render_scale(), producer.frame_ms())
        });
        (Some(render), None)
    };

    let mut stats = Stats {
        interval: vec![],
        cpu: vec![],
        done: vec![],
        view_frames: 0,
        final_scale: 1.0,
        view_ms: 0.0,
    };
    let mut pacer = FramePacer::new(60.0);
    let mut last_done: Option<Instant> = None;
    let mut n = 0u64;
    while start.elapsed().as_secs_f32() < seconds {
        pacer.wait();
        let t0 = Instant::now();
        n += 1;
        ui.set_text(readout, format!("frame {n}"));
        if let Some((producer, (scene, ids, camera))) = &mut serial {
            turn(scene, ids, start.elapsed().as_secs_f32());
            camera.set_aspect(producer.aspect());
            producer.render(scene, camera);
        }
        ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
        view.update(&ui, &mut pass, 1.0);
        pass.draw(&gpu.device, &gpu.queue, &ui.scene(), 1.0, &target_view, (W, H), FORMAT);
        let index = gpu.queue.submit([]);
        let cpu = t0.elapsed();
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("poll");
        let done = Instant::now();
        // The first second warms pipelines and caches; leave it out.
        if start.elapsed().as_secs_f32() > 1.0 {
            stats.cpu.push(cpu.as_secs_f32() * 1000.0);
            stats.done.push((done - t0).as_secs_f32() * 1000.0);
            if let Some(last) = last_done {
                stats.interval.push((done - last).as_secs_f32() * 1000.0);
            }
        }
        last_done = Some(done);
    }
    stats.view_frames = view.frames().published();
    if let Some((producer, _)) = &serial {
        (stats.final_scale, stats.view_ms) = (producer.render_scale(), producer.frame_ms());
    }
    drop(view);
    if let Some(r) = render {
        (stats.final_scale, stats.view_ms) = r.join().expect("render thread");
    }
    let _ = serial.take();
    stats
}

fn summary(v: &mut [f32]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(f32::total_cmp);
    let q = |p: f32| v[((v.len() - 1) as f32 * p).round() as usize];
    format!("p50 {:6.1}  p95 {:6.1}  max {:6.1}", q(0.5), q(0.95), q(1.0))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let knots: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(64);
    let seconds: f32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(5.0);
    let gpu = GpuContext::headless_blocking().expect("a GPU");
    println!(
        "{} — {knots} knots, {W}×{H}, UI capped at 60 fps, {seconds} s each (first second dropped)\n",
        gpu.adapter.get_info().name
    );
    for (mode, name) in [
        (Mode::Serial, "serial"),
        (Mode::Threaded, "threaded"),
        (Mode::Adaptive, "adaptive"),
    ] {
        let mut s = run(&gpu, mode, knots, seconds);
        let ui_fps = (s.interval.len() + 1) as f32 / (seconds - 1.0);
        println!(
            "{name:9} UI {ui_fps:5.1} fps, 3D {:5.1} fps at scale {:.3}, {:.1} ms GPU a 3D frame",
            s.view_frames as f32 / seconds,
            s.final_scale,
            s.view_ms
        );
        println!("  interval  {}", summary(&mut s.interval));
        println!("  cpu       {}", summary(&mut s.cpu));
        println!("  done      {}\n", summary(&mut s.done));
    }
}
