//! Several views on one producer, the loop that schedules them, and the
//! thread that runs it: each view draws its own scene at its own size and
//! rate, on demand or continuously, not at all while hidden; the UI's commands
//! reach the render thread; a panic there is reported, not frozen; a shared
//! GPU budget takes resolution from the least important view first.
//!
//! Skipped when no GPU adapter is available.

use std::time::{Duration, Instant};

use threers::gpu::GpuContext;
use threers::renderer::{
    Redraw, RenderLoop, RenderThread, SharedGpu, ViewFrames, ViewId, ViewOptions, ViewProducer,
};
use threers::{Color, PerspectiveCamera, Scene};

fn gpu() -> Option<SharedGpu> {
    match GpuContext::headless_blocking() {
        Ok(g) => Some(SharedGpu::from(&g)),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn scene(hex: u32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(hex);
    scene
}

fn camera() -> PerspectiveCamera {
    PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0)
}

fn rgb(hex: u32) -> [u8; 3] {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
}

/// Every pixel of the frame within ±1 of `hex`.
fn all_near(px: &[u8], hex: u32) -> bool {
    px.chunks(4).all(|p| (0..3).all(|i| p[i].abs_diff(rgb(hex)[i]) <= 1))
}

#[test]
fn views_on_one_producer_draw_their_own_scene_at_their_own_size() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut first) = ViewProducer::on(&gpu, ViewOptions::default());
    let (b, mut second) = producer.add_view();
    let (c, mut third) = producer.add_view();
    assert_eq!(producer.view_ids().collect::<Vec<_>>(), [ViewId::PRIMARY, b, c]);
    let views = [(ViewId::PRIMARY, 0xff0000, (40, 30)), (b, 0x00ff00, (24, 56)), (c, 0x2050a0, (64, 16))];
    first.set_size(40, 30);
    second.set_size(24, 56);
    third.set_size(64, 16);
    // Rounds over all three, as a split view draws: sizes alternate on one
    // multisampled renderer.
    for round in 0..4 {
        for &(id, hex, _) in &views {
            assert!(producer.view(id).render(&mut scene(hex), &camera()), "round {round}");
        }
        for (frames, &(_, hex, size)) in [&mut first, &mut second, &mut third].into_iter().zip(&views) {
            let f = frames.latest().expect("a frame");
            assert!(f.new);
            assert_eq!(f.size, size);
            assert!(all_near(&f.read_rgba(&gpu.device, &gpu.queue), hex), "{hex:06x} in round {round}");
        }
    }
    // Dropping a view's frames stops it; the others go on.
    drop(second);
    assert_eq!(producer.view_ids().collect::<Vec<_>>(), [ViewId::PRIMARY, c]);
    assert!(!producer.view(b).render(&mut scene(0), &camera()));
    assert!(!producer.is_closed());
}

#[test]
fn a_ui_that_asks_before_submitting_never_sees_its_frame_drawn_over() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = ViewProducer::on(&gpu, ViewOptions::default().msaa(1));
    frames.set_size(4, 4);
    // The two textures the UI may still be reading: the one it shows, and the
    // one it showed before (asked for the next frame, not yet submitted).
    let mut held: Vec<wgpu::Texture> = Vec::new();
    for i in 0..40u32 {
        for _ in 0..(i % 3) {
            producer.render_with(|r, view, _| {
                assert!(!held.iter().any(|t| t == view.texture()), "drawing into a frame the UI holds");
                r.render(&mut scene(0x808080), &camera(), view, false);
            });
        }
        if let Some(f) = frames.latest()
            && f.new
        {
            held.push(f.texture);
            if held.len() > 2 {
                held.remove(0);
            }
        }
    }
}

/// A loop over a producer's views counting draws per view.
fn counting_loop(gpu: &SharedGpu) -> (RenderLoop<Vec<u32>>, ViewFrames) {
    let (producer, frames) = ViewProducer::on(gpu, ViewOptions::default().msaa(1));
    let render = RenderLoop::new(producer, Vec::new(), |counts: &mut Vec<u32>, view| {
        let i = view.id().index();
        if counts.len() <= i {
            counts.resize(i + 1, 0);
        }
        counts[i] += 1;
        view.render(&mut scene(0x101010), &camera());
    });
    (render, frames)
}

fn draws(l: &mut RenderLoop<Vec<u32>>, view: ViewId) -> u32 {
    l.state().get(view.index()).copied().unwrap_or(0)
}

#[test]
fn on_demand_views_draw_once_per_request_and_hidden_views_not_at_all() {
    let Some(gpu) = gpu() else { return };
    let (mut l, frames) = counting_loop(&gpu);
    let v = ViewId::PRIMARY;
    l.set_redraw(v, Redraw::OnDemand);
    l.set_fps(v, 0.0);
    // No size yet: nothing to draw, nothing due.
    l.producer().take_request(v);
    assert_eq!(l.step(Instant::now()), None);
    frames.set_size(8, 8);
    l.step(Instant::now());
    assert_eq!(draws(&mut l, v), 1, "the resize asked for a frame");
    assert_eq!(l.step(Instant::now()), None, "nothing asked: nothing due");
    assert_eq!(draws(&mut l, v), 1);
    frames.request_redraw();
    l.step(Instant::now());
    assert_eq!(draws(&mut l, v), 2);
    l.update(|_| {});
    l.step(Instant::now());
    assert_eq!(draws(&mut l, v), 3, "an update asks every view");

    // Hidden: neither continuous drawing nor requests draw it.
    l.set_redraw(v, Redraw::Continuous);
    frames.set_visible(false);
    for _ in 0..5 {
        l.step(Instant::now());
    }
    assert_eq!(draws(&mut l, v), 3);
    frames.set_visible(true);
    l.step(Instant::now());
    assert_eq!(draws(&mut l, v), 4);
    // Paused likewise.
    l.set_paused(v, true);
    assert_eq!(l.step(Instant::now()), None);
    assert_eq!(draws(&mut l, v), 4);
}

#[test]
fn each_view_keeps_its_own_frame_rate() {
    let Some(gpu) = gpu() else { return };
    let (mut l, fast) = counting_loop(&gpu);
    let (slow_id, slow) = l.add_view();
    fast.set_size(8, 8);
    slow.set_size(8, 8);
    l.set_fps(ViewId::PRIMARY, 80.0);
    l.set_fps(slow_id, 20.0);
    let end = Instant::now() + Duration::from_millis(600);
    while Instant::now() < end {
        let next = l.step(Instant::now()).expect("continuous views are always due again");
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let (f, s) = (draws(&mut l, ViewId::PRIMARY), draws(&mut l, slow_id));
    // 48 and 12 at the rates asked; a loaded machine draws fewer of the fast.
    assert!((25..=50).contains(&f), "fast view drew {f}");
    assert!((10..=14).contains(&s), "slow view drew {s}");
}

#[test]
fn a_shared_budget_takes_resolution_from_the_least_important_view_first() {
    let Some(gpu) = gpu() else { return };
    let (mut l, main) = counting_loop(&gpu);
    let (side_id, side) = l.add_view();
    main.set_size(64, 64);
    side.set_size(64, 64);
    main.set_priority(10);
    l.set_fps(ViewId::PRIMARY, 0.0);
    l.set_fps(side_id, 0.0);
    // Any frame is over this budget.
    l.set_gpu_budget(Some(1e-6));
    let scales = |l: &mut RenderLoop<Vec<u32>>| {
        (l.producer().view(ViewId::PRIMARY).render_scale(), l.producer().view(side_id).render_scale())
    };
    let mut seen = Vec::new();
    for _ in 0..200 {
        l.step(Instant::now());
        let s = scales(&mut l);
        if seen.last() != Some(&s) {
            seen.push(s);
        }
    }
    let (main_s, side_s) = scales(&mut l);
    assert_eq!((main_s, side_s), (0.25, 0.25), "both at the floor in the end: {seen:?}");
    // The side view reached the floor while the main one was still whole.
    let floor_at = seen.iter().position(|s| s.1 == 0.25).expect("side reached the floor");
    assert!(seen[..=floor_at].iter().all(|s| s.0 == 1.0), "main gave way early: {seen:?}");

    // Room to spare: the main view gets its resolution back first.
    l.set_gpu_budget(Some(1e6));
    let mut seen = Vec::new();
    for _ in 0..200 {
        l.step(Instant::now());
        let s = scales(&mut l);
        if seen.last() != Some(&s) {
            seen.push(s);
        }
    }
    assert_eq!(scales(&mut l), (1.0, 1.0), "{seen:?}");
    let whole_at = seen.iter().position(|s| s.0 == 1.0).expect("main back to whole");
    assert!(seen[..=whole_at].iter().all(|s| s.1 == 0.25), "side grew first: {seen:?}");
}

#[test]
fn a_render_thread_takes_commands_and_reports_a_panic() {
    let Some(gpu) = gpu() else { return };
    struct World {
        scene: Scene,
        drawn: u32,
        explode: bool,
    }
    let (producer, mut frames) = ViewProducer::on(&gpu, ViewOptions::default().msaa(1));
    let world = World { scene: scene(0xff0000), drawn: 0, explode: false };
    let render = RenderThread::spawn(RenderLoop::new(producer, world, |w: &mut World, view| {
        assert!(!w.explode, "the scene exploded");
        w.drawn += 1;
        view.render(&mut w.scene, &camera());
    }));
    render.set_redraw(ViewId::PRIMARY, Redraw::OnDemand);
    frames.set_size(16, 16);

    // An update changes the scene and asks for a frame; it shows.
    render.update(|w| w.scene.background = Color::from_hex(0x00ff00));
    let deadline = Instant::now() + Duration::from_secs(10);
    let shown = loop {
        if let Some(f) = frames.latest()
            && all_near(&f.read_rgba(&gpu.device, &gpu.queue), 0x00ff00)
        {
            break f;
        }
        assert!(Instant::now() < deadline, "the update never showed");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(shown.frame >= 1);
    // A query answers from the render thread.
    let drawn = render.query(|w| w.drawn).recv().expect("an answer");
    assert!(drawn >= 1);
    // On demand: idle while nothing asks.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(render.query(|w| w.drawn).recv().unwrap(), drawn, "drew with nothing asked");
    assert!(render.is_running());

    // A panic in the drawing ends the loop and says why, on both sides.
    render.update(|w| w.explode = true);
    let deadline = Instant::now() + Duration::from_secs(10);
    while render.is_running() {
        assert!(Instant::now() < deadline, "the loop kept running");
        std::thread::sleep(Duration::from_millis(5));
    }
    let why = render.failure().expect("a failure");
    assert!(why.contains("the scene exploded"), "{why}");
    assert_eq!(frames.failure().as_deref(), Some(why.as_str()));
    assert!(frames.is_closed());
    // Asking a stopped loop is an error, not a hang.
    assert!(render.query(|w| w.drawn).recv().is_err());
}

#[test]
fn stats_count_frames_and_warm_up_leaves_nothing_shown() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = ViewProducer::on(&gpu, ViewOptions::default());
    let mut s = scene(0x3a9a6a);
    producer.warm_up(&mut s, &camera());
    assert!(!frames.has_frame());
    frames.set_size(32, 32);
    for _ in 0..5 {
        producer.render(&mut s, &camera());
        frames.latest();
        std::thread::sleep(Duration::from_millis(10));
    }
    // Two drawn and never looked at.
    producer.render(&mut s, &camera());
    producer.render(&mut s, &camera());
    frames.latest();
    let stats = frames.stats();
    assert!(frames.has_frame());
    assert_eq!(stats.published, 7);
    assert_eq!(stats.shown, 6);
    assert_eq!(stats.scale, 1.0);
    assert!(stats.fps > 0.0, "{stats:?}");
    assert!(stats.latency_ms >= 0.0);
}
