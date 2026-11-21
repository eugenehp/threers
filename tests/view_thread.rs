//! A 3D view drawn on one thread and taken by another: frames arrive in
//! order, at the size asked for, with the pixels drawn; a slow producer never
//! blocks the consumer; dropping either side ends the other's loop.
//!
//! Skipped when no GPU adapter is available.

use std::time::{Duration, Instant};

use threers::gpu::GpuContext;
use threers::renderer::{FramePacer, RendererOptions, SubmitGate, ViewFrames, ViewProducer};
use threers::{Color, PerspectiveCamera, Scene};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn pair(gpu: &GpuContext, gate: Option<SubmitGate>) -> (ViewProducer, ViewFrames) {
    ViewProducer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        RendererOptions::default(),
        gate,
    )
}

fn scene(hex: u32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(hex);
    scene
}

/// The texel at (0, 0) of `texture`, as RGBA bytes.
fn first_texel(gpu: &GpuContext, texture: &wgpu::Texture) -> [u8; 4] {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buffer.slice(..).get_mapped_range().expect("mapped");
    [data[0], data[1], data[2], data[3]]
}

fn near(a: [u8; 4], hex: u32) -> bool {
    let want = [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8];
    (0..3).all(|i| a[i].abs_diff(want[i]) <= 1)
}

#[test]
fn nothing_is_drawn_before_a_size_and_frames_carry_their_pixels() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = pair(&gpu, None);
    let camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);

    assert!(!producer.render(&mut scene(0xff0000), &camera), "no size yet");
    assert!(frames.latest().is_none());

    frames.set_size(32, 16);
    assert_eq!(producer.size(), Some((32, 16)));
    assert!(producer.render(&mut scene(0xff0000), &camera));
    let f = frames.latest().expect("a frame");
    assert!(f.new);
    assert_eq!((f.size, f.frame), ((32, 16), 1));
    assert!(near(first_texel(&gpu, &f.texture), 0xff0000));

    // Asked again with nothing new: the same frame, not new.
    let again = frames.latest().expect("still shown");
    assert!(!again.new);
    assert_eq!(again.frame, 1);

    // Two frames before the consumer looks: it skips to the newer.
    assert!(producer.render(&mut scene(0x00ff00), &camera));
    assert!(producer.render(&mut scene(0x0000ff), &camera));
    let f = frames.latest().expect("a frame");
    assert!(f.new);
    assert_eq!(f.frame, 3);
    assert!(near(first_texel(&gpu, &f.texture), 0x0000ff));
    assert_eq!(frames.published(), 3);

    // A resize: frames that follow come at the new size.
    frames.set_size(8, 24);
    assert!(producer.render(&mut scene(0x3a9a6a), &camera));
    let f = frames.latest().expect("a frame");
    assert_eq!(f.size, (8, 24));
    assert!(near(first_texel(&gpu, &f.texture), 0x3a9a6a));
}

#[test]
fn a_render_scale_draws_fewer_texels_and_says_so() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = pair(&gpu, None);
    let camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    frames.set_size(101, 40);
    producer.set_render_scale(0.5);
    assert!(producer.render(&mut scene(0xff8800), &camera));
    let f = frames.latest().expect("a frame");
    assert_eq!((f.size, f.scale), ((51, 20), 0.5));
    assert!(near(first_texel(&gpu, &f.texture), 0xff8800));
    // The aspect is the box's, not the rounded texels'.
    assert_eq!(producer.aspect(), 101.0 / 40.0);

    producer.set_render_scale(1.0);
    assert!(producer.render(&mut scene(0xff8800), &camera));
    let f = frames.latest().expect("a frame");
    assert_eq!((f.size, f.scale), ((101, 40), 1.0));
}

#[test]
fn adaptive_scale_lowers_the_resolution_of_frames_over_budget() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = pair(&gpu, None);
    let camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    frames.set_size(64, 64);
    // A budget no frame can meet: the scale steps down to the minimum.
    let mut adaptive = threers::renderer::AdaptiveScale::new(1e-6);
    adaptive.settle = 0;
    producer.set_adaptive_scale(Some(adaptive));
    for _ in 0..12 {
        producer.render(&mut scene(0x808080), &camera);
    }
    let f = frames.latest().expect("a frame");
    assert_eq!((f.scale, f.size), (0.25, (16, 16)), "cost {} ms", producer.frame_ms());
}

#[test]
fn the_consumer_never_holds_a_texture_the_producer_draws_into() {
    let Some(gpu) = gpu() else { return };
    let (mut producer, mut frames) = pair(&gpu, None);
    let camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    frames.set_size(4, 4);
    let mut shown: Option<wgpu::Texture> = None;
    for i in 0..20u32 {
        // Draw 0-2 frames between looks, so every hand-over order happens.
        for _ in 0..(i % 3) {
            producer.render_with(|r, view, _| {
                // `drawn` is what the producer is drawing into now.
                let drawn = view.texture();
                assert!(shown.as_ref() != Some(drawn), "drawing into the frame on screen");
                r.render(&mut scene(0x808080), &camera, view, false);
            });
        }
        if let Some(f) = frames.latest() {
            shown = Some(f.texture);
        }
    }
}

#[test]
fn a_render_thread_feeds_a_ui_thread_that_never_waits_for_it() {
    let Some(gpu) = gpu() else { return };
    let gate = SubmitGate::new();
    let (mut producer, mut frames) = pair(&gpu, Some(gate.clone()));
    frames.set_size(64, 64);

    // Each frame of the producer takes 40 ms of CPU, as a heavy scene would.
    let render = std::thread::spawn(move || {
        let mut camera = PerspectiveCamera::new(45.0, producer.aspect(), 0.1, 100.0);
        let mut drawn = 0u32;
        while !producer.is_closed() {
            camera.aspect = producer.aspect();
            let hex = [0xff0000, 0x00ff00][drawn as usize % 2];
            producer.render_with(|r, view, _| {
                std::thread::sleep(Duration::from_millis(40));
                r.render(&mut scene(hex), &camera, view, false);
            });
            drawn += 1;
        }
        drawn
    });

    // The UI side asks for its frame every 4 ms for half a second. Taking the
    // latest frame must never wait out a producer frame. The gate is
    // reconfigured now and then, as a window resize would.
    let start = Instant::now();
    let mut slowest = Duration::ZERO;
    let mut last = 0;
    let mut news = 0;
    let mut pacer = FramePacer::new(250.0);
    while start.elapsed() < Duration::from_millis(500) {
        pacer.wait();
        let t = Instant::now();
        if let Some(f) = frames.latest() {
            assert!(f.frame >= last, "frames go forwards");
            if f.new {
                news += 1;
                assert!(f.frame > last);
            }
            last = f.frame;
        }
        slowest = slowest.max(t.elapsed());
        if news % 3 == 1 {
            drop(gate.reconfigure());
        }
    }
    drop(frames);
    let drawn = render.join().expect("render thread");
    assert!(drawn >= 3, "the producer drew {drawn}");
    assert!(news >= 2, "the consumer saw {news} new frames");
    assert!(
        slowest < Duration::from_millis(20),
        "taking a frame took {slowest:?}, as long as a producer frame"
    );
}

#[test]
fn dropping_either_side_ends_the_other() {
    let Some(gpu) = gpu() else { return };
    let (producer, frames) = pair(&gpu, None);
    let waiter = std::thread::spawn(move || {
        let mut woken = 0;
        while producer.wait_for_work() {
            woken += 1;
        }
        woken
    });
    frames.set_size(10, 10);
    std::thread::sleep(Duration::from_millis(20));
    drop(frames);
    assert!(waiter.join().expect("waiter") <= 1);

    let (producer, frames) = pair(&gpu, None);
    drop(producer);
    assert!(frames.is_closed());

    // A timeout returns without a request, still open.
    let (producer, _frames) = pair(&gpu, None);
    let t = Instant::now();
    assert!(producer.wait_for_work_timeout(Some(Duration::from_millis(30))));
    assert!(t.elapsed() >= Duration::from_millis(25));
}
