//! Path-traced video, with and without the denoiser, from one render.
//!
//! ```text
//! cargo run --release --features raytrace,video,parallel --example path_trace_video
//! cargo run --release --features raytrace,video,parallel --example path_trace_video -- \
//!     --frames 120 --spp 256 --size 1280x720
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--frames N` | frames in the orbit (default 60) |
//! | `--spp N` | samples per pixel per frame (default 64) |
//! | `--size WxH` | resolution (default 960x540) |
//! | `--fps N` | frame rate (default 30) |
//! | `--out DIR` | where everything goes (default `out/path_trace_video`) |
//! | `--gpu` | use the compute backend |
//! | `--only-denoised` | skip the unfiltered pass |
//! | `--learned FILE` | also render a pass through a trained denoiser |
//! | `--mirrored` | average four mirrored passes of it (~2.7% better) |
//! | `--rolling-seed` | give every frame its own noise (see below) |
//! | `--keep-frames` | also write the PNG stills |
//!
//! Writes `denoised.mp4` and `raw.mp4` — the same samples resolved twice, so
//! the pair isolates the denoiser and nothing else. With `--learned` it writes
//! `learned.mp4` from the same samples again, which makes the three-way
//! comparison a comparison of filters rather than of renders:
//!
//! ```text
//! cargo run --release --features raytrace,video,parallel,learned-denoise \
//!     --example path_trace_video -- --learned weights.bin
//! ```
//!
//! # Why video is harder than a still
//!
//! A path-traced still is noisy. A path-traced *sequence* is noisy in a
//! different place every frame, and the eye is far more sensitive to that than
//! to the noise itself: static grain reads as film, and grain that reseeds every
//! frame reads as boiling. It is the single worst artefact in Monte-Carlo
//! animation, and it does not appear in any single frame you might inspect.
//!
//! Three things here address it, and they are independent:
//!
//! **The seed is fixed by default.** Every frame starts its sampler at the same
//! place, so a pixel looking at the same surface draws the same sequence and its
//! error barely changes between frames. The noise stops boiling and starts
//! sliding with the image. `--rolling-seed` gives each frame its own, which is
//! what an unthinking implementation does; render both and the difference is
//! obvious in motion and invisible in a screenshot.
//!
//! **The denoiser runs per frame.** It removes most of what remains, and
//! because the guides it filters by — albedo and normal — are noise-free and
//! move smoothly with the camera, its output moves smoothly too.
//!
//! **Samples buy more than resolution does.** Noise falls as `1/sqrt(n)`, so
//! four times the samples halves it, while twice the resolution costs the same
//! four times and halves nothing. For a shot that is going to be watched rather
//! than pixel-peeped, spend the budget on `--spp` first.
//!
//! # Cost
//!
//! The geometry does not move, so the BVH is built once and every frame after
//! that is [`RaytraceRenderer::set_camera`] — for this scene that is the
//! difference between a rebuild per frame and none.

use std::f32::consts::TAU;

use threers::raytrace::{RaytraceRenderer, RaytraceSettings};
use threers::{
    render_sequence, BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, PlaneGeometry, Quaternion, Scene, SequenceOptions, SphereGeometry,
    StandardMaterial, ToneMapping, Vector3,
};

fn arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn main() {
    let frames: usize = arg("--frames").and_then(|s| s.parse().ok()).unwrap_or(60);
    let spp: u32 = arg("--spp").and_then(|s| s.parse().ok()).unwrap_or(64);
    let fps: u32 = arg("--fps").and_then(|s| s.parse().ok()).unwrap_or(30);
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((960u32, 540u32));
    let out = arg("--out").unwrap_or_else(|| "out/path_trace_video".into());
    let rolling = flag("--rolling-seed");
    let keep_frames = flag("--keep-frames");
    let both = !flag("--only-denoised");
    let learned_weights = arg("--learned");
    let _ = std::fs::create_dir_all(&out);

    let mut scene = build_scene();
    let mut camera = PerspectiveCamera::new(38.0, w as f32 / h as f32, 0.05, 100.0);

    let mut renderer = RaytraceRenderer::new(w, h);
    #[cfg(not(target_arch = "wasm32"))]
    if flag("--gpu") {
        match threers::raytrace::gpu::GpuBackend::headless() {
            Ok(backend) => renderer = RaytraceRenderer::with_backend(w, h, Box::new(backend)),
            Err(e) => println!("gpu backend unavailable ({e}); staying on the cpu"),
        }
    }
    renderer.set_settings(RaytraceSettings {
        samples_per_pixel: spp,
        max_bounces: 6,
        min_bounces: 3,
        tone_mapping: ToneMapping::AcesFilmic,
        exposure: 1.0,
        // Adaptive sampling moves the budget to the noisy pixels, which for a
        // sequence also means the *amount* of noise varies between frames as
        // the camera moves. Off here: an even budget gives an even grain, and
        // an even grain is what the denoiser and the eye both prefer.
        adaptive_threshold: 0.0,
        denoise: true,
        ..Default::default()
    });

    println!(
        "{w}x{h}  {frames} frames  {spp} spp  {} backend  seed {}",
        renderer.backend_name(),
        if rolling { "per frame" } else { "fixed" }
    );

    // Build the acceleration structure once. Nothing in the scene moves — only
    // the camera — so every later frame is a camera swap.
    camera.position = orbit(0.0);
    camera.look_at(TARGET);
    renderer.prepare(&mut scene, &camera);
    if let Some(report) = renderer.report() {
        println!("scene: {report:?}");
    }

    // Frames land on disk as they are traced rather than piling up in memory:
    // 120 frames of 1280x720 is 440 MB per variant, and the pair would be most
    // of a gigabyte held for the whole render to buy nothing. The stills are
    // the durable artefact anyway — re-encoding is free, re-tracing is not.
    let mut variants: Vec<&str> = vec!["denoised"];
    if both {
        variants.push("raw");
    }

    // The trained denoiser resolves the same film a third time. Loading it here
    // rather than at the first frame means a bad path fails before the render
    // rather than after it.
    #[cfg(feature = "learned-denoise")]
    let mut learned = match &learned_weights {
        Some(path) => {
            // Whatever device this build was compiled for — Metal with
            // `learned-denoise-metal`, wgpu with `learned-denoise-gpu`.
            match threers::raytrace::learned::LearnedDenoiser::open(path) {
                Ok(mut d) => {
                    if flag("--mirrored") {
                        d.set_passes(threers::raytrace::learned::Passes::Mirrored);
                    }
                    println!(
                        "learned denoiser: {path}{}",
                        if flag("--mirrored") { ", mirrored" } else { "" }
                    );
                    variants.push("learned");
                    Some(d)
                }
                Err(e) => {
                    eprintln!("could not load {path}: {e}");
                    None
                }
            }
        }
        None => None,
    };
    #[cfg(not(feature = "learned-denoise"))]
    if learned_weights.is_some() {
        eprintln!("--learned needs the `learned-denoise` feature; ignoring it");
    }
    for name in &variants {
        std::fs::create_dir_all(format!("{out}/{name}")).expect("frame directory");
    }

    let started = std::time::Instant::now();
    for i in 0..frames {
        let t = i as f32 / frames as f32;
        camera.position = orbit(t);
        camera.look_at(TARGET);

        if rolling {
            // A fresh seed per frame is the naive choice, and the reason
            // Monte-Carlo animation boils. Kept as a flag so the difference can
            // be seen rather than argued about.
            let mut settings = renderer.settings().clone();
            settings.seed = 0x5eed_0000 + i as u64;
            renderer.set_settings(settings);
            renderer.prepare(&mut scene, &camera);
        } else if !renderer.set_camera(&camera) {
            renderer.prepare(&mut scene, &camera);
        }

        if let Err(e) = renderer.accumulate(spp) {
            eprintln!("frame {i}: {e}");
        }
        for name in &variants {
            let rgba = match *name {
                #[cfg(feature = "learned-denoise")]
                "learned" => {
                    let scale = renderer.traced_scene().map(|s| s.scale()).unwrap_or(1.0);
                    let hdr = learned
                        .as_mut()
                        .expect("the learned variant exists only when it loaded")
                        .apply(renderer.film(), scale)
                        .expect("learned denoiser");
                    renderer.film().beauty_rgba8(&hdr, renderer.settings())
                }
                "denoised" => renderer.resolve_rgba_denoised(true),
                _ => renderer.resolve_rgba_denoised(false),
            };
            std::fs::write(frame_path(&out, name, i), threers::encode_png(w, h, &rgba))
                .expect("write frame");
        }
        progress(i, frames, started);
    }
    println!(
        "\rtraced {frames} frames in {:.1}s ({:.2}s/frame)            ",
        started.elapsed().as_secs_f32(),
        started.elapsed().as_secs_f32() / frames as f32
    );

    for name in &variants {
        encode(&out, name, w, h, frames, fps, keep_frames);
    }
    if both {
        println!("the two differ only by the filter — identical samples, identical camera");
    }
}

fn frame_path(out: &str, name: &str, index: usize) -> String {
    format!("{out}/{name}/{name}_{index:05}.png")
}

/// Encode one variant from the stills already on disk.
///
/// Reading each frame back costs a PNG decode, which against the path tracing
/// that produced it is free — and it means only one frame is in memory at a
/// time however long the shot is.
fn encode(out: &str, name: &str, w: u32, h: u32, frames: usize, fps: u32, keep_frames: bool) {
    let opts = SequenceOptions::new(frames)
        .fps(fps)
        .video_to(format!("{out}/{name}.mp4"))
        // A path-traced frame has residual grain even after denoising, and
        // grain is the hardest thing for an encoder to hold on to. 16 rather
        // than the default 18 for that reason.
        .crf(16)
        .progress(false);
    let result = render_sequence(w, h, &opts, |i| {
        std::fs::read(frame_path(out, name, i))
            .ok()
            .and_then(|bytes| threers::decode_png(&bytes).ok())
            .map(|img| img.rgba)
            .unwrap_or_else(|| vec![0u8; w as usize * h as usize * 4])
    });
    match result {
        Ok(report) => println!(
            "  {name}.mp4  {} frames, {:.1}s to encode",
            report.frames, report.elapsed_secs
        ),
        Err(e) => eprintln!("  {name}: {e}"),
    }
    if !keep_frames {
        let _ = std::fs::remove_dir_all(format!("{out}/{name}"));
    }
}

const TARGET: Vector3 = Vector3::new(0.0, 0.55, 0.0);

/// A slow orbit that keeps the subject centred.
fn orbit(t: f32) -> Vector3 {
    let a = t * TAU;
    Vector3::new(a.cos() * 3.6, 1.5 + (a * 2.0).sin() * 0.25, a.sin() * 3.6)
}

fn progress(index: usize, frames: usize, started: std::time::Instant) {
    use std::io::Write;
    let done = index + 1;
    let per = started.elapsed().as_secs_f32() / done as f32;
    print!(
        "\r  frame {done}/{frames}  {per:.2}s/frame  {:.0}s left     ",
        per * (frames - done) as f32
    );
    let _ = std::io::stdout().flush();
}

/// A scene chosen to make the noise visible: one small bright light, so
/// everything off it is lit indirectly and every path has to find it.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.015, 0.02, 0.03);

    // Floor.
    let mut floor_mat = PhysicalMaterial::new(Color::new(0.55, 0.55, 0.58));
    floor_mat.roughness = 0.35;
    floor_mat.metalness = 0.0;
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(24.0, 24.0),
        Material::Physical(floor_mat),
    ));
    floor.quaternion = Quaternion::from_euler_xyz(-std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    scene.add(floor);

    // A rough metal sphere: glossy interreflection is where a path tracer's
    // noise lives, and where a denoiser earns its keep.
    let mut metal = PhysicalMaterial::new(Color::new(0.95, 0.78, 0.42));
    metal.metalness = 1.0;
    metal.roughness = 0.18;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 64, 32),
        Material::Physical(metal),
    ));
    ball.position = Vector3::new(-0.85, 0.55, 0.1);
    scene.add(ball);

    // Glass: long specular chains, the noisiest thing in the frame.
    let mut glass = PhysicalMaterial::new(Color::new(0.98, 0.99, 1.0));
    glass.transmission = 1.0;
    glass.roughness = 0.02;
    glass.ior = 1.5;
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.5, 64, 32),
        Material::Physical(glass),
    ));
    sphere.position = Vector3::new(0.55, 0.5, -0.35);
    scene.add(sphere);

    // A diffuse block, for a surface with nothing to hide behind.
    let mut chalk = PhysicalMaterial::new(Color::new(0.75, 0.25, 0.22));
    chalk.roughness = 0.9;
    let mut block = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.7, 0.9, 0.7),
        Material::Physical(chalk),
    ));
    block.position = Vector3::new(0.35, 0.45, 0.85);
    scene.add(block);

    // One small emitter, high and to the side. Small is the point: a large soft
    // light converges quickly and would hide the thing this example is about.
    let mut lamp_mat = StandardMaterial::new(Color::WHITE);
    lamp_mat.emissive = Color::new(1.0, 0.94, 0.85);
    lamp_mat.emissive_intensity = 120.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.22, 24, 12),
        Material::Standard(lamp_mat),
    ));
    lamp.position = Vector3::new(1.7, 3.1, 1.4);
    scene.add(lamp);

    scene
}
