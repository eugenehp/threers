//! One scene, four ways: raw, À-Trous, a trained network, and the reference.
//!
//! ```text
//! cargo run --release --features raytrace,parallel --example denoise_compare
//! cargo run --release --features raytrace,parallel,learned-denoise-metal \
//!     --example denoise_compare -- --learned out/g12_oidn.bin --spp 8
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--spp N` | samples for the noisy render (default 4) |
//! | `--ref N` | samples for the reference (default 1024) |
//! | `--scene N` | which procedural scene (default 5000) |
//! | `--size WxH` | resolution (default 512x512) |
//! | `--learned FILE` | weights for the trained denoiser |
//! | `--mirrored` | average four mirrored passes (4x inference, no retraining) |
//! | `--prefilter` | clean the albedo and normal guides first, as Cycles does |
//! | `--out DIR` | where the PNGs go (default `out/compare`) |
//! | `--gpu` | trace on the compute backend |
//!
//! Writes `raw.png`, `atrous.png`, `learned.png`, `reference.png` and a
//! `strip.png` of all four side by side, and prints each one's error against
//! the reference.
//!
//! # Reading it
//!
//! The number to watch is the ratio, not the absolute error, because the error
//! is set as much by the scene and the sample count as by the filter. And no
//! filter can reach zero: the reference is itself a Monte-Carlo estimate, and
//! two independent references of the same scene at 1024 samples differ by about
//! 0.0135 — so an error near 0.01 is the floor, not a failure.
//!
//! The strip is there because the numbers do not say everything a denoiser gets
//! wrong. Over-smoothing, lost contact shadows and a plastic look all score
//! well and read badly, which is why this writes images and not just a table.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use threers::raytrace::{
    denoise, DenoiseGuides, DenoiseParams, RaytraceRenderer, RaytraceSettings,
};
use threers::{
    encode_png, BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PhysicalMaterial,
    PlaneGeometry, Quaternion, Scene, SphereGeometry, StandardMaterial, Texture, TextureFormat,
    TextureWrap, ToneMapping, Vector2, Vector3,
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
    let spp: u32 = arg("--spp").and_then(|s| s.parse().ok()).unwrap_or(4);
    let reference_spp: u32 = arg("--ref").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let scene_index: u32 = arg("--scene").and_then(|s| s.parse().ok()).unwrap_or(5000);
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((512u32, 512u32));
    let out = arg("--out").unwrap_or_else(|| "out/compare".into());
    let _ = std::fs::create_dir_all(&out);

    let settings = |samples: u32| RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::AcesFilmic,
        ..Default::default()
    };

    println!("scene {scene_index} at {w}x{h}: {spp} spp against {reference_spp}");

    // The noisy render, kept whole: every filter below sees the same samples,
    // so the comparison is of filters and not of renders.
    let mut r = make_renderer(w, h);
    r.set_settings(settings(spp));
    let (mut scene, camera) = build_scene(scene_index, w as f32 / h as f32);
    r.render(&mut scene, &camera).expect("render");
    let raw = r.resolve_rgba_denoised(false);
    let raw_hdr = r.film().resolve_hdr();
    let scale = r.traced_scene().map(|s| s.scale()).unwrap_or(1.0);

    // À-Trous, given every guide it wants.
    let atrous_hdr = denoise(
        w,
        h,
        &raw_hdr,
        &DenoiseGuides {
            albedo: &r.film().resolve_albedo(),
            normal: &r.film().resolve_normal(),
            depth: &r.film().resolve_depth(),
            variance: &r.film().resolve_variance(),
            scene_scale: scale,
        },
        &DenoiseParams::default(),
    );
    let atrous = r.film().beauty_rgba8(&atrous_hdr, r.settings());

    // The trained network, from the same film.
    #[cfg(feature = "learned-denoise")]
    let learned = arg("--learned").and_then(|path| {
        match threers::raytrace::learned::LearnedDenoiser::open(&path) {
            Ok(mut d) => {
                if flag("--mirrored") {
                    d.set_passes(threers::raytrace::learned::Passes::Mirrored);
                }
                let prefilter = if flag("--prefilter-atrous") {
                    threers::raytrace::learned::Prefilter::Atrous
                } else if flag("--prefilter") {
                    threers::raytrace::learned::Prefilter::Guides
                } else {
                    threers::raytrace::learned::Prefilter::None
                };
                d.set_prefilter(prefilter);
                match d.apply(r.film(), scale) {
                    Ok(hdr) => {
                        println!(
                            "learned denoiser: {path}{} prefilter={prefilter:?}",
                            if flag("--mirrored") { " mirrored" } else { "" }
                        );
                        Some((r.film().beauty_rgba8(&hdr, r.settings()), hdr))
                    }
                    Err(e) => {
                        eprintln!("learned denoiser failed: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("could not load {path}: {e}");
                None
            }
        }
    });
    #[cfg(not(feature = "learned-denoise"))]
    let learned: Option<(Vec<u8>, Vec<f32>)> = {
        if arg("--learned").is_some() {
            eprintln!("--learned needs the `learned-denoise` feature");
        }
        None
    };

    // The reference.
    let mut rr = make_renderer(w, h);
    rr.set_settings(settings(reference_spp));
    let (mut scene, camera) = build_scene(scene_index, w as f32 / h as f32);
    rr.render(&mut scene, &camera).expect("reference");
    let reference = rr.resolve_rgba_denoised(false);
    let reference_hdr = rr.film().resolve_hdr();

    let write = |name: &str, rgba: &[u8]| {
        std::fs::write(format!("{out}/{name}.png"), encode_png(w, h, rgba)).expect("write");
    };
    write("raw", &raw);
    write("atrous", &atrous);
    write("reference", &reference);
    if let Some((rgba, _)) = &learned {
        write("learned", rgba);
    }

    let mut panels: Vec<(&str, &[u8])> = vec![("raw", &raw), ("atrous", &atrous)];
    if let Some((rgba, _)) = &learned {
        panels.push(("learned", rgba));
    }
    panels.push(("reference", &reference));
    std::fs::write(format!("{out}/strip.png"), strip(w, h, &panels)).expect("write strip");

    println!("\n{:<11}{:>10}{:>9}", "", "error", "vs raw");
    let base = relative_error(&raw_hdr, &reference_hdr);
    println!("{:<11}{base:>10.5}{:>9}", "raw", "-");
    let row = |name: &str, hdr: &[f32]| {
        let e = relative_error(hdr, &reference_hdr);
        println!("{name:<11}{e:>10.5}{:>8.2}x", base / e.max(1e-9));
    };
    row("atrous", &atrous_hdr);
    if let Some((_, hdr)) = &learned {
        row("learned", hdr);
    }
    // The floor is not a constant: it is the reference's own Monte-Carlo error,
    // which falls as 1/sqrt(n) like everything else. Measured at 1024 samples,
    // two independent references of these scenes differ by 0.0135, so one sits
    // 0.0135/sqrt(2) from the truth — scale that to whatever `--ref` was.
    const MEASURED_AT: f32 = 1024.0;
    const PAIR_AT_1024: f32 = 0.0135;
    let floor =
        PAIR_AT_1024 / std::f32::consts::SQRT_2 * (MEASURED_AT / reference_spp as f32).sqrt();
    println!(
        "\nfloor: this {reference_spp}-sample reference is itself about {floor:.4} from the truth,\n\
         so no filter measured against it can do better than that.\n\
         wrote {out}/strip.png"
    );
}

fn make_renderer(w: u32, h: u32) -> RaytraceRenderer {
    if flag("--gpu") {
        match threers::raytrace::gpu::GpuBackend::headless() {
            Ok(b) => return RaytraceRenderer::with_backend(w, h, Box::new(b)),
            Err(e) => println!("no gpu ({e}); tracing on the cpu"),
        }
    }
    RaytraceRenderer::new(w, h)
}

/// Relative L2 over RGB, the same measure `rlx-denoise` trains and reports on.
fn relative_error(image: &[f32], reference: &[f32]) -> f32 {
    const EPSILON: f64 = 0.01;
    let compress = |v: f32| {
        if !v.is_finite() {
            return if v > 0.0 { 1.0 } else { 0.0 };
        }
        v.max(0.0) / (1.0 + v.max(0.0))
    };
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for i in 0..reference.len() / 4 {
        for c in 0..3 {
            let y = compress(image[i * 4 + c]) as f64;
            let t = compress(reference[i * 4 + c]) as f64;
            sum += (y - t) * (y - t) / (t * t + EPSILON);
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

/// Lay the panels out left to right with a thin divider.
fn strip(w: u32, h: u32, panels: &[(&str, &[u8])]) -> Vec<u8> {
    const GAP: u32 = 4;
    let total = panels.len() as u32 * w + GAP * (panels.len() as u32 - 1);
    let mut out = vec![24u8; (total * h * 4) as usize];
    for i in 0..(total * h) as usize {
        out[i * 4 + 3] = 255;
    }
    for (i, (_, rgba)) in panels.iter().enumerate() {
        let x0 = i as u32 * (w + GAP);
        for y in 0..h {
            for x in 0..w {
                let src = ((y * w + x) * 4) as usize;
                let dst = (((y * total) + x0 + x) * 4) as usize;
                out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
            }
        }
    }
    encode_png(total, h, &out)
}

// ------------------------------------------------------------------ scene

/// A representative scene, built here rather than shared with
/// `denoise_dataset`.
///
/// It is deliberately *not* the training distribution: this example is for
/// looking at what a denoiser does to an image, and a model should be looked at
/// on something it was not fitted to. The number it prints is one scene and
/// should be read as an illustration.
///
/// For a figure that means something, score the model on a held-out tile set —
/// `rlx-denoise eval --val testset.bin --weights w.bin` — which covers 256
/// tiles across 64 scenes rather than one camera angle.
fn build_scene(index: u32, aspect: f32) -> (Scene, PerspectiveCamera) {
    let mut rng = Rng::new(index as u64 + 1);
    let mut scene = Scene::new();
    scene.background = Color::new(0.015, 0.02, 0.03);

    let mut floor_mat = PhysicalMaterial::new(Color::new(0.55, 0.55, 0.58));
    floor_mat.roughness = 0.25 + rng.next() * 0.5;
    if rng.next() < 0.5 {
        floor_mat.map = Some(Arc::new(checker(64)));
    }
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(24.0, 24.0),
        Material::Physical(floor_mat),
    ));
    floor.quaternion = Quaternion::from_euler_xyz(-FRAC_PI_2, 0.0, 0.0);
    scene.add(floor);

    let mut metal = PhysicalMaterial::new(Color::new(0.95, 0.78, 0.42));
    metal.metalness = 1.0;
    metal.roughness = 0.08 + rng.next() * 0.3;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 64, 32),
        Material::Physical(metal),
    ));
    ball.position = Vector3::new(-0.85, 0.55, 0.1);
    scene.add(ball);

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

    let mut chalk = PhysicalMaterial::new(Color::new(0.75, 0.25, 0.22));
    chalk.roughness = 0.9;
    let mut block = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.7, 0.9, 0.7),
        Material::Physical(chalk),
    ));
    block.position = Vector3::new(0.35, 0.45, 0.85);
    scene.add(block);

    let mut lamp_mat = StandardMaterial::new(Color::WHITE);
    lamp_mat.emissive = Color::new(1.0, 0.94, 0.85);
    lamp_mat.emissive_intensity = 90.0 + rng.next() * 60.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.2 + rng.next() * 0.12, 24, 12),
        Material::Standard(lamp_mat),
    ));
    lamp.position = Vector3::new(1.7, 3.1, 1.4);
    scene.add(lamp);

    let mut camera = PerspectiveCamera::new(38.0, aspect, 0.05, 100.0);
    let a = rng.next() * std::f32::consts::TAU;
    camera.position = Vector3::new(a.cos() * 3.6, 1.4 + rng.next() * 0.5, a.sin() * 3.6);
    camera.look_at(Vector3::new(0.0, 0.55, 0.0));
    (scene, camera)
}

fn checker(size: u32) -> Texture {
    let mut data = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 220 } else { 60 };
            let i = ((y * size + x) * 4) as usize;
            data[i] = v;
            data[i + 1] = v;
            data[i + 2] = v;
            data[i + 3] = 255;
        }
    }
    let mut t = Texture::new(size, size, TextureFormat::Rgba8UnormSrgb, data);
    t.wrap_s = TextureWrap::Repeat;
    t.wrap_t = TextureWrap::Repeat;
    t.repeat = Vector2::new(6.0, 6.0);
    t.flip_y = false;
    t
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32
    }
}
