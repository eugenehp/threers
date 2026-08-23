//! Export training pairs for a learned denoiser.
//!
//! A path tracer is its own dataset generator: render a scene at a low sample
//! count and again at a high one, and the pair is a training example. No corpus
//! needs collecting and no licence attaches to the result — which is the reason
//! a denoiser is worth training here at all, when the two that ship with
//! production renderers are either proprietary (OptiX, inside the driver) or a
//! large pretrained binary (OpenImageDenoise).
//!
//! ```text
//! GPU=1 cargo run --release --example denoise_dataset --features raytrace,parallel
//! cargo run --release --example denoise_dataset --features raytrace,parallel
//! SCENES=64 SPP=4 REF=1024 cargo run --release --example denoise_dataset --features raytrace
//! BASELINE=1 cargo run --release --example denoise_dataset --features raytrace
//! ```
//!
//! `BASELINE=1` writes nothing and instead reports what the hand-fitted À-Trous
//! filter achieves on the held-out scenes. That number is the one a trained
//! network has to beat to be worth its weight, and it is measured the same way
//! — relative L2 in the same compressed range, over the same tiles, summed
//! across the set with a single root at the end.
//!
//! Writes `out/denoise_train.bin` and `out/denoise_val.bin`.
//!
//! # Format
//!
//! ```text
//! magic   "RLXDN002"   8 bytes
//! tile    u32          side length in pixels, a multiple of 8
//! count   u32          number of tiles
//! inputs  u32          input planes per tile
//! outputs u32          target planes per tile
//! tiles   f32 x count x (inputs + outputs) x tile x tile   little-endian, planar
//! ```
//!
//! Eleven input planes then three target planes. Planar rather than
//! interleaved because that is the NCHW layout the network wants, so the loader
//! is a straight read. The plane counts are in the header so a file says what
//! shape of network it belongs to, rather than a mismatch showing up as a
//! network that trains on the wrong thing.
//!
//! `RLXDN001` files are still read, as nine inputs and three outputs.
//!
//! # Guides
//!
//! | planes | |
//! |---|---|
//! | 0-2 | colour, compressed |
//! | 3-5 | first-hit albedo |
//! | 6-8 | first-hit shading normal |
//! | 9 | first-hit depth, scene-relative |
//! | 10 | per-pixel standard error |
//! | 11-13 | the converged colour, compressed |
//!
//! Depth and standard error are here because the renderer already measures them
//! and the hand-tuned À-Trous filter already uses them — leaving them out gave
//! the fixed filter strictly more information than the learned one. The
//! standard error matters most: it is the renderer telling the network how far
//! it should trust each pixel, which the network otherwise has to infer from
//! the noise it is trying to remove.
//!
//! # Range
//!
//! Radiance is unbounded and a convolution stack is not: fed raw HDR, the first
//! layer's activations span four orders of magnitude and training diverges.
//! Colour and target are stored compressed by `x / (1 + x)`, which is bounded
//! in `[0, 1)`, monotone, and invertible as `x' / (1 - x')`. Albedo is already
//! a reflectance and normals are already unit vectors, so both are stored as
//! they are.

use std::cell::RefCell;
use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use threers::raytrace::{
    denoise, DenoiseGuides, DenoiseParams, RaytraceRenderer, RaytraceSettings,
};
use threers::{
    BoxGeometry, Color, CylinderGeometry, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, PlaneGeometry, PmremGenerator, Scene, SphereGeometry, StandardMaterial,
    Texture, TextureFormat, TextureWrap, ToneMapping, TorusGeometry, Vector2, Vector3,
};

/// Input planes per tile: colour, albedo, normal, depth, standard error.
const IN_PLANES: u32 = 11;
/// Target planes: the converged colour.
const OUT_PLANES: u32 = 3;

/// Side of one training tile. A multiple of 8, because the network halves its
/// resolution three times.
const TILE: u32 = 128;
/// Rendered frames are this many tiles on a side.
const TILES_PER_SIDE: u32 = 2;
const FRAME: u32 = TILE * TILES_PER_SIDE;

fn main() {
    let scenes = env_u32("SCENES", 40);
    // Scene content is a pure function of its index, so an offset gives a set
    // disjoint from an earlier run's — which is how a dataset grows without
    // regenerating what already exists.
    let offset = env_u32("OFFSET", 0);
    // 0 cycles NOISY_SPP across scenes; a value pins every scene to it.
    let noisy_spp = env_u32("SPP", 0);
    let reference_spp = env_u32("REF", 1024);
    // The last eighth is held out. Fitting and judging on the same renders says
    // nothing about whether the network learned the problem or the pictures.
    let validation_from = scenes - (scenes / 8).max(1);

    if env_u32("GUIDENOISE", 0) > 0 {
        guide_noise(scenes, offset, reference_spp);
        return;
    }

    if env_u32("PREVIEW", 0) > 0 {
        preview(scenes, offset);
        return;
    }

    if env_u32("AGREE", 0) > 0 {
        agree(scenes, offset, reference_spp);
        return;
    }

    if env_u32("SWEEP", 0) > 0 {
        sweep(scenes, offset, reference_spp);
        return;
    }

    if env_u32("FLOOR", 0) > 0 {
        floor(scenes, offset, reference_spp);
        return;
    }

    if env_u32("BASELINE", 0) > 0 {
        baseline(scenes, validation_from, offset, noisy_spp, reference_spp);
        return;
    }

    std::fs::create_dir_all("out").expect("create out/");
    println!(
        "{scenes} scenes at {FRAME}x{FRAME}, {} vs {reference_spp} samples \
         -> {} tiles of {TILE} ({IN_PLANES}+{OUT_PLANES} planes)",
        if noisy_spp > 0 {
            noisy_spp.to_string()
        } else {
            format!("{NOISY_SPP:?}")
        },
        scenes * TILES_PER_SIDE * TILES_PER_SIDE
    );

    let mut train: Vec<f32> = Vec::new();
    let mut train_tiles = 0u32;
    let mut val: Vec<f32> = Vec::new();
    let mut val_tiles = 0u32;

    let start = std::time::Instant::now();
    for index in 0..scenes {
        let spp = if noisy_spp > 0 {
            noisy_spp
        } else {
            NOISY_SPP[index as usize % NOISY_SPP.len()]
        };
        let (noisy, guides, reference, extra) =
            render_pair_with_depth(offset + index, spp, reference_spp);
        let (sink, count) = if index < validation_from {
            (&mut train, &mut train_tiles)
        } else {
            (&mut val, &mut val_tiles)
        };
        for ty in 0..TILES_PER_SIDE {
            for tx in 0..TILES_PER_SIDE {
                write_tile(sink, &noisy, &guides, &extra, &reference, tx, ty);
                *count += 1;
            }
        }
        println!(
            "  scene {:>3}/{scenes}  {spp:>2}spp  {:.0}s",
            index + 1,
            start.elapsed().as_secs_f32()
        );
    }

    let prefix = std::env::var("PREFIX").unwrap_or_else(|_| "out/denoise".into());
    write_set(&format!("{prefix}_train.bin"), train_tiles, &train);
    write_set(&format!("{prefix}_val.bin"), val_tiles, &val);
}

/// How noisy are the guide planes themselves?
///
/// Cycles prefilters albedo and normal before using them, on the grounds that a
/// noisy guide misleads the denoiser about where surfaces end. Whether that is
/// worth copying depends on how noisy ours actually are — a guide that is
/// already clean at four samples needs no filter, and building one would be
/// three times the inference for nothing.
///
/// Reported against the same guides at `REF` samples, next to the colour's own
/// error for scale.
fn guide_noise(scenes: u32, offset: u32, reference_spp: u32) {
    println!("guide noise against {reference_spp} samples, {scenes} scenes\n");
    let settings = |samples: u32| RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        ..Default::default()
    };
    let render = |index: u32, spp: u32| {
        let (mut scene, camera) = build_scene(index);
        let mut r = renderer(FRAME, FRAME);
        r.set_settings(settings(spp));
        r.render(&mut scene, &camera).expect("render");
        (
            r.film().resolve_hdr(),
            r.film().resolve_albedo(),
            r.film().resolve_normal(),
        )
    };

    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}",
        "spp", "colour", "albedo", "normal"
    );
    for spp in [1u32, 2, 4, 8] {
        let (mut c, mut a, mut n) = (0.0f64, 0.0f64, 0.0f64);
        let mut count = 0usize;
        for index in 0..scenes {
            let (rc, ra, rn) = render(offset + index, reference_spp);
            let (nc, na, nn) = render(offset + index, spp);
            for i in 0..(FRAME * FRAME) as usize {
                for k in 0..3 {
                    // Relative to each quantity's own scale, so the three are
                    // comparable: albedo and normals are bounded, radiance is
                    // not.
                    let dc = compress(nc[i * 4 + k]) - compress(rc[i * 4 + k]);
                    c += (dc * dc) as f64;
                    let da = na[i][k] - ra[i][k];
                    a += (da * da) as f64;
                    let dn = nn[i][k] - rn[i][k];
                    n += (dn * dn) as f64;
                    count += 1;
                }
            }
        }
        let rms = |v: f64| (v / count as f64).sqrt();
        println!(
            "{spp:>5}  {:>10.5}  {:>10.5}  {:>10.5}",
            rms(c),
            rms(a),
            rms(n)
        );
    }
    println!(
        "\nalbedo is in [0,1] and normals are unit vectors, so these are absolute\n\
         and directly comparable with each other."
    );
}

/// Render a few scenes to PNGs, to see what the generator is actually making.
///
/// A distribution is easy to describe and easy to get wrong — "a third are
/// enclosed rooms" is a claim about code that has to be looked at to be
/// believed. This also reports the split, counted rather than assumed.
fn preview(scenes: u32, offset: u32) {
    let _ = std::fs::create_dir_all("out/preview");
    let mut enclosed = 0u32;
    let mut lit_by_sky = 0u32;
    for index in 0..scenes {
        let (mut scene, camera) = build_scene(offset + index);
        // Counted, not assumed: "a quarter are sky-lit" is a claim about a
        // branch, and branches drift.
        let sky = scene.environment.is_some();
        if sky {
            lit_by_sky += 1;
        }
        let mut r = renderer(FRAME, FRAME);
        r.set_settings(RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 5,
            min_bounces: 3,
            clamp_indirect: 100.0,
            adaptive_threshold: 0.0,
            denoise: false,
            tone_mapping: ToneMapping::AcesFilmic,
            ..Default::default()
        });
        r.render(&mut scene, &camera).expect("render");
        // Five walls and a ceiling put a room well above an open floor's count,
        // so the mesh tally separates them without inspecting the graph. This
        // only holds for the procedural scenes: a CAD assembly is many meshes
        // and no room at all, and counting it as one would report 50% enclosed
        // for a set that has no rooms in it.
        let meshes = r.report().map(|b| b.meshes).unwrap_or(0);
        if meshes >= 9 && std::env::var("REAL").is_err() {
            enclosed += 1;
        }
        std::fs::write(
            format!("out/preview/scene_{:05}.png", offset + index),
            threers::encode_png(FRAME, FRAME, &r.resolve_rgba_denoised(false)),
        )
        .expect("write preview");
        println!(
            "  scene {:>5}  {meshes} meshes{}{}",
            offset + index,
            if meshes >= 9 { "  [room]" } else { "" },
            if sky { "  [sky]" } else { "" }
        );
    }
    println!(
        "\n{enclosed}/{scenes} enclosed ({:.0}%), {lit_by_sky}/{scenes} sky-lit ({:.0}%) \
         — wrote out/preview/",
        100.0 * enclosed as f32 / scenes as f32,
        100.0 * lit_by_sky as f32 / scenes as f32
    );
}

/// Do the two backends produce the same dataset?
///
/// A dataset rendered on the GPU has to be the one the CPU would have produced,
/// or every number measured against it is measured against a different
/// renderer. Monte-Carlo means the two will not be bit-identical — they draw
/// the same sequence but accumulate in a different order — so this reports the
/// difference against what two *seeds* of the same backend differ by. Below
/// that, the backends agree as well as the estimator does.
fn agree(scenes: u32, offset: u32, reference_spp: u32) {
    println!("cpu vs gpu over {scenes} scenes at {reference_spp} spp\n");
    let mut across = (0.0f64, 0usize);
    let mut within = (0.0f64, 0usize);

    for index in 0..scenes {
        let cpu = render_reference_on(offset + index, reference_spp, 7, false);
        let gpu = render_reference_on(offset + index, reference_spp, 7, true);
        // The same backend at another seed, as the yardstick.
        let cpu_b = render_reference_on(offset + index, reference_spp, 99, false);
        accumulate(&mut across, &cpu, &gpu);
        accumulate(&mut within, &cpu, &cpu_b);
        println!(
            "  scene {:>3}   cpu-vs-gpu {:.6}   cpu-vs-cpu {:.6}",
            index + 1,
            root(across),
            root(within)
        );
    }

    let a = root(across);
    let w = root(within);
    println!("\ncpu vs gpu          {a:.6}");
    println!("cpu vs cpu (seed)   {w:.6}");
    if a <= w * 1.5 {
        println!("\nthe backends agree within the estimator's own spread.");
    } else {
        println!(
            "\nthe backends DISAGREE by more than seed noise — do not mix them in one dataset."
        );
    }
}

/// One reference render, on a named backend.
fn render_reference_on(index: u32, samples: u32, seed: u64, gpu: bool) -> Vec<f32> {
    let (mut scene, camera) = build_scene(index);
    let mut r = if gpu {
        match threers::raytrace::gpu::GpuBackend::headless() {
            Ok(b) => RaytraceRenderer::with_backend(FRAME, FRAME, Box::new(b)),
            Err(e) => {
                println!("  (no gpu: {e})");
                RaytraceRenderer::new(FRAME, FRAME)
            }
        }
    } else {
        RaytraceRenderer::new(FRAME, FRAME)
    };
    r.set_settings(RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed,
        ..Default::default()
    });
    r.render(&mut scene, &camera).expect("render");
    r.film().resolve_hdr()
}

/// What does each error level cost in samples?
///
/// One reference per scene, then the same scene at a ladder of sample counts,
/// unfiltered and À-Trous filtered. Error falls as `1/sqrt(n)`, so this says
/// directly what an error target implies for render time — and where the target
/// stops being reachable by sampling at all.
fn sweep(scenes: u32, offset: u32, reference_spp: u32) {
    const LADDER: [u32; 7] = [1, 4, 16, 64, 256, 1024, 4096];
    println!("error against a {reference_spp}-sample reference, {scenes} scenes\n");

    let mut raw = [(0.0f64, 0usize); LADDER.len()];
    let mut filtered = [(0.0f64, 0usize); LADDER.len()];
    for index in 0..scenes {
        let reference = render_reference(offset + index, reference_spp, 0x1111_2222_3333_4444);
        for (slot, &spp) in LADDER.iter().enumerate() {
            let (noisy, guides, _, extra) = render_pair_with_depth(offset + index, spp, 1);
            accumulate(&mut raw[slot], &noisy, &reference);
            let cleaned = denoise(
                FRAME,
                FRAME,
                &noisy,
                &DenoiseGuides {
                    albedo: &guides.albedo,
                    normal: &guides.normal,
                    depth: &extra.depth,
                    variance: &extra.variance,
                    scene_scale: extra.scene_scale,
                },
                &DenoiseParams::default(),
            );
            accumulate(&mut filtered[slot], &cleaned, &reference);
        }
        println!("  scene {:>3}/{scenes}", index + 1);
    }

    println!(
        "\n{:>7}  {:>10}  {:>10}  {:>8}",
        "spp", "unfiltered", "a-trous", "filter"
    );
    for (slot, spp) in LADDER.iter().enumerate() {
        let r = root(raw[slot]);
        let f = root(filtered[slot]);
        println!("{spp:>7}  {r:>10.5}  {f:>10.5}  {:>7.2}x", r / f.max(1e-9));
    }
}

/// How well does the reference agree with itself?
///
/// Every error in this crate is measured against a `REF`-sample render, which
/// is not the true image but an estimate of it with its own Monte-Carlo error.
/// No denoiser can be shown to beat that error, because below it the target is
/// noise. This renders each scene twice at `REF` with different seeds and
/// reports the relative L2 between the two — an unbiased estimate of how far
/// *either* of them sits from the truth, up to a factor of sqrt(2).
///
/// Run it before believing any error target: if two references disagree by
/// 1e-2, an error of 1e-6 against one of them is not a better reconstruction,
/// it is a claim to have predicted that reference's particular noise.
fn floor(scenes: u32, offset: u32, reference_spp: u32) {
    println!("reference self-agreement over {scenes} scenes at {reference_spp} spp");
    let mut pair = (0.0f64, 0usize);
    let mut sum_abs = 0.0f64;
    let mut peak = 0.0f32;

    for index in 0..scenes {
        let a = render_reference(offset + index, reference_spp, 0x1111_2222_3333_4444);
        let b = render_reference(offset + index, reference_spp, 0x9999_8888_7777_6666);
        accumulate(&mut pair, &a, &b);
        for i in 0..(FRAME * FRAME) as usize {
            for c in 0..3 {
                let d = (compress(a[i * 4 + c]) - compress(b[i * 4 + c])).abs();
                sum_abs += d as f64;
                peak = peak.max(d);
            }
        }
        println!("  scene {:>3}   running {:.6}", index + 1, root(pair));
    }

    let two_refs = root(pair);
    // Two independent estimates each carry the error; the distance between them
    // is sqrt(2) times either one's distance from the truth.
    let one_ref = two_refs / std::f32::consts::SQRT_2;
    println!("\nreference vs reference   {two_refs:.6}");
    println!(
        "one reference vs truth   {one_ref:.6}  (the floor for any error measured against it)"
    );
    println!("mean |difference|        {:.2e}", sum_abs / pair.1 as f64);
    println!("largest difference       {peak:.2e}");
    println!(
        "\nto halve this floor, render {}x the samples: error falls as 1/sqrt(n).",
        4
    );
}

/// One reference render at a given seed.
fn render_reference(index: u32, samples: u32, seed: u64) -> Vec<f32> {
    let (mut scene, camera) = build_scene(index);
    let mut r = renderer(FRAME, FRAME);
    r.set_settings(RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed,
        ..Default::default()
    });
    r.render(&mut scene, &camera).expect("render");
    r.film().resolve_hdr()
}

/// Score the À-Trous filter on the held-out scenes.
///
/// The filter gets guides the network never sees — depth, and the per-pixel
/// variance the renderer measured — so this is the filter at its best, not a
/// handicapped version of it.
fn baseline(scenes: u32, validation_from: u32, offset: u32, noisy_spp: u32, reference_spp: u32) {
    let params = DenoiseParams::default();
    println!(
        "À-Trous on scenes {}..{}  {params:?}",
        offset + validation_from,
        offset + scenes
    );

    let mut raw = (0.0f64, 0usize);
    let mut filtered = (0.0f64, 0usize);
    // The whole range, not just the held-out tail: in this mode nothing is
    // written, so there is no train/validation split to respect and covering
    // every scene makes the number comparable with an evaluation over a whole
    // set.
    let _ = validation_from;
    for index in 0..scenes {
        let spp = if noisy_spp > 0 {
            noisy_spp
        } else {
            NOISY_SPP[index as usize % NOISY_SPP.len()]
        };
        let (noisy, guides, reference, extra) =
            render_pair_with_depth(offset + index, spp, reference_spp);
        let cleaned = denoise(
            FRAME,
            FRAME,
            &noisy,
            &DenoiseGuides {
                albedo: &guides.albedo,
                normal: &guides.normal,
                depth: &extra.depth,
                variance: &extra.variance,
                scene_scale: extra.scene_scale,
            },
            &params,
        );
        accumulate(&mut raw, &noisy, &reference);
        accumulate(&mut filtered, &cleaned, &reference);
        println!(
            "  scene {:>3}  {spp:>2}spp   raw {:.5}   filtered {:.5}",
            index + 1,
            root(raw),
            root(filtered)
        );
    }
    println!("\nunfiltered  {:.5}", root(raw));
    println!(
        "À-Trous     {:.5}   {:.2}x",
        root(filtered),
        root(raw) / root(filtered).max(1e-9)
    );
}

/// Sum `(a - b)^2 / (b^2 + eps)` over the RGB of an RGBA image, in the same
/// compressed range the dataset stores, so every number here is comparable
/// with the network's.
fn accumulate(into: &mut (f64, usize), image: &[f32], reference: &[f32]) {
    const EPSILON: f64 = 0.01;
    for i in 0..(FRAME * FRAME) as usize {
        for c in 0..3 {
            let y = compress(image[i * 4 + c]) as f64;
            let t = compress(reference[i * 4 + c]) as f64;
            let d = y - t;
            into.0 += d * d / (t * t + EPSILON);
            into.1 += 1;
        }
    }
}

fn root((sum, n): (f64, usize)) -> f32 {
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

fn write_set(path: &str, count: u32, data: &[f32]) {
    let mut bytes = Vec::with_capacity(16 + data.len() * 4);
    bytes.extend_from_slice(b"RLXDN002");
    bytes.extend_from_slice(&TILE.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&IN_PLANES.to_le_bytes());
    bytes.extend_from_slice(&OUT_PLANES.to_le_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("write dataset");
    println!(
        "wrote {path}: {count} tiles, {:.1} MB",
        bytes.len() as f32 / (1 << 20) as f32
    );
}

/// Cut one `TILE x TILE` square out of the rendered frame, planar, inputs then
/// target.
fn write_tile(
    out: &mut Vec<f32>,
    noisy: &[f32],
    guides: &Guides,
    extra: &FilterGuides,
    reference: &[f32],
    tx: u32,
    ty: u32,
) {
    let (x0, y0) = (tx * TILE, ty * TILE);
    let at = |x: u32, y: u32| ((y0 + y) * FRAME + x0 + x) as usize;

    // Colour, compressed.
    for c in 0..3 {
        for y in 0..TILE {
            for x in 0..TILE {
                out.push(compress(noisy[at(x, y) * 4 + c]));
            }
        }
    }
    // Albedo.
    for c in 0..3 {
        for y in 0..TILE {
            for x in 0..TILE {
                out.push(guides.albedo[at(x, y)][c]);
            }
        }
    }
    // Normal.
    for c in 0..3 {
        for y in 0..TILE {
            for x in 0..TILE {
                out.push(guides.normal[at(x, y)][c]);
            }
        }
    }
    // Depth, made scene-relative and bounded. A raw distance is unbounded and
    // means nothing without the scene's size; `d / (d + scale)` puts a hit at
    // the scene's own scale halfway up the range whatever the units are, and a
    // miss — infinity — lands exactly at 1.
    for y in 0..TILE {
        for x in 0..TILE {
            let d = extra.depth[at(x, y)];
            out.push(if d.is_finite() {
                compress(d / extra.scene_scale.max(1e-6))
            } else {
                1.0
            });
        }
    }
    // Standard error of the pixel's mean *relative to that mean* — what
    // adaptive sampling thresholds on.
    //
    // The absolute standard error was tried first and is nearly useless here:
    // measured across a tile set it varied 17 times more between tiles than
    // within them, because its size is set by the scene's brightness and the
    // sample count rather than by anything local. It encoded "how many samples
    // was this scene rendered at", which the network can already read off the
    // colour. Dividing by the pixel's own mean removes both, and what is left
    // is the per-pixel question actually worth asking: how far can this pixel
    // be trusted, compared with its neighbours.
    for y in 0..TILE {
        for x in 0..TILE {
            out.push(compress(extra.error[at(x, y)]));
        }
    }
    // Target, compressed the same way.
    for c in 0..3 {
        for y in 0..TILE {
            for x in 0..TILE {
                out.push(compress(reference[at(x, y) * 4 + c]));
            }
        }
    }
}

/// `x / (1 + x)` — bounded, monotone, and invertible as `y / (1 - y)`.
///
/// Total on purpose. The per-pixel variance is `INFINITY` wherever a pixel took
/// fewer than two samples — which is every pixel of a 1-sample render, a
/// quarter of this dataset — and `inf / (1 + inf)` is NaN, not 1. A single NaN
/// plane poisons every convolution downstream of it, silently, so the boundary
/// is handled here rather than assumed away.
fn compress(v: f32) -> f32 {
    if !v.is_finite() {
        // Infinite variance is maximum uncertainty, which is the top of the
        // range. NaN has no meaning to carry, so it goes to the bottom.
        return if v > 0.0 { 1.0 } else { 0.0 };
    }
    v.max(0.0) / (1.0 + v.max(0.0))
}

struct Guides {
    albedo: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
}

/// The guides the À-Trous filter uses that the network does not take.
struct FilterGuides {
    depth: Vec<f32>,
    variance: Vec<f32>,
    /// Per-pixel relative error — the standard error over the pixel's own mean.
    error: Vec<f32>,
    scene_scale: f32,
}

/// A device and queue kept alive together for the lifetime of the process.
type GpuHandles = (Arc<wgpu::Device>, Arc<wgpu::Queue>);

thread_local! {
    /// One wgpu device for the whole run.
    ///
    /// Adapter setup costs more than a small render, and a dataset is thousands
    /// of them. `None` once we have tried and failed, so the fallback message
    /// is printed once rather than per scene.
    static GPU: RefCell<Option<Option<GpuHandles>>> = const { RefCell::new(None) };
}

/// A renderer on the GPU when `GPU=1` and one is available, else the CPU.
///
/// The dataset has to be the same whichever produced it — see the
/// `AGREE=1` mode, which renders the same scene both ways and reports the
/// difference.
fn renderer(width: u32, height: u32) -> RaytraceRenderer {
    if env_u32("GPU", 0) == 0 {
        return RaytraceRenderer::new(width, height);
    }
    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(match threers::raytrace::gpu::GpuBackend::headless() {
                Ok(backend) => {
                    println!("gpu backend ready");
                    Some(backend.device_and_queue())
                }
                Err(e) => {
                    println!("gpu backend unavailable ({e}); staying on the cpu");
                    None
                }
            });
        }
        match slot.as_ref().and_then(|o| o.clone()) {
            Some((device, queue)) => RaytraceRenderer::with_backend(
                width,
                height,
                Box::new(threers::raytrace::gpu::GpuBackend::with_device(
                    device, queue,
                )),
            ),
            None => RaytraceRenderer::new(width, height),
        }
    })
}

/// Sample counts to draw the noisy half from, cycled across scenes.
///
/// A denoiser trained at one sample count learns that count's noise and is
/// wrong at every other, so the set spans the range a renderer actually stops
/// at. All of them are low on purpose: with next-event estimation, multiple
/// importance sampling and a Sobol sequence, this renderer at 32 samples is
/// already within a third of a percent of converged on scenes this simple —
/// there is nothing there to learn to remove.
const NOISY_SPP: [u32; 4] = [1, 2, 4, 8];

fn render_pair_with_depth(
    index: u32,
    noisy_spp: u32,
    reference_spp: u32,
) -> (Vec<f32>, Guides, Vec<f32>, FilterGuides) {
    let settings = |samples: u32| RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        // High enough to keep a stray path from dominating a whole tile, but
        // not the 10.0 that quietly removes the fireflies a denoiser exists to
        // handle. Both halves clamp identically, so the pair stays consistent.
        clamp_indirect: 100.0,
        // Every pixel takes the full budget: the network is being taught what a
        // given sample count looks like, not what an adaptive sampler left.
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        ..Default::default()
    };

    let (mut scene, camera) = build_scene(index);
    let mut r = renderer(FRAME, FRAME);
    r.set_settings(settings(noisy_spp));
    r.render(&mut scene, &camera).expect("render");
    let noisy = r.film().resolve_hdr();
    let guides = Guides {
        albedo: r.film().resolve_albedo(),
        normal: r.film().resolve_normal(),
    };
    let filter_guides = FilterGuides {
        depth: r.film().resolve_depth(),
        variance: r.film().resolve_variance(),
        error: r.film().resolve_error(),
        scene_scale: r.traced_scene().map(|s| s.scale()).unwrap_or(1.0),
    };

    let (mut scene, camera) = build_scene(index);
    let mut r = renderer(FRAME, FRAME);
    r.set_settings(settings(reference_spp));
    r.render(&mut scene, &camera).expect("render");
    let reference = r.film().resolve_hdr();

    (noisy, guides, reference, filter_guides)
}

// -------------------------------------------------------------- scene set

/// Whether this scene will be enclosed — drawn before the sky so the two
/// choices stay independent of each other and of the room's dimensions.
fn enclosed_pick(rng: &mut Rng) -> bool {
    rng.next() < 0.34
}

/// A sky: a small very bright sun, a gradient above the horizon, a dim warm
/// bounce below.
///
/// The ratio matters more than the shape. A sun thousands of times the zenith
/// is what a real sky has, and it is what makes an environment a hard sampling
/// problem rather than a soft fill.
fn generate_sky(w: usize, h: usize, rng: &mut Rng) -> Vec<f32> {
    let elevation = rng.range(0.12, 1.2);
    let azimuth = rng.range(0.0, std::f32::consts::TAU);
    let sun = Vector3::new(
        elevation.cos() * azimuth.cos(),
        elevation.sin(),
        elevation.cos() * azimuth.sin(),
    );
    let sun_radius = rng.range(0.015, 0.06);
    let sun_radiance = rng.range(2000.0, 14000.0);
    let zenith = Color::new(0.28, 0.45, 0.9);
    let horizon = Color::new(0.85, 0.82, 0.72);

    let mut out = vec![0.0f32; w * h * 4];
    for y in 0..h {
        let theta = (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        for x in 0..w {
            let phi = ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
            let d = Vector3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );
            let up = d.y.max(0.0);
            let mut c = if d.y >= 0.0 {
                Color::new(
                    horizon.r + (zenith.r - horizon.r) * up.powf(0.6),
                    horizon.g + (zenith.g - horizon.g) * up.powf(0.6),
                    horizon.b + (zenith.b - horizon.b) * up.powf(0.6),
                )
            } else {
                // Ground bounce: dim, warm, and not black, or everything below
                // the horizon renders as a hole.
                Color::new(0.12, 0.10, 0.09)
            };
            let scale = rng_free_intensity(elevation);
            c = Color::new(c.r * scale, c.g * scale, c.b * scale);
            if d.dot(sun) > sun_radius.cos() {
                c = Color::new(sun_radiance, sun_radiance * 0.95, sun_radiance * 0.85);
            }
            let i = (y * w + x) * 4;
            out[i] = c.r;
            out[i + 1] = c.g;
            out[i + 2] = c.b;
            out[i + 3] = 1.0;
        }
    }
    out
}

/// Overall sky brightness, dimmer when the sun is low — a sunset should not be
/// as bright as noon, and a network that only ever sees one exposure learns it.
fn rng_free_intensity(elevation: f32) -> f32 {
    0.6 + 2.4 * elevation.sin().max(0.0)
}

/// A deterministic pseudo-random stream, so scene `n` is always scene `n`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn color(&mut self) -> Color {
        Color::new(
            self.range(0.05, 0.9),
            self.range(0.05, 0.9),
            self.range(0.05, 0.9),
        )
    }
}

/// One scene from the family, varied by index.
///
/// Variety matters more than any single scene: a denoiser trained on one
/// arrangement learns that arrangement. Each draw changes the camera, the
/// light, the floor's texture and roughness, and the material of every object —
/// across diffuse, rough metal, clearcoated plastic and glass, which are the
/// four the guides behave differently for.
fn build_scene(index: u32) -> (Scene, PerspectiveCamera) {
    if std::env::var("CLASSIC").is_ok() {
        return build_classic_scene(index);
    }
    if std::env::var("REAL").is_ok() {
        return build_real_scene(index);
    }
    let mut rng = Rng::new(index as u64 + 1);
    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.025, 0.04);

    // Floor: half the scenes get a chequer, so the albedo guide has something
    // to carry that the colour cannot see through the noise.
    let mut floor_mat = StandardMaterial::new(rng.color());
    floor_mat.roughness = rng.range(0.25, 0.9);
    if rng.next() < 0.5 {
        floor_mat.map = Some(chequer(rng.range(3.0, 10.0)));
        floor_mat.color = Color::WHITE;
    }
    // A quarter of scenes are lit by a sky rather than by panels.
    //
    // A different category of light transport again: an environment lights
    // every surface from every direction at once, so there is no shadow-casting
    // source to importance sample toward and the noise is low-frequency rather
    // than the speckle a small emitter gives. Every scene so far was lit by
    // rectangles, which taught the network one kind of noise.
    let sky = !enclosed_pick(&mut rng) && rng.next() < 0.34;
    if sky {
        let (w, h) = (256usize, 128usize);
        let equirect = generate_sky(w, h, &mut rng);
        let cube = PmremGenerator::from_equirect_f32(&equirect, w as u32, h as u32, 64);
        scene.environment = Some(Arc::new(cube));
        scene.background = Color::BLACK;
    }

    // A third of scenes are enclosed rather than open ground.
    //
    // This is the largest single source of variety here, and it is about light
    // transport rather than looks: an open floor under one panel is almost all
    // direct light, and every scene built that way teaches the network the same
    // noise. Walls send most of the energy round at least twice, so indirect
    // illumination dominates, colour bleeds between surfaces and the variance
    // is far higher for the same sample count — which is the regime a denoiser
    // is actually bought for.
    // A sky inside a sealed room would never reach the camera, so the two are
    // exclusive; drawn first so the choice is independent of the room's size.
    let enclosed = !sky && rng.next() < 0.34;
    // Half the rooms are tight, strongly coloured, and **open at the front**,
    // with the camera outside looking in through the missing wall.
    //
    // The first attempt at this made them tight and saturated but left them
    // sealed with the camera inside, and it moved the Cornell box by nothing at
    // all — 13.3% behind before, 13.5% after. A sealed room seen from within
    // shows one or two walls and whatever the camera happens to face. The
    // Cornell arrangement is the opposite: the whole box in one frame, three
    // walls and the floor and the ceiling and the light all visible together,
    // every one of them bouncing onto every other. That is a different
    // visibility structure, and it was the one missing.
    let tight = enclosed && rng.next() < 0.5;
    let room = if tight {
        rng.range(2.2, 3.4)
    } else {
        rng.range(5.0, 9.0)
    };
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(60.0, 60.0),
        Material::Standard(floor_mat),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    if enclosed {
        let half = room * 0.5;
        // Two coloured side walls and two neutral ones: coloured bounce is what
        // makes indirect light legible, and a guide that carries it is what
        // separates a denoiser from a blur.
        // A tight room gets saturated facing walls, the way the Cornell box
        // does: one channel near its maximum and the other two near zero is
        // what makes bounced light legibly coloured rather than merely tinted.
        let (left_wall, right_wall) = if tight {
            let a = rng.range(0.5, 0.75);
            let b = rng.range(0.3, 0.5);
            if rng.next() < 0.5 {
                (Color::new(a, 0.05, 0.05), Color::new(0.1, b, 0.14))
            } else {
                (Color::new(0.1, b, 0.14), Color::new(a, 0.05, 0.05))
            }
        } else {
            (rng.color(), rng.color())
        };
        let walls: [(Vector3, f32, Color); 4] = [
            (Vector3::new(-half, half, 0.0), FRAC_PI_2, left_wall),
            (Vector3::new(half, half, 0.0), -FRAC_PI_2, right_wall),
            (
                Vector3::new(0.0, half, -half),
                0.0,
                Color::new(0.72, 0.72, 0.70),
            ),
            (
                Vector3::new(0.0, half, half),
                std::f32::consts::PI,
                Color::new(0.72, 0.72, 0.70),
            ),
        ];
        for (i, (position, yaw, colour)) in walls.into_iter().enumerate() {
            // An open front: skip the wall the camera sits behind. Index 3 is
            // the +z wall, and a tight room puts the camera on +z below.
            if tight && i == 3 {
                continue;
            }
            let mut m = StandardMaterial::new(colour);
            m.roughness = rng.range(0.6, 1.0);
            let mut wall = Object3D::mesh(Mesh::new(
                PlaneGeometry::new(room, room),
                Material::Standard(m),
            ));
            wall.position = position;
            wall.rotate_y(yaw);
            scene.add(wall);
        }
        let mut ceiling_mat = StandardMaterial::new(Color::new(0.8, 0.8, 0.78));
        ceiling_mat.roughness = 0.9;
        let mut ceiling = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(room, room),
            Material::Standard(ceiling_mat),
        ));
        ceiling.position = Vector3::new(0.0, room, 0.0);
        ceiling.rotate_x(FRAC_PI_2);
        scene.add(ceiling);
    }

    let objects = 2 + (rng.next() * 3.0) as u32;
    for _i in 0..objects {
        let radius = rng.range(0.5, 1.0);
        let x = rng.range(-2.4, 2.4);
        let z = rng.range(-1.6, 1.6);
        let kind = rng.next();
        let material = if kind < 0.35 {
            // Diffuse.
            let mut m = StandardMaterial::new(rng.color());
            m.roughness = rng.range(0.5, 1.0);
            Material::Standard(m)
        } else if kind < 0.6 {
            // Metal, rough to nearly mirrored.
            let mut m = StandardMaterial::new(rng.color());
            m.roughness = rng.range(0.05, 0.55);
            m.metalness = 1.0;
            Material::Standard(m)
        } else if kind < 0.85 {
            // Clearcoated plastic.
            let mut m = PhysicalMaterial::new(rng.color());
            m.roughness = rng.range(0.3, 0.7);
            m.clearcoat = 1.0;
            m.clearcoat_roughness = rng.range(0.02, 0.2);
            Material::Physical(m)
        } else {
            // Glass, whose guides are deferred through it.
            let mut m = PhysicalMaterial::new(Color::WHITE);
            m.transmission = 1.0;
            m.roughness = rng.range(0.0, 0.15);
            m.ior = rng.range(1.3, 1.6);
            Material::Physical(m)
        };
        // Shape chosen at random rather than alternating: a fixed alternation
        // correlates shape with position, and the network can learn that
        // instead of learning the surface.
        let shape = rng.next();
        let mut obj = if shape < 0.3 {
            Object3D::mesh(Mesh::new(SphereGeometry::new(radius, 32, 24), material))
        } else if shape < 0.55 {
            Object3D::mesh(Mesh::new(
                BoxGeometry::new(radius * 1.7, radius * 1.7, radius * 1.7),
                material,
            ))
        } else if shape < 0.75 {
            // A torus is the awkward case for a guide-driven filter: thin, with
            // curvature that turns the normal right round inside a few pixels.
            Object3D::mesh(Mesh::new(
                TorusGeometry::new(radius * 0.8, radius * 0.32, 24, 48, std::f32::consts::TAU),
                material,
            ))
        } else if shape < 0.9 {
            Object3D::mesh(Mesh::new(
                CylinderGeometry::new(
                    radius * 0.7,
                    radius * 0.7,
                    radius * 2.0,
                    32,
                    1,
                    false,
                    0.0,
                    std::f32::consts::TAU,
                ),
                material,
            ))
        } else {
            Object3D::mesh(Mesh::new(
                CylinderGeometry::new(
                    0.0,
                    radius * 0.9,
                    radius * 2.0,
                    32,
                    1,
                    false,
                    0.0,
                    std::f32::consts::TAU,
                ),
                material,
            ))
        };
        obj.position = Vector3::new(x, radius, z);
        obj.rotate_y(rng.range(0.0, std::f32::consts::PI));
        scene.add(obj);
    }

    // A fifth of scenes get a spread of lights rather than one panel: four
    // emitters whose radii span 10x and whose radiance goes as 1/r², so the
    // smallest is a hundred times brighter than the largest.
    //
    // Scored on the Veach arrangement — built for exactly this — the network
    // came out 10.6% behind OpenImageDenoise, against 4.7% *ahead* on the CAD
    // parts it had seen. Every scene here had one panel of one size, so a frame
    // holding a near-mirror highlight from a pinpoint source beside a broad soft
    // one from a large source was a regime it had never been shown.
    if rng.next() < 0.2 {
        for k in 0..4 {
            let radius = 0.06 * 2.15f32.powi(k);
            let mut m = StandardMaterial::new(Color::BLACK);
            let warm = rng.range(0.0, 1.0);
            m.emissive = Color::new(1.0, 0.88 + 0.12 * warm, 0.74 + 0.26 * warm);
            m.emissive_intensity = 2.4 / (radius * radius);
            let mut light = Object3D::mesh(Mesh::new(
                SphereGeometry::new(radius, 24, 16),
                Material::Standard(m),
            ));
            light.position = if tight {
                Vector3::new(
                    rng.range(-0.3, 0.3) * room,
                    room * rng.range(0.75, 0.95),
                    rng.range(-0.3, 0.3) * room,
                )
            } else {
                Vector3::new(
                    rng.range(-2.6, 2.6),
                    rng.range(2.6, 5.0),
                    rng.range(-2.6, 2.6),
                )
            };
            scene.add(light);
        }
    } else {
        // One area light overhead, varying in size and colour temperature. A
        // small one gives hard shadows and high variance; a large one the
        // opposite.
        let mut em = StandardMaterial::new(Color::BLACK);
        let warm = rng.range(0.0, 1.0);
        em.emissive = Color::new(1.0, 0.85 + 0.15 * warm, 0.7 + 0.3 * warm);
        let size = rng.range(0.8, 3.5);
        em.emissive_intensity = rng.range(15.0, 45.0) / size;
        let mut panel = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(size, size),
            Material::Standard(em),
        ));
        // Inside a tight room the ceiling is at `room`, so the usual 3.5-6.0
        // would put the emitter *outside* the box and light the interior only
        // by leakage — which is what the first open-fronted previews showed.
        // Cornell's light is a small panel just under the ceiling; this is that.
        panel.position = if tight {
            Vector3::new(
                rng.range(-0.2, 0.2) * room,
                room - 0.02,
                rng.range(-0.2, 0.2) * room,
            )
        } else {
            Vector3::new(
                rng.range(-1.5, 1.5),
                rng.range(3.5, 6.0),
                rng.range(-1.5, 1.5),
            )
        };
        panel.rotate_x(FRAC_PI_2);
        scene.add(panel);
    }

    // A second, dimmer light in a third of scenes. One light gives every scene
    // the same shadow structure; two give overlapping penumbrae and a shadow
    // that is dark in one direction and lit from another.
    if rng.next() < 0.34 {
        let mut fill = StandardMaterial::new(Color::BLACK);
        let tint = rng.range(0.0, 1.0);
        fill.emissive = Color::new(0.7 + 0.3 * tint, 0.8, 1.0);
        let fill_size = rng.range(0.6, 2.2);
        fill.emissive_intensity = rng.range(4.0, 16.0) / fill_size;
        let mut second = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(fill_size, fill_size),
            Material::Standard(fill),
        ));
        let a = rng.range(0.0, std::f32::consts::TAU);
        second.position = Vector3::new(
            a.cos() * rng.range(2.0, 4.0),
            rng.range(1.5, 3.5),
            a.sin() * rng.range(2.0, 4.0),
        );
        second.rotate_x(FRAC_PI_2);
        scene.add(second);
    }

    let mut camera = PerspectiveCamera::new(rng.range(35.0, 55.0), 1.0, 0.1, 200.0);
    let angle = rng.range(0.0, std::f32::consts::TAU);
    // A sealed room has to be shot from inside it or the frame is a wall; an
    // open-fronted one is shot from outside, square on to the opening, which is
    // what puts the whole interior in one frame.
    let distance = if tight {
        room * rng.range(1.5, 2.1)
    } else if enclosed {
        rng.range(1.8, room * 0.42)
    } else {
        rng.range(5.0, 8.0)
    };
    camera.position = if tight {
        // On +z, where the missing wall is, and near the box's own mid-height.
        Vector3::new(
            rng.range(-0.15, 0.15) * room,
            room * rng.range(0.4, 0.62),
            distance,
        )
    } else {
        Vector3::new(
            angle.cos() * distance,
            if enclosed {
                rng.range(1.0, room * 0.6)
            } else {
                rng.range(1.4, 3.2)
            },
            angle.sin() * distance,
        )
    };
    camera.target = if tight {
        Vector3::new(0.0, room * rng.range(0.42, 0.58), 0.0)
    } else {
        Vector3::new(0.0, rng.range(0.4, 1.2), 0.0)
    };
    (scene, camera)
}

/// The scenes renderers are demonstrated and compared on.
///
/// These are for **scoring only** — never training. Their value is that they are
/// not ours: the Cornell box is a physical measurement other renderers publish
/// against, and the Veach arrangement was built to expose exactly the sampling
/// failure a denoiser has to survive. A number on those means something to
/// someone who has never seen this repo, in a way that a number on a generator
/// we also wrote does not.
///
/// Each is deliberately hostile to a denoiser in a different way:
///
/// | | what it is | what it breaks |
/// |---|---|---|
/// | 0 | Cornell box | almost all light is indirect; colour bleeds off the walls |
/// | 1 | Cornell box, glass and metal | a caustic under the glass, which NEE cannot sample |
/// | 2 | Veach MIS | four glossy plates against four light sizes at once |
/// | 3 | furnace test | uniform environment, so the answer is a known constant |
fn build_classic_scene(index: u32) -> (Scene, PerspectiveCamera) {
    match index % 4 {
        0 => cornell_box(false),
        1 => cornell_box(true),
        2 => veach_mis(),
        _ => furnace_test(),
    }
}

/// The Cornell box, at the proportions the original measurements imply.
///
/// `objects` swaps the two diffuse blocks for a glass sphere and a metal one.
/// The glass is the harder case by far: the bright spot it focuses onto the
/// floor arrives only along BSDF-sampled paths, so it is the noisiest thing in
/// any of these scenes and the first place over-smoothing shows.
fn cornell_box(objects: bool) -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;

    let white = || Material::Standard(StandardMaterial::new(Color::new(0.73, 0.73, 0.73)));
    let red = || Material::Standard(StandardMaterial::new(Color::new(0.65, 0.05, 0.05)));
    let green = || Material::Standard(StandardMaterial::new(Color::new(0.12, 0.45, 0.15)));
    let s = 2.0;

    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    let mut ceiling = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    ceiling.position = Vector3::new(0.0, s, 0.0);
    ceiling.rotate_x(FRAC_PI_2);
    scene.add(ceiling);

    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    back.position = Vector3::new(0.0, s * 0.5, -s * 0.5);
    scene.add(back);

    let mut left = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), red()));
    left.position = Vector3::new(-s * 0.5, s * 0.5, 0.0);
    left.rotate_y(FRAC_PI_2);
    scene.add(left);

    let mut right = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), green()));
    right.position = Vector3::new(s * 0.5, s * 0.5, 0.0);
    right.rotate_y(-FRAC_PI_2);
    scene.add(right);

    // Emissive geometry rather than a light, so the panel casts soft shadows
    // and shows up in anything reflective.
    let mut lamp_material = StandardMaterial::new(Color::BLACK);
    lamp_material.emissive = Color::new(1.0, 0.92, 0.78);
    lamp_material.emissive_intensity = 22.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.6, 0.6),
        Material::Standard(lamp_material),
    ));
    lamp.position = Vector3::new(0.0, s - 0.01, 0.0);
    lamp.rotate_x(FRAC_PI_2);
    scene.add(lamp);

    if objects {
        let mut glass = PhysicalMaterial::new(Color::WHITE);
        glass.transmission = 1.0;
        glass.roughness = 0.0;
        glass.ior = 1.52;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.35, 64, 48),
            Material::Physical(glass),
        ));
        ball.position = Vector3::new(0.38, 0.35, 0.25);
        scene.add(ball);

        let mut metal = PhysicalMaterial::new(Color::new(0.9, 0.9, 0.92));
        metal.metalness = 1.0;
        metal.roughness = 0.08;
        let mut mirror = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.35, 64, 48),
            Material::Physical(metal),
        ));
        mirror.position = Vector3::new(-0.38, 0.35, -0.25);
        scene.add(mirror);
    } else {
        let mut tall = StandardMaterial::new(Color::new(0.75, 0.75, 0.72));
        tall.roughness = 0.85;
        let mut block = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.6, 1.2, 0.6),
            Material::Standard(tall),
        ));
        block.position = Vector3::new(-0.35, 0.6, -0.3);
        block.rotate_y(0.3);
        scene.add(block);

        let mut short = StandardMaterial::new(Color::new(0.75, 0.75, 0.72));
        short.roughness = 0.85;
        let mut cube = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.6, 0.6, 0.6),
            Material::Standard(short),
        ));
        cube.position = Vector3::new(0.35, 0.3, 0.35);
        cube.rotate_y(-0.3);
        scene.add(cube);
    }

    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.0, 3.9);
    camera.target = Vector3::new(0.0, 1.0, 0.0);
    (scene, camera)
}

/// Veach's multiple-importance-sampling scene.
///
/// Four plates of increasing roughness face four spherical lights of increasing
/// size and decreasing intensity. Every combination of "sample the light" and
/// "sample the surface" is right somewhere in the frame and wrong somewhere
/// else, which is what the scene was built to show. For a denoiser it means a
/// single image containing near-mirror highlights beside broad rough ones — the
/// filter has to be gentle in one place and aggressive a few pixels away.
fn veach_mis() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.01, 0.012, 0.02);

    let mut backdrop_mat = StandardMaterial::new(Color::new(0.35, 0.35, 0.38));
    backdrop_mat.roughness = 0.9;
    let mut backdrop = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(20.0, 20.0),
        Material::Standard(backdrop_mat),
    ));
    backdrop.position = Vector3::new(0.0, 0.0, -4.0);
    scene.add(backdrop);

    // Roughness spans mirror-like to nearly diffuse. The tilt is what makes the
    // scene work: it has to put the lights on the mirror direction from the
    // camera, or the plates reflect the empty room and the arrangement shows
    // nothing. With the lights behind and above and the camera in front, that
    // angle is the half-vector between the two — about ten degrees, so the
    // plates lie nearly flat. Tilting them up to meet the lights points the
    // mirror direction into the backdrop instead, which is the failure this
    // arrangement is easy to build by accident.
    for (i, roughness) in [0.005f32, 0.02, 0.08, 0.25].iter().enumerate() {
        let mut mat = PhysicalMaterial::new(Color::new(0.9, 0.9, 0.9));
        mat.metalness = 1.0;
        mat.roughness = *roughness;
        let mut plate = Object3D::mesh(Mesh::new(
            BoxGeometry::new(4.2, 0.06, 1.0),
            Material::Physical(mat),
        ));
        // Receding like steps, each further plate higher than the last, so all
        // four are in frame at once rather than the nearest hiding the rest.
        plate.position = Vector3::new(0.0, -1.5 + i as f32 * 0.44, -0.3 - i as f32 * 1.25);
        plate.rotate_x(0.18);
        scene.add(plate);
    }

    // Constant total power, so the small light is intense and the large one
    // faint — radiance goes as 1/r² for a fixed emitted power. That is the
    // whole point: sampling the light wins for the small ones and sampling the
    // surface wins for the large ones, and both are in frame at once.
    for (i, radius) in [0.05f32, 0.15, 0.3, 0.5].iter().enumerate() {
        let mut mat = StandardMaterial::new(Color::BLACK);
        mat.emissive = Color::new(1.0, 0.95, 0.85);
        mat.emissive_intensity = 3.0 / (radius * radius);
        let mut light = Object3D::mesh(Mesh::new(
            SphereGeometry::new(*radius, 32, 24),
            Material::Standard(mat),
        ));
        light.position = Vector3::new(-2.1 + i as f32 * 1.4, 2.2, -2.6);
        scene.add(light);
    }

    let mut camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 2.4, 5.0);
    camera.target = Vector3::new(0.0, -0.9, -2.2);
    (scene, camera)
}

/// The furnace test: a grey sphere inside a uniform emissive environment.
///
/// With albedo `a` under uniform illumination `L` the sphere must converge to
/// exactly `L` — the multiple-scattering series sums to one — so the whole frame
/// should be a single flat value and the object should vanish. Anything visible
/// is energy the BSDF gained or lost. It is included here because it is the one
/// scene whose correct answer is known in closed form rather than by rendering
/// it for longer, which makes it the only entry that can catch a denoiser
/// *inventing* structure: any edge it draws around the sphere is an artifact by
/// construction.
fn furnace_test() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.5, 0.5, 0.5);

    // A large emissive shell standing in for a uniform environment.
    let mut shell_mat = StandardMaterial::new(Color::BLACK);
    shell_mat.emissive = Color::WHITE;
    shell_mat.emissive_intensity = 1.0;
    // BackSide: the camera is inside the shell, so its inward faces are the
    // ones that have to emit.
    shell_mat.side = 1;
    let shell = Object3D::mesh(Mesh::new(
        SphereGeometry::new(30.0, 48, 32),
        Material::Standard(shell_mat),
    ));
    scene.add(shell);

    for (x, roughness, metalness) in [(-1.1f32, 0.35f32, 0.0f32), (1.1, 0.15, 1.0)] {
        let mut mat = PhysicalMaterial::new(Color::new(0.8, 0.8, 0.8));
        mat.roughness = roughness;
        mat.metalness = metalness;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.9, 64, 48),
            Material::Physical(mat),
        ));
        ball.position = Vector3::new(x, 0.0, 0.0);
        scene.add(ball);
    }

    let mut camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 5.5);
    camera.target = Vector3::ZERO;
    (scene, camera)
}

/// Solids from `tests/openscad-corpus`, which exist to check the OpenSCAD front
/// end and happen to be exactly the geometry a procedural sphere-and-box
/// generator never produces. `surf_dat` is omitted: it reads a sibling `.dat`
/// and the pair is about the data loader rather than about a shape.
const SCAD_MODELS: [&str; 13] = [
    "booleans_2d",
    "box_notch",
    "box_slot",
    "default_cyl",
    "mink_box",
    "mink2d",
    "plate_2d",
    "plate_with_hole",
    "proj_cut",
    "resize_box",
    "rounded_2d",
    "sheared",
    "stacked_boxes",
];

/// A scene built from the repo's CAD assemblies rather than from primitives.
///
/// Every measurement so far has been taken on procedural rooms of spheres,
/// boxes and tori — the same distribution the network trained on. A denoiser
/// scored only on the shapes it was taught cannot tell you whether it has
/// learned to denoise or learned that distribution, and the number it produces
/// is the same either way.
///
/// These are real parts: a NEMA 17 with its lamination stack, tie-rods and
/// wiring, and a printer assembly with extrusions, belts and a gantry. Thin
/// features, long thin triangles, coplanar faces, small concave corners that
/// trap light — none of which a sphere on a floor produces. The set is meant
/// for scoring, not training.
fn build_real_scene(index: u32) -> (Scene, PerspectiveCamera) {
    use threers::{build_nema17, build_printer, parse_scad_file};

    let mut rng = Rng::new(index as u64 + 7717);
    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.025, 0.04);

    // Two assemblies alone are still a narrow set — the whole reason the
    // procedural corpus failed. The `.scad` corpus adds thirteen more solids,
    // and they carry the features the assemblies do not: through-holes, notches
    // and slots, minkowski fillets, sheared and resized bodies. Cheap breadth
    // from geometry the repo already has to hand.
    let parts = match index % 4 {
        0 => build_nema17(),
        1 => build_printer(
            rng.range(180.0, 240.0),
            rng.range(180.0, 240.0),
            rng.range(180.0, 250.0),
            rng.range(20.0, 40.0),
            rng.range(0.2, 0.8),
            rng.range(0.2, 0.8),
            rng.range(0.2, 0.8),
        ),
        _ => {
            // Indexed by `index`, not `index / 4`: the latter is constant across
            // the pair of scenes that reach this arm, so every model would be
            // rendered twice in a row and the corpus would cycle half as fast.
            let pick = SCAD_MODELS[index as usize % SCAD_MODELS.len()];
            let path = format!("tests/openscad-corpus/{pick}.scad");
            match parse_scad_file(&path) {
                // A handful of these are 2D or degenerate once evaluated; fall
                // back rather than emitting a tile of empty sky, which would
                // teach the network that this geometry means nothing is there.
                Ok(solid) => {
                    let g = solid.to_geometry();
                    let verts = g.positions().into_iter().flatten().count();
                    // Several of these sources are 2D. A flat sheet renders as
                    // an edge-on sliver or fills the frame depending on the
                    // camera, and either way it is not the CAD geometry this
                    // arm exists to add — so require thickness in all three
                    // axes, not merely a non-empty vertex list.
                    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                    for p in g.positions().into_iter().flatten() {
                        let v = [p.x, p.y, p.z];
                        for a in 0..3 {
                            lo[a] = lo[a].min(v[a]);
                            hi[a] = hi[a].max(v[a]);
                        }
                    }
                    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
                    let longest = span.iter().cloned().fold(0.0f32, f32::max);
                    let thinnest = span.iter().cloned().fold(f32::INFINITY, f32::min);
                    let solid_enough = verts >= 3 && longest > 0.0 && thinnest / longest > 0.02;
                    if std::env::var("PREVIEW").is_ok() {
                        println!(
                            "    {pick}: {verts} verts, span {:.2}x{:.2}x{:.2}{}",
                            span[0],
                            span[1],
                            span[2],
                            if solid_enough {
                                ""
                            } else {
                                "  <- too flat, skipped"
                            }
                        );
                    }
                    if solid_enough {
                        vec![(g, [0.72, 0.74, 0.78])]
                    } else {
                        build_nema17()
                    }
                }
                Err(_) => build_nema17(),
            }
        }
    };

    // Fit the assembly into roughly the same world scale as the procedural
    // scenes, so the light intensities and camera distances below carry over
    // and the two sets are lit comparably.
    let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for (geom, _) in &parts {
        for p in geom.positions().into_iter().flatten() {
            let v = [p.x, p.y, p.z];
            for a in 0..3 {
                mn[a] = mn[a].min(v[a]);
                mx[a] = mx[a].max(v[a]);
            }
        }
    }
    let extent = (0..3).fold(0.0f32, |m, a| m.max(mx[a] - mn[a])).max(1e-6);
    let k = 2.4 / extent;
    let centre = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];

    for (geom, colour) in &parts {
        let mut mat = StandardMaterial::new(Color::new(colour[0], colour[1], colour[2]));
        // Machined parts: mostly metal, a few of them nearly mirrored. This is
        // the material mix the procedural set under-samples.
        mat.metalness = if rng.next() < 0.6 { 1.0 } else { 0.1 };
        mat.roughness = rng.range(0.15, 0.75);
        let mut obj = Object3D::mesh(Mesh::new(geom.clone(), Material::Standard(mat)));
        obj.scale = Vector3::new(k, k, k);
        obj.position = Vector3::new(-centre[0] * k, -mn[1] * k, -centre[2] * k);
        scene.add(obj);
    }

    // Ground, so there is somewhere for the assembly's shadow to land.
    let mut floor_mat = StandardMaterial::new(rng.color());
    floor_mat.roughness = rng.range(0.3, 0.9);
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(60.0, 60.0),
        Material::Standard(floor_mat),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    let mut em = StandardMaterial::new(Color::BLACK);
    let warm = rng.range(0.0, 1.0);
    em.emissive = Color::new(1.0, 0.85 + 0.15 * warm, 0.7 + 0.3 * warm);
    let size = rng.range(1.0, 3.0);
    em.emissive_intensity = rng.range(20.0, 50.0) / size;
    let mut panel = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(size, size),
        Material::Standard(em),
    ));
    panel.position = Vector3::new(
        rng.range(-1.5, 1.5),
        rng.range(3.5, 6.0),
        rng.range(-1.5, 1.5),
    );
    panel.rotate_x(FRAC_PI_2);
    scene.add(panel);

    let mut camera = PerspectiveCamera::new(rng.range(35.0, 55.0), 1.0, 0.1, 200.0);
    let angle = rng.range(0.0, std::f32::consts::TAU);
    // Matched to the procedural scenes: they frame objects about two units
    // across from five to eight units away. The assemblies are normalised to a
    // 2.4-unit extent above, so anything closer fills the frame with one face
    // and the tiles stop containing a shadow, a floor or a silhouette — the
    // things the guides are supposed to help with.
    let distance = rng.range(5.0, 8.0);
    camera.position = Vector3::new(
        angle.cos() * distance,
        rng.range(1.4, 3.2),
        angle.sin() * distance,
    );
    camera.target = Vector3::new(0.0, rng.range(0.5, 1.4), 0.0);
    (scene, camera)
}

fn chequer(repeat: f32) -> Arc<Texture> {
    let n = 16u32;
    let mut px = vec![255u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let on = (x + y) % 2 == 0;
            let i = ((y * n + x) * 4) as usize;
            px[i] = if on { 235 } else { 30 };
            px[i + 1] = if on { 215 } else { 45 };
            px[i + 2] = if on { 190 } else { 65 };
        }
    }
    let mut tex = Texture::new(n, n, TextureFormat::Rgba8UnormSrgb, px);
    tex.wrap_s = TextureWrap::Repeat;
    tex.wrap_t = TextureWrap::Repeat;
    tex.repeat = Vector2::new(repeat, repeat);
    tex.flip_y = false;
    Arc::new(tex)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
