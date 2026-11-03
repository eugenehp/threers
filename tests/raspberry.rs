//! Invariants for the procedural berry in `examples/raspberry.rs`.
//!
//! The generator is seeded, so these are cheap: build a berry, assert something
//! that has to be true of any berry, and look at the picture it makes.
//!
//! Every test here is a bug that happened. The packing came out as one lumpy
//! hull rather than separate drupelets; the skin's `vertex_color.a` is a
//! *storage buffer index* and nothing was checking it was in range; the fuzz
//! was first built four times life size, which is a dandelion; and the studio
//! was balanced thirty to one, which clips to a single flat red and reads as
//! plastic. None of those is visible in a compile, and three of the four are
//! only visible if you look at the image.

use std::sync::Arc;

use threers::{HeadlessRenderer, PmremGenerator, Scene, ToneMapping, Vector3};

#[allow(dead_code)]
#[path = "../examples/raspberry.rs"]
mod berry;

use berry::*;

/// A small render of the same scene the example composes.
fn render(fuzz: usize, w: u32, h: u32) -> Option<Vec<u8>> {
    let lights = studio_lights();
    let (equirect, sh) = build_studio(&lights);
    let cube = PmremGenerator::from_equirect_f32(&equirect, SKY_W as u32, SKY_H as u32, 128);
    let cube = Arc::new(PmremGenerator::generate_pmrem(&cube, 128));

    let drups = pack_drupelets(DRUPELETS);
    let skin = build_skin(&drups);

    let args = Args {
        w,
        h,
        fuzz,
        supersample: 2,
        ..Args::default()
    };
    let camera = studio_camera(w as f32 / h as f32);

    let mut r = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(args.supersample)
        .build()
        .ok()?;
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    let mut scene = build_scene(&drups, &skin, &sh, &lights, &args, &camera, cube);
    Some(r.render_to_rgba_resolved(&mut scene, &camera))
}

fn luma(px: &[u8], i: usize) -> f32 {
    0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
}

fn children(scene: &Scene) -> usize {
    scene
        .arena
        .get(scene.root)
        .expect("scene root")
        .children
        .len()
}

/// Drupelets have to be separate bodies that merely touch. Packed any tighter
/// they fuse into one lumpy hull, which is what the first version did and it
/// rendered a blackberry.
#[test]
fn the_drupelets_touch_their_neighbours_without_swallowing_them() {
    let drups = pack_drupelets(DRUPELETS);
    assert_eq!(drups.len(), DRUPELETS);

    let mut min_neighbours = usize::MAX;
    for (i, a) in drups.iter().enumerate() {
        let mut overlaps = 0;
        for (j, b) in drups.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = (a.center - b.center).length();
            assert!(
                d + a.radius.min(b.radius) > a.radius.max(b.radius),
                "drupelet {i} is swallowed whole by {j}"
            );
            if d < a.radius + b.radius {
                overlaps += 1;
            }
        }
        min_neighbours = min_neighbours.min(overlaps);
    }
    // Hexagonal packing gives six; the apex and the rim give fewer. Below three
    // the berry has holes in it.
    assert!(
        min_neighbours >= 3,
        "the loneliest drupelet touches only {min_neighbours} others"
    );
}

/// `vertex_color.a` is not a colour: it is the index the skin shader uses to
/// read the drupelet's centre and radius out of `u_s0`. Out of range it reads
/// whatever is next in the buffer — another drupelet's centre — so the bead
/// maths comes out wrong, silently, on a handful of triangles.
#[test]
fn every_skin_vertex_names_a_drupelet_that_exists() {
    let drups = pack_drupelets(DRUPELETS);
    let skin = build_skin(&drups);
    let col = skin
        .attributes
        .get("color")
        .expect("the skin carries a colour attribute");
    assert_eq!(col.item_size, 4);

    let n = drups.len() as f32;
    let mut seen = vec![false; drups.len()];
    for v in col.array.chunks_exact(4) {
        for (k, c) in v.iter().take(3).enumerate() {
            assert!(
                (0.0..=1.0).contains(c),
                "mask channel {k} is {c}, outside [0, 1]"
            );
        }
        let idx = v[3];
        assert!(
            idx >= 0.0 && idx < n && (idx - idx.round()).abs() < 1e-3,
            "drupelet index {idx} is not a whole number in [0, {n})"
        );
        seen[idx as usize] = true;
    }
    assert!(
        seen.iter().all(|s| *s),
        "{} drupelets contributed no vertices at all",
        seen.iter().filter(|s| !**s).count()
    );
}

/// The shells interpenetrate, so a good deal of every sphere is inside its
/// neighbours and must not be drawn. If the cull stops working the triangle
/// count jumps and the crevices fill with shells fighting for depth.
#[test]
fn the_faces_nobody_can_see_are_gone() {
    let drups = pack_drupelets(DRUPELETS);
    let skin = build_skin(&drups);
    let tris = skin.index.as_ref().expect("indexed").len() / 3;
    let uncut = DRUPELETS * DRUP_U * DRUP_V * 2;
    assert!(
        tris < uncut * 95 / 100,
        "the cull kept {tris} of {uncut} drupelet triangles — it is not culling"
    );
    assert!(
        tris > uncut / 3,
        "the cull kept only {tris} of {uncut} — it is eating the visible surface"
    );
}

/// A hair is about fifteen microns across and the berry is two centimetres, so
/// at any sane resolution a hair covers a fraction of a pixel. Drawn at its
/// true width it is hit or missed depending on where the pixel centre falls,
/// which is a definition of aliasing — so a sub-pixel hair is widened to a
/// pixel and faded by the same factor.
///
/// The test is in two halves and the second is the one that matters: without
/// it the rule could be "never widen anything" and still pass.
#[test]
fn no_hair_is_drawn_thinner_than_a_pixel() {
    let drups = pack_drupelets(DRUPELETS);
    let lights = studio_lights();
    let (_, sh) = build_studio(&lights);
    let args = Args {
        fuzz: 4_000,
        w: 400,
        h: 500,
        supersample: 2,
        ..Args::default()
    };
    let camera = studio_camera(args.w as f32 / args.h as f32);

    let g = build_fuzz(&drups, &sh, &lights, &args, &camera);
    let col = g.attributes.get("color").expect("fuzz colour");
    let alphas: Vec<f32> = col.array.chunks_exact(4).map(|v| v[3]).collect();
    assert!(!alphas.is_empty(), "no hairs were built at all");
    for a in &alphas {
        assert!((0.0..=1.0).contains(a), "a hair came back with coverage {a}");
    }
    let faded = alphas.iter().filter(|a| **a < 0.95).count();
    assert!(
        faded * 2 > alphas.len(),
        "only {faded} of {} hair vertices were faded — at this resolution most \
         of them are thinner than a pixel and should have been",
        alphas.len()
    );
}

/// The studio is projected onto nine spherical harmonics on the CPU and the
/// same basis is reconstructed in WGSL. A sign or an axis swapped between the
/// two is invisible in a still — the berry simply lights from somewhere else —
/// so the projection is checked against the sky it is a projection of.
#[test]
fn the_spherical_harmonics_agree_with_the_sky_they_came_from() {
    let lights = studio_lights();
    let (_, sh) = build_studio(&lights);

    // Band 0 is the mean radiance over the sphere, which is also what a large
    // number of samples of the sky itself averages to.
    let n = 20_000;
    let mut mean = [0.0f64; 3];
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let a = std::f32::consts::PI * (3.0 - 5.0f32.sqrt()) * i as f32;
        let c = studio_radiance(Vector3::new(r * a.cos(), z, r * a.sin()), &lights);
        for (m, v) in mean.iter_mut().zip(c.iter()) {
            *m += *v as f64;
        }
    }
    for k in 0..3 {
        let from_sh = (sh[0][k] * 0.282_095) as f64;
        let direct = mean[k] / n as f64;
        let err = (from_sh - direct).abs() / direct.max(1e-6);
        assert!(
            err < 0.05,
            "channel {k}: band 0 says {from_sh:.3}, the sky averages {direct:.3}"
        );
    }

    // And the direction, which is the half an axis swap breaks: a surface
    // turned towards the key has to receive far more than one turned away.
    let key = studio_lights()[0].dir;
    let towards = sh_irradiance(&sh, key)[0];
    let away = sh_irradiance(&sh, Vector3::new(-key.x, -key.y, -key.z))[0];
    // Three to one, measured. The fill and the rim strip light the other side
    // too, so this is not a shadow test — it is only asking that the bright
    // half of the sky is the half the key is in.
    assert!(
        towards > 2.5 * away,
        "facing the key gives {towards:.2} and facing away {away:.2}; the \
         first band is not pointing where the key is"
    );
}

/// What the picture has to be: a red berry with real tonal range.
///
/// Flat and clipped is the failure this catches, and it kept happening. The
/// symptom is one saturated red with no form in it, and the cause is a studio
/// balanced too evenly — prefiltering averages a panel that dim down to the
/// ambient, which leaves nothing over to be a highlight.
///
/// Measured on the red channel and not on luma, because luma is the wrong
/// instrument here: a saturated red at (152, 8, 9) has a luma of 39, so a berry
/// with a hundred levels of modelling in it reads as nearly flat in luma.
#[test]
fn the_berry_renders_red_with_tonal_range() {
    let Some(px) = render(6_000, 180, 225) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let total = 180 * 225;

    // The backdrop is a warm grey pool, so red dominance separates subject from
    // background far more cleanly than brightness does.
    let redness: Vec<f32> = (0..px.len())
        .step_by(4)
        .map(|i| px[i] as f32 - 0.5 * (px[i + 1] as f32 + px[i + 2] as f32))
        .collect();
    let peak = redness.iter().cloned().fold(0.0f32, f32::max);
    assert!(peak > 80.0, "nothing in the frame is strongly red");

    let body: Vec<usize> = (0..redness.len())
        .filter(|k| redness[*k] > 0.18 * peak)
        .collect();
    let n = body.len();
    assert!(
        n * 4 > total && n * 4 < total * 3,
        "the berry covers {n} of {total} pixels, which is not a berry in frame"
    );

    let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
    for k in &body {
        let i = k * 4;
        r += px[i] as f64;
        g += px[i + 1] as f64;
        b += px[i + 2] as f64;
    }
    let (r, g, b) = (r / n as f64, g / n as f64, b / n as f64);
    assert!(
        r > 3.0 * g && r > 3.0 * b,
        "mean berry colour is ({r:.0}, {g:.0}, {b:.0}) — not red"
    );

    let mut red: Vec<u8> = body.iter().map(|k| px[k * 4]).collect();
    red.sort_unstable();
    let p05 = red[n / 20] as i32;
    let p95 = red[n - 1 - n / 20] as i32;
    assert!(
        p95 - p05 > 70,
        "the red channel spans only {} between the 5th and 95th percentile \
         ({p05}..{p95}) — the berry is flat",
        p95 - p05
    );
    assert!(p95 < 250, "the 95th percentile is {p95}: the berry is clipping");
}

/// The fuzz is there to break the silhouette. A berry without it has a hard
/// edge and reads as plastic — a claim worth measuring rather than asserting in
/// a doc comment, so: the hairs have to put light *outside* the body's outline,
/// as much as the hairs themselves account for, and more hairs have to put more
/// of it.
///
/// The amount is not a constant picked off an old render. This used to demand
/// 200 pixels lifted by more than 4 luma, which held only while transparent
/// draws were blended in linear light: a hair is ~15 µm across, so its
/// coverage of a pixel is a few percent, and `encode(α·c)` lifts a near-black
/// backdrop several times more than three.js's `α·encode(c)` on canvas bytes.
/// When blending moved to encoded values the count fell to ~130 with the hairs
/// unchanged. [`predicted_halo`] now derives the expectation from the fuzz
/// geometry and three.js's compositing; it lands within ~10% of the GPU in
/// total lift, and the linear-light rule it rules out is ~3.5x away.
#[test]
fn the_fuzz_reaches_past_the_silhouette() {
    let Some(bare) = render(0, 180, 225) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // The outline of the berry itself, taken from the render with no hairs.
    let red_at = |px: &[u8], k: usize| -> f32 {
        px[k * 4] as f32 - 0.5 * (px[k * 4 + 1] as f32 + px[k * 4 + 2] as f32)
    };
    let count = bare.len() / 4;
    let peak = (0..count).map(|k| red_at(&bare, k)).fold(0.0f32, f32::max);
    let body: Vec<bool> = (0..count).map(|k| red_at(&bare, k) > 0.18 * peak).collect();

    // Pixels outside that outline which the hairs have lifted off the backdrop.
    let halo = |px: &[u8]| -> usize {
        (0..count)
            .filter(|k| !body[*k])
            .filter(|k| luma(px, k * 4) > luma(&bare, k * 4) + 4.0)
            .count()
    };
    assert_eq!(halo(&bare), 0, "the hairless render differs from itself");

    // Total luma the hairs add outside the outline: the halo's energy, which
    // unlike a thresholded count does not hinge on pixels sitting near the
    // threshold.
    let lift = |px: &[u8]| -> f32 {
        (0..count)
            .filter(|k| !body[*k])
            .map(|k| (luma(px, k * 4) - luma(&bare, k * 4)).max(0.0))
            .sum()
    };
    let some_px = render(12_000, 180, 225).expect("render");
    let some = halo(&some_px);
    let more = halo(&render(36_000, 180, 225).expect("render"));

    let (want_px, want_lift) = predicted_halo(12_000, 180, 225, &bare, &body, false);
    let (_, linear_lift) = predicted_halo(12_000, 180, 225, &bare, &body, true);
    assert!(
        linear_lift > 2.0 * want_lift,
        "linear-light and encoded compositing predict {linear_lift:.0} and \
         {want_lift:.0}: too close for this test to tell them apart"
    );
    let got_lift = lift(&some_px);
    assert!(
        (got_lift / want_lift - 1.0).abs() < 0.25,
        "twelve thousand hairs added {got_lift:.0} luma outside the body; their \
         geometry, composited as three.js does, adds {want_lift:.0} \
         (linear-light blending would add {linear_lift:.0})"
    );
    assert!(
        some as f32 > 0.5 * want_px as f32 && want_px > 100,
        "twelve thousand hairs lit {some} pixels outside the body by more than \
         4 luma; their geometry predicts {want_px}"
    );
    assert!(
        more > some,
        "tripling the hair count did not widen the halo ({some}, then {more})"
    );
}

/// The scene has the four meshes it says it has, and drops the two that are
/// optional when they are switched off.
#[test]
fn the_scene_carries_backdrop_skin_styles_and_fuzz() {
    let drups = pack_drupelets(DRUPELETS);
    let skin = build_skin(&drups);
    let lights = studio_lights();
    let (equirect, sh) = build_studio(&lights);
    let cube = || {
        Arc::new(PmremGenerator::from_equirect_f32(
            &equirect,
            SKY_W as u32,
            SKY_H as u32,
            32,
        ))
    };
    let camera = studio_camera(0.8);

    let all = Args {
        fuzz: 500,
        ..Args::default()
    };
    let scene = build_scene(&drups, &skin, &sh, &lights, &all, &camera, cube());
    assert_eq!(children(&scene), 4, "backdrop, skin, styles and fuzz");

    let bare = Args {
        fuzz: 0,
        styles: false,
        ..Args::default()
    };
    let scene = build_scene(&drups, &skin, &sh, &lights, &bare, &camera, cube());
    assert_eq!(children(&scene), 2, "backdrop and skin only");
}

/// The halo the fuzz *should* make — pixels outside `body` lifted by more than
/// 4 luma, and the total lift — rasterised on the CPU from the very
/// geometry `build_fuzz` hands the GPU — so the expectation comes from the
/// hairs and from three.js's compositing rules, not from an earlier render.
///
/// Each ribbon triangle is sampled at supersample-pixel centres; a fragment
/// does what `raspberry_fuzz.wgsl` does (round profile, the 0.02 discard,
/// exposure, `pbr_neutral`, sRGB encode) and is then blended the way a WebGL
/// canvas blends: `src·α + dst·(1 − α)` on the *encoded* bytes, since the
/// drawing buffer is a plain RGBA8 surface and the shader has already encoded.
/// The supersampled frame is box-filtered in linear light, as
/// `render_to_rgba_resolved` does, and the backdrop under each hair is the
/// hairless render.
///
/// `linear_blend` composites in linear light instead — what a hardware sRGB
/// target does, and what the renderer did before encoded blending landed.
fn predicted_halo(
    fuzz: usize,
    w: u32,
    h: u32,
    bare: &[u8],
    body: &[bool],
    linear_blend: bool,
) -> (usize, f32) {
    let lights = studio_lights();
    let (_, sh) = build_studio(&lights);
    let drups = pack_drupelets(DRUPELETS);
    let ss = 2u32;
    let args = Args { w, h, fuzz, supersample: ss, ..Args::default() };
    let camera = studio_camera(w as f32 / h as f32);
    let g = build_fuzz(&drups, &sh, &lights, &args, &camera);
    let pos = &g.get_attribute("position").expect("position").array;
    let col = &g.get_attribute("color").expect("color").array;
    let uv = &g.get_attribute("uv").expect("uv").array;
    let index = g.index.as_ref().expect("indexed").to_vec();

    let (sw, sh_) = ((w * ss) as usize, (h * ss) as usize);
    // Camera basis; three.js's look-at with +Y up.
    let sub = |a: Vector3, b: Vector3| Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z);
    let dot = |a: Vector3, b: Vector3| a.x * b.x + a.y * b.y + a.z * b.z;
    let cross = |a: Vector3, b: Vector3| {
        Vector3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    };
    let unit = |a: Vector3| {
        let l = dot(a, a).sqrt();
        Vector3::new(a.x / l, a.y / l, a.z / l)
    };
    let f = unit(sub(camera.target, camera.position));
    let r = unit(cross(f, Vector3::new(0.0, 1.0, 0.0)));
    let u = cross(r, f);
    let t = (camera.fov * 0.5).tan();
    let project = |v: usize| -> [f32; 2] {
        let d = sub(Vector3::new(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]), camera.position);
        let z = dot(d, f);
        let nx = dot(d, r) / (z * t * camera.aspect);
        let ny = dot(d, u) / (z * t);
        [(nx + 1.0) * 0.5 * sw as f32, (1.0 - ny) * 0.5 * sh_ as f32]
    };

    let decode = |s: f32| if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) };
    let encode = |l: f32| {
        let l = l.clamp(0.0, 1.0);
        if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
    };
    // `pbr_neutral` from raspberry_common.wgsl.
    let neutral = |c: [f32; 3]| -> [f32; 3] {
        let start = 0.8 - 0.04;
        let x = c[0].min(c[1]).min(c[2]);
        let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
        let c = c.map(|v| v - offset);
        let peak = c[0].max(c[1]).max(c[2]);
        if peak < start {
            return c;
        }
        let d = 1.0 - start;
        let new_peak = 1.0 - d * d / (peak + d - start);
        let c = c.map(|v| v * new_peak / peak.max(1e-6));
        let g = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0);
        c.map(|v| v + (new_peak - v) * g)
    };

    // The supersampled frame, held in whichever space blending happens in.
    let mut dst: Vec<[f32; 3]> = (0..sw * sh_)
        .map(|i| {
            let (x, y) = (i % sw / ss as usize, i / sw / ss as usize);
            let k = (y * w as usize + x) * 4;
            let e = [bare[k], bare[k + 1], bare[k + 2]].map(|b| b as f32 / 255.0);
            if linear_blend { e.map(decode) } else { e }
        })
        .collect();

    for tri in index.chunks_exact(3) {
        let v = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p = v.map(project);
        let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
            - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let x1 = (p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil() as usize).min(sw);
        let y0 = p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let y1 = (p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil() as usize).min(sh_);
        for y in y0..y1 {
            for x in x0..x1 {
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                let edge = |a: [f32; 2], b: [f32; 2]| {
                    (b[0] - a[0]) * (cy - a[1]) - (cx - a[0]) * (b[1] - a[1])
                };
                let wts = [p[1], p[2], p[0]]
                    .iter()
                    .zip([p[2], p[0], p[1]])
                    .map(|(&a, b)| edge(a, b) / area)
                    .collect::<Vec<_>>();
                if wts.iter().any(|&b| b < 0.0) {
                    continue;
                }
                let lerp = |arr: &[f32], stride: usize, c: usize| {
                    (0..3).map(|j| wts[j] * arr[v[j] * stride + c]).sum::<f32>()
                };
                let across = lerp(uv, 2, 0);
                let alpha = lerp(col, 4, 3) * (1.0 - across * across * 0.6);
                if alpha < 0.02 {
                    continue;
                }
                let hdr = [0, 1, 2].map(|c| lerp(col, 4, c) * args.exposure);
                let lin = neutral(hdr).map(|c| c.clamp(0.0, 1.0));
                let src = if linear_blend { lin } else { lin.map(encode) };
                let d = &mut dst[y * sw + x];
                for c in 0..3 {
                    d[c] = src[c] * alpha + d[c] * (1.0 - alpha);
                }
            }
        }
    }

    // Resolve: average each block in linear light, encode, quantise.
    let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let mut count = 0;
    let mut lift = 0.0f32;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let k = y * w as usize + x;
            if body[k] {
                continue;
            }
            let mut acc = [0.0f32; 3];
            for sy in 0..ss as usize {
                for sx in 0..ss as usize {
                    let d = dst[(y * ss as usize + sy) * sw + x * ss as usize + sx];
                    let l = if linear_blend { d } else { d.map(decode) };
                    for c in 0..3 {
                        acc[c] += l[c] / (ss * ss) as f32;
                    }
                }
            }
            let out = acc.map(|l| (encode(l) * 255.0).round());
            let base = [bare[k * 4], bare[k * 4 + 1], bare[k * 4 + 2]].map(|b| b as f32);
            if luma(out) > luma(base) + 4.0 {
                count += 1;
            }
            lift += (luma(out) - luma(base)).max(0.0);
        }
    }
    (count, lift)
}
