//! Every post-fx pass keeps the picture the right way up, natively.
//!
//! `POSTFX_SHADER` (the effects behind `applyPostFx` / `applyPostFxToRT` and the
//! three.js shim's passes) used to hand its fragment stage a uv with v = 0 at
//! the *bottom* of the target, GL's convention, while every wgpu texture has
//! v = 0 at its top row. Each pass was therefore a vertical flip unless a
//! per-call `resolution.w` flag asked for the compensating one, and which calls
//! asked was decided case by case — never into a render target. Chains of an
//! even number of passes came out right by accident; the rest did not, and the
//! screen-space effects reconstructed positions against the mirrored image.
//!
//! The scenes here are asymmetric on purpose: a marker in the top-left quadrant
//! only, so a vertical flip moves it to the bottom-left and a horizontal one to
//! the top-right. Every kind is checked into a render target, onto a
//! "canvas" (output id 0 — the path the browser takes for the swapchain), and
//! between half-float targets, which the shim's composer uses. SSAO and SSR are
//! checked against geometry: the occlusion has to land in the crease that is
//! really there, and the reflection below the object it reflects.
//!
//! The same orientation questions are asked in the browser, on WebGL2 and
//! WebGPU, by `tests/web_postfx`.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use threers::renderer::{readback, PostFxCamera};
use threers::{
    BasicMaterial, BoxGeometry, Camera, Color, HeadlessRenderer, Material, Mesh,
    NormalMaterial, Object3D, OrthographicCamera, PerspectiveCamera, PlaneGeometry, RenderTarget,
    Scene, Vector3,
};

const W: u32 = 64;
const H: u32 = 64;
const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;
const F16: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba16Float;

fn headless() -> Option<HeadlessRenderer> {
    match HeadlessRenderer::builder()
        .size(W, H)
        .color_format(FORMAT)
        .build()
    {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

/// A fresh target registered under `id`.
fn target(r: &mut HeadlessRenderer, id: u32, format: threers::wgpu::TextureFormat) -> Arc<RenderTarget> {
    let rt = Arc::new(RenderTarget::new(r.device(), W, H, format));
    r.renderer().register_render_target(id, &rt);
    rt
}

/// The target's pixels as RGBA in `[0, 1]`, top row first.
fn read(r: &HeadlessRenderer, rt: &RenderTarget) -> Vec<f32> {
    let bytes = readback::texture_bytes(r.device(), r.queue(), &rt.color_texture);
    if rt.format == F16 {
        // Clamped as a display would: dot screen, for one, writes far
        // outside [0, 1] into a float target.
        readback::as_f32(&bytes).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
    } else {
        bytes.iter().map(|&b| b as f32 / 255.0).collect()
    }
}

/// Mean of r+g+b over each quadrant: `[top-left, top-right, bottom-left, bottom-right]`.
fn quadrants(px: &[f32], w: u32, h: u32) -> [f32; 4] {
    let mut sum = [0f32; 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let q = (if y < h / 2 { 0 } else { 2 }) + (if x < w / 2 { 0 } else { 1 });
            sum[q] += px[i] + px[i + 1] + px[i + 2];
        }
    }
    let n = (w * h / 4) as f32;
    sum.map(|s| s / n)
}

/// `"top-left"` when that quadrant clearly carries the most, else the four
/// means — the answer a flip would change.
fn marker_quadrant(px: &[f32], w: u32, h: u32) -> String {
    let q = quadrants(px, w, h);
    // Measured above the darkest quadrant, so an effect that lifts the whole
    // frame (afterimage's blend with a grey fallback) cannot hide the marker.
    let floor = q.iter().cloned().fold(f32::INFINITY, f32::min);
    let e = q.map(|v| v - floor);
    let others = e[1].max(e[2]).max(e[3]);
    if e[0] > 0.05 && e[0] > others * 1.5 {
        "top-left".into()
    } else {
        format!("unclear: TL {:.3} TR {:.3} BL {:.3} BR {:.3}", q[0], q[1], q[2], q[3])
    }
}

/// White square over x ∈ [-0.75, -0.25], y ∈ [0.25, 0.75] of `front_ortho`'s
/// view: inside the top-left quadrant, away from both centre lines so edge
/// filters keep it there too.
fn marker_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut quad = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.5, 0.5),
        BasicMaterial::new(Color::WHITE).into(),
    ));
    quad.position = Vector3::new(-0.5, 0.5, 0.0);
    scene.add(quad);
    scene.update_world();
    scene
}

fn front_ortho() -> OrthographicCamera {
    let mut c = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

/// `(name, kind, time, params2, params3)` for every kind that has a picture
/// to keep. Omitted: 11, 12 and 26 (SSAO, SSR, bokeh — they need geometry,
/// see their own tests), 23 (LUT: a pointwise colour map read from a second
/// texture, so with none bound its output is a constant), 29 (solid clear: no
/// input at all) and 34 (adapt luminance: a 1×1 average in, one value out).
/// Name, kind, `p0`, `p1`, `p2` for one post-fx case.
type Case = (&'static str, u32, f32, [f32; 4], [f32; 4]);

fn kinds() -> Vec<Case> {
    let pi = std::f32::consts::PI;
    let (w, h) = (W as f32, H as f32);
    // Glitch with its bands moved off-screen and no displacement: a small RGB
    // split plus a little snow, both everywhere, so the marker still dominates.
    let glitch2 = [0.004, 0.3, 2.0, 2.0];
    let glitch3 = [0.0, 0.0, 0.05, 0.0];
    vec![
        ("0 copy", 0, 0.0, [0.0; 4], [0.0; 4]),
        ("0 copy, sRGB-decode (p2.y)", 0, 0.0, [0.0, 1.0, 0.0, 0.0], [0.0; 4]),
        ("1 FXAA", 1, 0.0, [1.0 / w, 1.0 / h, 0.0, 0.0], [0.0; 4]),
        ("2 film", 2, 0.25, [0.5, 0.0, 0.0, 0.0], [0.0; 4]),
        ("3 dot screen", 3, 0.0, [1.57, 1.0, 0.5, 0.5], [0.0; 4]),
        ("4 halftone", 4, 0.0, [4.0, 0.0, 1.0, 1.0], [pi / 12.0, pi / 6.0, pi / 4.0, 1.0]),
        ("5 glitch", 5, 0.5, glitch2, glitch3),
        ("31 glitch, no snow", 31, 0.5, glitch2, glitch3),
        ("6 bloom threshold", 6, 0.0, [0.5, 0.1, 0.0, 0.0], [0.0; 4]),
        ("7 blur h", 7, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("8 blur v", 8, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("9 bloom composite", 9, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("10 sobel", 10, 0.0, [1.0, 1.0, 1.0, 1.0], [0.0; 4]),
        ("13 SSAO blur", 13, 0.0, [0.0; 4], [0.0; 4]),
        ("14 SSAO composite", 14, 0.0, [0.0; 4], [0.0; 4]),
        ("15 SSAO multiply", 15, 0.0, [0.0; 4], [0.0; 4]),
        ("16 outline edge", 16, 0.0, [1.0, 1.0, 1.0, 1.0], [0.0; 4]),
        ("17 outline overlay", 17, 0.0, [3.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("18 outline blur h", 18, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("19 outline blur v", 19, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("20 bilinear resample", 20, 0.0, [0.0; 4], [0.0; 4]),
        ("21 afterimage", 21, 0.0, [0.95, 0.0, 0.0, 0.0], [0.0; 4]),
        ("22 transition", 22, 0.0, [0.0; 4], [0.0; 4]),
        ("24 pixelated", 24, 0.0, [4.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("25 adaptive tone map", 25, 0.0, [0.6, 0.01, 16.0, 1.0], [0.0; 4]),
        ("27 SAO", 27, 0.0, [0.5, 0.0, 0.0, 0.0], [0.0; 4]),
        ("28 SMAA", 28, 0.0, [0.0; 4], [0.0; 4]),
        ("30 output", 30, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("33 luminosity", 33, 0.0, [0.0; 4], [0.0; 4]),
    ]
}

/// Every post-fx kind keeps the marker in the top-left: into an 8-bit target,
/// onto the canvas path, and between half-float targets.
#[test]
fn every_post_fx_kind_keeps_the_marker_top_left() {
    let Some(mut r) = headless() else { return };
    let camera = front_ortho();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for (input_format, output_format, label) in [
        (FORMAT, FORMAT, "rgba8 target"),
        (F16, F16, "f16 target"),
    ] {
        let input = target(&mut r, 1, input_format);
        r.renderer().render_to(&mut marker_scene(), &camera, &input);
        let rt_in = read(&r, &input);
        assert_eq!(
            marker_quadrant(&rt_in, W, H),
            "top-left",
            "{label}: the scene render itself is not top-left"
        );
        let output = target(&mut r, 2, output_format);
        // An unregistered target standing in for the swapchain: output id 0
        // is what the browser passes for the canvas.
        let canvas = RenderTarget::new(r.device(), W, H, FORMAT);
        for (name, kind, time, p2, p3) in kinds() {
            for to_canvas in [false, true] {
                // The canvas is 8-bit, and the multiply blend only has an
                // 8-bit pipeline; the shim never asks for either with f16.
                if (to_canvas || kind == 15) && input_format == F16 {
                    continue;
                }
                let (out_id, out_rt) = if to_canvas { (0, &canvas) } else { (2, &*output) };
                // Kind 15 multiplies onto what is there, so give it white.
                if kind == 15 {
                    r.renderer().apply_postfx_by_id(
                        1, out_id, &out_rt.color_view, 0, 0, 29, 0.0,
                        [1.0; 4], [0.0; 4], W, H, false, PostFxCamera::default(),
                    );
                }
                r.renderer().apply_postfx_by_id(
                    1, out_id, &out_rt.color_view, 0, 0, kind, time, p2, p3, W, H, false,
                    PostFxCamera::default(),
                );
                let px = read(&r, out_rt);
                let at = marker_quadrant(&px, W, H);
                checked += 1;
                if at != "top-left" {
                    let dest = if to_canvas { "canvas" } else { label };
                    wrong.push(format!("{name} → {dest}: {at}"));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {checked} post-fx passes moved the marker:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// The three.js SSAO kernel shape: points in the +z hemisphere, denser near
/// the origin. A fixed LCG so the test is deterministic.
fn ssao_kernel() -> Vec<f32> {
    let mut s: u32 = 0x5a00426;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s as f32 / 4294967296.0
    };
    let mut k = Vec::new();
    for i in 0..32 {
        let mut v = Vector3::new(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd());
        v = v.normalize();
        let t = i as f32 / 32.0;
        let scale = 0.1 + (1.0 - 0.1) * t * t;
        k.extend_from_slice(&[v.x * scale, v.y * scale, v.z * scale]);
    }
    k
}

/// A floor, and a back wall standing on it over the left half only, seen
/// from above and in front so the crease where they meet is in the upper
/// half of the frame. Red wall, blue floor, so the crease row can be found
/// in the colour render rather than assumed.
fn crease_scene(material: impl Fn(Color) -> Material) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut floor = Object3D::mesh(Mesh::new(
        BoxGeometry::new(8.0, 0.1, 8.0),
        material(Color::new(0.0, 0.0, 1.0)),
    ));
    floor.position = Vector3::new(0.0, -0.05, 0.0);
    scene.add(floor);
    let mut wall = Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 3.0, 0.1),
        material(Color::new(1.0, 0.0, 0.0)),
    ));
    wall.position = Vector3::new(-2.0, 1.5, -1.05);
    scene.add(wall);
    scene.update_world();
    scene
}

fn crease_camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 20.0);
    c.position = Vector3::new(0.0, 1.5, 3.0);
    c.look_at(Vector3::new(0.0, 0.0, 0.5));
    c
}

/// The first row, scanning down column `x`, where red (wall) gives way to
/// blue (floor).
fn crease_row(px: &[f32], x: u32) -> Option<u32> {
    let red = |y: u32| {
        let i = ((y * W + x) * 4) as usize;
        px[i] > 0.3 && px[i + 2] < 0.2
    };
    let blue = |y: u32| {
        let i = ((y * W + x) * 4) as usize;
        px[i + 2] > 0.3 && px[i] < 0.2
    };
    (1..H).find(|&y| red(y - 1) && blue(y))
}

/// Mean ambient-occlusion factor (output `.r`, 1 = open) over a block.
fn ao_mean(px: &[f32], xs: std::ops::Range<u32>, ys: std::ops::Range<u32>) -> f32 {
    let mut s = 0.0;
    let mut n = 0.0;
    for y in ys {
        for x in xs.clone() {
            s += px[((y * W + x) * 4) as usize];
            n += 1.0;
        }
    }
    s / n
}

/// SSAO (kind 11) darkens the crease that is really there — upper half, left
/// half — and leaves the open floor alone, into a target and onto the canvas.
/// A uv flip anywhere between the depth fetch and the position reconstruction
/// moves the darkening to the mirrored rows, or smears it over flat floor.
#[test]
fn ssao_darkens_the_crease_where_it_is() {
    let Some(mut r) = headless() else { return };
    let camera = crease_camera();
    let colour = target(&mut r, 1, FORMAT);
    let normals = target(&mut r, 2, FORMAT);
    let ao = target(&mut r, 3, FORMAT);
    let canvas = RenderTarget::new(r.device(), W, H, FORMAT);

    r.renderer()
        .render_to(&mut crease_scene(|c| BasicMaterial::new(c).into()), &camera, &colour);
    let colour_px = read(&r, &colour);
    let crease = crease_row(&colour_px, W / 4).expect("no wall/floor crease in the left half");
    assert!(
        crease < H / 2 - 4,
        "the crease is at row {crease}; the scene is meant to put it in the upper half"
    );
    assert!(
        crease_row(&colour_px, W * 3 / 4).is_none(),
        "the wall is meant to stand over the left half only"
    );

    // Normal prepass; its depth attachment is what kind 11 reads when the
    // depth and normal ids are the same.
    let mut normal_scene = crease_scene(|_| NormalMaterial::new().into());
    normal_scene.background = Color::from_hex(0x7777ff);
    r.renderer().render_to(&mut normal_scene, &camera, &normals);
    r.renderer().set_ssao_kernel(&ssao_kernel());

    let (near, far) = camera.near_far();
    let proj = camera.projection_matrix();
    let cam = PostFxCamera {
        near,
        far,
        kernel_radius: 0.6,
        kernel_size: 32,
        proj: proj.elements,
        inv_proj: proj.invert().elements,
    };
    let mut failures = Vec::new();
    for (dest, out_id, out_rt) in [("target", 3u32, &*ao), ("canvas", 0, &canvas)] {
        r.renderer().apply_postfx_by_id(
            1, out_id, &out_rt.color_view, 2, 2, 11, 0.0,
            [0.0005, 0.1, 0.0, 0.0], [0.0; 4], W, H, false, cam.clone(),
        );
        let px = read(&r, out_rt);
        let left = 2..W / 2 - 4;
        let right = W * 5 / 8..W - 2;
        let band = crease.saturating_sub(2)..crease + 3;
        let mirrored = (H - 1 - crease).saturating_sub(2)..(H - 1 - crease) + 3;
        let in_crease = ao_mean(&px, left.clone(), band.clone());
        let at_mirror = ao_mean(&px, left.clone(), mirrored);
        let right_of_crease = ao_mean(&px, right, band);
        let open_floor = ao_mean(&px, left, H - 8..H - 2);
        let report = format!(
            "{dest}: crease row {crease}: AO {in_crease:.3} in the crease, {at_mirror:.3} at the \
             mirrored rows, {right_of_crease:.3} on the same rows right of the wall, \
             {open_floor:.3} on open floor"
        );
        eprintln!("{report}");
        if !(in_crease < 0.85
            && at_mirror > in_crease + 0.1
            && right_of_crease > in_crease + 0.1
            && open_floor > 0.9)
        {
            failures.push(report);
        }
    }
    assert!(failures.is_empty(), "SSAO misplaced:\n  {}", failures.join("\n  "));
}

/// SSR (kind 12) puts a reflection below the object it reflects, in the same
/// half, and leaves the object where it was.
///
/// The colour is a red block over texture rows v ∈ [0.1, 0.4] of the left
/// half; the depth marks only its top strip, v ∈ [0.1, 0.15], as near. The
/// pass mirrors about the horizontal midline and searches at most twelve
/// pixels (its minimum step), so the reflection of that strip lands at
/// v ∈ [0.66, 0.89] and nowhere else. A flipped frame puts the block at
/// [0.6, 0.9] and the reflection at [0.11, 0.34], so the bottom of the block
/// (v = 0.38) and the gap above the reflection (v = 0.62) tell the two apart.
#[test]
fn ssr_reflects_below_the_object() {
    let Some(mut r) = headless() else { return };
    let camera = front_ortho();
    let colour = target(&mut r, 1, FORMAT);
    let depth = target(&mut r, 2, FORMAT);
    let out = target(&mut r, 3, FORMAT);
    let canvas = RenderTarget::new(r.device(), W, H, FORMAT);

    // A block over world y ∈ [y - height/2, y + height/2], x ∈ [-0.8, -0.2].
    let block = |c: Color, height: f32, y: f32, bg: Color| {
        let mut s = Scene::new();
        s.background = bg;
        let mut q = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(0.6, height),
            BasicMaterial::new(c).into(),
        ));
        q.position = Vector3::new(-0.5, y, 0.0);
        s.add(q);
        s.update_world();
        s
    };
    let grey = Color::new(0.3, 0.3, 0.3);
    // World y ∈ [0.2, 0.8] is texture v ∈ [0.1, 0.4].
    r.renderer()
        .render_to(&mut block(Color::new(1.0, 0.0, 0.0), 0.6, 0.5, grey), &camera, &colour);
    // Near (white) over world y ∈ [0.7, 0.8], texture v ∈ [0.1, 0.15].
    r.renderer()
        .render_to(&mut block(Color::WHITE, 0.1, 0.75, Color::BLACK), &camera, &depth);

    let red_at = |px: &[f32], fx: f32, fy: f32| {
        let (x, y) = ((fx * W as f32) as u32, (fy * H as f32) as u32);
        let i = ((y * W + x) * 4) as usize;
        px[i] - px[i + 1] > 0.4
    };
    let mut failures = Vec::new();
    for (dest, out_id, out_rt) in [("target", 3u32, &*out), ("canvas", 0, &canvas)] {
        r.renderer().apply_postfx_by_id(
            1, out_id, &out_rt.color_view, 2, 0, 12, 0.0,
            [0.5, 0.1, 1.0, 1.0], [0.0; 4], W, H, false, PostFxCamera::default(),
        );
        let px = read(&r, out_rt);
        let checks = [
            ("the block's top", 0.25, 0.12, true),
            ("the block's bottom", 0.25, 0.38, true),
            ("open background between", 0.25, 0.5, false),
            ("the gap above the reflection", 0.25, 0.62, false),
            ("the reflection", 0.25, 0.87, true),
            ("nothing right of the reflection", 0.75, 0.87, false),
            ("nothing right of the block", 0.75, 0.25, false),
        ];
        for (what, fx, fy, want) in checks {
            if red_at(&px, fx, fy) != want {
                failures.push(format!(
                    "{dest}: {what} at ({fx}, {fy}) is {}red",
                    if want { "not " } else { "" }
                ));
            }
        }
    }
    assert!(failures.is_empty(), "SSR misplaced:\n  {}", failures.join("\n  "));
}

/// The native `EffectComposer` keeps the marker top-left however many passes
/// it runs. Its vertex stage used GL's bottom-left uv, so each shader pass —
/// and the final copy — was a flip: an odd total came out upside down.
#[test]
fn native_composer_chains_keep_the_marker_top_left() {
    use threers::postprocessing::*;
    let Some(mut r) = headless() else { return };
    let camera = front_ortho();
    type Chain = (&'static str, Vec<Box<dyn Pass>>);
    let copy = || Box::new(ShaderPass::copy()) as Box<dyn Pass>;
    let fxaa = || Box::new(ShaderPass::fxaa(W, H)) as Box<dyn Pass>;
    let chains: Vec<Chain> = vec![
        ("render only", vec![]),
        ("copy", vec![copy()]),
        ("copy, copy", vec![copy(), copy()]),
        ("fxaa", vec![fxaa()]),
        ("unreal bloom, fxaa", vec![Box::new(UnrealBloomPass::new(W, H, 1.0, 0.4, 0.5)), fxaa()]),
        ("bloom, copy", vec![Box::new(BloomPass::default()), copy()]),
        ("film", vec![Box::new(FilmPass::default())]),
        ("glitch", vec![Box::new(GlitchPass::default())]),
        ("output", vec![Box::new(OutputPass::new())]),
        ("outline", vec![Box::new(OutlinePass::new(W, H, 0, 0, Vec::new()))]),
        ("ssao", vec![Box::new(SsaoPass::new(0, 0, W, H, 32))]),
        ("ssao, copy", vec![Box::new(SsaoPass::new(0, 0, W, H, 32)), copy()]),
        ("afterimage", vec![Box::new(AfterimagePass::default())]),
        ("smaa", vec![Box::new(SmaaPass::new(W, H))]),
    ];
    let mut wrong = Vec::new();
    for (name, passes) in chains {
        let mut composer = EffectComposer::new(r.device().clone(), r.queue().clone(), W, H, FORMAT);
        composer.add_pass(RenderPass::new());
        for p in passes {
            composer.add_boxed(p);
        }
        let out = RenderTarget::new(r.device(), W, H, FORMAT);
        composer.render(r.renderer(), &mut marker_scene(), &camera, &out.color_view);
        let px = read(&r, &out);
        let at = marker_quadrant(&px, W, H);
        if at != "top-left" {
            wrong.push(format!("{name}: {at}"));
        }
    }
    assert!(wrong.is_empty(), "composer chains moved the marker:\n  {}", wrong.join("\n  "));
}

/// The headless renderer's own bloom and SSAO run on the shared fullscreen
/// stage; they keep the marker where it is.
#[test]
fn headless_bloom_and_ssao_keep_the_marker_top_left() {
    let Some(mut r) = headless() else { return };
    let camera = front_ortho();
    r.set_bloom(1.0, 0.5, 0.5);
    r.set_ssao(1.0, 0.5);
    let px: Vec<f32> = r
        .render_to_rgba(&mut marker_scene(), &camera)
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();
    let (w, h) = r.render_size();
    assert_eq!(marker_quadrant(&px, w, h), "top-left");
}
