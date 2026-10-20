//! What the post-fx kinds behind the three.js shim's passes compute, natively.
//!
//! `tests/postfx_orientation.rs` asks whether each kind keeps the picture the
//! right way up; this asks whether a kind is the effect it is named for. Kinds
//! 25 and 26 — the shim's `AdaptiveToneMappingPass` and `BokehPass` — were
//! also two of the glitch variants, and the glitch branch came first in the
//! shader, so both passes ran glitch. The glitch variants are 31 and 32 now.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use threers::renderer::{readback, PostFxCamera};
use threers::{
    BasicMaterial, Camera, Color, HeadlessRenderer, Mesh, Object3D, OrthographicCamera,
    PerspectiveCamera, PlaneGeometry, RenderTarget, Scene, Vector3,
};

const W: u32 = 64;
const H: u32 = 64;
const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;
const F16: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba16Float;

fn headless() -> Option<HeadlessRenderer> {
    match HeadlessRenderer::builder().size(W, H).color_format(FORMAT).build() {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

/// A fresh `w`×`h` target registered under `id`.
fn target(
    r: &mut HeadlessRenderer,
    id: u32,
    format: threers::wgpu::TextureFormat,
    w: u32,
    h: u32,
) -> Arc<RenderTarget> {
    let rt = Arc::new(RenderTarget::new(r.device(), w, h, format));
    r.renderer().register_render_target(id, &rt);
    rt
}

/// The target's pixels as RGBA floats, top row first, unclamped.
fn read(r: &HeadlessRenderer, rt: &RenderTarget) -> Vec<f32> {
    let bytes = readback::texture_bytes(r.device(), r.queue(), &rt.color_texture);
    if rt.format == F16 {
        readback::as_f32(&bytes)
    } else {
        bytes.iter().map(|&b| b as f32 / 255.0).collect()
    }
}

/// One post-fx pass from target `input` into target `output`. `second` is
/// the (depth, normal) pair; a pass that reads only `normal_tex` gets depth 0,
/// since naming one target twice binds it as depth alone unless the pass reads
/// hardware depth.
#[allow(clippy::too_many_arguments)]
fn pass(
    r: &mut HeadlessRenderer,
    input: u32,
    output: (u32, &RenderTarget),
    second: (u32, u32),
    kind: u32,
    time: f32,
    p2: [f32; 4],
    p3: [f32; 4],
    camera: PostFxCamera,
) {
    let (out_id, out_rt) = output;
    r.renderer().apply_postfx_by_id(
        input, out_id, &out_rt.color_view, second.0, second.1, kind, time, p2, p3,
        out_rt.width, out_rt.height, false, camera,
    );
}

/// Fill target `id` with one colour (kind 29), reading from `scratch`.
fn fill(r: &mut HeadlessRenderer, scratch: u32, id: u32, rt: &RenderTarget, c: [f32; 4]) {
    pass(r, scratch, (id, rt), (0, 0), 29, 0.0, c, [0.0; 4], PostFxCamera::default());
}

/// three.js `luminance()`.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126729 * c[0] + 0.7151522 * c[1] + 0.0721750 * c[2]
}

/// three.js r165 ToneMapShader's operator on a grey of value `v`.
fn tone_map(v: f32, average: f32, middle_grey: f32, min_lum: f32, max_lum: f32) -> f32 {
    let lum = luminance([v; 3]);
    let scaled = lum * middle_grey / min_lum.max(average);
    let compressed = scaled * (1.0 + scaled / (max_lum * max_lum)) / (1.0 + scaled);
    compressed * v
}

/// Kind 25 is three.js's adaptive tone mapping: with a fixed average it is
/// ToneMapShader's operator, and the adaptive chain the shim builds — kind 33
/// (luminosity), kind 20 halvings down to 1×1, kind 34 (adaption) — feeds it
/// the frame's own luminance, squared and adapted, so a bright frame comes out
/// far darker. Kind 25 used to run glitch, which passes a flat frame through.
#[test]
fn adaptive_tone_mapping_maps_a_bright_frame_down() {
    let Some(mut r) = headless() else { return };
    let (grey, middle, min_lum, max_lum) = (4.0f32, 0.6f32, 0.01f32, 16.0f32);
    let scratch = target(&mut r, 99, FORMAT, W, H);
    let _ = scratch;
    let frame = target(&mut r, 1, F16, W, H);
    fill(&mut r, 99, 1, &frame, [grey, grey, grey, 1.0]);
    let out = target(&mut r, 2, F16, W, H);
    let at_centre = |px: &[f32], w: u32| px[(((H / 2) * w + w / 2) * 4) as usize];

    // Fixed average luminance 1.
    pass(
        &mut r, 1, (2, &out), (0, 0), 25, 0.0, [middle, min_lum, max_lum, 1.0], [0.0; 4],
        PostFxCamera::default(),
    );
    let fixed = at_centre(&read(&r, &out), W);
    let want = tone_map(grey, 1.0, middle, min_lum, max_lum);
    assert!(
        (fixed - want).abs() < 0.01 * want,
        "kind 25, average 1: {grey} → {fixed}, three.js gives {want}"
    );

    // Adaptive: the luminance, averaged down to one texel.
    let lum = target(&mut r, 3, F16, W, H);
    pass(&mut r, 1, (3, &lum), (0, 0), 33, 0.0, [0.0; 4], [0.0; 4], PostFxCamera::default());
    let mut src = 3;
    let mut size = W;
    let mut id = 10;
    let mut halvings = Vec::new();
    while size > 1 {
        size /= 2;
        let rt = target(&mut r, id, F16, size, size);
        pass(&mut r, src, (id, &rt), (0, 0), 20, 0.0, [0.0; 4], [0.0; 4], PostFxCamera::default());
        halvings.push(rt);
        src = id;
        id += 1;
    }
    let average = read(&r, halvings.last().unwrap())[0];
    assert!(
        (average - luminance([grey; 3])).abs() < 0.01,
        "average luminance of a flat {grey} frame: {average}"
    );

    // Adaption from three.js's initial 0x777777 towards this frame.
    let initial = ((0x77 as f32 / 255.0 + 0.055) / 1.055).powf(2.4);
    let previous = target(&mut r, 4, F16, 1, 1);
    let adapted = target(&mut r, 5, F16, 1, 1);
    for (delta, tau) in [(0.1f32, 1.0f32), (100.0, 1.0)] {
        fill(&mut r, 99, 4, &previous, [initial, initial, initial, 1.0]);
        pass(
            &mut r, src, (5, &adapted), (0, 4), 34, 0.0, [min_lum, delta, tau, 0.0], [0.0; 4],
            PostFxCamera::default(),
        );
        let got = read(&r, &adapted)[0];
        let current = luminance([grey; 3]).max(min_lum).powi(2);
        let last = initial.max(min_lum);
        let want = last + (current - last) * (1.0 - (-delta * tau).exp());
        assert!(
            (got - want).abs() < 0.01 * want,
            "kind 34, delta {delta}: adapted {got}, three.js gives {want}"
        );
    }

    // Tone mapped against the converged adaption (16, the luminance squared).
    pass(
        &mut r, 1, (2, &out), (0, 5), 25, 0.0, [middle, min_lum, max_lum, 1.0],
        [1.0, 0.0, 0.0, 0.0], PostFxCamera::default(),
    );
    let mapped = at_centre(&read(&r, &out), W);
    let want = tone_map(grey, luminance([grey; 3]).powi(2), middle, min_lum, max_lum);
    assert!(
        (mapped - want).abs() < 0.01 * want.max(0.1),
        "kind 25, adaptive: {grey} → {mapped}, three.js gives {want}"
    );
    assert!(mapped < 0.25 * grey, "a bright frame is meant to come down: {grey} → {mapped}");
}

/// Vertical stripes two pixels wide, white on black, over the whole frame.
fn stripes() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let px = 2.0 / W as f32;
    for i in 0..W / 4 {
        let mut q = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0 * px, 2.0),
            BasicMaterial::new(Color::WHITE).into(),
        ));
        q.position = Vector3::new(-1.0 + (4 * i + 1) as f32 * px, 0.0, 0.0);
        scene.add(q);
    }
    scene.update_world();
    scene
}

fn front_ortho() -> OrthographicCamera {
    let mut c = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

/// Mean difference between horizontally adjacent pixels (red) over columns
/// `xs` and the middle rows: how much of the stripes survives.
fn contrast(px: &[f32], xs: std::ops::Range<u32>) -> f32 {
    let (mut s, mut n) = (0.0, 0.0);
    for y in H / 4..H * 3 / 4 {
        for x in xs.clone() {
            let i = ((y * W + x) * 4) as usize;
            s += (px[i] - px[i + 4]).abs();
            n += 1.0;
        }
    }
    s / n
}

/// Kind 26 is three.js's BokehShader: a wall at the focus distance over the
/// left half stays sharp, one far behind it over the right half is blurred,
/// with the depth read from the prepass target's hardware depth. Kind 26 used
/// to run glitch, which shifts channels rather than blurring by depth.
#[test]
fn bokeh_blurs_what_is_out_of_focus_and_keeps_the_focus_sharp() {
    let Some(mut r) = headless() else { return };
    let colour = target(&mut r, 1, FORMAT, W, H);
    let depth = target(&mut r, 2, FORMAT, W, H);
    let out = target(&mut r, 3, FORMAT, W, H);
    r.renderer().render_to(&mut stripes(), &front_ortho(), &colour);

    // Looking down -z: a wall at distance 2 left of x = 0, one at 10 right of it.
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 20.0);
    camera.position = Vector3::ZERO;
    camera.look_at(Vector3::new(0.0, 0.0, -1.0));
    let mut walls = Scene::new();
    walls.background = Color::BLACK;
    for (distance, half_width) in [(2.0f32, 3.0f32), (10.0, 15.0)] {
        let mut q = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(half_width, 2.0 * half_width),
            BasicMaterial::new(Color::WHITE).into(),
        ));
        let side = if distance < 5.0 { -0.5 } else { 0.5 };
        q.position = Vector3::new(side * half_width, 0.0, -distance);
        walls.add(q);
    }
    walls.update_world();
    r.renderer().render_to(&mut walls, &camera, &depth);

    let (near, far) = camera.near_far();
    let proj = camera.projection_matrix();
    let cam = PostFxCamera {
        near,
        far,
        kernel_radius: 0.0,
        kernel_size: 0,
        proj: proj.elements,
        inv_proj: proj.invert().elements,
    };
    // focus 2, aperture 0.025, maxblur 0.1, aspect 1.
    pass(&mut r, 1, (3, &out), (2, 2), 26, 0.0, [2.0, 0.025, 0.1, 1.0], [0.0; 4], cam);
    let before = read(&r, &colour);
    let after = read(&r, &out);
    let (left, right) = (4..W / 2 - 8, W / 2 + 8..W - 5);
    let report = format!(
        "stripe contrast: in focus {:.3} → {:.3}, out of focus {:.3} → {:.3}",
        contrast(&before, left.clone()),
        contrast(&after, left.clone()),
        contrast(&before, right.clone()),
        contrast(&after, right.clone()),
    );
    assert!(contrast(&before, left.clone()) > 0.3, "no stripes to blur: {report}");
    assert!(
        contrast(&after, left.clone()) > 0.95 * contrast(&before, left),
        "the wall in focus was blurred: {report}"
    );
    assert!(
        contrast(&after, right.clone()) < 0.3 * contrast(&before, right),
        "the far wall was not blurred: {report}"
    );
}

/// White square in the top-left quadrant on black.
fn marker() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut q = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.5, 0.5),
        BasicMaterial::new(Color::WHITE).into(),
    ));
    q.position = Vector3::new(-0.5, 0.5, 0.0);
    scene.add(q);
    scene.update_world();
    scene
}

/// Kind 31 — the glitch without snow, renumbered from 25 — still glitches: the red channel is read `amount`
/// to the right (angle 0), so red shows left of the square where green does
/// not. Kind 5 does the same plus snow.
#[test]
fn the_renumbered_glitch_kinds_still_split_the_channels() {
    let Some(mut r) = headless() else { return };
    let input = target(&mut r, 1, FORMAT, W, H);
    let out = target(&mut r, 2, F16, W, H);
    r.renderer().render_to(&mut marker(), &front_ortho(), &input);
    // Bands off-screen, no displacement, a 0.05 (3.2 px) split.
    let p2 = [0.05, 0.0, 2.0, 2.0];
    let p3 = [0.0, 0.0, 0.05, 0.0];
    {
        let kind = 31u32;
        pass(&mut r, 1, (2, &out), (0, 0), kind, 0.5, p2, p3, PostFxCamera::default());
        let px = read(&r, &out);
        // The square spans x, y ∈ [8, 24): two pixels left of it.
        let split = (12..20)
            .filter(|&y| {
                let i = ((y * W + 6) * 4) as usize;
                px[i] > 0.5 && px[i + 1] < 0.2
            })
            .count();
        assert!(split >= 6, "kind {kind}: no red left of the square ({split} rows of 8)");
    }
}

/// three.js DigitalGlitch's `rand`, in f32 as the GPU does it.
#[allow(clippy::excessive_precision)] // three.js's constants, digit for digit
fn glitch_rand(x: f32, y: f32) -> f32 {
    let d = x * 12.9898 + y * 78.233;
    (d.sin() * 43758.5453).rem_euclid(1.0)
}

/// Kind 5's snow is three.js's: `rand(vec2(xs * seed, ys * seed * 50.))` with
/// `xs`, `ys` the floors of `gl_FragCoord / 0.5` — pixel centres counted from
/// the bottom-left. Over black with the split and the bands out of the way the
/// output is the snow alone, `200 * amount * rand * 0.2`, so it can be checked
/// pixel by pixel against that formula for a fixed seed. The seed is tiny so
/// the hash's argument stays under 1, where f32 `sin` on the CPU and on the GPU
/// agree to a few parts in a hundred after the ×43758 — at the usual seeds the
/// hash is chaotic and no two implementations agree on a single pixel.
///
/// It used `(1 - v) * (H - 1) + 0.5` for the y: rows counted from the top,
/// i.e. the snow mirrored top to bottom, and off the centres.
#[test]
fn glitch_snow_hashes_gl_frag_coord() {
    let Some(mut r) = headless() else { return };
    let scratch = target(&mut r, 99, FORMAT, W, H);
    let _ = scratch;
    let input = target(&mut r, 1, F16, W, H);
    fill(&mut r, 99, 1, &input, [0.0, 0.0, 0.0, 0.0]);
    let out = target(&mut r, 2, F16, W, H);
    let (seed, amount) = (2.0e-6f32, 0.025f32);
    pass(
        &mut r, 1, (2, &out), (0, 0), 5, seed, [amount, 0.0, 2.0, 2.0], [0.0, 0.0, 0.05, 0.0],
        PostFxCamera::default(),
    );
    let px = read(&r, &out);
    let snow = |xs: f32, ys: f32| 200.0 * amount * glitch_rand(xs * seed, ys * seed * 50.0) * 0.2;
    // Circular distance: the hash wraps at 1.
    let close = |a: f32, b: f32| {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d) < 0.08
    };
    let (mut gl, mut mirrored, mut n) = (0, 0, 0);
    for row in 0..H {
        for col in 0..W {
            let got = px[((row * W + col) * 4) as usize];
            let xs = (2 * col + 1) as f32;
            if close(got, snow(xs, (2 * (H - 1 - row) + 1) as f32)) {
                gl += 1;
            }
            if close(got, snow(xs, (2 * row + 1) as f32)) {
                mirrored += 1;
            }
            n += 1;
        }
    }
    assert!(
        gl * 10 >= n * 9,
        "snow matches three.js's gl_FragCoord hash at {gl} of {n} pixels \
         (the top-down rows would match {mirrored})"
    );
}
