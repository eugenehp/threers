//! Post-fx passes onto the canvas keep the picture the right way up, shared by
//! the WebGL2 suite (`tests/webgl.rs`) and the WebGPU one (`tests/webgpu.rs`).
//!
//! The scene is deliberately asymmetric — green over the top half only, black
//! below — so a vertical flip anywhere between the render target and the
//! canvas shows up as green at the bottom. The render target's own read-back is
//! checked first, so the reference orientation is established on the same
//! backend rather than assumed.
//!
//! Each including file supplies `super::views_renderer()`.

use threers::wasm::{
    WebBufferAttribute, WebBufferGeometry, WebCamera, WebColor, WebEffectComposer, WebEuler,
    WebGeometry, WebMaterial, WebMesh, WebPass, WebRenderTarget, WebRenderer, WebScene,
    WebVector3,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

/// The canvas's current picture, RGBA, top-left origin — `drawImage` onto a 2D
/// canvas in the same task as the draw, as `web_views::read_canvas` does.
fn read_canvas(canvas: &web_sys::HtmlCanvasElement) -> Vec<u8> {
    let document = web_sys::window().unwrap().document().unwrap();
    let copy = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let (w, h) = (canvas.width(), canvas.height());
    copy.set_width(w);
    copy.set_height(h);
    let ctx: JsValue = copy.get_context("2d").unwrap().unwrap().into();
    let call = |name: &str, args: &[JsValue]| -> JsValue {
        let f: js_sys::Function = js_sys::Reflect::get(&ctx, &name.into())
            .unwrap()
            .dyn_into()
            .unwrap();
        let arr = js_sys::Array::new();
        for a in args {
            arr.push(a);
        }
        f.apply(&ctx, &arr).expect(name)
    };
    call("drawImage", &[canvas.into(), 0.into(), 0.into()]);
    let image = call(
        "getImageData",
        &[0.into(), 0.into(), (w as f64).into(), (h as f64).into()],
    );
    let data = js_sys::Reflect::get(&image, &"data".into()).unwrap();
    js_sys::Uint8ClampedArray::new(&data).to_vec()
}

async fn read_target(renderer: &WebRenderer, id: u32, w: u32, h: u32) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(id, 0, 0, w, h);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
}

/// Where the green is in a top-left-origin RGBA buffer: `"top"`, `"bottom"`,
/// or something else worth printing.
fn orientation(buf: &[u8], w: u32, h: u32) -> String {
    let at = |y: u32| {
        let i = ((y * w + w / 2) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2]]
    };
    let green = |p: [u8; 3]| p[1] > 150 && p[0] < 60 && p[2] < 60;
    let black = |p: [u8; 3]| p.iter().all(|&c| c < 40);
    let (upper, lower) = (at(h / 4), at(h * 3 / 4));
    match (green(upper), black(upper), green(lower), black(lower)) {
        (true, _, _, true) => "top".into(),
        (_, true, true, _) => "bottom".into(),
        _ => format!("unclear: upper {upper:?}, lower {lower:?}"),
    }
}

/// Green over `y ∈ [0, 1]` — the top half of `front_ortho`'s view.
fn top_half_scene() -> WebScene {
    let mut g = WebBufferGeometry::new();
    g.set_attribute(
        "position",
        WebBufferAttribute::new(
            vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0],
            3,
        ),
    );
    g.set_attribute("normal", WebBufferAttribute::new([0.0, 0.0, 1.0].repeat(4), 3));
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&WebMesh::new(
        &WebGeometry::from_buffer_geometry(g),
        &WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0)),
    ));
    scene
}

fn front_ortho() -> WebCamera {
    let mut c = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.set_position(0.0, 0.0, 5.0);
    c.look_at(0.0, 0.0, 0.0);
    c
}

/// `applyPostFx(rt, kind, …)` onto the canvas with the shim's defaults.
fn post_fx_to_canvas(renderer: &mut WebRenderer, rt: u32, kind: u32, p2: [f32; 4]) {
    let identity = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    renderer.apply_post_fx(
        rt,
        0,
        0,
        kind,
        0.0,
        p2[0],
        p2[1],
        p2[2],
        p2[3],
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0.1,
        100.0,
        0.5,
        16,
        identity.clone(),
        identity,
    );
}

/// Reset the shared renderer, draw the scene into a canvas-sized target and
/// check the target and a direct canvas render are both green-on-top.
async fn setup(
    renderer: &mut WebRenderer,
    canvas: &web_sys::HtmlCanvasElement,
) -> (WebRenderTarget, u32, u32) {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    renderer.set_pixel_ratio(1.0);
    let (w, h) = (canvas.width(), canvas.height());
    let mut scene = top_half_scene();
    let camera = front_ortho();
    let target = WebRenderTarget::new(renderer, w, h);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let rt_pixels = read_target(renderer, target.id, w, h).await;
    renderer.set_render_target(0);
    assert_eq!(orientation(&rt_pixels, w, h), "top", "render target read-back");
    renderer.render(&mut scene, &camera);
    assert_eq!(orientation(&read_canvas(canvas), w, h), "top", "direct canvas render");
    (target, w, h)
}

/// A render target drawn with green on top reaches the canvas with green on
/// top through the post-fx paths that sample it the right way round: the
/// copy through a scratch target (`applyPostFx(rt, 0, 0, [1, 0, 0, 0])`, what
/// the shim's composers use), the sRGB-decode copy (`p2.y = 1`, also the
/// WebGL2 persistent canvas buffer's present), and single-pass effects. Plus
/// a canvas render that goes through the persistent buffer on WebGL2.
#[wasm_bindgen_test]
async fn post_fx_to_the_canvas_keeps_the_picture_upright() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let (target, w, h) = setup(renderer, &canvas).await;

    let cases: [(&str, u32, [f32; 4]); 4] = [
        ("kind 0, p2.x = 1 (via scratch target)", 0, [1.0, 0.0, 0.0, 0.0]),
        ("kind 0, p2.y = 1 (sRGB-decode copy)", 0, [0.0, 1.0, 0.0, 0.0]),
        ("kind 1 (FXAA)", 1, [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0]),
        ("kind 20 (bilinear resample)", 20, [0.0; 4]),
    ];
    let mut results = Vec::new();
    for (name, kind, p2) in cases {
        post_fx_to_canvas(renderer, target.id, kind, p2);
        results.push((name, orientation(&read_canvas(&canvas), w, h)));
    }
    // A scissor test switches canvas draws on WebGL2 to the persistent back
    // buffer, presented with the sRGB-decode copy.
    renderer.set_scissor_test(true);
    renderer.set_scissor_test(false);
    renderer.render(&mut top_half_scene(), &front_ortho());
    results.push(("canvas render after a scissor test", orientation(&read_canvas(&canvas), w, h)));

    let wrong: Vec<_> = results.iter().filter(|(_, o)| o != "top").collect();
    assert!(
        wrong.is_empty(),
        "backend {}: not upright: {wrong:?} (all: {results:?})",
        renderer.backend()
    );
}

/// The plain kind-0 copy onto the canvas (`applyPostFx(rt, 0)` with default
/// `p2`) and render-target to render-target passes keep the picture upright.
///
/// `POSTFX_SHADER`'s `vs_main` used to put uv (0, 0) at the bottom-left of
/// clip space while wgpu textures have v = 0 at the top row, so a pass was a
/// vertical flip unless `resolution.w` asked for the compensating flip, which
/// `apply_postfx_by_id` did only on some canvas writes — never into a render
/// target, and not for the kind-0 byte copy to the canvas. The uv is texture
/// space now, on WebGPU and WebGL2 alike, and the flag is gone.
#[wasm_bindgen_test]
async fn post_fx_copies_keep_orientation_everywhere() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let (target, w, h) = setup(renderer, &canvas).await;
    let mut results = Vec::new();
    post_fx_to_canvas(renderer, target.id, 0, [0.0; 4]);
    results.push(("canvas kind 0 (plain copy)", orientation(&read_canvas(&canvas), w, h)));
    let second = WebRenderTarget::new(renderer, w, h);
    let identity = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    for (name, kind, p2) in [
        ("target kind 0 (copy)", 0u32, [0.0f32; 4]),
        ("target kind 1 (FXAA)", 1, [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0]),
    ] {
        renderer.apply_post_fx_to_rt(
            target.id, second.id, 0, 0, kind, 0.0, p2[0], p2[1], p2[2], p2[3], 0.0, 0.0, 0.0,
            0.0, 0.1, 100.0, 0.5, 16, identity.clone(), identity.clone(),
        );
        let px = read_target(renderer, second.id, w, h).await;
        results.push((name, orientation(&px, w, h)));
    }
    let wrong: Vec<_> = results.iter().filter(|(_, o)| o != "top").collect();
    assert!(
        wrong.is_empty(),
        "backend {}: not upright: {wrong:?} (all: {results:?})",
        renderer.backend()
    );
}

// ---------------------------------------------------------------------------
// Every kind, and the screen-space effects against geometry.
//
// The same questions `tests/postfx_orientation.rs` asks natively, asked here
// of the browser backends: a marker in the top-left quadrant only, so a
// vertical flip moves it to the bottom-left and a horizontal one to the right.

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Post-fx parameters beyond the input: what both entry points take.
#[derive(Clone)]
struct Fx {
    depth: u32,
    normal: u32,
    kind: u32,
    time: f32,
    p2: [f32; 4],
    p3: [f32; 4],
    near: f32,
    far: f32,
    kernel_radius: f32,
    proj: Vec<f32>,
    inv_proj: Vec<f32>,
}

impl Fx {
    fn kind(kind: u32, time: f32, p2: [f32; 4], p3: [f32; 4]) -> Fx {
        Fx {
            depth: 0,
            normal: 0,
            kind,
            time,
            p2,
            p3,
            near: 0.1,
            far: 100.0,
            kernel_radius: 0.5,
            proj: IDENTITY.to_vec(),
            inv_proj: IDENTITY.to_vec(),
        }
    }
}

fn fx_to_canvas(renderer: &mut WebRenderer, input: u32, f: &Fx) {
    renderer.apply_post_fx(
        input, f.depth, f.normal, f.kind, f.time, f.p2[0], f.p2[1], f.p2[2], f.p2[3], f.p3[0],
        f.p3[1], f.p3[2], f.p3[3], 0, f.near, f.far, f.kernel_radius, 32, f.proj.clone(),
        f.inv_proj.clone(),
    );
}

fn fx_to_target(renderer: &mut WebRenderer, input: u32, output: u32, f: &Fx) {
    renderer.apply_post_fx_to_rt(
        input, output, f.depth, f.normal, f.kind, f.time, f.p2[0], f.p2[1], f.p2[2], f.p2[3],
        f.p3[0], f.p3[1], f.p3[2], f.p3[3], f.near, f.far, f.kernel_radius, 32, f.proj.clone(),
        f.inv_proj.clone(),
    );
}

/// RGBA8 as `[0, 1]` floats.
fn unit(buf: &[u8]) -> Vec<f32> {
    buf.iter().map(|&b| b as f32 / 255.0).collect()
}

fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => s * f32::INFINITY,
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A half-float target's pixels, clamped to `[0, 1]` as a display would.
async fn read_target_f16(renderer: &WebRenderer, id: u32, w: u32, h: u32) -> Vec<f32> {
    let promise = renderer.read_render_target_f16(id, 0, 0, w, h);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("f16 read-back rejected");
    let bytes = js_sys::Uint8Array::new(&value).to_vec();
    bytes
        .chunks_exact(2)
        .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])).clamp(0.0, 1.0))
        .collect()
}

/// `"top-left"` when that quadrant clearly carries the most r+g+b, else the
/// four quadrant means.
fn marker_quadrant(px: &[f32], w: u32, h: u32) -> String {
    let mut sum = [0f32; 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let q = (if y < h / 2 { 0 } else { 2 }) + (if x < w / 2 { 0 } else { 1 });
            sum[q] += px[i] + px[i + 1] + px[i + 2];
        }
    }
    let q = sum.map(|s| s / (w * h / 4) as f32);
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

/// An axis-aligned quad at depth `z` over `x0..x1`, `y0..y1`, facing +z.
fn quad(x0: f32, x1: f32, y0: f32, y1: f32, z: f32, color: &WebColor) -> WebMesh {
    let mut g = WebBufferGeometry::new();
    g.set_attribute(
        "position",
        WebBufferAttribute::new(vec![x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z], 3),
    );
    g.set_attribute("normal", WebBufferAttribute::new([0.0, 0.0, 1.0].repeat(4), 3));
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    WebMesh::new(&WebGeometry::from_buffer_geometry(g), &WebMaterial::basic(color))
}

/// White square over x ∈ [-0.75, -0.25], y ∈ [0.25, 0.75] of `front_ortho`:
/// inside the top-left quadrant and clear of both centre lines.
fn marker_scene() -> WebScene {
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&quad(-0.75, -0.25, 0.25, 0.75, 0.0, &WebColor::new(1.0, 1.0, 1.0)));
    scene
}

/// `(name, Fx)` for every kind with a picture to keep. Not here: 11 and 12
/// (SSAO, SSR — against geometry below), 26 and 34 (bokeh needs depth, the
/// luminance adaption a 1×1 average), 23 (LUT: a pointwise colour map from
/// a second texture, constant with none bound), 29 (solid clear, no input)
/// and 15 (multiply onto what the target holds — into a target only).
fn every_kind(w: u32, h: u32) -> Vec<(&'static str, Fx)> {
    let pi = std::f32::consts::PI;
    let glitch2 = [0.004, 0.3, 2.0, 2.0];
    let glitch3 = [0.0, 0.0, 0.05, 0.0];
    let k = Fx::kind;
    vec![
        ("0 copy", k(0, 0.0, [0.0; 4], [0.0; 4])),
        ("0 copy, sRGB-decode", k(0, 0.0, [0.0, 1.0, 0.0, 0.0], [0.0; 4])),
        ("1 FXAA", k(1, 0.0, [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0], [0.0; 4])),
        ("2 film", k(2, 0.25, [0.5, 0.0, 0.0, 0.0], [0.0; 4])),
        ("3 dot screen", k(3, 0.0, [1.57, 1.0, 0.5, 0.5], [0.0; 4])),
        ("4 halftone", k(4, 0.0, [4.0, 0.0, 1.0, 1.0], [pi / 12.0, pi / 6.0, pi / 4.0, 1.0])),
        ("5 glitch", k(5, 0.5, glitch2, glitch3)),
        ("31 glitch, no snow", k(31, 0.5, glitch2, glitch3)),
        ("6 bloom threshold", k(6, 0.0, [0.5, 0.1, 0.0, 0.0], [0.0; 4])),
        ("7 blur h", k(7, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("8 blur v", k(8, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("9 bloom composite", k(9, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("10 sobel", k(10, 0.0, [1.0, 1.0, 1.0, 1.0], [0.0; 4])),
        ("13 SSAO blur", k(13, 0.0, [0.0; 4], [0.0; 4])),
        ("14 SSAO composite", k(14, 0.0, [0.0; 4], [0.0; 4])),
        ("16 outline edge", k(16, 0.0, [1.0, 1.0, 1.0, 1.0], [0.0; 4])),
        ("17 outline overlay", k(17, 0.0, [3.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("18 outline blur h", k(18, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("19 outline blur v", k(19, 0.0, [2.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("20 bilinear resample", k(20, 0.0, [0.0; 4], [0.0; 4])),
        ("21 afterimage", k(21, 0.0, [0.95, 0.0, 0.0, 0.0], [0.0; 4])),
        ("22 transition", k(22, 0.0, [0.0; 4], [0.0; 4])),
        ("24 pixelated", k(24, 0.0, [4.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("25 adaptive tone map", k(25, 0.0, [0.6, 0.01, 16.0, 1.0], [0.0; 4])),
        ("27 SAO", k(27, 0.0, [0.5, 0.0, 0.0, 0.0], [0.0; 4])),
        ("28 SMAA", k(28, 0.0, [0.0; 4], [0.0; 4])),
        ("30 output", k(30, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 4])),
        ("33 luminosity", k(33, 0.0, [0.0; 4], [0.0; 4])),
    ]
}

/// Every post-fx kind keeps the top-left marker top-left: onto the canvas,
/// into an 8-bit target, and from a half-float target into another (the
/// shim's `EffectComposer` targets are half-float).
#[wasm_bindgen_test]
async fn every_post_fx_kind_keeps_the_marker_top_left() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let camera = front_ortho();

    let input = WebRenderTarget::new(renderer, w, h);
    let output = WebRenderTarget::new(renderer, w, h);
    let input_f16 = WebRenderTarget::new_half_float(renderer, w, h);
    let output_f16 = WebRenderTarget::new_half_float(renderer, w, h);
    for id in [input.id, input_f16.id] {
        renderer.set_render_target(id);
        renderer.render(&mut marker_scene(), &camera);
        renderer.render(&mut marker_scene(), &camera);
    }
    renderer.set_render_target(0);
    let seen = unit(&read_target(renderer, input.id, w, h).await);
    assert_eq!(marker_quadrant(&seen, w, h), "top-left", "the 8-bit scene render");
    let seen = read_target_f16(renderer, input_f16.id, w, h).await;
    assert_eq!(marker_quadrant(&seen, w, h), "top-left", "the half-float scene render");

    let mut wrong = Vec::new();
    let mut checked = 0;
    for (name, f) in every_kind(w, h) {
        fx_to_canvas(renderer, input.id, &f);
        let at = marker_quadrant(&unit(&read_canvas(&canvas)), w, h);
        if at != "top-left" {
            wrong.push(format!("{name} → canvas: {at}"));
        }
        fx_to_target(renderer, input.id, output.id, &f);
        let at = marker_quadrant(&unit(&read_target(renderer, output.id, w, h).await), w, h);
        if at != "top-left" {
            wrong.push(format!("{name} → target: {at}"));
        }
        fx_to_target(renderer, input_f16.id, output_f16.id, &f);
        let at = marker_quadrant(&read_target_f16(renderer, output_f16.id, w, h).await, w, h);
        if at != "top-left" {
            wrong.push(format!("{name} → f16 target: {at}"));
        }
        checked += 3;
    }
    // Kind 15 multiplies onto the target's contents: give it white first.
    let white = Fx::kind(29, 0.0, [1.0; 4], [0.0; 4]);
    fx_to_target(renderer, input.id, output.id, &white);
    fx_to_target(renderer, input.id, output.id, &Fx::kind(15, 0.0, [0.0; 4], [0.0; 4]));
    let at = marker_quadrant(&unit(&read_target(renderer, output.id, w, h).await), w, h);
    if at != "top-left" {
        wrong.push(format!("15 SSAO multiply → target: {at}"));
    }
    checked += 1;
    assert!(
        wrong.is_empty(),
        "backend {}: {} of {checked} passes moved the marker:\n  {}",
        renderer.backend(),
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// What the shim's `AfterimagePass` does: blend the frame with the previous
/// composite (kind 21, the previous one bound as the second input) into a
/// target, then copy that to the canvas. Each step used to flip, and two flips
/// made it look right.
#[wasm_bindgen_test]
async fn afterimage_chain_keeps_the_marker_top_left() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let frame = WebRenderTarget::new(renderer, w, h);
    let old = WebRenderTarget::new(renderer, w, h);
    let comp = WebRenderTarget::new(renderer, w, h);
    renderer.set_render_target(frame.id);
    renderer.render(&mut marker_scene(), &front_ortho());
    renderer.set_render_target(0);
    let mut blend = Fx::kind(21, 0.0, [0.96, 0.0, 0.0, 0.0], [0.0; 4]);
    blend.normal = old.id;
    fx_to_target(renderer, frame.id, comp.id, &blend);
    let in_target = marker_quadrant(&unit(&read_target(renderer, comp.id, w, h).await), w, h);
    fx_to_canvas(renderer, comp.id, &Fx::kind(0, 0.0, [0.0; 4], [0.0; 4]));
    let on_canvas = marker_quadrant(&unit(&read_canvas(&canvas)), w, h);
    assert_eq!(
        (in_target.as_str(), on_canvas.as_str()),
        ("top-left", "top-left"),
        "backend {}",
        renderer.backend()
    );
}

/// The canvas copy needs no scratch target. `applyPostFx(rt, 0, [1, …])`
/// used to go render target → scratch → canvas, two flips, and the comment
/// on it blamed a Dawn silhouette glitch — a flipped copy of a near-symmetric
/// scene differs from the original only at its silhouettes. The direct copy
/// now matches that route byte for byte, from an 8-bit and a half-float target.
#[wasm_bindgen_test]
async fn the_canvas_copy_matches_the_route_through_a_target() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    // Something with gradients and partial coverage, not just flat fills.
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.1, 0.2, 0.3));
    scene.add(&quad(-0.9, 0.3, -0.2, 0.8, 0.0, &WebColor::new(0.9, 0.4, 0.1)));
    let tilted = scene.add(&quad(-0.4, 0.6, -0.6, 0.4, 0.5, &WebColor::new(0.2, 0.8, 0.5)));
    scene.set_transform(&tilted, &WebVector3::new(0.0, 0.0, 0.0), &WebEuler::new(0.0, 0.0, 0.4));
    let camera = front_ortho();
    let mut differ = Vec::new();
    for half in [false, true] {
        let src = if half {
            WebRenderTarget::new_half_float(renderer, w, h)
        } else {
            WebRenderTarget::new(renderer, w, h)
        };
        // The old scratch target was canvas-format, 8-bit.
        let via = WebRenderTarget::new(renderer, w, h);
        renderer.set_render_target(src.id);
        renderer.render(&mut scene, &camera);
        renderer.set_render_target(0);
        let copy = Fx::kind(0, 0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 4]);
        fx_to_canvas(renderer, src.id, &copy);
        let direct = read_canvas(&canvas);
        fx_to_target(renderer, src.id, via.id, &copy);
        fx_to_canvas(renderer, via.id, &copy);
        let routed = read_canvas(&canvas);
        let n = direct.iter().zip(&routed).filter(|(a, b)| a.abs_diff(**b) > 1).count();
        if n > 0 {
            differ.push(format!("{} input: {n} bytes differ", if half { "f16" } else { "8-bit" }));
        }
    }
    assert!(differ.is_empty(), "backend {}: {differ:?}", renderer.backend());
}

/// The three.js SSAO kernel shape (points in the +z hemisphere, denser near
/// the origin) from a fixed LCG.
fn ssao_kernel() -> Vec<f32> {
    let mut s: u32 = 0x5a00426;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s as f32 / 4294967296.0
    };
    let mut k = Vec::new();
    for i in 0..32 {
        let v = [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd()];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        let t = i as f32 / 32.0;
        let scale = 0.1 + 0.9 * t * t;
        k.extend(v.iter().map(|c| c / len * scale));
    }
    k
}

/// A floor and, over the left half only, a back wall standing on it; the
/// camera above and in front puts the crease in the upper half. Green wall,
/// blue floor, so the crease row is read off the colour render — colours that
/// survive the red/blue swap of a BGRA canvas-format target's read-back.
fn crease_scene(material: impl Fn(&WebColor) -> WebMaterial, background: &WebColor) -> WebScene {
    let mut scene = WebScene::new();
    scene.set_background(background);
    let floor = scene.add(&WebMesh::new(
        &WebGeometry::box_(8.0, 0.1, 8.0),
        &material(&WebColor::new(0.0, 0.0, 1.0)),
    ));
    scene.set_transform(&floor, &WebVector3::new(0.0, -0.05, 0.0), &WebEuler::new(0.0, 0.0, 0.0));
    let wall = scene.add(&WebMesh::new(
        &WebGeometry::box_(4.0, 3.0, 0.1),
        &material(&WebColor::new(0.0, 1.0, 0.0)),
    ));
    scene.set_transform(&wall, &WebVector3::new(-2.0, 1.5, -1.05), &WebEuler::new(0.0, 0.0, 0.0));
    scene
}

/// SSAO (kind 11) darkens the crease that is there — upper half, left half
/// — and not its mirror image or the open floor; into a target and onto the
/// canvas. A uv flip between the depth fetch and the reconstruction moves
/// the darkening or smears it over flat floor.
#[wasm_bindgen_test]
async fn ssao_darkens_the_crease_where_it_is() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let mut camera = WebCamera::perspective(50.0, w as f32 / h as f32, 0.1, 20.0);
    camera.set_position(0.0, 1.5, 3.0);
    camera.look_at(0.0, 0.0, 0.5);

    let colour = WebRenderTarget::new(renderer, w, h);
    let normals = WebRenderTarget::new(renderer, w, h);
    let ao = WebRenderTarget::new(renderer, w, h);
    let black = WebColor::new(0.0, 0.0, 0.0);
    renderer.set_render_target(colour.id);
    renderer.render(&mut crease_scene(WebMaterial::basic, &black), &camera);
    renderer.set_render_target(normals.id);
    renderer.render(
        &mut crease_scene(|_| WebMaterial::normal_mat(), &WebColor::new(0.467, 0.467, 1.0)),
        &camera,
    );
    renderer.set_render_target(0);
    let colour_px = unit(&read_target(renderer, colour.id, w, h).await);
    let crease_at = |x: u32| {
        let at = |y: u32| {
            let i = ((y * w + x) * 4) as usize;
            // Red and blue together: which is which depends on BGRA.
            (colour_px[i + 1], colour_px[i] + colour_px[i + 2])
        };
        (1..h).find(|&y| {
            let (g0, rb0) = at(y - 1);
            let (g1, rb1) = at(y);
            g0 > 0.3 && rb0 < 0.2 && rb1 > 0.3 && g1 < 0.2
        })
    };
    let crease = crease_at(w / 4).expect("no wall/floor crease in the left half");
    assert!(crease < h / 2 - 4, "the crease is at row {crease}, meant to be in the upper half");
    assert!(crease_at(w * 3 / 4).is_none(), "the wall is meant to cover the left half only");

    let normal_px = read_target(renderer, normals.id, w, h).await;
    let sample = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        [normal_px[i], normal_px[i + 1], normal_px[i + 2]]
    };
    // Normal prepass sanity, reported with any failure below.
    let normal_note = format!(
        "normal RT: wall {:?}, floor {:?}",
        sample(w / 4, crease / 2),
        sample(w / 4, h - 4)
    );
    // Both backends store that prepass alike: the packed normal itself —
    // (128, 62, 237) for the wall's (0.502, 0.243, 0.929) — as three.js does.
    // WebGL2 has no linear canvas format and no view formats, so every target
    // there is Rgba8UnormSrgb and encodes on write; the normal material
    // decodes first there (`framebuffer_data`), and the post-fx pass re-encodes
    // what its decoding view returns (`postfx_flag`). Before, WebGL2 stored the
    // encoding (188, 135, 247) and its AO output was encoded once more than on
    // WebGPU. Depth differed too: on WebGL2 the hardware-depth read returned 0
    // everywhere (see `sample_postfx_depth`), so the occlusion was computed
    // against a scene flattened onto the near plane.
    renderer.set_ssao_kernel(&ssao_kernel());
    let proj = camera.projection_matrix();
    let mut m = [0f32; 16];
    m.copy_from_slice(&proj[..16]);
    let inv = threers::Matrix4 { elements: m }.invert().elements.to_vec();
    let mut f = Fx::kind(11, 0.0, [0.0005, 0.1, 0.0, 0.0], [0.0; 4]);
    (f.depth, f.normal, f.near, f.far, f.kernel_radius) = (normals.id, normals.id, 0.1, 20.0, 0.6);
    (f.proj, f.inv_proj) = (proj, inv);

    let mean = |px: &[f32], xs: std::ops::Range<u32>, ys: std::ops::Range<u32>| {
        let (mut s, mut n) = (0.0, 0.0);
        for y in ys {
            for x in xs.clone() {
                s += px[((y * w + x) * 4) as usize];
                n += 1.0;
            }
        }
        s / n
    };
    let mut failures = Vec::new();
    for dest in ["target", "canvas"] {
        let px = if dest == "target" {
            fx_to_target(renderer, colour.id, ao.id, &f);
            unit(&read_target(renderer, ao.id, w, h).await)
        } else {
            fx_to_canvas(renderer, colour.id, &f);
            unit(&read_canvas(&canvas))
        };
        let left = 2..w / 2 - 4;
        let band = crease.saturating_sub(2)..crease + 3;
        let mirror = h - 1 - crease;
        let in_crease = mean(&px, left.clone(), band.clone());
        let at_mirror = mean(&px, left.clone(), mirror.saturating_sub(2)..mirror + 3);
        let right_of_crease = mean(&px, w * 5 / 8..w - 2, band);
        let open_floor = mean(&px, left, h - 8..h - 2);
        // A coarse map of the AO, for the failure message.
        let map: Vec<String> = (0..6)
            .map(|j| {
                (0..8)
                    .map(|i| {
                        let (x0, y0) = (i * w / 8, j * h / 6);
                        format!("{:.2}", mean(&px, x0..x0 + w / 8, y0..y0 + h / 6))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let report = format!(
            "{dest}: crease row {crease}: AO {in_crease:.3} in the crease, {at_mirror:.3} at the \
             mirrored rows, {right_of_crease:.3} right of the wall, {open_floor:.3} open floor; \
             AO by block:\n    {}",
            map.join("\n    ")
        );
        if !(in_crease < 0.85
            && at_mirror > in_crease + 0.1
            && right_of_crease > in_crease + 0.1
            && open_floor > 0.9)
        {
            failures.push(report);
        }
    }
    assert!(
        failures.is_empty(),
        "backend {}: SSAO misplaced ({normal_note}):\n  {}",
        renderer.backend(),
        failures.join("\n  ")
    );
}

/// The shim's `SSAOPass` default output, three.js r165's: the beauty pass
/// times the blurred AO (kind 13 of kind 11) — three.js draws a copy and then
/// the AO over it with a multiply; kind 14 with `p2.y` is the two in one pass.
/// Measured against kind 0's legacy copy, which is what the shim drew before:
/// darker in the crease and the same on open floor, where nothing occludes.
/// One pass rather than three.js's two draws, which WebGL2 used to lose —
/// the second started from a cleared canvas (see
/// `bloom_over_its_copy_keeps_both_on_the_canvas`).
#[wasm_bindgen_test]
async fn ssao_default_output_darkens_the_beauty_in_the_crease_only() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let mut camera = WebCamera::perspective(50.0, w as f32 / h as f32, 0.1, 20.0);
    camera.set_position(0.0, 1.5, 3.0);
    camera.look_at(0.0, 0.0, 0.5);
    let beauty = WebRenderTarget::new(renderer, w, h);
    let normals = WebRenderTarget::new(renderer, w, h);
    let ao = WebRenderTarget::new(renderer, w, h);
    let blur = WebRenderTarget::new(renderer, w, h);
    // A grey floor and wall, so the multiply shows on every channel.
    renderer.set_render_target(beauty.id);
    renderer.render(
        &mut crease_scene(|_| WebMaterial::basic(&WebColor::new(0.8, 0.8, 0.8)), &WebColor::new(0.0, 0.0, 0.0)),
        &camera,
    );
    renderer.set_render_target(normals.id);
    renderer.render(
        &mut crease_scene(|_| WebMaterial::normal_mat(), &WebColor::new(0.467, 0.467, 1.0)),
        &camera,
    );
    renderer.set_render_target(0);
    renderer.set_ssao_kernel(&ssao_kernel());
    let proj = camera.projection_matrix();
    let mut m = [0f32; 16];
    m.copy_from_slice(&proj[..16]);
    let inv = threers::Matrix4 { elements: m }.invert().elements.to_vec();
    let mut f = Fx::kind(11, 0.0, [0.0005, 0.1, 0.0, 0.0], [0.0; 4]);
    (f.depth, f.normal, f.near, f.far, f.kernel_radius) = (normals.id, normals.id, 0.1, 20.0, 0.6);
    (f.proj, f.inv_proj) = (proj, inv);
    fx_to_target(renderer, beauty.id, ao.id, &f);
    fx_to_target(renderer, ao.id, blur.id, &Fx::kind(13, 0.0, [0.0; 4], [0.0; 4]));

    let copy = Fx::kind(0, 0.0, [0.0, 1.0, 0.0, 0.0], [0.0; 4]);
    fx_to_canvas(renderer, beauty.id, &copy);
    let plain = unit(&read_canvas(&canvas));
    let mut composite = Fx::kind(14, 0.0, [0.0, 1.0, 0.0, 0.0], [0.0; 4]);
    composite.depth = blur.id;
    fx_to_canvas(renderer, beauty.id, &composite);
    let shaded = unit(&read_canvas(&canvas));

    // The crease row, off the beauty pass: the last row of wall above floor
    // in the left half. Grey on grey, so found off the AO target instead —
    // the darkest row of the left half's upper part.
    let ao_px = unit(&read_target(renderer, blur.id, w, h).await);
    let mean = |px: &[f32], xs: std::ops::Range<u32>, ys: std::ops::Range<u32>| {
        let (mut s, mut n) = (0.0, 0.0);
        for y in ys {
            for x in xs.clone() {
                let i = ((y * w + x) * 4) as usize;
                s += px[i] + px[i + 1] + px[i + 2];
                n += 3.0;
            }
        }
        s / n
    };
    let left = 2..w / 2 - 4;
    let crease = (2..h / 2)
        .min_by(|&a, &b| mean(&ao_px, left.clone(), a..a + 1).total_cmp(&mean(&ao_px, left.clone(), b..b + 1)))
        .unwrap();
    let band = crease.saturating_sub(1)..crease + 2;
    let floor = h - 8..h - 2;
    let (plain_crease, shaded_crease) = (mean(&plain, left.clone(), band.clone()), mean(&shaded, left.clone(), band));
    let (plain_floor, shaded_floor) = (mean(&plain, left.clone(), floor.clone()), mean(&shaded, left, floor));
    let report = format!(
        "backend {}: crease row {crease}: beauty {plain_crease:.3} → {shaded_crease:.3}; \
         open floor {plain_floor:.3} → {shaded_floor:.3}",
        renderer.backend()
    );
    assert!(plain_crease > 0.2 && plain_floor > 0.2, "the beauty pass is not grey: {report}");
    assert!(shaded_crease < plain_crease * 0.9, "no AO in the crease: {report}");
    assert!((shaded_floor - plain_floor).abs() < 2.5 / 255.0, "open floor changed: {report}");
}

/// SSR (kind 12) reflects below the object, in its own half, and leaves the
/// object where it is. Green (BGRA-proof) block over texture rows v ∈ [0.1, 0.4] of the left
/// half; depth marks only its top strip v ∈ [0.1, 0.15] as near, so the one
/// reflection is of that strip, at v ∈ [0.66, 0.89] (mirror about the
/// midline, searched at most twelve pixels). A flipped frame has the block at
/// [0.6, 0.9] instead: v = 0.38 and v = 0.62 tell the two apart.
#[wasm_bindgen_test]
async fn ssr_reflects_below_the_object() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let camera = front_ortho();
    let colour = WebRenderTarget::new(renderer, w, h);
    let depth = WebRenderTarget::new(renderer, w, h);
    let out = WebRenderTarget::new(renderer, w, h);

    let mut colour_scene = WebScene::new();
    colour_scene.set_background(&WebColor::new(0.3, 0.3, 0.3));
    colour_scene.add(&quad(-0.8, -0.2, 0.2, 0.8, 0.0, &WebColor::new(0.0, 1.0, 0.0)));
    let mut depth_scene = WebScene::new();
    depth_scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    depth_scene.add(&quad(-0.8, -0.2, 0.7, 0.8, 0.0, &WebColor::new(1.0, 1.0, 1.0)));
    renderer.set_render_target(colour.id);
    renderer.render(&mut colour_scene, &camera);
    renderer.set_render_target(depth.id);
    renderer.render(&mut depth_scene, &camera);
    renderer.set_render_target(0);

    let mut f = Fx::kind(12, 0.0, [0.5, 0.1, 1.0, 1.0], [0.0; 4]);
    f.depth = depth.id;
    let mut failures = Vec::new();
    for dest in ["target", "canvas"] {
        let px = if dest == "target" {
            fx_to_target(renderer, colour.id, out.id, &f);
            unit(&read_target(renderer, out.id, w, h).await)
        } else {
            fx_to_canvas(renderer, colour.id, &f);
            unit(&read_canvas(&canvas))
        };
        let green_at = |fx: f32, fy: f32| {
            let (x, y) = ((fx * w as f32) as u32, (fy * h as f32) as u32);
            let i = ((y * w + x) * 4) as usize;
            px[i + 1] - px[i].max(px[i + 2]) > 0.4
        };
        for (what, fx, fy, want) in [
            ("the block's top", 0.25, 0.12, true),
            ("the block's bottom", 0.25, 0.38, true),
            ("open background between", 0.25, 0.5, false),
            ("the gap above the reflection", 0.25, 0.62, false),
            ("the reflection", 0.25, 0.87, true),
            ("nothing right of the reflection", 0.75, 0.87, false),
            ("nothing right of the block", 0.75, 0.25, false),
        ] {
            if green_at(fx, fy) != want {
                failures.push(format!(
                    "{dest}: {what} at ({fx}, {fy}) is {}green",
                    if want { "not " } else { "" }
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "backend {}: SSR misplaced:\n  {}",
        renderer.backend(),
        failures.join("\n  ")
    );
}

/// The exact sRGB decode, as a texture view or blend unit applies it.
fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Computed passes store the same bytes on WebGL2 as on WebGPU.
///
/// Every `POSTFX_SHADER` branch returns the byte a plain `Unorm` target —
/// the WebGPU canvas and its targets — stores. WebGL2 has neither a linear
/// canvas format nor view formats, so its canvas and every 8-bit target are
/// sRGB and encode on write: a computed pass (film, tone mapping, the SSAO
/// blur) came out encoded once more there than on WebGPU, and the raw views
/// the copies and blurs read through decoded. Each case's byte is worked out
/// here from the input's own read-back, the same rule on both backends, so
/// this suite in each browser holds the two to the same numbers (±2).
#[wasm_bindgen_test]
async fn computed_post_fx_stores_the_same_bytes_on_every_backend() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let input = WebRenderTarget::new(renderer, w, h);
    let output = WebRenderTarget::new(renderer, w, h);
    // Red equal to blue, so a BGRA target's read-back needs no swizzle.
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.2, 0.5, 0.2));
    renderer.set_render_target(input.id);
    renderer.render(&mut scene, &front_ortho());
    renderer.set_render_target(0);
    let centre = |px: &[u8]| {
        let i = (((h / 2) * w + w / 2) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    let stored = centre(&read_target(renderer, input.id, w, h).await);
    let lin = stored.map(|b| srgb_decode(b as f32 / 255.0));
    let lum = 0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2];
    let scaled = lum * 0.6 / 1.0;
    let tone = scaled * (1.0 + scaled / (16.0 * 16.0)) / (1.0 + scaled);
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let cases: Vec<(&str, Fx, [u8; 3])> = vec![
        ("0 copy", Fx::kind(0, 0.0, [0.0; 4], [0.0; 4]), stored),
        ("0 copy, sRGB-decode", Fx::kind(0, 0.0, [0.0, 1.0, 0.0, 0.0], [0.0; 4]), lin.map(byte)),
        ("2 film, intensity 0", Fx::kind(2, 0.0, [0.0; 4], [0.0; 4]), lin.map(byte)),
        (
            "25 tone mapping",
            Fx::kind(25, 0.0, [0.6, 0.01, 16.0, 1.0], [0.0; 4]),
            lin.map(|c| byte(tone * c)),
        ),
        ("33 luminosity", Fx::kind(33, 0.0, [0.0; 4], [0.0; 4]), [byte(lum); 3]),
        ("20 resample", Fx::kind(20, 0.0, [0.0; 4], [0.0; 4]), lin.map(|c| byte(c.powf(1.0 / 2.2)))),
        ("13 SSAO blur", Fx::kind(13, 0.0, [0.0; 4], [0.0; 4]), [stored[0]; 3]),
    ];
    let mut wrong = Vec::new();
    for (name, f, want) in cases {
        fx_to_canvas(renderer, input.id, &f);
        let on_canvas = centre(&read_canvas(&canvas));
        fx_to_target(renderer, input.id, output.id, &f);
        let in_target = centre(&read_target(renderer, output.id, w, h).await);
        for (dest, got) in [("canvas", on_canvas), ("target", in_target)] {
            if got.iter().zip(want).any(|(g, w)| g.abs_diff(w) > 2) {
                wrong.push(format!("{name} → {dest}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "backend {}: input stored {stored:?}:\n  {}",
        renderer.backend(),
        wrong.join("\n  ")
    );
}

/// The shim's `UnrealBloomPass` onto the canvas: a copy of the frame, then
/// the blurred highlights added over it (kind 9, additive). On WebGL2 the
/// second draw used to start from a cleared canvas — wgpu's GL backend hands
/// the canvas out as a fresh texture each time — so the frame vanished and
/// only the bloom was left. Canvas post-fx there now draws through the
/// persistent canvas buffer, as viewports do: both must be on the canvas.
#[wasm_bindgen_test]
async fn bloom_over_its_copy_keeps_both_on_the_canvas() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let input = WebRenderTarget::new(renderer, w, h);
    let threshold = WebRenderTarget::new(renderer, w, h);
    let blur_h = WebRenderTarget::new(renderer, w, h);
    let blur_v = WebRenderTarget::new(renderer, w, h);
    // A dim grey frame with a white square in the top-left quadrant: only
    // the square passes the threshold.
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.05, 0.05, 0.05));
    scene.add(&quad(-0.75, -0.25, 0.25, 0.75, 0.0, &WebColor::new(1.0, 1.0, 1.0)));
    renderer.set_render_target(input.id);
    renderer.render(&mut scene, &front_ortho());
    renderer.set_render_target(0);
    let k = Fx::kind;
    fx_to_target(renderer, input.id, threshold.id, &k(6, 0.0, [0.85, 0.1, 0.0, 0.0], [0.0; 4]));
    fx_to_target(renderer, threshold.id, blur_h.id, &k(7, 0.0, [4.7, 0.0, 0.0, 0.0], [0.0; 4]));
    fx_to_target(renderer, blur_h.id, blur_v.id, &k(8, 0.0, [4.7, 0.0, 0.0, 0.0], [0.0; 4]));

    fx_to_canvas(renderer, input.id, &k(0, 0.0, [0.0; 4], [0.0; 4]));
    let copy = read_canvas(&canvas);
    renderer.apply_post_fx(
        blur_v.id, 0, 0, 9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.1, 100.0, 0.5, 32,
        IDENTITY.to_vec(), IDENTITY.to_vec(),
    );
    let bloomed = read_canvas(&canvas);
    let at = |px: &[u8], x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        px[i] as i32
    };
    // The square's right edge is at x = 3w/8; its rows span h/8..3h/8.
    let (edge, row) = (w * 3 / 8, h / 4);
    let report = format!(
        "backend {}: copy: square {}, beside it {}, far {}; with bloom: square {}, beside it {}, far {}",
        renderer.backend(),
        at(&copy, w / 4, row),
        at(&copy, edge + 3, row),
        at(&copy, w * 7 / 8, h * 7 / 8),
        at(&bloomed, w / 4, row),
        at(&bloomed, edge + 3, row),
        at(&bloomed, w * 7 / 8, h * 7 / 8),
    );
    // The frame: the square still white, the far background still its grey.
    assert!(at(&bloomed, w / 4, row) >= 250, "the square is gone: {report}");
    assert!(
        (at(&bloomed, w * 7 / 8, h * 7 / 8) - at(&copy, w * 7 / 8, h * 7 / 8)).abs() <= 2
            && at(&copy, w * 7 / 8, h * 7 / 8) > 20,
        "the background is gone: {report}"
    );
    // The bloom: light spilled past the square's edge. Less of it on WebGL2:
    // an sRGB target blends in linear light, so the same addition lifts a
    // dark byte less than it does on WebGPU's `Unorm` canvas, which adds
    // bytes.
    assert!(
        at(&bloomed, edge + 3, row) > at(&copy, edge + 3, row) + 5,
        "no bloom beside the square: {report}"
    );
}

/// The composer's chains, through `WebEffectComposer` and `WebPass` — the
/// Rust sequencing the shim's classes wrap — keep the marker top-left on
/// the canvas, whatever the last pass is: a shader pass, one that draws in
/// place (a scene render), and the passes that blend several draws onto the
/// screen (UnrealBloomPass's copy-plus-bloom, SSAOPass's beauty-times-AO).
#[wasm_bindgen_test]
async fn composer_chains_keep_the_marker_top_left() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let camera = front_ortho();
    type Chain = (&'static str, fn(u32, u32) -> Vec<WebPass>);
    let chains: Vec<Chain> = vec![
        ("render", |_, _| vec![]),
        ("copy", |_, _| vec![WebPass::shader_pass("CopyShader")]),
        ("fxaa", |_, _| vec![WebPass::shader_pass("FXAAShader")]),
        ("unreal bloom", |w, h| vec![WebPass::unreal_bloom_pass(w, h, 1.0, 0.4, 0.5)]),
        ("ssao", |w, h| vec![WebPass::ssao_pass(w, h, 16)]),
        ("smaa, output", |w, h| vec![WebPass::smaa_pass(w, h), WebPass::output_pass()]),
        ("afterimage", |_, _| vec![WebPass::afterimage_pass(0.96)]),
    ];
    let mut wrong = Vec::new();
    for (name, make) in chains {
        let mut composer = WebEffectComposer::new(w, h, 1.0);
        composer.add_pass(&WebPass::render_pass());
        for p in make(w, h) {
            composer.add_pass(&p);
        }
        let mut scene = marker_scene();
        composer.render1(renderer, &mut scene, &camera, &camera, &camera, &camera, Some(0.0));
        let px = unit(&read_canvas(&canvas));
        let at = marker_quadrant(&px, w, h);
        if at != "top-left" {
            wrong.push(format!("{name}: {at}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "backend {}: composer chains moved the marker:\n  {}",
        renderer.backend(),
        wrong.join("\n  ")
    );
}

/// UnrealBloomPass as the last pass draws a copy of the frame and adds the
/// bloom over it — two draws onto the screen. Both reach the canvas. The two
/// draws are blended in a stand-in that holds display values and copied
/// once, so both backends blend as three.js does on its canvas — in display
/// values — rather than in linear light on WebGL2's sRGB canvas and in bytes
/// on WebGPU's `Unorm` one.
#[wasm_bindgen_test]
async fn composer_bloom_onto_the_canvas_keeps_frame_and_glow() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let _ = setup(renderer, &canvas).await;
    let (w, h) = (canvas.width(), canvas.height());
    let camera = front_ortho();
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.05, 0.05, 0.05));
    scene.add(&quad(-0.75, -0.25, 0.25, 0.75, 0.0, &WebColor::new(1.0, 1.0, 1.0)));
    let render = |renderer: &mut WebRenderer, scene: &mut WebScene, bloom: bool| {
        let mut composer = WebEffectComposer::new(w, h, 1.0);
        composer.add_pass(&WebPass::render_pass());
        let b = WebPass::unreal_bloom_pass(w, h, 1.0, 0.4, 0.5);
        b.set_enabled(bloom);
        composer.add_pass(&b);
        composer.render1(renderer, scene, &camera, &camera, &camera, &camera, Some(0.0));
    };
    render(renderer, &mut scene, false);
    let plain = read_canvas(&canvas);
    render(renderer, &mut scene, true);
    let bloomed = read_canvas(&canvas);
    let at = |px: &[u8], x: u32, y: u32| px[((y * w + x) * 4) as usize] as i32;
    let (edge, row) = (w * 3 / 8, h / 4);
    let report = format!(
        "backend {}: plain: square {}, beside {}, far {}; bloomed: square {}, beside {}, far {}",
        renderer.backend(),
        at(&plain, w / 4, row),
        at(&plain, edge + 3, row),
        at(&plain, w * 7 / 8, h * 7 / 8),
        at(&bloomed, w / 4, row),
        at(&bloomed, edge + 3, row),
        at(&bloomed, w * 7 / 8, h * 7 / 8),
    );
    assert!(at(&bloomed, w / 4, row) >= 250, "the square is gone: {report}");
    let near = at(&bloomed, edge + 3, row) - at(&plain, edge + 3, row);
    let far = at(&bloomed, w * 7 / 8, h * 7 / 8) - at(&plain, w * 7 / 8, h * 7 / 8);
    assert!(near > 20, "no bloom beside the square: {report}");
    // The widest mips reach across a small canvas, but faintly.
    assert!(far >= 0 && far * 4 < near, "no fall-off away from the square: {report}");
}

/// A pass's properties and uniforms forward to the Rust pass by name, from
/// the schema it describes — what the shim's `pass.strength = …` and
/// `pass.uniforms.X.value = …` call.
#[wasm_bindgen_test]
fn pass_parameters_forward_by_name() {
    let bloom = WebPass::unreal_bloom_pass(64, 32, 1.5, 0.4, 0.85);
    assert_eq!(bloom.name(), "UnrealBloomPass");
    assert!(bloom.set("strength", wasm_bindgen::JsValue::from_f64(2.5)));
    assert_eq!(bloom.get("strength").as_f64(), Some(2.5));
    assert!(!bloom.set("noSuchThing", wasm_bindgen::JsValue::from_f64(1.0)));
    let film = WebPass::film_pass(0.35, false);
    let schema = film.param_schema();
    let uniform_names: Vec<String> = schema
        .iter()
        .filter(|o| js_sys::Reflect::get(o, &"uniform".into()).unwrap().as_bool() == Some(true))
        .map(|o| js_sys::Reflect::get(&o, &"name".into()).unwrap().as_string().unwrap())
        .collect();
    for n in ["intensity", "grayscale", "time"] {
        assert!(uniform_names.iter().any(|u| u == n), "FilmPass has no uniform {n}: {uniform_names:?}");
    }
    let dot = WebPass::dot_screen_pass(0.25, 0.75, 1.0, 2.0);
    let center = js_sys::Array::from(&dot.get("center"));
    assert_eq!(center.get(0).as_f64(), Some(0.25));
    assert_eq!(center.get(1).as_f64(), Some(0.75));
}
