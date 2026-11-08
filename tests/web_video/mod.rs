//! Video in a real browser, shared by the WebGL2 and WebGPU suites: the
//! browser's own decoded pictures copied into a texture GPU to GPU, and the
//! player decoding with WebCodecs.
//!
//! Needs the `native-decode` feature:
//!
//! ```sh
//! WEBGPU=1 FEATURES=captions,native-decode TEST=webgpu scripts/wasm-browser-test.sh
//! FEATURES=webgl,native-decode TEST=webgl scripts/wasm-browser-test.sh
//! ```

use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

const W: u32 = 64;
const H: u32 = 48;

/// Wait `ms` milliseconds, letting the browser's decoder run.
async fn sleep(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

/// Draw `texture` over the whole of a `W × H` target and read it back.
async fn show(renderer: &mut threers::wasm::WebRenderer, texture: &threers::wasm::WebDataTexture) -> Vec<u8> {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    renderer.set_pixel_ratio(1.0);
    let target = threers::wasm::WebRenderTarget::new(renderer, W, H);
    let mut scene = threers::wasm::WebScene::new();
    let mut material = threers::wasm::WebMaterial::basic(&threers::wasm::WebColor::new(1.0, 1.0, 1.0));
    material.set_map_data(texture);
    scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::plane(2.0, 2.0),
        &material,
    ));
    let mut camera = threers::wasm::WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.set_position(0.0, 0.0, 2.0);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    let promise = renderer.read_render_target_pixels(target.id, 0, 0, W, H);
    let pixels = js_sys::Uint8Array::new(&wasm_bindgen_futures::JsFuture::from(promise).await.unwrap()).to_vec();
    renderer.set_render_target(0);
    pixels
}

/// The pixel at `(x, y)`, top row first, as render targets read back.
fn at(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

fn near(got: [u8; 3], want: [u8; 3], tolerance: u8) -> bool {
    (0..3).all(|c| got[c].abs_diff(want[c]) <= tolerance)
}

/// Linear to sRGB.
fn encode(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// A WebCodecs `VideoFrame` of a canvas painted red over blue.
fn red_over_blue() -> JsValue {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: web_sys::HtmlCanvasElement = document.create_element("canvas").unwrap().dyn_into().unwrap();
    canvas.set_width(W);
    canvas.set_height(H);
    let ctx: web_sys::CanvasRenderingContext2d = canvas.get_context("2d").unwrap().unwrap().dyn_into().unwrap();
    ctx.set_fill_style_str("rgb(220, 30, 30)");
    ctx.fill_rect(0.0, 0.0, W as f64, H as f64 / 2.0);
    ctx.set_fill_style_str("rgb(30, 40, 220)");
    ctx.fill_rect(0.0, H as f64 / 2.0, W as f64, H as f64 / 2.0);
    let init = js_sys::Object::new();
    js_sys::Reflect::set(&init, &"timestamp".into(), &0.into()).unwrap();
    let class = js_sys::Reflect::get(&js_sys::global(), &"VideoFrame".into()).unwrap();
    let args = js_sys::Array::of2(&canvas, &init);
    js_sys::Reflect::construct(class.unchecked_ref::<js_sys::Function>(), &args).expect("a VideoFrame")
}

/// A frame the browser holds goes into the texture on the GPU, the right way
/// up, with nothing read back to the CPU on the way.
#[wasm_bindgen_test]
async fn a_browser_video_frame_is_copied_on_the_gpu() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let texture = threers::wasm::WebDataTexture::new_streamed(W, H);
    assert!(texture.write_video(red_over_blue()), "a VideoFrame is taken");
    assert!(!texture.write_video(JsValue::from_str("not a picture")));
    let out = show(&mut renderer, &texture).await;
    let (top, bottom) = (at(&out, W / 2, H / 4), at(&out, W / 2, 3 * H / 4));
    // `newStreamed` is the shim's texture: linear, and not turned over
    // (`flip_y` off), so it is drawn bottom row first as a three.js
    // DataTexture is, and the shim's UV transform rights it. The canvas's
    // bytes land as they are, exactly as `writeFrame` would put them, and
    // are encoded to sRGB on the way out: 220 shows as 239, 30 as 96.
    let shown = |c: [u8; 3]| c.map(|v| (encode(v as f32 / 255.0) * 255.0).round() as u8);
    assert!(near(bottom, shown([220, 30, 30]), 4), "the canvas's top row, drawn at the bottom: {bottom:?}");
    assert!(near(top, shown([30, 40, 220]), 4), "and its bottom row at the top: {top:?}");
}

/// The player hands the file to the browser's decoder, and what it shows is
/// what the native decoders make of the same file.
#[wasm_bindgen_test]
async fn the_player_decodes_with_webcodecs() {
    // VP9 in WebM: every Chrome decodes it, Chrome for Testing included.
    let file = include_bytes!("../support/video/tone.webm").to_vec();
    let mut want = threers::codec::decode::VideoReader::open_software(file.clone()).unwrap();
    let first = want.next_frame().unwrap().unwrap().1.to_rgba();

    let mut player = threers::wasm::WebVideoPlayer::open(file).unwrap();
    assert_eq!(player.decoder(), "browser", "WebCodecs, not the native decoders");
    let mut waited = 0;
    while player.frames_shown() < 1.0 {
        assert!(waited < 200, "no picture from WebCodecs: {:?}", player.error());
        sleep(25).await;
        player.update(0.0);
        waited += 1;
    }
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let out = show(&mut renderer, &player.texture()).await;
    // The whole picture against the native decoders' picture of the same
    // frame. The browser converts YUV to RGB its own way (its chroma
    // upsampling differs at this tiny test pattern's hard edges), so it is
    // the mean that has to agree, not every pixel.
    let total: u64 = out
        .chunks_exact(4)
        .zip(first.chunks_exact(4))
        .map(|(a, b)| (0..3).map(|c| a[c].abs_diff(b[c]) as u64).sum::<u64>())
        .sum();
    let mean = total as f64 / (W * H * 3) as f64;
    let bias: Vec<f64> = (0..3)
        .map(|c| out.chunks_exact(4).zip(first.chunks_exact(4)).map(|(a, b)| a[c] as f64 - b[c] as f64).sum::<f64>() / (W * H) as f64)
        .collect();
    // Measured: a mean of 8.3 on WebGPU, under 8 on WebGL2, with the bias
    // by channel within 3.5 — edges, not a colour shift. A wrong matrix or
    // range moves the bias by 10 and more.
    assert!(mean < 12.0, "mean difference {mean:.1} from the native decode, bias by channel {bias:?}");
    assert!(bias.iter().all(|b| b.abs() < 5.0), "a colour shift: bias by channel {bias:?}");

    // It plays on: half a second later a later picture is up.
    let shown = player.frames_shown();
    for _ in 0..20 {
        player.update(0.025);
        sleep(25).await;
    }
    assert!(player.frames_shown() > shown, "WebCodecs kept up");
    assert!(player.error().is_none(), "{:?}", player.error());
    // Still the browser's decoder: a refusal would have handed the file to
    // the native decoders without a word.
    assert_eq!(player.decoder(), "browser");
}
