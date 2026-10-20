//! The renderer on **WebGPU**, in a real browser.
//!
//! The WebGL2 suite (`tests/webgl.rs`) hides `navigator.gpu` so the renderer
//! starts on WebGL2. This one needs the opposite, and runs the same shared
//! modules (`web_views`, `web_postfx`, `web_maps`, `web_scenes`), so the two
//! backends are held to the same checks. There is a reason to want it beyond
//! coverage: naga, which validates every WGSL module here natively, accepts
//! WGSL that Tint — Chrome's compiler — rejects, and a rejected module fails
//! its pipelines silently (a black canvas at a full frame rate). A WebGPU run
//! in real Chrome is the only test of the WGSL as a browser sees it.
//!
//! ```sh
//! WEBGPU=1 FEATURES=captions TEST=webgpu scripts/wasm-browser-test.sh
//! ```
//!
//! `WEBGPU=1` launches Chrome with `--enable-unsafe-webgpu` (Metal on macOS).
//! Every test fails — rather than silently passing on WebGL2 — if the browser
//! hands over anything but a WebGPU adapter.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[path = "web_views/mod.rs"]
mod web_views;

// Post-fx onto the canvas, right way up — shared with the other backend.
#[path = "web_postfx/mod.rs"]
mod web_postfx;

// Material maps: shader variants and every sampler — shared with the other
// backend.
#[path = "web_maps/mod.rs"]
mod web_maps;

// Whole scenes: lit, skinned, refracting, shadowed and post-processed draws,
// ShaderMaterial, morph targets and groups — shared with the other backend.
#[path = "web_scenes/mod.rs"]
mod web_scenes;

type SharedViews = std::rc::Rc<std::cell::RefCell<threers::wasm::WebRenderer>>;

thread_local! {
    static VIEWS: std::cell::RefCell<Option<(SharedViews, web_sys::HtmlCanvasElement)>> =
        const { std::cell::RefCell::new(None) };
}

/// The one WebGPU renderer the suite shares, and its canvas.
async fn views_renderer() -> (SharedViews, web_sys::HtmlCanvasElement) {
    // A test that panicked never dropped its borrow (wasm panics do not
    // unwind), so after a failure the next test gets a fresh renderer rather
    // than a cascade of "already borrowed".
    if let Some(v) = VIEWS.with(|v| v.borrow().clone()) {
        if v.0.try_borrow_mut().is_ok() {
            return v;
        }
    }
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(128);
    canvas.set_height(96);
    document.body().unwrap().append_child(&canvas).unwrap();
    let renderer = threers::wasm::WebRenderer::new(canvas.clone())
        .await
        .expect("no renderer — is WebGPU enabled in this browser (WEBGPU=1)?");
    assert_eq!(
        renderer.backend(),
        "webgpu",
        "expected a WebGPU adapter; run with WEBGPU=1"
    );
    let v = (std::rc::Rc::new(std::cell::RefCell::new(renderer)), canvas);
    VIEWS.with(|slot| *slot.borrow_mut() = Some(v.clone()));
    v
}

/// The full renderer builds every pipeline and draws a lit frame on WebGPU —
/// the check that the WGSL survives Tint, which the shared pixel tests then
/// extend to every other pipeline. And WebGPU offers the compute stage WebGL2
/// lacks (`tests/webgl.rs` checks the opposite), so the compute-only passes run.
#[wasm_bindgen_test]
async fn the_renderer_draws_a_lit_frame_on_webgpu() {
    let (shared, _canvas) = views_renderer().await;
    let mut renderer = shared.borrow_mut();
    assert!(renderer.supports_compute(), "WebGPU reported no compute stage");
    let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
    let mut scene = threers::wasm::WebScene::new();
    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    let material = threers::wasm::WebMaterial::standard(
        &threers::wasm::WebColor::new(0.9, 0.3, 0.1),
        0.4,
        0.0,
    );
    scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(1.5, 1.5, 1.5),
        &material,
    ));
    let mut sun = threers::wasm::WebLight::directional(&white, 2.5);
    sun.set_direction(-0.4, -0.7, -0.6);
    scene.add_light(&sun);
    scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.4));
    let mut camera = threers::wasm::WebCamera::perspective(60.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.0, 4.0);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let promise = renderer.read_render_target_pixels(target.id, 0, 0, 64, 64);
    renderer.set_render_target(0);
    drop(renderer);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    let pixels = js_sys::Uint8Array::new(&value).to_vec();
    let lit = pixels.chunks(4).filter(|p| p[0] > 40 && p[0] > p[2] + 60).count();
    assert!(
        lit > 50,
        "only {lit} of {} pixels show the lit cube",
        pixels.len() / 4
    );
}
