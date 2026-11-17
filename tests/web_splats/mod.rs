//! Gaussian splats (`threers::splats`) in a real browser, shared by the WebGL2
//! suite (`tests/webgl.rs`) and the WebGPU one (`tests/webgpu.rs`), through
//! the JavaScript API a page uses: `WebScene::addSplats` on a file's bytes,
//! and a `render` that sorts them.
//!
//! On WebGL2 the splat data comes from textures, not storage buffers, and the
//! shader goes through naga's GLSL and ANGLE; on WebGPU through Tint. A shader
//! either rejects draws nothing, without an error — so every check is on
//! read-back pixels.
//!
//! Each including file supplies `super::views_renderer()`.

use threers::math::Vector3;
use threers::splats::GaussianSplats;
use wasm_bindgen_test::wasm_bindgen_test;

const N: u32 = 64;

fn reset(renderer: &mut threers::wasm::WebRenderer) {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    renderer.set_pixel_ratio(1.0);
}

async fn read_target(renderer: &threers::wasm::WebRenderer, target_id: u32) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(target_id, 0, 0, N, N);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
}

/// `bytes` (a file's) drawn over black from `eye`, looking at the origin.
async fn draw(bytes: &[u8], eye: [f32; 3]) -> Vec<u8> {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    reset(&mut renderer);
    let target = threers::wasm::WebRenderTarget::new(&renderer, N, N);
    let mut scene = threers::wasm::WebScene::new();
    scene.set_background(&threers::wasm::WebColor::new(0.0, 0.0, 0.0));
    scene.add_splats(bytes, None).expect("the splats load");
    let mut camera = threers::wasm::WebCamera::perspective(50.0, 1.0, 0.1, 100.0);
    camera.set_position(eye[0], eye[1], eye[2]);
    camera.look_at(0.0, 0.0, 0.0);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(&renderer, target.id).await;
    assert_eq!(px.len(), (N * N * 4) as usize, "short read-back");
    px
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    px[((y * N + x) * 4) as usize..][..4].try_into().unwrap()
}

/// One splat, from a `.splat` file: its colour at the centre, falling off as
/// a Gaussian — bright at the centre, dimmer one standard deviation out,
/// nothing at four.
#[wasm_bindgen_test]
async fn a_splat_file_draws_its_gaussian() {
    let mut s = GaussianSplats::default();
    s.push(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 1.0, [1.0, 0.6, 0.2]);
    let px = draw(&s.to_splat().unwrap(), [0.0, 0.0, 5.0]).await;
    // One standard deviation in pixels at 5 units: 0.3 · (N/2) / (5 tan 25°).
    let sigma = 0.3 * (N as f32 / 2.0) / (5.0 * 25f32.to_radians().tan());
    let c = N / 2;
    let centre = at(&px, c, c);
    let one = at(&px, c + sigma.round() as u32, c);
    let four = at(&px, c + (4.0 * sigma) as u32, c);
    // 255 · 0.99 · (1, 0.6, 0.2), within a few levels of the blend.
    let want = [252.0, 151.0, 50.0];
    for ch in 0..3 {
        assert!(
            (centre[ch] as f32 - want[ch]).abs() < 8.0,
            "centre {centre:?}, expected about {want:?}"
        );
    }
    assert!(
        one[0] > 100 && one[0] < 230 && one[0] < centre[0],
        "one sigma out {one:?} against centre {centre:?}"
    );
    assert!(four[0] < 12, "four sigma out {four:?}");
}

/// Two nearly opaque splats, one behind the other: the nearer covers the
/// farther from either side — so the render sorted them for its camera.
#[wasm_bindgen_test]
async fn the_render_sorts_the_splats_for_its_camera() {
    let mut s = GaussianSplats::default();
    s.push(Vector3::new(0.0, 0.0, 0.5), [0.5, 0.5, 0.02], [0.0, 0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 0.0]);
    s.push(Vector3::new(0.0, 0.0, -0.5), [0.5, 0.5, 0.02], [0.0, 0.0, 0.0, 1.0], 1.0, [0.0, 0.0, 1.0]);
    let bytes = s.to_ply().unwrap();
    let c = N / 2;
    let p = at(&draw(&bytes, [0.0, 0.0, 5.0]).await, c, c);
    assert!(p[0] > 200 && p[2] < 40, "from +z red is nearer: {p:?}");
    let p = at(&draw(&bytes, [0.0, 0.0, -5.0]).await, c, c);
    assert!(p[2] > 200 && p[0] < 40, "from -z blue is nearer: {p:?}");
}

/// Degree-3 view-dependent colour from a PLY, seen from three sides,
/// against `color_towards` on the CPU: the packed bands are read back right
/// from storage (WebGPU) or a texture (WebGL2).
#[wasm_bindgen_test]
async fn view_dependent_colour_matches_the_cpu() {
    let mut s = GaussianSplats { sh_degree: 3, ..Default::default() };
    s.push(Vector3::ZERO, [0.4; 3], [0.0, 0.0, 0.0, 1.0], 1.0, [0.5, 0.45, 0.55]);
    let n = s.sh.len();
    s.sh = (0..n).map(|k| ((k as f32) * 1.3).sin() * 0.12).collect();
    let bytes = s.to_ply().unwrap();
    let c = N / 2;
    for eye in [[0.0, 0.0, 5.0], [5.0, 0.0, 0.0], [-3.0, 3.0, 2.0]] {
        let p = at(&draw(&bytes, eye).await, c, c);
        let want = s.color_towards(0, -Vector3::new(eye[0], eye[1], eye[2]));
        for ch in 0..3 {
            let got = p[ch] as f32 / 255.0 / 0.99;
            let w = want[ch].clamp(0.0, 1.0);
            assert!(
                (got - w).abs() < 0.04,
                "eye {eye:?} channel {ch}: {got} against {w} ({p:?})"
            );
        }
    }
}

/// `.spz` in wasm: version 3 through the crate's inflate, version 4 through
/// zstd, each decoding to what splat-transform's fixtures hold (see
/// `tests/splats.rs`), and drawn: from `-z` the splat at the origin is in
/// front, colour `(2/3, 2/3, 1/2)` at opacity 0.9, over one coloured
/// `(2/3, 1/3, 1/2)` — green near red says the front one is on top.
#[wasm_bindgen_test]
async fn spz_files_load_and_draw() {
    for bytes in [
        &include_bytes!("../splat_fixtures/minimal-v3.spz")[..],
        &include_bytes!("../splat_fixtures/minimal-v4.spz")[..],
    ] {
        let s = GaussianSplats::from_bytes(bytes).expect("the .spz decodes");
        assert_eq!(s.len(), 4);
        assert!(s.positions[3].iter().all(|v| v.abs() < 1e-3), "{:?}", s.positions);
        assert!((s.opacities[3] - 0.9).abs() < 0.003);
        let px = draw(bytes, [0.0, 0.0, -5.0]).await;
        let c = at(&px, N / 2, N / 2);
        assert!(
            c[0] > 120 && c[1] as f32 > 0.8 * c[0] as f32 && c[2] < c[1],
            "centre {c:?}: expected mostly the origin splat's (2/3, 2/3, 1/2)"
        );
    }
}
