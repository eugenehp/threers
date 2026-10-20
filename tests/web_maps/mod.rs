//! Material maps in a real browser, shared by the WebGL2 suite
//! (`tests/webgl.rs`) and the WebGPU one (`tests/webgpu.rs`).
//!
//! A material whose maps disagree on filter, wrap or uv transform is drawn
//! with a shader variant (`renderer::material_maps`) that declares each map
//! with its own sampler and transform. That WGSL is generated, so the only
//! place to learn whether ANGLE's GLSL translation and Tint accept it is here.
//! The texture samplers — one for every filter and wrap a texture can ask
//! for, `MirroredRepeat` among them — likewise.
//!
//! The quad fills an orthographic view, so a pixel's x is its u.
//!
//! Each including file supplies `super::views_renderer()`.

use threers::wasm::{
    WebCamera, WebColor, WebDataTexture, WebGeometry, WebLight, WebMaterial, WebMesh,
    WebRenderTarget, WebRenderer, WebScene,
};
use wasm_bindgen_test::wasm_bindgen_test;

const N: u32 = 64;

async fn read_target(renderer: &WebRenderer, id: u32) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(id, 0, 0, N, N);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
}

/// three.js's `NearestFilter`, `LinearFilter`,
/// `ClampToEdgeWrapping` and `MirroredRepeatWrapping`.
const NEAREST: u32 = 1003;
const LINEAR: u32 = 1006;
const CLAMP: u32 = 1001;
const MIRROR: u32 = 1002;

/// A 2×1 nearest-filtered strip, `left | right`, with `wrap` on both axes.
fn strip(left: [u8; 4], right: [u8; 4], wrap: u32) -> WebDataTexture {
    let mut t = WebDataTexture::new(2, 1, [left, right].concat());
    t.set_filters(NEAREST, NEAREST, wrap, wrap);
    t
}

/// A white Lambert quad filling the view under ambient light at π, which
/// shades it to exactly its map (`BRDF_Lambert` is albedo / π), as the pixel
/// row through its middle.
async fn draw(renderer: &mut WebRenderer, material: &WebMaterial) -> Vec<[u8; 3]> {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add_light(&WebLight::ambient(
        &WebColor::new(1.0, 1.0, 1.0),
        std::f32::consts::PI,
    ));
    scene.add(&WebMesh::new(&WebGeometry::plane(2.0, 2.0), material));
    let mut camera = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);
    let target = WebRenderTarget::new(renderer, N, N);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(renderer, target.id).await;
    renderer.set_render_target(0);
    (0..N)
        .map(|x| {
            let i = ((N / 2 * N + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2]]
        })
        .collect()
}

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];
const WHITE: [u8; 4] = [255, 255, 255, 255];

/// A colour map and an emissive map with different uv transforms — the
/// emissive one mirrored (`repeat.x = −1`, `offset.x = 1`) — each read through
/// its own: white on the left (red + white), green on the right (green +
/// black). One transform shared between them would give red, then white.
#[wasm_bindgen_test]
async fn each_map_keeps_its_own_uv_transform_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let mut emissive = strip(BLACK, WHITE, CLAMP);
    emissive.set_uv_transform(1.0, 0.0, -1.0, 1.0);
    let mut material = WebMaterial::lambert(&WebColor::new(1.0, 1.0, 1.0));
    material.set_map_data(&strip(RED, GREEN, CLAMP));
    material.set_emissive(&WebColor::new(1.0, 1.0, 1.0));
    material.set_emissive_map_data(&emissive);
    let row = draw(&mut renderer, &material).await;
    assert_eq!(
        row[N as usize / 4],
        [255, 255, 255],
        "left: red map + mirrored emissive white"
    );
    assert_eq!(
        row[N as usize * 3 / 4],
        [0, 255, 0],
        "right: green map + mirrored emissive black"
    );
}

/// `MirroredRepeatWrapping` with `repeat.x = 2`: red | green, then mirrored
/// green | red — where `RepeatWrapping` would give red | green twice.
#[wasm_bindgen_test]
async fn mirrored_repeat_wrapping_mirrors_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let mut map = strip(RED, GREEN, MIRROR);
    map.set_uv_transform(0.0, 0.0, 2.0, 1.0);
    let mut material = WebMaterial::lambert(&WebColor::new(1.0, 1.0, 1.0));
    material.set_map_data(&map);
    let row = draw(&mut renderer, &material).await;
    let quarters: Vec<[u8; 3]> = [1, 3, 5, 7]
        .iter()
        .map(|k| row[(N as usize * k) / 8])
        .collect();
    // Green alone, which no red/blue swizzle moves.
    let greens: Vec<u8> = quarters.iter().map(|p| p[1]).collect();
    assert_eq!(
        greens,
        vec![0, 255, 255, 0],
        "mirrored repeat: red, green | green, red; got {quarters:?}"
    );
}

/// three.js `alphaMap` — a role only a shader variant binds — under
/// `alphaTest` 0.5: green 0 on the left cuts the quad to the black
/// background, green 255 on the right keeps it white. Its red channel is the
/// other way round, so reading the wrong channel fails too.
#[wasm_bindgen_test]
async fn alpha_map_cuts_by_its_green_channel_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let mut material = WebMaterial::lambert(&WebColor::new(1.0, 1.0, 1.0));
    material.set_alpha_map_data(&strip([255, 0, 255, 255], [0, 255, 0, 255], CLAMP));
    material.set_alpha_test(0.5);
    let row = draw(&mut renderer, &material).await;
    assert_eq!(row[N as usize / 4], [0, 0, 0], "left: cut by green 0");
    assert_eq!(
        row[N as usize * 3 / 4],
        [255, 255, 255],
        "right: kept by green 255"
    );
}

/// A `bumpMap` ramp tilting the normal 45° toward −x, under a light from the
/// left at 45°, meets it head on: white. Without the bump (or with its slope
/// the wrong way) the quad is grey or black. The bump takes screen-space
/// derivatives, which Tint only allows in uniform control flow — a check naga
/// does not make. And a light map at intensity π, with no light, shades to
/// exactly its texel.
#[wasm_bindgen_test]
async fn bump_and_light_maps_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    // Heights 0 | 1, linear: 2 per unit of u, and u moves 1/64 a pixel here,
    // so a scale of 32 steps the height by 1 a pixel.
    let mut ramp = WebDataTexture::new(2, 1, vec![0, 0, 0, 255, 255, 255, 255, 255]);
    ramp.set_filters(LINEAR, LINEAR, CLAMP, CLAMP);
    let mut bumped = WebMaterial::lambert(&WebColor::new(1.0, 1.0, 1.0));
    bumped.set_bump_map_data(&ramp);
    bumped.set_bump_scale(N as f32 / 2.0);
    let mut sun = WebLight::directional(&WebColor::new(1.0, 1.0, 1.0), std::f32::consts::PI);
    sun.set_direction(1.0, 0.0, -1.0);
    let row = draw_lit(&mut renderer, &bumped, Some(sun)).await;
    assert_eq!(row[N as usize / 2], [255, 255, 255], "the bumped normal faces the light");

    let mut lit = WebMaterial::lambert(&WebColor::new(1.0, 1.0, 1.0));
    lit.set_light_map_data(&WebDataTexture::new(1, 1, vec![0, 255, 0, 255]));
    lit.set_light_map_intensity(std::f32::consts::PI);
    let row = draw_lit(&mut renderer, &lit, None).await;
    assert_eq!(row[N as usize / 2], [0, 255, 0], "the light map's green, exactly");
}

/// [`draw`] with one light (or none) in place of the ambient light.
async fn draw_lit(renderer: &mut WebRenderer, material: &WebMaterial, light: Option<WebLight>) -> Vec<[u8; 3]> {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    if let Some(l) = light {
        scene.add_light(&l);
    }
    scene.add(&WebMesh::new(&WebGeometry::plane(2.0, 2.0), material));
    let mut camera = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.set_position(0.0, 0.0, 5.0);
    camera.look_at(0.0, 0.0, 0.0);
    let target = WebRenderTarget::new(renderer, N, N);
    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(renderer, target.id).await;
    renderer.set_render_target(0);
    (0..N)
        .map(|x| {
            let i = ((N / 2 * N + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2]]
        })
        .collect()
}

/// three.js `envMap` on a Basic quad facing an orthographic camera, `Mix`
/// combine: the reflected view ray is +z, so the quad shows the +Z face
/// (magenta); with `envMapRotation.y` a quarter turn it shows +X (red).
#[wasm_bindgen_test]
async fn env_maps_in_the_browser() {
    use threers::wasm::WebCubeTexture;
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let face = |c: [u8; 3]| [c[0], c[1], c[2], 255].repeat(16);
    let cube = WebCubeTexture::new(
        4,
        face([255, 0, 0]),
        face([0, 255, 0]),
        face([0, 0, 255]),
        face([255, 255, 0]),
        face([255, 0, 255]),
        face([0, 255, 255]),
    );
    let mut material = WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0));
    assert!(material.set_env_map(&cube), "Basic takes an envMap");
    material.set_combine(1);
    let row = draw_lit(&mut renderer, &material, None).await;
    assert_eq!(row[N as usize / 2], [255, 0, 255], "head-on: the +Z face");
    material.set_env_map_rotation(0.0, std::f32::consts::FRAC_PI_2, 0.0, "XYZ");
    let row = draw_lit(&mut renderer, &material, None).await;
    assert_eq!(row[N as usize / 2], [255, 0, 0], "turned a quarter about y: the +X face");
}

/// A MeshPhysicalMaterial variant — sheen, clearcoat with its own normal map,
/// iridescence, anisotropy and specular maps — compiles in the browser's
/// shader compiler and draws. Its texture reads sit in branches on uniform
/// material state, which Tint's uniformity analysis checks more strictly than
/// naga does; a module it rejects draws nothing at all.
#[wasm_bindgen_test]
async fn physical_maps_compile_and_draw_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let mut material = WebMaterial::physical(&WebColor::new(0.6, 0.6, 0.6), 0.4, 0.0, 1.0, 0.2);
    material.set_physical(1.0, 0.2, 1.5, 1.0, 1.0, 0.5, 1.0, 1.3, 100.0, 400.0, 0.5, 0.3, 0.0, 0.0, f32::INFINITY, 0.0);
    let grey = || strip([200, 200, 200, 255], [200, 200, 200, 255], 0);
    for name in ["sheenColorMap", "clearcoatNormalMap", "iridescenceMap", "anisotropyMap", "specularIntensityMap"] {
        assert!(material.set_named_map_data(name, &grey()), "{name} is a Physical slot");
    }
    let row = draw(&mut renderer, &material).await;
    let centre = row[N as usize / 2];
    assert!(centre.iter().any(|&c| c > 20), "the physical variant drew nothing: {centre:?}");
}

/// MeshToonMaterial's `gradientMap` in the browser: a light of intensity π
/// straight on reads the ramp at `coord = 1`, its last texel — here green —
/// so the white toon quad draws pure green; without the map, white. The ramp
/// is sampled with explicit derivatives inside the light loop, which ANGLE's
/// GLSL and Tint must both accept.
#[wasm_bindgen_test]
async fn toon_gradient_map_in_the_browser() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut renderer = shared.borrow_mut();
    let light = || {
        let mut l = WebLight::directional(&WebColor::new(1.0, 1.0, 1.0), std::f32::consts::PI);
        l.set_direction(0.0, 0.0, -1.0);
        l
    };
    let mut material = WebMaterial::toon(&WebColor::new(1.0, 1.0, 1.0));
    let plain = draw_lit(&mut renderer, &material, Some(light())).await;
    assert_eq!(plain[N as usize / 2], [255, 255, 255], "no gradientMap: all of the light");
    assert!(material.set_named_map_data("gradientMap", &strip(RED, GREEN, CLAMP)), "Toon has a gradientMap");
    let row = draw_lit(&mut renderer, &material, Some(light())).await;
    assert_eq!(row[N as usize / 2], [0, 255, 0], "the ramp's last texel");
}
