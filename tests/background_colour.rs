//! `scene.background` and `clear()` colours land on the image as the hex they
//! were written as.
//!
//! [`Color::from_hex`] decodes sRGB to linear, as three.js does with colour
//! management on, and the output encodes back, so an empty scene with a hex
//! background must read back as that hex — on an sRGB target (the hardware
//! encodes), a plain `Unorm` one (the renderer encodes, as a `bgra8unorm`
//! WebGPU canvas needs) and a float one (read back through the encode).
//!
//! The probes include near-black values, where a double decode collapses to
//! zero, and saturated primaries, where a wrong curve shows on one channel and
//! not the others.
//!
//! Skipped when no GPU adapter is available.

use threers::{Color, HeadlessRenderer, PerspectiveCamera, Scene};

const W: u32 = 64;
const H: u32 = 48;

const HEXES: &[u32] = &[
    0x20262c, // the reported dark slate
    0x0a0b0c, // near black, in the linear segment of the curve
    0x010203,
    0x000000,
    0xffffff,
    0x808080,
    0xff0000,
    0x00ff00,
    0x0000ff,
    0xff8800,
    0x3a9a6a,
    0x7f3f1f,
];

const FORMATS: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
];

fn renderer(format: wgpu::TextureFormat, ss: u32) -> Option<HeadlessRenderer> {
    match HeadlessRenderer::builder()
        .size(W, H)
        .supersample(ss)
        .color_format(format)
        .build()
    {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn camera() -> PerspectiveCamera {
    PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 100.0)
}

fn bytes(hex: u32) -> [u8; 3] {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
}

/// Every pixel within ±1 of `hex`, alpha opaque.
fn assert_all(img: &[u8], hex: u32, what: &str) {
    let want = bytes(hex);
    for (i, px) in img.chunks_exact(4).enumerate() {
        let ok = (0..3).all(|c| (px[c] as i32 - want[c] as i32).abs() <= 1) && px[3] == 255;
        assert!(
            ok,
            "{what}: #{hex:06x} came back as {:?} at pixel {i} (wanted {want:?})",
            &px[..4]
        );
    }
}

#[test]
fn background_hex_round_trips_on_every_target_format() {
    for &format in FORMATS {
        let Some(mut hr) = renderer(format, 1) else { return };
        for &hex in HEXES {
            let mut scene = Scene::new();
            scene.background = Color::from_hex(hex);
            let img = hr.render_to_rgba(&mut scene, &camera());
            assert_all(&img, hex, &format!("{format:?} render"));
        }
    }
}

/// The MSAA opaque pass clears a multisampled colour target and resolves it —
/// a different attachment from the one the plain path clears.
#[test]
fn background_hex_round_trips_through_msaa() {
    for &format in FORMATS {
        let Some(mut hr) = renderer(format, 1) else { return };
        hr.set_msaa(4);
        for &hex in HEXES {
            let mut scene = Scene::new();
            scene.background = Color::from_hex(hex);
            let img = hr.render_to_rgba(&mut scene, &camera());
            assert_all(&img, hex, &format!("{format:?} msaa"));
        }
    }
}

/// Supersampled, read through the downsample.
#[test]
fn background_hex_round_trips_through_the_supersample_resolve() {
    let Some(mut hr) = renderer(wgpu::TextureFormat::Rgba8UnormSrgb, 2) else { return };
    for &hex in HEXES {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(hex);
        hr.render(&mut scene, &camera());
        let img = hr.read_rgba_resolved();
        assert_all(&img, hex, "supersample 2");
    }
}

/// `HeadlessRenderer::clear` — the three.js `renderer.clear()` path — agrees
/// with the background clear, whole-target and scissored.
#[test]
fn clear_colour_round_trips_whole_and_scissored() {
    for &format in FORMATS {
        let Some(mut hr) = renderer(format, 1) else { return };
        for &hex in HEXES {
            hr.clear(Color::from_hex(hex), 1.0);
            let img = hr.read_rgba();
            assert_all(&img, hex, &format!("{format:?} clear"));
        }

        // Scissored: the partial clear is a fullscreen triangle writing the
        // blend constant, not a `LoadOp::Clear`.
        hr.clear(Color::from_hex(0x000000), 1.0);
        let r = hr.renderer();
        r.set_scissor_test(true);
        r.set_scissor(0.0, 0.0, W as f32 / 2.0, H as f32);
        hr.clear(Color::from_hex(0x20262c), 1.0);
        let img = hr.read_rgba();
        let r = hr.renderer();
        r.set_scissor_test(false);
        r.reset_scissor();
        let left: Vec<u8> = img
            .chunks_exact(4)
            .enumerate()
            .filter(|(i, _)| (*i as u32 % W) < W / 2)
            .flat_map(|(_, p)| p.to_vec())
            .collect();
        assert_all(&left, 0x20262c, &format!("{format:?} scissored clear"));
    }
}

/// A scissored *render*: the background goes in through the main pass's own
/// partial clear (blend constant), which has to agree with the others.
#[test]
fn scissored_render_background_round_trips() {
    for &format in FORMATS {
        for msaa in [1, 4] {
            let Some(mut hr) = renderer(format, 1) else { return };
            hr.set_msaa(msaa);
            hr.clear(Color::from_hex(0x000000), 1.0);
            let mut scene = Scene::new();
            scene.background = Color::from_hex(0xff8800);
            let r = hr.renderer();
            r.set_scissor_test(true);
            r.set_scissor(W as f32 / 2.0, 0.0, W as f32 / 2.0, H as f32);
            let img = hr.render_to_rgba(&mut scene, &camera());
            let r = hr.renderer();
            r.set_scissor_test(false);
            r.reset_scissor();
            let right: Vec<u8> = img
                .chunks_exact(4)
                .enumerate()
                .filter(|(i, _)| (*i as u32 % W) >= W / 2)
                .flat_map(|(_, p)| p.to_vec())
                .collect();
            assert_all(&right, 0xff8800, &format!("{format:?} msaa {msaa} scissored render"));
        }
    }
}

/// `Color::new` is linear, as `new THREE.Color(r, g, b)` is: linear 0.5 is
/// sRGB byte 188, not 128.
#[test]
fn a_linear_background_is_encoded_on_the_way_out() {
    for &format in FORMATS {
        let Some(mut hr) = renderer(format, 1) else { return };
        let mut scene = Scene::new();
        scene.background = Color::new(0.5, 0.5, 0.5);
        let img = hr.render_to_rgba(&mut scene, &camera());
        assert_all(&img, 0xbcbcbc, &format!("{format:?} linear 0.5"));
    }
}
