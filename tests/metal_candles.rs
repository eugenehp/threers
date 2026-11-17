//! Candles on the Metal backend: the flame's WGSL `ShaderMaterial` has to
//! translate, or it draws a flat fallback colour and still "works".
#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use threers::extras::candle::{Candle, FlameStyle};
use threers::metal::{MetalError, MetalHeadlessRenderer};
use threers::prelude::*;

fn scene(style: FlameStyle) -> (Scene, threers::extras::candle::CandleHandle) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x050403);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.05));
    let candle = Candle::new(style).with_seed(1.0).add_to(&mut scene, Vector3::new(0.0, 0.0, 0.0));
    candle.update(&mut scene, 2.3);
    (scene, candle)
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(30.0, 1.0, 0.01, 10.0);
    c.position = Vector3::new(0.0, 0.16, 0.42);
    c.look_at(Vector3::new(0.0, 0.13, 0.0));
    c
}

/// Warm, bright pixels in the upper middle, where the flame is — and not
/// white: a flame that renders as a saturated box has the right brightness
/// and none of the gradient.
fn flame_pixels(rgba: &[u8], size: u32) -> (usize, usize) {
    let (mut warm, mut white) = (0, 0);
    for y in 0..size / 2 {
        for x in size / 3..2 * size / 3 {
            let i = ((y * size + x) * 4) as usize;
            let (r, g, b) = (rgba[i] as i32, rgba[i + 1] as i32, rgba[i + 2] as i32);
            if r > 120 && r >= g && g > b + 20 {
                warm += 1;
            }
            if r > 245 && g > 245 && b > 245 {
                white += 1;
            }
        }
    }
    (warm, white)
}

/// Both flames burn on Metal, and look as they do on wgpu: the flame's WGSL
/// goes through the material translator, which is where a reserved name or
/// a split `++` turns a flame into a fallback colour or a white box.
#[test]
fn both_flames_burn_on_metal_as_on_wgpu() {
    let size = 128;
    let mut metal = match MetalHeadlessRenderer::builder().size(size, size).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let mut wgpu = HeadlessRenderer::builder().size(size, size).build().ok();
    for style in [FlameStyle::Simple, FlameStyle::Realistic] {
        let (mut scene, _) = scene(style);
        let m = metal.render_to_rgba(&mut scene, &camera()).unwrap();
        let (warm, white) = flame_pixels(&m, size);
        assert!(warm > 20, "{style:?}: only {warm} warm flame pixels");
        assert!(white < warm, "{style:?}: {white} white pixels against {warm} warm: a box, not a flame");
        if let Some(r) = wgpu.as_mut() {
            let (mut scene, _) = self::scene(style);
            let w = r.render_to_rgba(&mut scene, &camera());
            let mean = m.iter().zip(&w).map(|(a, b)| (*a as i32 - *b as i32).abs() as f64).sum::<f64>()
                / m.len() as f64;
            if std::env::var_os("CANDLE_PNG").is_some() {
                std::fs::create_dir_all("out").unwrap();
                std::fs::write(format!("out/metal_candle_{style:?}.png"), threers::encode_png(size, size, &m)).unwrap();
                std::fs::write(format!("out/wgpu_candle_{style:?}.png"), threers::encode_png(size, size, &w)).unwrap();
            }
            // The realistic flame jitters its march start per pixel; the two
            // backends' hashes agree, their rounding a little less.
            assert!(mean < 3.0, "{style:?}: Metal and wgpu differ by {mean:.2}/255 on average");
        }
    }
}

/// A flame given a colour burns in it — blue here, which the black body never
/// is above its base — on both backends, the light with it.
#[test]
fn a_tinted_flame_burns_in_its_colour() {
    let size = 128;
    let blue = |rgba: &[u8]| {
        let mut n = 0;
        for y in 0..size / 2 {
            for x in size / 3..2 * size / 3 {
                let i = ((y * size + x) * 4) as usize;
                let (r, b) = (rgba[i] as i32, rgba[i + 2] as i32);
                if b > 100 && b > r + 30 {
                    n += 1;
                }
            }
        }
        n
    };
    let mut metal = match MetalHeadlessRenderer::builder().size(size, size).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    let mut wgpu = HeadlessRenderer::builder().size(size, size).build().ok();
    for style in [FlameStyle::Simple, FlameStyle::Realistic] {
        let (mut scene, candle) = scene(style);
        candle.set_flame_color(&mut scene, Color::new(0.2, 0.5, 1.0), 1.0);
        let m = metal.render_to_rgba(&mut scene, &camera()).unwrap();
        assert!(blue(&m) > 20, "{style:?} on Metal: only {} blue flame pixels", blue(&m));
        if std::env::var_os("CANDLE_PNG").is_some() {
            std::fs::write(format!("out/metal_candle_{style:?}_blue.png"), threers::encode_png(size, size, &m)).unwrap();
        }
        if let Some(r) = wgpu.as_mut() {
            let w = r.render_to_rgba(&mut scene, &camera());
            assert!(blue(&w) > 20, "{style:?} on wgpu: only {} blue flame pixels", blue(&w));
        }
    }
}
