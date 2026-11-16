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

/// Warm, bright pixels in the upper middle, where the flame is.
fn flame_pixels(rgba: &[u8], size: u32) -> usize {
    let mut n = 0;
    for y in 0..size / 2 {
        for x in size / 3..2 * size / 3 {
            let i = ((y * size + x) * 4) as usize;
            let (r, g, b) = (rgba[i] as i32, rgba[i + 1] as i32, rgba[i + 2] as i32);
            if r > 120 && r >= g && g > b {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn both_flames_burn_on_metal() {
    let size = 128;
    let mut r = match MetalHeadlessRenderer::builder().size(size, size).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("{e}"),
    };
    for style in [FlameStyle::Simple, FlameStyle::Realistic] {
        let (mut scene, _) = scene(style);
        let rgba = r.render_to_rgba(&mut scene, &camera()).unwrap();
        let n = flame_pixels(&rgba, size);
        if std::env::var_os("CANDLE_PNG").is_some() {
            std::fs::create_dir_all("out").unwrap();
            std::fs::write(format!("out/metal_candle_{style:?}.png"), threers::encode_png(size, size, &rgba)).unwrap();
        }
        assert!(n > 20, "{style:?}: only {n} warm flame pixels");
    }
}
