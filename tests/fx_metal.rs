//! `threers::fx` on the Metal renderer — its own port of the surface shader —
//! against the shapes' distance fields and against the wgpu backend.
//!
//! The Metal renderer runs no WGSL, so without the port a surface would draw
//! as its material's fallback: a flat quad over the whole stage. These tests
//! would see that as a frame that disagrees with wgpu everywhere.
//!
//! Skips without a Metal device (or a wgpu adapter, for the comparison).

#![cfg(feature = "metal")]

use threers::fx::{Backdrop, Figure, Gloss, GlossStyle, Look, Shape, Stage, StageInput, Venom, VenomState};
use threers::math::Vector2;
use threers::metal::MetalHeadlessRenderer;
use threers::overlay::Overlay;
use threers::prelude::*;

const W: u32 = 360;
const H: u32 = 240;

fn pill() -> Shape {
    Shape::pill(Vector2::new(110.0, 150.0), 150.0, 48.0)
}
fn star() -> Shape {
    Shape::star(Vector2::new(270.0, 90.0), 5, 55.0, 24.0, 4.0)
}
fn ball() -> Shape {
    Shape::circle(Vector2::new(280.0, 190.0), 28.0)
}

/// A stage with a venom pill caught holding a pointer, a gloss star and a
/// chrome ball, run for a second so the strands are out.
fn scene() -> (Scene, Stage) {
    let mut scene = Scene::new();
    let mut stage = Stage::new();
    stage.set_backdrop(&mut scene, Backdrop::STUDIO);
    let venom = stage.add(&mut scene, Figure::new(pill()), Venom::default());
    stage.add(&mut scene, Figure::new(star()), Gloss::new(GlossStyle::default().with_look(Look::GOLD)));
    stage.add(&mut scene, Figure::new(ball()), Gloss::new(GlossStyle::default().with_look(Look::CHROME)));
    let mut overlay = Overlay::new(W, H);
    let input = StageInput { viewport: Vector2::new(W as f32, H as f32), dpr: 1.0, dt: 1.0 / 60.0, pointer: None, pointer_down: false };
    let hand = Vector2::new(150.0, 60.0);
    for _ in 0..60 {
        overlay.clear();
        stage.update_each(&mut scene, &mut overlay, input, &|id| (id == venom).then_some(hand));
    }
    assert_eq!(stage.effect::<Venom>(venom).unwrap().state(), VenomState::Holding);
    (scene, stage)
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 2.0);
    c.look_at(Vector3::ZERO);
    c
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
}

/// Pixels well inside each shape are its surface, and well outside every
/// shape (and the strands' reach) the backdrop.
fn check_against_the_fields(name: &str, px: &[u8]) {
    let (mut inside, mut inside_dark, mut outside, mut outside_pale) = (0, 0, 0, 0);
    for y in 0..H {
        for x in 0..W {
            let p = Vector2::new(x as f32 + 0.5, y as f32 + 0.5);
            let d_pill = pill().distance(p);
            let nearest = d_pill.min(star().distance(p)).min(ball().distance(p));
            if d_pill < -4.0 {
                inside += 1;
                inside_dark += (luma(px, x, y) < 60.0) as u32;
            } else if nearest > 45.0 && !(x > 100 && y < 150) {
                // clear of every shape, its shadow, and the strands up to the hand
                outside += 1;
                outside_pale += (luma(px, x, y) > 200.0) as u32;
            }
        }
    }
    let (a, b) = (inside_dark as f32 / inside as f32, outside_pale as f32 / outside as f32);
    eprintln!("{name}: lacquer inside the pill dark {a:.3}, backdrop pale {b:.3}");
    assert!(a > 0.97, "{name}: the pill's inside is not lacquer ({a})");
    assert!(b > 0.97, "{name}: the backdrop is not showing around the shapes ({b})");
    // the gold star is warm, the chrome ball is grey
    let i = ((90 * W + 270) * 4) as usize;
    assert!(px[i] as i32 - px[i + 2] as i32 > 30, "{name}: the star is not gold: {:?}", &px[i..i + 3]);
}

#[test]
fn surfaces_draw_on_metal() {
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(W, H).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let (mut scene, _stage) = scene();
    let px = metal.render_to_rgba(&mut scene, &camera()).unwrap();
    check_against_the_fields("metal", &px);
}

#[test]
fn metal_matches_wgpu() {
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(W, H).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = threers::renderer::headless::HeadlessRenderer::builder().size(W, H).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let (mut scene, _stage) = scene();
    let m = metal.render_to_rgba(&mut scene, &camera()).unwrap();
    let w = hr.render_to_rgba(&mut scene, &camera());
    check_against_the_fields("wgpu", &w);
    if let Ok(dir) = std::env::var("FX_METAL_DUMP") {
        std::fs::write(format!("{dir}/fx_metal.png"), threers::utils::png::encode_png(W, H, &m)).unwrap();
        std::fs::write(format!("{dir}/fx_wgpu.png"), threers::utils::png::encode_png(W, H, &w)).unwrap();
    }
    let (mut sum, mut big) = (0.0f64, 0usize);
    for (a, b) in m.chunks_exact(4).zip(w.chunks_exact(4)) {
        let d = (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap();
        sum += d as f64;
        big += (d > 24) as usize;
    }
    let mean = sum / (W * H) as f64;
    eprintln!("metal vs wgpu: mean {mean:.3} levels, {big} pixels over 24");
    // Anti-aliased edges and the blend's space can differ by a level or two;
    // a wrong surface differs by tens over thousands of pixels.
    assert!(mean < 1.5, "mean difference {mean:.2} levels");
    assert!(big < (W * H) as usize / 200, "{big} pixels differ by more than 24 levels");
}
