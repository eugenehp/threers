//! The GPU alpha strip has to produce exactly what the CPU path produces.
//!
//! This is a shortcut on the hot path — a compute pass whose output goes
//! straight into a video file — so the thing worth testing is not that it runs
//! but that it is byte-for-byte the RGBA readback with the fourth byte removed.
//! A packing bug here does not crash; it shifts every pixel by one byte and
//! tints the whole film, which is exactly the kind of fault that survives a
//! glance at a thumbnail.
//!
//! Sizes are chosen to exercise the parts that are easy to get wrong: a pixel
//! count that is not a multiple of the four-pixel group, and a row width whose
//! RGBA stride is not 256-byte aligned, so the two paths pad differently.

use threers::prelude::*;
use threers::HeadlessRenderer;

/// A scene with enough colour variation that a channel swap or a one-byte
/// shift cannot hide in it.
fn scene_with_colour() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.05, 0.10, 0.35);

    for (i, c) in [
        Color::new(1.0, 0.0, 0.0),
        Color::new(0.0, 1.0, 0.0),
        Color::new(0.0, 0.0, 1.0),
        Color::new(0.9, 0.7, 0.1),
    ]
    .into_iter()
    .enumerate()
    {
        let g = BoxGeometry::new(0.7, 0.7, 0.7);
        let mut m = StandardMaterial::new(c);
        m.roughness = 0.6;
        let mut o = Object3D::mesh(Mesh::new(g, m.into()));
        o.position = Vector3::new(i as f32 * 0.9 - 1.35, 0.0, 0.0);
        scene.add(o);
    }
    scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 1.2)
            .with_direction(Vector3::new(-0.4, -0.5, -0.8).normalize()),
    );

    let mut cam = PerspectiveCamera::new(45.0, 1.6, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.8, 4.0);
    cam.look_at(Vector3::ZERO);
    (scene, cam)
}

fn render_both(w: u32, h: u32) -> Option<(Vec<u8>, Vec<u8>)> {
    let (mut scene, cam) = scene_with_colour();

    let mut a = HeadlessRenderer::builder().size(w, h).build().ok()?;
    assert!(
        a.rgb_readback_supported(),
        "an 8-bit colour target must be packable"
    );
    // Pipelined readbacks hand back the PREVIOUS frame, so render twice and
    // take the one that comes out; `finish_readback` would do as well.
    a.render(&mut scene, &cam);
    let _ = a.read_rgba_resolved_pipelined();
    a.render(&mut scene, &cam);
    let rgba = a.read_rgba_resolved_pipelined()?;

    let mut b = HeadlessRenderer::builder().size(w, h).build().ok()?;
    b.render(&mut scene, &cam);
    let _ = b.read_rgb_resolved_pipelined();
    b.render(&mut scene, &cam);
    let rgb = b.read_rgb_resolved_pipelined()?;

    Some((rgba, rgb))
}

#[test]
fn packs_exactly_the_rgba_frame_without_its_alpha() {
    // 259 x 101 = 26159 pixels: not a multiple of 4, so the last workgroup
    // writes past the end and the trim has to be right. 259 * 4 = 1036 bytes a
    // row, which is not 256-aligned either, so the RGBA readback takes its
    // row-by-row unpadding path while this one is a flat copy.
    let Some((rgba, rgb)) = render_both(259, 101) else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    let px = 259 * 101;
    assert_eq!(rgba.len(), px * 4);
    assert_eq!(rgb.len(), px * 3, "three bytes a pixel, no padding");

    let mut mismatched = 0;
    for (i, (a, b)) in rgba.chunks_exact(4).zip(rgb.chunks_exact(3)).enumerate() {
        if a[..3] != *b {
            if mismatched < 4 {
                eprintln!("pixel {i}: rgba {:?} vs rgb {:?}", &a[..3], b);
            }
            mismatched += 1;
        }
    }
    assert_eq!(mismatched, 0, "{mismatched} of {px} pixels differ");
}

/// The failure this is really guarding against: a pack that is off by one byte,
/// or that writes BGR. Both leave the length right and the histogram similar,
/// so they pass anything that only checks "is there an image".
#[test]
fn does_not_shift_or_swap_channels() {
    let Some((rgba, rgb)) = render_both(64, 64) else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    // The scene is deliberately not grey, so a channel swap changes the totals.
    let sum = |v: &[u8], stride: usize, off: usize| -> u64 {
        v.iter().skip(off).step_by(stride).map(|b| *b as u64).sum()
    };
    for c in 0..3 {
        assert_eq!(
            sum(&rgba, 4, c),
            sum(&rgb, 3, c),
            "channel {c} totals differ — a swap or a shift"
        );
    }
    // And a shift by one would make neighbouring channels line up instead.
    let shifted = sum(&rgba, 4, 0) == sum(&rgb, 3, 1);
    assert!(
        !shifted || sum(&rgba, 4, 0) == sum(&rgba, 4, 1),
        "red matches green's slot: the pack is off by a byte"
    );
}
