//! Pixel probes for a `wireframe: true` quad, shared by the wgpu suite
//! (`tests/wgpu_wireframe.rs`) and the Metal one (`tests/metal_backend.rs`) so
//! the two backends are held to the same picture.
//!
//! The quad spans `±HALF` on the `z = 0` plane and is seen straight down −Z
//! through an orthographic camera whose frustum is the unit square, into a
//! `SIZE`×`SIZE` image. NDC is then world XY, so where a point lands is worked
//! out here rather than asked of the renderer.

#![allow(dead_code)]

use threers::core::BufferGeometry;
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, Material};
use threers::math::Color;

pub const SIZE: u32 = 64;
pub const HALF: f32 = 0.6;

/// Where a world point on `z = 0` lands, in pixels from the top-left.
pub fn to_px(x: f32, y: f32) -> (f32, f32) {
    ((x + 1.0) * 0.5 * SIZE as f32, (1.0 - y) * 0.5 * SIZE as f32)
}

pub fn lit_at(rgba: &[u8], x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= SIZE as i32 || y >= SIZE as i32 {
        return false;
    }
    let i = (y as usize * SIZE as usize + x as usize) * 4;
    rgba[i].max(rgba[i + 1]).max(rgba[i + 2]) > 128
}

/// How many of 13 samples over the segment's middle 60% have a lit pixel
/// within one pixel (a 1px line can land either side of a centre that falls
/// on a pixel boundary), and the sample count.
pub fn segment_drawn(rgba: &[u8], a: (f32, f32), b: (f32, f32)) -> (usize, usize) {
    let (pa, pb) = (to_px(a.0, a.1), to_px(b.0, b.1));
    let samples = 13;
    let mut hits = 0;
    for k in 0..samples {
        let t = 0.2 + 0.6 * k as f32 / (samples - 1) as f32;
        let (x, y) = (pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t);
        let (x, y) = (x.floor() as i32, y.floor() as i32);
        hits += (-1..=1).any(|dy| (-1..=1).any(|dx| lit_at(rgba, x + dx, y + dy))) as usize;
    }
    (hits, samples)
}

/// The quad's corners, counter-clockwise from the bottom-left.
pub fn corners() -> [(f32, f32); 4] {
    [(-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)]
}

/// How many of the quad's two diagonals are drawn at every sample.
pub fn diagonals_drawn(rgba: &[u8]) -> usize {
    let c = corners();
    [(0, 2), (1, 3)]
        .iter()
        .filter(|&&(a, b)| {
            let (hits, n) = segment_drawn(rgba, c[a], c[b]);
            hits == n
        })
        .count()
}

/// A point inside the quad away from its edges and diagonals.
pub fn interior_lit(rgba: &[u8]) -> bool {
    let (cx, cy) = to_px(0.0, 0.3);
    lit_at(rgba, cx as i32, cy as i32)
}

/// The four edges drawn, neither diagonal, and nothing filled.
pub fn assert_outline_only(rgba: &[u8], what: &str) {
    let c = corners();
    for i in 0..4 {
        let (hits, n) = segment_drawn(rgba, c[i], c[(i + 1) % 4]);
        assert_eq!(hits, n, "{what}: edge {i} drawn at only {hits}/{n} samples");
    }
    for (a, b) in [(0, 2), (1, 3)] {
        let (hits, _) = segment_drawn(rgba, c[a], c[b]);
        assert_eq!(hits, 0, "{what}: diagonal {a}-{b} lit at {hits} samples");
    }
    assert!(!interior_lit(rgba), "{what}: interior lit");
}

/// The probed quad as a `PlaneGeometry`, which records it as one polygon.
pub fn plane() -> BufferGeometry {
    PlaneGeometry::new(2.0 * HALF, 2.0 * HALF)
}

pub fn wire_material() -> Material {
    let mut m = BasicMaterial::new(Color::WHITE);
    m.wireframe = true;
    Material::Basic(m)
}
