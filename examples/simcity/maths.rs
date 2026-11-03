//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Small maths helpers.
// ---------------------------------------------------------------------------

pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub(crate) fn mix_color(a: Color, b: Color, t: f32) -> Color {
    Color::new(mix(a.r, b.r, t), mix(a.g, b.g, t), mix(a.b, b.b, t))
}

/// Put a linear colour through the same tail the shader gives every fragment:
/// ACES, then the sRGB OETF.
///
/// The clear colour is NOT tone mapped — with a non-sRGB target the renderer
/// writes `scene.background` into the framebuffer verbatim, while shaded
/// fragments arrive via `aces_tonemap` then `linear_to_srgb`. Set the
/// background to a raw linear colour and the sky ends up visibly darker than
/// the fully-fogged terrain in front of it, with a hard seam along the far
/// edge of the ground plane. Pre-applying the tail makes the two agree.
pub(crate) fn shaded_like_a_fragment(c: Color) -> Color {
    fn aces(x: f32) -> f32 {
        ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    }
    fn linear_to_srgb(v: f32) -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }
    Color::new(
        linear_to_srgb(aces(c.r)),
        linear_to_srgb(aces(c.g)),
        linear_to_srgb(aces(c.b)),
    )
}

pub(crate) fn scale_color(c: Color, k: f32) -> Color {
    Color::new(c.r * k, c.g * k, c.b * k)
}

/// Hash-based value noise on a unit lattice — the zoning map's irregularity.
pub(crate) fn city_hash2(x: i32, y: i32) -> f32 {
    let mut h = x.wrapping_mul(374_761_393).wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) as u32 as f32) / (u32::MAX as f32)
}

pub(crate) fn city_noise(x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (tx, ty) = (x - xi, y - yi);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let (x0, y0) = (xi as i32, yi as i32);
    let a = city_hash2(x0, y0);
    let b = city_hash2(x0 + 1, y0);
    let c = city_hash2(x0, y0 + 1);
    let d = city_hash2(x0 + 1, y0 + 1);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}
