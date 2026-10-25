//! Small overlay helpers shared by the cursor and the effects.

use crate::math::Vector2;
use crate::overlay::Overlay;
use crate::sdf::d2;

/// Bounding box of `points` as `(x0, y0, x1, y1)`.
pub(crate) fn bounds(points: &[Vector2]) -> (f32, f32, f32, f32) {
    points.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |(a, b, c, d), p| {
        (a.min(p.x), b.min(p.y), c.max(p.x), d.max(p.y))
    })
}

/// A run of round-capped segments `half` either side of their centre line,
/// in the overlay's current stroke colour, as one stroke — so the joints are
/// covered once and a hair thinner than a pixel keeps its weight.
pub(crate) fn stroke(o: &mut Overlay, pts: &[Vector2], half: f32) {
    if pts.len() < 2 {
        return;
    }
    o.stroke_weight(half * 2.0);
    o.polyline(pts);
}

/// A run of segments whose radius tapers from point to point, in the current
/// stroke colour, with each pixel's coverage taken against the stroke's local
/// width — so the fine end of a taper fades as it thins below a pixel rather
/// than staying a pixel-wide dark line.
pub(crate) fn taper(o: &mut Overlay, pts: &[Vector2], radii: &[f32]) {
    if pts.len() < 2 {
        return;
    }
    let (x0, y0, x1, y1) = bounds(pts);
    let pad = radii.iter().cloned().fold(0.0, f32::max);
    o.stroke_field_with_width(x0 - pad, y0 - pad, x1 + pad, y1 + pad, |p| {
        // the piece whose edge is nearest, and its radius there
        let (mut best, mut radius) = (f32::MAX, 0.0);
        for k in 0..pts.len() - 1 {
            let (d, r) = d2::tapered_capsule_radius(p, pts[k], pts[k + 1], radii[k], radii[k + 1]);
            if d < best {
                best = d;
                radius = r;
            }
        }
        (best + radius, radius)
    });
}
