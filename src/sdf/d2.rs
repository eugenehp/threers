//! 2D primitives, exact, negative inside. In WGSL as `sdf2_*` — see
//! [`WGSL`](super::WGSL).

use crate::math::Vector2;

/// A disc of radius `r` at the origin.
#[inline]
pub fn circle(p: Vector2, r: f32) -> f32 {
    p.length() - r
}

/// An axis-aligned box at the origin, `half` its half-extents.
#[inline]
pub fn rect(p: Vector2, half: Vector2) -> f32 {
    let qx = p.x.abs() - half.x;
    let qy = p.y.abs() - half.y;
    let (ox, oy) = (qx.max(0.0), qy.max(0.0));
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0)
}

/// [`rect`] with its corners rounded by `r`, keeping the same outer extent.
/// With `r` half the shorter side it is a pill (a stadium).
#[inline]
pub fn round_rect(p: Vector2, half: Vector2, r: f32) -> f32 {
    let r = r.max(0.0).min(half.x.min(half.y).max(0.0));
    rect(p, Vector2::new(half.x - r, half.y - r)) - r
}

/// Distance to the segment from `a` to `b`. Unsigned — a segment has no inside.
#[inline]
pub fn segment(p: Vector2, a: Vector2, b: Vector2) -> f32 {
    let v = b - a;
    let w = p - a;
    let len2 = v.dot(v);
    let t = if len2 > 0.0 { (w.dot(v) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (w - v * t).length()
}

/// A segment thickened to radius `r`: a stroke with round caps.
#[inline]
pub fn capsule(p: Vector2, a: Vector2, b: Vector2, r: f32) -> f32 {
    segment(p, a, b) - r
}

/// A capsule whose radius runs from `ra` at `a` to `rb` at `b` — the convex
/// hull of the two discs, joined by their outer tangents.
///
/// What a tapering stroke is, and the building block of anything organic: a
/// tentacle, a stem, a brush mark. Chain them end to end with matching radii
/// and the joints are seamless without any blending.
#[inline]
pub fn tapered_capsule(p: Vector2, a: Vector2, b: Vector2, ra: f32, rb: f32) -> f32 {
    tapered_capsule_radius(p, a, b, ra, rb).0
}

/// [`tapered_capsule`], and the capsule's radius at the point along its axis
/// nearest `p` — a local thickness, for shading the stroke as a round tube.
pub fn tapered_capsule_radius(p: Vector2, a: Vector2, b: Vector2, ra: f32, rb: f32) -> (f32, f32) {
    let pb = b - a;
    let p = p - a;
    let h = pb.dot(pb);
    let t = (p.dot(pb) / (h + 1e-12)).clamp(0.0, 1.0);
    let r = ra + (rb - ra) * t;
    let bb = ra - rb;
    if h <= bb * bb {
        // One disc contains the other: there are no tangents, only the
        // larger disc.
        return if ra > rb { (p.length() - ra, ra) } else { ((p - pb).length() - rb, rb) };
    }
    let qx = (p.x * pb.y - p.y * pb.x).abs() / h;
    let qy = p.dot(pb) / h;
    let cx = (h - bb * bb).sqrt();
    let k = cx * qy - bb * qx;
    let n = qx * qx + qy * qy;
    let d = if k < 0.0 {
        (h * n).sqrt() - ra
    } else if k > cx {
        (h * (n + 1.0 - 2.0 * qy)).sqrt() - rb
    } else {
        cx * qx + bb * qy - ra
    };
    (d, r)
}

/// A closed polygon. The sign comes from a crossing count, not the winding,
/// so either orientation — and a self-intersecting outline, by the even-odd
/// rule — gives the same answer.
pub fn polygon(p: Vector2, points: &[Vector2]) -> f32 {
    let n = points.len();
    if n == 0 {
        return f32::MAX;
    }
    let mut best = f32::MAX;
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + n - 1) % n]);
        best = best.min(segment(p, a, b));
        if (p.y >= a.y) != (p.y >= b.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if a.x + t * (b.x - a.x) > p.x {
                inside = !inside;
            }
        }
    }
    if inside {
        -best
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32) -> Vector2 {
        Vector2::new(x, y)
    }

    #[test]
    fn boxes_are_exact_at_corners_and_inside() {
        let half = v(2.0, 1.0);
        assert!((rect(v(5.0, 5.0), half) - (9.0f32 + 16.0).sqrt()).abs() < 1e-6);
        assert!((rect(v(0.0, 0.0), half) + 1.0).abs() < 1e-6);
        assert!((rect(v(1.9, 0.0), half) + 0.1).abs() < 1e-6);
        // A pill: the rounded box with r = half height.
        assert!((round_rect(v(3.0, 0.0), half, 1.0) - 1.0).abs() < 1e-6);
        // The corner arc of a 2x1 pill is centred on (1, 0).
        assert!((round_rect(v(1.0 + 3.0, 4.0), half, 1.0) - 4.0).abs() < 1e-5);
        // An oversized radius clamps rather than inverting the box.
        assert!((round_rect(v(0.0, 3.0), half, 9.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn segments_and_capsules() {
        let (a, b) = (v(0.0, 0.0), v(4.0, 0.0));
        assert_eq!(segment(v(2.0, 3.0), a, b), 3.0);
        assert_eq!(segment(v(7.0, 4.0), a, b), 5.0);
        assert_eq!(segment(v(1.0, 1.0), a, a), 2f32.sqrt());
        assert_eq!(capsule(v(2.0, 0.5), a, b, 1.0), -0.5);
    }

    /// The hull of two discs is the union of every disc between them, so
    /// outside it the distance is the least distance to any of those.
    #[test]
    fn tapered_capsule_matches_the_union_of_discs() {
        let cases = [
            (v(0.0, 0.0), v(10.0, 3.0), 3.0, 0.5),
            (v(-2.0, 1.0), v(4.0, -6.0), 0.2, 2.5),
            (v(0.0, 0.0), v(0.0, 5.0), 1.0, 1.0),
        ];
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for (a, b, ra, rb) in cases {
            for _ in 0..400 {
                let p = v(rnd() * 30.0 - 12.0, rnd() * 30.0 - 15.0);
                let d = tapered_capsule(p, a, b, ra, rb);
                let mut brute = f32::MAX;
                for i in 0..=4000 {
                    let t = i as f32 / 4000.0;
                    let c = a + (b - a) * t;
                    brute = brute.min((p - c).length() - (ra + (rb - ra) * t));
                }
                if brute > 0.0 {
                    assert!((d - brute).abs() < 2e-3, "{p:?}: {d} vs {brute}");
                } else {
                    assert!(d <= 1e-4, "{p:?} is inside but measured {d}");
                }
            }
        }
    }

    #[test]
    fn a_tapered_capsule_inside_out_is_its_larger_disc() {
        let (d, r) = tapered_capsule_radius(v(5.0, 0.0), v(0.0, 0.0), v(1.0, 0.0), 3.0, 1.0);
        assert!((d - 2.0).abs() < 1e-6);
        assert_eq!(r, 3.0);
        let (d, _) = tapered_capsule_radius(v(0.0, 0.0), v(0.0, 0.0), v(0.0, 0.0), 1.0, 1.0);
        assert!((d + 1.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_sign_ignores_winding() {
        let square = [v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)];
        let mut reversed = square;
        reversed.reverse();
        for pts in [&square[..], &reversed[..]] {
            assert!((polygon(v(1.0, 1.0), pts) + 1.0).abs() < 1e-6);
            assert!((polygon(v(3.0, 1.0), pts) - 1.0).abs() < 1e-6);
        }
        assert_eq!(polygon(v(0.0, 0.0), &[]), f32::MAX);
    }
}
