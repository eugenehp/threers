//! 3D primitives, exact, negative inside. In WGSL as `sdf3_*` — see
//! [`WGSL`](super::WGSL).

use crate::math::Vector3;

/// A ball of radius `r` at the origin.
#[inline]
pub fn sphere(p: Vector3, r: f32) -> f32 {
    p.length() - r
}

/// An axis-aligned box at the origin, `half` its half-extents.
#[inline]
pub fn cuboid(p: Vector3, half: Vector3) -> f32 {
    let q = Vector3::new(p.x.abs() - half.x, p.y.abs() - half.y, p.z.abs() - half.z);
    let o = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
    o.length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// [`cuboid`] with its edges rounded by `r`, keeping the same outer extent.
#[inline]
pub fn round_cuboid(p: Vector3, half: Vector3, r: f32) -> f32 {
    let r = r.max(0.0).min(half.x.min(half.y.min(half.z)).max(0.0));
    cuboid(p, Vector3::new(half.x - r, half.y - r, half.z - r)) - r
}

/// The segment from `a` to `b` thickened to radius `r`.
#[inline]
pub fn capsule(p: Vector3, a: Vector3, b: Vector3, r: f32) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let len2 = ba.dot(ba);
    let t = if len2 > 0.0 { (pa.dot(ba) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (pa - ba * t).length() - r
}

/// A capsule whose radius runs from `ra` at `a` to `rb` at `b`: the convex
/// hull of the two balls (a "round cone"). See [`super::d2::tapered_capsule`].
pub fn tapered_capsule(p: Vector3, a: Vector3, b: Vector3, ra: f32, rb: f32) -> f32 {
    let ba = b - a;
    let l2 = ba.dot(ba);
    let rr = ra - rb;
    let a2 = l2 - rr * rr;
    if a2 <= 0.0 {
        // One ball contains the other.
        return if ra > rb { (p - a).length() - ra } else { (p - b).length() - rb };
    }
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = pa.dot(ba);
    let z = y - l2;
    let x2 = (pa * l2 - ba * y).length_sq();
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        return (x2 + z2).sqrt() * il2 - rb;
    }
    if y.signum() * a2 * y2 < k {
        return (x2 + y2).sqrt() * il2 - ra;
    }
    ((x2 * a2 * il2).sqrt() + y * rr) * il2 - ra
}

/// A torus at the origin lying in the xz plane: `major` the radius of the
/// ring, `minor` of the tube.
#[inline]
pub fn torus(p: Vector3, major: f32, minor: f32) -> f32 {
    let qx = (p.x * p.x + p.z * p.z).sqrt() - major;
    (qx * qx + p.y * p.y).sqrt() - minor
}

/// A capped cylinder at the origin along y, `half_height` above and below.
#[inline]
pub fn cylinder(p: Vector3, half_height: f32, r: f32) -> f32 {
    let dx = (p.x * p.x + p.z * p.z).sqrt() - r;
    let dy = p.y.abs() - half_height;
    let (ox, oy) = (dx.max(0.0), dy.max(0.0));
    dx.max(dy).min(0.0) + (ox * ox + oy * oy).sqrt()
}

/// The half-space below the plane `dot(p, normal) + offset = 0`; `normal`
/// must be unit length.
#[inline]
pub fn plane(p: Vector3, normal: Vector3, offset: f32) -> f32 {
    p.dot(normal) + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3::new(x, y, z)
    }

    #[test]
    fn simple_shapes_are_exact() {
        assert!((cuboid(v(3.0, 4.0, 1.0), v(1.0, 1.0, 1.0)) - (4.0f32 + 9.0).sqrt()).abs() < 1e-6);
        assert!((cuboid(v(0.0, 0.0, 0.0), v(1.0, 2.0, 3.0)) + 1.0).abs() < 1e-6);
        assert!((round_cuboid(v(0.0, 3.0, 0.0), v(1.0, 1.0, 1.0), 0.5) - 2.0).abs() < 1e-6);
        assert!((torus(v(2.0, 0.0, 0.0), 2.0, 0.5) + 0.5).abs() < 1e-6);
        assert!((torus(v(0.0, 0.0, 0.0), 2.0, 0.5) - 1.5).abs() < 1e-6);
        assert!((cylinder(v(0.0, 5.0, 0.0), 1.0, 2.0) - 4.0).abs() < 1e-6);
        assert!((cylinder(v(5.0, 0.0, 0.0), 1.0, 2.0) - 3.0).abs() < 1e-6);
        assert!((cylinder(v(0.0, 0.0, 0.0), 1.0, 2.0) + 1.0).abs() < 1e-6);
        assert_eq!(plane(v(0.0, 3.0, 0.0), v(0.0, 1.0, 0.0), -1.0), 2.0);
        assert!((capsule(v(1.0, 2.0, 0.0), v(0.0, 0.0, 0.0), v(3.0, 0.0, 0.0), 0.5) - 1.5).abs() < 1e-6);
    }

    /// As in 2D: outside the hull of two balls, the distance is the least
    /// distance to any ball between them.
    #[test]
    fn tapered_capsule_matches_the_union_of_balls() {
        let cases = [
            (v(0.0, 0.0, 0.0), v(4.0, 1.0, -2.0), 1.5, 0.3),
            (v(1.0, -1.0, 0.0), v(-1.0, 3.0, 2.0), 0.2, 1.2),
            (v(0.0, 0.0, 0.0), v(0.0, 0.0, 3.0), 0.7, 0.7),
        ];
        let mut seed = 777u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for (a, b, ra, rb) in cases {
            for _ in 0..300 {
                let p = v(rnd() * 12.0 - 6.0, rnd() * 12.0 - 6.0, rnd() * 12.0 - 6.0);
                let d = tapered_capsule(p, a, b, ra, rb);
                let mut brute = f32::MAX;
                for i in 0..=3000 {
                    let t = i as f32 / 3000.0;
                    brute = brute.min((p - a.lerp(b, t)).length() - (ra + (rb - ra) * t));
                }
                if brute > 0.0 {
                    assert!((d - brute).abs() < 2e-3, "{p:?}: {d} vs {brute}");
                } else {
                    assert!(d <= 1e-4, "{p:?} is inside but measured {d}");
                }
            }
        }
        // Nested balls: the larger one.
        assert!((tapered_capsule(v(4.0, 0.0, 0.0), v(0.0, 0.0, 0.0), v(0.5, 0.0, 0.0), 2.0, 1.0) - 2.0).abs() < 1e-6);
    }
}
