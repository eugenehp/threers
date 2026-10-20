use super::{Box3, Matrix4, Plane, Sphere, Triangle, Vector3};

/// Mirrors three.js's `Ray`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

impl Default for Ray {
    fn default() -> Self {
        Self {
            origin: Vector3::ZERO,
            direction: Vector3::new(0.0, 0.0, -1.0),
        }
    }
}

impl Ray {
    pub const fn new(origin: Vector3, direction: Vector3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vector3 {
        self.origin + self.direction * t
    }

    pub fn look_at(&mut self, target: Vector3) -> &mut Self {
        self.direction = (target - self.origin).normalize();
        self
    }

    pub fn recast(&self, t: f32) -> Self {
        Self {
            origin: self.at(t),
            direction: self.direction,
        }
    }

    pub fn closest_point_to_point(&self, p: Vector3) -> Vector3 {
        let t = (p - self.origin).dot(self.direction);
        if t < 0.0 {
            self.origin
        } else {
            self.at(t)
        }
    }

    pub fn distance_to_point(&self, p: Vector3) -> f32 {
        self.distance_sq_to_point(p).sqrt()
    }

    pub fn distance_sq_to_point(&self, p: Vector3) -> f32 {
        let t = (p - self.origin).dot(self.direction);
        if t < 0.0 {
            (self.origin - p).length_sq()
        } else {
            (self.at(t) - p).length_sq()
        }
    }

    /// Squared distance between this ray and the segment `v0`–`v1`, with the
    /// closest point on the ray and the closest point on the segment, in that
    /// order. Mirrors three.js's `distanceSqToSegment` (Geometric Tools'
    /// `DistRaySegment`), region for region. The direction must be unit length.
    pub fn distance_sq_to_segment(&self, v0: Vector3, v1: Vector3) -> (f32, Vector3, Vector3) {
        let seg_center = (v0 + v1) * 0.5;
        let seg_dir = (v1 - v0).normalize();
        let diff = self.origin - seg_center;
        let seg_extent = v0.distance_to(v1) * 0.5;
        let a01 = -self.direction.dot(seg_dir);
        let b0 = diff.dot(self.direction);
        let b1 = -diff.dot(seg_dir);
        let c = diff.length_sq();
        let det = (1.0 - a01 * a01).abs();
        let (s0, s1, sqr_dist);
        if det > 0.0 {
            // The ray and segment are not parallel.
            let (mut t0, mut t1) = (a01 * b1 - b0, a01 * b0 - b1);
            let ext_det = seg_extent * det;
            if t0 >= 0.0 {
                if t1 >= -ext_det {
                    if t1 <= ext_det {
                        // Region 0: interior points of ray and segment.
                        let inv_det = 1.0 / det;
                        t0 *= inv_det;
                        t1 *= inv_det;
                        sqr_dist =
                            t0 * (t0 + a01 * t1 + 2.0 * b0) + t1 * (a01 * t0 + t1 + 2.0 * b1) + c;
                    } else {
                        // Region 1.
                        t1 = seg_extent;
                        t0 = (-(a01 * t1 + b0)).max(0.0);
                        sqr_dist = -t0 * t0 + t1 * (t1 + 2.0 * b1) + c;
                    }
                } else {
                    // Region 5.
                    t1 = -seg_extent;
                    t0 = (-(a01 * t1 + b0)).max(0.0);
                    sqr_dist = -t0 * t0 + t1 * (t1 + 2.0 * b1) + c;
                }
            } else if t1 <= -ext_det {
                // Region 4.
                t0 = (-(-a01 * seg_extent + b0)).max(0.0);
                t1 = if t0 > 0.0 {
                    -seg_extent
                } else {
                    (-b1).max(-seg_extent).min(seg_extent)
                };
                sqr_dist = -t0 * t0 + t1 * (t1 + 2.0 * b1) + c;
            } else if t1 <= ext_det {
                // Region 3.
                t0 = 0.0;
                t1 = (-b1).max(-seg_extent).min(seg_extent);
                sqr_dist = t1 * (t1 + 2.0 * b1) + c;
            } else {
                // Region 2.
                t0 = (-(a01 * seg_extent + b0)).max(0.0);
                t1 = if t0 > 0.0 {
                    seg_extent
                } else {
                    (-b1).max(-seg_extent).min(seg_extent)
                };
                sqr_dist = -t0 * t0 + t1 * (t1 + 2.0 * b1) + c;
            }
            s0 = t0;
            s1 = t1;
        } else {
            // Parallel.
            s1 = if a01 > 0.0 { -seg_extent } else { seg_extent };
            s0 = (-(a01 * s1 + b0)).max(0.0);
            sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
        }
        (
            sqr_dist,
            self.origin + self.direction * s0,
            seg_center + seg_dir * s1,
        )
    }

    /// Returns the t along the ray where it enters the sphere, or None.
    ///
    /// three.js's `intersectSphere`, except that the squared distance from
    /// the centre to the ray is measured as the length of the perpendicular,
    /// not as `|oc|² − tca²`: in f32 that difference loses most of its digits
    /// when the ray starts far from a small sphere, enough to miss a ray
    /// through the sphere's own surface.
    pub fn intersect_sphere(&self, s: &Sphere) -> Option<f32> {
        let oc = s.center - self.origin;
        let tca = oc.dot(self.direction);
        let d2 = (oc - self.direction * tca).length_sq();
        let r2 = s.radius * s.radius;
        if d2 > r2 {
            return None;
        }
        let thc = (r2 - d2).sqrt();
        let t0 = tca - thc;
        let t1 = tca + thc;
        if t1 < 0.0 {
            return None;
        }
        if t0 < 0.0 {
            Some(t1)
        } else {
            Some(t0)
        }
    }

    pub fn intersects_sphere(&self, s: &Sphere) -> bool {
        self.intersect_sphere(s).is_some()
    }

    /// three.js `Ray.intersectBox`, line for line: the ray parameter of the
    /// first point on the box at or ahead of the origin — the entry, or the
    /// exit when the origin is inside — or `None`.
    ///
    /// A direction component of zero (either sign) makes its reciprocal
    /// infinite, and when the origin lies exactly on one of that axis's faces
    /// `0 · ∞` is NaN. three.js drops a NaN bound rather than letting it fail
    /// the comparisons, so a ray grazing along a face — a pick on the very
    /// edge of a flat plane, whose box has zero thickness — still enters the
    /// box. The slab order follows the reciprocal's sign, not a comparison of
    /// the two bounds, for the same reason.
    pub fn intersect_box(&self, b: &Box3) -> Option<f32> {
        let invdirx = 1.0 / self.direction.x;
        let invdiry = 1.0 / self.direction.y;
        let invdirz = 1.0 / self.direction.z;
        let o = self.origin;
        let (mut tmin, mut tmax) = slab(o.x, b.min.x, b.max.x, invdirx);
        let (tymin, tymax) = slab(o.y, b.min.y, b.max.y, invdiry);
        if tmin > tymax || tymin > tmax {
            return None;
        }
        if tymin > tmin || tmin.is_nan() {
            tmin = tymin;
        }
        if tymax < tmax || tmax.is_nan() {
            tmax = tymax;
        }
        let (tzmin, tzmax) = slab(o.z, b.min.z, b.max.z, invdirz);
        if tmin > tzmax || tzmin > tmax {
            return None;
        }
        if tzmin > tmin || tmin.is_nan() {
            tmin = tzmin;
        }
        if tzmax < tmax || tmax.is_nan() {
            tmax = tzmax;
        }
        // The point closest to the ray's origin on its positive side.
        if tmax < 0.0 {
            return None;
        }
        Some(if tmin >= 0.0 { tmin } else { tmax })
    }

    pub fn intersects_box(&self, b: &Box3) -> bool {
        self.intersect_box(b).is_some()
    }

    pub fn intersect_plane(&self, plane: &Plane) -> Option<f32> {
        let denom = plane.normal.dot(self.direction);
        if denom == 0.0 {
            // Ray is parallel to plane. Hit only if origin is on the plane.
            if plane.distance_to_point(self.origin) == 0.0 {
                return Some(0.0);
            }
            return None;
        }
        let t = -(self.origin.dot(plane.normal) + plane.constant) / denom;
        if t >= 0.0 {
            Some(t)
        } else {
            None
        }
    }

    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        self.intersect_plane(plane).is_some()
    }

    /// three.js `Ray.intersectTriangle`: the ray parameter of the hit, or
    /// `None`. See [`intersect_triangle_weights`](Self::intersect_triangle_weights).
    pub fn intersect_triangle(&self, tri: &Triangle, backface_culling: bool) -> Option<f32> {
        self.intersect_triangle_weights(tri.a, tri.b, tri.c, backface_culling)
            .map(|(t, _, _)| t)
    }

    /// three.js `Ray.intersectTriangle` (Geometric Tools' `IntrRay3Triangle3`),
    /// in f64 as JavaScript evaluates it: the ray parameter and the weights
    /// `(wb, wc)` of `b` and `c`, so the hit is `a + wb·(b − a) + wc·(c − a)`.
    /// With `backface_culling` a triangle seen from behind misses; edge-on is
    /// always a miss. The tests are inclusive, so a ray through a shared edge
    /// or vertex can hit each triangle there — exactly when three.js's does,
    /// which is why this is computed at JavaScript's precision.
    pub fn intersect_triangle_weights(
        &self,
        a: Vector3,
        b: Vector3,
        c: Vector3,
        backface_culling: bool,
    ) -> Option<(f32, f32, f32)> {
        let v = |p: Vector3| [p.x as f64, p.y as f64, p.z as f64];
        intersect_triangle_f64(
            v(self.origin),
            v(self.direction),
            v(a),
            v(b),
            v(c),
            backface_culling,
        )
        .map(|(t, wb, wc)| (t as f32, wb as f32, wc as f32))
    }

    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        let e = &m.elements;
        let w = e[3] * self.origin.x + e[7] * self.origin.y + e[11] * self.origin.z + e[15];
        let inv_w = if w == 0.0 { 1.0 } else { 1.0 / w };
        let origin = Vector3::new(
            (e[0] * self.origin.x + e[4] * self.origin.y + e[8] * self.origin.z + e[12]) * inv_w,
            (e[1] * self.origin.x + e[5] * self.origin.y + e[9] * self.origin.z + e[13]) * inv_w,
            (e[2] * self.origin.x + e[6] * self.origin.y + e[10] * self.origin.z + e[14]) * inv_w,
        );
        let direction = Vector3::new(
            e[0] * self.direction.x + e[4] * self.direction.y + e[8] * self.direction.z,
            e[1] * self.direction.x + e[5] * self.direction.y + e[9] * self.direction.z,
            e[2] * self.direction.x + e[6] * self.direction.y + e[10] * self.direction.z,
        )
        .normalize();
        Self { origin, direction }
    }
}

/// One axis of three.js's slab test: the near and far ray parameters of the
/// pair of planes, ordered by the sign of the reciprocal direction (`-0.0`
/// gives `-∞`, so it counts as negative, as it does in JavaScript).
fn slab(o: f32, min: f32, max: f32, inv_d: f32) -> (f32, f32) {
    if inv_d >= 0.0 {
        ((min - o) * inv_d, (max - o) * inv_d)
    } else {
        ((max - o) * inv_d, (min - o) * inv_d)
    }
}

/// three.js `Ray.intersectTriangle` in f64, for a ray given as `origin` and
/// `direction`: the ray parameter and the weights of `b` and `c`. See
/// [`Ray::intersect_triangle_weights`].
pub(crate) fn intersect_triangle_f64(
    origin: [f64; 3],
    dir: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    backface_culling: bool,
) -> Option<(f64, f64, f64)> {
    type V = [f64; 3];
    let sub = |p: V, q: V| -> V { [p[0] - q[0], p[1] - q[1], p[2] - q[2]] };
    let dot = |p: V, q: V| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    let cross = |p: V, q: V| -> V {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let edge1 = sub(b, a);
    let edge2 = sub(c, a);
    let normal = cross(edge1, edge2);
    let mut ddn = dot(dir, normal);
    let sign = if ddn > 0.0 {
        if backface_culling {
            return None;
        }
        1.0
    } else if ddn < 0.0 {
        ddn = -ddn;
        -1.0
    } else {
        return None;
    };
    let diff = sub(origin, a);
    let ddqxe2 = sign * dot(dir, cross(diff, edge2));
    if ddqxe2 < 0.0 {
        return None;
    }
    let dde1xq = sign * dot(dir, cross(edge1, diff));
    if dde1xq < 0.0 || ddqxe2 + dde1xq > ddn {
        return None;
    }
    let qdn = -sign * dot(diff, normal);
    if qdn < 0.0 {
        return None;
    }
    Some((qdn / ddn, ddqxe2 / ddn, dde1xq / ddn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_unit_sphere_from_minus_z() {
        let r = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let s = Sphere::new(Vector3::ZERO, 1.0);
        let t = r.intersect_sphere(&s).expect("hit");
        assert!((t - 4.0).abs() < 1e-5);
    }

    #[test]
    fn intersect_axis_aligned_box() {
        let r = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let b = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
        let t = r.intersect_box(&b).expect("hit");
        assert!((t - 4.0).abs() < 1e-5);
    }

    /// three.js r165 `Ray.intersectBox` on rays with zero direction
    /// components whose origin lies exactly on that axis's faces: the NaN
    /// from `0 · ∞` is dropped, not allowed to fail the test.
    #[test]
    fn intersect_box_ignores_the_nan_of_a_grazing_axis() {
        let flat = Box3::new(Vector3::new(-1.0, -1.0, 0.0), Vector3::new(1.0, 1.0, 0.0));
        for (o, d) in [
            ([1.0, 0.3, 5.0], [0.0, 0.0, -1.0]),
            ([1.0, 0.3, 5.0], [-0.0, 0.0, -1.0]),
            ([-1.0, 0.3, 5.0], [0.0, -0.0, -1.0]),
            ([-1.0, -1.0, 5.0], [-0.0, -0.0, -1.0]),
            ([1.0, 1.0, 5.0], [0.0, 0.0, -1.0]),
        ] {
            let r = Ray::new(
                Vector3::new(o[0], o[1], o[2]),
                Vector3::new(d[0], d[1], d[2]),
            );
            assert_eq!(r.intersect_box(&flat), Some(5.0), "{o:?} {d:?}");
        }
        // Just outside, and behind: misses, as in three.js.
        let r = Ray::new(Vector3::new(1.5, 0.3, 5.0), Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(r.intersect_box(&flat), None);
        let r = Ray::new(Vector3::new(1.0, 0.3, 5.0), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(r.intersect_box(&flat), None);
        // Inside a solid box: the exit.
        let unit = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
        let r = Ray::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(r.intersect_box(&unit), Some(1.0));
    }

    #[test]
    fn intersect_triangle_basic() {
        let r = Ray::new(Vector3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let tri = Triangle::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let t = r.intersect_triangle(&tri, false).expect("hit");
        assert!((t - 1.0).abs() < 1e-5);
    }
}
