//! Rigid-transform and inertia-tensor math built on the `threers` vector types.
//!
//! Everything here is deliberately thin: [`Isometry`] is a position + rotation
//! pair (no scale — rigid bodies do not scale), and [`Mat3`] is a row-major 3x3
//! used for inertia tensors, which need a `matrix * vector` product that
//! `threers::Matrix3` does not expose.

use threers::math::{Box3, Matrix4, Quaternion, Vector3};

/// Axis-aligned bounding box. Alias of [`threers::math::Box3`] so AABBs pass
/// straight into the rest of the engine (`Ray::intersect_box`, `MeshBvh`, …).
pub type Aabb = Box3;

/// A rigid transform: rotate, then translate. No scale.
///
/// ```
/// use threers_physics::prelude::*;
/// use std::f32::consts::FRAC_PI_2;
///
/// let iso = Isometry::new(
///     Vector3::new(0.0, 1.0, 0.0),
///     Quaternion::from_axis_angle(Vector3::UP, FRAC_PI_2),
/// );
/// let p = iso.transform_point(Vector3::new(1.0, 0.0, 0.0));
/// assert!((p.z - -1.0).abs() < 1e-5);
/// assert!((p.y - 1.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Isometry {
    pub translation: Vector3,
    pub rotation: Quaternion,
}

impl Default for Isometry {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Isometry {
    pub const IDENTITY: Self = Self {
        translation: Vector3::ZERO,
        rotation: Quaternion::identity(),
    };

    pub const fn new(translation: Vector3, rotation: Quaternion) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    pub const fn from_translation(translation: Vector3) -> Self {
        Self {
            translation,
            rotation: Quaternion::identity(),
        }
    }

    pub const fn from_rotation(rotation: Quaternion) -> Self {
        Self {
            translation: Vector3::ZERO,
            rotation,
        }
    }

    /// Local point → world point.
    #[inline]
    pub fn transform_point(&self, p: Vector3) -> Vector3 {
        p.apply_quaternion(self.rotation) + self.translation
    }

    /// Local direction → world direction (ignores translation).
    #[inline]
    pub fn transform_vector(&self, v: Vector3) -> Vector3 {
        v.apply_quaternion(self.rotation)
    }

    /// World point → local point.
    #[inline]
    pub fn inverse_transform_point(&self, p: Vector3) -> Vector3 {
        (p - self.translation).apply_quaternion(self.rotation.conjugate())
    }

    /// World direction → local direction.
    #[inline]
    pub fn inverse_transform_vector(&self, v: Vector3) -> Vector3 {
        v.apply_quaternion(self.rotation.conjugate())
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.conjugate();
        Self {
            translation: (-self.translation).apply_quaternion(inv_rot),
            rotation: inv_rot,
        }
    }

    /// `self * other` — apply `other` first, then `self`.
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            translation: self.transform_point(other.translation),
            rotation: self.rotation.multiply(other.rotation),
        }
    }

    /// The transform taking `self`-local coordinates into `other`-local ones.
    pub fn inv_mul(&self, other: &Self) -> Self {
        self.inverse().mul(other)
    }

    /// Interpolate for render-time smoothing between fixed physics ticks.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
        }
    }

    pub fn to_matrix4(&self) -> Matrix4 {
        Matrix4::compose(self.translation, self.rotation, Vector3::ONE)
    }

    /// Drops any scale in `m`.
    pub fn from_matrix4(m: &Matrix4) -> Self {
        let (t, r, _) = m.decompose();
        Self::new(t, r)
    }

    /// Renormalize the rotation. Called after integration to stop quaternion
    /// drift from accumulating.
    pub fn renormalize(&mut self) {
        self.rotation = self.rotation.normalize();
    }
}

/// Row-major 3x3 matrix: `m[row * 3 + col]`.
///
/// Used for inertia tensors, which are rotated every step
/// (`R * I * Rᵀ`) and multiplied by angular-velocity vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [f32; 9],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const ZERO: Self = Self { m: [0.0; 9] };
    pub const IDENTITY: Self = Self {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    pub const fn from_diagonal(d: Vector3) -> Self {
        Self {
            m: [d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z],
        }
    }

    pub fn diagonal(&self) -> Vector3 {
        Vector3::new(self.m[0], self.m[4], self.m[8])
    }

    /// Rotation matrix for a unit quaternion.
    pub fn from_quat(q: Quaternion) -> Self {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self {
            m: [
                1.0 - (yy + zz),
                xy - wz,
                xz + wy,
                xy + wz,
                1.0 - (xx + zz),
                yz - wx,
                xz - wy,
                yz + wx,
                1.0 - (xx + yy),
            ],
        }
    }

    /// Cross-product matrix: `skew(a) * b == a.cross(b)`.
    pub fn skew(v: Vector3) -> Self {
        Self {
            m: [0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0],
        }
    }

    /// Outer product `a ⊗ b`.
    pub fn outer(a: Vector3, b: Vector3) -> Self {
        Self {
            m: [
                a.x * b.x,
                a.x * b.y,
                a.x * b.z,
                a.y * b.x,
                a.y * b.y,
                a.y * b.z,
                a.z * b.x,
                a.z * b.y,
                a.z * b.z,
            ],
        }
    }

    #[inline]
    pub fn mul_vec(&self, v: Vector3) -> Vector3 {
        let m = &self.m;
        Vector3::new(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z,
        )
    }

    pub fn mul(&self, o: &Self) -> Self {
        let mut r = [0.0f32; 9];
        for row in 0..3 {
            for col in 0..3 {
                r[row * 3 + col] = self.m[row * 3] * o.m[col]
                    + self.m[row * 3 + 1] * o.m[3 + col]
                    + self.m[row * 3 + 2] * o.m[6 + col];
            }
        }
        Self { m: r }
    }

    pub fn transpose(&self) -> Self {
        let m = &self.m;
        Self {
            m: [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        Self {
            m: std::array::from_fn(|i| self.m[i] + o.m[i]),
        }
    }

    pub fn sub(&self, o: &Self) -> Self {
        Self {
            m: std::array::from_fn(|i| self.m[i] - o.m[i]),
        }
    }

    pub fn scaled(&self, s: f32) -> Self {
        Self {
            m: std::array::from_fn(|i| self.m[i] * s),
        }
    }

    pub fn determinant(&self) -> f32 {
        let m = &self.m;
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Matrix inverse, or [`Mat3::ZERO`] if singular.
    ///
    /// Returning zero is deliberate: a singular inverse-inertia means "infinite
    /// inertia about that axis", and a zero inverse-inertia tensor is exactly
    /// how the solver represents a non-rotating body.
    ///
    /// # Why the singularity test is relative
    ///
    /// A determinant has the cube of the matrix's units, so "close to zero" is
    /// meaningless without knowing the scale. Both matrices that reach this
    /// function are small when the scene is: an inertia tensor goes as
    /// mass × length², and a constraint's effective mass goes as *inverse*
    /// mass. A 200 × 40 × 20 mm part modelled in metres has principal moments
    /// around 1e-7, so its determinant is around 1e-21 — and an absolute
    /// threshold of 1e-20 called that singular and quietly gave the part
    /// infinite rotational inertia. It could not turn, on any hinge, ever, and
    /// nothing reported it.
    ///
    /// Measuring the determinant against the matrix's own scale instead makes
    /// the test say what it means. At ordinary scales it decides exactly as the
    /// absolute one did; below them it stops lying.
    pub fn inverse(&self) -> Self {
        let m = &self.m;
        let scale = m.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        if scale <= 0.0 || !scale.is_finite() {
            return Self::ZERO;
        }
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if det.abs() <= 1e-9 * scale * scale * scale {
            return Self::ZERO;
        }
        let inv = 1.0 / det;
        Self {
            m: [
                c00 * inv,
                (m[2] * m[7] - m[1] * m[8]) * inv,
                (m[1] * m[5] - m[2] * m[4]) * inv,
                c01 * inv,
                (m[0] * m[8] - m[2] * m[6]) * inv,
                (m[2] * m[3] - m[0] * m[5]) * inv,
                c02 * inv,
                (m[1] * m[6] - m[0] * m[7]) * inv,
                (m[0] * m[4] - m[1] * m[3]) * inv,
            ],
        }
    }

    /// Rotate a body-frame tensor into world space: `R * self * Rᵀ`.
    pub fn rotated(&self, q: Quaternion) -> Self {
        let r = Self::from_quat(q);
        r.mul(self).mul(&r.transpose())
    }

    pub fn to_threers(&self) -> threers::math::Matrix3 {
        // threers::Matrix3 is column-major, so this is a transpose.
        let m = &self.m;
        threers::math::Matrix3 {
            elements: [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
        }
    }
}

/// Any vector perpendicular to `n`, chosen to stay well-conditioned for every
/// input direction. Used to build friction tangent bases.
pub fn orthonormal_basis(n: Vector3) -> (Vector3, Vector3) {
    // Branchless Frisvad-style construction; the sign trick avoids the
    // singularity at n ≈ (0, 0, -1) that the naive version has.
    let sign = if n.z >= 0.0 { 1.0f32 } else { -1.0f32 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vector3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// Closest points between two segments `[p1, q1]` and `[p2, q2]`.
///
/// Returns `(s, t, c1, c2)` where `c1 = p1 + s * (q1 - p1)`. Handles the
/// degenerate parallel and zero-length cases.
pub fn closest_points_segment_segment(
    p1: Vector3,
    q1: Vector3,
    p2: Vector3,
    q2: Vector3,
) -> (f32, f32, Vector3, Vector3) {
    const EPS: f32 = 1e-9;
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (mut s, mut t);
    if a <= EPS && e <= EPS {
        return (0.0, 0.0, p1, p2);
    }
    if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                // Parallel segments — any s works; 0 keeps it stable.
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    (s, t, p1 + d1 * s, p2 + d2 * t)
}

/// Closest point on segment `[a, b]` to `p`.
pub fn closest_point_on_segment(p: Vector3, a: Vector3, b: Vector3) -> Vector3 {
    let ab = b - a;
    let len_sq = ab.dot(ab);
    if len_sq < 1e-12 {
        return a;
    }
    a + ab * ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
}

/// Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision
/// Detection*, §5.1.5).
pub fn closest_point_on_triangle(p: Vector3, a: Vector3, b: Vector3, c: Vector3) -> Vector3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        if denom.abs() > 1e-12 {
            return a + ab * (d1 / denom);
        }
        return a;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        if denom.abs() > 1e-12 {
            return a + ac * (d2 / denom);
        }
        return a;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        if denom.abs() > 1e-12 {
            return b + (c - b) * ((d4 - d3) / denom);
        }
        return b;
    }

    let denom = va + vb + vc;
    if denom.abs() < 1e-12 {
        return a;
    }
    let inv = 1.0 / denom;
    a + ab * (vb * inv) + ac * (vc * inv)
}

/// Safe normalize: returns `None` for vectors shorter than `1e-12`.
#[inline]
pub fn try_normalize(v: Vector3) -> Option<Vector3> {
    let len_sq = v.length_sq();
    if len_sq < 1e-24 {
        None
    } else {
        Some(v * (1.0 / len_sq.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn close(a: Vector3, b: Vector3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn a_small_but_perfectly_invertible_matrix_is_not_called_singular() {
        // The inertia tensor of a 200 × 40 × 20 mm part, in metres at unit
        // density: principal moments around 1e-7, determinant around 1e-21.
        // An absolute singularity threshold called this singular and gave the
        // part infinite rotational inertia — it could not turn on any hinge.
        let i = Mat3::from_diagonal(Vector3::new(2.67e-8, 5.6e-7, 5.7e-7));
        let inv = i.inverse();
        assert_ne!(inv, Mat3::ZERO, "a real inertia tensor read as singular");

        // And it is a real inverse, not merely non-zero.
        let round_trip = i.mul(&inv);
        for k in 0..3 {
            let d = round_trip.diagonal();
            let v = [d.x, d.y, d.z][k];
            assert!((v - 1.0).abs() < 1e-3, "diagonal {k} came back as {v}");
        }
    }

    #[test]
    fn the_same_matrix_inverts_the_same_way_at_any_scale() {
        // Scale invariance is the property the relative test buys: the same
        // shape of body behaves the same whether the model is drawn in metres
        // or in millimetres.
        let base = Mat3::from_diagonal(Vector3::new(2.0, 3.0, 5.0));
        for scale in [1e-9f32, 1e-4, 1.0, 1e4] {
            let inv = base.scaled(scale).inverse();
            assert_ne!(inv, Mat3::ZERO, "singular at scale {scale}");
            let d = inv.diagonal();
            assert!(
                (d.x * 2.0 * scale - 1.0).abs() < 1e-3,
                "scale {scale} gave {d:?}"
            );
        }
    }

    #[test]
    fn a_genuinely_singular_matrix_still_reports_singular() {
        // A body with no rotational inertia at all — what a fixed body carries.
        assert_eq!(Mat3::ZERO.inverse(), Mat3::ZERO);
        // And one that is rank-deficient rather than empty.
        let flat = Mat3::from_diagonal(Vector3::new(1.0, 1.0, 0.0));
        assert_eq!(flat.inverse(), Mat3::ZERO);
        // Scaling a singular matrix does not make it invertible.
        assert_eq!(flat.scaled(1e-8).inverse(), Mat3::ZERO);
    }

    #[test]
    fn isometry_inverse_round_trips() {
        let iso = Isometry::new(
            Vector3::new(3.0, -2.0, 5.0),
            Quaternion::from_euler_xyz(0.3, -1.1, 2.0).normalize(),
        );
        let p = Vector3::new(1.0, 2.0, 3.0);
        assert!(close(iso.inverse().transform_point(iso.transform_point(p)), p));
        assert!(close(iso.inverse_transform_point(iso.transform_point(p)), p));
    }

    #[test]
    fn isometry_mul_matches_sequential_application() {
        let a = Isometry::new(
            Vector3::new(1.0, 0.0, 0.0),
            Quaternion::from_axis_angle(Vector3::UP, FRAC_PI_2),
        );
        let b = Isometry::new(
            Vector3::new(0.0, 2.0, 0.0),
            Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), 0.5),
        );
        let p = Vector3::new(0.5, -1.0, 2.0);
        assert!(close(
            a.mul(&b).transform_point(p),
            a.transform_point(b.transform_point(p))
        ));
    }

    #[test]
    fn mat3_inverse_round_trips() {
        let m = Mat3 {
            m: [4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0],
        };
        let id = m.mul(&m.inverse());
        for i in 0..9 {
            let want = if i % 4 == 0 { 1.0 } else { 0.0 };
            assert!((id.m[i] - want).abs() < 1e-4, "i={i} got {}", id.m[i]);
        }
    }

    #[test]
    fn mat3_singular_inverse_is_zero() {
        assert_eq!(Mat3::ZERO.inverse(), Mat3::ZERO);
    }

    #[test]
    fn skew_matches_cross_product() {
        let a = Vector3::new(1.0, -2.0, 3.0);
        let b = Vector3::new(-4.0, 0.5, 2.0);
        assert!(close(Mat3::skew(a).mul_vec(b), a.cross(b)));
    }

    #[test]
    fn from_quat_matches_apply_quaternion() {
        let q = Quaternion::from_euler_xyz(0.4, 1.2, -0.7).normalize();
        let v = Vector3::new(2.0, -1.0, 0.5);
        assert!(close(Mat3::from_quat(q).mul_vec(v), v.apply_quaternion(q)));
    }

    #[test]
    fn rotating_a_tensor_preserves_the_trace() {
        let i = Mat3::from_diagonal(Vector3::new(2.0, 5.0, 9.0));
        let q = Quaternion::from_euler_xyz(0.9, -0.2, 1.4).normalize();
        let r = i.rotated(q);
        let trace_of = |m: &Mat3| m.m[0] + m.m[4] + m.m[8];
        assert!((trace_of(&r) - trace_of(&i)).abs() < 1e-3);
    }

    #[test]
    fn orthonormal_basis_is_orthonormal_everywhere() {
        // Includes the (0, 0, -1) pole that the naive construction breaks on.
        let dirs = [
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-0.3, 0.5, -0.81).normalize(),
        ];
        for n in dirs {
            let (t1, t2) = orthonormal_basis(n);
            assert!(t1.dot(n).abs() < 1e-4, "t1·n for {n:?}");
            assert!(t2.dot(n).abs() < 1e-4, "t2·n for {n:?}");
            assert!(t1.dot(t2).abs() < 1e-4, "t1·t2 for {n:?}");
            assert!((t1.length() - 1.0).abs() < 1e-4);
            assert!((t2.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn segment_segment_handles_crossing_and_parallel() {
        let (_, _, c1, c2) = closest_points_segment_segment(
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        );
        assert!(close(c1, Vector3::ZERO));
        assert!(close(c2, Vector3::new(0.0, 0.0, 1.0)));

        let (_, _, c1, c2) = closest_points_segment_segment(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        );
        assert!((c1 - c2).length() - 1.0 < 1e-4);
    }

    #[test]
    fn closest_point_on_triangle_covers_all_regions() {
        let (a, b, c) = (
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        // Above the interior.
        assert!(close(
            closest_point_on_triangle(Vector3::new(0.25, 0.25, 3.0), a, b, c),
            Vector3::new(0.25, 0.25, 0.0)
        ));
        // Past vertex a.
        assert!(close(
            closest_point_on_triangle(Vector3::new(-2.0, -2.0, 0.0), a, b, c),
            a
        ));
        // Past vertex b, and past vertex c.
        assert!(close(
            closest_point_on_triangle(Vector3::new(4.0, -1.0, 0.0), a, b, c),
            b
        ));
        assert!(close(
            closest_point_on_triangle(Vector3::new(-1.0, 4.0, 0.0), a, b, c),
            c
        ));
        // Beyond edge bc.
        assert!(close(
            closest_point_on_triangle(Vector3::new(1.0, 1.0, 0.0), a, b, c),
            Vector3::new(0.5, 0.5, 0.0)
        ));
    }
}
