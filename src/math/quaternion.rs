use super::Vector3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Quaternion {
    pub const fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_axis_angle(axis: Vector3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: half.cos(),
        }
    }

    /// Three.js XYZ Euler-to-quaternion order.
    pub fn from_euler_xyz(x: f32, y: f32, z: f32) -> Self {
        let (c1, s1) = ((x / 2.0).cos(), (x / 2.0).sin());
        let (c2, s2) = ((y / 2.0).cos(), (y / 2.0).sin());
        let (c3, s3) = ((z / 2.0).cos(), (z / 2.0).sin());
        Self {
            x: s1 * c2 * c3 + c1 * s2 * s3,
            y: c1 * s2 * c3 - s1 * c2 * s3,
            z: c1 * c2 * s3 + s1 * s2 * c3,
            w: c1 * c2 * c3 - s1 * s2 * s3,
        }
    }

    pub fn multiply(&self, other: Self) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    pub fn normalize(&self) -> Self {
        let len_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if len_sq == 0.0 {
            Self::identity()
        } else {
            let inv = 1.0 / len_sq.sqrt();
            Self {
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
                w: self.w * inv,
            }
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// For a unit quaternion, invert == conjugate. Matches three.js.
    pub fn invert(&self) -> Self {
        self.conjugate()
    }

    /// `q' = other * self` (pre-multiply on the left).
    pub fn premultiply(&self, other: Self) -> Self {
        other.multiply(*self)
    }

    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Spherical linear interpolation. Matches three.js's `Quaternion.slerp`.
    pub fn slerp(&self, other: Self, t: f32) -> Self {
        if t == 0.0 {
            return *self;
        }
        if t == 1.0 {
            return other;
        }
        let mut cos_half = self.dot(other);
        let mut q = other;
        if cos_half < 0.0 {
            q = Self {
                x: -other.x,
                y: -other.y,
                z: -other.z,
                w: -other.w,
            };
            cos_half = -cos_half;
        }
        if cos_half >= 1.0 - 1e-6 {
            return Self {
                x: self.x + (q.x - self.x) * t,
                y: self.y + (q.y - self.y) * t,
                z: self.z + (q.z - self.z) * t,
                w: self.w + (q.w - self.w) * t,
            }
            .normalize();
        }
        let sin_half_sq = 1.0 - cos_half * cos_half;
        let sin_half = sin_half_sq.sqrt();
        let half = sin_half.atan2(cos_half);
        let ratio_a = ((1.0 - t) * half).sin() / sin_half;
        let ratio_b = (t * half).sin() / sin_half;
        Self {
            x: self.x * ratio_a + q.x * ratio_b,
            y: self.y * ratio_a + q.y * ratio_b,
            z: self.z * ratio_a + q.z * ratio_b,
            w: self.w * ratio_a + q.w * ratio_b,
        }
    }

    /// Extract a unit quaternion from a Matrix4 (assumes its upper-left 3x3 is a
    /// pure rotation). Mirrors three.js's `Quaternion.setFromRotationMatrix`.
    pub fn from_rotation_matrix(m: &super::Matrix4) -> Self {
        let te = &m.elements;
        let m11 = te[0];
        let m12 = te[4];
        let m13 = te[8];
        let m21 = te[1];
        let m22 = te[5];
        let m23 = te[9];
        let m31 = te[2];
        let m32 = te[6];
        let m33 = te[10];
        let trace = m11 + m22 + m33;
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Self {
                x: (m32 - m23) * s,
                y: (m13 - m31) * s,
                z: (m21 - m12) * s,
                w: 0.25 / s,
            }
        } else if m11 > m22 && m11 > m33 {
            let s = 2.0 * (1.0 + m11 - m22 - m33).sqrt();
            Self {
                x: 0.25 * s,
                y: (m12 + m21) / s,
                z: (m13 + m31) / s,
                w: (m32 - m23) / s,
            }
        } else if m22 > m33 {
            let s = 2.0 * (1.0 + m22 - m11 - m33).sqrt();
            Self {
                x: (m12 + m21) / s,
                y: 0.25 * s,
                z: (m23 + m32) / s,
                w: (m13 - m31) / s,
            }
        } else {
            let s = 2.0 * (1.0 + m33 - m11 - m22).sqrt();
            Self {
                x: (m13 + m31) / s,
                y: (m23 + m32) / s,
                z: 0.25 * s,
                w: (m21 - m12) / s,
            }
        }
    }
}
/// Composition, so rotations read the way they are spoken about.
///
/// `Vector3` has carried `Add`, `Sub`, `Mul<f32>` and `Neg` all along while the
/// quaternion had only the named method, so composing three rotations came out
/// as `a.multiply(b).multiply(c)` — the one place in the crate where the maths
/// has to be written inside out. Same order as [`multiply`](Quaternion::multiply)
/// and as three.js: `a * b` applies `b` first, then `a`.
impl std::ops::Mul for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.multiply(rhs)
    }
}

/// Rotating a vector, the same way round: `q * v`.
impl std::ops::Mul<Vector3> for Quaternion {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Vector3 {
        rhs.apply_quaternion(self)
    }
}

#[cfg(test)]
mod op_tests {
    use super::*;

    #[test]
    fn mul_is_multiply_and_composes_right_to_left() {
        let a = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.7);
        let b = Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -0.3);
        let (m, o) = (a.multiply(b), a * b);
        for (l, r) in [(m.x, o.x), (m.y, o.y), (m.z, o.z), (m.w, o.w)] {
            assert!((l - r).abs() < 1e-6, "operator must match multiply");
        }
        // Right-to-left: applying the product must equal applying b then a.
        let v = Vector3::new(0.3, -0.6, 0.8);
        let stepwise = v.apply_quaternion(b).apply_quaternion(a);
        let combined = (a * b) * v;
        for (l, r) in [
            (stepwise.x, combined.x),
            (stepwise.y, combined.y),
            (stepwise.z, combined.z),
        ] {
            assert!(
                (l - r).abs() < 1e-6,
                "a * b must apply b first: {stepwise:?} vs {combined:?}"
            );
        }
    }

    #[test]
    fn mul_vector_matches_apply_quaternion() {
        let q = Quaternion::from_euler_xyz(0.4, -0.9, 0.2);
        let v = Vector3::new(1.0, 2.0, -3.0);
        let (a, b) = (v.apply_quaternion(q), q * v);
        for (l, r) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
            assert!((l - r).abs() < 1e-6);
        }
    }
}
