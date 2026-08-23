use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    /// `+X`, the other half of three.js's right-handed Y-up frame.
    pub const RIGHT: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    /// `+Z` — toward the viewer in the default camera, as in three.js.
    pub const FORWARD: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn set(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::ZERO
        } else {
            *self * (1.0 / len)
        }
    }

    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn distance_to(&self, other: Self) -> f32 {
        (*self - other).length()
    }

    pub fn lerp(&self, other: Self, t: f32) -> Self {
        *self + (other - *self) * t
    }

    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn min(&self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    pub fn max(&self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.clamp(min.x, max.x),
            self.y.clamp(min.y, max.y),
            self.z.clamp(min.z, max.z),
        )
    }

    /// Rotate this vector by a unit quaternion. Matches three.js's `applyQuaternion`.
    pub fn apply_quaternion(&self, q: crate::math::Quaternion) -> Self {
        let ix = q.w * self.x + q.y * self.z - q.z * self.y;
        let iy = q.w * self.y + q.z * self.x - q.x * self.z;
        let iz = q.w * self.z + q.x * self.y - q.y * self.x;
        let iw = -q.x * self.x - q.y * self.y - q.z * self.z;
        Self::new(
            ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
            iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
            iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
        )
    }

    /// Transform as a position (homogeneous w=1). Matches three.js's `applyMatrix4`.
    pub fn apply_matrix4(&self, m: &crate::math::Matrix4) -> Self {
        let e = &m.elements;
        let w = e[3] * self.x + e[7] * self.y + e[11] * self.z + e[15];
        let inv_w = if w == 0.0 { 1.0 } else { 1.0 / w };
        Self::new(
            (e[0] * self.x + e[4] * self.y + e[8] * self.z + e[12]) * inv_w,
            (e[1] * self.x + e[5] * self.y + e[9] * self.z + e[13]) * inv_w,
            (e[2] * self.x + e[6] * self.y + e[10] * self.z + e[14]) * inv_w,
        )
    }

    /// Project from NDC (or any inverse-projection matrix) back to world space.
    pub fn unproject(
        &self,
        view: &crate::math::Matrix4,
        projection: &crate::math::Matrix4,
    ) -> Self {
        let inv_proj = projection.invert();
        let inv_view = view.invert();
        self.apply_matrix4(&inv_proj).apply_matrix4(&inv_view)
    }
}

impl Add for Vector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for Vector3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_product_right_handed() {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vector3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn normalize_unit_length() {
        let v = Vector3::new(3.0, 0.0, 4.0).normalize();
        assert!((v.length() - 1.0).abs() < 1e-6);
    }
}
