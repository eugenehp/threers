use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn set(&mut self, x: f32, y: f32) -> &mut Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
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
        self.x * other.x + self.y * other.y
    }

    /// 2D "cross" — returns the scalar z-component of the 3D cross product.
    pub fn cross(&self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    pub fn distance_to(&self, other: Self) -> f32 {
        (*self - other).length()
    }

    pub fn distance_to_sq(&self, other: Self) -> f32 {
        (*self - other).length_sq()
    }

    pub fn lerp(&self, other: Self, t: f32) -> Self {
        *self + (other - *self) * t
    }

    pub fn angle(&self) -> f32 {
        self.y.atan2(self.x)
    }

    pub fn rotate_around(&self, center: Self, angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let x = self.x - center.x;
        let y = self.y - center.y;
        Self::new(x * c - y * s + center.x, x * s + y * c + center.y)
    }

    pub fn min(&self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    pub fn max(&self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self::new(self.x.clamp(min.x, max.x), self.y.clamp(min.y, max.y))
    }

    pub fn to_array(&self) -> [f32; 2] {
        [self.x, self.y]
    }
}

impl Add for Vector2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vector2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f32> for Vector2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Neg for Vector2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit_length() {
        let v = Vector2::new(3.0, 4.0).normalize();
        assert!((v.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cross_z_component() {
        let a = Vector2::new(1.0, 0.0);
        let b = Vector2::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotate_around_origin_90() {
        let p = Vector2::new(1.0, 0.0);
        let r = p.rotate_around(Vector2::ZERO, std::f32::consts::FRAC_PI_2);
        assert!((r.x - 0.0).abs() < 1e-6);
        assert!((r.y - 1.0).abs() < 1e-6);
    }
}
