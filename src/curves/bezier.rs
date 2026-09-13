use super::{Curve2, Curve3};
use crate::math::{Vector2, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct QuadraticBezierCurve {
    pub v0: Vector2,
    pub v1: Vector2,
    pub v2: Vector2,
}
impl QuadraticBezierCurve {
    pub const fn new(v0: Vector2, v1: Vector2, v2: Vector2) -> Self {
        Self { v0, v1, v2 }
    }
}
impl Curve2 for QuadraticBezierCurve {
    fn get_point(&self, t: f32) -> Vector2 {
        quad(self.v0, self.v1, self.v2, t)
    }

    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        Some(super::svg_path::quadratic_segments(self))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuadraticBezierCurve3 {
    pub v0: Vector3,
    pub v1: Vector3,
    pub v2: Vector3,
}
impl QuadraticBezierCurve3 {
    pub const fn new(v0: Vector3, v1: Vector3, v2: Vector3) -> Self {
        Self { v0, v1, v2 }
    }
}
impl Curve3 for QuadraticBezierCurve3 {
    fn get_point(&self, t: f32) -> Vector3 {
        quad3(self.v0, self.v1, self.v2, t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CubicBezierCurve {
    pub v0: Vector2,
    pub v1: Vector2,
    pub v2: Vector2,
    pub v3: Vector2,
}
impl CubicBezierCurve {
    pub const fn new(v0: Vector2, v1: Vector2, v2: Vector2, v3: Vector2) -> Self {
        Self { v0, v1, v2, v3 }
    }
}
impl Curve2 for CubicBezierCurve {
    fn get_point(&self, t: f32) -> Vector2 {
        cubic(self.v0, self.v1, self.v2, self.v3, t)
    }

    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        Some(super::svg_path::cubic_segments(self))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CubicBezierCurve3 {
    pub v0: Vector3,
    pub v1: Vector3,
    pub v2: Vector3,
    pub v3: Vector3,
}
impl CubicBezierCurve3 {
    pub const fn new(v0: Vector3, v1: Vector3, v2: Vector3, v3: Vector3) -> Self {
        Self { v0, v1, v2, v3 }
    }
}
impl Curve3 for CubicBezierCurve3 {
    fn get_point(&self, t: f32) -> Vector3 {
        cubic3(self.v0, self.v1, self.v2, self.v3, t)
    }
}

fn quad(a: Vector2, b: Vector2, c: Vector2, t: f32) -> Vector2 {
    let inv = 1.0 - t;
    a * (inv * inv) + b * (2.0 * inv * t) + c * (t * t)
}
fn quad3(a: Vector3, b: Vector3, c: Vector3, t: f32) -> Vector3 {
    let inv = 1.0 - t;
    a * (inv * inv) + b * (2.0 * inv * t) + c * (t * t)
}
fn cubic(a: Vector2, b: Vector2, c: Vector2, d: Vector2, t: f32) -> Vector2 {
    let inv = 1.0 - t;
    a * (inv * inv * inv) + b * (3.0 * inv * inv * t) + c * (3.0 * inv * t * t) + d * (t * t * t)
}
fn cubic3(a: Vector3, b: Vector3, c: Vector3, d: Vector3, t: f32) -> Vector3 {
    let inv = 1.0 - t;
    a * (inv * inv * inv) + b * (3.0 * inv * inv * t) + c * (3.0 * inv * t * t) + d * (t * t * t)
}
