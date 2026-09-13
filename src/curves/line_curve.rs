use super::{Curve2, Curve3};
use crate::math::{Vector2, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct LineCurve {
    pub v1: Vector2,
    pub v2: Vector2,
}

impl LineCurve {
    pub const fn new(v1: Vector2, v2: Vector2) -> Self {
        Self { v1, v2 }
    }
}

impl Curve2 for LineCurve {
    fn get_point(&self, t: f32) -> Vector2 {
        self.v1.lerp(self.v2, t)
    }

    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        Some(super::svg_path::line_segments(self))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LineCurve3 {
    pub v1: Vector3,
    pub v2: Vector3,
}

impl LineCurve3 {
    pub const fn new(v1: Vector3, v2: Vector3) -> Self {
        Self { v1, v2 }
    }
}

impl Curve3 for LineCurve3 {
    fn get_point(&self, t: f32) -> Vector3 {
        self.v1.lerp(self.v2, t)
    }
}
