use super::Curve2;
use crate::math::Vector2;

/// 2D Catmull-Rom spline. Matches three.js's `SplineCurve`.
#[derive(Debug, Clone)]
pub struct SplineCurve {
    pub points: Vec<Vector2>,
}

impl SplineCurve {
    pub fn new(points: Vec<Vector2>) -> Self {
        Self { points }
    }
}

impl Curve2 for SplineCurve {
    fn get_point(&self, t: f32) -> Vector2 {
        let n = self.points.len();
        if n == 0 {
            return Vector2::ZERO;
        }
        if n == 1 {
            return self.points[0];
        }
        let segments = n - 1;
        let p = t * segments as f32;
        let i = (p.floor() as usize).min(segments - 1);
        let weight = p - i as f32;
        let idx = |k: i32| -> usize { (i as i32 + k).clamp(0, n as i32 - 1) as usize };
        let p0 = self.points[idx(-1)];
        let p1 = self.points[idx(0)];
        let p2 = self.points[idx(1)];
        let p3 = self.points[idx(2)];
        catmull_rom(p0, p1, p2, p3, weight, 0.5)
    }

    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        Some(super::svg_path::spline_segments(self))
    }
}

fn catmull_rom(
    p0: Vector2,
    p1: Vector2,
    p2: Vector2,
    p3: Vector2,
    t: f32,
    tension: f32,
) -> Vector2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let m1 = (p2 - p0) * tension;
    let m2 = (p3 - p1) * tension;
    p1 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + p2 * (-2.0 * t3 + 3.0 * t2)
        + m1 * (t3 - 2.0 * t2 + t)
        + m2 * (t3 - t2)
}
