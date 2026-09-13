use super::{CubicBezierCurve, Curve2, CurvePath, EllipseCurve, LineCurve, QuadraticBezierCurve};
use crate::math::Vector2;

/// 2D path with a "pen" cursor. Mirrors three.js's `Path` build-up API:
/// `move_to`, `line_to`, `bezier_curve_to`, `quadratic_curve_to`, `arc`, etc.
pub struct Path {
    pub current: Vector2,
    pub curve_path: CurvePath,
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Path")
            .field("current", &self.current)
            .field("curve_path", &self.curve_path)
            .finish()
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

impl Path {
    pub fn new() -> Self {
        Self {
            current: Vector2::ZERO,
            curve_path: CurvePath::new(),
        }
    }

    pub fn move_to(&mut self, p: Vector2) -> &mut Self {
        self.current = p;
        self
    }

    pub fn line_to(&mut self, p: Vector2) -> &mut Self {
        let c = LineCurve::new(self.current, p);
        self.curve_path.add(Box::new(c));
        self.current = p;
        self
    }

    pub fn quadratic_curve_to(&mut self, cp: Vector2, p: Vector2) -> &mut Self {
        let c = QuadraticBezierCurve::new(self.current, cp, p);
        self.curve_path.add(Box::new(c));
        self.current = p;
        self
    }

    pub fn bezier_curve_to(&mut self, cp1: Vector2, cp2: Vector2, p: Vector2) -> &mut Self {
        let c = CubicBezierCurve::new(self.current, cp1, cp2, p);
        self.curve_path.add(Box::new(c));
        self.current = p;
        self
    }

    pub fn arc(
        &mut self,
        center: Vector2,
        radius: f32,
        a_start: f32,
        a_end: f32,
        clockwise: bool,
    ) -> &mut Self {
        let c = EllipseCurve::new(center, radius, radius, a_start, a_end, clockwise, 0.0);
        self.curve_path.add(Box::new(c));
        self.current = c.get_point(1.0);
        self
    }

    pub fn get_points(&self, divisions: usize) -> Vec<Vector2> {
        self.curve_path
            .get_points(divisions * self.curve_path.curves.len().max(1))
    }
}
