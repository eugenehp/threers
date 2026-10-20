//! three.js `CurvePath`, `Path` and `Shape`.

use super::curve::{finite_difference_tangent, lengths_of, u_to_t, ShapeCurve, Vec2d};

/// three.js r165 `Path` (a `CurvePath` with a pen): a sequence of 2D curves
/// sampled exactly as three.js samples it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    pub curves: Vec<ShapeCurve>,
    /// `autoClose`: `getPoints` / `getSpacedPoints` repeat the first point at
    /// the end.
    pub auto_close: bool,
    /// `currentPoint`: where the next drawing command starts.
    pub current_point: Vec2d,
}

impl Path {
    pub fn new() -> Self {
        Self::default()
    }

    /// `new Path(points)` / `setFromPoints`: a `moveTo` then `lineTo`s.
    pub fn from_points(points: &[Vec2d]) -> Self {
        let mut p = Self::new();
        p.set_from_points(points);
        p
    }

    pub fn set_from_points(&mut self, points: &[Vec2d]) -> &mut Self {
        if let Some(first) = points.first() {
            self.move_to(first.x, first.y);
            for q in &points[1..] {
                self.line_to(q.x, q.y);
            }
        }
        self
    }

    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.current_point = Vec2d::new(x, y);
        self
    }

    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        let to = Vec2d::new(x, y);
        self.curves.push(ShapeCurve::Line {
            v1: self.current_point,
            v2: to,
        });
        self.current_point = to;
        self
    }

    pub fn quadratic_curve_to(&mut self, cpx: f64, cpy: f64, x: f64, y: f64) -> &mut Self {
        let to = Vec2d::new(x, y);
        self.curves.push(ShapeCurve::QuadraticBezier {
            v0: self.current_point,
            v1: Vec2d::new(cpx, cpy),
            v2: to,
        });
        self.current_point = to;
        self
    }

    pub fn bezier_curve_to(
        &mut self,
        cp1x: f64,
        cp1y: f64,
        cp2x: f64,
        cp2y: f64,
        x: f64,
        y: f64,
    ) -> &mut Self {
        let to = Vec2d::new(x, y);
        self.curves.push(ShapeCurve::CubicBezier {
            v0: self.current_point,
            v1: Vec2d::new(cp1x, cp1y),
            v2: Vec2d::new(cp2x, cp2y),
            v3: to,
        });
        self.current_point = to;
        self
    }

    /// `splineThru(pts)`: a Catmull–Rom spline from the pen through `pts`.
    pub fn spline_thru(&mut self, pts: &[Vec2d]) -> &mut Self {
        let Some(&last) = pts.last() else {
            return self;
        };
        let mut points = Vec::with_capacity(pts.len() + 1);
        points.push(self.current_point);
        points.extend_from_slice(pts);
        self.curves.push(ShapeCurve::Spline { points });
        self.current_point = last;
        self
    }

    /// `arc(aX, aY, …)`: centre relative to the pen.
    pub fn arc(
        &mut self,
        ax: f64,
        ay: f64,
        radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
    ) -> &mut Self {
        let (x0, y0) = (self.current_point.x, self.current_point.y);
        self.absarc(ax + x0, ay + y0, radius, start, end, clockwise)
    }

    pub fn absarc(
        &mut self,
        ax: f64,
        ay: f64,
        radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
    ) -> &mut Self {
        self.absellipse(ax, ay, radius, radius, start, end, clockwise, 0.0)
    }

    /// `ellipse(…)`: centre relative to the pen.
    #[allow(clippy::too_many_arguments)]
    pub fn ellipse(
        &mut self,
        ax: f64,
        ay: f64,
        x_radius: f64,
        y_radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
        rotation: f64,
    ) -> &mut Self {
        let (x0, y0) = (self.current_point.x, self.current_point.y);
        self.absellipse(
            ax + x0,
            ay + y0,
            x_radius,
            y_radius,
            start,
            end,
            clockwise,
            rotation,
        )
    }

    /// `absellipse(…)`: joined to the previous curve by a line when it does
    /// not start where the pen is.
    #[allow(clippy::too_many_arguments)]
    pub fn absellipse(
        &mut self,
        ax: f64,
        ay: f64,
        x_radius: f64,
        y_radius: f64,
        start: f64,
        end: f64,
        clockwise: bool,
        rotation: f64,
    ) -> &mut Self {
        let curve = ShapeCurve::Ellipse {
            x: ax,
            y: ay,
            x_radius,
            y_radius,
            start_angle: start,
            end_angle: end,
            clockwise,
            rotation,
        };
        if !self.curves.is_empty() {
            let first = curve.get_point(0.0);
            if first != self.current_point {
                self.line_to(first.x, first.y);
            }
        }
        let last = curve.get_point(1.0);
        self.curves.push(curve);
        self.current_point = last;
        self
    }

    /// `CurvePath.add(curve)`: append as is, the pen untouched.
    pub fn add(&mut self, curve: ShapeCurve) -> &mut Self {
        self.curves.push(curve);
        self
    }

    /// `CurvePath.closePath()`: a line back to the start when the ends differ.
    /// Like three.js, the pen does not move.
    pub fn close_path(&mut self) -> &mut Self {
        if let (Some(first), Some(last)) = (self.curves.first(), self.curves.last()) {
            let start = first.get_point(0.0);
            let end = last.get_point(1.0);
            if start != end {
                self.curves.push(ShapeCurve::Line { v1: end, v2: start });
            }
        }
        self
    }

    /// `getCurveLengths()`: cumulative lengths of the curves.
    pub fn get_curve_lengths(&self) -> Vec<f64> {
        let mut sums = 0.0;
        self.curves
            .iter()
            .map(|c| {
                sums += c.get_length();
                sums
            })
            .collect()
    }

    /// `getLength()`.
    pub fn get_length(&self) -> f64 {
        self.get_curve_lengths().last().copied().unwrap_or(0.0)
    }

    fn point_with(&self, t: f64, curve_lengths: &[f64], total: f64) -> Option<Vec2d> {
        let d = t * total;
        for (i, &len) in curve_lengths.iter().enumerate() {
            if len >= d {
                let diff = len - d;
                let curve = &self.curves[i];
                let segment_length = curve.get_length();
                let u = if segment_length == 0.0 {
                    0.0
                } else {
                    1.0 - diff / segment_length
                };
                return Some(curve.get_point_at(u));
            }
        }
        None
    }

    /// `getPoint(t)`: by distance along the whole path. `None` where three.js
    /// returns `null` (an empty path, or `t` past the end).
    pub fn get_point(&self, t: f64) -> Option<Vec2d> {
        let lengths = self.get_curve_lengths();
        let total = lengths.last().copied().unwrap_or(0.0);
        self.point_with(t, &lengths, total)
    }

    /// `getLengths(divisions)` of the whole path (its arc-length table).
    pub fn get_lengths(&self, divisions: usize) -> Vec<f64> {
        let lengths = self.get_curve_lengths();
        let total = lengths.last().copied().unwrap_or(0.0);
        lengths_of(
            |t| self.point_with(t, &lengths, total).unwrap_or_default(),
            divisions,
        )
    }

    /// `getPointAt(u)`.
    pub fn get_point_at(&self, u: f64) -> Option<Vec2d> {
        let t = u_to_t(&self.get_lengths(super::curve::ARC_LENGTH_DIVISIONS), u);
        self.get_point(t)
    }

    /// `getTangent(t)` (the finite difference of `Curve`).
    pub fn get_tangent(&self, t: f64) -> Vec2d {
        finite_difference_tangent(|t| self.get_point(t).unwrap_or_default(), t)
    }

    /// `getTangentAt(u)`.
    pub fn get_tangent_at(&self, u: f64) -> Vec2d {
        let t = u_to_t(&self.get_lengths(super::curve::ARC_LENGTH_DIVISIONS), u);
        self.get_tangent(t)
    }

    /// `getSpacedPoints(divisions)` (40 in three.js by default).
    pub fn get_spaced_points(&self, divisions: usize) -> Vec<Vec2d> {
        let lengths = self.get_curve_lengths();
        let total = lengths.last().copied().unwrap_or(0.0);
        let mut points: Vec<Vec2d> = (0..=divisions)
            .map(|i| {
                self.point_with(i as f64 / divisions as f64, &lengths, total)
                    .unwrap_or_default()
            })
            .collect();
        if self.auto_close {
            points.push(points[0]);
        }
        points
    }

    /// `getPoints(divisions)` (12 in three.js by default): each curve at its
    /// own resolution, consecutive duplicates dropped, closed when
    /// `auto_close`.
    pub fn get_points(&self, divisions: usize) -> Vec<Vec2d> {
        let mut points: Vec<Vec2d> = Vec::new();
        let mut last: Option<Vec2d> = None;
        for curve in &self.curves {
            for point in curve.get_points(curve.path_resolution(divisions)) {
                if last == Some(point) {
                    continue;
                }
                points.push(point);
                last = Some(point);
            }
        }
        if self.auto_close && points.len() > 1 && points[points.len() - 1] != points[0] {
            points.push(points[0]);
        }
        points
    }
}

/// three.js `Shape`: a [`Path`] outline plus hole paths.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Shape {
    pub path: Path,
    pub holes: Vec<Path>,
}

/// `Shape.extractPoints`: the outline's and every hole's `getPoints`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapePoints {
    pub shape: Vec<Vec2d>,
    pub holes: Vec<Vec<Vec2d>>,
}

impl Shape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_path(path: Path) -> Self {
        Self {
            path,
            holes: Vec::new(),
        }
    }

    /// `new Shape(points)`.
    pub fn from_points(points: &[Vec2d]) -> Self {
        Self::from_path(Path::from_points(points))
    }

    /// `getPointsHoles(divisions)`.
    pub fn get_points_holes(&self, divisions: usize) -> Vec<Vec<Vec2d>> {
        self.holes.iter().map(|h| h.get_points(divisions)).collect()
    }

    /// `extractPoints(divisions)`.
    pub fn extract_points(&self, divisions: usize) -> ShapePoints {
        ShapePoints {
            shape: self.path.get_points(divisions),
            holes: self.get_points_holes(divisions),
        }
    }
}

impl std::ops::Deref for Shape {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.path
    }
}

impl std::ops::DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Path {
        &mut self.path
    }
}
