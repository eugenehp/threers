//! The 2D curves a three.js `Path` is built from, and the arc-length machinery
//! of three.js's `Curve` base class, operation for operation in `f64`.

/// A 2D point in JavaScript number precision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2d {
    pub x: f64,
    pub y: f64,
}

impl Vec2d {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, o: Vec2d) -> f64 {
        let dx = self.x - o.x;
        let dy = self.y - o.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// `Vector2.normalize`: a multiply by `1 / (length || 1)`.
    pub fn normalize(self) -> Vec2d {
        let l = (self.x * self.x + self.y * self.y).sqrt();
        let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
        Vec2d::new(self.x * inv, self.y * inv)
    }
}

/// three.js `Curve.arcLengthDivisions`.
pub const ARC_LENGTH_DIVISIONS: usize = 200;

/// One curve of a path: three.js's `LineCurve`, `QuadraticBezierCurve`,
/// `CubicBezierCurve`, `SplineCurve` and `EllipseCurve` (`ArcCurve` is an
/// ellipse with equal radii).
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeCurve {
    Line {
        v1: Vec2d,
        v2: Vec2d,
    },
    QuadraticBezier {
        v0: Vec2d,
        v1: Vec2d,
        v2: Vec2d,
    },
    CubicBezier {
        v0: Vec2d,
        v1: Vec2d,
        v2: Vec2d,
        v3: Vec2d,
    },
    Spline {
        points: Vec<Vec2d>,
    },
    Ellipse {
        x: f64,
        y: f64,
        x_radius: f64,
        y_radius: f64,
        start_angle: f64,
        end_angle: f64,
        clockwise: bool,
        rotation: f64,
    },
}

/// three.js `Interpolations.CatmullRom`: the uniform Catmull-Rom spline
/// through `p1` (at `t = 0`) and `p2` (at `t = 1`), shaped by `p0` and `p3`.
pub fn catmull_rom(t: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let v0 = (p2 - p0) * 0.5;
    let v1 = (p3 - p1) * 0.5;
    let t2 = t * t;
    let t3 = t * t2;
    (2.0 * p1 - 2.0 * p2 + v0 + v1) * t3 + (-3.0 * p1 + 3.0 * p2 - 2.0 * v0 - v1) * t2 + v0 * t + p1
}

/// three.js `Interpolations.QuadraticBezier`: the quadratic Bézier with
/// endpoints `p0`, `p2` and control point `p1`, at `t` in `0..=1`.
pub fn quadratic_bezier(t: f64, p0: f64, p1: f64, p2: f64) -> f64 {
    let k = 1.0 - t;
    k * k * p0 + 2.0 * (1.0 - t) * t * p1 + t * t * p2
}

/// three.js `Interpolations.CubicBezier`: the cubic Bézier with endpoints
/// `p0`, `p3` and control points `p1`, `p2`, at `t` in `0..=1`.
pub fn cubic_bezier(t: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let k = 1.0 - t;
    k * k * k * p0 + 3.0 * k * k * t * p1 + 3.0 * (1.0 - t) * t * t * p2 + t * t * t * p3
}

/// The cumulative arc lengths at `divisions + 1` evenly spaced `t`, as
/// `Curve.getLengths` measures them.
pub(crate) fn lengths_of(get_point: impl Fn(f64) -> Vec2d, divisions: usize) -> Vec<f64> {
    let mut cache = Vec::with_capacity(divisions + 1);
    let mut last = get_point(0.0);
    let mut sum = 0.0;
    cache.push(0.0);
    for p in 1..=divisions {
        let current = get_point(p as f64 / divisions as f64);
        sum += current.distance_to(last);
        cache.push(sum);
        last = current;
    }
    cache
}

/// `Curve.getUtoTmapping(u)` over precomputed `getLengths()`.
pub(crate) fn u_to_t(arc_lengths: &[f64], u: f64) -> f64 {
    let il = arc_lengths.len();
    if il < 2 {
        return u;
    }
    let target = u * arc_lengths[il - 1];
    let (mut low, mut high) = (0_isize, il as isize - 1);
    while low <= high {
        let i = low + (high - low) / 2;
        let comparison = arc_lengths[i as usize] - target;
        if comparison < 0.0 {
            low = i + 1;
        } else if comparison > 0.0 {
            high = i - 1;
        } else {
            high = i;
            break;
        }
    }
    let i = high.max(0) as usize;
    if arc_lengths[i] == target {
        return i as f64 / (il - 1) as f64;
    }
    let before = arc_lengths[i];
    let after = arc_lengths[(i + 1).min(il - 1)];
    let fraction = (target - before) / (after - before);
    (i as f64 + fraction) / (il - 1) as f64
}

/// `Curve.getTangent`: the difference of two points `0.0001` either side,
/// clamped to the curve.
pub(crate) fn finite_difference_tangent(get_point: impl Fn(f64) -> Vec2d, t: f64) -> Vec2d {
    let delta = 0.0001;
    let t1 = (t - delta).max(0.0);
    let t2 = (t + delta).min(1.0);
    let p1 = get_point(t1);
    let p2 = get_point(t2);
    Vec2d::new(p2.x - p1.x, p2.y - p1.y).normalize()
}

impl ShapeCurve {
    /// `getPoint(t)`.
    pub fn get_point(&self, t: f64) -> Vec2d {
        match self {
            ShapeCurve::Line { v1, v2 } => {
                if t == 1.0 {
                    *v2
                } else {
                    Vec2d::new((v2.x - v1.x) * t + v1.x, (v2.y - v1.y) * t + v1.y)
                }
            }
            ShapeCurve::QuadraticBezier { v0, v1, v2 } => Vec2d::new(
                quadratic_bezier(t, v0.x, v1.x, v2.x),
                quadratic_bezier(t, v0.y, v1.y, v2.y),
            ),
            ShapeCurve::CubicBezier { v0, v1, v2, v3 } => Vec2d::new(
                cubic_bezier(t, v0.x, v1.x, v2.x, v3.x),
                cubic_bezier(t, v0.y, v1.y, v2.y, v3.y),
            ),
            ShapeCurve::Spline { points } => {
                let n = points.len();
                if n == 0 {
                    return Vec2d::default();
                }
                let p = (n - 1) as f64 * t;
                let int_point = p.floor();
                let weight = p - int_point;
                let ip = int_point.max(0.0) as usize;
                let ip = ip.min(n - 1);
                let p0 = points[if ip == 0 { 0 } else { ip - 1 }];
                let p1 = points[ip];
                let p2 = points[if ip + 2 > n {
                    n - 1
                } else {
                    (ip + 1).min(n - 1)
                }];
                let p3 = points[if ip + 3 > n { n - 1 } else { ip + 2 }];
                Vec2d::new(
                    catmull_rom(weight, p0.x, p1.x, p2.x, p3.x),
                    catmull_rom(weight, p0.y, p1.y, p2.y, p3.y),
                )
            }
            ShapeCurve::Ellipse {
                x: ax,
                y: ay,
                x_radius,
                y_radius,
                start_angle,
                end_angle,
                clockwise,
                rotation,
            } => {
                let two_pi = std::f64::consts::PI * 2.0;
                let mut delta = end_angle - start_angle;
                let same_points = delta.abs() < f64::EPSILON;
                while delta < 0.0 {
                    delta += two_pi;
                }
                while delta > two_pi {
                    delta -= two_pi;
                }
                if delta < f64::EPSILON {
                    delta = if same_points { 0.0 } else { two_pi };
                }
                if *clockwise && !same_points {
                    delta = if delta == two_pi {
                        -two_pi
                    } else {
                        delta - two_pi
                    };
                }
                let angle = start_angle + t * delta;
                let mut x = ax + x_radius * angle.cos();
                let mut y = ay + y_radius * angle.sin();
                if *rotation != 0.0 {
                    let cos = rotation.cos();
                    let sin = rotation.sin();
                    let tx = x - ax;
                    let ty = y - ay;
                    x = tx * cos - ty * sin + ax;
                    y = tx * sin + ty * cos + ay;
                }
                Vec2d::new(x, y)
            }
        }
    }

    /// `getPoints(divisions)`: `divisions + 1` points at even `t`.
    pub fn get_points(&self, divisions: usize) -> Vec<Vec2d> {
        (0..=divisions)
            .map(|d| self.get_point(d as f64 / divisions as f64))
            .collect()
    }

    /// `getLengths(divisions)`.
    pub fn get_lengths(&self, divisions: usize) -> Vec<f64> {
        lengths_of(|t| self.get_point(t), divisions)
    }

    /// `getLength()` — measured over `arcLengthDivisions` (200) chords, even
    /// for a line, as three.js does.
    pub fn get_length(&self) -> f64 {
        *self
            .get_lengths(ARC_LENGTH_DIVISIONS)
            .last()
            .unwrap_or(&0.0)
    }

    /// `getUtoTmapping(u)`.
    pub fn u_to_t(&self, u: f64) -> f64 {
        u_to_t(&self.get_lengths(ARC_LENGTH_DIVISIONS), u)
    }

    /// `getPointAt(u)`: by arc length (a line's `t` already is).
    pub fn get_point_at(&self, u: f64) -> Vec2d {
        match self {
            ShapeCurve::Line { .. } => self.get_point(u),
            _ => self.get_point(self.u_to_t(u)),
        }
    }

    /// `getTangent(t)`.
    pub fn get_tangent(&self, t: f64) -> Vec2d {
        match self {
            ShapeCurve::Line { v1, v2 } => Vec2d::new(v2.x - v1.x, v2.y - v1.y).normalize(),
            _ => finite_difference_tangent(|t| self.get_point(t), t),
        }
    }

    /// `getTangentAt(u)`.
    pub fn get_tangent_at(&self, u: f64) -> Vec2d {
        match self {
            ShapeCurve::Line { .. } => self.get_tangent(u),
            _ => self.get_tangent(self.u_to_t(u)),
        }
    }

    /// The resolution `CurvePath.getPoints(divisions)` samples this curve at:
    /// twice for ellipses, once for lines, per control point for splines.
    pub fn path_resolution(&self, divisions: usize) -> usize {
        match self {
            ShapeCurve::Ellipse { .. } => divisions * 2,
            ShapeCurve::Line { .. } => 1,
            ShapeCurve::Spline { points } => divisions * points.len(),
            _ => divisions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_end_is_last_point() {
        let c = ShapeCurve::Spline {
            points: vec![
                Vec2d::new(0.0, 0.0),
                Vec2d::new(1.0, 2.0),
                Vec2d::new(3.0, 1.0),
            ],
        };
        assert_eq!(c.get_point(1.0), Vec2d::new(3.0, 1.0));
        assert_eq!(c.get_point(0.0), Vec2d::new(0.0, 0.0));
    }

    #[test]
    fn full_circle_when_angles_equal_modulo_two_pi() {
        let c = ShapeCurve::Ellipse {
            x: 0.0,
            y: 0.0,
            x_radius: 1.0,
            y_radius: 1.0,
            start_angle: 0.0,
            end_angle: std::f64::consts::PI * 2.0,
            clockwise: false,
            rotation: 0.0,
        };
        let p = c.get_point(0.25);
        assert!((p.x).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
    }
}
