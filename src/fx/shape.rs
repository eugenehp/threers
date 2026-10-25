//! What an element is shaped like: [`Shape`], and its outline walked by arc
//! length ([`Outline`]) for effects that live on the edge.

use crate::curves::{parse_svg_subpaths, SvgPathError};
use crate::math::Vector2;
use crate::sdf::{self, d2};

/// Polygon shapes are capped at this many points (the surface shader walks
/// them per pixel). Longer outlines are decimated evenly.
pub const MAX_POLYGON_POINTS: usize = 128;

/// A 2D shape in CSS pixels, as a signed distance field the CPU and the GPU
/// both evaluate.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// A box with rounded corners; `half` its half-extents. A pill is
    /// `radius` = half the height ([`Shape::pill`]).
    RoundRect { center: Vector2, half: Vector2, radius: f32 },
    Circle { center: Vector2, radius: f32 },
    /// Any closed outline, either winding, grown outward by `radius` — which
    /// rounds every corner. At most [`MAX_POLYGON_POINTS`] points.
    Polygon { points: Vec<Vector2>, radius: f32 },
}

impl Shape {
    /// A pill (a stadium), `width` x `height` centred on `center`.
    pub fn pill(center: Vector2, width: f32, height: f32) -> Self {
        let half = Vector2::new(width * 0.5, height * 0.5);
        Shape::RoundRect { center, half, radius: half.x.min(half.y) }
    }

    /// A box from its top-left corner, with corners rounded by `radius`.
    pub fn rect(x: f32, y: f32, width: f32, height: f32, radius: f32) -> Self {
        let half = Vector2::new(width * 0.5, height * 0.5);
        Shape::RoundRect { center: Vector2::new(x + half.x, y + half.y), half, radius }
    }

    pub fn circle(center: Vector2, radius: f32) -> Self {
        Shape::Circle { center, radius }
    }

    /// A closed polygon, its corners rounded by `radius`.
    pub fn polygon(points: Vec<Vector2>, radius: f32) -> Self {
        let points = if points.len() > MAX_POLYGON_POINTS {
            let step = points.len() as f32 / MAX_POLYGON_POINTS as f32;
            (0..MAX_POLYGON_POINTS).map(|i| points[(i as f32 * step) as usize]).collect()
        } else {
            points
        };
        Shape::Polygon { points, radius }
    }

    /// A regular star of `points` points centred on `center`, `outer` to its
    /// tips and `inner` to its notches, corners rounded by `radius`.
    pub fn star(center: Vector2, points: usize, outer: f32, inner: f32, radius: f32) -> Self {
        let n = points.max(2) * 2;
        let pts = (0..n)
            .map(|k| {
                let a = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::TAU / n as f32;
                let r = if k % 2 == 0 { outer } else { inner };
                center + Vector2::new(a.cos() * r, a.sin() * r)
            })
            .collect();
        Shape::polygon(pts, radius)
    }

    /// The first subpath of SVG path data, scaled by `scale` and placed with
    /// its origin at `origin`, as a polygon rounded by `radius`. Curves and
    /// arcs are flattened; long outlines are decimated to
    /// [`MAX_POLYGON_POINTS`].
    pub fn from_svg_path(d: &str, origin: Vector2, scale: f32, radius: f32) -> Result<Self, SvgPathError> {
        let subpaths = parse_svg_subpaths(d)?;
        let mut points: Vec<Vector2> = Vec::new();
        if let Some(path) = subpaths.first() {
            let curves = path.curve_path.curves.len().max(1);
            // Enough samples per curve to stay smooth without blowing the cap.
            let per = (MAX_POLYGON_POINTS / curves).clamp(1, 16);
            for c in &path.curve_path.curves {
                for p in c.get_points(per) {
                    let q = origin + p * scale;
                    if points.last().is_none_or(|l| (*l - q).length() > 0.25) {
                        points.push(q);
                    }
                }
            }
            if points.len() > 1 && (points[0] - points[points.len() - 1]).length() < 0.25 {
                points.pop();
            }
        }
        Ok(Shape::polygon(points, radius))
    }

    /// Signed distance to the outline, negative inside.
    pub fn distance(&self, p: Vector2) -> f32 {
        match self {
            Shape::RoundRect { center, half, radius } => d2::round_rect(p - *center, *half, *radius),
            Shape::Circle { center, radius } => d2::circle(p - *center, *radius),
            Shape::Polygon { points, radius } => sdf::round(d2::polygon(p, points), *radius),
        }
    }

    pub fn contains(&self, p: Vector2) -> bool {
        self.distance(p) <= 0.0
    }

    /// Top-left and bottom-right of the bounding box.
    pub fn bounds(&self) -> (Vector2, Vector2) {
        match self {
            Shape::RoundRect { center, half, .. } => (*center - *half, *center + *half),
            Shape::Circle { center, radius } => {
                let r = Vector2::new(*radius, *radius);
                (*center - r, *center + r)
            }
            Shape::Polygon { points, radius } => {
                let (mut lo, mut hi) = (Vector2::new(f32::MAX, f32::MAX), Vector2::new(f32::MIN, f32::MIN));
                for p in points {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
                let r = Vector2::new(*radius, *radius);
                (lo - r, hi + r)
            }
        }
    }

    pub fn center(&self) -> Vector2 {
        let (lo, hi) = self.bounds();
        (lo + hi) * 0.5
    }

    /// The same shape scaled by `k` about its centre — what a pressed button does.
    pub fn scaled(&self, k: f32) -> Self {
        match self {
            Shape::RoundRect { center, half, radius } => Shape::RoundRect { center: *center, half: *half * k, radius: radius * k },
            Shape::Circle { center, radius } => Shape::Circle { center: *center, radius: radius * k },
            Shape::Polygon { points, radius } => {
                let c = self.center();
                Shape::Polygon { points: points.iter().map(|p| c + (*p - c) * k).collect(), radius: radius * k }
            }
        }
    }

    /// The same shape moved so its bounding box is centred on `center`.
    pub fn centered_at(&self, center: Vector2) -> Self {
        self.translated(center - self.center())
    }

    /// The same shape moved by `offset`.
    pub fn translated(&self, offset: Vector2) -> Self {
        match self {
            Shape::RoundRect { center, half, radius } => Shape::RoundRect { center: *center + offset, half: *half, radius: *radius },
            Shape::Circle { center, radius } => Shape::Circle { center: *center + offset, radius: *radius },
            Shape::Polygon { points, radius } => Shape::Polygon { points: points.iter().map(|p| *p + offset).collect(), radius: *radius },
        }
    }

    /// How deep the shape's own round profile is: its inradius, which is how
    /// far in a surface keeps curving before its top goes flat.
    pub fn inradius(&self) -> f32 {
        match self {
            Shape::RoundRect { half, .. } => half.x.min(half.y),
            Shape::Circle { radius, .. } => *radius,
            Shape::Polygon { .. } => {
                let (lo, hi) = self.bounds();
                let mut best = 0.0f32;
                for j in 0..24 {
                    for i in 0..24 {
                        let p = Vector2::new(
                            lo.x + (hi.x - lo.x) * (i as f32 + 0.5) / 24.0,
                            lo.y + (hi.y - lo.y) * (j as f32 + 0.5) / 24.0,
                        );
                        best = best.max(-self.distance(p));
                    }
                }
                best
            }
        }
    }
}

/// A point on an outline and the outward normal there.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub point: Vector2,
    pub normal: Vector2,
}

/// A shape's outline, resampled evenly and walked by arc length — for an
/// effect that roots things on the edge and slides them round it, whatever
/// the shape.
#[derive(Debug, Clone)]
pub struct Outline {
    pts: Vec<Vector2>,
    nrm: Vec<Vector2>,
    /// Arc length at each point; the loop closes back to the first.
    cum: Vec<f32>,
    perimeter: f32,
}

impl Outline {
    pub fn new(shape: &Shape) -> Self {
        let raw = raw_loop(shape);
        // Resample to an even ~2 px spacing, so arc length is uniform whatever
        // the source's sampling was.
        let n = raw.len();
        let total: f32 = (0..n).map(|i| (raw[(i + 1) % n] - raw[i]).length()).sum();
        let count = ((total / 2.0).ceil() as usize).clamp(16, 4096);
        let step = total / count as f32;
        let mut pts = Vec::with_capacity(count);
        let (mut seg, mut along) = (0usize, 0.0f32);
        for k in 0..count {
            let want = k as f32 * step;
            loop {
                let (a, b) = (raw[seg % n], raw[(seg + 1) % n]);
                let l = (b - a).length();
                if along + l >= want || seg >= 2 * n {
                    let t = if l > 0.0 { ((want - along) / l).clamp(0.0, 1.0) } else { 0.0 };
                    pts.push(a + (b - a) * t);
                    break;
                }
                along += l;
                seg += 1;
            }
        }
        // Normals from the field itself: outward whichever way the outline winds.
        let h = 0.5;
        let nrm = pts
            .iter()
            .map(|p| {
                let gx = shape.distance(Vector2::new(p.x + h, p.y)) - shape.distance(Vector2::new(p.x - h, p.y));
                let gy = shape.distance(Vector2::new(p.x, p.y + h)) - shape.distance(Vector2::new(p.x, p.y - h));
                let g = Vector2::new(gx, gy);
                if g.length() > 1e-6 { g.normalize() } else { Vector2::new(0.0, -1.0) }
            })
            .collect();
        let cum = (0..count).map(|k| k as f32 * step).collect();
        Self { pts, nrm, cum, perimeter: total.max(1.0) }
    }

    pub fn perimeter(&self) -> f32 {
        self.perimeter
    }

    /// The point `s` along the outline, wrapping.
    pub fn at(&self, s: f32) -> Edge {
        let s = s.rem_euclid(self.perimeter);
        let i = match self.cum.binary_search_by(|c| c.partial_cmp(&s).unwrap_or(std::cmp::Ordering::Less)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let j = (i + 1) % self.pts.len();
        let next = if j == 0 { self.perimeter } else { self.cum[j] };
        let t = ((s - self.cum[i]) / (next - self.cum[i]).max(1e-6)).clamp(0.0, 1.0);
        Edge {
            point: self.pts[i] + (self.pts[j] - self.pts[i]) * t,
            normal: (self.nrm[i] + (self.nrm[j] - self.nrm[i]) * t).normalize(),
        }
    }

    /// Arc length of the outline point nearest `q`.
    pub fn nearest(&self, q: Vector2) -> f32 {
        let n = self.pts.len();
        let (mut best, mut best_s) = (f32::MAX, 0.0);
        for i in 0..n {
            let (a, b) = (self.pts[i], self.pts[(i + 1) % n]);
            let ab = b - a;
            let l2 = ab.dot(ab);
            let t = if l2 > 0.0 { ((q - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = (q - (a + ab * t)).length_sq();
            if d < best {
                best = d;
                best_s = self.cum[i] + l2.sqrt() * t;
            }
        }
        best_s
    }

    /// The shorter way round from `a` to `b`, signed.
    pub fn loop_diff(&self, a: f32, b: f32) -> f32 {
        let p = self.perimeter;
        let mut d = (b - a) % p;
        if d > p / 2.0 {
            d -= p;
        }
        if d < -p / 2.0 {
            d += p;
        }
        d
    }
}

/// A closed loop round the shape, before resampling.
fn raw_loop(shape: &Shape) -> Vec<Vector2> {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    match shape {
        Shape::RoundRect { center, half, radius } => {
            let r = radius.max(0.0).min(half.x.min(half.y));
            let (ix, iy) = (half.x - r, half.y - r);
            // Clockwise on screen from the top-right corner.
            let corners = [(ix, -iy, -FRAC_PI_2), (ix, iy, 0.0), (-ix, iy, FRAC_PI_2), (-ix, -iy, PI)];
            let mut out = Vec::new();
            for (cx, cy, a0) in corners {
                for k in 0..=16 {
                    let a = a0 + FRAC_PI_2 * k as f32 / 16.0;
                    out.push(Vector2::new(center.x + cx + a.cos() * r, center.y + cy + a.sin() * r));
                }
            }
            out
        }
        Shape::Circle { center, radius } => (0..96)
            .map(|k| {
                let a = -FRAC_PI_2 + TAU * k as f32 / 96.0;
                Vector2::new(center.x + a.cos() * radius, center.y + a.sin() * radius)
            })
            .collect(),
        Shape::Polygon { points, radius } => {
            let n = points.len();
            if n < 3 {
                return vec![Vector2::ZERO, Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0)];
            }
            if *radius <= 0.0 {
                return points.clone();
            }
            // The outline grown by `radius`: an arc round each convex corner,
            // and the offset edges' crossing at each concave one.
            let area: f32 = (0..n).map(|i| points[i].cross(points[(i + 1) % n])).sum();
            let sign = if area >= 0.0 { 1.0 } else { -1.0 };
            let normal = |a: Vector2, b: Vector2| {
                let e = (b - a).normalize();
                Vector2::new(e.y, -e.x) * sign
            };
            let mut out = Vec::new();
            for i in 0..n {
                let (a, p, b) = (points[(i + n - 1) % n], points[i], points[(i + 1) % n]);
                let (n1, n2) = (normal(a, p), normal(p, b));
                if (p - a).cross(b - p) * sign >= 0.0 {
                    let a1 = n1.y.atan2(n1.x);
                    let mut delta = n2.y.atan2(n2.x) - a1;
                    while delta > PI {
                        delta -= TAU;
                    }
                    while delta < -PI {
                        delta += TAU;
                    }
                    let steps = ((delta.abs() / (PI / 16.0)).ceil() as usize).max(1);
                    for k in 0..=steps {
                        let t = a1 + delta * k as f32 / steps as f32;
                        out.push(p + Vector2::new(t.cos(), t.sin()) * *radius);
                    }
                } else {
                    let bis = (n1 + n2).normalize();
                    out.push(p + bis * (*radius / bis.dot(n1).max(0.2)));
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32) -> Vector2 {
        Vector2::new(x, y)
    }

    #[test]
    fn shapes_measure_themselves() {
        let pill = Shape::pill(v(100.0, 50.0), 200.0, 40.0);
        assert!((pill.distance(v(100.0, 50.0)) + 20.0).abs() < 1e-4);
        assert!((pill.distance(v(100.0, 0.0)) - 30.0).abs() < 1e-4);
        assert_eq!(pill.bounds(), (v(0.0, 30.0), v(200.0, 70.0)));
        let square = Shape::polygon(vec![v(0.0, 0.0), v(10.0, 0.0), v(10.0, 10.0), v(0.0, 10.0)], 2.0);
        assert!((square.distance(v(15.0, 5.0)) - 3.0).abs() < 1e-4);
        assert!(square.contains(v(11.0, 5.0)));
        assert!((square.inradius() - 7.0).abs() < 0.5, "{}", square.inradius());
        assert_eq!(pill.scaled(0.5).bounds(), (v(50.0, 40.0), v(150.0, 60.0)));
        assert_eq!(pill.centered_at(v(0.0, 0.0)).center(), v(0.0, 0.0));
        let star = Shape::star(v(0.0, 0.0), 5, 50.0, 20.0, 0.0);
        assert!(star.contains(v(0.0, -45.0)) && !star.contains(v(30.0, -30.0)));
    }

    #[test]
    fn long_polygons_are_capped() {
        let ring: Vec<Vector2> = (0..1000)
            .map(|i| {
                let a = i as f32 / 1000.0 * std::f32::consts::TAU;
                v(a.cos() * 50.0, a.sin() * 50.0)
            })
            .collect();
        let Shape::Polygon { points, .. } = Shape::polygon(ring, 0.0) else { unreachable!() };
        assert_eq!(points.len(), MAX_POLYGON_POINTS);
    }

    #[test]
    fn svg_shapes_land_where_the_path_says() {
        let heart = "M0 -3C-4 -8 -10 -3 0 6C10 -3 4 -8 0 -3Z";
        let shape = Shape::from_svg_path(heart, v(100.0, 100.0), 10.0, 0.0).unwrap();
        let (lo, hi) = shape.bounds();
        assert!((hi.y - 160.0).abs() < 1.0, "{hi:?}");
        assert!(lo.x < 100.0 && hi.x > 100.0);
        assert!(shape.contains(v(100.0, 120.0)));
        assert!(!shape.contains(v(100.0, 300.0)));
    }

    #[test]
    fn the_outline_walks_the_surface_with_outward_normals() {
        for shape in [
            Shape::pill(v(0.0, 0.0), 240.0, 64.0),
            Shape::circle(v(10.0, 20.0), 40.0),
            Shape::polygon(vec![v(0.0, 0.0), v(100.0, 0.0), v(50.0, 80.0)], 6.0),
            Shape::polygon(vec![v(0.0, 80.0), v(50.0, 0.0), v(100.0, 80.0)], 0.0),
            Shape::star(v(0.0, 0.0), 5, 60.0, 25.0, 5.0),
        ] {
            let o = Outline::new(&shape);
            for k in 0..200 {
                let e = o.at(o.perimeter() * k as f32 / 200.0);
                assert!(shape.distance(e.point).abs() < 0.6, "{shape:?}: {:?} is {} off", e.point, shape.distance(e.point));
                assert!(shape.distance(e.point + e.normal * 3.0) > 1.5, "{shape:?}: inward normal at {:?}", e.point);
            }
            // whatever `nearest` finds is at least as close as where we started
            for k in 0..20 {
                let e = o.at(o.perimeter() * k as f32 / 20.0);
                let q = e.point + e.normal * 10.0;
                let found = o.at(o.nearest(q)).point;
                assert!((q - found).length() <= (q - e.point).length() + 0.5, "{shape:?} near {:?}", e.point);
            }
        }
    }

    #[test]
    fn a_pill_outline_has_the_pills_perimeter() {
        let o = Outline::new(&Shape::pill(v(0.0, 0.0), 240.0, 64.0));
        let want = 2.0 * (240.0 - 64.0) + std::f32::consts::PI * 64.0;
        assert!((o.perimeter() - want).abs() < 1.0, "{} vs {want}", o.perimeter());
    }
}
