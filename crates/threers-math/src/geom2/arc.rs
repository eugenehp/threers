//! Circular arcs: through three points or about a centre, their points,
//! bounds and polylines.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use super::angle::in_sweep;
use super::{circle_step, cross, sub, P2};

/// An arc about `center`, of `radius`, from angle `start` through `sweep`
/// radians (counter-clockwise for a positive sweep, y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arc {
    pub center: P2,
    pub radius: f64,
    pub start: f64,
    pub sweep: f64,
}

/// The centre of the circle through three points; `None` where they are
/// on one line.
pub fn circumcenter(a: P2, b: P2, c: P2) -> Option<P2> {
    let (ab, ac) = (sub(b, a), sub(c, a));
    let d = 2.0 * cross(ab, ac);

    if d == 0.0 {
        return None;
    }

    let (ab2, ac2) = (ab[0] * ab[0] + ab[1] * ab[1], ac[0] * ac[0] + ac[1] * ac[1]);
    let ux = (ac[1] * ab2 - ab[1] * ac2) / d;
    let uy = (ab[0] * ac2 - ac[0] * ab2) / d;
    Some([a[0] + ux, a[1] + uy])
}

impl Arc {
    pub fn new(center: P2, radius: f64, start: f64, sweep: f64) -> Self {
        Arc {
            center,
            radius,
            start,
            sweep,
        }
    }

    /// A whole circle, from angle 0.
    pub fn circle(center: P2, radius: f64) -> Self {
        Arc::new(center, radius, 0.0, TAU)
    }

    /// The arc from `start` through `mid` to `end`; `None` where the three
    /// are on one line.
    pub fn through(start: P2, mid: P2, end: P2) -> Option<Self> {
        let center = circumcenter(start, mid, end)?;
        let angle = |p: P2| (p[1] - center[1]).atan2(p[0] - center[0]);
        let (a0, am, a1) = (angle(start), angle(mid), angle(end));
        let ccw = |from: f64, to: f64| (to - from).rem_euclid(TAU);

        // the way round that passes the middle point
        let sweep = if ccw(a0, am) <= ccw(a0, a1) {
            ccw(a0, a1)
        } else {
            ccw(a0, a1) - TAU
        };
        let radius = super::dist(start, center);
        Some(Arc::new(center, radius, a0, sweep))
    }

    /// The point at angle `a` on the circle.
    pub fn point_at_angle(&self, a: f64) -> P2 {
        [
            self.center[0] + self.radius * a.cos(),
            self.center[1] + self.radius * a.sin(),
        ]
    }

    /// The point at parameter `t`, 0 at the start and 1 at the end.
    pub fn at(&self, t: f64) -> P2 {
        self.point_at_angle(self.start + self.sweep * t)
    }

    pub fn start_point(&self) -> P2 {
        self.at(0.0)
    }

    pub fn end_point(&self) -> P2 {
        self.at(1.0)
    }

    pub fn length(&self) -> f64 {
        self.radius * self.sweep.abs()
    }

    /// The bounding box, `[min, max]`: the ends and whichever of the four
    /// axis extremes the arc passes.
    pub fn bounds(&self) -> [P2; 2] {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        let mut take = |p: P2| {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        };

        take(self.start_point());
        take(self.end_point());

        for quadrant in [0.0, FRAC_PI_2, PI, 1.5 * PI] {
            if in_sweep(quadrant, self.start, self.sweep) {
                take(self.point_at_angle(quadrant));
            }
        }

        [min, max]
    }

    /// The arc as a polyline whose chords stay within `max_error` of it,
    /// its ends included.
    pub fn flatten(&self, max_error: f64) -> Vec<P2> {
        let step = circle_step(self.radius, max_error.max(f64::MIN_POSITIVE));
        let n = (self.sweep.abs() / step).ceil().max(1.0) as usize;
        (0..=n).map(|i| self.at(i as f64 / n as f64)).collect()
    }

    /// The distance from `p` to the arc.
    pub fn distance(&self, p: P2) -> f64 {
        let a = (p[1] - self.center[1]).atan2(p[0] - self.center[0]);

        if in_sweep(a, self.start, self.sweep) {
            (super::dist(p, self.center) - self.radius).abs()
        } else {
            super::dist(p, self.start_point()).min(super::dist(p, self.end_point()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn through_three_points() {
        // the upper half of the unit circle, right to left
        let arc = Arc::through([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]).unwrap();
        assert!(arc.center[0].abs() < 1e-12 && arc.center[1].abs() < 1e-12);
        assert!((arc.sweep - PI).abs() < 1e-12);

        // the lower half, the other way round
        let arc = Arc::through([1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]).unwrap();
        assert!((arc.sweep + PI).abs() < 1e-12);
        assert!(Arc::through([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]).is_none());
    }

    #[test]
    fn bounds_and_flatten() {
        let arc = Arc::new([0.0, 0.0], 2.0, 0.0, PI);
        let [min, max] = arc.bounds();
        assert!((min[0] + 2.0).abs() < 1e-12 && (max[0] - 2.0).abs() < 1e-12);
        assert!(min[1].abs() < 1e-12 && (max[1] - 2.0).abs() < 1e-12);

        let error = 0.01;
        let pts = arc.flatten(error);

        for w in pts.windows(2) {
            let mid = [(w[0][0] + w[1][0]) / 2.0, (w[0][1] + w[1][1]) / 2.0];
            assert!(arc.distance(mid) <= error + 1e-12);
        }

        assert!(super::super::dist(pts[pts.len() - 1], [-2.0, 0.0]) < 1e-12);
    }
}
