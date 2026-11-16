//! A rotated ellipse, or an arc of one: points by parameter, an exact
//! bounding box, and a polyline within a chord error.
//!
//! The ellipse is `center + R(rotation) * (a cos t, b sin t)` for the
//! parameter `t` (not the polar angle), with `a` the major and `b` the minor
//! radius. It is the image of the circle of radius `a` under a scale of
//! `b / a` across the major axis, then a rotation; neither lengthens any
//! distance, so a chord that stays within `e` of the circle's arc stays
//! within `e` of the ellipse's. That gives [`Ellipse::flatten`] a uniform
//! parameter step with a proven bound and no recursion.

use std::f64::consts::TAU;

/// An ellipse, or an arc of one when `sweep` is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipse {
    pub center: [f64; 2],
    /// The major radius `a` and the minor radius `b`, `a >= b >= 0`.
    pub radii: [f64; 2],
    /// The major axis's angle from +x, in radians.
    pub rotation: f64,
    /// The parameter the arc starts at and how far it runs (negative runs
    /// backwards), in radians; `None` for the whole ellipse.
    pub sweep: Option<(f64, f64)>,
}

impl Ellipse {
    /// A whole ellipse; the radii in either order, the larger made the
    /// major one (the rotation turned a quarter to keep the shape).
    pub fn new(center: [f64; 2], rx: f64, ry: f64, rotation: f64) -> Self {
        let (rx, ry) = (rx.abs(), ry.abs());

        if rx >= ry {
            Ellipse {
                center,
                radii: [rx, ry],
                rotation,
                sweep: None,
            }
        } else {
            Ellipse {
                center,
                radii: [ry, rx],
                rotation: rotation + TAU / 4.0,
                sweep: None,
            }
        }
    }

    /// This ellipse's arc from parameter `start` through `sweep` (the
    /// parameters of [`Ellipse::new`]'s major axis).
    pub fn arc(self, start: f64, sweep: f64) -> Self {
        Ellipse {
            sweep: Some((start, sweep)),
            ..self
        }
    }

    /// The point at parameter `t`.
    pub fn point(&self, t: f64) -> [f64; 2] {
        let (s, c) = self.rotation.sin_cos();
        let (x, y) = (self.radii[0] * t.cos(), self.radii[1] * t.sin());
        [
            self.center[0] + x * c - y * s,
            self.center[1] + x * s + y * c,
        ]
    }

    /// The parameter range covered, increasing: `[start, end]`.
    fn range(&self) -> (f64, f64) {
        match self.sweep {
            None => (0.0, TAU),
            Some((start, sweep)) if sweep >= 0.0 => (start, start + sweep.min(TAU)),
            Some((start, sweep)) => (start + sweep.max(-TAU), start),
        }
    }

    /// Whether parameter `t` lies on the arc (always, for a whole ellipse).
    fn covers(&self, t: f64) -> bool {
        let (lo, hi) = self.range();

        if hi - lo >= TAU {
            return true;
        }

        let k = ((t - lo) / TAU).floor();
        let t = t - k * TAU;
        t <= hi
    }

    /// The bounding box, `[min, max]`: the arc's ends and whichever of the
    /// four axis extremes it covers.
    pub fn bounds(&self) -> [[f64; 2]; 2] {
        let (s, c) = self.rotation.sin_cos();
        let [a, b] = self.radii;
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        let mut take = |p: [f64; 2]| {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        };

        if self.sweep.is_some() {
            let (lo, hi) = self.range();
            take(self.point(lo));
            take(self.point(hi));
        }

        // where dx/dt = 0 and dy/dt = 0, and half a turn on
        let tx = (-b * s).atan2(a * c);
        let ty = (b * c).atan2(a * s);

        for t in [tx, tx + TAU / 2.0, ty, ty + TAU / 2.0] {
            if self.covers(t) {
                take(self.point(t));
            }
        }

        [min, max]
    }

    /// The parameter step whose chords stay within `max_error` of the curve.
    fn step(&self, max_error: f64) -> f64 {
        let a = self.radii[0];

        if a <= max_error || a == 0.0 {
            return TAU / 4.0;
        }

        // the sagitta a (1 - cos(h / 2)) of the bounding circle, at most max_error
        2.0 * (1.0 - max_error / a).acos()
    }

    /// The curve as a polyline, each chord within `max_error` of it: the
    /// arc's ends both included; a whole ellipse's points once round, the
    /// first not repeated at the end.
    pub fn flatten(&self, max_error: f64) -> Vec<[f64; 2]> {
        let (lo, hi) = self.range();
        let n = ((hi - lo) / self.step(max_error.max(f64::MIN_POSITIVE)))
            .ceil()
            .max(1.0) as usize;
        let whole = self.sweep.is_none();
        let count = if whole { n.max(3) } else { n };
        let h = (hi - lo) / count as f64;
        let ends = if whole { count } else { count + 1 };

        (0..ends).map(|i| self.point(lo + h * i as f64)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_ellipse(e: &Ellipse, p: [f64; 2]) -> f64 {
        let (s, c) = e.rotation.sin_cos();
        let (dx, dy) = (p[0] - e.center[0], p[1] - e.center[1]);
        let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
        (u / e.radii[0]).powi(2) + (v / e.radii[1]).powi(2)
    }

    #[test]
    fn major_axis_kept_larger() {
        let e = Ellipse::new([0.0, 0.0], 1.0, 2.0, 0.0);
        assert_eq!(e.radii, [2.0, 1.0]);
        assert!((e.rotation - TAU / 4.0).abs() < 1e-12);
    }

    #[test]
    fn flattened_within_error() {
        let e = Ellipse::new([3.0, -2.0], 50.0, 20.0, 0.7);
        let error = 0.05;
        let pts = e.flatten(error);

        for p in &pts {
            assert!((on_ellipse(&e, *p) - 1.0).abs() < 1e-9);
        }

        // every chord's middle within the error of the curve: check against a fine sampling
        for w in pts
            .windows(2)
            .chain(std::iter::once([pts[pts.len() - 1], pts[0]].as_slice()))
        {
            let mid = [(w[0][0] + w[1][0]) / 2.0, (w[0][1] + w[1][1]) / 2.0];
            let nearest = (0..20_000)
                .map(|i| e.point(TAU * i as f64 / 20_000.0))
                .map(|q| ((q[0] - mid[0]).powi(2) + (q[1] - mid[1]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= error + 1e-3, "{nearest}");
        }
    }

    #[test]
    fn bounds_of_whole_and_arc() {
        // an axis-aligned ellipse
        let e = Ellipse::new([0.0, 0.0], 4.0, 1.0, 0.0);
        let [min, max] = e.bounds();
        assert!((min[0] + 4.0).abs() < 1e-12 && (max[0] - 4.0).abs() < 1e-12);
        assert!((min[1] + 1.0).abs() < 1e-12 && (max[1] - 1.0).abs() < 1e-12);

        // its upper half: from +x round through +y to -x
        let [min, max] = e.arc(0.0, TAU / 2.0).bounds();
        assert!(min[1].abs() < 1e-12 && (max[1] - 1.0).abs() < 1e-12);
        assert!((min[0] + 4.0).abs() < 1e-12 && (max[0] - 4.0).abs() < 1e-12);

        // a rotated one against a dense sampling
        let r = Ellipse::new([1.0, 2.0], 5.0, 2.0, 0.4);
        let [min, max] = r.bounds();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);

        for i in 0..100_000 {
            let p = r.point(TAU * i as f64 / 100_000.0);
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }

        for k in 0..2 {
            assert!((min[k] - lo[k]).abs() < 1e-6 && (max[k] - hi[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn arc_ends_included() {
        let e = Ellipse::new([0.0, 0.0], 3.0, 2.0, 0.0).arc(0.5, -1.0);
        let pts = e.flatten(0.01);
        let (first, last) = (pts[0], pts[pts.len() - 1]);
        let (lo, hi) = (e.point(-0.5), e.point(0.5));
        assert!((first[0] - lo[0]).abs() < 1e-12 && (first[1] - lo[1]).abs() < 1e-12);
        assert!((last[0] - hi[0]).abs() < 1e-9 && (last[1] - hi[1]).abs() < 1e-9);
    }
}
