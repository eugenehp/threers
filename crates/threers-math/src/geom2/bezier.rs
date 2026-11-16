//! Quadratic and cubic Bézier curves: points, bounds, splitting and
//! polylines within a chord error.

use super::{dist, lerp, P2};

/// A cubic Bézier curve from `p0` to `p3`, pulled by `p1` and `p2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cubic {
    pub p0: P2,
    pub p1: P2,
    pub p2: P2,
    pub p3: P2,
}

/// A quadratic Bézier curve from `p0` to `p2`, pulled by `p1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadratic {
    pub p0: P2,
    pub p1: P2,
    pub p2: P2,
}

impl Quadratic {
    pub fn new(p0: P2, p1: P2, p2: P2) -> Self {
        Quadratic { p0, p1, p2 }
    }

    /// The same curve as a cubic (degree elevation).
    pub fn to_cubic(&self) -> Cubic {
        Cubic::new(
            self.p0,
            lerp(self.p0, self.p1, 2.0 / 3.0),
            lerp(self.p2, self.p1, 2.0 / 3.0),
            self.p2,
        )
    }

    pub fn at(&self, t: f64) -> P2 {
        lerp(lerp(self.p0, self.p1, t), lerp(self.p1, self.p2, t), t)
    }

    pub fn flatten(&self, max_error: f64) -> Vec<P2> {
        self.to_cubic().flatten(max_error)
    }
}

impl Cubic {
    pub fn new(p0: P2, p1: P2, p2: P2, p3: P2) -> Self {
        Cubic { p0, p1, p2, p3 }
    }

    /// The point at `t` in `[0, 1]` (de Casteljau).
    pub fn at(&self, t: f64) -> P2 {
        self.split(t).0.p3
    }

    /// The two halves at `t`: from the start to `t`, and from `t` on.
    pub fn split(&self, t: f64) -> (Cubic, Cubic) {
        let (a, b, c) = (
            lerp(self.p0, self.p1, t),
            lerp(self.p1, self.p2, t),
            lerp(self.p2, self.p3, t),
        );
        let (d, e) = (lerp(a, b, t), lerp(b, c, t));
        let m = lerp(d, e, t);
        (Cubic::new(self.p0, a, d, m), Cubic::new(m, e, c, self.p3))
    }

    /// How far the curve can stray from its chord: the control points'
    /// largest distance from the line through the ends bounds it.
    fn flatness(&self) -> f64 {
        let chord = super::segment::Segment::new(self.p0, self.p3);
        chord.distance(self.p1).max(chord.distance(self.p2))
    }

    /// The curve as a polyline whose chords stay within `max_error` of it:
    /// halved until each piece's control points lie within the error of its
    /// chord (the curve lies in their hull), its ends included.
    pub fn flatten(&self, max_error: f64) -> Vec<P2> {
        let mut out = vec![self.p0];
        self.flatten_into(max_error.max(f64::MIN_POSITIVE), 24, &mut out);
        out
    }

    fn flatten_into(&self, max_error: f64, depth: u32, out: &mut Vec<P2>) {
        if depth == 0 || self.flatness() <= max_error {
            out.push(self.p3);
            return;
        }

        let (a, b) = self.split(0.5);
        a.flatten_into(max_error, depth - 1, out);
        b.flatten_into(max_error, depth - 1, out);
    }

    /// The bounding box, `[min, max]`: the ends and where the derivative
    /// of either coordinate is zero inside the curve.
    pub fn bounds(&self) -> [P2; 2] {
        let mut min = [self.p0[0].min(self.p3[0]), self.p0[1].min(self.p3[1])];
        let mut max = [self.p0[0].max(self.p3[0]), self.p0[1].max(self.p3[1])];

        for k in 0..2 {
            let (a, b, c, d) = (self.p0[k], self.p1[k], self.p2[k], self.p3[k]);
            // the derivative's coefficients, over three: qa t² + qb t + qc
            let qa = -a + 3.0 * b - 3.0 * c + d;
            let qb = 2.0 * (a - 2.0 * b + c);
            let qc = b - a;
            let mut roots = Vec::new();

            if qa.abs() < 1e-12 {
                if qb.abs() > 1e-12 {
                    roots.push(-qc / qb);
                }
            } else {
                let disc = qb * qb - 4.0 * qa * qc;

                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    roots.push((-qb + sq) / (2.0 * qa));
                    roots.push((-qb - sq) / (2.0 * qa));
                }
            }

            for t in roots.into_iter().filter(|t| *t > 0.0 && *t < 1.0) {
                let v = self.at(t)[k];
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
            }
        }

        [min, max]
    }

    /// The length, to within about `max_error` per piece.
    pub fn length(&self, max_error: f64) -> f64 {
        self.flatten(max_error)
            .windows(2)
            .map(|w| dist(w[0], w[1]))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_within_error() {
        let c = Cubic::new([0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]);
        let error = 0.01;
        let pts = c.flatten(error);
        assert_eq!(pts[0], c.p0);
        assert_eq!(*pts.last().unwrap(), c.p3);

        // a dense sampling of the curve stays near the polyline
        let poly_dist = |p: P2| {
            pts.windows(2)
                .map(|w| super::super::segment::Segment::new(w[0], w[1]).distance(p))
                .fold(f64::INFINITY, f64::min)
        };

        for i in 0..=1000 {
            assert!(poly_dist(c.at(i as f64 / 1000.0)) <= error + 1e-9);
        }
    }

    #[test]
    fn bounds_and_elevation() {
        let c = Cubic::new([0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]);
        let [min, max] = c.bounds();
        assert_eq!(min, [0.0, 0.0]);
        assert!((max[1] - 7.5).abs() < 1e-12 && max[0] == 10.0);

        let q = Quadratic::new([0.0, 0.0], [5.0, 10.0], [10.0, 0.0]);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let (a, b) = (q.at(t), q.to_cubic().at(t));
            assert!(dist(a, b) < 1e-12);
        }
    }
}
