//! Line segments: nearest points, distances, sides and intersections.

use super::{add, cross, dist, dot, lerp, scale, sub, P2};

/// The segment from `a` to `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: P2,
    pub b: P2,
}

/// How two segments meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Crossing {
    /// They do not meet.
    None,
    /// They meet at one point.
    Point(P2),
    /// They lie on one line and share the stretch between these points.
    Overlap(P2, P2),
}

impl Segment {
    pub fn new(a: P2, b: P2) -> Self {
        Segment { a, b }
    }

    pub fn length(&self) -> f64 {
        dist(self.a, self.b)
    }

    /// The point at parameter `t`, 0 at `a` and 1 at `b`.
    pub fn at(&self, t: f64) -> P2 {
        lerp(self.a, self.b, t)
    }

    /// The parameter of the point of the segment nearest `p`, in `[0, 1]`.
    pub fn project(&self, p: P2) -> f64 {
        let d = sub(self.b, self.a);
        let len2 = dot(d, d);

        if len2 == 0.0 {
            return 0.0;
        }

        (dot(sub(p, self.a), d) / len2).clamp(0.0, 1.0)
    }

    /// The point of the segment nearest `p`.
    pub fn nearest(&self, p: P2) -> P2 {
        self.at(self.project(p))
    }

    /// The distance from `p` to the segment.
    pub fn distance(&self, p: P2) -> f64 {
        dist(p, self.nearest(p))
    }

    /// Which side of the line through the segment `p` is on: positive to
    /// the left of `a` to `b` (y up), negative to the right, zero on it.
    pub fn side(&self, p: P2) -> f64 {
        cross(sub(self.b, self.a), sub(p, self.a))
    }

    /// Where the two segments meet; `eps` is the tolerance on the cross
    /// products that decide parallel and collinear.
    pub fn crossing(&self, other: &Segment, eps: f64) -> Crossing {
        let r = sub(self.b, self.a);
        let s = sub(other.b, other.a);
        let qp = sub(other.a, self.a);
        let denom = cross(r, s);

        if denom.abs() <= eps {
            // parallel: collinear when the other's start is on this line
            if cross(qp, r).abs() > eps {
                return Crossing::None;
            }

            let rr = dot(r, r);

            if rr == 0.0 {
                return if other.distance(self.a) <= eps.sqrt() {
                    Crossing::Point(self.a)
                } else {
                    Crossing::None
                };
            }

            // the other's ends along this one, ordered
            let t0 = dot(qp, r) / rr;
            let t1 = t0 + dot(s, r) / rr;
            let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));

            return if lo > hi {
                Crossing::None
            } else if lo == hi {
                Crossing::Point(self.at(lo))
            } else {
                Crossing::Overlap(self.at(lo), self.at(hi))
            };
        }

        let t = cross(qp, s) / denom;
        let u = cross(qp, r) / denom;

        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            Crossing::Point(add(self.a, scale(r, t)))
        } else {
            Crossing::None
        }
    }

    /// The least distance between the two segments (0 where they meet).
    pub fn distance_to(&self, other: &Segment) -> f64 {
        if !matches!(self.crossing(other, 0.0), Crossing::None) {
            return 0.0;
        }

        [
            self.distance(other.a),
            self.distance(other.b),
            other.distance(self.a),
            other.distance(self.b),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_and_distance() {
        let s = Segment::new([0.0, 0.0], [10.0, 0.0]);
        assert_eq!(s.nearest([3.0, 4.0]), [3.0, 0.0]);
        assert_eq!(s.distance([-3.0, 4.0]), 5.0);
        assert!(s.side([1.0, 1.0]) > 0.0 && s.side([1.0, -1.0]) < 0.0);
    }

    #[test]
    fn crossings() {
        let s = Segment::new([0.0, 0.0], [10.0, 10.0]);
        assert_eq!(
            s.crossing(&Segment::new([0.0, 10.0], [10.0, 0.0]), 1e-12),
            Crossing::Point([5.0, 5.0])
        );
        assert_eq!(
            s.crossing(&Segment::new([0.0, 1.0], [10.0, 11.0]), 1e-12),
            Crossing::None
        );
        assert_eq!(
            s.crossing(&Segment::new([5.0, 5.0], [20.0, 20.0]), 1e-12),
            Crossing::Overlap([5.0, 5.0], [10.0, 10.0])
        );
        assert_eq!(
            s.crossing(&Segment::new([10.0, 10.0], [20.0, 20.0]), 1e-12),
            Crossing::Point([10.0, 10.0])
        );
        assert_eq!(
            Segment::new([0.0, 0.0], [1.0, 0.0]).distance_to(&Segment::new([0.0, 2.0], [1.0, 2.0])),
            2.0
        );
    }
}
