//! Planar geometry in `f64`: segments, circular arcs, Bézier curves,
//! polylines, outlines of rounded shapes, hatching, and a static spatial
//! index. Points are `[f64; 2]`.
//!
//! Curves turn into polylines within a given chord error, the largest
//! distance between a chord and the curve it stands for, so callers trade
//! accuracy for point count with one number.

pub mod angle;
pub mod arc;
pub mod bezier;
pub mod hatch;
pub mod outline;
pub mod polyline;
pub mod rtree;
pub mod segment;

/// A point, or a vector, in the plane.
pub type P2 = [f64; 2];

pub(crate) fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}

pub(crate) fn scale(a: P2, s: f64) -> P2 {
    [a[0] * s, a[1] * s]
}

pub(crate) fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// The z of the 3D cross product: positive when `b` turns
/// counter-clockwise from `a` (y up).
pub(crate) fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub(crate) fn norm(a: P2) -> f64 {
    a[0].hypot(a[1])
}

pub(crate) fn dist(a: P2, b: P2) -> f64 {
    norm(sub(a, b))
}

pub(crate) fn lerp(a: P2, b: P2, t: f64) -> P2 {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// The parameter step that keeps a circle of radius `r`'s chords within
/// `max_error` of it: the sagitta `r (1 - cos(h / 2))` held to the error.
/// A quarter turn where the error is as large as the radius.
pub(crate) fn circle_step(r: f64, max_error: f64) -> f64 {
    if r <= max_error || r <= 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }

    2.0 * (1.0 - max_error / r).acos()
}
