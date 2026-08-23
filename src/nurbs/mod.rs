//! NURBS curves and surfaces — the surface primitive the B-rep layer stands on.
//!
//! Stage 0 of [`docs/brep-nurbs-plan.md`](../../docs/brep-nurbs-plan.md), behind
//! the `nurbs` feature.
//!
//! # What this is, and what `curves::nurbs` is
//!
//! [`crate::curves::NURBSCurve`] / [`crate::curves::NURBSSurface`] are the
//! three.js parity types: f32, evaluation only, unconditional. They stay exactly
//! where they are. This module is the real thing — f64, with derivatives, knot
//! operations, exact conic/quadric constructors, and adaptive tessellation — and
//! the parity types convert into it via `TryFrom`.
//!
//! # Why f64
//!
//! The exact-CSG kernel is f64 throughout (`exact_csg::V3 = [f64; 3]`), and its
//! `orient2d`/`orient3d` predicates are only exact on inputs that carry full
//! double precision. Intersection points derived from f32 surface evaluations
//! would arrive pre-rounded and the predicates would be adjudicating noise. So
//! everything here is f64 and narrows to f32 only at the `BufferGeometry`
//! boundary in [`crate::nurbs::tessellate`].
//!
//! # Exactness
//!
//! Circles, spheres, cylinders, cones and tori built by [`crate::nurbs::construct`] are
//! *exact*, not faceted — a circular arc is a rational quadratic with weight
//! `cos(Δθ/2)` on its middle control point. Sampling one at 10 000 parameters
//! reproduces the radius to ~1e-15. This is the property that makes surface
//! provenance (Stage 1) worth carrying: the analytic shape survives.

pub mod basis;
pub mod construct;
pub mod curve;
pub mod knot;
pub mod surface;
pub mod tessellate;

pub use basis::{basis_funcs, ders_basis_funcs, find_span, uniform_clamped_knots};
pub use construct::{loft, loft_with_params, sweep};
pub use curve::NurbsCurve;
pub use knot::{compact, remove_knot};
pub use surface::NurbsSurface;
pub use tessellate::{tessellate_curve, tessellate_surface, TessellationOptions};

/// A point or vector in f64. Deliberately the same shape as `exact_csg::V3`, so
/// intersection work in later stages passes values across without conversion.
pub type V3 = [f64; 3];

/// A homogeneous control point `(w·x, w·y, w·z, w)`.
///
/// Note the pre-multiplication: the Cartesian point is `(x, y, z)` and the
/// stored triple is that *scaled by the weight*. Storing un-premultiplied
/// coordinates alongside a weight is the more readable layout but the wrong one
/// to compute in — de Boor, knot insertion and the derivative formulas are all
/// linear in the homogeneous coordinates and nonlinear in the Cartesian ones.
/// [`NurbsCurve::new`] takes Cartesian points plus weights and does the
/// multiplication for you.
pub type V4 = [f64; 4];

/// Vector helpers. Small enough to inline, kept here so `curve`, `surface`,
/// `construct` and `tessellate` share one definition.
pub(crate) mod v3 {
    use super::V3;

    pub fn add(a: V3, b: V3) -> V3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    pub fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    pub fn scale(a: V3, s: f64) -> V3 {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    pub fn dot(a: V3, b: V3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    pub fn cross(a: V3, b: V3) -> V3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    pub fn norm(a: V3) -> f64 {
        dot(a, a).sqrt()
    }
    pub fn dist(a: V3, b: V3) -> f64 {
        norm(sub(a, b))
    }
    /// Normalize, returning `None` rather than NaN or a garbage direction when
    /// the input is degenerate. Callers decide what a zero-length vector means;
    /// silently returning the unnormalized value (as some of the f32 code in
    /// this crate does) hides pole degeneracies instead of surfacing them.
    pub fn normalize(a: V3) -> Option<V3> {
        let l = norm(a);
        if l < 1e-300 {
            None
        } else {
            Some([a[0] / l, a[1] / l, a[2] / l])
        }
    }
}

/// Everything that can be wrong with a NURBS definition, reported rather than
/// panicked — these values routinely arrive from files (STEP, IGES) and from
/// JS, where a malformed knot vector is a data error, not a bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NurbsError {
    /// Fewer than `degree + 1` control points — no valid span exists.
    DegreeTooHigh { degree: usize, n_ctrl: usize },
    /// Knot vector length must be exactly `n_ctrl + degree + 1`.
    KnotCount { expected: usize, got: usize },
    /// Knots must be non-decreasing.
    KnotsNotMonotonic,
    /// `knots[n_ctrl] <= knots[degree]` — the parametric domain is empty.
    DegenerateDomain,
    /// Weight array length must match the control point count.
    WeightCount { expected: usize, got: usize },
    /// Weights must be strictly positive; a zero or negative weight puts a pole
    /// inside the domain.
    NonPositiveWeight(usize),
    /// Control point grid length must be `n_u * n_v`.
    GridSize { expected: usize, got: usize },
    /// Two curves could not be brought to a common degree and knot vector.
    Incompatible(&'static str),
}

impl std::fmt::Display for NurbsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegreeTooHigh { degree, n_ctrl } => write!(
                f,
                "degree {degree} needs at least {} control points, got {n_ctrl}",
                degree + 1
            ),
            Self::KnotCount { expected, got } => {
                write!(f, "knot vector must have {expected} entries, got {got}")
            }
            Self::KnotsNotMonotonic => write!(f, "knot vector is not non-decreasing"),
            Self::DegenerateDomain => write!(f, "parametric domain is empty"),
            Self::WeightCount { expected, got } => {
                write!(f, "expected {expected} weights, got {got}")
            }
            Self::NonPositiveWeight(i) => write!(f, "weight {i} is not strictly positive"),
            Self::GridSize { expected, got } => {
                write!(f, "control grid must have {expected} points, got {got}")
            }
            Self::Incompatible(why) => write!(f, "curves are incompatible: {why}"),
        }
    }
}

impl std::error::Error for NurbsError {}

/// Project a homogeneous point to Cartesian.
///
/// `w` is guaranteed positive by construction (see [`NurbsError::NonPositiveWeight`]),
/// and a convex combination of positive weights stays positive, so the divisor
/// cannot vanish for any `u` in the domain. The guard is for callers who built
/// a curve via `from_homogeneous_unchecked`.
pub(crate) fn project(h: V4) -> V3 {
    if h[3].abs() < 1e-300 {
        [h[0], h[1], h[2]]
    } else {
        [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
    }
}
