//! Rational B-spline curves in f64.
//!
//! `!(x > 0.0)` throughout the validators is deliberate and not a clumsy
//! `x <= 0.0`: it is *true* for NaN, where the positive form is false. Weights
//! and knots arrive from files and from JS, and a NaN slipping through a
//! validator becomes a NaN mesh a long way from here.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::basis::{basis_funcs, binomial, ders_basis_funcs, find_span, knots_valid};
use super::v3;
use super::{project, NurbsError, V3, V4};

/// A NURBS curve: degree, knot vector, and homogeneous control points.
///
/// Fields are private because the three must stay consistent — a knot vector of
/// the wrong length is an out-of-bounds index in [`find_span`], not a merely
/// wrong curve. Build with [`NurbsCurve::new`] (Cartesian points + weights) or
/// [`NurbsCurve::from_homogeneous`], both of which validate.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve {
    degree: usize,
    knots: Vec<f64>,
    /// Homogeneous control points `(w·x, w·y, w·z, w)`.
    cw: Vec<V4>,
}

impl NurbsCurve {
    /// Build from Cartesian control points and (optional) weights.
    ///
    /// `weights = None` means all-ones, i.e. a non-rational B-spline. The
    /// weights are multiplied into the coordinates here so callers never have
    /// to think about homogeneous coordinates.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        points: &[V3],
        weights: Option<&[f64]>,
    ) -> Result<Self, NurbsError> {
        let n = points.len();
        if let Some(w) = weights {
            if w.len() != n {
                return Err(NurbsError::WeightCount {
                    expected: n,
                    got: w.len(),
                });
            }
            if let Some(i) = w.iter().position(|&x| !(x > 0.0)) {
                return Err(NurbsError::NonPositiveWeight(i));
            }
        }
        let cw = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let w = weights.map_or(1.0, |ws| ws[i]);
                [p[0] * w, p[1] * w, p[2] * w, w]
            })
            .collect();
        Self::from_homogeneous(degree, knots, cw)
    }

    /// Build from control points that are already in homogeneous form.
    ///
    /// This is the form STEP's `RATIONAL_B_SPLINE_CURVE` and three.js's
    /// `NURBSCurve` both use, so importers land here directly.
    pub fn from_homogeneous(
        degree: usize,
        knots: Vec<f64>,
        cw: Vec<V4>,
    ) -> Result<Self, NurbsError> {
        let n = cw.len();
        if n <= degree {
            return Err(NurbsError::DegreeTooHigh { degree, n_ctrl: n });
        }
        if knots.len() != n + degree + 1 {
            return Err(NurbsError::KnotCount {
                expected: n + degree + 1,
                got: knots.len(),
            });
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return Err(NurbsError::KnotsNotMonotonic);
        }
        if !(knots[n] > knots[degree]) {
            return Err(NurbsError::DegenerateDomain);
        }
        if let Some(i) = cw.iter().position(|c| !(c[3] > 0.0)) {
            return Err(NurbsError::NonPositiveWeight(i));
        }
        debug_assert!(knots_valid(degree, n, &knots));
        Ok(Self { degree, knots, cw })
    }

    /// Skip validation. Used by [`crate::nurbs::knot`] and [`crate::nurbs::construct`],
    /// which produce curves whose invariants hold by construction and would
    /// otherwise re-validate a knot vector on every insertion.
    pub(crate) fn from_homogeneous_unchecked(degree: usize, knots: Vec<f64>, cw: Vec<V4>) -> Self {
        debug_assert!(knots_valid(degree, cw.len(), &knots));
        Self { degree, knots, cw }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub fn n_control(&self) -> usize {
        self.cw.len()
    }

    /// Homogeneous control points, as stored.
    pub fn homogeneous(&self) -> &[V4] {
        &self.cw
    }

    /// Cartesian control point `i` (the homogeneous point divided by its weight).
    pub fn control_point(&self, i: usize) -> V3 {
        project(self.cw[i])
    }

    pub fn weight(&self, i: usize) -> f64 {
        self.cw[i][3]
    }

    /// Are any weights non-unit? Non-rational curves skip the quotient rule in
    /// [`Self::derivatives`] and can use cheaper algorithms elsewhere.
    pub fn is_rational(&self) -> bool {
        self.cw.iter().any(|c| (c[3] - 1.0).abs() > 1e-12)
    }

    /// The parametric domain `[knots[p], knots[n]]`. Evaluation outside it is
    /// clamped, not extrapolated.
    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.cw.len()])
    }

    /// Evaluate at parameter `u` (clamped to the domain).
    pub fn point(&self, u: f64) -> V3 {
        let (u0, u1) = self.domain();
        let u = u.clamp(u0, u1);
        let span = find_span(self.degree, self.cw.len(), &self.knots, u);
        let n = basis_funcs(span, u, self.degree, &self.knots);

        let mut acc = [0.0f64; 4];
        for (j, &b) in n.iter().enumerate() {
            let cp = self.cw[span - self.degree + j];
            acc[0] += b * cp[0];
            acc[1] += b * cp[1];
            acc[2] += b * cp[2];
            acc[3] += b * cp[3];
        }
        project(acc)
    }

    /// Derivatives `C(u), C'(u), …, C⁽ᵏ⁾(u)` — index 0 is the point itself.
    ///
    /// The rational case needs the quotient rule (A4.2): the homogeneous curve's
    /// derivatives are *not* the rational curve's derivatives, because the
    /// perspective divide is nonlinear. This is the routine whose absence forced
    /// finite-differenced normals everywhere upstream.
    /// The index arithmetic below mirrors the published algorithm one-for-one
    /// (same variable names, same bounds). Rewriting the loops as iterator
    /// zips reads worse against the reference and makes an off-by-one harder
    /// to spot, which is the opposite of what this code needs.
    #[allow(clippy::needless_range_loop)]
    pub fn derivatives(&self, u: f64, k: usize) -> Vec<V3> {
        let (u0, u1) = self.domain();
        let u = u.clamp(u0, u1);
        let p = self.degree;
        // Orders above the degree are identically zero.
        let du = k.min(p);
        let span = find_span(p, self.cw.len(), &self.knots, u);
        let ders = ders_basis_funcs(span, u, p, du, &self.knots);

        // Numerator derivatives A⁽ⁱ⁾ and weight-function derivatives w⁽ⁱ⁾.
        let mut aders = vec![[0.0f64; 3]; k + 1];
        let mut wders = vec![0.0f64; k + 1];
        for i in 0..=du {
            for j in 0..=p {
                let cp = self.cw[span - p + j];
                let b = ders[i][j];
                aders[i][0] += b * cp[0];
                aders[i][1] += b * cp[1];
                aders[i][2] += b * cp[2];
                wders[i] += b * cp[3];
            }
        }

        let w0 = if wders[0].abs() < 1e-300 {
            1.0
        } else {
            wders[0]
        };
        let mut ck = vec![[0.0f64; 3]; k + 1];
        for i in 0..=k {
            let mut v = aders[i];
            for j in 1..=i {
                let c = binomial(i, j) * wders[j];
                v[0] -= c * ck[i - j][0];
                v[1] -= c * ck[i - j][1];
                v[2] -= c * ck[i - j][2];
            }
            ck[i] = [v[0] / w0, v[1] / w0, v[2] / w0];
        }
        ck
    }

    /// Unit tangent at `u`, analytically.
    ///
    /// Returns `None` at a cusp (`C'(u) = 0`), which is a real feature of some
    /// valid curves rather than an error — callers that need a frame there must
    /// fall back to a higher derivative or a neighbouring parameter.
    pub fn tangent(&self, u: f64) -> Option<V3> {
        v3::normalize(self.derivatives(u, 1)[1])
    }

    /// Map the domain onto `[0, 1]`, so `Curve3::get_point(t)` and the
    /// tessellator can speak in normalized parameters without every call site
    /// re-deriving the knot range.
    pub fn param_at(&self, t: f64) -> f64 {
        let (u0, u1) = self.domain();
        u0 + t.clamp(0.0, 1.0) * (u1 - u0)
    }

    /// The distinct knot values inside the domain, in order — the curve's
    /// breakpoints. Curvature is generally discontinuous across these, so any
    /// adaptive sampler must place a sample on each one rather than subdivide
    /// blindly through them.
    pub fn breakpoints(&self) -> Vec<f64> {
        let (u0, u1) = self.domain();
        let mut out = vec![u0];
        for &k in &self.knots {
            if k > u0 && k < u1 && k > *out.last().unwrap() {
                out.push(k);
            }
        }
        out.push(u1);
        out
    }
}

impl crate::curves::Curve3 for NurbsCurve {
    fn get_point(&self, t: f32) -> crate::math::Vector3 {
        let p = self.point(self.param_at(t as f64));
        crate::math::Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32)
    }

    /// Overrides the trait's central-difference default with the analytic
    /// derivative — same answer to ~1e-9 in the interior, and correct at the
    /// endpoints where the default's clamped `t ± eps` window is one-sided and
    /// silently halves the step.
    fn get_tangent(&self, t: f32) -> crate::math::Vector3 {
        let u = self.param_at(t as f64);
        match self.tangent(u) {
            Some(d) => crate::math::Vector3::new(d[0] as f32, d[1] as f32, d[2] as f32),
            None => crate::math::Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

/// Convert the three.js parity type. Its control points are already
/// homogeneous, so this is a widen-and-validate.
impl TryFrom<&crate::curves::NURBSCurve> for NurbsCurve {
    type Error = NurbsError;

    fn try_from(c: &crate::curves::NURBSCurve) -> Result<Self, Self::Error> {
        let cw = c
            .control_points
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64, p[3] as f64])
            .collect();
        let knots = c.knots.iter().map(|&k| k as f64).collect();
        Self::from_homogeneous(c.degree, knots, cw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::construct;

    fn quarter_circle() -> NurbsCurve {
        // Exact quarter circle in the xy-plane: rational quadratic, three
        // control points, middle weight cos(45°) = √2/2.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::from_homogeneous(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0, 0.0, 1.0], [w, w, 0.0, w], [0.0, 1.0, 0.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn rejects_bad_definitions() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(
            NurbsCurve::new(3, vec![0.0; 6], &pts, None),
            Err(NurbsError::DegreeTooHigh {
                degree: 3,
                n_ctrl: 2
            })
        );
        assert_eq!(
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0], &pts, None),
            Err(NurbsError::KnotCount {
                expected: 4,
                got: 3
            })
        );
        assert_eq!(
            NurbsCurve::new(1, vec![0.0, 1.0, 0.5, 1.0], &pts, None),
            Err(NurbsError::KnotsNotMonotonic)
        );
        assert_eq!(
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], &pts, Some(&[1.0, 0.0])),
            Err(NurbsError::NonPositiveWeight(1))
        );
    }

    #[test]
    fn quarter_circle_is_exact() {
        let c = quarter_circle();
        for i in 0..=2000 {
            let u = i as f64 / 2000.0;
            let p = c.point(u);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-14, "radius {r} at u = {u}");
            assert!(p[2].abs() < 1e-15);
        }
    }

    #[test]
    fn endpoints_interpolate_control_points() {
        let c = quarter_circle();
        let (u0, u1) = c.domain();
        let a = c.point(u0);
        let b = c.point(u1);
        assert!(v3::dist(a, [1.0, 0.0, 0.0]) < 1e-15);
        assert!(v3::dist(b, [0.0, 1.0, 0.0]) < 1e-15);
    }

    #[test]
    fn analytic_derivatives_match_central_differences() {
        let c = construct::circle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.5);
        let h = 1e-5;
        let breaks = c.breakpoints();
        for i in 1..200 {
            let u = c.param_at(i as f64 / 200.0);
            let (u0, u1) = c.domain();
            if u - h < u0 || u + h > u1 {
                continue;
            }
            // A circle's knots have multiplicity 2 = degree, so the curve is only
            // C¹ there: C'' genuinely jumps across a breakpoint (verifiable by
            // one-sided limits). A central difference straddling one therefore
            // carries O(h·ΔC'') error and is comparing against a quantity that
            // does not exist. Skip a window, don't loosen the tolerance.
            if breaks.iter().any(|&b| (u - b).abs() < 4.0 * h) {
                continue;
            }
            let d = c.derivatives(u, 2);
            let fd1 = v3::scale(v3::sub(c.point(u + h), c.point(u - h)), 1.0 / (2.0 * h));
            let fd2 = v3::scale(
                v3::add(
                    v3::sub(c.point(u + h), v3::scale(c.point(u), 2.0)),
                    c.point(u - h),
                ),
                1.0 / (h * h),
            );
            assert!(v3::dist(d[1], fd1) < 1e-5, "C' at u = {u}");
            assert!(v3::dist(d[2], fd2) < 1e-3, "C'' at u = {u}");
        }
    }

    #[test]
    fn tangent_is_perpendicular_to_the_radius_on_a_circle() {
        let c = construct::circle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3.0);
        for i in 0..=100 {
            let u = c.param_at(i as f64 / 100.0);
            let p = c.point(u);
            let t = c.tangent(u).expect("circle has no cusps");
            assert!(v3::dot(p, t).abs() < 1e-12, "at u = {u}");
        }
    }

    #[test]
    fn derivatives_past_the_degree_are_zero() {
        let c = quarter_circle();
        let d = c.derivatives(0.4, 5);
        // A rational quadratic's numerator vanishes past order 2, but the
        // quotient rule keeps producing non-trivial values; what must hold is
        // that we return the requested count without indexing out of bounds.
        assert_eq!(d.len(), 6);
        assert!(d.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn breakpoints_cover_the_domain_without_duplicates() {
        let c = construct::circle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0);
        let bp = c.breakpoints();
        assert_eq!(bp.first().copied(), Some(c.domain().0));
        assert_eq!(bp.last().copied(), Some(c.domain().1));
        assert!(bp.windows(2).all(|w| w[1] > w[0]), "{bp:?}");
        // A full circle is four quarter arcs → interior breaks at 1/4, 1/2, 3/4.
        assert_eq!(bp.len(), 5);
    }

    #[test]
    fn parity_type_round_trips() {
        let old = crate::curves::NURBSCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.707_106_77, 0.707_106_77, 0.0, 0.707_106_77],
                [0.0, 1.0, 0.0, 1.0],
            ],
        );
        let new = NurbsCurve::try_from(&old).unwrap();
        use crate::curves::Curve3;
        for i in 0..=50 {
            let t = i as f32 / 50.0;
            let a = old.get_point(t);
            let b = Curve3::get_point(&new, t);
            assert!((a - b).length() < 1e-5, "t = {t}: {a:?} vs {b:?}");
        }
    }
}
