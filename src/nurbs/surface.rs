//! Rational B-spline surfaces in f64.
//!
//! `!(x > 0.0)` throughout the validators is deliberate and not a clumsy
//! `x <= 0.0`: it is *true* for NaN, where the positive form is false. Weights
//! and knots arrive from files and from JS, and a NaN slipping through a
//! validator becomes a NaN mesh a long way from here.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::basis::{basis_funcs, binomial, ders_basis_funcs, find_span, knots_valid};
use super::v3;
use super::{project, NurbsError, V3, V4};

/// A NURBS surface: bidegree, two knot vectors, and a homogeneous control grid.
///
/// The grid is **u-major**: `control[i * n_v + j]` is the control point at
/// u-index `i`, v-index `j`. This matches Piegl & Tiller's `P[i][j]` and STEP's
/// `control_points_list`. (The three.js parity type in `curves::nurbs` stores
/// the transpose — v-major — which is why [`TryFrom`] below is not a memcpy.)
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSurface {
    degree_u: usize,
    degree_v: usize,
    knots_u: Vec<f64>,
    knots_v: Vec<f64>,
    control: Vec<V4>,
    n_u: usize,
    n_v: usize,
}

impl NurbsSurface {
    /// Build from a Cartesian control grid (u-major) and optional weights.
    ///
    /// Eight parameters, because a NURBS surface is eight independent things.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        n_u: usize,
        n_v: usize,
        points: &[V3],
        weights: Option<&[f64]>,
    ) -> Result<Self, NurbsError> {
        if points.len() != n_u * n_v {
            return Err(NurbsError::GridSize {
                expected: n_u * n_v,
                got: points.len(),
            });
        }
        if let Some(w) = weights {
            if w.len() != points.len() {
                return Err(NurbsError::WeightCount {
                    expected: points.len(),
                    got: w.len(),
                });
            }
        }
        let control = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let w = weights.map_or(1.0, |ws| ws[i]);
                [p[0] * w, p[1] * w, p[2] * w, w]
            })
            .collect();
        Self::from_homogeneous(degree_u, degree_v, knots_u, knots_v, n_u, n_v, control)
    }

    /// Build from an already-homogeneous control grid (u-major).
    #[allow(clippy::too_many_arguments)]
    pub fn from_homogeneous(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        n_u: usize,
        n_v: usize,
        control: Vec<V4>,
    ) -> Result<Self, NurbsError> {
        if control.len() != n_u * n_v {
            return Err(NurbsError::GridSize {
                expected: n_u * n_v,
                got: control.len(),
            });
        }
        for (deg, n, knots) in [(degree_u, n_u, &knots_u), (degree_v, n_v, &knots_v)] {
            if n <= deg {
                return Err(NurbsError::DegreeTooHigh {
                    degree: deg,
                    n_ctrl: n,
                });
            }
            if knots.len() != n + deg + 1 {
                return Err(NurbsError::KnotCount {
                    expected: n + deg + 1,
                    got: knots.len(),
                });
            }
            if knots.windows(2).any(|w| w[1] < w[0]) {
                return Err(NurbsError::KnotsNotMonotonic);
            }
            if !(knots[n] > knots[deg]) {
                return Err(NurbsError::DegenerateDomain);
            }
        }
        if let Some(i) = control.iter().position(|c| !(c[3] > 0.0)) {
            return Err(NurbsError::NonPositiveWeight(i));
        }
        Ok(Self {
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            control,
            n_u,
            n_v,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_homogeneous_unchecked(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        n_u: usize,
        n_v: usize,
        control: Vec<V4>,
    ) -> Self {
        debug_assert_eq!(control.len(), n_u * n_v);
        debug_assert!(knots_valid(degree_u, n_u, &knots_u));
        debug_assert!(knots_valid(degree_v, n_v, &knots_v));
        Self {
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            control,
            n_u,
            n_v,
        }
    }

    pub fn degree_u(&self) -> usize {
        self.degree_u
    }
    pub fn degree_v(&self) -> usize {
        self.degree_v
    }
    pub fn knots_u(&self) -> &[f64] {
        &self.knots_u
    }
    pub fn knots_v(&self) -> &[f64] {
        &self.knots_v
    }
    pub fn n_u(&self) -> usize {
        self.n_u
    }
    pub fn n_v(&self) -> usize {
        self.n_v
    }
    pub fn homogeneous(&self) -> &[V4] {
        &self.control
    }

    pub fn control_point(&self, i: usize, j: usize) -> V3 {
        project(self.control[i * self.n_v + j])
    }

    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.control[i * self.n_v + j][3]
    }

    pub fn is_rational(&self) -> bool {
        self.control.iter().any(|c| (c[3] - 1.0).abs() > 1e-12)
    }

    pub fn domain_u(&self) -> (f64, f64) {
        (self.knots_u[self.degree_u], self.knots_u[self.n_u])
    }

    pub fn domain_v(&self) -> (f64, f64) {
        (self.knots_v[self.degree_v], self.knots_v[self.n_v])
    }

    /// Map `[0, 1]²` onto the parametric domain.
    pub fn param_at(&self, s: f64, t: f64) -> (f64, f64) {
        let (u0, u1) = self.domain_u();
        let (v0, v1) = self.domain_v();
        (
            u0 + s.clamp(0.0, 1.0) * (u1 - u0),
            v0 + t.clamp(0.0, 1.0) * (v1 - v0),
        )
    }

    /// Distinct interior knot values in u, plus the domain ends — the surface's
    /// u-breakpoints. See [`crate::nurbs::NurbsCurve::breakpoints`].
    pub fn breakpoints_u(&self) -> Vec<f64> {
        breaks(&self.knots_u, self.domain_u())
    }

    pub fn breakpoints_v(&self) -> Vec<f64> {
        breaks(&self.knots_v, self.domain_v())
    }

    /// Evaluate at `(u, v)`, clamped to the domain.
    pub fn point(&self, u: f64, v: f64) -> V3 {
        let (u0, u1) = self.domain_u();
        let (v0, v1) = self.domain_v();
        let (u, v) = (u.clamp(u0, u1), v.clamp(v0, v1));
        let (p, q) = (self.degree_u, self.degree_v);

        let su = find_span(p, self.n_u, &self.knots_u, u);
        let sv = find_span(q, self.n_v, &self.knots_v, v);
        let nu = basis_funcs(su, u, p, &self.knots_u);
        let nv = basis_funcs(sv, v, q, &self.knots_v);

        // Contract in v first, then in u — (p+1)(q+1) products either way, but
        // this order touches each control row contiguously.
        let mut acc = [0.0f64; 4];
        for (i, &bu) in nu.iter().enumerate() {
            let mut row = [0.0f64; 4];
            for (j, &bv) in nv.iter().enumerate() {
                let c = self.control[(su - p + i) * self.n_v + (sv - q + j)];
                row[0] += bv * c[0];
                row[1] += bv * c[1];
                row[2] += bv * c[2];
                row[3] += bv * c[3];
            }
            acc[0] += bu * row[0];
            acc[1] += bu * row[1];
            acc[2] += bu * row[2];
            acc[3] += bu * row[3];
        }
        project(acc)
    }

    /// Partial derivatives up to total order `k`.
    ///
    /// `skl[a][b]` is `∂^(a+b) S / ∂uᵃ ∂vᵇ`; `skl[0][0]` is the point. Entries
    /// with `a + b > k` are left zero — A4.4 only defines the triangle, and
    /// filling the rest would mean computing derivatives the caller did not pay
    /// for.
    #[allow(clippy::needless_range_loop)]
    pub fn derivatives(&self, u: f64, v: f64, k: usize) -> Vec<Vec<V3>> {
        let (u0, u1) = self.domain_u();
        let (v0, v1) = self.domain_v();
        let (u, v) = (u.clamp(u0, u1), v.clamp(v0, v1));
        let (p, q) = (self.degree_u, self.degree_v);
        let du = k.min(p);
        let dv = k.min(q);

        let su = find_span(p, self.n_u, &self.knots_u, u);
        let sv = find_span(q, self.n_v, &self.knots_v, v);
        let ndu = ders_basis_funcs(su, u, p, du, &self.knots_u);
        let ndv = ders_basis_funcs(sv, v, q, dv, &self.knots_v);

        // Homogeneous derivatives: numerator (aders) and weight (wders).
        let mut aders = vec![vec![[0.0f64; 3]; k + 1]; k + 1];
        let mut wders = vec![vec![0.0f64; k + 1]; k + 1];

        for a in 0..=du {
            // temp[s] = Σ_r N⁽ᵃ⁾_r(u) · P[su-p+r][sv-q+s]
            let mut temp = vec![[0.0f64; 4]; q + 1];
            for (s, t) in temp.iter_mut().enumerate() {
                for r in 0..=p {
                    let c = self.control[(su - p + r) * self.n_v + (sv - q + s)];
                    let b = ndu[a][r];
                    t[0] += b * c[0];
                    t[1] += b * c[1];
                    t[2] += b * c[2];
                    t[3] += b * c[3];
                }
            }
            let d_max = (k - a).min(dv);
            for b in 0..=d_max {
                let mut acc = [0.0f64; 4];
                for (s, t) in temp.iter().enumerate() {
                    let bv = ndv[b][s];
                    acc[0] += bv * t[0];
                    acc[1] += bv * t[1];
                    acc[2] += bv * t[2];
                    acc[3] += bv * t[3];
                }
                aders[a][b] = [acc[0], acc[1], acc[2]];
                wders[a][b] = acc[3];
            }
        }

        // A4.4 — the rational quotient rule in two parameters.
        let w00 = if wders[0][0].abs() < 1e-300 {
            1.0
        } else {
            wders[0][0]
        };
        let mut skl = vec![vec![[0.0f64; 3]; k + 1]; k + 1];
        for a in 0..=k {
            for b in 0..=(k - a) {
                let mut val = aders[a][b];

                for j in 1..=b {
                    let c = binomial(b, j) * wders[0][j];
                    for e in 0..3 {
                        val[e] -= c * skl[a][b - j][e];
                    }
                }
                for i in 1..=a {
                    let ci = binomial(a, i);
                    for e in 0..3 {
                        val[e] -= ci * wders[i][0] * skl[a - i][b][e];
                    }
                    let mut v2 = [0.0f64; 3];
                    for j in 1..=b {
                        let cj = binomial(b, j) * wders[i][j];
                        for e in 0..3 {
                            v2[e] += cj * skl[a - i][b - j][e];
                        }
                    }
                    for e in 0..3 {
                        val[e] -= ci * v2[e];
                    }
                }
                skl[a][b] = [val[0] / w00, val[1] / w00, val[2] / w00];
            }
        }
        skl
    }

    /// Unit surface normal at `(u, v)`, from the analytic partials.
    ///
    /// # Poles
    ///
    /// At a degenerate parameter — a whole control row collapsed to a point, as
    /// at the poles of a revolved sphere — one partial vanishes and the cross
    /// product carries no direction. The normal is still well defined as a
    /// limit, so rather than return garbage this walks a short distance into the
    /// interior along the degenerate direction and takes the normal there. The
    /// normal field is continuous away from the pole, so the orientation is the
    /// correct one; the error is `O(nudge)` in angle, which at the default
    /// 1e-6-of-domain is far below any tessellation tolerance.
    ///
    /// Returns `None` only if the surface is degenerate over a whole
    /// neighbourhood, which no valid surface is.
    pub fn normal(&self, u: f64, v: f64) -> Option<V3> {
        if let Some(n) = self.normal_raw(u, v) {
            return Some(n);
        }

        let (u0, u1) = self.domain_u();
        let (v0, v1) = self.domain_v();
        let span_u = u1 - u0;
        let span_v = v1 - v0;

        // Nudge toward the interior, starting small (most accurate) and growing
        // until the cross product has a direction.
        for exp in (2..=6).rev() {
            let eps = 10f64.powi(-exp);
            let du = if u <= u0 + span_u * 0.5 { 1.0 } else { -1.0 } * eps * span_u;
            let dv = if v <= v0 + span_v * 0.5 { 1.0 } else { -1.0 } * eps * span_v;
            for (nu, nv) in [(u + du, v), (u, v + dv), (u + du, v + dv)] {
                if let Some(n) = self.normal_raw(nu.clamp(u0, u1), nv.clamp(v0, v1)) {
                    return Some(n);
                }
            }
        }
        None
    }

    /// The normal with no pole handling — `None` where `Sᵤ × Sᵥ` degenerates.
    ///
    /// Two distinct degeneracies have to be caught, and the obvious single test
    /// catches neither reliably:
    ///
    /// * **A collapsed row** (a pole). One partial vanishes. Testing
    ///   `|Sᵤ × Sᵥ|` against `1e-9·|Sᵤ|·|Sᵥ|` does *not* find this — that
    ///   product is itself vanishing, so the ratio stays O(1) and the test
    ///   passes while the direction is pure roundoff. It has to be caught by
    ///   comparing each partial against the *other*.
    /// * **Parallel partials.** Both non-zero but collinear, so there is no
    ///   local frame. That one is what the cross-product ratio is for.
    fn normal_raw(&self, u: f64, v: f64) -> Option<V3> {
        let d = self.derivatives(u, v, 1);
        let (su, sv) = (d[1][0], d[0][1]);
        let (lu, lv) = (v3::norm(su), v3::norm(sv));
        let big = lu.max(lv);
        if big < 1e-300 {
            return None;
        }
        if lu <= 1e-9 * big || lv <= 1e-9 * big {
            return None;
        }
        let n = v3::cross(su, sv);
        if v3::norm(n) <= 1e-9 * lu * lv {
            return None;
        }
        v3::normalize(n)
    }
}

fn breaks(knots: &[f64], (a, b): (f64, f64)) -> Vec<f64> {
    let mut out = vec![a];
    for &k in knots {
        if k > a && k < b && k > *out.last().unwrap() {
            out.push(k);
        }
    }
    out.push(b);
    out
}

/// Convert the three.js parity type, transposing its v-major grid to u-major.
impl TryFrom<&crate::curves::NURBSSurface> for NurbsSurface {
    type Error = NurbsError;

    fn try_from(s: &crate::curves::NURBSSurface) -> Result<Self, Self::Error> {
        let (n_u, n_v) = (s.cols, s.rows);
        if s.control_points.len() != n_u * n_v {
            return Err(NurbsError::GridSize {
                expected: n_u * n_v,
                got: s.control_points.len(),
            });
        }
        // Old layout: control_points[j * cols + i] with i over u, j over v.
        let mut control = vec![[0.0f64; 4]; n_u * n_v];
        for i in 0..n_u {
            for j in 0..n_v {
                let o = s.control_points[j * n_u + i];
                control[i * n_v + j] = [o[0] as f64, o[1] as f64, o[2] as f64, o[3] as f64];
            }
        }
        Self::from_homogeneous(
            s.degree_u,
            s.degree_v,
            s.knots_u.iter().map(|&k| k as f64).collect(),
            s.knots_v.iter().map(|&k| k as f64).collect(),
            n_u,
            n_v,
            control,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::construct;

    fn flat_patch() -> NurbsSurface {
        // Bilinear unit square in the xy-plane.
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            2,
            2,
            &[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn bilinear_patch_interpolates() {
        let s = flat_patch();
        assert!(v3::dist(s.point(0.0, 0.0), [0.0, 0.0, 0.0]) < 1e-15);
        assert!(v3::dist(s.point(1.0, 1.0), [1.0, 1.0, 0.0]) < 1e-15);
        assert!(v3::dist(s.point(0.5, 0.25), [0.5, 0.25, 0.0]) < 1e-15);
    }

    #[test]
    fn flat_patch_normal_is_constant() {
        let s = flat_patch();
        for i in 0..=10 {
            for j in 0..=10 {
                let n = s.normal(i as f64 / 10.0, j as f64 / 10.0).unwrap();
                assert!(v3::dist(n, [0.0, 0.0, 1.0]) < 1e-12, "{n:?}");
            }
        }
    }

    #[test]
    fn sphere_is_exact_and_its_normal_is_radial() {
        let s = construct::sphere([1.0, -2.0, 0.5], 3.0);
        for i in 0..=60 {
            for j in 0..=60 {
                let (u, v) = s.param_at(i as f64 / 60.0, j as f64 / 60.0);
                let p = s.point(u, v);
                let r = v3::dist(p, [1.0, -2.0, 0.5]);
                assert!((r - 3.0).abs() < 1e-12, "radius {r} at ({u}, {v})");

                let n = s.normal(u, v).expect("sphere normal is defined everywhere");
                let radial = v3::normalize(v3::sub(p, [1.0, -2.0, 0.5])).unwrap();
                let algn = v3::dot(n, radial).abs();
                assert!(
                    algn > 1.0 - 1e-6,
                    "normal off-radial by {algn} at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn sphere_poles_still_produce_a_normal() {
        // The pole rows collapse to a point, so Sᵤ × Sᵥ vanishes there — this is
        // exactly the case the raw cross product cannot handle.
        let s = construct::sphere([0.0, 0.0, 0.0], 1.0);
        let (v0, v1) = s.domain_v();
        for &v in &[v0, v1] {
            assert!(s.normal_raw(s.domain_u().0, v).is_none(), "expected a pole");
            let n = s
                .normal(s.domain_u().0, v)
                .expect("nudge should recover it");
            assert!((v3::norm(n) - 1.0).abs() < 1e-12);
            // At a pole the normal must be ±the axis.
            assert!(n[0].abs() < 1e-4 && n[1].abs() < 1e-4, "{n:?}");
        }
    }

    #[test]
    fn analytic_partials_match_central_differences() {
        let s = construct::torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 4.0, 1.25);
        let h = 1e-5;
        // Knots of multiplicity = degree make the second derivative jump, so a
        // central difference across a breakpoint measures nothing meaningful.
        // See the note on the curve's equivalent test.
        let (bu, bv) = (s.breakpoints_u(), s.breakpoints_v());
        for i in 1..20 {
            for j in 1..20 {
                let (u, v) = s.param_at(i as f64 / 20.0, j as f64 / 20.0);
                if bu.iter().any(|&b| (u - b).abs() < 4.0 * h)
                    || bv.iter().any(|&b| (v - b).abs() < 4.0 * h)
                {
                    continue;
                }
                let d = s.derivatives(u, v, 1);
                let fd_u = v3::scale(
                    v3::sub(s.point(u + h, v), s.point(u - h, v)),
                    1.0 / (2.0 * h),
                );
                let fd_v = v3::scale(
                    v3::sub(s.point(u, v + h), s.point(u, v - h)),
                    1.0 / (2.0 * h),
                );
                assert!(v3::dist(d[1][0], fd_u) < 1e-4, "Sᵤ at ({u}, {v})");
                assert!(v3::dist(d[0][1], fd_v) < 1e-4, "Sᵥ at ({u}, {v})");
            }
        }
    }

    #[test]
    fn derivative_table_leaves_the_unpaid_corner_zero() {
        let s = flat_patch();
        let d = s.derivatives(0.5, 0.5, 2);
        assert_eq!(d.len(), 3);
        // a + b > k is not computed.
        assert_eq!(d[2][1], [0.0, 0.0, 0.0]);
        assert_eq!(d[2][2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn parity_type_transposes_correctly() {
        // 3 (u) × 2 (v) grid, distinguishable in both directions.
        let old = crate::curves::NURBSSurface {
            degree_u: 1,
            degree_v: 1,
            knots_u: vec![0.0, 0.0, 0.5, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            // v-major: control_points[j * cols + i], cols = 3
            control_points: vec![
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 5.0, 0.0, 1.0],
                [1.0, 5.0, 0.0, 1.0],
                [2.0, 5.0, 0.0, 1.0],
            ],
            cols: 3,
            rows: 2,
        };
        let new = NurbsSurface::try_from(&old).unwrap();
        for i in 0..=8 {
            for j in 0..=8 {
                let (u, v) = (i as f32 / 8.0, j as f32 / 8.0);
                let a = old.point(u, v);
                let b = new.point(u as f64, v as f64);
                let delta = ((a.x as f64 - b[0]).powi(2) + (a.y as f64 - b[1]).powi(2)).sqrt();
                assert!(delta < 1e-5, "({u}, {v}): {a:?} vs {b:?}");
            }
        }
    }
}
