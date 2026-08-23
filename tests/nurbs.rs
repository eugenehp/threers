//! Stage 0 acceptance suite for the `nurbs` feature.
//!
//! These are the criteria from `docs/brep-nurbs-plan.md` § Stage 0, written as
//! executable checks against the public API. The per-module unit tests inside
//! `src/nurbs/` cover mechanism; this file covers the contract:
//!
//! * partition of unity, over a randomized degree/knot sweep
//! * conics and quadrics are *exact*, not faceted
//! * analytic derivatives agree with Richardson-extrapolated differences
//! * knot insertion, refinement and degree elevation preserve the curve
//! * tessellation honours its chord tolerance
//! * the f32 parity types round-trip into the f64 kernel types
//!
//! Also asserted: the feature is genuinely additive. `curves::NURBSCurve` and
//! `NURBSSurface` still exist and behave identically with `nurbs` on.

#![cfg(feature = "nurbs")]

use threers::nurbs::{
    basis, construct, knot, tessellate, NurbsCurve, NurbsSurface, TessellationOptions,
};

const TAU: f64 = std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// A deterministic LCG. Randomized sweeps must be reproducible — a test that
/// fails only on some runs is a test nobody can act on.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_f64() * (hi - lo + 1) as f64) as usize % (hi - lo + 1)
    }
}

/// A random valid clamped knot vector: `p+1` zeros, sorted interior, `p+1` ones.
fn random_knots(rng: &mut Lcg, degree: usize, n_ctrl: usize) -> Vec<f64> {
    let interior = n_ctrl - degree - 1;
    let mut mid: Vec<f64> = (0..interior).map(|_| rng.next_f64()).collect();
    mid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut k = vec![0.0; degree + 1];
    k.extend(mid);
    k.extend(std::iter::repeat_n(1.0, degree + 1));
    k
}

fn sample_same(a: &NurbsCurve, b: &NurbsCurve, tol: f64, what: &str) {
    for i in 0..=400 {
        let t = i as f64 / 400.0;
        let d = dist(a.point(a.param_at(t)), b.point(b.param_at(t)));
        assert!(d < tol, "{what}: curves differ by {d} at t = {t}");
    }
}

// ---------------------------------------------------------------------------
// basis
// ---------------------------------------------------------------------------

#[test]
fn partition_of_unity_over_a_randomized_sweep() {
    let mut rng = Lcg(0x5eed_1234);
    for _ in 0..200 {
        let degree = rng.next_usize(1, 5);
        let n_ctrl = rng.next_usize(degree + 1, degree + 9);
        let knots = random_knots(&mut rng, degree, n_ctrl);

        for i in 0..=64 {
            let u = i as f64 / 64.0;
            let span = basis::find_span(degree, n_ctrl, &knots, u);
            let n = basis::basis_funcs(span, u, degree, &knots);
            let sum: f64 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "degree {degree}, n {n_ctrl}: Σ N = {sum} at u = {u}"
            );
            assert!(
                n.iter().all(|&x| (-1e-14..=1.0 + 1e-14).contains(&x)),
                "basis outside [0, 1]: {n:?}"
            );
        }
    }
}

#[test]
fn derivative_orders_above_the_degree_vanish() {
    let mut rng = Lcg(0xabc_def);
    for _ in 0..50 {
        let degree = rng.next_usize(1, 4);
        let n_ctrl = rng.next_usize(degree + 1, degree + 6);
        let knots = random_knots(&mut rng, degree, n_ctrl);
        let u = 0.5;
        let span = basis::find_span(degree, n_ctrl, &knots, u);
        let ders = basis::ders_basis_funcs(span, u, degree, degree + 3, &knots);
        for (k, row) in ders.iter().enumerate().skip(degree + 1) {
            assert!(
                row.iter().all(|&x| x == 0.0),
                "order {k} > degree {degree} is non-zero: {row:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// exactness
// ---------------------------------------------------------------------------

#[test]
fn circle_radius_is_exact_to_machine_precision() {
    let c = construct::circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0);
    let mut worst = 0.0f64;
    for i in 0..=10_000 {
        let u = c.param_at(i as f64 / 10_000.0);
        let p = c.point(u);
        worst = worst.max(((p[0] * p[0] + p[1] * p[1]).sqrt() - 1.0).abs());
    }
    // The plan's bar is 1e-12; a rational quadratic actually lands near 1e-16.
    assert!(worst < 1e-12, "worst radius error {worst}");
}

#[test]
fn sphere_and_torus_radii_are_exact() {
    let s = construct::sphere([0.0; 3], 1.0);
    let mut worst = 0.0f64;
    for i in 0..=100 {
        for j in 0..=100 {
            let (u, v) = s.param_at(i as f64 / 100.0, j as f64 / 100.0);
            worst = worst.max((norm(s.point(u, v)) - 1.0).abs());
        }
    }
    assert!(worst < 1e-12, "sphere radius error {worst}");

    let t = construct::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0);
    let mut worst = 0.0f64;
    for i in 0..=100 {
        for j in 0..=100 {
            let (u, v) = t.param_at(i as f64 / 100.0, j as f64 / 100.0);
            let p = t.point(u, v);
            let radial = (p[0] * p[0] + p[1] * p[1]).sqrt();
            worst = worst.max((((radial - 3.0).powi(2) + p[2] * p[2]).sqrt() - 1.0).abs());
        }
    }
    assert!(worst < 1e-12, "torus tube radius error {worst}");
}

#[test]
fn a_faceted_sphere_is_measurably_worse_than_the_exact_one() {
    // The claim that motivates the whole feature: a tessellated primitive is an
    // approximation whose error is bounded by its segment count, and the NURBS
    // sphere is not an approximation at all.
    use threers::SphereGeometry;
    let g = SphereGeometry::new(1.0, 32, 16);
    let pos = &g.get_attribute("position").unwrap().array;

    // Every *vertex* of a UV sphere is on the sphere; the error lives in the
    // faces. Measure a face centroid.
    let idx = g.index.as_ref().unwrap();
    let mut worst_faceted = 0.0f64;
    for tri in idx.chunks_exact(3) {
        let mut c = [0.0f64; 3];
        for &i in tri {
            let o = i as usize * 3;
            c[0] += pos[o] as f64 / 3.0;
            c[1] += pos[o + 1] as f64 / 3.0;
            c[2] += pos[o + 2] as f64 / 3.0;
        }
        worst_faceted = worst_faceted.max((norm(c) - 1.0).abs());
    }

    let s = construct::sphere([0.0; 3], 1.0);
    let mut worst_exact = 0.0f64;
    for i in 0..=50 {
        for j in 0..=50 {
            let (u, v) = s.param_at(i as f64 / 50.0, j as f64 / 50.0);
            worst_exact = worst_exact.max((norm(s.point(u, v)) - 1.0).abs());
        }
    }

    assert!(
        worst_faceted > 1e-3,
        "expected visible faceting, got {worst_faceted}"
    );
    assert!(
        worst_exact < worst_faceted * 1e-9,
        "exact {worst_exact} vs faceted {worst_faceted}"
    );
}

// ---------------------------------------------------------------------------
// derivatives
// ---------------------------------------------------------------------------

#[test]
fn analytic_derivatives_match_richardson_extrapolation() {
    // Richardson: the h and h/2 central differences combine to cancel the O(h²)
    // term, giving an O(h⁴) reference — accurate enough that a real error in the
    // analytic path cannot hide inside the finite-difference error.
    let c = construct::circle([1.0, -2.0, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3.0);
    let breaks = c.breakpoints();
    let h = 1e-3;

    let mut worst = 0.0f64;
    for i in 1..400 {
        let u = c.param_at(i as f64 / 400.0);
        let (u0, u1) = c.domain();
        if u - h < u0 || u + h > u1 {
            continue;
        }
        // C'' jumps across a knot of multiplicity = degree, so a difference
        // straddling one is not measuring the derivative at all.
        if breaks.iter().any(|&b| (u - b).abs() < 2.0 * h) {
            continue;
        }

        let fd = |step: f64| {
            let a = c.point(u - step);
            let b = c.point(u + step);
            [
                (b[0] - a[0]) / (2.0 * step),
                (b[1] - a[1]) / (2.0 * step),
                (b[2] - a[2]) / (2.0 * step),
            ]
        };
        let (d1, d2) = (fd(h), fd(h / 2.0));
        let rich = [
            (4.0 * d2[0] - d1[0]) / 3.0,
            (4.0 * d2[1] - d1[1]) / 3.0,
            (4.0 * d2[2] - d1[2]) / 3.0,
        ];
        worst = worst.max(dist(c.derivatives(u, 1)[1], rich));
    }
    assert!(worst < 1e-8, "worst derivative mismatch {worst}");
}

#[test]
fn surface_normals_are_radial_on_a_sphere_including_the_poles() {
    let s = construct::sphere([2.0, 3.0, -1.0], 1.5);
    let center = [2.0, 3.0, -1.0];
    for i in 0..=80 {
        for j in 0..=80 {
            let (u, v) = s.param_at(i as f64 / 80.0, j as f64 / 80.0);
            let p = s.point(u, v);
            let n = s.normal(u, v).expect("sphere normal is defined everywhere");
            assert!(
                (norm(n) - 1.0).abs() < 1e-12,
                "normal not unit at ({u}, {v})"
            );

            let r = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            let rl = norm(r);
            let dot = (n[0] * r[0] + n[1] * r[1] + n[2] * r[2]) / rl;
            assert!(dot.abs() > 1.0 - 1e-5, "off-radial by {dot} at ({u}, {v})");
        }
    }
}

#[test]
fn cone_apex_and_sphere_poles_still_yield_a_normal() {
    let cases: Vec<(&str, NurbsSurface, f64)> = vec![
        ("sphere", construct::sphere([0.0; 3], 1.0), 0.0),
        (
            "cone",
            construct::cone([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 1.0, 2.0),
            0.0,
        ),
    ];
    for (name, s, v) in cases {
        let (v0, v1) = s.domain_v();
        let v = if v == 0.0 { v0 } else { v1 };
        for i in 0..=8 {
            let (u, _) = s.param_at(i as f64 / 8.0, 0.0);
            let n = s
                .normal(u, v)
                .unwrap_or_else(|| panic!("{name}: no normal at the degenerate row"));
            assert!((norm(n) - 1.0).abs() < 1e-9, "{name}: normal not unit");
        }
    }
}

// ---------------------------------------------------------------------------
// knot operations — shape preservation
// ---------------------------------------------------------------------------

#[test]
fn knot_operations_preserve_randomized_curves() {
    let mut rng = Lcg(0xfeed_face);
    for case in 0..40 {
        let degree = rng.next_usize(2, 4);
        let n_ctrl = rng.next_usize(degree + 2, degree + 7);
        let knots = random_knots(&mut rng, degree, n_ctrl);
        let points: Vec<[f64; 3]> = (0..n_ctrl)
            .map(|_| {
                [
                    rng.next_f64() * 10.0 - 5.0,
                    rng.next_f64() * 10.0 - 5.0,
                    rng.next_f64() * 10.0 - 5.0,
                ]
            })
            .collect();
        let weights: Vec<f64> = (0..n_ctrl).map(|_| 0.25 + rng.next_f64() * 3.0).collect();

        let c = NurbsCurve::new(degree, knots, &points, Some(&weights))
            .unwrap_or_else(|e| panic!("case {case}: {e}"));

        let at = 0.1 + rng.next_f64() * 0.8;
        sample_same(&c, &knot::insert_knot(&c, at, 1), 1e-11, "insert");
        sample_same(
            &c,
            &knot::refine(&c, &[0.15, 0.4, 0.65, 0.9]),
            1e-11,
            "refine",
        );
        sample_same(&c, &knot::elevate_degree(&c, 1), 1e-9, "elevate 1");
        sample_same(&c, &knot::elevate_degree(&c, 2), 1e-9, "elevate 2");

        if let Some((left, right)) = knot::split(&c, at) {
            for i in 0..=100 {
                let t = i as f64 / 100.0;
                let (u0, u1) = c.domain();
                let ul = u0 + (at - u0) * t;
                let ur = at + (u1 - at) * t;
                assert!(dist(left.point(ul), c.point(ul)) < 1e-10, "split left");
                assert!(dist(right.point(ur), c.point(ur)) < 1e-10, "split right");
            }
        }
    }
}

#[test]
fn elevation_and_refinement_keep_a_circle_exactly_circular() {
    let c = construct::circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 7.0);
    let variants = [
        ("refined", knot::refine(&c, &[0.05, 0.3, 0.55, 0.8, 0.95])),
        ("elevated", knot::elevate_degree(&c, 3)),
        (
            "both",
            knot::elevate_degree(&knot::refine(&c, &[0.2, 0.7]), 1),
        ),
    ];
    for (name, v) in variants {
        let mut worst = 0.0f64;
        for i in 0..=2000 {
            let u = v.param_at(i as f64 / 2000.0);
            let p = v.point(u);
            worst = worst.max(((p[0] * p[0] + p[1] * p[1]).sqrt() - 7.0).abs());
        }
        assert!(worst < 1e-10, "{name}: radius error {worst}");
    }
}

// ---------------------------------------------------------------------------
// tessellation
// ---------------------------------------------------------------------------

#[test]
fn tessellation_honours_its_chord_tolerance() {
    let surfaces: Vec<(&str, NurbsSurface)> = vec![
        ("sphere", construct::sphere([0.0; 3], 2.0)),
        (
            "torus",
            construct::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0),
        ),
        (
            "cylinder",
            construct::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.5, 4.0),
        ),
    ];

    for (name, s) in &surfaces {
        for &tol in &[1e-2, 1e-3] {
            let opts = TessellationOptions::with_tolerance(tol);
            let (pu, pv) = tessellate::sample_grid(s, &opts);
            assert!(
                tessellate::sample_grid_meets_tolerance(s, &pu, &pv, tol),
                "{name} @ {tol}: sampler reported it did not converge"
            );

            // Independent check: measure the grid-edge midpoint deviation
            // directly instead of trusting the sampler's own report.
            let mut worst = 0.0f64;
            for w in pu.windows(2) {
                for &v in &pv {
                    let (a, b) = (s.point(w[0], v), s.point(w[1], v));
                    let mid = [
                        0.5 * (a[0] + b[0]),
                        0.5 * (a[1] + b[1]),
                        0.5 * (a[2] + b[2]),
                    ];
                    worst = worst.max(dist(mid, s.point(0.5 * (w[0] + w[1]), v)));
                }
            }
            for w in pv.windows(2) {
                for &u in &pu {
                    let (a, b) = (s.point(u, w[0]), s.point(u, w[1]));
                    let mid = [
                        0.5 * (a[0] + b[0]),
                        0.5 * (a[1] + b[1]),
                        0.5 * (a[2] + b[2]),
                    ];
                    worst = worst.max(dist(mid, s.point(u, 0.5 * (w[0] + w[1]))));
                }
            }
            assert!(worst <= tol, "{name} @ {tol}: chord deviation {worst}");
        }
    }
}

#[test]
fn tessellated_geometry_is_well_formed() {
    let s = construct::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0);
    let g = threers::NurbsGeometry::with_tolerance(&s, 1e-3);

    let n = g.get_attribute("position").unwrap().count();
    assert_eq!(g.get_attribute("normal").unwrap().count(), n);
    assert_eq!(g.get_attribute("uv").unwrap().count(), n);

    let idx = g.index.as_ref().expect("indexed");
    assert_eq!(idx.len() % 3, 0);
    assert!(idx.iter().all(|&i| (i as usize) < n));

    // Every emitted normal is unit — the pole fallback must not leak a zero.
    let nor = &g.get_attribute("normal").unwrap().array;
    for c in nor.chunks_exact(3) {
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        assert!((l - 1.0).abs() < 1e-4, "non-unit normal, length {l}");
    }
}

#[test]
fn retessellation_at_a_finer_tolerance_gets_strictly_closer() {
    // The Stage 1 payoff in miniature: the surface is retained, so the mesh can
    // be regenerated at any resolution instead of being fixed at build time.
    let s = construct::sphere([0.0; 3], 1.0);
    let mut last = f64::INFINITY;
    for &tol in &[1e-1, 1e-2, 1e-3] {
        let opts = TessellationOptions::with_tolerance(tol);
        let (pu, pv) = tessellate::sample_grid(&s, &opts);
        let mut worst = 0.0f64;
        for w in pu.windows(2) {
            for &v in &pv {
                let (a, b) = (s.point(w[0], v), s.point(w[1], v));
                let mid = [
                    0.5 * (a[0] + b[0]),
                    0.5 * (a[1] + b[1]),
                    0.5 * (a[2] + b[2]),
                ];
                worst = worst.max((1.0 - norm(mid)).abs());
            }
        }
        assert!(
            worst < last,
            "tolerance {tol} did not improve on the previous"
        );
        last = worst;
    }
}

// ---------------------------------------------------------------------------
// the feature is additive
// ---------------------------------------------------------------------------

#[test]
fn the_three_js_parity_types_still_work_and_round_trip() {
    use threers::curves::Curve3;

    let w = std::f64::consts::FRAC_1_SQRT_2 as f32;
    let old = threers::NURBSCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![[1.0, 0.0, 0.0, 1.0], [w, w, 0.0, w], [0.0, 1.0, 0.0, 1.0]],
    );

    // Still evaluable on its own terms.
    let mid = old.get_point(0.5);
    assert!((mid.length() - 1.0).abs() < 1e-5, "parity type broke");

    // And convertible into the kernel type, which agrees with it.
    let new = NurbsCurve::try_from(&old).expect("valid definition");
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let a = old.get_point(t);
        let b = Curve3::get_point(&new, t);
        assert!((a - b).length() < 1e-5, "t = {t}: {a:?} vs {b:?}");
    }
}

#[test]
fn the_parity_surface_transposes_into_the_kernel_layout() {
    // The two types disagree about grid order — the parity type is v-major and
    // the kernel is u-major (Piegl & Tiller / STEP). Getting this backwards
    // produces a plausible-looking but transposed surface, so pick a control
    // grid that is not symmetric under transposition.
    let old = threers::NURBSSurface {
        degree_u: 1,
        degree_v: 1,
        knots_u: vec![0.0, 0.0, 0.5, 1.0, 1.0],
        knots_v: vec![0.0, 0.0, 1.0, 1.0],
        control_points: vec![
            [0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [6.0, 0.0, 1.0, 1.0],
            [0.0, 9.0, 0.0, 1.0],
            [3.0, 9.0, 0.0, 1.0],
            [6.0, 9.0, 1.0, 1.0],
        ],
        cols: 3,
        rows: 2,
    };
    let new = NurbsSurface::try_from(&old).expect("valid definition");
    for i in 0..=10 {
        for j in 0..=10 {
            let (s, t) = (i as f32 / 10.0, j as f32 / 10.0);
            let a = old.point(s, t);
            let b = new.point(s as f64, t as f64);
            let d = dist([a.x as f64, a.y as f64, a.z as f64], b);
            assert!(d < 1e-5, "({s}, {t}): differ by {d}");
        }
    }
}

#[test]
fn malformed_definitions_are_reported_not_panicked() {
    // NURBS data arrives from files and from JS; a bad knot vector is a data
    // error to surface, not an index panic in the middle of a render.
    let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
    assert!(
        NurbsCurve::new(5, vec![0.0; 9], &pts, None).is_err(),
        "degree > points"
    );
    assert!(
        NurbsCurve::new(1, vec![0.0; 3], &pts, None).is_err(),
        "short knots"
    );
    assert!(
        NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 0.5, 1.0], &pts, None).is_err(),
        "non-monotonic knots"
    );
    assert!(
        NurbsCurve::new(1, vec![0.0; 5], &pts, None).is_err(),
        "empty domain"
    );
    assert!(
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            &pts,
            Some(&[1.0, -1.0, 1.0])
        )
        .is_err(),
        "negative weight"
    );
    assert!(
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            &pts,
            Some(&[1.0, f64::NAN, 1.0])
        )
        .is_err(),
        "NaN weight must be rejected, not propagated into the mesh"
    );
}

#[test]
fn revolve_and_extrude_agree_with_their_named_shortcuts() {
    // `cylinder` is `revolve(line)` and nothing more; if the shortcut and the
    // general path ever disagree, one of them is wrong.
    //
    // The profile has to start where `cylinder` starts its own. The seam
    // direction is `construct::perpendicular(axis)` — for `+Z` that is `+Y`,
    // not the `+X` one might assume — and placing it elsewhere yields the same
    // cylinder under a rotated parameterization, which is a different surface
    // as far as `(u, v)` is concerned.
    let x = construct::perpendicular([0.0, 0.0, 1.0]);
    let profile = construct::line(
        [2.0 * x[0], 2.0 * x[1], 2.0 * x[2]],
        [2.0 * x[0], 2.0 * x[1], 2.0 * x[2] + 5.0],
    );
    let by_revolve = construct::revolve(&profile, [0.0; 3], [0.0, 0.0, 1.0], TAU);
    let shortcut = construct::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0);

    for i in 0..=30 {
        for j in 0..=30 {
            let (s, t) = (i as f64 / 30.0, j as f64 / 30.0);
            let (u1, v1) = by_revolve.param_at(s, t);
            let (u2, v2) = shortcut.param_at(s, t);
            let d = dist(by_revolve.point(u1, v1), shortcut.point(u2, v2));
            assert!(d < 1e-12, "({s}, {t}): differ by {d}");
        }
    }
}

#[test]
fn a_traced_curve_can_be_read_as_a_curve() {
    // A traced intersection is kept as the points it was traced through, and
    // that freezes it: whoever samples it first fixes the boundary, and the face
    // across it, a later refinement and a second boolean all have to chase that
    // sampling or come apart from it. Read as a curve it can be evaluated
    // anywhere, so two askers at one parameter get one point.
    //
    // Interpolating, not approximating — the input points are already vertices
    // two faces share, and moving them would move a boundary.
    use std::f64::consts::TAU;
    let n = 24;
    let ring: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = TAU * i as f64 / n as f64;
            [2.0 * t.cos(), 2.0 * t.sin(), 0.0]
        })
        .collect();

    let (curve, _) =
        threers::nurbs::construct::interpolate(&ring, true).expect("a ring interpolates");
    let (lo, hi) = curve.domain();

    // Every input point is on it.
    for (i, p) in ring.iter().enumerate() {
        let u = lo + (hi - lo) * i as f64 / n as f64;
        let q = curve.point(u);
        let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2)).sqrt();
        assert!(
            d < 1e-9,
            "point {i} is {d} off the curve that should pass through it"
        );
    }

    // And it closes *exactly*, which a distance test on samples cannot promise.
    let (a, b) = (curve.point(lo), curve.point(hi));
    let gap = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    assert!(gap < 1e-12, "the ends are {gap} apart");

    // Between the points it is a circle to well under a tolerance the polyline
    // could not hold: the chord of a 24-gon on this radius sags 0.0171.
    let mut worst: f64 = 0.0;
    for k in 0..500 {
        let u = lo + (hi - lo) * k as f64 / 500.0;
        let q = curve.point(u);
        worst = worst.max((q[0].hypot(q[1]) - 2.0).abs());
    }
    assert!(
        worst < 1e-3,
        "the curve wanders {worst} off the circle it interpolates"
    );
}
