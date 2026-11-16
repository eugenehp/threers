//! FEM against closed-form answers: patch tests the elements must pass
//! exactly, and beam and conduction problems they must converge to.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;
use threers_fem::{Elastic, FemError, Heat, LinearSolver, Material, Order, TetMesh};

const EPS: f64 = 1e-9;

/// A tetrahedral sphere from the mesher, through its SDF.
fn sphere(r: f64, h: f64) -> TetMesh {
    TetMesh::from_sdf([-r; 3], [r; 3], h, |p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r).unwrap()
}

#[test]
fn uniform_tension_on_an_unstructured_mesh() {
    // Rollers on three faces, a total force on the opposite x face: nearly
    // a uniform stress state, here on a mesh from the mesher rather than the
    // tidy grid. The mesher bevels the box's edges by a fraction of the
    // spacing, so the stress is the force over the *meshed* section, and
    // the force goes on the faces that lie flat on the end — a bevel's
    // triangles have a corner off it.
    let (l, e, nu, force) = (2.0, 1000.0, 0.3, 5.0);
    let mesh = TetMesh::from_sdf([0.0; 3], [l, 1.0, 1.0], 0.21, |p| {
        let q = [p[0] - l / 2.0, p[1] - 0.5, p[2] - 0.5];
        let d = [q[0].abs() - l / 2.0, q[1].abs() - 0.5, q[2].abs() - 0.5];
        let out = (d[0].max(0.0).powi(2) + d[1].max(0.0).powi(2) + d[2].max(0.0).powi(2)).sqrt();
        out + d[0].max(d[1]).max(d[2]).min(0.0)
    })
    .unwrap();
    let tol = 1e-6;
    // The mean meshed section; the bevels take a sliver off the full 1×1.
    let area = mesh.volume() / l;
    assert!(area > 0.95 && area < 1.0, "section {area}");
    let sigma = force / area;
    for order in [Order::Linear, Order::Quadratic] {
        let s = Elastic::new(&mesh, Material::new(e, nu))
            .order(order)
            .displace(|p| p[0] < tol, [Some(0.0), None, None])
            .displace(|p| p[1] < tol, [None, Some(0.0), None])
            .displace(|p| p[2] < tol, [None, None, Some(0.0)])
            .force(|p| p[0] > l - tol, [force, 0.0, 0.0])
            .solve()
            .unwrap();
        let (i, _) = s
            .nodes
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let d = |p: &[f64; 3]| (p[0] - 1.0).powi(2) + (p[1] - 0.5).powi(2) + (p[2] - 0.5).powi(2);
                d(a.1).total_cmp(&d(b.1))
            })
            .unwrap();
        let p = s.nodes[i];
        assert!((s.stress[i][0] / sigma - 1.0).abs() < 0.03, "{order:?}: σxx {} vs {sigma}", s.stress[i][0]);
        let ux = s.displacement[i][0];
        assert!((ux / (sigma / e * p[0]) - 1.0).abs() < 0.03, "{order:?}: ux {ux}");
        // Equilibrium is exact whatever the mesh.
        assert!((s.reaction[0] + force).abs() < 1e-9);
        assert!(s.reaction[1].abs() < 1e-9 && s.reaction[2].abs() < 1e-9);
    }
}

#[test]
fn patch_test_on_the_grid_is_exact_to_rounding() {
    let mesh = TetMesh::box_grid([0.0; 3], [3.0, 1.0, 2.0], [3, 2, 2]);
    for order in [Order::Linear, Order::Quadratic] {
        let s = Elastic::new(&mesh, Material::new(70.0, 0.25))
            .order(order)
            .displace(|p| p[0] < EPS, [Some(0.0), None, None])
            .displace(|p| p[1] < EPS, [None, Some(0.0), None])
            .displace(|p| p[2] < EPS, [None, None, Some(0.0)])
            .force(|p| p[2] > 2.0 - EPS, [0.0, 0.0, 9.0]) // over a 3×1 face: σzz = 3
            .solve()
            .unwrap();
        for (p, (d, st)) in s.nodes.iter().zip(s.displacement.iter().zip(&s.stress)) {
            let ez = 3.0 / 70.0;
            assert!((d[2] - ez * p[2]).abs() < 1e-12, "{order:?}");
            assert!((d[0] + 0.25 * ez * p[0]).abs() < 1e-12, "{order:?}");
            assert!((st[2] - 3.0).abs() < 1e-10 && st[0].abs() < 1e-10 && st[5].abs() < 1e-10);
        }
        assert!((s.reaction[2] + 9.0).abs() < 1e-10);
    }
}

/// Timoshenko tip deflection of an end-loaded cantilever.
fn cantilever_deflection(f: f64, l: f64, b: f64, h: f64, e: f64, nu: f64) -> f64 {
    let i = b * h.powi(3) / 12.0;
    let g = e / (2.0 * (1.0 + nu));
    let kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu);
    f * l.powi(3) / (3.0 * e * i) + f * l / (kappa * g * b * h)
}

#[test]
fn quadratic_cantilever_matches_beam_theory_and_linear_locks() {
    let (l, b, h, e, nu, f): (f64, f64, f64, f64, f64, f64) = (10.0, 1.0, 1.0, 1000.0, 0.3, -1.0);
    let mesh = TetMesh::box_grid([0.0; 3], [l, b, h], [20, 2, 2]);
    let want = cantilever_deflection(f.abs(), l, b, h, e, nu);
    let tip = |order| {
        let s = Elastic::new(&mesh, Material::new(e, nu))
            .order(order)
            .fix(|p| p[0] < EPS)
            .force(|p| p[0] > l - EPS, [0.0, f, 0.0])
            .solve()
            .unwrap();
        let mut y = 0.0;
        let mut n = 0.0;
        for (p, d) in s.nodes.iter().zip(&s.displacement) {
            if p[0] > l - EPS {
                y += d[1];
                n += 1.0;
            }
        }
        (-(y / n), s)
    };
    let (quad, s) = tip(Order::Quadratic);
    assert!((quad / want - 1.0).abs() < 0.02, "Tet10 tip {quad} vs {want}");
    // Root stress, away from the clamp's corner singularity: M c / I at
    // x = L/2 on the top fibre.
    let i = b * h.powi(3) / 12.0;
    let want_sigma = f.abs() * (l / 2.0) * (h / 2.0) / i;
    let (k, _) = s
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| (p[0] - l / 2.0).abs() < EPS && (p[1] - h).abs() < EPS)
        .min_by(|a, b| (a.1[2] - 0.5).abs().total_cmp(&(b.1[2] - 0.5).abs()))
        .unwrap();
    assert!((s.stress[k][0] / want_sigma - 1.0).abs() < 0.05, "σ {} vs {want_sigma}", s.stress[k][0]);
    // Reaction balances the load.
    assert!((s.reaction[1] + f).abs() < 1e-9);
    // Constant-strain tetrahedra lock: far too stiff at this resolution.
    let (lin, _) = tip(Order::Linear);
    assert!(lin < 0.6 * want, "Tet4 tip {lin} should lock well below {want}");
}

#[test]
fn iterative_and_direct_solvers_agree() {
    let mesh = TetMesh::box_grid([0.0; 3], [4.0, 1.0, 1.0], [8, 2, 2]);
    let run = |solver| {
        Elastic::new(&mesh, Material::new(1.0, 0.3))
            .solver(solver)
            .fix(|p| p[0] < EPS)
            .gravity([0.0, -1.0, 0.0])
            .solve()
            .unwrap()
    };
    let a = run(LinearSolver::Direct);
    let b = run(LinearSolver::ITERATIVE);
    assert!((a.max_displacement() - b.max_displacement()).abs() < 1e-7 * a.max_displacement());
    // Gravity on a 4 m³ block of density 1 weighs 4.
    assert!((a.reaction[1] - 4.0).abs() < 1e-9);
}

#[test]
fn unsupported_and_misplaced_problems_fail_clearly() {
    let mesh = TetMesh::box_grid([0.0; 3], [1.0; 3], [2, 2, 2]);
    let free = Elastic::new(&mesh, Material::new(1.0, 0.3))
        .force(|p| p[0] > 1.0 - EPS, [1.0, 0.0, 0.0])
        .solve();
    assert!(matches!(free, Err(FemError::InvalidInput(_))));
    let missed = Elastic::new(&mesh, Material::new(1.0, 0.3))
        .fix(|p| p[0] < -5.0)
        .solve();
    assert_eq!(missed.unwrap_err(), FemError::EmptyRegion("support"));
    // Held on one point only: still free to rotate.
    let pinned = Elastic::new(&mesh, Material::new(1.0, 0.3))
        .order(Order::Linear)
        .solver(LinearSolver::Direct)
        .fix(|p| p == [0.0, 0.0, 0.0])
        .force(|p| p[0] > 1.0 - EPS, [0.0, 1.0, 0.0])
        .solve();
    assert_eq!(pinned.unwrap_err(), FemError::NotPositiveDefinite);
    let bad = Elastic::new(&mesh, Material::new(1.0, 0.5)).fix(|p| p[0] < EPS).solve();
    assert!(matches!(bad, Err(FemError::InvalidInput(_))));
}

#[test]
fn cantilever_first_frequencies_match_euler_bernoulli() {
    // Slender, so shear and rotary inertia are small corrections.
    let (l, b, h, e, rho) = (20.0, 1.0, 1.0, 1000.0, 1.0);
    let mesh = TetMesh::box_grid([0.0; 3], [l, b, h], [40, 2, 2]);
    let modes = Elastic::new(&mesh, Material::new(e, 0.3).density(rho))
        .fix(|p| p[0] < EPS)
        .modes(4)
        .unwrap();
    let i = b * h.powi(3) / 12.0;
    let a = b * h;
    let f = |beta_l: f64| beta_l * beta_l / (2.0 * PI * l * l) * (e * i / (rho * a)).sqrt();
    let f1 = f(1.875_104_07);
    let got = modes.frequencies();
    // The square section bends the same both ways: two modes at f1.
    for k in 0..2 {
        assert!((got[k] / f1 - 1.0).abs() < 0.02, "mode {k}: {} vs {f1}", got[k]);
    }
    let f2 = f(4.694_091_13);
    assert!((got[2] / f2 - 1.0).abs() < 0.04, "mode 2: {} vs {f2}", got[2]);
    assert!(modes.modes[0].shape.iter().all(|d| d.iter().all(|v| v.is_finite())));
}

#[test]
fn a_free_part_has_six_rigid_modes() {
    let mesh = TetMesh::box_grid([0.0; 3], [2.0, 1.0, 1.0], [4, 2, 2]);
    let modes = Elastic::new(&mesh, Material::new(1000.0, 0.3)).modes(8).unwrap();
    let f = modes.frequencies();
    for k in 0..6 {
        assert!(f[k] < 1e-3 * f[6], "rigid mode {k} at {} vs first elastic {}", f[k], f[6]);
    }
    assert!(f[6] > 0.5);
}

#[test]
fn heat_with_uniform_source_is_exact_for_quadratic_elements() {
    // T = Q x (L − x) / 2k between two faces held at zero, insulated
    // elsewhere — a quadratic, which ten-node elements represent exactly.
    let (l, k, q) = (2.0, 3.0, 6.0);
    let mesh = TetMesh::box_grid([0.0; 3], [l, 0.5, 0.5], [3, 1, 1]);
    let s = Heat::new(&mesh, k)
        .temperature(|p| p[0] < EPS, 0.0)
        .temperature(|p| p[0] > l - EPS, 0.0)
        .source(q)
        .solve()
        .unwrap();
    for x in [0.1, 0.37, 1.0, 1.81] {
        let t = s.temperature_at([x, 0.2, 0.3]);
        let want = q * x * (l - x) / (2.0 * k);
        assert!((t - want).abs() < 1e-10, "{t} vs {want}");
    }
    // All the heat generated leaves through the held faces.
    assert!((s.fixed_heat_inflow + q * l * 0.25).abs() < 1e-10);
}

#[test]
fn heat_convection_matches_the_one_dimensional_solution() {
    // Held at T0 at x = 0, convecting to T∞ at x = L:
    // T(L) = (k T0 / L + h T∞) / (k / L + h).
    let (l, k, h, t0, tinf) = (1.0, 2.0, 5.0, 100.0, 20.0);
    let mesh = TetMesh::box_grid([0.0; 3], [l, 0.3, 0.3], [4, 1, 1]);
    for order in [Order::Linear, Order::Quadratic] {
        let s = Heat::new(&mesh, k)
            .order(order)
            .temperature(|p| p[0] < EPS, t0)
            .convection(|p| p[0] > l - EPS, h, tinf)
            .solve()
            .unwrap();
        let tl = (k * t0 / l + h * tinf) / (k / l + h);
        assert!((s.temperature_at([l, 0.1, 0.1]) - tl).abs() < 1e-9);
        let q = h * (tl - tinf) * 0.09;
        assert!((s.convected_out - q).abs() < 1e-9);
        assert!((s.fixed_heat_inflow - q).abs() < 1e-9);
    }
}

#[test]
fn heat_in_a_sphere_converges() {
    // T(r) = Q (R² − r²) / 6k with the surface at zero.
    let (r, k, q) = (1.0, 1.0, 6.0);
    let mesh = sphere(r, 0.2);
    let s = Heat::new(&mesh, k)
        .temperature(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() > r - 1e-6, 0.0)
        .source(q)
        .solve()
        .unwrap();
    let centre = s.temperature_at([0.0, 0.0, 0.0]);
    assert!((centre - 1.0).abs() < 0.01, "centre {centre}");
}
