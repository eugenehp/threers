//! BEM against closed-form answers on spheres and boxes.

use std::f64::consts::PI;
use threers::geometries::{BoxGeometry, IcosahedronGeometry};
use threers_fem::bem::{Helmholtz, Laplace, Potential};
use threers_fem::{Complex, TriSurface};

/// `20·(detail + 1)²` triangles, as three.js counts detail.
fn icosphere(r: f64, detail: usize) -> TriSurface {
    TriSurface::from_geometry(&IcosahedronGeometry::new(r as f32, detail)).unwrap()
}

fn len(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

#[test]
fn interior_dirichlet_recovers_the_flux_of_a_linear_field() {
    // u = x is harmonic; on the sphere its outward flux is n_x.
    let s = icosphere(1.0, 7);
    let lap = Laplace::interior(&s).unwrap();
    let sol = lap.solve(|c, _| Potential::Value(c[0])).unwrap();
    let mut err = 0.0f64;
    for (q, n) in sol.flux.iter().zip(lap.normals()) {
        err = err.max((q - n[0]).abs());
    }
    assert!(err < 0.06, "max flux error {err}");
    for p in [[0.2, 0.1, -0.3], [0.0, 0.5, 0.0], [-0.6, 0.0, 0.1]] {
        let u = sol.potential_at(p);
        assert!((u - p[0]).abs() < 0.01, "u at {p:?}: {u}");
    }
}

/// Solve u = 2x + 1 in a 2×1×1 box with `seg` segments per unit length:
/// values on the x faces, flux (zero) elsewhere. Returns the mean and
/// largest flux error on the x faces, the largest potential error on the
/// others, and the potential at an interior point.
fn mixed_box(seg: u32) -> (f64, f64, f64, f64) {
    let g = BoxGeometry::with_segments(2.0, 1.0, 1.0, 2 * seg, seg, seg);
    let s = TriSurface::from_geometry(&g).unwrap();
    let lap = Laplace::interior(&s).unwrap();
    let sol = lap
        .solve(|c, n| {
            if n[0].abs() > 0.5 {
                Potential::Value(2.0 * c[0] + 1.0)
            } else {
                Potential::Flux(2.0 * n[0])
            }
        })
        .unwrap();
    let mut flux_err = Vec::new();
    let mut u_err: f64 = 0.0;
    for (i, n) in lap.normals().iter().enumerate() {
        if n[0].abs() > 0.5 {
            flux_err.push((sol.flux[i] - 2.0 * n[0]).abs());
        } else {
            let want = 2.0 * lap.centroids()[i][0] + 1.0;
            u_err = u_err.max((sol.potential[i] - want).abs());
        }
    }
    let mean = flux_err.iter().sum::<f64>() / flux_err.len() as f64;
    let max = flux_err.iter().copied().fold(0.0, f64::max);
    (mean, max, u_err, sol.potential_at([0.3, 0.1, 0.0]))
}

#[test]
fn interior_mixed_problem_on_a_box_converges() {
    // Flux is the hard half for constant elements: it is singular at the
    // box's edges, and the triangles against them are the worst. What
    // matters is that refining shrinks it.
    let coarse = mixed_box(4);
    let fine = mixed_box(8);
    assert!(fine.0 < 0.65 * coarse.0, "mean flux error {} -> {}", coarse.0, fine.0);
    assert!(fine.0 < 0.07, "mean flux error {}", fine.0);
    assert!(fine.2 < 0.03, "potential error {}", fine.2);
    assert!((fine.3 - 1.6).abs() < 0.01, "interior u {}", fine.3);
}

#[test]
fn exterior_sphere_potential_and_charge() {
    let r = 2.0;
    let s = icosphere(r, 7);
    let lap = Laplace::exterior(&s).unwrap();
    // Held at 1: u = R/r outside, total flux −4πR.
    let sol = lap.solve(|_, _| Potential::Value(1.0)).unwrap();
    for p in [[5.0, 0.0, 0.0], [0.0, -3.0, 2.5], [10.0, 10.0, 0.0]] {
        let u = sol.potential_at(p);
        let want = r / len(p);
        assert!((u / want - 1.0).abs() < 0.02, "u at {p:?}: {u} vs {want}");
    }
    let c = lap.capacitance().unwrap();
    assert!((c / (4.0 * PI * r) - 1.0).abs() < 0.02, "C {c}");
    // Inside an exterior problem's body the formula gives zero.
    assert!(sol.potential_at([0.1, 0.2, 0.0]).abs() < 0.02);
    // Neumann: flux −1/R² everywhere is u = 1/r.
    let sol = lap.solve(|_, _| Potential::Flux(-1.0 / (r * r))).unwrap();
    let mean = sol.potential.iter().sum::<f64>() / sol.potential.len() as f64;
    assert!((mean - 1.0 / r).abs() < 0.02 / r, "{mean}");
}

#[test]
fn interior_neumann_everywhere_is_refused() {
    let lap = Laplace::interior(&icosphere(1.0, 1)).unwrap();
    assert!(lap.solve(|_, n| Potential::Flux(n[0])).is_err());
}

/// Pulsating sphere: p = A e^{ikr}/r with ∂p/∂r = q at r = a.
fn monopole(a: f64, k: f64, q: Complex, r: f64) -> Complex {
    let amp = q * (a * a) * Complex::cis(-k * a) / Complex::new(-1.0, k * a);
    amp * Complex::cis(k * r) / r
}

#[test]
fn pulsating_sphere_radiates_a_monopole() {
    let (a, k) = (1.0, 2.0);
    let s = icosphere(a, 7);
    let h = Helmholtz::new(&s, k).unwrap();
    assert!(h.elements_per_wavelength() > 10.0);
    let sol = h.radiate(|_, _| Complex::ONE).unwrap();
    for p in [[3.0, 0.0, 0.0], [0.0, 2.0, 2.0], [-1.5, -1.5, 1.0]] {
        let got = sol.pressure_at(p);
        let want = monopole(a, k, Complex::ONE, len(p));
        assert!((got - want).abs() < 0.03 * want.abs(), "{p:?}: {got:?} vs {want:?}");
    }
    // Far field: F = A, the r → ∞ limit of r e^{-ikr} p.
    let want = monopole(a, k, Complex::ONE, 1.0) * Complex::cis(-k);
    let ff = sol.far_field([0.0, 0.0, 1.0]);
    assert!((ff - want).abs() < 0.03 * want.abs(), "{ff:?} vs {want:?}");
}

#[test]
fn chief_removes_the_irregular_frequency() {
    // ka = π is the first interior Dirichlet eigenvalue of the unit sphere.
    let (a, k) = (1.0, PI);
    let s = icosphere(a, 7);
    let probe = [2.5, 0.0, 0.0];
    let want = monopole(a, k, Complex::ONE, len(probe));
    let plain = Helmholtz::new(&s, k)
        .unwrap()
        .chief_points(vec![])
        .radiate(|_, _| Complex::ONE)
        .unwrap()
        .pressure_at(probe);
    let fixed_sol = Helmholtz::new(&s, k).unwrap().radiate(|_, _| Complex::ONE).unwrap();
    assert!(!fixed_sol.chief_points.is_empty());
    let fixed = fixed_sol.pressure_at(probe);
    let e_plain = (plain - want).abs() / want.abs();
    let e_fixed = (fixed - want).abs() / want.abs();
    assert!(e_fixed < 0.04, "with CHIEF {e_fixed}");
    assert!(e_plain > 3.0 * e_fixed, "without CHIEF {e_plain} should be much worse than {e_fixed}");
}

/// Spherical Hankel h_n^(1) and spherical Bessel j_n for n = 0..=nmax.
fn spherical(nmax: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    // j by Miller's downward recurrence, normalised to j0 = sin x / x.
    let start = nmax + 30 + x as usize;
    let mut j = vec![0.0; start + 2];
    j[start + 1] = 0.0;
    j[start] = 1e-30;
    for n in (1..=start).rev() {
        j[n - 1] = (2 * n + 1) as f64 / x * j[n] - j[n + 1];
    }
    let norm = (x.sin() / x) / j[0];
    let j: Vec<f64> = j[..=nmax].iter().map(|v| v * norm).collect();
    // y upward, which is stable.
    let mut y = vec![-x.cos() / x, -x.cos() / (x * x) - x.sin() / x];
    for n in 1..nmax {
        let next = (2 * n + 1) as f64 / x * y[n] - y[n - 1];
        y.push(next);
    }
    y.truncate(nmax + 1);
    (j, y)
}

fn legendre(nmax: usize, x: f64) -> Vec<f64> {
    let mut p = vec![1.0, x];
    for n in 1..nmax {
        p.push(((2 * n + 1) as f64 * x * p[n] - n as f64 * p[n - 1]) / (n + 1) as f64);
    }
    p.truncate(nmax + 1);
    p
}

#[test]
fn rigid_sphere_scattering_matches_the_series() {
    let (a, k) = (1.0, 1.5);
    let s = icosphere(a, 7);
    let sol = Helmholtz::new(&s, k)
        .unwrap()
        .scatter_plane_wave([0.0, 0.0, 1.0], Complex::ONE)
        .unwrap();
    let nmax = 25;
    let (ja, ya) = spherical(nmax + 1, k * a);
    // Derivative f_n' = f_{n-1} − (n+1)/x f_n, with f_{-1} from f_1:
    // j0' = −j1, y0' = −y1.
    let d = |f: &[f64], n: usize, x: f64| if n == 0 { -f[1] } else { f[n - 1] - (n + 1) as f64 / x * f[n] };
    for p in [[0.0, 0.0, -2.5], [0.0, 2.0, 0.0], [1.5, 0.0, 1.5]] {
        let r = len(p);
        let cos = p[2] / r;
        let (jr, yr) = spherical(nmax, k * r);
        let pl = legendre(nmax, cos);
        let mut want = Complex::ZERO;
        for n in 0..=nmax {
            let jp = d(&ja, n, k * a);
            let hp = Complex::new(jp, d(&ya, n, k * a));
            let hr = Complex::new(jr[n], yr[n]);
            let i_n = Complex::cis(PI / 2.0 * n as f64);
            want -= i_n * ((2 * n + 1) as f64 * pl[n]) * (Complex::from(jp) / hp) * hr;
        }
        let got = sol.pressure_at(p);
        assert!((got - want).abs() < 0.04 * want.abs().max(0.05), "{p:?}: {got:?} vs {want:?}");
    }
}

#[test]
fn radiated_power_matches_the_far_field_flux() {
    // An oscillating sphere (∂p/∂n = cos θ), in a fluid with ρ = c = 1:
    // power through the surface equals ∫|F|² dΩ / 2ρc at infinity.
    let k = 1.0;
    let s = icosphere(1.0, 7);
    let sol = Helmholtz::new(&s, k).unwrap().radiate(|_, n| Complex::from(n[2])).unwrap();
    let w_surface = sol.sound_power(1.0, 1.0);
    let m = 400;
    let golden = PI * (3.0 - 5f64.sqrt());
    let mut w_far = 0.0;
    for i in 0..m {
        let z = 1.0 - (i as f64 + 0.5) * 2.0 / m as f64;
        let rr = (1.0 - z * z).sqrt();
        let phi = golden * i as f64;
        let f = sol.far_field([rr * phi.cos(), rr * phi.sin(), z]);
        w_far += f.norm_sqr() * 4.0 * PI / m as f64;
    }
    w_far /= 2.0;
    assert!(w_surface > 0.0);
    assert!((w_surface / w_far - 1.0).abs() < 0.03, "{w_surface} vs {w_far}");
}
