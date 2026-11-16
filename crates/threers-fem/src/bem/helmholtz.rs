//! Exterior acoustics: the Helmholtz equation `∇²p + k²p = 0` outside a
//! closed surface, with the Sommerfeld condition that waves only leave.
//!
//! Time dependence is `e^{-iωt}`, so the free-space Green's function is
//! `G = e^{ikr} / 4πr`, and for a surface vibrating with normal velocity
//! `v_n` the pressure gradient is `∂p/∂n = iωρ v_n`.
//!
//! # Irregular frequencies, and CHIEF
//!
//! The exterior equation `(½I − H) p = −G q` breaks down at the resonant
//! frequencies of the *interior* — not because anything physical happens
//! there, but because the integral operator acquires a null space. Near them
//! the answer is garbage. For a sphere of radius `a` the first is at
//! `ka = π`.
//!
//! The cure here is CHIEF (Schenck, 1968): the same representation formula,
//! evaluated at points inside the body, must give zero; adding those rows
//! and solving in the least-squares sense removes the null space as long as
//! no point sits on a node of the offending interior mode. A dozen points
//! scattered through the body are chosen automatically; set
//! [`chief_points`](Helmholtz::chief_points) to choose them, or to none.

use super::{integrate, integrate_far, integrate_self, interior_points, inv_r_self, rows, Elements};
use crate::complex::Complex;
use crate::dense::Matrix;
use crate::surface::TriSurface;
use crate::vec3::*;
use crate::FemError;
use std::f64::consts::PI;

/// An exterior acoustic problem at one frequency.
///
/// ```
/// use threers_fem::{bem::Helmholtz, Complex, TriSurface};
/// use threers::geometries::IcosahedronGeometry;
///
/// // A pulsating sphere: radius a = 1, uniform ∂p/∂n = q = 1, k = 1. The
/// // exact pressure on it is q·a / (ika − 1) = 1 / (i − 1).
/// let sphere = TriSurface::from_geometry(&IcosahedronGeometry::new(1.0, 3)).unwrap();
/// let s = Helmholtz::new(&sphere, 1.0)
///     .unwrap()
///     .radiate(|_, _| Complex::ONE)
///     .unwrap();
/// let want = Complex::ONE / Complex::new(-1.0, 1.0);
/// assert!((s.pressure[0] - want).abs() < 0.03 * want.abs());
/// ```
pub struct Helmholtz {
    e: Elements,
    k: f64,
    chief: Option<Vec<V3>>,
}

impl Helmholtz {
    /// Wavenumber `k = ω / c = 2π / λ`.
    pub fn new(surface: &TriSurface, wavenumber: f64) -> Result<Self, FemError> {
        if wavenumber.is_nan() || wavenumber <= 0.0 {
            return Err(FemError::InvalidInput("wavenumber must be positive"));
        }
        Ok(Self {
            e: Elements::new(surface)?,
            k: wavenumber,
            chief: None,
        })
    }

    /// At `frequency` (Hz) in a medium with sound speed `c` — 343 m/s in
    /// air at 20 °C, 1481 m/s in water.
    pub fn acoustic(surface: &TriSurface, frequency: f64, c: f64) -> Result<Self, FemError> {
        Self::new(surface, 2.0 * PI * frequency / c)
    }

    pub fn wavenumber(&self) -> f64 {
        self.k
    }

    /// CHIEF points inside the body. Empty turns CHIEF off; the default
    /// picks twelve.
    pub fn chief_points(mut self, points: Vec<V3>) -> Self {
        self.chief = Some(points);
        self
    }

    pub fn len(&self) -> usize {
        self.e.len()
    }

    pub fn is_empty(&self) -> bool {
        self.e.len() == 0
    }

    pub fn centroids(&self) -> &[V3] {
        &self.e.centroids
    }

    pub fn normals(&self) -> &[V3] {
        &self.e.normals
    }

    pub fn areas(&self) -> &[f64] {
        &self.e.areas
    }

    /// Elements per wavelength, by the largest triangle — six is the least
    /// for a usable answer, ten for a good one.
    pub fn elements_per_wavelength(&self) -> f64 {
        let lambda = 2.0 * PI / self.k;
        lambda / self.e.sizes.iter().copied().fold(0.0, f64::max)
    }

    /// Radiation: `dpdn(centroid, normal)` gives `∂p/∂n` on each triangle.
    pub fn radiate(&self, dpdn: impl Fn(V3, V3) -> Complex) -> Result<HelmholtzSolution, FemError> {
        let q: Vec<Complex> = (0..self.len())
            .map(|i| dpdn(self.e.centroids[i], self.e.normals[i]))
            .collect();
        self.solve_neumann(q, None)
    }

    /// Radiation from a surface vibrating with normal velocity
    /// `velocity(centroid, normal)` (outward positive, complex amplitude) in a
    /// fluid of `density`: `∂p/∂n = iωρ v_n` with `ω = k c`.
    pub fn radiate_velocity(
        &self,
        density: f64,
        sound_speed: f64,
        velocity: impl Fn(V3, V3) -> Complex,
    ) -> Result<HelmholtzSolution, FemError> {
        let omega = self.k * sound_speed;
        let f = Complex::new(0.0, omega * density);
        self.radiate(|c, n| f * velocity(c, n))
    }

    /// Scattering of the plane wave `amplitude · e^{ik d·x}` travelling along
    /// unit `direction` by the body as a rigid, motionless obstacle. The
    /// solution holds the *scattered* field; [`HelmholtzSolution::total_at`]
    /// adds the incident one back.
    pub fn scatter_plane_wave(&self, direction: V3, amplitude: Complex) -> Result<HelmholtzSolution, FemError> {
        let d = normalize(direction);
        let k = self.k;
        let incident = move |x: V3| amplitude * Complex::cis(k * dot(d, x));
        // Rigid: ∂(p_inc + p_s)/∂n = 0.
        let q: Vec<Complex> = (0..self.len())
            .map(|i| {
                let x = self.e.centroids[i];
                -(incident(x) * Complex::new(0.0, k * dot(d, self.e.normals[i])))
            })
            .collect();
        self.solve_neumann(q, Some((d, amplitude)))
    }

    fn solve_neumann(&self, q: Vec<Complex>, incident: Option<(V3, Complex)>) -> Result<HelmholtzSolution, FemError> {
        let n = self.len();
        let k = self.k;
        let e = &self.e;
        // Rows of ½I − H and of G, at the centroids.
        let row_data = rows(n, |i| {
            let x = e.centroids[i];
            let mut a = vec![Complex::ZERO; n];
            let mut g = vec![Complex::ZERO; n];
            let mut h0_sum = 0.0;
            for j in 0..n {
                let nj = e.normals[j];
                if i == j {
                    // G: the 1/r part analytically, the rest is bounded.
                    let smooth: Complex = integrate_self(e.tris[j], &|y: V3| {
                        let r = dist(x, y);
                        (Complex::cis(k * r) - Complex::ONE) * (1.0 / (4.0 * PI * r))
                    });
                    g[j] = smooth + Complex::from(inv_r_self(x, e.tris[j]) / (4.0 * PI));
                    continue;
                }
                let (gk, hk, h0) = kernels(x, e.tris[j], nj, k);
                g[j] = gk;
                // ½I − H_k = (I − (½I + H₀)) − (H_k − H₀); the static part's
                // diagonal comes from its row sum, as in Laplace.
                a[j] = -hk;
                h0_sum += h0;
            }
            // (½I + H₀)_ii = −Σ_{j≠i} H₀_ij, so (I − (½I + H₀))_ii = 1 + Σ.
            a[i] = Complex::from(1.0 + h0_sum);
            (a, g)
        });
        let chief = match &self.chief {
            Some(p) => p.clone(),
            None => interior_points(e, 12),
        };
        let chief_rows = rows(chief.len(), |c| {
            let x = chief[c];
            let mut a = vec![Complex::ZERO; n];
            let mut g = vec![Complex::ZERO; n];
            for j in 0..n {
                let (gk, hk, _) = kernels(x, e.tris[j], e.normals[j], k);
                // 0 = H p − G q at an interior point.
                a[j] = hk;
                g[j] = gk;
            }
            (a, g)
        });
        let m_rows = n + chief_rows.len();
        let mut m = Matrix::<Complex>::zeros(m_rows, n);
        let mut rhs = vec![Complex::ZERO; m_rows];
        for (i, (a, g)) in row_data.iter().enumerate() {
            m.row_mut(i).copy_from_slice(a);
            // (½I − H) p = −G q.
            rhs[i] = -g.iter().zip(&q).map(|(g, q)| *g * *q).sum::<Complex>();
        }
        for (c, (a, g)) in chief_rows.iter().enumerate() {
            m.row_mut(n + c).copy_from_slice(a);
            rhs[n + c] = g.iter().zip(&q).map(|(g, q)| *g * *q).sum::<Complex>();
        }
        let p = if m_rows > n { m.least_squares(&rhs)? } else { m.solve(&rhs)? };
        Ok(HelmholtzSolution {
            e: e.clone(),
            k,
            pressure: p,
            dpdn: q,
            incident,
            chief_points: chief,
        })
    }
}

/// `∫ G_k`, `∫ ∂G_k/∂n_y` and the static `∫ ∂G₀/∂n_y` over one triangle.
fn kernels(x: V3, tri: [V3; 3], n: V3, k: f64) -> (Complex, Complex, f64) {
    let t: Triple = integrate(x, tri, &|y: V3| {
        let d = sub(y, x);
        let r2 = norm2(d);
        let r = r2.sqrt();
        let e = Complex::cis(k * r);
        let g = e * (1.0 / (4.0 * PI * r));
        let rn = dot(d, n) / r;
        // dG/dr = e^{ikr}(ikr − 1) / 4πr².
        let h = e * Complex::new(-1.0, k * r) * (rn / (4.0 * PI * r2));
        let h0 = -rn / (4.0 * PI * r2);
        Triple(g, h, h0)
    });
    (t.0, t.1, t.2)
}

#[derive(Clone, Copy, Default)]
struct Triple(Complex, Complex, f64);

impl std::ops::Add for Triple {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Triple(self.0 + o.0, self.1 + o.1, self.2 + o.2)
    }
}

impl std::ops::Mul<f64> for Triple {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Triple(self.0 * s, self.1 * s, self.2 * s)
    }
}

/// Surface pressure and the field anywhere outside.
#[derive(Clone, Debug)]
pub struct HelmholtzSolution {
    e: Elements,
    k: f64,
    /// Pressure on each triangle — the scattered part, for a scattering
    /// problem.
    pub pressure: Vec<Complex>,
    /// `∂p/∂n` on each triangle, as given or as the rigid condition implies.
    pub dpdn: Vec<Complex>,
    incident: Option<(V3, Complex)>,
    /// The CHIEF points used.
    pub chief_points: Vec<V3>,
}

impl HelmholtzSolution {
    /// The radiated or scattered pressure at a point outside the body.
    pub fn pressure_at(&self, x: V3) -> Complex {
        let mut s = Complex::ZERO;
        for j in 0..self.e.len() {
            let (g, h, _) = kernels(x, self.e.tris[j], self.e.normals[j], self.k);
            s += h * self.pressure[j] - g * self.dpdn[j];
        }
        s
    }

    /// [`pressure_at`](Self::pressure_at) for many points, across threads
    /// with `parallel`.
    pub fn pressure_at_points(&self, points: &[V3]) -> Vec<Complex> {
        rows(points.len(), |i| self.pressure_at(points[i]))
    }

    /// Incident plus scattered pressure at a point outside; for a radiation
    /// problem, the same as [`pressure_at`](Self::pressure_at).
    pub fn total_at(&self, x: V3) -> Complex {
        let p = self.pressure_at(x);
        match self.incident {
            Some((d, a)) => p + a * Complex::cis(self.k * dot(d, x)),
            None => p,
        }
    }

    /// Far-field pattern `F(d̂)`, with `p ≈ F · e^{ikr} / r` far away in
    /// direction `d̂`: the representation formula with the Green's function
    /// replaced by its far-field limit.
    pub fn far_field(&self, direction: V3) -> Complex {
        let d = normalize(direction);
        let k = self.k;
        let mut s = Complex::ZERO;
        for j in 0..self.e.len() {
            let nj = self.e.normals[j];
            let t: Triple = integrate_far(self.e.tris[j], &|y: V3| {
                let ph = Complex::cis(-k * dot(d, y)) * (1.0 / (4.0 * PI));
                // ∂/∂n_y of e^{-ik d·y} = −ik (d·n) e^{-ik d·y}.
                Triple(ph, ph * Complex::new(0.0, -k * dot(d, nj)), 0.0)
            });
            s += t.1 * self.pressure[j] - t.0 * self.dpdn[j];
        }
        s
    }

    /// Radiated sound power, `½ Re ∫ p v̄_n dS` with `v_n = ∂p/∂n / iωρ`,
    /// in a fluid of `density` and `sound_speed`.
    pub fn sound_power(&self, density: f64, sound_speed: f64) -> f64 {
        let omega = self.k * sound_speed;
        let mut s = 0.0;
        for j in 0..self.e.len() {
            let v = self.dpdn[j] / Complex::new(0.0, omega * density);
            s += 0.5 * (self.pressure[j] * v.conj()).re * self.e.areas[j];
        }
        s
    }
}

/// Sound pressure level in dB re 20 µPa of a complex pressure amplitude (in
/// pascals; the RMS is `|p|/√2`).
pub fn spl(p: Complex) -> f64 {
    20.0 * (p.abs() / std::f64::consts::SQRT_2 / 20e-6).log10()
}
