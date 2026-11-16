//! Laplace's equation by constant-element collocation.
//!
//! With `G(x, y) = 1 / 4π|x − y|`, `H_ij = ∫_j ∂G/∂n_y` and `G_ij = ∫_j G`,
//! the boundary integral equations at the centroids are
//!
//! ```text
//! inside:   (½I + H) u = G q
//! outside:  (½I − H) u = −G q
//! ```
//!
//! for potential `u` and flux `q = ∂u/∂n` along the outward normal. `H`'s
//! diagonal vanishes on a flat triangle; the `½I + H` diagonal is then set
//! from the rows instead, using the fact that a constant potential has no
//! flux (`(½I + H)·1 = 0`). That absorbs the quadrature error of the
//! near-singular neighbours into the diagonal and roughly halves the error
//! on a coarse mesh.

use super::{integrate, inv_r_self, rows, Elements};
use crate::dense::{Lu, Matrix};
use crate::surface::TriSurface;
use crate::vec3::*;
use crate::FemError;
use std::f64::consts::PI;

/// What is known on one triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Potential {
    /// The potential (a Dirichlet condition): voltage on a conductor,
    /// temperature on a held face, velocity potential on an inlet.
    Value(f64),
    /// The outward normal derivative (a Neumann condition): flux out of an
    /// interior region, zero on an insulated wall or an impermeable body.
    Flux(f64),
}

/// A potential problem inside or outside a closed surface.
///
/// ```
/// use threers_fem::{bem::Laplace, TriSurface};
/// use threers::geometries::IcosahedronGeometry;
///
/// // The capacitance of a unit sphere is 4πε₀R; in units of ε₀, 4π.
/// let sphere = TriSurface::from_geometry(&IcosahedronGeometry::new(1.0, 3)).unwrap();
/// let c = Laplace::exterior(&sphere).unwrap().capacitance().unwrap();
/// assert!((c / (4.0 * std::f64::consts::PI) - 1.0).abs() < 0.02);
/// ```
pub struct Laplace {
    e: Elements,
    exterior: bool,
    /// `A` and `B` of `A u = B q`.
    a: Matrix<f64>,
    b: Matrix<f64>,
}

impl Laplace {
    /// The region inside `surface`.
    pub fn interior(surface: &TriSurface) -> Result<Self, FemError> {
        Self::new(surface, false)
    }

    /// The unbounded region outside `surface`, with the potential decaying
    /// to zero at infinity.
    pub fn exterior(surface: &TriSurface) -> Result<Self, FemError> {
        Self::new(surface, true)
    }

    fn new(surface: &TriSurface, exterior: bool) -> Result<Self, FemError> {
        let e = Elements::new(surface)?;
        let n = e.len();
        let row_data = rows(n, |i| {
            let x = e.centroids[i];
            let mut h = vec![0.0; n];
            let mut g = vec![0.0; n];
            for j in 0..n {
                if i == j {
                    g[j] = inv_r_self(x, e.tris[j]) / (4.0 * PI);
                    continue;
                }
                let nj = e.normals[j];
                let (gv, hv): (f64, f64) = integrate(x, e.tris[j], &|y: V3| {
                    let d = sub(y, x);
                    let r2 = norm2(d);
                    let r = r2.sqrt();
                    Pair(1.0 / (4.0 * PI * r), -dot(d, nj) / (4.0 * PI * r2 * r))
                })
                .into();
                g[j] = gv;
                h[j] = hv;
            }
            // (½I + H)·1 = 0 fixes the diagonal.
            h[i] = -h.iter().sum::<f64>();
            (h, g)
        });
        let mut a = Matrix::zeros(n, n);
        let mut b = Matrix::zeros(n, n);
        for (i, (h, g)) in row_data.into_iter().enumerate() {
            for j in 0..n {
                if exterior {
                    // ½I − H = I − (½I + H).
                    a[(i, j)] = if i == j { 1.0 } else { 0.0 } - h[j];
                    b[(i, j)] = -g[j];
                } else {
                    a[(i, j)] = h[j];
                    b[(i, j)] = g[j];
                }
            }
        }
        Ok(Self { e, exterior, a, b })
    }

    /// Number of triangles, and so of unknowns.
    pub fn len(&self) -> usize {
        self.e.len()
    }

    pub fn is_empty(&self) -> bool {
        self.e.len() == 0
    }

    /// Triangle centroids — where `condition` is asked, and where the
    /// solution's values live.
    pub fn centroids(&self) -> &[V3] {
        &self.e.centroids
    }

    /// Outward unit normal per triangle.
    pub fn normals(&self) -> &[V3] {
        &self.e.normals
    }

    pub fn areas(&self) -> &[f64] {
        &self.e.areas
    }

    /// Solve with `condition(centroid, normal)` giving what is known on each
    /// triangle.
    pub fn solve(&self, condition: impl Fn(V3, V3) -> Potential) -> Result<LaplaceSolution, FemError> {
        let n = self.len();
        let known: Vec<Potential> = (0..n)
            .map(|i| condition(self.e.centroids[i], self.e.normals[i]))
            .collect();
        if !self.exterior && known.iter().all(|k| matches!(k, Potential::Flux(_))) {
            return Err(FemError::InvalidInput(
                "an interior problem with flux given everywhere fixes the potential only up to a constant: give a value somewhere",
            ));
        }
        // Column j holds the unknown of triangle j: A u − B q = 0.
        let mut m = Matrix::zeros(n, n);
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                match known[j] {
                    Potential::Value(u) => {
                        m[(i, j)] = -self.b[(i, j)];
                        rhs[i] -= self.a[(i, j)] * u;
                    }
                    Potential::Flux(q) => {
                        m[(i, j)] = self.a[(i, j)];
                        rhs[i] += self.b[(i, j)] * q;
                    }
                }
            }
        }
        let x = Lu::factor(m)?.solve(&rhs)?;
        let mut u = vec![0.0; n];
        let mut q = vec![0.0; n];
        for j in 0..n {
            match known[j] {
                Potential::Value(v) => {
                    u[j] = v;
                    q[j] = x[j];
                }
                Potential::Flux(v) => {
                    q[j] = v;
                    u[j] = x[j];
                }
            }
        }
        Ok(LaplaceSolution {
            e: self.e.clone(),
            exterior: self.exterior,
            potential: u,
            flux: q,
        })
    }

    /// Capacitance of the surface as a conductor in free space, in units of
    /// the permittivity: hold it at potential 1 and total the charge,
    /// `−∫ ∂u/∂n`. Multiply by ε₀ = 8.854e-12 F/m for farads with lengths in
    /// metres. Needs an [`exterior`](Self::exterior) problem.
    pub fn capacitance(&self) -> Result<f64, FemError> {
        if !self.exterior {
            return Err(FemError::InvalidInput("capacitance needs an exterior problem"));
        }
        let s = self.solve(|_, _| Potential::Value(1.0))?;
        Ok(-s.total_flux())
    }
}

/// Potential and flux on every triangle, and the field anywhere else.
#[derive(Clone, Debug)]
pub struct LaplaceSolution {
    e: Elements,
    exterior: bool,
    /// Potential at each triangle.
    pub potential: Vec<f64>,
    /// Normal derivative along the outward normal at each triangle.
    pub flux: Vec<f64>,
}

impl LaplaceSolution {
    /// `∫ ∂u/∂n dS` over the whole surface.
    pub fn total_flux(&self) -> f64 {
        self.flux.iter().zip(&self.e.areas).map(|(q, a)| q * a).sum()
    }

    /// Potential at a point off the surface, by the representation formula.
    /// Inside the body for an interior problem, outside it for an exterior
    /// one; on the wrong side it is zero, to quadrature accuracy.
    pub fn potential_at(&self, x: V3) -> f64 {
        let mut s = 0.0;
        for j in 0..self.e.len() {
            let nj = self.e.normals[j];
            let (g, h): (f64, f64) = integrate(x, self.e.tris[j], &|y: V3| {
                let d = sub(y, x);
                let r2 = norm2(d);
                let r = r2.sqrt();
                Pair(1.0 / (4.0 * PI * r), -dot(d, nj) / (4.0 * PI * r2 * r))
            })
            .into();
            s += g * self.flux[j] - h * self.potential[j];
        }
        if self.exterior { -s } else { s }
    }

    /// [`potential_at`](Self::potential_at) for many points, across threads
    /// with `parallel`.
    pub fn potential_at_points(&self, points: &[V3]) -> Vec<f64> {
        rows(points.len(), |i| self.potential_at(points[i]))
    }

    /// The field `−∇u` at a point, by central differences of the potential
    /// at a step of `h` — a small fraction of the distance to the surface.
    pub fn field_at(&self, x: V3, h: f64) -> V3 {
        [0, 1, 2].map(|k| {
            let mut a = x;
            let mut b = x;
            a[k] += h;
            b[k] -= h;
            -(self.potential_at(a) - self.potential_at(b)) / (2.0 * h)
        })
    }
}

/// Two reals integrated together, so one pass of quadrature serves both
/// kernels.
#[derive(Clone, Copy, Default)]
pub(crate) struct Pair(pub f64, pub f64);

impl std::ops::Add for Pair {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Pair(self.0 + o.0, self.1 + o.1)
    }
}

impl std::ops::Mul<f64> for Pair {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Pair(self.0 * s, self.1 * s)
    }
}

impl From<Pair> for (f64, f64) {
    fn from(p: Pair) -> Self {
        (p.0, p.1)
    }
}
