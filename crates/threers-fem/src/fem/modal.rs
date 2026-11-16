//! Natural frequencies and mode shapes: `K φ = ω² M φ`.
//!
//! Shift-and-invert subspace iteration. A block of vectors is repeatedly
//! multiplied by `(K + σM)⁻¹ M` — one factorisation, then a pair of
//! triangular solves per vector — and re-diagonalised by Rayleigh–Ritz each
//! time, so the lowest modes emerge first and together. The shift `σ` is a
//! millionth of the mesh's highest frequency squared; it makes `K + σM`
//! positive definite even with no supports, which is how a free part's six
//! rigid modes come out at zero instead of breaking the factorisation.
//!
//! The mass matrix is the consistent one, so frequencies converge from above
//! as the mesh is refined.

use super::elastic::Elastic;
use super::per_chunk;
use crate::dense::{cholesky, symmetric_eigen, Matrix};
use crate::sparse::{Csr, SpdFactor, Triplets};
use crate::vec3::*;
use crate::FemError;

/// One mode of vibration.
#[derive(Clone, Debug)]
pub struct Mode {
    /// Frequency in cycles per unit time — hertz, in SI.
    pub frequency: f64,
    /// Angular frequency `ω = 2πf`.
    pub omega: f64,
    /// Shape: displacement of each mesh node, scaled so the largest has
    /// length 1. Its sign is arbitrary.
    pub shape: Vec<V3>,
}

/// The lowest modes of a part, ascending.
#[derive(Clone, Debug)]
pub struct Modes {
    /// The mesh nodes, undeformed.
    pub nodes: Vec<V3>,
    pub modes: Vec<Mode>,
    /// Subspace iterations taken.
    pub iterations: usize,
}

impl Modes {
    pub fn frequencies(&self) -> Vec<f64> {
        self.modes.iter().map(|m| m.frequency).collect()
    }
}

pub(crate) fn solve(problem: &Elastic, count: usize) -> Result<Modes, FemError> {
    if count == 0 {
        return Err(FemError::InvalidInput("asked for zero modes"));
    }
    let density = problem.material().density;
    if density.is_nan() || density <= 0.0 {
        return Err(FemError::InvalidInput("modal analysis needs a positive density"));
    }
    let sys = problem.assemble()?;
    let space = &sys.space;
    let order = problem.order_();
    let r = order.reference();
    let npe = order.nodes();
    let ndof = 3 * space.nodes.len();

    let parts = per_chunk(space.elem_count(), |range| {
        let mut t = Triplets::with_capacity(ndof, range.len() * npe * npe * 3);
        for e in range {
            let nodes = space.elem(e);
            let w = density * space.volumes[e];
            for a in 0..npe {
                for b in 0..npe {
                    let m = w * r.mass[a * npe + b];
                    for i in 0..3 {
                        t.add(3 * nodes[a] as usize + i, 3 * nodes[b] as usize + i, m);
                    }
                }
            }
        }
        t
    });
    let mut trip = Triplets::new(ndof);
    for p in parts {
        trip.extend(p);
    }
    let m_full = trip.to_csr();

    let free: Vec<bool> = sys.fixed.iter().map(|f| f.is_none()).collect();
    let (k, map) = sys.k.submatrix(&free);
    let (m, _) = m_full.submatrix(&free);
    let n = k.n;
    if n == 0 {
        return Err(FemError::InvalidInput("every degree of freedom is fixed"));
    }
    let count = count.min(n);
    let p = (2 * count).max(count + 8).min(n);

    // Shift: a millionth of the largest diagonal ratio, a stand-in for the
    // highest frequency squared.
    let kd = k.diagonal();
    let md = m.diagonal();
    let top = kd
        .iter()
        .zip(&md)
        .filter(|(_, m)| **m > 0.0)
        .map(|(k, m)| k / m)
        .fold(0.0, f64::max);
    let sigma = 1e-6 * top;
    let a = add_scaled(&k, &m, sigma);
    let all_coords = space.dof_coords(3);
    let coords: Vec<V3> = (0..ndof).filter(|&i| free[i]).map(|i| all_coords[i]).collect();
    // Every iteration solves `p` systems, so a direct factor pays off at a
    // larger size than for one static solve.
    let factor = SpdFactor::with_coords(a, &coords, problem.solver_(), 30.0 * p as f64)?;
    // For an iterative factor, the previous Ritz vector over its Ritz value
    // is nearly the next solve's answer once the block settles.
    let mut guesses: Option<Vec<Vec<f64>>> = None;

    // Start from smooth-ish deterministic vectors: the diagonal ratio
    // weights low-stiffness freedoms, plus a hash for independence.
    let mut x: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let h = (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((j as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    let h = (h ^ (h >> 31)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    ((h >> 11) as f64 / (1u64 << 53) as f64 - 0.5) + if j == 0 { 1.0 } else { 0.0 }
                })
                .collect()
        })
        .collect();

    let mut previous = vec![f64::INFINITY; count];
    let mut values = Vec::new();
    let mut iterations = 0;
    for it in 0..300 {
        iterations = it + 1;
        // Y = A⁻¹ M X.
        let mx: Vec<Vec<f64>> = x.iter().map(|v| m.mul_vec(v)).collect();
        let y: Vec<Vec<f64>> = mx
            .iter()
            .enumerate()
            .map(|(j, b)| factor.solve_from(b, guesses.as_ref().map(|g| g[j].as_slice())))
            .collect::<Result<_, _>>()?;
        // Rayleigh–Ritz: (Yᵀ A Y) q = μ (Yᵀ M Y) q, with A Y = M X.
        let my: Vec<Vec<f64>> = y.iter().map(|v| m.mul_vec(v)).collect();
        let mut ka = Matrix::<f64>::zeros(p, p);
        let mut ma = Matrix::<f64>::zeros(p, p);
        for i in 0..p {
            for j in i..p {
                let kij = dot_n(&y[i], &mx[j]);
                let kji = dot_n(&y[j], &mx[i]);
                ka[(i, j)] = 0.5 * (kij + kji);
                ka[(j, i)] = ka[(i, j)];
                ma[(i, j)] = dot_n(&y[i], &my[j]);
                ma[(j, i)] = ma[(i, j)];
            }
        }
        let (mu, q) = generalized_eigen(&ka, &ma)?;
        x = (0..p)
            .map(|j| {
                let mut v = vec![0.0; n];
                for i in 0..p {
                    let c = q[(i, j)];
                    if c != 0.0 {
                        for (vk, yk) in v.iter_mut().zip(&y[i]) {
                            *vk += c * yk;
                        }
                    }
                }
                v
            })
            .collect();
        if !factor.is_direct() {
            guesses = Some(
                x.iter()
                    .zip(&mu)
                    .map(|(v, m)| v.iter().map(|e| e / m.max(1e-300)).collect())
                    .collect(),
            );
        }
        values = mu;
        let converged = (0..count).all(|i| {
            let scale = values[i].abs().max(sigma);
            (values[i] - previous[i]).abs() <= 1e-10 * scale
        });
        previous = values[..count].to_vec();
        if converged {
            break;
        }
    }
    if iterations == 300 {
        return Err(FemError::NoConvergence { iterations });
    }

    let nodes = problem.mesh().nodes.clone();
    let modes = (0..count)
        .map(|j| {
            let lambda = (values[j] - sigma).max(0.0);
            let omega = lambda.sqrt();
            let mut shape = vec![[0.0; 3]; space.corner_count];
            for (node, s) in shape.iter_mut().enumerate() {
                for i in 0..3 {
                    let g = map[3 * node + i];
                    if g != u32::MAX {
                        s[i] = x[j][g as usize];
                    }
                }
            }
            let biggest = shape.iter().map(|d| norm(*d)).fold(0.0, f64::max);
            if biggest > 0.0 {
                for s in &mut shape {
                    *s = scale(*s, 1.0 / biggest);
                }
            }
            Mode {
                frequency: omega / (2.0 * std::f64::consts::PI),
                omega,
                shape,
            }
        })
        .collect();
    Ok(Modes {
        nodes,
        modes,
        iterations,
    })
}

fn dot_n(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a + s·b` for two matrices on the same pattern family.
fn add_scaled(a: &Csr, b: &Csr, s: f64) -> Csr {
    let mut t = Triplets::with_capacity(a.n, a.nnz() + b.nnz());
    for i in 0..a.n {
        let (c, v) = a.row(i);
        for (c, v) in c.iter().zip(v) {
            t.add(i, *c as usize, *v);
        }
        let (c, v) = b.row(i);
        for (c, v) in c.iter().zip(v) {
            t.add(i, *c as usize, s * v);
        }
    }
    t.to_csr()
}

/// `A q = μ B q` for small dense symmetric `A` and positive definite `B`,
/// ascending, with `B`-orthonormal vectors.
fn generalized_eigen(a: &Matrix<f64>, b: &Matrix<f64>) -> Result<(Vec<f64>, Matrix<f64>), FemError> {
    let n = a.rows;
    let l = cholesky(b).ok_or(FemError::NotPositiveDefinite)?;
    // C = L⁻¹ A L⁻ᵀ.
    let linv = lower_inverse(&l);
    let mut tmp = Matrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..=i {
                s += linv[(i, k)] * a[(k, j)];
            }
            tmp[(i, j)] = s;
        }
    }
    let mut c = Matrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..=j {
                s += tmp[(i, k)] * linv[(j, k)];
            }
            c[(i, j)] = s;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let m = 0.5 * (c[(i, j)] + c[(j, i)]);
            c[(i, j)] = m;
            c[(j, i)] = m;
        }
    }
    let (w, z) = symmetric_eigen(&c);
    // q = L⁻ᵀ z. The smallest μ, λ + σ, are the lowest modes.
    let mut q = Matrix::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let mut s = 0.0;
            for k in i..n {
                s += linv[(k, i)] * z[(k, j)];
            }
            q[(i, j)] = s;
        }
    }
    Ok((w, q))
}

fn lower_inverse(l: &Matrix<f64>) -> Matrix<f64> {
    let n = l.rows;
    let mut inv = Matrix::<f64>::zeros(n, n);
    for j in 0..n {
        inv[(j, j)] = 1.0 / l[(j, j)];
        for i in j + 1..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[(i, k)] * inv[(k, j)];
            }
            inv[(i, j)] = -s / l[(i, i)];
        }
    }
    inv
}
