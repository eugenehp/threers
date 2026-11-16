//! Dense linear algebra: what the boundary element method solves with, and
//! the small Rayleigh–Ritz problems inside the modal solver.
//!
//! Row-major storage, generic over [`Scalar`] so one LU and one QR serve both
//! the real Laplace and the complex Helmholtz systems.

// Matrix code: the index is the meaning.
#![allow(clippy::needless_range_loop)]

use crate::complex::Complex;
use crate::FemError;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A field the dense solvers work over: `f64` or [`Complex`].
pub trait Scalar:
    Copy
    + Default
    + PartialEq
    + std::fmt::Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn conj(self) -> Self;
    fn abs(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn scale(self, s: f64) -> Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn conj(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn scale(self, s: f64) -> Self {
        self * s
    }
}

impl Scalar for Complex {
    const ZERO: Self = Complex::ZERO;
    const ONE: Self = Complex::ONE;
    fn conj(self) -> Self {
        Complex::conj(self)
    }
    fn abs(self) -> f64 {
        Complex::abs(self)
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v, 0.0)
    }
    fn scale(self, s: f64) -> Self {
        Complex::scale(self, s)
    }
}

/// A dense row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::ZERO; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = T::ONE;
        }
        m
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// `A·x`.
    pub fn mul_vec(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                let mut s = T::ZERO;
                for (a, b) in self.row(i).iter().zip(x) {
                    s += *a * *b;
                }
                s
            })
            .collect()
    }

    /// Solve `A·x = b` by LU with partial pivoting. `A` must be square.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, FemError> {
        Lu::factor(self.clone())?.solve(b)
    }

    /// The `x` minimising `‖A·x − b‖₂`, by Householder QR. `A` must have at
    /// least as many rows as columns and full column rank.
    pub fn least_squares(&self, b: &[T]) -> Result<Vec<T>, FemError> {
        assert_eq!(b.len(), self.rows);
        assert!(self.rows >= self.cols, "least squares needs rows ≥ cols");
        let (m, n) = (self.rows, self.cols);
        let mut a = self.clone();
        let mut rhs = b.to_vec();
        let scale = a.data.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for k in 0..n {
            // Householder vector for column k, rows k..m.
            let mut norm2 = 0.0;
            for i in k..m {
                norm2 += a[(i, k)].abs().powi(2);
            }
            let norm = norm2.sqrt();
            if norm <= 1e-14 * scale {
                return Err(FemError::Singular);
            }
            let akk = a[(k, k)];
            // alpha = -e^{i·arg(akk)}·norm, so v = x - alpha·e1 does not cancel.
            let phase = if akk.abs() > 0.0 {
                akk.scale(1.0 / akk.abs())
            } else {
                T::ONE
            };
            let alpha = -(phase.scale(norm));
            let mut v: Vec<T> = (k..m).map(|i| a[(i, k)]).collect();
            v[0] -= alpha;
            let vnorm2: f64 = v.iter().map(|x| x.abs().powi(2)).sum();
            if vnorm2 == 0.0 {
                continue;
            }
            // Apply H = I - 2 v v^H / (v^H v) to the remaining columns and b:
            // w = v^H A by sweeping rows, then A -= v (2w / v^H v). Rows are
            // contiguous here; a column-at-a-time sweep strides the whole
            // matrix per entry and was several times slower.
            let mut w = vec![T::ZERO; n - k];
            for (r, vi) in v.iter().enumerate() {
                let c = vi.conj();
                let row = &a.row(k + r)[k..];
                for (wj, x) in w.iter_mut().zip(row) {
                    *wj += c * *x;
                }
            }
            for wj in &mut w {
                *wj = wj.scale(2.0 / vnorm2);
            }
            for (r, vi) in v.iter().enumerate() {
                let row = &mut a.row_mut(k + r)[k..];
                for (x, wj) in row.iter_mut().zip(&w) {
                    *x -= *vi * *wj;
                }
            }
            let mut s = T::ZERO;
            for (r, vi) in v.iter().enumerate() {
                s += vi.conj() * rhs[k + r];
            }
            let f = s.scale(2.0 / vnorm2);
            for (r, vi) in v.iter().enumerate() {
                rhs[k + r] -= *vi * f;
            }
        }
        // Back substitution on the upper triangle.
        let mut x = vec![T::ZERO; n];
        for i in (0..n).rev() {
            let mut s = rhs[i];
            for j in i + 1..n {
                s -= a[(i, j)] * x[j];
            }
            x[i] = s / a[(i, i)];
        }
        Ok(x)
    }
}

impl<T: Scalar> std::ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.cols + j]
    }
}

impl<T: Scalar> std::ops::IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.cols + j]
    }
}

/// An LU factorisation with partial pivoting, reusable across right-hand
/// sides — a BEM matrix is factored once and solved for each load.
#[derive(Clone, Debug)]
pub struct Lu<T: Scalar> {
    lu: Matrix<T>,
    perm: Vec<usize>,
}

impl<T: Scalar> Lu<T> {
    pub fn factor(mut a: Matrix<T>) -> Result<Self, FemError> {
        assert_eq!(a.rows, a.cols, "LU needs a square matrix");
        let n = a.rows;
        let mut perm: Vec<usize> = (0..n).collect();
        let scale = a.data.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for k in 0..n {
            let mut p = k;
            let mut best = a[(k, k)].abs();
            for i in k + 1..n {
                let v = a[(i, k)].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best <= 1e-14 * scale || best == 0.0 {
                return Err(FemError::Singular);
            }
            if p != k {
                perm.swap(p, k);
                for j in 0..n {
                    a.data.swap(k * n + j, p * n + j);
                }
            }
            let pivot = a[(k, k)];
            let (top, bottom) = a.data.split_at_mut((k + 1) * n);
            let row_k = &top[k * n..(k + 1) * n];
            for row_i in bottom.chunks_mut(n) {
                let f = row_i[k] / pivot;
                row_i[k] = f;
                if f == T::ZERO {
                    continue;
                }
                for j in k + 1..n {
                    let t = f * row_k[j];
                    row_i[j] -= t;
                }
            }
        }
        Ok(Self { lu: a, perm })
    }

    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, FemError> {
        let n = self.lu.rows;
        assert_eq!(b.len(), n);
        let mut x: Vec<T> = self.perm.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let row = self.lu.row(i);
            let mut s = x[i];
            for j in 0..i {
                s -= row[j] * x[j];
            }
            x[i] = s;
        }
        for i in (0..n).rev() {
            let row = self.lu.row(i);
            let mut s = x[i];
            for j in i + 1..n {
                s -= row[j] * x[j];
            }
            x[i] = s / row[i];
        }
        Ok(x)
    }
}

/// Eigenvalues and eigenvectors of a real symmetric matrix by cyclic Jacobi
/// rotations, ascending. Columns of the returned matrix are the vectors.
///
/// For the small Rayleigh–Ritz problems of [`modal`](crate::fem::modal): a
/// few dozen rows, where Jacobi is accurate to the last bits and simple.
pub fn symmetric_eigen(a: &Matrix<f64>) -> (Vec<f64>, Matrix<f64>) {
    let n = a.rows;
    assert_eq!(n, a.cols);
    let mut a = a.clone();
    let mut v = Matrix::identity(n);
    for _sweep in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..n {
            diag += a[(i, i)] * a[(i, i)];
            for j in i + 1..n {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[(p, q)];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[(k, p)];
                    let akq = a[(k, q)];
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[(p, k)];
                    let aqk = a[(q, k)];
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].total_cmp(&a[(j, j)]));
    let values = order.iter().map(|&i| a[(i, i)]).collect();
    let mut vectors = Matrix::zeros(n, n);
    for (new, &old) in order.iter().enumerate() {
        for k in 0..n {
            vectors[(k, new)] = v[(k, old)];
        }
    }
    (values, vectors)
}

/// Cholesky factor `L` of a small symmetric positive definite matrix, or
/// `None` if it is not positive definite.
pub fn cholesky(a: &Matrix<f64>) -> Option<Matrix<f64>> {
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[(i, i)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_and_qr_agree_on_a_square_complex_system() {
        let n = 7;
        let mut a = Matrix::<Complex>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = Complex::new(
                    ((i * 7 + j * 3) % 11) as f64 - 5.0,
                    ((i * 5 + j * 2) % 13) as f64 * 0.3,
                );
            }
            a[(i, i)] += Complex::new(10.0, 0.0);
        }
        let b: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64, 1.0)).collect();
        let x1 = a.solve(&b).unwrap();
        let x2 = a.least_squares(&b).unwrap();
        for (p, q) in x1.iter().zip(&x2) {
            assert!((*p - *q).abs() < 1e-11, "{p:?} vs {q:?}");
        }
        let r = a.mul_vec(&x1);
        for (p, q) in r.iter().zip(&b) {
            assert!((*p - *q).abs() < 1e-11);
        }
    }

    #[test]
    fn jacobi_eigen() {
        let mut a = Matrix::<f64>::zeros(3, 3);
        let v = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        for i in 0..3 {
            for j in 0..3 {
                a[(i, j)] = v[i][j];
            }
        }
        let (w, _) = symmetric_eigen(&a);
        let s2 = 2f64.sqrt();
        for (got, want) in w.iter().zip([2.0 - s2, 2.0, 2.0 + s2]) {
            assert!((got - want).abs() < 1e-13);
        }
    }
}
