//! Sparse symmetric systems: what finite element assembly produces.
//!
//! Two ways to solve one, chosen by [`LinearSolver`]:
//!
//! - **Direct** — a nested dissection ordering, then a sparse Cholesky
//!   factorisation computed row by row (the up-looking method of Davis's
//!   CSparse). Exact to rounding and indifferent to conditioning, which
//!   matters for thin parts and for the modal solver, which solves against one
//!   factor dozens of times. Its cost is memory: the factor of a 3D mesh grows
//!   like `n^{4/3}`, so a chunky part of a few hundred thousand unknowns can
//!   outgrow it.
//! - **Iterative** — conjugate gradients with a Jacobi preconditioner. Memory
//!   is the matrix and four vectors, so it goes as large as you like, at the
//!   price of iterations that grow with the mesh and the element aspect ratio
//!   — thousands, on a thin-walled part of quadratic elements.
//!
//! [`LinearSolver::Auto`] takes the direct route when the factor would fit in
//! [`DIRECT_BUDGET`] entries and falls back to iterations when it would not.
//!
//! The ordering is George's automatic nested dissection: split the graph at
//! the middle level of a breadth-first search from a far vertex, number both
//! halves first and the separating level last, and recurse. It needs no
//! coordinates and on tetrahedral meshes keeps the factor within a small
//! multiple of the matrix.

// Matrix code: the index is the meaning.
#![allow(clippy::needless_range_loop)]

use crate::FemError;

/// Factor entries the automatic choice allows a direct solve — 64 M, about
/// 770 MB with their row indices. Past this the iterative solver is used.
pub const DIRECT_BUDGET: usize = 64 << 20;

/// Multiply-adds the automatic choice allows a direct factorisation —
/// around ten seconds on one core. Past this conjugate gradients usually
/// finish first, unless the system will be solved many times over.
pub const DIRECT_FLOPS: f64 = 1.5e10;

/// How a sparse symmetric positive definite system is solved.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LinearSolver {
    /// Direct when the factor fits [`DIRECT_BUDGET`], iterative otherwise.
    #[default]
    Auto,
    /// Sparse Cholesky after nested dissection, whatever it costs.
    Direct,
    /// Jacobi-preconditioned conjugate gradients to a relative residual of
    /// `tolerance`, giving up after `max_iterations`.
    Iterative {
        tolerance: f64,
        max_iterations: usize,
    },
}

impl LinearSolver {
    /// [`Iterative`](Self::Iterative) at a relative residual of 1e-10 and
    /// up to 20 000 iterations.
    pub const ITERATIVE: Self = Self::Iterative {
        tolerance: 1e-10,
        max_iterations: 20_000,
    };
}

/// Coordinate-format entries, summed on conversion: the assembly buffer.
#[derive(Clone, Debug, Default)]
pub struct Triplets {
    pub n: usize,
    entries: Vec<(u32, u32, f64)>,
}

impl Triplets {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(n: usize, cap: usize) -> Self {
        Self {
            n,
            entries: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.entries.push((i as u32, j as u32, v));
    }

    pub fn extend(&mut self, other: Triplets) {
        self.entries.extend(other.entries);
    }

    /// Sum duplicates into compressed rows.
    pub fn to_csr(mut self) -> Csr {
        self.entries
            .sort_unstable_by_key(|&(i, j, _)| ((i as u64) << 32) | j as u64);
        let mut row_ptr = vec![0usize; self.n + 1];
        let mut cols = Vec::with_capacity(self.entries.len() / 4);
        let mut vals: Vec<f64> = Vec::with_capacity(self.entries.len() / 4);
        let mut last = (u32::MAX, u32::MAX);
        for (i, j, v) in self.entries {
            if (i, j) == last {
                *vals.last_mut().unwrap() += v;
            } else {
                cols.push(j);
                vals.push(v);
                row_ptr[i as usize + 1] += 1;
                last = (i, j);
            }
        }
        for i in 0..self.n {
            row_ptr[i + 1] += row_ptr[i];
        }
        Csr {
            n: self.n,
            row_ptr,
            cols,
            vals,
        }
    }
}

/// A square matrix in compressed sparse rows. Both triangles are stored.
#[derive(Clone, Debug)]
pub struct Csr {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub cols: Vec<u32>,
    pub vals: Vec<f64>,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let r = self.row_ptr[i]..self.row_ptr[i + 1];
        (&self.cols[r.clone()], &self.vals[r])
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let (c, v) = self.row(i);
                c.iter()
                    .zip(v)
                    .find(|(c, _)| **c as usize == i)
                    .map_or(0.0, |(_, v)| *v)
            })
            .collect()
    }

    /// `y = A·x`.
    pub fn mul_vec_into(&self, x: &[f64], y: &mut [f64]) {
        let row = |i: usize| {
            let (c, v) = self.row(i);
            let mut s = 0.0;
            for (c, v) in c.iter().zip(v) {
                s += v * x[*c as usize];
            }
            s
        };
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            if self.n > 20_000 {
                y.par_iter_mut().enumerate().for_each(|(i, y)| *y = row(i));
                return;
            }
        }
        for (i, y) in y.iter_mut().enumerate() {
            *y = row(i);
        }
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        self.mul_vec_into(x, &mut y);
        y
    }

    /// The rows and columns where `keep[i]` holds, renumbered in order, and
    /// the map from old indices to new (`u32::MAX` where dropped).
    pub fn submatrix(&self, keep: &[bool]) -> (Csr, Vec<u32>) {
        let mut map = vec![u32::MAX; self.n];
        let mut m = 0u32;
        for (i, k) in keep.iter().enumerate() {
            if *k {
                map[i] = m;
                m += 1;
            }
        }
        let mut row_ptr = Vec::with_capacity(m as usize + 1);
        row_ptr.push(0);
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..self.n {
            if !keep[i] {
                continue;
            }
            let (c, v) = self.row(i);
            for (c, v) in c.iter().zip(v) {
                let nc = map[*c as usize];
                if nc != u32::MAX {
                    cols.push(nc);
                    vals.push(*v);
                }
            }
            row_ptr.push(cols.len());
        }
        (
            Csr {
                n: m as usize,
                row_ptr,
                cols,
                vals,
            },
            map,
        )
    }
}

/// Nested dissection ordering of a symmetric pattern: `perm[new] = old`.
pub fn nested_dissection(a: &Csr) -> Vec<usize> {
    struct Nd<'a> {
        a: &'a Csr,
        mark: Vec<u32>,
        seen: Vec<u32>,
        stamp: u32,
    }
    impl Nd<'_> {
        fn next(&mut self) -> u32 {
            self.stamp += 1;
            self.stamp
        }

        /// BFS from `start` over vertices marked `set`; returns the levels.
        fn bfs(&mut self, start: usize, set: u32) -> Vec<Vec<usize>> {
            let s = self.next();
            self.seen[start] = s;
            let mut levels = vec![vec![start]];
            loop {
                let mut next = Vec::new();
                for &u in levels.last().unwrap() {
                    let (c, _) = self.a.row(u);
                    for &v in c {
                        let v = v as usize;
                        if self.mark[v] == set && self.seen[v] != s {
                            self.seen[v] = s;
                            next.push(v);
                        }
                    }
                }
                if next.is_empty() {
                    return levels;
                }
                levels.push(next);
            }
        }

        fn degree(&self, v: usize) -> usize {
            self.a.row_ptr[v + 1] - self.a.row_ptr[v]
        }

        fn dissect(&mut self, set: &[usize], out: &mut [usize]) {
            debug_assert_eq!(set.len(), out.len());
            if set.len() <= 64 {
                out.copy_from_slice(set);
                return;
            }
            let id = self.next();
            for &v in set {
                self.mark[v] = id;
            }
            let mut levels = self.bfs(set[0], id);
            let reached: usize = levels.iter().map(Vec::len).sum();
            if reached < set.len() {
                // Disconnected: number each component on its own.
                let mut comps: Vec<Vec<usize>> = vec![levels.concat()];
                let s = self.seen[set[0]];
                let mut rest: Vec<usize> = set.iter().copied().filter(|&v| self.seen[v] != s).collect();
                while !rest.is_empty() {
                    let lv = self.bfs(rest[0], id);
                    let s = self.seen[rest[0]];
                    comps.push(lv.concat());
                    rest.retain(|&v| self.seen[v] != s);
                }
                let mut at = 0;
                for c in comps {
                    let len = c.len();
                    self.dissect(&c, &mut out[at..at + len]);
                    at += len;
                }
                return;
            }
            // Pseudo-peripheral start: hop to the far end until the depth
            // stops growing.
            for _ in 0..4 {
                let far = *levels.last().unwrap().iter().min_by_key(|&&v| self.degree(v)).unwrap();
                let l2 = self.bfs(far, id);
                if l2.len() <= levels.len() {
                    break;
                }
                levels = l2;
            }
            if levels.len() < 3 {
                out.copy_from_slice(set);
                return;
            }
            // The median level separates the levels before it from those after.
            let half = set.len() / 2;
            let mut acc = 0;
            let mut m = 1;
            for (i, l) in levels.iter().enumerate() {
                acc += l.len();
                if acc >= half {
                    m = i.clamp(1, levels.len() - 2);
                    break;
                }
            }
            let a: Vec<usize> = levels[..m].concat();
            let b: Vec<usize> = levels[m + 1..].concat();
            let sep = &levels[m];
            let (na, nb) = (a.len(), b.len());
            out[na + nb..].copy_from_slice(sep);
            self.dissect(&a, &mut out[..na]);
            self.dissect(&b, &mut out[na..na + nb]);
        }
    }
    let n = a.n;
    let mut nd = Nd {
        a,
        mark: vec![0; n],
        seen: vec![0; n],
        stamp: 0,
    };
    let all: Vec<usize> = (0..n).collect();
    let mut out = vec![0; n];
    nd.dissect(&all, &mut out);
    out
}

/// Nested dissection by coordinates: `coords[i]` is where unknown `i`
/// lives (the three displacements of one node share a position). Each set is
/// cut at the median of its longest axis, and the separator is whichever
/// side's frontier — vertices with a neighbour across the cut — is smaller.
/// On a mesh this finds plane separators of about `n^{2/3}` vertices, which
/// is what keeps the 3D factor at `n^{4/3}`.
pub fn nested_dissection_geometric(a: &Csr, coords: &[[f64; 3]]) -> Vec<usize> {
    assert_eq!(coords.len(), a.n);
    let n = a.n;
    let mut side = vec![0u8; n];
    let mut member = vec![0u32; n];
    let mut stamp = 0u32;
    let mut out = vec![0usize; n];
    // Explicit stack of (set, output offset).
    let mut work: Vec<(Vec<usize>, usize)> = vec![((0..n).collect(), 0)];
    while let Some((mut set, at)) = work.pop() {
        if set.len() <= 64 {
            out[at..at + set.len()].copy_from_slice(&set);
            continue;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &v in &set {
            for k in 0..3 {
                lo[k] = lo[k].min(coords[v][k]);
                hi[k] = hi[k].max(coords[v][k]);
            }
        }
        let axis = (0..3).max_by(|&x, &y| (hi[x] - lo[x]).total_cmp(&(hi[y] - lo[y]))).unwrap();
        if hi[axis] <= lo[axis] {
            out[at..at + set.len()].copy_from_slice(&set);
            continue;
        }
        let mid = set.len() / 2;
        set.select_nth_unstable_by(mid, |&x, &y| coords[x][axis].total_cmp(&coords[y][axis]));
        let cut = coords[set[mid]][axis];
        stamp += 1;
        for &v in &set {
            member[v] = stamp;
            // Ties go up, so unknowns sharing a node stay together.
            side[v] = if coords[v][axis] < cut { 0 } else { 1 };
        }
        // Frontier of each side within the set.
        let mut frontier = [Vec::new(), Vec::new()];
        for &v in &set {
            let (c, _) = a.row(v);
            if c.iter().any(|&u| member[u as usize] == stamp && side[u as usize] != side[v]) {
                frontier[side[v] as usize].push(v);
            }
        }
        let s = if frontier[0].len() <= frontier[1].len() { 0 } else { 1 };
        for &v in &frontier[s] {
            side[v] = 2;
        }
        let part = |k: u8| -> Vec<usize> { set.iter().copied().filter(|&v| side[v] == k).collect() };
        let (pa, pb, sep) = (part(0), part(1), part(2));
        if pa.is_empty() || pb.is_empty() {
            out[at..at + set.len()].copy_from_slice(&set);
            continue;
        }
        let (na, nb) = (pa.len(), pb.len());
        out[at + na + nb..at + set.len()].copy_from_slice(&sep);
        work.push((pa, at));
        work.push((pb, at + na));
    }
    out
}

const NONE: usize = usize::MAX;

/// A sparse Cholesky factor `L` by columns: column `j` holds the diagonal
/// first, then its rows below in increasing order.
#[derive(Clone, Debug)]
pub struct SparseCholesky {
    n: usize,
    lp: Vec<usize>,
    li: Vec<u32>,
    lx: Vec<f64>,
}

impl SparseCholesky {
    /// The elimination tree of a symmetric pattern.
    fn etree(a: &Csr) -> Vec<usize> {
        let n = a.n;
        let mut parent = vec![NONE; n];
        let mut ancestor = vec![NONE; n];
        for k in 0..n {
            let (c, _) = a.row(k);
            for &i in c {
                let mut i = i as usize;
                while i != NONE && i < k {
                    let next = ancestor[i];
                    ancestor[i] = k;
                    if next == NONE {
                        parent[i] = k;
                    }
                    i = next;
                }
            }
        }
        parent
    }

    /// The pattern of row `k` of `L` (excluding the diagonal) in
    /// topological order, written to `stack[top..]`; returns `top`.
    fn ereach(a: &Csr, k: usize, parent: &[usize], flag: &mut [usize], stack: &mut [usize]) -> usize {
        let n = a.n;
        let mut top = n;
        flag[k] = k;
        let (c, _) = a.row(k);
        for &i in c {
            let mut i = i as usize;
            if i > k {
                continue;
            }
            let mut len = 0;
            while flag[i] != k {
                stack[len] = i;
                len += 1;
                flag[i] = k;
                i = parent[i];
            }
            while len > 0 {
                len -= 1;
                top -= 1;
                stack[top] = stack[len];
            }
        }
        top
    }

    /// Entries the factor of `a` will have and the multiply-adds it will
    /// take, or `None` once the entries pass `limit`.
    pub fn factor_size(a: &Csr, limit: usize) -> Option<(usize, f64)> {
        let n = a.n;
        let parent = Self::etree(a);
        let mut flag = vec![NONE; n];
        let mut stack = vec![0; n];
        let mut count = vec![1usize; n];
        let mut total = 0;
        for k in 0..n {
            let top = Self::ereach(a, k, &parent, &mut flag, &mut stack);
            total += n - top + 1;
            if total > limit {
                return None;
            }
            for &i in &stack[top..] {
                count[i] += 1;
            }
        }
        let flops = count.iter().map(|&c| (c as f64) * (c as f64)).sum();
        Some((total, flops))
    }

    /// Factor `a`, which must already be in the order it is to be factored.
    pub fn factor(a: &Csr) -> Result<Self, FemError> {
        let n = a.n;
        let parent = Self::etree(a);
        let mut flag = vec![NONE; n];
        let mut stack = vec![0; n];
        // Symbolic: column counts.
        let mut count = vec![1usize; n];
        for k in 0..n {
            let top = Self::ereach(a, k, &parent, &mut flag, &mut stack);
            for &i in &stack[top..] {
                count[i] += 1;
            }
        }
        let mut lp = vec![0usize; n + 1];
        for j in 0..n {
            lp[j + 1] = lp[j] + count[j];
        }
        let nnz = lp[n];
        let mut li = vec![0u32; nnz];
        let mut lx = vec![0.0f64; nnz];
        // Next free slot per column; slot lp[j] is the diagonal, filled last.
        let mut next: Vec<usize> = (0..n).map(|j| lp[j] + 1).collect();
        let mut x = vec![0.0f64; n];
        flag.iter_mut().for_each(|f| *f = NONE);
        for k in 0..n {
            let top = Self::ereach(a, k, &parent, &mut flag, &mut stack);
            let (c, v) = a.row(k);
            let mut akk = 0.0;
            for (c, v) in c.iter().zip(v) {
                let i = *c as usize;
                if i < k {
                    x[i] = *v;
                } else if i == k {
                    akk = *v;
                }
            }
            let mut d = akk;
            for &i in &stack[top..] {
                let lki = x[i] / lx[lp[i]];
                x[i] = 0.0;
                for p in lp[i] + 1..next[i] {
                    x[li[p] as usize] -= lx[p] * lki;
                }
                d -= lki * lki;
                let p = next[i];
                next[i] += 1;
                li[p] = k as u32;
                lx[p] = lki;
            }
            if d.is_nan() || d <= 1e-12 * akk.abs() {
                return Err(FemError::NotPositiveDefinite);
            }
            li[lp[k]] = k as u32;
            lx[lp[k]] = d.sqrt();
        }
        Ok(Self { n, lp, li, lx })
    }

    /// Entries in the factor.
    pub fn nnz(&self) -> usize {
        self.lx.len()
    }

    pub fn solve_in_place(&self, x: &mut [f64]) {
        for j in 0..self.n {
            let r = self.lp[j]..self.lp[j + 1];
            x[j] /= self.lx[r.start];
            let xj = x[j];
            for p in r.start + 1..r.end {
                x[self.li[p] as usize] -= self.lx[p] * xj;
            }
        }
        for j in (0..self.n).rev() {
            let r = self.lp[j]..self.lp[j + 1];
            let mut s = x[j];
            for p in r.start + 1..r.end {
                s -= self.lx[p] * x[self.li[p] as usize];
            }
            x[j] = s / self.lx[r.start];
        }
    }
}

/// Conjugate gradients on `a` with a Jacobi preconditioner, from `x` as the
/// initial guess. Returns the iteration count.
pub fn pcg(
    a: &Csr,
    b: &[f64],
    x: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<usize, FemError> {
    let n = a.n;
    let inv_diag: Vec<f64> = a
        .diagonal()
        .iter()
        .map(|d| if *d > 0.0 { 1.0 / d } else { 1.0 })
        .collect();
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        return Ok(0);
    }
    let mut r = a.mul_vec(x);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let mut z: Vec<f64> = r.iter().zip(&inv_diag).map(|(r, d)| r * d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut ap = vec![0.0; n];
    for it in 0..max_iterations {
        let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rnorm <= tolerance * bnorm {
            return Ok(it);
        }
        a.mul_vec_into(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            return Err(FemError::NotPositiveDefinite);
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Err(FemError::NoConvergence {
        iterations: max_iterations,
    })
}

/// A sparse SPD matrix prepared for repeated solves.
pub enum SpdFactor {
    Direct {
        perm: Vec<usize>,
        factor: SparseCholesky,
    },
    Iterative {
        a: Csr,
        tolerance: f64,
        max_iterations: usize,
    },
}

impl SpdFactor {
    /// Prepare `a`, ordering it by graph alone.
    pub fn new(a: Csr, how: LinearSolver) -> Result<Self, FemError> {
        Self::prepare(a, None, how, 1.0)
    }

    /// Prepare `a`, whose unknown `i` lives at `coords[i]` — a far better
    /// ordering on a mesh. `solves` is how many right-hand sides will follow:
    /// the more there are, the more a direct factor is worth.
    pub fn with_coords(a: Csr, coords: &[[f64; 3]], how: LinearSolver, solves: f64) -> Result<Self, FemError> {
        Self::prepare(a, Some(coords), how, solves)
    }

    fn prepare(a: Csr, coords: Option<&[[f64; 3]]>, how: LinearSolver, solves: f64) -> Result<Self, FemError> {
        let (tolerance, max_iterations) = match how {
            LinearSolver::Iterative {
                tolerance,
                max_iterations,
            } => (tolerance, max_iterations),
            _ => (1e-10, 20_000),
        };
        if !matches!(how, LinearSolver::Iterative { .. }) {
            let perm = match coords {
                Some(c) => nested_dissection_geometric(&a, c),
                None => nested_dissection(&a),
            };
            let p = permute(&a, &perm);
            let fits = how == LinearSolver::Direct
                || SparseCholesky::factor_size(&p, DIRECT_BUDGET)
                    .is_some_and(|(_, flops)| flops <= DIRECT_FLOPS * solves.max(1.0).sqrt());
            if fits {
                let factor = SparseCholesky::factor(&p)?;
                return Ok(Self::Direct { perm, factor });
            }
        }
        Ok(Self::Iterative {
            a,
            tolerance,
            max_iterations,
        })
    }

    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, FemError> {
        self.solve_from(b, None)
    }

    /// Solve, starting an iterative solve from `guess` (ignored by a direct
    /// factor, which needs none).
    pub fn solve_from(&self, b: &[f64], guess: Option<&[f64]>) -> Result<Vec<f64>, FemError> {
        match self {
            Self::Direct { perm, factor } => {
                let mut x: Vec<f64> = perm.iter().map(|&o| b[o]).collect();
                factor.solve_in_place(&mut x);
                let mut out = vec![0.0; b.len()];
                for (new, &old) in perm.iter().enumerate() {
                    out[old] = x[new];
                }
                Ok(out)
            }
            Self::Iterative {
                a,
                tolerance,
                max_iterations,
            } => {
                let mut x = guess.map_or_else(|| vec![0.0; b.len()], <[f64]>::to_vec);
                pcg(a, b, &mut x, *tolerance, *max_iterations)?;
                Ok(x)
            }
        }
    }
}

/// `P·A·Pᵀ` for `perm[new] = old`.
pub fn permute(a: &Csr, perm: &[usize]) -> Csr {
    let mut inv = vec![0u32; a.n];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new as u32;
    }
    let mut row_ptr = Vec::with_capacity(a.n + 1);
    row_ptr.push(0);
    let mut cols = Vec::with_capacity(a.nnz());
    let mut vals = Vec::with_capacity(a.nnz());
    let mut row: Vec<(u32, f64)> = Vec::new();
    for &old in perm {
        let (c, v) = a.row(old);
        row.clear();
        row.extend(c.iter().zip(v).map(|(c, v)| (inv[*c as usize], *v)));
        row.sort_unstable_by_key(|e| e.0);
        for (c, v) in &row {
            cols.push(*c);
            vals.push(*v);
        }
        row_ptr.push(cols.len());
    }
    Csr {
        n: a.n,
        row_ptr,
        cols,
        vals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_2d(m: usize) -> Csr {
        let n = m * m;
        let mut t = Triplets::new(n);
        for y in 0..m {
            for x in 0..m {
                let i = y * m + x;
                t.add(i, i, 4.0 + 1e-3);
                if x + 1 < m {
                    t.add(i, i + 1, -1.0);
                    t.add(i + 1, i, -1.0);
                }
                if y + 1 < m {
                    t.add(i, i + m, -1.0);
                    t.add(i + m, i, -1.0);
                }
            }
        }
        t.to_csr()
    }

    #[test]
    fn direct_and_iterative_agree() {
        let a = laplacian_2d(30);
        let b: Vec<f64> = (0..a.n).map(|i| ((i * 37) % 17) as f64 - 8.0).collect();
        let d = SpdFactor::new(a.clone(), LinearSolver::Direct)
            .unwrap()
            .solve(&b)
            .unwrap();
        let it = SpdFactor::new(a.clone(), LinearSolver::ITERATIVE)
            .unwrap()
            .solve(&b)
            .unwrap();
        let r = a.mul_vec(&d);
        for i in 0..a.n {
            assert!((r[i] - b[i]).abs() < 1e-9);
            assert!((d[i] - it[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn nested_dissection_beats_the_natural_order() {
        let a = laplacian_2d(60);
        let natural = SparseCholesky::factor_size(&a, usize::MAX).unwrap().0;
        let perm = nested_dissection(&a);
        let mut seen = vec![false; a.n];
        for &p in &perm {
            assert!(!seen[p]);
            seen[p] = true;
        }
        let nd = SparseCholesky::factor_size(&permute(&a, &perm), usize::MAX).unwrap().0;
        assert!(nd * 2 < natural, "{nd} vs {natural}");
        let coords: Vec<[f64; 3]> = (0..a.n).map(|i| [(i % 60) as f64, (i / 60) as f64, 0.0]).collect();
        let geo = nested_dissection_geometric(&a, &coords);
        let mut sorted = geo.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &p)| i == p));
        let g = SparseCholesky::factor_size(&permute(&a, &geo), usize::MAX).unwrap().0;
        assert!(g * 2 < natural, "{g} vs {natural}");
    }

    #[test]
    fn singular_matrix_is_refused() {
        // A graph Laplacian with no grounding: constants are in its null space.
        let mut t = Triplets::new(3);
        for (i, j) in [(0, 1), (1, 2)] {
            t.add(i, i, 1.0);
            t.add(j, j, 1.0);
            t.add(i, j, -1.0);
            t.add(j, i, -1.0);
        }
        assert!(matches!(
            SpdFactor::new(t.to_csr(), LinearSolver::Direct),
            Err(FemError::NotPositiveDefinite)
        ));
    }
}
