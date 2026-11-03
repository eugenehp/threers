//! Timoshenko-style 3D frame model for cuboct lattices (Jenett et al. 2020).
//!
//! Builds a beam network from the same segment generator as [`super::cuboct`],
//! assembles axial + weak bending stiffness, and runs quasi-static compression
//! tests for effective modulus, Poisson ratio, and chiral twist per strain.

use super::cuboct::{cell_segments, ChiralRule, Cuboct};
use super::Field;
use crate::math::{Box3, Vector3};
use std::collections::HashMap;

/// Material and section properties for the reduced-order frame model.
#[derive(Clone, Copy, Debug)]
pub struct FrameMaterial {
    /// Young's modulus (same units as stress output).
    pub e: f64,
    /// Beam cross-section area.
    pub a: f64,
    /// Second moment (used for weak bending regularisation).
    pub i: f64,
}

impl Default for FrameMaterial {
    fn default() -> Self {
        Self {
            e: 1.0,
            a: 1.0,
            i: 0.01,
        }
    }
}

/// Quasi-static test results for a cuboct specimen.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameResponse {
    /// Effective axial stiffness F/δ.
    pub stiffness: f64,
    /// Effective modulus E* = stiffness * H / A.
    pub effective_modulus: f64,
    /// ν = ε_trans / ε_axial (negative ⇒ auxetic).
    pub poisson: f64,
    /// Degrees twist per unit axial strain (chiral columns).
    pub twist_per_strain: f64,
}

/// Frame mesh extracted from a periodic cuboct lattice block.
pub struct CuboctFrame {
    nodes: Vec<[f64; 3]>,
    /// `(node_a, node_b, length)`.
    elements: Vec<(usize, usize, f64)>,
    pitch: f64,
    cells: [usize; 3],
    mat: FrameMaterial,
}

impl CuboctFrame {
    /// Build a frame model for `cells` voxels of `kind` at `pitch`.
    pub fn new(cells: [usize; 3], kind: Cuboct, pitch: f32, shape: f32) -> Self {
        Self::with_program(cells, pitch, shape, None, |_, _, _| kind)
    }

    /// Per-cell programming, same convention as [`super::CuboctAssembly::program`].
    pub fn with_program(
        cells: [usize; 3],
        pitch: f32,
        shape: f32,
        rule: Option<ChiralRule>,
        mut kind_at: impl FnMut(i32, i32, i32) -> Cuboct,
    ) -> Self {
        let pitch = pitch as f64;
        let [nx, ny, nz] = cells;
        let n = nx as i32;
        let weld = pitch * 1e-4;
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut elements = Vec::new();
        let mut index = HashMap::new();

        let mut node_id = |p: [f64; 3]| -> usize {
            for (i, q) in nodes.iter().enumerate() {
                let d = [(p[0] - q[0]).abs(), (p[1] - q[1]).abs(), (p[2] - q[2]).abs()];
                if d[0] < weld && d[1] < weld && d[2] < weld {
                    return i;
                }
            }
            let id = nodes.len();
            nodes.push(p);
            index.insert(
                (
                    (p[0] / weld).round() as i64,
                    (p[1] / weld).round() as i64,
                    (p[2] / weld).round() as i64,
                ),
                id,
            );
            id
        };

        let origin = |ix: i32, iy: i32, iz: i32| {
            [
                (ix as f64 - nx as f64 * 0.5) * pitch,
                (iy as f64 - ny as f64 * 0.5) * pitch,
                (iz as f64 - nz as f64 * 0.5) * pitch,
            ]
        };

        for iz in 0..nz as i32 {
            for iy in 0..ny as i32 {
                for ix in 0..nx as i32 {
                    let kind = kind_at(ix, iy, iz);
                    let o = origin(ix, iy, iz);
                    let segs = cell_segments(kind, shape, ix, iy, iz, rule, n);
                    for seg in segs {
                        let a = [
                            o[0] + seg[0][0] as f64 * pitch,
                            o[1] + seg[0][1] as f64 * pitch,
                            o[2] + seg[0][2] as f64 * pitch,
                        ];
                        let b = [
                            o[0] + seg[1][0] as f64 * pitch,
                            o[1] + seg[1][1] as f64 * pitch,
                            o[2] + seg[1][2] as f64 * pitch,
                        ];
                        let ia = node_id(a);
                        let ib = node_id(b);
                        if ia == ib {
                            continue;
                        }
                        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2))
                            .sqrt()
                            .max(weld);
                        elements.push((ia, ib, len));
                    }
                }
            }
        }

        Self {
            nodes,
            elements,
            pitch,
            cells,
            mat: FrameMaterial::default(),
        }
    }

    pub fn material(mut self, mat: FrameMaterial) -> Self {
        self.mat = mat;
        self
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Uniaxial compression along +Z with strain `epsilon` (positive = compress).
    pub fn compress_z(&self, epsilon: f64) -> FrameResponse {
        self.solve_compress_z(epsilon).1
    }

    /// Axial stress in every beam under the same test, at the beam's midpoint.
    ///
    /// Negative in compression. This is the other half of a field-driven
    /// workflow: solve the frame, and grade the lattice on what it says.
    ///
    /// ```no_run
    /// use threers::{Cuboct, CuboctFrame, Lattice, LatticeKind, Vector3};
    ///
    /// let frame = CuboctFrame::new([4, 4, 4], Cuboct::Rigid, 10.0, 0.15);
    /// let stress = frame.stress_field(0.01, [12, 12, 12]).map(f32::abs);
    ///
    /// let geom = Lattice::new(LatticeKind::Cuboct(Cuboct::Rigid))
    ///     .size(Vector3::new(40.0, 40.0, 40.0))
    ///     .cells([4, 4, 4])
    ///     .thickness(0.8)
    ///     // Thin where the frame says nothing is happening.
    ///     .grade(stress.into_grade(0.6, 1.6))
    ///     .build();
    /// # let _ = geom;
    /// ```
    pub fn element_stress(&self, epsilon: f64) -> Vec<(Vector3, f32)> {
        let (u, _) = self.solve_compress_z(epsilon);
        if u.is_empty() {
            return Vec::new();
        }
        self.elements
            .iter()
            .map(|&(a, b, len)| {
                let (pa, pb) = (self.nodes[a], self.nodes[b]);
                let mut strain = 0.0;
                for d in 0..3 {
                    strain += (u[b * 3 + d] - u[a * 3 + d]) * (pb[d] - pa[d]) / len;
                }
                let mid = Vector3::new(
                    ((pa[0] + pb[0]) * 0.5) as f32,
                    ((pa[1] + pb[1]) * 0.5) as f32,
                    ((pa[2] + pb[2]) * 0.5) as f32,
                );
                (mid, (self.mat.e * strain / len) as f32)
            })
            .collect()
    }

    /// [`element_stress`](Self::element_stress) resampled onto a grid, ready
    /// to hand to [`grade`](super::Lattice::grade).
    ///
    /// An empty frame gives a field of zeros rather than an empty one, so a
    /// grade built from it is a no-op rather than a panic.
    pub fn stress_field(&self, epsilon: f64, dims: [usize; 3]) -> Field {
        let samples = self.element_stress(epsilon);
        if samples.is_empty() {
            let unit = Box3::from_center_and_size(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0));
            return Field::scattered(unit, [2, 2, 2], &[]);
        }
        let points: Vec<Vector3> = samples.iter().map(|(p, _)| *p).collect();
        let mut bounds = Box3::from_points(&points);
        // Half a cell of margin: the beams' midpoints stop short of the
        // block's own faces, and a lattice filling that block samples past
        // them.
        bounds.expand_by_scalar(self.pitch as f32 * 0.5);
        Field::scattered(bounds, dims, &samples)
    }

    fn solve_compress_z(&self, epsilon: f64) -> (Vec<f64>, FrameResponse) {
        let ndof = self.nodes.len() * 3;
        if ndof == 0 {
            return (Vec::new(), FrameResponse::default());
        }
        let mut k = vec![0.0f64; ndof * ndof];
        for &(a, b, len) in &self.elements {
            add_truss(&mut k, ndof, a, b, self.mat.e * self.mat.a / len, &self.nodes);
            let bend = self.mat.e * self.mat.i / len.powi(3);
            add_bending(&mut k, ndof, a, b, bend, &self.nodes);
        }

        let zmin = self
            .nodes
            .iter()
            .map(|p| p[2])
            .fold(f64::INFINITY, f64::min);
        let zmax = self
            .nodes
            .iter()
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let tol = self.pitch * 0.05;

        let mut fixed = vec![false; ndof];
        for (i, p) in self.nodes.iter().enumerate() {
            if (p[2] - zmin).abs() < tol {
                fixed[i * 3] = true;
                fixed[i * 3 + 1] = true;
                fixed[i * 3 + 2] = true;
            }
        }

        let height = zmax - zmin;
        let dz = -epsilon * height;
        let mut prescribed = vec![0.0f64; ndof];
        for (i, p) in self.nodes.iter().enumerate() {
            if (p[2] - zmax).abs() < tol {
                prescribed[i * 3 + 2] = dz;
                fixed[i * 3 + 2] = true;
            }
        }

        let u = solve_reduced(&k, &prescribed, &fixed);

        let area = (self.cells[0] as f64 * self.pitch) * (self.cells[1] as f64 * self.pitch);
        let reaction = reaction_z(&k, &u, ndof, zmin, tol, &self.nodes);
        let stiffness = reaction.abs() / dz.abs().max(1e-12);
        let effective_modulus = stiffness * height / area.max(1e-12);

        let cx = self.nodes.iter().map(|p| p[0]).sum::<f64>() / self.nodes.len() as f64;
        let cy = self.nodes.iter().map(|p| p[1]).sum::<f64>() / self.nodes.len() as f64;
        let mut lat0 = 0.0;
        let mut lat1 = 0.0;
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        for (i, p) in self.nodes.iter().enumerate() {
            let ux = u[i * 3];
            let uy = u[i * 3 + 1];
            let r = ((p[0] + ux - cx).powi(2) + (p[1] + uy - cy).powi(2)).sqrt();
            if (p[2] - zmin).abs() < tol {
                lat0 += r;
                n0 += 1;
            } else if (p[2] - zmax).abs() < tol {
                lat1 += r;
                n1 += 1;
            }
        }
        let eps_ax = dz / height;
        let r0 = lat0 / n0.max(1) as f64;
        let r1 = lat1 / n1.max(1) as f64;
        let eps_lat = (r1 - r0) / (self.pitch * 0.5).max(1e-12);
        let poisson = if eps_ax.abs() > 1e-9 {
            eps_lat / eps_ax
        } else {
            0.0
        };

        let mut twist = 0.0;
        let mut twist_n = 0usize;
        let cx = self.nodes.iter().map(|p| p[0]).sum::<f64>() / self.nodes.len() as f64;
        let cy = self.nodes.iter().map(|p| p[1]).sum::<f64>() / self.nodes.len() as f64;
        for (i, p) in self.nodes.iter().enumerate() {
            if (p[2] - zmax).abs() < tol {
                let rx = p[0] - cx;
                let ry = p[1] - cy;
                let r = (rx * rx + ry * ry).sqrt();
                if r > self.pitch * 0.05 {
                    let ux = u[i * 3];
                    let uy = u[i * 3 + 1];
                    twist += (rx * uy - ry * ux) / (r * r);
                    twist_n += 1;
                }
            }
        }
        let twist_per_strain = if twist_n > 0 && epsilon.abs() > 1e-9 {
            twist / twist_n as f64 / epsilon
        } else {
            0.0
        };

        (
            u,
            FrameResponse {
                stiffness,
                effective_modulus,
                poisson,
                twist_per_strain,
            },
        )
    }
}

fn add_truss(k: &mut [f64], ndof: usize, a: usize, b: usize, ea_l: f64, nodes: &[[f64; 3]]) {
    let mut c = [0.0; 3];
    let pa = nodes[a];
    let pb = nodes[b];
    let len = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2) + (pb[2] - pa[2]).powi(2))
        .sqrt()
        .max(1e-12);
    for d in 0..3 {
        c[d] = (pb[d] - pa[d]) / len;
    }
    let idx = [a * 3, a * 3 + 1, a * 3 + 2, b * 3, b * 3 + 1, b * 3 + 2];
    for i in 0..6 {
        for j in 0..6 {
            let ci = c[i % 3];
            let cj = c[j % 3];
            let sign = if (i < 3) == (j < 3) { 1.0 } else { -1.0 };
            add_k(k, ndof, idx[i], idx[j], ea_l * ci * cj * sign);
        }
    }
}

fn add_bending(k: &mut [f64], ndof: usize, a: usize, b: usize, ei_l3: f64, _nodes: &[[f64; 3]]) {
    for d in 0..3 {
        let ia = a * 3 + d;
        let ib = b * 3 + d;
        add_k(k, ndof, ia, ia, ei_l3);
        add_k(k, ndof, ib, ib, ei_l3);
        add_k(k, ndof, ia, ib, -ei_l3);
        add_k(k, ndof, ib, ia, -ei_l3);
    }
}

fn add_k(k: &mut [f64], n: usize, i: usize, j: usize, v: f64) {
    k[i * n + j] += v;
}

fn solve_reduced(k: &[f64], prescribed: &[f64], fixed: &[bool]) -> Vec<f64> {
    let n = prescribed.len();
    let free: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
    let m = free.len();
    let mut u = vec![0.0; n];
    if m == 0 {
        for i in 0..n {
            if fixed[i] {
                u[i] = prescribed[i];
            }
        }
        return u;
    }
    let mut kff = vec![0.0; m * m];
    let mut ff = vec![0.0; m];
    for (ii, &i) in free.iter().enumerate() {
        for j in 0..n {
            if fixed[j] {
                ff[ii] -= k[i * n + j] * prescribed[j];
            }
        }
        for (jj, &j) in free.iter().enumerate() {
            kff[ii * m + jj] = k[i * n + j];
        }
    }
    gauss_solve(&mut kff, &mut ff, m);
    for (ii, &i) in free.iter().enumerate() {
        u[i] = ff[ii];
    }
    for i in 0..n {
        if fixed[i] {
            u[i] = prescribed[i];
        }
    }
    u
}

fn gauss_solve(a: &mut [f64], b: &mut [f64], n: usize) {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best < 1e-14 {
            continue;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let diag = a[col * n + col];
        for k in col..n {
            a[col * n + k] /= diag;
        }
        b[col] /= diag;
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row * n + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
}

fn reaction_z(
    k: &[f64],
    u: &[f64],
    ndof: usize,
    zmin: f64,
    tol: f64,
    nodes: &[[f64; 3]],
) -> f64 {
    let mut fz = 0.0;
    for (i, p) in nodes.iter().enumerate() {
        if (p[2] - zmin).abs() < tol {
            for j in 0..ndof {
                fz += k[(i * 3 + 2) * ndof + j] * u[j];
            }
        }
    }
    fz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_modulus_rises_with_cell_count() {
        let e1 = CuboctFrame::new([1, 1, 1], Cuboct::Rigid, 1.0, 0.0)
            .compress_z(0.01)
            .effective_modulus;
        let e2 = CuboctFrame::new([2, 2, 2], Cuboct::Rigid, 1.0, 0.0)
            .compress_z(0.01)
            .effective_modulus;
        assert!(e1 > 0.0 && e2 > e1, "E*1={e1} E*2={e2}");
    }

    #[test]
    fn compliant_modulus_near_constant_across_n() {
        let e1 = CuboctFrame::new([1, 1, 1], Cuboct::Compliant, 1.0, 0.15)
            .compress_z(0.02)
            .effective_modulus;
        let e2 = CuboctFrame::new([2, 2, 2], Cuboct::Compliant, 1.0, 0.15)
            .compress_z(0.02)
            .effective_modulus;
        assert!(
            (e1 - e2).abs() / e1.max(1e-9) < 0.5,
            "compliant E* drifted: {e1} vs {e2}"
        );
    }

    #[test]
    fn a_compressed_frame_reports_where_the_load_is() {
        let frame = CuboctFrame::new([2, 2, 3], Cuboct::Rigid, 1.0, 0.0);
        let stress = frame.element_stress(0.01);
        assert_eq!(stress.len(), frame.element_count());
        // Compression, so the beams that carry it are in compression.
        assert!(stress.iter().any(|&(_, s)| s < 0.0), "nothing was loaded");
        assert!(stress.iter().all(|&(_, s)| s.is_finite()));
        // And the load is not the same everywhere — a lattice graded on a
        // constant field is a lattice that was not graded.
        let (lo, hi) = stress
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(_, s)| {
                (lo.min(s), hi.max(s))
            });
        assert!(hi - lo > 1e-9, "uniform stress: {lo} to {hi}");
    }

    #[test]
    fn the_stress_field_covers_the_block_it_was_solved_on() {
        let frame = CuboctFrame::new([2, 2, 2], Cuboct::Rigid, 10.0, 0.0);
        let field = frame.stress_field(0.01, [8, 8, 8]).map(f32::abs);
        let (lo, hi) = field.range();
        assert!(lo >= 0.0 && hi > lo, "{lo} to {hi}");
        // The block runs from -10 to +10 on each axis; the field has to be
        // defined out to its corners, where a lattice filling it will sample.
        let corner = field.value(Vector3::new(-10.0, -10.0, -10.0));
        assert!(corner.is_finite() && corner >= 0.0, "{corner}");
    }

    #[test]
    fn an_empty_frame_gives_a_field_of_nothing() {
        let frame = CuboctFrame::new([0, 0, 0], Cuboct::Rigid, 1.0, 0.0);
        assert!(frame.element_stress(0.01).is_empty());
        let field = frame.stress_field(0.01, [4, 4, 4]);
        assert_eq!(field.value(Vector3::ZERO), 0.0);
    }

    #[test]
    fn auxetic_negative_poisson() {
        let rigid = CuboctFrame::new([2, 2, 2], Cuboct::Rigid, 1.0, 0.0).compress_z(0.05);
        let aux = CuboctFrame::new([2, 2, 2], Cuboct::Auxetic, 1.0, 0.2).compress_z(0.05);
        assert!(
            aux.poisson < rigid.poisson,
            "auxetic nu={} should be below rigid nu={}",
            aux.poisson,
            rigid.poisson
        );
    }

    #[test]
    fn chiral_column_twists() {
        let r = CuboctFrame::with_program([1, 1, 4], 1.0, 0.12, Some(ChiralRule::R1), |_, _, k| {
            Cuboct::column_half(k, 4)
        })
        .compress_z(0.05);
        assert!(r.twist_per_strain.abs() > 5e-4, "twist = {}", r.twist_per_strain);
    }
}
