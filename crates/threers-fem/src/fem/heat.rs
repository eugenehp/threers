//! Steady heat conduction: `−∇·(k∇T) = Q`.
//!
//! The same equation governs electric potential in a conductor (σ for k,
//! current for heat), steady diffusion and groundwater seepage, so this is
//! the scalar Poisson solver for all of them; only the names are thermal.

use super::{gradient_products, per_chunk, solve_constrained, Order, Region, Space};
use crate::mesh::TetMesh;
use crate::sparse::{LinearSolver, Triplets};
use crate::vec3::*;
use crate::FemError;

enum Boundary<'a> {
    /// Heat flux into the body per unit area.
    Flux(Region<'a>, f64),
    /// Convection to a fluid at `ambient` with film coefficient `h`.
    Convection(Region<'a>, f64, f64),
}

/// A steady conduction analysis of a tetrahedral mesh.
///
/// ```
/// use threers_fem::{Heat, TetMesh};
///
/// // A 1 m slab at 100° on one face and 0° on the other: linear in between.
/// let mesh = TetMesh::box_grid([0.0; 3], [1.0, 0.2, 0.2], [5, 1, 1]);
/// let s = Heat::new(&mesh, 50.0)
///     .temperature(|p| p[0] < 1e-9, 100.0)
///     .temperature(|p| p[0] > 1.0 - 1e-9, 0.0)
///     .solve()
///     .unwrap();
/// let mid = s.temperature_at([0.4, 0.1, 0.1]);
/// assert!((mid - 60.0).abs() < 1e-9);
/// // q = k ΔT / L = 5000 W/m², along +x.
/// assert!((s.flux[0][0] - 5000.0).abs() < 1e-6);
/// // What enters at the hot face leaves at the cold one.
/// assert!(s.fixed_heat_inflow.abs() < 1e-6);
/// ```
pub struct Heat<'a> {
    mesh: &'a TetMesh,
    conductivity: [[f64; 3]; 3],
    order: Order,
    solver: LinearSolver,
    fixed: Vec<(Region<'a>, f64)>,
    boundaries: Vec<Boundary<'a>>,
    source: Option<Box<dyn Fn(V3) -> f64 + Send + Sync + 'a>>,
}

impl<'a> Heat<'a> {
    /// Isotropic conductivity `k`.
    pub fn new(mesh: &'a TetMesh, conductivity: f64) -> Self {
        let k = conductivity;
        Self::anisotropic(mesh, [[k, 0.0, 0.0], [0.0, k, 0.0], [0.0, 0.0, k]])
    }

    /// A conductivity tensor — a laminate, a printed part, a lattice from
    /// `homogenize_conduction`.
    pub fn anisotropic(mesh: &'a TetMesh, conductivity: [[f64; 3]; 3]) -> Self {
        Self {
            mesh,
            conductivity,
            order: Order::Quadratic,
            solver: LinearSolver::Auto,
            fixed: Vec::new(),
            boundaries: Vec::new(),
            source: None,
        }
    }

    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

    pub fn solver(mut self, solver: LinearSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Hold the nodes in `region` at temperature `t`.
    pub fn temperature(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, t: f64) -> Self {
        self.fixed.push((Box::new(region), t));
        self
    }

    /// Heat flux `q` per unit area *into* the body through the boundary faces
    /// in `region` (negative draws heat out). Faces with nothing set are
    /// insulated.
    pub fn flux(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, q: f64) -> Self {
        self.boundaries.push(Boundary::Flux(Box::new(region), q));
        self
    }

    /// Convection on the faces in `region` to a fluid at `ambient`, with film
    /// coefficient `h` (W/m²K): a flux of `h (ambient − T)` in.
    pub fn convection(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, h: f64, ambient: f64) -> Self {
        self.boundaries.push(Boundary::Convection(Box::new(region), h, ambient));
        self
    }

    /// Uniform heat generation per unit volume.
    pub fn source(self, q: f64) -> Self {
        self.source_fn(move |_| q)
    }

    /// Heat generation per unit volume as a function of position. It is
    /// sampled at element nodes and interpolated.
    pub fn source_fn(mut self, q: impl Fn(V3) -> f64 + Send + Sync + 'a) -> Self {
        self.source = Some(Box::new(q));
        self
    }

    pub fn solve(&self) -> Result<HeatSolution, FemError> {
        if self.mesh.tets.is_empty() {
            return Err(FemError::InvalidInput("mesh has no tetrahedra"));
        }
        let space = Space::new(self.mesh, self.order);
        let r = self.order.reference();
        let npe = self.order.nodes();
        let n = space.nodes.len();
        let kappa = self.conductivity;

        let parts = per_chunk(space.elem_count(), |range| {
            let mut t = Triplets::with_capacity(n, range.len() * npe * npe);
            let mut gp = Vec::new();
            for e in range {
                gradient_products(r, &space.gradients[e], space.volumes[e], &mut gp);
                let nodes = space.elem(e);
                for a in 0..npe {
                    for b in 0..npe {
                        let g = &gp[(a * npe + b) * 9..(a * npe + b + 1) * 9];
                        let mut s = 0.0;
                        for i in 0..3 {
                            for j in 0..3 {
                                s += kappa[i][j] * g[i * 3 + j];
                            }
                        }
                        t.add(nodes[a] as usize, nodes[b] as usize, s);
                    }
                }
            }
            t
        });
        let mut trip = Triplets::new(n);
        for p in parts {
            trip.extend(p);
        }

        let mut f = vec![0.0; n];
        if let Some(q) = &self.source {
            for e in 0..space.elem_count() {
                let nodes = space.elem(e);
                let qv: Vec<f64> = nodes.iter().map(|&i| q(space.nodes[i as usize])).collect();
                // ∫ Nₐ Σ_b N_b q_b = V Σ_b M_ab q_b.
                for a in 0..npe {
                    let mut s = 0.0;
                    for b in 0..npe {
                        s += r.mass[a * npe + b] * qv[b];
                    }
                    f[nodes[a] as usize] += space.volumes[e] * s;
                }
            }
        }
        let nf = self.order.face_nodes();
        let mut convective = Vec::new();
        for b in &self.boundaries {
            match b {
                Boundary::Flux(region, q) => {
                    let faces = space.faces_where(region);
                    if faces.is_empty() {
                        return Err(FemError::EmptyRegion("flux"));
                    }
                    for fi in faces {
                        let (_, area) = space.face_normal_area(fi);
                        for (a, &node) in space.face(fi).iter().enumerate() {
                            f[node as usize] += area * r.face_load[a] * q;
                        }
                    }
                }
                Boundary::Convection(region, h, ambient) => {
                    let faces = space.faces_where(region);
                    if faces.is_empty() {
                        return Err(FemError::EmptyRegion("convection"));
                    }
                    for &fi in &faces {
                        let (_, area) = space.face_normal_area(fi);
                        let nodes = space.face(fi);
                        for a in 0..nf {
                            f[nodes[a] as usize] += area * r.face_load[a] * h * ambient;
                            for b in 0..nf {
                                trip.add(
                                    nodes[a] as usize,
                                    nodes[b] as usize,
                                    area * h * r.face_mass[a * nf + b],
                                );
                            }
                        }
                    }
                    convective.push((faces, *h, *ambient));
                }
            }
        }
        let k = trip.to_csr();

        let mut fixed = vec![None; n];
        for (region, t) in &self.fixed {
            let nodes = space.nodes_where(region);
            if nodes.is_empty() {
                return Err(FemError::EmptyRegion("temperature"));
            }
            for i in nodes {
                fixed[i] = Some(*t);
            }
        }
        if fixed.iter().all(|f| f.is_none()) && convective.is_empty() {
            return Err(FemError::InvalidInput(
                "no fixed temperature or convection: the temperature level is undetermined",
            ));
        }
        let (temp, residual) = solve_constrained(&k, &f, &fixed, &space.dof_coords(1), self.solver)?;
        let fixed_heat_inflow = residual
            .iter()
            .zip(&fixed)
            .filter(|(_, f)| f.is_some())
            .map(|(r, _)| r)
            .sum();
        // Heat convected away: ∫ h (T − T∞) dA.
        let mut convected_out = 0.0;
        for (faces, h, ambient) in &convective {
            for &fi in faces {
                let (_, area) = space.face_normal_area(fi);
                let mean: f64 = space
                    .face(fi)
                    .iter()
                    .enumerate()
                    .map(|(a, &node)| r.face_load[a] * temp[node as usize])
                    .sum();
                convected_out += h * area * (mean - ambient);
            }
        }
        let flux = space.average_to_corners::<3>(|e, c| {
            let g = &space.gradients[e];
            let nodes = space.elem(e);
            let mut grad = [0.0; 3];
            for a in 0..npe {
                for kk in 0..4 {
                    let w = r.grad_at_corner[(c * npe + a) * 4 + kk];
                    if w != 0.0 {
                        grad = add(grad, scale(g[kk], w * temp[nodes[a] as usize]));
                    }
                }
            }
            // q = −κ ∇T
            [0, 1, 2].map(|i| -(0..3).map(|j| kappa[i][j] * grad[j]).sum::<f64>())
        });
        Ok(HeatSolution {
            nodes: self.mesh.nodes.clone(),
            tets: self.mesh.tets.clone(),
            temperature: temp[..space.corner_count].to_vec(),
            all_temperature: temp,
            elems: space.elems.clone(),
            order: self.order,
            flux,
            fixed_heat_inflow,
            convected_out,
        })
    }
}

/// The answer to a [`Heat`] analysis.
#[derive(Clone, Debug)]
pub struct HeatSolution {
    /// The mesh nodes.
    pub nodes: Vec<V3>,
    tets: Vec<[u32; 4]>,
    /// Temperature at each mesh node.
    pub temperature: Vec<f64>,
    all_temperature: Vec<f64>,
    elems: Vec<u32>,
    order: Order,
    /// Heat flux vector `−k∇T` at each mesh node.
    pub flux: Vec<V3>,
    /// Net heat entering through the fixed-temperature nodes. With no other
    /// sources or sinks it is zero: what enters at the hot face leaves at the
    /// cold one.
    pub fixed_heat_inflow: f64,
    /// Heat leaving by convection.
    pub convected_out: f64,
}

impl HeatSolution {
    pub fn max_temperature(&self) -> f64 {
        self.temperature.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_temperature(&self) -> f64 {
        self.temperature.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Temperature at any point inside the mesh, interpolated with the
    /// element's own shape functions. `NaN` outside.
    pub fn temperature_at(&self, p: V3) -> f64 {
        let npe = self.order.nodes();
        for (e, tet) in self.tets.iter().enumerate() {
            let c = tet.map(|i| self.nodes[i as usize]);
            let Some(l) = barycentric(c, p) else { continue };
            let nodes = &self.elems[e * npe..(e + 1) * npe];
            let mut t = 0.0;
            for (a, &n) in nodes.iter().enumerate() {
                t += shape_value(self.order, a, l) * self.all_temperature[n as usize];
            }
            return t;
        }
        f64::NAN
    }
}

/// Barycentric coordinates of `p` in tetrahedron `c`, if inside (with a
/// small tolerance).
pub(crate) fn barycentric(c: [V3; 4], p: V3) -> Option<[f64; 4]> {
    let v = tet_volume6(c[0], c[1], c[2], c[3]);
    if v == 0.0 {
        return None;
    }
    let l1 = tet_volume6(c[0], p, c[2], c[3]) / v;
    let l2 = tet_volume6(c[0], c[1], p, c[3]) / v;
    let l3 = tet_volume6(c[0], c[1], c[2], p) / v;
    let l0 = 1.0 - l1 - l2 - l3;
    let l = [l0, l1, l2, l3];
    if l.iter().all(|&x| x >= -1e-9) { Some(l) } else { None }
}

pub(crate) fn shape_value(order: Order, a: usize, l: [f64; 4]) -> f64 {
    match order {
        Order::Linear => l[a],
        Order::Quadratic => {
            if a < 4 {
                l[a] * (2.0 * l[a] - 1.0)
            } else {
                let (p, q) = super::TET10_EDGES[a - 4];
                4.0 * l[p] * l[q]
            }
        }
    }
}
