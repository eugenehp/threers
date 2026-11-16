//! Linear elasticity: small displacements of a solid under loads and supports.

// Tensor and matrix code: the index is the meaning.
#![allow(clippy::needless_range_loop)]

use super::{gradient_products, per_chunk, solve_constrained, Order, Region, Space};
use crate::mesh::TetMesh;
use crate::sparse::{Csr, LinearSolver, Triplets};
use crate::vec3::*;
use crate::FemError;

/// A linear elastic material.
///
/// Isotropic from [`new`](Self::new); anisotropic — a 3D-printed part, a
/// homogenised lattice, wood — from [`anisotropic`](Self::anisotropic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Young's modulus.
    pub youngs: f64,
    /// Poisson's ratio, in `(-1, 0.5)`.
    pub poisson: f64,
    /// Mass per unit volume: what gravity and the modal solver use.
    pub density: f64,
    /// A full 6×6 stiffness in Voigt order xx, yy, zz, yz, xz, xy with
    /// engineering shear strains. When set it replaces `youngs` and
    /// `poisson`, which then only describe the material for display.
    pub anisotropic: Option<[[f64; 6]; 6]>,
}

impl Material {
    /// Isotropic, with density 1.
    pub fn new(youngs: f64, poisson: f64) -> Self {
        Self {
            youngs,
            poisson,
            density: 1.0,
            anisotropic: None,
        }
    }

    pub fn density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// A general stiffness, in the Voigt order and convention of
    /// `threers::geometries::lattice::Stiffness::c` — so a homogenised lattice,
    /// scaled by its base material's modulus, goes straight in.
    pub fn anisotropic(c: [[f64; 6]; 6], density: f64) -> Self {
        // An equivalent isotropic pair, for display only: E from the mean
        // axial compliance, ν from the mean off-diagonal ratio.
        let e = (c[0][0] + c[1][1] + c[2][2]) / 3.0;
        let off = (c[0][1] + c[0][2] + c[1][2]) / 3.0;
        let nu = off / (e + off);
        Self {
            youngs: e * (1.0 + nu) * (1.0 - 2.0 * nu) / (1.0 - nu),
            poisson: nu,
            density,
            anisotropic: Some(c),
        }
    }

    /// Structural steel, SI units: 200 GPa, 0.3, 7850 kg/m³.
    pub fn steel() -> Self {
        Self::new(200e9, 0.3).density(7850.0)
    }

    /// 6061 aluminium, SI units: 69 GPa, 0.33, 2700 kg/m³.
    pub fn aluminium() -> Self {
        Self::new(69e9, 0.33).density(2700.0)
    }

    /// Ti-6Al-4V, SI units: 114 GPa, 0.34, 4430 kg/m³.
    pub fn titanium() -> Self {
        Self::new(114e9, 0.34).density(4430.0)
    }

    /// Printed PLA, SI units: 3.5 GPa, 0.36, 1240 kg/m³ — a solid-infill
    /// part loaded in-plane; across layers it is weaker, which this ignores.
    pub fn pla() -> Self {
        Self::new(3.5e9, 0.36).density(1240.0)
    }

    /// Shear modulus `E / 2(1 + ν)`.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs / (2.0 * (1.0 + self.poisson))
    }

    /// The 6×6 Voigt stiffness.
    pub fn voigt(&self) -> [[f64; 6]; 6] {
        if let Some(c) = self.anisotropic {
            return c;
        }
        let (e, nu) = (self.youngs, self.poisson);
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lambda;
            }
            c[i][i] = lambda + 2.0 * mu;
            c[i + 3][i + 3] = mu;
        }
        c
    }

    pub(crate) fn validate(&self) -> Result<(), FemError> {
        if self.anisotropic.is_none()
            && !(self.youngs > 0.0 && self.poisson > -1.0 && self.poisson < 0.5)
        {
            return Err(FemError::InvalidInput(
                "material needs E > 0 and -1 < Poisson's ratio < 0.5",
            ));
        }
        if self.density.is_nan() || self.density < 0.0 {
            return Err(FemError::InvalidInput("density must be non-negative"));
        }
        Ok(())
    }
}

/// Voigt index of tensor component `(i, j)`: xx, yy, zz, yz, xz, xy.
pub(crate) fn voigt_index(i: usize, j: usize) -> usize {
    match (i.min(j), i.max(j)) {
        (0, 0) => 0,
        (1, 1) => 1,
        (2, 2) => 2,
        (1, 2) => 3,
        (0, 2) => 4,
        _ => 5,
    }
}

/// Von Mises equivalent of a Voigt stress.
pub fn von_mises(s: [f64; 6]) -> f64 {
    let [xx, yy, zz, yz, xz, xy] = s;
    (0.5 * ((xx - yy).powi(2) + (yy - zz).powi(2) + (zz - xx).powi(2))
        + 3.0 * (yz * yz + xz * xz + xy * xy))
        .sqrt()
}

/// Principal stresses of a Voigt stress, descending.
pub fn principal_stresses(s: [f64; 6]) -> [f64; 3] {
    let [xx, yy, zz, yz, xz, xy] = s;
    let mut m = crate::dense::Matrix::<f64>::zeros(3, 3);
    let v = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = v[i][j];
        }
    }
    let (w, _) = crate::dense::symmetric_eigen(&m);
    [w[2], w[1], w[0]]
}

enum Load<'a> {
    /// A total force spread evenly over the area of the selected faces.
    Force(Region<'a>, V3),
    /// Force per unit area on the selected faces.
    Traction(Region<'a>, V3),
    /// Normal pressure on the selected faces, positive pushing in.
    Pressure(Region<'a>, f64),
    /// A force at the mesh node nearest a point.
    Point(V3, V3),
}

/// A linear elastic analysis of a tetrahedral mesh.
///
/// ```
/// use threers_fem::{Elastic, Material, TetMesh};
///
/// // A 10 × 1 × 1 bar, held at x = 0 and pulled along x by 1000 at x = 10.
/// let mesh = TetMesh::box_grid([0.0; 3], [10.0, 1.0, 1.0], [10, 2, 2]);
/// let s = Elastic::new(&mesh, Material::new(1e6, 0.0))
///     .fix(|p| p[0] < 1e-9)
///     .force(|p| p[0] > 10.0 - 1e-9, [1000.0, 0.0, 0.0])
///     .solve()
///     .unwrap();
/// // σ = F/A = 1000, δ = FL/(AE) = 0.01.
/// assert!((s.max_von_mises() - 1000.0).abs() < 1e-6);
/// assert!((s.max_displacement() - 0.01).abs() < 1e-9);
/// ```
pub struct Elastic<'a> {
    mesh: &'a TetMesh,
    material: Material,
    order: Order,
    solver: LinearSolver,
    supports: Vec<(Region<'a>, [Option<f64>; 3])>,
    loads: Vec<Load<'a>>,
    body_force: V3,
}

/// What [`Elastic`] assembles: the space, the full stiffness, the load and
/// the prescribed displacements. The modal solver reuses it.
pub(crate) struct ElasticSystem {
    pub space: Space,
    pub k: Csr,
    pub f: Vec<f64>,
    pub fixed: Vec<Option<f64>>,
}

impl<'a> Elastic<'a> {
    pub fn new(mesh: &'a TetMesh, material: Material) -> Self {
        Self {
            mesh,
            material,
            order: Order::Quadratic,
            solver: LinearSolver::Auto,
            supports: Vec::new(),
            loads: Vec::new(),
            body_force: [0.0; 3],
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

    /// Clamp the nodes where `region` holds: no displacement in any
    /// direction.
    pub fn fix(self, region: impl Fn(V3) -> bool + Send + Sync + 'a) -> Self {
        self.displace(region, [Some(0.0); 3])
    }

    /// Prescribe some displacement components where `region` holds, leaving
    /// the `None` ones free — `[Some(0.0), None, None]` is a roller that
    /// holds x and lets y and z slide, which is how a symmetry plane is
    /// modelled.
    pub fn displace(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, d: [Option<f64>; 3]) -> Self {
        self.supports.push((Box::new(region), d));
        self
    }

    /// A total force `f`, spread evenly by area over the boundary faces in
    /// `region`.
    pub fn force(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, f: V3) -> Self {
        self.loads.push(Load::Force(Box::new(region), f));
        self
    }

    /// Force per unit area on the boundary faces in `region`.
    pub fn traction(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, t: V3) -> Self {
        self.loads.push(Load::Traction(Box::new(region), t));
        self
    }

    /// Pressure on the boundary faces in `region`, acting along the inward
    /// normal: positive pushes in, as a fluid does.
    pub fn pressure(mut self, region: impl Fn(V3) -> bool + Send + Sync + 'a, p: f64) -> Self {
        self.loads.push(Load::Pressure(Box::new(region), p));
        self
    }

    /// A force at the mesh node nearest `at`. A point load on a continuum
    /// has infinite stress under it; read stresses away from it.
    pub fn point_force(mut self, at: V3, f: V3) -> Self {
        self.loads.push(Load::Point(at, f));
        self
    }

    /// Gravity: a body force of `density · g` per unit volume.
    pub fn gravity(mut self, g: V3) -> Self {
        self.body_force = add(self.body_force, scale(g, self.material.density));
        self
    }

    /// A body force per unit volume.
    pub fn body_force(mut self, f: V3) -> Self {
        self.body_force = add(self.body_force, f);
        self
    }

    pub(crate) fn material(&self) -> &Material {
        &self.material
    }

    pub(crate) fn order_(&self) -> Order {
        self.order
    }

    pub(crate) fn mesh(&self) -> &TetMesh {
        self.mesh
    }

    pub(crate) fn solver_(&self) -> LinearSolver {
        self.solver
    }

    pub(crate) fn assemble(&self) -> Result<ElasticSystem, FemError> {
        self.material.validate()?;
        if self.mesh.tets.is_empty() {
            return Err(FemError::InvalidInput("mesh has no tetrahedra"));
        }
        let space = Space::new(self.mesh, self.order);
        let r = self.order.reference();
        let npe = self.order.nodes();
        let ndof = 3 * space.nodes.len();

        // The fourth-order tensor C_imjn from the Voigt matrix.
        let d = self.material.voigt();
        let mut c = [[[[0.0; 3]; 3]; 3]; 3];
        for i in 0..3 {
            for m in 0..3 {
                for j in 0..3 {
                    for n in 0..3 {
                        c[i][m][j][n] = d[voigt_index(i, m)][voigt_index(j, n)];
                    }
                }
            }
        }

        let parts = per_chunk(space.elem_count(), |range| {
            let mut t = Triplets::with_capacity(ndof, range.len() * npe * npe * 9);
            let mut gp = Vec::new();
            for e in range {
                gradient_products(r, &space.gradients[e], space.volumes[e], &mut gp);
                let nodes = space.elem(e);
                for a in 0..npe {
                    for b in 0..npe {
                        let g = &gp[(a * npe + b) * 9..(a * npe + b + 1) * 9];
                        for i in 0..3 {
                            for j in 0..3 {
                                let mut s = 0.0;
                                for m in 0..3 {
                                    for n in 0..3 {
                                        s += c[i][m][j][n] * g[m * 3 + n];
                                    }
                                }
                                t.add(3 * nodes[a] as usize + i, 3 * nodes[b] as usize + j, s);
                            }
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
        let k = trip.to_csr();

        let mut f = vec![0.0; ndof];
        // Body force.
        if self.body_force != [0.0; 3] {
            for e in 0..space.elem_count() {
                for (a, &n) in space.elem(e).iter().enumerate() {
                    let w = space.volumes[e] * r.load[a];
                    for i in 0..3 {
                        f[3 * n as usize + i] += w * self.body_force[i];
                    }
                }
            }
        }
        // Surface loads.
        let add_traction = |f: &mut Vec<f64>, faces: &[usize], t: &dyn Fn(usize) -> V3| {
            for &fi in faces {
                let (_, area) = space.face_normal_area(fi);
                let tv = t(fi);
                for (a, &n) in space.face(fi).iter().enumerate() {
                    let w = area * r.face_load[a];
                    for i in 0..3 {
                        f[3 * n as usize + i] += w * tv[i];
                    }
                }
            }
        };
        for load in &self.loads {
            match load {
                Load::Force(region, force) => {
                    let faces = space.faces_where(region);
                    let area: f64 = faces.iter().map(|&fi| space.face_normal_area(fi).1).sum();
                    if faces.is_empty() || area <= 0.0 {
                        return Err(FemError::EmptyRegion("force"));
                    }
                    let t = scale(*force, 1.0 / area);
                    add_traction(&mut f, &faces, &|_| t);
                }
                Load::Traction(region, t) => {
                    let faces = space.faces_where(region);
                    if faces.is_empty() {
                        return Err(FemError::EmptyRegion("traction"));
                    }
                    add_traction(&mut f, &faces, &|_| *t);
                }
                Load::Pressure(region, p) => {
                    let faces = space.faces_where(region);
                    if faces.is_empty() {
                        return Err(FemError::EmptyRegion("pressure"));
                    }
                    add_traction(&mut f, &faces, &|fi| scale(space.face_normal_area(fi).0, -*p));
                }
                Load::Point(at, force) => {
                    let n = nearest(&space.nodes[..space.corner_count], *at);
                    for i in 0..3 {
                        f[3 * n + i] += force[i];
                    }
                }
            }
        }

        let mut fixed = vec![None; ndof];
        for (region, d) in &self.supports {
            let nodes = space.nodes_where(region);
            if nodes.is_empty() {
                return Err(FemError::EmptyRegion("support"));
            }
            for n in nodes {
                for i in 0..3 {
                    if let Some(v) = d[i] {
                        fixed[3 * n + i] = Some(v);
                    }
                }
            }
        }
        Ok(ElasticSystem { space, k, f, fixed })
    }

    /// Solve for displacements and stresses.
    ///
    /// The supports must stop every rigid motion — three translations and
    /// three rotations — or the stiffness is singular and this returns
    /// [`FemError::NotPositiveDefinite`] (direct) or fails to converge
    /// (iterative).
    pub fn solve(&self) -> Result<ElasticSolution, FemError> {
        let sys = self.assemble()?;
        if sys.fixed.iter().all(|f| f.is_none()) {
            return Err(FemError::InvalidInput(
                "no supports: a free body has no unique static solution",
            ));
        }
        let (u, reaction) = solve_constrained(&sys.k, &sys.f, &sys.fixed, &sys.space.dof_coords(3), self.solver)?;
        let space = &sys.space;
        let r = self.order.reference();
        let npe = self.order.nodes();
        let d = self.material.voigt();

        let stress = space.average_to_corners::<6>(|e, c| {
            let g = &space.gradients[e];
            let nodes = space.elem(e);
            let mut grad_u = [[0.0; 3]; 3];
            for a in 0..npe {
                let mut dn = [0.0; 3];
                for k in 0..4 {
                    let w = r.grad_at_corner[(c * npe + a) * 4 + k];
                    if w != 0.0 {
                        dn = add(dn, scale(g[k], w));
                    }
                }
                let n = nodes[a] as usize;
                for i in 0..3 {
                    for m in 0..3 {
                        grad_u[i][m] += u[3 * n + i] * dn[m];
                    }
                }
            }
            let eps = [
                grad_u[0][0],
                grad_u[1][1],
                grad_u[2][2],
                grad_u[1][2] + grad_u[2][1],
                grad_u[0][2] + grad_u[2][0],
                grad_u[0][1] + grad_u[1][0],
            ];
            let mut s = [0.0; 6];
            for p in 0..6 {
                for q in 0..6 {
                    s[p] += d[p][q] * eps[q];
                }
            }
            s
        });
        let mut total_reaction = [0.0; 3];
        for (i, r) in reaction.iter().enumerate() {
            if sys.fixed[i].is_some() {
                total_reaction[i % 3] += r;
            }
        }
        let ku = sys.k.mul_vec(&u);
        let energy = 0.5 * u.iter().zip(&ku).map(|(a, b)| a * b).sum::<f64>();
        let work = u.iter().zip(&sys.f).map(|(a, b)| a * b).sum::<f64>();
        Ok(ElasticSolution {
            nodes: self.mesh.nodes.clone(),
            displacement: (0..space.corner_count)
                .map(|n| [u[3 * n], u[3 * n + 1], u[3 * n + 2]])
                .collect(),
            von_mises: stress.iter().map(|s| von_mises(*s)).collect(),
            stress,
            reaction: total_reaction,
            strain_energy: energy,
            compliance: work,
        })
    }

    /// The natural frequencies and mode shapes of the part, lowest first,
    /// under this problem's supports (prescribed displacements are taken as
    /// zero; loads are ignored). With no supports the first six modes are
    /// the rigid motions, at zero frequency.
    pub fn modes(&self, count: usize) -> Result<super::Modes, FemError> {
        super::modal::solve(self, count)
    }
}

fn nearest(nodes: &[V3], p: V3) -> usize {
    nodes
        .iter()
        .enumerate()
        .min_by(|a, b| norm2(sub(*a.1, p)).total_cmp(&norm2(sub(*b.1, p))))
        .map(|(i, _)| i)
        .unwrap()
}

/// The answer to an [`Elastic`] analysis, per mesh node.
#[derive(Clone, Debug)]
pub struct ElasticSolution {
    /// The mesh nodes, undeformed.
    pub nodes: Vec<V3>,
    /// Displacement of each mesh node.
    pub displacement: Vec<V3>,
    /// Stress at each mesh node, Voigt xx, yy, zz, yz, xz, xy — the
    /// volume-weighted average of the elements meeting there.
    pub stress: Vec<[f64; 6]>,
    /// Von Mises equivalent stress at each node.
    pub von_mises: Vec<f64>,
    /// Total force the supports exert on the part. Balances the applied
    /// loads.
    pub reaction: V3,
    /// `½ uᵀ K u`.
    pub strain_energy: f64,
    /// Work of the applied loads, `fᵀu` — the compliance a topology
    /// optimiser minimises. Twice the strain energy when all supports are
    /// fixed at zero.
    pub compliance: f64,
}

impl ElasticSolution {
    pub fn max_von_mises(&self) -> f64 {
        self.von_mises.iter().copied().fold(0.0, f64::max)
    }

    pub fn max_displacement(&self) -> f64 {
        self.displacement.iter().map(|d| norm(*d)).fold(0.0, f64::max)
    }

    /// Displacement magnitude per node.
    pub fn displacement_magnitude(&self) -> Vec<f64> {
        self.displacement.iter().map(|d| norm(*d)).collect()
    }

    /// The nodes moved by `scale` times their displacement — 1 for the true
    /// shape, more to make a small deflection visible.
    pub fn deformed_nodes(&self, scale: f64) -> Vec<V3> {
        self.nodes
            .iter()
            .zip(&self.displacement)
            .map(|(p, d)| add(*p, crate::vec3::scale(*d, scale)))
            .collect()
    }

    /// Safety factor against yield by von Mises: `yield / max σ_vm`.
    pub fn safety_factor(&self, yield_strength: f64) -> f64 {
        yield_strength / self.max_von_mises().max(1e-300)
    }
}
