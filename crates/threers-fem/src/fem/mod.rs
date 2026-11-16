//! The finite element method on tetrahedra.
//!
//! [`Elastic`] solves linear elasticity — displacements, stresses, reactions —
//! and its [`modes`](Elastic::modes) are the natural frequencies of the same
//! part. [`Heat`] solves steady conduction with fixed temperatures, fluxes,
//! convection and internal sources.
//!
//! # Elements
//!
//! [`Order::Quadratic`] (ten-node tetrahedra, the default) is what to use for
//! anything that bends: four-node tetrahedra have constant strain, so a beam
//! of them is far too stiff unless very finely meshed — twice too stiff on
//! the cantilever in the tests at two elements through the depth, where the
//! quadratic one is within 2 %. [`Order::Linear`] is cheaper and is fine for
//! conduction, and for a quick look.
//!
//! Elements are straight-sided: the midside nodes sit at edge midpoints. All
//! element integrals — stiffness, mass, loads — are then polynomials in
//! barycentric coordinates and are computed exactly, with no quadrature
//! rule to get wrong.
//!
//! # Choosing where loads and supports go
//!
//! Supports and loads are placed by predicates on position, `|p| p[0] < 1e-9`,
//! evaluated at solve time. A node is selected when its position satisfies
//! the predicate; a boundary face when its three corners do. The midside
//! node of a boundary edge goes with its corners, since on a curved surface
//! it lies on the chord, just inside the surface the predicate describes. Meshes
//! from [`TetMesh::from_sdf`](crate::TetMesh::from_sdf) put boundary nodes
//! exactly on the surface, so a predicate on a face's plane selects it — with
//! a tolerance of a millionth of the mesh spacing to allow for the root
//! finder.
//!
//! # Units
//!
//! Any consistent set. In SI — metres, pascals, kilograms per cubic metre,
//! newtons — frequencies come out in hertz and stresses in pascals; in
//! millimetres, megapascals and tonnes per cubic millimetre they do too, and
//! stresses come out in megapascals.

pub mod elastic;
pub mod heat;
pub mod modal;
pub mod voxel;

pub use elastic::{Elastic, ElasticSolution, Material};
pub use heat::{Heat, HeatSolution};
pub use modal::{Mode, Modes};

use crate::mesh::{TetMesh, TET_FACES};
use crate::vec3::*;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Polynomial order of the elements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Order {
    /// Four-node tetrahedra: linear fields, constant gradients.
    Linear,
    /// Ten-node tetrahedra: quadratic fields, linear gradients.
    #[default]
    Quadratic,
}

impl Order {
    /// Nodes per element.
    pub fn nodes(self) -> usize {
        match self {
            Self::Linear => 4,
            Self::Quadratic => 10,
        }
    }

    /// Nodes per face.
    pub fn face_nodes(self) -> usize {
        match self {
            Self::Linear => 3,
            Self::Quadratic => 6,
        }
    }

    pub(crate) fn reference(self) -> &'static Reference {
        static LIN: OnceLock<Reference> = OnceLock::new();
        static QUAD: OnceLock<Reference> = OnceLock::new();
        match self {
            Self::Linear => LIN.get_or_init(|| Reference::new(self)),
            Self::Quadratic => QUAD.get_or_init(|| Reference::new(self)),
        }
    }
}

/// Midside node of a ten-node element per corner pair, in the VTK order.
pub(crate) const TET10_EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)];

fn tet10_edge(a: usize, b: usize) -> usize {
    4 + TET10_EDGES
        .iter()
        .position(|&(p, q)| (p, q) == (a, b) || (p, q) == (b, a))
        .unwrap()
}

/// A polynomial in the four barycentric coordinates.
#[derive(Clone, Debug, Default)]
struct Poly(Vec<([u8; 4], f64)>);

impl Poly {
    fn term(e: [u8; 4], c: f64) -> Self {
        Self(vec![(e, c)])
    }

    fn mul(&self, o: &Poly) -> Poly {
        let mut out = Vec::with_capacity(self.0.len() * o.0.len());
        for (ea, ca) in &self.0 {
            for (eb, cb) in &o.0 {
                out.push(([ea[0] + eb[0], ea[1] + eb[1], ea[2] + eb[2], ea[3] + eb[3]], ca * cb));
            }
        }
        Poly(out)
    }

    fn d(&self, k: usize) -> Poly {
        Poly(
            self.0
                .iter()
                .filter(|(e, _)| e[k] > 0)
                .map(|(e, c)| {
                    let mut e2 = *e;
                    e2[k] -= 1;
                    (e2, c * e[k] as f64)
                })
                .collect(),
        )
    }

    /// `∫ p dV / V` over any tetrahedron: `3!·Πeᵢ! / (Σeᵢ + 3)!` per monomial.
    fn integrate_tet(&self) -> f64 {
        self.0
            .iter()
            .map(|(e, c)| {
                let num: f64 = e.iter().map(|&k| fact(k as usize)).product();
                let s: usize = e.iter().map(|&k| k as usize).sum();
                c * 6.0 * num / fact(s + 3)
            })
            .sum()
    }

    /// `∫ p dA / A` over a triangle, with `e[3]` required to be zero.
    fn integrate_tri(&self) -> f64 {
        self.0
            .iter()
            .map(|(e, c)| {
                assert_eq!(e[3], 0);
                let num: f64 = e[..3].iter().map(|&k| fact(k as usize)).product();
                let s: usize = e[..3].iter().map(|&k| k as usize).sum();
                c * 2.0 * num / fact(s + 2)
            })
            .sum()
    }

    fn eval(&self, l: [f64; 4]) -> f64 {
        self.0
            .iter()
            .map(|(e, c)| c * (0..4).map(|k| l[k].powi(e[k] as i32)).product::<f64>())
            .sum()
    }
}

fn fact(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

fn shape_functions(order: Order) -> Vec<Poly> {
    let l = |k: usize| {
        let mut e = [0u8; 4];
        e[k] = 1;
        Poly::term(e, 1.0)
    };
    match order {
        Order::Linear => (0..4).map(l).collect(),
        Order::Quadratic => {
            let mut n: Vec<Poly> = (0..4)
                .map(|k| {
                    // L(2L − 1)
                    let mut e2 = [0u8; 4];
                    e2[k] = 2;
                    let mut e1 = [0u8; 4];
                    e1[k] = 1;
                    Poly(vec![(e2, 2.0), (e1, -1.0)])
                })
                .collect();
            for (a, b) in TET10_EDGES {
                n.push(l(a).mul(&l(b)).mul(&Poly::term([0; 4], 4.0)));
            }
            n
        }
    }
}

/// Element integrals on the reference tetrahedron, per unit volume. Every
/// straight-sided element's integrals are these times its volume and its
/// barycentric gradients.
pub(crate) struct Reference {
    pub npe: usize,
    /// `∫ Nₐ N_b / V`.
    pub mass: Vec<f64>,
    /// `∫ ∂Nₐ/∂L_k · ∂N_b/∂L_l / V`, indexed `[(a·4 + k)·npe·4 + b·4 + l]`.
    pub grad: Vec<f64>,
    /// `∫ Nₐ / V`.
    pub load: Vec<f64>,
    /// On a face: `∫ Nₐ / A` for the face's nodes (corners, then midsides).
    pub face_load: Vec<f64>,
    /// On a face: `∫ Nₐ N_b / A`.
    pub face_mass: Vec<f64>,
    /// `∂Nₐ/∂L_k` at each corner `c`, indexed `[(c·npe + a)·4 + k]` — for
    /// recovering stresses and fluxes at the nodes.
    pub grad_at_corner: Vec<f64>,
}

impl Reference {
    fn new(order: Order) -> Self {
        let n = shape_functions(order);
        let npe = n.len();
        let dn: Vec<Vec<Poly>> = n.iter().map(|p| (0..4).map(|k| p.d(k)).collect()).collect();
        let mut mass = vec![0.0; npe * npe];
        let mut grad = vec![0.0; npe * 4 * npe * 4];
        for a in 0..npe {
            for b in 0..npe {
                mass[a * npe + b] = n[a].mul(&n[b]).integrate_tet();
                for k in 0..4 {
                    for l in 0..4 {
                        grad[(a * 4 + k) * npe * 4 + b * 4 + l] = dn[a][k].mul(&dn[b][l]).integrate_tet();
                    }
                }
            }
        }
        let load = n.iter().map(|p| p.integrate_tet()).collect();
        // A face's shape functions are the element's restricted to L3 = 0,
        // with face node order corners 0,1,2 then midsides 01, 12, 20.
        let face: Vec<Poly> = match order {
            Order::Linear => vec![n[0].clone(), n[1].clone(), n[2].clone()],
            Order::Quadratic => vec![
                n[0].clone(),
                n[1].clone(),
                n[2].clone(),
                n[tet10_edge(0, 1)].clone(),
                n[tet10_edge(1, 2)].clone(),
                n[tet10_edge(2, 0)].clone(),
            ],
        };
        let nf = face.len();
        let face_load = face.iter().map(|p| p.integrate_tri()).collect();
        let mut face_mass = vec![0.0; nf * nf];
        for a in 0..nf {
            for b in 0..nf {
                face_mass[a * nf + b] = face[a].mul(&face[b]).integrate_tri();
            }
        }
        let mut grad_at_corner = vec![0.0; 4 * npe * 4];
        for c in 0..4 {
            let mut l = [0.0; 4];
            l[c] = 1.0;
            for a in 0..npe {
                for k in 0..4 {
                    grad_at_corner[(c * npe + a) * 4 + k] = dn[a][k].eval(l);
                }
            }
        }
        Self {
            npe,
            mass,
            grad,
            load,
            face_load,
            face_mass,
            grad_at_corner,
        }
    }
}

/// The barycentric gradients `∇L_k` of a tetrahedron and its volume.
pub(crate) fn barycentric_gradients(p: [V3; 4]) -> ([V3; 4], f64) {
    let e1 = sub(p[1], p[0]);
    let e2 = sub(p[2], p[0]);
    let e3 = sub(p[3], p[0]);
    let det = dot(e1, cross(e2, e3));
    // Rows of the inverse Jacobian are the gradients of L1, L2, L3.
    let g1 = scale(cross(e2, e3), 1.0 / det);
    let g2 = scale(cross(e3, e1), 1.0 / det);
    let g3 = scale(cross(e1, e2), 1.0 / det);
    let g0 = scale(add(add(g1, g2), g3), -1.0);
    ([g0, g1, g2, g3], det / 6.0)
}

/// `∫ ∂Nₐ/∂xᵢ ∂N_b/∂xⱼ dV` for every `a, b, i, j` of one element, indexed
/// `[((a·npe + b)·3 + i)·3 + j]`.
pub(crate) fn gradient_products(r: &Reference, g: &[V3; 4], vol: f64, out: &mut Vec<f64>) {
    let npe = r.npe;
    out.clear();
    out.resize(npe * npe * 9, 0.0);
    // Precompute g_k[i]·g_l[j] for all k, l, i, j.
    let mut gg = [[[[0.0; 3]; 3]; 4]; 4];
    for k in 0..4 {
        for l in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    gg[k][l][i][j] = g[k][i] * g[l][j];
                }
            }
        }
    }
    for a in 0..npe {
        for b in 0..npe {
            let o = (a * npe + b) * 9;
            for k in 0..4 {
                for l in 0..4 {
                    let w = r.grad[(a * 4 + k) * npe * 4 + b * 4 + l];
                    if w == 0.0 {
                        continue;
                    }
                    let w = w * vol;
                    for i in 0..3 {
                        for j in 0..3 {
                            out[o + i * 3 + j] += w * gg[k][l][i][j];
                        }
                    }
                }
            }
        }
    }
}

/// The nodes and element connectivity of a finite element space on a mesh:
/// the mesh's own nodes first, then (for quadratic elements) one per edge.
pub(crate) struct Space {
    pub order: Order,
    pub nodes: Vec<V3>,
    /// Mesh nodes; `nodes[..corner_count]` are the mesh's.
    pub corner_count: usize,
    /// Element nodes, `order.nodes()` per element.
    pub elems: Vec<u32>,
    /// Boundary faces: `order.face_nodes()` per face, corners then
    /// midsides, wound outward.
    pub faces: Vec<u32>,
    /// Per element: volume.
    pub volumes: Vec<f64>,
    /// Per element: barycentric gradients.
    pub gradients: Vec<[V3; 4]>,
    /// For each node past `corner_count`, the edge it is the midpoint of.
    pub midside: Vec<(u32, u32)>,
    /// Whether each node lies on a boundary face.
    pub on_boundary: Vec<bool>,
}

impl Space {
    pub fn new(mesh: &TetMesh, order: Order) -> Self {
        let npe = order.nodes();
        let mut nodes = mesh.nodes.clone();
        let mut elems = Vec::with_capacity(mesh.tets.len() * npe);
        let mut edge_node: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midside = Vec::new();
        let mut volumes = Vec::with_capacity(mesh.tets.len());
        let mut gradients = Vec::with_capacity(mesh.tets.len());
        for (t, tet) in mesh.tets.iter().enumerate() {
            elems.extend_from_slice(tet);
            if order == Order::Quadratic {
                for (a, b) in TET10_EDGES {
                    let (p, q) = (tet[a], tet[b]);
                    let key = (p.min(q), p.max(q));
                    let id = *edge_node.entry(key).or_insert_with(|| {
                        nodes.push(lerp(mesh.nodes[p as usize], mesh.nodes[q as usize], 0.5));
                        midside.push(key);
                        (nodes.len() - 1) as u32
                    });
                    elems.push(id);
                }
            }
            let (g, v) = barycentric_gradients(mesh.corners(t));
            gradients.push(g);
            volumes.push(v);
        }
        let nf = order.face_nodes();
        let mut faces = Vec::new();
        for f in mesh.boundary_faces() {
            let e = &elems[f.tet as usize * npe..(f.tet as usize + 1) * npe];
            let local = TET_FACES[f.local as usize];
            faces.extend(local.iter().map(|&c| e[c]));
            if order == Order::Quadratic {
                for k in 0..3 {
                    faces.push(e[tet10_edge(local[k], local[(k + 1) % 3])]);
                }
            }
            debug_assert_eq!(faces.len() % nf, 0);
        }
        let mut on_boundary = vec![false; nodes.len()];
        for &n in &faces {
            on_boundary[n as usize] = true;
        }
        Self {
            order,
            nodes,
            corner_count: mesh.nodes.len(),
            elems,
            faces,
            volumes,
            gradients,
            midside,
            on_boundary,
        }
    }

    pub fn elem(&self, e: usize) -> &[u32] {
        let n = self.order.nodes();
        &self.elems[e * n..(e + 1) * n]
    }

    pub fn elem_count(&self) -> usize {
        self.volumes.len()
    }

    pub fn face(&self, f: usize) -> &[u32] {
        let n = self.order.face_nodes();
        &self.faces[f * n..(f + 1) * n]
    }

    pub fn face_count(&self) -> usize {
        self.faces.len() / self.order.face_nodes()
    }

    /// Outward unit normal and area of boundary face `f`.
    pub fn face_normal_area(&self, f: usize) -> (V3, f64) {
        let c = self.face(f);
        let [a, b, d] = [c[0], c[1], c[2]].map(|i| self.nodes[i as usize]);
        let n = cross(sub(b, a), sub(d, a));
        let l = norm(n);
        (scale(n, 1.0 / l), 0.5 * l)
    }

    /// Boundary faces whose three corners all satisfy `pred`.
    pub fn faces_where(&self, pred: &dyn Fn(V3) -> bool) -> Vec<usize> {
        (0..self.face_count())
            .filter(|&f| self.face(f)[..3].iter().all(|&n| pred(self.nodes[n as usize])))
            .collect()
    }

    /// Nodes that satisfy `pred` — and boundary midside nodes whose edge's
    /// two corners do. A midside node sits on the chord, so on a curved
    /// surface it falls just inside: `|p| r(p) > R − ε` misses it, and a
    /// face held at its corners but free at its midsides is barely held.
    pub fn nodes_where(&self, pred: &dyn Fn(V3) -> bool) -> Vec<usize> {
        let corner: Vec<bool> = self.nodes[..self.corner_count].iter().map(|p| pred(*p)).collect();
        let mut out: Vec<usize> = (0..self.corner_count).filter(|&n| corner[n]).collect();
        for (k, &(a, b)) in self.midside.iter().enumerate() {
            let n = self.corner_count + k;
            let via_edge = self.on_boundary[n] && corner[a as usize] && corner[b as usize];
            if via_edge || pred(self.nodes[n]) {
                out.push(n);
            }
        }
        out
    }

    /// Average per-element-corner values onto the mesh nodes, weighted by
    /// element volume. `value(e, c)` is element `e`'s value at its corner `c`.
    pub fn average_to_corners<const N: usize>(&self, value: impl Fn(usize, usize) -> [f64; N]) -> Vec<[f64; N]> {
        let mut sum = vec![[0.0; N]; self.corner_count];
        let mut w = vec![0.0; self.corner_count];
        for e in 0..self.elem_count() {
            let v = self.volumes[e];
            for c in 0..4 {
                let n = self.elem(e)[c] as usize;
                let x = value(e, c);
                for k in 0..N {
                    sum[n][k] += v * x[k];
                }
                w[n] += v;
            }
        }
        for (s, w) in sum.iter_mut().zip(&w) {
            if *w > 0.0 {
                for v in s.iter_mut() {
                    *v /= w;
                }
            }
        }
        sum
    }
}

/// Solve `K u = f` with some entries of `u` prescribed: eliminate them,
/// solve for the rest, and return `u` with the residual `K u − f`, which at
/// the prescribed entries is the reaction the supports supply.
pub(crate) fn solve_constrained(
    k: &crate::sparse::Csr,
    f: &[f64],
    fixed: &[Option<f64>],
    coords: &[V3],
    how: crate::sparse::LinearSolver,
) -> Result<(Vec<f64>, Vec<f64>), crate::FemError> {
    let n = f.len();
    let mut u: Vec<f64> = fixed.iter().map(|v| v.unwrap_or(0.0)).collect();
    let free: Vec<bool> = fixed.iter().map(|v| v.is_none()).collect();
    if free.iter().any(|f| *f) {
        // rhs = f − K u_prescribed, on the free rows.
        let ku = k.mul_vec(&u);
        let (kff, map) = k.submatrix(&free);
        let mut rhs = vec![0.0; kff.n];
        for i in 0..n {
            if map[i] != u32::MAX {
                rhs[map[i] as usize] = f[i] - ku[i];
            }
        }
        let free_coords: Vec<V3> = (0..n).filter(|&i| map[i] != u32::MAX).map(|i| coords[i]).collect();
        let x = crate::sparse::SpdFactor::with_coords(kff, &free_coords, how, 1.0)?.solve(&rhs)?;
        for i in 0..n {
            if map[i] != u32::MAX {
                u[i] = x[map[i] as usize];
            }
        }
    }
    let mut r = k.mul_vec(&u);
    for i in 0..n {
        r[i] -= f[i];
    }
    Ok((u, r))
}

impl Space {
    /// The position of each of `per_node` unknowns per node, in order.
    pub fn dof_coords(&self, per_node: usize) -> Vec<V3> {
        self.nodes.iter().flat_map(|p| std::iter::repeat_n(*p, per_node)).collect()
    }
}

/// A boxed position predicate, as the problem builders store them.
pub(crate) type Region<'a> = Box<dyn Fn(V3) -> bool + Send + Sync + 'a>;

/// Run `f` over elements in chunks — across threads with `parallel` — and
/// collect what each chunk produced.
pub(crate) fn per_chunk<R: Send>(count: usize, f: impl Fn(std::ops::Range<usize>) -> R + Sync) -> Vec<R> {
    let chunk = 2048;
    let ranges: Vec<_> = (0..count.div_ceil(chunk))
        .map(|i| i * chunk..((i + 1) * chunk).min(count))
        .collect();
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        return ranges.into_par_iter().map(&f).collect();
    }
    #[allow(unreachable_code)]
    ranges.into_iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_integrals_are_partitions_of_unity() {
        for order in [Order::Linear, Order::Quadratic] {
            let r = order.reference();
            // Σ ∫N = V, Σ∫N N = V.
            let sum: f64 = r.load.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14);
            let msum: f64 = r.mass.iter().sum();
            assert!((msum - 1.0).abs() < 1e-14);
            let fsum: f64 = r.face_load.iter().sum();
            assert!((fsum - 1.0).abs() < 1e-14);
        }
        // Tet10: corner loads are −1/20, midsides 1/5.
        let r = Order::Quadratic.reference();
        assert!((r.load[0] + 0.05).abs() < 1e-14);
        assert!((r.load[4] - 0.2).abs() < 1e-14);
        // And on a face, corners carry nothing, midsides a third each.
        assert!(r.face_load[0].abs() < 1e-14);
        assert!((r.face_load[3] - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn gradient_products_reproduce_a_linear_field() {
        // For u = x, Σ_b K_ab u_b = ∫∇N_a·∇u = ∫∂N_a/∂x, and Σ_a of it is 0.
        let mesh = TetMesh::box_grid([0.0; 3], [1.0; 3], [1, 1, 1]);
        for order in [Order::Linear, Order::Quadratic] {
            let s = Space::new(&mesh, order);
            let r = order.reference();
            let mut gp = Vec::new();
            let npe = order.nodes();
            for e in 0..s.elem_count() {
                gradient_products(r, &s.gradients[e], s.volumes[e], &mut gp);
                let mut total = 0.0;
                for a in 0..npe {
                    for b in 0..npe {
                        let x = s.nodes[s.elem(e)[b] as usize][0];
                        total += gp[((a * npe + b) * 3) * 3] * x;
                    }
                }
                assert!(total.abs() < 1e-12);
                // ∫ |∇x|² = V.
                let mut energy = 0.0;
                for a in 0..npe {
                    for b in 0..npe {
                        let xa = s.nodes[s.elem(e)[a] as usize][0];
                        let xb = s.nodes[s.elem(e)[b] as usize][0];
                        energy += xa * gp[((a * npe + b) * 3) * 3] * xb;
                    }
                }
                assert!((energy - s.volumes[e]).abs() < 1e-12, "{energy} vs {}", s.volumes[e]);
            }
        }
    }
}
