//! Finite elements on voxels: a closed mesh filled with cubes, solved for
//! how it deforms and where it is stressed.
//!
//! Beside the tetrahedral [`Elastic`](super::Elastic), which needs a mesh,
//! this takes any closed triangle mesh as it is — indexed or not, welded or
//! not — and fills it with cubes. Each is an eight-node brick. A plain brick
//! is too stiff in bending when only a few span a part's thickness (it locks
//! in shear), so each carries Wilson's nine incompatible modes, condensed
//! out: a cantilever four cubes thick bends within 5 % of beam theory. All
//! the cubes are alike, so one 24 × 24 stiffness serves every element and
//! the solve needs no assembled matrix — conjugate gradients preconditioned
//! by the diagonal, the products split among threads under the `parallel`
//! feature. Stresses come from each cube's corners, averaged where cubes
//! meet.
//!
//! The price of cubes is the stepped surface: a single node's stress where
//! the steps make a notch is the grid's, not the part's, so read peaks over
//! a small share of the most stressed nodes rather than the one most.
//!
//! Supports hold a node in any of the three directions — all three for a
//! clamp, one to keep it on a symmetry plane. The cubes sit on the lattice of
//! multiples of the cube size, so a layer of nodes lies on every coordinate
//! plane through the origin.
//!
//! Units are any consistent set, as for the rest of the crate.
//!
//! ```
//! use threers_fem::fem::voxel::{self, VoxelBody};
//! use threers_fem::fem::Material;
//!
//! // A bar 40 × 4 × 4, held at one end, pushed down 1 N at the other.
//! let body = VoxelBody::new(&threers::BoxGeometry::new(40.0, 4.0, 4.0), 1.0).unwrap();
//! let x: Vec<f64> = body.nodes.iter().map(|q| q[0]).collect();
//! let (lo, hi) = (x.iter().cloned().fold(f64::MAX, f64::min), x.iter().cloned().fold(f64::MIN, f64::max));
//! let held: Vec<(u32, [bool; 3])> = (0..x.len()).filter(|&n| x[n] < lo + 0.1).map(|n| (n as u32, [true; 3])).collect();
//! let end: Vec<u32> = (0..x.len() as u32).filter(|&n| x[n as usize] > hi - 0.1).collect();
//! let push: Vec<(u32, [f64; 3])> = end.iter().map(|&n| (n, [0.0, -1.0 / end.len() as f64, 0.0])).collect();
//! let s = voxel::solve(&body, &Material::new(2000.0, 0.3), &held, &push).unwrap();
//! // Down by about F L³ / 3 E I.
//! let beam = (hi - lo).powi(3) / (3.0 * 2000.0 * 4f64.powi(4) / 12.0);
//! assert!((s.displacement[end[0] as usize][1] / -beam - 1.0).abs() < 0.1);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use threers::BufferGeometry;

use super::{per_chunk, Material};
use crate::vec3::V3;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[path = "voxel_gpu.rs"]
mod gpu;
use crate::FemError;

/// A grid of cubes `res` across from `origin`, `size` cubes each way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    pub origin: [f32; 3],
    pub res: f32,
    pub size: [usize; 3],
}

impl VoxelGrid {
    /// A grid round everything in `meshes`, a cube to spare each side.
    pub fn around(meshes: &[&BufferGeometry], res: f32) -> VoxelGrid {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for g in meshes {
            if let Some(p) = g.get_attribute("position") {
                for v in p.array.chunks(3) {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
            }
        }
        let origin = lo.map(|v| v - res);
        let size = [0, 1, 2].map(|k| ((hi[k] - lo[k]) / res).ceil().max(0.0) as usize + 3);
        VoxelGrid { origin, res, size }
    }

    /// Which cubes have their middles inside `g`, a closed mesh (indexed or
    /// not, welded or not): each column of cubes filled between the points
    /// where it crosses the surface, in and out in turn. Indexed x fastest,
    /// then y, then z.
    pub fn fill(&self, g: &BufferGeometry) -> Vec<bool> {
        let [nx, ny, nz] = self.size;
        let mut cells = vec![false; nx * ny * nz];
        let Some(position) = g.get_attribute("position") else {
            return cells;
        };
        let pos = &position.array;
        let n = pos.len() as u32 / 3;
        let idx = g.index.clone().unwrap_or_else(|| (0..n).collect());
        let at = |i: u32| {
            let i = i as usize * 3;
            [pos[i], pos[i + 1], pos[i + 2]]
        };
        // The columns run through the cubes' middles, nudged off the grid's
        // lines so that none passes exactly through a vertex or an edge.
        let nudge = self.res * 0.013_7;
        let column = |i: usize, j: usize| {
            [
                self.origin[0] + (i as f32 + 0.5) * self.res + nudge,
                self.origin[1] + (j as f32 + 0.5) * self.res + nudge * 0.61,
            ]
        };
        let mut hits: Vec<Vec<f32>> = vec![vec![]; nx * ny];
        for t in idx.chunks(3) {
            let [a, b, c] = [at(t[0]), at(t[1]), at(t[2])];
            let d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if d.abs() < 1e-12 {
                continue;
            }
            let cell = |v: f32, k: usize| ((v - self.origin[k]) / self.res - 0.5).floor().max(0.0) as usize;
            let (i0, i1) = (cell(a[0].min(b[0]).min(c[0]), 0), cell(a[0].max(b[0]).max(c[0]), 0) + 1);
            let (j0, j1) = (cell(a[1].min(b[1]).min(c[1]), 1), cell(a[1].max(b[1]).max(c[1]), 1) + 1);
            for j in j0..=j1.min(ny - 1) {
                for i in i0..=i1.min(nx - 1) {
                    let [x, y] = column(i, j);
                    // The column's barycentric weights in the triangle's
                    // shadow on the grid's floor.
                    let wb = ((x - a[0]) * (c[1] - a[1]) - (y - a[1]) * (c[0] - a[0])) / d;
                    let wc = ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / d;
                    if wb >= 0.0 && wc >= 0.0 && wb + wc <= 1.0 {
                        hits[j * nx + i].push(a[2] + (b[2] - a[2]) * wb + (c[2] - a[2]) * wc);
                    }
                }
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let h = &mut hits[j * nx + i];
                h.sort_by(f32::total_cmp);
                for pair in h.chunks(2).filter(|p| p.len() == 2) {
                    let k0 = ((pair[0] - self.origin[2]) / self.res - 0.5).ceil().max(0.0) as usize;
                    let k1 = (((pair[1] - self.origin[2]) / self.res - 0.5).floor() as isize).min(nz as isize - 1);
                    for k in k0 as isize..=k1 {
                        cells[(k as usize * ny + j) * nx + i] = true;
                    }
                }
            }
        }
        cells
    }

    /// The volume of the cubes marked.
    pub fn volume(&self, cells: &[bool]) -> f32 {
        cells.iter().filter(|&&c| c).count() as f32 * self.res.powi(3)
    }
}

/// The volume two closed meshes share, on cubes `res` across.
pub fn overlap(a: &BufferGeometry, b: &BufferGeometry, res: f32) -> f32 {
    let grid = VoxelGrid::around(&[a, b], res);
    let (ca, cb) = (grid.fill(a), grid.fill(b));
    ca.iter().zip(&cb).filter(|(x, y)| **x && **y).count() as f32 * res.powi(3)
}

/// A closed mesh as cubes joined at their corners.
pub struct VoxelBody {
    pub res: f32,
    /// Each node's position.
    pub nodes: Vec<V3>,
    /// Each cube's eight nodes, in corner order x + 2y + 4z.
    elements: Vec<[u32; 8]>,
    at: HashMap<[i32; 3], u32>,
    origin: [f32; 3],
}

impl VoxelBody {
    /// `g`, a closed mesh, filled with cubes `res` across. The cubes sit on
    /// the lattice of multiples of `res`, so a layer of nodes lies on every
    /// coordinate plane through the origin (a symmetry plane, say).
    pub fn new(g: &BufferGeometry, res: f32) -> Result<VoxelBody, FemError> {
        // Written so a NaN size fails too.
        if res.is_nan() || res <= 0.0 {
            return Err(FemError::InvalidInput("the cube size must be above zero"));
        }
        let mut grid = VoxelGrid::around(&[g], res);
        for k in 0..3 {
            grid.origin[k] = (grid.origin[k] / res).floor() * res;
            grid.size[k] += 1;
        }
        let cells = grid.fill(g);
        let [nx, ny, _] = grid.size;
        let mut at: HashMap<[i32; 3], u32> = HashMap::new();
        let mut nodes = vec![];
        let mut elements = vec![];
        for (k, &filled) in cells.iter().enumerate() {
            if !filled {
                continue;
            }
            let (i, j, l) = ((k % nx) as i32, ((k / nx) % ny) as i32, (k / (nx * ny)) as i32);
            let mut e = [0u32; 8];
            for (c, slot) in e.iter_mut().enumerate() {
                let key = [i + (c & 1) as i32, j + ((c >> 1) & 1) as i32, l + ((c >> 2) & 1) as i32];
                *slot = *at.entry(key).or_insert_with(|| {
                    nodes.push([0, 1, 2].map(|k| grid.origin[k] as f64 + key[k] as f64 * res as f64));
                    nodes.len() as u32 - 1
                });
            }
            elements.push(e);
        }
        if elements.is_empty() {
            return Err(FemError::InvalidSurface("no cube's middle lies inside the mesh: is it closed, and larger than a cube?"));
        }
        Ok(VoxelBody { res, nodes, elements, at, origin: grid.origin })
    }

    /// How many cubes.
    pub fn elements(&self) -> usize {
        self.elements.len()
    }

    /// Each cube's centre, in the order [`solve_graded`] takes their
    /// stiffnesses.
    pub fn centres(&self) -> Vec<V3> {
        self.elements
            .iter()
            .map(|cube| {
                let (a, b) = (self.nodes[cube[0] as usize], self.nodes[cube[7] as usize]);
                [0, 1, 2].map(|k| (a[k] + b[k]) / 2.0)
            })
            .collect()
    }

    /// The node nearest `q` among those within two cubes of it.
    pub fn nearest(&self, q: V3) -> Option<u32> {
        let base = [0, 1, 2].map(|k| ((q[k] - self.origin[k] as f64) / self.res as f64).round() as i32);
        let mut best: Option<(f64, u32)> = None;
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    if let Some(&n) = self.at.get(&[base[0] + dx, base[1] + dy, base[2] + dz]) {
                        let d = crate::vec3::dist(self.nodes[n as usize], q);
                        if best.is_none_or(|(b, _)| d < b) {
                            best = Some((d, n));
                        }
                    }
                }
            }
        }
        best.map(|(_, n)| n)
    }
}

/// What [`solve`] finds: each node's displacement and von Mises stress.
pub struct VoxelSolution {
    pub displacement: Vec<V3>,
    pub stress: Vec<f64>,
    /// Conjugate-gradient iterations taken.
    pub iterations: usize,
    /// Whether they ran on a GPU ([`solve_gpu`] with a device that did not
    /// fail) rather than here.
    pub gpu: bool,
}

/// Solves `body` of `material` under `forces` at nodes, each node in `held`
/// kept from moving in the directions marked (all three to clamp it; one, to
/// hold it to a symmetry plane). Held too little to stop the body moving as
/// a whole, it does not converge.
pub fn solve(body: &VoxelBody, material: &Material, held: &[(u32, [bool; 3])], forces: &[(u32, V3)]) -> Result<VoxelSolution, FemError> {
    solve_on(body, material, None, held, forces, false)
}

/// [`solve`] with each cube's stiffness scaled by its entry in `stiffness`
/// (one per cube, in the order of [`VoxelBody::centres`], above zero): a part
/// whose inside is a lattice, say, at the lattice's effective stiffness
/// under a solid skin. Stresses are the solid material's, from its strain.
/// On the CPU.
pub fn solve_graded(
    body: &VoxelBody,
    material: &Material,
    stiffness: &[f64],
    held: &[(u32, [bool; 3])],
    forces: &[(u32, V3)],
) -> Result<VoxelSolution, FemError> {
    if stiffness.len() != body.elements.len() {
        return Err(FemError::InvalidInput("a stiffness for each cube"));
    }
    if stiffness.iter().any(|k| !(k.is_finite() && *k > 0.0)) {
        return Err(FemError::InvalidInput("each cube's stiffness above zero"));
    }
    solve_on(body, material, Some(stiffness), held, forces, false)
}

/// [`solve`] with the iteration on a GPU through wgpu, many times faster on
/// a large body: the whole conjugate gradient stays on the device and only
/// the convergence test comes back. In single precision, so to a residual a
/// little looser than the CPU's (see `voxel_gpu.rs`). Without the `gpu`
/// feature or an adapter, or should the device fail, it solves on the CPU,
/// so it always answers.
pub fn solve_gpu(body: &VoxelBody, material: &Material, held: &[(u32, [bool; 3])], forces: &[(u32, V3)]) -> Result<VoxelSolution, FemError> {
    solve_on(body, material, None, held, forces, true)
}

/// Bodies with more cubes than this solve faster with the multigrid's
/// cycles on a GPU ([`solve_gpu`]); fewer, on the CPU. Measured on an Apple
/// M4 Pro against 14 cores: at 5 000 cubes the GPU took twice as long, at
/// 40 000 as long or less, for under half the CPU time.
pub const GPU_WORTH_ABOVE: usize = 20_000;

/// [`solve`], or [`solve_graded`] given each cube's `stiffness`, on whichever
/// is the faster for this body: the GPU past [`GPU_WORTH_ABOVE`] cubes, when
/// there is one, otherwise the CPU.
pub fn solve_fastest(
    body: &VoxelBody,
    material: &Material,
    stiffness: Option<&[f64]>,
    held: &[(u32, [bool; 3])],
    forces: &[(u32, V3)],
) -> Result<VoxelSolution, FemError> {
    if let Some(k) = stiffness {
        if k.len() != body.elements.len() {
            return Err(FemError::InvalidInput("a stiffness for each cube"));
        }
        if k.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(FemError::InvalidInput("each cube's stiffness above zero"));
        }
    }
    let on_gpu = body.elements.len() > GPU_WORTH_ABOVE && gpu_available();
    solve_on(body, material, stiffness, held, forces, on_gpu)
}

/// Whether [`solve_gpu`] has a device to run on.
pub fn gpu_available() -> bool {
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    {
        gpu::device().is_some()
    }
    #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
    {
        false
    }
}

fn solve_on(
    body: &VoxelBody,
    material: &Material,
    stiffness: Option<&[f64]>,
    held: &[(u32, [bool; 3])],
    forces: &[(u32, V3)],
    on_gpu: bool,
) -> Result<VoxelSolution, FemError> {
    material.validate()?;
    if material.anisotropic.is_some() {
        return Err(FemError::InvalidInput("voxel solver needs an isotropic material"));
    }
    let named = held.iter().map(|(n, _)| *n).chain(forces.iter().map(|(n, _)| *n));
    if named.into_iter().any(|n| n as usize >= body.nodes.len()) {
        return Err(FemError::InvalidInput("a support or load names a node the body does not have"));
    }
    let shared = element(material.poisson);
    let el: &Element = &shared;
    let n = body.nodes.len() * 3;
    let scale = material.youngs * body.res as f64;
    let mut fixed = vec![false; n];
    for &(h, which) in held {
        for c in 0..3 {
            fixed[h as usize * 3 + c] |= which[c];
        }
    }
    let mut f = vec![0.0f64; n];
    for &(node, v) in forces {
        for c in 0..3 {
            f[node as usize * 3 + c] += v[c];
        }
    }
    // Each cube's stiffness: the material's at the cube's size, times its own
    // share when graded.
    let factor: Vec<f64> = match stiffness {
        Some(k) => k.iter().map(|k| k * scale).collect(),
        None => vec![scale; body.elements.len()],
    };
    let mut diag = vec![0.0f64; n];
    for (e, fe) in body.elements.iter().zip(&factor) {
        for (c, &node) in e.iter().enumerate() {
            for d in 0..3 {
                diag[node as usize * 3 + d] += el.k[3 * c + d][3 * c + d] * fe;
            }
        }
    }
    // y += K_e x over some elements.
    let apply = |range: std::ops::Range<usize>, x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        let mut u = [0.0f64; 24];
        for (e, fe) in body.elements[range.clone()].iter().zip(&factor[range]) {
            for (c, &node) in e.iter().enumerate() {
                for d in 0..3 {
                    u[3 * c + d] = x[node as usize * 3 + d];
                }
            }
            for (c, &node) in e.iter().enumerate() {
                for d in 0..3 {
                    let s: f64 = el.k[3 * c + d].iter().zip(&u).map(|(a, b)| a * b).sum();
                    y[node as usize * 3 + d] += s * fe;
                }
            }
        }
        y
    };
    let product = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for part in per_chunk(body.elements.len(), |range| apply(range, x)) {
            for (a, b) in y.iter_mut().zip(part) {
                *a += b;
            }
        }
        for (i, v) in y.iter_mut().enumerate() {
            if fixed[i] {
                *v = 0.0;
            }
        }
        y
    };
    // Multigrid: the same body on cubes two, four, eight… times as big,
    // where the slow bending modes a diagonal preconditioner leaves to
    // thousands of iterations are cheap. Small bodies keep the diagonal.
    let hierarchy = (body.elements.len() > MULTIGRID_ABOVE).then(|| Multigrid::new(body, el, &factor, &fixed));
    let diagonal = |r: &[f64]| -> Vec<f64> {
        r.iter().zip(&diag).enumerate().map(|(i, (v, d))| if fixed[i] || *d == 0.0 { 0.0 } else { v / d }).collect()
    };
    let product = |x: &[f64]| -> Vec<f64> {
        match &hierarchy {
            Some(m) => m.levels[0].apply(el, x),
            None => product(x),
        }
    };
    // On a GPU, the cycles there, the iteration still here in double.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    let device_cycle = if on_gpu { hierarchy.as_ref().and_then(|m| gpu::DeviceCycle::new(m, el)) } else { None };
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    let cycled_on_device = std::cell::Cell::new(false);
    let precondition = |r: &[f64]| -> Vec<f64> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        if let Some(z) = device_cycle.as_ref().and_then(|d| d.cycle(r)) {
            cycled_on_device.set(true);
            return z;
        }
        match &hierarchy {
            Some(m) => m.cycle(el, 0, r),
            None => diagonal(r),
        }
    };
    // On the device, if asked and there is one; else, or should it fail,
    // here.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    // Small bodies, without a hierarchy, the whole iteration on the device.
    let on_device = if on_gpu && stiffness.is_none() && hierarchy.is_none() {
        gpu::displacements(body, el, scale, &diag, &fixed, &f, &product)
    } else {
        None
    };
    #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
    let on_device: Option<(Vec<f64>, usize)> = {
        let _ = on_gpu;
        None
    };
    let whole = on_device.is_some();
    let (x, iterations) = match on_device {
        Some(found) => found,
        None => cg(n, &product, &precondition, &f, &fixed)?,
    };
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    let gpu = whole || cycled_on_device.get();
    #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
    let gpu = whole;
    let mut solution = stresses(body, el, material, x, iterations)?;
    solution.gpu = gpu;
    Ok(solution)
}

/// The conjugate gradient here, in double precision: `x` with `K x = f`,
/// the held directions kept at zero; and how many iterations it took.
fn cg(n: usize, product: &dyn Fn(&[f64]) -> Vec<f64>, precondition: &dyn Fn(&[f64]) -> Vec<f64>, f: &[f64], fixed: &[bool]) -> Result<(Vec<f64>, usize), FemError> {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut x = vec![0.0f64; n];
    let mut r: Vec<f64> = f.iter().enumerate().map(|(i, v)| if fixed[i] { 0.0 } else { *v }).collect();
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let target = dot(&r, &r).sqrt() * 1e-7;
    let mut iterations = 0;
    while dot(&r, &r).sqrt() > target {
        if iterations == MOST_ITERATIONS {
            return Err(FemError::NoConvergence { iterations });
        }
        let q = product(&p);
        let alpha = rz / dot(&p, &q);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        // Flexible (Polak–Ribière): the same as the usual step for an exact,
        // symmetric preconditioner, and still sound for one worked out in
        // single precision on a GPU.
        let before = z;
        z = precondition(&r);
        let next = dot(&r, &z);
        let beta = (next - dot(&r, &before)) / rz;
        rz = next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        iterations += 1;
    }
    Ok((x, iterations))
}

/// The von Mises stress at each node from the displacements `x`.
fn stresses(body: &VoxelBody, el: &Element, material: &Material, x: Vec<f64>, iterations: usize) -> Result<VoxelSolution, FemError> {
    // The stress at each cube's corners, averaged where they meet. Over a
    // cube of side `res` the strain is the displacement over `res`.
    let to_stress = material.youngs / body.res as f64;
    let mut sum = vec![0.0f64; body.nodes.len()];
    let mut count = vec![0u32; body.nodes.len()];
    for nodes in &body.elements {
        let mut u = [0.0f64; 24];
        for (c, &node) in nodes.iter().enumerate() {
            for d in 0..3 {
                u[3 * c + d] = x[node as usize * 3 + d];
            }
        }
        for (c, &node) in nodes.iter().enumerate() {
            let s = el.stress(&u, corner(c)).map(|v| v * to_stress);
            sum[node as usize] += super::elastic::von_mises(s);
            count[node as usize] += 1;
        }
    }
    Ok(VoxelSolution {
        displacement: (0..body.nodes.len()).map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]]).collect(),
        stress: sum.iter().zip(&count).map(|(s, c)| *s / (*c).max(1) as f64).collect(),
        iterations,
        gpu: false,
    })
}

/// Bodies of more cubes than this are preconditioned by multigrid.
const MULTIGRID_ABOVE: usize = 500;
/// The coarsest level has no more cubes than this, and is relaxed there by
/// this many Jacobi sweeps.
const COARSEST: usize = 100;
const COARSE_SWEEPS: usize = 30;
/// Jacobi sweeps before and after each coarse correction.
const SMOOTH: usize = 1;

/// A geometric multigrid preconditioner on the voxel grid: each level the
/// one below in cubes twice the size, two by two by two.
///
/// A coarse cube's stiffness is the shared element matrix at its size,
/// scaled by how many of its eight fine cubes are filled, so a thin part is
/// not stiffened by its coarse cubes being full. Values pass down by
/// trilinear interpolation inside the coarse cube holding each fine node,
/// and residuals up by its transpose; a coarse node is held in a direction
/// where any fine node it reaches is. Each level's Jacobi damping comes from
/// its own largest eigenvalue, estimated by power iteration. Damped Jacobi smooths, as many sweeps after the
/// correction as before, and a fixed number of sweeps relaxes the coarsest
/// level: the cycle is a fixed, symmetric linear operator, as the conjugate
/// gradient it preconditions needs.
struct Multigrid {
    levels: Vec<Level>,
}

/// One level: its cubes, each cube's stiffness factor (modulus times size
/// times how full it is), its held directions, its inverse diagonal (zero
/// where held), and for each of its nodes the coarser level's nodes and
/// weights it is interpolated from (empty on the coarsest).
struct Level {
    cubes: Vec<[u32; 8]>,
    factor: Vec<f64>,
    fixed: Vec<bool>,
    inverse: Vec<f64>,
    from_coarse: Vec<Vec<(u32, f64)>>,
    /// The Jacobi damping: 4/3 over the largest eigenvalue of D⁻¹K, the
    /// usual choice for smoothing (it must stay under 2 over it).
    damping: f64,
    /// Each node's cube at each corner (the cube whose corner `c` it is),
    /// for working out `K x` a node at a time.
    around: Vec<[u32; 8]>,
}

/// A node's corner with no cube there.
const NO_CUBE: u32 = u32::MAX;
/// Levels with fewer nodes than this are worked one node after another.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
const PARALLEL_NODES: usize = 4096;

/// `f` at each of `nodes` nodes, three values each: across threads under
/// the `parallel` feature (on native targets), one after another otherwise.
fn by_node(nodes: usize, f: impl Fn(usize) -> [f64; 3] + Sync) -> Vec<f64> {
    let mut y = vec![0.0f64; nodes * 3];
    // Small levels (a cycle's coarse ones) one after another: waking every
    // thread for a few hundred nodes, thousands of times a solve, costs
    // more than the arithmetic.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if nodes >= PARALLEL_NODES {
        use rayon::prelude::*;
        y.par_chunks_mut(3 * 256).enumerate().for_each(|(block, out)| {
            for (k, o) in out.chunks_mut(3).enumerate() {
                o.copy_from_slice(&f(block * 256 + k));
            }
        });
        return y;
    }
    for (i, o) in y.chunks_mut(3).enumerate() {
        o.copy_from_slice(&f(i));
    }
    y
}

impl Level {
    fn new(cubes: Vec<[u32; 8]>, factor: Vec<f64>, fixed: Vec<bool>, el: &Element) -> Level {
        let mut diag = vec![0.0f64; fixed.len()];
        for (cube, f) in cubes.iter().zip(&factor) {
            for (c, &node) in cube.iter().enumerate() {
                for d in 0..3 {
                    diag[node as usize * 3 + d] += el.k[3 * c + d][3 * c + d] * f;
                }
            }
        }
        let inverse = diag.iter().zip(&fixed).map(|(d, held)| if *held || *d == 0.0 { 0.0 } else { 1.0 / d }).collect();
        let mut around = vec![[NO_CUBE; 8]; fixed.len() / 3];
        for (e, cube) in cubes.iter().enumerate() {
            for (c, &node) in cube.iter().enumerate() {
                around[node as usize][c] = e as u32;
            }
        }
        let mut level = Level { cubes, factor, fixed, inverse, from_coarse: vec![], damping: 0.0, around };
        level.damping = 4.0 / 3.0 / level.largest_eigenvalue(el);
        level
    }

    /// The largest eigenvalue of D⁻¹K over the free directions, by power
    /// iteration from a fixed start, rounded up a little for safety.
    fn largest_eigenvalue(&self, el: &Element) -> f64 {
        let n = self.fixed.len();
        let mut v: Vec<f64> = (0..n).map(|i| if self.fixed[i] { 0.0 } else { 1.0 + ((i * 7919) % 13) as f64 / 13.0 }).collect();
        let mut lambda = 1.0;
        for _ in 0..12 {
            let kv = self.apply(el, &v);
            let w: Vec<f64> = kv.iter().zip(&self.inverse).map(|(a, b)| a * b).collect();
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            let before = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm == 0.0 || before == 0.0 {
                return 1.0;
            }
            lambda = norm / before;
            v = w.iter().map(|x| x / norm).collect();
        }
        lambda * 1.1
    }

    /// K x, held directions zero: each node's three rows from the cubes
    /// around it, the nodes shared among threads with nothing to add up
    /// afterwards.
    fn apply(&self, el: &Element, x: &[f64]) -> Vec<f64> {
        by_node(self.around.len(), |node| {
            let mut out = [0.0f64; 3];
            for (c, &e) in self.around[node].iter().enumerate() {
                if e == NO_CUBE {
                    continue;
                }
                let cube = &self.cubes[e as usize];
                let f = self.factor[e as usize];
                for d in 0..3 {
                    let row = &el.k[3 * c + d];
                    let mut sum = 0.0;
                    for (b, &m) in cube.iter().enumerate() {
                        let m = m as usize * 3;
                        sum += row[3 * b] * x[m] + row[3 * b + 1] * x[m + 1] + row[3 * b + 2] * x[m + 2];
                    }
                    out[d] += sum * f;
                }
            }
            for (d, o) in out.iter_mut().enumerate() {
                if self.fixed[node * 3 + d] {
                    *o = 0.0;
                }
            }
            out
        })
    }

    /// `sweeps` of damped Jacobi on `K x = b`, from `x`.
    fn smooth(&self, el: &Element, x: &mut [f64], b: &[f64], sweeps: usize) {
        for _ in 0..sweeps {
            let kx = self.apply(el, x);
            for i in 0..x.len() {
                x[i] += self.damping * self.inverse[i] * (b[i] - kx[i]);
            }
        }
    }
}

impl Multigrid {
    fn new(body: &VoxelBody, el: &Element, factor: &[f64], fixed: &[bool]) -> Multigrid {
        // Each node's grid corner, and where on that grid the origin is, so
        // the coarse planes keep a symmetry plane through it.
        let mut keys = vec![[0i64; 3]; body.nodes.len()];
        for (k, &n) in &body.at {
            keys[n as usize] = k.map(i64::from);
        }
        let mut zero = [0, 1, 2].map(|a| (-(body.origin[a] / body.res)).round() as i64);
        let mut levels = vec![Level::new(body.elements.clone(), factor.to_vec(), fixed.to_vec(), el)];
        while levels.last().unwrap().cubes.len() > COARSEST && levels.len() < 8 {
            let fine = levels.last().unwrap();
            let off = zero.map(|z| z.rem_euclid(2));
            let coarse_of = |k: [i64; 3]| [0, 1, 2].map(|a| (k[a] - off[a]).div_euclid(2));
            // The coarse cubes, and their stiffness: their fine cubes' summed,
            // at twice the size, over the eight a full one holds.
            let mut cells: HashMap<[i64; 3], usize> = HashMap::new();
            let mut factor = vec![];
            let mut cell_of = Vec::with_capacity(fine.cubes.len());
            for (cube, f) in fine.cubes.iter().zip(&fine.factor) {
                let c = coarse_of(keys[cube[0] as usize]);
                let id = *cells.entry(c).or_insert_with(|| {
                    factor.push(0.0);
                    factor.len() - 1
                });
                factor[id] += f / 4.0;
                cell_of.push((c, id));
            }
            let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
            let mut coarse_keys = vec![];
            let mut cubes = vec![[0u32; 8]; factor.len()];
            // In order, so the numbering (and the rounding with it) is the
            // same every run.
            let mut ordered: Vec<(&[i64; 3], &usize)> = cells.iter().collect();
            ordered.sort();
            for (&c, &id) in ordered {
                for (corner, slot) in cubes[id].iter_mut().enumerate() {
                    let k = [c[0] + (corner & 1) as i64, c[1] + ((corner >> 1) & 1) as i64, c[2] + ((corner >> 2) & 1) as i64];
                    *slot = *ids.entry(k).or_insert_with(|| {
                        coarse_keys.push(k);
                        coarse_keys.len() as u32 - 1
                    });
                }
            }
            let mut held = vec![false; coarse_keys.len() * 3];
            // Each fine node from the corners of a coarse cube holding it.
            let mut from_coarse: Vec<Vec<(u32, f64)>> = vec![vec![]; keys.len()];
            for (cube, (c, id)) in fine.cubes.iter().zip(&cell_of) {
                for &node in cube {
                    let slot = &mut from_coarse[node as usize];
                    if !slot.is_empty() {
                        continue;
                    }
                    let k = keys[node as usize];
                    let t = [0, 1, 2].map(|a| (k[a] - off[a] - 2 * c[a]) as f64 / 2.0);
                    for (corner, &coarse) in cubes[*id].iter().enumerate() {
                        let w: f64 = (0..3).map(|a| if (corner >> a) & 1 == 1 { t[a] } else { 1.0 - t[a] }).product();
                        if w > 0.0 {
                            slot.push((coarse, w));
                            // Held where any fine node it reaches is: a
                            // coarse level held less than the fine one
                            // drifts and turns as a whole, which the fine
                            // level cannot use.
                            for d in 0..3 {
                                held[coarse as usize * 3 + d] |= fine.fixed[node as usize * 3 + d];
                            }
                        }
                    }
                }
            }
            let _ = &fine;
            levels.last_mut().unwrap().from_coarse = from_coarse;
            levels.push(Level::new(cubes, factor, held, el));
            keys = coarse_keys;
            zero = [0, 1, 2].map(|a| (zero[a] - off[a]) / 2);
        }
        Multigrid { levels }
    }

    /// One V-cycle at `level` for the residual `b`: the correction.
    fn cycle(&self, el: &Element, level: usize, b: &[f64]) -> Vec<f64> {
        let here = &self.levels[level];
        let mut x = vec![0.0f64; b.len()];
        if level + 1 == self.levels.len() {
            here.smooth(el, &mut x, b, COARSE_SWEEPS);
            return x;
        }
        here.smooth(el, &mut x, b, SMOOTH);
        let kx = here.apply(el, &x);
        let coarse = &self.levels[level + 1];
        let mut rc = vec![0.0f64; coarse.fixed.len()];
        for (node, from) in here.from_coarse.iter().enumerate() {
            for &(c, w) in from {
                for d in 0..3 {
                    rc[c as usize * 3 + d] += w * (b[node * 3 + d] - kx[node * 3 + d]);
                }
            }
        }
        for (v, held) in rc.iter_mut().zip(&coarse.fixed) {
            if *held {
                *v = 0.0;
            }
        }
        let ec = self.cycle(el, level + 1, &rc);
        for (node, from) in here.from_coarse.iter().enumerate() {
            for &(c, w) in from {
                for d in 0..3 {
                    if !here.fixed[node * 3 + d] {
                        x[node * 3 + d] += w * ec[c as usize * 3 + d];
                    }
                }
            }
        }
        here.smooth(el, &mut x, b, SMOOTH);
        x
    }
}

/// Iterations before a solve gives up: far past what a body held enough to
/// stand needs.
const MOST_ITERATIONS: usize = 20_000;

/// The element: a cube of side 1 and modulus 1 (its stiffness scales with
/// both), and the matrix recovering its incompatible modes from its
/// corners' displacements.
struct Element {
    /// Condensed stiffness, corners in the order x + 2y + 4z, three
    /// displacements each.
    k: [[f64; 24]; 24],
    /// The incompatible modes from the corners: α = R u.
    r: [[f64; 24]; 9],
    nu: f64,
}

/// Natural coordinates of corner `c`, each −1 or +1.
fn corner(c: usize) -> [f64; 3] {
    [(c & 1) as f64 * 2.0 - 1.0, ((c >> 1) & 1) as f64 * 2.0 - 1.0, ((c >> 2) & 1) as f64 * 2.0 - 1.0]
}

/// The strain–displacement matrix at natural point `x` of a unit cube: 24
/// columns for the corners, 9 for the incompatible modes.
fn b_matrix(x: [f64; 3]) -> [[f64; 33]; 6] {
    let mut b = [[0.0; 33]; 6];
    let mut put = |col: usize, g: [f64; 3]| {
        b[0][col] = g[0];
        b[1][col + 1] = g[1];
        b[2][col + 2] = g[2];
        b[3][col] = g[1];
        b[3][col + 1] = g[0];
        b[4][col + 1] = g[2];
        b[4][col + 2] = g[1];
        b[5][col] = g[2];
        b[5][col + 2] = g[0];
    };
    // dN/dx = 2 dN/dξ on a cube of side 1.
    for c in 0..8 {
        let n = corner(c);
        let f = |k: usize| 1.0 + x[k] * n[k];
        put(3 * c, [n[0] * f(1) * f(2) / 4.0, f(0) * n[1] * f(2) / 4.0, f(0) * f(1) * n[2] / 4.0]);
    }
    // The modes 1 − ξ², 1 − η², 1 − ζ², each in all three directions.
    for m in 0..3 {
        let mut g = [0.0; 3];
        g[m] = -4.0 * x[m];
        put(24 + 3 * m, g);
    }
    b
}

/// Isotropic elasticity at modulus 1.
fn d_matrix(nu: f64) -> [[f64; 6]; 6] {
    let l = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 1.0 / (2.0 * (1.0 + nu));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        d[i][..3].fill(l);
        d[i][i] = l + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    d
}

impl Element {
    fn new(nu: f64) -> Element {
        let d = d_matrix(nu);
        let g = 1.0 / 3f64.sqrt();
        let mut full = [[0.0f64; 33]; 33];
        // 2 × 2 × 2 Gauss points, weight 1 each, and the cube's Jacobian 1/8.
        for c in 0..8 {
            let b = b_matrix(corner(c).map(|v| v * g));
            for i in 0..33 {
                for j in 0..33 {
                    let mut s = 0.0;
                    for p in 0..6 {
                        for q in 0..6 {
                            s += b[p][i] * d[p][q] * b[q][j];
                        }
                    }
                    full[i][j] += s / 8.0;
                }
            }
        }
        // Condense the modes out: K = Kuu − Kua Kaa⁻¹ Kau, R = −Kaa⁻¹ Kau.
        let mut kaa = [[0.0; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                kaa[i][j] = full[24 + i][24 + j];
            }
        }
        let inv = invert9(kaa);
        let mut r = [[0.0; 24]; 9];
        for i in 0..9 {
            for j in 0..24 {
                r[i][j] = -(0..9).map(|m| inv[i][m] * full[24 + m][j]).sum::<f64>();
            }
        }
        let mut k = [[0.0; 24]; 24];
        for i in 0..24 {
            for j in 0..24 {
                k[i][j] = full[i][j] + (0..9).map(|m| full[i][24 + m] * r[m][j]).sum::<f64>();
            }
        }
        Element { k, r, nu }
    }

    /// The stress (modulus 1, unit cube) at natural point `x`.
    fn stress(&self, u: &[f64; 24], x: [f64; 3]) -> [f64; 6] {
        let mut all = [0.0; 33];
        all[..24].copy_from_slice(u);
        for i in 0..9 {
            all[24 + i] = (0..24).map(|j| self.r[i][j] * u[j]).sum();
        }
        let b = b_matrix(x);
        let strain: Vec<f64> = (0..6).map(|p| (0..33).map(|j| b[p][j] * all[j]).sum()).collect();
        let d = d_matrix(self.nu);
        let mut s = [0.0; 6];
        for p in 0..6 {
            s[p] = (0..6).map(|q| d[p][q] * strain[q]).sum();
        }
        s
    }
}

/// The elements made so far, by their Poisson's ratio's bits.
type Elements = Mutex<Vec<(u64, Arc<Element>)>>;

/// The element for Poisson's ratio `nu`, made once.
fn element(nu: f64) -> Arc<Element> {
    static CACHE: OnceLock<Elements> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = nu.to_bits();
    if let Some((_, e)) = cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return e.clone();
    }
    let e = Arc::new(Element::new(nu));
    cache.lock().unwrap().push((key, e.clone()));
    e
}

fn invert9(mut a: [[f64; 9]; 9]) -> [[f64; 9]; 9] {
    let mut inv = [[0.0; 9]; 9];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..9 {
        let pivot = (col..9).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap();
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..9 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for row in 0..9 {
            if row != col {
                let f = a[row][col];
                for j in 0..9 {
                    a[row][j] -= f * a[col][j];
                    inv[row][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}
