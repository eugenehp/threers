//! Filling a solid with tetrahedra: isosurface stuffing on a body-centred
//! cubic lattice, after Labelle and Shewchuk (SIGGRAPH 2007).
//!
//! The solid is anything with a signed distance — a closed triangle surface
//! through [`SignedDistance`], an SDF from `threers::sdf`, a closure. The
//! lattice is two interleaved cubic grids, corners and centres, whose
//! tetrahedra are all congruent and well shaped. Where a lattice edge crosses
//! the surface:
//!
//! 1. **Warp.** If the crossing lies close to one end — within a quarter of a
//!    long edge or two-fifths of a short one — that lattice vertex moves onto
//!    the crossing. This is what keeps slivers out: a crossing near a vertex is
//!    the one that would cut a sliver off.
//! 2. **Cut.** Every lattice tetrahedron is then clipped to the inside. The
//!    clipped piece is a tetrahedron, a pyramid or a prism; the last two are
//!    split by Dompierre's rule — each quadrilateral face cut along the
//!    diagonal through its lowest-numbered vertex — which needs no extra
//!    points and makes the two tetrahedra sharing a face split it the same way.
//!
//! Every boundary node lies on the surface (to the root finder's tolerance of
//! a millionth of the spacing). What the mesh does not keep is a sharp
//! feature narrower than the spacing: a cube's edges come out bevelled by up
//! to about one lattice step. Mesh a part with sharp edges at a spacing small
//! against the features whose stress you care about, or, for a box,
//! use [`TetMesh::box_grid`], which is exact.
//!
//! Labelle and Shewchuk prove dihedral bounds of 10.7°–164.8° for their own
//! case tables; the generic clip here gives up that proof for brevity. The
//! tests require a minimum dihedral of 1° and a mean radius ratio above 0.6;
//! the demo's L-bracket comes out at 21°–122° at a 4 mm spacing.
//! [`TetMesh::quality`] reports what a mesh got.

use crate::mesh::TetMesh;
use crate::surface::{SignedDistance, TriSurface};
use crate::vec3::*;
use crate::FemError;
use std::collections::HashMap;
use threers::core::BufferGeometry;

/// Warp threshold on long (axis-aligned) lattice edges, from the paper.
const ALPHA_LONG: f64 = 0.24999;
/// Warp threshold on short (diagonal) lattice edges.
const ALPHA_SHORT: f64 = 0.41189;

impl TetMesh {
    /// Mesh a closed triangle surface with tetrahedra about `spacing` across.
    pub fn from_surface(surface: &TriSurface, spacing: f64) -> Result<Self, FemError> {
        let sd = SignedDistance::new(surface)?;
        let (lo, hi) = surface.bounds();
        Self::from_sdf(lo, hi, spacing, |p| sd.distance(p))
    }

    /// Mesh a closed renderable geometry, welded first — see
    /// [`TriSurface::from_geometry`].
    pub fn from_geometry(geometry: &BufferGeometry, spacing: f64) -> Result<Self, FemError> {
        Self::from_surface(&TriSurface::from_geometry(geometry)?, spacing)
    }

    /// Mesh the region where `sdf` is negative inside the box `lo..hi`.
    ///
    /// `sdf` should be a true distance, or at least Lipschitz with constant
    /// near 1 — the root finder assumes the zero crossing along an edge is
    /// unique.
    pub fn from_sdf<F>(lo: V3, hi: V3, spacing: f64, sdf: F) -> Result<Self, FemError>
    where
        F: Fn(V3) -> f64 + Sync,
    {
        if spacing.is_nan() || spacing <= 0.0 {
            return Err(FemError::InvalidInput("spacing must be positive"));
        }
        let h = spacing;
        // One cell of padding on every side, so the outer layer is outside.
        let origin = sub(lo, [h; 3]);
        let n = [0, 1, 2].map(|k| (((hi[k] - lo[k]) / h).ceil() as usize + 2).max(2));
        let cells = n[0] * n[1] * n[2];
        if cells > 64 << 20 {
            return Err(FemError::InvalidInput(
                "spacing too small: over 64 million lattice cells",
            ));
        }
        let lattice = Lattice { origin, h, n };
        let count = lattice.count();
        let mut pos: Vec<V3> = (0..count).map(|v| lattice.position(v)).collect();
        let mut f: Vec<f64> = map_maybe_parallel(&pos, &|p: &V3| sdf(*p));

        // Every lattice edge once, with whether it is long.
        let edges = lattice.edges();

        // 1. Warp vertices that a crossing comes too close to.
        let mut warp: Vec<Option<(f64, V3)>> = vec![None; count];
        let crossings: Vec<(usize, f64, V3)> = {
            let crossing_edges: Vec<(u32, u32, bool)> = edges
                .iter()
                .copied()
                .filter(|&(a, b, _)| crosses(f[a as usize], f[b as usize]))
                .collect();
            let roots: Vec<(f64, V3)> = map_maybe_parallel(&crossing_edges, &|&(a, b, _): &(u32, u32, bool)| {
                root(&sdf, pos[a as usize], pos[b as usize], f[a as usize], f[b as usize], h)
            });
            let mut out = Vec::new();
            for (&(a, b, long), (t, p)) in crossing_edges.iter().zip(roots) {
                let alpha = if long { ALPHA_LONG } else { ALPHA_SHORT };
                if t < alpha {
                    out.push((a as usize, t, p));
                }
                if 1.0 - t < alpha {
                    out.push((b as usize, 1.0 - t, p));
                }
            }
            out
        };
        for (v, t, p) in crossings {
            // The closest violating crossing wins.
            if warp[v].is_none_or(|(best, _)| t < best) {
                warp[v] = Some((t, p));
            }
        }
        for (v, w) in warp.iter().enumerate() {
            if let Some((_, p)) = w {
                pos[v] = *p;
                f[v] = 0.0;
            }
        }

        // 2. Crossings that survive the warp become new nodes.
        let mut cut_index: HashMap<(u32, u32), u32> = HashMap::new();
        let surviving: Vec<(u32, u32)> = edges
            .iter()
            .filter(|&&(a, b, _)| crosses(f[a as usize], f[b as usize]))
            .map(|&(a, b, _)| (a.min(b), a.max(b)))
            .collect();
        let cut_points: Vec<V3> = map_maybe_parallel(&surviving, &|&(a, b): &(u32, u32)| {
            root(&sdf, pos[a as usize], pos[b as usize], f[a as usize], f[b as usize], h).1
        });
        for (i, e) in surviving.iter().enumerate() {
            cut_index.insert(*e, (count + i) as u32);
        }
        pos.extend(cut_points);

        // 3. Clip every lattice tetrahedron to the inside.
        let sign = |v: u32| {
            let x = f[v as usize];
            if x < 0.0 {
                -1
            } else if x > 0.0 {
                1
            } else {
                0
            }
        };
        let cut = |a: u32, b: u32| cut_index[&(a.min(b), a.max(b))];
        let mut tets: Vec<[u32; 4]> = Vec::new();
        lattice.for_each_tet(|t| {
            let s = t.map(sign);
            let neg: Vec<u32> = (0..4).filter(|&i| s[i] < 0).map(|i| t[i]).collect();
            if neg.is_empty() {
                return;
            }
            let zer: Vec<u32> = (0..4).filter(|&i| s[i] == 0).map(|i| t[i]).collect();
            let pos_: Vec<u32> = (0..4).filter(|&i| s[i] > 0).map(|i| t[i]).collect();
            match (neg.len(), zer.len(), pos_.len()) {
                (_, _, 0) => tets.push(t),
                (1, _, _) => {
                    let a = neg[0];
                    let mut tet = vec![a];
                    tet.extend(&zer);
                    tet.extend(pos_.iter().map(|&p| cut(a, p)));
                    tets.push([tet[0], tet[1], tet[2], tet[3]]);
                }
                (2, 0, 2) => {
                    let (a, b) = (neg[0], neg[1]);
                    let (c, d) = (pos_[0], pos_[1]);
                    prism(
                        [a, cut(a, c), cut(a, d), b, cut(b, c), cut(b, d)],
                        &mut tets,
                    );
                }
                (2, 1, 1) => {
                    let (a, b, z, c) = (neg[0], neg[1], zer[0], pos_[0]);
                    pyramid([a, b, cut(b, c), cut(a, c)], z, &mut tets);
                }
                (3, 0, 1) => {
                    let (a, b, c, d) = (neg[0], neg[1], neg[2], pos_[0]);
                    prism([a, b, c, cut(a, d), cut(b, d), cut(c, d)], &mut tets);
                }
                _ => unreachable!("sign pattern {s:?}"),
            }
        });

        // Orient, and drop pieces flattened to nothing by warping two of
        // their corners onto the same crossing.
        let floor = 1e-12 * h * h * h;
        let mut mesh = TetMesh {
            nodes: pos,
            tets: Vec::with_capacity(tets.len()),
        };
        for mut t in tets {
            let [a, b, c, d] = t.map(|i| mesh.nodes[i as usize]);
            let v = tet_volume6(a, b, c, d);
            if v.abs() <= floor {
                continue;
            }
            if v < 0.0 {
                t.swap(2, 3);
            }
            mesh.tets.push(t);
        }
        if mesh.tets.is_empty() {
            return Err(FemError::InvalidInput(
                "nothing inside: the solid is empty or thinner than the spacing",
            ));
        }
        mesh.compact();
        Ok(mesh)
    }
}

#[inline]
fn crosses(a: f64, b: f64) -> bool {
    (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)
}

/// The zero of `sdf` on segment `pa → pb`, as a parameter from `pa` and a
/// point, by the Illinois variant of false position.
fn root<F: Fn(V3) -> f64>(sdf: &F, pa: V3, pb: V3, fa: f64, fb: f64, h: f64) -> (f64, V3) {
    let (mut t0, mut t1) = (0.0, 1.0);
    let (mut f0, mut f1) = (fa, fb);
    let len = dist(pa, pb).max(1e-300);
    let tol = 1e-6 * h / len;
    let mut side = 0;
    let mut t = 0.5;
    for _ in 0..60 {
        t = (t0 * f1 - t1 * f0) / (f1 - f0);
        if !t.is_finite() {
            t = 0.5 * (t0 + t1);
        }
        let ft = sdf(lerp(pa, pb, t));
        if ft == 0.0 || (t1 - t0) < tol {
            break;
        }
        if (ft < 0.0) == (f0 < 0.0) {
            t0 = t;
            f0 = ft;
            if side == -1 {
                f1 *= 0.5;
            }
            side = -1;
        } else {
            t1 = t;
            f1 = ft;
            if side == 1 {
                f0 *= 0.5;
            }
            side = 1;
        }
        if (t1 - t0) < tol {
            t = 0.5 * (t0 + t1);
            break;
        }
    }
    (t, lerp(pa, pb, t))
}

/// Split prism `v0 v1 v2 / v3 v4 v5` (with `vi` joined to `vi+3`) into three
/// tetrahedra, every quadrilateral face cut through its lowest-numbered vertex.
fn prism(v: [u32; 6], out: &mut Vec<[u32; 4]>) {
    // Relabel so the smallest vertex is v0, keeping the prism's structure:
    // a rotation of the triangles, and a swap of top and bottom.
    let m = (0..6).min_by_key(|&i| v[i]).unwrap();
    let (bottom, top) = if m < 3 { ([0, 1, 2], [3, 4, 5]) } else { ([3, 4, 5], [0, 1, 2]) };
    let r = m % 3;
    let w = [
        v[bottom[r]],
        v[bottom[(r + 1) % 3]],
        v[bottom[(r + 2) % 3]],
        v[top[r]],
        v[top[(r + 1) % 3]],
        v[top[(r + 2) % 3]],
    ];
    if w[1].min(w[5]) < w[2].min(w[4]) {
        out.push([w[0], w[1], w[2], w[5]]);
        out.push([w[0], w[1], w[5], w[4]]);
        out.push([w[0], w[4], w[5], w[3]]);
    } else {
        out.push([w[0], w[1], w[2], w[4]]);
        out.push([w[0], w[4], w[2], w[5]]);
        out.push([w[0], w[4], w[5], w[3]]);
    }
}

/// Split a pyramid with base `q` (in cyclic order) and apex `z` into two
/// tetrahedra, the base cut through its lowest-numbered vertex.
fn pyramid(q: [u32; 4], z: u32, out: &mut Vec<[u32; 4]>) {
    let m = (0..4).min_by_key(|&i| q[i]).unwrap();
    if m % 2 == 0 {
        out.push([q[0], q[1], q[2], z]);
        out.push([q[0], q[2], q[3], z]);
    } else {
        out.push([q[1], q[2], q[3], z]);
        out.push([q[1], q[3], q[0], z]);
    }
}

/// The body-centred cubic lattice: corner vertices of an `n`-cell grid, then
/// one vertex at each cell centre.
struct Lattice {
    origin: V3,
    h: f64,
    n: [usize; 3],
}

impl Lattice {
    fn corners(&self) -> usize {
        (self.n[0] + 1) * (self.n[1] + 1) * (self.n[2] + 1)
    }

    fn count(&self) -> usize {
        self.corners() + self.n[0] * self.n[1] * self.n[2]
    }

    fn corner(&self, i: usize, j: usize, k: usize) -> u32 {
        (i + (self.n[0] + 1) * (j + (self.n[1] + 1) * k)) as u32
    }

    fn centre(&self, i: usize, j: usize, k: usize) -> u32 {
        (self.corners() + i + self.n[0] * (j + self.n[1] * k)) as u32
    }

    fn position(&self, v: usize) -> V3 {
        let c = self.corners();
        let (ijk, off) = if v < c {
            let nx = self.n[0] + 1;
            let ny = self.n[1] + 1;
            ([v % nx, (v / nx) % ny, v / (nx * ny)], 0.0)
        } else {
            let v = v - c;
            let (nx, ny) = (self.n[0], self.n[1]);
            ([v % nx, (v / nx) % ny, v / (nx * ny)], 0.5)
        };
        [0, 1, 2].map(|a| self.origin[a] + (ijk[a] as f64 + off) * self.h)
    }

    /// Every lattice edge once, flagged long (axis-aligned, length h) or
    /// short (centre to corner, length h·√3/2).
    fn edges(&self) -> Vec<(u32, u32, bool)> {
        let [nx, ny, nz] = self.n;
        let mut e = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let a = self.corner(i, j, k);
                    if i < nx {
                        e.push((a, self.corner(i + 1, j, k), true));
                    }
                    if j < ny {
                        e.push((a, self.corner(i, j + 1, k), true));
                    }
                    if k < nz {
                        e.push((a, self.corner(i, j, k + 1), true));
                    }
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = self.centre(i, j, k);
                    if i + 1 < nx {
                        e.push((c, self.centre(i + 1, j, k), true));
                    }
                    if j + 1 < ny {
                        e.push((c, self.centre(i, j + 1, k), true));
                    }
                    if k + 1 < nz {
                        e.push((c, self.centre(i, j, k + 1), true));
                    }
                    for dk in 0..2 {
                        for dj in 0..2 {
                            for di in 0..2 {
                                e.push((c, self.corner(i + di, j + dj, k + dk), false));
                            }
                        }
                    }
                }
            }
        }
        e
    }

    /// Every lattice tetrahedron: two neighbouring centres and an edge of the
    /// square face between them.
    fn for_each_tet(&self, mut f: impl FnMut([u32; 4])) {
        let [nx, ny, nz] = self.n;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = self.centre(i, j, k);
                    // The face toward +x, +y and +z, each shared with the
                    // neighbouring cell.
                    for axis in 0..3 {
                        let mut nb = [i, j, k];
                        nb[axis] += 1;
                        if nb[0] >= nx || nb[1] >= ny || nb[2] >= nz {
                            continue;
                        }
                        let d = self.centre(nb[0], nb[1], nb[2]);
                        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                        // The square: corners at axis = +1, u and v in {0,1}.
                        let sq = |a: usize, b: usize| {
                            let mut p = [i, j, k];
                            p[axis] += 1;
                            p[u] += a;
                            p[v] += b;
                            self.corner(p[0], p[1], p[2])
                        };
                        let ring = [sq(0, 0), sq(1, 0), sq(1, 1), sq(0, 1)];
                        for r in 0..4 {
                            f([c, d, ring[r], ring[(r + 1) % 4]]);
                        }
                    }
                }
            }
        }
    }
}

fn map_maybe_parallel<T: Sync, R: Send>(items: &[T], f: &(impl Fn(&T) -> R + Sync)) -> Vec<R> {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        return items.par_iter().map(f).collect();
    }
    #[allow(unreachable_code)]
    items.iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prism_split_is_three_positive_volume_tets() {
        let p: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        for perm in [[0u32, 1, 2, 3, 4, 5], [4, 2, 5, 0, 3, 1], [5, 4, 3, 2, 1, 0]] {
            // Relabel vertex ids without moving the points.
            let mut out = Vec::new();
            prism(perm, &mut out);
            let pos = |id: u32| p[perm.iter().position(|&q| q == id).unwrap()];
            let vol: f64 = out
                .iter()
                .map(|t| tet_volume6(pos(t[0]), pos(t[1]), pos(t[2]), pos(t[3])).abs() / 6.0)
                .sum();
            assert!((vol - 0.5).abs() < 1e-12, "{vol}");
        }
    }
}
