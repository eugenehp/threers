//! Tetrahedral meshes.

use crate::surface::TriSurface;
use crate::vec3::*;
use std::collections::HashMap;

/// Outward faces of a positively oriented tetrahedron, as local corner
/// indices; face `k` is opposite corner `k`.
pub const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// A linear tetrahedral mesh. Every tetrahedron is positively oriented:
/// [`tet_volume6`] of its corners in order is positive.
#[derive(Clone, Debug, Default)]
pub struct TetMesh {
    pub nodes: Vec<V3>,
    pub tets: Vec<[u32; 4]>,
}

/// A boundary face of a [`TetMesh`]: three nodes wound counter-clockwise
/// from outside, and the tetrahedron it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryFace {
    pub nodes: [u32; 3],
    pub tet: u32,
    /// Which face of the tetrahedron, as an index into [`TET_FACES`].
    pub local: u8,
}

/// Shape statistics of a mesh, for judging whether to trust a solve on it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshQuality {
    /// Smallest dihedral angle of any tetrahedron, in degrees. A regular
    /// tetrahedron has 70.5°; stress results get unreliable below about 5°.
    pub min_dihedral: f64,
    /// Largest dihedral angle, in degrees. 180° is a flat element.
    pub max_dihedral: f64,
    /// Smallest radius ratio `3·r_in / r_circ` (1 for a regular tetrahedron).
    pub min_radius_ratio: f64,
    /// Mean radius ratio.
    pub mean_radius_ratio: f64,
}

impl TetMesh {
    pub fn new(nodes: Vec<V3>, tets: Vec<[u32; 4]>) -> Self {
        let mut m = Self { nodes, tets };
        m.orient();
        m
    }

    /// A box from `lo` to `hi` cut into `n[0]·n[1]·n[2]` cubes of six
    /// tetrahedra each (the Freudenthal–Kuhn split, conforming across cubes).
    /// Exact geometry, which makes it the mesh to test a solver on.
    pub fn box_grid(lo: V3, hi: V3, n: [usize; 3]) -> Self {
        let [nx, ny, nz] = n.map(|v| v.max(1));
        let id = |i: usize, j: usize, k: usize| (i + (nx + 1) * (j + (ny + 1) * k)) as u32;
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push([
                        lo[0] + (hi[0] - lo[0]) * i as f64 / nx as f64,
                        lo[1] + (hi[1] - lo[1]) * j as f64 / ny as f64,
                        lo[2] + (hi[2] - lo[2]) * k as f64 / nz as f64,
                    ]);
                }
            }
        }
        const PERMS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut tets = Vec::with_capacity(6 * nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    for p in PERMS {
                        let mut c = [i, j, k];
                        let mut t = [id(c[0], c[1], c[2]); 4];
                        for (s, axis) in p.iter().enumerate() {
                            c[*axis] += 1;
                            t[s + 1] = id(c[0], c[1], c[2]);
                        }
                        tets.push(t);
                    }
                }
            }
        }
        Self::new(nodes, tets)
    }

    /// Flip any negatively oriented tetrahedron.
    pub fn orient(&mut self) {
        for t in &mut self.tets {
            let [a, b, c, d] = t.map(|i| self.nodes[i as usize]);
            if tet_volume6(a, b, c, d) < 0.0 {
                t.swap(2, 3);
            }
        }
    }

    pub fn corners(&self, t: usize) -> [V3; 4] {
        self.tets[t].map(|i| self.nodes[i as usize])
    }

    pub fn tet_volume(&self, t: usize) -> f64 {
        let [a, b, c, d] = self.corners(t);
        tet_volume6(a, b, c, d) / 6.0
    }

    pub fn volume(&self) -> f64 {
        (0..self.tets.len()).map(|t| self.tet_volume(t)).sum()
    }

    pub fn bounds(&self) -> (V3, V3) {
        bounds(&self.nodes)
    }

    /// Faces that belong to exactly one tetrahedron, wound outward.
    pub fn boundary_faces(&self) -> Vec<BoundaryFace> {
        let mut seen: HashMap<[u32; 3], (u32, u8, bool)> = HashMap::with_capacity(self.tets.len() * 2);
        for (t, tet) in self.tets.iter().enumerate() {
            for (k, f) in TET_FACES.iter().enumerate() {
                let mut key = f.map(|i| tet[i]);
                key.sort_unstable();
                seen.entry(key)
                    .and_modify(|e| e.2 = false)
                    .or_insert((t as u32, k as u8, true));
            }
        }
        let mut out: Vec<BoundaryFace> = seen
            .into_values()
            .filter(|e| e.2)
            .map(|(t, k, _)| BoundaryFace {
                nodes: TET_FACES[k as usize].map(|i| self.tets[t as usize][i]),
                tet: t,
                local: k,
            })
            .collect();
        // Hash order is not deterministic; the solvers' results should be.
        out.sort_unstable_by_key(|f| (f.tet, f.local));
        out
    }

    /// The boundary as a triangle surface over the boundary nodes, and for
    /// each surface vertex the mesh node it came from.
    pub fn boundary_surface(&self) -> (TriSurface, Vec<u32>) {
        let faces = self.boundary_faces();
        let mut map: HashMap<u32, u32> = HashMap::new();
        let mut back = Vec::new();
        let mut vertices = Vec::new();
        let mut triangles = Vec::with_capacity(faces.len());
        for f in &faces {
            let t = f.nodes.map(|n| {
                *map.entry(n).or_insert_with(|| {
                    back.push(n);
                    vertices.push(self.nodes[n as usize]);
                    (vertices.len() - 1) as u32
                })
            });
            triangles.push(t);
        }
        (TriSurface::new(vertices, triangles), back)
    }

    /// Faces shared by more than two tetrahedra, plus boundary edges not
    /// shared by exactly two boundary faces. Zero for a conforming mesh of a
    /// closed solid.
    pub fn defects(&self) -> usize {
        let mut count: HashMap<[u32; 3], u32> = HashMap::new();
        for tet in &self.tets {
            for f in TET_FACES {
                let mut key = f.map(|i| tet[i]);
                key.sort_unstable();
                *count.entry(key).or_default() += 1;
            }
        }
        let over = count.values().filter(|c| **c > 2).count();
        over + self.boundary_surface().0.defects()
    }

    /// Drop nodes no tetrahedron uses, renumbering the rest in order.
    pub fn compact(&mut self) {
        let mut used = vec![false; self.nodes.len()];
        for t in &self.tets {
            for &i in t {
                used[i as usize] = true;
            }
        }
        let mut map = vec![u32::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (i, u) in used.iter().enumerate() {
            if *u {
                map[i] = nodes.len() as u32;
                nodes.push(self.nodes[i]);
            }
        }
        for t in &mut self.tets {
            *t = t.map(|i| map[i as usize]);
        }
        self.nodes = nodes;
    }

    pub fn quality(&self) -> MeshQuality {
        let mut q = MeshQuality {
            min_dihedral: 180.0,
            max_dihedral: 0.0,
            min_radius_ratio: f64::INFINITY,
            mean_radius_ratio: 0.0,
        };
        for t in 0..self.tets.len() {
            let p = self.corners(t);
            let (lo, hi) = dihedral_range(p);
            q.min_dihedral = q.min_dihedral.min(lo);
            q.max_dihedral = q.max_dihedral.max(hi);
            let r = radius_ratio(p);
            q.min_radius_ratio = q.min_radius_ratio.min(r);
            q.mean_radius_ratio += r;
        }
        if !self.tets.is_empty() {
            q.mean_radius_ratio /= self.tets.len() as f64;
        } else {
            q.min_radius_ratio = 0.0;
        }
        q
    }
}

/// Smallest and largest dihedral angle of a tetrahedron, in degrees.
pub fn dihedral_range(p: [V3; 4]) -> (f64, f64) {
    let normals = TET_FACES.map(|f| normalize(cross(sub(p[f[1]], p[f[0]]), sub(p[f[2]], p[f[0]]))));
    let mut lo: f64 = 180.0;
    let mut hi: f64 = 0.0;
    for i in 0..4 {
        for j in i + 1..4 {
            let a = 180.0 - dot(normals[i], normals[j]).clamp(-1.0, 1.0).acos().to_degrees();
            lo = lo.min(a);
            hi = hi.max(a);
        }
    }
    (lo, hi)
}

/// `3·r_in / r_circ`: 1 for a regular tetrahedron, 0 for a flat one.
pub fn radius_ratio(p: [V3; 4]) -> f64 {
    let v = tet_volume6(p[0], p[1], p[2], p[3]).abs() / 6.0;
    let area: f64 = TET_FACES
        .iter()
        .map(|f| 0.5 * norm(cross(sub(p[f[1]], p[f[0]]), sub(p[f[2]], p[f[0]]))))
        .sum();
    if v <= 0.0 || area <= 0.0 {
        return 0.0;
    }
    let r_in = 3.0 * v / area;
    // Circumradius: |a² (b×c) + b² (c×a) + c² (a×b)| / (12 V), edges from p0.
    let a = sub(p[1], p[0]);
    let b = sub(p[2], p[0]);
    let c = sub(p[3], p[0]);
    let num = add(
        add(scale(cross(b, c), norm2(a)), scale(cross(c, a), norm2(b))),
        scale(cross(a, b), norm2(c)),
    );
    let r_circ = norm(num) / (12.0 * v);
    3.0 * r_in / r_circ
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_grid_is_conforming_and_exact() {
        let m = TetMesh::box_grid([0.0; 3], [2.0, 1.0, 3.0], [4, 3, 5]);
        assert_eq!(m.tets.len(), 6 * 4 * 3 * 5);
        assert!((m.volume() - 6.0).abs() < 1e-12);
        assert!(m.tets.iter().enumerate().all(|(t, _)| m.tet_volume(t) > 0.0));
        assert_eq!(m.defects(), 0);
        let (s, _) = m.boundary_surface();
        assert!((s.area() - 2.0 * (2.0 + 6.0 + 3.0)).abs() < 1e-12);
        assert!((s.volume() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn regular_tet_quality() {
        let p = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let (lo, hi) = dihedral_range(p);
        assert!((lo - 70.528779).abs() < 1e-5 && (hi - 70.528779).abs() < 1e-5);
        assert!((radius_ratio(p) - 1.0).abs() < 1e-12);
    }
}
