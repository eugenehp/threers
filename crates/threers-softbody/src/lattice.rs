//! The lattice a soft part is simulated on: cubes over the part's field, each
//! split into five tetrahedra; their rest state; and the binding of other
//! points (a skin, chain roots, things inside) to the tetrahedron each lies in.
//!
//! A port of Axolotl Jelly's `buildPart` / `tetLattice` / `embedPoints`
//! (Dream Meridian, MIT), arithmetic for arithmetic: values are computed in
//! f64 and stored in f32 where the original stores into a `Float32Array`, and
//! maps keep insertion order where the original's `Map` does.

use std::collections::HashMap;

/// An axis-aligned box, `min` and `max` corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// The tetrahedral lattice over a field: nodes on a grid of spacing `h`
/// starting at the table (`y = 0`), and the tetrahedra of every cell whose
/// centre is inside or near the surface.
#[derive(Debug, Clone)]
pub struct Lattice {
    pub origin: [f64; 3],
    pub h: f64,
    pub nx: i64,
    pub ny: i64,
    pub nz: i64,
    /// Node positions, xyz.
    pub nodes: Vec<f32>,
    /// Four node indices per tetrahedron, positively oriented.
    pub tets: Vec<u32>,
    /// The tetrahedra of each kept cell, by the cell's linear index.
    cell_tets: HashMap<i64, Vec<u32>>,
}

const TET_EVEN: [[usize; 4]; 5] = [[1, 2, 4, 7], [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7]];
const TET_ODD: [[usize; 4]; 5] = [[0, 3, 5, 6], [1, 0, 3, 5], [2, 0, 3, 6], [4, 0, 5, 6], [7, 3, 5, 6]];

fn tet_volume(p: &[f32], t: &[u32]) -> f64 {
    let at = |i: u32, c: usize| p[3 * i as usize + c] as f64;
    let (a, b, c, d) = (t[0], t[1], t[2], t[3]);
    let (ux, uy, uz) = (at(b, 0) - at(a, 0), at(b, 1) - at(a, 1), at(b, 2) - at(a, 2));
    let (vx, vy, vz) = (at(c, 0) - at(a, 0), at(c, 1) - at(a, 1), at(c, 2) - at(a, 2));
    let (wx, wy, wz) = (at(d, 0) - at(a, 0), at(d, 1) - at(a, 1), at(d, 2) - at(a, 2));
    ((uy * vz - uz * vy) * wx + (uz * vx - ux * vz) * wy + (ux * vy - uy * vx) * wz) / 6.0
}

fn invert3(m: [f64; 9], out: &mut [f32]) {
    let [a, b, c, d, e, f, g, h, i] = m;
    let (ca, cb, cc) = (e * i - f * h, -(d * i - f * g), d * h - e * g);
    let r = 1.0 / (a * ca + b * cb + c * cc);
    out[0] = (ca * r) as f32;
    out[1] = (-(b * i - c * h) * r) as f32;
    out[2] = ((b * f - c * e) * r) as f32;
    out[3] = (cb * r) as f32;
    out[4] = ((a * i - c * g) * r) as f32;
    out[5] = (-(a * f - c * d) * r) as f32;
    out[6] = (cc * r) as f32;
    out[7] = (-(a * h - b * g) * r) as f32;
    out[8] = ((a * e - b * d) * r) as f32;
}

impl Lattice {
    /// Cubes of side `h` over `bounds` (from the table up), kept where the
    /// field at the cube's centre is below `0.6 h`, five tetrahedra each in
    /// alternating orientations so neighbouring cubes share their faces.
    pub fn new(f: &dyn Fn(f64, f64, f64) -> f64, bounds: &Bounds, h: f64) -> Lattice {
        let o = [bounds.min[0], 0.0, bounds.min[2]];
        let nx = ((bounds.max[0] - o[0]) / h).ceil() as i64;
        let ny = ((bounds.max[1] - o[1]) / h).ceil() as i64;
        let nz = ((bounds.max[2] - o[2]) / h).ceil() as i64;
        let mut node_id: HashMap<i64, u32> = HashMap::new();
        let (mut pos, mut tets) = (Vec::<f32>::new(), Vec::<u32>::new());
        let mut cell_tets = HashMap::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (o[0] + (i as f64 + 0.5) * h, o[1] + (j as f64 + 0.5) * h, o[2] + (k as f64 + 0.5) * h);
                    if f(c.0, c.1, c.2) > h * 0.6 {
                        continue;
                    }
                    let mut corner = [0u32; 8];
                    for (b, slot) in corner.iter_mut().enumerate() {
                        let (ii, jj, kk) = (i + (b as i64 & 1), j + ((b as i64 >> 1) & 1), k + ((b as i64 >> 2) & 1));
                        let key = (kk * (ny + 1) + jj) * (nx + 1) + ii;
                        *slot = *node_id.entry(key).or_insert_with(|| {
                            let n = (pos.len() / 3) as u32;
                            pos.push((o[0] + ii as f64 * h) as f32);
                            pos.push((o[1] + jj as f64 * h) as f32);
                            pos.push((o[2] + kk as f64 * h) as f32);
                            n
                        });
                    }
                    let mut list = Vec::with_capacity(5);
                    for t in if (i + j + k) & 1 == 1 { &TET_ODD } else { &TET_EVEN } {
                        list.push((tets.len() / 4) as u32);
                        tets.extend(t.iter().map(|&q| corner[q]));
                    }
                    cell_tets.insert((k * ny + j) * nx + i, list);
                }
            }
        }
        for t in tets.chunks_exact_mut(4) {
            if tet_volume(&pos, t) < 0.0 {
                t.swap(2, 3);
            }
        }
        Lattice { origin: o, h, nx, ny, nz, nodes: pos, tets, cell_tets }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len() / 3
    }
    pub fn tet_count(&self) -> usize {
        self.tets.len() / 4
    }
}

/// Points bound to a lattice: for each, the tetrahedron it lies in (or the
/// nearest) and its first three barycentric coordinates there.
#[derive(Debug, Clone, Default)]
pub struct Embedding {
    pub tet_of: Vec<u32>,
    pub bary: Vec<f32>,
}

/// Bind each point (xyz, flat) to the tetrahedron that contains it, or failing
/// that the one it is least outside of, searching outward up to three cells.
pub fn embed_points(points: &[f32], lat: &Lattice, dm_inv: &[f32]) -> Embedding {
    let nv = points.len() / 3;
    let mut emb = Embedding { tet_of: vec![0; nv], bary: vec![0.0; nv * 3] };
    let (o, h, nx, ny) = (lat.origin, lat.h, lat.nx, lat.ny);
    for v in 0..nv {
        let (px, py, pz) = (points[3 * v] as f64, points[3 * v + 1] as f64, points[3 * v + 2] as f64);
        let ci = ((px - o[0]) / h).floor() as i64;
        let cj = ((py - o[1]) / h).floor() as i64;
        let ck = ((pz - o[2]) / h).floor() as i64;
        let (mut best, mut best_score, mut bb) = (0u32, f64::NEG_INFINITY, [0.0f64; 3]);
        let mut r = 0i64;
        while r <= 3 && best_score < -1e-4 {
            for dk in -r..=r {
                for dj in -r..=r {
                    for di in -r..=r {
                        if di.abs().max(dj.abs()).max(dk.abs()) != r {
                            continue;
                        }
                        let Some(list) = lat.cell_tets.get(&(((ck + dk) * ny + (cj + dj)) * nx + (ci + di))) else {
                            continue;
                        };
                        for &t in list {
                            let a = 3 * lat.tets[4 * t as usize] as usize;
                            let m = 9 * t as usize;
                            let x = &lat.nodes;
                            let (rx, ry, rz) = (px - x[a] as f64, py - x[a + 1] as f64, pz - x[a + 2] as f64);
                            let d = |i: usize| dm_inv[m + i] as f64;
                            let b1 = d(0) * rx + d(3) * ry + d(6) * rz;
                            let b2 = d(1) * rx + d(4) * ry + d(7) * rz;
                            let b3 = d(2) * rx + d(5) * ry + d(8) * rz;
                            let score = (1.0 - b1 - b2 - b3).min(b1).min(b2).min(b3);
                            if score > best_score {
                                best_score = score;
                                best = t;
                                bb = [b1, b2, b3];
                            }
                        }
                    }
                }
            }
            r += 1;
        }
        emb.tet_of[v] = best;
        for c in 0..3 {
            emb.bary[3 * v + c] = bb[c] as f32;
        }
    }
    emb
}

/// One soft part: its lattice and everything the solver needs about it at
/// rest — inverse rest edge matrices, rest volumes, unique edges (in the
/// order tetrahedra first name them), the boundary triangles, and each
/// node's collision radius — plus its skin bound to the lattice.
#[derive(Debug, Clone)]
pub struct Part {
    pub lattice: Lattice,
    /// Per tetrahedron, the inverse of its rest edge matrix, column-major.
    pub dm_inv: Vec<f32>,
    pub rest_vol: Vec<f32>,
    pub edges: Vec<u32>,
    /// Triangles on the lattice's surface, three node indices each.
    pub boundary: Vec<u32>,
    /// How far out from each node another body is kept.
    pub radius: Vec<f32>,
    pub skin: Embedding,
}

impl Part {
    /// Build the lattice of the field `f` over `bounds` at spacing `h`, and
    /// bind the `skin` points (xyz, flat) to it.
    pub fn new(f: &dyn Fn(f64, f64, f64) -> f64, bounds: &Bounds, h: f64, skin: &[f32]) -> Part {
        let lat = Lattice::new(f, bounds, h);
        let (nt, nn) = (lat.tet_count(), lat.node_count());
        let mut dm_inv = vec![0.0f32; nt * 9];
        let mut rest_vol = vec![0.0f32; nt];
        // insertion-ordered sets, as the original's Maps
        let mut edge_at: HashMap<u64, usize> = HashMap::new();
        let mut edge_list: Vec<(u32, u32)> = Vec::new();
        let mut face_at: HashMap<[u32; 3], usize> = HashMap::new();
        let mut face_list: Vec<(u32, [u32; 3])> = Vec::new();
        let p = &lat.nodes;
        for t in 0..nt {
            let q = [lat.tets[4 * t], lat.tets[4 * t + 1], lat.tets[4 * t + 2], lat.tets[4 * t + 3]];
            let a = 3 * q[0] as usize;
            let mut m = [0.0f64; 9];
            for c in 1..=3 {
                let b = 3 * q[c] as usize;
                for k in 0..3 {
                    m[3 * (c - 1) + k] = p[b + k] as f64 - p[a + k] as f64;
                }
            }
            invert3(m, &mut dm_inv[9 * t..9 * t + 9]);
            rest_vol[t] = tet_volume(p, &q) as f32;
            for i in 0..4 {
                for j in i + 1..4 {
                    let (lo, hi) = (q[i].min(q[j]), q[i].max(q[j]));
                    let key = lo as u64 * nn as u64 + hi as u64;
                    if let Some(&at) = edge_at.get(&key) {
                        edge_list[at] = (lo, hi);
                    } else {
                        edge_at.insert(key, edge_list.len());
                        edge_list.push((lo, hi));
                    }
                }
            }
            for fc in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
                let tri = [q[fc[0]], q[fc[1]], q[fc[2]]];
                let mut key = tri;
                key.sort_unstable();
                if let Some(&at) = face_at.get(&key) {
                    face_list[at].0 += 1;
                } else {
                    face_at.insert(key, face_list.len());
                    face_list.push((1, tri));
                }
            }
        }
        let edges = edge_list.iter().flat_map(|&(a, b)| [a, b]).collect();
        let boundary = face_list.iter().filter(|f| f.0 == 1).flat_map(|f| f.1).collect();
        let radius = (0..nn)
            .map(|i| {
                let d = f(p[3 * i] as f64, p[3 * i + 1] as f64, p[3 * i + 2] as f64);
                (0.35 * h - d).clamp(0.0, h * 1.1) as f32
            })
            .collect();
        let skin = embed_points(skin, &lat, &dm_inv);
        Part { lattice: lat, dm_inv, rest_vol, edges, boundary, radius, skin }
    }

    pub fn node_count(&self) -> usize {
        self.lattice.node_count()
    }
    pub fn embed(&self, points: &[f32]) -> Embedding {
        embed_points(points, &self.lattice, &self.dm_inv)
    }
}
