//! Marching cubes over a sampled scalar field, welded and watertight.
//!
//! The textbook implementation drives a 256-entry triangle table transcribed by
//! hand. This one derives that table on first use by contouring each cube face
//! and chaining the segments into loops, which buys two things the transcribed
//! table does not:
//!
//! - **Ambiguous faces cannot crack.** A face with four sign changes can be
//!   contoured two ways. The rule here reads only the face's four corner signs,
//!   and the two cells sharing that face see the same four signs, so they always
//!   pick the same pairing and their surfaces meet edge to edge.
//! - **Winding falls out of the construction.** Face segments are directed —
//!   walking a face counter-clockwise from outside, each crossing *into* the
//!   solid runs to the next crossing *out* of it — so chaining them yields
//!   loops that are already wound counter-clockwise seen from outside the
//!   surface, and no triangle needs a normal test to orient it.
//!
//! Vertices are keyed by the grid edge they sit on, so neighbouring cells share
//! them: the output is indexed and, for a field that is negative all around the
//! sampled box, closed.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;
use std::sync::OnceLock;

/// Corner `c` of a cell sits at `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`.
/// Edges are grouped by axis, so `edge / 4` is 0 for x, 1 for y, 2 for z.
const EDGES: [(u8, u8); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// The six faces, corners counter-clockwise as seen from *outside* the cell.
/// Adjacent faces therefore traverse a shared edge in opposite directions,
/// which is what makes every cut edge exactly one segment start and one end.
const FACES: [[u8; 4]; 6] = [
    [0, 4, 6, 2], // -x
    [1, 3, 7, 5], // +x
    [0, 1, 5, 4], // -y
    [2, 6, 7, 3], // +y
    [0, 2, 3, 1], // -z
    [4, 5, 7, 6], // +z
];

const fn edge_index(a: u8, b: u8) -> u8 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let mut e = 0;
    while e < 12 {
        if EDGES[e].0 == lo && EDGES[e].1 == hi {
            return e as u8;
        }
        e += 1;
    }
    panic!("corners are not a cube edge");
}

/// The oriented contour loops for one of the 256 corner sign patterns, as cell
/// edge indices. Bit `c` of `case` is set when corner `c` is inside.
fn loops_for(case: u8) -> Vec<Vec<u8>> {
    let inside = |c: u8| case & (1 << c) != 0;

    // `next[e]` — the cut edge the contour leaves for after entering on `e`.
    // Each cut edge starts a segment on one of its two faces and ends one on
    // the other, so this is a permutation of the cut edges.
    let mut next = [u8::MAX; 12];
    for face in FACES {
        let mut kind = [0i8; 4];
        for i in 0..4 {
            let (a, b) = (face[i], face[(i + 1) % 4]);
            kind[i] = match (inside(a), inside(b)) {
                (true, false) => 1,  // leaving the solid
                (false, true) => -1, // entering it
                _ => 0,
            };
        }
        // Signs alternate around a closed face, so each "entering" crossing is
        // matched with the next "leaving" one. On a four-crossing face that
        // choice is the ambiguity, and it depends only on the corner signs.
        //
        // Entering-to-leaving, rather than the other way round, is what puts
        // the solid on the right of the contour and so makes the loops wind
        // counter-clockwise seen from outside the surface.
        for i in 0..4 {
            if kind[i] != -1 {
                continue;
            }
            for d in 1..4 {
                let j = (i + d) % 4;
                if kind[j] == 1 {
                    let from = edge_index(face[i], face[(i + 1) % 4]);
                    let to = edge_index(face[j], face[(j + 1) % 4]);
                    next[from as usize] = to;
                    break;
                }
            }
        }
    }

    let mut loops = Vec::new();
    let mut seen = [false; 12];
    for start in 0..12u8 {
        if next[start as usize] == u8::MAX || seen[start as usize] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut e = start;
        while !seen[e as usize] {
            seen[e as usize] = true;
            cycle.push(e);
            e = next[e as usize];
        }
        if cycle.len() >= 3 {
            loops.push(cycle);
        }
    }
    loops
}

fn case_table() -> &'static [Vec<Vec<u8>>; 256] {
    static TABLE: OnceLock<[Vec<Vec<u8>>; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|case| loops_for(case as u8)))
}

/// A scalar field sampled on a regular grid, positive inside the solid.
pub(crate) struct IsoGrid {
    /// Sample counts per axis; at least 2 each.
    pub dims: [usize; 3],
    /// World position of sample `(0, 0, 0)`.
    pub origin: Vector3,
    /// World spacing between samples.
    pub step: Vector3,
    /// `dims.x * dims.y * dims.z` samples, x fastest.
    pub values: Vec<f32>,
}

impl IsoGrid {
    fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[(k * self.dims[1] + j) * self.dims[0] + i]
    }

    /// Central difference at a sample, one-sided on the boundary. Points into
    /// the solid, so the outward normal is its negation.
    fn gradient(&self, i: usize, j: usize, k: usize) -> Vector3 {
        let [nx, ny, nz] = self.dims;
        let (x0, x1) = (i.saturating_sub(1), (i + 1).min(nx - 1));
        let (y0, y1) = (j.saturating_sub(1), (j + 1).min(ny - 1));
        let (z0, z1) = (k.saturating_sub(1), (k + 1).min(nz - 1));
        Vector3::new(
            (self.at(x1, j, k) - self.at(x0, j, k)) / ((x1 - x0) as f32 * self.step.x),
            (self.at(i, y1, k) - self.at(i, y0, k)) / ((y1 - y0) as f32 * self.step.y),
            (self.at(i, j, z1) - self.at(i, j, z0)) / ((z1 - z0) as f32 * self.step.z),
        )
    }

    /// Triangulate the zero level set. UVs are a planar projection onto the
    /// plane the vertex normal faces most directly, normalised over the grid —
    /// enough for tiling detail maps, with a seam wherever that axis flips.
    pub(crate) fn triangulate(&self) -> BufferGeometry {
        let [nx, ny, nz] = self.dims;
        let table = case_table();

        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Which vertex sits on each grid edge, so neighbouring cells share it.
        //
        // A cell only ever touches edges on its own z-layer and the one above,
        // so two layers of `nx × ny × 3` are enough and the table rolls forward
        // as the sweep does. Keyed by arithmetic rather than hashed: a hash map
        // over every crossed edge was the single largest cost in here, and it
        // was also the largest allocation — tens of megabytes of table for a
        // mesh whose *live* front is a couple of hundred kilobytes.
        let plane = nx * ny;
        let mut rows: [Vec<u32>; 2] = [vec![u32::MAX; plane * 3], vec![u32::MAX; plane * 3]];

        let extent = Vector3::new(
            (nx - 1) as f32 * self.step.x,
            (ny - 1) as f32 * self.step.y,
            (nz - 1) as f32 * self.step.z,
        );

        let mut corner = [0.0f32; 8];
        let mut cell_edge = [u32::MAX; 12];
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let mut case = 0u8;
                    for (c, slot) in corner.iter_mut().enumerate() {
                        *slot = self.at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                        if *slot > 0.0 {
                            case |= 1 << c;
                        }
                    }
                    let loops = &table[case as usize];
                    if loops.is_empty() {
                        continue;
                    }

                    cell_edge.fill(u32::MAX);
                    for cycle in loops {
                        for &e in cycle {
                            let slot = &mut cell_edge[e as usize];
                            if *slot != u32::MAX {
                                continue;
                            }
                            let (a, b) = EDGES[e as usize];
                            let axis = (e / 4) as usize;
                            let (ai, aj, ak) = (
                                i + (a as usize & 1),
                                j + ((a as usize >> 1) & 1),
                                k + ((a as usize >> 2) & 1),
                            );
                            // x- and y-parallel edges sit on layer `k` or `k+1`;
                            // z-parallel ones start on `k`. Either way the
                            // difference is 0 or 1.
                            let row = &mut rows[ak - k][axis * plane + aj * nx + ai];
                            if *row != u32::MAX {
                                *slot = *row;
                                continue;
                            }
                            let (va, vb) = (corner[a as usize], corner[b as usize]);
                            let denom = va - vb;
                            let t = if denom.abs() > f32::EPSILON {
                                (va / denom).clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            let (bi, bj, bk) = (
                                i + (b as usize & 1),
                                j + ((b as usize >> 1) & 1),
                                k + ((b as usize >> 2) & 1),
                            );
                            let mut p = Vector3::new(
                                self.origin.x + ai as f32 * self.step.x,
                                self.origin.y + aj as f32 * self.step.y,
                                self.origin.z + ak as f32 * self.step.z,
                            );
                            match axis {
                                0 => p.x += t * self.step.x,
                                1 => p.y += t * self.step.y,
                                _ => p.z += t * self.step.z,
                            }
                            let ga = self.gradient(ai, aj, ak);
                            let gb = self.gradient(bi, bj, bk);
                            let n = (ga * (1.0 - t) + gb * t) * -1.0;
                            let n = if n.length_sq() > 0.0 {
                                n.normalize()
                            } else {
                                Vector3::new(0.0, 0.0, 1.0)
                            };
                            let index = (positions.len() / 3) as u32;
                            positions.extend_from_slice(&[p.x, p.y, p.z]);
                            normals.extend_from_slice(&[n.x, n.y, n.z]);
                            *row = index;
                            *slot = index;
                        }
                        if cycle.len() == 3 {
                            indices.extend_from_slice(&[
                                cell_edge[cycle[0] as usize],
                                cell_edge[cycle[1] as usize],
                                cell_edge[cycle[2] as usize],
                            ]);
                            continue;
                        }
                        // Fanning a bigger loop from one of its own vertices
                        // folds the triangles back on themselves when the loop
                        // saddles, which it does whenever the cell straddles a
                        // neck. A centre vertex follows the loop instead: the
                        // triangles stay on the same side and stay well shaped,
                        // and the loop's own edges — the ones the neighbouring
                        // cells have to meet — are untouched.
                        let hub = {
                            let inv = 1.0 / cycle.len() as f32;
                            let (mut p, mut n) = (Vector3::ZERO, Vector3::ZERO);
                            for &e in cycle {
                                let v = cell_edge[e as usize] as usize * 3;
                                p = p + Vector3::new(
                                    positions[v],
                                    positions[v + 1],
                                    positions[v + 2],
                                );
                                n = n + Vector3::new(normals[v], normals[v + 1], normals[v + 2]);
                            }
                            p = p * inv;
                            let n = if n.length_sq() > 0.0 {
                                n.normalize()
                            } else {
                                Vector3::new(0.0, 0.0, 1.0)
                            };
                            let index = (positions.len() / 3) as u32;
                            positions.extend_from_slice(&[p.x, p.y, p.z]);
                            normals.extend_from_slice(&[n.x, n.y, n.z]);
                            index
                        };
                        for w in 0..cycle.len() {
                            indices.extend_from_slice(&[
                                hub,
                                cell_edge[cycle[w] as usize],
                                cell_edge[cycle[(w + 1) % cycle.len()] as usize],
                            ]);
                        }
                    }
                }
            }
            // Roll the table forward. The upper layer's x- and y-edges become
            // the lower layer's, which is exactly what the next sweep needs;
            // its z-edges were never written, and the layer being retired is
            // cleared to take the new upper one.
            rows.swap(0, 1);
            rows[1].fill(u32::MAX);
        }

        let mut geom = BufferGeometry::new();
        if positions.is_empty() {
            return geom;
        }
        let mut uvs = Vec::with_capacity(positions.len() / 3 * 2);
        for v in 0..positions.len() / 3 {
            let p = Vector3::new(
                positions[v * 3] - self.origin.x,
                positions[v * 3 + 1] - self.origin.y,
                positions[v * 3 + 2] - self.origin.z,
            );
            let n = [
                normals[v * 3].abs(),
                normals[v * 3 + 1].abs(),
                normals[v * 3 + 2].abs(),
            ];
            let (u, w) = if n[0] >= n[1] && n[0] >= n[2] {
                (p.z / extent.z, p.y / extent.y)
            } else if n[1] >= n[2] {
                (p.x / extent.x, p.z / extent.z)
            } else {
                (p.x / extent.x, p.y / extent.y)
            };
            uvs.extend_from_slice(&[u, w]);
        }

        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        geom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Sample a field over a box padded far enough that it is negative all the
    /// way round, so the surface is closed.
    fn grid_of(res: usize, half: f32, field: impl Fn(Vector3) -> f32) -> IsoGrid {
        let step = 2.0 * half / (res - 1) as f32;
        let origin = Vector3::new(-half, -half, -half);
        let mut values = Vec::with_capacity(res * res * res);
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    values.push(field(Vector3::new(
                        origin.x + i as f32 * step,
                        origin.y + j as f32 * step,
                        origin.z + k as f32 * step,
                    )));
                }
            }
        }
        IsoGrid {
            dims: [res; 3],
            origin,
            step: Vector3::new(step, step, step),
            values,
        }
    }

    fn triangles(geom: &BufferGeometry) -> Vec<[u32; 3]> {
        let idx = geom.index.as_ref().expect("indexed");
        idx.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect()
    }

    #[test]
    fn every_case_yields_closed_loops() {
        // A loop must visit each cut edge once and close: the permutation the
        // face rule builds has no fixed points and no stragglers.
        for case in 0..256u16 {
            let case = case as u8;
            let loops = loops_for(case);
            let mut visited: Vec<u8> = loops.iter().flatten().copied().collect();
            visited.sort_unstable();
            let before = visited.len();
            visited.dedup();
            assert_eq!(before, visited.len(), "case {case} repeats an edge");

            let inside = |c: u8| case & (1 << c) != 0;
            let mut cut: Vec<u8> = (0..12u8)
                .filter(|&e| {
                    let (a, b) = EDGES[e as usize];
                    inside(a) != inside(b)
                })
                .collect();
            cut.sort_unstable();
            assert_eq!(cut, visited, "case {case} misses a cut edge");
        }
    }

    #[test]
    fn sphere_is_watertight_and_outward_facing() {
        let r = 0.6f32;
        let geom = grid_of(24, 1.0, |p| r - p.length()).triangulate();
        let tris = triangles(&geom);
        assert!(!tris.is_empty());

        // Watertight and consistently wound: every directed edge appears once,
        // and its reverse appears once — the signature of an oriented manifold.
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (key, dir) = if e.0 < e.1 {
                    ((e.0, e.1), 1)
                } else {
                    ((e.1, e.0), -1)
                };
                *edges.entry(key).or_insert(0) += dir;
            }
        }
        assert!(
            edges.values().all(|&v| v == 0),
            "open or inconsistently wound edges"
        );

        // Outward: the winding normal agrees with the radial direction.
        let pos = geom.get_attribute("position").unwrap();
        let vert = |i: u32| {
            let i = i as usize * 3;
            Vector3::new(pos.array[i], pos.array[i + 1], pos.array[i + 2])
        };
        let inward = tris
            .iter()
            .filter(|t| {
                let (a, b, c) = (vert(t[0]), vert(t[1]), vert(t[2]));
                let n = (b - a).cross(c - a);
                n.dot(a + b + c) < 0.0
            })
            .count();
        assert_eq!(inward, 0, "of {} triangles", tris.len());

        // And it is the right sphere: every vertex sits on it.
        for i in 0..pos.count() as u32 {
            assert!((vert(i).length() - r).abs() < 0.05);
        }
    }

    #[test]
    fn normals_point_out() {
        let geom = grid_of(20, 1.0, |p| 0.6 - p.length()).triangulate();
        let pos = geom.get_attribute("position").unwrap();
        let nrm = geom.get_attribute("normal").unwrap();
        for i in 0..pos.count() {
            let p = Vector3::new(pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]);
            let n = Vector3::new(nrm.array[i * 3], nrm.array[i * 3 + 1], nrm.array[i * 3 + 2]);
            assert!(
                n.dot(p.normalize()) > 0.9,
                "normal off the radial by too far"
            );
        }
    }

    #[test]
    fn saddle_cells_do_not_fold_back() {
        // A gyroid puts a saddle through most cells, so its loops are the long
        // non-planar ones — exactly the case a fan from a loop vertex inverts.
        let geom = grid_of(28, 2.0, |p| {
            let (x, y, z) = (p.x * 2.0, p.y * 2.0, p.z * 2.0);
            0.4 - (x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()).abs()
        })
        .triangulate();
        let pos = geom.get_attribute("position").unwrap();
        let nrm = geom.get_attribute("normal").unwrap();
        let vec = |attr: &BufferAttribute, i: u32| {
            let i = i as usize * 3;
            Vector3::new(attr.array[i], attr.array[i + 1], attr.array[i + 2])
        };
        let mut inverted = 0;
        let tris = triangles(&geom);
        for t in &tris {
            let (a, b, c) = (vec(pos, t[0]), vec(pos, t[1]), vec(pos, t[2]));
            let face = (b - a).cross(c - a);
            let shading = vec(nrm, t[0]) + vec(nrm, t[1]) + vec(nrm, t[2]);
            if face.dot(shading) < 0.0 {
                inverted += 1;
            }
        }
        assert_eq!(inverted, 0, "of {} triangles", tris.len());
    }

    #[test]
    fn empty_field_makes_empty_geometry() {
        let geom = grid_of(8, 1.0, |_| -1.0).triangulate();
        assert!(geom.get_attribute("position").is_none());
    }

    #[test]
    fn two_disjoint_blobs_both_survive() {
        // Two spheres in one grid — the cell-local table must not merge them.
        let geom = grid_of(32, 2.0, |p| {
            let a = 0.5 - (p - Vector3::new(-0.9, 0.0, 0.0)).length();
            let b = 0.5 - (p - Vector3::new(0.9, 0.0, 0.0)).length();
            a.max(b)
        })
        .triangulate();
        let pos = geom.get_attribute("position").unwrap();
        let (mut left, mut right) = (0, 0);
        for i in 0..pos.count() {
            if pos.array[i * 3] < 0.0 {
                left += 1;
            } else {
                right += 1;
            }
        }
        assert!(left > 100 && right > 100, "lost a blob: {left}/{right}");
    }
}
