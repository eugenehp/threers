//! Catmull–Clark subdivision of a polygon cage, and
//! [`BufferGeometry::subdivide`] on top of it.
//!
//! One round turns every n-gon into n quads:
//!
//! - a **face point** at the average of each face's vertices;
//! - an **edge point** at the average of the edge's two ends and the two face
//!   points either side — or just the edge's midpoint if it has one face, which
//!   is what keeps a boundary from shrinking away;
//! - each original vertex moved toward its neighbours by
//!   `(F + 2R + (n-3)P) / n`, where `F` averages the touching face points and
//!   `R` averages the touching edge midpoints.
//!
//! Creases and corners are honoured to the extent [`SubdivisionOptions`]
//! states them: a listed edge or vertex is held in place.
//!
//! The same code refines USD subdivision surfaces (`crate::loaders::usd`); it
//! lives here so it needs no feature. three.js shipped this job as the
//! `SubdivisionModifier` addon (Loop subdivision, triangles only) and its
//! successor `LoopSubdivision`; this is Catmull–Clark, which is what quad
//! modellers and USD mean by a subdivision surface, and it works on triangles
//! too (each becomes three quads).

use crate::core::{BufferAttribute, BufferGeometry};
use std::collections::HashMap;

/// A polygon mesh — the control cage — before triangulation: positions,
/// corners per face, and the faces' corner indices back to back.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cage {
    pub positions: Vec<[f32; 3]>,
    /// How many vertices each face has.
    pub counts: Vec<u32>,
    pub indices: Vec<u32>,
}

/// Creases, corners and boundary rule for a refinement.
#[derive(Debug, Clone, Default)]
pub struct SubdivisionOptions {
    /// Vertices held in place entirely.
    pub corners: Vec<u32>,
    /// Edges held in place, as vertex pairs.
    pub creases: Vec<(u32, u32)>,
    /// Whether a boundary edge is held. USD's default is to hold it, which is
    /// what stops an open surface from pulling away from its own border.
    pub sharp_boundary: bool,
}

/// Refine a cage `level` times with Catmull–Clark.
pub fn catmull_clark(mut cage: Cage, level: usize, options: &SubdivisionOptions) -> Cage {
    for _ in 0..level {
        cage = once(cage, options);
    }
    cage
}

/// An edge, keyed so that `(a, b)` and `(b, a)` are the same edge.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn average(points: &[[f32; 3]]) -> [f32; 3] {
    let n = points.len().max(1) as f32;
    let mut out = [0.0; 3];
    for p in points {
        for i in 0..3 {
            out[i] += p[i];
        }
    }
    for value in &mut out {
        *value /= n;
    }
    out
}

fn once(cage: Cage, options: &SubdivisionOptions) -> Cage {
    let Cage {
        positions,
        counts,
        indices,
    } = cage;
    if positions.is_empty() || counts.is_empty() {
        return Cage {
            positions,
            counts,
            indices,
        };
    }

    // The faces, as slices of the index buffer.
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(counts.len());
    let mut at = 0usize;
    for count in &counts {
        let count = *count as usize;
        if at + count > indices.len() {
            break;
        }
        faces.push(indices[at..at + count].to_vec());
        at += count;
    }

    // A face point per face.
    let face_points: Vec<[f32; 3]> = faces
        .iter()
        .map(|face| {
            let corners: Vec<[f32; 3]> = face
                .iter()
                .filter_map(|i| positions.get(*i as usize).copied())
                .collect();
            average(&corners)
        })
        .collect();

    // Which faces touch each edge, and the edges in the order they are first
    // met — so the new vertices come out in the same order every run, rather
    // than in a hash map's.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut edge_order: Vec<(u32, u32)> = Vec::new();
    for (f, face) in faces.iter().enumerate() {
        for i in 0..face.len() {
            let edge = key(face[i], face[(i + 1) % face.len()]);
            let touching = edge_faces.entry(edge).or_default();
            if touching.is_empty() {
                edge_order.push(edge);
            }
            touching.push(f);
        }
    }

    let creases: std::collections::HashSet<(u32, u32)> =
        options.creases.iter().map(|(a, b)| key(*a, *b)).collect();
    let corners: std::collections::HashSet<u32> = options.corners.iter().copied().collect();

    // An edge point per edge.
    let mut new_positions = positions.clone();
    let mut edge_points: HashMap<(u32, u32), u32> = HashMap::new();
    for edge in &edge_order {
        let touching = &edge_faces[edge];
        let (a, b) = *edge;
        let (Some(pa), Some(pb)) = (positions.get(a as usize), positions.get(b as usize)) else {
            continue;
        };
        let midpoint = average(&[*pa, *pb]);
        // A crease, a boundary, or an edge with an odd number of faces keeps
        // its midpoint; anything else pulls toward the faces either side.
        let point = if creases.contains(edge)
            || touching.len() != 2
            || (options.sharp_boundary && touching.len() == 1)
        {
            midpoint
        } else {
            average(&[*pa, *pb, face_points[touching[0]], face_points[touching[1]]])
        };
        edge_points.insert(*edge, new_positions.len() as u32);
        new_positions.push(point);
    }

    // The face points, after the edge points so the indices stay stable.
    let face_point_base = new_positions.len() as u32;
    new_positions.extend_from_slice(&face_points);

    // Move each original vertex toward its neighbourhood.
    let mut touching_faces: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];
    for (f, face) in faces.iter().enumerate() {
        for v in face {
            if let Some(slot) = touching_faces.get_mut(*v as usize) {
                slot.push(f);
            }
        }
    }
    let mut touching_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); positions.len()];
    for edge in &edge_order {
        for v in [edge.0, edge.1] {
            if let Some(slot) = touching_edges.get_mut(v as usize) {
                slot.push(*edge);
            }
        }
    }

    for v in 0..positions.len() {
        let p = positions[v];
        let index = v as u32;
        // A corner does not move, and neither does a vertex on a held
        // boundary or between two creases.
        let boundary: Vec<(u32, u32)> = touching_edges[v]
            .iter()
            .copied()
            .filter(|e| {
                creases.contains(e)
                    || (options.sharp_boundary && edge_faces.get(e).map(|f| f.len()) == Some(1))
            })
            .collect();
        if corners.contains(&index) || boundary.len() > 2 {
            continue;
        }
        // USD's default is `interpolateBoundary = "edgeAndCorner"`: where a
        // held boundary turns a corner — two boundary edges meeting at a
        // vertex with a single face — the corner is held rather than smoothed.
        // Without this an open quad's corners creep inward.
        if boundary.len() == 2 && touching_faces[v].len() <= 1 {
            continue;
        }
        if boundary.len() == 2 {
            // On a crease, a vertex follows the crease rather than the surface.
            let ends: Vec<[f32; 3]> = boundary
                .iter()
                .filter_map(|(a, b)| {
                    let other = if *a == index { *b } else { *a };
                    positions.get(other as usize).copied()
                })
                .collect();
            if ends.len() == 2 {
                let mut moved = [0.0; 3];
                for i in 0..3 {
                    moved[i] = (ends[0][i] + 6.0 * p[i] + ends[1][i]) / 8.0;
                }
                new_positions[v] = moved;
            }
            continue;
        }

        let n = touching_faces[v].len() as f32;
        if n < 3.0 {
            continue;
        }
        let f = average(
            &touching_faces[v]
                .iter()
                .map(|f| face_points[*f])
                .collect::<Vec<_>>(),
        );
        let r = average(
            &touching_edges[v]
                .iter()
                .filter_map(|(a, b)| {
                    let (pa, pb) = (positions.get(*a as usize)?, positions.get(*b as usize)?);
                    Some(average(&[*pa, *pb]))
                })
                .collect::<Vec<_>>(),
        );
        let mut moved = [0.0; 3];
        for i in 0..3 {
            moved[i] = (f[i] + 2.0 * r[i] + (n - 3.0) * p[i]) / n;
        }
        new_positions[v] = moved;
    }

    // Every n-gon becomes n quads, each one corner of the original.
    let mut new_counts = Vec::new();
    let mut new_indices = Vec::new();
    for (f, face) in faces.iter().enumerate() {
        let face_point = face_point_base + f as u32;
        for i in 0..face.len() {
            let previous = face[(i + face.len() - 1) % face.len()];
            let current = face[i];
            let next = face[(i + 1) % face.len()];
            let (Some(before), Some(after)) = (
                edge_points.get(&key(previous, current)),
                edge_points.get(&key(current, next)),
            ) else {
                continue;
            };
            new_counts.push(4);
            new_indices.extend_from_slice(&[current, *after, face_point, *before]);
        }
    }

    Cage {
        positions: new_positions,
        counts: new_counts,
        indices: new_indices,
    }
}

impl BufferGeometry {
    /// Catmull–Clark subdivision, `levels` rounds, with an open boundary held
    /// where it is (USD's default `edgeAndCorner`: border edges stay on their
    /// curve and border corners stay put). See
    /// [`subdivide_with`](Self::subdivide_with).
    pub fn subdivide(&self, levels: usize) -> BufferGeometry {
        self.subdivide_with(
            levels,
            &SubdivisionOptions {
                sharp_boundary: true,
                ..Default::default()
            },
        )
    }

    /// Catmull–Clark subdivision with explicit creases, corners and boundary
    /// rule. Crease and corner indices refer to the **cage** vertices — this
    /// geometry's vertices welded by position, numbered in order of first use
    /// by [`faces`](Self::faces) — which [`subdivision_cage`](Self::subdivision_cage)
    /// returns.
    ///
    /// The cage is the geometry's polygons ([`faces`](Self::faces): a quad
    /// read from OBJ or made by a generator is refined as a quad), or its
    /// triangles where there is no grouping, with vertices welded by position
    /// so a seam that splits vertices for UVs or normals does not tear the
    /// surface open.
    ///
    /// The result is indexed, every face a quad (for `levels ≥ 1`) fanned into
    /// two triangles, with the polygon grouping set so wireframes, exporters
    /// and picking see quads. It carries `position` and smooth `normal`
    /// (recomputed) only: the cage has no face-varying data, so UVs, colours
    /// and other attributes are **dropped** rather than interpolated wrongly
    /// across a seam.
    pub fn subdivide_with(&self, levels: usize, options: &SubdivisionOptions) -> BufferGeometry {
        let refined = catmull_clark(self.subdivision_cage(), levels, options);
        let mut index = Vec::with_capacity(refined.indices.len() / 4 * 6);
        let mut polygons = Vec::with_capacity(refined.counts.len());
        let mut at = 0usize;
        for &n in &refined.counts {
            let n = n as usize;
            if n >= 3 && at + n <= refined.indices.len() {
                let face = &refined.indices[at..at + n];
                for k in 1..n - 1 {
                    index.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                }
                polygons.push(n as u32 - 2);
            }
            at += n;
        }
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(refined.positions.iter().flatten().copied().collect(), 3),
        );
        g.set_index(index);
        crate::utils::compute_vertex_normals(&mut g);
        g.set_polygons(polygons);
        g
    }

    /// The control cage [`subdivide`](Self::subdivide) refines: one face per
    /// [`faces`](Self::faces) entry, vertices welded by position (numbered in
    /// order of first use; `-0` and `0` are one).
    pub fn subdivision_cage(&self) -> Cage {
        let Some(position) = self.get_attribute("position").filter(|p| p.item_size == 3) else {
            return Cage::default();
        };
        let p = &position.array;
        let vertices = position.count();
        let canonical = |x: f32| if x == 0.0 { 0u32 } else { x.to_bits() };
        let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
        let mut weld: Vec<Option<u32>> = vec![None; vertices];
        let mut cage = Cage::default();
        for face in self.faces() {
            if face.iter().any(|&v| v as usize >= vertices) {
                continue;
            }
            cage.counts.push(face.len() as u32);
            for v in face {
                let v = v as usize;
                let id = *weld[v].get_or_insert_with(|| {
                    let at = [p[v * 3], p[v * 3 + 1], p[v * 3 + 2]];
                    *ids.entry(at.map(canonical)).or_insert_with(|| {
                        cage.positions.push(at);
                        cage.positions.len() as u32 - 1
                    })
                });
                cage.indices.push(id);
            }
        }
        cage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quads_everywhere(g: &BufferGeometry) {
        let counts = g.polygon_counts().expect("grouped");
        assert!(counts.iter().all(|&c| c == 2), "every face two triangles");
        for r in g.polygon_ranges() {
            assert_eq!(
                g.polygon_loop(r).map(|c| c.len()),
                Some(4),
                "every face a quad"
            );
        }
    }

    #[test]
    fn a_box_refines_to_24_then_96_quads() {
        let cube = crate::geometries::BoxGeometry::new(2.0, 2.0, 2.0);
        // 24 split vertices weld to the 8 corners of one closed cage.
        let cage = cube.subdivision_cage();
        assert_eq!(cage.positions.len(), 8);
        assert_eq!(cage.counts, vec![4; 6]);
        let one = cube.subdivide(1);
        assert_eq!(one.polygon_counts().unwrap().len(), 24);
        assert_eq!(one.get_attribute("position").unwrap().count(), 8 + 12 + 6);
        quads_everywhere(&one);
        let two = cube.subdivide(2);
        assert_eq!(two.polygon_counts().unwrap().len(), 96);
        quads_everywhere(&two);
        assert!(two.get_attribute("normal").is_some());
        assert!(two.get_attribute("uv").is_none(), "dropped, see the docs");
    }

    /// The limit of a cube is a rounded blob inside it: every vertex strictly
    /// inside, the whole thing symmetric under the cube's own symmetries, and
    /// much rounder than the cube (a cube's corner-to-face-centre ratio is √3).
    #[test]
    fn a_cube_rounds_toward_a_sphere_inside_itself() {
        let g = crate::geometries::BoxGeometry::new(2.0, 2.0, 2.0).subdivide(3);
        let p: Vec<[f32; 3]> = g
            .get_attribute("position")
            .unwrap()
            .array
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let has = |q: [f32; 3]| p.iter().any(|v| (0..3).all(|i| (v[i] - q[i]).abs() < 1e-5));
        let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
        for v in &p {
            assert!(v.iter().all(|c| c.abs() < 1.0), "{v:?} not strictly inside");
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            lo = lo.min(r);
            hi = hi.max(r);
            for mirrored in [[-v[0], v[1], v[2]], [v[0], -v[1], v[2]], [v[1], v[2], v[0]]] {
                assert!(has(mirrored), "{v:?} has no mirror {mirrored:?}");
            }
        }
        assert!(hi / lo < 1.2, "radius ratio {}", hi / lo);
        quads_everywhere(&g);
    }

    /// An open sheet keeps its border: corners stay, the outline stays on the
    /// square's sides, and nothing leaves the square.
    #[test]
    fn an_open_boundary_is_held() {
        let plane = crate::geometries::PlaneGeometry::with_segments(2.0, 2.0, 2, 2);
        let g = plane.subdivide(2);
        quads_everywhere(&g);
        let p = &g.get_attribute("position").unwrap().array;
        for c in [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
            assert!(
                p.chunks_exact(3).any(|v| v[0] == c[0] && v[1] == c[1]),
                "corner {c:?} moved"
            );
        }
        assert!(p
            .chunks_exact(3)
            .all(|v| v[0].abs() <= 1.0 && v[1].abs() <= 1.0 && v[2] == 0.0));
        // Each outline edge (used by one quad) lies on a side of the square.
        let edges = g.polygon_edges();
        let mut uses = HashMap::new();
        for r in g.polygon_ranges() {
            let c = g.polygon_loop(r).unwrap();
            for k in 0..4 {
                *uses.entry(key(c[k], c[(k + 1) % 4])).or_insert(0) += 1;
            }
        }
        let border: Vec<_> = edges
            .iter()
            .filter(|e| uses[&key(e[0], e[1])] == 1)
            .collect();
        assert_eq!(border.len(), 4 * 2 * 4, "sides × cells × 4 per cell side");
        for e in border {
            let (a, b) = (e[0] as usize * 3, e[1] as usize * 3);
            let on_side = (p[a] == p[b] && p[a].abs() == 1.0)
                || (p[a + 1] == p[b + 1] && p[a + 1].abs() == 1.0);
            assert!(on_side, "border edge {e:?} left the square");
        }
    }

    /// Triangles refine as Catmull–Clark on triangles: three quads each.
    #[test]
    fn triangles_become_three_quads_each() {
        let mut tet = BufferGeometry::new();
        tet.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
                ],
                3,
            ),
        );
        tet.set_index(vec![0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2]);
        let g = tet.subdivide(1);
        assert_eq!(g.polygon_counts().unwrap().len(), 12);
        quads_everywhere(&g);
    }

    /// The geometry entry point is the cage function on the welded cage, and
    /// both are deterministic.
    #[test]
    fn matches_the_cage_function_and_repeats_exactly() {
        let sphere = crate::geometries::SphereGeometry::new(1.0, 8, 6);
        let a = sphere.subdivide(2);
        let b = sphere.subdivide(2);
        assert_eq!(
            a.get_attribute("position").unwrap().array,
            b.get_attribute("position").unwrap().array
        );
        assert_eq!(a.index, b.index);
        let options = SubdivisionOptions {
            sharp_boundary: true,
            ..Default::default()
        };
        let direct = catmull_clark(sphere.subdivision_cage(), 2, &options);
        let flat: Vec<f32> = direct.positions.iter().flatten().copied().collect();
        assert_eq!(a.get_attribute("position").unwrap().array, flat);
        quads_everywhere(&a);
    }

    #[test]
    fn level_zero_is_the_welded_cage() {
        let g = crate::geometries::BoxGeometry::new(1.0, 1.0, 1.0).subdivide(0);
        assert_eq!(g.get_attribute("position").unwrap().count(), 8);
        quads_everywhere(&g);
        assert!(BufferGeometry::new().subdivide(2).triangle_count() == 0);
    }
}
