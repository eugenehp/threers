//! Patch boundaries, extracted from a tagged mesh and shared between the faces
//! that meet along them.
//!
//! This is the piece that makes re-tessellation *watertight* across more than
//! one surface, and it is the core of Stage 3a: tessellate each shared boundary
//! **once**, then constrain every face to it.
//!
//! # Why re-meshing each patch independently cannot work
//!
//! A cylinder's side sampled at 40 angular steps against a cap rim sampled at 31
//! leaves a ring of T-junctions. The two faces agree about *where* their shared
//! edge is — both derive it from the same surfaces — but not about which points
//! along it to use, and a mesh is made of points. No amount of tightening the
//! tolerance fixes that; the boundary has to be sampled once and used twice.
//!
//! # What a boundary is, here
//!
//! An edge of the input mesh whose two triangles carry *different* surfaces.
//! That is exactly the seam between two faces, and it comes for free with
//! provenance — without it, "which edges are face boundaries" is precisely the
//! question a triangle soup cannot answer.
//!
//! Chained into polylines, these become the shared constraint every adjacent
//! patch triangulates against.

use std::collections::HashMap;

use crate::core::BufferGeometry;
use crate::nurbs::v3;
use crate::nurbs::V3;

use super::table::{triangle_indices, triangle_vertices};
use super::SurfaceTable;

/// A boundary shared by two surface patches.
#[derive(Debug, Clone)]
pub struct Boundary {
    /// The two surfaces that meet here.
    pub surfaces: (usize, usize),
    /// The polyline, in order. Closed loops repeat their first point last.
    pub points: Vec<V3>,
    pub closed: bool,
}

/// The mesh's coordinate extent, as a scale for tolerances.
fn mesh_extent(geometry: &BufferGeometry) -> f64 {
    let Some(pos) = geometry.get_attribute("position") else {
        return 1.0;
    };
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &c in &pos.array {
        lo = lo.min(c as f64);
        hi = hi.max(c as f64);
    }
    if lo.is_finite() {
        (hi - lo).max(1.0)
    } else {
        1.0
    }
}

/// Quantized key so a vertex arrived at from two triangles is one vertex.
fn key(p: V3, quantum: f64) -> [i64; 3] {
    [
        (p[0] / quantum).round() as i64,
        (p[1] / quantum).round() as i64,
        (p[2] / quantum).round() as i64,
    ]
}

/// Extract every patch boundary of a tagged mesh, chained into polylines.
///
/// Boundaries are keyed by the *unordered* surface pair, so the same seam is one
/// boundary however the two faces are ordered — which is what lets both of them
/// look it up.
pub fn patch_boundaries(geometry: &BufferGeometry, table: &SurfaceTable) -> Vec<Boundary> {
    // The weld quantum is derived from the mesh, not passed in.
    //
    // It has to absorb f32 drift: a cylinder's seam column is generated twice, at
    // `θ = 0` and `θ = 2π`, and `sin(2π)` in f32 is −1.7e-7 rather than zero — so
    // the two copies of a seam vertex sit 3.5e-7 apart at radius 2. A quantum of
    // 1e-9 keys them differently and the rim loop silently fails to close, which
    // is the sort of parameter no caller should be asked to guess. Scaled to the
    // model it is comfortably below any real vertex spacing.
    let quantum = mesh_extent(geometry) * 1e-5;
    // Edge -> the surfaces of the triangles using it.
    let mut edge_surfaces: HashMap<([i64; 3], [i64; 3]), Vec<usize>> = HashMap::new();
    let mut edge_points: HashMap<([i64; 3], [i64; 3]), (V3, V3)> = HashMap::new();

    for tri in 0..table.triangle_count() {
        let (Some(verts), Some(_)) = (
            triangle_vertices(geometry, tri),
            triangle_indices(geometry, tri),
        ) else {
            continue;
        };
        let Some(si) = table.surface_index_of(tri) else {
            continue;
        };
        for k in 0..3 {
            let (a, b) = (verts[k], verts[(k + 1) % 3]);
            let (ka, kb) = (key(a, quantum), key(b, quantum));
            let ek = if ka <= kb { (ka, kb) } else { (kb, ka) };
            edge_surfaces.entry(ek).or_default().push(si);
            edge_points
                .entry(ek)
                .or_insert(if ka <= kb { (a, b) } else { (b, a) });
        }
    }

    // A boundary edge is one whose triangles disagree about their surface.
    let mut by_pair: HashMap<(usize, usize), Vec<(V3, V3)>> = HashMap::new();
    for (ek, surfaces) in &edge_surfaces {
        let mut distinct: Vec<usize> = surfaces.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != 2 {
            continue;
        }
        let pair = (distinct[0], distinct[1]);
        if let Some(&seg) = edge_points.get(ek) {
            by_pair.entry(pair).or_default().push(seg);
        }
    }

    let mut out = Vec::new();
    for (pair, segments) in by_pair {
        for (points, closed) in chain(&segments, quantum) {
            if points.len() >= 2 {
                out.push(Boundary {
                    surfaces: pair,
                    points,
                    closed,
                });
            }
        }
    }
    // Deterministic order: callers index these and the output must not depend on
    // hash iteration.
    out.sort_by(|a, b| {
        a.surfaces
            .cmp(&b.surfaces)
            .then_with(|| a.points.len().cmp(&b.points.len()))
            .then_with(|| {
                a.points[0]
                    .partial_cmp(&b.points[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    out
}

/// Chain unordered segments into polylines.
///
/// A patch boundary is usually a single closed loop (a cylinder's rim) but need
/// not be — a partial revolution leaves open arcs, and a face can border the
/// same neighbour in several places.
fn chain(segments: &[(V3, V3)], quantum: f64) -> Vec<(Vec<V3>, bool)> {
    let mut adjacency: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, (a, b)) in segments.iter().enumerate() {
        adjacency.entry(key(*a, quantum)).or_default().push(i);
        adjacency.entry(key(*b, quantum)).or_default().push(i);
    }

    let mut used = vec![false; segments.len()];
    let mut out = Vec::new();

    // Open chains first: starting from an endpoint used by exactly one segment
    // guarantees the walk covers the whole chain rather than starting mid-way.
    //
    // Sorted, because `adjacency` is a `HashMap` and callers index the result —
    // an order that varies run to run is a reproducibility bug waiting to be
    // blamed on the geometry.
    let mut starts: Vec<usize> = adjacency
        .iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(_, v)| v[0])
        .collect();
    starts.sort_unstable();
    starts.dedup();

    let walk = |seed: usize, used: &mut Vec<bool>, from_end: bool| -> Option<(Vec<V3>, bool)> {
        if used[seed] {
            return None;
        }
        used[seed] = true;
        let (a, b) = segments[seed];
        let mut points = if from_end { vec![b, a] } else { vec![a, b] };
        loop {
            let tail = *points.last().unwrap();
            let tk = key(tail, quantum);
            let next = adjacency
                .get(&tk)
                .into_iter()
                .flatten()
                .copied()
                .find(|&i| !used[i]);
            let Some(i) = next else { break };
            used[i] = true;
            let (p, q) = segments[i];
            points.push(if key(p, quantum) == tk { q } else { p });
        }
        let closed =
            points.len() > 2 && key(points[0], quantum) == key(*points.last().unwrap(), quantum);
        if closed {
            // Rotate to start at the lexicographically smallest point. A closed
            // loop has no intrinsic start, and leaving it wherever the walk began
            // makes the output depend on hash order even once the seeds are
            // sorted — the loop is entered from whichever segment came first.
            let n = points.len() - 1; // the last repeats the first
            let pivot = (0..n)
                .min_by(|&i, &j| {
                    points[i]
                        .partial_cmp(&points[j])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let mut rotated: Vec<V3> = points[pivot..n].to_vec();
            rotated.extend_from_slice(&points[..pivot]);
            rotated.push(rotated[0]);
            points = rotated;
        }
        Some((points, closed))
    };

    for seed in starts {
        if let Some(chain) = walk(seed, &mut used, false) {
            out.push(chain);
        }
    }
    for seed in 0..segments.len() {
        if let Some(chain) = walk(seed, &mut used, false) {
            out.push(chain);
        }
    }
    out
}

/// Resample a boundary polyline so its chords stay within `tolerance` of both
/// surfaces, keeping the original vertices.
///
/// The original vertices are kept rather than replaced: they are where the two
/// faces already agree, and moving them would be the one way this could open a
/// seam it was meant to close. New points are only ever *inserted* between them,
/// on the segment, pushed onto the surfaces.
pub fn densify_to(
    boundary: &Boundary,
    a: &super::Surface,
    b: &super::Surface,
    tolerance: f64,
) -> Vec<V3> {
    let mut out = Vec::with_capacity(boundary.points.len());
    for w in boundary.points.windows(2) {
        out.push(w[0]);
        subdivide(w[0], w[1], a, b, tolerance, 0, &mut out);
    }
    if let Some(last) = boundary.points.last() {
        out.push(*last);
    }
    out
}

/// Insert points along `p → q` until the chord is within `tolerance` of both
/// surfaces. Bounded, because a boundary the surfaces do not actually share
/// would otherwise subdivide forever.
fn subdivide(
    p: V3,
    q: V3,
    a: &super::Surface,
    b: &super::Surface,
    tolerance: f64,
    depth: usize,
    out: &mut Vec<V3>,
) {
    const MAX_DEPTH: usize = 10;
    if depth >= MAX_DEPTH {
        return;
    }
    let mid = v3::scale(v3::add(p, q), 0.5);
    if a.distance(mid).max(b.distance(mid)) <= tolerance {
        return;
    }
    let Some(m) = push_to_both(mid, a, b) else {
        return;
    };
    subdivide(p, m, a, b, tolerance, depth + 1, out);
    out.push(m);
    subdivide(m, q, a, b, tolerance, depth + 1, out);
}

/// Move a point onto both surfaces by alternating projection.
///
/// Cheap and adequate: the input is already within a tessellation chord of both,
/// so a handful of alternations converges. Returns `None` if it does not, and
/// the caller keeps the chord — a boundary point that is not on both surfaces
/// would be worse than a slightly coarse one.
fn push_to_both(p: V3, a: &super::Surface, b: &super::Surface) -> Option<V3> {
    let mut q = p;
    for _ in 0..16 {
        let (da, db) = (a.distance(q), b.distance(q));
        if da < 1e-12 && db < 1e-12 {
            return Some(q);
        }
        q = closest_on(&q, a)?;
        q = closest_on(&q, b)?;
    }
    let (da, db) = (a.distance(q), b.distance(q));
    (da < 1e-6 && db < 1e-6).then_some(q)
}

fn closest_on(p: &V3, s: &super::Surface) -> Option<V3> {
    let (u, v) = s.invert(*p)?;
    Some(s.point(u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::{BoxGeometry, CylinderGeometry, SphereGeometry};

    const TAU: f32 = std::f32::consts::PI * 2.0;

    #[test]
    fn a_capped_cylinder_has_two_rim_boundaries() {
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let table = g.surface_table().unwrap();
        let b = patch_boundaries(&g, table);

        assert_eq!(b.len(), 2, "one rim per cap");
        for boundary in &b {
            assert!(boundary.closed, "a rim is a closed loop");
            // 24 segments, so 25 points with the first repeated.
            assert_eq!(boundary.points.len(), 25);
            // Every point is on the cylinder *and* the cap it bounds.
            let (sa, sb) = boundary.surfaces;
            for p in &boundary.points {
                assert!(table.surfaces()[sa].distance(*p) < 1e-5);
                assert!(table.surfaces()[sb].distance(*p) < 1e-5);
            }
        }
    }

    #[test]
    fn a_box_has_twelve_edges() {
        let g = BoxGeometry::new(2.0, 3.0, 4.0);
        let table = g.surface_table().unwrap();
        let b = patch_boundaries(&g, table);
        assert_eq!(b.len(), 12, "a box has twelve edges");
        for boundary in &b {
            assert!(!boundary.closed);
            assert_eq!(boundary.points.len(), 2, "a box edge is one segment");
        }
    }

    #[test]
    fn a_single_surface_mesh_has_no_boundaries() {
        // A sphere is one patch, so there is nothing to stitch.
        let g = SphereGeometry::new(1.0, 16, 8);
        let table = g.surface_table().unwrap();
        assert!(patch_boundaries(&g, table).is_empty());
    }

    #[test]
    fn an_open_cylinder_has_no_boundaries() {
        let g = CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, true, 0.0, TAU);
        let table = g.surface_table().unwrap();
        assert!(patch_boundaries(&g, table).is_empty());
    }

    #[test]
    fn boundaries_are_deterministic() {
        // They are indexed by the caller, so hash iteration order must not leak.
        let g = CylinderGeometry::new(1.0, 1.0, 2.0, 12, 1, false, 0.0, TAU);
        let table = g.surface_table().unwrap();
        let first = patch_boundaries(&g, table);
        for _ in 0..8 {
            let again = patch_boundaries(&g, table);
            assert_eq!(first.len(), again.len());
            for (a, b) in first.iter().zip(&again) {
                assert_eq!(a.surfaces, b.surfaces);
                assert_eq!(a.points.len(), b.points.len());
                assert!(v3::dist(a.points[0], b.points[0]) < 1e-15);
            }
        }
    }

    #[test]
    fn densifying_keeps_the_original_points() {
        // The originals are where the two faces already agree; moving them is the
        // one way this could open a seam it was meant to close.
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 8, 1, false, 0.0, TAU);
        let table = g.surface_table().unwrap();
        let b = patch_boundaries(&g, table);
        let boundary = &b[0];
        let (sa, sb) = boundary.surfaces;
        let dense = densify_to(boundary, &table.surfaces()[sa], &table.surfaces()[sb], 1e-4);

        assert!(
            dense.len() >= boundary.points.len(),
            "densify removed points"
        );
        for p in &boundary.points {
            assert!(
                dense.iter().any(|q| v3::dist(*p, *q) < 1e-12),
                "an original boundary point was dropped"
            );
        }
        // And every inserted point really is on both surfaces.
        for p in &dense {
            assert!(table.surfaces()[sa].distance(*p) < 1e-5);
            assert!(table.surfaces()[sb].distance(*p) < 1e-5);
        }
    }
}
