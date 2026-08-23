//! Arrangement stage — step 1: the intersection **segments** where two solids'
//! surfaces cross. Each straddling triangle pair contributes the segment along
//! `plane(A) ∩ plane(B)` clipped to both triangles. These segments are the
//! constraints the (still-to-come) per-triangle constrained re-triangulation
//! inserts before classification finishes the boolean.
//!
//! Transversal cuts only; coplanar/degenerate contact is deferred to the exact
//! kernel (M2), consistent with [`super::tri_tri_intersect`].

use super::{aabb, aabb_overlap, cross, dot, sub, triangles, V3};
use crate::core::BufferGeometry;

fn lerp(p: V3, q: V3, t: f64) -> V3 {
    [
        p[0] + (q[0] - p[0]) * t,
        p[1] + (q[1] - p[1]) * t,
        p[2] + (q[2] - p[2]) * t,
    ]
}

/// Unnormalized plane (normal, offset) of a triangle: `dot(n, x) = d` on-plane.
fn plane(tri: &[V3; 3]) -> (V3, f64) {
    let n = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
    (n, dot(n, tri[0]))
}

/// The chord where `tri` crosses plane `(n, d)` — its two edge-crossing points,
/// or `None` unless exactly two edges strictly change side (degenerate cases,
/// incl. a vertex exactly on the plane, are skipped).
fn tri_plane_chord(tri: &[V3; 3], n: V3, d: f64) -> Option<(V3, V3)> {
    let dist = [dot(n, tri[0]) - d, dot(n, tri[1]) - d, dot(n, tri[2]) - d];
    let mut pts: Vec<V3> = Vec::new();
    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
        let (di, dj) = (dist[i], dist[j]);
        if (di > 0.0 && dj < 0.0) || (di < 0.0 && dj > 0.0) {
            let t = di / (di - dj);
            pts.push(lerp(tri[i], tri[j], t));
        }
    }
    if pts.len() == 2 {
        Some((pts[0], pts[1]))
    } else {
        None
    }
}

/// The intersection segment of two transversally-crossing triangles, or `None`
/// if their planes are parallel or their cross-sections don't overlap.
pub fn tri_tri_segment(a: &[V3; 3], b: &[V3; 3]) -> Option<(V3, V3)> {
    let (na, da) = plane(a);
    let (nb, db) = plane(b);
    let dir = cross(na, nb);
    if dot(dir, dir) < 1e-20 {
        return None; // parallel / coplanar planes
    }
    // Each triangle's chord on the *other's* plane lies on the shared line L=dir.
    let ca = tri_plane_chord(a, nb, db)?;
    let cb = tri_plane_chord(b, na, da)?;

    // Parametrize both chords by arc position along L and overlap the intervals.
    let s = |p: V3| dot(p, dir);
    let order = |c: (V3, V3)| {
        let (p, q) = ((s(c.0), c.0), (s(c.1), c.1));
        if p.0 <= q.0 {
            (p, q)
        } else {
            (q, p)
        }
    };
    let (a_lo, a_hi) = order(ca);
    let (b_lo, b_hi) = order(cb);
    let lo = if a_lo.0 >= b_lo.0 { a_lo } else { b_lo };
    let hi = if a_hi.0 <= b_hi.0 { a_hi } else { b_hi };
    if lo.0 > hi.0 + 1e-12 {
        return None; // cross-sections don't overlap
    }
    Some((lo.1, hi.1))
}

/// All surface-intersection segments between two meshes (naive AABB broad
/// phase). Zero-length segments (grazing contact) are dropped.
pub fn intersection_segments(a: &BufferGeometry, b: &BufferGeometry) -> Vec<(V3, V3)> {
    let ta = triangles(a);
    let tb = triangles(b);
    let boxes_b: Vec<(V3, V3)> = tb.iter().map(aabb).collect();
    let mut segs = Vec::new();
    for ai in &ta {
        let ba = aabb(ai);
        for (bj, bb) in tb.iter().zip(&boxes_b) {
            if !aabb_overlap(&ba, bb) {
                continue;
            }
            if let Some(seg) = tri_tri_segment(ai, bj) {
                if dot(sub(seg.1, seg.0), sub(seg.1, seg.0)) > 1e-18 {
                    segs.push(seg);
                }
            }
        }
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube;

    fn approx(p: V3, q: V3) -> bool {
        dot(sub(p, q), sub(p, q)) < 1e-9
    }

    fn is_endpoint(seg: &(V3, V3), p: V3) -> bool {
        approx(seg.0, p) || approx(seg.1, p)
    }

    #[test]
    fn segment_of_two_crossing_triangles() {
        // Triangle A in z=0; triangle B in the y=0.5 plane piercing it.
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let b = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0]];
        let seg = tri_tri_segment(&a, &b).expect("triangles cross");
        // Hand-computed intersection: (0.5,0.5,0) — (1.0,0.5,0).
        assert!(is_endpoint(&seg, [0.5, 0.5, 0.0]));
        assert!(is_endpoint(&seg, [1.0, 0.5, 0.0]));
    }

    #[test]
    fn no_segment_when_apart() {
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let apart = [[0.5, 0.5, 1.0], [0.5, 0.5, 2.0], [1.5, 0.5, 2.0]];
        assert!(tri_tri_segment(&a, &apart).is_none());
    }

    #[test]
    fn overlapping_solids_produce_a_cut_loop() {
        // A sphere poking through a box's faces — a *transversal* intersection
        // (sphere triangles pierce box-face interiors). Note: two axis-aligned
        // boxes would be degenerate (the cut runs along box edges), which the
        // transversal-only stage deliberately skips until M2.
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = crate::sphere(1.3).to_geometry();
        let segs = intersection_segments(&a, &b);
        assert!(
            !segs.is_empty(),
            "sphere through box must yield cut segments"
        );
        // No coincident-endpoint degenerates. (Genuine near-tangent grazes can be
        // legitimately tiny — geometric sliver cleanup is an M2/exact-arith concern.)
        for s in &segs {
            assert!(
                dot(sub(s.1, s.0), sub(s.1, s.0)) > 1e-18,
                "no coincident-point segs"
            );
        }
    }
}
