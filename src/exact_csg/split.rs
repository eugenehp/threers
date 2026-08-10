//! Arrangement stage — step 2 (atom): the conforming triangle/plane split.
//!
//! Cuts a triangle by a plane into sub-triangles, each of which lies wholly on
//! one side (so it never straddles the cut and the winding-number classifier can
//! label it by a single centroid test). Winding is preserved. This is the
//! primitive that splits a triangle against a cutter triangle's supporting plane.
//!
//! What this is **not** yet: the bounded-segment refinement. A real cut is the
//! *bounded* segment `tri_tri_segment` returns, not the full plane — so a
//! complete re-triangulation must clip splits to the cutter's extent (creating
//! interior T-junctions) and share cut vertices across both meshes so the seam is
//! watertight. That constrained triangulation is the remaining M1/M2 work.

use super::{dot, V3};

const EPS: f64 = 1e-12;

fn lerp(p: V3, q: V3, t: f64) -> V3 {
    [
        p[0] + (q[0] - p[0]) * t,
        p[1] + (q[1] - p[1]) * t,
        p[2] + (q[2] - p[2]) * t,
    ]
}

/// Split `tri` by plane `dot(n, x) = d` into sub-triangles that each lie on a
/// single side. Returns the triangle unchanged if it doesn't cleanly cross the
/// plane. A vertex exactly on the plane is a degenerate touch and is not split
/// (deferred to the M2 exact predicates).
pub fn split_triangle_by_plane(tri: &[V3; 3], n: V3, d: f64) -> Vec<[V3; 3]> {
    let dist = [
        dot(n, tri[0]) - d,
        dot(n, tri[1]) - d,
        dot(n, tri[2]) - d,
    ];
    let pos = dist.iter().filter(|&&x| x > EPS).count();
    let neg = dist.iter().filter(|&&x| x < -EPS).count();
    if pos == 0 || neg == 0 || pos + neg != 3 {
        return vec![*tri]; // no clean 1-vs-2 crossing (parallel, on-side, or on-plane vertex)
    }

    // The "lone" vertex is the one on the minority side; rotate it to the front
    // (a cyclic rotation, so winding is preserved).
    let lone = if pos == 1 {
        dist.iter().position(|&x| x > EPS).unwrap()
    } else {
        dist.iter().position(|&x| x < -EPS).unwrap()
    };
    let (a, b, c) = (tri[lone], tri[(lone + 1) % 3], tri[(lone + 2) % 3]);
    let (da, db, dc) = (dist[lone], dist[(lone + 1) % 3], dist[(lone + 2) % 3]);

    let pab = lerp(a, b, da / (da - db)); // crossing on edge a→b
    let pac = lerp(a, c, da / (da - dc)); // crossing on edge a→c

    vec![
        [a, pab, pac],  // corner triangle at the lone vertex
        [pab, b, c],    // quad half 1
        [pab, c, pac],  // quad half 2
    ]
}

#[cfg(test)]
mod tests {
    use super::super::{cross, sub};
    use super::*;

    fn area(t: &[V3; 3]) -> f64 {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        0.5 * dot(n, n).sqrt()
    }
    fn normal(t: &[V3; 3]) -> V3 {
        cross(sub(t[1], t[0]), sub(t[2], t[0]))
    }

    #[test]
    fn no_split_when_plane_misses() {
        let t = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        // Plane y = 5 is entirely above the triangle.
        let out = split_triangle_by_plane(&t, [0.0, 1.0, 0.0], 5.0);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn split_conserves_area_and_conforms() {
        let t = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let (n, d) = ([0.0, 1.0, 0.0], 0.5); // y = 0.5 cuts across it
        let sub = split_triangle_by_plane(&t, n, d);
        assert_eq!(sub.len(), 3, "1-vs-2 crossing → 3 sub-triangles");

        // Areas partition the original.
        let total: f64 = sub.iter().map(area).sum();
        assert!((total - area(&t)).abs() < 1e-9, "area conserved");

        let n0 = normal(&t);
        for s in &sub {
            // Winding preserved: every sub-triangle faces the same way.
            assert!(dot(n0, normal(s)) > 0.0, "winding preserved");
            // Conformance: no sub-triangle straddles the plane.
            let ds: Vec<f64> = s.iter().map(|v| dot(n, *v) - d).collect();
            let all_above = ds.iter().all(|&x| x >= -1e-9);
            let all_below = ds.iter().all(|&x| x <= 1e-9);
            assert!(all_above || all_below, "sub-triangle lies on one side");
        }
    }

    #[test]
    fn oblique_split_conserves_area() {
        let t = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 1.0]];
        let (n, d) = ([1.0, 1.0, 0.0], 2.0); // oblique cutting plane
        let sub = split_triangle_by_plane(&t, n, d);
        let total: f64 = sub.iter().map(area).sum();
        assert!((total - area(&t)).abs() < 1e-9);
    }
}
