//! Arrangement stage — step 2 (substrate): incremental point-insertion
//! triangulation. Inserts Steiner points into a triangle (each interior point
//! splits its containing sub-triangle 1→3), producing a valid triangulation that
//! includes every point as a vertex.
//!
//! This is the vertex-insertion half of the constrained triangulation. The other
//! half — recovering the cut *segments* as triangulation edges (edge flips) so no
//! sub-triangle straddles a constraint — is the remaining M1 step. Points are
//! assumed in general position (no point landing exactly on an edge created by an
//! earlier insertion); exact-arithmetic handling of on-edge/degenerate insertion
//! is M2.

use super::{dot, sub, V3};

/// Barycentric coordinates `[u, v, w]` of `p` in triangle `t` (p assumed
/// coplanar with t). `None` if the triangle is degenerate.
fn bary(t: &[V3; 3], p: V3) -> Option<[f64; 3]> {
    let v0 = sub(t[1], t[0]);
    let v1 = sub(t[2], t[0]);
    let v2 = sub(p, t[0]);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-18 {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    Some([1.0 - v - w, v, w])
}

/// Triangulate `tri` so it includes each point of `pts` as a vertex. Each point
/// splits the sub-triangle that contains it into three, preserving winding.
/// Points outside `tri` are ignored.
pub fn triangulate_with_points(tri: &[V3; 3], pts: &[V3]) -> Vec<[V3; 3]> {
    let mut tris = vec![*tri];
    const EPS: f64 = 1e-9;
    for &p in pts {
        let idx = tris.iter().position(|t| {
            matches!(bary(t, p), Some([u, v, w]) if u >= -EPS && v >= -EPS && w >= -EPS)
        });
        let Some(i) = idx else { continue };
        let t = tris.swap_remove(i);
        tris.push([t[0], t[1], p]);
        tris.push([t[1], t[2], p]);
        tris.push([t[2], t[0], p]);
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::super::cross;
    use super::*;

    fn area(t: &[V3; 3]) -> f64 {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        0.5 * dot(n, n).sqrt()
    }
    fn normal(t: &[V3; 3]) -> V3 {
        cross(sub(t[1], t[0]), sub(t[2], t[0]))
    }

    const T: [V3; 3] = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];

    #[test]
    fn single_interior_point_splits_into_three() {
        let out = triangulate_with_points(&T, &[[1.0, 1.0, 0.0]]);
        assert_eq!(out.len(), 3);
        assert!((out.iter().map(area).sum::<f64>() - area(&T)).abs() < 1e-9);
        let n0 = normal(&T);
        for s in &out {
            assert!(dot(n0, normal(s)) > 0.0, "winding preserved");
            assert!(area(s) > 1e-9, "no degenerate slivers");
        }
    }

    #[test]
    fn two_points_give_euler_count_and_area() {
        // Two general-position interior points → 1 + 2·2 = 5 triangles.
        let out = triangulate_with_points(&T, &[[1.0, 1.0, 0.0], [2.0, 0.5, 0.0]]);
        assert_eq!(out.len(), 5);
        assert!((out.iter().map(area).sum::<f64>() - area(&T)).abs() < 1e-9);
    }

    #[test]
    fn points_appear_as_vertices() {
        let p = [1.0, 1.0, 0.0];
        let out = triangulate_with_points(&T, &[p]);
        let has = |q: V3| out.iter().flatten().any(|v| dot(sub(*v, q), sub(*v, q)) < 1e-18);
        assert!(has(p) && has(T[0]) && has(T[1]) && has(T[2]));
    }

    #[test]
    fn point_outside_is_ignored() {
        let out = triangulate_with_points(&T, &[[9.0, 9.0, 0.0]]);
        assert_eq!(out.len(), 1);
    }
}
