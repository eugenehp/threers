//! Triangle–triangle transversal-intersection test, used to detect whether two
//! solids' surfaces cross (a "straddle"). Built from Möller–Trumbore
//! segment/triangle tests over all six edges.
//!
//! This detects transversal (edge-piercing) intersections — the cases that force
//! sub-triangulation in the arrangement. It does **not** flag purely coplanar
//! overlaps; those are degenerate cases the exact kernel (M2) will own, and
//! shared-face contact between adjacent operands is deliberately not a straddle.

use super::{cross, dot, sub, V3};

const EPS: f64 = 1e-12;

/// Does segment `p0→p1` intersect triangle `(v0, v1, v2)`?
fn seg_tri(p0: V3, p1: V3, v0: V3, v1: V3, v2: V3) -> bool {
    let dir = sub(p1, p0);
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if det.abs() < EPS {
        return false; // segment parallel to triangle plane
    }
    let inv = 1.0 / det;
    let tvec = sub(p0, v0);
    let u = dot(tvec, pvec) * inv;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return false;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return false;
    }
    let t = dot(e2, qvec) * inv;
    (-EPS..=1.0 + EPS).contains(&t)
}

/// Do triangles `a` and `b` cross transversally?
pub fn tri_tri_intersect(a: &[V3; 3], b: &[V3; 3]) -> bool {
    let a_edges = [(a[0], a[1]), (a[1], a[2]), (a[2], a[0])];
    for (p0, p1) in a_edges {
        if seg_tri(p0, p1, b[0], b[1], b[2]) {
            return true;
        }
    }
    let b_edges = [(b[0], b[1]), (b[1], b[2]), (b[2], b[0])];
    for (p0, p1) in b_edges {
        if seg_tri(p0, p1, a[0], a[1], a[2]) {
            return true;
        }
    }
    false
}
