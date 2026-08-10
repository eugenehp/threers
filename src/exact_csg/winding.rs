//! Generalized (solid-angle) winding number — the inside/outside classifier for
//! the arrangement boolean. For a closed, outward-wound triangle mesh the winding
//! number of a point is ≈ ±1 inside and ≈ 0 outside; it degrades gracefully on
//! open/non-manifold input (Barill et al. 2018, via the Van Oosterom–Strackee
//! solid-angle formula).

use super::{cross, dot, norm, sub, V3};

/// Sum of signed solid angles subtended by every triangle, in turns (÷4π).
pub fn winding_number(tris: &[[V3; 3]], p: V3) -> f64 {
    let mut total = 0.0;
    for t in tris {
        let a = sub(t[0], p);
        let b = sub(t[1], p);
        let c = sub(t[2], p);
        let (la, lb, lc) = (norm(a), norm(b), norm(c));
        let numer = dot(a, cross(b, c));
        let denom = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
        total += 2.0 * numer.atan2(denom);
    }
    total / (4.0 * std::f64::consts::PI)
}

/// True when `p` lies inside the solid bounded by `tris`.
pub fn point_in_mesh(tris: &[[V3; 3]], p: V3) -> bool {
    winding_number(tris, p).abs() > 0.5
}
