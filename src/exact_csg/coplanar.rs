//! Coplanar face overlap — the refinement primitive for coincident faces (M2).
//!
//! When a triangle of A is exactly coplanar with a triangle of B, they can't
//! cross transversally (no `tri_tri_segment`); instead they *overlap* in a 2D
//! region of their shared plane. `coplanar_clip` returns that overlap
//! (triangulated, in 3D). Its boundary is what the refinement inserts so the
//! coincident region can be classified by the aligned/opposite rule (next step)
//! rather than the ambiguous winding number.
//!
//! Clipping decisions use exact [`orient2d`], so the combinatorics are robust;
//! only the intersection *coordinates* are f64.

use super::predicates::{coplanar, orient2d};
use super::{cross, dot, normalize, sub, V3};

fn signed_area2(t: &[[f64; 2]; 3]) -> f64 {
    orient2d(t[0], t[1], t[2])
}

/// Intersection point of line `a→b` with segment `p→q` (assumed to cross it).
fn line_seg_x(a: [f64; 2], b: [f64; 2], p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    let d1 = orient2d(a, b, p);
    let d2 = orient2d(a, b, q);
    let t = d1 / (d1 - d2);
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/// Sutherland–Hodgman: clip convex `poly` to the left half-plane of `a→b`.
fn clip_by_edge(poly: &[[f64; 2]], a: [f64; 2], b: [f64; 2]) -> Vec<[f64; 2]> {
    let n = poly.len();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let inside = |p: [f64; 2]| orient2d(a, b, p) >= 0.0;
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let (cin, pin) = (inside(cur), inside(prev));
        if cin {
            if !pin {
                out.push(line_seg_x(a, b, prev, cur));
            }
            out.push(cur);
        } else if pin {
            out.push(line_seg_x(a, b, prev, cur));
        }
    }
    out
}

/// Overlap **polygon** of two coplanar triangles as ordered 3D vertices (in
/// `a`'s plane). Empty if not coplanar or no overlap. Its edges are the
/// constraints `refine_mesh` inserts to split a face along a coincident boundary.
pub fn coplanar_overlap_poly(a: &[V3; 3], b: &[V3; 3]) -> Vec<V3> {
    if !coplanar(a, b) {
        return Vec::new();
    }
    let n = cross(sub(a[1], a[0]), sub(a[2], a[0]));
    if dot(n, n) < 1e-18 {
        return Vec::new();
    }
    let origin = a[0];
    let ux = normalize(sub(a[1], a[0]));
    let uy = normalize(cross(n, ux));
    let to2 = |p: V3| [dot(sub(p, origin), ux), dot(sub(p, origin), uy)];

    let a2 = [to2(a[0]), to2(a[1]), to2(a[2])];
    let mut clip = [to2(b[0]), to2(b[1]), to2(b[2])];
    if signed_area2(&clip) < 0.0 {
        clip.reverse(); // clip triangle must be CCW so "inside = left"
    }

    let mut poly: Vec<[f64; 2]> = a2.to_vec();
    for e in 0..3 {
        if poly.is_empty() {
            break;
        }
        poly = clip_by_edge(&poly, clip[e], clip[(e + 1) % 3]);
    }
    if poly.len() < 3 {
        return Vec::new();
    }
    poly.iter()
        .map(|q| {
            [
                origin[0] + q[0] * ux[0] + q[1] * uy[0],
                origin[1] + q[0] * ux[1] + q[1] * uy[1],
                origin[2] + q[0] * ux[2] + q[1] * uy[2],
            ]
        })
        .collect()
}

/// Overlap of two **coplanar** triangles, triangulated and lifted to 3D with
/// `a`'s winding. Empty if they are not coplanar or do not overlap.
pub fn coplanar_clip(a: &[V3; 3], b: &[V3; 3]) -> Vec<[V3; 3]> {
    let poly = coplanar_overlap_poly(a, b);
    if poly.len() < 3 {
        return Vec::new();
    }
    let n = cross(sub(a[1], a[0]), sub(a[2], a[0]));
    let mut out = Vec::new();
    for i in 1..poly.len() - 1 {
        let (p0, p1, p2) = (poly[0], poly[i], poly[i + 1]);
        if dot(cross(sub(p1, p0), sub(p2, p0)), n) >= 0.0 {
            out.push([p0, p1, p2]);
        } else {
            out.push([p0, p2, p1]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area3(t: &[V3; 3]) -> f64 {
        0.5 * {
            let c = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            dot(c, c).sqrt()
        }
    }
    fn total(ov: &[[V3; 3]]) -> f64 {
        ov.iter().map(area3).sum()
    }

    #[test]
    fn contained_overlap_is_the_inner_triangle() {
        let a = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]; // area 8
        let b = [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]]; // area 0.5, inside a
        assert!((total(&coplanar_clip(&a, &b)) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn partial_overlap_area() {
        // a: x+y≤2 quadrant triangle; b: below y=x. Overlap = (0,0),(2,0),(1,1), area 1.
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let b = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]];
        assert!((total(&coplanar_clip(&a, &b)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disjoint_coplanar_is_empty() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0]];
        assert!(coplanar_clip(&a, &b).is_empty());
    }

    #[test]
    fn non_coplanar_is_empty() {
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let b = [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 0.5, 1.0]]; // tilted out of z=0
        assert!(coplanar_clip(&a, &b).is_empty());
    }
}
