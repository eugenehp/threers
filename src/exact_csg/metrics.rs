//! Mesh-comparison metrics — **M3 validation harness core**.
//!
//! The four-metric gate for "solid parity" against a reference mesh (e.g. one
//! exported by OpenSCAD/CGAL): both watertight, volumes agree, same Euler
//! characteristic (topology), and near-zero Hausdorff distance (geometry). These
//! operate on triangle soups, so the same `compare` runs on our kernel's output
//! vs a reference STL loaded by `StlLoader` (see `scripts/ci-openscad.sh`).

use super::{cross, dot, is_closed_manifold, sqlen, sub, triangles, V3};
use crate::core::BufferGeometry;
use std::collections::HashSet;

// Weld grid at 1e-4: coarse enough to merge vertices after f32 STL round-trip
// (~2e-6 ulp at model scale ~10s), fine enough to keep distinct features apart.
fn key(p: V3) -> (i64, i64, i64) {
    ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
}

/// Signed volume (divergence theorem).
pub fn volume(tris: &[[V3; 3]]) -> f64 {
    tris.iter().map(|t| dot(t[0], cross(t[1], t[2]))).sum::<f64>() / 6.0
}

/// Euler characteristic `V − E + F` over the welded mesh (2 for a genus-0
/// closed surface; `2 − 2g` in general).
pub fn euler_characteristic(tris: &[[V3; 3]]) -> i64 {
    let mut verts = HashSet::new();
    let mut edges = HashSet::new();
    for t in tris {
        for k in 0..3 {
            verts.insert(key(t[k]));
            let (mut u, mut v) = (key(t[k]), key(t[(k + 1) % 3]));
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            edges.insert((u, v));
        }
    }
    verts.len() as i64 - edges.len() as i64 + tris.len() as i64
}

fn unique_verts(tris: &[[V3; 3]]) -> Vec<V3> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in tris {
        for &v in t {
            if seen.insert(key(v)) {
                out.push(v);
            }
        }
    }
    out
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Closest point on triangle `(a,b,c)` to `p` (Ericson, Real-Time Collision
/// Detection §5.1.5 — Voronoi-region case analysis).
fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    add(add(a, scale(ab, vb * denom)), scale(ac, vc * denom))
}

fn point_tri_dist2(p: V3, t: &[V3; 3]) -> f64 {
    sqlen(sub(p, closest_on_tri(p, t[0], t[1], t[2])))
}

/// Symmetric **point-to-surface** Hausdorff distance — `max` over each mesh's
/// vertices of the nearest distance to the *other mesh's triangles* (not just its
/// vertices). Near-zero for two different tessellations of the same solid, which
/// is exactly the CGAL-vs-ours case.
pub fn hausdorff(a: &[[V3; 3]], b: &[[V3; 3]]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let one_way = |verts: &[V3], tris: &[[V3; 3]]| -> f64 {
        verts
            .iter()
            .map(|p| tris.iter().map(|t| point_tri_dist2(*p, t)).fold(f64::INFINITY, f64::min))
            .fold(0.0, f64::max)
            .sqrt()
    };
    one_way(&unique_verts(a), b).max(one_way(&unique_verts(b), a))
}

/// The four-metric comparison result.
#[derive(Debug)]
pub struct MeshReport {
    pub watertight: bool,
    pub vol_rel_err: f64,
    pub euler_match: bool,
    pub hausdorff: f64,
    pub pass: bool,
}

/// Compare `ours` against a `reference` mesh with volume and Hausdorff tolerances.
pub fn compare(
    ours: &BufferGeometry,
    reference: &BufferGeometry,
    vol_tol: f64,
    haus_tol: f64,
) -> MeshReport {
    let (to, tr) = (triangles(ours), triangles(reference));
    let watertight = is_closed_manifold(&to) && is_closed_manifold(&tr);
    let (vo, vr) = (volume(&to).abs(), volume(&tr).abs());
    let vol_rel_err = if vr > 1e-12 { (vo - vr).abs() / vr } else { vo };
    let euler_match = euler_characteristic(&to) == euler_characteristic(&tr);
    let hausdorff = hausdorff(&to, &tr);
    let pass = watertight && vol_rel_err < vol_tol && euler_match && hausdorff < haus_tol;
    MeshReport { watertight, vol_rel_err, euler_match, hausdorff, pass }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cube, sphere};

    #[test]
    fn euler_of_a_closed_box_is_two() {
        let g = cube([2.0, 2.0, 2.0]).to_geometry();
        assert_eq!(euler_characteristic(&triangles(&g)), 2, "genus-0 closed surface");
    }

    #[test]
    fn hausdorff_of_identical_mesh_is_zero() {
        let g = cube([2.0, 3.0, 1.0]).to_geometry();
        let t = triangles(&g);
        assert!(hausdorff(&t, &t) < 1e-9);
    }

    #[test]
    fn hausdorff_measures_offset() {
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = cube([2.0, 2.0, 2.0]).translate([0.1, 0.0, 0.0]).to_geometry();
        let h = hausdorff(&triangles(&a), &triangles(&b));
        assert!((h - 0.1).abs() < 1e-6, "shift of 0.1 → Hausdorff 0.1, got {h}");
    }

    #[test]
    fn hausdorff_zero_across_retessellation() {
        // Same flat square, two tessellations. Point-to-surface distance is 0,
        // whereas vertex-only would report ~0.7 (the added centre vertex).
        let plain = [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        let c = [0.5, 0.5, 0.0];
        let fan = [
            [c, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [c, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [c, [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [c, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        assert!(hausdorff(&plain, &fan) < 1e-9, "retessellation of the same surface → 0");
    }

    #[test]
    fn compare_passes_identical_and_fails_different() {
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        assert!(compare(&a, &a, 1e-6, 1e-6).pass, "identical meshes must pass");

        let big = cube([2.2, 2.0, 2.0]).to_geometry(); // 10% wider → volume differs
        let r = compare(&a, &big, 1e-3, 1e-6);
        assert!(!r.pass && r.vol_rel_err > 0.05, "different volume must fail");

        // A sphere vs the box: same watertightness but wrong topology/geometry.
        let s = sphere(1.0).to_geometry();
        assert!(!compare(&a, &s, 1e-3, 1e-6).pass);
    }
}

