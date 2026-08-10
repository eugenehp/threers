//! 2D constrained triangulation of one convex triangle with interior/boundary
//! Steiner points and required constraint edges — the linchpin of the arrangement.
//! Incremental point insertion + Sloan-style flip recovery of constraint edges.
//!
//! Correctness is bounded: f64, an iteration cap on recovery, and an assumption
//! that constraints don't cross except at shared endpoints (true for a manifold
//! intersection curve). On any failure it returns `None`, so the caller's
//! manifold/oracle gate can fall back rather than trust a bad triangulation.
//! Robust/degenerate handling is M2 (exact predicates).

pub type Tri = [usize; 3];

/// Largest per-face point count the constrained triangulation will attempt before
/// declining (→ float fallback). Edge recovery is O(crossings·n) with the
/// adjacency map (see `recover_edge`), and point insertion / Delaunay cleanup are
/// O(n²), so faces of many hundreds of points triangulate in milliseconds; this
/// bound is a final safety net against genuinely pathological faces, not the
/// common case.
const MAX_CDT_POINTS: usize = 256;

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    // Exactly-signed orientation — robust point location & flip decisions even
    // on near-collinear inputs (M2). Same sign convention as the cross product.
    super::predicates::orient2d(a, b, c)
}

/// Does open segment `p0p1` properly cross open segment `q0q1`?
fn seg_cross(p0: [f64; 2], p1: [f64; 2], q0: [f64; 2], q1: [f64; 2]) -> bool {
    let d1 = orient(p0, p1, q0);
    let d2 = orient(p0, p1, q1);
    let d3 = orient(q0, q1, p0);
    let d4 = orient(q0, q1, p1);
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

/// `> 0` iff `d` lies strictly inside the circumcircle of CCW triangle `(a,b,c)`.
fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
    let (bx, by) = (b[0] - d[0], b[1] - d[1]);
    let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx)
}

/// Constrained triangulation. `pts[0..3]` are the triangle corners; the rest lie
/// inside or on it. Returns triangles (index triples) conforming to `constraints`,
/// or `None` if recovery fails.
pub fn triangulate_constrained(pts: &[[f64; 2]], constraints: &[[usize; 2]]) -> Option<Vec<Tri>> {
    if pts.len() < 3 {
        return None;
    }
    // Safety net for genuinely huge coplanar faces. Recovery itself is now
    // O(crossings·n) (see `recover_edge`), so faces of a few hundred points
    // triangulate in milliseconds; but folding a boolean *sequence* over such a
    // face (each step re-triangulating a slightly bigger face) still grows fast,
    // so beyond this bound we decline and let the caller fall back to the float
    // kernel for that boolean. The bound sits well above any face the parity
    // corpus builds, so it never changes an exact result there.
    if pts.len() > MAX_CDT_POINTS {
        return None;
    }
    // Seed with the CCW corner triangle.
    let mut tris: Vec<Tri> = vec![if orient(pts[0], pts[1], pts[2]) >= 0.0 {
        [0, 1, 2]
    } else {
        [0, 2, 1]
    }];

    for i in 3..pts.len() {
        insert_point(&mut tris, pts, i)?;
    }
    for &[u, v] in constraints {
        recover_edge(&mut tris, pts, u, v)?;
    }
    // Restore the Delaunay property on non-constraint edges. Incremental insertion +
    // constraint recovery can leave skinny/near-collinear slivers (a vertex almost on
    // a neighbour's edge); the Lawson flips below replace them with well-shaped
    // triangles, which is what keeps a finely-tessellated cut (e.g. a sphere arc) from
    // degenerating into zero-area triangles that later have to be dropped.
    let mut cons: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for &[u, v] in constraints {
        cons.insert((u.min(v), u.max(v)));
    }
    delaunay_flip(&mut tris, pts, &cons);
    Some(tris)
}

/// Lawson flips: while some non-constraint interior edge fails the Delaunay incircle
/// test and its two triangles form a convex quad, flip it. Each pass rebuilds an
/// edge→triangles map (O(n)) and applies one flip, so the whole routine is ~O(n·f)
/// for `f` flips rather than the O(n²)-per-flip of a linear neighbour scan. A clear
/// incircle margin is required to flip, which prevents float-noise oscillation.
fn delaunay_flip(tris: &mut Vec<Tri>, pts: &[[f64; 2]], cons: &std::collections::HashSet<(usize, usize)>) {
    let cap = 8 * tris.len() + 64; // total-flip bound (Delaunay converges well within this)
    for _ in 0..cap {
        // Edge → the (up to two) triangles sharing it: (tri0, tri1, count).
        let mut adj: std::collections::HashMap<(usize, usize), (usize, usize, i32)> =
            std::collections::HashMap::with_capacity(tris.len() * 3);
        for (k, t) in tris.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let slot = adj.entry((a.min(b), a.max(b))).or_insert((k, 0, 0));
                if slot.2 == 1 {
                    slot.1 = k; // second incident triangle
                }
                slot.2 += 1;
            }
        }
        let mut did = false;
        for (&(a, b), &(k1, k2, count)) in &adj {
            if count != 2 || cons.contains(&(a, b)) {
                continue;
            }
            let (Some(c), Some(d)) = (apex(tris[k1], a, b), apex(tris[k2], a, b)) else { continue };
            let convex = orient(pts[a], pts[b], pts[c]) * orient(pts[a], pts[b], pts[d]) < 0.0
                && orient(pts[c], pts[d], pts[a]) * orient(pts[c], pts[d], pts[b]) < 0.0;
            if !convex {
                continue;
            }
            let ccw_tri = ccw([a, b, c], pts);
            if in_circle(pts[ccw_tri[0]], pts[ccw_tri[1]], pts[ccw_tri[2]], pts[d]) > 1e-9 {
                let (hi, lo) = (k1.max(k2), k1.min(k2));
                tris.swap_remove(hi);
                tris.swap_remove(lo);
                tris.push(ccw([c, d, a], pts));
                tris.push(ccw([d, c, b], pts));
                did = true;
                break;
            }
        }
        if !did {
            return;
        }
    }
}

fn insert_point(tris: &mut Vec<Tri>, pts: &[[f64; 2]], i: usize) -> Option<()> {
    let p = pts[i];
    const EPS: f64 = 1e-9;
    // Locate: triangle that contains p (strictly inside, or on an edge).
    let mut host = None;
    for (k, t) in tris.iter().enumerate() {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        let (oa, ob, oc) = (orient(a, b, p), orient(b, c, p), orient(c, a, p));
        if oa >= -EPS && ob >= -EPS && oc >= -EPS {
            host = Some((k, oa < EPS, ob < EPS, oc < EPS));
            break;
        }
    }
    let (k, on_ab, on_bc, on_ca) = host?;
    let t = tris[k];

    // On an edge → split that edge across both incident triangles.
    let edge = if on_ab {
        Some((t[0], t[1]))
    } else if on_bc {
        Some((t[1], t[2]))
    } else if on_ca {
        Some((t[2], t[0]))
    } else {
        None
    };

    if let Some((a, b)) = edge {
        // Split every triangle using edge (a,b) — the host and its neighbour.
        let affected: Vec<usize> = tris
            .iter()
            .enumerate()
            .filter(|(_, tr)| has_edge(**tr, a, b))
            .map(|(idx, _)| idx)
            .collect();
        let mut new_tris = Vec::new();
        for &idx in &affected {
            let tr = tris[idx];
            let apex = tr.iter().copied().find(|&x| x != a && x != b)?;
            // Preserve orientation: replace (x,y,apex) where (x,y) is (a,b) in tr's order.
            let (x, y) = ordered_edge(tr, a, b);
            new_tris.push([x, i, apex]);
            new_tris.push([i, y, apex]);
        }
        // Remove affected (descending) and append.
        let mut aff = affected;
        aff.sort_unstable_by(|x, y| y.cmp(x));
        for idx in aff {
            tris.swap_remove(idx);
        }
        tris.extend(new_tris);
    } else {
        // Strictly interior → 1→3 split, preserving orientation.
        tris.swap_remove(k);
        tris.push([t[0], t[1], i]);
        tris.push([t[1], t[2], i]);
        tris.push([t[2], t[0], i]);
    }
    Some(())
}

/// `(a,b)` in the cyclic order they appear in `t`.
fn ordered_edge(t: Tri, a: usize, b: usize) -> (usize, usize) {
    for k in 0..3 {
        let (x, y) = (t[k], t[(k + 1) % 3]);
        if (x == a && y == b) || (x == b && y == a) {
            return (x, y);
        }
    }
    (a, b)
}

fn has_edge(t: Tri, a: usize, b: usize) -> bool {
    t.contains(&a) && t.contains(&b)
}

fn edge_exists(tris: &[Tri], u: usize, v: usize) -> bool {
    tris.iter().any(|t| has_edge(*t, u, v))
}

/// Apex of triangle `t` (the vertex that is not on edge `(a,b)`).
fn apex(t: Tri, a: usize, b: usize) -> Option<usize> {
    t.iter().copied().find(|&x| x != a && x != b)
}

fn recover_edge(tris: &mut Vec<Tri>, pts: &[[f64; 2]], u: usize, v: usize) -> Option<()> {
    if u == v || edge_exists(tris, u, v) {
        return Some(());
    }
    let (su, sv) = (pts[u], pts[v]);
    let key = |a: usize, b: usize| (a.min(b), a.max(b));
    // Neighbour-lookup strategy. The per-flip work is finding the triangle across
    // a crossing edge. A linear `find` is O(n), which for a small face is a tiny
    // cache-friendly scan that beats hashing — but on a large coplanar face (a
    // plate drilled by many holes) it dominates and the whole recovery grinds.
    // Past a threshold we switch to an edge→triangle adjacency map, built once per
    // pass in O(n) and looked up in O(1); a pass is then O(n) and recovery is
    // O(flips·n) — O(crossings·n) in practice (the flip count is bounded linearly
    // below). The map is allocated once and cleared per pass so it stays off the
    // hot path for the common small faces (which never take this branch). Flip
    // selection is identical in both paths, so the resulting triangulation is
    // unchanged — only the lookup cost differs.
    const MAP_THRESHOLD: usize = 64;
    let mut adj: std::collections::HashMap<(usize, usize), (usize, usize, u8)> =
        std::collections::HashMap::new();
    let mut guard = 0;
    // Recovering one edge flips at most O(edges crossing it) = O(n) triangles, so
    // a legitimate recovery converges well within a linear bound (measured: a
    // 430-constraint face needs ~1 flip per constraint). A run that blows past
    // this isn't making progress — a degenerate near-collinear face where no
    // convex flip advances — so bail to `None` (→ float fallback) promptly rather
    // than grinding to the old O(n²) ceiling (which is what made a big drilled
    // face take minutes).
    let cap = 16 * tris.len() + 128;
    loop {
        guard += 1;
        if guard > cap {
            return None; // no convergence → caller falls back
        }
        let use_map = tris.len() > MAP_THRESHOLD;
        if use_map {
            adj.clear();
            adj.reserve(tris.len() * 3);
            for (k, t) in tris.iter().enumerate() {
                for e in 0..3 {
                    let slot = adj.entry(key(t[e], t[(e + 1) % 3])).or_insert((k, usize::MAX, 0));
                    if slot.2 == 1 {
                        slot.1 = k;
                    }
                    slot.2 += 1;
                }
            }
            if adj.contains_key(&key(u, v)) {
                return Some(());
            }
        } else if edge_exists(tris, u, v) {
            return Some(());
        }
        // Find the first interior edge (a,b) (triangle order) that properly
        // crosses (u,v) and whose two triangles form a convex quad, then flip it.
        let mut flipped = false;
        'search: for k1 in 0..tris.len() {
            let t1 = tris[k1];
            for e in 0..3 {
                let (a, b) = (t1[e], t1[(e + 1) % 3]);
                if a == u || a == v || b == u || b == v {
                    continue;
                }
                if !seg_cross(su, sv, pts[a], pts[b]) {
                    continue;
                }
                // neighbour across (a,b): O(1) via the map for large faces, else scan.
                let k2 = if use_map {
                    let (n0, n1, count) = adj[&key(a, b)];
                    if count != 2 {
                        continue; // boundary (or degenerate) edge — nothing to flip
                    }
                    if n0 == k1 {
                        n1
                    } else {
                        n0
                    }
                } else {
                    match (0..tris.len()).find(|&j| j != k1 && has_edge(tris[j], a, b)) {
                        Some(k2) => k2,
                        None => continue,
                    }
                };
                let c = apex(t1, a, b)?;
                let d = apex(tris[k2], a, b)?;
                // Convex quad a,c,b,d ⇔ c and d on opposite sides of (a,b) AND
                // a,b on opposite sides of (c,d).
                if orient(pts[a], pts[b], pts[c]) * orient(pts[a], pts[b], pts[d]) < 0.0
                    && orient(pts[c], pts[d], pts[a]) * orient(pts[c], pts[d], pts[b]) < 0.0
                {
                    // Flip (a,b) → (c,d): replace the two triangles.
                    let (hi, lo) = (k1.max(k2), k1.min(k2));
                    tris.swap_remove(hi);
                    tris.swap_remove(lo);
                    tris.push(ccw([c, d, a], pts));
                    tris.push(ccw([d, c, b], pts));
                    flipped = true;
                    break 'search;
                }
            }
        }
        if !flipped {
            return None; // no convex flip available → fall back
        }
    }
}

/// Return the triangle in CCW orientation.
fn ccw(t: Tri, pts: &[[f64; 2]]) -> Tri {
    if orient(pts[t[0]], pts[t[1]], pts[t[2]]) >= 0.0 {
        t
    } else {
        [t[0], t[2], t[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_area(pts: &[[f64; 2]], t: Tri) -> f64 {
        0.5 * orient(pts[t[0]], pts[t[1]], pts[t[2]]).abs()
    }
    fn total_area(pts: &[[f64; 2]], tris: &[Tri]) -> f64 {
        tris.iter().map(|t| tri_area(pts, *t)).sum()
    }
    const CORNERS: [[f64; 2]; 3] = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];

    #[test]
    fn plain_triangle() {
        let out = triangulate_constrained(&CORNERS, &[]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn interior_point_no_constraint() {
        let mut pts = CORNERS.to_vec();
        pts.push([1.0, 1.0]);
        let out = triangulate_constrained(&pts, &[]).unwrap();
        assert_eq!(out.len(), 3);
        assert!((total_area(&pts, &out) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn interior_interior_constraint_recovered() {
        // The hard case: a constraint between two interior points (a pure
        // T-junction at both ends), which insertion won't produce as an edge.
        let mut pts = CORNERS.to_vec();
        pts.push([1.0, 1.0]); // idx 3
        pts.push([2.0, 0.5]); // idx 4
        let out = triangulate_constrained(&pts, &[[3, 4]]).unwrap();
        assert!(edge_exists(&out, 3, 4), "constraint edge present");
        assert!((total_area(&pts, &out) - 8.0).abs() < 1e-9, "area conserved");
        for t in &out {
            assert!(tri_area(&pts, *t) > 1e-9, "no degenerate triangle");
        }
    }

    #[test]
    fn interior_to_boundary_constraint() {
        let mut pts = CORNERS.to_vec();
        pts.push([1.0, 1.0]); // interior, idx 3
        pts.push([2.0, 2.0]); // on hypotenuse x+y=4, idx 4
        let out = triangulate_constrained(&pts, &[[3, 4]]).unwrap();
        assert!(edge_exists(&out, 3, 4));
        assert!((total_area(&pts, &out) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn large_face_ring_recovers_fast() {
        // A big constrained triangulation that exercises the adjacency-map recovery
        // path (well past MAP_THRESHOLD triangles): a bounding triangle enclosing a
        // 120-gon whose edges are all interior-to-interior constraints — the hard
        // recovery case, ×120. If recovery ever regressed to the old O(n⁴) this
        // would crawl; with the O(crossings·n) map path it's milliseconds. Also a
        // correctness check: every ring edge recovered, area conserved, no slivers.
        use std::f64::consts::TAU;
        let n = 120usize;
        let mut pts = vec![[-20.0, -20.0], [20.0, -20.0], [0.0, 20.0]]; // CCW corners
        for i in 0..n {
            let a = i as f64 * TAU / n as f64;
            pts.push([3.0 * a.cos(), 3.0 * a.sin()]);
        }
        let cons: Vec<[usize; 2]> = (0..n).map(|i| [3 + i, 3 + (i + 1) % n]).collect();
        let out = triangulate_constrained(&pts, &cons).expect("large constrained CDT");
        for i in 0..n {
            assert!(edge_exists(&out, 3 + i, 3 + (i + 1) % n), "ring edge {i} recovered");
        }
        assert!((total_area(&pts, &out) - 800.0).abs() < 1e-6, "area conserved (½·40·40)");
        assert!(out.iter().all(|t| tri_area(&pts, *t) > 1e-9), "no degenerate triangles");
    }
}

#[cfg(test)]
mod degen {
    use super::*;
    fn area(pts:&[[f64;2]],t:Tri)->f64{0.5*orient(pts[t[0]],pts[t[1]],pts[t[2]]).abs()}
    #[test]
    fn point_on_edge_with_constraint() {
        // Bounding triangle; a constraint from an interior point to a point that
        // lies EXACTLY on the hypotenuse edge (the degenerate case).
        let pts = vec![
            [0.0,0.0],[4.0,0.0],[0.0,4.0], // corners
            [1.0,1.0],                     // interior (idx 3)
            [2.0,2.0],                     // ON hypotenuse x+y=4 (idx 4)
        ];
        let tris = triangulate_constrained(&pts, &[[3, 4]]).expect("CDT on point-on-edge");
        let tot: f64 = tris.iter().map(|t| area(&pts, *t)).sum();
        assert!((tot - 8.0).abs() < 1e-9, "area conserved with a vertex on the edge");
        assert!(edge_exists(&tris, 3, 4), "constraint recovered");
        assert!(tris.iter().all(|t| area(&pts, *t) > 1e-12), "no degenerate slivers");
    }
    #[test]
    fn rectangle_face_with_super_triangle() {
        // A cylinder side quad (rectangle 1.53×12) with two horizontal cuts and its
        // vertical edges split at the cut heights — the real failing case. Seeded
        // with a super-triangle (idx 0,1,2). A valid CDT must conserve area.
        let sup = 400.0;
        let mut pts = vec![[-sup, -sup], [sup, -sup], [0.0, sup]]; // super-triangle
        pts.extend([
            [1.5307, 12.0], [0.0, 12.0], [1.5307, 0.0], [0.0, 0.0], // 3,4,5,6 corners
            [1.4032, 11.0], [1.5307, 11.0], [1.5307, 1.0], [0.1276, 1.0], // 7,8,9,10
            [0.0, 11.0], [0.0, 1.0], // 11,12
        ]);
        let cons = [
            [4, 11], [11, 12], [12, 6], // left edge
            [5, 9], [9, 8], [8, 3],     // right edge
            [3, 4], [6, 5],             // top, bottom
            [11, 7], [7, 8],            // y=11 cut
            [12, 10], [10, 9],          // y=1 cut
        ];
        let tris = triangulate_constrained(&pts, &cons).expect("CDT on rectangle face");
        // Area of triangles strictly inside the rectangle must equal the rectangle.
        let inside = |t: &Tri| {
            let c = [
                (pts[t[0]][0] + pts[t[1]][0] + pts[t[2]][0]) / 3.0,
                (pts[t[0]][1] + pts[t[1]][1] + pts[t[2]][1]) / 3.0,
            ];
            c[0] > 0.0 && c[0] < 1.5307 && c[1] > 0.0 && c[1] < 12.0
        };
        let a: f64 = tris.iter().filter(|t| inside(t)).map(|t| area(&pts, *t)).sum();
        assert!((a - 1.5307 * 12.0).abs() < 1e-6, "rectangle area conserved, got {a}");
    }

    #[test]
    fn constraint_along_edge() {
        // Two points on the same edge with a constraint between them (sub-segment
        // of the triangle boundary) — arises when a cut runs along a shared edge.
        let pts = vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]];
        let tris = triangulate_constrained(&pts, &[[3, 4]]).expect("CDT along-edge constraint");
        assert!(edge_exists(&tris, 3, 4), "along-edge constraint recovered");
    }
}
