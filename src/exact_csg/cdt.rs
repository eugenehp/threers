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
/// declining (→ float fallback).
///
/// This is a safety net against a pathological face, not a working limit. It used
/// to be 256, which real models cross easily — a 100 mm plate cut by four
/// `$fn=32` spheres puts ~274 points on one face — and crossing it silently
/// dropped the whole boolean to the float evaluator. Point insertion and the
/// Delaunay cleanup are O(n²), so this bound is set where the worst case is still
/// seconds rather than minutes.
const MAX_CDT_POINTS: usize = 20_000;

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

/// The margin `in_circle` must clear before a flip is allowed.
///
/// The shape is Shewchuk's first-stage filter — the *permanent* (the same
/// expression with every term made positive) bounds the rounding error in the
/// determinant — but [`ICC_REL`] is set far above that bound deliberately. See
/// its note.
///
/// The point is that the bound has to scale with the geometry. `in_circle` is a
/// determinant in squared distances, so it grows as the fourth power of the
/// coordinate scale, while the margin it was compared against was the constant
/// `1e-9`. On a model measured in tens of millimetres that constant sits far
/// below the noise, so float noise decided flips.
fn in_circle_errbound(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
    let (bx, by) = (b[0] - d[0], b[1] - d[1]);
    let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let perm = ax.abs() * ((by * c2).abs() + (b2 * cy).abs())
        + ay.abs() * ((bx * c2).abs() + (b2 * cx).abs())
        + a2.abs() * ((bx * cy).abs() + (by * cx).abs());
    ICC_REL * perm
}
/// Deliberately conservative: about a thousand times the ~1.1e-15 that actually
/// bounds the rounding error. The two failure modes are not symmetric. Flipping
/// too eagerly on a near-cocircular quad costs termination — the flip and its
/// reverse both look improving, and the pass loop cycles until its iteration cap
/// and then just stops, wherever it happens to be. Declining to flip costs at
/// most one sliver left slightly non-Delaunay. So the margin is set where
/// near-cocircular quads are left alone.
///
/// Measured over one frame of the reference animation, moving from the constant
/// `1e-9` to this: 12,720 flip passes → 1,797, and **44 faces that were hitting
/// the iteration cap → 0**. Output is byte-identical, and the flip counts on
/// well-conditioned faces are unchanged to the flip — this only suppresses the
/// cycling.
const ICC_REL: f64 = 1.0e-12;

/// Constrained triangulation. `pts[0..3]` are the triangle corners; the rest lie
/// inside or on it. Returns triangles (index triples) conforming to `constraints`,
/// or `None` if recovery fails.
pub fn triangulate_constrained(pts: &[[f64; 2]], constraints: &[[usize; 2]]) -> Option<Vec<Tri>> {
    if pts.len() < 3 {
        return None;
    }
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

/// Lawson flips: while some non-constraint interior edge fails the Delaunay
/// incircle test and its two triangles form a convex quad, flip it.
///
/// Driven by a work list of edges rather than by repeated full passes. A flip
/// can only disturb the four outer edges of the quad it sits in, so those are
/// the only ones worth re-testing; pushing them and popping until the list
/// drains costs O(1) per flip instead of a full O(n log n) rebuild-and-rescan
/// per round of them.
///
/// The batched version this replaces rebuilt the edge→triangle map, sorted its
/// keys, and rescanned everything on every pass, and got 6.7 flips out of each
/// of those passes — 1797 passes to perform 12,118 flips across one frame. It
/// was itself a fix for a worse version that rebuilt per *flip*.
///
/// Determinism matters here and is easy to lose: which flips happen, and in
/// which order, decides the output triangulation. The seed list is sorted, not
/// taken in `HashMap` order — hash order is randomly seeded per process, and
/// with it the kernel converged on one run and hit the flip cap on the next,
/// falling back to the float evaluator silently and with a non-manifold result.
fn delaunay_flip(
    tris: &mut [Tri],
    pts: &[[f64; 2]],
    cons: &std::collections::HashSet<(usize, usize)>,
) {
    let mut adj: EdgeMap = std::collections::HashMap::with_capacity(tris.len() * 3);
    for (k, t) in tris.iter().enumerate() {
        attach_tri(&mut adj, *t, k);
    }

    // Seeded with every edge, ascending: `pop` takes the back, so reverse first.
    let mut work: Vec<Edge> = adj.keys().copied().collect();
    work.sort_unstable();
    work.reverse();
    let mut queued: std::collections::HashSet<Edge> = work.iter().copied().collect();

    // Each flip strictly improves the triangulation — the incircle margin is set
    // wide enough that a near-cocircular quad is left alone — so this terminates
    // on its own. The cap is a backstop against a degenerate face, not the
    // mechanism, and it is counted in edge *tests* rather than flips.
    let cap = 64 * tris.len() + 4096;
    let mut guard = 0usize;
    while let Some(e) = work.pop() {
        queued.remove(&e);
        guard += 1;
        if guard > cap {
            return;
        }
        let (a, b) = e;
        if cons.contains(&e) {
            continue;
        }
        let Some(&(k1, k2, count)) = adj.get(&e) else {
            continue;
        };
        if count != 2 {
            continue; // boundary, or a non-manifold edge — nothing to flip
        }
        let (Some(c), Some(d)) = (apex(tris[k1], a, b), apex(tris[k2], a, b)) else {
            continue;
        };
        let convex = orient(pts[a], pts[b], pts[c]) * orient(pts[a], pts[b], pts[d]) < 0.0
            && orient(pts[c], pts[d], pts[a]) * orient(pts[c], pts[d], pts[b]) < 0.0;
        if !convex {
            continue;
        }
        let ccw_tri = ccw([a, b, c], pts);
        let (p0, p1, p2, pd) = (pts[ccw_tri[0]], pts[ccw_tri[1]], pts[ccw_tri[2]], pts[d]);
        if in_circle(p0, p1, p2, pd) <= in_circle_errbound(p0, p1, p2, pd).max(1e-9) {
            continue;
        }

        // Flip (a,b) → (c,d). Written in place rather than removed and pushed, so
        // every other triangle index — and so the whole adjacency — stays valid.
        let (old1, old2) = (tris[k1], tris[k2]);
        detach_tri(&mut adj, old1, k1);
        detach_tri(&mut adj, old2, k2);
        let (new1, new2) = (ccw([c, d, a], pts), ccw([d, c, b], pts));
        tris[k1] = new1;
        tris[k2] = new2;
        attach_tri(&mut adj, new1, k1);
        attach_tri(&mut adj, new2, k2);

        // Only the quad's own boundary can have been made non-Delaunay by this.
        for (x, y) in [(a, c), (c, b), (b, d), (d, a)] {
            let key = edge_key(x, y);
            if !cons.contains(&key) && queued.insert(key) {
                work.push(key);
            }
        }
    }
}

type Edge = (usize, usize);
/// Edge → the (up to two) triangles sharing it, and how many claim it. The count
/// is kept separately because a non-manifold edge has more than the two slots can
/// hold, and such an edge must be left alone rather than flipped.
type EdgeMap = std::collections::HashMap<Edge, (usize, usize, u32)>;
const NO_TRI: usize = usize::MAX;

fn edge_key(a: usize, b: usize) -> Edge {
    (a.min(b), a.max(b))
}

fn attach_tri(adj: &mut EdgeMap, t: Tri, k: usize) {
    for e in 0..3 {
        let slot = adj
            .entry(edge_key(t[e], t[(e + 1) % 3]))
            .or_insert((NO_TRI, NO_TRI, 0));
        if slot.0 == NO_TRI {
            slot.0 = k;
        } else if slot.1 == NO_TRI {
            slot.1 = k;
        }
        slot.2 += 1;
    }
}

fn detach_tri(adj: &mut EdgeMap, t: Tri, k: usize) {
    for e in 0..3 {
        let key = edge_key(t[e], t[(e + 1) % 3]);
        let Some(slot) = adj.get_mut(&key) else {
            continue;
        };
        if slot.0 == k {
            slot.0 = slot.1;
            slot.1 = NO_TRI;
        } else if slot.1 == k {
            slot.1 = NO_TRI;
        }
        slot.2 = slot.2.saturating_sub(1);
        if slot.2 == 0 {
            adj.remove(&key);
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
    // Neighbour lookup is a linear `find`. This was once an edge→triangle hash map
    // for faces past 64 triangles, on the reasoning that O(1) lookups must beat an
    // O(n) scan on a large coplanar face. Measured, it was the reverse, by a lot:
    //
    //   face (triangles)   with map   linear scan
    //   1597                122.5 ms       31.8 ms
    //   1337                 38.3 ms        8.2 ms
    //    455                 13.0 ms        4.3 ms
    //
    // The map has to be rebuilt after every flip, since a flip changes the mesh —
    // so it cost a full O(n) of *hashing* per iteration to save lookups on the
    // handful of edges that actually cross `(u,v)`. On one face that was 6.6
    // million hash insertions to serve 1378 flips. The scan, by contrast, is a
    // cache-friendly walk over a flat `Vec<[usize; 3]>`, and the search usually
    // breaks early on the first reducing flip without reaching the end.
    //
    // The crossover is around 27 crossing edges per iteration; below that the scan
    // wins. If a model ever does exceed it, the fix is an adjacency map maintained
    // *incrementally* across flips — not one rebuilt from scratch each time.
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
        if edge_exists(tris, u, v) {
            return Some(());
        }
        // Find an interior edge (a,b) that properly crosses (u,v) and whose two
        // triangles form a convex quad, then flip it.
        //
        // PREFER a flip that strictly reduces the number of crossings, i.e. one
        // whose replacement diagonal (c,d) does not itself cross (u,v). Flipping an
        // arbitrary convex crossing edge is not monotone — the new diagonal can
        // cross the segment again — so the search can cycle between two states and
        // grind until the iteration guard trips, at which point the whole boolean
        // silently falls back to the float evaluator. Requiring monotone progress
        // bounds recovery at one flip per crossing edge. A non-reducing convex flip
        // is retained as a fallback for configurations that need a detour.
        let mut reducing: Option<(usize, usize, usize, usize, usize, usize)> = None;
        let mut fallback: Option<(usize, usize, usize, usize, usize, usize)> = None;
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
                // The triangle across (a,b). None means a boundary edge, with
                // nothing on the far side to flip against.
                let Some(k2) = (0..tris.len()).find(|&j| j != k1 && has_edge(tris[j], a, b)) else {
                    continue;
                };
                let (Some(c), Some(d)) = (apex(t1, a, b), apex(tris[k2], a, b)) else {
                    continue;
                };
                // Convex quad a,c,b,d ⇔ c and d on opposite sides of (a,b) AND
                // a,b on opposite sides of (c,d).
                if orient(pts[a], pts[b], pts[c]) * orient(pts[a], pts[b], pts[d]) < 0.0
                    && orient(pts[c], pts[d], pts[a]) * orient(pts[c], pts[d], pts[b]) < 0.0
                {
                    if !seg_cross(su, sv, pts[c], pts[d]) {
                        reducing = Some((k1, k2, a, b, c, d));
                        break 'search;
                    }
                    if fallback.is_none() {
                        fallback = Some((k1, k2, a, b, c, d));
                    }
                }
            }
        }
        let Some((k1, k2, a, b, c, d)) = reducing.or(fallback) else {
            return None; // no convex flip available → fall back
        };
        // Flip (a,b) → (c,d): replace the two triangles.
        let (hi, lo) = (k1.max(k2), k1.min(k2));
        tris.swap_remove(hi);
        tris.swap_remove(lo);
        tris.push(ccw([c, d, a], pts));
        tris.push(ccw([d, c, b], pts));
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
        assert!(
            (total_area(&pts, &out) - 8.0).abs() < 1e-9,
            "area conserved"
        );
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
        // A big constrained triangulation on the hard recovery path: a bounding
        // triangle enclosing a 120-gon whose edges are all interior-to-interior
        // constraints — the hard recovery case, ×120. If recovery ever regressed to
        // the old O(n⁴) this would crawl; at O(crossings·n) it's milliseconds. Also a
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
            assert!(
                edge_exists(&out, 3 + i, 3 + (i + 1) % n),
                "ring edge {i} recovered"
            );
        }
        assert!(
            (total_area(&pts, &out) - 800.0).abs() < 1e-6,
            "area conserved (½·40·40)"
        );
        assert!(
            out.iter().all(|t| tri_area(&pts, *t) > 1e-9),
            "no degenerate triangles"
        );
    }
}

#[cfg(test)]
mod degen {
    use super::*;
    fn area(pts: &[[f64; 2]], t: Tri) -> f64 {
        0.5 * orient(pts[t[0]], pts[t[1]], pts[t[2]]).abs()
    }
    #[test]
    fn point_on_edge_with_constraint() {
        // Bounding triangle; a constraint from an interior point to a point that
        // lies EXACTLY on the hypotenuse edge (the degenerate case).
        let pts = vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0], // corners
            [1.0, 1.0], // interior (idx 3)
            [2.0, 2.0], // ON hypotenuse x+y=4 (idx 4)
        ];
        let tris = triangulate_constrained(&pts, &[[3, 4]]).expect("CDT on point-on-edge");
        let tot: f64 = tris.iter().map(|t| area(&pts, *t)).sum();
        assert!(
            (tot - 8.0).abs() < 1e-9,
            "area conserved with a vertex on the edge"
        );
        assert!(edge_exists(&tris, 3, 4), "constraint recovered");
        assert!(
            tris.iter().all(|t| area(&pts, *t) > 1e-12),
            "no degenerate slivers"
        );
    }
    #[test]
    fn rectangle_face_with_super_triangle() {
        // A cylinder side quad (rectangle 1.53×12) with two horizontal cuts and its
        // vertical edges split at the cut heights — the real failing case. Seeded
        // with a super-triangle (idx 0,1,2). A valid CDT must conserve area.
        let sup = 400.0;
        let mut pts = vec![[-sup, -sup], [sup, -sup], [0.0, sup]]; // super-triangle
        pts.extend([
            [1.5307, 12.0],
            [0.0, 12.0],
            [1.5307, 0.0],
            [0.0, 0.0], // 3,4,5,6 corners
            [1.4032, 11.0],
            [1.5307, 11.0],
            [1.5307, 1.0],
            [0.1276, 1.0], // 7,8,9,10
            [0.0, 11.0],
            [0.0, 1.0], // 11,12
        ]);
        let cons = [
            [4, 11],
            [11, 12],
            [12, 6], // left edge
            [5, 9],
            [9, 8],
            [8, 3], // right edge
            [3, 4],
            [6, 5], // top, bottom
            [11, 7],
            [7, 8], // y=11 cut
            [12, 10],
            [10, 9], // y=1 cut
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
        let a: f64 = tris
            .iter()
            .filter(|t| inside(t))
            .map(|t| area(&pts, *t))
            .sum();
        assert!(
            (a - 1.5307 * 12.0).abs() < 1e-6,
            "rectangle area conserved, got {a}"
        );
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
