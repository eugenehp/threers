//! Robust mesh-arrangement boolean kernel — the watertight path that replaces the
//! three-bvh-csg float kernel for OpenSCAD-parity work.
//!
//! `boolean` resolves two triangle meshes in three stages:
//! 1. **Co-refinement** (`corefine`) — compute every triangle–triangle
//!    intersection segment (BVH broad phase), snap shared endpoints identically on
//!    both meshes, and re-triangulate each mesh one flat *face* at a time with the
//!    constrained-Delaunay routine in `cdt`, so the seam is subdivided the same
//!    way on both sides.
//! 2. **Classification** — keep each refined sub-triangle by the op and which side
//!    of the *other* solid it lies on (multi-ray parity, with a coincident-face
//!    rule for shared planar faces).
//! 3. **Verification** — accept the result only if it is a closed 2-manifold
//!    (welding + T-junction healing first, then a hard gate). If it can't be
//!    verified, return `BooleanOutcome::NeedsArrangement` so the caller falls
//!    back to the float kernel rather than trust an unproven mesh — the kernel is
//!    never wrong, only sometimes deferential.
//!
//! This resolves genuinely *curved∧curved* crossings — sphere∪sphere,
//! cylinder∩cylinder, cone/sphere/cylinder mixes — to watertight meshes (which the
//! float kernel cannot: its sphere∩sphere is degenerate). Predicates use exact
//! `orient2d`/`orient3d`; the remaining fallbacks are measure-zero degeneracies
//! (e.g. two identical primitives translated along an axis, whose mirror-coincident
//! seam facets a tiny perturbation would remove) and very large coplanar faces
//! (bounded by the CDT size guard).

/// Closed-form surface-pair resolution, when both meshes carry provenance.
/// Strictly an accelerator — see the module docs. Stage 2 of
/// `docs/brep-nurbs-plan.md`.
#[cfg(feature = "brep-csg")]
pub mod analytic;
mod arrangement;
mod bvh;
mod cdt;
mod coplanar;
pub mod metrics;

/// Booleans via the Manifold kernel — see the module docs for why.
#[cfg(feature = "manifold")]
pub mod manifold_backend;

/// The tolerance ladder this kernel balances on, in one place.
pub mod tolerance;

mod predicates;
/// Mesh diagnostics that report WHERE a solid is broken.
pub mod report;
mod split;
mod tri_intersect;
mod triangulate;
mod winding;

pub use arrangement::{intersection_segments, tri_tri_segment};
pub use coplanar::coplanar_clip;
pub use predicates::{coplanar, orient2d, orient3d};

// ---------------------------------------------------------------------------
// Unverified-result accounting
// ---------------------------------------------------------------------------

/// How many booleans have returned a mesh this kernel could **not** verify as
/// watertight, since the last [`reset_unverified_booleans`].
///
/// When the arrangement declines, callers fall back to the float evaluator; when
/// that result is not closed either and cannot be healed, the geometry is
/// returned anyway rather than failing the whole model. That is a defensible
/// trade for a preview and indefensible for anything downstream — a mesh with
/// cracks cannot be printed, cannot have mass properties computed, and its
/// defects compound through the next boolean. It used to be reported only as a
/// `log::warn`, which in practice means nobody sees it.
///
/// Read this after building a model to find out whether what you have is a
/// solid. `tests/watertight_gate.rs` asserts on it so the number cannot silently
/// grow.
pub fn unverified_booleans() -> usize {
    UNVERIFIED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reset the [`unverified_booleans`] counter to zero.
pub fn reset_unverified_booleans() {
    UNVERIFIED.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Bump the counter from a test, so the gate in `tests/watertight_gate.rs` can
/// prove the wiring works rather than trusting it.
#[doc(hidden)]
pub fn note_unverified_boolean_for_test() {
    note_unverified_boolean();
}

/// Record that a boolean produced a mesh that could not be verified.
pub(crate) fn note_unverified_boolean() {
    UNVERIFIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

static UNVERIFIED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "brep-csg")]
pub use analytic::{analytic_report, AnalyticReport};
pub use split::split_triangle_by_plane;
pub use tri_intersect::tri_tri_intersect;
pub use triangulate::triangulate_with_points;
pub use winding::{point_in_mesh, winding_number};

use crate::compute_vertex_normals;
use crate::core::{BufferAttribute, BufferGeometry};
use std::collections::HashMap;

/// A three-vector in f64 (the crate's `Vector3` is f32; winding sums want f64).
pub(crate) type V3 = [f64; 3];

pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub(crate) fn sqlen(a: V3) -> f64 {
    dot(a, a)
}
fn normalize(a: V3) -> V3 {
    let l = norm(a);
    if l < 1e-18 {
        a
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

/// The three boolean operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Union,
    Difference,
    Intersection,
}

/// Result of `boolean`.
#[derive(Debug)]
pub enum BooleanOutcome {
    /// Surfaces don't cross — this mesh is the exact boolean for that class.
    Exact(BufferGeometry),
    /// Surfaces cross; the arrangement (sub-triangulation) is required (M2).
    NeedsArrangement,
}

/// Read a geometry's triangles as f64 vertex triples (indexed or soup).
pub(crate) fn triangles(g: &BufferGeometry) -> Vec<[V3; 3]> {
    let pos = match g.get_attribute("position") {
        Some(p) => &p.array,
        None => return Vec::new(),
    };
    let v = |i: usize| -> V3 {
        [
            pos[i * 3] as f64,
            pos[i * 3 + 1] as f64,
            pos[i * 3 + 2] as f64,
        ]
    };
    let mut out = Vec::new();
    if let Some(idx) = &g.index {
        for t in idx.chunks_exact(3) {
            out.push([v(t[0] as usize), v(t[1] as usize), v(t[2] as usize)]);
        }
    } else {
        for k in 0..pos.len() / 9 {
            out.push([v(k * 3), v(k * 3 + 1), v(k * 3 + 2)]);
        }
    }
    out
}

fn centroid(t: &[V3; 3]) -> V3 {
    [
        (t[0][0] + t[1][0] + t[2][0]) / 3.0,
        (t[0][1] + t[1][1] + t[2][1]) / 3.0,
        (t[0][2] + t[1][2] + t[2][2]) / 3.0,
    ]
}

pub(crate) fn aabb(t: &[V3; 3]) -> (V3, V3) {
    let mut mn = t[0];
    let mut mx = t[0];
    for v in &t[1..] {
        for k in 0..3 {
            mn[k] = mn[k].min(v[k]);
            mx[k] = mx[k].max(v[k]);
        }
    }
    (mn, mx)
}

pub(crate) fn aabb_overlap(a: &(V3, V3), b: &(V3, V3)) -> bool {
    for k in 0..3 {
        if a.0[k] > b.1[k] || b.0[k] > a.1[k] {
            return false;
        }
    }
    true
}

/// Do the meshes interact at all — any transversal crossing or coplanar face
/// pair? BVH-accelerated broad phase; short-circuits on the first hit.
fn interacts(ta: &[[V3; 3]], tb: &[[V3; 3]], bvh_b: &bvh::Bvh) -> bool {
    let mut cand = Vec::new();
    for a in ta {
        bvh_b.overlaps(aabb(a), &mut cand);
        for &j in &cand {
            if tri_tri_intersect(a, &tb[j]) || coplanar(a, &tb[j]) {
                return true;
            }
        }
    }
    false
}

fn build_geometry(tris: &[[V3; 3]]) -> BufferGeometry {
    let mut pos = Vec::with_capacity(tris.len() * 9);
    for t in tris {
        for v in t {
            pos.push(v[0] as f32);
            pos.push(v[1] as f32);
            pos.push(v[2] as f32);
        }
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    compute_vertex_normals(&mut g);
    g
}

/// Quantized undirected edge key on the fine weld grid.
type EdgeKey = (i64, i64, i64, i64, i64, i64);
/// A point snapped to the welding grid, used as a map key.
type PointKey = (i64, i64, i64);
/// An undirected edge between two snapped points.
type SnappedEdge = (PointKey, PointKey);
/// A plane snapped to the grouping grid.
type PlaneKey = (i64, i64, i64, i64);
/// The two meshes, each re-triangulated against the other.
type Corefined = (Vec<[V3; 3]>, Vec<[V3; 3]>);
fn ekey(a: V3, b: V3) -> EdgeKey {
    let k = |p: V3| {
        (
            (p[0] * 1e5).round() as i64,
            (p[1] * 1e5).round() as i64,
            (p[2] * 1e5).round() as i64,
        )
    };
    let (ka, kb) = (k(a), k(b));
    let (lo, hi) = if ka <= kb { (ka, kb) } else { (kb, ka) };
    (lo.0, lo.1, lo.2, hi.0, hi.1, hi.2)
}

/// Oriented-plane key (normal + signed offset), quantized. Two triangles with the
/// same key are coplanar and same-facing — one flat *face* of the input.
///
/// `pub(crate)` so the `brep` provenance tests can check this hash against
/// ground truth: with real face identity available, the grouping it *guesses*
/// becomes falsifiable rather than merely load-bearing.
pub(crate) fn plane_key(t: &[V3; 3]) -> (i64, i64, i64, i64) {
    let n = normalize(face_normal(t));
    let d = dot(n, t[0]);
    let q = |x: f64| (x * 1e4).round() as i64;
    (q(n[0]), q(n[1]), q(n[2]), q(d))
}

/// Re-triangulate a whole coplanar **face** (group of triangles) in one CDT — the
/// fix for T-junctions: cuts crossing a face's internal diagonals become shared
/// vertices instead of mismatched splits. `boundary` is the face's outline (edges
/// on the group's border), `segs` the cut segments on it, and `tj` the T-junction
/// points from neighbouring faces landing on this face's boundary. `None` on CDT
/// failure (→ caller falls back).
fn cdt_face(
    group: &[usize],
    tris: &[[V3; 3]],
    boundary: &[(V3, V3)],
    segs: &[(V3, V3)],
    xpts: &[V3],
) -> Option<Vec<[V3; 3]>> {
    let t0 = &tris[group[0]];
    let n = normalize(face_normal(t0));
    if sqlen(n) < 1e-18 {
        return None;
    }
    let origin = t0[0];
    let ux = normalize(sub(t0[1], t0[0]));
    let uy = cross(n, ux);
    let project = |p: V3| [dot(sub(p, origin), ux), dot(sub(p, origin), uy)];
    let lift = |q: [f64; 2]| {
        [
            origin[0] + q[0] * ux[0] + q[1] * uy[0],
            origin[1] + q[0] * ux[1] + q[1] * uy[1],
            origin[2] + q[0] * ux[2] + q[1] * uy[2],
        ]
    };

    // Reserve indices 0,1,2 for a super-triangle (filled once the face's 2D bbox is
    // known). Sentinel 3D coords keep find_or_add from ever welding a real point
    // onto a super-corner.
    let mut pts3: Vec<V3> = vec![[1e30, 1e30, 1e30], [2e30, 2e30, 2e30], [3e30, 3e30, 3e30]];
    let mut pts2: Vec<[f64; 2]> = vec![[0.0; 2], [0.0; 2], [0.0; 2]];
    // Register every point first (segment/boundary endpoints + T-junctions), so a
    // point that lands on another segment is available to split it below.
    let mut raw: Vec<(V3, V3)> = Vec::with_capacity(boundary.len() + segs.len());
    for &(a, b) in boundary.iter().chain(segs.iter()) {
        find_or_add(&mut pts3, &mut pts2, a, project(a));
        find_or_add(&mut pts3, &mut pts2, b, project(b));
        raw.push((a, b));
    }
    // Cross-face T-junction heal: any global intersection point that lies on this
    // face's plane AND strictly on one of its edges/cuts must be a vertex here too,
    // so the shared edge is split identically on both incident faces.
    for &p in xpts {
        if dot(sub(p, origin), n).abs() > tolerance::PLANE_GATE {
            continue;
        }
        if raw.iter().any(|&(a, b)| on_segment_interior(a, b, p)) {
            find_or_add(&mut pts3, &mut pts2, p, project(p));
        }
    }
    // Build the enclosing super-triangle from the real points' 2D bounding box.
    let (mut mnx, mut mny, mut mxx, mut mxy) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for q in &pts2[3..] {
        mnx = mnx.min(q[0]);
        mny = mny.min(q[1]);
        mxx = mxx.max(q[0]);
        mxy = mxy.max(q[1]);
    }
    if !mnx.is_finite() {
        return None;
    }
    let span = (mxx - mnx).max(mxy - mny).max(1.0);
    let d = span * 20.0;
    let (cx, cy) = ((mnx + mxx) * 0.5, (mny + mxy) * 0.5);
    let sv = [[cx - d, cy - d], [cx + d, cy - d], [cx, cy + d]];
    for k in 0..3 {
        pts2[k] = sv[k];
        pts3[k] = lift(sv[k]);
    }

    // Split each raw segment at every registered point lying on it, so no vertex is
    // ever strictly interior to a constraint (which recover_edge cannot heal). This
    // is what makes shared boundary edges and internal cuts conform without
    // T-junctions.
    let mut cons: Vec<[usize; 2]> = Vec::new();
    for &(a, b) in &raw {
        let (a2, b2) = (project(a), project(b));
        let ab = [b2[0] - a2[0], b2[1] - a2[1]];
        let len2 = ab[0] * ab[0] + ab[1] * ab[1];
        if len2 < 1e-18 {
            continue;
        }
        let tol = 1e-9 * len2; // squared perpendicular distance threshold
        let mut on: Vec<(f64, usize)> = Vec::new();
        for (i, &q) in pts2.iter().enumerate().skip(3) {
            let t = ((q[0] - a2[0]) * ab[0] + (q[1] - a2[1]) * ab[1]) / len2;
            if !(-1e-9..=1.0 + 1e-9).contains(&t) {
                continue;
            }
            let proj = [a2[0] + t * ab[0], a2[1] + t * ab[1]];
            let d2 = (q[0] - proj[0]).powi(2) + (q[1] - proj[1]).powi(2);
            if d2 <= tol {
                on.push((t.clamp(0.0, 1.0), i));
            }
        }
        on.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        on.dedup_by_key(|x| x.1);
        for w in on.windows(2) {
            if w[0].1 != w[1].1 {
                cons.push([w[0].1, w[1].1]);
            }
        }
    }

    let idx = cdt::triangulate_constrained(&pts2, &cons)?;

    // Keep the real (non-super) CDT triangles.
    let mut work: Vec<[usize; 3]> = idx
        .into_iter()
        .filter(|t| t[0] >= 3 && t[1] >= 3 && t[2] >= 3)
        .collect();

    // Heal T-junctions: the flip-based recovery can leave a vertex lying on the
    // *edge* of a neighbouring triangle (when a collinear split point wasn't routed
    // through both sides). Split any such triangle so every on-edge vertex becomes a
    // shared corner — otherwise dropping the resulting sliver would open a crack.
    let on_edge = |e0: usize, e1: usize, p: usize| -> bool {
        let (a, b, q) = (pts2[e0], pts2[e1], pts2[p]);
        if orient2d(a, b, q).abs() > 1e-9 {
            return false; // not collinear
        }
        let ab = [b[0] - a[0], b[1] - a[1]];
        let len2 = ab[0] * ab[0] + ab[1] * ab[1];
        if len2 < 1e-18 {
            return false;
        }
        let t = ((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / len2;
        t > 1e-7 && t < 1.0 - 1e-7
    };
    // Work-stack heal: pop a triangle, and if any registered point lies strictly on
    // one of its edges, split it there and re-process the two halves; otherwise it's
    // clean. Points are fixed (from pts2), so each triangle splits a bounded number
    // of times and both triangles sharing an edge split at the same point → no
    // T-junctions. The cap guards against a pathological blow-up (→ caller falls back).
    let np = pts2.len();
    let mut healed: Vec<[usize; 3]> = Vec::with_capacity(work.len());
    let cap = (work.len() + np) * 32 + 1024;
    while let Some(t) = work.pop() {
        if healed.len() + work.len() > cap {
            return None;
        }
        let mut split = None;
        'find: for e in 0..3 {
            let (u, v, w) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
            for p in 3..np {
                if p != u && p != v && p != w && on_edge(u, v, p) {
                    split = Some((u, v, w, p));
                    break 'find;
                }
            }
        }
        match split {
            Some((u, v, w, p)) => {
                work.push([u, p, w]);
                work.push([p, v, w]);
            }
            None => healed.push(t),
        }
    }
    let work = healed;

    // Group triangles in 2D, to reject CDT triangles outside a non-convex face.
    let face2: Vec<[[f64; 2]; 3]> = group
        .iter()
        .map(|&i| {
            let t = &tris[i];
            [project(t[0]), project(t[1]), project(t[2])]
        })
        .collect();
    let mut out = Vec::with_capacity(work.len());
    for t in work {
        let (i0, i1, i2) = (t[0], t[1], t[2]);
        // Reject slivers: the flip-based recovery can leave a near-collinear triangle
        // straddling a split edge (all vertices ~on one line). It covers no real
        // surface but injects spurious/duplicate edges → non-manifold. Use a *relative*
        // test (min height vs longest edge) so it fires regardless of the face's scale
        // or the projection's obliqueness, where an absolute area threshold wouldn't.
        let area2 = orient2d(pts2[i0], pts2[i1], pts2[i2]).abs();
        let elen = [
            sqlen(sub(pts3[i1], pts3[i0])),
            sqlen(sub(pts3[i2], pts3[i1])),
            sqlen(sub(pts3[i0], pts3[i2])),
        ];
        let longest = elen[0].max(elen[1]).max(elen[2]).max(1e-30).sqrt();
        if area2 / longest < 1e-6 {
            continue; // min height < ~1e-6 → degenerate sliver
        }
        let c2 = [
            (pts2[i0][0] + pts2[i1][0] + pts2[i2][0]) / 3.0,
            (pts2[i0][1] + pts2[i1][1] + pts2[i2][1]) / 3.0,
        ];
        if !face2.iter().any(|f| point_in_tri_2d(f, c2)) {
            continue; // outside the face (convex-hull fill on a non-convex group)
        }
        let (a, b, c) = (pts3[i0], pts3[i1], pts3[i2]);
        if dot(cross(sub(b, a), sub(c, a)), n) >= 0.0 {
            out.push([a, b, c]);
        } else {
            out.push([a, c, b]);
        }
    }
    Some(out)
}

fn find_or_add(pts3: &mut Vec<V3>, pts2: &mut Vec<[f64; 2]>, p: V3, p2: [f64; 2]) -> usize {
    for (i, q) in pts3.iter().enumerate() {
        if sqlen(sub(*q, p)) < 1e-12 {
            return i;
        }
    }
    pts3.push(p);
    pts2.push(p2);
    pts3.len() - 1
}

/// Co-refine both meshes against **one shared** set of intersection segments, so
/// the seam vertices are bit-identical on both sides (the fix for curved seams —
/// each `tri_tri_segment`/coplanar-overlap is computed once and inserted into
/// both the A-triangle and the B-triangle it lies on).
fn corefine(
    ta: &[[V3; 3]],
    tb: &[[V3; 3]],
    bvh_b: &bvh::Bvh,
    #[cfg(feature = "brep-csg")] analytic: &mut analytic::Analytic<'_>,
) -> Option<Corefined> {
    let mut a_segs: Vec<Vec<(V3, V3)>> = vec![Vec::new(); ta.len()];
    let mut b_segs: Vec<Vec<(V3, V3)>> = vec![Vec::new(); tb.len()];
    let mut cand = Vec::new();

    // How far a seam endpoint may slide along its own edge to become exact.
    // `tri_tri_segment` carries ~1e-6 at unit scale; anything larger means the
    // root find landed on a different crossing of the same edge, and the
    // numeric point is kept. The accelerator may sharpen; it may not relocate.
    #[cfg(feature = "brep-csg")]
    let edge_snap = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in ta.iter().chain(tb.iter()) {
            for v in t {
                for c in v {
                    lo = lo.min(*c);
                    hi = hi.max(*c);
                }
            }
        }
        if lo.is_finite() {
            (hi - lo).max(1.0) * 1e-5
        } else {
            1e-5
        }
    };

    for (i, a) in ta.iter().enumerate() {
        bvh_b.overlaps(aabb(a), &mut cand); // O(log M) candidate B-triangles
        for &j in &cand {
            let b = &tb[j];

            // Ask the surfaces first, when both meshes know them.
            #[cfg(feature = "brep-csg")]
            let decision = if analytic.active() {
                analytic.decide(i, j)
            } else {
                analytic::Decision::Numeric
            };
            #[cfg(feature = "brep-csg")]
            match decision {
                // Provably no intersection.
                analytic::Decision::Skip => continue,
                // Same surface: the triangles cannot transversally cross, so any
                // segment `tri_tri_segment` would report is noise off a
                // near-coincidence. Take only the coplanar overlap, which is the
                // correct refinement for a shared face.
                analytic::Decision::SameSurface => {
                    if coplanar(a, b) {
                        let poly = coplanar::coplanar_overlap_poly(a, b);
                        for k in 0..poly.len() {
                            let s = (poly[k], poly[(k + 1) % poly.len()]);
                            if sqlen(sub(s.1, s.0)) > 1e-18 {
                                a_segs[i].push(s);
                                b_segs[j].push(s);
                            }
                        }
                    }
                    continue;
                }
                // `Curves` is *not* used to relocate seam points. A
                // `tri_tri_segment` endpoint lies on a triangle edge of one of
                // the two meshes, and the refinement downstream depends on that:
                // projecting it onto the surfaces' exact intersection curve
                // moves it off its edge, and the CDT then fails to close. The
                // exact curve is where the *surfaces* meet; the mesh seam is a
                // polyline approximating it whose vertices must stay on mesh
                // edges. Making the seam exact means intersecting each mesh edge
                // with the other surface — see the Stage 2 notes in
                // `docs/brep-nurbs-plan.md`. Measured: snapping here turns two
                // overlapping spheres from `Exact` into `NeedsArrangement`.
                analytic::Decision::Refine | analytic::Decision::Numeric => {}
            }

            if let Some(s) = tri_tri_segment(a, b) {
                // Sharpen the endpoints along the edges they already lie on.
                #[cfg(feature = "brep-csg")]
                let s = if decision == analytic::Decision::Refine {
                    match analytic.surfaces_of(i, j) {
                        Some((sa, sb)) => {
                            let q0 = analytic::exact_on_edge(s.0, a, b, sa, sb, edge_snap);
                            let q1 = analytic::exact_on_edge(s.1, a, b, sa, sb, edge_snap);
                            for (p, q) in [(s.0, q0), (s.1, q1)] {
                                let d = sqlen(sub(p, q)).sqrt();
                                if d > 0.0 {
                                    analytic.sharpened += 1;
                                    analytic.max_sharpen = analytic.max_sharpen.max(d);
                                }
                            }
                            (q0, q1)
                        }
                        None => s,
                    }
                } else {
                    s
                };
                if sqlen(sub(s.1, s.0)) > 1e-18 {
                    #[cfg(feature = "brep-csg")]
                    {
                        analytic.numeric_hits += 1;
                    }
                    a_segs[i].push(s);
                    b_segs[j].push(s);
                }
            } else if coplanar(a, b) {
                let poly = coplanar::coplanar_overlap_poly(a, b);
                for k in 0..poly.len() {
                    let s = (poly[k], poly[(k + 1) % poly.len()]);
                    if sqlen(sub(s.1, s.0)) > 1e-18 {
                        a_segs[i].push(s);
                        b_segs[j].push(s);
                    }
                }
            }
        }
    }
    // Global vertex/edge snapping. Intersection endpoints from `tri_tri_segment`
    // carry ~1e-6 error, which leaves a point a hair off the mesh edge it should
    // lie on — producing near-collinear slivers within a face *and*, worse, a seam
    // that doesn't quite coincide between the two meshes. Snapping every endpoint to
    // its canonical position (a nearby mesh vertex, else onto a nearby mesh edge)
    // fixes both, and — being a pure function of the point — snaps the *shared*
    // endpoint identically on both sides, so the seam stays watertight.
    let mut verts: Vec<V3> = Vec::new();
    let mut edges: Vec<(V3, V3)> = Vec::new();
    let mut eseen: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
    let mut vseen: std::collections::HashSet<(i64, i64, i64)> = std::collections::HashSet::new();
    for t in ta.iter().chain(tb.iter()) {
        for k in 0..3 {
            let vk = (
                (t[k][0] * 1e6).round() as i64,
                (t[k][1] * 1e6).round() as i64,
                (t[k][2] * 1e6).round() as i64,
            );
            if vseen.insert(vk) {
                verts.push(t[k]);
            }
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if eseen.insert(ekey(a, b)) {
                edges.push((a, b));
            }
        }
    }
    // Global set of *all* intersection points (both meshes), snapped. A point where
    // one mesh's cuts meet (a segment endpoint there) can fall in the middle of the
    // other mesh's seam segment; feeding this global set to both refinements lets
    // each face split its seam at the other mesh's points too — so the shared seam
    // is subdivided identically on both sides (watertight across the meshes).
    let mut global_pts: Vec<V3> = Vec::new();
    for (a, b) in a_segs.iter().chain(b_segs.iter()).flatten() {
        global_pts.push(snap_to_mesh(*a, &verts, &edges));
        global_pts.push(snap_to_mesh(*b, &verts, &edges));
    }
    Some((
        refine_with(ta, &a_segs, &verts, &edges, &global_pts)?,
        refine_with(tb, &b_segs, &verts, &edges, &global_pts)?,
    ))
}

/// Snap a point to a coincident mesh vertex (within ~1e-4) if there is one, else
/// leave it. Seam consistency across the two meshes comes from `corefine` *sharing*
/// each intersection segment (so both sides carry the identical point); this only
/// welds a computed point that lands essentially on an existing mesh vertex. Edge
/// snapping is deliberately avoided — with a loose tolerance it misroutes points on
/// fine tessellations; within-face slivers are handled by the degeneracy filter and
/// the T-junction heal instead. `edges` is retained for signature stability.
fn snap_to_mesh(p: V3, verts: &[V3], _edges: &[(V3, V3)]) -> V3 {
    let mut best: Option<(f64, V3)> = None;
    for &v in verts {
        let d = sqlen(sub(p, v));
        if d < 1e-8 && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, v));
        }
    }
    best.map_or(p, |(_, v)| v)
}

/// Refine a mesh against its cut segments, one flat **face** (coplanar group) at a
/// time. Doing the whole face in a single CDT — rather than each triangle
/// independently — means a cut crossing the face's internal diagonals produces
/// shared vertices, not T-junctions; and any cut endpoint that lands on a face's
/// boundary edge is inserted into the adjacent face too, so faces line up exactly.
fn refine_with(
    tris: &[[V3; 3]],
    segs: &[Vec<(V3, V3)>],
    verts: &[V3],
    edges: &[(V3, V3)],
    allpts: &[V3],
) -> Option<Vec<[V3; 3]>> {
    // Snap every cut endpoint to its canonical mesh position first. This removes the
    // ~1e-6 slop from tri_tri_segment (which otherwise leaves near-collinear slivers
    // within a face) and — because the snap is a pure function shared with the other
    // mesh's refinement — keeps the seam coincident on both sides.
    let segs: Vec<Vec<(V3, V3)>> = segs
        .iter()
        .map(|v| {
            v.iter()
                .map(|&(a, b)| (snap_to_mesh(a, verts, edges), snap_to_mesh(b, verts, edges)))
                .collect()
        })
        .collect();
    // Group triangles by their oriented plane (one entry per flat face); sorted for
    // deterministic output (HashMap iteration order is otherwise randomized).
    let mut groups: HashMap<(i64, i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        groups.entry(plane_key(t)).or_default().push(i);
    }
    let mut groups: Vec<(PlaneKey, Vec<usize>)> = groups.into_iter().collect();
    groups.sort_by_key(|(k, _)| *k);
    // `allpts` (passed in) is the global cut-point set from *both* meshes — a face
    // splits its edges at any of these that land on them, so shared edges between
    // faces (and across the two meshes) are subdivided identically (no T-junction).

    let mut out = Vec::new();
    for (_, group) in &groups {
        let group = group.as_slice();
        let gsegs: Vec<(V3, V3)> = group
            .iter()
            .flat_map(|&i| segs[i].iter().copied())
            .collect();
        // The face's boundary = edges used by exactly one triangle of the group.
        let mut ecount: HashMap<EdgeKey, ((V3, V3), u32)> = HashMap::new();
        for &i in group {
            let t = &tris[i];
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                ecount.entry(ekey(a, b)).or_insert(((a, b), 0)).1 += 1;
            }
        }
        let mut boundary: Vec<(V3, V3)> = ecount
            .values()
            .filter(|(_, c)| *c == 1)
            .map(|(e, _)| *e)
            .collect();
        boundary.sort_by_key(|&(a, b)| ekey(a, b)); // deterministic CDT input order

        // Does any global cut point land on this face (on its plane and an edge)?
        let n = normalize(face_normal(&tris[group[0]]));
        let o = tris[group[0]][0];
        let touched = gsegs.is_empty()
            && allpts.iter().any(|&p| {
                dot(sub(p, o), n).abs() <= tolerance::PLANE_GATE
                    && boundary.iter().any(|&(a, b)| on_segment_interior(a, b, p))
            });

        if gsegs.is_empty() && !touched {
            for &i in group {
                out.push(tris[i]); // untouched face — pass through unchanged
            }
        } else {
            out.extend(cdt_face(group, tris, &boundary, &gsegs, allpts)?);
        }
    }
    Some(out)
}

fn sgn(x: f64) -> i32 {
    (x > 0.0) as i32 - (x < 0.0) as i32
}

const RAY_DIRS: [V3; 6] = [
    [0.31, 0.53, 0.79],
    [0.71, 0.13, 0.69],
    [0.47, 0.87, 0.19],
    [0.91, 0.29, 0.33],
    [0.17, 0.61, 0.77],
    [0.83, 0.41, 0.23],
];

/// Exact ray-parity crossing count along one direction, over the BVH candidates
/// only. `None` if the ray is degenerate (grazes a plane/edge) → caller retries.
fn ray_parity(
    bvh: &bvh::Bvh,
    tris: &[[V3; 3]],
    p: V3,
    d: V3,
    cand: &mut Vec<usize>,
) -> Option<bool> {
    let far = [d[0] * 1e6, d[1] * 1e6, d[2] * 1e6];
    bvh.ray_leaves(p, far, cand);
    let q = [p[0] + far[0], p[1] + far[1], p[2] + far[2]];
    let mut crossings = 0u32;
    for &i in cand.iter() {
        let t = &tris[i];
        let s1 = sgn(orient3d(p, t[0], t[1], t[2]));
        let s2 = sgn(orient3d(q, t[0], t[1], t[2]));
        if s1 == 0 || s2 == 0 {
            return None;
        }
        if s1 == s2 {
            continue;
        }
        let a = sgn(orient3d(p, q, t[0], t[1]));
        let b = sgn(orient3d(p, q, t[1], t[2]));
        let c = sgn(orient3d(p, q, t[2], t[0]));
        if a == 0 || b == 0 || c == 0 {
            return None;
        }
        if a == b && b == c {
            crossings += 1;
        }
    }
    Some(crossings % 2 == 1)
}

/// Exact ray-parity inside test for a closed mesh, BVH-accelerated: only the
/// triangles whose AABB the ray crosses are tested, and each with exact
/// `orient3d`. Robust where winding is ambiguous (point on/near the surface).
fn point_inside(bvh: &bvh::Bvh, tris: &[[V3; 3]], p: V3) -> bool {
    let mut cand = Vec::new();
    for d in RAY_DIRS {
        if let Some(inside) = ray_parity(bvh, tris, p, d, &mut cand) {
            return inside;
        }
    }
    point_in_mesh(tris, p) // every direction grazed (practically impossible)
}

/// Is `v` strictly interior to segment `a→b` (collinear, between, not an endpoint)?
fn on_segment_interior(a: V3, b: V3, v: V3) -> bool {
    if sqlen(sub(v, a)) < 1e-14 || sqlen(sub(v, b)) < 1e-14 {
        return false;
    }
    let ab = sub(b, a);
    let ablen = sqlen(ab);
    if ablen < 1e-18 {
        return false;
    }
    if sqlen(cross(ab, sub(v, a))) > 1e-12 * ablen {
        return false; // not on the line
    }
    let t = dot(sub(v, a), ab) / ablen;
    t > 1e-9 && t < 1.0 - 1e-9
}

/// Weld coincident vertices (to a fine grid) and heal T-junctions, processing
/// **only boundary edges** (used once) — efficient because a near-watertight
/// mesh has few of those. Welding first collapses coincident-but-unmerged
/// vertices; then any vertex lying exactly on a boundary edge splits that edge.
/// Geometry-preserving (splits are on the edge; welds are within the grid).
fn repair(tris: &[[V3; 3]]) -> Vec<[V3; 3]> {
    let key = |p: V3| {
        (
            (p[0] * 1e5).round() as i64,
            (p[1] * 1e5).round() as i64,
            (p[2] * 1e5).round() as i64,
        )
    };
    let mut vmap: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut verts: Vec<V3> = Vec::new();
    let mut idx: Vec<[u32; 3]> = Vec::new();
    for tr in tris {
        let mut ix = [0u32; 3];
        for k in 0..3 {
            let ky = key(tr[k]);
            ix[k] = *vmap.entry(ky).or_insert_with(|| {
                verts.push(tr[k]);
                (verts.len() - 1) as u32
            });
        }
        if ix[0] != ix[1] && ix[1] != ix[2] && ix[2] != ix[0] {
            let (a, b, c) = (
                verts[ix[0] as usize],
                verts[ix[1] as usize],
                verts[ix[2] as usize],
            );
            let area = norm(cross(sub(b, a), sub(c, a)));
            let longest = sqlen(sub(b, a))
                .max(sqlen(sub(c, b)))
                .max(sqlen(sub(a, c)))
                .max(1e-30)
                .sqrt();
            if area / longest >= 1e-6 {
                idx.push(ix); // drop collinear slivers up front so heal can converge
            }
        }
    }
    // Spatial hash over the (fixed) vertex set, so the "is any vertex strictly on
    // this open edge?" test below is a local lookup instead of a scan over every
    // vertex. That scan made the whole routine O(passes · triangles · vertices):
    // healing a large float-fallback mesh (~150k triangles, ~77k vertices) took
    // tens of seconds, which is most of the cost of any boolean that falls back.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for v in &verts {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    let diag = if verts.is_empty() {
        1.0
    } else {
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
    };
    let cell = (diag / 128.0).max(1e-9);
    let ckey = |p: V3| -> PointKey {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, v) in verts.iter().enumerate() {
        grid.entry(ckey(*v)).or_default().push(i as u32);
    }
    // Lowest-indexed vertex lying strictly inside open edge (a,b), or None. Matching
    // the old `position(..)` (first in index order) keeps the output identical.
    let on_edge_vertex = |a: u32, b: u32, verts: &Vec<V3>| -> Option<u32> {
        let (pa, pb) = (verts[a as usize], verts[b as usize]);
        let (c0, c1) = (ckey(pa), ckey(pb));
        let (mut best, mut cells) = (None::<u32>, 0usize);
        for cx in c0.0.min(c1.0) - 1..=c0.0.max(c1.0) + 1 {
            for cy in c0.1.min(c1.1) - 1..=c0.1.max(c1.1) + 1 {
                for cz in c0.2.min(c1.2) - 1..=c0.2.max(c1.2) + 1 {
                    cells += 1;
                    if cells > 4096 {
                        return best; // pathologically long edge — stop widening
                    }
                    let Some(list) = grid.get(&(cx, cy, cz)) else {
                        continue;
                    };
                    for &i in list {
                        if i != a
                            && i != b
                            && best.is_none_or(|bi| i < bi)
                            && on_segment_interior(pa, pb, verts[i as usize])
                        {
                            best = Some(i);
                        }
                    }
                }
            }
        }
        best
    };

    // Bound growth: a seam that can't be healed makes splits cascade (each pass can
    // spawn a fresh T-junction), so idx would blow up exponentially. If it grows past
    // this, give up — the caller's manifold gate then falls back to the float CSG.
    let cap = tris.len() * 4 + 256;
    for _ in 0..64 {
        if idx.len() > cap {
            break;
        }
        let mut ec: HashMap<(u32, u32), i32> = HashMap::new();
        for tr in &idx {
            for k in 0..3 {
                let (mut a, mut b) = (tr[k], tr[(k + 1) % 3]);
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                *ec.entry((a, b)).or_insert(0) += 1;
            }
        }
        if ec.values().all(|&c| c == 2) {
            break;
        }
        let mut next = Vec::with_capacity(idx.len());
        let mut changed = false;
        for tr in &idx {
            let mut split = None;
            for e in 0..3 {
                let (a, b) = (tr[e], tr[(e + 1) % 3]);
                let (mut x, mut y) = (a, b);
                if x > y {
                    std::mem::swap(&mut x, &mut y);
                }
                if ec.get(&(x, y)) == Some(&2) {
                    continue; // interior edge, fine
                }
                if let Some(vi) = on_edge_vertex(a, b, &verts) {
                    split = Some((e, vi));
                    break;
                }
            }
            match split {
                Some((e, vi)) => {
                    let (a, b, c) = (tr[e], tr[(e + 1) % 3], tr[(e + 2) % 3]);
                    next.push([a, vi, c]);
                    next.push([vi, b, c]);
                    changed = true;
                }
                None => next.push(*tr),
            }
        }
        idx = next;
        if !changed {
            break;
        }
    }
    idx.iter()
        .map(|t| {
            [
                verts[t[0] as usize],
                verts[t[1] as usize],
                verts[t[2] as usize],
            ]
        })
        .collect()
}

/// Is the triangle soup a closed 2-manifold (every welded edge used exactly
/// twice)? The safety gate: a boolean that fails this is refused, never emitted.
pub(crate) fn is_closed_manifold(tris: &[[V3; 3]]) -> bool {
    if tris.is_empty() {
        return false;
    }
    let key = |p: V3| -> PointKey {
        (
            (p[0] * 1e6).round() as i64,
            (p[1] * 1e6).round() as i64,
            (p[2] * 1e6).round() as i64,
        )
    };
    let mut edges: HashMap<SnappedEdge, i32> = HashMap::new();
    for t in tris {
        for k in 0..3 {
            let (mut u, mut v) = (key(t[k]), key(t[(k + 1) % 3]));
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            *edges.entry((u, v)).or_insert(0) += 1;
        }
    }
    edges.values().all(|&c| c == 2)
}

fn face_normal(t: &[V3; 3]) -> V3 {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
}

fn point_in_tri_2d(o: &[[f64; 2]; 3], p: [f64; 2]) -> bool {
    let s0 = orient2d(o[0], o[1], p);
    let s1 = orient2d(o[1], o[2], p);
    let s2 = orient2d(o[2], o[0], p);
    (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
}

/// If `t` lies fully within a coincident (exactly coplanar) face of `other`,
/// return that face's unnormalized outward normal; else `None`.
fn coincident_face(
    t: &[V3; 3],
    other: &[[V3; 3]],
    other_boxes: &[(V3, V3)],
    // When both meshes carry provenance, surface identity replaces the exact
    // coplanarity gate — see `analytic::surfaces_coincide` for why that gate is
    // unreliable on *refined* triangles.
    #[cfg(feature = "brep-csg")] same_surface: Option<&SameSurfaceFn<'_>>,
) -> Option<V3> {
    // Degenerate slivers (collinear verts) can read as "coplanar" with a face
    // whose plane they happen to lie on — classify those by winding as before.
    if sqlen(face_normal(t)) < 1e-12 {
        return None;
    }
    let bt = aabb(t);
    for (oi, (o, ob)) in other.iter().zip(other_boxes).enumerate() {
        if !aabb_overlap(&bt, ob) {
            continue;
        }
        #[cfg(feature = "brep-csg")]
        let related = match same_surface {
            Some(f) => coplanar(o, t) || f(t, oi),
            None => coplanar(o, t),
        };
        #[cfg(not(feature = "brep-csg"))]
        let related = coplanar(o, t);
        if !related {
            continue;
        }
        let _ = oi;
        let n = face_normal(o);
        if sqlen(n) < 1e-18 {
            continue;
        }
        let ux = normalize(sub(o[1], o[0]));
        let uy = normalize(cross(n, ux));
        let origin = o[0];
        let pr = |p: V3| [dot(sub(p, origin), ux), dot(sub(p, origin), uy)];
        let o2 = [pr(o[0]), pr(o[1]), pr(o[2])];
        // Centroid test (not all-vertices): robust when the two faces are
        // tessellated with different diagonals. After refinement each sub-triangle
        // is wholly inside or outside the coincident region, so the centroid decides.
        if point_in_tri_2d(&o2, pr(centroid(t))) {
            return Some(n);
        }
    }
    None
}

/// "Are this refined sub-triangle and that original triangle of the other mesh
/// on the same surface?" — supplied by the analytic layer when both meshes carry
/// provenance.
#[cfg(feature = "brep-csg")]
type SameSurfaceFn<'a> = dyn Fn(&[V3; 3], usize) -> bool + Sync + 'a;

/// Keep/drop rule for a sub-triangle coincident with the other solid's face.
/// `COPLANAR_ALIGNED` (same normal direction) keeps exactly one copy — by
/// convention the A operand's; `COPLANAR_OPPOSITE` drops both, except difference
/// keeps A's (its boundary survives the subtraction).
fn coplanar_keep(op: Op, is_a: bool, aligned: bool) -> bool {
    match (op, aligned) {
        (Op::Union, true) | (Op::Intersection, true) => is_a,
        (Op::Difference, false) => is_a,
        _ => false,
    }
}

/// Full mesh-arrangement boolean: refine both meshes along their intersection,
/// classify each sub-triangle by winding number, assemble. Returns `Some` only
/// when the result passes the closed-manifold gate; `None` otherwise (so the
/// caller reports `NeedsArrangement` rather than trusting an unverified mesh).
///
/// The refined-and-classified triangle soup, *before* the closed-manifold gate.
/// Diagnostics only: this is what the gate rejects when it rejects something, so
/// it is the only way to see where a deferral actually comes from.
#[cfg(feature = "brep-csg")]
#[doc(hidden)]
pub fn debug_refined(a: &BufferGeometry, b: &BufferGeometry, op: Op) -> Vec<[V3; 3]> {
    arrangement_soup(a, b, op).unwrap_or_default()
}

pub fn boolean_arrangement(
    a: &BufferGeometry,
    b: &BufferGeometry,
    op: Op,
) -> Option<BufferGeometry> {
    let out = arrangement_soup(a, b, op)?;
    // Most results are already watertight. Otherwise weld + boundary-heal, and
    // accept it only if that actually closes the mesh (never emit an unverified
    // result — so we still fall back rather than trust a repair that didn't work).
    let out = if is_closed_manifold(&out) {
        out
    } else {
        let fixed = repair(&out);
        if is_closed_manifold(&fixed) {
            fixed
        } else {
            return None;
        }
    };
    Some(build_geometry(&out))
}

/// Refine both meshes against their intersection and classify every
/// sub-triangle. Split out of [`boolean_arrangement`] so the diagnostics above
/// can see the result the manifold gate is about to judge.
fn arrangement_soup(a: &BufferGeometry, b: &BufferGeometry, op: Op) -> Option<Vec<[V3; 3]>> {
    let ta = triangles(a);
    let tb = triangles(b);
    let bvh_a = bvh::Bvh::build(&ta);
    let bvh_b = bvh::Bvh::build(&tb);
    #[cfg(feature = "brep-csg")]
    let mut an = analytic::Analytic::new(a.surface_table(), b.surface_table());
    let (ra, rb) = corefine(
        &ta,
        &tb,
        &bvh_b,
        #[cfg(feature = "brep-csg")]
        &mut an,
    )?;
    let ta_boxes: Vec<(V3, V3)> = ta.iter().map(aabb).collect();
    let tb_boxes: Vec<(V3, V3)> = tb.iter().map(aabb).collect();

    // Classifying a sub-triangle means a BVH ray-parity test against the *other*
    // mesh — independent per triangle, so map it in parallel where the build
    // allows; `par_map` keeps input order, so the output triangle list is
    // identical either way. Not the dominant cost, though it reads like it should
    // be: measured over a whole animation it is 9.4% of the boolean against
    // corefine's 90.6%, effectively all of which is the per-face CDT.
    //
    // Ray parity is a coin flip for a point lying *on* the surface it is tested
    // against, so this looks like the obvious suspect for the kernel's remaining
    // non-watertight results (4 of 17 booleans on the reference model decline and
    // fall back). It was measured and it is not. Relaxing `coincident_face`'s
    // exact-coplanarity gate to a distance test on the centroid — so a
    // sub-triangle sitting on a face is classified by the coplanar rule instead of
    // by parity — changes nothing at any tolerance from 0 up to 1e-4 of the model
    // extent, at which point it starts misclassifying genuinely distinct faces and
    // the volume balloons. The failing sub-triangles are simply not near the other
    // surface.
    //
    // What the bad edges actually are, counted on the declined soups: about a
    // third have another vertex lying strictly inside them (the two refinements
    // subdivided the shared seam differently — a T-junction across the seam), and
    // the rest are genuinely dangling, with no vertex inside and no matching
    // triangle from either side. See the CHANGELOG's known-issues entry.
    use crate::utils::parallel::par_map;

    // Surface identity as a fallback for the exact coplanarity gate.
    //
    // A refined sub-triangle has no table entry — `refine_with` returns a fresh
    // list — so its surface has to be recovered. Measuring distance to each
    // candidate surface does *not* work: a tessellated cylinder's facet is a
    // chord, and its interior points sit a full sagitta off the analytic
    // cylinder (0.0096 for `$fn = 32` at radius 2, four orders of magnitude
    // above any numerical tolerance). Recovering it by *plane key* does work,
    // and reuses the grouping `refine_with` already computes: a sub-triangle is
    // coplanar with the facet it was cut from, and the key's 1e-4 quantization
    // absorbs the CDT's few-ULP drift while still separating adjacent facets.
    #[cfg(feature = "brep-csg")]
    let (table_a, table_b) = (a.surface_table(), b.surface_table());
    #[cfg(feature = "brep-csg")]
    let plane_surface = |tris: &[[V3; 3]], table: &crate::brep::SurfaceTable| {
        let mut m: HashMap<(i64, i64, i64, i64), u32> = HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            if let Some(si) = table.surface_index_of(i) {
                m.entry(plane_key(t)).or_insert(si as u32);
            }
        }
        m
    };
    #[cfg(feature = "brep-csg")]
    let (map_a, map_b) = match (table_a, table_b) {
        (Some(x), Some(y)) => (plane_surface(&ta, x), plane_surface(&tb, y)),
        _ => (HashMap::new(), HashMap::new()),
    };
    #[cfg(feature = "brep-csg")]
    let same_a: Option<Box<SameSurfaceFn<'_>>> = match (table_a, table_b) {
        (Some(x), Some(y)) => Some(Box::new(move |t: &[V3; 3], oi: usize| {
            let (Some(&si), Some(sb)) = (map_a.get(&plane_key(t)), y.surface_of(oi)) else {
                return false;
            };
            analytic::surfaces_coincide(&x.surfaces()[si as usize], sb)
        })),
        _ => None,
    };
    #[cfg(feature = "brep-csg")]
    let same_b: Option<Box<SameSurfaceFn<'_>>> = match (table_a, table_b) {
        (Some(x), Some(y)) => Some(Box::new(move |t: &[V3; 3], oi: usize| {
            let (Some(&si), Some(sa)) = (map_b.get(&plane_key(t)), x.surface_of(oi)) else {
                return false;
            };
            analytic::surfaces_coincide(&y.surfaces()[si as usize], sa)
        })),
        _ => None,
    };

    let kept_a = par_map(&ra, |t| {
        let keep = if let Some(nf) = coincident_face(
            t,
            &tb,
            &tb_boxes,
            #[cfg(feature = "brep-csg")]
            same_a.as_deref(),
        ) {
            coplanar_keep(op, true, dot(face_normal(t), nf) > 0.0)
        } else {
            let inside = point_inside(&bvh_b, &tb, centroid(t));
            match op {
                Op::Union | Op::Difference => !inside,
                Op::Intersection => inside,
            }
        };
        keep.then_some(*t)
    });
    let kept_b = par_map(&rb, |t| {
        if let Some(nf) = coincident_face(
            t,
            &ta,
            &ta_boxes,
            #[cfg(feature = "brep-csg")]
            same_b.as_deref(),
        ) {
            return coplanar_keep(op, false, dot(face_normal(t), nf) > 0.0).then_some(*t);
        }
        let inside = point_inside(&bvh_a, &ta, centroid(t));
        let (keep, flip) = match op {
            Op::Union => (!inside, false),
            Op::Intersection => (inside, false),
            Op::Difference => (inside, true),
        };
        keep.then(|| if flip { [t[0], t[2], t[1]] } else { *t })
    });

    let mut out: Vec<[V3; 3]> = Vec::with_capacity(kept_a.len() + kept_b.len());
    out.extend(kept_a.into_iter().flatten());
    out.extend(kept_b.into_iter().flatten());

    #[cfg(feature = "brep-csg")]
    if std::env::var("THREERS_CSG_DEBUG").is_ok() {
        eprintln!(
            "  [csg] active={} skip={} same={} numeric={} sharp={} maxmove={:e} | ra={} rb={} kept={}",
            an.active(),
            an.skipped,
            an.same_surface,
            an.numeric_hits,
            an.sharpened,
            an.max_sharpen,
            ra.len(),
            rb.len(),
            out.len()
        );
    }
    Some(out)
}

/// Try to make a mesh watertight by welding coincident vertices and healing
/// T-junctions (the seam cracks left by the float CSG on curved booleans).
/// Returns the repaired geometry **only if it is actually closed** — otherwise
/// `None`, so callers never trust a repair that didn't work.
pub fn repair_geometry(g: &BufferGeometry) -> Option<BufferGeometry> {
    let tris = triangles(g);
    if is_closed_manifold(&tris) {
        return None; // already watertight — nothing to do
    }
    let fixed = repair(&tris);
    is_closed_manifold(&fixed).then(|| build_geometry(&fixed))
}

/// Boolean of two solids by winding-number classification.
///
/// When surfaces don't cross, classifies whole triangles directly. When they do,
/// runs the mesh arrangement ([`boolean_arrangement`]) behind a closed-manifold
/// gate, reporting `BooleanOutcome::NeedsArrangement` only if that can't be
/// verified.
pub fn boolean(a: &BufferGeometry, b: &BufferGeometry, op: Op) -> BooleanOutcome {
    let ta = triangles(a);
    let tb = triangles(b);
    let bvh_b = bvh::Bvh::build(&tb);
    if interacts(&ta, &tb, &bvh_b) {
        return match boolean_arrangement(a, b, op) {
            Some(g) => BooleanOutcome::Exact(g),
            None => BooleanOutcome::NeedsArrangement,
        };
    }

    let mut out: Vec<[V3; 3]> = Vec::new();
    let bvh_a = bvh::Bvh::build(&ta);

    // A's triangles: keep the ones on the side the op wants.
    let keep_a_inside = op == Op::Intersection;
    for t in &ta {
        let inside = point_inside(&bvh_b, &tb, centroid(t));
        if inside == keep_a_inside {
            out.push(*t);
        }
    }

    // B's triangles: union keeps the outside; intersection/difference keep the
    // inside, and difference flips their winding (the void wall faces inward).
    for t in &tb {
        let inside = point_inside(&bvh_a, &ta, centroid(t));
        let (keep, flip) = match op {
            Op::Union => (!inside, false),
            Op::Intersection => (inside, false),
            Op::Difference => (inside, true),
        };
        if keep {
            out.push(if flip { [t[0], t[2], t[1]] } else { *t });
        }
    }

    BooleanOutcome::Exact(build_geometry(&out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cube, sphere};

    fn box_tris(size: f32, at: [f32; 3]) -> Vec<[V3; 3]> {
        triangles(&cube([size, size, size]).translate(at).to_geometry())
    }

    fn signed_volume(g: &BufferGeometry) -> f64 {
        triangles(g)
            .iter()
            .map(|t| dot(t[0], cross(t[1], t[2])))
            .sum::<f64>()
            / 6.0
    }

    /// Curved∧curved exact CSG: two faceted surfaces genuinely crossing (spheres,
    /// cylinders, cones, and mixed pairs) resolve to a watertight mesh for all
    /// three ops — the case the float kernel can't do (its sphere∩sphere is
    /// degenerate). The oracle is *self-consistency* — V(A∪B)=V(A)+V(B)−V(A∩B) and
    /// V(A−B)=V(A)−V(A∩B) — plus watertightness, since the float result is the
    /// unreliable one here. (A perfectly symmetric identical-primitive-on-axis
    /// overlap is a measure-zero degeneracy that still falls back to float; every
    /// case here is a real transversal crossing.)
    #[test]
    fn curved_curved_booleans_are_exact() {
        use crate::{cone, cylinder, sphere_fn};
        type SolidFn = Box<dyn Fn() -> crate::Solid>;
        let cases: Vec<(&str, SolidFn, SolidFn)> = vec![
            (
                "sph∪sph deep asym",
                Box::new(|| sphere_fn(1.3, 32)),
                Box::new(|| sphere_fn(1.1, 32).translate([0.9, 0.17, 0.11])),
            ),
            (
                "sph∪sph shallow",
                Box::new(|| sphere_fn(1.3, 32)),
                Box::new(|| sphere_fn(1.0, 32).translate([1.8, 0.2, 0.1])),
            ),
            (
                "cyl∪cyl parallel",
                Box::new(|| cylinder(3.0, 1.0)),
                Box::new(|| cylinder(3.0, 1.0).translate([1.2, 0.0, 0.0])),
            ),
            (
                "cyl⊥cyl cross",
                Box::new(|| cylinder(4.0, 0.8)),
                Box::new(|| cylinder(4.0, 0.8).rotate_x(std::f32::consts::FRAC_PI_2)),
            ),
            (
                "sph∪cyl",
                Box::new(|| sphere_fn(1.5, 32)),
                Box::new(|| cylinder(4.0, 0.7)),
            ),
            (
                "cone∪cyl",
                Box::new(|| cone(3.0, 1.4, 0.2)),
                Box::new(|| cylinder(4.0, 0.6).translate([0.6, 0.0, 0.0])),
            ),
        ];
        for (name, fa, fb) in &cases {
            let (a, b) = (fa().to_geometry(), fb().to_geometry());
            let (va, vb) = (signed_volume(&a).abs(), signed_volume(&b).abs());
            let vol = |op| match boolean(&a, &b, op) {
                BooleanOutcome::Exact(g) => {
                    assert!(
                        is_closed_manifold(&triangles(&g)),
                        "{name} {op:?}: not watertight"
                    );
                    signed_volume(&g).abs()
                }
                BooleanOutcome::NeedsArrangement => panic!("{name} {op:?}: fell back to float"),
            };
            let (uv, iv, dv) = (vol(Op::Union), vol(Op::Intersection), vol(Op::Difference));
            // Self-consistency across the three ops (float-oracle-independent).
            assert!(
                (uv - (va + vb - iv)).abs() / va < 0.02,
                "{name}: union inconsistent"
            );
            assert!(
                (dv - (va - iv)).abs() / va < 0.02,
                "{name}: difference inconsistent"
            );
            assert!(
                iv > 0.0 && iv < va.min(vb),
                "{name}: intersection out of range"
            );
        }
    }

    #[test]
    fn arrangement_resolves_a_real_overlap() {
        // Sphere piercing a box — a genuine transversal overlap. The arrangement
        // resolves the difference to a watertight mesh whose volume matches the
        // existing float CsgEvaluator (used purely as an oracle).
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = sphere(1.3).to_geometry();
        let float = cube([2.0, 2.0, 2.0]).difference(sphere(1.3)).to_geometry();

        let g =
            boolean_arrangement(&a, &b, Op::Difference).expect("difference resolves + passes gate");
        let (va, vf) = (signed_volume(&g).abs(), signed_volume(&float).abs());
        assert!(
            (va - vf).abs() / vf < 1e-3,
            "arrangement {va:.5} vs float {vf:.5}"
        );
        assert!(is_closed_manifold(&triangles(&g)), "watertight");
        assert!(matches!(
            boolean(&a, &b, Op::Difference),
            BooleanOutcome::Exact(_)
        ));
    }

    #[test]
    fn all_ops_watertight_on_oblique_boxes() {
        // Two obliquely-rotated boxes: a fully transversal, non-degenerate
        // overlap (no axis-aligned/coplanar faces, no polar slivers). All three
        // ops must resolve to a watertight mesh matching the float oracle —
        // proving union AND intersection work, not just difference.
        let a = cube([2.0, 2.0, 2.0]);
        let b = cube([2.0, 2.0, 2.0])
            .rotate_x(0.6)
            .rotate_y(0.4)
            .translate([0.7, 0.5, 0.3]);
        let (ga, gb) = (a.clone().to_geometry(), b.clone().to_geometry());
        for op in [Op::Difference, Op::Union, Op::Intersection] {
            let float = match op {
                Op::Difference => a.clone().difference(b.clone()),
                Op::Union => a.clone().union(b.clone()),
                Op::Intersection => a.clone().intersection(b.clone()),
            }
            .to_geometry();
            let g = boolean_arrangement(&ga, &gb, op)
                .unwrap_or_else(|| panic!("{op:?}: oblique boxes should resolve"));
            let (va, vf) = (signed_volume(&g).abs(), signed_volume(&float).abs());
            assert!(
                (va - vf).abs() / vf < 1e-2,
                "{op:?}: arrangement {va:.4} vs float {vf:.4}"
            );
            assert!(is_closed_manifold(&triangles(&g)), "{op:?}: watertight");
            assert!(
                matches!(boolean(&ga, &gb, op), BooleanOutcome::Exact(_)),
                "{op:?}: Exact"
            );
        }
    }

    #[test]
    fn fuzz_boolean_volume_identities() {
        // Adversarial property test: many random obliquely-rotated box overlaps.
        // For every pair where all three ops resolve to Exact, the results must
        // satisfy the inclusion–exclusion identities exactly (oracle-free):
        //   V(A∪B) + V(A∩B) == V(A) + V(B)
        //   V(A−B) + V(A∩B) == V(A)
        // A boolean that miscounts a region breaks these by whole percent.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f64) / ((1u64 << 24) as f64) // [0,1)
        };
        let mut checked = 0;
        for _ in 0..60 {
            let sa = 1.6 + rng() * 0.8;
            let sb = 1.4 + rng() * 0.9;
            let a = cube([sa as f32, sa as f32, sa as f32]);
            let b = cube([sb as f32, sb as f32, sb as f32])
                .rotate_x((0.25 + rng() * 0.9) as f32)
                .rotate_y((0.25 + rng() * 0.9) as f32)
                .rotate_z((0.25 + rng() * 0.9) as f32)
                .translate([
                    (rng() * 1.4 - 0.7) as f32,
                    (rng() * 1.4 - 0.7) as f32,
                    (rng() * 1.4 - 0.7) as f32,
                ]);
            let (ga, gb) = (a.to_geometry(), b.to_geometry());
            let (u, d, i) = (
                boolean(&ga, &gb, Op::Union),
                boolean(&ga, &gb, Op::Difference),
                boolean(&ga, &gb, Op::Intersection),
            );
            if let (
                BooleanOutcome::Exact(gu),
                BooleanOutcome::Exact(gd),
                BooleanOutcome::Exact(gi),
            ) = (u, d, i)
            {
                let va = signed_volume(&ga).abs();
                let vb = signed_volume(&gb).abs();
                let (vu, vd, vi) = (
                    signed_volume(&gu).abs(),
                    signed_volume(&gd).abs(),
                    signed_volume(&gi).abs(),
                );
                assert!(
                    ((vu + vi) - (va + vb)).abs() / va < 1e-2,
                    "inclusion-exclusion: u{vu:.4}+i{vi:.4} vs a{va:.4}+b{vb:.4}"
                );
                assert!(
                    ((vd + vi) - va).abs() / va < 1e-2,
                    "difference identity: d{vd:.4}+i{vi:.4} vs a{va:.4}"
                );
                for g in [&gu, &gd, &gi] {
                    assert!(
                        is_closed_manifold(&triangles(g)),
                        "Exact result must be watertight"
                    );
                }
                checked += 1;
            }
        }
        assert!(
            checked >= 8,
            "expected several fully-resolved random cases, got {checked}"
        );
    }

    #[test]
    fn coplanar_keep_truth_table() {
        // Union/Intersection: aligned keeps A's copy only; opposite drops both.
        for op in [Op::Union, Op::Intersection] {
            assert!(coplanar_keep(op, true, true) && !coplanar_keep(op, false, true));
            assert!(!coplanar_keep(op, true, false) && !coplanar_keep(op, false, false));
        }
        // Difference: aligned drops both; opposite keeps A's boundary only.
        assert!(!coplanar_keep(Op::Difference, true, true));
        assert!(
            coplanar_keep(Op::Difference, true, false)
                && !coplanar_keep(Op::Difference, false, false)
        );
    }

    #[test]
    fn coplanar_stacked_boxes_resolve() {
        // Two boxes sharing the z=1 face exactly (A's +z coincident-opposite B's −z).
        // A previously-refused coplanar case — now handled by the aligned/opposite rule.
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = cube([2.0, 2.0, 2.0])
            .translate([0.0, 0.0, 2.0])
            .to_geometry();
        // Union = 2×2×4 box (coincident-opposite faces dropped) → vol 16.
        match boolean(&a, &b, Op::Union) {
            BooleanOutcome::Exact(g) => {
                assert!(
                    (signed_volume(&g).abs() - 16.0).abs() < 1e-3,
                    "stacked union = 2×2×4 box"
                );
                assert!(is_closed_manifold(&triangles(&g)), "watertight");
            }
            BooleanOutcome::NeedsArrangement => panic!("stacked union should resolve"),
        }
        // Difference A−B: B only touches A (no volume overlap) → A−B = A, vol 8.
        // (Coincident-opposite: difference keeps A's face, drops B's.)
        match boolean(&a, &b, Op::Difference) {
            BooleanOutcome::Exact(g) => {
                assert!(
                    (signed_volume(&g).abs() - 8.0).abs() < 1e-3,
                    "A−B = A, vol 8"
                );
                assert!(is_closed_manifold(&triangles(&g)), "watertight");
            }
            BooleanOutcome::NeedsArrangement => panic!("stacked difference should resolve"),
        }
    }

    #[test]
    fn coplanar_overlapping_boxes_union_resolves() {
        // Two axis-aligned boxes overlapping in x∈[0,1] — the headline coplanar
        // case (partial coincidence). Union resolves via face-splitting along the
        // coincident boundary + the aligned rule.
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = cube([2.0, 2.0, 2.0])
            .translate([1.0, 0.0, 0.0])
            .to_geometry();
        // All three resolve (T-junction healing stitches the coincident seam):
        // union = 3×2×2 = 12, difference = 1×2×2 = 4, intersection = 1×2×2 = 4.
        for (op, want) in [
            (Op::Union, 12.0),
            (Op::Difference, 4.0),
            (Op::Intersection, 4.0),
        ] {
            match boolean(&a, &b, op) {
                BooleanOutcome::Exact(g) => {
                    assert!(
                        (signed_volume(&g).abs() - want).abs() < 1e-3,
                        "{op:?} volume = {want}"
                    );
                    assert!(is_closed_manifold(&triangles(&g)), "{op:?} watertight");
                }
                BooleanOutcome::NeedsArrangement => {
                    panic!("{op:?} overlapping boxes should resolve")
                }
            }
        }
    }

    #[test]
    fn square_through_hole_resolves() {
        // cube(10) minus a 4×4 box poking through z → a square hole, volume 840.
        let a = cube([10.0, 10.0, 10.0]).to_geometry();
        let b = cube([4.0, 4.0, 12.0]).to_geometry();
        match boolean(&a, &b, Op::Difference) {
            BooleanOutcome::Exact(g) => {
                assert!((signed_volume(&g).abs() - 840.0).abs() < 1e-2, "vol 840");
                assert!(is_closed_manifold(&triangles(&g)), "watertight hole");
            }
            BooleanOutcome::NeedsArrangement => panic!("square through-hole should resolve"),
        }
    }

    #[test]
    fn kernel_is_never_wrong_across_ops() {
        // Contract: for every op, the kernel either refuses (NeedsArrangement) or
        // returns a mesh that is watertight AND volume-matches the float oracle.
        // It must never emit a mesh that disagrees. (Union/Intersection currently
        // hit a few near-seam f64 misclassifications and are safely refused —
        // exact predicates in M2 will close that gap.)
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = sphere(1.3).to_geometry();
        let oracle = |op: Op| match op {
            Op::Difference => cube([2.0, 2.0, 2.0]).difference(sphere(1.3)).to_geometry(),
            Op::Union => cube([2.0, 2.0, 2.0]).union(sphere(1.3)).to_geometry(),
            Op::Intersection => cube([2.0, 2.0, 2.0])
                .intersection(sphere(1.3))
                .to_geometry(),
        };
        for op in [Op::Difference, Op::Union, Op::Intersection] {
            if let BooleanOutcome::Exact(g) = boolean(&a, &b, op) {
                let (va, vf) = (signed_volume(&g).abs(), signed_volume(&oracle(op)).abs());
                assert!(
                    (va - vf).abs() / vf < 1e-3,
                    "{op:?} claimed Exact but volume disagrees"
                );
                assert!(
                    is_closed_manifold(&triangles(&g)),
                    "{op:?} claimed Exact but not watertight"
                );
            }
        }
    }

    #[test]
    fn winding_inside_and_outside_a_box() {
        let t = box_tris(2.0, [0.0, 0.0, 0.0]);
        assert!(point_in_mesh(&t, [0.0, 0.0, 0.0]), "origin is inside");
        assert!(!point_in_mesh(&t, [5.0, 0.0, 0.0]), "far point is outside");
        assert!((winding_number(&t, [0.0, 0.0, 0.0]).abs() - 1.0).abs() < 1e-6);
        assert!(winding_number(&t, [5.0, 0.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn tri_tri_detects_crossing() {
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let crossing = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0]];
        let apart = [[0.5, 0.5, 1.0], [0.5, 0.5, 2.0], [1.5, 0.5, 2.0]];
        assert!(tri_tri_intersect(&a, &crossing));
        assert!(!tri_tri_intersect(&a, &apart));
    }

    #[test]
    fn disjoint_union_keeps_all_triangles() {
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = cube([2.0, 2.0, 2.0])
            .translate([10.0, 0.0, 0.0])
            .to_geometry();
        let na = triangles(&a).len();
        let nb = triangles(&b).len();
        match boolean(&a, &b, Op::Union) {
            BooleanOutcome::Exact(g) => assert_eq!(triangles(&g).len(), na + nb),
            BooleanOutcome::NeedsArrangement => panic!("disjoint boxes should not straddle"),
        }
    }

    #[test]
    fn nested_difference_is_exact_and_hollow() {
        // Small box strictly inside a big box: surfaces don't touch.
        let big = cube([4.0, 4.0, 4.0]).to_geometry();
        let small = cube([1.0, 1.0, 1.0]).to_geometry();
        let (nb, ns) = (triangles(&big).len(), triangles(&small).len());
        match boolean(&big, &small, Op::Difference) {
            BooleanOutcome::Exact(g) => {
                // All of big's shell (outside small) + all of small's shell (inside big).
                assert_eq!(triangles(&g).len(), nb + ns);
            }
            BooleanOutcome::NeedsArrangement => panic!("nested boxes should not straddle"),
        }
    }

    #[test]
    fn nested_intersection_is_the_inner_solid() {
        let big = cube([4.0, 4.0, 4.0]).to_geometry();
        let small = cube([1.0, 1.0, 1.0]).to_geometry();
        let ns = triangles(&small).len();
        match boolean(&big, &small, Op::Intersection) {
            BooleanOutcome::Exact(g) => assert_eq!(triangles(&g).len(), ns),
            BooleanOutcome::NeedsArrangement => panic!("nested boxes should not straddle"),
        }
    }

    #[test]
    fn genuinely_degenerate_still_refused_or_correct() {
        // A zero-thickness (degenerate) overlap must never yield a wrong mesh.
        let a = cube([2.0, 2.0, 2.0]).to_geometry();
        let b = cube([2.0, 2.0, 0.0]).to_geometry(); // flat
        if let BooleanOutcome::Exact(g) = boolean(&a, &b, Op::Union) {
            assert!(
                is_closed_manifold(&triangles(&g)),
                "any Exact result must be watertight"
            );
        }
    }
}

#[cfg(test)]
mod coverage {
    use super::*;
    use crate::{cone, cube, frustum, linear_extrude, polyhedron, sphere_fn, Solid};

    fn svol(g: &BufferGeometry) -> f64 {
        triangles(g)
            .iter()
            .map(|t| dot(t[0], cross(t[1], t[2])))
            .sum::<f64>()
            / 6.0
    }
    fn tetra() -> Solid {
        let p = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        polyhedron(
            &p,
            &[vec![0, 1, 2], vec![0, 3, 1], vec![0, 2, 3], vec![1, 3, 2]],
        )
    }

    /// Exercise every primitive shape against every other in all three booleans.
    /// Contract: every `Exact` result is watertight, and where all three ops of a
    /// pair resolve, the inclusion–exclusion volume identities hold. Prints a
    /// coverage map and guards a floor on the resolved count.
    #[test]
    fn all_shapes_all_ops_never_wrong() {
        let shapes: Vec<(&str, Solid)> = vec![
            ("cube", cube([2.0, 2.0, 2.0])),
            ("sphere", sphere_fn(1.2, 12)),
            ("cyl", frustum(2.0, 1.0, 1.0, 12)),
            ("cone", cone(2.0, 1.0, 0.0)),
            ("hex", frustum(2.0, 1.0, 1.0, 6)),
            ("pent", frustum(2.0, 1.0, 1.0, 5)),
            ("tri3", frustum(2.0, 1.0, 1.0, 3)),
            ("tet", tetra()),
            (
                "prism",
                linear_extrude(2.0, &[[-1.0, -0.8], [1.0, -0.8], [0.0, 1.0]])
                    .translate([0.0, 0.0, -1.0]),
            ),
        ];
        let off = [0.7, 0.5, 0.3];
        let ops = [Op::Union, Op::Difference, Op::Intersection];
        let (mut resolved, mut total, mut identity_checks) = (0usize, 0usize, 0usize);

        for (na, a) in &shapes {
            for (nb, b) in &shapes {
                let ga = a.clone().to_geometry();
                let gb = b.clone().translate(off).to_geometry();
                let mut vols = [None; 3];
                for (k, &op) in ops.iter().enumerate() {
                    total += 1;
                    if let BooleanOutcome::Exact(g) = boolean(&ga, &gb, op) {
                        let t = triangles(&g);
                        assert!(
                            is_closed_manifold(&t),
                            "{na}∖{nb} {op:?}: Exact but not watertight"
                        );
                        vols[k] = Some(svol(&g).abs());
                        resolved += 1;
                    }
                }
                // If a pair fully resolves, the volume identities must hold exactly.
                if let [Some(vu), Some(vd), Some(vi)] = vols {
                    let (va, vb) = (svol(&ga).abs(), svol(&gb).abs());
                    assert!(
                        ((vu + vi) - (va + vb)).abs() / va < 2e-2,
                        "{na}/{nb} incl-excl"
                    );
                    assert!(
                        ((vd + vi) - va).abs() / va < 2e-2,
                        "{na}/{nb} difference identity"
                    );
                    identity_checks += 1;
                }
            }
        }
        println!(
            "COVERAGE: {resolved}/{total} shape×op combinations resolve exactly; \
             {identity_checks} pairs fully resolved (identities verified)"
        );
        assert!(
            resolved * 100 / total >= 40,
            "coverage regressed: {resolved}/{total}"
        );
    }
}
