//! Predicates over a triangle soup.
//!
//! Everything here takes `&[Tri]` and nothing else. No index buffer, no vertex
//! sharing, no winding convention beyond "closed" where a function says it needs
//! one — because the meshes these run on come out of a boolean kernel, and the
//! only thing a boolean kernel reliably hands back is triangles.
//!
//! # Cost
//!
//! [`crate::assembly::mesh::nearest`] and [`crate::assembly::mesh::interfere`] are O(n·m) in the worst case and nothing like
//! it in practice. Both reject a triangle against the *whole* of the other body
//! before testing it against any of that body's triangles, which is the
//! difference between one test and m of them for every face that is nowhere
//! near — and on a real part most faces are nowhere near. `nearest` additionally
//! visits its triangles nearest-first, so its running bound starts tight enough
//! to reject the rest.
//!
//! Measured on a drilled block of 39,778 triangles against a pin of 252, in
//! release:
//!
//! | | |
//! |---|---|
//! | `nearest`, pin in the bore | 30 ms |
//! | `nearest`, pin 460 away | 0.8 ms |
//! | `interfere`, 0.2 of clearance | 14 ms |
//! | `rigid_key` | 1.3 ms |
//!
//! There is no acceleration structure. Building one costs more than the query at
//! the part counts a mechanism has, and it would be a second copy of
//! [`crate::mesh_bvh`] besides.

use super::Tri;
use std::collections::HashMap;

/// Per-triangle bounds.
///
/// The cheap half of every pair test: a box that does not reach the other box
/// cannot contain a triangle that reaches the other triangle.
pub fn tri_bbs(tris: &[Tri]) -> Vec<([f64; 3], [f64; 3])> {
    tris.iter().map(tri_bb).collect()
}

fn tri_bb(t: &Tri) -> ([f64; 3], [f64; 3]) {
    let mut lo = t[0];
    let mut hi = t[0];
    for v in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    (lo, hi)
}

/// Squared distance between two boxes. Zero when they overlap or touch.
fn bb_dist_sq(a: &([f64; 3], [f64; 3]), b: &([f64; 3], [f64; 3])) -> f64 {
    let mut d = 0.0;
    for k in 0..3 {
        let gap = (a.0[k] - b.1[k]).max(b.0[k] - a.1[k]).max(0.0);
        d += gap * gap;
    }
    d
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

/// Split a soup into the separate bodies it contains.
///
/// Triangles belong to the same body when they share a vertex, so this is the
/// connected components of the mesh — which for the output of a boolean is
/// exactly its separate solids. A model is one soup; an assembly is several
/// bodies inside it, and every cross-pose check starts by finding out which is
/// which.
///
/// `weld` is how close two vertices must be to count as one. Pass `0.0` to take
/// a millionth of the model's diagonal, which is loose enough to survive the
/// rounding a boolean leaves on a shared face and tight enough not to fuse parts
/// that merely touch.
///
/// # Bodies that touch come back as one
///
/// This is connectivity, and two solids drawn exactly against each other are
/// connected. A lid resting on a box shares that face's four corners to the
/// last bit, so they weld into one shell — whether the kernel unioned them or
/// an `assembly()` kept their meshes apart. No tolerance separates them again,
/// because the coordinates are not merely close, they are equal.
///
/// Measured, on a box with a lid sitting on it:
///
/// | Drawn | Shells |
/// |---|---|
/// | touching, unioned | 1 |
/// | touching, `assembly()` | 1 |
/// | 0.2 of clearance | 2 |
///
/// So a soup is not always enough to find the bodies in it, and quietly getting
/// one body where there are two is the sort of thing that goes unnoticed until a
/// cross-pose check compares the wrong things. Two ways out, in order of
/// preference: have the model *name* its bodies — which is what
/// `threers::openscad::mechanism`'s `part()` is for, and why the checks that
/// matter take declared parts rather than connected components — or draw the
/// clearance the parts really have, since a real lid does not share a plane with
/// its box.
///
/// A sealed void is the mirror image of this and is reported honestly: a cavity
/// fully enclosed in solid is a second surface, disconnected from the outer one,
/// and comes back as its own shell.
///
/// ```
/// use threers::assembly::shells;
/// // Two triangles sharing an edge are one body; two such pairs, far apart,
/// // are two.
/// let quad = |x: f64| vec![
///     [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]],
///     [[x + 1.0, 0.0, 0.0], [x + 1.0, 1.0, 0.0], [x, 1.0, 0.0]],
/// ];
/// let mut soup = quad(0.0);
/// soup.extend(quad(10.0));
/// assert_eq!(shells(&soup, 0.0).len(), 2);
/// ```
pub fn shells(tris: &[Tri], weld: f64) -> Vec<Vec<Tri>> {
    if tris.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = super::aabb(tris);
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    // A tolerance relative to the model, so the same model checks the same way
    // whether it is drawn in metres or in millimetres.
    let tol = if weld > 0.0 {
        weld
    } else if diag > 0.0 {
        diag * 1e-6
    } else {
        1e-9
    };

    // Weld by a grid whose cells are the tolerance across. A grid alone would
    // split two points that straddle a cell boundary, so each lookup also tries
    // the 26 neighbouring cells — which is what makes the tolerance mean
    // "within", rather than "in the same bucket".
    let cell = |p: &[f64; 3]| -> [i64; 3] {
        [
            (p[0] / tol).floor() as i64,
            (p[1] / tol).floor() as i64,
            (p[2] / tol).floor() as i64,
        ]
    };
    let mut lookup: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let mut points: Vec<[f64; 3]> = Vec::new();
    let tol_sq = tol * tol;

    let mut ids: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for t in tris {
        let mut tri_ids = [0u32; 3];
        for (k, v) in t.iter().enumerate() {
            let c = cell(v);
            let mut found = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = [c[0] + dx, c[1] + dy, c[2] + dz];
                        let Some(bucket) = lookup.get(&key) else {
                            continue;
                        };
                        for &id in bucket {
                            let q = &points[id as usize];
                            let d = (q[0] - v[0]).powi(2)
                                + (q[1] - v[1]).powi(2)
                                + (q[2] - v[2]).powi(2);
                            if d <= tol_sq {
                                found = Some(id);
                                break 'search;
                            }
                        }
                    }
                }
            }
            tri_ids[k] = match found {
                Some(id) => id,
                None => {
                    let id = points.len() as u32;
                    points.push(*v);
                    lookup.entry(c).or_default().push(id);
                    id
                }
            };
        }
        ids.push(tri_ids);
    }

    // Union-find over the welded vertices.
    let mut parent: Vec<u32> = (0..points.len() as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for t in &ids {
        let a = find(&mut parent, t[0]);
        for &v in &t[1..] {
            let b = find(&mut parent, v);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }

    // Group triangles by root, keeping the order they were first seen so the
    // same soup always splits the same way.
    let mut order: HashMap<u32, usize> = HashMap::new();
    let mut out: Vec<Vec<Tri>> = Vec::new();
    for (i, t) in tris.iter().enumerate() {
        let root = find(&mut parent, ids[i][0]);
        let slot = *order.entry(root).or_insert_with(|| {
            out.push(Vec::new());
            out.len() - 1
        });
        out[slot].push(*t);
    }
    out
}

// ---------------------------------------------------------------------------
// Containment
// ---------------------------------------------------------------------------

/// Whether a point is inside a closed body.
///
/// By solid angle rather than by casting a ray: a ray has to choose a direction,
/// and every direction is the wrong one for some mesh — through a shared edge,
/// along a coplanar face, out through a crack. Summing the solid angle each
/// triangle subtends has no direction to choose and no special cases, and it
/// degrades gracefully on a mesh that is not quite closed instead of flipping
/// its answer.
///
/// Needs consistent winding to know inside from outside. On a mesh with reversed
/// faces the sum still converges, to the wrong sign.
///
/// A point exactly *on* the surface is not inside. See [`crate::assembly::mesh::winding`] for what such
/// a point actually returns and why it is not a special case.
pub fn inside(tris: &[Tri], p: [f64; 3]) -> bool {
    winding(tris, p).abs() > 0.5
}

/// How many times a closed body wraps around a point. One inside, zero outside.
///
/// The number [`crate::assembly::mesh::inside`] thresholds. Worth reading directly when a mesh might
/// not be closed: a value near neither 0 nor 1 means the body has a hole in it,
/// and the containment question does not have an answer.
///
/// # On the surface
///
/// A point lying on the surface returns the fraction of a full turn the body
/// subtends there: a half on a face, a quarter on a right-angled edge, an eighth
/// at a cube's corner. None of those is greater than a half, so [`crate::assembly::mesh::inside`] says
/// no — which is the answer that makes two parts resting against each other read
/// as touching rather than as one containing the other.
///
/// This needs no special case, and must not have one. A triangle seen from a
/// point in its own plane subtends nothing, and the formula below returns
/// exactly zero for it; a guard that noticed the degeneracy and substituted
/// "inside" would fire precisely where two parts share a vertex, which is
/// precisely where parts touch.
pub fn winding(tris: &[Tri], p: [f64; 3]) -> f64 {
    let mut total = 0.0;
    for t in tris {
        let a = sub(t[0], p);
        let b = sub(t[1], p);
        let c = sub(t[2], p);
        let (la, lb, lc) = (len(a), len(b), len(c));
        let numer = dot(a, cross(b, c));
        let denom = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
        // `atan2(0, 0)` is zero, which is the right contribution from a triangle
        // the point is coplanar with — including one it is a vertex of.
        total += 2.0 * numer.atan2(denom);
    }
    total / (4.0 * std::f64::consts::PI)
}

/// How many faces to probe from before giving up. See `shares_volume`.
///
/// A ceiling on pathology rather than a sampling rate: the candidates are
/// already filtered to faces that reach the other body, and a face that does not
/// reach it cannot be inside it.
const PROBES: usize = 256;

/// Whether any of `probe`'s own interior lies inside `body`.
///
/// The test that catches what a surface-crossing test cannot. Two solids can
/// share volume and cross nothing: give two boxes the same height and depth and
/// overlap them along their length, and every place their surfaces meet is a
/// face laid on a face or an edge laid on an edge. Nothing passes *through*
/// anything, and yet half of one is inside the other. The same happens whenever
/// a body is swallowed whole.
///
/// So the probe is a point just *inside* `probe`'s surface — a face centroid
/// stepped along the inward normal — rather than a point on it. A point on the
/// surface answers neither question, and a vertex is the worst choice of all,
/// because a shared vertex is what touching parts have.
///
/// The step is a millionth of the face's own size, so it stays inside the body
/// however the model is scaled, and it clears a coincident face of the other
/// body by far more than the rounding that put them in the same plane.
///
/// # Where the probes go
///
/// Only at faces whose bounds reach `body`. That filter is what makes a fixed
/// number of probes enough: a face that cannot reach the other body cannot be
/// inside it, so spreading probes evenly over the whole surface spends nearly
/// all of them where the answer is already known. On a large part touching
/// another along a small patch, the candidates *are* the patch.
///
/// Falls back to [`PROBES`] evenly spread when more faces than that qualify,
/// which needs the two bodies to be interleaved over a wide region — and an
/// overlap that wide is not boundary-aligned, so [`crate::assembly::mesh::interfere`]'s crossing test
/// has already answered it.
///
/// # Cost
///
/// Each probe is a winding sum over all of `body`, so this is
/// `probes × body.len()`. It runs only when the crossing test found nothing,
/// which for two parts that are nowhere near each other is never — their bounds
/// do not overlap and there are no candidates at all.
fn shares_volume(body: &[Tri], probe: &[Tri], probes: usize) -> bool {
    if body.is_empty() || probe.is_empty() {
        return false;
    }
    let reach = super::aabb(body);
    let candidates: Vec<&Tri> = probe
        .iter()
        .filter(|t| bb_dist_sq(&tri_bb(t), &reach) <= 0.0)
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let step = candidates.len().div_ceil(probes.max(1)).max(1);
    for t in candidates.into_iter().step_by(step).take(probes.max(1)) {
        let Some(p) = just_inside(t) else { continue };
        if winding(body, p).abs() > 0.5 {
            return true;
        }
    }
    false
}

/// A point a hair inside the surface, behind the middle of this face.
fn just_inside(t: &Tri) -> Option<[f64; 3]> {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let l = len(n);
    if l <= 0.0 {
        return None; // degenerate face, no inward to speak of
    }
    // Scaled by the face, so the step is small against this triangle rather
    // than against some absolute idea of small.
    let inward = scale(n, -(l.sqrt() * 1e-6) / l);
    Some([
        (t[0][0] + t[1][0] + t[2][0]) / 3.0 + inward[0],
        (t[0][1] + t[1][1] + t[2][1]) / 3.0 + inward[1],
        (t[0][2] + t[1][2] + t[2][2]) / 3.0 + inward[2],
    ])
}

// ---------------------------------------------------------------------------
// Proximity
// ---------------------------------------------------------------------------

/// Closest approach between two bodies. Zero when they touch or overlap.
///
/// This is the measurement behind every cross-pose check: contact is a distance
/// below a tolerance, engagement is that distance staying below it, and coming
/// apart is it rising above.
///
/// ```
/// use threers::assembly::nearest;
/// let a = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
/// let b = vec![[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]];
/// assert!((nearest(&a, &b) - 2.0).abs() < 1e-9);
/// ```
pub fn nearest(a: &[Tri], b: &[Tri]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let abb = tri_bbs(a);
    let bbb = tri_bbs(b);
    let whole_b = super::aabb(b);

    // Visit A's triangles nearest-to-B first. The running bound then starts
    // tight, and every later box test rejects instead of descending.
    //
    // The key is computed once per triangle rather than inside the comparator,
    // which would evaluate it the `n log n` times a sort compares rather than
    // the `n` times there are triangles.
    let mut order: Vec<(f64, usize)> = abb
        .iter()
        .map(|bb| bb_dist_sq(bb, &whole_b))
        .zip(0..a.len())
        .collect();
    order.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut best = f64::INFINITY;
    for &(reach, i) in &order {
        if reach >= best * best {
            break; // sorted, so nothing after this one is closer either
        }
        for (j, bj) in bbb.iter().enumerate() {
            if bb_dist_sq(&abb[i], bj) >= best * best {
                continue;
            }
            let d = tri_tri_distance(&a[i], &b[j]);
            if d < best {
                best = d;
                if best == 0.0 {
                    return 0.0;
                }
            }
        }
    }
    best
}

/// Whether two bodies occupy the same space.
///
/// Touching is not interference: two faces resting on each other share a plane
/// and no volume, and this reports `false` for them. What it reports `true` for
/// is a surface of one crossing a surface of the other, or one body swallowed
/// whole by the other.
///
/// Tested twice over, because one test is not enough. A crossing catches the
/// ordinary case cheaply. Shared volume with *no* crossing is the case that
/// looks impossible and is not: two boxes of the same height and depth,
/// overlapped along their length, meet only face-on-face and edge-on-edge, and
/// half of one is inside the other. One body swallowed by another is the same
/// shape of problem. `shares_volume` is what sees those.
///
/// # What it cannot see
///
/// Two bodies overlapping in a region so thin that no face centroid lies inside
/// the other — a shaving thinner than the mesh is fine. [`crate::assembly::mesh::nearest`] returns zero
/// for those, which is the measurement that says the pair is in contact; this is
/// the one that says the contact has depth.
pub fn interfere(a: &[Tri], b: &[Tri]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let bbb = tri_bbs(b);
    let whole_b = super::aabb(b);
    let mut touched = false;

    for ai in a {
        let abi = tri_bb(ai);
        // Against the whole of `b` before against each of its triangles. On a
        // part with tens of thousands of faces most of them are nowhere near
        // the other body, and this rejects each in one test rather than in one
        // per triangle over there.
        if bb_dist_sq(&abi, &whole_b) > 0.0 {
            continue;
        }
        for (j, bj) in bbb.iter().enumerate() {
            if bb_dist_sq(&abi, bj) > 0.0 {
                continue;
            }
            if tri_tri_intersect(ai, &b[j]) {
                return true;
            }
            // Whether the two so much as touch, gathered on the way past. It
            // decides how hard the probe below has to look, and computing it
            // here costs one pass instead of a second one.
            if !touched && tri_tri_distance(ai, &b[j]) == 0.0 {
                touched = true;
            }
        }
    }

    // No crossing does not mean no shared volume — see `shares_volume`.
    //
    // Two bodies that do not touch anywhere are either disjoint or one is
    // wholly within the other, and a handful of probes settles which: with no
    // contact at all, a connected surface is entirely inside or entirely
    // outside, so there is nothing for more probes to discover. Only a pair
    // that touches can be partly inside without crossing, and only that pair
    // pays for the full sweep.
    let probes = if touched { PROBES } else { 8 };
    shares_volume(b, a, probes) || shares_volume(a, b, probes)
}

// ---------------------------------------------------------------------------
// Triangle kernels
// ---------------------------------------------------------------------------

/// Whether two triangles cross.
///
/// Six segment-triangle tests: if two non-coplanar triangles meet, an edge of
/// one passes through the other, so testing every edge against the opposite face
/// finds it. Coplanar overlap crosses nothing and is not reported — see
/// [`crate::assembly::mesh::interfere`].
///
/// **Crossing, not touching.** Two triangles meeting at a shared vertex, along a
/// shared edge, or with one's corner resting on the other's face all pass
/// through nothing and return `false`. [`tri_tri_distance`] returns zero for
/// them, which is the measurement that says they are in contact.
pub fn tri_tri_intersect(a: &Tri, b: &Tri) -> bool {
    for k in 0..3 {
        if seg_tri(a[k], a[(k + 1) % 3], b) || seg_tri(b[k], b[(k + 1) % 3], a) {
            return true;
        }
    }
    false
}

/// Möller–Trumbore, clamped to a segment and required to pass *through*.
///
/// Strictly interior on both counts: the crossing must be inside the triangle
/// rather than on its rim, and inside the segment rather than at an end. That is
/// the difference between two solids sharing volume and two solids resting
/// against each other — a face laid on a face crosses at every boundary and
/// passes through nothing, and reporting it as interference would flag every
/// assembled joint in the model.
///
/// The margin is on `u`, `v` and `t`, which are fractions, so it means the same
/// thing whatever the model is drawn in. The parallel test is scaled by the
/// triangle and the segment for the same reason: `det` has the units of a
/// volume, and comparing one to a fixed number is a statement about how big the
/// model is rather than about whether the segment is parallel.
fn seg_tri(p0: [f64; 3], p1: [f64; 3], t: &Tri) -> bool {
    /// A billionth of the way across a triangle. Anything closer to the rim
    /// than this is touching, and is not our business.
    const EDGE: f64 = 1e-9;
    let dir = sub(p1, p0);
    let e1 = sub(t[1], t[0]);
    let e2 = sub(t[2], t[0]);
    let h = cross(dir, e2);
    let det = dot(e1, h);
    if det.abs() <= 1e-12 * len(dir) * len(e1) * len(e2) {
        return false; // parallel to the face, including lying in it
    }
    let inv = 1.0 / det;
    let s = sub(p0, t[0]);
    let u = dot(s, h) * inv;
    if u <= EDGE || u >= 1.0 - EDGE {
        return false;
    }
    let q = cross(s, e1);
    let v = dot(dir, q) * inv;
    if v <= EDGE || u + v >= 1.0 - EDGE {
        return false;
    }
    let time = dot(e2, q) * inv;
    time > EDGE && time < 1.0 - EDGE
}

/// Closest approach between two triangles.
///
/// Zero when they cross; otherwise the minimum over the boundary, which for two
/// disjoint convex sets is where it always lies.
pub fn tri_tri_distance(a: &Tri, b: &Tri) -> f64 {
    if tri_tri_intersect(a, b) {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for v in a {
        best = best.min(point_tri_distance(*v, b));
    }
    for v in b {
        best = best.min(point_tri_distance(*v, a));
    }
    for i in 0..3 {
        for j in 0..3 {
            best = best.min(seg_seg_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]));
        }
    }
    best
}

/// Distance from a point to a triangle, by Voronoi region.
pub fn point_tri_distance(p: [f64; 3], t: &Tri) -> f64 {
    len(sub(p, closest_on_tri(p, t)))
}

/// Closest point on a triangle to `p` (Ericson, *Real-Time Collision Detection*).
fn closest_on_tri(p: [f64; 3], t: &Tri) -> [f64; 3] {
    let (a, b, c) = (t[0], t[1], t[2]);
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
        let v = if d1 - d3 != 0.0 { d1 / (d1 - d3) } else { 0.0 };
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 - d6 != 0.0 { d2 / (d2 - d6) } else { 0.0 };
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom != 0.0 { (d4 - d3) / denom } else { 0.0 };
        return add(b, scale(sub(c, b), w));
    }
    let denom = va + vb + vc;
    if denom == 0.0 {
        return a; // degenerate triangle
    }
    let v = vb / denom;
    let w = vc / denom;
    add(add(a, scale(ab, v)), scale(ac, w))
}

/// Distance between two segments, including the parallel case.
pub fn seg_seg_distance(p1: [f64; 3], q1: [f64; 3], p2: [f64; 3], q2: [f64; 3]) -> f64 {
    const EPS: f64 = 1e-15;
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    let (mut s, mut t);
    if a <= EPS && e <= EPS {
        return len(r); // both degenerate to points
    }
    if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            s = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0 // parallel: any s will do, then t is solved for it
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    len(sub(add(p1, scale(d1, s)), add(p2, scale(d2, t))))
}

// ---------------------------------------------------------------------------
// Vector helpers, kept local so this module depends on nothing
// ---------------------------------------------------------------------------

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
