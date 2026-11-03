//! Delaunay triangulation, convex hull and Voronoi cells, in the core crate.
//!
//! d3-delaunay and d3-polygon's hull, and the reason they are worth having: a
//! scattered set of points is the most common input in generative work, and
//! turning one into a surface, an outline or a set of regions is the step
//! between having data and having geometry.
//!
//! All three existed in this tree already and none was reachable. The exact
//! integer-predicate Delaunay in [`crate::rlx::geo`] sits behind the `rlx-geo`
//! feature and an external crate; the constrained triangulator in the CSG
//! kernel is specialised to one convex triangle; the 2D hull was private to the
//! OpenSCAD front end. So `cargo add threers` had three implementations of
//! computational geometry and access to none of them.
//!
//! These are the plain floating-point versions — good for scattered samples,
//! generative layouts and heightfields. Where the input is near-degenerate on
//! purpose (four points on a circle, three nearly collinear), the exact
//! predicates behind `rlx-geo` are the ones to reach for; the difference is
//! spelled out there.
//!
//! ```
//! use threers::math::{delaunay, convex_hull, Vector2};
//!
//! let points = [
//!     Vector2::new(0.0, 0.0),
//!     Vector2::new(1.0, 0.0),
//!     Vector2::new(1.0, 1.0),
//!     Vector2::new(0.0, 1.0),
//! ];
//! // A square is two triangles, whichever way the diagonal falls.
//! assert_eq!(delaunay(&points).len(), 2);
//! // And its hull is all four corners.
//! assert_eq!(convex_hull(&points).len(), 4);
//! ```

use crate::math::{Box2, Vector2};

/// Twice the signed area of `abc`; positive when counter-clockwise.
#[inline]
fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Positive when `d` lies inside the circle through `a`, `b`, `c` — assuming
/// `abc` is counter-clockwise.
#[inline]
fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
    let (bx, by) = (b[0] - d[0], b[1] - d[1]);
    let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
    (ax * ax + ay * ay) * (bx * cy - by * cx) - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
}

/// Delaunay triangulation of a point set, as triangles of indices into it.
///
/// Incremental insertion: start from one non-degenerate triangle, and for each
/// further point delete every triangle whose circumcircle contains it, then fill
/// the cavity by joining the point to each of the cavity's boundary edges. A
/// point landing outside the mesh extends it — the hull edges it can see are
/// part of the cavity too, which is what keeps the mesh covering the convex hull
/// of everything inserted so far.
///
/// There is deliberately no super-triangle. The usual trick — wrap the input in
/// one huge triangle and discard whatever still touches it — quietly loses thin
/// triangles along the hull. Those have enormous circumcircles (a chord of 3
/// with a sagitta of 0.004 gives a radius of 271), so a super-triangle vertex
/// at any finite distance eventually lands inside one, the sliver fails the
/// circumcircle test, and the result is a crack between the triangulation and
/// its own hull. Pushing the super-triangle further out trades that for a loss
/// of conditioning in the circumcircle predicate, where the fourth powers of
/// the coordinates swamp the term being tested. Starting from a real triangle
/// has neither problem and no magic constant.
///
/// O(n²): every insertion scans every triangle. That is the same order as
/// [`voronoi_cells`] and the right trade for the scattered-sample sizes this is
/// for; a quarter of a million points wants a sweep-hull implementation.
///
/// Fewer than three points, or points that are all collinear, give no
/// triangles rather than an error — both are legitimate inputs with no
/// triangulation, and there is nothing to report about them. Duplicates are
/// skipped.
///
/// Triangles come back counter-clockwise.
pub fn delaunay(points: &[Vector2]) -> Vec<[u32; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    let p: Vec<[f64; 2]> = points.iter().map(|v| [v.x as f64, v.y as f64]).collect();
    let n = p.len();

    // A seed triangle: the first point, the first one distinct from it, and the
    // first one off that line. Linear, rather than searching every triple,
    // because an all-collinear input would make that cubic.
    let Some(j) = (1..n).find(|&j| p[j] != p[0]) else {
        return Vec::new();
    };
    let Some(k) = (j + 1..n).find(|&k| orient(p[0], p[j], p[k]) != 0.0) else {
        return Vec::new();
    };
    let seed = if orient(p[0], p[j], p[k]) > 0.0 {
        [0, j, k]
    } else {
        [0, k, j]
    };

    let mut tris: Vec<[usize; 3]> = vec![seed];
    let mut bad: Vec<usize> = Vec::new();
    let mut cavity: Vec<(usize, usize)> = Vec::new();
    let mut directed: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut fresh: Vec<[usize; 3]> = Vec::new();

    for i in 0..n {
        if i == 0 || i == j || i == k {
            continue;
        }
        bad.clear();
        cavity.clear();
        fresh.clear();
        for (t, tri) in tris.iter().enumerate() {
            if in_circle(p[tri[0]], p[tri[1]], p[tri[2]], p[i]) > 0.0 {
                bad.push(t);
            }
        }

        // The cavity is the bad triangles plus, when the point lies outside the
        // mesh, the wedge between it and the hull edges it can see. Collect
        // both as directed edges and keep the ones that appear once: a boundary
        // edge swallowed by the cavity from both sides cancels out.
        for &t in &bad {
            let tri = tris[t];
            for e in 0..3 {
                cavity.push((tri[e], tri[(e + 1) % 3]));
            }
        }

        // A point inside a triangle is inside that triangle's circumcircle, so
        // if any triangle contains it, one of the bad ones does — no need to
        // scan the rest. And the mesh is convex at every step, so a point
        // inside it cannot see a boundary edge and the wedge pass has nothing
        // to add. That matters for speed and not just tidiness: the pass below
        // hashes every edge in the mesh, and most points in a scattered set
        // land inside, so skipping it here is the difference between hashing
        // O(n²) edges over the run and hashing almost none.
        let inside = bad.iter().any(|&t| {
            let tri = tris[t];
            orient(p[tri[0]], p[tri[1]], p[i]) >= 0.0
                && orient(p[tri[1]], p[tri[2]], p[i]) >= 0.0
                && orient(p[tri[2]], p[tri[0]], p[i]) >= 0.0
        });
        if !inside {
            // Every directed edge in the mesh. Triangles are counter-clockwise,
            // so an edge with no opposite twin is on the mesh boundary and has
            // the interior on its left.
            directed.clear();
            directed.reserve(tris.len() * 3);
            for tri in &tris {
                for e in 0..3 {
                    directed.insert((tri[e] as u32, tri[(e + 1) % 3] as u32));
                }
            }
            for tri in &tris {
                for e in 0..3 {
                    let (u, v) = (tri[e], tri[(e + 1) % 3]);
                    if directed.contains(&(v as u32, u as u32)) {
                        continue; // interior: another triangle owns the far side
                    }
                    if orient(p[u], p[v], p[i]) < 0.0 {
                        // Visible from outside, so the wedge behind it joins the
                        // cavity. Reversed, because that virtual triangle faces
                        // the other way.
                        cavity.push((v, u));
                    }
                }
            }
        }

        for a in 0..cavity.len() {
            let (u, v) = cavity[a];
            let shared = cavity
                .iter()
                .enumerate()
                .any(|(b, &(x, y))| b != a && ((x == u && y == v) || (x == v && y == u)));
            if !shared {
                fresh.push([u, v, i]);
            }
        }
        if fresh.is_empty() {
            // A duplicate, or a point exactly on the hull's line that no
            // circumcircle claims and no edge can see. Nothing to add — and
            // nothing to delete either, because a non-empty set of triangles
            // always has a non-empty boundary, so an empty cavity means the bad
            // set was empty too. Asserted rather than assumed: were it ever
            // false, falling through here would delete triangles and replace
            // them with nothing, which is the one way this loop could punch a
            // hole in the mesh.
            debug_assert!(bad.is_empty(), "cavity vanished around {} bad triangles", bad.len());
            continue;
        }

        for &t in bad.iter().rev() {
            tris.swap_remove(t);
        }
        tris.append(&mut fresh);
    }

    tris.into_iter()
        .map(|t| {
            // Counter-clockwise, always, so a caller can trust the winding.
            // Normally a no-op: a cavity edge is directed with the cavity on
            // its left and the new point is inside the cavity, so `[u, v, i]`
            // comes out counter-clockwise by construction — which is also why
            // `in_circle`'s precondition holds throughout the loop above. This
            // is here to make the guarantee unconditional rather than
            // conditional on that reasoning, and it costs one cross product
            // per triangle, once.
            if orient(p[t[0]], p[t[1]], p[t[2]]) < 0.0 {
                [t[0] as u32, t[2] as u32, t[1] as u32]
            } else {
                [t[0] as u32, t[1] as u32, t[2] as u32]
            }
        })
        .collect()
}

/// Convex hull of a 2D point set, as indices in counter-clockwise order.
///
/// Andrew's monotone chain. Collinear points on an edge are dropped — the hull
/// is its corners, not every point that happens to lie on it.
pub fn convex_hull(points: &[Vector2]) -> Vec<u32> {
    if points.len() < 3 {
        return (0..points.len() as u32).collect();
    }
    let mut order: Vec<u32> = (0..points.len() as u32).collect();
    order.sort_by(|&a, &b| {
        let (pa, pb) = (points[a as usize], points[b as usize]);
        pa.x
            .partial_cmp(&pb.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(pa.y.partial_cmp(&pb.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    order.dedup_by(|a, b| points[*a as usize] == points[*b as usize]);
    if order.len() < 3 {
        return order;
    }

    let cross = |o: u32, a: u32, b: u32| {
        let (o, a, b) = (
            points[o as usize],
            points[a as usize],
            points[b as usize],
        );
        orient(
            [o.x as f64, o.y as f64],
            [a.x as f64, a.y as f64],
            [b.x as f64, b.y as f64],
        )
    };

    let mut hull: Vec<u32> = Vec::with_capacity(order.len() + 1);
    for &i in &order {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    let lower = hull.len() + 1;
    for &i in order.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    hull.pop();
    hull
}

/// Voronoi cells for `sites`, each clipped to `bounds`, as a polygon per site
/// in input order.
///
/// Built by half-plane clipping — start from the bounding rectangle and cut it
/// back by the perpendicular bisector against other sites — rather than by
/// walking the Delaunay dual and closing the open cells afterwards. Clipping
/// closes them for free, because it never opens one: a cell on the hull is
/// unbounded in theory and bounded by the rectangle in practice, which is what
/// a caller drawing it wants anyway.
///
/// *Which* other sites comes from the dual, though. A cell is bounded only by
/// the bisectors it shares with its Delaunay neighbours — that is what the two
/// structures being duals means — and a site has about six of those however
/// many points there are. Clipping against all of them instead is the obvious
/// way to write this and it is the reason the first version took 379 ms for two
/// thousand sites where the triangulation it skipped takes 38 ms.
///
/// The all-pairs form is still here, as the fallback for input the
/// triangulation cannot describe: fewer than three sites, sites that are all
/// collinear, and duplicates, which are skipped during triangulation and so
/// have no neighbours to be bounded by. [`voronoi_cells`] is the same function
/// either way, and a test holds the two paths to the same answer.
///
/// A site outside `bounds`, or one coincident with another, yields an empty
/// polygon rather than a degenerate one.
pub fn voronoi_cells(sites: &[Vector2], bounds: Box2) -> Vec<Vec<Vector2>> {
    match delaunay_neighbours(sites) {
        Some(adjacency) => voronoi_from_adjacency(sites, bounds, &adjacency),
        None => voronoi_all_pairs(sites, bounds),
    }
}

/// Who shares a Delaunay edge with whom, or `None` when the triangulation does
/// not account for every site and so cannot answer the question.
///
/// One check covers every case that has to fall back, which is why there are no
/// others: a site in no triangle has no neighbours, and that is equally true of
/// a duplicate (skipped during insertion), of any site at all when the input is
/// collinear or shorter than three points (no triangles exist), and of a point
/// the predicates could not place. An empty neighbour list is the single
/// symptom, so it is the single test.
fn delaunay_neighbours(sites: &[Vector2]) -> Option<Vec<Vec<u32>>> {
    let tris = delaunay(sites);
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); sites.len()];
    for t in &tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if !adjacency[a as usize].contains(&b) {
                adjacency[a as usize].push(b);
            }
            if !adjacency[b as usize].contains(&a) {
                adjacency[b as usize].push(a);
            }
        }
    }
    // Off for the whole input rather than wrong for one site of it: a cell is
    // the intersection of half-planes with its neighbours, and a site without
    // any would come back as the entire bounding rectangle.
    if adjacency.iter().any(|a| a.is_empty()) {
        return None;
    }
    Some(adjacency)
}

/// The rectangle, counter-clockwise.
fn bounds_polygon(bounds: Box2) -> [Vector2; 4] {
    [
        Vector2::new(bounds.min.x, bounds.min.y),
        Vector2::new(bounds.max.x, bounds.min.y),
        Vector2::new(bounds.max.x, bounds.max.y),
        Vector2::new(bounds.min.x, bounds.max.y),
    ]
}

/// Clip `cell` by the bisector between `s` and `o`, keeping the half nearer
/// `s`. `false` means the two sites are coincident and neither has a cell.
fn clip_by_bisector(
    cell: &mut Vec<Vector2>,
    scratch: &mut Vec<Vector2>,
    s: Vector2,
    o: Vector2,
) -> bool {
    let d = Vector2::new(o.x - s.x, o.y - s.y);
    if d.x * d.x + d.y * d.y < 1e-20 {
        // Coincident sites have no bisector between them, and the cell of
        // either is undefined. Empty says so.
        cell.clear();
        return false;
    }
    let mid = Vector2::new((s.x + o.x) * 0.5, (s.y + o.y) * 0.5);
    clip_half_plane(cell, mid, d, scratch);
    std::mem::swap(cell, scratch);
    true
}

fn voronoi_from_adjacency(
    sites: &[Vector2],
    bounds: Box2,
    adjacency: &[Vec<u32>],
) -> Vec<Vec<Vector2>> {
    let rect = bounds_polygon(bounds);
    // Two buffers, swapped rather than reallocated: a cell is clipped half a
    // dozen times and every clip would otherwise be its own allocation.
    let mut cell: Vec<Vector2> = Vec::with_capacity(16);
    let mut scratch: Vec<Vector2> = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(sites.len());
    for (i, &s) in sites.iter().enumerate() {
        cell.clear();
        cell.extend_from_slice(&rect);
        for &j in &adjacency[i] {
            if cell.is_empty() {
                break;
            }
            if !clip_by_bisector(&mut cell, &mut scratch, s, sites[j as usize]) {
                break;
            }
        }
        out.push(cell.clone());
    }
    out
}

/// Every site against every other. The fallback, and the reference the fast
/// path is tested against.
fn voronoi_all_pairs(sites: &[Vector2], bounds: Box2) -> Vec<Vec<Vector2>> {
    let rect = bounds_polygon(bounds);
    let mut cell: Vec<Vector2> = Vec::with_capacity(16);
    let mut scratch: Vec<Vector2> = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(sites.len());
    for (i, &s) in sites.iter().enumerate() {
        cell.clear();
        cell.extend_from_slice(&rect);
        for (j, &o) in sites.iter().enumerate() {
            if i == j || cell.is_empty() {
                continue;
            }
            if !clip_by_bisector(&mut cell, &mut scratch, s, o) {
                break;
            }
        }
        out.push(cell.clone());
    }
    out
}

/// Sutherland–Hodgman against one half-plane: keep what is on the negative side
/// of the line through `point` with normal `normal`, writing the result to
/// `out` rather than returning it so a chain of clips allocates nothing.
fn clip_half_plane(poly: &[Vector2], point: Vector2, normal: Vector2, out: &mut Vec<Vector2>) {
    out.clear();
    if poly.is_empty() {
        return;
    }
    let side = |p: Vector2| (p.x - point.x) * normal.x + (p.y - point.y) * normal.y;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let (sa, sb) = (side(a), side(b));
        if sa <= 0.0 {
            out.push(a);
        }
        if (sa <= 0.0) != (sb <= 0.0) {
            let t = sa / (sa - sb);
            out.push(Vector2::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t));
        }
    }
}

/// Triangles filling a polygon with holes, by ear clipping.
///
/// The polygon outline `outer` and every hole are rings of 2D points, in
/// either winding; a ring may repeat its first point at the end, as GeoJSON
/// rings do, and that closing point is dropped. Indices count the points of
/// `outer` and then of each hole in turn, *after* dropping it — the order
/// [`polygon_points`] returns them in.
///
/// Unlike [`delaunay`], which triangulates the convex hull of scattered
/// points, this keeps to the outline: a C-shaped country stays C-shaped.
///
/// ```
/// use threers::math::{polygon_points, triangulate_polygon};
/// let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
/// let hole = vec![[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]];
/// let tris = triangulate_polygon(&square, &[hole.clone()]);
/// assert_eq!(polygon_points(&square, &[hole]).len(), 8);
/// assert_eq!(tris.len(), 8);
/// ```
pub fn triangulate_polygon(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[u32; 3]> {
    use crate::curves::shapes::{shape_utils, Vec2d};
    let ring = |r: &[[f64; 2]]| -> Vec<Vec2d> { open_ring(r).iter().map(|p| Vec2d::new(p[0], p[1])).collect() };
    let mut contour = ring(outer);
    let mut hs: Vec<Vec<Vec2d>> = holes.iter().map(|h| ring(h)).collect();
    if contour.len() < 3 {
        return Vec::new();
    }
    shape_utils::triangulate_shape(&mut contour, &mut hs)
}

/// The points [`triangulate_polygon`]'s indices refer to: `outer` then each
/// hole, closing points dropped.
pub fn polygon_points(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[f64; 2]> {
    let mut out = open_ring(outer).to_vec();
    for h in holes {
        out.extend_from_slice(open_ring(h));
    }
    out
}

/// `ring` without a closing point that repeats its first.
pub(crate) fn open_ring(ring: &[[f64; 2]]) -> &[[f64; 2]] {
    match ring {
        [first, .., last] if ring.len() > 2 && first == last => &ring[..ring.len() - 1],
        _ => ring,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A jittered grid. Jittered because an exact grid is four-points-on-a-
    /// circle everywhere, which is the degenerate case, and testing only that
    /// would hide every ordinary failure.
    ///
    /// The jitter comes from `Rng` and not from arithmetic on the indices. The
    /// first version used `((x * 7 + y * 13) % 5) * 0.013`, which is linear in
    /// x along the bottom row and linear in y up the right column, so both came
    /// out exactly collinear — and a point set whose hull edges have three
    /// points on them is not the general case the counts below assume.
    fn grid(n: usize) -> Vec<Vector2> {
        seeded_grid(n, 0x5EED)
    }

    fn seeded_grid(n: usize, seed: u64) -> Vec<Vector2> {
        let mut rng = crate::math::Rng::new(seed);
        let mut out = Vec::new();
        for y in 0..n {
            for x in 0..n {
                out.push(Vector2::new(
                    x as f32 + rng.range(-0.18, 0.18),
                    y as f32 + rng.range(-0.18, 0.18),
                ));
            }
        }
        out
    }

    /// Unsigned area of a polygon.
    fn shoelace(poly: &[Vector2]) -> f64 {
        let mut two = 0.0f64;
        for w in 0..poly.len() {
            let a = poly[w];
            let b = poly[(w + 1) % poly.len()];
            two += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
        }
        two.abs() * 0.5
    }

    fn area_of(p: &[Vector2], tris: &[[u32; 3]]) -> f64 {
        let c = |i: u32| [p[i as usize].x as f64, p[i as usize].y as f64];
        tris.iter()
            .map(|t| orient(c(t[0]), c(t[1]), c(t[2])).abs() * 0.5)
            .sum()
    }

    fn hull_area_of(p: &[Vector2], hull: &[u32]) -> f64 {
        let c = |i: u32| [p[i as usize].x as f64, p[i as usize].y as f64];
        let mut two = 0.0f64;
        for w in 0..hull.len() {
            let (a, b) = (c(hull[w]), c(hull[(w + 1) % hull.len()]));
            two += a[0] * b[1] - b[0] * a[1];
        }
        two.abs() * 0.5
    }

    /// A triangulation covers its own convex hull, exactly. This is the check
    /// that matters most and the one that found the bug this module was
    /// rewritten for: the super-triangle version left thin triangles off the
    /// hull, because a sliver with a 0.004 sagitta on a chord of 3 has a
    /// circumradius of 271 and the super-triangle vertex sat at 142, inside it.
    /// The mesh stayed consistent — closed boundary, no edge used three times,
    /// no point dropped — so only comparing areas showed the crack.
    #[test]
    fn the_triangulation_covers_its_own_hull() {
        for n in [3usize, 4, 5, 8, 12] {
            for seed in [0x5EED, 1, 2, 0xABCDEF] {
                let p = seeded_grid(n, seed);
                let tris = delaunay(&p);
                let hull = convex_hull(&p);
                let (got, want) = (area_of(&p, &tris), hull_area_of(&p, &hull));
                assert!(
                    (got - want).abs() <= want * 1e-9,
                    "{n}x{n} seed {seed:#x}: triangles cover {got:.9} of a {want:.9} hull \
                     (deficit {:.3e})",
                    want - got
                );
            }
        }
    }

    /// Points on a circle are the degenerate case for every circumcircle test
    /// at once: each triangle's circumcircle passes through every other point.
    /// Coverage still has to be exact.
    #[test]
    fn cocircular_points_still_tile_their_hull() {
        for count in [3usize, 4, 7, 32] {
            let p: Vec<Vector2> = (0..count)
                .map(|i| {
                    let a = i as f32 / count as f32 * std::f32::consts::TAU;
                    Vector2::new(a.cos() * 3.0, a.sin() * 3.0)
                })
                .collect();
            let tris = delaunay(&p);
            let hull = convex_hull(&p);
            assert_eq!(hull.len(), count, "{count} on a circle are all hull points");
            let (got, want) = (area_of(&p, &tris), hull_area_of(&p, &hull));
            assert!(
                (got - want).abs() <= want * 1e-6,
                "{count} on a circle: covered {got:.9} of {want:.9}"
            );
            assert_eq!(tris.len(), count - 2, "a convex polygon fans into n-2");
        }
    }

    /// A deliberately thin point set, which is where the hull slivers live.
    #[test]
    fn a_nearly_degenerate_strip_is_covered() {
        let mut rng = crate::math::Rng::new(7);
        let p: Vec<Vector2> = (0..40)
            .map(|i| Vector2::new(i as f32 * 0.25, rng.range(-0.002, 0.002)))
            .collect();
        let tris = delaunay(&p);
        let hull = convex_hull(&p);
        let (got, want) = (area_of(&p, &tris), hull_area_of(&p, &hull));
        assert!(
            (got - want).abs() <= want * 1e-6,
            "strip: covered {got:.9} of {want:.9}"
        );
    }

    #[test]
    fn a_square_is_two_triangles() {
        let p = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        assert_eq!(delaunay(&p).len(), 2);
    }

    /// Euler's formula pins the triangle count exactly: for `n` points with `h`
    /// on the hull, a triangulation has `2n - h - 2` triangles. A triangulator
    /// that drops a triangle or leaves a hole fails this and nothing else
    /// catches it.
    #[test]
    fn the_triangle_count_matches_eulers_formula() {
        for n in [3usize, 4, 5, 8, 12] {
            for seed in [0x5EED, 1, 2, 0xABCDEF] {
                let p = seeded_grid(n, seed);
                let tris = delaunay(&p);
                let hull = convex_hull(&p);
                let expect = 2 * p.len() - hull.len() - 2;
                assert_eq!(
                    tris.len(),
                    expect,
                    "{}x{} grid seed {:#x}: {} triangles for {} points and {} on the hull",
                    n,
                    n,
                    seed,
                    tris.len(),
                    p.len(),
                    hull.len()
                );
            }
        }
    }

    /// The defining property: no point may lie inside any triangle's
    /// circumcircle. This is what separates a Delaunay triangulation from any
    /// other triangulation of the same points, and a flip-based implementation
    /// that stops one flip early still looks plausible without it.
    #[test]
    fn no_point_lies_inside_a_circumcircle() {
        let p = grid(9);
        let tris = delaunay(&p);
        assert!(!tris.is_empty());
        let c: Vec<[f64; 2]> = p.iter().map(|v| [v.x as f64, v.y as f64]).collect();
        for t in &tris {
            for (i, d) in c.iter().enumerate() {
                if t.contains(&(i as u32)) {
                    continue;
                }
                let v = in_circle(
                    c[t[0] as usize],
                    c[t[1] as usize],
                    c[t[2] as usize],
                    *d,
                );
                assert!(
                    v <= 1e-6,
                    "point {i} is inside the circumcircle of {t:?} by {v:.3e}"
                );
            }
        }
    }

    #[test]
    fn every_triangle_comes_back_counter_clockwise() {
        let p = grid(7);
        let c: Vec<[f64; 2]> = p.iter().map(|v| [v.x as f64, v.y as f64]).collect();
        for t in delaunay(&p) {
            let a = orient(c[t[0] as usize], c[t[1] as usize], c[t[2] as usize]);
            assert!(a > 0.0, "{t:?} is wound the other way (area {a:.3e})");
        }
    }

    #[test]
    fn degenerate_input_gives_no_triangles_rather_than_nonsense() {
        assert!(delaunay(&[]).is_empty());
        assert!(delaunay(&[Vector2::new(0.0, 0.0)]).is_empty());
        // All collinear: there is no triangulation, and saying so beats
        // inventing slivers.
        let line: Vec<Vector2> = (0..8).map(|i| Vector2::new(i as f32, 0.0)).collect();
        assert!(delaunay(&line).is_empty());
        // Duplicates are skipped, not inserted as zero-area triangles.
        let dupes = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(0.0, 0.0),
        ];
        assert_eq!(delaunay(&dupes).len(), 1);
    }

    #[test]
    fn the_hull_is_its_corners_and_not_every_point_on_it() {
        // A square with a point in the middle of each edge and one inside.
        let p = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(2.0, 2.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(1.0, 0.0), // on an edge
            Vector2::new(1.0, 1.0), // interior
        ];
        let hull = convex_hull(&p);
        assert_eq!(hull.len(), 4, "got {hull:?}");
        for i in [0u32, 1, 2, 3] {
            assert!(hull.contains(&i), "corner {i} missing from {hull:?}");
        }
    }

    #[test]
    fn the_hull_is_counter_clockwise_and_contains_every_point() {
        for n in [3usize, 4, 6, 9] {
            for seed in [0x5EED, 1, 2, 0xABCDEF] {
                check_hull(&seeded_grid(n, seed));
            }
        }
    }

    fn check_hull(p: &[Vector2]) {
        let hull = convex_hull(p);
        let c = |i: u32| {
            let v = p[i as usize];
            [v.x as f64, v.y as f64]
        };
        // Convex and counter-clockwise: every consecutive triple turns left.
        for w in 0..hull.len() {
            let a = orient(
                c(hull[w]),
                c(hull[(w + 1) % hull.len()]),
                c(hull[(w + 2) % hull.len()]),
            );
            assert!(a > -1e-9, "hull turns right at {w} ({a:.3e})");
        }
        // And nothing is outside it.
        for (i, q) in p.iter().enumerate() {
            for w in 0..hull.len() {
                let a = orient(c(hull[w]), c(hull[(w + 1) % hull.len()]), [
                    q.x as f64,
                    q.y as f64,
                ]);
                assert!(a > -1e-6, "point {i} is outside hull edge {w}");
            }
        }
    }

    /// The property that makes a Voronoi diagram one: every point of a cell is
    /// nearer to its own site than to any other.
    #[test]
    fn every_cell_point_is_nearest_to_its_own_site() {
        let sites = grid(5);
        let bounds = Box2::new(Vector2::new(-1.0, -1.0), Vector2::new(5.0, 5.0));
        let cells = voronoi_cells(&sites, bounds);
        assert_eq!(cells.len(), sites.len());

        let d2 = |a: Vector2, b: Vector2| (a.x - b.x).powi(2) + (a.y - b.y).powi(2);
        for (i, cell) in cells.iter().enumerate() {
            assert!(cell.len() >= 3, "cell {i} is degenerate: {cell:?}");
            // Test the centroid rather than the corners: a corner is
            // equidistant from several sites by construction, so it proves
            // nothing, while the interior is unambiguous.
            let n = cell.len() as f32;
            let centroid = Vector2::new(
                cell.iter().map(|p| p.x).sum::<f32>() / n,
                cell.iter().map(|p| p.y).sum::<f32>() / n,
            );
            let own = d2(centroid, sites[i]);
            for (j, s) in sites.iter().enumerate() {
                if i != j {
                    assert!(
                        own <= d2(centroid, *s) + 1e-4,
                        "cell {i}'s centroid is nearer site {j}"
                    );
                }
            }
        }
    }

    /// The cells of sites inside the rectangle partition that rectangle: no
    /// gaps, no overlaps, nothing spilling out. Checked by area, which is the
    /// one measurement that catches all three at once — a bisector clipped on
    /// the wrong side leaves a gap and an overlap of the same shape, so
    /// inspecting any single cell would not show it.
    #[test]
    fn the_cells_tile_the_bounds_exactly() {
        let bounds = Box2::new(Vector2::new(-0.5, -0.5), Vector2::new(6.5, 6.5));
        let want = 7.0f64 * 7.0;
        for n in [2usize, 3, 5, 7] {
            let sites = grid(n);
            let cells = voronoi_cells(&sites, bounds);
            let mut total = 0.0f64;
            for cell in &cells {
                assert!(cell.len() >= 3, "a site inside the bounds has a cell");
                total += shoelace(cell);
            }
            assert!(
                (total - want).abs() < want * 1e-4,
                "{n}x{n} sites: cells cover {total:.6} of a {want:.6} rectangle"
            );
        }
    }

    /// The shortcut and the thing it is a shortcut for must agree. This is the
    /// test that licenses clipping against six neighbours instead of two
    /// thousand sites: the duality it relies on is a theorem, but the
    /// implementation of it is not, and a missing neighbour would leave a cell
    /// too *large* in a way that still tiles and still looks plausible.
    #[test]
    fn the_dual_shortcut_agrees_with_clipping_against_everything() {
        let bounds = Box2::new(Vector2::new(-1.0, -1.0), Vector2::new(13.0, 13.0));
        for n in [3usize, 5, 8, 12] {
            for seed in [0x5EED, 1, 2, 0xABCDEF] {
                let sites = seeded_grid(n, seed);
                let fast = voronoi_cells(&sites, bounds);
                let slow = voronoi_all_pairs(&sites, bounds);
                assert!(
                    delaunay_neighbours(&sites).is_some(),
                    "{n}x{n} seed {seed:#x} should take the fast path"
                );
                assert_eq!(fast.len(), slow.len());
                for (i, (f, s)) in fast.iter().zip(slow.iter()).enumerate() {
                    // Compared as polygons and not as lists. Both paths clip the
                    // same rectangle by the same half-planes but in a different
                    // order, so the ring starts at a different vertex — the
                    // shape is what has to match, not where the list begins.
                    let sorted = |c: &[Vector2]| {
                        let mut v: Vec<(f32, f32)> = c.iter().map(|p| (p.x, p.y)).collect();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v
                    };
                    let (fv, sv) = (sorted(f), sorted(s));
                    assert_eq!(
                        fv.len(),
                        sv.len(),
                        "cell {i} of {n}x{n} seed {seed:#x}: {f:?} vs {s:?}"
                    );
                    for (a, b) in fv.iter().zip(sv.iter()) {
                        assert!(
                            (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4,
                            "cell {i} of {n}x{n} seed {seed:#x}: {a:?} vs {b:?}\n  fast {f:?}\n  slow {s:?}"
                        );
                    }
                    assert!(
                        (shoelace(f) - shoelace(s)).abs() < 1e-4,
                        "cell {i} area {} vs {}",
                        shoelace(f),
                        shoelace(s)
                    );
                }
            }
        }
    }

    /// And the input that has to fall back actually does, because the fast path
    /// would be wrong rather than slow for it.
    #[test]
    fn degenerate_input_falls_back_to_all_pairs() {
        // All collinear: no triangulation at all, and the cells are strips.
        let line: Vec<Vector2> = (0..6).map(|i| Vector2::new(i as f32, 0.0)).collect();
        assert!(delaunay_neighbours(&line).is_none());
        let bounds = Box2::new(Vector2::new(-1.0, -2.0), Vector2::new(6.0, 2.0));
        let cells = voronoi_cells(&line, bounds);
        // Six strips tiling a 7x4 rectangle. Had the fast path been taken each
        // site would have got the whole rectangle, so the total would be 6x too
        // big — which is exactly the failure this guards.
        let total: f64 = cells.iter().map(|c| shoelace(c)).sum();
        assert!(
            (total - 28.0).abs() < 1e-3,
            "collinear sites: cells cover {total:.4} of 28.0"
        );

        // A duplicate is in no triangle, so it has no neighbours.
        let mut dupes = seeded_grid(3, 11);
        dupes.push(dupes[4]);
        assert!(delaunay_neighbours(&dupes).is_none());
        let cells = voronoi_cells(&dupes, Box2::new(Vector2::new(-1.0, -1.0), Vector2::new(3.0, 3.0)));
        assert!(
            cells[4].is_empty() && cells[9].is_empty(),
            "coincident sites have no cell: {:?} {:?}",
            cells[4],
            cells[9]
        );
    }

    #[test]
    fn cells_stay_inside_the_bounds_they_were_given() {
        let sites = grid(4);
        let bounds = Box2::new(Vector2::new(0.0, 0.0), Vector2::new(3.0, 3.0));
        for cell in voronoi_cells(&sites, bounds) {
            for p in cell {
                assert!(
                    p.x >= -1e-4 && p.x <= 3.0 + 1e-4 && p.y >= -1e-4 && p.y <= 3.0 + 1e-4,
                    "{p:?} escaped the bounds"
                );
            }
        }
    }

    #[test]
    fn a_single_site_owns_the_whole_rectangle() {
        let bounds = Box2::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0));
        let cells = voronoi_cells(&[Vector2::new(1.0, 1.0)], bounds);
        assert_eq!(cells[0].len(), 4);
        // Coincident sites have no bisector and no cell, rather than a sliver.
        let dup = voronoi_cells(
            &[Vector2::new(1.0, 1.0), Vector2::new(1.0, 1.0)],
            bounds,
        );
        assert!(dup[0].is_empty() && dup[1].is_empty());
    }
}
