//! Fillets and chamfers: an edge replaced by a face.
//!
//! The operation a mesh cannot do. Rounding an edge of a triangle soup means
//! guessing which triangles were the two faces and where the edge was; here the
//! edge is an [`Edge`], it names the two surfaces it joins, and the blend is the
//! surface tangent to both — a cylinder under a rolling ball, a torus around a
//! rim, a sphere in a corner — stated exactly rather than sampled.
//!
//! ```ignore
//! let body = Body::cuboid([4.0, 4.0, 4.0]);
//! let all: Vec<usize> = (0..body.edges().len()).collect();
//! let rounded = body.fillet(&all, 0.5, 1e-3)?;
//! let bevelled = body.chamfer(&[0, 2], 0.25, 1e-3)?;
//! ```
//!
//! # What it blends
//!
//! * **A straight edge between two planes**, convex or concave, whose ends are
//!   corners where exactly three planar faces meet. Any number of the three
//!   edges at a corner may be selected: one ends the blend on the third face,
//!   two mitre into each other, three meet in a point or, with
//!   [`ChamferCorner::Facet`](super::ChamferCorner), a triangular facet
//!   (chamfer), or a spherical patch (fillet).
//! * **A circular rim** where a plane caps a cylinder square to its axis: the
//!   end of a shaft, the mouth of a hole, the foot of a boss, the floor of a
//!   blind bore. A chamfer there is a cone and a fillet a torus.
//! * **A smooth ring round a flat face**, each of its edges between the face
//!   and a plane or a cylinder square to it, the sides tangent where they meet:
//!   the top of a rounded rectangle, a slot, the mouth or floor of a pocket with
//!   rounded corners. Selecting any of its edges blends the whole ring, as a
//!   blend that follows tangent edges does, in planes and cones (chamfer) or
//!   cylinders and tori (fillet) that meet each other tangentially. Where the
//!   ring has sharp corners too — a notch cut in from the side — the blend
//!   runs along the smooth stretch to the sharp corner either side and ends on
//!   the flat face there, using up an edge it reaches the end of exactly.
//!
//! # What it declines
//!
//! Everything else, by name — see [`BlendDeclined`]. The contract is the
//! boolean's: a result is a closed solid, or there is no result and a reason.
//! In particular an edge that ends on a curved face, or on an edge that is not
//! straight, is declined unless it is part of such a ring, and so is a blend
//! whose strip another edge of the face reaches into — a hole near the edge,
//! which the blend would have to be trimmed by.
//!
//! # Tolerance
//!
//! Curved boundaries are sampled here, once, to `tolerance`, and both faces on
//! a boundary share the samples. [`Body::refine_edges`] can still add to the
//! ones that cross — a fillet's end on a flat face, a mitre — but not to the
//! ones along which two surfaces *touch*: a torus meets its cap in a tangent
//! circle, and the alternating projection that refines an edge has no slope to
//! follow there. So blend at the tolerance the result will be tessellated at.

use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use crate::nurbs::construct::perpendicular;
use crate::nurbs::{v3, V3};

use super::body::{Body, Edge, Face, TrimLoop};
use super::Surface;

/// Why a fillet or chamfer was not attempted, or was abandoned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlendDeclined {
    /// A body that is not a closed solid has no inside to round towards.
    NotASolid,
    /// A radius, distance or tolerance that is not a positive finite number.
    BadSize,
    /// An edge index past the end of [`Body::edges`].
    NoSuchEdge { edge: usize },
    /// An edge that is neither a straight line between two planes nor a
    /// circular rim between a plane and a cylinder square to it.
    UnsupportedEdge { edge: usize },
    /// A selected edge ends at a vertex that is not the meeting of exactly
    /// three planar faces along three straight edges.
    UnsupportedVertex { vertex: usize },
    /// The selected edges meeting at a vertex are not all convex or all
    /// concave. One blend there removes material and the other adds it, and the
    /// patch that joins them is not a sphere.
    MixedConvexity { vertex: usize },
    /// Two blends reach the unselected edge at a vertex at different depths, so
    /// they do not mitre: one would have to be cut by the other's neighbour.
    UnequalSetback { vertex: usize },
    /// The blend is too large for this edge: it would consume the edge, or a
    /// neighbouring one, or the face beside it.
    TooLarge { edge: usize },
    /// A face the blend would have to re-trim carries a boundary that could not
    /// be restated.
    TrimmedFace { face: usize },
    /// The blend was built and is not a closed solid. Returned instead of the
    /// body, because a caller cannot tell a result with a hole in it is wrong.
    NotWatertight { defects: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chamfer,
    Fillet,
}

impl Body {
    /// Round the given edges with a rolling ball of `radius`.
    ///
    /// `edges` index [`Body::edges`]. The blend faces are exact — cylinders,
    /// tori and spheres tangent to the faces they join — and their boundaries
    /// are sampled to `tolerance`. Indices of edges that were not selected are
    /// unchanged in the result.
    pub fn fillet(
        &self,
        edges: &[usize],
        radius: f64,
        tolerance: f64,
    ) -> Result<Body, BlendDeclined> {
        blend(self, edges, radius, Kind::Fillet, tolerance)
    }

    /// Bevel the given edges, cutting each face back by `distance` from the
    /// edge.
    ///
    /// The setback is measured along each face, so the bevel on a right-angled
    /// edge is at 45° and `distance·√2` wide.
    pub fn chamfer(
        &self,
        edges: &[usize],
        distance: f64,
        tolerance: f64,
    ) -> Result<Body, BlendDeclined> {
        blend(self, edges, distance, Kind::Chamfer, tolerance)
    }
}

/// A body taken apart, so it can be edited and put back.
struct Parts {
    surfaces: Vec<Surface>,
    vertices: Vec<V3>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

impl Parts {
    fn of(body: &Body) -> Parts {
        Parts {
            surfaces: body.surfaces().to_vec(),
            vertices: body.vertices().to_vec(),
            edges: body.edges().to_vec(),
            faces: body.faces().to_vec(),
        }
    }

    fn body(&self) -> Body {
        Body::from_parts(
            self.surfaces.clone(),
            self.vertices.clone(),
            self.edges.clone(),
            self.faces.clone(),
        )
    }

    fn vertex(&mut self, p: V3) -> usize {
        self.vertices.push(p);
        self.vertices.len() - 1
    }

    /// Drop the vertices nothing names any more — the corners a blend replaced.
    fn compact(&mut self) {
        let mut used = vec![false; self.vertices.len()];
        for e in &self.edges {
            for &v in &e.vertices {
                used[v] = true;
            }
        }
        for l in self.faces.iter().filter_map(|f| f.loops.as_ref()).flatten() {
            for &v in &l.vertices {
                used[v] = true;
            }
        }
        let mut renamed = vec![usize::MAX; self.vertices.len()];
        let mut kept = Vec::with_capacity(self.vertices.len());
        for (old, p) in self.vertices.iter().enumerate() {
            if used[old] {
                renamed[old] = kept.len();
                kept.push(*p);
            }
        }
        self.vertices = kept;
        for e in &mut self.edges {
            for v in &mut e.vertices {
                *v = renamed[*v];
            }
        }
        for l in self
            .faces
            .iter_mut()
            .filter_map(|f| f.loops.as_mut())
            .flatten()
        {
            for v in &mut l.vertices {
                *v = renamed[*v];
            }
        }
    }
}

fn blend(
    body: &Body,
    selection: &[usize],
    size: f64,
    kind: Kind,
    tolerance: f64,
) -> Result<Body, BlendDeclined> {
    if !(size > 0.0 && size.is_finite() && tolerance > 0.0 && tolerance.is_finite()) {
        return Err(BlendDeclined::BadSize);
    }
    let mut selection = selection.to_vec();
    selection.sort_unstable();
    selection.dedup();
    if let Some(&edge) = selection.iter().find(|&&e| e >= body.edges().len()) {
        return Err(BlendDeclined::NoSuchEdge { edge });
    }
    if !body.is_valid_solid(tolerance) {
        return Err(BlendDeclined::NotASolid);
    }
    if selection.is_empty() {
        return Ok(body.clone());
    }

    // Edges on smooth rings round a flat face blend round the whole ring.
    let eps = 1e-9 * extent(body.vertices());
    let mut rings: Vec<TangentLoop> = Vec::new();
    let mut on_rings: HashSet<usize> = HashSet::new();
    for &e in &selection {
        if on_rings.contains(&e) || body.edges()[e].closed {
            continue;
        }
        if let Some(ring) = tangent_loop(body, e, eps) {
            on_rings.extend(ring.edges.iter().map(|r| r.0));
            rings.push(ring);
        }
    }
    for ring in &rings {
        if let Some(r) = ring.risers.iter().flatten().find(|r| selection.contains(r)) {
            return Err(BlendDeclined::UnsupportedEdge { edge: *r });
        }
        for end in ring.ends.iter().flatten() {
            if selection.contains(&end.on_cap) || selection.contains(&end.on_side) {
                return Err(BlendDeclined::UnsupportedVertex { vertex: end.vertex });
            }
        }
    }
    selection.retain(|e| !on_rings.contains(e));

    let (rims, straight): (Vec<usize>, Vec<usize>) =
        selection.iter().partition(|&&e| body.edges()[e].closed);

    // The straight edges go together, because the ones that share a corner have
    // to agree about it. A rim has no corners and is its own problem; it is
    // done afterwards, on indices the first pass leaves where they were.
    let mut parts = if straight.is_empty() {
        Parts::of(body)
    } else {
        blend_straight(body, &straight, size, kind, tolerance)?
    };
    for &rim in &rims {
        blend_rim(&mut parts, rim, size, kind, tolerance)?;
    }
    for ring in &rings {
        blend_loop(&mut parts, ring, size, kind, tolerance)?;
    }
    parts.compact();

    let out = parts.body();
    let defects = out.defects(tolerance);
    if !defects.is_empty() || !out.shells().iter().all(|s| s.closed) {
        return Err(BlendDeclined::NotWatertight {
            defects: defects.len(),
        });
    }
    Ok(out)
}

fn extent(vertices: &[V3]) -> f64 {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if vertices.is_empty() {
        1.0
    } else {
        v3::dist(lo, hi).max(1e-12)
    }
}

fn pair(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// A plane's normal as the surface states it, and as the solid sees it.
fn plane_normals(body: &Body, face: usize) -> Option<(V3, V3)> {
    let f = &body.faces()[face];
    match &body.surfaces()[f.surface] {
        Surface::Plane { normal, .. } => Some((
            *normal,
            if f.flipped {
                v3::scale(*normal, -1.0)
            } else {
                *normal
            },
        )),
        _ => None,
    }
}

/// An open edge whose points all lie on the line between its ends: the ends,
/// the unit direction from the first to the last, and the length.
fn straight_edge(body: &Body, edge: usize, eps: f64) -> Option<(usize, usize, V3, f64)> {
    let e = &body.edges()[edge];
    if e.closed || e.vertices.len() < 2 {
        return None;
    }
    let (first, last) = (e.vertices[0], *e.vertices.last()?);
    let (p, q) = (body.vertices()[first], body.vertices()[last]);
    let length = v3::dist(p, q);
    let dir = v3::normalize(v3::sub(q, p))?;
    if first == last || length <= eps {
        return None;
    }
    for &v in &e.vertices[1..e.vertices.len() - 1] {
        let d = v3::sub(body.vertices()[v], p);
        let off = v3::sub(d, v3::scale(dir, v3::dot(d, dir)));
        if v3::norm(off) > eps {
            return None;
        }
    }
    Some((first, last, dir, length))
}

/// Which way a planar face lies from one of its edges.
///
/// Read off the face's boundary rather than off its other edges. The obvious
/// shortcut — the face lies towards its *next* edge at the corner — is wrong at
/// a reflex corner, where the next edge leads out of the face rather than
/// across it, and an L-shaped cap has one. A ring runs anticlockwise about the
/// surface normal with the face on its left, holes included, so the answer is
/// the normal crossed with the direction the ring walks the edge in.
struct Sides<'a> {
    body: &'a Body,
    loops: HashMap<usize, Vec<TrimLoop>>,
}

impl Sides<'_> {
    fn inward(&mut self, face: usize, edge: usize) -> Option<V3> {
        let body = self.body;
        let (normal, _) = plane_normals(body, face)?;
        let e = &body.edges()[edge];
        let (p, q) = (*e.vertices.first()?, *e.vertices.get(1)?);
        let loops = self
            .loops
            .entry(face)
            .or_insert_with(|| body.face_loops(face));
        let mut sign = 0.0;
        'rings: for l in loops.iter() {
            let n = l.vertices.len();
            for i in 0..n {
                let step = (l.vertices[i], l.vertices[(i + 1) % n]);
                if step == (p, q) {
                    sign = 1.0;
                    break 'rings;
                }
                if step == (q, p) {
                    sign = -1.0;
                    break 'rings;
                }
            }
        }
        if sign == 0.0 {
            return None;
        }
        let along = v3::sub(body.vertices()[q], body.vertices()[p]);
        v3::normalize(v3::scale(v3::cross(normal, along), sign))
    }
}

/// A selected straight edge and the face that replaces it.
struct Chosen {
    edge: usize,
    /// Its two ends, and the unit direction from the first to the second.
    ends: [usize; 2],
    dir: V3,
    /// The two faces it joins, and their outward normals.
    faces: [usize; 2],
    normals: [V3; 2],
    /// `+1` where the solid is convex across the edge, `-1` where it is
    /// concave. A convex blend removes material and a concave one adds it; the
    /// geometry is the same construction on the other side.
    convex: f64,
    /// How far each face is cut back from the edge.
    setback: f64,
    /// The blend surface, and the edges bounding the blend face.
    surface: usize,
    boundary: Vec<usize>,
}

/// A vertex a selected edge ends at: three edges, three faces.
struct Corner {
    vertex: usize,
    edges: [usize; 3],
    /// Unit directions along each edge, away from the vertex.
    dirs: [V3; 3],
    /// `faces[i]` is the face *opposite* `edges[i]` — the one bounded by the
    /// other two.
    faces: [usize; 3],
    selected: [bool; 3],
    /// Where an unselected edge now ends, having been cut back by its
    /// neighbours' blends.
    ends: [Option<usize>; 3],
    /// The point that replaces the vertex in `faces[i]`, for a face bounded by
    /// at least one selected edge.
    points: [Option<usize>; 3],
}

impl Corner {
    fn point_on(&self, face: usize) -> Option<usize> {
        (0..3)
            .find(|&i| self.faces[i] == face)
            .and_then(|i| self.points[i])
    }
}

/// Does the segment `p`–`q` touch the axis-aligned box `lo`..`hi`?
fn segment_meets_box(p: [f64; 2], q: [f64; 2], lo: [f64; 2], hi: [f64; 2]) -> bool {
    let (mut enter, mut leave) = (0.0f64, 1.0f64);
    for k in 0..2 {
        let d = q[k] - p[k];
        if d.abs() < 1e-300 {
            if p[k] < lo[k] || p[k] > hi[k] {
                return false;
            }
            continue;
        }
        let (a, b) = ((lo[k] - p[k]) / d, (hi[k] - p[k]) / d);
        enter = enter.max(a.min(b));
        leave = leave.min(a.max(b));
        if enter > leave {
            return false;
        }
    }
    true
}

fn solve3(rows: [V3; 3], rhs: V3) -> Option<V3> {
    let det = v3::dot(rows[0], v3::cross(rows[1], rows[2]));
    if det.abs() < 1e-12 {
        return None;
    }
    let x = v3::add(
        v3::add(
            v3::scale(v3::cross(rows[1], rows[2]), rhs[0]),
            v3::scale(v3::cross(rows[2], rows[0]), rhs[1]),
        ),
        v3::scale(v3::cross(rows[0], rows[1]), rhs[2]),
    );
    Some(v3::scale(x, 1.0 / det))
}

/// The curve where a fillet's cylinder is cut by a plane, between two points
/// already on it: the new vertices strictly between them.
///
/// Every end of a fillet is one of these. Cut by the face the edge ran into, it
/// is the arc that face gains; cut by the plane bisecting two fillets' axes, it
/// is the mitre they share — two cylinders of one radius whose axes cross meet
/// in a pair of ellipses, each of them planar; cut square to the axis, it is
/// the circle a corner's sphere touches the cylinder along.
///
/// Sampled by the cylinder's own angle and slid along the axis onto the plane,
/// so every point is on the cylinder exactly and on the plane exactly.
fn section(
    cylinder: &Surface,
    origin: V3,
    normal: V3,
    from: V3,
    to: V3,
    tolerance: f64,
    parts: &mut Parts,
) -> Option<Vec<usize>> {
    let Surface::Cylinder { axis, radius, .. } = cylinder else {
        return None;
    };
    let lean = v3::dot(*axis, normal);
    if lean.abs() < 1e-9 {
        return None;
    }
    let (ua, _) = cylinder.invert(from)?;
    let (ub, _) = cylinder.invert(to)?;
    let at = |u: f64| {
        let p = cylinder.point(u, 0.0);
        v3::add(
            p,
            v3::scale(*axis, v3::dot(v3::sub(origin, p), normal) / lean),
        )
    };
    // The two ends were worked out from the faces; the curve was worked out
    // from the surfaces. If they disagree the construction does not apply here.
    let slack = 1e-7 * (1.0 + radius + v3::dist(from, to));
    if v3::dist(at(ua), from) > slack || v3::dist(at(ub), to) > slack {
        return None;
    }
    let u = |k: f64, n: usize| ua + (ub - ua) * k / n as f64;
    let mut n = 2usize;
    while n < 2048 {
        let worst = (0..n)
            .map(|s| {
                let (a, b) = (at(u(s as f64, n)), at(u(s as f64 + 1.0, n)));
                v3::dist(at(u(s as f64 + 0.5, n)), v3::scale(v3::add(a, b), 0.5))
            })
            .fold(0.0, f64::max);
        if worst <= tolerance {
            break;
        }
        n *= 2;
    }
    Some((1..n).map(|s| parts.vertex(at(u(s as f64, n)))).collect())
}

fn blend_straight(
    body: &Body,
    selection: &[usize],
    size: f64,
    kind: Kind,
    tolerance: f64,
) -> Result<Parts, BlendDeclined> {
    let eps = 1e-9 * extent(body.vertices());
    let mut parts = Parts::of(body);
    let mut sides = Sides {
        body,
        loops: HashMap::new(),
    };

    // Who uses what.
    let mut edge_faces: Vec<Vec<usize>> = vec![Vec::new(); body.edges().len()];
    for (fi, f) in body.faces().iter().enumerate() {
        for &e in &f.edges {
            edge_faces[e].push(fi);
        }
    }
    let mut ending_at: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut passed_through: HashSet<usize> = HashSet::new();
    for (ei, e) in body.edges().iter().enumerate() {
        let n = e.vertices.len();
        if e.closed || n < 2 {
            passed_through.extend(e.vertices.iter().copied());
            continue;
        }
        ending_at.entry(e.vertices[0]).or_default().push(ei);
        ending_at.entry(e.vertices[n - 1]).or_default().push(ei);
        passed_through.extend(e.vertices[1..n - 1].iter().copied());
    }

    // The selected edges: what they join, which way they turn, how far back
    // each face is cut.
    let mut chosen: Vec<Chosen> = Vec::with_capacity(selection.len());
    let mut chosen_at: HashMap<usize, usize> = HashMap::new();
    for &edge in selection {
        let unsupported = BlendDeclined::UnsupportedEdge { edge };
        let (first, last, dir, _) = straight_edge(body, edge, eps).ok_or(unsupported.clone())?;
        let &[a, b] = edge_faces[edge].as_slice() else {
            return Err(unsupported);
        };
        if a == b {
            return Err(unsupported);
        }
        let (Some((_, na)), Some((_, nb))) = (plane_normals(body, a), plane_normals(body, b))
        else {
            return Err(unsupported);
        };
        let cos = v3::dot(na, nb);
        if cos.abs() > 1.0 - 1e-9 {
            return Err(unsupported);
        }
        // Face A falls away behind face B across a convex edge, and rises in
        // front of it across a concave one.
        let side = v3::dot(sides.inward(a, edge).ok_or(unsupported.clone())?, nb);
        if side.abs() < 1e-9 {
            return Err(unsupported);
        }
        let setback = match kind {
            Kind::Chamfer => size,
            // The ball touches each face `r·tan(φ/2)` from the edge, `φ` being
            // the angle between the normals: `r` itself at a right angle.
            Kind::Fillet => size * ((1.0 - cos) / (1.0 + cos)).sqrt(),
        };
        chosen_at.insert(edge, chosen.len());
        chosen.push(Chosen {
            edge,
            ends: [first, last],
            dir,
            faces: [a, b],
            normals: [na, nb],
            convex: if side < 0.0 { 1.0 } else { -1.0 },
            setback,
            surface: usize::MAX,
            boundary: Vec::new(),
        });
    }

    // The corners they end at.
    let mut touched: Vec<usize> = chosen.iter().flat_map(|c| c.ends).collect();
    touched.sort_unstable();
    touched.dedup();

    let mut corners: Vec<Corner> = Vec::with_capacity(touched.len());
    let mut corner_at: HashMap<usize, usize> = HashMap::new();
    let mut cut_back: HashMap<usize, f64> = HashMap::new();
    for &vertex in &touched {
        let unsupported = BlendDeclined::UnsupportedVertex { vertex };
        let at = body.vertices()[vertex];
        let Some(&[e0, e1, e2]) = ending_at.get(&vertex).map(Vec::as_slice) else {
            return Err(unsupported);
        };
        if passed_through.contains(&vertex) {
            return Err(unsupported);
        }
        let edges = [e0, e1, e2];
        let mut dirs = [[0.0; 3]; 3];
        for i in 0..3 {
            let (first, _, dir, _) =
                straight_edge(body, edges[i], eps).ok_or(unsupported.clone())?;
            dirs[i] = if first == vertex {
                dir
            } else {
                v3::scale(dir, -1.0)
            };
        }
        let mut faces = [0usize; 3];
        for (i, face) in faces.iter_mut().enumerate() {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let mut common = edge_faces[edges[j]]
                .iter()
                .filter(|f| edge_faces[edges[k]].contains(f));
            let (Some(&f), None) = (common.next(), common.next()) else {
                return Err(unsupported);
            };
            if plane_normals(body, f).is_none() {
                return Err(unsupported);
            }
            *face = f;
        }
        if faces[0] == faces[1] || faces[1] == faces[2] || faces[2] == faces[0] {
            return Err(unsupported);
        }
        let selected = [0, 1, 2].map(|i| chosen_at.contains_key(&edges[i]));
        let of = |i: usize| &chosen[chosen_at[&edges[i]]];
        let mut turn = (0..3).filter(|&i| selected[i]).map(|i| of(i).convex);
        let first_turn = turn.next().unwrap_or(1.0);
        if turn.any(|t| t != first_turn) {
            return Err(BlendDeclined::MixedConvexity { vertex });
        }

        // An unselected edge is cut back to where its neighbour's blend leaves
        // the face they share. With both neighbours selected the two blends
        // have to reach it at the same depth, or they do not meet in a mitre.
        let mut ends = [None; 3];
        for k in 0..3 {
            if selected[k] {
                continue;
            }
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            let mut depth: Option<f64> = None;
            for (face, other) in [(faces[i], j), (faces[j], i)] {
                if !selected[other] {
                    continue;
                }
                let into = sides
                    .inward(face, edges[other])
                    .ok_or(unsupported.clone())?;
                let reach = v3::dot(dirs[k], into);
                // Not positive at a reflex corner: the blend runs past the
                // vertex and into the face beyond, which is another shape.
                if reach < 1e-6 {
                    return Err(unsupported);
                }
                let d = of(other).setback / reach;
                if depth.is_some_and(|prev| (prev - d).abs() > 1e-7 * (1.0 + d.abs())) {
                    return Err(BlendDeclined::UnequalSetback { vertex });
                }
                depth = Some(d);
            }
            if let Some(d) = depth {
                *cut_back.entry(edges[k]).or_insert(0.0) += d;
                ends[k] = Some(parts.vertex(v3::add(at, v3::scale(dirs[k], d))));
            }
        }

        // And the vertex itself becomes one point per face: where the face's
        // two setback lines cross, or where its one setback line meets the
        // edge that was cut back.
        let mut points = [None; 3];
        for i in 0..3 {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            points[i] = match (selected[j], selected[k]) {
                (true, true) => {
                    let into_j = sides
                        .inward(faces[i], edges[j])
                        .ok_or(unsupported.clone())?;
                    let into_k = sides
                        .inward(faces[i], edges[k])
                        .ok_or(unsupported.clone())?;
                    let (along_j, along_k) = (v3::dot(dirs[j], into_k), v3::dot(dirs[k], into_j));
                    if along_j.abs() < 1e-6 || along_k.abs() < 1e-6 {
                        return Err(unsupported);
                    }
                    let p = v3::add(
                        at,
                        v3::add(
                            v3::scale(dirs[j], of(k).setback / along_j),
                            v3::scale(dirs[k], of(j).setback / along_k),
                        ),
                    );
                    Some(parts.vertex(p))
                }
                (true, false) => ends[k],
                (false, true) => ends[j],
                (false, false) => None,
            };
        }

        corner_at.insert(vertex, corners.len());
        corners.push(Corner {
            vertex,
            edges,
            dirs,
            faces,
            selected,
            ends,
            points,
        });
    }

    // Is there room? An edge cut back from both ends has to have something
    // left, and so does each side of a blend.
    for (&edge, &cut) in &cut_back {
        let (_, _, _, length) =
            straight_edge(body, edge, eps).ok_or(BlendDeclined::UnsupportedEdge { edge })?;
        if cut >= length - eps {
            return Err(BlendDeclined::TooLarge { edge });
        }
    }
    let rail = |c: &Chosen, side: usize| -> Option<[usize; 2]> {
        Some([
            corners[corner_at[&c.ends[0]]].point_on(c.faces[side])?,
            corners[corner_at[&c.ends[1]]].point_on(c.faces[side])?,
        ])
    };
    for c in &chosen {
        for side in 0..2 {
            let [p, q] = rail(c, side).ok_or(BlendDeclined::UnsupportedEdge { edge: c.edge })?;
            let run = v3::dot(v3::sub(parts.vertices[q], parts.vertices[p]), c.dir);
            if run <= eps {
                return Err(BlendDeclined::TooLarge { edge: c.edge });
            }
        }
    }

    // And the strip of each face a blend takes has to be the face's to give:
    // nothing else that bounds the face — a hole, a notch, the far side — may
    // reach into it. The edges at the blend's own two corners are the strip's
    // ends and are judged above.
    for c in &chosen {
        let declined = BlendDeclined::TooLarge { edge: c.edge };
        let from = body.vertices()[c.ends[0]];
        let length = v3::dist(from, body.vertices()[c.ends[1]]);
        let mut own: Vec<usize> = c
            .ends
            .iter()
            .flat_map(|v| corners[corner_at[v]].edges)
            .collect();
        own.push(c.edge);
        for &face in &c.faces {
            let into = sides
                .inward(face, c.edge)
                .ok_or(BlendDeclined::UnsupportedEdge { edge: c.edge })?;
            let flat = |v: usize| {
                let d = v3::sub(body.vertices()[v], from);
                [v3::dot(d, c.dir), v3::dot(d, into)]
            };
            for &e in &body.faces()[face].edges {
                if own.contains(&e) {
                    continue;
                }
                for w in body.edges()[e].vertices.windows(2) {
                    if segment_meets_box(
                        flat(w[0]),
                        flat(w[1]),
                        [eps, -eps],
                        [length - eps, c.setback + eps],
                    ) {
                        return Err(declined);
                    }
                }
            }
        }
    }

    // The blend surfaces.
    for c in &mut chosen {
        let unsupported = BlendDeclined::UnsupportedEdge { edge: c.edge };
        let [na, nb] = c.normals;
        let between = v3::add(na, nb);
        let surface = match kind {
            Kind::Chamfer => {
                let [pa, _] = rail(c, 0).ok_or(unsupported.clone())?;
                let [pb, _] = rail(c, 1).ok_or(unsupported.clone())?;
                let (pa, pb) = (parts.vertices[pa], parts.vertices[pb]);
                let n = v3::normalize(v3::cross(c.dir, v3::sub(pb, pa))).ok_or(unsupported)?;
                let n = if v3::dot(n, between) < 0.0 {
                    v3::scale(n, -1.0)
                } else {
                    n
                };
                Surface::plane(pa, n)
            }
            Kind::Fillet => {
                // The axis is where the two faces, each moved `r` towards the
                // ball, intersect. `u = 0` is pointed at the edge, so the patch
                // straddles it and stays clear of the branch cut.
                let cos = v3::dot(na, nb);
                let origin = v3::sub(
                    body.vertices()[c.ends[0]],
                    v3::scale(between, c.convex * size / (1.0 + cos)),
                );
                Surface::cylinder(origin, c.dir, size).with_x_dir(v3::scale(between, c.convex))
            }
        };
        parts.surfaces.push(surface);
        c.surface = parts.surfaces.len() - 1;
    }

    // Edges. An unselected edge keeps its index and moves an end; a selected
    // one becomes the rail along its first face, and the rail along its second
    // is new.
    for corner in &corners {
        for k in 0..3 {
            let Some(end) = corner.ends[k] else { continue };
            let e = &mut parts.edges[corner.edges[k]];
            let (first, last) = (e.vertices[0], e.vertices[e.vertices.len() - 1]);
            e.vertices = if first == corner.vertex {
                vec![end, last]
            } else {
                vec![first, end]
            };
        }
    }
    for c in &mut chosen {
        let unsupported = BlendDeclined::UnsupportedEdge { edge: c.edge };
        let [a, b] = c.faces;
        let (sa, sb) = (body.faces()[a].surface, body.faces()[b].surface);
        let (ra, rb) = (
            rail(c, 0).ok_or(unsupported.clone())?,
            rail(c, 1).ok_or(unsupported)?,
        );
        parts.edges[c.edge] = Edge {
            surfaces: pair(sa, c.surface),
            vertices: ra.to_vec(),
            closed: false,
        };
        parts.edges.push(Edge {
            surfaces: pair(sb, c.surface),
            vertices: rb.to_vec(),
            closed: false,
        });
        let rail_b = parts.edges.len() - 1;
        for slot in parts.faces[b].edges.iter_mut() {
            if *slot == c.edge {
                *slot = rail_b;
            }
        }
        c.boundary = vec![c.edge, rail_b];
    }

    // The ends of each blend, corner by corner.
    let mut fresh: Vec<usize> = Vec::new();
    for corner in &corners {
        let vertex = corner.vertex;
        let unsupported = BlendDeclined::UnsupportedVertex { vertex };
        let at = body.vertices()[vertex];
        let which = |i: usize| chosen_at[&corner.edges[i]];
        let count = corner.selected.iter().filter(|s| **s).count();
        let convex = (0..3)
            .find(|&i| corner.selected[i])
            .map_or(1.0, |i| chosen[which(i)].convex);
        // Where the three faces, each moved `r` towards the ball, meet: the
        // point every fillet axis at this corner passes through.
        let centre = || -> Option<V3> {
            let n = [0, 1, 2].map(|i| plane_normals(body, corner.faces[i]).map(|(_, out)| out));
            let w = solve3([n[0]?, n[1]?, n[2]?], [1.0; 3])?;
            Some(v3::sub(at, v3::scale(w, convex * size)))
        };

        match count {
            // One blend, ending on the third face.
            1 => {
                let i = (0..3).find(|&i| corner.selected[i]).unwrap_or(0);
                let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                let ci = which(i);
                let (Some(from), Some(to)) = (corner.points[j], corner.points[k]) else {
                    return Err(unsupported);
                };
                let third = corner.faces[i];
                let (normal, _) = plane_normals(body, third).ok_or(unsupported.clone())?;
                let between = match kind {
                    Kind::Chamfer => Vec::new(),
                    Kind::Fillet => section(
                        &parts.surfaces[chosen[ci].surface].clone(),
                        at,
                        normal,
                        parts.vertices[from],
                        parts.vertices[to],
                        tolerance,
                        &mut parts,
                    )
                    .ok_or(unsupported)?,
                };
                let mut vertices = vec![from];
                vertices.extend(between);
                vertices.push(to);
                parts.edges.push(Edge {
                    surfaces: pair(chosen[ci].surface, body.faces()[third].surface),
                    vertices,
                    closed: false,
                });
                let e = parts.edges.len() - 1;
                parts.faces[third].edges.push(e);
                chosen[ci].boundary.push(e);
            }
            // Two blends, mitred into each other.
            2 => {
                let k = (0..3).find(|&k| !corner.selected[k]).unwrap_or(0);
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                let (ci, cj) = (which(i), which(j));
                let (Some(from), Some(to)) = (corner.points[k], corner.ends[k]) else {
                    return Err(unsupported);
                };
                let between = match kind {
                    Kind::Chamfer => Vec::new(),
                    Kind::Fillet => {
                        let unequal = BlendDeclined::UnequalSetback { vertex };
                        let o = centre().ok_or(unsupported)?;
                        let (p, q) = (parts.vertices[from], parts.vertices[to]);
                        // Two planes bisect a pair of crossing axes. The mitre
                        // is in the one both its ends are on.
                        let off = |m: V3| {
                            v3::normalize(m).map_or(f64::MAX, |m| {
                                v3::dot(v3::sub(p, o), m).abs() + v3::dot(v3::sub(q, o), m).abs()
                            })
                        };
                        let (plus, minus) = (
                            v3::add(corner.dirs[i], corner.dirs[j]),
                            v3::sub(corner.dirs[i], corner.dirs[j]),
                        );
                        let normal = if off(plus) < off(minus) { plus } else { minus };
                        let normal = v3::normalize(normal).ok_or(unequal.clone())?;
                        section(
                            &parts.surfaces[chosen[ci].surface].clone(),
                            o,
                            normal,
                            p,
                            q,
                            tolerance,
                            &mut parts,
                        )
                        .ok_or(unequal)?
                    }
                };
                let mut vertices = vec![from];
                vertices.extend(between);
                vertices.push(to);
                parts.edges.push(Edge {
                    surfaces: pair(chosen[ci].surface, chosen[cj].surface),
                    vertices,
                    closed: false,
                });
                let e = parts.edges.len() - 1;
                chosen[ci].boundary.push(e);
                chosen[cj].boundary.push(e);
            }
            // Three: bevels meet in a point, fillets around a sphere.
            _ => match kind {
                Kind::Chamfer
                    if super::settings::settings().chamfer_corner
                        == super::settings::ChamferCorner::Facet =>
                {
                    // A facet through the three points that replace the
                    // vertex, one on each original face. The bevel of edge
                    // `k` contains the two on the faces that edge lies
                    // between, so it meets the facet along the line joining
                    // them.
                    let mut at = [0usize; 3];
                    for (k, slot) in at.iter_mut().enumerate() {
                        *slot = corner.points[k].ok_or(unsupported.clone())?;
                    }
                    let p = at.map(|v| parts.vertices[v]);
                    let normal = v3::normalize(v3::cross(v3::sub(p[1], p[0]), v3::sub(p[2], p[0])))
                        .ok_or(unsupported.clone())?;
                    // Outward: towards the vertex that was cut off at a
                    // convex corner, away from it at a concave one.
                    let gone = v3::sub(parts.vertices[vertex], p[0]);
                    let outward = if v3::dot(normal, gone) * convex >= 0.0 {
                        normal
                    } else {
                        v3::scale(normal, -1.0)
                    };
                    parts.surfaces.push(Surface::plane(p[0], outward));
                    let facet = parts.surfaces.len() - 1;
                    let mut around = Vec::with_capacity(3);
                    for k in 0..3 {
                        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                        let ck = which(k);
                        parts.edges.push(Edge {
                            surfaces: pair(chosen[ck].surface, facet),
                            vertices: vec![at[i], at[j]],
                            closed: false,
                        });
                        let e = parts.edges.len() - 1;
                        chosen[ck].boundary.push(e);
                        around.push(e);
                    }
                    fresh.push(parts.faces.len());
                    parts.faces.push(Face {
                        surface: facet,
                        edges: around,
                        flipped: false,
                        u_range: (0.0, 0.0),
                        v_range: (0.0, 0.0),
                        u_wraps: false,
                        v_wraps: false,
                        loops: None,
                    });
                }
                Kind::Chamfer => {
                    let mut rows = [[0.0; 3]; 3];
                    let mut rhs = [0.0; 3];
                    for i in 0..3 {
                        let Surface::Plane { origin, normal, .. } =
                            &parts.surfaces[chosen[which(i)].surface]
                        else {
                            return Err(unsupported);
                        };
                        rows[i] = *normal;
                        rhs[i] = v3::dot(*normal, *origin);
                    }
                    let apex = solve3(rows, rhs).ok_or(unsupported.clone())?;
                    let apex = parts.vertex(apex);
                    for k in 0..3 {
                        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                        let (ci, cj) = (which(i), which(j));
                        let to = corner.points[k].ok_or(unsupported.clone())?;
                        parts.edges.push(Edge {
                            surfaces: pair(chosen[ci].surface, chosen[cj].surface),
                            vertices: vec![apex, to],
                            closed: false,
                        });
                        let e = parts.edges.len() - 1;
                        chosen[ci].boundary.push(e);
                        chosen[cj].boundary.push(e);
                    }
                }
                Kind::Fillet => {
                    let o = centre().ok_or(unsupported.clone())?;
                    // Aimed so the patch sits on the equator at `u = 0`, as far
                    // from the poles and the branch cut as a sphere allows.
                    let mut towards = [0.0; 3];
                    for i in 0..3 {
                        let (_, out) =
                            plane_normals(body, corner.faces[i]).ok_or(unsupported.clone())?;
                        towards = v3::add(towards, v3::scale(out, convex));
                    }
                    let towards = v3::normalize(towards).ok_or(unsupported.clone())?;
                    for i in 0..3 {
                        let (_, out) =
                            plane_normals(body, corner.faces[i]).ok_or(unsupported.clone())?;
                        if v3::dot(towards, out) * convex < 0.2 {
                            return Err(unsupported);
                        }
                    }
                    parts.surfaces.push(
                        Surface::sphere(o, size)
                            .with_axis(perpendicular(towards))
                            .with_x_dir(towards),
                    );
                    let sphere = parts.surfaces.len() - 1;
                    let mut around = Vec::with_capacity(3);
                    for i in 0..3 {
                        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                        let ci = which(i);
                        let (Some(from), Some(to)) = (corner.points[j], corner.points[k]) else {
                            return Err(unsupported);
                        };
                        let cylinder = parts.surfaces[chosen[ci].surface].clone();
                        let Surface::Cylinder { axis, .. } = &cylinder else {
                            return Err(unsupported);
                        };
                        let between = section(
                            &cylinder,
                            o,
                            *axis,
                            parts.vertices[from],
                            parts.vertices[to],
                            tolerance,
                            &mut parts,
                        )
                        .ok_or(unsupported.clone())?;
                        let mut vertices = vec![from];
                        vertices.extend(between);
                        vertices.push(to);
                        parts.edges.push(Edge {
                            surfaces: pair(chosen[ci].surface, sphere),
                            vertices,
                            closed: false,
                        });
                        let e = parts.edges.len() - 1;
                        chosen[ci].boundary.push(e);
                        around.push(e);
                    }
                    fresh.push(parts.faces.len());
                    parts.faces.push(Face {
                        surface: sphere,
                        edges: around,
                        flipped: convex < 0.0,
                        u_range: (0.0, 0.0),
                        v_range: (0.0, 0.0),
                        u_wraps: false,
                        v_wraps: false,
                        loops: None,
                    });
                }
            },
        }
    }

    for c in &chosen {
        fresh.push(parts.faces.len());
        parts.faces.push(Face {
            surface: c.surface,
            edges: c.boundary.clone(),
            // A concave fillet is the inside of its cylinder.
            flipped: kind == Kind::Fillet && c.convex < 0.0,
            u_range: (0.0, 0.0),
            v_range: (0.0, 0.0),
            u_wraps: false,
            v_wraps: false,
            loops: None,
        });
    }

    // A flat face that was re-trimmed goes back to being bounded by its edges:
    // they are all it is bounded by, and a ring stated before the blend names
    // the corner that is no longer there.
    let mut retrimmed: Vec<usize> = corners.iter().flat_map(|c| c.faces).collect();
    retrimmed.sort_unstable();
    retrimmed.dedup();
    for &f in &retrimmed {
        parts.faces[f].loops = None;
    }

    // The new faces' rings, read off the edges just built. A curved one keeps
    // its ring: a fillet between two mitres is not a parameter rectangle.
    let probe = parts.body();
    for &f in &retrimmed {
        if !probe.face_loops(f).iter().any(|l| l.is_outer()) {
            return Err(BlendDeclined::TrimmedFace { face: f });
        }
    }
    for &f in &fresh {
        let loops = probe.face_loops(f);
        let [ring] = loops.as_slice() else {
            return Err(BlendDeclined::NotWatertight { defects: 0 });
        };
        if !ring.is_outer() {
            return Err(BlendDeclined::NotWatertight { defects: 0 });
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for c in &ring.uv {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let face = &mut parts.faces[f];
        face.u_range = (lo[0], hi[0]);
        face.v_range = (lo[1], hi[1]);
        if !matches!(parts.surfaces[face.surface], Surface::Plane { .. }) {
            face.loops = Some(loops);
        }
    }
    Ok(parts)
}

/// Blend the circular rim where a plane caps a cylinder.
///
/// There are four of these and they are one construction. The cylinder is a
/// shaft or a bore; the plane closes it from the material side (a shaft's end,
/// a hole's mouth — convex) or from the open side (a boss's foot, a blind
/// hole's floor — concave). In every case the plane's rim moves radially *into
/// the plane's face* and the cylinder's rim moves axially *into the cylinder's
/// face*, by the same amount, and the blend is the cone or the torus through
/// the two circles that result.
fn blend_rim(
    parts: &mut Parts,
    edge: usize,
    size: f64,
    kind: Kind,
    tolerance: f64,
) -> Result<(), BlendDeclined> {
    let unsupported = BlendDeclined::UnsupportedEdge { edge };
    let too_large = BlendDeclined::TooLarge { edge };
    let eps = 1e-9 * extent(&parts.vertices);

    let users: Vec<usize> = (0..parts.faces.len())
        .filter(|&f| parts.faces[f].edges.contains(&edge))
        .collect();
    let &[f0, f1] = users.as_slice() else {
        return Err(unsupported);
    };
    let is_plane = |f: usize| {
        matches!(
            parts.surfaces[parts.faces[f].surface],
            Surface::Plane { .. }
        )
    };
    let (cap, wall) = if is_plane(f0) { (f0, f1) } else { (f1, f0) };
    let (
        Surface::Plane {
            origin: cap_origin,
            normal: cap_normal,
            ..
        },
        Surface::Cylinder {
            origin,
            axis,
            x_dir,
            radius,
        },
    ) = (
        parts.surfaces[parts.faces[cap].surface].clone(),
        parts.surfaces[parts.faces[wall].surface].clone(),
    )
    else {
        return Err(unsupported);
    };
    if v3::dot(cap_normal, axis).abs() < 1.0 - 1e-9 {
        return Err(unsupported);
    }
    // The wall has to be the plain band between two rims; one a boolean has
    // trimmed states rings this does not know how to move.
    if parts.faces[wall].loops.is_some() {
        return Err(BlendDeclined::TrimmedFace { face: wall });
    }
    if !parts.faces[wall].u_wraps {
        return Err(unsupported);
    }

    // Heights along the axis, which is the wall's `v`.
    let height = v3::dot(v3::sub(cap_origin, origin), axis);
    let (v0, v1) = parts.faces[wall].v_range;
    let near_end = if (v0 - height).abs() <= (v1 - height).abs() {
        0
    } else {
        1
    };
    let far = if near_end == 0 { v1 } else { v0 };
    let into_wall = if far > height { 1.0 } else { -1.0 };
    if (far - height).abs() <= size + eps {
        return Err(too_large);
    }

    let cap_out = if parts.faces[cap].flipped {
        v3::scale(cap_normal, -1.0)
    } else {
        cap_normal
    };
    // Convex when the wall leaves the cap on the cap's material side.
    let convex = if v3::dot(axis, cap_out) * into_wall < 0.0 {
        1.0
    } else {
        -1.0
    };
    // A shaft is solid inside its cylinder, a bore solid outside it.
    let shaft = if parts.faces[wall].flipped { -1.0 } else { 1.0 };
    let grows = -shaft * convex;
    let cap_radius = radius + grows * size;
    let wall_height = height + into_wall * size;
    if cap_radius <= eps {
        return Err(too_large);
    }
    // Whatever else bounds the cap must be clear of the ring the rim sweeps.
    // Judged by segment: a square plate's sides come nearest the hole between
    // their ends, not at them.
    let on_axis = |h: f64| v3::add(origin, v3::scale(axis, h));
    let (inner, outer) = (radius.min(cap_radius), radius.max(cap_radius));
    let hub = on_axis(height);
    for &e in &parts.faces[cap].edges {
        if e == edge {
            continue;
        }
        for w in parts.edges[e].vertices.windows(2) {
            let (p, q) = (parts.vertices[w[0]], parts.vertices[w[1]]);
            let pq = v3::sub(q, p);
            let t = (v3::dot(v3::sub(hub, p), pq) / v3::dot(pq, pq).max(1e-300)).clamp(0.0, 1.0);
            let nearest = v3::dist(hub, v3::add(p, v3::scale(pq, t)));
            let farthest = v3::dist(hub, p).max(v3::dist(hub, q));
            if nearest < outer + eps && farthest > inner - eps {
                return Err(too_large);
            }
        }
    }
    let (surface, v_range, flipped) = match kind {
        Kind::Chamfer => {
            // A 45° cone opening from the smaller circle to the larger.
            let widens = if cap_radius > radius {
                -into_wall
            } else {
                into_wall
            };
            let apex = on_axis(wall_height - widens * radius);
            (
                Surface::cone(apex, v3::scale(axis, widens), FRAC_PI_4).with_x_dir(x_dir),
                (radius.min(cap_radius), radius.max(cap_radius)),
                shaft < 0.0,
            )
        }
        Kind::Fillet => {
            // The tube's centre is under the cap's new rim and beside the
            // wall's: `size` from each.
            let at_cap = -into_wall * FRAC_PI_2;
            let at_wall = if grows < 0.0 { 0.0 } else { -into_wall * PI };
            (
                Surface::torus(on_axis(wall_height), axis, cap_radius, size).with_x_dir(x_dir),
                (at_cap.min(at_wall), at_cap.max(at_wall)),
                convex < 0.0,
            )
        }
    };

    // Both rims at the same angles, fine enough for the larger.
    let widest = radius.max(cap_radius);
    let n = if tolerance < widest {
        ((PI / (1.0 - tolerance / widest).acos()).ceil() as usize).clamp(16, 4096)
    } else {
        16
    };
    let y_dir = v3::cross(axis, x_dir);
    let ring = |h: f64, r: f64, parts: &mut Parts| -> Vec<usize> {
        let mut ids: Vec<usize> = (0..n)
            .map(|i| {
                let (s, c) = (-PI + TAU * i as f64 / n as f64).sin_cos();
                let radial = v3::add(v3::scale(x_dir, c), v3::scale(y_dir, s));
                parts.vertex(v3::add(on_axis(h), v3::scale(radial, r)))
            })
            .collect();
        ids.push(ids[0]);
        ids
    };
    let cap_rim = ring(height, cap_radius, parts);
    let wall_rim = ring(wall_height, radius, parts);

    parts.surfaces.push(surface);
    let blend = parts.surfaces.len() - 1;
    let (cap_surface, wall_surface) = (parts.faces[cap].surface, parts.faces[wall].surface);
    parts.edges[edge] = Edge {
        surfaces: pair(cap_surface, blend),
        vertices: cap_rim,
        closed: true,
    };
    parts.edges.push(Edge {
        surfaces: pair(wall_surface, blend),
        vertices: wall_rim,
        closed: true,
    });
    let wall_edge = parts.edges.len() - 1;

    let face = &mut parts.faces[wall];
    for slot in face.edges.iter_mut() {
        if *slot == edge {
            *slot = wall_edge;
        }
    }
    if near_end == 0 {
        face.v_range.0 = wall_height;
    } else {
        face.v_range.1 = wall_height;
    }
    // The cap is flat, so its edges are all that bound it.
    parts.faces[cap].loops = None;
    parts.faces.push(Face {
        surface: blend,
        edges: vec![edge, wall_edge],
        flipped,
        u_range: (-PI, PI),
        v_range,
        u_wraps: true,
        v_wraps: false,
        loops: None,
    });
    if !parts.body().face_loops(cap).iter().any(|l| l.is_outer()) {
        return Err(BlendDeclined::TrimmedFace { face: cap });
    }
    Ok(())
}

/// A smooth chain of edges bounding a flat face — the *cap* — each between
/// the cap and a side square to it, a plane or a cylinder whose axis is the
/// cap's normal, the sides tangent where they meet: the top of a rounded
/// rectangle, a slot, the mouth of a pocket with rounded corners.
///
/// A blend that starts along one of its edges runs on along the chain, as
/// OpenCASCADE's does where it follows tangent edges: round the whole ring
/// where every vertex of it is smooth, and otherwise to the sharp corners
/// either side, where it ends on the third face there as a single straight
/// blend does. Its pieces are the rim blends and the straight ones, a plane
/// or a cone (chamfer), a cylinder or a torus (fillet) for each edge, meeting
/// their neighbours tangentially along the line from the cap's new boundary
/// to the side's.
struct TangentLoop {
    cap: usize,
    /// The chain's edges in the order the cap's ring walks them, and whether
    /// it walks each against its stored order.
    edges: Vec<(usize, bool)>,
    /// The edge leaving the chain at each vertex along it, `risers[i]` at the
    /// start of `edges[i]`; for an open chain one more, and `None` at its two
    /// ends.
    risers: Vec<Option<usize>>,
    /// Where an open chain stops: at the start of its first edge and the end
    /// of its last.
    ends: Option<[ChainEnd; 2]>,
}

/// A sharp corner an open chain ends at: three straight edges, the chain's,
/// one more on the cap and one on the side, and the face across them.
struct ChainEnd {
    vertex: usize,
    on_cap: usize,
    on_side: usize,
    third: usize,
}

/// The faces using each edge.
fn edge_faces(edges: &[Edge], faces: &[Face]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); edges.len()];
    for (fi, f) in faces.iter().enumerate() {
        for &e in &f.edges {
            out[e].push(fi);
        }
    }
    out
}

/// A face's outward normal at a point on it, for a plane or a cylinder.
fn outward_at(body: &Body, face: usize, p: V3) -> Option<V3> {
    let f = &body.faces()[face];
    let n = match &body.surfaces()[f.surface] {
        Surface::Plane { normal, .. } => *normal,
        Surface::Cylinder { origin, axis, .. } => {
            let d = v3::sub(p, *origin);
            v3::normalize(v3::sub(d, v3::scale(*axis, v3::dot(d, *axis))))?
        }
        _ => return None,
    };
    Some(if f.flipped { v3::scale(n, -1.0) } else { n })
}

/// How many pieces an arc of `radius` through `angle` needs to be within
/// `tolerance` of its chords.
fn arc_pieces(radius: f64, angle: f64, tolerance: f64) -> usize {
    if tolerance >= radius {
        return 2;
    }
    let step = 2.0 * (1.0 - tolerance / radius).acos();
    ((angle.abs() / step).ceil() as usize).clamp(2, 4096)
}

/// The chain along a flat face that `edge` bounds, if it is a
/// [`TangentLoop`] with at least one curved edge.
fn tangent_loop(body: &Body, edge: usize, eps: f64) -> Option<TangentLoop> {
    let users = edge_faces(body.edges(), body.faces());
    let e = &body.edges()[edge];
    if e.closed || e.vertices.len() < 2 {
        return None;
    }
    let mut ending: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut inner: HashSet<usize> = HashSet::new();
    for (ei, e) in body.edges().iter().enumerate() {
        let n = e.vertices.len();
        if e.closed || n < 2 {
            inner.extend(e.vertices.iter().copied());
            continue;
        }
        ending.entry(e.vertices[0]).or_default().push(ei);
        ending.entry(e.vertices[n - 1]).or_default().push(ei);
        inner.extend(e.vertices[1..n - 1].iter().copied());
    }
    let other = |e: usize, cap: usize| -> Option<usize> {
        match *users[e].as_slice() {
            [a, b] if a == cap && b != cap => Some(b),
            [a, b] if b == cap && a != cap => Some(a),
            _ => None,
        }
    };
    'caps: for &cap in &users[edge] {
        let Some((normal, _)) = plane_normals(body, cap) else {
            continue;
        };
        // The ring through the edge, as steps between vertices, each named by
        // the edge it is part of.
        let (p, q) = (e.vertices[0], e.vertices[1]);
        let loops = body.face_loops(cap);
        let Some(ring) = loops.iter().find(|l| {
            let n = l.vertices.len();
            (0..n).any(|i| {
                let s = (l.vertices[i], l.vertices[(i + 1) % n]);
                s == (p, q) || s == (q, p)
            })
        }) else {
            continue;
        };
        let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
        for &ce in &body.faces()[cap].edges {
            for w in body.edges()[ce].vertices.windows(2) {
                owner.insert(pair(w[0], w[1]), ce);
            }
        }
        let n = ring.vertices.len();
        let mut steps = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (ring.vertices[i], ring.vertices[(i + 1) % n]);
            let Some(&ce) = owner.get(&pair(a, b)) else {
                continue 'caps;
            };
            steps.push((ce, a));
        }
        // Start at an edge's first step.
        let Some(start) = (0..n).find(|&i| steps[i].0 != steps[(i + n - 1) % n].0) else {
            continue;
        };
        steps.rotate_left(start);
        let mut ring_edges: Vec<(usize, bool)> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        for (i, &(ce, a)) in steps.iter().enumerate() {
            if i > 0 && steps[i - 1].0 == ce {
                continue;
            }
            let v = &body.edges()[ce].vertices;
            if v[0] != a && v[v.len() - 1] != a {
                continue 'caps;
            }
            ring_edges.push((ce, v[0] != a));
            starts.push(a);
        }
        // An edge met twice is a ring that doubles back on itself.
        let mut seen = HashSet::new();
        let m = ring_edges.len();
        if ring_edges.iter().any(|(ce, _)| !seen.insert(*ce)) || m < 2 {
            continue;
        }
        let Some(k) = ring_edges.iter().position(|r| r.0 == edge) else {
            continue;
        };

        // Which edges have a side square to the cap (and which of those are
        // curved), and which vertices are smooth, with a riser straight and
        // square to the cap too.
        let sides: Vec<Option<(usize, bool)>> = ring_edges
            .iter()
            .map(|&(ce, _)| {
                let side = other(ce, cap)?;
                match &body.surfaces()[body.faces()[side].surface] {
                    Surface::Plane { normal: m, .. } => (v3::dot(*m, normal).abs() <= 1e-9
                        && straight_edge(body, ce, eps).is_some())
                    .then_some((side, false)),
                    Surface::Cylinder { axis, .. } => (v3::dot(*axis, normal).abs() >= 1.0 - 1e-9
                        && !body.faces()[side].u_wraps)
                        .then_some((side, true)),
                    _ => None,
                }
            })
            .collect();
        let smooth = |i: usize| -> Option<usize> {
            let prev = (i + m - 1) % m;
            let ((side, _), (before, _)) = (sides[i]?, sides[prev]?);
            let v = starts[i];
            let at = ending.get(&v)?;
            if at.len() != 3 || inner.contains(&v) || before == side {
                return None;
            }
            let (ce, pe) = (ring_edges[i].0, ring_edges[prev].0);
            let &riser = at.iter().find(|&&r| r != ce && r != pe)?;
            let (_, _, dir, _) = straight_edge(body, riser, eps)?;
            if v3::dot(dir, normal).abs() < 1.0 - 1e-9 {
                return None;
            }
            let at_v = body.vertices()[v];
            let (m0, m1) = (
                outward_at(body, before, at_v)?,
                outward_at(body, side, at_v)?,
            );
            (v3::dot(m0, m1) >= 1.0 - 1e-6).then_some(riser)
        };
        if sides[k].is_none() {
            continue;
        }
        let all: Vec<Option<usize>> = (0..m).map(smooth).collect();
        if all.iter().all(Option::is_some) {
            if !sides.iter().flatten().any(|s| s.1) {
                continue;
            }
            return Some(TangentLoop {
                cap,
                edges: ring_edges,
                risers: all,
                ends: None,
            });
        }

        // Open: from the edge back and on to the sharp corners either side.
        let (mut lo, mut len) = (k, 1usize);
        while all[lo].is_some() {
            lo = (lo + m - 1) % m;
            len += 1;
        }
        while all[(lo + len) % m].is_some() {
            len += 1;
        }
        let chain: Vec<usize> = (0..len).map(|i| (lo + i) % m).collect();
        if !chain.iter().any(|&i| sides[i].is_some_and(|s| s.1)) {
            continue;
        }
        let end_at =
            |vertex_index: usize, chain_edge: usize, neighbour: usize| -> Option<ChainEnd> {
                let v = starts[vertex_index];
                let (side, curved) = sides[chain_edge]?;
                if curved {
                    return None;
                }
                let at = ending.get(&v)?;
                if at.len() != 3 || inner.contains(&v) {
                    return None;
                }
                let ce = ring_edges[chain_edge].0;
                let on_cap = ring_edges[neighbour].0;
                let &on_side = at.iter().find(|&&r| r != ce && r != on_cap)?;
                straight_edge(body, on_cap, eps)?;
                straight_edge(body, on_side, eps)?;
                let third = *users[on_cap]
                    .iter()
                    .find(|f| **f != cap && **f != side && users[on_side].contains(f))?;
                plane_normals(body, third)?;
                Some(ChainEnd {
                    vertex: v,
                    on_cap,
                    on_side,
                    third,
                })
            };
        let last = chain[len - 1];
        let (Some(first_end), Some(last_end)) = (
            end_at(lo, lo, (lo + m - 1) % m),
            end_at((last + 1) % m, last, (last + 1) % m),
        ) else {
            continue;
        };
        let mut risers: Vec<Option<usize>> = chain.iter().map(|&i| all[i]).collect();
        risers[0] = None;
        risers.push(None);
        return Some(TangentLoop {
            cap,
            edges: chain.iter().map(|&i| ring_edges[i]).collect(),
            risers,
            ends: Some([first_end, last_end]),
        });
    }
    None
}

/// Blend every edge of a [`TangentLoop`].
fn blend_loop(
    parts: &mut Parts,
    plan: &TangentLoop,
    size: f64,
    kind: Kind,
    tolerance: f64,
) -> Result<(), BlendDeclined> {
    let probe = parts.body();
    let body = &probe;
    let eps = 1e-9 * extent(body.vertices());
    let users = edge_faces(body.edges(), body.faces());
    let first = plan.edges[0].0;
    let unsupported = BlendDeclined::UnsupportedEdge { edge: first };
    let (normal, cap_out) = plane_normals(body, plan.cap).ok_or(unsupported.clone())?;
    let count = plan.edges.len();
    let open = plan.ends.is_some();
    // Vertex `i` starts edge `i`; an open chain has one more, ending it.
    let corners = if open { count + 1 } else { count };
    let next = |i: usize| if open { i + 1 } else { (i + 1) % count };

    // Each edge as the ring walks it.
    let walked: Vec<Vec<usize>> = plan
        .edges
        .iter()
        .map(|&(e, back)| {
            let mut v = body.edges()[e].vertices.clone();
            if back {
                v.reverse();
            }
            v
        })
        .collect();
    let vertex_at = |i: usize| -> usize {
        if i < count {
            walked[i][0]
        } else {
            walked[count - 1][walked[count - 1].len() - 1]
        }
    };
    let side_of = |i: usize| -> usize {
        let e = plan.edges[i].0;
        if users[e][0] == plan.cap {
            users[e][1]
        } else {
            users[e][0]
        }
    };

    // Into the cap, square to the ring, at a point of edge `i`.
    let inward = |i: usize, p: V3| -> Option<V3> {
        let w = &walked[i];
        let (a, b) = (body.vertices()[w[0]], body.vertices()[w[1]]);
        let rough = v3::normalize(v3::cross(normal, v3::sub(b, a)))?;
        match &body.surfaces()[body.faces()[side_of(i)].surface] {
            Surface::Cylinder { origin, axis, .. } => {
                let d = v3::sub(p, *origin);
                let radial = v3::normalize(v3::sub(d, v3::scale(*axis, v3::dot(d, *axis))))?;
                Some(if v3::dot(radial, rough) < 0.0 {
                    v3::scale(radial, -1.0)
                } else {
                    radial
                })
            }
            _ => Some(rough),
        }
    };
    // The edge a vertex is measured from: its own, or the last one's at the
    // end of an open chain.
    let edge_at = |i: usize| i.min(count - 1);

    // Away from the cap along the sides, and how far there is to go.
    let mut down: Option<V3> = None;
    for (i, r) in plan.risers.iter().enumerate() {
        let Some(r) = *r else { continue };
        let v = vertex_at(i);
        let rv = &body.edges()[r].vertices;
        let far = if rv[0] == v { rv[rv.len() - 1] } else { rv[0] };
        let d = v3::sub(body.vertices()[far], body.vertices()[v]);
        if v3::norm(d) <= size + eps {
            if std::env::var("THREERS_BREP_DEBUG").is_ok() {
                eprintln!("  [blend] riser {r} is only {}", v3::norm(d));
            }
            return Err(BlendDeclined::TooLarge { edge: r });
        }
        let d = v3::normalize(d).ok_or(unsupported.clone())?;
        if down.is_some_and(|prev| v3::dot(prev, d) < 1.0 - 1e-9) {
            return Err(unsupported);
        }
        down = Some(d);
    }
    // Along an edge leaving a sharp end down the side: away from the cap,
    // whichever way that is.
    let away = |e: usize, from: usize| -> Option<V3> {
        let (a, b, dir, _) = straight_edge(body, e, eps)?;
        if a == from {
            Some(dir)
        } else if b == from {
            Some(v3::scale(dir, -1.0))
        } else {
            None
        }
    };
    if let Some(ends) = &plan.ends {
        for end in ends {
            let d = away(end.on_side, end.vertex).ok_or(unsupported.clone())?;
            let along = v3::dot(d, normal);
            if along.abs() < 1e-6 {
                return Err(BlendDeclined::UnsupportedVertex { vertex: end.vertex });
            }
            let d = if along > 0.0 {
                normal
            } else {
                v3::scale(normal, -1.0)
            };
            if down.is_some_and(|prev| v3::dot(prev, d) < 1.0 - 1e-9) {
                return Err(BlendDeclined::UnsupportedVertex { vertex: end.vertex });
            }
            down = Some(d);
        }
    }
    let down = down.ok_or(unsupported.clone())?;

    // One way across every edge: a smooth chain cannot turn from convex to
    // concave.
    let a0 = body.vertices()[walked[0][0]];
    let c0 = inward(0, a0).ok_or(unsupported.clone())?;
    let m0 = outward_at(body, side_of(0), a0).ok_or(unsupported.clone())?;
    let convex = if v3::dot(c0, m0) < 0.0 { 1.0 } else { -1.0 };

    // The strips the blend takes must be clear of everything else bounding
    // the cap and the sides.
    let chain: Vec<usize> = plan.edges.iter().map(|e| e.0).collect();
    let segments = |e: usize| -> Vec<(V3, V3)> {
        body.edges()[e]
            .vertices
            .windows(2)
            .map(|w| (body.vertices()[w[0]], body.vertices()[w[1]]))
            .collect()
    };
    // A point of another edge in the strip a chain edge's blend takes: beside
    // one of its segments, and between it and `size` into the face, `into`
    // giving the way into the face at a point of chain edge `i`. Points
    // beyond the ends are judged by the ends' own edges.
    let in_strip = |i: usize, p: V3, into: &dyn Fn(usize, V3) -> Option<V3>| -> bool {
        walked[i].windows(2).any(|w| {
            let (a, b) = (body.vertices()[w[0]], body.vertices()[w[1]]);
            let ab = v3::sub(b, a);
            let len2 = v3::dot(ab, ab);
            if len2 <= 0.0 {
                return false;
            }
            let t = v3::dot(v3::sub(p, a), ab) / len2;
            if !(1e-9..=1.0 - 1e-9).contains(&t) {
                return false;
            }
            let foot = v3::add(a, v3::scale(ab, t));
            let Some(d) = into(i, foot) else { return false };
            let off = v3::sub(p, foot);
            let depth = v3::dot(off, d);
            let lateral = v3::norm(v3::sub(off, v3::scale(d, depth)));
            depth > eps && depth < size + eps && lateral <= size
        })
    };
    let clear = |face: usize,
                 own: &[usize],
                 of: &[usize],
                 into: &dyn Fn(usize, V3) -> Option<V3>|
     -> bool {
        body.faces()[face]
            .edges
            .iter()
            .filter(|e| !own.contains(e))
            .all(|&e| {
                // Each segment sampled finely enough that a strip `size`
                // deep cannot fall between two samples.
                let ok = segments(e).iter().all(|&(p, q)| {
                    let n = ((v3::dist(p, q) / (0.25 * size)).ceil() as usize).clamp(1, 4096);
                    (0..=n).all(|k| {
                        let x = v3::add(p, v3::scale(v3::sub(q, p), k as f64 / n as f64));
                        !of.iter().any(|&i| in_strip(i, x, into))
                    })
                });
                if !ok && std::env::var("THREERS_BREP_DEBUG").is_ok() {
                    eprintln!("  [blend] face {face}: edge {e} reaches into the blend of {of:?}");
                }
                ok
            })
    };
    let mut cap_own = chain.clone();
    let mut end_of: [Option<&ChainEnd>; 2] = [None, None];
    if let Some(ends) = &plan.ends {
        cap_own.extend(ends.iter().map(|e| e.on_cap));
        end_of = [Some(&ends[0]), Some(&ends[1])];
    }
    let all_edges: Vec<usize> = (0..count).collect();
    let into_cap = |i: usize, p: V3| inward(i, p);
    if !clear(plan.cap, &cap_own, &all_edges, &into_cap) {
        return Err(BlendDeclined::TooLarge { edge: first });
    }
    let end_at = |i: usize| -> Option<&ChainEnd> {
        if !open {
            None
        } else if i == 0 {
            end_of[0]
        } else if i == count {
            end_of[1]
        } else {
            None
        }
    };
    for i in 0..count {
        let mut own = vec![plan.edges[i].0];
        for v in [i, next(i)] {
            if let Some(r) = plan.risers[v] {
                own.push(r);
            }
            if let Some(end) = end_at(v) {
                own.push(end.on_side);
            }
        }
        let into_side = |_: usize, _: V3| Some(down);
        if !clear(side_of(i), &own, &[i], &into_side) {
            return Err(BlendDeclined::TooLarge { edge: own[0] });
        }
    }

    // The new corners: where each vertex moves on the cap and on the sides.
    // At a sharp end that is along the two edges it leaves by, cut back to
    // where the blend reaches them.
    let mut on_cap = Vec::with_capacity(corners);
    let mut on_side = Vec::with_capacity(corners);
    // An edge a sharp end's blend reaches the far end of, which it uses up:
    // `(corner, 0 on the cap or 1 down the side, the edge)`.
    let mut consumed: Vec<(usize, usize, usize)> = Vec::new();
    for i in 0..corners {
        let v = vertex_at(i);
        let p = body.vertices()[v];
        let c = inward(edge_at(i), p).ok_or(unsupported.clone())?;
        let ids = match end_at(i) {
            Some(end) => {
                let too_large = BlendDeclined::TooLarge {
                    edge: plan.edges[edge_at(i)].0,
                };
                let mut ids = [usize::MAX; 2];
                for (slot, (e, to)) in [(end.on_cap, c), (end.on_side, down)]
                    .into_iter()
                    .enumerate()
                {
                    let d = away(e, v).ok_or(BlendDeclined::UnsupportedVertex { vertex: v })?;
                    let reach = v3::dot(d, to);
                    if reach < 1e-6 {
                        return Err(BlendDeclined::UnsupportedVertex { vertex: v });
                    }
                    let (a, b, _, length) =
                        straight_edge(body, e, eps).ok_or(unsupported.clone())?;
                    let back = size / reach;
                    ids[slot] = if (back - length).abs() <= 1e-7 * (1.0 + length) {
                        // To the far end exactly: a chamfer as deep as the
                        // bevel beside it meets that bevel's own edge.
                        consumed.push((i, slot, e));
                        if a == v {
                            b
                        } else {
                            a
                        }
                    } else if back > length {
                        return Err(too_large);
                    } else {
                        parts.vertex(v3::add(p, v3::scale(d, back)))
                    };
                }
                if consumed.iter().filter(|c| c.0 == i).count() > 1 {
                    return Err(BlendDeclined::UnsupportedVertex { vertex: v });
                }
                ids
            }
            None => [
                parts.vertex(v3::add(p, v3::scale(c, size))),
                parts.vertex(v3::add(p, v3::scale(down, size))),
            ],
        };
        on_cap.push(ids[0]);
        on_side.push(ids[1]);
    }

    // Each edge's two new boundaries and its blend surface.
    let mut blend_surfaces = Vec::with_capacity(count);
    let mut cap_rails = Vec::with_capacity(count);
    let mut side_rails = Vec::with_capacity(count);
    for i in 0..count {
        let j = next(i);
        let edge = plan.edges[i].0;
        let w = &walked[i];
        let (a, b) = (body.vertices()[w[0]], body.vertices()[w[w.len() - 1]]);
        let side = side_of(i);
        let side_surface = body.surfaces()[body.faces()[side].surface].clone();
        let (cap_rail, side_rail, surface) = match &side_surface {
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                // Angles along the arc as the ring walks it, unwrapped.
                let mut us: Vec<f64> = Vec::with_capacity(w.len());
                for &v in w {
                    let (u, _) = side_surface
                        .invert(body.vertices()[v])
                        .ok_or(unsupported.clone())?;
                    let u = match us.last() {
                        Some(&prev) => u + TAU * ((prev - u) / TAU).round(),
                        None => u,
                    };
                    us.push(u);
                }
                let (u0, u1) = (us[0], us[us.len() - 1]);
                let height = v3::dot(v3::sub(a, *origin), *axis);
                let c = inward(i, a).ok_or(unsupported.clone())?;
                let ra = v3::sub(v3::sub(a, *origin), v3::scale(*axis, height));
                let grows = if v3::dot(c, ra) > 0.0 { 1.0 } else { -1.0 };
                let cap_radius = radius + grows * size;
                if cap_radius <= eps {
                    return Err(BlendDeclined::TooLarge { edge });
                }
                let n = arc_pieces(radius.max(cap_radius), u1 - u0, tolerance);
                let at =
                    |k: usize| side_surface.point(u0 + (u1 - u0) * k as f64 / n as f64, height);
                let mut cap_rail = vec![on_cap[i]];
                let mut side_rail = vec![on_side[i]];
                for k in 1..n {
                    let p = at(k);
                    let c = inward(i, p).ok_or(unsupported.clone())?;
                    cap_rail.push(parts.vertex(v3::add(p, v3::scale(c, size))));
                    side_rail.push(parts.vertex(v3::add(p, v3::scale(down, size))));
                }
                cap_rail.push(on_cap[j]);
                side_rail.push(on_side[j]);
                // Aimed at the arc's middle, clear of the branch cut.
                let middle = v3::sub(at(n / 2), *origin);
                let middle = v3::sub(middle, v3::scale(*axis, v3::dot(middle, *axis)));
                let foot = |h: f64| v3::add(*origin, v3::scale(*axis, h));
                let side_height = height + v3::dot(down, *axis) * size;
                let surface = match kind {
                    Kind::Chamfer => {
                        // 45° from the smaller circle to the larger.
                        let (wide_height, wide) = if cap_radius > *radius {
                            (height, cap_radius)
                        } else {
                            (side_height, *radius)
                        };
                        let narrow_height = if cap_radius > *radius {
                            side_height
                        } else {
                            height
                        };
                        let towards = v3::normalize(v3::scale(*axis, wide_height - narrow_height))
                            .ok_or(unsupported.clone())?;
                        let apex = v3::sub(foot(wide_height), v3::scale(towards, wide));
                        Surface::cone(apex, towards, FRAC_PI_4).with_x_dir(middle)
                    }
                    Kind::Fillet => Surface::torus(foot(side_height), *axis, cap_radius, size)
                        .with_x_dir(middle),
                };
                (cap_rail, side_rail, surface)
            }
            _ => {
                let c = inward(i, a).ok_or(unsupported.clone())?;
                let t = v3::normalize(v3::sub(b, a)).ok_or(unsupported.clone())?;
                let surface = match kind {
                    Kind::Chamfer => {
                        let pc = v3::add(a, v3::scale(c, size));
                        let ps = v3::add(a, v3::scale(down, size));
                        let n = v3::normalize(v3::cross(t, v3::sub(ps, pc)))
                            .ok_or(unsupported.clone())?;
                        let m = outward_at(body, side, a).ok_or(unsupported.clone())?;
                        let between = v3::add(cap_out, m);
                        Surface::plane(
                            pc,
                            if v3::dot(n, between) < 0.0 {
                                v3::scale(n, -1.0)
                            } else {
                                n
                            },
                        )
                    }
                    Kind::Fillet => {
                        let o = v3::add(a, v3::scale(v3::add(c, down), size));
                        let aim = v3::normalize(v3::scale(v3::add(c, down), -1.0))
                            .ok_or(unsupported.clone())?;
                        Surface::cylinder(o, t, size).with_x_dir(aim)
                    }
                };
                (
                    vec![on_cap[i], on_cap[j]],
                    vec![on_side[i], on_side[j]],
                    surface,
                )
            }
        };
        parts.surfaces.push(surface);
        blend_surfaces.push(parts.surfaces.len() - 1);
        cap_rails.push(cap_rail);
        side_rails.push(side_rail);
    }

    // Across the blend at every vertex: between neighbouring blends, from the
    // cap to the side, a straight bevel or a quarter of the ball's circle;
    // at a sharp end, where the blend meets the third face.
    let mut seams = Vec::with_capacity(corners);
    let mut thirds: Vec<(usize, usize)> = Vec::new();
    for i in 0..corners {
        let p = body.vertices()[vertex_at(i)];
        let mut vertices = vec![on_cap[i]];
        let surfaces = match end_at(i) {
            Some(end) => {
                let blend = blend_surfaces[edge_at(i)];
                if kind == Kind::Fillet {
                    let (third_normal, _) =
                        plane_normals(body, end.third).ok_or(unsupported.clone())?;
                    let (from, to) = (parts.vertices[on_cap[i]], parts.vertices[on_side[i]]);
                    let cylinder = parts.surfaces[blend].clone();
                    let between = section(&cylinder, p, third_normal, from, to, tolerance, parts)
                        .ok_or(BlendDeclined::UnsupportedVertex { vertex: end.vertex })?;
                    vertices.extend(between);
                }
                pair(blend, body.faces()[end.third].surface)
            }
            None => {
                let prev = (i + count - 1) % count;
                if kind == Kind::Fillet {
                    let c = inward(i, p).ok_or(unsupported.clone())?;
                    let o = v3::add(p, v3::scale(v3::add(c, down), size));
                    let n = arc_pieces(size, FRAC_PI_2, tolerance);
                    for k in 1..n {
                        let (s, co) = (FRAC_PI_2 * k as f64 / n as f64).sin_cos();
                        let off = v3::add(v3::scale(down, -co), v3::scale(c, -s));
                        vertices.push(parts.vertex(v3::add(o, v3::scale(off, size))));
                    }
                }
                pair(blend_surfaces[prev], blend_surfaces[i])
            }
        };
        vertices.push(on_side[i]);
        let edge = Edge {
            surfaces,
            vertices,
            closed: false,
        };
        // A used-up edge already lies on the third face, where the end goes:
        // it becomes the end, and leaves the face it bounded with the third.
        if let Some(&(_, slot, e)) = consumed.iter().find(|c| c.0 == i) {
            parts.edges[e] = edge;
            seams.push(e);
            let gone = if slot == 0 {
                plan.cap
            } else {
                side_of(edge_at(i))
            };
            parts.faces[gone].edges.retain(|&x| x != e);
            continue;
        }
        parts.edges.push(edge);
        seams.push(parts.edges.len() - 1);
        if let Some(end) = end_at(i) {
            thirds.push((end.third, parts.edges.len() - 1));
        }
    }

    // Risers lose the stretch the blend took; so do the edges a sharp end
    // leaves by, on the cap and down the side.
    for (i, r) in plan.risers.iter().enumerate() {
        let Some(r) = *r else { continue };
        let v = vertex_at(i);
        for slot in parts.edges[r].vertices.iter_mut() {
            if *slot == v {
                *slot = on_side[i];
            }
        }
    }
    for i in [0, count] {
        let Some(end) = end_at(i) else { continue };
        for (e, to) in [(end.on_cap, on_cap[i]), (end.on_side, on_side[i])] {
            if consumed.iter().any(|c| c.0 == i && c.2 == e) {
                continue;
            }
            for slot in parts.edges[e].vertices.iter_mut() {
                if *slot == end.vertex {
                    *slot = to;
                }
            }
        }
    }
    for &(third, e) in &thirds {
        parts.faces[third].edges.push(e);
    }

    // Each chain edge becomes its blend's boundary on the cap; its boundary
    // on the side is new.
    let cap_surface = body.faces()[plan.cap].surface;
    let mut fresh = Vec::with_capacity(count);
    let mut sides = Vec::with_capacity(count);
    for i in 0..count {
        let (edge, _) = plan.edges[i];
        let side = side_of(i);
        let side_surface = body.faces()[side].surface;
        parts.edges[edge] = Edge {
            surfaces: pair(cap_surface, blend_surfaces[i]),
            vertices: cap_rails[i].clone(),
            closed: false,
        };
        parts.edges.push(Edge {
            surfaces: pair(side_surface, blend_surfaces[i]),
            vertices: side_rails[i].clone(),
            closed: false,
        });
        let rail = parts.edges.len() - 1;
        for slot in parts.faces[side].edges.iter_mut() {
            if *slot == edge {
                *slot = rail;
            }
        }
        sides.push(side);
        fresh.push(parts.faces.len());
        let curved = !matches!(parts.surfaces[blend_surfaces[i]], Surface::Plane { .. });
        let flipped = match kind {
            Kind::Fillet => convex < 0.0,
            // A cone faces away from its axis: inwards where the side is a
            // bore.
            Kind::Chamfer => curved && body.faces()[side].flipped,
        };
        parts.faces.push(Face {
            surface: blend_surfaces[i],
            edges: vec![edge, rail, seams[i], seams[next(i)]],
            flipped,
            u_range: (0.0, 0.0),
            v_range: (0.0, 0.0),
            u_wraps: false,
            v_wraps: false,
            loops: None,
        });
    }

    // Faces bounded afresh, their rings read off the edges.
    let mut flat = vec![plan.cap];
    flat.extend(thirds.iter().map(|t| t.0));
    if let Some(ends) = &plan.ends {
        flat.extend(ends.iter().map(|e| e.third));
    }
    flat.sort_unstable();
    flat.dedup();
    for &f in &flat {
        parts.faces[f].loops = None;
    }
    for &s in &sides {
        parts.faces[s].loops = None;
    }
    let probe = parts.body();
    for &f in &flat {
        if !probe.face_loops(f).iter().any(|l| l.is_outer()) {
            return Err(BlendDeclined::TrimmedFace { face: f });
        }
    }
    for &f in sides.iter().chain(&fresh) {
        let loops = probe.face_loops(f);
        let Some(outer) = loops.iter().find(|l| l.is_outer()) else {
            return Err(BlendDeclined::TrimmedFace { face: f });
        };
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for c in &outer.uv {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let face = &mut parts.faces[f];
        face.u_range = (lo[0], hi[0]);
        face.v_range = (lo[1], hi[1]);
        if !matches!(parts.surfaces[face.surface], Surface::Plane { .. }) {
            face.loops = Some(loops);
        }
    }
    Ok(())
}
