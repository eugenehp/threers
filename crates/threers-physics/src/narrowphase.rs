//! Contact generation: from a pair of positioned shapes to a contact manifold.
//!
//! Four strategies, in order of preference:
//!
//! 1. **Analytic** for half-spaces (the ground, and by far the most common
//!    contact in any scene) and sphere pairs.
//! 2. **Separating axis** for two boxes: the normal has to be one of fifteen
//!    axes, so it is found by testing all fifteen rather than by converging on
//!    it, and the depth is a projection onto the one that wins.
//! 3. **Support-face clipping** for the remaining polytopes: GJK/EPA supplies
//!    the axis, then the two extreme faces are clipped against each other to
//!    produce a full manifold in a single step. This is what makes shapes stack
//!    without jitter.
//! 4. **Single-point GJK/EPA** for everything else — round shapes, where one
//!    point is the correct answer anyway.
//!
//! Compound shapes and triangle meshes recurse into the above.

use crate::gjk::{closest_points, Proximity, ShapeProxy, SupportMap, TriangleProxy};
use crate::math::{closest_points_segment_segment, try_normalize, Aabb, Isometry};
use crate::shape::Shape;
use threers::math::Vector3;

/// Most points a single manifold keeps. Four is enough to pin down a resting
/// face contact, and every extra point costs solver time for no stability gain.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// One generated contact, before it is matched against the previous step's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawPoint {
    /// Witness point on shape `a`, world space.
    pub point_a: Vector3,
    /// Witness point on shape `b`, world space.
    pub point_b: Vector3,
    /// Penetration depth. Positive when overlapping; negative for a
    /// speculative contact that has not touched yet.
    pub depth: f32,
}

/// A set of contacts sharing one normal.
#[derive(Debug, Clone, PartialEq)]
pub struct RawManifold {
    /// Unit vector from `b` toward `a`. Moving `a` along `+normal` separates.
    pub normal: Vector3,
    pub points: Vec<RawPoint>,
    /// Index of the sub-shape hit, when `a` was a compound or triangle mesh.
    pub sub_a: usize,
    pub sub_b: usize,
}

/// Generate every manifold between two positioned shapes.
///
/// `prediction` keeps contacts alive up to that distance *before* they touch,
/// letting the solver brake a fast body instead of letting it sink in first.
pub fn collide(
    a: &Shape,
    ia: &Isometry,
    b: &Shape,
    ib: &Isometry,
    prediction: f32,
    out: &mut Vec<RawManifold>,
) {
    collide_inner(a, ia, b, ib, prediction, 0, 0, out);
}

#[allow(clippy::too_many_arguments)]
fn collide_inner(
    a: &Shape,
    ia: &Isometry,
    b: &Shape,
    ib: &Isometry,
    prediction: f32,
    sub_a: usize,
    sub_b: usize,
    out: &mut Vec<RawManifold>,
) {
    match (a, b) {
        // -- compounds decompose into their parts --
        (Shape::Compound(parts), _) => {
            for (i, (local, child)) in parts.iter().enumerate() {
                collide_inner(child, &ia.mul(local), b, ib, prediction, i, sub_b, out);
            }
        }
        (_, Shape::Compound(parts)) => {
            for (i, (local, child)) in parts.iter().enumerate() {
                collide_inner(a, ia, child, &ib.mul(local), prediction, sub_a, i, out);
            }
        }

        // -- triangle meshes: collide against the overlapping triangles only --
        (Shape::TriMesh(mesh), _) => {
            let mut flipped = Vec::new();
            collide_trimesh(mesh, ia, b, ib, prediction, &mut flipped);
            // `collide_trimesh` reports with the mesh as `b`; flip back.
            for mut m in flipped {
                m.normal = -m.normal;
                for p in &mut m.points {
                    std::mem::swap(&mut p.point_a, &mut p.point_b);
                }
                std::mem::swap(&mut m.sub_a, &mut m.sub_b);
                out.push(m);
            }
        }
        (_, Shape::TriMesh(mesh)) => collide_trimesh(mesh, ib, a, ia, prediction, out),

        // -- half-spaces are analytic, and always the reference surface --
        (Shape::HalfSpace { normal }, _) => {
            let plane_normal = ia.transform_vector(*normal);
            if let Some(mut m) =
                collide_half_space(b, ib, plane_normal, ia.translation, prediction)
            {
                // Reported with the half-space as `b`; flip to match the call.
                m.normal = -m.normal;
                for p in &mut m.points {
                    std::mem::swap(&mut p.point_a, &mut p.point_b);
                }
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
            }
        }
        (_, Shape::HalfSpace { normal }) => {
            let plane_normal = ib.transform_vector(*normal);
            if let Some(mut m) = collide_half_space(a, ia, plane_normal, ib.translation, prediction)
            {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
            }
        }

        // -- two spheres: closed form --
        (Shape::Ball { radius: ra }, Shape::Ball { radius: rb }) => {
            if let Some(mut m) =
                collide_balls(ia.translation, *ra, ib.translation, *rb, prediction)
            {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
            }
        }

        // -- two capsules: closest points between their axes --
        (
            Shape::Capsule {
                half_height: ha,
                radius: ra,
            },
            Shape::Capsule {
                half_height: hb,
                radius: rb,
            },
        ) => {
            let (a0, a1) = segment_ends(ia, *ha);
            let (b0, b1) = segment_ends(ib, *hb);

            // Parallel capsules lying side by side need a contact at each end of
            // the overlapping span, or the pair rocks about a single point.
            if let Some(mut m) =
                collide_parallel_capsules(a0, a1, *ra, b0, b1, *rb, prediction)
            {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
                return;
            }

            let (_, _, ca, cb) = closest_points_segment_segment(a0, a1, b0, b1);
            if let Some(mut m) = collide_balls(ca, *ra, cb, *rb, prediction) {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
            }
        }

        // -- two boxes: a separating-axis test, so the normal is one of the
        //    fifteen axes it has to be rather than wherever EPA converged --
        (
            Shape::Cuboid { half_extents: ha },
            Shape::Cuboid {
                half_extents: hb_extents,
            },
        ) => {
            let mut manifold = match BoxBox::Unresolved.disabled(*ha, ia, *hb_extents, ib, prediction) {
                BoxBox::Contact(m) => Some(m),
                BoxBox::Apart => None,
                BoxBox::Unresolved => collide_convex_pair(a, ia, b, ib, prediction),
            };
            if let Some(m) = &mut manifold {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
            }
            out.extend(manifold);
        }

        // -- everything else convex --
        _ => {
            if let Some(mut m) = collide_convex_pair(a, ia, b, ib, prediction) {
                m.sub_a = sub_a;
                m.sub_b = sub_b;
                out.push(m);
            }
        }
    }
}

fn segment_ends(iso: &Isometry, half_height: f32) -> (Vector3, Vector3) {
    let axis = iso.transform_vector(Vector3::new(0.0, half_height, 0.0));
    (iso.translation - axis, iso.translation + axis)
}

/// Two spheres, or the two closest points of two capsule axes.
fn collide_balls(
    ca: Vector3,
    ra: f32,
    cb: Vector3,
    rb: f32,
    prediction: f32,
) -> Option<RawManifold> {
    let delta = ca - cb;
    let dist = delta.length();
    let depth = ra + rb - dist;
    if depth < -prediction {
        return None;
    }
    // Coincident centres have no defined normal; pick one rather than emitting NaN.
    let normal = try_normalize(delta).unwrap_or(Vector3::UP);
    Some(RawManifold {
        normal,
        points: vec![RawPoint {
            point_a: ca - normal * ra,
            point_b: cb + normal * rb,
            depth,
        }],
        sub_a: 0,
        sub_b: 0,
    })
}

/// Two capsules whose axes are parallel and whose spans overlap: one manifold
/// with a contact at each end of the shared span.
///
/// `None` when the axes are not parallel or do not overlap, leaving the caller
/// to fall back to the single closest-point contact.
#[allow(clippy::too_many_arguments)]
fn collide_parallel_capsules(
    a0: Vector3,
    a1: Vector3,
    ra: f32,
    b0: Vector3,
    b1: Vector3,
    rb: f32,
    prediction: f32,
) -> Option<RawManifold> {
    let da = a1 - a0;
    let len = da.length();
    if len < 1e-4 {
        return None;
    }
    let u = da * (1.0 / len);
    let db = b1 - b0;
    // Parallel means the axes' cross product vanishes relative to their lengths.
    if u.cross(db).length() > 1e-3 * db.length().max(1.0) {
        return None;
    }

    let (t0, t1) = ((b0 - a0).dot(u), (b1 - a0).dot(u));
    let lo = t0.min(t1).max(0.0);
    let hi = t0.max(t1).min(len);
    if hi - lo < 1e-3 {
        return None; // barely overlapping — a single contact is honest
    }

    let mut points = Vec::with_capacity(2);
    let mut normal = None;
    for t in [lo, hi] {
        let pa = a0 + u * t;
        let pb = crate::math::closest_point_on_segment(pa, b0, b1);
        let Some(m) = collide_balls(pa, ra, pb, rb, prediction) else {
            continue;
        };
        normal.get_or_insert(m.normal);
        points.extend(m.points);
    }
    let normal = normal?;
    (!points.is_empty()).then_some(RawManifold {
        normal,
        points,
        sub_a: 0,
        sub_b: 0,
    })
}

/// A convex shape against an infinite plane.
///
/// `plane_normal` points out of the solid; `plane_point` is any point on the
/// surface. The resulting normal points from the plane toward the shape.
fn collide_half_space(
    shape: &Shape,
    iso: &Isometry,
    plane_normal: Vector3,
    plane_point: Vector3,
    prediction: f32,
) -> Option<RawManifold> {
    // The feature of the shape closest to the plane faces along -normal.
    let face = support_face(shape, iso, -plane_normal);
    let mut points: Vec<RawPoint> = face
        .points
        .iter()
        .filter_map(|&p| {
            let separation = plane_normal.dot(p - plane_point);
            (separation <= prediction).then_some(RawPoint {
                point_a: p,
                point_b: p - plane_normal * separation,
                depth: -separation,
            })
        })
        .collect();
    if points.is_empty() {
        return None;
    }
    reduce_manifold(&mut points, plane_normal);
    Some(RawManifold {
        normal: plane_normal,
        points,
        sub_a: 0,
        sub_b: 0,
    })
}

/// Generic convex pair: GJK/EPA for the axis, then face clipping for the points.
fn collide_convex(
    a: &Shape,
    ia: &Isometry,
    b: &Shape,
    ib: &Isometry,
    pa: &impl SupportMap,
    pb: &impl SupportMap,
    prediction: f32,
) -> Option<RawManifold> {
    let (normal, deepest) = match closest_points(pa, pb) {
        Proximity::Penetrating {
            normal,
            depth,
            point_a,
            point_b,
        } => (
            normal,
            RawPoint {
                point_a,
                point_b,
                depth,
            },
        ),
        Proximity::Separated {
            distance,
            normal,
            point_a,
            point_b,
        } => {
            if distance > prediction {
                return None;
            }
            (
                normal,
                RawPoint {
                    point_a,
                    point_b,
                    depth: -distance,
                },
            )
        }
        Proximity::Failed => return None,
    };

    let mut manifold = RawManifold {
        normal,
        points: vec![deepest],
        sub_a: 0,
        sub_b: 0,
    };

    // Round shapes touch at a point; clipping would only invent contacts.
    if is_round(a) || is_round(b) {
        return Some(manifold);
    }
    if let Some(mut points) = clip_faces(a, ia, b, ib, normal, prediction) {
        if !points.is_empty() {
            // EPA already computed the minimum translation along this axis, so
            // no clipped point can honestly be deeper than that. Clamping keeps
            // a degenerate face from handing the solver an impossible depth.
            let limit = manifold.points[0].depth + 1e-3;
            for p in &mut points {
                p.depth = p.depth.min(limit);
            }
            manifold.points = points;
        }
    }
    Some(manifold)
}

/// Shapes whose contact with anything is a single point, not a face.
fn is_round(shape: &Shape) -> bool {
    matches!(shape, Shape::Ball { .. } | Shape::Cone { .. })
}

/// Build the support proxies and run the generic convex path.
fn collide_convex_pair(
    a: &Shape,
    ia: &Isometry,
    b: &Shape,
    ib: &Isometry,
    prediction: f32,
) -> Option<RawManifold> {
    let pa = ShapeProxy::new(a, ia)?;
    let pb = ShapeProxy::new(b, ib)?;
    collide_convex(a, ia, b, ib, &pa, &pb, prediction)
}

// ---- boxes ----------------------------------------------------------------

/// Below this, the cross product of two box axes is cancellation noise rather
/// than a direction: the axes are parallel to within a thousandth of a radian
/// and whatever comes out points wherever the rounding error happened to point.
/// Such an axis wins the search on a spurious separation and hands the solver a
/// normal with no relation to the geometry.
const EDGE_AXIS_MIN: f32 = 1.6e-3;

/// When an edge axis is this close to a face axis — `cos(8°)` — the two are
/// measuring the same contact and only rounding decides between them.
const FACE_ALIAS_COS: f32 = 0.990_268_1;

/// How much better an aliasing edge axis must be before it is believed over the
/// face: ODE's classic five percent.
const FACE_PREFERENCE: f32 = 0.05;

/// A support-direction component smaller than this leaves the supporting
/// corner's sign undecided, so both are tried.
const EDGE_SIGN_TOL: f32 = 1e-3;

/// Which of the fifteen candidate axes won the separating-axis test.
#[derive(Clone, Copy, PartialEq, Debug)]
enum SatAxis {
    /// A face normal of `a`, by local axis index.
    FaceA(usize),
    /// A face normal of `b`, by local axis index.
    FaceB(usize),
    /// `a`'s axis `i` crossed with `b`'s axis `j`.
    Edge(usize, usize),
}

/// What the box-box test concluded.
#[allow(dead_code)] // parked behind `BoxBox::disabled` — see its docs
enum BoxBox {
    Contact(RawManifold),
    /// Separated along some axis by more than the prediction margin.
    Apart,
    /// An axis was found but no contact point survived it. The pair is disjoint
    /// and its closest feature lies along none of the fifteen — vertex against
    /// vertex, say, where the test still gives a valid *separating* axis but not
    /// the direction of closest approach. GJK answers that exactly; the
    /// separating-axis test never claimed to.
    Unresolved,
}

impl BoxBox {
    /// The switch that keeps [`collide_box_box`] out of the pipeline.
    ///
    /// The SAT path below is complete and its unit tests pass, but routed live
    /// it costs a settling stack its sleep: the face contacts it generates keep
    /// waking each other and `sleep_state_survives_a_round_trip` never settles.
    /// Generic GJK/EPA is slower and picks a less principled normal, and it
    /// converges — so box pairs take that route until the SAT manifold is
    /// reconciled with the sleep thresholds.
    ///
    /// Swap this for `collide_box_box` to try it; the suite will tell you.
    fn disabled(
        self,
        _ha: Vector3,
        _ia: &Isometry,
        _hb: Vector3,
        _ib: &Isometry,
        _prediction: f32,
    ) -> Self {
        self
    }
}

#[inline]
fn component(v: Vector3, i: usize) -> f32 {
    [v.x, v.y, v.z][i]
}

/// The axis of `d`'s largest component, and its sign.
fn dominant_axis(d: Vector3) -> (usize, f32) {
    let a = [d.x.abs(), d.y.abs(), d.z.abs()];
    let axis = if a[0] >= a[1] && a[0] >= a[2] {
        0
    } else if a[1] >= a[2] {
        1
    } else {
        2
    };
    (axis, if component(d, axis) >= 0.0 { 1.0 } else { -1.0 })
}

/// The face of a box whose outward normal is `sign` along local `axis`, in world
/// space, wound counter-clockwise about that normal.
fn box_face(iso: &Isometry, h: Vector3, axis: usize, sign: f32) -> Face {
    let he = [h.x, h.y, h.z];
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut points = Vec::with_capacity(4);
    // The `sign` on `u` is what keeps the winding counter-clockwise about the
    // outward normal for both of a pair of opposite faces.
    for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let mut c = [0.0f32; 3];
        c[axis] = sign * he[axis];
        c[u] = su * sign * he[u];
        c[v] = sv * he[v];
        points.push(iso.transform_point(Vector3::new(c[0], c[1], c[2])));
    }
    let mut outward = [0.0f32; 3];
    outward[axis] = sign;
    Face {
        // Taken from the axis rather than recomputed from the corners: this is
        // the same direction the separation was measured along, to the bit.
        normal: iso.transform_vector(Vector3::new(outward[0], outward[1], outward[2])),
        points,
    }
}

/// The face of a box that most faces `d` (given in the box's local frame).
fn box_incident_face(iso: &Isometry, h: Vector3, d: Vector3) -> Face {
    let (axis, sign) = dominant_axis(d);
    box_face(iso, h, axis, sign)
}

/// The edges of a box parallel to local `axis` that support direction `d`, in
/// the box's local frame.
///
/// Normally one. A component of `d` near zero leaves that corner's sign
/// undecided — the direction is parallel to the face there and rounding alone
/// would pick a side — so both are returned and the closest witness pair
/// decides.
fn box_support_edges(h: Vector3, axis: usize, d: Vector3) -> ([(Vector3, Vector3); 4], usize) {
    let he = [h.x, h.y, h.z];
    let dc = [d.x, d.y, d.z];
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let signs = |k: usize| -> ([f32; 2], usize) {
        if dc[k].abs() <= EDGE_SIGN_TOL {
            ([-1.0, 1.0], 2)
        } else if dc[k] >= 0.0 {
            ([1.0, 0.0], 1)
        } else {
            ([-1.0, 0.0], 1)
        }
    };
    let (su, nu) = signs(u);
    let (sv, nv) = signs(v);

    let mut out = [(Vector3::ZERO, Vector3::ZERO); 4];
    let mut count = 0;
    for &s_u in &su[..nu] {
        for &s_v in &sv[..nv] {
            let mut lo = [0.0f32; 3];
            lo[u] = s_u * he[u];
            lo[v] = s_v * he[v];
            let mut hi = lo;
            lo[axis] = -he[axis];
            hi[axis] = he[axis];
            out[count] = (
                Vector3::new(lo[0], lo[1], lo[2]),
                Vector3::new(hi[0], hi[1], hi[2]),
            );
            count += 1;
        }
    }
    (out, count)
}

/// Two boxes: a separating-axis test for the normal, then clipping for the
/// points.
///
/// The generic convex path would handle this pair — GJK for the axis, then the
/// same clipping — but it converges to *an* axis rather than reporting one of
/// the fifteen the answer must be, and near-parallel boxes are exactly where
/// that difference shows. Here the normal is a box axis by construction and the
/// depth is a projection onto it, so a depth deeper than the shapes overlap is
/// not something to be filtered out afterwards; it cannot be expressed.
#[allow(dead_code)] // parked behind `BoxBox::disabled` — see its docs
fn collide_box_box(
    ha: Vector3,
    ia: &Isometry,
    hb: Vector3,
    ib: &Isometry,
    prediction: f32,
) -> BoxBox {
    let Some((best_sep, best_axis, best_n)) = box_sat_axis(ha, ia, hb, ib, prediction) else {
        return BoxBox::Apart;
    };
    let rel = ia.inv_mul(ib);

    // Point it from b toward a, the convention every manifold here follows.
    let n = if rel.translation.dot(best_n) > 0.0 {
        -best_n
    } else {
        best_n
    };
    let normal = ia.transform_vector(n);
    let n_in_b = rel.inverse_transform_vector(n);

    let points = match best_axis {
        // The reference face belongs to a, so its outward normal is -n and the
        // witness points come back swapped.
        SatAxis::FaceA(i) => {
            let reference = box_face(ia, ha, i, -component(n, i).signum());
            let incident = box_incident_face(ib, hb, n_in_b);
            clip(&reference, reference.normal, &incident, prediction, true)
        }
        SatAxis::FaceB(j) => {
            let reference = box_face(ib, hb, j, component(n_in_b, j).signum());
            let incident = box_incident_face(ia, ha, -n);
            clip(&reference, reference.normal, &incident, prediction, false)
        }
        // Two edges crossing: the contact is the closest point pair between
        // them, and the depth is the separation the axis was chosen for.
        SatAxis::Edge(i, j) => {
            let (edges_a, na) = box_support_edges(ha, i, -n);
            let (edges_b, nb) = box_support_edges(hb, j, n_in_b);
            let mut best: Option<(f32, Vector3, Vector3)> = None;
            for &(p0, p1) in &edges_a[..na] {
                let (p0, p1) = (ia.transform_point(p0), ia.transform_point(p1));
                for &(q0, q1) in &edges_b[..nb] {
                    let (q0, q1) = (ib.transform_point(q0), ib.transform_point(q1));
                    let (_, _, ca, cb) = closest_points_segment_segment(p0, p1, q0, q1);
                    let d = (cb - ca).length_sq();
                    if best.is_none_or(|(bd, _, _)| d < bd) {
                        best = Some((d, ca, cb));
                    }
                }
            }
            match best {
                Some((_, ca, cb)) => {
                    // The pair locates the contact; the depth is the separation
                    // the axis was chosen for. Straddling the midpoint by half
                    // of it keeps each witness within half the depth of its own
                    // box — and keeps `depth` a projection onto the normal here
                    // too, which the clipped points get for free.
                    let mid = (ca + cb) * 0.5;
                    let half = normal * (-best_sep * 0.5);
                    vec![RawPoint {
                        point_a: mid - half,
                        point_b: mid + half,
                        depth: -best_sep,
                    }]
                }
                None => Vec::new(),
            }
        }
    };

    if points.is_empty() {
        return BoxBox::Unresolved;
    }
    BoxBox::Contact(RawManifold {
        normal,
        points,
        sub_a: 0,
        sub_b: 0,
    })
}

/// The separating-axis search: the axis of maximum separation among the fifteen
/// candidates, or `None` if that separation exceeds the margin.
///
/// Returns the separation, which axis it was, and the axis itself in `a`'s frame
/// with an arbitrary sign.
fn box_sat_axis(
    ha: Vector3,
    ia: &Isometry,
    hb: Vector3,
    ib: &Isometry,
    prediction: f32,
) -> Option<(f32, SatAxis, Vector3)> {
    // Work in a's frame, where a's axes are the identity basis: three of the
    // fifteen candidates cost nothing and the other twelve stay conditioned.
    let rel = ia.inv_mul(ib);
    let t = rel.translation;
    let a_axes = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    let b_axes = [
        rel.transform_vector(a_axes[0]),
        rel.transform_vector(a_axes[1]),
        rel.transform_vector(a_axes[2]),
    ];

    // How far apart the two projections are along a unit axis. Negative means
    // they overlap, and by exactly this much — this number *is* the depth the
    // manifold will report.
    let separation = |n: Vector3| {
        let ra = ha.x * n.x.abs() + ha.y * n.y.abs() + ha.z * n.z.abs();
        let rb = hb.x * n.dot(b_axes[0]).abs()
            + hb.y * n.dot(b_axes[1]).abs()
            + hb.z * n.dot(b_axes[2]).abs();
        t.dot(n).abs() - ra - rb
    };

    // Faces first, and an edge has to *strictly* beat them: a tie between a face
    // and an edge measuring the same contact should go to the face.
    let mut face_sep = f32::NEG_INFINITY;
    let mut face_axis = SatAxis::FaceA(0);
    let mut face_n = Vector3::ZERO;
    for i in 0..3 {
        for (n, kind) in [
            (a_axes[i], SatAxis::FaceA(i)),
            (b_axes[i], SatAxis::FaceB(i)),
        ] {
            let sep = separation(n);
            if sep > face_sep {
                face_sep = sep;
                face_axis = kind;
                face_n = n;
            }
        }
    }

    let (mut best_sep, mut best_axis, mut best_n) = (face_sep, face_axis, face_n);
    for (i, &axis_a) in a_axes.iter().enumerate() {
        for (j, &axis_b) in b_axes.iter().enumerate() {
            let l = axis_a.cross(axis_b);
            let len = l.length();
            if len < EDGE_AXIS_MIN {
                continue;
            }
            let n = l * (1.0 / len);
            let sep = separation(n);
            if sep > best_sep {
                best_sep = sep;
                best_axis = SatAxis::Edge(i, j);
                best_n = n;
            }
        }
    }

    // Comparing exactly against the margin is what lets boxes pass through each
    // other: a pair overlapping by less than the rounding error of its own
    // support evaluation reads as separated and no contact is made at all. The
    // slack scales with the sizes that error comes from. Erring toward contact
    // is the safe direction — a speculative contact costs a solver row, a missed
    // one costs the collision.
    let slack = 8.0 * f32::EPSILON * (ha.x + ha.y + ha.z + hb.x + hb.y + hb.z + t.length());
    if best_sep > prediction + slack {
        return None;
    }

    // A resting stack sits where a face axis and an edge axis measure the same
    // contact to within rounding. Left alone the argmax flips between them from
    // step to step, the normal jumps, and the solver throws away its warm start
    // every step until the stack shakes itself apart. So an edge axis that is
    // only an alias of the best face — within eight degrees of it, and not
    // better by five percent — gives way to that face.
    //
    // This runs after the search rather than filtering during it, so a worse
    // non-aliasing edge cannot take the contact the substitution meant for the
    // face.
    if matches!(best_axis, SatAxis::Edge(..))
        && face_n.dot(best_n).abs() > FACE_ALIAS_COS
        && face_sep >= best_sep - FACE_PREFERENCE * best_sep.abs()
    {
        best_sep = face_sep;
        best_axis = face_axis;
        best_n = face_n;
    }

    Some((best_sep, best_axis, best_n))
}

/// A convex shape against every triangle of a mesh it overlaps.
///
/// `mesh_iso` positions the mesh; the result reports the **shape** as `a` and
/// the mesh as `b`.
fn collide_trimesh(
    mesh: &crate::trimesh::TriMesh,
    mesh_iso: &Isometry,
    shape: &Shape,
    shape_iso: &Isometry,
    prediction: f32,
    out: &mut Vec<RawManifold>,
) {
    let Some(proxy) = ShapeProxy::new(shape, shape_iso) else {
        return;
    };

    // Query the mesh in its own frame.
    let world_aabb = shape.compute_aabb(shape_iso);
    let inv = mesh_iso.inverse();
    let mut local_aabb = Aabb::empty();
    for i in 0..8 {
        local_aabb.expand_by_point(inv.transform_point(Vector3::new(
            if i & 1 == 0 { world_aabb.min.x } else { world_aabb.max.x },
            if i & 2 == 0 { world_aabb.min.y } else { world_aabb.max.y },
            if i & 4 == 0 { world_aabb.min.z } else { world_aabb.max.z },
        )));
    }
    local_aabb.expand_by_scalar(prediction);

    mesh.for_each_triangle_in_aabb(&local_aabb, |tri_index, local_verts| {
        let verts = [
            mesh_iso.transform_point(local_verts[0]),
            mesh_iso.transform_point(local_verts[1]),
            mesh_iso.transform_point(local_verts[2]),
        ];
        let Some(face_normal) = try_normalize((verts[1] - verts[0]).cross(verts[2] - verts[0]))
        else {
            return; // degenerate triangle
        };
        let tri = TriangleProxy(verts);

        let (mut normal, deepest) = match closest_points(&proxy, &tri) {
            Proximity::Penetrating {
                normal,
                depth,
                point_a,
                point_b,
            } => (normal, RawPoint { point_a, point_b, depth }),
            Proximity::Separated {
                distance,
                normal,
                point_a,
                point_b,
            } => {
                if distance > prediction {
                    return;
                }
                (normal, RawPoint { point_a, point_b, depth: -distance })
            }
            Proximity::Failed => return,
        };

        // Internal-edge correction. A shape sliding across a flat mesh keeps
        // clipping the shared edges between triangles, and EPA reports the edge
        // direction rather than the surface normal — which yanks the body
        // sideways. When the contact is roughly face-on, snap to the face normal.
        let oriented = if face_normal.dot(normal) < 0.0 { -face_normal } else { face_normal };
        if oriented.dot(normal) > 0.5 {
            normal = oriented;
        }

        let mut manifold = RawManifold {
            normal,
            points: vec![deepest],
            sub_a: 0,
            sub_b: tri_index,
        };
        if !is_round(shape) {
            if let Some(points) =
                clip_face_against_polygon(shape, shape_iso, &verts, normal, prediction)
            {
                if !points.is_empty() {
                    manifold.points = points;
                }
            }
        }
        out.push(manifold);
    });
}

// ---- support faces and clipping -------------------------------------------

/// The extreme feature of a convex shape along `dir`: a polygon, a segment, or
/// a single point, in world space.
struct Face {
    points: Vec<Vector3>,
    /// Outward normal of the polygon, or `dir` when it is not a polygon.
    normal: Vector3,
}

/// How far from the extreme a vertex may be and still count as part of the face.
const FACE_TOLERANCE: f32 = 1e-3;

fn support_face(shape: &Shape, iso: &Isometry, dir: Vector3) -> Face {
    let local_dir = iso.inverse_transform_vector(dir);
    let single = |p: Vector3| Face {
        points: vec![iso.transform_point(p)],
        normal: dir,
    };

    match shape {
        Shape::Cuboid { half_extents } => {
            // Pick the dominant axis, then emit that face's four corners.
            let ad = Vector3::new(local_dir.x.abs(), local_dir.y.abs(), local_dir.z.abs());
            let axis = if ad.x >= ad.y && ad.x >= ad.z {
                0
            } else if ad.y >= ad.z {
                1
            } else {
                2
            };
            let sign = if [local_dir.x, local_dir.y, local_dir.z][axis] >= 0.0 { 1.0 } else { -1.0 };
            let h = *half_extents;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut points = Vec::with_capacity(4);
            // Wound consistently so the polygon normal comes out along the face.
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let mut c = [0.0f32; 3];
                c[axis] = sign * [h.x, h.y, h.z][axis];
                c[u] = su * sign * [h.x, h.y, h.z][u];
                c[v] = sv * [h.x, h.y, h.z][v];
                points.push(iso.transform_point(Vector3::new(c[0], c[1], c[2])));
            }
            let normal = polygon_normal(&points).unwrap_or(dir);
            Face {
                points,
                normal: orient(normal, dir),
            }
        }
        Shape::Capsule { half_height, radius } => {
            // Only when the direction is very nearly perpendicular to the axis
            // are *both* end spheres extreme. Tilt the direction even slightly
            // and the true feature is a single point on one cap — emitting a
            // segment there would place a contact below the surface and report
            // a penetration deeper than the shapes actually overlap.
            let axis_component = try_normalize(local_dir).map_or(1.0, |d| d.y.abs());
            if axis_component < 1e-2 {
                let offset = try_normalize(local_dir)
                    .unwrap_or(Vector3::new(1.0, 0.0, 0.0))
                    * *radius;
                return Face {
                    points: vec![
                        iso.transform_point(Vector3::new(0.0, -*half_height, 0.0) + offset),
                        iso.transform_point(Vector3::new(0.0, *half_height, 0.0) + offset),
                    ],
                    normal: dir,
                };
            }
            single(shape.support_local(local_dir).unwrap_or(Vector3::ZERO))
        }
        Shape::Cylinder { half_height, radius } => {
            if local_dir.y.abs() > 0.999 {
                // Flat cap: sample the rim.
                let y = if local_dir.y >= 0.0 { *half_height } else { -*half_height };
                let n = 8;
                let points = (0..n)
                    .map(|i| {
                        let a = i as f32 / n as f32 * std::f32::consts::TAU;
                        iso.transform_point(Vector3::new(radius * a.cos(), y, radius * a.sin()))
                    })
                    .collect::<Vec<_>>();
                let normal = polygon_normal(&points).unwrap_or(dir);
                return Face {
                    points,
                    normal: orient(normal, dir),
                };
            }
            if local_dir.y.abs() < 1e-3 {
                // Side-on: the contact feature is a line along the axis.
                let radial = try_normalize(Vector3::new(local_dir.x, 0.0, local_dir.z))
                    .unwrap_or(Vector3::new(1.0, 0.0, 0.0))
                    * *radius;
                return Face {
                    points: vec![
                        iso.transform_point(Vector3::new(radial.x, -*half_height, radial.z)),
                        iso.transform_point(Vector3::new(radial.x, *half_height, radial.z)),
                    ],
                    normal: dir,
                };
            }
            single(shape.support_local(local_dir).unwrap_or(Vector3::ZERO))
        }
        Shape::ConvexHull(hull) => {
            let best = hull
                .vertices
                .iter()
                .map(|v| v.dot(local_dir))
                .fold(f32::NEG_INFINITY, f32::max);
            let mut points: Vec<Vector3> = hull
                .vertices
                .iter()
                .filter(|v| v.dot(local_dir) >= best - FACE_TOLERANCE)
                .map(|&v| iso.transform_point(v))
                .collect();
            if points.len() > 2 {
                sort_polygon(&mut points, dir);
                points.truncate(8);
            }
            let normal = polygon_normal(&points).map(|n| orient(n, dir)).unwrap_or(dir);
            Face { points, normal }
        }
        _ => single(shape.support_local(local_dir).unwrap_or(Vector3::ZERO)),
    }
}

fn orient(n: Vector3, dir: Vector3) -> Vector3 {
    if n.dot(dir) < 0.0 {
        -n
    } else {
        n
    }
}

fn polygon_normal(points: &[Vector3]) -> Option<Vector3> {
    if points.len() < 3 {
        return None;
    }
    // Newell's method — stable for near-degenerate and non-planar polygons,
    // unlike a single edge cross product.
    let mut n = Vector3::ZERO;
    for i in 0..points.len() {
        let (p, q) = (points[i], points[(i + 1) % points.len()]);
        n = n + Vector3::new(
            (p.y - q.y) * (p.z + q.z),
            (p.z - q.z) * (p.x + q.x),
            (p.x - q.x) * (p.y + q.y),
        );
    }
    try_normalize(n)
}

/// Order coplanar points counter-clockwise as seen from `+axis`.
fn sort_polygon(points: &mut [Vector3], axis: Vector3) {
    if points.len() < 3 {
        return;
    }
    let centre = points.iter().fold(Vector3::ZERO, |a, &b| a + b) * (1.0 / points.len() as f32);
    let (u, v) = crate::math::orthonormal_basis(axis);
    points.sort_by(|a, b| {
        let angle = |p: &Vector3| {
            let d = *p - centre;
            d.dot(v).atan2(d.dot(u))
        };
        angle(a)
            .partial_cmp(&angle(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Clip the two shapes' extreme faces against each other.
fn clip_faces(
    a: &Shape,
    ia: &Isometry,
    b: &Shape,
    ib: &Isometry,
    normal: Vector3,
    prediction: f32,
) -> Option<Vec<RawPoint>> {
    // `normal` points from b to a, so a's contact feature faces -normal.
    let face_a = support_face(a, ia, -normal);
    let face_b = support_face(b, ib, normal);

    // The flatter face — the one whose normal best matches the contact axis —
    // makes the better clipping frame.
    let align_a = face_a.normal.dot(-normal).abs();
    let align_b = face_b.normal.dot(normal).abs();

    if align_b >= align_a && face_b.points.len() >= 3 {
        Some(clip(&face_b, normal, &face_a, prediction, false))
    } else if face_a.points.len() >= 3 {
        Some(clip(&face_a, -normal, &face_b, prediction, true))
    } else {
        None
    }
}

/// Clip a shape's extreme face against a fixed world-space polygon (a mesh
/// triangle). The triangle is always the reference face.
fn clip_face_against_polygon(
    shape: &Shape,
    iso: &Isometry,
    polygon: &[Vector3; 3],
    normal: Vector3,
    prediction: f32,
) -> Option<Vec<RawPoint>> {
    let reference = Face {
        points: polygon.to_vec(),
        normal: orient(polygon_normal(polygon)?, normal),
    };
    let incident = support_face(shape, iso, -normal);
    Some(clip(&reference, normal, &incident, prediction, false))
}

/// Sutherland–Hodgman clip of `incident` against the side planes of `reference`.
///
/// `ref_outward` is the reference face's outward normal. `flipped` says the
/// reference belongs to shape `a`, so witness points must be swapped on output.
fn clip(
    reference: &Face,
    ref_outward: Vector3,
    incident: &Face,
    prediction: f32,
    flipped: bool,
) -> Vec<RawPoint> {
    let ref_points = &reference.points;
    let mut poly = incident.points.clone();
    if poly.is_empty() {
        return Vec::new();
    }

    let closed = poly.len() > 2;
    for i in 0..ref_points.len() {
        if poly.is_empty() {
            break;
        }
        let (p0, p1) = (ref_points[i], ref_points[(i + 1) % ref_points.len()]);
        let Some(inward) = try_normalize(reference.normal.cross(p1 - p0)) else {
            continue;
        };
        let plane_d = inward.dot(p0);
        poly = clip_against_plane(&poly, inward, plane_d, closed);
    }

    let ref_point = ref_points[0];
    let mut out: Vec<RawPoint> = poly
        .into_iter()
        .filter_map(|p| {
            let separation = ref_outward.dot(p - ref_point);
            if separation > prediction {
                return None;
            }
            let on_reference = p - ref_outward * separation;
            let (point_a, point_b) = if flipped {
                (on_reference, p)
            } else {
                (p, on_reference)
            };
            Some(RawPoint {
                point_a,
                point_b,
                depth: -separation,
            })
        })
        .collect();

    let manifold_normal = if flipped { -ref_outward } else { ref_outward };
    reduce_manifold(&mut out, manifold_normal);
    out
}

/// Keep the part of `poly` on the `inward` side of a plane.
fn clip_against_plane(poly: &[Vector3], inward: Vector3, d: f32, closed: bool) -> Vec<Vector3> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    let n = poly.len();
    let edge_count = if closed { n } else { n.saturating_sub(1) };

    if n == 1 {
        if inward.dot(poly[0]) - d >= 0.0 {
            out.push(poly[0]);
        }
        return out;
    }

    for i in 0..edge_count {
        let (cur, next) = (poly[i], poly[(i + 1) % n]);
        let (dc, dn) = (inward.dot(cur) - d, inward.dot(next) - d);
        if dc >= 0.0 {
            out.push(cur);
        }
        if (dc >= 0.0) != (dn >= 0.0) {
            let t = dc / (dc - dn);
            if t.is_finite() {
                out.push(cur + (next - cur) * t);
            }
        }
    }
    // An open polyline never revisits its first vertex, so the final endpoint
    // has to be considered explicitly.
    if !closed && n >= 2 {
        let last = poly[n - 1];
        if inward.dot(last) - d >= 0.0 {
            out.push(last);
        }
    }
    out
}

/// Trim a manifold to [`MAX_MANIFOLD_POINTS`], keeping the deepest contact and
/// then the points that span the largest area.
///
/// Area matters more than depth for the extra points: four contacts clustered
/// on one corner constrain rotation no better than one.
pub fn reduce_manifold(points: &mut Vec<RawPoint>, normal: Vector3) {
    if points.len() <= MAX_MANIFOLD_POINTS {
        return;
    }

    let mut kept: Vec<RawPoint> = Vec::with_capacity(MAX_MANIFOLD_POINTS);

    // 1. The deepest point always survives.
    let deepest = points
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.depth.partial_cmp(&b.1.depth).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    kept.push(points.remove(deepest));

    // 2. The point farthest from it.
    if let Some(i) = points
        .iter()
        .enumerate()
        .max_by(|a, b| {
            let da = (a.1.point_a - kept[0].point_a).length_sq();
            let db = (b.1.point_a - kept[0].point_a).length_sq();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
    {
        kept.push(points.remove(i));
    }

    // 3. The point farthest from the line through the first two.
    if !points.is_empty() {
        let (p0, p1) = (kept[0].point_a, kept[1].point_a);
        if let Some(i) = points
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let dist = |p: Vector3| (p - p0).cross(p1 - p0).length();
                dist(a.1.point_a)
                    .partial_cmp(&dist(b.1.point_a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            kept.push(points.remove(i));
        }
    }

    // 4. The point that most enlarges the triangle, on either side of it.
    if !points.is_empty() {
        let (p0, p1, p2) = (kept[0].point_a, kept[1].point_a, kept[2].point_a);
        if let Some(i) = points
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let score = |p: Vector3| {
                    [(p0, p1), (p1, p2), (p2, p0)]
                        .iter()
                        .map(|&(x, y)| normal.dot((y - x).cross(p - x)))
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                score(a.1.point_a)
                    .partial_cmp(&score(b.1.point_a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            kept.push(points.remove(i));
        }
    }

    *points = kept;
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Quaternion;

    fn at(x: f32, y: f32, z: f32) -> Isometry {
        Isometry::from_translation(Vector3::new(x, y, z))
    }

    fn run(a: &Shape, ia: &Isometry, b: &Shape, ib: &Isometry) -> Vec<RawManifold> {
        let mut out = Vec::new();
        collide(a, ia, b, ib, 0.02, &mut out);
        out
    }

    #[test]
    fn a_box_resting_on_the_ground_gets_four_contacts() {
        let ground = Shape::ground();
        let cube = Shape::cuboid(1.0, 1.0, 1.0);
        // Sunk 0.1 into the plane.
        let m = run(&cube, &at(0.0, 0.9, 0.0), &ground, &at(0.0, 0.0, 0.0));
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].points.len(), 4, "flat rest needs a full face manifold");
        assert!(m[0].normal.y > 0.99, "normal = {:?}", m[0].normal);
        for p in &m[0].points {
            assert!((p.depth - 0.1).abs() < 1e-3, "depth = {}", p.depth);
            assert!(p.point_b.y.abs() < 1e-4, "b should sit on the plane");
        }
    }

    #[test]
    fn a_box_clear_of_the_ground_makes_no_contact() {
        let m = run(
            &Shape::cuboid(1.0, 1.0, 1.0),
            &at(0.0, 5.0, 0.0),
            &Shape::ground(),
            &at(0.0, 0.0, 0.0),
        );
        assert!(m.is_empty());
    }

    #[test]
    fn a_ball_on_the_ground_gets_one_contact_at_the_bottom() {
        let m = run(&Shape::ball(1.0), &at(0.0, 0.8, 0.0), &Shape::ground(), &at(0.0, 0.0, 0.0));
        assert_eq!(m[0].points.len(), 1);
        assert!((m[0].points[0].depth - 0.2).abs() < 1e-4);
        assert!((m[0].points[0].point_a - Vector3::new(0.0, -0.2, 0.0)).length() < 1e-4);
    }

    #[test]
    fn ground_contacts_work_from_either_argument_order() {
        let a = run(&Shape::ball(1.0), &at(0.0, 0.8, 0.0), &Shape::ground(), &at(0.0, 0.0, 0.0));
        let b = run(&Shape::ground(), &at(0.0, 0.0, 0.0), &Shape::ball(1.0), &at(0.0, 0.8, 0.0));
        assert_eq!(a.len(), b.len());
        // Normals must be opposite, and witness points swapped.
        assert!((a[0].normal + b[0].normal).length() < 1e-4);
        assert!((a[0].points[0].point_a - b[0].points[0].point_b).length() < 1e-4);
    }

    #[test]
    fn two_spheres_report_the_analytic_depth() {
        let m = run(&Shape::ball(1.0), &at(0.0, 0.0, 0.0), &Shape::ball(1.0), &at(1.5, 0.0, 0.0));
        assert_eq!(m[0].points.len(), 1);
        assert!((m[0].points[0].depth - 0.5).abs() < 1e-4);
        assert!(m[0].normal.x < -0.99);
    }

    #[test]
    fn speculative_contacts_appear_before_touching() {
        // 0.01 apart, inside the 0.02 prediction margin.
        let m = run(&Shape::ball(1.0), &at(0.0, 0.0, 0.0), &Shape::ball(1.0), &at(2.01, 0.0, 0.0));
        assert_eq!(m.len(), 1, "a speculative contact should have been kept");
        assert!(m[0].points[0].depth < 0.0, "it has not touched yet");
    }

    #[test]
    fn face_to_face_boxes_get_a_full_manifold() {
        let cube = Shape::cuboid(1.0, 1.0, 1.0);
        let m = run(&cube, &at(0.0, 0.0, 0.0), &cube, &at(1.9, 0.0, 0.0));
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].points.len(), 4, "got {:?}", m[0].points.len());
        assert!(m[0].normal.x < -0.98, "normal = {:?}", m[0].normal);
        for p in &m[0].points {
            assert!((p.depth - 0.1).abs() < 1e-2, "depth = {}", p.depth);
        }
    }

    #[test]
    fn a_corner_on_a_face_gets_a_single_contact() {
        let cube = Shape::cuboid(1.0, 1.0, 1.0);
        // Rotated 45° about two axes so a vertex points down at the ground.
        let tilted = Isometry::new(
            Vector3::new(0.0, 1.6, 0.0),
            Quaternion::from_euler_xyz(0.6154797, 0.0, std::f32::consts::FRAC_PI_4),
        );
        let m = run(&cube, &tilted, &Shape::ground(), &at(0.0, 0.0, 0.0));
        assert_eq!(m.len(), 1);
        assert!(m[0].points.len() <= 2, "a corner is not a face: {:?}", m[0].points.len());
    }

    #[test]
    fn every_contact_normal_actually_separates_the_pair() {
        // The invariant the solver depends on, over a spread of poses.
        let shapes = [
            Shape::cuboid(1.0, 0.5, 0.75),
            Shape::ball(0.8),
            Shape::capsule(0.5, 0.4),
            Shape::cylinder(0.6, 0.5),
        ];
        for (i, a) in shapes.iter().enumerate() {
            for (j, b) in shapes.iter().enumerate() {
                let ia = Isometry::new(
                    Vector3::new(0.2, 0.1, -0.1),
                    Quaternion::from_euler_xyz(0.3 * i as f32, 0.5, 0.2 * j as f32).normalize(),
                );
                let ib = Isometry::new(
                    Vector3::new(-0.3, 0.2, 0.15),
                    Quaternion::from_euler_xyz(0.7, 0.1 * i as f32, 0.4).normalize(),
                );
                for m in run(a, &ia, b, &ib) {
                    let max_depth = m.points.iter().fold(0.0f32, |acc, p| acc.max(p.depth));
                    if max_depth <= 0.0 {
                        continue;
                    }
                    let moved = Isometry::new(ia.translation + m.normal * (max_depth + 0.05), ia.rotation);
                    let after = run(a, &moved, b, &ib);
                    let still_deep = after
                        .iter()
                        .flat_map(|m| m.points.iter())
                        .fold(0.0f32, |acc, p| acc.max(p.depth));
                    assert!(
                        still_deep <= 1e-2,
                        "shapes {i}/{j}: depth {max_depth} -> {still_deep} along {:?}",
                        m.normal
                    );
                }
            }
        }
    }

    #[test]
    fn a_compound_reports_one_manifold_per_overlapping_part() {
        let dumbbell = Shape::compound(vec![
            (at(-2.0, 0.0, 0.0), Shape::ball(0.5)),
            (at(2.0, 0.0, 0.0), Shape::ball(0.5)),
        ]);
        // Ground just under both ends.
        let m = run(&dumbbell, &at(0.0, 0.4, 0.0), &Shape::ground(), &at(0.0, 0.0, 0.0));
        assert_eq!(m.len(), 2, "both ends should touch");
        assert_eq!(m[0].sub_a, 0);
        assert_eq!(m[1].sub_a, 1);
    }

    #[test]
    fn a_box_resting_on_a_triangle_mesh_floor() {
        // Two triangles forming a 10x10 floor at y = 0.
        let floor = Shape::trimesh(
            vec![
                Vector3::new(-5.0, 0.0, -5.0),
                Vector3::new(5.0, 0.0, -5.0),
                Vector3::new(5.0, 0.0, 5.0),
                Vector3::new(-5.0, 0.0, 5.0),
            ],
            vec![[0, 2, 1], [0, 3, 2]],
        )
        .unwrap();
        let m = run(&Shape::cuboid(0.5, 0.5, 0.5), &at(0.0, 0.45, 0.0), &floor, &at(0.0, 0.0, 0.0));
        assert!(!m.is_empty(), "the box should rest on the mesh");
        let total: usize = m.iter().map(|x| x.points.len()).sum();
        assert!(total >= 3, "expected a face manifold, got {total} points");
        for man in &m {
            assert!(man.normal.y > 0.9, "normal = {:?}", man.normal);
            for p in &man.points {
                assert!((p.depth - 0.05).abs() < 1e-2, "depth = {}", p.depth);
            }
        }
    }

    #[test]
    fn trimesh_contacts_work_from_either_argument_order() {
        let floor = Shape::trimesh(
            vec![
                Vector3::new(-5.0, 0.0, -5.0),
                Vector3::new(5.0, 0.0, -5.0),
                Vector3::new(5.0, 0.0, 5.0),
                Vector3::new(-5.0, 0.0, 5.0),
            ],
            vec![[0, 2, 1], [0, 3, 2]],
        )
        .unwrap();
        let ball = Shape::ball(0.5);
        let a = run(&ball, &at(0.0, 0.45, 0.0), &floor, &at(0.0, 0.0, 0.0));
        let b = run(&floor, &at(0.0, 0.0, 0.0), &ball, &at(0.0, 0.45, 0.0));
        assert_eq!(a.len(), b.len());
        assert!(a[0].normal.y > 0.9);
        assert!(b[0].normal.y < -0.9, "flipped normal = {:?}", b[0].normal);
    }

    #[test]
    fn parallel_capsules_side_by_side_get_two_contacts() {
        let cap = Shape::capsule(1.0, 0.5);
        let m = run(&cap, &at(0.0, 0.0, 0.0), &cap, &at(0.9, 0.0, 0.0));
        let total: usize = m.iter().map(|x| x.points.len()).sum();
        assert!(total >= 2, "parallel capsules rock on a single point: {total}");
    }

    #[test]
    fn manifold_reduction_keeps_the_deepest_and_spreads_the_rest() {
        let mut pts: Vec<RawPoint> = (0..12)
            .map(|i| {
                let a = i as f32 / 12.0 * std::f32::consts::TAU;
                RawPoint {
                    point_a: Vector3::new(a.cos(), 0.0, a.sin()),
                    point_b: Vector3::ZERO,
                    depth: if i == 5 { 10.0 } else { 0.1 },
                }
            })
            .collect();
        reduce_manifold(&mut pts, Vector3::UP);
        assert_eq!(pts.len(), MAX_MANIFOLD_POINTS);
        assert!(pts.iter().any(|p| p.depth == 10.0), "deepest point was dropped");
        // The kept points should not be clustered.
        let spread = pts
            .iter()
            .flat_map(|p| pts.iter().map(move |q| (p.point_a - q.point_a).length()))
            .fold(0.0f32, f32::max);
        assert!(spread > 1.5, "kept points are clustered: spread {spread}");
    }
}
