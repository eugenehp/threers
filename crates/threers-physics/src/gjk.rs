//! GJK distance and EPA penetration depth for convex shapes.
//!
//! Together these answer the one question the narrow phase cannot answer
//! analytically: for two arbitrary convex shapes, how far apart are they, or how
//! deep do they overlap and along which axis?
//!
//! # Normal convention
//!
//! Every result here reports a unit `normal` **pointing from `b` toward `a`**:
//! translating `a` by `normal * depth` separates the pair. The same convention
//! holds throughout [`crate::contact`] and [`crate::solver`].

use crate::math::{try_normalize, Isometry};
use crate::shape::Shape;
use threers::math::Vector3;

const MAX_ITERATIONS: usize = 64;
const EPS: f32 = 1e-7;

/// Anything GJK can ask for extreme points of.
pub trait SupportMap {
    /// Farthest point of the shape along `dir`, in world space. `dir` is not
    /// necessarily normalised and is never zero.
    fn support(&self, dir: Vector3) -> Vector3;
}

/// A convex [`Shape`] positioned by an [`Isometry`].
pub struct ShapeProxy<'a> {
    shape: &'a Shape,
    iso: &'a Isometry,
}

impl<'a> ShapeProxy<'a> {
    /// `None` if the shape is not convex — GJK has no meaning for those.
    pub fn new(shape: &'a Shape, iso: &'a Isometry) -> Option<Self> {
        shape.is_convex().then_some(Self { shape, iso })
    }
}

impl SupportMap for ShapeProxy<'_> {
    fn support(&self, dir: Vector3) -> Vector3 {
        let local_dir = self.iso.inverse_transform_vector(dir);
        let p = self.shape.support_local(local_dir).unwrap_or(Vector3::ZERO);
        self.iso.transform_point(p)
    }
}

/// A single point — lets point queries reuse the GJK machinery.
pub struct PointProxy(pub Vector3);

impl SupportMap for PointProxy {
    fn support(&self, _dir: Vector3) -> Vector3 {
        self.0
    }
}

/// A world-space triangle, for collision against [`crate::trimesh::TriMesh`].
pub struct TriangleProxy(pub [Vector3; 3]);

impl SupportMap for TriangleProxy {
    fn support(&self, dir: Vector3) -> Vector3 {
        let mut best = self.0[0];
        let mut best_dot = best.dot(dir);
        for &v in &self.0[1..] {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = v;
            }
        }
        best
    }
}

/// A convex shape swept along a translation — used by shape casts.
pub struct TranslatedProxy<'a, S: SupportMap> {
    pub inner: &'a S,
    pub offset: Vector3,
}

impl<S: SupportMap> SupportMap for TranslatedProxy<'_, S> {
    fn support(&self, dir: Vector3) -> Vector3 {
        self.inner.support(dir) + self.offset
    }
}

/// Outcome of a convex-convex proximity query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Proximity {
    /// The shapes are apart. `distance` is positive.
    Separated {
        distance: f32,
        /// Closest point on `a`.
        point_a: Vector3,
        /// Closest point on `b`.
        point_b: Vector3,
        /// Unit vector from `b` toward `a`.
        normal: Vector3,
    },
    /// The shapes overlap. Translating `a` by `normal * depth` separates them.
    Penetrating {
        depth: f32,
        point_a: Vector3,
        point_b: Vector3,
        normal: Vector3,
    },
    /// GJK could not converge — degenerate or unbounded input.
    Failed,
}

impl Proximity {
    /// Signed gap: positive when apart, negative when overlapping.
    pub fn signed_distance(&self) -> Option<f32> {
        match self {
            Self::Separated { distance, .. } => Some(*distance),
            Self::Penetrating { depth, .. } => Some(-*depth),
            Self::Failed => None,
        }
    }
}

/// A vertex of the Minkowski difference `a ⊖ b`, remembering where it came from
/// so witness points can be recovered by barycentric interpolation.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    /// `pa - pb`.
    w: Vector3,
    pa: Vector3,
    pb: Vector3,
}

fn minkowski_support(a: &impl SupportMap, b: &impl SupportMap, dir: Vector3) -> Vertex {
    let pa = a.support(dir);
    let pb = b.support(-dir);
    Vertex { w: pa - pb, pa, pb }
}

/// Distance (or penetration) between two convex shapes.
pub fn closest_points(a: &impl SupportMap, b: &impl SupportMap) -> Proximity {
    let mut simplex: Vec<Vertex> = Vec::with_capacity(4);

    // Seed the simplex with a real support point rather than a bare direction.
    // If the seed happens to already be the closest feature — a ray straight
    // down a cylinder's axis, say — the loop terminates on its first pass, and
    // an empty simplex at that point has no witness points to report.
    let seed = minkowski_support(a, b, Vector3::UP);
    let mut v = seed.w;
    simplex.push(seed);
    if v.length_sq() <= EPS {
        return penetration(a, b, &mut simplex);
    }

    for _ in 0..MAX_ITERATIONS {
        let Some(dir) = try_normalize(-v) else {
            // v is the origin: the shapes touch or overlap.
            return penetration(a, b, &mut simplex);
        };
        let w = minkowski_support(a, b, dir);

        // Termination: the support point along -v is no closer to the origin
        // than v itself, so v is the true minimum-norm point of `a ⊖ b`.
        let progress = v.dot(v) - v.dot(w.w);
        if progress <= EPS * v.dot(v).max(1.0) {
            break;
        }

        simplex.push(w);
        match reduce_simplex(&mut simplex) {
            Some(closest) => v = closest,
            // The origin is enclosed by the simplex.
            None => return penetration(a, b, &mut simplex),
        }
        if v.length_sq() <= EPS {
            return penetration(a, b, &mut simplex);
        }
    }

    if simplex.is_empty() {
        return Proximity::Failed;
    }
    let (point_a, point_b) = witness(&simplex, v);
    let distance = v.length();
    match try_normalize(point_a - point_b) {
        // `a - b` points from b to a, which is exactly the convention.
        Some(normal) => Proximity::Separated {
            distance,
            point_a,
            point_b,
            normal,
        },
        None => penetration(a, b, &mut simplex),
    }
}

/// Whether two convex shapes overlap. Cheaper than [`closest_points`] — it
/// stops as soon as the answer is known.
pub fn intersect(a: &impl SupportMap, b: &impl SupportMap) -> bool {
    match closest_points(a, b) {
        Proximity::Penetrating { .. } => true,
        Proximity::Separated { distance, .. } => distance <= EPS,
        Proximity::Failed => false,
    }
}

/// Closest point of the origin on the current simplex, reducing it to the
/// smallest supporting face. `None` means the origin is inside a tetrahedron.
fn reduce_simplex(simplex: &mut Vec<Vertex>) -> Option<Vector3> {
    match simplex.len() {
        1 => Some(simplex[0].w),
        2 => {
            let (w, keep) = closest_origin_segment(simplex[0].w, simplex[1].w);
            retain_mask(simplex, keep);
            Some(w)
        }
        3 => {
            let (w, keep) = closest_origin_triangle(simplex[0].w, simplex[1].w, simplex[2].w);
            retain_mask(simplex, keep);
            Some(w)
        }
        4 => {
            let (w, keep) = closest_origin_tetrahedron(
                simplex[0].w,
                simplex[1].w,
                simplex[2].w,
                simplex[3].w,
            )?;
            retain_mask(simplex, keep);
            Some(w)
        }
        // A simplex never exceeds four points; drop the oldest if it somehow does.
        _ => {
            simplex.remove(0);
            reduce_simplex(simplex)
        }
    }
}

fn retain_mask(simplex: &mut Vec<Vertex>, keep: u8) {
    let mut i = 0;
    simplex.retain(|_| {
        let k = keep & (1 << i) != 0;
        i += 1;
        k
    });
}

/// Barycentric weights of the origin's projection onto the surviving simplex,
/// interpolated onto each shape to recover the witness points.
fn witness(simplex: &[Vertex], closest: Vector3) -> (Vector3, Vector3) {
    match simplex.len() {
        0 => (Vector3::ZERO, Vector3::ZERO),
        1 => (simplex[0].pa, simplex[0].pb),
        2 => {
            let (a, b) = (simplex[0].w, simplex[1].w);
            let ab = b - a;
            let len_sq = ab.length_sq();
            let t = if len_sq > EPS {
                ((closest - a).dot(ab) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (
                simplex[0].pa.lerp(simplex[1].pa, t),
                simplex[0].pb.lerp(simplex[1].pb, t),
            )
        }
        _ => {
            let bary = barycentric(closest, simplex[0].w, simplex[1].w, simplex[2].w);
            (
                simplex[0].pa * bary[0] + simplex[1].pa * bary[1] + simplex[2].pa * bary[2],
                simplex[0].pb * bary[0] + simplex[1].pb * bary[1] + simplex[2].pb * bary[2],
            )
        }
    }
}

fn barycentric(p: Vector3, a: Vector3, b: Vector3, c: Vector3) -> [f32; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    [1.0 - v - w, v, w]
}

fn closest_origin_segment(a: Vector3, b: Vector3) -> (Vector3, u8) {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= EPS {
        return (a, 0b01);
    }
    let t = (-a).dot(ab) / len_sq;
    if t <= 0.0 {
        (a, 0b01)
    } else if t >= 1.0 {
        (b, 0b10)
    } else {
        (a + ab * t, 0b11)
    }
}

/// Ericson, *Real-Time Collision Detection* §5.1.5, specialised to `p = origin`
/// and extended to report which vertices form the supporting feature.
fn closest_origin_triangle(a: Vector3, b: Vector3, c: Vector3) -> (Vector3, u8) {
    let ab = b - a;
    let ac = c - a;
    let d1 = ab.dot(-a);
    let d2 = ac.dot(-a);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 0b001);
    }

    let d3 = ab.dot(-b);
    let d4 = ac.dot(-b);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 0b010);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let t = if denom.abs() > EPS { d1 / denom } else { 0.0 };
        return (a + ab * t, 0b011);
    }

    let d5 = ab.dot(-c);
    let d6 = ac.dot(-c);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0b100);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let t = if denom.abs() > EPS { d2 / denom } else { 0.0 };
        return (a + ac * t, 0b101);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let t = if denom.abs() > EPS {
            (d4 - d3) / denom
        } else {
            0.0
        };
        return (b + (c - b) * t, 0b110);
    }

    let denom = va + vb + vc;
    if denom.abs() < EPS {
        return (a, 0b001);
    }
    let inv = 1.0 / denom;
    (a + ab * (vb * inv) + ac * (vc * inv), 0b111)
}

/// `None` when the origin lies inside the tetrahedron.
fn closest_origin_tetrahedron(
    a: Vector3,
    b: Vector3,
    c: Vector3,
    d: Vector3,
) -> Option<(Vector3, u8)> {
    // Faces wound so that the fourth vertex is the interior reference.
    const FACES: [[usize; 4]; 4] = [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 3, 2, 0],
    ];
    let pts = [a, b, c, d];

    let mut best: Option<(Vector3, u8, f32)> = None;
    let mut outside_any = false;
    let mut any_valid_face = false;

    for face in FACES {
        let (p0, p1, p2, interior) = (pts[face[0]], pts[face[1]], pts[face[2]], pts[face[3]]);
        // Normalise before testing sides. A sliver tetrahedron has tiny face
        // normals, and the unnormalised sign product is then dominated by
        // rounding — which shows up as a separated pair being reported as
        // penetrating.
        let Some(n) = try_normalize((p1 - p0).cross(p2 - p0)) else {
            continue;
        };
        any_valid_face = true;
        // Orient outward: away from the remaining vertex.
        let n = if n.dot(interior - p0) > 0.0 { -n } else { n };
        // True signed distance of the origin from the face plane.
        if n.dot(-p0) <= 0.0 {
            continue; // origin is on the inner side of this face
        }
        outside_any = true;

        let (closest, mask) = closest_origin_triangle(p0, p1, p2);
        let dist_sq = closest.length_sq();
        if best.is_none_or(|(_, _, d)| dist_sq < d) {
            // Lift the face-local mask back onto the tetrahedron's indices.
            let mut full = 0u8;
            for (bit, &idx) in face[..3].iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    full |= 1 << idx;
                }
            }
            best = Some((closest, full, dist_sq));
        }
    }

    if !any_valid_face {
        // Fully degenerate tetrahedron (all four points collinear or
        // coincident). Fall back to the triangle it collapsed to rather than
        // claiming the origin is enclosed by a shape with no volume.
        let (closest, mask) = closest_origin_triangle(a, b, c);
        return Some((closest, mask));
    }
    if !outside_any {
        return None; // origin enclosed — penetrating
    }
    best.map(|(p, mask, _)| (p, mask))
        .or(Some((a, 0b0001)))
}

// ---- EPA ------------------------------------------------------------------

/// Penetration depth by expanding the GJK simplex out to the Minkowski boundary.
fn penetration(a: &impl SupportMap, b: &impl SupportMap, simplex: &mut Vec<Vertex>) -> Proximity {
    if !expand_to_tetrahedron(a, b, simplex) {
        return Proximity::Failed;
    }

    #[derive(Clone, Copy)]
    struct Face {
        idx: [usize; 3],
        normal: Vector3,
        dist: f32,
    }

    let mut verts = simplex.clone();

    // Build an outward-oriented tetrahedron. The centroid is interior, so a
    // face whose plane has the centroid on the positive side is wound wrong.
    let centroid = (verts[0].w + verts[1].w + verts[2].w + verts[3].w) * 0.25;
    let make_face = |verts: &[Vertex], i: usize, j: usize, k: usize| -> Option<Face> {
        let (p0, p1, p2) = (verts[i].w, verts[j].w, verts[k].w);
        let n = try_normalize((p1 - p0).cross(p2 - p0))?;
        let (idx, n) = if n.dot(centroid - p0) > 0.0 {
            ([i, k, j], -n)
        } else {
            ([i, j, k], n)
        };
        Some(Face {
            idx,
            normal: n,
            dist: n.dot(p0),
        })
    };

    let mut faces: Vec<Face> = Vec::with_capacity(16);
    for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        match make_face(&verts, i, j, k) {
            Some(f) => faces.push(f),
            None => return Proximity::Failed,
        }
    }

    for _ in 0..MAX_ITERATIONS {
        // Face of the current polytope closest to the origin.
        let Some((best_i, best)) = faces
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| x.dist.partial_cmp(&y.dist).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, f)| (i, *f))
        else {
            return Proximity::Failed;
        };

        let w = minkowski_support(a, b, best.normal);
        let reach = w.w.dot(best.normal);

        // The polytope already touches the Minkowski boundary here.
        if reach - best.dist <= 1e-4 {
            return epa_result(&verts, &best.idx, best.normal, best.dist.max(0.0));
        }

        // Remove every face this new point can see, then stitch the hole shut.
        let p = w.w;
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut kept: Vec<Face> = Vec::with_capacity(faces.len() + 4);
        let mut removed_any = false;
        for (fi, f) in faces.iter().enumerate() {
            if f.normal.dot(p) - f.dist > 1e-6 || fi == best_i {
                removed_any = true;
                horizon.push((f.idx[0], f.idx[1]));
                horizon.push((f.idx[1], f.idx[2]));
                horizon.push((f.idx[2], f.idx[0]));
            } else {
                kept.push(*f);
            }
        }
        if !removed_any {
            return epa_result(&verts, &best.idx, best.normal, best.dist.max(0.0));
        }

        verts.push(w);
        let new_idx = verts.len() - 1;
        faces = kept;
        for &(e0, e1) in &horizon {
            // An edge shared by two removed faces is interior to the hole.
            if horizon.iter().any(|&(f0, f1)| f0 == e1 && f1 == e0) {
                continue;
            }
            if let Some(f) = make_face(&verts, e0, e1, new_idx) {
                faces.push(f);
            }
        }
        if faces.is_empty() {
            return Proximity::Failed;
        }
    }

    // Out of iterations — report the best face found so far rather than nothing.
    faces
        .iter()
        .min_by(|x, y| x.dist.partial_cmp(&y.dist).unwrap_or(std::cmp::Ordering::Equal))
        .map(|f| epa_result(&verts, &f.idx, f.normal, f.dist.max(0.0)))
        .unwrap_or(Proximity::Failed)
}

fn epa_result(verts: &[Vertex], idx: &[usize; 3], normal: Vector3, depth: f32) -> Proximity {
    let (v0, v1, v2) = (verts[idx[0]], verts[idx[1]], verts[idx[2]]);
    let bary = barycentric(normal * depth, v0.w, v1.w, v2.w);
    let point_a = v0.pa * bary[0] + v1.pa * bary[1] + v2.pa * bary[2];
    let point_b = v0.pb * bary[0] + v1.pb * bary[1] + v2.pb * bary[2];
    Proximity::Penetrating {
        depth,
        point_a,
        point_b,
        // `normal` points from the origin toward the closest boundary point of
        // `a ⊖ b`, i.e. the direction `a` must move *back* along. Negate it to
        // get the separating direction for `a`.
        normal: -normal,
    }
}

/// Grow a degenerate simplex into a tetrahedron enclosing the origin, which EPA
/// requires as its starting polytope.
fn expand_to_tetrahedron(a: &impl SupportMap, b: &impl SupportMap, simplex: &mut Vec<Vertex>) -> bool {
    if simplex.is_empty() {
        simplex.push(minkowski_support(a, b, Vector3::UP));
    }
    if simplex.len() == 1 {
        // Probe along each axis for a second, distinct point.
        for dir in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ] {
            let w = minkowski_support(a, b, dir);
            if (w.w - simplex[0].w).length_sq() > 1e-8 {
                simplex.push(w);
                break;
            }
        }
    }
    if simplex.len() < 2 {
        return false;
    }
    if simplex.len() == 2 {
        // Any direction perpendicular to the segment gives a third point.
        let axis = simplex[1].w - simplex[0].w;
        let (t1, t2) = crate::math::orthonormal_basis(
            try_normalize(axis).unwrap_or(Vector3::UP),
        );
        for dir in [t1, t2, -t1, -t2] {
            let w = minkowski_support(a, b, dir);
            let area = (simplex[1].w - simplex[0].w).cross(w.w - simplex[0].w);
            if area.length_sq() > 1e-10 {
                simplex.push(w);
                break;
            }
        }
    }
    if simplex.len() < 3 {
        return false;
    }
    if simplex.len() == 3 {
        let n = (simplex[1].w - simplex[0].w).cross(simplex[2].w - simplex[0].w);
        let Some(n) = try_normalize(n) else {
            return false;
        };
        for dir in [n, -n] {
            let w = minkowski_support(a, b, dir);
            if (w.w - simplex[0].w).dot(n).abs() > 1e-6 {
                simplex.push(w);
                break;
            }
        }
    }
    simplex.len() == 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Quaternion;

    fn at(x: f32, y: f32, z: f32) -> Isometry {
        Isometry::from_translation(Vector3::new(x, y, z))
    }

    fn query(sa: &Shape, ia: &Isometry, sb: &Shape, ib: &Isometry) -> Proximity {
        let a = ShapeProxy::new(sa, ia).unwrap();
        let b = ShapeProxy::new(sb, ib).unwrap();
        closest_points(&a, &b)
    }

    #[test]
    fn separated_boxes_report_the_gap() {
        let s = Shape::cuboid(1.0, 1.0, 1.0);
        let r = query(&s, &at(0.0, 0.0, 0.0), &s, &at(5.0, 0.0, 0.0));
        match r {
            Proximity::Separated {
                distance, normal, ..
            } => {
                assert!((distance - 3.0).abs() < 1e-3, "distance = {distance}");
                // b is at +x, so the direction from b to a is -x.
                assert!(normal.x < -0.99, "normal = {normal:?}");
            }
            other => panic!("expected separation, got {other:?}"),
        }
    }

    #[test]
    fn overlapping_boxes_report_the_shallow_axis() {
        let s = Shape::cuboid(1.0, 1.0, 1.0);
        // Overlap 0.5 along x, 1.5 along y — the minimum translation is along x.
        let r = query(&s, &at(0.0, 0.0, 0.0), &s, &at(1.5, 0.5, 0.0));
        match r {
            Proximity::Penetrating { depth, normal, .. } => {
                assert!((depth - 0.5).abs() < 1e-2, "depth = {depth}");
                assert!(normal.x < -0.98, "normal = {normal:?}");
            }
            other => panic!("expected penetration, got {other:?}"),
        }
    }

    #[test]
    fn sphere_distance_matches_the_analytic_answer() {
        let a = Shape::ball(1.0);
        let b = Shape::ball(2.0);
        let r = query(&a, &at(0.0, 0.0, 0.0), &b, &at(10.0, 0.0, 0.0));
        let Proximity::Separated { distance, point_a, point_b, .. } = r else {
            panic!("expected separation, got {r:?}");
        };
        assert!((distance - 7.0).abs() < 1e-2, "distance = {distance}");
        assert!((point_a - Vector3::new(1.0, 0.0, 0.0)).length() < 1e-2);
        assert!((point_b - Vector3::new(8.0, 0.0, 0.0)).length() < 1e-2);
    }

    #[test]
    fn sphere_penetration_matches_the_analytic_answer() {
        let a = Shape::ball(1.0);
        let b = Shape::ball(1.0);
        let r = query(&a, &at(0.5, 0.0, 0.0), &b, &at(-0.5, 0.0, 0.0));
        let Proximity::Penetrating { depth, normal, .. } = r else {
            panic!("expected penetration, got {r:?}");
        };
        assert!((depth - 1.0).abs() < 1e-2, "depth = {depth}");
        // a is at +x relative to b, so a must move further +x to separate.
        assert!(normal.x > 0.98, "normal = {normal:?}");
    }

    #[test]
    fn the_normal_actually_separates_the_pair() {
        // Translating `a` by normal * depth must remove the overlap — the
        // property everything downstream relies on.
        let a = Shape::cuboid(1.0, 0.5, 0.75);
        let b = Shape::cylinder(1.0, 0.8);
        let ia = Isometry::new(
            Vector3::new(0.3, 0.4, -0.2),
            Quaternion::from_euler_xyz(0.3, 0.2, 0.7).normalize(),
        );
        let ib = Isometry::new(
            Vector3::new(-0.4, 0.1, 0.3),
            Quaternion::from_euler_xyz(-0.5, 1.0, 0.2).normalize(),
        );
        let Proximity::Penetrating { depth, normal, .. } = query(&a, &ia, &b, &ib) else {
            panic!("shapes should overlap");
        };
        // Push slightly beyond the reported depth to clear rounding.
        let moved = Isometry::new(ia.translation + normal * (depth + 0.02), ia.rotation);
        let after = query(&a, &moved, &b, &ib);
        assert!(
            matches!(after, Proximity::Separated { .. }),
            "still overlapping after separation: {after:?}"
        );
    }

    #[test]
    fn deep_penetration_finds_the_true_minimum_axis() {
        // A thin slab deeply overlapping a big box: the answer must be the thin
        // axis, which is where naive EPA seeding tends to fail.
        let slab = Shape::cuboid(2.0, 0.1, 2.0);
        let big = Shape::cuboid(1.0, 1.0, 1.0);
        let r = query(&slab, &at(0.0, 1.05, 0.0), &big, &at(0.0, 0.0, 0.0));
        let Proximity::Penetrating { depth, normal, .. } = r else {
            panic!("expected penetration, got {r:?}");
        };
        assert!((depth - 0.05).abs() < 1e-2, "depth = {depth}");
        assert!(normal.y > 0.98, "normal = {normal:?}");
    }

    #[test]
    fn point_containment_via_gjk() {
        let s = Shape::ball(1.0);
        let iso = at(0.0, 0.0, 0.0);
        let inside = closest_points(&ShapeProxy::new(&s, &iso).unwrap(), &PointProxy(Vector3::new(0.2, 0.0, 0.0)));
        assert!(matches!(inside, Proximity::Penetrating { .. }));
        let outside = closest_points(&ShapeProxy::new(&s, &iso).unwrap(), &PointProxy(Vector3::new(3.0, 0.0, 0.0)));
        match outside {
            Proximity::Separated { distance, .. } => assert!((distance - 2.0).abs() < 1e-2),
            other => panic!("expected separation, got {other:?}"),
        }
    }

    #[test]
    fn triangle_proxy_collides_with_a_box() {
        let tri = TriangleProxy([
            Vector3::new(-2.0, 0.0, -2.0),
            Vector3::new(2.0, 0.0, -2.0),
            Vector3::new(0.0, 0.0, 2.0),
        ]);
        let s = Shape::cuboid(0.5, 0.5, 0.5);
        // Box centre 0.4 above the plane: overlaps by 0.1.
        let iso = at(0.0, 0.4, 0.0);
        let r = closest_points(&ShapeProxy::new(&s, &iso).unwrap(), &tri);
        let Proximity::Penetrating { depth, normal, .. } = r else {
            panic!("expected penetration, got {r:?}");
        };
        assert!((depth - 0.1).abs() < 1e-2, "depth = {depth}");
        assert!(normal.y > 0.9, "normal = {normal:?}");
    }

    #[test]
    fn touching_shapes_do_not_panic() {
        let s = Shape::cuboid(1.0, 1.0, 1.0);
        let r = query(&s, &at(0.0, 0.0, 0.0), &s, &at(2.0, 0.0, 0.0));
        assert!(r.signed_distance().is_some(), "got {r:?}");
        assert!(r.signed_distance().unwrap().abs() < 1e-2);
    }

    #[test]
    fn distance_is_symmetric_under_swapping() {
        let a = Shape::cuboid(0.6, 1.2, 0.4);
        let b = Shape::ball(0.9);
        let (ia, ib) = (at(0.0, 0.0, 0.0), at(3.0, 1.0, -1.0));
        let d1 = query(&a, &ia, &b, &ib).signed_distance().unwrap();
        let d2 = query(&b, &ib, &a, &ia).signed_distance().unwrap();
        assert!((d1 - d2).abs() < 1e-3, "{d1} vs {d2}");
    }
}
