//! Scene queries: rays, shape casts, overlap tests and point lookups.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! # world.add_body(RigidBody::fixed().shape(Shape::ground()));
//!
//! // What is under the crosshair?
//! let ray = Ray::new(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
//! if let Some(hit) = world.raycast(&ray, 100.0, QueryFilter::default()) {
//!     println!("hit body {:?} at {:?}", hit.body, hit.point);
//! }
//!
//! // Can the player capsule move 2 m forward without clipping anything?
//! let cast = world.cast_shape(
//!     &Shape::capsule(0.5, 0.3),
//!     &Isometry::from_translation(Vector3::new(0.0, 1.0, 0.0)),
//!     Vector3::new(0.0, 0.0, 1.0),
//!     2.0,
//!     QueryFilter::default(),
//! );
//! ```

use crate::body::{BodyId, RigidBody};
use crate::gjk::{closest_points, PointProxy, Proximity, ShapeProxy, SupportMap, TranslatedProxy};
use crate::material::InteractionGroups;
use crate::math::{closest_point_on_segment, try_normalize, Isometry};
use crate::shape::Shape;
use crate::world::World;
use threers::math::{Ray, Vector3};

/// Which bodies a query is allowed to see.
///
/// ```
/// use threers_physics::prelude::*;
///
/// // Ignore the body doing the looking, and ignore trigger volumes.
/// # let player = None;
/// let filter = QueryFilter::default().exclude(player);
/// assert!(filter.exclude_sensors);
/// ```
/// An extra per-body test a query runs on top of the flags.
pub type QueryPredicate<'a> = &'a dyn Fn(BodyId, &RigidBody) -> bool;

#[derive(Clone, Copy)]
pub struct QueryFilter<'a> {
    /// Groups the query itself belongs to and looks for.
    pub groups: InteractionGroups,
    /// Skip this body — nearly always the one casting.
    pub exclude_body: Option<BodyId>,
    /// Skip sensor colliders. On by default: a trigger volume is not something
    /// you want a camera ray or a footstep check to hit.
    pub exclude_sensors: bool,
    pub include_fixed: bool,
    pub include_dynamic: bool,
    pub include_kinematic: bool,
    /// Arbitrary extra test. Return `false` to skip a body.
    pub predicate: Option<QueryPredicate<'a>>,
}

impl Default for QueryFilter<'_> {
    fn default() -> Self {
        Self {
            groups: InteractionGroups::ALL,
            exclude_body: None,
            exclude_sensors: true,
            include_fixed: true,
            include_dynamic: true,
            include_kinematic: true,
            predicate: None,
        }
    }
}

impl<'a> QueryFilter<'a> {
    /// Ignore a body. Accepts `Option` so `filter.exclude(self.body)` works
    /// whether or not you have a handle.
    pub fn exclude(mut self, body: impl Into<Option<BodyId>>) -> Self {
        self.exclude_body = body.into();
        self
    }

    pub fn groups(mut self, groups: InteractionGroups) -> Self {
        self.groups = groups;
        self
    }

    /// Include sensor colliders in the results.
    pub fn include_sensors(mut self) -> Self {
        self.exclude_sensors = false;
        self
    }

    /// Only see immovable level geometry.
    pub fn only_fixed(mut self) -> Self {
        self.include_fixed = true;
        self.include_dynamic = false;
        self.include_kinematic = false;
        self
    }

    /// Only see bodies the simulation moves.
    pub fn only_dynamic(mut self) -> Self {
        self.include_fixed = false;
        self.include_dynamic = true;
        self.include_kinematic = false;
        self
    }

    pub fn predicate(mut self, f: &'a dyn Fn(BodyId, &RigidBody) -> bool) -> Self {
        self.predicate = Some(f);
        self
    }

    fn accepts_body(&self, id: BodyId, body: &RigidBody) -> bool {
        if !body.enabled || Some(id) == self.exclude_body {
            return false;
        }
        let type_ok = match body.body_type {
            crate::body::BodyType::Fixed => self.include_fixed,
            crate::body::BodyType::Dynamic => self.include_dynamic,
            crate::body::BodyType::Kinematic => self.include_kinematic,
        };
        if !type_ok {
            return false;
        }
        self.predicate.is_none_or(|f| f(id, body))
    }

    fn accepts_collider(&self, collider: &crate::collider::Collider) -> bool {
        collider.enabled
            && !(self.exclude_sensors && collider.is_sensor)
            && self.groups.test(&collider.groups)
    }
}

/// Where a ray met a collider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub body: BodyId,
    /// Index into the body's collider list.
    pub collider: usize,
    /// Distance along the ray. The ray direction is normalised first, so this is
    /// a true distance in world units.
    pub toi: f32,
    pub point: Vector3,
    /// Surface normal at the hit, pointing out of the surface.
    pub normal: Vector3,
    /// [`crate::collider::Collider::user_data`] of the collider hit.
    pub user_data: u64,
}

/// Where a swept shape first touched something.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeHit {
    pub body: BodyId,
    pub collider: usize,
    /// Distance travelled before contact.
    pub toi: f32,
    /// Contact point on the cast shape, at the moment of impact.
    pub witness_cast: Vector3,
    /// Contact point on the obstacle.
    pub witness_target: Vector3,
    /// Normal pointing from the obstacle back toward the cast shape.
    pub normal: Vector3,
    pub user_data: u64,
}

/// Nearest point on a collider to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointProjection {
    pub body: BodyId,
    pub collider: usize,
    pub point: Vector3,
    /// Distance to the surface. Zero when the query point is inside.
    pub distance: f32,
    pub is_inside: bool,
}

const MAX_ADVANCE_ITERATIONS: usize = 48;
/// Gap below which a swept shape counts as touching.
const CONTACT_EPS: f32 = 1e-4;

impl World {
    /// Nearest hit along a ray, or `None`.
    ///
    /// `ray.direction` is normalised, so `toi` is a distance.
    pub fn raycast(&self, ray: &Ray, max_toi: f32, filter: QueryFilter) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        self.for_each_ray_hit(ray, max_toi, filter, &mut |hit| {
            if best.is_none_or(|b| hit.toi < b.toi) {
                best = Some(hit);
            }
            true
        });
        best
    }

    /// Every hit along a ray, unordered. Sort by `toi` if you need them in order.
    pub fn raycast_all(&self, ray: &Ray, max_toi: f32, filter: QueryFilter) -> Vec<RayHit> {
        let mut hits = Vec::new();
        self.for_each_ray_hit(ray, max_toi, filter, &mut |hit| {
            hits.push(hit);
            true
        });
        hits
    }

    /// Whether anything at all blocks the segment — the cheap "can A see B?"
    /// test, since it stops at the first hit.
    pub fn ray_is_blocked(&self, ray: &Ray, max_toi: f32, filter: QueryFilter) -> bool {
        let mut blocked = false;
        self.for_each_ray_hit(ray, max_toi, filter, &mut |_| {
            blocked = true;
            false // stop early
        });
        blocked
    }

    /// Visit every ray hit. Return `false` from `visit` to stop.
    pub fn for_each_ray_hit(
        &self,
        ray: &Ray,
        max_toi: f32,
        filter: QueryFilter,
        visit: &mut dyn FnMut(RayHit) -> bool,
    ) {
        let Some(direction) = try_normalize(ray.direction) else {
            return;
        };
        let ray = Ray::new(ray.origin, direction);

        for (id, body) in self.bodies().iter() {
            if !filter.accepts_body(id, body) {
                continue;
            }
            // Cheap whole-body rejection first.
            let aabb = body.compute_aabb();
            if aabb.is_empty() {
                continue;
            }
            // Only reject on the entry distance, and only when the ray starts
            // outside. `intersect_box` reports the *exit* for an origin inside
            // the box, which for an unbounded shape like a ground plane is
            // enormous — testing that against `max_toi` would skip the very
            // body the ray is aimed at.
            if !aabb.contains_point(ray.origin) {
                match ray.intersect_box(&aabb) {
                    Some(t) if t <= max_toi => {}
                    _ => continue,
                }
            }

            for (index, collider) in body.colliders().iter().enumerate() {
                if !filter.accepts_collider(collider) {
                    continue;
                }
                let iso = collider.world_transform(&body.position);
                let local = Ray::new(
                    iso.inverse_transform_point(ray.origin),
                    iso.inverse_transform_vector(ray.direction),
                );
                let Some((toi, local_normal)) = raycast_shape(&collider.shape, &local, max_toi)
                else {
                    continue;
                };
                if toi > max_toi {
                    continue;
                }
                let hit = RayHit {
                    body: id,
                    collider: index,
                    toi,
                    point: ray.at(toi),
                    normal: iso.transform_vector(local_normal),
                    user_data: collider.user_data,
                };
                if !visit(hit) {
                    return;
                }
            }
        }
    }

    /// Sweep a convex shape along `direction` and report the first thing it hits.
    ///
    /// This is how you move a character without tunnelling: cast, move to `toi`,
    /// slide along the normal, repeat.
    ///
    /// The shape must be convex; compound and triangle-mesh casts return `None`.
    pub fn cast_shape(
        &self,
        shape: &Shape,
        start: &Isometry,
        direction: Vector3,
        max_toi: f32,
        filter: QueryFilter,
    ) -> Option<ShapeHit> {
        let direction = try_normalize(direction)?;
        let cast_proxy = ShapeProxy::new(shape, start)?;

        // Bounds covering the whole sweep, for cheap rejection.
        let mut swept = shape.compute_aabb(start);
        swept = swept.union(&shape.compute_aabb(&Isometry::new(
            start.translation + direction * max_toi,
            start.rotation,
        )));

        let mut best: Option<ShapeHit> = None;
        for (id, body) in self.bodies().iter() {
            if !filter.accepts_body(id, body) {
                continue;
            }
            let aabb = body.compute_aabb();
            if aabb.is_empty() || !aabb.intersects_box(&swept) {
                continue;
            }
            for (index, collider) in body.colliders().iter().enumerate() {
                if !filter.accepts_collider(collider) {
                    continue;
                }
                let target_iso = collider.world_transform(&body.position);
                let limit = best.map_or(max_toi, |b| b.toi);
                let Some(hit) = cast_against(
                    &cast_proxy,
                    &collider.shape,
                    &target_iso,
                    direction,
                    limit,
                ) else {
                    continue;
                };
                let (toi, witness_cast, witness_target, normal) = hit;
                if best.is_none_or(|b| toi < b.toi) {
                    best = Some(ShapeHit {
                        body: id,
                        collider: index,
                        toi,
                        witness_cast,
                        witness_target,
                        normal,
                        user_data: collider.user_data,
                    });
                }
            }
        }
        best
    }

    /// Every collider overlapping `shape` placed at `iso`.
    pub fn intersections_with_shape(
        &self,
        shape: &Shape,
        iso: &Isometry,
        filter: QueryFilter,
    ) -> Vec<(BodyId, usize)> {
        let mut out = Vec::new();
        let query_aabb = shape.compute_aabb(iso);
        for (id, body) in self.bodies().iter() {
            if !filter.accepts_body(id, body) {
                continue;
            }
            let aabb = body.compute_aabb();
            if aabb.is_empty() || !aabb.intersects_box(&query_aabb) {
                continue;
            }
            for (index, collider) in body.colliders().iter().enumerate() {
                if !filter.accepts_collider(collider) {
                    continue;
                }
                let target = collider.world_transform(&body.position);
                let mut raw = Vec::new();
                crate::narrowphase::collide(shape, iso, &collider.shape, &target, 0.0, &mut raw);
                if raw.iter().any(|m| m.points.iter().any(|p| p.depth >= 0.0)) {
                    out.push((id, index));
                }
            }
        }
        out
    }

    /// Bodies whose colliders contain `point`.
    pub fn bodies_at_point(&self, point: Vector3, filter: QueryFilter) -> Vec<BodyId> {
        let mut out = Vec::new();
        for (id, body) in self.bodies().iter() {
            if !filter.accepts_body(id, body) {
                continue;
            }
            let aabb = body.compute_aabb();
            if aabb.is_empty() || !aabb.contains_point(point) {
                continue;
            }
            let hit = body.colliders().iter().any(|c| {
                filter.accepts_collider(c)
                    && c.shape
                        .contains_point_local(c.world_transform(&body.position).inverse_transform_point(point))
            });
            if hit {
                out.push(id);
            }
        }
        out
    }

    /// Nearest collider surface to `point`.
    pub fn project_point(&self, point: Vector3, filter: QueryFilter) -> Option<PointProjection> {
        let mut best: Option<PointProjection> = None;
        for (id, body) in self.bodies().iter() {
            if !filter.accepts_body(id, body) {
                continue;
            }
            for (index, collider) in body.colliders().iter().enumerate() {
                if !filter.accepts_collider(collider) {
                    continue;
                }
                let iso = collider.world_transform(&body.position);
                let Some((surface, distance, inside)) = project_onto_shape(&collider.shape, &iso, point)
                else {
                    continue;
                };
                if best.is_none_or(|b| distance < b.distance) {
                    best = Some(PointProjection {
                        body: id,
                        collider: index,
                        point: surface,
                        distance,
                        is_inside: inside,
                    });
                }
            }
        }
        best
    }
}

/// Ray against one shape, in that shape's local frame.
///
/// Returns `(toi, outward normal)`.
fn raycast_shape(shape: &Shape, ray: &Ray, max_toi: f32) -> Option<(f32, Vector3)> {
    match shape {
        Shape::Ball { radius } => {
            let sphere = threers::math::Sphere::new(Vector3::ZERO, *radius);
            let toi = ray.intersect_sphere(&sphere)?;
            Some((toi, try_normalize(ray.at(toi))?))
        }
        Shape::Cuboid { half_extents } => {
            let aabb = crate::math::Aabb::new(-*half_extents, *half_extents);
            let toi = ray.intersect_box(&aabb)?;
            Some((toi, box_normal(ray.at(toi), *half_extents)))
        }
        Shape::HalfSpace { normal } => {
            let denominator = normal.dot(ray.direction);
            // Parallel, or travelling away from the surface.
            if denominator.abs() < 1e-9 {
                return None;
            }
            let toi = -normal.dot(ray.origin) / denominator;
            (toi >= 0.0).then_some((toi, *normal))
        }
        Shape::Capsule {
            half_height,
            radius,
        } => raycast_capsule(ray, *half_height, *radius),
        Shape::ConvexHull(hull) => {
            // A convex polyhedron is an intersection of half-spaces, so clip the
            // ray interval against every face plane.
            let (mut enter, mut exit) = (0.0f32, f32::INFINITY);
            let mut enter_normal = Vector3::UP;
            for face in &hull.faces {
                let denominator = face.normal.dot(ray.direction);
                let distance = face.normal.dot(ray.origin) - face.offset;
                if denominator.abs() < 1e-9 {
                    if distance > 0.0 {
                        return None; // parallel and outside this face
                    }
                    continue;
                }
                let t = -distance / denominator;
                if denominator < 0.0 {
                    if t > enter {
                        enter = t;
                        enter_normal = face.normal;
                    }
                } else if t < exit {
                    exit = t;
                }
                if enter > exit {
                    return None;
                }
            }
            (enter <= exit && enter >= 0.0).then_some((enter, enter_normal))
        }
        Shape::TriMesh(mesh) => {
            let (toi, tri) = mesh.raycast(ray, max_toi, false)?;
            let normal = mesh.triangle_normal(tri)?;
            // Orient against the ray so the caller always gets a facing normal.
            Some((
                toi,
                if normal.dot(ray.direction) > 0.0 { -normal } else { normal },
            ))
        }
        Shape::Compound(parts) => {
            let mut best: Option<(f32, Vector3)> = None;
            for (local, child) in parts.iter() {
                let inner = Ray::new(
                    local.inverse_transform_point(ray.origin),
                    local.inverse_transform_vector(ray.direction),
                );
                if let Some((toi, n)) = raycast_shape(child, &inner, max_toi) {
                    if best.is_none_or(|(b, _)| toi < b) {
                        best = Some((toi, local.transform_vector(n)));
                    }
                }
            }
            best
        }
        Shape::Cone { half_height, .. } => {
            let (toi, normal) = raycast_by_advancement(shape, ray, max_toi)?;
            // The apex is a point, not a surface. Every lateral normal of the
            // cone meets there and none of them is the answer — a ray arriving
            // down the axis gets whichever facet the solver happened to land on,
            // which on a wide cone can be more sideways than up. Hand back the
            // axis instead: it is the one direction the shape is symmetric
            // about, and it is what a caller resolving a contact there wants.
            let apex = Vector3::new(0.0, *half_height, 0.0);
            if (ray.at(toi) - apex).length() < half_height.abs().max(1.0) * 1e-3 {
                return Some((toi, Vector3::UP));
            }
            Some((toi, normal))
        }
        // Cylinders have no tidy closed form worth the code; advancing
        // along the ray by the GJK distance converges quickly and is exact in
        // the limit.
        _ => raycast_by_advancement(shape, ray, max_toi),
    }
}

fn box_normal(point: Vector3, half: Vector3) -> Vector3 {
    // Whichever face the point is closest to is the one it exited through.
    let d = [
        (half.x - point.x.abs(), Vector3::new(point.x.signum(), 0.0, 0.0)),
        (half.y - point.y.abs(), Vector3::new(0.0, point.y.signum(), 0.0)),
        (half.z - point.z.abs(), Vector3::new(0.0, 0.0, point.z.signum())),
    ];
    d.iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, n)| *n)
        .unwrap_or(Vector3::UP)
}

/// Ray against a Y-aligned sphere-swept segment.
fn raycast_capsule(ray: &Ray, half_height: f32, radius: f32) -> Option<(f32, Vector3)> {
    let (a, b) = (
        Vector3::new(0.0, -half_height, 0.0),
        Vector3::new(0.0, half_height, 0.0),
    );
    let axis = b - a;
    let oa = ray.origin - a;
    let axis_len_sq = axis.length_sq().max(1e-12);

    // Project the ray and its origin onto the plane across the axis, reducing
    // the side surface to a 2D circle intersection.
    let dir_along = ray.direction.dot(axis) / axis_len_sq;
    let oa_along = oa.dot(axis) / axis_len_sq;
    let dir_perp = ray.direction - axis * dir_along;
    let oa_perp = oa - axis * oa_along;

    let qa = dir_perp.length_sq();
    let qb = 2.0 * dir_perp.dot(oa_perp);
    let qc = oa_perp.length_sq() - radius * radius;

    let mut best: Option<f32> = None;
    if qa > 1e-12 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            for t in [(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)] {
                if t < 0.0 {
                    continue;
                }
                // Only counts if it lands on the cylindrical section.
                let h = oa_along + t * dir_along;
                if (0.0..=1.0).contains(&h) && best.is_none_or(|x| t < x) {
                    best = Some(t);
                }
            }
        }
    }

    // The two end caps.
    for centre in [a, b] {
        let sphere = threers::math::Sphere::new(centre, radius);
        if let Some(t) = ray.intersect_sphere(&sphere) {
            if t >= 0.0 && best.is_none_or(|x| t < x) {
                best = Some(t);
            }
        }
    }

    let toi = best?;
    let point = ray.at(toi);
    let normal = try_normalize(point - closest_point_on_segment(point, a, b))?;
    Some((toi, normal))
}

/// Walk along the ray in steps of the current distance to the shape. Each step
/// is guaranteed not to pass through the surface, so this converges on the first
/// hit from outside.
fn raycast_by_advancement(shape: &Shape, ray: &Ray, max_toi: f32) -> Option<(f32, Vector3)> {
    let iso = Isometry::IDENTITY;
    let proxy = ShapeProxy::new(shape, &iso)?;
    let mut toi = 0.0f32;
    // GJK reports the direction from the query point toward the shape; the
    // outward surface normal is the opposite.
    let mut last_normal = -ray.direction;
    for _ in 0..MAX_ADVANCE_ITERATIONS {
        let point = ray.at(toi);
        match closest_points(&proxy, &PointProxy(point)) {
            Proximity::Separated {
                distance, normal, ..
            } => {
                last_normal = -normal;
                if distance < CONTACT_EPS {
                    return Some((toi, last_normal));
                }
                // Stop just short of the surface. Advancing the full gap lands
                // the query point exactly on it, where the Minkowski difference
                // has no volume and EPA has nothing to expand — so the step that
                // should report the hit fails instead.
                toi += (distance - CONTACT_EPS * 0.5).max(CONTACT_EPS);
                if toi > max_toi {
                    return None;
                }
            }
            // Started inside, or arrived.
            Proximity::Penetrating { normal, .. } => return Some((toi, -normal)),
            // Degenerate configuration after we had already closed on the
            // surface: report where we got to rather than losing the hit.
            Proximity::Failed => return (toi > 0.0).then_some((toi, last_normal)),
        }
    }
    None
}

/// Conservative advancement: repeatedly step the cast shape forward by the
/// current gap, which can never take it through the target.
fn cast_against(
    cast: &impl SupportMap,
    target_shape: &Shape,
    target_iso: &Isometry,
    direction: Vector3,
    max_toi: f32,
) -> Option<(f32, Vector3, Vector3, Vector3)> {
    // A half-space has no bounded support function, so GJK will not take it and
    // `ShapeProxy::new` refuses. It does not need GJK: the cast shape's own
    // support toward the plane gives the gap in closed form and exactly. Without
    // this, a shape cast simply cannot see a ground plane — which is what the
    // character controller stands on.
    if let Shape::HalfSpace { normal } = target_shape {
        let n = try_normalize(target_iso.transform_vector(*normal))?;
        // Direction first. A character resting on the ground is touching this
        // plane, and reporting that as a hit at t = 0 for a *sideways* cast
        // would pin it in place and hide whatever it was walking into.
        let closing = -direction.dot(n);
        if closing <= 1e-6 {
            return None; // parallel to the plane, or moving away from it
        }
        let deepest = cast.support(-n);
        let gap = (deepest - target_iso.translation).dot(n);
        if gap <= CONTACT_EPS {
            // Already touching, or started inside, and moving further in.
            return Some((0.0, deepest, deepest - n * gap, n));
        }
        let toi = gap / closing;
        if toi > max_toi {
            return None;
        }
        let contact = deepest + direction * toi;
        return Some((toi, contact, contact, n));
    }

    let target = ShapeProxy::new(target_shape, target_iso)?;
    let mut toi = 0.0f32;

    for _ in 0..MAX_ADVANCE_ITERATIONS {
        let moved = TranslatedProxy {
            inner: cast,
            offset: direction * toi,
        };
        match closest_points(&moved, &target) {
            Proximity::Separated {
                distance,
                point_a,
                point_b,
                normal,
            } => {
                if distance < CONTACT_EPS {
                    return Some((toi, point_a, point_b, normal));
                }
                // How fast the gap closes as we advance. Non-positive means the
                // shapes are separating — they will never meet along this ray.
                let closing = -normal.dot(direction);
                if closing <= 1e-6 {
                    return None;
                }
                // Same near-miss margin as the ray version: stopping exactly on
                // the surface leaves EPA with a degenerate simplex.
                toi += ((distance - CONTACT_EPS * 0.5) / closing).max(CONTACT_EPS);
                if toi > max_toi {
                    return None;
                }
            }
            Proximity::Penetrating {
                point_a,
                point_b,
                normal,
                ..
            } => return Some((toi, point_a, point_b, normal)),
            Proximity::Failed => return None,
        }
    }
    None
}

/// Closest point on a shape's surface to `point`, plus whether it is inside.
fn project_onto_shape(
    shape: &Shape,
    iso: &Isometry,
    point: Vector3,
) -> Option<(Vector3, f32, bool)> {
    let local = iso.inverse_transform_point(point);
    if shape.contains_point_local(local) {
        return Some((point, 0.0, true));
    }
    match shape {
        Shape::Compound(parts) => {
            let mut best: Option<(Vector3, f32, bool)> = None;
            for (offset, child) in parts.iter() {
                if let Some(r) = project_onto_shape(child, &iso.mul(offset), point) {
                    if best.is_none_or(|b| r.1 < b.1) {
                        best = Some(r);
                    }
                }
            }
            best
        }
        Shape::TriMesh(mesh) => {
            let (surface, distance, _) = mesh.bvh().closest_point_to_point(local);
            Some((iso.transform_point(surface), distance, false))
        }
        Shape::HalfSpace { normal } => {
            let distance = normal.dot(local);
            Some((iso.transform_point(local - *normal * distance), distance.abs(), false))
        }
        _ => {
            let proxy = ShapeProxy::new(shape, iso)?;
            match closest_points(&proxy, &PointProxy(point)) {
                Proximity::Separated {
                    distance, point_a, ..
                } => Some((point_a, distance, false)),
                Proximity::Penetrating { point_a, .. } => Some((point_a, 0.0, true)),
                Proximity::Failed => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::collider::Collider;
    use crate::world::World;

    fn down_ray(x: f32, y: f32) -> Ray {
        Ray::new(Vector3::new(x, y, 0.0), Vector3::new(0.0, -1.0, 0.0))
    }

    #[test]
    fn a_ray_hits_a_ball_at_its_top_and_reports_the_right_normal() {
        let mut world = World::new();
        let ball = world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(1.0))
                .translation(Vector3::new(0.0, 0.0, 0.0)),
        );
        let hit = world.raycast(&down_ray(0.0, 5.0), 100.0, QueryFilter::default()).unwrap();
        assert_eq!(hit.body, ball);
        assert!((hit.toi - 4.0).abs() < 1e-3, "toi = {}", hit.toi);
        assert!((hit.point - Vector3::new(0.0, 1.0, 0.0)).length() < 1e-3);
        assert!(hit.normal.y > 0.99, "normal = {:?}", hit.normal);
    }

    #[test]
    fn a_ray_misses_what_is_not_there() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ball(1.0)));
        assert!(world.raycast(&down_ray(50.0, 5.0), 100.0, QueryFilter::default()).is_none());
        // And respects max_toi.
        assert!(world.raycast(&down_ray(0.0, 5.0), 1.0, QueryFilter::default()).is_none());
    }

    #[test]
    fn the_nearest_of_several_hits_wins() {
        let mut world = World::new();
        let near = world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 3.0, 0.0)),
        );
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        let hit = world.raycast(&down_ray(0.0, 10.0), 100.0, QueryFilter::default()).unwrap();
        assert_eq!(hit.body, near);
        assert_eq!(world.raycast_all(&down_ray(0.0, 10.0), 100.0, QueryFilter::default()).len(), 2);
    }

    #[test]
    fn filters_exclude_bodies_types_and_sensors() {
        let mut world = World::new();
        let dynamic = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 3.0, 0.0)),
        );
        let fixed = world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        let ray = down_ray(0.0, 10.0);

        // Excluding the near body falls through to the far one.
        let hit = world
            .raycast(&ray, 100.0, QueryFilter::default().exclude(dynamic))
            .unwrap();
        assert_eq!(hit.body, fixed);

        // Type filters.
        assert_eq!(
            world.raycast(&ray, 100.0, QueryFilter::default().only_fixed()).unwrap().body,
            fixed
        );
        assert_eq!(
            world.raycast(&ray, 100.0, QueryFilter::default().only_dynamic()).unwrap().body,
            dynamic
        );

        // Sensors are invisible by default, visible on request.
        let sensor = world.add_body(
            RigidBody::fixed()
                .collider(Collider::new(Shape::ball(0.5)).sensor(true))
                .translation(Vector3::new(0.0, 6.0, 0.0)),
        );
        assert_ne!(world.raycast(&ray, 100.0, QueryFilter::default()).unwrap().body, sensor);
        assert_eq!(
            world
                .raycast(&ray, 100.0, QueryFilter::default().include_sensors())
                .unwrap()
                .body,
            sensor
        );
    }

    #[test]
    fn a_ray_hits_every_shape_kind() {
        let shapes = [
            ("ball", Shape::ball(1.0), 1.0),
            ("cuboid", Shape::cuboid(1.0, 1.0, 1.0), 1.0),
            ("capsule", Shape::capsule(0.5, 0.5), 1.0),
            ("cylinder", Shape::cylinder(1.0, 1.0), 1.0),
            ("cone", Shape::cone(1.0, 1.0), 1.0),
        ];
        for (name, shape, top) in shapes {
            let mut world = World::new();
            world.add_body(RigidBody::fixed().shape(shape));
            let hit = world
                .raycast(&down_ray(0.0, 5.0), 100.0, QueryFilter::default())
                .unwrap_or_else(|| panic!("{name}: no hit"));
            assert!(
                (hit.toi - (5.0 - top)).abs() < 0.05,
                "{name}: toi {} expected {}",
                hit.toi,
                5.0 - top
            );
            assert!(hit.normal.y > 0.5, "{name}: normal {:?}", hit.normal);
        }
    }

    #[test]
    fn a_ray_hits_a_ground_plane_and_a_triangle_mesh() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let hit = world.raycast(&down_ray(3.0, 5.0), 100.0, QueryFilter::default()).unwrap();
        assert!((hit.toi - 5.0).abs() < 1e-3);
        assert!(hit.normal.y > 0.99);

        let mut world = World::new();
        world.add_body(
            RigidBody::fixed().shape(
                Shape::trimesh(
                    vec![
                        Vector3::new(-5.0, 0.0, -5.0),
                        Vector3::new(5.0, 0.0, -5.0),
                        Vector3::new(5.0, 0.0, 5.0),
                        Vector3::new(-5.0, 0.0, 5.0),
                    ],
                    vec![[0, 2, 1], [0, 3, 2]],
                )
                .unwrap(),
            ),
        );
        let hit = world.raycast(&down_ray(1.0, 5.0), 100.0, QueryFilter::default()).unwrap();
        assert!((hit.toi - 5.0).abs() < 1e-3, "toi = {}", hit.toi);
        assert!(hit.normal.y > 0.99, "normal = {:?}", hit.normal);
    }

    #[test]
    fn a_rotated_collider_is_hit_in_world_space() {
        let mut world = World::new();
        // A thin slab, rotated 90 degrees so it stands upright across the ray.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(2.0, 0.1, 2.0))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        let hit = world.raycast(&down_ray(0.0, 5.0), 100.0, QueryFilter::default()).unwrap();
        assert!((hit.toi - 3.9).abs() < 1e-2, "toi = {}", hit.toi);
    }

    #[test]
    fn ray_blocking_is_a_cheap_line_of_sight_test() {
        let mut world = World::new();
        assert!(!world.ray_is_blocked(&down_ray(0.0, 5.0), 100.0, QueryFilter::default()));
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(1.0, 1.0, 1.0))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        assert!(world.ray_is_blocked(&down_ray(0.0, 5.0), 100.0, QueryFilter::default()));
    }

    #[test]
    fn a_shape_cast_stops_at_the_obstacle() {
        let mut world = World::new();
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(0.5, 5.0, 5.0))
                .translation(Vector3::new(5.0, 0.0, 0.0)),
        );
        let hit = world
            .cast_shape(
                &Shape::ball(0.5),
                &Isometry::IDENTITY,
                Vector3::new(1.0, 0.0, 0.0),
                20.0,
                QueryFilter::default(),
            )
            .unwrap();
        // Ball surface at 0.5, wall face at 4.5 → 4.0 of travel.
        assert!((hit.toi - 4.0).abs() < 0.05, "toi = {}", hit.toi);
        assert!(hit.normal.x < -0.9, "normal = {:?}", hit.normal);
    }

    #[test]
    fn a_shape_cast_that_hits_nothing_returns_none() {
        let mut world = World::new();
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 50.0, 0.0)),
        );
        assert!(world
            .cast_shape(
                &Shape::ball(0.5),
                &Isometry::IDENTITY,
                Vector3::new(1.0, 0.0, 0.0),
                20.0,
                QueryFilter::default(),
            )
            .is_none());
        // Nor when the obstacle is behind the sweep.
        assert!(world
            .cast_shape(
                &Shape::ball(0.5),
                &Isometry::from_translation(Vector3::new(0.0, 50.0, 10.0)),
                Vector3::new(0.0, 0.0, 1.0),
                5.0,
                QueryFilter::default(),
            )
            .is_none());
    }

    #[test]
    fn overlap_tests_find_what_is_inside_the_query_shape() {
        let mut world = World::new();
        let inside = world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(0.0, 0.0, 0.0)),
        );
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(10.0, 0.0, 0.0)),
        );
        let found = world.intersections_with_shape(
            &Shape::ball(1.0),
            &Isometry::IDENTITY,
            QueryFilter::default(),
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, inside);
    }

    #[test]
    fn point_queries_report_containment_and_the_nearest_surface() {
        let mut world = World::new();
        let cube = world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(1.0, 1.0, 1.0))
                .translation(Vector3::new(0.0, 0.0, 0.0)),
        );
        assert_eq!(
            world.bodies_at_point(Vector3::new(0.5, 0.5, 0.5), QueryFilter::default()),
            vec![cube]
        );
        assert!(world
            .bodies_at_point(Vector3::new(9.0, 0.0, 0.0), QueryFilter::default())
            .is_empty());

        let outside = world
            .project_point(Vector3::new(4.0, 0.0, 0.0), QueryFilter::default())
            .unwrap();
        assert!(!outside.is_inside);
        assert!((outside.distance - 3.0).abs() < 1e-2, "distance = {}", outside.distance);
        assert!((outside.point - Vector3::new(1.0, 0.0, 0.0)).length() < 1e-2);

        let inside = world
            .project_point(Vector3::new(0.1, 0.0, 0.0), QueryFilter::default())
            .unwrap();
        assert!(inside.is_inside);
        assert_eq!(inside.distance, 0.0);
    }

    #[test]
    fn queries_see_bodies_where_they_are_after_stepping() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0)),
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        // The ball came to rest on the ground; a downward ray must meet it first.
        let hit = world.raycast(&down_ray(0.0, 10.0), 100.0, QueryFilter::default()).unwrap();
        assert!((hit.toi - 9.0).abs() < 0.1, "toi = {}", hit.toi);
    }

    #[test]
    fn a_ray_starting_inside_a_collider_still_reports_a_hit() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::cuboid(2.0, 2.0, 2.0)));
        let hit = world.raycast(&down_ray(0.0, 0.0), 100.0, QueryFilter::default());
        assert!(hit.is_some(), "a ray from inside a box found nothing");
    }
}
