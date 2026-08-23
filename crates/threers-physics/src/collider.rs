//! Colliders — a shape, where it sits on its body, and how it behaves on contact.

use crate::material::{InteractionGroups, PhysicsMaterial};
use crate::math::{Aabb, Isometry};
use crate::shape::{ColliderFit, MassProperties, Shape};
use threers::core::BufferGeometry;
use threers::math::{Quaternion, Vector3};

/// One collision shape attached to a body.
///
/// A body may carry several. Build them fluently:
///
/// ```
/// use threers_physics::prelude::*;
///
/// let wheel = Collider::new(Shape::cylinder(0.1, 0.4))
///     .translation(Vector3::new(0.0, -0.5, 0.0))
///     .friction(1.2)
///     .density(2.5);
/// assert_eq!(wheel.density, 2.5);
/// ```
#[derive(Debug, Clone)]
pub struct Collider {
    pub shape: Shape,
    /// Placement relative to the body's origin.
    pub local_transform: Isometry,
    pub material: PhysicsMaterial,
    /// Mass per unit volume, used when the body computes its own mass.
    pub density: f32,
    /// Sensors report overlaps but generate no contact response.
    pub is_sensor: bool,
    pub groups: InteractionGroups,
    pub enabled: bool,
    /// Free-form tag, echoed back on contact and query results.
    pub user_data: u64,
}

impl From<Shape> for Collider {
    fn from(shape: Shape) -> Self {
        Self::new(shape)
    }
}

impl Collider {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            local_transform: Isometry::IDENTITY,
            material: PhysicsMaterial::default(),
            density: 1.0,
            is_sensor: false,
            groups: InteractionGroups::default(),
            enabled: true,
            user_data: 0,
        }
    }

    pub fn translation(mut self, t: Vector3) -> Self {
        self.local_transform.translation = t;
        self
    }

    pub fn rotation(mut self, r: Quaternion) -> Self {
        self.local_transform.rotation = r;
        self
    }

    /// A collider fitted to `geometry` — the shape `fit` asks for, already
    /// placed where the mesh actually sits.
    ///
    /// `None` if the geometry has no usable positions, or if the fit needs a
    /// volume the mesh does not enclose.
    pub fn fit_to_geometry(geometry: &BufferGeometry, fit: ColliderFit) -> Option<Self> {
        let (shape, placement) = Shape::fit_to_geometry(geometry, fit)?;
        Some(Self::new(shape).transform(placement))
    }

    pub fn transform(mut self, iso: Isometry) -> Self {
        self.local_transform = iso;
        self
    }

    pub fn material(mut self, material: PhysicsMaterial) -> Self {
        self.material = material;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        self.material.friction = friction.max(0.0);
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.material.restitution = restitution.clamp(0.0, 1.0);
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = density.max(0.0);
        self
    }

    /// Detect overlaps without pushing anything — trigger volumes, pickups,
    /// water. Sensors still show up in `intersection_events`.
    pub fn sensor(mut self, is_sensor: bool) -> Self {
        self.is_sensor = is_sensor;
        self
    }

    pub fn groups(mut self, groups: InteractionGroups) -> Self {
        self.groups = groups;
        self
    }

    pub fn user_data(mut self, data: u64) -> Self {
        self.user_data = data;
        self
    }

    /// World transform, given the owning body's transform.
    #[inline]
    pub fn world_transform(&self, body: &Isometry) -> Isometry {
        body.mul(&self.local_transform)
    }

    /// World-space bounds, given the owning body's transform.
    pub fn compute_aabb(&self, body: &Isometry) -> Aabb {
        self.shape.compute_aabb(&self.world_transform(body))
    }

    /// Mass properties in the **body's** frame (local transform applied).
    pub fn mass_properties(&self) -> MassProperties {
        self.shape
            .mass_properties(self.density)
            .transformed_by(&self.local_transform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains_read_left_to_right() {
        let c = Collider::new(Shape::ball(1.0))
            .translation(Vector3::new(1.0, 2.0, 3.0))
            .friction(0.3)
            .restitution(0.9)
            .density(7.0)
            .sensor(true)
            .user_data(42);
        assert_eq!(c.local_transform.translation, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(c.material.friction, 0.3);
        assert_eq!(c.material.restitution, 0.9);
        assert_eq!(c.density, 7.0);
        assert!(c.is_sensor);
        assert_eq!(c.user_data, 42);
    }

    #[test]
    fn a_shape_converts_straight_into_a_collider() {
        let c: Collider = Shape::ball(0.5).into();
        assert_eq!(c.local_transform, Isometry::IDENTITY);
        assert_eq!(c.density, 1.0);
    }

    #[test]
    fn the_local_offset_moves_the_centre_of_mass() {
        let c = Collider::new(Shape::ball(1.0)).translation(Vector3::new(4.0, 0.0, 0.0));
        let mp = c.mass_properties();
        assert!((mp.center_of_mass - Vector3::new(4.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn world_aabb_follows_the_body() {
        let c = Collider::new(Shape::ball(1.0)).translation(Vector3::new(2.0, 0.0, 0.0));
        let body = Isometry::from_translation(Vector3::new(0.0, 5.0, 0.0));
        let aabb = c.compute_aabb(&body);
        assert!((aabb.center() - Vector3::new(2.0, 5.0, 0.0)).length() < 1e-5);
    }
}
