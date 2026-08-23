//! Physics for OpenSCAD solids, without a round trip through STL.
//!
//! A [`threers::openscad::Solid`] evaluates straight to a [`threers::BufferGeometry`] — the same buffer the
//! renderer draws from — so there is no reason to write a mesh file and read it
//! back. This module attaches physics metadata to a solid and hands you both
//! halves at once: the geometry to render and the body to simulate, built from
//! the *same* evaluation.
//!
//! That shared evaluation is the point. Exporting to STL and re-importing gives
//! you two meshes that are nearly but not exactly the same, and "nearly" is
//! where a collider that does not match what you can see comes from.
//!
//! Requires the `openscad` feature.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//! use threers_physics::scad::SolidPhysics;
//! use threers::openscad::{cube, cylinder, difference_all};
//!
//! // A block with a hole through it — one CSG evaluation, used twice.
//! let part = difference_all(vec![
//!     cube([4.0, 2.0, 2.0]),
//!     cylinder(3.0, 0.5),
//! ]);
//!
//! let built = SolidPhysics::dynamic()
//!     .density(2.7)                       // aluminium, near enough
//!     .fit(ColliderFit::ConvexHull)
//!     .build(part)
//!     .unwrap();
//!
//! // `built.geometry` goes to a Mesh; `built.body` goes to the World.
//! let mut world = World::new();
//! let id = world.add_body(built.body);
//! assert!(world.body(id).unwrap().mass() > 0.0);
//! ```

use crate::body::{BodyType, RigidBody, RigidBodyBuilder};
use crate::collider::Collider;
use crate::material::PhysicsMaterial;
use crate::shape::{ColliderFit, Shape};
use threers::core::{BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId};
use threers::materials::Material;
use threers::math::Vector3;
use threers::openscad::Solid;

/// How a solid should behave once it is in the world.
///
/// This is the "meta layer": everything the physics engine needs to know about
/// a modelled part that the model itself does not say.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidPhysics {
    pub body_type: BodyType,
    /// How to turn the evaluated mesh into a collider.
    ///
    /// Defaults to [`ColliderFit::ConvexHull`] for movable parts and
    /// [`ColliderFit::TriMesh`] for fixed ones, which is almost always what you
    /// want: a triangle mesh is exact but hollow, so it belongs on scenery
    /// rather than on anything the solver has to push out of a wall.
    pub fit: ColliderFit,
    pub density: f32,
    /// Exact total mass, overriding density.
    pub mass: Option<f32>,
    pub material: PhysicsMaterial,
    pub translation: Vector3,
    /// Evaluate with the watertight exact-CSG kernel rather than the float one.
    ///
    /// Slower, and worth it when the boolean result feeds a triangle-mesh
    /// collider: a crack the renderer hides is a hole the collider does not.
    pub exact: bool,
}

impl Default for SolidPhysics {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            fit: ColliderFit::ConvexHull,
            density: 1.0,
            mass: None,
            material: PhysicsMaterial::default(),
            translation: Vector3::ZERO,
            exact: false,
        }
    }
}

impl SolidPhysics {
    /// A part the simulation moves.
    pub fn dynamic() -> Self {
        Self::default()
    }

    /// Immovable scenery. Defaults to an exact triangle-mesh collider, since
    /// that is both safe and free for something that never moves.
    pub fn fixed() -> Self {
        Self {
            body_type: BodyType::Fixed,
            fit: ColliderFit::TriMesh,
            ..Default::default()
        }
    }

    /// A part you move yourself that still pushes the simulated ones.
    pub fn kinematic() -> Self {
        Self {
            body_type: BodyType::Kinematic,
            fit: ColliderFit::ConvexHull,
            ..Default::default()
        }
    }

    pub fn fit(mut self, fit: ColliderFit) -> Self {
        self.fit = fit;
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = density.max(0.0);
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        self.mass = Some(mass.max(0.0));
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

    pub fn material(mut self, material: PhysicsMaterial) -> Self {
        self.material = material;
        self
    }

    /// Where to place the part in the world.
    pub fn translation(mut self, translation: Vector3) -> Self {
        self.translation = translation;
        self
    }

    /// Use the watertight exact-CSG kernel.
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Evaluate the solid once and produce both halves.
    ///
    /// `None` if the solid evaluates to nothing usable — an empty difference, or
    /// a shape too degenerate for the requested collider fit.
    pub fn build(&self, solid: Solid) -> Option<SolidPart> {
        let geometry = if self.exact {
            solid.to_geometry_exact()
        } else {
            solid.to_geometry()
        };
        self.build_from_geometry(geometry)
    }

    /// The same, for geometry that has already been evaluated.
    pub fn build_from_geometry(&self, geometry: BufferGeometry) -> Option<SolidPart> {
        let collider = Collider::fit_to_geometry(&geometry, self.fit)?
            .material(self.material)
            .density(self.density);

        let mut body = RigidBodyBuilder::new(self.body_type)
            .collider(collider)
            .translation(self.translation);
        if let Some(mass) = self.mass {
            body = body.mass(mass);
        }

        Some(SolidPart { geometry, body })
    }
}

/// One modelled part: what to draw, and what to simulate.
#[derive(Debug, Clone)]
pub struct SolidPart {
    /// The evaluated mesh. Hand it to a [`Mesh`] — no file goes near this.
    pub geometry: BufferGeometry,
    /// The body, already carrying a collider fitted to that same geometry.
    pub body: RigidBodyBuilder,
}

impl SolidPart {
    /// Add the mesh to a scene and the body to a world, linked so the body
    /// drives the node.
    ///
    /// Returns the scene node and the body handle.
    pub fn spawn(
        self,
        arena: &mut ObjectArena,
        parent: Option<ObjectId>,
        world: &mut crate::world::World,
        material: Material,
    ) -> (ObjectId, crate::body::BodyId) {
        let node = arena.insert(Object3D::mesh(Mesh::new(self.geometry, material)));
        if let Some(parent) = parent {
            arena.add_child(parent, node);
        }
        let body = world.add_body(self.body.scene_object(node));
        (node, body)
    }

    /// The body on its own, for callers doing their own rendering.
    pub fn into_body(self) -> RigidBody {
        self.body.build()
    }
}

/// Build a collider straight from a solid, without a body.
///
/// For attaching several modelled parts to one rigid assembly.
pub fn collider_from_solid(solid: Solid, fit: ColliderFit, exact: bool) -> Option<Collider> {
    let geometry = if exact {
        solid.to_geometry_exact()
    } else {
        solid.to_geometry()
    };
    Collider::fit_to_geometry(&geometry, fit)
}

/// Build a shape straight from a solid.
pub fn shape_from_solid(solid: Solid, fit: ColliderFit, exact: bool) -> Option<Shape> {
    let geometry = if exact {
        solid.to_geometry_exact()
    } else {
        solid.to_geometry()
    };
    Shape::fit_to_geometry(&geometry, fit).map(|(shape, _)| shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;
    use threers::openscad::{cube, cylinder, difference_all, sphere, union_all};

    #[test]
    fn a_solid_becomes_a_mesh_and_a_body_from_one_evaluation() {
        let part = SolidPhysics::dynamic()
            .density(2.0)
            .build(cube([2.0, 2.0, 2.0]))
            .unwrap();

        // The geometry is real and drawable.
        let positions = part.geometry.get_attribute("position").unwrap();
        assert!(positions.count() > 0);

        // And the body has mass derived from that same shape: a 2x2x2 cube at
        // density 2 weighs 16.
        let body = part.into_body();
        assert!((body.mass() - 16.0).abs() < 0.5, "mass was {}", body.mass());
    }

    #[test]
    fn a_csg_result_produces_a_collider_that_matches_it() {
        // A block with a hole: the convex hull fills the hole in, which is the
        // documented behaviour and why the fit is a choice.
        let drilled = difference_all(vec![
            cube([4.0, 2.0, 2.0]),
            cylinder(3.0, 0.4),
        ]);
        let part = SolidPhysics::dynamic()
            .fit(ColliderFit::ConvexHull)
            .build(drilled)
            .unwrap();
        let body = part.into_body();
        assert_eq!(body.colliders().len(), 1);
        // The hull of a 4x2x2 block is the block.
        assert!((body.mass() - 16.0).abs() < 1.0, "mass was {}", body.mass());
    }

    #[test]
    fn every_fit_works_on_a_modelled_part() {
        for fit in [
            ColliderFit::TriMesh,
            ColliderFit::ConvexHull,
            ColliderFit::Box,
            ColliderFit::Ball,
            ColliderFit::Capsule,
            ColliderFit::Cylinder,
        ] {
            let part = SolidPhysics::fixed()
                .fit(fit)
                .build(cube([2.0, 4.0, 2.0]));
            assert!(part.is_some(), "{fit:?} produced nothing");
        }
    }

    #[test]
    fn a_modelled_part_falls_and_lands_in_a_world() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));

        let part = SolidPhysics::dynamic()
            .translation(Vector3::new(0.0, 5.0, 0.0))
            .build(cube([1.0, 1.0, 1.0]))
            .unwrap();
        let id = world.add_body(part.body);

        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let y = world.body(id).unwrap().translation().y;
        assert!((y - 0.5).abs() < 0.06, "the part rested at {y}, expected 0.5");
    }

    #[test]
    fn a_fixed_part_defaults_to_an_exact_mesh_collider() {
        let meta = SolidPhysics::fixed();
        assert_eq!(meta.fit, ColliderFit::TriMesh);
        assert_eq!(meta.body_type, BodyType::Fixed);

        // And a dynamic one defaults to a hull, which is safe to move.
        assert_eq!(SolidPhysics::dynamic().fit, ColliderFit::ConvexHull);
    }

    #[test]
    fn a_part_spawns_into_a_scene_and_a_world_together() {
        use threers::materials::BasicMaterial;
        use threers::math::Color;

        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));

        let part = SolidPhysics::dynamic()
            .translation(Vector3::new(0.0, 4.0, 0.0))
            .build(sphere(0.5))
            .unwrap();
        let (node, body) = part.spawn(
            &mut arena,
            Some(root),
            &mut world,
            Material::Basic(BasicMaterial::new(Color::from_hex(0xcccccc))),
        );

        // The body drives the node.
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        world.sync_to_scene(&mut arena);
        let drawn = arena.get(node).unwrap().position.y;
        let simulated = world.body(body).unwrap().translation().y;
        assert!((drawn - simulated).abs() < 1e-5);
        assert!(simulated < 4.0, "it never fell");
    }

    #[test]
    fn material_and_mass_overrides_reach_the_body() {
        let part = SolidPhysics::dynamic()
            .mass(42.0)
            .friction(1.3)
            .restitution(0.75)
            .build(cube([1.0, 1.0, 1.0]))
            .unwrap();
        let body = part.into_body();
        assert!((body.mass() - 42.0).abs() < 1e-3);
        let collider = &body.colliders()[0];
        assert_eq!(collider.material.friction, 1.3);
        assert_eq!(collider.material.restitution, 0.75);
    }

    #[test]
    fn an_assembly_can_share_one_body() {
        // Several modelled parts welded into a single rigid body.
        let mut builder = RigidBody::dynamic();
        for (x, size) in [(-1.5f32, 0.5f32), (0.0, 0.8), (1.5, 0.5)] {
            let collider =
                collider_from_solid(cube([size, size, size]), ColliderFit::Box, false)
                    .unwrap()
                    .translation(Vector3::new(x, 0.0, 0.0));
            builder = builder.collider(collider);
        }
        let body = builder.build();
        assert_eq!(body.colliders().len(), 3);
        assert!(body.mass() > 0.0);
        // The centre of mass sits at the middle, where the big piece is.
        assert!(body.local_center_of_mass().x.abs() < 0.1);
    }

    #[test]
    fn a_union_of_solids_is_one_part() {
        let welded = union_all(vec![
            cube([1.0, 1.0, 1.0]),
            sphere(0.7),
        ]);
        let part = SolidPhysics::dynamic().build(welded).unwrap();
        assert!(part.geometry.get_attribute("position").unwrap().count() > 0);
        assert!(part.into_body().mass() > 0.0);
    }
}
