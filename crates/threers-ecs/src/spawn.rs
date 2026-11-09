//! Spawning a node and setting it up in one expression.

use crate::world::{Bundle, Entity, World};
use std::sync::Arc;
use threers::core::{BufferGeometry, Mesh, Object3D, ObjectKind};
use threers::lights::DirectionalLight;
use threers::materials::Material;
use threers::math::{Color, Quaternion, Vector3};

/// A node just spawned, to finish setting up. Dropping it leaves the node as it
/// stands, so [`id`](Self::id) is needed only to keep the entity.
///
/// ```
/// # use threers::prelude::*;
/// # use threers_ecs::prelude::*;
/// struct Crate;
/// impl Component for Crate {}
///
/// let mut w = World::new();
/// let id = w
///     .mesh(BoxGeometry::new(1.0, 1.0, 1.0), StandardMaterial::new(Color::WHITE))
///     .at(Vector3::new(0.0, 2.0, 0.0))
///     .named("crate")
///     .with(Crate)
///     .id();
/// assert!(w.has::<Crate>(id));
/// ```
pub struct Spawned<'w> {
    pub(crate) world: &'w mut World,
    pub(crate) id: Entity,
}

impl Spawned<'_> {
    /// Put it at `position`, relative to its parent.
    pub fn at(self, position: Vector3) -> Self {
        if let Some(node) = self.world.scene.get_mut(self.id) {
            node.position = position;
        }
        self
    }

    /// Turn it to `rotation`.
    pub fn rotated(self, rotation: Quaternion) -> Self {
        if let Some(node) = self.world.scene.get_mut(self.id) {
            node.quaternion = rotation;
        }
        self
    }

    /// Name it, as the inspector and the console's `select` show it.
    pub fn named(self, name: impl Into<String>) -> Self {
        if let Some(node) = self.world.scene.get_mut(self.id) {
            node.name = name.into();
        }
        self
    }

    /// Whether it casts and receives shadows.
    pub fn shadows(self, cast: bool, receive: bool) -> Self {
        if let Some(node) = self.world.scene.get_mut(self.id) {
            node.cast_shadow = cast;
            node.receive_shadow = receive;
        }
        self
    }

    /// Mark it scenery that will not move, for [`World::bake_static`] to merge
    /// with the rest of its kind.
    pub fn static_mesh(self) -> Self {
        self.world.insert(self.id, crate::batch::Static);
        self
    }

    /// Add components, as [`World::insert`] does.
    pub fn with(self, bundle: impl Bundle) -> Self {
        self.world.insert(self.id, bundle);
        self
    }

    /// Give it a rigid body, after its transform is set.
    #[cfg(feature = "physics")]
    pub fn body(self, body: impl Into<threers_physics::prelude::RigidBody>) -> Self {
        use crate::physics::AttachBody;
        self.world.attach_body(self.id, body);
        self
    }

    /// Make it solid: a fixed body shaped exactly like its mesh, for level
    /// geometry. Does nothing for a node with no mesh or an empty one.
    #[cfg(feature = "physics")]
    pub fn solid(self) -> Self {
        use crate::physics::{AttachBody, RigidBody, Shape};
        let shape = match self.world.scene.get(self.id).map(|node| &node.kind) {
            Some(threers::core::ObjectKind::Mesh(mesh)) => {
                Shape::trimesh_from_geometry(&mesh.geometry)
            }
            _ => None,
        };
        if let Some(shape) = shape {
            self.world
                .attach_body(self.id, RigidBody::fixed().shape(shape));
        }
        self
    }

    /// Make it fixed level geometry with a collider fitted to what it draws,
    /// the way `fit` asks: `ColliderFit::Box` for a crate, `TriMesh` for a
    /// floor (which is what [`solid`](Self::solid) does).
    #[cfg(feature = "physics")]
    pub fn solid_as(self, fit: threers_physics::prelude::ColliderFit) -> Self {
        use threers_physics::prelude::RigidBody;
        self.body_from_mesh(RigidBody::fixed(), fit)
    }

    /// Give it `body` with a collider fitted to everything it draws, its own
    /// mesh and every mesh under it: a car made of several parts gets one
    /// hull the shape of the car. Does nothing for a node that draws nothing.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// let mut app = App::new();
    /// app.plugin(PhysicsPlugin::default());
    /// let barrel = app
    ///     .world
    ///     .mesh(CylinderGeometry::new(0.5, 0.5, 1.2, 16, 1, false, 0.0, 6.283), StandardMaterial::new(Color::WHITE))
    ///     .at(Vector3::new(0.0, 3.0, 0.0))
    ///     .body_from_mesh(RigidBody::dynamic().mass(40.0), ColliderFit::ConvexHull)
    ///     .id();
    /// assert!(app.world.has::<Body>(barrel));
    /// ```
    #[cfg(feature = "physics")]
    pub fn body_from_mesh(
        self,
        body: threers_physics::prelude::RigidBodyBuilder,
        fit: threers_physics::prelude::ColliderFit,
    ) -> Self {
        use crate::physics::AttachBody;
        use threers_physics::prelude::Collider;
        let collider =
            drawn_geometry(self.world, self.id).and_then(|g| Collider::fit_to_geometry(&g, fit));
        if let Some(collider) = collider {
            self.world.attach_body(self.id, body.collider(collider));
        }
        self
    }

    /// Make it a car, after its transform is set.
    #[cfg(feature = "physics")]
    pub fn car(self, spec: crate::car::CarSpec) -> Self {
        use crate::car::AttachCar;
        self.world.attach_car(self.id, spec);
        self
    }

    /// Play `player` on this mesh: its texture becomes the material's `map`,
    /// and the app moves it on with game time, sound and all (with the `audio`
    /// feature and plugin). The material is copied for this mesh first if
    /// others share it. Needs the `video` feature. For a silent one, insert
    /// [`Video::muted`](crate::video::Video::muted) instead.
    #[cfg(feature = "video")]
    pub fn video(self, player: crate::video::VideoPlayer) -> Self {
        if let Some(node) = self.world.scene.get_mut(self.id)
            && let ObjectKind::Mesh(mesh) = &mut node.kind
            && let Some(slot) = std::sync::Arc::make_mut(&mut mesh.material).map_slot_mut("map")
        {
            *slot = Some(player.texture());
        }
        self.world.insert(self.id, crate::video::Video::new(player));
        self
    }

    /// The entity, to keep.
    pub fn id(self) -> Entity {
        self.id
    }
}

/// Every mesh at and under `root`, as one geometry in `root`'s own frame.
#[cfg(feature = "physics")]
pub(crate) fn drawn_geometry(world: &World, root: Entity) -> Option<BufferGeometry> {
    use threers::geometries::MeshBuilder;
    use threers::math::Matrix4;
    fn visit(world: &World, id: Entity, frame: Matrix4, out: &mut MeshBuilder, any: &mut bool) {
        let Some(node) = world.scene.get(id) else {
            return;
        };
        if let ObjectKind::Mesh(mesh) = &node.kind {
            out.append(&mesh.geometry, &frame);
            *any = true;
        }
        for &child in &node.children {
            if let Some(c) = world.scene.get(child) {
                let local = Matrix4::compose(c.position, c.quaternion, c.scale);
                visit(world, child, frame.multiply(&local), out, any);
            }
        }
    }
    let mut builder = MeshBuilder::new();
    let mut any = false;
    visit(world, root, Matrix4::identity(), &mut builder, &mut any);
    any.then(|| builder.build())
}

impl World {
    /// Spawn a mesh, to finish with [`Spawned`]'s methods.
    ///
    /// `geometry` is a built-in shape or a [`MeshBuilder`](threers::geometries::MeshBuilder).
    pub fn mesh(
        &mut self,
        geometry: impl Into<BufferGeometry>,
        material: impl Into<Material>,
    ) -> Spawned<'_> {
        let id = self.spawn(Object3D::mesh(Mesh::new(geometry.into(), material.into())));
        Spawned { world: self, id }
    }

    /// Change a mesh's geometry while the game runs, and redraw it. Returns `false`
    /// if `id` is not a mesh.
    ///
    /// Nothing else has to be told: the GPU copy and bounds follow. A collider made
    /// from the old shape does not, since rebuilding one costs far more than this.
    pub fn edit_mesh(&mut self, id: Entity, edit: impl FnOnce(&mut BufferGeometry)) -> bool {
        let Some(node) = self.scene.get_mut(id) else {
            return false;
        };
        let ObjectKind::Mesh(mesh) = &mut node.kind else {
            return false;
        };
        let geometry = Arc::make_mut(&mut mesh.geometry);
        edit(geometry);
        geometry.bump_version();
        geometry.compute_bounding_box();
        geometry.compute_bounding_sphere();
        true
    }

    /// A sun for a game: a directional light with shadows that follow the camera.
    ///
    /// A plain `DirectionalLight` casts nothing until told to, and then only
    /// over a few units around the origin, which suits one object and not a
    /// level. This one covers `reach` units from the camera, in cascades.
    pub fn sun(&mut self, direction: Vector3, intensity: f32, reach: f32) -> Entity {
        let mut sun = DirectionalLight::new(Color::WHITE, intensity);
        sun.direction = direction.normalize();
        sun.cast_shadow = true;
        sun.shadow.cascades = 3;
        sun.shadow.cascade_distance = reach;
        self.scene.add_light(sun)
    }

    /// Set up an entity that already exists with [`Spawned`]'s methods: give a
    /// loaded node a body, a name or components.
    ///
    /// # Panics
    ///
    /// If `id` is not alive: the methods would quietly do nothing.
    pub fn entity(&mut self, id: Entity) -> Spawned<'_> {
        assert!(self.is_alive(id), "entity {id:?} does not exist");
        Spawned { world: self, id }
    }

    /// Spawn any node, to finish with [`Spawned`]'s methods.
    pub fn object(&mut self, object: Object3D) -> Spawned<'_> {
        let id = self.spawn(object);
        Spawned { world: self, id }
    }
}
