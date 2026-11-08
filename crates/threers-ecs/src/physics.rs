//! Rigid bodies on entities, from [`threers-physics`](threers_physics).
//!
//! ```
//! use threers::prelude::*;
//! use threers_ecs::physics::*;
//! use threers_ecs::prelude::*;
//! use threers_physics::prelude::{RigidBody, Shape};
//!
//! let mut app = App::new();
//! app.plugin(PhysicsPlugin::default());
//!
//! let floor = app.world.spawn(Object3D::group());
//! app.world.attach_body(floor, RigidBody::fixed().shape(Shape::ground()));
//!
//! let ball = app.world.spawn(Object3D::group());
//! app.world.scene.get_mut(ball).unwrap().position.y = 5.0;
//! app.world.attach_body(ball, RigidBody::dynamic().shape(Shape::ball(0.5)));
//!
//! for _ in 0..60 {
//!     app.frame(1.0 / 60.0);
//! }
//! assert!(app.world.scene.get(ball).unwrap().position.y < 5.0);
//! ```
//!
//! # Who owns the transform
//!
//! | Body | Moved by | Node |
//! |---|---|---|
//! | Dynamic | The solver | Written every frame. Move the body, not the node |
//! | Kinematic | The game | Read every step. Write `position` and the body follows, pushing what it meets |
//! | [`Character`] | [`character_step`] | Written every frame. Move it with [`Character::teleport`] |
//! | Fixed | Nothing | Left alone |
//!
//! A dynamic node holds the pose *drawn* this frame, which trails the simulation
//! by up to one step. Gameplay that needs the exact pose reads the body:
//! `w.resource::<Physics>().body(id)`.
//!
//! # Order
//!
//! [`physics_step`] is a fixed system and [`physics_sync`] a variable one. Order
//! your own against them: forces go `.before(physics_step)`, and a camera that
//! follows a body goes `.after(physics_sync)`. Whatever sets [`Character::walk`]
//! goes `.before(character_step)`. [`Collision`]s are events, so any system can
//! read them with [`World::events`].
//!
//! # Parents
//!
//! A body lives in world space. A node under a moved or rotated parent works:
//! its local fields are converted both ways. A scaled ancestor is refused, since
//! a rigid body has no way to be scaled.

use crate::app::{App, Plugin, Time};
use crate::query::Node;
use crate::schedule::System;
use crate::world::{Component, Entity, World};
use std::ops::{Deref, DerefMut};
use threers::math::{Quaternion, Vector3};

#[allow(unused_imports)]
use threers::easing::Easing;
pub use threers_physics::prelude::World as Solver;
/// All of the physics crate's prelude, so a game needs no import of its own. Not
/// its `World`, which the explicit import above hides: that name is the game's
/// [`World`], and the physics one is [`Solver`] here. `Easing` is hidden the same
/// way, for `threers`'s.
pub use threers_physics::prelude::*;

/// The solver, as a resource. Derefs to its `World` for gravity,
/// joints, raycasts and the rest of its API.
pub struct Physics(pub Solver);

impl Deref for Physics {
    type Target = Solver;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Physics {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An entity's rigid body. Removing this component removes the body, however it
/// goes: [`World::remove`], a despawn or a replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Body(pub BodyId);
impl Component for Body {}

/// Two bodies starting or stopping contact. An entity is `None` for a body added
/// straight to the solver with no node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Collision {
    /// One body's entity.
    pub a: Option<Entity>,
    /// The other body's entity.
    pub b: Option<Entity>,
    /// True when the pair began touching, false when it stopped.
    pub started: bool,
    /// One side is a sensor, so nothing was pushed.
    pub is_sensor: bool,
}

impl Collision {
    /// If `entity` is in this collision, what it hit.
    pub fn other(&self, entity: Entity) -> Option<Option<Entity>> {
        if self.a == Some(entity) {
            Some(self.b)
        } else if self.b == Some(entity) {
            Some(self.a)
        } else {
            None
        }
    }
}

/// Sets up [`Physics`], its systems, and the clean-up of bodies. The solver's own `timestep` is replaced by the app's fixed rate.
#[derive(Default)]
pub struct PhysicsPlugin {
    world: Solver,
}

impl PhysicsPlugin {
    /// Start from a solver that is already set up, with its own gravity or
    /// bodies.
    pub fn with_world(world: Solver) -> Self {
        Self { world }
    }
}

impl Plugin for PhysicsPlugin {
    fn build(self, app: &mut App) {
        app.resource(Physics(self.world))
            .fixed_system(physics_step)
            .fixed_system(System::new(character_step).before(physics_step))
            .fixed_system(System::new(crate::car::car_step).before(physics_step))
            .system(physics_sync)
            .system(System::new(crate::car::car_wheels).after(physics_sync));
        app.world.after_load(follow_load);
        app.world.on_remove::<Body>(|w, _, body| {
            let body = body.0;
            match w.get_resource_mut::<Physics>() {
                Some(physics) => {
                    physics.remove_body(body);
                }
                // Lent out by `with_resource`, which will have it back by the
                // time commands are next applied.
                None => w.commands().add(move |w| {
                    if let Some(physics) = w.get_resource_mut::<Physics>() {
                        physics.remove_body(body);
                    }
                }),
            }
        });
    }
}

/// A load moves nodes under the solver's feet: put bodies and characters back
/// with them, at rest.
fn follow_load(w: &mut World) {
    let mut moves = Vec::new();
    w.each_ref::<(Entity, &Body)>(|(id, body)| moves.push((id, body.0)));
    for (id, body) in moves {
        let Some(pose) = world_pose(w, id) else {
            continue;
        };
        if let Some(character) = w.get_mut::<Character>(id) {
            character.teleport(pose.translation);
        }
        if let Some(body) = w.resource_mut::<Physics>().body_mut(body) {
            body.set_position(pose);
            body.linear_velocity = Vector3::ZERO;
            body.angular_velocity = Vector3::ZERO;
        }
    }
}

/// Advance the solver one step, at the rate the [`App`] runs fixed systems.
///
/// The timestep is set on every call because the solver steps by its own, and
/// would otherwise keep running at 60 Hz whatever the app's rate.
pub fn physics_step(w: &mut World) {
    let timestep = w.resource::<Time>().delta;
    w.with_resource::<Physics, _>(|w, physics| {
        physics.timestep = timestep;
        physics.sync_from_scene(&w.scene.arena);
        physics.step_fixed();

        let entity = |body| physics.body(body).and_then(|b: &RigidBody| b.scene_object);
        let events: Vec<CollisionEvent> = physics.collision_events().to_vec();
        for event in events {
            w.send(Collision {
                a: entity(event.body_a),
                b: entity(event.body_b),
                started: event.started,
                is_sensor: event.is_sensor,
            });
        }
    });
}

/// Pose dynamic nodes for this frame, between the last two steps by
/// [`Time::alpha`], so motion stays smooth when the display outruns the solver.
/// Only dynamic ones: kinematic nodes belong to the game.
pub fn physics_sync(w: &mut World) {
    let alpha = w.resource::<Time>().alpha;
    w.with_resource::<Physics, _>(|w, physics| {
        physics.set_interpolation_alpha(alpha);
        physics.sync_to_scene_interpolated_where(&mut w.scene.arena, |_, body| body.is_dynamic());
        let mut poses = Vec::new();
        w.each_ref::<(Entity, &Character)>(|(id, character)| {
            let at = character.previous.lerp(character.position, alpha);
            poses.push((id, Isometry::from_translation(at)));
        });
        for (id, pose) in poses {
            place(w, id, pose);
        }
    });
}

/// Where `id`'s parent is, as a rigid transform.
fn parent_pose(w: &World, id: Entity) -> Isometry {
    let mut chain = Vec::new();
    let mut next = w.scene.get(id).and_then(|node| node.parent);
    while let Some(ancestor) = next {
        let Some(node) = w.scene.get(ancestor) else {
            break;
        };
        chain.push(Isometry::new(node.position, node.quaternion));
        next = node.parent;
    }
    chain
        .into_iter()
        .rev()
        .fold(Isometry::IDENTITY, |pose, local| pose.mul(&local))
}

/// Put a node at a world pose, through its local fields.
fn place(w: &mut World, id: Entity, pose: Isometry) {
    let local = parent_pose(w, id).inverse().mul(&pose);
    if let Some(node) = w.scene.get_mut(id) {
        node.position = local.translation;
        node.quaternion = local.rotation;
        node.update_matrix();
    }
}

/// Where a node is in the world.
fn world_pose(w: &World, id: Entity) -> Option<Isometry> {
    let node = w.scene.get(id)?;
    Some(parent_pose(w, id).mul(&Isometry::new(node.position, node.quaternion)))
}

/// A walking, jumping entity: a swept shape that slides along what it hits.
///
/// Set [`walk`](Self::walk) and call [`jump`](Self::jump); [`character_step`] does
/// the rest. Add one with [`AttachBody::attach_character`].
#[derive(Debug, Clone)]
pub struct Character {
    /// The shape, slope limit and step height it moves with.
    pub controller: CharacterController,
    /// Wanted velocity along the ground, in units per second.
    pub walk: Vector3,
    /// Current speed along the controller's `up`.
    pub vertical_speed: f32,
    /// Downward acceleration. Separate from the solver's gravity because a
    /// character that falls realistically usually feels floaty.
    pub gravity: f32,
    /// Standing on something after the last step.
    pub grounded: bool,
    /// Where the simulation has it. The node is drawn between this and
    /// `previous`, so it stays smooth on a display faster than the fixed rate.
    position: Vector3,
    previous: Vector3,
}

impl Component for Character {}

impl Character {
    /// Standing still, with a gravity of 9.81 units/s².
    pub fn new(controller: CharacterController) -> Self {
        Self {
            controller,
            walk: Vector3::ZERO,
            vertical_speed: 0.0,
            gravity: 9.81,
            grounded: false,
            position: Vector3::ZERO,
            previous: Vector3::ZERO,
        }
    }

    /// Where the simulation has it now, which the drawn node trails slightly.
    pub fn position(&self) -> Vector3 {
        self.position
    }

    /// Move straight to `position`, with no fall and no sweep along the way.
    pub fn teleport(&mut self, position: Vector3) {
        self.position = position;
        self.previous = position;
        self.vertical_speed = 0.0;
    }

    /// Leave the ground at `speed`. Does nothing in mid-air.
    pub fn jump(&mut self, speed: f32) {
        if self.grounded {
            self.vertical_speed = speed;
            self.grounded = false;
        }
    }
}

/// Move every [`Character`] for this step.
pub fn character_step(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.with_resource::<Physics, _>(|w, physics| {
        w.each::<(Node, &mut Character, Option<&Body>)>(|(node, character, body)| {
            character.vertical_speed -= character.gravity * dt;
            let up = character.controller.up;
            let wanted = (character.walk + up * character.vertical_speed) * dt;
            let start = Isometry::new(character.position, node.quaternion);
            let filter = QueryFilter::default().exclude(body.map(|b| b.0));
            let moved = character
                .controller
                .move_and_slide(physics, start, wanted, filter);

            character.previous = character.position;
            character.position = character.position + moved.translation;
            // Where the solver reads a kinematic body's target from. The frame's
            // sync puts the interpolated pose back for drawing.
            node.position = character.position;
            let falling = character.vertical_speed <= 0.0;
            character.grounded = moved.grounded;
            if moved.grounded && falling || moved.hit_ceiling && !falling {
                character.vertical_speed = 0.0;
            }
        });
    });
}

/// Attaching bodies to entities.
pub trait AttachBody {
    /// Give `entity` a rigid body, replacing any it had. Returns `None`, adding
    /// nothing, if the entity is dead.
    ///
    /// A body left at the origin starts at the node's transform. One given a
    /// position with `.translation()` or `.rotation()` keeps it, and the node
    /// moves there.
    ///
    /// # Panics
    ///
    /// If an ancestor is scaled or sets its own matrix, or there is no [`Physics`].
    fn attach_body(&mut self, entity: Entity, body: impl Into<RigidBody>) -> Option<BodyId>;

    /// Make `entity` a [`Character`], with a kinematic body of the same shape so
    /// that sensors and dynamic bodies can meet it. Same conditions as
    /// [`attach_body`](Self::attach_body).
    fn attach_character(
        &mut self,
        entity: Entity,
        controller: CharacterController,
    ) -> Option<BodyId>;

    /// Move a body and its node to a pose, at rest: for respawning, where a
    /// moved node alone would be pulled back by the solver. Returns whether
    /// `entity` has a body.
    fn teleport_body(&mut self, entity: Entity, position: Vector3, rotation: Quaternion) -> bool;

    /// An entity's rigid body, to read: its velocity, mass, sleep state.
    fn body_of(&self, entity: Entity) -> Option<&RigidBody>;

    /// An entity's rigid body, to push or change.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// let mut app = App::new();
    /// app.plugin(PhysicsPlugin::default());
    /// let ball = app.world.object(Object3D::group()).body(RigidBody::dynamic().shape(Shape::ball(0.5))).id();
    /// app.world.body_mut(ball).unwrap().apply_impulse(Vector3::new(0.0, 5.0, 0.0));
    /// ```
    fn body_mut(&mut self, entity: Entity) -> Option<&mut RigidBody>;
}

impl AttachBody for World {
    fn attach_body(&mut self, entity: Entity, body: impl Into<RigidBody>) -> Option<BodyId> {
        let mut body = body.into();
        let authored = world_pose(self, entity)?;
        assert!(
            ancestors_are_rigid(self, entity),
            "attach_body needs a node with no scaled ancestor: bodies cannot scale"
        );

        if body.position == Isometry::IDENTITY {
            body.set_position(authored);
        }
        let pose = body.position;
        body.scene_object = Some(entity);
        let id = self.resource_mut::<Physics>().add_body(body);

        place(self, entity, pose);
        self.insert(entity, Body(id));
        Some(id)
    }

    fn attach_character(
        &mut self,
        entity: Entity,
        controller: CharacterController,
    ) -> Option<BodyId> {
        let shape = controller.shape.clone();
        let id = self.attach_body(entity, RigidBody::kinematic().shape(shape))?;
        let mut character = Character::new(controller);
        character.teleport(world_pose(self, entity)?.translation);
        self.insert(entity, character);
        Some(id)
    }

    fn body_of(&self, entity: Entity) -> Option<&RigidBody> {
        let id = self.get::<Body>(entity)?.0;
        self.get_resource::<Physics>()?.body(id)
    }

    fn body_mut(&mut self, entity: Entity) -> Option<&mut RigidBody> {
        let id = self.get::<Body>(entity)?.0;
        self.get_resource_mut::<Physics>()?.body_mut(id)
    }

    fn teleport_body(&mut self, entity: Entity, position: Vector3, rotation: Quaternion) -> bool {
        let Some(id) = self.get::<Body>(entity).map(|body| body.0) else {
            return false;
        };
        let pose = Isometry::new(position, rotation);
        let Some(body) = self.resource_mut::<Physics>().body_mut(id) else {
            return false;
        };
        body.set_position(pose);
        body.linear_velocity = Vector3::ZERO;
        body.angular_velocity = Vector3::ZERO;
        place(self, entity, pose);
        true
    }
}

fn ancestors_are_rigid(w: &World, entity: Entity) -> bool {
    let mut next = w.scene.get(entity).and_then(|node| node.parent);
    while let Some(id) = next {
        let Some(node) = w.scene.get(id) else {
            return true;
        };
        // Without auto-update the node's matrix is its own, whatever these fields
        // say; `Scene::set_root_transform` works that way.
        if !node.matrix_auto_update || node.scale != Vector3::ONE {
            return false;
        }
        next = node.parent;
    }
    true
}
