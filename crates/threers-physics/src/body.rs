//! Rigid bodies and the arena that stores them.

use crate::collider::Collider;
use crate::material::PhysicsMaterial;
use crate::math::{Aabb, Isometry, Mat3};
use crate::shape::{MassProperties, Shape};
use threers::core::ObjectId;
use threers::math::{Quaternion, Vector3};

/// How a body participates in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyType {
    /// Moved by forces, gravity and contacts.
    #[default]
    Dynamic,
    /// Moved only by you, but pushes dynamic bodies out of the way. Behaves as
    /// though it had infinite mass. Use for lifts, doors and animated props.
    Kinematic,
    /// Never moves. Use for level geometry.
    Fixed,
}

/// Degrees of freedom removed from a body.
///
/// ```
/// use threers_physics::prelude::*;
///
/// // The classic upright character capsule.
/// let axes = LockedAxes::ROTATION;
/// assert!(axes.is_rotation_locked(1));
///
/// // A 2D-in-3D setup: movement confined to the XY plane.
/// let planar = LockedAxes::TRANSLATION_Z | LockedAxes::ROTATION_X | LockedAxes::ROTATION_Y;
/// assert!(planar.is_translation_locked(2));
/// assert!(!planar.is_rotation_locked(2));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LockedAxes(u8);

impl LockedAxes {
    pub const NONE: Self = Self(0);
    pub const TRANSLATION_X: Self = Self(1 << 0);
    pub const TRANSLATION_Y: Self = Self(1 << 1);
    pub const TRANSLATION_Z: Self = Self(1 << 2);
    pub const ROTATION_X: Self = Self(1 << 3);
    pub const ROTATION_Y: Self = Self(1 << 4);
    pub const ROTATION_Z: Self = Self(1 << 5);
    /// All three translation axes.
    pub const TRANSLATION: Self = Self(0b000_111);
    /// All three rotation axes.
    pub const ROTATION: Self = Self(0b111_000);
    /// Completely frozen, while still colliding.
    pub const ALL: Self = Self(0b111_111);

    #[inline]
    pub fn is_translation_locked(self, axis: usize) -> bool {
        self.0 & (1 << axis) != 0
    }

    #[inline]
    pub fn is_rotation_locked(self, axis: usize) -> bool {
        self.0 & (1 << (axis + 3)) != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for LockedAxes {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for LockedAxes {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A simulated rigid body.
///
/// Build one with [`RigidBody::dynamic`], [`RigidBody::fixed`] or
/// [`RigidBody::kinematic`], then hand it to [`crate::world::World::add_body`].
///
/// ```
/// use threers_physics::prelude::*;
///
/// let ball = RigidBody::dynamic()
///     .translation(Vector3::new(0.0, 10.0, 0.0))
///     .shape(Shape::ball(0.5))
///     .restitution(0.7)
///     .build();
/// assert!(ball.mass() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub body_type: BodyType,
    /// World transform of the body's **origin** — not its centre of mass. This
    /// is the transform that maps onto a scene node.
    pub position: Isometry,
    /// Linear velocity of the centre of mass, in world space.
    pub linear_velocity: Vector3,
    /// Angular velocity in world space, in radians per second. Its direction is
    /// the rotation axis and its length the rate.
    pub angular_velocity: Vector3,
    /// Fraction of linear velocity shed per second. `0` never slows down.
    pub linear_damping: f32,
    pub angular_damping: f32,
    /// Multiplier on world gravity. `0` floats, `-1` falls upward.
    pub gravity_scale: f32,
    pub locked_axes: LockedAxes,
    /// Sweep this body against obstacles instead of teleporting it each step.
    ///
    /// The discrete pass only ever asks whether two shapes overlap *now*, so a
    /// body that moves further in one step than the thing in its way is thick
    /// is already on the far side by the time anything is tested — and no
    /// amount of solver iteration finds a contact that was never generated.
    /// With this on, the step is swept instead: cast the shape along the motion
    /// it just made, and put it back at the impact point if it hit something.
    /// The response is left to the ordinary contact solver on the next step.
    ///
    /// Only steps longer than the body is thick are swept; below that the
    /// discrete pass could not have missed anything and the cast would be pure
    /// cost. Worth it for bullets and fast small objects, not for a crate.
    pub ccd_enabled: bool,
    /// Whether this body may be put to sleep when it comes to rest.
    pub can_sleep: bool,
    pub enabled: bool,
    /// Free-form tag, echoed back on contacts and query hits.
    pub user_data: u64,
    /// Scene node driven by this body. Set it and [`crate::world::World::sync_to_scene`]
    /// writes the transform back every step.
    pub scene_object: Option<ObjectId>,

    colliders: Vec<Collider>,
    mass_override: Option<f32>,
    /// Balance point, overriding what the colliders imply. A weeble rights
    /// itself because its mass sits below its centre.
    com_override: Option<Vector3>,
    /// Principal moments about the local axes, overriding the collider's. Lets a
    /// body resist spin about one axis far more than its shape suggests.
    inertia_override: Option<Vector3>,

    mass: f32,
    inv_mass: f32,
    local_com: Vector3,
    local_inv_inertia: Mat3,
    world_inv_inertia: Mat3,

    force: Vector3,
    torque: Vector3,

    sleeping: bool,
    sleep_timer: f32,

    /// Transform at the start of the current step, for CCD and render
    /// interpolation.
    previous_position: Isometry,

    /// Where a kinematic body has been told to be by the end of the next step.
    kinematic_target: Option<Isometry>,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            position: Isometry::IDENTITY,
            linear_velocity: Vector3::ZERO,
            angular_velocity: Vector3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            locked_axes: LockedAxes::NONE,
            ccd_enabled: false,
            can_sleep: true,
            enabled: true,
            user_data: 0,
            scene_object: None,
            colliders: Vec::new(),
            mass_override: None,
            com_override: None,
            inertia_override: None,
            mass: 0.0,
            inv_mass: 0.0,
            local_com: Vector3::ZERO,
            local_inv_inertia: Mat3::ZERO,
            world_inv_inertia: Mat3::ZERO,
            force: Vector3::ZERO,
            torque: Vector3::ZERO,
            sleeping: false,
            sleep_timer: 0.0,
            previous_position: Isometry::IDENTITY,
            kinematic_target: None,
        }
    }
}

impl RigidBody {
    /// Start building a body moved by forces and contacts.
    pub fn dynamic() -> RigidBodyBuilder {
        RigidBodyBuilder::new(BodyType::Dynamic)
    }

    /// Start building an immovable body.
    pub fn fixed() -> RigidBodyBuilder {
        RigidBodyBuilder::new(BodyType::Fixed)
    }

    /// Start building a body you move yourself that still pushes dynamic bodies.
    pub fn kinematic() -> RigidBodyBuilder {
        RigidBodyBuilder::new(BodyType::Kinematic)
    }

    // ---- classification ---------------------------------------------------

    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType::Kinematic
    }

    pub fn is_fixed(&self) -> bool {
        self.body_type == BodyType::Fixed
    }

    /// Can this body be moved by the solver at all?
    pub fn is_movable(&self) -> bool {
        self.is_dynamic() && self.enabled
    }

    /// Moments of inertia about the local axes. Infinite moments read as zero,
    /// which is what a fixed body has.
    pub fn principal_inertia(&self) -> Vector3 {
        let inv = self.local_inv_inertia.diagonal();
        let axis = |i: f32| if i > 0.0 { 1.0 / i } else { 0.0 };
        Vector3::new(axis(inv.x), axis(inv.y), axis(inv.z))
    }

    /// Change what kind of body this is, at runtime.
    ///
    /// Mass properties are recomputed, because a fixed body has none and a
    /// dynamic one needs them — a ragdoll switching from kinematic to dynamic
    /// would otherwise wake up massless and fall through the floor. The body is
    /// woken too: a sleeping body that has just changed kind is not at rest in
    /// the sense sleeping means.
    pub fn set_body_type(&mut self, body_type: BodyType) {
        if self.body_type == body_type {
            return;
        }
        self.body_type = body_type;
        self.recompute_mass_properties();
        self.wake_up();
    }

    /// How long this body has been still, in seconds — the timer that decides
    /// when it falls asleep.
    pub fn rest_time(&self) -> f32 {
        self.sleep_timer
    }

    /// Drive a kinematic body to `target` by the end of the next step.
    ///
    /// This is how a kinematic body should be moved: writing
    /// [`position`](Self::position) directly teleports it, so it arrives without
    /// ever having had a velocity and sweeps nothing out of its way. Given a
    /// target, the body works out the velocity that lands it there and pushes
    /// what it meets — while still arriving exactly, whatever it was carrying.
    pub fn set_kinematic_target(&mut self, target: Isometry) {
        self.kinematic_target = Some(target);
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    /// The pose this body was last told to reach, if any.
    pub fn kinematic_target(&self) -> Option<Isometry> {
        self.kinematic_target
    }

    /// Turn a kinematic target into the velocity that reaches it in `dt`.
    ///
    /// Consumed here rather than applied as a teleport, so the body carries a
    /// real velocity through the step and the solver can push things with it.
    pub(crate) fn drive_to_kinematic_target(&mut self, dt: f32) {
        if dt <= 0.0 || !self.is_kinematic() {
            return;
        }
        let Some(target) = self.kinematic_target.take() else {
            return;
        };

        self.linear_velocity = (target.translation - self.position.translation) * (1.0 / dt);

        // The turn from here to there, as an axis and an angle. `-q` and `q` are
        // the same orientation, so take the shorter of the two or a small turn
        // reads as one just under a full revolution the other way.
        let mut delta = target.rotation.multiply(self.position.rotation.conjugate());
        if delta.w < 0.0 {
            delta = Quaternion::new(-delta.x, -delta.y, -delta.z, -delta.w);
        }
        let sin_half = (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z).sqrt();
        self.angular_velocity = if sin_half > 1e-6 {
            let angle = 2.0 * sin_half.atan2(delta.w.clamp(-1.0, 1.0));
            Vector3::new(delta.x, delta.y, delta.z) * (angle / (sin_half * dt))
        } else {
            Vector3::ZERO
        };
    }

    /// Put a captured state back, derived quantities included.
    ///
    /// Everything here is state a step reads, so restoring all of it is what
    /// makes a rollback resume the same simulation rather than a similar one:
    /// leaving out the sleep timer, say, wakes a body a fraction of a second
    /// later than it originally woke.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_state(
        &mut self,
        position: Isometry,
        previous_position: Isometry,
        linear_velocity: Vector3,
        angular_velocity: Vector3,
        body_type: BodyType,
        sleeping: bool,
        rest_time: f32,
        kinematic_target: Option<Isometry>,
        enabled: bool,
    ) {
        self.position = position;
        self.previous_position = previous_position;
        self.linear_velocity = linear_velocity;
        self.angular_velocity = angular_velocity;
        self.body_type = body_type;
        self.sleeping = sleeping;
        self.sleep_timer = rest_time;
        self.kinematic_target = kinematic_target;
        self.enabled = enabled;
        // Forces are per-step and were already consumed by the step that this
        // state was captured before; carrying them over would apply them twice.
        self.force = Vector3::ZERO;
        self.torque = Vector3::ZERO;
        self.update_world_inertia();
    }

    // ---- mass -------------------------------------------------------------

    /// The single density that reproduces this body's mass over its own volume.
    ///
    /// For the usual uniform body this is just the collider's density; for a
    /// compound it is the mass-weighted blend, so `mass() / effective_density()`
    /// is the displaced volume whatever the body is made of. Zero if nothing
    /// here has a volume to speak of — a half-space, or a mass set by hand.
    pub fn effective_density(&self) -> f32 {
        let mut mass = 0.0;
        let mut volume = 0.0;
        for c in self.colliders.iter().filter(|c| c.enabled && !c.is_sensor) {
            let m = c.mass_properties().mass;
            if m > 0.0 && c.density > 0.0 {
                mass += m;
                volume += m / c.density;
            }
        }
        if volume > 0.0 {
            mass / volume
        } else {
            0.0
        }
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }

    pub fn inv_mass(&self) -> f32 {
        self.inv_mass
    }

    /// Per-axis inverse mass, with locked translation axes zeroed.
    #[inline]
    pub fn inv_mass_axes(&self) -> Vector3 {
        Vector3::new(
            if self.locked_axes.is_translation_locked(0) { 0.0 } else { self.inv_mass },
            if self.locked_axes.is_translation_locked(1) { 0.0 } else { self.inv_mass },
            if self.locked_axes.is_translation_locked(2) { 0.0 } else { self.inv_mass },
        )
    }

    /// World-space inverse inertia tensor, with locked rotation axes zeroed.
    #[inline]
    pub fn world_inv_inertia(&self) -> Mat3 {
        self.world_inv_inertia
    }

    /// Centre of mass in the body's local frame.
    pub fn local_center_of_mass(&self) -> Vector3 {
        self.local_com
    }

    /// Centre of mass in world space — the point the body actually rotates about.
    #[inline]
    pub fn world_center_of_mass(&self) -> Vector3 {
        self.position.transform_point(self.local_com)
    }

    /// Move the body so its centre of mass lands on `com`.
    pub fn set_world_center_of_mass(&mut self, com: Vector3) {
        self.position.translation = com - self.local_com.apply_quaternion(self.position.rotation);
    }

    /// Override the total mass, keeping the inertia *distribution* from the
    /// colliders. Pass `None` to go back to density-derived mass.
    pub fn set_mass(&mut self, mass: Option<f32>) {
        self.mass_override = mass.map(|m| m.max(0.0));
        self.recompute_mass_properties();
    }

    /// Recompute mass, centre of mass and inertia from the colliders. Call this
    /// after changing a collider's shape, density or local transform.
    pub fn recompute_mass_properties(&mut self) {
        if !self.is_dynamic() {
            self.mass = 0.0;
            self.inv_mass = 0.0;
            self.local_inv_inertia = Mat3::ZERO;
            self.local_com = self
                .colliders
                .first()
                .map(|c| c.local_transform.translation)
                .unwrap_or(Vector3::ZERO);
            // A fixed body still needs a defined centre of rotation, but its
            // inertia is infinite, so the tensor stays zero.
            self.local_com = Vector3::ZERO;
            self.update_world_inertia();
            return;
        }

        let mut props = MassProperties::ZERO;
        for c in self.colliders.iter().filter(|c| c.enabled && !c.is_sensor) {
            props = props.merged(&c.mass_properties());
        }

        // A dynamic body needs finite, non-zero mass. Half-spaces and open
        // triangle meshes have none, so fall back to a unit-density box over the
        // collider bounds; without this the body would silently behave as fixed.
        if props.mass <= 0.0 {
            props = self.synthesized_mass_properties();
        }
        if let Some(m) = self.mass_override {
            props = props.with_mass(m);
        }
        if let Some(com) = self.com_override {
            props.center_of_mass = com;
        }
        if let Some(i) = self.inertia_override {
            props.inertia = Mat3::from_diagonal(i);
        }

        self.mass = props.mass;
        self.inv_mass = if props.mass > 0.0 { 1.0 / props.mass } else { 0.0 };
        self.local_com = props.center_of_mass;
        self.local_inv_inertia = props.inertia.inverse();
        self.update_world_inertia();
    }

    fn synthesized_mass_properties(&self) -> MassProperties {
        let mut bounds = Aabb::empty();
        for c in self.colliders.iter().filter(|c| c.enabled && !c.is_sensor) {
            // Unbounded shapes would swamp the bounds; skip them.
            if c.shape.is_unbounded() {
                continue;
            }
            bounds = bounds.union(&c.compute_aabb(&Isometry::IDENTITY));
        }
        if bounds.is_empty() {
            return MassProperties::new(
                1.0,
                Vector3::ZERO,
                Mat3::from_diagonal(Vector3::new(1.0, 1.0, 1.0)),
            );
        }
        let half = bounds.size() * 0.5;
        Shape::cuboid(half.x, half.y, half.z)
            .mass_properties(1.0)
            .transformed_by(&Isometry::from_translation(bounds.center()))
    }

    fn update_world_inertia(&mut self) {
        let mut i = self.local_inv_inertia.rotated(self.position.rotation);
        // Locking a rotation axis means infinite inertia about it, i.e. a zero
        // row and column in the inverse tensor.
        for axis in 0..3 {
            if self.locked_axes.is_rotation_locked(axis) {
                for k in 0..3 {
                    i.m[axis * 3 + k] = 0.0;
                    i.m[k * 3 + axis] = 0.0;
                }
            }
        }
        self.world_inv_inertia = i;
    }

    // ---- colliders --------------------------------------------------------

    pub fn colliders(&self) -> &[Collider] {
        &self.colliders
    }

    /// Mutable access to one collider. Changing its shape, density or transform
    /// requires a follow-up [`Self::recompute_mass_properties`]; changing its
    /// material or groups does not.
    pub fn collider_mut(&mut self, index: usize) -> Option<&mut Collider> {
        self.colliders.get_mut(index)
    }

    pub fn add_collider(&mut self, collider: impl Into<Collider>) -> usize {
        self.colliders.push(collider.into());
        self.recompute_mass_properties();
        self.colliders.len() - 1
    }

    pub fn remove_collider(&mut self, index: usize) -> Option<Collider> {
        if index >= self.colliders.len() {
            return None;
        }
        let c = self.colliders.remove(index);
        self.recompute_mass_properties();
        Some(c)
    }

    /// World-space bounds of every enabled collider.
    pub fn compute_aabb(&self) -> Aabb {
        let mut b = Aabb::empty();
        for c in self.colliders.iter().filter(|c| c.enabled) {
            b = b.union(&c.compute_aabb(&self.position));
        }
        b
    }

    // ---- kinematics -------------------------------------------------------

    /// Velocity of the world-space point `p` on this body, including the
    /// contribution from spin.
    #[inline]
    pub fn velocity_at_point(&self, p: Vector3) -> Vector3 {
        self.linear_velocity + self.angular_velocity.cross(p - self.world_center_of_mass())
    }

    /// Teleport the body. Clears accumulated sleep, since a moved body is not
    /// at rest.
    pub fn set_position(&mut self, position: Isometry) {
        self.position = position;
        self.previous_position = position;
        self.update_world_inertia();
        self.wake_up();
    }

    /// Set the linear velocity, waking the body.
    ///
    /// Writing [`linear_velocity`](Self::linear_velocity) directly is the same
    /// thing minus the wake, which on a sleeping body means nothing happens.
    pub fn set_linear_velocity(&mut self, velocity: Vector3) {
        self.linear_velocity = velocity;
        self.wake_up();
    }

    /// Set the angular velocity, waking the body.
    pub fn set_angular_velocity(&mut self, velocity: Vector3) {
        self.angular_velocity = velocity;
        self.wake_up();
    }

    pub fn set_translation(&mut self, translation: Vector3) {
        self.position.translation = translation;
        self.previous_position = self.position;
        self.wake_up();
    }

    pub fn set_rotation(&mut self, rotation: Quaternion) {
        self.position.rotation = rotation.normalize();
        self.previous_position = self.position;
        self.update_world_inertia();
        self.wake_up();
    }

    pub fn translation(&self) -> Vector3 {
        self.position.translation
    }

    pub fn rotation(&self) -> Quaternion {
        self.position.rotation
    }

    /// Transform at the start of the current step. Interpolate between this and
    /// [`Self::position`] to draw at a frame rate above the physics rate.
    pub fn previous_position(&self) -> Isometry {
        self.previous_position
    }

    // ---- forces -----------------------------------------------------------

    /// Accumulate a force, applied at the centre of mass, until the next step.
    /// Forces are cleared every step.
    pub fn add_force(&mut self, force: Vector3) {
        if self.is_dynamic() {
            self.force = self.force + force;
            self.wake_up();
        }
    }

    /// Accumulate a force at a world-space point, producing torque as well.
    pub fn add_force_at_point(&mut self, force: Vector3, point: Vector3) {
        if self.is_dynamic() {
            self.force = self.force + force;
            self.torque = self.torque + (point - self.world_center_of_mass()).cross(force);
            self.wake_up();
        }
    }

    pub fn add_torque(&mut self, torque: Vector3) {
        if self.is_dynamic() {
            self.torque = self.torque + torque;
            self.wake_up();
        }
    }

    /// Change velocity immediately. Unlike a force, this ignores the time step —
    /// it is the right tool for jumps, hits and explosions.
    pub fn apply_impulse(&mut self, impulse: Vector3) {
        if !self.is_dynamic() {
            return;
        }
        let inv = self.inv_mass_axes();
        self.linear_velocity = self.linear_velocity
            + Vector3::new(impulse.x * inv.x, impulse.y * inv.y, impulse.z * inv.z);
        self.wake_up();
    }

    pub fn apply_impulse_at_point(&mut self, impulse: Vector3, point: Vector3) {
        if !self.is_dynamic() {
            return;
        }
        let r = point - self.world_center_of_mass();
        self.apply_impulse(impulse);
        self.apply_torque_impulse(r.cross(impulse));
    }

    pub fn apply_torque_impulse(&mut self, impulse: Vector3) {
        if !self.is_dynamic() {
            return;
        }
        self.angular_velocity =
            self.angular_velocity + self.world_inv_inertia.mul_vec(impulse);
        self.wake_up();
    }

    pub fn reset_forces(&mut self) {
        self.force = Vector3::ZERO;
        self.torque = Vector3::ZERO;
    }

    pub fn accumulated_force(&self) -> Vector3 {
        self.force
    }

    pub fn accumulated_torque(&self) -> Vector3 {
        self.torque
    }

    // ---- sleeping ---------------------------------------------------------

    pub fn is_sleeping(&self) -> bool {
        self.sleeping
    }

    /// Wake the body and restart its rest timer.
    pub fn wake_up(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    /// Force the body to sleep immediately, zeroing its velocity.
    pub fn sleep(&mut self) {
        self.sleeping = true;
        self.linear_velocity = Vector3::ZERO;
        self.angular_velocity = Vector3::ZERO;
    }

    pub(crate) fn update_sleep(&mut self, dt: f32, linear_threshold: f32, angular_threshold: f32, time_to_sleep: f32) {
        if !self.can_sleep || !self.is_dynamic() {
            self.sleep_timer = 0.0;
            return;
        }
        let at_rest = self.linear_velocity.length_sq() < linear_threshold * linear_threshold
            && self.angular_velocity.length_sq() < angular_threshold * angular_threshold;
        if at_rest {
            self.sleep_timer += dt;
            if self.sleep_timer >= time_to_sleep {
                self.sleep();
            }
        } else {
            self.sleep_timer = 0.0;
        }
    }

    // ---- integration ------------------------------------------------------

    /// Take custody of the body for the step, and refuse a transform that
    /// cannot be simulated.
    ///
    /// Returns `(velocity_repaired, position_repaired)`, which the world counts
    /// into [`Warning::NonFiniteVelocity`] and [`Warning::NonFinitePosition`]
    /// respectively. They are reported apart because they say different things
    /// about where the bad value came from, and a caller chasing one is not
    /// helped by being told the other.
    ///
    /// This is the last point at which the position can be repaired: after it,
    /// it has been copied into `previous_position`, the known-good value is
    /// gone, and by the end of the step every body compared against this one has
    /// the same infinity in it.
    ///
    /// [`Warning::NonFiniteVelocity`]: crate::diagnostics::Warning
    /// [`Warning::NonFinitePosition`]: crate::diagnostics::Warning
    pub(crate) fn begin_step(&mut self) -> (bool, bool) {
        let finite = |v: Vector3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        let sane = |iso: &Isometry| {
            finite(iso.translation)
                && iso.rotation.x.is_finite()
                && iso.rotation.y.is_finite()
                && iso.rotation.z.is_finite()
                && iso.rotation.w.is_finite()
        };
        // Velocities and accumulators too, and for every body rather than only
        // the ones about to be integrated: `integrate_velocity` returns early
        // for a sleeping body, so poison written into one would sit there
        // untouched and be handed to the solver the moment it woke.
        let mut velocity = false;
        if !finite(self.linear_velocity) {
            self.linear_velocity = Vector3::ZERO;
            velocity = true;
        }
        if !finite(self.angular_velocity) {
            self.angular_velocity = Vector3::ZERO;
            velocity = true;
        }
        if !finite(self.force) {
            self.force = Vector3::ZERO;
            velocity = true;
        }
        if !finite(self.torque) {
            self.torque = Vector3::ZERO;
            velocity = true;
        }
        let mut repaired = false;
        if !sane(&self.position) {
            // Prefer where the engine last had it; if that is gone too, the
            // origin is the only value left that is certainly usable.
            self.position = if sane(&self.previous_position) {
                self.previous_position
            } else {
                Isometry::IDENTITY
            };
            self.linear_velocity = Vector3::ZERO;
            self.angular_velocity = Vector3::ZERO;
            repaired = true;
        }
        self.previous_position = self.position;
        (velocity, repaired)
    }

    /// Semi-implicit Euler on velocity: gravity and accumulated forces, then
    /// damping.
    /// Returns whether a non-finite velocity had to be zeroed, which the world
    /// counts into [`Warning::NonFiniteVelocity`](crate::diagnostics::Warning).
    pub(crate) fn integrate_velocity(&mut self, dt: f32, gravity: Vector3) -> bool {
        if !self.is_dynamic() || self.sleeping || !self.enabled {
            return false;
        }
        let inv = self.inv_mass_axes();
        // Gravity is an acceleration, so it is applied without the mass term —
        // but still suppressed on locked axes.
        let g = gravity * self.gravity_scale;
        let acc = Vector3::new(
            if inv.x == 0.0 { 0.0 } else { g.x + self.force.x * inv.x },
            if inv.y == 0.0 { 0.0 } else { g.y + self.force.y * inv.y },
            if inv.z == 0.0 { 0.0 } else { g.z + self.force.z * inv.z },
        );
        self.linear_velocity = self.linear_velocity + acc * dt;
        self.angular_velocity =
            self.angular_velocity + self.world_inv_inertia.mul_vec(self.torque) * dt;

        // Exponential damping — stable at any step size, unlike `v *= 1 - k*dt`
        // which reverses direction once `k * dt > 1`.
        //
        // The gyroscopic term (ω × Iω) is deliberately omitted: it is unstable
        // at game time steps and its absence is imperceptible outside of
        // free-spinning asymmetric bodies.
        self.linear_velocity = self.linear_velocity * (1.0 / (1.0 + dt * self.linear_damping.max(0.0)));
        self.angular_velocity =
            self.angular_velocity * (1.0 / (1.0 + dt * self.angular_damping.max(0.0)));

        // One NaN reaches every body this one touches by the end of the step, so
        // it is zeroed here and reported rather than left to propagate.
        let finite = |v: Vector3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        let mut repaired = false;
        if !finite(self.linear_velocity) {
            self.linear_velocity = Vector3::ZERO;
            repaired = true;
        }
        if !finite(self.angular_velocity) {
            self.angular_velocity = Vector3::ZERO;
            repaired = true;
        }

        self.apply_locks();
        repaired
    }

    /// Advance the transform, rotating about the centre of mass.
    pub(crate) fn integrate_position(&mut self, dt: f32) {
        if !self.enabled || self.sleeping {
            return;
        }
        if !(self.is_dynamic() || self.is_kinematic()) {
            return;
        }
        self.apply_locks();

        let com = self.world_center_of_mass() + self.linear_velocity * dt;

        // q' = q + (dt/2) ω ⊗ q
        let w = self.angular_velocity;
        let q = self.position.rotation;
        let half_dt = dt * 0.5;
        let dq = Quaternion::new(
            w.x * q.w + w.y * q.z - w.z * q.y,
            w.y * q.w + w.z * q.x - w.x * q.z,
            w.z * q.w + w.x * q.y - w.y * q.x,
            -(w.x * q.x + w.y * q.y + w.z * q.z),
        );
        let rotated = Quaternion::new(
            q.x + dq.x * half_dt,
            q.y + dq.y * half_dt,
            q.z + dq.z * half_dt,
            q.w + dq.w * half_dt,
        )
        .normalize();

        self.position.rotation = rotated;
        self.set_world_center_of_mass(com);
        self.update_world_inertia();
    }

    /// Move the body by the solver's pseudo velocities.
    ///
    /// Overlap is pushed apart with a velocity that exists for this step only
    /// and is never added to the real one. That separation is the point: a deep
    /// interpenetration resolved through the ordinary velocity would store the
    /// push as momentum and fire the bodies apart on the next step.
    ///
    /// Kinematic and fixed bodies are left alone — being unmoved by what they
    /// overlap is what makes them kinematic.
    pub(crate) fn apply_position_correction(&mut self, linear: Vector3, angular: Vector3, dt: f32) {
        if !self.enabled || self.sleeping || !self.is_dynamic() {
            return;
        }
        if linear.length_sq() <= 0.0 && angular.length_sq() <= 0.0 {
            return;
        }

        let com = self.world_center_of_mass() + linear * dt;

        // Same q' = q + (dt/2) ω ⊗ q as `integrate_position`, on the pseudo
        // angular velocity.
        let w = angular;
        let q = self.position.rotation;
        let half_dt = dt * 0.5;
        let dq = Quaternion::new(
            w.x * q.w + w.y * q.z - w.z * q.y,
            w.y * q.w + w.z * q.x - w.x * q.z,
            w.z * q.w + w.x * q.y - w.y * q.x,
            -(w.x * q.x + w.y * q.y + w.z * q.z),
        );
        let rotated = Quaternion::new(
            q.x + dq.x * half_dt,
            q.y + dq.y * half_dt,
            q.z + dq.z * half_dt,
            q.w + dq.w * half_dt,
        )
        .normalize();

        self.position.rotation = rotated;
        self.set_world_center_of_mass(com);
        self.update_world_inertia();
    }

    fn apply_locks(&mut self) {
        if self.locked_axes.is_empty() {
            return;
        }
        for axis in 0..3 {
            if self.locked_axes.is_translation_locked(axis) {
                match axis {
                    0 => self.linear_velocity.x = 0.0,
                    1 => self.linear_velocity.y = 0.0,
                    _ => self.linear_velocity.z = 0.0,
                }
            }
            if self.locked_axes.is_rotation_locked(axis) {
                match axis {
                    0 => self.angular_velocity.x = 0.0,
                    1 => self.angular_velocity.y = 0.0,
                    _ => self.angular_velocity.z = 0.0,
                }
            }
        }
    }
}

/// Fluent constructor for [`RigidBody`].
///
/// Every setter returns `Self`, and the builder can be handed straight to
/// [`crate::world::World::add_body`] without calling [`Self::build`].
#[derive(Debug, Clone)]
pub struct RigidBodyBuilder {
    body: RigidBody,
    density: Option<f32>,
    material: Option<PhysicsMaterial>,
}

impl RigidBodyBuilder {
    pub fn new(body_type: BodyType) -> Self {
        Self {
            body: RigidBody {
                body_type,
                ..Default::default()
            },
            density: None,
            material: None,
        }
    }

    pub fn position(mut self, position: Isometry) -> Self {
        self.body.position = position;
        self
    }

    pub fn translation(mut self, translation: Vector3) -> Self {
        self.body.position.translation = translation;
        self
    }

    pub fn rotation(mut self, rotation: Quaternion) -> Self {
        self.body.position.rotation = rotation.normalize();
        self
    }

    /// Attach a shape with no offset. Shorthand for `.collider(Collider::new(shape))`.
    pub fn shape(self, shape: Shape) -> Self {
        self.collider(Collider::new(shape))
    }

    pub fn collider(mut self, collider: impl Into<Collider>) -> Self {
        self.body.colliders.push(collider.into());
        self
    }

    pub fn linear_velocity(mut self, v: Vector3) -> Self {
        self.body.linear_velocity = v;
        self
    }

    pub fn angular_velocity(mut self, v: Vector3) -> Self {
        self.body.angular_velocity = v;
        self
    }

    pub fn linear_damping(mut self, d: f32) -> Self {
        self.body.linear_damping = d.max(0.0);
        self
    }

    pub fn angular_damping(mut self, d: f32) -> Self {
        self.body.angular_damping = d.max(0.0);
        self
    }

    pub fn gravity_scale(mut self, s: f32) -> Self {
        self.body.gravity_scale = s;
        self
    }

    /// Density applied to every collider on this body. Ignored for colliders
    /// whose density you set individually **after** `build`.
    pub fn density(mut self, density: f32) -> Self {
        self.density = Some(density.max(0.0));
        self
    }

    /// Exact total mass, overriding whatever density implies.
    pub fn mass(mut self, mass: f32) -> Self {
        self.body.mass_override = Some(mass.max(0.0));
        self
    }

    /// Put the balance point somewhere other than where the colliders say.
    ///
    /// In the body's own frame. A weeble rights itself because its mass sits
    /// below its centre, and no arrangement of a single ball collider says so.
    pub fn center_of_mass(mut self, com: Vector3) -> Self {
        self.body.com_override = Some(com);
        self
    }

    /// State the moments of inertia about the local axes directly.
    ///
    /// For faking a mass distribution the collider does not have: a flywheel is
    /// a thin cylinder that must be hard to spin about its axle and easy to
    /// tumble, which its own shape would have exactly backwards.
    pub fn principal_inertia(mut self, inertia: Vector3) -> Self {
        self.body.inertia_override = Some(inertia);
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        self.material.get_or_insert_with(PhysicsMaterial::default).friction = friction.max(0.0);
        self
    }

    pub fn restitution(mut self, restitution: f32) -> Self {
        self.material
            .get_or_insert_with(PhysicsMaterial::default)
            .restitution = restitution.clamp(0.0, 1.0);
        self
    }

    pub fn material(mut self, material: PhysicsMaterial) -> Self {
        self.material = Some(material);
        self
    }

    pub fn locked_axes(mut self, axes: LockedAxes) -> Self {
        self.body.locked_axes = axes;
        self
    }

    /// Keep the body upright — the usual choice for characters.
    pub fn lock_rotation(self) -> Self {
        let axes = self.body.locked_axes | LockedAxes::ROTATION;
        self.locked_axes(axes)
    }

    /// Enable continuous collision detection for this body — see
    /// [`RigidBody::ccd_enabled`].
    pub fn ccd(mut self, enabled: bool) -> Self {
        self.body.ccd_enabled = enabled;
        self
    }

    pub fn can_sleep(mut self, can_sleep: bool) -> Self {
        self.body.can_sleep = can_sleep;
        self
    }

    pub fn user_data(mut self, data: u64) -> Self {
        self.body.user_data = data;
        self
    }

    /// Bind this body to a scene node so [`crate::world::World::sync_to_scene`] drives it.
    pub fn scene_object(mut self, id: ObjectId) -> Self {
        self.body.scene_object = Some(id);
        self
    }

    pub fn build(mut self) -> RigidBody {
        if let Some(d) = self.density {
            for c in &mut self.body.colliders {
                c.density = d;
            }
        }
        if let Some(m) = self.material {
            for c in &mut self.body.colliders {
                c.material = m;
            }
        }
        let mut body = self.body;
        body.previous_position = body.position;
        body.recompute_mass_properties();
        body
    }
}

impl From<RigidBodyBuilder> for RigidBody {
    fn from(b: RigidBodyBuilder) -> Self {
        b.build()
    }
}

/// Stable handle to a body. Remains valid until the body is removed; afterwards
/// lookups return `None` rather than aliasing a recycled slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId {
    index: u32,
    generation: u32,
}

impl BodyId {
    /// Dense slot index — useful as an array key inside a single step.
    #[inline]
    pub fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Debug, Clone)]
struct Slot {
    generation: u32,
    body: Option<RigidBody>,
}

/// Generational arena of rigid bodies.
#[derive(Debug, Clone, Default)]
pub struct BodySet {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
}

impl BodySet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, body: impl Into<RigidBody>) -> BodyId {
        let body = body.into();
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.body = Some(body);
            return BodyId {
                index,
                generation: slot.generation,
            };
        }
        let index = self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 0,
            body: Some(body),
        });
        BodyId {
            index,
            generation: 0,
        }
    }

    pub fn remove(&mut self, id: BodyId) -> Option<RigidBody> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let body = slot.body.take()?;
        // Bumping the generation is what makes stale handles fail instead of
        // silently addressing whichever body reuses the slot next.
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
        self.len -= 1;
        Some(body)
    }

    pub fn get(&self, id: BodyId) -> Option<&RigidBody> {
        let slot = self.slots.get(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.body.as_ref())?
    }

    pub fn get_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        let slot = self.slots.get_mut(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.body.as_mut())?
    }

    /// Two distinct bodies at once — what every constraint solve needs.
    /// `None` if the handles are equal or either is stale.
    pub fn get_two_mut(&mut self, a: BodyId, b: BodyId) -> Option<(&mut RigidBody, &mut RigidBody)> {
        if a.index == b.index {
            return None;
        }
        let (lo, hi) = if a.index < b.index { (a, b) } else { (b, a) };
        let (left, right) = self.slots.split_at_mut(hi.index as usize);
        let lo_slot = left.get_mut(lo.index as usize)?;
        let hi_slot = right.first_mut()?;
        if lo_slot.generation != lo.generation || hi_slot.generation != hi.generation {
            return None;
        }
        let lo_body = lo_slot.body.as_mut()?;
        let hi_body = hi_slot.body.as_mut()?;
        Some(if a.index < b.index {
            (lo_body, hi_body)
        } else {
            (hi_body, lo_body)
        })
    }

    pub fn contains(&self, id: BodyId) -> bool {
        self.get(id).is_some()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.len = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = (BodyId, &RigidBody)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.body.as_ref().map(|b| {
                (
                    BodyId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    b,
                )
            })
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (BodyId, &mut RigidBody)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            let generation = s.generation;
            s.body.as_mut().map(|b| {
                (
                    BodyId {
                        index: i as u32,
                        generation,
                    },
                    b,
                )
            })
        })
    }

    pub fn ids(&self) -> impl Iterator<Item = BodyId> + '_ {
        self.iter().map(|(id, _)| id)
    }

    // -- dense access, for the solver's inner loops --

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn by_index(&self, index: usize) -> Option<&RigidBody> {
        self.slots.get(index)?.body.as_ref()
    }

    pub(crate) fn by_index_mut(&mut self, index: usize) -> Option<&mut RigidBody> {
        self.slots.get_mut(index)?.body.as_mut()
    }

    pub(crate) fn id_at(&self, index: usize) -> Option<BodyId> {
        let slot = self.slots.get(index)?;
        slot.body.as_ref().map(|_| BodyId {
            index: index as u32,
            generation: slot.generation,
        })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dynamic_body_derives_mass_from_its_shape() {
        let b = RigidBody::dynamic().shape(Shape::ball(1.0)).density(3.0).build();
        let want = 3.0 * (4.0 / 3.0) * std::f32::consts::PI;
        assert!((b.mass() - want).abs() < 1e-3, "{}", b.mass());
        assert!((b.inv_mass() - 1.0 / want).abs() < 1e-4);
    }

    #[test]
    fn an_explicit_mass_overrides_density() {
        let b = RigidBody::dynamic()
            .shape(Shape::ball(1.0))
            .density(100.0)
            .mass(2.0)
            .build();
        assert!((b.mass() - 2.0).abs() < 1e-5);
        // Inertia is rescaled, not discarded.
        assert!(b.world_inv_inertia().m[0] > 0.0);
    }

    #[test]
    fn fixed_and_kinematic_bodies_have_infinite_mass() {
        for b in [
            RigidBody::fixed().shape(Shape::ball(1.0)).build(),
            RigidBody::kinematic().shape(Shape::ball(1.0)).build(),
        ] {
            assert_eq!(b.inv_mass(), 0.0);
            assert_eq!(b.world_inv_inertia(), Mat3::ZERO);
        }
    }

    #[test]
    fn a_dynamic_body_with_a_massless_shape_still_gets_mass() {
        // A half-space has no volume; without a fallback this body would be
        // silently immovable.
        let b = RigidBody::dynamic().shape(Shape::ground()).build();
        assert!(b.mass() > 0.0);
        assert!(b.inv_mass() > 0.0);
    }

    #[test]
    fn offset_colliders_shift_the_centre_of_mass() {
        let b = RigidBody::dynamic()
            .collider(Collider::new(Shape::ball(0.5)).translation(Vector3::new(2.0, 0.0, 0.0)))
            .build();
        assert!((b.local_center_of_mass() - Vector3::new(2.0, 0.0, 0.0)).length() < 1e-4);

        // And the world COM follows the body transform.
        let mut b2 = b.clone();
        b2.set_translation(Vector3::new(0.0, 10.0, 0.0));
        assert!((b2.world_center_of_mass() - Vector3::new(2.0, 10.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn locked_axes_zero_the_matching_inverse_terms() {
        let b = RigidBody::dynamic()
            .shape(Shape::cuboid(1.0, 1.0, 1.0))
            .locked_axes(LockedAxes::TRANSLATION_Y | LockedAxes::ROTATION_X)
            .build();
        let inv = b.inv_mass_axes();
        assert!(inv.x > 0.0 && inv.z > 0.0);
        assert_eq!(inv.y, 0.0);
        let i = b.world_inv_inertia();
        assert_eq!(i.m[0], 0.0); // row x
        assert_eq!(i.m[3], 0.0); // col x
        assert!(i.m[4] > 0.0); // y still free
    }

    #[test]
    fn a_locked_body_ignores_impulses_on_that_axis() {
        let mut b = RigidBody::dynamic()
            .shape(Shape::ball(1.0))
            .locked_axes(LockedAxes::TRANSLATION_Y)
            .build();
        b.apply_impulse(Vector3::new(5.0, 5.0, 0.0));
        assert!(b.linear_velocity.x > 0.0);
        assert_eq!(b.linear_velocity.y, 0.0);
    }

    #[test]
    fn gravity_accelerates_at_the_same_rate_regardless_of_mass() {
        let mut light = RigidBody::dynamic().shape(Shape::ball(1.0)).mass(0.1).build();
        let mut heavy = RigidBody::dynamic().shape(Shape::ball(1.0)).mass(1000.0).build();
        let g = Vector3::new(0.0, -10.0, 0.0);
        for _ in 0..10 {
            light.integrate_velocity(0.01, g);
            heavy.integrate_velocity(0.01, g);
        }
        assert!((light.linear_velocity.y - heavy.linear_velocity.y).abs() < 1e-4);
    }

    #[test]
    fn integration_rotates_about_the_centre_of_mass_not_the_origin() {
        // Collider offset from the origin: spinning must swing the origin around
        // the offset centre of mass, not the other way round.
        let mut b = RigidBody::dynamic()
            .collider(Collider::new(Shape::ball(0.2)).translation(Vector3::new(1.0, 0.0, 0.0)))
            .build();
        b.angular_velocity = Vector3::new(0.0, 1.0, 0.0);
        let com_before = b.world_center_of_mass();
        for _ in 0..100 {
            b.integrate_position(0.01);
        }
        let com_after = b.world_center_of_mass();
        assert!(
            (com_before - com_after).length() < 1e-3,
            "centre of mass drifted: {com_before:?} -> {com_after:?}"
        );
        // The origin, being offset, must have moved.
        assert!(b.translation().length() > 0.1);
    }

    #[test]
    fn velocity_at_point_includes_spin() {
        let mut b = RigidBody::dynamic().shape(Shape::ball(1.0)).build();
        b.angular_velocity = Vector3::new(0.0, 0.0, 1.0);
        let v = b.velocity_at_point(Vector3::new(1.0, 0.0, 0.0));
        assert!((v - Vector3::new(0.0, 1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn damping_never_reverses_velocity_even_at_absurd_steps() {
        let mut b = RigidBody::dynamic().shape(Shape::ball(1.0)).build();
        b.linear_damping = 50.0;
        b.linear_velocity = Vector3::new(10.0, 0.0, 0.0);
        b.integrate_velocity(1.0, Vector3::ZERO);
        assert!(b.linear_velocity.x > 0.0, "{:?}", b.linear_velocity);
        assert!(b.linear_velocity.x < 10.0);
    }

    #[test]
    fn arena_handles_survive_unrelated_removals() {
        let mut set = BodySet::new();
        let a = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        let b = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        let c = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        assert_eq!(set.len(), 3);
        set.remove(b);
        assert_eq!(set.len(), 2);
        assert!(set.get(a).is_some());
        assert!(set.get(c).is_some());
        assert!(set.get(b).is_none());
    }

    #[test]
    fn a_stale_handle_never_aliases_a_recycled_slot() {
        let mut set = BodySet::new();
        let old = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        set.remove(old);
        let new = set.insert(RigidBody::dynamic().shape(Shape::ball(2.0)));
        assert_eq!(old.index(), new.index(), "slot should have been reused");
        assert!(set.get(old).is_none(), "stale handle resolved");
        assert!(set.get(new).is_some());
    }

    #[test]
    fn get_two_mut_returns_the_pair_in_the_order_asked() {
        let mut set = BodySet::new();
        let a = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)).user_data(1));
        let b = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)).user_data(2));
        let (x, y) = set.get_two_mut(a, b).unwrap();
        assert_eq!((x.user_data, y.user_data), (1, 2));
        let (x, y) = set.get_two_mut(b, a).unwrap();
        assert_eq!((x.user_data, y.user_data), (2, 1));
        assert!(set.get_two_mut(a, a).is_none());
    }

    #[test]
    fn sleeping_takes_the_configured_time_and_any_impulse_wakes_it() {
        let mut b = RigidBody::dynamic().shape(Shape::ball(1.0)).build();
        for _ in 0..10 {
            b.update_sleep(0.05, 0.05, 0.05, 0.5);
        }
        assert!(b.is_sleeping());
        b.apply_impulse(Vector3::new(0.0, 1.0, 0.0));
        assert!(!b.is_sleeping());
    }
}
