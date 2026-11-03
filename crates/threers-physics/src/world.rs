//! [`World`] — the object you actually talk to.
//!
//! ```
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! world.add_body(RigidBody::fixed().shape(Shape::ground()));
//! let ball = world.add_body(
//!     RigidBody::dynamic()
//!         .shape(Shape::ball(0.5))
//!         .translation(Vector3::new(0.0, 5.0, 0.0)),
//! );
//!
//! for _ in 0..180 {
//!     world.step(1.0 / 60.0);
//! }
//!
//! // It fell and came to rest on the ground.
//! let y = world.body(ball).unwrap().translation().y;
//! assert!((y - 0.5).abs() < 0.05, "ball rested at {y}");
//! ```

use crate::body::{BodyId, BodySet, RigidBody};
use crate::broadphase::BroadPhase;
use crate::contact::{CollisionEvent, ContactPoint, ContactSet, ManifoldKey};
use crate::diagnostics::{Diagnostics, Warning};
use crate::island::IslandSet;
use crate::joint::{Joint, JointId, JointSet};
use crate::narrowphase::{collide, RawManifold};
use crate::solver::{Solver, SolverConfig};
use crate::tendon::{Tendon, TendonId, TendonSet};
use std::collections::HashSet;
use threers::core::{ObjectArena, ObjectId};
use threers::math::Vector3;

/// When bodies are allowed to stop simulating.
///
/// Sleeping is what keeps a scene of a thousand settled props cheap. A sleeping
/// body is skipped by the broad phase, the solver and integration, and wakes on
/// contact, on an impulse, or when moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SleepConfig {
    pub enabled: bool,
    /// Linear speed below which a body counts as at rest.
    pub linear_threshold: f32,
    /// Angular speed below which a body counts as at rest.
    pub angular_threshold: f32,
    /// How long a body must stay at rest before it sleeps.
    pub time_to_sleep: f32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            linear_threshold: 0.05,
            angular_threshold: 0.05,
            time_to_sleep: 0.5,
        }
    }
}

/// An attraction law applied on top of uniform [`World::gravity`].
///
/// Uniform gravity is a flat-earth approximation: every body falls the same way
/// however far apart they are. That is right for a level and wrong for anything
/// in orbit, so this is the escape hatch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GravityModel {
    /// Uniform gravity only.
    #[default]
    None,
    /// Attraction toward a fixed point, as though a mass sat there without
    /// being simulated — a planet the satellites do not perturb.
    ///
    /// `mu` is `G·M`. The circular orbital speed at radius `r` is `sqrt(mu / r)`,
    /// and it does not depend on the orbiting body's own mass.
    Point {
        centre: Vector3,
        mu: f32,
        /// Floor on the separation used in the inverse square, so a body that
        /// passes through the centre is not flung to infinity.
        min_distance: f32,
    },
    /// Every body pulls on every other. `O(n²)`, so it is for a handful of
    /// bodies rather than a scene.
    Mutual {
        /// The gravitational constant, in whatever units the scene uses.
        g: f32,
        min_distance: f32,
    },
}

/// The simulation.
pub struct World {
    /// Acceleration applied to every dynamic body, scaled per body by
    /// [`RigidBody::gravity_scale`]. Defaults to Earth gravity along `-Y`.
    pub gravity: Vector3,
    pub solver_config: SolverConfig,
    pub sleep: SleepConfig,
    /// Fixed physics step. Simulation quality depends on this being constant,
    /// which is why [`World::step`] accumulates real time rather than using it
    /// directly.
    pub timestep: f32,
    /// Most fixed steps run per [`World::step`] call. Caps the "spiral of
    /// death" where a slow frame asks for more simulation than it can afford.
    pub max_substeps: usize,
    /// How far apart two shapes may be and still get a contact. Lets the solver
    /// brake a fast body *before* it interpenetrates.
    pub prediction_distance: f32,
    /// Attraction law applied on top of [`Self::gravity`]. [`GravityModel::None`]
    /// by default, which is the flat uniform field almost every scene wants.
    pub gravity_model: GravityModel,
    /// Speed ceiling applied to every dynamic body each substep, counted into
    /// [`Warning::VelocityClamped`]. Infinite by default — set it when a scene
    /// can produce a velocity that would tunnel through the world in one step.
    pub max_velocity: f32,

    bodies: BodySet,
    joints: JointSet,
    tendons: TendonSet,
    contacts: ContactSet,
    broadphase: BroadPhase,
    solver: Solver,

    /// What the engine noticed while it was running: timings, counters and
    /// warnings. Reset at the top of each [`World::step`].
    diagnostics: Diagnostics,
    /// Fixed steps actually run by the last [`World::step`] call.
    pub substeps: usize,

    accumulator: f32,
    interpolation_alpha: f32,

    // Scratch, reused every step to keep the hot path allocation-free.
    pairs: Vec<(u32, u32)>,
    /// Pairs supplied from outside, used instead of the CPU broad phase for the
    /// next step and then forgotten.
    external_pairs: Option<Vec<(u32, u32)>>,
    raw: Vec<RawManifold>,
    no_collide: HashSet<(u32, u32)>,
    /// Rebuilt at the end of each step so `counters().islands` can report it.
    /// Kept on the world rather than made locally so the union-find's buffers
    /// are reused instead of reallocated every step.
    islands: IslandSet,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            solver_config: SolverConfig::default(),
            sleep: SleepConfig::default(),
            timestep: 1.0 / 60.0,
            max_substeps: 4,
            prediction_distance: 0.02,
            gravity_model: GravityModel::None,
            max_velocity: f32::INFINITY,
            bodies: BodySet::new(),
            joints: JointSet::new(),
            tendons: TendonSet::new(),
            contacts: ContactSet::new(),
            broadphase: BroadPhase::new(),
            solver: Solver::new(),
            diagnostics: Diagnostics::default(),
            substeps: 0,
            accumulator: 0.0,
            interpolation_alpha: 0.0,
            pairs: Vec::new(),
            external_pairs: None,
            raw: Vec::new(),
            no_collide: HashSet::new(),
            islands: IslandSet::new(),
        }
    }

    pub fn with_gravity(mut self, gravity: Vector3) -> Self {
        self.gravity = gravity;
        self
    }

    /// A world with no uniform gravity — the starting point for anything in
    /// orbit, in free fall, or in space.
    pub fn zero_gravity() -> Self {
        Self::new().with_gravity(Vector3::ZERO)
    }

    /// Attract bodies by a law rather than a constant field. See
    /// [`GravityModel`].
    pub fn with_gravity_model(mut self, model: GravityModel) -> Self {
        self.gravity_model = model;
        self
    }

    /// Change gravity, waking everything asleep.
    ///
    /// Bodies that have gone to sleep are not integrating, so they would ignore
    /// a change in gravity until something else disturbed them — a scene that
    /// has settled would simply keep hanging there when gravity turned over.
    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
        for (_, body) in self.bodies.iter_mut() {
            body.wake_up();
        }
    }

    /// Solve at a named quality rather than by tuning iteration counts.
    pub fn with_quality(mut self, quality: crate::solver::SimulationQuality) -> Self {
        self.solver_config = quality.config();
        self
    }

    /// Set the fixed physics rate, in steps per second.
    pub fn with_timestep_hz(mut self, hz: f32) -> Self {
        self.timestep = 1.0 / hz.max(1.0);
        self
    }

    // ---- bodies and joints ------------------------------------------------

    pub fn add_body(&mut self, body: impl Into<RigidBody>) -> BodyId {
        self.bodies.insert(body)
    }

    /// Remove a body, along with every contact, joint and tendon that referenced
    /// it. A cable routed through a body that no longer exists has no path left
    /// to follow, so it goes with it rather than pulling on a hole.
    pub fn remove_body(&mut self, id: BodyId) -> Option<RigidBody> {
        let body = self.bodies.remove(id)?;
        self.contacts.remove_body(id);
        self.joints.remove_body(id);
        // Tendons too: a cable routed through a body that no longer exists has
        // no path left to follow, and leaving it in place means the next step
        // pulls on a hole.
        self.tendons.remove_body(id);
        Some(body)
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.get(id)
    }

    /// Mutable access. Note that changing a body's transform directly
    /// teleports it — use [`RigidBody::set_position`] and friends, which also
    /// wake it.
    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.get_mut(id)
    }

    pub fn bodies(&self) -> &BodySet {
        &self.bodies
    }

    /// The bounds the broad phase compared on the last step, prediction distance
    /// already added.
    ///
    /// Before the first step the broad phase has nothing in it yet, so this
    /// falls back to the bodies' own bounds — a debug view asked for the boxes
    /// wants to see them on frame zero too, not after something has moved.
    pub fn broad_bounds(&self) -> Vec<crate::math::Aabb> {
        if self.broadphase.is_empty() {
            return self
                .bodies
                .iter()
                .map(|(_, b)| b.compute_aabb())
                .filter(|a| !a.is_empty())
                .collect();
        }
        self.broadphase
            .entries()
            .iter()
            .map(|e| crate::math::Aabb::new(e.min, e.max))
            .collect()
    }

    /// Use `pairs` for the next step instead of running the broad phase.
    ///
    /// Body-slot indices, each pair low-then-high. For handing back the result
    /// of a broad phase computed elsewhere — see
    /// [`GpuBroadPhase`](crate::gpu::GpuBroadPhase). Consumed by the next step;
    /// miss a step and the CPU sweep runs as usual.
    pub fn set_broadphase_pairs(&mut self, pairs: Vec<(u32, u32)>) {
        self.external_pairs = Some(pairs);
    }

    pub fn bodies_mut(&mut self) -> &mut BodySet {
        &mut self.bodies
    }

    pub fn add_joint(&mut self, joint: Joint) -> JointId {
        self.joints.insert(joint)
    }

    pub fn remove_joint(&mut self, id: JointId) -> Option<Joint> {
        self.joints.remove(id)
    }

    pub fn joint(&self, id: JointId) -> Option<&Joint> {
        self.joints.get(id)
    }

    pub fn joint_mut(&mut self, id: JointId) -> Option<&mut Joint> {
        self.joints.get_mut(id)
    }

    pub fn joints(&self) -> &JointSet {
        &self.joints
    }

    pub fn joints_mut(&mut self) -> &mut JointSet {
        &mut self.joints
    }

    /// Real time left over from the last [`World::step`], not yet consumed by a
    /// fixed step. Part of the simulation state: a rollback that ignores it
    /// resumes a fraction of a step out.
    pub fn accumulator_remaining(&self) -> f32 {
        self.accumulator
    }

    pub(crate) fn set_accumulator(&mut self, accumulator: f32) {
        self.accumulator = accumulator;
    }

    /// Put the captured contact set back, wholesale.
    ///
    /// Patching impulses onto whatever manifolds happen to be live is not
    /// enough: after simulating on and rewinding, the live set is the one from
    /// the abandoned future. Contacts that existed at snapshot time may be gone,
    /// ones that did not may be present, and those that survive have points that
    /// have since moved — so the next step matches nothing, warm-starts cold,
    /// and the replay diverges on its very first step.
    pub(crate) fn restore_contacts(&mut self, saved: &[crate::snapshot::ContactSnapshot]) {
        self.contacts.restore(saved.iter().map(|entry| {
            let points = entry
                .points
                .iter()
                .map(|&(local_a, normal_impulse, tangent_impulse)| ContactPoint {
                    point_a: Vector3::ZERO,
                    point_b: Vector3::ZERO,
                    local_a,
                    local_b: Vector3::ZERO,
                    depth: 0.0,
                    normal_impulse,
                    tangent_impulse,
                })
                .collect();
            (entry.key, points, entry.touching)
        }));
    }

    /// Re-file the bodies in the broad phase against their current positions.
    ///
    /// Needed after a restore, where the bodies moved without a step running:
    /// until this happens a raycast is answered from where they used to be.
    ///
    /// Deliberately *not* a full collision pass. Re-running the narrow phase
    /// here would begin a new step on the contact set and throw away the
    /// warm-start impulses the restore just put back, and those impulses are
    /// part of the state — without them the first step after a restore solves
    /// from a different guess and the replay diverges immediately.
    pub(crate) fn refresh_queries(&mut self) {
        self.rebuild_broadphase();
    }

    // ---- tendons ----------------------------------------------------------

    /// Add a cable routed through the world — see [`crate::tendon`].
    pub fn add_tendon(&mut self, tendon: Tendon) -> TendonId {
        // A path that cannot be read as written is a modelling mistake, and it
        // is a mistake the moment it is made rather than once per step — so it
        // is reported here, where the caller still has the stack that built it.
        if tendon.resolve(&self.bodies).malformed {
            self.diagnostics.warn(Warning::MalformedTendonPath, 1);
        }
        self.tendons.insert(tendon)
    }

    pub fn remove_tendon(&mut self, id: TendonId) -> Option<Tendon> {
        self.tendons.remove(id)
    }

    pub fn tendon(&self, id: TendonId) -> Option<&Tendon> {
        self.tendons.get(id)
    }

    pub fn tendon_mut(&mut self, id: TendonId) -> Option<&mut Tendon> {
        self.tendons.get_mut(id)
    }

    pub fn tendons(&self) -> &TendonSet {
        &self.tendons
    }

    /// How long the path is right now, summed over every segment.
    ///
    /// This is the number a rope, a spring or a winch is measured against, and
    /// the one to read off a freshly assembled scene to find its rest length.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// let hook = world.add_body(RigidBody::fixed().shape(Shape::ball(0.05)));
    /// let load = world.add_body(
    ///     RigidBody::dynamic()
    ///         .shape(Shape::ball(0.1))
    ///         .translation(Vector3::new(0.0, -2.0, 0.0)),
    /// );
    /// let rope = world.add_tendon(Tendon::rope(
    ///     vec![
    ///         TendonPoint::new(hook, Vector3::ZERO),
    ///         TendonPoint::new(load, Vector3::ZERO),
    ///     ],
    ///     2.0,
    /// ));
    /// assert!((world.tendon_length(rope).unwrap() - 2.0).abs() < 1e-5);
    /// ```
    pub fn tendon_length(&self, id: TendonId) -> Option<f32> {
        Some(self.tendons.get(id)?.length(&self.bodies))
    }

    /// The tension the cable carried through the last step, in newtons.
    ///
    /// Positive is pulling; a slack rope reads zero. Measured over one substep,
    /// which is the interval the impulse was actually applied over.
    pub fn tendon_tension(&self, id: TendonId) -> Option<f32> {
        let substep = self.timestep / self.substeps.max(1) as f32;
        Some(self.tendons.get(id)?.tension(substep))
    }

    /// Where a hinge, slider or screw currently stands, and how fast it is
    /// moving.
    ///
    /// `None` for a joint that has no single coordinate — a ball socket has
    /// three rotations and no one number describes it — or whose bodies have
    /// gone.
    ///
    /// The coordinate is the one the joint's own limits are measured in, so a
    /// joint sitting on its stop reports exactly the limit value rather than a
    /// hair past it.
    ///
    /// ```
    /// # #[cfg(feature = "mechanism")] {
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.5, 0.05, 0.5)));
    /// let flap = world.add_body(
    ///     RigidBody::dynamic()
    ///         .shape(Shape::cuboid(0.5, 0.05, 0.5))
    ///         .translation(Vector3::new(1.0, 0.0, 0.0)),
    /// );
    /// let hinge = world.add_joint(
    ///     Joint::revolute_at_point(world.bodies(), frame, flap, Vector3::new(0.5, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0))
    ///         .unwrap()
    ///         .with_limits(-0.5, 0.0),
    /// );
    ///
    /// // It starts level, then gravity swings it down onto its stop.
    /// assert!(world.joint_angle(hinge).unwrap().abs() < 1e-4);
    /// for _ in 0..600 {
    ///     world.step(1.0 / 60.0);
    /// }
    /// assert!((world.joint_angle(hinge).unwrap() + 0.5).abs() < 0.02);
    /// # }
    /// ```
    #[cfg(feature = "mechanism")]
    pub fn joint_state(&self, id: JointId) -> Option<crate::joint::JointState> {
        let joint = self.joints.get(id)?;
        let a = self.bodies.get(joint.body_a)?;
        let b = self.bodies.get(joint.body_b)?;
        joint.state(a, b)
    }

    /// A hinge's angle in radians, or `None` if the joint is not one.
    #[cfg(feature = "mechanism")]
    pub fn joint_angle(&self, id: JointId) -> Option<f32> {
        let state = self.joint_state(id)?;
        (state.kind == crate::joint::AxialKind::Angle).then_some(state.coordinate)
    }

    /// A slider's or screw's travel in world units, or `None` if the joint is
    /// neither.
    #[cfg(feature = "mechanism")]
    pub fn joint_offset(&self, id: JointId) -> Option<f32> {
        let state = self.joint_state(id)?;
        (state.kind == crate::joint::AxialKind::Offset).then_some(state.coordinate)
    }

    /// How fast the joint's coordinate is changing, in rad/s or units/s.
    #[cfg(feature = "mechanism")]
    pub fn joint_speed(&self, id: JointId) -> Option<f32> {
        Some(self.joint_state(id)?.speed)
    }

    // ---- diagnostics ------------------------------------------------------

    /// What the engine repaired, dropped or ignored, and where its time went.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// world.add_body(RigidBody::fixed().shape(Shape::ground()));
    /// world.add_body(RigidBody::dynamic().shape(Shape::ball(0.5))
    ///     .translation(Vector3::new(0.0, 2.0, 0.0)));
    /// world.step(1.0 / 60.0);
    ///
    /// assert!(world.diagnostics().is_clean(), "nothing to report on a sane scene");
    /// assert_eq!(world.diagnostics().counters().bodies, 2);
    /// ```
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// Forget every recorded warning, keeping the step count.
    pub fn clear_warnings(&mut self) {
        self.diagnostics.clear_warnings();
    }

    /// Collect per-stage timings for every step from now on.
    ///
    /// Off by default and free when off — the clock is never read. Not
    /// available on `wasm32`, which has no clock in `std`; there this is a
    /// no-op and [`crate::diagnostics::Timings::supported`] is `false`.
    pub fn set_profiling(&mut self, on: bool) {
        self.diagnostics.set_profiling(on);
    }

    /// Total mechanical energy: kinetic plus gravitational potential, in joules
    /// for a scene in metres and kilograms.
    ///
    /// Not used by the step — it is a *measurement*, and the number it gives is
    /// the one that catches a solver adding energy it should not. With gravity
    /// off and no driven joints, this must not rise; if it does, something is
    /// injecting energy. Potential is measured against the world origin along
    /// `-gravity`, so only its *changes* are meaningful.
    ///
    /// Sleeping bodies contribute their potential but no kinetic energy, which
    /// is what they physically have.
    ///
    /// A body with any rotation axis locked contributes no *rotational* energy:
    /// the engine stores the inverse inertia, and locking makes it singular. It
    /// is exact for a fully locked body, which cannot rotate at all, and drops a
    /// term for a partially locked one.
    pub fn energy(&self) -> f32 {
        let gravity = self.gravity;
        let mut total = 0.0;
        for (_, body) in self.bodies.iter() {
            if !body.is_dynamic() || !body.enabled {
                continue;
            }
            let mass = body.mass();
            if mass > 0.0 && !body.is_sleeping() {
                let v = body.linear_velocity;
                total += 0.5 * mass * v.dot(v);
                let w = body.angular_velocity;
                // Singular — a locked axis — inverts to zero, so those axes
                // simply contribute nothing rather than an infinity.
                total += 0.5 * w.dot(body.world_inv_inertia().inverse().mul_vec(w));
            }
            // Potential rises against gravity: -m g·x with g pointing down.
            total -= mass * gravity.dot(body.world_center_of_mass()) * body.gravity_scale;
        }
        total
    }

    /// Every live contact manifold.
    pub fn contacts(&self) -> &ContactSet {
        &self.contacts
    }

    /// Pairs that started or stopped touching during the last step, including
    /// sensor overlaps.
    pub fn collision_events(&self) -> &[CollisionEvent] {
        self.contacts.events()
    }

    // ---- stepping ---------------------------------------------------------

    /// Advance by `dt` seconds of real time.
    ///
    /// Real time is accumulated and consumed in fixed [`Self::timestep`] chunks,
    /// so the simulation behaves identically whether the frame rate is 30, 60 or
    /// 144 — and a stalled frame cannot destabilise it. Leftover time carries
    /// over to the next call; read [`Self::interpolation_alpha`] to draw
    /// smoothly between steps.
    pub fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            self.diagnostics.warn(Warning::InvalidTimestep, 1);
            return;
        }
        // Discard time beyond what `max_substeps` can consume, rather than
        // letting the accumulator grow without bound and make every subsequent
        // frame worse than the last.
        let budget = self.timestep * self.max_substeps as f32;
        let wanted = self.accumulator + dt;
        if wanted > budget {
            self.diagnostics.warn(Warning::TimeDiscarded, 1);
        }
        self.accumulator = wanted.min(budget);
        let mut steps = 0;
        while self.accumulator >= self.timestep && steps < self.max_substeps {
            self.step_fixed();
            self.accumulator -= self.timestep;
            steps += 1;
        }
        self.substeps = steps;
        self.interpolation_alpha = (self.accumulator / self.timestep).clamp(0.0, 1.0);
    }

    /// Run exactly one fixed step, ignoring the accumulator. Use for
    /// deterministic and headless simulation.
    pub fn step_fixed(&mut self) {
        let dt = self.timestep;
        self.substeps = 1;

        // Every clock read here costs nothing while profiling is off: the
        // stopwatch simply never starts, and reads back zero.
        let whole = self.diagnostics.stopwatch();
        let mut timings = crate::diagnostics::Timings::default();

        let clock = self.diagnostics.stopwatch();
        self.integrate_velocities(dt);
        timings.integration = clock.read();

        let clock = self.diagnostics.stopwatch();
        self.detect_collisions();
        timings.narrow_phase = clock.read();

        let clock = self.diagnostics.stopwatch();
        self.solver.solve(
            &mut self.bodies,
            &mut self.contacts,
            &mut self.joints,
            &mut self.tendons,
            dt,
            &self.solver_config,
        );
        timings.solver = clock.read();

        let clock = self.diagnostics.stopwatch();
        self.integrate_positions(dt);
        self.resolve_ccd();
        timings.integration += clock.read();

        let clock = self.diagnostics.stopwatch();
        self.update_sleeping(dt);
        timings.sleeping = clock.read();

        timings.step = whole.read();
        let counters = self.counters();
        self.diagnostics.end_step(timings, counters);
    }

    /// Pull back any body that crossed something during the step it just took.
    ///
    /// The discrete pass only ever asks "do these overlap *now*", so a body that
    /// moves further in one step than the thing in its way is thick is on the
    /// far side before anything is tested, and no amount of solver iteration
    /// finds a contact that was never generated. Bodies opted in with
    /// [`RigidBody::ccd`] get their step swept instead: cast the shape along the
    /// motion it just made, and if it hit something, put it back at the impact
    /// point. The response is left to the ordinary contact solver on the next
    /// step, which now has an overlap to work with — this only ensures there is
    /// one to find.
    ///
    /// Only bodies that moved further than they are thick are swept. Below that
    /// the discrete pass could not have missed anything, and the cast would be
    /// pure cost.
    fn resolve_ccd(&mut self) {
        let mut stops: Vec<(BodyId, Vector3)> = Vec::new();
        for (id, body) in self.bodies.iter() {
            if !body.ccd_enabled || !body.is_dynamic() || !body.enabled || body.is_sleeping() {
                continue;
            }
            let Some(collider) = body.colliders().first() else {
                continue;
            };
            let from = body.previous_position();
            let motion = body.position.translation - from.translation;
            let distance = motion.length();
            let Some(direction) = crate::math::try_normalize(motion) else {
                continue;
            };
            let bounds = collider.shape.compute_aabb(&crate::math::Isometry::IDENTITY);
            let size = bounds.max - bounds.min;
            let thickness = size.x.min(size.y).min(size.z).max(1e-4);
            if distance <= thickness {
                continue;
            }
            let filter = crate::query::QueryFilter::default().exclude(id);
            let start = collider.world_transform(&from);
            if let Some(hit) = self.cast_shape(&collider.shape, &start, direction, distance, filter)
            {
                // Stop a hair short, so the next step's broad phase sees a pair
                // and the narrow phase a contact, rather than an exact touch
                // that rounds either way.
                let skin = (self.prediction_distance * 0.5).min(hit.toi);
                stops.push((id, from.translation + direction * (hit.toi - skin).max(0.0)));
            }
        }
        for (id, translation) in stops {
            if let Some(body) = self.bodies.get_mut(id) {
                body.position.translation = translation;
            }
        }
    }

    /// What the step just did, in counts.
    fn counters(&mut self) -> crate::diagnostics::Counters {
        // Everything else here is already lying around; the island split is
        // not, because nothing else in the step needs it yet. Building it is a
        // union-find over the contacts and joints — cheap next to the solve,
        // and it reuses this world's buffers.
        self.islands
            .build(&self.bodies, &self.contacts, &self.joints, &self.tendons);
        let island_count = self.islands.len();
        let mut counters = crate::diagnostics::Counters {
            substeps: self.substeps as u32,
            manifolds: self.contacts.len(),
            islands: island_count,
            ..Default::default()
        };
        for (_, body) in self.bodies.iter() {
            counters.bodies += 1;
            if body.is_dynamic() && body.enabled && !body.is_sleeping() {
                counters.awake_bodies += 1;
            }
        }
        for m in self.contacts.iter() {
            counters.contact_points += m.points.len();
        }
        counters
    }

    /// How far into the next physics step the last [`Self::step`] left us, in
    /// `0..1`. Interpolate each body between
    /// [`RigidBody::previous_position`] and [`RigidBody::position`] by this
    /// amount to draw without stutter.
    pub fn interpolation_alpha(&self) -> f32 {
        self.interpolation_alpha
    }

    /// Accelerations from [`Self::gravity_model`], one per body slot.
    ///
    /// Computed as accelerations rather than forces so a body's mass cancels
    /// exactly where the physics says it should: two satellites at the same
    /// radius orbit at the same speed however much they weigh.
    fn model_accelerations(&self) -> Vec<Vector3> {
        let mut out = vec![Vector3::ZERO; self.bodies.slot_count()];
        match self.gravity_model {
            GravityModel::None => {}
            GravityModel::Point {
                centre,
                mu,
                min_distance,
            } => {
                let floor = min_distance.max(1e-4);
                for (i, slot) in out.iter_mut().enumerate() {
                    let Some(b) = self.bodies.by_index(i) else {
                        continue;
                    };
                    if !b.is_dynamic() {
                        continue;
                    }
                    let to = centre - b.world_center_of_mass();
                    let r = to.length().max(floor);
                    *slot = to * (mu / (r * r * r));
                }
            }
            GravityModel::Mutual { g, min_distance } => {
                let floor = min_distance.max(1e-4);
                let n = self.bodies.slot_count();
                for i in 0..n {
                    let Some(a) = self.bodies.by_index(i) else {
                        continue;
                    };
                    for j in (i + 1)..n {
                        let Some(b) = self.bodies.by_index(j) else {
                            continue;
                        };
                        let to = b.world_center_of_mass() - a.world_center_of_mass();
                        let r = to.length().max(floor);
                        let unit = to * (1.0 / r);
                        let scale = g / (r * r);
                        // Equal and opposite forces, so the accelerations differ
                        // by the other body's mass — which is the whole point.
                        if a.is_dynamic() {
                            out[i] = out[i] + unit * (scale * b.mass());
                        }
                        if b.is_dynamic() {
                            out[j] = out[j] - unit * (scale * a.mass());
                        }
                    }
                }
            }
        }
        out
    }

    fn integrate_velocities(&mut self, dt: f32) {
        let gravity = self.gravity;
        let model = self.model_accelerations();
        let limit = self.max_velocity;
        let mut non_finite = 0u32;
        let mut clamped = 0u32;
        let mut non_finite_position = 0u32;
        for (index, (_, body)) in self.bodies.iter_mut().enumerate() {
            let (bad_velocity, bad_position) = body.begin_step();
            if bad_velocity {
                non_finite += 1;
            }
            if bad_position {
                non_finite_position += 1;
            }
            // A kinematic body's velocity comes from where it was told to go,
            // not from forces, so this happens before the integration.
            body.drive_to_kinematic_target(dt);
            if let Some(a) = model.get(index) {
                if *a != Vector3::ZERO && body.is_dynamic() && !body.is_sleeping() {
                    body.linear_velocity = body.linear_velocity + *a * (dt * body.gravity_scale);
                }
            }
            if body.integrate_velocity(dt, gravity) {
                non_finite += 1;
            }
            // A finite but absurd velocity is not a bug the way a NaN is — it is
            // usually a stiff stack venting — so it is capped and counted rather
            // than zeroed.
            if limit.is_finite() && limit > 0.0 {
                let speed = body.linear_velocity.length();
                if speed > limit {
                    body.linear_velocity = body.linear_velocity * (limit / speed);
                    clamped += 1;
                }
            }
        }
        if non_finite > 0 {
            self.diagnostics
                .warn(Warning::NonFiniteVelocity, non_finite as u64);
        }
        if clamped > 0 {
            self.diagnostics
                .warn(Warning::VelocityClamped, clamped as u64);
        }
        if non_finite_position > 0 {
            self.diagnostics
                .warn(Warning::NonFinitePosition, non_finite_position as u64);
        }
    }

    fn integrate_positions(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            body.integrate_position(dt);
            body.reset_forces();
        }
    }

    /// Re-file every body's bounds and re-pair them. Touches nothing else, which
    /// is what makes it safe to run outside a step.
    fn rebuild_broadphase(&mut self) {
        self.broadphase.clear();
        for i in 0..self.bodies.slot_count() {
            let Some(body) = self.bodies.by_index(i) else {
                continue;
            };
            if !body.enabled || body.colliders().is_empty() {
                continue;
            }
            let mut aabb = body.compute_aabb();
            if aabb.is_empty() {
                continue;
            }
            aabb.expand_by_scalar(self.prediction_distance);
            let active = !body.is_fixed() && !body.is_sleeping();
            self.broadphase.add(i as u32, &aabb, active);
        }
        // An external broad phase — a GPU one, say — replaces the pair search
        // but not the bounds: queries still have to know where things are.
        match self.external_pairs.take() {
            Some(pairs) => {
                self.pairs.clear();
                self.pairs.extend(pairs);
            }
            None => self.broadphase.find_pairs(&mut self.pairs),
        }
    }

    fn detect_collisions(&mut self) {
        // Bodies tied by a joint overlap at the joint by design; colliding them
        // makes the contact and the constraint fight.
        self.no_collide.clear();
        for (_, joint) in self.joints.iter() {
            if joint.is_active() && !joint.collide_connected {
                let (a, b) = (joint.body_a.index() as u32, joint.body_b.index() as u32);
                self.no_collide.insert((a.min(b), a.max(b)));
            }
        }

        self.rebuild_broadphase();

        self.contacts.begin_step();
        // The broad phase drops pairs with no active body, so a contact between
        // two sleeping bodies is never regenerated. Hold on to it rather than
        // letting it expire — see `ContactSet::keep_dormant`.
        {
            let Self {
                contacts, bodies, ..
            } = self;
            let dormant = |id| {
                bodies
                    .get(id)
                    .is_none_or(|b| !b.enabled || b.is_fixed() || b.is_sleeping())
            };
            contacts.keep_dormant(|key| dormant(key.body_a) && dormant(key.body_b));
        }
        let pairs = std::mem::take(&mut self.pairs);
        for &(ia, ib) in &pairs {
            if self.no_collide.contains(&(ia, ib)) {
                continue;
            }
            let (Some(id_a), Some(id_b)) = (
                self.bodies.id_at(ia as usize),
                self.bodies.id_at(ib as usize),
            ) else {
                continue;
            };
            let (Some(a), Some(b)) = (
                self.bodies.by_index(ia as usize),
                self.bodies.by_index(ib as usize),
            ) else {
                continue;
            };

            for (ci, ca) in a.colliders().iter().enumerate() {
                if !ca.enabled {
                    continue;
                }
                for (cj, cb) in b.colliders().iter().enumerate() {
                    if !cb.enabled || !ca.groups.test(&cb.groups) {
                        continue;
                    }
                    self.raw.clear();
                    collide(
                        &ca.shape,
                        &ca.world_transform(&a.position),
                        &cb.shape,
                        &cb.world_transform(&b.position),
                        self.prediction_distance,
                        &mut self.raw,
                    );
                    if self.raw.is_empty() {
                        continue;
                    }
                    let material = ca.material.combine(&cb.material);
                    let is_sensor = ca.is_sensor || cb.is_sensor;
                    for m in &self.raw {
                        self.contacts.update(
                            ManifoldKey {
                                body_a: id_a,
                                body_b: id_b,
                                collider_a: ci as u32,
                                collider_b: cj as u32,
                                sub_a: m.sub_a as u32,
                                sub_b: m.sub_b as u32,
                            },
                            m,
                            &a.position,
                            &b.position,
                            material.friction,
                            material.restitution,
                            is_sensor,
                        );
                    }
                }
            }
        }
        self.pairs = pairs;
        self.contacts.end_step();

        self.wake_touching_bodies();
    }

    /// A moving body must wake anything it touches, or it will pass through a
    /// sleeping neighbour that never got the chance to respond.
    fn wake_touching_bodies(&mut self) {
        let threshold = self.sleep.linear_threshold;
        let mut wake: Vec<BodyId> = Vec::new();
        for m in self.contacts.iter().filter(|m| m.touching && !m.is_sensor) {
            let (a, b) = (m.key.body_a, m.key.body_b);
            let moving = |id: BodyId| {
                self.bodies
                    .get(id)
                    .is_some_and(|x| !x.is_sleeping() && x.linear_velocity.length() > threshold)
            };
            let sleeping = |id: BodyId| self.bodies.get(id).is_some_and(|x| x.is_sleeping());
            if moving(a) && sleeping(b) {
                wake.push(b);
            }
            if moving(b) && sleeping(a) {
                wake.push(a);
            }
        }
        // A joint transmits motion too: waking one end must wake the other.
        for (_, joint) in self.joints.iter() {
            if !joint.is_active() {
                continue;
            }
            let awake = |id: BodyId| self.bodies.get(id).is_some_and(|x| !x.is_sleeping());
            let sleeping = |id: BodyId| self.bodies.get(id).is_some_and(|x| x.is_sleeping());
            if awake(joint.body_a) && sleeping(joint.body_b) {
                wake.push(joint.body_b);
            }
            if awake(joint.body_b) && sleeping(joint.body_a) {
                wake.push(joint.body_a);
            }
        }
        for id in wake {
            if let Some(b) = self.bodies.get_mut(id) {
                b.wake_up();
            }
        }
    }

    fn update_sleeping(&mut self, dt: f32) {
        if !self.sleep.enabled {
            return;
        }
        let s = self.sleep;
        for (_, body) in self.bodies.iter_mut() {
            body.update_sleep(dt, s.linear_threshold, s.angular_threshold, s.time_to_sleep);
        }
    }

    /// Wake every body. Useful after changing gravity or moving level geometry.
    pub fn wake_all(&mut self) {
        for (_, body) in self.bodies.iter_mut() {
            body.wake_up();
        }
    }

    // ---- scene graph ------------------------------------------------------

    /// Copy every body's transform onto its bound scene node.
    ///
    /// Bind a node with [`crate::body::RigidBodyBuilder::scene_object`] or by setting
    /// [`RigidBody::scene_object`]. Nodes without a body are untouched.
    ///
    /// Scale is left alone: rigid bodies do not scale, and overwriting it would
    /// silently reset whatever the mesh was authored with.
    pub fn sync_to_scene(&self, arena: &mut ObjectArena) {
        self.sync_to_scene_where(arena, |_, _| true);
    }

    /// Like [`Self::sync_to_scene`], but only for bodies matching `include`.
    ///
    /// Use this with animation↔physics blends: during a handover the blend owns
    /// the drawn pose, so those bodies must be skipped here or the crossfade is
    /// overwritten by the raw simulation pose.
    ///
    /// ```ignore
    /// world.sync_to_scene_where(&mut arena, |id, _| !ragdoll.owns_scene_write(id));
    /// ```
    pub fn sync_to_scene_where(
        &self,
        arena: &mut ObjectArena,
        mut include: impl FnMut(BodyId, &RigidBody) -> bool,
    ) {
        for (body_id, body) in self.bodies.iter() {
            if !include(body_id, body) {
                continue;
            }
            let Some(id) = body.scene_object else { continue };
            let Some(object) = arena.get_mut(id) else {
                continue;
            };
            object.position = body.position.translation;
            object.quaternion = body.position.rotation;
            object.update_matrix();
        }
    }

    /// Copy transforms onto scene nodes, interpolated by
    /// [`Self::interpolation_alpha`].
    ///
    /// Use instead of [`Self::sync_to_scene`] when the render rate is higher
    /// than [`Self::timestep`] — it removes the stutter you get from drawing the
    /// same physics pose across several frames.
    pub fn sync_to_scene_interpolated(&self, arena: &mut ObjectArena) {
        self.sync_to_scene_interpolated_where(arena, |_, _| true);
    }

    /// Interpolated scene sync with a body filter — see [`Self::sync_to_scene_where`].
    pub fn sync_to_scene_interpolated_where(
        &self,
        arena: &mut ObjectArena,
        mut include: impl FnMut(BodyId, &RigidBody) -> bool,
    ) {
        let t = self.interpolation_alpha;
        for (body_id, body) in self.bodies.iter() {
            if !include(body_id, body) {
                continue;
            }
            let Some(id) = body.scene_object else { continue };
            let Some(object) = arena.get_mut(id) else {
                continue;
            };
            let pose = body.previous_position().lerp(&body.position, t);
            object.position = pose.translation;
            object.quaternion = pose.rotation;
            object.update_matrix();
        }
    }

    /// Drive kinematic bodies from their scene nodes.
    ///
    /// Animate the scene however you like — a tween, an [`threers::AnimationMixer`],
    /// hand-written code — then call this before [`Self::step`] and the physics
    /// world will push things out of the way accordingly.
    pub fn sync_from_scene(&mut self, arena: &ObjectArena) {
        for (_, body) in self.bodies.iter_mut() {
            if !body.is_kinematic() {
                continue;
            }
            let Some(id) = body.scene_object else { continue };
            let Some(object) = arena.get(id) else { continue };
            let target = crate::math::Isometry::new(object.position, object.quaternion);
            let dt = self.timestep;
            if dt <= 0.0 {
                continue;
            }
            // Set the velocity that *carries* the body to the target over this
            // step, rather than teleporting it there. Integration then lands it
            // exactly on the target, and — crucially — the contact solver sees a
            // real velocity, so the body pushes things aside instead of passing
            // straight through them.
            body.linear_velocity =
                (target.translation - body.position.translation) * (1.0 / dt);

            // Same for spin: the shortest rotation from here to the target,
            // expressed as an angular velocity.
            let mut delta = target.rotation.multiply(body.position.rotation.conjugate());
            if delta.w < 0.0 {
                delta = threers::math::Quaternion::new(-delta.x, -delta.y, -delta.z, -delta.w);
            }
            let sin_half = (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z).sqrt();
            body.angular_velocity = if sin_half > 1e-6 {
                let angle = 2.0 * sin_half.atan2(delta.w);
                Vector3::new(delta.x, delta.y, delta.z) * (angle / (sin_half * dt))
            } else {
                Vector3::ZERO
            };
            body.wake_up();
        }
    }

    /// Bind an existing body to a scene node.
    pub fn link_scene_object(&mut self, body: BodyId, object: ObjectId) {
        if let Some(b) = self.bodies.get_mut(body) {
            b.scene_object = Some(object);
        }
    }

    // ---- bulk operations --------------------------------------------------

    /// Number of live bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Bodies currently asleep.
    pub fn sleeping_count(&self) -> usize {
        self.bodies.iter().filter(|(_, b)| b.is_sleeping()).count()
    }

    /// Remove everything.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.joints.clear();
        self.contacts.clear();
        self.broadphase.clear();
        self.accumulator = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::Collider;
    use crate::joint::Joint;
    use crate::material::InteractionGroups;
    use crate::shape::Shape;
    use threers::core::Object3D;

    fn ground(world: &mut World) -> BodyId {
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8))
    }

    #[test]
    fn a_ball_falls_and_settles() {
        let mut world = World::new();
        ground(&mut world);
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0)),
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let y = world.body(ball).unwrap().translation().y;
        assert!((y - 0.5).abs() < 0.05, "settled at {y}");
    }

    #[test]
    fn the_fixed_timestep_makes_the_result_frame_rate_independent() {
        let simulate = |frame_dt: f32, frames: usize| {
            let mut world = World::new();
            ground(&mut world);
            let ball = world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.5))
                    .translation(Vector3::new(0.0, 5.0, 0.0))
                    .can_sleep(false),
            );
            for _ in 0..frames {
                world.step(frame_dt);
            }
            world.body(ball).unwrap().translation().y
        };
        // One second of simulation at three very different frame rates.
        let a = simulate(1.0 / 30.0, 30);
        let b = simulate(1.0 / 60.0, 60);
        let c = simulate(1.0 / 120.0, 120);
        assert!((a - b).abs() < 0.02, "30 vs 60 Hz: {a} vs {b}");
        assert!((b - c).abs() < 0.02, "60 vs 120 Hz: {b} vs {c}");
    }

    #[test]
    fn a_huge_frame_time_cannot_run_away() {
        let mut world = World::new();
        ground(&mut world);
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0)),
        );
        // A five second hitch must not try to run 300 steps.
        world.step(5.0);
        assert!(world.interpolation_alpha() <= 1.0);
        // And the world is still sane afterwards.
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        assert!(world.body(world.bodies().ids().nth(1).unwrap()).unwrap().translation().y > 0.4);
    }

    #[test]
    fn settled_bodies_fall_asleep_and_a_new_arrival_wakes_them() {
        let mut world = World::new();
        ground(&mut world);
        let resting = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0)),
        );
        for _ in 0..180 {
            world.step(1.0 / 60.0);
        }
        assert!(world.body(resting).unwrap().is_sleeping(), "box never slept");

        // Drop something on it.
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(0.0, 4.0, 0.0)),
        );
        for _ in 0..120 {
            world.step(1.0 / 60.0);
            if !world.body(resting).unwrap().is_sleeping() {
                return;
            }
        }
        panic!("the sleeping box was never woken by the falling ball");
    }

    #[test]
    fn collision_groups_let_bodies_pass_through_each_other() {
        const A: u32 = 1 << 0;
        const B: u32 = 1 << 1;
        let mut world = World::new();
        // A floor that only interacts with group A.
        world.add_body(
            RigidBody::fixed()
                .collider(Collider::new(Shape::ground()).groups(InteractionGroups::new(A, A))),
        );
        let ghost = world.add_body(
            RigidBody::dynamic()
                .collider(Collider::new(Shape::ball(0.5)).groups(InteractionGroups::new(B, B)))
                .translation(Vector3::new(0.0, 2.0, 0.0))
                .can_sleep(false),
        );
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        assert!(
            world.body(ghost).unwrap().translation().y < -1.0,
            "the ghost should have fallen through, it is at {}",
            world.body(ghost).unwrap().translation().y
        );
    }

    #[test]
    fn a_sensor_reports_overlap_without_pushing() {
        let mut world = World::new();
        world.add_body(
            RigidBody::fixed()
                .collider(Collider::new(Shape::cuboid(2.0, 0.1, 2.0)).sensor(true))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(0.0, 3.0, 0.0))
                .can_sleep(false),
        );
        let mut saw_start = false;
        for _ in 0..120 {
            world.step(1.0 / 60.0);
            if world.collision_events().iter().any(|e| e.started && e.is_sensor) {
                saw_start = true;
            }
        }
        assert!(saw_start, "the sensor never reported an overlap");
        assert!(
            world.body(ball).unwrap().translation().y < 0.0,
            "the sensor blocked the ball instead of letting it pass"
        );
    }

    #[test]
    fn contact_events_fire_on_landing() {
        let mut world = World::new();
        let floor = ground(&mut world);
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 2.0, 0.0)),
        );
        let mut started = 0;
        for _ in 0..180 {
            world.step(1.0 / 60.0);
            started += world
                .collision_events()
                .iter()
                .filter(|e| {
                    e.started
                        && ((e.body_a == ball && e.body_b == floor)
                            || (e.body_a == floor && e.body_b == ball))
                })
                .count();
        }
        assert!(started >= 1, "landing produced no contact event");
    }

    #[test]
    fn removing_a_body_removes_its_joints_and_contacts() {
        let mut world = World::new();
        ground(&mut world);
        let a = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0)),
        );
        let b = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 2.0, 0.0)),
        );
        world.add_joint(Joint::distance(a, b, Vector3::ZERO, Vector3::ZERO, 1.5));
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        assert_eq!(world.joints().len(), 1);
        world.remove_body(b);
        assert_eq!(world.joints().len(), 0);
        assert!(world.contacts().contacts_with(b).count() == 0);
        assert!(world.body(b).is_none());
        // And stepping afterwards must not panic on the dangling handle.
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
    }

    #[test]
    fn jointed_bodies_do_not_fight_a_contact_between_themselves() {
        // Two overlapping spheres welded together: without contact filtering the
        // contact pushes them apart while the joint pulls them together, and the
        // pair buzzes.
        let mut world = World::new();
        let a = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 0.0, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        let b = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.4, 0.0, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        world.add_joint(Joint::distance(a, b, Vector3::ZERO, Vector3::ZERO, 0.4));
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        let d = (world.body(b).unwrap().translation() - world.body(a).unwrap().translation()).length();
        assert!((d - 0.4).abs() < 0.02, "welded spheres drifted to {d} apart");
        let v = world.body(a).unwrap().linear_velocity.length();
        assert!(v < 0.05, "the pair is buzzing at {v}");
    }

    #[test]
    fn transforms_are_written_onto_bound_scene_nodes() {
        let mut arena = ObjectArena::new();
        let node = arena.insert(Object3D::group());

        let mut world = World::new();
        ground(&mut world);
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0))
                .scene_object(node),
        );
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }
        world.sync_to_scene(&mut arena);

        let body_y = world.body(ball).unwrap().translation().y;
        assert!((arena.get(node).unwrap().position.y - body_y).abs() < 1e-5);
        assert!(body_y < 5.0, "the ball never fell");
        // Scale must be left alone.
        assert_eq!(arena.get(node).unwrap().scale, Vector3::ONE);
    }

    #[test]
    fn interpolated_sync_lands_between_the_two_poses() {
        let mut arena = ObjectArena::new();
        let node = arena.insert(Object3D::group());
        let mut world = World::new();
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0))
                .scene_object(node),
        );
        // Step by a partial timestep so the accumulator leaves a remainder.
        world.step(1.0 / 60.0);
        world.step(1.0 / 240.0);
        world.sync_to_scene_interpolated(&mut arena);

        let body = world.body(ball).unwrap();
        let (lo, hi) = (body.position.translation.y, body.previous_position().translation.y);
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        let drawn = arena.get(node).unwrap().position.y;
        assert!(drawn >= lo - 1e-5 && drawn <= hi + 1e-5, "{drawn} not in [{lo}, {hi}]");
    }

    #[test]
    fn a_kinematic_body_driven_from_the_scene_pushes_dynamic_ones() {
        let mut arena = ObjectArena::new();
        let node = arena.insert(Object3D::group());

        let mut world = World::new();
        let piston = world.add_body(
            RigidBody::kinematic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(-3.0, 0.5, 0.0))
                .scene_object(node),
        );
        let crate_body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );

        // Drive the piston across by animating the scene node.
        for i in 0..120 {
            arena.get_mut(node).unwrap().position = Vector3::new(-3.0 + i as f32 * 0.02, 0.5, 0.0);
            world.sync_from_scene(&arena);
            world.step(1.0 / 60.0);
        }
        assert!(
            world.body(crate_body).unwrap().translation().x > 0.3,
            "the kinematic piston did not push the crate, it is at {:?}",
            world.body(crate_body).unwrap().translation()
        );
        // And the piston itself went exactly where the scene put it.
        assert!((world.body(piston).unwrap().translation().x - (-3.0 + 119.0 * 0.02)).abs() < 1e-3);
    }

    #[test]
    fn gravity_can_be_changed_and_reversed() {
        let mut world = World::new().with_gravity(Vector3::new(0.0, 5.0, 0.0));
        let ball = world.add_body(RigidBody::dynamic().shape(Shape::ball(0.5)).can_sleep(false));
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }
        assert!(world.body(ball).unwrap().translation().y > 1.0, "it should have risen");
    }

    #[test]
    fn a_body_with_no_collider_still_integrates() {
        let mut world = World::new();
        let ghost = world.add_body(
            RigidBody::dynamic()
                .mass(1.0)
                .translation(Vector3::new(0.0, 10.0, 0.0))
                .can_sleep(false),
        );
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }
        assert!(world.body(ghost).unwrap().translation().y < 9.0);
    }

    #[test]
    fn an_empty_world_steps_without_complaint() {
        let mut world = World::new();
        for _ in 0..10 {
            world.step(1.0 / 60.0);
        }
        assert_eq!(world.body_count(), 0);
        // Nonsense frame times are ignored rather than poisoning the accumulator.
        world.step(f32::NAN);
        world.step(-1.0);
        world.step(0.0);
        assert_eq!(world.body_count(), 0);
    }
}
