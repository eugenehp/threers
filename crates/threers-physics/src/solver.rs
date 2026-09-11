//! Constraint solver: sequential impulses with warm starting.
//!
//! Each step runs three passes over the constraint set:
//!
//! 1. **Warm start** — re-apply the impulses that worked last step. This is what
//!    lets a stack converge in a couple of iterations instead of dozens.
//! 2. **Velocity** — iterate contacts and joints, correcting relative velocity.
//!    Friction is solved before the normal so it uses a settled normal impulse.
//! 3. **Position** — a second solve on *pseudo* velocities that pushes
//!    overlapping bodies apart. Keeping this separate from the real velocities
//!    is why boxes do not launch themselves off the floor when they start the
//!    step interpenetrating.

use crate::body::{BodySet, RigidBody};
use crate::contact::{ContactManifold, ContactSet};
use crate::joint::{Dof, DofMotion, Joint, JointKind, JointSet, JointSpring, Softness};
use crate::math::{orthonormal_basis, try_normalize, Mat3};
use crate::tendon::{ResolvedPath, Tendon, TendonKind, TendonSet};
use threers::math::{Quaternion, Vector3};

/// How a constraint's bias and impulse are scaled — rigid or spring-like.
///
/// A rigid constraint pushes its error toward zero with a fixed fraction per
/// step (`baumgarte`) and never gives ground. A soft one behaves like a spring
/// of a chosen frequency instead, which needs two more numbers: the solved
/// impulse is scaled down, and a fraction of the impulse applied so far is
/// handed back. That last term is what makes the constraint *yield* rather than
/// merely converge more slowly.
///
/// Both cases go through the same arithmetic, with the rigid one falling out as
/// `mass_scale = 1`, `impulse_scale = 0`.
#[derive(Debug, Clone, Copy)]
struct SoftParams {
    bias_rate: f32,
    mass_scale: f32,
    impulse_scale: f32,
}

impl SoftParams {
    fn rigid(dt: f32, baumgarte: f32) -> Self {
        Self {
            bias_rate: baumgarte / dt,
            mass_scale: 1.0,
            impulse_scale: 0.0,
        }
    }

    /// Whether this constraint is allowed to yield.
    ///
    /// A rigid constraint hands its position error to the pseudo velocities, so
    /// that closing a gap does not also hand the bodies momentum. A soft one
    /// cannot: its restoring force *is* the momentum, and a spring whose push
    /// gets thrown away at the end of the step is not a spring. So the two take
    /// different routes for the same error, and this is the fork.
    fn is_soft(&self) -> bool {
        self.impulse_scale != 0.0
    }

    fn new(softness: Softness, dt: f32, config: &SolverConfig) -> Self {
        if softness.is_rigid() {
            return Self::rigid(dt, config.baumgarte);
        }
        // The frequency is turned into these three numbers without ever naming a
        // stiffness, which is why the same `Softness` behaves the same on a
        // marble and on a shipping container: the constraint's effective mass
        // has already divided out.
        let omega = std::f32::consts::TAU * softness.frequency;
        let a1 = 2.0 * softness.damping_ratio + dt * omega;
        let a2 = dt * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimulationQuality {
    /// Fewest iterations. Fine for debris and particles, visibly soft under a
    /// stack.
    Fast,
    /// The default — what [`SolverConfig::default`] already is.
    #[default]
    Balanced,
    /// Twice the iterations. For tall stacks and long jointed chains, where
    /// sequential impulses need the passes to carry information end to end.
    High,
}

impl SimulationQuality {
    /// The tuning this preset stands for.
    pub fn config(self) -> SolverConfig {
        let base = SolverConfig::default();
        match self {
            Self::Fast => SolverConfig {
                velocity_iterations: 4,
                position_iterations: 2,
                ..base
            },
            Self::Balanced => base,
            Self::High => SolverConfig {
                velocity_iterations: base.velocity_iterations * 2,
                position_iterations: base.position_iterations * 2,
                ..base
            },
        }
    }

    /// Names for a UI to populate a menu from.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::High => "high",
        }
    }
}


/// Solver tuning. The defaults are chosen for a 60 Hz step and metre-scale
/// scenes; the doc on each field says which way to move it.
///
/// # Tall stacks need more than the default
///
/// Sequential impulses carry information roughly one contact per iteration, so
/// how well a stack converges depends on its height against the iteration
/// budget. Measured, as steps for a settling stack of unit boxes to fall asleep:
///
/// | boxes high | 4×4 (default) | 8×8 |
/// |---|---|---|
/// | 1–5 | ~30 | ~30 |
/// | 10 | ~1100 | ~54 |
/// | 15 | ~2100 | ~96 |
///
/// Below six high the default settles almost immediately. Above that the top of
/// the stack is still learning about the floor several iterations late, and the
/// column sways — visibly, and for seconds — before the error works its way out.
/// It is *stable*, not unstable: nothing sinks or explodes, it just rings. If
/// your scenes are built out of tall stacks, raise [`crate::world::World::substeps`] and
/// `velocity_iterations` together rather than either alone.
///
/// Alternating the sweep direction between iterations (symmetric Gauss-Seidel),
/// which is the textbook remedy, was tried and measured here: it helps at high
/// iteration counts and makes the default *worse* — 10 high went from ~1100
/// steps to ~2800 — because reversing the order fights the warm start. It is
/// deliberately not done.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Velocity iterations. More is more rigid and more expensive. Below about
    /// 4, tall stacks sag.
    pub velocity_iterations: usize,
    /// Position (overlap-recovery) iterations.
    pub position_iterations: usize,
    /// Fraction of remaining overlap corrected per step, in `0..1`. Higher
    /// separates faster but overshoots.
    pub baumgarte: f32,
    /// Overlap left uncorrected. A little slop stops resting contacts from
    /// oscillating between touching and separated.
    pub penetration_slop: f32,
    /// Cap on how fast overlap recovery may push bodies apart, in units/s.
    /// Without it, a deeply overlapping pair explodes.
    pub max_correction_velocity: f32,
    /// Impacts slower than this do not bounce, however elastic the material.
    /// Prevents a resting body from buzzing against the floor.
    pub restitution_threshold: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            baumgarte: 0.2,
            penetration_slop: 0.005,
            max_correction_velocity: 3.0,
            restitution_threshold: 1.0,
        }
    }
}

/// Per-body scratch state.
///
/// The solver works on this flat array rather than on [`RigidBody`] directly:
/// contacts touch the same body from many constraints, and a plain indexable
/// array sidesteps the borrow juggling entirely.
#[derive(Debug, Clone, Copy)]
struct SolverBody {
    linear: Vector3,
    angular: Vector3,
    pseudo_linear: Vector3,
    pseudo_angular: Vector3,
    inv_mass: Vector3,
    inv_inertia: Mat3,
    com: Vector3,
    /// Whether impulses may move this body at all.
    movable: bool,
}

impl Default for SolverBody {
    fn default() -> Self {
        Self {
            linear: Vector3::ZERO,
            angular: Vector3::ZERO,
            pseudo_linear: Vector3::ZERO,
            pseudo_angular: Vector3::ZERO,
            inv_mass: Vector3::ZERO,
            inv_inertia: Mat3::ZERO,
            com: Vector3::ZERO,
            movable: false,
        }
    }
}

impl SolverBody {
    fn from_body(body: &RigidBody) -> Self {
        let movable = body.is_dynamic() && body.enabled && !body.is_sleeping();
        Self {
            linear: body.linear_velocity,
            angular: body.angular_velocity,
            pseudo_linear: Vector3::ZERO,
            pseudo_angular: Vector3::ZERO,
            // A non-movable body keeps zero inverse mass, so every impulse
            // applied to it is silently absorbed — exactly the behaviour of an
            // infinitely heavy object.
            inv_mass: if movable { body.inv_mass_axes() } else { Vector3::ZERO },
            inv_inertia: if movable { body.world_inv_inertia() } else { Mat3::ZERO },
            com: body.world_center_of_mass(),
            movable,
        }
    }

    #[inline]
    fn velocity_at(&self, r: Vector3) -> Vector3 {
        self.linear + self.angular.cross(r)
    }

    #[inline]
    fn pseudo_velocity_at(&self, r: Vector3) -> Vector3 {
        self.pseudo_linear + self.pseudo_angular.cross(r)
    }

    #[inline]
    fn apply_impulse(&mut self, r: Vector3, impulse: Vector3) {
        self.linear = self.linear + mul_axes(self.inv_mass, impulse);
        self.angular = self.angular + self.inv_inertia.mul_vec(r.cross(impulse));
    }

    #[inline]
    fn apply_pseudo_impulse(&mut self, r: Vector3, impulse: Vector3) {
        self.pseudo_linear = self.pseudo_linear + mul_axes(self.inv_mass, impulse);
        self.pseudo_angular = self.pseudo_angular + self.inv_inertia.mul_vec(r.cross(impulse));
    }

    #[inline]
    fn apply_angular_impulse(&mut self, impulse: Vector3) {
        self.angular = self.angular + self.inv_inertia.mul_vec(impulse);
    }

    /// `1 / (J M⁻¹ Jᵀ)` for a unit constraint direction applied at `r`.
    fn effective_mass(&self, r: Vector3, dir: Vector3) -> f32 {
        let rn = r.cross(dir);
        mul_axes(self.inv_mass, dir).dot(dir) + rn.dot(self.inv_inertia.mul_vec(rn))
    }

    /// Apply a constraint whose angular row is *not* simply `r × impulse`.
    ///
    /// Every joint above pushes at a point, so its torque falls out of the lever
    /// arm and [`Self::apply_impulse`] can derive it. A coupling does not work
    /// that way: a screw's thread converts force into torque through its lead,
    /// and a gear applies pure torque at no point at all. Those need the two
    /// halves of the Jacobian given separately.
        #[inline]
    fn apply_generalized(&mut self, linear: Vector3, angular: Vector3) {
        self.linear = self.linear + mul_axes(self.inv_mass, linear);
        self.angular = self.angular + self.inv_inertia.mul_vec(angular);
    }

    /// `1 / (J M⁻¹ Jᵀ)` for a Jacobian given as separate linear and angular
    /// rows.
        #[inline]
    fn generalized_mass(&self, linear: Vector3, angular: Vector3) -> f32 {
        mul_axes(self.inv_mass, linear).dot(linear)
            + angular.dot(self.inv_inertia.mul_vec(angular))
    }
}

#[inline]
fn mul_axes(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// One contact point, prepared for solving.
#[derive(Debug, Clone, Copy)]
struct PointConstraint {
    r_a: Vector3,
    r_b: Vector3,
    normal_mass: f32,
    tangent_mass: [f32; 2],
    /// Target relative normal velocity: restitution bounce, or the closing rate
    /// a speculative contact is still allowed.
    velocity_bias: f32,
    depth: f32,
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
    position_impulse: f32,
    /// Where to write the impulses back for next step's warm start.
    manifold: usize,
    point: usize,
}

#[derive(Debug, Clone)]
struct ContactConstraint {
    body_a: usize,
    body_b: usize,
    normal: Vector3,
    tangent: [Vector3; 2],
    friction: f32,
    points: Vec<PointConstraint>,
}

/// Reusable solver scratch space.
#[derive(Debug, Default, Clone)]
pub struct Solver {
    bodies: Vec<SolverBody>,
    contacts: Vec<ContactConstraint>,
    /// One row per tendon, rebuilt each substep because the path moves with the
    /// bodies it is threaded through.
    tendons: Vec<TendonRow>,
    /// Joint anchors in world space, gathered once per substep. Only the
    /// coupling-phase integrator reads them, so without `mechanism` the field
    /// and the per-substep clone that fills it are both dead weight.
    #[cfg(feature = "mechanism")]
    frames: Vec<JointFrame>,
}

impl Solver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Solve one step's contacts and joints, writing velocities and overlap
    /// corrections back into `bodies`.
    pub fn solve(
        &mut self,
        bodies: &mut BodySet,
        contacts: &mut ContactSet,
        joints: &mut JointSet,
        tendons: &mut TendonSet,
        dt: f32,
        config: &SolverConfig,
    ) {
        if dt <= 0.0 {
            return;
        }

        self.bodies.clear();
        self.bodies.resize(bodies.slot_count(), SolverBody::default());
        for i in 0..bodies.slot_count() {
            if let Some(b) = bodies.by_index(i) {
                self.bodies[i] = SolverBody::from_body(b);
            }
        }

        // Tendons index the same body slots the rows above do, so they are
        // reduced once the scratch exists and before anything is solved.
        let mut cables: Vec<&mut Tendon> = tendons.iter_mut().map(|(_, t)| t).collect();
        self.prepare_tendons(&cables, bodies);

        // Speculative manifolds (every point still a hair apart) are solved too.
        // They carry a `velocity_bias` that permits closing exactly the
        // remaining gap and no more. Skipping them lets a body that has just
        // been pushed out of contact accelerate freely for a frame, which is
        // what keeps a settled stack quietly buzzing instead of falling asleep.
        let mut manifolds: Vec<&mut ContactManifold> =
            contacts.iter_mut().filter(|m| !m.is_sensor).collect();
        self.prepare_contacts(&manifolds, dt, config);

        // Joints need their anchors in world space; gather once.
        let mut joint_frames: Vec<JointFrame> = Vec::new();
        for (_, joint) in joints.iter() {
            joint_frames.push(JointFrame::build(joint, bodies, dt));
        }

        self.warm_start(joints, &joint_frames, &mut cables);

        for _ in 0..config.velocity_iterations {
            self.solve_joint_velocities(joints, &joint_frames, dt, config);
            self.solve_contact_velocities();
            self.solve_tendon_lengths(&mut cables, dt, config);
        }

        // The couplings integrate their leftover on the settled velocities, so
        // this runs after the iteration loop rather than inside it.
        #[cfg(feature = "mechanism")]
        {
            self.frames = joint_frames.clone();
            let mut driven: Vec<&mut Joint> = joints.values_mut().collect();
            self.integrate_coupling_phase(&mut driven, dt);
        }

        for _ in 0..config.position_iterations {
            self.solve_contact_positions(dt, config);
        }

        // Write impulses back for next step's warm start.
        for c in &self.contacts {
            for p in &c.points {
                let m = &mut manifolds[p.manifold];
                m.points[p.point].normal_impulse = p.normal_impulse;
                m.points[p.point].tangent_impulse = p.tangent_impulse;
            }
        }

        // Break joints whose impulse exceeded their tolerance.
        for joint in joints.values_mut() {
            if let Some(limit) = joint.break_impulse {
                if !joint.broken && joint.applied_impulse() > limit {
                    joint.broken = true;
                    joint.reset_impulses();
                }
            }
        }

        for i in 0..self.bodies.len() {
            let state = self.bodies[i];
            if !state.movable {
                continue;
            }
            let Some(body) = bodies.by_index_mut(i) else {
                continue;
            };
            body.linear_velocity = state.linear;
            body.angular_velocity = state.angular;
            body.apply_position_correction(state.pseudo_linear, state.pseudo_angular, dt);
        }
    }

    fn prepare_contacts(
        &mut self,
        manifolds: &[&mut ContactManifold],
        dt: f32,
        config: &SolverConfig,
    ) {
        self.contacts.clear();
        for (mi, m) in manifolds.iter().enumerate() {
            let ia = m.key.body_a.index();
            let ib = m.key.body_b.index();
            if ia >= self.bodies.len() || ib >= self.bodies.len() {
                continue;
            }
            if !self.bodies[ia].movable && !self.bodies[ib].movable {
                continue;
            }
            let (t1, t2) = orthonormal_basis(m.normal);
            let mut constraint = ContactConstraint {
                body_a: ia,
                body_b: ib,
                normal: m.normal,
                tangent: [t1, t2],
                friction: m.friction,
                points: Vec::with_capacity(m.points.len()),
            };

            for (pi, p) in m.points.iter().enumerate() {
                let a = &self.bodies[ia];
                let b = &self.bodies[ib];
                let r_a = p.point_a - a.com;
                let r_b = p.point_b - b.com;

                let inv_normal_mass =
                    a.effective_mass(r_a, m.normal) + b.effective_mass(r_b, m.normal);
                if inv_normal_mass <= 0.0 {
                    continue; // neither body can respond along this axis
                }

                let mut tangent_mass = [0.0f32; 2];
                for (k, t) in [t1, t2].iter().enumerate() {
                    let inv = a.effective_mass(r_a, *t) + b.effective_mass(r_b, *t);
                    tangent_mass[k] = if inv > 0.0 { 1.0 / inv } else { 0.0 };
                }

                let relative = a.velocity_at(r_a) - b.velocity_at(r_b);
                let vn = relative.dot(m.normal);
                let bounce = if vn < -config.restitution_threshold {
                    -m.restitution * vn
                } else {
                    0.0
                };
                let velocity_bias = if p.depth >= 0.0 {
                    bounce
                } else if bounce > 0.0 {
                    // A fast approach into a speculative contact must bounce, or
                    // the speculative term brakes the body to a standstill at the
                    // margin and the impact is silently eaten. Bouncing a step
                    // early, a couple of centimetres out, is imperceptible;
                    // losing the bounce entirely is not.
                    bounce
                } else {
                    // Otherwise close exactly the remaining gap, no further.
                    p.depth / dt
                };

                constraint.points.push(PointConstraint {
                    r_a,
                    r_b,
                    normal_mass: 1.0 / inv_normal_mass,
                    tangent_mass,
                    velocity_bias,
                    depth: p.depth,
                    normal_impulse: p.normal_impulse,
                    tangent_impulse: p.tangent_impulse,
                    position_impulse: 0.0,
                    manifold: mi,
                    point: pi,
                });
            }

            if !constraint.points.is_empty() {
                self.contacts.push(constraint);
            }
        }
    }

    fn warm_start(
        &mut self,
        joints: &mut JointSet,
        frames: &[JointFrame],
        tendons: &mut [&mut Tendon],
    ) {
        for ci in 0..self.contacts.len() {
            let c = &self.contacts[ci];
            let (ia, ib, normal, tangent) = (c.body_a, c.body_b, c.normal, c.tangent);
            for pi in 0..self.contacts[ci].points.len() {
                let p = self.contacts[ci].points[pi];
                let impulse = normal * p.normal_impulse
                    + tangent[0] * p.tangent_impulse[0]
                    + tangent[1] * p.tangent_impulse[1];
                self.apply_pair(ia, ib, p.r_a, p.r_b, impulse);
            }
        }
        for ((_, joint), frame) in joints.iter_mut().zip(frames.iter()) {
            if !joint.is_active() || !frame.valid {
                continue;
            }
            self.apply_pair(
                frame.body_a,
                frame.body_b,
                frame.r_a,
                frame.r_b,
                joint.linear_impulse,
            );
            if let Some(a) = self.bodies.get_mut(frame.body_a) {
                a.apply_angular_impulse(-joint.angular_impulse);
            }
            if let Some(b) = self.bodies.get_mut(frame.body_b) {
                b.apply_angular_impulse(joint.angular_impulse);
            }
            // Springs on the joint's own coordinate warm-start too — their
            // impulses survive `reset_step_impulses` for exactly that reason.
            self.warm_start_springs(joint, frame);
            // Only the accumulators that were *not* just applied are cleared.
            // See `Joint::reset_step_impulses` for why the other two have to
            // keep running.
            joint.reset_step_impulses();
        }
        for k in 0..self.tendons.len() {
            let Some(tendon) = tendons.get_mut(k) else {
                continue;
            };
            if !tendon.is_active() || !self.tendons[k].valid {
                continue;
            }
            // Carried across substeps like a joint's main impulse rather than
            // cleared like a limit's: a cable holding a load is the case this
            // exists for, and rediscovering the same tension from zero four
            // times a step is what makes it sag visibly while doing it.
            self.apply_tendon_impulse(k, tendon.impulse);
        }
    }

    /// Re-apply last substep's spring impulses along this substep's axes.
    ///
    /// The axes are re-derived rather than stored, because the bodies have
    /// moved: an impulse remembered as a scalar about "the hinge axis" only
    /// means anything against wherever that axis points now.
    fn warm_start_springs(&mut self, joint: &Joint, frame: &JointFrame) {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let angular = |s: &mut Self, axis: Vector3, lambda: f32| {
            if lambda == 0.0 {
                return;
            }
            if let Some(x) = s.bodies.get_mut(ia) {
                x.apply_angular_impulse(axis * -lambda);
            }
            if let Some(x) = s.bodies.get_mut(ib) {
                x.apply_angular_impulse(axis * lambda);
            }
        };
        match &joint.kind {
            JointKind::Revolute { spring, .. } if spring.is_some() => {
                angular(self, frame.axis_a, joint.spring_impulse[3]);
            }
            JointKind::Prismatic { spring, .. } if spring.is_some() => {
                let lambda = joint.spring_impulse[0];
                if lambda != 0.0 {
                    self.apply_pair(ia, ib, frame.r_a, frame.r_b, frame.axis_a * lambda);
                }
            }
            JointKind::Generic {
                linear,
                angular: a,
                ..
            } => {
                for (i, dof) in linear.iter().enumerate() {
                    let lambda = joint.spring_impulse[i];
                    if dof.spring.is_none() || lambda == 0.0 {
                        continue;
                    }
                    // The flipped axis `solve_generic` measures on, so the sign
                    // of the remembered impulse still means what it did.
                    let axis = frame.joint_axes[i] * -1.0;
                    self.apply_pair(ia, ib, frame.r_a, frame.r_b, axis * lambda);
                }
                for (i, dof) in a.iter().enumerate() {
                    if dof.spring.is_none() {
                        continue;
                    }
                    angular(self, frame.joint_axes[i], joint.spring_impulse[3 + i]);
                }
            }
            _ => {}
        }
    }

    /// The same, on the pseudo velocities — position correction that moves the
    /// bodies without handing them momentum.
    fn apply_pseudo_pair(
        &mut self,
        a: usize,
        b: usize,
        r_a: Vector3,
        r_b: Vector3,
        impulse: Vector3,
    ) {
        if let Some(x) = self.bodies.get_mut(a) {
            x.apply_pseudo_impulse(r_a, impulse);
        }
        if let Some(x) = self.bodies.get_mut(b) {
            x.apply_pseudo_impulse(r_b, -impulse);
        }
    }

    /// Apply `+impulse` to `a` and `-impulse` to `b`.
    fn apply_pair(&mut self, a: usize, b: usize, r_a: Vector3, r_b: Vector3, impulse: Vector3) {
        if let Some(x) = self.bodies.get_mut(a) {
            x.apply_impulse(r_a, impulse);
        }
        if let Some(x) = self.bodies.get_mut(b) {
            x.apply_impulse(r_b, -impulse);
        }
    }

    fn solve_contact_velocities(&mut self) {
        for ci in 0..self.contacts.len() {
            let (ia, ib) = (self.contacts[ci].body_a, self.contacts[ci].body_b);
            let normal = self.contacts[ci].normal;
            let tangent = self.contacts[ci].tangent;
            let friction = self.contacts[ci].friction;

            for pi in 0..self.contacts[ci].points.len() {
                // Friction first: it needs a normal impulse to bound itself, and
                // last iteration's is a better estimate than zero.
                let normal_impulse = self.contacts[ci].points[pi].normal_impulse;
                let max_friction = friction * normal_impulse;
                for (k, &direction) in tangent.iter().enumerate() {
                    let p = self.contacts[ci].points[pi];
                    if p.tangent_mass[k] == 0.0 {
                        continue;
                    }
                    let relative = self.bodies[ia].velocity_at(p.r_a)
                        - self.bodies[ib].velocity_at(p.r_b);
                    let vt = relative.dot(direction);
                    let delta = -p.tangent_mass[k] * vt;
                    let old = p.tangent_impulse[k];
                    // Clamping the *accumulated* impulse rather than the
                    // increment is what makes the friction cone hold.
                    let new = (old + delta).clamp(-max_friction, max_friction);
                    self.contacts[ci].points[pi].tangent_impulse[k] = new;
                    self.apply_pair(ia, ib, p.r_a, p.r_b, tangent[k] * (new - old));
                }

                let p = self.contacts[ci].points[pi];
                let relative =
                    self.bodies[ia].velocity_at(p.r_a) - self.bodies[ib].velocity_at(p.r_b);
                let vn = relative.dot(normal);
                let delta = -p.normal_mass * (vn - p.velocity_bias);
                let old = p.normal_impulse;
                // Contacts push, never pull.
                let new = (old + delta).max(0.0);
                self.contacts[ci].points[pi].normal_impulse = new;
                self.apply_pair(ia, ib, p.r_a, p.r_b, normal * (new - old));
            }
        }
    }

    fn solve_contact_positions(&mut self, dt: f32, config: &SolverConfig) {
        for ci in 0..self.contacts.len() {
            let (ia, ib) = (self.contacts[ci].body_a, self.contacts[ci].body_b);
            let normal = self.contacts[ci].normal;
            for pi in 0..self.contacts[ci].points.len() {
                let p = self.contacts[ci].points[pi];
                let error = p.depth - config.penetration_slop;
                if error <= 0.0 {
                    continue;
                }
                let target =
                    (config.baumgarte * error / dt).min(config.max_correction_velocity);
                let relative = self.bodies[ia].pseudo_velocity_at(p.r_a)
                    - self.bodies[ib].pseudo_velocity_at(p.r_b);
                let vn = relative.dot(normal);
                let delta = p.normal_mass * (target - vn);
                let old = p.position_impulse;
                let new = (old + delta).max(0.0);
                self.contacts[ci].points[pi].position_impulse = new;
                let impulse = normal * (new - old);
                if let Some(x) = self.bodies.get_mut(ia) {
                    x.apply_pseudo_impulse(p.r_a, impulse);
                }
                if let Some(x) = self.bodies.get_mut(ib) {
                    x.apply_pseudo_impulse(p.r_b, -impulse);
                }
            }
        }
    }

    // ---- joints -----------------------------------------------------------

    fn solve_joint_velocities(
        &mut self,
        joints: &mut JointSet,
        frames: &[JointFrame],
        dt: f32,
        config: &SolverConfig,
    ) {
        for ((_, joint), frame) in joints.iter_mut().zip(frames.iter()) {
            if !joint.is_active() || !frame.valid {
                continue;
            }
            let (ia, ib) = (frame.body_a, frame.body_b);
            if ia >= self.bodies.len() || ib >= self.bodies.len() {
                continue;
            }
            if !self.bodies[ia].movable && !self.bodies[ib].movable {
                continue;
            }

            let soft = SoftParams::new(joint.softness, dt, config);
            // Limits and motors stay rigid however soft the joint is: a limit
            // that yields is not a limit, and a motor that yields is a motor
            // that cannot reach its target speed.
            let hard = SoftParams::rigid(dt, config.baumgarte);

            match &joint.kind {
                JointKind::Spherical => {
                    self.solve_point_constraint(joint, frame, soft);
                }
                JointKind::Fixed { rest_rotation } => {
                    let error = fixed_angular_error(frame, *rest_rotation);
                    self.solve_angular_constraint(joint, frame, error, soft);
                    self.solve_point_constraint(joint, frame, soft);
                }
                JointKind::Revolute {
                    limits,
                    motor,
                    spring,
                    ..
                } => {
                    let (limits, motor, spring) = (*limits, *motor, *spring);
                    let axis = frame.axis_a;
                    let (t1, t2) = orthonormal_basis(axis);
                    // Two angular constraints keep the axes aligned.
                    let misalignment = frame.axis_a.cross(frame.axis_b);
                    for (slot, t) in [(3usize, t1), (4, t2)] {
                        joint.dof_impulse[slot] = self.solve_scalar_angular(
                            ia,
                            ib,
                            t,
                            misalignment.dot(t),
                            joint.dof_impulse[slot],
                            soft,
                        );
                    }
                    // The return spring, before the drive and the limit: it is a
                    // force the joint always exerts, and both of those get to
                    // overrule it in the same iteration. Slot 3 is the one
                    // `warm_start_springs` re-applies along this same axis.
                    if let Some(s) = spring.filter(JointSpring::is_active) {
                        joint.spring_impulse[3] = self.solve_angular_spring(
                            ia,
                            ib,
                            axis,
                            frame.hinge_angle,
                            s,
                            joint.spring_impulse[3],
                            dt,
                            config,
                        );
                    }
                    if let Some(m) = motor {
                        let impulse = self.solve_motor(
                            ia,
                            ib,
                            axis,
                            m.target_velocity,
                            m.max_force * dt,
                            joint.motor_impulse,
                            true,
                        );
                        joint.motor_impulse = impulse;
                    }
                    // The drive runs before the limit, so a servo commanded past
                    // the end of travel is corrected by the limit in the same
                    // iteration rather than a step later.
                    #[cfg(feature = "mechanism")]
                    if let Some(s) = joint.servo {
                        joint.servo_impulse = self.solve_angular_servo(
                            ia,
                            ib,
                            axis,
                            frame.hinge_angle,
                            s,
                            joint.servo_impulse,
                            dt,
                            config,
                        );
                    }
                    if let Some(l) = limits {
                        let angle = frame.hinge_angle;
                        joint.limit_impulse = self.solve_limit(
                            ia,
                            ib,
                            axis,
                            angle,
                            l.min,
                            l.max,
                            joint.limit_impulse,
                            hard,
                            true,
                        );
                    }
                    self.solve_point_constraint(joint, frame, soft);
                }
                JointKind::Prismatic {
                    limits,
                    motor,
                    spring,
                    ..
                } => {
                    let (limits, motor, spring) = (*limits, *motor, *spring);
                    let axis = frame.axis_a;
                    // Lock all rotation.
                    let error = fixed_angular_error(frame, frame.rest_rotation);
                    self.solve_angular_constraint(joint, frame, error, soft);
                    // Lock the two translations across the axis.
                    let (t1, t2) = orthonormal_basis(axis);
                    let delta = frame.anchor_a - frame.anchor_b;
                    for (slot, t) in [(0usize, t1), (1, t2)] {
                        joint.dof_impulse[slot] = self.solve_scalar_linear(
                            frame,
                            t,
                            delta.dot(t),
                            joint.dof_impulse[slot],
                            soft,
                        );
                    }
                    if let Some(s) = spring.filter(JointSpring::is_active) {
                        joint.spring_impulse[0] = self.solve_linear_spring(
                            frame,
                            axis,
                            delta.dot(axis),
                            s,
                            joint.spring_impulse[0],
                            dt,
                            config,
                        );
                    }
                    if let Some(m) = motor {
                        let impulse = self.solve_linear_motor(
                            frame,
                            axis,
                            m.target_velocity,
                            m.max_force * dt,
                            joint.motor_impulse,
                        );
                        joint.motor_impulse = impulse;
                    }
                    #[cfg(feature = "mechanism")]
                    if let Some(s) = joint.servo {
                        joint.servo_impulse = self.solve_linear_servo(
                            frame,
                            axis,
                            delta.dot(axis),
                            s,
                            joint.servo_impulse,
                            dt,
                            config,
                        );
                    }
                    if let Some(l) = limits {
                        joint.limit_impulse = self.solve_linear_limit(
                            frame,
                            axis,
                            delta.dot(axis),
                            l.min,
                            l.max,
                            joint.limit_impulse,
                            hard,
                        );
                    }
                }
                JointKind::Distance { min, max } => {
                    let (min, max) = (*min, *max);
                    let delta = frame.anchor_a - frame.anchor_b;
                    let length = delta.length();
                    let Some(dir) = try_normalize(delta) else {
                        continue;
                    };
                    let error = if length > max {
                        length - max
                    } else if length < min {
                        length - min
                    } else {
                        continue; // within range — a rope with slack does nothing
                    };
                    joint.axial_impulse =
                        self.solve_scalar_linear(frame, dir, error, joint.axial_impulse, soft);
                }
                JointKind::Spring {
                    rest_length,
                    stiffness,
                    damping,
                } => {
                    let (rest_length, stiffness, damping) = (*rest_length, *stiffness, *damping);
                    let delta = frame.anchor_a - frame.anchor_b;
                    let length = delta.length();
                    let Some(dir) = try_normalize(delta) else {
                        continue;
                    };
                    let relative = self.bodies[ia].velocity_at(frame.r_a)
                        - self.bodies[ib].velocity_at(frame.r_b);
                    // An explicit force, not a constraint: a spring is meant to
                    // be soft, so it must not be able to win against contacts.
                    let force = -stiffness * (length - rest_length) - damping * relative.dot(dir);
                    let impulse = dir * (force * dt);
                    self.apply_pair(ia, ib, frame.r_a, frame.r_b, impulse);
                    joint.axial_impulse = force * dt;
                }
                #[cfg(feature = "mechanism")]
                JointKind::Gear { ratio, .. } => {
                    let ratio = *ratio;
                    let (n_a, n_b) = (frame.axis_a, frame.axis_b);
                    // C = (ω_a − ω_c)·n_a − ratio (ω_b − ω_c)·n_b, so the
                    // carrier's row is whatever the other two leave over. With
                    // no carrier it drops out and this is the world-referenced
                    // constraint it has always been.
                    let j_c = n_b * ratio - n_a;
                    let ic = frame.body_c;
                    let mut k = n_a.dot(self.bodies[ia].inv_inertia.mul_vec(n_a))
                        + ratio * ratio * n_b.dot(self.bodies[ib].inv_inertia.mul_vec(n_b));
                    let mut rate = self.bodies[ia].angular.dot(n_a)
                        - ratio * self.bodies[ib].angular.dot(n_b);
                    if let Some(c) = ic.and_then(|i| self.bodies.get(i)) {
                        k += j_c.dot(c.inv_inertia.mul_vec(j_c));
                        rate += c.angular.dot(j_c);
                    }
                    if k <= 0.0 {
                        continue;
                    }
                    let lambda = -(rate + soft.bias_rate * joint.coupling_phase) / k
                        * soft.mass_scale
                        - joint.coupling_impulse * soft.impulse_scale;
                    if let Some(x) = self.bodies.get_mut(ia) {
                        x.apply_angular_impulse(n_a * lambda);
                    }
                    if let Some(x) = self.bodies.get_mut(ib) {
                        x.apply_angular_impulse(n_b * (-ratio * lambda));
                    }
                    if let Some(x) = ic.and_then(|i| self.bodies.get_mut(i)) {
                        x.apply_angular_impulse(j_c * lambda);
                    }
                    joint.coupling_impulse += lambda;
                }
                #[cfg(feature = "mechanism")]
                JointKind::RackPinion { radius, .. } => {
                    let radius = *radius;
                    let (slide, spin) = (frame.axis_b, frame.axis_a);
                    // B advances along its own axis as A turns about its own, so
                    // the angular row carries the pitch radius as well as the
                    // usual lever arm.
                    let j_a = frame.r_a.cross(slide) + spin * radius;
                    let j_b = frame.r_b.cross(slide) + spin * radius;
                    let k = self.bodies[ia].generalized_mass(slide, j_a)
                        + self.bodies[ib].generalized_mass(slide, j_b);
                    if k <= 0.0 {
                        continue;
                    }
                    let rate = (self.bodies[ib].velocity_at(frame.r_b)
                        - self.bodies[ia].velocity_at(frame.r_a))
                    .dot(slide)
                        - radius * (self.bodies[ia].angular - self.bodies[ib].angular).dot(spin);
                    let lambda = -(rate + soft.bias_rate * joint.coupling_phase) / k
                        * soft.mass_scale
                        - joint.coupling_impulse * soft.impulse_scale;
                    if let Some(x) = self.bodies.get_mut(ia) {
                        x.apply_generalized(slide * -lambda, j_a * -lambda);
                    }
                    if let Some(x) = self.bodies.get_mut(ib) {
                        x.apply_generalized(slide * lambda, j_b * lambda);
                    }
                    joint.coupling_impulse += lambda;
                }
                #[cfg(feature = "mechanism")]
                JointKind::Screw {
                    lead,
                    rest_offset,
                    rest_angle,
                    limits,
                    motor,
                    ..
                } => {
                    let (lead, rest_offset, rest_angle) = (*lead, *rest_offset, *rest_angle);
                    let (limits, motor) = (*limits, *motor);
                    let axis = frame.axis_a;
                    // A screw is a hinge and a slider that are not allowed to
                    // disagree: hold the axes together and the two off-axis
                    // translations, then tie what is left to itself.
                    let misalignment = frame.axis_a.cross(frame.axis_b);
                    let (t1, t2) = orthonormal_basis(axis);
                    for (slot, t) in [(3usize, t1), (4, t2)] {
                        joint.dof_impulse[slot] = self.solve_scalar_angular(
                            ia,
                            ib,
                            t,
                            misalignment.dot(t),
                            joint.dof_impulse[slot],
                            soft,
                        );
                    }
                    let delta = frame.anchor_a - frame.anchor_b;
                    for (slot, t) in [(0usize, t1), (1, t2)] {
                        joint.dof_impulse[slot] = self.solve_scalar_linear(
                            frame,
                            t,
                            delta.dot(t),
                            joint.dof_impulse[slot],
                            soft,
                        );
                    }

                    let offset = delta.dot(axis);

                    // Drives first, then the thread, then the stop. A motor is
                    // an actuator with a ceiling and the thread is a hard
                    // constraint, so the thread has to be solved after it —
                    // otherwise a bolt driven against a bottomed-out limit
                    // keeps spinning happily while going nowhere, which is not
                    // what a thread does.
                    if let Some(m) = motor {
                        joint.motor_impulse = self.solve_motor(
                            ia,
                            ib,
                            axis,
                            m.target_velocity,
                            m.max_force * dt,
                            joint.motor_impulse,
                            true,
                        );
                    }
                    #[cfg(feature = "mechanism")]
                    if let Some(s) = joint.servo {
                        joint.servo_impulse = self.solve_linear_servo(
                            frame,
                            axis,
                            offset,
                            s,
                            joint.servo_impulse,
                            dt,
                            config,
                        );
                    }
                    self.solve_screw_coupling(joint, frame, axis, offset, rest_offset, rest_angle, lead, soft);
                    if let Some(l) = limits {
                        joint.limit_impulse = self.solve_linear_limit(
                            frame,
                            axis,
                            offset,
                            l.min,
                            l.max,
                            joint.limit_impulse,
                            hard,
                        );
                    }
                }
                JointKind::Generic {
                    linear, angular, ..
                } => {
                    let (linear, angular) = (*linear, *angular);
                    self.solve_generic(joint, frame, &linear, &angular, soft, hard, dt, config);
                }
            }
        }
    }

    /// Six independently configured degrees of freedom.
    ///
    /// The three linear axes act on the anchor separation, the three angular
    /// ones on the orientation error, and each is locked, limited, free, or
    /// driven on its own. Locked axes are solved before limits and motors so a
    /// drive works against a settled frame rather than a drifting one.
    #[allow(clippy::too_many_arguments)]
    fn solve_generic(
        &mut self,
        joint: &mut Joint,
        frame: &JointFrame,
        linear: &[Dof; 3],
        angular: &[Dof; 3],
        soft: SoftParams,
        hard: SoftParams,
        dt: f32,
        config: &SolverConfig,
    ) {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let delta = frame.anchor_a - frame.anchor_b;

        // A joint with all three translations locked is a point constraint, and
        // solving it as one — coupled, through the full 3×3 matrix — converges
        // far better than three separate axes fighting each other.
        if linear.iter().all(|d| d.motion == DofMotion::Locked && d.motor.is_none()) {
            self.solve_point_constraint(joint, frame, soft);
        } else {
            for (i, dof) in linear.iter().enumerate() {
                // Everything the user sees is **B relative to A**: a limit of
                // `-0.3..0` lets B hang 0.3 below A, and a motor target of +2
                // drives B along +axis. The scalar solvers measure the other way
                // round, so they are handed the flipped axis and the coordinate
                // works out with no sign juggling at the call sites.
                let axis = frame.joint_axes[i] * -1.0;
                let coordinate = -delta.dot(frame.joint_axes[i]);
                match dof.motion {
                    DofMotion::Locked => {
                        joint.dof_impulse[i] = self.solve_scalar_linear(
                            frame,
                            axis,
                            coordinate,
                            joint.dof_impulse[i],
                            soft,
                        );
                    }
                    DofMotion::Limited(l) => {
                        joint.dof_impulse[i] = self.solve_linear_limit(
                            frame,
                            axis,
                            coordinate,
                            l.min,
                            l.max,
                            joint.dof_impulse[i],
                            hard,
                        );
                    }
                    DofMotion::Free => {}
                }
                if let Some(s) = dof.spring.filter(JointSpring::is_active) {
                    joint.spring_impulse[i] = self.solve_linear_spring(
                        frame,
                        axis,
                        coordinate,
                        s,
                        joint.spring_impulse[i],
                        dt,
                        config,
                    );
                }
                if let Some(m) = dof.motor {
                    joint.dof_motor_impulse[i] = self.solve_linear_motor(
                        frame,
                        axis,
                        m.target_velocity,
                        m.max_force * frame.dt,
                        joint.dof_motor_impulse[i],
                    );
                }
            }
        }

        // Likewise: three locked rotations is a weld, and the coupled solve is
        // both cheaper and stiffer than three scalar ones.
        if angular.iter().all(|d| d.motion == DofMotion::Locked && d.motor.is_none()) {
            let error = fixed_angular_error(frame, frame.rest_rotation);
            self.solve_angular_constraint(joint, frame, error, soft);
        } else {
            let error = fixed_angular_error(frame, frame.rest_rotation);
            for (i, dof) in angular.iter().enumerate() {
                let axis = frame.joint_axes[i];
                let slot = 3 + i;
                match dof.motion {
                    DofMotion::Locked => {
                        joint.dof_impulse[slot] = self.solve_scalar_angular(
                            ia,
                            ib,
                            axis,
                            error.dot(axis),
                            joint.dof_impulse[slot],
                            soft,
                        );
                    }
                    DofMotion::Limited(l) => {
                        joint.dof_impulse[slot] = self.solve_limit(
                            ia,
                            ib,
                            axis,
                            frame.euler[i],
                            l.min,
                            l.max,
                            joint.dof_impulse[slot],
                            hard,
                            true,
                        );
                    }
                    DofMotion::Free => {}
                }
                if let Some(s) = dof.spring.filter(JointSpring::is_active) {
                    joint.spring_impulse[slot] = self.solve_angular_spring(
                        ia,
                        ib,
                        axis,
                        frame.euler[i],
                        s,
                        joint.spring_impulse[slot],
                        dt,
                        config,
                    );
                }
                if let Some(m) = dof.motor {
                    joint.dof_motor_impulse[slot] = self.solve_motor(
                        ia,
                        ib,
                        axis,
                        m.target_velocity,
                        m.max_force * frame.dt,
                        joint.dof_motor_impulse[slot],
                        true,
                    );
                }
            }
        }
    }

    /// Three-DOF point-to-point constraint holding the anchors together.
    fn solve_point_constraint(&mut self, joint: &mut Joint, frame: &JointFrame, soft: SoftParams) {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let (a, b) = (self.bodies[ia], self.bodies[ib]);
        let k = point_constraint_matrix(&a, &b, frame.r_a, frame.r_b);
        let inv_k = k.inverse();
        if inv_k == Mat3::ZERO {
            return;
        }
        let error = frame.anchor_a - frame.anchor_b;
        // The velocity constraint, on the real velocities. A soft joint carries
        // its position error here too — see `SoftParams::is_soft`.
        let relative = a.velocity_at(frame.r_a) - b.velocity_at(frame.r_b);
        let bias = if soft.is_soft() {
            error * soft.bias_rate
        } else {
            Vector3::ZERO
        };
        let impulse = inv_k.mul_vec(-(relative + bias)) * soft.mass_scale
            - joint.linear_impulse * soft.impulse_scale;
        joint.linear_impulse = joint.linear_impulse + impulse;
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, impulse);

        // The position error, on the *pseudo* velocities — the same split the
        // contact solver already uses.
        //
        // Discrete integration walks a rotating anchor off its circle a little
        // every step. Correcting that by adding real velocity is momentum the
        // joint never earned, and on a fast hinge it compounds: the drift feeds
        // the correction, the correction feeds the spin, and a free door with
        // nothing driving it reverses and spins up. A pseudo impulse moves the
        // bodies and is thrown away.
        if !soft.is_soft() && error != Vector3::ZERO && soft.bias_rate > 0.0 {
            let (pa, pb) = (self.bodies[ia], self.bodies[ib]);
            let pseudo_rel =
                pa.pseudo_velocity_at(frame.r_a) - pb.pseudo_velocity_at(frame.r_b);
            let correction =
                inv_k.mul_vec(-(pseudo_rel + error * soft.bias_rate)) * soft.mass_scale;
            self.apply_pseudo_pair(ia, ib, frame.r_a, frame.r_b, correction);
        }
    }

    /// Three-DOF angular constraint holding a relative orientation.
    ///
    /// Angular constraints all measure **`b` relative to `a`** — the same sense
    /// the error vectors are expressed in — and apply `+impulse` to `b`.
    fn solve_angular_constraint(
        &mut self,
        joint: &mut Joint,
        frame: &JointFrame,
        error: Vector3,
        soft: SoftParams,
    ) {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let k = self.bodies[ia].inv_inertia.add(&self.bodies[ib].inv_inertia);
        let inv_k = k.inverse();
        if inv_k == Mat3::ZERO {
            return;
        }
        let relative = self.bodies[ib].angular - self.bodies[ia].angular;
        // Drive the error toward zero: the target relative spin undoes it.
        let impulse = inv_k.mul_vec(-(relative + error * soft.bias_rate)) * soft.mass_scale
            - joint.angular_impulse * soft.impulse_scale;
        joint.angular_impulse = joint.angular_impulse + impulse;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(-impulse);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(impulse);
        }
    }

    /// One-DOF angular constraint about `axis`, returning the new accumulated
    /// impulse.
    fn solve_scalar_angular(
        &mut self,
        ia: usize,
        ib: usize,
        axis: Vector3,
        error: f32,
        accumulated: f32,
        soft: SoftParams,
    ) -> f32 {
        let inv_mass = axis.dot(self.bodies[ia].inv_inertia.mul_vec(axis))
            + axis.dot(self.bodies[ib].inv_inertia.mul_vec(axis));
        if inv_mass <= 0.0 {
            return accumulated;
        }
        // Velocity on the real spins, alignment error on the pseudo ones — the
        // same split as the point constraint, and for the same reason: a hinge
        // whose axes are re-aligned by handing the bodies real angular momentum
        // does not conserve it.
        let relative = (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let bias = if soft.is_soft() {
            soft.bias_rate * error
        } else {
            0.0
        };
        let lambda =
            -(relative + bias) / inv_mass * soft.mass_scale - accumulated * soft.impulse_scale;
        let impulse = axis * lambda;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(-impulse);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(impulse);
        }

        if !soft.is_soft() && error != 0.0 && soft.bias_rate > 0.0 {
            let pseudo_rel =
                (self.bodies[ib].pseudo_angular - self.bodies[ia].pseudo_angular).dot(axis);
            let correction =
                axis * (-(pseudo_rel + soft.bias_rate * error) / inv_mass * soft.mass_scale);
            if let Some(x) = self.bodies.get_mut(ia) {
                x.pseudo_angular = x.pseudo_angular - x.inv_inertia.mul_vec(correction);
            }
            if let Some(x) = self.bodies.get_mut(ib) {
                x.pseudo_angular = x.pseudo_angular + x.inv_inertia.mul_vec(correction);
            }
        }

        accumulated + lambda
    }

    /// One-DOF linear constraint along `dir`, returning the new accumulated
    /// impulse.
    fn solve_scalar_linear(
        &mut self,
        frame: &JointFrame,
        dir: Vector3,
        error: f32,
        accumulated: f32,
        soft: SoftParams,
    ) -> f32 {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let inv_mass = self.bodies[ia].effective_mass(frame.r_a, dir)
            + self.bodies[ib].effective_mass(frame.r_b, dir);
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let relative = (self.bodies[ia].velocity_at(frame.r_a)
            - self.bodies[ib].velocity_at(frame.r_b))
        .dot(dir);
        // Same split as the point constraint: a rigid constraint's position
        // error goes to the pseudo velocities, not the real ones. Handing a
        // stiff chain real momentum to close its gaps is a pump — the stretch
        // buys velocity, the velocity buys more stretch, and twenty links of it
        // ends up moving faster than anything in the scene threw it. A soft one
        // still needs the error on the real velocities, because there it *is*
        // the restoring force.
        let bias = if soft.is_soft() {
            soft.bias_rate * error
        } else {
            0.0
        };
        let lambda =
            -(relative + bias) / inv_mass * soft.mass_scale - accumulated * soft.impulse_scale;
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, dir * lambda);

        if !soft.is_soft() && error != 0.0 && soft.bias_rate > 0.0 {
            let pseudo_rel = (self.bodies[ia].pseudo_velocity_at(frame.r_a)
                - self.bodies[ib].pseudo_velocity_at(frame.r_b))
            .dot(dir);
            let correction =
                -(pseudo_rel + soft.bias_rate * error) / inv_mass * soft.mass_scale;
            self.apply_pseudo_pair(ia, ib, frame.r_a, frame.r_b, dir * correction);
        }
        accumulated + lambda
    }

    /// Drive a rotational axis toward a target speed, bounded by `max_impulse`.
    #[allow(clippy::too_many_arguments)]
    fn solve_motor(
        &mut self,
        ia: usize,
        ib: usize,
        axis: Vector3,
        target: f32,
        max_impulse: f32,
        accumulated: f32,
        angular: bool,
    ) -> f32 {
        debug_assert!(angular);
        let inv_mass = axis.dot(self.bodies[ia].inv_inertia.mul_vec(axis))
            + axis.dot(self.bodies[ib].inv_inertia.mul_vec(axis));
        if inv_mass <= 0.0 {
            return accumulated;
        }
        // `target` is the speed of `b` about the axis relative to `a` — the way
        // "the hinge turns at 3 rad/s" naturally reads.
        let relative = (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let lambda = (target - relative) / inv_mass;
        let new = (accumulated + lambda).clamp(-max_impulse, max_impulse);
        let applied = new - accumulated;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(axis * -applied);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(axis * applied);
        }
        new
    }

    fn solve_linear_motor(
        &mut self,
        frame: &JointFrame,
        axis: Vector3,
        target: f32,
        max_impulse: f32,
        accumulated: f32,
    ) -> f32 {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let inv_mass = self.bodies[ia].effective_mass(frame.r_a, axis)
            + self.bodies[ib].effective_mass(frame.r_b, axis);
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let relative = (self.bodies[ia].velocity_at(frame.r_a)
            - self.bodies[ib].velocity_at(frame.r_b))
        .dot(axis);
        let lambda = (target - relative) / inv_mass;
        let new = (accumulated + lambda).clamp(-max_impulse, max_impulse);
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, axis * (new - accumulated));
        new
    }

    // ---- tendons ----------------------------------------------------------

    /// One pass over this core's tendons.
    ///
    /// Sign convention throughout: a **positive impulse shortens the path**, so
    /// a tendon's accumulated impulse reads directly as the tension it is
    /// carrying and a pull-only cable is one clamped to stay non-negative.
    /// Resolve every tendon against the bodies as they stand and reduce each to
    /// one constraint row.
    ///
    /// Rebuilt per substep rather than warm-started as geometry: the path runs
    /// where the bodies are now, and a row built against last substep's tangent
    /// points pulls in the wrong direction as soon as anything turns.
    fn prepare_tendons(&mut self, tendons: &[&mut Tendon], bodies: &BodySet) {
        let mut rows = Vec::with_capacity(tendons.len());
        for tendon in tendons {
            if !tendon.is_active() {
                rows.push(TendonRow::default());
                continue;
            }
            let path = tendon.resolve(bodies);
            let locals: Vec<(usize, usize)> = path
                .segments()
                .iter()
                .map(|s| (s.a_body.index(), s.b_body.index()))
                .collect();
            rows.push(TendonRow::build(&path, &locals, &self.bodies));
        }
        self.tendons = rows;
    }

    fn solve_tendon_lengths(
        &mut self,
        tendons: &mut [&mut Tendon],
        dt: f32,
        config: &SolverConfig,
    ) {
        for k in 0..self.tendons.len() {
            let Some(tendon) = tendons.get_mut(k) else {
                continue;
            };
            if !tendon.is_active() {
                continue;
            }
            let (valid, inv_mass, length) = {
                let row = &self.tendons[k];
                (row.valid, row.inv_mass, row.length)
            };
            if !valid || inv_mass <= 0.0 {
                continue;
            }
            let rate = self.tendon_rate(k);
            let accumulated = tendon.impulse;

            // `lambda` is the change this iteration wants; `clamp` is the range
            // the *accumulated* impulse is allowed to end up in.
            let (lambda, clamp) = match tendon.kind {
                TendonKind::Limit { min, max } => {
                    let error = if length > max {
                        length - max
                    } else if length < min {
                        length - min
                    } else {
                        // Within range: slack, and a slack cable pulls nothing.
                        tendon.impulse = 0.0;
                        continue;
                    };
                    let hard = SoftParams::rigid(dt, config.baumgarte);
                    let lambda = (rate + hard.bias_rate * error) / inv_mass;
                    // Too long pulls, too short pushes, and neither is allowed
                    // to change its mind — an accumulator that crosses zero is
                    // a cable applying the force of a rod.
                    let clamp = if error > 0.0 {
                        (0.0, f32::INFINITY)
                    } else {
                        (f32::NEG_INFINITY, 0.0)
                    };
                    (lambda, clamp)
                }
                TendonKind::Force { tension } => {
                    // A constant pull is not solved for, it is *held*: the
                    // accumulated impulse should be `tension·dt` however many
                    // iterations run, so each one asks only for the shortfall.
                    // Applying it per iteration instead would multiply the
                    // tension by the iteration count.
                    let want = tension.max(0.0) * dt;
                    (want - accumulated, (0.0, f32::INFINITY))
                }
                TendonKind::Spring {
                    rest_length,
                    stiffness,
                    damping,
                } => {
                    // `spring_lambda` measures a coordinate its impulse
                    // increases; a tendon's impulse shortens the path. Handing
                    // it the negated error and rate is handing it `-length` —
                    // how much has been reeled in — and the impulse comes back
                    // in the tendon's own sign.
                    let spring = JointSpring::new(stiffness, 0.0, damping);
                    let lambda = spring_lambda(
                        spring,
                        rest_length - length,
                        -rate,
                        inv_mass,
                        accumulated,
                        dt,
                    );
                    (lambda, (0.0, f32::INFINITY))
                }
                TendonKind::Servo {
                    target,
                    max_force,
                    max_speed,
                    softness,
                } => {
                    // A winch that has reeled to its target is a rope of that
                    // length: it hauls the path in while it is too long and
                    // lets it be while it is not. Solving it two-sided instead
                    // — clamping the impulse positive but leaving the velocity
                    // term running below the target — makes a ratchet, which is
                    // exactly what it sounds like: every attempt to pay line
                    // back out is resisted and the length walks to zero.
                    if length <= target {
                        tendon.impulse = 0.0;
                        continue;
                    }
                    let params = SoftParams::new(softness, dt, config);
                    let bias = servo_bias(params.bias_rate, length - target, max_speed);
                    let lambda = (rate + bias) / inv_mass * params.mass_scale
                        - accumulated * params.impulse_scale;
                    (lambda, (0.0, max_force.max(0.0) * dt))
                }
            };

            let new = (accumulated + lambda).clamp(clamp.0, clamp.1);
            let applied = new - accumulated;
            tendon.impulse = new;
            self.apply_tendon_impulse(k, applied);
        }
    }

    /// How fast the path is lengthening, in units per second.
    fn tendon_rate(&self, k: usize) -> f32 {
        self.tendons[k]
            .rows
            .iter()
            .map(|(i, lin, ang)| {
                let body = &self.bodies[*i];
                body.linear.dot(*lin) + body.angular.dot(*ang)
            })
            .sum()
    }

    /// Pull the path in by `impulse`, spread over every body it touches.
    fn apply_tendon_impulse(&mut self, k: usize, impulse: f32) {
        if impulse == 0.0 {
            return;
        }
        for r in 0..self.tendons[k].rows.len() {
            let (i, lin, ang) = self.tendons[k].rows[r];
            if let Some(body) = self.bodies.get_mut(i) {
                body.apply_generalized(lin * -impulse, ang * -impulse);
            }
        }
    }

    /// A spring on a rotational coordinate, pulling it back toward its rest
    /// angle.
    #[allow(clippy::too_many_arguments)]
    fn solve_angular_spring(
        &mut self,
        ia: usize,
        ib: usize,
        axis: Vector3,
        coordinate: f32,
        spring: JointSpring,
        accumulated: f32,
        dt: f32,
        _config: &SolverConfig,
    ) -> f32 {
        let inv_mass = axis.dot(self.bodies[ia].inv_inertia.mul_vec(axis))
            + axis.dot(self.bodies[ib].inv_inertia.mul_vec(axis));
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let rate = (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let lambda = spring_lambda(
            spring,
            coordinate - spring.rest,
            rate,
            inv_mass,
            accumulated,
            dt,
        );
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(axis * -lambda);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(axis * lambda);
        }
        accumulated + lambda
    }

    /// The same along a linear coordinate — a sprung slide.
    #[allow(clippy::too_many_arguments)]
    fn solve_linear_spring(
        &mut self,
        frame: &JointFrame,
        axis: Vector3,
        coordinate: f32,
        spring: JointSpring,
        accumulated: f32,
        dt: f32,
        _config: &SolverConfig,
    ) -> f32 {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let inv_mass = self.bodies[ia].effective_mass(frame.r_a, axis)
            + self.bodies[ib].effective_mass(frame.r_b, axis);
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let rate = (self.bodies[ia].velocity_at(frame.r_a) - self.bodies[ib].velocity_at(frame.r_b))
            .dot(axis);
        let lambda = spring_lambda(
            spring,
            coordinate - spring.rest,
            rate,
            inv_mass,
            accumulated,
            dt,
        );
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, axis * lambda);
        accumulated + lambda
    }

    /// One-sided angular limit: pushes back only while outside `[min, max]`.
    #[allow(clippy::too_many_arguments)]
    fn solve_limit(
        &mut self,
        ia: usize,
        ib: usize,
        axis: Vector3,
        value: f32,
        min: f32,
        max: f32,
        accumulated: f32,
        soft: SoftParams,
        angular: bool,
    ) -> f32 {
        debug_assert!(angular);
        let error = if value < min {
            value - min // negative: the angle must increase
        } else if value > max {
            value - max // positive: the angle must decrease
        } else {
            return 0.0; // inside the range — the limit is not engaged
        };
        let inv_mass = axis.dot(self.bodies[ia].inv_inertia.mul_vec(axis))
            + axis.dot(self.bodies[ib].inv_inertia.mul_vec(axis));
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let relative = (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let lambda = -(relative + soft.bias_rate * error) / inv_mass;
        // A limit only ever pushes back toward the allowed range, so the
        // accumulated impulse is clamped to one sign.
        let new = if error < 0.0 {
            (accumulated + lambda).max(0.0)
        } else {
            (accumulated + lambda).min(0.0)
        };
        let applied = new - accumulated;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(axis * -applied);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(axis * applied);
        }
        new
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_linear_limit(
        &mut self,
        frame: &JointFrame,
        axis: Vector3,
        value: f32,
        min: f32,
        max: f32,
        accumulated: f32,
        soft: SoftParams,
    ) -> f32 {
        let error = if value < min {
            value - min
        } else if value > max {
            value - max
        } else {
            return 0.0;
        };
        let (ia, ib) = (frame.body_a, frame.body_b);
        let inv_mass = self.bodies[ia].effective_mass(frame.r_a, axis)
            + self.bodies[ib].effective_mass(frame.r_b, axis);
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let relative = (self.bodies[ia].velocity_at(frame.r_a)
            - self.bodies[ib].velocity_at(frame.r_b))
        .dot(axis);
        let lambda = -(relative + soft.bias_rate * error) / inv_mass;
        let new = if error < 0.0 {
            (accumulated + lambda).max(0.0)
        } else {
            (accumulated + lambda).min(0.0)
        };
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, axis * (new - accumulated));
        new
    }

    // ---- machine elements -------------------------------------------------

    /// Hold a hinge at an angle, within a torque ceiling.
    ///
    /// The same arithmetic as every other constraint here, with two
    /// differences: the error is measured against a target rather than against
    /// zero, and the accumulated impulse is clamped — which is what turns a
    /// constraint into an actuator that can be beaten.
    #[cfg(feature = "mechanism")]
    #[allow(clippy::too_many_arguments)]
    fn solve_angular_servo(
        &mut self,
        ia: usize,
        ib: usize,
        axis: Vector3,
        coordinate: f32,
        servo: crate::joint::Servo,
        accumulated: f32,
        dt: f32,
        config: &SolverConfig,
    ) -> f32 {
        let inv_mass = axis.dot(self.bodies[ia].inv_inertia.mul_vec(axis))
            + axis.dot(self.bodies[ib].inv_inertia.mul_vec(axis));
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let params = SoftParams::new(servo.softness, dt, config);
        let bias = servo_bias(params.bias_rate, coordinate - servo.target, servo.max_speed);
        let relative = (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let lambda =
            -(relative + bias) / inv_mass * params.mass_scale - accumulated * params.impulse_scale;
        let new = (accumulated + lambda).clamp(-servo.max_force * dt, servo.max_force * dt);
        let applied = new - accumulated;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_angular_impulse(axis * -applied);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_angular_impulse(axis * applied);
        }
        new
    }

    /// The same for a slider or a screw, measured along the axis.
    #[cfg(feature = "mechanism")]
    #[allow(clippy::too_many_arguments)]
    fn solve_linear_servo(
        &mut self,
        frame: &JointFrame,
        axis: Vector3,
        coordinate: f32,
        servo: crate::joint::Servo,
        accumulated: f32,
        dt: f32,
        config: &SolverConfig,
    ) -> f32 {
        let (ia, ib) = (frame.body_a, frame.body_b);
        let inv_mass = self.bodies[ia].effective_mass(frame.r_a, axis)
            + self.bodies[ib].effective_mass(frame.r_b, axis);
        if inv_mass <= 0.0 {
            return accumulated;
        }
        let params = SoftParams::new(servo.softness, dt, config);
        let bias = servo_bias(params.bias_rate, coordinate - servo.target, servo.max_speed);
        let relative = (self.bodies[ia].velocity_at(frame.r_a)
            - self.bodies[ib].velocity_at(frame.r_b))
        .dot(axis);
        let lambda =
            -(relative + bias) / inv_mass * params.mass_scale - accumulated * params.impulse_scale;
        let new = (accumulated + lambda).clamp(-servo.max_force * dt, servo.max_force * dt);
        self.apply_pair(ia, ib, frame.r_a, frame.r_b, axis * (new - accumulated));
        new
    }

    /// Tie a screw's travel to its rotation.
    ///
    /// Both halves of the helix are measured from the pose the joint was built
    /// in, so the constraint is on a real coordinate rather than an integrated
    /// rate — a thread cannot lose count the way a gear can.
    #[cfg(feature = "mechanism")]
    #[allow(clippy::too_many_arguments)]
    fn solve_screw_coupling(
        &mut self,
        joint: &mut Joint,
        frame: &JointFrame,
        axis: Vector3,
        offset: f32,
        rest_offset: f32,
        rest_angle: f32,
        lead: f32,
        soft: SoftParams,
    ) {
        let (ia, ib) = (frame.body_a, frame.body_b);
        // The axial coordinate runs A-relative-to-B and the angle runs
        // B-relative-to-A, so the lead enters with a minus to come out as the
        // right-hand thread everyone means: turn `+` about the axis, advance
        // `+` along it.
        let j_a = frame.r_a.cross(axis) - axis * lead;
        let j_b = frame.r_b.cross(axis) - axis * lead;
        let k = self.bodies[ia].generalized_mass(axis, j_a)
            + self.bodies[ib].generalized_mass(axis, j_b);
        if k <= 0.0 {
            return;
        }
        let error = (offset - rest_offset) + lead * (frame.hinge_angle - rest_angle);
        let rate = (self.bodies[ia].velocity_at(frame.r_a)
            - self.bodies[ib].velocity_at(frame.r_b))
        .dot(axis)
            + lead * (self.bodies[ib].angular - self.bodies[ia].angular).dot(axis);
        let lambda = -(rate + soft.bias_rate * error) / k * soft.mass_scale
            - joint.coupling_impulse * soft.impulse_scale;
        if let Some(x) = self.bodies.get_mut(ia) {
            x.apply_generalized(axis * lambda, j_a * lambda);
        }
        if let Some(x) = self.bodies.get_mut(ib) {
            x.apply_generalized(axis * -lambda, j_b * -lambda);
        }
        joint.coupling_impulse += lambda;
    }

    /// Integrate what the rate couplings failed to cancel.
    ///
    /// Run once per substep on the settled velocities, so the leftover is the
    /// real one rather than whatever a mid-iteration state happened to show.
    #[cfg(feature = "mechanism")]
    fn integrate_coupling_phase(&mut self, joints: &mut [&mut Joint], dt: f32) {
        for k in 0..self.frames.len() {
            let frame = self.frames[k];
            let Some(joint) = joints.get_mut(k) else {
                continue;
            };
            if !joint.is_active() || !frame.valid {
                continue;
            }
            let (ia, ib) = (frame.body_a, frame.body_b);
            if ia >= self.bodies.len() || ib >= self.bodies.len() {
                continue;
            }
            let rate = match &joint.kind {
                JointKind::Gear { ratio, .. } => {
                    let mut rate = self.bodies[ia].angular.dot(frame.axis_a)
                        - *ratio * self.bodies[ib].angular.dot(frame.axis_b);
                    if let Some(c) = frame.body_c.and_then(|i| self.bodies.get(i)) {
                        rate += c.angular.dot(frame.axis_b * *ratio - frame.axis_a);
                    }
                    rate
                }
                JointKind::RackPinion { radius, .. } => {
                    (self.bodies[ib].velocity_at(frame.r_b)
                        - self.bodies[ia].velocity_at(frame.r_a))
                    .dot(frame.axis_b)
                        - *radius
                            * (self.bodies[ia].angular - self.bodies[ib].angular)
                                .dot(frame.axis_a)
                }
                _ => continue,
            };
            if rate.is_finite() {
                joint.coupling_phase += rate * dt;
            }
        }
    }
}

/// How fast a drive is told to close its remaining error.
///
/// Capping this rather than the impulse is what makes `max_speed` behave like
/// an actuator's top speed: the drive still applies its full force, it just
/// stops asking to arrive any sooner than it could.
// Not gated on `mechanism`: a tendon servo is a drive too, and tendons are
// always compiled.
#[inline]
fn servo_bias(bias_rate: f32, error: f32, max_speed: f32) -> f32 {
    let bias = bias_rate * error;
    if max_speed > 0.0 {
        bias.clamp(-max_speed, max_speed)
    } else {
        bias
    }
}

/// One tendon reduced to a single constraint row.
///
/// A tendon constrains one scalar — its total length — so however many bodies
/// it threads, it contributes one row. `rows` holds that row's Jacobian, one
/// entry per body the cable actually pulls on.
#[derive(Debug, Clone, Default)]
struct TendonRow {
    /// `(core-local body index, linear direction, angular direction)`, summed
    /// per body so a run with both ends on one body cancels to nothing.
    rows: Vec<(usize, Vector3, Vector3)>,
    /// Path length as resolved, strand divisors and wrap arcs already in it.
    length: f32,
    /// Generalised inverse mass of the row along its own direction.
    inv_mass: f32,
    /// False for a path that could not be resolved; such a row is skipped.
    valid: bool,
}

impl TendonRow {
    /// Reduce a tendon to its Jacobian against the solver's bodies.
    ///
    /// `locals` gives the core-local index of each route point's body, in route
    /// order; `usize::MAX` for a point whose body is not in this core, which
    /// cannot happen for a well-built island and is skipped if it does.
    fn build(path: &ResolvedPath, locals: &[(usize, usize)], core: &[SolverBody]) -> Self {
        if !path.valid || path.segments.len() != locals.len() || path.is_empty() {
            return Self::default();
        }

        let mut row = Self {
            rows: Vec::new(),
            // Taken from the resolution rather than re-summed: it already
            // includes the arcs the path runs along and the strand divisors,
            // neither of which is visible in the straight runs alone.
            length: path.length,
            inv_mass: 0.0,
            valid: true,
        };
        for (segment, &(local_a, local_b)) in path.segments.iter().zip(locals) {
            let Some(u) = try_normalize(segment.b - segment.a) else {
                continue;
            };
            // `d|p₁-p₀|/dp₀ = -u` and `d/dp₁ = +u`, scaled by the strand's
            // divisor. Both go in against the body carrying that point, summed —
            // a run whose two ends are on the *same* body cancels to nothing,
            // which is correct: its length cannot change. That is also what
            // makes a wrap right, since the arc between two tangent points on
            // one obstacle contributes no direction of its own.
            let u = u * (1.0 / segment.divisor);
            row.push(core, local_a, segment.a, -u);
            row.push(core, local_b, segment.b, u);
        }
        row.inv_mass = row
            .rows
            .iter()
            .map(|(i, lin, ang)| core[*i].generalized_mass(*lin, *ang))
            .sum();
        row
    }

    fn push(&mut self, core: &[SolverBody], local: usize, point: Vector3, dir: Vector3) {
        if local >= core.len() {
            return;
        }
        let ang = (point - core[local].com).cross(dir);
        if let Some(slot) = self.rows.iter_mut().find(|(i, _, _)| *i == local) {
            slot.1 = slot.1 + dir;
            slot.2 = slot.2 + ang;
        } else {
            self.rows.push((local, dir, ang));
        }
    }
}

/// The impulse one iteration of a [`JointSpring`] asks for.
///
/// Shared by the angular and linear cases, which differ only in how `error`,
/// `rate` and `inv_mass` were measured. `inv_mass` is the constraint's
/// generalised inverse mass along its own axis.
///
/// # Backward Euler, in one line
///
/// The impulse a spring-damper wants over one substep is the one consistent
/// with the velocity **it will itself produce**, not with the velocity it found:
///
/// ```text
/// λ = −dt·( k(θ + dt·ω′) + d·ω′ )      ω′ = ω + inv_mass·λ
/// ```
///
/// Solving that for `λ` and writing it as the *increment* from whatever has
/// been applied so far — so a second pass over the same joint asks for nothing
/// — gives what this returns, with `a = dt(k·dt + d)`:
///
/// ```text
/// λ⁺ = −( dt·k·θ + a·ω + λ ) / (1 + a·inv_mass)
/// ```
///
/// A pure damper is the `k = 0` case of the same expression, so there is one
/// formula and not two.
///
/// # Why not the explicit force, and why not [`Softness`]
///
/// Explicit — `τ = −k·θ` integrated forward — is stable only while `k·dt²/I`
/// stays under about one, which a steel backbone discretised at `n·EI/L` blows
/// through by three orders of magnitude. This is stable for any `k`, any `d`
/// and any `dt`; too stiff a spring merely converges to a rigid joint.
///
/// Routing it through [`SoftParams`] instead — deriving a frequency and damping
/// ratio from the stiffness and handing them to the soft-constraint machinery —
/// gets the same static answer and was tried. It fails on exactly the case this
/// is for. A rod's stations are light, the effective inertia at one falling as
/// `1/n³`, so `ζ` reaches four thousand times critical by thirty links; the
/// soft parameters then put `impulse_scale` at a millionth, the accumulator
/// takes thousands of iterations to reach the impulse it wants, and it has to
/// be carried across substeps to get there. Carried, it pumps energy into a
/// long chain — a thirty-link rod visibly stretched to three times its length.
/// This form reaches the whole impulse on the **first** iteration of every
/// substep and keeps nothing between them.
fn spring_lambda(
    spring: JointSpring,
    error: f32,
    rate: f32,
    inv_mass: f32,
    accumulated: f32,
    dt: f32,
) -> f32 {
    let a = dt * (spring.stiffness * dt + spring.damping);
    let denominator = 1.0 + a * inv_mass;
    if denominator <= 0.0 || !denominator.is_finite() {
        return 0.0;
    }
    -(dt * spring.stiffness * error + a * rate + accumulated) / denominator
}

/// `K = diag(invMa + invMb) - [ra]ˣ Ia⁻¹ [ra]ˣ - [rb]ˣ Ib⁻¹ [rb]ˣ`
fn point_constraint_matrix(a: &SolverBody, b: &SolverBody, r_a: Vector3, r_b: Vector3) -> Mat3 {
    let linear = Mat3::from_diagonal(a.inv_mass + b.inv_mass);
    let sa = Mat3::skew(r_a);
    let sb = Mat3::skew(r_b);
    linear
        .sub(&sa.mul(&a.inv_inertia).mul(&sa))
        .sub(&sb.mul(&b.inv_inertia).mul(&sb))
}

/// World-space joint geometry, recomputed once per step.
#[derive(Debug, Clone, Copy)]
struct JointFrame {
    body_a: usize,
    body_b: usize,
    /// Anchor offsets from each body's centre of mass.
    r_a: Vector3,
    r_b: Vector3,
    anchor_a: Vector3,
    anchor_b: Vector3,
    axis_a: Vector3,
    axis_b: Vector3,
    rotation_a: Quaternion,
    rotation_b: Quaternion,
    rest_rotation: Quaternion,
    hinge_angle: f32,
    /// World-space x, y and z of the joint frame, for [`JointKind::Generic`].
    joint_axes: [Vector3; 3],
    /// Rotation of B relative to A about each joint axis, in radians.
    euler: [f32; 3],
    /// The substep length, so motor force limits can be turned into impulses
    /// without threading `dt` through every call.
    dt: f32,
    /// The body a coupling is carried on, when it is carried on one that moves.
    /// See [`crate::joint::JointKind::Gear::carrier`].
    ///
    /// Gear couplings are the only thing that sets or reads this, and they are
    /// behind the same flag — so without it the field is a word of padding and
    /// a dead-code warning.
    #[cfg(feature = "mechanism")]
    body_c: Option<usize>,
    valid: bool,
}

impl JointFrame {
    fn build(joint: &Joint, bodies: &BodySet, dt: f32) -> Self {
        let mut frame = Self {
            body_a: joint.body_a.index(),
            body_b: joint.body_b.index(),
            r_a: Vector3::ZERO,
            r_b: Vector3::ZERO,
            anchor_a: Vector3::ZERO,
            anchor_b: Vector3::ZERO,
            axis_a: Vector3::UP,
            axis_b: Vector3::UP,
            rotation_a: Quaternion::identity(),
            rotation_b: Quaternion::identity(),
            rest_rotation: Quaternion::identity(),
            hinge_angle: 0.0,
            joint_axes: [Vector3::RIGHT, Vector3::UP, Vector3::FORWARD],
            euler: [0.0; 3],
            dt,
            #[cfg(feature = "mechanism")]
            body_c: None,
            valid: false,
        };
        let (Some(a), Some(b)) = (bodies.get(joint.body_a), bodies.get(joint.body_b)) else {
            return frame;
        };
        frame.anchor_a = a.position.transform_point(joint.local_anchor_a);
        frame.anchor_b = b.position.transform_point(joint.local_anchor_b);
        frame.r_a = frame.anchor_a - a.world_center_of_mass();
        frame.r_b = frame.anchor_b - b.world_center_of_mass();
        frame.rotation_a = a.rotation();
        frame.rotation_b = b.rotation();

        match &joint.kind {
            JointKind::Revolute {
                local_axis_a,
                local_axis_b,
                ..
            }
            | JointKind::Prismatic {
                local_axis_a,
                local_axis_b,
                ..
            } => {
                frame.axis_a = a.position.transform_vector(*local_axis_a);
                frame.axis_b = b.position.transform_vector(*local_axis_b);
                // Twist angle of B relative to A about the hinge axis.
                frame.hinge_angle =
                    crate::joint::axial_angle(frame.rotation_a, frame.rotation_b, *local_axis_a);
            }
            #[cfg(feature = "mechanism")]
            JointKind::Gear {
                local_axis_a,
                local_axis_b,
                carrier,
                ..
            } => {
                frame.axis_a = a.position.transform_vector(*local_axis_a);
                frame.axis_b = b.position.transform_vector(*local_axis_b);
                frame.hinge_angle =
                    crate::joint::axial_angle(frame.rotation_a, frame.rotation_b, *local_axis_a);
                // A carrier that is not in the set is simply absent, and the
                // coupling falls back to being measured in the world — which is
                // also what a *fixed* carrier gives, since it contributes no
                // inverse inertia and no rate.
                frame.body_c = carrier
                    .filter(|c| bodies.get(*c).is_some())
                    .map(|c| c.index());
            }
            #[cfg(feature = "mechanism")]
            JointKind::RackPinion {
                local_axis_a,
                local_axis_b,
                ..
            }
            | JointKind::Screw {
                local_axis_a,
                local_axis_b,
                ..
            } => {
                frame.axis_a = a.position.transform_vector(*local_axis_a);
                frame.axis_b = b.position.transform_vector(*local_axis_b);
                frame.hinge_angle =
                    crate::joint::axial_angle(frame.rotation_a, frame.rotation_b, *local_axis_a);
            }
            JointKind::Fixed { rest_rotation } => frame.rest_rotation = *rest_rotation,
            JointKind::Generic {
                frame_a,
                frame_b,
                rest_rotation,
                ..
            } => {
                frame.rest_rotation = *rest_rotation;
                let world_a = frame.rotation_a.multiply(*frame_a);
                frame.joint_axes = [
                    Vector3::RIGHT.apply_quaternion(world_a),
                    Vector3::UP.apply_quaternion(world_a),
                    Vector3::FORWARD.apply_quaternion(world_a),
                ];
                // Angles are measured from the pose the joint was built in, so
                // "zero" is where the user put the bodies rather than wherever
                // the world axes happen to point.
                let world_b = frame.rotation_b.multiply(*frame_b);
                let relative = world_a
                    .multiply(*rest_rotation)
                    .conjugate()
                    .multiply(world_b);
                frame.euler = euler_xyz(relative);
            }
            _ => {}
        }
        frame.valid = true;
        frame
    }
}

/// Rotation split into turns about x, then y, then z.
///
/// Only used for limits on a generic joint. The decomposition is singular when
/// the y angle reaches ±90°, where x and z stop being distinguishable; the
/// clamp keeps `asin` in range so the result stays finite rather than NaN, but
/// the two outer angles are meaningless there. [`crate::joint::Joint::generic`]
/// says so.
fn euler_xyz(q: Quaternion) -> [f32; 3] {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let sin_y = (2.0 * (x * z + w * y)).clamp(-1.0, 1.0);
    if sin_y.abs() > 0.9999 {
        // At the pole, fold the lost degree of freedom into x and report z as 0.
        return [
            2.0 * x.atan2(w),
            sin_y.asin(),
            0.0,
        ];
    }
    [
        (-2.0 * (y * z - w * x)).atan2(1.0 - 2.0 * (x * x + y * y)),
        sin_y.asin(),
        (-2.0 * (x * y - w * z)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

/// Rotation vector taking B's actual orientation to the one the joint wants.
fn fixed_angular_error(frame: &JointFrame, rest: Quaternion) -> Vector3 {
    let desired = frame.rotation_a.multiply(rest);
    let mut error = frame.rotation_b.multiply(desired.conjugate());
    // A quaternion and its negation are the same rotation; pick the short way
    // round, or the joint will drive the bodies the long way to get there.
    if error.w < 0.0 {
        error = Quaternion::new(-error.x, -error.y, -error.z, -error.w);
    }
    Vector3::new(error.x, error.y, error.z) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{BodyId, RigidBody};
    use crate::contact::ManifoldKey;
    use crate::narrowphase::{collide, RawManifold};
    use std::collections::HashSet;
    use crate::shape::Shape;

    /// Minimal step loop: integrate, collide, solve, integrate. The full
    /// version lives in `World`; this keeps the solver tests independent of it.
    fn step(bodies: &mut BodySet, contacts: &mut ContactSet, joints: &mut JointSet, dt: f32) {
        let gravity = Vector3::new(0.0, -9.81, 0.0);
        let config = SolverConfig::default();
        let mut solver = Solver::new();

        for (_, b) in bodies.iter_mut() {
            let _ = b.begin_step();
            b.integrate_velocity(dt, gravity);
        }

        contacts.begin_step();
        // Mirror `World::detect_collisions`: bodies tied by a joint overlap at
        // the joint by design, so colliding them makes the contact and the
        // constraint fight. A helper that skips this filter does not simulate
        // what callers run, and the difference is not subtle — a hinge whose
        // anchor collides with its own door is shoved off its pivot every step
        // and has its spin scrubbed off by the contact's friction.
        let mut no_collide: HashSet<(u32, u32)> = HashSet::new();
        for (_, joint) in joints.iter() {
            if joint.is_active() && !joint.collide_connected {
                let (a, b) = (joint.body_a.index() as u32, joint.body_b.index() as u32);
                no_collide.insert((a.min(b), a.max(b)));
            }
        }
        let ids: Vec<BodyId> = bodies.ids().collect();
        for i in 0..ids.len() {
            for j in i + 1..ids.len() {
                let (a, b) = (ids[i], ids[j]);
                let (ia, ib) = (a.index() as u32, b.index() as u32);
                if no_collide.contains(&(ia.min(ib), ia.max(ib))) {
                    continue;
                }
                let (ba, bb) = (bodies.get(a).unwrap(), bodies.get(b).unwrap());
                if !ba.is_dynamic() && !bb.is_dynamic() {
                    continue;
                }
                for (ci, ca) in ba.colliders().iter().enumerate() {
                    for (cj, cb) in bb.colliders().iter().enumerate() {
                        let mut raw: Vec<RawManifold> = Vec::new();
                        collide(
                            &ca.shape,
                            &ca.world_transform(&ba.position),
                            &cb.shape,
                            &cb.world_transform(&bb.position),
                            0.02,
                            &mut raw,
                        );
                        for m in &raw {
                            contacts.update(
                                ManifoldKey {
                                    body_a: a,
                                    body_b: b,
                                    collider_a: ci as u32,
                                    collider_b: cj as u32,
                                    sub_a: m.sub_a as u32,
                                    sub_b: m.sub_b as u32,
                                },
                                m,
                                &ba.position,
                                &bb.position,
                                ca.material.combine(&cb.material).friction,
                                ca.material.combine(&cb.material).restitution,
                                false,
                            );
                        }
                    }
                }
            }
        }
        contacts.end_step();

        solver.solve(bodies, contacts, joints, &mut TendonSet::new(), dt, &config);

        for (_, b) in bodies.iter_mut() {
            b.integrate_position(dt);
            b.reset_forces();
        }
    }

    fn ground(bodies: &mut BodySet) -> BodyId {
        bodies.insert(RigidBody::fixed().shape(Shape::ground()).friction(0.8))
    }

    #[test]
    fn a_falling_box_comes_to_rest_on_the_ground() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        ground(&mut bodies);
        let cube = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 3.0, 0.0)),
        );

        for _ in 0..240 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }

        let y = bodies.get(cube).unwrap().translation().y;
        assert!((y - 0.5).abs() < 0.02, "box settled at y = {y}, expected 0.5");
        let v = bodies.get(cube).unwrap().linear_velocity.length();
        assert!(v < 0.05, "box is still moving at {v}");
    }

    #[test]
    fn a_resting_box_does_not_sink_over_time() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        ground(&mut bodies);
        let cube = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0))
                .can_sleep(false),
        );
        for _ in 0..600 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let y = bodies.get(cube).unwrap().translation().y;
        assert!(y > 0.48, "box sank to y = {y} over 10 seconds");
    }

    #[test]
    fn a_bouncy_ball_bounces_and_a_dead_one_does_not() {
        let run = |restitution: f32| {
            let mut bodies = BodySet::new();
            let mut contacts = ContactSet::new();
            let mut joints = JointSet::new();
            ground(&mut bodies);
            let ball = bodies.insert(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.5))
                    .translation(Vector3::new(0.0, 3.0, 0.0))
                    .restitution(restitution)
                    .can_sleep(false),
            );
            let mut peak_after_bounce: f32 = 0.0;
            let mut touched = false;
            for _ in 0..400 {
                step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
                let y = bodies.get(ball).unwrap().translation().y;
                if y < 0.55 {
                    touched = true;
                }
                if touched {
                    peak_after_bounce = peak_after_bounce.max(y);
                }
            }
            peak_after_bounce
        };
        assert!(run(0.9) > 1.2, "an elastic ball should bounce back up");
        assert!(run(0.0) < 0.6, "an inelastic ball should stay down");
    }

    #[test]
    fn friction_stops_a_sliding_box_and_frictionless_keeps_it_going() {
        let run = |friction: f32| {
            let mut bodies = BodySet::new();
            let mut contacts = ContactSet::new();
            let mut joints = JointSet::new();
            bodies.insert(RigidBody::fixed().shape(Shape::ground()).friction(friction));
            let cube = bodies.insert(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vector3::new(5.0, 0.0, 0.0))
                    .friction(friction)
                    .can_sleep(false),
            );
            for _ in 0..180 {
                step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
            }
            bodies.get(cube).unwrap().linear_velocity.x
        };
        assert!(run(1.0) < 0.2, "friction should have stopped the box");
        assert!(run(0.0) > 4.5, "a frictionless box should keep its speed");
    }

    #[test]
    fn a_stack_of_boxes_stays_stacked() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        ground(&mut bodies);
        let mut cubes = Vec::new();
        for i in 0..5 {
            cubes.push(bodies.insert(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(0.0, 0.5 + i as f32 * 1.001, 0.0))
                    .friction(0.8),
            ));
        }
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        for (i, c) in cubes.iter().enumerate() {
            let b = bodies.get(*c).unwrap();
            let want = 0.5 + i as f32;
            assert!(
                (b.translation().y - want).abs() < 0.1,
                "box {i} at y = {}, expected about {want}",
                b.translation().y
            );
            assert!(
                b.translation().x.abs() < 0.15 && b.translation().z.abs() < 0.15,
                "box {i} slid sideways to {:?}",
                b.translation()
            );
        }
    }

    #[test]
    fn a_heavy_box_does_not_push_a_light_one_through_the_floor() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        ground(&mut bodies);
        let light = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0))
                .mass(0.1),
        );
        bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 1.55, 0.0))
                .mass(500.0),
        );
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let y = bodies.get(light).unwrap().translation().y;
        assert!(y > 0.4, "the light box was crushed through the floor to y = {y}");
    }

    #[test]
    fn a_spherical_joint_holds_a_pendulum_at_its_pivot() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let bob = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.2))
                .translation(Vector3::new(2.0, 0.0, 0.0))
                .can_sleep(false),
        );
        joints.insert(Joint::spherical(
            anchor,
            bob,
            Vector3::ZERO,
            Vector3::new(-2.0, 0.0, 0.0),
        ));

        // Track the swing rather than sampling it. A pendulum released from the
        // horizontal returns to the horizontal every half period, so the height
        // at any one step says only where in the cycle that step landed — the
        // lowest point it reached is the thing that means "it swung".
        let mut lowest = f32::MAX;
        let mut radius_error: f32 = 0.0;
        for _ in 0..600 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
            let p = bodies.get(bob).unwrap().translation();
            lowest = lowest.min(p.y);
            radius_error = radius_error.max((p.length() - 2.0).abs());
        }
        // Whatever it swung to, it must have stayed 2 units from the pivot
        // throughout, not merely come back to 2 by the end.
        assert!(
            radius_error < 0.05,
            "pendulum drifted to a radius error of {radius_error}"
        );
        // Released horizontally, it should reach roughly the bottom of its arc.
        assert!(lowest < -1.5, "the pendulum only ever got down to y = {lowest}");
    }

    #[test]
    fn a_distance_joint_holds_its_length_under_gravity() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let bob = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.2))
                .translation(Vector3::new(0.0, -1.5, 0.0))
                .can_sleep(false),
        );
        joints.insert(Joint::distance(
            anchor,
            bob,
            Vector3::ZERO,
            Vector3::ZERO,
            1.5,
        ));
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let d = bodies.get(bob).unwrap().translation().length();
        assert!((d - 1.5).abs() < 0.05, "distance joint stretched to {d}");
    }

    #[test]
    fn a_rope_goes_slack_but_never_stretches() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        // Starts well inside the rope's reach.
        let bob = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.2))
                .translation(Vector3::new(0.0, -0.5, 0.0))
                .can_sleep(false),
        );
        joints.insert(Joint::rope(anchor, bob, Vector3::ZERO, Vector3::ZERO, 2.0));
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let d = bodies.get(bob).unwrap().translation().length();
        assert!(d <= 2.05, "rope stretched to {d}");
        assert!(d > 1.9, "rope should have gone taut, got {d}");
    }

    #[test]
    fn a_fixed_joint_welds_two_bodies_rigidly() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let arm = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.1, 0.1))
                .translation(Vector3::new(1.0, 0.0, 0.0))
                .can_sleep(false),
        );
        let j = Joint::fixed(
            &bodies,
            anchor,
            arm,
            Vector3::ZERO,
            Vector3::new(-1.0, 0.0, 0.0),
        );
        joints.insert(j);
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let b = bodies.get(arm).unwrap();
        assert!(
            (b.translation() - Vector3::new(1.0, 0.0, 0.0)).length() < 0.08,
            "welded body drifted to {:?}",
            b.translation()
        );
        // And it must not have rotated: a weld holds orientation too.
        assert!(
            b.rotation().dot(Quaternion::identity()).abs() > 0.99,
            "welded body rotated to {:?}",
            b.rotation()
        );
    }

    #[test]
    fn a_hinge_allows_its_own_axis_and_blocks_the_others() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let door = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.05))
                .translation(Vector3::new(0.5, 0.0, 0.0))
                .gravity_scale(0.0)
                .angular_velocity(Vector3::new(0.0, 2.0, 0.0))
                .can_sleep(false),
        );
        joints.insert(Joint::revolute(
            anchor,
            door,
            Vector3::ZERO,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::UP,
            Vector3::UP,
        ));
        for _ in 0..120 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        let w = bodies.get(door).unwrap().angular_velocity;
        // Not still 2.0, and it never could be. The anchor is fixed, so its
        // reaction acts *at* the hinge and exerts no torque about it — angular
        // momentum about the hinge is conserved, not angular velocity. A door
        // handed w = (0, 2, 0) about its own centre of mass therefore drops
        // immediately to 2*I_yy / (I_yy + m*d^2), which for this one is 0.50,
        // and that is what the first step produces.
        //
        // It then bleeds off slowly: the constraint is solved at the velocity
        // level each step, so the residual it removes after integration takes a
        // little energy with it. What the hinge must do is keep the turn on its
        // own axis and put none of it anywhere else.
        assert!(w.y.abs() > 0.1, "the hinge stopped turning about y: {w:?}");
        assert!(
            w.y.abs() < 0.51,
            "the hinge gained spin it was never given: {w:?}"
        );
        assert!(w.x.abs() < 0.01 && w.z.abs() < 0.01, "off-axis spin leaked: {w:?}");
        // The pivot must still be at the origin.
        let pivot = bodies.get(door).unwrap().position.transform_point(Vector3::new(-0.5, 0.0, 0.0));
        assert!(pivot.length() < 0.05, "hinge pivot drifted to {pivot:?}");
    }

    #[test]
    fn a_hinge_motor_drives_and_limits_stop_it() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let arm = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.05, 0.05))
                .translation(Vector3::new(0.5, 0.0, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        joints.insert(
            Joint::revolute(
                anchor,
                arm,
                Vector3::ZERO,
                Vector3::new(-0.5, 0.0, 0.0),
                Vector3::UP,
                Vector3::UP,
            )
            .with_motor(3.0, 100.0)
            .with_limits(-0.5, 0.5),
        );
        for _ in 0..300 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        // The motor drives toward +y rotation; the limit must catch it near 0.5 rad.
        let q = bodies.get(arm).unwrap().rotation();
        let angle = 2.0 * q.y.atan2(q.w);
        assert!(
            angle > 0.3 && angle < 0.75,
            "motor+limit settled at {angle} rad, expected near the 0.5 limit"
        );
    }

    #[test]
    fn a_breakable_joint_snaps_under_load() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        let anchor = bodies.insert(RigidBody::fixed().shape(Shape::ball(0.1)));
        let heavy = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, -1.0, 0.0))
                .mass(500.0)
                .can_sleep(false),
        );
        let id = joints.insert(
            Joint::distance(anchor, heavy, Vector3::ZERO, Vector3::ZERO, 1.0).breakable(1.0),
        );
        for _ in 0..60 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        assert!(joints.get(id).unwrap().broken, "the joint should have snapped");
        // And the body should now be falling freely.
        assert!(bodies.get(heavy).unwrap().translation().y < -1.2);
    }

    #[test]
    fn a_sleeping_body_is_not_moved_by_the_solver() {
        let mut bodies = BodySet::new();
        let mut contacts = ContactSet::new();
        let mut joints = JointSet::new();
        ground(&mut bodies);
        let cube = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0)),
        );
        bodies.get_mut(cube).unwrap().sleep();
        let before = bodies.get(cube).unwrap().position;
        for _ in 0..60 {
            step(&mut bodies, &mut contacts, &mut joints, 1.0 / 60.0);
        }
        assert_eq!(bodies.get(cube).unwrap().position, before);
    }
}



