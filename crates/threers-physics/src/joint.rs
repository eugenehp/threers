//! Joints — constraints that tie two bodies together.
//!
//! | Joint | Removes | Leaves |
//! |---|---|---|
//! | [`Joint::fixed`] | everything | nothing — a rigid weld |
//! | [`Joint::spherical`] | 3 translations | 3 rotations — a ball socket |
//! | [`Joint::revolute`] | 3 translations, 2 rotations | spin about one axis — a hinge |
//! | [`Joint::prismatic`] | 2 translations, 3 rotations | slide along one axis |
//! | [`Joint::distance`] | 1 translation | a rigid rod, or a rope with slack |
//! | [`Joint::spring`] | nothing (soft) | a damped spring along the anchor line |
//! | [`Joint::generic`] | whichever of the 6 you choose | the rest |
//!
//! Anchors are given in each body's **local** frame. When you know where the
//! joint should sit in the world instead, use the `*_at_point` constructors,
//! which convert for you.
//!
//! # Machine elements
//!
//! With the `mechanism` feature on, three more kinds join the table — the ones
//! a mechanism needs and a pile of loose bodies never does:
//!
//! | Joint | Removes | Leaves |
//! |---|---|---|
//! | [`Joint::gear`] | 1 rotation, *rate only* | a fixed turns ratio |
//! | [`Joint::rack_pinion`] | 1 rate | travel tied to a pinion's spin |
//! | [`Joint::screw`] | everything but a helix | turn-and-advance together |
//!
//! The same feature adds [`Servo`], which drives a hinge or slider to a
//! *position* under a torque ceiling, rather than at a speed.
//!
//! # Rigid and soft
//!
//! Every joint is rigid by default: the solver drives its error to zero as hard
//! as it can each step. [`Joint::soft`] relaxes that into a spring, specified as
//! a frequency and a damping ratio rather than a stiffness — see [`Softness`].
//!
//! # Elastic
//!
//! Softness is about the constraint the joint *holds*. The freedom it leaves
//! can have a spring in it too — [`Joint::with_spring`] gives a hinge a rest
//! angle it is pulled back toward, and [`Joint::with_damping`] gives it viscous
//! resistance with nowhere to pull. Three ways of giving, easily confused:
//!
//! | | What it does |
//! |---|---|
//! | [`Joint::soft`] | the joint's own constraint yields under load |
//! | [`Joint::spring`] | a spring between the two anchors, along the line between them |
//! | [`Joint::with_spring`] | a spring on the joint's coordinate — the elastic hinge |
//!
//! A chain of hinges each carrying a [`JointSpring`] of `n·EI/L` is a flexible
//! rod, which is what [`threers-continuum`](https://docs.rs/threers-continuum)
//! builds on.

use crate::body::{BodyId, BodySet};
use threers::math::{Quaternion, Vector3};

/// How hard a joint fights to hold its constraint.
///
/// Stiffness in newtons per metre is the wrong unit to expose: the right value
/// depends on the masses involved, so a number that works for a door is useless
/// for a crane. A *frequency* does not have that problem — "this suspension
/// oscillates at 4 Hz" means the same thing whatever it is holding up, because
/// the solver scales it by the constraint's own effective mass.
///
/// ```
/// use threers_physics::prelude::*;
///
/// # let mut world = World::new();
/// # let a = world.add_body(RigidBody::dynamic().shape(Shape::ball(1.0)));
/// # let b = world.add_body(RigidBody::dynamic().shape(Shape::ball(1.0)));
/// // A ball socket, then the same socket with some give in it.
/// let rigid = Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO);
/// let springy = rigid.clone().soft(6.0, 0.7);
/// assert!(rigid.softness.is_rigid());
/// assert!(!springy.softness.is_rigid());
/// ```
///
/// Damping ratio reads the usual way: below 1 overshoots and rings, 1 is
/// critically damped, above 1 crawls back without overshooting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Softness {
    /// Oscillations per second. Zero — the default — means rigid.
    ///
    /// Keep it well under half the step rate. A 30 Hz spring on a 60 Hz step has
    /// two samples per cycle and will alias into noise rather than a spring.
    pub frequency: f32,
    /// Fraction of critical damping.
    pub damping_ratio: f32,
}

impl Softness {
    /// No give at all — the solver's default behaviour.
    pub const RIGID: Self = Self {
        frequency: 0.0,
        damping_ratio: 0.0,
    };

    pub fn new(frequency: f32, damping_ratio: f32) -> Self {
        Self {
            frequency: frequency.max(0.0),
            damping_ratio: damping_ratio.max(0.0),
        }
    }

    /// Whether this asks for the rigid solve.
    pub fn is_rigid(&self) -> bool {
        self.frequency <= 0.0
    }
}

impl Default for Softness {
    fn default() -> Self {
        Self::RIGID
    }
}

/// What one axis of a [`Joint::generic`] is allowed to do.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DofMotion {
    /// Held at the rest value.
    #[default]
    Locked,
    /// Unconstrained.
    Free,
    /// Free within a range, blocked outside it.
    Limited(JointLimits),
}

/// One degree of freedom of a [`Joint::generic`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dof {
    pub motion: DofMotion,
    /// Drive along or about this axis. A motor on a `Locked` axis does nothing —
    /// the lock wins — so free or limit the axis you want to drive.
    pub motor: Option<Motor>,
    /// Pull this axis back to a rest value like a spring rather than holding it.
    pub spring: Option<JointSpring>,
}

impl Dof {
    /// Held rigid.
    pub const LOCKED: Self = Self {
        motion: DofMotion::Locked,
        motor: None,
        spring: None,
    };
    /// Unconstrained.
    pub const FREE: Self = Self {
        motion: DofMotion::Free,
        motor: None,
        spring: None,
    };

    /// Free between `min` and `max`.
    pub fn limited(min: f32, max: f32) -> Self {
        Self {
            motion: DofMotion::Limited(JointLimits::new(min, max)),
            motor: None,
            spring: None,
        }
    }

    /// Add a drive.
    pub fn driven(mut self, target_velocity: f32, max_force: f32) -> Self {
        self.motor = Some(Motor::new(target_velocity, max_force));
        self
    }

    /// Add a spring pulling this axis back toward `rest`.
    ///
    /// A chain of these is a flexible rod: give each station `n·EI/L` and the
    /// links bend like a beam rather than hanging like a chain.
    pub fn sprung(mut self, stiffness: f32, rest: f32, damping: f32) -> Self {
        self.spring = Some(JointSpring::new(stiffness, rest, damping));
        self
    }
}

/// A powered axis on a hinge or slider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Motor {
    /// Angular velocity (rad/s) for a hinge, linear (units/s) for a slider.
    pub target_velocity: f32,
    /// Cap on the impulse the motor may apply per second. Keep it finite, or
    /// the motor will happily fling the whole assembly.
    pub max_force: f32,
}

impl Motor {
    pub fn new(target_velocity: f32, max_force: f32) -> Self {
        Self {
            target_velocity,
            max_force: max_force.max(0.0),
        }
    }
}

/// A spring on the joint's own coordinate: an angle for a hinge, a distance for
/// a slider.
///
/// This is the elastic hinge — a return spring, a torsion bar, a rubber bush,
/// or one station of a flexible rod discretised into rigid links. It differs
/// from [`Joint::spring`], which pulls two *anchors* together along the line
/// between them, and from [`Joint::soft`], which does not pull anywhere: soft
/// says how hard the joint fights to hold the constraint it already has, and
/// this says the joint has somewhere it would rather be.
///
/// ```
/// use threers_physics::prelude::*;
///
/// # let mut world = World::new();
/// # let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(1.0, 1.0, 0.1)));
/// # let flap = world.add_body(RigidBody::dynamic().shape(Shape::cuboid(1.0, 1.0, 0.1)));
/// // A flap that springs shut: 4 N·m/rad back toward zero, lightly damped.
/// let hinge = Joint::revolute(frame, flap, Vector3::ZERO, Vector3::ZERO, Vector3::UP, Vector3::UP)
///     .with_spring(4.0, 0.0, 0.1);
/// # let _ = hinge;
/// ```
///
/// # Why stiffness is safe to name here
///
/// [`Softness`] deliberately refuses to take a stiffness, because the number
/// that works for a door is useless for a crane — the right value depends on
/// the mass. A spring is the case where that reasoning inverts: `4 N·m/rad` is
/// a property of the spring, not of what it happens to be holding, and a rod's
/// `EI/L` comes out of beam theory with nothing to say about mass at all.
///
/// It is still solved implicitly rather than as an explicit force. The solver
/// divides the stiffness by the constraint's own effective mass to recover the
/// frequency [`Softness`] wanted all along, so a stiffness far too large to
/// integrate explicitly — a steel backbone at `n·EI/L` — neither explodes nor
/// forces the timestep down. That is the whole reason this is a constraint and
/// not a call to [`crate::body::RigidBody::apply_torque_impulse`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointSpring {
    /// Torque per radian (hinge) or force per unit of travel (slider). Zero
    /// leaves a pure damper.
    pub stiffness: f32,
    /// The coordinate it pulls toward, measured the way
    /// [`crate::world::World::joint_angle`] reports it.
    pub rest: f32,
    /// Torque per rad/s, or force per unit/s. Opposes motion in the joint
    /// whether or not the spring is doing anything, so it is also how you damp
    /// a free hinge: `stiffness = 0`, `damping > 0`.
    pub damping: f32,
}

impl JointSpring {
    /// Pull toward `rest` at `stiffness`, damped by `damping`.
    pub fn new(stiffness: f32, rest: f32, damping: f32) -> Self {
        Self {
            stiffness: stiffness.max(0.0),
            rest,
            damping: damping.max(0.0),
        }
    }

    /// Damping alone — a dashpot across the joint, with nowhere it pulls to.
    pub fn damper(damping: f32) -> Self {
        Self::new(0.0, 0.0, damping)
    }

    /// Whether this does anything at all.
    pub fn is_active(&self) -> bool {
        self.stiffness > 0.0 || self.damping > 0.0
    }
}

/// A position drive: a hinge or slider told *where* to be rather than how fast
/// to move.
///
/// A [`Motor`] is an open loop — it asks for a speed and has no idea where the
/// joint ended up. Anything that wants an angle out of it has to close the loop
/// itself, which in practice means re-deriving a PID controller per project.
/// This is that loop, solved as a constraint instead of as a controller, so it
/// gets the solver's substepping and warm starting for free.
///
/// ```
/// # #[cfg(feature = "mechanism")] {
/// use threers_physics::prelude::*;
///
/// # let mut world = World::new();
/// # let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(1.0, 1.0, 0.1)));
/// # let door = world.add_body(RigidBody::dynamic().shape(Shape::cuboid(1.0, 1.0, 0.1)));
/// // A door driven to 90°, by a motor that can hold 12 N·m and no more.
/// let hinge = Joint::revolute(frame, door, Vector3::ZERO, Vector3::ZERO, Vector3::UP, Vector3::UP)
///     .with_servo(Servo::new(std::f32::consts::FRAC_PI_2, 12.0));
/// # let _ = hinge;
/// # }
/// ```
///
/// # Why the force ceiling is not optional
///
/// `max_force` is what keeps a driven joint honest. Every other constraint in
/// the world — a contact, a limit — is rigid, so a drive with no ceiling wins
/// every argument and pushes the part it is driving straight through whatever
/// is in the way. Bounded, it loses those arguments exactly as the real
/// mechanism would: the door reaches the doorstop and stops, the lid meets the
/// box and stops, and the drive sits there stalled against it.
///
/// That is the whole trick behind motion that cannot violate the physics. Pick
/// the number off the motor you intend to fit; if you do not have one,
/// [`crate::assembly::Assembly`] sizes it from the part it has to lift.
#[cfg(feature = "mechanism")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Servo {
    /// Where to hold the joint: radians for a hinge, world units for a slider.
    ///
    /// Measured the same way [`crate::world::World::joint_angle`] reports it, which for
    /// a bare joint is *from where the two bodies' own frames coincide* rather
    /// than from the pose you built it in.
    pub target: f32,
    /// How stiffly it chases the target. Rigid — the default — is a servo with
    /// no give at all, which is right for a machine tool and wrong for a hobby
    /// servo; give it a frequency and damping ratio for something springier.
    pub softness: Softness,
    /// Torque (hinge) or force (slider) ceiling. See the type docs: this is what
    /// lets contacts and limits beat the drive.
    pub max_force: f32,
    /// Fastest the drive will slew, in rad/s or units/s. Zero means unlimited,
    /// which lets a joint far from its target lunge at it; a real actuator has
    /// a top speed and so should this.
    pub max_speed: f32,
}

#[cfg(feature = "mechanism")]
impl Servo {
    /// Hold `target`, with at most `max_force` of torque or force to do it.
    pub fn new(target: f32, max_force: f32) -> Self {
        Self {
            target,
            softness: Softness::RIGID,
            max_force: max_force.max(0.0),
            max_speed: 0.0,
        }
    }

    /// Cap the slew rate, in rad/s or units/s.
    pub fn max_speed(mut self, max_speed: f32) -> Self {
        self.max_speed = max_speed.max(0.0);
        self
    }

    /// Give the drive some spring rather than holding the target rigidly — see
    /// [`Softness`]. A 3 Hz, 0.7-damped servo visibly settles into position; the
    /// rigid default arrives and stays.
    pub fn response(mut self, frequency: f32, damping_ratio: f32) -> Self {
        self.softness = Softness::new(frequency, damping_ratio);
        self
    }

    /// Move the target, keeping the tuning.
    pub fn to(mut self, target: f32) -> Self {
        self.target = target;
        self
    }
}

/// Inclusive travel range for a hinge (radians) or slider (world units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointLimits {
    pub min: f32,
    pub max: f32,
}

impl JointLimits {
    pub fn new(min: f32, max: f32) -> Self {
        Self {
            min: min.min(max),
            max: max.max(min),
        }
    }
}

/// What a joint constrains.
///
/// The variants differ a lot in size — `Generic` carries six `Dof`s where
/// `Spherical` carries nothing — and boxing the big one would shrink every
/// `Joint`. Left flat deliberately: this enum is public and matched on
/// everywhere, so the indirection would show up in every caller's patterns to
/// save memory on a struct that is already one per joint rather than one per
/// contact.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum JointKind {
    /// Rigid weld. `rest_rotation` is the relative orientation to hold, captured
    /// when the joint is built.
    Fixed { rest_rotation: Quaternion },
    /// Ball socket: the anchors coincide, rotation is free.
    Spherical,
    /// Hinge about `local_axis_a` / `local_axis_b`.
    Revolute {
        local_axis_a: Vector3,
        local_axis_b: Vector3,
        limits: Option<JointLimits>,
        motor: Option<Motor>,
        /// Elastic hinge: torque back toward a rest angle. See [`JointSpring`].
        spring: Option<JointSpring>,
    },
    /// Slider along `local_axis_a` / `local_axis_b`.
    Prismatic {
        local_axis_a: Vector3,
        local_axis_b: Vector3,
        limits: Option<JointLimits>,
        motor: Option<Motor>,
        /// Sprung slide: force back toward a rest offset. See [`JointSpring`].
        spring: Option<JointSpring>,
    },
    /// Anchor separation held within `[min, max]`. Equal bounds give a rigid
    /// rod; `min = 0` gives a rope that only resists stretching.
    Distance { min: f32, max: f32 },
    /// Soft spring along the anchor line. Never rigid, so it cannot fight other
    /// constraints for control of the body.
    Spring {
        rest_length: f32,
        /// Force per unit of extension.
        stiffness: f32,
        /// Force per unit of closing speed. Zero oscillates forever.
        damping: f32,
    },
    /// Two spinning parts geared together.
    ///
    /// Constrains the *rates* only — `ratio` turns of A for every turn of B —
    /// and nothing else, so both parts still need whatever hinge holds them in
    /// place. That is what a gear is: an extra relationship between two axes
    /// that already exist.
    ///
    /// A positive ratio turns them the same way about their own axes, which is
    /// a belt, a chain, or an internal gear. Two external gears meshing turn
    /// opposite ways, so those take a negative ratio.
    ///
    /// Rates are measured in the world, not against a carrier, so a gearbox
    /// whose *case* is also spinning — an epicyclic train — is not this. Mount
    /// the pair on something that holds still and it is exact.
    #[cfg(feature = "mechanism")]
    Gear {
        local_axis_a: Vector3,
        local_axis_b: Vector3,
        /// Turns of A per turn of B. Negative for meshing external gears.
        ratio: f32,
        /// The body the pair is mounted on, when that body moves too.
        ///
        /// `None` measures both rates against the world, which is right for a
        /// gearbox bolted to something that holds still and wrong the moment the
        /// case itself turns: an epicyclic train, a slew drive on a rotating
        /// carrier, a nozzle bearing driven by a pinion that rides on the
        /// segment upstream of it. Name the carrier and the ratio is enforced on
        /// the rates *relative to it*, which is the pair of numbers a gear
        /// actually relates.
        ///
        /// The carrier has to be jointed to at least one of the pair, which in
        /// a real train it always is — the pinion is in a bearing in the case.
        /// That is what puts all three in one solver island.
        carrier: Option<BodyId>,
    },
    /// A pinion driving a rack: rotation of A about its axis becomes
    /// translation of B along its own.
    ///
    /// `radius` is the pitch radius, so B advances `radius` units for every
    /// radian A turns. Like [`JointKind::Gear`] this is a rate relationship and
    /// holds nothing in place on its own — the pinion needs a hinge and the
    /// rack needs a slider.
    #[cfg(feature = "mechanism")]
    RackPinion {
        /// Pinion spin axis, in A's frame.
        local_axis_a: Vector3,
        /// Rack travel direction, in B's frame.
        local_axis_b: Vector3,
        /// Pitch radius: units of travel per radian.
        radius: f32,
    },
    /// A threaded pair — a lead screw, a jackscrew, a bolt going into a hole.
    ///
    /// One degree of freedom, not two: turning and sliding are locked to each
    /// other by the thread, so B advances `lead` units along the axis for every
    /// radian it turns relative to A. Unlike the gear this is a joint in its own
    /// right, holding the parts on a common axis as well as coupling them.
    ///
    /// `lead` is per **radian**, which is the number the constraint actually
    /// wants; a thread quoted as pitch-per-turn is that over 2π, and
    /// [`Joint::screw_from_pitch`] does the division for you.
    #[cfg(feature = "mechanism")]
    Screw {
        local_axis_a: Vector3,
        local_axis_b: Vector3,
        /// Axial travel per radian of relative rotation. Negative is a
        /// left-hand thread.
        lead: f32,
        /// Where the pair stood when the joint was built — the coupling holds
        /// travel and rotation to each other *from here*, so the pose you
        /// assemble in is the one that reads as zero.
        rest_offset: f32,
        rest_angle: f32,
        /// Travel limits along the axis, in world units. A bolt bottoming out.
        limits: Option<JointLimits>,
        /// Drives the *rotation*, since that is the end a screw is turned from.
        motor: Option<Motor>,
    },
    /// Every degree of freedom configured independently.
    ///
    /// The other kinds are the useful presets; this is the general case behind
    /// them. All six axes are measured in the joint frame — body A's frame turned
    /// by `frame_a` — with `linear` in world units and `angular` in radians.
    Generic {
        /// Joint frame in each body's local space. For a joint built by
        /// [`Joint::generic`] these are identity, meaning the bodies' own axes.
        frame_a: Quaternion,
        frame_b: Quaternion,
        /// Relative orientation the angular locks hold, captured at construction.
        rest_rotation: Quaternion,
        /// Translation along the joint x, y and z axes.
        linear: [Dof; 3],
        /// Rotation about the joint x, y and z axes.
        angular: [Dof; 3],
    },
}

/// A constraint between two bodies.
#[derive(Debug, Clone, PartialEq)]
pub struct Joint {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub local_anchor_a: Vector3,
    pub local_anchor_b: Vector3,
    pub kind: JointKind,
    pub enabled: bool,
    /// Impulse magnitude that snaps the joint. `None` never breaks.
    pub break_impulse: Option<f32>,
    /// Set once the joint has broken. A broken joint stops constraining but is
    /// not removed, so you can inspect or repair it.
    pub broken: bool,
    /// Whether the two bodies still collide with each other. Off by default:
    /// jointed bodies usually overlap by construction, and colliding them
    /// fights the constraint.
    pub collide_connected: bool,
    /// Give in the constraint itself. [`Softness::RIGID`] is the default solve.
    pub softness: Softness,
    /// A position drive on the joint's own axis, if it has one.
    #[cfg(feature = "mechanism")]
    pub servo: Option<Servo>,

    // Accumulated impulses, kept across steps for warm starting.
    pub(crate) linear_impulse: Vector3,
    pub(crate) angular_impulse: Vector3,
    pub(crate) axial_impulse: f32,
    pub(crate) limit_impulse: f32,
    pub(crate) motor_impulse: f32,
    /// Per-axis constraint impulses for [`JointKind::Generic`] — three linear
    /// slots then three angular.
    pub(crate) dof_impulse: [f32; 6],
    /// Per-axis motor impulses, same slot order as `dof_impulse`.
    pub(crate) dof_motor_impulse: [f32; 6],
    /// Per-axis spring impulses, same slot order as `dof_impulse`.
    pub(crate) spring_impulse: [f32; 6],
    pub(crate) servo_impulse: f32,
    pub(crate) coupling_impulse: f32,
    /// Accumulated coupling error for a gear, rack or screw mate — how far the
    /// two driven axes have drifted out of their declared ratio.
    pub(crate) coupling_phase: f32,
}

impl Joint {
    fn base(body_a: BodyId, body_b: BodyId, anchor_a: Vector3, anchor_b: Vector3, kind: JointKind) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            kind,
            enabled: true,
            break_impulse: None,
            broken: false,
            collide_connected: false,
            softness: Softness::RIGID,
            #[cfg(feature = "mechanism")]
            servo: None,
            linear_impulse: Vector3::ZERO,
            angular_impulse: Vector3::ZERO,
            axial_impulse: 0.0,
            limit_impulse: 0.0,
            motor_impulse: 0.0,
            dof_impulse: [0.0; 6],
            dof_motor_impulse: [0.0; 6],
            spring_impulse: [0.0; 6],
            servo_impulse: 0.0,
            coupling_impulse: 0.0,
            coupling_phase: 0.0,
        }
    }

    /// Ball socket at the given local anchors.
    pub fn spherical(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
    ) -> Self {
        Self::base(body_a, body_b, anchor_a, anchor_b, JointKind::Spherical)
    }

    /// Rigid weld holding the bodies' current relative pose.
    pub fn fixed(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
    ) -> Self {
        let rest = match (bodies.get(body_a), bodies.get(body_b)) {
            (Some(a), Some(b)) => a.rotation().conjugate().multiply(b.rotation()),
            _ => Quaternion::identity(),
        };
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Fixed {
                rest_rotation: rest,
            },
        )
    }

    /// Hinge. `axis_a` and `axis_b` are the rotation axis in each body's frame;
    /// they are normalised for you.
    pub fn revolute(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        axis_a: Vector3,
        axis_b: Vector3,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Revolute {
                local_axis_a: normalize_or_up(axis_a),
                local_axis_b: normalize_or_up(axis_b),
                limits: None,
                motor: None,
                spring: None,
            },
        )
    }

    /// Slider along a shared axis.
    pub fn prismatic(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        axis_a: Vector3,
        axis_b: Vector3,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Prismatic {
                local_axis_a: normalize_or_up(axis_a),
                local_axis_b: normalize_or_up(axis_b),
                limits: None,
                motor: None,
                spring: None,
            },
        )
    }

    /// Rigid rod of the given length.
    pub fn distance(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        length: f32,
    ) -> Self {
        let l = length.max(0.0);
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Distance { min: l, max: l },
        )
    }

    /// Rope: resists stretching past `max`, goes slack below it.
    pub fn rope(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        max_length: f32,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Distance {
                min: 0.0,
                max: max_length.max(0.0),
            },
        )
    }

    /// Damped spring between the anchors.
    pub fn spring(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Spring {
                rest_length: rest_length.max(0.0),
                stiffness: stiffness.max(0.0),
                damping: damping.max(0.0),
            },
        )
    }

    /// Gear two spinning parts together — see [`JointKind::Gear`].
    ///
    /// Anchors are irrelevant to a rate constraint, so there are none: this
    /// couples the axes wherever the parts happen to be.
    ///
    /// ```
    /// # #[cfg(feature = "mechanism")] {
    /// use threers_physics::prelude::*;
    ///
    /// # let mut world = World::new();
    /// # let pinion = world.add_body(RigidBody::dynamic().shape(Shape::cylinder(0.1, 0.2)));
    /// # let wheel = world.add_body(RigidBody::dynamic().shape(Shape::cylinder(0.1, 0.6)));
    /// // A 3:1 reduction between two meshing spur gears, so they counter-rotate.
    /// world.add_joint(Joint::gear(pinion, wheel, Vector3::UP, Vector3::UP, -3.0));
    /// # }
    /// ```
    #[cfg(feature = "mechanism")]
    pub fn gear(
        body_a: BodyId,
        body_b: BodyId,
        axis_a: Vector3,
        axis_b: Vector3,
        ratio: f32,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            Vector3::ZERO,
            Vector3::ZERO,
            JointKind::Gear {
                local_axis_a: normalize_or_up(axis_a),
                local_axis_b: normalize_or_up(axis_b),
                ratio,
                carrier: None,
            },
        )
    }

    /// Gear two spinning parts that are both carried by a third — an epicyclic
    /// train, or any drive whose case is not standing still.
    ///
    /// [`Self::gear`] relates the two rates in the world. This relates them to
    /// the carrier: `(ω_a − ω_c)·â = ratio · (ω_b − ω_c)·b̂`, which is what a
    /// mesh does. The two agree exactly when the carrier is fixed, so this is
    /// the general form and `gear` is the shortcut.
    ///
    /// ```
    /// # #[cfg(feature = "mechanism")] {
    /// use threers_physics::prelude::*;
    ///
    /// # let mut world = World::new();
    /// # let arm = world.add_body(RigidBody::dynamic().shape(Shape::cuboid(0.4, 0.05, 0.05)));
    /// # let sun = world.add_body(RigidBody::dynamic().shape(Shape::cylinder(0.05, 0.1)));
    /// # let planet = world.add_body(RigidBody::dynamic().shape(Shape::cylinder(0.05, 0.05)));
    /// // A planet meshing with a sun, both carried on a rotating arm. Turn the
    /// // arm with the sun held and the planet still turns 2:1 against it.
    /// world.add_joint(Joint::gear_on_carrier(
    ///     planet, sun, arm, Vector3::UP, Vector3::UP, -2.0,
    /// ));
    /// # }
    /// ```
    #[cfg(feature = "mechanism")]
    pub fn gear_on_carrier(
        body_a: BodyId,
        body_b: BodyId,
        carrier: BodyId,
        axis_a: Vector3,
        axis_b: Vector3,
        ratio: f32,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            Vector3::ZERO,
            Vector3::ZERO,
            JointKind::Gear {
                local_axis_a: normalize_or_up(axis_a),
                local_axis_b: normalize_or_up(axis_b),
                ratio,
                carrier: Some(carrier),
            },
        )
    }

    /// Turn a pinion's rotation into a rack's travel — see
    /// [`JointKind::RackPinion`]. `radius` is the pitch radius.
    ///
    /// The anchors matter here, unlike a gear: the rack's travel is measured
    /// between them, so put them on the pitch line.
    #[cfg(feature = "mechanism")]
    pub fn rack_pinion(
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        pinion_axis: Vector3,
        rack_axis: Vector3,
        radius: f32,
    ) -> Self {
        Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::RackPinion {
                local_axis_a: normalize_or_up(pinion_axis),
                local_axis_b: normalize_or_up(rack_axis),
                radius,
            },
        )
    }

    /// A threaded pair — see [`JointKind::Screw`]. `lead` is axial travel per
    /// radian.
    ///
    /// Zero is wherever the parts stand now, so build it with the thread
    /// engaged where you want it to read zero.
    #[cfg(feature = "mechanism")]
    #[allow(clippy::too_many_arguments)]
    pub fn screw(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        axis_a: Vector3,
        axis_b: Vector3,
        lead: f32,
    ) -> Option<Self> {
        let a = bodies.get(body_a)?;
        let b = bodies.get(body_b)?;
        let local_axis_a = normalize_or_up(axis_a);
        let local_axis_b = normalize_or_up(axis_b);
        let world_axis = a.position.transform_vector(local_axis_a);
        let delta =
            a.position.transform_point(anchor_a) - b.position.transform_point(anchor_b);
        Some(Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Screw {
                local_axis_a,
                local_axis_b,
                lead,
                rest_offset: delta.dot(world_axis),
                rest_angle: axial_angle(a.rotation(), b.rotation(), local_axis_a),
                limits: None,
                motor: None,
            },
        ))
    }

    /// The same, from a thread quoted the way threads are quoted: travel per
    /// full turn. An M8×1.25 gives `pitch = 1.25` in millimetre units.
    #[cfg(feature = "mechanism")]
    #[allow(clippy::too_many_arguments)]
    pub fn screw_from_pitch(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        axis_a: Vector3,
        axis_b: Vector3,
        pitch: f32,
    ) -> Option<Self> {
        Self::screw(
            bodies,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            pitch / std::f32::consts::TAU,
        )
    }

    /// Every degree of freedom configured by hand.
    ///
    /// The axes are body A's own, in the order x, y, z. Angular values are
    /// measured from the bodies' relative orientation at construction, so
    /// whatever pose they are in when you build the joint is the rest pose.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// let base = world.add_body(RigidBody::fixed().shape(Shape::cuboid(1.0, 0.2, 1.0)));
    /// let head = world.add_body(
    ///     RigidBody::dynamic()
    ///         .shape(Shape::cuboid(0.3, 0.3, 0.3))
    ///         .translation(Vector3::new(0.0, 1.0, 0.0)),
    /// );
    ///
    /// // A turret: spins freely about y, tilts ±30° about x, rigid otherwise.
    /// world.add_joint(
    ///     Joint::generic(world.bodies(), base, head, Vector3::new(0.0, 0.2, 0.0), Vector3::new(0.0, -0.3, 0.0))
    ///         .unwrap()
    ///         .with_angular_dof(0, Dof::limited(-0.52, 0.52))
    ///         .with_angular_dof(1, Dof::FREE.driven(2.0, 40.0)),
    /// );
    /// ```
    ///
    /// Angular limits are read off Euler angles about the joint axes, which lose
    /// meaning once the middle axis (y) passes ±90°. Lock or limit that axis if
    /// the joint needs to work through large rotations — a hinge with a limit is
    /// [`Joint::revolute`], which measures its angle directly and has no such
    /// restriction.
    pub fn generic(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
    ) -> Option<Self> {
        let a = bodies.get(body_a)?;
        let b = bodies.get(body_b)?;
        Some(Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Generic {
                frame_a: Quaternion::identity(),
                frame_b: Quaternion::identity(),
                rest_rotation: a.rotation().conjugate().multiply(b.rotation()),
                linear: [Dof::LOCKED; 3],
                angular: [Dof::LOCKED; 3],
            },
        ))
    }

    /// The same, with the six axes measured in a frame of your choosing rather
    /// than in body A's own.
    ///
    /// [`Joint::generic`] measures its degrees of freedom along the bodies' x, y
    /// and z, which is only what you want when the thing being modelled happens
    /// to lie along one of them. `frame_a` and `frame_b` rotate the joint's axes
    /// within each body, so a slider up an arbitrary diagonal is
    /// `shortest_arc(X, diagonal)` and then DOF 0.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    /// use threers_physics::ik::shortest_arc;
    ///
    /// # let mut world = World::new();
    /// # let base = world.add_body(RigidBody::fixed().shape(Shape::ball(1.0)));
    /// # let shaft = world.add_body(RigidBody::dynamic().shape(Shape::ball(1.0)));
    /// // A plain bearing on the y axis: free to turn *and* slide, nothing else.
    /// let axis = Vector3::new(0.0, 1.0, 0.0);
    /// let frame = shortest_arc(Vector3::new(1.0, 0.0, 0.0), axis);
    /// world.add_joint(
    ///     Joint::generic_framed(world.bodies(), base, shaft, Vector3::ZERO, Vector3::ZERO, frame, frame)
    ///         .unwrap()
    ///         .with_linear_dof(0, Dof::FREE)
    ///         .with_angular_dof(0, Dof::FREE),
    /// );
    /// ```
    pub fn generic_framed(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vector3,
        anchor_b: Vector3,
        frame_a: Quaternion,
        frame_b: Quaternion,
    ) -> Option<Self> {
        let a = bodies.get(body_a)?;
        let b = bodies.get(body_b)?;
        let (frame_a, frame_b) = (frame_a.normalize(), frame_b.normalize());
        // Rest is between the *joint frames*, not the bodies — which is the
        // same quaternion whenever the frames are identity, and so leaves
        // `generic` behaving exactly as it did.
        let world_a = a.rotation().multiply(frame_a);
        let world_b = b.rotation().multiply(frame_b);
        Some(Self::base(
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            JointKind::Generic {
                frame_a,
                frame_b,
                rest_rotation: world_a.conjugate().multiply(world_b),
                linear: [Dof::LOCKED; 3],
                angular: [Dof::LOCKED; 3],
            },
        ))
    }

    /// Configure translation along joint axis `axis` (0 = x, 1 = y, 2 = z).
    /// Ignored by every kind but [`JointKind::Generic`].
    pub fn with_linear_dof(mut self, axis: usize, dof: Dof) -> Self {
        if let JointKind::Generic { linear, .. } = &mut self.kind {
            if let Some(slot) = linear.get_mut(axis) {
                *slot = dof;
            }
        }
        self
    }

    /// Configure rotation about joint axis `axis` (0 = x, 1 = y, 2 = z).
    /// Ignored by every kind but [`JointKind::Generic`].
    pub fn with_angular_dof(mut self, axis: usize, dof: Dof) -> Self {
        if let JointKind::Generic { angular, .. } = &mut self.kind {
            if let Some(slot) = angular.get_mut(axis) {
                *slot = dof;
            }
        }
        self
    }

    /// Ball socket pinned at a world-space point, converting the anchors for you.
    ///
    /// ```no_run
    /// # use threers_physics::prelude::*;
    /// # let mut world = World::new();
    /// # let arm = world.add_body(RigidBody::dynamic().shape(Shape::ball(1.0)));
    /// # let hand = world.add_body(RigidBody::dynamic().shape(Shape::ball(1.0)));
    /// let elbow = Vector3::new(0.0, 1.5, 0.0);
    /// world.add_joint(Joint::spherical_at_point(world.bodies(), arm, hand, elbow).unwrap());
    /// ```
    pub fn spherical_at_point(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        world_point: Vector3,
    ) -> Option<Self> {
        let (aa, ab) = world_anchors(bodies, body_a, body_b, world_point)?;
        Some(Self::spherical(body_a, body_b, aa, ab))
    }

    /// Hinge pinned at a world point, about a world axis.
    pub fn revolute_at_point(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        world_point: Vector3,
        world_axis: Vector3,
    ) -> Option<Self> {
        let (aa, ab) = world_anchors(bodies, body_a, body_b, world_point)?;
        let a = bodies.get(body_a)?;
        let b = bodies.get(body_b)?;
        Some(Self::revolute(
            body_a,
            body_b,
            aa,
            ab,
            a.position.inverse_transform_vector(world_axis),
            b.position.inverse_transform_vector(world_axis),
        ))
    }

    /// Weld two bodies where they currently stand, pinned at a world point.
    pub fn fixed_at_point(
        bodies: &BodySet,
        body_a: BodyId,
        body_b: BodyId,
        world_point: Vector3,
    ) -> Option<Self> {
        let (aa, ab) = world_anchors(bodies, body_a, body_b, world_point)?;
        Some(Self::fixed(bodies, body_a, body_b, aa, ab))
    }

    // ---- modifiers --------------------------------------------------------

    /// Restrict a hinge's angle or a slider's travel. Ignored by other kinds.
    pub fn with_limits(mut self, min: f32, max: f32) -> Self {
        let l = Some(JointLimits::new(min, max));
        match &mut self.kind {
            JointKind::Revolute { limits, .. } | JointKind::Prismatic { limits, .. } => *limits = l,
            #[cfg(feature = "mechanism")]
            JointKind::Screw { limits, .. } => *limits = l,
            _ => {}
        }
        self
    }

    /// Drive a hinge or slider. Ignored by other kinds.
    pub fn with_motor(mut self, target_velocity: f32, max_force: f32) -> Self {
        let m = Some(Motor::new(target_velocity, max_force));
        match &mut self.kind {
            JointKind::Revolute { motor, .. } | JointKind::Prismatic { motor, .. } => *motor = m,
            #[cfg(feature = "mechanism")]
            JointKind::Screw { motor, .. } => *motor = m,
            _ => {}
        }
        self
    }

    /// Make a hinge or slider elastic — see [`JointSpring`]. Ignored by other
    /// kinds.
    ///
    /// `rest` is in the same coordinate the joint's limits and servo use, so a
    /// spring cannot disagree with them about where the joint is.
    pub fn with_spring(mut self, stiffness: f32, rest: f32, damping: f32) -> Self {
        let s = Some(JointSpring::new(stiffness, rest, damping));
        match &mut self.kind {
            JointKind::Revolute { spring, .. } | JointKind::Prismatic { spring, .. } => *spring = s,
            _ => {}
        }
        self
    }

    /// Damp a hinge or slider without pulling it anywhere — viscous friction in
    /// the joint's own coordinate.
    ///
    /// `Bearing` is the other kind of resistance and the two are not
    /// interchangeable: a bearing opposes *load* and can hold a joint still, a
    /// damper opposes *speed* and can only slow it down.
    pub fn with_damping(self, damping: f32) -> Self {
        self.with_spring(0.0, 0.0, damping)
    }

    /// Drive the joint to a position rather than at a speed — see [`Servo`].
    ///
    /// Hinges, sliders and screws. The target is in the same coordinate the
    /// joint's own limits use, so a servo and a limit can never disagree about
    /// where the joint is.
    #[cfg(feature = "mechanism")]
    pub fn with_servo(mut self, servo: Servo) -> Self {
        if self.axial_coordinate_kind().is_some() {
            self.servo = Some(servo);
        }
        self
    }

    /// Which one-dimensional coordinate this joint has, if any. `None` for the
    /// kinds a servo cannot drive.
    #[cfg(feature = "mechanism")]
    pub fn axial_coordinate_kind(&self) -> Option<AxialKind> {
        match &self.kind {
            JointKind::Revolute { .. } => Some(AxialKind::Angle),
            JointKind::Prismatic { .. } => Some(AxialKind::Offset),
            JointKind::Screw { .. } => Some(AxialKind::Offset),
            _ => None,
        }
    }

    /// Give the joint some spring instead of holding it rigid.
    ///
    /// `frequency` is in hertz and `damping_ratio` is a fraction of critical —
    /// see [`Softness`]. This softens the joint's *main* constraint (the point
    /// weld, or the angular weld on a fixed joint); limits and motors stay
    /// rigid, since a soft limit is a limit that does not hold.
    ///
    /// [`Joint::spring`] is a different thing: an explicit force along one axis
    /// that constrains nothing. Reach for `soft` when you want a joint that
    /// mostly holds but gives under load, and for `spring` when you want a
    /// suspension strut.
    pub fn soft(mut self, frequency: f32, damping_ratio: f32) -> Self {
        self.softness = Softness::new(frequency, damping_ratio);
        self
    }

    /// Snap the joint once its impulse exceeds `impulse`.
    pub fn breakable(mut self, impulse: f32) -> Self {
        self.break_impulse = Some(impulse.max(0.0));
        self
    }

    /// Whether the joint is currently doing anything.
    pub fn is_active(&self) -> bool {
        self.enabled && !self.broken
    }

    /// Repair a broken joint and clear its accumulated impulses.
    pub fn repair(&mut self) {
        self.broken = false;
        self.reset_impulses();
    }

    /// Clear the accumulators the solver rebuilds from nothing each substep,
    /// leaving the two it warm-starts from.
    ///
    /// The distinction matters more than it looks. `linear_impulse` and
    /// `angular_impulse` are re-applied at the top of every substep, so they
    /// have to keep meaning *the total this constraint has applied* — zeroing
    /// them turns them into "the correction added since the warm start", which
    /// is a different number and lands at zero whenever the joint is already
    /// converged. A [`Softness`] reads that total back to decide how much to
    /// yield, so getting it wrong halves the give; and a rigid joint ends up
    /// warm-starting from the real impulse one substep and from nothing the
    /// next.
    ///
    /// The rest are not warm-started, so for them the accumulator *is* the
    /// total, and clearing is right.
    pub(crate) fn reset_step_impulses(&mut self) {
        self.axial_impulse = 0.0;
        self.limit_impulse = 0.0;
        self.motor_impulse = 0.0;
        self.dof_impulse = [0.0; 6];
        self.dof_motor_impulse = [0.0; 6];
        // The phase of a coupling is deliberately *not* cleared: it is the
        // integrated drift, not an impulse, and zeroing it every substep would
        // throw away the whole point of tracking it.
        #[cfg(feature = "mechanism")]
        {
            self.servo_impulse = 0.0;
            self.coupling_impulse = 0.0;
        }
    }

    pub(crate) fn reset_impulses(&mut self) {
        self.linear_impulse = Vector3::ZERO;
        self.angular_impulse = Vector3::ZERO;
        self.axial_impulse = 0.0;
        self.limit_impulse = 0.0;
        self.motor_impulse = 0.0;
        self.dof_impulse = [0.0; 6];
        self.dof_motor_impulse = [0.0; 6];
        #[cfg(feature = "mechanism")]
        {
            self.servo_impulse = 0.0;
            self.coupling_impulse = 0.0;
            self.coupling_phase = 0.0;
        }
    }

    /// The impulses carried across steps for warm starting, for [`crate::snapshot`].
    pub fn warm_start_state(&self) -> (Vector3, Vector3, f32) {
        (
            self.linear_impulse,
            self.angular_impulse,
            self.axial_impulse,
        )
    }

    /// Put a captured warm-start state back.
    ///
    /// Restoring these is what makes a rollback resume rather than restart: the
    /// step after a restore then begins from the same guess the original one did.
    pub fn set_warm_start_state(&mut self, state: (Vector3, Vector3, f32)) {
        self.linear_impulse = state.0;
        self.angular_impulse = state.1;
        self.axial_impulse = state.2;
    }

    /// Total impulse the joint applied last step.
    pub fn applied_impulse(&self) -> f32 {
        #[cfg(feature = "mechanism")]
        let coupling = self.coupling_impulse * self.coupling_impulse;
        #[cfg(not(feature = "mechanism"))]
        let coupling = 0.0;
        (self.linear_impulse.length_sq()
            + self.angular_impulse.length_sq()
            + self.axial_impulse * self.axial_impulse
            + coupling
            + self.dof_impulse.iter().map(|i| i * i).sum::<f32>())
        .sqrt()
    }

    /// Where this joint stands, given the two bodies it connects.
    ///
    /// [`crate::world::World::joint_state`] is the way to reach this; it looks the
    /// bodies up for you.
    #[cfg(feature = "mechanism")]
    pub fn state(&self, a: &crate::body::RigidBody, b: &crate::body::RigidBody) -> Option<JointState> {
        let kind = self.axial_coordinate_kind()?;
        let local_axis_a = match &self.kind {
            JointKind::Revolute { local_axis_a, .. }
            | JointKind::Prismatic { local_axis_a, .. }
            | JointKind::Screw { local_axis_a, .. } => *local_axis_a,
            _ => return None,
        };
        let axis = a.position.transform_vector(local_axis_a);
        Some(match kind {
            AxialKind::Angle => JointState {
                kind,
                coordinate: axial_angle(a.rotation(), b.rotation(), local_axis_a),
                speed: (b.angular_velocity - a.angular_velocity).dot(axis),
            },
            AxialKind::Offset => {
                let anchor_a = a.position.transform_point(self.local_anchor_a);
                let anchor_b = b.position.transform_point(self.local_anchor_b);
                JointState {
                    kind,
                    coordinate: (anchor_a - anchor_b).dot(axis),
                    speed: (a.velocity_at_point(anchor_a) - b.velocity_at_point(anchor_b)).dot(axis),
                }
            }
        })
    }

    /// How far a rate coupling has drifted out of phase, in radians of A.
    ///
    /// Zero for every other kind. Worth watching on a long gear train: a phase
    /// that grows without settling means the coupling is losing the argument to
    /// something else, usually a motor with more torque than the gear can pass.
    #[cfg(feature = "mechanism")]
    pub fn coupling_phase(&self) -> f32 {
        self.coupling_phase
    }
}

fn normalize_or_up(v: Vector3) -> Vector3 {
    crate::math::try_normalize(v).unwrap_or(Vector3::UP)
}

/// What a one-dimensional joint coordinate measures.
#[cfg(feature = "mechanism")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxialKind {
    /// Radians, about the joint axis.
    Angle,
    /// World units, along the joint axis.
    Offset,
}

/// Where a one-degree-of-freedom joint currently stands.
///
/// # Which way is positive
///
/// A hinge reads **B relative to A**: positive is B turning about the axis by
/// the right-hand rule. A slider reads the other way round — **A relative to
/// B** — because that is the sense its limits have always been measured in, and
/// a readout that disagreed with the limits beside it would be worse than one
/// that reads backwards. [`crate::assembly`] normalises both to B-relative-to-A
/// before showing them to anyone.
#[cfg(feature = "mechanism")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointState {
    pub kind: AxialKind,
    /// Radians for a hinge, world units for a slider or screw.
    pub coordinate: f32,
    /// Rate of change of `coordinate`, in rad/s or units/s.
    pub speed: f32,
}

/// Twist of `rotation_b` relative to `rotation_a` about an axis given in A's
/// own frame, in radians.
///
/// The solver measures a hinge this way, so limits, servos and
/// [`crate::world::World::joint_angle`] all read the same number — which matters more
/// than it sounds. A readout derived independently drifts from the one the
/// limits use by a hair, and then a joint clamped at its maximum reports an
/// angle just past it, which reads as a bug in the limit.
///
/// Zero is where the two bodies' frames coincide, *not* where the joint was
/// built. A hinge assembled with its parts at some angle to each other reports
/// that angle at rest; subtract it if you want the assembled pose to read zero.
///
/// # It wraps, and so does everything built on it
///
/// The angle comes out of a quaternion, and a quaternion cannot tell a turn
/// from a turn plus a revolution. Past ±360° this wraps, which makes limits and
/// [`Servo`] targets beyond one revolution meaningless: a hinge told to reach
/// 720° is told to reach 0°, and drives itself back to where it started.
///
/// This is a limit on *counting* turns, not on making them. A joint spins as
/// freely as the solver allows; it is only the readout that is modular. Anything
/// that turns continuously — a shaft, a gear train, a wheel — wants a [`Motor`],
/// which asks for a speed and never asks where the joint is.
pub fn axial_angle(rotation_a: Quaternion, rotation_b: Quaternion, local_axis_a: Vector3) -> f32 {
    let relative = rotation_a.conjugate().multiply(rotation_b);
    let along = Vector3::new(relative.x, relative.y, relative.z).dot(local_axis_a);
    2.0 * along.atan2(relative.w)
}

fn world_anchors(
    bodies: &BodySet,
    a: BodyId,
    b: BodyId,
    world_point: Vector3,
) -> Option<(Vector3, Vector3)> {
    let ba = bodies.get(a)?;
    let bb = bodies.get(b)?;
    Some((
        ba.position.inverse_transform_point(world_point),
        bb.position.inverse_transform_point(world_point),
    ))
}

/// Stable handle to a joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointId {
    index: u32,
    generation: u32,
}

#[derive(Debug, Clone)]
struct Slot {
    generation: u32,
    joint: Option<Joint>,
}

/// Generational arena of joints, mirroring [`BodySet`].
#[derive(Debug, Clone, Default)]
pub struct JointSet {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
}

impl JointSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, joint: Joint) -> JointId {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.joint = Some(joint);
            return JointId {
                index,
                generation: slot.generation,
            };
        }
        self.slots.push(Slot {
            generation: 0,
            joint: Some(joint),
        });
        JointId {
            index: self.slots.len() as u32 - 1,
            generation: 0,
        }
    }

    pub fn remove(&mut self, id: JointId) -> Option<Joint> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let j = slot.joint.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
        self.len -= 1;
        Some(j)
    }

    pub fn get(&self, id: JointId) -> Option<&Joint> {
        let slot = self.slots.get(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.joint.as_ref())?
    }

    pub fn get_mut(&mut self, id: JointId) -> Option<&mut Joint> {
        let slot = self.slots.get_mut(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.joint.as_mut())?
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

    pub fn iter(&self) -> impl Iterator<Item = (JointId, &Joint)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.joint.as_ref().map(|j| {
                (
                    JointId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    j,
                )
            })
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (JointId, &mut Joint)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            let generation = s.generation;
            s.joint.as_mut().map(|j| {
                (
                    JointId {
                        index: i as u32,
                        generation,
                    },
                    j,
                )
            })
        })
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut Joint> {
        self.slots.iter_mut().filter_map(|s| s.joint.as_mut())
    }

    /// Drop every joint attached to `body`.
    pub fn remove_body(&mut self, body: BodyId) {
        for i in 0..self.slots.len() {
            let hit = self.slots[i]
                .joint
                .as_ref()
                .is_some_and(|j| j.body_a == body || j.body_b == body);
            if hit {
                self.slots[i].joint = None;
                self.slots[i].generation = self.slots[i].generation.wrapping_add(1);
                self.free.push(i as u32);
                self.len -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::shape::Shape;

    fn two_bodies() -> (BodySet, BodyId, BodyId) {
        let mut set = BodySet::new();
        let a = set.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(1.0))
                .translation(Vector3::new(-1.0, 0.0, 0.0)),
        );
        let b = set.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(1.0))
                .translation(Vector3::new(1.0, 0.0, 0.0)),
        );
        (set, a, b)
    }

    #[test]
    fn a_world_anchor_converts_into_both_local_frames() {
        let (set, a, b) = two_bodies();
        let j = Joint::spherical_at_point(&set, a, b, Vector3::ZERO).unwrap();
        assert!((j.local_anchor_a - Vector3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((j.local_anchor_b - Vector3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn limits_and_motors_only_attach_to_axial_joints() {
        let (set, a, b) = two_bodies();
        let hinge = Joint::revolute(a, b, Vector3::ZERO, Vector3::ZERO, Vector3::UP, Vector3::UP)
            .with_limits(-1.0, 1.0)
            .with_motor(2.0, 50.0);
        match &hinge.kind {
            JointKind::Revolute { limits, motor, .. } => {
                assert_eq!(limits.unwrap(), JointLimits::new(-1.0, 1.0));
                assert_eq!(motor.unwrap().target_velocity, 2.0);
            }
            other => panic!("wrong kind: {other:?}"),
        }

        // A ball socket has no axis to limit, so the call is a no-op, not an error.
        let ball = Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO).with_limits(-1.0, 1.0);
        assert_eq!(ball.kind, JointKind::Spherical);
        let _ = set;
    }

    #[test]
    fn reversed_limits_are_normalised() {
        let l = JointLimits::new(2.0, -2.0);
        assert_eq!((l.min, l.max), (-2.0, 2.0));
    }

    #[test]
    fn a_fixed_joint_captures_the_current_relative_rotation() {
        let mut set = BodySet::new();
        let a = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        let b = set.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(1.0))
                .rotation(Quaternion::from_axis_angle(Vector3::UP, 1.0)),
        );
        let j = Joint::fixed(&set, a, b, Vector3::ZERO, Vector3::ZERO);
        match j.kind {
            JointKind::Fixed { rest_rotation } => {
                let expected = Quaternion::from_axis_angle(Vector3::UP, 1.0);
                assert!((rest_rotation.dot(expected).abs() - 1.0).abs() < 1e-4);
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn joint_handles_are_generational() {
        let (_, a, b) = two_bodies();
        let mut set = JointSet::new();
        let id = set.insert(Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO));
        assert!(set.get(id).is_some());
        set.remove(id);
        let id2 = set.insert(Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO));
        assert!(set.get(id).is_none(), "stale handle resolved");
        assert!(set.get(id2).is_some());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn removing_a_body_removes_its_joints() {
        let (_, a, b) = two_bodies();
        let mut set = JointSet::new();
        set.insert(Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO));
        set.insert(Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO));
        assert_eq!(set.len(), 2);
        set.remove_body(b);
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn a_broken_joint_stops_being_active_until_repaired() {
        let (_, a, b) = two_bodies();
        let mut j = Joint::spherical(a, b, Vector3::ZERO, Vector3::ZERO).breakable(10.0);
        assert!(j.is_active());
        j.broken = true;
        assert!(!j.is_active());
        j.repair();
        assert!(j.is_active());
        assert_eq!(j.applied_impulse(), 0.0);
    }

    #[test]
    fn axes_are_normalised_and_degenerate_axes_fall_back() {
        let (_, a, b) = two_bodies();
        let j = Joint::revolute(
            a,
            b,
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::ZERO,
        );
        match j.kind {
            JointKind::Revolute {
                local_axis_a,
                local_axis_b,
                ..
            } => {
                assert!((local_axis_a.length() - 1.0).abs() < 1e-5);
                assert_eq!(local_axis_b, Vector3::UP, "zero axis should fall back");
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }
}
