//! Mechanical assembly: parts, mates, and motion that cannot cheat.
//!
//! Building a hinge out of raw joints means typing the pivot's world
//! coordinates and its axis by hand, twice, and hoping the pin you modelled is
//! actually where you said it was. Nothing checks it. The parts start wherever
//! you left them and the first step of the simulation snaps them into place,
//! which looks like the mechanism kicking itself.
//!
//! This module is the layer above that. You declare *where the joint lives on
//! each part*, in the part's own coordinates — the coordinates the model was
//! drawn in — and the assembly does the rest:
//!
//! | Stage | What it does |
//! |---|---|
//! | [`Assembly::solve`] | moves the parts until the mates line up, before anything simulates |
//! | [`Assembly::check`] | reports parts that overlap, mates that could not be met, and the assembly's degrees of freedom |
//! | [`Assembly::sweep`] | drives one mate through its travel and reports where it fouls |
//! | [`Assembly::build`] | turns parts into bodies and mates into joints, in one consistent pose |
//! | [`Assembly::drive`] | authors motion as joint *targets*, realised by the solver |
//!
//! ```no_run
//! use threers_physics::prelude::*;
//! use threers_physics::assembly::deg;
//! # use threers::core::BufferGeometry;
//! # fn box_geometry() -> BufferGeometry { unimplemented!() }
//! # fn lid_geometry() -> BufferGeometry { unimplemented!() }
//!
//! let mut asm = Assembly::new();
//! let body = asm.part_from_geometry("box", box_geometry(), PartPhysics::fixed());
//! let lid = asm.part_from_geometry("lid", lid_geometry(), PartPhysics::dynamic().density(0.7));
//!
//! // The hinge is declared where it sits on each part, not in world space.
//! let hinge = asm.mate(
//!     Mate::hinge(
//!         Feature::axis(lid, Axis::x_at([0.0, 40.0, 8.0])),
//!         Feature::axis(body, Axis::x_at([0.0, 40.0, 8.0])),
//!     )
//!     .limits(0.0, deg(105.0)),
//! );
//!
//! asm.solve().unwrap();               // parts snap together
//! assert!(asm.check().clear());       // nothing fouls, every mate met
//!
//! let mut world = World::new();
//! asm.build(&mut world).unwrap();
//! asm.drive(hinge).to(deg(105.0)).over(1.2);
//!
//! for _ in 0..120 {
//!     asm.update(1.0 / 60.0, &mut world);
//!     world.step(1.0 / 60.0);
//! }
//! ```
//!
//! # Why the motion cannot cheat
//!
//! An authored animation normally *sets* a transform, which makes the pose a
//! function of time and nothing else — so a lid told to open 105° opens 105°
//! whether or not something is in the way, and a hinge told to go past its stop
//! goes past it.
//!
//! [`Assembly::drive`] never sets a pose. It moves a [`Servo`]'s target, and the
//! servo is a constraint with a torque ceiling like any other. The joint limit
//! is rigid and the drive is not, so the limit wins. Contacts are solved after
//! joints, so they win too. What you get out is the pose the mechanism can
//! actually reach: the lid opens until it hits its stop, or until it hits
//! whatever is on top of it, and then it stalls there. The animation lags rather
//! than lying.
//!
//! That is also why [`Drive::torque`] exists. The ceiling is the whole mechanism
//! of compliance — an infinite one would push through anything — so if you do
//! not pick a number, the assembly sizes it from the mass it has to lift.
//!
//! # Coordinates
//!
//! Everything user-facing reads **A relative to B**: the first feature you name
//! is the part that moves, the second is what it moves against. A hinge mated
//! `(lid, box)` reports the lid's angle, opening positive by the right-hand
//! rule about the axis. Zero is the pose you assembled in, not wherever the two
//! bodies' frames happen to coincide — which is what [`crate::world::World::joint_angle`]
//! reports for a bare joint, and rarely what a CAD model means by zero.

use crate::body::{BodyId, BodyType, RigidBodyBuilder};
use crate::collider::Collider;
use crate::joint::{Joint, JointId, JointKind, JointLimits, Motor, Servo, Softness};
use crate::material::PhysicsMaterial;
use crate::math::{try_normalize, Isometry};
use crate::shape::{ColliderFit, MassProperties, Shape};
use crate::world::World;
use threers::core::{BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId};
use threers::materials::Material;
use threers::math::{Quaternion, Vector3};

/// Degrees to radians, for models drawn in degrees.
///
/// `deg(105.0)` beside a `0.0` reads as a hinge range; `1.8325` beside a `0.0`
/// reads as nothing at all.
pub fn deg(degrees: f32) -> f32 {
    degrees.to_radians()
}

/// Anything that names a point or a direction.
///
/// A CAD model is written in `[x, y, z]` arrays, and converting each one at the
/// call site buries the number you care about in punctuation.
pub trait IntoVec3 {
    fn into_vec3(self) -> Vector3;
}

impl IntoVec3 for Vector3 {
    fn into_vec3(self) -> Vector3 {
        self
    }
}

impl IntoVec3 for [f32; 3] {
    fn into_vec3(self) -> Vector3 {
        Vector3::new(self[0], self[1], self[2])
    }
}

impl IntoVec3 for (f32, f32, f32) {
    fn into_vec3(self) -> Vector3 {
        Vector3::new(self.0, self.1, self.2)
    }
}

// ---- features ------------------------------------------------------------

/// A line in a part's own coordinates: where an axle goes.
///
/// Hinges, sliders and screws are all declared with one of these on each part.
/// The assembly's job is to make the two lines the same line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    pub origin: Vector3,
    /// Normalised on construction.
    pub direction: Vector3,
}

impl Axis {
    pub fn new(origin: impl IntoVec3, direction: impl IntoVec3) -> Self {
        Self {
            origin: origin.into_vec3(),
            direction: try_normalize(direction.into_vec3()).unwrap_or(Vector3::UP),
        }
    }

    /// Along +X through `origin`.
    pub fn x_at(origin: impl IntoVec3) -> Self {
        Self::new(origin, [1.0, 0.0, 0.0])
    }

    /// Along +Y through `origin`.
    pub fn y_at(origin: impl IntoVec3) -> Self {
        Self::new(origin, [0.0, 1.0, 0.0])
    }

    /// Along +Z through `origin`.
    pub fn z_at(origin: impl IntoVec3) -> Self {
        Self::new(origin, [0.0, 0.0, 1.0])
    }

    /// Through two points — the way a hole is usually dimensioned.
    pub fn through(from: impl IntoVec3, to: impl IntoVec3) -> Self {
        let from = from.into_vec3();
        Self::new(from, to.into_vec3() - from)
    }

    /// The same line in world space, given the part's placement.
    pub fn to_world(self, transform: &Isometry) -> Self {
        Self {
            origin: transform.transform_point(self.origin),
            direction: transform.transform_vector(self.direction),
        }
    }
}

/// What a mate attaches to, and on which part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feature {
    pub part: PartId,
    pub kind: FeatureKind,
}

/// The geometry a mate is declared against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureKind {
    /// A line: a bore, a shaft, a slot's direction.
    Axis(Axis),
    /// A single point: a ball socket's centre.
    Point(Vector3),
    /// A face: an origin and the direction it looks along.
    Plane { origin: Vector3, normal: Vector3 },
}

impl Feature {
    pub fn axis(part: PartId, axis: Axis) -> Self {
        Self {
            part,
            kind: FeatureKind::Axis(axis),
        }
    }

    pub fn point(part: PartId, point: impl IntoVec3) -> Self {
        Self {
            part,
            kind: FeatureKind::Point(point.into_vec3()),
        }
    }

    pub fn plane(part: PartId, origin: impl IntoVec3, normal: impl IntoVec3) -> Self {
        Self {
            part,
            kind: FeatureKind::Plane {
                origin: origin.into_vec3(),
                normal: try_normalize(normal.into_vec3()).unwrap_or(Vector3::UP),
            },
        }
    }

    /// Where the feature sits, in the part's own frame.
    pub fn origin(&self) -> Vector3 {
        match self.kind {
            FeatureKind::Axis(a) => a.origin,
            FeatureKind::Point(p) => p,
            FeatureKind::Plane { origin, .. } => origin,
        }
    }

    /// The feature's direction in the part's own frame: an axis's line, a
    /// plane's normal, or nothing for a point.
    pub fn direction(&self) -> Option<Vector3> {
        match self.kind {
            FeatureKind::Axis(a) => Some(a.direction),
            FeatureKind::Point(_) => None,
            FeatureKind::Plane { normal, .. } => Some(normal),
        }
    }
}

// ---- mates ---------------------------------------------------------------

/// What a mate holds, and what it leaves free.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MateKind {
    /// One rotation about the shared axis. A door, a lid, an elbow.
    Hinge,
    /// One translation along the shared axis. A drawer, a linear stage.
    Slider,
    /// Both: turn and slide on the same axis. A shaft in a plain bearing.
    Cylindrical,
    /// A thread: turning and sliding, locked to each other. `lead` is axial
    /// travel per radian — see [`Mate::screw_from_pitch`].
    Screw { lead: f32 },
    /// Three rotations about a shared point. A ball socket.
    Ball,
    /// Nothing. A weld, a bolted flange, two parts that are one part.
    Rigid,
    /// Two translations in the shared plane and one rotation in it. A part
    /// resting on a face.
    Planar,
    /// A ratio between two axes that are already held by something else. Gears,
    /// a belt, a chain — see [`crate::joint::JointKind::Gear`] for the sign.
    Gear { ratio: f32 },
    /// A pinion's rotation becoming a rack's travel. `radius` is the pitch
    /// radius.
    Rack { radius: f32 },
}

impl MateKind {
    /// How many degrees of freedom this leaves between the two parts, for the
    /// mobility count.
    fn freedoms(self) -> i32 {
        match self {
            MateKind::Hinge | MateKind::Slider | MateKind::Screw { .. } => 1,
            MateKind::Cylindrical => 2,
            MateKind::Ball => 3,
            MateKind::Rigid => 0,
            MateKind::Planar => 3,
            // A coupling removes one freedom from a pair that some other mate
            // already holds, so it contributes as a 5-freedom joint would.
            MateKind::Gear { .. } | MateKind::Rack { .. } => 5,
        }
    }

    /// Whether this mate places its parts. Couplings do not — they constrain
    /// rates between parts held by other mates.
    fn places(self) -> bool {
        !matches!(self, MateKind::Gear { .. } | MateKind::Rack { .. })
    }

    /// Whether [`Mate::limits`] means anything for this kind.
    ///
    /// A ball socket has three rotations and a gear has no travel at all, so a
    /// range written on one is a mistake rather than a preference — and
    /// [`Assembly::check`] says so rather than dropping it.
    fn accepts_limits(self) -> bool {
        matches!(
            self,
            MateKind::Hinge | MateKind::Slider | MateKind::Screw { .. } | MateKind::Cylindrical
        )
    }

    /// What the mate's coordinate measures, if it has one.
    ///
    /// `None` for the kinds that have no single number describing where they
    /// stand — a ball socket has three rotations, a gear has a rate and no
    /// position at all.
    pub fn axial(self) -> Option<crate::joint::AxialKind> {
        use crate::joint::AxialKind;
        match self {
            MateKind::Hinge => Some(AxialKind::Angle),
            MateKind::Slider | MateKind::Screw { .. } => Some(AxialKind::Offset),
            _ => None,
        }
    }
}

/// A relationship between two parts.
///
/// The first feature names the part that moves; the second names what it moves
/// against. Everything reported afterwards — angles, travel, drive targets — is
/// in that sense.
#[derive(Debug, Clone)]
pub struct Mate {
    pub name: String,
    pub a: Feature,
    pub b: Feature,
    pub kind: MateKind,
    /// Travel range, in the mate's own coordinate. Radians or world units.
    pub limits: Option<JointLimits>,
    pub motor: Option<Motor>,
    pub servo: Option<Servo>,
    pub softness: Softness,
    pub break_impulse: Option<f32>,
    /// Whether the two parts also collide with each other. Off by default:
    /// mated parts overlap at the joint by design.
    pub collide: bool,
    /// What a coupling is mounted on, when it is not mounted on the world.
    ///
    /// See [`crate::joint::Joint::gear_on_carrier`]. Ignored on every mate that
    /// is not a coupling, since those relate two parts and nothing else.
    pub carrier: Option<PartId>,
}

impl Mate {
    fn base(a: Feature, b: Feature, kind: MateKind) -> Self {
        Self {
            name: String::new(),
            a,
            b,
            kind,
            limits: None,
            motor: None,
            servo: None,
            softness: Softness::RIGID,
            break_impulse: None,
            collide: false,
            carrier: None,
        }
    }

    /// A hinge on the shared axis.
    pub fn hinge(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Hinge)
    }

    /// A slider on the shared axis.
    pub fn slider(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Slider)
    }

    /// Free to turn *and* slide on the shared axis.
    pub fn cylindrical(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Cylindrical)
    }

    /// A thread, given its lead in travel per radian.
    pub fn screw(a: Feature, b: Feature, lead: f32) -> Self {
        Self::base(a, b, MateKind::Screw { lead })
    }

    /// A thread, given its pitch the way threads are quoted: travel per turn.
    pub fn screw_from_pitch(a: Feature, b: Feature, pitch: f32) -> Self {
        Self::screw(a, b, pitch / std::f32::consts::TAU)
    }

    /// A ball socket on the shared point.
    pub fn ball(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Ball)
    }

    /// Welded: two parts that move as one.
    pub fn rigid(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Rigid)
    }

    /// Face to face, free to slide and spin in the plane.
    pub fn planar(a: Feature, b: Feature) -> Self {
        Self::base(a, b, MateKind::Planar)
    }

    /// A gear ratio between two axes that other mates already hold. Negative
    /// for meshing external gears, which counter-rotate.
    pub fn gear(a: Feature, b: Feature, ratio: f32) -> Self {
        Self::base(a, b, MateKind::Gear { ratio })
    }

    /// A pinion driving a rack, given the pitch radius.
    pub fn rack(a: Feature, b: Feature, radius: f32) -> Self {
        Self::base(a, b, MateKind::Rack { radius })
    }

    /// Restrict the travel, in the mate's own coordinate.
    pub fn limits(mut self, min: f32, max: f32) -> Self {
        self.limits = Some(JointLimits::new(min, max));
        self
    }

    /// Drive at a speed, with a force ceiling. See [`Motor`].
    pub fn motor(mut self, target_velocity: f32, max_force: f32) -> Self {
        self.motor = Some(Motor::new(target_velocity, max_force));
        self
    }

    /// Drive to a position, with a force ceiling. See [`Servo`].
    ///
    /// [`Assembly::drive`] installs one of these for you, sized from the part,
    /// if the mate has none.
    pub fn servo(mut self, servo: Servo) -> Self {
        self.servo = Some(servo);
        self
    }

    /// Give the mate some spring instead of holding it rigidly.
    pub fn soft(mut self, frequency: f32, damping_ratio: f32) -> Self {
        self.softness = Softness::new(frequency, damping_ratio);
        self
    }

    /// Let the mate fail above an impulse — a shear pin, a snap fit.
    pub fn breakable(mut self, impulse: f32) -> Self {
        self.break_impulse = Some(impulse.max(0.0));
        self
    }

    /// Let the two mated parts collide with each other as well.
    /// Mount a coupling on a part that moves, so the ratio is enforced against
    /// it rather than against the world.
    ///
    /// The carrier has to be jointed to at least one of the pair — in a real
    /// train it is, because the pinion runs in a bearing in the case.
    pub fn carried_by(mut self, carrier: PartId) -> Self {
        self.carrier = Some(carrier);
        self
    }

    pub fn collide(mut self, collide: bool) -> Self {
        self.collide = collide;
        self
    }

    /// Name it, for reports.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

// ---- parts ---------------------------------------------------------------

/// Handle to a part in an [`Assembly`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartId(pub(crate) u32);

impl PartId {
    /// Where the part sits in [`Assembly::parts`], which is the order it was
    /// added in.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Handle to a mate in an [`Assembly`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MateId(pub(crate) u32);

impl MateId {
    /// Where the mate sits in [`Assembly::mates`].
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Everything the simulation needs to know about a part that the model does not
/// say.
#[derive(Debug, Clone, PartialEq)]
pub struct PartPhysics {
    pub body_type: BodyType,
    /// How to turn the drawn mesh into a collider.
    ///
    /// This is also what [`Assembly::check`] tests interference against, which
    /// makes it two decisions at once: a [`ColliderFit::ConvexHull`] on a part
    /// with a pocket in it reports the pocket as solid, so a mating part that
    /// sits in the pocket reads as fouling when it does not. Turn on
    /// [`Self::decompose`] for a moving part whose real shape matters.
    pub fit: ColliderFit,
    /// Split a concave part into convex pieces, so it can both move *and* keep
    /// its shape. Costs a voxel decomposition per part, once.
    pub decompose: bool,
    pub density: f32,
    /// Exact total mass, overriding density.
    pub mass: Option<f32>,
    pub material: PhysicsMaterial,
}

impl Default for PartPhysics {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            fit: ColliderFit::ConvexHull,
            decompose: false,
            density: 1.0,
            mass: None,
            material: PhysicsMaterial::default(),
        }
    }
}

impl PartPhysics {
    /// A part the mechanism moves.
    pub fn dynamic() -> Self {
        Self::default()
    }

    /// A part that holds still and grounds whatever is mated to it. Defaults to
    /// an exact triangle-mesh collider, which is free for something that never
    /// moves.
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
            ..Default::default()
        }
    }

    pub fn fit(mut self, fit: ColliderFit) -> Self {
        self.fit = fit;
        self
    }

    /// Keep a concave part concave, by splitting it into convex pieces.
    pub fn decomposed(mut self) -> Self {
        self.decompose = true;
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
}

/// One modelled part: what to draw, what to collide with, and where it sits.
#[derive(Debug, Clone)]
pub struct Part {
    pub name: String,
    /// The drawn mesh, when the part was built from one. Hand it to a [`Mesh`].
    pub geometry: Option<BufferGeometry>,
    pub colliders: Vec<Collider>,
    pub physics: PartPhysics,
    /// Where the part sits. Set by [`Assembly::place`], then corrected by
    /// [`Assembly::solve`].
    pub transform: Isometry,
    /// The scene node, once [`Assembly::spawn`] has made one.
    pub scene_object: Option<ObjectId>,
}

impl Part {
    /// Whether this part grounds the mechanism.
    pub fn is_grounded(&self) -> bool {
        self.physics.body_type == BodyType::Fixed
    }

    /// Mass and centre of mass in the part's own frame, from its colliders.
    fn mass_properties(&self) -> MassProperties {
        let mut total: Option<MassProperties> = None;
        for c in &self.colliders {
            let mp = c.mass_properties();
            total = Some(match total {
                Some(t) => t.merged(&mp),
                None => mp,
            });
        }
        let mut mp = total.unwrap_or(MassProperties {
            mass: 0.0,
            center_of_mass: Vector3::ZERO,
            inertia: crate::math::Mat3::ZERO,
        });
        if let Some(m) = self.physics.mass {
            mp = mp.with_mass(m);
        }
        mp
    }
}

// ---- reports -------------------------------------------------------------

/// Two parts that occupy the same space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interference {
    pub a: PartId,
    pub b: PartId,
    /// Where they overlap, in world space.
    pub point: Vector3,
    /// How deep, in world units.
    pub depth: f32,
}

/// What [`Assembly::check`] found.
#[derive(Debug, Clone, Default)]
pub struct AssemblyReport {
    /// Parts occupying the same space, worst first.
    pub interferences: Vec<Interference>,
    /// Mates the pose solve could not satisfy, with the error left over.
    pub unsatisfied: Vec<(MateId, f32)>,
    /// Pairs that could not be tested at all.
    ///
    /// Two triangle-mesh colliders have no volume between them to measure — a
    /// mesh is a surface — so the pair is reported here rather than silently
    /// passing. Give one of the two a convex fit or
    /// [`PartPhysics::decomposed`] and it becomes testable.
    pub unchecked: Vec<(PartId, PartId)>,
    /// Mates given a travel range their kind cannot use.
    ///
    /// A ball socket has three rotations and a gear has no travel, so a range
    /// on one is a mistake. Reported rather than dropped, because a limit that
    /// is quietly ignored is indistinguishable from a limit that is not doing
    /// its job.
    pub ignored_limits: Vec<MateId>,
    /// Degrees of freedom the mechanism has, by the Grübler–Kutzbach count.
    ///
    /// 1 is a mechanism with one input — a door, a slider-crank. 0 is a
    /// structure. Negative means over-constrained on paper, which is extremely
    /// common in real assemblies (two hinges on one door) and usually fine: the
    /// count assumes the constraints are independent, and coaxial ones are not.
    /// Positive above 1 means something is free that you may not have meant to
    /// leave free.
    pub mobility: i32,
    /// No part is fixed, so the whole assembly is free to drift.
    pub ungrounded: bool,
}

impl AssemblyReport {
    /// Nothing fouls, every mate is satisfied, and every pair was testable.
    ///
    /// An untestable pair counts against a clean bill deliberately: a check that
    /// could not run is not a check that passed.
    pub fn clear(&self) -> bool {
        self.interferences.is_empty()
            && self.unsatisfied.is_empty()
            && self.unchecked.is_empty()
            && self.ignored_limits.is_empty()
    }

    /// The deepest overlap found, or zero.
    pub fn worst_interference(&self) -> f32 {
        self.interferences.first().map(|i| i.depth).unwrap_or(0.0)
    }
}

/// How far a mate can actually travel — see [`Assembly::sweep`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sweep {
    pub mate: MateId,
    /// The travel the mate was declared to have.
    pub declared: (f32, f32),
    /// The travel it reaches before something fouls, either side of the
    /// assembled pose.
    pub clear: (f32, f32),
    /// Where it first fouled, if it did.
    pub blocked_at: Option<f32>,
    /// What it fouled on.
    pub blocker: Option<(PartId, PartId)>,
}

impl Sweep {
    /// Whether the mate reaches both its declared limits.
    pub fn reaches_limits(&self) -> bool {
        self.blocked_at.is_none()
    }
}

/// Why an assembly could not be put together.
#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyError {
    /// A mate names a part that does not exist.
    UnknownPart(PartId),
    /// A mate needs geometry its features do not carry — a hinge given a point
    /// rather than an axis, say.
    WrongFeature {
        mate: MateId,
        wanted: &'static str,
    },
    /// The pose solve could not satisfy this mate. The residual is in world
    /// units and radians added together, so it is a magnitude rather than a
    /// measurement.
    Unsatisfiable { mate: MateId, residual: f32 },
    /// A part produced no collider at all — an empty or degenerate mesh.
    EmptyPart(PartId),
}

impl std::fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPart(p) => write!(f, "mate refers to unknown part {}", p.0),
            Self::WrongFeature { mate, wanted } => {
                write!(f, "mate {} needs {wanted}", mate.0)
            }
            Self::Unsatisfiable { mate, residual } => {
                write!(f, "mate {} left a residual of {residual}", mate.0)
            }
            Self::EmptyPart(p) => write!(f, "part {} produced no collider", p.0),
        }
    }
}

impl std::error::Error for AssemblyError {}

/// A `.scad` model whose declarations do not line up with each other.
///
/// Both cases are typos, and both would otherwise show up much later as a joint
/// that quietly does nothing.
#[cfg(feature = "openscad")]
#[derive(Debug, Clone, PartialEq)]
pub enum ScadMechanismError {
    /// A mate names a part the model never declared.
    UnknownPart { mate: String, part: String },
    /// A drive names a mate the model never declared.
    UnknownMate { drive: String },
}

#[cfg(feature = "openscad")]
impl std::fmt::Display for ScadMechanismError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPart { mate, part } => {
                write!(f, "{mate}() names part \"{part}\", which no part() declared")
            }
            Self::UnknownMate { drive } => {
                write!(f, "drive(\"{drive}\") names a mate that does not exist")
            }
        }
    }
}

#[cfg(feature = "openscad")]
impl std::error::Error for ScadMechanismError {}

/// Gravity for a model drawn the way OpenSCAD draws one.
///
/// OpenSCAD is **z up** and threers, following three.js, is y up. A mechanism
/// converted straight from a `.scad` model therefore falls sideways under
/// [`World`]'s default gravity, and the failure looks like a solver bug rather
/// than a units mistake. Set this on the world, or let
/// [`crate::mechanism::ScadMechanism`] do it for you.
///
/// ```
/// # #[cfg(feature = "openscad")] {
/// use threers_physics::prelude::*;
/// use threers_physics::assembly::SCAD_GRAVITY;
///
/// let mut world = World::new();
/// world.gravity = SCAD_GRAVITY;
/// # }
/// ```
#[cfg(feature = "openscad")]
pub const SCAD_GRAVITY: Vector3 = Vector3::new(0.0, 0.0, -9.81);

/// Convert a limit pair to radians when the mate measures an angle.
fn to_radians_if(angular: bool, min: f32, max: f32) -> (f32, f32) {
    if angular {
        (deg(min), deg(max))
    } else {
        (min, max)
    }
}

// ---- authored motion -----------------------------------------------------

/// How a drive moves between two targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// Constant rate. Starts and stops abruptly, which reads as mechanical —
    /// often exactly right for a machine.
    Linear,
    /// Smoothstep: eases in and out. The default, because a part that starts
    /// moving instantly looks like a rendering error.
    #[default]
    Smooth,
}

impl Easing {
    fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::Smooth => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// One authored move, in progress.
#[derive(Debug, Clone, Copy)]
struct Motion {
    /// Resolved when the move actually starts, from where the joint is then —
    /// not from where the author assumed it would be. A move queued behind
    /// another cannot know its own starting point until the one in front has
    /// finished.
    from: Option<f32>,
    to: f32,
    duration: f32,
    elapsed: f32,
    easing: Easing,
    /// Seconds still to wait before the move begins.
    delay: f32,
}

/// Authors a move on one mate. Commit it with [`Drive::over`] or
/// [`Drive::at_speed`].
pub struct Drive<'a> {
    assembly: &'a mut Assembly,
    mate: MateId,
    to: f32,
    easing: Easing,
    torque: Option<f32>,
    max_speed: Option<f32>,
    delay: f32,
}

impl Drive<'_> {
    /// Where to end up, in the mate's own coordinate: radians for a hinge,
    /// world units for a slider.
    pub fn to(mut self, target: f32) -> Self {
        self.to = target;
        self
    }

    /// How the move is shaped.
    pub fn eased(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    /// Override the force ceiling the assembly would have sized.
    ///
    /// Newton-metres for a hinge, newtons for a slider. Lower it until the
    /// mechanism stalls where a real one would.
    pub fn torque(mut self, max_force: f32) -> Self {
        self.torque = Some(max_force.max(0.0));
        self
    }

    /// Cap the drive's own speed, in rad/s or units/s.
    pub fn max_speed(mut self, speed: f32) -> Self {
        self.max_speed = Some(speed.max(0.0));
        self
    }

    /// Wait this many seconds before starting.
    ///
    /// How a sequence is written: a latch that releases at 1.6 s and a lid that
    /// opens at 1.8 s are two drives with two delays, not a state machine.
    pub fn after(mut self, seconds: f32) -> Self {
        self.delay = seconds.max(0.0);
        self
    }

    /// Take `seconds` to get there.
    ///
    /// A duration is a request, not a promise: if the mechanism cannot move
    /// that fast under the torque it has, or something is in the way, it
    /// arrives late or not at all. That is the point.
    pub fn over(self, seconds: f32) {
        let motion = Motion {
            from: None,
            to: self.to,
            duration: seconds.max(1e-4),
            elapsed: 0.0,
            easing: self.easing,
            delay: self.delay,
        };
        self.commit(motion);
    }

    /// Move at a fixed rate, in rad/s or units/s, however long that takes.
    pub fn at_speed(self, per_second: f32) {
        // The distance is not known until the first update resolves `from`, so
        // the duration is stored as a rate and turned into one there.
        let motion = Motion {
            from: None,
            to: self.to,
            duration: -per_second.abs().max(1e-4),
            elapsed: 0.0,
            easing: Easing::Linear,
            delay: self.delay,
        };
        self.commit(motion);
    }

    /// Hold the current target — used to stop a move part way.
    ///
    /// Drops everything queued behind it too: stopping a move and then
    /// performing the next one anyway is not what "hold" can mean.
    pub fn hold(self) {
        let index = self.mate.0 as usize;
        if index < self.assembly.motions.len() {
            self.assembly.motions[index].clear();
        }
    }

    fn commit(self, motion: Motion) {
        let index = self.mate.0 as usize;
        if index >= self.assembly.mates.len() {
            return;
        }
        if let Some(t) = self.torque {
            self.assembly.drive_force[index] = Some(t);
        }
        if let Some(s) = self.max_speed {
            self.assembly.drive_speed[index] = Some(s);
        }
        // Queued behind whatever is already authored, rather than replacing it.
        // A machine does one thing and then another with the same joint — close
        // the gate, try to open it, release the latch and open it — and each
        // move starts from wherever the one in front actually got to, which is
        // not always where it was aimed.
        self.assembly.motions[index].push(motion);
    }
}

// ---- the assembly --------------------------------------------------------

/// A mechanism: parts, the mates between them, and the motion they are told to
/// perform.
///
/// See the [module docs](self) for the shape of a session.
#[derive(Debug, Clone, Default)]
pub struct Assembly {
    parts: Vec<Part>,
    mates: Vec<Mate>,
    /// Each mate's coordinate in the assembled pose, in *joint* space. What
    /// makes the assembled pose read as zero.
    rest: Vec<f32>,
    /// Residual left by the last [`Self::solve`].
    residual: Vec<f32>,
    /// Queued moves per mate, in authored order — a mate can be told to do
    /// several things in a row, each starting from where the last got to.
    motions: Vec<Vec<Motion>>,
    drive_force: Vec<Option<f32>>,
    drive_speed: Vec<Option<f32>>,
    bodies: Vec<Option<BodyId>>,
    joints: Vec<Option<JointId>>,
    /// Overlaps shallower than this are not interference. Defaults to 1e-3 of a
    /// world unit — a micron on a model drawn in millimetres.
    pub tolerance: f32,
    /// Iterations the pose solve is allowed.
    pub solve_iterations: usize,
}

impl Assembly {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-3,
            solve_iterations: 128,
            ..Default::default()
        }
    }

    // ---- building it up ---------------------------------------------------

    /// Add a part from a drawn mesh.
    ///
    /// The geometry is kept, so the same evaluation both draws and collides —
    /// there is no file in between to disagree with itself.
    pub fn part_from_geometry(
        &mut self,
        name: impl Into<String>,
        geometry: BufferGeometry,
        physics: PartPhysics,
    ) -> PartId {
        let colliders = fit_colliders(&geometry, &physics);
        self.push_part(Part {
            name: name.into(),
            geometry: Some(geometry),
            colliders,
            physics,
            transform: Isometry::IDENTITY,
            scene_object: None,
        })
    }

    /// Add a part from a collision shape, with no mesh to draw.
    pub fn part_from_shape(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        physics: PartPhysics,
    ) -> PartId {
        let collider = Collider::new(shape)
            .material(physics.material)
            .density(physics.density);
        self.push_part(Part {
            name: name.into(),
            geometry: None,
            colliders: vec![collider],
            physics,
            transform: Isometry::IDENTITY,
            scene_object: None,
        })
    }

    /// Add a part modelled in OpenSCAD.
    ///
    /// One evaluation produces the mesh that is drawn and the collider that is
    /// simulated, so they cannot drift apart.
    #[cfg(feature = "openscad")]
    pub fn part(
        &mut self,
        name: impl Into<String>,
        solid: threers::openscad::Solid,
        physics: PartPhysics,
    ) -> PartId {
        self.part_from_geometry(name, solid.to_geometry(), physics)
    }

    /// The same, evaluated with the watertight exact-CSG kernel.
    ///
    /// Worth the extra time when the result feeds a triangle-mesh collider: a
    /// crack the renderer hides is a hole the collider does not.
    #[cfg(feature = "openscad")]
    pub fn part_exact(
        &mut self,
        name: impl Into<String>,
        solid: threers::openscad::Solid,
        physics: PartPhysics,
    ) -> PartId {
        self.part_from_geometry(name, solid.to_geometry_exact(), physics)
    }

    /// Build an assembly from what a `.scad` model said about itself.
    ///
    /// The model declares its parts and joints with `part()`, `hinge()` and the
    /// rest — see [`threers::openscad::mechanism`] — and this turns that into
    /// bodies and mates. Two conversions happen here and nowhere else: angles
    /// come in as **degrees**, because that is what the rest of OpenSCAD uses,
    /// and names become handles.
    ///
    /// The parts arrive already assembled. A model draws the pin and the bore
    /// at the same coordinate, so model space *is* each part's own space and the
    /// pose solve has nothing left to do — [`Self::solve`] on a freshly
    /// converted mechanism should report no residual at all. That is worth
    /// running anyway: if it does not, the model and the declarations disagree,
    /// and it is better to hear about it here than to watch the first frame
    /// snap.
    ///
    /// ```no_run
    /// use threers_physics::prelude::*;
    /// use threers::parse_scad_mechanism_file;
    ///
    /// let spec = parse_scad_mechanism_file("box.scad").unwrap();
    /// let mut asm = Assembly::from_scad(&spec).unwrap();
    /// asm.solve().unwrap();
    /// assert!(asm.check().clear());
    ///
    /// let hinge = asm.mate_named("lid_pivot").unwrap();
    /// # let _ = hinge;
    /// ```
    /// Evaluated with the watertight kernel. A declared mechanism's geometry
    /// becomes its colliders and its interference checks, and the float kernel
    /// is measurably wrong on curved booleans — see
    /// [`crate::mechanism::ScadMechanism::from_spec`]. Use
    /// [`Self::from_scad_parts`] to evaluate them some other way.
    #[cfg(feature = "openscad")]
    pub fn from_scad(
        spec: &threers::openscad::MechanismSpec,
    ) -> Result<Self, ScadMechanismError> {
        Self::from_scad_parts(spec, |part| part.solid.clone().to_geometry_exact())
    }

    /// The same, with the caller deciding how each part's geometry is produced.
    ///
    /// The hook exists because a renderer usually wants a part split into its
    /// `color()` pieces *and* one mesh to collide against, and evaluating the
    /// model twice to get both is the obvious waste this avoids.
    #[cfg(feature = "openscad")]
    pub fn from_scad_parts(
        spec: &threers::openscad::MechanismSpec,
        mut geometry_of: impl FnMut(&threers::openscad::mechanism::PartSpec) -> BufferGeometry,
    ) -> Result<Self, ScadMechanismError> {
        use threers::openscad::mechanism::MateSpecKind;

        use threers::openscad::mechanism::PartFit;

        let mut asm = Self::new();
        for part in &spec.parts {
            let mut physics = if part.fixed {
                PartPhysics::fixed()
            } else {
                PartPhysics::dynamic()
            };
            if let Some(density) = part.density {
                physics = physics.density(density);
            }
            if let Some(mass) = part.mass {
                physics = physics.mass(mass);
            }
            // `collider = "…"`. Left alone, a fixed part keeps every triangle
            // and a moving one gets a convex hull — which is right until the
            // part's job is to *catch* something, and the hull fills in the hook.
            if let Some(fit) = part.fit {
                physics = match fit {
                    PartFit::Hull => physics.fit(ColliderFit::ConvexHull),
                    PartFit::Mesh => physics.fit(ColliderFit::TriMesh),
                    PartFit::Box => physics.fit(ColliderFit::Box),
                    PartFit::Ball => physics.fit(ColliderFit::Ball),
                    PartFit::Capsule => physics.fit(ColliderFit::Capsule),
                    PartFit::Cylinder => physics.fit(ColliderFit::Cylinder),
                    // Decomposition produces convex pieces, so the fit under it
                    // is the hull of each piece rather than of the whole part.
                    PartFit::Decompose => physics.fit(ColliderFit::ConvexHull).decomposed(),
                };
            }
            if let Some(friction) = part.friction {
                physics = physics.friction(friction);
            }
            if let Some(restitution) = part.restitution {
                physics = physics.restitution(restitution);
            }
            asm.part_from_geometry(part.name.clone(), geometry_of(part), physics);
        }

        for mate in &spec.mates {
            let moving = asm.part_named(&mate.parts[0]).ok_or_else(|| {
                ScadMechanismError::UnknownPart {
                    mate: mate.name.clone(),
                    part: mate.parts[0].clone(),
                }
            })?;
            let base = asm.part_named(&mate.parts[1]).ok_or_else(|| {
                ScadMechanismError::UnknownPart {
                    mate: mate.name.clone(),
                    part: mate.parts[1].clone(),
                }
            })?;

            // Model space is each part's own space, so the one coordinate the
            // model wrote serves as the feature on both parts.
            let axis_on = |part: PartId| Feature::axis(part, Axis::new(mate.at, mate.axis));
            let point_on = |part: PartId| Feature::point(part, mate.at);
            let plane_on = |part: PartId| Feature::plane(part, mate.at, mate.axis);

            let mut built = match mate.kind {
                MateSpecKind::Hinge => Mate::hinge(axis_on(moving), axis_on(base)),
                MateSpecKind::Slider => Mate::slider(axis_on(moving), axis_on(base)),
                MateSpecKind::Cylindrical => Mate::cylindrical(axis_on(moving), axis_on(base)),
                MateSpecKind::Ball => Mate::ball(point_on(moving), point_on(base)),
                MateSpecKind::Weld => Mate::rigid(axis_on(moving), axis_on(base)),
                MateSpecKind::Planar => Mate::planar(plane_on(moving), plane_on(base)),
                MateSpecKind::Screw { pitch } => {
                    Mate::screw_from_pitch(axis_on(moving), axis_on(base), pitch)
                }
                MateSpecKind::Gear { ratio } => {
                    let mut gear = Mate::gear(axis_on(moving), axis_on(base), ratio);
                    // A carrier naming a part that does not exist is caught by
                    // `dangling_parts` on the spec; here it simply leaves the
                    // coupling world-referenced rather than failing the build.
                    if let Some(carrier) = mate
                        .carrier
                        .as_deref()
                        .and_then(|name| asm.part_named(name))
                    {
                        gear = gear.carried_by(carrier);
                    }
                    gear
                }
                MateSpecKind::Rack { radius } => {
                    // The pinion spins about `axis`; the rack travels along
                    // `rack_axis`, falling back to the same line when the model
                    // did not say.
                    let travel = mate.axis_b.unwrap_or(mate.axis);
                    Mate::rack(
                        Feature::axis(moving, Axis::new(mate.at, mate.axis)),
                        Feature::axis(base, Axis::new(mate.at, travel)),
                        radius,
                    )
                }
            }
            .named(mate.name.clone());

            if let Some([min, max]) = mate.range {
                let (min, max) = to_radians_if(mate.kind.is_angular(), min, max);
                built = built.limits(min, max);
            }
            asm.mate(built);
        }

        // In the order they happen, not the order they were written. Several
        // drives on one mate queue up, and a model that lists the release before
        // the close means the release to run second all the same.
        let mut ordered: Vec<&threers::openscad::mechanism::DriveSpec> = spec.drives.iter().collect();
        ordered.sort_by(|a, b| a.start.total_cmp(&b.start));

        for drive in ordered {
            let mate = asm.mate_named(&drive.mate).ok_or_else(|| {
                ScadMechanismError::UnknownMate {
                    drive: drive.mate.clone(),
                }
            })?;
            let angular = spec
                .mate(&drive.mate)
                .map(|m| m.kind.is_angular())
                .unwrap_or(false);
            let scale = if angular { deg(1.0) } else { 1.0 };

            let Some(target) = drive.to else {
                // No target: a motor, turning for as long as the simulation
                // runs. The only thing that works for a shaft making full
                // turns, since a hinge angle wraps at ±360° and a position
                // target past that is not expressible.
                let speed = drive.max_speed.unwrap_or(0.0) * scale;
                let force = drive
                    .torque
                    .unwrap_or_else(|| asm.size_drive(mate.index(), 9.81));
                asm.mates[mate.index()].motor = Some(Motor::new(speed, force));
                continue;
            };

            let mut builder = asm.drive(mate).to(target * scale).after(drive.start);
            if let Some(torque) = drive.torque {
                builder = builder.torque(torque);
            }
            if let Some(speed) = drive.max_speed {
                builder = builder.max_speed(speed * scale);
            }
            match drive.over {
                Some(seconds) => builder.over(seconds),
                // No duration given: move at whatever the drive can manage.
                None => builder.at_speed(if angular { deg(120.0) } else { 50.0 }),
            }
        }

        Ok(asm)
    }

    fn push_part(&mut self, part: Part) -> PartId {
        self.parts.push(part);
        self.bodies.push(None);
        PartId(self.parts.len() as u32 - 1)
    }

    /// Where a part starts out. The pose solve corrects it from here, so a
    /// rough placement is enough — but a wildly wrong one can converge to a
    /// different assembly than you meant.
    pub fn place(&mut self, part: PartId, transform: Isometry) -> &mut Self {
        if let Some(p) = self.parts.get_mut(part.0 as usize) {
            p.transform = transform;
        }
        self
    }

    /// Move a part without rotating it.
    pub fn place_at(&mut self, part: PartId, translation: impl IntoVec3) -> &mut Self {
        self.place(part, Isometry::from_translation(translation.into_vec3()))
    }

    /// Add a mate.
    pub fn mate(&mut self, mate: Mate) -> MateId {
        self.mates.push(mate);
        self.rest.push(0.0);
        self.residual.push(0.0);
        self.motions.push(Vec::new());
        self.drive_force.push(None);
        self.drive_speed.push(None);
        self.joints.push(None);
        MateId(self.mates.len() as u32 - 1)
    }

    // ---- reading it back --------------------------------------------------

    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    pub fn part_of(&self, id: PartId) -> Option<&Part> {
        self.parts.get(id.0 as usize)
    }

    /// Find a part by the name it was added under.
    pub fn part_named(&self, name: &str) -> Option<PartId> {
        self.parts
            .iter()
            .position(|p| p.name == name)
            .map(|i| PartId(i as u32))
    }

    /// Find a mate by the name it was added under.
    pub fn mate_named(&self, name: &str) -> Option<MateId> {
        self.mates
            .iter()
            .position(|m| m.name == name)
            .map(|i| MateId(i as u32))
    }

    pub fn mates(&self) -> &[Mate] {
        &self.mates
    }

    pub fn mate_of(&self, id: MateId) -> Option<&Mate> {
        self.mates.get(id.0 as usize)
    }

    /// The body a part became, once [`Self::build`] has run.
    pub fn body_of(&self, part: PartId) -> Option<BodyId> {
        self.bodies.get(part.0 as usize).copied().flatten()
    }

    /// The joint a mate became, once [`Self::build`] has run.
    pub fn joint_of(&self, mate: MateId) -> Option<JointId> {
        self.joints.get(mate.0 as usize).copied().flatten()
    }

    /// Where a mate currently stands, measured from the assembled pose.
    ///
    /// Positive is the first-named part moving by the right-hand rule about the
    /// axis, or along it. `None` for a mate with no single coordinate, or one
    /// that has not been built.
    pub fn coordinate(&self, mate: MateId, world: &World) -> Option<f32> {
        let index = mate.0 as usize;
        let joint = self.joint_of(mate)?;
        let state = world.joint_state(joint)?;
        Some(self.sign(index) * (state.coordinate - self.rest.get(index).copied().unwrap_or(0.0)))
    }

    /// How fast a mate is moving, in rad/s or units/s.
    pub fn speed(&self, mate: MateId, world: &World) -> Option<f32> {
        let joint = self.joint_of(mate)?;
        Some(self.sign(mate.0 as usize) * world.joint_state(joint)?.speed)
    }

    /// Whether every authored move has finished.
    ///
    /// A move that has stalled against something solid still counts as running,
    /// which is what you want when waiting on one: it never finishes, and the
    /// mechanism is telling you why.
    pub fn motion_complete(&self) -> bool {
        self.motions.iter().all(|m| m.is_empty())
    }

    /// A hinge's joint coordinate runs the same way the mate does; a slider's
    /// runs the other way. See [`crate::joint::JointState`].
    fn sign(&self, mate_index: usize) -> f32 {
        match self.mates.get(mate_index).map(|m| m.kind.axial()) {
            Some(Some(crate::joint::AxialKind::Offset)) => -1.0,
            _ => 1.0,
        }
    }

    // ---- the pose solve ---------------------------------------------------

    /// Move the parts until the mates line up.
    ///
    /// Gauss-Seidel over the mates: each one is asked what correction would
    /// satisfy it, and the two parts share that correction according to how free
    /// they are to move — a fixed part takes none of it. Repeat until nothing
    /// is left, which for a well-posed assembly takes a few dozen passes.
    ///
    /// This is not the physics solver and does not pretend to be. It has no
    /// masses, no contacts and no time; it just puts the parts where the mates
    /// say they go, so that the physics starts from a pose it agrees with
    /// instead of spending its first step fixing yours.
    pub fn solve(&mut self) -> Result<(), AssemblyError> {
        self.validate()?;

        let placing: Vec<usize> = (0..self.mates.len())
            .filter(|&i| self.mates[i].kind.places())
            .collect();

        for _ in 0..self.solve_iterations.max(1) {
            let mut worst = 0.0f32;
            for &i in &placing {
                worst = worst.max(self.relax(i));
            }
            if worst <= self.tolerance {
                break;
            }
        }

        // One more pass to record what is left, without moving anything.
        let mut worst: Option<(MateId, f32)> = None;
        for i in 0..self.mates.len() {
            let residual = if self.mates[i].kind.places() {
                self.measure(i)
            } else {
                0.0
            };
            self.residual[i] = residual;
            if residual > self.tolerance
                && worst.map(|(_, r)| residual > r).unwrap_or(true)
            {
                worst = Some((MateId(i as u32), residual));
            }
        }

        self.capture_rest();

        match worst {
            Some((mate, residual)) => Err(AssemblyError::Unsatisfiable { mate, residual }),
            None => Ok(()),
        }
    }

    fn validate(&self) -> Result<(), AssemblyError> {
        for (i, mate) in self.mates.iter().enumerate() {
            for feature in [&mate.a, &mate.b] {
                if feature.part.0 as usize >= self.parts.len() {
                    return Err(AssemblyError::UnknownPart(feature.part));
                }
            }
            let wants_axis = matches!(
                mate.kind,
                MateKind::Hinge
                    | MateKind::Slider
                    | MateKind::Cylindrical
                    | MateKind::Screw { .. }
                    | MateKind::Gear { .. }
                    | MateKind::Rack { .. }
                    | MateKind::Planar
            );
            if wants_axis && (mate.a.direction().is_none() || mate.b.direction().is_none()) {
                return Err(AssemblyError::WrongFeature {
                    mate: MateId(i as u32),
                    wanted: "an axis or plane on both parts",
                });
            }
        }
        for (i, part) in self.parts.iter().enumerate() {
            if part.colliders.is_empty() {
                return Err(AssemblyError::EmptyPart(PartId(i as u32)));
            }
        }
        Ok(())
    }

    /// The correction one mate wants, in world space: how to move part A onto
    /// part B.
    fn correction(&self, index: usize) -> Correction {
        let mate = &self.mates[index];
        let ta = self.parts[mate.a.part.0 as usize].transform;
        let tb = self.parts[mate.b.part.0 as usize].transform;

        let origin_a = ta.transform_point(mate.a.origin());
        let origin_b = tb.transform_point(mate.b.origin());
        let dir_a = mate.a.direction().map(|d| ta.transform_vector(d));
        let dir_b = mate.b.direction().map(|d| tb.transform_vector(d));

        let rotation = match (dir_a, dir_b) {
            (Some(a), Some(b)) => crate::ik::shortest_arc(a, b),
            _ => Quaternion::identity(),
        };

        // A hinge pins the point outright; anything free to slide only cares
        // about the offset *across* its axis.
        let separation = origin_b - origin_a;
        let translation = match mate.kind {
            MateKind::Slider | MateKind::Cylindrical | MateKind::Screw { .. } => {
                match dir_b {
                    Some(axis) => separation - axis * separation.dot(axis),
                    None => separation,
                }
            }
            MateKind::Planar => match dir_b {
                Some(normal) => normal * separation.dot(normal),
                None => separation,
            },
            _ => separation,
        };

        Correction {
            pivot: origin_a,
            rotation,
            translation,
        }
    }

    /// How far one mate is from satisfied.
    fn measure(&self, index: usize) -> f32 {
        let c = self.correction(index);
        c.translation.length() + rotation_angle(c.rotation)
    }

    /// Apply one mate's correction, shared between its two parts.
    fn relax(&mut self, index: usize) -> f32 {
        let mate = &self.mates[index];
        let (pa, pb) = (mate.a.part.0 as usize, mate.b.part.0 as usize);
        if pa == pb {
            return 0.0;
        }
        let free_a = !self.parts[pa].is_grounded();
        let free_b = !self.parts[pb].is_grounded();
        let (wa, wb) = match (free_a, free_b) {
            (true, true) => (0.5, 0.5),
            (true, false) => (1.0, 0.0),
            (false, true) => (0.0, 1.0),
            // Both fixed: nothing can move, so the residual stands and
            // `check` reports it.
            (false, false) => return self.measure(index),
        };

        let c = self.correction(index);
        let error = c.translation.length() + rotation_angle(c.rotation);

        // A moves toward B, and B toward A, each by its share.
        if wa > 0.0 {
            let r = partial_rotation(c.rotation, wa);
            nudge(&mut self.parts[pa], c.pivot, r, c.translation * wa);
        }
        if wb > 0.0 {
            let r = partial_rotation(c.rotation, -wb);
            let pivot = self.parts[pb]
                .transform
                .transform_point(self.mates[index].b.origin());
            nudge(&mut self.parts[pb], pivot, r, c.translation * -wb);
        }
        error
    }

    /// Record each mate's coordinate in the pose it was assembled in, so that
    /// pose reads as zero from here on.
    fn capture_rest(&mut self) {
        for i in 0..self.mates.len() {
            self.rest[i] = self.joint_coordinate_from_transforms(i).unwrap_or(0.0);
        }
    }

    /// A mate's coordinate in *joint* space, read straight off the part
    /// transforms rather than out of a world.
    fn joint_coordinate_from_transforms(&self, index: usize) -> Option<f32> {
        use crate::joint::AxialKind;
        let mate = &self.mates[index];
        let kind = mate.kind.axial()?;
        // The joint is built with B as body A — the thing being moved against.
        let t_ref = self.parts.get(mate.b.part.0 as usize)?.transform;
        let t_moving = self.parts.get(mate.a.part.0 as usize)?.transform;
        let local_axis = mate.b.direction()?;
        Some(match kind {
            AxialKind::Angle => {
                crate::joint::axial_angle(t_ref.rotation, t_moving.rotation, local_axis)
            }
            AxialKind::Offset => {
                let anchor_ref = t_ref.transform_point(mate.b.origin());
                let anchor_moving = t_moving.transform_point(mate.a.origin());
                (anchor_ref - anchor_moving).dot(t_ref.transform_vector(local_axis))
            }
        })
    }

    // ---- checking it ------------------------------------------------------

    /// Interference, unmet mates, and the mechanism's degrees of freedom.
    ///
    /// Run it after [`Self::solve`] and before you commit to the design. It
    /// tests the assembled pose only — for whether a hinge can actually reach
    /// its stop, see [`Self::sweep`].
    pub fn check(&self) -> AssemblyReport {
        let mut report = AssemblyReport {
            ungrounded: !self.parts.iter().any(Part::is_grounded),
            mobility: self.mobility(),
            ..Default::default()
        };

        for (i, &residual) in self.residual.iter().enumerate() {
            if residual > self.tolerance {
                report.unsatisfied.push((MateId(i as u32), residual));
            }
        }
        for (i, mate) in self.mates.iter().enumerate() {
            if mate.limits.is_some() && !mate.kind.accepts_limits() {
                report.ignored_limits.push(MateId(i as u32));
            }
        }

        let mated = self.mated_pairs();
        for a in 0..self.parts.len() {
            for b in (a + 1)..self.parts.len() {
                if mated.contains(&(a, b)) {
                    continue;
                }
                match self.interference_between(a, b) {
                    PairResult::Clear => {}
                    PairResult::Overlap { point, depth } => report.interferences.push(Interference {
                        a: PartId(a as u32),
                        b: PartId(b as u32),
                        point,
                        depth,
                    }),
                    PairResult::Untestable => report
                        .unchecked
                        .push((PartId(a as u32), PartId(b as u32))),
                }
            }
        }
        report
            .interferences
            .sort_by(|x, y| y.depth.total_cmp(&x.depth));
        report
    }

    /// Grübler–Kutzbach: `6(n − 1 − j) + Σf`, counting the grounded parts as one
    /// link.
    fn mobility(&self) -> i32 {
        let grounded = self.parts.iter().filter(|p| p.is_grounded()).count();
        // Every fixed part is the same link — the ground — however many there
        // are, so they collapse into one.
        let links = self.parts.len() as i32 - grounded as i32 + if grounded > 0 { 1 } else { 0 };
        let joints = self.mates.len() as i32;
        let freedoms: i32 = self.mates.iter().map(|m| m.kind.freedoms()).sum();
        6 * (links - 1 - joints) + freedoms
    }

    /// Pairs of part indices that a mate connects, so their designed overlap is
    /// not reported as a fault.
    fn mated_pairs(&self) -> std::collections::HashSet<(usize, usize)> {
        let mut set = std::collections::HashSet::new();
        for mate in &self.mates {
            if mate.collide {
                continue;
            }
            let (a, b) = (mate.a.part.0 as usize, mate.b.part.0 as usize);
            set.insert((a.min(b), a.max(b)));
        }
        set
    }

    fn interference_between(&self, a: usize, b: usize) -> PairResult {
        let (pa, pb) = (&self.parts[a], &self.parts[b]);
        self.interference_at(pa, &pa.transform, pb, &pb.transform)
    }

    /// The same, with the poses given rather than taken from the parts — what
    /// the sweep needs.
    fn interference_at(
        &self,
        pa: &Part,
        ta: &Isometry,
        pb: &Part,
        tb: &Isometry,
    ) -> PairResult {
        let mut deepest: Option<(Vector3, f32)> = None;
        let mut testable = false;
        let mut manifolds = Vec::new();

        for ca in &pa.colliders {
            for cb in &pb.colliders {
                if is_surface(&ca.shape) && is_surface(&cb.shape) {
                    continue;
                }
                testable = true;
                manifolds.clear();
                crate::narrowphase::collide(
                    &ca.shape,
                    &ca.world_transform(ta),
                    &cb.shape,
                    &cb.world_transform(tb),
                    0.0,
                    &mut manifolds,
                );
                for m in &manifolds {
                    for p in &m.points {
                        if p.depth > self.tolerance
                            && deepest.map(|(_, d)| p.depth > d).unwrap_or(true)
                        {
                            deepest = Some((p.point_a, p.depth));
                        }
                    }
                }
            }
        }

        match (deepest, testable) {
            (Some((point, depth)), _) => PairResult::Overlap { point, depth },
            (None, true) => PairResult::Clear,
            (None, false) => PairResult::Untestable,
        }
    }

    /// Drive one mate through its declared travel and find where it fouls.
    ///
    /// This is the question a drawing cannot answer: the lid is *specified* to
    /// open 105°, but does it, or does its corner catch the box at 92°? The
    /// sweep moves everything on the mate's free side rigidly through the range
    /// and tests it against everything else at each step.
    ///
    /// `None` when the mate has no travel to sweep, or when both its sides are
    /// tied to ground — a closed loop cannot be swept one mate at a time.
    ///
    /// Sampling is what it is: a foul narrower than the step between samples is
    /// missed, so raise `samples` when the clearance is tight.
    pub fn sweep(&self, mate: MateId, samples: usize) -> Option<Sweep> {
        let index = mate.0 as usize;
        let m = self.mates.get(index)?;
        let kind = m.kind.axial()?;
        let limits = m.limits?;
        let (moving, reversed) = self.free_side(index)?;

        let axis_part = if reversed { m.a.part } else { m.b.part };
        let axis_feature = if reversed { &m.a } else { &m.b };
        let t = self.parts[axis_part.0 as usize].transform;
        let axis = t.transform_vector(axis_feature.direction()?);
        let pivot = t.transform_point(axis_feature.origin());
        // Moving the other side means travelling the other way.
        let orientation = if reversed { -1.0 } else { 1.0 };

        let samples = samples.max(2);
        let mut clear = (0.0f32, 0.0f32);
        let mut blocked_at = None;
        let mut blocker = None;

        // Walk outward from the assembled pose in both directions, stopping at
        // the first foul: what matters is the travel that is reachable, and past
        // a blockage nothing is.
        for direction in [1.0f32, -1.0] {
            let span = if direction > 0.0 { limits.max } else { limits.min };
            if span * direction <= 0.0 {
                continue;
            }
            let steps = samples;
            for step in 1..=steps {
                let value = span * (step as f32 / steps as f32);
                let placed = self.pose_at(&moving, kind, value * orientation, pivot, axis);
                if let Some((a, b, _)) = self.first_foul(&moving, &placed) {
                    blocked_at = Some(match blocked_at {
                        Some(previous) if value.abs() >= f32::abs(previous) => previous,
                        _ => value,
                    });
                    blocker = blocker.or(Some((PartId(a as u32), PartId(b as u32))));
                    break;
                }
                if direction > 0.0 {
                    clear.1 = value;
                } else {
                    clear.0 = value;
                }
            }
        }

        Some(Sweep {
            mate,
            declared: (limits.min, limits.max),
            clear,
            blocked_at,
            blocker,
        })
    }

    /// Where the moving parts sit when the mate has travelled `value`.
    fn pose_at(
        &self,
        moving: &[usize],
        kind: crate::joint::AxialKind,
        value: f32,
        pivot: Vector3,
        axis: Vector3,
    ) -> Vec<Isometry> {
        use crate::joint::AxialKind;
        moving
            .iter()
            .map(|&i| {
                let t = self.parts[i].transform;
                match kind {
                    AxialKind::Angle => {
                        let r = Quaternion::from_axis_angle(axis, value);
                        Isometry::new(
                            pivot + (t.translation - pivot).apply_quaternion(r),
                            r.multiply(t.rotation).normalize(),
                        )
                    }
                    AxialKind::Offset => {
                        Isometry::new(t.translation + axis * value, t.rotation)
                    }
                }
            })
            .collect()
    }

    /// First interference between the displaced parts and everything else.
    fn first_foul(
        &self,
        moving: &[usize],
        placed: &[Isometry],
    ) -> Option<(usize, usize, f32)> {
        let mated = self.mated_pairs();
        for (slot, &a) in moving.iter().enumerate() {
            for b in 0..self.parts.len() {
                if moving.contains(&b) {
                    continue;
                }
                let key = (a.min(b), a.max(b));
                if mated.contains(&key) {
                    continue;
                }
                if let PairResult::Overlap { depth, .. } = self.interference_at(
                    &self.parts[a],
                    &placed[slot],
                    &self.parts[b],
                    &self.parts[b].transform,
                ) {
                    return Some((a, b, depth));
                }
            }
        }
        None
    }

    /// Which side of a mate is free to move, and whether it is the second one.
    ///
    /// Cut the mate out of the graph and see which of the two halves still
    /// reaches ground. If neither does the mechanism is a loop and there is
    /// nothing single-mate to sweep; if both do, likewise.
    fn free_side(&self, index: usize) -> Option<(Vec<usize>, bool)> {
        let mate = &self.mates[index];
        let side_a = self.component_without(mate.a.part.0 as usize, index);
        let side_b = self.component_without(mate.b.part.0 as usize, index);
        if side_a.contains(&(mate.b.part.0 as usize)) {
            return None; // the two sides are the same component: a closed loop
        }
        let grounded = |side: &[usize]| side.iter().any(|&i| self.parts[i].is_grounded());
        match (grounded(&side_a), grounded(&side_b)) {
            (false, _) => Some((side_a, false)),
            (true, false) => Some((side_b, true)),
            (true, true) => None,
        }
    }

    /// Parts reachable from `start` through placing mates, ignoring one mate.
    fn component_without(&self, start: usize, skip: usize) -> Vec<usize> {
        let mut seen = vec![false; self.parts.len()];
        let mut stack = vec![start];
        seen[start] = true;
        let mut out = Vec::new();
        while let Some(part) = stack.pop() {
            out.push(part);
            for (i, mate) in self.mates.iter().enumerate() {
                if i == skip || !mate.kind.places() {
                    continue;
                }
                let (a, b) = (mate.a.part.0 as usize, mate.b.part.0 as usize);
                let next = if a == part {
                    b
                } else if b == part {
                    a
                } else {
                    continue;
                };
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        out.sort_unstable();
        out
    }

    // ---- into a world -----------------------------------------------------

    /// Create the bodies and joints.
    ///
    /// Every part becomes one body at the pose the solve put it in, and every
    /// mate becomes the joint that holds it. Because the anchors and axes are
    /// the ones you declared on the parts — in part coordinates, not world ones
    /// — the joint agrees with the geometry by construction rather than by
    /// arithmetic you had to get right.
    pub fn build(&mut self, world: &mut World) -> Result<(), AssemblyError> {
        self.validate()?;
        if self.rest.iter().all(|r| *r == 0.0) {
            // Never solved: the current poses are the assembled ones.
            self.capture_rest();
        }

        for (i, part) in self.parts.iter().enumerate() {
            let mut builder = RigidBodyBuilder::new(part.physics.body_type)
                .position(part.transform);
            for collider in &part.colliders {
                builder = builder.collider(collider.clone());
            }
            if let Some(mass) = part.physics.mass {
                builder = builder.mass(mass);
            }
            if let Some(node) = part.scene_object {
                builder = builder.scene_object(node);
            }
            self.bodies[i] = Some(world.add_body(builder));
        }

        let gravity = world.gravity.length().max(1.0);
        for i in 0..self.mates.len() {
            let Some(joint) = self.make_joint(i, world, gravity) else {
                continue;
            };
            self.joints[i] = Some(world.add_joint(joint));
        }
        Ok(())
    }

    /// Build the bodies, the joints, *and* a scene node per part.
    ///
    /// The bodies drive the nodes, so one [`World::sync_to_scene`] per frame
    /// keeps the picture and the simulation in step.
    pub fn spawn(
        &mut self,
        arena: &mut ObjectArena,
        parent: Option<ObjectId>,
        world: &mut World,
        material: Material,
    ) -> Result<(), AssemblyError> {
        for part in &mut self.parts {
            let Some(geometry) = part.geometry.clone() else {
                continue;
            };
            let node = arena.insert(Object3D::mesh(Mesh::new(geometry, material.clone())));
            if let Some(parent) = parent {
                arena.add_child(parent, node);
            }
            part.scene_object = Some(node);
        }
        self.build(world)
    }

    fn make_joint(&mut self, index: usize, world: &World, gravity: f32) -> Option<Joint> {
        let mate = &self.mates[index];
        // The joint's A is the part being mated *against*, so its B-relative-to-A
        // coordinate is the moving part's — which is the one anybody asking
        // about a hinge means.
        let body_a = self.body_of(mate.b.part)?;
        let body_b = self.body_of(mate.a.part)?;
        let anchor_a = mate.b.origin();
        let anchor_b = mate.a.origin();
        let axis_a = mate.b.direction().unwrap_or(Vector3::UP);
        let axis_b = mate.a.direction().unwrap_or(Vector3::UP);
        let rest = self.rest[index];

        let mut joint = match mate.kind {
            MateKind::Hinge => {
                Joint::revolute(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
            }
            MateKind::Slider => {
                Joint::prismatic(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
            }
            MateKind::Cylindrical => {
                // A cylindrical pair is a hinge that is also free to slide, and
                // the generic joint is the only kind that can say that. Its
                // axes have to be turned onto the mate's, or the pair would
                // only be right for a bore that happened to lie along x.
                let (fa, fb) = joint_frames(axis_a, axis_b);
                // Two freedoms and one `limits`, so the limit is the *travel* —
                // a shaft's end float, which is the one people write down. The
                // rotation stays free; a cylindrical pair with a limited angle
                // is a hinge, and there is a mate for that.
                let slide = match mate.limits {
                    Some(l) => crate::joint::Dof::limited(l.min, l.max),
                    None => crate::joint::Dof::FREE,
                };
                Joint::generic_framed(
                    world.bodies(),
                    body_a,
                    body_b,
                    anchor_a,
                    anchor_b,
                    fa,
                    fb,
                )?
                .with_linear_dof(0, slide)
                .with_angular_dof(0, crate::joint::Dof::FREE)
            }
            MateKind::Screw { lead } => Joint::screw(
                world.bodies(),
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                axis_a,
                axis_b,
                lead,
            )?,
            MateKind::Ball => Joint::spherical(body_a, body_b, anchor_a, anchor_b),
            MateKind::Rigid => Joint::fixed(world.bodies(), body_a, body_b, anchor_a, anchor_b),
            MateKind::Planar => {
                // The joint's x is the plane's normal, so what is left free is
                // sliding in the plane and spinning about the normal.
                let (fa, fb) = joint_frames(axis_a, axis_b);
                Joint::generic_framed(
                    world.bodies(),
                    body_a,
                    body_b,
                    anchor_a,
                    anchor_b,
                    fa,
                    fb,
                )?
                .with_linear_dof(1, crate::joint::Dof::FREE)
                .with_linear_dof(2, crate::joint::Dof::FREE)
                .with_angular_dof(0, crate::joint::Dof::FREE)
            }
            // The couplings do *not* take the swap the placement mates take.
            //
            // A placement mate puts the base first so the joint's
            // B-relative-to-A coordinate comes out as the moving part's, which
            // is what anyone asking a hinge its angle means. A coupling has no
            // coordinate to report, and the swap would instead invert what it
            // constrains: `ratio` reads "turns of the first-named part per turn
            // of the second", and reversing the pair silently turns a 2:1
            // reduction into a 1:2 overdrive.
            MateKind::Gear { ratio } => match mate.carrier.and_then(|c| self.body_of(c)) {
                Some(carrier) => {
                    Joint::gear_on_carrier(body_b, body_a, carrier, axis_b, axis_a, ratio)
                }
                None => Joint::gear(body_b, body_a, axis_b, axis_a, ratio),
            },
            MateKind::Rack { radius } => Joint::rack_pinion(
                // The pinion is the first-named part and the rack the second,
                // matching `Joint::rack_pinion`'s own order.
                body_b, body_a, anchor_b, anchor_a, axis_b, axis_a, radius,
            ),
        };

        // Limits and drives are declared in the mate's coordinate; the joint
        // measures its own. Convert once, here, so nothing downstream has to.
        let sign = self.sign(index);
        if let Some(l) = mate.limits {
            let (lo, hi) = (rest + sign * l.min, rest + sign * l.max);
            joint = joint.with_limits(lo.min(hi), lo.max(hi));
        }
        if let Some(m) = mate.motor {
            joint = joint.with_motor(sign * m.target_velocity, m.max_force);
        }
        if let Some(s) = mate.servo {
            joint = joint.with_servo(Servo {
                target: rest + sign * s.target,
                ..s
            });
        } else if !self.motions[index].is_empty() {
            // A move was authored before the build, so the drive it needs has
            // to exist by the time the joint does.
            //
            // Keyed on there being a *move*, not on there being a stated force.
            // A continuous drive records its ceiling too, and fitting a servo
            // for it would hold the shaft at the angle it started from while
            // its own motor tried to turn it — two drives on one joint, pulling
            // against each other, and the joint goes nowhere.
            let force = self.drive_force[index].unwrap_or_else(|| self.size_drive(index, gravity));
            let mut servo = Servo::new(rest, force);
            if let Some(speed) = self.drive_speed[index] {
                servo = servo.max_speed(speed);
            }
            joint = joint.with_servo(servo);
        }
        joint.softness = mate.softness;
        joint.break_impulse = mate.break_impulse;
        joint.collide_connected = mate.collide;
        Some(joint)
    }

    /// Torque (or force) a drive needs to hold the part it moves against
    /// gravity, with headroom.
    ///
    /// Statics, not guesswork: the part's own mass at its own lever arm about
    /// the mate's axis is the worst the drive has to hold, and four times that
    /// leaves room to accelerate it. It is a default, not a specification —
    /// [`Drive::torque`] is there for when you know the motor.
    fn size_drive(&self, index: usize, gravity: f32) -> f32 {
        const HEADROOM: f32 = 4.0;
        let mate = &self.mates[index];
        let Some(part) = self.parts.get(mate.a.part.0 as usize) else {
            return 1.0;
        };
        let mp = part.mass_properties();
        let weight = mp.mass * gravity;
        match mate.kind.axial() {
            Some(crate::joint::AxialKind::Angle) => {
                // Distance from the hinge axis to the centre of mass, in the
                // part's own frame — which is where both are expressed.
                let axis = mate.a.direction().unwrap_or(Vector3::UP);
                let offset = mp.center_of_mass - mate.a.origin();
                let lever = (offset - axis * offset.dot(axis)).length();
                (weight * lever * HEADROOM).max(1e-3)
            }
            _ => (weight * HEADROOM).max(1e-3),
        }
    }

    // ---- driving it -------------------------------------------------------

    /// Author a move on one mate.
    ///
    /// ```
    /// # use threers_physics::prelude::*;
    /// # use threers_physics::assembly::deg;
    /// # let mut asm = Assembly::new();
    /// # let base = asm.part_from_shape("base", Shape::cuboid(1.0, 0.1, 1.0), PartPhysics::fixed());
    /// # let arm = asm.part_from_shape("arm", Shape::cuboid(1.0, 0.1, 1.0), PartPhysics::dynamic());
    /// # let hinge = asm.mate(Mate::hinge(
    /// #     Feature::axis(arm, Axis::x_at([0.0, 0.0, 0.0])),
    /// #     Feature::axis(base, Axis::x_at([0.0, 0.0, 0.0])),
    /// # ));
    /// asm.drive(hinge).to(deg(105.0)).over(1.2);
    /// ```
    ///
    /// The move is realised by a [`Servo`], so it obeys the mate's limits and
    /// anything it runs into. Call [`Self::update`] once per frame to advance
    /// it.
    pub fn drive(&mut self, mate: MateId) -> Drive<'_> {
        Drive {
            assembly: self,
            mate,
            to: 0.0,
            easing: Easing::default(),
            torque: None,
            max_speed: None,
            delay: 0.0,
        }
    }

    /// Advance the authored moves by `dt` and hand the solver its new targets.
    ///
    /// Call this immediately before [`World::step`]. It writes servo targets and
    /// nothing else — no transform is ever set — so a part that cannot reach its
    /// target simply does not.
    pub fn update(&mut self, dt: f32, world: &mut World) {
        for index in 0..self.motions.len() {
            // Every queued move counts down on the same clock, so a start time
            // means what a model says it means: "at 2.5s" is 2.5s into the run,
            // not 2.5s after whatever was in front happened to finish.
            for queued in self.motions[index].iter_mut() {
                queued.delay -= dt;
            }
            let Some(mut motion) = self.motions[index].first().copied() else {
                continue;
            };
            let Some(joint_id) = self.joint_of(MateId(index as u32)) else {
                continue;
            };
            // A move that has not come round yet holds the joint wherever it
            // is — which is a servo doing its job, not an idle one.
            if motion.delay > 0.0 {
                continue;
            }
            // A move authored after the build finds a joint with no drive on
            // it. Fit one now rather than silently doing nothing, which is what
            // "call `drive` whenever you like" has to mean.
            self.fit_drive(index, world);
            let sign = self.sign(index);
            let rest = self.rest[index];

            // Resolve the start from where the joint actually is, the first
            // time round. Authoring happens before the mechanism has moved;
            // this runs after it has.
            let from = match motion.from {
                Some(v) => v,
                None => {
                    let current = world
                        .joint_state(joint_id)
                        .map(|s| sign * (s.coordinate - rest))
                        .unwrap_or(0.0);
                    motion.from = Some(current);
                    if motion.duration < 0.0 {
                        // Stored as a rate: now that the distance is known, it
                        // becomes a duration.
                        let rate = -motion.duration;
                        motion.duration = ((motion.to - current).abs() / rate).max(1e-4);
                    }
                    current
                }
            };

            motion.elapsed += dt;
            let t = (motion.elapsed / motion.duration).clamp(0.0, 1.0);
            let value = from + (motion.to - from) * motion.easing.apply(t);

            if let Some(joint) = world.joint_mut(joint_id) {
                if let Some(servo) = &mut joint.servo {
                    servo.target = rest + sign * value;
                }
            }
            // A body asleep on its stop will not notice a new target, so both
            // ends are woken whenever one is asked to move.
            self.wake(index, world);

            if t >= 1.0 {
                self.motions[index].remove(0);
            } else {
                self.motions[index][0] = motion;
            }
        }
    }

    /// Make sure the mate's joint carries a servo, and that it matches whatever
    /// ceiling and slew rate the author last asked for.
    fn fit_drive(&mut self, index: usize, world: &mut World) {
        let Some(joint_id) = self.joint_of(MateId(index as u32)) else {
            return;
        };
        let sized = self.size_drive(index, world.gravity.length().max(1.0));
        let force = self.drive_force[index];
        let speed = self.drive_speed[index];
        let rest = self.rest[index];
        let Some(joint) = world.joint_mut(joint_id) else {
            return;
        };
        if joint.axial_coordinate_kind().is_none() {
            return;
        }
        match &mut joint.servo {
            Some(servo) => {
                if let Some(f) = force {
                    servo.max_force = f;
                }
                if let Some(s) = speed {
                    servo.max_speed = s;
                }
            }
            slot @ None => {
                let mut servo = Servo::new(rest, force.unwrap_or(sized));
                if let Some(s) = speed {
                    servo = servo.max_speed(s);
                }
                *slot = Some(servo);
            }
        }
    }

    /// Re-derive every drive whose force this crate chose, from the world's
    /// gravity as it stands now.
    ///
    /// A drive with no stated ceiling is sized from the weight it has to hold,
    /// which means it depends on gravity — and gravity is often set *after* the
    /// joints are built, because the units a model is drawn in are a property of
    /// the model rather than of the assembly. A mechanism built at 9.81 and then
    /// told it is in millimetres has every automatic drive a thousand times too
    /// weak, and shows it by not moving.
    ///
    /// Ceilings given explicitly are left alone: those are in the caller's own
    /// units and are not this function's business.
    pub fn resize_drives(&mut self, world: &mut World) {
        let gravity = world.gravity.length().max(1.0);
        for index in 0..self.mates.len() {
            if self.drive_force[index].is_some() {
                continue; // stated, not derived
            }
            let sized = self.size_drive(index, gravity);
            let Some(id) = self.joint_of(MateId::from_index(index)) else {
                continue;
            };
            let Some(joint) = world.joint_mut(id) else {
                continue;
            };
            if let Some(servo) = &mut joint.servo {
                servo.max_force = sized;
            }
            set_motor_force(&mut joint.kind, sized);
        }
    }

    fn wake(&self, index: usize, world: &mut World) {
        let Some(mate) = self.mates.get(index) else {
            return;
        };
        for part in [mate.a.part, mate.b.part] {
            if let Some(body) = self.body_of(part) {
                if let Some(b) = world.body_mut(body) {
                    b.wake_up();
                }
            }
        }
    }

    /// Point a mate at a target and leave it there, with no timed move.
    ///
    /// The servo holds it against whatever load arrives — which is what a
    /// mechanism that is *holding* a position does, as distinct from one moving
    /// to it.
    pub fn hold(&mut self, mate: MateId, target: f32, world: &mut World) {
        let index = mate.0 as usize;
        if index >= self.mates.len() {
            return;
        }
        // Holding a position abandons whatever was queued: performing the next
        // authored move anyway would not be holding.
        self.motions[index].clear();
        self.fit_drive(index, world);
        let (sign, rest) = (self.sign(index), self.rest[index]);
        if let Some(joint_id) = self.joint_of(mate) {
            if let Some(joint) = world.joint_mut(joint_id) {
                if let Some(servo) = &mut joint.servo {
                    servo.target = rest + sign * target;
                }
            }
        }
        self.wake(index, world);
    }
}

/// The move one mate wants, in world space.
struct Correction {
    /// The point the rotation turns about.
    pivot: Vector3,
    rotation: Quaternion,
    translation: Vector3,
}

enum PairResult {
    Clear,
    Overlap { point: Vector3, depth: f32 },
    Untestable,
}

/// Joint frames that put the generic joint's x along each part's mate axis.
///
/// [`Joint::generic`] measures its six degrees of freedom along the bodies' own
/// x, y and z. A bore that does not lie along x needs the axes turned onto it
/// first, or "free to slide along the axis" frees the wrong direction.
fn joint_frames(axis_a: Vector3, axis_b: Vector3) -> (Quaternion, Quaternion) {
    let x = Vector3::new(1.0, 0.0, 0.0);
    (
        crate::ik::shortest_arc(x, axis_a),
        crate::ik::shortest_arc(x, axis_b),
    )
}

/// Turn and shift a part, rotating about a point rather than its own origin.
fn nudge(part: &mut Part, pivot: Vector3, rotation: Quaternion, translation: Vector3) {
    let offset = part.transform.translation - pivot;
    part.transform = Isometry::new(
        pivot + offset.apply_quaternion(rotation) + translation,
        rotation.multiply(part.transform.rotation).normalize(),
    );
}

/// A fraction of a rotation. Negative fractions go the other way.
fn partial_rotation(rotation: Quaternion, fraction: f32) -> Quaternion {
    if fraction >= 0.0 {
        Quaternion::identity().slerp(rotation, fraction)
    } else {
        Quaternion::identity().slerp(rotation.conjugate(), -fraction)
    }
}

/// How far a rotation turns, in radians.
fn rotation_angle(q: Quaternion) -> f32 {
    let w = q.w.abs().clamp(0.0, 1.0);
    2.0 * w.acos()
}

/// Raise the force ceiling on whatever drive a joint carries.
///
/// Sizing a drive is a property of the mate, not of which kind of joint the mate
/// turned into, so this reaches through every kind that has a motor and leaves
/// the ones that do not alone.
fn set_motor_force(kind: &mut JointKind, max_force: f32) {
    let apply = |motor: &mut Option<Motor>| {
        if let Some(m) = motor {
            m.max_force = max_force;
        }
    };
    match kind {
        JointKind::Revolute { motor, .. }
        | JointKind::Prismatic { motor, .. }
        | JointKind::Screw { motor, .. } => apply(motor),
        JointKind::Generic { linear, angular, .. } => {
            for dof in linear.iter_mut().chain(angular.iter_mut()) {
                apply(&mut dof.motor);
            }
        }
        _ => {}
    }
}

/// Whether a shape is a surface rather than a volume, and so cannot be tested
/// against another surface.
fn is_surface(shape: &Shape) -> bool {
    match shape {
        Shape::TriMesh(_) => true,
        Shape::Compound(parts) => parts.iter().all(|(_, s)| is_surface(s)),
        _ => false,
    }
}

/// Build a part's colliders from its mesh.
fn fit_colliders(geometry: &BufferGeometry, physics: &PartPhysics) -> Vec<Collider> {
    let base = |shape: Shape, placement: Isometry| {
        Collider::new(shape)
            .transform(placement)
            .material(physics.material)
            .density(physics.density)
    };

    if physics.decompose {
        if let Some(mesh) = crate::trimesh::TriMesh::from_geometry(geometry) {
            let config = crate::decompose::DecompositionConfig::default();
            if let Some(shape) = crate::decompose::decompose_to_shape(&mesh, &config) {
                return vec![base(shape, Isometry::IDENTITY)];
            }
        }
    }

    match Shape::fit_to_geometry(geometry, physics.fit) {
        Some((shape, placement)) => vec![base(shape, placement)],
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use threers::geometries::BoxGeometry;

    fn block(width: f32, height: f32, depth: f32) -> BufferGeometry {
        BoxGeometry::new(width, height, depth)
    }

    /// A box with a lid hinged along its back top edge.
    ///
    /// The axis points along **−X** so that a positive angle lifts the lid: by
    /// the right-hand rule about +X, the lid's front edge would swing down
    /// through the box instead, and every reading would come out backwards.
    fn hinged_box() -> (Assembly, PartId, PartId, MateId) {
        let mut asm = Assembly::new();
        let body = asm.part_from_geometry("box", block(2.0, 1.0, 1.0), PartPhysics::fixed());
        let lid = asm.part_from_geometry(
            "lid",
            block(2.0, 0.1, 1.0),
            PartPhysics::dynamic().density(50.0),
        );
        // The hinge is on the lid's own back edge and on the box's back top
        // edge — both in their own coordinates, and nowhere near each other
        // until the assembly moves them together.
        let hinge = asm.mate(
            Mate::hinge(
                Feature::axis(lid, Axis::new([0.0, -0.05, -0.5], [-1.0, 0.0, 0.0])),
                Feature::axis(body, Axis::new([0.0, 0.5, -0.5], [-1.0, 0.0, 0.0])),
            )
            .limits(0.0, deg(100.0))
            .named("lid"),
        );
        (asm, body, lid, hinge)
    }

    #[test]
    fn a_mate_pulls_its_parts_together_before_anything_simulates() {
        let (mut asm, _, lid, _) = hinged_box();
        // The lid starts at the origin, on top of the box rather than hinged
        // to it.
        asm.place_at(lid, [0.0, 3.0, 2.0]);
        asm.solve().unwrap();

        let lid_part = asm.part_of(lid).unwrap();
        // Its hinge edge now sits exactly on the box's.
        let world_hinge = lid_part.transform.transform_point(Vector3::new(0.0, -0.05, -0.5));
        assert!(
            (world_hinge - Vector3::new(0.0, 0.5, -0.5)).length() < 1e-3,
            "hinge landed at {world_hinge:?}"
        );
    }

    #[test]
    fn a_solved_assembly_reports_clear() {
        let (mut asm, _, lid, _) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();
        let report = asm.check();
        assert!(report.unsatisfied.is_empty());
        assert!(!report.ungrounded);
        // One hinge between ground and one link: a one-degree-of-freedom
        // mechanism, which is what a lid is.
        assert_eq!(report.mobility, 1);
    }

    #[test]
    fn interference_is_reported_with_a_depth() {
        let mut asm = Assembly::new();
        let a = asm.part_from_geometry("a", block(1.0, 1.0, 1.0), PartPhysics::fixed().fit(ColliderFit::Box));
        let b = asm.part_from_geometry("b", block(1.0, 1.0, 1.0), PartPhysics::dynamic());
        asm.place_at(b, [0.6, 0.0, 0.0]);
        let report = asm.check();
        assert_eq!(report.interferences.len(), 1, "{report:?}");
        let hit = report.interferences[0];
        assert!((hit.depth - 0.4).abs() < 0.05, "depth was {}", hit.depth);
        assert_eq!((hit.a, hit.b), (a, b));
    }

    #[test]
    fn two_mesh_parts_are_reported_as_untestable_rather_than_clear() {
        let mut asm = Assembly::new();
        // Both fixed, so both default to a triangle-mesh collider — two
        // surfaces, with no volume between them to measure.
        asm.part_from_geometry("a", block(1.0, 1.0, 1.0), PartPhysics::fixed());
        let b = asm.part_from_geometry("b", block(1.0, 1.0, 1.0), PartPhysics::fixed());
        asm.place_at(b, [0.5, 0.0, 0.0]);
        let report = asm.check();
        assert!(report.interferences.is_empty());
        assert_eq!(report.unchecked.len(), 1);
        assert!(!report.clear(), "an untestable pair must not read as a pass");
    }

    #[test]
    fn a_built_hinge_reads_zero_in_the_assembled_pose() {
        let (mut asm, _, lid, hinge) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();

        let mut world = World::new();
        asm.build(&mut world).unwrap();
        let angle = asm.coordinate(hinge, &world).unwrap();
        assert!(angle.abs() < 1e-4, "assembled pose read {angle} rad");
    }

    #[test]
    fn a_drive_moves_the_hinge_and_the_limit_stops_it() {
        let (mut asm, _, lid, hinge) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();

        let mut world = World::new();
        world.gravity = Vector3::ZERO; // isolate the drive from the weight
        asm.build(&mut world).unwrap();

        // Ask for more than the limit allows.
        asm.drive(hinge).to(deg(140.0)).over(0.5);
        for _ in 0..180 {
            asm.update(1.0 / 60.0, &mut world);
            world.step(1.0 / 60.0);
        }

        let angle = asm.coordinate(hinge, &world).unwrap();
        assert!(
            (angle - deg(100.0)).abs() < deg(3.0),
            "the limit should have stopped it at 100°, not {}°",
            angle.to_degrees()
        );
    }

    #[test]
    fn a_drive_stalls_against_something_solid() {
        let (mut asm, _, lid, hinge) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();

        let mut world = World::new();
        world.gravity = Vector3::ZERO;
        asm.build(&mut world).unwrap();

        // A shelf over the box, 0.7 above the hinge — the lid's tip reaches it
        // at about 44°, well before the 100° its limit allows.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(1.5, 0.1, 1.0))
                .translation(Vector3::new(0.0, 1.3, 0.0)),
        );

        asm.drive(hinge).to(deg(100.0)).over(0.5);
        for _ in 0..240 {
            asm.update(1.0 / 60.0, &mut world);
            world.step(1.0 / 60.0);
        }

        let angle = asm.coordinate(hinge, &world).unwrap();
        assert!(
            angle < deg(60.0),
            "the shelf should have stopped the lid, but it reached {}°",
            angle.to_degrees()
        );
        assert!(angle > deg(20.0), "it barely moved: {}°", angle.to_degrees());
    }

    #[test]
    fn a_sweep_finds_where_a_lid_fouls() {
        let (mut asm, _, lid, hinge) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();

        // Nothing in the way: the lid reaches its stop.
        let open = asm.sweep(hinge, 48).unwrap();
        assert!(open.reaches_limits(), "{open:?}");
        assert!((open.clear.1 - deg(100.0)).abs() < deg(5.0));

        // Now put a shelf over it. The lid's tip travels on a 1.0 radius from a
        // hinge at y = 0.5, so an obstacle at y = 1.2 stops it near 44°.
        let shelf = asm.part_from_geometry(
            "shelf",
            block(3.0, 0.2, 2.0),
            PartPhysics::fixed().fit(ColliderFit::Box),
        );
        asm.place_at(shelf, [0.0, 1.3, 0.0]);
        let blocked = asm.sweep(hinge, 48).unwrap();
        assert!(!blocked.reaches_limits(), "the shelf should be in the way");
        assert!(
            blocked.clear.1 < deg(60.0),
            "clear travel was {}°",
            blocked.clear.1.to_degrees()
        );
        assert!(
            blocked.clear.1 > deg(30.0),
            "it should still open part way, not {}°",
            blocked.clear.1.to_degrees()
        );
        assert_eq!(blocked.blocker.map(|(_, b)| b), Some(shelf));
    }

    /// The case the whole module exists for: two parts modelled in OpenSCAD,
    /// hinged on a bore that is part of the model rather than a number typed
    /// into the physics.
    #[cfg(feature = "openscad")]
    #[test]
    fn two_scad_parts_hinge_on_their_own_bores() {
        use threers::openscad::{cube, cylinder, difference_all};

        // A bracket with a 3-radius bore through it, along Y at x = 15.
        let bracket = difference_all(vec![
            cube([40.0, 20.0, 8.0]),
            cylinder(30.0, 3.0).translate([15.0, 0.0, 0.0]),
        ]);
        // A lever with the matching bore at one end, at x = -12.
        let lever = difference_all(vec![
            cube([30.0, 8.0, 8.0]),
            cylinder(20.0, 3.0).translate([-12.0, 0.0, 0.0]),
        ]);

        let mut asm = Assembly::new();
        let base = asm.part("bracket", bracket, PartPhysics::fixed());
        let arm = asm.part("lever", lever, PartPhysics::dynamic().density(0.0027));

        // Each axis is the bore's own, in the part it belongs to.
        let pivot = asm.mate(
            Mate::hinge(
                Feature::axis(arm, Axis::y_at([-12.0, 0.0, 0.0])),
                Feature::axis(base, Axis::y_at([15.0, 0.0, 0.0])),
            )
            .limits(deg(-90.0), deg(90.0)),
        );

        // The lever starts somewhere else entirely.
        asm.place_at(arm, [0.0, 60.0, 40.0]);
        asm.solve().unwrap();

        // Its bore is now on the bracket's bore, to within a micron of a part
        // 40 units across.
        let lever_bore = asm
            .part_of(arm)
            .unwrap()
            .transform
            .transform_point(Vector3::new(-12.0, 0.0, 0.0));
        assert!(
            (lever_bore - Vector3::new(15.0, 0.0, 0.0)).length() < 1e-3,
            "the bore landed at {lever_bore:?}"
        );

        let report = asm.check();
        assert_eq!(report.mobility, 1);
        assert!(report.clear(), "{report:?}");

        let mut world = World::new();
        asm.build(&mut world).unwrap();
        assert!(asm.coordinate(pivot, &world).unwrap().abs() < 1e-4);
        // And the mass came off the model, hole and all.
        let mass = world.body(asm.body_of(arm).unwrap()).unwrap().mass();
        assert!(mass > 0.0, "the lever weighs nothing");
    }

    /// The whole path, end to end: a model that describes its own mechanism,
    /// turned into one, checked, and driven.
    #[cfg(feature = "openscad")]
    #[test]
    fn a_scad_model_becomes_a_mechanism_that_runs() {
        // Drawn the way a model is drawn: the lid sits on the box, both in the
        // same coordinates, and the hinge is written once.
        let spec = threers::parse_scad_mechanism(
            r#"
            part("box", fixed = true) cube([0.30, 0.20, 0.16]);
            part("lid") translate([0, 0, 0.16]) cube([0.30, 0.20, 0.012]);

            hinge("lid_pivot", parts = ["lid", "box"],
                  at = [0.15, 0, 0.16], axis = [1, 0, 0], range = [0, 105]);

            drive("lid_pivot", to = 105, over = 1.0);
            "#,
        )
        .unwrap();

        let mut asm = Assembly::from_scad(&spec).unwrap();
        assert_eq!(asm.parts().len(), 2);
        assert!(asm.part_named("box").unwrap() != asm.part_named("lid").unwrap());

        // Degrees on the model side, radians on this one.
        let hinge = asm.mate_named("lid_pivot").unwrap();
        let limits = asm.mate_of(hinge).unwrap().limits.unwrap();
        assert!((limits.max - deg(105.0)).abs() < 1e-5, "{limits:?}");

        // The model drew the parts where they belong, so the pose solve has
        // nothing to do — and says so rather than shifting anything.
        let before = asm.part_of(asm.part_named("lid").unwrap()).unwrap().transform;
        asm.solve().unwrap();
        let after = asm.part_of(asm.part_named("lid").unwrap()).unwrap().transform;
        assert!((before.translation - after.translation).length() < 1e-5);
        assert!(asm.check().clear(), "{:?}", asm.check());

        // Then it runs, under the gravity the model was drawn for.
        let mut world = World::new();
        world.gravity = SCAD_GRAVITY;
        asm.build(&mut world).unwrap();
        assert!(asm.coordinate(hinge, &world).unwrap().abs() < 1e-4);

        for _ in 0..180 {
            asm.update(1.0 / 60.0, &mut world);
            world.step(1.0 / 60.0);
        }
        let angle = asm.coordinate(hinge, &world).unwrap();
        assert!(
            (angle - deg(105.0)).abs() < deg(3.0),
            "the model asked for 105°, the mechanism reached {}°",
            angle.to_degrees()
        );
    }

    /// Drives carry their own place on a timeline, so a model can choreograph.
    #[cfg(feature = "openscad")]
    #[test]
    fn a_drive_waits_for_its_turn() {
        let spec = threers::parse_scad_mechanism(
            r#"
            part("base", fixed = true) cube([0.20, 0.20, 0.02]);
            part("arm") translate([0, 0, 0.02]) cube([0.20, 0.04, 0.02]);
            hinge("elbow", parts = ["arm", "base"],
                  at = [0.10, 0, 0.02], axis = [1, 0, 0], range = [0, 90]);
            drive("elbow", to = 90, over = 0.5, at = 1.0);
            "#,
        )
        .unwrap();

        let mut asm = Assembly::from_scad(&spec).unwrap();
        asm.solve().unwrap();
        let mut world = World::new();
        world.gravity = SCAD_GRAVITY;
        asm.build(&mut world).unwrap();
        let elbow = asm.mate_named("elbow").unwrap();

        // Half a second in, it has not started.
        for _ in 0..30 {
            asm.update(1.0 / 60.0, &mut world);
            world.step(1.0 / 60.0);
        }
        assert!(
            asm.coordinate(elbow, &world).unwrap().abs() < deg(2.0),
            "it started early: {}°",
            asm.coordinate(elbow, &world).unwrap().to_degrees()
        );

        // Two seconds in, it has been and gone.
        for _ in 0..90 {
            asm.update(1.0 / 60.0, &mut world);
            world.step(1.0 / 60.0);
        }
        assert!(
            (asm.coordinate(elbow, &world).unwrap() - deg(90.0)).abs() < deg(3.0),
            "it never arrived: {}°",
            asm.coordinate(elbow, &world).unwrap().to_degrees()
        );
    }

    #[cfg(feature = "openscad")]
    #[test]
    fn a_mate_naming_a_part_that_does_not_exist_is_an_error() {
        let spec = threers::parse_scad_mechanism(
            r#"
            part("lid") cube(10);
            hinge("h", parts = ["lid", "bxo"], at = [0,0,0], axis = [1,0,0]);
            "#,
        )
        .unwrap();
        let err = Assembly::from_scad(&spec).unwrap_err();
        assert_eq!(
            err,
            ScadMechanismError::UnknownPart {
                mate: "h".into(),
                part: "bxo".into()
            }
        );
        assert!(err.to_string().contains("bxo"));
    }

    /// A cylindrical pair frees exactly two motions, and it has to be the two
    /// belonging to the *mate's* axis rather than to the body's x.
    #[test]
    fn a_cylindrical_mate_slides_along_its_own_axis_and_not_across_it() {
        let mut asm = Assembly::new();
        let housing = asm.part_from_shape(
            "housing",
            Shape::cuboid(0.2, 0.6, 0.2),
            PartPhysics::fixed(),
        );
        // A shaft in a vertical bore: free to drop through it and to spin in
        // it, held in the other four.
        let shaft = asm.part_from_shape(
            "shaft",
            Shape::cylinder(0.5, 0.05),
            PartPhysics::dynamic().density(7800.0),
        );
        asm.mate(Mate::cylindrical(
            Feature::axis(shaft, Axis::y_at([0.0, 0.0, 0.0])),
            Feature::axis(housing, Axis::y_at([0.0, 0.0, 0.0])),
        ));
        asm.place_at(shaft, [0.4, 0.3, 0.4]);
        asm.solve().unwrap();

        let mut world = World::new();
        asm.build(&mut world).unwrap();
        let body = asm.body_of(shaft).unwrap();
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }

        let p = world.body(body).unwrap().translation();
        // Gravity is along the bore, so it falls freely.
        assert!(p.y < -1.0, "the shaft should have dropped, it is at y = {}", p.y);
        // And is held in the two directions across it. Were the joint still
        // measuring along the body's x, one of these would be the free one.
        assert!(
            p.x.abs() < 1e-3 && p.z.abs() < 1e-3,
            "the shaft wandered out of its bore to ({}, {})",
            p.x,
            p.z
        );
    }

    #[test]
    fn gravity_alone_swings_the_lid_shut_against_the_stop() {
        let (mut asm, _, lid, hinge) = hinged_box();
        asm.place_at(lid, [0.0, 2.0, 0.0]);
        asm.solve().unwrap();

        let mut world = World::new();
        asm.build(&mut world).unwrap();
        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        // The range is 0..100°, and the lid hangs at the bottom of it.
        let angle = asm.coordinate(hinge, &world).unwrap();
        assert!(angle.abs() < deg(2.0), "it settled at {}°", angle.to_degrees());
    }
}
