//! Tendons — a cable routed through a chain of points, constrained by its
//! total length.
//!
//! A [`Joint::distance`](crate::joint::Joint::distance) ties two anchors
//! together and knows nothing about anything in between. A tendon is the same
//! idea with the middle put back: a path through any number of via-points on
//! any number of bodies, with **one** scalar coordinate — the summed length of
//! every segment — and one constraint on it.
//!
//! That single coordinate is the whole point. Pulling on the end of a real
//! cable does not pull each guide separately; it shortens the path, and every
//! body the path touches feels its share along whatever direction the cable
//! happens to leave that guide. Getting that from ordinary two-body joints
//! needs one per segment and a fiction about where the slack goes.
//!
//! | Kind | What it is |
//! |---|---|
//! | [`TendonKind::Limit`] | a rope: resists stretching past `max`, slack below it |
//! | [`TendonKind::Force`] | a constant pull — a hanging weight, or pretension |
//! | [`TendonKind::Spring`] | an elastic cord, or a length actuator with gain `stiffness` |
//! | [`TendonKind::Servo`] | a winch: reels to a length under a force ceiling |
//!
//! # What the path is made of
//!
//! A path is a list of [`TendonNode`]s, and a list of nothing but
//! [`TendonNode::Via`] is the polyline above — a cable through eyelets.
//!
//! | Node | What it adds |
//! |---|---|
//! | [`TendonNode::Via`] | a point the path passes through |
//! | [`TendonNode::Sphere`] | an obstacle the path bends around, on tangents |
//! | [`TendonNode::Cylinder`] | the same, about an axis — a pulley wheel or a drum |
//! | [`TendonNode::Pulley`] | a break into a separate strand, counted for `1/divisor` |
//!
//! A via point is a guide with no size, which is the wrong model for the one
//! part everybody wants: a cable over a *wheel* leaves and rejoins it on
//! tangents, is longer than the straight line by the arc it runs along, and
//! pushes the wheel along the bisector of the two tangents with no torque about
//! its axle. Routed through the wheel's centre instead, it is short, it is
//! straight through solid material, and it applies no moment arm at all — which
//! is the single number a pulley exists to provide.
//!
//! An obstacle engages only while the straight line would cut it, so a taut
//! cable that clears the wheel costs nothing to have declared.
//!
//! ```
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! # let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.1, 0.1, 0.1)));
//! # let arm = world.add_body(RigidBody::dynamic().shape(Shape::cuboid(0.4, 0.02, 0.02))
//! #     .translation(Vector3::new(0.4, 0.0, 0.0)));
//! // A cable from the frame, over a guide on the arm, to the arm's tip.
//! let cable = world.add_tendon(Tendon::new(vec![
//!     TendonPoint::new(frame, Vector3::ZERO),
//!     TendonPoint::new(arm, Vector3::new(-0.2, 0.03, 0.0)),
//!     TendonPoint::new(arm, Vector3::new(0.4, 0.0, 0.0)),
//! ]));
//! // Its rest length is whatever the route measures where it was built.
//! let slack = world.tendon_length(cable).unwrap();
//! world.tendon_mut(cable).unwrap().kind = TendonKind::rope(slack);
//! ```
//!
//! # A bare cable only pulls
//!
//! A rope that pushed would be a rod. [`TendonKind::Force`],
//! [`TendonKind::Spring`] and [`TendonKind::Servo`] therefore shorten the path
//! and never lengthen it, and a slack one applies nothing at all rather than a
//! small negative force.
//!
//! [`TendonKind::Limit`] is the exception, and the `min` bound is why: a cable
//! in a sheath *can* push, and so can a belt trapped on its pulleys. Leave
//! `min` at zero for anything bare.
//!
//! # Cost
//!
//! Unlike the machine elements in [`crate::joint`], tendons are not behind a
//! feature. A world with none pays one emptiness check per step, because the
//! cost is per tendon rather than per constraint row in a hot loop, and cfg-ing
//! a field of [`World`](crate::world::World) buys a build combination that can
//! break instead.

use crate::body::{BodyId, BodySet};
use crate::joint::Softness;
use crate::math::{orthonormal_basis, try_normalize, Isometry};
use threers::math::Vector3;

/// One point the cable passes through, fixed to a body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonPoint {
    pub body: BodyId,
    /// Where on the body, in its own frame.
    pub local: Vector3,
}

impl TendonPoint {
    pub fn new(body: BodyId, local: Vector3) -> Self {
        Self { body, local }
    }
}

/// A round body-fixed obstacle the path bends around.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonObstacle {
    pub body: BodyId,
    /// Centre in the body's frame. For a cylinder, any point on its axis.
    pub center: Vector3,
    pub radius: f32,
    /// Cylinder axis in the body's frame; ignored by a sphere.
    pub axis: Vector3,
    /// Which way round to pass, as a point in the body's frame.
    ///
    /// A cable over a cylinder can go either way, and the two are different
    /// machines: over the top of a winch drum and under it wind opposite
    /// directions. Left `None`, the shorter way wins — which is right for a
    /// pulley the cable is merely deflected by, and wrong for a drum, because
    /// the shorter way flips to the other side as the load swings past centre.
    pub side: Option<Vector3>,
}

/// One element of a tendon's path.
///
/// A bare list of [`TendonNode::Via`] is a cable through a set of eyelets: the
/// path is the polyline between them, and the guide it passes is a point with no
/// size. That is the whole model for most cables and it is wrong for exactly the
/// case people build tendons for — a pulley. A cable "through" the centre of a
/// pulley wheel takes the wrong route, has the wrong length, and applies its
/// force with no moment arm about the wheel at all, which is the one number the
/// pulley exists to provide.
///
/// [`TendonNode::Sphere`] and [`TendonNode::Cylinder`] put the size back: the
/// path leaves the previous point on a tangent, runs along the surface, and
/// leaves on a tangent again. It engages only while the straight line would cut
/// the obstacle, so a taut cable that clears the wheel costs nothing and a slack
/// one wraps.
///
/// [`TendonNode::Pulley`] is a different thing with a confusing name — it splits
/// the path into strands. See its own documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TendonNode {
    /// A point the path passes through, fixed to a body.
    Via(TendonPoint),
    /// A sphere the path wraps around. Must sit between two [`Self::Via`]s.
    Sphere(TendonObstacle),
    /// An infinite cylinder the path wraps around. Must sit between two
    /// [`Self::Via`]s.
    ///
    /// Infinite along its axis, so it is the right model for a pulley wheel or a
    /// drum and the wrong one for the end of a roller, where the cable would
    /// slide off in reality and here will not.
    Cylinder(TendonObstacle),
    /// Start a new strand, whose length counts for `1/divisor`.
    ///
    /// Nothing connects the strand before a pulley to the strand after it, and
    /// they need not even be near each other: a tendon is a length and a
    /// gradient, and both are sums over the strands. That is what makes this
    /// worth having — the strands of a reeving really are separate pieces of
    /// rope sharing one coordinate.
    ///
    /// **The divisor belongs to the strand its pulley starts**, so one physical
    /// pulley splitting a line in two is *two* of these, one before each half,
    /// each dividing by 2. A path with no pulley in it is a single strand
    /// divided by 1.
    ///
    /// # What dividing actually does
    ///
    /// It changes what the tendon's one coordinate measures, and the force
    /// follows by duality. Two strands to a load, each halved, make the
    /// coordinate track *the load's travel* rather than the rope's: the load
    /// falls a metre, both strands lengthen a metre, and the coordinate gains
    /// one metre rather than two. The force conjugate to that coordinate is then
    /// the force at the load — the whole weight — while each strand physically
    /// carries half.
    ///
    /// That is mechanical advantage expressed as a change of coordinate, and it
    /// is the reason [`Tendon::tension`] reads differently on a reeved path than
    /// on a plain one. Leave the divisors out and the coordinate is the rope,
    /// and the tension is the rope's.
    Pulley { divisor: f32 },
}

impl TendonNode {
    /// A point the path passes through.
    pub fn via(body: BodyId, local: Vector3) -> Self {
        Self::Via(TendonPoint::new(body, local))
    }

    /// A sphere fixed to `body`, centred at `center` in its frame.
    pub fn sphere(body: BodyId, center: Vector3, radius: f32) -> Self {
        Self::Sphere(TendonObstacle {
            body,
            center,
            radius: radius.max(0.0),
            axis: Vector3::UP,
            side: None,
        })
    }

    /// An infinite cylinder about `axis` through `center`, both in `body`'s
    /// frame. This is the pulley wheel.
    pub fn cylinder(body: BodyId, center: Vector3, axis: Vector3, radius: f32) -> Self {
        Self::Cylinder(TendonObstacle {
            body,
            center,
            radius: radius.max(0.0),
            axis,
            side: None,
        })
    }

    /// Split the path into a new strand counting for `1/divisor`.
    pub fn pulley(divisor: f32) -> Self {
        Self::Pulley { divisor }
    }

    /// Pass the obstacle on the side `point` (in the obstacle body's frame) is
    /// on, rather than whichever way is shorter.
    pub fn with_side(mut self, point: Vector3) -> Self {
        if let Self::Sphere(o) | Self::Cylinder(o) = &mut self {
            o.side = Some(point);
        }
        self
    }

    /// The body this element is attached to, if any.
    pub fn body(&self) -> Option<BodyId> {
        match self {
            Self::Via(p) => Some(p.body),
            Self::Sphere(o) | Self::Cylinder(o) => Some(o.body),
            Self::Pulley { .. } => None,
        }
    }

    fn obstacle(&self) -> Option<(&TendonObstacle, bool)> {
        match self {
            Self::Sphere(o) => Some((o, false)),
            Self::Cylinder(o) => Some((o, true)),
            _ => None,
        }
    }
}

impl From<TendonPoint> for TendonNode {
    fn from(p: TendonPoint) -> Self {
        Self::Via(p)
    }
}

// ---- wrapping geometry ----------------------------------------------------

/// Do two 2D segments cross?
///
/// Used to reject a tangent pair that would have the cable pass through itself,
/// which is the signature of having taken the wrong way round.
fn segments_cross(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> bool {
    let det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1]);
    if det.abs() < 1e-12 {
        return false;
    }
    let a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det;
    let b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det;
    (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)
}

/// Arc length from `p0` to `p1` around a circle of `radius` at the origin.
///
/// `long_way` picks which of the two arcs, since both are valid paths and only
/// the tangent construction knows which one it built.
fn arc_length(p0: [f32; 2], p1: [f32; 2], long_way: bool, radius: f32) -> f32 {
    let n0 = (p0[0] * p0[0] + p0[1] * p0[1]).sqrt();
    let n1 = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
    if n0 < 1e-9 || n1 < 1e-9 {
        return 0.0;
    }
    let cos = ((p0[0] * p1[0] + p0[1] * p1[1]) / (n0 * n1)).clamp(-1.0, 1.0);
    let mut angle = cos.acos();
    let cross = p0[1] * p1[0] - p0[0] * p1[1];
    if (cross > 0.0) == long_way {
        angle = std::f32::consts::TAU - angle;
    }
    radius * angle
}

/// Where a taut line from `e0` to `e1` leaves and rejoins a circle of `radius`
/// centred at the origin, and the arc between.
///
/// `None` when it does not touch: either endpoint inside the circle, or the
/// straight line already clears it. `side` biases which way round; without it
/// the shorter path wins.
fn wrap_circle(
    e0: [f32; 2],
    e1: [f32; 2],
    side: Option<[f32; 2]>,
    radius: f32,
) -> Option<([f32; 2], [f32; 2], f32)> {
    let sq0 = e0[0] * e0[0] + e0[1] * e0[1];
    let sq1 = e1[0] * e1[0] + e1[1] * e1[1];
    let sqrad = radius * radius;
    // An endpoint inside the obstacle has no tangent, and a circle of no size
    // has nothing to wrap.
    if sq0 < sqrad || sq1 < sqrad || radius < 1e-9 {
        return None;
    }

    let dif = [e1[0] - e0[0], e1[1] - e0[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if dd < 1e-12 {
        return None;
    }

    // Closest approach of the straight line to the centre. If that clears the
    // circle the cable is taut and the obstacle is not touched — unless a side
    // hint says the cable is on the far side and has to come round.
    let a = (-(dif[0] * e0[0] + dif[1] * e0[1]) / dd).clamp(0.0, 1.0);
    let near = [a * dif[0] + e0[0], a * dif[1] + e0[1]];
    let clears = near[0] * near[0] + near[1] * near[1] > sqrad;
    if clears && side.is_none_or(|s| s[0] * near[0] + s[1] * near[1] >= 0.0) {
        return None;
    }

    let root0 = (sq0 - sqrad).sqrt();
    let root1 = (sq1 - sqrad).sqrt();

    // The two tangent pairs — one round each side — scored, then the better
    // taken. A pair whose two straight runs cross each other has the cable
    // passing through itself and is never the answer.
    let mut best: Option<([f32; 2], [f32; 2], f32, bool)> = None;
    for i in 0..2 {
        let sgn = if i == 0 { 1.0 } else { -1.0 };
        let t0 = [
            (e0[0] * sqrad + sgn * radius * e0[1] * root0) / sq0,
            (e0[1] * sqrad - sgn * radius * e0[0] * root0) / sq0,
        ];
        let t1 = [
            (e1[0] * sqrad - sgn * radius * e1[1] * root1) / sq1,
            (e1[1] * sqrad + sgn * radius * e1[0] * root1) / sq1,
        ];
        if segments_cross(e0, t0, e1, t1) {
            continue;
        }
        let score = match side {
            // Closest to the side we were told to pass.
            Some(s) => {
                let mid = [t0[0] + t1[0], t0[1] + t1[1]];
                let n = (mid[0] * mid[0] + mid[1] * mid[1]).sqrt();
                if n < 1e-9 {
                    f32::NEG_INFINITY
                } else {
                    (mid[0] * s[0] + mid[1] * s[1]) / n
                }
            }
            // Otherwise the shorter way: the tangent points closest together.
            None => {
                let d = [t0[0] - t1[0], t0[1] - t1[1]];
                -(d[0] * d[0] + d[1] * d[1])
            }
        };
        if best.is_none_or(|(_, _, b, _)| score > b) {
            best = Some((t0, t1, score, i == 1));
        }
    }

    let (t0, t1, _, long_way) = best?;
    Some((t0, t1, arc_length(t0, t1, long_way, radius)))
}

/// One resolved wrap: where the path met the obstacle and left it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonArc {
    /// Centre of the circle the arc runs on, in world space.
    pub center: Vector3,
    /// Axis the arc turns about, in world space.
    pub axis: Vector3,
    /// Where the path touched down and where it left.
    pub from: Vector3,
    pub to: Vector3,
    /// Swept angle, always positive, about `axis`.
    pub angle: f32,
    /// Length along the surface — longer than `angle * radius` on a cylinder,
    /// where the path also travels along the axis.
    pub length: f32,
}

impl TendonArc {
    /// Sample the arc, `from` and `to` included.
    ///
    /// On a cylinder this is a helix, so the axial travel is spread along it
    /// rather than taken all at one end.
    pub fn sample(&self, count: usize) -> Vec<Vector3> {
        let count = count.max(2);
        let axial = |p: Vector3| (p - self.center).dot(self.axis);
        let (z0, z1) = (axial(self.from), axial(self.to));
        let radial = (self.from - self.center) - self.axis * z0;
        (0..count)
            .map(|i| {
                let t = i as f32 / (count - 1) as f32;
                let q = threers::math::Quaternion::from_axis_angle(self.axis, self.angle * t);
                self.center + radial.apply_quaternion(q) + self.axis * (z0 + (z1 - z0) * t)
            })
            .collect()
    }
}

/// Why a wrap did not happen.
enum NoWrap {
    /// The straight line already clears the obstacle — the ordinary case for a
    /// taut cable, and not a problem.
    Clears,
    /// A path point is inside the obstacle, so there is no tangent to leave on.
    Inside,
}

/// Resolve one obstacle between two world-space path points.
fn wrap_obstacle(
    obstacle: &TendonObstacle,
    is_cylinder: bool,
    iso: &Isometry,
    p0: Vector3,
    p1: Vector3,
) -> Result<TendonArc, NoWrap> {
    let center = iso.transform_point(obstacle.center);
    let (d0, d1) = (p0 - center, p1 - center);

    // The frame the circle lives in: for a cylinder its own axis, for a sphere
    // the plane through both points and the centre — a sphere's wrap is a great
    // circle, and that is the great circle it lies on.
    let (u, v, axis) = if is_cylinder {
        let Some(axis) = try_normalize(iso.transform_vector(obstacle.axis)) else {
            return Err(NoWrap::Clears);
        };
        let (u, v) = orthonormal_basis(axis);
        (u, v, axis)
    } else {
        let Some(u) = try_normalize(d0) else {
            return Err(NoWrap::Inside);
        };
        // Collinear points leave the plane undetermined; any plane through them
        // contains the same great circle, so take one.
        let axis = try_normalize(u.cross(d1)).unwrap_or_else(|| orthonormal_basis(u).0);
        let v = axis.cross(u);
        (u, v, axis)
    };

    let flat = |d: Vector3| [d.dot(u), d.dot(v)];
    let (e0, e1) = (flat(d0), flat(d1));
    if e0[0] * e0[0] + e0[1] * e0[1] < obstacle.radius * obstacle.radius
        || e1[0] * e1[0] + e1[1] * e1[1] < obstacle.radius * obstacle.radius
    {
        return Err(NoWrap::Inside);
    }

    let side = obstacle.side.map(|s| {
        let f = flat(iso.transform_point(s) - center);
        let n = (f[0] * f[0] + f[1] * f[1]).sqrt();
        if n < 1e-9 {
            [0.0, 0.0]
        } else {
            [f[0] / n * obstacle.radius, f[1] / n * obstacle.radius]
        }
    });

    let Some((t0, t1, planar_arc)) = wrap_circle(e0, e1, side, obstacle.radius) else {
        return Err(NoWrap::Clears);
    };

    let lift = |t: [f32; 2]| u * t[0] + v * t[1];
    let (mut from, mut to) = (lift(t0), lift(t1));
    let mut length = planar_arc;

    if is_cylinder {
        // The tangent points are only fixed in the plane; along the axis the
        // path climbs steadily, so give each its share of the climb by how far
        // along the whole route it sits. Then the arc is a helix and longer than
        // its shadow.
        let (z0, z1) = (d0.dot(axis), d1.dot(axis));
        let run0 = ((e0[0] - t0[0]).powi(2) + (e0[1] - t0[1]).powi(2)).sqrt();
        let run1 = ((e1[0] - t1[0]).powi(2) + (e1[1] - t1[1]).powi(2)).sqrt();
        let total = run0 + planar_arc + run1;
        if total > 1e-9 {
            let za = z0 + (z1 - z0) * run0 / total;
            let zb = z0 + (z1 - z0) * (run0 + planar_arc) / total;
            from = from + axis * za;
            to = to + axis * zb;
            let climb = zb - za;
            length = (planar_arc * planar_arc + climb * climb).sqrt();
        }
    }

    let angle = if obstacle.radius > 1e-9 {
        planar_arc / obstacle.radius
    } else {
        0.0
    };
    // Orient the axis so that sweeping `from` by `+angle` about it arrives at
    // `to`. The cross product says which way the short way round goes; when the
    // arc is the long way round — over half the circle — it goes the other.
    let short_way = lift(t0).cross(lift(t1)).dot(axis) > 0.0;
    let long_arc = angle > std::f32::consts::PI;
    let axis = if short_way == long_arc { -axis } else { axis };

    Ok(TendonArc {
        center,
        axis,
        from: center + from,
        to: center + to,
        angle,
        length,
    })
}

/// What a tendon does about its length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TendonKind {
    /// Length held within `[min, max]`.
    ///
    /// `min = 0` is a rope: it resists stretching past `max` and does nothing
    /// below it. A non-zero `min` is a path that also cannot get *shorter*,
    /// which a bare cable cannot do and a sheathed one can — that end pushes.
    /// Equal bounds give an inextensible path in both directions.
    Limit { min: f32, max: f32 },
    /// A constant pull along the path, whatever its length.
    ///
    /// The tension a hanging weight puts through a cable over pulleys, or the
    /// pretension a tendon-driven robot is assembled with. Unbounded in travel:
    /// it will happily reel the path down to nothing if nothing stops it.
    Force { tension: f32 },
    /// A damped spring on the length: an elastic cord, or a length actuator
    /// whose gain is `stiffness`.
    ///
    /// `stiffness` is force per unit of extension, in newtons per world unit,
    /// and is solved implicitly — see
    /// [`JointSpring`](crate::joint::JointSpring) for why that matters and why
    /// naming a stiffness is safe here when [`Softness`] refuses to.
    Spring {
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
    /// A winch: reel the path to `target` and hold it, within a force ceiling.
    ///
    /// The ceiling is what keeps it honest, for the same reason
    /// [`Servo`](crate::joint::Servo)'s is: a drive that cannot be beaten
    /// drags whatever it is pulling straight through everything else.
    Servo {
        target: f32,
        max_force: f32,
        /// Fastest it reels, in units/s. Zero is unlimited.
        max_speed: f32,
        /// How stiffly it holds. Rigid by default.
        softness: Softness,
    },
}

impl TendonKind {
    /// Resists stretching past `length`, goes slack below it.
    pub fn rope(length: f32) -> Self {
        Self::Limit {
            min: 0.0,
            max: length.max(0.0),
        }
    }

    /// Neither stretches nor goes slack — a belt.
    pub fn belt(length: f32) -> Self {
        let l = length.max(0.0);
        Self::Limit { min: l, max: l }
    }

    /// A winch with a rigid hold.
    pub fn winch(target: f32, max_force: f32) -> Self {
        Self::Servo {
            target: target.max(0.0),
            max_force: max_force.max(0.0),
            max_speed: 0.0,
            softness: Softness::RIGID,
        }
    }
}

/// One straight run of a resolved path.
///
/// A wrap contributes two of these — the approach and the departure — with the
/// arc between them accounted for in the length but not here: a frictionless
/// wrap pulls along its tangents and nowhere else, so the arc carries no
/// direction of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PathSegment {
    pub a_body: BodyId,
    pub a: Vector3,
    pub b_body: BodyId,
    pub b: Vector3,
    /// Strand divisor in force here.
    pub divisor: f32,
}

/// A tendon path worked out against the bodies as they stand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedPath {
    pub(crate) segments: Vec<PathSegment>,
    /// Where the path met each obstacle it touched, in path order.
    pub arcs: Vec<TendonArc>,
    /// Total length, strand divisors already applied.
    pub length: f32,
    /// A path point is inside an obstacle it was supposed to wrap, so that wrap
    /// was skipped. Surfaced as
    /// [`Warning::TendonInsideObstacle`](crate::diagnostics::Warning).
    pub degenerate: bool,
    /// An element could not be read as written and was skipped. Surfaced as
    /// [`Warning::MalformedTendonPath`](crate::diagnostics::Warning).
    pub malformed: bool,
    /// Every body on the path is still alive.
    pub valid: bool,
}

impl ResolvedPath {
    /// Add one straight run and its length.
    fn push(&mut self, a_body: BodyId, a: Vector3, b_body: BodyId, b: Vector3, divisor: f32) {
        self.length += (b - a).length() / divisor;
        self.segments.push(PathSegment {
            a_body,
            a,
            b_body,
            b,
            divisor,
        });
    }

    /// Whether the path resolved to anything the solver can constrain.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The straight runs, in path order.
    pub(crate) fn segments(&self) -> &[PathSegment] {
        &self.segments
    }
}

/// A cable, and what it does about its length.
#[derive(Debug, Clone, PartialEq)]
pub struct Tendon {
    /// The path, in order. Fewer than two via points constrains nothing.
    pub path: Vec<TendonNode>,
    pub kind: TendonKind,
    pub enabled: bool,
    /// Accumulated impulse, kept across substeps for warm starting.
    pub(crate) impulse: f32,
}

impl Tendon {
    /// A tendon along `route`, slack: it does nothing until given a
    /// [`TendonKind`].
    ///
    /// The route is *not* checked for length here — measure it with
    /// [`World::tendon_length`](crate::world::World::tendon_length) once the
    /// bodies are placed, which is the number a rope or a winch wants.
    pub fn new(route: Vec<TendonPoint>) -> Self {
        Self {
            path: route.into_iter().map(TendonNode::from).collect(),
            kind: TendonKind::Force { tension: 0.0 },
            enabled: true,
            impulse: 0.0,
        }
    }

    /// A rope of the given maximum length.
    pub fn rope(route: Vec<TendonPoint>, max_length: f32) -> Self {
        let mut t = Self::new(route);
        t.kind = TendonKind::rope(max_length);
        t
    }

    /// A cable pulled with constant tension.
    pub fn pulled(route: Vec<TendonPoint>, tension: f32) -> Self {
        let mut t = Self::new(route);
        t.kind = TendonKind::Force { tension };
        t
    }

    /// An elastic cord, or a length actuator of gain `stiffness`.
    pub fn sprung(route: Vec<TendonPoint>, rest_length: f32, stiffness: f32, damping: f32) -> Self {
        let mut t = Self::new(route);
        t.kind = TendonKind::Spring {
            rest_length: rest_length.max(0.0),
            stiffness: stiffness.max(0.0),
            damping: damping.max(0.0),
        };
        t
    }

    /// Whether this constrains anything at all.
    ///
    /// Obstacles and pulleys are not endpoints: a path has to have two points
    /// fixed to bodies before there is a length to hold.
    pub fn is_active(&self) -> bool {
        self.enabled
            && self
                .path
                .iter()
                .filter(|n| matches!(n, TendonNode::Via(_)))
                .count()
                >= 2
    }

    /// The bodies the path touches, in first-appearance order.
    pub fn bodies(&self) -> impl Iterator<Item = BodyId> + '_ {
        let mut seen: Vec<BodyId> = Vec::new();
        self.path
            .iter()
            .filter_map(move |n| {
                let body = n.body()?;
                if seen.contains(&body) {
                    None
                } else {
                    seen.push(body);
                    Some(body)
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Where the via points sit right now, in world space. Empty if a body is
    /// gone. Wraps are not in here — for the path as the solver sees it, with
    /// its arcs and strand divisors, resolve it against the bodies instead.
    pub fn world_route(&self, bodies: &BodySet) -> Vec<Vector3> {
        let mut out = Vec::with_capacity(self.path.len());
        for node in &self.path {
            let TendonNode::Via(p) = node else {
                continue;
            };
            let Some(body) = bodies.get(p.body) else {
                return Vec::new();
            };
            out.push(body.position.transform_point(p.local));
        }
        out
    }

    /// Work the path out against the bodies as they stand: straight runs, the
    /// arcs where it wraps, and the total length with strand divisors applied.
    ///
    /// This is what the solver constrains. A [`TendonNode::Pulley`] ends the
    /// strand before it, so nothing is drawn across the break — the strands of a
    /// reeving are separate pieces of rope that happen to share one coordinate.
    pub fn resolve(&self, bodies: &BodySet) -> ResolvedPath {
        let mut out = ResolvedPath {
            valid: true,
            ..Default::default()
        };

        // `prev` is where the last strand got to: a world point and the body it
        // is fixed to. `None` at the start of a strand.
        let mut prev: Option<(BodyId, Vector3)> = None;
        let mut pending: Option<(TendonObstacle, bool)> = None;
        let mut divisor = 1.0f32;

        for node in &self.path {
            match node {
                TendonNode::Pulley { divisor: d } => {
                    // A new strand: the previous one simply ends here.
                    divisor = if *d > 1e-9 { *d } else { 1.0 };
                    prev = None;
                    if pending.take().is_some() {
                        out.malformed = true;
                    }
                }
                TendonNode::Sphere(_) | TendonNode::Cylinder(_) => {
                    let Some((obstacle, is_cylinder)) = node.obstacle() else {
                        out.malformed = true;
                        continue;
                    };
                    if pending.replace((*obstacle, is_cylinder)).is_some() {
                        // Two obstacles with no via point between them: the
                        // second has nothing to be tangent from.
                        out.malformed = true;
                    }
                }
                TendonNode::Via(p) => {
                    let Some(body) = bodies.get(p.body) else {
                        out.valid = false;
                        return out;
                    };
                    let point = body.position.transform_point(p.local);
                    let Some((prev_body, prev_point)) = prev else {
                        // First via of a strand — nothing to join it to yet.
                        if pending.take().is_some() {
                            out.malformed = true;
                        }
                        prev = Some((p.body, point));
                        continue;
                    };

                    match pending.take() {
                        None => out.push(prev_body, prev_point, p.body, point, divisor),
                        Some((obstacle, is_cylinder)) => {
                            let Some(host) = bodies.get(obstacle.body) else {
                                out.valid = false;
                                return out;
                            };
                            match wrap_obstacle(
                                &obstacle,
                                is_cylinder,
                                &host.position,
                                prev_point,
                                point,
                            ) {
                                Ok(arc) => {
                                    out.push(
                                        prev_body,
                                        prev_point,
                                        obstacle.body,
                                        arc.from,
                                        divisor,
                                    );
                                    // The arc is length the straight runs cannot
                                    // see, and it moves with the obstacle's body
                                    // rather than adding a direction of its own.
                                    out.length += arc.length / divisor;
                                    out.push(obstacle.body, arc.to, p.body, point, divisor);
                                    out.arcs.push(arc);
                                }
                                Err(NoWrap::Clears) => {
                                    out.push(prev_body, prev_point, p.body, point, divisor)
                                }
                                Err(NoWrap::Inside) => {
                                    out.degenerate = true;
                                    out.push(prev_body, prev_point, p.body, point, divisor);
                                }
                            }
                        }
                    }
                    prev = Some((p.body, point));
                }
            }
        }

        if pending.is_some() {
            // An obstacle with no via point after it never gets to wrap.
            out.malformed = true;
        }
        out
    }

    /// The summed length of every segment, right now — wraps and strand
    /// divisors included, so this is the number the solver constrains.
    pub fn length(&self, bodies: &BodySet) -> f32 {
        self.resolve(bodies).length
    }

    /// The length the constraint is measured against — what the kind wants the
    /// path to be. `None` for the kinds that have no target.
    pub fn target_length(&self) -> Option<f32> {
        match self.kind {
            TendonKind::Limit { min, max } => Some(if min == max { min } else { max }),
            TendonKind::Spring { rest_length, .. } => Some(rest_length),
            TendonKind::Servo { target, .. } => Some(target),
            TendonKind::Force { .. } => None,
        }
    }

    /// Tension carried last step, in newtons — positive is pulling.
    ///
    /// Read after [`World::step`](crate::world::World::step); it is the
    /// impulse the constraint applied divided by the substep it applied over,
    /// which is the number a load cell spliced into the cable would show.
    pub fn tension(&self, dt: f32) -> f32 {
        if dt > 0.0 {
            self.impulse / dt
        } else {
            0.0
        }
    }

}

/// Handle to a tendon in a [`TendonSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TendonId {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

#[derive(Debug, Clone, Default)]
struct Slot {
    generation: u32,
    tendon: Option<Tendon>,
}

/// The world's tendons, with generation-checked handles.
#[derive(Debug, Clone, Default)]
pub struct TendonSet {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
}

impl TendonSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, tendon: Tendon) -> TendonId {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.tendon = Some(tendon);
            return TendonId {
                index,
                generation: slot.generation,
            };
        }
        self.slots.push(Slot {
            generation: 0,
            tendon: Some(tendon),
        });
        TendonId {
            index: self.slots.len() as u32 - 1,
            generation: 0,
        }
    }

    pub fn remove(&mut self, id: TendonId) -> Option<Tendon> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let t = slot.tendon.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
        self.len -= 1;
        Some(t)
    }

    pub fn get(&self, id: TendonId) -> Option<&Tendon> {
        let slot = self.slots.get(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.tendon.as_ref())?
    }

    pub fn get_mut(&mut self, id: TendonId) -> Option<&mut Tendon> {
        let slot = self.slots.get_mut(id.index as usize)?;
        (slot.generation == id.generation).then_some(slot.tendon.as_mut())?
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

    pub fn iter(&self) -> impl Iterator<Item = (TendonId, &Tendon)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.tendon.as_ref().map(|t| {
                (
                    TendonId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    t,
                )
            })
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (TendonId, &mut Tendon)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            let generation = s.generation;
            s.tendon.as_mut().map(|t| {
                (
                    TendonId {
                        index: i as u32,
                        generation,
                    },
                    t,
                )
            })
        })
    }

    /// Drop every tendon whose path touches `body`.
    ///
    /// A cable that has lost one of its guides has no meaningful path left —
    /// silently dropping the point instead would leave a tendon that quietly
    /// measures something else.
    pub fn remove_body(&mut self, body: BodyId) {
        for i in 0..self.slots.len() {
            let hit = self.slots[i]
                .tendon
                .as_ref()
                .is_some_and(|t| t.path.iter().any(|n| n.body() == Some(body)));
            if hit {
                self.slots[i].tendon = None;
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
        let mut bodies = BodySet::new();
        let a = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.1))
                .translation(Vector3::ZERO),
        );
        let b = bodies.insert(
            RigidBody::dynamic()
                .shape(Shape::ball(0.1))
                .translation(Vector3::new(3.0, 4.0, 0.0)),
        );
        (bodies, a, b)
    }

    #[test]
    fn length_is_the_sum_of_the_segments() {
        let (bodies, a, b) = two_bodies();
        let t = Tendon::new(vec![
            TendonPoint::new(a, Vector3::ZERO),
            TendonPoint::new(b, Vector3::ZERO),
        ]);
        assert!((t.length(&bodies) - 5.0).abs() < 1e-5, "3-4-5");
    }

    #[test]
    fn a_route_visiting_a_body_twice_lists_it_once() {
        let (_, a, b) = two_bodies();
        let t = Tendon::new(vec![
            TendonPoint::new(a, Vector3::ZERO),
            TendonPoint::new(b, Vector3::ZERO),
            TendonPoint::new(a, Vector3::new(0.0, 1.0, 0.0)),
        ]);
        assert_eq!(t.bodies().collect::<Vec<_>>(), vec![a, b]);
    }

    #[test]
    fn a_one_point_tendon_constrains_nothing() {
        let (_, a, _) = two_bodies();
        assert!(!Tendon::new(vec![TendonPoint::new(a, Vector3::ZERO)]).is_active());
        assert!(!Tendon::new(vec![]).is_active());
    }

    #[test]
    fn removing_a_body_removes_the_tendons_through_it() {
        let (_, a, b) = two_bodies();
        let mut set = TendonSet::new();
        set.insert(Tendon::new(vec![
            TendonPoint::new(a, Vector3::ZERO),
            TendonPoint::new(b, Vector3::ZERO),
        ]));
        assert_eq!(set.len(), 1);
        set.remove_body(b);
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn stale_handles_do_not_resolve() {
        let (_, a, b) = two_bodies();
        let mut set = TendonSet::new();
        let route = vec![TendonPoint::new(a, Vector3::ZERO), TendonPoint::new(b, Vector3::ZERO)];
        let id = set.insert(Tendon::new(route.clone()));
        set.remove(id);
        let id2 = set.insert(Tendon::new(route));
        assert!(set.get(id).is_none(), "stale handle resolved");
        assert!(set.get(id2).is_some());
    }
}
