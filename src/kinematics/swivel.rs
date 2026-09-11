//! Three-bearing swivel nozzles: a duct steered by rotating obliquely-cut joints.
//!
//! Cut a round duct with a plane that is not square to it and the join is an
//! ellipse. Put a bearing on that ellipse and turn it, and the duct downstream
//! sweeps a cone: at half a turn it has folded over by twice the cant angle,
//! and at a full turn it is straight again. Three of those in series is a
//! [`SwivelNozzle`] — the Three-Bearing Swivel Module that turns a STOVL
//! fighter's exhaust 95° downward without anything sliding, hinging or leaving
//! the gas path.
//!
//! ```
//! use threers::kinematics::swivel::SwivelNozzle;
//!
//! let nozzle = SwivelNozzle::three_bearing();
//! assert!((nozzle.envelope() - 95.0).abs() < 1e-9);
//!
//! // Stowed, the module is a straight pipe and the jet goes aft.
//! assert!(nozzle.deflection(&[0.0; 3]) < 1e-9);
//!
//! // The centre segment half a turn against both neighbours puts it at the
//! // corner of the envelope, and the roll bearing puts it in the vertical plane.
//! assert!((nozzle.deflection(&[-180.0, 180.0, 180.0]) - 95.0).abs() < 1e-6);
//!
//! // The whole deployment, closed form, exactly in plane.
//! let schedule = nozzle.roll_schedule(95.0, 400);
//! assert!(schedule.worst_lateral() < 1e-9);
//! ```
//!
//! # The arrangement matters more than the angles
//!
//! [`SwivelNozzle::three_bearing`] is the published one: a plain **roll bearing**
//! aft of the turbine, then two canted joints at ∓23.75°. Turning the centre
//! segment half a turn against both of its neighbours folds the duct by
//! `2(23.75 + 23.75)`, and four times 23.75 is 95 exactly.
//!
//! The roll bearing is what makes it a machine you can fly. It has no cant, so
//! it does nothing to the jet on its own; what it does is turn the whole folded
//! assembly about the engine axis, which is one-for-one with the jet's azimuth.
//! So the deployment schedule is two lines and no search
//! ([`SwivelNozzle::roll_schedule`]): sweep the two canted joints together, and
//! roll the front bearing by minus the azimuth that produces. The jet stays in
//! the vertical plane to floating point, not to a tolerance.
//!
//! # And the arrangement that does need a solver
//!
//! Cant all three instead — 11.875 / −23.75 / 11.875 also reaches 95° — and
//! every one of those properties goes away:
//!
//! - No bearing sets azimuth, so there is no closed form.
//! - Driving them on any fixed ratio takes the jet **68° out of the vertical
//!   plane** on the way down, which on an aircraft in transition is a yawing
//!   kick worth more than the thrust it is vectoring.
//! - The ratio that cancels the sideways motion is only correct to first order,
//!   and past a few degrees of deflection the mechanism is simply not linear.
//!
//! [`SwivelNozzle::searched_schedule`] is for that case: damped least squares
//! tracking a commanded deflection down the envelope, one small step at a time.
//! Held to it the same mechanism stays within a hundredth of a degree of the
//! plane — at the cost of a table, an open-loop break-out, and a few degrees of
//! yaw inside it. [`SwivelNozzle::schedule`] picks whichever applies.
//!
//! # The stowed position is singular either way
//!
//! Straight, every bearing axis lies in the pitch plane and the jet points along
//! the duct — so every column of the direction Jacobian is the cross product of
//! an in-plane axis with an in-plane vector, and all of them come out sideways.
//! On the published arrangement the roll bearing's column is not merely sideways
//! but exactly zero: a roll bearing cannot move a jet that is already on its
//! axis. The module has **no first-order pitch authority at all** when stowed.
//!
//! This is a property of the geometry rather than of the maths, and any swivel
//! nozzle that is a straight pipe in cruise has it. What differs is whether it
//! matters. With a roll bearing it does not — the two canted joints leave stowed
//! on a path whose azimuth is well defined the whole way, and the roll bearing
//! cancels it, so the closed form simply does not care. Without one,
//! [`SwivelNozzle::searched_schedule`] has to break out open-loop before it can
//! close the loop, and [`Schedule::worst_lateral`] reports what that costs.
//!
//! [`SwivelNozzle::authority`] is the number to watch: it is the second
//! singular value of the direction Jacobian, in jet-degrees per bearing-degree,
//! and it goes to zero at both ends of the envelope.

use super::chain::{cross, rot, scale, sub, RevoluteJoint, SerialChain};
use super::{dot, norm, unit, Goal, Ik, Solution, V3};

/// One obliquely-cut rotary joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bearing {
    /// Angle between the bearing axis and the duct, degrees. The sign is which
    /// way the cut leans: positive tips the axis from the duct toward the
    /// direction the jet deflects.
    ///
    /// Signs have to alternate along the chain for the bends to add. Three
    /// bearings deflect the jet by `2 * (c0 - c1 + c2)`, so equal signs
    /// subtract and a nozzle built that way barely moves.
    pub cant: f64,
    /// Where the joint plane crosses the duct centreline, measured along it
    /// from the first bearing.
    pub station: f64,
}

/// A duct steered by a chain of obliquely-cut bearings.
///
/// The frame is the engine's: **+X aft along the duct**, +Z up, and the jet
/// deflects toward −Z. Angles are degrees and lengths are metres, matching
/// [`SerialChain`] with `metres = 1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct SwivelNozzle {
    pub bearings: Vec<Bearing>,
    /// Where the exit plane crosses the centreline, along +X from bearing 0.
    pub exit: f64,
    /// Mass of the duct outboard of each bearing and inboard of the next — the
    /// part that bearing swings. The last entry is the nozzle itself.
    pub masses: Vec<f64>,
    /// Centre of mass of each of those, as an offset from its own bearing in
    /// the stowed frame. Half a segment length down the duct for a plain tube;
    /// a real one is off the centreline, because an obliquely-cut duct is not
    /// symmetric about it.
    pub com: Vec<V3>,
    /// The point moments are reported about — the airframe's centre of gravity,
    /// in the same frame. Defaults to the first bearing.
    pub reference: V3,
    /// Standoff used to turn a direction command into a point goal for the IK.
    /// Only the units of the residual depend on it; one metre keeps `lambda`
    /// and `tol` in metres of miss at a metre.
    pub aim: f64,
    /// How far to drive the bearings together, in bearing-degrees, before
    /// closing the loop. See the module docs on the stowed singularity.
    pub breakout: f64,
}

impl SwivelNozzle {
    /// The three-bearing module of a STOVL fighter, in metres.
    ///
    /// 95° of deflection over a 1.68 m module on a 1 m duct, in the arrangement
    /// the published description gives: a plain roll bearing aft of the turbine,
    /// then two canted joints at ∓23.75°.
    ///
    /// The split is not a guess. Lockheed describe three duct segments "cut on an
    /// angle and joined by two airtight circular bearings", with "the forward and
    /// aft segments maintain\[ing\] alignment with each other" while "the center
    /// segment rotates 180 degrees relative to them", and a third bearing "aft of
    /// the turbine stage" through which the nozzle "provides yaw control". Two
    /// mitre joints at 23.75° with the middle segment at half a turn against both
    /// fold the duct by `2(23.75 + 23.75)` — and 4 × 23.75 is 95 exactly.
    ///
    /// That first bearing is what makes this mechanism tractable. It has no cant,
    /// so it does nothing to the jet on its own; what it does is roll the whole
    /// folded assembly about the engine axis, which is one-for-one with the jet's
    /// azimuth. So the schedule is closed form — see [`Self::roll_schedule`] —
    /// where a chain with three canted bearings needs [`Self::schedule`] and a
    /// search.
    ///
    /// The masses and centres of mass are what
    /// `crates/threers-physics/examples/three_bearing_swivel.scad` comes to,
    /// read off its meshes, so the analytic chain and the simulated one weigh
    /// the same and hang the same way. Override them with [`Self::masses`] and
    /// [`Self::com_offsets`] when the geometry is not that.
    pub fn three_bearing() -> Self {
        Self {
            bearings: vec![
                Bearing { cant: 0.0, station: 0.000 },
                Bearing { cant: -23.750, station: 0.560 },
                Bearing { cant: 23.750, station: 1.280 },
            ],
            exit: 1.680,
            // Each figure is a whole link of the chain, not just its duct: the
            // ring gear bolted to it, and the drive unit and pinion for the next
            // bearing down, which ride on it.
            masses: vec![463.5, 463.3, 306.7],
            // Well off the centreline, in both directions and for two different
            // reasons. An obliquely-cut duct has more metal on the long side of
            // the cut than the short one, and the two ends of a segment lean
            // opposite ways — that is the ±105 mm in z. The 93 mm in y is the
            // drive train, which is all on one side because the duct folds in
            // the other plane and there is nowhere else to put it.
            com: vec![
                [0.269, 0.093, -0.107],
                [0.246, 0.093, 0.105],
                [0.106, 0.002, -0.045],
            ],
            reference: [0.0, 0.0, 0.0],
            aim: 1.0,
            breakout: 30.0,
        }
    }

    /// Replace the segment masses, in kilograms, outboard-of-each-bearing order.
    ///
    /// The point of this is to take them from the geometry rather than from a
    /// guess: the simulation derives each part's mass from the mesh it draws,
    /// and feeding those back means the torques here are the torques there.
    pub fn masses(mut self, masses: impl Into<Vec<f64>>) -> Self {
        self.masses = masses.into();
        self
    }

    /// Replace the centre-of-mass offsets, each measured from its own bearing.
    ///
    /// Same reason as [`Self::masses`]: an obliquely-cut duct's centre of mass
    /// is not on the centreline, and taking it from the mesh rather than from
    /// the middle of the segment is the difference between a holding torque
    /// that matches the simulation and one that is a few per cent out.
    pub fn com_offsets(mut self, com: impl Into<Vec<V3>>) -> Self {
        self.com = com.into();
        self
    }

    /// Bearings from cant/station pairs, with the exit plane at `exit`.
    ///
    /// For building the kinematic model out of what a `.scad` model declared,
    /// rather than restating it: the cants come back out of the hinge axes and
    /// the stations out of the hinge points.
    pub fn from_bearings(bearings: impl Into<Vec<Bearing>>, exit: f64) -> Self {
        let bearings = bearings.into();
        let n = bearings.len();
        let mut nozzle = Self {
            bearings,
            exit,
            masses: vec![0.0; n],
            com: vec![[0.0; 3]; n],
            ..Self::three_bearing()
        };
        let segments = nozzle.segments();
        nozzle.com = segments.iter().map(|l| [l * 0.5, 0.0, 0.0]).collect();
        nozzle
    }

    /// Report moments about this point rather than the first bearing.
    pub fn reference(mut self, reference: V3) -> Self {
        self.reference = reference;
        self
    }

    pub fn n(&self) -> usize {
        self.bearings.len()
    }

    /// The bearing axes with every joint at zero, in the engine frame.
    ///
    /// Also each axis in its own joint's local frame, which is the same thing:
    /// stowed, every frame in the chain coincides with the engine's.
    pub fn axes(&self) -> Vec<V3> {
        self.bearings
            .iter()
            .map(|b| {
                let c = b.cant.to_radians();
                [c.cos(), 0.0, -c.sin()]
            })
            .collect()
    }

    /// Length of the duct each bearing swings, up to the next one or the exit.
    pub fn segments(&self) -> Vec<f64> {
        let n = self.n();
        (0..n)
            .map(|i| {
                let next = if i + 1 < n {
                    self.bearings[i + 1].station
                } else {
                    self.exit
                };
                next - self.bearings[i].station
            })
            .collect()
    }

    /// Largest deflection the chain can reach, degrees.
    ///
    /// Every bearing at half a turn, where the mitre bends line up and add:
    /// `2 * (c0 - c1 + c2 - …)`. Reached exactly, and only there.
    pub fn envelope(&self) -> f64 {
        2.0 * self
            .bearings
            .iter()
            .enumerate()
            .map(|(i, b)| if i % 2 == 0 { b.cant } else { -b.cant })
            .sum::<f64>()
    }

    /// The chain, for kinematics and for the torques gravity puts on each
    /// bearing.
    ///
    /// Each joint's link runs down the duct centreline to the next joint, so
    /// [`SerialChain::tool`] returns the exit centre and the direction the jet
    /// leaves in — the two things this whole mechanism exists to place.
    ///
    /// The dynamics it carries treat a segment as a thin rod along its own
    /// centreline, which is exact for gravity (that depends only on where the
    /// mass is, not how it is spread) and understates the polar inertia of a
    /// duct spun about a nearly-axial bearing. Use it for the static and
    /// quasi-static side; the rigid-body solver in `threers-physics` builds
    /// real inertia tensors from the same geometry for the rest.
    pub fn chain(&self) -> SerialChain {
        let axes = self.axes();
        let segments = self.segments();
        let joints = (0..self.n())
            .map(|i| {
                let len = segments[i];
                RevoluteJoint {
                    axis: axes[i],
                    link: [len, 0.0, 0.0],
                    mass: self.masses.get(i).copied().unwrap_or(0.0),
                    com: self.com.get(i).copied().unwrap_or([len * 0.5, 0.0, 0.0]),
                    limits: (-180.0, 180.0),
                }
            })
            .collect();
        SerialChain {
            joints,
            gravity: [0.0, 0.0, -9.81],
            metres: 1.0,
        }
    }

    /// Where the exit centre sits and which way the jet leaves, for bearing
    /// angles `q` in degrees.
    pub fn exhaust(&self, q: &[f64]) -> (V3, V3) {
        self.chain().tool(q)
    }

    /// Angle between the jet and the engine axis, degrees. 0 stowed, up to
    /// [`Self::envelope`].
    pub fn deflection(&self, q: &[f64]) -> f64 {
        let (_, e) = self.exhaust(q);
        unit(e)[0].clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Which way round the engine axis the jet has swung, degrees. 0 is
    /// straight down, ±90 is sideways. Undefined, and reported as 0, when there
    /// is no deflection to have an azimuth.
    pub fn azimuth(&self, q: &[f64]) -> f64 {
        let e = unit(self.exhaust(q).1);
        if e[1].abs() < 1e-12 && e[2].abs() < 1e-12 {
            return 0.0;
        }
        e[1].atan2(-e[2]).to_degrees()
    }

    /// How far out of the vertical plane the jet is, degrees. This is the
    /// number a transition has to keep small — it is a yawing kick, and the
    /// aircraft has nothing but rudder and roll posts to answer it with.
    pub fn lateral(&self, q: &[f64]) -> f64 {
        unit(self.exhaust(q).1)[1].clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// Unit jet direction for a deflection and azimuth, both degrees.
    pub fn direction(deflection: f64, azimuth: f64) -> V3 {
        let (d, a) = (deflection.to_radians(), azimuth.to_radians());
        [d.cos(), d.sin() * a.sin(), -d.sin() * a.cos()]
    }

    /// How the jet direction moves with each bearing: columns are
    /// `axis_i × direction`, in per-radian units.
    ///
    /// Rank two at best — a unit vector has only two ways to move — so the
    /// smallest singular value is always zero and the *second* one is what says
    /// whether the mechanism can steer. See [`Self::authority`].
    pub fn direction_jacobian(&self, q: &[f64]) -> [[f64; 3]; 3] {
        let chain = self.chain();
        let poses = chain.poses(q);
        let e = unit(chain.tool(q).1);
        let mut j = [[0.0; 3]; 3];
        for (c, pose) in poses.iter().enumerate().take(3) {
            let col = cross(pose.axis, e);
            for r in 0..3 {
                j[r][c] = col[r];
            }
        }
        j
    }

    /// Steering authority: the second singular value of
    /// [`Self::direction_jacobian`], in jet-radians per bearing-radian.
    ///
    /// One is a mechanism whose jet moves as fast as its bearings. Zero is one
    /// that cannot steer at all in some direction, which happens twice: stowed,
    /// where every axis is coplanar with the jet, and at the far corner of the
    /// envelope, where the three bends have run out. Both ends of a deployment
    /// are singular and the middle is not.
    pub fn authority(&self, q: &[f64]) -> f64 {
        let j = self.direction_jacobian(q);
        // JJᵀ, symmetric 3×3; its eigenvalues are the squared singular values.
        let mut a = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] = (0..3).map(|k| j[r][k] * j[c][k]).sum();
            }
        }
        sym3_eigenvalues(a)[1].max(0.0).sqrt()
    }

    /// The solver, tuned for a direction goal at [`Self::aim`] metres.
    ///
    /// `lambda` is the damping that biases the answer toward the smallest
    /// bearing move, which is what keeps a tracked schedule on one branch
    /// instead of jumping between the several bearing triples that all put the
    /// jet in the same place.
    pub fn solver(&self) -> Ik {
        Ik {
            lambda: 0.02 * self.aim,
            max_step: 4.0,
            tol: 1e-6 * self.aim,
            max_iters: 200,
            limits: (-180.0, 180.0),
            ..Ik::default()
        }
    }

    /// Bearing angles that point the jet at `deflection` / `azimuth`, starting
    /// the search from `seed`.
    ///
    /// The goal is a direction and the chain has three bearings, so one degree
    /// of freedom is left over; damped least squares spends it on the smallest
    /// move from the seed. Seed each solve with the last answer and the whole
    /// deployment stays on one branch — which is the point, because the branches
    /// are far apart and switching between them mid-transition would mean
    /// slewing a bearing a hundred degrees to hold the jet still.
    pub fn solve(&self, seed: &[f64], deflection: f64, azimuth: f64) -> Solution {
        let want = Self::direction(deflection, azimuth);
        let chain = self.chain();
        let aim = self.aim;
        let mut sol = self.solver().solve(
            seed,
            &Goal::Point(scale(want, aim)),
            |q| {
                let e = unit(chain.tool(q).1);
                (scale(e, aim), e)
            },
        );
        // Report the miss as an angle, which is what the caller commanded in.
        // The chord at `aim` is what the solver minimised; it is not the number
        // anyone wants to read.
        sol.axis_err = 2.0 * (sol.position_err / (2.0 * aim)).clamp(0.0, 1.0).asin().to_degrees();
        sol
    }

    /// Whether the first bearing is a plain roll bearing.
    ///
    /// No cant means it does nothing to the jet by itself: it turns the whole
    /// assembly downstream of it about the duct axis, which when the duct is
    /// straight is a rotation that maps the duct to itself and when it is folded
    /// is exactly the jet's azimuth. That is what makes the schedule closed form.
    pub fn has_roll_bearing(&self) -> bool {
        self.bearings.first().is_some_and(|b| b.cant.abs() < 1e-9)
    }

    /// The schedule for a nozzle with a roll bearing in front — exactly, with no
    /// search anywhere in it.
    ///
    /// Turn the two canted bearings together and the jet sweeps out to the corner
    /// of the envelope on a path that leaves the vertical plane. Roll the front
    /// bearing by minus that azimuth and it is back in the plane, because rolling
    /// the assembly *is* the azimuth. Two lines, and the residual is floating
    /// point rather than a tolerance.
    ///
    /// [`Self::schedule`] solves the same problem for a chain that has no such
    /// bearing, and on this one the two agree — which is the useful check, since
    /// they have nothing in common but the mechanism.
    pub fn roll_schedule(&self, to: f64, steps: usize) -> Schedule {
        let steps = steps.max(1);
        let mut points = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = 180.0 * i as f64 / steps as f64;
            let mut q = vec![t; self.n()];
            q[0] = 0.0;
            // Where the fold put the jet, and the roll that undoes it.
            q[0] = -self.azimuth(&q);
            let deflection = self.deflection(&q);
            points.push(self.sample(&q, 0.0));
            if deflection >= to {
                break;
            }
        }
        Schedule {
            points,
            breakout: 0.0,
        }
    }

    /// The bearing schedule for a deployment, from stowed to `to` degrees of
    /// deflection in the vertical plane.
    ///
    /// This is the table a nozzle controller carries. On a nozzle with a roll
    /// bearing in front it is [`Self::roll_schedule`], which is exact; on one
    /// without, it is [`Self::searched_schedule`], which is not and cannot be.
    /// Callers who want the table do not have to know which.
    pub fn schedule(&self, to: f64, steps: usize) -> Schedule {
        if self.has_roll_bearing() {
            self.roll_schedule(to, steps)
        } else {
            self.searched_schedule(to, steps)
        }
    }

    /// The schedule for a chain with no roll bearing, solved rather than derived.
    ///
    /// Every bearing is canted, so none of them sets azimuth on its own and there
    /// is no closed form — see the module docs. Damped least squares tracks a
    /// commanded deflection down the envelope one small step at a time, after an
    /// open-loop break-out to get off the stowed singularity.
    ///
    /// `steps` is how many closed-loop points to solve after break-out. More is
    /// a smoother table and a slower build; the solve is seeded from its
    /// predecessor either way, so cost is linear.
    pub fn searched_schedule(&self, to: f64, steps: usize) -> Schedule {
        let n = self.n();
        let steps = steps.max(1);
        let mut points = Vec::with_capacity(steps + 17);

        // Break-out: all bearings together, open loop, off the singularity.
        // Sixteen points is enough to resolve a fraction of a degree of jet.
        const BREAKOUT_POINTS: usize = 16;
        let mut q = vec![0.0; n];
        for i in 0..=BREAKOUT_POINTS {
            let t = self.breakout * i as f64 / BREAKOUT_POINTS as f64;
            q = vec![t; n];
            points.push(self.sample(&q, 0.0));
        }
        let start = points.last().map(|p| p.deflection).unwrap_or(0.0);

        // Closed loop from there to the commanded deflection.
        let to = to.min(self.envelope());
        if to > start {
            for i in 1..=steps {
                let want = start + (to - start) * i as f64 / steps as f64;
                let sol = self.solve(&q, want, 0.0);
                q = sol.joints;
                points.push(self.sample(&q, sol.axis_err));
            }
        }

        Schedule { points, breakout: start }
    }

    fn sample(&self, q: &[f64], residual: f64) -> SchedulePoint {
        SchedulePoint {
            deflection: self.deflection(q),
            lateral: self.lateral(q),
            bearings: q.to_vec(),
            authority: self.authority(q),
            residual,
        }
    }

    /// What the jet does to the airframe.
    ///
    /// `thrust` is the gross thrust in newtons, along the jet. The force
    /// reported is the reaction on the aircraft, which is the opposite way.
    pub fn thrust(&self, q: &[f64], thrust: f64) -> ThrustState {
        let (exit, dir) = self.exhaust(q);
        let dir = unit(dir);
        // The jet leaves along `dir`; the airframe is pushed the other way.
        let force = scale(dir, -thrust);
        let arm = sub(exit, self.reference);
        ThrustState {
            direction: dir,
            exit,
            deflection: dir[0].clamp(-1.0, 1.0).acos().to_degrees(),
            lateral: dir[1].clamp(-1.0, 1.0).asin().to_degrees(),
            force,
            moment: cross(arm, force),
            thrust,
        }
    }

    /// Torque each bearing's actuator has to hold, in N·m, for a static pose.
    ///
    /// Gravity only: the module's own weight, hanging off the back of the
    /// engine. This is exact — a static moment depends on where the mass is and
    /// not on how it is spread — and it is the number that sizes the actuators,
    /// because a nozzle spends far longer holding a deflection than reaching it.
    pub fn holding_torque(&self, q: &[f64]) -> Vec<f64> {
        let n = self.n();
        self.chain()
            .inverse_dynamics(q, &vec![0.0; n], &vec![0.0; n], true)
    }

    /// Torque each bearing needs to hold the pose *and* slew at `rate`
    /// degrees a second with `accel` degrees a second squared.
    ///
    /// Thin-rod inertia — see [`Self::chain`] — so read the inertial part as an
    /// estimate and the gravity part as exact.
    pub fn slew_torque(&self, q: &[f64], rate: &[f64], accel: &[f64]) -> Vec<f64> {
        let rad: Vec<f64> = rate.iter().map(|v| v.to_radians()).collect();
        let acc: Vec<f64> = accel.iter().map(|v| v.to_radians()).collect();
        self.chain().inverse_dynamics(q, &rad, &acc, true)
    }
}

/// One row of a deployment schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulePoint {
    /// Where the jet actually ended up, degrees off the engine axis.
    pub deflection: f64,
    /// And how far out of the vertical plane, degrees. Zero everywhere the loop
    /// is closed; a degree or two through break-out.
    pub lateral: f64,
    /// The bearing angles that do it, degrees.
    pub bearings: Vec<f64>,
    /// [`SwivelNozzle::authority`] here.
    pub authority: f64,
    /// How far the solve missed by, degrees. Zero through break-out, which is
    /// open loop and commands nothing.
    pub residual: f64,
}

/// A deployment, as the table a controller would carry.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    /// In order of increasing deflection.
    pub points: Vec<SchedulePoint>,
    /// Deflection at which the loop closed — below this the schedule is the
    /// open-loop break-out and the jet is not held in plane.
    pub breakout: f64,
}

impl Schedule {
    /// Bearing angles for a commanded deflection, interpolated between rows.
    ///
    /// Clamped at both ends: below the first row it holds stowed, above the last
    /// it holds the corner of the envelope. A controller asking for more than
    /// the nozzle has gets the most it has, which is what the mechanism would
    /// do anyway.
    ///
    /// Rows are keyed by the deepest deflection reached so far rather than by
    /// their own. The two are the same everywhere except inside break-out,
    /// where the jet backs up by a few hundredths of a degree — see
    /// [`Self::breakout_reversal`] — and a table indexed by a coordinate that
    /// briefly runs backwards would return the wrong row for it.
    pub fn bearings_at(&self, deflection: f64) -> Vec<f64> {
        if self.points.is_empty() {
            return Vec::new();
        }
        let first = &self.points[0];
        if deflection <= first.deflection {
            return first.bearings.clone();
        }
        let mut key = first.deflection;
        for w in self.points.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let next = b.deflection.max(key);
            if deflection <= next {
                let span = next - key;
                let t = if span > 1e-12 {
                    (deflection - key) / span
                } else {
                    0.0
                };
                return a
                    .bearings
                    .iter()
                    .zip(&b.bearings)
                    .map(|(x, y)| x + (y - x) * t)
                    .collect();
            }
            key = next;
        }
        self.points.last().unwrap().bearings.clone()
    }

    /// Cumulative bearing travel to each row, degrees, starting at zero.
    ///
    /// The distance an actuator has actually moved, measured as the largest
    /// single-bearing move — which is the one that sets how long the deployment
    /// takes, since they all run at once.
    pub fn travel(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.points.len());
        let mut total = 0.0;
        out.push(0.0);
        for w in self.points.windows(2) {
            total += w[0]
                .bearings
                .iter()
                .zip(&w[1].bearings)
                .fold(0.0f64, |m, (a, b)| m.max((b - a).abs()));
            out.push(total);
        }
        out
    }

    /// Bearing angles a fraction `u` of the way through the deployment,
    /// measured in bearing travel rather than in jet angle.
    ///
    /// This is the one to drive a mechanism with. Interpolating on deflection
    /// crams break-out — a third of the total bearing travel — into the first
    /// degree of jet, and asks the actuators to cover it in whatever slice of
    /// time that degree gets. Interpolating on travel gives them a constant
    /// rate, which is what an actuator has.
    pub fn bearings_at_travel(&self, u: f64) -> Vec<f64> {
        if self.points.is_empty() {
            return Vec::new();
        }
        let travel = self.travel();
        let total = travel.last().copied().unwrap_or(0.0);
        if total <= 0.0 {
            return self.points[0].bearings.clone();
        }
        let want = (u.clamp(0.0, 1.0)) * total;
        for (i, w) in travel.windows(2).enumerate() {
            if want <= w[1] {
                let span = w[1] - w[0];
                let t = if span > 1e-12 { (want - w[0]) / span } else { 0.0 };
                return self.points[i]
                    .bearings
                    .iter()
                    .zip(&self.points[i + 1].bearings)
                    .map(|(a, b)| a + (b - a) * t)
                    .collect();
            }
        }
        self.points.last().unwrap().bearings.clone()
    }

    /// The furthest the jet ever backs up during break-out, degrees.
    ///
    /// Break-out drives the bearings together at the ratio that cancels the
    /// sideways motion, and that ratio also very nearly cancels the *downward*
    /// motion — the deflection leaving the stowed position is third order in
    /// bearing angle, not second. So the jet wanders by a few hundredths of a
    /// degree before it commits. It is far too small to fly, and it is the
    /// reason the schedule is keyed the way it is rather than by raw deflection.
    pub fn breakout_reversal(&self) -> f64 {
        let mut peak = f64::NEG_INFINITY;
        let mut worst = 0.0f64;
        for p in &self.points {
            peak = peak.max(p.deflection);
            worst = worst.max(peak - p.deflection);
        }
        worst
    }

    /// The most deflection this schedule reaches, degrees.
    pub fn reach(&self) -> f64 {
        self.points.last().map(|p| p.deflection).unwrap_or(0.0)
    }

    /// The worst the jet ever leaves the vertical plane, degrees.
    ///
    /// All of it is in break-out. Past that the loop is closed and this is a
    /// solver tolerance rather than a mechanism property.
    pub fn worst_lateral(&self) -> f64 {
        self.points.iter().fold(0.0f64, |m, p| m.max(p.lateral.abs()))
    }

    /// The worst any closed-loop solve missed its command by, degrees.
    pub fn worst_residual(&self) -> f64 {
        self.points.iter().fold(0.0f64, |m, p| m.max(p.residual))
    }

    /// The least steering authority anywhere on the schedule, and where.
    pub fn worst_authority(&self) -> (f64, f64) {
        self.points
            .iter()
            .fold((f64::INFINITY, 0.0), |(a, d), p| {
                if p.authority < a {
                    (p.authority, p.deflection)
                } else {
                    (a, d)
                }
            })
    }

    /// Whether the closed-loop part of the table rises monotonically.
    ///
    /// This is the check that can really fail: damped least squares returns the
    /// smallest move from its seed, but a chain with a spare degree of freedom
    /// has several branches that put the jet in the same place, and a solve that
    /// jumps between them shows up here as a deflection that goes backwards
    /// while the bearings swing a hundred degrees.
    ///
    /// Break-out is excluded — it is open loop, and its own reversal is
    /// [`Self::breakout_reversal`].
    pub fn is_monotonic(&self) -> bool {
        self.points
            .iter()
            .filter(|p| p.deflection >= self.breakout - 1e-9)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[1].deflection >= w[0].deflection - 1e-9)
    }

    /// Largest bearing move between consecutive rows, degrees — how fast the
    /// actuators have to work at the tightest point in the table.
    pub fn worst_step(&self) -> f64 {
        self.points.windows(2).fold(0.0f64, |m, w| {
            w[0].bearings
                .iter()
                .zip(&w[1].bearings)
                .fold(m, |m, (a, b)| m.max((b - a).abs()))
        })
    }
}

/// What a vectored jet does to the aircraft carrying it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrustState {
    /// Unit vector the jet leaves along.
    pub direction: V3,
    /// Where it leaves from.
    pub exit: V3,
    /// Angle off the engine axis, degrees.
    pub deflection: f64,
    /// Out-of-plane angle, degrees.
    pub lateral: f64,
    /// Reaction on the airframe, newtons, in the engine frame.
    pub force: V3,
    /// And the moment it makes about [`SwivelNozzle::reference`], N·m.
    pub moment: V3,
    /// Gross thrust it was computed from, newtons.
    pub thrust: f64,
}

impl ThrustState {
    /// Upward force, newtons. This is what holds the aircraft up in the hover.
    pub fn lift(&self) -> f64 {
        self.force[2]
    }

    /// Forward force, newtons — positive accelerates. Goes negative past 90° of
    /// deflection, where the jet has swung far enough forward to brake with.
    pub fn axial(&self) -> f64 {
        -self.force[0]
    }

    /// Sideways force, newtons. Zero on a schedule that holds the plane.
    pub fn side(&self) -> f64 {
        self.force[1]
    }

    /// Pitching moment about the reference point, N·m, positive nose-up.
    ///
    /// Vectored lift from a nozzle behind the centre of gravity is nose-*down*,
    /// so this goes negative as the jet swings under the aircraft — and it is a
    /// large number, hundreds of kN·m. Something else has to answer it, which
    /// on a real STOVL aircraft is the lift fan out in front.
    pub fn pitch(&self) -> f64 {
        self.moment[1]
    }

    /// Fraction of gross thrust turned into lift, 0 to 1.
    pub fn lift_fraction(&self) -> f64 {
        if self.thrust.abs() < 1e-12 {
            0.0
        } else {
            self.lift() / self.thrust
        }
    }
}

/// Eigenvalues of a symmetric 3×3, descending. Closed form — no iteration, and
/// no dependency for the one place this crate needs singular values.
fn sym3_eigenvalues(a: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    if p1 < 1e-30 {
        let mut d = [a[0][0], a[1][1], a[2][2]];
        d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return d;
    }
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let mut b = a;
    for (i, row) in b.iter_mut().enumerate() {
        row[i] -= q;
        for v in row.iter_mut() {
            *v /= p;
        }
    }
    let det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}

/// The rotation a bearing at `cant` makes when turned by `angle`, both degrees.
/// Exposed because it is the whole mechanism in one line, and because a test
/// that checks the chain against it is checking something independent.
pub fn bearing_rotation(cant: f64, angle: f64) -> [[f64; 3]; 3] {
    let c = cant.to_radians();
    rot([c.cos(), 0.0, -c.sin()], angle)
}

/// The cant a bearing axis implies, degrees, in the engine frame.
///
/// The inverse of what [`SwivelNozzle::axes`] builds: an axis of
/// `[cos c, 0, -sin c]` reads back as `c`. Any component along +Y is out of the
/// deflection plane and is ignored here, so a model that declares one gets a
/// cant that describes only the part of it this reads.
pub fn cant_of(axis: V3) -> f64 {
    let a = unit(axis);
    (-a[2]).atan2(a[0]).to_degrees()
}

/// Angle between two directions, degrees.
pub fn angle_between(a: V3, b: V3) -> f64 {
    dot(unit(a), unit(b)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Length of a vector. Re-exported so callers reading [`ThrustState::force`]
/// need not reach for the private helper.
pub fn magnitude(v: V3) -> f64 {
    norm(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nozzle() -> SwivelNozzle {
        SwivelNozzle::three_bearing()
    }

    #[test]
    fn stowed_is_a_straight_pipe() {
        let n = nozzle();
        let (exit, dir) = n.exhaust(&[0.0; 3]);
        assert!((exit[0] - n.exit).abs() < 1e-12, "exit at {exit:?}");
        assert!(exit[1].abs() < 1e-12 && exit[2].abs() < 1e-12, "exit at {exit:?}");
        assert!(angle_between(dir, [1.0, 0.0, 0.0]) < 1e-9);
        assert!(n.deflection(&[0.0; 3]) < 1e-9);
    }

    #[test]
    fn half_a_turn_on_every_bearing_reaches_the_envelope() {
        let n = nozzle();
        assert!((n.envelope() - 95.0).abs() < 1e-9, "envelope {}", n.envelope());
        let q = [180.0; 3];
        assert!((n.deflection(&q) - 95.0).abs() < 1e-6, "reached {}", n.deflection(&q));
        // And it lands exactly in the vertical plane, pointing down.
        assert!(n.lateral(&q).abs() < 1e-6);
        assert!(n.exhaust(&q).1[2] < 0.0, "it deflected upward");
    }

    #[test]
    fn nothing_in_the_envelope_exceeds_the_envelope() {
        let n = nozzle();
        let mut worst: f64 = 0.0;
        let mut q = [0.0f64; 3];
        for a in (0..360).step_by(17) {
            for b in (0..360).step_by(19) {
                for c in (0..360).step_by(23) {
                    q = [a as f64, b as f64, c as f64];
                    worst = worst.max(n.deflection(&q));
                }
            }
        }
        let _ = q;
        assert!(worst <= n.envelope() + 1e-6, "found {worst} past {}", n.envelope());
        assert!(worst > n.envelope() - 3.0, "the sweep never got near it: {worst}");
    }

    #[test]
    fn signs_have_to_alternate_for_the_bends_to_add() {
        let mut same = nozzle();
        for b in &mut same.bearings {
            b.cant = b.cant.abs();
        }
        assert!(same.envelope().abs() < 1e-9, "equal signs cancel: {}", same.envelope());
        assert!(same.deflection(&[180.0; 3]) < 1e-6);
    }

    #[test]
    fn the_stowed_pose_has_no_first_order_pitch_authority() {
        let n = nozzle();
        let j = n.direction_jacobian(&[0.0; 3]);
        // Nothing can pitch the jet from stowed, and for two different reasons.
        // The roll bearing is on the jet's own axis, so it does nothing at all;
        // the canted pair are coplanar with it, so all they can do is yaw.
        for (c, (axial, pitch)) in j[0].iter().zip(j[2].iter()).enumerate() {
            assert!(axial.abs() < 1e-12, "column {c} has an axial part");
            assert!(pitch.abs() < 1e-12, "column {c} can pitch: {pitch}");
        }
        assert!(
            (0..3).all(|r| j[r][0].abs() < 1e-12),
            "a roll bearing should move a jet on its own axis nowhere: {:?}",
            [j[0][0], j[1][0], j[2][0]]
        );
        for (c, yaw) in j[1].iter().enumerate().skip(1) {
            assert!(yaw.abs() > 1e-3, "canted column {c} does nothing either");
        }
        assert!(n.authority(&[0.0; 3]) < 1e-9, "authority {}", n.authority(&[0.0; 3]));

        // And it does not matter, which is the point of the roll bearing. The
        // schedule leaves stowed on the two canted joints, whose azimuth is
        // well defined the whole way, and the roll bearing cancels it.
        let s = n.roll_schedule(95.0, 400);
        let q = s.bearings_at(45.0);
        assert!(n.authority(&q) > 0.2, "authority at 45 deg: {}", n.authority(&q));
    }

    #[test]
    fn the_roll_schedule_is_exact() {
        let n = nozzle();
        assert!(n.has_roll_bearing());
        let s = n.roll_schedule(95.0, 400);
        assert!(s.is_monotonic(), "the table doubles back");
        assert!((s.reach() - 95.0).abs() < 1e-6, "reached {}", s.reach());
        assert_eq!(s.breakout, 0.0, "a closed form needs no break-out");
        // Not "within tolerance" — exactly in the plane, to floating point.
        assert!(
            s.worst_lateral() < 1e-9,
            "worst lateral {:e}",
            s.worst_lateral()
        );
        // It ends with the centre segment half a turn against both neighbours,
        // which is how the published description puts it.
        let end = &s.points.last().unwrap().bearings;
        assert!((end[1].abs() - 180.0).abs() < 1e-6, "{end:?}");
        assert!((end[2].abs() - 180.0).abs() < 1e-6, "{end:?}");
        assert!((end[1] - end[2]).abs() < 1e-6, "the two canted joints move together");
    }

    #[test]
    fn the_search_and_the_closed_form_agree() {
        // Two answers with nothing in common but the mechanism: one is a
        // rearrangement of the geometry, the other is damped least squares over
        // a residual. Where both are defined they have to be the same nozzle.
        let n = nozzle();
        let exact = n.roll_schedule(95.0, 400);
        for want in [10.0, 25.0, 45.0, 65.0, 85.0] {
            let seed = exact.bearings_at(want);
            let sol = n.solve(&seed, want, 0.0);
            assert!(sol.axis_err < 1e-3, "the search missed by {}", sol.axis_err);
            // The closed form is already a solution, so a solver biased toward
            // the smallest move from its seed should barely move at all.
            let moved = seed
                .iter()
                .zip(&sol.joints)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(moved < 0.5, "at {want} deg the search moved {moved} deg");
        }
    }

    #[test]
    fn the_direction_jacobian_matches_finite_differences() {
        let n = nozzle();
        for q in [[20.0, 35.0, 60.0], [120.0, -40.0, 150.0], [-70.0, 90.0, 10.0]] {
            let j = n.direction_jacobian(&q);
            let base = unit(n.exhaust(&q).1);
            const H: f64 = 1e-6; // radians
            for c in 0..3 {
                let mut probe = q;
                probe[c] += H.to_degrees();
                let moved = unit(n.exhaust(&probe).1);
                for r in 0..3 {
                    let fd = (moved[r] - base[r]) / H;
                    assert!(
                        (fd - j[r][c]).abs() < 1e-5,
                        "q={q:?} row {r} col {c}: analytic {} vs {fd}",
                        j[r][c]
                    );
                }
            }
        }
    }

    #[test]
    fn driving_the_bearings_in_step_takes_the_jet_out_of_the_plane() {
        // The reason this module has a solver in it. A ganged schedule reaches
        // the envelope and swings the jet most of a right angle sideways doing
        // it, which on an aircraft is a yaw upset rather than a vectored thrust.
        let n = nozzle();
        let worst = (0..=180)
            .map(|t| n.lateral(&[t as f64; 3]).abs())
            .fold(0.0f64, f64::max);
        assert!(worst > 60.0, "the ganged path only wandered {worst} degrees");
    }

    #[test]
    fn the_solved_schedule_holds_the_plane_across_the_envelope() {
        let n = nozzle();
        // The general search, on a nozzle that does not need it: this one has a
        // roll bearing and a closed form. What is being checked is that the
        // search still gets there, because the same code has to serve a chain
        // that has no such bearing — see the module docs. It costs a bigger
        // break-out and a few degrees of yaw inside it, which is exactly the
        // price the roll bearing exists to avoid.
        let s = n.searched_schedule(95.0, 600);
        assert!(s.is_monotonic(), "the closed loop doubled back");
        assert!(s.reach() > 94.9, "only reached {}", s.reach());
        assert!(s.breakout < 8.0, "break-out ran to {} degrees", s.breakout);

        // Past break-out the loop is closed and the jet is in the plane.
        for p in s.points.iter().filter(|p| p.deflection > s.breakout + 1e-9) {
            assert!(
                p.lateral.abs() < 0.01,
                "at {} deg the jet was {} deg out of plane",
                p.deflection,
                p.lateral
            );
            assert!(p.residual < 0.01, "missed by {} deg", p.residual);
        }
        // All of the excursion is in break-out, and on this layout it is several
        // degrees — where `roll_schedule` on the same nozzle is exact.
        assert!(s.worst_lateral() < 8.0, "worst lateral {}", s.worst_lateral());
        assert!(
            n.roll_schedule(95.0, 600).worst_lateral() < 1e-9,
            "the closed form should not wander at all"
        );
    }

    #[test]
    fn travel_parametrises_the_deployment_evenly() {
        let n = nozzle();
        let s = n.schedule(95.0, 400);
        let travel = s.travel();
        assert_eq!(travel.len(), s.points.len());
        assert!(travel.windows(2).all(|w| w[1] >= w[0]), "travel went backwards");
        let total = *travel.last().unwrap();
        assert!(total > 180.0, "the deployment only moved {total} deg of bearing");

        // Ends land on the ends.
        assert!(s.bearings_at_travel(0.0).iter().all(|v| v.abs() < 1e-9));
        assert_eq!(s.bearings_at_travel(1.0), s.points.last().unwrap().bearings);

        // Even in travel means no lurch: every tenth of the way costs about a
        // tenth of the movement, which is exactly what interpolating on
        // deflection does not give.
        let mut prev = s.bearings_at_travel(0.0);
        let mut worst: f64 = 0.0;
        for i in 1..=100 {
            let q = s.bearings_at_travel(i as f64 / 100.0);
            worst = worst.max(
                q.iter()
                    .zip(&prev)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs())),
            );
            prev = q;
        }
        assert!(worst < 0.02 * total, "a lurch of {worst} deg in one percent of travel");

        // And it really does sweep the envelope on the way.
        assert!(n.deflection(&s.bearings_at_travel(1.0)) > 94.9);
    }

    #[test]
    fn the_schedule_is_singular_at_both_ends_and_not_in_between() {
        let n = nozzle();
        let s = n.schedule(95.0, 400);
        let (worst, at) = s.worst_authority();
        assert!(worst < 1e-6, "the ends should be singular, got {worst} at {at}");
        let middle = s
            .points
            .iter()
            .filter(|p| p.deflection > 20.0 && p.deflection < 80.0)
            .fold(f64::INFINITY, |m, p| m.min(p.authority));
        // Weakest interior point sits around 24 degrees, at a shade over 0.12
        // jet-degrees per bearing-degree. That is a working mechanism: the
        // actuators move eight times the jet does at the worst of it.
        assert!(middle > 0.10, "the middle of the envelope is weak: {middle}");
    }

    #[test]
    fn the_table_interpolates_and_clamps() {
        let n = nozzle();
        let s = n.schedule(95.0, 200);
        let at_zero = s.bearings_at(-5.0);
        assert!(at_zero.iter().all(|v| v.abs() < 1e-9), "{at_zero:?}");
        let past_end = s.bearings_at(120.0);
        assert_eq!(past_end, s.points.last().unwrap().bearings);
        // A commanded deflection comes back out of the mechanism it went in as.
        for want in [10.0, 30.0, 55.0, 80.0, 92.0] {
            let q = s.bearings_at(want);
            assert!(
                (n.deflection(&q) - want).abs() < 0.5,
                "asked {want}, table gives {}",
                n.deflection(&q)
            );
            assert!(n.lateral(&q).abs() < 0.5, "interpolation left the plane");
        }
    }

    #[test]
    fn full_deflection_is_lift_and_a_little_braking() {
        let n = nozzle();
        let q = [180.0; 3];
        let t = n.thrust(&q, 80_000.0);
        assert!(t.lift_fraction() > 0.99, "lift fraction {}", t.lift_fraction());
        // Past 90 degrees the jet points slightly forward, so the reaction on
        // the airframe is rearward: the nozzle brakes as well as lifts.
        assert!(t.axial() < 0.0, "axial {}", t.axial());
        assert!(t.axial() > -0.15 * t.thrust, "braking far too hard: {}", t.axial());
        assert!(t.side().abs() < 1.0, "side force {}", t.side());

        // Stowed it is all thrust and no lift.
        let s = n.thrust(&[0.0; 3], 80_000.0);
        assert!((s.axial() - 80_000.0).abs() < 1e-3);
        assert!(s.lift().abs() < 1e-6);
        assert!(s.pitch().abs() < 1e-6, "a straight jet on the axis has no moment");
    }

    #[test]
    fn a_deflected_jet_pitches_the_aircraft() {
        // Thrust reported about a centre of gravity forward of the nozzle: lift
        // applied behind the CG pitches the nose down.
        let n = nozzle().reference([-4.0, 0.0, 0.0]);
        let s = n.schedule(95.0, 200);
        let q = s.bearings_at(90.0);
        let t = n.thrust(&q, 80_000.0);
        assert!(t.lift() > 0.9 * t.thrust);
        assert!(t.pitch() < 0.0, "lift behind the CG is nose-down: {}", t.pitch());
        // The arm is a little over five metres, so the moment is that times lift.
        let arm = t.exit[0] - (-4.0);
        assert!(
            (-t.pitch() - arm * t.lift()).abs() < 0.02 * t.pitch().abs() + 1.0,
            "moment {} vs lift*arm {}",
            t.pitch(),
            arm * t.lift()
        );
        // And stowed there is no vertical force to pitch with at all.
        let stowed = n.thrust(&[0.0; 3], 80_000.0);
        assert!(stowed.pitch().abs() < 1e-6, "{}", stowed.pitch());
    }

    #[test]
    fn holding_torque_peaks_where_the_nozzle_hangs_furthest_out() {
        let n = nozzle();
        let stowed = n.holding_torque(&[0.0; 3]);
        // Straight and level, every bearing axis is within 24 degrees of the
        // duct and the weight hangs on the structure rather than the actuators —
        // but not at zero, because the axes are canted.
        assert!(stowed.iter().all(|t| t.abs() < 6000.0), "{stowed:?}");

        let s = n.schedule(95.0, 200);
        let worst = (0..=95)
            .map(|d| {
                let q = s.bearings_at(d as f64);
                n.holding_torque(&q)
                    .iter()
                    .fold(0.0f64, |m, t| m.max(t.abs()))
            })
            .fold(0.0f64, f64::max);
        // Real numbers for a half-tonne of duct on a metre of arm: thousands of
        // newton-metres, not tens and not millions.
        assert!(
            (500.0..50_000.0).contains(&worst),
            "peak holding torque {worst} N.m looks wrong"
        );
    }

    #[test]
    fn accelerating_a_bearing_costs_torque_on_top_of_holding_it() {
        let n = nozzle();
        let s = n.schedule(95.0, 200);
        let q = s.bearings_at(45.0);
        let hold = n.holding_torque(&q);

        // Standing still is holding, exactly.
        let still = n.slew_torque(&q, &[0.0; 3], &[0.0; 3]);
        for (a, b) in still.iter().zip(&hold) {
            assert!((a - b).abs() < 1e-9, "{still:?} vs {hold:?}");
        }

        // At rest, the inertial part is linear in acceleration, so doubling the
        // acceleration doubles the departure from the holding torque. Not that
        // slewing always costs *more*: on some joints at some poses the inertia
        // pulls the same way as gravity and the actuator has an easier time.
        let one = n.slew_torque(&q, &[0.0; 3], &[60.0; 3]);
        let two = n.slew_torque(&q, &[0.0; 3], &[120.0; 3]);
        for i in 0..3 {
            let d1 = one[i] - hold[i];
            let d2 = two[i] - hold[i];
            assert!(d1.abs() > 1e-6, "joint {i} felt no acceleration at all");
            assert!(
                (d2 / d1 - 2.0).abs() < 1e-6,
                "joint {i}: {d1} then {d2}, not double"
            );
        }

        // Turning at a steady rate is not free either: a chain rotating about
        // three axes at once has centrifugal and Coriolis terms, and those go as
        // the square of the rate rather than linearly.
        let slow = n.slew_torque(&q, &[20.0; 3], &[0.0; 3]);
        let fast = n.slew_torque(&q, &[40.0; 3], &[0.0; 3]);
        for i in 0..3 {
            let d1 = slow[i] - hold[i];
            let d2 = fast[i] - hold[i];
            assert!(d1.abs() > 1e-6, "joint {i} felt no rotation at all");
            assert!(
                (d2 / d1 - 4.0).abs() < 1e-5,
                "joint {i}: {d1} then {d2}, not four times"
            );
        }
    }

    #[test]
    fn the_solver_lands_on_a_commanded_direction() {
        let n = nozzle();
        let mut seed = vec![n.breakout; 3];
        for want in [5.0, 20.0, 45.0, 70.0, 90.0] {
            let sol = n.solve(&seed, want, 0.0);
            assert!(
                sol.axis_err < 1e-3,
                "asked {want} deg, missed by {} ({:?})",
                sol.axis_err,
                sol.stop
            );
            seed = sol.joints;
        }
        // Off-plane commands work too — the mechanism is not restricted to
        // pitch, it is only usually asked for it.
        let sol = n.solve(&seed, 40.0, 25.0);
        assert!(sol.axis_err < 1e-3, "off-plane miss {}", sol.axis_err);
        assert!((n.azimuth(&sol.joints) - 25.0).abs() < 0.01);
    }

    #[test]
    fn symmetric_3x3_eigenvalues_come_back_descending_and_right() {
        // A matrix with known eigenvalues 6, 3, 1.
        let a = [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 4.0]];
        let e = sym3_eigenvalues(a);
        assert!(e[0] >= e[1] && e[1] >= e[2]);
        let trace = a[0][0] + a[1][1] + a[2][2];
        assert!((e.iter().sum::<f64>() - trace).abs() < 1e-9);
        // Diagonal input takes the early exit and still sorts.
        let d = sym3_eigenvalues([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]]);
        assert_eq!(d, [5.0, 3.0, 1.0]);
    }

    #[test]
    fn a_bearing_rotation_is_the_chain_in_one_line() {
        // One bearing turned by 180 degrees folds the duct by twice its cant.
        // Checked against the rotation directly, not against the chain that
        // produced it.
        use super::super::chain::apply;
        for cant in [5.0, 11.875, 23.75, 40.0] {
            let r = bearing_rotation(cant, 180.0);
            let bent = apply(r, [1.0, 0.0, 0.0]);
            assert!(
                (angle_between(bent, [1.0, 0.0, 0.0]) - 2.0 * cant).abs() < 1e-9,
                "cant {cant} folded {}",
                angle_between(bent, [1.0, 0.0, 0.0])
            );
        }
    }

    #[test]
    fn segments_and_masses_line_up_with_the_stations() {
        let n = nozzle();
        let seg = n.segments();
        assert_eq!(seg.len(), 3);
        let stations: Vec<f64> = n.bearings.iter().map(|b| b.station).collect();
        assert!((seg[0] - (stations[1] - stations[0])).abs() < 1e-12);
        assert!((seg[1] - (stations[2] - stations[1])).abs() < 1e-12);
        assert!((seg[2] - (n.exit - stations[2])).abs() < 1e-12);
        assert!((seg.iter().sum::<f64>() - n.exit).abs() < 1e-12);

        // Masses substituted from geometry reach the torques.
        let heavy = nozzle().masses([900.0, 900.0, 600.0]);
        let a = n.holding_torque(&[0.0, 45.0, 0.0]);
        let b = heavy.holding_torque(&[0.0, 45.0, 0.0]);
        assert!(b[0].abs() > a[0].abs(), "{b:?} vs {a:?}");
    }
}
