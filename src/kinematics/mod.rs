//! Inverse kinematics, servo physics, and gear-train simulation.
//!
//! # IK
//!
//! Inverse kinematics: joint angles that put a tool where you want it.
//!
//! Forward kinematics is a function; inverse kinematics is a search, and for a
//! redundant arm — more joints than the task constrains — it is a search with a
//! continuum of answers. This module solves it by damped least squares, which
//! among that continuum returns the *smallest* joint move from the seed. That
//! property is the reason to prefer it over a general optimiser: seeding each
//! station of a trajectory from the previous one then keeps the whole path on
//! one IK branch, instead of snapping between elbow-up and elbow-down halfway
//! through a move because both solve the pose equally well.
//!
//! # The failure this exists to prevent
//!
//! The obvious way to ask for "reach this point AND hold the tool square to it"
//! is to score a pose by `distance + weight * angle` and minimise that. It does
//! not work, and it fails quietly. A weighted sum is an exchange rate, so the
//! search will happily buy 50 mm of reach with 30 degrees of lean whenever the
//! rate allows; raising the weight only moves the price. On a docking approach
//! that reads as a tool arriving diagonally at a latch that only closes square.
//!
//! Least squares separates them. Position and orientation are distinct rows of
//! one residual, and where a pose satisfying both exists the solve drives every
//! row to zero — it arrives on target AND square, with no exchange rate to
//! settle short of it. Where no such pose exists it cannot invent one, and then
//! [`crate::kinematics::Ik::rot_weight`] does decide how the unavoidable error is split; the
//! difference from a scalar cost is that the split is monotonic in the weight
//! and only ever spends what the geometry actually forbids. Wind the weight up
//! and the tool holds exactly square while the shortfall is reported in full,
//! which is a number you can act on rather than a lean you have to notice.
//!
//! ```
//! use threers::kinematics::{Goal, Ik};
//!
//! // A two-link planar arm in XZ, joints in degrees about +Y, tool along -Z.
//! let tool = |q: &[f64]| {
//!     let (a, b) = (q[0].to_radians(), q[0].to_radians() + q[1].to_radians());
//!     let p = [400.0 * a.sin() + 300.0 * b.sin(), 0.0, 400.0 * a.cos() + 300.0 * b.cos()];
//!     ([p[0], p[1], p[2]], [b.sin(), 0.0, b.cos()])
//! };
//! let s = Ik::default().solve(&[10.0, 20.0], &Goal::Point([500.0, 0.0, 300.0]), tool);
//! assert!(s.position_err < 1e-3, "{s:?}");
//! ```
//!
//! # Servo + drive physics
//!
//! [`crate::kinematics::Drive`] covers geared motors ([`crate::kinematics::GearTrain`] + temp-scaled Stribeck),
//! hydraulics ([`crate::kinematics::HydraulicFluid`] ν(T), leak, bulk modulus), and tendons
//! ([`crate::kinematics::TendonMaterial`] stretch, CTE pretension, routing μ). [`crate::kinematics::DriveEnv`] holds
//! operating temperature. [`crate::kinematics::ServoJoint`] / [`crate::kinematics::ArmPlant`] step through the drive;
//! [`crate::kinematics::IkServoSim`] reports tip error vs ideal IK.
//!
//! [`crate::kinematics::PlanarArm`] is a ready-made 3R FK for demos and tests. Render a side-by-side
//! comparison with:
//!
//! ```text
//! cargo run --release --example ik_servo_demo
//! cargo run --release --example ik_servo_sim
//! ```
//!
//! ```
//! use threers::kinematics::{
//!     sim::{IkServoSim, Waypoint},
//!     servo::presets,
//!     Goal, Ik,
//! };
//!
//! let ik = Ik::default();
//! let seed = [10.0, 20.0, 30.0];
//! let waypoints = vec![Waypoint {
//!     t: 0.5,
//!     goal: Goal::Point([500.0, 0.0, 300.0]),
//! }];
//! let tool = |q: &[f64]| {
//!     let (a, b) = (q[0].to_radians(), (q[0] + q[1]).to_radians());
//!     ([400.0 * a.sin() + 300.0 * b.sin(), 0.0, 400.0 * a.cos() + 300.0 * b.cos()], [0.0, 0.0, 1.0])
//! };
//! let mut joints = presets::qdd(3);
//! let report = IkServoSim::new(ik).with_label("qdd").run(&seed, &waypoints, &mut joints, tool);
//! assert!(!report.frames.is_empty());
//! ```

pub mod actuator;
pub mod chain;
pub mod design;
pub mod materials;
pub mod plant;
pub mod servo;
pub mod sim;
/// Three-bearing swivel nozzles: obliquely-cut rotary joints that vector a jet.
pub mod swivel;
pub mod transmission;
pub mod world;

#[cfg(test)]
pub(crate) mod tests_support;

pub use actuator::{Drive, DriveOutput, HydraulicDrive, TendonDrive};
pub use chain::{JointPose, PlanarArm, RevoluteJoint, SerialChain};
pub use design::{
    assemble_joint, parts_flat, ActuatorDesign, DesignPart, GearMechanism, HydraulicMechanism,
    PartRole, PartShape, TendonMechanism, PART_STRIDE,
};
pub use materials::{DriveEnv, HydraulicFluid, TendonMaterial};
pub use plant::ArmPlant;
pub use servo::{ServoJoint, ServoMode};
pub use sim::{ideal_envelope, IkServoSim, SimFrame, SimReport, Waypoint};
pub use swivel::{Bearing, Schedule, SchedulePoint, SwivelNozzle, ThrustState};
pub use transmission::{GearTrain, StribeckFriction};
pub use world::{ArmWorld, Contact, JointState, LinkBody, LinkMaterial, WorldObstacles};

pub(crate) fn wrap_deg(e: f64) -> f64 {
    let mut x = e % 360.0;
    if x > 180.0 {
        x -= 360.0;
    }
    if x < -180.0 {
        x += 360.0;
    }
    x
}

pub type V3 = [f64; 3];

pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(v: V3) -> f64 {
    dot(v, v).sqrt()
}

pub(crate) fn unit(v: V3) -> V3 {
    let n = norm(v);
    if n < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

/// What the tool has to achieve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Goal {
    /// Put the tool point here. Orientation is left free — with a redundant arm
    /// that means the solver will use the slack for whatever costs least, which
    /// is fine for a waypoint and wrong for a docking approach.
    Point(V3),
    /// Put the tool point here *and* aim the tool axis along `along`. Roll about
    /// the axis stays free, which is what you want: a latch cares which way the
    /// tool points, not how it is clocked around its own centreline.
    PointAlong {
        at: V3,
        /// Need not be normalised.
        along: V3,
    },
}

impl Goal {
    /// Where the tool point has to end up, whichever variant this is.
    pub fn at(&self) -> V3 {
        match *self {
            Goal::Point(p) => p,
            Goal::PointAlong { at, .. } => at,
        }
    }
    fn rows(&self) -> usize {
        match self {
            Goal::Point(_) => 3,
            Goal::PointAlong { .. } => 6,
        }
    }
}

/// Why the iteration stopped. Not all of these are failures, but only one of
/// them means the answer is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// Residual fell below the tolerance. The pose is the answer.
    Converged,
    /// A step stopped reducing the residual. For a target outside the envelope
    /// — or outside it *at that orientation* — this is the normal ending, and
    /// the returned pose is the closest approach.
    Stalled,
    /// The damped normal equations went singular. Rare, and worth seeing.
    Singular,
    /// Ran out of iterations while still improving. Raise `max_iters`.
    MaxIters,
}

/// The pose, and an honest account of how well it met the goal.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Joint values, in the same units and order as the seed.
    pub joints: Vec<f64>,
    /// Distance from the tool point to the target, in the caller's length unit.
    pub position_err: f64,
    /// Angle between the tool axis and the requested one, in degrees. Zero for
    /// a [`Goal::Point`], which does not constrain it.
    pub axis_err: f64,
    pub iters: usize,
    pub stop: Stop,
}

impl Solution {
    /// Did it meet both tolerances? `pos` is a length, `axis` is degrees.
    pub fn within(&self, pos: f64, axis: f64) -> bool {
        self.position_err <= pos && self.axis_err <= axis
    }
}

/// A damped least squares IK solver.
///
/// Joint values are in **degrees** throughout — seed, solution and limits. The
/// Jacobian is taken with respect to radians internally, which is what keeps
/// `lambda` and `rot_weight` in units you can reason about.
#[derive(Debug, Clone)]
pub struct Ik {
    /// Levenberg-Marquardt damping, in the residual's length unit. Larger is
    /// steadier and slower; it is also what biases the answer toward the
    /// smallest joint move, so do not drive it to zero to chase convergence.
    pub lambda: f64,
    /// How many length units one unit of axis error is worth. The axis residual
    /// is a difference of unit vectors, so one degree is 0.01745 units: the
    /// default 400 makes a degree cost about 7 mm, which is the right order for
    /// an arm working in millimetres.
    pub rot_weight: f64,
    /// Largest single-joint change per iteration, degrees. This is not a speed
    /// limit but a branch lock — one unbounded step can fling a joint across its
    /// range into a different IK solution family and strand the trajectory.
    pub max_step: f64,
    /// Stop when the weighted residual norm falls below this.
    pub tol: f64,
    pub max_iters: usize,
    /// Applied to every joint when [`Self::joint_limits`] is empty, degrees.
    pub limits: (f64, f64),
    /// Per-joint travel, degrees. Empty means use [`Self::limits`] for all.
    pub joint_limits: Vec<(f64, f64)>,
}

impl Default for Ik {
    fn default() -> Self {
        Self {
            lambda: 12.0,
            rot_weight: 400.0,
            max_step: 15.0,
            tol: 1e-3,
            max_iters: 60,
            limits: (-180.0, 180.0),
            joint_limits: Vec::new(),
        }
    }
}

impl Ik {
    fn clamp_joint(&self, i: usize, q: f64) -> f64 {
        let (lo, hi) = self
            .joint_limits
            .get(i)
            .copied()
            .unwrap_or(self.limits);
        q.clamp(lo, hi)
    }

    /// Solve against a serial chain using the geometric Jacobian
    /// (`axis × (tip − hinge)`), not a finite difference.
    pub fn solve_chain(&self, seed: &[f64], goal: &Goal, chain: &chain::SerialChain) -> Solution {
        self.solve_inner(seed, goal, |q| chain.tool(q), Some(chain))
    }

    /// Solve from `seed`, with `tool(joints) -> (point, axis)` as the forward
    /// kinematics. `axis` may be unnormalised and is ignored for a
    /// [`Goal::Point`], so a caller that has no tool direction can return
    /// anything for it.
    ///
    /// Never panics on a bad goal and never reports success it did not achieve:
    /// an unreachable target returns [`Stop::Stalled`] with the miss in
    /// `position_err`.
    pub fn solve<F>(&self, seed: &[f64], goal: &Goal, tool: F) -> Solution
    where
        F: Fn(&[f64]) -> (V3, V3),
    {
        self.solve_inner(seed, goal, tool, None)
    }

    fn solve_inner<F>(
        &self,
        seed: &[f64],
        goal: &Goal,
        tool: F,
        chain: Option<&chain::SerialChain>,
    ) -> Solution
    where
        F: Fn(&[f64]) -> (V3, V3),
    {
        let n = seed.len();
        let rows = goal.rows();
        let target = goal.at();
        let along = match *goal {
            Goal::PointAlong { along, .. } => unit(along),
            Goal::Point(_) => [0.0, 0.0, 0.0],
        };

        // e = [ tcp - target ,  w * (axis - along) ].
        //
        // The orientation rows are a DIFFERENCE of unit vectors, not the cross
        // product the textbook reaches for. Both vanish only when the axis is
        // aligned, but `axis x along` also vanishes when the tool points exactly
        // BACKWARDS — a stationary point of the residual, so a seed anywhere
        // near anti-parallel converges smoothly to pointing the wrong way and
        // reports itself solved. The difference has no such point: its magnitude
        // is 2*sin(theta/2), which climbs monotonically to 2 at anti-parallel
        // and pushes hardest exactly where the cross product gives up.
        let resid = |q: &[f64], out: &mut [f64]| {
            let (p, a) = tool(q);
            out[0] = p[0] - target[0];
            out[1] = p[1] - target[1];
            out[2] = p[2] - target[2];
            if rows == 6 {
                let a = unit(a);
                for i in 0..3 {
                    out[3 + i] = self.rot_weight * (a[i] - along[i]);
                }
            }
        };
        let len = |e: &[f64]| e.iter().map(|v| v * v).sum::<f64>().sqrt();

        let mut cur = seed.to_vec();
        let mut e = vec![0.0; rows];
        resid(&cur, &mut e);

        // Scratch, allocated once: this runs thousands of times per trajectory.
        let mut e2 = vec![0.0; rows];
        let mut j = vec![0.0; rows * n]; // row-major, rows x n
        let mut a = vec![0.0; rows * rows];
        let mut y = vec![0.0; rows];
        let mut probe = vec![0.0; n];
        let mut next = vec![0.0; n];
        let mut step = vec![0.0; n];

        let mut stop = Stop::MaxIters;
        let mut iters = 0;
        // Damping adapts. `self.lambda` is the floor, not a constant: a step
        // that fails means the linearisation was trusted too far, and the fix
        // is to shorten it and try again from the same Jacobian — not to
        // conclude the arm cannot get there. Treating one failed step as the
        // end is how a solver reports a 200 mm miss on a target it can reach.
        let mut lambda = self.lambda;
        let lambda_max = self.lambda * 1e8;
        for it in 0..self.max_iters {
            iters = it + 1;
            if len(&e) < self.tol {
                stop = Stop::Converged;
                break;
            }

            if let Some(ch) = chain {
                let (jv, jw) = ch.jacobian(&cur);
                let (_, tool_axis) = tool(&cur);
                let ua = unit(tool_axis);
                for c in 0..n {
                    for r in 0..3 {
                        j[r * n + c] = jv[c][r];
                    }
                    if rows == 6 {
                        let d = chain::cross(jw[c], ua);
                        for k in 0..3 {
                            j[(3 + k) * n + c] = self.rot_weight * d[k];
                        }
                    }
                }
            } else {
                // Numeric Jacobian: one extra forward evaluation per joint.
                const H: f64 = 1e-4; // radians
                let h_deg = H.to_degrees();
                for c in 0..n {
                    probe.copy_from_slice(&cur);
                    probe[c] += h_deg;
                    resid(&probe, &mut e2);
                    for r in 0..rows {
                        j[r * n + c] = (e2[r] - e[r]) / H;
                    }
                }
            }

            // Shorten the step until it actually helps. Every retry reuses the
            // Jacobian — it is still the one taken at `cur` — so a rejection
            // costs one forward evaluation, not another n+1 of them.
            let mut accepted = false;
            let mut singular = false;
            while lambda <= lambda_max {
                // A = J J^T + lambda^2 I, then solve A y = e.
                for r in 0..rows {
                    for c in 0..rows {
                        a[r * rows + c] = (0..n).map(|k| j[r * n + k] * j[c * n + k]).sum();
                    }
                    a[r * rows + r] += lambda * lambda;
                }
                y.copy_from_slice(&e);
                if !solve_in_place(&mut a, &mut y, rows) {
                    // Damping only ever makes this better conditioned, so more
                    // of it is the answer here too.
                    lambda *= 10.0;
                    singular = true;
                    continue;
                }
                singular = false;

                // dtheta = -J^T y, radians to degrees, scaled to the step limit
                // as a whole so the step keeps its direction.
                let mut worst = 0.0f64;
                for c in 0..n {
                    step[c] = -(0..rows)
                        .map(|r| j[r * n + c] * y[r])
                        .sum::<f64>()
                        .to_degrees();
                    worst = worst.max(step[c].abs());
                }
                let scale = if worst > self.max_step {
                    self.max_step / worst
                } else {
                    1.0
                };
                for c in 0..n {
                    next[c] = self.clamp_joint(c, cur[c] + step[c] * scale);
                }

                resid(&next, &mut e2);
                if len(&e2) < len(&e) {
                    accepted = true;
                    break;
                }
                lambda *= 10.0;
            }

            if !accepted {
                // Even a step next to zero cannot reduce the residual: this is a
                // real local minimum, which for a target outside the envelope —
                // or outside it at that orientation — is the expected ending.
                // `cur` is the closest approach, and that is the useful answer.
                stop = if singular {
                    Stop::Singular
                } else {
                    Stop::Stalled
                };
                break;
            }
            cur.copy_from_slice(&next);
            e.copy_from_slice(&e2);
            // Loosen again on success, so the next step can be longer than the
            // one that just worked. Never below the configured floor, which is
            // what keeps the bias toward the smallest joint move.
            lambda = (lambda / 3.0).max(self.lambda);
        }

        let (p, ax) = tool(&cur);
        let position_err = norm([p[0] - target[0], p[1] - target[1], p[2] - target[2]]);
        let axis_err = if rows == 6 {
            dot(unit(ax), along).clamp(-1.0, 1.0).acos().to_degrees()
        } else {
            0.0
        };
        Solution {
            joints: cur,
            position_err,
            axis_err,
            iters,
            stop,
        }
    }
}

/// Gauss-Jordan with partial pivoting on an `n x n` system, in place. Returns
/// false if it hits a zero pivot. `n` is 3 or 6 here, far too small to be worth
/// a linear algebra dependency.
fn solve_in_place(m: &mut [f64], y: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if m[r * n + i].abs() > m[piv * n + i].abs() {
                piv = r;
            }
        }
        if m[piv * n + i].abs() < 1e-12 {
            return false;
        }
        if piv != i {
            for c in 0..n {
                m.swap(i * n + c, piv * n + c);
            }
            y.swap(i, piv);
        }
        let d = m[i * n + i];
        for c in i..n {
            m[i * n + c] /= d;
        }
        y[i] /= d;
        for r in 0..n {
            if r != i && m[r * n + i] != 0.0 {
                let f = m[r * n + i];
                for c in i..n {
                    m[r * n + c] -= f * m[i * n + c];
                }
                y[r] -= f * y[i];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests;
