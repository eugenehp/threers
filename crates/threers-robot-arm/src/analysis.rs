//! What the machine would actually need, in newton-metres and kilograms.
//!
//! # Why this exists
//!
//! Everything up to here has been geometry that looks plausible. Whether the
//! motors chosen can turn the links drawn is a different question, and it is not
//! one you can answer by looking — a link that reads as slender on screen can
//! easily want ten times the torque a NEMA 17 has. The only way to know is to
//! work out the torque the trajectory demands and compare it with what the
//! motors deliver.
//!
//! # How
//!
//! Inverse dynamics, the same calculation used to size real robot motors. For
//! each axis, take every body beyond it and sum what the joint has to supply:
//!
//! ```text
//! τ = Σ [ (c − p) × m(a − g)  +  I α  +  ω × I ω ] · axle
//! ```
//!
//! — the moment of accelerating each mass and holding it against gravity, plus
//! the moment of angularly accelerating it and the gyroscopic term. Accelerations
//! come from differencing the simulated motion, so this measures the trajectory
//! that was actually run rather than an idealised one.
//!
//! Run it against the kinematic arm. That one follows the commanded path
//! exactly, so the answer is "what would a real machine need to do this", which
//! is the question motor sizing asks.

use crate::arm::{ArmDriver, KinematicArm, AXIS_TORQUE, COLUMN, LINKS};
use crate::{bin_for, programme_for, PART, PICK_X};
use threers_physics::prelude::*;

/// One link, weighed.
#[derive(Debug, Clone)]
pub struct LinkReport {
    pub name: &'static str,
    pub mass: f32,
    pub length: f32,
    pub radius: f32,
    /// Second moment about the joint it hangs from, in kg·m².
    pub inertia_about_joint: f32,
}

/// One axis, sized.
#[derive(Debug, Clone)]
pub struct AxisReport {
    pub name: &'static str,
    pub motor: &'static str,
    /// What the motor delivers at the joint, after its reduction.
    pub available: f32,
    /// Worst demand over the whole programme.
    pub peak: f32,
    /// Root-mean-square demand, which is what heats the motor.
    pub rms: f32,
    /// Peak demand from gravity alone, with the arm still.
    pub holding: f32,
}

impl AxisReport {
    /// Available over peak. Under 1 means the motor stalls.
    pub fn headroom(&self) -> f32 {
        if self.peak > 1e-6 {
            self.available / self.peak
        } else {
            f32::INFINITY
        }
    }
}

#[derive(Debug, Clone)]
pub struct Performance {
    pub links: Vec<LinkReport>,
    pub axes: Vec<AxisReport>,
    pub arm_mass: f32,
    pub payload: f32,
    pub reach: f32,
    pub cycle: f32,
}

impl Performance {
    /// The axis with the least headroom — the one that decides the design.
    pub fn worst_axis(&self) -> &AxisReport {
        self.axes
            .iter()
            .min_by(|a, b| a.headroom().total_cmp(&b.headroom()))
            .expect("there is always at least one axis")
    }

    pub fn feasible(&self) -> bool {
        self.axes.iter().all(|a| a.headroom() >= 1.0)
    }
}

/// One body's state, for differencing.
#[derive(Clone, Copy, Default)]
struct State {
    com: Vector3,
    linear: Vector3,
    angular: Vector3,
    mass: f32,
    inertia: Mat3,
}

/// Run the programme and work out what it costs.
pub fn analyse() -> Performance {
    const DT: f32 = 1.0 / 60.0;
    const G: Vector3 = Vector3::new(0.0, -9.81, 0.0);

    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
    let mut arm = KinematicArm::build(&mut world, Vector3::ZERO);
    let payload = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(PART, PART, PART))
            .mass(0.04)
            .translation(Vector3::new(PICK_X, PART, 0.0))
            .friction(1.2)
            .can_sleep(false),
    );
    let bin = bin_for(Vector3::ZERO);
    for (dx, dz, hx, hz) in [
        (0.0f32, -0.06f32, 0.06f32, 0.006f32),
        (0.0, 0.06, 0.06, 0.006),
        (-0.06, 0.0, 0.006, 0.06),
        (0.06, 0.0, 0.006, 0.06),
    ] {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(hx, 0.03, hz))
                .translation(bin + Vector3::new(dx, 0.03, dz)),
        );
    }

    // Everything the arm has to carry, nearest the base first. Axis `j` is
    // responsible for links `j..`, and for the tool, jaws and payload always.
    let mut carried: Vec<BodyId> = arm.bodies().to_vec();
    let tail_start = carried.len();
    carried.extend_from_slice(&arm.gripper().jaws);
    carried.push(payload);

    let read = |world: &World, id: BodyId| -> State {
        world
            .body(id)
            .map(|b| State {
                com: b.world_center_of_mass(),
                linear: b.linear_velocity,
                angular: b.angular_velocity,
                mass: b.mass(),
                // The solver keeps the inverse because that is what it needs;
                // inverse dynamics needs the other one.
                inertia: b.world_inv_inertia().inverse(),
            })
            .unwrap_or_default()
    };

    let mut previous: Vec<State> = carried.iter().map(|id| read(&world, *id)).collect();
    let mut peak = [0.0f32; 4];
    let mut sum_squares = [0.0f32; 4];
    let mut holding = [0.0f32; 4];
    let mut samples = 0usize;

    let programme = programme_for(Vector3::ZERO);
    let mut frames = 0usize;
    for step in &programme {
        let start = arm.chain().tip();
        match step.grip {
            Some(true) => {
                arm.close_gripper(&mut world, payload);
            }
            Some(false) => arm.open_gripper(&mut world),
            None => {}
        }
        for frame in 0..step.frames {
            let t = (frame + 1) as f32 / step.frames as f32;
            let smooth = t * t * (3.0 - 2.0 * t);
            arm.aim(&mut world, start + (step.target - start) * smooth);
            world.step(DT);
            frames += 1;

            let now: Vec<State> = carried.iter().map(|id| read(&world, *id)).collect();
            let joints: Vec<Vector3> = arm
                .chain()
                .chain
                .joints
                .iter()
                .map(|j| j.position)
                .collect();
            let heading = arm.chain().heading();
            let axle = Vector3::new(0.0, 1.0, 0.0).cross(heading).normalize();

            for axis in 0..4 {
                // Axis 0 turns about the vertical at the column; the rest turn
                // about the shared axle at their own joint.
                let (pivot, direction) = if axis == 0 {
                    (Vector3::new(0.0, COLUMN, 0.0), Vector3::new(0.0, 1.0, 0.0))
                } else {
                    (joints[axis - 1], axle)
                };
                // Links from this axis outward, plus everything on the end.
                let first = if axis == 0 { 0 } else { axis - 1 };
                let mut torque = 0.0f32;
                let mut gravity_only = 0.0f32;
                for (i, id) in carried.iter().enumerate() {
                    let _ = id;
                    if i < tail_start && i < first {
                        continue;
                    }
                    let (a, b) = (previous[i], now[i]);
                    if b.mass <= 0.0 {
                        continue;
                    }
                    let accel = (b.linear - a.linear) * (1.0 / DT);
                    let alpha = (b.angular - a.angular) * (1.0 / DT);
                    let arm_vector = b.com - pivot;

                    // What the joint must supply: enough to accelerate the mass
                    // *and* to hold it up.
                    let force = (accel - G) * b.mass;
                    let spin = b.inertia.mul_vec(alpha) + b.angular.cross(b.inertia.mul_vec(b.angular));
                    torque += (arm_vector.cross(force) + spin).dot(direction);
                    gravity_only += arm_vector.cross(G * -b.mass).dot(direction);
                }
                let magnitude = torque.abs();
                peak[axis] = peak[axis].max(magnitude);
                sum_squares[axis] += magnitude * magnitude;
                holding[axis] = holding[axis].max(gravity_only.abs());
            }
            samples += 1;
            previous = now;
        }
    }

    let links = LINKS
        .iter()
        .enumerate()
        .map(|(i, length)| {
            let radius = crate::arm::link_radius(i);
            let state = read(&world, arm.bodies()[i]);
            LinkReport {
                name: ["upper arm", "forearm", "wrist"][i],
                mass: state.mass,
                length: *length,
                radius,
                // Parallel axis: about the joint rather than the centre.
                inertia_about_joint: state.inertia.mul_vec(Vector3::new(0.0, 0.0, 1.0)).z
                    + state.mass * (length * 0.5).powi(2),
            }
        })
        .collect::<Vec<_>>();

    let names = ["base yaw", "shoulder", "elbow", "wrist"];
    let motors = ["NEMA 23", "NEMA 23", "NEMA 17", "NEMA 17"];
    let axes = (0..4)
        .map(|i| AxisReport {
            name: names[i],
            motor: motors[i],
            available: AXIS_TORQUE[i],
            peak: peak[i],
            rms: (sum_squares[i] / samples.max(1) as f32).sqrt(),
            holding: holding[i],
        })
        .collect();

    Performance {
        arm_mass: links.iter().map(|l| l.mass).sum(),
        links,
        axes,
        payload: 0.04,
        reach: LINKS.iter().sum(),
        cycle: frames as f32 * DT,
    }
}
