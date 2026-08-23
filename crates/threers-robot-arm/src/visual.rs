//! The parts of the machine that are not colliders.
//!
//! The physics only needs three capsules and a ball — that is the whole arm as
//! far as the solver is concerned, and adding geometry to it would cost
//! collision work for nothing. But three capsules do not read as a robot.
//!
//! These parts are drawn and never simulated. Their transforms come from the
//! same solved chain the colliders do, so they cannot drift out of agreement
//! with it, and they cost one quaternion each per frame.
//!
//! # What actually turns the joints
//!
//! Stepper motors, on the axis of the joint they drive, each behind a reduction
//! housing. Not a cylinder standing in for "a joint", and not a linear ram —
//! a ram is a hydraulic or pneumatic part and belongs to a different machine
//! entirely. An arm built this way is driven by rotary motors and has no
//! cylinders on it anywhere.
//!
//! The sizes are the real NEMA frames, in metres, matching the model in
//! [`threers::build_nema17`]: a 42.3 mm square body, a ⌀22 mm front boss and a
//! ⌀5 mm shaft for a 17, scaled through the standard sizes for the others.
//!
//! # On whether the motors are big enough
//!
//! Honestly: only with the reduction. A NEMA 34 holds a few newton-metres, and
//! this arm's shoulder carries about 1.7 m of link and payload — tens of
//! newton-metres. The housings each joint sits in are what closes that gap, and
//! they are drawn at the size a harmonic drive of the necessary ratio would
//! be. The motors get smaller toward the tool for the same reason a real arm's
//! do: each one only has to move what is beyond it.

use crate::arm::{Chain, COLUMN};
use threers_physics::prelude::*;

/// A NEMA frame, in metres.
#[derive(Clone, Copy)]
pub struct Nema {
    /// Across the square body.
    pub frame: f32,
    /// Along the shaft.
    pub body: f32,
    /// The raised ring the motor locates on.
    pub boss_radius: f32,
    pub boss_depth: f32,
    pub shaft_radius: f32,
    pub shaft_length: f32,
}

/// 86 mm frame. The largest common size, and what a metre-scale arm's first two
/// axes are built around.
pub const NEMA_34: Nema = Nema {
    frame: 0.086,
    body: 0.098,
    boss_radius: 0.0365,
    boss_depth: 0.002,
    shaft_radius: 0.00635,
    shaft_length: 0.032,
};
/// 57 mm frame.
pub const NEMA_23: Nema = Nema {
    frame: 0.057,
    body: 0.056,
    boss_radius: 0.0191,
    boss_depth: 0.0016,
    shaft_radius: 0.00318,
    shaft_length: 0.021,
};
/// 42.3 mm frame — the one in `build_nema17`, to its own dimensions.
pub const NEMA_17: Nema = Nema {
    frame: 0.0423,
    body: 0.042,
    boss_radius: 0.011,
    boss_depth: 0.0025,
    shaft_radius: 0.0025,
    shaft_length: 0.024,
};
/// 28 mm frame, for the gripper.
pub const NEMA_11: Nema = Nema {
    frame: 0.0282,
    body: 0.045,
    boss_radius: 0.011,
    boss_depth: 0.0015,
    shaft_radius: 0.0025,
    shaft_length: 0.020,
};

/// The motor on each axis, base first. Each drives what is beyond it, so they
/// shrink toward the tool.
const MOTORS: [Nema; 5] = [NEMA_34, NEMA_34, NEMA_23, NEMA_17, NEMA_11];
/// Radius of the reduction housing each joint turns in.
const HOUSING: [f32; 3] = [0.115, 0.092, 0.058];

/// How many visual parts each arm contributes, in the order [`transforms`]
/// writes them.
pub const PARTS_PER_ARM: usize = 2 + 5 * 3 + 3 + 2;

/// Kind, half-size and role for one arm's worth of visual parts.
///
/// Roles: 3 base structure, 5 reduction housing, 6 gripper jaw, 8 motor.
/// Kinds: 1 box, 3 cylinder.
pub fn drawables() -> Vec<(u32, [f32; 3], u32)> {
    let mut out = Vec::with_capacity(PARTS_PER_ARM);
    // Turret and column.
    out.push((3, [0.175, 0.070, 0.0], 3));
    out.push((3, [0.090, 0.190, 0.0], 3));
    // Every motor: body, boss, shaft.
    for motor in MOTORS {
        out.push((1, [motor.frame * 0.5, motor.body * 0.5, motor.frame * 0.5], 8));
        out.push((3, [motor.boss_radius, motor.boss_depth * 0.5, 0.0], 8));
        out.push((3, [motor.shaft_radius, motor.shaft_length * 0.5, 0.0], 8));
    }
    // Reduction housings at the three pitch joints.
    for radius in HOUSING {
        out.push((3, [radius, radius * 0.85, 0.0], 5));
    }
    // Two gripper jaws.
    out.push((1, [0.010, 0.048, 0.026], 6));
    out.push((1, [0.010, 0.048, 0.026], 6));
    out
}

/// Position and rotation of every visual part, appended to `out` as 7 floats
/// each in the order [`drawables`] lists them.
pub fn transforms(chain: &Chain, gripping: bool, out: &mut Vec<f32>) {
    let base = chain.base;
    let joints: Vec<Vector3> = chain.chain.joints.iter().map(|j| j.position).collect();
    let (shoulder, elbow, wrist, tip) = (joints[0], joints[1], joints[2], joints[3]);

    // Which way the machine faces, and the axle every pitch joint turns about.
    // The same construction the IK uses for its elbow hinge, so a motor's shaft
    // lines up with the axis the joint actually has rather than merely looking
    // like it does.
    let flat = Vector3::new(tip.x - shoulder.x, 0.0, tip.z - shoulder.z);
    let heading = if flat.length() > 1e-4 {
        flat.normalize()
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let up = Vector3::new(0.0, 1.0, 0.0);
    let axle = up.cross(heading).normalize();

    let along = |dir: Vector3| crate::arm::rotation_from_to(up, dir);
    let mut push = |p: Vector3, q: Quaternion| {
        out.extend_from_slice(&[p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
    };

    // Turret and column yaw with the machine.
    let yaw = crate::arm::rotation_from_to(Vector3::new(1.0, 0.0, 0.0), heading);
    push(base + Vector3::new(0.0, 0.17, 0.0), yaw);
    push(base + Vector3::new(0.0, COLUMN * 0.5 + 0.09, 0.0), yaw);

    // A motor mounted so its shaft runs along `axis` into a housing of radius
    // `clearance` centred on `at`. The body sits behind the face plate, which is
    // where a real one goes — bolted to the far side of the joint plate with the
    // shaft passing through it.
    let mut motor = |m: Nema, at: Vector3, axis: Vector3, clearance: f32| {
        let rotation = along(axis);
        let face = at - axis * clearance;
        push(face - axis * (m.body * 0.5), rotation);
        push(face + axis * (m.boss_depth * 0.5), rotation);
        push(face + axis * (m.shaft_length * 0.5), rotation);
    };

    // Base yaw: shaft vertical, motor under the turret.
    motor(MOTORS[0], base + Vector3::new(0.0, 0.10, 0.0), up, 0.0);
    // The three pitch joints, each on the axle.
    motor(MOTORS[1], shoulder, axle, HOUSING[0]);
    motor(MOTORS[2], elbow, axle, HOUSING[1]);
    motor(MOTORS[3], wrist, axle, HOUSING[2]);

    // Gripper motor: at the tool, shaft across the jaws so it can drive them.
    let tool_dir = if (tip - wrist).length() > 1e-4 {
        (tip - wrist).normalize()
    } else {
        Vector3::new(0.0, -1.0, 0.0)
    };
    motor(MOTORS[4], tip - tool_dir * 0.045, axle, 0.03);

    // The reduction housings themselves.
    let axle_rotation = along(axle);
    for at in [shoulder, elbow, wrist] {
        push(at, axle_rotation);
    }

    // Jaws: either side of the tool, closing across the axle.
    let jaw_rotation = along(tool_dir);
    let spread = if gripping { 0.028 } else { 0.056 };
    let seat = tip + tool_dir * 0.018;
    push(seat + axle * spread, jaw_rotation);
    push(seat - axle * spread, jaw_rotation);
}
