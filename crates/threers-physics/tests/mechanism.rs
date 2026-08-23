//! Machine elements: position drives, joint readout, and the couplings.
//!
//! Everything here is behind the `mechanism` feature, so the whole file is
//! compiled out without it rather than failing to build.

#![cfg(feature = "mechanism")]

use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

fn settle(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

/// A hinge between a fixed post and a bar sticking out along +X, turning about
/// +Z. Gravity is off unless a test wants it, so a drive is measured on its own.
fn hinge_rig(gravity: bool) -> (World, BodyId, BodyId, JointId) {
    let mut world = World::new();
    if !gravity {
        world.gravity = Vector3::ZERO;
    }
    let post = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.1, 0.1, 0.1)));
    let bar = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.5, 0.05, 0.05))
            .mass(1.0)
            .translation(Vector3::new(0.5, 0.0, 0.0))
            .can_sleep(false),
    );
    let hinge = world.add_joint(
        Joint::revolute_at_point(
            world.bodies(),
            post,
            bar,
            Vector3::ZERO,
            Vector3::new(0.0, 0.0, 1.0),
        )
        .unwrap(),
    );
    (world, post, bar, hinge)
}

// ---- joint state ---------------------------------------------------------

#[test]
fn a_hinge_reports_the_angle_it_is_actually_at() {
    let (mut world, _, bar, hinge) = hinge_rig(false);
    assert!(world.joint_angle(hinge).unwrap().abs() < 1e-5);

    // Turn the bar a quarter turn by hand and the joint says so.
    world
        .body_mut(bar)
        .unwrap()
        .set_rotation(Quaternion::from_axis_angle(
            Vector3::new(0.0, 0.0, 1.0),
            std::f32::consts::FRAC_PI_2,
        ));
    let angle = world.joint_angle(hinge).unwrap();
    assert!(
        (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
        "read {angle}"
    );

    // A hinge has an angle, not an offset.
    assert!(world.joint_offset(hinge).is_none());
}

#[test]
fn a_slider_reports_its_travel_and_a_ball_socket_reports_nothing() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let rail = world.add_body(RigidBody::fixed().shape(Shape::cuboid(2.0, 0.1, 0.1)));
    let carriage = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .translation(Vector3::new(0.5, 0.0, 0.0)),
    );
    let slide = world.add_joint(Joint::prismatic(
        rail,
        carriage,
        Vector3::ZERO,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
    ));
    // A prismatic measures `anchor_a - anchor_b` along the axis, so a carriage
    // half a unit up the rail reads -0.5. `assembly` is what turns that back
    // into the direction a person would quote.
    let offset = world.joint_offset(slide).unwrap();
    assert!((offset + 0.5).abs() < 1e-5, "read {offset}");

    let ball = world.add_joint(Joint::spherical(
        rail,
        carriage,
        Vector3::ZERO,
        Vector3::ZERO,
    ));
    assert!(world.joint_state(ball).is_none(), "a ball socket has no one coordinate");
}

#[test]
fn joint_speed_is_zero_at_rest_and_signed_while_moving() {
    let (mut world, _, _, hinge) = hinge_rig(false);
    assert!(world.joint_speed(hinge).unwrap().abs() < 1e-6);

    world.joint_mut(hinge).unwrap().kind = match world.joint(hinge).unwrap().kind.clone() {
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            limits,
            ..
        } => JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            limits,
            motor: Some(Motor::new(2.0, 100.0)),
            spring: None,
        },
        other => other,
    };
    settle(&mut world, 30);
    let speed = world.joint_speed(hinge).unwrap();
    assert!((speed - 2.0).abs() < 0.2, "the motor asked for 2 rad/s, got {speed}");
}

// ---- servo ---------------------------------------------------------------

#[test]
fn a_servo_holds_the_angle_it_was_given() {
    let (mut world, _, _, hinge) = hinge_rig(false);
    let target = 1.0;
    world.joint_mut(hinge).unwrap().servo = Some(Servo::new(target, 50.0));
    settle(&mut world, 240);

    let angle = world.joint_angle(hinge).unwrap();
    assert!((angle - target).abs() < 0.02, "settled at {angle}, wanted {target}");
    assert!(
        world.joint_speed(hinge).unwrap().abs() < 0.05,
        "it arrived but never stopped"
    );
}

#[test]
fn a_servo_holds_position_under_load() {
    // The point of a position drive over a velocity one: gravity is pulling the
    // bar down the whole time and it stays where it was put.
    let (mut world, _, _, hinge) = hinge_rig(true);
    world.joint_mut(hinge).unwrap().servo = Some(Servo::new(0.6, 200.0));
    settle(&mut world, 400);
    let angle = world.joint_angle(hinge).unwrap();
    assert!((angle - 0.6).abs() < 0.05, "drooped to {angle}");
}

#[test]
fn a_servo_cannot_beat_a_limit() {
    let (mut world, _, _, hinge) = hinge_rig(false);
    {
        let joint = world.joint_mut(hinge).unwrap();
        *joint = joint.clone().with_limits(-0.4, 0.4);
        joint.servo = Some(Servo::new(2.0, 80.0));
    }
    settle(&mut world, 300);
    let angle = world.joint_angle(hinge).unwrap();
    assert!(
        angle <= 0.4 + 0.02,
        "the servo pushed through the limit to {angle}"
    );
    assert!(angle > 0.35, "it should be sitting on the limit, not at {angle}");
}

#[test]
fn a_weak_servo_stalls_where_a_strong_one_arrives() {
    // Same target, same load, different ceiling: the whole reason `max_force`
    // is not optional. The bar needs about 4.9 N·m to lift off at all, so 0.05
    // cannot and 200 easily can.
    let mut reached = Vec::new();
    for torque in [0.05f32, 200.0] {
        let (mut world, _, _, hinge) = hinge_rig(true);
        world.joint_mut(hinge).unwrap().servo = Some(Servo::new(1.2, torque));
        settle(&mut world, 400);
        reached.push(world.joint_angle(hinge).unwrap());
    }
    assert!(reached[0] < 0.5, "a 0.05 N·m servo lifted the bar to {}", reached[0]);
    assert!(
        (reached[1] - 1.2).abs() < 0.05,
        "a 200 N·m servo should have got there, not {}",
        reached[1]
    );
}

#[test]
fn a_speed_capped_servo_takes_the_time_it_should() {
    let (mut world, _, _, hinge) = hinge_rig(false);
    // 1 rad at 0.5 rad/s is two seconds, so it is nowhere near done after one.
    world.joint_mut(hinge).unwrap().servo = Some(Servo::new(1.0, 100.0).max_speed(0.5));
    settle(&mut world, 60);
    let half_way = world.joint_angle(hinge).unwrap();
    assert!(
        (half_way - 0.5).abs() < 0.1,
        "after a second it should be about half way, not {half_way}"
    );
    settle(&mut world, 120);
    let angle = world.joint_angle(hinge).unwrap();
    assert!((angle - 1.0).abs() < 0.03, "it never finished: {angle}");
}

#[test]
fn a_slider_servo_drives_travel() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let rail = world.add_body(RigidBody::fixed().shape(Shape::cuboid(2.0, 0.1, 0.1)));
    let carriage = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .can_sleep(false),
    );
    let slide = world.add_joint(
        Joint::prismatic(
            rail,
            carriage,
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        )
        .with_servo(Servo::new(-0.8, 200.0)),
    );
    settle(&mut world, 300);
    let offset = world.joint_offset(slide).unwrap();
    assert!((offset + 0.8).abs() < 0.02, "settled at {offset}");
    // Which puts the carriage 0.8 up the rail.
    let x = world.body(carriage).unwrap().translation().x;
    assert!((x - 0.8).abs() < 0.02, "the carriage is at {x}");
}

// ---- couplings -----------------------------------------------------------

/// Two wheels on their own hinges, ready to be geared together.
fn gear_rig(ratio: f32) -> (World, JointId, JointId, JointId) {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.1, 0.1, 0.1)));
    let axis = Vector3::new(0.0, 1.0, 0.0);

    let mut wheel = |x: f32, radius: f32| {
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cylinder(0.05, radius))
                .translation(Vector3::new(x, 0.0, 0.0))
                .can_sleep(false),
        );
        let hinge = world.add_joint(
            Joint::revolute_at_point(
                world.bodies(),
                frame,
                body,
                Vector3::new(x, 0.0, 0.0),
                axis,
            )
            .unwrap(),
        );
        (body, hinge)
    };

    let (driver, drive_hinge) = wheel(0.0, 0.2);
    let (driven, driven_hinge) = wheel(0.6, 0.4);
    let gear = world.add_joint(Joint::gear(driver, driven, axis, axis, ratio));
    (world, drive_hinge, driven_hinge, gear)
}

#[test]
fn a_gear_pair_turns_at_the_ratio_it_was_given() {
    // Two turns of the driver per turn of the driven wheel, counter-rotating.
    let (mut world, drive_hinge, driven_hinge, _) = gear_rig(-2.0);
    world.joint_mut(drive_hinge).unwrap().kind = drive_with_motor(&world, drive_hinge, 4.0);
    settle(&mut world, 240);

    let fast = world.joint_speed(drive_hinge).unwrap();
    let slow = world.joint_speed(driven_hinge).unwrap();
    assert!((fast - 4.0).abs() < 0.3, "the driver ran at {fast}");
    assert!(
        (slow + fast / 2.0).abs() < 0.3,
        "a -2:1 pair should give {} against the driver's {fast}, got {slow}",
        -fast / 2.0
    );
}

#[test]
fn a_gear_pair_stays_in_phase_over_a_long_run() {
    let (mut world, drive_hinge, _, gear) = gear_rig(-3.0);
    world.joint_mut(drive_hinge).unwrap().kind = drive_with_motor(&world, drive_hinge, 6.0);
    settle(&mut world, 900);
    // Rate constraints drift; the phase feedback is what stops it accumulating.
    let phase = world.joint(gear).unwrap().coupling_phase();
    assert!(
        phase.abs() < 0.05,
        "fifteen seconds of running left {phase} rad of phase error"
    );
}

#[test]
fn a_gear_passes_torque_both_ways() {
    // Hold the driven wheel and the driver cannot turn either: a gear is a
    // constraint, not a one-way transmission.
    let (mut world, drive_hinge, driven_hinge, _) = gear_rig(-2.0);
    world.joint_mut(drive_hinge).unwrap().kind = drive_with_motor(&world, drive_hinge, 4.0);
    world.joint_mut(driven_hinge).unwrap().kind = drive_with_motor(&world, driven_hinge, 0.0);
    settle(&mut world, 180);
    let fast = world.joint_speed(drive_hinge).unwrap();
    assert!(
        fast.abs() < 1.5,
        "the held wheel should have slowed the driver, which still ran at {fast}"
    );
}

/// Replace a hinge's motor without rebuilding the joint.
fn drive_with_motor(world: &World, hinge: JointId, speed: f32) -> JointKind {
    match world.joint(hinge).unwrap().kind.clone() {
        JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            limits,
            ..
        } => JointKind::Revolute {
            local_axis_a,
            local_axis_b,
            limits,
            motor: Some(Motor::new(speed, 500.0)),
            spring: None,
        },
        other => other,
    }
}

#[test]
fn a_rack_advances_as_its_pinion_turns() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let frame = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.1, 0.1, 0.1)));
    let spin = Vector3::new(0.0, 0.0, 1.0);
    let slide = Vector3::new(1.0, 0.0, 0.0);

    let pinion = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cylinder(0.05, 0.25))
            .can_sleep(false),
    );
    let pinion_hinge = world.add_joint(
        Joint::revolute_at_point(world.bodies(), frame, pinion, Vector3::ZERO, spin).unwrap(),
    );
    let rack = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(1.0, 0.05, 0.05))
            .translation(Vector3::new(0.0, 0.25, 0.0))
            .can_sleep(false),
    );
    let rack_slide = world.add_joint(Joint::prismatic(
        frame,
        rack,
        Vector3::new(0.0, 0.25, 0.0),
        Vector3::ZERO,
        slide,
        slide,
    ));
    world.add_joint(Joint::rack_pinion(
        pinion,
        rack,
        Vector3::ZERO,
        Vector3::ZERO,
        spin,
        slide,
        0.25,
    ));

    world.joint_mut(pinion_hinge).unwrap().kind = drive_with_motor(&world, pinion_hinge, 2.0);
    settle(&mut world, 120);

    // Two seconds at 2 rad/s is 4 radians, times a 0.25 pitch radius: one unit.
    let travel = -world.joint_offset(rack_slide).unwrap();
    assert!(
        (travel - 1.0).abs() < 0.1,
        "the rack should have advanced a unit, not {travel}"
    );
}

#[test]
fn a_screw_advances_as_it_turns_and_cannot_do_one_without_the_other() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let nut = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.2, 0.2, 0.2)));
    let bolt = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cylinder(0.4, 0.08))
            .can_sleep(false),
    );
    let axis = Vector3::new(0.0, 1.0, 0.0);
    // A 2 mm-per-turn thread, in a model drawn in millimetres.
    let screw = world
        .add_joint(
            Joint::screw_from_pitch(
                world.bodies(),
                nut,
                bolt,
                Vector3::ZERO,
                Vector3::ZERO,
                axis,
                axis,
                2.0,
            )
            .unwrap()
            .with_motor(3.0, 500.0),
        );

    settle(&mut world, 120);

    // Two seconds at 3 rad/s is 6 radians — a little under one turn — so the
    // bolt should have backed out by 6 × (2 / 2π) ≈ 1.91 mm.
    let expected = 6.0 * 2.0 / std::f32::consts::TAU;
    let travel = -world.joint_offset(screw).unwrap();
    assert!(
        (travel - expected).abs() < 0.25,
        "expected about {expected} of travel, got {travel}"
    );

    // And it is the thread that tied them: the travel is the rotation times the
    // lead, not an independent slide that happened to land nearby.
    //
    // A screw's own coordinate is its *travel* — that is what its limits are in
    // — so the rotation is read off the bodies.
    let angle = threers_physics::joint::axial_angle(
        world.body(nut).unwrap().rotation(),
        world.body(bolt).unwrap().rotation(),
        axis,
    );
    let lead = 2.0 / std::f32::consts::TAU;
    assert!(
        (travel - lead * angle).abs() < 1e-2,
        "{travel} of travel is not {angle} rad through a {lead} lead"
    );
}

#[test]
fn a_screw_that_is_held_cannot_turn() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let nut = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.2, 0.2, 0.2)));
    let bolt = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cylinder(0.4, 0.08))
            .can_sleep(false),
    );
    let axis = Vector3::new(0.0, 1.0, 0.0);
    let screw = world.add_joint(
        Joint::screw(
            world.bodies(),
            nut,
            bolt,
            Vector3::ZERO,
            Vector3::ZERO,
            axis,
            axis,
            0.3,
        )
        .unwrap()
        // Bottomed out: no travel available in either direction.
        .with_limits(0.0, 0.0)
        .with_motor(5.0, 200.0),
    );
    settle(&mut world, 180);

    // With the travel blocked, the thread blocks the rotation too — which is
    // what "bottomed out" means, and what a coupling buys over two joints.
    let spin = world.body(bolt).unwrap().angular_velocity.dot(axis);
    assert!(spin.abs() < 0.5, "the bolt kept turning at {spin} rad/s");
    assert!(world.joint_offset(screw).unwrap().abs() < 0.05);
}
