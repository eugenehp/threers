//! The behaviours a kinematically-driven robot arm depends on.
//!
//! `examples/robot_arm.rs` builds a working cell out of these; this file checks
//! each of them on its own, so that when the example breaks it is obvious which
//! piece went. They are the four claims an arm rests on:
//!
//! - A kinematic link **moves what it touches**, and is **not moved back**.
//! - IK **reaches** a target, **keeps the links the length they are**, and says
//!   so when it cannot.
//! - Joint limits **hold during the solve**, not merely at the end of it.
//! - A joint made at runtime **carries a load and lets go of it**.

use std::f32::consts::PI;
use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

// ---- kinematic links ------------------------------------------------------

#[test]
fn a_kinematic_link_sweeps_a_dynamic_part_along() {
    // The property that makes the whole approach work. A servo with enough
    // torque moves the load; the load does not move the servo.
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.6));
    let part = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .mass(2.0)
            .translation(Vector3::new(0.0, 0.1, 0.0))
            .friction(0.6)
            .can_sleep(false),
    );
    let paddle = world.add_body(
        RigidBody::kinematic()
            .shape(Shape::cuboid(0.05, 0.3, 0.4))
            .translation(Vector3::new(-0.6, 0.3, 0.0))
            .friction(0.9),
    );

    // Sweep the paddle steadily through where the part is sitting.
    for step in 0..180 {
        let x = -0.6 + step as f32 * 0.005;
        world
            .body_mut(paddle)
            .unwrap()
            .set_kinematic_target(Isometry::from_translation(Vector3::new(x, 0.3, 0.0)));
        world.step(DT);
    }

    let moved = world.body(part).unwrap().translation().x;
    assert!(
        moved > 0.15,
        "the paddle swept past the part and left it at x = {moved}"
    );
    // And the paddle went exactly where it was told, regardless of the load.
    let paddle_x = world.body(paddle).unwrap().translation().x;
    assert!(
        (paddle_x - 0.295).abs() < 0.01,
        "a 2 kg part pushed the arm off its commanded path, to x = {paddle_x}"
    );
}

#[test]
fn a_kinematic_link_is_not_shoved_by_what_lands_on_it() {
    let mut world = World::new();
    let shelf = world.add_body(
        RigidBody::kinematic()
            .shape(Shape::cuboid(0.5, 0.05, 0.5))
            .translation(Vector3::new(0.0, 1.0, 0.0)),
    );
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .mass(500.0)
            .translation(Vector3::new(0.0, 3.0, 0.0))
            .can_sleep(false),
    );
    let before = world.body(shelf).unwrap().translation();
    for _ in 0..240 {
        world.step(DT);
    }
    let after = world.body(shelf).unwrap().translation();
    assert_eq!(
        before, after,
        "half a tonne dropped on a kinematic link and it gave way"
    );
}

// ---- inverse kinematics ---------------------------------------------------

/// A three-link arm rooted at the origin, reaching along +X.
fn arm() -> IkChain {
    let mut chain = IkChain::from_points(&[
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.8, 0.0, 0.0),
        Vector3::new(1.5, 0.0, 0.0),
        Vector3::new(1.72, 0.0, 0.0),
    ]);
    chain.max_iterations = 40;
    chain.tolerance = 0.002;
    chain
}

fn link_lengths(chain: &IkChain) -> Vec<f32> {
    (0..chain.joints.len() - 1)
        .map(|i| (chain.joints[i + 1].position - chain.joints[i].position).length())
        .collect()
}

#[test]
fn the_tool_reaches_targets_all_over_the_workspace() {
    let original = link_lengths(&arm());
    let mut worst_error: f32 = 0.0;
    let mut worst_stretch: f32 = 0.0;

    // A sweep through the reachable volume rather than a handful of poses: a
    // solver can look fine on the poses someone chose and stall in between.
    for i in 0..12 {
        for j in 0..8 {
            let yaw = i as f32 / 12.0 * 2.0 * PI;
            let pitch = j as f32 / 8.0 * PI - PI * 0.5;
            let radius = 1.0;
            let target = Vector3::new(
                yaw.cos() * pitch.cos() * radius,
                pitch.sin() * radius,
                yaw.sin() * pitch.cos() * radius,
            );

            let mut chain = arm();
            let result = chain.solve_ccd(target);
            worst_error = worst_error.max((chain.tip() - target).length());
            for (k, length) in link_lengths(&chain).iter().enumerate() {
                worst_stretch = worst_stretch.max((length - original[k]).abs());
            }
            assert!(!result.out_of_reach, "{target:?} is 1.0 from a 1.72 reach");
        }
    }

    assert!(
        worst_error < 0.01,
        "the worst target was missed by {:.1} mm",
        worst_error * 1000.0
    );
    assert!(
        worst_stretch < 1e-4,
        "a link changed length by {worst_stretch} m — the arm is stretching, not bending"
    );
}

#[test]
fn a_target_beyond_the_arm_is_reported_rather_than_faked() {
    // Reaching for something too far away is a normal thing for a programme to
    // ask. The wrong answers are pretending it worked, and stretching the arm.
    let mut chain = arm();
    let original = link_lengths(&chain);
    let far = Vector3::new(5.0, 0.0, 0.0);
    let result = chain.solve_ccd(far);

    assert!(result.out_of_reach, "5 m is outside a 1.72 m reach");
    assert!(!result.reached);
    assert!(
        (result.error - (5.0 - 1.72)).abs() < 0.05,
        "reported a shortfall of {:.2}, expected about 3.28",
        result.error
    );
    for (k, length) in link_lengths(&chain).iter().enumerate() {
        assert!(
            (length - original[k]).abs() < 1e-4,
            "link {k} stretched from {} to {length} trying to reach",
            original[k]
        );
    }
}

#[test]
fn a_hinge_limit_holds_through_the_whole_solve() {
    // Checking the limit only at the end would miss a solver that swings the
    // joint through the forbidden range and happens to come back.
    let mut chain = arm();
    let (min, max) = (0.3f32, 2.2f32);
    chain.set_constraint(
        1,
        IkConstraint::hinge_limited(Vector3::new(0.0, 0.0, 1.0), min, max),
    );

    let mut worst_low = f32::MAX;
    let mut worst_high = f32::MIN;
    for i in 0..40 {
        // Targets that would fold the elbow right up if nothing stopped them.
        let a = i as f32 / 40.0 * 2.0 * PI;
        let target = Vector3::new(a.cos() * 0.45, a.sin() * 0.45, 0.0);
        chain.solve_ccd(target);

        let upper = chain.joints[1].position - chain.joints[0].position;
        let fore = chain.joints[2].position - chain.joints[1].position;
        let bend = upper
            .normalize()
            .dot(fore.normalize())
            .clamp(-1.0, 1.0)
            .acos();
        worst_low = worst_low.min(bend);
        worst_high = worst_high.max(bend);
    }

    // A little slack: the constraint is applied per iteration, so the final
    // pose can sit a hair outside while converging.
    assert!(
        worst_low > min - 0.05,
        "the elbow closed to {worst_low:.2} rad, past its {min} limit"
    );
    assert!(
        worst_high < max + 0.05,
        "the elbow opened to {worst_high:.2} rad, past its {max} limit"
    );
}

#[test]
fn the_same_commanded_pose_gives_the_same_arm_every_time() {
    // Repeatability in joint space, which is what a controller's home position
    // actually is. Reaching for the same *point* need not repeat — a redundant
    // arm has a family of poses that all satisfy it — but replaying the same
    // configuration must.
    let reference = arm();
    let pose: Vec<Vector3> = reference.joints.iter().map(|j| j.position).collect();

    for _ in 0..5 {
        let mut chain = arm();
        chain.solve_ccd(Vector3::new(0.3, 1.2, 0.4)); // wander off
        for (joint, p) in chain.joints.iter_mut().zip(&pose) {
            joint.position = *p;
        }
        for (k, joint) in chain.joints.iter().enumerate() {
            assert_eq!(joint.position, pose[k], "joint {k} did not come back exactly");
        }
    }
}

// ---- gripping -------------------------------------------------------------

#[test]
fn a_joint_made_at_runtime_carries_a_load_and_lets_it_go() {
    // How a vacuum cup or an electromagnet behaves: it holds whatever it met
    // until told otherwise, and the load's mass keeps mattering while it does.
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
    // The part rests on the ground and the tool comes down onto it, so the
    // grip is made where the two actually touch. The heights are not free: a
    // part left in mid-air falls during the settle below, and the joint would
    // then be built around wherever it landed rather than where it was drawn.
    // 0.08 half-extent resting on the ground puts its top at 0.16; a 0.06 ball
    // whose underside sits on that has its centre at 0.22.
    let tool = world.add_body(
        RigidBody::kinematic()
            .shape(Shape::ball(0.06))
            .translation(Vector3::new(0.0, 0.22, 0.0)),
    );
    let part = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.08, 0.08, 0.08))
            .mass(1.0)
            .translation(Vector3::new(0.0, 0.08, 0.0))
            .can_sleep(false),
    );
    for _ in 0..30 {
        world.step(DT);
    }

    let grip = world
        .add_joint(Joint::fixed_at_point(world.bodies(), tool, part, Vector3::new(0.0, 0.16, 0.0)).unwrap());
    let grip_offset = {
        let (t, p) = (world.body(tool).unwrap(), world.body(part).unwrap());
        (p.translation() - t.translation()).length()
    };

    // Lift and traverse.
    for step in 0..150 {
        let t = step as f32 / 150.0;
        world
            .body_mut(tool)
            .unwrap()
            .set_kinematic_target(Isometry::from_translation(Vector3::new(
                t * 1.2,
                0.22 + t * 1.0,
                0.0,
            )));
        world.step(DT);
    }

    let carried = world.body(part).unwrap().translation();
    assert!(
        carried.y > 1.0 && carried.x > 1.0,
        "the part did not come along: it is at {carried:?}"
    );
    // It should still be exactly where the grip put it, not swinging or
    // sagging: measured against the offset the joint was built with, since that
    // is what a rigid grip promises to keep.
    let tool_at = world.body(tool).unwrap().translation();
    let held = (carried - tool_at).length();
    assert!(
        (held - grip_offset).abs() < 0.02,
        "the grip was made at {grip_offset:.3} and is now {held:.3}, so it has stretched"
    );

    world.remove_joint(grip);
    for _ in 0..240 {
        world.step(DT);
    }
    let dropped = world.body(part).unwrap().translation();
    assert!(
        dropped.y < 0.2,
        "the part was released at height {:.2} and is still at {:.2}",
        carried.y,
        dropped.y
    );
}

#[test]
fn releasing_hands_the_load_its_momentum() {
    // A part let go mid-swing should carry on, not stop dead. Otherwise a
    // place-on-the-move programme puts everything short of where it aimed.
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let tool = world.add_body(RigidBody::kinematic().shape(Shape::ball(0.06)));
    let part = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.05, 0.05, 0.05))
            .mass(0.5)
            .translation(Vector3::new(0.0, -0.12, 0.0))
            .can_sleep(false),
    );
    let grip = world
        .add_joint(Joint::fixed_at_point(world.bodies(), tool, part, Vector3::new(0.0, -0.06, 0.0)).unwrap());

    for step in 0..90 {
        let x = step as f32 * 0.02;
        world
            .body_mut(tool)
            .unwrap()
            .set_kinematic_target(Isometry::from_translation(Vector3::new(x, 0.0, 0.0)));
        world.step(DT);
    }
    let speed = world.body(part).unwrap().linear_velocity.x;
    assert!(
        speed > 0.8,
        "the carried part is only doing {speed} units/s while the tool does 1.2"
    );

    world.remove_joint(grip);
    let at_release = world.body(part).unwrap().translation();
    for _ in 0..60 {
        world.step(DT);
    }
    let coasted = world.body(part).unwrap().translation().x - at_release.x;
    assert!(
        coasted > 0.5,
        "the part stopped dead on release, coasting only {coasted}"
    );
}


