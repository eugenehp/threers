//! The generic six-DOF joint and soft constraints.
//!
//! Both are things the presets cannot express: a joint whose axes you pick one
//! at a time, and a joint that gives under load instead of holding rigid.

use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

fn settle(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

/// An anchor and a body hanging half a metre below it.
fn hanging(mass: f32) -> (World, BodyId, BodyId) {
    let mut world = World::new();
    let anchor = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .translation(Vector3::new(0.0, 2.0, 0.0)),
    );
    let load = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .mass(mass)
            .translation(Vector3::new(0.0, 1.5, 0.0))
            .can_sleep(false),
    );
    (world, anchor, load)
}

// ---- generic joint --------------------------------------------------------

#[test]
fn all_axes_locked_is_a_weld() {
    let (mut world, anchor, load) = hanging(1.0);
    world.add_joint(
        Joint::generic(
            world.bodies(),
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
        )
        .unwrap(),
    );
    let start = world.body(load).unwrap().translation();
    settle(&mut world, 240);
    let end = world.body(load).unwrap().translation();
    assert!(
        (end - start).length() < 0.02,
        "a fully locked generic joint let the body move {:?} -> {:?}",
        start,
        end
    );
    // And the orientation is held too.
    let spin = world.body(load).unwrap().angular_velocity.length();
    assert!(spin < 0.05, "the weld is not holding rotation: {spin}");
}

#[test]
fn all_axes_free_constrains_nothing() {
    let (mut world, anchor, load) = hanging(1.0);
    let mut joint = Joint::generic(
        world.bodies(),
        anchor,
        load,
        Vector3::new(0.0, -0.1, 0.0),
        Vector3::new(0.0, 0.4, 0.0),
    )
    .unwrap();
    for axis in 0..3 {
        joint = joint
            .with_linear_dof(axis, Dof::FREE)
            .with_angular_dof(axis, Dof::FREE);
    }
    world.add_joint(joint);

    settle(&mut world, 60);
    let y = world.body(load).unwrap().translation().y;
    // One second of free fall from 1.5 is about -3.4.
    assert!(y < -2.0, "an all-free joint still held the body up at y = {y}");
}

#[test]
fn a_linear_limit_bounds_travel_along_one_axis() {
    let (mut world, anchor, load) = hanging(1.0);
    world.add_joint(
        Joint::generic(
            world.bodies(),
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
        )
        .unwrap()
        // Free to slide down the y axis, but only by 0.3.
        .with_linear_dof(1, Dof::limited(-0.3, 0.0)),
    );

    settle(&mut world, 300);
    let y = world.body(load).unwrap().translation().y;
    assert!(
        (y - 1.2).abs() < 0.05,
        "the body should hang at the bottom of its 0.3 travel (y = 1.2), not {y}"
    );
}

#[test]
fn an_angular_limit_stops_a_swing() {
    let mut world = World::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let post = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .translation(Vector3::new(0.0, 2.0, 0.0)),
    );
    // An arm sticking out sideways, hinged at the post.
    let arm = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.5, 0.05, 0.05))
            .mass(1.0)
            .translation(Vector3::new(0.5, 2.0, 0.0))
            .can_sleep(false),
    );
    world.add_joint(
        Joint::generic(
            world.bodies(),
            post,
            arm,
            Vector3::ZERO,
            Vector3::new(-0.5, 0.0, 0.0),
        )
        .unwrap()
        // Free to rotate about z (the arm swings down), limited to 45°.
        .with_angular_dof(2, Dof::limited(-0.785, 0.785)),
    );

    settle(&mut world, 400);
    let p = world.body(arm).unwrap().translation();
    // 45° down from horizontal puts the tip at y = 2 - 0.5·sin45 ≈ 1.646.
    let drop = 2.0 - p.y;
    assert!(
        (0.25..0.45).contains(&drop),
        "the arm should stop at about 45° (drop 0.354), but dropped {drop}"
    );
}

#[test]
fn a_motor_drives_a_free_axis() {
    let mut world = World::new();
    world.set_gravity(Vector3::ZERO);
    let base = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.3, 0.1, 0.3))
            .translation(Vector3::new(0.0, 1.0, 0.0)),
    );
    let head = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .mass(1.0)
            .translation(Vector3::new(0.0, 1.3, 0.0))
            .can_sleep(false),
    );
    world.add_joint(
        Joint::generic(
            world.bodies(),
            base,
            head,
            Vector3::new(0.0, 0.1, 0.0),
            Vector3::new(0.0, -0.2, 0.0),
        )
        .unwrap()
        .with_angular_dof(1, Dof::FREE.driven(3.0, 100.0)),
    );

    settle(&mut world, 120);
    let spin = world.body(head).unwrap().angular_velocity;
    assert!(
        (spin.y - 3.0).abs() < 0.2,
        "the motor should hold 3 rad/s about y, got {spin:?}"
    );
    // And the other two axes stay locked.
    assert!(
        spin.x.abs() < 0.1 && spin.z.abs() < 0.1,
        "the locked axes leaked: {spin:?}"
    );
}

#[test]
fn a_motor_drives_a_linear_axis() {
    let mut world = World::new();
    world.set_gravity(Vector3::ZERO);
    let rail = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .translation(Vector3::ZERO),
    );
    let carriage = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .mass(2.0)
            .can_sleep(false),
    );
    world.add_joint(
        Joint::generic(world.bodies(), rail, carriage, Vector3::ZERO, Vector3::ZERO)
            .unwrap()
            .with_linear_dof(0, Dof::FREE.driven(1.5, 200.0)),
    );

    settle(&mut world, 120);
    let v = world.body(carriage).unwrap().linear_velocity;
    assert!(
        (v.x - 1.5).abs() < 0.1,
        "the linear motor should hold 1.5 units/s along +x, got {v:?}"
    );
}

#[test]
fn angles_are_measured_from_the_pose_the_joint_was_built_in() {
    // A joint built on already-rotated bodies must read zero, not the world
    // angle between them — otherwise every limit is offset by however the parts
    // happened to be posed.
    let mut world = World::new();
    world.set_gravity(Vector3::ZERO);
    let turn = Quaternion::from_axis_angle(Vector3::UP, 1.0);
    let a = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .rotation(turn),
    );
    let b = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.2, 0.2, 0.2))
            .rotation(turn)
            .translation(Vector3::new(0.0, 0.5, 0.0))
            .can_sleep(false),
    );
    world.add_joint(
        Joint::generic(world.bodies(), a, b, Vector3::new(0.0, 0.25, 0.0), Vector3::new(0.0, -0.25, 0.0))
            .unwrap()
            // A limit of exactly zero: if the rest angle were misread as 1 rad,
            // the joint would immediately snap the body round to correct it.
            .with_angular_dof(1, Dof::limited(0.0, 0.0)),
    );

    settle(&mut world, 120);
    let spin = world.body(b).unwrap().angular_velocity.length();
    assert!(spin < 0.05, "the joint fought a rest pose it should have adopted: {spin}");
}

// ---- softness -------------------------------------------------------------

#[test]
fn a_soft_joint_sags_and_a_rigid_one_does_not() {
    let mut sag = Vec::new();
    for softness in [None, Some(3.0f32)] {
        let (mut world, anchor, load) = hanging(1.0);
        let mut joint = Joint::spherical(
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
        );
        if let Some(f) = softness {
            joint = joint.soft(f, 1.0);
        }
        world.add_joint(joint);
        let start = world.body(load).unwrap().translation().y;
        settle(&mut world, 400);
        sag.push(start - world.body(load).unwrap().translation().y);
    }
    assert!(sag[0].abs() < 0.005, "the rigid joint sagged {}", sag[0]);
    assert!(sag[1] > 0.01, "the soft joint did not give: {}", sag[1]);
}

#[test]
fn sag_matches_the_frequency_that_was_asked_for() {
    // A spring of angular frequency omega holding a mass against gravity settles
    // where k·x = m·g. Since k = m·omega², the mass cancels and x = g/omega².
    // That identity is the whole point of specifying softness as a frequency, so
    // it is worth checking directly.
    for frequency in [2.0f32, 4.0] {
        let (mut world, anchor, load) = hanging(1.0);
        world.add_joint(
            Joint::spherical(
                anchor,
                load,
                Vector3::new(0.0, -0.1, 0.0),
                Vector3::new(0.0, 0.4, 0.0),
            )
            .soft(frequency, 1.0),
        );
        let start = world.body(load).unwrap().translation().y;
        settle(&mut world, 600);
        let sag = start - world.body(load).unwrap().translation().y;

        let omega = std::f32::consts::TAU * frequency;
        let expected = 9.81 / (omega * omega);
        assert!(
            (sag - expected).abs() < expected * 0.25,
            "{frequency} Hz should sag {expected:.4}, got {sag:.4}"
        );
    }
}

#[test]
fn softness_means_the_same_thing_whatever_the_mass() {
    // The reason the API takes a frequency rather than a stiffness: one number
    // has to work for a marble and a shipping container.
    let mut sags = Vec::new();
    for mass in [0.5f32, 50.0] {
        let (mut world, anchor, load) = hanging(mass);
        world.add_joint(
            Joint::spherical(
                anchor,
                load,
                Vector3::new(0.0, -0.1, 0.0),
                Vector3::new(0.0, 0.4, 0.0),
            )
            .soft(3.0, 1.0),
        );
        let start = world.body(load).unwrap().translation().y;
        settle(&mut world, 600);
        sags.push(start - world.body(load).unwrap().translation().y);
    }
    let ratio = sags[0] / sags[1];
    assert!(
        (0.8..1.25).contains(&ratio),
        "a 100x mass change moved the sag from {} to {}",
        sags[0],
        sags[1]
    );
}

#[test]
fn an_underdamped_joint_rings_and_a_damped_one_does_not() {
    let mut swings = Vec::new();
    for damping in [0.05f32, 1.0] {
        let (mut world, anchor, load) = hanging(1.0);
        world.add_joint(
            Joint::spherical(
                anchor,
                load,
                Vector3::new(0.0, -0.1, 0.0),
                Vector3::new(0.0, 0.4, 0.0),
            )
            .soft(3.0, damping),
        );
        // Yank it down, then count direction changes as it recovers.
        world
            .body_mut(load)
            .unwrap()
            .set_linear_velocity(Vector3::new(0.0, -2.0, 0.0));

        let mut reversals = 0;
        let mut previous = -2.0f32;
        for _ in 0..300 {
            world.step(DT);
            let v = world.body(load).unwrap().linear_velocity.y;
            if v.abs() > 0.05 && v.signum() != previous.signum() {
                reversals += 1;
            }
            if v.abs() > 0.05 {
                previous = v;
            }
        }
        swings.push(reversals);
    }
    assert!(
        swings[0] > swings[1],
        "damping 0.05 rang {} times, damping 1.0 rang {} — damping did nothing",
        swings[0],
        swings[1]
    );
    assert!(
        swings[1] <= 2,
        "a critically damped joint should barely overshoot, but reversed {} times",
        swings[1]
    );
}

#[test]
fn a_soft_joint_still_returns_to_its_rest_pose() {
    // Giving under load is fine; failing to recover is a broken joint.
    let (mut world, anchor, load) = hanging(1.0);
    world.set_gravity(Vector3::ZERO);
    world.add_joint(
        Joint::spherical(
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
        )
        .soft(4.0, 0.8),
    );
    world
        .body_mut(load)
        .unwrap()
        .apply_impulse(Vector3::new(3.0, 0.0, 2.0));

    // Recovery is measured on the joint, not on the load's position, because a
    // spherical joint does not determine a position. It pins one point and
    // leaves rotation free, so with gravity off every point on the 0.4 sphere
    // around the pivot satisfies it equally, and the kick leaves the load
    // orbiting there: angular momentum about the pivot is conserved because the
    // constraint force acts *at* the pivot and exerts no torque about it.
    // Asking the load to come back to where it started is asking the joint to
    // undo a rotation it never resisted.
    let pivot = Vector3::new(0.0, 1.9, 0.0);
    let separation = |w: &World| {
        let b = w.body(load).unwrap();
        (b.position.transform_point(Vector3::new(0.0, 0.4, 0.0)) - pivot).length()
    };

    settle(&mut world, 100);
    let early = separation(&world);
    settle(&mut world, 500);
    let late = separation(&world);

    assert!(
        late < 0.02,
        "the soft joint was still pulled {late:.4} apart after 10 s"
    );
    assert!(
        late < early,
        "the joint gave {early:.4} under the kick and never took it back ({late:.4})"
    );
    // Still on the sphere the joint defines, not dragged off it.
    let radius = (world.body(load).unwrap().translation() - pivot).length();
    assert!(
        (radius - 0.4).abs() < 0.02,
        "the load left its constraint sphere: {radius:.4} from the pivot, wanted 0.4"
    );
}

#[test]
fn softness_does_not_disturb_a_rigid_joint() {
    // `Softness::RIGID` has to reproduce the old path exactly, not approximately:
    // it is the default, so any drift here would change every existing scene.
    let build = |soft: bool| {
        let (mut world, anchor, load) = hanging(1.0);
        let mut j = Joint::revolute(
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        if soft {
            j = j.soft(0.0, 0.0); // explicitly rigid
        }
        world.add_joint(j);
        settle(&mut world, 200);
        world.body(load).unwrap().translation()
    };
    assert_eq!(build(false), build(true));
}

#[test]
fn a_soft_generic_joint_gives_on_its_locked_axes() {
    let (mut world, anchor, load) = hanging(1.0);
    world.add_joint(
        Joint::generic(
            world.bodies(),
            anchor,
            load,
            Vector3::new(0.0, -0.1, 0.0),
            Vector3::new(0.0, 0.4, 0.0),
        )
        .unwrap()
        // Two locks and one limit, so the coupled point solve is not used and
        // the per-axis path has to carry the softness itself. The limit goes on
        // a horizontal axis: it is here to break the all-locked fast path, not
        // to carry the load, and a limit on the axis gravity pulls along would
        // measure the limit stopping the fall rather than the softness holding
        // it.
        .with_linear_dof(0, Dof::limited(-1.0, 1.0))
        .soft(3.0, 1.0),
    );
    let start = world.body(load).unwrap().translation().y;
    settle(&mut world, 600);
    let sag = start - world.body(load).unwrap().translation().y;
    let expected = 9.81 / (std::f32::consts::TAU * 3.0f32).powi(2);
    assert!(
        (sag - expected).abs() < expected * 0.4,
        "expected about {expected:.4} of sag, got {sag:.4}"
    );
}

