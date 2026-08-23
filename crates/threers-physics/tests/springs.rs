//! Elastic joints: a hinge with somewhere it would rather be.
//!
//! The property that matters is not that a spring pulls — an explicit torque
//! does that — but that a stiffness far too large to integrate forward still
//! behaves. That is what makes a chain of these usable as a flexible rod.

use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;

fn settle(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step_fixed();
    }
    let _ = DT;
}

/// A rod hinged at the world origin about Z, sticking out along +X with its
/// centre of mass `arm` away. Gravity bends it down.
fn cantilever(mass: f32, arm: f32, spring: Option<JointSpring>) -> (World, BodyId, JointId) {
    let mut world = World::new();
    let base = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.02, 0.02, 0.02)));
    let rod = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(arm, 0.02, 0.02))
            .mass(mass)
            .translation(Vector3::new(arm, 0.0, 0.0))
            .can_sleep(false),
    );
    let mut hinge = Joint::revolute(
        base,
        rod,
        Vector3::ZERO,
        Vector3::new(-arm, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
    );
    if let Some(s) = spring {
        hinge = hinge.with_spring(s.stiffness, s.rest, s.damping);
    }
    let id = world.add_joint(hinge);
    (world, rod, id)
}

/// Where the rod hangs, as an angle below horizontal.
fn droop(world: &World, rod: BodyId, arm: f32) -> f32 {
    let y = world.body(rod).unwrap().translation().y;
    (-y / arm).clamp(-1.0, 1.0).asin()
}

/// The equilibrium of `k·θ = m·g·L·cos θ`, by bisection.
fn predicted_droop(stiffness: f32, mass: f32, arm: f32) -> f32 {
    let (mut lo, mut hi) = (0.0f32, std::f32::consts::FRAC_PI_2);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if stiffness * mid < mass * G * arm * mid.cos() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn a_sprung_hinge_settles_where_the_torque_balance_says() {
    let (mass, arm, k) = (1.0, 0.5, 50.0);
    let (mut world, rod, _) = cantilever(mass, arm, Some(JointSpring::new(k, 0.0, 2.0)));
    settle(&mut world, 900);

    let measured = droop(&world, rod, arm);
    let expected = predicted_droop(k, mass, arm);
    assert!(
        (measured - expected).abs() < 0.1 * expected,
        "sprung cantilever settled at {measured:.4} rad, beam balance says {expected:.4}"
    );
}

#[test]
fn four_times_the_stiffness_is_a_quarter_of_the_deflection() {
    // The units check: in the small-angle regime θ ≈ mgL/k, so the ratio of two
    // deflections is the inverse ratio of the stiffnesses — and it is that
    // proportionality, not any one number, that says N·m/rad means N·m/rad.
    let (mass, arm) = (1.0, 0.5);
    let mut angles = Vec::new();
    for k in [50.0f32, 200.0] {
        let (mut world, rod, _) = cantilever(mass, arm, Some(JointSpring::new(k, 0.0, 2.0)));
        settle(&mut world, 900);
        angles.push(droop(&world, rod, arm));
    }
    let ratio = angles[0] / angles[1];
    assert!(
        (ratio - 4.0).abs() < 0.4,
        "quadrupling the stiffness changed the deflection by {ratio:.3}x, wanted 4x ({angles:?})"
    );
}

#[test]
#[cfg(feature = "mechanism")] // reads the joint coordinate back
fn a_rest_angle_holds_the_joint_away_from_where_it_was_built() {
    // Built horizontal, told to rest 30° up, and stiff enough that gravity is
    // a rounding error: it should end up near the rest angle, not near zero.
    let rest = std::f32::consts::FRAC_PI_6;
    let (mut world, _rod, hinge) = cantilever(0.2, 0.4, Some(JointSpring::new(500.0, rest, 5.0)));
    settle(&mut world, 900);
    let angle = world.joint_angle(hinge).unwrap();
    assert!(
        (angle - rest).abs() < 0.05,
        "hinge with rest {rest} settled at {angle}"
    );
}

#[test]
#[cfg(feature = "mechanism")] // reads the joint coordinate back
fn a_stiffness_that_would_explode_explicitly_stays_put() {
    // 1e6 N·m/rad on a 10 g link at 60 Hz. Integrated forward as a torque this
    // diverges within a handful of steps — `k·dt²/I` is nine orders of
    // magnitude past the stability limit. Solved as a constraint it is simply
    // very stiff.
    let (mut world, rod, hinge) = cantilever(0.01, 0.05, Some(JointSpring::new(1.0e6, 0.0, 100.0)));
    settle(&mut world, 600);

    let body = world.body(rod).unwrap();
    let (p, v, w) = (
        body.translation(),
        body.linear_velocity,
        body.angular_velocity,
    );
    assert!(
        p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
        "position went non-finite: {p:?}"
    );
    assert!(
        v.length() < 1.0 && w.length() < 1.0,
        "a stiff spring is feeding energy in: v={v:?} w={w:?}"
    );
    let angle = world.joint_angle(hinge).unwrap();
    assert!(
        angle.abs() < 1.0e-3,
        "a 1e6 N·m/rad hinge sagged {angle} rad under 0.01 kg"
    );
}

#[test]
fn damping_alone_slows_a_hinge_and_does_not_hold_it() {
    // The distinction from `Bearing`, which is the other way round: friction
    // can hold a joint still and a damper can only ever slow it down.
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let base = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.02, 0.02, 0.02)));
    let wheel = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cylinder(0.02, 0.2))
            .mass(1.0)
            .can_sleep(false),
    );
    world.add_joint(
        Joint::revolute(
            base,
            wheel,
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::UP,
            Vector3::UP,
        )
        .with_damping(0.5),
    );
    world.body_mut(wheel).unwrap().angular_velocity = Vector3::new(0.0, 10.0, 0.0);

    settle(&mut world, 300);
    let spin = world.body(wheel).unwrap().angular_velocity.y;
    assert!(
        spin.abs() < 1.0,
        "a damped hinge is still turning at {spin} rad/s after 5 s"
    );
    assert!(spin > -0.1, "the damper drove the wheel backwards: {spin}");
}

#[test]
#[cfg(feature = "mechanism")] // reads the joint coordinate back
fn a_sprung_slider_pushes_back_along_its_axis() {
    let mut world = World::new();
    world.gravity = Vector3::new(0.0, -G, 0.0);
    let post = world.add_body(RigidBody::fixed().shape(Shape::cuboid(0.05, 0.05, 0.05)));
    let load = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .mass(2.0)
            .translation(Vector3::new(0.0, -0.5, 0.0))
            .can_sleep(false),
    );
    // A 400 N/m spring under 2 kg should compress by mg/k ≈ 49 mm.
    let slider = world.add_joint(
        Joint::prismatic(
            post,
            load,
            Vector3::ZERO,
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::UP,
            Vector3::UP,
        )
        .with_spring(400.0, 0.0, 30.0),
    );
    settle(&mut world, 900);

    let travel = world.joint_state(slider).unwrap().coordinate.abs();
    let expected = 2.0 * G / 400.0;
    assert!(
        (travel - expected).abs() < 0.2 * expected,
        "a sprung slider settled {travel:.4} m along, Hooke says {expected:.4} m"
    );
}
