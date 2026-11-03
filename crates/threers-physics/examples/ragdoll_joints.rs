//! Joints: a hinged bridge, a motorised arm, a rope, and a breakable link.
//!
//! ```text
//! cargo run -p threers-physics --example ragdoll_joints
//! ```

use std::f32::consts::FRAC_PI_2;
use threers_physics::prelude::*;

fn main() {
    hinged_bridge();
    motorised_arm();
    rope_and_break();
}

/// A chain of planks hinged end to end, anchored at both ends.
fn hinged_bridge() {
    println!("-- hinged bridge --");
    let mut world = World::new();

    let plank_half = 0.5f32;
    let count = 8;
    let mut planks = Vec::new();

    // Anchors carry no collider: they exist only to pin the ends.
    let left = world.add_body(RigidBody::fixed().translation(Vector3::new(-4.5, 3.0, 0.0)));
    let right = world.add_body(RigidBody::fixed().translation(Vector3::new(
        -4.5 + count as f32 * plank_half * 2.0,
        3.0,
        0.0,
    )));

    for i in 0..count {
        planks.push(world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(plank_half, 0.08, 0.6))
                .translation(Vector3::new(
                    -4.5 + plank_half + i as f32 * plank_half * 2.0,
                    3.0,
                    0.0,
                ))
                .can_sleep(false),
        ));
    }

    // Hinge each plank to the previous one about z, at their shared edge.
    let mut previous = left;
    let mut anchor_on_previous = Vector3::ZERO;
    for &plank in &planks {
        world.add_joint(Joint::revolute(
            previous,
            plank,
            anchor_on_previous,
            Vector3::new(-plank_half, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ));
        previous = plank;
        anchor_on_previous = Vector3::new(plank_half, 0.0, 0.0);
    }
    world.add_joint(Joint::revolute(
        previous,
        right,
        anchor_on_previous,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
    ));

    for _ in 0..300 {
        world.step(1.0 / 60.0);
    }
    let sag = planks
        .iter()
        .filter_map(|id| world.body(*id))
        .map(|b| b.translation().y)
        .fold(f32::MAX, f32::min);
    println!("  the bridge sags to y = {sag:.2} (anchored at y = 3.0)\n");
}

/// A hinge with a motor and end stops — a robot joint.
fn motorised_arm() {
    println!("-- motorised arm --");
    let mut world = World::new();
    let base = world.add_body(RigidBody::fixed());
    let arm = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.6, 0.06, 0.06))
            .translation(Vector3::new(0.6, 0.0, 0.0))
            // Ignore gravity so the motor's behaviour is the only thing on show.
            .gravity_scale(0.0)
            .can_sleep(false),
    );

    let hinge = world.add_joint(
        Joint::revolute(
            base,
            arm,
            Vector3::ZERO,
            Vector3::new(-0.6, 0.0, 0.0),
            Vector3::UP,
            Vector3::UP,
        )
        // Drive at 1.5 rad/s, but stop at 90 degrees.
        .with_motor(1.5, 200.0)
        .with_limits(0.0, FRAC_PI_2),
    );

    for frame in 0..180 {
        world.step(1.0 / 60.0);
        if frame % 45 == 44 {
            let q = world.body(arm).unwrap().rotation();
            println!("  t = {:.2}s  angle = {:.3} rad", (frame + 1) as f32 / 60.0, 2.0 * q.y.atan2(q.w));
        }
    }

    // Reverse the motor.
    if let Some(JointKind::Revolute { motor, .. }) = world.joint_mut(hinge).map(|j| &mut j.kind) {
        *motor = Some(Motor::new(-1.5, 200.0));
    }
    world.wake_all();
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }
    let q = world.body(arm).unwrap().rotation();
    println!("  after reversing: angle = {:.3} rad\n", 2.0 * q.y.atan2(q.w));
}

/// A rope that only resists stretching, and a link that snaps under load.
fn rope_and_break() {
    println!("-- rope and breakable link --");
    let mut world = World::new();
    let hook = world.add_body(RigidBody::fixed().translation(Vector3::new(0.0, 5.0, 0.0)));

    let weight = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.3))
            .translation(Vector3::new(0.0, 4.5, 0.0))
            .mass(5.0)
            .can_sleep(false),
    );
    world.add_joint(Joint::rope(hook, weight, Vector3::ZERO, Vector3::ZERO, 2.0));

    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }
    let drop = 5.0 - world.body(weight).unwrap().translation().y;
    println!("  the rope let the weight fall {drop:.2} m, then held (max 2.0)");

    // Now hang something far too heavy from a link rated for a small impulse.
    let anchor = world.add_body(RigidBody::fixed().translation(Vector3::new(5.0, 5.0, 0.0)));
    let anvil = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.3))
            .translation(Vector3::new(5.0, 4.0, 0.0))
            .mass(400.0)
            .can_sleep(false),
    );
    let link = world.add_joint(
        Joint::distance(anchor, anvil, Vector3::ZERO, Vector3::ZERO, 1.0).breakable(2.0),
    );

    for frame in 0..120 {
        world.step(1.0 / 60.0);
        if world.joint(link).is_some_and(|j| j.broken) {
            println!("  the link snapped after {} frames", frame + 1);
            break;
        }
    }
    println!("  the anvil is now at y = {:.2}", world.body(anvil).unwrap().translation().y);
}
