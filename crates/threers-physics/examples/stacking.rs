//! Stacking, friction and sleeping.
//!
//! Builds a pyramid of boxes, lets it settle, then fires a heavy sphere into it
//! and watches the sleeping bodies wake up.
//!
//! ```text
//! cargo run -p threers-physics --example stacking
//! ```

use threers_physics::prelude::*;

fn main() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.9));

    // A pyramid: five boxes on the bottom row, one fewer each row up.
    let half = 0.5;
    let mut boxes = Vec::new();
    for row in 0..5 {
        let count = 5 - row;
        for i in 0..count {
            let x = (i as f32 - (count - 1) as f32 * 0.5) * (half * 2.05);
            let y = half + row as f32 * half * 2.0;
            boxes.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(half, half, half))
                    .translation(Vector3::new(x, y, 0.0))
                    .friction(0.9),
            ));
        }
    }
    println!("built a pyramid of {} boxes", boxes.len());

    // Settle. Anything that stops moving for half a second falls asleep and
    // stops costing anything.
    for _ in 0..240 {
        world.step(1.0 / 60.0);
    }
    println!(
        "after 4 s: {} of {} boxes asleep",
        world.sleeping_count(),
        world.body_count()
    );
    report_heights(&world, &boxes);

    // Fire a cannonball at it. The impact wakes whatever it touches, and that
    // wake spreads outward through the contacts.
    println!("\nfiring a 50 kg sphere at the stack...");
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.4))
            .translation(Vector3::new(-6.0, 1.5, 0.0))
            .linear_velocity(Vector3::new(18.0, 0.0, 0.0))
            .mass(50.0)
            // Small and fast: sweep it instead of teleporting it each step.
            .ccd(true),
    );

    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }
    println!(
        "half a second later: {} of {} asleep",
        world.sleeping_count(),
        world.body_count()
    );

    for _ in 0..240 {
        world.step(1.0 / 60.0);
    }
    println!("\nafter the dust settles:");
    report_heights(&world, &boxes);
}

fn report_heights(world: &World, boxes: &[BodyId]) {
    let heights: Vec<f32> = boxes
        .iter()
        .filter_map(|id| world.body(*id))
        .map(|b| b.translation().y)
        .collect();
    let highest = heights.iter().copied().fold(f32::MIN, f32::max);
    let toppled = heights.iter().filter(|&&y| y < 0.4).count();
    println!("  highest box at y = {highest:.2}, {toppled} on the floor");
}
