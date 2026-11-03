//! Gravity and restitution — the "hello world" of a physics engine.
//!
//! Drops three balls of different bounciness onto a ground plane and prints how
//! high each one comes back up.
//!
//! ```text
//! cargo run -p threers-physics --example bouncing_balls
//! ```

use threers_physics::prelude::*;

fn main() {
    let mut world = World::new();

    // An infinite ground plane at y = 0. `Shape::ground()` is
    // `half_space(Vector3::UP)`: solid everywhere below the surface.
    world.add_body(RigidBody::fixed().shape(Shape::ground()));

    let materials = [
        ("dead     ", 0.0),
        ("rubbery  ", 0.6),
        ("superball", 0.95),
    ];

    let balls: Vec<(&str, BodyId)> = materials
        .iter()
        .enumerate()
        .map(|(i, &(name, restitution))| {
            let body = world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.5))
                    .translation(Vector3::new(i as f32 * 2.0, 5.0, 0.0))
                    .restitution(restitution)
                    // Bouncing bodies must not doze off between bounces.
                    .can_sleep(false),
            );
            (name, body)
        })
        .collect();

    // Track the highest point each ball reaches *after* its first landing.
    let mut landed = vec![false; balls.len()];
    let mut rebound = vec![0.0f32; balls.len()];

    // Five seconds at 60 Hz. `step` takes real frame time and consumes it in
    // fixed internal steps, so this is exactly five seconds of simulation.
    for _ in 0..300 {
        world.step(1.0 / 60.0);

        for (i, (_, id)) in balls.iter().enumerate() {
            let y = world.body(*id).unwrap().translation().y;
            if y < 0.55 {
                landed[i] = true;
            }
            if landed[i] {
                rebound[i] = rebound[i].max(y - 0.5);
            }
        }
    }

    println!("dropped from 4.5 m above the ground:\n");
    for (i, (name, id)) in balls.iter().enumerate() {
        let body = world.body(*id).unwrap();
        println!(
            "  {name}  restitution {:.2}  bounced back {:.2} m  resting at y = {:.3}",
            materials[i].1,
            rebound[i],
            body.translation().y
        );
    }

    println!("\n{} bodies, {} asleep", world.body_count(), world.sleeping_count());
}
