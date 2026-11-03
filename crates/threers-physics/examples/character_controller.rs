//! A kinematic character walking, sliding along walls and climbing a ramp.
//!
//! ```text
//! cargo run -p threers-physics --example character_controller
//! ```

use threers_physics::prelude::*;

fn main() {
    let mut world = World::new();

    // Floor.
    world.add_body(RigidBody::fixed().shape(Shape::ground()));

    // A wall running along z at x = 3.
    world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.2, 2.0, 8.0))
            .translation(Vector3::new(3.0, 2.0, 0.0)),
    );

    // A 30 degree ramp to walk up, off to the side.
    world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(3.0, 0.2, 2.0))
            .translation(Vector3::new(-4.0, 1.0, 0.0))
            .rotation(threers_physics::prelude::Quaternion::from_axis_angle(
                Vector3::new(0.0, 0.0, 1.0),
                30f32.to_radians(),
            )),
    );

    // A capsule 1.6 m tall: 0.6 half-height plus 0.3 radius at each end.
    let controller = CharacterController::new(Shape::capsule(0.6, 0.3))
        .max_slope_degrees(45.0)
        .snap_to_ground(0.3);

    let mut position = Isometry::from_translation(Vector3::new(0.0, 1.0, 0.0));
    let mut vertical_speed = 0.0f32;
    let dt = 1.0 / 60.0;

    println!("-- walking into a wall --");
    // Walk diagonally at the wall: x is blocked, z should keep going.
    for frame in 0..120 {
        vertical_speed -= 9.81 * dt;
        let wish = Vector3::new(4.0, 0.0, 2.0) * dt + Vector3::new(0.0, vertical_speed * dt, 0.0);

        let result = controller.move_and_slide(&world, position, wish, QueryFilter::default());
        position.translation = position.translation + result.translation;
        if result.grounded {
            vertical_speed = 0.0;
        }

        if frame % 40 == 39 {
            println!(
                "  t={:.2}s  pos ({:.2}, {:.2}, {:.2})  grounded={}  wall={}",
                (frame + 1) as f32 * dt,
                position.translation.x,
                position.translation.y,
                position.translation.z,
                result.grounded,
                result.hit_wall
            );
        }
    }
    println!(
        "  stopped at x={:.2} (the wall is at 3.0) but slid to z={:.2}",
        position.translation.x, position.translation.z
    );

    println!("\n-- jumping --");
    vertical_speed = 5.0; // an impulse straight up
    let mut peak = position.translation.y;
    for _ in 0..120 {
        vertical_speed -= 9.81 * dt;
        let wish = Vector3::new(0.0, vertical_speed * dt, 0.0);
        let result = controller.move_and_slide(&world, position, wish, QueryFilter::default());
        position.translation = position.translation + result.translation;
        peak = peak.max(position.translation.y);
        if result.grounded && vertical_speed < 0.0 {
            vertical_speed = 0.0;
        }
    }
    println!("  jumped to y={peak:.2}, landed back at y={:.2}", position.translation.y);

    println!("\n-- climbing a 30 degree ramp --");
    position = Isometry::from_translation(Vector3::new(-1.0, 1.0, 0.0));
    vertical_speed = 0.0;
    for _ in 0..240 {
        vertical_speed -= 9.81 * dt;
        // Walk left at 2 units/s, falling all the while.
        let wish = Vector3::new(-2.0 * dt, vertical_speed * dt, 0.0);
        let result = controller.move_and_slide(&world, position, wish, QueryFilter::default());
        position.translation = position.translation + result.translation;
        if result.grounded {
            vertical_speed = 0.0;
        }
    }
    println!(
        "  walked to ({:.2}, {:.2}) — the ramp lifted the character {:.2} m",
        position.translation.x,
        position.translation.y,
        position.translation.y - 1.0
    );

    let drop = controller.ground_distance(&world, position, 100.0, QueryFilter::default());
    println!("  ground is {:.2} m below", drop.unwrap_or(f32::NAN));
}
