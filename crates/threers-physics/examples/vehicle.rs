//! A car driving a lap: throttle, steering, brakes and a ramp.
//!
//! ```text
//! cargo run -p threers-physics --example vehicle
//! ```
//!
//! Prints a trace of what the car is doing each half-second. The point is to
//! show the shape of the API — build a chassis, bolt on four wheels, feed it
//! three numbers a frame — and to show that the interesting behaviour
//! (weight transfer, a wheel leaving the ground on a ramp, the tyres running
//! out of grip in a corner) falls out rather than being scripted.

use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

fn main() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(1.0));

    // A ramp to launch off, a little way down the road.
    world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(4.0, 0.5, 3.0))
            .translation(Vector3::new(0.0, 0.0, -30.0))
            .rotation(Quaternion::from_axis_angle(
                Vector3::new(1.0, 0.0, 0.0),
                -0.18,
            ))
            .friction(1.0),
    );

    let chassis = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.9, 0.4, 2.0))
            .mass(1200.0)
            .translation(Vector3::new(0.0, 1.0, 0.0))
            .can_sleep(false),
    );

    let mut car = Vehicle::new(chassis);
    for (x, z, front) in [
        (-0.8f32, 1.4f32, true),
        (0.8, 1.4, true),
        (-0.8, -1.4, false),
        (0.8, -1.4, false),
    ] {
        car.add_wheel(
            WheelConfig::new(Vector3::new(x, -0.2, z), 0.35)
                .steering(front)   // front wheels turn
                .powered(!front),  // rear wheels drive
        );
    }

    println!(" time  speed  wheels  steer  what it is doing");
    println!("─────────────────────────────────────────────────────────");

    for frame in 0..900 {
        let t = frame as f32 * DT;

        // A little script: accelerate, corner, brake, accelerate again.
        let doing = match frame {
            0..=120 => {
                car.set_drive(0.0);
                car.set_brake(0.0);
                "settling on its suspension"
            }
            121..=420 => {
                car.set_drive(5000.0);
                car.set_brake(0.0);
                car.set_steering(0.0);
                "accelerating, straight — into the ramp"
            }
            421..=600 => {
                car.set_drive(3000.0);
                car.set_steering(0.35);
                "cornering under power"
            }
            601..=750 => {
                car.set_drive(0.0);
                car.set_brake(8000.0);
                car.set_steering(0.0);
                "braking hard"
            }
            _ => {
                car.set_brake(0.0);
                car.set_drive(2000.0);
                "trundling away"
            }
        };

        car.update(&mut world, DT);
        world.step(DT);

        if frame % 30 == 0 {
            let body = world.body(chassis).unwrap();
            let slipping = car.wheels.iter().filter(|w| w.slip > 0.05).count();
            let note = if car.wheels_on_ground() < 4 {
                " — airborne!"
            } else if slipping > 0 {
                " — tyres sliding"
            } else {
                ""
            };
            println!(
                "{t:5.1}s {:6.1} {:^7} {:6.2}  {doing}{note}",
                car.speed(&world),
                format!("{}/4", car.wheels_on_ground()),
                car.wheels[0].steering,
            );
            let _ = body;
        }
    }

    let end = world.body(chassis).unwrap().translation();
    println!("\nEnded at ({:.1}, {:.1}, {:.1}).", end.x, end.y, end.z);
    println!(
        "Suspension travel at rest: {:.3} of {:.3} available.",
        car.wheels[0].compression(),
        car.wheels[0].config.max_travel
    );
}
