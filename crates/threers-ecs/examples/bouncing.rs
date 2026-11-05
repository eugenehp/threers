//! Balls falling under a fixed step, removed once they stop, with no window.
//!
//! ```text
//! cargo run -p threers-ecs --example bouncing
//! ```

use threers_ecs::prelude::*;

struct Velocity(Vector3);
impl Component for Velocity {}

struct Bounces(u32);
impl Component for Bounces {}

#[derive(Default)]
struct Removed(u32);

fn gravity(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.each::<&mut Velocity>(|velocity| velocity.0.y -= 9.81 * dt);
}

fn movement(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.each::<(Node, &Velocity)>(|(node, velocity)| {
        node.position = node.position + velocity.0 * dt;
    });
}

fn bounce(w: &mut World) {
    w.each::<(Node, &mut Velocity, &mut Bounces)>(|(node, velocity, bounces)| {
        if node.position.y < 0.0 {
            node.position.y = 0.0;
            velocity.0.y *= -0.6;
            bounces.0 += 1;
        }
    });
}

fn remove_tired(w: &mut World) {
    let commands = w.commands();
    w.with_resource::<Removed, _>(|w, removed| {
        w.each::<(Entity, &Bounces)>(|(id, bounces)| {
            if bounces.0 >= 3 {
                commands.despawn(id);
                removed.0 += 1;
            }
        });
    });
}

fn main() {
    let mut app = App::new();
    app.resource(Removed::default())
        .fixed_system(remove_tired)
        .fixed_system(System::new(bounce).before(remove_tired).after(movement))
        .fixed_system(System::new(movement).after(gravity))
        .fixed_system(gravity);

    for i in 0..5 {
        let ball = app.world.spawn(Object3D::group());
        app.world.scene.get_mut(ball).unwrap().position.y = 1.0 + i as f32;
        app.world
            .insert(ball, (Velocity(Vector3::ZERO), Bounces(0)));
    }

    for frame in 0..600 {
        app.frame(1.0 / 60.0);
        if frame % 60 == 0 {
            let left = app.world.entities::<With<Bounces>>().len();
            println!("{:>4.1}s  {left} balls", frame as f32 / 60.0);
        }
    }
    println!("removed {}", app.world.resource::<Removed>().0);
}
