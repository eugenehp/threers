//! Animation drives a kinematic capsule; [`PhysicsBlend`] hands it to the
//! solver on impact and recovers without a visible snap.
//!
//! Frame order (do not call bare `sync_to_scene` during a handover):
//!
//! 1. sample animation → animated pose
//! 2. `blend.update` → pose to draw
//! 3. `blend.apply_to_scene`
//! 4. `world.sync_from_scene` (remaining kinematics)
//! 5. `world.step`
//! 6. `world.sync_to_scene_where(|id, _| !blend.owns_scene_write() || id != body)`
//!
//! ```text
//! cargo run -p threers-animation --example anim_physics_blend --features physics
//! ```

use threers::core::{Object3D, ObjectArena};
use threers::math::Vector3;
use threers_animation::prelude::*;
use threers_physics::prelude::*;

fn main() {
    let mut arena = ObjectArena::new();
    let root = arena.insert(Object3D::group());
    let character = arena.insert(Object3D::group());
    arena.add_child(root, character);
    arena.get_mut(character).unwrap().position = Vector3::new(0.0, 1.0, 0.0);

    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));
    let body = world.add_body(
        RigidBody::kinematic()
            .shape(Shape::capsule(0.5, 0.3))
            .translation(Vector3::new(0.0, 1.0, 0.0))
            .scene_object(character),
    );

    let mut blend = PhysicsBlend::new(body, 0.3);
    let dt = 1.0 / 60.0;

    println!("-- walk under animation --");
    for frame in 0..90 {
        // Scripted walk (stand-in for AnimationMixer writing the node).
        let t = frame as f32 * dt;
        let x = t * 2.0;
        arena.get_mut(character).unwrap().position = Vector3::new(x, 1.0, 0.0);

        let animated = Isometry::from_translation(arena.get(character).unwrap().position);
        let drawn = blend.update(dt, animated, &mut world);
        blend.apply_to_scene(drawn, &world, &mut arena);
        world.sync_from_scene(&arena);
        world.step(dt);
        world.sync_to_scene_where(&mut arena, |id, _| !(id == body && blend.owns_scene_write()));

        if frame % 30 == 29 {
            let p = arena.get(character).unwrap().position;
            println!(
                "  frame {frame}: mesh ({:.2}, {:.2})  state {:?}",
                p.x,
                p.y,
                blend.state()
            );
        }
    }

    println!("-- go limp mid-stride --");
    blend.go_limp(&mut world);
    assert!(world.body(body).unwrap().is_dynamic());
    let v = world.body(body).unwrap().linear_velocity;
    println!("  inherited velocity ≈ ({:.2}, {:.2}, {:.2})", v.x, v.y, v.z);

    for frame in 0..120 {
        // Animation would keep walking; blend freezes the limp pose.
        let animated = Isometry::from_translation(Vector3::new(50.0, 1.0, 0.0));
        world.step(dt);
        let drawn = blend.update(dt, animated, &mut world);
        blend.apply_to_scene(drawn, &world, &mut arena);
        world.sync_to_scene_where(&mut arena, |id, _| !(id == body && blend.owns_scene_write()));

        if frame % 40 == 39 {
            let p = arena.get(character).unwrap().position;
            println!(
                "  frame {frame}: mesh ({:.2}, {:.2})  body y {:.2}  state {:?}",
                p.x,
                p.y,
                world.body(body).unwrap().translation().y,
                blend.state()
            );
        }
    }
    assert_eq!(blend.state(), BlendState::Simulated);

    println!("-- recover to animation --");
    blend.recover(&mut world);
    let stand = Isometry::from_translation(Vector3::new(2.0, 1.0, 0.0));
    for frame in 0..60 {
        world.step(dt);
        let drawn = blend.update(dt, stand, &mut world);
        blend.apply_to_scene(drawn, &world, &mut arena);
        world.sync_to_scene_where(&mut arena, |id, _| !(id == body && blend.owns_scene_write()));
        if frame % 20 == 19 {
            let p = arena.get(character).unwrap().position;
            println!(
                "  frame {frame}: mesh ({:.2}, {:.2})  state {:?}",
                p.x,
                p.y,
                blend.state()
            );
        }
    }
    assert_eq!(blend.state(), BlendState::Animated);
    assert!(world.body(body).unwrap().is_kinematic());
    println!("done — no snap on limp or recover.");
}
