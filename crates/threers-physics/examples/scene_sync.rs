//! Driving a `threers` scene graph from the physics world.
//!
//! Shows the two directions of the binding:
//!
//! - **physics → scene** for dynamic bodies, so meshes follow the simulation
//! - **scene → physics** for kinematic bodies, so animation pushes things around
//!
//! ```text
//! cargo run -p threers-physics --example scene_sync
//! ```

use threers::core::{Object3D, ObjectArena};
use threers::geometries::{BoxGeometry, SphereGeometry};
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Matrix4};
use threers::core::Mesh;
use threers_physics::prelude::*;

fn main() {
    let mut arena = ObjectArena::new();
    let scene_root = arena.insert(Object3D::group());
    let mut world = World::new();

    // --- static level geometry, straight from the drawn mesh ---
    // A triangle-mesh collider is exact and free: it reuses the same
    // BufferGeometry the renderer already has.
    let floor_geometry = BoxGeometry::new(20.0, 0.5, 20.0);
    let floor_shape = Shape::trimesh_from_geometry(&floor_geometry)
        .expect("box geometry always makes a valid mesh");
    let floor_node = arena.insert(Object3D::mesh(Mesh::new(
        floor_geometry,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x404040))),
    )));
    arena.add_child(scene_root, floor_node);
    world.add_body(
        RigidBody::fixed()
            .shape(floor_shape)
            .translation(Vector3::new(0.0, -0.25, 0.0))
            .scene_object(floor_node),
    );

    // --- dynamic props: the body drives the mesh ---
    let ball_geometry = SphereGeometry::new(0.4, 16, 12);
    let mut balls = Vec::new();
    for i in 0..5 {
        let node = arena.insert(Object3D::mesh(Mesh::new(
            ball_geometry.clone(),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xe05050))),
        )));
        arena.add_child(scene_root, node);

        // `convex_hull_from_geometry` fits the collider to whatever you drew.
        // For a sphere the analytic `Shape::ball` is cheaper and exact, so use
        // that here — the hull path is for irregular meshes.
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(i as f32 * 0.9 - 1.8, 3.0 + i as f32 * 0.9, 0.0))
                .restitution(0.4)
                .scene_object(node),
        );
        balls.push(body);
    }

    // --- a kinematic paddle: the scene drives the body ---
    let paddle_geometry = BoxGeometry::new(4.0, 1.0, 4.0);
    let paddle_node = arena.insert(Object3D::mesh(Mesh::new(
        paddle_geometry,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x3080f0))),
    )));
    arena.add_child(scene_root, paddle_node);
    world.add_body(
        RigidBody::kinematic()
            .shape(Shape::cuboid(2.0, 0.5, 2.0))
            .translation(Vector3::new(-6.0, 1.0, 0.0))
            .scene_object(paddle_node),
    );

    println!("scene: {} nodes, world: {} bodies\n", 8, world.body_count());

    // --- the frame loop ---
    let dt = 1.0 / 60.0;
    for frame in 0..300 {
        // 1. Animate whatever you animate. Here, sweep the paddle across.
        let t = frame as f32 * dt;
        arena.get_mut(paddle_node).unwrap().position =
            Vector3::new(-6.0 + t * 2.0, 1.0, (t * 2.0).sin() * 2.0);

        // 2. Push those animated transforms into the kinematic bodies. This
        //    derives a velocity, so the paddle *sweeps* rather than teleporting
        //    and actually shoves the balls out of the way.
        world.sync_from_scene(&arena);

        // 3. Step the simulation.
        world.step(dt);

        // 4. Pull the results back onto the meshes. The interpolated variant
        //    blends between physics steps, which matters as soon as the render
        //    rate stops matching `world.timestep`.
        world.sync_to_scene_interpolated(&mut arena);

        if frame % 100 == 99 {
            arena.update_world_matrices(scene_root, Matrix4::identity());
            println!("frame {}:", frame + 1);
            for (i, id) in balls.iter().enumerate() {
                let node = world.body(*id).unwrap().scene_object.unwrap();
                let object = arena.get(node).unwrap();
                println!(
                    "  ball {i}: mesh at ({:.2}, {:.2}, {:.2})  world y {:.2}",
                    object.position.x,
                    object.position.y,
                    object.position.z,
                    object.world_position().y
                );
            }
            println!("  {} of {} bodies asleep", world.sleeping_count(), world.body_count());
        }
    }

    // Every mesh transform now matches its body exactly.
    println!("\nfinal check — mesh transforms track their bodies:");
    for id in &balls {
        let body = world.body(*id).unwrap();
        let node = arena.get(body.scene_object.unwrap()).unwrap();
        let drift = (node.position - body.translation()).length();
        println!("  drift {drift:.6} (interpolation makes this non-zero mid-step)");
    }
}
