//! Scene queries: rays, shape casts, overlap tests and point lookups.
//!
//! ```text
//! cargo run -p threers-physics --example raycast_picking
//! ```

use threers_physics::prelude::*;

// Collision groups, exactly like three.js `Layers`.
const SCENERY: u32 = 1 << 0;
const PROPS: u32 = 1 << 1;

fn main() {
    let mut world = World::new();

    world.add_body(
        RigidBody::fixed()
            .collider(Collider::new(Shape::ground()).groups(InteractionGroups::new(SCENERY, !0))),
    );

    // Three props in a row, tagged via `user_data` so query hits identify
    // themselves without a side table.
    let names = ["crate", "barrel", "statue"];
    for (i, name) in names.iter().enumerate() {
        world.add_body(
            RigidBody::dynamic()
                .collider(
                    Collider::new(Shape::cuboid(0.5, 0.5, 0.5))
                        .groups(InteractionGroups::new(PROPS, !0))
                        .user_data(i as u64),
                )
                .translation(Vector3::new(i as f32 * 3.0, 0.5, 0.0)),
        );
        println!("prop {i} = {name}");
    }

    // A trigger volume. Sensors are invisible to queries unless asked for.
    world.add_body(
        RigidBody::fixed()
            .collider(Collider::new(Shape::cuboid(1.0, 1.0, 1.0)).sensor(true))
            .translation(Vector3::new(3.0, 3.0, 0.0)),
    );

    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }

    // --- picking: what is under this point? ---
    println!("\n-- raycast --");
    let ray = Ray::new(Vector3::new(3.0, 8.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
    match world.raycast(&ray, 100.0, QueryFilter::default()) {
        Some(hit) => println!(
            "  straight down at x=3: hit {} at {:.2} m, normal {:?}",
            names[hit.user_data as usize], hit.toi, hit.normal
        ),
        None => println!("  nothing there"),
    }

    // The sensor sits above the barrel and is skipped by default.
    let with_sensors = world
        .raycast(&ray, 100.0, QueryFilter::default().include_sensors())
        .unwrap();
    println!(
        "  including sensors: first hit is now {:.2} m away (the trigger volume)",
        with_sensors.toi
    );

    // --- every hit along the ray ---
    let mut all = world.raycast_all(&ray, 100.0, QueryFilter::default());
    all.sort_by(|a, b| a.toi.partial_cmp(&b.toi).unwrap());
    println!("  all hits: {:?}", all.iter().map(|h| h.toi).collect::<Vec<_>>());

    // --- line of sight, ignoring the body doing the looking ---
    println!("\n-- line of sight --");
    let eye = Vector3::new(-4.0, 0.5, 0.0);
    let target = Vector3::new(6.0, 0.5, 0.0);
    let sight = Ray::new(eye, target - eye);
    let blocked = world.ray_is_blocked(&sight, (target - eye).length(), QueryFilter::default());
    println!("  can the statue be seen from x=-4? {}", if blocked { "no" } else { "yes" });

    // Only look at scenery — the props no longer block.
    let scenery_only = QueryFilter::default().groups(InteractionGroups::new(!0, SCENERY));
    let blocked = world.ray_is_blocked(&sight, (target - eye).length(), scenery_only);
    println!("  ignoring props (group filter)?     {}", if blocked { "no" } else { "yes" });

    // --- shape cast: will a 1 m sphere fit through here? ---
    println!("\n-- shape cast --");
    let hit = world.cast_shape(
        &Shape::ball(0.5),
        &Isometry::from_translation(Vector3::new(-4.0, 0.5, 0.0)),
        Vector3::new(1.0, 0.0, 0.0),
        20.0,
        QueryFilter::default(),
    );
    match hit {
        Some(h) => println!(
            "  a rolling sphere travels {:.2} m before hitting {}",
            h.toi, names[h.user_data as usize]
        ),
        None => println!("  clear all the way"),
    }

    // --- overlap and point queries ---
    println!("\n-- overlap and point queries --");
    let overlapping = world.intersections_with_shape(
        &Shape::ball(2.0),
        &Isometry::from_translation(Vector3::new(3.0, 0.5, 0.0)),
        QueryFilter::default(),
    );
    println!("  {} colliders within 2 m of the barrel", overlapping.len());

    let inside = world.bodies_at_point(Vector3::new(3.0, 0.5, 0.0), QueryFilter::default());
    println!("  bodies containing the barrel's centre: {}", inside.len());

    let projection = world
        .project_point(Vector3::new(3.0, 4.0, 0.0), QueryFilter::default())
        .unwrap();
    println!(
        "  nearest surface to (3, 4, 0) is {:.2} m away at {:?}",
        projection.distance, projection.point
    );
}
