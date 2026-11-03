//! Zero-gravity simulation: orbits, tumbling debris and n-body attraction.
//!
//! ```text
//! cargo run -p threers-physics --example zero_gravity_orbits
//! ```

use threers_physics::prelude::*;

fn main() {
    drifting_debris();
    circular_orbits();
    n_body();
    mass_and_density();
}

/// With no gravity and no damping, motion simply continues.
fn drifting_debris() {
    println!("-- drifting debris --");
    // `zero_gravity()` is `World::new().with_gravity(Vector3::ZERO)`.
    let mut world = World::zero_gravity();

    let probe = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.3, 0.2, 0.5))
            .linear_velocity(Vector3::new(2.0, 0.0, 0.0))
            .angular_velocity(Vector3::new(0.3, 1.2, 0.0))
            // A vacuum damps nothing. The defaults include a little angular
            // damping, which is right on a planet and wrong in space.
            .linear_damping(0.0)
            .angular_damping(0.0)
            .can_sleep(false),
    );

    for _ in 0..600 {
        world.step(1.0 / 60.0);
    }
    let body = world.body(probe).unwrap();
    println!(
        "  after 10 s: at x = {:.2} (10 s at 2 m/s), still tumbling at {:.2} rad/s",
        body.translation().x,
        body.angular_velocity.length()
    );
}

/// A planet at the origin, and satellites that fall around it forever.
fn circular_orbits() {
    println!("\n-- circular orbits --");
    // `mu` is G*M. The circular orbital speed at radius r is sqrt(mu / r).
    let mu = 400.0;
    let mut world = World::zero_gravity().with_gravity_model(GravityModel::Point {
        centre: Vector3::ZERO,
        mu,
        min_distance: 1.0,
    });

    let mut satellites = Vec::new();
    for (radius, mass) in [(6.0, 1.0), (10.0, 0.05), (10.0, 5000.0), (16.0, 1.0)] {
        let speed = (mu / radius).sqrt();
        satellites.push((
            radius,
            mass,
            world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.2))
                    .mass(mass)
                    .translation(Vector3::new(radius, 0.0, 0.0))
                    .linear_velocity(Vector3::new(0.0, 0.0, speed))
                    .linear_damping(0.0)
                    .can_sleep(false),
            ),
        ));
    }

    // The gravity model is evaluated inside the step, once per substep, so
    // there is nothing to call each frame.
    let mut extremes = vec![(f32::MAX, 0.0f32); satellites.len()];
    for _ in 0..3600 {
        world.step(1.0 / 60.0);
        for (i, (_, _, id)) in satellites.iter().enumerate() {
            let r = world.body(*id).unwrap().translation().length();
            extremes[i].0 = extremes[i].0.min(r);
            extremes[i].1 = extremes[i].1.max(r);
        }
    }

    println!("  60 s of orbiting:");
    for (i, (radius, mass, _)) in satellites.iter().enumerate() {
        let (min, max) = extremes[i];
        println!(
            "    r = {radius:>4.1}, mass {mass:>7.2}  ->  radius stayed between {min:.2} and {max:.2}",
        );
    }
    println!("  note the two satellites at r = 10: a 100000x mass difference,");
    println!("  identical orbits. Gravity is an acceleration, not a force.");
}

/// No central body — everything pulls on everything.
fn n_body() {
    println!("\n-- n-body cluster --");
    let mut world = World::zero_gravity().with_gravity_model(GravityModel::Mutual {
        g: 2.0,
        min_distance: 0.5,
    });

    let mut bodies = Vec::new();
    for i in 0..8 {
        let angle = i as f32 / 8.0 * std::f32::consts::TAU;
        bodies.push(world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(angle.cos() * 8.0, 0.0, angle.sin() * 8.0))
                // A little tangential motion, not enough to escape.
                .linear_velocity(Vector3::new(-angle.sin() * 1.5, 0.0, angle.cos() * 1.5))
                .mass(10.0)
                .linear_damping(0.0)
                .restitution(0.4)
                .can_sleep(false),
        ));
    }

    let spread = |w: &World| -> f32 {
        let centre = bodies
            .iter()
            .filter_map(|id| w.body(*id))
            .fold(Vector3::ZERO, |acc, b| acc + b.translation())
            * (1.0 / bodies.len() as f32);
        bodies
            .iter()
            .filter_map(|id| w.body(*id))
            .map(|b| (b.translation() - centre).length())
            .fold(0.0f32, f32::max)
    };

    println!("  starting spread: {:.2}", spread(&world));
    for _ in 0..900 {
        world.step(1.0 / 60.0);
    }
    println!("  after 15 s:      {:.2}  (they draw together and collide)", spread(&world));

    // Momentum is conserved: nothing pushed the cluster anywhere.
    let momentum = bodies
        .iter()
        .filter_map(|id| world.body(*id))
        .fold(Vector3::ZERO, |acc, b| acc + b.linear_velocity * b.mass());
    println!("  net momentum:    {:.4}  (equal and opposite pulls)", momentum.length());
}

/// Mass, density and where a body balances.
fn mass_and_density() {
    println!("\n-- configuring mass --");

    // Density scales with volume; the same material makes a bigger thing heavier.
    for radius in [0.5f32, 1.0, 2.0] {
        let b = RigidBody::dynamic().shape(Shape::ball(radius)).density(2.5).build();
        println!(
            "  ball r={radius:.1} at density 2.5  ->  mass {:.3}, back-computed density {:.2}",
            b.mass(),
            b.effective_density()
        );
    }

    // Or state the mass outright and let the inertia scale with it.
    let exact = RigidBody::dynamic().shape(Shape::ball(1.0)).mass(12.0).build();
    println!("  explicit mass 12.0                ->  mass {:.3}", exact.mass());

    // Move the balance point: a weeble rights itself because its mass sits low.
    let weeble = RigidBody::dynamic()
        .shape(Shape::ball(0.5))
        .center_of_mass(Vector3::new(0.0, -0.35, 0.0))
        .build();
    println!("  centre of mass override           ->  {:?}", weeble.local_center_of_mass());

    // Or fake a mass distribution the collider does not have.
    let flywheel = RigidBody::dynamic()
        .shape(Shape::cylinder(0.1, 1.0))
        .principal_inertia(Vector3::new(1.0, 50.0, 1.0))
        .build();
    println!(
        "  inertia override                  ->  {:?}  (hard to spin about y)",
        flywheel.principal_inertia()
    );
}
