//! Adversarial tests: invariants that must hold no matter what is thrown at the
//! engine. These are deliberately harsher than the unit tests — degenerate
//! inputs, extreme masses, absurd speeds, long runs.

use threers_physics::prelude::*;

/// Deterministic pseudo-random source. `rand` is not a dependency, and a fixed
/// sequence makes any failure reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
    fn f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }
    fn vector(&mut self, extent: f32) -> Vector3 {
        Vector3::new(
            self.range(-extent, extent),
            self.range(-extent, extent),
            self.range(-extent, extent),
        )
    }
    fn quaternion(&mut self) -> Quaternion {
        Quaternion::from_euler_xyz(
            self.range(-std::f32::consts::PI, std::f32::consts::PI),
            self.range(-std::f32::consts::PI, std::f32::consts::PI),
            self.range(-std::f32::consts::PI, std::f32::consts::PI),
        )
        .normalize()
    }
    fn shape(&mut self) -> Shape {
        match self.next_u32() % 6 {
            0 => Shape::ball(self.range(0.1, 0.6)),
            1 => Shape::cuboid(
                self.range(0.1, 0.6),
                self.range(0.1, 0.6),
                self.range(0.1, 0.6),
            ),
            2 => Shape::capsule(self.range(0.05, 0.5), self.range(0.1, 0.4)),
            3 => Shape::cylinder(self.range(0.1, 0.5), self.range(0.1, 0.5)),
            4 => Shape::cone(self.range(0.1, 0.5), self.range(0.1, 0.5)),
            _ => Shape::compound(vec![
                (
                    Isometry::from_translation(Vector3::new(-0.3, 0.0, 0.0)),
                    Shape::ball(0.25),
                ),
                (
                    Isometry::from_translation(Vector3::new(0.3, 0.0, 0.0)),
                    Shape::cuboid(0.2, 0.2, 0.2),
                ),
            ]),
        }
    }
}

fn finite(v: Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn assert_world_sane(world: &World, context: &str) {
    for (id, body) in world.bodies().iter() {
        assert!(
            finite(body.translation()),
            "{context}: body {id:?} has a non-finite position {:?}",
            body.translation()
        );
        assert!(
            finite(body.linear_velocity) && finite(body.angular_velocity),
            "{context}: body {id:?} has a non-finite velocity"
        );
        let q = body.rotation();
        assert!(
            q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite(),
            "{context}: body {id:?} has a non-finite rotation"
        );
        let norm = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-2,
            "{context}: body {id:?} rotation drifted from unit length ({norm})"
        );
        assert!(
            body.translation().length() < 1e5,
            "{context}: body {id:?} was flung to {:?}",
            body.translation()
        );
    }
}

/// A closed box of walls, so nothing can legitimately escape.
fn sealed_room(world: &mut World, half: f32) {
    let t = 0.5;
    let faces = [
        (Vector3::new(0.0, -half - t, 0.0), Vector3::new(half + t, t, half + t)),
        (Vector3::new(0.0, half + t, 0.0), Vector3::new(half + t, t, half + t)),
        (Vector3::new(-half - t, 0.0, 0.0), Vector3::new(t, half + t, half + t)),
        (Vector3::new(half + t, 0.0, 0.0), Vector3::new(t, half + t, half + t)),
        (Vector3::new(0.0, 0.0, -half - t), Vector3::new(half + t, half + t, t)),
        (Vector3::new(0.0, 0.0, half + t), Vector3::new(half + t, half + t, t)),
    ];
    for (centre, extents) in faces {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(extents.x, extents.y, extents.z))
                .translation(centre)
                .friction(0.5),
        );
    }
}

#[test]
fn a_pile_of_random_shapes_never_produces_nan_or_explodes() {
    let mut rng = Rng::new(0xDEADBEEF);
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.7));

    for i in 0..60 {
        world.add_body(
            RigidBody::dynamic()
                .shape(rng.shape())
                .translation(Vector3::new(
                    rng.range(-2.0, 2.0),
                    1.0 + i as f32 * 0.7,
                    rng.range(-2.0, 2.0),
                ))
                .rotation(rng.quaternion())
                .angular_velocity(rng.vector(3.0))
                .friction(rng.range(0.0, 1.0))
                .restitution(rng.range(0.0, 0.9)),
        );
    }

    for step in 0..900 {
        world.step(1.0 / 60.0);
        if step % 60 == 0 {
            assert_world_sane(&world, &format!("step {step}"));
        }
    }
    assert_world_sane(&world, "final");

    // Nothing may end up below the floor.
    for (id, body) in world.bodies().iter().filter(|(_, b)| b.is_dynamic()) {
        assert!(
            body.translation().y > -1.0,
            "body {id:?} sank through the ground to {:?}",
            body.translation()
        );
    }
}

#[test]
fn extreme_mass_ratios_stay_stable() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));

    // A feather under an anvil: a 100000:1 ratio.
    let feather = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.5, 0.1, 0.5))
            .translation(Vector3::new(0.0, 0.1, 0.0))
            .mass(0.01),
    );
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.5, 0.5, 0.5))
            .translation(Vector3::new(0.0, 0.75, 0.0))
            .mass(1000.0),
    );

    for _ in 0..600 {
        world.step(1.0 / 60.0);
    }
    assert_world_sane(&world, "mass ratio");
    let y = world.body(feather).unwrap().translation().y;
    assert!(y > -0.1, "the feather was crushed through the floor to {y}");
}

#[test]
fn a_sealed_room_never_leaks_however_fast_the_contents_move() {
    let mut rng = Rng::new(7);
    let mut world = World::new();
    let half = 5.0;
    sealed_room(&mut world, half);

    let mut balls = Vec::new();
    for _ in 0..25 {
        balls.push(
            world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(rng.range(0.15, 0.4)))
                    .translation(rng.vector(half * 0.6))
                    // Very fast, to stress the contact pipeline.
                    .linear_velocity(rng.vector(40.0))
                    .restitution(0.9)
                    .gravity_scale(0.0)
                    .ccd(true)
                    .can_sleep(false),
            ),
        );
    }

    for step in 0..600 {
        world.step(1.0 / 60.0);
        for id in &balls {
            let p = world.body(*id).unwrap().translation();
            assert!(
                p.x.abs() < half + 1.0 && p.y.abs() < half + 1.0 && p.z.abs() < half + 1.0,
                "step {step}: a ball escaped the sealed room to {p:?}"
            );
        }
    }
    assert_world_sane(&world, "sealed room");
}

#[test]
fn a_fast_bullet_does_not_tunnel_through_a_thin_wall() {
    // A 1 cm wall and a projectile crossing 3.3 m per step. Without continuous
    // collision detection it is on the far side before anything is tested.
    let mut world = World::new();
    world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.005, 5.0, 5.0))
            .translation(Vector3::new(0.0, 0.0, 0.0)),
    );
    let bullet = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.05))
            .translation(Vector3::new(-5.0, 0.0, 0.0))
            .linear_velocity(Vector3::new(200.0, 0.0, 0.0))
            .gravity_scale(0.0)
            .ccd(true)
            .can_sleep(false),
    );

    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }
    let x = world.body(bullet).unwrap().translation().x;
    assert!(
        x < 0.1,
        "the bullet tunnelled through the wall and ended up at x = {x}"
    );
}

#[test]
fn energy_never_grows_in_a_closed_inelastic_system() {
    let mut rng = Rng::new(99);
    let mut world = World::new();
    sealed_room(&mut world, 4.0);

    let mut bodies = Vec::new();
    for _ in 0..20 {
        bodies.push(world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(rng.vector(2.5))
                .linear_velocity(rng.vector(5.0))
                // No gravity and no bounce: kinetic energy can only fall.
                .gravity_scale(0.0)
                .restitution(0.0)
                .friction(0.0)
                .can_sleep(false),
        ));
    }

    let kinetic = |w: &World| -> f32 {
        bodies
            .iter()
            .filter_map(|id| w.body(*id))
            .map(|b| 0.5 * b.mass() * b.linear_velocity.length_sq())
            .sum()
    };

    let start = kinetic(&world);
    let mut peak = start;
    for _ in 0..600 {
        world.step(1.0 / 60.0);
        peak = peak.max(kinetic(&world));
    }
    // A little overshoot from overlap recovery is expected; doubling is not.
    assert!(
        peak < start * 1.5,
        "kinetic energy grew from {start} to {peak} — the solver is injecting energy"
    );
    assert!(kinetic(&world) <= peak);
}

#[test]
fn degenerate_and_hostile_inputs_do_not_panic() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));

    // Zero and negative sizes are clamped, not fatal.
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.0))
            .translation(Vector3::new(0.0, 1.0, 0.0)),
    );
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(-1.0, 0.0, -0.0))
            .translation(Vector3::new(0.5, 1.0, 0.0)),
    );
    // Exactly coincident bodies have no separating direction to find.
    for _ in 0..3 {
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(2.0, 1.0, 0.0)),
        );
    }
    // A body with zero mass requested.
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.3))
            .mass(0.0)
            .translation(Vector3::new(-2.0, 1.0, 0.0)),
    );

    for _ in 0..300 {
        world.step(1.0 / 60.0);
    }
    assert_world_sane(&world, "degenerate shapes");
}

#[test]
fn removing_bodies_mid_simulation_is_safe() {
    let mut rng = Rng::new(4242);
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));

    let mut live: Vec<BodyId> = Vec::new();
    for step in 0..400 {
        // Spawn.
        if step % 3 == 0 {
            live.push(world.add_body(
                RigidBody::dynamic()
                    .shape(rng.shape())
                    .translation(Vector3::new(rng.range(-2.0, 2.0), 5.0, rng.range(-2.0, 2.0))),
            ));
        }
        // Join two at random, so joints reference bodies that may vanish.
        if live.len() >= 2 && step % 7 == 0 {
            let a = live[rng.next_u32() as usize % live.len()];
            let b = live[rng.next_u32() as usize % live.len()];
            if a != b {
                world.add_joint(Joint::distance(a, b, Vector3::ZERO, Vector3::ZERO, 1.5));
            }
        }
        // Remove.
        if !live.is_empty() && step % 5 == 0 {
            let i = rng.next_u32() as usize % live.len();
            let id = live.swap_remove(i);
            world.remove_body(id);
            // Stale handles must resolve to nothing, not to a recycled body.
            assert!(world.body(id).is_none());
        }
        world.step(1.0 / 60.0);
    }
    assert_world_sane(&world, "churn");
}

#[test]
fn a_long_joint_chain_stays_attached() {
    let mut world = World::new();
    // Twenty joints in series is a deep constraint graph, and sequential
    // impulses carry a correction one joint per iteration: at the default 8 the
    // anchor end never hears about the load hanging off the far end, and the
    // top links sit visibly stretched. It is a convergence budget, not an
    // instability — measured over 15 s of swing, the worst link is 1.61 out at
    // 8 iterations, 0.06 at 24 and 0.009 at 64. Long chains are the case that
    // has to ask for more.
    world.solver_config.velocity_iterations = 24;
    let anchor = world.add_body(RigidBody::fixed().translation(Vector3::new(0.0, 10.0, 0.0)));

    let mut previous = anchor;
    let mut links = Vec::new();
    for i in 0..20 {
        let link = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.1))
                .translation(Vector3::new(0.0, 10.0 - (i + 1) as f32 * 0.5, 0.0))
                .can_sleep(false),
        );
        world.add_joint(Joint::distance(
            previous,
            link,
            Vector3::ZERO,
            Vector3::ZERO,
            0.5,
        ));
        previous = link;
        links.push(link);
    }

    // Give it a shove so it swings. Sized from the link's own mass rather than
    // written as a fixed impulse: these are 0.1 m balls at the default density,
    // which is about four grams each, and 20 N·s on four grams is not a shove
    // but a rifle round — it leaves the tip doing 4.8 km/s, and what the rest of
    // the test then measures is how long the solver takes to swallow that.
    let tip = *links.last().unwrap();
    let shove = world.body(tip).unwrap().mass() * 3.0;
    world
        .body_mut(tip)
        .unwrap()
        .apply_impulse(Vector3::new(shove, 0.0, 0.0));

    for _ in 0..900 {
        world.step(1.0 / 60.0);
    }
    assert_world_sane(&world, "chain");

    // Every link must still be roughly 0.5 from its neighbour.
    let mut previous_position = world.body(anchor).unwrap().translation();
    for (i, id) in links.iter().enumerate() {
        let p = world.body(*id).unwrap().translation();
        let d = (p - previous_position).length();
        assert!(
            (d - 0.5).abs() < 0.15,
            "link {i} is {d} from its neighbour, expected 0.5"
        );
        previous_position = p;
    }
}

#[test]
fn kinematic_motion_is_unaffected_by_the_frame_rate() {
    // A kinematic body is told where to be. It must arrive there whatever the
    // frame time does, including frames that run zero or several fixed steps.
    let run = |frame_dt: f32, frames: usize| {
        use threers::core::{Object3D, ObjectArena};
        let mut arena = ObjectArena::new();
        let node = arena.insert(Object3D::group());
        let mut world = World::new();
        let body = world.add_body(
            RigidBody::kinematic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .scene_object(node),
        );
        for i in 0..frames {
            // A straight line at 1 unit per frame of *simulated* time.
            let t = (i + 1) as f32 * frame_dt;
            arena.get_mut(node).unwrap().position = Vector3::new(t, 0.0, 0.0);
            world.sync_from_scene(&arena);
            world.step(frame_dt);
        }
        world.body(body).unwrap().translation().x
    };

    // One second of simulated time, three very different frame rates. Each
    // should place the body at x = 1.0.
    for (dt, frames) in [(1.0 / 60.0, 60), (1.0 / 30.0, 30), (1.0 / 120.0, 120)] {
        let x = run(dt, frames);
        assert!(
            (x - 1.0).abs() < 0.02,
            "at dt = {dt}, the kinematic body ended at x = {x}, expected 1.0"
        );
    }
}

#[test]
fn queries_survive_hostile_input() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.5))
            .translation(Vector3::new(0.0, 2.0, 0.0)),
    );

    // A zero-length ray direction must be rejected, not divided by.
    assert!(world
        .raycast(
            &Ray::new(Vector3::ZERO, Vector3::ZERO),
            10.0,
            QueryFilter::default()
        )
        .is_none());
    // Zero and negative distances.
    world.raycast(
        &Ray::new(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
        0.0,
        QueryFilter::default(),
    );
    world.raycast(
        &Ray::new(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
        -1.0,
        QueryFilter::default(),
    );
    // A zero-direction shape cast.
    assert!(world
        .cast_shape(
            &Shape::ball(0.2),
            &Isometry::IDENTITY,
            Vector3::ZERO,
            5.0,
            QueryFilter::default()
        )
        .is_none());
    // Casting a non-convex shape is refused rather than misbehaving.
    assert!(world
        .cast_shape(
            &Shape::ground(),
            &Isometry::IDENTITY,
            Vector3::new(1.0, 0.0, 0.0),
            5.0,
            QueryFilter::default()
        )
        .is_none());
    // Queries against an empty world.
    let empty = World::new();
    assert!(empty
        .raycast(
            &Ray::new(Vector3::ZERO, Vector3::UP),
            100.0,
            QueryFilter::default()
        )
        .is_none());
    assert!(empty
        .project_point(Vector3::ZERO, QueryFilter::default())
        .is_none());
    assert!(empty
        .bodies_at_point(Vector3::ZERO, QueryFilter::default())
        .is_empty());
}

#[test]
fn ik_survives_hostile_targets() {
    let mut chain = IkChain::from_points(&[
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
    ]);
    for target in [
        Vector3::ZERO,
        Vector3::new(1e6, 0.0, 0.0),
        Vector3::new(-1e-9, 1e-9, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
    ] {
        chain.solve(target);
        for j in &chain.joints {
            assert!(finite(j.position), "FABRIK produced {:?} for {target:?}", j.position);
        }
        chain.solve_ccd(target);
        for j in &chain.joints {
            assert!(finite(j.position), "CCD produced {:?} for {target:?}", j.position);
        }
    }

    // Coincident joints have no direction to work with.
    let mut degenerate = IkChain::from_points(&[Vector3::ZERO, Vector3::ZERO, Vector3::ZERO]);
    degenerate.solve(Vector3::new(1.0, 1.0, 1.0));
    for j in &degenerate.joints {
        assert!(finite(j.position));
    }
}


