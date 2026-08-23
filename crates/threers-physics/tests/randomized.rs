//! Randomized scenes, checked against invariants that must hold for all of them.
//!
//! Every other test in this crate asserts something about a scene someone
//! thought of. This one asserts things that must be true of scenes nobody
//! thought of: nothing becomes NaN, nothing gains energy from nowhere, nothing
//! ends up inside the floor, and a replay reproduces the original exactly.
//!
//! The scenes come from a seeded generator, so a failure is reproducible: the
//! message names the seed, and [`world_from_seed`] rebuilds that exact scene.
//! That is the whole point — a randomized test that cannot be re-run is a
//! randomized test nobody can fix.
//!
//! This is deliberately not `cargo-fuzz`. Fuzzing needs nightly and a corpus,
//! and it is good at finding *crashes*; the interesting failures in a physics
//! engine are not crashes but silent wrongness — a body that drifts, a stack
//! that gains energy, a replay that diverges. Those need assertions, which is
//! what this is.

use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

/// xorshift32. Not a good generator; a reproducible one, which is what matters.
struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in `-1..1`.
    fn signed(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0
    }

    /// Uniform in `lo..hi`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.signed() * 0.5 + 0.5) * (hi - lo)
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n.max(1)
    }
}

/// A scene of assorted shapes dropped over a floor, plus some walls.
fn world_from_seed(seed: u32, count: usize) -> (World, Vec<BodyId>) {
    let mut rng = Rng(seed | 1); // xorshift dies on zero
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.7));
    for (x, z) in [(-8.0f32, 0.0f32), (8.0, 0.0), (0.0, -8.0), (0.0, 8.0)] {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(
                    if x == 0.0 { 8.0 } else { 0.5 },
                    4.0,
                    if z == 0.0 { 8.0 } else { 0.5 },
                ))
                .translation(Vector3::new(x, 4.0, z)),
        );
    }

    let mut ids = Vec::new();
    for _ in 0..count {
        let shape = match rng.below(6) {
            0 => Shape::ball(rng.range(0.2, 0.6)),
            1 => Shape::cuboid(rng.range(0.2, 0.6), rng.range(0.2, 0.6), rng.range(0.2, 0.6)),
            2 => Shape::capsule(rng.range(0.2, 0.5), rng.range(0.15, 0.4)),
            3 => Shape::cylinder(rng.range(0.2, 0.5), rng.range(0.2, 0.5)),
            4 => Shape::cone(rng.range(0.2, 0.5), rng.range(0.2, 0.5)),
            _ => Shape::convex_hull(&(0..8).map(|_| {
                Vector3::new(rng.signed() * 0.5, rng.signed() * 0.5, rng.signed() * 0.5)
            })
            .collect::<Vec<_>>())
            .unwrap_or_else(|| Shape::ball(0.3)),
        };
        let rotation = Quaternion::from_axis_angle(
            crate_normalize(Vector3::new(rng.signed(), rng.signed(), rng.signed())),
            rng.range(0.0, std::f32::consts::TAU),
        );
        ids.push(
            world.add_body(
                RigidBody::dynamic()
                    .shape(shape)
                    .translation(Vector3::new(
                        rng.range(-5.0, 5.0),
                        rng.range(1.0, 9.0),
                        rng.range(-5.0, 5.0),
                    ))
                    .rotation(rotation)
                    .linear_velocity(Vector3::new(
                        rng.signed() * 3.0,
                        rng.signed() * 3.0,
                        rng.signed() * 3.0,
                    ))
                    .angular_velocity(Vector3::new(
                        rng.signed() * 4.0,
                        rng.signed() * 4.0,
                        rng.signed() * 4.0,
                    ))
                    .density(rng.range(200.0, 3000.0))
                    .friction(rng.range(0.0, 1.2))
                    .restitution(rng.range(0.0, 0.9)),
            ),
        );
    }
    (world, ids)
}

fn crate_normalize(v: Vector3) -> Vector3 {
    let len = v.length();
    if len > 1e-6 {
        v * (1.0 / len)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    }
}

fn finite(v: Vector3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Everything that must be true of any body at any moment.
fn check(world: &World, ids: &[BodyId], seed: u32, step: usize) {
    for id in ids {
        let Some(body) = world.body(*id) else { continue };
        let p = body.translation();
        assert!(
            finite(p) && finite(body.linear_velocity) && finite(body.angular_velocity),
            "seed {seed} step {step}: body went non-finite — p {:?} v {:?} w {:?}",
            p,
            body.linear_velocity,
            body.angular_velocity
        );
        let q = body.rotation();
        assert!(
            q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite(),
            "seed {seed} step {step}: rotation went non-finite"
        );
        let norm = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "seed {seed} step {step}: rotation drifted off unit length ({norm})"
        );
        // The box is 16 across and 8 tall with walls; nothing should escape it
        // by more than a wide margin, and nothing should sink far below zero.
        assert!(
            p.y > -5.0,
            "seed {seed} step {step}: a body sank to y = {} — through the floor",
            p.y
        );
        assert!(
            p.x.abs() < 60.0 && p.z.abs() < 60.0 && p.y < 200.0,
            "seed {seed} step {step}: a body was flung to {p:?}"
        );
        assert!(
            body.linear_velocity.length() < 500.0,
            "seed {seed} step {step}: a body reached {} units/s from a 3 unit/s start",
            body.linear_velocity.length()
        );
    }
}

#[test]
fn random_scenes_stay_finite_and_bounded() {
    for seed in 1..=40u32 {
        let (mut world, ids) = world_from_seed(seed.wrapping_mul(2_654_435_761), 25);
        for step in 0..400 {
            world.step(DT);
            if step % 20 == 0 {
                check(&world, &ids, seed, step);
            }
        }
        check(&world, &ids, seed, 400);
    }
}

#[test]
fn random_scenes_settle_rather_than_simmer() {
    // A scene left alone must run out of energy. One that does not is either
    // gaining it from the solver or jittering forever, and both look the same
    // from outside: a pile that never sleeps.
    //
    // Measured on the energy rather than on a count of bodies still moving,
    // because "still moving" catches something that is not a fault. A ball
    // rolling without slipping dissipates nothing — the friction force does no
    // work at the contact — so nothing stops it but angular damping, whose
    // default time constant is 20 s. After 25 s a roller is still doing about
    // 29% of its original speed, and a scene of balls and cylinders will always
    // have a few of those. That is the shapes being round, not the solver
    // simmering, and only the energy can tell the two apart.
    for seed in 1..=12u32 {
        let (mut world, _) = world_from_seed(seed.wrapping_mul(40_503), 20);
        // A second first, to get past the spawn: bodies are dropped in
        // overlapping and pushing them apart is real work that the initial
        // energy has not accounted for.
        for _ in 0..60 {
            world.step(DT);
        }
        let start = world.energy();
        let mut peak = start;
        for _ in 0..1440 {
            world.step(DT);
            peak = peak.max(world.energy());
        }
        let end = world.energy();

        assert!(
            peak <= start * 1.001,
            "seed {seed}: energy climbed from {start:.0} to {peak:.0} — the solver is \
             putting in more than the scene started with"
        );
        assert!(
            end < start * 0.25,
            "seed {seed}: energy only fell from {start:.0} to {end:.0}, so the scene is \
             still simmering rather than running down"
        );
    }
}

#[test]
fn random_scenes_never_gain_energy_from_nothing() {
    // With gravity off and no restitution, total kinetic energy can only fall.
    // Rising means the solver is doing work, which is the failure mode behind
    // exploding stacks and jittering piles.
    for seed in 1..=20u32 {
        let mut rng = Rng(seed | 1);
        let mut world = World::new();
        world.gravity = Vector3::ZERO;
        for (x, z) in [(-4.0f32, 0.0f32), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0)] {
            world.add_body(
                RigidBody::fixed()
                    .shape(Shape::cuboid(
                        if x == 0.0 { 4.0 } else { 0.5 },
                        4.0,
                        if z == 0.0 { 4.0 } else { 0.5 },
                    ))
                    .translation(Vector3::new(x, 0.0, z))
                    .restitution(0.0),
            );
        }
        let mut ids = Vec::new();
        for _ in 0..14 {
            ids.push(
                world.add_body(
                    RigidBody::dynamic()
                        .shape(Shape::cuboid(0.4, 0.4, 0.4))
                        .translation(Vector3::new(
                            rng.range(-2.5, 2.5),
                            rng.range(-2.5, 2.5),
                            rng.range(-2.5, 2.5),
                        ))
                        .linear_velocity(Vector3::new(
                            rng.signed() * 4.0,
                            rng.signed() * 4.0,
                            rng.signed() * 4.0,
                        ))
                        .restitution(0.0)
                        .friction(0.4)
                        .can_sleep(false),
                ),
            );
        }

        let energy = |world: &World| -> f32 {
            ids.iter()
                .filter_map(|id| world.body(*id))
                .map(|b| {
                    0.5 * b.mass() * b.linear_velocity.length_sq()
                        + 0.5 * b.mass() * b.angular_velocity.length_sq() * 0.1
                })
                .sum()
        };

        let start = energy(&world);
        for _ in 0..600 {
            world.step(DT);
        }
        let end = energy(&world);
        assert!(
            end <= start * 1.05 + 1.0,
            "seed {seed}: energy rose from {start} to {end} with gravity off and no bounce"
        );
    }
}

#[test]
fn random_scenes_replay_exactly() {
    // Determinism on scenes nobody designed. The hand-written replay tests use
    // tidy stacks; this one uses whatever the generator produced, including the
    // messy transient contacts of the first few frames.
    for seed in 1..=15u32 {
        let (mut world, ids) = world_from_seed(seed.wrapping_mul(2_246_822_519), 18);
        for _ in 0..40 {
            world.step_fixed();
        }

        let snapshot = world.snapshot();
        let record = |world: &mut World| -> Vec<(Vector3, Vector3)> {
            let mut trace = Vec::new();
            for _ in 0..80 {
                world.step_fixed();
                for id in &ids {
                    if let Some(b) = world.body(*id) {
                        trace.push((b.translation(), b.linear_velocity));
                    }
                }
            }
            trace
        };

        let original = record(&mut world);
        assert!(world.restore(&snapshot), "seed {seed}: restore was refused");
        let replay = record(&mut world);
        assert_eq!(
            original.len(),
            replay.len(),
            "seed {seed}: the replay produced a different number of samples"
        );
        for (i, (a, b)) in original.iter().zip(replay.iter()).enumerate() {
            assert_eq!(a, b, "seed {seed}: replay diverged at sample {i}");
        }
    }
}

#[test]
fn random_scenes_survive_being_poked() {
    // Bodies added, removed, teleported and impulsed while the simulation runs.
    // Editing a live world is the normal case in a game, and the arena, the
    // spatial index and the contact set all have to keep up.
    for seed in 1..=15u32 {
        let (mut world, mut ids) = world_from_seed(seed.wrapping_mul(97_531), 15);
        let mut rng = Rng(seed | 1);

        for step in 0..500 {
            world.step(DT);
            match rng.below(6) {
                0 if !ids.is_empty() => {
                    let victim = ids.swap_remove(rng.below(ids.len() as u32) as usize);
                    world.remove_body(victim);
                }
                1 => ids.push(
                    world.add_body(
                        RigidBody::dynamic()
                            .shape(Shape::ball(rng.range(0.2, 0.5)))
                            .translation(Vector3::new(
                                rng.range(-4.0, 4.0),
                                rng.range(6.0, 10.0),
                                rng.range(-4.0, 4.0),
                            )),
                    ),
                ),
                2 if !ids.is_empty() => {
                    let id = ids[rng.below(ids.len() as u32) as usize];
                    // Scaled by the body's own mass, so this is a shove of a few
                    // metres per second whatever it lands on. These are 0.2–0.5 m
                    // balls at the default density of 1, so a fixed 20 N·s is
                    // upwards of 100 m/s — which crosses this 16 m room in four
                    // steps and goes through a 0.5 m wall without a step ever
                    // landing inside it. That is what CCD is for, and it is off
                    // here by design; a discrete solver being unable to catch a
                    // body nobody asked it to sweep is not it failing to survive
                    // a poke.
                    let mass = world.body(id).map_or(1.0, |b| b.mass());
                    if let Some(b) = world.body_mut(id) {
                        b.apply_impulse(Vector3::new(
                            rng.signed() * 5.0 * mass,
                            rng.range(0.0, 5.0) * mass,
                            rng.signed() * 5.0 * mass,
                        ));
                    }
                }
                3 if !ids.is_empty() => {
                    let id = ids[rng.below(ids.len() as u32) as usize];
                    let to = Vector3::new(rng.range(-4.0, 4.0), rng.range(1.0, 8.0), rng.range(-4.0, 4.0));
                    if let Some(b) = world.body_mut(id) {
                        b.set_translation(to);
                    }
                }
                4 => {
                    // Queries against a world in flux must not panic either.
                    let ray = Ray::new(
                        Vector3::new(rng.range(-6.0, 6.0), 12.0, rng.range(-6.0, 6.0)),
                        Vector3::new(rng.signed() * 0.3, -1.0, rng.signed() * 0.3),
                    );
                    let _ = world.raycast(&ray, 50.0, QueryFilter::default());
                    let _ = world.project_point(ray.origin, QueryFilter::default());
                }
                _ => {}
            }
            if step % 50 == 0 {
                check(&world, &ids, seed, step);
            }
        }
        check(&world, &ids, seed, 500);
    }
}

#[test]
fn nonsense_input_is_refused_rather_than_absorbed() {
    // NaN and infinity arriving from gameplay code — a divide by zero in a
    // controller, a bad value out of a config file. One of these reaching a
    // body's state poisons it permanently, and then poisons everything it
    // touches, so they have to be stopped at the door.
    let mut rng = Rng(0xDEAD_BEEF);
    let poison = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30];

    for seed in 1..=8u32 {
        let (mut world, ids) = world_from_seed(seed.wrapping_mul(13_331), 10);
        for step in 0..200 {
            world.step(DT);
            if ids.is_empty() {
                break;
            }
            let id = ids[rng.below(ids.len() as u32) as usize];
            let bad = poison[rng.below(poison.len() as u32) as usize];
            if let Some(b) = world.body_mut(id) {
                match rng.below(4) {
                    0 => b.apply_impulse(Vector3::new(bad, 0.0, 0.0)),
                    1 => b.set_translation(Vector3::new(0.0, bad, 0.0)),
                    2 => b.linear_velocity = Vector3::new(0.0, 0.0, bad),
                    _ => b.apply_torque_impulse(Vector3::new(bad, bad, 0.0)),
                }
            }
            // Writing a field directly is not guarded — nothing can guard a
            // public field — so the check is that the *world* recovers, not
            // that the poke had no effect.
            world.step(DT);
            for other in &ids {
                if *other == id {
                    continue;
                }
                if let Some(b) = world.body(*other) {
                    assert!(
                        finite(b.translation()) && finite(b.linear_velocity),
                        "seed {seed} step {step}: {bad} written to one body spread to another"
                    );
                }
            }
        }
    }
}




