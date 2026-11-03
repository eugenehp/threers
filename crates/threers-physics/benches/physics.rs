//! What the engine costs, measured.
//!
//! These exist to catch silent regressions, not to win comparisons. The tuning
//! defaults — four substeps, four iterations, the BVH margins — were each chosen
//! by measuring, and without something watching them a change that triples the
//! cost of a step looks exactly like a change that does not.
//!
//! Run everything with `cargo bench -p threers-physics`, or one group with
//! `cargo bench -p threers-physics -- stack`. Criterion writes a baseline into
//! `target/criterion` and reports the delta on the next run, so the useful
//! workflow is: measure, change, measure again.
//!
//! The scenes are deliberately dull — a stack, a pile, a chain — because those
//! are the shapes real content takes, and because a benchmark nobody can picture
//! is a benchmark nobody will trust when it moves.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use threers_physics::prelude::*;

/// A deterministic scatter, so every run measures the same scene.
///
/// Nothing here should depend on the *quality* of the randomness; it just has to
/// be the same jumble every time, which a real generator would not give us for
/// free across platforms.
struct Scatter(u32);

impl Scatter {
    fn next(&mut self) -> f32 {
        // xorshift32, then mapped to -0.5..0.5.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0 - 0.5
    }
}

fn with_ground() -> World {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
    world
}

/// `height` boxes stacked squarely on top of one another.
fn stack(height: usize) -> World {
    let mut world = with_ground();
    for i in 0..height {
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5 + i as f32 * 1.01, 0.0))
                .friction(0.8),
        );
    }
    world
}

/// `count` bodies dropped into a heap, cycling through the primitive shapes.
fn pile(count: usize) -> World {
    let mut world = with_ground();
    let mut scatter = Scatter(0x9E3779B9);
    let side = (count as f32).cbrt().ceil() as usize;
    for i in 0..count {
        let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
        let shape = match i % 4 {
            0 => Shape::cuboid(0.4, 0.4, 0.4),
            1 => Shape::ball(0.45),
            2 => Shape::capsule(0.3, 0.25),
            _ => Shape::cylinder(0.35, 0.35),
        };
        world.add_body(
            RigidBody::dynamic()
                .shape(shape)
                .translation(Vector3::new(
                    x as f32 * 1.1 - side as f32 * 0.55 + scatter.next() * 0.1,
                    0.6 + y as f32 * 1.1,
                    z as f32 * 1.1 - side as f32 * 0.55 + scatter.next() * 0.1,
                ))
                .friction(0.6),
        );
    }
    world
}

/// A hanging chain of `links` boxes — the joint solver's worst case, because
/// every link's error feeds into the next.
fn chain(links: usize) -> World {
    let mut world = World::new();
    let mut previous = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(0.1, 0.1, 0.1))
            .translation(Vector3::new(0.0, links as f32 * 0.5 + 1.0, 0.0)),
    );
    for i in 0..links {
        let y = links as f32 * 0.5 + 1.0 - (i as f32 + 1.0) * 0.5;
        let link = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.05, 0.2, 0.05))
                .translation(Vector3::new(0.0, y, 0.0))
                .can_sleep(false),
        );
        world.add_joint(Joint::spherical(
            previous,
            link,
            Vector3::new(0.0, if i == 0 { 0.0 } else { -0.2 }, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
        ));
        previous = link;
    }
    world
}

/// Step a world until it stops changing much, so a benchmark measures the
/// steady state rather than the initial collapse.
fn settled(mut world: World, steps: usize) -> World {
    for _ in 0..steps {
        world.step_fixed();
    }
    world
}

// ---- the step itself ------------------------------------------------------

fn bench_stack(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack");
    for height in [10usize, 50, 100] {
        group.bench_function(format!("{height}_boxes"), |b| {
            b.iter_batched_ref(
                || stack(height),
                |world| world.step_fixed(),
                BatchSize::SmallInput,
            )
        });
        // The same stack once it has gone to sleep. A settled scene should cost
        // almost nothing — if this ever approaches the active number, sleeping
        // has stopped working and every idle scene in every game got slower.
        group.bench_function(format!("{height}_boxes_asleep"), |b| {
            b.iter_batched_ref(
                || settled(stack(height), 400),
                |world| world.step_fixed(),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_pile(c: &mut Criterion) {
    let mut group = c.benchmark_group("pile");
    group.sample_size(30);
    for count in [125usize, 500, 1000] {
        group.bench_function(format!("{count}_bodies"), |b| {
            b.iter_batched_ref(
                || settled(pile(count), 30),
                |world| world.step_fixed(),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain");
    for links in [10usize, 50] {
        group.bench_function(format!("{links}_links"), |b| {
            b.iter_batched_ref(
                || settled(chain(links), 20),
                |world| world.step_fixed(),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ---- the tuning knobs -----------------------------------------------------

/// Substeps against iterations at matched total work.
///
/// This is the measurement the default is built on: four substeps of four
/// iterations beats one step of sixteen by a wide margin in accuracy. These
/// numbers are the other half of that trade — what the accuracy costs.
fn bench_solver_budget(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_budget");
    for (substeps, iterations) in [(1usize, 16usize), (2, 8), (4, 4), (8, 2), (16, 1)] {
        group.bench_function(format!("{substeps}x{iterations}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut world = settled(pile(200), 60);
                    world.substeps = substeps;
                    world.solver_config.velocity_iterations = iterations;
                    world
                },
                |world| world.step_fixed(),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---- queries --------------------------------------------------------------

fn bench_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("queries");
    // Queries do not mutate, so one settled world serves every sample.
    let world = settled(pile(1000), 60);
    let mut scatter = Scatter(0x1234_5678);
    let rays: Vec<Ray> = (0..256)
        .map(|_| {
            Ray::new(
                Vector3::new(scatter.next() * 20.0, 30.0, scatter.next() * 20.0),
                Vector3::new(scatter.next() * 0.3, -1.0, scatter.next() * 0.3),
            )
        })
        .collect();

    group.bench_function("raycast_first", |b| {
        b.iter(|| {
            let mut hits = 0;
            for ray in &rays {
                if world.raycast(ray, 100.0, QueryFilter::default()).is_some() {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });

    group.bench_function("raycast_all", |b| {
        b.iter(|| {
            let mut hits = 0;
            for ray in &rays {
                hits += world.raycast_all(ray, 100.0, QueryFilter::default()).len();
            }
            black_box(hits)
        })
    });

    let probe = Shape::ball(1.0);
    group.bench_function("shape_cast", |b| {
        b.iter(|| {
            let mut hits = 0;
            for ray in &rays {
                let start = Isometry::from_translation(ray.origin);
                if world
                    .cast_shape(&probe, &start, ray.direction, 100.0, QueryFilter::default())
                    .is_some()
                {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });

    group.bench_function("overlap", |b| {
        b.iter(|| {
            let mut found = 0;
            for ray in &rays {
                let at = Isometry::from_translation(Vector3::new(ray.origin.x, 1.0, ray.origin.z));
                found += world
                    .intersections_with_shape(&probe, &at, QueryFilter::default())
                    .len();
            }
            black_box(found)
        })
    });

    group.bench_function("project_point", |b| {
        b.iter(|| {
            let mut total = 0.0f32;
            for ray in &rays {
                if let Some(p) = world.project_point(ray.origin, QueryFilter::default()) {
                    total += p.point.y;
                }
            }
            black_box(total)
        })
    });

    group.finish();
}

// ---- narrow phase ---------------------------------------------------------

/// One pair of shapes, collided head-on a thousand times.
///
/// Isolating this matters because the convex path (GJK, then EPA when they
/// overlap) is a different order of magnitude from the analytic ones, and a
/// change that pushes a common pair onto the general path would otherwise hide
/// inside the whole-scene numbers.
fn bench_narrowphase(c: &mut Criterion) {
    use threers_physics::narrowphase::{collide, RawManifold};

    let mut group = c.benchmark_group("narrowphase");
    let hull = Shape::convex_hull(&[
        Vector3::new(-0.5, -0.5, -0.5),
        Vector3::new(0.6, -0.4, -0.5),
        Vector3::new(-0.4, 0.6, -0.45),
        Vector3::new(-0.5, -0.5, 0.55),
        Vector3::new(0.5, 0.5, 0.5),
        Vector3::new(0.2, -0.6, 0.3),
    ])
    .expect("six points in general position make a hull");

    let pairs: [(&str, Shape, Shape); 6] = [
        ("ball_ball", Shape::ball(0.5), Shape::ball(0.5)),
        (
            "cuboid_cuboid",
            Shape::cuboid(0.5, 0.5, 0.5),
            Shape::cuboid(0.5, 0.5, 0.5),
        ),
        ("ball_cuboid", Shape::ball(0.5), Shape::cuboid(0.5, 0.5, 0.5)),
        (
            "capsule_capsule",
            Shape::capsule(0.4, 0.25),
            Shape::capsule(0.4, 0.25),
        ),
        (
            "cuboid_ground",
            Shape::cuboid(0.5, 0.5, 0.5),
            Shape::ground(),
        ),
        ("hull_hull", hull.clone(), hull),
    ];

    for (name, a, b) in &pairs {
        // Overlapping by 0.1, which is the case that reaches EPA.
        let iso_a = Isometry::from_translation(Vector3::new(0.0, 0.45, 0.0));
        let iso_b = Isometry::from_translation(Vector3::ZERO);
        group.bench_function(*name, |bench| {
            let mut out: Vec<RawManifold> = Vec::new();
            bench.iter(|| {
                out.clear();
                collide(a, &iso_a, b, &iso_b, 0.02, &mut out);
                black_box(out.len())
            })
        });
    }
    group.finish();
}

// ---- broad phase ----------------------------------------------------------

fn bench_broadphase(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadphase");
    for count in [500usize, 5000] {
        // Everything moving, which is the case the tree cannot skip.
        group.bench_function(format!("{count}_moving"), |b| {
            b.iter_batched_ref(
                || {
                    let mut world = pile(count);
                    world.gravity = Vector3::new(0.5, -9.81, 0.3);
                    settled(world, 5)
                },
                |world| world.step_fixed(),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_stack,
    bench_pile,
    bench_chain,
    bench_solver_budget,
    bench_queries,
    bench_narrowphase,
    bench_broadphase,
);
criterion_main!(benches);
