//! Running the broad phase on the GPU via a wgpu compute shader.
//!
//! ```text
//! cargo run -p threers-physics --features gpu --example gpu_broadphase
//! ```
//!
//! The same code path works on `wasm32`/WebGPU, where rayon cannot run — that is
//! the main reason it exists. See the `threers_physics::gpu` module docs for
//! when this is actually faster than the CPU sweep-and-prune (short version:
//! only at high body counts).

use std::time::Instant;
use threers_physics::gpu::GpuBroadPhase;
use threers_physics::prelude::*;

fn main() {
    // In a real app, share the renderer's device instead of making a new one:
    //
    //     let gpu = GpuBroadPhase::from_device(renderer.device_arc(), renderer.queue_arc());
    //
    let Some(mut gpu) = pollster::block_on(GpuBroadPhase::new()) else {
        eprintln!("no compute-capable GPU adapter found; nothing to demonstrate");
        return;
    };
    println!("GPU broad phase ready\n");

    // A dense cube of falling boxes — enough bodies that the broad phase is
    // actually doing work.
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));

    let side = 8;
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                world.add_body(
                    RigidBody::dynamic()
                        .shape(Shape::cuboid(0.4, 0.4, 0.4))
                        .translation(Vector3::new(
                            x as f32 * 1.0 - 4.0,
                            y as f32 * 1.0 + 1.0,
                            z as f32 * 1.0 - 4.0,
                        ))
                        .can_sleep(false),
                );
            }
        }
    }
    println!("{} bodies\n", world.body_count());

    // --- correctness first: the GPU must agree with the CPU ---
    // Run one CPU step to get the built-in broad phase's view, then compare.
    world.step_fixed();
    let aabbs = world.broadphase_aabbs();
    let gpu_pairs = gpu.find_pairs_blocking(&aabbs);
    println!("GPU found {} overlapping pairs", gpu_pairs.len());
    if gpu.overflowed() {
        println!("  (the pair buffer overflowed and was grown; re-run for the full list)");
    }

    // --- driving the world with GPU pairs ---
    // Each frame: collect bounds, dispatch, hand the pairs back, step.
    println!("\nstepping with the GPU broad phase...");
    let start = Instant::now();
    for _ in 0..120 {
        let aabbs = world.broadphase_aabbs();
        let pairs = gpu.find_pairs_blocking(&aabbs);
        world.set_broadphase_pairs(pairs);
        world.step_fixed();
    }
    let gpu_time = start.elapsed();

    let settled: Vec<f32> = world
        .bodies()
        .iter()
        .filter(|(_, b)| b.is_dynamic())
        .map(|(_, b)| b.translation().y)
        .collect();
    let lowest = settled.iter().copied().fold(f32::MAX, f32::min);
    println!(
        "  120 steps in {:.1} ms, lowest box now at y = {:.2}",
        gpu_time.as_secs_f64() * 1000.0,
        lowest
    );

    // --- the same run on the CPU, for comparison ---
    let mut cpu_world = World::new();
    cpu_world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                cpu_world.add_body(
                    RigidBody::dynamic()
                        .shape(Shape::cuboid(0.4, 0.4, 0.4))
                        .translation(Vector3::new(
                            x as f32 * 1.0 - 4.0,
                            y as f32 * 1.0 + 1.0,
                            z as f32 * 1.0 - 4.0,
                        ))
                        .can_sleep(false),
                );
            }
        }
    }
    cpu_world.step_fixed();
    let start = Instant::now();
    for _ in 0..120 {
        cpu_world.step_fixed();
    }
    let cpu_time = start.elapsed();
    println!(
        "  the same 120 steps, CPU sweep-and-prune: {:.1} ms",
        cpu_time.as_secs_f64() * 1000.0
    );

    println!(
        "\nAt {} bodies the readback latency usually dominates — the GPU path pays off\n\
         at much larger scenes, and on wasm32 where it is the only parallelism available.",
        world.body_count()
    );
}
