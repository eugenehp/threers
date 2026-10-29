//! Where a soft-body step spends its time: the gummy jellyfish and octopus of
//! `web/examples/creatures.js`, built from the same shapes, run for 300 steps
//! with the solver's phase timer on.
//!
//! ```text
//! cargo run --release --example softbody_bench           # the CPU solver, by phase
//! cargo run --release --example softbody_bench -- --gpu  # and the WebGPU-compute one
//! ```

use std::f32::consts::PI;
use std::time::Instant;

use threers::math::Vector3;
use threers::sdf::Shape;
use threers::softbody::{ChainSpec, Creature, Marks, ShapeSpec, SoftBody};

include!("gummy_creatures.inc");

fn clock() -> f64 {
    thread_local!(static T0: Instant = Instant::now());
    T0.with(|t| t.elapsed().as_secs_f64() * 1000.0)
}

fn bench(name: &str, mut sim: SoftBody, steps: usize) {
    sim.clock = Some(clock);
    // a hand on the first part, pulling, so the glue and the tear are in it
    let slot = sim.leg_slots[0][0];
    let p = sim.pivot_now(0);
    sim.start_grab([p[0] * 1.6, p[1] + 0.02, p[2] * 1.6], slot);
    let t = Instant::now();
    for k in 0..steps {
        if let Some(g) = sim.grab.as_mut() {
            g.target[1] += 0.002;
            let _ = k;
        }
        sim.step(1.0 / 60.0);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let p = sim.profile;
    let n = p.steps as f64;
    println!("{name}: {} particles, {ms:.2} ms a step", sim.n);
    for (k, v) in [
        ("integrate", p.integrate),
        ("grab", p.grab),
        ("edges", p.edges),
        ("volumes", p.volumes),
        ("glue", p.glue),
        ("floor+walls", p.contact),
        ("damping", p.damping),
        ("chains", p.chains),
        ("drops", p.drops),
        ("frame (growth, muscles, hash, nodeF)", p.frame),
    ] {
        println!("  {k:<38} {:7.3} ms  {:5.1} %", v / n, 100.0 * v / n / ms);
    }
}

/// The same creature on the GPU solver: the CPU's share of a step (the
/// prologue, encoding, the CPU half) apart from the GPU's own time.
fn bench_gpu(name: &str, mut sim: SoftBody, steps: usize) {
    let Some(gpu) = threers::gpu::testing::context("softbody bench") else {
        println!("{name} on the GPU: no adapter");
        return;
    };
    let mut solver = threers::softbody::gpu::GpuSolver::new(gpu.device.clone(), gpu.queue.clone(), &mut sim);
    solver.wait = false;
    let (mut cpu, mut wait) = (0.0, 0.0);
    for k in 0..steps + 10 {
        let t = Instant::now();
        solver.step(&mut sim, 1.0 / 60.0);
        let c = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if k >= 10 {
            cpu += c;
            wait += t.elapsed().as_secs_f64();
        }
    }
    let n = steps as f64;
    println!(
        "{name} on the GPU: {:.2} ms a step on the CPU, {:.2} ms on the GPU; {} dispatches a substep",
        cpu * 1000.0 / n,
        wait * 1000.0 / n,
        solver.batch_count()
    );
}

fn main() {
    let gpu = std::env::args().any(|a| a == "--gpu");
    let t = Instant::now();
    let j = jellyfish();
    println!("jellyfish built in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    bench("jellyfish", j.into_soft_body(7), 300);
    if gpu {
        bench_gpu("jellyfish", jellyfish().into_soft_body(7), 300);
    }
    let t = Instant::now();
    let o = octopus();
    println!("octopus built in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    bench("octopus", o.into_soft_body(7), 300);
    if gpu {
        bench_gpu("octopus", octopus().into_soft_body(7), 300);
    }
}
