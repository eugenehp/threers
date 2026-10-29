//! The GPU solver against the CPU one, on the gummy jellyfish.
//!
//! They are not meant to agree bit for bit — the GPU solves its constraints in
//! colour batches where the CPU goes one at a time — but they are the same
//! physics, so they must agree on what can be seen: resting, the body keeps
//! its volume and sits where the CPU's does; pulled, the same tentacle tears
//! at about the same moment.
//!
//! Skipped when no GPU adapter is available.

#[allow(dead_code)]
mod bench {
    use std::f32::consts::PI;
    use threers::math::Vector3;
    use threers::sdf::Shape;
    use threers::softbody::{ChainSpec, Creature, Marks, ShapeSpec};
    include!("../examples/gummy_creatures.inc");
}

use threers::softbody::gpu::GpuSolver;
use threers::softbody::{Event, SoftBody};

fn centroid(sim: &SoftBody) -> [f64; 3] {
    let n = sim.topos[0].n;
    let mut c = [0.0f64; 3];
    for p in sim.x[..3 * n].chunks_exact(3) {
        for (ck, &v) in c.iter_mut().zip(p) {
            *ck += v as f64 / n as f64;
        }
    }
    c
}

#[test]
fn at_rest_the_gpu_body_is_the_cpus() {
    let Some(gpu) = threers::gpu::testing::context("softbody gpu") else {
        return;
    };
    let mut cpu = bench::jellyfish().into_soft_body(7);
    let mut gsim = bench::jellyfish().into_soft_body(7);
    let mut solver = GpuSolver::new(gpu.device.clone(), gpu.queue.clone(), &mut gsim);
    for _ in 0..120 {
        cpu.step(1.0 / 60.0);
        solver.step(&mut gsim, 1.0 / 60.0);
    }
    assert!(gsim.x.iter().all(|v| v.is_finite()), "the GPU body went to NaN");
    let (mc, mg) = (cpu.measure(), gsim.measure());
    let (cc, cg) = (centroid(&cpu), centroid(&gsim));
    eprintln!("volume: cpu {:.4} gpu {:.4}; width: cpu {:.4} gpu {:.4}", mc.volume, mg.volume, mc.length, mg.length);
    eprintln!("centroid: cpu {cc:?} gpu {cg:?}");
    assert!((mg.volume - 1.0).abs() < 0.02, "the GPU body lost volume: {:.4}", mg.volume);
    assert!((mg.volume - mc.volume).abs() < 0.01, "volume {:.4} against the CPU's {:.4}", mg.volume, mc.volume);
    assert!((mg.length - mc.length).abs() < 0.01, "width {:.4} against the CPU's {:.4}", mg.length, mc.length);
    let d = ((cc[0] - cg[0]).powi(2) + (cc[1] - cg[1]).powi(2) + (cc[2] - cg[2]).powi(2)).sqrt();
    assert!(d < 0.005, "the bodies sit {d:.4} apart");
}

fn pull_until_torn(sim: &mut SoftBody, mut step: impl FnMut(&mut SoftBody)) -> Option<usize> {
    let slot = sim.leg_slots[0][0];
    let p = sim.pivot_now(0);
    let hand = [p[0] * 1.6, p[1] + 0.02, p[2] * 1.6];
    sim.start_grab(hand, slot);
    for frame in 1..=240 {
        if let Some(g) = sim.grab.as_mut() {
            let k = frame as f64;
            g.target = [hand[0] * (1.0 + 0.012 * k), hand[1] + 0.004 * k, hand[2] * (1.0 + 0.012 * k)];
        }
        step(sim);
        if sim.events.drain(..).any(|e| matches!(e, Event::Tear { leg: 0 })) {
            return Some(frame);
        }
    }
    None
}

#[test]
fn pulled_the_same_tentacle_tears_on_both() {
    let Some(gpu) = threers::gpu::testing::context("softbody gpu") else {
        return;
    };
    let mut cpu = bench::jellyfish().into_soft_body(7);
    let tc = pull_until_torn(&mut cpu, |s| s.step(1.0 / 60.0));
    let mut gsim = bench::jellyfish().into_soft_body(7);
    let mut solver = GpuSolver::new(gpu.device.clone(), gpu.queue.clone(), &mut gsim);
    let tg = pull_until_torn(&mut gsim, |s| solver.step(s, 1.0 / 60.0));
    eprintln!("torn at frame: cpu {tc:?} gpu {tg:?}");
    let (tc, tg) = (tc.expect("the CPU tentacle tears"), tg.expect("the GPU tentacle tears"));
    assert!(tc.abs_diff(tg) <= 12, "the GPU tore at {tg}, the CPU at {tc}");
    // and after: a loose copy, and a bud regrowing
    assert!(gsim.blocks.iter().any(|b| b.leg == 0 && b.slot > 0 && b.active));
    assert!(gsim.blocks[gsim.leg_slots[0][0]].grow < 1.0);
}

/// Not a check: the two solvers' step times (`--ignored --nocapture`).
#[test]
#[ignore]
fn step_times() {
    let Some(gpu) = threers::gpu::testing::context("softbody gpu") else {
        return;
    };
    for (name, make) in [("jellyfish", bench::jellyfish as fn() -> _), ("octopus", bench::octopus)] {
        let mut cpu = make().into_soft_body(7);
        let t = std::time::Instant::now();
        for _ in 0..120 {
            cpu.step(1.0 / 60.0);
        }
        let c = t.elapsed().as_secs_f64() * 1000.0 / 120.0;
        let mut g = make().into_soft_body(7);
        let mut solver = GpuSolver::new(gpu.device.clone(), gpu.queue.clone(), &mut g);
        for _ in 0..10 {
            solver.step(&mut g, 1.0 / 60.0);
        }
        let t = std::time::Instant::now();
        for _ in 0..120 {
            solver.step(&mut g, 1.0 / 60.0);
        }
        let gm = t.elapsed().as_secs_f64() * 1000.0 / 120.0;
        eprintln!("{name}: {} particles — cpu {c:.2} ms, gpu {gm:.2} ms a step (with the readback waited for)", cpu.n);
        // the same, the encoding (and the CPU half) apart from the GPU's own time
        solver.wait = false;
        let (mut enc, mut wait) = (0.0, 0.0);
        for _ in 0..120 {
            let t = std::time::Instant::now();
            solver.step(&mut g, 1.0 / 60.0);
            enc += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            wait += t.elapsed().as_secs_f64();
        }
        let batches = solver.batch_count();
        eprintln!(
            "  {:.2} ms on the CPU (prologue, encoding, the CPU half), {:.2} ms waiting on the GPU; {batches} colour batches a substep",
            enc * 1000.0 / 120.0,
            wait * 1000.0 / 120.0
        );
    }
}

/// Switching solvers mid-pull — what the demo's "auto" does when the GPU
/// starts or stops paying — keeps hold of the tentacle and the body whole:
/// it still tears, at about the frame either solver alone tears it.
#[test]
fn switching_solvers_mid_pull_keeps_the_grab() {
    let Some(gpu) = threers::gpu::testing::context("softbody gpu") else {
        return;
    };
    let mut sim = bench::jellyfish().into_soft_body(7);
    let mut solver = GpuSolver::new(gpu.device.clone(), gpu.queue.clone(), &mut sim);
    solver.suspend(&mut sim);
    let mut on_gpu = false;
    let mut frame = 0;
    let torn = pull_until_torn(&mut sim, |s| {
        frame += 1;
        // swap every seven frames, either way
        if frame % 7 == 0 {
            on_gpu = !on_gpu;
            if on_gpu { solver.resume(s) } else { solver.suspend(s) }
        }
        if on_gpu { solver.step(s, 1.0 / 60.0) } else { s.step(1.0 / 60.0) }
    });
    eprintln!("torn at frame {torn:?} while switching every 7 frames");
    assert!(sim.x.iter().all(|v| v.is_finite()), "switching sent the body to NaN");
    let t = torn.expect("the tentacle tears while the solvers swap");
    assert!((25..=50).contains(&t), "torn at {t}; alone, the CPU tears it at 35 and the GPU at 39");
    let v = sim.measure().volume;
    assert!((v - 1.0).abs() < 0.03, "the body's volume went to {v:.4}");
}
