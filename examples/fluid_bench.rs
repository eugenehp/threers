//! Where a fluid frame goes, measured rather than guessed.
//!
//!     cargo run --release --features fluid --example fluid_bench
//!
//! Prints the cost of each stage at each quality level, and the residual
//! divergence each pressure solver leaves — the two numbers that trade against
//! each other. Knobs:
//!
//! * `FLUID_SIZE=WxH` — canvas size (default 1920x1080).
//! * `FLUID_FRAMES=<n>` — frames per measurement (default 60).
//!
//! # How it times
//!
//! Each stage is submitted on its own and the queue drained before the clock
//! is read, which is the only way to attribute GPU time without timestamp
//! queries — and `TIMESTAMP_QUERY` is not on every adapter. The serialisation
//! costs a submit per stage, so the *total* here is pessimistic against a real
//! frame; the *split between stages* is what it is for.

use std::time::Instant;

use threers::fluid::{Advection, FluidConfig, FluidSim, PressureSolver, QUALITY_LEVELS};
use threers::gpu::GpuContext;

fn main() {
    let (width, height) = size();
    let frames = env("FLUID_FRAMES").unwrap_or(60u32);

    let gpu = match GpuContext::headless_blocking() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    println!(
        "{}  {width}x{height}, {frames} frames a measurement\n",
        gpu.describe()
    );

    // A target to composite into, so `render` is measured doing what it
    // actually does rather than into nothing.
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fluid bench target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let drain = || {
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    };
    // Milliseconds per call, averaged, with the queue drained either side.
    let time = |label: &str, n: u32, mut f: Box<dyn FnMut() + '_>| -> f32 {
        f();
        drain();
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        drain();
        let ms = t.elapsed().as_secs_f32() * 1000.0 / n as f32;
        println!("    {label:<22} {ms:7.3} ms");
        ms
    };

    println!("stage costs, by quality level");
    for level in QUALITY_LEVELS {
        let config = FluidConfig::default().with_quality(level);
        let Ok(mut sim) =
            FluidSim::new(gpu.device.clone(), gpu.queue.clone(), width, height, config)
        else {
            continue;
        };
        let ((dw, dh), (sw, sh)) = sim.grid_sizes();
        println!(
            "\n  {} — dye {dw}x{dh}, sim {sw}x{sh}, {} MB",
            level.name,
            sim.memory_bytes() / (1024 * 1024)
        );
        sim.multiple_splats(8);
        for _ in 0..8 {
            sim.update(1.0 / 60.0);
        }

        let step = time("step (solver)", frames, Box::new(|| sim.step(1.0 / 60.0)));

        // Each display stage on its own, by difference: a stage's cost is what
        // the frame stops costing when it is switched off.
        let all = {
            let c = FluidConfig {
                shading: true,
                bloom: true,
                sunrays: true,
                ..*sim.config()
            };
            sim.set_config(c);
            time(
                "render (everything)",
                frames,
                Box::new(|| sim.render(&target)),
            )
        };
        let no_bloom = {
            let c = FluidConfig {
                bloom: false,
                ..*sim.config()
            };
            sim.set_config(c);
            time("  without bloom", frames, Box::new(|| sim.render(&target)))
        };
        let no_rays = {
            let c = FluidConfig {
                sunrays: false,
                ..*sim.config()
            };
            sim.set_config(c);
            time(
                "  without sunrays too",
                frames,
                Box::new(|| sim.render(&target)),
            )
        };
        let bare = {
            let c = FluidConfig {
                shading: false,
                ..*sim.config()
            };
            sim.set_config(c);
            time("  dye only", frames, Box::new(|| sim.render(&target)))
        };
        println!(
            "    => bloom {:.3}, sunrays {:.3}, shading {:.3}, composite {:.3} ms",
            all - no_bloom,
            no_bloom - no_rays,
            no_rays - bare,
            bare
        );
        println!(
            "    => a frame is {:.3} ms, {:.0} fps, {:.0}% of it the solver",
            step + all,
            1000.0 / (step + all),
            100.0 * step / (step + all)
        );

        // A burst, with no step in it: the splat passes alone.
        let burst = time("25 splats", 16, Box::new(|| sim.multiple_splats(25)));
        let one = time("1 splat", 16, Box::new(|| sim.multiple_splats(1)));
        println!(
            "    => {:.3} ms for the batch, {:.3} for one — {:.1}x cheaper than \
             one each",
            burst,
            one,
            one * 25.0 / burst
        );
    }

    println!("\nadvection at `high` — the extra pass, on the two biggest grids");
    {
        let Ok(mut sim) = FluidSim::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            width,
            height,
            FluidConfig::default(),
        ) else {
            return;
        };
        for advection in [Advection::SemiLagrangian, Advection::MacCormack] {
            sim.set_config(FluidConfig {
                advection,
                ..*sim.config()
            });
            sim.multiple_splats(6);
            for _ in 0..4 {
                sim.update(1.0 / 60.0);
            }
            time(
                &format!("{advection:?}"),
                frames,
                Box::new(|| sim.step(1.0 / 60.0)),
            );
        }
    }

    println!("\npressure solvers at `high` — cost against what they leave behind");
    let Ok(mut sim) = FluidSim::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        width,
        height,
        FluidConfig::default(),
    ) else {
        return;
    };
    for (label, solver, iters, cycles) in [
        ("jacobi 20 (default)", PressureSolver::Jacobi, 20u32, 1u32),
        ("jacobi 80", PressureSolver::Jacobi, 80, 1),
        ("jacobi 320", PressureSolver::Jacobi, 320, 1),
        ("multigrid 1 cycle", PressureSolver::Multigrid, 20, 1),
        ("multigrid 2 cycles", PressureSolver::Multigrid, 20, 2),
    ] {
        let config = FluidConfig {
            pressure_solver: solver,
            pressure_iterations: iters,
            multigrid_cycles: cycles,
            gradient_scale: 0.5,
            curl: 0.0,
            ..FluidConfig::default()
        };
        sim.set_config(config);
        sim.multiple_splats(6);
        for _ in 0..4 {
            sim.update(1.0 / 60.0);
        }
        let ms = time(label, frames, Box::new(|| sim.step(1.0 / 60.0)));
        let _ = ms;
    }
    println!(
        "\n(residuals for the same solvers are in `cargo test --features fluid \\
         --lib -- fluid::tests::multigrid --nocapture`)"
    );
}

fn size() -> (u32, u32) {
    std::env::var("FLUID_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1920, 1080))
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse().ok()
}
