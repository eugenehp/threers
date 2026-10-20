//! Where a *volume* frame goes, measured rather than guessed.
//!
//!     cargo run --release --features fluid --example fluid3d_bench
//!
//! A volume costs the cube of its side, so the question "what resolution can
//! this machine run" has a different answer at every stage of the frame. This
//! prints the three numbers that decide it: the solve, the ray march, and the
//! memory.
//!
//! Knobs:
//!
//! * `FLUID_SIZE=WxH` — output size for the march (default 1280x720).
//! * `FLUID_FRAMES=<n>` — frames per measurement (default 30).
//! * `FLUID_REPEATS=<n>` — measurements per number, best taken (default 5).
//! * `FLUID_GRIDS=a,b,c` — solver resolutions to try (default 32,48,64,96).
//! * `FLUID_CPU_GRIDS=a,b,c` — resolutions for the CPU/GPU comparison.
//! * `FLUID_COMPACT=1` — store the dye in the packed `rg11b10ufloat` instead
//!   of `rgba16float`, to measure what the packing is worth.
//! * `FLUID_SKIP=0` — march every step instead of stepping over the empty
//!   bricks.
//!
//! # How it times
//!
//! Each measurement submits its work and drains the queue before the clock is
//! read. `TIMESTAMP_QUERY` is not on every adapter, so this is the portable
//! way to attribute GPU time; the cost is a submit per measurement, which
//! makes the absolute numbers pessimistic and the *ratios* — which is what
//! this is for — sound.
//!
//! `jacobi` and `v-cycle` are both whole steps, with the solver swapped. The
//! two are not comparable pass for pass — a V-cycle is a chain of small grids
//! where sweeping is one grid over and over — so what is worth comparing is
//! what a frame pays, against what the solve is worth: the residual numbers
//! are in `the_v_cycle_beats_the_sweeps`.
//!
//! Every number here is the *best* of several runs. The fastest run is the
//! one with the least interference from everything else on the machine; an
//! average over a shared GPU measures the neighbours.

use std::time::Instant;

use threers::fluid::{FluidConfig3d, FluidSim3d, PressureSolver};
use threers::gpu::GpuContext;
use threers::{PerspectiveCamera, Vector3};

fn main() {
    let (width, height) = size();
    let frames = env("FLUID_FRAMES").unwrap_or(30u32);
    let repeats = env("FLUID_REPEATS").unwrap_or(5u32);
    let grids: Vec<u32> = std::env::var("FLUID_GRIDS")
        .ok()
        .map(|v| v.split(',').filter_map(|g| g.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![32, 48, 64, 96]);

    let gpu = match GpuContext::headless_blocking() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    let feats = gpu.adapter.features();
    println!(
        "{}  march {width}x{height}, {frames} frames a measurement",
        gpu.describe()
    );
    println!(
        "timestamps: query {}, inside encoders {}, inside passes {}\n",
        feats.contains(wgpu::Features::TIMESTAMP_QUERY),
        feats.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        feats.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
    );

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fluid3d bench target"),
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
    let view = target.create_view(&Default::default());
    let drain = || {
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    };
    let camera = {
        let mut c = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.05, 20.0);
        c.position = Vector3::new(1.1, 0.18, 1.24);
        c.look_at(Vector3::new(0.0, -0.02, 0.0));
        c
    };

    // Time `body` over `frames` runs *without waiting for the GPU*: what the
    // CPU spends recording the work. A stage whose total is its `cpu` is not
    // waiting on the GPU at all, and making its shaders faster buys nothing.
    let cpu_time = |body: &mut dyn FnMut()| -> f64 {
        for _ in 0..frames.min(8) {
            body();
        }
        drain();
        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            let t = Instant::now();
            for _ in 0..frames {
                body();
            }
            best = best.min(t.elapsed().as_secs_f64() * 1000.0 / frames as f64);
            drain();
        }
        best
    };

    // Time `body` over `frames` runs, in milliseconds a run, best of
    // `repeats`.
    let time = |body: &mut dyn FnMut()| -> f64 {
        for _ in 0..frames.min(8) {
            body();
        }
        drain();
        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            let t = Instant::now();
            for _ in 0..frames {
                body();
            }
            drain();
            best = best.min(t.elapsed().as_secs_f64() * 1000.0 / frames as f64);
        }
        best
    };

    println!(
        "{:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}",
        "solver", "dye", "step", "cpu", "jacobi", "v-cycle", "march", "frame", "memory"
    );
    println!("{}", "─".repeat(84));

    for &grid in &grids {
        let base = FluidConfig3d {
            resolution: grid,
            dye_resolution: grid * 2,
            compact_dye: env::<u32>("FLUID_COMPACT").unwrap_or(0) != 0,
            empty_space_skip: env::<u32>("FLUID_SKIP").unwrap_or(1) != 0,
            ..FluidConfig3d::default()
        };
        let sweeps = base.pressure_iterations;
        let extra = 40;

        let Some(mut sim) = build(&gpu, base) else {
            continue;
        };
        sim.multiple_splats(6);
        let (solver, dye) = sim.grids();
        let bytes = sim.memory_bytes();

        let step = time(&mut || sim.step(1.0 / 60.0));
        let cpu = cpu_time(&mut || sim.step(1.0 / 60.0));

        // What each solver costs, as the whole step: the two are not
        // interchangeable pass for pass, and the step is what a frame pays.
        let with_solver = |sim: &mut FluidSim3d, solver, cycles| {
            sim.set_config(FluidConfig3d {
                pressure_solver: solver,
                multigrid_cycles: cycles,
                ..base
            });
        };
        with_solver(&mut sim, PressureSolver::Jacobi, 1);
        let jacobi = time(&mut || sim.step(1.0 / 60.0));
        with_solver(&mut sim, PressureSolver::Multigrid, 1);
        let vcycle = time(&mut || sim.step(1.0 / 60.0));
        sim.set_config(base);
        let _ = (sweeps, extra);

        let march = time(&mut || {
            sim.render_to_view(
                &camera,
                &view,
                wgpu::TextureFormat::Rgba8Unorm,
                (width, height),
            )
        });

        println!(
            "{solver:>4}³  {dye:>4}³  {step:>7.2}ms  {cpu:>7.2}ms  {jacobi:>7.2}ms  {vcycle:>7.2}ms  {march:>7.2}ms  {:>7.2}ms  {:>5} MB",
            step + march,
            bytes / 1_048_576,
        );
    }

    // How much of a plume is nothing at all.
    //
    // This is the number that says what empty-space skipping could buy: a ray
    // marching a volume that is nine tenths empty spends nine tenths of its
    // samples fetching zero. Bricks, not voxels, because a skip has to be
    // coarse enough to be worth testing.
    {
        let base = FluidConfig3d {
            compact_dye: env::<u32>("FLUID_COMPACT").unwrap_or(0) != 0,
            ..FluidConfig3d::default()
        };
        if let Some(mut sim) = build(&gpu, base) {
            sim.multiple_splats(6);
            for _ in 0..40 {
                sim.step(1.0 / 60.0);
            }
            let dye = threers::renderer::readback::texture_channel_f32(
                &gpu.device,
                &gpu.queue,
                sim.dye_texture(),
                0,
            );
            let atlas = sim.dye_atlas();
            let (aw, _) = atlas.size();
            let n = atlas.grid as usize;
            let across = (aw / atlas.grid) as usize;
            let at = |x: usize, y: usize, z: usize| -> f32 {
                let (row, col) = (z / across, z % across);
                dye[(row * n + y) * aw as usize + col * n + x]
            };
            for brick in [4usize, 8] {
                let (mut empty, mut total) = (0u64, 0u64);
                for bz in (0..n).step_by(brick) {
                    for by in (0..n).step_by(brick) {
                        for bx in (0..n).step_by(brick) {
                            let mut peak = 0.0f32;
                            for z in bz..(bz + brick).min(n) {
                                for y in by..(by + brick).min(n) {
                                    for x in bx..(bx + brick).min(n) {
                                        peak = peak.max(at(x, y, z));
                                    }
                                }
                            }
                            total += 1;
                            // Below this a brick cannot change a pixel: the
                            // march's own early-out gives up at 0.003
                            // transmittance.
                            if peak < 0.003 {
                                empty += 1;
                            }
                        }
                    }
                }
                println!(
                    "\n{brick}³ bricks of a settled plume: {:.1}% hold nothing \
                     ({empty} of {total})",
                    100.0 * empty as f64 / total as f64
                );
            }
        }
    }

    // What keeping the bind groups and uniform blocks is worth, both ways
    // round in one process. The step is where it shows: recording fifty
    // passes is most of what the CPU does in a frame, and all fifty say the
    // same thing they said last frame.
    println!("\nkeeping the bind groups between frames:");
    {
        let both = |cache: bool| -> Option<(f64, f64)> {
            let mut sim = build(
                &gpu,
                FluidConfig3d {
                    cache_bindings: cache,
                    ..FluidConfig3d::default()
                },
            )?;
            sim.multiple_splats(6);
            let step = time(&mut || sim.step(1.0 / 60.0));
            let cpu = cpu_time(&mut || sim.step(1.0 / 60.0));
            Some((step, cpu))
        };
        if let (Some(on), Some(off)) = (both(true), both(false)) {
            println!("  cached  step {:>6.2}ms  cpu {:>6.2}ms", on.0, on.1);
            println!(
                "  rebuilt step {:>6.2}ms  cpu {:>6.2}ms   ({:.2}x on the cpu)",
                off.0,
                off.1,
                off.1 / on.1
            );
        }
    }

    // What stepping over the empty bricks is worth, both ways round in one
    // process: the march alone, on a plume that has been left to rise.
    //
    // *On a plume*, because what it saves depends entirely on what is in the
    // volume. Six fresh splats fill the box and leave nothing to skip; a
    // volume the dissipation has emptied is all skip and no march. Neither is
    // the picture anybody renders.
    println!("\nstepping over empty bricks, whole frames on a settled plume:");
    {
        let plume = |skip: bool| -> Option<f64> {
            let mut sim = build(
                &gpu,
                FluidConfig3d {
                    empty_space_skip: skip,
                    buoyancy: 46.0,
                    curl: 26.0,
                    density_dissipation: 0.2,
                    ..FluidConfig3d::default()
                },
            )?;
            for frame in 0..52 {
                if frame < 34 && frame % 3 == 0 {
                    sim.splat([0.5, 0.11, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
                }
                sim.step(1.0 / 60.0);
            }
            // A whole frame, not just the march: the occupancy has to be
            // rebuilt whenever the dye moves, so timing a render loop that
            // never steps would count the saving and not the cost.
            Some(time(&mut || {
                sim.step(1.0 / 60.0);
                sim.render_to_view(
                    &camera,
                    &view,
                    wgpu::TextureFormat::Rgba8Unorm,
                    (width, height),
                );
            }))
        };
        if let (Some(on), Some(off)) = (plume(true), plume(false)) {
            println!("  skipping   {on:>7.2}ms");
            println!("  every step {off:>7.2}ms   ({:.2}x)", off / on);
        }
    }

    // The same solver on the CPU, against the same solver on the GPU.
    //
    // Step only — no ray march — because that is the part both of them do.
    // Same configuration, same stages, same V-cycle; the CPU version is a
    // transcription of the shaders and agrees with them on what it leaves
    // behind (0.337 of a puff's divergence against 0.324).
    {
        println!("\nthe step, cpu against gpu (ms a step, and what 30 fps allows):");
        println!(
            "  {:>6} {:>6}  {:>10} {:>10} {:>10}  {:>8}",
            "solver", "dye", "cpu 1 core", "cpu all", "gpu", "cpu/gpu"
        );
        let cpu_grids: Vec<u32> = std::env::var("FLUID_CPU_GRIDS")
            .ok()
            .map(|v| v.split(',').filter_map(|g| g.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![16, 24, 32, 48, 64, 96]);
        for &n in &cpu_grids {
            let cfg = FluidConfig3d {
                resolution: n,
                dye_resolution: n * 2,
                buoyancy: 46.0,
                curl: 26.0,
                ..FluidConfig3d::default()
            };
            // Fewer frames as the grid grows: the CPU is the slow one and
            // the numbers are stable long before the clock is.
            let runs = (frames / (1 + n / 24)).max(3);
            let cpu = |threaded: bool| -> f64 {
                let mut sim = threers::fluid::CpuFluid3d::new(cfg);
                sim.threaded = threaded;
                sim.splat([0.5, 0.2, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
                sim.step(1.0 / 60.0);
                let t = Instant::now();
                for _ in 0..runs {
                    sim.step(1.0 / 60.0);
                }
                t.elapsed().as_secs_f64() * 1000.0 / runs as f64
            };
            let one = cpu(false);
            let all = cpu(true);
            let gpu = match build(&gpu, cfg) {
                Some(mut sim) => {
                    sim.splat([0.5, 0.2, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
                    let mut best = f64::INFINITY;
                    for _ in 0..repeats {
                        let t = Instant::now();
                        for _ in 0..frames {
                            sim.step(1.0 / 60.0);
                        }
                        drain();
                        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / frames as f64);
                    }
                    best
                }
                None => f64::NAN,
            };
            println!(
                "  {n:>4}³  {:>4}³  {one:>9.2}ms {all:>9.2}ms {gpu:>9.2}ms  {:>7.0}x",
                n * 2,
                all / gpu
            );
        }
    }

    // Where the solve's time goes.
    //
    // A V-cycle on a 64³ grid is about forty render passes and most of them
    // are on grids of 32³ down to 4³, where the arithmetic is free and the
    // pass is not. If that is what the solve costs, then stopping the
    // coarsening earlier buys back most of it — and the residual says what
    // it costs in convergence.
    if gpu
        .adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        println!("\nthe solve, against how far it coarsens:");
        println!(
            "  {:>8} {:>7}  {:>9}  {:>9}",
            "coarsest", "levels", "solve", "whole step"
        );
        for coarsest in [4u32, 8, 16, 32] {
            let base = FluidConfig3d {
                multigrid_coarsest: coarsest,
                buoyancy: 46.0,
                curl: 26.0,
                ..FluidConfig3d::default()
            };
            let Some(mut sim) = build(&gpu, base) else {
                continue;
            };
            if !sim.profile() {
                break;
            }
            sim.multiple_splats(6);
            let mut solve = f32::INFINITY;
            for _ in 0..frames.max(60) {
                sim.step(1.0 / 60.0);
                sim.render_to_view(
                    &camera,
                    &view,
                    wgpu::TextureFormat::Rgba8Unorm,
                    (width, height),
                );
                let st = sim.stage_ms();
                if st[2] > 0.0 {
                    solve = solve.min(st[2]);
                }
            }
            let mut total = f32::INFINITY;
            for _ in 0..frames.max(60) {
                sim.step(1.0 / 60.0);
                sim.render_to_view(
                    &camera,
                    &view,
                    wgpu::TextureFormat::Rgba8Unorm,
                    (width, height),
                );
                let st: f32 = sim.stage_ms().iter().sum();
                if st > 0.0 {
                    total = total.min(st);
                }
            }
            println!(
                "  {coarsest:>7}³ {:>7}  {:>7.3}ms  {:>7.3}ms",
                sim.solver_levels(),
                solve,
                total
            );
        }
    }

    // Whether the optimisations stack, on the GPU's own clock.
    //
    // Wall-clock whole-frame timing cannot answer this on a machine with
    // anything else running: the effects are tens of percent and the spread
    // is more. Timestamp queries are not affected by what the CPU is doing,
    // so each stage's number here is stable to a few percent — which is what
    // it takes to see a 10% effect, let alone whether two of them add.
    if gpu
        .adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        println!("\ngpu time a frame, by stage (ms), on a settled plume:");
        print!("  {:>5} {:>5} {:>7}", "skip", "cache", "packed");
        for name in threers::fluid::STAGE_NAMES {
            print!("  {name:>10}");
        }
        println!("  {:>8}  {:>8}", "total", "vs none");

        let mut baseline = 0.0f32;
        for mask in 0..8u32 {
            let Some(mut sim) = build(
                &gpu,
                FluidConfig3d {
                    empty_space_skip: mask & 1 != 0,
                    cache_bindings: mask & 2 != 0,
                    compact_dye: mask & 4 != 0,
                    buoyancy: 46.0,
                    curl: 26.0,
                    density_dissipation: 0.2,
                    ..FluidConfig3d::default()
                },
            ) else {
                continue;
            };
            if !sim.profile() {
                break;
            }
            for frame in 0..52 {
                if frame < 34 && frame % 3 == 0 {
                    sim.splat([0.5, 0.11, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
                }
                sim.step(1.0 / 60.0);
            }
            // Each stage's own minimum over the run, not the stages of the
            // cheapest frame. The GPU clocks up and down and the readback
            // lands on whichever frame it lands on, so one frame's six
            // numbers are six independent samples of a noisy quantity — the
            // first version of this took them together and reported the
            // solve, which is identical work in all eight rows, varying by
            // two and a half times.
            let mut best = [f32::INFINITY; threers::fluid::STAGE_NAMES.len()];
            for _ in 0..frames.max(60) {
                sim.step(1.0 / 60.0);
                sim.render_to_view(
                    &camera,
                    &view,
                    wgpu::TextureFormat::Rgba8Unorm,
                    (width, height),
                );
                let st = sim.stage_ms();
                if st.iter().sum::<f32>() <= 0.0 {
                    continue;
                }
                for (slot, v) in best.iter_mut().zip(st) {
                    if v > 0.0 {
                        *slot = slot.min(v);
                    }
                }
            }
            for slot in best.iter_mut() {
                if !slot.is_finite() {
                    *slot = 0.0;
                }
            }
            let best_total: f32 = best.iter().sum();
            if mask == 0 {
                baseline = best_total;
            }
            let yn = |b: bool| if b { "on" } else { "–" };
            print!(
                "  {:>5} {:>5} {:>7}",
                yn(mask & 1 != 0),
                yn(mask & 2 != 0),
                yn(mask & 4 != 0)
            );
            for v in best {
                print!("  {v:>10.3}");
            }
            println!("  {best_total:>7.3}ms  {:>7.2}x", baseline / best_total);
        }
    }

    // Whether the optimisations stack.
    //
    // They cannot be read off one another's individual numbers, because they
    // do not act on the same bottleneck: stepping over empty bricks is GPU
    // work in the march, keeping the bind groups is CPU work in the step, and
    // packing the dye is both. Whichever of the two is behind the other is
    // hidden, and which that is depends on the machine.
    //
    // All eight are built up front and then timed *interleaved*, round after
    // round, keeping each one's best. Measuring them one after another
    // instead lets the machine drift between configurations — which it does,
    // by more than the effect being measured, and the drift then reads as a
    // result.
    println!("\ndo they stack? whole frames, settled plume, every combination:");
    {
        let mut sims: Vec<(u32, FluidSim3d)> = Vec::new();
        for mask in 0..8u32 {
            let Some(mut sim) = build(
                &gpu,
                FluidConfig3d {
                    empty_space_skip: mask & 1 != 0,
                    cache_bindings: mask & 2 != 0,
                    compact_dye: mask & 4 != 0,
                    buoyancy: 46.0,
                    curl: 26.0,
                    density_dissipation: 0.2,
                    ..FluidConfig3d::default()
                },
            ) else {
                continue;
            };
            for frame in 0..52 {
                if frame < 34 && frame % 3 == 0 {
                    sim.splat([0.5, 0.11, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
                }
                sim.step(1.0 / 60.0);
            }
            sims.push((mask, sim));
        }
        let mut best = [f64::INFINITY; 8];
        let mut worst = [0.0f64; 8];
        for round in 0..repeats + 1 {
            for (mask, sim) in sims.iter_mut() {
                let t = Instant::now();
                for _ in 0..frames {
                    sim.step(1.0 / 60.0);
                    sim.render_to_view(
                        &camera,
                        &view,
                        wgpu::TextureFormat::Rgba8Unorm,
                        (width, height),
                    );
                }
                drain();
                let ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
                // The first round is the warm-up: first touch of each
                // texture, and every pipeline's first use at this format.
                if round > 0 {
                    best[*mask as usize] = best[*mask as usize].min(ms);
                    worst[*mask as usize] = worst[*mask as usize].max(ms);
                }
            }
        }
        // The spread is the point of this column. These effects are tens of
        // percent, and on a machine with anything else running the round to
        // round spread is larger than that — in which case the ratios below
        // are noise wearing a number, and the thing to do is to go and
        // measure in a regime where the effect is big: see the browser demo's
        // Optimisations panel, where the CPU is the binding constraint and
        // the same switches move the frame rate by a third.
        println!(
            "  {:>5} {:>5} {:>7}   {:>8}  {:>8}  {:>7}",
            "skip", "cache", "packed", "frame", "vs none", "spread"
        );
        let baseline = best[0];
        for mask in 0..8usize {
            if !best[mask].is_finite() {
                continue;
            }
            let yn = |b: bool| if b { "on" } else { "–" };
            println!(
                "  {:>5} {:>5} {:>7}   {:>6.2}ms  {:>7.2}x  {:>6.0}%",
                yn(mask & 1 != 0),
                yn(mask & 2 != 0),
                yn(mask & 4 != 0),
                best[mask],
                baseline / best[mask],
                100.0 * (worst[mask] - best[mask]) / best[mask]
            );
        }
    }

    // What the march costs is set by how many samples each ray takes, and
    // that is the one knob here that is free to change at runtime.
    println!("\nmarch steps, at the default resolution:");
    let base = FluidConfig3d {
        compact_dye: env::<u32>("FLUID_COMPACT").unwrap_or(0) != 0,
        empty_space_skip: env::<u32>("FLUID_SKIP").unwrap_or(1) != 0,
        ..FluidConfig3d::default()
    };
    if let Some(mut sim) = build(&gpu, base) {
        sim.multiple_splats(6);
        for _ in 0..20 {
            sim.step(1.0 / 60.0);
        }
        for steps in [32u32, 64, 96, 128, 192, 256] {
            sim.set_config(FluidConfig3d {
                march_steps: steps,
                ..base
            });
            let ms = time(&mut || {
                sim.render_to_view(
                    &camera,
                    &view,
                    wgpu::TextureFormat::Rgba8Unorm,
                    (width, height),
                )
            });
            println!("  {steps:>3} steps  {ms:>7.2}ms");
        }
    }
}

fn build(gpu: &GpuContext, config: FluidConfig3d) -> Option<FluidSim3d> {
    match FluidSim3d::new(gpu.device.clone(), gpu.queue.clone(), config) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("{}³: {e}", config.resolution);
            None
        }
    }
}

fn size() -> (u32, u32) {
    std::env::var("FLUID_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1280, 720))
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse().ok()
}
