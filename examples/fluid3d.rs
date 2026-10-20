//! A three-dimensional fluid: a rising plume of coloured smoke, ray-marched.
//!
//!     cargo run --release --features fluid --example fluid3d
//!
//! Writes `out/fluid3d.png`. Knobs:
//!
//! * `FLUID_SIZE=WxH` — output size (default 1000x1000).
//! * `FLUID_QUALITY=<level>` — `low`, `medium`, `high` (the default),
//!   `ultra`, `extreme`. Sets the three below at once; see
//!   [`threers::fluid::QUALITY_LEVELS_3D`] for what each one costs.
//! * `FLUID_RES=<n>` — solver voxels a side, overriding the level. The cube
//!   law bites: 128 is eight times the work of 64.
//! * `FLUID_DYE=<n>` — dye voxels a side.
//! * `FLUID_STEPS=<n>` — ray-march samples (default 160).
//! * `FLUID_FPS=1` — draw the frame rate over the picture.
//! * `FLUID_ACCUM=<n>` — average the march over `n` frames, holding the
//!   fluid and the camera still. The volume stops moving, so this is the
//!   picture a still camera converges to rather than a frame of animation —
//!   which is exactly what a capture wants, and what the biggest volumes
//!   need, since past about 256³ the dye is finer than an affordable number
//!   of samples can resolve.
//! * `FLUID_SEED=<n>` — a different set of hues.
//!
//! This is the 3D counterpart of `examples/fluid_render.rs`, and it is a
//! separate example on purpose: the two-dimensional fluid is unchanged and
//! still the one the module leads with. What differs is in
//! [`threers::fluid::FluidSim3d`] — the curl becomes a vector, buoyancy
//! becomes worth having because there is an up, and the volume has to be
//! rendered rather than simply displayed.
//!
//! Everything here is deterministic: the hues come from
//! [`FluidConfig3d::seed`] and the clock is a fixed `dt`.

use threers::fluid::{quality3d, quality3d_names, FluidConfig3d, FluidSim3d};
use threers::gpu::GpuContext;
use threers::{encode_png, PerspectiveCamera, Vector3};

/// Simulated seconds a frame. Fixed, not measured.
const DT: f32 = 1.0 / 60.0;
/// Frames of plume before the picture is taken.
const FRAMES: u32 = 52;
/// How long the source keeps emitting. After that it is left to rise.
const EMIT_UNTIL: u32 = 34;
/// Frames timed after the plume has settled, best taken.
const TIMED_FRAMES: u32 = 12;

fn main() {
    let (width, height) = size();
    let gpu = match GpuContext::headless_blocking() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    println!("adapter: {}", gpu.describe());

    // A named level first, then anything asked for by hand on top of it.
    let level = std::env::var("FLUID_QUALITY").ok();
    let base = match level.as_deref() {
        None => FluidConfig3d::default(),
        Some(name) => match quality3d(name) {
            Some(q) => FluidConfig3d::default().with_quality(q),
            None => {
                let names: Vec<_> = quality3d_names().collect();
                eprintln!("unknown quality '{name}' (have {})", names.join(", "));
                return;
            }
        },
    };
    let config = FluidConfig3d {
        resolution: env("FLUID_RES").unwrap_or(base.resolution),
        dye_resolution: env("FLUID_DYE").unwrap_or(base.dye_resolution),
        march_steps: env("FLUID_STEPS").unwrap_or(160),
        seed: env("FLUID_SEED").unwrap_or(FluidConfig3d::default().seed),
        // Enough confinement that the plume curls rather than mushrooming
        // smoothly — the same knob as in two dimensions and the same job.
        curl: 26.0,
        buoyancy: 46.0,
        density_dissipation: 0.2,
        emission: 1.3,
        absorption: 34.0,
        show_fps: env::<u32>("FLUID_FPS").unwrap_or(0) != 0,
        ..base
    };
    let mut sim = match FluidSim3d::new(gpu.device.clone(), gpu.queue.clone(), config) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let (solver, dye) = sim.grids();
    println!(
        "sim {solver}^3, dye {dye}^3, fields {} MB, {} march steps",
        sim.memory_bytes() / (1024 * 1024),
        config.march_steps
    );

    for frame in 0..FRAMES {
        // A source near the floor that wanders in a slow circle, so the plume
        // has something to braid rather than rising as one straight column.
        if frame < EMIT_UNTIL && frame % 3 == 0 {
            let t = frame as f32 / EMIT_UNTIL as f32;
            let a = t * 7.5;
            sim.splat(
                [0.5 + 0.06 * a.cos(), 0.11, 0.5 + 0.06 * a.sin()],
                [45.0 * a.sin(), 300.0, 45.0 * a.cos()],
                // Hot at the base, cooling as it climbs: the colour is the
                // dye's and the dye is what rises, so a ramp over emission
                // time reads as a temperature gradient up the plume.
                [1.15 - 0.8 * t, 0.26 + 0.34 * t, 0.2 + 0.9 * t],
            );
        }
        sim.step(DT);
    }

    let mut camera = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.05, 20.0);
    camera.position = Vector3::new(1.1, 0.18, 1.24);
    camera.look_at(Vector3::new(0.0, -0.02, 0.0));

    // What a frame of this costs, as a real-time loop would pay it: a step
    // and a march, with the queue drained so the clock is not reading the
    // submission rather than the work.
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fluid3d timing target"),
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
    let mut best = f64::INFINITY;
    for _ in 0..TIMED_FRAMES {
        let t = std::time::Instant::now();
        sim.step(DT);
        sim.render_to_view(
            &camera,
            &view,
            wgpu::TextureFormat::Rgba8Unorm,
            (width, height),
        );
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "frame {best:.1}ms — {:.0} fps at {width}x{height}, {} march steps",
        1000.0 / best,
        config.march_steps
    );

    // Hold everything still and let the march average itself out.
    let accumulate: u32 = env("FLUID_ACCUM").unwrap_or(1);
    if accumulate > 1 {
        sim.set_config(FluidConfig3d {
            temporal_samples: accumulate,
            ..config
        });
        for _ in 0..accumulate * 3 {
            sim.capture_rgba(&camera, width, height);
        }
        println!("averaged the march over {} frames", accumulate * 3);
    }
    let rgba = sim.capture_rgba(&camera, width, height);
    write("out/fluid3d.png", &rgba, width, height);
}

fn size() -> (u32, u32) {
    std::env::var("FLUID_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1000, 1000))
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn write(path: &str, rgba: &[u8], width: u32, height: u32) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, encode_png(width, height, rgba)).expect("write png");
    println!("wrote {path} ({width}x{height})");
}
