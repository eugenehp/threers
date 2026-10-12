//! The fluid simulation, driven by a script instead of a hand.
//!
//!     cargo run --release --features fluid --example fluid_render
//!
//! Writes `out/fluid_render.png` — four strokes drawn across the canvas and
//! then left to curl for a second — and `out/fluid_render_dye.png`, the same
//! frame with the shading, bloom and sunrays turned off, which is what the dye
//! field actually holds.
//!
//! Everything here is deterministic: the hues and the splat positions come from
//! [`FluidConfig::seed`] and the clock is a fixed `dt`, so the two images are
//! the same bytes on every run and on every machine with the same adapter. That
//! is what makes them usable as documentation.
//!
//! Knobs:
//!
//! * `FLUID_SIZE=WxH` — output size (default 1600x900, try `3840x2160`).
//! * `FLUID_QUALITY=<name>` — `low` … `extreme`; `high` is the default and is
//!   the original's settings. `ultra` quadruples the dye grid and doubles the
//!   solver's, which is the difference between ink that is smooth at 1600 px
//!   and ink that is still smooth at 4K.
//! * `FLUID_SOLVER=multigrid` — a V-cycle instead of the original's twenty
//!   Jacobi sweeps. Writes `out/fluid_render_multigrid.png` as well, so the
//!   two can be put side by side.
//! * `FLUID_CURL=<n>` — vorticity confinement, 0 to 50.
//! * `FLUID_SEED=<n>` — a different set of hues.

use threers::encode_png;
use threers::fluid::{quality, quality_names, FluidConfig, FluidSim, PressureSolver};
use threers::gpu::GpuContext;

/// Simulated seconds per frame. Fixed, not measured: a wall clock would make
/// the output depend on how fast the machine is.
const DT: f32 = 1.0 / 60.0;

/// One scripted drag: where it goes, and when.
struct Stroke {
    id: i32,
    /// Frame it presses down on, and how many frames it lasts.
    start: u32,
    frames: u32,
    /// Position in uv at a parameter from 0 to 1.
    path: fn(f32) -> (f32, f32),
}

/// Four strokes, chosen to put the fluid through everything it does: a long
/// sweep that leaves a wake, a fast diagonal that crosses it, a tight spiral
/// that winds vorticity up in one place, and a short jab that collides with the
/// first wake while it is still moving.
const STROKES: &[Stroke] = &[
    Stroke {
        id: 1,
        start: 0,
        frames: 28,
        path: |t| {
            (
                0.08 + 0.84 * t,
                0.62 - 0.30 * (t * std::f32::consts::TAU).sin(),
            )
        },
    },
    Stroke {
        id: 2,
        start: 14,
        frames: 20,
        path: |t| (0.88 - 0.72 * t, 0.12 + 0.70 * t * t),
    },
    Stroke {
        id: 3,
        start: 34,
        frames: 34,
        path: |t| {
            let a = t * 3.0 * std::f32::consts::TAU;
            let r = 0.04 + 0.16 * t;
            (0.30 + r * a.cos(), 0.70 + r * a.sin() * 0.72)
        },
    },
    Stroke {
        id: 4,
        start: 62,
        frames: 16,
        path: |t| (0.62 + 0.20 * t, 0.86 - 0.52 * t),
    },
];

/// Frames to keep running after the last stroke, so the vortices have time to
/// form. A fluid photographed while it is still being pushed looks like paint;
/// it reads as a fluid about a second later.
const SETTLE: u32 = 34;

fn main() {
    let (width, height) = size();
    let level = match std::env::var("FLUID_QUALITY") {
        Ok(name) => match quality(name.trim()) {
            Some(q) => q,
            None => {
                let names: Vec<_> = quality_names().collect();
                eprintln!("unknown quality '{name}' (have {})", names.join(", "));
                return;
            }
        },
        Err(_) => quality("high").expect("high"),
    };
    let multigrid = std::env::var("FLUID_SOLVER").as_deref() == Ok("multigrid");
    let config = FluidConfig {
        pressure_solver: if multigrid {
            PressureSolver::Multigrid
        } else {
            PressureSolver::Jacobi
        },
        curl: env("FLUID_CURL").unwrap_or(30.0),
        seed: env::<u64>("FLUID_SEED").unwrap_or(FluidConfig::default().seed),
        // A splat per frame from a scripted pointer is a smaller push than a
        // real drag, which moves several pixels between frames.
        splat_force: 9000.0,
        ..FluidConfig::default().with_quality(level)
    };

    // An ordinary threers device — the same one a scene would be rendered on,
    // which is what lets a frame hold both. See `threers::gpu`.
    let gpu = match GpuContext::headless_blocking() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    println!("adapter: {}", gpu.describe());
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());

    let mut sim = FluidSim::new(device, queue, width, height, config).expect("fluid");
    let ((dw, dh), (sw, sh)) = sim.grid_sizes();
    println!(
        "quality {:<8} dye {dw}x{dh}  sim {sw}x{sh}  sweeps {}  fields {} MB",
        level.name,
        config.pressure_iterations,
        sim.memory_bytes() / (1024 * 1024),
    );

    let last = STROKES
        .iter()
        .map(|s| s.start + s.frames)
        .max()
        .unwrap_or(0);
    for frame in 0..last + SETTLE {
        for stroke in STROKES {
            let Some(step) = frame.checked_sub(stroke.start) else {
                continue;
            };
            if step > stroke.frames {
                continue;
            }
            let t = step as f32 / stroke.frames as f32;
            let (u, v) = (stroke.path)(t);
            let (x, y) = (u * width as f32, v * height as f32);
            if step == 0 {
                sim.pointer_down(stroke.id, x, y);
            } else if step == stroke.frames {
                sim.pointer_up(stroke.id);
            } else {
                sim.pointer_move(stroke.id, x, y);
            }
        }
        sim.update(DT);
    }

    write(
        if multigrid {
            "out/fluid_render_multigrid.png"
        } else {
            "out/fluid_render.png"
        },
        &sim.capture_rgba(width, height),
        width,
        height,
    );

    // The same state with the display stages off. Bloom, sunrays and the
    // height-field shading are most of what the picture looks like, and seeing
    // the dye on its own is the only way to tell which is which.
    sim.set_config(FluidConfig {
        shading: false,
        bloom: false,
        sunrays: false,
        ..*sim.config()
    });
    write(
        if multigrid {
            "out/fluid_render_multigrid_dye.png"
        } else {
            "out/fluid_render_dye.png"
        },
        &sim.capture_rgba(width, height),
        width,
        height,
    );
}

fn size() -> (u32, u32) {
    std::env::var("FLUID_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1600, 900))
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
