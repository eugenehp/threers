//! The fluid, on the Metal backend — no wgpu underneath it at all.
//!
//!     cargo run --release --features "metal,fluid" --example metal_fluid
//!
//! Writes `out/metal_fluid.png`, and with `--compare` also renders the same
//! simulation through wgpu and reports how far apart the two pictures are.
//!
//! Knobs:
//!
//! * `FLUID_SIZE=WxH` — output size (default 1280x720).
//! * `FLUID_SEED=<n>` — a different set of hues.
//! * `FLUID_FPS=1` — draw the frame-rate read-out and its sparkline over the
//!   picture. Ignored with `--compare`: each backend reports its own measured
//!   rate, so the two pictures would differ by clock jitter rather than by
//!   anything about the solver.
//! * `FLUID_FRAMES=<n>` — steps before the picture is taken. With
//!   `--compare`, a short run is the one that says whether the two backends
//!   agree: a fluid is chaotic, so after a few hundred frames even an exact
//!   match in the arithmetic would drift apart.
//!
//! # What this demonstrates
//!
//! `src/metal` talks to Metal directly through the Objective-C runtime: no
//! `metal-rs`, no `objc` crate, no build script, and no wgpu. That matters for
//! an app that already owns a `MTLDevice` and a `CAMetalLayer` and does not
//! want a second graphics abstraction inside it. The solver is the same solver
//! — `src/metal/fluid.metal` is a transcription of `src/fluid/shader.rs` and
//! the two share the uniform block — and `--compare` is what keeps it honest.

use threers::encode_png;
use threers::fluid::FluidConfig;
use threers::metal::{fluid_shader_source, MetalDevice, MetalFluid, MetalRenderTarget};

/// Simulated seconds a frame, fixed so the picture repeats.
const DT: f32 = 1.0 / 60.0;
/// What `capture_rgba` writes into, so the two paths produce the same bytes.
const FORMAT: usize = threers::metal::enums::pixel_format::RGBA8_UNORM;
const DEFAULT_FRAMES: u32 = 90;

fn main() {
    let (width, height) = size();
    let frames: u32 = env("FLUID_FRAMES").unwrap_or(DEFAULT_FRAMES);
    let compare = std::env::args().any(|a| a == "--compare");

    let device = match MetalDevice::with_shader_source(&fluid_shader_source()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            return;
        }
    };
    println!("metal: {}", device.name());

    // The read-out measures renders, so it needs a render a step rather
    // than one at the end — and with `--compare` there is nothing to learn
    // from two clocks disagreeing.
    let show_fps = env::<u32>("FLUID_FPS").unwrap_or(0) != 0 && !compare;
    let config = FluidConfig {
        seed: env("FLUID_SEED").unwrap_or(FluidConfig::default().seed),
        show_fps,
        // The Metal backend implements the solver and the display, not the
        // bloom and sunrays — see `threers::metal::fluid`. Switching them off
        // on the wgpu side too is what makes `--compare` a comparison of the
        // same thing.
        bloom: false,
        sunrays: false,
        // Hue cycling is a pointer-driven feature and this backend does not
        // have it; off, so `unsupported` has nothing to report and the
        // comparison is of what the two actually share.
        colorful: false,
        ..FluidConfig::default()
    };

    let missing = MetalFluid::unsupported(&config);
    if !missing.is_empty() {
        // Said out loud rather than silently ignored: a picture that looks
        // like a fluid is not evidence that it ran what was asked for.
        eprintln!("this backend ignores: {}", missing.join(", "));
    }

    let mut fluid = match MetalFluid::new(&device, width, height, config) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let ((dw, dh), (sw, sh)) = fluid.grid_sizes();
    println!("dye {dw}x{dh}, sim {sw}x{sh}");

    fluid.multiple_splats(&device, 18);
    // With the read-out on, a frame is drawn per step into a target this
    // example owns — a chart of one sample is not a chart, and the readback
    // `capture_rgba` does is not part of a frame, so timing it would report
    // the PNG and not the fluid.
    let screen = if show_fps {
        match MetalRenderTarget::new(&device, width, height, FORMAT, 1) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("no render target: {e}");
                return;
            }
        }
    } else {
        None
    };
    for _ in 0..frames {
        fluid.update(&device, DT);
        if let Some(t) = &screen {
            if let Err(e) = fluid.render(&device, t.color_texture(), FORMAT, (width, height)) {
                eprintln!("render failed: {e}");
                return;
            }
        }
    }
    if show_fps {
        println!(
            "read-out: {:.1} fps over the last {frames} frames",
            fluid.fps()
        );
    }
    // One last frame, so the picture written out is the one the read-out
    // just measured.
    let shot = match screen.as_ref() {
        Some(t) => fluid
            .render(&device, t.color_texture(), FORMAT, (width, height))
            .and_then(|()| t.read_rgba(&device)),
        None => fluid.capture_rgba(&device),
    };
    let rgba = match shot {
        Ok(p) => p,
        Err(e) => {
            eprintln!("readback failed: {e}");
            return;
        }
    };
    write("out/metal_fluid.png", &rgba, width, height);

    if compare {
        match wgpu_reference(width, height, config, frames) {
            Some(other) => {
                write("out/metal_fluid_wgpu.png", &other, width, height);
                report(&rgba, &other);
            }
            None => eprintln!("no wgpu adapter, so nothing to compare against"),
        }
    }
}

/// The same simulation through `threers::fluid`, for `--compare`.
fn wgpu_reference(width: u32, height: u32, config: FluidConfig, frames: u32) -> Option<Vec<u8>> {
    use threers::fluid::FluidSim;
    use threers::gpu::GpuContext;

    let gpu = GpuContext::headless_blocking().ok()?;
    let mut sim =
        FluidSim::new(gpu.device.clone(), gpu.queue.clone(), width, height, config).ok()?;
    sim.multiple_splats(18);
    for _ in 0..frames {
        sim.update(DT);
    }
    Some(sim.capture_rgba(width, height))
}

/// How far apart the two pictures are.
///
/// Not "are they identical": two backends never are. Half-float rounding
/// differs, the samplers round the last bit differently, and a fluid is
/// chaotic enough to turn either into a visible difference after ninety
/// frames. What is worth reporting is the *shape* of the disagreement — a mean
/// that is small with a large maximum is two pictures of the same fluid, and a
/// mean that is large is two different fluids.
fn report(a: &[u8], b: &[u8]) {
    let n = a.len().min(b.len());
    let mut sum = 0.0f64;
    let mut worst = 0u8;
    let mut over_16 = 0usize;
    for i in 0..n {
        let d = a[i].abs_diff(b[i]);
        sum += d as f64;
        worst = worst.max(d);
        if d > 16 {
            over_16 += 1;
        }
    }
    println!(
        "metal vs wgpu: mean |diff| {:.2}/255, worst {worst}, {:.2}% of channels over 16",
        sum / n as f64,
        100.0 * over_16 as f64 / n as f64
    );
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

fn write(path: &str, rgba: &[u8], width: u32, height: u32) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, encode_png(width, height, rgba)).expect("write png");
    println!("wrote {path} ({width}x{height})");
}
