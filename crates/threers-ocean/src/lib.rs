//! The threers ocean-water demo, as a browser package.
//!
//! ```text
//! crates/threers-ocean/build.sh
//! python3 -m http.server -d web        # then open /ocean.html
//! ```
//!
//! # What is shared, and what is not
//!
//! The simulation modules are **included from the native example** rather than
//! copied — `#[path]` points each `mod` at `examples/ocean_water/`. There is one
//! ocean, and a browser build that drifts from the desktop one is worse than no
//! browser build. Only the shell differs: the native example opens a winit window
//! and runs an event loop; this asks the canvas for a WebGPU surface and runs on
//! `requestAnimationFrame`.
//!
//! Everything the ocean does is already portable — the wave field, the cascades,
//! the foam and the particles are all compute passes on the renderer's own
//! device, and nothing reads back or touches the filesystem or a clock the web
//! does not have.

// Sharing the source has a cost: the browser shell drives a subset of what the
// native example does, so everything only the desktop shell reaches reads as
// dead here. Deleting any of it would break the example that does use it, and
// splitting the files would reintroduce exactly the drift the sharing prevents.
#![allow(dead_code)]

// The ocean itself, shared verbatim with `examples/ocean_water`.
#[path = "../../../examples/ocean_water/fft.rs"]
mod fft;
#[path = "../../../examples/ocean_water/grid.rs"]
mod grid;
#[path = "../../../examples/ocean_water/ocean_fft.rs"]
mod ocean_fft;
#[path = "../../../examples/ocean_water/particles.rs"]
mod particles;
#[path = "../../../examples/ocean_water/preset.rs"]
mod preset;
#[path = "../../../examples/ocean_water/probe.rs"]
mod probe;
#[path = "../../../examples/ocean_water/shader.rs"]
mod shader;
#[path = "../../../examples/ocean_water/spectrum.rs"]
mod spectrum;
#[path = "../../../examples/ocean_water/surface_state.rs"]
mod surface_state;
#[path = "../../../examples/ocean_water/terrain.rs"]
mod terrain;
#[path = "../../../examples/ocean_water/waterline.rs"]
mod waterline;
#[path = "../../../examples/ocean_water/waves_gpu.rs"]
mod waves_gpu;
#[path = "../../../examples/ocean_water/world.rs"]
mod world;

#[cfg(target_arch = "wasm32")]
mod web;

/// Preset and quality names, for a browser UI to populate a menu from without
/// hardcoding them.
pub fn preset_names() -> Vec<&'static str> {
    preset::all().iter().map(|p| p.name).collect()
}

pub fn quality_names() -> Vec<&'static str> {
    preset::QUALITY_LEVELS.iter().map(|q| q.name).collect()
}
