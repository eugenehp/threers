//! An interactive 2D fluid, on the GPU.
//!
//! Dye poured into a velocity field that is advected, confined, projected and
//! dissipated every frame. Drag a pointer across it and the ink follows the
//! motion, curls into vortices behind it, and diffuses away.
//!
//! ```no_run
//! use threers::fluid::{FluidConfig, FluidSim};
//! # fn demo(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>,
//! #         frame: &wgpu::Texture) -> Result<(), String> {
//! let mut fluid = FluidSim::new(device, queue, 1280, 720, FluidConfig::default())?;
//! fluid.multiple_splats(20);
//!
//! // Per frame:
//! fluid.update(1.0 / 60.0);
//! fluid.render(frame);
//! # Ok(()) }
//! ```
//!
//! # What it is
//!
//! A port of Pavel Dobryakov's [WebGL Fluid Simulation] onto wgpu, so the same
//! fluid runs natively and in a browser from one Rust implementation. The
//! numerics are Jos Stam's *Stable Fluids* on a staggered-free grid, the shape
//! of the frame is Mark Harris's [GPU Gems 38], and the look — the vorticity
//! confinement, the bloom, the sunrays, the dye lit as a height field — is
//! Dobryakov's.
//!
//! Per frame, over five textures:
//!
//! | pass | what it does |
//! |---|---|
//! | curl | the local spin of the velocity field |
//! | vorticity | pushes along concentrated spin, putting back what the grid ate |
//! | divergence | how much the field is creating or destroying fluid |
//! | pressure × 20 | Jacobi sweeps of ∇²p = ∇·u |
//! | gradient subtract | removes ∇p, leaving an incompressible field |
//! | advect velocity | the field carries itself |
//! | advect dye | the field carries the ink |
//!
//! then bloom, sunrays and a display composite on top. Every one is a
//! fullscreen fragment pass — there is no compute stage anywhere in it, which
//! is what lets it run on WebGL2 as well as WebGPU.
//!
//! # Why a fluid is cheap and a fluid solver is not
//!
//! The expensive part is the pressure solve: making the field divergence-free
//! is a global constraint, and twenty Jacobi sweeps is not a converged Poisson
//! solve. It does not need to be. What is wanted is a field whose remaining
//! divergence is smooth and small enough that the dye cannot show it, and
//! twenty sweeps at 128² reaches that in well under a millisecond. Each sweep
//! is also only a 4-tap kernel, so the whole solve is bandwidth, not
//! arithmetic — which is why the simulation grid is 128² while the dye it
//! carries is 1024². The eye reads the dye's resolution, not the solver's.
//!
//! # Texture space, and why the picture is a mirror
//!
//! In OpenGL a texture's origin is at the bottom left; in WebGPU it is at the
//! top left. The simulation lives entirely in uv, so rather than flipping
//! anything, the whole field runs in a frame where y points *down* the screen.
//! Reflection is a symmetry of the Navier-Stokes equations and — as the
//! derivation in `src/fluid/shader.rs` shows — of this vorticity-confinement
//! term too, so every formula ports verbatim and the result is the mirror image
//! of the same initial conditions. It also means pointer coordinates go in as
//! they arrive from the window system, with no `1 - y`.
//!
//! # Where it differs from the original
//!
//! * The nineteen GLSL programs are one WGSL module with nineteen entry points
//!   over one bind group — see `src/fluid/shader.rs`.
//! * `SHADING` / `BLOOM` / `SUNRAYS` were three `#define`s and up to eight
//!   compiled display variants. Here they are bits in the uniform block.
//! * The dither under the bloom is interleaved gradient noise computed in the
//!   shader rather than a sampled blue-noise texture, so the crate ships no
//!   asset and the simulation stays a function of its own state.
//! * Float formats are chosen per adapter ([`FluidFormats::for_adapter`]), and
//!   the one case the original handles that is not handled here is a device
//!   that can render to a half-float target but not filter one. WebGPU
//!   guarantees filtering; WebGL2 has it in core for half float.
//! * Rendering into an sRGB target applies the inverse transfer first, so the
//!   bytes that land match the ones GL wrote into a plain `RGBA8` canvas.
//!
//! [WebGL Fluid Simulation]: https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
//! [GPU Gems 38]: https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu

mod passes;
mod shader;
#[cfg(target_arch = "wasm32")]
mod web;

use std::sync::Arc;

use crate::renderer::fullscreen::{self, draw, DoubleFbo, Fbo, Load};
use passes::{
    batch_ubo, bind, ubo, Params, Pipelines, FLAG_BLOOM, FLAG_ENCODE, FLAG_SHADING, FLAG_SUNRAYS,
    MAX_BATCH,
};

pub use passes::FluidFormats;

/// Every knob the simulation has, with the original's defaults.
///
/// The three that change the picture most: `curl` decides whether it looks like
/// smoke or like spilled paint, `density_dissipation` how long a stroke stays
/// on screen, and `sim_resolution` how fine the motion is (as opposed to
/// `dye_resolution`, which is how fine the *ink* is).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidConfig {
    /// Shortest side of the velocity / pressure grid, in texels.
    pub sim_resolution: u32,
    /// Shortest side of the dye grid. Independent of the solver's, and usually
    /// several times larger.
    pub dye_resolution: u32,
    /// How fast dye fades, per second.
    pub density_dissipation: f32,
    /// How fast motion fades, per second.
    pub velocity_dissipation: f32,
    /// How much of last frame's pressure to start this frame's solve from.
    pub pressure: f32,
    /// Jacobi sweeps per frame. With [`PressureSolver::Multigrid`] this is the
    /// smoothing budget instead — see there.
    pub pressure_iterations: u32,
    /// How the velocity and the dye are carried along.
    pub advection: Advection,
    /// How the pressure equation is solved.
    pub pressure_solver: PressureSolver,
    /// V-cycles per frame, for [`PressureSolver::Multigrid`]. One is usually
    /// enough; a second buys about another factor of ten.
    pub multigrid_cycles: u32,
    /// Coefficient on the pressure gradient that the projection subtracts.
    ///
    /// **1.0, which is twice the analytically consistent value.**
    /// `(pR - pL, pT - pB)` is a central difference over *two* texels, so
    /// turning it into a gradient takes the 0.5 that GPU Gems 38 calls
    /// `halfrdx` and that the divergence pass carries. The original omits it
    /// there and only there.
    ///
    /// Measured, that is a compensation rather than a mistake. Jacobi converges
    /// slowly on a Poisson problem whose solution is long-ranged, and twenty
    /// sweeps is nowhere near converged — so the pressure is too small, and so
    /// is the correction it produces. Doubling the correction puts back more of
    /// what the unconverged solve left out than the consistent coefficient
    /// does: on one splat at the default twenty sweeps, 1.0 leaves 65 % of the
    /// divergence where 0.5 leaves 82 %.
    ///
    /// The trade appears as soon as the solve *is* converged. At 320 sweeps the
    /// order reverses — 0.5 leaves 37 %, and 1.0 leaves 127 %, which is more
    /// divergence than not projecting at all. So this belongs with
    /// [`pressure_iterations`](Self::pressure_iterations): 1.0 with twenty of
    /// them, 0.5 if you raise them. `the_missing_half_compensates_for_an_
    /// unconverged_solve` in the tests holds both halves of that.
    pub gradient_scale: f32,
    /// Vorticity confinement strength. 0 turns it off.
    pub curl: f32,
    /// Radius of a pointer splat, as a fraction of the canvas.
    pub splat_radius: f32,
    /// How hard a pointer's motion pushes the fluid.
    pub splat_force: f32,
    /// Light the dye as a height field.
    pub shading: bool,
    /// Cycle pointer colours through the hue circle.
    pub colorful: bool,
    /// Hue cycling speed, in changes per second.
    pub color_update_speed: f32,
    /// Flat colour behind the dye, when not `transparent`.
    pub back_color: [f32; 3],
    /// Keep the dye's own alpha instead of compositing it over `back_color`.
    pub transparent: bool,
    pub bloom: bool,
    /// Levels in the bloom chain. Each is half the size of the last.
    pub bloom_iterations: u32,
    pub bloom_resolution: u32,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    /// How gradually the threshold comes in, as a fraction of it.
    pub bloom_soft_knee: f32,
    pub sunrays: bool,
    pub sunrays_resolution: u32,
    pub sunrays_weight: f32,
    /// Seed for splat positions and hues. Fixed by default, so a headless
    /// render of N frames is the same picture every time.
    pub seed: u64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            sim_resolution: 128,
            dye_resolution: 1024,
            density_dissipation: 1.0,
            velocity_dissipation: 0.2,
            pressure: 0.8,
            pressure_iterations: 20,
            advection: Advection::SemiLagrangian,
            pressure_solver: PressureSolver::Jacobi,
            multigrid_cycles: 1,
            gradient_scale: 1.0,
            curl: 30.0,
            splat_radius: 0.25,
            splat_force: 6000.0,
            shading: true,
            colorful: true,
            color_update_speed: 10.0,
            back_color: [0.0, 0.0, 0.0],
            transparent: false,
            bloom: true,
            bloom_iterations: 8,
            bloom_resolution: 256,
            bloom_intensity: 0.8,
            bloom_threshold: 0.6,
            bloom_soft_knee: 0.7,
            sunrays: true,
            sunrays_resolution: 196,
            sunrays_weight: 1.0,
            seed: 0x5EED_F101_D1CE,
        }
    }
}

/// How a field is carried along by the velocity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Advection {
    /// Trace backwards, fetch, done — Stam's semi-Lagrangian step, and the
    /// original's. Unconditionally stable at any time step, which is the
    /// whole reason it is the standard choice.
    ///
    /// It is first order, and its error is *diffusion*: the backward trace
    /// lands between texels and the bilinear fetch averages them, so every
    /// step blurs a little. That blur is most of why dye goes muddy and why
    /// small vortices stop existing after a few hundred frames.
    #[default]
    SemiLagrangian,
    /// Advect forward, advect the result back, and take off half of however
    /// far it failed to return — MacCormack, second order in space for one
    /// extra pass.
    ///
    /// The correction is an extrapolation and can overshoot, so the result is
    /// clamped to the range of the four texels the forward pass actually read.
    /// That limiter is what makes it unconditionally stable (Selle et al.
    /// 2008) rather than merely sharper.
    ///
    /// Costs one extra full-resolution pass per advected field, which on the
    /// dye grid is the most expensive pass in the frame — see the numbers in
    /// `examples/fluid_bench.rs`.
    MacCormack,
}

/// How the pressure equation gets solved.
///
/// Both answer the same question — what pressure field makes the velocity
/// divergence-free — and the difference is entirely how fast they get there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PressureSolver {
    /// Twenty Jacobi sweeps, as the original. The default, because it is what
    /// produces the original's picture.
    ///
    /// A Jacobi sweep moves information one texel, so the error it is slowest
    /// to remove is the one spread across the whole grid — and the pressure
    /// field from a localised puff of velocity is exactly that: long-ranged,
    /// falling off like log r. Sixteen times the sweeps still leaves a third
    /// of the divergence, and the shortfall is not a floor but a rate.
    #[default]
    Jacobi,
    /// A multigrid V-cycle: smooth, take the residual to a grid half the size,
    /// solve *that*, interpolate the correction back, smooth again.
    ///
    /// The point is that "spread across the whole grid" stops being true when
    /// the grid is eight texels across, so the mode Jacobi cannot touch is the
    /// one the coarsest level solves outright. The cost is about one and a
    /// half times the default sweep count in passes — most of them at reduced
    /// resolution — and the measured result is in
    /// `multigrid_beats_jacobi_by_an_order_of_magnitude`.
    ///
    /// Not the default: a visibly more incompressible fluid is a *different*
    /// fluid, and the original's look is what this module is a port of.
    Multigrid,
}

impl FluidConfig {
    /// Apply a [`FluidQuality`], leaving every other setting alone.
    ///
    /// ```
    /// # use threers::fluid::{quality, FluidConfig};
    /// let config = FluidConfig::default().with_quality(quality("ultra").unwrap());
    /// assert_eq!(config.dye_resolution, 2048);
    /// ```
    pub fn with_quality(mut self, quality: &FluidQuality) -> Self {
        self.sim_resolution = quality.sim_resolution;
        self.dye_resolution = quality.dye_resolution;
        self.pressure_iterations = quality.pressure_iterations;
        self.bloom_iterations = quality.bloom_iterations;
        self.bloom_resolution = quality.bloom_resolution;
        self.sunrays_resolution = quality.sunrays_resolution;
        self
    }

    /// Everything the picture does not strictly need: no shading, no bloom, no
    /// sunrays, a coarse dye grid. What the original falls back to on a device
    /// that cannot filter a float texture.
    pub fn plain() -> Self {
        Self {
            dye_resolution: 512,
            shading: false,
            bloom: false,
            sunrays: false,
            ..Self::default()
        }
    }
}

/// A named quality level: the four resolutions and the two iteration counts
/// that decide what the simulation costs.
///
/// Everything else in [`FluidConfig`] is a look or a feel. These six are the
/// bill, and they are grouped so a menu can offer one choice instead of six.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidQuality {
    pub name: &'static str,
    /// Shortest side of the velocity and pressure grids.
    pub sim_resolution: u32,
    /// Shortest side of the dye grid.
    pub dye_resolution: u32,
    pub pressure_iterations: u32,
    pub bloom_iterations: u32,
    pub bloom_resolution: u32,
    pub sunrays_resolution: u32,
}

/// The quality levels, coarsest first.
///
/// `high` is the original's defaults, and the two above it are not in the
/// original at all — it was written for 2017 mobile browsers. What they buy is
/// different at each end: **dye** resolution is detail in the ink, which is
/// what the eye reads; **sim** resolution is detail in the *motion*, which is
/// what decides how fine the vortices can get before the grid eats them; and
/// **sweeps** decide how incompressible the result is.
///
/// What they cost is mostly dye. The dye field is two `rgba16float` textures at
/// the canvas's aspect, so a 16:9 canvas at `ultra` carries 2048 × 3641 twice —
/// about 240 MB — against 15 MB at `high`. The solver grids are a few hundred
/// kilobytes at any of these.
///
/// `extreme` asks for grids large enough that a wide canvas would exceed what
/// the device will allocate as a texture at all; the sizes are clamped to
/// `max_texture_dimension_2d` on both axes together, so it degrades into
/// `ultra` rather than failing.
pub const QUALITY_LEVELS: &[FluidQuality] = &[
    FluidQuality {
        name: "low",
        sim_resolution: 64,
        dye_resolution: 256,
        pressure_iterations: 12,
        bloom_iterations: 6,
        bloom_resolution: 128,
        sunrays_resolution: 128,
    },
    FluidQuality {
        name: "medium",
        sim_resolution: 96,
        dye_resolution: 512,
        pressure_iterations: 16,
        bloom_iterations: 7,
        bloom_resolution: 192,
        sunrays_resolution: 160,
    },
    // The original's defaults, and what `FluidConfig::default` gives.
    FluidQuality {
        name: "high",
        sim_resolution: 128,
        dye_resolution: 1024,
        pressure_iterations: 20,
        bloom_iterations: 8,
        bloom_resolution: 256,
        sunrays_resolution: 196,
    },
    FluidQuality {
        name: "ultra",
        sim_resolution: 256,
        dye_resolution: 2048,
        pressure_iterations: 32,
        bloom_iterations: 9,
        bloom_resolution: 512,
        sunrays_resolution: 384,
    },
    FluidQuality {
        name: "extreme",
        sim_resolution: 512,
        dye_resolution: 4096,
        pressure_iterations: 48,
        bloom_iterations: 10,
        bloom_resolution: 1024,
        sunrays_resolution: 768,
    },
];

/// Look a quality level up by name, for a menu or a command line.
pub fn quality(name: &str) -> Option<&'static FluidQuality> {
    QUALITY_LEVELS.iter().find(|q| q.name == name)
}

/// Every quality level's name, in order.
pub fn quality_names() -> impl Iterator<Item = &'static str> {
    QUALITY_LEVELS.iter().map(|q| q.name)
}

/// One finger or one mouse, tracked between frames so its motion can push the
/// fluid. Create them through [`FluidSim::pointer_down`].
#[derive(Clone, Copy, Debug)]
struct Pointer {
    id: i32,
    /// Current and previous position, in uv.
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    down: bool,
    moved: bool,
    color: [f32; 3],
}

/// xorshift64*, so splat positions and hues are reproducible from the seed.
/// Nothing here needs a good generator; it needs the same numbers twice.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        // Top 24 bits: every f32 in [0, 1) with a 2^-24 step and no bias.
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0
    }
}

/// A fully saturated colour at the given hue, as the original's `HSVtoRGB`.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// [`grid_size`], then shrunk to fit what the device will allocate.
///
/// Both axes are scaled by the same factor, because the point of `grid_size`
/// is that the texels come out square — clipping the long side alone would
/// stretch every vortex. The effect is that asking for more than the device
/// has gets you the most it does have, at the right shape, rather than a
/// texture-creation failure.
fn grid_size_limited(resolution: u32, width: u32, height: u32, max: u32) -> (u32, u32) {
    let (w, h) = grid_size(resolution, width, height);
    let longest = w.max(h);
    if longest <= max.max(1) {
        return (w, h);
    }
    let scale = max as f64 / longest as f64;
    (
        ((w as f64 * scale).floor() as u32).max(1),
        ((h as f64 * scale).floor() as u32).max(1),
    )
}

/// The shortest side of a grid is `resolution`; the longer one follows the
/// canvas's aspect, so the simulation's texels stay square. A non-square texel
/// would make a vortex elliptical.
fn grid_size(resolution: u32, width: u32, height: u32) -> (u32, u32) {
    let (width, height) = (width.max(1), height.max(1));
    let aspect = width as f32 / height as f32;
    let aspect = if aspect < 1.0 { 1.0 / aspect } else { aspect };
    let min = resolution.max(1);
    let max = ((resolution as f32 * aspect).round() as u32).max(1);
    if width > height {
        (max, min)
    } else {
        (min, max)
    }
}

/// One coarse level of the multigrid pyramid.
///
/// Level 0 is not here: it is the simulation's own `pressure` and `divergence`
/// textures, which already exist and which the cycle smooths in place.
struct MgLevel {
    /// The correction being solved for at this level, ping-ponged by the
    /// smoother.
    pressure: DoubleFbo,
    /// The restricted residual from the level above.
    rhs: Fbo,
    /// This level's own residual, on the way down.
    residual: Fbo,
    /// Uniform blocks: smoothing, residual, restriction, prolongation.
    /// Per level because the chain is one submit and `write_buffer` only
    /// orders against submits.
    ubo_smooth: wgpu::Buffer,
    ubo_residual: wgpu::Buffer,
    ubo_restrict: wgpu::Buffer,
    ubo_prolong: wgpu::Buffer,
}

/// Coarse levels for a grid of `width` x `height`, halving until a level would
/// be smaller than this.
///
/// Four texels, not two: the coarsest grid is solved by smoothing it hard, and
/// below about four across there is nothing left for a five-point stencil to
/// average that clamp-to-edge is not already giving it.
const MG_COARSEST: u32 = 4;

/// Relaxation weight for the multigrid smoother: `p <- (1-w)p + w*jacobi(p)`.
///
/// **One, and the textbook answer is 4/5.** The reason to damp a multigrid
/// smoother is the checkerboard mode: plain Jacobi's amplification factor is
/// `(cos kx + cos ky) / 2`, which at that mode is exactly -1, so the error
/// changes sign every sweep and never shrinks — and a V-cycle depends on its
/// smoother removing precisely the frequencies the coarse grid cannot
/// represent. Weighting to 4/5 takes that factor to 0.6 a sweep.
///
/// It does not pay here, because this right-hand side has no checkerboard
/// content to remove: the divergence is a *central* difference over two
/// texels, and that operator has a zero at the Nyquist frequency. Damping then
/// only slows the smoother on the modes that do exist. Measured on twelve
/// splats, residual as a fraction of what went in:
///
/// | w | 1 cycle | 2 cycles | 3 cycles |
/// |---|---|---|---|
/// | 0.6 | 0.267 | 0.100 | 0.065 |
/// | 0.8 | 0.208 | 0.076 | 0.060 |
/// | **1.0** | **0.175** | **0.068** | **0.058** |
///
/// It stays a parameter of the sweep rather than being folded away, because
/// the argument is a property of the divergence operator — change that and 4/5
/// is right again.
const MG_WEIGHT: f32 = 1.0;

fn mg_levels(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Vec<MgLevel> {
    let mut levels = Vec::new();
    let (mut w, mut h) = (width, height);
    while w / 2 >= MG_COARSEST && h / 2 >= MG_COARSEST && levels.len() < 8 {
        w /= 2;
        h /= 2;
        levels.push(MgLevel {
            pressure: DoubleFbo::new(device, "fluid mg pressure", w, h, format),
            rhs: Fbo::new(device, "fluid mg rhs", w, h, format),
            residual: Fbo::new(device, "fluid mg residual", w, h, format),
            ubo_smooth: ubo(device, "fluid mg smooth"),
            ubo_residual: ubo(device, "fluid mg residual"),
            ubo_restrict: ubo(device, "fluid mg restrict"),
            ubo_prolong: ubo(device, "fluid mg prolong"),
        });
    }
    levels
}

/// The uniform buffers, one per pass that runs in a single submit.
struct Ubos {
    copy: wgpu::Buffer,
    curl: wgpu::Buffer,
    vorticity: wgpu::Buffer,
    divergence: wgpu::Buffer,
    clear: wgpu::Buffer,
    pressure: wgpu::Buffer,
    gradient: wgpu::Buffer,
    advect_velocity: wgpu::Buffer,
    advect_dye: wgpu::Buffer,
    correct_velocity: wgpu::Buffer,
    correct_dye: wgpu::Buffer,
    splat_velocity: wgpu::Buffer,
    splat_dye: wgpu::Buffer,
    /// The batch itself — one per field, since both passes are in one submit.
    batch_velocity: wgpu::Buffer,
    batch_dye: wgpu::Buffer,
    /// Zeroes, bound by every pass that is not a splat. One buffer, shared.
    batch_idle: wgpu::Buffer,
    bloom_prefilter: wgpu::Buffer,
    bloom_final: wgpu::Buffer,
    /// One per blur in the chain, down and back up: each level has its own
    /// texel size.
    bloom_chain: Vec<wgpu::Buffer>,
    sunrays_mask: wgpu::Buffer,
    sunrays: wgpu::Buffer,
    blur_h: wgpu::Buffer,
    blur_v: wgpu::Buffer,
    display: wgpu::Buffer,
    background: wgpu::Buffer,
}

impl Ubos {
    fn new(device: &wgpu::Device, chain: usize) -> Self {
        Self {
            copy: ubo(device, "fluid copy"),
            curl: ubo(device, "fluid curl"),
            vorticity: ubo(device, "fluid vorticity"),
            divergence: ubo(device, "fluid divergence"),
            clear: ubo(device, "fluid clear"),
            pressure: ubo(device, "fluid pressure"),
            gradient: ubo(device, "fluid gradient"),
            advect_velocity: ubo(device, "fluid advect velocity"),
            advect_dye: ubo(device, "fluid advect dye"),
            correct_velocity: ubo(device, "fluid correct velocity"),
            correct_dye: ubo(device, "fluid correct dye"),
            splat_velocity: ubo(device, "fluid splat velocity"),
            splat_dye: ubo(device, "fluid splat dye"),
            batch_velocity: batch_ubo(device, "fluid splat batch velocity"),
            batch_dye: batch_ubo(device, "fluid splat batch dye"),
            batch_idle: batch_ubo(device, "fluid splat batch idle"),
            bloom_prefilter: ubo(device, "fluid bloom prefilter"),
            bloom_final: ubo(device, "fluid bloom final"),
            bloom_chain: (0..chain.max(1) * 2)
                .map(|_| ubo(device, "fluid bloom chain"))
                .collect(),
            sunrays_mask: ubo(device, "fluid sunrays mask"),
            sunrays: ubo(device, "fluid sunrays"),
            blur_h: ubo(device, "fluid blur h"),
            blur_v: ubo(device, "fluid blur v"),
            display: ubo(device, "fluid display"),
            background: ubo(device, "fluid background"),
        }
    }
}

/// A fluid simulation and the textures it lives in.
///
/// Owns its own wgpu resources and touches nothing else — it has no `Scene`, no
/// camera and no renderer. Hand it a device and a texture to draw into and it
/// will do both halves: advance the fluid, and composite it.
pub struct FluidSim {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: FluidConfig,
    formats: FluidFormats,
    /// Size of what is being rendered into, which sets both grids' aspect.
    width: u32,
    height: u32,
    /// `max_texture_dimension_2d`, so a quality level can ask for more than the
    /// device has and get the most it does have instead of a failed texture.
    max_dim: u32,

    dye: DoubleFbo,
    velocity: DoubleFbo,
    pressure: DoubleFbo,
    divergence: Fbo,
    curl: Fbo,
    /// The top level's residual, at the simulation's own resolution — the one
    /// size the pyramid below does not have.
    residual: Fbo,
    /// Somewhere for the MacCormack correction to land: it reads the forward
    /// result and the original at once, so it cannot write over either.
    /// Allocated whatever the advection, because they are small beside the
    /// dye and switching scheme mid-run should not reallocate.
    scratch_velocity: Fbo,
    scratch_dye: Fbo,
    /// Coarse levels for the multigrid solver. Allocated whatever the solver,
    /// because they are a few hundred kilobytes and switching solver mid-run
    /// should not reallocate.
    mg: Vec<MgLevel>,
    bloom: Fbo,
    bloom_chain: Vec<Fbo>,
    sunrays: Fbo,
    sunrays_temp: Fbo,

    pipes: Pipelines,
    sampler: wgpu::Sampler,
    ubos: Ubos,

    pointers: Vec<Pointer>,
    rng: Rng,
    color_timer: f32,
    paused: bool,
}

/// A queued splat, in the order it was asked for.
#[derive(Clone, Copy)]
struct Splat {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    color: [f32; 3],
}

impl FluidSim {
    /// Build a simulation sized for a `width` × `height` canvas, using the
    /// float formats WebGPU guarantees. Use [`FluidSim::with_formats`] and
    /// [`FluidFormats::for_adapter`] where the device may be a WebGL2 one.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        config: FluidConfig,
    ) -> Result<Self, String> {
        Self::with_formats(
            device,
            queue,
            width,
            height,
            config,
            FluidFormats::default(),
        )
    }

    /// As [`FluidSim::new`], with the storage formats chosen by the caller.
    pub fn with_formats(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        config: FluidConfig,
        formats: FluidFormats,
    ) -> Result<Self, String> {
        if config.pressure_iterations == 0 {
            return Err("fluid: pressure_iterations must be at least 1".into());
        }
        let (width, height) = (width.max(1), height.max(1));
        let pipes = Pipelines::new(&device, formats);
        let sampler = fullscreen::linear_sampler(&device);
        let ubos = Ubos::new(&device, config.bloom_iterations as usize);

        let max_dim = device.limits().max_texture_dimension_2d;
        let (sw, sh) = grid_size_limited(config.sim_resolution, width, height, max_dim);
        let (dw, dh) = grid_size_limited(config.dye_resolution, width, height, max_dim);
        let (bw, bh) = grid_size_limited(config.bloom_resolution, width, height, max_dim);
        let (rw, rh) = grid_size_limited(config.sunrays_resolution, width, height, max_dim);

        let sim = Self {
            dye: DoubleFbo::new(&device, "fluid dye", dw, dh, formats.dye),
            velocity: DoubleFbo::new(&device, "fluid velocity", sw, sh, formats.velocity),
            pressure: DoubleFbo::new(&device, "fluid pressure", sw, sh, formats.scalar),
            divergence: Fbo::new(&device, "fluid divergence", sw, sh, formats.scalar),
            curl: Fbo::new(&device, "fluid curl", sw, sh, formats.scalar),
            residual: Fbo::new(&device, "fluid residual", sw, sh, formats.scalar),
            scratch_velocity: Fbo::new(&device, "fluid advect scratch", sw, sh, formats.velocity),
            scratch_dye: Fbo::new(&device, "fluid dye scratch", dw, dh, formats.dye),
            mg: mg_levels(&device, sw, sh, formats.scalar),
            bloom: Fbo::new(&device, "fluid bloom", bw, bh, formats.dye),
            bloom_chain: bloom_chain(&device, bw, bh, config.bloom_iterations, formats.dye),
            sunrays: Fbo::new(&device, "fluid sunrays", rw, rh, formats.scalar),
            sunrays_temp: Fbo::new(&device, "fluid sunrays temp", rw, rh, formats.scalar),
            device,
            queue,
            config,
            formats,
            width,
            height,
            max_dim,
            pipes,
            sampler,
            ubos,
            pointers: Vec::new(),
            rng: Rng::new(config.seed),
            color_timer: 0.0,
            paused: false,
        };
        // The dye and velocity fields start at zero, which is a fluid at rest
        // with nothing in it — correct, and invisible. The caller decides
        // whether to seed it.
        Ok(sim)
    }

    pub fn config(&self) -> &FluidConfig {
        &self.config
    }

    /// Replace the configuration, reallocating any grid whose resolution
    /// changed. The fields themselves are preserved where the size is the same,
    /// so turning bloom on mid-stroke does not reset the fluid.
    pub fn set_config(&mut self, config: FluidConfig) {
        let resized = config.sim_resolution != self.config.sim_resolution
            || config.dye_resolution != self.config.dye_resolution
            || config.bloom_resolution != self.config.bloom_resolution
            || config.bloom_iterations != self.config.bloom_iterations
            || config.sunrays_resolution != self.config.sunrays_resolution;
        if config.bloom_iterations != self.config.bloom_iterations {
            self.ubos.bloom_chain = (0..config.bloom_iterations.max(1) as usize * 2)
                .map(|_| ubo(&self.device, "fluid bloom chain"))
                .collect();
        }
        self.config = config;
        if resized {
            self.reallocate();
        }
    }

    /// Size of the canvas being rendered into.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn formats(&self) -> FluidFormats {
        self.formats
    }

    /// What the grids actually came out as, in texels: the dye field and the
    /// solver field. A quality level is a request — these are the answer, after
    /// the canvas's aspect and the device's texture limit have had their say.
    pub fn grid_sizes(&self) -> ((u32, u32), (u32, u32)) {
        (
            (self.dye.width(), self.dye.height()),
            (self.velocity.width(), self.velocity.height()),
        )
    }

    /// Bytes of GPU memory the fields occupy. Dominated by the dye, which is
    /// two `rgba16float` textures at the canvas's aspect — at `ultra` on a 16:9
    /// canvas that alone is about 240 MB.
    pub fn memory_bytes(&self) -> u64 {
        let bytes = |fbo: &Fbo| {
            // `readback::bytes_per_texel`, not `target_pixel_byte_cost` — the
            // latter is the padded cost of a *render attachment* and reports 8
            // for a four-byte format, which would overstate this by double.
            let texel = crate::renderer::readback::bytes_per_texel(fbo.texture.format()) as u64;
            fbo.width as u64 * fbo.height as u64 * texel
        };
        let double = |f: &DoubleFbo| bytes(&f.read) + bytes(&f.write);
        double(&self.dye)
            + double(&self.velocity)
            + double(&self.pressure)
            + bytes(&self.divergence)
            + bytes(&self.curl)
            + bytes(&self.bloom)
            + self.bloom_chain.iter().map(bytes).sum::<u64>()
            + bytes(&self.sunrays)
            + bytes(&self.sunrays_temp)
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// The dye field, which is what the picture is of. Exposed so it can be
    /// sampled by something else — a material, a post pass, a readback.
    pub fn dye_texture(&self) -> &wgpu::Texture {
        &self.dye.read.texture
    }

    /// The velocity field. Two channels, in sim texels per second.
    pub fn velocity_texture(&self) -> &wgpu::Texture {
        &self.velocity.read.texture
    }

    /// Tell the simulation the canvas changed size. The grids are rebuilt at
    /// the new aspect; dye and velocity are resampled into them rather than
    /// cleared, so a resize does not wipe the picture.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.reallocate();
    }

    /// Rebuild every grid for the current size and configuration, carrying dye
    /// and velocity across.
    fn reallocate(&mut self) {
        let (w, h) = (self.width, self.height);
        let c = &self.config;
        let max_dim = self.max_dim;
        let (sw, sh) = grid_size_limited(c.sim_resolution, w, h, max_dim);
        let (dw, dh) = grid_size_limited(c.dye_resolution, w, h, max_dim);
        let (bw, bh) = grid_size_limited(c.bloom_resolution, w, h, max_dim);
        let (rw, rh) = grid_size_limited(c.sunrays_resolution, w, h, max_dim);
        let (dye_fmt, vel_fmt, scalar) =
            (self.formats.dye, self.formats.velocity, self.formats.scalar);
        let iterations = c.bloom_iterations;

        self.resample(dw, dh, dye_fmt, true);
        self.resample(sw, sh, vel_fmt, false);

        // Pressure, divergence and curl are rebuilt from scratch: they are
        // derived from the velocity field every frame, so there is nothing in
        // them worth resampling.
        self.pressure = DoubleFbo::new(&self.device, "fluid pressure", sw, sh, scalar);
        self.divergence = Fbo::new(&self.device, "fluid divergence", sw, sh, scalar);
        self.curl = Fbo::new(&self.device, "fluid curl", sw, sh, scalar);
        self.residual = Fbo::new(&self.device, "fluid residual", sw, sh, scalar);
        self.scratch_velocity = Fbo::new(&self.device, "fluid advect scratch", sw, sh, vel_fmt);
        self.scratch_dye = Fbo::new(&self.device, "fluid dye scratch", dw, dh, dye_fmt);
        self.mg = mg_levels(&self.device, sw, sh, scalar);
        self.bloom = Fbo::new(&self.device, "fluid bloom", bw, bh, dye_fmt);
        self.bloom_chain = bloom_chain(&self.device, bw, bh, iterations, dye_fmt);
        self.sunrays = Fbo::new(&self.device, "fluid sunrays", rw, rh, scalar);
        self.sunrays_temp = Fbo::new(&self.device, "fluid sunrays temp", rw, rh, scalar);
    }

    /// Copy one double-buffered field into a new size, through the copy pass.
    fn resample(&mut self, w: u32, h: u32, format: wgpu::TextureFormat, dye: bool) {
        let field = if dye { &self.dye } else { &self.velocity };
        if field.width() == w && field.height() == h {
            return;
        }
        let label = if dye { "fluid dye" } else { "fluid velocity" };
        let fresh = Fbo::new(&self.device, label, w, h, format);
        write(&self.queue, &self.ubos.copy, Params::default());
        let bg = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            &self.ubos.copy,
            &self.ubos.batch_idle,
            &field.read.view,
            None,
            None,
        );
        let pipeline = if dye {
            &self.pipes.copy_dye
        } else {
            &self.pipes.copy_velocity
        };
        let mut enc = self.encoder("fluid resize");
        draw(&mut enc, &fresh.view, Load::Clear, pipeline, &bg);
        self.queue.submit([enc.finish()]);

        let replacement = DoubleFbo {
            read: fresh,
            write: Fbo::new(&self.device, label, w, h, format),
        };
        if dye {
            self.dye = replacement;
        } else {
            self.velocity = replacement;
        }
    }

    // -- input ------------------------------------------------------------

    /// Press at a pixel position in the canvas, with an id that identifies the
    /// mouse or the touch. `-1` is the convention for a mouse.
    pub fn pointer_down(&mut self, id: i32, x: f32, y: f32) {
        let color = self.generate_color();
        let (u, v) = (x / self.width as f32, y / self.height as f32);
        let p = Pointer {
            id,
            x: u,
            y: v,
            dx: 0.0,
            dy: 0.0,
            down: true,
            moved: false,
            color,
        };
        match self.pointers.iter_mut().find(|p| p.id == id || !p.down) {
            Some(slot) => *slot = p,
            None => self.pointers.push(p),
        }
    }

    /// Move a pressed pointer. Ignored for a pointer that is not down, which is
    /// what keeps a hovering mouse from painting.
    pub fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
        let (w, h) = (self.width as f32, self.height as f32);
        let aspect = w / h;
        let Some(p) = self.pointers.iter_mut().find(|p| p.id == id && p.down) else {
            return;
        };
        let (px, py) = (p.x, p.y);
        p.x = x / w;
        p.y = y / h;
        // The delta drives the splat's force, and is corrected for aspect so a
        // diagonal drag pushes diagonally on a non-square canvas.
        p.dx = (p.x - px) * if aspect < 1.0 { aspect } else { 1.0 };
        p.dy = (p.y - py) / if aspect > 1.0 { aspect } else { 1.0 };
        p.moved = p.dx != 0.0 || p.dy != 0.0;
    }

    /// Release a pointer. Its slot is reused by the next press.
    pub fn pointer_up(&mut self, id: i32) {
        if let Some(p) = self.pointers.iter_mut().find(|p| p.id == id) {
            p.down = false;
        }
    }

    /// `count` splats at random positions, with random hues and enough velocity
    /// to throw the dye across the canvas. What the original opens with, and
    /// what the space bar does.
    pub fn multiple_splats(&mut self, count: u32) {
        let mut batch = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut color = self.generate_color();
            // Ten times a pointer's dye: a burst is meant to read as a flash,
            // and at a pointer's intensity it would not.
            for c in &mut color {
                *c *= 10.0;
            }
            let x = self.rng.next_f32();
            let y = self.rng.next_f32();
            let dx = 1000.0 * (self.rng.next_f32() - 0.5);
            let dy = 1000.0 * (self.rng.next_f32() - 0.5);
            batch.push(Splat {
                x,
                y,
                dx,
                dy,
                color,
            });
        }
        self.record_splats(&batch);
    }

    /// One splat: `x`, `y` in uv with the origin at the top left, `dx`, `dy`
    /// the velocity to add in sim texels per second, `color` the dye.
    ///
    /// Applied at once rather than queued for the next
    /// [`update`](Self::update), so a headless script can splat and step in
    /// either order and get the same picture.
    pub fn splat(&mut self, x: f32, y: f32, dx: f32, dy: f32, color: [f32; 3]) {
        self.record_splats(&[Splat {
            x,
            y,
            dx,
            dy,
            color,
        }]);
    }

    /// Set the whole velocity field to one vector, in sim texels per second.
    ///
    /// A uniform flow, which is both the simplest thing to impose — a wind, a
    /// current — and the one case where advection has an exact answer: a
    /// constant field advects itself unchanged, its divergence is zero so the
    /// projection does nothing, and its curl is zero so confinement does
    /// nothing. Anything that happens to the *dye* under it is therefore the
    /// advection scheme's own error and nothing else, which is what
    /// `maccormack_advection_diffuses_less` measures.
    pub fn set_uniform_velocity(&mut self, x: f32, y: f32) {
        write(
            &self.queue,
            &self.ubos.clear,
            Params {
                color: [x, y, 0.0, 1.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.clear, &self.velocity.read, None, None);
        let mut enc = self.encoder("fluid fill velocity");
        draw(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.fill_velocity,
            &bg,
        );
        self.queue.submit([enc.finish()]);
        self.velocity.swap();
    }

    fn generate_color(&mut self) -> [f32; 3] {
        let c = hsv_to_rgb(self.rng.next_f32(), 1.0, 1.0);
        [c[0] * 0.15, c[1] * 0.15, c[2] * 0.15]
    }

    // -- the frame --------------------------------------------------------

    /// Advance by `dt` seconds: apply queued input, cycle the hues, and step
    /// the solver unless paused.
    ///
    /// Input is applied even while paused, so a stroke drawn on a paused fluid
    /// still appears.
    pub fn update(&mut self, dt: f32) {
        self.update_colors(dt);
        self.apply_inputs();
        if !self.paused {
            self.step(dt);
        }
    }

    fn update_colors(&mut self, dt: f32) {
        if !self.config.colorful {
            return;
        }
        self.color_timer += dt * self.config.color_update_speed;
        if self.color_timer >= 1.0 {
            self.color_timer = self.color_timer.fract();
            for i in 0..self.pointers.len() {
                let color = self.generate_color();
                self.pointers[i].color = color;
            }
        }
    }

    /// Splat for every pointer that moved since the last frame, and forget the
    /// motion so it is not applied twice.
    fn apply_inputs(&mut self) {
        let moved: Vec<Splat> = self
            .pointers
            .iter_mut()
            .filter(|p| p.moved)
            .map(|p| {
                p.moved = false;
                Splat {
                    x: p.x,
                    y: p.y,
                    dx: p.dx,
                    dy: p.dy,
                    color: p.color,
                }
            })
            .collect();
        let force = self.config.splat_force;
        // One submit for every finger, rather than one each.
        let scaled: Vec<Splat> = moved
            .into_iter()
            .map(|s| Splat {
                dx: s.dx * force,
                dy: s.dy * force,
                ..s
            })
            .collect();
        self.record_splats(&scaled);
    }

    /// Every queued splat, in two passes and one submit.
    ///
    /// Two passes *total*, not two per splat: the shader loops over the batch
    /// and sums the Gaussians itself. A pass each is what this replaced, and
    /// at a 4096-texel dye grid that was two full-screen passes per blob — a
    /// burst of twenty-five cost eighteen milliseconds, about four frames.
    ///
    /// Batches longer than [`MAX_BATCH`] are split, which costs a submit each;
    /// thirty-two is past any plausible number of fingers or of blobs in a
    /// burst.
    fn record_splats(&mut self, splats: &[Splat]) {
        for chunk in splats.chunks(MAX_BATCH) {
            self.record_splat_batch(chunk);
        }
    }

    fn record_splat_batch(&mut self, splats: &[Splat]) {
        if splats.is_empty() {
            return;
        }
        let aspect = self.width as f32 / self.height as f32;
        // The radius is in canvas fractions, and widening it with the aspect
        // keeps a splat the same physical size on a wide canvas.
        let mut radius = self.config.splat_radius / 100.0;
        if aspect > 1.0 {
            radius *= aspect;
        }

        // Two `vec4`s per splat: (point, radius, _) and (colour, _).
        let mut velocity_batch = [0.0f32; MAX_BATCH * 8];
        let mut dye_batch = [0.0f32; MAX_BATCH * 8];
        for (k, s) in splats.iter().enumerate() {
            let at = k * 8;
            for batch in [&mut velocity_batch, &mut dye_batch] {
                batch[at] = s.x;
                batch[at + 1] = s.y;
                batch[at + 2] = radius;
            }
            velocity_batch[at + 4] = s.dx;
            velocity_batch[at + 5] = s.dy;
            dye_batch[at + 4] = s.color[0];
            dye_batch[at + 5] = s.color[1];
            dye_batch[at + 6] = s.color[2];
        }
        self.queue.write_buffer(
            &self.ubos.batch_velocity,
            0,
            bytemuck::cast_slice(&velocity_batch),
        );
        self.queue
            .write_buffer(&self.ubos.batch_dye, 0, bytemuck::cast_slice(&dye_batch));

        // `value` carries the count; the rest of the block is shared.
        let params = Params {
            aspect_ratio: aspect,
            value: splats.len() as f32,
            ..Default::default()
        };
        write(&self.queue, &self.ubos.splat_velocity, params);
        write(&self.queue, &self.ubos.splat_dye, params);

        let bg_velocity = self.bind_batched(
            &self.ubos.splat_velocity,
            &self.ubos.batch_velocity,
            &self.velocity.read,
            None,
            None,
        );
        let bg_dye = self.bind_batched(
            &self.ubos.splat_dye,
            &self.ubos.batch_dye,
            &self.dye.read,
            None,
            None,
        );

        let mut enc = self.encoder("fluid splat");
        draw(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.splat_velocity,
            &bg_velocity,
        );
        draw(
            &mut enc,
            &self.dye.write.view,
            Load::Clear,
            &self.pipes.splat_dye,
            &bg_dye,
        );
        self.queue.submit([enc.finish()]);
        self.velocity.swap();
        self.dye.swap();
    }

    /// One step of the solver. Called by [`FluidSim::update`]; public so a
    /// caller driving its own clock can skip the input handling.
    pub fn step(&mut self, dt: f32) {
        let c = self.config;
        let texel = self.velocity.texel();
        let mut enc = self.encoder("fluid step");

        // 1. Curl of the velocity field.
        let p = Params {
            texel_size: texel,
            ..Default::default()
        };
        write(&self.queue, &self.ubos.curl, p);
        let bg = self.bind(&self.ubos.curl, &self.velocity.read, None, None);
        draw(
            &mut enc,
            &self.curl.view,
            Load::Clear,
            &self.pipes.curl,
            &bg,
        );

        // 2. Vorticity confinement, from the curl just computed.
        write(
            &self.queue,
            &self.ubos.vorticity,
            Params {
                texel_size: texel,
                dt,
                curl: c.curl,
                ..Default::default()
            },
        );
        let bg = self.bind(
            &self.ubos.vorticity,
            &self.velocity.read,
            Some(&self.curl),
            None,
        );
        draw(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.vorticity,
            &bg,
        );
        self.velocity.swap();

        // 3. Divergence of the result.
        self.record_divergence(&mut enc);

        // 4. Start the pressure solve from a fraction of last frame's answer.
        write(
            &self.queue,
            &self.ubos.clear,
            Params {
                value: c.pressure,
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.clear, &self.pressure.read, None, None);
        draw(
            &mut enc,
            &self.pressure.write.view,
            Load::Clear,
            &self.pipes.clear,
            &bg,
        );
        self.pressure.swap();

        // 5. Solve for the pressure.
        match c.pressure_solver {
            PressureSolver::Jacobi => self.record_jacobi(&mut enc, c.pressure_iterations, 1.0),
            PressureSolver::Multigrid => self.record_multigrid(&mut enc),
        }

        // 6. Subtract the gradient: the field is now divergence-free.
        write(
            &self.queue,
            &self.ubos.gradient,
            Params {
                texel_size: texel,
                value: c.gradient_scale,
                ..Default::default()
            },
        );
        let bg = self.bind(
            &self.ubos.gradient,
            &self.pressure.read,
            Some(&self.velocity.read),
            None,
        );
        draw(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.gradient,
            &bg,
        );
        self.velocity.swap();

        // 7. The field carries itself, and 8. it carries the dye. Note the
        // texel size is the velocity's in both: the lookup is an offset in uv,
        // which is the same number whatever resolution the dye is stored at.
        self.record_advection(&mut enc, dt, texel, true);
        self.record_advection(&mut enc, dt, texel, false);

        self.queue.submit([enc.finish()]);
    }

    /// Advect one field — the velocity itself, or the dye.
    ///
    /// Semi-Lagrangian is one pass. MacCormack is two: the same forward trace,
    /// then a correction that advects the result back and takes off half of
    /// however far it failed to return. The forward pass does not dissipate in
    /// that case; the correction applies it once at the end, so the fade is
    /// the same either way.
    fn record_advection(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        dt: f32,
        texel: [f32; 2],
        velocity_field: bool,
    ) {
        let c = self.config;
        let maccormack = c.advection == Advection::MacCormack;
        let dissipation = if velocity_field {
            c.velocity_dissipation
        } else {
            c.density_dissipation
        };
        let field_texel = if velocity_field {
            self.velocity.texel()
        } else {
            self.dye.texel()
        };
        let (forward_ubo, correct_ubo) = if velocity_field {
            (&self.ubos.advect_velocity, &self.ubos.correct_velocity)
        } else {
            (&self.ubos.advect_dye, &self.ubos.correct_dye)
        };
        let (forward_pipe, correct_pipe) = if velocity_field {
            (&self.pipes.advect_velocity, &self.pipes.correct_velocity)
        } else {
            (&self.pipes.advect_dye, &self.pipes.correct_dye)
        };

        let params = Params {
            texel_size: texel,
            dye_texel_size: field_texel,
            dt,
            // The forward half of a MacCormack step must not fade the field
            // twice.
            dissipation: if maccormack { 0.0 } else { dissipation },
            ..Default::default()
        };
        write(&self.queue, forward_ubo, params);

        {
            let (source, target) = if velocity_field {
                (&self.velocity.read, &self.velocity.write)
            } else {
                (&self.dye.read, &self.dye.write)
            };
            let bg = self.bind(forward_ubo, &self.velocity.read, Some(source), None);
            draw(enc, &target.view, Load::Clear, forward_pipe, &bg);
        }

        if !maccormack {
            if velocity_field {
                self.velocity.swap();
            } else {
                self.dye.swap();
            }
            return;
        }

        // The correction reads the velocity, the forward result and the field
        // as it was — three textures, which is exactly the bind group's
        // capacity, and the reason the limiter samples its four source texels
        // from the original rather than re-tracing through a fourth.
        write(
            &self.queue,
            correct_ubo,
            Params {
                dissipation,
                ..params
            },
        );
        if velocity_field {
            // The forward result is in `write` and the original in `read`, so
            // the corrected value has nowhere to go but back over `read` —
            // which is safe, because nothing reads it after this.
            let bg = self.bind_three(
                correct_ubo,
                &self.velocity.read,
                &self.velocity.write,
                &self.velocity.read,
            );
            draw(
                enc,
                &self.scratch_velocity.view,
                Load::Clear,
                correct_pipe,
                &bg,
            );
            std::mem::swap(&mut self.velocity.read, &mut self.scratch_velocity);
        } else {
            let bg = self.bind_three(
                correct_ubo,
                &self.velocity.read,
                &self.dye.write,
                &self.dye.read,
            );
            draw(enc, &self.scratch_dye.view, Load::Clear, correct_pipe, &bg);
            std::mem::swap(&mut self.dye.read, &mut self.scratch_dye);
        }
    }

    /// `count` Jacobi sweeps on the top-level pressure field, relaxed by
    /// `weight` — 1 for the plain solver, [`MG_WEIGHT`] when smoothing.
    ///
    /// The field ping-pongs, so there are two bind groups and the loop
    /// alternates between them rather than making twenty of them.
    fn record_jacobi(&mut self, enc: &mut wgpu::CommandEncoder, count: u32, weight: f32) {
        write(
            &self.queue,
            &self.ubos.pressure,
            Params {
                texel_size: self.velocity.texel(),
                value: weight,
                ..Default::default()
            },
        );
        let bg_even = self.bind(
            &self.ubos.pressure,
            &self.pressure.read,
            Some(&self.divergence),
            None,
        );
        let bg_odd = self.bind(
            &self.ubos.pressure,
            &self.pressure.write,
            Some(&self.divergence),
            None,
        );
        for i in 0..count {
            let (bg, target) = if i % 2 == 0 {
                (&bg_even, &self.pressure.write.view)
            } else {
                (&bg_odd, &self.pressure.read.view)
            };
            draw(enc, target, Load::Clear, &self.pipes.pressure, bg);
        }
        if count % 2 == 1 {
            self.pressure.swap();
        }
    }

    /// A multigrid V-cycle: down to the coarsest grid, then back up.
    ///
    /// The shape is the textbook one. On the way down each level smooths a
    /// little, measures what it failed to satisfy, and hands that residual to
    /// a grid half the size as a fresh problem starting from zero. At the
    /// bottom — four to eight texels across — a few sweeps *are* a solve,
    /// because there is no longer any such thing as a long-wavelength error.
    /// On the way up each correction is interpolated into the level above and
    /// smoothed again to take the interpolation's own roughness out.
    ///
    /// Everything here reuses the Jacobi sweep as the smoother; the only new
    /// stages are the residual, the restriction and the prolongation.
    fn record_multigrid(&mut self, enc: &mut wgpu::CommandEncoder) {
        if self.mg.is_empty() {
            // Too small to coarsen — a grid this size is already the coarsest
            // level, and sweeping it is the solve.
            self.record_jacobi(enc, self.config.pressure_iterations, 1.0);
            return;
        }
        // Two sweeps either side of the recursion is the classic choice: one
        // is not enough to remove the high-frequency error that restriction
        // cannot represent, and three buys little.
        let nu = 2;
        // The coarsest grid gets the rest of the budget, which is what makes
        // the long-wavelength error actually go away rather than be deferred.
        let coarse_sweeps = self.config.pressure_iterations.max(nu * 2);

        for cycle in 0..self.config.multigrid_cycles.max(1) {
            // The first cycle starts from last frame's scaled pressure, as
            // Jacobi does; later cycles continue from where the last left off.
            let _ = cycle;

            // --- down ---------------------------------------------------
            self.record_smooth_top(enc, nu);
            for level in 0..self.mg.len() {
                self.record_residual(enc, level);
                self.record_restrict(enc, level);
                // The coarse problem is for a *correction*, so it starts at
                // zero — a clear-only pass, no draw.
                let _ = fullscreen::begin(enc, &self.mg[level].pressure.read.view, Load::Clear);
                let sweeps = if level + 1 == self.mg.len() {
                    coarse_sweeps
                } else {
                    nu
                };
                self.record_smooth_level(enc, level, sweeps);
            }
            // --- up -----------------------------------------------------
            for level in (0..self.mg.len()).rev() {
                self.record_prolong(enc, level);
                if level == 0 {
                    self.record_smooth_top(enc, nu);
                } else {
                    self.record_smooth_level(enc, level - 1, nu);
                }
            }
        }
    }

    /// Smooth the top-level pressure against the divergence.
    fn record_smooth_top(&mut self, enc: &mut wgpu::CommandEncoder, sweeps: u32) {
        self.record_jacobi(enc, sweeps, MG_WEIGHT);
    }

    /// Smooth coarse `level`'s correction against its own right-hand side.
    fn record_smooth_level(&mut self, enc: &mut wgpu::CommandEncoder, level: usize, sweeps: u32) {
        let l = &self.mg[level];
        write(
            &self.queue,
            &l.ubo_smooth,
            Params {
                texel_size: l.pressure.texel(),
                value: MG_WEIGHT,
                ..Default::default()
            },
        );
        let bg_even = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            &l.ubo_smooth,
            &self.ubos.batch_idle,
            &l.pressure.read.view,
            Some(&l.rhs.view),
            None,
        );
        let bg_odd = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            &l.ubo_smooth,
            &self.ubos.batch_idle,
            &l.pressure.write.view,
            Some(&l.rhs.view),
            None,
        );
        for i in 0..sweeps {
            let (bg, target) = if i % 2 == 0 {
                (&bg_even, &l.pressure.write.view)
            } else {
                (&bg_odd, &l.pressure.read.view)
            };
            draw(enc, target, Load::Clear, &self.pipes.pressure, bg);
        }
        if sweeps % 2 == 1 {
            self.mg[level].pressure.swap();
        }
    }

    /// What `level`'s solution fails to satisfy. Level 0 means the simulation's
    /// own pressure against the divergence.
    fn record_residual(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let (pressure, rhs, texel) = if level == 0 {
            (&self.pressure.read, &self.divergence, self.pressure.texel())
        } else {
            let l = &self.mg[level - 1];
            (&l.pressure.read, &l.rhs, l.pressure.texel())
        };
        let slot = &self.mg[level].ubo_residual;
        write(
            &self.queue,
            slot,
            Params {
                texel_size: texel,
                ..Default::default()
            },
        );
        let bg = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            slot,
            &self.ubos.batch_idle,
            &pressure.view,
            Some(&rhs.view),
            None,
        );
        // At this level's own resolution; the restriction that follows is
        // what takes it down a level. Writing it straight into the coarse
        // texture instead would be a point sample, which is the one thing a
        // restriction must not be — it would alias exactly the high-frequency
        // error the smoother was there to remove.
        let target = if level == 0 {
            &self.residual
        } else {
            &self.mg[level - 1].residual
        };
        draw(enc, &target.view, Load::Clear, &self.pipes.residual, &bg);
    }

    /// Average `level`'s residual onto the half-size grid below it.
    fn record_restrict(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let (residual, fine_texel) = if level == 0 {
            (&self.residual, self.pressure.texel())
        } else {
            let l = &self.mg[level - 1];
            (&l.residual, l.pressure.texel())
        };
        let slot = &self.mg[level].ubo_restrict;
        write(
            &self.queue,
            slot,
            Params {
                texel_size: fine_texel,
                ..Default::default()
            },
        );
        let bg = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            slot,
            &self.ubos.batch_idle,
            &residual.view,
            None,
            None,
        );
        draw(
            enc,
            &self.mg[level].rhs.view,
            Load::Clear,
            &self.pipes.restrict,
            &bg,
        );
    }

    /// Interpolate `level`'s correction into the level above and add it.
    fn record_prolong(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let coarse = &self.mg[level].pressure.read;
        let slot = &self.mg[level].ubo_prolong;
        write(&self.queue, slot, Params::default());
        let (fine_read, fine_write) = if level == 0 {
            (&self.pressure.read, &self.pressure.write)
        } else {
            let l = &self.mg[level - 1];
            (&l.pressure.read, &l.pressure.write)
        };
        let bg = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            slot,
            &self.ubos.batch_idle,
            &coarse.view,
            Some(&fine_read.view),
            None,
        );
        draw(enc, &fine_write.view, Load::Clear, &self.pipes.prolong, &bg);
        if level == 0 {
            self.pressure.swap();
        } else {
            self.mg[level - 1].pressure.swap();
        }
    }

    /// The divergence of the current velocity field, into the divergence
    /// texture. Part of [`FluidSim::step`], and factored out because it is also
    /// the only way to *measure* what the projection did — see the tests.
    fn record_divergence(&self, enc: &mut wgpu::CommandEncoder) {
        write(
            &self.queue,
            &self.ubos.divergence,
            Params {
                texel_size: self.velocity.texel(),
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.divergence, &self.velocity.read, None, None);
        draw(
            enc,
            &self.divergence.view,
            Load::Clear,
            &self.pipes.divergence,
            &bg,
        );
    }

    // -- display ----------------------------------------------------------

    /// Composite the fluid into a texture, as it would appear on a canvas:
    /// over `back_color`, or over a checkerboard if `transparent` is set.
    pub fn render(&mut self, target: &wgpu::Texture) {
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_into(fullscreen::Target::from_texture(target, &view), true);
    }

    /// Composite for a capture rather than for a screen: no checkerboard, and
    /// with `transparent` set the dye keeps its own alpha instead of being
    /// flattened onto the background.
    pub fn render_capture(&mut self, target: &wgpu::Texture) {
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_into(fullscreen::Target::from_texture(target, &view), false);
    }

    /// The general form: any view, given its format and size. `on_screen`
    /// selects between the two compositing conventions above.
    ///
    /// Equivalent to [`render_into`](Self::render_into) with a
    /// [`Target`](crate::renderer::fullscreen::Target) assembled from the
    /// three arguments.
    pub fn render_to_view(
        &mut self,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        on_screen: bool,
    ) {
        self.render_into(
            fullscreen::Target {
                view: target,
                format,
                width: size.0,
                height: size.1,
            },
            on_screen,
        )
    }

    /// Composite into a [`Target`](crate::renderer::fullscreen::Target) — a
    /// view that knows its own format and size, so the three cannot drift
    /// apart across a resize.
    ///
    /// This is what a [`DrawPass`](crate::postprocessing::DrawPass) hands over,
    /// so putting the fluid in a post-processing chain is one line:
    ///
    /// ```no_run
    /// # use threers::postprocessing::DrawPass;
    /// # fn demo(mut fluid: threers::fluid::FluidSim) -> DrawPass {
    /// DrawPass::new("fluid", move |ctx| {
    ///     fluid.update(1.0 / 60.0);
    ///     fluid.render_into(ctx.target(), true);
    /// })
    /// # }
    /// ```
    pub fn render_into(&mut self, target: fullscreen::Target<'_>, on_screen: bool) {
        let c = self.config;
        let (format, size) = (target.format, target.size());
        let target = target.view;
        let mut enc = self.encoder("fluid render");
        if c.bloom {
            self.record_bloom(&mut enc);
        }
        if c.sunrays {
            self.record_sunrays(&mut enc);
        }

        let encode = if format.is_srgb() { FLAG_ENCODE } else { 0 };
        let mut flags = encode;
        if c.shading {
            flags |= FLAG_SHADING;
        }
        if c.bloom {
            flags |= FLAG_BLOOM;
        }
        if c.sunrays {
            flags |= FLAG_SUNRAYS;
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        write(
            &self.queue,
            &self.ubos.display,
            Params {
                // The shading normal is a gradient over the *output* grid, so
                // this is the target's texel, not the dye's.
                texel_size: [1.0 / w as f32, 1.0 / h as f32],
                flags,
                ..Default::default()
            },
        );
        write(
            &self.queue,
            &self.ubos.background,
            Params {
                color: [c.back_color[0], c.back_color[1], c.back_color[2], 1.0],
                aspect_ratio: w as f32 / h as f32,
                flags: encode,
                ..Default::default()
            },
        );
        let bg_display = self.bind(
            &self.ubos.display,
            &self.dye.read,
            Some(&self.bloom),
            Some(&self.sunrays),
        );
        let bg_background = self.bind(&self.ubos.background, &self.dye.read, None, None);

        // One pass, up to two draws: the fluid composites over whatever the
        // first draw left, which is what the original's blend did over its
        // background fill.
        let pipes = self.pipes.output(&self.device, format);
        let display = if on_screen || !c.transparent {
            &pipes.display_over
        } else {
            &pipes.display_replace
        };
        {
            let mut pass = fullscreen::begin(&mut enc, target, Load::Clear);
            if !c.transparent {
                pass.set_pipeline(&pipes.color);
                pass.set_bind_group(0, &bg_background, &[]);
                pass.draw(0..3, 0..1);
            } else if on_screen {
                pass.set_pipeline(&pipes.checkerboard);
                pass.set_bind_group(0, &bg_background, &[]);
                pass.draw(0..3, 0..1);
            }
            pass.set_pipeline(display);
            pass.set_bind_group(0, &bg_display, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Threshold the dye, blur it down a halving chain, add the levels back up,
    /// and leave the result in the bloom texture.
    fn record_bloom(&mut self, enc: &mut wgpu::CommandEncoder) {
        if self.bloom_chain.len() < 2 {
            return;
        }
        let c = self.config;
        let knee = c.bloom_threshold * c.bloom_soft_knee + 0.0001;
        write(
            &self.queue,
            &self.ubos.bloom_prefilter,
            Params {
                curve: [
                    c.bloom_threshold - knee,
                    knee * 2.0,
                    0.25 / knee,
                    c.bloom_threshold,
                ],
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.bloom_prefilter, &self.dye.read, None, None);
        draw(
            enc,
            &self.bloom.view,
            Load::Clear,
            &self.pipes.bloom_prefilter,
            &bg,
        );

        // Down: each level blurs the one above it at half the size.
        let mut last = &self.bloom;
        for (i, dest) in self.bloom_chain.iter().enumerate() {
            let slot = &self.ubos.bloom_chain[i];
            write(
                &self.queue,
                slot,
                Params {
                    texel_size: last.texel(),
                    ..Default::default()
                },
            );
            let bg = bind(
                &self.device,
                &self.pipes.bgl,
                &self.sampler,
                slot,
                &self.ubos.batch_idle,
                &last.view,
                None,
                None,
            );
            draw(enc, &dest.view, Load::Clear, &self.pipes.bloom_blur, &bg);
            last = dest;
        }

        // Up: blur each level into the one above, *adding* — which is what
        // turns a chain of small blurs into one wide one.
        let n = self.bloom_chain.len();
        for i in (0..n - 1).rev() {
            let base = &self.bloom_chain[i];
            let slot = &self.ubos.bloom_chain[n + i];
            write(
                &self.queue,
                slot,
                Params {
                    texel_size: last.texel(),
                    ..Default::default()
                },
            );
            let bg = bind(
                &self.device,
                &self.pipes.bgl,
                &self.sampler,
                slot,
                &self.ubos.batch_idle,
                &last.view,
                None,
                None,
            );
            draw(enc, &base.view, Load::Keep, &self.pipes.bloom_blur_add, &bg);
            last = base;
        }

        write(
            &self.queue,
            &self.ubos.bloom_final,
            Params {
                texel_size: last.texel(),
                value: c.bloom_intensity,
                ..Default::default()
            },
        );
        let bg = bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            &self.ubos.bloom_final,
            &self.ubos.batch_idle,
            &last.view,
            None,
            None,
        );
        draw(
            enc,
            &self.bloom.view,
            Load::Clear,
            &self.pipes.bloom_final,
            &bg,
        );
    }

    /// Mask the dye into an occlusion map, march it radially, and blur the
    /// result once on each axis.
    fn record_sunrays(&mut self, enc: &mut wgpu::CommandEncoder) {
        write(&self.queue, &self.ubos.sunrays_mask, Params::default());
        let bg = self.bind(&self.ubos.sunrays_mask, &self.dye.read, None, None);
        // The mask goes into the dye's spare buffer: it is the one scratch
        // texture of the right size and format that is free at this point in
        // the frame, and the fields do not swap, so nothing is lost.
        draw(
            enc,
            &self.dye.write.view,
            Load::Clear,
            &self.pipes.sunrays_mask,
            &bg,
        );

        write(
            &self.queue,
            &self.ubos.sunrays,
            Params {
                value: self.config.sunrays_weight,
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.sunrays, &self.dye.write, None, None);
        draw(
            enc,
            &self.sunrays.view,
            Load::Clear,
            &self.pipes.sunrays,
            &bg,
        );

        // One separable blur, to take the stepping out of the march.
        let texel = self.sunrays.texel();
        write(
            &self.queue,
            &self.ubos.blur_h,
            Params {
                texel_size: [texel[0], 0.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.blur_h, &self.sunrays, None, None);
        draw(
            enc,
            &self.sunrays_temp.view,
            Load::Clear,
            &self.pipes.blur,
            &bg,
        );
        write(
            &self.queue,
            &self.ubos.blur_v,
            Params {
                texel_size: [0.0, texel[1]],
                ..Default::default()
            },
        );
        let bg = self.bind(&self.ubos.blur_v, &self.sunrays_temp, None, None);
        draw(enc, &self.sunrays.view, Load::Clear, &self.pipes.blur, &bg);
    }

    // -- helpers ----------------------------------------------------------

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Render into a fresh `Rgba8Unorm` texture and read it back — a
    /// screenshot, as the original's capture button takes.
    ///
    /// Native only: reading a texture back means waiting for the queue, and a
    /// browser's main thread cannot be blocked. On the web, render into a
    /// canvas and let the page do the capture.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_rgba(&mut self, width: u32, height: u32) -> Vec<u8> {
        let (width, height) = (width.max(1), height.max(1));
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fluid capture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Not the sRGB variant: the display pass already hands back
            // display-referred values, so storing them untouched is what
            // reproduces the GL canvas byte for byte.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.render_capture(&target);
        crate::renderer::readback::texture_bytes(&self.device, &self.queue, &target)
    }

    fn bind(
        &self,
        params: &wgpu::Buffer,
        a: &Fbo,
        b: Option<&Fbo>,
        c: Option<&Fbo>,
    ) -> wgpu::BindGroup {
        self.bind_batched(params, &self.ubos.batch_idle, a, b, c)
    }

    /// Three distinct textures, which only the MacCormack correction needs.
    fn bind_three(&self, params: &wgpu::Buffer, a: &Fbo, b: &Fbo, c: &Fbo) -> wgpu::BindGroup {
        self.bind_batched(params, &self.ubos.batch_idle, a, Some(b), Some(c))
    }

    /// [`Self::bind`] with a splat batch rather than the idle one.
    fn bind_batched(
        &self,
        params: &wgpu::Buffer,
        batch: &wgpu::Buffer,
        a: &Fbo,
        b: Option<&Fbo>,
        c: Option<&Fbo>,
    ) -> wgpu::BindGroup {
        bind(
            &self.device,
            &self.pipes.bgl,
            &self.sampler,
            params,
            batch,
            &a.view,
            b.map(|f| &f.view),
            c.map(|f| &f.view),
        )
    }
}

fn bloom_chain(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    iterations: u32,
    format: wgpu::TextureFormat,
) -> Vec<Fbo> {
    let mut chain = Vec::new();
    for i in 0..iterations {
        let w = width >> (i + 1);
        let h = height >> (i + 1);
        // Below two texels a blur has no neighbours left to average, and the
        // chain stops wherever that happens rather than at a fixed depth.
        if w < 2 || h < 2 {
            break;
        }
        chain.push(Fbo::new(device, "fluid bloom level", w, h, format));
    }
    chain
}

fn write(queue: &wgpu::Queue, buffer: &wgpu::Buffer, params: Params) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&params));
}

#[cfg(test)]
mod tests;
