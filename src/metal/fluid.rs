//! The fluid solver on the Metal backend.
//!
//! ```no_run
//! # #[cfg(all(feature = "metal", feature = "fluid"))]
//! # fn demo() -> Result<(), threers::metal::MetalError> {
//! use threers::fluid::FluidConfig;
//! use threers::metal::{fluid_shader_source, MetalDevice, MetalFluid};
//!
//! let device = MetalDevice::with_shader_source(&fluid_shader_source())?;
//! let mut fluid = MetalFluid::new(&device, 1280, 720, FluidConfig::default())?;
//! fluid.multiple_splats(&device, 20);
//! fluid.update(&device, 1.0 / 60.0);
//! let rgba = fluid.capture_rgba(&device)?;
//! # let _ = rgba;
//! # Ok(()) }
//! ```
//!
//! # Why a second implementation at all
//!
//! The same reason the rest of `src/metal` exists: it talks to Metal directly
//! through the Objective-C runtime, with no wgpu underneath, which is what
//! makes it usable from an app that already owns a `MTLDevice` and a
//! `CAMetalLayer` and does not want a second graphics abstraction inside it.
//! The simulation is the same simulation, and the shaders are the same
//! shaders: every pass is written once, with [`passes!`](crate::passes), in
//! `src/fluid/shader.rs`, and that source is emitted as the wgpu module's
//! WGSL and as this backend's MSL (the read-out is the exception; see
//! below). `metal_fluid_matches_wgpu` runs both from the same seed and
//! compares the pictures: they come back identical, to within one level in a
//! channel after ninety frames.
//!
//! # What it covers
//!
//! All of it: the solver and the splats — curl, vorticity confinement,
//! divergence, the Jacobi sweeps or a multigrid V-cycle, the projection,
//! semi-Lagrangian or MacCormack advection — the dye lit as a height field,
//! bloom, sunrays, the dither, and the frame-rate read-out.
//! `bloom_sunrays_maccormack_and_multigrid_match_wgpu` holds each to the
//! wgpu module's picture, and [`resize`](MetalFluid::resize) resamples the
//! dye and velocity into the new grids as it does.
//!
//! The read-out is the one piece where the two backends share Rust rather
//! than only a shape: `crate::fluid::read_out::Rate` owns the ring of frame
//! times, the smoothing and the rule for what counts as a frame, and all
//! three solvers call it. Only the glyphs and the chart are transcribed, and
//! `the_two_read_outs_agree` renders the same pinned history through both
//! and compares the pictures — which came back identical to the byte.


use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg1, msg2, msg3, sel, AutoreleasePool, Id, Owned};
use super::postfx::{encode_fullscreen, fullscreen_pipeline_blend, fullscreen_pipeline_with};
use super::resources::{new_texture_2d, upload_texture_2d};
use super::target::{blit_texture_to_buffer_on_cmd, finish_texture_readback_into};

use crate::fluid::passes::{Params, FLAG_BLOOM, FLAG_ENCODE, FLAG_SHADING, FLAG_SUNRAYS};
use crate::fluid::read_out::{Frames, Rate};
use crate::fluid::{Advection, FluidConfig, PressureSolver};

/// Every field is four-channel half float, where the wgpu module narrows the
/// velocity to two channels and the pressure to one.
///
/// Metal has `r16Float` and `rg16Float` and would take them; the reason not to
/// is that the wgpu side picks its formats per *adapter*, so the two backends
/// would be storing the same field differently and the parity test would be
/// comparing two discretisations rather than two implementations. The cost is
/// memory nobody reads.
const FORMAT: NSUInteger = pixel_format::RGBA16_FLOAT;

/// A field that is read and written in the same pass, so it needs two copies.
struct Pair {
    read: Owned,
    write: Owned,
}

impl Pair {
    fn new(device: &MetalDevice, w: u32, h: u32) -> Result<Self, MetalError> {
        Ok(Self {
            read: field(device, w, h)?,
            write: field(device, w, h)?,
        })
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }
}

/// One coarse level of the multigrid V-cycle: its correction, the problem it
/// solves for, and the residual it hands down.
struct Level {
    pressure: Pair,
    rhs: Owned,
    residual: Owned,
    size: (u32, u32),
}

/// Every texture whose size follows the canvas and the configuration, built
/// in one place for [`MetalFluid::new`], `resize` and `set_config`.
struct Grids {
    sim: (u32, u32),
    dye_size: (u32, u32),
    velocity: Pair,
    dye: Pair,
    pressure: Pair,
    divergence: Owned,
    curl: Owned,
    /// MacCormack's corrected fields, swapped in for the ones they replace.
    scratch_velocity: Owned,
    scratch_dye: Owned,
    /// The top level's residual, and the coarser levels below it.
    residual: Owned,
    mg: Vec<Level>,
    /// The thresholded dye, and the halving chain it is blurred down.
    bloom: (Owned, (u32, u32)),
    bloom_chain: Vec<(Owned, (u32, u32))>,
    /// The rays, and the blur's other half.
    sunrays: (Owned, (u32, u32)),
    sunrays_temp: Owned,
}

impl Grids {
    fn new(device: &MetalDevice, c: &FluidConfig, width: u32, height: u32) -> Result<Self, MetalError> {
        let sim = grid(c.sim_resolution, width, height);
        let dye_size = grid(c.dye_resolution, width, height);
        let bloom = grid(c.bloom_resolution, width, height);
        let rays = grid(c.sunrays_resolution, width, height);
        // As `mg_levels` and `bloom_chain` on the wgpu side.
        let mut mg = Vec::new();
        let (mut w, mut h) = sim;
        while w / 2 >= crate::fluid::MG_COARSEST && h / 2 >= crate::fluid::MG_COARSEST && mg.len() < 8 {
            w /= 2;
            h /= 2;
            mg.push(Level {
                pressure: Pair::new(device, w, h)?,
                rhs: field(device, w, h)?,
                residual: field(device, w, h)?,
                size: (w, h),
            });
        }
        let mut bloom_chain = Vec::new();
        for i in 0..c.bloom_iterations {
            let (w, h) = (bloom.0 >> (i + 1), bloom.1 >> (i + 1));
            if w < 2 || h < 2 {
                break;
            }
            bloom_chain.push((field(device, w, h)?, (w, h)));
        }
        Ok(Self {
            sim,
            dye_size,
            velocity: Pair::new(device, sim.0, sim.1)?,
            dye: Pair::new(device, dye_size.0, dye_size.1)?,
            pressure: Pair::new(device, sim.0, sim.1)?,
            divergence: field(device, sim.0, sim.1)?,
            curl: field(device, sim.0, sim.1)?,
            scratch_velocity: field(device, sim.0, sim.1)?,
            scratch_dye: field(device, dye_size.0, dye_size.1)?,
            residual: field(device, sim.0, sim.1)?,
            mg,
            bloom: (field(device, bloom.0, bloom.1)?, bloom),
            bloom_chain,
            sunrays: (field(device, rays.0, rays.1)?, rays),
            sunrays_temp: field(device, rays.0, rays.1)?,
        })
    }

    /// Every texture, for zeroing.
    fn all(&self) -> Vec<Id> {
        let mut out = vec![
            self.velocity.read.id(),
            self.velocity.write.id(),
            self.dye.read.id(),
            self.dye.write.id(),
            self.pressure.read.id(),
            self.pressure.write.id(),
            self.divergence.id(),
            self.curl.id(),
            self.scratch_velocity.id(),
            self.scratch_dye.id(),
            self.residual.id(),
            self.bloom.0.id(),
            self.sunrays.0.id(),
            self.sunrays_temp.id(),
        ];
        for l in &self.mg {
            out.extend([l.pressure.read.id(), l.pressure.write.id(), l.rhs.id(), l.residual.id()]);
        }
        out.extend(self.bloom_chain.iter().map(|(t, _)| t.id()));
        out
    }
}

/// One texel, in uv, of a grid of `size`.
fn texel_of(size: (u32, u32)) -> [f32; 2] {
    [1.0 / size.0 as f32, 1.0 / size.1 as f32]
}

fn field(device: &MetalDevice, w: u32, h: u32) -> Result<Owned, MetalError> {
    new_texture_2d(
        device,
        w,
        h,
        FORMAT,
        texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
        storage_mode::PRIVATE,
        1,
        1,
    )
}

/// The pipelines, one per fragment function. Every field is the same format,
/// so unlike the wgpu side there is no pipeline-per-format cache — except for
/// the display, whose target belongs to the caller.
struct Pipelines {
    clear: Owned,
    splat: Owned,
    advection: Owned,
    divergence: Owned,
    curl: Owned,
    vorticity: Owned,
    pressure: Owned,
    gradient: Owned,
    advect_correct: Owned,
    residual: Owned,
    restrict: Owned,
    prolong: Owned,
    bloom_prefilter: Owned,
    bloom_blur: Owned,
    /// The blur back up the chain, adding: the premultiplied blend, which
    /// over a source alpha of zero is `src + dst`.
    bloom_blur_add: Owned,
    bloom_final: Owned,
    sunrays_mask: Owned,
    sunrays: Owned,
    blur: Owned,
    /// Keyed by the caller's colour format. `display` replaces the target,
    /// `display_over` composites onto it — see
    /// [`fullscreen_pipeline_blend`].
    display: Vec<(NSUInteger, Owned)>,
    display_over: Vec<(NSUInteger, Owned)>,
    color: Vec<(NSUInteger, Owned)>,
    /// The frame-rate read-out, composited over the finished picture.
    overlay: Vec<(NSUInteger, Owned)>,
}

/// A 2D fluid simulation running on Metal.
pub struct MetalFluid {
    config: FluidConfig,
    width: u32,
    height: u32,
    grids: Grids,

    pipes: Pipelines,

    rng: u64,
    colour_timer: f32,
    paused: bool,
    /// The mouse and fingers, kept as the wgpu module keeps them.
    pointers: crate::fluid::Pointers,

    /// The read-out's state. `rate` is the same counter and the same ring
    /// the wgpu module keeps; `history` is this backend's own 64x1 texture
    /// for it, uploaded with `replaceRegion` rather than a queue write.
    rate: Rate,
    history: Owned,
}

impl MetalFluid {
    /// `device` must have been built with [`fluid_shader_source`] — the
    /// default library does not carry the fluid's functions.
    ///
    /// [`fluid_shader_source`]: crate::metal::fluid_shader_source
    pub fn new(
        device: &MetalDevice,
        width: u32,
        height: u32,
        config: FluidConfig,
    ) -> Result<Self, MetalError> {
        if config.pressure_iterations == 0 {
            return Err(MetalError::Unsupported(
                "fluid: pressure_iterations must be at least 1".into(),
            ));
        }
        let (width, height) = (width.max(1), height.max(1));

        let p = |name: &str| fullscreen_pipeline_with(device, "fluid_vs", name, FORMAT);
        let pipes = Pipelines {
            clear: p("fluid_fs_clear")?,
            splat: p("fluid_fs_splat")?,
            advection: p("fluid_fs_advection")?,
            divergence: p("fluid_fs_divergence")?,
            curl: p("fluid_fs_curl")?,
            vorticity: p("fluid_fs_vorticity")?,
            pressure: p("fluid_fs_pressure")?,
            gradient: p("fluid_fs_gradient")?,
            advect_correct: p("fluid_fs_advect_correct")?,
            residual: p("fluid_fs_residual")?,
            restrict: p("fluid_fs_restriction")?,
            prolong: p("fluid_fs_prolong")?,
            bloom_prefilter: p("fluid_fs_bloom_prefilter")?,
            bloom_blur: p("fluid_fs_bloom_blur")?,
            bloom_blur_add: fullscreen_pipeline_blend(device, "fluid_vs", "fluid_fs_bloom_blur", FORMAT, true)?,
            bloom_final: p("fluid_fs_bloom_final")?,
            sunrays_mask: p("fluid_fs_sunrays_mask")?,
            sunrays: p("fluid_fs_sunrays")?,
            blur: fullscreen_pipeline_with(device, "fluid_vs_blur", "fluid_fs_blur", FORMAT)?,
            display: Vec::new(),
            display_over: Vec::new(),
            color: Vec::new(),
            overlay: Vec::new(),
        };

        let fluid = Self {
            grids: Grids::new(device, &config, width, height)?,
            config,
            width,
            height,
            pipes,
            rng: config.seed | 1,
            colour_timer: 0.0,
            paused: false,
            pointers: Default::default(),
            rate: Rate::default(),
            history: new_texture_2d(
                device,
                Frames::LEN as u32,
                1,
                pixel_format::RGBA8_UNORM,
                texture_usage::SHADER_READ,
                storage_mode::SHARED,
                1,
                1,
            )?,
        };
        fluid.zero_fields(device)?;
        Ok(fluid)
    }

    /// Clear every field to zero.
    ///
    /// A fresh `MTLTexture` holds whatever was in that memory. wgpu clears a
    /// new texture before anything can read it, so the other backend never
    /// sees this; here, one NaN left in the velocity reaches the advection on
    /// the first step, the backward trace goes to NaN, and the dye it fetches
    /// comes back as nothing. The symptom is a picture that is uniformly
    /// empty however many splats went in — which is what
    /// `the_solver_keeps_the_dye` pins.
    fn zero_fields(&self, device: &MetalDevice) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let cmd = self.command_buffer(device);
        for t in self.grids.all() {
            clear_to_zero(cmd, t)?;
        }
        commit(cmd);
        Ok(())
    }

    pub fn config(&self) -> &FluidConfig {
        &self.config
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The dye and solver grids, in texels.
    pub fn grid_sizes(&self) -> ((u32, u32), (u32, u32)) {
        (self.grids.dye_size, self.grids.sim)
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Replace the configuration, rebuilding the grids if a resolution
    /// changed — the dye and velocity resampled into them, as on wgpu.
    pub fn set_config(&mut self, device: &MetalDevice, config: FluidConfig) -> Result<(), MetalError> {
        let resized = config.sim_resolution != self.config.sim_resolution
            || config.dye_resolution != self.config.dye_resolution
            || config.bloom_resolution != self.config.bloom_resolution
            || config.bloom_iterations != self.config.bloom_iterations
            || config.sunrays_resolution != self.config.sunrays_resolution;
        self.config = config;
        if resized {
            self.reallocate(device)?;
        }
        Ok(())
    }

    /// Tell the simulation the canvas changed size: the grids are rebuilt at
    /// the new aspect, the dye and velocity resampled into them, as on wgpu.
    pub fn resize(&mut self, device: &MetalDevice, width: u32, height: u32) -> Result<(), MetalError> {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        self.reallocate(device)
    }

    /// Every grid for the current size and configuration, zeroed — but the
    /// dye and the velocity resampled into theirs, bilinearly, as the wgpu
    /// module's `resample`, so a resize does not wipe the picture.
    fn reallocate(&mut self, device: &MetalDevice) -> Result<(), MetalError> {
        let old = std::mem::replace(&mut self.grids, Grids::new(device, &self.config, self.width, self.height)?);
        self.zero_fields(device)?;
        let _pool = AutoreleasePool::new();
        let cmd = self.command_buffer(device);
        // The clear pass at a value of one is a copy at the target's uv.
        let u = Params { value: 1.0, ..Default::default() };
        for (src, dst) in [
            (old.dye.read.id(), self.grids.dye.read.id()),
            (old.velocity.read.id(), self.grids.velocity.read.id()),
        ] {
            encode(cmd, &self.pipes.clear, dst, |enc| bind(enc, u, &[src]))?;
        }
        commit(cmd);
        Ok(())
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    // -- input ------------------------------------------------------------

    /// Press at a pixel position in the canvas; `id` names the mouse (`-1`)
    /// or a touch. As [`FluidSim::pointer_down`](crate::fluid::FluidSim::pointer_down).
    pub fn pointer_down(&mut self, id: i32, x: f32, y: f32) {
        let color = self.generate_color();
        self.pointers.down(id, x / self.width as f32, y / self.height as f32, color);
    }

    /// Move a pressed pointer; the next [`update`](Self::update) pushes the
    /// fluid along the motion.
    pub fn pointer_move(&mut self, id: i32, x: f32, y: f32) {
        let (w, h) = (self.width as f32, self.height as f32);
        self.pointers.move_to(id, x / w, y / h, w / h);
    }

    /// Release a pointer.
    pub fn pointer_up(&mut self, id: i32) {
        self.pointers.up(id);
    }

    /// A pointer's dye: a random hue, dim, as the wgpu module draws it.
    fn generate_color(&mut self) -> [f32; 3] {
        let c = crate::fluid::hsv_to_rgb_for_backends(self.next_f32(), 1.0, 1.0);
        [c[0] * 0.15, c[1] * 0.15, c[2] * 0.15]
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0
    }

    /// `count` splats at random positions, with random hues — the same
    /// generator and the same sequence as the wgpu module, so a shared seed
    /// gives both backends the same input.
    pub fn multiple_splats(&mut self, device: &MetalDevice, count: u32) {
        let mut burst = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let hue = self.next_f32();
            let c = crate::fluid::hsv_to_rgb_for_backends(hue, 1.0, 1.0);
            let colour = [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5];
            let x = self.next_f32();
            let y = self.next_f32();
            let dx = 1000.0 * (self.next_f32() - 0.5);
            let dy = 1000.0 * (self.next_f32() - 0.5);
            burst.push((x, y, dx, dy, colour));
        }
        // In one batch, as the wgpu module sums them.
        let _ = self.splats(device, &burst);
    }

    /// One splat: `x`, `y` in uv with the origin at the top left.
    pub fn splat(
        &mut self,
        device: &MetalDevice,
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
        color: [f32; 3],
    ) -> Result<(), MetalError> {
        self.splats(device, &[(x, y, dx, dy, color)])
    }

    /// Splats in batches of up to [`MAX_BATCH`](crate::fluid::passes::MAX_BATCH),
    /// two passes and one submit a batch, as `record_splat_batch` on the
    /// wgpu side: the pass sums them, `value` of them.
    fn splats(&mut self, device: &MetalDevice, list: &[(f32, f32, f32, f32, [f32; 3])]) -> Result<(), MetalError> {
        use crate::fluid::passes::MAX_BATCH;
        let aspect = self.width as f32 / self.height as f32;
        // In canvas fractions, widened with the aspect so a splat keeps its
        // physical size on a wide canvas.
        let mut radius = self.config.splat_radius / 100.0;
        if aspect > 1.0 {
            radius *= aspect;
        }
        for chunk in list.chunks(MAX_BATCH) {
            let _pool = AutoreleasePool::new();
            // Two `vec4`s a splat: (point, radius, _) and (colour, _).
            let mut velocity = vec![0.0f32; chunk.len() * 8];
            let mut dye = vec![0.0f32; chunk.len() * 8];
            for (k, &(x, y, dx, dy, c)) in chunk.iter().enumerate() {
                for batch in [&mut velocity, &mut dye] {
                    batch[k * 8..k * 8 + 3].copy_from_slice(&[x, y, radius]);
                }
                velocity[k * 8 + 4..k * 8 + 6].copy_from_slice(&[dx, dy]);
                dye[k * 8 + 4..k * 8 + 7].copy_from_slice(&c);
            }
            let u = Params {
                aspect_ratio: aspect,
                value: chunk.len() as f32,
                ..Default::default()
            };
            let cmd = self.command_buffer(device);
            for (field, batch) in [(&self.grids.velocity, &velocity), (&self.grids.dye, &dye)] {
                encode(cmd, &self.pipes.splat, field.write.id(), |enc| {
                    bind(enc, u, &[field.read.id()]);
                    bind_batch(enc, batch);
                })?;
            }
            commit(cmd);
            self.grids.velocity.swap();
            self.grids.dye.swap();
        }
        Ok(())
    }

    // -- the frame --------------------------------------------------------

    /// A frame: the solver stepped by `dt` scaled by
    /// [`FluidConfig::time_scale`](crate::fluid::FluidConfig::time_scale).
    ///
    /// As on the wgpu side, [`step`](Self::step) takes the timestep it is
    /// given and this one applies the knob — so the two backends can be held
    /// to the same picture at a fixed `dt` while an application still has a
    /// speed control.
    pub fn update(&mut self, device: &MetalDevice, dt: f32) {
        let dt = dt * self.config.time_scale.max(0.0);
        // The hues cycle, and the pointers' motion goes in even while paused,
        // as on the wgpu side.
        if self.config.colorful {
            self.colour_timer += dt * self.config.color_update_speed;
            if self.colour_timer >= 1.0 {
                self.colour_timer = self.colour_timer.fract();
                let mut pointers = std::mem::take(&mut self.pointers);
                pointers.recolor(|| self.generate_color());
                self.pointers = pointers;
            }
        }
        let moved = self.pointers.take_moved(self.config.splat_force);
        let _ = self.splats(device, &moved);
        if !self.paused && dt > 0.0 {
            let _ = self.step(device, dt);
        }
    }

    /// Set how fast the fluid runs, as a multiple of real time.
    pub fn set_time_scale(&mut self, scale: f32) {
        self.config.time_scale = scale.max(0.0);
    }

    /// One step of the solver: the same eight passes, in the same order.
    pub fn step(&mut self, device: &MetalDevice, dt: f32) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let c = self.config;
        let texel = [1.0 / self.grids.sim.0 as f32, 1.0 / self.grids.sim.1 as f32];
        let cmd = self.command_buffer(device);

        // 1. Curl.
        let plain = Params {
            texel_size: texel,
            ..Default::default()
        };
        let u = plain;
        encode(cmd, &self.pipes.curl, self.grids.curl.id(), |enc| {
            bind(enc, u, &[self.grids.velocity.read.id()]);
        })?;

        // 2. Vorticity confinement.
        let u = Params {
            texel_size: texel,
            dt,
            curl: c.curl,
            ..Default::default()
        };
        encode(
            cmd,
            &self.pipes.vorticity,
            self.grids.velocity.write.id(),
            |enc| {
                bind(enc, u, &[self.grids.velocity.read.id(), self.grids.curl.id()]);
            },
        )?;
        self.grids.velocity.swap();

        // 3. Divergence.
        let u = plain;
        encode(cmd, &self.pipes.divergence, self.grids.divergence.id(), |enc| {
            bind(enc, u, &[self.grids.velocity.read.id()]);
        })?;

        // 4. Start the solve from a fraction of last frame's answer.
        let u = Params {
            value: c.pressure,
            ..Default::default()
        };
        encode(cmd, &self.pipes.clear, self.grids.pressure.write.id(), |enc| {
            bind(enc, u, &[self.grids.pressure.read.id()]);
        })?;
        self.grids.pressure.swap();

        // 5. Solve for the pressure.
        match c.pressure_solver {
            PressureSolver::Jacobi => self.jacobi(cmd, None, c.pressure_iterations, 1.0)?,
            PressureSolver::Multigrid => self.multigrid(cmd)?,
        }

        // 6. Subtract the gradient.
        let u = Params {
            texel_size: texel,
            value: c.gradient_scale,
            ..Default::default()
        };
        encode(cmd, &self.pipes.gradient, self.grids.velocity.write.id(), |enc| {
            bind(enc, u, &[self.grids.pressure.read.id(), self.grids.velocity.read.id()]);
        })?;
        self.grids.velocity.swap();

        // 7. The field carries itself, and 8. it carries the dye. The texel
        // size is the velocity's in both: the lookup is an offset in uv, the
        // same number at any dye size.
        self.advect(cmd, dt, texel, true)?;
        self.advect(cmd, dt, texel, false)?;

        commit(cmd);
        Ok(())
    }

    /// Advect the velocity or the dye, as `record_advection` on the wgpu
    /// side: one pass semi-Lagrangian; for MacCormack the same forward pass
    /// without the fade, then the correction, which fades once.
    fn advect(&mut self, cmd: Id, dt: f32, texel: [f32; 2], velocity_field: bool) -> Result<(), MetalError> {
        let c = self.config;
        let maccormack = c.advection == Advection::MacCormack;
        let dissipation = if velocity_field { c.velocity_dissipation } else { c.density_dissipation };
        let g = &self.grids;
        let (field, scratch, size) = if velocity_field {
            (&g.velocity, g.scratch_velocity.id(), g.sim)
        } else {
            (&g.dye, g.scratch_dye.id(), g.dye_size)
        };
        let (velocity, src, fwd) = (g.velocity.read.id(), field.read.id(), field.write.id());
        let params = Params {
            texel_size: texel,
            dye_texel_size: texel_of(size),
            dt,
            dissipation: if maccormack { 0.0 } else { dissipation },
            ..Default::default()
        };
        let u = params;
        encode(cmd, &self.pipes.advection, fwd, |enc| bind(enc, u, &[velocity, src]))?;
        if maccormack {
            let u = Params { dissipation, ..params };
            encode(cmd, &self.pipes.advect_correct, scratch, |enc| bind(enc, u, &[velocity, fwd, src]))?;
        }
        let g = &mut self.grids;
        match (velocity_field, maccormack) {
            (true, false) => g.velocity.swap(),
            (false, false) => g.dye.swap(),
            (true, true) => std::mem::swap(&mut g.velocity.read, &mut g.scratch_velocity),
            (false, true) => std::mem::swap(&mut g.dye.read, &mut g.scratch_dye),
        }
        Ok(())
    }

    /// `count` Jacobi sweeps relaxed by `weight`, on the simulation's own
    /// pressure (`None`) or a multigrid level's correction, against its
    /// right-hand side.
    fn jacobi(&mut self, cmd: Id, level: Option<usize>, count: u32, weight: f32) -> Result<(), MetalError> {
        let (size, rhs) = match level {
            None => (self.grids.sim, self.grids.divergence.id()),
            Some(l) => (self.grids.mg[l].size, self.grids.mg[l].rhs.id()),
        };
        let u = Params {
            texel_size: texel_of(size),
            value: weight,
            ..Default::default()
        };
        let pair = match level {
            None => &mut self.grids.pressure,
            Some(l) => &mut self.grids.mg[l].pressure,
        };
        for i in 0..count {
            let (src, dst) = if i % 2 == 0 {
                (pair.read.id(), pair.write.id())
            } else {
                (pair.write.id(), pair.read.id())
            };
            encode(cmd, &self.pipes.pressure, dst, |enc| bind(enc, u, &[src, rhs]))?;
        }
        if count % 2 == 1 {
            pair.swap();
        }
        Ok(())
    }

    /// A multigrid V-cycle, as `record_multigrid` on the wgpu side: smooth,
    /// hand the residual a level down as a problem starting from zero, solve
    /// at the bottom, and interpolate the corrections back up, smoothing
    /// again at each level.
    fn multigrid(&mut self, cmd: Id) -> Result<(), MetalError> {
        use crate::fluid::MG_WEIGHT;
        let n = self.grids.mg.len();
        if n == 0 {
            return self.jacobi(cmd, None, self.config.pressure_iterations, 1.0);
        }
        let nu = 2;
        let coarse_sweeps = self.config.pressure_iterations.max(nu * 2);
        for _ in 0..self.config.multigrid_cycles.max(1) {
            self.jacobi(cmd, None, nu, MG_WEIGHT)?;
            for level in 0..n {
                // r = rhs - L(p) at the level above, at its own size…
                let g = &self.grids;
                let (pressure, rhs, fine, residual) = if level == 0 {
                    (g.pressure.read.id(), g.divergence.id(), g.sim, g.residual.id())
                } else {
                    let l = &g.mg[level - 1];
                    (l.pressure.read.id(), l.rhs.id(), l.size, l.residual.id())
                };
                let coarse = (g.mg[level].rhs.id(), g.mg[level].pressure.read.id());
                let u = Params { texel_size: texel_of(fine), ..Default::default() };
                encode(cmd, &self.pipes.residual, residual, |enc| bind(enc, u, &[pressure, rhs]))?;
                // …averaged down onto this one, whose correction starts at zero.
                let u = Params { texel_size: texel_of(fine), ..Default::default() };
                encode(cmd, &self.pipes.restrict, coarse.0, |enc| bind(enc, u, &[residual]))?;
                clear_to_zero(cmd, coarse.1)?;
                let sweeps = if level + 1 == n { coarse_sweeps } else { nu };
                self.jacobi(cmd, Some(level), sweeps, MG_WEIGHT)?;
            }
            for level in (0..n).rev() {
                let correction = self.grids.mg[level].pressure.read.id();
                let u = Params::default();
                let fine = if level == 0 {
                    &mut self.grids.pressure
                } else {
                    &mut self.grids.mg[level - 1].pressure
                };
                let (src, dst) = (fine.read.id(), fine.write.id());
                encode(cmd, &self.pipes.prolong, dst, |enc| bind(enc, u, &[correction, src]))?;
                fine.swap();
                self.jacobi(cmd, level.checked_sub(1), nu, MG_WEIGHT)?;
            }
        }
        Ok(())
    }

    /// Composite the dye into `target`, a texture of `format`.
    pub fn render(
        &mut self,
        device: &MetalDevice,
        target: Id,
        format: NSUInteger,
        size: (u32, u32),
    ) -> Result<(), MetalError> {
        self.composite(device, target, format, size, std::ptr::null_mut())
    }

    /// Composite the dye into a window's drawable — `frame`, from
    /// [`MetalSurface::next_frame`](crate::metal::MetalSurface::next_frame)
    /// — and present it. The surface's sample count does not matter: the
    /// picture goes to the drawable itself.
    pub fn present(&mut self, device: &MetalDevice, frame: &crate::metal::SurfaceFrame) -> Result<(), MetalError> {
        let a = frame.attachments();
        let target = if a.resolve.is_null() { a.color } else { a.resolve };
        self.composite(device, target, a.color_format, (a.width, a.height), a.drawable)
    }

    /// [`render`](Self::render), presenting `drawable` with the same command
    /// buffer when it is not null.
    fn composite(
        &mut self,
        device: &MetalDevice,
        target: Id,
        format: NSUInteger,
        size: (u32, u32),
        drawable: Id,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        self.ensure_output_pipelines(device, format)?;
        let cmd = self.command_buffer(device);

        let c = self.config;
        if c.bloom {
            self.bloom(cmd)?;
        }
        if c.sunrays {
            self.sunrays(cmd)?;
        }
        let encode_srgb = if pixel_format::is_srgb(format) {
            FLAG_ENCODE
        } else {
            0
        };
        let (w, h) = (size.0.max(1), size.1.max(1));

        if !c.transparent {
            let u = Params {
                color: [c.back_color[0], c.back_color[1], c.back_color[2], 1.0],
                flags: encode_srgb,
                ..Default::default()
            };
            let pipeline = self.output(&self.pipes.color, format);
            encode(cmd, pipeline, target, |enc| {
                bind(enc, u, &[self.grids.dye.read.id()]);
            })?;
        }

        let mut flags = encode_srgb;
        if c.shading {
            flags |= FLAG_SHADING;
        }
        if c.bloom {
            flags |= FLAG_BLOOM;
        }
        if c.sunrays {
            flags |= FLAG_SUNRAYS;
        }
        let u = Params {
            texel_size: [1.0 / w as f32, 1.0 / h as f32],
            flags,
            ..Default::default()
        };
        // Loads rather than clears when the background has just been drawn,
        // and composites rather than replaces — the fluid has holes in it, and
        // its alpha where there is no dye is zero. That is the blend the wgpu
        // module gets from one pass with two draws.
        let over = !c.transparent;
        let pipeline = if over {
            self.output(&self.pipes.display_over, format)
        } else {
            self.output(&self.pipes.display, format)
        };
        let textures = [self.grids.dye.read.id(), self.grids.bloom.0.id(), self.grids.sunrays.0.id()];
        encode_over(cmd, pipeline, target, over, |enc| {
            bind(enc, u, &textures);
        })?;

        if c.show_fps {
            // A third pass over the finished picture. The wgpu module gets
            // this as a third draw inside the display pass; here every pass
            // is its own encoder, so it is a pass that loads.
            upload_texture_2d(
                &self.history,
                Frames::LEN as u32,
                1,
                4,
                &self.rate.frames().row(),
            )?;
            let u = Params {
                value: self.rate.fps(),
                radius: self.rate.scale_ms(),
                curl: self.rate.target_ms(),
                ..Default::default()
            };
            let pipeline = self.output(&self.pipes.overlay, format);
            encode_over(cmd, pipeline, target, true, |enc| {
                bind(enc, u, &[self.history.id()]);
            })?;
        }

        if !drawable.is_null() {
            unsafe {
                let _: () = msg1(cmd, sel!("presentDrawable:"), drawable);
            }
        }
        commit(cmd);
        if c.show_fps {
            // The next frame rewrites the history texture from the CPU
            // (`replaceRegion`), which must not land under this one's read.
            unsafe {
                let _: () = msg0(cmd, sel!("waitUntilCompleted"));
            }
        }
        self.rate.tick();
        Ok(())
    }

    /// Threshold the dye, blur it down the halving chain and back up,
    /// adding, and leave the result in the bloom texture — `record_bloom`.
    fn bloom(&mut self, cmd: Id) -> Result<(), MetalError> {
        if self.grids.bloom_chain.len() < 2 {
            return Ok(());
        }
        let c = self.config;
        let knee = c.bloom_threshold * c.bloom_soft_knee + 0.0001;
        let u = Params {
            curve: [c.bloom_threshold - knee, knee * 2.0, 0.25 / knee, c.bloom_threshold],
            ..Default::default()
        };
        let (dye, bloom) = (self.grids.dye.read.id(), self.grids.bloom.0.id());
        encode(cmd, &self.pipes.bloom_prefilter, bloom, |enc| bind(enc, u, &[dye]))?;

        let chain: Vec<(Id, (u32, u32))> = self.grids.bloom_chain.iter().map(|(t, s)| (t.id(), *s)).collect();
        let mut last = (bloom, self.grids.bloom.1);
        for &dest in &chain {
            let u = Params { texel_size: texel_of(last.1), ..Default::default() };
            encode(cmd, &self.pipes.bloom_blur, dest.0, |enc| bind(enc, u, &[last.0]))?;
            last = dest;
        }
        for &base in chain[..chain.len() - 1].iter().rev() {
            let u = Params { texel_size: texel_of(last.1), ..Default::default() };
            encode_over(cmd, &self.pipes.bloom_blur_add, base.0, true, |enc| bind(enc, u, &[last.0]))?;
            last = base;
        }
        let u = Params {
            texel_size: texel_of(last.1),
            value: c.bloom_intensity,
            ..Default::default()
        };
        encode(cmd, &self.pipes.bloom_final, bloom, |enc| bind(enc, u, &[last.0]))
    }

    /// Mask the dye into an occlusion map, march it towards the centre, and
    /// blur the rays once on each axis — `record_sunrays`. The mask goes in
    /// the dye's spare buffer, free at this point in the frame.
    fn sunrays(&mut self, cmd: Id) -> Result<(), MetalError> {
        let g = &self.grids;
        let (dye, mask, rays, temp, size) =
            (g.dye.read.id(), g.dye.write.id(), g.sunrays.0.id(), g.sunrays_temp.id(), g.sunrays.1);
        let u = Params::default();
        encode(cmd, &self.pipes.sunrays_mask, mask, |enc| bind(enc, u, &[dye]))?;
        let u = Params { value: self.config.sunrays_weight, ..Default::default() };
        encode(cmd, &self.pipes.sunrays, rays, |enc| bind(enc, u, &[mask]))?;
        let t = texel_of(size);
        let u = Params { texel_size: [t[0], 0.0], ..Default::default() };
        encode(cmd, &self.pipes.blur, temp, |enc| bind(enc, u, &[rays]))?;
        let u = Params { texel_size: [0.0, t[1]], ..Default::default() };
        encode(cmd, &self.pipes.blur, rays, |enc| bind(enc, u, &[temp]))
    }

    /// The smoothed frame rate the read-out shows.
    pub fn fps(&self) -> f32 {
        self.rate.fps()
    }

    /// Render into a fresh texture and read it back as tightly packed RGBA.
    pub fn capture_rgba(&mut self, device: &MetalDevice) -> Result<Vec<u8>, MetalError> {
        // The blit's command buffer is autoreleased: drain it here, not at the
        // thread's exit.
        let _pool = AutoreleasePool::new();
        let (w, h) = (self.width, self.height);
        let target = new_texture_2d(
            device,
            w,
            h,
            pixel_format::RGBA8_UNORM,
            texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        self.render(device, target.id(), pixel_format::RGBA8_UNORM, (w, h))?;

        let padded = (w as usize * 4).div_ceil(256) * 256;
        let staging = device.new_buffer(&vec![0u8; padded * h as usize])?;
        let cmd = self.command_buffer(device);
        blit_texture_to_buffer_on_cmd(cmd, target.id(), w, h, &staging, padded)?;
        unsafe {
            let _: () = msg0(cmd, sel!("commit"));
        }
        let mut out = Vec::new();
        finish_texture_readback_into(cmd, &staging, w, h, pixel_format::RGBA8_UNORM, &mut out)?;
        Ok(out)
    }

    // -- helpers ----------------------------------------------------------

    fn command_buffer(&self, device: &MetalDevice) -> Id {
        unsafe { msg0(device.queue_id(), sel!("commandBuffer")) }
    }

    fn ensure_output_pipelines(
        &mut self,
        device: &MetalDevice,
        format: NSUInteger,
    ) -> Result<(), MetalError> {
        if !self.pipes.display.iter().any(|(f, _)| *f == format) {
            let p = fullscreen_pipeline_with(device, "fluid_vs", "fluid_fs_display", format)?;
            self.pipes.display.push((format, p));
        }
        if !self.pipes.display_over.iter().any(|(f, _)| *f == format) {
            let p =
                fullscreen_pipeline_blend(device, "fluid_vs", "fluid_fs_display", format, true)?;
            self.pipes.display_over.push((format, p));
        }
        if !self.pipes.color.iter().any(|(f, _)| *f == format) {
            let p = fullscreen_pipeline_with(device, "fluid_vs", "fluid_fs_color", format)?;
            self.pipes.color.push((format, p));
        }
        if self.config.show_fps && !self.pipes.overlay.iter().any(|(f, _)| *f == format) {
            let p =
                fullscreen_pipeline_blend(device, "fluid_vs", "fs_fluid_overlay", format, true)?;
            self.pipes.overlay.push((format, p));
        }
        Ok(())
    }

    fn output<'a>(&'a self, list: &'a [(NSUInteger, Owned)], format: NSUInteger) -> &'a Owned {
        list.iter()
            .find(|(f, _)| *f == format)
            .map(|(_, p)| p)
            .expect("ensure_output_pipelines ran first")
    }
}

/// The shortest side of a grid is `resolution`; the longer follows the
/// canvas's aspect, so the texels stay square.
fn grid(resolution: u32, width: u32, height: u32) -> (u32, u32) {
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    let aspect = if aspect < 1.0 { 1.0 / aspect } else { aspect };
    let min = resolution.max(1);
    let max = ((resolution as f32 * aspect).round() as u32).max(1);
    if width > height {
        (max, min)
    } else {
        (min, max)
    }
}

/// A pass's uniforms and textures. The block is copied into the command
/// buffer as it is encoded (`set…Bytes`), so a submit owns its own and the
/// CPU never has to wait for the GPU before writing the next.
fn bind(enc: Id, params: Params, textures: &[Id]) {
    let bytes = bytemuck::bytes_of(&params);
    unsafe {
        let _: () = msg3(enc, sel!("setFragmentBytes:length:atIndex:"), bytes.as_ptr(), bytes.len(), 0usize);
        // The vertex stage reads the same block: it is where the neighbour
        // taps come from.
        let _: () = msg3(enc, sel!("setVertexBytes:length:atIndex:"), bytes.as_ptr(), bytes.len(), 0usize);
        for (i, t) in textures.iter().enumerate() {
            let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), *t, i);
        }
    }
}

/// A splat pass's batch (`batch(k)` in the shared source), at fragment
/// buffer 1.
fn bind_batch(enc: Id, batch: &[f32]) {
    unsafe {
        let _: () = msg3(enc, sel!("setFragmentBytes:length:atIndex:"), batch.as_ptr(), batch.len() * 4, 1usize);
    }
}

fn encode(cmd: Id, pipeline: &Owned, dst: Id, bind: impl FnOnce(Id)) -> Result<(), MetalError> {
    encode_fullscreen(cmd, pipeline.id(), dst, 0, bind)
}

/// As [`encode`], but loading the target instead of clearing it — for the
/// display pass drawn over a background already in place.
fn encode_over(
    cmd: Id,
    pipeline: &Owned,
    dst: Id,
    load: bool,
    binder: impl FnOnce(Id),
) -> Result<(), MetalError> {
    if !load {
        return encode(cmd, pipeline, dst, binder);
    }
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = super::postfx::render_pass_descriptor(dst);
        let attachments: Id = msg0(desc, sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setLoadAction:"), load_action::LOAD);
        let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
        if enc.is_null() {
            return Err(MetalError::Unsupported(
                "fluid render encoder failed".into(),
            ));
        }
        let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline.id());
        binder(enc);
        let _: () = msg3(
            enc,
            sel!("drawPrimitives:vertexStart:vertexCount:"),
            primitive::TRIANGLE,
            0usize,
            3usize,
        );
        let _: () = msg0(enc, sel!("endEncoding"));
    }
    Ok(())
}

/// A render pass that clears `dst` to transparent black and draws nothing.
///
/// `render_pass_descriptor` clears to opaque black, which is right for a
/// picture and wrong for a field: an alpha of one in the dye is a texel that
/// is fully covered, and in the velocity it is a component the solver reads.
fn clear_to_zero(cmd: Id, dst: Id) -> Result<(), MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = super::postfx::render_pass_descriptor(dst);
        let attachments: Id = msg0(desc, sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(
            color,
            sel!("setClearColor:"),
            MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            },
        );
        let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
        if enc.is_null() {
            return Err(MetalError::Unsupported("fluid clear encoder failed".into()));
        }
        let _: () = msg0(enc, sel!("endEncoding"));
    }
    Ok(())
}

/// Submit without waiting: one queue runs its command buffers in order, and
/// each owns its uniforms (see [`bind`]) and retains its textures.
fn commit(cmd: Id) {
    unsafe {
        let _: () = msg0(cmd, sel!("commit"));
    }
}

#[cfg(all(test, metal_backend, feature = "fluid"))]
mod gpu_tests {
    use super::*;
    use crate::metal::fluid_shader_source;

    /// The device, or `None` on a machine without Metal.
    fn device() -> Option<MetalDevice> {
        match MetalDevice::with_shader_source(&fluid_shader_source()) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping: {e}");
                None
            }
        }
    }

    /// Copy one of the float fields into an 8-bit texture and read it back,
    /// which is the only way a test can see what the solver is holding.
    fn dump(fluid: &mut MetalFluid, device: &MetalDevice, src: Id) -> Vec<u8> {
        let (w, h) = (128, 128);
        let target = new_texture_2d(
            device,
            w,
            h,
            pixel_format::RGBA8_UNORM,
            texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
            storage_mode::PRIVATE,
            1,
            1,
        )
        .unwrap();
        let pipeline = fullscreen_pipeline_with(
            device,
            "fluid_vs",
            "fluid_fs_clear",
            pixel_format::RGBA8_UNORM,
        )
        .unwrap();
        let cmd = fluid.command_buffer(device);
        let u = (Params {
            value: 1.0,
            ..Default::default()
        });
        encode(cmd, &pipeline, target.id(), |enc| bind(enc, u, &[src])).unwrap();
        commit(cmd);

        let padded = (w as usize * 4).div_ceil(256) * 256;
        let staging = device.new_buffer(&vec![0u8; padded * h as usize]).unwrap();
        let cmd = fluid.command_buffer(device);
        blit_texture_to_buffer_on_cmd(cmd, target.id(), w, h, &staging, padded).unwrap();
        unsafe {
            let _: () = msg0(cmd, sel!("commit"));
        }
        let mut out = Vec::new();
        finish_texture_readback_into(cmd, &staging, w, h, pixel_format::RGBA8_UNORM, &mut out)
            .unwrap();
        out
    }

    fn brightest(rgba: &[u8]) -> u8 {
        rgba.chunks_exact(4)
            .map(|p| p[0].max(p[1]).max(p[2]))
            .max()
            .unwrap_or(0)
    }

    /// The MSL carries the row width and the millisecond range as
    /// literals, because `FLUID_SHADER_SOURCE` is a public
    /// constant. This is what stops them drifting from the Rust the
    /// wgpu shaders get them substituted from: change `Frames::LEN` and the
    /// chart on this backend would read the wrong texels, silently, and the
    /// picture would still be a chart.
    #[test]
    fn the_msl_read_out_agrees_about_the_row() {
        let find = |name: &str| -> String {
            let at = super::super::FLUID_SHADER_SOURCE
                .find(name)
                .unwrap_or_else(|| panic!("`{name}` is not declared in fluid.metal"));
            let rest = &super::super::FLUID_SHADER_SOURCE[at + name.len()..];
            let eq = rest.find('=').expect("a constant is assigned");
            rest[eq + 1..]
                .split(';')
                .next()
                .expect("terminated")
                .trim()
                .to_string()
        };
        assert_eq!(
            find("FLUID_HISTORY_COLUMNS"),
            Frames::LEN.to_string(),
            "the MSL chart reads a different number of columns than Rust uploads"
        );
        assert_eq!(
            find("FLUID_HISTORY_SCALE_MS").parse::<f32>().unwrap(),
            Frames::SCALE_MS,
            "the MSL chart decodes a byte as a different number of milliseconds \
             than Rust encodes it as"
        );
    }

    /// The read-out draws, and changes nothing outside its own box.
    ///
    /// The second half is the half that catches things: a `discard` that is
    /// not reached leaves the whole picture tinted, and an overlay drawn
    /// without a load action erases it.
    #[test]
    fn the_metal_read_out_draws_only_where_it_says() {
        let Some(device) = device() else { return };
        let (w, h) = (160u32, 64u32);
        let render = |show: bool| -> Vec<u8> {
            let mut fluid = MetalFluid::new(
                &device,
                w,
                h,
                FluidConfig {
                    show_fps: show,
                    ..FluidConfig::default()
                },
            )
            .unwrap();
            fluid.multiple_splats(&device, 4);
            for _ in 0..4 {
                fluid.update(&device, 1.0 / 60.0);
                // Twice, so there is a rate to report rather than a zero.
                fluid.capture_rgba(&device).unwrap();
            }
            fluid.capture_rgba(&device).unwrap()
        };
        let plain = render(false);
        let with = render(true);

        // Three digits at 3x5 scaled by three, ten in from each edge, and
        // the chart under them — 64 samples two pixels wide, 18 tall,
        // starting at y = 29. The same box the wgpu tests use.
        let in_read_out = |x: u32, y: u32| {
            let digits = x < 10 + 12 * 3 && y < 10 + 5 * 3;
            let chart = (10..10 + 64 * 2).contains(&x) && (29..29 + 18).contains(&y);
            digits || chart
        };
        let (mut inside, mut outside) = (0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if plain[i..i + 4] == with[i..i + 4] {
                    continue;
                }
                if in_read_out(x, y) {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }
        eprintln!("  the read-out changed {inside} pixels inside its box, {outside} outside");
        assert!(inside > 20, "the read-out drew {inside} pixels");
        assert_eq!(outside, 0, "it changed {outside} pixels outside its box");
    }

    /// Both backends draw the same read-out from the same frame times.
    ///
    /// The transcription has two places to go wrong that a picture of a
    /// plausible chart would not show: the glyph table, and the byte that
    /// encodes a millisecond. Pinning the history and the rate on both
    /// sides makes the overlay the only thing in the frame that is not
    /// black, so the comparison is of the read-out and nothing else — which
    /// is also what proves the premultiplied blend here lands in the same
    /// place as the wgpu side's straight alpha.
    #[test]
    fn the_two_read_outs_agree() {
        use crate::gpu::GpuContext;
        let Some(device) = device() else { return };
        let Ok(gpu) = GpuContext::headless_blocking() else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        let (w, h) = (160u32, 64u32);
        let config = FluidConfig {
            show_fps: true,
            bloom: false,
            sunrays: false,
            colorful: false,
            transparent: false,
            back_color: [0.0; 3],
            ..FluidConfig::default()
        };
        // A flat history with a hitch in it, so the comparison covers a
        // clipped bar, a half-height one and the target line.
        let mut times = [1000.0f32 / 60.0; 64];
        times[40] = 1000.0 / 10.0;
        times[41] = 1000.0 / 240.0;

        let mut metal = MetalFluid::new(&device, w, h, config).unwrap();
        metal.rate.pin_frame_times(&times);
        let from_metal = metal.capture_rgba(&device).unwrap();

        let mut sim =
            crate::fluid::FluidSim::new(gpu.device.clone(), gpu.queue.clone(), w, h, config)
                .unwrap();
        sim.set_frame_times_for_test(&times);
        let from_wgpu = sim.capture_rgba(w, h);

        let mut worst = 0i32;
        let mut differing = 0usize;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 4;
                for ch in 0..3 {
                    let d = from_metal[i + ch] as i32 - from_wgpu[i + ch] as i32;
                    worst = worst.max(d.abs());
                    if d != 0 {
                        differing += 1;
                    }
                }
            }
        }
        let lit = from_wgpu
            .chunks_exact(4)
            .filter(|p| p[0] > 10 || p[1] > 10 || p[2] > 10)
            .count();
        eprintln!(
            "  the wgpu read-out lit {lit} pixels; {differing} channels differ, worst {worst}/255"
        );
        assert!(lit > 400, "the reference drew only {lit} pixels");
        // One step, for the rounding of a blend done in a different order.
        assert!(
            worst <= 1,
            "the two read-outs differ by {worst}/255 over {differing} channels"
        );
    }

    #[test]
    fn a_splat_lands_in_the_dye() {
        let Some(device) = device() else { return };
        let mut fluid = MetalFluid::new(&device, 256, 256, FluidConfig::default()).unwrap();
        fluid
            .splat(&device, 0.5, 0.5, 0.0, 0.0, [1.0, 0.5, 0.25])
            .unwrap();
        let src = fluid.grids.dye.read.id();
        let dye = dump(&mut fluid, &device, src);
        assert!(
            brightest(&dye) > 200,
            "a splat at the centre left nothing in the dye (brightest {})",
            brightest(&dye)
        );
    }

    /// A drag paints where it goes, as on the wgpu side: down, along and up
    /// in the top-left quadrant of a paused fluid leaves dye there and
    /// nowhere else, and a hovering pointer paints nothing.
    #[test]
    fn a_drag_paints_under_the_pointer() {
        let Some(device) = device() else { return };
        let mut fluid = MetalFluid::new(&device, 256, 256, FluidConfig::default()).unwrap();
        fluid.set_paused(true);
        fluid.pointer_move(-1, 200.0, 200.0);
        fluid.update(&device, 1.0 / 60.0);
        let src = fluid.grids.dye.read.id();
        assert_eq!(brightest(&dump(&mut fluid, &device, src)), 0, "a hovering pointer painted");

        fluid.pointer_down(-1, 40.0, 40.0);
        for k in 1..=10 {
            fluid.pointer_move(-1, 40.0 + 6.0 * k as f32, 50.0);
            fluid.update(&device, 1.0 / 60.0);
        }
        fluid.pointer_up(-1);
        let src = fluid.grids.dye.read.id();
        let dye = dump(&mut fluid, &device, src);
        let quadrant = |x0: usize, y0: usize| {
            (y0..y0 + 64)
                .flat_map(|y| (x0..x0 + 64).map(move |x| (y, x)))
                .map(|(y, x)| {
                    let p = &dye[(y * 128 + x) * 4..][..3];
                    p[0].max(p[1]).max(p[2])
                })
                .max()
                .unwrap()
        };
        assert!(quadrant(0, 0) > 10, "the drag left no dye under it ({})", quadrant(0, 0));
        assert_eq!(quadrant(64, 64), 0, "dye away from the drag");
    }

    /// A resize and a new dye resolution rebuild the grids at the new size,
    /// keeping the dye.
    #[test]
    fn resize_and_set_config_rebuild_the_grids() {
        let Some(device) = device() else { return };
        let mut fluid = MetalFluid::new(&device, 256, 256, FluidConfig::default()).unwrap();
        fluid.splat(&device, 0.5, 0.5, 0.0, 0.0, [1.0, 0.5, 0.25]).unwrap();
        fluid.resize(&device, 512, 256).unwrap();
        let ((dw, dh), (sw, sh)) = fluid.grid_sizes();
        assert_eq!((dw, dh, sw, sh), (dw.max(dh), dw.min(dh), sw.max(sh), sw.min(sh)), "landscape grids");
        assert_eq!(dw, 2 * dh);
        let c = FluidConfig {
            dye_resolution: 256,
            ..*fluid.config()
        };
        fluid.set_config(&device, c).unwrap();
        assert_eq!(fluid.grid_sizes().0, (512, 256));
        // What was there survives the rebuild, resampled, as on wgpu.
        let src = fluid.grids.dye.read.id();
        assert!(brightest(&dump(&mut fluid, &device, src)) > 100, "the resize wiped the dye");
    }

    #[test]
    fn the_solver_keeps_the_dye() {
        let Some(device) = device() else { return };
        let mut fluid = MetalFluid::new(&device, 256, 256, FluidConfig::default()).unwrap();
        fluid
            .splat(&device, 0.5, 0.5, 0.0, 0.0, [1.0, 0.5, 0.25])
            .unwrap();
        for _ in 0..10 {
            fluid.update(&device, 1.0 / 60.0);
        }
        let src = fluid.grids.dye.read.id();
        let dye = dump(&mut fluid, &device, src);
        assert!(
            brightest(&dye) > 100,
            "ten steps erased the dye (brightest {})",
            brightest(&dye)
        );
    }

    /// The two backends are the same simulation: the solver and the display,
    /// the post passes off (see [`parity`]).
    #[test]
    #[cfg_attr(not(feature = "fluid"), ignore)]
    fn metal_fluid_matches_wgpu() {
        let config = FluidConfig {
            bloom: false,
            sunrays: false,
            colorful: false,
            ..FluidConfig::default()
        };
        if let Some(mean) = parity(config) {
            assert!(mean < 0.05, "the two backends drew different fluids (mean |diff| {mean:.3}/255)");
        }
    }

    /// The rest of the wgpu module, one piece at a time and then all
    /// together: bloom, sunrays, MacCormack advection and the multigrid
    /// solve draw the same fluid on both backends.
    #[test]
    #[cfg_attr(not(feature = "fluid"), ignore)]
    fn bloom_sunrays_maccormack_and_multigrid_match_wgpu() {
        use crate::fluid::{Advection, PressureSolver};
        let base = FluidConfig {
            bloom: false,
            sunrays: false,
            colorful: false,
            ..FluidConfig::default()
        };
        let cases = [
            ("bloom", FluidConfig { bloom: true, ..base }),
            ("sunrays", FluidConfig { sunrays: true, ..base }),
            ("maccormack", FluidConfig { advection: Advection::MacCormack, ..base }),
            ("multigrid", FluidConfig { pressure_solver: PressureSolver::Multigrid, ..base }),
            (
                "everything",
                FluidConfig {
                    bloom: true,
                    sunrays: true,
                    advection: Advection::MacCormack,
                    pressure_solver: PressureSolver::Multigrid,
                    ..base
                },
            ),
        ];
        // Measured: 0.000 between the backends — one shader source — where
        // each piece on one side and off on the other differs by 43 (bloom),
        // 6.0 (sunrays), 1.1 (MacCormack), 4.1 (multigrid). The bound sits
        // well under the smallest of those.
        let mut failures = Vec::new();
        for (name, config) in cases {
            let Some(mean) = parity(config) else { return };
            eprintln!("{name}: mean |diff| {mean:.3}/255");
            if mean >= 0.05 {
                failures.push(format!("{name}: mean |diff| {mean:.2}/255"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Both backends from the same seed, five frames, captured: the mean
    /// difference per channel, out of 255. `None` without both GPUs.
    ///
    /// One shader source and the same batching, so the two come back equal:
    /// 0.000 here, and within one level after ninety frames
    /// (`examples/metal_fluid.rs --compare`). It was 4.79 at ninety while the
    /// MSL was a hand transcription that splatted a burst one blob at a time
    /// — a fluid amplifies any difference in rounding, so a short run is
    /// still what catches a pass that is wrong rather than merely different.
    fn parity(config: FluidConfig) -> Option<f64> {
        use crate::gpu::GpuContext;
        let device = device()?;
        let Ok(gpu) = GpuContext::headless_blocking() else {
            eprintln!("skipping: no wgpu adapter");
            return None;
        };
        let (w, h) = (320, 192);
        let mut metal = MetalFluid::new(&device, w, h, config).unwrap();
        let mut wgpu_sim =
            crate::fluid::FluidSim::new(gpu.device.clone(), gpu.queue.clone(), w, h, config).unwrap();
        assert_eq!(metal.grid_sizes(), wgpu_sim.grid_sizes(), "the two backends sized their grids differently");
        metal.multiple_splats(&device, 8);
        wgpu_sim.multiple_splats(8);
        for _ in 0..5 {
            metal.update(&device, 1.0 / 60.0);
            wgpu_sim.update(1.0 / 60.0);
        }
        let a = metal.capture_rgba(&device).unwrap();
        let b = wgpu_sim.capture_rgba(w, h);
        assert_eq!(a.len(), b.len());
        Some(a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / a.len() as f64)
    }

    /// `transparent: false` means every pixel is opaque, dye or no dye — the
    /// display pass has to blend over the background rather than replace it.
    #[test]
    fn the_background_survives_the_display_pass() {
        let Some(device) = device() else { return };
        let config = FluidConfig {
            back_color: [0.1, 0.2, 0.3],
            transparent: false,
            ..FluidConfig::default()
        };
        let mut fluid = MetalFluid::new(&device, 64, 64, config).unwrap();
        let rgba = fluid.capture_rgba(&device).unwrap();
        assert!(
            rgba.chunks_exact(4).all(|p| p[3] == 255),
            "the picture is not opaque"
        );
        assert!(
            rgba.chunks_exact(4).all(|p| p[0] > 0 && p[2] > p[0]),
            "the background colour is not in the picture"
        );
    }
}
