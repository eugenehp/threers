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
//! The simulation is the same simulation — `fluid.metal` is a transcription of
//! `src/fluid/shader.rs`, function for function and constant for constant,
//! and the two share the uniform block rather than each declaring their own.
//!
//! What holds them together is not the transcription being careful, it is
//! `metal_fluid_matches_wgpu`: the same seed, the same frames, both pictures,
//! compared. A port that drifts has somewhere to be caught.
//!
//! # What this is a subset of
//!
//! The solver, the splats, the display and the frame-rate read-out — curl,
//! vorticity confinement, divergence, the Jacobi sweeps, the projection,
//! both advections, and the dye lit as a height field. **Not** bloom,
//! sunrays or the dither**,** which are the wgpu module's display polish and
//! would be four more pipelines and a mip chain for no new ideas. The
//! backend documents its subsets rather than approximating them, which is
//! the convention the rest of `src/metal` already follows.
//!
//! The read-out is the one piece where the two backends share Rust rather
//! than only a shape: `crate::fluid::read_out::Rate` owns the ring of frame
//! times, the smoothing and the rule for what counts as a frame, and all
//! three solvers call it. Only the glyphs and the chart are transcribed, and
//! `the_two_read_outs_agree` renders the same pinned history through both
//! and compares the pictures — which came back identical to the byte.

use std::mem::size_of;

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg1, msg2, msg3, sel, AutoreleasePool, Id, Owned};
use super::postfx::{encode_fullscreen, fullscreen_pipeline_blend, fullscreen_pipeline_with};
use super::resources::{new_texture_2d, upload_texture_2d};
use super::target::{blit_texture_to_buffer_on_cmd, finish_texture_readback_into};

use crate::fluid::passes::{Params, FLAG_ENCODE, FLAG_SHADING};
use crate::fluid::read_out::{Frames, Rate};
use crate::fluid::FluidConfig;

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
    sim: (u32, u32),
    dye_size: (u32, u32),

    velocity: Pair,
    dye: Pair,
    pressure: Pair,
    divergence: Owned,
    curl: Owned,

    pipes: Pipelines,
    /// One uniform buffer per pass in a submit. `Queue::write_buffer` has no
    /// Metal equivalent that orders against a command buffer, so each pass
    /// gets its own slot and they are all written before the buffer is
    /// committed — the same reason the wgpu side has one buffer a pass.
    params: Vec<Owned>,
    next_param: usize,

    rng: u64,
    colour_timer: f32,
    paused: bool,

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
        let sim = grid(config.sim_resolution, width, height);
        let dye_size = grid(config.dye_resolution, width, height);

        let p = |name: &str| fullscreen_pipeline_with(device, "vs_fluid", name, FORMAT);
        let pipes = Pipelines {
            clear: p("fs_fluid_clear")?,
            splat: p("fs_fluid_splat")?,
            advection: p("fs_fluid_advection")?,
            divergence: p("fs_fluid_divergence")?,
            curl: p("fs_fluid_curl")?,
            vorticity: p("fs_fluid_vorticity")?,
            pressure: p("fs_fluid_pressure")?,
            gradient: p("fs_fluid_gradient")?,
            display: Vec::new(),
            display_over: Vec::new(),
            color: Vec::new(),
            overlay: Vec::new(),
        };

        // Enough slots for the longest submit: a step is eight passes plus the
        // sweeps, which share one.
        let params = (0..16)
            .map(|_| device.new_buffer(&[0u8; size_of::<Params>()]))
            .collect::<Result<Vec<_>, _>>()?;

        let fluid = Self {
            velocity: Pair::new(device, sim.0, sim.1)?,
            dye: Pair::new(device, dye_size.0, dye_size.1)?,
            pressure: Pair::new(device, sim.0, sim.1)?,
            divergence: field(device, sim.0, sim.1)?,
            curl: field(device, sim.0, sim.1)?,
            config,
            width,
            height,
            sim,
            dye_size,
            pipes,
            params,
            next_param: 0,
            rng: config.seed | 1,
            colour_timer: 0.0,
            paused: false,
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
        for t in [
            self.velocity.read.id(),
            self.velocity.write.id(),
            self.dye.read.id(),
            self.dye.write.id(),
            self.pressure.read.id(),
            self.pressure.write.id(),
            self.divergence.id(),
            self.curl.id(),
        ] {
            clear_to_zero(cmd, t)?;
        }
        commit(cmd);
        Ok(())
    }

    /// Settings this backend does not implement, named, for a `config` that
    /// has them set.
    ///
    /// Empty is the answer you want. The point is that the alternative to
    /// naming them is *silence*: ask for a multigrid solve here and you get
    /// a Jacobi one, with no error and a picture that looks like a fluid —
    /// and `metal_fluid_matches_wgpu` would then be comparing two different
    /// algorithms and reporting how far apart they are.
    ///
    /// The bloom and sunrays are deliberate — this is the solver and the
    /// display, and the post chain belongs to `src/metal/postfx.rs`. The
    /// rest are gaps.
    pub fn unsupported(config: &FluidConfig) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !matches!(config.pressure_solver, crate::fluid::PressureSolver::Jacobi) {
            out.push("pressure_solver: only Jacobi is implemented here");
        }
        if !matches!(config.advection, crate::fluid::Advection::SemiLagrangian) {
            out.push("advection: only semi-Lagrangian is implemented here");
        }
        if config.bloom {
            out.push("bloom");
        }
        if config.sunrays {
            out.push("sunrays");
        }
        if config.colorful {
            out.push("colorful: splat hues do not advance on their own here");
        }
        out
    }

    pub fn config(&self) -> &FluidConfig {
        &self.config
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The dye and solver grids, in texels.
    pub fn grid_sizes(&self) -> ((u32, u32), (u32, u32)) {
        (self.dye_size, self.sim)
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    // -- input ------------------------------------------------------------

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
        for _ in 0..count {
            let hue = self.next_f32();
            let c = crate::fluid::hsv_to_rgb_for_backends(hue, 1.0, 1.0);
            let colour = [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5];
            let x = self.next_f32();
            let y = self.next_f32();
            let dx = 1000.0 * (self.next_f32() - 0.5);
            let dy = 1000.0 * (self.next_f32() - 0.5);
            let _ = self.splat(device, x, y, dx, dy, colour);
        }
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
        let _pool = AutoreleasePool::new();
        let aspect = self.width as f32 / self.height as f32;
        let mut radius = self.config.splat_radius / 100.0;
        if aspect > 1.0 {
            radius *= aspect;
        }
        let base = Params {
            point: [x, y],
            aspect_ratio: aspect,
            radius,
            ..Default::default()
        };
        let cmd = self.command_buffer(device);
        self.next_param = 0;

        let u = self.write_params(Params {
            color: [dx, dy, 0.0, 1.0],
            ..base
        });
        encode(cmd, &self.pipes.splat, self.velocity.write.id(), |enc| {
            bind(enc, u, &[self.velocity.read.id()]);
        })?;
        let u = self.write_params(Params {
            color: [color[0], color[1], color[2], 1.0],
            ..base
        });
        encode(cmd, &self.pipes.splat, self.dye.write.id(), |enc| {
            bind(enc, u, &[self.dye.read.id()]);
        })?;
        commit(cmd);
        self.velocity.swap();
        self.dye.swap();
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
        if !self.paused && dt > 0.0 {
            let _ = self.step(device, dt);
        }
        let _ = self.colour_timer;
    }

    /// Set how fast the fluid runs, as a multiple of real time.
    pub fn set_time_scale(&mut self, scale: f32) {
        self.config.time_scale = scale.max(0.0);
    }

    /// One step of the solver: the same eight passes, in the same order.
    pub fn step(&mut self, device: &MetalDevice, dt: f32) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let c = self.config;
        let texel = [1.0 / self.sim.0 as f32, 1.0 / self.sim.1 as f32];
        let cmd = self.command_buffer(device);
        self.next_param = 0;

        // 1. Curl.
        let plain = Params {
            texel_size: texel,
            ..Default::default()
        };
        let u = self.write_params(plain);
        encode(cmd, &self.pipes.curl, self.curl.id(), |enc| {
            bind(enc, u, &[self.velocity.read.id()]);
        })?;

        // 2. Vorticity confinement.
        let u = self.write_params(Params {
            texel_size: texel,
            dt,
            curl: c.curl,
            ..Default::default()
        });
        encode(
            cmd,
            &self.pipes.vorticity,
            self.velocity.write.id(),
            |enc| {
                bind(enc, u, &[self.velocity.read.id(), self.curl.id()]);
            },
        )?;
        self.velocity.swap();

        // 3. Divergence.
        let u = self.write_params(plain);
        encode(cmd, &self.pipes.divergence, self.divergence.id(), |enc| {
            bind(enc, u, &[self.velocity.read.id()]);
        })?;

        // 4. Start the solve from a fraction of last frame's answer.
        let u = self.write_params(Params {
            value: c.pressure,
            ..Default::default()
        });
        encode(cmd, &self.pipes.clear, self.pressure.write.id(), |enc| {
            bind(enc, u, &[self.pressure.read.id()]);
        })?;
        self.pressure.swap();

        // 5. Jacobi sweeps, ping-ponging. `value` is the relaxation weight,
        // which is 1 here as it is for the wgpu module's plain solver.
        let u = self.write_params(Params {
            texel_size: texel,
            value: 1.0,
            ..Default::default()
        });
        for i in 0..c.pressure_iterations {
            let (src, dst) = if i % 2 == 0 {
                (self.pressure.read.id(), self.pressure.write.id())
            } else {
                (self.pressure.write.id(), self.pressure.read.id())
            };
            encode(cmd, &self.pipes.pressure, dst, |enc| {
                bind(enc, u, &[src, self.divergence.id()]);
            })?;
        }
        if c.pressure_iterations % 2 == 1 {
            self.pressure.swap();
        }

        // 6. Subtract the gradient.
        let u = self.write_params(Params {
            texel_size: texel,
            value: c.gradient_scale,
            ..Default::default()
        });
        encode(cmd, &self.pipes.gradient, self.velocity.write.id(), |enc| {
            bind(enc, u, &[self.pressure.read.id(), self.velocity.read.id()]);
        })?;
        self.velocity.swap();

        // 7. The field carries itself.
        let u = self.write_params(Params {
            texel_size: texel,
            dt,
            dissipation: c.velocity_dissipation,
            ..Default::default()
        });
        encode(
            cmd,
            &self.pipes.advection,
            self.velocity.write.id(),
            |enc| {
                bind(enc, u, &[self.velocity.read.id(), self.velocity.read.id()]);
            },
        )?;
        self.velocity.swap();

        // 8. …and it carries the dye. The texel size is still the velocity's:
        // the lookup is an offset in uv, the same number at any dye size.
        let u = self.write_params(Params {
            texel_size: texel,
            dt,
            dissipation: c.density_dissipation,
            ..Default::default()
        });
        encode(cmd, &self.pipes.advection, self.dye.write.id(), |enc| {
            bind(enc, u, &[self.velocity.read.id(), self.dye.read.id()]);
        })?;
        self.dye.swap();

        commit(cmd);
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
        let _pool = AutoreleasePool::new();
        self.ensure_output_pipelines(device, format)?;
        let cmd = self.command_buffer(device);
        self.next_param = 0;

        let c = self.config;
        let encode_srgb = if format == pixel_format::RGBA8_UNORM_SRGB {
            FLAG_ENCODE
        } else {
            0
        };
        let (w, h) = (size.0.max(1), size.1.max(1));

        if !c.transparent {
            let u = self.write_params(Params {
                color: [c.back_color[0], c.back_color[1], c.back_color[2], 1.0],
                flags: encode_srgb,
                ..Default::default()
            });
            let pipeline = self.output(&self.pipes.color, format);
            encode(cmd, pipeline, target, |enc| {
                bind(enc, u, &[self.dye.read.id()]);
            })?;
        }

        let mut flags = encode_srgb;
        if c.shading {
            flags |= FLAG_SHADING;
        }
        let u = self.write_params(Params {
            texel_size: [1.0 / w as f32, 1.0 / h as f32],
            flags,
            ..Default::default()
        });
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
        encode_over(cmd, pipeline, target, over, |enc| {
            bind(enc, u, &[self.dye.read.id()]);
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
            let u = self.write_params(Params {
                value: self.rate.fps(),
                radius: self.rate.scale_ms(),
                curl: self.rate.target_ms(),
                ..Default::default()
            });
            let pipeline = self.output(&self.pipes.overlay, format);
            encode_over(cmd, pipeline, target, true, |enc| {
                bind(enc, u, &[self.history.id()]);
            })?;
        }

        commit(cmd);
        self.rate.tick();
        Ok(())
    }

    /// The smoothed frame rate the read-out shows.
    pub fn fps(&self) -> f32 {
        self.rate.fps()
    }

    /// Render into a fresh texture and read it back as tightly packed RGBA.
    pub fn capture_rgba(&mut self, device: &MetalDevice) -> Result<Vec<u8>, MetalError> {
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

    /// Take the next uniform slot and fill it.
    fn write_params(&mut self, params: Params) -> Id {
        let i = self.next_param.min(self.params.len() - 1);
        self.next_param = i + 1;
        let buffer = self.params[i].id();
        unsafe {
            let contents: *mut u8 = msg0(buffer, sel!("contents"));
            std::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(&params).as_ptr(),
                contents,
                size_of::<Params>(),
            );
        }
        buffer
    }

    fn ensure_output_pipelines(
        &mut self,
        device: &MetalDevice,
        format: NSUInteger,
    ) -> Result<(), MetalError> {
        if !self.pipes.display.iter().any(|(f, _)| *f == format) {
            let p = fullscreen_pipeline_with(device, "vs_fluid", "fs_fluid_display", format)?;
            self.pipes.display.push((format, p));
        }
        if !self.pipes.display_over.iter().any(|(f, _)| *f == format) {
            let p =
                fullscreen_pipeline_blend(device, "vs_fluid", "fs_fluid_display", format, true)?;
            self.pipes.display_over.push((format, p));
        }
        if !self.pipes.color.iter().any(|(f, _)| *f == format) {
            let p = fullscreen_pipeline_with(device, "vs_fluid", "fs_fluid_color", format)?;
            self.pipes.color.push((format, p));
        }
        if self.config.show_fps && !self.pipes.overlay.iter().any(|(f, _)| *f == format) {
            let p =
                fullscreen_pipeline_blend(device, "vs_fluid", "fs_fluid_overlay", format, true)?;
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

fn bind(enc: Id, params: Id, textures: &[Id]) {
    unsafe {
        let _: () = msg3(
            enc,
            sel!("setFragmentBuffer:offset:atIndex:"),
            params,
            0usize,
            0usize,
        );
        // The vertex stage reads the same block: it is where the neighbour
        // taps come from.
        let _: () = msg3(
            enc,
            sel!("setVertexBuffer:offset:atIndex:"),
            params,
            0usize,
            0usize,
        );
        for (i, t) in textures.iter().enumerate() {
            let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), *t, i);
        }
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

fn commit(cmd: Id) {
    unsafe {
        let _: () = msg0(cmd, sel!("commit"));
        let _: () = msg0(cmd, sel!("waitUntilCompleted"));
    }
}

#[cfg(all(
    test,
    feature = "metal",
    feature = "fluid",
    any(target_os = "macos", target_os = "ios")
))]
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
            "vs_fluid",
            "fs_fluid_clear",
            pixel_format::RGBA8_UNORM,
        )
        .unwrap();
        let cmd = fluid.command_buffer(device);
        fluid.next_param = 0;
        let u = fluid.write_params(Params {
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
    /// literals, because `FLUID_SHADER_SOURCE` is public and has to compile
    /// as it stands. This is what stops them drifting from the Rust the
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
        assert!(
            MetalFluid::unsupported(&config).is_empty(),
            "this backend does not implement {:?}",
            MetalFluid::unsupported(&config)
        );
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
        let src = fluid.dye.read.id();
        let dye = dump(&mut fluid, &device, src);
        assert!(
            brightest(&dye) > 200,
            "a splat at the centre left nothing in the dye (brightest {})",
            brightest(&dye)
        );
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
        let src = fluid.dye.read.id();
        let dye = dump(&mut fluid, &device, src);
        assert!(
            brightest(&dye) > 100,
            "ten steps erased the dye (brightest {})",
            brightest(&dye)
        );
    }

    /// The two backends are the same simulation.
    ///
    /// Not bit-for-bit: the half floats round differently and a fluid
    /// amplifies that, so this runs few enough steps that chaos has not had
    /// time to separate them — measured, the mean difference is 0.00/255 at
    /// zero steps, 0.04 at five and 4.79 at ninety. A regression in a pass
    /// shows up here immediately because it is not a divergence, it is an
    /// offset from the first frame.
    #[test]
    #[cfg_attr(not(feature = "fluid"), ignore)]
    fn metal_fluid_matches_wgpu() {
        use crate::gpu::GpuContext;
        let Some(device) = device() else { return };
        let Ok(gpu) = GpuContext::headless_blocking() else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        let (w, h) = (320, 192);
        let config = FluidConfig {
            // Everything this backend does not implement, off — so the
            // comparison is of the solver and the display, which is what
            // the two share. `unsupported` below is what keeps this list
            // honest as either side grows.
            bloom: false,
            sunrays: false,
            colorful: false,
            ..FluidConfig::default()
        };
        let mut metal = MetalFluid::new(&device, w, h, config).unwrap();
        let mut wgpu_sim =
            crate::fluid::FluidSim::new(gpu.device.clone(), gpu.queue.clone(), w, h, config)
                .unwrap();
        assert_eq!(
            metal.grid_sizes(),
            wgpu_sim.grid_sizes(),
            "the two backends sized their grids differently"
        );

        assert!(
            MetalFluid::unsupported(&config).is_empty(),
            "the configuration under comparison uses {:?}, which this backend \
             does not implement — so this would be measuring the distance \
             between two different algorithms",
            MetalFluid::unsupported(&config)
        );

        metal.multiple_splats(&device, 8);
        wgpu_sim.multiple_splats(8);
        for _ in 0..5 {
            metal.update(&device, 1.0 / 60.0);
            wgpu_sim.update(1.0 / 60.0);
        }
        let a = metal.capture_rgba(&device).unwrap();
        let b = wgpu_sim.capture_rgba(w, h);
        assert_eq!(a.len(), b.len());
        let mean = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64;
        assert!(
            mean < 2.0,
            "the two backends drew different fluids (mean |diff| {mean:.2}/255)"
        );
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
