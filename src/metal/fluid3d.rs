//! The 3D fluid on the Metal backend.
//!
//! ```no_run
//! # #[cfg(feature = "metal-fluid")]
//! # fn demo() -> Result<(), threers::metal::MetalError> {
//! use threers::fluid::FluidConfig3d;
//! use threers::metal::{MetalDevice, MetalFluid3d, pixel_format_rgba8};
//! use threers::PerspectiveCamera;
//!
//! let device = MetalDevice::new()?;
//! let mut fluid = MetalFluid3d::new(&device, FluidConfig3d::default())?;
//! fluid.multiple_splats(&device, 6)?;
//! fluid.update(&device, 1.0 / 60.0)?;
//! # let camera = PerspectiveCamera::new(40.0, 1.0, 0.05, 20.0);
//! # let target: threers::metal::objc::Id = std::ptr::null_mut();
//! fluid.render(&device, &camera, target, pixel_format_rgba8(), (512, 512))?;
//! # Ok(()) }
//! ```
//!
//! # One shader source
//!
//! The wgpu module's WGSL is the only copy of the shaders: naga translates it
//! to MSL here, with every pass's resources pinned to fixed slots — the
//! uniform block at buffer 0, the three textures at 0 to 2, and the sampler
//! inlined as a `constexpr` clamp-to-edge linear one, which is what the wgpu
//! side binds. `metal_fluid3d_matches_wgpu` holds the two to one picture.
//!
//! # What differs from the wgpu module
//!
//! The fields are the formats the wgpu module picks on a device that has them
//! all — four channels of half float for vectors and the dye, one for
//! scalars — so the parity is like for like. There is no uniform-buffer pool
//! and no bind-group cache: the block is 368 bytes and goes in with
//! `setFragmentBytes`, which Metal copies at encode time. A step's GPU time
//! comes from the command buffer's own clock rather than timestamp queries.

use std::collections::HashMap;

use super::device::{compile_library, MetalDevice, MetalError};
use super::enums::*;
use super::fluid::{blit_row_to_texture, clear_to_zero, commit};
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg3, nsstring, sel, AutoreleasePool, Id,
    Owned, NIL, YES,
};
use super::pass::new_texture_2d;

use crate::cameras::Camera;
use crate::fluid::read_out::{Frames, Rate};
use crate::fluid::volume_config::{self, Atlas, FluidConfig3d, Vol, MG_COARSEST_3D};
use crate::fluid::{PressureSolver, Rng};

/// Vectors (velocity, curl): there is no signed three-channel float format.
const VEC: NSUInteger = pixel_format::RGBA16_FLOAT;
/// The dye. The wgpu module can pack it into `rg11b10ufloat`; off by default
/// there, and not offered here.
const DYE: NSUInteger = pixel_format::RGBA16_FLOAT;
/// Pressure, divergence, the multigrid levels and the occupancy bricks.
const SCALAR: NSUInteger = pixel_format::R16_FLOAT;
/// Bytes a texel of each, for [`volume_config::memory_bytes`].
const BYTES: [u64; 3] = [8, 8, 2];

/// `MTLPixelFormatRGBA8Unorm`, for callers that do not want the enums.
pub fn pixel_format_rgba8() -> NSUInteger {
    pixel_format::RGBA8_UNORM
}

/// The fullscreen triangle, as `renderer::fullscreen::VS` — repeated here so
/// `metal-fluid` builds without the renderer; `the_vertex_stage_is_the_renderers`
/// holds the two together.
const VS: &str = crate::wgsl!(
    r#"
struct FullscreenVertex {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> FullscreenVertex {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out : FullscreenVertex;
    out.uv = vec2<f32>(x, y);
    out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}
"#
);

/// The volume's shaders in MSL, and the MSL name of each WGSL entry point.
///
/// naga keeps names unless they collide with something in MSL, and says so
/// when it does; asking is cheaper than finding out from a missing function.
pub fn fluid3d_shader_source() -> Result<(String, HashMap<String, String>), MetalError> {
    use naga::back::msl;
    let wgsl = format!(
        "{VS}\n{}\n{}",
        crate::fluid::read_out::glyphs(),
        crate::fluid::volume_shader::SHADER
    );
    let fail =
        |what: &str, e: String| MetalError::ShaderCompilation(format!("fluid3d {what}: {e}"));
    let module =
        naga::front::wgsl::parse_str(&wgsl).map_err(|e| fail("parse", e.emit_to_string(&wgsl)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| fail("validate", format!("{e:?}")))?;

    // One layout for every pass, as the wgpu module's bind group layout.
    let binding = |b: u32| naga::ResourceBinding {
        group: 0,
        binding: b,
    };
    let mut resources = msl::BindingMap::new();
    resources.insert(
        binding(0),
        msl::BindTarget {
            buffer: Some(0),
            ..Default::default()
        },
    );
    resources.insert(
        binding(1),
        msl::BindTarget {
            texture: Some(0),
            ..Default::default()
        },
    );
    resources.insert(
        binding(2),
        msl::BindTarget {
            texture: Some(1),
            ..Default::default()
        },
    );
    resources.insert(
        binding(4),
        msl::BindTarget {
            texture: Some(2),
            ..Default::default()
        },
    );
    resources.insert(
        binding(3),
        msl::BindTarget {
            sampler: Some(msl::BindSamplerTarget::Inline(0)),
            ..Default::default()
        },
    );
    let per_entry_point_map = module
        .entry_points
        .iter()
        .map(|ep| {
            (
                ep.name.clone(),
                msl::EntryPointResources {
                    resources: resources.clone(),
                    ..Default::default()
                },
            )
        })
        .collect();
    let options = msl::Options {
        lang_version: (2, 4),
        per_entry_point_map,
        // `fullscreen::linear_sampler`: linear, clamped on every axis.
        inline_samplers: vec![msl::sampler::InlineSampler {
            coord: msl::sampler::Coord::Normalized,
            address: [msl::sampler::Address::ClampToEdge; 3],
            border_color: msl::sampler::BorderColor::TransparentBlack,
            mag_filter: msl::sampler::Filter::Linear,
            min_filter: msl::sampler::Filter::Linear,
            mip_filter: None,
            lod_clamp: None,
            max_anisotropy: None,
            compare_func: msl::sampler::CompareFunc::Never,
        }],
        fake_missing_bindings: false,
        // What wgpu's Metal backend asks for with runtime checks on (the
        // default): out-of-range indices and texel loads clamp into range.
        // Unchecked, a load past the screen's edge in the resolve's
        // neighbourhood reads whatever is there, and the two backends'
        // pictures part at the border.
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        ..Default::default()
    };
    let (source, translated) =
        msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
            .map_err(|e| fail("translate", format!("{e:?}")))?;
    let mut names = HashMap::new();
    for (ep, name) in module.entry_points.iter().zip(translated.entry_point_names) {
        let name = name.map_err(|e| fail("entry point", format!("{}: {e:?}", ep.name)))?;
        names.insert(ep.name.clone(), name);
    }
    Ok((source, names))
}

/// A field read and written by the same pass, so two copies.
struct Pair {
    read: Owned,
    write: Owned,
}

impl Pair {
    fn new(device: &MetalDevice, atlas: Atlas, format: NSUInteger) -> Result<Self, MetalError> {
        Ok(Self {
            read: field(device, atlas, format)?,
            write: field(device, atlas, format)?,
        })
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }
}

fn field(device: &MetalDevice, atlas: Atlas, format: NSUInteger) -> Result<Owned, MetalError> {
    let (w, h) = atlas.size();
    texture(device, w, h, format)
}

fn texture(device: &MetalDevice, w: u32, h: u32, format: NSUInteger) -> Result<Owned, MetalError> {
    new_texture_2d(
        device,
        w,
        h,
        format,
        texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
        storage_mode::PRIVATE,
        1,
        1,
    )
}

/// One coarse level of the multigrid solver.
struct Level {
    atlas: Atlas,
    pressure: Pair,
    rhs: Owned,
    residual: Owned,
}

/// Every texture whose size follows the grids.
struct Fields {
    velocity: Pair,
    dye: Pair,
    pressure: Pair,
    divergence: Owned,
    curl: Owned,
    residual: Owned,
    mg: Vec<Level>,
    /// The dye's largest value per brick, at half its side and a quarter.
    occupancy: [(Atlas, Owned); 2],
}

impl Fields {
    fn new(
        device: &MetalDevice,
        c: &FluidConfig3d,
        sim: Atlas,
        dye: Atlas,
    ) -> Result<Self, MetalError> {
        // As `mg_levels3`.
        let mut mg = Vec::new();
        let mut n = sim.grid;
        let coarsest = c.multigrid_coarsest.max(MG_COARSEST_3D);
        while n / 2 >= coarsest && mg.len() < 6 {
            n /= 2;
            let atlas = Atlas::new(n);
            mg.push(Level {
                atlas,
                pressure: Pair::new(device, atlas, SCALAR)?,
                rhs: field(device, atlas, SCALAR)?,
                residual: field(device, atlas, SCALAR)?,
            });
        }
        let level = |n: u32| -> Result<(Atlas, Owned), MetalError> {
            let atlas = Atlas::new(n.max(2));
            Ok((atlas, field(device, atlas, SCALAR)?))
        };
        Ok(Self {
            velocity: Pair::new(device, sim, VEC)?,
            dye: Pair::new(device, dye, DYE)?,
            pressure: Pair::new(device, sim, SCALAR)?,
            divergence: field(device, sim, SCALAR)?,
            curl: field(device, sim, VEC)?,
            residual: field(device, sim, SCALAR)?,
            mg,
            occupancy: [level(dye.grid / 2)?, level(dye.grid / 4)?],
        })
    }

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
            self.residual.id(),
            self.occupancy[0].1.id(),
            self.occupancy[1].1.id(),
        ];
        for l in &self.mg {
            out.extend([
                l.pressure.read.id(),
                l.pressure.write.id(),
                l.rhs.id(),
                l.residual.id(),
            ]);
        }
        out
    }
}

/// One pipeline per fragment function and target format.
struct Pipes {
    fill: Owned,
    buoyancy: Owned,
    splat_velocity: Owned,
    splat_dye: Owned,
    advect_velocity: Owned,
    advect_dye: Owned,
    divergence: Owned,
    curl: Owned,
    vorticity: Owned,
    pressure: Owned,
    gradient: Owned,
    maxdown: Owned,
    residual: Owned,
    restrict: Owned,
    prolong: Owned,
}

/// The temporal accumulation's buffers, at the size of the target.
struct Accum {
    size: (u32, u32),
    current: Owned,
    depth: Owned,
    history: Pair,
}

/// A 3D fluid on Metal: the same solver, the same march, and the same
/// configuration as [`FluidSim3d`](crate::fluid::FluidSim3d).
pub struct MetalFluid3d {
    config: FluidConfig3d,
    sim: Atlas,
    dye_atlas: Atlas,
    fields: Fields,
    occupancy_stale: bool,

    library: Owned,
    names: HashMap<String, String>,
    vertex: Owned,
    pipes: Pipes,
    /// The march, the resolve and the read-out, keyed by the caller's format.
    march: Vec<(NSUInteger, Owned)>,
    resolve: Vec<(NSUInteger, Owned)>,
    overlay: Vec<(NSUInteger, Owned)>,
    accum_march: Option<Owned>,
    accum: Option<Accum>,
    view_proj: [f32; 16],
    view_proj_inverse: [f32; 16],
    accum_from: Option<[f32; 16]>,
    accum_frame: u32,

    rate: Rate,
    history: Owned,
    /// The last step's command buffer, kept to read its GPU clock once done.
    last_step: Option<Owned>,
    solver_ms: f32,
    passes: std::cell::Cell<u32>,
    rng: Rng,
    paused: bool,
    /// The frame [`render_over`](Self::render_over) marches into, and its size.
    #[cfg_attr(not(metal_backend), allow(dead_code))]
    over: Option<(Owned, u32, u32)>,
}

/// What a pass writes: one texture, or several at once.
enum Targets<'a> {
    One(Id),
    Many(&'a [Id]),
}

impl MetalFluid3d {
    /// A volume on `device`, which may have been built with any shader
    /// source: the volume compiles a library of its own (about 50 ms).
    pub fn new(device: &MetalDevice, config: FluidConfig3d) -> Result<Self, MetalError> {
        if config.pressure_iterations == 0 {
            return Err(MetalError::Unsupported(
                "fluid3d: pressure_iterations must be at least 1".into(),
            ));
        }
        let _pool = AutoreleasePool::new();
        let (source, names) = fluid3d_shader_source()?;
        let library = unsafe { compile_library(device.device_id(), &source)? };
        let vertex = function(&library, &names, "vs_main")?;
        let max = max_texture_side(device);
        let sim = Atlas::largest_within(config.resolution, max);
        let dye_atlas = Atlas::largest_within(config.dye_resolution, max);

        let p = |fs: &str, format: NSUInteger| -> Result<Owned, MetalError> {
            pipeline(
                device,
                &vertex,
                &function(&library, &names, fs)?,
                &[format],
                false,
            )
        };
        let pipes = Pipes {
            fill: p("fs_fill3", VEC)?,
            buoyancy: p("fs_buoyancy3", VEC)?,
            splat_velocity: p("fs_splat3", VEC)?,
            splat_dye: p("fs_splat3", DYE)?,
            advect_velocity: p("fs_advect3_self", VEC)?,
            advect_dye: p("fs_advect3", DYE)?,
            divergence: p("fs_divergence3", SCALAR)?,
            curl: p("fs_curl3", VEC)?,
            vorticity: p("fs_vorticity3", VEC)?,
            pressure: p("fs_pressure3", SCALAR)?,
            gradient: p("fs_gradient3", VEC)?,
            maxdown: p("fs_maxdown3", SCALAR)?,
            residual: p("fs_residual3", SCALAR)?,
            restrict: p("fs_restrict3", SCALAR)?,
            prolong: p("fs_prolong3", SCALAR)?,
        };
        let fluid = Self {
            fields: Fields::new(device, &config, sim, dye_atlas)?,
            occupancy_stale: true,
            library,
            names,
            vertex,
            pipes,
            march: Vec::new(),
            resolve: Vec::new(),
            overlay: Vec::new(),
            accum_march: None,
            accum: None,
            view_proj: [0.0; 16],
            view_proj_inverse: [0.0; 16],
            accum_from: None,
            accum_frame: 0,
            rate: Rate::default(),
            history: new_texture_2d(
                device,
                Frames::LEN as u32,
                1,
                pixel_format::RGBA8_UNORM,
                texture_usage::SHADER_READ,
                storage_mode::PRIVATE,
                1,
                1,
            )?,
            last_step: None,
            solver_ms: 0.0,
            passes: std::cell::Cell::new(0),
            rng: Rng::new(config.seed),
            paused: false,
            over: None,
            config,
            sim,
            dye_atlas,
        };
        fluid.clear_fields(device)?;
        Ok(fluid)
    }

    /// A fresh `MTLTexture` holds whatever was in that memory, and every
    /// stage reads before it writes — so zero them all, as the 2D fluid does.
    fn clear_fields(&self, device: &MetalDevice) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let cmd = device.command_buffer();
        for t in self.fields.all() {
            clear_to_zero(cmd, t)?;
        }
        commit(cmd);
        Ok(())
    }

    pub fn config(&self) -> &FluidConfig3d {
        &self.config
    }

    /// Replace the configuration. A change of grid throws the volume away, as
    /// on wgpu — see `FluidSim3d::set_resolution`.
    pub fn set_config(
        &mut self,
        device: &MetalDevice,
        config: FluidConfig3d,
    ) -> Result<(), MetalError> {
        let rebuild = config.resolution != self.config.resolution
            || config.dye_resolution != self.config.dye_resolution
            || config.multigrid_coarsest != self.config.multigrid_coarsest;
        self.config = config;
        if rebuild {
            let max = max_texture_side(device);
            self.sim = Atlas::largest_within(config.resolution.max(2), max);
            self.dye_atlas = Atlas::largest_within(config.dye_resolution.max(2), max);
            self.fields = Fields::new(device, &self.config, self.sim, self.dye_atlas)?;
            self.occupancy_stale = true;
            self.clear_fields(device)?;
        }
        Ok(())
    }

    /// The solver and dye grids, in voxels a side.
    pub fn grids(&self) -> (u32, u32) {
        (self.sim.grid, self.dye_atlas.grid)
    }

    /// What the fields occupy — [`memory_for`](Self::memory_for) at the
    /// grids that came out.
    pub fn memory_bytes(&self) -> u64 {
        Self::memory_for(self.sim.grid, self.dye_atlas.grid)
    }

    /// What a volume of this size will allocate, before allocating it.
    pub fn memory_for(resolution: u32, dye_resolution: u32) -> u64 {
        volume_config::memory_bytes(resolution, dye_resolution, BYTES)
    }

    /// The largest `(solver, dye)` within `budget` bytes, `ratio` dye voxels to
    /// a solver voxel a side — for a phone, where asking for too much is the
    /// watchdog rather than a slow frame.
    pub fn within_memory(budget: u64, ratio: u32) -> (u32, u32) {
        volume_config::within_bytes(budget, ratio, BYTES)
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Render passes the last step recorded.
    pub fn passes_per_step(&self) -> u32 {
        self.passes.get()
    }

    /// Milliseconds the GPU spent on the last finished step, from the command
    /// buffer's own clock; zero until one has finished.
    pub fn solver_ms(&mut self) -> f32 {
        if let Some(cmd) = &self.last_step {
            // MTLCommandBufferStatusCompleted.
            let status: NSUInteger = unsafe { msg0(cmd.id(), sel!("status")) };
            if status == 4 {
                let start: f64 = unsafe { msg0(cmd.id(), sel!("GPUStartTime")) };
                let end: f64 = unsafe { msg0(cmd.id(), sel!("GPUEndTime")) };
                self.solver_ms = ((end - start) * 1000.0).max(0.0) as f32;
                self.last_step = None;
            }
        }
        self.solver_ms
    }

    /// The smoothed rate `render` is being called at.
    pub fn fps(&self) -> f32 {
        self.rate.fps()
    }

    /// The dye volume (`rgba16Float`), as the atlas [`dye_atlas`](Self::dye_atlas)
    /// describes. Borrowed; a different texture after every step or splat.
    pub fn dye_texture(&self) -> Id {
        self.fields.dye.read.id()
    }

    pub fn dye_atlas(&self) -> Atlas {
        self.dye_atlas
    }

    // -- input ------------------------------------------------------------

    /// A ball of dye and a push, at `at` in the unit cube.
    pub fn splat(
        &mut self,
        device: &MetalDevice,
        at: [f32; 3],
        velocity: [f32; 3],
        color: [f32; 3],
    ) -> Result<(), MetalError> {
        self.splat_batch(device, &[(at, velocity, color)])
    }

    /// A push into the volume where a point on the screen looks into it; as
    /// `FluidSim3d::impulse_at_screen`.
    pub fn impulse_at_screen(
        &mut self,
        device: &MetalDevice,
        camera: &dyn Camera,
        uv: [f32; 2],
        drag: [f32; 2],
        strength: f32,
        colour: [f32; 3],
    ) -> Result<(), MetalError> {
        match volume_config::impulse(&self.config, self.sim.grid, camera, uv, drag, strength) {
            Some((at, velocity)) => self.splat(device, at, velocity, colour),
            None => Ok(()),
        }
    }

    /// `count` puffs at random, from the same sequence as the wgpu module.
    pub fn multiple_splats(&mut self, device: &MetalDevice, count: u32) -> Result<(), MetalError> {
        let batch = volume_config::random_splats(&mut self.rng, count);
        self.splat_batch(device, &batch)
    }

    /// Several splats — `(at, velocity, colour)` each, as [`splat`](Self::splat)
    /// — in one command buffer: emitters that feed the volume every frame.
    pub fn splat_batch(
        &mut self,
        device: &MetalDevice,
        splats: &[([f32; 3], [f32; 3], [f32; 3])],
    ) -> Result<(), MetalError> {
        if splats.is_empty() {
            return Ok(());
        }
        let _pool = AutoreleasePool::new();
        let radius = self.config.splat_radius / 100.0;
        let cmd = device.command_buffer();
        for (at, vel, color) in splats {
            for dye in [false, true] {
                let (atlas, value) = if dye {
                    (self.dye_atlas, *color)
                } else {
                    (self.sim, *vel)
                };
                let u = self.params(
                    atlas,
                    Vol {
                        point: [at[0], at[1], at[2], radius],
                        color: [value[0], value[1], value[2], 1.0],
                        ..Default::default()
                    },
                );
                let (pipe, field) = if dye {
                    (&self.pipes.splat_dye, &self.fields.dye)
                } else {
                    (&self.pipes.splat_velocity, &self.fields.velocity)
                };
                encode(
                    cmd,
                    pipe,
                    Targets::One(field.write.id()),
                    false,
                    &u,
                    [field.read.id(); 3],
                )?;
            }
            self.fields.velocity.swap();
            self.fields.dye.swap();
        }
        commit(cmd);
        self.occupancy_stale = true;
        Ok(())
    }

    /// Set the whole velocity field to one vector, in voxels per second.
    pub fn set_uniform_velocity(
        &mut self,
        device: &MetalDevice,
        v: [f32; 3],
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let cmd = device.command_buffer();
        let u = self.params(
            self.sim,
            Vol {
                color: [v[0], v[1], v[2], 0.0],
                ..Default::default()
            },
        );
        let f = &self.fields.velocity;
        encode(
            cmd,
            &self.pipes.fill,
            Targets::One(f.write.id()),
            false,
            &u,
            [f.read.id(); 3],
        )?;
        commit(cmd);
        self.fields.velocity.swap();
        Ok(())
    }

    // -- the frame --------------------------------------------------------

    /// A frame: the step scaled by `time_scale`, unless paused.
    pub fn update(&mut self, device: &MetalDevice, dt: f32) -> Result<(), MetalError> {
        let dt = dt * self.config.time_scale.max(0.0);
        if !self.paused && dt > 0.0 {
            self.step(device, dt)?;
        }
        Ok(())
    }

    /// One step of the solver, committed on the device's queue.
    pub fn step(&mut self, device: &MetalDevice, dt: f32) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let cmd = device.command_buffer();
        self.step_on(cmd, dt)?;
        commit(cmd);
        self.last_step = unsafe { Owned::retain(cmd) };
        Ok(())
    }

    /// [`step`](Self::step), encoded into `cmd` for the caller to commit.
    pub fn step_on(&mut self, cmd: Id, dt: f32) -> Result<(), MetalError> {
        let c = self.config;
        self.passes.set(0);
        let sim = self.sim;

        // Buoyancy: dye is hot, hot fluid rises.
        let axis = {
            let a = c.buoyancy_axis;
            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if len > 1e-6 {
                [a[0] / len, a[1] / len, a[2] / len]
            } else {
                [0.0; 3]
            }
        };
        if c.buoyancy != 0.0 && axis != [0.0; 3] {
            let u = self.params_mixed(
                sim,
                sim,
                self.dye_atlas,
                Vol {
                    texel: [0.0, 0.0, 0.0, dt],
                    params: [0.0, c.buoyancy, 0.0, 0.0],
                    color: [axis[0], axis[1], axis[2], 0.0],
                    ..Default::default()
                },
            );
            let f = &self.fields;
            self.pass(
                cmd,
                &self.pipes.buoyancy,
                f.velocity.write.id(),
                &u,
                [f.velocity.read.id(), f.dye.read.id()],
            )?;
            self.fields.velocity.swap();
        }

        // 1. Curl, and 2. confinement from it.
        if c.curl != 0.0 {
            let u = self.params(sim, Vol::default());
            let f = &self.fields;
            self.pass(
                cmd,
                &self.pipes.curl,
                f.curl.id(),
                &u,
                [f.velocity.read.id(); 2],
            )?;
            let u = self.params(
                sim,
                Vol {
                    texel: [0.0, 0.0, 0.0, dt],
                    params: [0.0, c.curl, 0.0, 0.0],
                    ..Default::default()
                },
            );
            self.pass(
                cmd,
                &self.pipes.vorticity,
                f.velocity.write.id(),
                &u,
                [f.velocity.read.id(), f.curl.id()],
            )?;
            self.fields.velocity.swap();
        }

        // 3. Divergence.
        let u = self.params(sim, Vol::default());
        let f = &self.fields;
        self.pass(
            cmd,
            &self.pipes.divergence,
            f.divergence.id(),
            &u,
            [f.velocity.read.id(); 2],
        )?;

        // 4. The solve.
        match c.pressure_solver {
            PressureSolver::Jacobi => self.jacobi(cmd, None, c.pressure_iterations, c.pressure)?,
            PressureSolver::Multigrid => self.multigrid(cmd)?,
        }

        // 6. Subtract the gradient.
        let u = self.params(
            sim,
            Vol {
                params: [0.0, 0.0, c.gradient_scale, 0.0],
                ..Default::default()
            },
        );
        let f = &self.fields;
        self.pass(
            cmd,
            &self.pipes.gradient,
            f.velocity.write.id(),
            &u,
            [f.pressure.read.id(), f.velocity.read.id()],
        )?;
        self.fields.velocity.swap();

        // 7. The field carries itself.
        let u = self.params(
            sim,
            Vol {
                texel: [0.0, 0.0, 0.0, dt],
                params: [c.velocity_dissipation, 0.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        let f = &self.fields;
        self.pass(
            cmd,
            &self.pipes.advect_velocity,
            f.velocity.write.id(),
            &u,
            [f.velocity.read.id(); 2],
        )?;
        self.fields.velocity.swap();

        // 8. …and it carries the dye, in the solver's voxels.
        let u = self.params_mixed(
            self.dye_atlas,
            sim,
            self.dye_atlas,
            Vol {
                texel: [0.0, 0.0, 0.0, dt],
                params: [c.density_dissipation, 0.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        let f = &self.fields;
        self.pass(
            cmd,
            &self.pipes.advect_dye,
            f.dye.write.id(),
            &u,
            [f.velocity.read.id(), f.dye.read.id()],
        )?;
        self.fields.dye.swap();
        self.occupancy_stale = true;
        Ok(())
    }

    /// `count` Jacobi sweeps on the top-level pressure (`None`) or a coarse
    /// level's correction, the first reading what it starts from scaled by
    /// `first_scale` — see `fs_pressure3`.
    fn jacobi(
        &mut self,
        cmd: Id,
        level: Option<usize>,
        count: u32,
        first_scale: f32,
    ) -> Result<(), MetalError> {
        if count == 0 {
            return Ok(());
        }
        let atlas = level.map_or(self.sim, |l| self.fields.mg[l].atlas);
        let scaled = self.params(
            atlas,
            Vol {
                params: [0.0, 0.0, first_scale, 0.0],
                ..Default::default()
            },
        );
        let plain = self.params(
            atlas,
            Vol {
                params: [0.0, 0.0, 1.0, 0.0],
                ..Default::default()
            },
        );
        let (pressure, rhs) = match level {
            None => (&self.fields.pressure, self.fields.divergence.id()),
            Some(l) => (&self.fields.mg[l].pressure, self.fields.mg[l].rhs.id()),
        };
        let (a, b) = (pressure.read.id(), pressure.write.id());
        for i in 0..count {
            let even = i % 2 == 0;
            let u = if i == 0 { &scaled } else { &plain };
            let (src, dst) = if even { (a, b) } else { (b, a) };
            encode(
                cmd,
                &self.pipes.pressure,
                Targets::One(dst),
                false,
                u,
                [src, rhs, src],
            )?;
        }
        self.passes.set(self.passes.get() + count);
        if count % 2 == 1 {
            match level {
                None => self.fields.pressure.swap(),
                Some(l) => self.fields.mg[l].pressure.swap(),
            }
        }
        Ok(())
    }

    /// A V-cycle, as `FluidSim3d::record_multigrid`.
    fn multigrid(&mut self, cmd: Id) -> Result<(), MetalError> {
        let c = self.config;
        if self.fields.mg.is_empty() {
            return self.jacobi(cmd, None, c.pressure_iterations, c.pressure);
        }
        let nu = c.multigrid_smooth.max(1);
        let coarsest = self.fields.mg.last().map(|l| l.atlas.grid).unwrap_or(4);
        let coarse_sweeps = (2 * coarsest).clamp(nu * 2, c.pressure_iterations.max(8));
        let levels = self.fields.mg.len();
        for cycle in 0..c.multigrid_cycles.max(1) {
            let scale = if cycle == 0 { c.pressure } else { 1.0 };
            self.jacobi(cmd, None, nu, scale)?;
            for level in 0..levels {
                self.residual(cmd, level)?;
                self.restrict(cmd, level)?;
                let sweeps = if level + 1 == levels {
                    coarse_sweeps
                } else {
                    nu
                };
                self.jacobi(cmd, Some(level), sweeps, 0.0)?;
            }
            for level in (0..levels).rev() {
                self.prolong(cmd, level)?;
                if level == 0 {
                    self.jacobi(cmd, None, nu, 1.0)?;
                } else {
                    self.jacobi(cmd, Some(level - 1), nu, 1.0)?;
                }
            }
        }
        Ok(())
    }

    /// What the level above `level` fails to satisfy, at its own resolution.
    fn residual(&mut self, cmd: Id, level: usize) -> Result<(), MetalError> {
        let f = &self.fields;
        let (atlas, pressure, rhs, target) = if level == 0 {
            (
                self.sim,
                f.pressure.read.id(),
                f.divergence.id(),
                f.residual.id(),
            )
        } else {
            let l = &f.mg[level - 1];
            (l.atlas, l.pressure.read.id(), l.rhs.id(), l.residual.id())
        };
        let u = self.params(atlas, Vol::default());
        self.pass(cmd, &self.pipes.residual, target, &u, [pressure, rhs])
    }

    /// Average `level`'s residual onto the half-size grid below it.
    fn restrict(&mut self, cmd: Id, level: usize) -> Result<(), MetalError> {
        let f = &self.fields;
        let (fine, residual) = if level == 0 {
            (self.sim, f.residual.id())
        } else {
            let l = &f.mg[level - 1];
            (l.atlas, l.residual.id())
        };
        let u = self.params_mixed(f.mg[level].atlas, fine, fine, Vol::default());
        self.pass(
            cmd,
            &self.pipes.restrict,
            f.mg[level].rhs.id(),
            &u,
            [residual; 2],
        )
    }

    /// Interpolate `level`'s correction into the level above and add it.
    fn prolong(&mut self, cmd: Id, level: usize) -> Result<(), MetalError> {
        let f = &self.fields;
        let coarse = f.mg[level].atlas;
        let (fine, read, write) = if level == 0 {
            (self.sim, f.pressure.read.id(), f.pressure.write.id())
        } else {
            let l = &f.mg[level - 1];
            (l.atlas, l.pressure.read.id(), l.pressure.write.id())
        };
        let u = self.params_mixed(fine, coarse, fine, Vol::default());
        self.pass(
            cmd,
            &self.pipes.prolong,
            write,
            &u,
            [f.mg[level].pressure.read.id(), read],
        )?;
        if level == 0 {
            self.fields.pressure.swap();
        } else {
            self.fields.mg[level - 1].pressure.swap();
        }
        Ok(())
    }

    /// The dye's largest value per brick, twice, for the march to skip by.
    fn occupancy(&mut self, cmd: Id) -> Result<(), MetalError> {
        let mut source = (self.dye_atlas, self.fields.dye.read.id());
        for level in 0..2 {
            let (atlas, target) = (
                self.fields.occupancy[level].0,
                self.fields.occupancy[level].1.id(),
            );
            let u = self.params_mixed(atlas, source.0, source.0, Vol::default());
            encode(
                cmd,
                &self.pipes.maxdown,
                Targets::One(target),
                false,
                &u,
                [source.1; 3],
            )?;
            source = (atlas, target);
        }
        Ok(())
    }

    /// March the volume from `camera` into `target` (an `MTLTexture` of
    /// `format`, `size` pixels) and commit.
    pub fn render(
        &mut self,
        device: &MetalDevice,
        camera: &dyn Camera,
        target: Id,
        format: NSUInteger,
        size: (u32, u32),
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let cmd = device.command_buffer();
        self.render_on(device, cmd, camera, target, format, size)?;
        commit(cmd);
        Ok(())
    }

    /// [`render`](Self::render), encoded into `cmd` — a drawable's frame, say,
    /// with the present on the same buffer.
    pub fn render_on(
        &mut self,
        device: &MetalDevice,
        cmd: Id,
        camera: &dyn Camera,
        target: Id,
        format: NSUInteger,
        size: (u32, u32),
    ) -> Result<(), MetalError> {
        let c = self.config;
        let accumulating = self.prepare_accum(device, size)?;
        let view_proj = camera.projection_matrix().multiply(&camera.view_matrix());
        let inv = view_proj.invert();
        self.view_proj = view_proj.elements;
        self.view_proj_inverse = inv.elements;
        let pos = camera.position();
        let skip_floor = if c.empty_space_skip { 0.0008 } else { 0.0 };
        let u = self.params_mixed(
            self.dye_atlas,
            self.dye_atlas,
            self.fields.occupancy[1].0,
            Vol {
                params: [skip_floor, self.accum_frame as f32, 0.0, 0.0],
                bounds_min: [c.bounds_min[0], c.bounds_min[1], c.bounds_min[2], 0.0],
                bounds_max: [c.bounds_max[0], c.bounds_max[1], c.bounds_max[2], 0.0],
                march: [
                    c.march_steps.max(1) as f32,
                    c.absorption,
                    c.emission,
                    c.exposure,
                ],
                background: [
                    c.background[0],
                    c.background[1],
                    c.background[2],
                    if c.transparent { 1.0 } else { 0.0 },
                ],
                cam_pos: [pos.x, pos.y, pos.z, 1.0],
                inv_view_proj: inv.elements,
                prev_view_proj: self.accum_from.unwrap_or(view_proj.elements),
                ..Default::default()
            },
        );
        if c.empty_space_skip && self.occupancy_stale {
            self.occupancy(cmd)?;
            self.occupancy_stale = false;
        }
        let textures = [
            self.fields.dye.read.id(),
            self.fields.occupancy[1].1.id(),
            self.fields.dye.read.id(),
        ];
        if accumulating {
            self.accumulated_march(device, cmd, target, format, &u, textures)?;
            self.accum_frame += 1;
            self.accum_from = Some(view_proj.elements);
        } else {
            let pipe = self.output(device, format, Output::March)?;
            encode(cmd, &pipe, Targets::One(target), false, &u, textures)?;
        }
        if c.show_fps {
            self.read_out(device, cmd, target, format)?;
        }
        self.rate.tick();
        Ok(())
    }

    /// Make sure the accumulation's buffers match `size`; false when it is off.
    fn prepare_accum(
        &mut self,
        device: &MetalDevice,
        size: (u32, u32),
    ) -> Result<bool, MetalError> {
        if self.config.temporal_samples <= 1 {
            self.accum = None;
            self.accum_from = None;
            return Ok(false);
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        if self.accum.as_ref().is_none_or(|a| a.size != (w, h)) {
            let make = |f| texture(device, w, h, f);
            let history = Pair {
                read: make(VEC)?,
                write: make(VEC)?,
            };
            let cmd = device.command_buffer();
            clear_to_zero(cmd, history.read.id())?;
            clear_to_zero(cmd, history.write.id())?;
            commit(cmd);
            self.accum = Some(Accum {
                size: (w, h),
                current: make(VEC)?,
                depth: make(SCALAR)?,
                history,
            });
            self.accum_from = None;
            self.accum_frame = 0;
        }
        Ok(true)
    }

    /// March into this frame's buffer and its depth, then resolve against
    /// the history into the history and the target at once.
    fn accumulated_march(
        &mut self,
        device: &MetalDevice,
        cmd: Id,
        target: Id,
        format: NSUInteger,
        u: &Vol,
        textures: [Id; 3],
    ) -> Result<(), MetalError> {
        if self.accum_march.is_none() {
            let fs = function(&self.library, &self.names, "fs_raymarch_depth")?;
            self.accum_march = Some(pipeline(device, &self.vertex, &fs, &[VEC, SCALAR], false)?);
        }
        let weight = if self.accum_frame == 0 {
            1.0
        } else {
            1.0 / self.config.temporal_samples.max(2) as f32
        };
        let resolve = self.output(device, format, Output::Resolve)?;
        let r = Vol {
            params: [0.0, 0.0, weight, self.config.temporal_clip],
            inv_view_proj: self.view_proj_inverse,
            prev_view_proj: self.accum_from.unwrap_or(self.view_proj),
            ..self.params(self.dye_atlas, Vol::default())
        };
        let a = self.accum.as_mut().expect("prepare_accum made one");
        let march = self.accum_march.as_ref().expect("just built");
        encode(
            cmd,
            march,
            Targets::Many(&[a.current.id(), a.depth.id()]),
            false,
            u,
            textures,
        )?;
        encode(
            cmd,
            &resolve,
            Targets::Many(&[a.history.write.id(), target]),
            false,
            &r,
            [a.current.id(), a.history.read.id(), a.depth.id()],
        )?;
        a.history.swap();
        Ok(())
    }

    /// The frame-rate read-out over the finished picture.
    fn read_out(
        &mut self,
        device: &MetalDevice,
        cmd: Id,
        target: Id,
        format: NSUInteger,
    ) -> Result<(), MetalError> {
        let row = device.new_buffer(&self.rate.frames().row())?;
        blit_row_to_texture(cmd, &row, &self.history, Frames::LEN as u32)?;
        let u = self.params(
            self.dye_atlas,
            Vol {
                params: [
                    self.rate.fps(),
                    self.rate.scale_ms(),
                    self.rate.target_ms(),
                    0.0,
                ],
                ..Default::default()
            },
        );
        let pipe = self.output(device, format, Output::Overlay)?;
        encode(
            cmd,
            &pipe,
            Targets::One(target),
            true,
            &u,
            [self.history.id(); 3],
        )
    }

    /// The pipelines whose target is the caller's, compiled on first sight
    /// of a format.
    fn output(
        &mut self,
        device: &MetalDevice,
        format: NSUInteger,
        which: Output,
    ) -> Result<Owned, MetalError> {
        let cache = match which {
            Output::March => &mut self.march,
            Output::Resolve => &mut self.resolve,
            Output::Overlay => &mut self.overlay,
        };
        if let Some((_, p)) = cache.iter().find(|(f, _)| *f == format) {
            return Ok(unsafe { Owned::retain(p.id()) }.expect("a live pipeline"));
        }
        let (fs, formats, blend): (&str, Vec<NSUInteger>, bool) = match which {
            Output::March => ("fs_raymarch", vec![format], false),
            Output::Resolve => ("fs_resolve3", vec![VEC, format], false),
            Output::Overlay => ("fs_overlay3", vec![format], true),
        };
        let p = pipeline(
            device,
            &self.vertex,
            &function(&self.library, &self.names, fs)?,
            &formats,
            blend,
        )?;
        let out = unsafe { Owned::retain(p.id()) }.expect("a live pipeline");
        cache.push((format, p));
        Ok(out)
    }

    /// March the volume over a frame the scene renderer drew: into a
    /// premultiplied frame of its own, blended over `pass` by `renderer`,
    /// which then presents the drawable if `pass` has one. Render the scene
    /// with [`without_drawable`](super::PassAttachments::without_drawable)
    /// first. Wants [`FluidConfig3d::transparent`]; without it the volume's
    /// background covers the scene. The volume is not depth-tested against
    /// the scene: it is drawn over whatever is there.
    #[cfg(metal_backend)]
    pub fn render_over(
        &mut self,
        renderer: &mut super::MetalRenderer,
        camera: &dyn Camera,
        pass: &super::PassAttachments,
    ) -> Result<(), MetalError> {
        let (w, h) = pass.size();
        let (w, h) = (w.max(1), h.max(1));
        if !matches!(&self.over, Some((_, ow, oh)) if (*ow, *oh) == (w, h)) {
            self.over = Some((texture(renderer.device(), w, h, pixel_format::RGBA16_FLOAT)?, w, h));
        }
        let target = self.over.as_ref().map(|(t, _, _)| t.id()).expect("made above");
        let device = renderer.device().clone();
        self.render(&device, camera, target, pixel_format::RGBA16_FLOAT, (w, h))?;
        renderer.blend_texture(target, pass)
    }

    /// Render into a fresh RGBA8 texture and read it back, tightly packed.
    #[cfg(metal_backend)]
    pub fn capture_rgba(
        &mut self,
        device: &MetalDevice,
        camera: &dyn Camera,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, MetalError> {
        use super::target::{blit_texture_to_buffer_on_cmd, finish_texture_readback_into};
        let _pool = AutoreleasePool::new();
        let (w, h) = (width.max(1), height.max(1));
        let target = texture(device, w, h, pixel_format::RGBA8_UNORM)?;
        self.render(
            device,
            camera,
            target.id(),
            pixel_format::RGBA8_UNORM,
            (w, h),
        )?;
        let padded = (w as usize * 4).div_ceil(256) * 256;
        let staging = device.new_buffer(&vec![0u8; padded * h as usize])?;
        let cmd = device.command_buffer();
        blit_texture_to_buffer_on_cmd(cmd, target.id(), w, h, &staging, padded)?;
        commit(cmd);
        let mut out = Vec::new();
        finish_texture_readback_into(cmd, &staging, w, h, pixel_format::RGBA8_UNORM, &mut out)?;
        Ok(out)
    }

    // -- helpers ----------------------------------------------------------

    fn params(&self, atlas: Atlas, rest: Vol) -> Vol {
        self.params_mixed(atlas, atlas, atlas, rest)
    }

    /// Each texture's own grid, separately: sampling one field with another's
    /// tiling is a quiet bug that still draws a plausible plume.
    fn params_mixed(&self, out: Atlas, a: Atlas, b: Atlas, rest: Vol) -> Vol {
        Vol {
            out_grid: out.into(),
            a_grid: a.into(),
            b_grid: b.into(),
            ..rest
        }
    }

    /// A pass of the step, counted.
    fn pass(
        &self,
        cmd: Id,
        pipe: &Owned,
        target: Id,
        u: &Vol,
        textures: [Id; 2],
    ) -> Result<(), MetalError> {
        self.passes.set(self.passes.get() + 1);
        encode(
            cmd,
            pipe,
            Targets::One(target),
            false,
            u,
            [textures[0], textures[1], textures[0]],
        )
    }
}

#[derive(Clone, Copy)]
enum Output {
    March,
    Resolve,
    Overlay,
}

fn max_texture_side(device: &MetalDevice) -> u32 {
    // Every Apple GPU family Metal 2.4 runs on takes 16384; the A-series
    // before the A9 took 8192 and cannot run this anyway.
    let _ = device;
    16384
}

fn function(
    library: &Owned,
    names: &HashMap<String, String>,
    wgsl: &str,
) -> Result<Owned, MetalError> {
    let name = names.get(wgsl).map(String::as_str).unwrap_or(wgsl);
    let _pool = AutoreleasePool::new();
    unsafe {
        let f: Id = msg1(library.id(), sel!("newFunctionWithName:"), nsstring(name));
        Owned::from_retained(f).ok_or_else(|| MetalError::MissingFunction(name.into()))
    }
}

/// A fullscreen pipeline writing `formats`, one attachment each; `blend` is
/// wgpu's `ALPHA_BLENDING` on the first, for the read-out.
fn pipeline(
    device: &MetalDevice,
    vs: &Owned,
    fs: &Owned,
    formats: &[NSUInteger],
    blend: bool,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
        let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
        let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
        for (i, format) in formats.iter().enumerate() {
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), i);
            let _: () = msg1(color, sel!("setPixelFormat:"), *format);
            if blend && i == 0 {
                let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
                let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
                let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
                let _: () = msg1(
                    color,
                    sel!("setSourceRGBBlendFactor:"),
                    blend_factor::SOURCE_ALPHA,
                );
                let _: () = msg1(
                    color,
                    sel!("setDestinationRGBBlendFactor:"),
                    blend_factor::ONE_MINUS_SOURCE_ALPHA,
                );
                let _: () = msg1(color, sel!("setSourceAlphaBlendFactor:"), blend_factor::ONE);
                let _: () = msg1(
                    color,
                    sel!("setDestinationAlphaBlendFactor:"),
                    blend_factor::ONE_MINUS_SOURCE_ALPHA,
                );
            }
        }
        let mut err: Id = NIL;
        let state: Id = msg2(
            device.device_id(),
            sel!("newRenderPipelineStateWithDescriptor:error:"),
            desc.id(),
            &mut err as *mut Id,
        );
        Owned::from_retained(state).ok_or_else(|| MetalError::PipelineCreation(error_message(err)))
    }
}

/// One fullscreen draw: `u` at fragment buffer 0, `textures` at 0 to 2,
/// writing `targets` — cleared to transparent black, or loaded when `load`.
fn encode(
    cmd: Id,
    pipe: &Owned,
    targets: Targets<'_>,
    load: bool,
    u: &Vol,
    textures: [Id; 3],
) -> Result<(), MetalError> {
    let one;
    let targets: &[Id] = match targets {
        Targets::One(t) => {
            one = [t];
            &one
        }
        Targets::Many(ts) => ts,
    };
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc: Id = msg0(
            class!("MTLRenderPassDescriptor"),
            sel!("renderPassDescriptor"),
        );
        let attachments: Id = msg0(desc, sel!("colorAttachments"));
        for (i, t) in targets.iter().enumerate() {
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), i);
            let _: () = msg1(color, sel!("setTexture:"), *t);
            let action = if load {
                load_action::LOAD
            } else {
                load_action::CLEAR
            };
            let _: () = msg1(color, sel!("setLoadAction:"), action);
            let _: () = msg1(color, sel!("setStoreAction:"), store_action::STORE);
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
        }
        let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
        if enc.is_null() {
            return Err(MetalError::Unsupported(
                "fluid3d render encoder failed".into(),
            ));
        }
        let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipe.id());
        let bytes = bytemuck::bytes_of(u);
        let _: () = msg3(
            enc,
            sel!("setFragmentBytes:length:atIndex:"),
            bytes.as_ptr(),
            bytes.len(),
            0usize,
        );
        for (i, t) in textures.iter().enumerate() {
            let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), *t, i);
        }
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

#[cfg(all(test, metal_core))]
mod tests {
    use super::*;
    use crate::{PerspectiveCamera, Vector3};

    fn device() -> Option<MetalDevice> {
        MetalDevice::new().ok()
    }

    fn camera() -> PerspectiveCamera {
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.05, 20.0);
        c.position = Vector3::new(0.6, 0.3, 1.8);
        c.look_at(Vector3::ZERO);
        c
    }

    fn small() -> FluidConfig3d {
        FluidConfig3d {
            resolution: 32,
            dye_resolution: 48,
            march_steps: 48,
            buoyancy: 30.0,
            ..FluidConfig3d::default()
        }
    }

    /// The WGSL goes through naga cleanly and every pass the driver builds
    /// comes out with a name.
    #[test]
    fn the_volume_translates_to_msl() {
        let (msl, names) = fluid3d_shader_source().expect("translates");
        for entry in [
            "vs_main",
            "fs_fill3",
            "fs_buoyancy3",
            "fs_splat3",
            "fs_advect3",
            "fs_advect3_self",
            "fs_divergence3",
            "fs_curl3",
            "fs_vorticity3",
            "fs_pressure3",
            "fs_gradient3",
            "fs_residual3",
            "fs_restrict3",
            "fs_prolong3",
            "fs_maxdown3",
            "fs_resolve3",
            "fs_overlay3",
            "fs_raymarch",
            "fs_raymarch_depth",
        ] {
            let name = names.get(entry).unwrap_or_else(|| panic!("no `{entry}`"));
            assert!(msl.contains(name.as_str()), "`{name}` is not in the MSL");
        }
    }

    /// The vertex stage repeated for `metal-fluid` is the renderer's.
    #[cfg(feature = "render")]
    #[test]
    fn the_vertex_stage_is_the_renderers() {
        assert_eq!(VS, crate::renderer::fullscreen::VS);
    }

    /// The memory estimate is what the budget search answers with, and grows
    /// with the cube of the side.
    #[test]
    fn the_budget_buys_a_volume_that_fits() {
        let (solver, dye) = MetalFluid3d::within_memory(64 << 20, 2);
        assert!(solver >= 32 && dye == solver * 2, "{solver}/{dye}");
        assert!(MetalFluid3d::memory_for(solver, dye) <= 64 << 20);
        assert!(MetalFluid3d::memory_for(solver + 8, dye + 16) > 64 << 20);
    }

    /// Everything a frame touches, on a real device, with nothing compared:
    /// splats, impulses, both solvers, accumulation and the read-out.
    #[test]
    fn a_volume_steps_and_draws_every_way() {
        let Some(device) = device() else { return };
        let cam = camera();
        for (solver, samples, fps) in [
            (PressureSolver::Multigrid, 1, false),
            (PressureSolver::Jacobi, 1, true),
            (PressureSolver::Multigrid, 4, true),
        ] {
            let mut fluid = MetalFluid3d::new(
                &device,
                FluidConfig3d {
                    pressure_solver: solver,
                    temporal_samples: samples,
                    show_fps: fps,
                    ..small()
                },
            )
            .unwrap();
            fluid.multiple_splats(&device, 4).unwrap();
            fluid
                .impulse_at_screen(
                    &device,
                    &cam,
                    [0.5, 0.5],
                    [0.05, -0.02],
                    6.0,
                    [1.0, 0.4, 0.2],
                )
                .unwrap();
            for _ in 0..6 {
                fluid.update(&device, 1.0 / 60.0).unwrap();
            }
            assert!(
                fluid.passes_per_step() > 10,
                "a step of {} passes",
                fluid.passes_per_step()
            );
            #[cfg(metal_backend)]
            {
                let rgba = fluid.capture_rgba(&device, &cam, 96, 96).unwrap();
                let lit = rgba
                    .chunks(4)
                    .filter(|p| p[0].max(p[1]).max(p[2]) > 40)
                    .count();
                assert!(
                    lit > 50,
                    "{solver:?}, {samples} samples: only {lit} pixels lit"
                );
            }
        }
    }

    /// The two backends are one simulation: the same seed, steps and camera
    /// draw the same picture, for each solver and with accumulation.
    #[cfg(all(metal_backend, feature = "fluid"))]
    #[test]
    fn metal_fluid3d_matches_wgpu() {
        use crate::fluid::FluidSim3d;
        let Some(device) = device() else { return };
        let Some(gpu) = crate::gpu::testing::context("fluid3d parity") else {
            return;
        };
        let cam = camera();
        for (solver, samples) in [
            (PressureSolver::Multigrid, 1),
            (PressureSolver::Jacobi, 1),
            (PressureSolver::Multigrid, 3),
        ] {
            let config = FluidConfig3d {
                pressure_solver: solver,
                temporal_samples: samples,
                ..small()
            };
            let mut wgpu_sim =
                FluidSim3d::new(gpu.device.clone(), gpu.queue.clone(), config).unwrap();
            let mut metal = MetalFluid3d::new(&device, config).unwrap();
            assert_eq!(metal.grids(), wgpu_sim.grids());
            assert_eq!(
                metal.memory_bytes(),
                wgpu_sim.memory_bytes(),
                "the two allocate differently"
            );
            wgpu_sim.multiple_splats(5);
            metal.multiple_splats(&device, 5).unwrap();
            wgpu_sim.impulse_at_screen(&cam, [0.45, 0.55], [0.04, 0.0], 6.0, [0.2, 0.6, 1.0]);
            metal
                .impulse_at_screen(
                    &device,
                    &cam,
                    [0.45, 0.55],
                    [0.04, 0.0],
                    6.0,
                    [0.2, 0.6, 1.0],
                )
                .unwrap();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            // Several renders when accumulating, so the history is exercised.
            for _ in 0..10 {
                wgpu_sim.step(1.0 / 60.0);
                metal.step(&device, 1.0 / 60.0).unwrap();
                a = wgpu_sim.capture_rgba(&cam, 128, 128);
                b = metal.capture_rgba(&device, &cam, 128, 128).unwrap();
            }
            let lit = a
                .chunks(4)
                .filter(|p| p[0].max(p[1]).max(p[2]) > 40)
                .count();
            assert!(lit > 100, "{solver:?}: wgpu drew almost nothing ({lit})");
            let mean = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (*x as i32 - *y as i32).abs() as f64)
                .sum::<f64>()
                / a.len() as f64;
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max()
                .unwrap_or(0);
            eprintln!("  {solver:?}, {samples} samples: mean |diff| {mean:.3}/255, worst {worst}");
            // One GPU runs both, and the MSL is naga's from one WGSL with
            // wgpu's own bounds checks: nothing should differ at all.
            assert!(worst <= 1, "{solver:?}, {samples} samples: the backends drew different volumes (mean {mean:.3}, worst {worst})");
        }
    }
}
