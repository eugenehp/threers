//! A three-dimensional fluid: the same solver, a volume instead of a plane.
//!
//! ```no_run
//! use threers::fluid::{FluidConfig3d, FluidSim3d};
//! # fn demo(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>,
//! #         camera: &dyn threers::cameras::Camera, frame: &wgpu::Texture) -> Result<(), String> {
//! let mut fluid = FluidSim3d::new(device, queue, FluidConfig3d::default())?;
//! fluid.splat([0.5, 0.2, 0.5], [0.0, 900.0, 0.0], [0.4, 0.2, 0.9]);
//!
//! // Per frame:
//! fluid.step(1.0 / 60.0);
//! fluid.render(camera, frame);
//! # Ok(()) }
//! ```
//!
//! # What changes between two dimensions and three
//!
//! Less than you would expect. Advection, divergence, the Jacobi sweep and the
//! projection are the same equations with a third term; the stencil grows from
//! five points to seven, so the pressure divisor is six rather than four.
//!
//! Two things genuinely change.
//!
//! **Vorticity is a vector.** In the plane the curl is a scalar — one number
//! saying how fast this point spins — and the confinement force is that
//! number times a quarter turn of the gradient. In space the curl is an axis,
//! and the quarter turn becomes the cross product it was always standing in
//! for: `f = ε (N × ω)`. That is the only formula in the solver whose *shape*
//! differs.
//!
//! **You cannot look at it.** A 2D fluid is already an image. A volume has to
//! be rendered, and since it has no surfaces that means marching rays through
//! it and accumulating emission against absorption — see
//! [`FluidSim3d::render`].
//!
//! # Why the volume lives in a 2D texture
//!
//! A render pass writes one attachment, and a 3D attachment is one z-slice, so
//! a 64³ volume would be sixty-four passes per stage and about eighteen
//! hundred a frame. Instead the slices are tiled across a 2D texture and every
//! stage is a single fullscreen pass, exactly as in two dimensions. It also
//! keeps what the rest of this module is built around: no compute stage, so it
//! runs on WebGL2, which has no 3D storage textures at all. The price is doing
//! trilinear filtering by hand — see `src/fluid/volume_shader.rs`.

use std::sync::Arc;

use crate::cameras::Camera;
use crate::renderer::fullscreen::{self, draw, draw_timed, DoubleFbo, Fbo, Load};

use super::passes::FluidFormats;
use super::volume_config::{Atlas, FluidConfig3d, Vol, MG_COARSEST_3D};
use super::PressureSolver;




/// What a volume of this size will allocate, in bytes, before allocating it.
///
/// `formats` must be [resolved](FluidFormats::resolve) against the
/// configuration's `compact_dye` first — what the device *allows* and what
/// the volume will *use* are different, and the difference is a factor of two
/// on the largest field.
///
/// The atlas rounds a cube up to a rectangle, so this is what the textures
/// actually cost rather than `n³ × bytes`: a 100³ volume tiles ten across and
/// ten down and wastes nothing, a 101³ volume tiles eleven across and ten
/// down and wastes a tenth.
///
/// Mirrors [`FluidSim3d::memory_bytes`] — `the_memory_estimate_is_the_memory`
/// holds the two together — so it can be used to choose a resolution on a
/// device where getting it wrong means the app is killed rather than slow.
pub fn memory_for(resolution: u32, dye_resolution: u32, formats: FluidFormats) -> u64 {
    let size = |f: wgpu::TextureFormat| crate::renderer::readback::bytes_per_texel(f) as u64;
    super::volume_config::memory_bytes(
        resolution,
        dye_resolution,
        [size(formats.dye), size(formats.volume_dye), size(formats.scalar)],
    )
}

/// The largest volume that fits a memory budget, as `(solver, dye)`.
///
/// `ratio` is how many dye voxels to a solver voxel along each axis — two is
/// the default everywhere else here, because dye is what the eye reads and
/// motion is what the solver has to resolve.
///
/// For a device where the number matters. A desktop can be asked for 272³ and
/// told no; a phone that is asked for 272³ is killed by the watchdog, and the
/// honest move is to ask for what fits.
///
/// ```
/// use threers::fluid::{within_memory, FluidConfig3d, FluidFormats};
/// // A couple of hundred megabytes, which is a reasonable share of what a
/// // phone will let one app have. Resolved against the configuration, or the
/// // dye is predicted at half what it will cost.
/// let config = FluidConfig3d::default();
/// let formats = FluidFormats::default().resolve(config.compact_dye);
/// let (solver, dye) = within_memory(200 << 20, 2, formats);
/// assert!(solver >= 64 && dye == solver * 2);
/// ```
pub fn within_memory(budget: u64, ratio: u32, formats: FluidFormats) -> (u32, u32) {
    let size = |f: wgpu::TextureFormat| crate::renderer::readback::bytes_per_texel(f) as u64;
    super::volume_config::within_bytes(
        budget,
        ratio,
        [size(formats.dye), size(formats.volume_dye), size(formats.scalar)],
    )
}







const VOL_SIZE: u64 = std::mem::size_of::<Vol>() as u64;

/// A three-dimensional fluid and the textures it lives in.
pub struct FluidSim3d {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: FluidConfig3d,
    formats: FluidFormats,

    sim: Atlas,
    dye_atlas: Atlas,

    velocity: DoubleFbo,
    dye: DoubleFbo,
    pressure: DoubleFbo,
    divergence: Fbo,
    curl: Fbo,
    /// The top level's residual, and the coarse chain under it.
    residual: Fbo,
    mg: Vec<MgLevel3>,
    /// The dye's largest value per brick, at half the dye's side and then a
    /// quarter, for the march to skip by. See `fs_maxdown3`.
    occupancy: [(Atlas, Fbo); 2],
    /// Whether the dye has changed since the occupancy was last built. The
    /// reduction costs the same whatever the render is: rebuilding it per
    /// render would be most of the cost of a small one.
    occupancy_stale: std::cell::Cell<bool>,

    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipes: Pipes,
    out: fullscreen::FormatPipelines<wgpu::RenderPipeline>,
    /// The march into `current`, which is the ordinary march at the
    /// accumulation's format.
    accum_march: Option<wgpu::RenderPipeline>,
    /// The resolve, which writes the history and the caller's target at
    /// once — so it is keyed on the second of those.
    resolve: fullscreen::FormatPipelines<wgpu::RenderPipeline>,
    /// The frame-rate read-out, composited over the target.
    overlay: fullscreen::FormatPipelines<wgpu::RenderPipeline>,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    ubos: std::cell::RefCell<Vec<wgpu::Buffer>>,
    /// Next free uniform block this frame; reset per submit.
    next_ubo: std::cell::Cell<usize>,
    /// What was last written to each uniform block, so a block whose contents
    /// have not changed since the last frame is not written again.
    ubo_contents: std::cell::RefCell<Vec<Vol>>,
    /// GPU timestamps, when something asked for them. `None` until
    /// [`profile`](Self::profile) is called, and on an adapter with no
    /// `TIMESTAMP_QUERY`.
    profiler: Option<crate::renderer::gpu_timer::GpuTimer>,
    /// When the last render was, and the smoothed rate since — for the
    /// read-out, and for anyone who wants the number without drawing it.
    rate: super::read_out::Rate,
    /// The last sixty-four frame times, for the chart under the number.
    history: super::read_out::History,
    /// This frame's march, before it is mixed with the history: half float,
    /// at the size of the target it was last asked for.
    current: Option<Fbo>,
    /// How far along each ray this frame's march saw what it saw, for the
    /// resolve to reproject by.
    depth: Option<Fbo>,
    /// This frame's camera, and its inverse, kept so the resolve can use
    /// them without being handed the camera again.
    view_proj: [f32; 16],
    view_proj_inverse: [f32; 16],
    /// The history, ping-ponged — the resolve reads one and writes the other.
    accum: Option<DoubleFbo>,
    /// The view-projection the accumulation was started from. Any change
    /// means the pixels no longer line up and the history is worthless.
    accum_from: Option<[f32; 16]>,
    /// Frames since the accumulation was reset, which moves the jitter.
    accum_frame: u32,
    /// Render passes the last `step` recorded.
    ///
    /// A deterministic companion to [`stage_ms`](Self::stage_ms): at small
    /// grids the solve is bound by how many passes it is, not by what they
    /// compute, and a pass count does not need a quiet machine to measure.
    passes: std::cell::Cell<u32>,
    /// Which mark the next stage writes. Reset at the top of a frame.
    mark: std::cell::Cell<u32>,
    /// Bind groups, kept between frames.
    ///
    /// A step records about fifty passes and every one of them wants a bind
    /// group over (uniform block, two textures). Those triples repeat exactly
    /// frame after frame — the pass order is fixed, the uniform blocks are
    /// handed out in the same order, and the textures only change when the
    /// grids are rebuilt — so building them again each frame is work whose
    /// answer is already known. Keyed on the slot and on *which* textures,
    /// because the ping-pong means the same slot alternates between two.
    binds: std::cell::RefCell<std::collections::HashMap<(usize, u64, u64), wgpu::BindGroup>>,

    rng: super::Rng,
    paused: bool,
}

struct Pipes {
    fill: wgpu::RenderPipeline,
    buoyancy: wgpu::RenderPipeline,
    splat_velocity: wgpu::RenderPipeline,
    splat_dye: wgpu::RenderPipeline,
    advect_velocity: wgpu::RenderPipeline,
    advect_dye: wgpu::RenderPipeline,
    divergence: wgpu::RenderPipeline,
    curl: wgpu::RenderPipeline,
    vorticity: wgpu::RenderPipeline,
    pressure: wgpu::RenderPipeline,
    gradient: wgpu::RenderPipeline,
    maxdown: wgpu::RenderPipeline,
    residual: wgpu::RenderPipeline,
    restrict: wgpu::RenderPipeline,
    prolong: wgpu::RenderPipeline,
}

/// The stages a frame is reported in, in the order they are recorded.
///
/// A mark is written at the start of each one's first pass and a last mark at
/// the end of the frame, so consecutive differences are the stages — see
/// [`GpuTimer::with_marks`](crate::renderer::gpu_timer::GpuTimer::with_marks).
pub const STAGE_NAMES: [&str; 5] = ["curl+vort", "divergence", "solve", "project", "advect"];

/// Marks: one opening each stage, plus one that closes the last.
///
/// Every mark is written at the *beginning* of a pass. Writing one at each
/// end of a single pass is the obvious way to time it and it does not work
/// here — asked for both on the ray march, this GPU returned a pair in the
/// wrong order on most frames, and a plausible number on some. Marks across
/// passes are reliable. So the closing mark goes on whatever pass comes next
/// — the occupancy reduction, or the march if skipping is off — and the
/// stage it opens is not one of these. The march is timed by rendering in a
/// loop and watching the clock, which works because it is one pass of real
/// work rather than tens of microseconds.
const MARKS: u32 = STAGE_NAMES.len() as u32 + 1;

/// Half and quarter of the dye's side, holding its largest value per brick.
///
/// Two levels because the march wants a brick big enough to be worth a test
/// and small enough to hug the plume: a quarter of the side is four dye
/// voxels, which measured out as 85% empty on a settled plume. Half a
/// megabyte at the default resolution, against the dye's thirty-three.
fn occupancy_levels(
    device: &wgpu::Device,
    dye_grid: u32,
    format: wgpu::TextureFormat,
) -> [(Atlas, Fbo); 2] {
    let level = |n: u32| {
        let atlas = Atlas::new(n.max(2));
        let (w, h) = atlas.size();
        (atlas, Fbo::new(device, "fluid3d occupancy", w, h, format))
    };
    [level(dye_grid / 2), level(dye_grid / 4)]
}

/// Which format the dye volume lives in.
///
/// `rg11b10ufloat` is filterable everywhere but a render target only where
/// the device enabled `RG11B10UFLOAT_RENDERABLE`, so this asks the device
/// rather than assuming. Falling back costs memory, not correctness.
fn packed_dye(
    device: &wgpu::Device,
    config: &FluidConfig3d,
    formats: FluidFormats,
) -> wgpu::TextureFormat {
    if !config.compact_dye {
        return formats.resolve(false).volume_dye;
    }
    if device
        .features()
        .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
    {
        wgpu::TextureFormat::Rg11b10Ufloat
    } else {
        formats.dye
    }
}

/// A filled uniform block, and which slot of the pool it came from.
///
/// The slot travels with the buffer because the bind-group cache is keyed on
/// it, and deriving it from "the last one handed out" is only correct while
/// every block is bound before the next is filled.
struct Ubo {
    buffer: wgpu::Buffer,
    slot: usize,
}


/// One coarse level of the volume's multigrid solver.
struct MgLevel3 {
    atlas: Atlas,
    /// The correction being solved for here, ping-ponged by the smoother.
    pressure: DoubleFbo,
    /// The restricted residual from the level above.
    rhs: Fbo,
    /// This level's own residual, on the way down.
    residual: Fbo,
}

/// Coarse levels for a grid of `grid` voxels a side, halving as far as it
/// goes.
///
/// They are cheap: a level is an eighth of the one above it, so the whole
/// chain below a 64³ grid costs about a seventh of one more 64³ field.
fn mg_levels3(
    device: &wgpu::Device,
    grid: u32,
    coarsest: u32,
    format: wgpu::TextureFormat,
) -> Vec<MgLevel3> {
    let mut levels = Vec::new();
    let mut n = grid;
    let coarsest = coarsest.max(MG_COARSEST_3D);
    while n / 2 >= coarsest && levels.len() < 6 {
        n /= 2;
        let atlas = Atlas::new(n);
        let (w, h) = atlas.size();
        levels.push(MgLevel3 {
            atlas,
            pressure: DoubleFbo::new(device, "fluid3d mg pressure", w, h, format),
            rhs: Fbo::new(device, "fluid3d mg rhs", w, h, format),
            residual: Fbo::new(device, "fluid3d mg residual", w, h, format),
        });
    }
    levels
}

impl FluidSim3d {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: FluidConfig3d,
    ) -> Result<Self, String> {
        Self::with_formats(device, queue, config, FluidFormats::default())
    }

    pub fn with_formats(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: FluidConfig3d,
        formats: FluidFormats,
    ) -> Result<Self, String> {
        if config.pressure_iterations == 0 {
            return Err("fluid3d: pressure_iterations must be at least 1".into());
        }
        let max = device.limits().max_texture_dimension_2d;
        let sim = Atlas::largest_within(config.resolution, max);
        let dye_atlas = Atlas::largest_within(config.dye_resolution, max);
        // The caller may have handed over the baseline formats, which cannot
        // know what this device will render to. Upgrade the dye here rather
        // than make every caller thread an adapter through.
        let formats = FluidFormats {
            volume_dye: packed_dye(&device, &config, formats),
            ..formats
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers fluid3d bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(VOL_SIZE),
                    },
                    count: None,
                },
                tex(1),
                tex(2),
                // A third, for the one pass that needs three: the resolve
                // reads this frame's march, the history, and the depth the
                // march saw. Every other pass binds the dye here and
                // ignores it — an entry the shader does not declare is
                // allowed, and one layout for everything is worth more than
                // a binding nobody reads.
                tex(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers fluid3d layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers fluid3d"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    fullscreen::VS,
                    super::read_out::glyphs(),
                    super::volume_shader::SHADER
                )
                .into(),
            ),
        });

        // Three channels of velocity and of curl, so four: there is no
        // *signed* three-channel float format. The dye is unsigned and gets
        // the packed one.
        let vec_fmt = formats.dye;
        let dye_fmt = formats.volume_dye;
        let scalar = formats.scalar;
        let p = |fs: &str, format| build(&device, &layout, &module, fs, format);
        let pipes = Pipes {
            fill: p("fs_fill3", vec_fmt),
            buoyancy: p("fs_buoyancy3", vec_fmt),
            splat_velocity: p("fs_splat3", vec_fmt),
            splat_dye: p("fs_splat3", dye_fmt),
            advect_velocity: p("fs_advect3_self", vec_fmt),
            advect_dye: p("fs_advect3", dye_fmt),
            divergence: p("fs_divergence3", scalar),
            curl: p("fs_curl3", vec_fmt),
            vorticity: p("fs_vorticity3", vec_fmt),
            pressure: p("fs_pressure3", scalar),
            gradient: p("fs_gradient3", vec_fmt),
            maxdown: p("fs_maxdown3", scalar),
            residual: p("fs_residual3", scalar),
            restrict: p("fs_restrict3", scalar),
            prolong: p("fs_prolong3", scalar),
        };

        let (sw, sh) = sim.size();
        let (dw, dh) = dye_atlas.size();
        Ok(Self {
            velocity: DoubleFbo::new(&device, "fluid3d velocity", sw, sh, vec_fmt),
            dye: DoubleFbo::new(&device, "fluid3d dye", dw, dh, dye_fmt),
            pressure: DoubleFbo::new(&device, "fluid3d pressure", sw, sh, scalar),
            divergence: Fbo::new(&device, "fluid3d divergence", sw, sh, scalar),
            curl: Fbo::new(&device, "fluid3d curl", sw, sh, vec_fmt),
            residual: Fbo::new(&device, "fluid3d residual", sw, sh, scalar),
            mg: mg_levels3(&device, sim.grid, config.multigrid_coarsest, scalar),
            occupancy: occupancy_levels(&device, dye_atlas.grid, scalar),
            occupancy_stale: std::cell::Cell::new(true),
            sampler: fullscreen::linear_sampler(&device),
            ubos: std::cell::RefCell::new((0..48).map(|_| ubo(&device)).collect()),
            next_ubo: std::cell::Cell::new(0),
            profiler: None,
            rate: Default::default(),
            history: Default::default(),
            current: None,
            depth: None,
            view_proj: [0.0; 16],
            view_proj_inverse: [0.0; 16],
            accum: None,
            accum_from: None,
            accum_frame: 0,
            passes: std::cell::Cell::new(0),
            mark: std::cell::Cell::new(0),
            ubo_contents: std::cell::RefCell::new(Vec::new()),
            binds: std::cell::RefCell::new(std::collections::HashMap::new()),
            rng: super::Rng::new(config.seed),
            device,
            queue,
            config,
            formats,
            sim,
            dye_atlas,
            bgl,
            pipes,
            out: fullscreen::FormatPipelines::new(),
            accum_march: None,
            resolve: fullscreen::FormatPipelines::new(),
            overlay: fullscreen::FormatPipelines::new(),
            layout,
            module,
            paused: false,
        })
    }

    pub fn config(&self) -> &FluidConfig3d {
        &self.config
    }

    /// Replace the configuration, rebuilding the grids if the resolution
    /// changed.
    ///
    /// A resolution change throws the fluid away — see
    /// [`set_resolution`](Self::set_resolution) for what that costs and why
    /// there is nothing to carry over.
    pub fn set_config(&mut self, config: FluidConfig3d) {
        let (solver, dye) = (config.resolution, config.dye_resolution);
        // `compact_dye` is a texture format, so changing it means new
        // textures — which `set_resolution` only does when a size changed.
        let refmt = packed_dye(&self.device, &config, self.formats) != self.formats.volume_dye;
        let relevel = config.multigrid_coarsest != self.config.multigrid_coarsest;
        self.config = config;
        if refmt || relevel {
            self.formats.volume_dye = packed_dye(&self.device, &self.config, self.formats);
            self.rebuild_fields();
        }
        self.set_resolution(solver, dye);
        // Anything cached names a uniform block or a texture whose contents
        // the new configuration may have changed.
        self.ubo_contents.borrow_mut().clear();
    }

    /// Rebuild the volume at a new size, in voxels a side.
    ///
    /// The fields are dropped and reallocated empty: an interpolated volume
    /// would be eight textures of resampling, and the thing it would preserve
    /// — a plume mid-flight — is a second of simulation. The splat colours
    /// and the camera survive because they are not in here.
    ///
    /// Both are requests. A volume costs the cube of its side, and the atlas
    /// that holds it is a 2D texture with a device limit on it, so the
    /// numbers that come back from [`grids`](Self::grids) are halved until
    /// they fit. Nothing is allocated if neither changed, so this is cheap to
    /// call from a slider's event handler.
    pub fn set_resolution(&mut self, solver: u32, dye: u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        let sim = Atlas::largest_within(solver.max(2), max);
        let dye_atlas = Atlas::largest_within(dye.max(2), max);
        self.config.resolution = solver;
        self.config.dye_resolution = dye;
        if sim.grid == self.sim.grid && dye_atlas.grid == self.dye_atlas.grid {
            return;
        }
        self.sim = sim;
        self.dye_atlas = dye_atlas;
        self.rebuild_fields();
    }

    /// Allocate every field at the current grids and formats, and empty it.
    fn rebuild_fields(&mut self) {
        let (sw, sh) = self.sim.size();
        let (dw, dh) = self.dye_atlas.size();
        let vec_fmt = self.formats.dye;
        let scalar = self.formats.scalar;
        self.velocity = DoubleFbo::new(&self.device, "fluid3d velocity", sw, sh, vec_fmt);
        self.dye = DoubleFbo::new(&self.device, "fluid3d dye", dw, dh, self.formats.volume_dye);
        self.pressure = DoubleFbo::new(&self.device, "fluid3d pressure", sw, sh, scalar);
        self.divergence = Fbo::new(&self.device, "fluid3d divergence", sw, sh, scalar);
        self.curl = Fbo::new(&self.device, "fluid3d curl", sw, sh, vec_fmt);
        self.residual = Fbo::new(&self.device, "fluid3d residual", sw, sh, scalar);
        self.mg = mg_levels3(
            &self.device,
            self.sim.grid,
            self.config.multigrid_coarsest,
            scalar,
        );
        self.occupancy = occupancy_levels(&self.device, self.dye_atlas.grid, scalar);
        self.occupancy_stale.set(true);
        // Every cached bind group names a texture that no longer exists.
        self.binds.borrow_mut().clear();
        // A fresh texture is undefined until something writes it, and every
        // stage here reads before it writes.
        self.clear_fields();
    }

    /// Zero every field, so a rebuilt volume starts from nothing rather than
    /// from whatever the allocator handed over.
    fn clear_fields(&mut self) {
        let mut enc = self.encoder("fluid3d clear");
        for target in [
            &self.velocity.read.view,
            &self.velocity.write.view,
            &self.pressure.read.view,
            &self.pressure.write.view,
            &self.divergence.view,
            &self.curl.view,
            &self.dye.read.view,
            &self.dye.write.view,
        ] {
            // A render pass that loads nothing and draws nothing: the clear
            // is the whole operation, and it costs one tile flush rather than
            // a fullscreen triangle.
            fullscreen::begin(&mut enc, target, Load::Clear);
        }
        self.queue.submit([enc.finish()]);
    }

    /// How many coarse levels the multigrid solver is using, and how many
    /// render passes a step records. The second is what the GPU time in the
    /// solve is mostly made of — see `fluid3d_bench`.
    pub fn solver_levels(&self) -> usize {
        self.mg.len()
    }

    /// The solver and dye grids, in voxels a side, after the device's texture
    /// limit has had its say.
    pub fn grids(&self) -> (u32, u32) {
        (self.sim.grid, self.dye_atlas.grid)
    }

    /// Bytes of GPU memory the fields occupy.
    pub fn memory_bytes(&self) -> u64 {
        let bytes = |f: &Fbo| {
            crate::renderer::readback::bytes_per_texel(f.texture.format()) as u64
                * f.width as u64
                * f.height as u64
        };
        let double = |f: &DoubleFbo| bytes(&f.read) + bytes(&f.write);
        double(&self.velocity)
            + double(&self.dye)
            + double(&self.pressure)
            + bytes(&self.divergence)
            + bytes(&self.curl)
            + bytes(&self.residual)
            + self
                .mg
                .iter()
                .map(|l| double(&l.pressure) + bytes(&l.rhs) + bytes(&l.residual))
                .sum::<u64>()
            + self.occupancy.iter().map(|(_, f)| bytes(f)).sum::<u64>()
    }

    pub fn formats(&self) -> FluidFormats {
        self.formats
    }

    /// Renders a second, smoothed, measured between calls to
    /// [`render`](Self::render). Zero until the second one.
    pub fn fps(&self) -> f32 {
        self.rate.fps()
    }

    /// Ask the GPU how long each stage of a frame takes.
    ///
    /// Wall-clock timing round `step` measures the CPU recording the work and
    /// almost none of the work; on a machine with anything else running it
    /// stops meaning anything at all. These come off the GPU's own clock.
    ///
    /// Returns false where the adapter has no `TIMESTAMP_QUERY`. The figures
    /// are a frame or two behind — reading them back without stalling the
    /// pipeline is the whole trick — so they are a statistic, not something
    /// to make a per-frame decision on.
    pub fn profile(&mut self) -> bool {
        if self.profiler.is_none() {
            self.profiler =
                crate::renderer::gpu_timer::GpuTimer::with_marks(&self.device, &self.queue, MARKS);
        }
        self.profiler.is_some()
    }

    /// Milliseconds the GPU spent in each of [`STAGE_NAMES`], from the most
    /// recent completed readback. All zero until one lands.
    ///
    /// The ray march is not among them: every mark is written at the start
    /// of a pass, so the last pass of a frame has nothing after it to close
    /// its stage. Time the march by rendering in a loop and watching the
    /// clock — it is one pass of real work, so the wall is good enough.
    pub fn stage_ms(&self) -> [f32; STAGE_NAMES.len()] {
        let mut out = [0.0; STAGE_NAMES.len()];
        if let Some(p) = &self.profiler {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = p.span_ms(i);
            }
        }
        out
    }

    /// Render passes the last [`step`](Self::step) recorded.
    ///
    /// Zero before the first one. The splats and the render are not counted:
    /// this is the solver's own cost, which is what the number is for.
    pub fn passes_per_step(&self) -> u32 {
        self.passes.get()
    }

    /// Record one pass, counting it.
    ///
    /// Every draw in a step goes through here so that the count cannot drift
    /// from the thing it counts.
    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        load: Load,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
        marks: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        self.passes.set(self.passes.get() + 1);
        draw_timed(enc, target, load, pipeline, bind, marks);
    }

    /// Take the timestamp writes that open stage `i`, once per frame.
    ///
    /// The first pass of a stage asks for this and the rest do not, so a
    /// stage whose first pass is conditional — buoyancy, confinement — still
    /// opens on whatever pass actually runs first.
    fn take_mark(&self, stage: u32) -> Option<u32> {
        // Strictly in order. A configuration that skips a stage — no
        // confinement, so no curl — would otherwise shift every later mark
        // into the wrong slot and report five plausible wrong numbers; this
        // way the frame is not reported at all and the last good one stands.
        if self.profiler.is_none() || self.mark.get() != stage {
            return None;
        }
        self.mark.set(stage + 1);
        Some(stage)
    }

    /// The timestamp writes for a mark index, if there is a profiler.
    ///
    /// Separate from [`take_mark`](Self::take_mark) because the result
    /// borrows the query set, and the ray march has to claim its mark before
    /// taking a mutable borrow to compile its pipeline.
    fn writes(&self, mark: Option<u32>) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        match (mark, &self.profiler) {
            (Some(i), Some(p)) => Some(p.mark(i)),
            _ => None,
        }
    }

    fn open(&self, stage: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let mark = self.take_mark(stage);
        self.writes(mark)
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// The dye volume, as the atlas it is stored in.
    pub fn dye_texture(&self) -> &wgpu::Texture {
        &self.dye.read.texture
    }

    /// How the dye volume is packed into that texture.
    pub fn dye_atlas(&self) -> Atlas {
        self.dye_atlas
    }

    // -- input ------------------------------------------------------------

    /// A ball of dye and a push, at `at` in the unit cube.
    pub fn splat(&mut self, at: [f32; 3], velocity: [f32; 3], color: [f32; 3]) {
        self.splat_batch(&[(at, velocity, color)]);
    }

    /// `count` splats at random places, as the 2D module's burst.
    /// A push into the volume, where a point on the screen looks into it.
    ///
    /// `uv` is in 0..1 with the origin at the top left — a pointer position
    /// over the canvas — and `drag` is how far it moved in the same units,
    /// zero for a click that did not move.
    ///
    /// # Where the impulse lands
    ///
    /// A screen point is a ray, not a place: everything along it is under
    /// the pointer. Something has to pick the depth, and there is no right
    /// answer without reading the volume back, which would mean a stall.
    ///
    /// So it lands on the plane through the volume's centre, facing the
    /// camera. Poking the middle of the box is what a pointer over the
    /// middle of the box looks like it should do, it is stable as the camera
    /// orbits — the plane turns with it — and it cannot land behind the
    /// viewer or outside the box's depth range. A ray that misses the box
    /// entirely does nothing.
    ///
    /// The push is the drag, in world units at that same depth, so dragging
    /// across the screen pushes across the screen however the camera is
    /// turned. `strength` scales it into voxels a second.
    pub fn impulse_at_screen(
        &mut self,
        camera: &dyn Camera,
        uv: [f32; 2],
        drag: [f32; 2],
        strength: f32,
        colour: [f32; 3],
    ) {
        if let Some((at, velocity)) =
            super::volume_config::impulse(&self.config, self.sim.grid, camera, uv, drag, strength)
        {
            self.splat(at, velocity, colour);
        }
    }

    pub fn multiple_splats(&mut self, count: u32) {
        let batch = super::volume_config::random_splats(&mut self.rng, count);
        self.splat_batch(&batch);
    }

    fn splat_batch(&mut self, splats: &[([f32; 3], [f32; 3], [f32; 3])]) {
        // A pass each, unlike the 2D module's batch: in three dimensions the
        // pass is over an atlas rather than an image, and the arithmetic per
        // texel is a single exponential either way, so batching would save
        // only the submits — and a 3D burst is a handful of blobs, not
        // twenty-five.
        let radius = self.config.splat_radius / 100.0;
        for (at, vel, color) in splats {
            let mut enc = self.encoder("fluid3d splat");
            self.next_ubo.set(0);
            for dye in [false, true] {
                let field = if dye { &self.dye } else { &self.velocity };
                let atlas = if dye { self.dye_atlas } else { self.sim };
                let value = if dye { *color } else { *vel };
                let u = self.params(
                    atlas,
                    Vol {
                        point: [at[0], at[1], at[2], radius],
                        color: [value[0], value[1], value[2], 1.0],
                        ..Default::default()
                    },
                );
                let bg = self.bind(&u, &field.read, None);
                let pipe = if dye {
                    &self.pipes.splat_dye
                } else {
                    &self.pipes.splat_velocity
                };
                draw(&mut enc, &field.write.view, Load::Clear, pipe, &bg);
            }
            self.queue.submit([enc.finish()]);
            self.velocity.swap();
            self.dye.swap();
            self.occupancy_stale.set(true);
        }
    }

    /// Set the whole velocity field to one vector, in voxels per second.
    pub fn set_uniform_velocity(&mut self, v: [f32; 3]) {
        let mut enc = self.encoder("fluid3d fill");
        self.next_ubo.set(0);
        let u = self.params(
            self.sim,
            Vol {
                color: [v[0], v[1], v[2], 0.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&u, &self.velocity.read, None);
        draw(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.fill,
            &bg,
        );
        self.queue.submit([enc.finish()]);
        self.velocity.swap();
    }

    // -- the frame --------------------------------------------------------

    pub fn update(&mut self, dt: f32) {
        let dt = dt * self.config.time_scale.max(0.0);
        if !self.paused && dt > 0.0 {
            self.step(dt);
        }
    }

    /// One step of the solver.
    pub fn step(&mut self, dt: f32) {
        let c = self.config;
        let mut enc = self.encoder("fluid3d step");
        self.next_ubo.set(0);
        self.mark.set(0);
        self.passes.set(0);

        // Buoyancy, before anything else: dye is hot, hot fluid rises. In two
        // dimensions this is a stylistic choice; in three there is an up.
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
                self.sim,
                self.sim,
                self.dye_atlas,
                Vol {
                    texel: [0.0, 0.0, 0.0, dt],
                    params: [0.0, c.buoyancy, 0.0, 0.0],
                    // `color` is unused by this pass, so the axis rides in
                    // it rather than widening the block for three floats.
                    color: [axis[0], axis[1], axis[2], 0.0],
                    ..Default::default()
                },
            );
            let bg = self.bind(&u, &self.velocity.read, Some(&self.dye.read));
            self.pass(
                &mut enc,
                &self.velocity.write.view,
                Load::Clear,
                &self.pipes.buoyancy,
                &bg,
                self.open(0),
            );
            self.velocity.swap();
        }

        // 1. Curl — a vector field in three dimensions. Two passes with a
        // seven-point stencil each, and nothing downstream reads the curl, so
        // with confinement off they are skipped rather than multiplied by
        // zero.
        if c.curl != 0.0 {
            let u = self.params(self.sim, Vol::default());
            let bg = self.bind(&u, &self.velocity.read, None);
            self.pass(
                &mut enc,
                &self.curl.view,
                Load::Clear,
                &self.pipes.curl,
                &bg,
                self.open(0),
            );

            // 2. Vorticity confinement, from it.
            let u = self.params(
                self.sim,
                Vol {
                    texel: [0.0, 0.0, 0.0, dt],
                    params: [0.0, c.curl, 0.0, 0.0],
                    ..Default::default()
                },
            );
            let bg = self.bind(&u, &self.velocity.read, Some(&self.curl));
            self.pass(
                &mut enc,
                &self.velocity.write.view,
                Load::Clear,
                &self.pipes.vorticity,
                &bg,
                None,
            );
            self.velocity.swap();
        }

        // 3. Divergence.
        let u = self.params(self.sim, Vol::default());
        let bg = self.bind(&u, &self.velocity.read, None);
        self.pass(
            &mut enc,
            &self.divergence.view,
            Load::Clear,
            &self.pipes.divergence,
            &bg,
            self.open(1),
        );

        // 4. The solve, starting from a fraction of last frame's answer —
        // which the first sweep applies to what it reads rather than a pass
        // of its own multiplying a whole grid by a constant.
        let open = self.take_mark(2);
        match c.pressure_solver {
            PressureSolver::Jacobi => {
                self.record_jacobi(&mut enc, c.pressure_iterations, c.pressure, open)
            }
            PressureSolver::Multigrid => self.record_multigrid(&mut enc, open),
        }

        // 6. Subtract the gradient.
        let u = self.params(
            self.sim,
            Vol {
                params: [0.0, 0.0, c.gradient_scale, 0.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&u, &self.pressure.read, Some(&self.velocity.read));
        self.pass(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.gradient,
            &bg,
            self.open(3),
        );
        self.velocity.swap();

        // 7. The field carries itself.
        let u = self.params(
            self.sim,
            Vol {
                texel: [0.0, 0.0, 0.0, dt],
                params: [c.velocity_dissipation, 0.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&u, &self.velocity.read, Some(&self.velocity.read));
        self.pass(
            &mut enc,
            &self.velocity.write.view,
            Load::Clear,
            &self.pipes.advect_velocity,
            &bg,
            self.open(4),
        );
        self.velocity.swap();

        // 8. …and it carries the dye. The velocity is in *solver* voxels a
        // second, and the advection steps in the sampled field's own voxels,
        // so the dye pass needs the solver's texel size rather than its own.
        let u = self.params_mixed(
            self.dye_atlas,
            self.sim,
            self.dye_atlas,
            Vol {
                texel: [0.0, 0.0, 0.0, dt],
                params: [c.density_dissipation, 0.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        let bg = self.bind(&u, &self.velocity.read, Some(&self.dye.read));
        self.pass(
            &mut enc,
            &self.dye.write.view,
            Load::Clear,
            &self.pipes.advect_dye,
            &bg,
            None,
        );
        self.dye.swap();
        self.occupancy_stale.set(true);

        self.queue.submit([enc.finish()]);
    }

    /// `count` Jacobi sweeps on the top-level pressure, ping-ponging.
    ///
    /// No relaxation weight, unlike the textbook multigrid smoother. The
    /// argument is the one the 2D module sets out at `MG_WEIGHT` and it
    /// carries over unchanged: damping is there to kill the checkerboard
    /// mode, this divergence is a central difference over two voxels, and
    /// that operator has a zero at exactly that frequency — so there is no
    /// checkerboard in the right-hand side to kill, and damping would only
    /// slow the modes that do exist.
    fn record_jacobi(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        count: u32,
        first_scale: f32,
        open: Option<u32>,
    ) {
        if count == 0 {
            return;
        }
        // Two uniform blocks, because the first sweep may scale what it
        // reads and the rest never do — see `fs_pressure3`.
        let scaled = self.params(
            self.sim,
            Vol {
                params: [0.0, 0.0, first_scale, 0.0],
                ..Default::default()
            },
        );
        let plain = self.params(
            self.sim,
            Vol {
                params: [0.0, 0.0, 1.0, 0.0],
                ..Default::default()
            },
        );
        let groups = [
            self.bind(&scaled, &self.pressure.read, Some(&self.divergence)),
            self.bind(&scaled, &self.pressure.write, Some(&self.divergence)),
            self.bind(&plain, &self.pressure.read, Some(&self.divergence)),
            self.bind(&plain, &self.pressure.write, Some(&self.divergence)),
        ];
        let mut marks = self.writes(open);
        for i in 0..count {
            let even = i % 2 == 0;
            let bg = &groups[usize::from(i > 0) * 2 + usize::from(!even)];
            let target = if even {
                &self.pressure.write.view
            } else {
                &self.pressure.read.view
            };
            self.pass(
                enc,
                target,
                Load::Clear,
                &self.pipes.pressure,
                bg,
                marks.take(),
            );
        }
        if count % 2 == 1 {
            self.pressure.swap();
        }
    }

    /// A multigrid V-cycle: down to the coarsest grid, then back up.
    ///
    /// The shape is the textbook one, and the same one the 2D module uses. On
    /// the way down each level smooths a little, measures what it failed to
    /// satisfy, and hands that residual to a grid half the size as a fresh
    /// problem starting from zero. At the bottom — four to eight voxels
    /// across — a few sweeps *are* a solve. On the way up each correction is
    /// interpolated into the level above and smoothed again, to take the
    /// interpolation's own roughness back out.
    ///
    /// It matters more here than in two dimensions. A sweep moves information
    /// one voxel either way, and a volume is wider in the sense that counts:
    /// the pressure from a puff falls off as 1/r rather than log r, and six
    /// neighbours share the correction instead of four.
    fn record_multigrid(&mut self, enc: &mut wgpu::CommandEncoder, open: Option<u32>) {
        if self.mg.is_empty() {
            let (count, scale) = (self.config.pressure_iterations, self.config.pressure);
            self.record_jacobi(enc, count, scale, open);
            return;
        }
        let nu = self.config.multigrid_smooth.max(1);
        // The bottom grid gets enough sweeps to *be* a solve, and no more.
        // Information crosses a grid one voxel a sweep, so twice the side is
        // the whole domain twice over — on a 4³ grid that is eight sweeps,
        // and the twenty the 2D module spends there would be twenty render
        // passes over sixty-four voxels. The arithmetic is free at that size;
        // the passes are not.
        let coarsest = self.mg.last().map(|l| l.atlas.grid).unwrap_or(4);
        let coarse_sweeps = (2 * coarsest).clamp(nu * 2, self.config.pressure_iterations.max(8));
        let mut open = open;
        for cycle in 0..self.config.multigrid_cycles.max(1) {
            // --- down -------------------------------------------------
            // Only the very first sweep carries the carry-over scale; after
            // that the pressure is this frame's and is read as it stands.
            let scale = if cycle == 0 {
                self.config.pressure
            } else {
                1.0
            };
            self.record_jacobi(enc, nu, scale, open.take());
            for level in 0..self.mg.len() {
                self.record_residual3(enc, level);
                self.record_restrict3(enc, level);
                let sweeps = if level + 1 == self.mg.len() {
                    coarse_sweeps
                } else {
                    nu
                };
                // The coarse problem is for a *correction*, so it starts at
                // zero. A scale of zero says so, and the sweep that used to
                // follow a clear-only pass now *is* the clear: it reads
                // nothing and writes `-rhs / 6`.
                self.record_smooth3(enc, level, sweeps, 0.0);
            }
            // --- up ---------------------------------------------------
            for level in (0..self.mg.len()).rev() {
                self.record_prolong3(enc, level);
                if level == 0 {
                    self.record_jacobi(enc, nu, 1.0, None);
                } else {
                    self.record_smooth3(enc, level - 1, nu, 1.0);
                }
            }
        }
    }

    /// Smooth coarse `level`'s correction against its own right-hand side.
    fn record_smooth3(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        level: usize,
        sweeps: u32,
        first_scale: f32,
    ) {
        if sweeps == 0 {
            return;
        }
        let atlas = self.mg[level].atlas;
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
        let l = &self.mg[level];
        let groups = [
            self.bind(&scaled, &l.pressure.read, Some(&l.rhs)),
            self.bind(&scaled, &l.pressure.write, Some(&l.rhs)),
            self.bind(&plain, &l.pressure.read, Some(&l.rhs)),
            self.bind(&plain, &l.pressure.write, Some(&l.rhs)),
        ];
        for i in 0..sweeps {
            let even = i % 2 == 0;
            let bg = &groups[usize::from(i > 0) * 2 + usize::from(!even)];
            let target = if even {
                &l.pressure.write.view
            } else {
                &l.pressure.read.view
            };
            self.pass(enc, target, Load::Clear, &self.pipes.pressure, bg, None);
        }
        if sweeps % 2 == 1 {
            self.mg[level].pressure.swap();
        }
    }

    /// What the level above `level` fails to satisfy, at its own resolution.
    ///
    /// At its own resolution and not straight into the coarse texture: that
    /// would be a point sample, which is the one thing a restriction must not
    /// be — it aliases exactly the high-frequency error the smoother was
    /// there to remove.
    fn record_residual3(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let (atlas, pressure, rhs, target) = if level == 0 {
            (
                self.sim,
                &self.pressure.read,
                &self.divergence,
                &self.residual,
            )
        } else {
            let l = &self.mg[level - 1];
            (l.atlas, &l.pressure.read, &l.rhs, &l.residual)
        };
        let u = self.params(atlas, Vol::default());
        let bg = self.bind(&u, pressure, Some(rhs));
        self.pass(
            enc,
            &target.view,
            Load::Clear,
            &self.pipes.residual,
            &bg,
            None,
        );
    }

    /// Average `level`'s residual onto the half-size grid below it.
    fn record_restrict3(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let (fine_atlas, residual) = if level == 0 {
            (self.sim, &self.residual)
        } else {
            let l = &self.mg[level - 1];
            (l.atlas, &l.residual)
        };
        let coarse = self.mg[level].atlas;
        let u = self.params_mixed(coarse, fine_atlas, fine_atlas, Vol::default());
        let bg = self.bind(&u, residual, None);
        self.pass(
            enc,
            &self.mg[level].rhs.view,
            Load::Clear,
            &self.pipes.restrict,
            &bg,
            None,
        );
    }

    /// Interpolate `level`'s correction into the level above and add it.
    fn record_prolong3(&mut self, enc: &mut wgpu::CommandEncoder, level: usize) {
        let coarse_atlas = self.mg[level].atlas;
        let (fine_atlas, fine_read, fine_write) = if level == 0 {
            (self.sim, &self.pressure.read, &self.pressure.write)
        } else {
            let l = &self.mg[level - 1];
            (l.atlas, &l.pressure.read, &l.pressure.write)
        };
        let u = self.params_mixed(fine_atlas, coarse_atlas, fine_atlas, Vol::default());
        let bg = self.bind(&u, &self.mg[level].pressure.read, Some(fine_read));
        self.pass(
            enc,
            &fine_write.view,
            Load::Clear,
            &self.pipes.prolong,
            &bg,
            None,
        );
        if level == 0 {
            self.pressure.swap();
        } else {
            self.mg[level - 1].pressure.swap();
        }
    }

    /// Reduce the dye to its largest value per brick, twice.
    ///
    /// Recorded at render rather than at the end of the step, because a splat
    /// lands outside the step: building it there would let a fresh splat fall
    /// into a brick still marked empty, and the march would step straight
    /// over it.
    fn record_occupancy(&self, enc: &mut wgpu::CommandEncoder, opening: Option<u32>) {
        let mut source = (self.dye_atlas, &self.dye.read);
        let mut marks = self.writes(opening);
        for level in 0..self.occupancy.len() {
            let (atlas, target) = &self.occupancy[level];
            let u = self.params_mixed(*atlas, source.0, source.0, Vol::default());
            let bg = self.bind(&u, source.1, None);
            self.pass(
                enc,
                &target.view,
                Load::Clear,
                &self.pipes.maxdown,
                &bg,
                marks.take(),
            );
            source = (*atlas, target);
        }
    }

    /// March the dye volume from `camera` into `target`.
    ///
    /// Emission and absorption, front to back, stopping a ray once what is
    /// left cannot change the pixel. The volume occupies
    /// [`bounds_min`](FluidConfig3d::bounds_min)`..`[`bounds_max`](FluidConfig3d::bounds_max)
    /// in world space, so it composites with anything else drawn with the same
    /// camera — though it writes alpha rather than reading depth, so it goes
    /// over a scene rather than into it.
    pub fn render(&mut self, camera: &dyn Camera, target: &wgpu::Texture) {
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let size = (target.width(), target.height());
        self.render_to_view(camera, &view, target.format(), size);
    }

    /// March into a view of `size` pixels.
    ///
    /// The size is not decoration: a `TextureView` does not carry its
    /// dimensions, and [`temporal_samples`](FluidConfig3d::temporal_samples)
    /// needs a buffer that matches the target exactly.
    pub fn render_to_view(
        &mut self,
        camera: &dyn Camera,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) {
        let c = self.config;
        self.next_ubo.set(0);
        let accumulating = self.prepare_accum(size);
        let view_proj = camera.projection_matrix().multiply(&camera.view_matrix());
        let inv = view_proj.invert();
        self.view_proj = view_proj.elements;
        self.view_proj_inverse = inv.elements;
        let pos = camera.position();
        // A brick below this cannot change a pixel: the march's own early-out
        // gives up at a thousandth of transmittance. Zero turns skipping off,
        // and the occupancy volume stays bound either way so that what the
        // uniform says about `b` is always what is in `b`.
        let skip_floor = if c.empty_space_skip { 0.0008 } else { 0.0 };
        let u = self.params_mixed(
            self.dye_atlas,
            self.dye_atlas,
            self.occupancy[1].0,
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
        let bg = self.bind(&u, &self.dye.read, Some(&self.occupancy[1].1));
        let mut enc = self.encoder("fluid3d render");
        // Here rather than at the end of the step, because the march is the
        // only thing that reads it: a simulation stepped a hundred times and
        // drawn once would otherwise reduce the dye a hundred times and use
        // the answer once. Two passes a step, which is five percent of one.
        //
        // It carries the mark that closes the last stage when it runs; the
        // march carries it when it does not.
        if c.empty_space_skip && self.occupancy_stale.get() {
            let closing = self.take_mark(STAGE_NAMES.len() as u32);
            self.record_occupancy(&mut enc, closing);
            self.occupancy_stale.set(false);
        }
        // Carries the mark that closes the last stage, if the occupancy
        // reduction did not — it is the next pass either way. Taken before
        // the pipelines, which borrow `self` mutably to compile one on first
        // sight of a format.
        let mark = self.take_mark(STAGE_NAMES.len() as u32);
        if accumulating {
            self.record_accumulated_march(&mut enc, target, format, &bg, mark);
        } else {
            let (device, layout, module) = (&self.device, &self.layout, &self.module);
            let pipeline = self.out.get_or_build(format, || {
                build(device, layout, module, "fs_raymarch", format)
            });
            let marks = match (mark, &self.profiler) {
                (Some(i), Some(p)) => Some(p.mark(i)),
                _ => None,
            };
            draw_timed(&mut enc, target, Load::Clear, pipeline, &bg, marks);
        }
        // The march is the last pass of the frame, so it also carries the
        // mark that closes it — and the resolve only happens when every
        // stage opened, so a frame that skipped one reports nothing rather
        // than six wrong numbers.
        let complete = self.mark.get() == MARKS;
        if let Some(p) = &self.profiler
            && complete
        {
            p.resolve(&mut enc);
        }
        if c.show_fps {
            self.record_overlay(&mut enc, target, format);
        }
        self.queue.submit([enc.finish()]);
        if let Some(p) = &mut self.profiler {
            p.poll(&self.device);
        }
        if accumulating {
            self.accum_frame += 1;
            self.accum_from = Some(view_proj.elements);
        }
        self.rate.tick();
    }

    /// Composite the read-out over the target.
    fn record_overlay(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        // The target the chart is drawn against: what the display is
        // actually delivering, rounded to a sensible refresh, so the line
        // means "keeping up" rather than an arbitrary sixty.
        let target_ms = self.rate.target_ms();
        // Twice the current frame time, so a steady run sits halfway up
        // however fast it is going.
        let scale_ms = self.rate.scale_ms();
        let u = self.params(
            self.dye_atlas,
            Vol {
                params: [self.rate.fps(), scale_ms, target_ms, 0.0],
                ..Default::default()
            },
        );
        let history = {
            let (device, queue) = (self.device.clone(), self.queue.clone());
            self.history
                .view(&device, &queue, self.rate.frames())
                .clone()
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid3d read-out"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&history),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&history),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&history),
                },
            ],
        });
        let (device, layout, module) = (&self.device, &self.layout, &self.module);
        let pipeline = self.overlay.get_or_build(format, || {
            build_blended(
                device,
                layout,
                module,
                "fs_overlay3",
                format,
                wgpu::BlendState::ALPHA_BLENDING,
            )
        });
        // Keeps what is already there: the read-out is lit pixels over a
        // picture, and every other pixel is discarded.
        draw(enc, target, Load::Keep, pipeline, &bind);
    }

    /// Make sure there is an accumulation buffer of the right size, and
    /// decide whether this frame continues the average or starts it.
    ///
    /// Returns false when accumulation is off, which is the only path that
    /// touches no extra memory.
    fn prepare_accum(&mut self, size: (u32, u32)) -> bool {
        if self.config.temporal_samples <= 1 {
            self.accum = None;
            self.current = None;
            self.depth = None;
            self.accum_from = None;
            return false;
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        let resized = !matches!(&self.current, Some(f) if f.width == w && f.height == h);
        if resized {
            let fmt = self.formats.dye;
            self.current = Some(Fbo::new(&self.device, "fluid3d march", w, h, fmt));
            self.accum = Some(DoubleFbo::new(&self.device, "fluid3d history", w, h, fmt));
            // One channel, and it holds a distance in world units rather
            // than a normalised depth, so it wants the range a float gives.
            self.depth = Some(Fbo::new(
                &self.device,
                "fluid3d march depth",
                w,
                h,
                self.formats.scalar,
            ));
        }
        // Any camera movement and the pixels no longer line up, so the
        // history is not an average of this view, it is a smear of the last
        // one. Nothing here reprojects: a volume has no single depth to
        // reproject *by*, and guessing one puts the plume's near face and
        // its far face in different places.
        // Not reset on movement any more: the resolve reprojects through
        // `accum_from`, so a moved camera still has a usable history. Only
        // a resize invalidates it outright, because then the pixels are a
        // different set.
        if resized {
            self.accum_from = None;
            self.accum_frame = 0;
        }
        true
    }

    /// March into the accumulation, then put it on the target.
    ///
    /// Two passes where there was one, and the second is a copy — which is
    /// the price of the first being able to blend with what it wrote last
    /// time. `set_blend_constant` carries the weight, so one pipeline serves
    /// every sample count and the frame after a reset can set it to 1 and
    /// simply overwrite.
    fn record_accumulated_march(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        bg: &wgpu::BindGroup,
        mark: Option<u32>,
    ) {
        let weight = if self.accum_frame == 0 {
            1.0
        } else {
            1.0 / self.config.temporal_samples.max(2) as f32
        };
        let accum_format = self.formats.dye;
        let depth_format = self.formats.scalar;
        if self.accum_march.is_none() {
            self.accum_march = Some(build_targets(
                &self.device,
                &self.layout,
                &self.module,
                "fs_raymarch_depth",
                &[Some(accum_format), Some(depth_format)],
            ));
        }
        {
            let current = self.current.as_ref().expect("prepare_accum made one");
            let depth = self.depth.as_ref().expect("prepare_accum made one");
            let marks = match (mark, &self.profiler) {
                (Some(i), Some(p)) => Some(p.mark(i)),
                _ => None,
            };
            fn attach(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
                wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }
            }
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fluid3d march"),
                color_attachments: &[Some(attach(&current.view)), Some(attach(&depth.view))],
                depth_stencil_attachment: None,
                timestamp_writes: marks,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.accum_march.as_ref().expect("just built"));
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // The resolve, writing the history and the caller's target together.
        // Two attachments rather than two passes: the value is the same and
        // only the format differs, so letting the hardware convert it twice
        // is cheaper than reading it back to convert it once.
        let u = self.params(
            self.dye_atlas,
            Vol {
                params: [0.0, 0.0, weight, self.config.temporal_clip],
                // The resolve unprojects this frame's pixel and projects it
                // through last frame's camera, so it needs both.
                inv_view_proj: self.view_proj_inverse,
                prev_view_proj: self.accum_from.unwrap_or(self.view_proj),
                ..Default::default()
            },
        );
        let current = self.current.take().expect("prepare_accum made one");
        let depth = self.depth.take().expect("prepare_accum made one");
        let mut accum = self.accum.take().expect("prepare_accum made one");
        let bind = self.bind3(&u, &current, &accum.read, &depth);
        {
            let (device, layout, module) = (&self.device, &self.layout, &self.module);
            let pipeline = self.resolve.get_or_build(format, || {
                build_targets(
                    device,
                    layout,
                    module,
                    "fs_resolve3",
                    &[Some(accum_format), Some(format)],
                )
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fluid3d resolve"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &accum.write.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        accum.swap();
        self.current = Some(current);
        self.depth = Some(depth);
        self.accum = Some(accum);
    }

    /// Render into a fresh `Rgba8Unorm` texture and read it back. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_rgba(&mut self, camera: &dyn Camera, width: u32, height: u32) -> Vec<u8> {
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fluid3d capture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.render(camera, &target);
        crate::renderer::readback::texture_bytes(&self.device, &self.queue, &target)
    }

    // -- helpers ----------------------------------------------------------

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Fill in the atlas geometry and take the next free uniform block. Every
    /// field on the same grid, which is every pass but the two that mix the
    /// velocity with the dye.
    fn params(&self, atlas: Atlas, rest: Vol) -> Ubo {
        self.params_mixed(atlas, atlas, atlas, rest)
    }

    /// As [`Self::params`], with each texture's own grid given separately.
    ///
    /// Sampling one field with another's tiling is the bug this exists to
    /// prevent, and it is a quiet one: the dye advection read the velocity
    /// atlas through the dye's tile layout and still produced a plausible
    /// plume.
    fn params_mixed(&self, out: Atlas, a: Atlas, b: Atlas, rest: Vol) -> Ubo {
        let v = Vol {
            out_grid: out.into(),
            a_grid: a.into(),
            b_grid: b.into(),
            ..rest
        };
        // One buffer a pass: `write_buffer` orders against submits, not
        // against draws, so two passes sharing a block in one submit both see
        // whatever was written last. The pool grows rather than wrapping —
        // wrapping is that bug with a cap on it, and a V-cycle needs five
        // blocks a level.
        let i = self.next_ubo.get();
        self.next_ubo.set(i + 1);
        let buffer = {
            let mut pool = self.ubos.borrow_mut();
            while pool.len() <= i {
                pool.push(ubo(&self.device));
            }
            pool[i].clone()
        };
        // Most of these blocks are the same every frame — a grid description
        // and a dt that has not changed — and the pass order is fixed, so
        // slot `i` sees the same contents it saw last frame. Comparing two
        // hundred and forty bytes is cheaper than staging them.
        if !self.config.cache_bindings {
            self.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&v));
            return Ubo { buffer, slot: i };
        }
        let mut seen = self.ubo_contents.borrow_mut();
        while seen.len() <= i {
            // A block nothing has written yet cannot match: `march.x` is a
            // step count and is never zero in a real one.
            seen.push(Vol::default());
        }
        if bytemuck::bytes_of(&seen[i]) != bytemuck::bytes_of(&v) {
            seen[i] = v;
            drop(seen);
            self.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&v));
        }
        Ubo { buffer, slot: i }
    }

    fn bind(&self, params: &Ubo, a: &Fbo, b: Option<&Fbo>) -> wgpu::BindGroup {
        if !self.config.cache_bindings {
            return self.build_bind(&params.buffer, a, b);
        }
        // The *uniform's own* slot, not whichever was handed out last. Those
        // are the same number only while every block is bound immediately
        // after it is filled — and the moment a pass wants two blocks before
        // it binds either, keying on the latter silently hands the second
        // pass the first one's bind group. Which is a solver that stops
        // converging and a sweep count that stops mattering.
        let key = (params.slot, a.id, b.map(|f| f.id).unwrap_or(a.id));
        if let Some(hit) = self.binds.borrow().get(&key) {
            return hit.clone();
        }
        let group = self.build_bind(&params.buffer, a, b);
        self.binds.borrow_mut().insert(key, group.clone());
        group
    }

    fn build_bind(&self, params: &wgpu::Buffer, a: &Fbo, b: Option<&Fbo>) -> wgpu::BindGroup {
        self.build_bind3(params, a, b, None)
    }

    fn build_bind3(
        &self,
        params: &wgpu::Buffer,
        a: &Fbo,
        b: Option<&Fbo>,
        c: Option<&Fbo>,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid3d bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        b.map(|f| &f.view).unwrap_or(&a.view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        c.map(|f| &f.view).unwrap_or(&a.view),
                    ),
                },
            ],
        })
    }

    /// [`bind`](Self::bind), with a third texture. Not cached: the one pass
    /// that uses it runs once a frame, and the cache is keyed on two.
    fn bind3(&self, params: &Ubo, a: &Fbo, b: &Fbo, c: &Fbo) -> wgpu::BindGroup {
        self.build_bind3(&params.buffer, a, Some(b), Some(c))
    }
}

fn tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn ubo(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fluid3d params"),
        size: VOL_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn build(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    build_targets(device, layout, module, fs, &[Some(format)])
}

/// [`build`], for a stage that writes more than one attachment — the resolve,
/// which puts the same value into the history and the caller's target and
/// lets the hardware convert it to each format.
fn build_targets(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs: &str,
    formats: &[Option<wgpu::TextureFormat>],
) -> wgpu::RenderPipeline {
    let targets: Vec<_> = formats
        .iter()
        .map(|f| {
            f.map(|format| wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    build_with(device, layout, module, fs, &targets)
}

/// [`build`], for a stage that composites over what is already there.
fn build_blended(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    build_with(
        device,
        layout,
        module,
        fs,
        &[Some(wgpu::ColorTargetState {
            format,
            blend: Some(blend),
            write_mask: wgpu::ColorWrites::ALL,
        })],
    )
}

fn build_with(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    fs: &str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fs),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(fullscreen::VS_ENTRY),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
#[path = "volume_tests.rs"]
mod tests;
