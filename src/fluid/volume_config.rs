//! What a 3D fluid is asked for, and how its volume is laid out: the
//! configuration, the named quality levels, the slice atlas and the memory
//! arithmetic. Shared by the wgpu solver (`FluidSim3d`) and the Metal one
//! (`metal::MetalFluid3d`), so it builds without wgpu.

use super::PressureSolver;
use crate::cameras::Camera;
use crate::Vector3;

/// Knobs for a 3D fluid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidConfig3d {
    /// Voxels along each axis of the solver's grid. Cubed, so this is the one
    /// number that decides everything: 64 is 262 144 voxels, 128 is two
    /// million.
    pub resolution: u32,
    /// Voxels along each axis of the dye. As in two dimensions it is usually
    /// finer than the solver's — the eye reads the ink, not the motion — but
    /// in three dimensions the cube law makes that expensive, so the default
    /// is only twice.
    pub dye_resolution: u32,
    pub density_dissipation: f32,
    pub velocity_dissipation: f32,
    /// How much of last frame's pressure to start this frame's solve from.
    pub pressure: f32,
    pub pressure_iterations: u32,
    /// How fast the fluid runs, as a multiple of real time. Applied by
    /// [`FluidSim3d::update`] and not by [`step`](FluidSim3d::step), for the
    /// reason the plane's [`time_scale`](super::FluidConfig::time_scale)
    /// gives.
    pub time_scale: f32,
    /// How the pressure equation is solved. The volume defaults to
    /// [`PressureSolver::Multigrid`] where the plane defaults to Jacobi, and
    /// the reason is the cube law — see [`QUALITY_LEVELS_3D`] and
    /// `the_v_cycle_beats_the_sweeps`.
    pub pressure_solver: PressureSolver,
    /// V-cycles a step, when the solver is the multigrid one.
    pub multigrid_cycles: u32,
    /// Smoothing sweeps either side of the recursion, per level.
    ///
    /// The textbook choice is two, and two is the default. It is here
    /// because it is most of the V-cycle's pass count — a 64³ solve with
    /// four levels spends sixteen of its forty-four passes on these — so it
    /// is the first place to look for work that is not buying anything.
    pub multigrid_smooth: u32,
    /// Stop coarsening at this many voxels a side.
    ///
    /// Each level costs a handful of render passes over a grid small enough
    /// that the arithmetic is free, which looks like pure overhead — and
    /// measured, it is not. Coarsening all the way to 4³ against stopping at
    /// 16³ is 10% of the solve for 23% of the residual
    /// (`the_v_cycle_needs_its_coarse_levels`), so the default goes all the
    /// way down. The knob is here because that balance is a property of the
    /// machine's per-pass cost, not of the fluid.
    pub multigrid_coarsest: u32,
    /// Keep the bind groups and uniform blocks between frames instead of
    /// rebuilding them.
    ///
    /// A step records about fifty passes whose descriptions are identical
    /// frame to frame, and recording them is most of what the CPU does here.
    /// Off is the way to measure that; there is no reason to run that way.
    pub cache_bindings: bool,
    /// Draw the frame rate over the picture.
    ///
    /// Off by default. The number is the volume's own render cadence,
    /// measured between calls to [`render`](FluidSim3d::render) — not the
    /// application's frame rate, which may be doing other things, and not
    /// the GPU's time in the solver, which is
    /// [`stage_ms`](FluidSim3d::stage_ms).
    pub show_fps: bool,
    /// Average the ray march over this many frames.
    ///
    /// One is off. Above that, each frame jitters its samples differently
    /// and blends into what came before with a weight of `1 / n`, so a march
    /// of 64 samples converges towards one of several hundred without ever
    /// costing more than 64 in a frame. That is the trade the big volumes
    /// need: past about 256³ the dye is finer than any affordable number of
    /// samples can resolve, and the picture is noise-limited rather than
    /// resolution-limited.
    ///
    /// It resets whenever the camera moves, so it buys nothing while you are
    /// dragging and everything the moment you stop. The fluid itself still
    /// moves, which it cannot reset on — a plume accumulated over eight
    /// frames is eight frames of motion blurred together, and that is the
    /// cost. Steady smoke takes it well; a splat landing does not.
    pub temporal_samples: u32,
    /// How far the history may stray from this frame, in standard
    /// deviations of its own neighbourhood, before it is rejected.
    ///
    /// Smaller rejects more. Measured, against a long march of the same
    /// moment — the first column is a still camera, where accumulating is
    /// supposed to help, and the second a rising plume, where it is supposed
    /// not to hurt:
    ///
    /// | clip | still | moving |
    /// |---|---|---|
    /// | 0.25 | 0.067 | 0.179 |
    /// | **0.5** | **0.024** | **0.235** |
    /// | 1.0 | 0.019 | 0.408 |
    /// | no clip | 0.019 | 1.233 |
    /// | no accumulation | 0.088 | 0.050 |
    ///
    /// Half a sigma is the knee: nearly all of the convergence, for a little
    /// over half the ghosting of one sigma. Below it the history is pinned
    /// so close to the local mean that accumulating stops buying anything.
    ///
    /// Only read when [`temporal_samples`](Self::temporal_samples) is above
    /// one.
    pub temporal_clip: f32,
    /// Step the ray march over the parts of the volume that hold nothing.
    ///
    /// Four fifths of a plume is empty, so this is most of the march. It
    /// costs two small reduction passes a frame to find the empty parts and
    /// changes the picture by a fraction of a level; set it false to measure
    /// either.
    pub empty_space_skip: bool,
    /// Pack the dye into `rg11b10ufloat` where the device allows it — half
    /// the memory and half the bandwidth of `rgba16float`, which on the
    /// largest field in the volume is the difference between one resolution
    /// level and the next. Set it false to measure what it costs; see
    /// [`FluidFormats::volume_dye`].
    pub compact_dye: bool,
    /// Vorticity confinement. The same knob as in two dimensions and the same
    /// effect: it is what decides whether this looks like smoke.
    pub curl: f32,
    /// Coefficient on the pressure gradient. 1.0 is the 2D module's default
    /// and over-relaxes by two, which compensates for an unconverged solve —
    /// see `FluidConfig::gradient_scale`.
    pub gradient_scale: f32,
    /// Radius of a splat, as a fraction of the volume.
    pub splat_radius: f32,
    /// How hard a drag pushes, for [`impulse_at_screen`](FluidSim3d::impulse_at_screen).
    ///
    /// The drag is already in world units at the depth it acts on, so this
    /// only has to turn "a fraction of the screen" into "voxels a second" —
    /// it is a feel knob, not a physical one.
    pub splat_force: f32,
    /// Buoyancy: how hard dye rises, in voxels per second squared. Smoke that
    /// does not rise reads as dust, and in three dimensions there is a clear
    /// up — which is the one force worth having that a 2D ink demo does not.
    pub buoyancy: f32,
    /// Which way buoyancy pushes. Up by default, because that is what
    /// buoyancy is — but a volume has no privileged axis, and the one case
    /// this exists for is a scene whose "up" is not the solver's: a plume
    /// along a tilted chimney, or a current in any direction at all.
    ///
    /// Normalised before use; a zero vector means no buoyancy.
    pub buoyancy_axis: [f32; 3],
    /// Ray-march steps through the volume. The cost of the picture, more or
    /// less entirely.
    pub march_steps: u32,
    /// How opaque the dye is.
    pub absorption: f32,
    /// How brightly it glows.
    pub emission: f32,
    pub exposure: f32,
    /// What is behind the volume when it is not `transparent`.
    pub background: [f32; 3],
    /// Keep the dye's own alpha, premultiplied, instead of compositing it
    /// onto [`background`](Self::background) — for drawing the volume over a
    /// scene rather than on its own.
    pub transparent: bool,
    /// The box the volume occupies, in world units.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Seed for splat positions and hues, so a headless render repeats.
    pub seed: u64,
}

/// A named size for a volume: what to allocate and how hard to work it.
///
/// Separate from the 2D [`FluidQuality`](crate::fluid::FluidQuality) because
/// the numbers mean different things. In two dimensions doubling the dye grid
/// costs four times the memory and the simulation stays interactive; here it
/// costs eight, and the march has to take more steps to see the extra detail,
/// so a level that reads as one notch finer is nearer three times the frame.
#[derive(Clone, Copy, Debug)]
pub struct FluidQuality3d {
    pub name: &'static str,
    /// Velocity and pressure grid, in voxels a side.
    pub resolution: u32,
    /// Dye grid, in voxels a side.
    pub dye_resolution: u32,
    pub pressure_iterations: u32,
    /// Samples a ray takes crossing the volume.
    pub march_steps: u32,
}

/// The volume's quality levels, coarsest first.
///
/// The names match the 2D levels so one dropdown can drive both, but the
/// numbers are the volume's own. Measured on an M4 Pro with
/// `examples/fluid3d_bench.rs`, marching into 1280x720, as step + march:
///
/// | level | solver | dye | memory | frame |
/// |---|---|---|---|---|
/// | low | 24³ | 48³ | 2 MB | 2.5 ms |
/// | medium | 40³ | 80³ | 10 MB | 3.0 ms |
/// | high | 64³ | 128³ | 41 MB | 3.2 ms |
/// | ultra | 96³ | 192³ | 139 MB | 3.9 ms |
/// | extreme | 128³ | 256³ | 324 MB | 5.2 ms |
///
/// Memory is the part to look at before `extreme`: three double-buffered
/// vector fields and a scalar pair, all at the cube of the side. The frame
/// grows far more slowly than the memory does, because half of it is the
/// march and the march is set by the pixels on the screen rather than by the
/// voxels behind them.
pub const QUALITY_LEVELS_3D: &[FluidQuality3d] = &[
    FluidQuality3d {
        name: "low",
        resolution: 24,
        dye_resolution: 48,
        pressure_iterations: 16,
        march_steps: 48,
    },
    FluidQuality3d {
        name: "medium",
        resolution: 40,
        dye_resolution: 80,
        pressure_iterations: 18,
        march_steps: 64,
    },
    // What `FluidConfig3d::default` gives.
    FluidQuality3d {
        name: "high",
        resolution: 64,
        dye_resolution: 128,
        pressure_iterations: 20,
        march_steps: 96,
    },
    FluidQuality3d {
        name: "ultra",
        resolution: 96,
        dye_resolution: 192,
        pressure_iterations: 24,
        march_steps: 128,
    },
    FluidQuality3d {
        name: "extreme",
        resolution: 128,
        dye_resolution: 256,
        pressure_iterations: 28,
        march_steps: 192,
    },
];

/// Look a volume quality level up by name.
pub fn quality3d(name: &str) -> Option<&'static FluidQuality3d> {
    QUALITY_LEVELS_3D.iter().find(|q| q.name == name)
}

/// Every volume quality level's name, coarsest first.
pub fn quality3d_names() -> impl Iterator<Item = &'static str> {
    QUALITY_LEVELS_3D.iter().map(|q| q.name)
}

impl FluidConfig3d {
    /// Apply a [`FluidQuality3d`], leaving every other setting alone.
    ///
    /// ```no_run
    /// use threers::fluid::{quality3d, FluidConfig3d};
    /// let config = FluidConfig3d {
    ///     curl: 26.0,
    ///     ..FluidConfig3d::default().with_quality(quality3d("ultra").unwrap())
    /// };
    /// ```
    pub fn with_quality(mut self, quality: &FluidQuality3d) -> Self {
        self.resolution = quality.resolution;
        self.dye_resolution = quality.dye_resolution;
        self.pressure_iterations = quality.pressure_iterations;
        self.march_steps = quality.march_steps;
        self
    }
}

impl Default for FluidConfig3d {
    fn default() -> Self {
        Self {
            resolution: 64,
            dye_resolution: 128,
            density_dissipation: 0.4,
            velocity_dissipation: 0.2,
            // All of it, not the 0.8 the plane inherits from the original.
            // More carry-over converges better — 0.269 of a splat's
            // divergence left after eight frames against 0.281 — and the
            // drift the damping is for does not happen here: see
            // `the_pressure_carry_over_is_damped_against_drift`, which runs
            // nine hundred frames with and without a coarse grid to anchor
            // the constant mode and finds both bounded.
            pressure: 1.0,
            pressure_iterations: 20,
            time_scale: 1.0,
            pressure_solver: PressureSolver::Multigrid,
            multigrid_cycles: 1,
            multigrid_smooth: 2,
            multigrid_coarsest: 4,
            cache_bindings: true,
            show_fps: false,
            temporal_samples: 1,
            temporal_clip: 0.5,
            empty_space_skip: true,
            // Off, and measured: it halves the dye's memory, and on an M4 Pro
            // it also doubles the frame — sampling the packed format is the
            // cost, not the bandwidth it saves. See `FluidFormats::volume_dye`.
            compact_dye: false,
            curl: 12.0,
            gradient_scale: 1.0,
            splat_radius: 0.6,
            splat_force: 6.0,
            buoyancy: 0.0,
            buoyancy_axis: [0.0, 1.0, 0.0],
            march_steps: 96,
            absorption: 28.0,
            emission: 1.2,
            exposure: 1.0,
            background: [0.015, 0.015, 0.025],
            transparent: false,
            bounds_min: [-0.5, -0.5, -0.5],
            bounds_max: [0.5, 0.5, 0.5],
            seed: 0x5EED_F101_D1CE,
        }
    }
}

/// How a volume's z-slices are laid out across a 2D texture.
///
/// Square-ish, so the atlas is as close to square as the slice count allows —
/// a long thin texture wastes more of the device's maximum dimension than it
/// needs to and is likelier to hit it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atlas {
    /// Voxels along each axis.
    pub grid: u32,
    /// Tiles across and down.
    pub tiles: (u32, u32),
}

impl Atlas {
    pub fn new(grid: u32) -> Self {
        let grid = grid.max(2);
        let across = (grid as f32).sqrt().ceil() as u32;
        let down = grid.div_ceil(across);
        Self {
            grid,
            tiles: (across, down),
        }
    }

    /// Size of the 2D texture holding it.
    pub fn size(&self) -> (u32, u32) {
        (self.grid * self.tiles.0, self.grid * self.tiles.1)
    }

    /// The largest grid whose atlas fits in `max` on both axes.
    pub fn largest_within(mut grid: u32, max: u32) -> Self {
        loop {
            let a = Self::new(grid);
            let (w, h) = a.size();
            if (w <= max && h <= max) || grid <= 8 {
                return a;
            }
            grid /= 2;
        }
    }
}

/// Below this many voxels a side there is nothing left for a seven-point
/// stencil to average that clamp-to-edge is not already giving it.
pub(crate) const MG_COARSEST_3D: u32 = 4;

/// What a volume of this size allocates, in bytes, given the bytes a texel of
/// each kind of field takes: `[vector, dye, scalar]`.
///
/// The atlas rounds a cube up to a rectangle, so this is what the textures
/// actually cost rather than `n³ × bytes`. Both backends allocate exactly this
/// set of fields, and their `memory_bytes` are held to it.
pub(crate) fn memory_bytes(resolution: u32, dye_resolution: u32, bytes: [u64; 3]) -> u64 {
    let texels = |grid: u32| {
        let a = Atlas::new(grid);
        let (w, h) = a.size();
        w as u64 * h as u64
    };
    let [vec3, dye_fmt, scalar] = bytes;
    let sim = texels(resolution.max(2));
    let dye = texels(dye_resolution.max(2));

    // velocity ×2, pressure ×2, divergence, curl, residual.
    let mut total = sim * (2 * vec3 + 2 * scalar + scalar + vec3 + scalar);
    total += dye * 2 * dye_fmt;
    // The coarse chain: each level is an eighth of the one above, four fields
    // apiece.
    let mut n = resolution.max(2);
    let mut levels = 0;
    while n / 2 >= MG_COARSEST_3D && levels < 6 {
        n /= 2;
        levels += 1;
        total += texels(n) * 4 * scalar;
    }
    // The occupancy pyramid: half the dye's side, then a quarter.
    total += (texels(dye_resolution / 2) + texels(dye_resolution / 4)) * scalar;
    total
}

/// One field's atlas, mirroring `Grid` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GridUniform {
    pub(crate) size: [f32; 4],
    pub(crate) tiles: [f32; 4],
}

impl From<Atlas> for GridUniform {
    fn from(a: Atlas) -> Self {
        let n = a.grid as f32;
        let (tx, ty) = (a.tiles.0 as f32, a.tiles.1 as f32);
        Self {
            size: [n, n, n, 1.0 / n],
            tiles: [tx, ty, 1.0 / tx, 1.0 / ty],
        }
    }
}

/// The uniform block, mirroring `Vol` in `volume_shader.rs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vol {
    pub(crate) out_grid: GridUniform,
    pub(crate) a_grid: GridUniform,
    pub(crate) b_grid: GridUniform,
    pub(crate) texel: [f32; 4],
    pub(crate) point: [f32; 4],
    pub(crate) color: [f32; 4],
    pub(crate) params: [f32; 4],
    pub(crate) bounds_min: [f32; 4],
    pub(crate) bounds_max: [f32; 4],
    pub(crate) march: [f32; 4],
    pub(crate) background: [f32; 4],
    pub(crate) cam_pos: [f32; 4],
    pub(crate) inv_view_proj: [f32; 16],
    /// Last frame's view-projection, for the resolve to reproject through.
    pub(crate) prev_view_proj: [f32; 16],
}

/// Where a push from the screen lands in the volume, and the velocity it adds:
/// `uv` and `drag` in 0..1 of the view, origin top left. `None` when the ray
/// misses the box. See `FluidSim3d::impulse_at_screen` for why it lands on the
/// plane through the centre facing the camera.
pub(crate) fn impulse(
    config: &FluidConfig3d,
    sim_grid: u32,
    camera: &dyn Camera,
    uv: [f32; 2],
    drag: [f32; 2],
    strength: f32,
) -> Option<([f32; 3], [f32; 3])> {
    let inv = camera
        .projection_matrix()
        .multiply(&camera.view_matrix())
        .invert();
    // uv, top-left origin, to a world ray.
    let ray = |uv: [f32; 2]| -> (Vector3, Vector3) {
        let ndc =
            |z: f32| Vector3::new(uv[0] * 2.0 - 1.0, 1.0 - uv[1] * 2.0, z).apply_matrix4(&inv);
        let near = ndc(0.0);
        let far = ndc(1.0);
        (near, (far - near).normalize())
    };
    let c = *config;
    let centre = Vector3::new(
        0.5 * (c.bounds_min[0] + c.bounds_max[0]),
        0.5 * (c.bounds_min[1] + c.bounds_max[1]),
        0.5 * (c.bounds_min[2] + c.bounds_max[2]),
    );
    let eye = camera.position();
    let normal = (centre - eye).normalize();
    // Where a ray meets the plane through the centre facing the camera.
    let on_plane = |uv: [f32; 2]| -> Option<Vector3> {
        let (origin, dir) = ray(uv);
        let denom = dir.dot(normal);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (centre - origin).dot(normal) / denom;
        if t <= 0.0 {
            return None;
        }
        Some(origin + dir * t)
    };
    let hit = on_plane(uv)?;
    let size = Vector3::new(
        c.bounds_max[0] - c.bounds_min[0],
        c.bounds_max[1] - c.bounds_min[1],
        c.bounds_max[2] - c.bounds_min[2],
    );
    let at = [
        (hit.x - c.bounds_min[0]) / size.x,
        (hit.y - c.bounds_min[1]) / size.y,
        (hit.z - c.bounds_min[2]) / size.z,
    ];
    if at.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return None;
    }
    // The drag, as a world direction at the same depth, in the solver's
    // voxels a second.
    let moved = on_plane([uv[0] + drag[0], uv[1] + drag[1]])
        .map(|to| to - hit)
        .unwrap_or(Vector3::ZERO);
    let scale = strength * sim_grid as f32;
    Some((
        at,
        [
            moved.x / size.x * scale,
            moved.y / size.y * scale,
            moved.z / size.z * scale,
        ],
    ))
}

/// `count` puffs at random places in the middle of the box, random hues,
/// pushed in random directions: `(at, velocity, colour)`. One sequence for
/// both backends, so a shared seed gives the same burst.
pub(crate) fn random_splats(
    rng: &mut super::Rng,
    count: u32,
) -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
    let mut batch = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let hue = rng.next_f32();
        let c = super::hsv_to_rgb(hue, 1.0, 1.0);
        let at = [
            0.2 + 0.6 * rng.next_f32(),
            0.2 + 0.6 * rng.next_f32(),
            0.2 + 0.6 * rng.next_f32(),
        ];
        // A direction, uniformly over the sphere, times a speed.
        //
        // Not three independent components: that is uniform over a
        // *cube*, which is denser towards its corners, so the pushes
        // would favour the diagonals. And the y term used to be
        // `260 * rand` rather than `rand - 0.5`, which is to say it was
        // never negative — every random splat went up, which is what
        // this looked like from outside.
        let z = 2.0 * rng.next_f32() - 1.0;
        let theta = std::f32::consts::TAU * rng.next_f32();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let speed = 120.0 + 180.0 * rng.next_f32();
        let v = [r * theta.cos() * speed, z * speed, r * theta.sin() * speed];
        batch.push((at, v, [c[0] * 1.5, c[1] * 1.5, c[2] * 1.5]));
    }
    batch
}

/// The largest `(solver, dye)` whose fields fit `budget` bytes, `ratio` dye
/// voxels to a solver voxel a side, at `bytes` per texel as in
/// [`memory_bytes`]. Eight at a time: the atlas pads a cube into a rectangle
/// and the waste is worst at awkward sizes.
pub(crate) fn within_bytes(budget: u64, ratio: u32, bytes: [u64; 3]) -> (u32, u32) {
    let ratio = ratio.max(1);
    let mut best = (8u32, 8 * ratio);
    for solver in (8..=512).step_by(8) {
        let dye = solver * ratio;
        if memory_bytes(solver, dye, bytes) > budget {
            break;
        }
        best = (solver, dye);
    }
    best
}
