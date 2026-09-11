//! Lattice generators — periodic infill as `BufferGeometry`.
//!
//! Five families, one pipeline:
//!
//! | Family | What it is | Count |
//! |--------|------------|-------|
//! | [`Tpms`] | Triply periodic minimal surfaces — gyroid, Schwarz P and D, Neovius, I-WP, Fischer–Koch S, Lidinoid, split P | 8 |
//! | [`Strut`] | Beam cells — simple cubic, BCC, BCC-Z, FCC, octet truss, diamond, Kelvin | 7 |
//! | [`Infill`] | What a slicer draws — rectilinear (turning and aligned), grid, triangles, tri-hexagon, honeycomb, cubic, quarter cubic, concentric | 9 |
//! | [`Cuboct`] | Face-connected cuboctahedra voxels — rigid, compliant, auxetic, chiral (Jenett et al. 2020) | 5 |
//! | [`Stochastic`] | Foams — Voronoi struts and walls, and four classes of spinodal random field | 6 |
//!
//! Each is evaluated as a scalar field that is positive inside the solid,
//! sampled on a regular grid, and contoured by marching cubes. Because the
//! field is intersected with the bounding box before it is sampled, the mesh
//! closes on itself: it is indexed, welded, and every edge is shared by exactly
//! two triangles, which is what a slicer or a boolean kernel needs.
//!
//! Every generator takes the same three numbers a slicer asks for — cell size,
//! line width, density — and any of them can be poured into a [`Region`]
//! rather than a box, graded across the part, skinned, or sectioned.
//!
//! ```
//! use threers::{Lattice, LatticeKind, Tpms, Vector3};
//!
//! // A 20 mm cube of gyroid, 5 mm cells, 0.8 mm walls.
//! let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .size(Vector3::new(20.0, 20.0, 20.0))
//!     .cell_size(Vector3::new(5.0, 5.0, 5.0))
//!     .thickness(0.8)
//!     .build();
//! assert!(geom.index.is_some());
//! ```
//!
//! # Thickness, or density
//!
//! [`thickness`](Lattice::thickness) is a length: wall thickness for
//! [`LatticeStyle::Sheet`], strut diameter for a beam lattice. When what you
//! actually care about is how much material ends up in the part, ask for that
//! instead and let [`fit_relative_density`](Lattice::fit_relative_density)
//! solve for the thickness that hits it.
//!
//! ```
//! # use threers::{Lattice, LatticeKind, Strut, Vector3};
//! let lattice = Lattice::new(LatticeKind::Strut(Strut::Octet))
//!     .size(Vector3::new(10.0, 10.0, 10.0))
//!     .cells([3, 3, 3])
//!     .fit_relative_density(0.25);
//! assert!((lattice.relative_density() - 0.25).abs() < 0.01);
//! ```
//!
//! # Filling a shape
//!
//! A lattice fills a box unless told otherwise. [`fill`](Lattice::fill) takes a
//! [`Region`] instead — a sphere, a cylinder, a tube swept along a curve, a
//! triangle mesh, an OpenSCAD model, or any union, intersection or difference
//! of those. The region's own bounds size the sample grid, so there is nothing
//! else to say, and the mesh closes over the cut so the result is still one
//! watertight shell.
//!
//! ```
//! # #[cfg(feature = "openscad")] {
//! use threers::{Lattice, LatticeKind, Region, Tpms, Vector3};
//!
//! let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .fill(Region::scad("difference(){ cube(40, center=true); sphere(24); }").unwrap())
//!     .cell_size(Vector3::new(6.0, 6.0, 6.0))
//!     .thickness(1.0)
//!     .build();
//! # let _ = geom;
//! # }
//! ```
//!
//! Four more knobs shape what comes out:
//!
//! | Call | What it does |
//! |------|--------------|
//! | [`skin`](Lattice::skin) | a solid wall on the fill's surface, unioned with the lattice — a printed part's perimeters and its infill in one mesh |
//! | [`clip`](Lattice::clip) | cuts the finished part, skin and all, so you can see inside a section of it |
//! | [`grade`](Lattice::grade) | scales thickness per point, so a part is dense where it is loaded and light where it is not |
//! | [`conform`](Lattice::conform) | lays the cells out in a mapped space, so they follow a curved part instead of being cut by it |
//!
//! ```
//! use threers::{Lattice, LatticeKind, Region, Tpms, Vector3};
//!
//! let geom = Lattice::new(LatticeKind::Tpms(Tpms::Diamond))
//!     .fill(Region::sphere(Vector3::ZERO, 2.0))
//!     .cells([2, 2, 2])
//!     .thickness(0.25)
//!     // Twice the wall on the left, half on the right.
//!     .grade(|p| 1.0 + p.x * 0.25)
//!     .skin(0.2)
//!     .build();
//! # assert!(geom.index.is_some());
//! ```
//!
//! # Conforming, rather than trimming
//!
//! [`fill`](Lattice::fill) cuts the lattice where the part ends. On a flat box
//! that is right; on a curved shell it leaves severed struts at the surface and
//! whatever fraction of a cell happened to fit. [`conform`](Lattice::conform)
//! maps the point into cell space first, so a tiling can close around a nozzle
//! or stack whole layers through a wall that curves — see [`Conform`].
//!
//! # Grading on something real
//!
//! [`grade`](Lattice::grade) takes a closure, which is not the form a solver
//! result, a scan or a sensor sweep arrives in. [`Field`] is the adapter:
//! build it from a grid, a function or scattered points, and it interpolates,
//! rescales and hands back the closure. [`CuboctFrame::stress_field`] closes
//! the loop for beam lattices — solve the block, grade the lattice on what the
//! solve said.
//!
//! # What it comes out as
//!
//! Two questions survive the build, and both have answers here rather than
//! opinions:
//!
//! - [`metrics`](Lattice::metrics) — porosity, internal surface area, pore
//!   size, ligament thickness, hydraulic diameter and a permeability estimate.
//!   The numbers a heat exchanger, a filter or a scaffold is specified by.
//! - [`homogenize`](Lattice::homogenize) — the cell's effective stiffness
//!   tensor, and the moduli, shear moduli, Poisson ratios and anisotropy read
//!   off it. What to hand a solver that is modelling the lattice as a solid.
//! - [`conductivity`](Lattice::conductivity) — and with it the electrical,
//!   diffusive and dielectric answers, which are the same equation.
//! - [`strength`](Lattice::strength) — where it gives way, from how unevenly
//!   the load spreads inside the cell.
//!
//! ```
//! # use threers::{Lattice, LatticeKind, Tpms, Vector3};
//! let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .size(Vector3::new(20.0, 20.0, 20.0))
//!     .cells([4, 4, 4])
//!     .fit_relative_density(0.25);
//!
//! let m = lattice.metrics();
//! assert!(m.porosity > 0.7 && m.pore_diameter > 0.0);
//! ```
//!
//! # Resolution
//!
//! Everything here is cubic in [`resolution`](Lattice::resolution), so the
//! useful question is not "how fine can I make it" but "how fine does this
//! lattice need". [`wall_samples`](Lattice::wall_samples) answers it: below
//! about two samples across the wall, marching cubes starts missing the wall
//! between samples and the mesh comes out as gravel — silently, because a field
//! never sampled inside a wall looks exactly like one with no wall there.
//! [`resolve_walls`](Lattice::resolve_walls) raises the resolution until that
//! number is met, which costs far less than paying the worst case everywhere:
//! at 25 % density a gyroid needs a third the sampling a Lidinoid does.
//!
//! [`max_samples`](Lattice::max_samples) is the backstop. Rather than attempt
//! an allocation that will not fit, the resolution is reduced until it does —
//! and `wall_samples` will then say so.
//!
//! Sampling parallelises with the crate's `parallel` feature (rayon, native
//! only); the gallery example builds about 3× faster with it on. Contouring is
//! still one thread. Results are identical either way: every sample has a fixed
//! address in the grid, so a build flag cannot change the geometry.

mod automation;
mod conform;
mod cuboct;
mod field;
mod frame;
// A `wgpu::Device` is neither `Send` nor `Sync` in the browser — it holds an
// `Rc` — so the process-wide device this keeps cannot exist there. `Solver::Gpu`
// falls back to the CPU on wasm, which is what it does anywhere without an
// adapter.
#[cfg(not(target_arch = "wasm32"))]
mod gpu_solve;
mod homogenize;
mod infill;
mod mesher;
mod metrics;
mod region;
mod stochastic;
mod strut;
mod tpms;
mod voxel;

pub use automation::{CuboctAssemblyPlan, CuboctAssemblyStep};
pub use conform::Conform;
pub use cuboct::{ChiralRule, Cuboct, Hand};
pub use field::Field;
pub use frame::{CuboctFrame, FrameMaterial, FrameResponse};
pub use homogenize::{
    homogenize, homogenize_conduction, homogenize_strength, Conductivity, SolidMaterial, Solver,
    Stiffness, Strength,
};
pub use infill::Infill;
pub use metrics::LatticeMetrics;
pub use region::Region;
pub use stochastic::Stochastic;
pub use strut::{Segment, Strut};
pub use tpms::Tpms;
pub use voxel::{CuboctAssembly, CuboctJoint, CuboctJointKind};

use crate::core::BufferGeometry;
use crate::math::{Box3, Vector3};
use mesher::IsoGrid;
use std::f32::consts::TAU;

/// Which periodic cell to fill space with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatticeKind {
    /// A triply periodic minimal surface, thickened into a wall or filled on
    /// one side.
    Tpms(Tpms),
    /// A cell of beams, thickened into round struts.
    Strut(Strut),
    /// A slicer's infill pattern, drawn as walls.
    Infill(Infill),
    /// A face-connected cuboctahedron voxel — rigid, compliant, auxetic, or
    /// chiral. Beam *shape* is [`Lattice::shape`]; chirality can be programmed
    /// per cell with [`Lattice::program`] and [`Lattice::chiral_rule`].
    Cuboct(Cuboct),
    /// A foam rather than a tiling: Voronoi struts or walls, or a spinodal
    /// random field. Randomness is set by [`Lattice::seed`] and, for the
    /// Voronoi cells, [`Lattice::jitter`].
    Stochastic(Stochastic),
}

impl LatticeKind {
    /// Every generator in every family, in declaration order.
    pub fn all() -> Vec<LatticeKind> {
        Tpms::ALL
            .into_iter()
            .map(LatticeKind::Tpms)
            .chain(Strut::ALL.into_iter().map(LatticeKind::Strut))
            .chain(Infill::ALL.into_iter().map(LatticeKind::Infill))
            .chain(Cuboct::ALL.into_iter().map(LatticeKind::Cuboct))
            .chain(Stochastic::ALL.into_iter().map(LatticeKind::Stochastic))
            .collect()
    }

    /// Lower-case identifier, for logs and CLI arguments.
    pub fn name(self) -> &'static str {
        match self {
            LatticeKind::Tpms(t) => t.name(),
            LatticeKind::Strut(s) => s.name(),
            LatticeKind::Infill(i) => i.name(),
            LatticeKind::Cuboct(c) => c.name(),
            LatticeKind::Stochastic(s) => s.name(),
        }
    }

    /// The generator with this [`name`](Self::name), from any of the three
    /// families — so a `--lattice gyroid` flag needs no table of its own.
    ///
    /// Names are unique across the families, which is why
    /// [`Strut::Cubic`] answers to `simple-cubic`: a slicer means the
    /// cube-on-a-corner pattern by "cubic", and that is [`Infill::Cubic`].
    pub fn from_name(name: &str) -> Option<LatticeKind> {
        Tpms::from_name(name)
            .map(LatticeKind::Tpms)
            .or_else(|| Strut::from_name(name).map(LatticeKind::Strut))
            .or_else(|| Infill::from_name(name).map(LatticeKind::Infill))
            .or_else(|| Cuboct::from_name(name).map(LatticeKind::Cuboct))
            .or_else(|| Stochastic::from_name(name).map(LatticeKind::Stochastic))
    }
}

/// What part of a [`Tpms`] level set becomes solid.
///
/// Ignored by [`LatticeKind::Strut`], [`LatticeKind::Infill`],
/// [`LatticeKind::Cuboct`] and the two Voronoi cells, which are always the
/// solid around their beams or walls. The spinodal cells are level sets like a
/// TPMS, and answer to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LatticeStyle {
    /// A wall of the given thickness centred on the surface, with open void on
    /// both sides. Two independent channel networks, no closed cells, and the
    /// most surface area per gram — the usual choice for infill, heat
    /// exchangers and scaffolds.
    #[default]
    Sheet,
    /// One of the two labyrinths, filled solid, its boundary offset outwards by
    /// half the thickness. Thickness `0` gives the bare minimal surface, so
    /// about half the volume; negative values thin it below that. Stiffer than
    /// a sheet of the same mass, and it leaves a single connected void.
    Solid,
}

/// Samples of overhang on each axis: two either side of the bounds so the box's
/// own faces are contoured with a live gradient, plus the two the half-step
/// offset needs.
const PADDING: usize = 6;

/// The smallest grid an axis can be reduced to — the padding alone.
const MIN_DIM: usize = PADDING;

/// How the cell size is pinned down: directly, or as a count across the bounds.
#[derive(Clone, Copy, Debug)]
enum Spacing {
    Size(Vector3),
    Count([usize; 3]),
}

/// A lattice, configured and then built.
///
/// See the [module docs](self) for the full picture.
pub struct Lattice<'a> {
    kind: LatticeKind,
    style: LatticeStyle,
    bounds: Box3,
    spacing: Spacing,
    thickness: f32,
    resolution: usize,
    max_samples: usize,
    skin: f32,
    /// Whether the caller pinned the bounds, so [`Lattice::fill`] knows not to
    /// take them from the region instead. Tracked rather than inferred from
    /// call order, so the two can be written either way round.
    pinned_bounds: bool,
    /// `Sync + Send` so the sample grid can be filled from several threads —
    /// required whether or not the `parallel` feature is on, so that turning it
    /// on is never a breaking change to calling code.
    grade: Option<std::sync::Arc<dyn Fn(Vector3) -> f32 + Sync + Send + 'a>>,
    trim: Option<std::sync::Arc<dyn Fn(Vector3) -> f32 + Sync + Send + 'a>>,
    clip: Option<std::sync::Arc<dyn Fn(Vector3) -> f32 + Sync + Send + 'a>>,
    /// Cuboct beam shape as a fraction of cell pitch: corrugation amplitude,
    /// reentrant indent, or chiral radius. Ignored by the other families.
    shape: f32,
    /// Per-cell cuboct type. When set, overrides [`LatticeKind::Cuboct`] at
    /// each integer cell. Ignored by the other families.
    program: Option<crate::geometries::lattice::voxel::CuboctProgram<'a>>,
    chiral_rule: Option<ChiralRule>,
    /// Where the cells are laid out, when that is not world space.
    conform: Option<Conform<'a>>,
    /// Which foam. Ignored by the periodic families, which have nothing to
    /// randomise.
    seed: u32,
    /// How far a Voronoi seed may wander from its cell centre, 0 to 1.
    jitter: f32,
    /// Where the homogenisation solves run. Nothing else reads it.
    solver: Solver,
}

impl<'a> Lattice<'a> {
    /// A lattice of `kind` filling the unit cube at the origin: 4 cells per
    /// axis, walls a twentieth of the box across, 16 samples per cell.
    pub fn new(kind: LatticeKind) -> Self {
        Self {
            kind,
            style: LatticeStyle::default(),
            bounds: Box3::from_center_and_size(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)),
            spacing: Spacing::Count([4, 4, 4]),
            thickness: 0.05,
            resolution: 16,
            max_samples: 8_000_000,
            skin: 0.0,
            pinned_bounds: false,
            grade: None,
            trim: None,
            clip: None,
            shape: 0.15,
            program: None,
            chiral_rule: None,
            conform: None,
            seed: 0,
            jitter: 1.0,
            solver: Solver::default(),
        }
    }

    /// Fill an explicit box.
    pub fn bounds(mut self, bounds: Box3) -> Self {
        self.bounds = bounds;
        self.pinned_bounds = true;
        self
    }

    /// Fill a box of this size, centred on the origin.
    pub fn size(mut self, size: Vector3) -> Self {
        self.bounds = Box3::from_center_and_size(Vector3::ZERO, size);
        self.pinned_bounds = true;
        self
    }

    /// Fill a [`Region`] instead of a box — a sphere, a swept curve, a mesh, an
    /// OpenSCAD model, or any combination of those.
    ///
    /// The region's own bounds become the sample domain, so this is the only
    /// call needed; [`bounds`](Self::bounds) or [`size`](Self::size) override
    /// that, whichever order they are written in. The lattice is cut where the
    /// region ends and the mesh closes over the cut, so the result is still one
    /// watertight shell.
    ///
    /// ```
    /// use threers::{Lattice, LatticeKind, Region, Strut, Vector3};
    ///
    /// let geom = Lattice::new(LatticeKind::Strut(Strut::Octet))
    ///     .fill(Region::cylinder(
    ///         Vector3::new(0.0, -8.0, 0.0),
    ///         Vector3::new(0.0, 8.0, 0.0),
    ///         5.0,
    ///     ))
    ///     .cell_size(Vector3::new(4.0, 4.0, 4.0))
    ///     .thickness(0.8)
    ///     .build();
    /// assert!(geom.index.is_some());
    /// ```
    pub fn fill(mut self, region: Region<'a>) -> Self {
        let bounds = region.bounds();
        // An intersection of two regions that do not overlap has inverted
        // bounds. Taking them would give a grid with negative extents; leaving
        // the ones already set gives an empty mesh, which is the right answer
        // for a region with nothing in it.
        if !self.pinned_bounds && !bounds.is_empty() {
            self.bounds = bounds;
        }
        self.trim = Some(region.into_field());
        self
    }

    /// Wrap the fill in a solid skin of this thickness, measured inwards from
    /// the surface — a printed part's perimeters and its infill, in one mesh.
    ///
    /// The skin follows whatever is being filled: the bounding box by default,
    /// or the [`Region`] given to [`fill`](Self::fill). The lattice is unioned
    /// with it rather than cut by it, so the two bond where they meet and the
    /// result stays a single shell.
    ///
    /// It counts towards [`relative_density`](Self::relative_density), which
    /// means [`fit_relative_density`](Self::fit_relative_density) sizes the
    /// lattice around it — ask for 20 % on a part whose skin is already 15 %
    /// and the infill is what makes up the difference.
    pub fn skin(mut self, thickness: f32) -> Self {
        self.skin = thickness.max(0.0);
        self
    }

    /// Set the period of the cell as a length per axis.
    ///
    /// Anisotropic sizes stretch the cell rather than clipping it, so a lattice
    /// can be made compliant along one axis without changing its topology.
    ///
    /// The exceptions are the [`Infill`] patterns built from regular polygons —
    /// triangles, tri-hexagon, honeycomb, and the two cubic ones. A regular
    /// hexagon does not tile a rectangle, so those keep their proportions and
    /// take their spacing from `cell.x` alone.
    pub fn cell_size(mut self, cell: Vector3) -> Self {
        self.spacing = Spacing::Size(cell);
        self
    }

    /// Set the period as a number of cells across the bounds.
    pub fn cells(mut self, cells: [usize; 3]) -> Self {
        self.spacing = Spacing::Count(cells);
        self
    }

    /// Cuboct beam *path*, as a fraction of cell pitch.
    ///
    /// | Cell | Meaning |
    /// |------|---------|
    /// | [`Cuboct::Compliant`] | corrugation amplitude \(a/P\) |
    /// | [`Cuboct::Auxetic`] | reentrant indent \(d/P\) |
    /// | [`Cuboct::ChiralCw`] / [`Cuboct::ChiralCcw`] | pinwheel radius \(r/P\) |
    /// | [`Cuboct::Rigid`] | unused |
    ///
    /// The other families ignore it. [`thickness`](Self::thickness) is still
    /// the strut diameter. Default `0.15`, the paper's mid-range for all three
    /// shaped cells.
    pub fn shape(mut self, shape: f32) -> Self {
        self.shape = shape.max(0.0);
        self
    }

    /// Pick a [`Cuboct`] cell per integer voxel, for heterogeneous and chiral
    /// lattices.
    ///
    /// The closure is `(i, j, k)` in cell index, not world units. Pair with
    /// [`chiral_rule`](Self::chiral_rule) to orient the faces of a chiral
    /// column so neighbouring pinwheels do not cancel. Ignored unless the
    /// lattice kind is cuboct.
    ///
    /// ```
    /// use threers::{ChiralRule, Cuboct, Lattice, LatticeKind, Vector3};
    ///
    /// let n = 3i32;
    /// let m = 12i32;
    /// let geom = Lattice::new(LatticeKind::Cuboct(Cuboct::ChiralCcw))
    ///     .size(Vector3::new(n as f32, n as f32, m as f32))
    ///     .cells([n as usize, n as usize, m as usize])
    ///     .thickness(0.08)
    ///     .chiral_rule(ChiralRule::R1)
    ///     .program(move |_, _, k| Cuboct::column_half(k, m))
    ///     .build();
    /// # let _ = geom;
    /// ```
    pub fn program(mut self, program: impl Fn(i32, i32, i32) -> Cuboct + Sync + Send + 'a) -> Self {
        self.program = Some(Box::new(program));
        self
    }

    /// Orient chiral faces by Jenett et al.'s column rules. No effect on rigid,
    /// compliant, or auxetic cells. [`R1`](ChiralRule::R1) for odd widths,
    /// [`R2`](ChiralRule::R2) for even; the width is the cell count along x.
    pub fn chiral_rule(mut self, rule: ChiralRule) -> Self {
        self.chiral_rule = Some(rule);
        self
    }

    /// Lay the cells out in a mapped space instead of world space, so they
    /// follow the part rather than being cut by it — see [`Conform`].
    ///
    /// The map applies to the lattice alone. The [`fill`](Self::fill),
    /// [`skin`](Self::skin) and [`clip`](Self::clip) all stay in world space,
    /// which is where the part is.
    ///
    /// [`cell_size`](Self::cell_size) is then a size in *mapped* units, and
    /// [`cells`](Self::cells) — a count across the world-space bounds — no
    /// longer means much; set the size.
    pub fn conform(mut self, conform: Conform<'a>) -> Self {
        self.conform = Some(conform);
        self
    }

    /// Where [`homogenize`](Self::homogenize), [`conductivity`](Self::conductivity)
    /// and [`strength`](Self::strength) run their linear solves.
    ///
    /// Nothing else reads it: the geometry, the mesh and the metrics are
    /// unaffected, and remain bit-for-bit what they were. See [`Solver`].
    ///
    /// ```no_run
    /// use threers::{Lattice, LatticeKind, Solver, Tpms, Vector3};
    ///
    /// let c = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    ///     .size(Vector3::new(10.0, 10.0, 10.0))
    ///     .cells([1, 1, 1])
    ///     .fit_relative_density(0.3)
    ///     .solver(Solver::Gpu)
    ///     .homogenize(32);
    /// // And whether it got one.
    /// assert_eq!(c.solver, Solver::Gpu);
    /// ```
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Which foam. Any two seeds give two different lattices of the same
    /// statistics; the same seed always gives the same lattice.
    ///
    /// Ignored by the periodic families.
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// How far a Voronoi seed may wander from its cell's centre, as a fraction
    /// of the cell. Clamped to `0..=1`; the default is 1.
    ///
    /// Zero puts every seed on a regular grid, which makes
    /// [`Stochastic::Voronoi`] the edges of a cube and
    /// [`Stochastic::VoronoiWall`] a plain box grid — a useful thing to sweep
    /// towards, because it is the ordered end of the same family. Ignored by
    /// everything but the Voronoi cells.
    pub fn jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Wall thickness ([`LatticeStyle::Sheet`]) or strut diameter
    /// ([`LatticeKind::Strut`] / [`LatticeKind::Cuboct`]), in world units.
    ///
    /// For [`LatticeStyle::Solid`] it is how far the surface is pushed into the
    /// void, and may be negative.
    pub fn thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness;
        self
    }

    /// Which side of a [`Tpms`] level set is solid. No effect on beam lattices.
    pub fn style(mut self, style: LatticeStyle) -> Self {
        self.style = style;
        self
    }

    /// Samples per cell per axis. Higher resolves thin walls and sharp nodes;
    /// cost and triangle count both go as the cube.
    pub fn resolution(mut self, samples_per_cell: usize) -> Self {
        self.resolution = samples_per_cell.max(2);
        self
    }

    /// Ceiling on the sample grid. If the requested resolution would exceed it,
    /// the resolution is lowered until the grid fits rather than the allocation
    /// being attempted.
    ///
    /// A hard cap, not a target. Two samples per cell is the *preferred* floor,
    /// because below it the cell starts aliasing away — but a part a thousand
    /// cells across cannot be sampled twice per cell inside any budget worth
    /// the name, and something has to give. It is the sampling rate:
    /// [`sample_grid`](Self::sample_grid) says what the grid came out as, and
    /// [`wall_samples`](Self::wall_samples) says what that cost.
    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples.max(64);
        self
    }

    /// The grid [`build`](Self::build) will sample, after the resolution has
    /// been reconciled with [`max_samples`](Self::max_samples). Four bytes a
    /// sample, and one field evaluation each.
    pub fn sample_grid(&self) -> [usize; 3] {
        self.grid().0 .0
    }

    /// How many samples fall across the wall, at the resolution that will
    /// actually be used.
    ///
    /// This is the number that decides whether the mesh is a lattice or
    /// gravel. Marching cubes only sees a wall where two neighbouring samples
    /// straddle it, so below about 2 the wall starts being missed between
    /// samples and comes out as disconnected specks — and it does so silently,
    /// because a field that is never sampled inside a wall is
    /// indistinguishable from one with no wall there. 3 is a floor for
    /// something to look at, 5 or more for something to print.
    ///
    /// The trap is that it is easy to reach by accident: the high-area
    /// surfaces, Fischer–Koch S and Lidinoid and split P, hit 25 % density on
    /// walls a third the thickness a gyroid needs, so the same cell size and
    /// the same resolution that render a gyroid cleanly will shatter those.
    /// [`resolve_walls`](Self::resolve_walls) raises the resolution to fix it.
    ///
    /// Measured against the thinnest wall the part actually has: the nominal
    /// [`thickness`](Self::thickness) times the smallest multiplier
    /// [`grade`](Self::grade) applies anywhere in the bounds, swept on a coarse
    /// grid. Grading a part down to two fifths and sizing the grid for the
    /// nominal wall is how the thin end comes out as gravel.
    pub fn wall_samples(&self) -> f32 {
        match self.wall_width() {
            Some(width) => {
                let step = self.grid().0 .2;
                width / step.x.max(step.y).max(step.z)
            }
            None => f32::INFINITY,
        }
    }

    /// The thin dimension the sampling has to resolve, if there is one.
    ///
    /// For everything but a solid TPMS that is the thickness: a wall, a strut
    /// diameter, an infill line. A solid TPMS has no such dimension — its
    /// `thickness` is an *offset* from the minimal surface, and the solid it
    /// bounds is a bulk labyrinth whose features scale with the cell. Reading
    /// the offset as a wall gets it wrong in both directions: zero offset, the
    /// commonest setting there is, reports zero samples across a lattice that
    /// samples perfectly well, and a small offset demands a resolution fine
    /// enough for a feature that does not exist.
    fn wall_width(&self) -> Option<f32> {
        if self.is_level_set() {
            None
        } else {
            Some(self.thickness.abs() * self.grade_floor())
        }
    }

    /// The smallest multiplier [`grade`](Self::grade) applies anywhere in the
    /// bounds, and so the fraction of the nominal thickness that has to be
    /// resolved.
    ///
    /// The whole point of [`wall_samples`](Self::wall_samples) is to catch a
    /// wall thinner than the sampling, and a grade is the commonest way to get
    /// one: a part graded from 1.8 down to 0.4 has its thinnest strut at two
    /// fifths of the thickness the grid was sized for, and the thin end comes
    /// out as gravel while the thick end looks perfect.
    ///
    /// Swept rather than solved, because the grade is an arbitrary closure.
    /// Eight samples an axis is enough for the smooth fields grades actually
    /// are, and cheap enough to pay for on every call.
    ///
    /// Floored at a twentieth: a grade that reaches zero has no wall left to
    /// resolve there, and no resolution would help.
    fn grade_floor(&self) -> f32 {
        let Some(grade) = self.grade.as_deref() else {
            return 1.0;
        };
        const SWEEP: usize = 8;
        let size = self.bounds.size();
        let mut floor = f32::INFINITY;
        for k in 0..SWEEP {
            for j in 0..SWEEP {
                for i in 0..SWEEP {
                    let at = |index: usize, lo: f32, extent: f32| {
                        lo + extent * (index as f32 + 0.5) / SWEEP as f32
                    };
                    let p = Vector3::new(
                        at(i, self.bounds.min.x, size.x),
                        at(j, self.bounds.min.y, size.y),
                        at(k, self.bounds.min.z, size.z),
                    );
                    floor = floor.min(grade(p).max(0.0));
                }
            }
        }
        if floor.is_finite() {
            floor.clamp(0.05, 1.0)
        } else {
            1.0
        }
    }

    /// Whether [`thickness`](Self::thickness) is an offset from a level set
    /// rather than the width of a wall.
    ///
    /// True for a solid TPMS and a solid spinodal — the two generators whose
    /// zero is a surface with half the volume on each side of it, which is what
    /// makes their thickness signed and their wall width meaningless.
    fn is_level_set(&self) -> bool {
        self.style == LatticeStyle::Solid
            && match self.kind {
                LatticeKind::Tpms(_) => true,
                LatticeKind::Stochastic(s) => s.is_field(),
                _ => false,
            }
    }

    /// Raise [`resolution`](Self::resolution) until the wall is at least
    /// `samples` across — see [`wall_samples`](Self::wall_samples).
    ///
    /// Never lowers it, and never overrides
    /// [`max_samples`](Self::max_samples): ask for a wall the sample budget
    /// cannot afford and the budget wins, which `wall_samples` will then say.
    /// Expect to raise the budget with it — four samples across a 0.25 mm wall
    /// in a 20 mm part is a 326³ grid, and the default budget stops at 8 M
    /// samples.
    ///
    /// Call it last: it reads the thickness, so
    /// [`fit_relative_density`](Self::fit_relative_density) has to have run
    /// first.
    pub fn resolve_walls(mut self, samples: f32) -> Self {
        // No wall, nothing to resolve — see `wall_width`.
        let Some(thickness) = self.wall_width() else {
            return self;
        };
        if thickness <= 0.0 || samples <= 0.0 {
            return self;
        }
        let cell = self.resolved_cell();
        let widest = cell.x.max(cell.y).max(cell.z);
        let needed = (samples * widest / thickness).ceil();
        if needed.is_finite() && needed > 0.0 {
            self.resolution = self.resolution.max(needed as usize);
        }
        self
    }

    /// Scale thickness per point — `1.0` leaves it alone, `2.0` doubles it.
    ///
    /// Negative returns are clamped to zero. The surface is the set of points
    /// whose distance to the cell equals the *local* half-thickness, so a
    /// smoothly varying grade gives a smoothly varying wall.
    ///
    /// # It scales, so it needs something to scale
    ///
    /// On a [`LatticeStyle::Solid`] the thickness is an offset from the minimal
    /// surface, and that surface — offset zero — is where half the volume is
    /// already solid. Scaling zero leaves zero, so grading a solid lattice at
    /// its default offset does **nothing at all**, silently: the field is the
    /// same everywhere however the grade varies.
    ///
    /// The parameterisation is what is wrong, not the grade. Every other kind
    /// has thickness zero meaning *no material*, which is what makes a
    /// multiplier the natural knob; a solid lattice's zero means *half*.
    /// Set a non-zero offset first — [`fit_relative_density`](Self::fit_relative_density)
    /// to anything but about half will do it — and the grade scales that.
    /// Where a lattice really has to run from nothing to solid across a part,
    /// [`LatticeStyle::Sheet`] grades over the whole range.
    pub fn grade(mut self, grade: impl Fn(Vector3) -> f32 + Sync + Send + 'a) -> Self {
        self.grade = Some(std::sync::Arc::new(grade));
        self
    }

    /// Intersect the lattice with another field: positive inside, negative out.
    ///
    /// Any signed distance function works — the value only has to have the
    /// right sign near its own zero crossing, and be smooth enough there for
    /// the surface to interpolate. [`fill`](Self::fill) is the same thing given
    /// a [`Region`], and sets the bounds for you.
    pub fn trim(mut self, keep: impl Fn(Vector3) -> f32 + Sync + Send + 'a) -> Self {
        self.trim = Some(std::sync::Arc::new(keep));
        self
    }

    /// Cut the finished part — [`skin`](Self::skin) included — with a region.
    ///
    /// The difference from [`fill`](Self::fill) is *when*: a fill is the shape
    /// of the part, so the skin wraps it and closes over it; a clip is a cut
    /// through a part that has already been made, so the skin stops dead at the
    /// cut face and you can see in. Sectioning a part to inspect its infill is
    /// the reason it exists.
    ///
    /// It does not move the bounds, and it is not counted by
    /// [`relative_density`](Self::relative_density) — a part shown in half is
    /// still the density it was made at.
    pub fn clip(mut self, region: Region<'a>) -> Self {
        self.clip = Some(region.into_field());
        self
    }

    /// Fraction of the part that ends up as material, `0.0` to `1.0`.
    ///
    /// Of the *part*, not of its bounding box: with a [`fill`](Self::fill)
    /// region set, both the material and the space it is measured against are
    /// counted inside the region. That is what a slicer's infill percentage
    /// means, and the difference is not small — a tube swept along a curve
    /// might occupy a fifth of its own bounding box, so 25 % of the box would
    /// be unreachable however thick the walls were made. Without a region the
    /// part is the box, and the two readings agree.
    ///
    /// Measured by sampling rather than derived from a formula, so
    /// [`grade`](Self::grade) and [`skin`](Self::skin) are both accounted for.
    pub fn relative_density(&self) -> f32 {
        self.occupancy(self.density_resolution())
    }

    /// Sampling rate for density work. Deliberately coarser than the mesh, and
    /// deliberately the same for measuring and for fitting, so that a fitted
    /// lattice reports back the density it was asked for.
    ///
    /// Eight a cell, because this is an estimate of a fraction and the error on
    /// one is `√(p(1−p)/N)`: at eight a cell a four-cell part is 32 000
    /// samples, which is a quarter of a percent, and going finer buys a
    /// third of that for eight times the work. The jitter is what makes that
    /// arithmetic apply — see [`occupancy`](Self::occupancy).
    fn density_resolution(&self) -> usize {
        self.resolution.min(8)
    }

    /// A ceiling on the density estimate's sample count, over and above
    /// [`max_samples`](Self::max_samples).
    ///
    /// Samples per cell is the wrong knob for a part that is a hundred cells
    /// across: it would ask for a hundred million points to estimate a number
    /// that a quarter of a million pins down to a tenth of a percent. The
    /// estimate does not get better with the size of the part, so it does not
    /// get more expensive with it either.
    const DENSITY_SAMPLES: usize = 250_000;

    /// And a floor under it, for the same reason from the other side.
    ///
    /// Samples per cell is just as wrong for a lattice that is *one* cell
    /// across — which is what a homogenisation asks for — where eight an axis
    /// is 512 points and a two per cent error on a fraction. Density is
    /// estimated from the total, so the total is what is held up.
    const DENSITY_MIN: usize = 32_768;

    /// Solve [`thickness`](Self::thickness) for a target
    /// [`relative_density`](Self::relative_density).
    ///
    /// Density rises monotonically with thickness, so this bisects. A target
    /// the lattice cannot reach — denser than the cell can pack, or thinner
    /// than a single wall — lands on the nearest thickness that can.
    ///
    /// Costs about fifteen density measurements, each a coarse pass over the
    /// bounds,
    /// so it is worth doing once and reusing the
    /// [`thickness`](Self::current_thickness) rather than calling it per frame.
    /// The region being filled is evaluated once for all twenty, not once each:
    /// it does not move as the thickness does, and a [`Region::mesh`] costs two
    /// BVH queries a sample where the lattice itself costs a few multiplies.
    pub fn fit_relative_density(mut self, target: f32) -> Self {
        let target = target.clamp(0.0, 1.0);
        let cell = self.resolved_cell();
        let span = cell.x.min(cell.y).min(cell.z).max(1e-6);
        // A solid level set is the only kind whose thickness goes negative: it
        // is an offset from the surface, and below zero it thins the labyrinth
        // rather than emptying the cell.
        let signed = self.is_level_set();
        let grid = self.density_grid(self.density_resolution());

        // Widen the bracket until it really does bracket the target. One cell
        // is the right *scale* for a starting guess and not a bound: how thick
        // a wall a given generator needs for a given density varies several
        // fold between them, and bisecting inside a bracket that does not
        // contain the answer converges neatly onto the bracket's own end and
        // returns it as though it were the answer. That is how a sheet asked
        // for 97 % came back with 94 %.
        let mut hi = span;
        for _ in 0..16 {
            self.thickness = hi;
            if self.occupancy_on(&grid) >= target {
                break;
            }
            hi *= 2.0;
        }
        let mut lo = if signed { -span } else { 0.0 };
        if signed {
            for _ in 0..16 {
                self.thickness = lo;
                if self.occupancy_on(&grid) <= target {
                    break;
                }
                lo *= 2.0;
            }
        }

        // Fourteen halvings of a bracket one cell wide: a micron on a
        // millimetre cell, which is finer than the thickness means anything to
        // and six passes cheaper than twenty.
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            self.thickness = mid;
            if self.occupancy_on(&grid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.thickness = 0.5 * (lo + hi);
        self
    }

    /// The thickness currently set, including one solved by
    /// [`fit_relative_density`](Self::fit_relative_density).
    pub fn current_thickness(&self) -> f32 {
        self.thickness
    }

    /// Sample the field and contour it.
    ///
    /// Empty geometry — no attributes, no index — when nothing crosses the
    /// surface, which is what a thickness of zero or a trim that excludes the
    /// whole box gives.
    pub fn build(&self) -> BufferGeometry {
        let (grid, sampler) = self.grid();
        let ([nx, ny, nz], origin, step) = grid;
        // `grid` bounds this, but sizing an allocation is not somewhere to
        // trust a caller's arithmetic: an overflow here is a panic, and an
        // empty mesh is the honest answer for a grid that cannot be made.
        let Some(total) = nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)) else {
            return BufferGeometry::new();
        };
        // A slab per z, written straight into the grid it will be contoured
        // from. Every sample has a fixed address, so the result cannot depend
        // on how the threads interleave — a build flag must not change the
        // geometry.
        let mut values = vec![0.0f32; total];
        crate::utils::parallel::par_fill_chunks(&mut values, nx * ny, |k, slab| {
            let z = origin.z + k as f32 * step.z;
            for j in 0..ny {
                let y = origin.y + j as f32 * step.y;
                for i in 0..nx {
                    let x = origin.x + i as f32 * step.x;
                    slab[j * nx + i] = sampler.value(Vector3::new(x, y, z));
                }
            }
        });
        IsoGrid {
            dims: [nx, ny, nz],
            origin,
            step,
            values,
        }
        .triangulate()
    }

    /// Cell count along x, which [`ChiralRule`] treats as the column width.
    fn chiral_n(&self) -> i32 {
        match self.spacing {
            Spacing::Count([nx, ..]) => nx as i32,
            Spacing::Size(_) => {
                let size = self.bounds.size();
                let cell = self.resolved_cell();
                (size.x / cell.x).round().max(0.0) as i32
            }
        }
    }

    /// Cell period in world units, however it was specified.
    fn resolved_cell(&self) -> Vector3 {
        let size = self.bounds.size();
        match self.spacing {
            Spacing::Size(c) => Vector3::new(
                c.x.abs().max(1e-6),
                c.y.abs().max(1e-6),
                c.z.abs().max(1e-6),
            ),
            Spacing::Count([nx, ny, nz]) => Vector3::new(
                size.x / nx.max(1) as f32,
                size.y / ny.max(1) as f32,
                size.z / nz.max(1) as f32,
            ),
        }
    }

    /// The sample grid, and the field to fill it with.
    ///
    /// The grid overhangs the bounds by two and a half samples on every side:
    /// two so the box's own faces are contoured with a live gradient either
    /// side, and the half so no sample plane lands exactly on a face, where the
    /// field would be identically zero and the surface ambiguous.
    fn grid(&self) -> (([usize; 3], Vector3, Vector3), Sampler<'_>) {
        let cell = self.resolved_cell();
        let size = self.bounds.size();
        let resolution = self.resolution.max(2);

        // Coarsen by growing the *step*, not by lowering samples-per-cell.
        //
        // The two are the same until the cells themselves are the problem: a
        // part a thousand cells across cannot be sampled twice per cell inside
        // any budget, and a loop that only ever reduced samples-per-cell would
        // bottom out at two, hand back a grid of billions, and leave `build` to
        // overflow the multiply that sizes its allocation. The budget has to be
        // able to win outright, so the step is what gives.
        let mut step = Vector3::new(
            cell.x / resolution as f32,
            cell.y / resolution as f32,
            cell.z / resolution as f32,
        );
        let dims_for = |step: Vector3| -> [usize; 3] {
            // Saturating on the way in: `extent / step` is `inf` for a zero
            // step and enormous for a tiny one, and both cast to a count this
            // loop can then shrink.
            let count = |extent: f32, step: f32| -> usize {
                // Negated deliberately: this has to catch NaN as well as zero
                // and negatives, and `step <= 0.0` is false for NaN.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(step > 0.0) {
                    return MIN_DIM;
                }
                ((extent / step).ceil() as usize)
                    .saturating_add(PADDING)
                    .max(MIN_DIM)
            };
            [
                count(size.x, step.x),
                count(size.y, step.y),
                count(size.z, step.z),
            ]
        };
        let mut dims = dims_for(step);
        for _ in 0..32 {
            let total = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
            // Six of the samples on each axis are the padding either side, so
            // the grid bottoms out at 6³ however hard the budget pushes.
            if total <= self.max_samples || dims == [MIN_DIM; 3] || !step.x.is_finite() {
                break;
            }
            let shrink = (self.max_samples as f64 / total as f64).cbrt();
            step = step * (1.0 / shrink.clamp(1e-9, 0.95) as f32);
            dims = dims_for(step);
        }

        let origin = Vector3::new(
            self.bounds.min.x - 2.5 * step.x,
            self.bounds.min.y - 2.5 * step.y,
            self.bounds.min.z - 2.5 * step.z,
        );
        ((dims, origin, step), self.sampler(cell, step, true))
    }

    fn sampler(&self, cell: Vector3, step: Vector3, clipped: bool) -> Sampler<'_> {
        let cuboct_cached = match (self.kind, self.program.is_some(), self.chiral_rule) {
            (LatticeKind::Cuboct(c), false, None) => Some(c.segments(self.shape)),
            _ => None,
        };
        // Built once here rather than per sample: two dozen directions and
        // phases, against `WAVES` trigonometric calls for every point of a
        // grid that is millions of points long.
        let waves = match self.kind {
            LatticeKind::Stochastic(s) if s.is_field() => {
                Some(stochastic::waves(s, cell, self.seed))
            }
            _ => None,
        };
        Sampler {
            kind: self.kind,
            style: self.style,
            // A conformal map hands back coordinates in its own frame, already
            // anchored at zero; the box's corner is only the phase when the
            // cells are laid out in world space.
            phase: if self.conform.is_some() {
                Vector3::ZERO
            } else {
                self.bounds.min
            },
            cell,
            omega: Vector3::new(TAU / cell.x, TAU / cell.y, TAU / cell.z),
            half: self.thickness * 0.5,
            margin: 3.0 * step.x.max(step.y).max(step.z),
            skin: self.skin,
            bounds: self.bounds,
            grade: self.grade.as_deref(),
            trim: self.trim.as_deref(),
            clip: if clipped { self.clip.as_deref() } else { None },
            cuboct_shape: self.shape,
            cuboct_rule: self.chiral_rule,
            cuboct_n: self.chiral_n(),
            cuboct_program: self.program.as_deref(),
            cuboct_cached,
            conform: self.conform.as_ref(),
            jitter: self.jitter,
            seed: self.seed,
            waves,
        }
    }

    /// The effective stiffness of the cell this lattice is built from, as a
    /// material — see [`Stiffness`], and [`homogenize`] for what it means.
    ///
    /// `resolution` is the voxel grid one cell is solved on, 2 to 48. Twelve
    /// ranks two cells against each other; twenty is a number to quote.
    ///
    /// The *cell* is what is measured, not the part: the trim, the skin, the
    /// clip and the grade are all left out, because a lattice that has been cut
    /// or graded is not periodic and a stiffness is only a material property of
    /// something that is.
    ///
    /// # Patterns that are not periodic
    ///
    /// Two of the [`Infill`] patterns do not repeat every cell, and what comes
    /// back describes a window of them rather than the whole:
    ///
    /// - [`Infill::Concentric`] has no unit cell at all — its rings are struck
    ///   from the part's outline. It is measured as what it is locally, which
    ///   is parallel walls at the line spacing.
    /// - The [`is_layered`](Infill::is_layered) patterns alternate direction
    ///   from layer to layer, so one cell holds one layer and misses the
    ///   cross-ply entirely. Give those a window of two or more.
    ///
    /// # A foam has no cell
    ///
    /// [`LatticeKind::Stochastic`] does not repeat, so there is nothing for a
    /// periodic boundary to be periodic with: wrapping one cell of a foam cuts
    /// whatever crossed the face, and one cell of a random medium is one
    /// bubble. Use [`homogenize_window`](Self::homogenize_window) for those —
    /// several cells across, and averaged over a few [`seed`](Self::seed)s if
    /// the number is going anywhere.
    pub fn homogenize(&self, resolution: usize) -> Stiffness {
        self.homogenize_with(resolution, SolidMaterial::default())
    }

    /// The same, in the units of a real material rather than as a fraction.
    pub fn homogenize_with(&self, resolution: usize, material: SolidMaterial) -> Stiffness {
        self.homogenize_window(1, resolution, material)
    }

    /// Homogenise a window of `cells` cells per axis rather than one, at
    /// `resolution` voxels per cell.
    ///
    /// For a periodic family this buys nothing but cost — a bigger window of a
    /// repeating cell is the same material, and the answer should come back
    /// the same. For a foam it is the whole difference between a measurement
    /// and an anecdote.
    ///
    /// The voxel grid is `cells · resolution` per axis and caps at 48, so three
    /// cells at 16 is the most that will be honoured.
    pub fn homogenize_window(
        &self,
        cells: usize,
        resolution: usize,
        material: SolidMaterial,
    ) -> Stiffness {
        let (occupancy, origin, window, voxels) = self.periodic_window(cells, resolution);
        homogenize(occupancy, origin, window, voxels, material, self.solver)
    }

    /// When the cell this lattice is built from gives way — see [`Strength`].
    ///
    /// Solves the stiffness on the way and hands it back inside the result, so
    /// this is not [`homogenize`](Self::homogenize) plus something: it is
    /// [`homogenize`](Self::homogenize) plus one pass over the elements.
    ///
    /// Yielding only. Elastic buckling is not modelled, and
    /// [`collapse_strain`](Strength::collapse_strain) is how to tell whether
    /// that matters for the lattice in hand.
    pub fn strength(&self, resolution: usize) -> Strength {
        self.strength_with(resolution, SolidMaterial::default())
    }

    /// The same, for a named material.
    pub fn strength_with(&self, resolution: usize, material: SolidMaterial) -> Strength {
        self.strength_window(1, resolution, material)
    }

    /// Over a window of `cells` cells per axis — see
    /// [`homogenize_window`](Self::homogenize_window), which this mirrors.
    pub fn strength_window(
        &self,
        cells: usize,
        resolution: usize,
        material: SolidMaterial,
    ) -> Strength {
        let (occupancy, origin, window, voxels) = self.periodic_window(cells, resolution);
        homogenize_strength(occupancy, origin, window, voxels, material, self.solver)
    }

    /// The effective conductivity of the cell this lattice is built from, as a
    /// fraction of the solid's — see [`Conductivity`].
    ///
    /// Heat, electricity, diffusion and permittivity are the same equation, so
    /// this is the same number for all of them; only the units of the solid's
    /// own conductivity differ. The caveats are
    /// [`homogenize`](Self::homogenize)'s, down to the last one: a foam has no
    /// cell, and wants [`conductivity_window`](Self::conductivity_window).
    ///
    /// ```
    /// use threers::{Lattice, LatticeKind, Strut, Vector3};
    ///
    /// let k = Lattice::new(LatticeKind::Strut(Strut::Cubic))
    ///     .size(Vector3::new(10.0, 10.0, 10.0))
    ///     .cells([1, 1, 1])
    ///     .fit_relative_density(0.3)
    ///     .conductivity(10);
    ///
    /// // Simple cubic is three sets of straight bars that share their nodes,
    /// // so the columns running along the gradient hold a little over half of
    /// // the material — not the third the three families would suggest.
    /// assert!((k.tortuosity_factor(1.0) - 0.53).abs() < 0.08);
    /// ```
    pub fn conductivity(&self, resolution: usize) -> Conductivity {
        self.conductivity_with(resolution, 1.0)
    }

    /// The same, in the units of a real material rather than as a fraction —
    /// `solid` being what the material it is printed in conducts at.
    pub fn conductivity_with(&self, resolution: usize, solid: f64) -> Conductivity {
        self.conductivity_window(1, resolution, solid)
    }

    /// Over a window of `cells` cells per axis rather than one — see
    /// [`homogenize_window`](Self::homogenize_window), which this mirrors.
    pub fn conductivity_window(
        &self,
        cells: usize,
        resolution: usize,
        solid: f64,
    ) -> Conductivity {
        let (occupancy, origin, window, voxels) = self.periodic_window(cells, resolution);
        homogenize_conduction(occupancy, origin, window, voxels, solid, self.solver)
    }

    /// The pieces both homogenisations need: a test for what is solid, and the
    /// periodic window to run it over.
    fn periodic_window(
        &self,
        cells: usize,
        resolution: usize,
    ) -> (
        impl Fn(Vector3) -> bool + Sync + Send + '_,
        Vector3,
        Vector3,
        usize,
    ) {
        let cells = cells.max(1);
        let cell = self.resolved_cell();
        // A zero step, because only the *sign* of the field is read here. The
        // step sets how far past the surface a beam lattice keeps searching
        // for struts so that the gradient stays exact for the contouring, and
        // a cell's worth of that is most of the cost of voxelising it — twenty
        // seven samples an element with nothing culled.
        let sampler = self.sampler(cell, Vector3::ZERO, false);
        let half = sampler.half;
        let origin = if self.conform.is_some() {
            Vector3::ZERO
        } else {
            self.bounds.min
        };
        (
            // Concentric infill is the one pattern that reads how deep into the
            // part it is, and it has no unit cell at all: its rings are drawn
            // from the part's outline, which does not repeat. What *does*
            // repeat is the spacing between them, so it is handed a depth that
            // ramps along x — the local structure of concentric infill, which
            // is parallel walls at the line spacing. Every other pattern
            // ignores the argument.
            //
            // A depth of infinity, which reads as "deep inside", is what this
            // used to pass, and it put the first ring infinitely far away: the
            // cell came back empty and the stiffness came back zero.
            move |p| sampler.solid_at(p, half, p.x - origin.x) > 0.0,
            origin,
            cell * cells as f32,
            resolution.max(2).saturating_mul(cells),
        )
    }

    /// Fraction of the bounds that is solid, by counting samples.
    ///
    /// The samples are stratified and jittered rather than laid on a regular
    /// grid. A regular grid is *commensurate* with the lattice — every cell
    /// gets sampled at the same relative points — so thousands of samples sit
    /// at exactly the same distance from a strut and all cross at exactly the
    /// same thickness. Density then moves in visible steps, and anything
    /// solving against it (see
    /// [`fit_relative_density`](Self::fit_relative_density)) can only land on a
    /// step edge. Jittering breaks the symmetry and leaves density effectively
    /// continuous, while the offsets stay a pure function of the sample index,
    /// so repeated calls agree exactly.
    fn occupancy(&self, samples_per_cell: usize) -> f32 {
        self.occupancy_on(&self.density_grid(samples_per_cell))
    }

    /// The points density is measured at, and the trim's value at each.
    ///
    /// Split out from the counting so a bisection can build it once. The trim
    /// values are the expensive half and the only half that does not change
    /// with the thickness.
    fn density_grid(&self, samples_per_cell: usize) -> DensityGrid {
        let cell = self.resolved_cell();
        let size = self.bounds.size();
        let mut res = samples_per_cell.max(2) as f32;
        let count =
            |extent: f32, cell: f32, res: f32| (((extent / cell) * res).ceil() as usize).max(1);
        // Bounded for the same reason the build grid is: a part that is
        // thousands of cells across would otherwise ask for a points vector
        // whose length overflows before it is ever allocated.
        let mut dims = [
            count(size.x, cell.x, res),
            count(size.y, cell.y, res),
            count(size.z, cell.z, res),
        ];
        let budget = self.max_samples.min(Self::DENSITY_SAMPLES);
        let floor = Self::DENSITY_MIN.min(budget);
        let resize = |res: f32| {
            [
                count(size.x, cell.x, res),
                count(size.y, cell.y, res),
                count(size.z, cell.z, res),
            ]
        };
        for _ in 0..32 {
            let total = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
            let scale = if total > budget {
                (budget as f64 / total as f64).cbrt().clamp(1e-9, 0.95)
            } else if total < floor {
                (floor as f64 / total as f64).cbrt().max(1.05)
            } else {
                break;
            };
            res *= scale as f32;
            let next = resize(res);
            if next == dims {
                break;
            }
            dims = next;
        }
        // Strata that exactly tile the bounds, so no sample lands outside it.
        let stride = Vector3::new(
            size.x / dims[0] as f32,
            size.y / dims[1] as f32,
            size.z / dims[2] as f32,
        );
        let mut points = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let at = jitter(i, j, k);
                    points.push(Vector3::new(
                        self.bounds.min.x + (i as f32 + at[0]) * stride.x,
                        self.bounds.min.y + (j as f32 + at[1]) * stride.y,
                        self.bounds.min.z + (k as f32 + at[2]) * stride.z,
                    ));
                }
            }
        }
        let cuts = match self.trim.as_deref() {
            Some(trim) => crate::utils::parallel::par_map(&points, |p| trim(*p)),
            None => Vec::new(),
        };
        DensityGrid {
            points,
            cuts,
            cell,
        }
    }

    /// Fraction of the part covered by `grid` that lands in solid.
    ///
    /// Both counts are taken inside the fill region: the numerator is samples
    /// in material, the denominator samples in the part at all. Measuring
    /// against the whole box instead would make the reading depend on how
    /// loosely the box happens to fit.
    fn occupancy_on(&self, grid: &DensityGrid) -> f32 {
        if grid.points.is_empty() {
            return 0.0;
        }
        // Without the clip: sectioning a part for inspection does not change
        // what it was built at. And with a zero step, because only the *sign*
        // of the field is read here — the step widens a beam lattice's search
        // radius so that the gradient stays exact a few samples past the
        // surface, which is worth paying for when contouring and is most of
        // the cost of counting. A wider radius stops the neighbouring cells
        // being culled, and a bisection pays for it twenty times over.
        let sampler = self.sampler(grid.cell, Vector3::ZERO, false);
        // Counted per chunk and summed in order, so the total does not depend
        // on how the threads interleave. `fit_relative_density` bisects on
        // this; a density that wobbled between calls would make it wander.
        let chunk = grid.points.len().div_ceil(64).max(1);
        let counts =
            crate::utils::parallel::par_map_range(grid.points.len().div_ceil(chunk), |c| {
                let lo = c * chunk;
                let hi = (lo + chunk).min(grid.points.len());
                let (mut solid, mut part) = (0usize, 0usize);
                for n in lo..hi {
                    let cut = grid.cuts.get(n).copied();
                    if cut.is_none_or(|c| c > 0.0) {
                        part += 1;
                        if sampler.value_at(grid.points[n], cut) > 0.0 {
                            solid += 1;
                        }
                    }
                }
                (solid, part)
            });
        let (solid, part) = counts
            .iter()
            .fold((0usize, 0usize), |acc, c| (acc.0 + c.0, acc.1 + c.1));
        if part == 0 {
            return 0.0;
        }
        solid as f32 / part as f32
    }
}

/// Where density is measured, and what the trim says there.
struct DensityGrid {
    points: Vec<Vector3>,
    /// The trim's value per point, or empty when there is no trim.
    cuts: Vec<f32>,
    cell: Vector3,
}

/// Three offsets in `[0, 1)` from a sample's index — a hash, not a sequence, so
/// the same sample always lands in the same place.
fn jitter(i: usize, j: usize, k: usize) -> [f32; 3] {
    let mut h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (k as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    let mut out = [0.0f32; 3];
    for axis in &mut out {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        *axis = (h >> 40) as f32 / (1u32 << 24) as f32;
    }
    out
}

/// The resolved field: everything `build` and `occupancy` need per sample.
struct Sampler<'a> {
    kind: LatticeKind,
    style: LatticeStyle,
    /// World point the cell grid starts from.
    phase: Vector3,
    cell: Vector3,
    /// Radians per world unit, per axis.
    omega: Vector3,
    half: f32,
    margin: f32,
    skin: f32,
    bounds: Box3,
    grade: Option<&'a (dyn Fn(Vector3) -> f32 + Sync + Send)>,
    trim: Option<&'a (dyn Fn(Vector3) -> f32 + Sync + Send)>,
    clip: Option<&'a (dyn Fn(Vector3) -> f32 + Sync + Send)>,
    cuboct_shape: f32,
    cuboct_rule: Option<ChiralRule>,
    cuboct_n: i32,
    cuboct_program: Option<&'a (dyn Fn(i32, i32, i32) -> Cuboct + Sync + Send)>,
    cuboct_cached: Option<Vec<Segment>>,
    /// The map into cell space, when the cells are not laid out in world
    /// space.
    conform: Option<&'a Conform<'a>>,
    jitter: f32,
    seed: u32,
    /// The plane waves a spinodal field is summed from. Ignored by everything
    /// else.
    waves: Option<Vec<stochastic::Wave>>,
}

impl Sampler<'_> {
    /// Positive inside the solid, negative outside, zero on the surface.
    fn value(&self, p: Vector3) -> f32 {
        self.value_at(p, self.trim.map(|t| t(p)))
    }

    /// The same, with the trim's value supplied rather than evaluated — for
    /// callers that already know it and would rather not pay for it twenty
    /// times over.
    fn value_at(&self, p: Vector3, cut: Option<f32>) -> f32 {
        let half = match self.grade {
            Some(g) => self.half * g(p).max(0.0),
            None => self.half,
        };
        // How deep into the part this point is. The concentric pattern lays its
        // rings against it, so it has to be known before the pattern is — and
        // it wants the depth measured in the layer plane, the way a slicer
        // offsets an outline, not through the top and bottom faces as well.
        let mut boundary = self.box_distance(p);
        let mut depth = self.plan_distance(p);
        if let Some(cut) = cut {
            boundary = boundary.min(cut);
            depth = depth.min(cut);
        }
        let filled = self.solid_at(p, half, depth).min(boundary);
        let clip = |v: f32| match self.clip {
            Some(c) => v.min(c(p)),
            None => v,
        };
        if self.skin <= 0.0 {
            return clip(filled);
        }
        // The skin is the band from the surface inwards, and it is *unioned*
        // with the lattice rather than intersected — so the two bond wherever
        // they touch, which is what a printed part does and what keeps this one
        // shell rather than two.
        clip(filled.max(boundary.min(self.skin - boundary)))
    }

    /// The lattice's own field at a point: positive inside a wall or a strut.
    ///
    /// No boundary, no trim, no skin and no clip — those are properties of the
    /// part, and this is the periodic thing filling it. `half` is the local
    /// half-thickness, already graded, and `depth` how far into the part the
    /// point lies, which only the concentric infill reads.
    fn solid_at(&self, p: Vector3, half: f32, depth: f32) -> f32 {
        // A conformal map takes the point into the space the cells are tiled
        // in, and says how much it stretched to get there. `half` is a length
        // in that space, so it is stretched with it and the answer divided back
        // out — see [`Conform`] for why that cannot be exact.
        let (p, stretch) = match self.conform {
            Some(c) => (c.point(p), c.stretch_at(p)),
            None => (p, 1.0),
        };
        let half = half * stretch;
        let margin = self.margin * stretch;
        let solid = match self.kind {
            LatticeKind::Strut(s) => {
                // Only struts nearer than the local radius can matter; the
                // margin keeps the gradient exact for a few samples beyond.
                let cull = half.max(0.0) + margin;
                let d = strut::distance(p, self.phase, self.cell, s.segments(), cull);
                half - d
            }
            LatticeKind::Cuboct(c) => {
                let cull = half.max(0.0) + margin;
                let d = if let Some(ref segs) = self.cuboct_cached {
                    strut::distance(p, self.phase, self.cell, segs, cull)
                } else {
                    cuboct::distance(
                        p,
                        self.phase,
                        self.cell,
                        cull,
                        |i, j, k| match self.cuboct_program {
                            Some(f) => f(i, j, k),
                            None => c,
                        },
                        self.cuboct_shape,
                        self.cuboct_rule,
                        self.cuboct_n,
                    )
                };
                half - d
            }
            LatticeKind::Infill(i) => half - infill::distance(i, p, self.phase, self.cell, depth),
            LatticeKind::Tpms(t) => {
                let sd = self.tpms_distance(t, p);
                match self.style {
                    LatticeStyle::Sheet => half - sd.abs(),
                    LatticeStyle::Solid => sd + half,
                }
            }
            // A spinodal field is a level set like a TPMS, and reads its
            // thickness the same way.
            LatticeKind::Stochastic(s) if s.is_field() => {
                let sd = self.spinodal_distance(p);
                match self.style {
                    LatticeStyle::Sheet => half - sd.abs(),
                    LatticeStyle::Solid => sd + half,
                }
            }
            LatticeKind::Stochastic(s) => {
                let d = stochastic::voronoi_distance(
                    p,
                    self.phase,
                    self.cell,
                    matches!(s, Stochastic::VoronoiWall),
                    self.jitter,
                    self.seed,
                );
                half - d
            }
        };
        solid / stretch
    }

    /// The spinodal field, divided by its own gradient so that the thickness
    /// asked for is a length.
    ///
    /// The same correction a TPMS needs and for the same reason — a sum of
    /// cosines climbs at different rates in different places, so a wall of
    /// constant field value is not a wall of constant thickness.
    fn spinodal_distance(&self, p: Vector3) -> f32 {
        let waves = self.waves.as_deref().unwrap_or(&[]);
        let (value, gradient) = stochastic::value_and_gradient(waves, p - self.phase);
        value / gradient.length().max(1e-6)
    }

    /// The nodal function rescaled into world units.
    ///
    /// `F` itself is not a distance — it climbs faster near the channels than
    /// across them — so a wall of constant `F` varies in real thickness by a
    /// factor of two or more. Dividing by `|∇F|` is the first-order distance to
    /// the surface, which is what makes `thickness` a length you can print to.
    ///
    /// Both come from one call and one set of sines and cosines. Differencing
    /// for the gradient instead would be seven evaluations a sample, and would
    /// need a step size small enough to track the surface but large enough to
    /// survive `f32` — a trade with no good answer at either end of the
    /// resolution range.
    fn tpms_distance(&self, t: Tpms, p: Vector3) -> f32 {
        let (value, g) = t.value_and_gradient(
            (p.x - self.phase.x) * self.omega.x,
            (p.y - self.phase.y) * self.omega.y,
            (p.z - self.phase.z) * self.omega.z,
        );
        // The gradient comes back per radian; the cell says how many radians a
        // unit of length is worth on each axis.
        let gradient = Vector3::new(
            g[0] * self.omega.x,
            g[1] * self.omega.y,
            g[2] * self.omega.z,
        );
        value / gradient.length().max(1e-6)
    }

    /// Distance into the bounding box measured in the layer plane only — the
    /// depth a slicer's outline offset would see, with the top and bottom
    /// faces ignored.
    fn plan_distance(&self, p: Vector3) -> f32 {
        let lo = self.bounds.min;
        let hi = self.bounds.max;
        (p.x - lo.x).min(hi.x - p.x).min(p.y - lo.y).min(hi.y - p.y)
    }

    /// Distance into the bounding box, negative outside it. Exact inside, which
    /// is where the surface it cuts actually lands.
    fn box_distance(&self, p: Vector3) -> f32 {
        let lo = self.bounds.min;
        let hi = self.bounds.max;
        (p.x - lo.x)
            .min(hi.x - p.x)
            .min(p.y - lo.y)
            .min(hi.y - p.y)
            .min(p.z - lo.z)
            .min(hi.z - p.z)
    }
}

/// Periodic infill as a `BufferGeometry`, in the shape of the other geometry
/// constructors. [`Lattice`] is the same generator with the whole knob set.
pub struct LatticeGeometry;

impl LatticeGeometry {
    /// A box of `size` centred on the origin, filled with `cells` cells of
    /// `kind` at the given wall thickness or strut diameter.
    pub fn new(
        kind: LatticeKind,
        size: Vector3,
        cells: [usize; 3],
        thickness: f32,
    ) -> BufferGeometry {
        Lattice::new(kind)
            .size(size)
            .cells(cells)
            .thickness(thickness)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn geom_of(kind: LatticeKind) -> BufferGeometry {
        Lattice::new(kind)
            .size(Vector3::new(2.0, 2.0, 2.0))
            .cells([2, 2, 2])
            .thickness(0.14)
            .resolution(12)
            .build()
    }

    /// Every directed edge cancelling with its reverse means the surface is
    /// closed *and* consistently wound — the property a slicer depends on.
    fn assert_watertight(geom: &BufferGeometry, what: &str) {
        let index = geom
            .index
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: no index"));
        assert!(!index.is_empty(), "{what}: no triangles");
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for t in index.chunks_exact(3) {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (key, dir) = if e.0 < e.1 {
                    ((e.0, e.1), 1)
                } else {
                    ((e.1, e.0), -1)
                };
                *edges.entry(key).or_insert(0) += dir;
            }
        }
        let open = edges.values().filter(|&&v| v != 0).count();
        assert_eq!(open, 0, "{what}: {open} unpaired edges");
    }

    #[test]
    fn every_tpms_builds_watertight() {
        for t in Tpms::ALL {
            let kind = LatticeKind::Tpms(t);
            assert_watertight(&geom_of(kind), t.name());
        }
    }

    #[test]
    fn every_strut_builds_watertight() {
        for s in Strut::ALL {
            let kind = LatticeKind::Strut(s);
            assert_watertight(&geom_of(kind), s.name());
        }
    }

    #[test]
    fn every_infill_builds_watertight() {
        for i in Infill::ALL {
            let kind = LatticeKind::Infill(i);
            assert_watertight(&geom_of(kind), i.name());
        }
    }

    #[test]
    fn every_cuboct_builds_watertight() {
        for c in Cuboct::ALL {
            let kind = LatticeKind::Cuboct(c);
            assert_watertight(&geom_of(kind), c.name());
        }
    }

    #[test]
    fn a_programmed_chiral_column_builds_watertight() {
        let m = 4i32;
        let geom = Lattice::new(LatticeKind::Cuboct(Cuboct::ChiralCcw))
            .size(Vector3::new(2.0, 2.0, 4.0))
            .cells([2, 2, 4])
            .thickness(0.14)
            .resolution(12)
            .chiral_rule(ChiralRule::R2)
            .program(move |_, _, k| Cuboct::column_half(k, m))
            .build();
        assert_watertight(&geom, "chiral column");
    }

    #[test]
    fn a_ring_of_cells_closes_on_itself() {
        // A cylindrical map is only seamless if a whole number of cells fits
        // the circumference — which is a property of the arithmetic, so it can
        // be checked exactly rather than looked at.
        let radius = 12.0f32;
        let count = 16usize;
        let axis = Vector3::new(0.0, 0.0, 1.0);
        // The largest disagreement anywhere in the wall between a point and
        // the same point turned by one cell.
        let worst_of = |pitch: f32| {
            let lattice = Lattice::new(LatticeKind::Strut(Strut::Cubic))
                .size(Vector3::new(30.0, 30.0, 10.0))
                .conform(Conform::cylindrical(Vector3::ZERO, axis, radius))
                .cell_size(Vector3::new(pitch, 3.0, 2.0))
                .thickness(0.6);
            let cell = lattice.resolved_cell();
            let sampler = lattice.sampler(cell, Vector3::new(0.05, 0.05, 0.05), false);
            let half = sampler.half;
            let angle = TAU / count as f32;
            let (sin, cos) = angle.sin_cos();
            let mut worst = 0.0f32;
            for i in 0..40 {
                let t = i as f32 / 40.0;
                let r = radius - 2.0 + 4.0 * t;
                let a = t * 2.3;
                let p = Vector3::new(r * a.cos(), r * a.sin(), t * 6.0 - 3.0);
                let turned = Vector3::new(p.x * cos - p.y * sin, p.x * sin + p.y * cos, p.z);
                let here = sampler.solid_at(p, half, f32::INFINITY);
                let there = sampler.solid_at(turned, half, f32::INFINITY);
                worst = worst.max((here - there).abs());
            }
            worst
        };

        assert!(
            worst_of(Conform::ring_pitch(radius, count)) < 1e-4,
            "a turn of one cell moved the lattice: {}",
            worst_of(Conform::ring_pitch(radius, count))
        );
        // And a pitch that does not divide the circumference does not close.
        assert!(
            worst_of(Conform::ring_pitch(radius, count) * 1.3) > 1e-3,
            "an ill-fitting pitch closed anyway"
        );
    }

    #[test]
    fn a_depth_map_stacks_whole_layers_through_a_curved_wall() {
        // Two points on one vertical line, one cell deeper below a sphere's
        // surface. Under the map they differ by exactly one cell and the
        // lattice reads the same; in world space they differ by something else
        // and it does not.
        let outer = 10.0f32;
        let cell_z = 2.0f32;
        let build = |conform: bool| {
            let mut lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(20.0, 20.0, 20.0))
                .cell_size(Vector3::new(4.0, 4.0, cell_z))
                .thickness(0.5);
            if conform {
                lattice = lattice.conform(Conform::depth(Region::sphere(Vector3::ZERO, outer)));
            }
            let cell = lattice.resolved_cell();
            let sampler = lattice.sampler(cell, Vector3::new(0.05, 0.05, 0.05), false);
            let half = sampler.half;
            // Radius 8 and radius 6: two millimetres deeper, along one line.
            let (a, b) = (1.0f32, 1.0f32);
            let at = |r: f32| {
                let z = (r * r - a * a - b * b).sqrt();
                sampler.solid_at(Vector3::new(a, b, z), half, f32::INFINITY)
            };
            (at(8.0), at(8.0 - cell_z))
        };

        let (near, deep) = build(true);
        assert!(
            (near - deep).abs() < 1e-3,
            "the layers did not line up: {near} vs {deep}"
        );
        let (near, deep) = build(false);
        assert!(
            (near - deep).abs() > 1e-3,
            "world-space cells lined up by accident: {near} vs {deep}"
        );
    }

    #[test]
    fn every_foam_fits_a_density_target() {
        for cell in Stochastic::ALL {
            let lattice = Lattice::new(LatticeKind::Stochastic(cell))
                .size(Vector3::new(12.0, 12.0, 12.0))
                .cells([4, 4, 4])
                .seed(7)
                .fit_relative_density(0.25);
            assert!(
                (lattice.relative_density() - 0.25).abs() < 0.02,
                "{}: {}",
                cell.name(),
                lattice.relative_density()
            );
            assert_watertight(&lattice.resolution(12).build(), cell.name());
        }
    }

    #[test]
    fn two_seeds_are_two_foams_of_the_same_density() {
        let build = |seed: u32| {
            Lattice::new(LatticeKind::Stochastic(Stochastic::Voronoi))
                .size(Vector3::new(10.0, 10.0, 10.0))
                .cells([3, 3, 3])
                .seed(seed)
                .fit_relative_density(0.2)
        };
        let a = build(1);
        let b = build(2);
        assert!((a.relative_density() - b.relative_density()).abs() < 0.02);
        // Same statistics, different foam: somewhere in the block they differ.
        let cell = a.resolved_cell();
        let step = Vector3::new(0.05, 0.05, 0.05);
        let (sa, sb) = (a.sampler(cell, step, false), b.sampler(cell, step, false));
        let differs = (0..50).any(|i| {
            let t = i as f32 * 0.17;
            let p = Vector3::new(t.sin() * 4.0, t.cos() * 4.0, t * 0.1);
            (sa.value(p) - sb.value(p)).abs() > 1e-3
        });
        assert!(differs, "two seeds made the same foam");
    }

    #[test]
    fn jitter_zero_is_a_periodic_foam() {
        // The ordered end of the family: seeds on a regular grid make a
        // Voronoi diagram that repeats with the cell, which a jittered one
        // never does. It is the cheapest possible check that the jitter is
        // wired to the seeds at all.
        let build = |jitter: f32| {
            Lattice::new(LatticeKind::Stochastic(Stochastic::VoronoiWall))
                .size(Vector3::new(9.0, 9.0, 9.0))
                .cells([3, 3, 3])
                .jitter(jitter)
                .seed(4)
                .thickness(0.4)
        };
        let cell = Vector3::new(3.0, 3.0, 3.0);
        let step = Vector3::new(0.05, 0.05, 0.05);
        let p = Vector3::new(0.7, -1.1, 0.4);
        let shifted = p + cell;
        for (jitter, want_same) in [(0.0f32, true), (1.0, false)] {
            let lattice = build(jitter);
            let sampler = lattice.sampler(cell, step, false);
            let half = sampler.half;
            let a = sampler.solid_at(p, half, f32::INFINITY);
            let b = sampler.solid_at(shifted, half, f32::INFINITY);
            assert_eq!(
                (a - b).abs() < 1e-4,
                want_same,
                "jitter {jitter}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn a_stretch_dominated_cell_beats_a_bending_dominated_one() {
        // The oldest result in the field, and the one a homogenisation has to
        // reproduce to be worth running: an octet truss carries load along its
        // struts, a BCC cell bends them, and at the same density the octet wins
        // by a lot.
        let modulus = |strut: Strut| {
            Lattice::new(LatticeKind::Strut(strut))
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([1, 1, 1])
                .fit_relative_density(0.25)
                .homogenize(12)
                .youngs_moduli()[2]
        };
        let octet = modulus(Strut::Octet);
        let bcc = modulus(Strut::Bcc);
        // The gap is wider than this in reality — a voxel grid this coarse
        // fattens every node and flatters the cell that depends on its nodes,
        // which is the bending one.
        assert!(octet > 1.5 * bcc, "octet {octet} vs bcc {bcc}");
        // And both are a fraction of the solid they are cut from.
        assert!(octet < 0.5, "{octet}");
    }

    #[test]
    fn density_buys_stiffness() {
        let modulus = |density: f32| {
            Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([1, 1, 1])
                .fit_relative_density(density)
                .homogenize(12)
                .youngs_moduli()[0]
        };
        let light = modulus(0.15);
        let heavy = modulus(0.45);
        assert!(heavy > 2.0 * light, "{light} then {heavy}");
    }

    #[test]
    fn a_lamellar_spinodal_is_anisotropic_and_an_isotropic_one_is_not() {
        let measure = |cell: Stochastic| {
            Lattice::new(LatticeKind::Stochastic(cell))
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([1, 1, 1])
                .seed(11)
                .fit_relative_density(0.3)
                .homogenize(12)
        };
        // Plates stacked across z: stiff in their own plane, soft across it.
        let lamellar = measure(Stochastic::SpinodalLamellar);
        let e = lamellar.youngs_moduli();
        assert!(e[0] > 2.0 * e[2], "in plane {} across {}", e[0], e[2]);
        // The isotropic class is not perfectly isotropic in one cell — it is a
        // random field, and one cell is one sample of it — but it is nothing
        // like as directional.
        let isotropic = measure(Stochastic::Spinodal);
        assert!(
            isotropic.anisotropy() < lamellar.anisotropy(),
            "isotropic {} vs lamellar {}",
            isotropic.anisotropy(),
            lamellar.anisotropy()
        );
    }

    #[test]
    fn a_field_grades_a_lattice_thicker_where_it_says() {
        let size = Vector3::new(20.0, 20.0, 20.0);
        let bounds = Box3::from_center_and_size(Vector3::ZERO, size);
        // Hot on the left, cold on the right — a stand-in for a solver.
        let field = Field::scattered(
            bounds,
            [8, 8, 8],
            &[
                (Vector3::new(-10.0, 0.0, 0.0), 100.0),
                (Vector3::new(10.0, 0.0, 0.0), 0.0),
            ],
        );
        let lattice = Lattice::new(LatticeKind::Strut(Strut::Octet))
            .bounds(bounds)
            .cells([4, 4, 4])
            .thickness(0.8)
            .grade(field.into_grade(2.0, 0.5));
        let cell = lattice.resolved_cell();
        let sampler = lattice.sampler(cell, Vector3::new(0.05, 0.05, 0.05), false);
        // The same point of the same cell, at both ends of the block: the
        // strut is thicker where the field is hot.
        let offset = Vector3::new(0.4, 0.4, 0.4);
        let hot = sampler.value(Vector3::new(-7.5, 0.0, 0.0) + offset);
        let cold = sampler.value(Vector3::new(7.5, 0.0, 0.0) + offset);
        assert!(hot > cold, "hot {hot} cold {cold}");
        assert!(lattice.build().index.is_some());
    }

    #[test]
    fn a_continuous_surface_conducts_better_per_gram_than_bars() {
        let measure = |kind| {
            Lattice::new(kind)
                .size(Vector3::new(10.0, 10.0, 10.0))
                .cells([1, 1, 1])
                .fit_relative_density(0.3)
                .conductivity(12)
        };
        let gyroid = measure(LatticeKind::Tpms(Tpms::Gyroid));
        let cubic = measure(LatticeKind::Strut(Strut::Cubic));
        // A sheet is one connected surface, so a share of it lies along every
        // direction at once; three sets of bars spend two thirds of themselves
        // across the gradient rather than along it.
        assert!(
            gyroid.tortuosity_factor(1.0) > cubic.tortuosity_factor(1.0),
            "gyroid {} vs cubic {}",
            gyroid.tortuosity_factor(1.0),
            cubic.tortuosity_factor(1.0)
        );
        for k in [gyroid, cubic] {
            // Both cells are cubic, so both conduct the same way on every axis.
            assert!((k.anisotropy() - 1.0).abs() < 0.05, "{}", k.anisotropy());
            // And nothing beats the solid it is cut from.
            assert!(k.axes()[0] < k.relative_density, "{:?}", k.axes());
            assert!(k.tortuosity_factor(1.0) < 1.0);
        }
    }

    #[test]
    fn plates_conduct_along_themselves_and_not_across() {
        let k = Lattice::new(LatticeKind::Stochastic(Stochastic::SpinodalLamellar))
            .size(Vector3::new(10.0, 10.0, 10.0))
            .cells([1, 1, 1])
            .seed(7)
            .fit_relative_density(0.3)
            .conductivity(12);
        let along = k.axes();
        assert!(along[0] > 0.1, "in plane {}", along[0]);
        assert!(along[2] < 0.01, "across {}", along[2]);
        assert!(k.anisotropy() > 50.0, "{}", k.anisotropy());
        // Nearly every gram is in a plate, and every plate runs along the flow.
        assert!(k.tortuosity_factor(1.0) > 0.9, "{}", k.tortuosity_factor(1.0));
        // A random field's plates are not laid out on the axes to the last
        // degree, so the tensor has small off-diagonal terms and the principal
        // conductivity is a little above the best axis — never below it, which
        // is what the largest eigenvalue of a symmetric matrix always is.
        let principal = k.principal();
        let best_axis = along[0].max(along[1]);
        assert!(principal[0] >= best_axis - 1e-9, "{principal:?} vs {along:?}");
        assert!(principal[0] < best_axis * 1.1, "{principal:?} vs {along:?}");
    }

    #[test]
    fn every_generator_homogenizes_to_a_real_material() {
        // Broad rather than deep: at eight voxels a cell none of these numbers
        // is quotable, but a generator that voxelises to nothing, or comes back
        // soft in every direction at once, is broken — and that is exactly what
        // concentric infill did when it was handed a depth of infinity.
        for kind in LatticeKind::all() {
            let lattice = Lattice::new(kind)
                .size(Vector3::new(8.0, 8.0, 8.0))
                .cells([4, 4, 4])
                .seed(3)
                .fit_relative_density(0.3);
            let c = lattice.homogenize(8);
            let name = kind.name();
            assert!(
                (c.relative_density - 0.3).abs() < 0.1,
                "{name}: voxelised to {}",
                c.relative_density
            );
            let e = c.youngs_moduli();
            assert!(e.iter().all(|v| v.is_finite()), "{name}: {e:?}");
            // Some cells really are soft on an axis — a stack of plates has
            // nothing across it — but nothing here is soft on all three.
            assert!(
                e.iter().any(|&v| v > 0.01),
                "{name}: soft in every direction, {e:?}"
            );
        }
    }

    #[test]
    fn a_grade_is_counted_in_the_wall_it_leaves() {
        let build = |graded: bool| {
            let lattice = Lattice::new(LatticeKind::Strut(Strut::Octet))
                .size(Vector3::new(20.0, 20.0, 20.0))
                .cells([4, 4, 4])
                .thickness(1.0)
                .resolution(10);
            if graded {
                // Two fifths at one end, untouched at the other.
                lattice.grade(|p| if p.x < 0.0 { 0.4 } else { 1.0 })
            } else {
                lattice
            }
        };
        let plain = build(false).wall_samples();
        let graded = build(true).wall_samples();
        assert!(
            (graded - 0.4 * plain).abs() < 0.05 * plain,
            "plain {plain} graded {graded}"
        );

        // And the resolution it asks for follows the thin end, not the nominal
        // one: sizing for the nominal wall is how a graded part comes out as
        // gravel where it is thinnest.
        let resolved = build(true).resolve_walls(3.0);
        assert!(resolved.wall_samples() >= 3.0, "{}", resolved.wall_samples());
        let nominal = build(false).resolve_walls(3.0);
        assert!(
            resolved.sample_grid()[0] > nominal.sample_grid()[0],
            "{:?} vs {:?}",
            resolved.sample_grid(),
            nominal.sample_grid()
        );
    }

    #[test]
    fn a_region_can_be_used_twice() {
        // Cloning is a reference count, so filling a shell and conforming to
        // the same surface costs one region, not two.
        let shell = Region::sphere(Vector3::ZERO, 8.0)
            .difference(Region::sphere(Vector3::ZERO, 5.0));
        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .conform(Conform::depth(shell.clone()))
            .fill(shell)
            .cell_size(Vector3::new(3.0, 3.0, 1.5))
            .thickness(0.4)
            .resolution(10)
            .build();
        assert_watertight(&geom, "shell used twice");
    }

    #[test]
    fn a_stretching_cell_uses_more_of_itself_than_a_bending_one() {
        // The strength counterpart of the stiffness ordering: an octet loads
        // its struts along their length, a BCC cell bends them, and a bent
        // strut has only its outer fibres working.
        let measure = |strut| {
            Lattice::new(LatticeKind::Strut(strut))
                .size(Vector3::new(10.0, 10.0, 10.0))
                .cells([1, 1, 1])
                .fit_relative_density(0.3)
                .strength(16)
        };
        let octet = measure(Strut::Octet);
        let bcc = measure(Strut::Bcc);
        assert!(octet.resolved(), "octet unresolved: {:?}", octet.efficiency());
        assert!(bcc.resolved(), "bcc unresolved: {:?}", bcc.efficiency());
        assert!(
            octet.efficiency()[2] > bcc.efficiency()[2],
            "octet {:?} bcc {:?}",
            octet.efficiency(),
            bcc.efficiency()
        );

        // Strength is a fraction of the solid's, and it scales with it.
        let at_500 = octet.uniaxial(500.0)[2];
        let at_1000 = octet.uniaxial(1000.0)[2];
        assert!((at_1000 - 2.0 * at_500).abs() < 1e-6);
        assert!(at_500 > 0.0 && at_500 < 500.0 * 0.3, "{at_500}");
        // And the stiffness came back in the same result rather than needing
        // a second solve.
        assert!(octet.stiffness.youngs_moduli()[2] > 0.0);
        assert_eq!(octet.stiffness.voxels, 16);
    }

    #[test]
    fn every_generator_is_named_once() {
        // The names are what a CLI or a log line selects on, so two generators
        // answering to one name would be a real ambiguity — and `from_name`
        // would silently hand back the wrong family.
        let all = LatticeKind::all();
        assert_eq!(
            all.len(),
            Tpms::ALL.len()
                + Strut::ALL.len()
                + Infill::ALL.len()
                + Cuboct::ALL.len()
                + Stochastic::ALL.len()
        );
        let mut names: Vec<&str> = all.iter().map(|k| k.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate generator name");

        for kind in all {
            assert_eq!(LatticeKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(LatticeKind::from_name("gyroidal"), None);
    }

    #[test]
    fn building_twice_gives_the_same_mesh() {
        // The sample grid is filled a slab at a time, in parallel when the
        // feature is on. Vertex order has to come out identical anyway — a
        // build flag or a thread schedule that reordered the mesh would break
        // every downstream diff and cache.
        let build = || {
            Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(3.0, 3.0, 3.0))
                .cells([3, 3, 3])
                .thickness(0.2)
                .resolution(14)
                .trim(|p| 1.4 - p.length())
                .build()
        };
        let (a, b) = (build(), build());
        assert_eq!(a.index, b.index);
        for name in ["position", "normal", "uv"] {
            assert_eq!(
                a.get_attribute(name).unwrap().array,
                b.get_attribute(name).unwrap().array,
                "{name} differs between builds"
            );
        }
    }

    #[test]
    fn resolve_walls_puts_samples_across_the_wall() {
        // A quarter-millimetre wall in a 6.7 mm cell is under one sample wide
        // at the default resolution, which is where lattices come out as
        // gravel.
        let thin = Lattice::new(LatticeKind::Tpms(Tpms::Lidinoid))
            .size(Vector3::new(20.0, 20.0, 20.0))
            .cells([3, 3, 3])
            .thickness(0.25);
        assert!(thin.wall_samples() < 1.0, "{}", thin.wall_samples());

        let resolved = Lattice::new(LatticeKind::Tpms(Tpms::Lidinoid))
            .size(Vector3::new(20.0, 20.0, 20.0))
            .cells([3, 3, 3])
            .thickness(0.25)
            // A wall that thin over a part that big needs the budget raised
            // with it — otherwise the ceiling takes the resolution back.
            .max_samples(64_000_000)
            .resolve_walls(4.0);
        assert!(
            resolved.wall_samples() >= 4.0,
            "only {} samples",
            resolved.wall_samples()
        );
        // And it only ever raises the resolution.
        assert!(resolved.sample_grid()[0] > thin.sample_grid()[0]);
    }

    #[test]
    fn the_sample_budget_still_wins_over_resolve_walls() {
        // Asking for a wall the budget cannot afford must not blow the budget;
        // `wall_samples` then reports what was actually possible.
        let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(20.0, 20.0, 20.0))
            .cells([3, 3, 3])
            .thickness(0.05)
            .max_samples(150_000)
            .resolve_walls(8.0);
        let dims = lattice.sample_grid();
        assert!(dims[0] * dims[1] * dims[2] <= 150_000, "{dims:?}");
        assert!(lattice.wall_samples() < 8.0);
    }

    #[test]
    fn solid_style_builds_watertight() {
        for t in [Tpms::Gyroid, Tpms::SchwarzP, Tpms::IWP] {
            let geom = Lattice::new(LatticeKind::Tpms(t))
                .size(Vector3::new(2.0, 2.0, 2.0))
                .cells([2, 2, 2])
                .style(LatticeStyle::Solid)
                .thickness(0.0)
                .resolution(12)
                .build();
            assert_watertight(&geom, t.name());
        }
    }

    #[test]
    fn geometry_stays_inside_the_bounds() {
        let geom = geom_of(LatticeKind::Tpms(Tpms::Gyroid));
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            for a in v {
                assert!(a.abs() <= 1.0 + 1e-3, "{a} escapes the box");
            }
        }
    }

    #[test]
    fn attributes_are_complete() {
        let geom = geom_of(LatticeKind::Strut(Strut::Octet));
        let count = geom.get_attribute("position").unwrap().count();
        assert_eq!(geom.get_attribute("normal").unwrap().count(), count);
        assert_eq!(geom.get_attribute("uv").unwrap().count(), count);
        for i in geom.index.as_ref().unwrap() {
            assert!((*i as usize) < count);
        }
    }

    #[test]
    fn thicker_walls_mean_more_material() {
        let base = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2]);
        let thin = base.thickness(0.2).relative_density();
        let thick = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .thickness(0.6)
            .relative_density();
        assert!(thin < thick, "{thin} !< {thick}");
        assert!(thin > 0.0 && thick < 1.0);
    }

    #[test]
    fn density_fit_lands_on_target() {
        for kind in [
            LatticeKind::Tpms(Tpms::Gyroid),
            LatticeKind::Tpms(Tpms::Diamond),
            LatticeKind::Strut(Strut::Octet),
            LatticeKind::Strut(Strut::Bcc),
            LatticeKind::Cuboct(Cuboct::Rigid),
            LatticeKind::Infill(Infill::Grid),
            LatticeKind::Infill(Infill::Honeycomb),
        ] {
            for target in [0.2f32, 0.45] {
                let lattice = Lattice::new(kind)
                    .size(Vector3::new(4.0, 4.0, 4.0))
                    .cells([2, 2, 2])
                    .fit_relative_density(target);
                let got = lattice.relative_density();
                assert!(
                    (got - target).abs() < 0.01,
                    "{} wanted {target}, got {got} at thickness {}",
                    kind.name(),
                    lattice.current_thickness()
                );
                assert!(lattice.current_thickness() > 0.0);
            }
        }
    }

    #[test]
    fn every_tpms_builds_solid_as_well_as_sheet() {
        for t in Tpms::ALL {
            let geom = Lattice::new(LatticeKind::Tpms(t))
                .size(Vector3::new(2.0, 2.0, 2.0))
                .cells([2, 2, 2])
                .style(LatticeStyle::Solid)
                .thickness(0.0)
                .resolution(12)
                .build();
            assert_watertight(&geom, t.name());
        }
    }

    #[test]
    fn solid_density_climbs_with_the_offset() {
        for t in [Tpms::Gyroid, Tpms::Diamond, Tpms::IWP] {
            let density = |offset: f32| {
                Lattice::new(LatticeKind::Tpms(t))
                    .size(Vector3::new(4.0, 4.0, 4.0))
                    .cells([2, 2, 2])
                    .style(LatticeStyle::Solid)
                    .thickness(offset)
                    .relative_density()
            };
            let steps: Vec<f32> = [-1.5f32, -0.75, 0.0, 0.75, 1.5]
                .into_iter()
                .map(density)
                .collect();
            for pair in steps.windows(2) {
                assert!(
                    pair[1] > pair[0],
                    "{}: density went {:?} as the offset grew",
                    t.name(),
                    steps
                );
            }
            assert!(steps[0] < 0.2 && *steps.last().unwrap() > 0.8, "{steps:?}");
        }
    }

    #[test]
    fn solid_style_fits_a_density_target() {
        for t in [Tpms::Gyroid, Tpms::SchwarzP, Tpms::Diamond] {
            for target in [0.2f32, 0.5, 0.8] {
                let lattice = Lattice::new(LatticeKind::Tpms(t))
                    .size(Vector3::new(4.0, 4.0, 4.0))
                    .cells([2, 2, 2])
                    .style(LatticeStyle::Solid)
                    .fit_relative_density(target);
                let got = lattice.relative_density();
                assert!(
                    (got - target).abs() < 0.02,
                    "{} solid wanted {target}, got {got} at offset {}",
                    t.name(),
                    lattice.current_thickness()
                );
            }
        }
    }

    #[test]
    fn a_solid_lattice_fills_a_region_and_takes_a_skin() {
        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([2, 2, 2])
            .style(LatticeStyle::Solid)
            .thickness(0.0)
            .resolution(14)
            .skin(0.3)
            .build();
        assert_watertight(&geom, "solid in a sphere with a skin");
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            let r = Vector3::new(v[0], v[1], v[2]).length();
            assert!(r <= 3.05, "{r} escaped the sphere");
        }
    }

    #[test]
    fn the_wall_guard_knows_a_solid_lattice_has_no_wall() {
        // `thickness` means an offset here, not a wall width. Reading it as one
        // says "zero samples across the wall" for the plainest solid lattice
        // there is, and demands a preposterous resolution for a thin offset —
        // 2.5 samples across a 0.05 mm offset in a 2 mm cell is 100 samples per
        // cell, for a feature that is not there.
        let solid = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .style(LatticeStyle::Solid)
            .thickness(0.05);
        assert!(
            solid.wall_samples().is_infinite(),
            "a solid lattice has no wall to resolve, got {}",
            solid.wall_samples()
        );
        let asked = solid.resolution;
        assert_eq!(
            Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([2, 2, 2])
                .style(LatticeStyle::Solid)
                .thickness(0.05)
                .resolve_walls(2.5)
                .resolution,
            asked,
            "resolve_walls should leave a solid lattice alone"
        );

        // The sheet of the same lattice does have a wall, and still reports it.
        let sheet = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .thickness(0.05);
        assert!(sheet.wall_samples().is_finite() && sheet.wall_samples() < 1.0);
    }

    #[test]
    fn the_density_fit_reaches_targets_outside_one_cell_of_thickness() {
        // One cell is the scale of a first guess, not a bound. Bisecting inside
        // a bracket that does not contain the answer converges tidily onto the
        // bracket's own end and hands it back as though it were the answer —
        // which is how a sheet asked for 97 % used to come back with 94 %, and
        // a solid asked for 2 % with 3 %.
        for (style, targets) in [
            (LatticeStyle::Sheet, [0.02f32, 0.97]),
            (LatticeStyle::Solid, [0.02, 0.97]),
        ] {
            for target in targets {
                let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                    .size(Vector3::new(4.0, 4.0, 4.0))
                    .cells([2, 2, 2])
                    .style(style)
                    .fit_relative_density(target);
                let got = lattice.relative_density();
                assert!(
                    (got - target).abs() < 0.02,
                    "{style:?} wanted {target}, got {got} at {}",
                    lattice.current_thickness()
                );
            }
        }
    }

    #[test]
    fn grading_a_solid_lattice_needs_an_offset_to_scale() {
        // Documented rather than fixed, because the parameterisation is what
        // is wrong: a solid lattice's zero thickness means *half solid*, so
        // there is nothing for a multiplier to act on. The test is here so the
        // behaviour cannot change without someone noticing.
        let halves = |offset: f32| {
            let side = |lo: f32, hi: f32| {
                Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                    .bounds(Box3::new(
                        Vector3::new(lo, -1.0, -1.0),
                        Vector3::new(hi, 1.0, 1.0),
                    ))
                    .cell_size(Vector3::new(1.0, 1.0, 1.0))
                    .style(LatticeStyle::Solid)
                    .thickness(offset)
                    .grade(|p| (p.x + 2.0) * 0.5)
                    .relative_density()
            };
            (side(-2.0, 0.0), side(0.0, 2.0))
        };

        // At the default offset the grade cannot bite.
        let (left, right) = halves(0.0);
        assert!((left - right).abs() < 1e-6, "{left} vs {right}");
        // Given something to scale, it does.
        let (left, right) = halves(0.3);
        assert!(right > left + 0.1, "{left} vs {right}");
    }

    #[test]
    fn every_combination_comes_out_closed() {
        // Generator × style × skin × fill, all of it. The interesting failures
        // are at the joins — a skin meeting a lattice, a fill cutting one — and
        // there are too many pairs to have an opinion about each.
        for kind in LatticeKind::all() {
            for style in [LatticeStyle::Sheet, LatticeStyle::Solid] {
                // Only a level set reads the style; for the others a zero
                // thickness would just mean no lattice.
                let solid_tpms = style == LatticeStyle::Solid
                    && match kind {
                        LatticeKind::Tpms(_) => true,
                        LatticeKind::Stochastic(s) => s.is_field(),
                        _ => false,
                    };
                for skin in [0.0f32, 0.25] {
                    for filled in [false, true] {
                        let lattice = Lattice::new(kind)
                            .size(Vector3::new(3.0, 3.0, 3.0))
                            .cells([2, 2, 2])
                            .style(style)
                            .thickness(if solid_tpms { 0.0 } else { 0.35 })
                            .resolution(10)
                            .skin(skin);
                        let lattice = if filled {
                            lattice.fill(Region::sphere(Vector3::ZERO, 1.5))
                        } else {
                            lattice
                        };
                        let geom = lattice.build();
                        let what = format!("{} {style:?} skin={skin} fill={filled}", kind.name());
                        assert_watertight(&geom, &what);
                        for name in ["position", "normal", "uv"] {
                            let attr = geom.get_attribute(name).unwrap();
                            assert!(
                                attr.array.iter().all(|v| v.is_finite()),
                                "{what}: non-finite {name}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_settings_produce_a_mesh_or_nothing_but_never_a_panic() {
        let cases: Vec<(&str, Lattice<'_>)> = vec![
            (
                "no cells",
                Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                    .size(Vector3::new(2.0, 2.0, 2.0))
                    .cells([0, 0, 0])
                    .thickness(0.2),
            ),
            (
                "no size",
                Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                    .size(Vector3::ZERO)
                    .cells([2, 2, 2])
                    .thickness(0.2),
            ),
            (
                "inside-out bounds",
                Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                    .size(Vector3::new(-2.0, -2.0, -2.0))
                    .cells([2, 2, 2])
                    .thickness(0.2),
            ),
            (
                "thickness past the cell",
                Lattice::new(LatticeKind::Strut(Strut::Bcc))
                    .size(Vector3::new(2.0, 2.0, 2.0))
                    .cells([2, 2, 2])
                    .thickness(500.0)
                    .resolution(8),
            ),
        ];
        for (what, lattice) in cases {
            let geom = lattice.build();
            if let Some(pos) = geom.get_attribute("position") {
                assert!(
                    pos.array.iter().all(|v| v.is_finite()),
                    "{what}: non-finite position"
                );
            }
            assert!(lattice.relative_density().is_finite(), "{what}: density");
        }
    }

    #[test]
    fn solid_style_at_zero_thickness_is_half_dense() {
        // The minimal surface splits space evenly, so filling one side of it
        // and offsetting by nothing has to come out near half.
        let density = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .style(LatticeStyle::Solid)
            .thickness(0.0)
            .relative_density();
        assert!((density - 0.5).abs() < 0.05, "{density}");
    }

    #[test]
    fn grading_moves_material_across_the_part() {
        // Vertex counts will not show this: a sheet's area barely changes with
        // its thickness, so the wall has to be weighed, not counted.
        let half = |lo: f32, hi: f32| {
            Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .bounds(Box3::new(
                    Vector3::new(lo, -1.0, -1.0),
                    Vector3::new(hi, 1.0, 1.0),
                ))
                .cell_size(Vector3::new(1.0, 1.0, 1.0))
                .thickness(0.2)
                // Nothing at the left edge, double at the right.
                .grade(|p| (p.x + 2.0) * 0.5)
                .relative_density()
        };
        let (left, right) = (half(-2.0, 0.0), half(0.0, 2.0));
        assert!(
            right > left * 2.0,
            "grading did not bias the wall: {left} / {right}"
        );

        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 2.0, 2.0))
            .cell_size(Vector3::new(1.0, 1.0, 1.0))
            .thickness(0.2)
            .resolution(12)
            .grade(|p| (p.x + 2.0) * 0.5)
            .build();
        assert_watertight(&geom, "graded");
    }

    #[test]
    fn filling_a_region_takes_its_bounds_and_its_shape() {
        let radius = 3.0f32;
        let lattice = Lattice::new(LatticeKind::Strut(Strut::Octet))
            .fill(Region::sphere(Vector3::new(1.0, 0.0, -0.5), radius))
            .cells([3, 3, 3])
            .thickness(0.4)
            .resolution(14);
        // The region sized the domain; nothing else was said.
        let bounds = Box3::from_center_and_size(
            Vector3::new(1.0, 0.0, -0.5),
            Vector3::new(1.0, 1.0, 1.0) * (radius * 2.0),
        );
        assert_eq!(lattice.bounds, bounds);

        let geom = lattice.build();
        assert_watertight(&geom, "sphere fill");
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            let r = (Vector3::new(v[0], v[1], v[2]) - Vector3::new(1.0, 0.0, -0.5)).length();
            assert!(r <= radius + 1e-2, "{r} escaped the sphere");
        }
    }

    #[test]
    fn an_explicit_box_beats_the_region_either_way_round() {
        let want = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
        let region = || Region::sphere(Vector3::ZERO, 8.0);
        let after = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(region())
            .bounds(want);
        let before = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .bounds(want)
            .fill(region());
        assert_eq!(after.bounds, want);
        assert_eq!(before.bounds, want);
    }

    #[test]
    fn a_lattice_fills_a_swept_curve() {
        use crate::curves::{CatmullRomCurve3, Curve3};
        let curve = CatmullRomCurve3::new(vec![
            Vector3::new(-3.0, 0.0, 0.0),
            Vector3::new(-1.0, 1.5, 0.5),
            Vector3::new(1.0, -1.5, -0.5),
            Vector3::new(3.0, 0.0, 0.0),
        ]);
        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(Region::tube(&curve, 1.0, 48))
            .cell_size(Vector3::new(0.9, 0.9, 0.9))
            .thickness(0.14)
            .resolution(12)
            .build();
        assert_watertight(&geom, "tube fill");
        // Nothing outside the tube: every vertex is within its radius of the
        // curve, give or take the polyline's own corner-cutting.
        let samples: Vec<Vector3> = (0..=64).map(|i| curve.get_point(i as f32 / 64.0)).collect();
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            let p = Vector3::new(v[0], v[1], v[2]);
            let near = samples
                .iter()
                .map(|s| (p - *s).length())
                .fold(f32::MAX, f32::min);
            assert!(near <= 1.05, "{near} outside the tube");
        }
    }

    #[test]
    fn density_is_measured_against_the_part_not_its_box() {
        // A sphere is π/6 of its box. If density counted the box, filling the
        // sphere solid would read 52 % and a target above that would be
        // unreachable no matter how thick the walls were made.
        let solid_sphere = Lattice::new(LatticeKind::Strut(Strut::Cubic))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([2, 2, 2])
            // Struts far thicker than the cell: the sphere fills in completely.
            .thickness(20.0)
            .relative_density();
        assert!(solid_sphere > 0.99, "{solid_sphere} — the part is full");

        // And a target above the box fraction is reachable.
        let fitted = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([2, 2, 2])
            .fit_relative_density(0.7);
        assert!((fitted.relative_density() - 0.7).abs() < 0.02);

        // Without a region the part *is* the box, so nothing changes there.
        let boxed = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .fit_relative_density(0.3);
        assert!((boxed.relative_density() - 0.3).abs() < 0.02);
    }

    #[test]
    fn a_skin_wraps_the_fill_and_bonds_to_it() {
        let plain = Lattice::new(LatticeKind::Strut(Strut::Bcc))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([3, 3, 3])
            .thickness(0.3)
            .resolution(14);
        let bare = plain.relative_density();

        let skinned = Lattice::new(LatticeKind::Strut(Strut::Bcc))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([3, 3, 3])
            .thickness(0.3)
            .resolution(14)
            .skin(0.4);
        assert!(
            skinned.relative_density() > bare,
            "a skin adds material: {bare} to {}",
            skinned.relative_density()
        );

        let geom = skinned.build();
        assert_watertight(&geom, "skinned");
        // The skin is solid right at the surface, wherever you look.
        for dir in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.577, 0.577, 0.577),
        ] {
            let just_inside = dir.normalize() * 2.9;
            let just_outside = dir.normalize() * 3.2;
            let sampler = skinned.grid().1;
            assert!(
                sampler.value(just_inside) > 0.0,
                "skin has a gap at {dir:?}"
            );
            assert!(sampler.value(just_outside) < 0.0, "skin bulges at {dir:?}");
        }
    }

    #[test]
    fn a_region_with_nothing_in_it_builds_nothing() {
        // Two spheres that do not touch: the intersection is empty and its
        // bounds come out inverted, which a grid built from them would not
        // survive.
        let empty = Region::sphere(Vector3::new(-9.0, 0.0, 0.0), 1.0)
            .intersection(Region::sphere(Vector3::new(9.0, 0.0, 0.0), 1.0));
        assert!(empty.bounds().is_empty());

        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(empty)
            .cells([2, 2, 2])
            .thickness(0.1)
            .build();
        assert!(geom.get_attribute("position").is_none());
    }

    #[test]
    fn a_clip_cuts_after_the_skin_rather_than_before() {
        // Sectioning half a skinned ball away: the skin has to stop at the cut
        // face, not wrap round it, or there is nothing to see inside.
        let keep_negative_x = || {
            Region::cuboid(Box3::new(
                Vector3::new(0.0, -9.0, -9.0),
                Vector3::new(9.0, 9.0, 9.0),
            ))
            .invert()
        };
        let sectioned = Lattice::new(LatticeKind::Strut(Strut::Bcc))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([3, 3, 3])
            .thickness(0.5)
            .resolution(16)
            .skin(0.4)
            .clip(keep_negative_x());
        let sampler = sectioned.grid().1;

        // Solid skin on the kept side…
        assert!(sampler.value(Vector3::new(-2.9, 0.0, 0.0)) > 0.0);
        // …nothing at all on the cut side…
        assert!(sampler.value(Vector3::new(2.9, 0.0, 0.0)) < 0.0);
        // …and the cut face is open where the lattice's own voids are, which is
        // what "the skin did not follow the cut" means. Just inside the face,
        // between struts, there is void.
        let void = (0..40)
            .map(|i| Vector3::new(-0.05, -2.6 + i as f32 * 0.13, 0.0))
            .any(|p| sampler.value(p) < 0.0);
        assert!(void, "the cut face was skinned over");

        assert_watertight(&sectioned.build(), "sectioned");

        // And a clip is a view of the part, not a change to it: the density is
        // still the density the part was built at.
        let whole = Lattice::new(LatticeKind::Strut(Strut::Bcc))
            .fill(Region::sphere(Vector3::ZERO, 3.0))
            .cells([3, 3, 3])
            .thickness(0.5)
            .resolution(16)
            .skin(0.4);
        assert!((sectioned.relative_density() - whole.relative_density()).abs() < 1e-6);
    }

    #[cfg(feature = "mesh-bvh")]
    #[test]
    fn a_lattice_fills_a_mesh() {
        use crate::geometries::SphereGeometry;
        // A mesh of a sphere, so the answer is one we already know.
        let sphere = SphereGeometry::new(2.0, 32, 24);
        let region = Region::mesh(&sphere).expect("a bvh over a sphere");
        assert!(region.contains(Vector3::ZERO));
        assert!(!region.contains(Vector3::new(3.0, 0.0, 0.0)));
        assert!((region.distance(Vector3::ZERO) - 2.0).abs() < 0.05);

        let geom = Lattice::new(LatticeKind::Strut(Strut::Octet))
            .fill(region)
            .cells([2, 2, 2])
            .thickness(0.3)
            .resolution(12)
            .build();
        assert_watertight(&geom, "mesh fill");
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            let r = Vector3::new(v[0], v[1], v[2]).length();
            assert!(r <= 2.05, "{r} escaped the mesh");
        }
    }

    #[cfg(feature = "openscad")]
    #[test]
    fn a_lattice_fills_a_scad_model() {
        let region = Region::scad("difference(){ cube(20, center=true); sphere(12, $fn=32); }")
            .expect("valid scad");
        // A cube with a sphere bitten out: the corners survive, the middle does
        // not.
        assert!(!region.contains(Vector3::ZERO));
        assert!(region.contains(Vector3::new(9.0, 9.0, 9.0)));

        let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .fill(region)
            .cells([3, 3, 3])
            .thickness(0.7)
            .resolution(10)
            .build();
        assert_watertight(&geom, "scad fill");
    }

    #[test]
    fn trimming_clips_to_the_field() {
        let radius = 0.8f32;
        let geom = Lattice::new(LatticeKind::Strut(Strut::Bcc))
            .size(Vector3::new(2.0, 2.0, 2.0))
            .cells([2, 2, 2])
            .thickness(0.16)
            .resolution(16)
            .trim(move |p| radius - p.length())
            .build();
        assert_watertight(&geom, "trimmed");
        let pos = geom.get_attribute("position").unwrap();
        for v in pos.array.chunks_exact(3) {
            let r = Vector3::new(v[0], v[1], v[2]).length();
            assert!(r <= radius + 1e-2, "{r} outside the trim sphere");
        }
    }

    #[test]
    fn anisotropic_cells_stretch_the_pattern() {
        // One cell across x, four across z: the surface must repeat four times
        // as often along z as along x.
        let crossings = |geom: &BufferGeometry, axis: usize| {
            let pos = geom.get_attribute("position").unwrap();
            let mut lo = 0;
            for v in pos.array.chunks_exact(3) {
                if v[axis] < 0.0 {
                    lo += 1;
                }
            }
            lo
        };
        let geom = Lattice::new(LatticeKind::Tpms(Tpms::SchwarzP))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cell_size(Vector3::new(4.0, 4.0, 1.0))
            .thickness(0.2)
            .resolution(10)
            .build();
        assert_watertight(&geom, "anisotropic");
        // Half the vertices below the midplane on both axes — the check here is
        // that it builds at all with a cell as wide as the box.
        assert!(crossings(&geom, 0) > 0 && crossings(&geom, 2) > 0);
    }

    #[test]
    fn zero_thickness_makes_nothing() {
        let geom = Lattice::new(LatticeKind::Strut(Strut::Cubic))
            .size(Vector3::new(2.0, 2.0, 2.0))
            .cells([2, 2, 2])
            .thickness(0.0)
            .build();
        assert!(geom.get_attribute("position").is_none());
    }

    #[test]
    fn the_sample_budget_is_respected() {
        // A resolution that would want a billion samples has to come back with
        // a mesh, not an allocation failure.
        let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(100.0, 100.0, 100.0))
            .cell_size(Vector3::new(10.0, 10.0, 10.0))
            .thickness(2.0)
            .resolution(64)
            .max_samples(200_000);
        let dims = lattice.sample_grid();
        let total = dims[0] * dims[1] * dims[2];
        assert!(total <= 200_000, "{dims:?} over budget");
        // Not merely under budget — still using most of it.
        assert!(total > 100_000, "{dims:?} gave up too much");
        assert!(!lattice.build().index.as_ref().unwrap().is_empty());
    }

    #[test]
    fn the_budget_wins_over_the_sampling_rate() {
        // A hundred cells across the part cannot be sampled twice per cell
        // inside a thousand samples, and something has to give. It used to be
        // the budget: the grid came back 206³ — over by a factor of 8700 — on
        // the reasoning that two samples per cell was a floor. That is not a
        // budget, and on a part a few thousand cells across the same reasoning
        // produced a grid whose sample count overflowed the multiply that sizes
        // the allocation, which is a panic rather than a large mesh.
        let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(100.0, 100.0, 100.0))
            .cell_size(Vector3::new(1.0, 1.0, 1.0))
            .max_samples(1_000);
        let dims = lattice.sample_grid();
        let total = dims[0] * dims[1] * dims[2];
        assert!(total <= 1_000, "{dims:?} is {total}, over a budget of 1000");
        // The sampling paid for it, and `wall_samples` is where that shows.
        assert!(lattice.thickness(0.2).wall_samples() < 1.0);
    }

    #[test]
    fn an_unpayable_grid_comes_back_as_a_mesh_rather_than_a_panic() {
        // Cells a millionth the size of the part — the shape of a units mix-up,
        // metres against millimetres. Every axis wants tens of millions of
        // samples and the product overflows a `usize`.
        for lattice in [
            Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(2.0, 2.0, 2.0))
                .cell_size(Vector3::ZERO)
                .thickness(0.2),
            Lattice::new(LatticeKind::Strut(Strut::Octet))
                .size(Vector3::new(1000.0, 1000.0, 1000.0))
                .cell_size(Vector3::new(0.001, 0.001, 0.001))
                .thickness(0.0005),
        ] {
            let lattice = lattice.max_samples(200_000);
            let dims = lattice.sample_grid();
            let total = dims[0] * dims[1] * dims[2];
            assert!(total <= 200_000, "{dims:?} is {total}");
            // Both of these used to panic in `build`, before they got here.
            let geom = lattice.build();
            if let Some(pos) = geom.get_attribute("position") {
                assert!(pos.array.iter().all(|v| v.is_finite()));
            }
            // The density path sizes its own vector the same way.
            assert!(lattice.relative_density().is_finite());
        }
    }

    #[test]
    fn the_one_shot_constructor_matches_the_builder() {
        let kind = LatticeKind::Tpms(Tpms::Gyroid);
        let size = Vector3::new(2.0, 2.0, 2.0);
        let direct = LatticeGeometry::new(kind, size, [2, 2, 2], 0.14);
        let built = Lattice::new(kind)
            .size(size)
            .cells([2, 2, 2])
            .thickness(0.14)
            .build();
        assert_eq!(
            direct.get_attribute("position").unwrap().count(),
            built.get_attribute("position").unwrap().count()
        );
    }
}

