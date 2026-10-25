# Lattice infill

Part of the [threers](../README.md) documentation.

# Lattice infill

35 lattices, all unconditional — no feature flag and no extra dependencies.

```bash
cargo run --release --example lattice   # every generator at 25 % density → out/lattice.png
cargo run --release --example cuboct    # cuboct voxels (Jenett 2020) → out/cuboct.png
cargo run --release --example lattice_engineering   # foams, conforming, metrics, stiffness
```

```rust
use threers::{Infill, Lattice, LatticeKind, Strut, Tpms, Vector3};

// A 20 mm cube of gyroid, 5 mm cells, 0.8 mm walls.
let gyroid = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cell_size(Vector3::new(5.0, 5.0, 5.0))
    .thickness(0.8)
    .build();

// Or say what a slicer would say — 25 % infill — and let it solve the
// strut diameter that gets there.
let octet = Lattice::new(LatticeKind::Strut(Strut::Octet))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .fit_relative_density(0.25);
let diameter = octet.current_thickness();
let mesh = octet.build();

// Face-connected cuboct voxel: the same assembly, four metamaterial
// behaviours. `shape` is amplitude / reentrant indent / chiral radius
// as a fraction of cell pitch.
let auxetic = Lattice::new(LatticeKind::Cuboct(threers::Cuboct::Auxetic))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .shape(0.2)
    .fit_relative_density(0.2)
    .build();

// Discrete assembly: six face parts per voxel, exploded so the joints show.
let exploded = threers::CuboctAssembly::new(threers::Cuboct::Rigid)
    .pitch(20.0)
    .cells([2, 2, 2])
    .explode(0.25)
    .build();

// Denser on the right than the left, and poured into a sphere.
let graded = Lattice::new(LatticeKind::Infill(Infill::Honeycomb))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .thickness(0.6)
    .grade(|p| 1.0 + (p.x / 20.0 + 0.5) * 2.0)
    .trim(|p| 10.0 - p.length())
    .build();
```

| Family | Generators |
|--------|-----------|
| `Tpms` | `Gyroid`, `SchwarzP`, `Diamond`, `Neovius`, `IWP`, `FischerKochS`, `Lidinoid`, `SplitP`, `Helicoid` |
| `Strut` | `Cubic`, `Bcc`, `BccZ`, `Fcc`, `Octet`, `Diamond`, `Kelvin`; textiles `Weave`, `Braid`, `HelixBundle` |
| `Cuboct` | `Rigid`, `Compliant`, `Auxetic`, `ChiralCw`, `ChiralCcw` |
| `Infill` | `Rectilinear`, `AlignedRectilinear`, `Grid`, `Triangles`, `TriHexagon`, `Honeycomb`, `Cubic`, `QuarterCubic`, `Concentric` |

`LatticeKind::from_name` takes any of those names, so a `--lattice gyroid` flag
needs no table of its own.

The three textiles are yarns rather than beams: their strands pass one another
instead of meeting at nodes, so below a certain diameter a textile is several
interlocked pieces — a cloth, a plait, a rope — and above it they fuse where
they cross. The diameters are a quarter of a cell for `Weave` and about 0.15
for `Braid` and `HelixBundle` (`Strut::is_textile` tells them apart).

![A plain weave, a braid, a six-fibre rope and helicoid screws; below, a spinodal, a broadband Gaussian field, hyperuniform fibres and a reaction–diffusion maze](images/lattice_textiles.jpg)

<sub>`cargo run --release --example lattice_textiles` — top row: weave, braid,
helix bundle, helicoid (`Tpms::Helicoid`, as a sheet); bottom row: spinodal,
`GaussianField`, `Hyperuniform`, `ReactionDiffusion`.</sub>

## Sheet or solid

A minimal surface bounds *two* interlocking labyrinths, and there are two ways
to make a solid out of that. `LatticeStyle` picks which (beam and infill
lattices are always the solid around their beams or walls):

| | |
|---|---|
| `Sheet` (default) | a wall of `thickness` centred on the surface, with open void either side. Two independent channel networks, no closed cells, the most surface area per gram — infill, heat exchangers, scaffolds. |
| `Solid` | one labyrinth filled, the other left as void. Stiffer than a sheet of the same mass, and it leaves a single connected void. |

For `Solid`, `thickness` is an **offset** from the minimal surface rather than a
wall width — `0.0` is the bare surface, so about half the volume, and negative
values thin it below that. Two consequences worth knowing: `wall_samples`
reports infinity, because there is no wall to resolve; and `grade` scales that
offset, so grading a solid lattice sitting at offset `0.0` does nothing at all
(zero times anything is zero). Give it a non-zero offset to scale, or grade a
sheet, which runs from nothing to solid.

```rust
let solid = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([3, 3, 3])
    .style(LatticeStyle::Solid)
    .fit_relative_density(0.25)   // solves the offset, here about −1.25 mm
    .build();
```

## Filling shapes, not just boxes

A lattice fills a box unless told otherwise. `fill` takes a `Region` — a signed
field plus the bounds it occupies — and the region's own bounds size the sample
grid, so there is nothing else to say. The mesh closes over the cut, so the
result is still one watertight shell.

```bash
cargo run --release --example lattice_shapes                       # 7 shapes
cargo run --release --example lattice_shapes --features openscad   # + a .scad model
```

```rust
use threers::{CatmullRomCurve3, Lattice, LatticeKind, Region, Tpms, Vector3};

// Primitives, swept curves, and constructive combinations of them.
let curve = CatmullRomCurve3::new(vec![/* … */]);
let shape = Region::sphere(Vector3::ZERO, 10.0)
    .union(Region::tube(&curve, 3.0, 64))
    .difference(Region::cylinder(
        Vector3::new(0.0, -20.0, 0.0),
        Vector3::new(0.0, 20.0, 0.0),
        3.0,
    ));

// …any closed triangle mesh (`mesh-bvh`)…
let shape = Region::mesh(&TorusKnotGeometry::new(10.0, 3.0, 128, 16, 2, 3)).unwrap();

// …or an OpenSCAD model, through the exact-CSG kernel (`openscad`).
let shape = Region::scad("difference(){ cube(40, center=true); sphere(24); }").unwrap();

let part = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    .fill(shape)
    .cell_size(Vector3::new(6.0, 6.0, 6.0))
    .fit_relative_density(0.25)   // 25 % of the *part*, not of its bounding box
    .skin(0.8)                    // perimeters and infill, in one mesh
    .build();
```

| Constructor | |
|-------------|--|
| `sphere`, `cuboid`, `rounded_cuboid`, `capsule`, `cylinder`, `cone`, `torus`, `half_space` | exact signed distances |
| `tube(curve, radius, segments)` | a swept `Curve3` — Catmull-Rom, Bézier, NURBS, … |
| `mesh(&BufferGeometry)`, `from_bvh` | any closed triangle mesh (`mesh-bvh`) |
| `solid(Solid)`, `scad(&str)` | an OpenSCAD model, through the exact-CSG kernel (`openscad`) |
| `new(bounds, field)` | your own field |
| `union`, `intersection`, `difference`, `smooth_union`, `offset`, `shell`, `invert` | combinators |
| `translate`, `rotate`, `scale`, `transform` | placement |

`skin(t)` adds a solid wall on the fill's surface and unions it with the
lattice, so the two bond and the part is a printed part rather than a lattice
in a bag. `clip(region)` cuts the finished part — skin included — for
sectioning it to look inside, and unlike `fill` the skin stops dead at the cut
face.

Density is measured against the **part**, not its bounding box. It has to be:
a tube swept along a curve might occupy a fifth of its own box, so 25 % of the
box would be unreachable however thick the walls were made.

Each is a scalar field, contoured by a marching cubes that builds its own case
table by chaining face contours into loops. That buys the two properties a
slicer or a boolean kernel actually needs: an ambiguous cell is resolved from
its corner signs alone, so the two cells sharing that face always agree and the
mesh cannot crack; and the loops come out oriented, so no triangle needs a
normal test to face the right way. Output is indexed, welded, and closed —
every edge shared by exactly two triangles, including where `trim` cuts the
lattice off.

Thickness is a length, not a level-set constant: the TPMS fields are divided by
their own gradient, so a 0.8 mm wall is 0.8 mm everywhere rather than varying
two-to-one between the channels and the necks. The gradients are closed-form
rather than differenced, which is one field evaluation a sample instead of
seven.

## Resolution, and the one way this fails quietly

With the `parallel` feature both halves of a build use every core: the field is
sampled a slab of the grid at a time, and the contour is drawn in slabs too and
welded back on their seams — to the same bytes as a single sweep, so a build
flag never changes a mesh.

A wall thinner than the sample step is missed *between* samples, and the mesh
comes out as disconnected specks. Nothing errors, because a field never sampled
inside a wall looks exactly like one with no wall there. It is easy to hit by
accident: at 25 % density Lidinoid needs walls a third as thick as a gyroid, so
the settings that render one cleanly shatter the other.

```rust
let lattice = Lattice::new(LatticeKind::Tpms(Tpms::Lidinoid))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([3, 3, 3])
    .fit_relative_density(0.25)   // density first — it sets the thickness
    .max_samples(12_000_000)
    .resolve_walls(2.5);          // …which sets how fine the sampling has to be
assert!(lattice.wall_samples() >= 2.5);
```

Sampling parallelises with the crate's `parallel` feature — the gallery example
builds 3.0× faster with it on (2.3 s → 0.76 s for all 37 tiles, 10 cores).
Results are byte-identical either way: every sample has a fixed address in the
grid, so a build flag cannot change the geometry.

## Foams

A beam lattice is stiff along its struts and soft between them, and however the
part is loaded some of those directions are wasted. `Stochastic` is the other
answer: a seeded random field with no special direction, which is what a real
foam, a bone and every energy-absorbing pad is.

```rust
use threers::{Lattice, LatticeKind, Stochastic, Vector3};

// Open-cell foam: struts on the edges of a Voronoi diagram.
let foam = Lattice::new(LatticeKind::Stochastic(Stochastic::Voronoi))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([5, 5, 5])
    .seed(7)          // any two seeds are two foams of the same statistics
    .jitter(1.0)      // 0 puts every seed on a grid — the ordered end of the family
    .fit_relative_density(0.2)
    .build();

// Spinodal decomposition — what a quenched alloy freezes into. Isotropic,
// or in one of three anisotropic classes (Kumar et al. 2020).
let spinodal = Lattice::new(LatticeKind::Stochastic(Stochastic::SpinodalLamellar))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .fit_relative_density(0.3)
    .build();
```

| Cell | |
|------|--|
| `Voronoi` | struts on the Voronoi edges — open cell, every void connects |
| `VoronoiWall` | the Voronoi faces as walls — closed cell, sealed bubbles |
| `VoronoiWet` | rounded bubbles parted by `thickness`-thin films, liquid gathered in Plateau borders along the edges (`plateau_border` sets how wet) |
| `Spinodal` | isotropic Gaussian random field |
| `SpinodalLamellar` | plates stacked across z |
| `SpinodalColumnar` | columns along z |
| `SpinodalCubic` | cubic symmetry |
| `GaussianField` | like `Spinodal`, but wavelengths spread ±45 % — a range of pore sizes |
| `Hyperuniform` | fibres: one elongated kernel per jittered cell, evenly spread with no order |
| `ReactionDiffusion` | a Gray–Scott Turing pattern grown on a periodic tile — a bicontinuous maze of even walls (about a second to grow, once per seed) |

The Voronoi cells measure the true distance to their walls and edges, so a
wall of `thickness` t is t thick everywhere. (The usual shortcut, half the gap
between the nearest seed distances, reads anywhere from t to three times it.)

Their seeds can also sit on a crystal lattice, which makes every cell one of a
few shapes — the ordered foams, and the ones a test rig can compare against:

```rust
use threers::{Lattice, LatticeKind, SeedLayout, Stochastic, Vector3};

// Weaire–Phelan: the least-wall equal-volume foam known.
let wp = Lattice::new(LatticeKind::Stochastic(Stochastic::VoronoiWall))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([2, 2, 2])
    .seed_layout(SeedLayout::WeairePhelan)
    .fit_relative_density(0.1)
    .build();
```

| `SeedLayout` | Seeds a cell | Cells |
|--------------|--------------|-------|
| `Random` | 1, moved by `jitter` | the random foam (the default) |
| `Kelvin` | 2 | truncated octahedra, 14 faces |
| `Fcc` | 4 | rhombic dodecahedra, 12 faces |
| `WeairePhelan` | 8 | 2 dodecahedra and 6 tetrakaidecahedra |
| `C15` | 24 | 16 cells of 12 faces and 8 of 16 |
| `Poisson` | `seeds_per_cell` (8), well spread | irregular, similar in size |
| `Relaxed` | the same, Lloyd-relaxed | irregular, nearly equal volumes |
| `Bimodal` | the same, a quarter large | two sizes (`size_ratio`, 2 by default) as a power diagram, so walls stay flat — open, closed or wet |

![Six seed layouts across, as open, closed and wet foams down](images/lattice_foams.jpg)

<sub>`cargo run --release --example lattice_foams` — columns: random, Kelvin,
Weaire–Phelan, C15, relaxed, bimodal; rows: `Voronoi`, `VoronoiWall`,
`VoronoiWet`, each fitted to one density.</sub>

The last three are one periodic tile, generated once per lattice from the seed,
so they still repeat with the cell and sample the same anywhere.

Nothing here stores a point set or an RNG: seeds are a hash of their cell index
and wave directions a hash of their own, so the field is a pure function of the
point and the seed. The same lattice samples the same at any resolution, on any
thread, in any process.

## Before it goes to the printer

`utils::mesh_check` answers the questions a slicer will ask, for any mesh:

![A Kelvin foam from below, overhangs in red, as built and simplified](images/lattice_print.jpg)

<sub>`cargo run --release --example lattice_print` — a Kelvin foam from below,
every face a 45° process must hold up in red and those within ten degrees of
it in amber; as built (102 816 triangles) and simplified (38 982), both closed,
volume within half a percent.</sub>

```rust
use threers::utils::mesh_check::{edge_report, print_report, split_pieces, triangle_quality};
use threers::{Lattice, LatticeKind, Stochastic, Vector3};

let part = Lattice::new(LatticeKind::Stochastic(Stochastic::Spinodal))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .fit_relative_density(0.25)
    .drop_specks(true) // the loose debris a random field leaves in the void
    .build();

assert!(edge_report(&part).is_closed());
let quality = triangle_quality(&part); // needles, worst-angle percentile
let pieces = split_pieces(&part); // solids, and sealed cavities (negative volume)
// Build along +z, faces flatter than 45° need support, footprint on a 512² raster.
let print = print_report(&part, Vector3::new(0.0, 0.0, 1.0), 45.0, 512);
println!(
    "{} pieces, {:.1} % needles, {:.0} mm² to support over {:.0} mm² of plate",
    pieces.len(), quality.needle_fraction * 100.0, print.support_area, print.support_footprint,
);
```

`utils::simplify` brings the triangle count down without opening the mesh: a
quadric edge collapse that refuses any collapse which would pinch the surface or
turn a triangle over, so a closed mesh stays closed. Flat regions go entirely —
a finely cut box comes back to twelve triangles — and curved ones to the
tolerance asked for:

```rust
use threers::utils::simplify::{simplify, SimplifyOptions};
# use threers::{Lattice, LatticeKind, Tpms};
# let part = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid)).build();
// Keep the tolerance well under the wall: these walls are 0.05 thick.
let (lighter, report) = simplify(&part, SimplifyOptions { target_triangles: 0, max_error: 0.002 });
```

`drop_specks` removes solid pieces under ten sample cells — but only while they
are debris: if they hold more than 2 % of the solid they are the design (an
under-resolved wall, a woven textile) and stay. Sealed cavities always stay.
`mesh_check::remove_loose_pieces` does the same for any mesh, and can keep the
largest piece with only its own cavities instead.

## Conforming, not trimming

`fill` cuts the lattice where the part ends. On a flat box that is right; on a
curved shell it leaves severed struts at the surface and whatever fraction of a
cell happened to fit. `conform` maps the point into cell space first.

```rust
use threers::{Conform, Lattice, LatticeKind, Region, Strut, Vector3};

let radius = 20.0;
let axis = Vector3::new(0.0, 0.0, 1.0);

// Sixteen cells around a nozzle, three through its wall, and the tiling
// closes on itself — `ring_pitch` is the cell size that makes it.
let nozzle = Lattice::new(LatticeKind::Strut(Strut::Cubic))
    .fill(
        Region::cylinder(Vector3::new(0.0, 0.0, -15.0), Vector3::new(0.0, 0.0, 15.0), radius)
            .difference(Region::cylinder(
                Vector3::new(0.0, 0.0, -16.0), Vector3::new(0.0, 0.0, 16.0), radius - 6.0)),
    )
    .conform(Conform::cylindrical(Vector3::ZERO, axis, radius))
    .cell_size(Vector3::new(Conform::ring_pitch(radius, 16), 5.0, 2.0))
    .thickness(0.8)
    .build();
```

| Map | |
|-----|--|
| `cylindrical(origin, axis, radius)` | x is arc length, y axial, z radial — a nozzle, a pipe, an isogrid |
| `spherical(centre, radius)` | two arc lengths and a radius — a helmet liner, a cup |
| `depth(region)` | z becomes depth below a surface — whole layers through a wall that curves |
| `new(map)` / `with_stretch(map, stretch)` | your own |

`thickness` is a length in *cell* space, and a map that is not an isometry does
not preserve lengths — so every map reports its stretch and the thickness is
divided by it. That makes the wall exact wherever the map is an isometry (at
the reference radius) and leaves the error elsewhere as the spread between the
map's principal stretches rather than their magnitude. `Conform::stretch_at`
exposes it if you want to cancel the rest with a `grade`.

## Grading on something real

`grade` takes a closure, which is not the shape a solver result, a CT scan or a
sensor sweep arrives in. `Field` is the adapter.

```rust
use threers::{Cuboct, CuboctFrame, Lattice, LatticeKind, Vector3};

// Solve a block of the lattice, then grade the lattice on what it said.
let frame = CuboctFrame::new([4, 4, 4], Cuboct::Rigid, 10.0, 0.15);
let stress = frame.stress_field(0.01, [12, 12, 12]).map(f32::abs);

let part = Lattice::new(LatticeKind::Cuboct(Cuboct::Rigid))
    .size(Vector3::new(40.0, 40.0, 40.0))
    .cells([4, 4, 4])
    .thickness(0.8)
    .grade(stress.into_grade(0.6, 1.6))   // thin where nothing is happening
    .build();
```

Build a `Field` from `grid`, `from_fn`, `scattered` (any bag of points with
values — solver output, sensor readings, a point cloud) or `from_mesh` (a value
per vertex). `map`, `normalized`, `clamped` and `smoothed` shape it;
`into_grade(at_min, at_max)` reads its range once and maps it onto two
thickness multipliers, so the units it happened to be in stop mattering.

## Measuring what came out

```rust
use threers::{Lattice, LatticeKind, Tpms, Vector3};

let m = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([4, 4, 4])
    .fit_relative_density(0.2)
    .metrics();

println!(
    "{:.0} % open, pores {:.2} mm, {:.2} mm²/mm³ of surface",
    m.porosity * 100.0, m.pore_diameter, m.surface_area_to_volume,
);
```

| Reading | |
|---------|--|
| `relative_density`, `porosity` | material and void fractions of the part |
| `open_porosity`, `closed_porosity` | void that reaches the outside, and void that does not |
| `percolates` | whether a connected path crosses the part, per axis |
| `largest_void_fraction` | the biggest connected void as a share of all of it |
| `surface_area`, `wetted_area` | every triangle; and only the internal ones |
| `specific_surface_area`, `surface_area_to_volume` | internal area per unit of part, and per unit of material |
| `pore_diameter` | largest sphere that fits in the void |
| `ligament_thickness` | largest sphere that fits in the material |
| `hydraulic_diameter` | 4 · void / wetted area |
| `permeability` | Kozeny–Carman estimate |
| `sample_spacing`, `wall_samples` | what it was measured at, and whether that was enough |

**Porous, connected and flowing are three different questions**, and they come
apart badly. The void is flood-filled and asked all three separately, because a
closed-cell foam is 80 % porous, mostly sealed and carries no flow at all; a
part with a `skin` on it is full of connected void that nothing can reach; and
a sheet TPMS is one wall between *two* separate labyrinths, which is exactly
why you would put one in a heat exchanger and would look like a defect to
anything that only counted holes.

```rust
use threers::{Lattice, LatticeKind, Stochastic, Tpms, Vector3};

let build = |kind| {
    Lattice::new(kind)
        .size(Vector3::new(24.0, 24.0, 24.0))
        .cells([4, 4, 4])
        .seed(5)
        .fit_relative_density(0.2)
        .metrics()
};

let closed = build(LatticeKind::Stochastic(Stochastic::VoronoiWall));
assert!(closed.porosity > 0.75);              // four fifths hole
assert!(closed.largest_void_fraction < 0.1);  // every bubble its own
assert!(!closed.percolates[0]);               // and nothing crosses it

let sheet = build(LatticeKind::Tpms(Tpms::Gyroid));
assert!((sheet.largest_void_fraction - 0.5).abs() < 0.1);   // two labyrinths
assert!(sheet.percolates.iter().all(|&p| p));               // both open
```

`metrics()` samples fine enough to see the wall before it measures anything —
a grid coarser than the wall reads a lattice at half its density with pores the
size of the cell, and `wall_samples` on the result says whether it managed.
Areas come off the real triangles, so a grade and a skin are in the number;
`wetted_area` drops the triangles lying in the part's own boundary, which is
the number a heat exchanger is sized on rather than `surface_area`.

## As a material, not a geometry

`homogenize` voxelises one periodic cell, solves six unit macroscopic strains
on it with periodic boundaries, and reads the effective stiffness off the
strain energy — the standard energy method, and the number to hand a solver
that is modelling the lattice as a solid.

```rust
use threers::{Lattice, LatticeKind, SolidMaterial, Strut, Vector3};

// Aluminium, 70 GPa.
let c = Lattice::new(LatticeKind::Strut(Strut::Octet))
    .size(Vector3::new(10.0, 10.0, 10.0))
    .cells([1, 1, 1])
    .fit_relative_density(0.3)
    .homogenize_with(20, SolidMaterial { modulus: 70_000.0, poisson: 0.33 });

let e = c.youngs_moduli();                             // E along x, y, z
let g = c.shear_moduli();                              // G about yz, xz, xy
let nu = c.poisson_ratios();                           // negative if auxetic
let diagonal = c.directional_modulus(Vector3::new(1.0, 1.0, 1.0));
println!("E {:.0} MPa, G {:.0}, ν {:.2}, on the diagonal {:.0}, anisotropy {:.2}",
    e[2], g[0], nu[2], diagonal, c.anisotropy());
```

`c.c` is the full 6×6 Voigt stiffness, `c.compliance()` its inverse, if what
you have to fill in is an orthotropic material card.

Conduction is the same solve with one unknown a node instead of three, so it
comes out of the same machinery — and because heat, electricity, diffusion and
permittivity are all the same equation, it is the same number for all of them:

```rust
use threers::{Lattice, LatticeKind, Tpms, Vector3};

let k = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    .size(Vector3::new(10.0, 10.0, 10.0))
    .cells([1, 1, 1])
    .fit_relative_density(0.3)
    .conductivity_with(16, 237.0);   // aluminium, W/m·K

k.axes();                  // along x, y, z
k.principal();             // the eigenvalues, which do not care about orientation
k.anisotropy();            // best direction over worst
k.tortuosity_factor(237.0);   // 0.71 — share of the material on the path
```

That last one is the number worth having. A gyroid sheet gets 0.71 of its
material onto every path; simple cubic manages 0.55, because two of its three
bar families sit across the gradient rather than along it. A lamellar spinodoid
reaches 0.92 in the plane of its plates and conducts essentially nothing across
them.

The two solves converge very differently, and it matters:

| | 12³ → 44³ | what to do |
|---|---|---|
| Stiffness | falls about a fifth, still falling | compare cells at one resolution; quote only from a grid you have watched converge |
| Conductivity | falls about 2 % | converged by 12³ for most purposes |

Elasticity is the sensitive one because a thin member's bending stiffness
depends on the exact shape of its surface and a staircase is not it. Conduction
only asks how much material lies along the path.

## When it gives way

Stiffness says how far a lattice moves; strength says how much it takes before
something stops coming back. The mechanism is the same one that makes a lattice
weaker than its density suggests — the load does not spread evenly, and the
worst-loaded ligament reaches yield long before the average one. `strength`
puts a number on that unevenness and solves the stiffness on the way, so it is
one set of solves and not two.

```rust
use threers::{Lattice, LatticeKind, SolidMaterial, Strut, Vector3};

let s = Lattice::new(LatticeKind::Strut(Strut::Octet))
    .size(Vector3::new(10.0, 10.0, 10.0))
    .cells([1, 1, 1])
    .fit_relative_density(0.3)
    .strength_with(24, SolidMaterial { modulus: 200_000.0, poisson: 0.3 });

s.concentration;            // local von Mises per unit macroscopic stress
s.uniaxial(500.0);          // 316L at 500 MPa → the lattice's yield stress
s.efficiency();             // share of the material at yield when it gives
s.stiffness.youngs_moduli() // solved on the way, not separately
;
assert!(s.resolved());      // …and whether the grid was fine enough to believe
```

At 30 % density a gyroid sheet reaches 73 MPa of a 500 MPa solid, an octet
63 MPa and a BCC cell 39 MPa — the same ordering their stiffnesses come in, and
for the same reason. `efficiency` is what separates them: 0.49, 0.43 and 0.26 of
the material at yield when the cell gives.

**This one converges from below.** A coarse grid cannot see the sharpest-loaded
corner of a ligament at all, so the concentration comes out too low and the
strength too high — the unsafe direction, and the opposite of the stiffness. An
octet at 30 % reads 6.2 at 12 voxels a cell, 8.0 at 16, 9.7 at 24, 10.8 at 32
and 11.3 at 40, still climbing; at 10 % density it is not worth reading below
about 32. `resolved()` catches the gross cases — efficiency above 1 is
impossible, and half the material has to be interior — but it is a necessary
condition, not a sufficient one. Refine until it stops moving.

Yielding only: **elastic buckling is not modelled**, and slender ligaments buckle
before they yield. `collapse_strain` is the tell — a lattice that only reaches
yield at several percent of macroscopic strain has bent over long before, and
its real collapse stress is lower than this.

A fully solid cell returns the base material exactly, which is the test that
keeps it honest. Voxel homogenisation converges *from above* — a stair-stepped
strut is over-connected — so the value falls as the resolution rises: about a
fifth between a 16³ grid and a 40³ one on an octet at 30 %, and still falling.
Compare two lattices at the same resolution; quote a number only from a grid
you have watched converge. This is the cell's own
stiffness and not a specimen's: a real block is stiffer at a bonded platen and
softer at a free surface, and at three cells across the surface is most of it.
For a block, solve the block — `CuboctFrame` does exactly that for beam cells.

## On the GPU

The linear solves run on a wgpu compute device with `.solver(Solver::Gpu)` —
the same conjugate gradient, ten times faster, and the same answer:

```rust
use threers::{Lattice, LatticeKind, Solver, Stochastic, Vector3};

let c = Lattice::new(LatticeKind::Stochastic(Stochastic::Voronoi))
    .size(Vector3::new(20.0, 20.0, 20.0))
    .cells([1, 1, 1])
    .fit_relative_density(0.3)
    .solver(Solver::Gpu)
    .homogenize(32);

c.solver;   // Solver::Gpu — or Solver::Cpu, if there was no adapter to fall to
```

| | CPU | GPU | |
|---|---|---|---|
| octet, 32³ | 3.7 s | 0.54 s | 6.9× |
| spinodal, 24³ window | 7.8 s | 0.58 s | 13.4× |
| voronoi foam, 24³ window | 9.3 s | 0.92 s | 10.1× |
| octet, 44³ | 16.7 s | 1.5 s | 11.2× |

The device solves in `f32` where the CPU solves in `f64`, and the moduli still
agree to five figures — 0.00 % apart on three of those four cases, 0.01 % on the
fourth. That is not luck: the effective tensor is read off a *strain energy*,
and energy is stationary at the solution, so an error in the displacement field
appears squared in the answer. A residual a thousand times looser than the
CPU's is still a tensor to five figures.

It is opt-in rather than automatic, because the two are not bit-identical and
this crate would rather you chose than be surprised. It falls back to the CPU
when there is no adapter, and `solver` on the result says which one ran.
Below about 16 voxels a cell the CPU wins outright — the solve is smaller than
the cost of talking to a device.

The geometry side is deliberately *not* on the GPU. Measured on a 198³ build,
sampling the field takes 55 ms for a gyroid and contouring it takes 367 ms, so
moving the sampling alone would buy about 15 %; for a beam lattice, where
sampling is the larger half, it would buy 2.5×. Neither is worth a second
implementation of every generator in WGSL that could drift from the Rust one.
