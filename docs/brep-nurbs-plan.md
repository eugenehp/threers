# threers B-rep / NURBS plan

Design doc and roadmap for an opt-in, **pure-Rust** boundary-representation layer
with NURBS surfaces, layered *on top of* the existing mesh-arrangement kernel
(`src/exact_csg/`) rather than replacing it.

Five feature flags, each independently useful and independently shippable:
`nurbs` → `brep` → `brep-csg` → `brep-kernel` → `step`.

No C/C++ ships in the crate (same commitment as
[docs/openscad-plan.md](openscad-plan.md)). FreeCAD/OpenCASCADE are used only as
an **offline test oracle** for STEP fixtures, never linked or bundled.

---

## Why

Everything below the modeling layer is triangle soup today, and analytic shape is
destroyed at the earliest possible moment.

| Symptom | Location |
|---|---|
| `sphere(r)` becomes a 32-facet mesh before the first boolean; the sphere is gone | `src/openscad/mod.rs:64` (`Solid::Leaf(BufferGeometry)`), `DEFAULT_FN` at `mod.rs:55` |
| Face identity is recovered by **hashing a quantized plane** out of triangles, from scratch, on every boolean — and only works for planes | `src/exact_csg/mod.rs:189` (`plane_key`) |
| Kernel defers to the float path on coaxial-primitive and large-coplanar degeneracies — exactly the cases analytic surface identity decides trivially | `src/exact_csg/mod.rs:20-25`, `BooleanOutcome::NeedsArrangement` at `mod.rs:97` |
| `$fn` is frozen before CSG and baked into the result forever — no LOD, no adaptive re-mesh | `src/openscad/mod.rs:99` |
| NURBS is a 126-line evaluation-only dead end: no derivatives, no knot insertion, `O(n)` span search, no path to a mesh, **zero** references from `src/wasm.rs` or `web/threejs-shim.js` | `src/curves/nurbs.rs` |
| Import is STL/OBJ/OFF/DXF/SVG — all mesh or polyline. No CAD interop | `src/openscad/scad/import.rs` |

The through-line: you don't have "no B-rep," you have a B-rep that gets lossily
hashed out of triangles at every boolean.

---

## Scope: committed vs rejected

| Meaning | Definition | Verdict |
|---|---|---|
| **(1) Surface provenance** | Every triangle knows the analytic surface it came from; normals, UVs and re-tessellation derive from that surface, not from the mesh | **Committed** (`brep`) |
| **(2) Analytic acceleration** | Surface-pair intersections with closed forms are solved in closed form; everything else falls through to today's numeric path, bit-identical | **Committed** (`brep-csg`) |
| **(3) True B-rep boolean** | `Shell/Face/Loop/Edge` topology, surface–surface intersection, trimming in UV, fillets/chamfers | **Committed, but staged and long** (`brep-kernel`) |
| **(4) STEP interop** | Read/write ISO 10303-21 AP203/AP214 advanced B-rep | **Committed** (`step`) |
| **(5) Full parametric CAD** | Constraint solver, feature tree with rollback, sketch solver, assembly mates | **Rejected.** Different product. The `Solid` expression tree plus `.scad` is the modeling front end; this plan changes what a *leaf* is, not what the tree is. |
| **(6) Kernel replacement** | B-rep as the only path; `exact_csg` retired | **Rejected.** See [Safety property](#safety-property). |

### Safety property

`src/exact_csg/mod.rs:15` states the current design philosophy: the kernel "is
never wrong, only sometimes deferential." Every stage here inherits that rule
verbatim:

> **The B-rep layer may only accelerate or refine. Any analytic path that cannot
> prove its result returns `Unknown` and falls through to the existing mesh
> arrangement. No stage may introduce a case where the answer is worse than
> today's.**

Concretely this is testable, and each stage's acceptance criteria include it:
with the analytic path forced to `Unknown` everywhere, output must be
**byte-identical** to the same build with the flag off.

---

## Flag chain

| Flag | Stage | Implies | Ships | Status |
|---|---|---|---|---|
| `nurbs` | 0 | — | Real NURBS: basis derivatives, knot insertion, adaptive tessellation, `NurbsGeometry`, wasm export | **Shipped** |
| `brep` | 1 | `nurbs` | `Surface` enum + `SurfaceTable` provenance on geometry; exact normals/UVs; re-tessellation | **Shipped** |
| `brep-csg` | 2 | `brep`, `openscad` | Closed-form surface–surface intersection fast paths inside `exact_csg` | **Shipped** |
| `brep-kernel` | 3 | `brep-csg` | `Body/Shell/Face/Loop/Edge` topology, UV trimming, B-rep boolean, fillets/chamfers | Declared |
| `step` | 4 | `brep-kernel` | ISO 10303-21 AP203/214 import + export | Declared |

"Declared" means the flag exists in `Cargo.toml` with its implications wired, and
enabling it today builds identically to leaving it off. The chain is in the
manifest so the implication graph is reviewable, not because the code is there.

```toml
# Cargo.toml [features] — appended to the existing block
#   nurbs        NURBS curves/surfaces: derivatives, knots, tessellation
#   brep         analytic surface provenance on geometry      (implies nurbs)
#   brep-csg     closed-form SSI fast paths in the CSG kernel (implies brep + openscad)
#   brep-kernel  true B-rep topology, boolean, fillets        (implies brep-csg)
#   step         STEP AP203/214 import + export               (implies brep-kernel)
nurbs = []
brep = ["nurbs"]
brep-csg = ["brep", "openscad"]
brep-kernel = ["brep-csg"]
step = ["brep-kernel"]
```

**`curves::NURBSCurve` / `NURBSSurface` stay unconditional.** They are part of the
three.js parity surface (`src/lib.rs:180`) and gating them off would be a
breaking API change. The `nurbs` flag adds a *new* `src/nurbs/` module; the old
types get a `From` impl into it and their bodies delegate when the flag is on.

**f64 throughout the new modules.** `src/curves/nurbs.rs` is f32; the exact kernel
is f64 (`V3 = [f64; 3]`, `src/exact_csg/mod.rs:53`). Everything under `src/nurbs/`
and `src/brep/` is f64 internally and narrows to f32 only at the `BufferGeometry`
boundary. Feeding f32-derived intersection points into `orient3d` would waste the
exact predicates.

### Web mirroring

Same shape as the existing flags — env var → cargo feature → `web/features.js`:

- `web/build.sh`: `NURBS=1`, `BREP=1`, `BREP_CSG=1`, `BREP_KERNEL=1`, `STEP=1`,
  appended to `FEATURE_LIST` with the same implication collapsing already used for
  `BVH_CSG`/`MESH_BVH` (`web/build.sh:24-39`).
- `web/scripts/generate-features-addon.mjs`: emit `nurbs`, `brep`, `brepCsg`,
  `brepKernel`, `step` into `features.js` plus `isNurbsEnabled()` etc., and
  stub/real addon pairs for `nurbs-addon.js` and `step-addon.js`.
- `web/scripts/generate-shim-types.mjs`: shim types for the new exports.

---

## Stage 0 — `nurbs` — **SHIPPED**

**Turns a dead 126-line evaluator into the surface primitive everything else
stands on.** Self-contained; no other stage's design is load-bearing on it
shipping first, but all of them need it.

### Shipped

```
src/nurbs/
  mod.rs          re-exports
  basis.rs        B-spline basis + derivative basis, binary-search span lookup
  curve.rs        NurbsCurve (f64, rational)
  surface.rs      NurbsSurface (f64, rational)
  knot.rs         Boehm insertion, refinement, Bézier extraction, degree elevation
  construct.rs    conics, quadrics, extrude/revolve/ruled/loft/sweep constructors
  tessellate.rs   curvature-adaptive → BufferGeometry
```

- **`find_span` by binary search** — replaces the linear scan at
  `src/curves/nurbs.rs:82-92`, and fixes the sketchy `k.saturating_sub(degree)`
  index math for `k < degree` while it's in there.
- **Derivatives** — `curve.derivatives(u, k)` and `surface.derivatives(u, v, k)`
  (Piegl & Tiller A3.2 / A3.6), with the rational quotient rule applied for
  weights. This is the piece whose absence forces finite-differenced normals today.
- **`surface.normal(u, v)`** = `normalize(Sᵤ × Sᵥ)`, with degenerate-pole
  handling — see `tests/polar_sampling.rs` and `examples/pole_probe.rs` for the
  artifacts this addresses.
- **Knot insertion / refinement / Bézier extraction / degree elevation / split**
  — prerequisites for splitting surfaces at intersection curves in Stage 3, plus
  `make_compatible` (used by `ruled`) and surface-direction insertion.
- **Exact constructors** — conic arcs as rational B-splines (weight `cos(Δθ/2)`),
  so circle/sphere/cylinder/cone/torus are *exact*, not faceted. Plus `line`,
  `plane`, `extrude`, `revolve`, `ruled`.
- **Curvature-adaptive tessellation** — chord-deviation driven subdivision to a
  tolerance, emitting position + analytic normal + surface-parameter UV. One
  tessellator, written once, instead of the per-primitive hand-rolled meshers in
  `src/geometries/`.
- **`NurbsGeometry`** in `src/geometries/` (gated), plus wasm bindings
  (`WebNurbsCurve` / `WebNurbsSurface` + `nurbs*` builders) and the
  `web/nurbs-addon.js` shim pair — NURBS is reachable from JS for the first time.

### Deviations from the plan as written

Recorded rather than quietly absorbed:

- **Pole normals use a nudge, not a second-derivative limit.** Walking a short
  way into the interior along the degenerate direction and taking the normal
  there is `O(nudge)` in angle — far below any tessellation tolerance at the
  default 1e-6 of the domain — and it cannot pick the wrong sign, which the
  L'Hôpital form can. The subtle part was the *detection*: comparing
  `|Sᵤ × Sᵥ|` against `ε·|Sᵤ|·|Sᵥ|` does not find a pole, because that product
  is itself vanishing there. Each partial has to be compared against the other.
- **`refine` is repeated single insertion**, `O(n·r)`, not A5.4's simultaneous
  `O(n + r)`. Correct by construction given `insert_knot` is; revisit when bulk
  surface refinement shows up in a profile.
- ~~**`elevate_degree` does not compact.**~~ **Now done.** `remove_knot` (A5.8)
  and `compact` remove knots that carry no shape, and elevation runs `compact`
  as its last step: interior knots come back at `original + t`. The residual
  check makes removability a property of the *geometry* — a knot that genuinely
  bends the curve survives, and the test asserts it.
- ~~**`loft` and `sweep` are not in.**~~ **Now done.** `loft` skins through N
  sections and **passes through every one**, which stacking them as control rows
  does not: a clamped B-spline is pulled toward its interior control points
  without reaching them, and a three-section loft built that way misses its
  middle by 0.625. So the control rows are solved for — chord-length parameters,
  an averaged knot vector, a dense system per control column, all in homogeneous
  coordinates so rational sections come through exactly. `loft_with_params`
  reports where each section landed, since chord-length parameters are not
  guessable. `sweep` carries a profile along a spine in a **double-reflection**
  frame (Wang et al. 2008) rather than a Frenet one, which flips through
  inflections and is undefined on a straight spine.
- **`max_samples` cannot push below the breakpoint count**, and when it binds the
  chord tolerance is not met. Rather than decimate a converged sample set (which
  would silently break a tolerance already achieved), the cap bounds recursion
  depth and `sample_grid_meets_tolerance` reports whether the result actually
  converged — the plan's "no silent caps" rule, made callable.

### Acceptance — all met

`tests/nurbs.rs` (17 cases) plus 66 unit tests in `src/nurbs/`. Run with
`cargo test --features nurbs`.

- Partition of unity: `Σ N_i,p(u) = 1` to 1e-13 over a randomized degree/knot
  sweep (200 curves × 65 parameters, seeded LCG so failures reproduce).
- Exact circle/sphere/torus: sampled radius error ≤ 1e-12 at 10k samples
  (measured ~1e-16), and a `SphereGeometry(1.0, 32, 16)` face centroid is off by
  >1e-3 on the same shape.
- Analytic derivative vs. Richardson-extrapolated central difference ≤ 1e-8,
  skipping breakpoints where `C''` genuinely jumps.
- Insertion / refinement / elevation / split all preserve randomized rational
  curves to ≤1e-9, and keep a circle circular to 1e-10.
- Tessellation: measured grid-edge chord deviation ≤ tolerance on
  sphere/torus/cylinder at 1e-2 and 1e-3, verified independently of the
  sampler's own convergence report.
- Round-trip: `curves::NURBSCurve` → `nurbs::NurbsCurve` agrees to 1e-5 (f32
  input precision), and `NURBSSurface`'s v-major grid transposes correctly —
  tested on a grid that is *not* symmetric under transposition, since a
  transposed surface otherwise looks plausible.
- Malformed input (short/non-monotonic knots, empty domain, negative **and NaN**
  weights) returns `NurbsError`, never panics.

### Verified

`cargo test --lib` 167 passed (unchanged from before the feature);
`--features nurbs` 233 passed; `--test nurbs` 17 passed. `cargo build --lib`
clean for `nurbs`, `brep`, `brep-csg`, `brep-kernel`, `step` and `--all-features`;
`--target wasm32-unknown-unknown` clean with and without the flag. Clippy adds no
new warnings beyond the repo-wide `new` -> `Self` convention every geometry
generator already trips.

### Non-goals

Trimmed surfaces (Stage 3), surface fitting/interpolation from point clouds, GPU
tessellation-shader evaluation.

---

## Stage 1 — `brep` — **SHIPPED**

**Every triangle knows what surface it came from.** No kernel changes yet; this is
pure annotation plus what falls out of it for free.

### Shipped

```rust
// src/brep/surface.rs — every variant carries a full frame, not the minimum
// data that pins the point set down: without a reference `x_dir` the parameter
// `u` is arbitrary, `invert` is not a function, and two tessellations of the
// same sphere cannot be compared.
pub enum Surface {
    Plane    { origin: V3, normal: V3, x_dir: V3 },
    Cylinder { origin: V3, axis: V3, x_dir: V3, radius: f64 },
    Sphere   { center: V3, axis: V3, x_dir: V3, radius: f64 },
    Cone     { apex: V3, axis: V3, x_dir: V3, half_angle: f64 },
    Torus    { center: V3, axis: V3, x_dir: V3, major: f64, minor: f64 },
    Nurbs(Box<NurbsSurface>),
}

impl Surface {
    fn point(&self, u: f64, v: f64) -> V3;
    fn normal(&self, u: f64, v: f64) -> Option<V3>;
    fn invert(&self, p: V3) -> Option<(f64, f64)>;        // point → uv
    fn distance(&self, p: V3) -> f64;                     // the self-check
    fn transform(&self, m: &Matrix4) -> Option<Surface>;  // None ⇒ drop the tag
    fn periodic(&self) -> (bool, bool);
}

// src/brep/table.rs — private fields, validated on construction: a table that
// indexes out of bounds would panic in the renderer rather than where the
// mistake was made.
pub struct SurfaceTable { /* surfaces, tri_face */ }
impl SurfaceTable {
    fn max_deviation(&self, g: &BufferGeometry) -> f64;  // is the tag *true*?
    fn groups(&self) -> Vec<(usize, Vec<usize>)>;
    fn transform(&self, m: &Matrix4) -> Option<SurfaceTable>;
}
```

Attached to geometry with the **same pattern already used for the BVH** at
`src/core/buffer_geometry.rs:20-21`:

```rust
#[cfg(feature = "brep")]
pub surfaces: Option<Arc<SurfaceTable>>,
```

- **Primitives populate it, by default** — `BoxGeometry` → 6 planes;
  `SphereGeometry` → 1 sphere; `CylinderGeometry` → cylinder-or-cone + up to 2 cap
  planes; `ConeGeometry`, `TorusGeometry`, `PlaneGeometry`, `CircleGeometry`,
  `RingGeometry`. One line per generator, calling `brep::primitives`, which is
  also where the axis conventions live (`SphereGeometry`/`CylinderGeometry` are
  Y-up, `TorusGeometry` is Z-up) so they cannot drift apart.
- **A truncated cone's apex is extrapolated** from its two radii. It is not a
  vertex, not on any face, and not inside the bounding box — the clearest case of
  something no amount of triangle inspection could recover.
- **Transform propagation** — rigid motion and uniform scale keep the analytic
  type; anything else drops the whole table. Planes and NURBS survive any
  invertible affine map (inverse-transpose normal; control points are exact in
  homogeneous coordinates).
- **`retessellate(&geometry, tolerance)`** — re-mesh from provenance, returning a
  `RetessellationReport`.
- **`exact_normals()` / `exact_uvs()`** — from the surface, not from averaged face
  normals or the mesher's guess.

### Deviations from the plan as written

- **Non-uniform scale drops quadrics rather than demoting them to NURBS.** The
  plan said demote — an ellipsoid *is* an exact rational patch — but an unbounded
  `Cylinder` or `Plane` has no natural bounded NURBS to demote *to*, and a
  half-populated table (some triangles tagged, some not) would force every
  consumer to carry that distinction when "no provenance" is already a supported,
  safe state. `transform` is therefore all-or-nothing.
- **`retessellate` trims planar patches but does not yet share vertices across
  every boundary.** Partly fixed, and the fix uncovered something worse than the
  T-junctions originally recorded.

  **A patch's parameter footprint is not in general a rectangle.** Meshing its
  bounding box instead is a wrong answer, not a coarse one: a cylinder's radius-2
  disk cap came back as a **square**, reaching 2.83 — the corner of the box its
  parameters span, on the right plane but nowhere near the solid. Planar patches
  are now filled against their actual boundary (a plane is exact under any
  triangulation, so this is not an approximation), and a box, a cone and a capped
  cylinder all come back the right shape.

  Boundaries themselves are recovered by [`brep::stitch`]: an edge whose two
  triangles carry *different* surfaces is a face boundary, which is a question a
  triangle soup cannot answer and provenance answers for free. Curved patches now
  also sample the shared boundary's parameters, so both faces place vertices at
  the same angles.

  Closure is **not** achievable on this path, and is no longer claimed on it.
  Welding by *position* cannot make two faces agree on which points to place
  along a shared rim: the side wants ~124 samples around a rim the cap has 24 of,
  and densifying the boundary first made things worse (a cone went from closed to
  118 open edges) because the two faces then reach nearby-but-unequal points by
  different arithmetic. That is what [`brep::Body`] is for — see Stage 3b below.
- **`retessellate` samples each direction to half the requested tolerance.** A
  quad's diagonal spans both parameter directions, so its chord deviation is
  bounded by the *sum* of the two. Sampling each to `tolerance` would leave every
  diagonal at up to `2·tolerance`.
- **UV seams are unwrapped over mesh connectivity, not per triangle.** `atan2` has
  a branch cut; choosing each vertex's branch independently cannot work on a
  shared-vertex mesh, because a vertex written by two triangles gets whichever
  branch was written last and the texture tears in a triangle-order-dependent
  place. The branch propagates breadth-first from a seed instead. Where a seam
  vertex is genuinely *shared* rather than duplicated, no assignment can be right;
  `UvReport::seam_wrapped` counts those.
- ~~**`LatheGeometry` / `TubeGeometry` / `ExtrudeGeometry` are not tagged.**~~
  **Now done.** A lathe is an exact NURBS revolution of its profile. A tube is
  **lofted through the generator's own circles**, not swept — a sweep would build
  its own rotation-minimising frame, and a surface on different frames puts its
  vertices somewhere else, so the tag would be false. An extrusion is all
  *planes* (two caps, one per contour edge), which makes it the one swept
  primitive whose provenance the Stage 2 analytic path can use.
- ~~**UV seam splitting reduces rather than eliminates.**~~ **Now complete.**
  `exact_uvs_split_seams` cuts every seam in one pass and leaves nothing behind.

  The fix was a change of *representation*, not of algorithm. Parameters are now
  assigned per **(triangle, corner)** rather than per vertex — made globally
  consistent by unwrapping each triangle and propagating whole-turn offsets
  across shared edges. A per-vertex assignment cannot even express the problem
  ("this vertex must hold two values"); a per-corner one makes the seam a list
  rather than a search: it is exactly the corners whose value disagrees, by whole
  turns, with the one their vertex ended up holding.

  The second half matters as much: parameters are computed **once, on the
  original mesh**, and reused after the split. Recomputing afterwards is what
  defeated two earlier attempts — splitting changes the connectivity the offsets
  propagate along, so the assignment shifts and conflicts reappear elsewhere. One
  version converged to four wrapped triangles instead of zero; the next went
  19 → 19. Computing first makes the split a pure re-indexing.

  Verified on the case it exists for: *weld* a sphere/cylinder/torus's duplicated
  seam column (which is what a CSG result or an STL import looks like), confirm
  the texture tears, split, and confirm no triangle spans more than half the
  texture in a periodic direction. The built-in generators duplicate their seams
  already, so the split correctly does nothing to them.

  Two bugs fell out of getting there. A sphere's **poles** have an arbitrary `u`,
  and seeding the branch there flagged every triangle around them —
  `Surface::parameter_degenerate_at` now identifies those and gives them the mean
  of their triangle's real corners. And a test helper was tagging
  `SphereGeometry` with a Z-up sphere where the generator is Y-up: the same point
  set under a rotated parameterization, so the mesh's rows stopped aligning with
  it and triangles spanned arbitrary ranges — which reads as a torn texture but
  is really a mismatched tag.

### Acceptance — all met

`tests/brep.rs` (19 cases) plus 49 unit tests in `src/brep/`. Run with
`cargo test --features brep,openscad`.

- **Every primitive's tag is true**: `max_deviation` over all twelve built-ins is
  below `1e-5 ×` model scale, which is the f32 vertex-buffer floor. This is the
  central claim — a tag is a *claim* about where triangles are, and this is what
  makes it falsifiable.
- Analytic normals are exact to 1e-9 on every curved primitive, while
  `compute_vertex_normals` error on the same mesh grows monotonically as the mesh
  coarsens (asserted, so the annotation is proven to be used).
- `plane_key` (`src/exact_csg/mod.rs:191`, now `pub(crate)`) partitions a box's
  triangles **identically** to `tri_face`. On a sphere it produces 50+ "faces"
  where provenance sees one surface — asserted as a gap, since closing it is what
  Stage 2 is for.
- A `16×8` sphere re-tessellates to chord error ≤ tolerance at 1e-2, 1e-3 and
  1e-4, closed, with provenance carried forward so it can be done again.
- Transform round-trip: four transforms × twelve primitives, deviation ≤
  `1e-4 ×` scale after mapping table and mesh independently.
- **Safety**: tagging alters no buffer (the pre-existing three.js parity tests
  pass unchanged with `brep` on — 167 default vs 280 with the feature, all
  originals green); every consumer declines without provenance; writing positions
  invalidates the table; a mismatched table is refused rather than stored.

### Non-goals

Trim loops (a face here is still "the set of triangles tagged with this surface"),
topology, edges, any kernel behavior change.

---

## Stage 2 — `brep-csg` — **SHIPPED** (one case still defers)

**Closed-form intersection where closed form exists; today's numeric path
everywhere else.**

The kernel's first named degeneracy — *"two identical primitives translated
along an axis"* — now returns `Exact` for all three ops. Crossed cylinders at one
particular radius still defer, for an unrelated reason; see
[What still defers](#what-still-defers).

**The fix was not where the plan expected it.** The plan assumed the win would
come from feeding better intersection segments into `corefine`. It did not:
`corefine` was already fine. The failure was in the *classifier*, and the fix
was to give it surface identity as a fallback for an exact-arithmetic test that
refined vertices cannot pass. See [Where it actually paid](#where-it-actually-paid).

### Shipped

```rust
// src/brep/intersect.rs
pub enum SsiResult {
    Disjoint,                        // proven no intersection
    Coincident { opposite: bool },   // proven same surface; sense recorded
    Curves(Vec<Curve3d>),            // exact intersection curves
    Unknown,                         // ⇒ fall through to numeric tri-tri
}
pub fn ssi(a: &Surface, b: &Surface) -> SsiResult;

// src/brep/curve3d.rs — Line | Circle | Ellipse | Point, each with a
// closed-form `project` (the ellipse is a scan-seeded, guarded Newton).
```

Closed forms implemented (everything else returns `Unknown`):

| Pair | Result |
|---|---|
| plane / plane | line, `Coincident`, or `Disjoint` (parallel) |
| plane / sphere | circle, tangent point, or `Disjoint` |
| plane / cylinder | line pair ∥ axis, circle ⊥ axis, or ellipse |
| plane / cone | conic section (ellipse / parabola / hyperbola / line pair / point) |
| plane / torus | circle (⊥ axis), Villarceau pair, or quartic → `Unknown` |
| sphere / sphere | circle, tangent point, `Coincident`, or `Disjoint` |
| sphere / cylinder | coaxial → circle pair; otherwise `Unknown` |
| cylinder / cylinder | coaxial → `Coincident`/`Disjoint`; parallel → line pair; equal-radius crossing → ellipse pair |
| cone / cone | shared apex → line pair; coaxial → circle |
| any / Nurbs | `Unknown` (Stage 3 handles it by marching) |

Wired into `corefine` (`src/exact_csg/analytic.rs`) with a per-boolean cache
keyed by *surface* pair — a cylinder is one surface and hundreds of triangles, so
without it the same `ssi` would run once per candidate pair. Two of the three
hooks are live:

- **`Disjoint` → skip the pair.** A proven negative cannot be wrong.
- **`Coincident` → same surface, so no transversal crossing.** Only the coplanar
  overlap is taken; any segment `tri_tri_segment` reports for such a pair is
  noise off a near-coincidence.
- **`Curves` → reported but not acted on.** See below.

**Also required, and easy to miss:** `openscad::bake_matrix` had to carry the
provenance through a transform (`SurfaceTable::transform`, which Stage 1 built
for exactly this). Without it every transformed operand reaches the kernel
untagged — and *every* interesting boolean has a transform on one side, so the
accelerator would have been dead code that still passed all its own tests.

### Where it actually paid

`coincident_face` — the classifier that decides whether a sub-triangle lies on a
face the other solid also has — gates on `coplanar()`, which is
`orient3d(...) == 0.0` **exactly**. That is right for original mesh triangles,
whose vertices are shared verbatim, and wrong for *refined* ones, whose vertices
come out of the CDT recomputed and land a few ULPs off the plane they belong to.

The test then fails, the sub-triangle falls through to a ray-parity
classification whose ray starts *on* the other solid's boundary, and the answer
is a coin flip. On `cylinder(4, 2) ∪ the same translated 2 along its axis` the
flip landed both-keep on some facets and neither-keep on others: duplicate
triangles in one place, holes in another, 34 malformed edges, every one on the
two rim circles.

Surface identity has no such fragility — two triangles are on the same surface
when their surfaces are the same surface, decided by comparing axes and radii.
Supplying `ssi`-based identity as a fallback for the coplanarity gate is the
entire fix, and it lifts detections from 102/318 to 128/318 on that model.

Recovering a refined sub-triangle's surface needs care. Measuring its distance to
each candidate surface does **not** work: a tessellated cylinder's facet is a
chord, and its interior points sit a full sagitta off the analytic cylinder
(0.0096 for `$fn = 32` at radius 2 — four orders of magnitude above any
numerical tolerance). Recovering it by **plane key** does work, and reuses the
grouping `refine_with` already computes.

### Seam exactness, done the right way

`Surface::signed_distance` and `Surface::edge_crossing` now let a seam endpoint
be made exact by sliding it **along the mesh edge it already lies on** until it
sits on the other surface — a bracketed root find, so the point cannot leave its
edge. Measured on two overlapping spheres: 48 endpoints corrected, by up to
2e-6, with the corpus unchanged.

The plan asked for the seam to land on the exact circle "to 1e-12". **That is
unachievable through a `BufferGeometry` regardless of the kernel**, because
positions are stored as f32: at radius 2 the quantum is ~1.2e-7, and the measured
seam error is 1.85e-7 with the correction on *and* off. The internal f64 seam is
now exact; the output format cannot carry it. Any future criterion of this kind
has to be stated against the f64 pipeline, not the vertex buffer.

### Two things that did not work

Both were implemented, measured against the corpus, and removed. The reasons
generalise, so they are recorded rather than quietly dropped.

**Projecting seam points onto the exact intersection curve.** A
`tri_tri_segment` endpoint lies on a triangle edge, and the refinement downstream
depends on that incidence; the exact curve is where the *surfaces* meet, not
where the triangles do. Two overlapping spheres went from `Exact` to
`NeedsArrangement`. Replaced by the edge root find above.

**Synthesising cuts for boundary contacts.** A capped cylinder's rim lies on the
cylinder, so a cap stacked flush against a side touches without crossing and
`tri_tri_segment` finds nothing — which looked like the cause of the coaxial
failure. Cutting the side facet with the cap's plane was neither necessary (the
coplanar-overlap path already makes those cuts) nor harmless (it broke
`cyl ∪ cyl parallel`, whose caps are coplanar, by adding cuts that path had
deliberately not made).

### What still defers

Two cylinders crossing at 90° at radius 1.5 remain `NeedsArrangement`. This is
**not** the coincident-face problem: their surfaces genuinely cross, the
accelerator correctly reports two exact ellipses, and the failure is sliver
triangles out of the CDT on a genuinely curved∧curved seam.

It is geometry-specific rather than categorical — the corpus's own
`cyl⊥cyl cross` case (`cylinder(4, 0.8)`) resolves. Closing it is work on the
CDT's sliver handling, not on the analytic layer. `tests/brep_csg.rs` asserts the
current state so that closing it breaks the test.

### Acceptance

`tests/brep_csg.rs` (13 cases) plus 29 unit tests in `src/brep/intersect.rs` and
`src/brep/curve3d.rs`.

| Criterion | Status |
|---|---|
| Reported curves lie on both surfaces (sampled, 1e-9) | **Met** |
| `Disjoint` only when provable (verified by dense sampling) | **Met** |
| Accelerator demonstrably fires on real booleans | **Met** — `analytic_report` |
| No regression: full corpus green with the flag on | **Met** — 535 pass vs 506 with it off, 0 failures |
| A tagged boolean is never worse than an untagged one | **Met** — and this caught the `Curves` bug |
| Tagged results still pass the closed-manifold gate | **Met** |
| Coaxial cylinders return `Exact` | **Met** — all three ops; defers again when provenance is stripped |
| Crossed cylinders return `Exact` | **Not met** — CDT slivers, not the analytic layer |
| NURBS point inversion is reliable | **Fixed while doing this** — the seed grid was fixed at 12×12 and the Newton unguarded, so on a lofted tube it reported a distance of **2.2** where the truth was 4.5e-3. Now sized to the control net, with a backtracking line search. |
| Sphere ∩ sphere seam on the exact circle to 1e-12 | **Unachievable as stated** — f32 vertex buffer floors it at ~1.85e-7. The f64 seam is exact. |

The safety property is asserted in a stronger, testable form than the plan's
original wording. "`ssi` forced to `Unknown` ⇒ byte-identical" cannot be checked
inside one build; **"a tagged boolean is never worse than the same boolean with
its provenance stripped"** can, and it is what caught the seam-projection
regression. `analytic_report` exists alongside it because a never-regresses
property is trivially satisfied by an accelerator that never fires.

### Measured

`Disjoint` pays in a narrower band than it looks. The AABB broad phase is cheaper
and already eliminates anything far apart — two *concentric* spheres a whole unit
apart produce **zero** candidate pairs, so the closed form is never consulted.
What it catches is near-miss geometry where the boxes overlap but the surfaces
cannot meet; a thin-walled tube (r = 2.0 against r = 1.9) yields 332 skips and
2048 same-surface resolutions out of 3624 candidates.

### Non-goals

Marching NURBS/NURBS intersections, tolerant modeling, topology. Still a *mesh*
boolean — just one that stops guessing where the seam is when it can know.

---

## Stage 3 — `brep-kernel`

**The real thing.** Multi-month; explicitly sub-staged so each piece lands
standalone. Do not start this before 0–2 are shipped and the corpus is green.

### Prerequisite refactor

`arrange_extract` (`src/openscad/scad.rs:2577`) is a complete 2D arrangement
kernel — splits every edge at crossings, classifies sub-edges by ε-offset point
sampling, traces boundary loops, nests holes. It is exactly the UV-space trimming
engine, and it is currently a private fn buried in a 5300-line file. **Lift it into
`src/geom2d/arrangement.rs`** as a public, generic-over-classifier module, with
`scad.rs` calling into it. No behavior change; the 2D corpus models (`booleans_2d`,
`rounded_2d`, `proj_cut`, `mink2d`) are the regression gate.

### 3a — topology + trimmed tessellation — **SHIPPED**

`src/brep/body.rs` — a `Body` of surfaces, shared `Edge`s and `Face`s, built from
a tagged mesh by `Body::from_tagged_mesh`. Faces share vertex **indices**, not
positions, which is the whole point: each edge is tessellated once and both
adjacent faces reference the result, so the mesh is closed by construction and no
tolerance is involved. `refine_edges` sharpens an edge on that shared list.

A capped cylinder, a cone and a box all tessellate **closed** at any tolerance —
including the capped cylinder the mesh-side path could not close.

Two things the edges alone could not express, and the face's recorded parameter
footprint can: a **degenerate end** (a cone's apex is not a boundary shared with
anything, so it collapses to one vertex and the quads become a fan) and a
**closed sweep** (a full revolution's grid has to join its last column back to
its first, or it is open along the seam).

Getting there turned up an inconsistency in `Surface::invert`: it returned `None`
on a cylinder's axis and at a cone's apex, where in fact only `u` is undetermined
and `v` is perfectly well defined. A cone's apex vertices therefore vanished from
its face's footprint, collapsing it to a single `v`. The sphere already did the
right thing — report a usable parameter and let `parameter_degenerate_at` flag
it — and the other two now match.

Boundary-*less* faces work too: a sphere and a torus are one face with no edges,
so there is nothing to share and the face samples its own footprint — with the
seam wrapping in `u` (a sphere) or both (a torus), and poles collapsing to a
single vertex. They used to produce an empty mesh, because the fill required a
rim to take its angular samples from.

| Primitive | Body tessellation |
|---|---|
| box, sphere, torus, cone, capped cylinder | **closed** |
| open cylinder | open, correctly — a tube has two boundaries |
| **truncated cone** | **open** — see below |

**The truncated cone does not close, and the reason is structural.** Refinement is
driven by curvature, so its radius-1 rim converges at 81 points where its
radius-3 rim needs 161. Both are right on their own, and the face between them
cannot be gridded: a tensor grid needs its first and last rows the same width, so
the wider rim ends up disconnected from its cap (240 open edges). Imposing the
union of the two `u` sets on both rims was tried and made it worse — 332 open
edges, and it broke the capped cylinder. The real fix is to stop requiring a
tensor grid at all: walk a strip between two polylines of different lengths,
advancing whichever side is behind in parameter. That is a change to *how a swept
face is filled*, not a patch to its inputs, and it is the right next piece.

**Everything above is now done.** In order:

* **The strip walk.** A swept face's two rows need not have the same width —
  a truncated cone's radius-1 rim converges at 81 points where its radius-3 rim
  needs 161 — so the rows are joined by walking them together and advancing
  whichever is behind in parameter, rather than by a tensor grid. Every
  primitive now tessellates closed, the truncated cone included.
* **Shells, validity and Euler.** `Body::shells` groups connected faces,
  `Body::defects` reports dangling references, unshared edges, unclosed loops
  and — the geometric one — edges that do not lie on the surfaces they claim to
  join. `Shell::euler` carries `euler_meaningful` alongside it, because a
  periodic face has an unrepresented seam and the alternating sum is then not a
  genus claim: a box gives 2, a capped cylinder gives 3, and saying so is more
  useful than quietly reporting 3.
* **Closure is not an edge count.** An open tube is one face with *no edges at
  all* — its rims border nothing — so a check that only counted edge use would
  call it closed. `face_free_ends` asks the footprint instead: an end is closed
  by a shared edge, by a degeneracy (a pole, an apex), or by wrapping, and
  anything else is free.
* **Authoring.** `Body::cuboid` / `sphere` / `torus` / `cylinder` / `cone` /
  `frustum` / `plate_with_hole` build the topology first and let the mesh come
  out of it. Until this, a `Body` could only be *derived* from a tagged mesh and
  so could never represent anything the mesh path had not already produced.
* **Trim loops.** `Face` has real loops in parameter space, derived from its
  edges and oriented so the outer one is positive and holes negative.
  `plate_with_hole` is the smallest solid that needs them — its top is one plane
  bounded by an outer rectangle *and* an inner circle, which no parameter
  rectangle describes.

Two things had to be written to make holes work. Hole **bridging**, because the
crate's `earcut` documents that it ignores its `hole_indices` — so a holed face
came back filled solid. And a **new ear clip**, because the polygon bridging
produces is pinched: it has coincident vertices by construction, and an
inclusive containment test vetoes every ear, makes no progress, and returns
nothing. Strict containment plus an explicit same-position check fixes it.

### 3a — as originally planned

```rust
// src/brep/topology.rs
pub struct Body   { shells: Vec<Shell> }
pub struct Shell  { faces: Vec<Face>, closed: bool }
pub struct Face   { surface: SurfaceId, loops: Vec<Loop>, sense: bool, tol: f64 }
pub struct Loop   { half_edges: Vec<HalfEdgeId>, kind: LoopKind }  // Outer | Hole
pub struct Edge   { curve: Curve3d, pcurve: [PCurve; 2], v0: VertexId, v1: VertexId, tol: f64 }
pub struct Vertex { point: [f64;3], tol: f64 }
```

Half-edge adjacency, extending `src/csg/half_edge_map.rs:9`. Trimmed-face
tessellation: sample the surface on a curvature-adaptive UV grid, constrain to the
trim loops, run `src/exact_csg/cdt.rs`. `Body → BufferGeometry` at any tolerance.
Validity checker: every edge has exactly two half-edges, loops close, senses agree,
shells are closed and orientable.

### 3b — B-rep boolean

SSI (Stage 2 closed forms, plus a marching-with-Newton-refinement path for the
NURBS cases that returned `Unknown`) → intersection curves → pcurves in each
face's UV → per-face 2D arrangement via `geom2d::arrangement` → classify subfaces
in/out of the other body → assemble result shells. Tolerant modeling throughout:
per-entity tolerance, healing pass after assembly.

Falls back to the Stage 2 mesh path whenever SSI marching fails to converge or
validity checking rejects the result — same deferential contract.

### 3c — fillets and chamfers

The capability that is *impossible* on triangle soup and that OpenSCAD itself
lacks. Rolling-ball edge blend: the new face is a pipe surface (NURBS) swept along
the edge; adjacent faces get re-trimmed against it. Constant radius first;
variable radius and vertex blends (three-face corner) after. Chamfer is the same
machinery with a ruled surface.

```rust
body.fillet(&[edge_id], radius)?;
body.chamfer(&[edge_id], distance)?;
```

### Acceptance

- 3a: every `.scad` corpus model round-trips mesh → B-rep → mesh with volume and
  Euler characteristic preserved; validity checker clean.
- 3b: corpus booleans through the B-rep path match the mesh kernel's volume to
  1e-9 and Hausdorff to 1e-6; watertight and 2-manifold on all.
- 3c: fillet radius `r` on all 12 cube edges — watertight, and volume matches the
  analytic `V = s³ − 12·(1 − π/4)·r²·s + (corner terms)` to 1e-6.
- **Safety**: `brep-kernel` off ⇒ Stage 2 behavior unchanged, byte-identical.

### Risk

Surface–surface intersection is where amateur B-rep kernels die — specifically
marching an intersection curve through tangential and near-tangential
configurations. Mitigation is structural: SSI failure is a *supported outcome*
that falls back to the mesh kernel, not an error. If 3b stalls, 3a and 3c are
still shippable (3c needs 3a plus local booleans, not the general one).

`truck` (pure Rust, Apache-2, wasm-friendly) is worth reading for its SSI approach
— it doesn't violate the no-C/C++ commitment even as a dependency, though the
intent here is to build natively.

---

## Stage 4 — `step`

**CAD interop.** ISO 10303-21 part-21 physical file, AP203/AP214 advanced-B-rep
subset.

> **Naming trap:** `src/csg/step_tests.rs` is about evaluation *steps* in the CSG
> hierarchy — entirely unrelated. Put this in `src/brep/step/` and `src/loaders/step.rs`.

### Ships

- Part-21 tokenizer + entity-graph parser (forward references, `#N` resolution).
- Entity subset — read and write:
  `ADVANCED_BREP_SHAPE_REPRESENTATION`, `MANIFOLD_SOLID_BREP`, `CLOSED_SHELL`,
  `ADVANCED_FACE`, `FACE_OUTER_BOUND`/`FACE_BOUND`, `EDGE_LOOP`, `ORIENTED_EDGE`,
  `EDGE_CURVE`, `VERTEX_POINT`, `CARTESIAN_POINT`, `DIRECTION`, `AXIS2_PLACEMENT_3D`;
  surfaces `PLANE`, `CYLINDRICAL_SURFACE`, `SPHERICAL_SURFACE`, `CONICAL_SURFACE`,
  `TOROIDAL_SURFACE`, `B_SPLINE_SURFACE_WITH_KNOTS`, `RATIONAL_B_SPLINE_SURFACE`;
  curves `LINE`, `CIRCLE`, `ELLIPSE`, `B_SPLINE_CURVE_WITH_KNOTS`.
- Units and tolerance headers; assembly structure flattened on import (v1).
- `import("model.step")` in the `.scad` front end, alongside the existing
  STL/OBJ/OFF/DXF/SVG handlers in `src/openscad/scad/import.rs`.
- `Solid::to_step()` next to `to_stl`/`to_obj`/`to_off`/`to_3mf`/`to_glb`
  (`src/openscad/mod.rs:445-462`).

### Acceptance

- Import a FreeCAD/Onshape `.step` → tessellate → Hausdorff ≤ 1e-5 vs. the same
  tool's STL export of the same model.
- Export → reimport in FreeCAD → volume and face count match.
- Round-trip through our own reader/writer is lossless on the corpus.
- Fixture set covers: analytic-surface-only, NURBS-surface, trimmed-face,
  multi-shell-with-voids, and an assembly.

---

## Sequencing

Stages 0–2 are incremental, each independently useful, and none destabilize the
mesh kernel — they can land in consecutive releases. Stage 3 is a different
magnitude and should not begin until 0–2 are green across the corpus. Stage 4 is
mostly parsing and mapping, gated on 3a's topology existing (not on 3b).

| | Stage | Rough size |
|---|---|---|
| 1 | `nurbs` | days |
| 2 | `brep` | 1–2 weeks |
| 3 | `brep-csg` | 2–3 weeks |
| 4 | `brep-kernel` 3a | 3–4 weeks |
| 5 | `step` | 2–3 weeks (needs 3a only) |
| 6 | `brep-kernel` 3b | months |
| 7 | `brep-kernel` 3c | weeks after 3b |

`step` deliberately jumps ahead of the general B-rep boolean: reading CAD data and
rendering it needs topology and tessellation (3a), not booleans.

---

## Also update

- `PLAN.md` — add the five flags to the "Ecosystem extensions" table with status.
- `Cargo.toml` — `[features]` block plus the doc comment header at `Cargo.toml:82-97`.
- `web/build.sh`, `web/scripts/generate-features-addon.mjs`,
  `web/scripts/generate-shim-types.mjs` — flag mirroring per
  [Web mirroring](#web-mirroring).
- `README.md` — feature table.


## Stage 4 — `brep-kernel`: the B-rep boolean — **SHIPPED**

`Body::boolean(&self, other, op, tolerance) -> Result<Body, Declined>`, over
Union / Difference / Intersection.

### Ships

- Closed-form SSI between every face pair, reusing the `brep-csg` intersector.
- Curves sampled **once** into a shared vertex list and referenced by index from
  both faces they bound. This is the whole design: two faces that each evaluate
  the same seam agree on where it is and still crack, because a crack is a
  disagreement about vertex *identity*, not position.
- Face trimming: `split_planar` turns a closed curve into a hole (and the region
  it encloses into an island); `split_swept` slices a swept face's v-range at a
  constant-v curve.
- Inside/outside classification by ray parity against the opposing body, with
  Difference flipping the walls it keeps from B.
- Untouched faces carried through with their original edges, keyed by
  (body, edge index) so both users resolve to one result edge.
- Carried-through loop corners welded by position into the same vertex list.

### Measured

`Body::cuboid([10,8,2]) ⊘ Body::cylinder([0,0,-3],[0,0,1],2,6)` at tol 1e-3,
volume by the divergence theorem, against the closed form:

| | computed | exact | error | open edges |
|---|---|---|---|---|
| A − B | 134.884 | 134.867 | 0.01% | 0 |
| A ∪ B | 210.203 | 210.265 | 0.03% | 0 |
| A ∩ B | 25.116 | 25.133 | 0.07% | 0 |
| sphere(3) − bore(1) | 94.618 | 94.782 | 0.17% | 0 |

The residual is tessellation chord error at the requested tolerance, not kernel
error — the B-rep itself is exact (a bore comes back as a `Cylinder` with
`v ∈ (2,4)`, and a holed face as a plane with a −4π hole loop).

The last row is the "napkin ring": a sphere less a coaxial bore, whose exact
volume is 4/3·π·(R²−a²)^{3/2}. It exercises a curved face split by a curved one,
where the seam is not a plane cut.

### Deviations

- **Declines rather than approximates**, as everywhere else in this stack:
  `NeedsArrangement` when a curve crosses a face boundary (a general 2D
  arrangement in parameter space is not implemented), `NoClosedForm` for a
  surface pair without one, `CoincidentFaces` for shared surfaces, `NotASolid`
  for an open input.
- Non-closed intersection curves (lines, and any open curve) are not handled;
  they reach a face as a subdivision rather than a hole and decline.
- `split_swept` only cuts at a **constant-v** curve. A curve that spirals across
  a swept face declines.
- Fillets and chamfers are **not** implemented, and are not planned here.

### Found on the way

`Body::tessellate` was emitting inconsistently-wound triangles and nothing
caught it: a mesh whose faces disagree about which way is out still uses every
edge exactly twice, so `is_closed()` was satisfied. It surfaced only as a
volume — a radius-2, height-6 cylinder measured 25.1 where 24π = 75.4. Fixed by
orienting in two steps: propagate consistency through triangle adjacency (a
property of the mesh, needing nothing from the B-rep), then pick each
component's sign by majority vote against the analytic surface normals, so one
sliver's unreliable normal cannot invert a shell.


## Stage 5 — `step`: ISO 10303 exchange — **SHIPPED**

Two layers, kept apart because they fail for different reasons.

`step::part21` is the exchange *syntax* — a header and a list of
`#id = NAME(args);` instances, with no opinion about what they mean. Nearly
every real-world STEP problem is lexical (a real written without its decimal
point, a quote inside a name, a complex instance), and all of those are testable
here against a round trip without constructing any geometry.

`step::export` / `step::import` are the AP203 *mapping*.

### Ships

- `export(&Body, name, tolerance) -> (String, ExportReport)` and
  `import(&str, tolerance) -> Result<(Body, ImportReport), ImportError>`.
- Plane, cylinder, sphere, cone and torus map to their own entities in both
  directions. NURBS surfaces map to `B_SPLINE_SURFACE_WITH_KNOTS`, rational or
  not, written as the complex instance AP203 requires.
- Edge geometry is *recovered from the edge's own vertices*: points on a common
  line become a `LINE`, points on a common circle become a `CIRCLE`, anything
  else a `POLYLINE`. So an edge exports as whatever it actually is, whether it
  came from a primitive, a boolean seam, or a file — and a bore's rims come back
  as two circles rather than four hundred segments.
- Closed surfaces are cut at their parametric seam on the way out (two faces for
  a sphere, four for a torus) because `ADVANCED_FACE` is a *bounded* region and a
  whole sphere has no boundary to give it. The pieces share the cut edges.
- The AP203 product/context boilerplate, so the file opens in a CAD system
  rather than merely parsing.
- Export is deterministic — no clock, no hash iteration order — so the same
  solid always produces the same bytes.

### Measured

Round trip, volume by the divergence theorem, all watertight:

| | out and back | exact |
|---|---|---|
| cuboid 2×3×4 | 24.000 | 24 |
| cylinder r2 h5 | 62.71 | 62.83 |
| sphere r2 | 33.44 | 33.51 |
| plate − bore | 134.6 | 134.87 |

An imported solid is still a solid: `import` → `boolean` → correct volume is a
test, because that is the difference between reading a file and reading a
*model*.

### Deviations

- Reports rather than approximates, as everywhere else: `Unsupported::Surface`
  for a surface with no AP203 counterpart, `UnboundedFace` for a closed NURBS
  face with no synthesisable seam, and `ImportReport::skipped` naming every
  entity the reader met and could not map.
- Only the first shell in a file is read; assemblies
  (`NEXT_ASSEMBLY_USAGE_OCCURRENCE`) are not.
- No `SURFACE_OF_REVOLUTION` / `SURFACE_OF_LINEAR_EXTRUSION` on import — they
  are reported, not silently tessellated.
- Units are written as metres and not converted on read.

### Found on the way

Three defects in `Body::tessellate`, each invisible until a round trip made a
face arrive by a route authoring never used:

1. A rim sitting **on the branch cut** — a seam-split sphere's meridian at
   u = ±π — measured a full turn of spread instead of none, because `invert`
   returns whichever sign the arithmetic lands on. Periodic spread now means
   the smallest arc containing the samples.
2. A **pole** contributed an arbitrary parameter to that measurement (`invert`
   is degenerate there), widening a meridian's spread from nothing to a half
   turn. Degenerate points are now excluded from the measurement, though not
   from the rim.
3. A face bounded only by **meridians** — one half of a seam-split sphere —
   could not be filled at all, since the walk assumed rims at constant `v`.
   It now runs transposed as a second attempt, and collapses degenerate column
   ends the way it already collapsed degenerate rows.


## After the stages: measuring what the boolean actually covers

All five flags shipped, so the next question is not what to build but what
already works. A twelve-pair matrix over all three operations, checking volume
against the closed form *and* watertightness, answered it — and the answer was
worse than the test suite suggested.

**Four of the thirty-six returned a solid with a hole in it.** They passed every
existing test, because every existing test used a case that worked. A boolean
that returns a broken solid is worse than one that returns nothing: the caller
cannot tell. That is a violation of the property the whole stack is built on,
and it was invisible until something measured the cases nobody had tried.

### Fixed

- **Silent drops.** `split_planar` and `split_face` dropped a piece whenever they
  could not find a point to classify it by — `Ok(Vec::new())`, no error. A box
  cut by a sphere came back with five walls and a dimple, its *top face missing
  entirely*, reported as success. Those paths now return
  `Declined::UnclassifiablePiece`.
- **The sample search was too narrow.** `sample_between` fanned triangles from
  one corner of the outer ring, so a quad yielded two candidates, both near the
  middle — and a hole in the middle swallowed both. The material is the *ring*
  around the hole, which nothing looked at. It now walks in from each vertex and
  then over a grid. This alone turned box/sphere from a wrong answer into an
  exact one, all three operations.
- **Degenerate rim loops.** A swept face's boundary loops are its rims, and a rim
  is a straight line in parameter space enclosing nothing, so ring containment
  rejected every candidate. Such a face is described by its parameter range
  instead.

### Enforced

`boolean` now tessellates its own result before returning and declines
`NotWatertight` if it is not closed. Whatever the cause — a range that does not
quite meet its neighbour's, a rim two faces describe differently — the promise is
enforced rather than argued for. It costs one tessellation, which is the right
price for the difference between declining and being quietly wrong.

### Then the two the gate caught

`NotWatertight` on coaxial cylinders and cone/cylinder was a real bug, and the
gate made it findable: the rejected body had exactly the right topology — four
faces, an annular cap at each end, a flipped bore wall — and 288 open edges.

288 is 2 × 144, and the cap's outer loop had **16** points where the rim *edge*
had 145. A loop is a fixed polyline; an edge gets refined to tolerance by
`refine_edges`. Carrying both means they describe the same circle until the
moment it is subdivided, and then they come apart along every segment in
between. Every straight-edged case passed throughout, because refining a
straight edge changes nothing — which is why a box drilled through was watertight
and a cylinder bored through was not.

The fix is to let the edges own the boundary wherever they back it: a boolean
piece keeps trim loops only when nothing else describes its boundary. Both cases
now resolve exactly (tube 125.627 against 125.66; cone 34.874 against 34.91).

### Where it stands

Twenty-one of thirty-six resolve, **all watertight, all with the correct
volume**; fifteen decline. Before this pass: seventeen correct, four wrong,
fifteen declines. The declines are the map of what is not implemented —
`NeedsArrangement` (box/box, sphere/box) wants a 2D arrangement in parameter
space, `NoClosedForm` (crossing cylinders, torus/cylinder) wants numeric SSI,
and `CoincidentFaces` wants the mesh kernel's keep/drop rules.

`tests/brep_boolean.rs` holds the matrix, so this cannot silently degrade.

### Also added

`Body::transform` / `Body::translated`, which the matrix needed and which were
missing: there was no way to position a `cuboid`. The topology is untouched — the
same faces and shared edges, so a moved solid is still a solid — and vertices go
through the same f64 affine the surfaces use rather than `Matrix4`'s f32, because
a vertex that lands even slightly off its surface stops being shared and the seam
it was holding closed opens. It declines a transform it cannot represent: a
non-uniform scale turns a sphere into an ellipsoid, which is not one of these
surfaces.


## Chaining: a result has to be usable as an input

The natural next question after "does one boolean work" is "does the next one".
Three bores drilled into a plate, each cut into the output of the last — which is
how anyone actually models — and the **second** one declined.

The watertightness gate made it diagnosable rather than merely wrong. The
rejected body's topology was flawless: eight faces, sixteen edges, every edge
used by exactly two faces, two bores. The failure was in filling it, and per-face
areas said where — the top face covered 920.751 of its 920.751, and the bottom
face covered **335.946**.

### The bridge had no visibility test

`bridge_holes` connects each hole to the ring by its rightmost vertex and the
nearest ring vertex to the right of it. With one hole that is always fine: the
outer boundary is the only thing there and it is always reachable. With two, the
ring already contains the first hole, and the nearest vertex may sit on its far
side — so the bridge runs straight *through* it. The ring is then
self-intersecting, and the ear clip fills part of the face and stops.

Whether it happened depended on ring ordering, which is why one face of a plate
was perfect and the other was a third filled.

The fix is the test that was missing: a candidate bridge is rejected if it
properly crosses any edge of the ring or of the holes not yet merged, and
candidates are tried nearest-first, rightward before anywhere. Chained booleans
now work — three bores giving 3761.50 / 3683.00 / 3604.50 against 3761.46 /
3682.92 / 3604.38 — and the finished part survives a STEP round trip with its
volume equal to twelve digits and all three bores still cylinders.

### A second degeneracy in the same place

`Body::plate_with_holes` (added for this, since there was no way to author a
multi-hole plate) then failed on three *collinear* holes — three open edges out
of some 2400, with the correct area covered.

Naming the three unpaired edges gave it away immediately: they were the three
edges of a single triangle whose corners were (−8, 2.5), (0, 2.5) and (8, 2.5) —
the three rims' extreme points, exactly collinear. A bridge runs into the
rightmost point of a circle, where the tangent is vertical, so the bridge is
collinear with the ring it lands on and the ear there is exactly flat. The clip
drops flat ears, and dropping that one takes a rim vertex the bore wall still
has: the face then shows one edge where the wall shows two.

Emitting the zero-area triangle instead does not help — its edges pair with
nothing either. Refusing the bridge does: tangency is now a reason to pick a
different target, falling back to clear-but-tangent only if nothing avoids both
(tangency costs a triangle, crossing costs the face).

### And a third: the ring was only weakly simple

Collinear centres on a *diagonal* still failed — six open edges, correct area.
The six were the three edges of one flat triangle on each face, each used
**once**, and the ear clip was not terminating early, so the triangulation was
completing and still leaving diagonals unpaired.

That only happens on a polygon that is not simple. Bridging chains the holes —
each one to the nearest vertex, which is usually the previous hole — and a chain
of bridges makes the ring *weakly* simple: the channels meet at shared vertices,
where an ear clip has no guarantee. What it clipped at the end was a zero-area
sliver joining three holes, and the sliver's edges paired with nothing.

The fix is not another degeneracy test but removing the cause: prefer an **outer
boundary** vertex over a hole vertex when choosing a bridge target, so each hole
reaches the boundary independently and its channel stays its own.

Emitting the flat ears instead was tried at two points and fixed nothing; both
attempts were removed rather than kept.

### Where it stands

Thirteen arrangements, all watertight with the correct volume: one hole; two,
adjacent and diagonal; three collinear at two spacings; three collinear on each
diagonal; four collinear; four on a diagonal; three in a zig-zag; three along
the short axis; five in a row; five including a centre hole. Before this pass,
five of the thirteen failed.


## Coaxial surfaces of revolution

The `NoClosedForm` declines were worth a second look: some were genuine, and
some were only a missing case.

A torus was handled against planes and nothing else, so a torus against a
cylinder, a sphere, a cone or another torus all fell through — including
*coaxial* pairs, which are exactly the ones that occur: an O-ring groove, a
rounded rim, a bore through a doughnut.

Two surfaces of revolution about the same axis meet in **circles**, so the whole
family reduces to intersecting their profiles in the `(ρ, h)` half-plane — a
three-dimensional quartic becomes a two-dimensional conic problem already solved
elsewhere in the file. The profiles are a vertical line (cylinder), a circle
(sphere, or a torus tube offset by its major radius), and a pair of rays (cone);
every crossing lifts to one circle.

Written as `Profile` + `profile_meet`, so adding a surface of revolution later
means adding one profile rather than one function per pair.

Off-axis pairs still return `Unknown` — those really are quartics.

### Measured

`Body::torus(major 4, minor 1)` against a coaxial cylinder, checked against
Pappus (a solid of revolution's volume is its profile area times the distance its
centroid travels; half a tube of radius 1 has area π/2 with its centroid 4r/3π
from the tube centre):

| | computed | exact |
|---|---|---|
| torus − cylinder(4) | 43.634 | π²(4 + 4/3π) = 43.67 |
| torus ∪ cylinder(4) | 345.123 | 345.26 |

Both watertight. A cylinder that passes clean through the hole now yields
`Disjoint` — an *empty* intersection, which is a result — where it used to
decline.

Coverage went from fourteen of thirty-three operations to **twenty of
thirty-six**, still with nothing wrong returned.


## `brep::planar` — the subdivision the remaining declines were waiting for

Every case still declining needed the same missing thing, so it was worth
measuring which ones before building anything. Four ordinary operations, all
`NeedsArrangement`:

| | |
|---|---|
| rounding an edge | a cylinder parked on a box's edge |
| a through slot | a cylinder crossing the block |
| a channel | a cylinder with its axis in the top face |
| clipping a corner | a sphere at a corner |

None of them is exotic, and all of them are the same shape of problem: a face
crossed by an **open** curve, one that enters and leaves through the face's
boundary rather than closing on it. A closed curve is a hole and the region
around it is the rest — that is a containment test. An open one cuts the face in
two, several cut it into several, and no containment test says which piece is
which.

`src/brep/planar.rs` is that subdivision: a planar graph of the boundary plus
the chords, walked half-edge by half-edge to read the pieces off. It works in
plain 2D parameter coordinates and is tested as such, without constructing a
solid to ask — eighteen tests covering one cut, cuts landing on corners,
parallel cuts, curved cuts, crossing cuts, and the refusals.

Two things it handles that a first attempt would not:

- **Crossing chords.** Two boxes overlapping at a corner put one cut from each
  of the other solid's side faces on the face between them, and they meet in the
  middle. Without the crossing as a node one chord passes through the other and
  the walk reads pieces that are not there.
- **Crossings that land on a polyline vertex.** A sampled curve crossing a cut
  through its own midpoint does this every time, and rejecting segment endpoints
  makes both adjacent segments disown it. Endpoints are accepted and the
  duplicate is deduped by position.

It declines rather than guessing, like everything else here: a chord that stops
*inside* the face is half a cut, and a chord lying along the boundary cuts
nothing off.

### Wired in, for planar faces

Two more pieces were needed to use it.

**Clipping.** A surface intersection is unbounded — two planes meet in a whole
line — but the faces are not, so the seam is only the stretch both of them
carry. `clip_to_face` samples and then bisects the two ends.

**Chord ends may land on another chord.** The first version required both ends
on the face's boundary, and two boxes overlapping at a corner immediately break
that: the two cuts on each face nearest the corner *stop at the corner they
share*. Neither reaches the boundary at both ends, and together they cross the
face. Crossings are now computed first, and a chord end resolves against the
boundary or against a junction.

### The bisection had to run to precision, not to tolerance

The first attempt still declined, and the reason is worth recording. The two
chords meeting at a corner came out as `(…, 0.0)` and `(0.0, −7.15e-7)` — 7e-7
apart, where the snapping threshold was 4e-7. They were not *near* each other by
accident: they are the same point, the corner of the other solid, found twice.

The bisection was stopping at `tolerance × 1e-3`, which is exactly that
magnitude. But these endpoints are not approximations of a position — they are
*identities*, and the subdivision has to see them as one node. Run to `1e-14` of
the span instead, they agree to 1e-13 and the cut closes.

### Measured

| | computed | exact |
|---|---|---|
| box ∩ box (corner) | 8.000 | 8 |
| box − box | 56.000 | 56 |
| box ∪ box | 120.000 | 120 |

All watertight, all faces still planes — nine of them where there were six,
since the three the other box cuts each come back in two pieces.

Coverage: **23 of 42 operations**, up from 20 of 36, with the two new probe
cases (rounding an edge, a sphere at a corner) added to the matrix as declines.

### Still declining, and why

- **Swept faces crossed by chords.** A cylinder parked on an edge needs the same
  subdivision in *its* parameter domain, which wraps — so its boundary is not a
  ring in `(u, v)` and the walk needs a periodic domain. Rounding an edge and
  clipping a corner with a sphere both wait on this.
- **Coincident faces.** Two boxes sharing a face plane. The overlap has to be
  computed in that plane and the keep/drop rules applied — a 2D boolean rather
  than a 2D subdivision.
- **A hole and a cut on one face at once.** Handled separately; together they
  want the holes carried through the subdivision.


## A wrapping parameter is a circle, not an interval

`torus ∩ cylinder` was the last case returning `NotWatertight` — 282 open edges
— while the *same pair's* difference and union were exact. The gate caught it;
the asymmetry said where to look.

A torus wraps in both parameters, so cuts across its tube divide a **circle**.
The stretch from the highest cut back round to the lowest is one band. Splitting
it as an interval — first band from the parameter origin, last band to it —
cuts that band in half at a seam that does not exist, and leaves the two halves
with nothing shared between them.

Difference and union kept the *other* band and never noticed. The intersection
keeps the one the false seam runs through, and opened along it.

`split_swept` now builds cyclic bands when the direction wraps: one per pair of
consecutive cuts, the last closing back onto the first.

| | computed | exact |
|---|---|---|
| torus ∩ cylinder | 35.265 | π²(4 − 4/3π) = 35.29 |
| torus − cylinder | 43.634 | 43.67 |
| torus ∪ cylinder | 345.123 | 345.26 |

A torus cut by a coaxial *sphere* works too, and its two pieces add back to the
whole torus — 45.723 + 33.174 against 78.96.

Coverage: **24 of 39 operations**. The remaining fifteen are two buckets and one
combination: coincident faces (two boxes sharing a face plane, which wants a 2D
boolean in that plane rather than a subdivision), swept faces crossed by chords
(whose domain wraps, so the subdivision needs a periodic one), and a hole and a
cut on the same face at once.


## Tracing what has no closed form

`NoClosedForm` was the last decline that meant *"this kind of surface pair is not
supported"* rather than *"this particular thing is not implemented"*. Two
cylinders crossing at different radii meet in a quartic; so does a torus with
anything off its axis. Declining a cross-drilled hole for that reason is
declining something entirely ordinary.

`intersect::march` traces them. The intersection of two smooth surfaces runs
along `n₁ × n₂`, so step that way and pull the point back onto both surfaces;
repeat until it closes or leaves. What makes it trustworthy rather than merely
plausible is the correction — a 2×2 solve along the two normals, driving each
point to within `tolerance × 1e-3` of *both* surfaces before it is kept, and
ending the trace rather than accepting a step that will not converge (which is
what a tangential meeting looks like).

`Curve3d` gained a `Sampled` variant to carry the result, since there is no conic
to name. A traced curve is not re-sampled anywhere: its points already hold the
tolerance it was traced at, and fitting a formula it does not have would only
lose accuracy.

Measured: a thin bore through a thick rod gives **two** closed loops, as it
should, every point within 1e-6 of both cylinders. Torus against an off-axis
cylinder likewise.

`NoClosedForm` is no longer reachable for analytic surfaces.

## Following the surface inside a trimmed face

A face trimmed to an arbitrary outline has no parameter rectangle to grid, so it
is ear-clipped in `(u, v)` — which fills the *boundary* and leaves the interior
spanned by flat sheets. On a plane that is the whole answer. On a cylinder it is
not.

`refine_on_surface` subdivides until it follows. The decision is made **per
edge**, from the sagitta between its endpoints, so two triangles sharing an edge
always reach the same verdict and the mesh cannot come apart along it. Boundary
edges are never split: they are shared with the neighbouring *face*, which is not
party to the decision.

Any face carrying trim loops now uses that fill, whatever surface it lies on.

## Where the remaining fifteen stand

Two buckets, and I want to be exact about the second.

**Coincident faces** (9 operations) — two boxes sharing a face plane. Needs the
overlap computed in that plane and keep/drop rules applied: a 2D *boolean*,
where everything above is a 2D *subdivision*. Not started.

**Swept faces crossed by curves** (6) — cylinders crossing, an off-axis torus, a
rounded edge, a sphere clipped by a box. These now get much further than they
did: the quartic is traced, the wrapping parameter is cut open in a gap no curve
occupies, a ring that wraps the domain completely is re-cut as a chord from one
seam edge to the other, and the pieces subdivide correctly. They fail at the
last step, and the reason is specific:

> A swept face's outline takes its `v`-sides from the **rim edges** it shares
> with the caps beyond them. Those rims are closed curves, so folded into the
> rectangle they start wherever they start — not at the seam. The outline's
> corner is then a diagonal rather than the seam edge, and a chord landing on
> the seam is not on the boundary at all.
>
> Adding the seam point to the outline fixes the outline and breaks the seam: the
> cap beyond it does not have that point, so the two faces meet along an edge one
> has split. The point has to go into the shared `Edge`, and both faces build
> their copy of it independently in their own `split_face` call.

That is a real design question about where carried edges are materialised, not a
missing formula, and it is where I stopped. The watertightness gate catches all
six, so they decline rather than returning anything wrong.


## The seam between a subdivided swept face and its neighbours

Another pass at the six `crossing cylinders` / off-axis operations. Four real
faults found and fixed; it still does not close.

- **Edges materialised once.** Every edge of both bodies becomes points *before*
  any face is split, so a seam point one face needs on a rim is seen by the face
  beyond it.
- **`seam_chord` sorted its points.** The same curve bounds a face on the other
  side too, where it is a plain ring in the order it was traced. Sorting gives
  the same *set* in a different order, so the two faces join them with different
  edges — and a seam is held by edges, not by points. It rotates now.
- **The opened ring lost its closing edge.** Cutting a ring at the seam has to
  put its *first point* back at the far edge, not a fabricated one, or the edge
  that closed the ring simply is not there on one side.
- **The seam crossing was not a vertex.** Where a curve crosses the seam has to
  be a point of the shared curve — the face on the other side joins the same
  points with an edge straight across, and one face splitting that edge while
  the other does not is exactly the gap they then leave. Inserting it needs a
  *bisection* for the point whose own parameter is the seam: the straight line
  between two traced points does not lie on the surface, so a fraction worked
  out from their parameters lands most of a step away.

And one regression caught by the suite rather than by reasoning: a curve at
constant `v` wraps the domain too, but it is a *band* cut — `split_swept`'s case,
which keeps the parameter rectangles. Forcing a seam onto the rims for it cost a
bored cone three edges. Only a wrapping curve that is **not** at constant `v`
needs one.

**Still open.** Both chords now reach both seam edges and carry every edge their
rings had, and `planar::subdivide` still refuses: one chord's near end lands
0.107 from the boundary — a whole sample step — where the other's is exact. I did
not get to the bottom of that.

Coverage is unchanged at **24 of 39**, with all seventeen boolean tests passing
and no regressions. The remaining work is the list in this document's previous
section, minus the four faults above.


## Coincident faces: attempted, reverted, one piece kept

Two faces on one surface. The plan was to cut the overlap out of both with the
existing subdivision, then let a rule decide which copy survives:

| normals | union | intersection | difference (A−B) |
|---|---|---|---|
| same way | keep A | keep A | drop both |
| facing | drop both | drop both | keep A |

The rule is written down and believed correct. What defeated it is the *cut*.
Feeding one face's boundary to the other as a closed curve only works when it
lies entirely inside — and `box/box edge`, where the two overlap partly, needs
their boundaries genuinely intersected in the plane. That is a 2D **boolean**,
which is what this was supposed to avoid needing.

Worse, `box/box flush` — the contained case, which should have worked — still
declined: the face is *also* crossed by the grazing cuts from the other box's
side walls, so it gets a hole and chords at once.

So the routing is reverted and `CoincidentFaces` is reported again. A rule that
keeps or drops whole faces which only partly overlap is worse than a refusal,
and the vaguer `NeedsArrangement` the half-done version produced was worse still
— a decline that no longer said what was wrong.

### Kept: `planar::subdivide` handles holes and closed paths

That came out of the attempt and stands on its own, with four new tests:

- **A closed path inside a region is a hole, not a cut.** It leaves the region
  connected. The walk finds such a ring wound the other way; each is now given
  to the smallest region enclosing it, and the one nothing encloses is the
  unbounded face and is dropped.
- **A hole and a cut together** on one face, which is what a face gets when a
  solid both crosses it and shares a wall with it. `split_planar` no longer
  refuses that combination — it passes the rings through as closed paths
  alongside the chords.

`Region` gained a `holes` field for it. Twenty-two tests.


## The polygon boolean, built out of the arrangement

Second attempt at coincident faces, this time building the missing primitive
deliberately rather than routing around it.

`planar::clip_to_region` trims a path to the part lying inside a region. That is
what turns a subdivision into a **boolean**: cutting one region by another's
outline works only when the outline is already inside, and when the two overlap
*partly* the outline has to be trimmed first — each surviving stretch then enters
and leaves through the boundary, which is a chord, which everything downstream
already handles.

Measured on the case that motivated it: a 4×4 region and a 4×4 overlapping it by
2×2 come apart into **4 and 12**, exactly. A corner cut likewise. Twenty-five
tests.

### It still is not enough for a shared wall

Not because the rule or the cut is wrong — both work — but because of the *other*
face pairs such a model produces. Two boxes sharing a wall also meet **edge-on**
along every adjoining face, and a curve lying exactly on both faces' boundaries
is a grazing contact: `clip_to_face` samples for the stretch inside a face, and
"inside" is not defined for a point on the boundary. It finds no run, or several,
and declines.

That is the classic hard case in solid modelling and it is not a detail to slip
in at the end. So `CoincidentFaces` is reported again, and the machinery that is
ready — the rule, `Piece::shared_wall`, the clip — waits on grazing contacts
being handled first.

Coverage remains **24 of 39**. What is new and permanent is the primitive, which
is the thing the next attempt needs and now exists, tested, on its own terms.


## Coincident faces — done

The blocker named last time was grazing contact, and thinking it through turned
out to make it small: **a curve lying along a face's boundary does not cut that
face**. The boundary already describes it. There is nothing to resolve — the
question was wrong, not hard.

Winding number decides a point *on* a boundary by whichever way the arithmetic
falls, so such a curve read as a scatter of in and out and both
`curve_reaches_face` and `clip_to_face` were confused by it. Both now ask
whether a point is inside by a **margin**, and a grazing curve is simply not
reaching.

That unblocked the shared-wall work, and two more distinctions had to be drawn
before it held. Both are about telling apart cases that look identical to a
vertex test:

- **Faces that are the same region** share all of it, and every vertex of each
  sits on the other's boundary.
- **Faces that merely adjoin** — every pair of walls of two solids side by side,
  on one plane — share *none* of it, and their vertices sit on each other's
  boundary too.

What separates them is whether an **interior** point of one falls in the other,
or whether their boundaries **properly cross**. Testing vertices answers
neither: with containment it made adjoining walls look shared and they declined;
with strict containment it made identical walls look separate, and one of the two
copies survived as an internal wall — a result that was watertight and *wrong* by
exactly its own contribution, 10.67 on a 128 union. The gate cannot see that, so
it is the kind of error worth going back for.

### Measured

Every configuration exact, every one watertight:

| | difference | union | intersection |
|---|---|---|---|
| identical solids | 0 | 8 | 8 |
| block on a plate | 64 | 72 | 0 |
| overlapping corner, sharing a plane | 48 | 112 | 16 |
| face to face | 64 | 128 | 0 |

The last is the one that caught the internal wall.

**Coverage: 36 of 51 operations**, up from 24 of 45 — every coincident-face case
now resolves, and `CoincidentFaces` no longer appears in the matrix at all. The
remaining fifteen are all swept faces crossed by curves.


## Swept faces: the traversal direction

With coincident faces done, the fifteen remaining declines are all one category.
Taking the most specific — two cylinders crossing — the subdivision was refusing
because one chord's near end landed 0.107 from the boundary, a whole sample step,
where the other's was exact.

The cause: `seam_chord` finds where the folded parameter jumps and rotates the
ring to start there. But **which way it jumps says which way the curve runs** —
a jump *up* means the curve was running down through the seam — and rotating to
the point after an upward jump lands on the *high* end of the range, not the low
one. One of the two curves on a cylinder is traced each way, so one worked and
one did not.

Taking the points in the direction of increasing `u` fixes it, and the
subdivision now returns the right three regions.

**Still not closed**: the pieces assemble with 160 open edges, all of them on one
face — the bored cylinder's wall, whose whole boundary goes unmatched.

Ruled out, so the next attempt need not repeat them:

- **Vertex indices.** The wall's hole loops carry the shared curves' own indices
  and its outline carries the rims', checked against the edge lists.
- **Edge refinement.** The pinning holds: every edge is the same length before
  and after `refine_edges` on the result.
- **A vertex in one loop and not the other.** The one found turned out to be a
  legitimate corner on the parameter rectangle's seam edge, belonging to that
  face's own boundary rather than to a shared curve.

What is left to look at is the *bore* face's region: its boundary is 121 points
against the two chords' 60 and 61, which leaves almost nothing for the stretches
of seam edge between them — and a region bounded by two chords at different `v`
must run along the seam from one to the other on both sides.

Coverage stays at **36 of 51**. What moved is that this case now gets through the
subdivision, which it never had before.


## Iteration: the rim seam

Chasing the 160 open edges on the bored cylinder's wall. Two fixes landed, two
attempts were reverted, and the cause moved but did not clear.

**Landed.**

- `rim_at` **sorted** the rim by parameter while the face beyond it walks the
  edge in stored order — the identical mistake `seam_chord` made, in the same
  file, for the same reason. It rotates now.
- `seam_origin` only produced a seam where a curve *wrapped*. A face being cut
  open needs its rims to reach the cut whether or not anything wraps, so it now
  answers for any face that will be subdivided — recognised by having a curve
  that is **not** at constant `v`, since those are `split_swept`'s bands and want
  no seam at all. Gated on the direction actually wrapping, or it costs the
  box cases twenty edges.

**Reverted twice, and worth recording why.** A chord's endpoint on the boundary
becomes an anonymous crossing node, welded by position — while the face on the
*other* side of that curve names the same point by index. Naming the node by the
chord's own vertex ought to fix it, and does fix the crossing cylinders' hole
seam. But the node's position and its index then disagree by up to the snap
tolerance, whichever of the two is used, and the weld puts them in different
places: box-on-box lost sixteen edges. Both placements were tried. The fix is
not the naming — it is making a chord's ends land *exactly* on the boundary
rather than within tolerance of it.

**Where the remaining edges are.** All on the rims, at the spacing
`adaptive_steps` produces — which means `rim_at` is returning `None` and
`parameter_outline` is falling back to sampling its own rim, which cannot match
the neighbouring cap's seventeen-point edge. That is the next thing to look at.

160 → 146 open edges. Coverage unchanged at **36 of 51**, everything green.


## Iteration: a chord of the boundary is still the boundary

`refine_on_surface` refuses to split *boundary* edges — an edge is shared with
the neighbouring face, which is not party to the decision. It recognised them as
edges consecutive in the ring, which is not enough.

Two points of a rim sit at the **same `v`**, so the midpoint between any two of
them is at that `v` too — on the rim, geometrically, however far apart along it
they are. Such a pair is a *diagonal* by the consecutive test, gets split, and
the new vertex lands on the boundary where the face beyond it has nothing. Every
one of those splits opens the seam a little further.

An edge is now also boundary when its midpoint falls on the ring.

**146 → 23 open edges** on crossing cylinders, everything else unchanged and
green. What remains is one hole-seam edge — the chord-endpoint naming problem
recorded above, still unsolved — and about twenty coarse rim edges that face 0
uses and the caps do not, which looks like the outline covering only part of the
rim after the seam rotation.


## Iteration: the loops match, so the fault is downstream

Chasing the twenty-odd rim edges left on crossing cylinders. The hypothesis was
that the outline covers only part of the rim after the seam rotation. **It does
not.** Printed side by side:

```
F0 outline  119, 119, 120 … 134, 119, 135, 150 …
F1 cap ring 134, 133 … 120, 119
```

The outline's rim side gives edges `(119,120) … (133,134), (134,119)` — exactly
the cap's sixteen, in the other direction. They pair. So does the second rim.
The trim loops of the two faces agree, and the seam still opens.

That moves the fault downstream of the loops, into `fill_planar` or the
refinement — the loops are no longer worth suspecting.

**Landed anyway**: the outline drops consecutive duplicate points. A seam point
can land on a rim vertex already there (the two showed as `119, 119`), and a
loop naming the same vertex twice in a row is a zero-length edge the ear clip is
entitled to make anything of. It did not change the count here, but it is not a
change that needs a failure to justify it.

Still 23 open edges. Everything green.


## Iteration: counting what each face emits

Rather than reasoning about the topology, count the edges each face's
*triangles* put on the rims. That settles it in one measurement:

```
F1 cap    16 of 16 rim edges
F2 cap    16 of 16
F0 wall   14 of 32
```

The wall's trim loop names every rim vertex — established last time — and its
triangulation covers barely half of them. Whatever is wrong is between the loop
and the triangles.

**Landed.** `parameter_outline` now deduplicates by where its points *land*
rather than by their parameters. A seam point can fall on a rim vertex already
there; their parameters differ by a hair while the surface puts them in the same
place, so they weld to one vertex and the loop names it twice in a row. The ring
was visiting one vertex three times. 14 → 16 rim edges, 23 → 22 open.

**Ruled out this iteration**, so the next need not re-check them:

- The ear clip terminating early — it completes.
- Degenerate triangles being dropped at emission — none are.

What is left is that the wall's ring covers the rim but its triangles do not,
with the clip completing and nothing being discarded. That combination should
not be possible, which suggests the ring-position→mesh-vertex mapping goes wrong
partway rather than anything being lost.


## Iteration: three changes to the ear clip, byte-identical output

Chasing the wall's missing rim coverage. Established first that the wall's
*ring* is complete — all thirty-two rim vertices are in it, in order — so nothing
is lost before the clip.

Then three separate changes to `earclip`, each plausible:

- Emit flat ears instead of dropping them. On a swept face the rim is a straight
  line in parameter space, so every ear along it is exactly degenerate — a real
  mechanism for losing exactly these vertices.
- Prefer real ears, taking flat ones only when nothing else remains, so the rim
  is not collapsed before any triangle reaches it.
- Both together.

**All three produced byte-identical output.** The first is explained: a flat
triangle emitted by the clip is dropped again by `degenerate_at` at emission, so
the net effect is nil. The others are not, and the conclusion is that the loss is
not in the clip at all. Reverted, per the rule that a change fixing nothing does
not stay.

One measurement error worth recording: the diagnostic counted only edges with
*both* endpoints in the rim's index range, which excludes a rim vertex paired
with a refinement vertex. The rim may well be reached and only its
boundary-to-boundary edges missing — a different fault, and the count was never
evidence against it.

Still 22 open edges. Everything green.


## Iteration: the diagnostics were lying

`triangle_face` recorded one entry per *index* rather than per triangle — three
each. `orient` looks it up by triangle number, so every triangle but the first
was voted under some other face's normal, and the majority that decides which way
a shell faces was reading the wrong ones. It also meant every per-face
measurement in the last three iterations attributed edges to the wrong face,
which is why "the wall covers half the rim" looked true and was not.

Fixed. It changed no outcome here — the vote survived being fed noise, which is
what a majority is for — but the measurements it corrupted had sent two
iterations down the wrong path.

With the attribution right, the twenty-two open edges are two faults, both
specific:

- **The same seam point, computed two ways, 1.3e-3 apart.** `insert_seam_vertex`
  bisects onto the *curve*; `planar::subdivide` intersects the chord's
  **polyline** with the boundary. The polyline's crossing differs from the true
  curve point by the sampling sagitta, which at this tolerance is larger than the
  weld quantum — so they stay two vertices, and the faces either side of the seam
  name different ones. This is the same wound as the earlier 118-vs-185, now
  measured rather than inferred.
- **A cap's fan diagonal emitted twice** — `(119,123)` used three times, twice by
  one face.

Neither was visible until the attribution was fixed.


## Iteration: the same fix, refused for the third time

With the attribution finally trustworthy, the seam fault is measured rather than
inferred: where a chord meets the boundary, the subdivision intersects the
chord's **polyline** with the boundary's, and that crossing sits a sagitta away
from the curve it is meant to be on — **1.3e-3 against a tolerance of 1e-3**. The
weld does not close that, so the two faces either side of the seam name different
vertices.

Snapping such a point onto the curve's own vertex **works**: crossing cylinders
go 22 → 18. It also **breaks two boxes overlapping at a corner**, which loses
twenty edges.

That is now three attempts, from three different angles:

1. Naming the node by the chord's vertex in `planar::subdivide`.
2. Placing the node at the chord's own point rather than its projection.
3. Snapping afterwards by proximity, at three different reaches — the modelling
   tolerance (too tight to catch it), half a sample step (which for a
   *two-point* chord is half its whole length, so it pulled in vertices from
   anywhere), and a few multiples of the tolerance.

All three fix the cylinders and break the boxes. Reverted again, since a
regression is not a trade worth taking.

The pattern is the finding: **box-on-box wants these nodes welded by position and
nothing else**, and every attempt to give them an identity takes that away. Why
is the thing to establish — not a fourth way to write the snap. A comment saying
so now sits where the fourth attempt would go.

22 open edges. Everything green.


## The blocker, resolved: curves have to be welded to each other

Three attempts had failed the same way, so this iteration asked *why* rather than
trying a fourth. Instrumenting box-on-box under the snap gave it immediately —
the unpaired edges ran between vertices at

```
(2.0, 0.0, 4.000043e-9)
(2.0, 4.000043e-9, 0.0)
```

Two distinct vertices at one corner, four **nanometres** apart. Three surfaces
pass through that point, so three curves meet there, and each arrives with its
own endpoint. Welding by position downstream merged them — which is why box/box
worked without the snap — and any attempt to name such a point *by the curve it
came from* picked a different vertex on each face, which is why it broke with it.

So the curves are now welded **to each other**, once, before anything is split.
A corner is then one vertex, and naming it is safe.

With that in place the snap can be turned on, and both cases work:

| | before | after |
|---|---|---|
| box/box (six operations) | exact | exact |
| crossing cylinders | 22 open edges | 18 |

The weld looks in the twenty-seven neighbouring cells as well as its own: two
points a nanometre apart can still fall either side of a grid boundary, and a
grid that only checks its own cell leaves them separate — which is the very
problem it exists to solve. Without that, two vertices 5.7e-9 apart survived into
the result.

Nothing regressed; everything green. The seam fault that has run through the last
seven iterations is fixed. What is left on crossing cylinders is eighteen edges
of some other cause.


## Iteration: the eighteen are a different shape entirely

With the seam fault gone, what is left on crossing cylinders reads as:

```
(119,123) x3 by [0, 1, 1]
(119,134) x1 by [1]
(123,124) x1 by [1]
```

`(119,123)` used **three times**, twice by the cap — and that pair is correct.
The cap's ear clip fans from vertex 119, so `(119,123)` is one of its interior
diagonals and belongs to two of its triangles. The third use is the wall's, and
that is the wrong one: the wall's outline runs **119 → 123 directly**, skipping
the rim points between.

So the wall's region boundary is taking a chord where the rim should be, and the
rim points it skipped are still on the cap's boundary with nothing to pair
against — which is the rest of the count.

A chord endpoint should not be landing on the rim at all here: the bore passes
through the wall nowhere near `z = −3`. Either a traced curve's clipped end is
landing there wrongly, or the subdivision is cutting the outline somewhere it
should not.

Not fixed. But the shape is different from everything before it — this is about
*where a chord ends*, not about how a point is named — and the previous fault is
confirmed gone rather than merely quieter.


## Iteration: two hypotheses eliminated, none confirmed

Last time's reading was that a chord endpoint must be landing on the wall's rim.
Measured, and it is not:

- The **bore's** face is the one with chords, and both end well inside it —
  `v = 3.04` and `6.96` in a range of `0..10`.
- The **wall's** face has no chords at all. Its two traced curves close on it, so
  it takes the ring path, and its outline is the full rectangle.
- That outline covers all sixteen rim vertices. The duplicate visible at the seam
  (`u = 3.142` twice) is removed by the position dedup that follows.

So the wall's outline is intact, its ring contains every rim vertex, the ear clip
completes, and nothing is discarded at emission — and its triangles still reach
only four of the sixteen rim edges before stopping.

Every step of that chain has now been measured individually and each is correct,
which means the fault is in how they compose. The next thing worth doing is not
another hypothesis but a direct trace: take one rim vertex the triangles never
reach, and follow it from the outline through `bridge_holes`, the ear clip and
`refine_on_surface`, checking at each stage that it is still there and still
adjacent to its neighbours.

18 open edges, unchanged. Everything green.


## A flat ear is still an ear

The direct trace found it in one measurement. The wall's ring has 157 positions
and produced **140** triangles where 155 are expected — and **fourteen positions
never appeared in any triangle at all**, positions 5 to 15 being rim vertices
124 to 134, every one of them at `v = 0`.

A rim is a *straight line* in parameter space, so every ear along it is exactly
flat. `earclip` removed those vertices without emitting anything — correct for a
zero-area triangle, and fatal here, because the cap beyond the rim was still
holding edges to them. A zero-area triangle draws nothing either, and keeps the
vertex attached; it is now emitted.

`crossing cylinders` difference and union resolve, watertight, and check against
`|A| + |B| = |A∪B| + |A∩B|` — which is the right test here, since the overlap of
two perpendicular cylinders of unequal radius is an elliptic integral and a
hand-computed expectation would be the less trustworthy half of the comparison.

Four faces: the rod's wall with two windows, its two ends, and the bore's wall,
all still analytic.

Coverage **28 of 39** in the matrix, up from 26.

### A process note

This fix had been attempted twice before and reported as "no change". It was not:
the patch never applied. The closure in the code reads `|&j| j != ia` and every
attempt had matched against `|j| j != &ia`, so the replacement silently did
nothing and I measured unchanged code twice while believing otherwise. The
elimination those iterations recorded — "the loss is not in the ear clip" — was
wrong, and cost two further iterations looking elsewhere. Applying a patch and
confirming it *landed* is not the same as applying it.


## An island on a curved surface keeps its ring

`crossing cylinders` intersection was the last of that trio, and the report said
`carried_through: 2` — two faces that did not fill at all, rather than a seam
that did not meet.

They were the two *windows* on the rod's wall: the regions inside the bore, which
`split_planar` emits as islands. An island had been made to rely on the `Edge` it
names rather than carrying a trim ring, so that the boundary refines with
everything else. That is right on a plane, where the fill reads the loops back
off the edges. On a **curved** surface it is not: such a face is filled from its
trim loops, and the parameter-rectangle fill that takes it instead wants rims at
constant `v` — which a traced seam is not. Without the ring the face simply does
not fill.

All three operations now resolve:

| | volume | faces |
|---|---|---|
| rod − bore | 62.614 | 4 |
| rod ∪ bore | 94.002 | 7 |
| rod ∩ bore | 12.149 | 3 |

and `|A| + |B| = 106.81` against `|A∪B| + |A∩B| = 106.15`, within 0.6% — checked
by that identity because the overlap of two perpendicular cylinders of unequal
radius is an elliptic integral.

Coverage **29 of 39**.


## A ring that does not fit is not a hole

The remaining four all declined at the same place: `split_planar`'s containment
test. Measured, two of them are on **planar** faces — a sphere meeting a box's
wall gives a circle of `v ∈ [−2.83, 2.83]` against a wall of `v ∈ [−1, 1]`.

Such a ring is not a hole in that face. Only the arc within the wall is a seam;
the rest is off the face entirely. Clipped with `planar::clip_to_region` — the
primitive built for coincident faces — each surviving arc enters and leaves
through the boundary, which is a chord, which the subdivision already handles.

`sphere/box` now gets through the subdivision for the first time. It does not
close yet (477 open edges), but it is past the refusal.

Also landed, though it changed none of these: a periodic **`v`** is now cut open
the same way `u` is. A torus wraps in both, and a curve can straddle the `v` seam
just as readily — folded relative to wherever the curve happens to start, its
parameters run outside the rectangle and containment rejects it out of hand.

The other three now decline in `clip_to_face` rather than in containment, which
is a different question: that is about an *open* curve's span on a face, not
about whether a closed one fits.

Coverage 29 of 39, unchanged.


## Iteration: mapping sphere/box, and a cleanup that was overdue

`sphere/box` now gets through the subdivision. What it produces:

- The sphere is cut by all four of the prism's walls and comes back as **four
  pieces**, each with one loop of about 120 points and one edge.
- The prism's walls each receive one curve and are split too.
- Every one of the four sphere pieces is **entirely** open — 119, 119, 120, 119
  edges — so their loops share nothing with the walls beside them.

That is a different failure from the crossing cylinders, where the seams met and
only the rim did not. Here nothing meets at all, which points at the sphere
pieces' loops rather than at any one stage of the fill.

**Cleanup.** Nine env-gated diagnostic blocks were still in `body.rs` from
earlier iterations — I had removed one occurrence at a time and believed the file
clean. They were inert (each behind an unset variable) but they had no business
shipping. All removed; the two `THREERS_BREP_DEBUG` hooks that predate this work
are left alone. Worth noting alongside the unapplied-patch mistake two iterations
back: both come from trusting an edit instead of checking it.

Coverage 29 of 39, unchanged.


## A closed curve is only a ring where all of it is on both faces

Following the four disconnected sphere pieces: each carried the vertices of a
*different* curve, so they were four **islands**, one per wall — not a
subdivision at all.

The cause: a plane cutting a sphere gives a full circle, and the wall doing the
cutting is two units wide. Only the arc across that wall is a seam; the rest of
the circle is nowhere. Taken whole it becomes an island on the sphere, and the
sphere comes back in as many disconnected pieces as there were walls, none of
them meeting anything.

A closed curve is now treated as a ring only where the *whole* of it lies on both
faces; otherwise it is clipped like any open curve. The test is asked only of a
face whose boundary encloses something — a swept face's loops are its rims,
which are straight lines in parameter space enclosing nothing, so there is no
region to be inside of and a curve on it is inside by construction. Without that
proviso the check refuses curves on every cylinder and cone, and takes the
working cases with it (measured: `a_bore_across_a_rod` and
`a_curved_face_split_by_a_curved_one` both broke before it was added).

`sphere/box` is now classified correctly and declines at the subdivision instead
of assembling something open — no better by the count, but from a sound footing
rather than an unsound one.

Coverage 29 of 39.


## Two gaps in `clip_to_face`, both ordinary

Following `sphere/box` past its misclassification, it declined twice more, each
time on something unexceptional.

**A face need not have a boundary.** `clip_to_face` began by finding an
enclosing loop and giving up without one — but a whole sphere has no edges at
all, and a swept face's loops are rims that are straight lines in parameter space
enclosing nothing. Such a face is bounded by its *parameter range*, and every
point of the surface within that range is on it. `curve_reaches_face` had read
them that way for a long time; this did not, so a sphere had no clippable region
and the whole operation declined.

**A curve can be on a face more than once.** It then returned a single interval
and refused when there was more than one. A circle crossing a strip is inside it
*twice* — a sphere cut by a narrow wall does exactly that — and taking one of the
two loses half the seam. It now returns every stretch, and the caller intersects
two *sets* of intervals, each survivor its own seam.

That was item three on the gaps list ("multiple clip intervals"), and it is now
closed.

`sphere/box` gets through with the right classification and assembles, though not
yet closed — 132 open edges. Nothing regressed.

Coverage 29 of 39.


## A sphere clipping a corner

The two `clip_to_face` fixes from the previous iteration unlocked a case that had
never been tried against them: a sphere centred on a box's corner, cutting three
walls at once. Each cut is an *arc* — the plane of a wall meets the sphere in a
whole circle, and only the part crossing that wall is a seam — and a circle can
be on a wall in two separate stretches, which is precisely what the interval-set
clip was for.

Exact on all three, all watertight:

| | computed | exact |
|---|---|---|
| block ∩ sphere | 14.13 | 4/3·π·27/8 = 14.137 |
| block − sphere | 305.87 | 305.863 |
| block ∪ sphere | 418.51 | 418.960 |

Seven faces for the difference — six walls, three of them cut, and the dimple,
which is still a `Sphere`.

It was not in the coverage matrix, which is why the count had not moved when it
started working. Added, along with a test of its own.

Coverage **32 of 42**.

## A chord that crosses the seam was never cut there

`sphere / box` had been failing at 132 open edges. Measuring the subdivision of
the sphere face showed twelve chords going in and **fifteen** regions coming
out, where a square column bored through a ball should give three: two caps and
the band between them.

The chords are the sphere's meets with the column's four side planes — two
closed loops of four arcs each. Neither loop leaves a gap in `u`, so
`seam_origin` has nowhere clean to put the seam and it lands in the middle of an
arc. That much is expected and already provided for: `insert_seam_vertex` puts a
vertex on the crossing before either face looks at the curve.

What was missing is that nothing then *cut the chord there*. The fold puts every
point back inside the rectangle one at a time, so the arc came out as a run of
points that jumps a whole period between two neighbours:

```
u[1.571,7.709] ends (2.36,1.08)->(7.07,1.08)
```

In parameter space that jump draws a line clear across the face at `v ≈ 1.08`,
and the fifteen regions are the slivers it leaves behind.

`split_at_seam` cuts the chord where it crosses. The first attempt closed each
run off at the neighbour brought back alongside it, which overshoots — a whole
step past the seam, `7.855` where the edge is at `7.782` — and a run that leaves
the rectangle is not a chord of it, so `subdivide` declined outright. Ending the
run *on* the edge, with `v` interpolated to the crossing, and starting the next
run at the same point on the far edge, gives:

```
u[7.069,7.782] ... , u[1.498,2.356] ...   -> Some(3)
```

Three regions, and the two sides of the cut meet at one point rather than across
a gap.

Open edges **132 → 6**. Not yet watertight, so `sphere / box` still declines and
the matrix does not move, but the failure is now six edges of a different kind
rather than a shredded face. All 57 B-rep tests stay green — the fold this
replaces was on the path every wrapping face takes, so cylinders and cones and
tori were the real risk, and none of them moved.

Coverage **32 of 42**.

## The seam moved between choosing it and cutting at it

The six edges left over from the last iteration were all at `(0, 1, ±2.8284)`,
and the vertex on the wrong side of them was:

```
(0.0728, 1.0027, -2.8265)
```

`y = 1.0027`. On the sphere, off the plane at `y = 1` — and its direction in
`xy` is `atan2(0.99738, 0.0724) = 1.4983`, which is exactly the seam. So it is a
point of the cut, and it is not on the surface the cut is shared with.

The first guess was that `split_at_seam` interpolated it. It does interpolate,
and `insert_seam_vertex` exists precisely so it does not have to: that one
bisects onto the surface instead. So the fix looked like *prefer the crossing
vertex when the curve carries one*. Made it; nothing changed. The curve carried
no such vertex.

Printing the origin at both places says why:

```
pre-origin  tag0 f0 sphere = 3.069130
split-origin     f0 sphere = 1.498333
```

The seam goes in the widest stretch of `u` no curve occupies. The pre-pass finds
that stretch, picks 3.069, and inserts a vertex there — which *fills* the
stretch it just measured. When `split_face` asks the same question of the same
curves, the widest gap is somewhere else, and the face is cut at 1.498 with no
vertex on it. The two faces of the seam then describe it with points neither of
them has.

Fixed by deciding once. `seam_origins: HashMap<(u8, usize), f64>` records what
the pre-pass chose; `split_face` looks it up and only re-derives it if there is
nothing recorded. Recomputing a choice that its own consequences invalidate is
the general shape of this bug, and the map is the general fix for it.

| sphere / column | computed | exact |
|---|---|---|
| difference | 89.893 | 90.011 |
| intersection | 23.085 | 23.0865 |
| union | 169.893 | 170.011 |

The column's share has no closed form; 23.0865 is `∫∫ 2√(9−x²−y²)` over
`[−1,1]²` by quadrature. The 0.13% on the two that contain the ball is the
sphere's own chordal deficit at `1e-3`, and it cancels in
`|A| + |B| = |A∪B| + |A∩B|` to within 0.12 — the same number, which is what
tells you it is the tessellation and not the boolean.

Three operations gained, and `a_square_column_bored_through_a_ball` asserts the
volumes rather than counting the case as resolved.

Coverage **35 of 42**.

## The weld was eating the seam vertices

The remaining declines are all `NeedsArrangement { face: 0 }`: `round an edge`
and `torus/cylinder off-axis`, three operations each. `box/box` — both the
overlapping and the flush case — resolves now and had not been re-measured
since.

Following the off-axis torus: the *torus* face subdivides. The **cylinder** face
is the one that declines, and its two chords are

```
u[1.911,8.194] ... ends (1.911,2.619)->(8.194,2.619)
u[1.924,8.194] ... ends (1.924,3.395)->(8.194,3.395)
```

The seam is at 1.911. The second chord starts 0.013 past it — a whole sample
step — so it dangles, and a cut that stops in open space is not a subdivision.

`seam_chord` forces its *far* end onto the seam but takes its near end from the
first sample past the crossing. That is exact only when there is a vertex
sitting on the seam, which is what `insert_seam_vertex` is for. And it had put
one there:

```
curve 1 on tag1 f0 cylinder origin=1.9113
  seam-insert n=293 at i=1 u=Some(1.9113013165535995)
  seam-insert done: len now 294
```

294 going in. 293 by the time the face was split, and the nearest input to the
seam 1.27e-2 away. Something removed it, and what removed it was the weld:

```rust
curve.vertices.dedup();
```

The weld runs *after* the seam pre-pass. A seam vertex lies between two samples
of a traced curve, and on a finely traced one it is within the weld quantum of a
neighbour — so the pre-pass put the vertex in and the weld took it straight back
out. It survived on the torus (one insert of two) and not on the cylinder, which
is why only one of the four chords was short.

Fixed by ordering: the pre-pass now runs after the weld. That is also where it
belongs for a second reason — the shared-wall cut adds curves of its own, and
running before it meant those never got a seam vertex at all.

All four chords now start exactly on the seam, and the case moves from
`NeedsArrangement` to `NotWatertight { open_edges: 138 }` — through the
subdivision for the first time, and declining later and for a different reason.
Coverage does not move.

Worth saying plainly: this was a silent fault, not a local one. Any seam vertex
on a finely traced curve was at risk, and the cases that pass today pass because
their vertex happened to fall outside the quantum.

Coverage **35 of 42**.

## `v` had the same seam bug `u` did

The 138 open edges on the off-axis torus were a single unbroken chain at
constant `z = -0.3968`, every one of them on the torus's biggest piece and on no
other face. On a torus of `R = 4, r = 1` a constant-`z` chain at `ρ ≈ 3.08` is a
circle of constant tube angle: `4 + cos v = 3.08`, `sin v = -0.397`, so
`v = -2.74`. The face's `v_range` began at `-2.734`. The crack ran along the
`v` seam.

Which is the bug already fixed for `u`, in the block immediately above it:

```rust
for (_, c, _) in chords.iter_mut() {
    for p in c.iter_mut() {
        p[1] = fold(p[1]);
    }
}
```

Folding point by point leaves a whole-period jump wherever a chord crosses, and
in parameter space that jump draws a line right round the tube.

`split_at_seam` now takes an `axis` and does either. The `v` split has to run
*last*, after the `u` handling, because the wrapping-ring branch turns rings
into chords and those need it as much as the rest — at `v`-fold time the torus's
curves are still rings and a split there would miss them entirely.

Open edges **138 → 34**, and the result gains a face.

What is left is not a seam problem. Two kinds:

```
(-4.4111,-2.1270,0.4417)-(-4.3744,-2.1094,0.5162)  faces [0, 0, 0, 0]
(-3.0000, 0.0000,0.0000)-(-2.9999, 0.0331,-0.0096) faces [2]
```

An edge used four times by one face is a loop doubling back on itself. And
`(-3, 0, 0)` is `ρ = 3` on the torus — its inner equator — which is exactly 4
from the cylinder's axis at `x = 1`. The two surfaces *touch* there. The
intersection pinches to a point, the two loops meet, and the four-times-used
edge is the curve folded back on itself at the pinch.

So `torus/cylinder off-axis` is a tangential-contact case, which is the item
already on the gap list as unimplemented (`settle` returns `None` on parallel
normals). It is not going to close by fixing seams, and the honest thing is to
say so rather than keep whittling at it. Coverage does not move.

The `v` split is worth having regardless: it is a correctness fix for every
torus whose cuts cross the tube seam, and those were silently cracking.

Coverage **35 of 42**.

## Rounding an edge: two points is a curve, and a ring is not its own hole

`round an edge` declined at `chosen_origin()` returning `None`. The cylinder's
face carries four curves:

```
seam_origin curve 0: 2 pts
seam_origin curve 1: 32 pts
seam_origin curve 2: 32 pts
seam_origin curve 3: 2 pts
```

The cylinder lies *on* the box's edge, so two of the box's faces contain its
axis and cut it in straight lines — and a straight seam arrives with exactly two
points. `seam_origin` skipped anything shorter than three. The 32-point curves
are the arcs where the cylinder leaves through the box's ends, and those are at
constant `v`, which it also skips. So every curve was skipped, `wants_seam`
stayed false, and a face that plainly needed a seam was told it wanted none.
Two points is a curve. Guard lowered to `< 2`.

That got to the arrangement, which then produced this:

```
outer u[3.9270,10.2102] v[0,12]
chords  u[6.2832,6.2832] v[2,10]  n=2     the two straight seams
        u[6.2832,7.8540] v[10,10] n=32    the two arcs
        u[6.2832,7.8540] v[2,2]   n=32
        u[7.8540,7.8540] v[2,10]  n=2
-> Some(["area=75.3982 holes=0", "area=0.0000 holes=1"])
```

The four chords close a rectangle *inside* the face. The right answer is
`62.83 with one hole` and `12.57 with none`. What came out was the whole face
with no hole at all, and a second region of zero area holding the hole — because
the cycle had been given to **itself**.

`subdivide` hands each inverted ring to the smallest region enclosing it:

```rust
if r.area <= area || !point_in_ring(&r.outer.uv, probe) { continue; }
```

A ring and its own reversal have the same area to the last bit, so `<=` is a
coin toss, and the probe is a vertex *of* that ring, lying exactly on the
boundary where `point_in_ring` may answer either way. The cycle enclosed itself,
its area cancelled to nothing, and the region that really contained it never got
its hole. Requiring the container to be strictly larger — `area * (1.0 + 1e-9)`
— settles it.

All three operations resolve:

| | computed | exact |
|---|---|---|
| difference | 263.758 | 263.451 |
| intersection | 56.242 | 56.549 |
| union | 600.656 | 602.743 |

Checked rather than assumed: `|A−B| + |A∩B| = 320.0000`, which is `|A|` to four
places. The shortfall is all in `|B−A|`, and every sign is what an *inscribed*
polygon does to a convex solid — the quarter cylinder measures short, so the box
minus it measures long by the same 0.31. Nothing is missing.

But it does not shrink with tolerance:

```
tol 1e-3  gap=2.1982
tol 1e-4  gap=2.3850
tol 1e-5  gap=2.4027
```

So there is a separate fault, and it is not in the boolean: a *trimmed* curved
face is filled from its stored trim loop, and the loop is not refined when the
face is re-tessellated. Working back from the deficit, the three-quarter wall
gets about 22 segments where `1e-4` asks for nearly three hundred. The claim
that a result re-tessellates at any tolerance holds for the faces that keep
their parameter rectangles and not for the ones that carry loops. Added to the
gap list; it also means the 1% volume checks are weaker on curved trimmed faces
than they look.

Three operations gained.

Coverage **38 of 42** — and of the four not counted, one is `box/box flush`
intersection, which is legitimately *empty* and never was a gap. Only the
off-axis torus declines, on the tangency found last iteration.

## Trim-loop refinement: located exactly, not fixed

The volume gap on `round an edge` does not shrink with tolerance, so it is not
tessellation noise. Measuring the loops of `cylinder − box` at two tolerances:

```
tol 1e-3: 7 faces, 37076 tris    F0 cylinder loops=[34, 64]  edges=2
tol 1e-5: 7 faces, 446090 tris   F0 cylinder loops=[34, 612] edges=2
```

The hole refines, 64 → 612. The **outer** does not: 34 points at both, which for
a face bounded by two full circles is sixteen segments each — the resolution
`Body::cylinder` was *constructed* at. So the three-quarter wall is a
sixteen-sided prism no matter what tolerance is asked for, and that is the
missing 2.4.

`refine_edges` says why, and says it deliberately:

> An edge a *trimmed* face also describes is left alone. […] The loop was
> sampled to tolerance when it was built, so nothing is lost by leaving both as
> they are.

The first half of that reasoning is sound and the second is wrong. A carried
loop is sampled where the *input body's* rim was, not to any tolerance of the
caller's.

Three attempts, and none of them ships:

* Refine the pinned edges anyway. `cylinder − box` comes back with **286** open
  edges. The pin is load-bearing.
* Carry the same new vertices into the loop by index. Never fires — the boolean
  clears a carried loop's vertex indices, because they number the input body and
  not the result.
* Carry them by *position*, matching consecutive loop points against the edge
  vertices they came from. 286 → **9**. Better, and still a regression.

Nine is not nearly there. The parts of a loop that no edge backs — the seam
segments — still do not move, and a loop that gains points where an edge backs
it and keeps them where nothing does is a different shape from either.

Reverted. A change that trades a known shortfall for open edges is worse than
the shortfall, and the rule has been that a result must never be worse than a
decline. What is bought is the diagnosis, which is now exact rather than
suspected, and it is written into the function so the next attempt starts from
it instead of from the old comment's assumption.

Doing it properly means refining a loop and every edge along it as one object,
with the parts no edge backs re-sampled on the surface — not splicing one into
the other. That is a bigger change than an iteration, and it is the next thing.

Also worth stating: this is why the 1% volume checks pass on curved trimmed
faces. They are not tight enough to see it. The identity that did catch it is
`|A−B| + |A∩B| = |A|`, which held to four places while `|B−A| + |A∩B|` was 2.4
short of `|B|`.

Coverage **38 of 42**, unchanged.

## Refining segments instead of edges — and the gap it turned out not to be

The plan was to stop treating an edge and the loop that describes it as two
things. A boundary segment is named by where its ends are; it is subdivided
once; every edge and every loop running along it takes the same points. There is
then nothing to keep in step, because there is only one answer — and a stretch
of loop no edge backs is a segment too, subdivided on its own face's surface.

It works, in the sense it was built for. The three-quarter cylinder wall's outer
loop went from 34 points to 258, and refines with the tolerance instead of
staying at whatever `Body::cylinder` was constructed at:

```
refine: 112 segments, 32 subdivided, 224 points added
F0 cylinder loops=[258, 64] edges=2
```

And it opens 48 edges on `a_bore_across_a_rod`. A seam shared by two curved
faces gets subdivided with whichever face claimed the segment first, so the
point lands on one surface rather than on the curve the two share; pairing the
surfaces properly — the segment's two claimants — does not fix it either.
Reverted, both attempts.

**But the measurement that matters is a different one.** Refining those loops
moved the volume gap by 0.09 out of 2.31. The rest does not go away as the
tolerance falls — it settles:

```
tol 1e-3   |AnB| = 56.3153    identity gap 2.11268
tol 1e-4   |AnB| = 56.2423    identity gap 2.29946
tol 1e-5   |AnB| = 56.2363    identity gap 2.31383
```

`|A∩B|` converges on 56.236. The quarter cylinder is `9π/4 × 8 = 56.5487`
exactly. That is 0.55% short in the limit, and a limit is not a sampling
artefact — the solid being built is the wrong shape by a fixed amount, and
`|A−B| + |A∩B| = |A|` holding to five places says the two share one displaced
boundary rather than one of them being wrong on its own.

So the last two iterations have been chasing the wrong thing. The trim-loop
resolution is a real limitation and worth fixing, but it is not what the volume
says. The next thing to find out is where that boundary actually sits — most
directly by taking the intersection's own trim loops and measuring the quarter
arc they describe against the circle it should be.

Reverted to the pinned refinement. Coverage **38 of 42**, unchanged, and the
diagnosis in `refine_edges` now records both failed approaches so the next
attempt does not repeat them.

## Where the half a percent goes: a fan across the wall

The gap converges, so the shape is wrong by a fixed amount. Measured on
`box ∩ cylinder` at `1e-5`, over the vertices the index buffer actually
references — the position buffer also carries vertices no triangle uses, and
taking the bounding box over all of them says `x[-5,8] y[-6,6] z[-2,5]`, which
is the *union's* extent and sent me looking in the wrong place for a while:

```
bbox   x[2.00000,5.00000] y[-4.00000,4.00000] z[-1.00000,2.00000]
wall radius [3.000000,3.000000]   (exact 3)
```

The boundary is exact. Every point of the wall is at radius 3 to six places,
and the bounding box is the quarter cylinder's to five. So the loops are right
and the vertices are right, and the volume is still 0.55% short.

It is the *interior*:

```
wall triangles 200515, worst sag 7.485738e-1   (tolerance 1e-5)
sag histogram by decade from 1e-6: [32339, 107660, 60089, 384, 36, 7]
```

A sag of 0.749 on a radius of 3 is a chord spanning 83 degrees — of a
90-degree wall. Seven triangles are at least that bad and four hundred are
within three decades of it, at 75,000 times the tolerance.

`refine_on_surface` refuses to split them, and its `boundary` test is right to.
Ear clipping *fans*, so it emits triangles with an edge running from one ring
vertex to another far along the same ring. On a cylinder every point of a rim
shares that rim's `v`, so in parameter space such a chord lies along the
boundary however long it is — and splitting it would put a vertex on the
boundary that the face across the seam does not have. The rule is sound; the
chord should never have existed.

Splitting those triangles at the centroid instead: 201432 triangles become
655070, and the volume comes back **56.23630** — the same to five decimal
places. That is not a weak result, it is a proof. The three children keep all
three of the parent's edges, so if the sag were anywhere but in the edges the
number would have moved. It did not. The chord is the whole of it.

Reverted, and the diagnosis written into `boundary` where the next attempt will
find it. The chord has to not be created: an edge flip against the neighbouring
triangle, or triangulating the parameter rectangle instead of fanning the loop.

That also settles the last two iterations' question. The trim-loop resolution
work was aimed at the boundary, and the boundary was never wrong.

Coverage **38 of 42**, unchanged.

## The chord fix works, and is wrong in one configuration

Replacing a boundary-collinear chord with the ring's own vertices between its
ends. Nothing invented: every point of the fan that replaces it is a ring vertex
the face across the seam names too, so the seam cannot open.

On the quarter cylinder it is decisive. The error stops growing with tolerance
and starts shrinking:

| tolerance | before | after |
|---|---|---|
| 1e-3 | 0.2334 | 0.0143 |
| 1e-4 | 0.3064 | 0.00196 |
| 1e-5 | 0.3124 | 0.00172 |

Watertight throughout, and the residual at `1e-5` is 0.003% against 0.55%.

Two things had to be got right along the way. The chord has to be replaced *by
edge* rather than per triangle — ear clipping also emits zero-area slivers
between a chord and the boundary it lies on, and those name the same edge, so
replacing it on one side and not the other left exactly one edge open. And the
ring path has to *advance* along the chord: a sphere's ring turns round at a
pole, and points either side of the turn lie on the chord while running back the
way they came.

It still fails `a_sphere_clipping_a_corner`: 19.82 where an eighth of a sphere of
radius 3 is 14.137. One chord is replaced there, and it is not a rim —

```
chord sphere (3.142,-0.000)->(3.142,-1.520) path 29 of n=93
```

— a meridian at `u = π`, 87 degrees of a great circle. That is the case the
operation is wrong for. The strip between such a chord and the arc it shortcuts
is not empty: the zero-area slivers already cover it, degenerate in parameter
space but real in three dimensions. Fanning the *other* triangle over the same
path lays a second sheet across it, and the eighth-sphere gains 5.68 of volume
it does not have.

So the rule is not "a chord along the boundary should follow the boundary". It
is that the strip between them belongs to whatever already covers it, and the
question is which triangle that is. The slivers are the honest owners — they are
exactly that strip — and the fix is to retriangulate *them* onto the surface
rather than to re-cover the strip from the interior side.

Reverted. Green again, and the measurement stands: the chord is worth 0.55% of
volume on a trimmed cylinder and the way to it is now narrow.

Also worth recording, because it cost most of an iteration: `timeout` is not on
this machine. Three "hangs" were `timeout: command not found` returning 127, and
the run underneath finished in 0.09 seconds with a one-edge failure. Check the
exit code before believing a timeout.

Coverage **38 of 42**, unchanged.

## Shipped: the chord goes back onto the boundary, and the slivers go

Third variant, and this one is right.

A boundary-collinear chord is replaced by the ring's own vertices between its
ends. Nothing is invented — every point of the fan is a ring vertex the face
across the seam names too — so the seam cannot open.

The part that took three attempts is what happens to the strip between the flat
chord and the curved boundary it shortcuts. That strip is not empty. Ear
clipping leaves *slivers* along such a chord: triangles with no area in
parameter space, whose corners are all on the ring. They look like nothing and
they are not nothing — in three dimensions they are exactly that strip, and they
are what was holding it closed.

* Fan the interior triangle and keep them: the strip is covered twice. An eighth
  of a sphere came back 19.82 where it is 14.137.
* Fan and drop every flat triangle naming the chord: five tests open. Those
  slivers do not name the chord — they join the run's own points to each other.
* Fan and drop every flat triangle lying wholly *inside the run*: correct. That
  is the set that tiles the strip, and it is identified by its vertices, not by
  an edge.

| exact | tolerance | before | after |
|---|---|---|---|
| quarter cylinder 56.548668 | 1e-3 | 0.2334 | **0.014317** |
| | 1e-4 | 0.3064 | **0.001959** |
| | 1e-5 | 0.3124 | **0.001720** |
| eighth sphere 14.137167 | 1e-3 | — | 0.004175 |
| | 1e-4 | — | 0.002491 |

The error used to *grow* as the tolerance fell — a finer boundary makes a longer
fan — and now it shrinks. That is the property `a_trimmed_curved_face_holds_its_tolerance`
asserts: not a fixed bound, but that asking for less error gets less error, which
is the thing that was actually broken. It also holds the volume to a tenth of a
percent, where the existing checks allow one percent and could not see this at
all.

Two guards earned along the way and kept: the replacement is decided *by edge*,
because doing it per triangle leaves the chord on one side and replaces it on the
other — exactly one edge open. And the ring path must *advance* along the chord,
because a sphere's ring turns round at a pole and the points either side of the
turn lie on the chord while running back the way they came.

Full suite green, including every case the two wrong variants broke.

Coverage **38 of 42**.

## Touching told apart from crossing

The only pair still declining is the off-axis torus, and it declined as
`NotWatertight { open_edges: 34 }` — which reads as "the kernel built something
broken". It is not broken. A cylinder of radius 4 about `x = 1` is exactly 4
from its own axis at `(-3, 0, 0)`, which is exactly the inner equator of a torus
of `R = 4, r = 1`. The two surfaces *touch* there. The intersection pinches to a
point, two of its branches meet, and the tracer cannot follow that because
`settle` has no direction to step in when the normals are parallel.

Whether two surfaces touch or cross is `|n1 × n2|` along the curve — the sine of
the angle between the normals. Measured rather than guessed:

```
torus / cylinder, off-axis   1.828e-3
torus / cylinder, coaxial    1.000e0
sphere / cylinder            9.428e-1
plane / sphere               1.000e0
plane / cylinder             1.000e0
cylinder / cylinder          8.661e-1
```

Three orders of separation, so the threshold is not a delicate choice; `1e-2`
sits five times above the tangent case and eighty times below the nearest pair
that genuinely crosses. `Declined::TangentialContact { face_a, face_b }` now
names it.

This is a report, not a solution. Resolving such a pair needs the pinch as a
*node* of the curve network with four branches leaving it, which is the
intersection-graph design and not an iteration's work. But a decline that says
what it means is worth having on its own: `NotWatertight` was carrying two
meanings, "this is not supported" and "this came out wrong", and only the second
is a bug. That conflation has been on the gap list since it was written.

A side effect worth noting: `brep_boolean` went from 63 seconds to 15. The torus
had been tessellating a broken result three times over before the gate caught
it.

Coverage **38 of 42** — the count does not move, and should not. What moved is
that all four uncounted operations are now honest: one is a legitimately empty
intersection, and three say the surfaces touch.

## NURBS, which nothing had ever tried

`ssi` has no case for `Surface::Nurbs`, so such a pair falls through to
`Unknown` — and `Unknown` is not a decline. It is marched, by the same numeric
tracer a cross-drilled hole's quartic seam uses. So the NURBS path through the
boolean was live, and nothing had ever run it. An untested path that *produces*
something deserves more attention than one that refuses to.

Testing it needs a surface whose answer is known independently. A rational
quadratic arc of three control points weighted `1, √½, 1` is exactly a quarter
circle; extruded, exactly a quarter cylinder. Checked first, because otherwise
the rest proves nothing:

```
point(0,0)     = [3, 0, 0]          radius 3.000000
point(0.5,0.5) = [2.1213, 2.1213, 2] radius 3.000000
point(1,1)     = [0, 3, 4]          radius 3.000000
```

Then cut at `z = 2`, where the seam has to be an arc of radius 3:

```
march -> 1 curve
  201 pts  radius [2.999850, 3.000000]   z [2.000000, 2.000000]
```

One curve, on the plane to twelve places, on the patch to 1.5e-4 against a
tolerance of 1e-3. It works, and now there is a test saying so.

Also re-measured, since the chord fix landed: the residual on `cylinder − box`
fell from **−2.31 to −0.151**, a twentieth of what it was. It still grows as the
tolerance falls — 0.048, 0.138, 0.151 — so the pinned trim loop is still real,
still the wrong direction, and now worth 0.045% instead of 0.7%. It stays on the
list; it is no longer the thing most worth doing.

Coverage **38 of 42**.

## Two gap-list items, closed by measuring them

**Multi-shell results already work.** It was on the list as unimplemented and it
is not. The union of two solids that never meet is one body with two shells, and
a kernel assuming one shell either drops a piece or comes back open. Measured:

```
disjoint cubes    difference 8.0000   intersection 0   union 16.0000 (12 faces)
disjoint spheres  difference 4.1846   intersection 0   union  8.3692 (2 faces)
nested cubes      difference 208.0000 intersection 8   union 216.0000
```

All closed, all exact. `solids_that_do_not_meet_still_have_all_three_answers`
now says so, including that an empty intersection is an *answer* rather than a
failure. Nested solids too, which meet nowhere either and were also untested.

**A point intersection is a touch, and now says so.** Two spheres exactly a
diameter apart declined as `NeedsArrangement { face: 0 }`. Nothing is arranged —
they graze. `sphere_sphere` says exactly that:

```rust
if h2 <= EPS * EPS {
    return SsiResult::Curves(vec![Curve3d::Point(center)]);
}
```

Three of the `ssi` cases produce a `Curve3d::Point`, and every one of them is a
tangency: a sphere on a plane, two spheres, and the cylinder pair. So a point
that reaches both faces is now `TangentialContact` like any other touch, which
is what `Curve3d::Point` being "ignored" was really costing — not a wrong
answer, but a decline that pointed at the wrong thing.

A hair further apart and the two do not touch at all, which the test also
checks, because a tangency detector that fires on everything nearby would be
worse than none.

That leaves, of everything on the list: the tangency *solved* rather than
reported (the intersection-graph design), the pinned trim loop at 0.045%, and
cost — two tessellations per boolean, `O(n²)` face pairs, `march` seeding on a
13³ grid.

Coverage **38 of 42**.

## Where a boolean actually spends its time

Timed by phase rather than guessed at, on a plate with a bore through it:

```
                       1e-3        1e-4
ssi + trace              0ns     5.79ms
sample curves        0.45ms     0.55ms
classify meshes      9.72ms   183.96ms      <- 58% of the whole thing
weld + seam + wall   4.23ms    41.16ms
TOTAL               20.38ms   317.36ms
```

The classify meshes are `solid_triangles` on both inputs, and they exist for one
purpose: to answer *is this sample point inside that solid* by ray parity. A
sample is chosen away from the boundary on purpose, so what such a mesh needs is
to place the point, not to be accurate — and it was being built at the modelling
tolerance, which is why it grew without limit as the tolerance fell.

A floor of `span × 1e-4` is 21 times faster and **wrong**. A plate came back
from its first bore not a valid solid, and the second bore then declined
`NotASolid`. So the classification really is delicate: coarsen it too far and
pieces change sides. `span × 1e-5` holds:

| | classify meshes | whole boolean |
|---|---|---|
| 1e-3 | unchanged — the floor does not bind | unchanged |
| 1e-4 | 197.6ms → **104.6ms** | 328ms → **158ms** |
| 1e-5 | 647.1ms → **101.4ms** | 1168ms → **489ms** |

Relative to the model, not absolute, because a millimetre is coarse on a watch
and fine on a bridge. And it is an *attempt*, not a decision: a body that has
already been cut carries trim loops sampled at whatever tolerance cut them, so
asking for a coarser tessellation than that can come back open — in which case
the modelling tolerance still works and is used.

Nothing changes at `1e-3`, which is where every test runs, and that is the point:
the cost was in exactly the regime the cost mattered.

Coverage **38 of 42**.

## Two more of the cost, and one thing that cannot be cut

With the classification meshes coarsened, they were still two thirds of a
boolean. Timed inside them, on the two inputs at `4e-4`:

```
plate: refine_edges  48µs, tessellate   0.15ms,   12 tris
drill: refine_edges 250µs, tessellate  24.59ms, 1020 tris
```

Twenty-four microseconds a triangle for a cylinder is not a tessellation cost,
it is a search cost. Two of them:

**`near` was recomputed on every call.** `on_ring` decides how close to the ring
counts as on it by walking the whole ring — and that value does not depend on
the point being tested. It was being worked out again for every edge of every
triangle on every round of refinement. Hoisted: 24.6ms → 20.9ms.

**The chord search ran on planes.** A plane is exact under any triangulation, so
a chord across one is not wrong and there is nothing to put back — but the
search still walked the ring for every edge of every triangle, and a drill's end
caps carry the finest rings in the model. It was looking for a fault planes
cannot have. Guarded: 20.9ms → **14.1ms**.

End to end, best of three, against the 328ms and 1168ms measured before any of
this iteration's work:

```
tol 1e-3   11.7ms
tol 1e-4  107.4ms
tol 1e-5  290.9ms
```

Roughly three times at `1e-4` and four at `1e-5`.

**And one that cannot go.** `solid_triangles` clones the body and calls
`refine_edges` before tessellating, which looked like pure waste for a mesh that
only answers *is this point inside*. It is not waste. Without it the edges stay
at the resolution the body was built at while the fills go to tolerance, the two
disagree, and the mesh is not closed — nine tests' worth, immediately. Noted in
the code so it is not tried again.

Coverage **38 of 42**.

## What is left of the cost, and what holds it there

Re-profiled after the last two rounds, at `1e-4`:

```
ssi + trace            0ns
sample curves       0.79ms
classify meshes    61.77ms      <- still 60%
weld + seam + wall  7.34ms
split A             0.18ms
TOTAL             102.58ms
```

Still the classification meshes, and the floor is the reason: `span × 1e-5` on
this model is `1.1e-4`, which at a tolerance of `1e-4` is barely coarser than
the model. It only bites at `1e-5`, where it is worth six times.

Why it cannot go further, measured rather than assumed. At `span × 1e-4` the
plate's *first* bore comes back:

```
bore x=-14: 8 faces, valid=false defects=[EdgeFaceCount { edge: 12, faces: 3 }]
bore x=0:   NotASolid
```

An edge with three faces on it — a piece kept that should have been dropped. Not
an open mesh, not a fallback that failed to fire: a piece put on the wrong side.
So a sample point was sitting within about 4e-3 of its own piece's boundary, and
a mesh deviating further than that moves the boundary past it.

Which says exactly what the remaining 60% is waiting on, and it is not the mesh.
A piece's sample point is chosen by its margin in *parameter* space
(`deep_in_ring`, `ring_margin`), and parameter margin is not distance. On a face
whose parameterisation bunches — near a pole, near a seam, on a cone — a point
can be deep by that measure and a hair from the boundary in space. Choose the
sample by its distance in space instead and the classification mesh can be
coarse, because the thing it has to resolve is no longer small.

That is the next piece of work, and it buys correctness as well as speed: a
sample point far from its boundary is one that a coarse mesh, a near-tangent
surface, and a thin sliver all classify the same way.

Coverage **38 of 42**.

## Choosing the sample point instead of taking it

`sample_between` returned the *first* candidate that landed inside the piece.
That point decides which side of the other solid the piece falls on, by a ray
cast against a tessellation of it — so how far it sits from its own boundary is
exactly how much that tessellation is allowed to deviate. Taking the first one
gave points a few thousandths from the edge, and that is what capped the
classification meshes.

Depth in *parameters* is not distance. Near a pole, near a seam, on a cone, the
parameterisation bunches and a point deep by that measure is a hair from the
boundary in space. So it now measures the margin where it matters — in space,
against the loops' own points, subsampled to at most 256 — keeps the best
candidate rather than the first, and stops early once one is a tenth of the
piece from the edge.

The same coarsening that broke a plate's first bore last iteration —

```
bore x=-14: 8 faces, valid=false defects=[EdgeFaceCount { edge: 12, faces: 3 }]
bore x=0:   NotASolid
```

— now passes, along with everything else. So does one step coarser again,
`span × 1e-3`, but `span × 1e-4` is the one kept: a sample a tenth of its piece
from the edge against a mesh out by a thousandth of the model is a couple of
hundred times more margin than it needs, and slivers do not get to be the
exception.

| tolerance | before this iteration | after |
|---|---|---|
| 1e-3 | 11.7ms | 13.8ms |
| 1e-4 | 107.4ms | **33.9ms** |
| 1e-5 | 290.9ms | **230.8ms** |

Slightly slower at `1e-3`, where the floor does not bind and the margin search
is pure addition; three times faster at `1e-4`. Against where the cost work
started — 328ms and 1168ms — that is roughly ten times at `1e-4` and five at
`1e-5`.

Worth saying which of these is the real result. The speed is the visible half;
the other is that a sample point far from its boundary is one that a coarse
mesh, a near-tangent surface and a thin sliver all classify the same way. The
old code was one unlucky centroid away from a wrong answer at any tolerance.

Coverage **38 of 42**.

## A boolean result is not always a boolean input

Setting out to close a hole in the gate, and finding a larger one behind it.

The hole in the gate is real: `boolean` checks that its result is *closed*, and
closure cannot see a piece that was kept when it should have been dropped,
because a doubled face has no open edge. The plate that came back with
`EdgeFaceCount { edge: 12, faces: 3 }` two iterations ago was returned `Ok`; the
*next* boolean to touch it declined `NotASolid`, with nothing to say why.

The obvious fix — gate on `defects()` too — fails, and the way it fails is the
finding. Twelve tests break, and their results are watertight and their volumes
right. Narrowing to edges with *more* than two faces still breaks four. So
`defects()` is stricter than what this layer produces, routinely, on results
that are correct by every other measure.

Which raises the question the other way round, since `boolean` *begins* by
requiring `is_valid_solid` of its inputs. Can it eat its own output?

```
corner overlap     Difference: defects 6, valid=false, reused -> NotASolid
corner overlap     Union:      defects 0, valid=true,  reused -> ok
bore across a rod  Difference: defects 4, valid=false, reused -> NotASolid
bore across a rod  Union:      defects 4, valid=false, reused -> NotASolid
plate + drill      Difference: defects 0, valid=true,  reused -> ok
plate + drill      Union:      defects 0, valid=true,  reused -> ok
```

Sometimes. A plate with a bore can be bored again — which is why the example and
`bores_can_be_drilled_one_after_another` work, and why this had not been
noticed. A rod with a cross-hole cannot be touched again at all.

The cause is the result's *edge list*, not its geometry. A face's boundary comes
from its trim loops, so tessellation is closed and volumes are exact whatever
the edges say; but `edges` is left partial, with edges one face claims and edges
three do, and `is_valid_solid` reads that list. Nothing downstream of
tessellation notices. Every boolean does.

So the gate cannot be tightened until the edge list is built properly, and the
edge list is worth building properly for its own sake: a modelling kernel whose
output cannot be re-cut is half a kernel. That is the next work, and it is
bigger than a gate — every piece has to record which shared curve or carried
edge each stretch of its boundary came from, and the assembly has to resolve
those into edges naming exactly two faces.

Reverted, since a gate that rejects correct results is worse than one that
misses wrong ones. Nothing shipped this iteration; what was bought is knowing
that `bores_can_be_drilled_one_after_another` passing does not mean results
compose, and which cases it does not cover.

Coverage **38 of 42**.

## One edge per curve is wrong, and inferring the right one does not work

The composability hole is in `from_pieces`, and it is one line of intent:

```rust
let e = *edge_of_curve[id].get_or_insert_with(|| { ... });
```

One `Edge` per shared curve. But a curve can bound *several* pieces at once —
subdivide a face along it and every region touching it names it — so that edge
collects three faces, or one when the piece on the far side was dropped. Which
is exactly what `is_valid_solid` reads, and exactly why a rod with a cross-hole
cannot be cut again.

Each *stretch* of a curve should be its own edge: two pieces using the same
stretch share one, two using different stretches get one each. The attempt was
to recover the stretches from what is already there — a piece's loops carry the
curve's vertex indices, so the vertices of curve `id` appearing in piece `p`'s
loops are the stretch `p` runs along, and two pieces with the same set share an
edge.

256 open edges. The reason is worth keeping: **a carried loop's vertices are
`usize::MAX`.** They are cleared on purpose — they number the input body, not
the result — so for those pieces the stretch comes back empty, the face silently
loses that edge, and the seam it was holding opens. The information is not
merely awkward to recover; for the pieces that most need it, it is not there.

So the note from last iteration was right in a way I had not appreciated when
writing it: each piece has to *record* which curve or carried edge every stretch
of its boundary came from, at the moment the face is split, and the assembly has
to resolve those. It cannot be worked out afterwards. That means a provenance
field on `TrimLoop` — per point, which curve and which index along it — carried
through `split_planar`, `subdivide_face` and the seam machinery, all of which
already thread the ordering (`Chord`'s third element is exactly this, for
chords). Extending it to every boundary point is the shape of the work.

Reverted. Two attempts at this now: gating on `defects()` rejects correct
results, and inferring stretches loses the carried ones. What is bought is
knowing that the fix is a data-model change and not a patch.

Coverage **38 of 42**.

## Correcting last iteration: the provenance is there

I wrote that a piece's boundary provenance "is not merely awkward to recover;
for the pieces that most need it, it is not there". That is wrong, and the way
it is wrong matters, so: measured, per piece, on the rod with a cross-hole —

```
piece 1 cylinder: loops [34, 59, 60], bounding hits [(0, 60/60), (1, 61/61)]
piece 4 cylinder: loops [121],        bounding hits []
```

Piece 4's single 121-point loop *is* the two curves, 60 and 61 points, joined.
Its loop vertices resolve fine. What is empty is its `bounding` list — and it is
empty on purpose:

> The seam is shared through the loop's vertex indices rather than through an
> `Edge`: a chord cut at a crossing bounds only part of each piece, so one edge
> per curve would not describe it.

True of the tessellation and false of the topology. The loops do hold the seam
shut, so the mesh closes and the volume is right; but the curve is then claimed
by one face instead of two, `is_valid_solid` reads exactly that, and the result
cannot be cut again.

And `planar::Source::Chord { chord, point }` names the curve *and* the index
along it for every boundary point the subdivision produced. The provenance was
there the whole time, in the sources, one field away from the `vertices` walk
that already reads them.

Filling `bounding` from those sources: defects on the rod fall from 4 to 3, the
two curve edges gain their second face, and everything still passes.

One of the remaining three is new to look at and not new to exist —
`EdgeOffSurface { edge: 1, deviation_scaled: 1 }`. An edge with only one face
has `surfaces.0 == surfaces.1`, so that check was passing trivially; giving it
its real second surface reveals a marched curve sitting a hair over tolerance
from one of them. Exposed, not caused.

Still not composable. Corner overlap is unchanged at 6 defects, and the rod is
not valid. The comment's objection is the reason and it stands: one edge per
curve is too coarse when a chord is cut at a crossing and each piece runs along
only part of it. The edges want splitting per *stretch* — which those same
sources describe, since they carry the index along the curve as well as its id.

What changed this iteration is that the next step is now a walk over data that
exists rather than a data-model change. Kept, because a piece that names the
curves it runs along is more nearly right than one that names none.

Coverage **38 of 42**.

## Per-stretch edges, built and measured against per-curve

Built it: `Piece::bounding` carrying `(curve, Option<stretch>)`, the stretch
being indices into that curve's own vertex list, taken from
`planar::Source::Chord { chord, point }` through the chord's ordering; `None`
meaning the whole curve, for the paths that bound a piece with a closed ring.
One edge per distinct `(curve, stretch)` in `from_pieces`.

Everything passes. And it is not better:

| case | one edge per curve | one edge per stretch |
|---|---|---|
| bore across a rod | **3** defects | 6 |
| column in ball | 20 / 24 | **12 / 16** |
| corner overlap | 6 / 0 | 6 / 0 |
| plate + drill | 0 / 0 | 0 / 0 |
| sphere + drill | 0 / 0 | 0 / 0 |

Splitting per stretch helps exactly where the reasoning said it would — a column
bored through a ball, where each loop is four arcs and a curve bounds many
regions, drops by a third. It hurts the rod, and for a reason worth stating: the
two sides of that seam derive their stretch by *different routes*. The face
split by `split_planar` bounds itself with a whole ring and says `None`; the face
split by `subdivide_face` lists the indices its sources recorded. When those
disagree by a single index the two no longer share a key, and one edge with two
faces becomes two edges with one face each.

So the derivation is the problem, not the idea. Both sides have to arrive at the
same description of the same stretch, and they cannot, because they come through
different code paths with different information — one has a ring, the other has
sources.

Which suggests the fix is not to derive the edges better but to *reconcile* them
afterwards: build them per curve as now, then walk the finished list and split
any edge more than two faces claim into the stretches its claimants actually
run along, merging those that coincide. A post-pass answers to what the faces
say rather than to how they were made, and that is the only thing both sides
share.

Reverted to per-curve, which is where the fewest cases are wrong today. Four
attempts on this now, and each has narrowed it: gating on `defects()` rejects
correct results; inferring stretches from loops loses the carried ones;
recording them per piece works but cannot be made consistent across split paths.
The next one is a post-pass, and it is the first that does not depend on the two
sides agreeing in advance.

Coverage **38 of 42**.

## The post-pass, and what it found by not working

Built the reconciling post-pass — split an edge more than two faces claim into
the stretches its claimants run along. It changed nothing, because I had been
building for a case that does not occur. Counted:

```
rod:    edge 4: 1 face, 18 verts;  edge 5: 1 face, 18 verts    {1: 2, 2: 4}
corner: six edges, 1 face, 2 verts each                        {1: 6, 2: 15}
```

**No edge has three faces.** Not one, in any case measured. Every edge this
assembly gets wrong has exactly *one* claimant, and every one is carried over
from an input body. Four iterations of reasoning about curves cut at crossings
and stretches bounding part of a piece, and the thing actually broken is a rim
with nothing on its far side.

Which makes sense once seen. Such a rim lies between two faces of one input
body; the operation keeps one of them, so the face now across it is a piece of
the *other* solid, and that piece has no reason to name an edge it never had.

So the post-pass was rewritten for the case that happens: find the partner by
asking which other face's loops name every vertex of the edge, and give it the
claim. That changed nothing either — and the reason is the real finding. **No
face names those vertices.** The piece across such a rim describes it with its
*own* points. The two coincide in space, which is why the mesh closes and the
volume is right, and they are different vertices, which is why the topology has
two descriptions of one rim.

That cannot be repaired by a post-pass over the edge list, because the thing
that is missing is not an edge — it is the agreement. The two sides have to
share the vertices along a carried rim the way they already share them along an
intersection curve, and `SharedCurve` is precisely the mechanism: sample once,
reference from both. A carried rim that becomes a boundary between the two
solids wants the same treatment, and does not get it because it was never an
intersection.

Both post-passes reverted; neither changed a number. Five attempts now, and the
question has moved a long way: from "tighten the gate" to "build the edges per
stretch" to "the two sides never shared the vertices in the first place". The
last one is upstream of all the others and is where the next attempt goes.

Coverage **38 of 42**.

## A piece was carrying rims it never reached

Six iterations of theory about stretches, and the answer was in a dump of the
faces:

```
face 3 cylinder loops=Some([121]) edges=[0, 1, 4, 5]
edge 4: claims [3], 18 verts, closed
edge 5: claims [3], 18 verts, closed
```

Edges 4 and 5 are the *drill's own end rims*, five units outside the rod it is
boring. The sliver of drill wall that survives inside the rod does not go
anywhere near them, and it was carrying them anyway — `subdivide_face` handed
every piece `carried.to_vec()`, the whole of its face's edge list. Each became
an edge with one face and no possible partner, because there is nothing on the
far side of a rim the result does not reach.

Not stretches. Not provenance. A filter that was never written.

The filter is: keep a rim any point of which lies on this piece's own loops.
*Any*, not every — a rim a piece runs along only part of is still a rim it runs
along, and dropping it takes the connection between two faces with it. Requiring
every point drops those, defects fall further, and a shell then reads as open:
corner overlap went to zero defects and stopped being a valid solid.

| | before | after |
|---|---|---|
| corner overlap | 6 / 0 | **0 / 0** |
| bore across a rod | 3 / 3 | **1 / 1** |
| column in ball | 20 / 24 | **12 / 16** |
| round an edge | 4 / 2 | **2 / 2** |
| plate + drill | 0 / 0 | 0 / 0 |
| sphere + drill | 0 / 0 | 0 / 0 |

Strictly down or level everywhere, no case made worse, and the whole suite
green.

Still five of twelve valid, so this is not the end of it — corner overlap's
difference now has *no* defects and is still not a valid solid, which means its
shells do not close, and that is the next thing to look at. But the edge lists
are a great deal closer to true than they were, and the remaining wrongness has
somewhere new to be.

Worth recording as method, too. The last five attempts all reasoned forward from
a hypothesis about what must be wrong. This one printed the faces and read what
was there.

Coverage **38 of 42**.

## A chord is not a ring

Following the shells. Corner overlap's difference has no defects and is still
not a valid solid, and the dump says why:

```
shell 0: 6 faces [0, 2, 3, 4, 5, 1], closed=true
shell 1: 3 faces [6, 7, 8],          closed=false
  face 6 plane loops=Some([4]) edges=[12, 13] free_ends=2
```

Two shells. Faces 6–8 are the three walls of the notch, and they share edges
only with *each other* — the notch's inner corner. Where they meet the cube's
faces there is no edge on either side, so the two groups never connect.

Attributing a region's `Boundary` points to the curve they lie on — the
subdivision only marks a point as `Chord` when the cut crosses the face, not
when it runs along its own outline — joins them: one shell, closed, and defects
across the whole set fall from 34 to 8. It also costs corner overlap's *union*
its validity, because a stretch then gets covered twice, once by a carried edge
and once by the new curve edge, and `chain_face_loop` walks every segment of
every edge of a face into one ring and cannot when a segment is doubled. Held
back for that; it is right in substance and needs the duplicate resolved.

What did ship is one line found on the way:

```rust
vertices: shared[id].vertices.clone(),
closed: true,
```

Every edge made from an intersection curve was declared closed. A chord is not a
ring. It went unnoticed because closed curves were most of what became edges,
and the moment more of them did, six `EdgeNotClosed` appeared and pointed at it.

Reporting it as the curve actually is:

| | defects before | after |
|---|---|---|
| corner overlap | 0 / 0 | 0 / 0 |
| bore across a rod | 1 / 1 | 1 / 1 |
| column in ball | 12 / 16 | **0 / 4** |
| round an edge | 2 / 2 | **0 / 0** |
| plate + drill, sphere + drill | 0 | 0 |
| **total** | **34** | **6** |

Five of twelve still valid — unchanged, no case worse — and the defect count is
down by a factor of six from one line. The remaining six are the rod's pair and
four on a column bored through a ball.

Coverage **38 of 42**.

## The attribution is right and the checker will not have it

Tried the exclusion the last iteration called for: attribute a region's
`Boundary` points to the curve they lie on, *unless* a carried rim already
covers that stretch, so nothing is described twice. It made no difference —
still four of twelve valid against five without it — so the duplicate was not
carried-against-curve after all.

What the attribution does achieve is worth setting down, because it is most of
the structure:

```
shell 0: [0, 9, 11, 2, 3, 4, 5, 1, 7, 8, 10, 6] closed=false
  face 2 edges=[10, 11, 2, 6, 12, 13] free_ends=2
  face 9 edges=[0, 17, 19, 23, 28, 29] free_ends=2
  (every other face 0, and no edge with a count other than 2)
```

One shell instead of two. Every edge claimed by exactly two faces, in every case
measured. The notch that used to float free is joined to the solid it was cut
from. What is left is two faces out of twelve whose edges `chain_face_loop` will
not walk into a ring — and faces 4, 7 and 11 have six edges each and chain
perfectly, so it is not the count.

So the data is right and the check disagrees with it, which is a different
problem from the one this started as. `face_free_ends` asks whether a plane's
edges chain into a loop, and `chain_face_loop` walks every segment of every edge
from the first one and takes whatever it can reach; a spur, a doubled segment or
a second ring stops it, and it cannot say which. That is the thing to look at
next, and it is a checker, not a kernel.

Held back again, for the same reason as last time: one case loses `valid` and no
case gains it, and validity is what decides whether a result can be cut again.
Two iterations now where the attribution is clearly the right shape and the
measurement will not endorse it. It stays out until the check it fails is
understood well enough to say whether the check or the data is wrong — and the
evidence points at the check.

Coverage **38 of 42**, 6 defects across the twelve results, 5 of them valid.

## The check was worse and the data was wrong too

Last iteration ended "the evidence points at the check". It did, and the check
was worse than it needed to be — but replacing it says the data is wrong as
well, which is the opposite of what I expected and the useful part.

`face_free_ends` asked, for a plane, whether the face's edges *chain* into a
loop: walk greedily from the first segment and take whatever can be reached.
That answers a harder question than the one being asked, and badly — a spur, a
second ring, or three segments meeting at a point all stop the walk, and it
cannot say which it hit.

A free end has a definition that needs no walking. Every vertex of a closed
boundary is an end of an even number of its segments; an open polyline has
exactly two that are not. Degree does not care how many rings there are, what
order they come in, or where they touch. (An absent boundary is still not a
closed one: a face with no edges at all — an open tube — keeps its two.)

On today's results the new check agrees with the old one exactly: five of twelve
valid, six defects, whole suite green. Which is the right outcome for a
definition swap and not the reason for making it.

The reason was to unblock attributing a region's `Boundary` points to the curve
they lie on. With the better check, that attribution *still* fails — the same
two faces, now for a reason the check can state. A curve edge whose end falls
partway along a carried rim leaves that vertex with three segments meeting at
it. A T-junction. The boundary really is not closed there, the old check was
right to refuse it, and it simply had no way to say so.

So the attribution is not merely right-in-substance-and-blocked: it needs the
carried rim *split* at the point the curve leaves it. That is the same
per-stretch question as before, arriving from the other side, and it is now
stated in terms of a specific vertex rather than a general worry.

Shipped: the degree check, and `chain_face_loop` deleted with it — nothing else
used it.

Coverage **38 of 42**, 6 defects across the twelve results, 5 valid.

## Three ways to attribute a boundary, all worse

The T-junction suggested over-attribution: `curve_at` gives a point to a curve
when it is merely within reach of one of its vertices, so a face that *touches*
a curve at a single point claims the whole of it, and the curve then ends
partway along one of that face's rims. Requiring a run instead — two consecutive
boundary points on the same curve before it counts — is the obvious correction.

It is worse. Twelve defects against eight for plain attribution and six for
none, and four of twelve valid against five. So a single-point touch is not what
was producing the T-junction either.

The attempts, all measured on the same twelve results:

| | defects | valid |
|---|---|---|
| no attribution (shipped) | **6** | **5** |
| attribute every boundary point | 8 | 4 |
| ...unless a carried rim covers it | 8 | 4 |
| ...only along a run of two or more | 12 | 4 |

Every one of them turns two shells into one and gives every edge exactly two
faces, which is why they keep looking right. Every one of them also leaves a
vertex where three segments meet, and the degree check — which is a definition,
not a heuristic — says that is not a closed boundary, correctly.

What that means is that the seam these faces share is genuinely described
differently by the two sides, and no rule for *labelling* the existing points
can fix it, because the points themselves do not line up. Attributing more
carefully, less carefully, or with exclusions all fail the same way. The
carried rim has to be cut where the curve meets it, so that both sides have a
vertex there and the boundary turns instead of branching.

That is the fourth time this has come back to splitting a rim, from four
different directions: per-stretch edges at assembly, inferred stretches from
loops, recorded stretches per piece, and now attribution. It is the work.

Reverted to the shipped state. Coverage **38 of 42**, 6 defects across the
twelve results, 5 of them valid.

## The rim is cut now, and it is still not enough

Four directions had converged on the same thing: a rim carried from an input
body is a polyline of that body's own points, and a curve running onto it stops
at a place that polyline has no vertex for. So put one there. Only a curve's
*ends* — its interior lies across the face, not along the rim — and only when
the point is genuinely on the rim rather than near it, measured as the detour it
adds between the two neighbours it falls between.

It ships, and on its own it is nearly invisible: the same five of twelve valid
and the same six defects. What it does move is the thing it was aimed at — a
plate with a bore, cut again, goes from eight open edges to **four**.

Then the test it was built for. Attributing a region's `Boundary` points to the
curve they lie on, *with* the rims now cut: four of twelve valid and eight
defects, exactly as without the cut. The fifth attempt at that attribution and
the fifth to come out worse.

| | defects | valid |
|---|---|---|
| shipped | **6** | **5** |
| + attribute boundary points | 8 | 4 |
| + …unless a carried rim covers it | 8 | 4 |
| + …only along a run of two or more | 12 | 4 |
| + …with the rims cut first | 8 | 4 |

The rim split was the last explanation I had for why attribution leaves a
T-junction, and it was wrong — or at least insufficient, since cutting the rims
changes nothing about the attributed result. So the vertex where three segments
meet is not the curve's end landing mid-rim. It is something else, and I no
longer have a candidate; the next iteration should find that vertex and print
what actually meets there rather than reason about what might.

Coverage **38 of 42**, 6 defects across the twelve results, 5 valid, and a bore
that can be re-cut with half the gap it had.

## Printed the vertex, and it says the attribution is simply wrong

Stopped reasoning about the T-junction and printed it. Every odd vertex in the
attributed corner-overlap union, and every one is the same shape:

```
vertex 0 at (2.000,0.000,2.000) degree 3, on edges [0, 4]
   edge 0: 2 verts, this is index 0     <- an endpoint
   edge 4: 3 verts, this is index 1     <- the middle
```

A two-point rim *ends* on the *middle* of a three-point curve. So it was the
reverse of the guess two iterations ago: the rim has its vertex, the curve runs
straight past a corner the face turns at. Which suggested splitting the curve
there — and both faces would get the same cut, since the point is already one of
its vertices.

Built it. The split lands, six new edges appear, and the free ends do not move:
twelve before and twelve after. Of course they do not. Cutting the curve at that
vertex makes it the end of *two* curve pieces — degree two — and the rim's
endpoint is still there. Two plus one is three. **A degree-three vertex cannot be
split into an even one.** Three edges genuinely meet at that point.

Which settles what six attempts could not. Three edges meeting on the boundary
of one face means one of them is not on that boundary: the face's edge already
describes that stretch, and the curve attributed to it is a *second* description
of the same locus. Not a missing cut, not a mislabelled point — a duplicate.
Excluding carried-covered points was the right instinct two iterations ago and
failed only because it compared against the rim's *points*, which do not
coincide with the curve's.

So attributing boundary points to curves is the wrong tool for joining the notch
to the solid. The notch and the cube already share that locus; what they do not
share is an edge *object*, and the way to give them one is to make the rim and
the curve the same edge — recognise at assembly that a carried rim and an
intersection curve lie on each other, and keep one.

Six attempts, all reverted, and the question is now specific enough to be worth
one more: not "which curve bounds this piece" but "which of these two edges are
the same edge".

Coverage **38 of 42**, 6 defects across the twelve results, 5 valid.

## The merge finds nothing, which completes the picture

Merging edges that lie on each other — keep the finer, let both faces name it —
finds no pair at all. Corner overlap comes back as two shells exactly as before.

Which is the last piece. Without attribution those faces have no curve edge to
merge *with*: the two descriptions are not two edges, one of them is simply
absent. With attribution the curve edge appears, and the degree-three vertex
from last iteration says what it looks like — a rim's endpoint landing on the
curve's *middle*. So the curve is not a duplicate of one rim. It is **longer**
than a rim: one curve runs along two of them, and the corner where they meet
falls inside it.

That accounts for every measurement taken over the last seven attempts, and it
says the fix is all three of the things tried separately, in order:

1. Attribute the boundary points, so the curve edge exists.
2. Split it at the vertices where the face's rims end — done last iteration, and
   on its own it leaves the degree at three, because the split point then has a
   rim end and two curve ends at it.
3. **Merge** each curve piece with the rim it now coincides with. That is what
   takes the degree from three to two: one edge for the stretch, not two, so the
   vertex has one end of each of its two neighbouring edges and nothing else.

Step three is why the first two kept failing individually — each was a third of
the answer, and a third of this answer is worse than none. Individually
measured: attribution alone 8 defects and four valid, split alone no change,
merge alone no change; shipped state 6 defects and five valid.

Seven attempts, all reverted. What is worth having is that the hypothesis now
explains the failures rather than just proposing something else: a rim ends
inside a curve because the curve spans two rims, and nothing that treats one
rim, one curve, or one label at a time can close that.

Coverage **38 of 42**, 6 defects across the twelve results, 5 valid.

## All three together, and a fourth thing underneath

Ran attribute-split-merge as one change, which the last iteration argued was the
whole answer. It is not, and the vertex says why:

```
v0 (2.00,0.00,2.00) deg 3: (edge,len,idx) [(0, 2, 0), (4, 2, 1), (31, 2, 0)]
```

Three two-point edges, all meeting there as *endpoints*. Not a curve passing
through a rim's end any more — that part the split did fix. Three distinct
stretches, which is also why the merge found nothing to merge: they are not
duplicates of each other.

Counting says what the third one is. Face 0 is the top of the first cube, an
L-shape where the second cube takes a bite out of it, and an L has six corners
and six sides. The face is carrying **eight** edges. At the L's inner corner
only two of the three belong; the third runs off into the square the other cube
removed.

So the split does its job and then keeps both halves. One of them is on the
face's boundary and one is in the part that is gone, and nothing tells them
apart — the same fault as the drill's end rims three iterations ago, which was
fixed by filtering a piece's carried edges to the ones its loops actually run
along. Split pieces need exactly that filter and do not have it.

That is a small, specific thing, which is a better place to be than the last six
iterations. It is also the fourth distinct rule this one problem has needed:
attribute the curve, cut it where rims end, merge each piece with its rim, and
drop the pieces that are not on the boundary. Each was invisible until the one
before it was in place.

Reverted, all four attempts this iteration. Coverage **38 of 42**, 6 defects
across the twelve results, 5 valid.

## Eight attempts, and the shape of the answer is not a post-pass

All four rules together — attribute, split, drop the pieces off the boundary,
merge each survivor with its rim — comes to 36 defects and four of twelve valid,
against six and five for doing none of it. The drop is what turns it: taking an
edge off a face leaves the edge with one face, which is the defect it was
supposed to remove.

The whole sequence, measured on the same twelve results:

| | defects | valid |
|---|---|---|
| shipped | **6** | **5** |
| attribute | 8 | 4 |
| attribute, excluding carried-covered points | 8 | 4 |
| attribute only along a run | 12 | 4 |
| attribute, with rims pre-cut | 8 | 4 |
| split edges where rims end | 16 | 4 |
| merge edges on one locus | 6 | 5 (no effect) |
| all four | 36 | 4 |

Each rule was found by measurement and each was right about the thing it looked
at. None of them helps, and the combination is the worst of all. That is not bad
luck with the details; it is the method being wrong. Every one of these is an
attempt to *reconstruct* topology after the fact from geometry — which edge lies
on which, which piece is on the boundary, which curve bounds which face — and
every reconstruction needs a tolerance, and every tolerance is right for some
faces and wrong for others. The failures do not accumulate into a fix; they
accumulate into a pile of thresholds.

The topology is known at the moment a face is split. `subdivide_face` has the
regions, their sources, which chord each boundary point came from and where
along it. That is the whole answer, in hand, and it is discarded — `Piece`
records a list of curve ids and nothing about which stretch, and everything
since has been an attempt to get that back out of the coordinates.

So the next thing here is not another post-pass. It is `Piece` carrying its
boundary's provenance per point — the same information `Chord` already carries
for chords — and `from_pieces` building edges from that rather than inferring
them. That was tried once, five iterations ago, and failed because only *one* of
the two split paths recorded it; the answer is to make the other path record it
too, not to infer it for that path.

Reverted, all four. Coverage **38 of 42**, 6 defects across the twelve results,
5 valid, and the boolean's own promise — closed or declined — intact throughout.

## A defect I had said did not occur

Stopped attacking the same face and looked at what the remaining six defects
actually are. Two of them are a new kind:

```
column in ball U: defects [EdgeFaceCount { edge: 12, faces: 4 }, ... × 4]
   face 1 plane edges=[1, 12, 13, 14]
   face 2 plane edges=[0, 2, 12, 13, 15]
   face 5 plane edges=[6, 8, 12, 16, 20]
   face 6 plane edges=[7, 12, 16, 21]
```

An edge with **four** faces — which four iterations ago I recorded as never
happening, having only measured the corner overlap and the rod. Edge 12 is the
column's vertical corner rim, and the sphere cuts each side of the column into
an upper piece and a lower one. Both pieces carry the whole rim; two sides
times two pieces is four.

The rim is not one edge there. It is two, one per piece, cut where each piece's
own boundary leaves it. Trimming each piece's carried rims to the part it runs
along, and keying the resulting edge by *which part* rather than by which rim:

| | defects | valid |
|---|---|---|
| shipped | 6 | **5** |
| trimmed | **2** | 4 |

Column in ball goes from four defects to none, and every other case holds — but
a corner overlap's union loses its shell closure, and neither state dominates
the other. Two more variants of the trim, against the loop's segments rather
than its points, and keeping the longest contiguous run rather than every point
that passes, land on exactly the same numbers, so it is not the predicate.

Reverted the trim. Kept the keying, which is inert today — with whole rims
carried, every piece produces the same key — but says what the key means, and
the case it is for is now measured rather than assumed.

That is the second time in this problem that a rule was right for the case it
was found on and wrong for the case next to it. The first was `.any()` against
`.all()` on the same filter, three iterations ago, and the two failures are the
same shape: one piece needs the rim whole to stay attached, the other needs it
cut to stay manifold, and a piece cannot tell which it is from its own loops.

Coverage **38 of 42**, 6 defects across the twelve results, 5 valid.

## Cut the rim only when someone else claims it

Last iteration ended: "one piece needs the rim whole to stay attached, the other
needs it cut to stay manifold, and a piece cannot tell which it is from its own
loops." That is true of a piece and false of the assembly, which has all of them
— and there the question is simply *does anyone else claim this rim*.

So keep every rim whole, as before, and cut the ones that turn out to need it:
an edge more than two faces claim, split by the stretch each claimant runs
along, and only when it comes apart cleanly — every stretch claimed by exactly
two faces, more than one stretch to find, otherwise left exactly as it was.

| | defects | valid |
|---|---|---|
| before | 6 | 5 |
| after | **2** | **5** |

Column in ball goes from four defects to none, both ways round. Corner overlap
keeps the shell closure that trimming unconditionally took from it. Nothing else
moves, and the whole suite is green.

Ten attempts at this problem, nine of them reverted, and what separates this one
is not a better predicate — the run test is the same one that failed last
iteration, character for character. It is *when* it is asked. Trimming at the
piece level asks a question a piece cannot answer and has to guess; asking after
assembly turns the guess into a lookup.

The two defects left are both on a rod with a cross-hole: one `EdgeOffSurface`,
a marched curve sitting a hair over tolerance from one of its two surfaces, and
its twin. Those are accuracy, not topology.

Coverage **38 of 42**, 2 defects across the twelve results, 5 valid.

## The last two defects want a solve, not an iteration

Both remaining defects are `EdgeOffSurface`, and the cause is one line in
`insert_seam_vertex`:

```rust
let mid = v3::add(pa, v3::scale(v3::sub(pb, pa), t));
vertices.push(mid);
```

The bisection above it finds *where along the step* the seam is, carefully,
because interpolating the parameter lands somewhere else. The point it then
stores is on the straight line between two samples of a traced curve — off both
surfaces by that step's sagitta, a thousandth on a rod with a cross-hole.

Three ways to move it, each measured on the same twelve results:

| | defects | notes |
|---|---|---|
| left on the chord (shipped) | 2 | the rod, off both by a sagitta |
| evaluated on this surface | 4 | rod fixed, column now 2× off the plane |
| alternating onto both, ending on the seam | 4 | column unchanged |
| alternating, not ending on the seam | **0** | and the rod stops resolving |

The last row is the interesting one. Every defect goes, and `a_bore_across_a_rod`
declines `NeedsArrangement` — because the whole purpose of this vertex is that
its `u` *is* the seam, and ending the alternation on the other surface gives that
up. Put it back and the column is where it started.

All three constraints are satisfiable together: on this surface, on the surface
across the curve, and at `u = origin`. That is three equations in three unknowns
with a solution the alternation is not finding — it converges to whichever pair
of them the last step enforced. It wants a Newton step on all three at once, not
projection back and forth.

Which is a small, well-posed piece of numerical work, and a much better place to
be than "an edge is off its surface".

Reverted. Coverage **38 of 42**, 2 defects across the twelve results, 5 valid.

## Not three unknowns, one

The last two defects, gone, and the reason the four attempts before this failed
is that they were all solving the wrong problem.

`insert_seam_vertex` bisects to find *where along a step* the curve crosses the
seam, and then stores a point on the straight line between the two samples —
off both surfaces by that step's sagitta. Moving it onto one surface puts it off
the other; alternating between the two lands on whichever the last step
enforced. It looked like three constraints in three unknowns wanting a Newton
step.

It is one constraint in one unknown. The vertex is on this surface's seam *by
definition*, and that seam is the curve `surface.point(origin, v)` — a single
parameter. Where it meets the surface across the curve is a root along `v`,
bracketed by the two samples the crossing lies between, and bisection on the
signed offset to that other surface finds it to fifteen places.

```
corner overlap     Difference / Union: defects []
bore across a rod  Difference / Union: defects []
plate + drill      Difference / Union: defects []
sphere + drill     Difference / Union: defects []
column in ball     Difference / Union: defects []
round an edge      Difference / Union: defects []
valid 5 of 12, 0 defects total
```

Every result of every operation measured, and not one topological or geometric
defect between them. The five that are not yet valid solids are shells that do
not close, which is a different question and the one left.

Worth keeping: "it wants solving, not iterating" was the right instinct and the
wrong size. The fix was not a bigger solver but a smaller problem — noticing
that one of the three constraints holds by construction and the other two meet
on a curve.

Coverage **38 of 42**, **0 defects** across the twelve results, 5 valid.

## What the unclosed shells are missing

With no defects left, everything that is not a valid solid is a shell that does
not close. Measured, per face:

```
corner overlap D: shells [(6, true), (3, false)]
   face 6 plane loops=Some([4]) edges=[12, 13] free=2
      v21 (0.00,0.00,4.00) deg 1: [(12, 3, 2)]     <- an edge stops here
      v22 (0.00,4.00,0.00) deg 1: [(13, 3, 2)]

round an edge D: shells [(7, false)]
   face 6 cylinder loops=Some([64]) edges=[7, 10] free=4
      v33 (2.00,4.00,2.00) deg 1: [(7, 32, 31)]
```

The same thing every time: a face of the *second* solid — a notch wall, a
fillet — carries only its inner edges, and the stretch where it meets the first
solid has no edge at all. The edges it does have simply stop, degree one, at the
point where that stretch ought to begin.

Which is the boundary attribution again, so I retried it — the conditions have
changed since it last failed: seam vertices are now exact on both surfaces, and
a rim more than two faces claim gets cut. It does what it is for, and joins the
notch to the solid: corner overlap goes from two shells to one. It also takes
the result from 0 defects and 5 valid to 2 and 4, exactly as before.

So: attributing the boundary attaches the shell and leaves it open, which is not
progress in itself but is a sharper statement than "two shells". The face knows
its boundary — its loops are right, and the tessellation closes on them. What it
does not have is an *edge* for the part of that boundary it shares with the other
solid, and the attempts to hand it one keep handing it the wrong shape: a whole
curve where it wants a stretch.

Reverted, ninth time. The state itself is the best it has been — no defects
anywhere, and the shells are the single remaining question.

Coverage **38 of 42**, 0 defects across the twelve results, 5 valid.

## The metric was lying

Attributing a *stretch* rather than a whole curve — derived, for every face
alike, as "which of this curve's vertices do your own loops name" — joins the
faces that share a seam. Corner overlap goes from two shells to one. And it
takes the count from five of twelve valid to four, which by nine iterations of
precedent means revert.

Except that this time the count is wrong. What the five contained:

```
union: 12 faces, valid true
  shell 0: 6 faces, closed=true, euler=2
  shell 1: 6 faces, closed=true, euler=2
```

The union of two overlapping cubes, reported as **two closed boxes**. Each group
carries its own faces, trimmed correctly, and the two never meet because the
seam between them has no edge — so each reads as a complete solid and
`is_valid_solid` says yes. It is not a solid, it is two, and the kernel cannot
tell.

That is worse than failing. A result like it is accepted as the *input* to
another boolean, which is exactly the thing validity is checked for. With the
stretch edges the same case comes back as one shell that does not close, and the
next boolean declines `NotASolid` rather than building on a fiction.

So: shipped, and the headline number goes down.

| | defects | valid | corner overlap union |
|---|---|---|---|
| before | 0 | 5 | two closed shells, reused as input |
| after | 0 | **4** | one open shell, declined |

Which is a reminder worth writing down. "Five of twelve valid" has been the
yardstick for six iterations, and part of what it was counting was the kernel
failing to notice that two things were separate. A measure that can be satisfied
by *missing* a connection will be, eventually, by anything optimised against it.

Coverage **38 of 42**, 0 defects across the twelve results, 4 valid and all four
of them true.

## A shell that closes, and the price of it

With the stretch edges in, every edge of the corner overlap is shared by exactly
two faces — thirty of thirty. What keeps the shell open is uniform:

```
face 0 edges=[0, 1, 2, 3, 24, 27] free=2
   v0 (2.00,0.00,2.00) deg 3: [(2, 3, 1), (24, 2, 0)]
```

Edge 2 is the cube's own rim, running the whole side of the face it bounded, and
it *passes through* the vertex where the seam's edge stops. The operation kept
half of that side; the boundary turns at the middle and the rim does not.

Cutting the rim there is not enough on its own — the two halves both stay on the
face and the vertex gets two of their ends plus the seam's. Each half has to go
to whichever faces run along it, which after the cut is a question about its two
ends and the loops that name them.

Done, and it works:

```
corner overlap Difference: shells [true]
corner overlap Union:      shells [true]
```

The first closed shell this problem has produced. And the reassignment costs the
edge counts: the whole set goes from 0 defects to 32, with a column bored
through a ball at twelve. Handing a half to every face whose loops name both its
ends gives some edges one face and some three, which is the invariant the
stretch pass had just established.

So closure is reachable and the two halves of the promise are still fighting:
this reassignment closes shells and breaks sharing, the stretch pass fixed
sharing and left them open. What is needed is a reassignment that preserves
exactly-two by construction rather than by hoping — which is the same shape as
the rim cut that *did* work two iterations ago, where the rule was "only when it
comes apart cleanly, otherwise leave it alone".

Reverted. Coverage **38 of 42**, 0 defects across the twelve results, 4 valid.

## Two takers, or none

The conservative cut — split a rim where the boundary turns, give each half to
the faces whose loops name both its ends, and only when every half finds exactly
two — blocked every case. Which is the wrong answer, and the counts say why:

```
half of edge 0: 2 verts, 2 takers [0, 2]
half of edge 0: 2 verts, 0 takers []
half of edge 2: 2 verts, 2 takers [0, 4]
half of edge 2: 2 verts, 0 takers []
```

Every rim splits into a half with two takers and a half with **none**. The
second is the part of the rim the operation took away — interior to the result,
on nobody's boundary — and there is nothing wrong with it finding no faces. It
should simply go.

So the rule is two *or none*, and the none-halves are dropped:

| | defects | valid | corner overlap difference |
|---|---|---|---|
| before | 0 | 4 | one open shell, declined |
| after | 0 | **6** | closed, valid, **reused: ok (9 faces)** |

Six of twelve, no defects anywhere, and a corner overlap can be cut again — the
first result of that shape this stack has produced that survives being an input.

Worth noting what made the difference, because it was not the idea. The
unconditional version of this closed the shells two iterations ago and wrecked
the edge counts; the guarded version blocked everything. The working one is the
same cut with the guard admitting a case the first guard called a failure — and
the only way to know that case existed was to print the counts rather than
reason about what a half ought to find.

Coverage **38 of 42**, 0 defects across the twelve results, **6 valid**.

## Every result is a solid

Two things, and the second was a check rather than a kernel.

**Naming the ends is not enough.** The cut that closed the corner overlap gives
each half of a rim to the faces whose loops name both its ends, and on a column
bored through a ball that found *three* takers — the two column sides and the
**sphere**, whose loop names both cut points because they lie on the curve where
the two meet, while the rim between them runs through the sphere's inside. The
middle of the piece has to be on the face's boundary too. With that, the four
column sides go from eight free ends each to none.

**And `face_free_ends` was asking the wrong question of a curved face.** It had
two branches: for a plane, whether the edges chain into a loop; for anything
else, where the boundary sat in the parameters. The second is a different
question, and it answered this one wrongly — the sphere in that same result has
every vertex of its boundary at even degree and still read as having two free
ends.

Degree is the definition, on any surface. What the parameter branch was really
protecting is one case: a face with *no* edges. A ball is one face bounded by
nothing and it is a solid — `u` goes right round, and its `v` ends are poles,
points with no length. An open tube has no edges either and is not closed, its
`v` ends being circles that border nothing. `u_wraps && (v_wraps || both ends
degenerate)` tells them apart, and `an_open_tube_is_reported_as_open_not_as_valid`
holds.

```
corner overlap     Difference / Union: valid, one closed shell
bore across a rod  Difference / Union: valid, one closed shell
plate + drill      Difference / Union: valid, one closed shell
sphere + drill     Difference / Union: valid, one closed shell
column in ball     Difference / Union: valid, one closed shell
round an edge      Difference / Union: valid, one closed shell
valid 12 of 12, 0 defects total
```

**Every result of every operation is now a valid solid**, one shell, closed, no
defects — and every one is accepted as the input to another boolean rather than
refused as not-a-solid. Where a second cut still fails it now fails on its own
geometry, which is an ordinary decline.

`what_comes_out_is_a_solid_that_can_go_back_in` asserts it: no defects, exactly
one shell, closed, valid, for all six pairs both ways round.

That is the composability hole shut. It was opened fourteen iterations ago by
noticing a plate could be bored twice and a rod could not.

Coverage **38 of 42**, 0 defects, **12 of 12 valid**.

## What chaining looks like now

With every result a valid solid, the next question is what happens when one is
cut again. First measurement, with a badly chosen probe — a box whose side sat
*flush* with the plate's:

```
open 4 of 8 faces:
  (3.0000, 4.0000, -1.0000)-(3.0000, 4.0000, 1.0000) faces [2]
  (3.0000, 4.0000, -1.0000)-(4.0000, 4.0000, -1.0000) faces [6]
```

A rectangle on `y = 4`, exactly where the probe's face lay on the plate's. That
is coincident faces, which this layer declines on purpose, and it says nothing
about chaining. Worth recording only because it nearly went down as one.

With a thin drill offset to meet nothing flush:

```
corner overlap     D / U: re-cut ok, valid
plate + drill      D / U: re-cut ok, valid
sphere + drill     D / U: re-cut ok, valid
bore across a rod  D / U: NotWatertight { 490 } / { 506 }
column in ball     D / U: NeedsArrangement { face: 0 }
round an edge      D / U: NotWatertight { 64 } / { 320 }
re-cut 6 of 12
```

Six of twelve, and each of the six comes back a valid solid in its own right —
so the chain does not degrade: a twice-cut result is as sound as a once-cut one.
The other six decline, which is the promise working; none of them returns
something wrong.

The three that fail are the three whose *first* result has the most complicated
boundary — a rod with a cross-hole, a bored ball, a rounded edge — and they fail
at the same two places everything failed before the last few iterations: the
subdivision, and the watertightness gate. Which is where the next work is, and
now on second-order inputs rather than on primitives.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## The rod's second cut, and an old debt coming due

490 open edges on a rod re-cut, and they are all at `x = -5`:

```
open-by-face [(1, 128), (2, 128), (3, 234)] (of 5 faces)
  (-5.0000,-0.9998,-0.0181)-(-5.0000,-0.9993, 0.0366) faces [3]
```

`x = -5` is the *cross drill's end cap*, five units outside the rod it bored.
Face 3 is that drill's wall, and in the twice-cut body every face reads
`loops=None`. In the once-cut body it does not:

```
F3 cylinder loops=Some([121]) edges=[0, 1] u(2.71,9.00) v(0.0,10.0)
```

Its parameter range is the whole drill, ten units of it, and what trims it to
the rod is the 121-point loop. The second boolean drops that loop — a face no
curve touches is reproduced from its edges, and carrying trim loops as well is
how a rim comes to be described twice, a fixed polyline beside an edge that
refines. So the wall fills from one end of the drill to the other.

Keeping the loop instead: `a_blind_hole` comes back 24.808 where it should be
25.133. The loop is coarser than the edges beside it, and the fill follows the
loop.

Which is the trim-loop refinement problem, unsolved since the iteration that
measured it — a carried loop keeps whatever resolution the *input body* was
constructed at, and cannot be refined without the edge and the loop drifting
apart. It cost 0.045% of a volume then. It now also costs a rod its second cut,
and the choice between them is the whole of it: drop the loop and the boundary
is wrong, keep it and the boundary is coarse.

So this is not a new problem, it is an old one presenting a second symptom, and
the two together are worth more than the one was. Reverted.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## The 48, at last, and what they were

Segment-keyed refinement failed twice before with exactly 48 open edges on a
bore across a rod, and I had never looked at where they were. They are at
`z = ±3` — the rod's own end caps — and the listing gives it away:

```
open-by-face [(0, 24), (1, 16), (2, 8)] (of 4 faces)
  (-0.7654,-1.8478,-3.0000)-(-0.6738,-1.8831,-3.0000) faces [0]
  (-0.7654,-1.8478,-3.0000)-(-0.6738,-1.8831,-3.0000) faces [1]
```

The same coordinates twice, once under the wall and once under a cap. Two map
entries, so two *different* points as far as the weld was concerned, printing
alike to four places. A loop stores parameters and an edge stores positions, and
the same rim point arriving by those two routes lands a little apart — further
apart than a quantum of `tolerance × 1e-3`. Every segment of the rim was
therefore two segments, and the seam opened along the whole of it.

At `tolerance × 0.25` the 48 go and the suite passes: 30 of 30, thirty-eight of
forty-two, twelve of twelve valid.

And re-cutting falls from six of twelve to four — sphere and drill stops
chaining — while the rod's own second cut stays at 490, which is the thing this
was for. Two lost, none gained. Reverted, and the quantum written down with it,
because the next attempt will otherwise spend a third iteration on those 48.

Three iterations have now ended with the same shape of answer: the mechanism is
right, the measurement will not have it. That is worth taking seriously rather
than trying a fourth variant — the loop-versus-edge split is not a bug with a
fix, it is two representations of one boundary, and something has to give up
being a representation.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## The bored ball's second cut is missing its curves

`column in ball` re-cut declines `NeedsArrangement`, which is a different
failure from the other two — those are `NotWatertight` — so it is worth its own
look. And the face that fails is not the one I expected:

```
subdiv f0 cylinder FAILED
  outer u[4.712,10.996] v[0.000,60.000] n=34
  chords ["u[8.395,8.395] v[27.221,32.779] n=2",
          "u[7.313,7.313] v[27.312,32.688] n=2"]
```

Not the bored ball. The *probe's own wall*, sixty units of cylinder, cut by two
straight chords that begin and end in the middle of it. Two chords that reach
nothing divide nothing, and `planar::subdivide` says so.

Those two are where the drill crosses the column's flat wall at `x = 1`. What is
missing is where it crosses the **sphere** — the drill enters and leaves the ball
as well, and those curves are not in the list. With them the four would close a
window; without them the two hang in the middle of the face.

So this one is not the loop-versus-edge duality that blocks the other two. It is
a face pair whose curves went missing between the first boolean and the second,
and that is a different question with a different answer — most likely in which
pairs are considered at all, since the sphere face of a once-cut body carries
twelve edges and a 218-point loop and no longer looks like a plain sphere.

Recorded rather than fixed; the iteration went on the loop/edge attempt above.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## A curve found and then thrown away

Following the bored ball's second cut to the pair that should have produced the
missing curves:

```
pair a0 sphere / b0 cylinder (marched): 1 curves, reaches [(false, true)]
```

The curve is there. `curve_reaches_face` says it does not reach the *sphere* —
the face it is on.

Because the sphere's face runs `u` from 3.069 to 9.352, that being where its
seam fell when the column was bored through it, and `Surface::invert` answers in
the surface's own terms: `u = 0.49` for a drill at `(1.3, 0.7)`. As far as that
face is concerned the point is at 6.77, but the raw value is tested against the
face's loops and its footprint, and 0.49 is outside both. Everything on the far
side of the seam is rejected out of hand.

Folding the parameter into the face's own range first:

```
pair a0 sphere / b0 cylinder (marched): 1 curves, reaches [(true, true)]
```

Re-cutting stays at six of twelve — the bored ball still declines, now further
along — but the rod's difference moves from `NotWatertight { 490 }` to
`NeedsArrangement`, which is a decline instead of an assembly of something with
four hundred and ninety open edges. And the bug it fixes is worse than the
count suggests: any face whose seam has moved, which is every periodic face a
boolean has touched, was silently dropping intersections that fell on the wrong
side of it. Nothing warned; the curves were found and thrown away.

`a_curve_is_found_on_a_face_whose_seam_has_moved` asserts the case: a drill
through a bored ball either changes it or declines, and does not come back
pretending nothing happened.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## The same mistake in three places

Following the bored ball's decline past `curve_reaches_face`, it moved to
`clip_to_face`, and past that to the subdivision itself — each time the same
thing. `Surface::invert` answers in the surface's own terms; a face cut open at
a seam runs its parameters from wherever that seam fell; and three separate
places compared one against the other without folding.

* `curve_reaches_face` — a curve at `u = 0.49` tested against a face running
  3.069 to 9.352, and rejected.
* `clip_to_face` — the same, so the curve could not be clipped to the face it
  was on.
* the ring and chord construction — each point unwrapped to sit near the one
  before it, which keeps a curve continuous and says nothing about where the
  curve as a whole belongs. The first point is taken as `invert` gives it. The
  ball's chords came back at `u ∈ [0.27, 0.73]` against an outline at
  `[3.069, 9.352]`, a whole period away and outside everything they were meant
  to cut.

Anchoring that first point to the middle of the face's own range puts them
where they belong:

```
outer  u[3.069,9.352] v[-1.571,1.571]
chords u[7.013,7.014] ... u[6.554,6.555] ...
```

Three fixes, all the same fix, and the suite stays green throughout — which is
the point worth noting. Nothing was testing a boolean on a body whose seam had
moved, so all three could be wrong at once and no test would say. It took
chaining a cut onto a bored ball to reach any of them.

The ball still declines, now for a reason further along and of a different kind:
its drill's circle arrives as **six fragments of two and three points**, not one
closed loop, and a subdivision cannot join what does not meet. That is the next
thing, and it is about how a marched curve is clipped rather than about
parameters.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Six fragments of a circle

The bored ball's second cut now reaches the subdivision with its chords in the
right place, and fails there because the drill's circle on the sphere arrives as
six pieces:

```
clip a0/b0 sampled: mine Some(6) theirs Some(7)
chords u[7.013,7.014] n=2 ... u[6.554,6.555] n=3 ... (six of them)
```

Six intervals from the sphere's side and seven from the cylinder's, for one
closed curve. Some splitting is right — the drill is at `(1.3, 0.7)` with radius
0.35, so it straddles the column's wall at `x = 1` and part of its circle is in
the window the column took out. But the pieces span two thousandths of a radian
each against a circle of extent 0.12, so these are slivers, not the two or three
arcs the geometry has.

`clip_to_face` samples along the curve and keeps runs of "inside". Where the
drill *grazes* the wall rather than crossing it, the curve runs along the loop's
boundary instead of through it, and containment flickers sample to sample. Six
runs is what flickering looks like.

Which is a different kind of problem from the last three: not a parameter
compared in the wrong frame, but a predicate asked a question it cannot answer
stably — is this point inside, for a point that is *on* the edge. The answers
that suggest themselves — merge runs across a short gap, drop runs below a
length — are both thresholds, and this file records what happens to thresholds
picked to fix one case.

Recorded rather than guessed at. Coverage **38 of 42**, 0 defects, 12 of 12
valid, 6 of 12 re-cuttable.

## Abstaining does not tell a graze from a crossing

The six fragments come from asking "is this point inside" of points that are
*on* the boundary. The obvious repair is to let those abstain — a sample within
tolerance of the loop has no honest answer — and fill each abstention from the
nearest sample that did decide.

Five tests fail. And the reason is that the abstention cannot tell the two cases
apart. Where a curve *grazes* the boundary, every sample along the graze
abstains and the neighbours either side agree, so filling them is right. Where a
curve *crosses* it, the samples nearest the crossing abstain too — and they get
filled from whichever side happens to be nearer, which moves the interval's end
away from the crossing. The bisection that follows then looks for the boundary
between two samples that no longer bracket it.

So the predicate is not what needs fixing; the *question* is wrong. Asking each
sample independently throws away the one piece of information that separates a
graze from a crossing, which is what the curve does either side. A graze has the
same answer on both sides of the ambiguous stretch and a crossing has opposite
ones — and that is decidable, but not by a rule about how far a point is from an
edge.

Reverted, and written into `clip_to_face` so the next attempt starts from the
distinction rather than from the threshold.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Every chord ends on the boundary

Second attempt at the six fragments, this time with the abstention resolved by
continuity — a stretch of ambiguous samples takes its neighbours' answer when
they agree, and splits between them when they differ, which is exactly the graze
against crossing distinction the last iteration asked for. It also guards the
bisection that follows, so it only narrows a bracket the raw predicate agrees
on.

Five tests fail, and one of them is `a_sphere_clipping_a_corner` — a
*first-order* case that has worked for twenty iterations. `NeedsArrangement`.

The reason is worth the whole attempt. **Every chord ends on the boundary.**
That is what makes it a chord: it runs from one edge of the face to another.
So the samples at both ends of an ordinary seam are ambiguous too, the run fills
out to the last sample, and the chord stops short of the edge it is supposed to
land on. The clip was not being made stable, it was being made blunt.

Which says what the distinction actually is, and it is not distance. A graze and
a chord's endpoint are both *on* the boundary; what separates them is
**direction** — a chord crosses the boundary, a graze runs along it. The test
wants the boundary's tangent and the curve's, and their angle, not how far apart
they are.

Two attempts this iteration, both reverted, and the note is in `clip_to_face` so
the third starts from direction.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Direction was sound and fired on nothing

Third attempt at the six fragments, using the distinction the last two arrived
at: a chord *crosses* the boundary, a graze *runs along* it, so compare the
curve's tangent with the tangent of the edge beneath it. Within about eight
degrees, the sample abstains and its confident neighbours decide; otherwise it
answers as before. That keeps a chord's endpoints — which are transverse —
deciding for themselves, which is what broke the last attempt.

Everything passes. And the clip is unchanged:

```
clip a0/b0: mine Some(6) theirs Some(7)
```

Six pieces and seven, exactly as before. The test never fires on this curve, so
the drill's circle is not being fragmented by lying along an edge. Three
hypotheses about *why* those six exist and all three wrong: it is not the
distance to the boundary, not the ambiguity of a point on it, and not the
direction the curve runs.

Reverted — a change that passes everything and moves nothing is not worth the
lines. What is worth having is that the field is now clear: the next attempt
should print the six intervals themselves and the samples either side of each
break, rather than proposing a fourth reason for them. Every time this session
has done that instead of theorising, it has taken one iteration; every time it
has theorised, three.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Six points for a circle

Printed the runs instead of proposing a fourth reason for them, and it took one
measurement:

```
clip f0 sphere: 6 runs of 512 samples, lo..hi = 0..6
  run 0..2:     (1.1057,0.9911,-2.6067)..(1.1147,0.9912,-2.6021)
  run 83..91:   (1.4777,0.9950,-2.4127)..(1.4988,0.9781,-2.4065)
  ...
  before run 83: surf-dist 1.16e-3
```

`lo..hi = 0..6`. The drill's whole circle on the sphere is **six points**. The
clip samples 512 places along it, all but a few land on the chords between those
six, and a chord sags 1.16e-3 off the sphere — past the 1e-3 tolerance. So
`inside` refuses them at its first test, "is this point even on the surface",
and the six runs are just the neighbourhoods of the six real points.

Not the boundary, not the ambiguity of a point on it, not direction. The curve
was never sampled finely enough to ask about.

`march` walks with `span(bounds) * 0.01`, and `bounds` is everything both solids
occupy. The probe here is sixty units long, so the step is 0.6 and a circle of
circumference 2.2 gets four of them. Every point is exactly on both surfaces —
`settle` sees to that — and the curve between them is a hexagon.

Putting points in afterwards until each chord is within tolerance of what it is
a chord of: all thirty-one tests pass, and re-cutting drops from six of twelve to
four. Denser curves change what the subdivision sees everywhere, not only where
they were wanted. Reverted.

The step should come from the curvature being walked, not from the size of the
model — which is a change inside `trace`, where the direction turns and the
sagitta of the next step can be measured before taking it. That is a real fix
rather than a pass over the output, and it is the next thing.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Density is right and costs two, twice over

Two ways of giving a traced curve the points it should have:

* a pass over the finished curve, splitting each chord until it lies within
  tolerance of both surfaces;
* an adaptive step inside `trace` — take the step, measure the sagitta of the
  chord it left, halve and retry until it is under tolerance, and lengthen again
  where the curve is straight.

The second is the better shape: the step decides where to start, the sagitta
decides where to stop, and only the parts that curve pay for it. Both pass all
thirty-one tests. Both take re-cutting from six of twelve to four, and it is the
same two that go — sphere and drill, both ways round.

Which says the cost is density itself, not how it was arrived at. A guess at the
mechanism: a traced curve's parameter *is* its point index, so `clip_to_face`
sampling a fixed 512 places along it looks through gaps once the curve has more
points than that, and where an interval ends is decided by which point was
landed on. Making the sample count follow the curve — 512 or four per point,
whichever is more — changes nothing. Still four of twelve.

So that guess is wrong too, and the mechanism is still unaccounted for. What is
established is narrower and worth having: **denser curves cost those two
results, by some path that is not the clip's sampling rate.** Every attempt to
improve the tracing will hit it, so it wants finding before another one is
made.

Reverted, both. Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12
re-cuttable.

## What density actually costs

Measured rather than guessed at, with the adaptive step in place, on the one
result it takes away:

```
subdiv f0 cylinder FAILED outer u[1.0356,7.3188] v[0.0000,60.0000] n=36
  chords ["u[1.1073,7.3188] v[27.2198,27.6200] n=59
           ends (1.1073,27.2996)->(7.3188,27.2996)"]
```

The probe's own wall, sixty units of it, and one chord: the drill's circle where
it crosses the sphere, now fifty-nine points instead of a handful. It ends
exactly on the far seam edge, 7.3188, and begins 0.0717 short of the near one —
about two thirds of a sample step.

That is the wrapping-curve case. A curve that goes all the way round `u` has to
be cut at the seam, and `seam_chord` rotates it to the first sample *past* the
cut, which is exact only when a vertex sits there. `insert_seam_vertex` puts one
there — when the curve wraps. With four points to a circle it did not wrap; with
fifty-nine it does, and something between the two means the vertex is not
landing.

So density does not break anything by being dense. It moves curves into a case
that was already imperfect and was not being reached: **the wrapping seam chord,
whose near end is short by a fraction of a step.** Every one of the last three
attempts at better tracing has been failing on that, one layer down from where I
was looking.

Which is a good place to stop looking at tracing. The seam chord is a known
mechanism with known machinery — `seam_origins`, `insert_seam_vertex`,
`seam_chord` — and the question is narrow: why is the vertex absent for this
curve. That is where the next iteration goes, and if it is answered the adaptive
step can go in behind it.

Reverted. Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## A sample landing on the seam

Traced the missing seam vertex, and it is not missing — it was never wanted:

```
pre-pass tag1 f0 cylinder origin 1.0356 curves [0]
  curve: 59 verts closed=true folded u[1.1073,7.3188] origin 1.0356
```

The face's period runs 1.0356 to 7.3188, and the curve's folded parameters reach
7.3188. One of its samples is *on the seam*. `insert_seam_vertex` sees that and
returns — rightly, there is nothing to add.

But `seam_chord` then rotates the ring to the sample *after* the jump, which is
the one at 1.1073, a step past the near edge, and the chord begins in open space.
The sample sitting on the seam is the one it should start from: `fold` puts it a
hair under `origin + period`, and that is the same place as a hair over `origin`.

Starting there when it is there:

```
sphere + drill, re-cut: NeedsArrangement -> NotWatertight { 58 }
```

The subdivision does its work, and the failure moves to the assembly. With the
sparse tracing this fires on nothing — six of twelve either way, every test
green — so by the rule this file has been kept to, it fixes nothing and should
go. It stays, and the reason is written down rather than assumed: it is the
precondition for the tracing work. Every attempt at a curve with enough points
has died on this chord, and it is the difference between a decline at the
subdivision and a fault at the gate.

The adaptive step still costs sphere and drill their second cut — 58 open edges
rather than a decline — so it is still out. But the thing it kept hitting is
now dealt with, and what is left of that case is one number instead of a
category.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## The adaptive step's last 58, and what they are

With the seam chord starting where it should, the adaptive step's remaining cost
is one number. Measured:

```
open-by-face [(0, 58)] (of 2 faces)
  (0.9501,0.7078,-2.7561)-(0.9515,0.6680,-2.7656) faces [0]
  F0 sphere   loops=Some([272, 58]) edges=3
  F1 cylinder loops=None            edges=2
```

Fifty-eight open edges and a fifty-eight point loop: the *whole* of the sphere's
hole, where the drill goes through it, unmatched along its entire length. The
face across that hole is the drill's wall, which has no loops at all and takes
its boundary from its edges.

So the sphere describes that circle with a loop and the wall describes it with
an edge, and once the curve is dense enough for the two to differ, they do. It
is the loop-versus-edge duality again — the same thing that pins `refine_edges`,
that drops a cross-hole's trim loop on a second cut, and that three attempts at
segment-keyed refinement could not resolve.

That is the fourth distinct symptom of it, and the first where it is unambiguous:
one circle, two descriptions, fifty-eight edges open. Everything else this
session has found in the tracing — a step scaled to the model, a clip that
fragments, a chord that starts in open space — has been real and has now been
either fixed or written down. What is left in the way of tracing curves properly
is not in the tracing.

Reverted the adaptive step; the seam chord fix stays. Coverage **38 of 42**, 0
defects, 12 of 12 valid, 6 of 12 re-cuttable.

## What the example and the README were still claiming

Ran the worked example rather than trusting it. The numbers hold — a plate bored
three times comes to 3604.504 against 3604.381 exact, re-tessellates at three
tolerances, round-trips through STEP with all nine faces analytic, and the
imported solid takes a fourth bore. But two claims had gone stale under the last
several weeks of changes:

The example said a cross-hole declines because that pair "has no closed-form
intersection here". It has none, and it has not been *refused* for that reason
since `march` went in: the curve is traced, and what declines is the assembly.
The printed reason says `NotWatertight { open_edges: 160 }` and always did; the
comment beside it explained a different failure.

The README listed `NoClosedForm` among the declines and did not mention that a
pair without one is traced, nor `TangentialContact`, which is the one the
off-axis torus actually gives. It also said nothing about a result being a
*solid* — one closed shell, every edge on exactly two faces — which is this
month's work and the thing that decides whether a result can be cut again.

Both corrected. Neither was wrong when written, which is the point: a claim
about behaviour ages with the behaviour, and nothing in a test suite checks the
prose.

Coverage **38 of 42**, 0 defects, 12 of 12 valid, 6 of 12 re-cuttable.

## Where the two descriptions come apart

The fifty-eight open edges are one circle described twice, so the question is
where the second description comes from. Measured on the first cut:

```
once: 2 faces
  F0 sphere   loops=None edges=[0, 1]
  F1 cylinder loops=None edges=[0, 1]
  edge 0: 72 verts    edge 1: 72 verts
after refine_edges: unchanged
```

A ball with a hole through it: two faces, neither carrying loops, both naming
the same two edges of seventy-two points, and `refine_edges` leaves them alone.
There is one description here and it is shared. Nothing to come apart.

It is the *second* cut that splits them. The sphere is subdivided again and
comes back with a fifty-eight point hole loop; the probe's wall is not touched
by any curve on that side and carries through with no loops at all, taking its
boundary from its edges. One circle, and now a loop on one side and an edge on
the other.

So the duality is not something a body carries. It is made, on each boolean,
between a face that was subdivided and a face that was not — and the two are
only equal while the curve is coarse enough that the subdivision's boundary and
the edge agree point for point.

Which is the sharpest statement of it yet and says where a fix has to go: not
into `refine_edges`, not into the tracing, but into what `split_face` gives a
piece that no curve touched. It has the loops; it is told to drop them because
its edges describe the same boundary; and that is true of the boundary and false
of its *resolution*.

Reverted the adaptive step again. Coverage **38 of 42**, 0 defects, 12 of 12
valid, 6 of 12 re-cuttable.

---

## Where this stands

Measured, not remembered, over the fourteen pairs in three operations each:

```
of 42 operations: 38 resolved, 1 legitimately empty, 3 declined
of the 38 resolved: 37 valid solids, 5 defects in total
  torus off-axis Difference / Union / Intersection: TangentialContact
```

**What resolves.** Thirty-eight of forty-two. The one that is neither resolved
nor declined is `box/box flush` intersection, which is *empty* — two solids
meeting at a wall have nothing in common — and that is an answer.

**What declines.** Three, all the same pair and all for the same reason: a
cylinder grazing a torus's inner equator touches it rather than crossing it, the
intersection pinches to a point, and the tracer has no direction to step in where
the normals are parallel. `TangentialContact` says so. Solving it needs the pinch
as a node of a curve network with four branches leaving it.

**What comes back.** Thirty-seven of the thirty-eight results are valid solids:
one closed shell, every edge on exactly two faces, no defects. They are
therefore usable as the input to another boolean, which for most of this
project's life they were not.

**The exception.** `box/box flush` union — two cubes sharing a wall — comes back
with five edges that only one face claims and two shells, neither closed. It is
the only first-order result that is not a solid, and it is the next thing worth
doing.

**Chaining.** Six of twelve second cuts succeed, and those six are valid solids
again, so the chain does not degrade. The rest decline. Behind them is one
mechanism, stated as sharply as this project has managed: a boolean makes a
loop-versus-edge duality *between* a face it subdivided and a face it did not,
and the two agree only while the curve is coarse enough for the subdivision's
boundary and the edge to match point for point. It blocks the adaptive tracing
step, the trim-loop refinement, and the second cut of a cross-holed rod, and it
is one problem wearing four hats.

**The promise.** Unchanged and enforced: a result is closed and manifold, or
there is no result and a reason. Every decline above is a reason, and no
operation in the matrix returns something wrong.

## A ring on one side, four sides on the other

The one first-order result that was not a solid, measured:

```
11 faces 25 edges, defects 5
  shell 0: [0, 2, 3, 4, 5, 1] closed=false
  shell 1: [6, 8, 9, 10, 7]   closed=false
  edge 12: 1 faces, 5 verts
  edge 16: 1 faces, 2 verts   edge 20: 1 faces, 2 verts
  edge 22: 1 faces, 2 verts   edge 24: 1 faces, 2 verts
```

Two cubes sharing a wall. The lower one's top face gets a square hole — one
closed edge of four corners — and the upper one's four walls each bring the side
they stand on. The same square, five edges, every one claimed by a single face,
and two shells that do not close.

Cutting the ring at its own corners and pairing each piece with the side it
coincides with — only edges no second face claims, and only where every piece
finds exactly one partner:

```
11 faces 24 edges, defects 0
  shell 0: [0, 2, 3, 4, 5, 1, 7, 8, 6, 9, 10] closed=true
```

One shell, closed. The last defect after that was the ring itself, left behind
with nothing naming it, so edges no face claims are now dropped — a body
carrying one reads as defective for a reason that is only bookkeeping.

```
38 resolved, 1 empty, 3 tangential; 38 valid, 0 defects
```

**Every operation in the matrix that resolves now returns a valid solid.** One
closed shell, every edge on exactly two faces, nothing left over. The three that
decline are the one tangential pair, and the empty one is an intersection that
is genuinely empty.

`a_result_is_always_watertight_or_declined_never_neither` now asserts it for all
forty-two: closed, no defects, a solid.

Coverage **38 of 42**, and for the first time the count of results that are
solids is the same number.

## A face's own holes were not in the arrangement

Re-measured chaining after the flush-stack fix, and added that case to the list:
six of fourteen second cuts, and five of the eight failures are
`NeedsArrangement`. The flush stack's, on the lower cube's top face:

```
subdiv f4 plane FAILED outer u[-2,2] v[-2,2] n=4
  chords ["u[0.8803,1.0500] v[-1.3000,-1.0000] n=8",
          "u[0.3500,1.0500] v[-1.6492,-1.0000] n=29"]
```

Both chords end on `v = -1`, which is the edge of the square hole the first
union left in that face. And `subdivide` is given the outer outline and the
chords and nothing else — the *new* intersection rings go in as closed paths,
but the holes the face already had do not. So both chords ran to a boundary the
arrangement had never heard of, and dangled.

They go in now, as bare paths: they carry no curve, so their points come back
with no index and are welded by position like any other point of a face's own
boundary. The provenance lookups needed guarding — a path index past the chords
is one of these — and one of them was indexing straight into `chords` and
panicking on the first attempt.

Face 4 subdivides. The flush stack's second cut still declines, now at face 0,
so the count is unchanged at six of fourteen and this is not the whole of that
case. It stays: a face's holes belong in its arrangement whether or not this
particular pair needed them, and the fix is one a second cut on any bored face
will want.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6 of 14
re-cuttable.

## The flush stack's second cut, one layer on

With the face's holes in the arrangement, the flush stack's re-cut moves off the
top face and stops here:

```
F0..F10 plane loops=None edges=4 (F4: 8)
decline@1521 -> NeedsArrangement { face: 0 }
```

`chosen_origin()` returns nothing: the drill's own wall cannot find a seam to be
cut open at. `seam_origin` gives `None` when every curve on the face lies at
constant `v` — a stack of bands, which `split_swept` handles and which wants no
seam — or when there are no curves at all.

The drill here goes through the lower cube's top and bottom and past the upper
cube's wall, so it should have both kinds: circles at constant `v` where it
crosses a floor, and cuts that run along it where it grazes the step at `x = 1`.
One of those two is not arriving, and which it is decides whether this is a
missing curve or a seam rule that does not fire — the same fork as the bored
ball two iterations ago, where it turned out to be three places comparing
parameters in the wrong frame.

Recorded rather than guessed: the next iteration should print what `seam_origin`
is given for that face before proposing anything.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6 of 14
re-cuttable.

## No seam wanted is not no answer

Printed what `seam_origin` is given for the face the flush stack's second cut
dies on — the drill's own wall:

```
seam_origin curve 0: 8 pts, v spread 0.0000 of 60.0000
seam_origin curve 1: 29 pts, v spread 0.0000
seam_origin curve 2: 43 pts, v spread 0.0000
seam_origin curve 3: 9 pts, v spread 0.0000
seam_origin: wants_seam=false u_wraps=true curves 4
```

Every curve level. `seam_origin` says no seam is wanted, which is right — a face
cut only at constant `v` is a stack of bands. That is `split_swept`'s case, and
rings at constant `v` do go there; but an *arc* at constant `v` is a chord, so
the face arrives here instead, where a missing origin was read as "cannot be cut
open" rather than "does not need to be", and declined.

Taking the face's existing seam when none is chosen gets past it, and exposes
what was behind:

```
outer u[-3.1416,3.1416]
paths u[0.0000,1.0297] ... u[2.1119,6.2832] ... u[0.0000,6.1336]
```

Chords between 0 and 2π against an outline from −π to π. Skipping the seam had
skipped the *fold* with it, and `invert` answers in the surface's own terms.
Folding at the existing origin puts them in frame and splits the one that
crosses.

One path then remained wrong: a closed ring spanning 0.976 of the period against
a wrapping threshold of 0.98, folded and never cut, jumping the rectangle's whole
width at the seam. Detecting the *jump* rather than measuring the span catches it
— the subdivision stops failing — and takes re-cutting from six of fourteen to
four, because the rings it newly calls wrapping include ones a drill through a
bored ball needs left alone. Reverted, and the threshold's weakness written down
beside it.

Two of the three stay. The flush stack still declines at face 0, further along
than it did. Coverage **38 of 42**, 38 of 38 valid, 6 of 14 re-cuttable.

## A threshold that was protecting something else

The wrapping test measures how wide a ring's parameters span, which misses a
ring straddling the seam. Two better tests:

* **Does it jump?** Catches the missed one — and also catches a *small* ring that
  merely straddles the seam, which is not wrapping at all.
* **How far does `u` turn before it closes?** Take each step the short way round
  and add them up. A ring that goes round comes to a whole period; one that only
  straddles the seam comes back to nothing. That tells the two apart exactly.

Both take re-cutting from six of fourteen to four, and the second is
unimprovable as a *test*, so the fault is elsewhere. Measured:

```
ring f0 7 pts, span 0.4572 of 6.2832 (0.073), jumps=false
ring f0 7 pts, span 5.2788 of 6.2832 (0.840), jumps=true
```

Seven points, turning a full period, spanning 0.84 of it because that is what
seven samples of a circle reach. It *does* wrap. The 0.98 threshold has been
calling it something else, and the wrapping path it would otherwise take leaves
seven edges open.

So the threshold is not a measurement that needs sharpening. It is a guard, by
accident, around `seam_chord` and a coarsely sampled ring — and every attempt to
measure wrapping properly walks into what it was hiding. That is worth knowing
before the next one: fix the coarse-ring case first, then the test can be
correct.

Reverted, with all three tests and their numbers written in beside it.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6 of 14
re-cuttable.

## A revert that did not take

Setting out to fix the coarse-ring case, I first measured what the code actually
does — and it was not what the last entry says. The wrapping test in the file was
the *turning* test, not the span test, and re-cutting stood at four of fourteen,
not six. The previous iteration recorded a revert and a measurement of six; the
revert did not take, and the six came from a stale test binary.

Reverted properly, and verified by looking rather than by reporting:

```
grep -c "turn.abs()"      0
grep -c "period * 0.98"   1
re-cut 6 of 14
```

The comment and the code had also come apart — the comment described the span
test while the code implemented the turn test — which is exactly the state a
half-applied edit leaves behind, and exactly the state that is invisible unless
the file is read.

Worth taking from it, because this file has been arguing all session for
measuring over reasoning: *the same discipline applies to one's own edits*. A
patch that reports success and a suite that stays green are not evidence the
patch did what it said. Three greps would have caught this at the time, and from
here a revert is checked the same way a fix is.

No change to the kernel this iteration beyond undoing one that should not have
been there. Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6
of 14 re-cuttable.

## The seven, and which ring they belong to

Applied the turning test again — verifying each edit landed this time — and
measured what it costs:

```
open 7 of 2 faces
  (0.9878,0.5417,-2.7804)-(0.9989,0.8785,-2.6889) faces [0]
  ... seven of them, all on face 0
  F0 sphere   loops=Some([272, 7])
  F1 cylinder loops=None
```

Seven open edges and a seven-point loop: the sphere's *hole*, where the first
drill went through it, open along its whole length. That drill is on the axis,
so its circle goes right round the sphere's `u` — it genuinely wraps, the
turning test is right about it, and `seam_chord` on a ring of seven points then
leaves every edge of it open.

Which settles what the 0.98 threshold has been doing. It is not a rough
measurement of wrapping. It is the line below which a ring is sent down the
*non*-wrapping path, and for this ring that path works and the wrapping one does
not. The threshold is load-bearing for the case it misclassifies.

So the order is: make `seam_chord` right for a coarsely sampled ring, then the
wrapping test can say what is true. Trying to correct the test first has now
failed three times — by span, by jump, by turning — each time on the same seven
edges, and each time the test was more right than the code it fed.

Reverted, verified by grep this time rather than by report: `turn.abs()` absent,
`period * 0.98` present, probes zero.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6 of 14
re-cuttable.

## The coarse ring has its seam vertex

Went to fix `seam_chord` for a coarsely sampled ring and measured first:

```
seam_chord f0 curve 0: 7 pts, origin 0.4677, nearest sample to seam 5.551e-17
```

There is nothing wrong with the ring. A sample sits *exactly* on the seam —
`insert_seam_vertex` put it there — and the rotation has an exact place to start
from. Seven points is coarse and it is not the problem.

The problem is that the cut happens on one side only. The sphere's `u` wraps, so
its ring becomes a chord running seam to seam; the drill's wall does not wrap
that way and keeps the ring closed. One circle, described on one face as a chord
with the seam point at both ends and on the other as a ring with it once — and
the seven edges that stay open are that difference.

So it is not `seam_chord` that is wrong for coarse rings, and it is not the
wrapping test. It is that converting a ring to a chord is a decision one face
makes about a curve *two* faces share, and the other is never told. Which is the
same shape as the loop-versus-edge duality — a description made on one side of a
seam and not the other — arriving from a fourth direction.

Reverted; grep-checked: `turn.abs()` 0, `period * 0.98` 1, probes 0.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, 6 of 14
re-cuttable.

## Chaining, held down by a test

Second cuts have been measured all month by scratch tests written and deleted
each time, which is how a revert once went unnoticed for two iterations. They
are a test now: `a_result_can_be_cut_again` takes three results — two boxes
overlapping at a corner, a bored plate, a bored ball — and cuts each of them
again, both ways round, asserting the second result is a valid solid and that
the drill left a trace.

Writing it turned up two things about the *measurement* rather than the kernel.

The plate's second cut "left no trace", and rightly: the probe I had been using
sits at radius 1.48 and the plate's existing bore is 2 wide, so the drill was
entirely inside the hole and removed nothing. Every measurement of that case
this month has been of a drill cutting air. Each case now has a drill placed
where it can bite.

And the bored ball's second cut depends on *where* the drill goes: at
`(1.3, 0.7)` it resolves, at `(1.6, 0.9)` it comes back with the same seven open
edges the last three iterations have been chasing. So that failure is not a
property of the shape, it is a property of where the second cut falls relative
to the seam — which is consistent with a ring being cut open on one face and not
the other, and is the sharpest evidence yet for it.

The test pins what works and does not claim the general case; the plan records
which cases decline and why.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, and the six
second cuts that work are now checked on every run.

## One root under four symptoms

Measured the bored ball's second cut where it fails, at `(1.6, 0.9)`:

```
open 7 of 2 faces
  F0 sphere   loops=Some([272, 7])
  F1 cylinder loops=None
```

The seven-point loop is the *new* drill's window. Seven points for a circle of
radius 0.35, because `march` walks with a hundredth of the model and the model
here contains a sixty-unit probe. The drill's wall opposite it has no loops and
takes its boundary from edges, and the two do not agree, so all seven edges of
that window stay open.

Which is the same seven this month's last four iterations have chased through
the wrapping test, `seam_chord`, and the coarse ring — and they all sit on one
root:

* `march`'s step comes from the size of the model, so a small feature is traced
  with a handful of points;
* a coarse curve makes a coarse loop on the face that is subdivided;
* the face across it describes the same curve with an *edge*;
* and while both are coarse in the same way they agree, so nothing shows.

Refining the curve — either afterwards or with an adaptive step — makes the two
disagree, which is why every attempt at better tracing has cost exactly the
results whose second cut relies on that accidental agreement. The tracing was
never the problem and neither was the seam. **A curve is described twice, and the
two descriptions only match while both are wrong in the same way.**

That is the whole of it, and it says the order plainly: one description or the
other has to go before the tracing can be improved. Nothing else in this cluster
is worth another attempt until it does.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, six second
cuts checked by test.

## Segment refinement, one test away

Retried it, now that the topology it depends on is sound — every edge on two
faces, every shell closed, seam vertices exact. It refines a boundary *segment*
once, named by where its two ends are, and hands the points to the edge and to
every loop running along it, so the two descriptions cannot drift because there
is only one.

The last attempt broke five tests. This one breaks one:

```
a bored ball Difference could not be cut again: NotWatertight { open_edges: 101 }
  open-by-face [(0, 5), (2, 96)] of 3 faces
  F0 sphere   loops=Some([272, 29]) edges=3
  F1 cylinder loops=None            edges=2
  F2 cylinder loops=Some([29])      edges=1
```

Ninety-six of the hundred and one are on F2 — the *new* drill's wall, which now
arrives carrying a 29-point loop where it previously carried none. Refining the
loops has given a face a loop it did not have, and its fill does not agree with
the face across it.

Which is the same duality once more, one level along: it is not that the loop and
the edge disagree about a boundary, it is that one face has a loop and the other
does not, and refinement makes that difference visible where coarseness hid it.

Still a regression, so still out — and worth noting that the thing that caught
it was `a_result_can_be_cut_again`, written last iteration precisely because
this measurement had only ever lived in scratch files.

Coverage **38 of 42**, 38 of 38 resolved results are valid solids, six second
cuts under test.

## The loop and the edge are not redundant, they conflict

Segment refinement left one test failing, and the diagnosis was that a carried-
through face arrives with a loop where its neighbour has none. So the obvious
pairing: keep the loops on *every* carried-through face, so no face of a seam
has one the other lacks. The one reason that had failed before — a loop is a
fixed polyline beside an edge that refines, and a blind hole lost 1.3% of its
volume to the coarser of the two — is exactly what segment refinement removes.

It is worse. Five tests, not one:

    a_result_can_be_cut_again
    bores_can_be_drilled_one_after_another
    a_curved_rim_survives_refinement
    a_bore_across_a_rod
    what_comes_out_is_a_solid_that_can_go_back_in

That number is worth more than the attempt cost. The standing theory has been
that a boundary is *described twice* and the two descriptions drift apart — a
redundancy, harmless while both are equally coarse. If that were the whole
story, giving both descriptions the same points would make the duplication
harmless in a second way, and the tests would improve or stay put.

They got worse, so it is not redundancy. When a face has both a loop and edges
naming the same boundary, the two are *used differently* downstream — the loop
trims a parameter region, the edges stitch a shell — and making them agree
point-for-point does not make those two uses agree. Dropping the loop where the
edges suffice is not a workaround for coarseness. It is load-bearing: it picks
which of the two mechanisms owns that boundary, and having both own it is worse
than having the coarse one own it.

Which sharpens the design answer given earlier in this session. "Separate the
topological boundary from the tessellation input" is not a tidying-up that can
be deferred behind better tracing. It is the blocker itself: as long as a face
can carry both a trim loop and edges over the same curve, every fix has to
choose one silently, and the choice is invisible at the call site. The loop
should stop being a boundary representation and become what it is used for — a
trimming input derived from the edges on demand — with the edges the single
owner of topology. That is a change to what a `Face` *is*, not to how a curve is
traced, and nothing in the refinement family will substitute for it.

Both attempts reverted, verified by grep (`let pinned` present, `segs`/`intern`/
`shoelace` gone, the carried-through `loops` drop restored). Suite green: 32
boolean, 21 brep, 13 brep_csg, 17 nurbs, 7 step, 638 lib. Coverage still 38/42,
chaining still 6/14.

## What the stored loops actually contain

Last entry concluded the blocker is that a `Face` can hold both a trim loop and
edges over one curve. Before rebuilding anything, measure how much is really in
there. Probe (kept at `scratchpad/loops_probe.rs`): run six solid pairs through
all three ops, and for every face that carries stored loops, ask two questions —
is each loop point already a vertex of one of that face's edges, and what ring
count would the edges alone derive?

    faces with stored loops:            6   (out of 18 booleans)
      all loop points already on edges: 2
      edges derive the same ring count: 2
      edges derive nothing at all:      0

Six faces. The override is far rarer than the argument around it suggested. And
the four that disagree are all one shape — a sphere carrying two rings where its
edges derive one:

    ring0: 128 pts, 128 off-edge, u in [3.1416, 9.4248], v in [-1.5708, 1.5708]
    ring1:  99 pts, ALL on edges

`ring1` is the intersection circle, and it is entirely edges. `ring0` is a full
2π turn in `u` starting at the seam origin and the whole pole-to-pole span in
`v`: it is the *parameter outline* — down one seam, around a pole, back up the
other. Not a boundary of the solid at all, an artifact of the parameterisation.

So the stored-loop override, in the measured corpus, carries exactly two things:

  * rings the edges already derive, which are pure duplication; and
  * the parameter outline of a periodic face, which no edge carries because the
    seam is not a topological boundary.

Neither is a genuine boundary that edges cannot express. That is the earlier
design answer — *make the seam a real edge* — arriving as a measurement rather
than an opinion, and it makes the change tractable: `face_loops` already derives
rings from edges and treats stored loops as an override, so the work is to give
the derivation the outline for a wrapping face and then stop storing. Six faces
is the whole blast radius.

### The seam vertices are already real

`parameter_outline` builds its ring with `vertices: vec![usize::MAX; n]`, which
reads like the outline names no vertices — that would have made the two
descriptions incomparable in principle, one with identity and one without.
Checked instead of assumed, and it is not so: by the time the loops reach the
finished body every one of those placeholders has been resolved by the weld.

    Union F0 ring0: 128 pts, 0 unnamed, area 19.4308     <- the outline
    Union F0 ring1:  99 pts, 0 unnamed, area -1.6796     <- the circle, on edges

So the 128 seam-and-pole points are real, welded, shared vertices of the body.
They are not a second parallel description with its own coordinates; they are
the same vertices everything else uses, and the only thing missing is an *edge*
naming them. Which makes the change additive rather than a reconciliation: the
identity is already there, the topology is not.

The shape of the work, then:

  * `parameter_outline` walks four sides — bottom rim, seam, top rim, seam — and
    only the two rims need `carried`; in `face_loops` those rims are the face's
    own edges, which it already chains. The seam runs are sampled from the
    surface and the ranges, needing nothing a `Body` lacks.
  * So the derivation can move into `face_loops` whole, and `Face::loops` can
    stop being an override.

The one thing that must be settled first is what a seam edge *is* to the
validity check: `face_free_ends` and the watertight test count an edge as
belonging to exactly two faces, and a seam has the same face on both sides.
`Edge::surfaces` being a pair of the same surface is representable; being on one
face twice is not, today. That is the next thing to measure, not to guess.

## A face is now allowed to meet itself

The measurement said the seam wants to be an edge and the vertices are already
there. So: what in the validity model forbids one? Two places, both saying the
same thing, both in `body.rs`:

    let uses = self.faces.iter().filter(|f| f.edges.contains(&ei)).count();
    if uses != 2 { … EdgeFaceCount … }

and the matching test inside `shells`. Both count *distinct faces*. The manifold
condition is not that: it is that a curve is walked twice by the boundaries
around it, and both walks may belong to the same face. That is what a seam is —
a cylinder's side meets itself where its parameterisation closes — and STEP
states it directly, two `ORIENTED_EDGE`s over one `EDGE_CURVE` inside a single
face's loop. Counting faces is a special case that coincides with the real rule
exactly while no face uses an edge twice.

Measured before changing: across four solid pairs, three ops and three
primitives, faces listing an edge twice = **0**. So the change is a no-op on
everything that exists and purely enabling — the whole suite agrees, unchanged.

Changed both counts to sum uses per face, and renamed the defect's field from
`faces` to `uses`, since it is no longer counting faces.

A no-op change with no test is a change that will be quietly undone, so
`a_face_may_meet_itself_along_a_seam` builds the case by hand: take a capped
cylinder, run an edge along the side between its two rims at one `u`, and let
the side name it twice. That body must have no defects and no open shell; naming
it once must still be `EdgeFaceCount { uses: 1 }`. Verified the test earns its
keep by restoring the face-count and watching it fail:

    a seam edge is not a defect: [EdgeFaceCount { edge: 2, uses: 1 }]

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 7 step, 639 lib (+1).
wasm32 and the example build.

Next: `parameter_outline`'s two rim walks are the face's own edges once a seam
edge exists, so the derivation can move into `face_loops` and `Face::loops` can
stop being an override. The blocker named at the end of the last entry is gone.

## The loop carries a bit the edges cannot

The plan after the seam-edge change was to move `parameter_outline` into
`face_loops` and retire the stored override. Built it as a probe first — rims
from the face's own edges, seams sampled from the ranges — and compared it
against the stored rings (`scratchpad/outline_probe.rs`):

    ball+ball Union F0 sphere: stored 19.4308 (128 pts) | derived 19.7392 | ratio  1.02
    ball+ball Union F1 sphere: stored  7.6209  (99 pts) | derived 19.7392 | ratio  2.59
    ball+ball Isect F0 sphere: stored  1.6796  (99 pts) | derived 19.7392 | ratio 11.75

19.7392 is 2π², the whole sphere rectangle — the derivation returns it for
*every* face, including the two that are small caps bounded only by the
intersection circle. The reason, printed for each:

    u_range (3.14159, 9.42478)  v_range (-1.5708, 1.5708)  wraps false/false  edges 1

Identical on all of them. A boolean output face keeps the *input surface's* full
domain, not the extent of the piece it actually is, and `u_wraps`/`v_wraps` are
false even on a face covering a whole sphere. So there is nothing in the ranges
to derive an outline from: they do not distinguish the cap from the remainder.

Which exposes the real reason the stored loops exist, and it is not the one the
last two entries assumed. **A closed curve on a closed surface bounds two
regions, and the edges alone can never say which one the face is.** One circle
on a sphere is the boundary of the cap and equally the boundary of everything
else. That is not duplication of what the edges hold; it is a bit of information
the edges structurally cannot carry, and the stored loop carries it — in its
winding and its area.

So "derive the loops from the edges and retire the override" is not achievable,
and the previous two entries were wrong to head there. The `Face::loops` doc was
right all along that a boolean result's faces must state something, though for a
different reason than it gives: not that intersection curves lack edges — they
have them now — but that a curve does not determine a side.

What is still worth doing is making that bit *small*. Today a face states its
region with a full polyline of 128 points, which is why it drifts from the edges
under refinement and why refining them apart opens seams. The same information
fits in the face's own parameter extent — the piece's true `u_range`/`v_range`,
which are currently stale — or in one flag per ring saying which side is inside.
Either is derivable-from and checkable-against the edges, neither drifts, and
both leave the edges the single owner of geometry.

That reframes the target: not "loops derived from edges" but "loops reduced to
the bit edges cannot hold". The seam-edge change stands; it was needed either
way and remains correct. Suite untouched and green.

## The missing bit, as a shipped bug

Last entry concluded that a trim loop carries something the edges cannot: which
of the two regions a closed curve bounds the face actually is. That was an
argument about representation. Followed it into the one place where the
representation has to leave the crate — STEP — and it is a live bug.

    Union:        before 121.769  after 0.0   | export skipped []  import skipped []
    Difference:   before  88.279  after 0.0   | export skipped []  import skipped []
    Intersection: before  24.723  after 0.0   | export skipped []  import skipped []

A boolean of two balls, written to STEP and read back, is a solid of **no
volume** — and both ends report complete success. Not a decline, not an
approximation: silent loss of the whole model, in the format whose entire point
is that the model survives.

Narrowed by comparison, which named the case exactly:

    plain sphere:   in 1 faces -> 2 ADVANCED_FACE, 2 CIRCLE -> vol 112.98 -> 112.98
    plate-bore:     in 7 faces -> 7 ADVANCED_FACE, 2 CIRCLE -> vol 134.88 -> 134.88
    ball-bore diff: in 2 faces -> 2 ADVANCED_FACE, 2 CIRCLE -> vol  94.62 ->  94.62
    ball-ball diff: in 2 faces -> 2 ADVANCED_FACE, 1 CIRCLE -> vol  88.28 ->   0.00

Every case with two bounding circles round-trips. The one with **one** circle
does not — a single closed curve on a closed surface, which is precisely the
ambiguous case the last entry described. `loops` appears nowhere in
`export.rs`: the trim loop, the only thing holding the side, is dropped, and the
reader is handed a boundary two regions share. It took the empty one.

Two things kept this hidden. `POLYLINE 0` made the file look clean and analytic
rather than lossy. And an empty tessellation reports `is_closed()` — vacuously,
with no edges to be open — so every volume helper in `tests/step.rs` asserted
watertightness on nothing and passed.

Fixed the way the rest of the stack behaves: `Unsupported::AmbiguousRegion`. A
face carrying a trim loop that names points none of its edges carry cannot be
stated in AP203, so it is reported instead of written. `ball-ball` now exports
zero faces and says why; every unambiguous case is untouched, including
`ball-bore`, whose two circles do pin the region between them.

Also added `assert!(report.triangles > 0)` to the volume helper, so a body that
exports to nothing can never again pass a watertightness check by being empty.

Test `a_face_whose_edges_do_not_say_which_side_is_reported_not_written` pins
both halves — the ambiguous case names itself and writes no face; the drilled
ball still round-trips to 1e-6.

This is a decline, not a repair: such a solid cannot be exported at all now,
where before it exported wrongly. Stating it correctly needs the region written
rather than implied — the face's true parameter extent, which is the same stale
`u_range`/`v_range` the last entry found, or the `seam_faces` grid path that
already exists for edgeless faces and is not reached here. That is the next
piece of work, and it is now backed by a failing-in-the-world case rather than
an argument.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 8 step (+1), 639 lib. wasm32
and the example build.

## The pole was a second silent loss, and AP203 already had the word for it

Quantified the reach of the export bug across nine solid pairs and three ops:

    21 results: 14 round-trip, 6 declined, 1 still wrong

The six declines are the ambiguous-region case caught last entry — each was a
silent corruption before. But one case was *still* silently wrong and did not
decline:

    WRONG ball+bore Intersection: 18.2890 -> 11.8322

A different mechanism. That face carries no stored loops at all, so the
ambiguity check never sees it. Printing the ranges either side of the trip named
it in one line:

    out: F0 sphere v=(-1.5708, -1.2310)      <- pole to rim
    in:  F0 sphere v=(-1.2310, -1.2310)      <- rim to rim

The cap collapses. Import recovers a face's extent from the vertices its edges
reach, and at a sphere's axis the iso-curve is a single point — there is no
curve for an edge to be. So the rim is all the file offers, the face ends where
it begins, and both caps tessellate to nothing. 18.289 − 11.832 = 6.46, which is
the two caps.

AP203 has had the answer since 1994: `VERTEX_LOOP`, a bound holding one vertex.
Grepped for it — absent from the crate, both directions. Added:

  * export: a face whose `v_range` end has every `u` landing on one point emits
    `VERTEX_POINT` → `VERTEX_LOOP` → `FACE_BOUND`, never the *outer* bound,
    since a point encloses nothing;
  * import: a bound whose loop is a `VERTEX_LOOP` contributes no edge, only the
    `v` its vertex sits at. Its `u` is meaningless — every `u` is the same point
    — so only `v` joins the extent.

    21 results: 15 round-trip, 6 declined, 0 still wrong

The cap now returns with the range it left with, to the last digit shown.
`a_face_that_runs_up_to_a_pole_keeps_its_cap` pins it: two `VERTEX_LOOP`s in the
file, `skipped` empty at both ends, volume within 1e-3.

Worth noting what the two bugs share. Both are the same shortfall this thread
has been circling — the boundary of a face is not always a set of curves, and a
representation that only knows curves loses whichever part isn't one. The pole
is that shortfall in its simplest form, and it has a standard fix. The seam is
the same shortfall where the standard fix is an arrangement, and that is still
open.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 9 step (+1), 639 lib.

## A guard for the class, not the two instances

Two silent losses found by hand in two ticks, by the same probe, in the same
place. A third is not worth waiting for, so the probe is now a test:
`no_solid_leaves_through_a_file_and_comes_back_a_different_size`. Nine solid
pairs, three ops each; every result must either come back the size it left or
have the export say what it could not carry. Silence *and* a wrong answer
together is the single outcome ruled out — which is exactly what both bugs were,
and why nothing downstream could have noticed either.

    14 round-trip, 6 declined, 0 silently wrong

Verified the guard earns its keep rather than assuming it: disabled the pole's
contribution to the parameter extent and it failed; restored and it passed.
Restore checked by grep, not by report — pole block 1, stub 0.

The six declines are all the same remaining case, and it is worth stating what
it is now that the pole is done, because the two looked alike and are not. A cap
needed a *bound* that is not a curve, and AP203 had one. The complement of a cap
— a whole sphere minus a disk — needs no such thing: it is a disk too, bounded
by the same circle, and what tells the two apart is the direction the boundary
is walked. That is ordinary STEP and the file can hold it.

What cannot hold it is this crate's own `Face`, which states a region as
`u_range × v_range` plus trim loops, and the complement is not a rectangle in
either parameter. So the export could write those six correctly today and our
own importer still could not read them back — it sets `loops: None` always and
recovers the extent from edge vertices. The decline is therefore honest in both
directions for now, and the work it stands in for is on the *import* side:
giving a face read from a file the same ability to state a non-rectangular
region that a face out of the boolean already has.

That is the first time this thread has had a reason to change the importer
rather than the kernel, and it is a smaller, better-bounded change than the seam
arrangement it was previously waiting behind.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 10 step (+1), 639 lib.

## What the complement actually needs, worked out before building it

The plan was to lift the six declines by encoding the side in the boundary's
direction, which is ordinary STEP. Worked the semantics through before writing
it, and it is not the small change it looked like.

**The bit has no channel.** Our importer collects a face's edges from the
`ORIENTED_EDGE` references and ignores their orientation entirely; the only
direction it reads is `ADVANCED_FACE`'s own flag, which is the surface normal,
not the boundary walk. So carrying "which side" means adding a channel, not
using one.

**The pole is two different things depending on the answer.** This is the part
worth having found now rather than after shipping it. A degenerate `v_range` end
is a genuine 3D boundary when the face stops there — a cap — and is merely a
corner of the parameter rectangle when the face runs past it. A sphere minus a
side-on cap reaches `v = ±π/2` at both ends and *contains both poles in its
interior*. The `VERTEX_LOOP` emission added last tick keys on the degenerate end
alone, so lifting the decline without more would have written two bounds the
solid does not have — a new silent wrongness, in the code that fixed the last
one. It is safe today only because every such face is declined earlier. Recorded
that dependency in the code, where someone lifting the decline will read it.

**Self-consistency is the only check available.** Writing a file that is correct
for other systems but unreadable by our own importer cannot be verified here —
there is no second CAD system in this repo to open it. So export and import have
to land together, and the volume round trip is the test. That rules out the
tempting half-step of "write it correctly and let our importer decline".

So the remaining work is one change with three parts that must arrive at once:
a channel for the boundary direction, an importer that can state a
non-rectangular region (it sets `loops: None` unconditionally today), and a pole
rule that asks the trim loop rather than the range. Each is small; none is
independently testable. That is why it has not been done in a tick, and naming
it is better than a fourth attempt at guessing which third of it comes first.

Suite unchanged and green: 10 step, 32 boolean, 639 lib.

## The complement, carried — and a correction to the last entry

The three parts landed together, as required, and the channel turned out not to
need any new semantics. AP203 already separates `FACE_OUTER_BOUND` from
`FACE_BOUND`: the first is the edge of the face, the second is a hole. A face
with bounds but *no outer one* is therefore everything else on its surface —
which is exactly the complement, said in the standard's own vocabulary. Our
importer had been collapsing the two with `.or_else`.

  * export: a face with one ring off its edges and the rest on them is a
    complement; write its rings as plain `FACE_BOUND`s and no outer bound.
    Spheres only for now, since the reader has to know the surface's own extent
    to rebuild the region and that extent is hardcoded there.
  * import: a face with no outer bound gets the surface's whole outline as its
    outer ring and each bound as a hole, which is how the kernel's own booleans
    state such a face.
  * import: a ring that does *not* go the whole way round in `u` encloses a disk
    in the parameters, and that disk is the face — not the box around it.

**A correction.** Last entry said the pole emission was safe because every face
that could misfire declines earlier. That was wrong, and this tick found it by
measuring rather than by re-reading. A cap sitting *side-on* to the axis — the
small ball's cap in a union — has all its loop points on its edges, so it never
declined; and its `v_range` is the stale full-sphere one, so both ends looked
degenerate and it was written with two pole bounds it does not have. The
consequence was masked only because the *other* face on that solid declined and
took the whole file with it. Lifting that decline exposed it immediately:

    F1 out:  loops [99] area 7.62         (a disk)
    F1 back: loops None, u (-1.445,1.445) (its bounding box)

The rule is now the one the last entry said was needed: ask the loops, not the
range. A face that states loops is bounded by them, and its `v_range` ends are
corners of the parameter domain rather than places the solid stops.

    before this thread:  14 round-trip,  0 declined,  7 silently wrong
    after the declines:  14 round-trip,  6 declined,  1 silently wrong
    after the pole:      15 round-trip,  6 declined,  0 silently wrong
    now:                 18 round-trip,  3 declined,  0 silently wrong

The three that remain are a sphere cut by a box, where a face has more than one
ring off its edges — a region with several disjoint pieces of outline, which the
single-outline rebuild does not cover. Still named, not written.

`a_face_whose_edges_do_not_say_which_side_is_reported_not_written` pinned the
decline and so had to go; it is replaced by
`a_face_that_is_everything_except_its_own_boundary_survives`, which pins the
round trip *and* keeps a live decline case.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 10 step, 671 lib. wasm32 and
the example build.

## The last three declines are the seam, and the seam is already legal

Measured the three remaining declines rather than trusting the last entry's
description of them, which was wrong. A ball with a square column through it:

    F0 sphere edges=12 u=(1.643, 7.926) v=(-1.571, 1.571)
      ring0: 218 pts, 102 off-edge, area 14.908, u [1.643,7.926] v [-1.231,1.231]

**One** ring, not several. Its `u` span is 6.283 — a full turn. This is a *band*
between the column's two polar holes, walked as a single ring: along the top
hole, down the seam, along the bottom hole, back up the seam. The 102 off-edge
points are the two seam runs. Last entry called this "more than one ring off its
edges"; it is one ring whose seam runs are off its edges, which is a different
problem with a different fix.

And the fix is one this thread has already made legal. Three weeks of entries
went into the question of whether a seam can be an edge, ending in
`a_face_may_meet_itself_along_a_seam` — an edge used twice by one face, which is
how STEP writes a seam and how the manifold condition is actually stated. That
change was a no-op on every body the crate builds, and it was recorded as
enabling with nothing yet enabled. This is the thing it enables.

If the boolean materialised a face's seam runs as a real edge, that 218-point
ring would be entirely on edges. Export would write it as an ordinary
`EDGE_LOOP` with the seam edge appearing twice — standard practice, not a
special case — and import would chain it back the way it chains any other loop.
The `AmbiguousRegion` decline, the outline rebuild added last tick, and the
`loops`-versus-edges duality that has run through this whole thread all shrink
to the same one change: the boundary a face states should be made of edges,
including where it runs along a seam.

That is a change to `from_pieces`, which is the most delicate code here and has
cost a full tick twice when approached without a measurement in hand. This one
now has one. Not attempted at the end of a long tick; recorded as the next
piece, with the shape of the answer known rather than guessed.

Suite unchanged and green: 10 step, 32 boolean, 671 lib.

## The seam has an edge

Measured the ball-with-a-square-column face before touching anything, and the
answer was better than the plan assumed:

    218 pts, 102 off-edge, 2 runs
      run0: 51 pts, u 7.9264->7.9264, v -1.2272-> 1.2272, vertices 134..184
      run1: 51 pts, u 1.6433->1.6433, v  1.2272->-1.2272, vertices 184..134
      shared vertices between the two runs: 51 of 51 and 51
      run1 reversed == run0? true

The two seam runs are *the same vertices in opposite order*. The curve is
already there and already shared; only the `Edge` was missing. So
`materialise_seams` is a post-pass over the finished body: find each ring's
maximal runs of vertices no edge of that face names, pair the runs that are
exact reverses of one another, and give each pair one edge which the face then
names twice. The pairing is the whole test — a run that appears twice reversed
is a seam whatever surface it lies on — and a run that does not pair is left
alone, so nothing that declined before starts guessing.

An edge used twice by one face is precisely what `EdgeFaceCount` was changed to
allow several entries ago, recorded then as enabling with nothing yet enabled.
This is the thing it enables, and the whole suite agrees: 32 boolean, 21 brep,
13 brep_csg, 17 nurbs, 639 lib, all green with seams now real edges.

**And then it was worse, briefly.** With the ring wholly on edges, export
stopped declining those faces and wrote them — and the round trip came back
121.89 → 17.24. The file is not the problem; an `EDGE_LOOP` naming one
`EDGE_CURVE` twice is ordinary practice. The reader is: it collects a face's
edges as a *set*, and rebuilding a band needs the bound walked in order with `u`
unwrapped along the way, so the two passes along the seam land at opposite ends
of the parameter domain instead of on top of each other. Without that they
collapse and the band has no area.

Writing a file this crate misreads is worse than not writing one, and there is
no second system here to check the other reading against — so `SeamBoundary`
names it and export declines, honestly, for a reason that is about the reader
rather than the format. The corpus is back to 18 round-trip, 3 declined, 0
wrong, with the guard test holding the line that caught this in the first place.

    before this thread:  14 round-trip,  0 declined,  7 silently wrong
    now:                 18 round-trip,  3 declined,  0 silently wrong

Next is that reader, and it is now specified rather than described: keep each
bound's `ORIENTED_EDGE` order and orientation, walk it, unwrap `u` against the
previous point as `body.rs` already does for its own rings, and build the trim
loop from that instead of an extent.

(One repair along the way: the helper was inserted into the middle of
`parameter_outline`'s doc comment, splitting it in two. Clippy caught it —
"empty line after doc comment" — which is the only reason the halves were
noticed and rejoined.)

## The reader can walk a seam; the writer cannot yet write one

Built the reader the last entry specified: keep each bound's `ORIENTED_EDGE`
order and direction, walk it, and unwrap `u` against the point before, so the
two passes along a seam land a full turn apart instead of on top of each other.
Then dropped the `SeamBoundary` decline to see what happened.

    WRONG ball+box Union:      121.8885 -> 127.0803
    WRONG ball+box Difference:  89.8885 ->  16.0399

Better than the 17.24 of the last entry and still wrong, so the reader was not
the whole story. Comparing the rings either side said why in one line:

    OUT:  F0 sphere loops [(218,  14.908)]
    BACK: F0 sphere loops [(96, -0.0), (56, -0.558), (56, 0.558)]

Three rings where there was one. The fault is on the *writing* side, and it is
not the seam edge: `face_rings` builds a face's loops by chaining its edges into
closed rings on their own, with no reference to the ring the face actually
states. A boundary that walks the top hole, the seam, the bottom hole and the
seam again therefore comes out as three loops — the two holes, and the seam as a
loop of no area. Nothing downstream can rebuild a band from that, because the
band was never written.

So the decline stands, with its reason rewritten to say which side is missing:
rings have to come from the face's stored trim loops, in their order, rather
than from chaining its edges.

**The reader is kept, and tested.** It is not dead code waiting on the writer:
one face with one `EDGE_CURVE` named twice is how other systems write a closed
surface, and files like that arrive from outside. Since nothing this crate
exports looks like that yet, `a_face_that_meets_itself_along_a_seam_reads` is a
STEP file written by hand — a cylinder as a single face, up the seam, round the
top, down the same seam, round the bottom. It asserts the seam is named twice by
the one face and that the region has the area of the whole parameter rectangle,
2π × 2. Verified it earns its keep by removing the unwrap:

    region area 0, expected about 12.566 — did the two passes along the seam collapse?

Restore checked by grep. Corpus unchanged at 18 round-trip, 3 declined, 0 wrong.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 11 step (+1), 671 lib.

## The writer, attempted; and what the ring is actually made of

Built the writing side the last entry called for: for a face that names an edge
twice, take the ring the face already states and match its edges into it in that
order, rather than chaining edges into closed loops on their own. Two real
findings and a revert.

**The ring does not start where an edge does.** It starts wherever the
arrangement left it, which is generally partway along one:

    ring 218 pts; face has 14 edges
      step0 at 0: NO MATCH; ring[0..6] = [124, 134, 135, 136, 137, 138]

Vertex 124 is an endpoint of nothing. An `EDGE_LOOP` is cyclic, so the walk has
to begin at a vertex some edge really ends at. Fixed, and still no match.

**Because the ring is not a concatenation of its edges.** Counting settles it.
The face's twelve hole edges have lengths 8, 15, 8, 8, 15, 8, 9, 16, 8, 8, 15, 8,
and consecutive edges share a vertex the ring names once, so they supply
`Σ(len−1) = 114` steps. The seam is 51 vertices walked twice, supplying 100. That
is **214** against a ring of **218**. Four points in the ring come from no edge
of the face at all.

So the ring and the edges still disagree, in the same way and for the same
reason they have disagreed throughout this thread — only now the disagreement is
four points wide instead of a hundred and two, and it is *countable*. The seam
edge closed the large gap; something else opens a small one. Likely candidates
are the seam vertices `insert_seam_vertex` puts on a ring without giving them to
a curve, but that is a guess and the next tick's measurement, not a conclusion.

Reverted the writing-side walk: it never succeeded, so keeping it would be
untested complexity standing in for a decline that already reads correctly. The
decline's comment now carries the count, which is the part worth keeping.

**One bug caught in passing, worth recording because of how it presented.** The
first version guarded the old chaining with `if !chains.is_empty() { break }` —
but the chaining *also* pushes to `chains`, one entry per closed edge, so after
the first closed edge it broke out and dropped every remaining edge of the face.
Nine previously-correct results went wrong at once:

    WRONG plate+bore Union: 210.2026 -> 140.1549
    WRONG ball+bore Union:  132.2666 ->  12.5500
    EMPTY ball+bore Difference

The corpus guard caught all nine immediately, which is the second time it has
paid for itself. A flag taken *before* the loop fixed it.

Corpus unchanged at 18 round-trip, 3 declined, 0 wrong. Suite: 32 boolean,
21 brep, 13 brep_csg, 17 nurbs, 11 step, 671 lib.

## Not four missing points — two edges that cross the seam

Measured what the last entry guessed at, and the guess was wrong in a useful
direction. First:

    ring 218 pts, 0 not named by any edge of the face

*Every* ring vertex is on an edge. So "four points come from no edge" was an
error of arithmetic, not a fact about the data. What the count was really
detecting:

    edge 6 (len  9) is NOT a contiguous run of the ring: [62, 124]..[68, 54]
    edge 7 (len 16) is NOT a contiguous run of the ring: [53, 71]..[83, 23]

Two of fourteen edges are walked by the ring in two pieces rather than one. And
the reason is exact:

    vertex 124: uv (1.6433, -1.2300)   ring positions [0, 161]
       ring uv at 0:   [7.9264, -1.2300]      <- u₁
       ring uv at 161: [1.6433, -1.2300]      <- u₀
    face u_range (1.6433, 7.9264)

124 and 125 are the seam's own endpoints — where the boundary meets the seam —
and the ring names each twice, once at either end of the parameter domain, which
is right. Edge 6 runs `[62, 124, 63, …]`: it *crosses the seam*, at 124. In three
dimensions it is one continuous curve; in the parameters it is two runs on
opposite sides. The ring walks both. The edge is one.

So this is not the ring and the edges disagreeing. It is the seam split, which
is inherent to a closed surface, and the fix is the one every CAD system makes:
**an edge that crosses a face's seam is split at the seam vertex into two
edges.** The vertices are already there — 124 and 125 exist, and both edges
already contain them — so the split is a topological subdivision with no new
geometry and nothing to converge.

With that, every edge is a contiguous run of its ring, the walk built and
reverted last entry succeeds unchanged, and the writer can state a seamed face.
That is the next piece, and after a long series of entries circling "the loop and
the edge disagree", it is worth saying plainly that they no longer do: the
remaining difference is two edges that need cutting where they cross a line the
parameterisation draws.

Suite unchanged and green: 11 step, 32 boolean, 671 lib. Corpus 18 / 3 / 0.

## The seam closes

Three changes, each measured before it was made, and the thread's long-running
complaint is answered.

**1. The seam edge was a vertex short at each end.** `materialise_seams` built
it from the run of vertices *no edge names*, and the junctions where the seam
meets the holes are named — so the edge started one inside the boundary and
shared no endpoint with anything. That is what the 218-against-214 was: not four
missing points, but four steps of the ring that no edge covered, two at each
junction, once per pass. Taking each run together with the vertices either side
of it fixes it, and the extended runs are still exact reverses of one another,
so the pairing test is untouched.

**2. An edge that crosses the seam is cut at it.** `split_at_seam_ends` cuts at
the seam edge's own ends, which is where a boundary meets a seam by definition.
The vertices are already in both the edge and the ring, so this is subdivision
of topology with no new geometry.

    before: ring 218, edges 14, steps 214, non-contiguous [6, 7]
    after:  ring 218, edges 16, steps 218, non-contiguous []

The edges now tile the ring exactly.

**3. The writer reads the ring the face states**, rather than chaining edges
into closed loops of their own, and matches the edges into it in that order —
starting the walk at a vertex some edge really ends at, since the ring begins
partway along one and an `EDGE_LOOP` is cyclic. This is the code written and
reverted two entries ago, restored unchanged; it failed then only because the
edges did not yet tile the ring.

    18 round-trip, 3 declined, 0 wrong      (before)
    20 round-trip, 1 declined, 0 wrong      (now)

    before this thread: 14 round-trip, 0 declined, 7 silently wrong

`a_band_bounded_by_its_own_seam_survives_a_file_round_trip` pins it, and says in
its comment why all three had to be true at once. The one remaining decline is a
sphere-and-box *intersection*, still an honest `AmbiguousRegion`.

Worth stating what has actually been settled, because it ran through a great
many entries. A face's boundary is now made of edges — all of it, including
where it runs along a seam and where it crosses one. The trim loop is no longer
a second description that drifts; it is the same edges in an order. The
`loops`-versus-edges duality that blocked refinement, opened seams under
tessellation, and lost solids through STEP was, in the end, three missing pieces
of topology: a seam with no edge, a junction with no vertex shared, and a
crossing with no cut.

Suite: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step (+1), 671 lib. wasm32
and the example build; the one clippy line is the pre-existing `provenance.rs`.

## Refinement, retried on the new footing — and what it cost

With a face's boundary now made of edges, the oldest pinned decision in this
file was worth re-testing: `refine_edges` leaves a trimmed face's edges alone,
because the loop beside them used to be a fixed polyline that would drift. That
justification is gone. A loop is the face's edges in an order, so the ring can be
*rebuilt* from the same walk after the edges move, and there is nothing left to
drift apart.

The walk reads cleanly now — `ring_walks` succeeded on every face it was asked
about, which is itself the confirmation that the three seam changes did what
they claimed. But the obvious implementation, refine everything then rebuild,
costs:

    a_square_column_bored_through_a_ball              NotWatertight { open_edges: 1 }
    a_curve_is_found_on_a_face_whose_seam_has_moved   NotWatertight { open_edges: 1 }
    what_comes_out_is_a_solid_that_can_go_back_in     NotWatertight { open_edges: 1 }
    coverage 38 -> 36

One open edge in each — a single junction, not a boundary. And the reach is
wider than the change looks: the boolean calls `refine_edges` on its own inputs,
so refining trimmed faces changes what the kernel *classifies against*, not only
what a caller tessellates. That is why one junction is enough to turn a resolved
operation into a decline.

Reverted, verified by grep (`let pinned` back, `ring_walks` gone). The pinned
comment now records the attempt and its numbers rather than the old reasoning,
which no longer applies.

**Chaining, on a fresh corpus:** seven solids cut twice, both ops — 8 of 12
second cuts are valid solids. Not comparable to the "6 of 14" recorded earlier;
those were different cases, and this is a new baseline rather than an
improvement. What fails:

    ball: bore then column   second cut not a solid / NotWatertight { open_edges: 3 }
    ball: ball then bore     NeedsArrangement { face: 1 } / NotWatertight { open_edges: 130 }

Both are spheres cut twice, which is where the seam work lives, so they are the
right next thing to look at with the new topology in hand.

Suite green and unchanged: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## The two halves fail on different cases

Went looking for the single junction the last entry left open, and found
something better by separating the change into its two halves and measuring each
alone.

**Unpinning alone** — refine every edge, leave the rings as they are:

    a_cylinder_parked_on_an_edge          a_bore_across_a_rod
    coverage 38 -> 35

**Unpinning with the rings rebuilt from their walks:**

    a_square_column_bored_through_a_ball  a_curve_is_found_on_a_face_whose_seam_has_moved
    coverage 38 -> 36

The two lists are **disjoint**. The rebuild is not a wrong idea that costs three
tests; it *repairs* both cases the bare unpinning breaks, and breaks two others
that the bare unpinning leaves alone. Something in it is right and something is
missing.

And the second list names where to look. `a_square_column_bored_through_a_ball`
is the seamed band — the one face in this suite whose ring walks a single edge
twice — so the rebuild's handling of a repeated edge is the first suspect. The
walk records `(edge, forward)` and concatenates, skipping a vertex that repeats
the previous one; for an edge used twice, the second pass starts at the vertex
the first ended on, and that skip may be eating a junction that the ring needs.
One open edge in each failure is consistent with exactly that.

Also worth recording, because it was a surprise: with refinement unpinned and
the rings left stale, the ball-and-column result still tessellates **closed** —
121782 triangles, 0 boundary edges. A stale ring beside a refined edge does not
open a seam on its own here, which is a weaker claim than the pin's original
justification made.

Reverted to pinned, restore verified by grep. The pinned comment now carries
both failure lists rather than one, since the difference between them is the
finding.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The walk is exact; the anchor was not; neither is the blocker

Chased the suspect the last entry named — that the rebuild's concatenation eats
a junction where a ring walks one edge twice — and it is innocent. Measured
without changing any source, by walking each ring of a bored ball and
concatenating it straight back:

    F0 ring0: 16 edges, seam true,  rebuilt 218 vs ring 218 — IDENTICAL
    F1 ring0:  5 edges, seam false, rebuilt  30 vs ring  30 — IDENTICAL
    …

Every ring, seamed one included. So the vertex walk is exact and the hypothesis
was wrong.

**The parameters were not**, and that is a real bug found:

    uv worst difference 6.283185 (first at 0)
      rebuilt [[1.6433, -1.2300], …]   want [[7.9264, -1.2300], …]

Exactly 2π, from the first point. The rebuild started with no anchor, so the
first vertex took whatever `invert` returns — the canonical `u₀` — and the whole
first pass along the seam followed a full turn from where the ring puts it. A
seam's two passes are the *same points* at opposite ends of the domain, so the
canonical parameter is a turn away from one of them by construction, and a
rebuild that chooses for itself moves the ring rather than refining it.

Seeding the anchor from the ring's own first parameter fixes it, and gives the
rebuild the property it should have had from the start: with nothing refined it
reproduces every ring **exactly**, vertices and parameters, worst difference
0.000000. A rebuild that is an identity before refinement can only add points
after it.

**And it is still not what the two tests are failing on.** With the anchor fixed
the same two fail the same way, coverage 36. So there is a third thing, and the
two lists from the last entry still bound it: it lives in what the rebuild does
that leaving the rings alone does not, and it shows up only after edges actually
move.

Reverted to pinned; restore verified by grep. The pinned comment now records the
walk's exactness, the anchor bug and its fix, and that neither closes it — which
is three fewer blind alleys for the next attempt than the last entry left.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Green, and reverted anyway

The last entry's two disjoint failure lists suggested a split, and both failures
the rebuild caused are on seam faces — `a_square_column_bored_through_a_ball` is
the seamed band itself. So: rebuild every face except those that walk an edge
twice, and pin the seamed ones.

    32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib — all green

The split is real. The rebuild is right for faces that do not run along a seam,
and only seamed ones break, which is a much smaller and better-aimed question
than "refinement breaks three tests".

**And it is reverted, because it improves nothing measurable.** The pin exists
so that a trimmed face keeps a boundary its loop still describes; the cost was
supposed to be that such a face cannot refine. Measured, that cost is not being
paid by anything:

    ball bored:  no loops at all — the pin never touched it
    plate bored: no loops at all
    ball+ball:   loop sizes [64, 32] and [32, 32], longest chord 0.39207
                 — identical at 1e-2, 1e-3 and 1e-4

The bored solids state no rings, so they were never pinned. The two balls do,
but their ring is the *parameter outline* — seam and poles — which no edge
carries, so the walk fails and their edges stay pinned regardless. What is left
walks cleanly and is planar, where subdividing a straight edge adds nothing.

So the obstacle is not the pin any more. It is that a complement face's outline
is the last part of a boundary still not made of edges — the same gap the seam
work closed for bands, left open for complements, where the "seam" runs pole to
pole through a face with nothing on the other side of it. Until that is closed,
unpinning buys a precision no face can spend, and shipping it would be untested
reach into the kernel's own closure gate for no gain.

`materialise_seams` pairs runs that appear twice reversed, and a complement's
outline is a single run that never pairs, which is exactly why it was left
alone. Making that ring out of edges is the next piece, and it is the same shape
as the work already done — the vertices exist and are welded; only the topology
is missing.

Reverted, verified by grep. The pinned comment now records what the rebuild
achieves, what it costs, and why neither is worth taking yet.

## The right criterion, waiting on the reader

The last entry named the obstacle: a complement face's outline is the last part
of a boundary not made of edges, and `materialise_seams` skips it because the
whole ring is off the edges and there is no on/off boundary to start a run from.
Measured that ring rather than assuming its shape:

    F0 ring0: 128 pts, 65 distinct, 63 named twice
      pos 1 u 9.4248 (u₁)  also at 127 u 3.1416 (u₀)
      pos 2 u 9.4248       also at 126 u 3.1416

It is the band's structure exactly, with poles where the band had hole edges:
pole, up the seam at `u₁`, pole, down the same seam at `u₀`. Positions 0 and 64
are the poles, named once because every parameter at a pole is the same point.

So "carried by no edge" was never the property. **A vertex the ring names twice
is on a seam** — the ring passes it once at each end of the domain — and that
holds whether or not an edge already carries it. It subsumes the existing rule:
the band's junctions are named twice too, which is why they were already inside
its runs.

Switched to it, with the run reaching out to a neighbour only where that
neighbour is on no edge (a pole), since where it is on one the junction is
already inside the run. The kernel result is what it should be:

    ball+ball: 2 faces 3 edges, valid true
      F0 ring0: 128 pts, 0 off-edge     F0 ring1: 99 pts, 0 off-edge
    ball+box:  5 faces 19 edges, valid true
      F0 ring0: 218 pts, 0 off-edge

Every ring of both is now made of edges, both are valid solids, and a sphere cut
by a box stops declining on export — 0 declines across the corpus for the first
time.

**Reverted, because the reader cannot take it.** Four results come back open:

    ball+ball Union 179, Difference 230, Intersection 151, ball+box Intersection 99

These rings run pole to pole, and at a pole every parameter names the same
point, so the importer's walk — which unwraps `u` against the point before — has
nothing to unwrap against and places the two passes on top of each other. That
is the same collapse the seamed *band* had, in the one place the fix for it does
not reach.

So the criterion is right and the kernel change is ready; what is missing is a
rule for the pole on the reading side. The obvious one is that a bound's pass
through a degenerate point should carry `u` across rather than re-derive it —
the walk already knows which side it came from.

Reverted, verified by grep. The comment in `materialise_seams` now records the
better criterion, what it achieves, and the four numbers that say why it waits.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The pole rule helps and does not finish

Took the reader's pole rule the last entry proposed: at a degenerate point every
`u` names the same place, so what `invert` returns there is arbitrary, and
unwrapping it means unwrapping noise. The walk knows which side it arrived from,
so carry that `u` across instead of re-deriving it.

    open edges on import
      before the rule:  ball+ball 179, 230, 151   ball+box 99
      after:            ball+ball  99,   0,  99   ball+box 69
      round-trip:       11 -> 17 of the corpus

A real improvement, and not enough: four results still fail, one of them by
tessellating to *nothing* — zero triangles with zero open edges, which is the
vacuous closure the corpus guard was given `triangles > 0` to catch.

Also tried starting the ring away from a pole, on the reasoning that a ring
beginning at a degenerate point begins with an arbitrary parameter that
everything after is carried from. The ring is cyclic so this is free, and it
changed **nothing** — same four, same counts. So the fault is not the pole's
parameter, and not where the walk starts. Something past both is still wrong.

All three changes reverted — kernel criterion, pole carry, rotation — verified
by grep. Baseline holds at 20 round-trip, 1 declined, 0 wrong; the criterion
would give 0 declined at the cost of 4 wrong, and wrong is the thing this thread
has spent its time removing.

What is now known about the remaining fault, which is more than the last entry
had: it survives a correct pole parameter, it survives a well-chosen starting
point, and one of its symptoms is an empty tessellation rather than a torn one.
An empty result from a ring that has points in it suggests the ring is being
*rejected* downstream rather than mis-sampled — a degenerate area, a winding
that came out the wrong way, or a hole that swallowed its outer. That is a
different place to look than the parameters, and it is where the next tick
should start.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Why unwrapping cannot tell a complement's two passes apart

Last entry left the fault "past the pole and past the starting point". Measured
the rings either side of the file rather than guessing again:

    OUT:  F0 rings [128 pts/+19.431, 99 pts/-1.680]
    BACK: F0 rings [122 pts/ -0.000, 99 pts/+1.680]

The outline came back with **no area** and the hole with its sign flipped. So
the two passes along the seam did land on top of each other, pole rule and all.

And there is a reason unwrapping *cannot* separate them. It works on a band
because the two passes are far apart in the walk: the holes between them turn a
whole circle, so by the time the walk comes back the anchor has drifted a period
and the nearest representative is the far one. On a whole sphere less a disk the
two passes are **adjacent** — only a pole between them — nothing has drifted, and
the nearest representative is the same one. The information that separates them
is not in the geometry at all.

What is: a seam edge's two uses are one period apart by definition. Placing the
second a period from the first fixes the collapse, with two corrections found by
measuring:

  * **once, not per vertex.** Applied per vertex it compounds and `u` walks off
    by a period a point — the ring came back spanning -373 to 3.
  * **as a fallback, not a maximum.** Choosing whichever variant encloses most is
    wrong: a band is already told apart by its drift, and a shifted version of it
    encloses more while describing a different solid — 233 where the answer is
    121. Take the plain ring unless it came out with no area.

    11 -> 17 -> 16 -> 18 round-trip as those went in and were corrected

**Reverted at 18 round-trip, 0 declined, 3 wrong**, against a baseline of 20 / 1
/ 0. The mechanism is right and the ring areas now come back correct — F0 at
19.416 against 19.431 — but three results still arrive open (151, 151, 69). With
the areas right, that is no longer a parameter problem, and the point counts say
where to look next: 122 and 100 points back from 128 out. At a pole many
parameters name one point, so they weld to one vertex and the walk drops the
repeats; that is correct in itself, but it means the ring no longer has a point
where the *other* face's boundary has one, and a junction that does not line up
is exactly an open edge.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Consolidation: does what this thread claims still hold?

A tick spent checking the work rather than extending it, since the STEP thread
has been reverting more than it lands and the claims have moved a long way.

The end-to-end example runs and every number in it is right:

    plate 3840.000 -> bored three times -> 3604.504, expected 3604.381
    re-tessellated at 1e-3 / 1e-4 / 1e-5: 1368 / 5400 / 21528 triangles, all watertight
    STEP: 9 faces, 18 edges, all analytic
    read back: 9 faces, 3 cylindrical surfaces, volume 3604.504
    drilled one more bore into the imported solid: 10 faces, volume 3576.254
    cross-hole declined: NotWatertight { open_edges: 160 }

Including the last line, which the example's own comment predicts: the cross-hole
is traced, not refused, so what declines is the assembly and the reason names it.

The README's B-rep decline list is accurate. Its STEP line was accurate but
weak — "anything unmappable is reported" was true before this thread too, while
being false in practice twice over. Replaced with the property that is now
actually tested: a corpus is written, read back and re-measured, and each result
must either come back the size it left or have the export name what it could not
carry. Both declines are named, and the ruled-out outcome — silence together
with a wrong answer — is stated, because that is what both bugs were.

The `step` module doc said the same thing about *surfaces* only. A face can be
unstateable too, which is less obvious and was found the hard way twice, so it
now says that as well, with both reasons and the fact that `skipped` was empty at
both ends when the rule was broken.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Why a bore cut twice is not a solid

Switched axis to the kernel's chaining failures, and the first one opened
cleanly. A ball with a bore, cut again by a column:

    second: 14 faces, valid false, tris 145562 open 0, defects 2
        EdgeFaceCount { edge: 16, uses: 6 }
        EdgeFaceCount { edge: 17, uses: 6 }

It tessellates *closed* and is still not a solid — the bookkeeping is wrong, not
the geometry. Edges 16 and 17 are the original bore's two rim circles, and the
wall they belong to came apart into five pieces, each claiming a whole rim:

    F1..F5 cylinder edges=[16, 17]

There is machinery for exactly this — cut a rim when more than two faces claim
it — and it never fires. Two guesses at why, both wrong and both cheap to
disprove:

  * that `run_of` bails because a wall piece states no loops. It does bail on
    that, but not here: every one of the six faces has loops.
  * that the sphere covers the whole rim while the pieces claim parts, so no
    stretch has exactly two claimants. It does not cover the whole rim.

Instrumented instead of guessing again, and the answer is neither:

    edge 16: 6 claims, 82 verts
      face 0 (sphere)   run (11, 31)
      face 1 (cylinder) run (41, 51)    face 2 (cylinder) run (53, 71)
      face 3 (cylinder) run (0, 10)     face 4 (cylinder) run (11, 30)
      face 5 (cylinder) run (32, 42)

Six faces, six stretches, **overlapping but unequal**. The sphere's (11, 31) and
the wall's (11, 30) are plainly the same arc of the rim, and the grouping keys on
the exact pair of ends, so they land in different groups, every group holds one
face, and the guard "every stretch claimed by exactly two faces" is never
satisfied. The rim cut is not disabled by a missing case; it is disabled by
equality where it needs overlap.

That is the fix — cluster the runs by overlap, then cut — and it is a contained
one, in a block that already has the plan/apply structure to hang it on. Both
guesses reverted; the comment in the block now carries the six runs, so the next
attempt starts from the measurement rather than from the same two hypotheses.

Suite green: 32 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The rim cut, specified

Last entry said the rim cut fails because it groups by equality where it needs
overlap. That was right about the symptom and wrong about the fix, and one more
measurement — every run per face, not just the longest — says why:

    rim of 82 vertices, six faces
      sphere    (0,10) (11,31) (32,51) (52,71) (72,82)
      wall 1    (41,51)          wall 2  (53,71)
      wall 3    (0,10) (72,82)   wall 4  (11,30)
      wall 5    (32,42)

Two things, not one.

**`run_of` keeps only the longest run.** The sphere borders four arcs of this rim
— `(72,82)`+`(0,10)` is one arc wrapped, which wall 3 correctly claims as both —
and all but `(11,31)` are thrown away before the grouping ever sees them. That
alone guarantees singleton groups.

**And the ends do not agree even with every run in hand.** The sphere's `(11,31)`
and wall 4's `(11,30)` are one arc, so equality fails; but its `(32,51)` spans
*two* wall pieces, `(32,42)` and `(41,51)`, so that stretch has three faces on it
and overlap-grouping would put all three together. Neither key works.

So the rim has to be cut at the **union of every run's ends**, not matched
between faces. And the one-vertex stretches between runs — `(10,11)`, `(30,31)`,
`(31,32)`, `(41,42)`, `(51,52)`, `(71,72)` — are where the column removed the rim
altogether: they belong to no face and have to be *dropped* rather than assigned,
which means the rim edge gets shorter, not merely divided.

That last part is why this is worth specifying before writing: the existing block
divides an edge among faces and never shortens one, so the fix is not a change of
grouping key but a change to what the operation does. Recorded in the block, with
the six faces' runs, so the next attempt starts from the data.

Instrumentation removed, verified by grep. Suite green: 32 boolean, 21 brep,
13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The rim cut, built

Implemented what the last entry specified, and it needed one more measurement
first — the per-vertex containment, which turned out far cleaner than the runs
suggested:

    face 0 (sphere) ##########.####################.###################.###################.#####…
    face 3          ##########..............................................................#####…
    face 4          ...........###################....................................................
    face 5          ................................##########........................................
    face 1          .........................................##########...............................
    face 2          .....................................................##################...........

The sphere reaches every vertex but four — 10, 31, 51 and 71, the column's four
corners — and each wall piece reaches one arc. Only three vertices are awkward:
30 and 52 reached by the sphere alone, 41 by three faces, all within a vertex of
a boundary, which is what a containment test a tolerance wide will do.

So the rule is per vertex, not per run: **a vertex reached by exactly two faces
belongs to the stretch they share; anything else carries the label of the vertex
before it.** The corners and the fuzz both fall out of that without a special
case, every vertex is used, and each stretch of one label becomes an edge those
two faces share. `run_of` became `hits_of`, returning the row rather than its
longest run.

Two corrections found by building it:

  * a stretch has to end on the *next* stretch's first vertex, or consecutive
    pieces of a rim do not meet. The walk is modular because one stretch runs
    through the rim's ends.
  * a face on several stretches names the rim once, so after the first takes its
    slot the rest are appended. The sphere borders every piece. Without this the
    six-face edge became six one-face edges — which is how it first came out.

    before: EdgeFaceCount { edge: 16, uses: 6 } and the same for 17
    after:  no EdgeFaceCount defects at all

**And the result is still not a solid**, because its shell does not close —
every face reports free ends, including the four planes the rim cut never
touches, so that fault is older and separate. Chaining is unchanged at 8 of 12.

Kept anyway, which is a departure from "revert what fixes nothing", and the
reason is that this fixes something measured: an edge shared by six faces is not
a boundary between two faces, and the topology was wrong whatever the shell did.
`a_rim_cut_a_second_time_is_shared_by_two_faces_at_a_time` pins it, and says in
its own text that the shell is a separate fault — so the day that closes, the
test is what will say so.

Suite: 33 boolean (+1), 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32
and the example build.

## The shell is open because a whole class of edge was never made

Followed the open shell the last entry set aside. Every face of the twice-cut
ball has free ends, and the first plane face says where they are:

    F6 plane edges [2, 3, 16]: v6 -e2- v8 -e16- v9 -e3- v11, open at v6 and v11
      v6  at [0.7500,  0.6614, -2.8284]
      v11 at [0.7500,  0.6614,  2.8284]

Those points are 1.0 from the axis and 3.0 from the origin: they are on the
bore's wall *and* on the sphere — the rim. So what is missing from this face is
the boundary between v6 and v11, which is where the column's plane cuts the
bore's wall. Counting the kinds of edge in the result says it plainly:

    cylinder/sphere 14   plane/plane 4   plane/sphere 16   sphere/sphere 1

**No plane/cylinder edges. Not one.** The wall came apart into five pieces, so
those curves were certainly found and used to trim; but no edge was made for any
of them. Each wall piece is bounded only by rim arcs and reports four free ends,
having neither of its vertical sides; each plane piece is missing the same curve
from the other direction.

The comparison confirms it is specific to chaining rather than to the shapes: the
ball cut by the same column *directly* is 5 faces, 19 edges, **valid**, with its
four plane/plane corners present and every face closed. Cutting the bore first
is what loses them.

So the fault is not in the arrangement, the rim, or the loops — those are now
right, and the last entry's rim cut removed the last defect of that kind. It is
that an intersection curve between a face of the first solid and a face of the
second produced a trim and no edge. That is a different part of `from_pieces`
than anything this thread has touched, and it is the whole reason a bore cannot
be cut twice.

Chaining stays 8 of 12; suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs,
12 step, 639 lib.

## No plane-against-cylinder edges because the arrangement forgets the chords

Traced the missing edge class back through `from_pieces`. An `Edge` is made from
a piece's `bounding`, and `bounding` is `along` — the curves the subdivision
recorded as sources for the points of the piece's outline. So the question is
what the arrangement says about the bore wall, and it says nothing:

    face (cylinder) region: along []  sources boundary=20 crossing=0 chord=0  chords.len=8
    face (cylinder) region: along []  sources boundary=8  crossing=0 chord=0  chords.len=8
    …nine regions, every one the same…
    face (plane)    region: along [4,5,6,7]  sources boundary=4 crossing=8 chord=4  chords.len=6
    face (plane)    region: along [4,5]      sources boundary=0 crossing=4 chord=2  chords.len=6

Eight chords are passed for the cylinder — the four column planes cut it in two
lines each — and they plainly work: the wall comes apart into nine regions,
which is four surviving arcs plus the seam splits. But every point of every
region's outline is recorded as `Boundary`, so `along` is empty, no edge is made,
and the wall pieces end up bounded by rim arcs alone with four free ends each.
The column's *own* plane faces, going through the same code with six chords,
record theirs normally.

So the fault is not that the intersection was missed, nor that the pieces are
wrong. The pieces are right. What is lost is the *provenance* of their
boundaries, and only for this face. That is a much narrower thing than the last
entry could say, and it is the whole reason a bore cannot be cut a second time.

Where to look next, in order: whether the cylinder's chords reach `subdivide` as
chords or as bare paths (they are counted as chords, so probably not it);
whether a chord that runs boundary-to-boundary across a *periodic* face is
treated as part of the boundary; and whether the seam handling rebuilds an
outline and drops the sources with it — the wall is the one face here that both
wraps and is cut, and nine regions from eight chords means the seam is involved.

Recorded at the `along` site so the next attempt starts from the counts. Suite
green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A straight curve left no trace, so it got no edge

The last entry narrowed the missing edges to lost provenance. The cause is one
line of the arrangement's design, and the counts name it exactly:

    face (sphere):   chord point counts [3, 3, 3, … ]        chord sources > 0
    face (cylinder): chord point counts [2, 2, 2, 2, 2, 2, 2, 2]   chord sources 0
    face (plane):    chord point counts [3, 3, 3, 3, 2, 2]   chord sources 4, not 6

`Source::Chord` is attached to a chord's *interior* points — the ones strictly
between its ends — because its ends are nodes shared with whatever they land on.
A curve with only two samples has no interior, so it leaves no trace at all, and
`from_pieces` makes no `Edge` for it.

And two samples is not a degenerate case. **A plane meets a cylinder in a
straight line, and two points describe a line exactly.** So every
plane-against-cylinder curve in the model was invisible to the thing that builds
edges — which is why a bore cut a second time had none of them, from either
side: the wall's eight chords all had two points, and the planes' six chords
contributed four, the two missing ones being the same lines seen from the other
face.

Fixed where the curves are finished rather than in the arrangement: a curve with
two vertices and no closure is given a midpoint. That is exact rather than a
refinement — the sampler stopping at two points is its statement that the curve
is straight to within tolerance.

    before: cylinder/plane 0   edges 35   free ends on 14 of 14 faces
    after:  cylinder/plane 8   edges 43   free ends on  4 of 14 faces, 0 defects

Ten of the fourteen faces now close where none did. The result is still not a
solid and chaining is unchanged at 8 of 12, so this is not the end of it — but
`a_rim_cut_a_second_time_is_shared_by_two_faces_at_a_time` now also asserts the
eight wall cuts exist, and asserts the body is *not* yet valid, so the day the
remaining four faces close, the test says the plan is out of date.

Suite: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32 and
the example build.

## An edge from a vertex to itself

With the wall cuts made, four of the fourteen faces still had free ends. Their
vertices are all on the rim — r = 1.0 from the axis and 3.0 from the origin — and
one edge in the list says what went wrong:

    e38 sphere/cylinder 2 verts 72..72

An edge from a vertex to itself. The rim cut built it: a closed rim repeats its
first vertex last, and that repeat is not a position of its own, so a stretch
running across it takes `vs[n-1]` and `vs[0]` as its two ends — which are the
same vertex. The stretches either side then have nothing to meet at.

Fixed by counting the rim's positions without the repeat, in both the labelling
and the part builder. Two degenerate edges become none, 43 edges become 41, and
the suite is unchanged.

**It is not what the free ends were.** They are exactly as before — [4, 2, 0, 4,
0, 2, …] — so the zero-length edges were a second fault sitting in the same
place, real but not this one. Worth having found separately: an edge from a
vertex to itself would have gone on producing wrong answers quietly once the
first fault was fixed.

`a_rim_cut_a_second_time_is_shared_by_two_faces_at_a_time` now also asserts no
edge names one vertex twice.

Suite: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The wall pieces are missing the seam between them

With the wall cuts made and the zero-length edges gone, four faces still do not
close. What they hold says why:

    F1 cylinder free=2 edges=[16, 30, 35]
       e16 cylinder/plane  42..47      e30 sphere/cylinder 110..42
       e35 sphere/cylinder 183..47     odd [110, 183]

    F3 cylinder free=4 edges=[19, 20, 32, 37]
       e19 cylinder/plane  35..30      e20 cylinder/plane 24..29
       e32 sphere/cylinder 35..72      e37 sphere/cylinder 30..145
       odd [24, 29, 72, 145]

F1 chains `110 → 42 → 47 → 183` and stops: it has **one** cylinder/plane line
where an arc of the wall needs two, and nothing joins 183 back to 110. F3 has the
opposite problem — `e20` connects to none of its other three edges at all.

The arithmetic says where the missing boundary is. Four column planes cut the
cylinder in two lines each, so eight lines, and eight `cylinder/plane` edges
exist. Those eight bound the four *surviving* arcs, two apiece, exactly. But
there are five wall pieces, not four: the parameter seam splits one arc in two,
and the boundary between those two halves is not a line, not a rim, and has no
edge. F1 is one of that pair, open at both ends of the seam.

So this is the same shortfall the STEP thread spent its ticks on — a face split
at the seam of its own parameterisation, with nothing to hold the halves
together — but inside the kernel this time, and reached from the other
direction. The seam-edge machinery added earlier (`materialise_seams`,
`split_at_seam_ends`) works from a face's *stated loops*; these pieces state
loops too, so the next thing to measure is why their seam runs are not being
paired there.

That F3 holds a line belonging to a different piece is a separate thread to pull,
and is worth keeping distinct from the missing seam rather than assuming one
explains the other — the last two entries were each a second fault sitting in the
same place as the first.

Suite green and unchanged: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## The unowned runs are not the missing boundary

Last entry guessed the wall halves were missing the seam between them, and that
`materialise_seams` could not see it because it pairs a run with *itself* — which
is a seam when one face walks it twice, and not what happens when two faces each
walk it once. Measured the unowned runs of every face to check:

    F0 sphere   off-edge runs [(8 verts, 73..80), (8 verts, 153..146)]
    F1 cylinder off-edge runs [(2 verts, 182..109)]
    F3 cylinder off-edge runs [(8 verts, 73..80), (8 verts, 153..146)]

F0 and F3 hold the *same two runs*, in the same order — so a cross-face pairing
that accepts either direction would match them, which the within-face test never
could. Built that, and it is **wrong**:

    free ends before [4, 2, 0, 4, 0, 2, …]
    free ends after  [8, 2, 0, 8, 0, 2, …]

Worse, and informatively so. Giving those runs an edge adds their *endpoints* as
odd-degree vertices on both faces, which can only happen if the runs do not meet
those faces' other edges. A stretch that is genuinely a shared boundary would
have closed two chains; this one opened four. So the runs are not the missing
boundary — they are something else lying loose in the same rings, and the
missing boundary is still missing.

Reverted, verified by grep. The negative is worth as much as the guess was: two
faces holding the same vertices in the same order is not sufficient evidence that
those vertices are their common edge, and the free-end count says so immediately.

F1's run of two vertices, `182..109`, remains unexplained and is the smaller and
more tractable of the two: two vertices, one face, and F1 is exactly the piece
that is open at both ends of the seam.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Two wall pieces meet on a line only one of them names

Pulled the smaller thread — F1's unowned run of two vertices — and it is not a
missing edge at all.

    F1 ring 21: .###################.
       [109, 110, …, 42, 71, 47, 190, …, 183, 182]
    F5 ring 21: #####################
       [41, 101, …, 108, 109, 182, 181, …, 36, 70]

    v109 [0.0000, -1.0000, -2.8284]   r = 1.0000
    v182 [0.0000, -1.0000,  2.8284]   r = 1.0000

Only F1's *first and last* vertices are unnamed, and they wrap into each other:
the run is the pair `182, 109`. Those two points have the same `x` and `y` at
radius 1 and opposite `z` — a vertical line up the bore's wall at (0, −1). F5's
ring holds the same pair, adjacent and named, so **the edge exists and F5 has
it; F1 simply does not claim it.**

That reframes the fault. The last two entries were looking for a boundary to
*build*; this one needs a boundary to be *claimed*. Which also explains why
building it made things worse: a second edge over the same line would have given
those vertices another degree rather than closing anything.

It also explains the shape of the earlier failure. The line at (0, −1) is not a
column plane — the column's planes are at ±0.75 and this is at 1.0 — so it is
where the arrangement cut the wall for its own reasons, and only the piece on one
side recorded it. Whatever assigns an edge to the faces either side of it is
seeing one of them.

Not attempted this tick: matching a face's unowned run against the edges that
already exist is a different operation from creating one, safer, and worth doing
deliberately rather than at the end of a long tick. The run to match is the
unowned pair *extended to its named neighbours* — `183, 182, 109, 110` — since
that is the stretch the boundary actually walks.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A rim stretch that straddles two pieces

The last entry said F5 names the line F1 does not. That was the wrong reading of
the right data. What F5 actually names is two *rim* stretches:

    e29: [41, 101, …, 108, 109, 110]   users [0, 5]
    e34: [36, 174, …, 181, 182, 183]   users [0, 5]

and its ring walks them only as far as 109 and 182 — while F1's ring *starts* at
109 and *ends* at 182. So each of these edges covers boundary belonging to two
different wall pieces, and only one of them is credited with it. F1's two
"unowned" vertices are the far ends of stretches it shares.

That is the rim cut from three entries ago, one vertex off. At 109 three faces
meet — the sphere, F5 and F1 — so `pair_at` finds no pair, the carry-forward rule
gives it the label before it, and it lands inside F5's stretch instead of ending
it. The same at 182.

Which sharpens the rule that entry settled on. Carrying a vertex that names no
pair is right for *fuzz* — a boundary the tolerance-wide containment test blurs
by a vertex, where the pairs either side are the same — and wrong for a
*junction*, where three faces meet because the stretch genuinely changes hands
there. The two look identical to `pair_at` and differ in what follows them: at a
junction the pair after is not the pair before.

So the fix is to end a stretch at a three-face vertex whose neighbours disagree,
rather than to carry it. That is a small change to code this thread already
wrote, in a place it already understands, and it is the fourth distinct fault
found in this one chained boolean — six-face rims, missing plane/cylinder edges,
zero-length edges, and now a stretch that changes hands mid-way. Each was real
and none was the last one.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A bore can be cut twice

Two changes closed it, and the second was only findable because the first moved
the fault into view.

**The junction rule.** A vertex on a rim that names no pair of faces is either
fuzz — the containment test is a tolerance wide — or a place three faces meet and
the stretch changes hands. Carrying the previous label is right for the first and
wrong for the second. What separates them is what comes *next*: at a junction the
pair after is not the pair before. Measured, index 41 of both rims is a three-face
vertex, and with the rule in place the stretches move:

    e29 [41 … 110]  ->  [41 … 109]      e30 now [109 … 42]
    e34 [36 … 183]  ->  [36 … 182]      e35 now [182 … 47]

They meet where they should. On its own it changed no free-end count, which is
what made the remaining fault legible:

    F1 chain: 109 -e30- 42 -e16- 47 -e35- 182     open at 109 and 182
    F5 chain: 109 -e29- 41 -e23- 36 -e34- 182     open at 109 and 182

**Both** halves of the wall arc open at the *same two vertices*, needing the same
boundary — the line at (0, −1) up the bore. And now the reason two entries of
searching missed it: both its ends are vertices of other edges, so nothing is
unnamed. The gap is a **segment**, not a vertex. `materialise_seams` looks for
vertices no edge names and structurally cannot see it.

So `materialise_shared` asks the other question: which consecutive pair of a
ring is walked by no edge anywhere, gathered into runs and paired across faces in
either direction, since which way two faces walk a shared boundary is a matter of
orientation.

    second: 14 faces 44 edges valid true defects 0 free [0 × 14]

    chaining 8 of 12 -> 9 of 12

A ball with a bore, cut again by a column, is a solid: one closed shell, every
edge between exactly two faces, ready to be cut a third time. Four faults stood
in the way and each was real — a rim six faces claimed, a class of edge never made
because a straight curve leaves no trace where provenance lives, an edge from a
vertex to itself across a closed rim's repeat, and a stretch that changed hands at
a junction.

`a_rim_cut_a_second_time_is_shared_by_two_faces_at_a_time` asserted the body was
*not* valid, so that the day it became one the test would say the plan was out of
date. It did, in those words. It now asserts the solid.

Suite: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32 and
the example build; clippy unchanged.

## Where the remaining chained cases stand

With a bore cutting twice, three of twelve chained cases still fail. Measured
each far enough to say what it is rather than that it fails.

**`ball: bore then column` (Union), `NotWatertight { open_edges: 3 }`.** The input
is sound — the union of a ball and a long cylinder is 5 faces, 4 edges, valid,
and tessellates closed with no open edges — so the fault is in the second cut,
and three open edges is close. Same shapes as the case that now works; the
difference is that the cylinder protrudes past the ball at both ends, so the
column crosses surfaces the difference case never presented.

**`ball: ball then bore` (both ops), `NeedsArrangement` and 130 open edges.** The
first result is two sphere faces, each stating

    loops [128, 99]   u (π, 3π)   v (−π/2, π/2)   1 edge

— the 128-point *parameter outline*, seam and poles, with the 99-point
intersection circle as a hole. That is the complement shape the STEP thread spent
many entries on, and it is the one kind of face whose boundary is still not made
of edges: one edge for a face with two rings. So when the bore arrives and asks
the arrangement to cut that face, it is cutting a region described by an outline
no edge backs, and it declines.

Which ties the two threads together. The twice-named criterion measured earlier
would give these faces real edges for their outlines, and was reverted only
because the *reader* could not take it — a STEP-side limitation. The kernel-side
gain was never in doubt and is what this case needs. That is a better reason to
return to it than the export declines were.

STEP corpus unchanged by this thread's edge work: 20 round-trip, 1 declined,
0 wrong. Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## The two threads were not the same thread

The last entry argued that the twice-named criterion — reverted earlier for
breaking the STEP reader — was what the remaining chained cases needed, and that
this was a better reason to take it than the export declines had been. Tried it
on the kernel as it now stands, four fixes further on.

It works, and it is genuinely better structurally:

    ball − ball, before: 2 faces, 1 edge,  off-edge per ring [128, 0]
    ball − ball, after:  2 faces, 3 edges, seam-backed, off-edge [0, 0], still valid

Both complement faces' boundaries are now entirely made of edges, which is the
thing that has been missing since this line of work began.

**And it changes nothing that was failing.** Chaining stays 9 of 12: cutting that
result with a bore still declines `NeedsArrangement { face: 1 }`. So the
arrangement's refusal is *not* about a boundary with no edge behind it, which is
what the last entry assumed when it tied the two threads together. That
assumption is now measured and false.

Meanwhile the cost is unchanged — the same two STEP tests fail, because the
reader still cannot place the two passes of a pole-to-pole seam. Structure alone
does not pay for a regression, so it is reverted again, this time with the reason
recorded in the code: it is not that the criterion is wrong, it is that nothing
measured yet needs it.

Which leaves `NeedsArrangement` needing a diagnosis of its own rather than an
inherited one. The face is a sphere piece stating a 128-point outline and a
99-point circle; the bore meets it in two closed curves near the poles, which
should be holes and should leave the region connected. Why `planar::subdivide`
refuses that is the next thing to measure, and there is no longer a standing
theory to test first — which, on this thread's record, is the better position to
start from.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The arrangement was handed the wrong curve

Diagnosed `NeedsArrangement` on its own terms, with no inherited theory to test
first — which the last entry noted was the better starting position, and it was.

`planar::subdivide` returns `None`, and what it is given explains why:

    face 1 (sphere): outer 128 pts, 65 paths, already 0 -> SUBDIVIDE RETURNED NONE
      [3.544, 0.000]..[3.544, 0.024]   [3.545, 0.042]..[3.545, 0.090]
      [3.546, 0.108]..[3.547, 0.155]   [3.547, 0.174]..[3.549, 0.221]

Sixty-five paths, each two points before the midpoint pass, all at very nearly
**constant `u`**, marching in `v` with gaps between them. They are fragments of
one curve running down a meridian.

But the bore in this case shares the sphere's axis, and a coaxial cylinder meets
a sphere in **two circles at constant `v`** — the exactly-representable case this
kernel advertises. A meridian is what you get from a plane through the axis, not
from this pair at all. So the curve is wrong before the arrangement ever sees it,
and `None` is the honest answer to what it was handed.

That relocates the fault a long way. Four entries have treated the chained-cut
failures as topology — rims, provenance, seams, junctions — and this one is not:
it is the intersection itself, on a pair that has a closed form and should be
two circles. Whatever produced a fragmented meridian is upstream of every piece
of machinery this thread has been fixing.

Also worth noting for the next attempt: the fragmentation into two-point pieces
with gaps is its own signal. A curve sampled to tolerance does not come out in
sixty-five disconnected parts; something is clipping it per-segment, and the gaps
are as informative as the pieces.

Recorded at the `subdivide` call with the numbers. Suite green: 33 boolean,
21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Correction: it is a traced pair, not a coaxial one

The last entry said the fragmented curve came from a bore *sharing the sphere's
axis*, which would mean a closed-form pair producing a wrong answer — alarming,
and wrong. Checked the surfaces rather than assuming which sphere the face was
on:

    F0: sphere centre [0, 0, 0] radius 3
    F1: sphere centre [2, 0, 0] radius 2

The face that declines is **F1**, on the *second* sphere of the difference, whose
centre is two units off the bore's axis. That pair has no closed form and is
traced. Face 0 is the coaxial one, and it cuts fine.

The geometry also says why tracing struggles. Sphere B spans `x ∈ [0, 4]`; the
bore is radius 0.8 about the axis. They meet only where B's surface comes within
0.8 of the axis — right at B's leftmost point, where the surface is turning away
fastest. It is a grazing intersection, which is the hardest thing the marcher
does, and sixty-five two-point fragments with gaps is what failing at it looks
like.

So the fault is neither the topology of the last four entries nor a broken closed
form: it is `march` on a near-tangential pair. That is a place with a long note
already in it — its step is `span(bounds) × 0.01`, scaled to the model rather
than to the feature — and this is the first case in this thread that points
straight at it.

Worth saying plainly: the previous entry's claim was checked and failed, and it
would have sent the next attempt to rewrite a closed form that was never
involved. Two entries ago this thread noted that having no standing theory was
the better starting position; the lesson repeats.

Code comment corrected. Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs,
12 step, 639 lib.

## Sixty-four stubs, and why the marcher makes them

Counted the curves the two booleans produce, which settles what the fragments
are:

    first boolean:  1 shared curve   (points, count) [(100, 1)]
    second boolean: 67 shared curves [(2, 1), (3, 64), (65, 2)]

The two 65-point curves are sphere A against the bore — coaxial, closed form,
correct. The **64 curves of exactly three points** are the grazing pair, and
three is not a coincidence: `march` discards anything shorter, so every one of
these is a trace that died at the shortest length it is allowed to keep. Sixty-
four seeds, each managing two steps before stopping.

That is Newton settling failing where it is documented to fail. `settle` pulls a
point onto both surfaces along their normals and returns `None` when they are
parallel; sphere B reaches the bore's wall exactly where its own surface turns
away fastest, so the normals there are nearly parallel and the correction has
almost no component to work with. The marcher then cannot advance, and every
seed leaves a stub.

So the chain is complete, from the decline back to the geometry:

    NeedsArrangement            because `subdivide` was given 65 fragments
    65 fragments                because 64 traces died at three points
    64 three-point traces       because `settle` will not converge
    `settle` will not converge  because the surfaces graze

**Not attempted, deliberately.** The obvious move — drop stubs and let the pair
report no intersection — is worse than the decline it replaces: a boolean that
believes two solids do not meet where they do returns a confident wrong answer,
which is the one outcome this stack is built to avoid. The honest improvement is
to recognise the stubs for what they are and decline `TangentialContact`, which
already exists and would say *the surfaces graze* instead of *the subdivision
failed*. That is a change to what the kernel reports rather than what it
computes, and worth doing on its own rather than at the end of a long thread.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Correction: the tracing is fine

The last entry read the shared-curve histogram — `[(2, 1), (3, 64), (65, 2)]` —
as sixty-four traces dying at the shortest length `march` will keep, and built a
whole causal chain on it, down to `settle` refusing to converge at a grazing
pair. Went to implement the decline that story called for, and checked the
source first.

    closed-form pair (0,0): 2 reaching, sizes [0, 0]     <- analytic, not sampled
    traced pair (1,0):      1 reaching, sizes [65]       <- one clean curve

The second boolean finds **three** curves, and every one is healthy. The traced
pair — the grazing one, the whole subject of the last entry — traces to a single
65-point curve. There are no stubs at the intersection stage at all.

So the chain was wrong from its second link. The sixty-four three-point curves
come into being *after* the surfaces are intersected and *before* the face is
split, and nothing measured yet says what does it. The `TangentialContact` guard
written for the stub theory could never have fired, and is reverted; the comment
at the `subdivide` call now records the counts from the source rather than the
inference from the histogram.

Two corrections in three entries on the same case, both from reading a symptom
backwards into a cause without checking the step between. The measurement that
would have caught either is the same one: count the thing where it is *made*,
not where it is used. The histogram was taken from `shared`, which is a long way
downstream of `march`, and every conclusion drawn from it about `march` was
unfounded.

What is now known: the curve is right when produced, wrong when consumed, and the
fragmentation is at constant `u` with gaps — which is what a curve cut at a seam
repeatedly would look like, and is the next thing to measure rather than assume.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The clip flickers once per sample

Followed the curve from where it is made to where it is used, one step at a
time, and the break is at the clip:

    pair (1,0): mine 65 intervals, theirs 27 intervals
      mine   (0.000, 0.359) (0.640, 1.359) (1.640, 2.359) (2.640, 3.359) …
      theirs (0.000, 10.294) (10.705, 11.217) (11.782, 12.170) …

Sixty-five intervals on a curve of sixty-five points, in a pattern whose period
is exactly **1.0**: about seven tenths of every sample interval is "on the face"
and the remaining three tenths is not, over and over. A curve that genuinely left
a face and came back would not do it once per sample, in step with the sampling,
for its whole length. This is a containment test changing its mind between one
sample and the next.

The open-curve path then takes the *cross product* of the two interval sets —
correctly, since a curve can lie on a face more than once — so 65 against 27
becomes the 64 three-point fragments the arrangement chokes on. The multiplier
is not the bug; it is what makes a flickering clip catastrophic rather than
merely untidy.

That is three links of the chain measured rather than inferred: the trace is
clean, the clip flickers, the cross product multiplies. The remaining question is
narrow and local — why `clip_to_face` answers differently at the two ends of one
sample step, on a face whose region is a 128-point outline with a 99-point hole.

Recorded at the cross-product site with the interval counts, since that is where
the damage becomes visible and where the next person will be standing.

Suite green: 33 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A chord is not the curve

The last entry left one narrow question: why `clip_to_face` answers differently
at the two ends of a single sample step. The answer is one line of it —

    if surface.distance(p) > tolerance.max(1e-9) { return false; }

— and `p` is `curve.point(t)`, which for a sampled curve walks the *chords*
between samples. On a curved surface a chord sags away from it. At a sample the
distance is zero and the point is on the face; a third of the way along the step
the sag passes the tolerance and it is not. That is the period-1.0 pattern
exactly, "in" for ±0.36 around each integer:

    (0.000, 0.359) (0.640, 1.359) (1.640, 2.359) (2.640, 3.359)

The curve does not leave the surface between samples. Only the straight line
drawn through it does, and the test was asking about the line.

A traced curve is on both surfaces by construction, so for a sampled curve the
gate has nothing left to protect against and is dropped. The effect on what
reaches the arrangement:

    before: 67 shared curves [(2, 1), (3, 64), (65, 2)]
    after:   3 shared curves [(65, 2), (66, 1)]

Three — the two coaxial circles and the one traced curve, which is what the
intersection found in the first place. The fragments are gone, and with them the
`NeedsArrangement`: that operation now gets far enough to fail on its own terms,
`NotWatertight { open_edges: 195 }`.

Which is still a decline, and chaining is unchanged at 9 of 12 — so this is kept
for the same reason the rim cut was, and it is worth being explicit about the
rule. A change earns its place by fixing something *measured*, not by flipping a
test: sixty-seven curves where there are three is wrong however the operation
ends, and a containment test that answers by where a chord sags is wrong even on
the cases it happens not to break.

`a_curve_is_clipped_by_where_it_runs_not_by_where_its_chords_sag` pins it, by the
only property a declining operation can offer: that it no longer declines *this*
way.

Suite: 34 boolean (+1), 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32
builds; clippy unchanged.

## The same bug, one function over, and not worth taking

`curve_reaches_face` asks the same question `clip_to_face` was asking wrongly:
whether a point of the curve lies on the surface, evaluated at a parameter that
walks the chords between samples. It has the same gate on the same quantity.

It matters less. A sagging sample is *skipped* rather than the curve rejected,
and one surviving sample carries the answer. But the sampling is 64 evenly spaced
positions along a curve of however many points, so most of them do fall between
two of the curve's own, and a curve running right across a face can be judged not
to reach it.

Applied the same exemption and measured: **nothing changed.** Same suite, same
chaining at 9 of 12, the same three failures with the same open-edge counts —
195, 130, 3. Not a single curve rescued anywhere the corpus looks.

So it is reverted, and the reasoning is worth stating because it cuts the other
way from the last entry. That one kept a change with no test flip because it
fixed something measured — 67 curves where there were 3. This one has nothing
measured behind it at all: only an argument that the code is wrong. An argument
is not a measurement, and a correct-looking change with no demonstrated effect is
indistinguishable from an untested one.

The comment stays where the code is, saying what the hazard is, that exempting
sampled curves changes nothing observable today, and that this is where to come
back to when something does depend on it.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## What 195 open edges is, seen from inside

The closure gate withholds the body it rejects, so opened it behind an
environment variable, looked, and put it back. The result is far smaller than
195 open edges suggests:

    3 faces 4 edges defects 2
      EdgeFaceCount { edge: 2, uses: 1 }
      EdgeFaceCount { edge: 3, uses: 1 }
      F0 sphere   edges=2 free=0 loops=None
      F1 sphere   edges=1 free=0 loops=Some([128])
      F2 cylinder edges=3 free=0 loops=Some([130, 65])

Four edges is exactly right for this solid: the sphere-against-sphere circle, two
circles where the coaxial bore meets the outer sphere, and the traced curve where
it meets the inner one. Nothing is missing and nothing is fragmented — the last
two entries' work holds.

What fails is *ownership*. Two edges are named by one face each: the bore's wall
claims the rims where it cuts the outer sphere, and the sphere piece does not
claim them back. Every face reports zero free ends, so each face's own boundary
closes; it is the pairing between faces that has a hole in it, which is why the
tessellation opens along a whole rim and produces a number like 195 from a body
with four edges.

That is the same class as the six-face rim fixed several entries ago — a carried
rim reaching the wrong set of pieces — but the opposite error: there, too many
faces claimed one rim; here, too few. Both come from the same place, which is
what decides which pieces carry which rims.

Note also `F1`: one edge, and a single ring of 128 points. That is the complement
shape again — a face whose outline is the parameter rectangle with no edge behind
it. It has appeared in the STEP thread, in the refinement thread, and now here.

Gate restored, verified by grep. Suite green: 34 boolean, 21 brep, 13 brep_csg,
17 nurbs, 12 step, 639 lib.

## A band drops the rim that crosses it

Named the two single-use edges, which settles what the last entry could only
describe:

    F0 sphere@[0,0,0]  edges=[0, 1]
    F1 sphere@[2,0,0]  edges=[2]
    F2 cylinder        edges=[1, 0, 3]

    e0 sphere/cylinder    65 verts  users=[0, 2]
    e1 sphere/cylinder    65 verts  users=[0, 2]
    e2 sphere/sphere     100 verts  users=[1]
    e3 cylinder/cylinder  66 verts  users=[2]

**e2 is the carried sphere-against-sphere circle, and the outer sphere's own
piece has dropped it.** The bore cuts that sphere at two constant-`v` circles, so
the face goes down the band path — split into stacked bands in `v` — and that
path keeps a carried rim only when its mean `v` sits at a band's end. The
sphere-sphere circle is at constant `x`. It crosses the bands instead of ending
one, so every band decides it is not theirs and it is dropped by all of them.

That is why three faces with no free ends between them tessellate to 195 open
edges: the body is right except that one rim has a face on one side and nothing
on the other.

**e3 says the same thing from the other end.** Its surfaces are recorded
`cylinder/cylinder` — both the bore — because that pair is filled in when a
*second* face claims the edge, and no second face ever did. The traced curve
where the bore meets the inner sphere is claimed by the wall alone; the inner
sphere's piece states a single ring of 128 points, the parameter outline, and was
never cut by that curve at all.

So two faults, both about a face not claiming a boundary it has, and the band
path's `at_end` test is the first and the more clearly wrong: a rim is a rim
whether or not it happens to lie at constant `v`, and the test asks the wrong
question of a face that was cut by something else before.

Recorded at the filter with the numbers. Gate restored, verified by grep. Suite
green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A rim is a rim whether or not it ends a band

Fixed the filter the last entry named. A band now keeps a carried rim that
*crosses* it as well as one that ends it — any of the rim's points falling in the
band's range, rather than only its mean landing on the boundary. The at-end case
is unaffected, since a face's own rims sit at the outer edges of its range and no
other band reaches them.

    e2 sphere/sphere 100 verts  users=[1]      ->  users=[0, 1]
    defects 2 -> 1
    open edges: 195 -> 193, and 130 -> 126 on the union

The rim the outer sphere had dropped is claimed by both its faces again, which is
what the change was for and is measured rather than argued.

**One defect remains, and it is the other half already named.** `e3` — the traced
curve where the bore meets the inner sphere — is still claimed by the wall alone,
and still records `cylinder/cylinder` because no second face ever filled its
pair in. The inner sphere's piece states one ring of 128 points, the parameter
outline, and has no hole where the bore passes through it: it was never cut by
that curve at all. So the fault there is upstream of ownership — the face was not
subdivided, not merely mis-assigned.

Two open edges fewer out of 195 is not a result, and the case still declines. It
is kept for the same reason as the last two structural fixes: a rim that every
band decides is not theirs is wrong however the operation ends, and it now has
two faces where it had one.

No test pins it. The operation still declines, so there is no public property to
assert — `defects` and edge ownership are only reachable on a body the gate
returns, and it does not return this one. Worth saying plainly rather than
inventing a proxy assertion that would pass for the wrong reason.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The ring reaches the face and does not survive it

Followed the missing hole one step at a time.

    split_face 1 (sphere): 1 rings [(2, 65)], 0 chords, wraps false/false

So the curve is *there*. The face's split is handed exactly the ring it should
cut a hole with, and the face that comes out states its 128-point outline and
nothing else. The ring is lost between those two points.

It is not lost in the arrangement, because that face never reaches it — printing
at `subdivide_face`'s exit, only the cylinder's split appears. Face 1 returns
from an earlier path, and the gate on that path is:

    rings.retain(|(id, r)| {
        if r.uv.iter().all(|p| point_in_ring(&outer.uv, *p)) { return true }
        …clip it into chords instead…

`outer` for this face is the parameter outline — the rectangle that runs down the
seam and back and collapses a row of points at each pole. A ring is kept whole
only if *every* one of its points is inside that polygon, and a point-in-polygon
test on a boundary that doubles back on itself is exactly the kind that answers
wrongly.

That is where to look next, and it is a narrow question with a cheap answer:
count how many of the ring's 65 points `point_in_ring` accepts against that
outline. If it is all of them, the loss is further on; if it is some, this is it.

Which would also be the third distinct failure this thread has traced to the
complement face's outline — after the STEP round trip and the refinement pin. A
boundary that is a parameter artifact rather than a curve keeps being asked
questions it cannot answer.

Recorded at the retain. Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs,
12 step, 639 lib.

## Half a ring outside, and why fixing that alone goes backwards

Counted what the last entry asked for:

    face 1 ring 2: 33 of 65 points inside the outline (128 pts, area 19.4308)

Thirty-three of sixty-five, against an outline whose area is the whole parameter
rectangle. Nothing is outside a rectangle that covers everything, so the test is
not measuring what it reads as.

The reason is the seam, again, and specifically *which* seam. The second sphere
of the difference sits at [2, 0, 0]; the bore is around the origin; so the ring
where they meet encircles that sphere's leftmost point — which in that sphere's
own parameterisation is `u = π`, its seam. Half the ring's parameters therefore
land on the far side of the face's `u_range` of (3.1416, 9.4248) and are read as
outside. It is not a ring leaving the face. It is a ring measured against the
wrong turn.

Folding into the face's range before asking — the correction `curve_reaches_face`
and `clip_to_face` already carry, for exactly this reason — fixes the count. And
it makes the operation **worse**: back to `NeedsArrangement { face: 1 }` from the
`NotWatertight { open_edges: 193 }` it had reached, which is a step earlier in
the pipeline, not a step later.

So keeping the ring whole is not sufficient on its own. The path that then has to
cut a hole with a seam-straddling ring cannot do it, and handing it one is worse
than clipping the ring into chords as now. Both halves have to move together.

Reverted, and the test written two entries ago —
`a_curve_is_clipped_by_where_it_runs_not_by_where_its_chords_sag`, which asserts
only that this operation does not decline *that particular way* — is what caught
it, immediately and without a probe. A test that pins a declining case by the
shape of its decline earned its place today.

The measurement stays in the code at the retain, with the 33-of-65 and the note
that fixing it alone regresses.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Not two halves — at least three

Went back for "both halves together". With the fold on, so the seam-straddling
ring is kept whole, instrumented `subdivide_face` to see what the hole-cutting
path is handed — and it prints nothing. **That face never reaches the
arrangement.** It is turned away earlier, by one of the other nine places that
raise `NeedsArrangement`, and the `NotWatertight` it had reached without the fold
was further down the pipeline than where it lands with it.

So the last entry's framing was too tidy. Keeping the ring whole is not one of
two halves; it is the first of at least three steps, and the one after it is
somewhere in the guards between the split's start and the arrangement.

The exact guard is not isolated. Tagging every `NeedsArrangement` construction
with a marker broke the build — one of them is `.ok_or(…)` on a continuation
line, not a statement, so a line-oriented edit put an `if` where a `;` belonged.
Restored rather than patched around: a mechanical edit that does not understand
the syntax it is editing is the wrong tool, and the honest report is that this
tick found the shape of the answer and not the answer.

What is now known and was not: the fold's regression is *not* the arrangement
failing on a harder input, which is what the last entry assumed when it said the
hole-cutting path "cannot do it". The arrangement never sees it.

Code comment corrected to say that. Suite green: 34 boolean, 21 brep, 13
brep_csg, 17 nurbs, 12 step, 639 lib.

## The same question, asked twice, answered two ways

The guard the last entry could not isolate is nine lines after the one it did:

    rings.retain(|(id, r)| {
        if r.uv.iter().all(|p| point_in_ring(&outer.uv, *p)) { return true }
        …
    // Strictly inside, or the split crosses a boundary and needs a genuine
    // planar subdivision.
    for (_, r) in &rings {
        if !r.uv.iter().all(|p| point_in_ring(&outer.uv, *p)) {
            return Err(Declined::NeedsArrangement { face: face_index });

The same containment question, asked twice. Folding the first and not the second
kept the ring and then threw the face out for having it — which is why the fold
alone looked like the arrangement failing on a harder input, and why nothing
reached the arrangement at all.

Folded both. The topology of the case is now **complete**:

    3 faces 4 edges defects 0
      e0 sphere/cylinder  users=[0, 2]      e1 sphere/cylinder  users=[0, 2]
      e2 sphere/sphere    users=[0, 1]      e3 sphere/cylinder  users=[1, 2]
      F1 sphere loops=Some([128, 65])

Every edge has exactly two faces. `e3` — which was `cylinder/cylinder` because no
second face ever claimed it — is now `sphere/cylinder` with both. The inner
sphere has its hole. Four entries of faults on this one operation, and the body
that comes out of it is right.

    defects 2 -> 0, open edges 193 -> 177, chaining still 9 of 12

**And it still declines**, because the *tessellation* of that body leaves 177
edges open. Which is a different layer from everything this thread has been
fixing: not which faces own which boundary, but how a face whose outline is the
parameter rectangle and whose hole straddles the seam gets filled. The topology
was the obstacle up to here; it is not the obstacle any more.

No test pins it — the operation still declines, so the sound topology is not
reachable from outside. Said plainly rather than asserted through a proxy.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Putting the hole on the same turn as its face

With the topology sound and the tessellation still open, the next question was
how a hole straddling the seam gets *filled*. The fill splices holes into the
outer ring and clips ears; a hole with half its parameters a period away from the
other half is not something any fill can splice.

So once the containment questions are settled, the rings are folded onto the
face's own turn — the same fold, applied to the data rather than to a test — and
their areas recomputed from the folded points.

    open edges 177 -> 128 on the difference

The union is unchanged at 126, so this is not what ails that one.

Three folds now, in the same function, for the same reason: the retain, the guard
after it, and the rings themselves. Worth naming the pattern rather than the
three instances — **a face cut open at a seam has its own idea of where `u`
starts, and every question asked of a curve against that face has to be asked in
those terms.** This thread has now found that in `curve_reaches_face`,
`clip_to_face`, twice in `split_face`'s containment tests, and now in the ring
data itself. Each was found the same way, by measuring a count that should have
been all-or-nothing and was neither.

Still declining, still 9 of 12. Kept for the measured improvement, as with the
rest of this run: 177 open edges to 128 is not an outcome, but it is not nothing,
and a hole spliced from two turns at once is wrong wherever it ends up.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32
builds; clippy unchanged.

## A band that says it is a rectangle when it is not

Opened the gate and looked at where the 128 open edges are:

    tris 123649 open 128 carried 1
    by face: [("F0 sphere", 124), ("none", 4)]
      [-0.451, 0.661, -2.891] -> [-0.514, 0.613, -2.891]

`z = ±2.891` is `√(9 − 0.64)` — the bore's rim on the outer sphere. So 124 of the
128 tears run along that rim, on one face: **F0**, the band piece, whose loops are
`None`.

A band is filled from its parameter rectangle, because a band *is* the rectangle
between two cuts and its range says so exactly. That is true when nothing else
bounds it, and false here: this sphere was already cut by another sphere, so the
surviving band carries that rim across itself. The band still claims the whole
rectangle, and the fill covers ground the face does not have — which is what
tears along the rim.

So it is the band path again, and the third distinct thing wrong with it in this
one operation: it dropped a crossing rim (fixed two entries ago), and it states a
region that ignores the same rim (here). Both come from the same assumption —
that a face cut into bands is cut *only* into bands — and that assumption is what
a chained boolean breaks, because the face it is cutting was already cut by
something else.

The remaining `carried 1` is separate: one face's interior could not be filled at
all, and its input triangles were carried through. On its own that is enough to
decline, so both have to go.

Recorded at the band's `loops: Vec::new()`. Suite green: 34 boolean, 21 brep,
13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Sending the band to the arrangement gets there, and no further

Acted on the last entry: `split_swept` already refuses a *ring* that is not at
constant `v`, and a carried rim that crosses the bands deserves the same
question. Added it — declining there is not a decline, since the caller falls
back to the subdivision that can state such a region.

It reaches it. Both ball-then-bore cases stop failing at `NotWatertight` and
start failing at **`UnclassifiablePiece { face: 0 }`**: the arrangement runs, the
region comes out, and then `sample_between` cannot find a point that says which
side of the other solid the piece is on.

So the band's rectangle really was the wrong region — routing past it changes the
failure — and the path it routes to cannot finish either. That is an earlier
failure than 128 open edges, and nothing measurable improved, so it is not kept.
The same rule as two entries ago, applied against my own change this time: a
principled argument that the code is wrong does not by itself earn a place, and
trading a late failure for an early one is not progress unless something else
says it is.

What the tick establishes, which the last one only conjectured: the region a
chained cut needs on a previously-cut face is beyond *both* paths. The band path
cannot state it, and the arrangement can state it but cannot classify it. That is
a more useful thing to know than another open-edge count, and it says the next
work is in `sample_between` — finding an interior point of a region bounded by a
parameter outline, a crossing rim, and a hole — rather than in the dispatch.

Comment left at the band with both halves of the result. Suite green: 34 boolean,
21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A piece that encloses nothing

The last entry put the next work in `sample_between`. Measured what it is
actually handed, on the four pieces it is asked about when the band path is
routed past:

    outer 128 pts (area 19.4308), 1 hole  [99]     63 of its own vertices read inside itself
    outer  99 pts (area  7.6209), 0 holes          49 of its own vertices read inside itself
    outer 128 pts (area 19.4308), 2 holes [64, 64] 63 of its own vertices read inside itself
    outer  64 pts (area  0.0000), 0 holes           0 of its own vertices read inside itself

The last one is the failure: **a piece whose outer ring encloses nothing.** Sixty-
four points, zero area. No point is inside a ring of no area, so `sample_between`
returning `None` is right and `UnclassifiablePiece` is an honest report of it. The
fault is upstream — something built a region that is not a region.

Also worth noting from the same measurement: the three healthy pieces read only
63 of 128 and 49 of 99 of their *own* vertices as inside themselves. That is the
degenerate-outline problem again, and it is tolerated here because a boundary
vertex is not expected to be strictly inside — but it means the same
`point_in_ring` that has misled this thread twice is what every candidate is
tested against.

Tried dropping zero-area regions where `subdivide_face` builds its pieces. It
changed nothing, so that ring is not coming from there: some other piece-building
path produced it, and which one is the next thing to find rather than guess.

Reverted to the state before this tick's experiments — the band decline included,
since it was only ever applied to reach this measurement. Suite green: 34
boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The ring that encloses nothing is a ring that goes all the way round

Labelled the three `sample_between` call sites to find which builds the zero-area
piece. The last call before the failure is the *island* path, with a ring of 64
points:

    ZZ site 2: sizes [128, 64, 64]      <- the face, with two 64-point holes
    ZZ site 3: sizes [64]               <- one of them, as an island. Area 0.0000

And that is self-explaining once seen. The bore's circles on the outer sphere are
at constant `v` and **wrap the whole way round in `u`**. A wrapping ring is not a
closed loop on the parameter rectangle — it is a line from one seam edge to the
other — so its shoelace area is zero, and nothing can be inside it.

There is machinery for precisely this: a ring that goes all the way round is
converted into a chord by `seam_chord`, which splits the rectangle instead of
trying to enclose part of it. It runs `if face.u_wraps`. **This face reports
`u_wraps = false` while covering the whole turn.**

Which is the stale-wraps observation from much earlier in this thread — a boolean
output face keeps the input surface's ranges and flags rather than its own —
finally with a consequence attached to it. It is not a tidiness problem. It is
why a ball cut by a ball cannot then be bored.

And the fix is not "correct the flag", which would be one more thing carried and
liable to go stale. **Wrapping is a property of the ring**, and the code that
needs to know already measures it — the `hi - lo < period * 0.98` test inside the
wrapping branch, with its long note about being the wrong measure. Asking the
face is what wants removing; asking the ring is what is already written.

Recorded at the branch. Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs,
12 step, 639 lib.

## Correction: the empty island was not on the shipped path

Tried the fix the last entry proposed — gate the wrapping branch on the surface
being periodic rather than on the face's `u_wraps` flag, since a boolean's output
face does not set that flag even when it covers the whole turn. It builds, the
suite passes, and it changes **nothing**: 128 and 126 open edges, exactly as
before.

Instrumented the branch to see why, and the answer corrects the last entry:

    face 0 (sphere)   period 6.2832  rings [(99 pts, span 1.445)]
    face 0 (cylinder) period 6.2832  rings [(64, 6.183), (64, 6.183), (65, 2.738)]

The sphere face's ring spans 1.445 of a 6.28 period — it does not wrap, so no
conversion was ever being skipped for it. The wrapping rings are the bore's
circles on the **cylinder**, and there the branch already runs.

So the zero-area island measured two entries ago was produced by the *band-decline
experiment*, which was applied to reach that measurement and is not in the tree.
Saying it "is what stops a ball-minus-ball from being bored" attributed a fault
in a configuration I had built to the code as it stands. That is the third time
this thread has read a measurement taken under one configuration as a fact about
another, and the discipline that catches it is the same each time: say which
build the number came from.

One thing worth keeping from the instrumentation. Those wrapping rings span 6.183
against a threshold of 0.98 × 6.2832 = 6.158 — four parts in a thousand of
margin. The long note at that test says the measure is wrong and that better ones
exist; this is the first number showing how little room it has.

Reverted; the gate is back to `face.u_wraps`, with both results recorded there.
Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The turning measure, retried, and the note holds

The wrapping test decides whether a ring goes all the way round by how wide its
samples span, with a 0.98 threshold. Last entry measured its margin at four parts
in a thousand. The note beside it records two better measures, both tried long
ago, both costing two of fourteen re-cuts — and says the reason is not the
measure but `seam_chord` behind it.

Retried the better one: sum how far `u` turns before the ring closes, each step
taken the short way round. A ring that goes right round comes to a period; one
that merely straddles the seam comes to nothing, however wide its samples span.

    then: two of fourteen re-cuts lost
    now:  one test, `a_result_can_be_cut_again`, NotWatertight { open_edges: 7 }

Seven open edges, for the seven-point ring the note names. So the measure is
right, the cost has more than halved as the rest of this thread's fixes landed,
and what remains is precisely what the note predicted: `seam_chord` cannot cut a
ring that coarse, and the threshold has been protecting it rather than measuring
anything.

Reverted, and the retry recorded beside the note rather than replacing it. A note
that predicts its own retry's outcome three months on is worth more than the code
change it is guarding, and this is the second time this thread has been saved a
blind attempt by one.

The next work there is `seam_chord` on a coarse ring — not the threshold, which
is a symptom, and not the measure, which is already correct.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The coarse ring is cut exactly, and still opens seven edges

The note at the wrapping threshold says it is protecting `seam_chord` from a
coarse ring. Tested that directly: put the turning measure in, so the seven-point
ring is classified as wrapping, and printed what `seam_chord` does with it.

    seam_chord: 7 pts, origin 0.4677, period 6.2832,
                first folded u 0.467688 (gap from seam 0.000000)

**The cut is exact.** The seam vertex is present, the rotation starts on it, and
the chord runs from one seam edge to the other with nothing fabricated. Two finer
cases in the same run — 64-point and 43-point rings — come out the same way and
succeed.

So the threshold is not protecting `seam_chord`, and the note's explanation of
its own cost is wrong even though its prediction of that cost was exactly right.
Whatever leaves seven edges open is downstream of the cut, on a boundary that
both faces name with the same seven points.

Which is a better place to be than the note left it: the suspect has an alibi,
and the remaining question is narrower — how two faces sharing seven points along
one curve end up with seven segments unmatched. Not the measure, not the cut.

Also worth noting how long this took to establish: three probes, two of them on
cases I had reconstructed wrongly from the test — a drill of the wrong length,
then the wrong radius — each of which *passed* and looked like evidence the fault
was elsewhere. Reading the case out of the test rather than rebuilding it from
its description is the cheaper habit.

Comment at the threshold corrected. Suite green: 34 boolean, 21 brep, 13
brep_csg, 17 nurbs, 12 step, 639 lib.

## Seven open edges is one missing face

Opened the gate on the case the turning measure exposes, and the shape of it is
not what the last entry expected:

    2 faces 4 edges defects 1
      EdgeFaceCount { edge: 0, uses: 1 }
    tris 141551 open 7 carried 0
      open [1.607, 0.531, -2.477] -> [1.277, 0.351, -2.692] on ["F0:sphere"]
      …seven of them, one closed loop around the drill's entry…

Two faces. A bored ball cut by a drill should have the sphere, the bore's wall
and the drill's wall; the drill's wall is not there. Its ring in the sphere
therefore has a face on one side and nothing on the other, and the seven open
edges are that rim.

So "two faces sharing seven points end up with seven segments unmatched" was the
wrong question — there is no second face to disagree with. The turning measure
changes which rings are classified as wrapping, and on the drill's wall that
classification is what decides whether the wall survives as a piece at all.

That is the third framing of this failure in three entries — first the coarse
ring, then the cut, now the missing face — and each was wrong in the same way:
inferred from the number rather than read off the body. Seven open edges looked
like seven segments of a boundary. It is one hole with no wall behind it.

Reverted; the turning measure was only applied to reach this. The comment at the
threshold now carries all three findings, in the order they were established, so
the next attempt inherits the corrections rather than repeating them.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Both measures, on both rings, at last

Printed what each measure says about each ring in the failing case, which is what
three entries of inference should have started with:

    sphere   ring 7 pts: span 0.4572 (thr 6.1575)  turn  0.0000  span: hole,  turn: hole
    cylinder ring 7 pts: span 5.2788 (thr 6.1575)  turn -6.2832  span: hole,  turn: wraps

The drill wall's ring turns **exactly one period**. It goes all the way round,
and the span measure misses it — 5.28 against 6.16 — because seven samples of a
full turn span less than the turn they make. So today that ring is treated as a
hole, which is *wrong*, and the operation succeeds regardless. Classified
correctly, it goes down the wrapping path, and the wrapping path loses the face
entirely.

That resolves the three entries of shifting explanations. The note at the
threshold was right that it protects the wrapping path; it was wrong about how,
and so was I twice over. `seam_chord`'s cut is exact. What cannot handle a
correctly-classified wrapping ring is whatever comes after it, on a face whose
*only* ring wraps — and that face is currently saved by a misclassification.

Which is worth saying plainly: this measure is not merely imprecise, it is wrong
in a way the code depends on. Fixing it in isolation removes a face from a solid.
The two have to be taken together, and the order is forced — the wrapping path
first, the measure second.

Reverted; the turning measure was applied only to reach this. All four findings
now sit at the threshold in the order established.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The wrapping path produces the face; something else throws it away

Last entry said the wrapping path loses the face and put it first in the order of
work. Measured it before starting:

    subdivide 0 (cylinder): outer 36 pts, chords [8]
      -> 2 pieces from face 0

One wrapping ring becomes one chord; one chord cuts the parameter rectangle into
two pieces. That is exactly right — a curve that goes right round a cylinder's
wall separates the wall into what is above it and what is below — and it is what
the wrapping path is for.

So the wrapping path works. Both pieces are then **discarded**, and the wall is
absent from a result that keeps every other face. What cannot cope with a
correctly-classified wrapping ring is the classification of the pieces, not the
producing of them.

That is the fifth reading of this one failure, and each has moved one step closer
to the thing itself: coarse ring, then the cut, then a missing face, then which
ring wraps, and now which piece survives. Every one was measured and every one
narrowed. What has cost the entries is that each measurement answered the
question asked and I asked the next one from the answer rather than from the
body.

The order the last entry set is therefore wrong as stated: it is not "the
wrapping path first, the measure second" but "classification of a wrapping
piece first". The measure still goes last.

Comment corrected at the threshold. Suite green: 34 boolean, 21 brep, 13
brep_csg, 17 nurbs, 12 step, 639 lib.

## The missing piece is the one inside the ball

Instrumented the keep/discard decision, which is where the last entry left it:

    piece from B on cylinder: sample [1.6495, 0.6804, -20.8609]  inside_other false  keep false
    piece from B on cylinder: sample [1.1501, 1.0163,  19.1391]  inside_other false  keep false

The drill is sixty units long and the ball has radius three. Those samples are
twenty units clear of it, so both pieces are the wall *beyond* the ball at each
end, and discarding them is correct. The piece the result needs — the wall inside
the ball — is not among them. It was never formed.

Because the drill passes right through: its wall meets the sphere in **two**
rings and wants cutting into three parts by two chords. It got one chord, and two
pieces. So only one of the two rings arrives at the split as a wrapping ring.

Which corrects the last entry again. Classification is not what cannot cope: it
is correct on everything it is given. What is wrong is what it is given, and the
question is now the narrowest it has been — where does the drill's second ring go?

Six readings of this failure across six entries, each measured, each nearer.
Worth recording the pattern rather than only the findings: every one of the six
was an inference from the *previous* measurement instead of a fresh look at the
body, and every one was wrong in the direction of blaming the machinery I had
just been reading. The one that finally moved was printing the samples — a
quantity I had not looked at once in five entries, because I had been reasoning
about which code ran rather than what it decided.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The drill's second ring was never traced

Counted the curves where they are made, per pair — the lesson from three entries
ago, finally applied to the right quantity:

    first  boolean, pair (0,0): 2 curves [Circle, Circle]
    second boolean, pair (0,0): 1 curve  [sampled 6 points, closed]

The drill runs parallel to the sphere's axis but two units off it, so that pair
has no closed form and is traced. The trace returns **one** ring. A drill passing
right through a ball meets its surface twice — entry and exit — and the second
ring does not exist at any point downstream.

Which explains the whole chain, backwards: one ring gives one chord, one chord
gives two pieces, both pieces are the wall beyond the ball, both are correctly
discarded, and the wall is missing. Every stage was right on what it was handed.

Six points is also the "seven-point ring" that has run through the last five
entries — six samples plus the closing repeat. It looked like a coarse ring
because it *is* one, and coarseness was the first suspect; but the fault was
never that it is coarse, it is that it is alone.

And the span measure survives this while the correct measure does not, which is
worth stating: misclassifying that ring as a hole leaves the drill's wall as an
island, and an island is a plausible enough face that the result closes. Getting
the classification right exposes the missing ring. That is why fixing the measure
in isolation removed a face — it was not the measure's fault at all.

So the work is in `march`/`seeds` on this pair: a sphere and an offset cylinder,
two rings, one found. That is where the next entry starts, and for the first time
in this run it is a question about geometry rather than bookkeeping.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## `march` finds both rings

The last entry put the work in `march`/`seeds` — a sphere and an offset cylinder,
two rings, one found. Called `march` directly on that pair before starting, which
needs no instrumentation since it is public:

    overlap box: 2 curves ["41 pts closed true z -2.780..-2.381",
                           "41 pts closed true z  2.380.. 2.780"]
    sphere box:  same

Both rings, forty-one points each, either bounds. The tracing is right and the
seeding is right.

So two rings of forty-one points become **one ring of six** by the time the
boolean holds them, and both the loss and the coarsening happen between `march`
returning and `shared` being built: the closed-curve re-sampling, the
`curve_reaches_face` filter, the clip. That is a short stretch of code and every
step of it is one this thread has already read for other reasons.

Seven readings now. The useful thing about this one is that it is the first to
*exonerate* rather than accuse: the last six each moved the blame one step
earlier, and this one stops the regress by testing the accused directly instead
of inferring from what reached it. Calling the function under suspicion with the
inputs it gets is cheaper than any of the six instrumentations that preceded it,
and it was available the whole time.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The bounds are the bug

Instrumented the stretch between `march` and `shared`, and the first line settles
seven entries of searching:

    ZZ traced pair (0,0): 1 curves

Inside the boolean, `march` returns **one** curve for the pair it returns two for
when called directly. Same surfaces. The difference is the third argument:

    box_both ([-4.001, -4.000, -33.001], [4.651, 4.051, 33.001])

That is the union of *both solids'* vertices, padded — `z` from −33 to 33,
because the drill is sixty long. Span 66.

And `seeds` keeps a candidate only if it is more than `span / 12` from every
candidate already kept. Sixty-six over twelve is **5.5**. The drill's two rings on
the sphere, entry and exit, are **5.2** apart. The second is discarded as a
duplicate of the first, and everything downstream — the coarse ring, the single
chord, the two outside pieces, the missing wall, the seven open edges — follows
from that one comparison.

So the seven-entry chain ends at a per-model scale used for a per-feature
question, which is precisely the fault the note on `march`'s step has recorded
for months: `span(bounds) * 0.01`, scaled to the model rather than to the curve.
Two symptoms, one cause, and the cause is a box computed once for the whole
operation and handed to every pair.

The fix is a box per face pair. Not attempted here: `box_both` feeds the seeding,
the step and the wander limit, and narrowing it changes what every traced pair
sees — that wants its own entry with the corpus in front of it, not the tail of
this one.

Recorded at `box_both` with both measurements. Suite green: 34 boolean, 21 brep,
13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The right box finds both rings, and the arrangement declines them

Implemented what the last entry named: bound a traced curve by where *both*
solids are — the intersection of their boxes — rather than by where either is.
A curve lying on both surfaces cannot leave either box, so the union was as much
too large as the longer solid is long.

    pair (0,0): 1 curve  ->  2 curves

Exactly the fix, at exactly the level it targets. The drill's entry and exit
rings are both found, the seed dedup no longer swallows one, and seven entries of
downstream symptoms have nothing left to feed on.

**And one test fails**: the same `a_result_can_be_cut_again`, now with
`NeedsArrangement { face: 0 }`. The sphere face arrives with the drill's two
holes and the arrangement cannot cut it.

So this is the second time in this run that a correct fix has been stopped by the
same wall. The turning measure classifies a wrapping ring rightly and the
wrapping path loses the face. The right box finds both rings and the arrangement
declines them. Both were reverted for a failing test; both are right.

Which is worth stating as a fact about the kernel rather than about either
change: **it currently passes some cases because information is lost upstream of
the part that cannot handle it.** A coarse ring that should be two rings, a
wrapping ring misread as a hole — each loss lands somewhere the code copes, and
correcting either delivers the truth to something that does not. That is not a
reason to keep the losses. It is a reason to expect the next real fix to fail a
test too, and to judge these changes by what they make true rather than by the
count of green tests on the way.

Reverted, with both measurements recorded at `box_both`. Suite green: 34 boolean,
21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## What blocks the box fix is the outline, not the arrangement

Last entry left the box fix blocked by `NeedsArrangement { face: 0 }` and read
that as the arrangement being unable to cut a face with two holes. Measured what
the arrangement is actually handed:

    subdivide face 0 (sphere): outer 272 pts area 15.4687, paths [22, 22], already 0

The drill's two rings do not arrive as rings. They arrive as **chords of 22
points**, having been clipped — which happens when a ring is judged not to lie
inside the face's outer boundary. That boundary is the 272-point parameter
outline of a band, and judging containment against it is the same test that read
33 of 65 points outside a rectangle covering everything, and that returned 63 of
128 of its own vertices as inside itself.

So the arrangement is not failing to cut two holes; it is being given two chords
that should have been two holes, and failing on those. The blocker for the box
fix is the outline, which is the same blocker as for three other things in this
file — the STEP complement, the refinement pin, and the ring fold two entries
back.

That is a more useful result than another blocked fix. Four separate pieces of
work now wait on one thing: a face whose boundary is a parameter artifact rather
than a curve cannot answer *is this point inside me*, and everything that asks
gets a wrong answer somewhere.

Reverted. Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## Correction: that outline is a clean rectangle

The last entry concluded the box fix is blocked by the parameter outline's
containment test, and that four pieces of work now wait on one thing. Checked the
number before building on it.

The outline in question has area **15.4687**. The face's ranges are `u ∈ (−π, π)`
and `v ∈ (−1.231, 1.231)`, and 2π × 2.462 = 15.4687 exactly. It is a clean
rectangle — not a polygon that doubles back on a seam — and a crossing test on it
is sound.

The rings are not near its edges either. The drill meets the sphere at
`z = ±2.38..2.78`, which is `v = ±0.92..1.18` against a range of ±1.231, at
`u ≈ 0.494` with the seam at ±π. Nothing in that configuration should fail a
containment test.

So "the blocker is the outline" does not follow from the measurement it was drawn
from, and the four-things-wait-on-one-thing claim with it. Both were pattern-
matching against three earlier findings that *were* about degenerate outlines —
the strongest kind of wrong guess, because it explains the evidence and has
precedent.

What remains true and measured: the box fix yields both rings, and they reach the
arrangement as two 22-point chords rather than two rings. Why is now genuinely
open, and the next tick starts by finding where a ring becomes a chord rather
than by assuming which test rejected it.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A body's vertices do not span the body

Followed the rings to where they stop being rings, and it is the sampler's first
guard:

    sample_closed on sphere: 21 pts, closed false, invert fails on 0

They arrive **open**, with 21 points. Not 41, and not closed. So nothing turned a
ring into a chord — they were never rings by the time anything looked, and the
open-curve path did exactly the right thing with an open curve.

Why open is the box fix's own fault. `bounds` is taken over a body's
**vertices**, and a bored ball's vertices are only the two bore rims — radius 1,
z ±2.83. Intersecting that with the drill's x ∈ [0.95, 1.65] leaves a sliver
about 0.05 wide, and the trace runs into its wall and stops.

Called by hand with a box around the actual overlap, the same pair gives two
closed rings of 41 points. That is what made the fix look right two entries ago:
the hand-built box and the computed one are different boxes, and only the
hand-built one is correct.

So the intersection is the right idea and a vertex hull is the wrong way to get
it. A `Body`'s vertices are where its *edges* are, which on a bored ball is two
small circles and nothing else — the surface between them has no vertices at all.
The box wants the surfaces' own extents over the faces' parameter ranges.

Worth noting what this says about the last three entries. The box fix "produced
two curves" and I took that as the fix working; it produced two *open* curves,
which is a different thing, and the count alone could not tell them apart. Each
subsequent conclusion — the arrangement cannot cut two holes, then the outline
blocks it — was drawn from that. Printing what a thing *is* rather than how many
there are would have caught it at the first step.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The box, done properly, gets a closed solid

Replaced the vertex hull with what the last entry said it wanted: each face's
surface sampled across its own parameter range, unioned with the vertices, and
the two solids' boxes intersected.

    3 faces 5 edges, shells closed [true], free ends [0, 0, 0]
    defects [EdgeOffSurface { edge: 1, deviation_scaled: 1 }]

A **closed solid**. One shell, no free ends, every edge between exactly two
faces. The drill's rings are found, the wall is present, and the topology is
sound throughout. The only complaint is that one edge's vertices sit one
tolerance off their surfaces — `deviation_scaled: 1` is the smallest value that
trips the check.

So the case that has run through the last nine entries now comes down to a traced
curve landing at the edge of what `defects` allows. That is a different kind of
problem from everything before it: not a boundary lost, not a piece dropped, not
a region mis-stated, but a curve accurate to tolerance being measured against a
threshold of exactly tolerance.

Reverted, because `a_result_can_be_cut_again` asserts validity and this fails it.
But the reason it fails has moved from "the second cut declines" through "the
second cut is not watertight" to "one edge is a tolerance out", and that last one
is worth saying out loud: the boolean is now producing the right solid and
failing its own inspection by a hair.

Next: whether a traced edge should be held to `> tolerance` against surfaces it
was traced onto *to* tolerance, or whether the trace should settle harder before
the curve is kept.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## One ring exact, the other a hair out

The last entry ended with a closed solid failing on `EdgeOffSurface {
deviation_scaled: 1 }` and asked whether the trace should settle harder. It
should not: `settle` already converges to `tolerance * 1e-3`, a thousandth of
what the check allows.

Measured the edges instead:

    e0 sphere/cylinder 41 verts closed true  worst off-surface 0.000000
    e1 sphere/cylinder 41 verts closed true  worst off-surface 0.001162
    e2 sphere/cylinder 73 verts closed true  worst off-surface 0.000000

Two rings from the same trace, same length, and one is exact while the other is
0.001162 against a tolerance of 0.001. Systematic error would move both. So
something displaces one ring's points *after* tracing, by about a tolerance.

The weld in `from_pieces` fits: its quantum is `tolerance * 0.5`, the first point
into a cell wins, and a later point in the same or a neighbouring cell is
replaced by it — a move of up to roughly that distance, and the check is a
whole tolerance with no allowance for the weld that precedes it.

Which would make this the last piece of the case: a curve traced to a thousandth
of tolerance, welded by half a tolerance, and inspected against one tolerance.
Two of those three are chosen here and can be made consistent.

Not attempted this tick — the weld is load-bearing for every shared boundary in
the result, and narrowing it or exempting traced vertices from the check are
different fixes with different blast radii. Recorded with the numbers so the next
attempt starts from them.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Not the weld, and not the marcher

Two hypotheses tested, both refuted, which is the useful shape of a tick.

**The weld.** Its quantum is `tolerance * 0.5`, and the last entry reasoned that
a point pulled into an existing cell could move about that far. Ran the case with
the quantum at `0.5` and at `0.1`:

    quantum tolerance*0.5: worst edge e1 off by 0.001162
    quantum tolerance*0.1: worst edge e1 off by 0.001162

Identical to the digit. The weld does not touch it.

**The marcher.** Called directly with a box around the overlap:

    curve 0: 37 pts closed true, worst off-surface 0.00000000, longest step 0.0655
    curve 1: 37 pts closed true, worst off-surface 0.00000005, longest step 0.0655

Exact to five parts in a hundred million. The tracing is not the source either.

What the second measurement also shows, and what the last three entries missed:
**37 points, where the result's edge has 41.** The boolean's computed box and my
hand-built one produce *different curves* for the same pair — not merely more or
fewer samples of one curve, since the accuracy differs by four orders of
magnitude. Every comparison I have drawn between "march called directly" and
"march inside the boolean" has been between two different traces.

So the remaining question is what the computed box makes the marcher do
differently, and the honest answer is that I have been treating the two as
interchangeable for four entries. The next measurement is the one I should have
taken first: print the box the boolean computes, march with exactly that, and
compare against the edge in the result.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The box sets the step, not the accuracy

Took the measurement the last entry said should have come first: march the pair
with each candidate box and compare.

    union box (shipped)   span 66.002  step 0.6600  1 curve,   6 pts, worst 7.2e-7
    surface-extent isect  span  6.600  step 0.0660  2 curves, 37 pts, worst 5.0e-8
    tight overlap         span  6.000  step 0.0600  2 curves, 41 pts, worst 3.7e-7

`march` is exact in **every** configuration. The box does not affect accuracy at
all; it sets the step — `span * 0.01` — and through it the seed spacing, which is
what loses the second ring. Two separate consequences of one number, and only the
first was ever in doubt.

So the 0.001162 on that edge is not a traced point. It is a point the arrangement
*added*, and the size names it: a step of 0.066 across a ring of radius ~0.35 has
a sagitta of about 0.0016, and a crossing computed by intersecting the chord's
polyline with a boundary's lands exactly a sagitta off the true curve.

Which is written down already. The note at `near_curve` says: *"the subdivision
intersects the chord's polyline with the boundary's, and that crossing lands a
sagitta from the curve — 1.3e-3 at a tolerance of 1e-3, which the weld will not
close."* Measured here: 1.162e-3. The same effect, the same magnitude, in a place
the existing snap does not reach — `near_curve` is applied to the outer ring's
crossings and this one is on a hole.

That closes the question the last four entries were circling, and it closes it on
something already understood rather than something new. The remaining work on the
box is real and separate: it costs the second ring through the seed spacing, and
that is worth fixing whether or not the crossing snap is extended.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The snap has nothing to snap to

Extended `near_curve`'s crossing snap from the outer ring to holes, which is what
the last entry identified: a crossing is found by intersecting two polylines and
lands a sagitta off the curve, the outer ring already pulls those back, and a
hole's crossings are no different.

Green on its own — whole suite passes. And it does not fix the case: with the box
fix alongside it, the same `EdgeOffSurface { edge: 1, deviation_scaled: 1 }`
survives on the same otherwise-closed solid.

Because there is nothing to snap to. `near_curve` searches *the chords'* shared
vertices for a point within four tolerances. A ring kept whole does not arrive as
a chord: it reaches `subdivide_face` as an `already` path — bare parameters, no
vertex identity, no `shared` id. So a crossing on a hole that came from a ring
has no curve to be pulled onto, and the sagitta stands.

That is the same shape as several earlier findings in this file, and worth naming
as a class now that it has recurred: **a boundary that arrives without identity
cannot be repaired later.** The parameter outline had no edges behind it; the
`already` paths have no vertices behind them; both defeat machinery that is
otherwise correct, and both are cheap to fix at the point where the identity is
dropped rather than where it is missed.

Reverted the hole snap too, by the rule this thread has been applying to itself:
green and principled is not enough without something measured moving. The finding
is what the tick produced.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## One vertex, on one surface

Gave `near_curve` the identity the last entry said it lacked — search every
shared curve, not only the chords', so a crossing on a ring kept whole has
something to snap to. Green. With the box fix and the hole snap alongside it,
still `EdgeOffSurface { edge: 1, deviation_scaled: 1 }`.

So printed the vertices instead of reasoning about which mechanism made them:

    e1 sphere/cylinder, 41 verts, 3 off
      idx 0 and 40, vertex 39: 2e-6            (the closing repeat, negligible)
      idx 37,       vertex 79: sphere 1.35e-4, cylinder 1.162e-3

**One vertex.** It lies on the sphere and is a thousandth off the cylinder. That
is not a crossing — a crossing sits off *both* surfaces by the sagitta of the
polylines that produced it, and would have snapped, since 1.16e-3 is well inside
the four-tolerance reach. It is a point minted from a parameter on one surface
without being settled onto the other.

Which means the last two entries' explanation was wrong in a way the numbers
happened to support: a sagitta of ~0.0016 was the right order for a 0.066 step,
and I took the match as identification. The asymmetry — exact on one surface,
out on the other — is the thing that identifies it, and I had not printed it.

Both changes reverted, neither having moved anything measured. What the tick
establishes is narrow and checkable: find where a vertex on a two-surface edge is
created from one surface's parameters, and settle it onto both.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The point is new, not a duplicate

Found where the offending vertex is made — `from_pieces`, the fallback for a loop
point that reached the body with no identity:

    let p = piece.surface.point(uv[0], uv[1]);
    *slot = *welded.entry(key(p)).or_insert_with(...)

It is minted from *that piece's* parameters, so it lands on that piece's surface
and nowhere else in particular. Exactly the asymmetry the last entry measured:
1.35e-4 off the sphere, 1.162e-3 off the cylinder.

Tried the obvious repair — before minting, look through the neighbouring weld
cells for an existing vertex within a tolerance, since the face on the other side
of a shared boundary should already have made the right one. Green, and it
changes nothing: the same vertex, the same 1162e-6.

**Because there is no vertex there to find.** The other side never made one. So
this is not a duplicate that the weld's cell size was too small to catch — the
hypothesis the last entry's arithmetic suggested — but a genuinely new point that
needs *settling* onto both surfaces rather than welding to something.

And that cannot be done where it is minted: only the piece's own surface is known
there. The edge's pair is known later, when edges are built, which is where a
settle belongs.

Reverted. Two negatives and one identification: the weld is not too coarse, there
is nothing to weld to, and the fix is a settle at edge construction.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Valid — and one regression away from landing

The last entry said the fix is a settle at edge construction, where an edge names
both its surfaces. Built it: Newton on the pair, applied to every shared vertex
more than a thousandth of a tolerance off either surface.

It did nothing, and the reason was my own guard — I only accepted moves up to one
tolerance, and the point is 1.162e-3 out, so the one move that mattered was the
one rejected. Raised to four tolerances:

    => valid true defects []
    34 boolean tests pass

The bored ball cut a second time is a **valid solid**. That is the case this file
has been following for a dozen entries, from `NeedsArrangement` through six
readings of a missing wall to a single vertex a thousandth off a cylinder.

**And it does not land**, because the box change it needs breaks something else:
with a box drawn where both solids are, a rod crossed by a rod comes back from a
STEP round trip open — 418, 492 and 292 edges for the three ops — while the
boolean's own result for those is closed. So the export or the import cannot take
what the narrower box produces for that pair.

Measured which half: the settle alone is green everywhere and moves nothing
measurable — chaining stays 9 of 12, the corpus stays clean. The box alone is
what breaks the round trip. They are independent and only the box is a problem.

Both reverted, with the working combination and its single blocker recorded at
the mint site. The next tick has a narrow question with a known-good state either
side of it: why a rod-across-rod result traced in a tighter box does not survive
STEP when the same result traced in a looser one does.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Correction: the box does not break the rod, it fixes it

The last entry recorded the box change as breaking a rod crossed by a rod through
STEP. Checked what the current build does with that pair before diagnosing the
difference, and it does not do it at all:

    rod across rod, Union, today: Err(NeedsArrangement { face: 0 })

It declines. The corpus guard therefore skips it, which is why it has never
appeared in these numbers. With the box change:

    out:  7 faces 6 edges valid true
          e0 cylinder/cylinder 147 verts closed
          e1 cylinder/cylinder 148 verts closed
    STEP: 7 ADVANCED_FACE, 4 CIRCLE, 2 POLYLINE
    back: 7 faces 6 edges, tris 694 open 418

The case **resolves**, and to a valid solid. What fails is the round trip of a
result that did not exist before: two traced curves go out as `POLYLINE`s and the
reader does not rebuild them watertight.

So the box fix has no regression against it. It makes a bored ball cut twice
valid, it makes a rod crossed by a rod resolve, and the only thing standing
against it is that STEP cannot carry one of the new results — which is a
limitation of the exporter and importer for traced polyline boundaries, not a
fault in the boolean.

That is a materially better position than the last entry described, and I would
not have found it by diagnosing the difference between two builds: the question
"what does the current build do with this pair" had to be asked first. The same
habit that caught the march comparison two entries ago — test the accused
directly — applies to the baseline as much as to the change.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The box fix is blocked by a gap already on the list

Looked at what STEP does to the newly-resolved rod-across-rod result, face by
face:

    OUT  F0 cylinder edges=4 loops=Some([34, 146, 147]) u=(3.14, 9.42) wraps=(false, false)
    BACK F0 cylinder edges=4 loops=None                u=(-π, π)      wraps=(true, false)

The wall leaves as an outline with **two holes** — where the crossing rod passes
through it — and comes back as a whole wrapping cylinder with no holes at all.
The fill then covers the holes, and the body is open by 418 edges. F3 and F4 go
the same way: a traced ring out, a plain `v` band back.

That is not new. It is the importer's standing limitation, recorded in this file
when the STEP complement was being chased: **`loops: None` always, extent
recovered from the edges' vertices**, which cannot state a region that is not a
parameter rectangle. A wall with two holes through it is exactly such a region.

So the box fix — which makes a bored ball cut twice valid *and* makes a rod
crossed by a rod resolve at all — is held up by a gap that was already on the
list for its own reasons, and that has now blocked three separate things: the
STEP complement export, the `NeedsArrangement` on a chained cut, and this.

Which changes what is worth doing next. The importer's inability to state a
trimmed region has stopped being a STEP-only concern; it is now the single thing
between this kernel and two measured improvements. Giving an imported face its
holes — the edges are all there, in the right places, and only the loops are
missing — is the work with the most behind it.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Deriving holes on import, and why the obvious rule collapses a solid

The last entry put the work on the importer: give a face its holes, since every
edge is already there and only the loops are missing. `Body::face_loops` chains a
face's edges into rings exactly as the kernel does, so the change is to ask it.

Asked it wherever a face's edges make more than one ring. The corpus guard caught
the result immediately:

    ball and bore Union: left as 132.26663, came back as 19.89509,
                         and nothing was reported

A cylinder wall between two rims has two rings and **neither encloses the
other** — it is an annulus, not a face with a hole. `face_loops` orders by signed
area and marks the smaller a hole, so the wall came back as a disc with a bite
out of it and five sixths of the solid disappeared. Silently, which is what the
guard exists for.

The exporter already knows this: *"On a face that is an annulus — a cylinder
between two rims — neither ring encloses the other, and the choice is arbitrary
but must still be made; longest wins."* The same fact, needed on the other side
and not asked there.

So the rule is not "more than one ring means holes". A ring is a hole when it is
*inside* another one, which is a containment test in parameter space and not an
area comparison. That is a small change to what I wrote, and the reason to
record rather than rush it is that the failure it caused was invisible — right
face count, right edge count, plausible geometry, and a fifth of the volume.

Reverted. Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## Containment is the right rule and not enough

Implemented the rule the last entry derived: the outer ring is the one with the
largest area, a ring is a hole only when *every* point of it lies inside that
one, and a face whose rings do not nest keeps its extent. Green everywhere —
the annulus that collapsed a ball with a bore is now left alone.

It does not fix the rod. Still 418 open edges, because that wall is **both**:

    OUT F0 cylinder edges=4 loops=Some([34, 146, 147])

Four edges — two rims, and the two rings the crossing rod cut through the wall.
The rims are an annulus, neither inside the other, so the nesting test declines
the face, correctly, and it comes back with no holes at all.

Its region is the rectangle *between* the rims with two holes in it. No chaining
of that face's edges will produce that outer boundary, because the boundary is
not made of edges: the kernel does not derive it either, it **synthesises** it
with `parameter_outline`. The importer would have to do the same, and then it
would have the third case — outline plus holes — rather than the two it can
almost express.

Reverted, since it moves nothing measured on its own. What the tick establishes:
the containment rule is right and is not the missing piece; the missing piece is
that an imported face has no way to say "the rectangle between my rims", which is
the same synthesised-outline gap the kernel solved years of entries ago and the
importer never got.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Synthesising the outline everywhere breaks the plate

Built what the last entry called for — where a face has a ring strictly inside
its parameter extent, take the extent's rectangle as the outer boundary and that
ring as a hole. It costs **three** STEP tests:

    a_bore_arrives_as_a_cylinder_not_as_triangles
    a_part_built_by_several_cuts_round_trips
    no_solid_leaves_through_a_file_and_comes_back_a_different_size

Because a plate with a bore matches the condition and must not. Its outer ring
touches its extent, its bore ring is interior — and its extent is only the
*bounding box of its edges*, so replacing a correct edge-derived boundary with
that rectangle grows the plate to its own bounding box.

The discriminator is the one the last entry already named for the other half of
this problem, applied the other way round: **does the non-interior ring enclose
the interior ones?** A plate's outer ring does, and its edges say the boundary
exactly. A band's rims do not — they are lines across the rectangle, not loops
around it — and only then does the outline have to be synthesised.

So the importer needs both rules, chosen by that test: chain the edges where they
enclose, synthesise where they do not. Neither alone is right, and this tick
proved the second alone as thoroughly as the previous one proved the first.

Reverted. Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step,
639 lib.

## Both rules, and the rod gets worse

Implemented the pair the last entry specified — chain the edges where a rim
encloses the interior rings, synthesise the extent's rectangle where none does.
Green on its own: all twelve STEP tests, all 34 boolean.

With the kernel's narrower trace box alongside it, the rod is **worse**: 839 open
edges against 418 with no import rules at all.

So the synthesised rectangle is not that wall's region either. Its rims run right
round a cylinder, so in the parameters they are two lines across a rectangle
whose `u` sides are the *same seam*; a plain four-sided outline says nothing
about that, and gives the fill a boundary that crosses itself at the seam. The
kernel's `parameter_outline` reuses the carried rims for two of its four sides
for precisely this reason, and a rectangle sampled from the ranges does not.

That is the fourth distinct thing tried against this one gap — derive by count,
derive by containment, synthesise always, synthesise where nothing encloses —
and each has been wrong in a way the previous one did not predict. What they have
in common is that the importer has no notion of a *seam*: every rule so far has
assumed the outer boundary is a loop in the plane, and on a closed surface it is
not.

Which is the same thing the kernel needed and got, in `parameter_outline`,
`seam_chord` and the wrapping branch. The importer needs the same vocabulary
before any of these rules can be right, and that is a larger piece of work than
another rule.

All four changes reverted. Suite green: 34 boolean, 21 brep, 13 brep_csg,
17 nurbs, 12 step, 639 lib.

## Nothing needs synthesising

Before trying a fifth import rule, checked the premise the last two entries were
built on — that the rod's wall has an outer boundary no edges describe:

    F0 cylinder edges=4  rings (pts, off-edge) [(34, 0), (146, 0), (147, 0)]

**All three rings are made entirely of edges.** The 34-point outline included:
`materialise_seams` gave the seam an edge several entries ago, so the outline has
one too. That is why `AmbiguousRegion` does not fire on export — the face's
boundary really is its edges, and the file it writes is honest.

So the last two entries' conclusion was wrong in its central claim. Nothing needs
synthesising; `parameter_outline` is not the missing vocabulary; the importer is
not being asked to invent a boundary. It is being asked to chain four edges into
the same three rings the kernel chained them into, and it does not.

Which is a much smaller problem than "the importer has no notion of a seam", and
it explains why four increasingly elaborate rules all failed: every one was
solving the wrong problem, and the second and third made things worse precisely
because they replaced a correct edge-derived boundary with an invented one.

The check that would have caught it is the one this file keeps relearning: before
building machinery to supply something, measure whether it is already there. Two
entries and four implementations, against one probe printing off-edge counts.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Four rings and no outer

Printed how the rod's wall chains, out and back:

    OUT  e0 147v  e1 148v  e2 18v  e3 18v
         derived rings [(34, outer), (146, hole), (147, hole)]
    BACK e0 101v  e1 101v  e2 148v  e3 147v
         derived rings [(100, ..), (100, ..), (147, outer), (146, ..)]

The kernel chains its two 18-point rims into **one** 34-point outer ring. Read
back, those rims arrive as 101-point circles — the exporter wrote them as
`CIRCLE`s and the importer sampled them to tolerance — and they stay **two**
rings. So the face has four rings and no outer at all, and the largest by area is
one of the traced holes.

That is why every rule failed, and it is the same reason each time: all four took
the largest ring for the outer, and on this body the largest ring is a hole. The
elaboration was beside the point.

It also says the fix is not in the rules at all. Two rims that bound a band must
chain into one ring, as they do in the kernel, and they cannot here because the
importer has no seam to join them along — the kernel's pair share seam vertices
and the imported pair do not, having been rebuilt independently from two
`CIRCLE`s.

So the earlier guess about seam vocabulary was right after the fact, but for the
wrong reason: not that the outer boundary must be synthesised — it must not — but
that two rims cannot be chained into one boundary without the seam that joins
them.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Where this thread stands

A long run of entries on one chained boolean has ended at a boundary between the
kernel and STEP, so it is worth stating the position plainly rather than opening
a fifth attempt on the same wall.

**Landed, and in the tree:**

  * seam runs made into edges, and edges cut where they cross a seam, so a face's
    boundary is made of edges — including along and across the seam;
  * rims cut per stretch, so an edge is shared by exactly two faces;
  * a two-point curve given a midpoint, so a straight intersection leaves
    provenance behind it;
  * a sampled curve clipped by where it runs rather than by where its chords sag;
  * `EdgeFaceCount` counting *uses*, so a face may meet itself;
  * STEP: `VERTEX_LOOP` for a pole, `AmbiguousRegion` and `SeamBoundary` where a
    region cannot be stated, and a corpus guard that has caught three silent
    losses since.

**Measured, correct, and not in the tree**, each blocked by the next:

  * a trace box drawn where *both* solids are. Takes a drill's two rings from one
    to two, makes a bored ball cut twice **valid**, and makes a rod crossed by a
    rod **resolve** where it declines today.
  * settling a shared vertex onto both its surfaces, which the above needs: a
    minted point was 1.35e-4 off one surface and 1.162e-3 off the other.
  * four rules for giving an imported face its holes, all wrong for the same
    reason — each took the largest ring for the outer, and on the body in
    question the largest ring is a hole.

**The one thing between them:** an imported face's two rims arrive as separate
circles and cannot chain into the single boundary the kernel gives them, because
the seam that joins them is not in the file. Everything else measured on this
path is sound.

The kernel is strictly better with the first two changes and strictly worse
through STEP with them, and this file's rule has been to ship neither half of
that. Which is right, and worth recording as the reason the box fix is not in the
tree despite being the only change in this run that improves two cases at once.

Suite: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib. wasm32 and
the example build; one pre-existing clippy line in `provenance.rs`.

## Not a shared vertex either

The last entry said two rims cannot chain into one boundary without the seam that
joins them. `face_loops` chains segments by shared vertices, so that is
checkable:

    OUT  rims 16 and 16 vertices, share 0
    BACK rims 100 and 100 vertices, share 0

Neither pair shares a vertex. So the kernel's 34-point outer ring is not its two
rims joined at a seam — and 34 is not 16 + 16 either, so it is not simply the two
rims at all. What makes one ring there and two here remains unaccounted for.

That is the fifth explanation offered for this one difference and the fifth to
fail, and they have been getting smaller: the importer needs synthesised
outlines, then it needs seam vocabulary, then it needs both hole rules, then it
needs the seam that joins the rims, now none of those. Each was a guess dressed
in the previous measurement.

Stopping this line here rather than offering a sixth. What is established and
worth keeping: the kernel changes are correct and measured, the STEP round trip
of one newly-resolved case is what blocks them, and the difference lives in how
`face_loops` chains one body's edges into three rings and another's into four.
The next attempt should start by printing that chaining step by step, not by
proposing what it must depend on.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## `face_loops` was answering a different question

Printed the chaining rather than proposing what it depends on, as the last entry
said to. Chaining both bodies' segments by hand:

    OUT  raw rings [147, 148, 17, 17]    face_loops [34, 146, 147]
    BACK raw rings [101, 101, 148, 147]  face_loops [100, 100, 147, 146]

Both chain into **four** rings. The kernel's `face_loops` reports three, and 34
is not among the four — because `face_loops` returns a face's **stored** loops
when it has any and only derives when it does not. The kernel's face stores its
loops; the imported one does not. Every comparison in the last five entries was
between a stored value and a derived one, which is why each explanation of the
difference fit the evidence and none survived the next measurement.

With that, the measurements reconcile. The 34-point outer is the outline the
boolean stored, running rim, seam, rim, seam. Its vertices are edge vertices —
hence nothing reads as off-edge — but its *order* is one no chaining of those
edges produces. Both facts were measured and looked contradictory; they are not.

So the conclusion four entries ago was right and its reasoning was wrong: an
imported face cannot recover that boundary by deriving, however the rings are
classified, because the boundary is not a derivation. It has to be given one —
which means the file has to carry it, or the reader has to rebuild it from the
seam, and neither is a rule about rings.

Five explanations, one probe. The probe was the one the entry before last said to
write, and writing it took less time than any of the five.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Two steps with no edge

The last entry concluded the file has to carry the stored outline, and the
export already knows how to write a stored ring — walk it, match edges into it.
So the question is whether this outline *can* be walked:

    ring0:  34 pts, 0 off-edge, 2 steps no edge covers
    ring1: 146 pts, 0 off-edge, 0 steps
    ring2: 147 pts, 0 off-edge, 0 steps

The outline is sixteen points along one rim, **one step across the seam**,
sixteen along the other, and one step back. Exactly two segments have no edge,
and since the rims close, both are the same line traversed in opposite
directions.

Which explains the last contradiction cleanly: every vertex is on an edge — hence
0 off-edge — and only the two *crossings* are uncovered. `materialise_seams`
looks for vertices no edge names and finds none. `materialise_shared` pairs
uncovered runs across two faces and these are one face's. Both miss it, and both
are one predicate away from catching it.

So the fix is small and now exactly specified: a run of uncovered *segments*
whose reverse also appears in the same ring is a seam, and wants one edge named
twice — the same conclusion `materialise_seams` reached for a band, arrived at
from the other end. With that, this face's outline becomes writable, the file
carries it, and the importer has something to chain.

That is the first statement in this run that is small enough to be wrong quickly.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The whole suite green, and one chained case lost

Implemented the predicate the last entry specified: a run of uncovered segments
whose reverse appears in the *same* ring is a seam, and gets one edge the face
names twice. With it, plus the trace box drawn where both solids are and the
settle at edge construction:

    34 boolean, 21 brep, 13 brep_csg, 17 nurbs, **12 step**, 671 lib — all green
    wasm32 and the example build; clippy unchanged

Every test in the repository passes, including the STEP corpus guard that has
blocked this combination for six entries. The bored ball cut twice is valid, the
rod crossed by a rod resolves and round-trips, and the file carries a face whose
outline runs rim, seam, rim, seam.

**And chaining goes from 9 of 12 to 8 of 14.** Two more *first* cuts now resolve,
which is why the total moved; but `ball: bore then bore Difference`, which
chained before, now comes back with 77 open edges. That case is in no test, so
the suite cannot see it.

Reverted on that. A green suite is not the standard this file has been holding —
the standard is that nothing measured gets worse — and a case that used to
produce a solid and now does not is worse, whatever the tests say. It is also
exactly the kind of thing the chaining probe exists to catch, and the reason it
is run every time rather than trusted to the suite.

What this establishes: the combination is complete enough to pass everything
written down, and one unwritten case away from being right. The next entry has a
single failing case to chase rather than a class of problem, and all three
changes are recorded in the file to reapply.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The trade is the box's alone

Isolated the case the last entry lost, by removing one change at a time:

    box + settle + seam pairing: NotWatertight { open_edges: 77 }
    box + settle:                NotWatertight { open_edges: 77 }
    box only:                    NotWatertight { open_edges: 77 }
    baseline:                    Ok, 4 faces, valid

The settle and the seam pairing are clean. **The trace box alone** costs
`ball: bore then bore Difference`, and it is the same change that makes a bored
ball cut twice valid and a rod crossed by a rod resolve and round-trip.

So the position is exact: one change, two cases gained, one lost, and the whole
written suite green either way. That is a trade rather than a bug, which is a
different thing from everything else in this run — every previous blocker was
something wrong that could be made right, and this is a box that is too tight
for one pair and too loose for two.

Which suggests the box should not be one box. It is computed once for the whole
operation and handed to every face pair; the pair that loses is a cross-bore
against a bored ball, where the intersection of the two *solids'* boxes is a poor
description of where that pair's curves can be. A box per face pair — the
intersection of the two faces' own extents — would be tighter still for the two
that gain and looser for the one that loses, and it is the thing this file first
suggested when the step and the seed spacing were traced to one number.

Recorded at `box_both` with the isolation. Suite green: 34 boolean, 21 brep,
13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A box per pair does not settle it either

The last entry proposed that the box should not be one box: computed once for the
whole operation and handed to every pair, it is a poor description of where any
particular pair's curves can be. So: a box per face pair, the intersection of the
two faces' own extents.

Whole suite green — 34 boolean, 12 step, 639 lib — and the same case still fails,
at **139** open edges rather than 77. Chaining 8 of 14, exactly as with the
coarser intersection box.

So the loss is not that one box serves every pair, and tightening further makes
that pair *worse*. A cross-bore against a bored ball wants the looser trace; the
bored ball cut twice and the rod crossed by a rod want the tighter one. Three
boxes tried — model union, solid intersection, face-pair intersection — and the
same one case is on the wrong side of every division.

Which is worth stating as the shape of the problem rather than another lead: this
is not a box that is wrong, it is a *step* that is derived from a box. `march`
takes `span * 0.01`, so a trace's fineness is set by the region it is allowed to
wander in rather than by the curve it is following. Every box tried has been an
attempt to control the step by proxy. The note on `march`'s step has said so from
the beginning, and it is the thing to change — the step from the curve's own
curvature, and the box left to be only a bound.

All three changes reverted. Suite green: 34 boolean, 21 brep, 13 brep_csg,
17 nurbs, 12 step, 639 lib.

## A step that follows the curve costs a case too

The last entry said the boxes were all proxies for a step, and the step should
come from the curve. Implemented that in `trace`: halve where the chord's
midpoint falls further from the curve than the tolerance allows, grow where it
falls much nearer, with the old step as the ceiling.

    a_result_can_be_cut_again: NotWatertight { open_edges: 58 }

With the settle alongside it: the same. The rest of the suite passes — 12 step,
639 lib, and the other 33 boolean tests — so the change is close to neutral, and
it costs the one case that has been the subject of this whole run.

Which is the fourth thing to cost that case, after three boxes, and the pattern
across all four is the same: any change to *how the curve is sampled* breaks it,
whether the samples get finer, coarser, or unevenly spaced. That is a stronger
statement than any of the four individually, and it points somewhere none of them
did — not at the sampling, but at whatever downstream depends on the sampling
being what it currently is.

The fixed step is uniform, and uniform is a property nothing has needed to state
because it has always been true. A curve traced at a constant step gives a face
the same points on both sides of every shared boundary, and the pieces either
side of it agree by construction rather than by tolerance. An adaptive step gives
each trace its own spacing, and two traces of the same curve — from different
seeds, or in different directions — need no longer agree.

Reverted, with that recorded at `march`'s note, which has asked for this change
since long before this run and now has a reason it cannot simply be made.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Correction: not uniformity

The last entry explained the adaptive step's cost by saying something downstream
depends on the sampling being uniform — a curve traced at a constant step gives
both sides of a boundary the same points, and an adaptive one does not. That was
written as a conclusion and not measured, which is the error this run keeps
correcting. Measured now:

    2 faces, 4 edges, defects [EdgeFaceCount { edge: 0, uses: 1 }]
    free ends [0, 0]
    e0 sphere/sphere 59 verts closed true

The drill's **wall is missing** — two faces where there should be three — and an
edge reads `sphere/sphere` because only one face ever claimed it and the pair was
never completed. That is the missing-piece failure from the middle of this run,
the one a *coarse* trace gives. Nothing about it says two samplings disagreed.

So an adaptive step loses a piece for the same reason a wrong box does, and the
uniformity story explained nothing. Both wrong guesses had the same form: a
plausible mechanism fitted to a number, with the body never opened.

What stands after this tick is narrower and duller than what the last one
claimed: four changes to how a curve is sampled, four losses of the same case,
and in the one instance opened the loss is a face that was not built. The next
attempt should open the body first — for every one of the four, not just this
one — and see whether the same face goes missing each time. That is one probe
against four hypotheses, and this run's record says it will beat them.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Identical topology, different tessellation

Opened the body under each sampling change, as the last entry said to — and the
first attempt probed the *wrong case*, the drill from `a_result_can_be_cut_again`
rather than the cross-bore from the chaining probe. That one is fine under the
box: 3 faces, 0 defects, closed shell, 0 open edges. Corrected, and the real case
says something quite different from either earlier guess:

    baseline 4 faces 7 edges 0 defects shells [true] free [0,0,0,0]  326114 tris,  0 open
    box      4 faces 7 edges 0 defects shells [true] free [0,0,0,0]  373283 tris, 77 open

**The topology is identical.** Same faces, same edges, no defects, no free ends,
one closed shell either way. Nothing is missing and nothing is mis-claimed. The
only difference is the sampling — 47,000 more triangles — and at the finer
sampling the fill leaves 77 edges open.

So the box does not lose a piece here. Both earlier readings were wrong: not a
missing wall (that was the adaptive step, a different configuration), and not two
samplings disagreeing. This is a **fill that does not survive its input getting
finer**, which is a different repair from anything tried against it — and a
better-conditioned one, because the body going in is provably right.

The wrong-case probe is worth recording too. Two configurations, two cases, and I
compared one config's case against the other's conclusion for a whole tick. The
habit that would have caught it is the one that keeps recurring here: state which
case a number came from, in the same breath as the number.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The fill opens because two faces disagree about one vertex

Followed last tick's finding — identical topology, finer sampling, 77 edges open
— down to the actual mechanism, by tagging every boundary edge with the face
whose triangle owns it.

Under a re-derived trace box the cross-bored ball leaves **2 open edges, both on
one face**, all three of their endpoints at x = -2.8914: the rim where the
cross-bore exits the sphere. Printing both rings along that rim side by side:

    sphere (f0 l2, 64 pts):   … (-0.2736,-0.7518)  (-2.8916, -0.2217, -0.7678)  (-0.1974,-0.7753) …
    bore   (f2 l0, 65 pts):   … (-0.2736,-0.7518)  (-2.8914, -0.2220, -0.7686)  (-0.1974,-0.7753) …
                                                    ^ and listed twice in a row

Every other point on that rim is shared to the last digit. One is not: the same
rim vertex, arrived at down two different faces, **8.9e-4 apart**. Inside the
1e-3 tolerance, so by definition one point — but the endpoint weld accepts a
match only within `tolerance * 0.5`, so it becomes two, and the two fills
disagree by exactly the two segments around it.

So the failure is not sampling density and not a lost piece. It is **vertex
identity across faces**, and the sampling change merely stops the two arrivals
from coinciding by luck.

Two candidate repairs, both measured and neither shipped:

* **Weld radius = tolerance** (from half). Whole suite green; all 45 cases of a
  chaining corpus byte-identical, decline reasons and open-edge counts included.
  Exactly inert — so it ships with one of the sampling changes, not before.
* **Drop repeated ring points.** Scanned every result in the corpus: not one
  loop anywhere has a repeated consecutive point or a same-vertex pair. Also
  specific to the box.

Recorded at the weld site in `boolean.rs`, which is where the next attempt at any
of the four sampling changes will land.

The re-derived box is *not* the variant measured two ticks ago as whole-suite
green: this one gains a chaining case (11→12/45, `cube: bore then cross` starts
resolving, `rod: cross then …` stops being `NotASolid`) but costs 3 tests. Worth
noting the corpus itself is new and larger — 45 chains, not the 12 quoted
earlier — so the two numbers are not comparable, only the diffs within a run are.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Why the two arrivals are far apart: the crossing vertex is off the curve

Chased last tick's 8.9e-4 to its source. First, two things it is *not*: marched
points are settled well (worst deviation over both surfaces 9.7e-7 across every
curve in these cases), and a closed trace's wrapping chord is *shorter* than its
regular segments (2.6e-2 against a step of 1.3e-1), not longer.

The rim that opens is `sphere × cross-cylinder` — closed form, an exact circle at
x = -2.8914. Dumping that ring at full precision, with each point's distance to
every other surface:

    f0 (sphere) ring #29, vertex 613   (-2.891604, -0.221748, -0.767759)
        distance to the cross-cylinder   8.59e-4
        every other point on that rim    < 1e-6

One point. A crossing vertex, spliced into the rim, **never settled onto the
other surface the rim lies on** — so it sits off the shared circle, and each face
mints its own copy from a different direction. That is what puts the two 8.9e-4
apart, and it is why widening the weld radius fixes the symptom.

Then the part that matters for shipping: this is **live in the current tree**,
not an artefact of the box. Scanning every ring of the 45-case corpus for points
near another surface but not on it:

    ball − cross    v226/v227   2.64e-3 off the cylinder
    rod  − cross    v80/v81     2.64e-3 off — and the *same* body vertex
                                reprojects 2.8e-3 apart between its two faces

Nearly three times the tolerance, in results that pass today. They pass because
the fill takes a ring vertex's position from the **body vertex**, not from that
face's own `uv` — so the two faces disagree about where the point is and the
disagreement never reaches the mesh. It is a crack held shut by the fill, not an
absent one; every sampling change that has failed here has failed by prying it
open.

So the repair to try is not the weld, which treats the symptom: **settle a
crossing vertex onto both surfaces its rim lies on, and recompute each ring's
`uv` from the settled point.** Untried. Recorded in `boolean.rs` at the weld,
with the vertices to check it against.

Two ticks running, the probe was written and run once with the env var unset and
returned nothing, which reads exactly like a clean result. Worth a guard when the
next one is written.

Suite green: 34 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: a crossing is put on both of the surfaces whose rim it cuts

Built the repair the last tick specified, and it holds.

`settle_shared_vertices`, run over the assembled body just before the seams are
materialised: take every vertex claimed by exactly two surfaces through the loops
that name it, run the marcher's own Newton step to put it on both, and recompute
each ring's `uv` from the settled point — unwrapping near the old value so a
periodic parameter does not jump a period. Guarded three ways: exactly two
claiming surfaces, a move no larger than a few tolerances, and only when it lands
nearer both surfaces than it started.

What it does, measured:

* **The off-rim points are gone.** Scanning every ring of 16 results for points
  near a surface without being on it: previously `ball − cross` at 2.64e-3 and
  `rod − cross` at 2.64e-3 (with the same body vertex reprojecting 2.8e-3 apart
  between its two faces). Now zero, everywhere.
* **Chaining 11/45 → 12/45.** `ball: bore then cross Union` resolves; it used to
  decline `NotWatertight { open_edges: 1 }` — one open edge, bought by one
  crossing that was on neither curve.
* **`rod − cross` is now a solid.** The five `rod: cross then …` chains stopped
  declining `NotASolid`, which they did because the *first* cut failed the
  solidity check. They now get as far as a real answer or a real reason.
* No case anywhere moved backwards, and the whole suite is green.

Two tests pin it, and both were checked to fail without the change — the first
with exactly the number above:

* `a_crossing_sits_on_both_of_the_surfaces_whose_rim_it_cuts` — for a bored ball
  and a bored rod, no ring point may sit *near* a surface it is not on. Close to
  another surface and not on it is the signature of a point meant to be shared
  and not shared: either it is on that surface, or it has no business being
  within a thousandth of it.
* `a_rod_can_be_added_to_a_ball_that_has_already_been_bored` — the union that the
  single open edge used to cost, asserted as a valid solid and strictly larger
  than the bored ball.

This is the crack the four sampling changes kept prying open. Worth re-running
each of them now that it is shut — the trace box, the per-face-pair box, the
adaptive step, and the settle — since the reason all four failed was measured to
be this and not their own idea.

Suite green: 36 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: a body with no vertices was invisible to the trace box

Re-ran the trace box with the crossing crack shut, as the last entry said to. It
gained a chaining case (12→13/45) but broke `touching_is_told_apart_from_crossing`
— the kernel stopped recognising a tangential contact, which is the safety rule
rather than a coverage number. Printing the box it computed for that pair:

    union box          [-3.401, -4.401, -3.401] .. [ 5.401, 4.401, 3.401]  -> 2 curves
    intersection box   [ 1.8e308, 1.8e308, 1.8e308] .. [-1.8e308, …]       -> 0 curves

The low corner is above the high one. **A whole torus has no vertices** — one
face, no edges — so a vertex-derived extent for it is empty, and intersecting an
empty extent with anything gives an inverted box. Three ticks of reading that
failure as evidence about the box's *idea*, when it was a hole in this particular
derivation. Guarding it (fall back when either body has no vertices) recovers the
tangency test outright.

Then the part that mattered more. The **shipped** box is vertex-derived too, so
two vertexless bodies have always handed the marcher an inverted box:

    torus − torus    -> ok, valid=true, 1 face
    torus − ball     -> ok, valid=true, 1 face

No curve found, so the boolean concluded the two do not meet and returned the
first body untouched — watertight, `is_valid_solid`, and wrong. A ball centred on
the tube's own centre circle and half again as wide as the tube, and the
difference kept **78.917 of 78.917**. That is precisely the outcome this file's
header forbids, and nothing downstream could have told.

`extent_of` now takes a vertexless body's extent from its faces, sampled across
their parameter ranges. Both cases decline with a named reason instead
(`TangentialContact`, `NotWatertight`), the 45-case corpus is byte-identical, and
the suite is green. Pinned by
`two_solids_with_no_vertices_are_not_assumed_to_miss_each_other`, which fails
without it with the volume above.

Still open on the box itself: with the vertexless guard it costs exactly one case
— `ball: bore then cross Difference`, 2 open edges — plus the step corpus guard,
and gains two. Widening the weld no longer changes that at all (identical
output), so those 2 edges are not the weld and want their own probe.

Suite green: 37 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The box's last case is a fill that tears, not a boolean that fails

Probed the 2 open edges the trace box still costs on `ball: bore then cross
Difference`, since the last two probes of this case each turned up a real bug.

The endpoints are all on their surfaces to 1e-6 — the crossing settle holds, so
these are not off-rim points. And the two faces' rings agree: vertices 96..104 in
the same order in both, with face 2's stretch at constant `v` and uniform `u`,
which is the rim circle exactly as it should be.

But the open edges are `(90,100)` and `(100,124)`, and those three sit at ring
positions #307, #254 and #278 of a 309-point ring. Not neighbours. So they are
not ring edges at all — they are **interior edges of the fill, used once**: a
hole punched through a triangulation whose boundary was correct going in.

Comparing the ring either way:

    shipped   n=109   u[-0.2493, 6.0339]   span 6.2832   area 13.0615
    box       n=309   u[ 2.8604, 9.1436]   span 6.2832   area 13.0616

The same closed loop, spanning exactly one period, with the same area. The only
difference is that the box's is sampled three times as finely — and the finer one
tears. My first reading of that 8.976 was that the ring wrapped more than once in
parameter space; it does not, the span is exactly one period in both.

So the box's remaining cost is not in the boolean. It is `fill_planar` failing to
tile a period-spanning ring as a function of how densely it is sampled, and
failing *silently*: the face is not `carried_through`, and the tear surfaces only
as open edges with no face named. Recorded on `fill_planar`, which is where a fix
belongs and which is a different subsystem from the curve work these ticks have
been in.

That closes the box for now. It stands at +2 cases, −1 case, −1 step guard, and
the −1 is this. Worth noting what the four sampling changes have actually been
worth: not one of them has shipped, but chasing why they fail has produced the
crossing settle, the vertexless extent, and this — two shipped fixes and a
located defect, all of them latent in the tree with the box nowhere near it.

Suite green: 37 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The tear reproduces without the box, and it is not the clip

Went after the fill tear the last entry located, starting with whether it needs
the box at all. It does not — the ring's density comes from the tolerance, so
asking for a finer one reproduces it:

    1e-3  ok, 0 open        5e-4  ok, 0 open
    2e-4  NotWatertight { open_edges: 4 }
    1e-4  ok, 0 open        5e-5  ok, 0 open

Non-monotonic, in the shipped tree, reachable by a caller who just wants a finer
result. Worth being exact about what that is: the 2e-4 case **declines**, so it
is a coverage gap, not a broken promise — the answer is honest, there just isn't
one.

Then two hypotheses, both wrong, both cheap to kill:

* **`earclip` stalls and returns a partial fan.** It has exactly that path —
  `if !clipped || guard > n * n + 8 { break }`, silently keeping whatever it had.
  But instrumented, it never stalls: not at 2e-4, not anywhere in the 45-case
  corpus, not anywhere in the suite. The silent-truncation path is latent, not
  live, so a guard on it would be untestable and is not worth writing yet.
* **The vertex is off-surface, like the crossing was.** Generalised
  `settle_shared_vertices` to vertices named by only *one* face's ring, finding
  the second surface by proximity. Changed nothing. Reverted.

What it actually is: three of the four open edges form a triangular gap
straddling two faces at the bore rim (z = -2.8284, radius 1), and the odd corner
is a body vertex that the sphere's ring carries and the bore wall's ring does
not. On both surfaces, correctly placed — just not shared. A **ring mismatch**,
where the box's failure was a hole inside one face's fill. Two different
failures, and last entry's framing covered only the first.

Left behind: `the_same_cut_asked_for_five_ways_is_either_right_or_declined`,
which asserts the file's rule across all five tolerances and records that four of
them resolve. It would catch the case that matters most — a finer tolerance
starting to return a solid instead of declining.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A ring may name a vertex twice, and that is the bridge, not a fault

Chased the 2e-4 ring mismatch. The two faces on the bore's rim carry 159 and 160
points, and diffing them by angle turned up no point present in one and absent
from the other — the counts differ because the cylinder's ring names **vertex 449
twice**, at ring positions 0 and 159 of a 320-point ring, with the whole rim
sitting in that first stretch.

That reads exactly like two rims concatenated into one polygon, which no ear clip
can tile. So: split a ring where it returns to a vertex it has already used.
Every one of the five tolerances went from four-of-five resolving to **none**,
379 to 959 open edges.

The premise was wrong, and measuring it says so plainly. Repeated vertices are
*normal* in a stored ring, because a hole is bridged into its outer ring by
running out to it and back:

    ball − cross    f0 l0   n=128   63 repeated vertices, gaps 126, 124, 122, …

That descending-by-two run is a there-and-back traversal read off directly. Two
rims joined into one ring are indistinguishable from it by this test — and the
passing 1e-3 case has repeats too (`ball-bore-cross f2 l0`, n=109, 2 of them). So
the repeat at 2e-4 is not the defect, and the 2e-4 failure is still unexplained.

Recorded on `face_loops`, with the numbers, so the next attempt does not spend a
tick re-deriving that a ring is allowed to revisit a vertex.

Three hypotheses killed on this failure now — the clip stalling, an off-surface
vertex, a pinched ring — each cheap, each leaving the tree as it found it. What
is established: at 2e-4 two faces produce a triangular gap at the bore rim, the
vertices involved are on their surfaces and correctly placed, the rings agree
angle-for-angle, and the ring structure is legitimate. The disagreement is in
what the fill does with them, not in what it is given.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The spur leaves at one vertex and comes back at its neighbour

Kept after the 2e-4 failure, and it is now located — in the boolean, not the
fill, which is where the last three entries had been looking.

First a fourth hypothesis killed. `refine_on_surface` drops a flat sliver when
`t.iter().all(|v| run.contains(v))`, and `run` is the **union** of every replaced
chord's points in that round — so a sliver whose own chord was left alone can be
dropped because its corners happen to appear among another chord's. Narrowed it
to one set per run. Inert: 2e-4 still fails with the same four edges. Reverted.

Then the direct question, which settled it. The odd vertex is

    V688 = (0.000000, 1.054213, -2.808671)

1.0542 from the axis, so **not on the bore's rim at all** — it is on the sphere,
and the sphere's stored ring reads:

    f0  … 528, 370, 688, 689 …    … 690, 689, 688, 371, 372 …
                 ^ out at 370              ^ back at 371
    f1  … 528, 370, 371, 372 …

A spur, run out and back — and it leaves at 370 and returns at **371**, the next
vertex along the rim. So the two legs bound a thin wedge that nothing covers, and
those legs are two of the four open edges; the third is the rim segment (370,371)
on the face opposite. A spur that returns to its own start bounds nothing and
costs nothing, which is why the same structure is harmless everywhere else.

`bridge_holes` is ruled out as the source: its splice is
`ring[..=b] + hole + ring[b..]`, repeating both ends, so a bridge it builds is a
slit of no width. This ring is a **stored** `TrimLoop` — the boolean's own output
— so the fault is upstream in `from_pieces`, and the fill is only where it shows.

Recorded on `face_loops` next to the constraint from last entry, since the two
are read together: a repeated vertex is normal, a spur that comes back to the
wrong vertex is not.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The excursion is a whole ring, and the anchor is emitted once instead of twice

Narrowed the 2e-4 spur further. Two facts settle where it is not.

**The rim edge is clean.** Of six edges in the result, exactly one touches the
vertices involved:

    EDGE 2  surfaces (0,1)  n=160  closed=true  head [370,371,372,…]  tail […,370]

A proper closed rim. So 688–690 appear in *no* edge — they exist only in the
stored `TrimLoop`.

**And it is not a whisker.** The excursion runs from ring position #159 to #572
of a 573-point ring: 413 points, most of the face. It is a second ring spliced
into the first, and the splice lands on 370 going out and 371 coming back.

The walk in `planar::subdivide` pushes `nodes[cur.from]` for *every* half-edge,
so a node the walk passes through twice is emitted twice — which is exactly what
a dangling chord needs, since the ring leaves and re-enters the boundary at the
same node. Written once, the ring jumps from the chord's far end to whatever came
after that node, and the wedge between the two legs is bounded by nothing. Rotate
the observed ring and it is precisely `raw[1..]`:

    walk      370, 371, …, 528, 370, 688, …, 688
    stored         371, …, 528, 370, 688, …, 688     → wraps 688 to 371

So the walk is right and one of the two anchor emissions is dropped downstream,
between a `Region` and the `TrimLoop` the face stores. That is a single, checkable
next step: print the region ring as `subdivide` returns it for that face and diff
it against the stored loop.

Recorded at the walk, where the invariant it must preserve now sits in writing.

Five hypotheses killed on this failure — the clip stalling, an off-surface
vertex, a pinched ring, the pooled sliver drop, and `bridge_holes` — and the
search has narrowed from "somewhere in the fill" to one copy step.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Not a spur — a seam crossed at matching v on one side only

Diffed the region ring against the stored loop, as the last entry said to, and it
corrected that entry rather than confirming it.

`ring.uv = region.outer.uv` is a straight move, so nothing is dropped in the
copy. And the stored ring has **zero duplicated positions** — all 573 `uv` points
are distinct. The repeated vertex *indices* are two different parameters naming
one 3D point, which on a sphere means the seam. Measured:

    f0 v688   #159  u=7.853982   |   #572  u=1.570796    du = -6.283185 = -2pi

Exactly one period. So the excursion is a seam wrap, not a path run out and back,
and last entry's spur reading was wrong.

What is actually wrong is narrower and sharper. Both sides of a seam are the same
line, so the non-periodic parameter must agree across it:

    #158  v370  u=7.853982  v=-1.230959      right of the seam
    #572  v688  u=1.570796  v=-1.211726      left of it
    #0    v371  u=1.610563  v=-1.230959      0.0398 *past* the seam

At `v = -1.230959`, the latitude where the bore's rim meets the seam, the right
side carries the rim vertex and the left side carries nothing — so the ring steps
to the next rim vertex along. The two legs of that step bound the wedge, and they
are two of the four open edges. It bites only when a seam crossing lands on a rim
vertex, which is exactly why the other four tolerances are clean and why this
took six hypotheses to corner.

Recorded on `split_at_seam` with the parameters above. The repair is stateable
now: a ring that crosses the seam must carry the crossing on both sides at the
same `v`, and inserting the missing one is a local edit to a ring that is
otherwise correct.

Six killed on this failure — clip stall, off-surface vertex, pinched ring, pooled
sliver drop, `bridge_holes`, dropped anchor emission — and the two that read as
confirmations (`spur`, `dropped copy`) were retracted by the next measurement.
Worth noting the pattern: every reading taken from *vertex indices* has been
wrong, and every one taken from *parameters* has held.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The obvious repair works on the case and is not the repair

Built what the last entry specified: `close_seam_runs`, a pass over stored loops
that finds a ring spanning exactly one period in `u`, takes the two runs along
the seam, and where one stops short writes in the point the other already
names — the same body vertex, at this side's `u`. Nothing invented.

On the case it works. **Four open edges to one.** The remaining one is a
different fault: a 66-degree chord across the *top* rim, `(608,638)`, length 1.09
— the fan-chord limitation `refine_on_surface` already documents, not a seam.

It also breaks five tests:

    a_cylinder_parked_on_an_edge
    a_bore_across_a_rod
    a_crossing_sits_on_both_of_the_surfaces_whose_rim_it_cuts
    what_comes_out_is_a_solid_that_can_go_back_in
    a_result_is_always_watertight_or_declined_never_neither

and `rod - cross` stops being a solid, which takes five chains out of the corpus
with it. Narrowing to a side short by exactly one point — the gap is 0.019233
against a run spacing of 0.019234, so this looked like the distinguishing
feature — changes nothing: the five that break are one-step gaps too.

So **"both runs along the seam span the same `v`" is not an invariant**, however
much it reads like one, and five tests are the evidence. That is the useful
result of the tick: not the repair, but the fact that the invariant it rests on
is false, measured rather than argued. Recorded on `split_at_seam` alongside the
case, so the next attempt starts from a true statement instead of this one.

Two other things now known and worth keeping together: the target ring's own
short side is real and its repair does close the wedge, and the last edge at that
tolerance is a fan chord, not a seam — so even a correct seam fix leaves 2e-4
declining, and the coverage gap needs both.

Suite green: 38 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: a seam is reached from both sides at the same place

The invariant the last entry refuted was refutable because it was stated too
widely. Narrowed, it is true and it ships.

The claim is not "a ring spanning a period has matching `v` at its extremes" —
that is false, and five tests said so. It is: **where a ring runs *along* the
seam on both sides — two or more points at one `u`, a vertical run in parameter
space — those two runs describe one line and must end together.** A ring can span
a whole period without ever lying along the seam, and then its extreme points are
single and have no reason to agree. Requiring the run is the whole difference:
with it the suite is green, without it those same five fail.

`close_seam_runs` writes in the point the short side is missing — the vertex the
other side already names, at this side's `u`, nothing invented — guarded by the
run on both sides and by the gap being one step of the run's own spacing.

Measured:

* `ball - bore - cross` at 2e-4: **four open edges to one**. The last is a
  different fault, the 66-degree fan chord across the top rim that
  `refine_on_surface` already documents. So 2e-4 still declines and the coverage
  gap needs both.
* The 45-case corpus is byte-identical, and the whole suite is green.
* It fires on two corpus cases that **already resolve** — `ball - cross` and
  `ball - cross - cross` — so it is correcting rings in results that ship today.

That last point is what makes it testable, and
`a_seam_is_reached_from_both_sides_at_the_same_place` pins it on `ball - cross`
at 1e-3. Without the fix it reports the ring running

    -1.570796 .. 1.521709   on one side
    -1.521709 .. 1.570796   on the other

— each side reaching the pole at one end and stopping 0.049 short at the other.
A crack held shut by the fill is still a crack; this is the one place in the
corpus it can be caught in a result that resolves.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: a chord is found even when its triangle has no area

The last fault at 2e-4 was the 66-degree fan chord across the top rim, and it
turned out to be one word in a filter.

`refine_on_surface` collects the chords to replace `from the triangles that have
area` — `faces.iter().filter(|t| !flat(t))`. Keying the result by *edge* is what
stops a chord being replaced on one side and left on the other, and that guard is
sound. But skipping flat triangles when **finding** the chords is a different
thing, and it misses a whole class: three corners on one rim is a triangle with
no area in parameter space, so a chord between two of them was never offered to
`between` and never fanned. That is exactly the top-rim chord — 1.09 long,
66 degrees of a rim — with nothing on the other side of it.

Dropping the filter:

    1e-3  ok      5e-4  ok      2e-4  ok      1e-4  ok      5e-5  ok

**All five tolerances resolve.** 2e-4 has gone from `NotWatertight { open_edges:
4 }` at the start of this run to a valid solid, in three steps: the crossing
settle, the seam runs, and this.

Checked it is *right* and not merely closed, which is the failure this corpus
exists to catch — the five volumes are

    86.560971  86.453356  86.449707  86.560666  86.450510

and the new one sits inside the band its neighbours already occupied. The 45-case
corpus is byte-identical and the whole suite is green.

`the_same_cut_asked_for_five_ways_is_either_right_or_declined` now asserts all
five resolve *and* that they agree to one per cent — a solid of the wrong size
being the thing to fear when a decline turns into a result.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The 407 open edges are one face that was never filled

Re-ran the trace box now that the fill faults it kept hitting are fixed. Fourth
attempt, same shape: +2 cases (`cube: bore then cross`), −1 (`ball: bore then
cross Difference`, 2 open edges), −2 boolean tests, −1 step test. It is not the
change, and four measurements say so.

So: the largest failure left in the corpus, `ball - cross` then `- bore`, 407
open edges. Attributed, it is not 407 faults:

    ZZOPEN total 407  per-face [(1,64), (2,64), (3,128), (4,151)]  faces 5  carried 1
      F0 sphere    loops None
      F1..F4 cylinder loops Some([109]) Some([109]) Some([43]) Some([146,43,43])

**`carried 1`.** One face is never filled — the sphere — and the 407 edges are
the four faces around it left closing against nothing. And it had loops going in:

    after `ball - cross`    F0 sphere  loops Some([130, 63, 63])
    after `- bore`          F0 sphere  loops None

A sphere with three holes is not a parameter rectangle, so with no stored loops
the grid fill takes it and fails. The loops exist before that second cut and not
after it.

Tried the fill-side workaround — fall back to `fill_planar`, which derives rings
from the edges, rather than dropping the face. It works: 407 open edges to 288,
142 to 55 on `ball: column then bore`, suite green. But it flips no case in the
45-chain corpus and fires **nowhere** in this crate's tests — six corpus cases
only, all still declining — so there is nothing to hold it in place, and by the
rule the rest of this run has followed it does not ship. Recorded on `tessellate`
with its numbers.

The next thing is upstream and now precisely stated: find where a face's stored
loops are lost between one boolean and the next.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The loops are dropped on purpose, and the reason is a known open problem

Found where `ball - cross` loses its sphere's rings. Not a bug — a decision, with
its cost already recorded beside it.

`split_face` returns early for a face this boolean does not split, and hands it
back as one piece with

    loops: if carried.is_empty() { loops } else { Vec::new() }

A loop is a fixed polyline and an edge gets *refined* to tolerance, so carrying
both makes them diverge the moment the boundary is curved — a cap holding a
16-point rim beside a wall that refined the same circle to 145, coming apart
along every one of the 144 segments between. The comment there already names what
keeping them costs (1.3% of a blind hole's volume) and what it is waiting on:
**the trim-loop refinement problem, still open**.

This is the third thing waiting on it, and the largest failure in the chaining
corpus: sphere `loops Some([130, 63, 63])` before the second cut, `None` after,
grid fill takes a sphere with holes, fails, face dropped whole, 407 open edges
around a face that is not there.

Tried keeping the loops only where the face has a hole — a rim is a rectangle's
border and a hole is not, so the grid can never draw one. Two tests went, and this
face was not among the ones it helped: its rings are not holes yet at that point.
Reverted. The fix is the refinement itself, not a rule about which loops to trust.

That makes trim-loop refinement the thing to build next, and it is well shaped:
where a ring is backed by edges, refining it *is* rebuilding it from those edges
after `refine_edges` runs — which is also why both faces on a boundary would then
agree by construction, the property the divergence breaks today.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Three ways to keep the rings, three ways to lose tests

Tried to unlock the 407-edge case by keeping the rings `split_face` drops. Every
form of it costs more than the face is worth:

    keep always                      9 boolean tests, 2 STEP tests, and the
                                     corpus gains four declines it did not have
    keep where the split face has
      a hole                         2 tests — and it never reaches this face,
                                     whose rings are not holes at that point
    keep here where the face has
      a hole                         4 boolean tests, 1 STEP test, though four
                                     `rod:` chains do start resolving

So the divergence the drop exists to prevent is real and large, and no rule about
*which* rings to trust gets around it. That is worth having measured rather than
inferred from the comment.

The prerequisite is already written down in `refine_edges`, from an earlier run:
the walk that rebuilds a ring from its edges after they move was **built,
measured exact on every ring of a bored ball, and left out** — because at the
time nothing measurable would gain by it. That last clause is what has changed,
and I have noted it there: this face is the thing that gains. With the rebuild in
place the rings and the edges cannot diverge, because the rings *are* the edges,
and then keeping them is free.

That is the next build, and it is a real one rather than another one-line
experiment — which is the honest read of a tick that spent three of them.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Built the rebuild; it is not the prerequisite after all

Wrote the trim-loop refinement the last entry called for: read each ring off the
edges as a sequence of steps, refine the edges unpinned wherever every ring of
every face using them reads, then rebuild each ring by concatenating what those
steps gained. Roughly a hundred lines in `refine_edges`.

It works, and it reproduces the earlier run's findings exactly: the walk reads,
the whole suite stays green, the 45-chain corpus is unchanged but for one
decline's edge count, and a bored plate's volume converges to the same six
figures at every tolerance — 291.962362, 291.750239, 291.728162 — identical with
the rebuild and without it.

Then the payoff, which does not come. Keeping the rings at `split_face`'s early
return **still costs eight boolean tests and a STEP test**, exactly as it did
without the rebuild. The reason is in the note that was already there: the rings
that get kept are the ones the walk *cannot* read — parameter outlines, seam and
poles, that no edge carries — so they stay pinned and coarse exactly as before.
The rebuild does not reach them.

So the claim I wrote last entry, that this face is the thing the rebuild would
gain, is **false**, and I have corrected it in `refine_edges` rather than leave a
plausible wrong reason sitting where the next attempt will read it. The real
prerequisite is the sentence that follows it there: a complement face's outline
is the last part of a boundary still not made of edges. Until it is, the rings
that matter cannot be rebuilt, cannot be refined, and cannot be kept.

That is the third independent confirmation of the same limitation, from three
directions — the pin, the drop, and now the rebuild. It is the bottleneck, and
building edges for complement outlines is the work that clears it.

Suite green: 39 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: the seam a ring crosses in one step is now an edge

Went at the bottleneck — a boundary not made of edges — by first measuring which
steps of which rings have nothing behind them:

    ball - cross   sphere outer   n=130   unbacked 130   pole, seam, pole, seam
    ball - cross   the two holes  n=63    unbacked 0
    rod  - cross   cylinder outer n=36    unbacked 2     `v90->v107`, `v107->v90`
    rod  - cross   f3             n=84    unbacked 2

A swept face is backed **except its two seam runs**, and those two are one line
walked up one side of the parameter domain and down the other. `materialise_seams`
misses them for a precise reason: it looks for a *run of vertices no edge names*,
which is what a sampled seam looks like, and a swept face's seam is not sampled —
it is one straight step whose ends are rim vertices that edges do name. The run
never starts.

`materialise_seam_steps` makes an edge of exactly that: a step no edge carries,
walked twice by the one face. Walked *twice* is the test that keeps it off a
boundary shared with a neighbour, which is walked once, and the edge is named
twice by the face because the face uses it twice.

    rod - cross    unbacked 2 -> 0, every ring fully backed
    ball - cross   unbacked 130 -> 2, and the two are the poles

Faces that run to a point are exempt, deliberately: the sphere version costs
`a_face_that_is_everything_except_its_own_boundary_survives`, because a
pole-to-pole seam is a boundary this crate's STEP reader cannot place — the same
cost `materialise_seams` already records. So the gain taken is the swept one,
which has no pole.

Suite green, 45-chain corpus byte-identical, and pinned by
`every_step_of_a_ring_is_a_step_of_an_edge`, which without the fix reports
`f0 l0 step #17 (v90 to v107) has no edge behind it`.

Half the bottleneck, then. What remains unbacked anywhere measured is a pole
step — a ring stepping from a vertex to itself — and that is a different problem
from a seam, since there is no line there to make an edge of.

Suite green: 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Backing the seam does not unlock keeping the rings, and the test says why

With a swept face's seam now an edge, re-ran the chain the last two entries were
aiming at: rebuild the rings from the edges, then stop dropping them.

It still costs nine boolean tests and a STEP test — the same as before the seam
work. But this time the failure list says what is wrong without any probing:

    every_step_of_a_ring_is_a_step_of_an_edge ... FAILED

The test written this run, failing under the experiment. The rings kept at
`split_face`'s early return are exactly the ones with steps no edge carries, so
keeping them puts unbacked rings back into the body and they drift from the edges
beside them — `a_bore_across_a_rod` comes back `NotWatertight { open_edges: 6 }`.

That is worth noting as a result about the tests rather than the kernel: three
ticks ago this took a full tick of probing to establish. Now a named test states
it on the first run. The invariant was worth writing down.

So the remaining blocker is sharp and small. Everything measured is backed except
a **pole step** — a ring stepping from a vertex to itself, crossing the whole
parameter range at `v = ±pi/2`. No edge can be made of it: there is no line
there, only a point the surface collapses to. Whatever answers it is not another
edge, and that is the shape of the next attempt.

Reverted; recorded in `refine_edges` beside the limitation it belongs to.

Suite green: 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Drift is not the reason the rings are dropped

Three experiments, three refutations, and the useful result is that a story three
ticks have been built on is wrong.

**Poles need no edge.** A ring stepping from a vertex to itself crosses a place
the surface collapses to a point: there is no line there to be an edge of, and
nothing that can drift, so the walk can simply carry it. Built that way the
rebuild reads those rings too and the whole suite stays green.

**Backing everything does not unlock keeping the rings.** With every seam given
an edge — poled faces included — poles carried, and rings rebuilt from the edges
after they move, keeping the rings at `split_face`'s early return **still costs
the same nine boolean tests**, plus a second STEP test for the pole-to-pole
seams. Not one test fewer than with none of it.

**And it is not the fill either.** If a kept ring hurt by pushing a face from the
grid fill to the loop fill, then using the grid unless the face has holes should
recover it. It is worse: sixteen boolean tests and three STEP tests.

So the comment at that early return — a loop is fixed, an edge is refined, carry
both and they diverge — describes something real, but it is **not the whole
reason the drop is there**, and three ticks of work aimed at removing the drift
have not moved the outcome by a single test. Recorded at that line, with the
instruction the next attempt should follow: open one failing case and look,
rather than reason from the comment.

`a_bore_across_a_rod` is the one to open. It fails with
`NotWatertight { open_edges: 6 }` — six, on a case that otherwise resolves, which
is small enough to attribute completely.

Suite green: 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The kept ring is stale, not drifting — and restating it moves the number

Opened `a_bore_across_a_rod` with the rings kept, as the last entry said to. It
was the union that failed, not the difference, and six edges is few enough to
read completely:

    face 3 (bore wall)   165 -> 166 -> 167
    face 5 (bore cap)    165 ---------> 167     the cap cuts the corner
      F0 cylinder loops [34, 59, 60]
      F1 plane    loops [16]      F3 cylinder loops [78]
      F2 plane    loops [16]      F4 cylinder loops [79]

**Caps holding sixteen-point rings, walls holding seventy-eight.** A face the
boolean does not split arrives with the sampling it already had, while the
boundary beside it was re-sampled — and no refinement of either afterwards makes
16 agree with 78. The ring is not drifting from its edges; it is *stale*.

That is a different repair from any of the last four attempts. `restate_rings`
writes every ring in the vertices of the edges it runs along, matching by
**endpoint** rather than by adjacency — the cap's step from one rim vertex to
another becomes the rim's own five points, and then the two cannot disagree.

    the union with rings kept      6 open edges -> 0
    restate_rings alone           whole suite green, corpus byte-identical
    restate_rings + keep rings    9 boolean tests -> 6, 2 STEP tests

Neither half ships: alone it is inert, together it is still six tests short. But
it is the first thing in four attempts to move that number at all, and it moved
it because a case was opened instead of the comment being reasoned from — which
is what the last entry told this one to do, and it was right.

Recorded at the drop, replacing the drift explanation with the stale one and the
numbers behind it. The six that remain are the next thing to open, one at a time.

Suite green: 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## STATE — working tree as of this entry

**The working tree currently holds an unfinished experiment and the suite is
red.** Recording it here because none of this run's work is committed, so the
only other copy of the shipped state is in a session-temporary scratchpad.

### Test status right now

    brep_boolean   35 passed, 5 FAILED
    step           11 passed, 1 FAILED
    brep 21, brep_csg 13, nurbs 17, lib 639 — all passing

Failing: `a_result_can_be_cut_again`, `a_curved_rim_survives_refinement`,
`bores_can_be_drilled_one_after_another`,
`every_step_of_a_ring_is_a_step_of_an_edge`,
`the_same_cut_asked_for_five_ways_is_either_right_or_declined`, and
`a_part_built_by_several_cuts_round_trips` (STEP).

### The three experimental edits, and how to undo each by hand

1. **`src/brep/body.rs` — `fn restate_rings`, called at the end of
   `refine_edges`.** New method, ~70 lines, writes every ring back in the
   vertices of the edges it runs along, matching steps by endpoint. *Undo:*
   delete the method and the `self.restate_rings();` call. The `fn near` helper
   above `signed_area` goes with it.

2. **`src/brep/body.rs` — the pin removed.** `refine_edges` now has
   `let pinned: Vec<bool> = vec![false; self.edges.len()];`. *Undo:* restore

       let pinned: Vec<bool> = (0..self.edges.len())
           .map(|i| {
               self.faces.iter().any(|f| {
                   f.edges.contains(&i) && f.loops.as_ref().is_some_and(|l| !l.is_empty())
               })
           })
           .collect();

3. **`src/brep/boolean.rs:1541` — `split_face`'s early return keeps its rings.**
   Reads `loops,`. *Undo:* restore

       loops: if carried.is_empty() { loops } else { Vec::new() },

   (Note `boolean.rs:2167` also reads `loops,` — that one is original, leave it.)

Also present: `tests/zz_probe.rs`, a scratch probe, delete it.

Undoing 1–3 returns the tree to green: 40 boolean, 21 brep, 13 brep_csg, 17
nurbs, 12 step, 639 lib.

### What this tick established before it was cut short

* **Unpinning fixes the precision cost.** `a_blind_hole` was out by 1.3% — the
  exact figure `split_face`'s comment records for a kept loop — because edges
  under a ring-stating face never refine. With `restate_rings` making a ring
  follow its edges, the pin is unnecessary; removing it fixed that test and took
  the count from 6 failures to 5, and STEP from 2 to 1.
* **`every_step_of_a_ring_is_a_step_of_an_edge` fails for a locatable reason.**
  It reports `f1 l0 step #7 (v91 to v89) has no edge behind it`, and edge 2 *does*
  hold both 91 and 89, one step apart through v90. The ring is simply not
  restated yet: `restate_rings` runs inside `refine_edges`, and that test reads
  the boolean's output directly. **The next step is to restate at assembly** — in
  `from_pieces`, after `Body::from_parts` — so the invariant holds where the
  boolean produces it, not only after a caller refines.

### Everything landed this run and still green under an undo

The crossing settle, the vertexless extent, the seam runs, the fan-chord filter,
`materialise_seam_steps`, and their tests. All uncommitted, like everything else.

## STATE SUPERSEDED — tree is green again

The experiment documented in the previous entry has been undone. The tree is back
to 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib, all passing, and
carries no scratch probe. Those undo instructions are now history; ignore them.

Before undoing it, two more steps were measured, and both worked:

* **Restate at assembly.** `restate_rings` called in `from_pieces` after
  `Body::from_parts`, so the boolean's own output holds the invariant instead of
  holding it only once a caller refines. That is what
  `every_step_of_a_ring_is_a_step_of_an_edge` was reporting — edge 2 held both
  v91 and v89 one step apart through v90, and the ring simply had not been
  restated yet. Fixed it. **Five failures to four.**
* **Reject the long way round.** A closed rim offers two runs between any two of
  its vertices and the shorter-by-count one is not always the right one, so a
  candidate run is now rejected when it is more than 1.5x the straight distance
  between its ends. Correct in principle; it changed nothing measurable here.

So the whole line stands at **nine boolean failures reduced to four**, by four
separate corrections, each of which held:

    restate rings by endpoint      9 -> 6
    unpin, rings now follow edges  6 -> 5   (and fixed `a_blind_hole`'s 1.3%)
    restate at assembly            5 -> 4
    reject the long way round      4 -> 4

The four left, with what they say:

    a_result_can_be_cut_again                  NotWatertight, 100 open edges
    bores_can_be_drilled_one_after_another      NotWatertight, 112 open edges
    the_same_cut_asked_for_five_ways_...        (not yet opened)
    a_curved_rim_survives_refinement            volume 11.08 of an expected 125.66

The last is the one to open next, and it is not a near miss: a quarter cylinder
coming back with a twelfth of its volume is a ring folded inside out, not a seam
a point short. The long-way guard was written for exactly that and did not move
it, so the fold has another cause. Everything else in this run has yielded to
opening one case and reading it, and that is the way in.

Rebuilding the experiment is a few edits from the shipped state, all described in
the entry above this one; the work is in the record rather than the tree because
a red tree is not a place to leave a loop.

## The fold is not the ring's orientation

Went after the one failure that is not a near miss — a quarter cylinder coming
back with 11.08 of an expected 125.66, which is a ring read as the wrong thing
rather than a boundary a point short.

The obvious suspect: `restate_rings` recomputes `area` with `signed_area`, and a
ring's `area` is stored `.abs()` at one site in `split_face` and signed at
another. A recomputed negative reads as a **hole**, and a face's outer ring
filled as a hole would lose almost exactly that much. Restating only adds points
along the path a ring already ran, so its orientation cannot have changed — keep
the sign it came with.

Measured, that is wrong twice over:

* it does not fix the case — `a_curved_rim_survives_refinement` stops returning
  11.08 and starts *declining* `NotWatertight { open_edges: 16 }` instead;
* and it costs more than it saves. The failure count goes 4 to 5 and the failing
  *set* changes: `a_blind_hole` comes back, which unpinning had fixed.

So the `.abs()` convention is not the fold, and preserving the sign is a
regression rather than a correction. Reverted, and the tree is green again — 40
boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

That leaves the line where the last entry left it, at four, with the fold's cause
still open. What is now excluded: the long way round a closed rim (guarded, no
change), and the ring's orientation (this entry). The next attempt should stop
guessing at the mechanism and open the case — dump that quarter cylinder's rings
before and after restating and find the one that differs, which is a probe this
run has used successfully perhaps a dozen times and did not use here.

## A ring of no area is not an outline — nine failures down to one

Opened the tube instead of guessing at it, which is what the last entry said to
do, and the dump answered in one line:

    ZZR f0 cylinder n 16 -> 16 area 0.0000 -> 0.0000  first [142,143,144,145,...]
    ZZR f0 cylinder n 16 -> 16 area 0.0000 -> 0.0000  first [158,159,160,161,...]

**Area zero.** Those two rings are the cylinder's rims — each a line at constant
`v` in parameter space — and a face whose "loops" are its two rims has been
handed two lines and no region. The grid fill draws that wall correctly from its
parameter range; the loop fill cannot draw it at all. Restating was never going
to help, and neither was orientation or path choice: the ring bounds nothing.

Keeping only rings that bound something —

    loops: if loops.iter().all(|l| l.area.abs() > 1e-12) { loops } else { Vec::new() }

— resolves the tube outright (0 open edges) and takes the count from four
failures to **one**, with STEP from two to one.

So the whole line now stands at:

    restate rings by endpoint          9 -> 6
    unpin, rings follow edges          6 -> 5
    restate at assembly                5 -> 4
    reject the long way round          4 -> 4
    keep only rings that bound         4 -> 1     (STEP 2 -> 1)

The two that remain, with what they say:

    a_blind_hole                       volume 24.808 of an expected 25.133 (1.3%)
    no_solid_leaves_through_a_file...  plate and bore Union: left as 209.630,
                                       came back as 210.199, nothing reported

Both are precision rather than structure — the second is the corpus guard doing
exactly its job, catching a silent size change through a round trip.

One more thing tried and rejected: dropping a face's rings when the edges cannot
state them, on the reasoning that such a ring will drift. It goes the wrong way,
**one failure to seven** — many faces hold rings no edge can state, parameter
outlines among them, and they need those rings to fill at all. Keeping an
unstatable ring is better than having none.

Reverted to green: 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.
The experiment is five described edits from here and worth rebuilding, because
one boolean failure and one round-trip failure is the closest this line has come.

## The pin is what costs the blind hole, and lifting it trades two for two

Opened `a_blind_hole` rather than reasoning about it. The failing check is the
*Intersection* — a cylinder r=2 h=2, coming back 24.808 of an expected 25.133,
which is a rim of about twenty points where the tolerance asks for a hundred and
forty. The dump is unambiguous:

    F2 plane  loops [16]  edges [1]
    E0 (plane, cylinder)  n=101      <- refined
    E1 (cylinder, plane)  n=17       <- not

    ZZE edge 0 (cylinder,plane)  17 -> 129
    ZZE edge 1 (cylinder,plane)  17 -> 129     (the *input* bodies)
    ZZE edge 0 (plane,cylinder) 101 -> 101     (the result: edge 1 never appears)

Edge 1 is skipped because it is **pinned** — the drill's end cap states a ring,
so every edge it names is held still, and the cap keeps a 16-point rim while the
wall beside it goes to 129. Also worth recording: the checkpoint this run rebuilt
from turned out to predate the unpin, which is why the failure was still there.

Lifting the pin fixes it, and fixes the STEP round trip too — the corpus guard's
`plate and bore Union` mismatch was the same coarse rim through a file. **STEP
goes to 12 of 12.** But two boolean tests go the other way:

    a_result_can_be_cut_again              a bored ball, 101 open edges
    the_same_cut_asked_for_five_ways_...   3 of 5 tolerances resolve, not 5

Both are spheres, whose outline is seam and poles and which `restate_rings`
cannot state — so their edges refine and their rings cannot follow.

Tried the obvious synthesis: pin only the faces whose rings *cannot* follow,
computed from whether every ring step is an edge step or joins two points of one
edge. No change — those sphere rings pass that test and still are not restated,
so the check is not measuring what restating actually does. That is the thread to
pull next: make the check ask `restate_rings` itself rather than approximate it.

Best configuration reached this run: **two boolean failures, everything else
green**, from nine. Reverted; the tree is green at 40 boolean, 21 brep, 13
brep_csg, 17 nurbs, 12 step, 639 lib.

## The fold is in restate's own unwrapping, and it is mine

Opened the bored ball cut twice, under the unpinned configuration, and the dump
found the bug rather than another hypothesis:

    ZZR f0 sphere   n 7 -> 29   area -0.0851 -> -0.0943
    ZZR f2 cylinder n 7 -> 29   area  0.3496 ->  0.1127     <- a third of it
    ZZR f0 sphere   n 272 -> 272                             <- the outline, unstatable

Both faces receive the *same* restated vertices, so the drill's rim is shared
correctly. But the cylinder's ring area collapses to a third without its shape
changing, which is a ring folded in parameter space.

The cause is in `restate_rings` as this run wrote it: inserted points are
unwrapped against a running anchor, and then the ring's *own* next point is
pushed unchanged — and that point can sit a period away from where the inserted
run ended. The ring folds there.

Carrying one anchor across the whole ring fixes the fold — areas become
consistent (-0.9479 -> -1.0715) and the ball goes from **101 open edges to 29** —
but it rewrites the ring's own parameters, which changes its signed area. The
drill wall's ring goes from +0.3496 to -0.9479, and a ring whose sign flips is
reclassified from outer to hole: one `carried_through`, and eight boolean tests.

So the repair is narrower than either version tried:

* the inserted points must be unwrapped so they lie **between** the step's two
  endpoints, not merely near the first of them;
* and the ring's own points must be left exactly as they are, because their
  parameters carry the orientation the rest of the kernel reads.

That is a precise statement of what to write next, and it came from opening the
case — the third time this run that a dump has replaced a wrong guess with the
actual line of code.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The fold is fixed, and one test stands between here and the whole thing

Wrote the narrow repair the last entry specified, and it is right — but the way
to it was one step further than that entry saw.

Unwrapping the inserted points against a running anchor and **discarding** a run
that does not land on the ring's own next point removes the fold: the drill
wall's rim goes from an area of 0.3496 -> 0.1127 to 0.3496 -> 0.3704, and the
bored ball cut twice from 101 open edges to 6. But discarding leaves the two
faces on one rim holding 29 points and 25.

A run that does not land is not wrong, only wound the other way. **Shift** it by
whole turns until its end meets the ring's next point and both faces get 29:

    bored ball cut twice   101 open edges -> 0

That leaves one boolean test, and the pin is what it turns on:

    full unpin                      sweep 3 of 5, blind hole ok, STEP ok
    pin where ends merely share
      an edge (first occurrence)    sweep ok, blind hole FAILS, STEP FAILS
    ... all occurrences             sweep FAILS, blind hole ok, STEP ok
    pin unless every step is an
      adjacent edge step            sweep FAILS, blind hole ok, STEP ok

The first-occurrence version pins the cap wrongly — a closed edge repeats its
first vertex last, so a ring's *closing* step reads as the whole way round and
trips the length guard. Fixing that unpins the sphere too, and adjacency does not
separate them either: **the sphere's outline is adjacent-edge-backed except at
its poles**. So it is not that restating fails to run on it; restating runs and
produces something wrong at fine tolerance — 403 open edges at 1e-4, 148 at 5e-5,
and nothing at all wrong at 2e-4 or coarser.

That is the whole remaining distance: one case, one test, and a defect that only
appears below 2e-4 on a sphere whose ring restates cleanly at coarser settings.

Best configuration reached: **one boolean failure, everything else green**, from
nine at the start of this line. Reverted; the tree is green at 40 boolean, 21
brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The last case is broad, and three more guesses missed it

Opened the one remaining failure — the sweep's two finest tolerances — and got a
useful shape but not the cause.

    2e-4   F0 sphere [574,142,142]  F1 [320,153,154]  F2 [297]  F3 [298]   0 open
    1e-4   F0 sphere [962,200,200]  F1 [452,302,299]  F2 [504]  F3 [501]   403 open

    ZZOPEN total 403  per-face [(0,179), (1,112), (2,57), (3,55)]
                      both-body 291   len 1.32e-2 .. 1.53e-1

Not a few stray vertices: the 403 are spread over **all four faces**, in segments
the length of an ordinary refined step (a chord at 1e-4 on radius 3 is about
0.05, and these run 0.013 to 0.15). Whole stretches of rim are unmatched, not
corners. The rings differ by three points on each shared rim — 302 against 299,
504 against 501 — where at 2e-4 they differ by one.

Three hypotheses, all measured, none of them it:

* **A vertex dropped when `surface.invert` fails**, which would drop it from one
  face only. Placed such points along the step instead of skipping them. No
  change.
* **The two faces choosing different edges.** Any edge holding both ends of a
  step will do geometrically, and at a fine tolerance several do, so `best` could
  differ per face. Restricted the search to edges the face itself names. No
  change.
* (and the earlier length guard, which was already in.)

So the mechanism is still open, and I spent this tick reasoning at it rather than
comparing the two sides of one rim directly — which is the move that has worked
every time it was used in this run and which I did not use here. **That is the
next step, stated concretely:** at 1e-4, take one open edge, find the two faces
whose rings should both contain it, and print both rings around that vertex. The
counts say three points; the comparison will say which three and where they came
from.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.
The experiment stands at one boolean failure and everything else green.

## The clip stalls, and forcing it past that is not free

Followed the instruction and compared the two sides of one open edge, which
answered in one line — and then the answer turned out to be a trap.

Both faces hold the **same vertices in the same order**:

    V486: f0 l0 n=962 #112 ctx [708, 709, 710, 486, 487, 488, 489]
    V486: f1 l0 n=452 #113 ctx [708, 709, 710, 486, 487, 488, 489]

So the rings agree and the *fills* differ. Three more guesses died on that
(the `sags` test being per-face, the two faces choosing different edges, the
pooled sliver drop — none of them). Instrumenting the clip found it:

    ZZEAR stall n=1057 left=116 clipped=false guard=942

`earclip` cannot find an ear with 116 vertices left, breaks, and returns a
partial fan — a 116-vertex hole, and 403 open edges around it. This is the silent
truncation recorded several entries ago as "latent, not live". Restating makes
rings much larger (452 + 302 + 299 here), and at that size it is live.

Taking the most convex corner instead of giving up closes it, and **the entire
suite goes green with the whole experiment in** — 40 boolean, 21 brep, 13
brep_csg, 17 nurbs, 12 step, 639 lib — with the corpus at 13/45 from 12 and every
remaining decline holding fewer open edges (407 -> 272, 450 -> 210, 26 -> 16).

And it is still not shippable, because forcing is an approximation:

    rod, bored, then bored again down the same axis
        once  75.390647   (expected 75.398224)
        twice 67.029114   watertight, `is_valid_solid`, and wrong

That case declined before. Forcing tiles a degenerate polygon with overlapping
triangles, and the result looks closed while enclosing the wrong volume — the one
outcome this crate forbids. The chaining corpus counts `is_valid_solid`, so it
scored that as a *gain*; the metric cannot see it, which is worth remembering.

Making a forced clip fail its face instead — decline rather than approximate —
restores safety and the double bore declines again. But the same forcing is
load-bearing at 1e-4, so the sweep drops to three of five and one test fails.

Both ends are therefore unshippable, and the shape of the real fix is now clear:
the stall exists because restating multiplies ring sizes, so either the fill must
stop depending on ear clipping at that size, or the rings must not grow that
large. That is the next question, and it is a design one rather than another
one-line experiment.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The polygon the fill is handed crosses itself

Asked whether a better triangulator would help, since this repo already has a
constrained Delaunay one in `exact_csg::cdt` with edge recovery and a 20,000
point cap. Before restructuring the feature graph to reach it — `exact_csg` is
behind `openscad`, brep is behind `nurbs` — measured whether the polygon is
tileable at all:

    ring 1366  crossings  0   coincident-pairs 4
    ring 1057  crossings 10   coincident-pairs 4      <- the one that stalls

**Ten self-crossings.** The polygon `fill_planar` is handed is not simple, and no
triangulator tiles a self-intersecting polygon correctly — a CDT would fail
recovering constraints that cross, exactly as the ear clip fails to find an ear.
So the CDT is not the answer and the feature graph stays as it is.

It also explains the trap from the last entry. Forcing the clip past the stall
does not repair the polygon, it tiles a crossing one with overlapping triangles —
which is why a rod bored twice down the same axis came back watertight,
`is_valid_solid`, and holding 67.03 of the 75.39 it should. The fill was never
going to be right; forcing only hid that it was wrong.

The four coincident pairs are the bridges, and expected. The ten crossings are
not: either `bridge_holes` runs a bridge through geometry its `blocked` test
missed, or two of the restated rings overlap each other in parameter space. The
1366-point ring on the same body has none, so it is not size alone.

That is the next question and it is well posed: for the 1057-point ring, print
the ten crossing pairs and see whether they involve a bridge segment or two ring
segments. Bridges are `bridge_holes`'s problem; two rings crossing is restating's.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The crossings are inside the restated holes, not the bridges

Classified the ten crossings, which is what the last entry asked for:

    ring 1366  rings [962, 200, 200]  crossings []
    ring 1057  rings [452, 302, 299]  crossings [ring1 x ring1: 5, ring2 x ring2: 5]

Not one of them involves a bridge. Every crossing is a ring crossing **itself**,
and the two rings are the restated holes on the bore wall — 302 and 299 points,
the rims where the cross-bore meets it. So `bridge_holes` is exonerated and this
is restating's fault, which is to say mine.

Two candidates ruled out along the way:

* **The whole-run shift.** Unwrapping each inserted point toward its interpolated
  position along the step, so no shift is ever needed, is better code and changes
  the crossings not at all: still five and five.
* **A ring spanning the seam.** Both holes sit in small, consistently unwrapped
  boxes — `u[6.9267, 8.7791]` and `u[3.7855, 5.6387]`, spans of 1.85 against a
  period of 6.28, both with area -2.2306. They do not wrap, and the two are
  identical in size and shape as they should be.

So a small number of inserted runs land **out of order** inside an otherwise
sound ring. Five crossings in 302 points is a handful of steps, not a systematic
error — which fits a path chosen in the wrong direction on the few steps where
the edge offers two.

Next: for that hole, print the five crossing segment pairs with the `uv` around
them. An out-of-order run shows up immediately as indices that go backwards, and
the step that produced it can be read off the same print.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The fold is a spike between two of the ring's own points

Printed the five crossing pairs, and the ring's shape at the fold is plain:

    #273 (8.61834, 5.59871)     #276 (8.61352, 5.59275)
    #274 (8.60797, 5.58601)     #277 (8.60305, 5.58012)
    #275 (8.59740, 5.57347)     #278 (8.59239, 5.56765)

Two nearly parallel descents, offset by about 0.005, interleaved. Tagging the
vertices behind them says which is which:

    #270..281   1195 1194 1193 1192 1191 | 398 439 | 1449 1448 1447 1446 1445

The low indices are the ring's **own** points; the high ones are runs put in by
restating. Both runs descend in `u` — and the ring's own pair between them,
`398 -> 439`, **ascends**. Down, up, down: a spike, and a spike crosses.

Ruled out this tick: that the runs are inserted backwards. Requiring every run to
progress from its step's start toward its end — each point no nearer the start
and no farther from the end than the one before — changes nothing, still five and
five. The runs are ordered; the pair between them is not.

So the spike lives between two consecutive vertices of the ring the *boolean*
produced, and restating does not create it — it lengthens the segments either
side until the spike crosses them. That also explains why 2e-4 is clean and 1e-4
is not: the same spike, with more room to cross.

Next: dump that hole as the boolean hands it over, before restating, and look at
`398 -> 439` there. If the spike is already in it, this was never a restating bug
at all, and the fix belongs where the hole is built.

Reverted; green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Shipped rings cross themselves, and it is two vertices swapped at the wrap

Asked whether the boolean's own rings self-intersect, without any of the
experiment. They do, in results that pass today:

    ball-bore-cross@1e-3  f1 l1 n=43  x1     f2 l0 n=109 x1
    rod-cross@1e-3        f0 l1 n=41  x1     f3 l0 n=84  x2
    rod-bore-cross@1e-3   f0 l1 n=41  x1     f4 l0 n=86  x1
    cube-cross-bore@1e-3  f6 l0 n=109 x1

So this is not restating's doing and never was. The fills survive it because ear
clipping copes at these sizes; at the sizes restating produces, it does not — and
that is the whole of the 403 open edges, the stall, and the forced clip that
returned a wrong solid. One latent defect, surfacing three ways.

The smallest instance is 41 points and reads completely. Its crossing is the
closing segment against the first, and the `u` values say why:

    ring order   v2 1.974  v1 1.954  v0 1.924  v39 1.930  v38 1.892
    sorted       v2 1.974  v1 1.954  v39 1.930  v0 1.924  v38 1.892

**v0 and v39 are transposed** — an edge's first and last vertices, swapped where
the ring wraps. The ring laps itself by a fraction of one segment, which is
exactly enough to cross.

That is a small, well-located defect with a large blast radius, and it is the
thing to fix next: find where a ring takes a closed edge's vertices and puts the
wrap pair the wrong way round. It is worth a test of its own regardless of the
restating work — no trim loop should cross itself, and today several do.

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Plan: two decompositions

Ten ticks of this line have been whack-a-mole because there is one assertion at
the end (`is_closed`) and everything before it is trusted. Two decompositions fix
that, and they answer different halves.

### The geometric one — recompute the region instead of trusting rings

`rings -> bridge_holes -> earclip` requires rings that are simple, oriented,
ordered and non-overlapping. Every defect found this week violates one of those:
the overshooting closed curve (not simple), the transposed wrap pair (not
ordered), the zero-area rims (not regions at all), the pole steps (degenerate),
and the bridges (weakly simple by construction).

    regionise(segments, pts) -> Vec<Cell>    // arrangement: crossings become vertices
    classify(cells, rings)   -> Vec<Cell>    // winding number, not ring orientation
    monotone(cell)           -> Vec<Polygon> // y-monotone pieces
    tile(polygon)            -> Vec<Tri>     // linear and *total*

The point is that the first stage does not care whether its input crosses itself,
and the last is total — no ear test, no stall, no forced corner, no silent
partial fan. Holes need no bridging; they are more segments, and the cells inside
them classify out. `planar::subdivide` already does a version of stages 1-2 for
the boolean. The fill is the consumer that never got it.

### The invariant one — a ladder, each rung a property test

    A  every ring is simple                          the overshoot, the wrap swap
    B  every ring step is an edge step               unbacked seams  (have it)
    C  two faces on an edge use the same vertices    stale 16-vs-78 sampling
    D  a face's triangles bound exactly its rings    the stall, the forced clip
    E  the body is closed and the right size         everything else  (have it)

A and D are the two that would have caught this week's defects on the first run
rather than the tenth. D in particular turns "403 open edges somewhere" into
"face 1 did not tile its ring".

**Order: A, then D, then the arrangement fill.** A and D are cheap, need no
design decisions, and fail on the shipped tree today — A does, on seven rings.
They also give the arrangement work a spec and a regression net, so it can land
in one piece instead of being probed into existence.

## Invariant A measured, and the overshoot is not its cause

Wrote A as a measurement before writing it as a test, which is the order the plan
above asks for, and it pays immediately:

    A: 2 crossing rings of 38, over 14 results
       cube-cross-bore f6 l1 n=42 crossings=2
       cube-cross-bore f6 l2 n=42 crossings=3

Two rings, both on one face of one result. Earlier scans looked worse only
because they counted the same body at two tolerances.

Then the overshoot repair — drop a marched closure that has already carried past
its start, which is exactly the fault read off the `rod - cross` rim
(`v38 -> v39 -> v0`, forward then backward). Measured:

* **A is unchanged.** Still two crossing rings, the same two. Those rims are not
  marched, so the overshoot cannot be what crosses them.
* And it **costs three tests** —
  `a_crossing_sits_on_both_of_the_surfaces_whose_rim_it_cuts`,
  `every_step_of_a_ring_is_a_step_of_an_edge`, and
  `the_same_cut_asked_for_five_ways_is_either_right_or_declined`.

So the overshoot is real (the coordinates show it plainly) but it is not the
cause of A's failures, and trimming the tail moves vertices that two faces had
agreed on. Removed.

This is the plan working as intended: one measurement retired a repair that four
ticks of reasoning had been converging on, before it was built into anything. The
next step is unchanged — A over the *specific* two rings on `cube-cross-bore`,
which are 42 points each and will read completely, as the 41-point one did.

## The overshoot is in the ring, not in either sampler

Read the crossing rings on `cube - cross - bore`. Four of them, each crossing
once:

    f6 l0 n=109   f7 l0 n=109   f8 l1 n=43   f8 l2 n=43

Worth flagging honestly: the previous entry's scan reported *two* crossing rings
at n=42 on `f6`, and this one finds four at 109 and 43 on `f6`/`f7`/`f8` for what
should be the same body. Two probes of mine disagree about a result they both
built the same way, and I have not reconciled them. Until that is understood,
neither count should be quoted as the number.

The 43-point ring reads completely, and its crossing is the same shape as the rod
rim from two entries ago:

    #41 v182 (1.91577, 5.27499)
    #42 v183 (1.79293, 5.23093)     <- past the start
    #0  v142 (1.81757, 5.23821)     <- closure comes back

The last point overshoots the first, and the closure doubles back over the first
segment. But neither sampler puts it there:

* **`march`** — the closure trim measured last entry changes A not at all, so
  these rings are not closed by that path.
* **`sample_closed_curve`** — the analytic branch steps `TAU * i / n` for `i` in
  `0..n`, which cannot overshoot, and would give 71 points at this radius and
  tolerance rather than 43. The sampled branch only inverts points it is handed.

The ring's vertices are `v142..v183` — one contiguous block of 42 minted
vertices, closed with a 43rd. So the extra point is added where the *ring* is
assembled, not where the curve is sampled, and that is the next place to look:
whatever appends a ring's final point is appending one that has already gone
round.

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## The two probes are reconciled, and A's real baseline is much worse

Ran both scan shapes in one binary against one tree. They agree exactly — four
crossing rings on `cube - cross - bore`, same faces, same counts. So they never
disagreed about the same tree.

What differed was the *tree*. The earlier `n=42` scan ran while the overshoot
patch was still in the working copy — the same patch that had `brep_boolean` red
that tick before I noticed it. So:

    without the patch   f6 l0 n=109 x1   f7 l0 n=109 x1   f8 l1/l2 n=43 x1
    with it             f6 l1 n=42 x2    f6 l2 n=42 x3

The patch does not reduce crossings; it shortens the rings by the trimmed tail
and leaves *more* crossings on the ones that remain. Another reason it is right
to have removed it, and a reminder that a measurement is only as good as the
knowledge of what was compiled.

With that settled, A's baseline on the shipped tree is not two rings. It is:

    A-BASELINE 17 crossing rings of 49, over 16 results

Nearly half of every trim loop the boolean produces crosses itself — on
`ball-bore-cross`, `rod-cross`, `rod-bore-cross`, `cube-cross-bore`, and more,
all of them results that pass today. The fills tolerate it at these sizes and
stop tolerating it when restating makes the rings bigger, which is the whole
story of the last ten entries in one number.

That makes A the right next piece of work, and it is no longer a tidy-up: it is
the single defect behind the stall, the forced clip, and the wrong-sized solid.

## Where the overshoot is made, and why neither repair at the curve works

Found the line. In `trace`:

    p = next;
    forward.push(p);                                    // kept first
    if forward.len() > 3 && dist(p, seed) <= step * 0.75 {
        forward.push(seed);                             // then closed

The sample is kept *before* the closure is tested, so one that has already
carried past the seed stays in the curve. `march` then pops only the trailing
seed, and the ring ends beyond where it began — its closing segment running back
over the first. That is the overshoot, exactly, and it is one statement.

Both repairs at the curve are worse than the defect:

* **Trim the tail after closing** (drop points whose closing runs against the
  direction of travel). Costs three tests, and measured against A it shortens
  rings without reducing crossings — 4 rings at one crossing each became 2 rings
  at two and three.
* **Decide closure on the candidate, before keeping it.** Never creates the
  point at all, which is the cleaner statement — and costs **seven** tests. The
  closing chord grows to as much as 1.75 steps, and the vertex set changes under
  faces that had agreed on it.

The common cause of both failures is that a traced curve's points are not private
to it: they become edge vertices and ring vertices that two faces share, so
removing one is a change to a boundary, not to a sampling. That is the constraint
any fix has to respect, and it rules out the whole family of "fix the curve"
repairs.

What is left is to handle it where the ring is *assembled* — both faces build
their rings from the same curve, so both would drop the same point and stay in
agreement. That is the next thing to try, and it is the first repair in this line
that would not move a shared vertex.

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## A ring does not start where its curve closed

Tried the repair the last entry proposed — drop a ring's last point when its
closing segment runs back against the one before it, at ring assembly where both
faces would drop the same point. **Seven tests.**

The reason is worth having: the ring's wrap is not the curve's closure. A ring is
walked out of the arrangement and starts wherever that walk began, so index
`n - 1 -> 0` is an arbitrary place in it. Testing only there drops a point that
was not the overshoot at all, on rings that never had one.

That is the third repair for this defect and the third failure, and all three
failed for the same underlying reason in different clothes:

    trim the curve after closing      moves a vertex two faces share
    close before keeping the sample   moves the vertex set, lengthens the chord
    drop the ring's last point        the ring's last point is not the overshoot

The overshoot is a *local* fold — two adjacent segments that run back over each
other — and it can sit anywhere in a ring. So the test has to be positional, not
terminal: scan every `i`, find where segment `i-1` and segment `i` overlap rather
than merely turn obtusely, and repair there. Obtuse alone is not enough; a
polyline around a tight arc turns obtusely all the time without crossing.

That is the next attempt, and it is narrower than the three before it: the
predicate is "these two segments overlap", not "this turn looks wrong".

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Four repairs, one conclusion: the fill has to stop needing simple rings

The positional repair, done properly. Drop the closing point only where a
crossing actually involves one of the two segments it touches, and only while
dropping keeps helping:

    predicate = closing segment only      suite green, A unchanged, corpus 12 -> 11
    predicate = both touched segments     A -> 0 crossings, corpus 12 -> 7, 4 tests

The second reaches zero the wrong way. A falls to *zero of 24 rings over 12
results* — from 49 rings over 16 — because the results that had crossing rings
stopped resolving at all. Counting only what survives is not a measurement, and
this is the second time in this line that a metric has scored destruction as
progress. Worth naming as a hazard of A: **it must be read beside the number of
rings and results it ran over**, or a repair that deletes geometry looks perfect.

So four repairs for this defect, four failures, and they exhaust the space:

    trim the curve after closing        moves a vertex two faces share
    close before keeping the sample     moves the vertex set, lengthens the chord
    drop the ring's last point          the ring's wrap is not the overshoot
    drop it only where it crosses       deletes geometry the result needed

The tension underneath is structural, not a bug in any one attempt. The overshoot
is baked into a curve whose points *are* the shared boundary; every repair either
changes that boundary — and the two faces stop agreeing — or leaves the crossing
in place. There is no third option at the ring.

Which settles the plan's open question. The geometric decomposition is not the
nice-to-have alternative to fixing the rings; **it is the only route left**. An
arrangement-based fill does not care whether its input crosses itself: crossings
become vertices, the region is chosen by winding, and the triangulation is total.
That is the next build, and this entry is the argument for it.

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Why no repair at the ring can work: the crossing belongs to the projection

Fifth attempt, and the most careful one: cut a self-crossing ring at its crossing
and keep the larger of the two pieces. Bounded where popping was not — the lap is
its own loop of two or three points, and keeping the larger half discards exactly
that and nothing else.

Three tests, with small numbers: `a bored ball ... NotWatertight { open_edges: 5 }`,
a rod-on-ball union with defects, and the tolerance sweep. Small is the tell.

And it finishes the argument. A ring is a curve **projected into one face's
parameter space**, and whether that projection laps itself is a property of the
projection, not of the curve. The sphere's ring can cross where the cylinder's
ring on the same curve does not — so a repair applied per face edits one side of
a shared boundary and not the other, and the two stop agreeing. Five attempts,
five different mechanisms, one cause:

    trim the curve after closing        moves a vertex two faces share
    close before keeping the sample     moves the vertex set, lengthens the chord
    drop the ring's last point          the ring's wrap is not the overshoot
    drop it only where it crosses       deletes geometry the result needed
    split at the crossing, keep larger  edits one projection, not the other

There is no repair at the ring, and now there is a reason rather than a tally.
The fill has to accept a ring that crosses itself — which is what an arrangement
does by construction, since a crossing simply becomes a vertex and the region is
chosen by winding rather than by trusting the loop.

So the next build is `regionise / classify / monotone / tile`, and every
alternative to it has now been measured and closed.

Green at 40 boolean, 21 brep, 13 brep_csg, 17 nurbs, 12 step, 639 lib.

## Landed: a traced curve can be read as a curve

Changed direction on the strength of a question worth asking — whether the
algorithm is missing a better curve representation. It is, and the crate already
has most of the pieces:

    src/nurbs/curve.rs      NurbsCurve, with point(u), derivatives(u, k), domain()
    src/brep/curve3d.rs     Curve3d: Line, Circle, Ellipse, Point, Sampled

There is no `Curve3d` variant that carries a spline, so every intersection
without a conic form is frozen into a polyline the moment it is traced. That
polyline is the common cause under most of what this run has fought:

* the **closure overshoot** exists only because closure is a distance test on
  samples — a periodic curve closes topologically and has no last sample to be
  past anything;
* the **16-against-78 divergence** happens because the polyline *is* the
  boundary, so whoever samples first wins and everyone else must chase;
* the **pin** and **`restate_rings`** are both workarounds for a ring that cannot
  follow an edge, because neither is re-evaluable;
* the **chord-sag hacks** in `clip_to_face` and `refine_on_surface` approximate a
  curvature bound a spline gives directly.

It also reframes the four sampling changes that never shipped. Every one was an
attempt to change a *fixed* sampling; a curve is what makes sampling not fixed.

Adding the variant is 42 match sites across `boolean.rs` and `intersect.rs` with
no catch-alls, so it is not a one-tick change. The fitter is needed first either
way, and it is landed: **`nurbs::construct::interpolate(points, closed)`**, a
cubic B-spline through *every* input point, Catmull-Rom tangents written as
Bezier spans, so there is no system to solve and no end conditions to pick.

Interpolating rather than approximating is the point: those points are already
vertices two faces share, and a fit that moved them would move a boundary. Pinned
by `a_traced_curve_can_be_read_as_a_curve` — every input point on the curve to
1e-9, the ends together to 1e-12 (exactly, not by a distance test), and the curve
within 1e-3 of the circle it interpolates where the 24-gon it was built from sags
0.0171.

Next: the `Curve3d::Spline` variant and the 42 sites, with `Sampled` kept as the
fallback where a fit cannot hold tolerance. Then re-run A — 17 crossing rings of
49 — which is the measurement that says whether this was the right diagnosis.

Green: 40 boolean, 21 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Landed: `Curve3d::Spline`, and the blast radius was a tenth of the estimate

Added the variant. It carries the interpolating cubic plus `at`, the parameter of
each traced point, so `t` still indexes the samples exactly as it does for
`Sampled` — the convention everything downstream is written against.

The 42 match sites turned out to be **three** compile errors: the import, `kind`,
and `project`. Nearly all of those sites construct or match one variant rather
than exhausting the enum. That estimate was made by counting `Curve3d::` and it
was wrong by an order of magnitude, which is worth remembering the next time a
refactor is sized by grep count rather than by compiling it.

Nothing produces a `Spline` yet, so the tree is unchanged in behaviour and green.
What is there:

    Curve3d::spline_through(points, closed)   the polyline read as a curve
    point(t)                                  same parameter, same points
    project(p)                                scan then walk, like the ellipse arm
    translated(t)                             shifts control points, keeps knots

Pinned by `a_traced_polyline_and_the_curve_through_it_agree_at_every_sample`,
which is the claim that matters: at every sample the polyline and the curve give
the same point to 1e-9, and between the samples

    the chord of a 32-gon at r=3    sags   1.44e-2
    the curve through those points  is out 1.04e-4

a hundred and forty times less. It is an interpolation and not the circle, so it
is not exact — my first assertion asked for 1e-4 and had to be relaxed to the
measured 1.04e-4. But it is wrong by less than the tolerances this kernel holds,
where the chord is wrong by more, and that is the whole difference.

Next: have `march` fit one and return it, keeping `Sampled` where a fit cannot
hold tolerance — then re-run A. Seventeen crossing rings of forty-nine is the
number that says whether this diagnosis was right.

Green: 40 boolean, 22 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## What the spline is actually for: not accuracy, re-evaluability

Measured the fit against the surfaces before wiring it in, which was the right
order — the answer changes what the variant is for.

    rod x cross     n=40   spline_off 1.33e-3   chord_off 2.67e-3   tol 1e-3
    ball x cross    n=35   spline_off 1.58e-3   chord_off 3.04e-3
    torus x cyl     n=233  spline_off 1.87e-4   chord_off 3.78e-4

The curve through the traced points is consistently **half** the chord's
deviation, and at a tolerance of 1e-3 *neither holds it*. A marched curve's raw
sampling is out by two to three times the tolerance it was asked for, and
interpolating those samples cannot fix that — an interpolation is only as true as
the points it passes through.

So the spline does not buy accuracy, and it was wrong to expect it to. What
already buys accuracy is `refine_edges`, whose `subdivide` settles each new
midpoint onto *both* surfaces — every inserted point is on the intersection to
1e-6, whatever the curve between them looks like.

Which sharpens what `Curve3d::Spline` is for, and it is still worth having:

* **a parameterisation to generate from.** A ring cannot refine because a
  polyline has nothing to ask. A curve does, and each generated point can then be
  settled onto both surfaces exactly as an edge's midpoints are — `Edge` already
  carries `surfaces: (usize, usize)`, so the machinery is in reach.
* **exact closure**, which no distance test on samples can promise, and which is
  the direct cause of 17 crossing rings in 49.

So the wiring is not "march returns a spline instead of points". It is: keep the
points, carry a curve alongside them, and let whatever needs a new point ask the
curve where and the surfaces what. That is a smaller and better-founded change
than replacing the representation, and it leaves `Sampled` alone.

Green: 40 boolean, 22 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## No fitted curve can be exact here, and that settles the representation

Two results this tick, one of them a mistake worth recording.

**The even re-sampling was a no-op.** Re-saying a closed curve from its own
spline at `t = n * i / n` is re-saying it at `t = i`, which is where the points
already were. A unchanged at 17 of 49, corpus unchanged at 12/45, suite green —
green because nothing happened. The idea was never tested; the code was.

**And the idea would not have helped anyway**, which the closed-form table
explains. `ssi` already covers plane against everything, sphere-sphere,
sphere-cylinder, cylinder-cylinder, cylinder-cone, cone-cone and coaxial tori.
What falls through to marching is the *general* quadric pair — two cylinders at
unequal radii, which is `rod x cross`, the case behind most of A's crossing
rings.

The intersection of two quadrics is a **quartic space curve**, and a quartic of
that kind is not rational except in degenerate configurations — tangency, or a
shared plane or line. A NURBS curve is rational by construction. **So no NURBS,
Bezier or arc-spline representation can be exact for the case that actually
needs one**, and the measurement two entries ago said the same thing
empirically: the fit halves the chord's error and still misses tolerance by 1.3x.

That closes the question of whether a better curve *type* helps. It does not.
What helps is not fitting at all:

* **the curve stays implicit** — it is the pair of surfaces, and any point on it
  comes from `settle`, which is exact to 1e-6 wherever it is asked;
* **the spline is only a parameterisation** — somewhere to ask *where* next,
  monotone and cheap, with no claim to be the curve;
* **closed forms stay preferred** where they exist, and the table is nearly full,
  so there is little headroom left there.

`Curve3d::Spline` is right to exist under that reading, and wrong to be filled by
`march` as a replacement for the points. It should carry the parameterisation
beside them.

Green: 40 boolean, 22 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The diagnosis is confirmed: 17 crossing rings to 1

Ran the experiment the no-op skipped. Re-say a closed traced curve **evenly in
the curve's own parameter** — not at the points it already has — and settle every
sample back onto both surfaces. The last point then lands strictly before the
close, which is exactly where the lap came from.

    A-BASELINE  17 crossing rings of 49, over 16 results
    A-NOW        1 crossing ring  of 47, over 16 results

Sixteen results either way and forty-seven rings against forty-nine, so this is
not the earlier trap where a "fix" scored well by destroying the geometry it was
measured on. The rings are still there; they have stopped crossing. **The
diagnosis of the last several entries is correct**: the crossings come from the
closure keeping a sample that has already gone round, and re-saying the curve
removes them.

And it costs coverage: the chaining corpus falls 12/45 to 10/45, with three
boolean tests and the STEP corpus guard. The telling one is

    every_step_of_a_ring_is_a_step_of_an_edge
        a bore through a solid resolves: NotWatertight { open_edges: 74 }

`rod - cross` — the very case whose ring was crossing — stops resolving once the
crossing is gone. The fill was not tolerating that ring in spite of the crossing;
it was tolerating *that sampling*, and any other sampling of the same curve fails
it differently.

So this is the sixth repair and the first that works on its own terms, and it
still cannot ship, for the reason every one before it could not: the fills are
tuned to the sampling they were handed. That is now a claim with a controlled
experiment behind it rather than an inference — same curve, same count, same
results, only the point placement changed, and four things broke.

Which is the argument for the arrangement fill in its strongest form yet. Until
the fill stops depending on where the points happen to fall, no improvement to
where they fall can be kept.

Green: 40 boolean, 22 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Landed: `planar::arrange`, the first stage of a fill that needs no simple ring

Built the arrangement's splitting stage. `arrange(rings, tolerance)` takes the
rings as they are — crossing or not — and returns a planar graph: every segment
cut where it meets another, points welded at tolerance so a crossing computed
from both directions lands once.

That is the precondition the whole line has been failing to satisfy by other
means. A trim loop that laps itself is not a polygon any triangulation can tile,
and six repairs have now failed to stop the lapping, the last of them working
perfectly and costing four tests anyway. An arrangement does not ask: the
crossing becomes a vertex and the impossible polygon becomes an ordinary graph.

Pinned by `a_ring_that_crosses_itself_becomes_a_graph_that_does_not`, on a square
whose last corner laps the first exactly as a traced closure does — the crossing
becomes a point, the two segments it splits become four, no pair of edges crosses
afterwards, and a ring that was already simple comes back untouched, four points
and four edges.

Two mistakes worth recording, both mine and both in the test rather than the
code. The first lapped ring did not actually cross — its overshoot went *outside*
the boundary instead of across it, which is a different shape from the defect and
would have passed for the wrong reason. And the no-crossing check tripped on
edges that merely share an endpoint, where the orientation is zero and a zero
reads as a sign change. A test that fails for the wrong reason is worth as little
as one that passes for the wrong reason.

Nothing consumes `arrange` yet, so the tree is unchanged and green. Next is the
half that turns a graph into regions: walk the faces of the arrangement, choose
the ones inside by winding rather than by trusting a ring's orientation, and hand
each to the existing fill — which will then be receiving a simple polygon for the
first time.

Green: 40 boolean, 23 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Landed: `planar::regions_of` — the region a lapped ring meant

The second stage. `regions_of(rings, tolerance)` arranges the rings, walks the
graph's faces by tightest turn, and keeps the cycles the rings **wind around**.
Winding is the point: a loop that laps itself has no consistent orientation to
trust, which is precisely why six repairs that trusted one have failed.

Pinned by `the_region_a_lapped_ring_meant_survives_its_lap`:

* a unit square whose last corner laps the first comes back as the region it
  winds around, and
* a bowtie comes back as exactly two triangles, neither of them the crossing.

The area is the interesting part, and my first assertion was wrong. I expected 1
and got 0.9774, which is not an error — **the lap shaves a corner**, so the
region the loop encloses really is smaller than the square. The shoelace of that
ring is 0.975; it differs from the arrangement's 0.9774 because the shoelace
counts the lapped triangle *negatively* while the arrangement counts it once, and
counting it once is the more honest reading of "the part this loop encloses".
Asserted at the measured value with that reasoning written down, rather than at
the round number I assumed.

Both halves of the geometric decomposition now exist and are tested:

    arrange(rings, tol)      crossings become vertices
    regions_of(rings, tol)   the enclosed cycles, by winding

Neither is wired in, so the tree is unchanged and green. What remains is the
integration, and it is the part that needs care rather than code: a region's
boundary must still be made of the *same vertices the face across it has*, or the
fill trades a self-crossing ring for an open seam — which is the trap that caught
every ring-level repair. The crossing point the arrangement introduces is the
thing to watch.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## `regions_of` passes its own tests and fails on real rings

Before wiring the arrangement in, asked the question integration depends on: is
`regions_of` the identity on a ring that is already simple? If it is not, wiring
it in perturbs the 32 rings in 49 that were never the problem.

    IDENTITY same 7  differ 4  multi 17  empty 0

Seven of twenty-eight. Seventeen come back as *several* regions. So no — and the
two tests it passes are not wrong, they are just narrow: a lapped square and a
bowtie are convex-ish, small, and wound the way the implementation assumes.

Two attempts at it, neither converging:

* **Keep by winding rather than by area sign.** The sign filter drops a hole's
  interior — a hole is wound the other way — so it was keeping the *outside* of
  every hole. Removing it made every ring multi: 28 of 28.
* **Probe just inside an edge rather than at the centroid of the corners.** A
  cycle need not contain the average of its own vertices and an outer face never
  does. No change: still 28 multi, and the lapped square came back as three
  regions totalling zero area.

Restored to the state whose tests pass, and the honest position is that
`regions_of` is not yet correct — it is correct on the two shapes it was written
against, which is exactly the failure mode this run has caught twice before in
its own probes. A face walk that returns the outer face as a region, and a
winding test whose probe is not reliably inside the cycle it is testing, are both
still open.

What the identity measurement is worth: it is the right gate, and it should stay
the gate. Anything that goes near the fill must leave a simple ring alone, and
this says plainly that the current implementation does not.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The identity gate was the wrong gate, and the arrangement is sound

Last entry concluded `regions_of` was broken because it returned several regions
for 17 of 28 real rings. That conclusion was wrong, and the reason is worth
having: **a real trim loop contains bridges**. A ring that runs out to a hole and
back visits a vertex twice, which is a degree-four vertex in the arrangement, and
several faces whose *union* is the region is the correct answer there. Identity
was never the right thing to ask.

The right gate is area, and it passes:

    AREA-GATE kept 16  lost 12  worst-rel 1.534e-5
        n=43  want 2.223393  got 2.223410
        n=41  want 1.022330  got 1.022337

Every ring's area is preserved to about one part in 1e5, and the twelve that
differ are exactly the crossing ones, where the arrangement counts the lap once
instead of negatively — the same reading documented for the lapped square. So the
arrangement does what it was built to do, and last entry's verdict on it should
be struck.

What integration actually costs is now measured too:

    INVENTED  n=109 new 2   n=43 new 2   n=41 new 2   n=84 new 6

Two to six new vertices per crossing ring, none on a simple one. Those are the
crossing points, and they land *on* the boundary — so the face across it would
not have them, and the fill would trade a self-crossing ring for an open seam.
That is the wall every ring-level repair has hit.

But this time there is a way through, and it is the opposite of what has been
tried. A crossing point is a point **of the curve**: it lies on the intersection,
in three dimensions, and both faces can be given it. Every previous repair tried
to *remove* a vertex, and removal is what breaks agreement — the two faces
disagree about what is gone. **Adding** does not: insert the crossing into the
shared curve, and both rings acquire it, and both stay simple.

That is the next thing to build, and it is the first plan in this line that does
not fight the shared-boundary constraint.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Adding the crossing halves A and costs exactly one case

Built the repair the last entry argued for: where the curve's projection onto
either surface crosses itself, put a vertex there — found on the chord, settled
onto both surfaces like every other sample, **added and never removed**, so both
faces get it and both loops gain what makes them simple.

    A          17 crossing rings of 49 over 16 results  ->  8 of 41 over 15
    suite      green, all six binaries
    corpus     12/45 -> 11/45

and the corpus diff is one line:

    DECLINE rod: bore then cross Difference -> NotWatertight { open_edges: 8 }

That is the first repair in this line to move A substantially **without the suite
objecting** — six previous attempts each cost between three and nine tests. The
whole cost is one chain case and eight edges.

Tried narrowing it to *local* laps only, on the reasoning that a closure that
went round too far crosses the segments beside it while a distant crossing is the
curve genuinely folding. Wrong: A goes 8 to 11 and the corpus 11 to 10, with two
declines instead of one. The distant crossings are worth fixing too.

Reverted, because one lost case is still a regression. But the shape of the
answer is now clear and close: adding is the operation that two faces can agree
on, it works, and what remains is eight open edges on a single case rather than a
structural objection.

Also worth recording: the build died mid-tick with missing dependency artifacts
and `target/debug/deps` gone from under it — not the change, and a full rebuild
took fourteen seconds. Worth knowing that a compile error naming a crate you did
not touch is worth one look at the tree before it is worth any thought.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The one case it costs is an asymmetry of a single vertex

Opened `rod: bore then cross Difference` under the crossing insertion. Six faces,
eight open edges, nothing carried — and the face list says what is wrong without
any further probing:

    F3 cylinder loops [146, 49, 48]
    F4 cylinder loops [97]
    F5 cylinder loops [96]

That solid is symmetric about the xz-plane. `F4` and `F5` are the two halves of
the cross-bore's wall and must match; `F3`'s two holes are the rims where that
bore meets the first one, and must match too. Both pairs differ by **exactly
one** — 49 against 48, 97 against 96.

So the insertion is not wrong in kind, it is wrong in *symmetry*: a crossing
found on one side is not found on its mirror. The curves either side are traced
from different seeds and are not exact mirrors of one another, so a crossing that
is just over the threshold in one projection is just under it in the other, and
one face gains a vertex its opposite number does not. Eight open edges is what a
single unmatched vertex costs.

That is a much better position than "it costs a case". The repair is right — it
halves A with the whole suite green — and what is left is a tie-break: a crossing
test that answers the same way on two curves that ought to be mirror images.
Snapping the found parameter, or looking for crossings on both projections and
taking their union rather than the first hit, are the two obvious ways in.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Three ways to make the insertion symmetric, and what each is worth

The insertion halves A with the whole suite green and costs one chaining case,
and the cause is an asymmetry of a single vertex on a mirrored solid. Three
attempts at that asymmetry:

    plain (first crossing found)   A 17 -> 8   suite green   corpus 12 -> 11
    a band around the test         A      8    suite green   corpus      11
    insert at the segment midpoint A      6    one test      corpus      11

The **band** — accept a crossing whose parameters land within 5% outside the
segment, so a near-miss counts — is exactly neutral. Every number identical. So
the two mirrored curves are not landing either side of a knife edge; their
projections genuinely differ in whether they cross at all, which is a topological
difference and not a numerical one.

The **midpoint** — insert at `t = 0.5` rather than at the computed crossing, so
both halves agree on where the vertex goes — improves A again, 8 to 6, and costs
a boolean test. It is treating the symptom: a vertex in the same *relative* place
is still not a vertex in the same place.

So the insertion stands at: right in kind, worth half of A, blocked by one case
whose two halves are traced from different seeds and are not mirrors of each
other. That last clause is the thing to attack next, and it is upstream of all
this — `seeds` walks a 13-cubed grid and `trace` walks from wherever that lands,
so a solid symmetric about a plane has no reason to produce symmetric curves.
Making the trace itself symmetric would fix this case and any other like it,
rather than patching the crossing test that reports it.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The two halves are the same curve at a different phase

Measured the claim rather than assuming it. For `rod - cross`, `march` returns
two curves that should be mirror images across `x = 0`:

    n=37 closed  x-range -2.0000..-1.8331
    n=37 closed  x-range  1.8331.. 2.0000
    MIRROR counts 37 vs 37   worst-partner-distance 3.649e-2

Same shape, same count, and **not** mirrors: every point of one is up to 3.6e-2
from the mirror of its nearest partner, against a sample spacing of about 0.136.
They are the same curve sampled at a **different phase**, because `trace` starts
wherever `seeds` happened to land and walks from there.

That is the whole asymmetry, and it explains why the three patches failed.
Whether a closure laps at all depends on where the samples fall relative to it —
so one half laps and the other does not, and no test applied to the crossing can
make two curves agree when the disagreement is in their sampling.

The obvious fix is to sample at a canonical phase, and it does not generalise. A
canonical anchor has to be *equivariant*: the anchor of a mirrored curve must be
the mirror of the anchor. Lexicographic extremes are not — mirroring turns a
maximum into a minimum. Farthest-from-centroid is equivariant but degenerate here,
because these rims are near-circles and every point is nearly equidistant, so the
choice is decided by noise.

So there is no general way to make the sampling agree, which is the same wall in
a new place: **the sampling cannot be made canonical, so the fill must stop
depending on it.** Every route out of this line now points at the arrangement
fill, and this is the fourth independent argument for it — after the five ring
repairs, the resample that worked and cost four tests, and the insertion that
halves A and costs one case.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Not symmetry after all: the inserted points do not reach one face's ring

Attributed the eight edges instead of inferring from face counts, and the answer
is different from the last entry's:

    face 3   146 -> 147 -> 148 -> 149 -> 150 -> 151 -> 155
    face 5   146 -> 65202,  155 -> 65202

Face 3 walks six consecutive segments through vertices 147..151. Face 5 jumps
straight from 146 to 155 through **65202** — an index far past the body's vertex
list, which is a *fresh* point `fill_planar` mints when a ring entry does not
resolve to a body vertex at all.

So the five inserted points reach one face's ring and not the other's, and the
one that misses them falls through `unwrap_or(usize::MAX)` and invents a vertex
of its own. That is the eight edges.

Which means the previous entry's diagnosis was wrong, or at least not the
operative cause. The two curves *are* sampled at different phases — that
measurement stands, 3.6e-2 against a spacing of 0.136 — but what breaks this case
is not the asymmetry. It is that a ring resolves its points to curve vertices by
**matching positions to 1e-12**, and a point inserted into the curve afterwards
has no match in the order array that mapping was built from.

That is a much better place to be: a mapping that goes stale when the curve gains
a point is a defect with an obvious shape, where "make the trace symmetric" had
no general answer at all. The insertion has to happen before the chords are
ordered, or the ordering has to be rebuilt after it.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Our projections already agree, so pcurves would not buy consistency

Looked at how the established kernels handle this. OpenCASCADE computes an
intersection as a **triple** — the 3D curve and a 2D pcurve in each surface's
parameters, approximated together by `ApproxInt_Approx` so all three hold one
tolerance. CGAL's `Arrangement_on_surface_2` does what `arrange`/`regions_of`
does but with filtered exact predicates. OCCT also has `ShapeFix_Wire`, whose
`FixSelfIntersection` exists precisely because self-intersecting wires happen and
are repaired rather than prevented — which is the same conclusion six repairs
reached here. And both OCCT and ACIS carry a **tolerance per entity**: a vertex
is a sphere, an edge a tube, so two computations of one point landing 8.9e-4
apart is absorbed rather than fatal.

The obvious lesson looked like "compute the pcurve during intersection instead of
projecting per face". Measured before building:

    rod-bore-cross    514 shared entries, 0 disagree
    ball-bore-cross   783 shared entries, 0 disagree

Every vertex that two faces on one surface both name gets the *same* `uv` from
both, to the last bit, modulo whole turns. **Our projections already agree**, so
a stored pcurve would buy no consistency at all, and the reasoning that made it
look attractive — the 16-against-78 divergence, the mirrored halves — is about
density and phase, not about disagreeing projections.

What that leaves from the survey is narrower and still useful:

* **tolerant entities** are the answer to our welding problems, and we have one
  global tolerance and exact index sharing — stricter than any production kernel;
* **a healing pass** is the answer to self-intersecting rings, which is what
  `arrange`/`regions_of` already are;
* **exact predicates** are the answer to the mirrored halves differing
  topologically, which no epsilon fixed.

None of those is the pcurve, and an afternoon would have gone into building one.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Welding cannot absorb the crossing, so the fill and the insertion are one change

Hoped the arrangement's own welding would make integration free: `arrange` merges
points within its tolerance, and the earlier "invented vertices" count was taken
at 1e-9. At the model tolerance it should reuse an existing ring point instead of
adding one.

    WELD tol 1e-9   invented 41 over 28 rings   worst-area-rel 1.53e-5
    WELD tol 1e-4   invented 41                 worst-area-rel 1.53e-5
    WELD tol 1e-3   invented 41                 worst-area-rel 7.16e-5

Identical. A lap's crossing lands in the *middle* of a segment — segments here
are 0.05 to 0.1 in `uv`, so the crossing is around 0.03 from either end, thirty
times a tolerance of 1e-3. Welding was never going to reach it, and raising the
tolerance only makes the area drift.

So the arrangement fill cannot avoid adding a boundary vertex, and the only way
both faces get one is to add it to the **curve** — which is the insertion built
two entries ago. They are not alternatives; they are one change, and the
insertion is the half that makes the other possible.

Which puts the whole line in one sentence: **the fill needs a vertex that only
the curve can give it**, and the remaining obstacle is the single case where a
clipped piece takes a corner-cutting crossing instead of the curve's own points.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The insertion cannot terminate: a projected crossing has no single 3D point

Read the coordinates of the five points the insertion added between vertex 146
and 155, and the gaps between them:

    146 -> 147   0.0120
    147 -> 148   0.0068
    148 -> 149   0.0031
    149 -> 150   0.0014
    150 -> 151   0.0006

They halve. The insertion is not resolving the crossing — it is **bisecting**
toward it and stopping only when the loop hits its cap of eight, leaving a
cluster of points where it gave up.

And the reason is structural rather than a bug in the loop. A ring crosses itself
in *projection*: two different points of the curve, far apart along it, land on
the same `(u, v)`. There is no single three-dimensional point there to insert.
Putting a vertex on one of the two segments splits that segment and leaves the
other still crossing it, so the next pass finds the same crossing a little closer
in, forever.

That closes the insertion, and with it the last idea for making a ring simple.
The five ring repairs failed because they removed a point two faces had agreed
on; this one fails because the point it wants to add does not exist.

So the fill must cover a self-overlapping boundary **using only the vertices it
already has** — which is possible, since a lap is covered twice rather than left
open, and is exactly what forcing the ear clip past its stall did: every test
green, the whole corpus intact, and one wrong volume on a rod bored twice down
the same axis, where the faces are coincident. That case is a separate defect
with its own name already in the code, `Declined::CoincidentFaces`, and it is not
being detected.

The package, then, is: force the clip, and decline coincident faces. Both halves
have been built and measured separately; neither has been tried together.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## "Decline coincident faces" would remove a feature, not add a guard

The plan from the last entry was: force the clip, and decline coincident faces,
since a rod bored twice down the same axis came back the wrong size. Checked
before building, and the second half is wrong.

Coincidence is already detected — `ssi` tests identity first and returns
`SsiResult::Coincident { opposite }` — and the boolean **handles** it:

    SsiResult::Coincident { opposite } => {
        if self.faces_overlap(ia, other, ib, tolerance) {
            coincident.push((ia, ib, opposite));
        }
    }

with a `shared_wall` rule downstream deciding which copy of a shared wall
survives. That is deliberate, and declining instead would take away a case the
kernel is built to do.

`Declined::CoincidentFaces` meanwhile is **never constructed anywhere in `src`**.
Its only appearance outside its own declaration is in a test's match pattern, as
one of the declines that test is willing to accept. A dead variant that reads
like a live guard is worth knowing about: I planned a tick's work around it on
the strength of the name.

So the double bore's wrong volume under a forced clip is not a missing decline.
It is the coincident-wall handling getting the wrong answer for that case — a
narrower and more delicate problem, and the one that actually stands between the
forced clip and shipping.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The double bore keeps two walls where it should keep one

Forced the clip so the case builds, and looked at what the coincident handling
produces:

    ONCE  vol 75.3906  faces 4  [cylinder, plane, plane, cylinder]
    TWICE vol 67.0291  faces 5  [cylinder, plane, plane, cylinder, cylinder]
    EXPECT 75.3982

Cutting the same bore twice should be a no-op and gives **an extra cylinder**.

The rule itself reads correctly for this case. A's bore wall faces into the hole
and the tool's faces out of the tool, so they are opposed, and

    (Difference, true,  false) => keep      A's wall survives
    _                          => drop      the tool's does not

is what should happen. Both walls being present means the tool's wall was not
dropped — that is, some of its pieces never carried `shared_wall` at all and fell
through to the ordinary rule, where `(Difference, false, inside) => (inside,
true)` keeps whatever lies inside A and flips it. That is exactly the wall A
already has.

So the defect is in the *marking*, not the rule: coincidence is decided per face
pair with `faces_overlap`, and the pieces those faces are cut into do not all
inherit it. A piece of the tool's wall that overlaps A's is kept as if it were an
ordinary piece of a difference.

That is a narrow and well-shaped bug, and it is the last thing between the forced
clip and a shipping change — the clip closes every case in the corpus and every
test, and this is the one wrong answer it exposes.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The wall is never registered, and the obvious reason is not the reason

Instrumented the marking and found the tool's wall is not a shared wall at all:

    ZZW B face 0 not a shared wall (3 pieces)

`wall_of_b` is empty, so every piece falls through to the ordinary difference
rule, which keeps whatever is inside A — the wall A already has. That is the
extra cylinder.

The cause looked certain. `faces_overlap` starts by finding an *outer* loop on
each face, and a whole cylinder wall has none: its two rims are lines at constant
`v` with no area in parameter space, so neither reads as outer and the function
returns `false` before testing anything. Falling back to the face's parameter
rectangle, which is what its footprint actually is, makes that test possible —
suite green, corpus byte-identical.

And it does not fix the double bore. Still five faces, still 67.0291 against
75.3982. So either `faces_overlap` still says no for a reason past the outer
loop, or coincidence never reaches it.

Recorded rather than shipped: the fallback is a genuine correction — a face whose
boundary has no area is not a face with no boundary — but it is inert by every
measure available, and this run does not ship inert changes. It is written down
here so the next attempt does not spend a tick rediscovering that a cylinder wall
states no outer ring.

The narrower question for next time: does `ssi` return `Coincident` for these two
at all? Everything above assumes it does, on the grounds that the surfaces are
identical, and that assumption has not been measured.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## `same_way` is a fact about surfaces, and the rule needs one about faces

Measured the assumption the last three entries rested on. `ssi` **does** see the
double bore's two walls as coincident:

    result[3] cylinder x tool[0] cylinder -> Coincident(opposite = false)
        A Cylinder { origin: [0,0,-6], axis: [0,0,1], x_dir: [0,1,0], radius: 1 }
        B Cylinder { origin: [0,0,-6], axis: [0,0,1], x_dir: [0,1,0], radius: 1 }

Identical surfaces, so `opposite` is false — **same way**. And the two *faces* on
those surfaces do not face the same way at all: the hole's wall points into the
void, the tool's points into the rod's material. What separates them is
`Face.flipped`, which `ssi` never sees, because it is given surfaces.

The keep rule then reads the wrong row:

    (Difference, _, true)      => drop both        <- taken
    (Difference, true, false)  => keep A's wall    <- correct for this case

So there are two defects here, not one, and neither is the one I set out to fix:

1. `faces_overlap` returns false for a whole cylinder wall, because it wants an
   outer loop and a rim has no area — so the pair never even reaches the rule.
2. `same_way` comes from surface geometry and the rule needs face orientation.
   Two faces on one surface with opposite `flipped` flags face opposite ways, and
   the rule cannot tell.

The second is the more interesting: it means the coincident-wall table has been
consulted with the wrong argument wherever a face is flipped, and the double bore
is simply the case that makes it visible. It is also cheap to test — `same_way`
should be `opposite ^ fa.flipped ^ fb.flipped`, or whatever the sign convention
works out to, and the corpus will say.

Three assumptions checked in three entries, three wrong: the wall was said to be
unmarked (true), the cause was said to be the outer loop (only half), and
coincidence was assumed undetected (false, it is detected and misread).

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## It compares rims, not footprints — and widening it costs a different case

Two more assumptions checked, one of them mine from the entry before.

**`same_way` is not the bug.** `ssi` reports `opposite = false` for the double
bore's two walls, which selects `(Difference, true, false) => keep A's wall` —
the correct row. The previous entry claimed the rule reads the wrong one; it does
not, and that claim is withdrawn.

**Nor is a missing outer loop.** Instrumented, `faces_overlap` *is* called for
the pair and sees `A f3 loops 2 outer 2 | B f0 loops 2 outer 2`. Both faces have
two rings and both count as outer. What it then does is take the **first** of
them — which is a *rim*, a line at constant `v` with no area — so it compares two
lines and finds no overlap however completely the walls coincide.

Taking the widest ring instead, and the parameter rectangle when none of them
bounds anything, is the obvious repair and it costs
`a_curve_along_a_face_boundary_does_not_cut_it`, which declines
`NeedsArrangement`. That is precisely the false positive the function's own
doc warns about — two walls of two solids side by side share a boundary and no
area, and a rectangle cannot tell that from a genuine overlap.

So the shape of the fix is now known and so is its trap: the footprint of a face
whose rings are rims is its parameter rectangle, *and* a rectangle overlapping
another is not the same as two faces sharing material. The test needs both — the
footprint to compare, and something better than area to decide, since area is
what the degenerate case lacks.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## Root cause found: a face does not contain its own interior

Traced the coincident-wall failure to one predicate. `point_on_face` takes the
first ring that is not a hole as the face's outer boundary:

    let Some(outer) = loops.iter().find(|l| !l.is_hole()) else { return false };
    point_in_ring(&outer.uv, [u, v])

A whole cylinder wall is bounded by its two **rims**, and a rim is a line at
constant `v` with no area. So "outer" is a line, and `point_in_ring` against a
line is false for every point — including the middle of the face's own
rectangle. Measured directly:

    ZZS sample u 0.000 v 6.000 -> A false B false

That point is the centre of A's own bore wall, and the face says it is not on it.
Everything else follows: `faces_overlap` compares two lines and finds nothing,
`wall_of_b` stays empty, no piece is marked, both copies of the shared wall
survive, and the rod bored twice comes back holding 67.03 of 75.40.

Fixed by giving such a face its parameter rectangle as its footprint, with
periodic folding. Whole suite green, 45-chain corpus byte-identical — and inert
by every measure available, including the most direct case there is:

    cylinder against itself, before and after
        Union         NotWatertight { open_edges: 256 }
        Intersection  NotWatertight { open_edges:  32 }
        Difference    NotWatertight { open_edges: 256 }

Unchanged. So a boolean of a solid with itself declines either way, and the fix
cannot be reached from outside the crate — which makes it untestable, and this
run does not ship untestable changes. Recorded instead, with the sample above,
because it is the root cause of a defect that took six entries to corner and the
next attempt should not have to find it again.

Worth noting what it says about the corpus, too: with the forced clip the chain
count *rises* to 13/45 while the double bore returns a wrong volume, because
`is_valid_solid` cannot see wrongness. That metric has now scored destruction as
progress twice and a wrong answer as progress once.

Green: 40 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 639 lib.

## The pin is a wrong answer, not a precision loss

The new inclusion-exclusion test found `rod - column` out by 2.5%, and the cause
is arithmetic rather than topology:

    f0 cylinder  loops None                       (grid fill, refines)
    f1 plane     loops [(16, 10.286), (8, -1.96)] (the cap)
    f2 plane     loops [(16, 10.286), (8, -1.96)]
    f3..f6 plane loops [(6, 11.2)]                (column walls, 1.4 x 8, exact)

A 16-gon inscribed at radius 2 has area 12.246 against `pi * 4 = 12.566`, and
`12.246 - 1.96 = 10.286` is exactly what the cap carries. The solid is being
tessellated as a **16-gon prism**: `12.246 * 8 - 15.68 = 82.29`, which is the
volume measured, to four figures.

The rod on its own measures 100.4906 against 100.5310, so it refines properly.
The boolean's *result* is coarser than its input: the cap stores the primitive's
sixteen-point circle, storing it pins that rim, and the wall — which would refine
to about 145 points — has to follow the pin.

So the pin does not cost precision, it costs **correctness**, and in a case that
passes every other test in this repository. `a_blind_hole` losing 1.3% was the
same defect in smaller print.

That reframes the restate-and-unpin work recorded several entries ago. It was
measured as *inert* and set aside on that basis — corpus unchanged, suite green.
It is not inert: it is the fix for a 2.5% wrong answer, and the reason it looked
inert is that nothing in the suite or the corpus was measuring size. Now
something is.

Green: 41 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 675 lib.

## Landed: the rings follow their edges, and a wrong answer goes with the pin

`restate_rings` and the unpin, shipped — the change that was measured as *inert*
eleven entries ago and set aside on that basis. It is not inert. It is the fix
for a 2.5% wrong answer, and the only reason it looked inert is that nothing was
measuring size until the inclusion-exclusion net went in last entry.

    rod - column   before  82.2870   a sixteen-gon prism
                   after   84.8106   = |A| - |A n B| exactly
                   exact   84.8510   (the rod itself measures 100.4906 of 100.5310)

A ring restated in its edges' own vertices cannot be left behind when they
refine, so the pin that held it is no longer needed — and the pin was what kept a
cap at the primitive's sixteen points while the wall beside it went to a hundred
and forty-five.

**What it costs, stated plainly.** `the_same_cut_asked_for_five_ways` goes from
five tolerances resolving to three. That is a real loss of coverage, and it is
the trade this file's header asks for in as many words: *a result is never worse
than a decline*. Two honest declines are worth more than one solid of the wrong
size. The count is asserted at three rather than removed, so a change that
recovers them is visible.

The `rod - column` skip in the new correctness test is gone with it — every pair
in the matrix now measures every other, with no exceptions carried.

Measured: 41 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 686 lib, all
passing; the 45-chain corpus holds at 12 with no case gained or lost; clippy
clean across `src/brep`.

Eleven entries between measuring this change and understanding what it was for.
The lesson is the one the plan opened with: the ladder of invariants comes first,
because a change can only be judged against what is being measured, and "inert"
meant "invisible to the tests that existed".

## A second cut by the same tool loses an eighth of the solid

Extended the correctness net to `(A - B) - B = A - B`, and it caught a live wrong
answer on its first run — then a second:

    ball - bore  again   94.6184 became 82.7862
    ball - ball2 again   88.2798 became 77.6824

Both of the second cuts this corpus can make. Both about an eighth of the solid.
Both perfectly watertight and `is_valid_solid`, which is why nothing in the suite
noticed, and why the chaining metric counted them as successes. The rod bored
twice down its axis is the same defect and *declines*, so it never surfaced as a
wrong answer at all.

Tried the `point_on_face` fix on it — the one recorded two entries ago as the
root cause of the coincident-wall failure, a face not containing its own
interior. No change. So that predicate is genuinely broken *and* genuinely not
the cause of this, which is worth having established rather than assumed twice.

Landed as `cutting_twice_with_one_tool_takes_nothing_the_second_time`, `#[ignore]`
with the numbers in the body and a one-line reproduction. A specification of what
the kernel owes, executable, rather than a paragraph of prose that drifts. The
passing half of the net — 26 identities over every pair — stays green.

Two entries, two live wrong answers found, one fixed. The net is earning its
runtime.

Green: 41 boolean (1 ignored), 24 brep, 13 brep_csg, 18 nurbs, 12 step, 686 lib.

## Landed: two walls that coincide are finally seen to

Opened the second cut and the picture was the double bore again, one step worse
because this one *returns* something:

    ONCE   vol 94.6184  faces 2  [sphere, cylinder]
    TWICE  vol 82.7862  faces 3  [sphere, cylinder, cylinder]

Two coincident bore walls, identical parameters, both kept. The cause is the one
recorded two entries ago and it needed **both** halves fixed, which is why
neither alone moved anything:

* `point_on_face` took the first ring that is not a hole as the face's boundary,
  and for a whole cylinder wall that is a *rim* — a line at constant `v` with no
  area — so the face did not contain its own middle;
* `faces_overlap` chose its rings the same way, so it compared two rims and found
  no overlap however completely the walls coincided.

With both, the pair is recognised, one copy of the wall is kept, and

    ball - bore cut again by bore    was 82.7862 of 94.6184, now declines

A wrong answer becomes an honest decline, which is the trade this file's header
asks for. The 45-chain corpus is unchanged at 12 and loses its one `INVALID`
line; the suite is green; `src/brep` carries one clippy warning, and it is not
one of mine.

`ball - ball2` cut twice is still wrong — 88.2798 becoming 77.6824 — so the
ignored specification stays, with its numbers narrowed to the case that still
fails and the fixed one written up beside it.

Three entries, three live wrong answers found by the net, two of them now gone.
None of them was visible to any test in this repository a week ago.

Green: 41 boolean (1 ignored), 24 brep, 13 brep_csg, 18 nurbs, 12 step, 686 lib.

## Landed: a second cut no longer loses an eighth of the solid

The sphere pair fell to a third cause, found the same way — printing what the
test actually saw:

    ZZP f1 x f0: u (3.1416, 9.4248) v (-1.5708, 1.5708)  onA 0  onB 25  both 0

`u` runs from pi to 3pi. A sphere cut by a sphere keeps a cavity wall whose rings
live a whole turn away from the canonical period, and `invert` answers *in* that
period — so every point of the face missed its own ring by 2pi, and nothing was
ever on it. Folding the point to where the ring lives fixes it.

Three causes, one defect, and each hid the next:

    faces_overlap compared two rims             (a rim has no area)
    point_on_face chose its ring the same way   (a face lacked its own middle)
    and tested the point a turn from the ring   (periodic mismatch)

With all three, both second cuts decline instead of lying:

    ball - bore  again   was 82.7862 of 94.6184   now declines
    ball - ball2 again   was 77.6824 of 88.2798   now declines

So `cutting_twice_with_one_tool_takes_nothing_the_second_time` is **no longer
ignored** — the property holds and is enforced from here. The corpus reports
**zero** `INVALID` results, where it had one; the chain count is unchanged at 12;
the whole suite is green at 42 boolean tests.

Four entries ago the corpus scored a wrong answer as a success and nothing in the
repository could tell. Now the identity net and the second-cut property both run
on every build, three predicates that had been quietly wrong are right, and two
twelve-per-cent errors are gone.

Green: 42 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 686 lib.

## Union and intersection commute, and two pairs answer differently by order

Extended the net to commutativity, since the last three additions each found a
live defect. This one does not, which is worth as much:

    COMMUTE 32 checked, 0 disagree

Every pair that resolves both ways gives the same volume both ways. No wrong
answers there.

Two pairs do resolve one way and decline the other, though — `bore` against
`cross`, in both union and intersection. For a commutative operation that is a
defect in itself: the same question asked in the other order gets a different
answer, and only one of them can be the best the kernel can do. It is not a wrong
answer, so the net's assertions would not catch it; it is a coverage asymmetry,
and worth its own check.

A process note, because it nearly became a false finding. Three re-runs reported
zero asymmetries and I was a keystroke from recording "nondeterministic". The lib
was not compiling — the concurrent lattice work has `Strut::Kelvin` uncovered in
a match, among four errors — and `grep` over a failed build returns nothing,
which reads exactly like a clean result. The same trap as the unset environment
variable several entries ago, and the same lesson: a probe that reports nothing
should be made to prove it ran.

`src/brep` carries none of those errors. Nothing here is blocked once the lattice
work compiles; the last verified state is 42 boolean, 24 brep, 13 brep_csg, 18
nurbs, 12 step, 686 lib, all green.

## Instrumentation had shipped with the landing

Checking the tree after the commutativity work turned up debug code inside the
landed change, carried in with the checkpoints it was restored from:

    boolean.rs   `... && std::env::var("ZZGATE").is_err()`   on the closure gate
    body.rs      a `ZZR` println inside `restate_rings`

The second is noise. The first is not: `ZZGATE` **disables the gate that decides
whether a result is watertight enough to return**. Set that variable and the
kernel hands back bodies it had judged unfit — precisely the "wrong answer rather
than a decline" this run has spent four entries removing, reintroduced as an
environment switch.

It landed because the restate-and-unpin change was assembled by restoring saved
files rather than by applying a diff, and those files were snapshots of a
debugging session. Both are gone; the suite is green with them out.

The habit that caught it was cheap and should be routine: grep the tree for the
instrumentation prefix before calling anything done, not just after reverting an
experiment. Every entry has ended with a "clean" line; this one was the first to
check the *landed* code rather than the reverted code.

Green: 42 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 696 lib.

## Argument order decides whether three operations resolve at all

The commutativity asymmetry, opened. It is not about the operation:

    Union         bore-first ok faces 7 | cross-first NeedsArrangement { face: 0 }
    Intersection  bore-first ok faces 3 | cross-first NeedsArrangement { face: 0 }
    Difference    bore-first ok faces 4 | cross-first NeedsArrangement { face: 0 }

All three resolve with the wider cylinder as `self` and all three decline with
the narrower one. So it is not commutativity that fails — it is that the
arrangement can split one wall by the other's curves and not the reverse, and
which body is `self` decides which wall gets split. A caller who swaps their
arguments gets an answer or a refusal, for the same solids.

A decline, so no wrong answer, and the net's assertions cannot see it. Recorded
as a coverage gap with a workaround that costs nothing: put the larger solid
first.

### And the net is three times cheaper

It had grown to dominate the suite. The identity holds at *any* tolerance — it is
about the operations agreeing with each other, not about either being exact — so
the only question was how much margin a coarser one leaves:

    1e-2   worst 1.40e-2   too coarse for the one per cent asserted
    5e-3   worst 3.91e-3   24s
    1e-3   worst 2.14e-3   74s

Moved to 5e-3: the same twenty-six pairs measure each other, with two and a half
times the margin still to spare, for a third of the time. `brep_boolean` goes
from 215s to 147s. A guard nobody wants to run guards nothing.

Green: 42 boolean, 24 brep, 13 brep_csg, 18 nurbs, 12 step, 696 lib.

## Landed: the model states what code had been inferring (items 1-3, and Surface)

Four changes to the definitions rather than the algorithms, each one removing a
question the code had been answering by guesswork — and each traceable to a
defect this run actually found.

**A ring says what it is.** `TrimLoop::area`'s *sign* said outer or hole and its
*zero* said nothing, and zero is the case that matters: a cylinder wall's rims
are lines with no area. Three predicates independently read a rim as a face's
boundary — `point_on_face`, `faces_overlap`, the trimmed fill — and it cost two
twelve-per-cent wrong answers. Now `is_degenerate`, `is_outer` and `is_hole`
answer it, scale-free, and the three ad-hoc `area.abs() > tolerance * tolerance`
clauses that had grown at the call sites are gone.

**A ring is asked about its own branch.** `TrimLoop::contains` folds the
parameter to where the ring lives before testing it. Six sites tested a point
against a ring and only two folded; a sphere's cavity wall with `u` in `(pi, 3pi)`
missed every point of itself by a turn.

**A curve carries its surfaces.** `Curve3d::OnSurfaces` is the intersection as it
is actually defined — the spline says where to look, `settle` says what is there.
Measured on two crossing cylinders: asked halfway between its samples, the chord
is out by more than 1e-3 and the curve by less than 1e-5. Not an approximation of
the curve, because a quadric pair meets in a quartic that no spline can be; the
spline is only a parameterisation.

**A surface states its period and answers on the branch asked for.** `periodic()`
gave a boolean and eighty-one places in this module reached for `TAU` to fold by
hand. `period()` gives the number and `invert_near(p, anchor)` gives the answer on
the branch you are working on, which makes the correct thing the easy thing.
`Surface::translated` also exists now — it simply did not, which is why a
procedural curve could not be moved.

Everything green: 42 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 715 lib.

Still to do from that list: **one source of truth per face** — `Face` carries
`u_range`/`v_range` *and* `loops`, two representations of one fact that
`restate_rings` exists to reconcile — and **per-entity tolerance**, which every
production kernel has and this one does not.

## Landed: one rule for what bounds a face (item 4)

`Face` carries a parameter range *and*, sometimes, trim loops, and the two answer
the same question. Which one was authoritative had been decided separately at
every site that asked — and this run found the codebase had discovered the rule
**four** times independently:

    tessellate      trimmed if `loops` is non-empty
    point_on_face   the first ring that is not a hole
    faces_overlap   the same, plus its own degeneracy clause
    all_of_it       `area.abs() > 1e-12 && uv.len() >= 3`, with a comment
                    describing the insight in its own words

The fourth is the tell. Its comment already says "a swept face's loops are its
rims, which enclose nothing, so such a face is bounded by its parameter range" —
the exact conclusion `point_on_face` and `faces_overlap` each cost a wrong answer
to reach separately.

    pub enum Footprint<'a> {
        Rectangle { u: (f64, f64), v: (f64, f64) },
        Rings(&'a [TrimLoop]),
    }

`Footprint::of(rings, u, v)` is the rule, taking rings rather than a face so that
a caller working from *derived* loops asks the same question as one working from
stored ones. All four sites now call it. A rim is an edge, not a footprint, and
there is one place that says so.

Green: 42 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 715 lib.

Item 5 — per-entity tolerance — is the one left, and it is the largest: every
comparison in the kernel currently takes a single global `tolerance`, where a
production kernel gives each vertex and edge its own.

## Item 5 measured before building, and the answer is smaller than the plan

Per-entity tolerance is the largest change on the list, so the first question was
whether it has a consumer here. It does — a chaining corpus carries

    15 results, 2769 vertices, 34 pairs closer than the tolerance

pairs of *distinct* vertices that the kernel's own tolerance says are one point.
That is the crack every later stage has to work around, and the most expensive
instance this run found — one crossing minted twice, 8.9e-4 apart, the two faces
either side of it disagreeing — was exactly this.

But the simple fix comes first. The weld ran at **half** the tolerance, and two
points closer than the tolerance are the same point. Widening it is green across
the suite and removes four of the thirty-four. Landed.

The other thirty are born *after* the weld, and the specification written to pin
this found something sharper than expected:

    ball - column: vertices 0 and 22 are 7.145e-16 apart

Not "within tolerance" — identical, to the last bit, and never merged. So this is
not a tolerance question at all for most of them: something mints a vertex that
already exists. A per-entity tolerance would not have fixed one of those, and
building it first would have been a large change aimed past the defect.

`no_two_vertices_of_a_result_are_the_same_point` is landed `#[ignore]`d with the
numbers and a one-command reproduction, beside the second-cut specification —
which was written the same way two entries ago and fixed two entries later.

Green: 42 boolean (1 ignored), 26 brep, 13 brep_csg, 18 nurbs, 12 step, 715 lib.

## Landed: no two vertices of a result are the same point

The thirty pairs left after widening the weld were not duplicates in the sense
the plan assumed. Every one of them was an **orphan**: for each pair, the
higher-numbered vertex was referenced by no edge and no ring.

    ball - column   v0 and v22, 7.145e-16 apart
                    v0  on edges 0 and 2, faces 0 and 1
                    v22 on nothing at all

Assembly mints a vertex whenever a piece names a point, and some of those points
end up on no boundary — a piece that was discarded, a crossing that was
superseded. They are invisible in the solid and they are not free: anything
asking whether two vertices of a body are the same point gets a yes about
geometry that is not there.

`prune_orphan_vertices` drops them and remaps what is left. A bored ball goes
from 157 vertices with twelve duplicate pairs to **137 with none**, and

    no_two_vertices_of_a_result_are_the_same_point

is no longer ignored. Written as an ignored specification one entry ago, enforced
the next — the same shape as the second-cut property, which was also written
ignored and fixed two entries later.

So item 5 closes without the change it was named for. Per-entity tolerance would
have been a signature change through the whole module aimed at thirty-four pairs,
four of which wanted a wider weld and thirty of which wanted deleting. Measuring
the demand first was worth more than the design.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## Re-baselined, and the resample is two tests away instead of four

With items 1-5 landed, re-measured the two invariants that were open:

    A     17 crossing rings of 85, over 15 results   (was 17 of 49)
    sweep 3 of 5 tolerances resolve, 403 and 148 open edges at the finest two

A is unchanged in count while the ring total has grown from 49 to 85, so
proportionally it has halved — more faces state rings now, and the new ones are
sound.

Then retried the even re-sample of closed traced curves. It was measured long ago
as fixing A outright and costing four tests, and the reason it cost them was
rings not following a changed sampling — which is exactly what `restate_rings`
now does. The retry:

    A      17 -> 1 crossing rings, same 85 rings over the same 15 results
    suite  two failures, down from four

    every_step_of_a_ring_is_a_step_of_an_edge   a bore through a solid: 77 open
    a_result_can_be_cut_again                   a bored ball: 24 open

Not shippable, but the direction is confirmed twice over: the overshoot is the
cause of the crossings, re-saying the curve removes them without destroying
anything, and the model work has halved what that costs. Reverted.

What the remaining two say is that some ring step stops being an edge step when
the sampling moves — which is the same shape as the failure `restate_rings` was
built for, one level further out: the *edges* are rebuilt from the curve, but a
piece's ring is clipped from it, and the clip does not follow.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## The resample loses a face because a piece's sample lands outside

Opened what the re-sample costs, and it is one face, not a scatter of edges.

    baseline    4 faces, 0 open
    re-sampled  3 faces, 77 open   [cylinder, plane, plane] — the bore wall gone

The rod's wall keeps holes of 38 and 39 points where the bore enters and leaves,
and nothing bounds them, which is the 77.

Followed it into classification, and the answer is exact:

    ZZP B face 0 cylinder made 3 pieces
    ZZP  piece from B: inside_other=false shared=false keep=false
    ZZP  piece from B: inside_other=false shared=false keep=false
    ZZP  piece from B: inside_other=false shared=false keep=false

The tool's wall is split into three pieces and **every one is judged outside the
rod**. A bore passing through a rod has its middle piece inside by construction,
so this is not a marginal call — the piece's representative point is landing
somewhere it should not be.

So the re-sample does not break the arrangement, the fill, or the ring
invariants. It moves `piece.sample`, and a piece is kept or dropped by where that
single point falls. That is a much narrower thing to fix than "the fill depends
on sampling", and it is the same shape as the defects this run has already
closed: one predicate answering about a representative rather than the thing it
represents.

`sample_between` is where to look next.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## A seventh site for the ring rule, and the re-sample is still two tests away

`sample_between` picks the point that decides whether a piece is kept, and it was
testing candidates against rings with a bare `point_in_ring` — unfolded, the same
periodic-branch mistake items 2 and 4 were about. On a cylinder wall whose rings
sit an unwrapped turn away, a candidate *outside* the piece can read as inside
it, and one point decides the whole piece.

Routed through `TrimLoop::contains`, which is now the only way a ring is asked
about a point. Green everywhere — the seventh site of one rule, and leaving it
unrouted was the inconsistency, not shipping it.

It did **not** unblock the re-sample, which still costs the same two tests:

    every_step_of_a_ring_is_a_step_of_an_edge   a bore through a solid, 77 open
    a_result_can_be_cut_again                   a bored ball, 24 open

So the lost bore wall is not a folding error. What is known about it now is
precise: the tool's wall splits into three pieces and all three are judged
outside the rod, when the middle one is inside by construction. The candidate is
being accepted or the classification is answering wrongly, and the fold was the
cheaper of the two to rule out.

Next is the other half: `point_in_mesh` against a tessellation of the other
solid, which is what `inside_other` actually asks. A piece deep inside a rod
reading as outside points at the ray cast, not at the point.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## The lost face is a sample on the rim, not in the middle

Followed the re-sample's lost bore wall to the point that decides it. The tool's
wall makes three pieces and every sample lands outside the rod:

    sample [-4.200, 0.037, -0.417]   r 4.200   correctly outside
    sample [-1.999, 1.108,  0.481]   r 2.286   the middle piece
    sample [ 1.796, 1.209, -0.345]   r 2.165

The rod has radius 2. The middle piece spans `x` from about -2 to +2, so its
representative belongs near `x = 0` — and it sits at `x = -1.999`, which is the
**rim**. Not a marginal misclassification: the point is on the piece's boundary,
where the answer is a coin toss, rather than in its interior where it is not.

So `inside_other` is answering correctly about the point it was given. The fault
is upstream, in which point `sample_between` chose, and its own comment is about
exactly this hazard — "the *furthest* candidate from the boundary, in space, not
the first one that is inside ... a classification mesh coarser than that put the
piece on the wrong side".

Under the re-sampled curves that search is coming back with a boundary point
anyway, which means either every interior candidate is being rejected as unusable
or none is generated for this piece's shape. Folding `usable` through the ring
(last entry) was the first of those two to rule out, and it did not change this.

So: why does a piece spanning the whole rod have no usable candidate away from
its rim? That is the question, and it is narrow enough to answer by printing the
candidates.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## Correction: the samples are right, the pieces are wrong

Counted the candidates `sample_between` generates for the pieces of the tool's
wall, expecting to find an interior point being rejected. The opposite:

    fan-hit outer 37 holes 0 tried 8  usable 8 best_margin 0.140 target 0.140
    fan-hit outer 37 holes 0 tried 7  usable 4 best_margin 0.304 target 0.140
    fan-hit outer 37 holes 0 tried 23 usable 7 best_margin 0.189 target 0.140

Every piece finds usable candidates and clears its target. The search is working.

And checking the geometry rather than assuming it, the samples are *correct*. The
tool's wall meets the rod where `x^2 + y^2 = 4`, so at `y = 1.209` the rim is at
`x = 1.593`, and the sample sits at `x = 1.796` — beyond it, genuinely outside.
The same at the other end: rim at `-1.665`, sample at `-1.999`.

So the previous entry's conclusion — "a sample on the rim, not in the middle" —
is **wrong** and withdrawn. The samples represent their pieces faithfully. What
is missing is the piece itself: the tool's wall is split into three parts and
none of them is the part inside the rod.

That moves the question from classification to `split_face`, and makes it sharper
than anything in the last four entries: a bore through a rod divides the bore's
wall into three by its two rims, and the arrangement is returning three pieces
that are all outside. Two rims, three pieces, and the middle one absent.

Green: 43 boolean, 26 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## Why the middle piece is missing: a rim read as a hole

Printed the parameter extent of every piece the tool's wall is cut into, both
ways. The whole answer is in six lines:

    baseline    v [0.00 4.48] 56pts   v [4.00 8.00] 76pts   v [7.52 12.00] 56pts   holes 0
    resampled   v [0.00 12.00] 36pts holes 2      v [4.00 4.48] 37pts   v [7.52 8.00] 37pts

The baseline reads each rim as a **divider** and gets three bands, the middle of
which (`v` 4.00–8.00) is the piece inside the rod. The resample reads each rim as
a **hole** in one full-height wall, so no band is inside anything and the two
leftovers are the rims themselves. Nothing is misclassified; the middle piece is
never built. This is the ambiguity `src/step/mod.rs` already names for export —
a closed curve on a closed surface bounds two regions without saying which —
appearing in the boolean, where it had not been recognised.

Here it is not actually ambiguous. Each "hole" runs `u` from 1.23 to 7.44, a span
of `2pi`: **a hole cannot go the whole way around the surface it is a hole in.**

Landed that as `TrimLoop::wraps(period)` — sum the steps folded into the period;
a ring that closes in the plane totals zero, one that wraps totals a period. It
separates the two readings that `area` cannot, and is scale- and branch-free.
Not yet consumed by the split; the predicate and its evidence come first, since
the last four entries all guessed at this mechanism and missed it.

Two corrections this tick, both from the same habit of reading a filtered stream:
a probe printed nothing at all and I nearly recorded "no pieces" — the probe test
had failed to *compile* (`shells` is a method, not a field). That is the third
time an empty grep has meant a broken build. The raw output is the check.

Green: 27 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## A graze found by luck: the same shape at twice the size got a different answer

Chasing the argument-order asymmetry, swept crossing cylinders at several radii
and found something worse sitting next to it:

    r 1 x 1   TangentialContact { face_a: 0, face_b: 0 }   both orders
    r 2 x 2   NotWatertight { open_edges: 340 / 363 }      both orders

Two equal cylinders crossing at right angles are tangent at two points whatever
the radius, so `TangentialContact` is right and the second line is not. The SSI
is identical at both sizes — two ellipses, the exact closed-form answer — so the
difference is downstream, at the test that asks whether the surfaces cross or
only touch.

That test asked each *sample* of the curve whether `|n1 x n2| < 1e-2`. Working
out the geometry: near the graze the sine runs about `sqrt(2)·x/r`, so the window
is `|x| < r·7.1e-3` — it does scale with the model. What does not scale is
whether a sample happens to land inside it. At `r = 1` one did; at `r = 2` none
did. **The detector was luck.**

Fixed by bracketing rather than sampling. Both normals are perpendicular to the
curve they meet along, so `n1 x n2` is parallel to its tangent, and taken *along*
the curve it is signed and passes through zero exactly at a graze.

The bracket alone over-fired — four tests, an ordinary cross-bore reading as
tangential. Printing the flip said why in one line:

    flip at 39/41 sine 0.9513 lean 1.262e-1 -> -1.901e-2

sine 0.95 is nowhere near a graze, so that sign change is a bad *chord*, not a
tangency: at the closing step of a loop the chord is not along the tangent and
`dot(x, step)` measures nothing. So the sign change is not the test — it is what
lets the threshold be loose. A graze must both reverse the lean and drive the
sine down, and `0.2` sits between the 1.8e-3 of a real graze and the 0.87 of
every genuine crossing. `r·0.2/sqrt(2)` is a window a hundred times wider than
before, which is what makes it robust instead of lucky.

Checked for coverage loss rather than assumed: twelve probe cases across three
radius pairs and both argument orders, all identical to before the change. The
suite ran 79s against 235s on one run, which is machine load — the second run
was 211s. Not a result, and not recorded as one.

The asymmetry itself reproduces at `r 1 x 0.9`, `NeedsArrangement { face: 0 }`
one way and three good answers the other, and is still open. Also new: `1 x 0.7`
declines `NotWatertight` where `2 x 0.7` — the same ratio, twice the size —
resolves. Another scale-dependent verdict, and the next thread to pull.

Green: 44 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 725 lib.

## The erratic ratios were a rim closing one turn early, and one of them lied

Two corrections to the last entry first. `1 x 0.7` and `2 x 0.7` are ratios 0.7
and 0.35 — not one shape at two sizes, so that was never evidence of anything.
Tested properly, scaling radii, lengths and tolerance together, every verdict is
identical across scales 0.5 to 4: **163 open edges at all four**. There is no
scale defect there. And the older workaround "put the larger solid first" is
wrong: at ratio 0.5 the thick one must come first and at 0.7 the thin one must.

What is real is stranger. Sweeping the tool-to-rod radius ratio:

    0.45 refused   0.50 answered   0.55 refused   0.65 answered   0.70 refused
    0.75 answered  0.80 refused    0.85 refused   0.90 refused    0.95 answered

Geometry does not alternate, so something discrete was deciding it. Holding the
shape and moving only the tolerance kept the refusal at every value while the
open-edge count swung 48, 82, 157, 163, 165, 190 — a persistent failure whose
*extent* is sampling noise.

Printing the tool wall's pieces at a working ratio and its failing neighbour:

    0.75   v [0.00 5.34]  v [5.00 7.00]  v [6.66 12.00]
    0.70   v [0.00 7.00]  v [6.71 12.00] v [5.00 5.29] wraps [true, _]

`5.00 5.29` is exactly the rim's own extent, and the middle band — the piece
inside the rod — is absent. Same failure as the resample entry, live in the
baseline. `TrimLoop::wraps` reads `[true, false]` on precisely those slivers and
`[false, false]` on all ten sound pieces beside them, at three ratios.

A guessed cause did not survive: the working ratios sit on branch `pi..3pi` and
0.70 straddles `u = 0`, so the seam looked responsible — refuted by 0.80, which
fails at `u [0.04 6.33]` without straddling anything, and by 0.85, which fails on
the shifted branch.

The rule that does hold: two rims bound a band by going out along one and back
along the other, net travel zero. A ring totalling a whole period closed itself
after one turn. Landed as a guard, and it needed one restriction — applied
everywhere it rejects four sound results, because where a surface closes to a
point a single wrapping ring is a genuine boundary: a spherical cap has one
circle and a pole, and a cone's apex the same. A cylinder closes nowhere, so
there a single turn is always a ring that stopped early.

**The guard's cost is negative.** Ratio 0.80 the thin way used to return a body:
watertight, two shells, seven faces — and 23.53 where 20.46 is right, 15% too
much material with nothing structural to show for it. Exactly the wrong answer
this kernel is arranged to prevent, and it is gone. Elsewhere the declines are
merely honest now: `NeedsArrangement` naming the face instead of `NotWatertight`
counting 146 to 163 edges it cannot explain.

Held by a new test that checks all twenty-two cases by *size*, against the
shared volume of two perpendicular cylinders done by quadrature — no closed form
needed for a number that only has to be right. Eleven resolve and every one is
within 1%. I wrote twelve from miscounting the sweep; the test said 11.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The trace loses a ring because its seeds are spaced by the size of the box

Went after the other half of the rim defect — the walk that closes a ring one
turn early — and the trail led somewhere else entirely.

The wrapping test in `split_face` measures a ring's *span*, and the comment there
already argues at length that the *turn* is the right measure and that swapping
them costs one test. Retried with everything landed since: it still costs exactly
`a_result_can_be_cut_again`, now 24 open edges rather than 7, and it buys
**nothing** — 11 of 22 ratios resolve before and after, every outcome identical.
Rejected again, and now with the gain measured rather than assumed.

That comment blames an upstream loss: two traced rings of 41 points arriving as
one ring of 6. Instrumented the boolean's own call:

    march (0,0) sphere x cylinder: ["sampled 6 closed z -2.780..-2.380"]

So `march` returns the six-point ring itself. The comment says "march is not the
one giving it. Called directly on that pair it returns both rings, in full" —
that was a different call with a different box, and inside the boolean it is not
true. One ring of two, at a sixth of the samples.

Then the box. `march` traces within `box_both`, the *union* of both bodies'
extents, so a ball of radius 3 drilled by a tool 60 long gets a box 60 tall for
two rings living inside `|z| < 3`. Tightening it to the overlap — a curve on
both faces is inside both bodies — recovered both rings, 39 points each, closed,
over their full extent. Exactly what the comment predicted was there.

But it costs three tests, and so does the `extent_of` correction it needs. That
correction is right on its own terms: `extent_of` returned early on a body's
*vertices*, and a bored ball has vertices only on the bore's rims, so the box
came back smaller than the solid. Harmless in a union, where the other body
covered it; not harmless in an overlap.

Four combinations, and they say one thing: a bigger box breaks tests, a smaller
box fixes the trace. Which is a fact about `seeds`, not about boxes:

    const N: usize = 12;
    let reach = span(bounds);
    ... if a.distance(p) > reach / N || b.distance(p) > reach / N { continue }
    ... if out.iter().all(|r| v3::dist(*r, q) > reach / N) { out.push(q) }

**The dedup radius is a twelfth of the box.** The two rings of this case sit 5.2
apart in `z`; a 60-tall box makes that radius 5.0, so the second ring is absorbed
into the first and never traced. Nothing about the geometry decided that — the
length of an unrelated drill did.

Same shape as the graze detector fixed two entries ago: a threshold taken from an
arbitrary global instead of from the thing being measured. The graze one was
worth fixing because the fix was local. This one is in the seeding of every
traced intersection, so it wants a measure of its own — feature size from the
surfaces and the tolerance, not from whatever box the caller passed — and that
is the next thing to build rather than another box.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Seeds spaced by the box threw away a ring, and the test could not see it

Both halves of last entry's diagnosis are in `march`, and both are `span(bounds)`:

    let step = span(bounds) * 0.01;                      // how finely it traces
    ... if out.iter().all(|r| dist(*r, q) > reach / N)   // when two seeds are one

A ball of radius 3 drilled by a tool 60 long gives a box 60 tall, so the step is
0.6 for a ring 2.2 around — the six-point ring — and the dedup radius is 5.0 for
two rings that sit 5.2 apart. The second ring was absorbed into the first and
never traced. Neither number came from the geometry.

The dedup one is landed. `march` already drops a seed lying on a curve it has, so
a repeated seed costs a `distance` and nothing else — the measure can err small,
and it now errs to the trace step.

**It costs one test, and that test was passing on a wrong answer.** The bored
ball cut again:

    bored ball   volume 94.6184  faces 2  closed  valid
    cut again    volume 94.6046  faces 3  closed  valid

The drill should take 2.01 out of it. It took **0.0138** — under a percent — and
the body came back closed, valid and a face richer, which is all
`a_result_can_be_cut_again` ever asked. A hole with an entry and no exit, because
the trace found one ring of two. That case now declines, and the test says so
explicitly, with a note to check what it weighs before believing it if it ever
resolves again.

The test itself is the other half of the fix: a face count says the drill was
*mentioned*, not that it cut. It now measures what came out, against `pi·r²`
times a thickness every case clears.

The step is not landed. Sized to the curve — `sqrt(8·r·tol)`, the longest step
holding a circle's sagitta within tolerance — it gives that ring 41 points and
both rings closed and full, which is exactly what this file predicted two entries
ago. It costs three more tests, all of them results lost on the wrapping path
that cannot take a face whose only ring wraps. That path is the blocker, and it
is now the only thing between here and a trace that answers to its curves. The
reasoning is left in the comment at the step so the next attempt starts from the
measurement rather than the idea.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The wrapping path: a cut that begins and ends at one point

Went at the blocker named last entry, with the step fix applied so the failing
case exists. It declines at one place — `planar::subdivide` returning `None`,
reached through an `.ok_or(...)?` that four earlier taggings of the `return Err`
sites all missed. Worth remembering: not every decline is a `return`.

The face is the *drill's* wall, cut by the ball it passes through:

    outer 35 pts u [2.1247 8.4079] v [0.0000 60.0000], 2 paths [48, 48]
    path 0: starts [2.1247 27.2199] ends [8.4079 27.2199]
    path 1: starts [8.4079 32.7801] ends [8.4079 32.7801]

Path 0 crosses the rectangle from one seam edge to the other. Path 1 begins and
ends at the *same point* on the far edge. A drill's entry and exit rings run
opposite ways round it, as entry and exit must, and the fold that places chord
ends was choosing by proximity — "keep an endpoint that landed exactly on the far
edge" — so both ends of the backwards one were kept there. A cut that begins and
ends at one point divides nothing.

Fixed by asking the ring which way it runs — the same turn measure as `wraps` —
and reversing the path when it runs backwards. Relabelling the ends alone is not
enough and the intermediate measurement said so: with only the ends pinned the
chords read correctly at both ends while the middle still ran the other way, and
`subdivide` refused exactly as before. Reversed, both paths are sound:

    path 0: [2.1247 27.2199] -> [8.4079 27.2199] u [2.1247 8.4079] max step 0.1496
    path 1: [2.1247 32.7801] -> [8.4079 32.7801] u [2.1247 8.4079] max step 0.1495

Landed — green on the tree as it stands, where it is a latent correctness fix
rather than a visible one.

It does not open the case, and the next thing is now exact rather than suspected.
The outline carries **one** point on that seam edge, `v = 60`, and the chords
meet the edge at 27.22 and 32.78 where there is no vertex to meet. The walk has
nothing to attach them to. `parameter_outline` has to carry the points where a
chord lands on a seam edge — and they need vertices, or invariant B goes. That
note is left at the call site.

Also measured on the way: with the step fix, the bored ball's *Difference* cut
resolves where it declined under the dedup fix alone. The step change costs three
tests and is worth what it fixes; it is one outline away.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## An outline that says where it is met, and the step fix down to one test

Built what the last entry pointed at: the seam edges of `parameter_outline` are
laid by `along`, which samples for *curvature*, and an edge at constant `u` on a
cylinder is straight — so it comes back as a single point and a chord meeting it
partway has nothing to attach to. Breaking the edge at those places is a few
lines.

Given to every face it costs four tests: 48 open edges on a bore across a rod,
and eleven of twenty-two ratios down to three. All declines, no wrong answers,
but lost. The reason is the one written down last entry — those points have no
edge behind them, and a seam is held by edges.

So it is a **second attempt** rather than a first. A face that splits without the
extra points keeps exactly the split it had; a face that does not was going to be
declined regardless. It can add a result and cannot take one away, which is a
shape worth remembering for anything this delicate. Landed, green.

Then the step fix on top, and the accounting has changed completely. It was three
tests. It is now one:

    a_result_can_be_cut_again          the bored ball RESOLVES — the assertion
                                       that it declines is what fails
    the_same_cut_asked_for_five_ways   4 of 5 tolerances, against assert_eq!(3)
    a_rod_can_be_added_to_a_ball...    NotWatertight { open_edges: 206 }

Two of the three are improvements the assertions cannot express. And the first is
the case this file has been circling for four entries:

    bored ball   volume 94.6184
    cut again    volume 92.6569      removed 1.9615

against `∫∫ 2·sqrt(9 − x² − y²) dA` over the drill's footprint, which is **1.997**.
Within 1.8%, on a body whose own tessellation runs 0.17% light. That cut has
never been right before — it removed 0.0138 when it resolved, and declined once
that was found.

So the step fix costs exactly one test now, and that test is the whole remaining
blocker: adding a rod to a bored ball, 206 open edges. Not landed until it is
zero, but it is one case rather than a class.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The last blocker, end to end: a ring with no ear

Traced the one remaining cost of the step fix — `a_rod_can_be_added_to_a_ball_
that_has_already_been_bored`, 206 open edges — to the bottom. Every link
measured:

    faces        7 both ways, the same seven, same bands
    baseline     face 1 cylinder [148, 43, 43]   face 3 cylinder [88]
    step fix     face 1 cylinder [146, 68, 69]   face 3 cylinder [139]
    defects      0  — the edge topology is sound; `is_valid_solid` would pass
    gate         boundary 206, unfilled 0  — every face triangulated
    earclip      gave up: 18 of 139 points left, clipped false, guard 122
    that ring    self-crossings 8

So: the finer trace makes the rod's band on the bore wall cross itself in eight
places, a polygon that crosses itself has no ear anywhere, the clip stops with 18
points unspent, and the face's fill never reaches its own boundary. The topology
is right the whole way — which is why `defects` says nothing and only the
tessellation gate catches it. The baseline's coarser 88-point ring does not cross
and does not give up.

That is invariant A, arriving from a new direction. And the repair for it exists:
`planar::regions_of` turns a crossing ring into simple regions and has since it
was built. What has always blocked wiring it is the same thing that blocks the
outline fix from the last entry: **a crossing is a new point, and a new point
needs a vertex both sides can name.** The outline wants a vertex where a chord
meets a seam edge; the fill wants one where a ring crosses itself. Two
independent paths, one missing capability.

That is the thing to build next, and it is now specified by two callers rather
than argued for by one.

Also: reached for `git checkout src/brep/body.rs` to drop instrumentation. It did
not run, and it would have thrown away `TrimLoop::wraps` and every other landed,
uncommitted change in that file. Nothing in this tree is committed. Remove
instrumentation by removing it.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## What the fill actually leaves, and a metric that lied

Counted `earclip` giving up across the green boolean suite: **eight times**, in
tests that pass. Measured the leftover polygon's area against the ring's and got
84%, 61%, 32%, 23% — and nearly recorded that a face in a passing test is filled
to a sixth of itself.

That metric is wrong. The ring handed to `earclip` is *bridged*: holes are joined
into the outer ring by bridges traversed twice, so the leftover's shoelace area
is not uncovered area. The honest measure is the area of the triangles emitted
against `signed_area(points)`, and by that measure six of the eight fill to
100.0% within a rounding error. Two do not:

    ring 1057:  filled 2.0995e1 of 3.1082e1  (67.5%),  116 points left
    ring  129:  filled 1.5953e0 of 1.0209e1  (15.6%),   61 points left

So there is something real here, and it is two cases rather than eight, and it
would not have been found by the measure that made it look like eight.

Also confirmed while reading: `clipped` resets each pass, so `!clipped` is a full
scan finding no ear — a genuine give-up, not a loop-guard artefact.

Work paused here at the user's request and written down in `TODO.md`: the vertex
that both sides can name, `regions_of` in the fill, the step sized to its curve,
these two fills, and the argument-order asymmetry — in that order, because the
first unblocks the second and third.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## A seam vertex that was not on the seam: 11 of 22 becomes 18

Item 1 of `TODO.md`, and the answer was not the one written there.

The theory was that the outline's seam-edge break needs a vertex nobody mints.
Tested it: made `insert_seam_vertex` return the `v` it cut at, collected those per
face, and handed them to `parameter_outline` on the *first* attempt. Still four
tests. So the point is not missing a vertex — the vertex is right there.

It is in the wrong place. `insert_seam_vertex` refines its point onto the surface
across the curve by bracketing a root along the seam, and where that bracket
fails the raw chord point stands. Measured on a rod bored across:

    seam vertex at v 4.001203: settled vs rebuilt 2.637e-3

2.637e-3, against a tolerance of 1e-3. Everything else that names the seam is
built from `surface.point(origin, v)`, so nothing could ever land on that vertex —
it is a seam vertex that is not on the seam. The fallback now projects onto this
surface's own seam, which the vertex is on by definition and which the comment
above it already said.

Gap goes to zero, and the case that gave `NotWatertight { open_edges: 48 }`
resolves. With the outline break landed as a *first* attempt on top:

    ratios resolved   11 of 22  ->  18 of 22
    what is left      open edges 4, 6, 10 — where it was 150 to 190

Every one of the eighteen is right by volume; that test checks them all and its
floor is now 18. The argument-order asymmetry is mostly gone with it: thin-first
resolves at ten of eleven ratios, thick-first at eight.

The step fix (item 3) is still not landable, and its bill has changed rather than
shrunk: 18 ratios down to 14, plus the self-crossing ring at 207 open edges
(item 2), against the bored ball resolving. The ratio loss is new — before this
entry the step fix left the count alone at 11 either way. Item 2 first.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Four open edges: one corner the outline does not put on the far seam edge

Went at the smallest of the four remaining ratio refusals — 0.85 thick-first,
four open edges — and it took two wrong turns to get to one line.

**Wrong turn one.** Printed each face's seam runs and read:

    face 0 ring 0: seam at u 0.0002 v ["0.0000", "8.0000"] | u 6.2834 v ["8.0000"]

and concluded `parameter_outline`'s dedup was folding the two sides of the seam
together, since on a cylinder `(u0, v)` and `(u1, v)` are the same point in space
and cutting the seam open is the act of needing both. Made the dedup refuse pairs
a period apart. Suite green — **and the outcome did not move**: still 18 of 22,
still four open edges, and the seam runs printed exactly as before. A change that
changes nothing measurable is not a fix, so it is out.

**Wrong turn two.** Printed the whole ring and saw a clean rectangle with both
right-hand corners, which contradicted the run above. The contradiction was my
own filter: the two right-hand corners differ in `u` by more than the `1e-9` I
was filtering on. At `1e-6` the ring says what it meant to:

    face 0 ring 0: at u 0.0002 [(v0.000, vertex 100), (v8.000, vertex 116)]
                 | at u 6.2834 [(v8.000, vertex 116)]
    face 3 ring 0: at u 3.1996 [(v6.999, 91), (v5.001, 90)]
                 | at u 9.4827 [(v5.001, 90), (v6.999, 91)]

Face 3 is right, and instructively so: both sides name the *same two vertices*,
90 and 91, which is what a seam looks like when it works. Face 0 names vertex 116
twice and vertex 100 once.

And it is there before anything splits. `parameter_outline` returns it:

    outline 34 pts: at u0 ["0.000", "8.000"] | at u1 ["8.000"]
    outline 40 pts: at u0 ["0.000","12.000","6.999","5.001"] | at u1 [the same four]

The second is the tool's wall, balanced, with the seam cuts this session added.
The first is the rod's wall, short one corner. The rim's own closing push is not
the cause — measured, its first point is off the seam by 0.000e0, so the push
fires. The corner is lost between that push and the ring being returned, in
thirty lines of one function.

Not fixed. Localised to `parameter_outline`'s tail, which is a much smaller place
than "the argument-order asymmetry".

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Stage A: the outline says which vertex each of its points is

The rework's first stage, and the diagnosis that motivated it was measurable:
`parameter_outline` was the *only* producer of a boundary without vertex
identity — `vertices: vec![usize::MAX; n]` — and 604 lines across eight passes
exist to put back what that drops, with 37 sites recovering identity by
comparing positions. The tell was in the type: `Carried = (u8, usize, Vec<V3>,
bool)` hands the outline an **edge index** and the outline uses only the points.

Four changes, each green:

1. `carried_vertices`, built once beside `carried_points`, naming every rim point
   in the result's own numbering and reusing a vertex a shared curve already put
   there. On a grid — the first cut was a scan of every rim point against every
   vertex, quadratic, and the chained-CSG suite went 7s to 51s. On a grid it is
   3s, faster than before the change.
2. `Carried` widened to carry those names, and `rim_at` keeps them through every
   transform it makes: the closed-rim pop, the rotation, the reverse, and the
   closing push — where the far corner now *takes the near corner's vertex*,
   which is true by construction rather than by a weld that could fail.
3. The outline assembled as points-with-names, its dedup preferring the named of
   a welded pair.
4. `insert_seam_vertex` returns which vertex it made, not only where, so a seam
   cut in the outline names the vertex the chord will land on.

Measured before step 4: outlines went from **0 named** to `34/34`, `36/36`, and
`36/40` — the stragglers being exactly the seam stops step 4 then names.
`vertices: vec![usize::MAX; n]` now appears **nowhere** in `src/brep`.

Two honest limits. The outcome has not moved — still 18 of 22 ratios, the same
four refusals — so this bought structure, not results. And the repairs are not
yet redundant: commenting out `close_seam_runs` costs two tests, so the 604 lines
have not started shrinking. Stage A is a precondition for B and C, and it should
be judged when those land, not now.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Stage B: the seam drawn once, and the repair it does not remove

Went at `close_seam_runs` — 85 lines the rework predicted would go — by first
asking what it still repairs. The census across the boolean suite:

    1 on a sphere at v -1.2310
   23 on a sphere at v -1.5708
   23 on a sphere at v  1.5708

47 repairs, every one on a sphere, 46 of them at `v = ±pi/2`: **the poles**. Not
a general seam problem at all.

Looking at the outlines those faces get: a whole sphere is 128 drawn points with
*none* named — no carried rims to name them from — and the pole sides collapse to
one point each because their sag is zero. So 126 of the 128 are the two seam
meridians, which are the same curve drawn twice. Measured, they never agreed:

    u0 has 64, u1 has 64, mirror-equal false        (x29)
    u0 has 128, u1 has 129, mirror-equal false

And the reason is structural rather than numerical. `along` emits `n` points and
leaves out its endpoint, so the run from `v0` to `v1` holds `v0` and the run back
holds `v1` — the two are offset by a whole sample, always. That offset is exactly
what `close_seam_runs` describes itself as fixing: "short by one point of the
run's own spacing".

Fixed by drawing the seam once and slicing it: one side keeps every point but its
far corner, the other is the same list reversed keeping every point but *its* far
corner. Verified against its own claim —

    cylinder seam: u0 4 pts, u1 4 pts, 4 values on both      (x27)
    sphere   seam: u0 128 pts, u1 129 pts, 128 values on both

— every value on one side is now on the other, where before none matched. The
count difference is the corner each side owns, which is right.

**And the repair count did not move: 47 before, 47 after.** So the mismatch
`close_seam_runs` fixes does not come from the outline's two runs, and removing
it still costs two tests. What the census does say is where it *does* come from,
which no previous entry knew: the pole, on a sphere, forty-six times out of
forty-seven. A pole is one point that a whole edge of parameter space maps to,
and that is a different repair from a seam.

Kept, because it does what it claims and the sides agreeing by construction is a
precondition for ever removing the repair. Not counted as progress on the 604
lines, which stand.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## What the seam repair still does, and why it cannot be narrowed

Printed the rings `close_seam_runs` repairs, in full. They say Stage A and B
worked and that the repair's job has moved:

    lo: -1.5708/158, -1.5217/159, -1.4726/160, ... 1.5217/221
    hi:              -1.5217/159, -1.4726/160, ... 1.5217/221, 1.5708/222

The two sides of the seam **already share every interior vertex** — 159 through
221, the same numbers on both — which is what drawing the seam once was for. What
differs is one point at each end: `lo` owns the south pole and `hi` owns the
north, each with its own name for a place that has only one.

So the pass is now doing pole work. Tried narrowing it to exactly that — repair
only where the surface runs to a point — on the strength of 46 of its 47 firings
being a sphere at `v = ±pi/2`.

**It costs a result.** `the_same_cut_asked_for_five_ways` goes from three
tolerances resolving to two. So the one non-pole repair in forty-seven is
load-bearing, and the general case earns its place. Reverted.

Nothing landed this entry. What it establishes is worth the runs: the seam's
interior is now held by identity rather than by rounding, and the 85 lines that
remain are not redundant work but a real repair for a case the outline still
cannot state — a pole, where one vertex is the whole of one edge of parameter
space, and neither side of a seam can own it alone.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The pole, stated properly — 47 seam repairs become 1

A sphere's parameter rectangle is not a rectangle on the surface. The `v = ±pi/2`
edges each collapse to a point, so the face is a lune: two meridians meeting at
two poles. Both meridians are the *same* curve, since `u0` and `u1` differ by a
period, so the boundary runs one edge twice in opposite directions and the poles
are its two ends.

`close_seam_runs` was compensating for a parameterisation artifact. A pole is one
place with an arbitrary `u`, so filtering the ring by `u` counts it on one side
only; the pass gave each side its own copy, `(u0, pi/2)` and `(u1, pi/2)` — same
point, different parameters — which is exactly right.

What stopped the outline doing that itself was the position dedup folding the
pair. Two changes:

* a `v` edge that is a *point* is not a side the ring travels along but the
  corner where both seam runs meet, so neither run drops it;
* the dedup never folds two parameters a period apart, because they are one place
  with two names and that is what cutting a closed surface open means.

That second one is the change I made two entries ago and reverted for changing
nothing measurable. It was the right change measured against the wrong thing: I
checked cylinders and seam prints, and never checked whether it retired the
repair.

    repairs across the boolean suite   47  ->  1
    without close_seam_runs at all     2 tests failing  ->  1

`a_seam_is_reached_from_both_sides_at_the_same_place` now holds with the pass
switched off entirely. The one repair left is the one non-pole case, a sphere at
`v = -1.2310` where a bore's rim meets the seam, and it is load-bearing: removing
the pass still costs `the_same_cut_asked_for_five_ways` a tolerance, three down
to two.

So 85 lines now exist for a single case, which is the right position to attack it
from. Nothing else moved — 18 of 22 ratios, all six binaries green.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## `close_seam_runs` is gone: 97 lines, and the rule that replaced it

The last repair was the mirror of the pole one. Printed in full:

    lo (u 1.5708): -1.2117/722 ... 1.2310/563
    hi (u 7.8540): -1.2310/387, -1.2117/722 ... 1.2310/563

Both sides share the interior *and* the top corner — vertex 563 on each — while
the bottom corner sits on `hi` alone. One corner shared, one not, which is not a
rule but an accident of which side happened to drop what.

The rule is the one the pole taught, stated generally: **a corner of the
parameter rectangle is one place with two parameters.** `(u0, v)` and `(u1, v)`
are the same point on any closed surface, so neither seam side owns a corner and
neither may drop one. Applied — both sides keep both corners, the degeneracy test
no longer needed — and the pass came off entirely:

    repairs across the boolean suite   47 -> 1 -> 0
    close_seam_runs                    97 lines deleted
    repair machinery                   604 -> 519 lines

All six binaries green with it gone, including the two tests that have blocked
its removal every time it was tried: `a_seam_is_reached_from_both_sides_at_the_
same_place` and `the_same_cut_asked_for_five_ways`, which still resolves three of
its five tolerances.

`materialise_seam_steps` was tried next and stays: without it
`every_step_of_a_ring_is_a_step_of_an_edge` fails, and rightly — its job is
*edges* behind the steps, not agreement between points, which is a different
thing that the outline cannot supply.

This is the first shrink the rework promised, and it took the whole chain to get
here: name the outline's points (Stage A), draw the seam once (Stage B), then see
that a corner is a place and not a side.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The turn measure, on the baseline the rework built: 18 of 22 becomes 21

Cleared the small items and re-measured the ones whose verdicts predated this
run's work. Two landed, and one of them was sitting in `TODO` marked "buys
nothing".

**`CoincidentFaces` is gone.** Never constructed anywhere in `src`, and it cannot
be: two bodies sharing a surface are *handled* now — `SsiResult::Coincident`
sends the pair to the keep/drop rules — so the variant named a refusal the kernel
no longer makes. It had a case; the case was solved.

**The turn measure is in.** `split_face` asked whether a ring goes all the way
round by the *span* of its parameters; the right question is how far it *turns*,
which `TrimLoop::wraps` has computed since it was added. Swapping them was tried
twice before and cost a test for nothing measurable. On the baseline this run
built — named outlines, one-list seams, corners as places — it costs nothing and
buys:

    ratios resolved     18 of 22  ->  21 of 22
    bored ball drilled off-axis: removed 1.9601 against an exact 1.997

That second one has been wrong or refused for the whole of this record. It
removed 0.0138 when it first resolved, then declined once the trace was fixed,
and now answers correctly. Its test asserts the *weight*, because weight is what
was wrong with it.

**Items 2 and 4 are one item, and it did not fall.** The under-filled fills are
crossing rings, and the crossings are always in the *holes*:

    ring 1057 filled 67.5%: crossings outer 0, holes [5, 5], bridged 14
    ring  738 filled 77.1%: crossings outer 0, holes [1, 2], bridged  9

Two repairs tried and both reverted on measurement. Making a crossing hole simple
in the fill costs four tests — the hole is shared, and a face that changes it
describes a rim its neighbour describes differently, which is the T-junction
`every_step_of_a_ring_is_a_step_of_an_edge` catches. Inserting the crossing into
the *curve* instead, the way a seam crossing is inserted, costs four or five: a
vertex placed at both passes makes the ring pinch rather than simplify, and at
one pass it duplicates a point. That is the seventh attempt at this repair in
this file.

**The step fix, re-measured:** 21 ratios down to 17, plus the crossing case. Its
bill has moved three times as the baseline moved, and it is still not payable.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The crossing is a lap of exactly two, and trimming it does not pay — yet

Eighth attempt at the crossing repair, and the first to say what the crossing
actually *is*.

Started from the premise written in `TODO`: the curve does not cross itself, only
its polyline does, so refine the chords until it stops. Built that — halve both
offending chords, settle each midpoint onto both surfaces, repeat — and measured
before believing it:

    368 refinements, and the same curve growing 26, 27, 28 ... 35 without ever
    resolving

Refinement never converges, so the crossing is not a sampling artifact and the
premise was **wrong**. Printing where it is:

    cylinder curve 42: segs 0,40 apart in space 1.0507e-1
    cylinder curve 46: segs 0,44 apart in space 6.9578e-3
    cylinder curve 52: segs 0,48 apart in space 2.8672e-3

Always the *first* segment against one of the *last*, with the two ends
approaching and never meeting. That is a closed trace whose tail runs past its
own head — a **lap**, not a fold, and no amount of chord-halving removes an
overshoot.

Trimming it instead — a crossing cuts a closed curve in two, keep the longer —
and the laps are startlingly uniform:

    every lap is exactly 2 points, on 14 curves of 42, 14 of 40, and a torus of 56

The trade is not payable, but it is close: ratios 21 of 22 down to **19**, and
`the_same_cut_asked_for_five_ways` up from three tolerances to **four**. Losing
two results to gain one is still losing, so it is reverted.

What is worth keeping is the characterisation, which eight attempts have been
missing: the crossing is a *two-point overshoot at the close of a traced ring*,
not a figure-eight and not coarse sampling. That points at `trace` rather than at
any repair — a marcher that stopped two samples late, or a closure test that
accepts a point already passed.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Two crossing sources, not one — and a correction

Followed the lead the last entry left: close a traced ring where it came *nearest*
its start rather than where the walk happened to stop. Built it, and the whole
suite stayed green — then measured what it bought, with the detector counting
crossings in the rings the fill actually sees:

    with the closure fix   ring 1057: filled 67.5%, crossings 14
    without it             ring 1057: filled 67.5%, crossings 14

Identical. Reverted, on the same rule that took out the seam dedup two entries
ago: a change that changes nothing measurable is not a fix, however good the
reasoning behind it.

And the comparison corrects the last entry. The laps it characterised — one per
curve, exactly two points, on cylinder curves of 26 to 42 — are **not** the
crossings that break the fill. Ring 1057's holes carry *five each*, on rings an
order of magnitude larger. Two different sources wearing the same symptom, and
the entry that said "the crossing is a two-point overshoot" was describing the
one that does not matter.

So item 1 is not one defect but two, and only one of them is characterised. What
is measured and holds:

* a closed trace can lap its own head by two points, uniformly — real, and
  fixable by trimming, which costs two ratio results and gains one;
* the rings that leave a face two-thirds filled cross themselves five times, in
  the *holes*, and nothing tried so far touches them.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## A ring that runs its own loop twice, and the rule for repairing a shared ring

Characterised the crossings the fill actually chokes on, as the last entry asked.
There are two kinds and they want opposite treatment.

**A doubled ring.** One hole of 142 points where every one of the first 71 is the
same *place* as the one 71 later:

    face 1 hole 0: 142 pts, 71/71 doubled, area 6.275e0, is_hole true
    ... 53-124(gap 0.0e0) 54-125(gap 0.0e0) ... 70-141(gap 0.0e0)

Both passes go the same way round, so the areas add instead of cancelling and it
reads as a perfectly good hole of twice the size — while crossing itself 124
times. Halved, and the 124-crossing ring is gone from the census. Landed.

**A lap.** What is left is the head against the tail — `0-299, 0-300, 1-300,
1-301, 299-301` on a ring of 302 — two or three points deep, the same overshoot
the previous entry found in traced curves. Cutting at the crossing and keeping the
longer loop costs four tests. Not landed.

**And the difference between them is the rule.** Both repairs are decided by the
ring's own points, so two faces sharing a ring make the same decision — that is
not what separates them. Halving removes *redundant* points: every step of the
ring survives, because the second pass repeats the first, so a neighbour matching
those steps still finds them all. Trimming a lap removes **real** steps, and the
neighbour still has them. That is what `every_step_of_a_ring_is_a_step_of_an_edge`
catches, and it is the reason nine attempts at this repair have failed: a shared
ring may be *deduplicated* but not *shortened*.

Honest limit: the halving fixes no test and moves no fill — 1057 stays at 67.5%,
129 at 15.6%, 738 at 77.1%. It is kept because it demonstrably does what it claims
and removes a ring that crosses itself 124 times, on the same footing as the seam
drawn once.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The lap, fixed where it is made: 22 of 22, and the last blocker is one edge

`trace` pushes a point and *then* asks whether it has come back to the seed:

    p = next;
    forward.push(p);
    if forward.len() > 3 && v3::dist(p, seed) <= step * 0.75 { forward.push(seed); ... }

So a walk that comes back *past* where it began keeps the points it overshot by,
and the ring laps its own head. That is the lap nine repairs have been aimed at,
and it is two lines from where it is made. The seed lying behind the direction of
travel is what "past" means, so points are dropped while that holds.

What it bought, and none of it was assumed:

    ratio sweep            21 of 22  ->  22 of 22
    five-tolerance sweep    3 of 5   ->   4 of 5
    under-filled rings      3        ->   1   (1057 at 67.5% and 738 at 77.1%, gone)

All twenty-two ratios now resolve and every one is right by volume; that test
asserts equality with 22 rather than a floor. The argument-order asymmetry, open
since it was first recorded, is closed.

**And the step fix is one open edge away.** Re-measured on this baseline it costs
a single test, `a_rod_can_be_added_to_a_ball_that_has_already_been_bored`, with
`NotWatertight { open_edges: 1 }` — where it was 207 two entries ago and 5 tests
before that. Diagnosed as far as: `defects` is empty, every ring step is matched
by another face, and the edge is there before `refine_edges` as well as after. So
it is the triangulation, not the model — and the fill drops any triangle naming a
vertex twice, which leaves exactly one boundary edge behind. That is where to
look next.

Also landed this entry: a hole that runs its own loop twice is halved. It fixes
no test and moves no fill, but it removes a ring that crossed itself 124 times,
and the rule it established is what made the trace fix findable — **a shared ring
may be deduplicated but not shortened**, because deduplication leaves every step
in place and shortening does not.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## The last open edge is not the ring, and the chord test encodes the old step

Chased the one edge the step fix still costs. The fill drops any triangle naming
a vertex twice, and the failing case makes plenty:

    face 2: dropped triangle [100 101 100] of ring 132
    face 2: dropped triangle [101 102 101] of ring 132

Several ring entries welding to one vertex, so removed the repeats — which is the
*allowed* operation on a shared ring, since a step from a vertex to itself is not
a step and every real step survives. Green on its own terms, and it changed
nothing: **14889 dropped triangles before and after**, the edge still there.

They do not come from the ring. `refine_on_surface` runs after the clip and mints
its own vertices, and the degenerate triangles are its. So the repeats in the
ring were never the cause, and the dedup is reverted for buying nothing.

The other cost is a test rather than a defect. `a_traced_polyline_and_the_curve_
through_it_agree_at_every_sample` asserts `worst_chord > 1e-3` — that a polyline
leaves the surfaces by more than tolerance, so the parameterised curve can be
shown to beat it. A step of `sqrt(8·r·tol)` puts the sagitta at *exactly*
tolerance by construction, and it measures 0.00099. The assertion encodes the old
coarse step; the test's real claim — the curve beats the chord — would hold if it
compared the two rather than comparing one to a constant.

So the step fix's bill is now: one test that should be rewritten, and one open
edge that should not. The edge is in the triangulation, after the clip, in
`refine_on_surface` — which is a smaller place than it was two entries ago, and
not the place anything has been looking.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.

## Done: the fan chord, the step, and the last of the list

The one open edge, located: from `[-2.8914 -0.5998 -0.5293]` to
`[-2.8914 0.3107 0.7372]`, length **1.56**. Both endpoints satisfy
`y² + z² = 0.64` and `x² + y² + z² = 9` — both on the rod's wall *and* on the
sphere, so both on the rim where the rod leaves the ball. A chord spanning 112
degrees of a circle of radius 0.8.

That is the fan chord `refine_on_surface` has described for a long time and
refuses to split, ending with: "The chord itself has to go, which means not
creating it." So it is refused at the clip instead. Ear clipping (a, b, c) draws
the edge a-c, and where the ring is a rim — one `v` on a cylinder, one latitude
on a sphere — that chord lies *on* the boundary in parameter space however long
it is, while the face beyond the rim has no such edge. An ear whose base runs
along the ring between two vertices that are not neighbours on it is not an ear.

With that, **the trace step sized to its curve lands.** Its remaining two costs
were both assertions, not defects:

* `the_same_cut_asked_for_five_ways` — **5 of 5** tolerances, where it asserted 4.
* the chord test asserted `worst_chord > 1e-3`, true only while the trace stepped
  by a hundredth of its bounding box. `sqrt(8·r·tol)` puts a chord's sagitta at
  the tolerance *by construction*, so it measured 0.00099 and failed for the
  sampling being right. Rewritten as what it always meant: the curve hugs the
  surfaces twenty times closer than its chords.

Where the whole list stands, measured:

    ratio sweep         22 of 22, every one right by volume
    tolerance sweep      5 of 5
    under-filled rings   0   (was 3)
    earclip give-ups     2, both covering their rings completely

And the last item with them: a fill that does not cover its face now says so and
is `carried_through`, instead of coming back short and leaving the watertight gate
to count edges it cannot explain. Judged by *area* — the ring is bridged, so a
leftover of sixty points can enclose nothing, and both remaining give-ups do.

Green: 45 boolean, 27 brep, 13 brep_csg, 18 nurbs, 12 step, 730 lib.
