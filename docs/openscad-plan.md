# threers openscad plan

Design doc and roadmap for an opt-in, **pure-Rust** OpenSCAD-style solid-modeling
front end backed by a **robust exact boolean kernel** (`wgpu` compute where it
helps). Feature flag: `openscad` (implies `bvh-csg`, which implies `mesh-bvh`).

No C/C++ ships in the crate. CGAL/OpenSCAD are used only as an **offline test
oracle** (fixtures), never linked or bundled.

---

## Goal & parity definition

"Parity with OpenSCAD's CGAL (F6) render" splits into two very different bars.
We commit to one and explicitly disclaim the other.

| Meaning | Definition | Verdict |
|---------|------------|---------|
| **(1) Solid parity** | Result is watertight + 2-manifold; its boundary is the same point set as CGAL's within ε; robust on every degenerate case (coplanar faces, edge-on-edge, shared vertices) | **Committed.** Pure-Rust feasible. |
| **(2) Mesh parity** | Identical vertex list, triangle list, and ordering (à la our three-bvh-csg `TriKey` parity) | **Rejected.** Requires porting CGAL `Nef_polyhedron_3` internals + OpenSCAD's tessellator — the C++ we are avoiding. Two independent *correct* exact kernels still disagree on triangulation/order. |

**Why (2) is out of scope.** Our `TriKey` parity against three-bvh-csg works
because we port *that specific JS algorithm* deterministically. CGAL's output
triangulation is an artifact of its sphere-map/half-edge traversal plus
OpenSCAD's facet tessellation. Matching the triangle list means reimplementing
those, not "an exact CSG." Parity is therefore defined at the **solid** level and
measured (below), not at the mesh level.

**Middle rung.** Vertices *forced* by geometry (intersection of two input planes)
are mathematically unique. With exact rational constructions rounded identically,
those coordinates match CGAL to the ULP — "same vertices, faces differ only by
triangulation." Available via the rational fork; see [Construction fork](#construction-fork).

---

## Non-goals (v1)

- **Mesh-level bit parity** with CGAL (see above).
- ~~**`.scad` language parsing.**~~ **NOW SHIPPED** (`src/openscad/scad.rs`,
  `parse_scad`) — a tokenizer → recursive-descent parser → evaluator for the core
  OpenSCAD language: 3D primitives, transforms (`translate`/`rotate`-in-degrees/
  `scale`/`mirror`), booleans, `for`/`if`/`let`, `module`/`function` defs, named +
  default args, `$fn`, and the expression language. **The CGAL harness now runs
  `parse_scad` on the real `.scad` corpus and compares to OpenSCAD rendering the
  same source** — `stacked_boxes`/`box_notch`/`box_slot`/`plate_2d` pass
  end-to-end (parse → evaluate → kernel → STL, Hausdorff ≤2e-5); the 3D-difference
  `plate_with_hole` fails only on the kernel's `$fn=32` degeneracy (parser output
  is correct — Hausdorff 5e-6). Full `children()` still out.
- ~~**2D subsystem** (`square`/`circle`/`polygon`) beyond what feeds
  `linear_extrude`/`rotate_extrude`.~~ **NOW SHIPPED.** `square`/`circle`/`polygon`
  (with `paths` holes) evaluate to a `Shape` (outer ring + holes); 2D `difference`
  turns subtracted outlines into holes. **Extrudes build closed manifolds
  directly** — an earcut-style **hole-bridging cap triangulator** (`triangulate_holes`)
  triangulates the outer-minus-holes cap, and `linear_extrude`/`rotate_extrude`
  stitch caps + walls into one watertight `polyhedron` (no float-CSG differencing,
  which stack-overflowed on coaxial prisms). **Consequence:** a round-hole-in-plate
  expressed via the 2D path (`plate_2d`: `linear_extrude difference(){square;circle;}`)
  reaches **CGAL parity** (watertight, vol err 0, Euler match, Hausdorff 2e-5) — the
  same shape the 3D-difference kernel still can't close. `offset`/`text` still out.
- ~~**`hull()`**~~ **NOW SHIPPED** (2D: Andrew's monotone chain; 3D: incremental
  convex hull → outward-wound `polyhedron`; dedups coincident mesh-soup inputs).
- ~~**`minkowski()`**~~ **NOW SHIPPED (convex)** — the convex Minkowski sum =
  convex hull of pairwise vertex sums, 2D + 3D. Exact for convex operands (box⊕box
  and box⊕sphere reach CGAL parity in volume/Euler; the round-with-sphere case
  matches to ~1%, the residue being our sphere's `$fn` faceting vs OpenSCAD's, not
  the sum). Non-convex Minkowski still needs a general kernel — deferred.
- ~~**`offset`/`resize`/`projection`**~~ **NOW SHIPPED.** `resize` fits the child's
  bounding box (with `auto`); `offset` grows/shrinks 2D contours (round `r`, mitred
  or chamfered `delta`); `projection(cut=true)` slices the mesh at z=0 and chains
  the segments into loops. All reach CGAL parity end-to-end (`resize_box`,
  `rounded_2d`, `proj_cut` corpus models).
- ~~**2D polygon booleans**~~ **NOW SHIPPED** — a real **arrangement kernel**
  (`arrange_extract`): every edge is split at all crossings, each sub-edge is
  classified by testing a point ε to either side (even-odd for booleans, nonzero
  winding for self-cleaning), boundary sub-edges are kept and traced into loops,
  and holes are nested into outers. Handles non-convex shapes, holes, coincident/
  collinear edges, and self-intersection. This powers correct 2D `union`/
  `difference`/`intersection` (replacing the old hole-stuffing / Sutherland–Hodgman
  approximations), self-cleaning `offset`, and `projection(cut=false)` (silhouette
  = union of projected triangles). The `booleans_2d` corpus model (a plus-with-hole)
  reaches CGAL parity. Ear-clipping was hardened for reflex corners on ear edges so
  concave caps stay watertight.
- ~~**`$fa`/`$fs` resolution**~~ **NOW SHIPPED** — circle/sphere/cylinder fragment
  counts follow OpenSCAD's `ceil(max(min(360/$fa, 2πr/$fs), 5))` when `$fn` is unset,
  so default-resolution curves match OpenSCAD's tessellation (volume + topology;
  vertex phase on curves is still a mesh-level non-goal).
- ~~**`multmatrix`, `import`, `include`/`use`**~~ **NOW SHIPPED.** `multmatrix`
  applies an arbitrary affine (row-major → column-major) matrix (`sheared` model at
  CGAL parity); `import("…stl")` loads a mesh via `StlLoader`; `include <…>` inlines
  a file textually and `use <…>` imports only its definitions, both resolved by
  `parse_scad_file` relative to the file's folder.
- ~~**tessellation phase / `text` / `surface` / non-convex `minkowski` / non-STL
  `import`**~~ **NOW SHIPPED.**
  - **Tessellation** — `circle`/`sphere`/`cylinder` are built with OpenSCAD's exact
    vertex placement (fragment count *and* phase, pole caps for spheres), so
    default-resolution curves reach vertex parity (`default_cyl` corpus model,
    Hausdorff 4e-6 vs the previous 0.058 half-facet offset).
  - **`text`** — glyph outlines from a bundled outline font (**DejaVu Sans**,
    Bitstream Vera/Arev license) or a supplied `font="…ttf"`, split into contours
    and filled by non-zero winding through the arrangement kernel, then extruded.
    Smooth, watertight glyphs that render from any angle (verified boundary=0,
    non-manifold=0); won't match OpenSCAD's Liberation Sans unless you pass its path.
  - **`surface`** — `.dat` heightmap → solid with the base one unit below the data
    minimum (matching OpenSCAD). `surf_dat` corpus model at Hausdorff 0.
  - **non-convex `minkowski` (2D)** — triangulate both shapes, convex-sum each pair,
    union via the arrangement kernel. Exact for non-convex operands (`mink2d` corpus
    model, an L⊕circle, at Hausdorff 6e-5).
  - **`import`** — STL/OBJ/OFF → solid; **DXF/SVG → 2D shape** (DXF `LWPOLYLINE`/
    `LINE`/`CIRCLE`; SVG `rect`/`circle`/`polygon`/`path` with bezier flattening).
- ~~**first-class functions / deep recursion / `assign` / DXF-SVG / non-convex-3D
  minkowski**~~ **NOW DONE.**
  - **First-class functions** — `function(x) …` literals as values, closures
    (capture the defining scope), higher-order calls, functions in lists, and
    postfix calls (`fs[1](3)`). `is_function` reports them.
  - **Recursion** — evaluation runs on a 1 GB-stack worker thread and a recursion
    guard (OpenSCAD's 20 000 limit) turns runaway recursion into a clean error;
    deep-but-finite recursion (thousands deep) no longer overflows the stack.
  - **`assign()`** — the deprecated synonym for `let()`.
  - **non-convex 3D `minkowski`** — detected (a vertex off the convex hull) and
    **rejected with a clear error** rather than silently returning the convex hull.
- **Crash-safety** — the float CSG fallback is guarded three ways: a depth cap in
  the dual-BVH traversal (a runaway-recursion bug), a `catch_unwind` per boolean
  degrading to the un-cut mesh, and a `catch_unwind` worker-thread backstop. So a
  curved 3D boolean that the exact kernel can't close degrades gracefully instead
  of aborting the process. A weld + T-junction repair is applied to the fallback
  and accepted only when it actually closes the mesh.
- **Exact curved-3D CSG — done for the common cases.** `cube − cylinder`
  (`plate_with_hole`) now reaches **CGAL parity**: watertight, `Δvol = 0.0000 %`
  vs the OpenSCAD/CGAL oracle. Oblique box∧box (all three ops), two crossing
  cylinders (union), and a sphere piercing a box are all watertight through the
  exact kernel. How it was closed, in `corefine` → `refine_with` → `cdt_face`:
  1. **Per-face CDT, not per-triangle.** Co-refined triangles are grouped by their
     oriented plane (one flat *face*), and each face is triangulated **once** with a
     super-triangle, its boundary edges, and all its cut segments as constraints —
     so a cut crossing the face's internal diagonal produces a shared vertex, not a
     T-junction.
  2. **Constraint splitting** at every registered point on a segment (no vertex is
     ever strictly interior to a constraint, which the flip-recovery can't heal),
     plus a **global cut-point set** shared by *both* meshes so a point where one
     mesh's cuts meet also splits the other mesh's seam identically.
  3. **Delaunay (Lawson) flipping** after constraint recovery — the decisive fix for
     finely-tessellated curved seams (a sphere arc): incremental insertion +
     recovery leaves near-collinear slivers, and the incircle flips replace them
     with well-shaped triangles instead of zero-area slivers that would open cracks.
  4. **Cross-face T-junction heal** (work-stack: split any output triangle with a
     registered vertex on its edge), a **relative** degeneracy filter (min-height vs
     longest edge), and a bounded weld+heal `repair` as the final safety net — all
     behind the **never-wrong gate**: emit `Exact` only if the result verifies as a
     closed 2-manifold, else `NeedsArrangement` → crash-safe float fallback.
  Corpus: **14/14 models at CGAL parity** (was 13/14; `plate_with_hole` graduated
  from KNOWN_LIMITATION to MUST_PASS).
  Remaining: some fine sphere∧sphere / large-`$fn` sphere booleans still fall back
  to float (verified, not silently wrong). Lower-priority: image (`.png`) `surface`,
  exact `text` font parity. Everything else in the core language is implemented.
- **No default enablement.** Apps opt in at build time; default wasm stays lean.

---

## Architecture

Two layers: a thin modeling front end that lowers to a boolean tree, and the
exact kernel that evaluates it.

```
Solid tree (src/openscad/)         ← OpenSCAD-style builder API
   │  lower + bake transforms
   ▼
Boolean plan (union/diff/intersect over triangle soups)
   │
   ▼
Exact mesh-arrangement kernel (src/exact_csg/)   ← the hard part
   │  broad phase → arrangement → classify → assemble
   ▼
BufferGeometry                     ← drops into Mesh like any threers geometry
```

### Front end: `Solid` tree → `BufferGeometry`

A builder tree that reads like SCAD and produces a normal `BufferGeometry`.
Primitives reuse existing generators; transforms are **baked into vertices** at
lowering time (no reliance on the parity-tuned CSG world-matrix path).

```rust
pub enum Solid {
    Leaf(BufferGeometry),                 // primitive, baked to identity pose
    Transform { m: Matrix4, child: Box<Solid> },
    Union(Vec<Solid>),
    Difference(Vec<Solid>),               // head − tail − tail …
    Intersection(Vec<Solid>),
}

// cube([x,y,z]) / sphere(r,$fn) / cylinder(h,r1,r2,$fn) / polyhedron(pts,faces)
// linear_extrude(h) polygon → ExtrudeGeometry ; rotate_extrude(a) → LatheGeometry
let part = cube([20.0, 20.0, 4.0])
    .difference(cylinder(10.0, 2.5, 2.5).translate([10.0, 10.0, 0.0]))
    .union(sphere(3.0).translate([0.0, 0.0, 4.0]));
let geom = part.to_geometry(&mut kernel);   // BufferGeometry
```

n-ary boolean semantics map directly onto left-to-right folds
(`union`=fold add, `difference`=head then subtract tail, `intersection`=fold
intersect) — identical to OpenSCAD.

Primitive coverage:

| OpenSCAD | Backing generator | Notes |
|----------|-------------------|-------|
| `cube` | `BoxGeometry` | `center` → baked translate |
| `sphere` | `SphereGeometry` | `$fn` → segments |
| `cylinder` / cone | `CylinderGeometry` | cone = `r2=0` |
| `polyhedron` | `PolyhedronGeometry` | triangulate faces first |
| `linear_extrude` | exact-vertex prism | +holes via `linear_extrude_holes` (outer − taller hole prisms) |
| `rotate_extrude` | `LatheGeometry` | profile = `&[Vector2]`, `angle` |

### Kernel: mesh arrangements + indirect predicates

**Not Nef polyhedra. Not float BSP** (the evanw/three-bvh-csg style in `src/csg/`
fails exactly the degenerate cases we need). The kernel is a **mesh arrangement**:

1. **Broad phase** — BVH×BVH overlap → candidate triangle pairs.
2. **Arrangement** — resolve all triangle-triangle intersections into a
   consistent sub-triangulation of the combined soup.
3. **Classification** — label each output triangle inside/outside each operand.
4. **Assemble** — select triangles per boolean op; emit watertight B-rep.

References:

- Zhou, Grinspun, Zorin, Jacobson — *Mesh Arrangements for Solid Geometry*, SIGGRAPH 2016 (libigl's approach; arrangement + winding-number classification).
- Cherchi, Livesu, Scateni, Attene — *Fast and Robust Mesh Arrangements using Floating-point Arithmetic*, SIGGRAPH Asia 2020.
- Cherchi, Pellacini, Attene, Livesu — *Interactive and Robust Mesh Booleans*, SIGGRAPH Asia 2022.
- Attene — *Indirect Predicates for Geometric Constructions*, CAD 2020.
- Barill et al. — *Fast Winding Numbers for Soups and Clouds*, SIGGRAPH 2018.

**Indirect predicates** are the key to speed *and* pure Rust: intersection points
stay **implicit** as triples of input planes; only orientation predicates are
evaluated over them, exactly, via Shewchuk expansion arithmetic. No
rational-coordinate blow-up, near-float speed, provably total.

### Pure-Rust arithmetic stack

Every exact primitive has a pure-Rust option — no `gmp`/`rug` (those bind C).

| Need | Crate / plan | C-free |
|------|--------------|--------|
| Direct predicates (`orient3d`, `insphere`) | `robust` (Shewchuk, georust) | ✅ |
| Indirect predicates over implicit points | new Rust — expansion arithmetic, port of Attene's generated code | ✅ |
| Arbitrary-precision rationals (rational fork only) | `dashu` or `malachite` | ✅ |
| Fast winding number | new Rust (Barill et al.) | ✅ |

There is **no existing pure-Rust exact mesh-boolean kernel to reuse** (`csgrs` et
al. are float BSP). This is a from-scratch, research-grade build — larger and
riskier than the native-codec effort, which had a fixed spec.

### Construction fork

Two ways to finalize output coordinates. Pick per build/config.

| | Indirect (Cherchi) — **default** | Rational (Zhou/libigl-style) |
|---|---|---|
| Speed | near-float | slow (coordinate blow-up) |
| Robustness | exact / total | exact / total |
| Output coords | snap-rounded → geometric parity within ε | exact rationals → f64 = **CGAL's forced vertices to the ULP** |
| Use when | valid solid, fast | closest-to-CGAL vertices required |

Neither reproduces CGAL's *triangulation* — that stays out of scope regardless.

---

## Parity: the CGAL oracle harness

Since mesh parity is rejected, parity is a **measured solid equivalence** against
reference STLs generated by the real OpenSCAD/CGAL binary. The oracle is a
**fixture generator** (dev-only, `scripts/`), never linked into the crate — the
shipped code stays pure Rust.

Per model in the corpus, gate on:

| Metric | Gate |
|--------|------|
| Watertight + 2-manifold (both meshes) | hard fail if not |
| Volume | agree within ε |
| Symmetric Hausdorff distance | ≈ 0 |
| Euler characteristic / genus | identical (topological identity) |

Corpus: hand-picked `.scad` models emphasizing the degenerate cases float BSP
fails — coplanar unions, shared-face subtractions, edge-on-edge, near-tangent
spheres, deep boolean trees — plus a fuzzer that emits random primitive stacks.
Reference STLs are cached; CI runs the Rust kernel and compares. Lives beside
`scripts/ci-bvh-csg.sh` as `scripts/ci-openscad.sh`.

---

## GPU acceleration (`wgpu`)

We already own a `wgpu` device (`HeadlessRenderer`). Compute kernels accelerate
the parallel float phases; **exactness stays on the CPU.**

**Fits (GPU):**

| Stage | Why |
|-------|-----|
| Broad phase (BVH×BVH → candidate pairs) | parallel, float bounds tests; reuses `mesh_bvh` |
| Float-filtered tri–tri intersection | fast 99% path; degenerate cases fall back to CPU exact |
| Inside/outside via fast winding number | embarrassingly parallel |

**Does not fit (stays CPU):**

- **Exact arithmetic.** Expansion / bignum predicates are a poor GPU fit; the
  exact fallback and the topological arrangement assembly are CPU. GPU makes easy
  cases *fast*; CPU makes hard cases *correct*.
- **SDF / raymarch CSG ≠ parity.** `min`/`max` of signed-distance fields on the
  GPU is great for an **F5-style live preview**, but it is a grid approximation
  and can never yield an exact B-rep. Preview only — never the parity path.

---

## What threers already provides (reuse)

| Asset | Reused for |
|-------|-----------|
| `src/mesh_bvh/` (three-mesh-bvh-compatible, indirect BVH) | broad phase input |
| `src/csg/` attribute plumbing (`attribute_data`, `geometry_prep`) | output assembly |
| `HeadlessRenderer` wgpu device | compute kernels |
| Geometry generators (`box`/`sphere`/`cylinder`/`extrude`/`lathe`/`polyhedron`) | front-end primitives |
| Feature-flag + JS-shim build pattern (`mesh-bvh`, `bvh-csg`) | `openscad` wiring |

Small new shared helper: `bake_matrix(&mut BufferGeometry, &Matrix4)` (transform
positions by `m`, normals by inverse-transpose) — we have `Object3D::apply_matrix4`
but no geometry-level bake today.

---

## Milestones (staged)

| # | Deliverable | Parity reached |
|---|-------------|----------------|
| **M0** ✅ | `Solid` front end + lowering + `bake_matrix`; evaluate via the **existing float `CsgEvaluator`** (`src/openscad/`) | none — proves the API + pipeline shape |
| **M1** 🚧 | f64 mesh-arrangement + winding-number booleans on top of `mesh_bvh` | passes well-conditioned models; fails degenerates |
| **M2** 🚧 | CPU exact predicates (from-scratch, pure Rust) → totality/robustness | **solid parity (meaning 1)** on the degenerate corpus |
| **M3** 🚧 | CGAL oracle harness + 4-metric gate in CI (`scripts/ci-openscad.sh`) | parity *measured & enforced* |
| **M4** 🚧 | BVH broad phase + point-in-mesh (CPU done; GPU next) | same results, faster |
| **M5** (opt) | Rational-construction mode; `hull()`/`minkowski()`; 2D subsystem | ULP vertex closeness; wider op coverage |

A shippable, honestly-parity kernel exists at the end of **M3**. M4–M5 are speed
and coverage.

**M4 progress (`src/exact_csg/bvh.rs`).** A compact median-split AABB BVH (built
once per mesh, verified no-miss vs brute force) now backs the two O(N·M) hot
paths: **broad-phase co-refinement** (`corefine` queries the BVH per triangle →
O(N log M)) and the **exact ray-parity inside test** (`point_inside` casts through
`ray_leaves` candidates, exact `orient3d` only on those). The dispatch's straddle
+ coplanar scan collapsed into one BVH pass (`interacts`). Effect: `sphere ∪
sphere` at 1520 triangles each resolves in **~8 ms** (naive broad phase = 2.3 M
pairs); correctness unchanged (all tests + 3 CGAL-parity models still green). GPU
compute (winding / broad phase) remains the next M4 step.

**M1 progress (`src/exact_csg/`).** Landed:
- **Classification** — generalized winding number (`winding_number` / `point_in_mesh`).
- **Straddle detection** — triangle–triangle transversal predicate (`tri_tri_intersect`).
- **Boolean (well-conditioned)** — `boolean(a, b, op)`: exact for disjoint / strictly
  nested solids, returns `BooleanOutcome::NeedsArrangement` when surfaces cross
  rather than emitting a corrupt mesh.
- **Arrangement step 1 — cut segments** — `tri_tri_segment` / `intersection_segments`:
  the exact `plane(A) ∩ plane(B)` segments clipped to both triangles, i.e. the
  constraints the re-triangulation will insert.
- **Arrangement step 2 (atom) — conforming plane split** — `split_triangle_by_plane`:
  cuts a triangle by a plane into sub-triangles that each lie wholly on one side
  (area-conserving, winding-preserving), the primitive for splitting a triangle
  against a cutter's supporting plane.
- **Arrangement step 2 (substrate) — point-insertion triangulation** —
  `triangulate_with_points`: inserts Steiner points into a triangle (each 1→3,
  winding-preserving); verified by Euler count (1+2k), area partition, and
  vertex inclusion.
- **Constrained triangulation** — `cdt::triangulate_constrained`: incremental
  insertion + Sloan-style flip recovery; verified on the hard interior–interior
  constraint (T-junction at both ends), area-conserving, degenerate-free.
- **Full arrangement boolean** — `boolean_arrangement`: refines **both** meshes
  along their intersection (shared cut vertices → watertight seam), classifies
  sub-triangles by winding number, assembles behind a **closed-manifold gate**.
  `boolean` now runs this on straddle and returns `Exact` only when the result is
  verified watertight, else `NeedsArrangement`.

**Working end-to-end — all three ops.** Verified against the float `CsgEvaluator`
oracle (volume match, 0 non-manifold edges):
- **Oblique boxes** (fully transversal, non-degenerate): **union, difference, and
  intersection** all resolve watertight.
- **Sphere through box**: **difference** resolves watertight; union/intersection
  are safely refused (see below).

**Never-wrong contract** holds for every op: the kernel either returns a mesh that
is watertight *and* volume-matches the oracle, or it refuses (`NeedsArrangement`).

**All-shapes coverage** (`all_shapes_all_ops_never_wrong`): 9 primitives (cube,
sphere, cylinder, cone, hexagonal/pentagonal/triangular prisms, tetrahedron,
extruded triangle) × every pair × 3 ops = **243 combinations**. **152 resolve
exactly** (~63%); **48 pairs fully resolve** with the inclusion–exclusion volume
identities verified; and the **never-wrong contract holds across all 243** (every
`Exact` result is watertight *and* identity-consistent). This test found and fixed
a real bug — `polyhedron()` now **auto-orients** inward-wound input to outward.

**Property-fuzzed** (oracle-free): across 60 random obliquely-rotated box overlaps,
**58** fully resolve all three ops to `Exact`, and every one satisfies the
inclusion–exclusion volume identities (`V(A∪B)+V(A∩B) = V(A)+V(B)`,
`V(A−B)+V(A∩B) = V(A)`) and is watertight; the other 2 are safely refused. And the
`Solid` front end uses this kernel via `to_geometry_exact()` (arrangement +
per-op float fallback), verified to match the float path on oblique boxes and to
render the full bracket identically. Output ships as **binary STL**
(`Solid::to_stl` / `geometry_to_stl`), round-trip-verified through `StlLoader`.

**Known degenerate case (→ M2).** Sphere-through-box union/intersection leave a
handful of cracks (6/2008, 3/1376 edges) localized to where the **UV-sphere's
poles land on the box seam** — the sphere's thin polar sliver triangles fall
exactly on the cut, so f64 centroid classification of those slivers is ambiguous.
This is a degenerate-tessellation artifact (difference keeps the well-conditioned
inner sphere and is unaffected), and the concrete motivation for **M2 exact
predicates**. The gate refuses it rather than emit a bad mesh.

**Remaining for M1:** exact-predicate robustness for polar-sliver / coplanar
degeneracies (M2, in progress), and swapping the naive AABB broad phase for
`mesh_bvh` (M4).

**M2 progress (`src/exact_csg/predicates.rs`).** Precision foundation landed —
from-scratch, pure-Rust **adaptive-exact predicates** (Shewchuk expansion
arithmetic: fast f64 filter, exact fallback only when the sign is in doubt; no
deps, contrary to the earlier `robust`-crate plan):
- **`orient2d`** — verified vs the `i128` determinant over 4000 near-collinear +
  random cases (naive f64 misjudges >100); wired into the CDT's point-location
  and flip-convexity decisions.
- **`orient3d`** — verified vs `i128` over 4000 random + 3000 coplanar cases
  (naive misjudges >50 coplanar).
- **`coplanar`** — exact coincident-face test built on `orient3d`.
- **`coplanar_clip`** (`coplanar.rs`) — 2D overlap of two coincident triangles
  (Sutherland–Hodgman with exact `orient2d` clipping decisions), triangulated and
  lifted to 3D; area-verified (contained / partial=1.0 / disjoint / non-coplanar).
  The refinement primitive for coincident faces.

- **Coplanar classification wired in** — `boolean_arrangement` now detects when a
  sub-triangle is coincident with the other solid's face (`coincident_face`, exact
  coplanar + centroid-containment, robust to mismatched face diagonals) and
  applies the **aligned/opposite keep-drop table** (`coplanar_keep`) instead of the
  ambiguous winding number; `boolean` routes coincident cases through the
  arrangement. **Lifts fully-coincident coplanar faces**: stacked/flush boxes now
  resolve to `Exact` — union = 2×2×4 (vol 16), difference = A (vol 8), watertight.
  The never-wrong contract held through the change (a degenerate-sliver false
  positive was caught by the oblique-box regression and guarded).

- **Partial coincidence — `coplanar_overlap_poly` wired into `refine_mesh`.** Faces
  are now split along the coincident-region boundary before classification, so
  *partially* coincident faces resolve. **Two axis-aligned boxes overlapping in
  volume now union correctly** (3×2×2 = vol 12, watertight) — the headline
  degenerate case, previously refused.

- **T-junction healing** (`heal_tjunctions`) — after classification, splits any
  triangle edge that has another vertex on its interior (geometry-preserving,
  no-op on a clean manifold, run only when the mesh isn't already watertight).
  This stitches the straight coincident seams, so **all three ops on
  volume-overlapping axis-aligned boxes now resolve** (union 12 / difference 4 /
  intersection 4, watertight). Verified against **real CGAL**: `box_notch`
  (overlapping-box difference) → PASS (volume exact, Euler match, Hausdorff 0).

**Global co-refinement implemented** (`corefine`): every intersection segment /
coplanar-overlap is computed **once** and its exact coordinates are fed into
**both** the A-triangle and the B-triangle it lies on, so seam vertices are
bit-identical on both sides. Plus **exact ray-parity classification**
(`point_inside`, via exact `orient3d`) and **conditional** T-junction healing
(only accepted if it actually closes the mesh — healing multiplies cracks on
curved seams, so it's guarded).

**Holes covered.** The through-hole/difference case resolves for axis-aligned and
most polygonal tessellations — `cube − prism` is exact for `$fn` = 3,4,5,6,12…;
`box_slot` (a square through-hole) **passes vs real CGAL** (volume, Euler,
Hausdorff 0). **Thirteen corpus models now gate at CGAL parity** — adding
`default_cyl` (exact facet phase), `mink2d` (non-convex 2D Minkowski), and
`surf_dat` (`surface` heightmap) to the ten below: `stacked_boxes`,
`box_notch`, `box_slot`, `plate_2d` (round-hole-in-plate via the **2D extrude path**
— the hole-bridging cap triangulator builds it watertight directly, so it reaches
parity even though the 3D-difference form of the same shape can't), plus the
operations `resize_box` (`resize`), `rounded_2d` (`offset(r)` → extrude), `proj_cut`
(`projection(cut=true)`), `mink_box` (convex `minkowski`), `booleans_2d` (2D
union/difference of a plus-with-hole through the arrangement kernel), and `sheared`
(`multmatrix`) — all watertight with Hausdorff ≤ 0.01 vs OpenSCAD/CGAL rendering the
same source. Broader showcase models live in `examples/scad-demos/` (curved ones
match volume + topology at default resolution and converge to Hausdorff parity as
`$fn` rises — mesh-level vertex parity on curves remains a non-goal).

**Still open — and now proven to require exact *construction*.** Round holes at
`$fn` = 8/16/32 **built as a 3D difference** (`cube − cylinder`; prism vertices
landing on the box diagonal) and sphere-box union/intersection fall back with a
handful of residual cracks. (Expressing the same hole via the 2D path sidesteps
this — see `plate_2d`.) Every f64-level
lever was tried and **none** lifts them: coplanar handling, shared-seam
co-refinement (`corefine`), exact predicates (`orient2d`/`orient3d`,
ray-parity classification), T-junction healing, and weld+boundary **repair**.
That exhaustion is the proof: the cracks are not coordinate mismatches (welding
handles those) nor classification errors (exact ray-parity handles those) — they
are structural inconsistencies produced when a cut coordinate lands *exactly* on a
triangle edge, which only **exact-arithmetic construction of the intersection
coordinates** (rational or Attene indirect implicit points — the
[Construction fork](#construction-fork)) resolves. This is why CGAL/libigl use
exact arithmetic here. It is a foundational change to how cut points are
represented across the whole pipeline (intersection → CDT → classification), not a
repair pass — a separate, multi-pass effort.

**Perf note.** The `repair` (weld + boundary-only T-junction heal) replaced the
O(T·V) heal and, with the BVH, cut the exact-CSG test time roughly in half. The remaining lever is exact-arithmetic
*construction* of the intersection coordinates (rational or Attene indirect
implicit points), not just exact *predicates* — the [Construction fork](#construction-fork).
That's a foundational change to how cut points are represented, deferred as its
own effort. Safely refused meanwhile — never-wrong holds (181 tests incl. fuzz
identities + oblique-box oracle green throughout; two models at CGAL parity).

**M3 progress (`src/exact_csg/metrics.rs`, `scripts/ci-openscad.sh`).** The
validation harness core is built and verified: `volume`, `euler_characteristic`
(V−E+F), vertex-sampled `hausdorff`, and a four-metric `compare` (watertight +
volume + Euler + Hausdorff), tested against synthetic meshes (box Euler = 2,
Hausdorff of a 0.1 shift = 0.1, pass/fail on identical vs scaled). The
`openscad_compare` example runs the gate on two STLs; `openscad_corpus` writes our
kernel's STL for a named model paired with a `.scad` in `tests/openscad-corpus/`;
`ci-openscad.sh` renders each model with real OpenSCAD (CGAL F6) and compares,
falling back to a self-consistency smoke test where OpenSCAD isn't installed.
Hausdorff is **point-to-surface** (closest-point-on-triangle), so it reads ~0
across differing tessellations of the same solid (verified: retessellated square → 0).

**Runs against real CGAL.** `ci-openscad.sh` auto-detects a local `openscad` or the
`openscad/openscad` **Docker image** (pulled on demand), rendering each `.scad`
with CGAL and comparing; self-consistency smoke test only if neither is present.
Measured result vs **real OpenSCAD/CGAL 2021.01**:
- `stacked_boxes` → **PASS** — watertight, volume exact, **Euler matches**,
  Hausdorff **0.0**. Genuine solid parity with CGAL on the coplanar union.
- `plate_with_hole` → volume exact + Hausdorff **5e-6** (shape matches CGAL) but
  watertight=false / Euler mismatch — the float-fallback mesh-quality gap, now
  *measured against CGAL*. Flagged known-limitation (non-gating).

**Remaining M3:** grow the corpus; move models up from known-limitation as the
kernel's exact coverage grows (e.g. once cylinder differences resolve exactly).

---

## Feature-flag wiring

```toml
# Cargo.toml
[features]
openscad = ["bvh-csg"]              # 3D booleans require the CSG/BVH stack

[[example]]
name = "openscad_bracket"
required-features = ["openscad"]
```

```rust
// src/lib.rs — beside the existing bvh-csg block
#[cfg(feature = "openscad")]
mod openscad;
#[cfg(feature = "openscad")]
mod exact_csg;                      // kernel; gated with openscad
#[cfg(feature = "openscad")]
pub use openscad::{cube, cylinder, sphere, Solid /* … */};
```

```bash
cargo run --example openscad_bracket --features openscad
scripts/ci-openscad.sh             # kernel vs CGAL oracle (dev machine w/ openscad installed)
```

JS shim: mirror the flag (`OPENSCAD=1 web/build.sh` → `web/features.js`) so the
browser can expose the `Solid` builder over wasm, matching the `mesh-bvh` /
`bvh-csg` addon pattern. Preview mode may expose the SDF raymarcher separately.

---

## Open questions / risks

- **Exact-arithmetic performance** on deep boolean trees — indirect predicates
  mitigate coordinate blow-up but the arrangement is still the cost center.
  Budget M4 for it.
- **Snap-rounding correctness** in the indirect fork — must preserve topology
  when collapsing implicit points to f64; this is the subtle part of Cherchi 2020.
- **Front-end fidelity vs OpenSCAD** — `$fn/$fa/$fs` tessellation of curved
  primitives affects the *input* soup; document that curved surfaces are faceted
  per our generators, so solid parity is against OpenSCAD run at matching `$fn`.
- **`polyhedron` input validation** — self-intersecting / non-manifold user input
  must fail closed, matching OpenSCAD's "not a valid 2-manifold" behavior.
- **Naming** — publish as OpenSCAD-*style*; the module doc must state results are
  not bit-compatible with CGAL's tessellation.
