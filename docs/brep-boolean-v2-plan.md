# threers B-rep boolean, second design

Stage 3b of [`brep-nurbs-plan.md`](brep-nurbs-plan.md) shipped as
`src/brep/boolean.rs`. It works: 51 boolean tests, and on the FreeCAD corpus
79 of the 95 booleans whose operands are exact come back exact, with OCCT's
volumes. This document is about why it should not be grown any further, and
what replaces it.

## Why

Every defect fixed in the last round had one shape: **one edge, built twice**.
A wall's carried edge and the outline a coincident wall cut beside it; a leg of
a bent cut and the straight cut beside it; a door's floor clipped to a ring's
outline but not to its courtyard. Each pair had the same geometry, two names,
and one face on each name, so `defects()` read two single-use edges.

Each was fixed by a pass that finds the duplicates afterwards and merges them.
There are now about twenty such passes (carried rims, ring canonicalisation,
curve aliasing, bent-run splitting, rim cutting by claim count…), and
`boolean.rs` is 6,900 lines. The passes interact: a fix for one corpus case is
measured against the rest, not derived. That is the signature of a missing
structure, not of missing cases.

The structure that is missing is in two places.

1. **The topology cannot say "the same edge".** An [`Edge`](../src/brep/body.rs)
   is a polyline of vertex indices and a pair of surface ids. It has no curve, no
   parameter range, no orientation; a [`Face`] holds an unordered list of edges
   and no wires. Two edges are the same only when their samples happen to weld,
   so sharing is recovered by position, after the fact, at a tolerance.
2. **The boolean has no shared intersection state.** Faces are split one at a
   time from curves computed per face pair. An edge of A lying in a face of B, a
   vertex of A on an edge of B, two coincident edges: each is discovered (or not)
   by whichever face split happens to meet it, and the two sides of it can
   disagree.

## What replaces it

### 1. Topology with identity — `brep::topo`

The standard boundary representation, as OCCT and Parasolid have it:

| Entity | Holds |
|---|---|
| `Vertex` | point, tolerance |
| `Edge` | exact `Curve3d`, parameter range, two `Vertex` ids, tolerance |
| `Coedge` | its `Edge`, sense, pcurve on the face's surface, next/prev in its loop |
| `Loop` | first `Coedge`; outer or hole |
| `Face` | surface, sense, loops |
| `Shell` / `Solid` | faces / shells |

An edge is shared because two coedges *refer to it*; nothing is welded. Tessellation is derived:
each edge is sampled once from its curve, at the requested tolerance, and both faces take those
points, so the result is closed by construction as 3a promised. `Body` stays as the
tessellation-facing view and is produced from `topo`, so every existing consumer
(STEP, blend, `fc-part`) keeps working during the move.

The validity checker becomes structural: each edge has exactly two coedges of opposite sense
in a closed shell, loops close in pcurve space and in 3D, vertices lie within tolerance of
their edges' curve ends, and edges lie within tolerance of their faces.

### 2. A general fuse — `brep::fuse`

The architecture of OCCT's `BOPAlgo`, which is one algorithm for every operation:

1. **Intersect everything, once, into one table.** In order: vertex/vertex, vertex/edge,
   edge/edge, vertex/face, edge/face, face/face. Each edge collects *paves* (parameters where it
   is split); each face collects the vertices and edges lying in it and the section curves
   crossing it.
2. **Common blocks.** A split piece of an edge of A that coincides with a split piece of an edge
   of B (or with a section curve) becomes **one** entity both refer to. Coincident faces are
   the same: one face-in-face record, not a separate wall path. This is the step whose absence
   produced every "built twice" defect.
3. **Build faces.** Per face, a 2D wire builder in its pcurve space takes the face's split edges
   plus its section edges and returns the faces it splits into (the existing
   [`planar::subdivide`](../src/brep/planar.rs) arrangement, generalised from chords to an edge set
   with identities).
4. **Classify.** One exact point-in-solid test per connected group of split faces, propagated
   across uncut edges. No tessellation and no `point_in_mesh`.
5. **Assemble.** Union, intersection and difference are a choice of which split faces to keep and
   which to flip. Shells are built from the shared edges, with nothing to weld.

### 3. Exact section curves

Plane–plane gives a line; plane–cylinder, a line pair, circle or ellipse; plane–cone and plane–sphere,
conics and circles; coaxial or parallel cylinders, lines and circles. These are closed forms
(the stage 2 SSI already decides most), and an edge should carry the curve rather than a
`Curve3d::Sampled` of it. The rest marches with Newton refinement and keeps a pcurve on both
surfaces, so the two faces cannot disagree about where the edge is.

### 4. Tolerances

- Per-entity tolerances (vertex, edge) instead of one absolute tolerance for the whole run,
  combined with the relative `SNAP` in `planar.rs`.
- Exact orientation and incircle predicates (Shewchuk-style adaptive arithmetic) in the 2D
  arrangement, so a crossing is never both found and missed.
- Coincidence decided symbolically for analytic surfaces (same plane, same axis and radius)
  before any sampling.

### 5. Testing by property

A generator of random compositions — boxes, cylinders, cones, spheres, at random placements,
including touching and coincident ones on a grid — and invariants checked on every result:

- `V(A∪B) + V(A∩B) = V(A) + V(B)` and `V(A−B) = V(A) − V(A∩B)`;
- every result is valid (`defects()` empty) or declined, never wrong;
- every result can go back in as an operand;
- grid placements, which make coincident faces and edges, at least as often as generic ones.

Corpus volumes from FreeCAD (the `freecad-rs` oracle) stay the external check.

## Stages

Each stage lands standalone and the current `boolean.rs` stays the fallback until stage D.

| Stage | Content | Gate |
|---|---|---|
| A | Property harness (§5) against today's boolean; baseline numbers recorded here | runs in CI-sized time; baseline committed |
| B | `brep::topo` (§1), `Body` ↔ `topo` both ways, structural checker; STEP and blend unchanged | all brep tests green through the round trip |
| C | `brep::fuse` (§2) on planar solids, with exact lines (§3) and exact 2D predicates (§4) | property harness on boxes: no wrong results; corpus planar cases ≥ today |
| D | Analytic curved surfaces, marching for NURBS; switch `Body::boolean` to `fuse`, old passes deleted | brep_boolean suite and corpus ≥ today; property harness clean |
| E | Classification without tessellation (§2.4); per-entity tolerances | — |

## Measured baseline (2026-11-14)

- `tests/brep_boolean.rs`: 51 pass. `--lib brep`: 163 pass. `tests/brep_blend.rs`: 25 pass.
- FreeCAD corpus booleans (`freecad-rs` `boolean_census`, Drilling_1 excluded): 79 exact,
  19 through the mesh because an operand is only a mesh, 16 declined or invalid:
  NeedsArrangement 7, NotWatertight 2, NotASolid 2, base not a solid 3, invalid input 1,
  invalid result 1 (fixed since, by clipping a wall outline to the face's holes).

Stage A, `tests/brep_boolean_property.rs` at `BREP_PROPERTY_CASES=60` (default seeds):

| Family | Right | Declined | Invalid | Wrong |
|---|---|---|---|---|
| Boxes on a half-unit grid (exact volumes) | 141 | 3 | 3 | 0 |
| Box and axis-aligned cylinder (identities) | 135 | 42 | 3 | 0 |

No wrong volumes and no broken identities, so the boolean keeps its contract: a result is
never quietly wrong. What it costs is the declines and the invalid results; the 6 box cases all
have faces of the two boxes coinciding in part. The 45 cylinder cases are not yet sorted by
cause: some have the axis or a rim lying on a box face, and some, such as a cylinder passing
through both the top and the bottom of a box, have no coincidence at all.
`BREP_PROPERTY_VERBOSE=1` lists them.

## Where it stands

`src/brep/fuse.rs` is stages C and most of D, built ahead of stage B: a general fuse over
the existing `Body`, with the pool of vertices standing in for vertex identity and edges
chained back out of shared sides. `Body::boolean` runs it first (setting `general_fuse`,
`KernelGeneralFuse` in `freecad-rs`) and takes the face-splitting path only for what it
declines; a graze it declines is returned as one.

Done:

- Planes: the line two planes share, clipped to both faces exactly; coplanar faces hand each
  other their outlines. Welded at the arithmetic's precision, not the caller's tolerance —
  at a 4.7 mm tolerance a wall's thinner membrane was merged and the volume came out wrong.
- Curved faces, through `ssi` and `march`, clipped to both faces by halving to the boundary;
  a stretch along a ring of either face takes the ring's own points.
- Faces that wrap: a period wide with the seam on both sides; sections laid down once per
  period they reach; pieces rejoined across the seam afterwards, a band with nothing cut
  out of it written as a face that wraps, as the primitives make one.
- Grazes along a curve or at a point; a section that stops inside a face is declined, not
  pruned.
- Every vertex settled onto every surface it is on, so an edge's points are its curve's.
- Edges chained between the same two faces, broken at corners: OpenCASCADE's topology on
  a corner cut (9 faces, 21 edges, 14 vertices) and on a drilled plate.
- Planar faces fill by a constrained Delaunay triangulation where hole bridging fails.

Measured after (same harness, same seeds, `BREP_PROPERTY_CASES=60`):

| Family | Right | Declined | Invalid | Wrong |
|---|---|---|---|---|
| Boxes on a half-unit grid | 147 | 0 | 0 | 0 |
| Box and axis-aligned cylinder | 160 | 19 | 1 | 0 |

FreeCAD corpus booleans: 102 exact (79 before). All suites green: 51 boolean, 172 lib,
25 blend, 27 brep; `freecad-rs` CI green, its oracles included.

How an edge keeps its curve without stage B: each edge of the inputs is remembered with the
two surfaces it lies between and, when its samples lie on one, its circle. A vertex of the
result found on such an edge — a sample of it, or a crossing on one of its chords — is
settled onto those surfaces and that circle; so is the end of a result edge whose inner
points all are. Where an arc meets a line or another circle at a shallow angle the crossing
is solved in closed form, since taking turns onto each crawls there (a line meeting an arc
at about a degree had its corner 0.48 out). An edge of two points that is not straight gets
its middle. With that, the results that did not convert back do, and come out the same at
every tolerance.

The tolerance is the caller's, and means what it says: points closer than it are one, and
curves are sampled to it. Flat faces need neither, and are worked at the arithmetic's
precision. `freecad-rs` calls the fuse at five millionths of the operands' size
(`BooleanFuseRelativeTolerance`) rather than its older boolean's two-thousandth: a
window's reveal in a curved wall of a building tens of metres across is thinner than the
latter, and was merged — a valid solid of the wrong volume, which the oracle caught.

Pieces are classified exactly where the other solid is bounded by planes, cylinders, cones
and spheres: a ray against the surfaces themselves, retried in another direction where it
grazes. A body cut many times over may not tessellate closed at a fine tolerance, and a mesh
that does not close has no inside — every point read as outside it, and a hole's wall went
missing. Tori still go by the mesh, the coarsest closed one if need be; splines by rays
first (below).

Solids with NURBS faces are classified by rays as well (2026-11-22, the first part of stage
E). A ray's hits on a NURBS face are the crossings of its seed grid's triangles, each polished
by Newton's method on `S(u, v) = p + t·d` onto the surface; a crossing that does not polish
gives no answer. With such faces two clean directions must agree, and where they do not, or a
hit is unsure, the closed mesh is made then and asked instead, once per boolean. On the
drilled CAM test part (`Drilling_1`, a pad with two extruded spline faces cut by eighteen
holes) a boolean took about 120 s, all of it tessellating at five millionths of the model
and inverting points against the mesh; it now takes under a second, at 77 MB, and the
shapes are the ones FreeCAD makes. Point inversion on a NURBS surface keeps the seed grid of
each surface it has seen (per thread, found again by the surface's whole definition) and
picks its four best seeds without sorting them all; `NurbsSurface::point` evaluates its basis
on the stack below degree 16.

Flat faces weld at a millionth of the model: inputs arrive agreeing only to single
precision (a face at y = 2199.99949 against 2200), and at a ten-millionth the cut between
them stopped short.

Tried and taken back: reading cones that do not close, hyperbolas, and a closed rim on a
seamed face back into `freecad-rs` by fitting. The kernel's results were sound — its own
mesh of them agrees — but the shapes read back had volumes out by one or two parts in ten
thousand, which the oracle caught. A check of each read-back shape against the kernel's
mesh could not tell a misread curve from the mesh's own error at that precision. The way
to do it is to read each such curve from the two surfaces that meet in it, not from its
samples.

What a closer look at those found. The cut behind PartDesignExample's second pocket — a bore
whose axis lies in the end wall of an earlier pocket and which touches that pocket's side
wall along the corner — is right in the kernel: its exact volume, by the divergence theorem
over its faces, is 588519.62 against OpenCASCADE's 588519.37. Two other things were wrong.

- The tessellation of its bore's wall, a band with a notch cut out of its top, comes out 86
  units of volume off on a test model (`a_bore_through_a_pockets_end_wall`), though closed:
  the fill accepts 99 % coverage of the parameter region, and long chords across the band
  are not refined to the surface. A mesh, not a solid; but a mesh is what the old path
  classifies with and what a check against one would believe.
- The model's inputs have the pocket's end at x = 37.50001 and the axis at 37.49999. Within
  the tolerance, the pocket floor's side, the line y = −25, comes out with a bend of 9e-4 at
  its end, where it meets the bore's circle just short of the tangent point. The shape is
  sound to the tolerance; read back by fitting, the bent line is not a line, and the top
  face's area came back 7 units out.

Ellipses and hyperbolas are now read back exactly, from the plane and the quadric they lie
on. Closed rims on a seamed face, and cones that do not close, read back exactly on the test
models too, but stay off until a tangency within the tolerance is resolved as one.

Still to do, in order:

1. A tangency within the tolerance resolved as a tangency (the bend above), then closed
   rims on seamed faces and partial cones read back again.
2. The fill of a trimmed curved face: a constrained Delaunay triangulation over the loops
   and points of the surface inside them, instead of an ear clip refined afterwards.
3. Curves that are neither lines nor circles drawn as outlines between pieces of one plane
   (an ellipse from a slanted cut, carried into a later boolean) — the same mechanism with
   a conic in place of the circle.
4. Stage B proper, then deleting the face-splitting path.

## Non-goals

- Linking OCCT or any C/C++ kernel. `truck` is worth reading; the build is native.
- Keeping `boolean.rs`'s passes alive inside `fuse`. If a case needs one, the table in §2.1–2.2 is
  missing an entry, and that is where it goes.
