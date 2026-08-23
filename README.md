# threers

A **drop-in three.js replacement** for Rust and the browser, backed by [wgpu](https://github.com/gfx-rs/wgpu). The same Rust core runs natively (winit) and as WebAssembly, with a JavaScript shim that exposes the familiar `THREE.*` API so existing three.js code can run with minimal changes.

**Current version:** [0.0.4](CHANGELOG.md) · [Changelog](CHANGELOG.md)

## Table of contents

- [Features](#features)
- [Prelude](#prelude)
- [Quick start](#quick-start)
  - [Native (desktop)](#native-desktop)
  - [Optional: mesh-bvh / CSG](#optional-mesh-bvh--csg)
  - [OpenSCAD solid modeling](#openscad-solid-modeling)
  - [Lattice infill](#lattice-infill)
  - [Kirigami corrugations](#kirigami-corrugations)
  - [Materials: metals, glass, anisotropy, thin film](#materials-metals-glass-anisotropy-thin-film)
  - [Headless render & video export](#headless-render--video-export)
  - [Path tracing (`raytrace`)](#path-tracing-raytrace)
  - [Metal backend (Apple)](#metal-backend-apple)
  - [visionOS](#visionos)
  - [Native codecs (GIF, APNG, VP9, H.264, HEVC)](#native-codecs-gif-apng-vp9-h264-hevc)
  - [Subtitles and captions](#subtitles-and-captions)
  - [Animating OpenSCAD models](#animating-openscad-models)
  - [Web (browser)](#web-browser)
  - [Earth, sun and lens flare](#earth-sun-and-lens-flare)
- [RLX bridge](#rlx-bridge)
- [Physics](#physics)
- [Parity compare UI](#parity-compare-ui)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Development](#development)
- [Status](#status)
- [License](#license)

## Features

- Scene graph (`Object3D`, `Scene`, transforms, layers)
- Buffer geometries, PBR materials, lights, shadows, fog
- **Extended PBR** — clearcoat (direct + IBL), transmission/IOR/dispersion, anisotropy (+rotation), sheen (direct + IBL), thin-film iridescence (+thickness map), volume absorption, normal mapping with vertex tangents, geometric specular AA, and measured-reflectance `materials::presets`
- **HDR pipeline** — `Rgba16Float` textures, float `.hdr` decoding, HDR environment prefiltering, and ACES tone mapping on `Renderer`
- **Area lights** — `RectAreaLight` shading (analytic diffuse form factor + representative-point specular)
- Custom WGSL materials via `ShaderMaterial`
- wgpu renderer with post-processing (FXAA, bloom, SSAO, glitch, halftone, …)
- Loaders (glTF, OBJ, STL, HDR, …), animation, controls, helpers
- **Headless** offscreen render → tightly packed RGBA (`HeadlessRenderer`, native)
- **Opt-in Metal backend** (`metal`, macOS/iOS) — a second renderer that talks to Metal directly through the Objective-C runtime: no `metal-rs`, no `objc` crate, no build script, no new dependency of any kind. Draws the same `Scene` to an offscreen target or a `CAMetalLayer`, and `PassAttachments::from_raw` will draw it into textures your app already owns. wgpu's `Renderer` stays the default and the complete one — see [Metal backend (Apple)](#metal-backend-apple) for what this covers and what it does not.
- **Opt-in visionOS** (`visionos`) — the CompositorServices frame loop and ARKit world tracking, again as plain `extern "C"` against the visionOS SDK. Both eyes are drawn in **one pass** into a texture array, depth runs reverse-Z (the only convention the compositor accepts), per-eye projections come from `cp_drawable_compute_projection`, and the device anchor keeps content still in the room instead of riding on the viewer's head. See [visionOS](#visionos).
- **Video export** (`video`) — frame sequence → ffmpeg; native GIF/APNG/H.264/VP9 when `native-codec` is on
- **Native codecs** (`native-codec`) — pure-Rust GIF (encode/decode), APNG, VP9/WebM, H.264/MP4, HEVC/MP4; wasm-safe browser download
- **Subtitles & captions** — SubRip / WebVTT parse + write, text rasterized on-screen over the 3D frame or burned into exported video, plus sidecar files and soft-subtitle tracks (MP4 `tx3g`, WebM WebVTT). Pure Rust, no font assets required, wasm-safe.
- **Height maps** — `displacementMap` / `displacementScale` / `displacementBias` on Standard and Physical materials, sampled in the vertex stage (plain, skinned, instanced and shadow paths)
- **Kirigami plate lattices** — Kirigami Expanded Miura corrugations (Parra Rubio et al., ASME 2023): planar and curved sandwich cores as plates with crease topology, developed SVG nets, and discrete origami cells. `cargo run --release --example kirigami`
- **Lattice generators** — 29 periodic infills as `BufferGeometry`, in one API: 8 triply periodic minimal surfaces (gyroid, Schwarz P and D, Neovius, I-WP, Fischer–Koch S, Lidinoid, split P), 7 beam cells (simple cubic, BCC, BCC-Z, FCC, octet truss, diamond, Kelvin), the 9 patterns a slicer draws (rectilinear turning and aligned, grid, triangles, tri-hexagon, honeycomb, cubic, quarter cubic, concentric), and 5 face-connected cuboctahedra voxels (rigid, compliant, auxetic, chiral CW/CCW — Jenett et al., Sci. Adv. 2020). Sized the way a slicer sizes them — ask for *25 % density* and the wall thickness is solved for you — and meshed by a marching cubes that derives its own case table, so ambiguous cells cannot crack and the output is indexed, welded and watertight. `grade` varies thickness across the part, `trim` pours the lattice into any shape you can write as a field, and both keep the mesh closed. `wall_samples` catches the failure mode that would otherwise be silent — a wall thinner than the sample step comes out as gravel, not as a wall — and `resolve_walls` fixes it per generator rather than paying the worst case on all of them. No feature flag, no dependencies.
- **Opt-in planets** (`planet`) — `Planet` / `Starfield` builders that stack a textured body, a cloud shell and an analytically shaded atmosphere; procedural equirectangular map generation; elevation → normal + roughness derivation; and a loader for NASA's public-domain Blue Marble, Black Marble, MODIS, GEBCO, CGI Moon Kit and Deep Star Maps imagery
- **Opt-in path tracer** (`raytrace`) — a physically-based renderer beside the rasteriser, not inside it. Rays leave the camera, scatter off surfaces by their BSDF, and either find a light or die trying; the image is the average of millions of such paths, so global illumination, colour bleeding, soft shadows, glossy interreflection and refraction fall out of the integral rather than being features bolted on. A principled BSDF (Lambert + anisotropic GGX + rough dielectric transmission + clearcoat, with Kulla–Conty energy compensation, so rough gold does not render grey), emissive geometry sampled as area lights, next-event estimation with the power heuristic, **Owen-scrambled Sobol sampling** (every 2D decision a path makes comes from a stratified low-discrepancy pair — 1.5–3× lower RMS error at the same sample count, which is 2–9× the effective samples), **importance-sampled environments** (an HDRI's sun is 6·10⁻⁵ of the sphere — cosine-weighted sampling finds it once in twenty thousand tries and returns speckle), **adaptive sampling** that stops each pixel when its own standard error settles, Russian roulette, Beer–Lambert absorption inside solids, a thin-lens camera with autofocus, and an AOV-guided À-Trous denoiser thresholded on each pixel's *measured* variance, with albedo and normal guides that follow the path through mirrors and glass the way Cycles' do. **Two backends over one scene form**: a portable CPU reference (wasm32 included, rayon-parallel with `parallel`) and a wgpu compute kernel — no new dependency, it reuses the renderer's own wgpu — that runs the same estimator and agrees with the CPU to within Monte-Carlo noise. Takes an ordinary `Scene` with ordinary `Material`s and `Light`s; nothing is authored twice. See [Path tracing](#path-tracing-raytrace).
- **Opt-in mesh BVH** (`mesh-bvh`) — accelerated raycast / shapecast (three-mesh-bvh–compatible)
- **Opt-in CSG** (`bvh-csg`) — boolean ops on `BufferGeometry` (three-bvh-csg–compatible); Rust native + JS addon
- **Opt-in OpenSCAD** (`openscad`) — a `.scad` interpreter **and** a `Solid`/`scad!` Rust DSL, evaluated by a pure-Rust **watertight exact-CSG kernel** (resolves curved∧curved booleans, float fallback where unverifiable). Mesh import/export (STL/OBJ/OFF/3MF/AMF/glTF-GLB, DXF/SVG, `.dat`/`.png` heightmaps) and a live browser playground.
- **Opt-in NURBS** (`nurbs`) — f64 rational curves and surfaces with analytic derivatives, Boehm knot insertion / refinement / Bézier extraction / degree elevation, and **exact** conic and quadric constructors (a circle is a rational quadratic, so a sampled sphere holds its radius to ~1e-15 at any parameter — not to whatever segment count you picked). Curvature-adaptive tessellation to `BufferGeometry` with analytic normals via `NurbsGeometry`, plus JS bindings (`NURBS=1 web/build.sh`). Stage 0 of the B-rep roadmap in [docs/brep-nurbs-plan.md](docs/brep-nurbs-plan.md).
- **Opt-in surface provenance** (`brep`) — every triangle of a built-in primitive knows the analytic `Surface` it was sampled from, so normals come from `Sᵤ × Sᵥ` instead of averaged face normals, UVs come from the surface's own parameterization, and a mesh can be **re-tessellated at any tolerance** after the fact rather than being frozen at whatever `$fn` was chosen. Recovers what triangles cannot hold — a truncated cone's apex is not a vertex, not on any face, and not inside the bounding box. Provenance is a sidecar: it never alters a mesh, and dropping it leaves every consumer behaving as before. Stage 1 of [docs/brep-nurbs-plan.md](docs/brep-nurbs-plan.md).
- **Opt-in analytic CSG acceleration** (`brep-csg`) — when both boolean operands carry provenance, a triangle pair is resolved by asking what its *surfaces* do rather than intersecting the triangles. **Fixes the exact-CSG kernel's longest-standing degeneracy**: two identical primitives translated along a shared axis, which used to fall back to the float kernel and now returns an exact watertight result. The classifier's exact-arithmetic coplanarity test cannot be passed by re-triangulated vertices; surface identity can. Strictly an accelerator — anything without a closed form falls through to the existing path, and a tagged boolean is asserted never to be worse than an untagged one. See [docs/brep-nurbs-plan.md](docs/brep-nurbs-plan.md) § Stage 2.
- **Opt-in B-rep boolean** (`brep-kernel`) — `Body::boolean` for union / difference / intersection on the B-rep itself, not on triangles. Surfaces are intersected in closed form, each face is trimmed by the resulting curves, and the seams become *shared edges*: both sides index the same vertices, so a result is watertight at any tolerance rather than merely coincident — and stays analytic, so a bore comes back as a `Cylinder` you can re-tessellate, not a band of triangles. Coaxial surfaces of revolution are exact, which covers a torus against a cylinder, sphere, cone or another torus — cases with no closed form in general position. A pair with no closed form is *traced* numerically rather than refused, so a cross-drilled hole resolves too. **It declines rather than approximates**: `TangentialContact` where two surfaces touch instead of crossing, `NeedsArrangement` when a face's subdivision does not come apart, `CoincidentFaces` where the two share a surface, and `NotWatertight` if the pieces do not assemble — the layer verifies its own result before returning it, so what you get back is closed or you get nothing. A result is also a *solid*: one closed shell with every edge on exactly two faces, which is what lets it be the input to the next boolean. Stage 4 of [docs/brep-nurbs-plan.md](docs/brep-nurbs-plan.md).
- **Opt-in STEP** (`step`) — ISO 10303-21 read and write, and the AP203/AP214 advanced-B-rep mapping to and from `Body`. Planes, cylinders, spheres, cones and tori map to their own entities, and so do the circles and lines bounding them: a drilled plate arrives in a CAD system as three cylinders it can re-dimension, not as facets it cannot. Import returns a *solid*, so a file you read can be cut again. Export is deterministic — no clock, no hash order — so the same model always produces the same bytes, and anything unmappable is reported rather than silently approximated. The promise is checked, not asserted: a corpus of booleans is written, read back and re-measured, and each must either come back the size it left or have the export name what it could not carry — a face whose edges bound two regions and do not say which (`AmbiguousRegion`), or one bounded partly by its own seam (`SeamBoundary`). Silence *and* a wrong answer is the one outcome ruled out, which is what a format whose point is that the model survives has to guarantee. `cargo run --example brep_part --features step`. Stage 5 of [docs/brep-nurbs-plan.md](docs/brep-nurbs-plan.md).
- **OpenSCAD animation** — the `$t` loop rendered headlessly: colored parts from `color()`, pickable palettes, a camera-relative studio light rig with shadows, a camera that fits or orbits the model (or obeys its `$vpr`/`$vpt`/`$vpd`), static-model caching, concurrent frame evaluation, and one-call PNG / PNG-sequence / video output.
- **Companion crate `threers-physics`** — rigid-body physics: gravity, collisions, joints, scene queries (raycast / shapecast), inverse kinematics, continuous collision detection and zero-gravity/orbital models. Pure Rust, wasm-ready, with optional rayon and wgpu-compute acceleration. See [`crates/threers-physics`](crates/threers-physics).
- **Companion crate `threers-animation`** — easing, tweens, springs, timelines, cinematic camera animation (fly/orbit/paths/shake/shots), and blending an animated pose into a simulated one (ragdolls). See [`crates/threers-animation`](crates/threers-animation), [`docs/animation.md`](docs/animation.md), and [`docs/camera-animation.md`](docs/camera-animation.md).
- **Published JS package `threers`** — npm / Deno / Node ESM from [`crates/threers-js`](crates/threers-js). Default **`threers/mini`** (small wasm); **`threers/full`** for CAD/codecs/path tracer. Arch-independent wasm.
- **Published native Node package `threers-node`** — napi-rs addon from [`crates/threers-node`](crates/threers-node) with per-arch binaries (`@threers/node-darwin-arm64`, …). Not Neon; Deno stays on the wasm package. See [`docs/bindings.md`](docs/bindings.md).
- **Published Python package `threers`** — PyPI wheels from [`crates/threers-py`](crates/threers-py) (maturin / PyO3; arm64 + x86_64 matrices).
- **Companion crate `threers-robot-arm`** — a four-axis pick-and-place cell run two ways (kinematic and servo-driven), sized from its own inverse dynamics, with a scrubbable browser player. See [`crates/threers-robot-arm`](crates/threers-robot-arm).
- **Companion crate `threers-probe`** — screen-space neural global illumination from a G-buffer and a set of lighting probes: trained in RLX, run anywhere RLX runs. See [`crates/threers-probe`](crates/threers-probe).
- **Companion crate `threers-continuum`** — tendon-driven continuum robots: a flexible rod as a chain of elastic hinges, cable routing, Clark coordinates and task-space tip control. See [`crates/threers-continuum`](crates/threers-continuum).
- **Opt-in RLX bridge** (`rlx`) — [RLX](https://crates.io/crates/rlx), an ML compiler + runtime, as a neighbour of the renderer rather than a layer under it. Vertex attributes and pixels convert to `rlx` tensors of a declared shape and back (NHWC/NCHW, sRGB↔linear, `Rgba16Float` included), and `GraphRunner` compiles a graph once so a per-frame filter does not pay the compiler sixty times a second. On top of that: `ConvFilter` (blur/sharpen/Sobel/emboss as depthwise conv2d), `Diffusion` (iterated `h ← h + κ∇²h` for terrain), `mesh::taubin_smooth` (smoothing as a graph over `[n, 3]` positions), `ColorGrade` (a colour transform **fitted** to a reference by autodiff — not applied, *learned*), and `Palette` (k-means with the assignment step on the device). Built with rlx's `cpu`, `tensor` and `gpu` backends, native **and** wasm32.
- **Opt-in exact Delaunay / Voronoi** (`rlx-geo`) — [`rlx-geo`](https://crates.io/crates/rlx-geo)'s integer predicates, so scattered samples triangulate to *the* Delaunay mesh rather than one a floating-point predicate rounded into a near-miss (a flipped triangle in a height field is a visible spike). `heightfield_geometry` returns `BufferGeometry` with normals and UVs; `refine_heightfield` samples adaptively, reaching ~9× lower error than a uniform grid of the same vertex count; Voronoi labels, site-distance, wall-distance and edge maps drive cell textures (cracked mud, stained glass) via `normal_map_from_height`.
- **Parallel CPU work** (`parallel`) — rayon-backed BVH/CSG evaluation; native only, and order-preserving so results are unchanged
- **Web**: `web/threejs-shim.js` + wasm — drop-in `THREE.*` replacement targeting three.js r165
- **Native**: winit examples for desktop development and debugging

## Prelude

`use threers::prelude::*;` brings in the scene graph, the common geometries and
materials, lights, cameras, textures, the renderers, the vector maths, and the
`Camera` trait — which has to be in scope for `camera.view_matrix()` to resolve,
and is the sort of thing a prelude is for.

```rust
use threers::prelude::*;

let mut scene = Scene::new();
scene.add(Object3D::mesh(Mesh::new(
    BoxGeometry::new(1.0, 1.0, 1.0),
    Material::Standard(StandardMaterial::new(Color::from_hex(0xff8844))),
)));
scene.add_light(DirectionalLight::new(Color::WHITE, 3.0));

let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
camera.position = Vector3::new(3.0, 2.0, 4.0);
camera.look_at(Vector3::ZERO);
```

It is a *curated* subset, not everything the crate exports: a glob import claims
every name in it, so the maths shapes (`Plane`, `Sphere`, `Triangle`, `Ray`,
`Box3`, …), the curve and path types (`Path`, `Shape`, …), and the subsystems you
reach for deliberately are left out of the root. Nested modules keep those
reachable without polluting the glob:

```rust
use threers::prelude::*;
use threers::prelude::controls::*;
use threers::prelude::animation::*;
```

All of them also remain at the crate root — `threers::Path`,
`threers::loaders::GltfLoader` — one `use` away. Feature-gated nested modules
(`prelude::csg`, `prelude::openscad`, `prelude::nurbs`, `prelude::raytrace`,
`prelude::captions`) appear when the matching Cargo feature is on. With the
`metal` feature on, the Metal renderers join the root prelude, because they are
renderers like the others.

Meta feature bundles (`cad`, `media`, `gi`, `apple`, `full`) turn on related leaf
flags in one go — see the feature table in the crate docs.

## Quick start

### Native (desktop)

Requires Rust 1.87+ (wgpu 30's minimum), a working wgpu backend (Metal/Vulkan/DX12), and dev dependencies from `Cargo.toml`.

```bash
cargo run --example cube          # spinning PBR cube
cargo run --example scene_graph   # hierarchy + lights
cargo run --example controls_orbit
cargo run --example shader_material
cargo run --release --example ocean_water   # spectral ocean: waves, foam, buoyancy
```

`ocean_water` is a **GPU FFT ocean**: a JONSWAP spectrum on a full lattice,
inverse-transformed each frame into three tiling cascades (swell / waves /
ripples) by a butterfly compute pass. Everything reads the cascades through one
shared sampler — a compute pass displaces a camera-anchored disc from them, the
`ShaderMaterial` fragment samples them per pixel for normals and breaking-wave
foam, a second compute pass accumulates persistent foam and wake, and the CPU
answers buoyancy from the dominant modes with no readback. The mesh is uploaded
once and written in place on the GPU, so per-frame CPU cost is flat in vertex
count (~0.3 ms at every quality level). Eight sea-state presets, five mesh
densities:
`cargo run --release --example ocean_water -- storm --quality ultra`,
`-- tropical --png ocean.png`, `-- --bench`, or `-- --list`.

Waves **shoal** over the analytic sea bed — slowing, shortening, growing by
Green's law and breaking past 0.78 of their own depth, which is what draws the
surf zone. It also does screen-space reflection and refraction, caustics on a
shader-lit sea floor, persistent foam and wake, spray/rain/underwater-mote
particles, an underwater view with Snell's window, water masking, sparkle and
multi-point buoyancy.

The FFT's butterfly table is validated on the CPU against both a naive DFT and
**rlx's own FFT** before it ever reaches the GPU (`--features rlx`), because a
butterfly index off by one produces plausible noise rather than an error. rlx is
not in the per-frame path: measured, its CPU backend is ~6.6 ms a frame for three
cascades, and its GPU backend builds its own device rather than borrowing this
one, so results would cross host memory either way. (It used to be a different
major of wgpu as well; since the wgpu 30 upgrade both are the same major, which
is what made rlx's GPU backend usable on wasm32 at all.)

Four small opt-in APIs carry that, all reusable outside the example:

- `BufferGeometry::gpu_writable` adds `STORAGE` usage to a mesh's vertex buffer,
  and `Renderer::vertex_buffer` hands it back — so a compute shader can write
  positions and normals directly, vertex animation with no CPU round trip.
- `ShaderMaterial::textures` binds up to four of your own texture views as
  `u_tex0..3`, for sampling something a compute pass wrote this frame.
- `ShaderMaterial::screen_space` draws the material in the refraction pass, where
  the captured opaque colour and depth are available — the prerequisite for any
  custom SSR or refraction.
- `ShaderMaterial::side` is now honoured (it was previously ignored and always
  culled back faces, which silently hid anything meant to be seen from inside).

### Optional: mesh-bvh / CSG

```bash
# BVH picking demo
cargo run --example mesh_bvh_picking --features mesh-bvh

# Hierarchy CSG (exact TriKey parity with JS three-bvh-csg@0.0.16)
cargo run --example bvh_csg_hierarchy --features bvh-csg
cargo run --example bvh_csg_steps --features bvh-csg -- 2 --live
```

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["bvh-csg"] }
use threers::{CsgBrush, CsgEvaluator, BoxGeometry, SUBTRACTION};

let mut ev = CsgEvaluator::new();
let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
let _geom = ev.evaluate(&mut a, &mut b, SUBTRACTION);
```

### OpenSCAD solid modeling

The `openscad` feature adds two front ends onto one `Solid` CSG tree — an OpenSCAD
`.scad` interpreter and a Rust DSL — evaluated by a pure-Rust **watertight
exact-CSG kernel** (`to_geometry_exact`), with mesh import/export.

```bash
cargo run --example openscad_gallery --features openscad   # 37-demo language tour → STL
cargo run --example dsl_showcase     --features openscad   # the Rust DSL → STL
# convert a .scad to any mesh format (format = output extension):
cargo run --example scad2stl --features openscad -- model.scad out.glb
```

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["openscad"] }
use threers::{cube, cylinder, sphere, scad, parse_scad};

// (a) Rust DSL — fluent builder + the `scad!` macro:
let part = scad! {
    difference() {
        cube([30.0, 30.0, 30.0]);
        translate([0.0, 0.0, -1.0]) { cylinder(40.0, 8.0); }
    }
};
let _stl = part.to_stl();          // also .to_obj/.to_off/.to_3mf/.to_glb

// (b) or parse an OpenSCAD program:
let solid = parse_scad("difference(){ cube(20,center=true); sphere(12,$fn=48); }").unwrap();
let _glb = solid.to_glb();
let _ = (sphere(1.0),);            // primitives are also free functions
```

Curved∧curved booleans (e.g. `sphere ∪ sphere`, `cylinder ∩ cylinder`) resolve to
watertight meshes; the kernel falls back to the float evaluator only where its
manifold gate can't verify a result, so it is never wrong — only sometimes
deferential. Try it live in the browser: **`web/openscad-playground.html`**.

### Kirigami corrugations

Kirigami Expanded Miura plate lattices — pads, inclined walls, custom curvature,
discrete origami cells, tiled assemblies, TPMS/strut lattice cores, and a
joined crease net.

```bash
cargo run --release --example kirigami         # gallery PNG + SVG nets → out/
cargo run --release --example kirigami_orbit   # windowed viewer (13 presets + net)
```

```rust
use threers::{KirigamiExpandedMiura, KirigamiPreset};

let geom = KirigamiExpandedMiura::new(5, 6)
    .cell(28.5, 29.5, 68.0)
    .height(50.0)
    .thickness(1.2)
    .build();

let saddle = KirigamiPreset::Saddle.evaluate_default(1.2);
let hybrid = saddle.to_geometry_with_core(KirigamiCoreLattice::Gyroid, 0.20);
let tiled = KirigamiAssembly::tile(&KirigamiPreset::Planar.evaluate_default(1.2), 2, 2, 8.0)
    .evaluate();
std::fs::write("net.svg", saddle.develop_joined().to_svg()).unwrap();
```

Browser: `web/examples/kirigami.html` (needs a wasm build with `web/build.sh`).

### Lattice infill

29 periodic lattices, all unconditional — no feature flag and no extra
dependencies.

```bash
cargo run --release --example lattice   # every generator at 25 % density → out/lattice.png
cargo run --release --example cuboct    # cuboct voxels (Jenett 2020) → out/cuboct.png
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
| `Tpms` | `Gyroid`, `SchwarzP`, `Diamond`, `Neovius`, `IWP`, `FischerKochS`, `Lidinoid`, `SplitP` |
| `Strut` | `Cubic`, `Bcc`, `BccZ`, `Fcc`, `Octet`, `Diamond`, `Kelvin` |
| `Cuboct` | `Rigid`, `Compliant`, `Auxetic`, `ChiralCw`, `ChiralCcw` |
| `Infill` | `Rectilinear`, `AlignedRectilinear`, `Grid`, `Triangles`, `TriHexagon`, `Honeycomb`, `Cubic`, `QuarterCubic`, `Concentric` |

`LatticeKind::from_name` takes any of those names, so a `--lattice gyroid` flag
needs no table of its own.

#### Sheet or solid

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

#### Filling shapes, not just boxes

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

#### Resolution, and the one way this fails quietly

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
builds 1.6× faster with it on (2.0 s → 1.3 s for all 25 tiles, 10 cores).
Results are byte-identical either way: every sample has a fixed address in the
grid, so a build flag cannot change the geometry.

### Materials: metals, glass, anisotropy, thin film

`PhysicalMaterial` (three.js `MeshPhysicalMaterial`) carries the full extended
PBR stack — clearcoat, transmission + IOR + dispersion, anisotropy with an
in-plane rotation, sheen, iridescence, and Beer-Lambert volume absorption — all
honored by the renderer, plus normal / roughness / metalness / AO / emissive maps.

```bash
cargo run --release --example material_chart        # contact sheet → PNG
cargo run --release --example spacecraft_materials  # interactive scene
cargo run --release --example spacecraft_render     # same scene, headless → PNG
```

In the browser, **`web/examples/materials.html`** renders all of them live via
wasm/wgpu (also procedural — no HDRI or texture downloads). The JS shim accepts
three.js's own option names, so this is portable three.js code:

```js
new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.03,
    transmission: 1.0, ior: 1.5, thickness: 0.6, dispersion: 0.2,
    attenuationColor: 0x66ddaa, attenuationDistance: 0.8,
    transparencyMode: 'refract',
});
// …or start from a preset and tweak:
THREE.MeshPhysicalMaterial.preset('gold_foil', 0, { normalMap: crinkle });
```

Measured-reflectance presets live in `materials::presets` (gold and silver MLI
foil, aluminium, brushed aluminium, titanium, heat-tinted titanium, solar cell,
thermal paint, black kapton, optical glass):

```rust
use threers::materials::presets;

let mut foil = presets::gold_foil();     // F0 (1.000, 0.766, 0.336), metalness 1
foil.normal_map = Some(crinkle);         // the crinkle is what makes it read as foil
let panel = presets::solar_cell();       // dark cells + cover-glass clearcoat
let bell  = presets::anodized_titanium(320.0);  // thin-film hue by thickness (nm)
```

### HDR environments and tone mapping

An 8-bit environment cannot hold a sun: capped at 1.0, a solar disc can only be
made to *read* bright by making it physically huge, and PMREM then smears it
across tens of degrees so every metal reflects the same grey wash. Load a real
one instead:

```rust
use threers::{loaders::HdrLoader, PmremGenerator, ToneMapping};

let (pixels, w, h) = HdrLoader::parse_f32(&bytes)?;      // linear f32, unclipped
let cube = PmremGenerator::from_equirect_f32(&pixels, w, h, 256);
scene.environment = Some(Arc::new(PmremGenerator::generate_pmrem(&cube, 256)));

// Without this, anything above 1.0 hard-clips to white and loses its hue.
renderer.set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
```

Tone mapping is **off by default** so existing scenes render unchanged.

Two things worth knowing before you reach for metals:

- **Metals need `scene.environment`.** At `metalness = 1.0` the BRDF has no
  diffuse term, so with no environment a gold sphere renders near-black no matter
  how many lights you add. Prefilter one with `PmremGenerator::generate_pmrem`;
  that mip chain is also what makes `roughness` mean anything for reflections.
- **Pick a non-sRGB offscreen format.** The mesh shader does its own
  linear→sRGB encode, so a `HeadlessRenderer` left on the default
  `Rgba8UnormSrgb` target encodes twice and washes the image out (mid-greys land
  ~2.5× too bright). Use `.color_format(wgpu::TextureFormat::Rgba8Unorm)`, as the
  examples above do.

For a *brushed* look, `anisotropy_rotation` aims the streak in UV tangent space;
the tangent frame is derived from UV screen-space derivatives, so it needs a mesh
with UVs (without them the frame falls back to a view-dependent one and the
rotation is arbitrary).

### Headless render & video export

Native-only. `HeadlessRenderer` owns the wgpu device and an offscreen target; `export_video` pipes RGBA frames to ffmpeg (or in-process GIF/APNG when `native-codec` is enabled).

```bash
cargo run --release --example headless_video --features video   # H.264 demo
```

Per-format examples (optional `--out PATH`):

| Example | Features | Output |
|---------|----------|--------|
| `export_h264` | `video` | `.mp4` (native H.264 with `native-codec`; else ffmpeg `libx264`) |
| `export_hevc` | `video` | `.mp4` (libx265) |
| `export_hevc_vt` | `video` | `.mp4` (hevc_videotoolbox) |
| `export_vp9` | `video` | `.webm` opaque |
| `export_vp9_alpha` | `video` | `.webm` + alpha |
| `export_gif` | `video,native-codec` | `.gif` (no ffmpeg) |
| `export_apng` | `video,native-codec` | animated PNG |
| `export_webm_native` | `native-codec` | `.webm` pure-Rust VP9 |

```bash
cargo run --release --example export_gif --features "video,native-codec"
cargo run --release --example export_vp9 --features video -- --out /tmp/cube.webm
```

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["video"] }
use threers::{export_video, HeadlessRenderer, VideoCodec, VideoOptions};

let mut hr = HeadlessRenderer::builder().size(1280, 720).build().unwrap();
let opts = VideoOptions::new("out.mp4").fps(30).codec(VideoCodec::H264);
// export_video(w, h, frame_count, &opts, |i| { /* advance scene */; hr.render_to_rgba(...) })
```

GIF / APNG without spawning ffmpeg:

```rust
// features = ["video", "native-codec"]
let opts = VideoOptions::new("out.gif")
    .fps(12)
    .codec(VideoCodec::Gif)
    .transparent(true)
    .gif_colors(128);
```

Browser-friendly bytes API (also used from wasm):

```rust
// features = ["native-codec"]
use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};
let bytes = encode_animation_rgba(
    &AnimationEncodeOptions {
        width: 320, height: 240, fps: 15,
        codec: BrowserCodec::Webm,
        transparent: false,
        gif_colors: 256,
    },
    frames, // Vec<Vec<u8>> of RGBA
).unwrap();
```

### Path tracing (`raytrace`)

`Renderer` draws a scene the way a GPU pipeline does: project the triangles,
shade each fragment from a fixed set of lights, approximate everything else.
Those approximations are what make it run at 60 fps and what make it wrong in
familiar ways — light does not bounce, a mirror cannot show what is behind the
camera, and a glass ball does not bend the room.

`RaytraceRenderer` answers the same question by simulating light transport. It
takes the **same `Scene`** — same materials, same lights, nothing authored twice.

```bash
cargo run --release --example path_trace     --features raytrace,parallel   # Cornell box, CPU
cargo run --release --example path_trace_gpu --features raytrace,parallel   # GPU, timed against the CPU
cargo run --release --example hdri_lighting  --features raytrace,parallel   # image-based lighting
```

```rust
use threers::raytrace::{RaytraceRenderer, RaytraceSettings};

let mut renderer = RaytraceRenderer::new(1920, 1080);
renderer.set_settings(RaytraceSettings::default().with_samples(256));
let rgba = renderer.render_to_rgba(&mut scene, &camera);   // same layout as HeadlessRenderer
std::fs::write("out.png", threers::encode_png(1920, 1080, &rgba))?;
```

Progressive, so a preview refines instead of appearing all at once — the scene is
flattened and its BVH built once, then samples are added to the same film:

```rust
renderer.prepare(&mut scene, &camera);
for _ in 0..20 {
    renderer.accumulate(8)?;
    let preview = renderer.resolve_rgba();   // sharper each time
}
```

**Backends.** `CpuBackend` is the reference — portable, wasm32 included, and the
definition of correct for the module. `gpu::GpuBackend` runs the same integrator
as a wgpu compute kernel over the same scene, and is selected by swapping one
argument:

```rust
use threers::raytrace::gpu::GpuBackend;
let backend = GpuBackend::headless()?;               // or ::with_device(device, queue)
let mut renderer = RaytraceRenderer::with_backend(1920, 1080, Box::new(backend));
```

The kernel is packed for throughput rather than convenience: the whole scene goes
into four storage buffers plus one texture atlas (WebGPU's baseline guarantees
only four storage buffers per stage, and bindless is a native-only
wgpu feature this kernel does not assume), so there is
one bind group for any scene; samples are traced in batches per dispatch, because
a single dispatch that runs for seconds trips the GPU watchdog on every platform;
the film accumulates on the device and is read back once, not once per batch; and
one invocation owns one pixel, so there are no atomics anywhere.

On an Apple M-series integrated GPU the `path_trace_gpu` example runs about
**7× faster** than the rayon-parallel CPU backend at identical settings, and the
two frames agree to four decimal places — which is what the CPU↔GPU test in the
suite checks, along with a white-furnace test on each.

**Image-based lighting.** The usual way to light a realistic render is one
HDRI and no lights at all — `hdri_lighting` does exactly that: no
`DirectionalLight`, no `AmbientLight`, every shadow and highlight and bit of
fill coming out of `scene.environment`. Point it at a real `.hdr` with
`HDRI=studio.hdr`, or let it synthesise a sky with a sun four orders of
magnitude brighter than the blue around it.

```rust
let (pixels, w, h) = HdrLoader::parse_f32(&std::fs::read("studio.hdr")?)?;
scene.environment = Some(Arc::new(PmremGenerator::from_equirect_f32(&pixels, w, h, 256)));
settings.background = BackgroundMode::Environment;   // lit by it *and* seen behind
```

That needs three things to be right, and an HDRI is unforgiving about all
three: the full dynamic range has to reach the integrator (the 8-bit copy
`CubeTexture` keeps for display is Reinhard-compressed, and integrating it caps
the sky at about 1.0), the sun has to be importance-sampled, and the cube faces
have to be read in the orientation they were written in.

**Sampling.** Three strategies, all combined by the power heuristic so none is
trusted where it is poor: the BSDF's own lobes; the lights, including emissive
geometry by an area×luminance distribution; and the environment, by a 256×128
density built over its own brightness.

Underneath all of them, the *numbers* are stratified rather than independent.
Nearly every decision a path makes is two-dimensional — a point on a light, a
direction off a BSDF, a point on the lens — and each one draws a pair from an
Owen-scrambled Sobol (0,2)-sequence, so sixteen samples of a pixel cover the
unit square evenly instead of clumping the way sixteen independent draws do.
Dimensions are allocated in a fixed block per bounce so that the same decision
draws the same dimension on every sample, which is what makes the stratification
worth anything. Measured against a converged reference: 1.5× lower RMS on a
plain area-light scene, 3× on one with an environment, a sun and an emitter —
the latter being roughly nine times the effective samples. That last one is not a refinement — an
HDRI puts most of its energy into a sun about 6·10⁻⁵ of the sphere across, which
a cosine-weighted direction finds roughly once in twenty thousand samples, so
without it an HDRI-lit render is white speckle that does not resolve at any
sample count you would wait for.

**Denoising.** Cycles uses OpenImageDenoise — a pretrained U-Net — which is
not something this crate can ship. What it *can* take from Cycles is the part
that makes any guided denoiser work, and that is how the guide passes are
built: the albedo and normal are not "whatever the first hit was". A mirror's
own colour and normal describe the mirror, not the image in it, and a glass
ball's describe neither the glass nor the room behind it. So the guides follow
the path through specular and transmissive surfaces, carrying their tint, until
they reach something rough enough to describe — Cycles' scheme, thresholds
included.

The reconstruction filter is edge-avoiding À-Trous, but the colour threshold is
the pixel's own accumulated variance rather than a constant: two estimates of
the same value differ by about their combined standard deviation, so dividing
by it asks "is this difference more than noise?" instead of comparing against a
number that is wrong at every sample count but one. Where a pixel has already
converged the filter stands down entirely — there is nothing left to remove and
a smooth gradient is all it could damage.

And the filter's parameters are **fitted, not guessed**. `denoise_fit` renders
a handful of scenes twice — 32 samples and 8192 — and searches the four widths
against that ground truth, judged on a held-out interior:

```bash
cargo run --release --example denoise_fit --features raytrace,parallel
```

Which was worth doing. The hand-picked values it replaced were worse on every
scene tried, and on a smooth area-lit one they were worse than not denoising at
all. Relative error on the held-out scene went 0.096 → 0.084, and on glass
beside rough metal 0.182 → 0.157. `DenoiseParams::fit` is public, so the same
can be done against your own content.

**A trained denoiser** (`--features learned-denoise`) sits beside the fitted
À-Trous filter. It is a U-Net over the same guides — colour, albedo, normal —
trained on pairs this renderer generates itself, so no corpus is collected and
no licence attaches to the result.

Measured against **the OpenImageDenoise library inside Blender.app**, driven at
the settings Cycles sets in `intern/cycles/integrator/denoiser_oidn_base.cpp`
(filter `RT`, hdr on, srgb off, quality `HIGH`) — so the comparison is with what
Blender actually ships, not an approximation of it:

| scored on | ours | Blender's OIDN | |
|---|---|---|---|
| renders of this repo's NEMA 17 and printer | **0.03545** (5.51×) | 0.03720 (5.25×) | 4.7% ahead |
| the furnace test | **0.0125** | 0.0132 | 5.2% ahead |
| Cornell box | 0.0558 | **0.0492** | 13.3% behind |
| Veach MIS | 0.1016 | **0.0919** | 10.6% behind |

Both directions are worth stating. On real machined geometry it is ahead of
what Cycles ships. On the classic demonstration scenes it is behind, and the
per-scene split says why: the training corpus had no tight saturated box and no
frame holding a pinpoint light beside a broad one. Those regimes are in the
generator now.

The **furnace test** is the row to keep. A grey sphere in a uniform emissive
environment must converge to exactly the environment's radiance — the
multiple-scattering series sums to one — so the correct answer is known in
closed form and any structure a denoiser draws there is *provably invented*.
Being ahead on that one says the filter hallucinates less, which is the property
a renderer actually wants and which none of the other numbers measure.

The forward pass is `raytrace::denoise_net`: plain Rust, no dependencies,
`wasm32` included. It is checked against the implementation that trained it
rather than by eye — worst per-pixel disagreement **0.000000**
(`examples/denoise_parity.rs`). Training lives in `rlx-denoise`, which is
GPL-3.0-only and is not linked by this crate.

**In a browser.** `wasm_denoise` splits the work into `plan` / `extract` /
`run` / `merge`, where only `run` is expensive and belongs on a worker; tiles
share no state, so N workers produce the same frame as one
(`tiling_a_frame_matches_denoising_it_whole` asserts exactly that). The module
docs carry the worker wiring.

Measured: ~0.7 s a 128×128 tile at the default widths on one core, ~1.1 s at
`wide`. Four times the parameters costs 1.6× the time, so the kernel is
memory-bound, not compute-bound. A 1080p frame is ~135 tiles, which across
eight workers is **12–19 seconds** — enough to denoise a finished render in the
browser, or to sharpen a progressive one between passes, and **not** an
interactive viewport. Getting there means putting the convolutions on the GPU,
which the crate already has the wgpu plumbing for.

**Progressive denoising.** `render_progressive` traces in batches and denoises
the accumulated film after each, so a viewport shows a usable image from the
first sample rather than noise until the end — which is what Cycles does, and
Blender's default is to denoise from sample 1. Worth knowing what that buys: on
this content the denoiser at ~4 spp matches a raw render at ~110 spp, about 29×
fewer samples.

**Adaptive sampling** tracks each pixel's own standard error and stops it once
that falls below `adaptive_threshold` (1 % by default). Noise falls as
`1/sqrt(n)`, so the last halving of the error costs three quarters of the
render, and most of an image gets there long before its worst pixels do —
typically 10–40 % of the sample budget goes unspent for an image that is
indistinguishable. It does not bias anything: a pixel that stops early is still
the mean of its own samples, and the film divides by each pixel's own count.
The decision reads only what the pixel has accumulated, so an adaptive render is
still independent of how it was batched. On the CPU the saved samples are saved
time; on the GPU a workgroup runs until its slowest pixel, so the saving only
lands where whole tiles converge. `RaytraceSettings::final_quality()` turns it
off.

**What is simulated:** multi-bounce diffuse and glossy GI; the principled BSDF
(Lambert + anisotropic GGX + rough dielectric transmission + clearcoat) with
Kulla–Conty energy compensation; emissive geometry as area lights;
`Directional` / `Point` / `Spot` / `RectArea` lights, optionally with an angular
or spherical size for soft shadows; `Ambient` and `Hemisphere` lights and
`scene.environment` as importance-sampled image-based lighting; alpha cutouts
whose shadows are cutout-shaped; Beer–Lambert absorption inside transmissive
solids; depth of field.

**What is not:** participating media, subsurface scattering, spectral dispersion,
and caustics through next-event estimation. Materials with no physical reading —
`MeshNormalMaterial`, `MeshToonMaterial`, `ShaderMaterial` — are mapped to the
nearest surface that has one, and every such choice is listed in
`renderer.report().approximated` rather than made silently.

### Metal backend (Apple)

`--features metal`, macOS and iOS. A second renderer for the same `Scene`, written
against the Objective-C runtime directly: `objc_msgSend` declared and transmuted
per call site in `src/metal/objc.rs`, frameworks pulled in with
`#[link(kind = "framework")]`. **No `metal-rs`, no `objc` crate, no build script,
and no new dependency** — the feature adds a directory, not a line to `Cargo.lock`.
On any other platform the feature compiles to nothing.

```bash
cargo run --release --features metal --example metal_headless          # → out/metal_headless.png
cargo run --release --features metal --example metal_stereo            # → out/metal_stereo.png (both eyes)
cargo run --release --features metal --example metal_window            # winit + CAMetalLayer
cargo run --release --features metal --example metal_window -- --frames 120
cargo test --features metal --test metal_backend                       # 24 tests, on a real GPU
```

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["metal"] }
use threers::metal::MetalHeadlessRenderer;

let mut renderer = MetalHeadlessRenderer::builder().size(1280, 720).msaa(4).build()?;
let rgba = renderer.render_to_rgba(&mut scene, &camera)?;   // tightly packed RGBA8
```

On screen, given an `NSView` from any window toolkit:

```rust
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface};

let device = MetalDevice::new()?;
let mut renderer = MetalRenderer::with_device(device.clone())?;
let surface = unsafe { MetalSurface::from_ns_view(&device, ns_view, 1280, 720, 4)? };

if let Some(frame) = surface.next_frame() {
    renderer.render(&mut scene, &camera, &frame.attachments())?;   // draws and presents
}
```

Or into textures you already own — an AVFoundation pipeline, an `MTKView`, a
Core Image chain — with `PassAttachments::from_raw`.

**Covers:** meshes, instanced meshes, line segments, points and sprites; `Basic`,
`Lambert`, `Phong`, `Standard`, `Physical`, `Normal`, `Depth`, `Toon`, `Matcap`,
`Line`, `Points` and `Sprite` materials; one base-colour map with the sampler,
wrap modes and UV transform from the `Texture`; ambient, directional, point, spot
and hemisphere lights; linear and exp² fog; alpha blend, alpha test, vertex
colours, wireframe, MSAA, and depth-sorted transparency. Geometry, textures,
samplers and pipeline states are cached across frames.

**Does not cover** — use the wgpu `Renderer` for these: shadow maps,
post-processing, environment maps and IBL (so a metal-heavy material has only its
specular highlights), skinning and morph targets, and the normal / roughness /
metalness / AO / emissive map slots. Materials outside the list draw unlit in
their base colour rather than failing, so a scene authored for the wgpu path
still renders. Both backends share the same conventions — 0..1 clip depth, CCW
front faces, `flip_y` applied at upload, three.js punctual attenuation — so the
two agree on a scene as far as this one goes.

### visionOS

`--features visionos` (which implies `metal`). An immersive visionOS app owns no
layer and no swap chain: SwiftUI hands it a `cp_layer_renderer_t` and it pulls
frames. That API is C rather than Objective-C, so this is `extern "C"`
declarations transcribed from the XROS SDK headers — still no new dependency, no
build script, no `metal-rs`.

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["visionos"] }
use threers::metal::visionos::ImmersiveRenderer;

#[no_mangle]
pub extern "C" fn threers_visionos_run(layer_renderer: *mut std::ffi::c_void) {
    let mut renderer = unsafe { ImmersiveRenderer::new(layer_renderer) }.unwrap();
    let mut scene = build_scene();
    while renderer.render_frame(&mut scene).unwrap() {}
}
```

```swift
// The Swift side is one call.
ImmersiveSpace(id: "scene") {
    CompositorLayer(configuration: MyConfiguration()) { layerRenderer in
        threers_visionos_run(Unmanaged.passUnretained(layerRenderer).toOpaque())
    }
}
```

What it does per frame: waits for the compositor's optimal input time, asks
ARKit for the device anchor at the predicted presentation time, takes the
drawable's textures and per-eye transforms, computes each eye's projection with
`cp_drawable_compute_projection`, draws, and presents on the same queue.

**Both eyes in one pass.** With the compositor's `layered` layout the eyes are
two slices of one texture array, and `render_views` draws them in a single pass —
the vertex stage reads its eye from the instance id and writes
`render_target_array_index`, so the second eye costs pixels, not a second walk
of the scene graph. The `dedicated` and `shared` layouts work too, one pass per
eye. **Reverse-Z** throughout, because the compositor accepts nothing else
(`drawable.h`: *"It only supports reverse-Z depth"*).

Both of those work on any Mac, which is where they are tested — `cargo test
--features metal` renders a stereo pair into a two-slice array offscreen and
checks each eye's parallax, and renders a reverse-Z scene and checks the depth
test inverted with it:

```bash
cargo run --release --features metal --example metal_stereo   # → out/metal_stereo.png, side by side
```

> **Building for the device.** The `visionos` code is written against the XROS
> SDK and verified against it (every `cp_*` and `ar_*` symbol checked against the
> SDK stubs, and the whole frame loop compiled and linked against the same two
> frameworks on macOS 26). Building the *whole crate* for
> `aarch64-apple-visionos` does not work yet, and the reason is upstream: wgpu
> 0.20 predates visionOS, so its `cfg(all(unix, not(ios), not(macos)))` routes
> the target into the Vulkan backend, which pulls `ash` → `libloading 0.7`,
> which has no `RTLD_*` constants for it. It clears when the crate moves to a
> wgpu that knows the target, or when `wgpu` becomes an optional dependency —
> nothing in `src/metal` needs it.

### Native codecs (GIF, APNG, VP9, H.264, HEVC)

Enable `native-codec` for dependency-free encoders (and a GIF decoder) that also build on `wasm32`:

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["native-codec"] }
use threers::{encode_gif, decode_gif, GifEncoder, GifOptions, PaletteMode};

let mut enc = GifEncoder::new(64, 64, 0)
    .colors(64)
    .dither(true)
    .diff_rects(true)
    .palette_mode(PaletteMode::Local);
// enc.add_frame(&rgba, 1, 10);
let bytes = enc.finish();
let (_info, frames) = decode_gif(&bytes).unwrap();
```

See `threers::codec` module docs and `tests/{gif,apng,webm,h264_*,hevc_*}.rs` for VP9/WebM, APNG, H.264/MP4, and HEVC usage.

### Subtitles and captions

`threers::captions` is always available — pure Rust, no dependencies, no font
files, and it builds for `wasm32`. It parses and writes SubRip and WebVTT,
rasterizes cue text, and hands the pixels to whichever path you need.

```bash
cargo run --example captions_render                                          # PNGs: captions over a 3D render
cargo run --release --example captions_video --features "video,native-codec" # every delivery mode
```

**On screen** — the renderer blends the active cue over the frame it just drew:

```rust
use threers::captions::{CaptionOverlay, CaptionTrack};

let track = CaptionTrack::parse(&std::fs::read_to_string("dialogue.vtt")?)?;
let mut overlay = CaptionOverlay::new(track).auto_scale(true);

renderer.render(&mut scene, &camera, &view, false);
renderer.draw_caption_overlay(&mut overlay, time_seconds, width, height, &view, format);
```

The text is rasterized only when the active cue changes, so a frame in the
middle of a cue costs one texture upload — and a frame with no cue costs
nothing.

**In an exported video** — pick how the cues ship:

```rust
use threers::{CaptionMode, CaptionTrack, VideoCodec, VideoExporter};

VideoExporter::new("out.mp4")
    .size(1280, 720).frames(300).fps(30).codec(VideoCodec::H264)
    .captions(CaptionTrack::parse_srt(&srt)?)
    .caption_mode(CaptionMode::BurnAndSidecar)   // pixels *and* out.srt
    .export(render)?;
```

| `CaptionMode` | Result |
|------|--------|
| `Burn` (default) | text composited into the frames — works with every codec, including GIF and APNG |
| `Sidecar` | a separate `out.srt` / `out.vtt` beside the video |
| `Embed` | a soft subtitle track in the container: MP4 `mov_text`, WebM WebVTT |
| `BurnAndSidecar` | both of the first two |

Styling is a `CaptionStyle` — type size, fill, outline, shadow, background box,
alignment, frame anchor, margins, and wrap width — and `for_height` rescales a
1080p-authored style for any output size. Text draws with a built-in 8×8 bitmap
face by default (zero assets, Latin-1 accents fold to their base letter); pass a
`.ttf` to `CaptionFont::from_ttf_bytes` / `VideoOptions::caption_font` for real
typography.

The muxers can also write soft subtitles without ffmpeg:
`codec::mp4::mux_hevc_with_captions` (a `tx3g` timed-text track) and
`codec::webm::mux_webm_with_captions` (a `D_WEBVTT/SUBTITLES` track). Both are
round-trip verified against ffmpeg in `tests/captions_mux.rs`.

In the browser, `renderer.setCaptions(overlay)` makes every `render()` draw the
current cue — see `web/examples/captions.html`:

```js
import THREE, { CaptionTrack, CaptionOverlay } from './web/threejs-shim.js';

const overlay = new CaptionOverlay(CaptionTrack.parse(vttText), { fontSize: 34 });
overlay.setAutoScale(true);
renderer.setCaptions(overlay);
renderer.captionTime = video.currentTime;   // then render() as usual
```

### Animating OpenSCAD models

OpenSCAD animates by re-evaluating the whole program with `$t` stepped from 0 to
1 — geometry is a function of time. `openscad::animate` drives that loop and
renders it.

```bash
cargo run --release --example scad_animate --features "openscad,video,native-codec"
```

```rust
use threers::openscad::animate::{ScadAnimation, ScadCamera, ScadRender};

let mut animation = ScadAnimation::from_file("chuck.scad").frames(120).fps(30);
ScadRender::new(1280, 720)
    .supersample(2)
    .camera(ScadCamera::turntable())
    .export_video(&mut animation, "chuck.mp4")?;
```

- **`color()` survives evaluation.** `Solid::parts()` splits the model into
  separately colored pieces — CSS names, `#rrggbb`, `[r,g,b,a]`, and alpha for
  see-through parts. Booleans distribute over the pieces exactly, so a cut
  through a colored body keeps its color. A model with no `color()` is still one
  part, identical to `to_geometry_exact()`.
- **Cameras that frame the model**: `Auto` fits it exactly (a per-corner frustum
  fit, not a loose bounding sphere), `Turntable` orbits at a fixed distance so
  the model does not breathe, `Viewport` obeys the model's own `$vpr`/`$vpt`/
  `$vpd` — including written as functions of `$t` — and `Fixed` takes an
  explicit eye and target. Z-up, like every model written for OpenSCAD.
- **It avoids the work where it can.** A model that never reads `$t` is
  evaluated once and shared by every frame; the rest are evaluated several at a
  time (`concurrency`, default `min(4, cores)`), which is ~3× on the example
  model. `--features parallel` additionally parallelises the CSG inside a frame.
- **Output**: `render_frames` (RGBA), `render_png`, `export_png_sequence`, and
  `export_video` / `export_video_with` — the latter takes a `VideoOptions`, so
  [subtitles](#subtitles-and-captions) and codec settings come along.

**Colors.** A `ScadPalette` is a background plus a cycle of part colors; eight
ship built in (`studio`, `cornfield`, `metallic`, `sunset`, `midnight`,
`blueprint`, `nature`, `monochrome`) and `ScadPalette::from_hex` takes your own.
`ScadColoring` decides where a part's color comes from:

```rust
use threers::openscad::animate::{ScadColoring, ScadPalette, ScadRender};

ScadRender::new(1280, 720)
    .palette(ScadPalette::blueprint())        // also sets the background
    .coloring(ScadColoring::Palette)          // ignore color(), re-skin the model
    // …or ModelThenPalette: keep color() and dress only the untagged parts
    .part_colors(vec![[1.0, 0.3, 0.2, 1.0], [0.2, 0.5, 0.9, 1.0]]);  // explicit list
```

**Lights.** The default is a three-point studio rig placed *relative to the
camera*, so a turntable never rotates the model into its own shadow — the usual
failure of a fixed rig. The key light casts shadows, sized to the model's bounds.

```rust
use threers::openscad::animate::{ScadLight, ScadLighting, ScadRender};

ScadRender::new(1280, 720)
    // Camera-relative key/fill/rim over a sky-and-ground hemisphere.
    .lighting(ScadLighting::studio().warmth(0.5).ambient(0.3))
    // …or a fixed sun, so the shading says which way a face points:
    .lighting(ScadLighting::sun(215.0, 38.0))
    // …or flat and shadowless, like the OpenSCAD GUI:
    .lighting(ScadLighting::flat())
    // …or exactly what you specify:
    .lighting(ScadLighting::custom(vec![ScadLight::Key {
        color: [1.0, 0.95, 0.9], intensity: 2.0, azimuth: 30.0, elevation: 25.0,
    }]));
```

Ambient light comes from the palette — each scheme carries its own sky and
ground colors, so `blueprint` is lit coolly and `sunset` warmly with no extra
setup.

**Speed** comes in three flavours, because "faster" can mean three things:

| Knob | Changes |
|------|---------|
| `speed(2.0)` / `seconds(4.0)` | how fast it *plays* — same frames, different frame rate |
| `ping_pong(true)` / `easing(…)` | how fast the *model moves* through the loop |
| `quality(ScadQuality::Draft)` | how fast it *renders* — kernel, curve resolution, supersampling |

```rust
use threers::openscad::animate::{ScadAnimation, ScadEasing, ScadQuality};

let animation = ScadAnimation::from_file("chuck.scad")
    .frames(120)
    .seconds(4.0)                       // retime the loop
    .speed(0.5)                         // …then play it at half speed
    .ping_pong(true)                    // out and back, no snap at the loop point
    .easing(ScadEasing::EaseInOut)      // accelerate away from rest, settle at the end
    .quality(ScadQuality::Draft);       // rough it in first
```

Driving a model from Rust rather than `$t` works the same way:

```rust
// Any root-scope name can be the animation variable.
let animation = ScadAnimation::from_file("arm.scad")
    .frames(90)
    .var("SHOULDER", |t| 90.0 * t)
    .var("ELBOW", |t| 45.0 * (1.0 - t));

// …or skip `.scad` entirely and animate the Rust DSL.
let animation = ScadAnimation::from_fn(|t| {
    threers::cube([20.0, 20.0, 20.0]).difference(threers::sphere(4.0 + 8.0 * t as f32))
});
```

### Web (browser)

Build the wasm package and serve the repo root (or use the parity server below):

```bash
# Default (no mesh-bvh / CSG addon)
wasm-pack build --target web --out-dir web/pkg && bash web/post-build.sh

# Or via the feature build scripts:
MESH_BVH=1 web/build.sh          # mesh-bvh addon
BVH_CSG=1 web/build.sh           # CSG addon (implies mesh-bvh)
NATIVE_CODEC=1 web/build.sh      # GIF/APNG/WebM/MP4 browser export bindings
OPENSCAD=1 web/build.sh          # OpenSCAD front end (scad_geometry/scadExport)
```

**OpenSCAD in the browser** (after `OPENSCAD=1` build): the live playground
[`web/openscad-playground.html`](web/openscad-playground.html) parses `.scad` code
to a watertight mesh and renders it with WebGPU, with STL/OBJ/OFF/3MF/GLB download.
See also the demo gallery and the parametric 3D-printer / NEMA-17 assemblies
(`web/openscad-gallery.html`, `web/openscad-printer.html`, `web/nema17.html`).

**Export video in the browser** (after `NATIVE_CODEC=1` build): open
[`web/examples/export-video.html`](web/examples/export-video.html) — render frames, encode GIF/APNG/WebM/MP4 in-process, download the file. No ffmpeg.

JS/TS API (`native-codec` wasm) — prefer `VideoExporter`:

```js
import {
  initThreers,
  VideoExporter,
  assertVideoExportAvailable,
} from '/web/threejs-shim.js';

await initThreers('/web/pkg/threers_bg.wasm');
assertVideoExportAvailable();

const result = await VideoExporter.from(renderer, scene, camera)
  .gif({ transparent: true })
  .fps(15)
  .parallel(3) // pipeline GPU readbacks (1–8)
  .frames(30)
  .update((i, n) => { cube.rotation.y = (i / n) * Math.PI * 2; })
  .onProgress(({ message }) => console.log(message))
  .download('cube');
// or events:
// exporter.on(VideoExportEvent.Progress, (e) => …)
// exporter.on(VideoExportEvent.Complete, ({ detail }) => …)
```

Trace progress with events:

```js
exporter
  .on(VideoExportEvent.Progress, (e) => console.log(e.message, e.ratio))
  .on(VideoExportEvent.Complete, ({ detail }) => console.log(detail.result.summary));
```

Rust (same event model via callbacks):

```rust
use threers::{VideoCodec, VideoExporter, VideoExportEvent};

VideoExporter::new("out/cube.gif")
    .size(320, 240)
    .frames(45)
    .fps(15)
    .codec(VideoCodec::Gif)
    .on_progress(|p| eprintln!("{}", p.message))
    .on_event(|ev| {
        if let VideoExportEvent::Complete { output, .. } = ev {
            eprintln!("wrote {output}");
        }
    })
    .export(|i| { /* return RGBA for frame i */ vec![] })?;
```

Shorthands: `.gif()`, `.apng()`, `.webm({ alpha: true })`, `.mp4()`. Format strings like `"webm-alpha"` and `"h264"` work via `.format(...)` / `parseVideoFormat`. Filenames need no extension (`"cube"` → `"cube.gif"`). Scene size defaults to the canvas; WebM sizes snap to multiples of 8, MP4 to even dimensions. `.parallel(N)` overlaps render + async pixel readback across N targets. `.worker(true)` streams frames into a Web Worker (`video-export-worker.js`) while capture continues.

Buffer-only: `VideoExporter.encode(frames, { width, height, format, … })` or `encodeVideoFramesInWorker(...)` from `/web/video-export.js`.

Headless check (needs Chromium + `NATIVE_CODEC=1` wasm):

```bash
NATIVE_CODEC=1 web/build.sh
cd web && npm run test:video-export
```

Then load `web/examples/index.html` or any page that imports `/web/threejs-shim.js` and calls `initThreers('/web/pkg/threers_bg.wasm')`.

```javascript
import THREE, { initThreers } from '/web/threejs-shim.js';

await initThreers('/web/pkg/threers_bg.wasm');
const renderer = await THREE.WebGLRenderer.create(document.querySelector('canvas'));
// … same patterns as three.js
```

CSG in the browser (after `BVH_CSG=1` build):

```javascript
import { installBvhCsg, Brush, Evaluator, ADDITION } from '/web/bvh-csg-addon.js';
installBvhCsg(THREE);
```

### Earth, sun and lens flare

```bash
cargo run --release --example earth_sun_flare                  # 4K maps
cargo run --release --example earth_sun_flare -- --texture 8192
cargo run --release --example earth_sun_flare -- --frames 120 --video out/earth.mp4
```

A full PBR scene with no downloaded assets. Five equirectangular maps — albedo,
normal, roughness, night lights and clouds — are generated from **one shared
elevation field**, so the coastlines in the colour map, the relief in the normal
map, the shine on the oceans and the shoreline glow of the city lights all line
up. At `--texture 8192` that is five 33.6-Mtexel maps, built in under a second
across cores.

The noise is sampled in 3D on the sphere rather than in UV space, which is what
keeps an equirectangular map free of both a date-line seam and the smearing that
2D noise suffers at the poles. The lens flare and the atmosphere rim are
composited in screen space after the render — a flare is a property of the
camera, not the scene, and doing it there is what lets it be occluded as Earth
passes in front of the sun.

Rendering it needs mip-mapped textures to not shimmer, which the renderer now
builds for every 2D texture.

### Planets (`planet`)

`earth_sun_flare` builds all of that by hand. The `planet` feature is the same
machinery as an API, so a believable planet is a builder rather than a thousand
lines of example:

```rust
// Cargo.toml: threers = { version = "0.0.4", features = ["planet"] }
use threers::planet::{EarthTextures, Planet, Starfield};

let mut scene = threers::Scene::new();
// Real NASA imagery where it is on disk, generated maps for whatever is not.
let maps = EarthTextures::from_dir("web/assets/earth").load();
let earth = Planet::earth().maps(maps).add_to(&mut scene);
Starfield::procedural(2048).add_to(&mut scene);
```

`Planet::add_to` puts three objects in the scene and returns their ids: the
body, a cloud shell above it, and an atmosphere above that. The details that are
easy to get wrong are handled — colour maps tagged sRGB and data maps not,
longitude wrapping while latitude clamps, cloud cover living in the alpha
channel, the night map gating the emissive so city lights do not flood the day
side, and the atmosphere shaded as a volume the view ray passes through rather
than as a surface.

```bash
scripts/fetch-earth-textures.sh --png --sky   # NASA imagery, public domain, gitignored
cargo run --release --features planet --example realistic_earth
cargo run --release --features planet --example realistic_earth -- --relief 0.05
cargo run --release --features planet --example realistic_earth -- --moon
cargo run --release --features planet --example realistic_earth -- --frames 120
```

The browser port (`web/examples/earth.html`) builds the globe from **eight
sphere patches**, each streaming its own NASA tile at 1024, 2048 or 4096 px
depending on how close the camera is and whether the patch faces it. That is a
16384x8192 globe at the top level — a single-texture globe cannot exceed
8192x4096, because that is all WebGPU guarantees for one texture. Sliders drive
the sun's longitude and height, city-light brightness, atmosphere strength,
cloud cover and relief.

The Moon is there at its real distance — 60.3 Earth radii, on a Keplerian
ellipse with the actual eccentricity, inclination and sidereal period, tidally
locked, sharing Earth's clock. That makes it four pixels wide from anywhere that
frames Earth, so the scene opens beside it and flies in: a held shot by the
Moon, a turn to Earth while it is still a distant blue marble, then an eased
descent into orbit that hands over to `OrbitControls`. `replay intro` runs it
again; `?flyover=0` skips it.

Without the download the example still runs: every map falls back to a
procedural one built from a shared elevation field. With it, you get Blue Marble
surface colour, Black Marble city lights, a MODIS cloud composite, relief and
ocean gloss derived from GEBCO elevation, the CGI Moon Kit Moon (LRO colour +
LOLA relief), and the Deep Star Maps sky — the last read straight out of its
ZIP-compressed OpenEXR as half-float, because over half of that map sits below
1/100th of full scale and clipping it to 8 bits leaves a black sky.

## RLX bridge

[RLX](https://crates.io/crates/rlx) is an ML compiler and runtime. The `rlx`
and `rlx-geo` features put it beside the renderer — the renderer is unchanged
either way — and convert between the two crates' data: vertex attributes and
pixels become tensors of a declared shape, tensors become attributes and
textures. Both features are off by default and both build for wasm32 as well as
native.

```toml
# Cargo.toml
threers = { version = "0.0.4", features = ["rlx", "rlx-geo"] }
```

```rust
use threers::rlx::{mesh, preferred_device, ColorGrade, ConvFilter, FitOptions, Kernel3x3};

// A 3×3 convolution, compiled once and run per frame.
let mut sharpen = ConvFilter::new(width, height, Kernel3x3::SHARPEN, preferred_device());
let crisp = sharpen.apply(&frame)?;

// Mesh smoothing as a graph over the [n, 3] positions tensor.
mesh::taubin_smooth(&mut geometry, 12, 0.5, -0.53, preferred_device())?;

// A colour grade *learned* from a reference, by gradient descent.
let report = ColorGrade::fit(&render, &reference, &FitOptions::default(), preferred_device())?;
let graded = report.grade.apply(&other_frame);
```

| Example | Features | What it shows |
|---|---|---|
| `rlx_frame_filter` | `rlx` | A Reinhard tone map as three IR nodes over a rendered frame |
| `rlx_conv_postfx` | `rlx` | Blur / sharpen / Sobel / Laplacian / emboss, plus CPU-vs-GPU timings |
| `rlx_mesh_smooth` | `rlx` | Laplacian against Taubin smoothing, with the volume each one keeps |
| `rlx_fit_grade` | `rlx` | Learns a colour grade from a reference, then transfers it to another shot |
| `rlx_video_grade` | `rlx,video,native-codec` | The compiled grade applied across 60 frames → GIF |
| `rlx_palette` | `rlx,rlx-geo` | k-means palette, posterise, and a Voronoi mosaic snapped to it |
| `rlx_geo_terrain` | `rlx-geo` | Scattered samples → exact Delaunay mesh, plus a Voronoi map |
| `rlx_geo_materials` | `rlx-geo` | Cell textures: albedo, roughness and normals from wall distance |
| `rlx_geo_refine` | `rlx-geo` | Adaptive refinement against a uniform grid at equal vertex budget |
| `rlx_geo_erosion` | `rlx,rlx-geo` | 120 diffusion passes over a terrain, one compiled graph |

```bash
cargo run --release --features rlx --example rlx_fit_grade
cargo run --release --features "rlx,rlx-geo" --example rlx_palette
```

### In the browser

`RLX=1 web/build.sh` (and `RLX_GEO=1`) exports the same work as `rlxConvolve`,
`rlxDiffuse`, `rlxSmoothMesh`, `rlxFitGrade`, `rlxApplyGrade`, `rlxPalette`,
`rlxPosterize`, `geoDelaunay`, `geoHeightfield`, `geoVoronoiLabels`,
`geoVoronoiWallDistance` and `geoNormalMap` — plain typed arrays in and out.

Run them in a worker. `web/rlx-worker.js` is one, and `web/rlx-bench.html`
drives it: every op is tens of milliseconds, which on the main thread is a
dropped frame or twenty. Pixel buffers go in the transfer list rather than
being cloned — a 4K frame is 33 MB, and copying it costs more than filtering
it does.

```js
const worker = new Worker('./rlx-worker.js', { type: 'module' });
worker.postMessage({ id: 1, op: 'convolve',
                     args: { rgba, width, height, kernel: 'sharpen' } },
                   [rgba.buffer]);   // ← transferred, not copied
```

**rlx's GPU backend is available on wasm32 as of the wgpu 30 upgrade.** It used
not to be: rlx was on wgpu 30 while this renderer was on wgpu 0.20, and two
majors in one wasm binary both generate bindings for the same WebGPU interfaces,
which `wasm-bindgen` rejects outright (`duplicate string enums:
GpuAutoLayoutMode`). The Rust compiled — it was the bindgen step that could not
resolve it — so the browser build left rlx's `gpu` feature off. Both are wgpu 30
now, one set of bindings, and `wasm-bindgen` accepts the build. Ask
`rlxDevices()` at startup rather than assuming: whether a given browser then
hands you a working WebGPU adapter is still its own question.

Two numbers from those examples, since both cut against the feature as much as
for it. Adaptive refinement reaches **~9× lower error** than a uniform grid of
the same vertex count — worth having. A 640×640 convolution costs **~10 ms per
frame** through host memory against ~5 ms to render the frame in the first
place — so for effects that belong to the frame, the renderer's own
post-processing chain is still the better tool. What the bridge buys is a
kernel in the same graph as the rest of the pipeline, and one autodiff can see.

### Speed

`preferred_device()` returns the **CPU**
for a reason — the GPU column is what rlx's own priority order would have
picked:

Milliseconds, best of five. Native is an M-series laptop; the browser column is
Chrome 131 running `web/rlx-bench.html`, measured by
`web/scripts/rlx-bench-validate.mjs`. `preferred_device()` returns the **CPU**
for a reason — the GPU column is what rlx's own priority order would pick:

| operation | native CPU | native GPU | Chrome (wasm) |
|---|---|---|---|
| `ConvFilter` gaussian, 512² | 11.3 | 17.6 | 45.3 |
| `Palette::extract`, 8 colours | 29.7 | 67.0 | 34.1 |
| `posterize`, 512² | 22.8 | 47.6 | 53.0 |
| `ColorGrade::fit`, 400 steps | 479 | 29 312 | 481 |
| `Diffusion`, 20 passes, 256² | 3.9 | 15.6 | 6.6 |
| `taubin_smooth`, 8 iterations | 8.9 | 33.3 | 12.5 |
| `voronoi_labels`, 400 sites, 512² | 106 | — | 87 |
| `delaunay`, 5 000 points | 0.9 | — | 8.3 |

Wasm costs 1–4× native across most of the table, and the **fit — the one that
matters — is at parity** (481 ms against 479). `delaunay` is the outlier at 9×,
and that is not the engine: the browser takes rlx-geo's Guibas-Stolfi path
because the Dwyer one calls a clock wasm does not have (see below). Measured on
the same target, that path costs 1.9–2.4× the Dwyer build for identical output.
Run-to-run variance on this machine is up to ~1.5×, and the fit is the noisiest
row.

The GPU column is the finding: every one of these is either small or a chain of
dependent steps, and dispatch latency eats the arithmetic. `fit` is 400 round
trips, hence 61×. Convolution crosses over around four megapixels — at
3840×2160 the GPU takes 373 ms against the CPU's 920 — so pass `Device::Gpu`
explicitly for work that size.

### Why the worker, measured

Five 512² convolutions, driven from the page, counting animation frames:

| | time | frames painted |
|---|---|---|
| on the main thread | 338 ms | **0** |
| in the worker | 341 ms | **22** |

Same work, same wasm, no measurable throughput cost — and the page keeps
painting instead of freezing solid. `node web/scripts/rlx-bench-validate.mjs`
reproduces both columns in headless Chrome.

### Payload

`wasm-bindgen --target web`, bytes, against the same build without the feature:

| build | raw | gzip | brotli |
|---|---|---|---|
| baseline (default features) | 1 368 963 | 395 840 | 300 916 |
| `+rlx-geo` | 1 439 365 | 418 025 | 317 030 |
| `+rlx` | 6 511 319 | 1 476 290 | 976 191 |
| `+rlx,rlx-geo` | 6 583 531 | 1 497 478 | 990 578 |

`rlx-geo` costs **16 KB** brotli — it is a triangulator and a predicate kernel.
`rlx` costs **675 KB** brotli, a 3.2× payload, because it brings a whole ML
compiler: IR, autodiff, fusion, the compile pipeline and a CPU backend. That is
a real decision for a web build, not a rounding error, which is why it is its
own flag and off by default.

## Physics

Rigid-body physics lives in the companion crate
[`threers-physics`](crates/threers-physics), a workspace member:

```toml
[dependencies]
threers = "0.0.4"
threers-physics = "0.0.1"
```

```rust
use threers_physics::prelude::*;

let mut world = World::new();
world.add_body(RigidBody::fixed().shape(Shape::ground()));
let ball = world.add_body(
    RigidBody::dynamic()
        .shape(Shape::ball(0.5))
        .translation(Vector3::new(0.0, 10.0, 0.0))
        .scene_object(node),          // bind it to a scene node
);

world.step(1.0 / 60.0);               // per frame
world.sync_to_scene_interpolated(&mut arena);
```

Shapes, joints, sleeping, collision filtering, sensors, raycasts, shape casts,
inverse kinematics (FABRIK and cyclic coordinate descent), a kinematic character
controller, raycast vehicles and routed tendons. Bodies can opt into continuous
collision detection, so a projectile is swept rather than sampled and cannot
cross a thin wall between two frames. Colliders can be built straight from the
geometry you are already drawing.

```bash
cargo run -p threers-physics --example bouncing_balls
cargo run -p threers-physics --example scene_sync
cargo run -p threers-physics --features gpu --example gpu_broadphase
```

Optional features: `parallel` (rayon narrow phase, native), `gpu` (broad phase on
a wgpu compute shader — works on wasm32/WebGPU, where rayon cannot run), `async`
(tokio), `openscad` (colliders straight from `Solid`s, no STL round trip). Full
documentation in the [crate README](crates/threers-physics/README.md).

### Mechanisms

> `threers-mechanism-tour` is `exclude`d from the workspace: it builds as a
> library, but its examples still call an older shape of its own API, so the
> `cargo run -p threers-mechanism-tour …` lines below need that reconciled first.
> `threers-robot-arm` and `threers-physics-bench` *are* members and build.
> See [docs/releasing.md](docs/releasing.md).

A `.scad` file can declare its own parts, joints and drives, and the assembly
stack reads them: it puts the mechanism together, checks it for interference,
sweeps its travel, runs it, records the run as poses, reduces that to keyframes,
and then asks the geometry whether the declarations were true.

[The mechanism tour](crates/threers-mechanism-tour) — twelve mechanisms in a browser.
(The banner picture is regenerated by `cargo run -p threers-mechanism-tour --example look --docs`,
which needs that crate back in the workspace first.)

[`threers-mechanism-tour`](crates/threers-mechanism-tour) is twelve of them in a
browser — a bench, a pendulum, a gear train, a four-bar, a slider-crank, a
universal joint, a worm and wheel, a Geneva, a cam, a ratchet, a stage and a
latched gate. Four have a closed form to check against and none has it written
down: the pendulum's period lands within 0.04% of `2π√(I/mgd)`, the slider-crank's
piston within 0.006 mm of `R·sinφ + √(L²−R²cos²φ)`, the universal joint's wobble
within 0.01° of `tan θ_out = cos β·tan θ_in`, and the cam's lift within 0.1 mm rms.

```bash
crates/threers-mechanism-tour/build.sh
python3 -m http.server --directory web 8080
open http://localhost:8080/mechanism-tour/
```

Joints have **bearings**: `friction` and `bearing = [mu, radius]` give a mate
Coulomb friction of its own, scaled by what it carries. That is what makes a worm
drive irreversible — driven backwards, a frictionless mesh spins the worm 3,022°
and one at μ 0.05 moves it 0.0°, while driving forward costs 1.2%.

### OpenSCAD parts, without an STL round trip

A `Solid` evaluates straight to the same `BufferGeometry` the renderer draws, so
the collider and the mesh come from one evaluation:

```rust
use threers_physics::scad::SolidPhysics;

let part = SolidPhysics::dynamic()
    .density(2.7)
    .fit(ColliderFit::ConvexHull)
    .build(difference_all(vec![cube([4.0, 2.0, 2.0]), cylinder(3.0, 0.5)]))
    .unwrap();

let (node, body) = part.spawn(&mut arena, Some(root), &mut world, material);
```

### A machine that has to survive its own numbers

[`threers-robot-arm`](crates/threers-robot-arm) is a four-axis pick-and-place
cell that runs the *same* programme on two different machines — one kinematic,
one built from dynamic bodies and PI position servos — so what you are looking
at is the difference between the models rather than between two demos.

```bash
cargo run -p threers-robot-arm --example console       # runs the cycle, prints a report
cargo run -p threers-robot-arm --example performance   # sizes the motors and the links
crates/threers-robot-arm/build.sh                      # browser player -> /robot-arm/
```

It is also where the physics gets held to account. `performance` runs the whole
trajectory, differentiates it for accelerations, and sums
`τ = Σ [(c−p) × m(a−g) + Iα + ω×Iω] · axle` per axis — so the motors are chosen
against measured demand, and every link section is sized against the peak bending
moment *and* torsion at its own root, checked for stiffness and for strength:

```text
axis       motor       holding      peak       rms  headroom
base yaw   NEMA 17      0.00Nm    0.33Nm    0.06Nm      6.9×
shoulder   NEMA 23      1.37Nm    1.89Nm    0.90Nm      2.9×
elbow      NEMA 17      0.56Nm    1.09Nm    0.36Nm      2.1×
wrist      NEMA 11      0.07Nm    0.36Nm    0.05Nm      1.9×

link              section    mass   bending   torsion   margin  sized by
upper arm     20×26  mm    92 g    1.89 Nm    0.38 Nm     4.1×  stiffness
forearm       16×21  mm    58 g    1.09 Nm    0.12 Nm     4.5×  stiffness
wrist         14×14  mm    35 g    0.36 Nm    0.09 Nm     6.6×  printability
```

The loop is closed — the simulation weighs what the optimiser specifies, to
within 1% — so those torques are the torques of the arm as drawn.

### Browser benchmark

```bash
crates/threers-physics-bench/build.sh          # needs wasm-pack
python3 -m http.server --directory web 8080    # then open /physics-bench/
```

Six scenes — stacking, bouncing, chains, orbits, mixed shapes, bullets — with
live step-time, body count and sleep stats, and switchable quality presets.

## Parity compare UI

Side-by-side **three.js r165 vs threers wasm** for 113 regression scenes:

```bash
cd tests/parity
npm install                    # puppeteer, pixelmatch (first time)
node server.js                 # http://localhost:8087
```

Open **http://localhost:8087/** — navigate scenes, toggle light/dark theme, view source (JS / TS / Rust tabs).

```bash
cd tests/parity
node run.js                    # core suite → out/compare-results.json
node run-mesh-bvh.js           # mesh-bvh scenes (MESH_BVH=1 build)
node run-bvh-csg.js            # CSG scenes (BVH_CSG=1 build)
```

CI helpers: `scripts/ci-mesh-bvh.sh`, `scripts/ci-bvh-csg.sh`.

After changing Rust or wasm, rebuild and hard-refresh the browser (⌘⇧R) or click **↻** in the compare UI so cached wasm/shim are busted.

## Project layout

```
src/                 Rust library (scene graph, renderer, loaders, …)
  codec/             Pure-Rust media codecs (`native-codec`)
  video.rs           Frame-sequence export (`video`, native)
  renderer/          wgpu pipelines, post-fx, headless offscreen
  metal/             Metal backend (`metal`, macOS/iOS) — objc runtime, MSL, headless + CAMetalLayer
  metal/visionos/    visionOS (`visionos`) — CompositorServices frame loop, ARKit world tracking
  prelude.rs         `use threers::prelude::*`
  materials/         PBR + ShaderMaterial
  csg/               Rust CSG (`bvh-csg`) — Evaluator, Brush, hierarchy
  mesh_bvh/          Rust BVH (`mesh-bvh`)
  openscad/          OpenSCAD front end (`openscad`) — scad interpreter, Solid/scad! DSL
  exact_csg/         Watertight mesh-arrangement CSG kernel (curved∧curved)
  rlx/               RLX bridge (`rlx`, `rlx-geo`) — tensors, graph runner, Delaunay
  wasm.rs            #[wasm_bindgen] exports (web only)
web/
  threejs-shim.js    THREE-compatible JS API over wasm
  openscad-*.html    OpenSCAD playground / gallery / assemblies
  csg/               JS three-bvh-csg port (loaded when BVH_CSG=1)
  robot-arm/         Scrubbable player for the robot cell
  physics-bench/     Physics benchmark UI
  mesh-bvh-*.js      mesh-bvh addon / stub
  pkg/               wasm-pack output (threers.js, threers_bg.wasm)
  post-build.sh      Patches wasm glue for Safari; generates .d.ts
crates/
  threers-physics/   Rigid bodies, joints, queries, IK, swept CCD, vehicles, tendons
  threers-probe/     Screen-space neural GI from a G-buffer and lighting probes
  threers-continuum/ Tendon-driven continuum robots: elastic-hinge rods, cables
  threers-mechanism-tour/  Twelve .scad mechanisms, assembled and checked
  threers-animation/ Easing, tweens, springs, timelines, ragdoll blending
  threers-js/        npm / Deno / Node ESM publish root (THREE shim + wasm)
  threers-node/      npm native Node addon (napi-rs; per-arch @threers/node-*)
  threers-py/        PyPI publish root (PyO3 / maturin)
  threers-ocean/     Ocean-water demo, as a browser package
  threers-physics-bench/  Browser benchmark (six scenes)
  threers-robot-arm/ Pick-and-place cell + inverse-dynamics sizing
tests/               Codec / video integration tests + parity/
examples/            Native demos (cube, headless_video, shader_material, …)
scripts/             Feature CI (mesh-bvh, bvh-csg)
CHANGELOG.md         Release history
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser: three.js app code (or parity scene HTML)      │
│       ↓ import                                            │
│  web/threejs-shim.js  ──►  wasm WebRenderer / WebScene  │
│  (+ optional mesh-bvh / bvh-csg addons)                 │
└──────────────────────────────┬──────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────┐
│  Rust: Scene → Renderer (wgpu) → surface or RenderTarget │
│        Materials / lights / post-fx (EffectComposer)     │
│        Optional: MeshBvh, CsgEvaluator                   │
│        Optional: HeadlessRenderer → video / native-codec │
└──────────────────────────────────────────────────────────┘
```

- **Native path**: `Renderer::new(device, queue, format)` → `render()` / `render_to()`; or `HeadlessRenderer` for offscreen RGBA.
- **Web path**: `WebRenderer` owns the wgpu surface for a `<canvas>`; render targets and post-fx are registered by integer id across the JS boundary.
- **Post-fx**: A single WGSL module (`src/renderer/shader.rs`) switches on `effect_kind` (copy, FXAA, glitch, bloom, …). The JS `EffectComposer` mirrors three.js pass order.
- **CSG**: Rust `CsgEvaluator` for native/tests; browser parity scenes use the JS port under `web/csg/` with wasm BVH for splits.
- **Codecs**: `native-codec` encodes/decodes in-process (no ffmpeg); `video` shells out to ffmpeg except for GIF/APNG/WebM/H.264 when both features are enabled.

## Development

| Task | Command |
|------|---------|
| Native check | `cargo build` |
| With CSG | `cargo build --features bvh-csg` |
| Video + codecs | `cargo build --features "video,native-codec"` |
| CSG unit/parity tests | `cargo test --features bvh-csg --lib csg` |
| GIF codec tests | `cargo test --features native-codec --lib codec::gif --test gif` |
| H.264 codec tests | `cargo test --features native-codec --test h264_ffmpeg --test h264_animation_parity` |
| H.264 wgpu GPU | `cargo test --features native-codec --test h264_gpu` |
| H.264 Metal GPU | `cargo test --features "metal,native-codec" --test h264_metal` |
| Planet tests (incl. GPU) | `cargo test --features planet --lib planet --test planet_render` |
| RLX bridge tests | `cargo test --features "rlx,rlx-geo" --lib rlx:: --test rlx_bridge` |
| No default features | `cargo build --no-default-features` (drops `captions`) |
| Wasm release | `wasm-pack build --target web --out-dir web/pkg && bash web/post-build.sh` |
| Feature wasm | `BVH_CSG=1` / `MESH_BVH=1` / `NATIVE_CODEC=1 web/build.sh` |
| Physics tests | `cargo test -p threers-physics -p threers-animation` |
| Robot cell | `cargo run -p threers-robot-arm --example console` |
| Robot cell (wasm player) | `crates/threers-robot-arm/build.sh` |
| Parity scenes | `cd tests/parity && node generate-scenes.js` |
| Rust scene snippets (parity UI Rust tab) | `cd tests/parity && node generate-rust-scenes.js` |
| Regenerate shim types | `node web/scripts/generate-shim-types.mjs` (also runs in post-build) |
| Docs | `cargo doc --features "video,native-codec" --open` |
| Release versions | `./scripts/release-versions.sh` |
| Release build/pack | `./scripts/release-all.sh pack` — npm mini+full, node, PyPI wheel |
| Release publish | `PUBLISH=1 ./scripts/release-all.sh publish` — see [docs/releasing.md](docs/releasing.md) |

## Status

Active development toward **pixel parity** with three.js r165 on the core parity scene set. Opt-in **mesh-bvh** and **bvh-csg** suites report 0% pixel diff on their dedicated manifests; hierarchy CSG also has **exact ordered TriKey** topology parity in Rust vs JS.

The **`openscad`** front end is functional end to end: the exact-CSG kernel produces watertight meshes for planar and curved∧curved booleans (verified by cross-op volume consistency + a 2-manifold gate), deferring to the float kernel only on measure-zero degeneracies. Remaining edges: `minkowski`/`color` in the DSL, some 3MF component metadata, and per-boolean cost on very large sequential folds.

Some core scenes are approximate (PMREM, SSR, etc.) — the compare UI marks these and stores diff percentages in `compare-results.json`. See [docs/](docs/) for the roadmap notes and [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
