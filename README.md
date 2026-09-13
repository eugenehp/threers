# threers

A **drop-in three.js replacement** for Rust and the browser, backed by [wgpu](https://github.com/gfx-rs/wgpu). The same Rust core runs natively (winit) and as WebAssembly, with a JavaScript shim that exposes the familiar `THREE.*` API so existing three.js code can run with minimal changes.

**Current version:** [0.0.6](CHANGELOG.md) · [Changelog](CHANGELOG.md)

## Features

The one-line version. Each links to where it is explained; the long form of this
list is in [docs/features.md](docs/features.md).

**Core** — scene graph, buffer geometries, PBR materials, lights, shadows, fog,
custom WGSL via `ShaderMaterial`, and a wgpu renderer with post-processing
(FXAA, bloom, SSAO, glitch, halftone, …). Loaders for glTF, OBJ, STL, PLY,
Collada, FBX, HDR and EXR, plus animation, controls and helpers.

| | |
|---|---|
| [Extended PBR](docs/materials.md) | Clearcoat, transmission, IOR, dispersion, anisotropy, sheen, iridescence, volume absorption, measured presets |
| [HDR pipeline](docs/materials.md) | `Rgba16Float`, `.hdr` decoding, environment prefiltering, ACES tone mapping, area lights |
| [Headless render](docs/headless-render.md) | Offscreen render to packed RGBA — no window, no display |
| [Video export](docs/codecs.md) | `video` via ffmpeg, or pure-Rust `native-codec`: GIF, APNG, VP9/WebM, H.264, HEVC |
| [Subtitles](docs/captions.md) | SRT and WebVTT — on-screen, burned in, or as soft tracks |
| [Vector SVG](docs/svg.md) | `SvgRenderer` draws a `Scene` as vectors; `Path`/`Shape` read SVG path data |
| [USD](docs/usd.md) (`usd`) | `.usda`, `.usdc` and `.usdz`, read **and written**, in pure Rust |
| [Lattices](docs/lattices.md) | 35 generators, foams, conformal cells, and properties from a solve |
| [Kirigami](docs/kirigami.md) | Plate lattices and corrugations that fold from a flat sheet |
| [Planets](docs/planets.md) (`planet`) | Atmospheres, starfields, map generation, Earth and lens flare |
| [Path tracing](docs/path-tracing.md) (`raytrace`) | Global illumination, area lights, refraction, depth of field — CPU and GPU |
| [Mesh BVH / CSG](docs/mesh-bvh-csg.md) (`mesh-bvh`, `bvh-csg`) | Accelerated raycast and shapecast; boolean ops with TriKey parity |
| [OpenSCAD](docs/openscad.md) (`openscad`) | `.scad` front end, `Solid` DSL, a watertight exact-CSG kernel, and animation |
| [NURBS and B-rep](docs/brep-nurbs-plan.md) (`nurbs`, `brep`, `step`) | Curved surfaces kept as surfaces, analytic booleans, ISO 10303-21 read and write |
| [Metal and visionOS](docs/apple-platforms.md) (`metal`, `visionos`) | A second renderer on Metal, and the CompositorServices frame loop |
| [RLX bridge](docs/rlx.md) (`rlx`, `rlx-geo`) | Tensors, learned colour grading, and exact Delaunay/Voronoi |
| [Physics](docs/physics.md) | `threers-physics`: rigid bodies, joints, mechanisms |
| [Bindings](docs/bindings.md) | Published to npm (ESM + Node) and PyPI |

**Runs where you do** — the same core natively on winit and in the browser as
WebAssembly, with `web/threejs-shim.js` exposing the `THREE.*` API for three.js
r165. `parallel` adds rayon-backed BVH and CSG work natively, order-preserving
so results are unchanged.

## Examples

39 examples render to an image; every one of them is in
**[docs/examples.md](docs/examples.md)** with its picture and the command that
produces it. A few of them:

|   |   |   |
|---|---|---|
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/realistic_earth.jpg" width="270" alt="realistic_earth"></a> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_engineering.jpg" width="270" alt="lattice_engineering"></a> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/path_trace_gpu.jpg" width="270" alt="path_trace_gpu"></a> |
| **`realistic_earth`** | **`lattice_engineering`** | **`path_trace_gpu`** |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/openscad_render.jpg" width="270" alt="openscad_render"></a> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/kirigami.jpg" width="270" alt="kirigami"></a> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_fit_grade.jpg" width="270" alt="rlx_fit_grade"></a> |
| **`openscad_render`** | **`kirigami`** | **`rlx_fit_grade`** |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/origami_miura.jpg" width="270" alt="origami_miura"></a> |  |  |
| **`origami_miura`** |  |  |

All headless — no window, no display — so they run the same on a laptop and
in CI. The rest of `examples/` is interactive, exports video, or prints
numbers rather than drawing.
<!-- examples-gallery:end -->


## Documentation

The guides live in [`docs/`](docs/). Each is self-contained; start with
[getting started](docs/getting-started.md).

| | |
|---|---|
| [Getting started](docs/getting-started.md) | A window, a scene, and the prelude — natively |
| [Web](docs/web.md) | The same core as WebAssembly, and the `THREE.*` shim |
| [Examples](docs/examples.md) | All 39 headless examples, with pictures |
| **Geometry** | |
| [Mesh BVH and CSG](docs/mesh-bvh-csg.md) | Accelerated raycast, shapecast, boolean ops |
| [OpenSCAD](docs/openscad.md) | `.scad` front end, the `Solid` DSL, and animating both |
| [Lattices](docs/lattices.md) | Foams, conformal cells, and numbers you can size a part with |
| [Kirigami](docs/kirigami.md) | Corrugations that fold from a flat sheet |
| [B-rep and NURBS](docs/brep-nurbs-plan.md) | Curved surfaces, STEP, and the exact kernel |
| **Look** | |
| [Materials](docs/materials.md) | Metals, glass, anisotropy, thin film, HDR environments |
| [Path tracing](docs/path-tracing.md) | Global illumination on CPU and GPU |
| [Planets](docs/planets.md) | Earth, atmospheres, starfields, lens flare |
| **Output** | |
| [Headless render and video](docs/headless-render.md) | Frames without a window |
| [Native codecs](docs/codecs.md) | GIF, APNG, VP9, H.264, HEVC — pure Rust |
| [Subtitles and captions](docs/captions.md) | SRT and WebVTT, on-screen or burned in |
| [Vector SVG](docs/svg.md) | The scene graph as vectors, and `Path`/`Shape` |
| [USD](docs/usd.md) | `.usda`, `.usdc` and `.usdz`, read and written |
| **Interchange and platforms** | |
| [Animation](docs/animation.md) · [Camera](docs/camera-animation.md) · [Controls](docs/controls.md) | Clips, tracks, and moving the camera |
| [Metal and visionOS](docs/apple-platforms.md) | The Apple backends |
| [Bindings](docs/bindings.md) | npm, PyPI and the Node addon |
| [Physics](docs/physics.md) · [RLX](docs/rlx.md) | The companion crates |
| **Project** | |
| [Architecture](docs/architecture.md) | Layout, how it fits together, and development |
| [Parity](docs/parity.md) | Comparing against three.js |
| [Releasing](docs/releasing.md) | How a version ships |

## Status

Active development toward **pixel parity** with three.js r165 on the core parity scene set. Opt-in **mesh-bvh** and **bvh-csg** suites report 0% pixel diff on their dedicated manifests; hierarchy CSG also has **exact ordered TriKey** topology parity in Rust vs JS.

The **`openscad`** front end is functional end to end: the exact-CSG kernel produces watertight meshes for planar and curved∧curved booleans (verified by cross-op volume consistency + a 2-manifold gate), deferring to the float kernel only on measure-zero degeneracies. Remaining edges: `minkowski`/`color` in the DSL, some 3MF component metadata, and per-boolean cost on very large sequential folds.

Some core scenes are approximate (PMREM, SSR, etc.) — the compare UI marks these and stores diff percentages in `compare-results.json`. See [docs/](docs/) for the roadmap notes and [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
