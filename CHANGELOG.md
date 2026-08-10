# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] — 2026-08-10

### Added
- `openscad` feature — OpenSCAD-style solid modeling, two front ends onto one
  `Solid` CSG tree:
  - **`.scad` interpreter** (`parse_scad` / `parse_scad_file`): primitives,
    transforms, booleans, `linear_extrude`/`rotate_extrude` (twist/scale/slices),
    `hull`/`minkowski`/`offset`/`projection`/`fill`, modules, first-class
    functions, list comprehensions, the `*`/`%`/`!`/`#` modifiers, special
    variables, and builtins including `rands()` and `fill()`.
  - **Rust DSL**: the `Solid` builder, the `scad!` macro, and
    `union!`/`difference!`/`intersection!`/`hull!` combinators; `rotate` (degrees),
    `rotate_axis`, `mirror`, and a `solid()` escape hatch.
  - **Watertight exact-CSG kernel** (`exact_csg`): mesh co-refinement + per-face
    constrained-Delaunay triangulation + winding/ray-parity classification behind
    a closed-2-manifold gate. Resolves genuinely *curved∧curved* booleans
    (sphere∪sphere, cylinder∩cylinder, cone/mixed), which the float kernel cannot;
    falls back to the float kernel only where a result can't be verified (never
    emits an unproven mesh).
  - **Mesh import**: STL, OBJ, OFF, 3MF, AMF, and 2D DXF/SVG; `surface()`
    heightmaps from `.dat` and `.png`.
  - **Mesh export**: STL, OBJ, OFF, 3MF (OPC/ZIP), and binary glTF 2.0 (`.glb`) —
    `Solid::to_stl`/`to_obj`/`to_off`/`to_3mf`/`to_glb`; the `scad2stl` example
    picks format by output extension.
  - **Browser**: a live [OpenSCAD playground](web/openscad-playground.html)
    (edit code → WebGPU render + STL/OBJ/OFF/3MF/GLB download), a demo gallery, and
    parametric 3D-printer / NEMA-17 assemblies. wasm bindings `scad_geometry`,
    `scadExport`, `scad_register_file`.

### Changed
- Renderer + headless: hardware 4× MSAA (`set_msaa`).
- Refactored the OpenSCAD format parsers into a `scad::import` submodule; gated the
  CSG parity/debug scaffolding behind `#[cfg(test)]` (clean, warning-free build).

### Fixed
- Kernel robustness (curved-boolean paths that previously hung or ground):
  - Bounded the dual-BVH `bvhcast` descent — a degenerate input could explore an
    exponential tree of node pairs and hang; now capped with partial-result bail.
  - Rewrote CDT constraint-edge recovery from ~O(n⁴) to O(crossings·n) via an
    edge→triangle adjacency map, with a linear flip cap.
  - Split-fragment and per-face CDT-size guards keep pathological booleans
    terminating — identically on wasm (no thread to time out).

## [0.0.2] — 2026-07-21

### Added
- `native-codec` feature: pure-Rust, wasm-safe media codecs (no ffmpeg / C bindings).
  - HEVC / H.265 encoder and MP4 mux helpers.
  - VP9 encoder (intra + inter ladder) and WebM mux with alpha (`BlockAdditional`).
  - APNG encoder from RGBA frames.
  - Full GIF89a encoder: local/global/auto palettes, octree & median-cut quantization, Floyd–Steinberg dithering, dirty-rect differencing, transparency, disposal modes, lossy indexing, deferred/adaptive LZW clears, comments, interlacing, and incremental `GifWriter`.
  - Native GIF decoder (`GifDecoder` / `decode_gif`) with disposal compositing, interlace, and structured errors.
- `video` feature: `export_video` frame-sequence export via system ffmpeg; with `native-codec`, `VideoCodec::Gif` streams through `GifWriter`.
- Headless RGBA rendering path and related examples/tests for codec round-trips.

### Changed
- Public re-exports for codec and video APIs when the corresponding features are enabled.
- Web package versions (`web/package.json`, `web/pkg`) aligned to the crate version.
- README updated with table of contents, headless/video/codec quick starts, and feature overview.
- Expanded rustdoc on GIF encode/decode, video export, headless rendering, and `ShaderMaterial`.
- Per-format `export_*` examples, `encode_animation_rgba` / `BrowserCodec`, and browser export demo (`NATIVE_CODEC=1`).
- JS/TS video export API: `encodeVideoFrames`, `exportSceneVideo`, `BrowserVideoFormat` (`web/video-export.js` + shim).

## [0.0.1] — 2026-07-16

### Added
- Initial public crate: three.js–shaped wgpu renderer for native and wasm.
- Scene graph, geometries, PBR materials, lights, post-processing, loaders, controls, helpers.
- Opt-in `mesh-bvh` and `bvh-csg` (three-mesh-bvh / three-bvh-csg parity).
- Web shim (`THREE.*`) and parity tooling.

[0.0.3]: https://github.com/eugenehp/threers/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/eugenehp/threers/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/eugenehp/threers/releases/tag/v0.0.1
