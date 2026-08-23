# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] — 2026-08-22

The renderer grew a second way to make a picture, a second backend to make it
on, and a second crate to make things move.

### Added
- **Browser wasm variants** — npm `threers` ships `./mini` (default, smallest)
  and `./full` (`wasm-full`: OpenSCAD/CSG, NURBS, codecs, planet, raytrace) so
  browsers only download the entry they import. `VARIANT=mini|full web/build.sh`.
- **Unified release scripts** — `./scripts/release-versions.sh` and
  `./scripts/release-all.sh` (build / pack / publish) for crates.io, npm ESM,
  npm native, and PyPI; multi-arch via tag push to CI.
- **Meta feature bundles** — `cad`, `media`, `gi`, `apple`, and `full` compose
  existing leaf flags; feature table in `src/lib.rs` now documents `nurbs` /
  `brep*` / `step` / `assembly-check` / `videotoolbox` / `learned-denoise*`.
  Also `wasm-full` for the browser kitchen sink.
- **`threers::wgpu`** — the `wgpu` this crate was built against, re-exported, so
  a caller on the native path cannot end up holding a `Device` from a different
  version of it than the renderer expects.
- **Nested prelude modules** — `prelude::{controls,animation,loaders,helpers}`
  plus feature-gated `csg` / `openscad` / `nurbs` / `raytrace` / `captions`.
- **Native Node package `threers-node`** — napi-rs (`#[napi]`) bindings mirroring
  the Python surface, with multi-arch optional deps (`@threers/node-*`). Deno and
  browsers stay on the wasm ESM package; Neon is not used. See
  [`docs/bindings.md`](docs/bindings.md).
- **Multi-arch release workflow** — `.github/workflows/release-language-packages.yml`
  builds npm ESM (wasm), napi platform binaries (arm64/x64), and PyPI wheels.
- **Published JS package** — `crates/threers-js` is the npm / Deno / Node ESM
  publish root (`threers`), staging the THREE shim + wasm via `./build.sh`.
- **Published Python package** — `crates/threers-py` is the PyPI wheel
  (maturin / PyO3): headless render, math, Tween, and PhysicsWorld.
- **Browser cinematic camera helpers** — `THREE.CameraPath`, `THREE.ShotTimeline`,
  and a richer `THREE.CameraAnimator` (speed ramps, path follow, Vertigo, look/FOV)
  on the three.js-compatible shim. Wasm `WebCamera.setView` batches eye + look-at + FOV
  for cinematic frames. Demo: `web/examples/camera-shots.html`. See
  [`docs/camera-animation.md`](docs/camera-animation.md).
- **General animation stack** — real `PropertyBinding` / `PropertyMixer`, mixer
  `fadeIn` / `fadeOut` / `crossFadeFrom` / `crossFadeTo`, additive blend mode,
  morph / material / bone path binding, `TrackModifier` (noise / cycles / stepped),
  plus `Tween`, `Spring`, `Timeline` (markers + time remap), `ObjectSpring`, and
  `PoseBlend`. Rust mixer gains weights, fades, cross-fade, morph weights, and
  glTF `weights` channels. Demo: `web/examples/animation-demo.html`.
- **Animation ↔ physics handoff** — `PhysicsBlend` freezes the limp pose (no
  yank from a continuing clip), inherits stride velocity, clears velocity on
  recover, and writes via `apply_to_scene`. `World::sync_to_scene_where` skips
  bodies the blend still owns so scene sync cannot undo the crossfade.
  Example: `cargo run -p threers-animation --example anim_physics_blend --features physics`.
- **Path tracing (`raytrace`)** — a physically based integrator beside the
  raster renderer: global illumination, area lights, refraction and depth of
  field, running on the CPU or as a wgpu compute pass (`pathtrace.wgsl`).
  Multiple-importance sampling, a BSDF set, HDRI environment lighting, and a
  progressive film that can be read back mid-render.
  - **Denoising** — an À-Trous edge-avoiding filter guided by albedo and normal
    G-buffers, and behind `learned-denoise`, a trained U-Net run natively
    (`learned-denoise-metal` / `learned-denoise-gpu` pick the backend).
- **NURBS, B-rep and STEP** — a four-step ladder, each implying the one below:
  - `nurbs`: rational curves and surfaces, knot insertion/refinement,
    derivatives, and tessellation.
  - `brep`: analytic surface provenance carried on geometry, so a face
    remembers the plane, cylinder or sphere it was cut from.
  - `brep-csg`: closed-form surface/surface-intersection fast paths in the CSG
    kernel, taken when both operands are analytic.
  - `brep-kernel`: real B-rep topology with its own boolean and fillets.
  - `step`: AP203/214 import and export over a Part 21 reader/writer.
- **Metal backend (`metal`)** — a second renderer talking to Metal through the
  Objective-C runtime on macOS/iOS, with its own headless path. `visionos` adds
  CompositorServices and ARKit on top.
- **`videotoolbox`** — in-process VideoToolbox encode on macOS, no ffmpeg
  process and no temporary frame directory.
- **Captions (`captions`, on by default)** — SRT and WebVTT parsing, layout and
  rasterization, drawn as an overlay pass or burned into an export.
- **H.264/MP4** in `native-codec`, joining the existing HEVC, VP9/WebM, APNG and
  GIF encoders — still pure Rust, still wasm-safe.
- **Planets (`planet`)** — planet and moon surfaces, starfields, and baked map
  generation, including NASA imagery ingest.
- **RLX bridge (`rlx`, `rlx-geo`)** — tensors to and from geometry and pixels, a
  graph runner, convolution post-fx, mesh smoothing, palette extraction, and
  exact Delaunay with adaptive refinement and Voronoi cell textures.
- **Geometry** — kirigami corrugations; a lattice family (struts, TPMS,
  cuboctahedral cells, voxel meshing and infill); rigid origami with crease
  patterns, folding and development sheets; NURBS geometry.
- **Kinematics** — chains, actuators, servos, transmissions and a plant model,
  with material and design helpers.
- **`assembly-check`** — body correspondence, axis recovery and swept-volume
  interference checks on an assembly's motion.
- **`parallel` / `async`** — rayon-backed CSG and BVH work (a deliberate no-op
  on wasm32, where rayon has no threads), and an optional tokio runtime for
  callers driving loaders concurrently.
- **`manifold`** — the Manifold kernel as an alternative backend behind the
  OpenSCAD booleans.
- **Post-processing** — bloom, SSAO, and a downsample pass; BC1 texture
  compression.
- **`threers::prelude`** — the common imports in one `use`.
- **Companion crates**:
  - [`threers-physics`](crates/threers-physics) — rigid bodies, shapes, joints,
    sleeping, collision filtering, sensors, raycasts and shape casts, FABRIK/CCD
    inverse kinematics, a kinematic character controller, and colliders built
    straight from a `Solid`. Optional `parallel`, `gpu` (a wgpu compute broad
    phase that works on wasm32/WebGPU, where rayon cannot), `async`, `assembly`
    and `openscad`.
  - [`threers-probe`](crates/threers-probe) — screen-space neural GI from a
    G-buffer and lighting probes, trained in RLX.
  - [`threers-animation`](crates/threers-animation) — easing, tweens, springs,
    timelines, scene-node clips, and **cinematic cameras** with Blender-parity
    extras (Track/Damped/Locked, Follow Path+tilt, F-curve modifiers, NLA,
    marker binds, drivers, lens shift, panoramics, sensor fit, DoF blades,
    stereo pivots, guides, walk/fly record, time remap, rolling shutter).
    `OrbitControls` gained damping + auto-rotate. Not published with 0.0.4;
    see [docs/releasing.md](docs/releasing.md).

### Changed
- **BREAKING (native Rust callers): wgpu 0.20 → 30.** `Renderer::new` takes a
  `wgpu::Device`, a `wgpu::Queue` and a `wgpu::TextureFormat`, and 36 other
  public functions take or return `wgpu` types. A Rust type belongs to the exact
  crate *version* it came from, so `wgpu::Device` from 0.20 and `wgpu::Device`
  from 30 are unrelated types that print identically — and cargo builds both into
  one graph without complaint, because they are semver-incompatible and allowed
  to coexist. A caller who stays on 0.20 therefore gets `expected `wgpu::Device`,
  found `wgpu::Device`` with nothing on the line to explain it. Move your own
  `wgpu` to 30, or better, use the `threers::wgpu` re-export added below and stop
  having a second copy to keep in step. `HeadlessRenderer` and the wasm/JS paths
  are unaffected: they never name a `wgpu` type.

  Ten major versions in one step, and it removes the last
  use of the unmaintained `block 0.1.6` from the tree: wgpu's Metal backend
  moved from `metal-rs` to `objc2` in 29, so the soundness warning that shipped
  with every macOS build of 0.0.3 is simply gone. It also collapses a duplicate
  — with `rlx` enabled the build compiled wgpu 0.20 *and* wgpu 30 side by side,
  two complete graphics stacks, and now compiles one.

  What changed on the surface: the `ImageCopy*` types are `TexelCopy*`,
  `Maintain` is `PollType` and returns a `Result`, `request_adapter` returns a
  `Result` rather than an `Option`, `present` moved from the surface texture to
  the queue, `get_current_texture` returns a `CurrentSurfaceTexture` enum
  instead of a `Result`, mapping a buffer range is fallible, and `BufferViewMut`
  is write-only because mapped memory may be write-combining.

  One thing was a real bug the upgrade exposed rather than caused: the main
  pass declares 20 sampled textures in its fragment stage while the headless
  device asked for `downlevel_defaults()`, which allows 16. wgpu 0.20 never
  checked; wgpu 30 does. The device now asks for the binding counts the adapter
  actually has, and the renderer has presumably been over that line since it
  was written.

  It also unlocks something that was blocked on the version skew: rlx's GPU
  backend is now available on wasm32. Two wgpu majors in one wasm binary both
  generate bindings for the same WebGPU interfaces, and `wasm-bindgen` rejected
  that outright, so the browser build had rlx pinned to its CPU backend. One
  major, one set of bindings, and the build goes through.
- `default` now includes `captions`; `video` implies it.
- `Vector3` gained `RIGHT` and `FORWARD` beside the existing `UP`.
- `KirigamiPreset::all` and `::count` are `const fn`, so a preset count can size
  an array.
- The crate `exclude`s `tests/parity/scenes/**` — 22 MB of reference dumps that
  only the parity harness reads, and that were counting against the package.
- The repository is a Cargo workspace; the library remains its root member.

### Fixed
- **`manifold` backend: a boolean could return the wrong solid, confidently.**
  The backend verified its *result* and not its *operands*, and a mesh Manifold
  declines to import behaves as an empty solid rather than as an error. So a
  subtraction whose second operand failed to import returned the first operand
  untouched — and since that operand was closed to begin with, the
  watertightness gate downstream had nothing to object to. A `sphere($fn=32)`
  bitten out of a cube came back as the whole cube. The operands are now checked
  on the way in, and a boolean that cannot be trusted falls through to the
  arrangement kernel as it was always meant to.
- **Shadows were lost entirely in any scene without an environment map.**
  Raising `map_size` reallocates the shadow depth texture, and the environment
  bind group that samples it was invalidated by clearing its two cache keys to
  `None`. A scene with no environment map already has `None` for both, so the
  `!=` guard detected nothing, the group was never rebuilt, and every frame
  sampled the *old* texture — one that nothing renders into. The effect was that
  a shadow-casting directional light lost its own contribution on the surfaces it
  lit: a rig with several lights merely looked dim, and a single-sun rig looked
  unlit. Invalidation is now an explicit flag, because it cannot be expressed by
  clearing keys that were already empty.
- A point light's six shadow cube faces all rendered with the last face's
  view-projection. The faces were encoded into one command buffer while writing
  their matrices into one shared uniform buffer, and `queue.write_buffer` lands
  before *any* submitted command — so the last write won for all six. Each face
  now has its own uniform buffer and bind group.
- `ShadowSettings::normal_bias` is read. It was declared, documented, and then
  never looked at — the same failure the renderer's own note about `map_size`
  describes: a caller sets it, sees no error, and gets nothing. The shadow
  lookup now steps off the surface along its normal before sampling.
- The OpenSCAD render's shadow frustum is fitted to the model instead of
  spanning `0.01 .. 8 × reach`. The depth bias is applied in NDC, so what a
  given bias is worth depends on how much world depth the near–far range covers;
  most of that range was empty space.
- **Two features did not compile on their own.** `native-codec` uses
  `crate::captions` in its muxers without depending on it, and `brep` calls the
  surface/surface intersector that only `brep-csg` compiles. Both are invisible
  from `--all-features` and from the default set — `captions` is a default, and
  anything that enables `brep` in a normal build tends to enable `brep-csg` too —
  so they only fail for the person who asks for one feature and nothing else,
  which is exactly what `default-features = false` is for. `native-codec` now
  depends on `captions` as `video` already did, and the `brep` curve falls back
  to its own parameterisation when the intersector is not there.
- **`Solid::parts` appended its pieces where it should have united them.**
  `parts` splits a model by `color()` and distributes the booleans over the
  groups, which is exact — `(a ∪ b) − c` becomes `(a − c)` and `(b − c)` — and
  then has to put the pieces sharing a colour back together. It joined them with
  a triangle-soup concatenation instead of a union, so geometry the model had
  merged came back as separate overlapping shells: an uncolored `cube ∪ cube`
  yielded 72 vertices of z-fighting where `to_geometry_exact` gives 36. Only the
  display path was affected — the export path was always correct — which is why
  it showed as shimmering interior faces and twice the triangles rather than a
  wrong mesh on disk. Each colour group is now folded back into one `Solid` and
  evaluated once, as its own documentation always said it did.
- Rigid origami: `fold_twist` walked its candidate crease assignments but `?`
  returned from the whole function on the first one that would not fold, so a
  later, foldable assignment was never reached.

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

[0.0.4]: https://github.com/eugenehp/threers/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/eugenehp/threers/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/eugenehp/threers/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/eugenehp/threers/releases/tag/v0.0.1
