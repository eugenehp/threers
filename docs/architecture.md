# Project layout and architecture

Part of the [threers](../README.md) documentation.

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
  connectome/        Connectome node-cloud viewer (page + viewer.js)
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
  threers-connectome/  Both fly connectomes as a browser node cloud + tree reader
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
| Release publish | `PUBLISH=1 ./scripts/release-all.sh publish` — see [docs/releasing.md](releasing.md) |
