# threers

A **drop-in three.js replacement** for Rust and the browser, backed by [wgpu](https://github.com/gfx-rs/wgpu). The same Rust core runs natively (winit) and as WebAssembly, with a JavaScript shim that exposes the familiar `THREE.*` API so existing three.js code can run with minimal changes.

**Current version:** [0.0.3](CHANGELOG.md) · [Changelog](CHANGELOG.md)

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
  - [Native (desktop)](#native-desktop)
  - [Optional: mesh-bvh / CSG](#optional-mesh-bvh--csg)
  - [OpenSCAD solid modeling](#openscad-solid-modeling)
  - [Headless render & video export](#headless-render--video-export)
  - [Native codecs (GIF, APNG, VP9, HEVC)](#native-codecs-gif-apng-vp9-hevc)
  - [Web (browser)](#web-browser)
- [Parity compare UI](#parity-compare-ui)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Development](#development)
- [Status](#status)
- [License](#license)

## Features

- Scene graph (`Object3D`, `Scene`, transforms, layers)
- Buffer geometries, PBR materials, lights, shadows, fog
- Custom WGSL materials via `ShaderMaterial`
- wgpu renderer with post-processing (FXAA, bloom, SSAO, glitch, halftone, …)
- Loaders (glTF, OBJ, STL, HDR, …), animation, controls, helpers
- **Headless** offscreen render → tightly packed RGBA (`HeadlessRenderer`, native)
- **Video export** (`video`) — frame sequence → ffmpeg; native GIF/APNG when `native-codec` is on
- **Native codecs** (`native-codec`) — pure-Rust GIF (encode/decode), APNG, VP9/WebM, HEVC/MP4; wasm-safe browser download
- **Opt-in mesh BVH** (`mesh-bvh`) — accelerated raycast / shapecast (three-mesh-bvh–compatible)
- **Opt-in CSG** (`bvh-csg`) — boolean ops on `BufferGeometry` (three-bvh-csg–compatible); Rust native + JS addon
- **Opt-in OpenSCAD** (`openscad`) — a `.scad` interpreter **and** a `Solid`/`scad!` Rust DSL, evaluated by a pure-Rust **watertight exact-CSG kernel** (resolves curved∧curved booleans, float fallback where unverifiable). Mesh import/export (STL/OBJ/OFF/3MF/AMF/glTF-GLB, DXF/SVG, `.dat`/`.png` heightmaps) and a live browser playground.
- **Web**: `web/threejs-shim.js` + wasm — drop-in `THREE.*` replacement targeting three.js r165
- **Native**: winit examples for desktop development and debugging

## Quick start

### Native (desktop)

Requires Rust 1.75+, a working wgpu backend (Metal/Vulkan/DX12), and dev dependencies from `Cargo.toml`.

```bash
cargo run --example cube          # spinning PBR cube
cargo run --example scene_graph   # hierarchy + lights
cargo run --example controls_orbit
cargo run --example shader_material
```

### Optional: mesh-bvh / CSG

```bash
# BVH picking demo
cargo run --example mesh_bvh_picking --features mesh-bvh

# Hierarchy CSG (exact TriKey parity with JS three-bvh-csg@0.0.16)
cargo run --example bvh_csg_hierarchy --features bvh-csg
cargo run --example bvh_csg_steps --features bvh-csg -- 2 --live
```

```rust
// Cargo.toml: threers = { version = "0.0.3", features = ["bvh-csg"] }
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
// Cargo.toml: threers = { version = "0.0.3", features = ["openscad"] }
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

### Headless render & video export

Native-only. `HeadlessRenderer` owns the wgpu device and an offscreen target; `export_video` pipes RGBA frames to ffmpeg (or in-process GIF/APNG when `native-codec` is enabled).

```bash
cargo run --release --example headless_video --features video   # H.264 demo
```

Per-format examples (optional `--out PATH`):

| Example | Features | Output |
|---------|----------|--------|
| `export_h264` | `video` | `.mp4` (libx264) |
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
// Cargo.toml: threers = { version = "0.0.3", features = ["video"] }
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

### Native codecs (GIF, APNG, VP9, HEVC)

Enable `native-codec` for dependency-free encoders (and a GIF decoder) that also build on `wasm32`:

```rust
// Cargo.toml: threers = { version = "0.0.3", features = ["native-codec"] }
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

See `threers::codec` module docs and `tests/{gif,apng,webm,hevc_*}.rs` for VP9/WebM, APNG, and HEVC usage.

### Web (browser)

Build the wasm package and serve the repo root (or use the parity server below):

```bash
# Default (no mesh-bvh / CSG addon)
wasm-pack build --target web --out-dir web/pkg && bash web/post-build.sh

# Or via the feature build scripts:
MESH_BVH=1 web/build.sh          # mesh-bvh addon
BVH_CSG=1 web/build.sh           # CSG addon (implies mesh-bvh)
NATIVE_CODEC=1 web/build.sh      # GIF/APNG/WebM browser export bindings
OPENSCAD=1 web/build.sh          # OpenSCAD front end (scad_geometry/scadExport)
```

**OpenSCAD in the browser** (after `OPENSCAD=1` build): the live playground
[`web/openscad-playground.html`](web/openscad-playground.html) parses `.scad` code
to a watertight mesh and renders it with WebGPU, with STL/OBJ/OFF/3MF/GLB download.
See also the demo gallery and the parametric 3D-printer / NEMA-17 assemblies
(`web/openscad-gallery.html`, `web/openscad-printer.html`, `web/nema17.html`).

**Export video in the browser** (after `NATIVE_CODEC=1` build): open
[`web/examples/export-video.html`](web/examples/export-video.html) — render frames, encode GIF/APNG/WebM in-process, download the file. No ffmpeg.

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

Shorthands: `.gif()`, `.apng()`, `.webm({ alpha: true })`. Format strings like `"webm-alpha"` work via `.format(...)` / `parseVideoFormat`. Filenames need no extension (`"cube"` → `"cube.gif"`). Scene size defaults to the canvas; WebM sizes snap to multiples of 8. `.parallel(N)` overlaps render + async pixel readback across N targets. `.worker(true)` encodes in a Web Worker (`video-export-worker.js`).

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
  materials/         PBR + ShaderMaterial
  csg/               Rust CSG (`bvh-csg`) — Evaluator, Brush, hierarchy
  mesh_bvh/          Rust BVH (`mesh-bvh`)
  openscad/          OpenSCAD front end (`openscad`) — scad interpreter, Solid/scad! DSL
  exact_csg/         Watertight mesh-arrangement CSG kernel (curved∧curved)
  wasm.rs            #[wasm_bindgen] exports (web only)
web/
  threejs-shim.js    THREE-compatible JS API over wasm
  openscad-*.html    OpenSCAD playground / gallery / assemblies
  csg/               JS three-bvh-csg port (loaded when BVH_CSG=1)
  mesh-bvh-*.js      mesh-bvh addon / stub
  pkg/               wasm-pack output (threers.js, threers_bg.wasm)
  post-build.sh      Patches wasm glue for Safari; generates .d.ts
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
- **Codecs**: `native-codec` encodes/decodes in-process (no ffmpeg); `video` shells out to ffmpeg except for GIF when both features are enabled.

## Development

| Task | Command |
|------|---------|
| Native check | `cargo build` |
| With CSG | `cargo build --features bvh-csg` |
| Video + codecs | `cargo build --features "video,native-codec"` |
| CSG unit/parity tests | `cargo test --features bvh-csg --lib csg` |
| GIF codec tests | `cargo test --features native-codec --lib codec::gif --test gif` |
| Wasm release | `wasm-pack build --target web --out-dir web/pkg && bash web/post-build.sh` |
| Feature wasm | `BVH_CSG=1` / `MESH_BVH=1` / `NATIVE_CODEC=1 web/build.sh` |
| Parity scenes | `cd tests/parity && node generate-scenes.js` |
| Rust scene snippets (parity UI Rust tab) | `cd tests/parity && node generate-rust-scenes.js` |
| Regenerate shim types | `node web/scripts/generate-shim-types.mjs` (also runs in post-build) |
| Docs | `cargo doc --features "video,native-codec" --open` |

## Status

Active development toward **pixel parity** with three.js r165 on the core parity scene set. Opt-in **mesh-bvh** and **bvh-csg** suites report 0% pixel diff on their dedicated manifests; hierarchy CSG also has **exact ordered TriKey** topology parity in Rust vs JS.

The **`openscad`** front end is functional end to end: the exact-CSG kernel produces watertight meshes for planar and curved∧curved booleans (verified by cross-op volume consistency + a 2-manifold gate), deferring to the float kernel only on measure-zero degeneracies. Remaining edges: `minkowski`/`color` in the DSL, some 3MF component metadata, and per-boolean cost on very large sequential folds.

Some core scenes are approximate (PMREM, SSR, etc.) — the compare UI marks these and stores diff percentages in `compare-results.json`. See [PLAN.md](PLAN.md) for feature roadmap notes and [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
