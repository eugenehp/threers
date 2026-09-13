# Web (browser)

Part of the [threers](../README.md) documentation.

# Web (browser)

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
[`web/openscad-playground.html`](../web/openscad-playground.html) parses `.scad` code
to a watertight mesh and renders it with WebGPU, with STL/OBJ/OFF/3MF/GLB download.
See also the demo gallery and the parametric 3D-printer / NEMA-17 assemblies
(`web/openscad-gallery.html`, `web/openscad-printer.html`, `web/nema17.html`).

**Export video in the browser** (after `NATIVE_CODEC=1` build): open
[`web/examples/export-video.html`](../web/examples/export-video.html) — render frames, encode GIF/APNG/WebM/MP4 in-process, download the file. No ffmpeg.

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
