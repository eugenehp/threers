# scroll-video

Map page scroll to video playback time, with an optional threers 3D overlay composited on top.

## Modules

| File | Role |
|------|------|
| `scroll-sync.js` | Scroll → time mapping, markers, JSON export |
| `video-composite.js` | `VideoSceneComposite` — canvas 3D or HTML-layer display |
| `video-element-texture.js` | Fast video → GPU texture upload |
| `clip-loader.js` | Load bundled or user video into a decode `<video>` |
| `paint-clip.js` | Procedural fallback frames |
| `assets/calibration-clip.mp4` | Bundled 10s H.264 test clip (Safari-safe) |

## Quick start

```html
<video id="v" playsinline muted preload="metadata" hidden></video>
<canvas id="c"></canvas>
<script type="module">
import THREE, { initThreers } from '../threejs-shim.js';
import { ScrollVideoSync } from './scroll-sync.js';
import { VideoSceneComposite } from './video-composite.js';
import { loadStageVideo } from './clip-loader.js';

await initThreers({ module_or_path: '../pkg/threers_bg.wasm' });
const video = document.getElementById('v');
const canvas = document.getElementById('c');
const duration = await loadStageVideo(video);

const sync = new ScrollVideoSync({
  mode: 'progress',
  scrollStart: 0,
  scrollEnd: 3000,
  videoDuration: duration,
  clipOut: duration,
});

const composite = new VideoSceneComposite({ canvas, video });
await composite.initRenderer();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.z = 6.8;
composite.setCamera(camera);

function frame(now) {
  requestAnimationFrame(frame);
  sync.update({ scrollY: window.scrollY, now });
  sync.applyToVideo(video);
  composite.updateVideoFrame(sync.time);
  composite.render();
}
frame(performance.now());
</script>
```

See also: `example.html` (minimal production page), `calibration.html` (full tuning UI).

## ScrollVideoSync

### Modes

- **`progress`** — scroll position maps linearly (optional smoothstep easing) to clip time.
- **`velocity`** — scroll speed integrates playback time; video may play while moving.

### Markers

Place elements in the document:

```html
<section data-scroll-marker data-time="3" data-label="beat">…</section>
```

Read and fit:

```js
import { readScrollMarkers, autoFitScrollRange, attachMarkerResizeObserver } from './scroll-sync.js';

const scrollRoot = document.getElementById('copy-scroll');
sync.refreshMarkers(document, { scrollRoot });
autoFitScrollRange(sync, sync.markers);
attachMarkerResizeObserver(sync, scrollRoot, document);
```

### JSON export

`sync.toJSON()` / `ScrollVideoSync.fromJSON()` — include in your page bundle:

```json
{
  "mode": "progress",
  "scrollStart": 0,
  "scrollEnd": 4000,
  "videoDuration": 10,
  "clipIn": 0,
  "clipOut": 10,
  "pxPerSecond": 900,
  "velocityGain": 1,
  "smoothing": 0.18,
  "easing": false,
  "markers": []
}
```

## VideoSceneComposite display modes

| Mode | Video | Canvas |
|------|-------|--------|
| `canvas-3d` | Hidden decode element | Video on a 3D backdrop plane |
| `html-layer` | Visible under canvas | Transparent overlay only |

### Performance notes

- Video textures use **`replaceRgba`** (no per-frame GPU texture allocation).
- **`requestVideoFrameCallback`** uploads only when a new decoded frame is ready (not while scrubbing).
- Texture resolution matches stage size × DPR (capped at 1280×720).
- Backdrop video uses **`object-fit: cover`** cropping.

## Regenerate bundled clip

```bash
ffmpeg -y -f lavfi -i "testsrc2=size=854x480:rate=24:duration=10" \
  -c:v libx264 -preset fast -crf 30 -pix_fmt yuv420p -movflags +faststart \
  web/scroll-video/assets/calibration-clip.mp4
```

## Local demo

```bash
cd tests/parity && node server.js
# http://localhost:8087/web/scroll-video/example.html
# http://localhost:8087/web/scroll-video/calibration.html
```
