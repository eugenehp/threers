# Camera controls — three.js vs threers

threers mirrors three.js-style controls through `web/threejs-shim.js`. The Rust implementations live in `src/controls/`.

| Control | three.js (`examples/jsm`) | threers shim (`THREE.*`) | Native Rust (`threers::`) |
|---------|----------------------------|--------------------------|---------------------------|
| Orbit | Drop-in, auto `domElement` | Manual pointer wiring | `PointerEvent` + `OrbitControls::update` |
| Trackball | Drop-in | **Drop-in** (auto events) | `TrackballControls::update` |
| First person | Drop-in | Drop-in (WASD + mouse) | `FirstPersonControls::update` |
| Drag | Drop-in | JS raycast drag | `DragControls` (Rust) |
| Transform | Full 3D gizmo | Simplified gizmo | — |

Runnable browser demos (live preview + source tabs):

- http://localhost:8087/web/examples/controls-threejs.html — JavaScript · TypeScript · HTML
- http://localhost:8087/web/examples/controls-threers.html — JavaScript · TypeScript · Rust · HTML
- http://localhost:8087/web/examples/ — index

Native demo: `cargo run --example controls_orbit`.

## TypeScript

Types ship with the browser shim:

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true
  }
}
```

```typescript
import THREE, {
  initThreers,
  type WebGLRenderer,
  type PerspectiveCamera,
} from '../web/threejs-shim.js';

await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

const renderer: WebGLRenderer = await THREE.WebGLRenderer.create(canvas);
const camera = new THREE.PerspectiveCamera(60, 1.6, 0.1, 100) as PerspectiveCamera;
```

- Definitions: `web/threejs-shim.d.ts` (auto-generated — `node web/scripts/generate-shim-types.mjs`)
- Low-level wasm bindings: `web/pkg/threers.d.ts`
- Validate: `cd web && npm install && npm run typecheck`
- Optional peer `three@^0.165` for cross-checking against canonical three.js types when porting apps

---

## OrbitControls

### three.js (JavaScript)

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const renderer = new THREE.WebGLRenderer({ canvas });
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
camera.position.set(2, 2, 4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;

function animate() {
    requestAnimationFrame(animate);
    controls.update(); // required every frame for damping / auto-rotate
    renderer.render(scene, camera);
}
animate();
```

threers shim parity: `enableDamping`, `dampingFactor`, `autoRotate`, `autoRotateSpeed`. Native Rust: `enable_damping`, `damping`, `auto_rotate`, `update_dt(...)`.

For scripted cinematic cameras (fly-to, paths, shake, shot timelines), see [camera-animation.md](./camera-animation.md).

### three.js (TypeScript)

```typescript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas });
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
camera.position.set(2, 2, 4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function animate(): void {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
```

### threers shim (JavaScript)

Orbit math runs in Rust/wasm. Wire pointer events yourself, or use **TrackballControls** (below) for a drop-in.

```javascript
import THREE, { initThreers } from '/web/threejs-shim.js';

await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

const canvas = document.getElementById('canvas');
const renderer = await THREE.WebGLRenderer.create(canvas);
const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 100);
camera.position.set(2, 2, 4);

const controls = new THREE.OrbitControls(camera, canvas);

let rotating = false, panning = false, lastX = 0, lastY = 0;

canvas.addEventListener('pointerdown', (e) => {
    rotating = e.button === 0;
    panning = e.button === 2;
    lastX = e.clientX;
    lastY = e.clientY;
});
canvas.addEventListener('pointermove', (e) => {
    if (!rotating && !panning) return;
    controls.update(e.clientX - lastX, e.clientY - lastY, 0, rotating, panning);
    lastX = e.clientX;
    lastY = e.clientY;
});
canvas.addEventListener('pointerup', () => { rotating = panning = false; });
canvas.addEventListener('wheel', (e) => controls.update(0, 0, e.deltaY, false, false));

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
```

### threers shim (TypeScript)

See `web/examples/controls-threers.ts` and `web/threejs-shim.d.ts`.

```typescript
import THREE, { initThreers } from '/web/threejs-shim.js';
import type { PerspectiveCamera, WebGLRenderer } from '/web/threejs-shim.js';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

const renderer: WebGLRenderer = await THREE.WebGLRenderer.create(canvas);
const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 100) as PerspectiveCamera;
camera.position.set(2, 2, 4);

const controls = new THREE.OrbitControls(camera, canvas);

let rotating = false, panning = false, lastX = 0, lastY = 0;

canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    rotating = e.button === 0;
    panning = e.button === 2;
    lastX = e.clientX;
    lastY = e.clientY;
});
canvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (!rotating && !panning) return;
    controls.update(e.clientX - lastX, e.clientY - lastY, 0, rotating, panning);
    lastX = e.clientX;
    lastY = e.clientY;
});
canvas.addEventListener('pointerup', () => { rotating = panning = false; });
canvas.addEventListener('wheel', (e: WheelEvent) => controls.update(0, 0, e.deltaY, false, false));

function animate(): void {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
```

### Native Rust

```rust
use threers::cameras::Camera;
use threers::{OrbitControls, PerspectiveCamera, PointerEvent, Vector3};

let mut camera = PerspectiveCamera::new(60.0, 16.0 / 9.0, 0.1, 100.0);
camera.position = Vector3::new(2.0, 2.0, 4.0);
camera.look_at(Vector3::ZERO);

let mut controls = OrbitControls::new(&camera);

// Each frame / input event — translate platform events into PointerEvent:
let ev = PointerEvent {
    dx: 12.0,
    dy: -4.0,
    wheel: 0.0,
    rotating: true,
    panning: false,
};
controls.update(ev, &mut camera, (800.0, 600.0));
```

Full winit example: `examples/controls_orbit.rs` (`cargo run --example controls_orbit`).

---

## TrackballControls (recommended drop-in for threers)

### three.js (JavaScript)

```javascript
import * as THREE from 'three';
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js';

const controls = new TrackballControls(camera, renderer.domElement);

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
```

### threers shim (JavaScript)

Events are wired on `domElement` — closest to three.js OrbitControls ergonomics.

```javascript
import THREE, { initThreers } from '/web/threejs-shim.js';

await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
const renderer = await THREE.WebGLRenderer.create(canvas);
const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.1, 100);

const controls = new THREE.TrackballControls(camera, renderer.domElement);
// left-drag orbit · right-drag pan · wheel zoom

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
```

### Native Rust

```rust
use threers::{TrackballControls, PerspectiveCamera, PointerEvent};

let mut controls = TrackballControls::new(&camera);
controls.update(
    PointerEvent { dx: 5.0, dy: 2.0, wheel: 0.0, rotating: true, panning: false },
    &mut camera,
    (width as f32, height as f32),
);
```

---

## FirstPersonControls

### three.js

```javascript
import { FirstPersonControls } from 'three/examples/jsm/controls/FirstPersonControls.js';
const controls = new FirstPersonControls(camera, domElement);
// animation loop:
controls.update(1 / 60);
```

### threers shim

```javascript
const controls = new THREE.FirstPersonControls(camera, canvas);
function animate() {
    controls.update(0, 0, 1 / 60, false);
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
```

### Native Rust

```rust
use threers::{FirstPersonControls, PointerEvent, Vector3};

let mut fp = FirstPersonControls::new(&camera);
fp.move_input = Vector3::new(1.0, 0.0, 0.0); // forward
fp.update(
    PointerEvent { dx: 0.0, dy: 0.0, wheel: 0.0, rotating: false, panning: false },
    &mut camera,
    1.0 / 60.0,
    false,
);
```

---

## DragControls

### three.js

```javascript
import { DragControls } from 'three/examples/jsm/controls/DragControls.js';
const drag = new DragControls([mesh], camera, domElement);
drag.addEventListener('drag', () => renderer.render(scene, camera));
```

### threers shim

```javascript
const drag = new THREE.DragControls([mesh], camera, canvas);
drag.addEventListener('drag', () => renderer.render(scene, camera));
```

---

## Import map cheat sheet

**Real three.js** — separate `examples/jsm` imports:

```html
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.165.0/build/three.module.js" } }
</script>
<script type="module">
import { OrbitControls } from 'https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js';
</script>
```

**threers** — everything on `THREE.*`:

```html
<script type="module">
import THREE, { initThreers } from '/web/threejs-shim.js';
await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
const controls = new THREE.TrackballControls(camera, canvas);
</script>
```

Do **not** mix three.js `OrbitControls` with `THREE.WebGLRenderer.create()` — the renderer and camera are wasm-backed types behind the shim.
