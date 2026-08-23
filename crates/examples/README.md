# Examples — THREE.* drop-in first

threers is a **drop-in three.js replacement**. Start with the same `THREE.*`
code you already know; only the import line changes.

| Runtime | Package | Example | Run |
|---------|---------|---------|-----|
| Browser | `threers` (npm) | [cube.html](../threers-js/examples/cube.html) | `cd crates/threers-js && npm run build && python3 -m http.server -d . 8765` → `/examples/cube.html` |
| npm / bundler | `threers` | [cube.mjs](../threers-js/examples/cube.mjs) | `import THREE from 'threers'` |
| Deno | `npm:threers` | [cube.deno.ts](../threers-js/examples/cube.deno.ts) | `deno run examples/cube.deno.ts` |
| Node headless PNG | `threers-node` | [cube_headless.mjs](../threers-node/examples/cube_headless.mjs) | `node examples/cube_headless.mjs` |
| Python headless PNG | `threers` (PyPI) | [cube.py](../threers-py/examples/cube.py) | `python examples/cube.py` |

## three.js → threers (JavaScript)

```js
// was:
// import * as THREE from 'three';
import THREE, { initThreers } from 'threers';

await initThreers();

const scene = new THREE.Scene();
const mesh = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshStandardMaterial({ color: 0xff6633 }),
);
scene.add(mesh);

const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
const renderer = await THREE.WebGLRenderer.create(canvas);
renderer.render(scene, camera);
```

Use `threers/full` when you need OpenSCAD, codecs, NURBS, or path tracing.

## Native bindings (Python / Node)

Python and `threers-node` expose a **small headless subset** today. Each example
maps the same scene graph to the native API and points back to the THREE shim for
full parity.

See [docs/bindings.md](../../docs/bindings.md).
