/**
 * Headless cube PNG — mirrors a minimal three.js scene graph, via native Node.
 *
 * For the **drop-in THREE.* API** (same code as three.js in the browser), use
 * the `threers` wasm package instead of `threers-node`:
 *
 *   import THREE, { initThreers } from 'threers';
 *   await initThreers();
 *   const scene = new THREE.Scene();
 *   …
 *
 * This example uses `threers-node` when you want a native .node addon and
 * offscreen PNG without loading wasm.
 *
 *   cd crates/threers-node && npm i && npm run build
 *   node examples/cube_headless.mjs
 */

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  Scene,
  PerspectiveCamera,
  HeadlessRenderer,
} from 'threers-node';

// three.js                          →  threers-node
// ─────────────────────────────────────────────────
// const scene = new THREE.Scene();  →  const scene = new Scene()
// scene.background = 0x101010       →  scene.setBackground(0x101010)
// new THREE.BoxGeometry(1,1,1)      →  scene.addBox(1, 1, 1, 0xff6633)
// new THREE.PerspectiveCamera(…)    →  new PerspectiveCamera(…)
// camera.position.set(3, 2, 5)      →  camera.setPosition(3, 2, 5)
// camera.lookAt(0, 0, 0)            →  camera.lookAt(0, 0, 0)
// renderer.render(scene, camera)    →  renderer.renderPng(scene, camera)

const scene = new Scene();
scene.setBackground(0x101010);
scene.addBox(1, 1, 1, 0xff6633);

const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 100);
camera.setPosition(3, 2, 5);
camera.lookAt(0, 0, 0);

const renderer = new HeadlessRenderer(800, 600);
const png = renderer.renderPng(scene, camera);

const out = fileURLToPath(new URL('./cube.png', import.meta.url));
writeFileSync(out, png);
console.log(`wrote ${out} (${png.length} bytes)`);
