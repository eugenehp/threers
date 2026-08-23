/**
 * Deno — same three.js-shaped API as npm (`THREE.*` drop-in).
 *
 *   cd crates/threers-js && ./build.sh
 *   deno run --allow-read --allow-net examples/cube.deno.ts
 *
 * Or from the registry (after publish):
 *   deno run --allow-read --allow-net examples/cube.deno.ts
 *   with import from "npm:threers"
 */

import THREE, { initThreers } from "../dist/mini/threejs-shim.js";
// Published: import THREE, { initThreers } from "npm:threers";

const html = `<!DOCTYPE html>
<html><body style="margin:0;background:#101010">
<canvas id="c" width="900" height="700" style="display:block;width:100vw;height:100vh"></canvas>
<script type="module" src="./cube.mjs"><\/script>
</body></html>`;

// Deno has no DOM — open cube.html in a browser for the live demo.
// This file shows the identical THREE setup you would paste into any three.js app:

await initThreers();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101010);

const mesh = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshStandardMaterial({ color: 0xff6633, roughness: 0.35, metalness: 0.1 }),
);
scene.add(mesh);

scene.add(new THREE.AmbientLight(0xffffff, 0.25));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(4, 6, 3);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
camera.position.set(3, 2, 5);
camera.lookAt(0, 0, 0);

console.log("THREE drop-in scene ready:", {
  objects: scene.children.length,
  camera: [camera.position.x, camera.position.y, camera.position.z],
});
console.log("Open examples/cube.html in a WebGPU browser for the spinning cube.");

export { scene, camera, mesh, html };
