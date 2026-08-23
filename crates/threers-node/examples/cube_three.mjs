/**
 * THREE.* drop-in — same imports as three.js, swap the package name.
 *
 * Browser / Deno / Node ESM with WebGPU:
 *   npm i threers
 *   import THREE, { initThreers } from 'threers';
 *
 * Run the live demo: see ../threers-js/examples/cube.html
 *
 * This file is the canonical three.js snippet; paste it into any app that
 * already uses three.js and change only the import line.
 */

import THREE, { initThreers } from 'threers';

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

// In the browser: WebGLRenderer + canvas + requestAnimationFrame (see cube.html).
// Headless PNG on native Node: use threers-node/examples/cube_headless.mjs instead.

export { scene, camera, mesh, THREE };
