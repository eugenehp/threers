/**
 * Spinning cube — the three.js "hello world", via the threers THREE.* shim.
 *
 * Browser: open examples/cube.html (after `npm run build` in this crate).
 *
 * Bundler / Node ESM (WebGPU canvas required):
 *   npm i threers
 *   import THREE, { initThreers } from 'threers';
 */

import THREE, { initThreers } from 'threers';

/** @param {HTMLCanvasElement} canvas */
export async function runCubeDemo(canvas) {
  await initThreers();

  const renderer = await THREE.WebGLRenderer.create(canvas);
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio ?? 1, 2));

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

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(3, 2, 5);
  camera.lookAt(0, 0, 0);

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  resize();
  globalThis.addEventListener?.('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    mesh.rotation.x += 0.008;
    mesh.rotation.y += 0.012;
    renderer.render(scene, camera);
  }
  animate();

  return { scene, camera, renderer, mesh };
}

// Run when executed directly in a browser with a #c canvas (e.g. via import map).
if (typeof document !== 'undefined') {
  const canvas = document.getElementById('c');
  if (canvas instanceof HTMLCanvasElement) {
    runCubeDemo(canvas);
  }
}
