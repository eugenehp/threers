// OrbitControls follows a camera pose set from JavaScript, as three.js's does:
// `update()` re-reads `camera.position` and `controls.target`, so a view set
// from outside (a saved view, an animation) is where the orbit continues
// from. Runs the shim over a real wasm build (`THREERS_PKG_DIR`, else web/pkg).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

async function loadShim(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  if (typeof m.WebOrbitControls?.prototype?.reseedFromCamera !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-orbit-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShim(pkgDir) : null;
const skip = !S && `no wasm build with WebOrbitControls.reseedFromCamera in ${pkgDir}`;
const near = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;

describe('OrbitControls and a pose set from outside', { skip }, () => {
  it('update() keeps a camera position and target set from JavaScript', () => {
    const cam = new S.PerspectiveCamera(50, 1, 0.1, 100);
    cam.position.set(0, 0, 5);
    cam.lookAt(0, 0, 0);
    const controls = new S.OrbitControls(cam, null);
    controls.update();
    cam.position.set(3, 2, -4);
    controls.target.set(0.5, 1, 0);
    controls.update();
    assert.ok(near(cam.position.x, 3) && near(cam.position.y, 2) && near(cam.position.z, -4),
      `position moved to ${cam.position.toArray()}`);
    assert.ok(near(controls.target.x, 0.5) && near(controls.target.y, 1) && near(controls.target.z, 0),
      `target moved to ${controls.target.toArray()}`);
    // And the camera looks at the target.
    const dir = cam.getWorldDirection(new S.Vector3());
    const want = new S.Vector3(0.5 - 3, 1 - 2, 0 + 4).normalize();
    assert.ok(dir.distanceTo(want) < 1e-3, `looks along ${dir.toArray()}, want ${want.toArray()}`);
  });

  it('camera.up stays the world up the orbit turns about, as in three.js', () => {
    // Views from behind and below the horizon, where the camera's own y axis
    // is far from world up.
    for (const [elev, azim] of [[38, 142], [28, 218], [120, 300]]) {
      const e = elev * Math.PI / 180, a = azim * Math.PI / 180;
      const p = [5 * Math.sin(e) * Math.sin(a), 5 * Math.cos(e), 5 * Math.sin(e) * Math.cos(a)];
      const cam = new S.PerspectiveCamera(50, 4 / 3, 0.1, 100);
      cam.position.set(...p);
      cam.lookAt(0, 0, 0);
      const controls = new S.OrbitControls(cam, null);
      controls.update();
      assert.deepEqual(cam.up.toArray(), [0, 1, 0], `up after construction at ${elev}/${azim}`);
      cam.position.set(p[2], p[1], -p[0]);
      controls.update();
      assert.deepEqual(cam.up.toArray(), [0, 1, 0], `up after a move at ${elev}/${azim}`);
      // three.js's lookAt with world up: the camera's x axis is level.
      const right = new S.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
      assert.ok(Math.abs(right.y) < 1e-4, `camera rolled at ${elev}/${azim}: right ${right.toArray()}`);
    }
  });

  it('an unchanged pose is left to the orbit', () => {
    const cam = new S.PerspectiveCamera(50, 1, 0.1, 100);
    cam.position.set(0, 0, 5);
    const controls = new S.OrbitControls(cam, null);
    controls.update();
    const before = cam.position.toArray();
    controls.update();
    assert.deepEqual(cam.position.toArray().map(v => +v.toFixed(5)), before.map(v => +v.toFixed(5)));
  });
});
