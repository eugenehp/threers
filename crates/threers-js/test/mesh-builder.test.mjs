// The JS MeshBuilder: the Rust builder behind three.js-style arguments, giving
// back an ordinary BufferGeometry. Runs the shim over a real wasm build
// (`THREERS_PKG_DIR`, else web/pkg), and skips without one.

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
  if (typeof m.MeshBuilder !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-mb-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const have = existsSync(join(pkgDir, 'threers_bg.wasm'));
const THREE = have ? await loadShim(pkgDir) : null;
const skip = THREE ? false : 'needs a wasm build with MeshBuilder (web/build.sh, or THREERS_PKG_DIR)';

function faceY(g, t) {
  const p = g.attributes.position.array, i = g.index.array;
  const v = k => [p[i[t * 3 + k] * 3], p[i[t * 3 + k] * 3 + 1], p[i[t * 3 + k] * 3 + 2]];
  const [a, b, c] = [v(0), v(1), v(2)];
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return u[2] * w[0] - u[0] * w[2];
}

describe('MeshBuilder', { skip }, () => {
  it('builds a BufferGeometry that faces the side asked for, whatever the order', () => {
    for (const order of [[0, 1, 2, 3], [3, 2, 1, 0]]) {
      const m = new THREE.MeshBuilder();
      const corners = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]];
      const ids = order.map(i => m.vertex(corners[i]));
      const g = m.facing(new THREE.Vector3(0, 1, 0)).quad(...ids).build();
      assert.ok(g instanceof THREE.BufferGeometry);
      assert.equal(g.index.count, 6);
      assert.ok(faceY(g, 0) > 0 && faceY(g, 1) > 0);
      assert.ok(g.attributes.normal.array[1] > 0.99);
    }
  });

  it('grids from a function or an array, and ribbons from Vector3s', () => {
    const fromFn = new THREE.MeshBuilder().grid(10, 10, 4, 4, (x, z) => x * 0.5).build();
    const heights = [];
    for (let r = 0; r <= 4; r++) for (let c = 0; c <= 4; c++) heights.push((c / 4 - 0.5) * 10 * 0.5);
    const fromArray = new THREE.MeshBuilder().grid(10, 10, 4, 4, heights).build();
    assert.deepEqual(Array.from(fromFn.attributes.position.array), Array.from(fromArray.attributes.position.array));
    const left = [0, 1, 2].map(i => new THREE.Vector3(i, 0, 0));
    const right = [0, 1, 2].map(i => new THREE.Vector3(i, 0, 1));
    const strip = new THREE.MeshBuilder().ribbon(right, left).build();
    assert.equal(strip.index.count, 12);
    assert.ok(faceY(strip, 0) > 0);
  });

  it('merges geometries with a Matrix4 and reports the result solid', () => {
    const box = new THREE.BoxGeometry(1, 1, 1);
    const m = new THREE.MeshBuilder();
    m.append(box, new THREE.Matrix4().makeTranslation(5, 0, 0));
    m.append(box, new THREE.Matrix4().makeTranslation(-5, 0, 0));
    const g = m.build();
    assert.equal(g.index.count, box.index.count * 2);
    g.computeBoundingBox();
    assert.ok(Math.abs(g.boundingBox.max.x - 5.5) < 1e-4);
    const report = THREE.meshReport(g);
    assert.equal(report.isSolid, true, JSON.stringify(report));
    assert.equal(THREE.meshReport(new THREE.PlaneGeometry(1, 1)).openEdges, 4);
  });

  it('says what is wrong where it is written', () => {
    const m = new THREE.MeshBuilder();
    const a = m.vertex(0, 0, 0);
    assert.throws(() => m.triangle(a, a, 7), /vertex 7 does not exist/);
    assert.throws(() => new THREE.MeshBuilder().grid(1, 1, 2, 2, [1, 2, 3]), /expected 9 heights/);
  });
});
