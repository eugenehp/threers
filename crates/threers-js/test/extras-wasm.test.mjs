// The shim's loaders, noise and debug helpers run on Rust (`src/wasm/extras.rs`);
// these check the bindings over a real wasm build against the references
// three.js r165 itself uses: Google's Draco decoder (the one three.js's
// DRACOLoader loads), three.js's own `SimplexNoise`, and three.js's helper
// objects, all run here in node.
//
// Needs a wasm build: `web/pkg`, or the directory in THREERS_PKG_DIR.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const three = join(repo, 'web/node_modules/three');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

let real = null;
if (existsSync(join(pkgDir, 'threers.js')) && existsSync(join(pkgDir, 'threers_bg.wasm'))) {
  const pkg = await import(pathToFileURL(join(pkgDir, 'threers.js')).href);
  pkg.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  if (typeof pkg.WebDracoLoader === 'function') real = pkg;
}
const THREE = existsSync(three) ? await import(pathToFileURL(join(three, 'build/three.module.js')).href) : null;
const skip = (!real && 'wasm build missing or without WebDracoLoader') || (!THREE && 'web/node_modules/three missing');

/// Google's Draco decoder as three.js ships it, evaluated as the CommonJS it is.
async function googleDraco() {
  const path = join(three, 'examples/jsm/libs/draco/draco_decoder.js');
  const module = { exports: {} };
  new Function('module', 'exports', 'require', '__filename', '__dirname', readFileSync(path, 'utf8'))(
    module, module.exports, createRequire(import.meta.url), path, dirname(path));
  return module.exports();
}

/// three.js r165 DRACOLoader's `decodeGeometry` for a `.drc` (default attribute
/// ids, Float32 arrays), without the worker around it.
function decodeLikeThree(draco, bytes) {
  const decoder = new draco.Decoder();
  const mesh = new draco.Mesh();
  const status = decoder.DecodeArrayToMesh(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), bytes.byteLength, mesh);
  assert.ok(status.ok(), status.error_msg());
  const out = {};
  for (const [name, type] of [['position', 'POSITION'], ['normal', 'NORMAL'], ['color', 'COLOR'], ['uv', 'TEX_COORD']]) {
    const id = decoder.GetAttributeId(mesh, draco[type]);
    if (id === -1) continue;
    const attribute = decoder.GetAttribute(mesh, id);
    const n = mesh.num_points() * attribute.num_components();
    const ptr = draco._malloc(n * 4);
    decoder.GetAttributeDataArrayForAllPoints(mesh, attribute, draco.DT_FLOAT32, n * 4, ptr);
    out[name] = new Float32Array(draco.HEAPF32.buffer, ptr, n).slice();
    draco._free(ptr);
  }
  const n = mesh.num_faces() * 3;
  const ptr = draco._malloc(n * 4);
  decoder.GetTrianglesUInt32Array(mesh, n * 4, ptr);
  out.index = new Uint32Array(draco.HEAPF32.buffer, ptr, n).slice();
  draco._free(ptr);
  draco.destroy(mesh);
  draco.destroy(decoder);
  return out;
}

describe('DRACOLoader over the Rust decoder', { skip }, () => {
  const fixtures = join(repo, 'src/loaders/draco/testdata');
  it('decodes every fixture to what three.js gets from Google\'s decoder', async () => {
    const draco = await googleDraco();
    const loader = new real.WebDracoLoader();
    for (const file of readdirSync(fixtures).filter((f) => f.endsWith('.drc'))) {
      const bytes = readFileSync(join(fixtures, file));
      const want = decodeLikeThree(draco, bytes);
      // Linear colours: `decodeDracoFile`'s default, so no sRGB step to undo.
      const got = loader.decode(bytes, false);
      assert.deepEqual(Array.from(got.getIndexArray()), Array.from(want.index), `${file}: index`);
      for (const name of ['position', 'normal', 'color', 'uv']) {
        const g = got.getAttributeArray(name);
        if (!want[name]) { assert.equal(g, undefined, `${file}: no ${name}`); continue; }
        assert.ok(g, `${file}: ${name} missing`);
        assert.equal(g.length, want[name].length, `${file}: ${name} length`);
        for (let i = 0; i < g.length; i++) {
          assert.ok(Math.abs(g[i] - want[name][i]) <= 1e-6 * Math.max(1, Math.abs(want[name][i])),
            `${file}: ${name}[${i}] ${g[i]} vs ${want[name][i]}`);
        }
      }
    }
  });
  it('refuses what is not Draco', () => {
    assert.throws(() => new real.WebDracoLoader().decode(new Uint8Array([1, 2, 3, 4]), true), /DRACOLoader/);
  });
});

describe('KTX2Loader over the Rust reader', { skip }, () => {
  const ktx = (name) => readFileSync(join(repo, 'src/loaders/testdata/ktx', name));
  it('reads RGBA8, zlib-supercompressed, mipmapped and BC7 files', () => {
    const loader = new real.WebKtx2Loader();
    for (const name of ['rgba8.ktx2', 'zlib.ktx2', 'mips.ktx2', 'bc7.ktx2']) {
      const header = new DataView(ktx(name).buffer, ktx(name).byteOffset);
      const t = loader.parseTexture(ktx(name));
      assert.equal(t.width(), header.getUint32(20, true), `${name}: width`);
      assert.equal(t.height(), Math.max(1, header.getUint32(24, true)), `${name}: height`);
    }
  });
  it('names what it cannot read', () => {
    const bytes = new Uint8Array(ktx('rgba8.ktx2'));
    new DataView(bytes.buffer).setUint32(12, 0, true); // vkFormat 0: Basis Universal
    assert.throws(() => new real.WebKtx2Loader().parseTexture(bytes), /Basis/);
    assert.throws(() => new real.WebKtx2Loader().parseTexture(new Uint8Array(16)), /KTX2Loader/);
  });
});

describe('SimplexNoise over WebThreeSimplexNoise', { skip }, async () => {
  const { SimplexNoise } = THREE
    ? await import(pathToFileURL(join(three, 'examples/jsm/math/SimplexNoise.js')).href)
    : { SimplexNoise: null };
  /// The shim's `seedRandom` generator.
  const lcg = (seed) => { let s = seed >>> 0; return { random: () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; } }; };
  it('gives three.js\'s numbers for the same random source', () => {
    const want = new SimplexNoise(lcg(7));
    const r = lcg(7);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = Math.floor(r.random() * 256);
    const got = new real.WebThreeSimplexNoise(p);
    const pts = lcg(99);
    for (let i = 0; i < 2000; i++) {
      const [x, y, z, w] = [0, 0, 0, 0].map(() => (pts.random() - 0.5) * 600);
      assert.equal(got.noise(x, y), want.noise(x, y));
      assert.equal(got.noise3d(x, y, z), want.noise3d(x, y, z));
      assert.equal(got.noise4d(x, y, z, w), want.noise4d(x, y, z, w));
    }
  });
});

/// Every line point of three.js's helper, in world space.
function threeLines(root) {
  root.updateMatrixWorld(true);
  const out = [];
  root.traverse((o) => {
    const g = o.geometry;
    if (!g) return;
    const pos = g.attributes.position;
    const idx = g.index ? Array.from(g.index.array) : [...Array(pos.count).keys()];
    const v = new THREE.Vector3();
    const P = (i) => { v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld); out.push(v.x, v.y, v.z); };
    if (o.isLineSegments) idx.forEach(P);
    else if (o.isLine) for (let k = 0; k + 1 < idx.length; k++) { P(idx[k]); P(idx[k + 1]); }
    else if (o.isMesh && o.material.wireframe) {
      for (let k = 0; k < idx.length; k += 3) for (const [i, j] of [[0, 1], [1, 2], [2, 0]]) { P(idx[k + i]); P(idx[k + j]); }
    }
  });
  return out;
}
function close(got, want, what) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < got.length; i++) {
    assert.ok(Math.abs(got[i] - want[i]) <= 1e-5 * Math.max(1, Math.abs(want[i])), `${what}[${i}]: ${got[i]} vs ${want[i]}`);
  }
}

describe('debug helper bindings against three.js', { skip }, () => {
  it('SpotLightHelper', () => {
    const light = new THREE.SpotLight(0xffffff, 1, 6, 0.4);
    light.position.set(1, 4, 2);
    light.target.position.set(-1, 0, 0);
    light.updateMatrixWorld(); light.target.updateMatrixWorld();
    const h = new THREE.SpotLightHelper(light);
    const g = real.helperSpotLight(new Float64Array(light.matrixWorld.elements), -1, 0, 0, 6, 0.4);
    close(g.lines, threeLines(h), 'spot');
  });
  it('CameraHelper', () => {
    const cam = new THREE.PerspectiveCamera(40, 2, 0.5, 8);
    cam.position.set(2, 1, 5); cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
    const h = new THREE.CameraHelper(cam);
    const proj = real.helperPerspectiveProjection(40, 2, 0.5, 8, 1, 0, 35, undefined);
    const c = (hex) => real.WebColor.fromHex(hex);
    const g = real.helperCamera(proj, new Float64Array(cam.matrixWorld.elements),
      c(0xffaa00), c(0xff0000), c(0x00aaff), c(0xffffff), c(0x333333));
    close(g.lines, threeLines(h), 'camera');
    close(g.colors, Array.from(h.geometry.attributes.color.array), 'camera colours');
  });
  it('PlaneHelper and ArrowHelper (lines and fills)', () => {
    const plane = new THREE.Plane(new THREE.Vector3(1, 1, 0).normalize(), 2);
    const ph = new THREE.PlaneHelper(plane, 3);
    const g = real.helperPlane(plane.normal.x, plane.normal.y, plane.normal.z, 2, 3);
    close(g.lines, threeLines(ph), 'plane');
    assert.equal(g.mesh.length, 18);
    const dir = new THREE.Vector3(0, -1, 1).normalize();
    const ah = new THREE.ArrowHelper(dir, new THREE.Vector3(1, 2, 3), 2);
    const q = real.helperArrowQuaternion(dir.x, dir.y, dir.z);
    assert.deepEqual(Array.from(q).map((v) => +v.toFixed(12)), ah.quaternion.toArray().map((v) => +v.toFixed(12)));
    ah.updateMatrixWorld(true);
    const a = real.helperArrow(new Float64Array(ah.matrixWorld.elements), 2, 0.4, 0.08);
    close(a.lines, threeLines(ah), 'arrow');
    assert.equal(a.mesh.length, 15 * 9); // 10 side + 5 cap triangles
  });
});
