// The shim's InterleavedBuffer / InterleavedBufferAttribute /
// InstancedInterleavedBuffer against three.js r165 (web/node_modules/three)
// on seeded random data: every accessor and setter, normalized Int8 / Uint8 /
// Int16 / Uint16 storage, applyMatrix4 / applyNormalMatrix /
// transformDirection, clone and toJSON with and without shared `data`, the
// buffer's copy / copyAt / set / updateRanges; then the math and geometry
// code that reads attributes (Vector2/3/4 and Color.fromBufferAttribute,
// Box3.setFromBufferAttribute, computeBoundingBox / computeBoundingSphere,
// rotateX / applyMatrix4 / translate, toNonIndexed, computeVertexNormals,
// BufferGeometryUtils' interleave / deinterleave / mergeAttributes), and
// last what reaches Rust: the wasm geometry built from an interleaved
// geometry carries exactly the attributes of its plain equivalent, a
// `needsUpdate` on the shared buffer re-syncs every geometry using it, and
// raycasts hit what three.js hits.
//
// The wasm comes from `THREERS_PKG_DIR` (a wasm-bindgen `--target web`
// output directory) or web/pkg.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
const threePath = join(repo, 'web/node_modules/three');

async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const pkg = await import(pkgUrl);
  pkg.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-interleaved-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threePath, 'build/three.module.js'));
const S = ready ? await loadShimOver(pkgDir) : null;
const THREE = ready ? await import(pathToFileURL(join(threePath, 'build/three.module.js')).href) : null;
const TU = ready ? await import(pathToFileURL(join(threePath, 'examples/jsm/utils/BufferGeometryUtils.js')).href) : null;
const skip = !ready && `no wasm in ${pkgDir} or no three`;

// three.js logs when an interleaved attribute is cloned or serialized
// without shared data; the shim does not.
const quiet = (f) => { const log = console.log; console.log = () => {}; try { return f(); } finally { console.log = log; } };

// ---- seeded inputs ----
let seed = 0x1234567;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const rnd = (a = -2, b = 2) => a + (b - a) * rand();

const TOL = 1e-12;
function close(a, b, tol = TOL) {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  return a === b || Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
function sameArray(got, want, what, tol = TOL) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++) assert.ok(close(got[i], want[i], tol), `${what}[${i}]: ${got[i]} vs ${want[i]}`);
}

const TYPES = [Float32Array, Int8Array, Uint8Array, Int16Array, Uint16Array];
/// A random array of `Type`, valid stored values for it.
function randomArray(Type, n) {
  const a = new Type(n);
  for (let i = 0; i < n; i++) {
    if (Type === Float32Array) a[i] = rnd(-10, 10);
    else {
      const bits = Type.BYTES_PER_ELEMENT * 8, signed = Type.name.startsWith('Int');
      const lo = signed ? -(2 ** (bits - 1)) : 0, hi = signed ? 2 ** (bits - 1) - 1 : 2 ** bits - 1;
      a[i] = lo + Math.floor(rand() * (hi - lo + 1));
    }
  }
  return a;
}
/// The same interleaved layout in both libraries: stride 9, attributes of
/// sizes 3 / 4 / 2 at offsets 0 / 3 / 7.
function pair(Type, normalized, verts = 8) {
  const arr = randomArray(Type, verts * 9);
  const mk = (L) => {
    const ib = new L.InterleavedBuffer(new Type(arr), 9);
    return { ib, a3: new L.InterleavedBufferAttribute(ib, 3, 0, normalized), a4: new L.InterleavedBufferAttribute(ib, 4, 3, normalized), a2: new L.InterleavedBufferAttribute(ib, 2, 7, normalized) };
  };
  return { s: mk(S), t: mk(THREE) };
}
/// A random value a setter of a normalized attribute of `Type` can store.
function setValue(Type, normalized) {
  if (!normalized) return Type === Float32Array ? rnd(-10, 10) : Math.round(rnd(0, 100));
  return Type.name.startsWith('Int') ? rnd(-1, 1) : rnd(0, 1);
}

/// Strip uuids (random on each side) for a structural comparison.
function stripUuids(o) {
  return JSON.parse(JSON.stringify(o, (k, v) => (k === 'uuid' || k === 'buffer' || k === 'data' && typeof v === 'string') ? '<id>' : v));
}

describe('InterleavedBufferAttribute API vs three.js r165', { skip }, () => {
  it('flags, count, array and the buffer fields', () => {
    const { s, t } = pair(Float32Array, false);
    for (const k of ['isInterleavedBufferAttribute', 'name', 'itemSize', 'offset', 'normalized', 'count']) assert.equal(s.a4[k], t.a4[k], k);
    assert.equal(s.a4.array, s.ib.array);
    for (const k of ['isInterleavedBuffer', 'stride', 'count', 'usage', 'version']) assert.equal(s.ib[k], t.ib[k], k);
    assert.deepEqual(s.ib.updateRanges, t.ib.updateRanges);
    assert.equal(typeof s.ib.uuid, 'string');
    // count is three.js's array.length / stride, not floored.
    assert.equal(new S.InterleavedBuffer(new Float32Array(10), 3).count, new THREE.InterleavedBuffer(new Float32Array(10), 3).count);
    s.a3.needsUpdate = true; t.a3.needsUpdate = true;
    assert.equal(s.ib.version, t.ib.version);
    s.ib.addUpdateRange(3, 9); t.ib.addUpdateRange(3, 9);
    assert.deepEqual(s.ib.updateRanges, t.ib.updateRanges);
    s.ib.clearUpdateRanges(); t.ib.clearUpdateRanges();
    assert.deepEqual(s.ib.updateRanges, t.ib.updateRanges);
    assert.equal(s.ib.setUsage(35048), s.ib); assert.equal(s.ib.usage, 35048);
    let called = 0;
    s.ib.onUpload(() => called++); s.ib.onUploadCallback();
    assert.equal(called, 1);
  });

  for (const Type of TYPES) {
    for (const normalized of Type === Float32Array ? [false, true] : [false, true]) {
      it(`get/set accessors on ${Type.name}${normalized ? ' normalized' : ''}`, () => {
        const { s, t } = pair(Type, normalized);
        for (const name of ['a3', 'a4', 'a2']) {
          const sa = s[name], ta = t[name];
          for (let i = 0; i < sa.count; i++) {
            for (const g of ['getX', 'getY', 'getZ', 'getW']) assert.ok(close(sa[g](i), ta[g](i)), `${name}.${g}(${i})`);
            for (let c = 0; c < sa.itemSize; c++) assert.ok(close(sa.getComponent(i, c), ta.getComponent(i, c)), `${name}.getComponent(${i},${c})`);
          }
        }
        // Setters, each with the same random values on both sides.
        for (let round = 0; round < 40; round++) {
          const i = Math.floor(rand() * s.a4.count);
          const v = [0, 1, 2, 3].map(() => setValue(Type, normalized));
          const op = Math.floor(rand() * 9);
          const name = ['a3', 'a4', 'a2'][Math.floor(rand() * 3)];
          const call = (a) => {
            switch (op) {
              case 0: return a.setX(i, v[0]);
              case 1: return a.setY(i, v[1]);
              case 2: return a.setZ(i, v[2]);
              case 3: return a.setW(i, v[3]);
              case 4: return a.setXY(i, v[0], v[1]);
              case 5: return a.setXYZ(i, v[0], v[1], v[2]);
              case 6: return a.setXYZW(i, v[0], v[1], v[2], v[3]);
              case 7: return a.setComponent(i, 1, v[1]);
              default: return a.setComponent(i, 0, v[0]);
            }
          };
          assert.equal(call(s[name]), s[name], 'setters chain');
          call(t[name]);
          sameArray(s.ib.array, t.ib.array, `after op ${op} on ${name}`);
        }
      });
    }
  }

  for (const Type of [Float32Array, Int16Array, Int8Array]) {
    it(`applyMatrix4 / applyNormalMatrix / transformDirection on ${Type.name}`, () => {
      const normalized = Type !== Float32Array;
      for (let round = 0; round < 6; round++) {
        const { s, t } = pair(Type, normalized);
        const e = Array.from({ length: 16 }, () => rnd());
        e[3] = e[7] = e[11] = 0; e[15] = 1;
        const sm = new S.Matrix4().fromArray(e), tm = new THREE.Matrix4().fromArray(e);
        const which = round % 3;
        if (which === 0) { assert.equal(s.a3.applyMatrix4(sm), s.a3); t.a3.applyMatrix4(tm); }
        if (which === 1) { s.a3.applyNormalMatrix(new S.Matrix3().getNormalMatrix(sm)); t.a3.applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(tm)); }
        if (which === 2) { s.a4.transformDirection(sm); t.a4.transformDirection(tm); }
        sameArray(s.ib.array, t.ib.array, `transform ${which}`);
      }
    });
  }

  it('clone and toJSON, with and without shared data', () => {
    for (const Type of TYPES) {
      const { s, t } = pair(Type, Type !== Float32Array);
      // De-interleaving clone.
      const sc = s.a4.clone(), tc = quiet(() => t.a4.clone());
      assert.equal(sc.itemSize, tc.itemSize);
      assert.equal(sc.normalized, tc.normalized);
      sameArray(Array.from(sc.array), Array.from(tc.array), `${Type.name} clone()`);
      // De-interleaving toJSON.
      assert.deepEqual(s.a4.toJSON(), quiet(() => t.a4.toJSON()));
      // Shared clone: both attributes land on one new buffer.
      const sd = {}, td = {};
      const s1 = s.a3.clone(sd), s2 = s.a2.clone(sd), t1 = t.a3.clone(td);
      assert.equal(s1.data, s2.data);
      assert.notEqual(s1.data, s.ib);
      assert.equal(s1.data.array.constructor, Type);
      sameArray(s1.data.array, t1.data.array, 'shared clone buffer');
      assert.equal(s1.offset, 0); assert.equal(s2.offset, 7);
      // Shared toJSON: interleavedBuffers + arrayBuffers, keyed by uuid.
      const sj = {}, tj = {};
      const a = s.a3.toJSON(sj); s.a2.toJSON(sj);
      const b = t.a3.toJSON(tj); t.a2.toJSON(tj);
      assert.deepEqual(stripUuids(a), stripUuids(b));
      assert.equal(Object.keys(sj.interleavedBuffers).length, 1);
      assert.deepEqual(stripUuids(Object.values(sj.interleavedBuffers)[0]), stripUuids(Object.values(tj.interleavedBuffers)[0]));
      assert.deepEqual(Object.values(sj.arrayBuffers)[0], Object.values(tj.arrayBuffers)[0]);
    }
  });

  it('InterleavedBuffer copy / copyAt / set / clone', () => {
    const { s, t } = pair(Int16Array, true);
    const so = new S.InterleavedBuffer(new Int16Array(72), 9), to = new THREE.InterleavedBuffer(new Int16Array(72), 9);
    so.copy(s.ib); to.copy(t.ib);
    sameArray(so.array, to.array, 'copy'); assert.equal(so.count, to.count); assert.notEqual(so.array, s.ib.array);
    so.copyAt(1, s.ib, 4); to.copyAt(1, t.ib, 4);
    sameArray(so.array, to.array, 'copyAt');
    so.set([1, 2, 3], 5); to.set([1, 2, 3], 5);
    sameArray(so.array, to.array, 'set');
    const sc = s.ib.clone({}), tc = t.ib.clone({});
    assert.equal(sc.constructor, S.InterleavedBuffer);
    sameArray(sc.array, tc.array, 'clone'); assert.equal(sc.usage, tc.usage);
  });

  it('InstancedInterleavedBuffer', () => {
    const arr = randomArray(Float32Array, 32);
    const s = new S.InstancedInterleavedBuffer(new Float32Array(arr), 8, 2), t = new THREE.InstancedInterleavedBuffer(new Float32Array(arr), 8, 2);
    assert.equal(s.isInstancedInterleavedBuffer, true); assert.equal(s.isInterleavedBuffer, true);
    assert.equal(s.meshPerAttribute, t.meshPerAttribute);
    const sc = s.clone({}), tc = t.clone({});
    assert.equal(sc.meshPerAttribute, tc.meshPerAttribute); assert.ok(sc instanceof S.InstancedInterleavedBuffer);
    const sco = new S.InstancedInterleavedBuffer(new Float32Array(8), 8).copy(s);
    assert.equal(sco.meshPerAttribute, 2);
    assert.deepEqual(stripUuids(s.toJSON({})), stripUuids(t.toJSON({})));
  });
});

describe('code reading interleaved attributes', { skip }, () => {
  it('Vector2/3/4 and Color fromBufferAttribute, Box3.setFromBufferAttribute', () => {
    for (const Type of [Float32Array, Uint8Array, Int16Array]) {
      const { s, t } = pair(Type, Type !== Float32Array);
      for (let i = 0; i < s.a4.count; i++) {
        assert.ok(new S.Vector2().fromBufferAttribute(s.a2, i).equals(new THREE.Vector2().fromBufferAttribute(t.a2, i)));
        const sv = new S.Vector3().fromBufferAttribute(s.a3, i), tv = new THREE.Vector3().fromBufferAttribute(t.a3, i);
        assert.ok(close(sv.x, tv.x) && close(sv.y, tv.y) && close(sv.z, tv.z), `Vector3 ${i}`);
        const s4 = new S.Vector4().fromBufferAttribute(s.a4, i), t4 = new THREE.Vector4().fromBufferAttribute(t.a4, i);
        assert.ok(close(s4.x, t4.x) && close(s4.w, t4.w), `Vector4 ${i}`);
      }
      const sb = new S.Box3().setFromBufferAttribute(s.a3), tb = new THREE.Box3().setFromBufferAttribute(t.a3);
      sameArray([...sb.min.toArray(), ...sb.max.toArray()], [...tb.min.toArray(), ...tb.max.toArray()], 'Box3');
    }
  });

  it('Color.fromBufferAttribute', { skip: S && typeof S.Color.prototype.fromBufferAttribute !== 'function' && 'shim Color has no fromBufferAttribute' }, () => {
    const { s, t } = pair(Uint8Array, true);
    for (let i = 0; i < s.a3.count; i++) {
      const sc = new S.Color().fromBufferAttribute(s.a3, i), tc = new THREE.Color().fromBufferAttribute(t.a3, i);
      assert.ok(close(sc.r, tc.r) && close(sc.g, tc.g) && close(sc.b, tc.b), `Color ${i}`);
    }
  });

  /// The same random indexed mesh, interleaved, in both libraries.
  function meshPair(L, f32, index) {
    const g = new L.BufferGeometry();
    const ib = new L.InterleavedBuffer(new Float32Array(f32), 8);
    g.setAttribute('position', new L.InterleavedBufferAttribute(ib, 3, 0));
    g.setAttribute('normal', new L.InterleavedBufferAttribute(ib, 3, 3));
    g.setAttribute('uv', new L.InterleavedBufferAttribute(ib, 2, 6));
    g.setIndex(index);
    return { g, ib };
  }
  function randomMesh(verts = 30, tris = 40) {
    const f32 = new Float32Array(verts * 8);
    for (let k = 0; k < verts; k++) {
      const n = [rnd(), rnd(), rnd()], l = Math.hypot(...n);
      f32.set([rnd(-3, 3), rnd(-3, 3), rnd(-3, 3), n[0] / l, n[1] / l, n[2] / l, rand(), rand()], k * 8);
    }
    const index = Array.from({ length: tris * 3 }, () => Math.floor(rand() * verts));
    return { f32, index };
  }

  it('computeBoundingBox / computeBoundingSphere', () => {
    for (let round = 0; round < 5; round++) {
      const { f32, index } = randomMesh();
      const s = meshPair(S, f32, index).g, t = meshPair(THREE, f32, index).g;
      s.computeBoundingBox(); t.computeBoundingBox();
      sameArray([...s.boundingBox.min.toArray(), ...s.boundingBox.max.toArray()], [...t.boundingBox.min.toArray(), ...t.boundingBox.max.toArray()], 'box');
      s.computeBoundingSphere(); t.computeBoundingSphere();
      sameArray([...s.boundingSphere.center.toArray(), s.boundingSphere.radius], [...t.boundingSphere.center.toArray(), t.boundingSphere.radius], 'sphere');
    }
  });

  it('rotateX / translate / scale / applyMatrix4 transform in place', () => {
    const { f32, index } = randomMesh();
    const s = meshPair(S, f32, index), t = meshPair(THREE, f32, index);
    for (const [k, args] of [['rotateX', [0.7]], ['rotateY', [-0.3]], ['rotateZ', [1.1]], ['translate', [1, -2, 0.5]], ['scale', [2, 0.5, 1.5]]]) {
      s.g[k](...args); t.g[k](...args);
      sameArray(s.ib.array, t.ib.array, k);
    }
    const e = Array.from({ length: 16 }, () => rnd()); e[3] = e[7] = e[11] = 0; e[15] = 1;
    s.g.applyMatrix4(new S.Matrix4().fromArray(e)); t.g.applyMatrix4(new THREE.Matrix4().fromArray(e));
    sameArray(s.ib.array, t.ib.array, 'applyMatrix4');
    // Still the same attribute objects, on the same buffer — as three.js.
    assert.equal(s.g.getAttribute('position').data, s.ib);
  });

  it('toNonIndexed and computeVertexNormals', () => {
    const { f32, index } = randomMesh();
    const s = meshPair(S, f32, index).g, t = meshPair(THREE, f32, index).g;
    const sn = s.toNonIndexed(), tn = t.toNonIndexed();
    for (const name of ['position', 'normal', 'uv']) sameArray(sn.getAttribute(name).array, tn.getAttribute(name).array, `toNonIndexed ${name}`);
    s.computeVertexNormals(); t.computeVertexNormals();
    // Rust accumulates in f32.
    const tN = t.getAttribute('normal');
    const sN = s.getAttribute('normal');
    for (let i = 0; i < tN.count; i++) {
      for (const g of ['getX', 'getY', 'getZ']) assert.ok(Math.abs(sN[g](i) - tN[g](i)) < 1e-5, `normal ${i} ${g}: ${sN[g](i)} vs ${tN[g](i)}`);
    }
  });

  it('BufferGeometryUtils: interleaveAttributes / deinterleaveAttribute / deinterleaveGeometry / mergeAttributes / estimateBytesUsed', () => {
    const p = randomArray(Float32Array, 30), n = randomArray(Float32Array, 30), u = randomArray(Float32Array, 20);
    const sAttrs = [new S.BufferAttribute(new Float32Array(p), 3), new S.BufferAttribute(new Float32Array(n), 3), new S.BufferAttribute(new Float32Array(u), 2)];
    const tAttrs = [new THREE.BufferAttribute(new Float32Array(p), 3), new THREE.BufferAttribute(new Float32Array(n), 3), new THREE.BufferAttribute(new Float32Array(u), 2)];
    const si = S.BufferGeometryUtils.interleaveAttributes(sAttrs), ti = TU.interleaveAttributes(tAttrs);
    assert.equal(si[0].data, si[2].data);
    sameArray(si[0].data.array, ti[0].data.array, 'interleaveAttributes');
    assert.deepEqual(si.map((a) => [a.itemSize, a.offset]), ti.map((a) => [a.itemSize, a.offset]));
    const sd = S.BufferGeometryUtils.deinterleaveAttribute(si[1]), td = TU.deinterleaveAttribute(ti[1]);
    sameArray(sd.array, td.array, 'deinterleaveAttribute');
    // Two interleaved attributes: three.js refuses to merge an interleaved
    // one with a plain one (their `gpuType`s differ) and returns null.
    const sm = S.BufferGeometryUtils.mergeAttributes([si[0], si[1]]), tm = quiet(() => TU.mergeAttributes([ti[0], ti[1]]));
    assert.ok(tm, 'three.js merged the two interleaved attributes');
    sameArray(sm.array, tm.array, 'mergeAttributes');
    const sg = new S.BufferGeometry(), tg = new THREE.BufferGeometry();
    ['position', 'normal', 'uv'].forEach((k, i) => { sg.setAttribute(k, si[i]); tg.setAttribute(k, ti[i]); });
    assert.equal(S.BufferGeometryUtils.estimateBytesUsed(sg), TU.estimateBytesUsed(tg));
    S.BufferGeometryUtils.deinterleaveGeometry(sg); TU.deinterleaveGeometry(tg);
    for (const k of ['position', 'normal', 'uv']) {
      assert.ok(!sg.getAttribute(k).isInterleavedBufferAttribute, k);
      sameArray(sg.getAttribute(k).array, tg.getAttribute(k).array, `deinterleaveGeometry ${k}`);
    }
  });
});

describe('interleaved geometry in Rust', { skip }, () => {
  /// Interleaved (stride 12: position 3, normal 3, uv 2, colour 4 as
  /// floats) and the same numbers as plain attributes, plus a normalized
  /// Uint8 colour buffer for the interleaved side.
  function build() {
    const verts = 24;
    const f32 = new Float32Array(verts * 8), u8 = new Uint8Array(verts * 4);
    for (let k = 0; k < verts; k++) {
      f32.set([rnd(-2, 2), rnd(-2, 2), rnd(-2, 2), rnd(), rnd(), rnd(), rand(), rand()], k * 8);
      u8.set([0, 1, 2, 3].map(() => Math.floor(rand() * 256)), k * 4);
    }
    const index = Array.from({ length: 60 }, () => Math.floor(rand() * verts));
    const ig = new S.BufferGeometry();
    const ib = new S.InterleavedBuffer(f32, 8), cb = new S.InterleavedBuffer(u8, 4);
    ig.setAttribute('position', new S.InterleavedBufferAttribute(ib, 3, 0));
    ig.setAttribute('normal', new S.InterleavedBufferAttribute(ib, 3, 3));
    ig.setAttribute('uv', new S.InterleavedBufferAttribute(ib, 2, 6));
    ig.setAttribute('color', new S.InterleavedBufferAttribute(cb, 4, 0, true));
    ig.setIndex(index);
    const pg = new S.BufferGeometry();
    const pick = (src, stride, off, size, scale = 1) => {
      const out = new Float32Array(verts * size);
      for (let k = 0; k < verts; k++) for (let c = 0; c < size; c++) out[k * size + c] = src[k * stride + off + c] / scale;
      return out;
    };
    pg.setAttribute('position', new S.BufferAttribute(pick(f32, 8, 0, 3), 3));
    pg.setAttribute('normal', new S.BufferAttribute(pick(f32, 8, 3, 3), 3));
    pg.setAttribute('uv', new S.BufferAttribute(pick(f32, 8, 6, 2), 2));
    pg.setAttribute('color', new S.BufferAttribute(pick(u8, 4, 0, 4, 255), 4));
    pg.setIndex(index);
    return { ig, pg, ib, cb, f32, index };
  }
  const wasmAttrs = (g) => {
    const w = g._syncWasmFromJs();
    return Object.fromEntries(['position', 'normal', 'uv', 'color'].map((k) => [k, [w.getAttributeItemSize(k), Array.from(w.getAttributeArray(k))]]));
  };

  it('the wasm geometry gets the plain equivalent attributes, bit for bit', () => {
    const { ig, pg } = build();
    assert.deepEqual(wasmAttrs(ig), wasmAttrs(pg));
  });

  it('needsUpdate on the shared buffer re-syncs every geometry drawing it', () => {
    const { ig, ib, f32, index } = build();
    // A second geometry over the same buffer.
    const other = new S.BufferGeometry();
    other.setAttribute('position', new S.InterleavedBufferAttribute(ib, 3, 0));
    other.setIndex(index);
    const m1 = new S.Mesh(ig, new S.MeshBasicMaterial()), m2 = new S.Mesh(other, new S.MeshBasicMaterial());
    const scene = new S.Scene(); scene.add(m1, m2);
    const w1 = ig._wasmGeometry(), w2 = other._wasmGeometry();
    const v1 = ig._version, v2 = other._version;
    ig.getAttribute('position').setXYZ(0, 9, 8, 7);
    ib.needsUpdate = true;
    assert.ok(ig._version > v1 && other._version > v2, 'both geometries bumped');
    assert.notEqual(ig._wasmGeometry(), w1); assert.notEqual(other._wasmGeometry(), w2);
    assert.deepEqual(Array.from(other._syncWasmFromJs().getAttributeArray('position').slice(0, 3)), [9, 8, 7]);
    assert.equal(f32[0], 9);
    // The attribute's own setter reaches the buffer's owners too.
    const v3 = ig._version;
    ig.getAttribute('normal').needsUpdate = true;
    assert.ok(ig._version > v3);
  });

  it('raycasts an interleaved mesh as three.js does', () => {
    const { f32, index } = (() => { const b = build(); return { f32: b.f32, index: b.index }; })();
    const mk = (L) => {
      const g = new L.BufferGeometry();
      const ib = new L.InterleavedBuffer(new Float32Array(f32), 8);
      g.setAttribute('position', new L.InterleavedBufferAttribute(ib, 3, 0));
      g.setAttribute('normal', new L.InterleavedBufferAttribute(ib, 3, 3));
      g.setIndex(index);
      const m = new L.Mesh(g, new L.MeshBasicMaterial({ side: L.DoubleSide }));
      m.position.set(0.1, -0.2, 0.3); m.rotation.set(0.3, 0.2, 0.1); m.updateMatrixWorld(true);
      return m;
    };
    const sm = mk(S), tm = mk(THREE);
    let hits = 0;
    for (let r = 0; r < 40; r++) {
      const o = [rnd(-4, 4), rnd(-4, 4), 6], d = [rnd(-0.3, 0.3), rnd(-0.3, 0.3), -1];
      const sr = new S.Raycaster(new S.Vector3(...o), new S.Vector3(...d).normalize());
      const tr = new THREE.Raycaster(new THREE.Vector3(...o), new THREE.Vector3(...d).normalize());
      const sh = sr.intersectObject(sm), th = tr.intersectObject(tm);
      assert.equal(sh.length, th.length, `ray ${r}: hit count`);
      for (let i = 0; i < th.length; i++) {
        assert.ok(Math.abs(sh[i].distance - th[i].distance) < 1e-4, `ray ${r} hit ${i}: distance ${sh[i].distance} vs ${th[i].distance}`);
        assert.equal(sh[i].faceIndex, th[i].faceIndex);
        hits++;
      }
    }
    assert.ok(hits > 0, 'some rays hit');
  });
});
