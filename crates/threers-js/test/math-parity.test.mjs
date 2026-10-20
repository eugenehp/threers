// The shim's math value types — Vector2/3/4, Matrix3, Box2, Box3, Sphere,
// Plane, Ray, Triangle, Frustum, Spherical, Cylindrical, Line3 and the OBB
// addon — against three.js r165 itself (web/node_modules/three), method by
// method on seeded random inputs: the receiver after the call and the result
// must agree to 1e-12. Then the operations that read scene data through Rust
// (`Box3.setFromObject` / `expandByObject` in both modes,
// `BufferGeometry.computeBoundingBox` / `computeBoundingSphere`, the
// object-level bounds of InstancedMesh and SkinnedMesh, and
// `Frustum.intersectsObject` / `intersectsSprite`) on the same small scene
// graph built through either library.
//
// The wasm comes from `THREERS_PKG_DIR` (a wasm-bindgen `--target web` output
// directory) or web/pkg; the suite skips when that build predates the bounds
// bindings.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
const threePath = join(repo, 'web/node_modules/three');

/// The shim as a module whose relative imports point at `pkgDir` and web/.
async function loadShimOver(pkgDir) {
  const pkgUrl = pathToFileURL(join(pkgDir, 'threers.js')).href;
  const pkg = await import(pkgUrl);
  if (typeof pkg.WebScene?.prototype?.vertexBounds !== 'function') return null;
  pkg.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-math-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threePath, 'build/three.module.js'));
const S = ready ? await loadShimOver(pkgDir) : null;
const THREE = ready ? await import(pathToFileURL(join(threePath, 'build/three.module.js')).href) : null;
const OBB3 = ready ? (await import(pathToFileURL(join(threePath, 'examples/jsm/math/OBB.js')).href)).OBB : null;
const skip = !S && `no wasm with the bounds bindings in ${pkgDir}`;

// ---- comparison ----
const TOL = 1e-12;
function close(a, b) {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (a === b) return true;
  return Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b));
}
/// `got` (shim) against `want` (three.js), over three.js's own fields.
function same(got, want, path, seen = new Set()) {
  if (typeof want === 'number') {
    if (typeof got !== 'number' || !close(got, want)) throw new Error(`${path}: ${got} !== ${want}`);
    return;
  }
  if (want === null || want === undefined || typeof want !== 'object') {
    if (got !== want) throw new Error(`${path}: ${String(got)} !== ${String(want)}`);
    return;
  }
  if (seen.has(want)) return;
  seen.add(want);
  if (got === null || typeof got !== 'object') throw new Error(`${path}: ${String(got)} is not an object`);
  if (Array.isArray(want) || ArrayBuffer.isView(want)) {
    if (got.length !== want.length) throw new Error(`${path}.length: ${got.length} !== ${want.length}`);
    for (let i = 0; i < want.length; i++) same(got[i], want[i], `${path}[${i}]`, seen);
    return;
  }
  // Vectors, quaternions and eulers by value; the rest by three.js's fields.
  const keys = 'w' in want || '_w' in want ? ['x', 'y', 'z', 'w'].filter((k) => k in want)
    : 'x' in want ? ['x', 'y', 'z'].filter((k) => k in want)
      : Object.keys(want).filter((k) => typeof want[k] !== 'function' && !k.startsWith('_'));
  for (const k of keys) same(got[k], want[k], `${path}.${k}`, seen);
}

// ---- seeded inputs ----
function rng(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/// An argument description, drawn once and built in each library.
function draw(kind, r) {
  const n = () => (r() - 0.5) * 10;
  const unit = () => { const v = [n(), n(), n()]; const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };
  switch (kind) {
    case 'n': return { k, v: n() };
    case 'p': return { k, v: 0.1 + r() * 5 };
    case 'u': return { k, v: r() };
    case 'a': return { k, v: (r() - 0.5) * 2 * Math.PI };
    case 'i2': return { k, v: Math.floor(r() * 2) };
    case 'i3': return { k, v: Math.floor(r() * 3) };
    case 'i4': return { k, v: Math.floor(r() * 4) };
    case 'b': return { k, v: r() < 0.5 };
    case 'eps': return { k, v: [0, 1e-3, Number.EPSILON][Math.floor(r() * 3)] };
    case 'v2': return { k, v: [n(), n()] };
    case 'v3': return { k, v: [n(), n(), n()] };
    case 'uv3': return { k, v: unit() };
    case 'v4': return { k, v: [n(), n(), n(), n()] };
    case 'q': { const v = [n(), n(), n(), n()]; const l = Math.hypot(...v); return { k, v: v.map((x) => x / l) }; }
    case 'e': return { k, v: [n(), n(), n(), ['XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY'][Math.floor(r() * 6)]] };
    case 'm3': return { k, v: Array.from({ length: 9 }, n) };
    case 'm4': return { k, v: { p: [n(), n(), n()], q: draw('q', r).v, s: [0.2 + r() * 3, 0.2 + r() * 3, 0.2 + r() * 3] } };
    case 'm4r': return { k, v: Array.from({ length: 16 }, n) };
    case 'box2': { const a = [n(), n()], b = [n(), n()]; return { k, v: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])] }; }
    case 'box3': { const a = [n(), n(), n()], b = [n(), n(), n()]; return { k, v: [0, 1, 2].map((i) => Math.min(a[i], b[i])).concat([0, 1, 2].map((i) => Math.max(a[i], b[i]))) }; }
    case 'sph': return { k, v: [n(), n(), n(), 0.1 + r() * 4] };
    case 'pl': return { k, v: [...unit(), n()] };
    case 'ray': return { k, v: [n(), n(), n(), ...unit()] };
    case 'tri': case 'l3': return { k, v: Array.from({ length: kind === 'tri' ? 9 : 6 }, n) };
    case 'sc': return { k, v: [0.1 + r() * 4, r() * Math.PI, (r() - 0.5) * 2 * Math.PI] };
    case 'cy': return { k, v: [0.1 + r() * 4, (r() - 0.5) * 2 * Math.PI, n()] };
    case 'fr': return { k, v: { fov: 30 + r() * 60, aspect: 0.5 + r() * 1.5, near: 0.1 + r(), far: 10 + r() * 50, view: draw('m4', r).v, sys: r() < 0.8 ? 2000 : 2001 } };
    case 'obb': return { k, v: { c: [n(), n(), n()], h: [0.1 + r() * 3, 0.1 + r() * 3, 0.1 + r() * 3], e: [n(), n(), n()] } };
    case 'col': return { k, v: [r(), r(), r()] };
    case 'arr': return { k, v: Array.from({ length: 24 }, n) };
    case 'pts2': return { k, v: Array.from({ length: 6 }, () => [n(), n()]) };
    case 'pts3': return { k, v: Array.from({ length: 6 }, () => [n(), n(), n()]) };
    case 'attr2': case 'attr3': case 'attr4': {
      const size = +kind.slice(4);
      return { k, v: { size, data: Array.from({ length: size * 6 }, n) } };
    }
    case 'cam': return { k, v: { view: draw('m4', r).v, fov: 30 + r() * 60, aspect: 0.5 + r() * 1.5, near: 0.1 + r(), far: 10 + r() * 50 } };
    default:
      if (kind.startsWith('T:') || kind === 'null' || kind === 'undef' || kind === 'emptyArr' || kind === 'off') return { k };
      throw new Error(`unknown kind ${kind}`);
  }
  function k() { return kind; }
}

/// Build a drawn argument in library `L` (THREE or the shim).
function build(L, d) {
  const kind = d.k(), v = d.v;
  const v3 = (a) => new L.Vector3(a[0], a[1], a[2]);
  const m4 = (o) => new L.Matrix4().compose(v3(o.p), new L.Quaternion(o.q[0], o.q[1], o.q[2], o.q[3]), v3(o.s));
  switch (kind) {
    case 'n': case 'p': case 'u': case 'a': case 'i2': case 'i3': case 'i4': case 'b': case 'eps': return v;
    case 'v2': return new L.Vector2(v[0], v[1]);
    case 'v3': case 'uv3': return v3(v);
    case 'v4': return new L.Vector4(v[0], v[1], v[2], v[3]);
    case 'q': return new L.Quaternion(v[0], v[1], v[2], v[3]);
    case 'e': return new L.Euler(v[0], v[1], v[2], v[3]);
    case 'm3': return new L.Matrix3().fromArray(v);
    case 'm4': return m4(v);
    case 'm4r': return new L.Matrix4().fromArray(v);
    case 'box2': return new L.Box2(new L.Vector2(v[0], v[1]), new L.Vector2(v[2], v[3]));
    case 'box3': return new L.Box3(v3(v), v3(v.slice(3)));
    case 'sph': return new L.Sphere(v3(v), v[3]);
    case 'pl': return new L.Plane(v3(v), v[3]);
    case 'ray': return new L.Ray(v3(v), v3(v.slice(3)));
    case 'tri': return new L.Triangle(v3(v), v3(v.slice(3)), v3(v.slice(6)));
    case 'l3': return new L.Line3(v3(v), v3(v.slice(3)));
    case 'sc': return new L.Spherical(v[0], v[1], v[2]);
    case 'cy': return new L.Cylindrical(v[0], v[1], v[2]);
    case 'fr': {
      const p = new L.Matrix4().makePerspective(-v.near * v.aspect, v.near * v.aspect, v.near, -v.near, v.near, v.far);
      return new L.Frustum().setFromProjectionMatrix(p.multiply(m4(v.view).invert()), v.sys);
    }
    case 'obb': {
      const O = L === THREE ? OBB3 : S.OBB;
      return new O(v3(v.c), v3(v.h), new L.Matrix3().setFromMatrix4(new L.Matrix4().makeRotationFromEuler(new L.Euler(v.e[0], v.e[1], v.e[2]))));
    }
    case 'col': return { r: v[0], g: v[1], b: v[2] };
    case 'arr': return v.slice();
    case 'pts2': return v.map((p) => new L.Vector2(p[0], p[1]));
    case 'pts3': return v.map((p) => v3(p));
    case 'attr2': case 'attr3': case 'attr4': return new L.BufferAttribute(new Float32Array(v.data), v.size);
    case 'cam': {
      const mw = m4(v.view);
      const pm = new L.Matrix4().makePerspective(-v.near * v.aspect, v.near * v.aspect, v.near, -v.near, v.near, v.far);
      return { matrixWorld: mw, matrixWorldInverse: mw.clone().invert(), projectionMatrix: pm, projectionMatrixInverse: pm.clone().invert() };
    }
    case 'null': return null;
    case 'undef': return undefined;
    case 'emptyArr': return [];
    case 'off': return 3;
    default: {
      const name = kind.slice(2);
      if (name === 'OBB') return L === THREE ? new OBB3() : new S.OBB();
      return new L[name]();
    }
  }
}

/// `[receiver kind, method, ...argument kinds]` for every method of a class.
const vecCommon = (V, t) => [
  `set:${t === 'v2' ? 'n,n' : t === 'v3' ? 'n,n,n' : 'n,n,n,n'}`, 'setScalar:n', 'setX:n', 'setY:n',
  `setComponent:${t === 'v2' ? 'i2' : t === 'v3' ? 'i3' : 'i4'},n`, `getComponent:${t === 'v2' ? 'i2' : t === 'v3' ? 'i3' : 'i4'}`,
  'clone:', `copy:${t}`, `add:${t}`, 'addScalar:n', `addVectors:${t},${t}`, `addScaledVector:${t},n`, `sub:${t}`,
  'subScalar:n', `subVectors:${t},${t}`, `multiply:${t}`, 'multiplyScalar:n', 'divideScalar:n', 'divideScalar:p',
  `min:${t}`, `max:${t}`, `clamp:${t},${t}`, 'clampScalar:n,p', 'clampLength:p,p', 'floor:', 'ceil:', 'round:',
  'roundToZero:', 'negate:', `dot:${t}`, 'lengthSq:', 'length:', 'manhattanLength:', 'normalize:', 'setLength:n',
  `lerp:${t},u`, `lerpVectors:${t},${t},u`, `equals:${t}`, 'fromArray:arr', 'fromArray:arr,off', 'toArray:',
  'toArray:emptyArr,off', `fromBufferAttribute:attr${t.slice(1)},i3`, 'random:',
].map((s) => `${V}|${s}`);
const SPECS = [
  ...vecCommon('v2', 'v2'),
  ...['width', 'height'].map((g) => `v2|@${g}`),
  ...['divide:v2', 'applyMatrix3:m3', 'cross:v2', 'angle:', 'angleTo:v2', 'distanceTo:v2', 'distanceToSquared:v2',
    'manhattanDistanceTo:v2', 'rotateAround:v2,a'].map((s) => `v2|${s}`),
  ...vecCommon('v3', 'v3'),
  ...['setZ:n', 'multiplyVectors:v3,v3', 'applyEuler:e', 'applyAxisAngle:uv3,a', 'applyMatrix3:m3', 'applyNormalMatrix:m3',
    'applyMatrix4:m4', 'applyMatrix4:m4r', 'applyQuaternion:q', 'project:cam', 'unproject:cam', 'transformDirection:m4',
    'divide:v3', 'cross:v3', 'crossVectors:v3,v3', 'projectOnVector:v3', 'projectOnPlane:uv3', 'reflect:uv3', 'angleTo:v3',
    'distanceTo:v3', 'distanceToSquared:v3', 'manhattanDistanceTo:v3', 'setFromSpherical:sc', 'setFromSphericalCoords:p,a,a',
    'setFromCylindrical:cy', 'setFromCylindricalCoords:p,a,n', 'setFromMatrixPosition:m4', 'setFromMatrixScale:m4',
    'setFromMatrixColumn:m4r,i4', 'setFromMatrix3Column:m3,i3', 'setFromEuler:e', 'setFromColor:col', 'randomDirection:'].map((s) => `v3|${s}`),
  ...vecCommon('v4', 'v4'),
  ...['width', 'height'].map((g) => `v4|@${g}`),
  ...['setZ:n', 'setW:n', 'applyMatrix4:m4r', 'setAxisAngleFromQuaternion:q', 'setAxisAngleFromRotationMatrix:m4',
    'setAxisAngleFromRotationMatrix:m4r'].map((s) => `v4|${s}`),
  ...['set:n,n,n,n,n,n,n,n,n', 'identity:', 'copy:m3', 'extractBasis:T:Vector3,T:Vector3,T:Vector3', 'setFromMatrix4:m4r',
    'multiply:m3', 'premultiply:m3', 'multiplyMatrices:m3,m3', 'multiplyScalar:n', 'determinant:', 'invert:', 'transpose:',
    'getNormalMatrix:m4', 'transposeIntoArray:emptyArr', 'setUvTransform:n,n,p,p,a,n,n', 'scale:n,n', 'rotate:a',
    'translate:n,n', 'makeTranslation:n,n', 'makeTranslation:v2', 'makeRotation:a', 'makeScale:n,n', 'equals:m3',
    'fromArray:arr', 'fromArray:arr,off', 'toArray:', 'toArray:emptyArr,off', 'clone:'].map((s) => `m3|${s}`),
  ...['set:v2,v2', 'setFromPoints:pts2', 'setFromCenterAndSize:v2,v2', 'clone:', 'copy:box2', 'makeEmpty:', 'isEmpty:',
    'getCenter:T:Vector2', 'getSize:T:Vector2', 'expandByPoint:v2', 'expandByVector:v2', 'expandByScalar:n',
    'containsPoint:v2', 'containsBox:box2', 'getParameter:v2,T:Vector2', 'intersectsBox:box2', 'clampPoint:v2,T:Vector2',
    'distanceToPoint:v2', 'intersect:box2', 'union:box2', 'translate:v2', 'equals:box2'].map((s) => `box2|${s}`),
  ...['set:v3,v3', 'setFromArray:arr', 'setFromBufferAttribute:attr3', 'setFromPoints:pts3', 'setFromCenterAndSize:v3,v3',
    'clone:', 'copy:box3', 'makeEmpty:', 'isEmpty:', 'getCenter:T:Vector3', 'getSize:T:Vector3', 'expandByPoint:v3',
    'expandByVector:v3', 'expandByScalar:n', 'containsPoint:v3', 'containsBox:box3', 'getParameter:v3,T:Vector3',
    'intersectsBox:box3', 'intersectsSphere:sph', 'intersectsPlane:pl', 'intersectsTriangle:tri', 'clampPoint:v3,T:Vector3',
    'distanceToPoint:v3', 'getBoundingSphere:T:Sphere', 'intersect:box3', 'union:box3', 'applyMatrix4:m4',
    'applyMatrix4:m4r', 'translate:v3', 'equals:box3'].map((s) => `box3|${s}`),
  ...['set:v3,p', 'setFromPoints:pts3', 'setFromPoints:pts3,v3', 'copy:sph', 'isEmpty:', 'makeEmpty:', 'containsPoint:v3',
    'distanceToPoint:v3', 'intersectsSphere:sph', 'intersectsBox:box3', 'intersectsPlane:pl', 'clampPoint:v3,T:Vector3',
    'getBoundingBox:T:Box3', 'applyMatrix4:m4', 'translate:v3', 'expandByPoint:v3', 'union:sph', 'equals:sph',
    'clone:'].map((s) => `sph|${s}`),
  ...['set:uv3,n', 'setComponents:n,n,n,n', 'setFromNormalAndCoplanarPoint:uv3,v3', 'setFromCoplanarPoints:v3,v3,v3',
    'copy:pl', 'normalize:', 'negate:', 'distanceToPoint:v3', 'distanceToSphere:sph', 'projectPoint:v3,T:Vector3',
    'intersectLine:l3,T:Vector3', 'intersectsLine:l3', 'intersectsBox:box3', 'intersectsSphere:sph',
    'coplanarPoint:T:Vector3', 'applyMatrix4:m4', 'applyMatrix4:m4,m3', 'translate:v3', 'equals:pl', 'clone:'].map((s) => `pl|${s}`),
  ...['set:v3,uv3', 'copy:ray', 'at:n,T:Vector3', 'lookAt:v3', 'recast:n', 'closestPointToPoint:v3,T:Vector3',
    'distanceToPoint:v3', 'distanceSqToPoint:v3', 'distanceSqToSegment:v3,v3', 'distanceSqToSegment:v3,v3,T:Vector3,T:Vector3',
    'intersectSphere:sph,T:Vector3', 'intersectsSphere:sph', 'distanceToPlane:pl', 'intersectPlane:pl,T:Vector3',
    'intersectsPlane:pl', 'intersectBox:box3,T:Vector3', 'intersectsBox:box3', 'intersectTriangle:v3,v3,v3,b,T:Vector3',
    'applyMatrix4:m4', 'equals:ray', 'clone:'].map((s) => `ray|${s}`),
  ...['set:v3,v3,v3', 'setFromPointsAndIndices:pts3,i3,i3,i3', 'setFromAttributeAndIndices:attr3,i3,i3,i3', 'clone:',
    'copy:tri', 'getArea:', 'getMidpoint:T:Vector3', 'getNormal:T:Vector3', 'getPlane:T:Plane', 'getBarycoord:v3,T:Vector3',
    'getInterpolation:v3,v2,v2,v2,T:Vector2', 'getInterpolation:v3,v3,v3,v3,T:Vector3', 'containsPoint:v3',
    'isFrontFacing:v3', 'intersectsBox:box3', 'closestPointToPoint:v3,T:Vector3', 'equals:tri'].map((s) => `tri|${s}`),
  ...['getNormal:v3,v3,v3,T:Vector3', 'getBarycoord:v3,v3,v3,v3,T:Vector3', 'containsPoint:v3,v3,v3,v3',
    'getInterpolation:v3,v3,v3,v3,v2,v2,v2,T:Vector2', 'isFrontFacing:v3,v3,v3,v3'].map((s) => `static:Triangle|${s}`),
  ...['set:pl,pl,pl,pl,pl,pl', 'copy:fr', 'intersectsSphere:sph', 'intersectsBox:box3', 'containsPoint:v3',
    'clone:'].map((s) => `fr|${s}`),
  ...['set:p,a,a', 'copy:sc', 'makeSafe:', 'setFromVector3:v3', 'setFromCartesianCoords:n,n,n', 'clone:'].map((s) => `sc|${s}`),
  ...['set:p,a,n', 'copy:cy', 'setFromVector3:v3', 'setFromCartesianCoords:n,n,n', 'clone:'].map((s) => `cy|${s}`),
  ...['set:v3,v3', 'copy:l3', 'getCenter:T:Vector3', 'delta:T:Vector3', 'distanceSq:', 'distance:', 'at:n,T:Vector3',
    'closestPointToPointParameter:v3,b', 'closestPointToPoint:v3,b,T:Vector3', 'applyMatrix4:m4', 'equals:l3',
    'clone:'].map((s) => `l3|${s}`),
  ...['set:v3,v3,m3', 'copy:obb', 'clone:', 'getSize:T:Vector3', 'clampPoint:v3,T:Vector3', 'containsPoint:v3',
    'intersectsBox3:box3', 'intersectsSphere:sph', 'intersectsOBB:obb', 'intersectsOBB:obb,eps', 'intersectsPlane:pl',
    'intersectRay:ray,T:Vector3', 'intersectsRay:ray', 'fromBox3:box3', 'equals:obb', 'applyMatrix4:m4'].map((s) => `obb|${s}`),
];

const RUNS = 25;

/// Run one spec on both libraries with the same draws; throw on a mismatch.
function runSpec(spec, seed) {
  const [recv, rest] = spec.split('|');
  const [method, argList] = rest.split(':', 2).length === 2 ? [rest.slice(0, rest.indexOf(':')), rest.slice(rest.indexOf(':') + 1)] : [rest, ''];
  const r = rng(seed);
  const argKinds = argList ? argList.match(/T:\w+|[^,]+/g) : [];
  const recvDraw = recv.startsWith('static:') ? null : draw(recv, r);
  const argDraws = argKinds.map((k) => draw(k, r));
  const results = [];
  for (const L of [THREE, S]) {
    const self = recvDraw ? build(L, recvDraw) : L[recv.slice(7)];
    const args = argDraws.map((d) => build(L, d));
    // Seeded Math.random, for `random()` / `randomDirection()`.
    const realRandom = Math.random;
    Math.random = rng(seed ^ 0x5bd1e995);
    let out;
    try {
      out = method.startsWith('@') ? self[method.slice(1)] : self[method](...args);
    } finally {
      Math.random = realRandom;
    }
    results.push({ self, out, args });
  }
  const [want, got] = results;
  same(got.out, want.out, `${spec} result`);
  if (recvDraw) same(got.self, want.self, `${spec} receiver`);
  for (let i = 0; i < want.args.length; i++) {
    if (/^(attr|cam)/.test(argKinds[i])) continue; // not math types; read only
    same(got.args[i], want.args[i], `${spec} arg ${i}`);
  }
}

describe('math value types match three.js r165', { skip }, () => {
  const groups = new Map();
  for (const spec of SPECS) {
    const key = spec.split('|')[0];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(spec);
  }
  for (const [key, specs] of groups) {
    it(`${key}: ${specs.length} methods × ${RUNS} random inputs`, () => {
      const failures = [];
      for (const spec of specs) {
        for (let run = 0; run < RUNS; run++) {
          try { runSpec(spec, 1000 * run + specs.indexOf(spec) + 7); } catch (e) { failures.push(`${e.message}`); break; }
        }
      }
      assert.deepEqual(failures, []);
    });
  }

  it('covers every method three.js defines on these classes', () => {
    const covered = new Map();
    for (const spec of SPECS) {
      const [recv, rest] = spec.split('|');
      const m = rest.split(':')[0].replace('@', '');
      (covered.get(recv) ?? covered.set(recv, new Set()).get(recv)).add(m);
    }
    const classes = { v2: 'Vector2', v3: 'Vector3', v4: 'Vector4', m3: 'Matrix3', box2: 'Box2', box3: 'Box3', sph: 'Sphere',
      pl: 'Plane', ray: 'Ray', tri: 'Triangle', fr: 'Frustum', sc: 'Spherical', cy: 'Cylindrical', l3: 'Line3' };
    // Scene-data methods are checked on a scene below.
    const elsewhere = new Set(['setFromObject', 'expandByObject', 'intersectsObject', 'intersectsSprite', 'setFromProjectionMatrix', 'constructor']);
    const missing = [];
    for (const [k, name] of Object.entries(classes)) {
      const proto = THREE[name].prototype;
      for (const m of Object.getOwnPropertyNames(proto)) {
        if (elsewhere.has(m) || m.startsWith('is') || covered.get(k)?.has(m)) continue;
        if (name === 'Vector2' || name === 'Vector4') { if (m === 'width' || m === 'height') continue; }
        missing.push(`${name}.${m}`);
      }
      // …and the shim has each of them.
      for (const m of Object.getOwnPropertyNames(proto)) {
        if (!(m in S[name].prototype)) missing.push(`shim lacks ${name}.${m}`);
      }
    }
    for (const m of Object.getOwnPropertyNames(OBB3.prototype)) {
      if (!covered.get('obb')?.has(m) && m !== 'constructor') missing.push(`OBB.${m}`);
      if (!(m in S.OBB.prototype)) missing.push(`shim lacks OBB.${m}`);
    }
    assert.deepEqual(missing, []);
  });
});

// ---- scene data ----

/// The same little scene in either library: a mesh (indexed BufferGeometry,
/// morph targets, rotated/scaled parent), points, a line, an instanced mesh, a
/// skinned mesh posed by two bones, and a sprite.
function scene(L, opts = {}) {
  const r = rng(99);
  const n = () => (r() - 0.5) * 6;
  const root = new L.Group();
  root.position.set(0.5, -1.25, 2);
  root.rotation.set(0.3, -0.7, 1.1);
  root.scale.set(1.5, 0.75, 2);
  const geo = new L.BufferGeometry();
  const pos = new Float32Array(Array.from({ length: 30 }, n));
  geo.setAttribute('position', new L.BufferAttribute(pos, 3));
  geo.setIndex([0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]);
  geo.morphAttributes.position = [
    new L.BufferAttribute(new Float32Array(Array.from({ length: 30 }, n)), 3),
    new L.BufferAttribute(new Float32Array(Array.from({ length: 30 }, n)), 3),
  ];
  if (opts.relative) geo.morphTargetsRelative = true;
  const mesh = new L.Mesh(geo, new L.MeshBasicMaterial());
  mesh.morphTargetInfluences = [0.25, 0.6];
  mesh.position.set(1, 2, 3);
  mesh.quaternion.set(0.1, 0.7, -0.2, 0.67).normalize();
  root.add(mesh);
  const pts = new L.Points(new L.BufferGeometry().setAttribute('position', new L.BufferAttribute(new Float32Array(Array.from({ length: 18 }, n)), 3)), new L.PointsMaterial());
  pts.position.set(-3, 0.5, 1);
  mesh.add(pts);
  const line = new L.Line(new L.BufferGeometry().setAttribute('position', new L.BufferAttribute(new Float32Array(Array.from({ length: 15 }, n)), 3)), new L.LineBasicMaterial());
  line.rotation.set(1, 2, 3);
  root.add(line);
  const im = new L.InstancedMesh(new L.BoxGeometry(1, 2, 3), new L.MeshBasicMaterial(), 3);
  const m = new L.Matrix4();
  for (let i = 0; i < 3; i++) im.setMatrixAt(i, m.compose(new L.Vector3(n(), n(), n()), new L.Quaternion(r(), r(), r(), r()).normalize(), new L.Vector3(1 + r(), 1 + r(), 1 + r())));
  im.position.set(4, -2, 0);
  root.add(im);
  // A skinned strip bound to two bones, the second bent.
  const sg = new L.BufferGeometry();
  const sp = [], si = [], sw = [];
  for (let i = 0; i < 8; i++) { sp.push((i % 2) - 0.5, Math.floor(i / 2), 0); si.push(0, 1, 0, 0); const w = Math.floor(i / 2) / 3; sw.push(1 - w, w, 0, 0); }
  sg.setAttribute('position', new L.BufferAttribute(new Float32Array(sp), 3));
  sg.setAttribute('skinIndex', new L.Uint16BufferAttribute(si, 4));
  sg.setAttribute('skinWeight', new L.Float32BufferAttribute(sw, 4));
  const b0 = new L.Bone(), b1 = new L.Bone();
  b1.position.y = 1.5; b0.add(b1);
  const sm = new L.SkinnedMesh(sg, new L.MeshBasicMaterial());
  sm.add(b0);
  sm.bind(new L.Skeleton([b0, b1]));
  b1.rotation.z = 0.6; b0.rotation.x = 0.2;
  sm.position.set(-2, 1, -1);
  root.add(sm);
  const sprite = new L.Sprite(new L.SpriteMaterial());
  sprite.position.set(2, 3, -4);
  sprite.scale.set(2, 3, 1);
  root.add(sprite);
  // A built-in geometry, as three.js builds it.
  const box = new L.Mesh(new L.BoxGeometry(1, 2, 3), new L.MeshBasicMaterial());
  box.position.set(-1, -2, 5);
  box.rotation.set(0.4, 0.5, 0.6);
  root.add(box);
  root.updateMatrixWorld(true);
  return { root, mesh, pts, line, im, sm, sprite, box };
}

describe('bounds from scene data match three.js r165', { skip }, () => {
  for (const relative of [false, true]) {
    it(`Box3.setFromObject, plain and precise (morphTargetsRelative ${relative})`, () => {
      const a = scene(THREE, { relative }), b = scene(S, { relative });
      for (const precise of [false, true]) {
        for (const k of Object.keys(a)) {
          same(new S.Box3().setFromObject(b[k], precise), new THREE.Box3().setFromObject(a[k], precise), `${k} precise=${precise}`);
        }
        same(new S.Box3(new S.Vector3(-20, 0, 0), new S.Vector3(-19, 1, 1)).expandByObject(b.root, precise),
          new THREE.Box3(new THREE.Vector3(-20, 0, 0), new THREE.Vector3(-19, 1, 1)).expandByObject(a.root, precise), `expandByObject precise=${precise}`);
      }
    });
    it(`geometry and object bounding volumes (morphTargetsRelative ${relative})`, () => {
      const a = scene(THREE, { relative }), b = scene(S, { relative });
      for (const k of ['mesh', 'pts', 'line']) {
        a[k].geometry.computeBoundingBox(); b[k].geometry.computeBoundingBox();
        a[k].geometry.computeBoundingSphere(); b[k].geometry.computeBoundingSphere();
        same(b[k].geometry.boundingBox, a[k].geometry.boundingBox, `${k} boundingBox`);
        same(b[k].geometry.boundingSphere, a[k].geometry.boundingSphere, `${k} boundingSphere`);
      }
      for (const k of ['im', 'sm']) {
        a[k].computeBoundingBox(); b[k].computeBoundingBox();
        a[k].computeBoundingSphere(); b[k].computeBoundingSphere();
        same(b[k].boundingBox, a[k].boundingBox, `${k} boundingBox`);
        same(b[k].boundingSphere, a[k].boundingSphere, `${k} boundingSphere`);
      }
    });
  }
  it('Frustum.intersectsObject / intersectsSprite', () => {
    const a = scene(THREE), b = scene(S);
    const r = rng(5);
    let hits = 0, misses = 0;
    for (let run = 0; run < 40; run++) {
      const d = draw('fr', r);
      const fa = build(THREE, d), fb = build(S, d);
      for (const k of ['mesh', 'pts', 'line', 'im', 'sm', 'box']) {
        const want = fa.intersectsObject(a[k]);
        assert.equal(fb.intersectsObject(b[k]), want, `${k} run ${run}`);
        want ? hits++ : misses++;
      }
      assert.equal(fb.intersectsSprite(b.sprite), fa.intersectsSprite(a.sprite), `sprite run ${run}`);
    }
    assert.ok(hits > 0 && misses > 0, `both outcomes exercised (${hits} hits, ${misses} misses)`);
    for (const k of ['mesh', 'pts', 'line']) same(b[k].geometry.boundingSphere, a[k].geometry.boundingSphere, `${k} cached sphere`);
  });
  it('Frustum.setFromProjectionMatrix in both coordinate systems', () => {
    const r = rng(17);
    for (let run = 0; run < 30; run++) {
      const d = draw('fr', r);
      same(build(S, d), build(THREE, d), `run ${run}`);
    }
  });
});
