// The shim's Shape / Path / ShapePath / ShapeUtils / ShapeGeometry /
// ExtrudeGeometry / Font / TextGeometry, run over a real wasm build, against
// three.js r165 itself (web/node_modules/three) on the same calls. The shapes
// are Rust (`curves::shapes`, `geometries::{ShapeGeometry, ExtrudeGeometry}`),
// so this checks the bindings and the JS wrappers end to end; buffers must
// match bit for bit.
//
// The wasm comes from `THREERS_PKG_DIR` (a wasm-bindgen `--target web` output
// directory) or web/pkg; the suite skips when that build predates the shape
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
  if (typeof pkg.WebShapeCurvePath !== 'function') return null;
  pkg.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-shapes-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threePath, 'build/three.module.js'));
const T = ready ? await loadShimOver(pkgDir) : null;
const THREE = ready ? await import(pathToFileURL(join(threePath, 'build/three.module.js')).href) : null;
const addons = ready ? {
  Font: (await import(pathToFileURL(join(threePath, 'examples/jsm/loaders/FontLoader.js')).href)).Font,
  TextGeometry: (await import(pathToFileURL(join(threePath, 'examples/jsm/geometries/TextGeometry.js')).href)).TextGeometry,
} : null;
const skip = !T && `no wasm with shape bindings in ${pkgDir}`;

/// The same shape built through either library.
function mixed(L) {
  const s = new L.Shape();
  s.moveTo(0, 0); s.lineTo(2, 0); s.quadraticCurveTo(3, 0.5, 2.5, 1.5);
  s.bezierCurveTo(2.2, 2.5, 1.0, 2.8, 0.5, 2.0);
  s.splineThru([new L.Vector2(0.2, 1.6), new L.Vector2(-0.4, 1.2), new L.Vector2(-0.3, 0.6)]);
  s.absellipse(-0.5, 0.3, 0.4, 0.25, 0.3, 4.0, true, 0.2);
  s.closePath();
  const hole = new L.Path(); hole.absarc(1.2, 1.0, 0.35, 0, Math.PI * 2, false);
  s.holes.push(hole);
  const hole2 = new L.Path(); hole2.moveTo(1.8, 0.3); hole2.lineTo(2.1, 0.3); hole2.lineTo(1.95, 0.6);
  s.holes.push(hole2);
  return s;
}
const triangle = (L) => new L.Shape([new L.Vector2(0, 0.5), new L.Vector2(-0.5, -0.5), new L.Vector2(0.5, -0.5)]);

function sameGeometry(ours, theirs, name) {
  for (const a of ['position', 'normal', 'uv']) {
    const x = ours.attributes[a]?.array, y = theirs.attributes[a].array;
    assert.ok(x, `${name}: no ${a}`);
    assert.equal(x.length, y.length, `${name}.${a} length`);
    for (let i = 0; i < y.length; i++) {
      if (!Object.is(Math.fround(x[i]), y[i])) assert.fail(`${name}.${a}[${i}]: ${x[i]} vs three.js ${y[i]}`);
    }
  }
  assert.deepEqual(ours.index ? Array.from(ours.index.array) : null, theirs.index ? Array.from(theirs.index.array) : null, `${name}: index`);
  assert.deepEqual(ours.groups.map((g) => [g.start, g.count, g.materialIndex]),
    theirs.groups.map((g) => [g.start, g.count, g.materialIndex]), `${name}: groups`);
}
const xy = (pts) => pts.flatMap((p) => [p.x, p.y]);
/// f64 point lists: equal to the last bit or two. Sines and cosines come from
/// Rust's libm rather than V8's, which can round an ulp apart; once stored as
/// f32 (every buffer above) that vanishes, and those compare exactly.
function closePoints(a, b, msg) {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 4e-16 * Math.max(1, Math.abs(b[i]))) assert.fail(`${msg}[${i}]: ${a[i]} vs three.js ${b[i]}`);
  }
}

describe('shim shapes over wasm vs three.js r165', { skip }, () => {
  it('samples paths identically', () => {
    const a = mixed(T), b = mixed(THREE);
    for (const d of [3, 12]) closePoints(xy(a.getPoints(d)), xy(b.getPoints(d)), `getPoints(${d})`);
    closePoints(xy(a.getSpacedPoints(9)), xy(b.getSpacedPoints(9)), 'getSpacedPoints');
    closePoints([a.getLength()], [b.getLength()], 'getLength');
    closePoints(xy([a.getPointAt(0.3)]), xy([b.getPointAt(0.3)]), 'getPointAt');
    closePoints(xy(a.extractPoints(4).holes.flat()), xy(b.extractPoints(4).holes.flat()), 'holes');
    assert.equal(a.curves.length, b.curves.length);
    assert.deepEqual(a.curves.map((c) => c.type), b.curves.map((c) => c.type));
  });
  it('ShapeUtils.triangulateShape, including the in-place closing-point pop', () => {
    const pts = (L) => [new L.Vector2(0, 0), new L.Vector2(4, 0), new L.Vector2(4, 4), new L.Vector2(0, 4), new L.Vector2(0, 0)];
    const hole = (L) => [new L.Vector2(1, 1), new L.Vector2(1, 2), new L.Vector2(2, 2), new L.Vector2(1, 1)];
    const ca = pts(T), ha = [hole(T)], cb = pts(THREE), hb = [hole(THREE)];
    assert.deepEqual(T.ShapeUtils.triangulateShape(ca, ha), THREE.ShapeUtils.triangulateShape(cb, hb));
    assert.equal(ca.length, cb.length);
    assert.equal(ha[0].length, hb[0].length);
    assert.equal(T.ShapeUtils.area(pts(T)), THREE.ShapeUtils.area(pts(THREE)));
    assert.equal(T.ShapeUtils.isClockWise(hole(T)), THREE.ShapeUtils.isClockWise(hole(THREE)));
  });
  it('ShapeGeometry, single and grouped', () => {
    sameGeometry(new T.ShapeGeometry(triangle(T)), new THREE.ShapeGeometry(triangle(THREE)), 'single');
    sameGeometry(new T.ShapeGeometry([mixed(T), triangle(T)], 5), new THREE.ShapeGeometry([mixed(THREE), triangle(THREE)], 5), 'grouped');
  });
  it('ExtrudeGeometry: bevel defaults, no bevel, options', () => {
    sameGeometry(new T.ExtrudeGeometry(mixed(T), { curveSegments: 4 }), new THREE.ExtrudeGeometry(mixed(THREE), { curveSegments: 4 }), 'defaults');
    const o = { depth: 0.3, steps: 3, bevelEnabled: false };
    sameGeometry(new T.ExtrudeGeometry([mixed(T), triangle(T)], o), new THREE.ExtrudeGeometry([mixed(THREE), triangle(THREE)], o), 'no bevel');
    const b = { depth: 0.35, bevelSize: 0.08, bevelThickness: 0.08, bevelOffset: -0.01, bevelSegments: 2 };
    sameGeometry(new T.ExtrudeGeometry(triangle(T), b), new THREE.ExtrudeGeometry(triangle(THREE), b), 'bevel options');
  });
  it('ExtrudeGeometry along a three.js curve (sampled in JS, swept in Rust)', () => {
    // The path is three.js's own CatmullRomCurve3 on both sides: only the
    // sweep differs.
    const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(-2, 0, 0), new THREE.Vector3(-1, 1, 0.5), new THREE.Vector3(1, -0.5, 1), new THREE.Vector3(2, 0.5, -0.5)]);
    sameGeometry(new T.ExtrudeGeometry(triangle(T), { steps: 8, extrudePath: curve }),
      new THREE.ExtrudeGeometry(triangle(THREE), { steps: 8, extrudePath: curve }), 'extrudePath');
  });
  it('a custom UVGenerator is called like three.js calls it', () => {
    const gen = {
      generateTopUV: (_g, v, a, b, c) => [a, b, c].map((i) => new THREE.Vector2(v[i * 3] * 2, v[i * 3 + 2])),
      generateSideWallUV: (_g, v, a, b, c, d) => [a, b, c, d].map((i) => new THREE.Vector2(v[i * 3 + 1], v.length)),
    };
    sameGeometry(new T.ExtrudeGeometry(mixed(T), { UVGenerator: gen, curveSegments: 3 }),
      new THREE.ExtrudeGeometry(mixed(THREE), { UVGenerator: gen, curveSegments: 3 }), 'UVGenerator');
  });
  it('ShapePath.toShapes sorts holes like three.js', () => {
    const build = (L) => {
      const p = new L.ShapePath();
      const loop = (pts) => { p.moveTo(...pts[0]); for (const q of pts.slice(1)) p.lineTo(...q); };
      loop([[0, 0], [0, 3], [3, 3], [3, 0]]);
      loop([[5, 0], [5, 3], [8, 3], [8, 0]]);
      loop([[1, 1], [2, 1], [2, 2], [1, 2]]);
      loop([[6, 1], [7, 1], [7, 2], [6, 2]]);
      return p;
    };
    for (const ccw of [false, true]) {
      const a = build(T).toShapes(ccw), b = build(THREE).toShapes(ccw);
      assert.deepEqual(a.map((s) => [xy(s.getPoints()), s.holes.map((h) => xy(h.getPoints()))]),
        b.map((s) => [xy(s.getPoints()), s.holes.map((h) => xy(h.getPoints()))]), `isCCW ${ccw}`); // lines only: exact
    }
  });
  it('Font + TextGeometry', () => {
    const json = JSON.parse(readFileSync(join(threePath, 'examples/fonts/helvetiker_regular.typeface.json'), 'utf8'));
    const ours = new T.Font(json), theirs = new addons.Font(json);
    const sa = ours.generateShapes('Ag8&\nB?é', 3), sb = theirs.generateShapes('Ag8&\nB?é', 3);
    assert.equal(sa.length, sb.length);
    sa.forEach((s, i) => {
      closePoints(xy(s.getPoints(4)), xy(sb[i].getPoints(4)), `glyph shape ${i}`);
      assert.equal(s.holes.length, sb[i].holes.length);
    });
    const p = { size: 0.5, depth: 0.1, curveSegments: 4, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.01, bevelSegments: 2 };
    sameGeometry(new T.TextGeometry('Ag8', { font: ours, ...p }), new addons.TextGeometry('Ag8', { font: theirs, ...p }), 'text');
    sameGeometry(new T.TextGeometry('o', { font: ours, size: 2 }), new addons.TextGeometry('o', { font: theirs, size: 2 }), 'text defaults');
  });
});
