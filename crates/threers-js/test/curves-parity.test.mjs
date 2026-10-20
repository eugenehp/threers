// The shim's Curve hierarchy — every three.js r165 curve, CurvePath, the base
// class's arc-length machinery, and user-defined `Curve` subclasses — run over
// a real wasm build against three.js r165 itself (web/node_modules/three).
// The curves are Rust (`curves::three`, f64 as JavaScript computes); f64
// results must agree to 1e-12, and the TubeGeometry / ExtrudeGeometry buffers
// built from them bit for bit (as f32).
//
// The wasm comes from `THREERS_PKG_DIR` (a wasm-bindgen `--target web` output
// directory) or web/pkg; the suite skips when that build predates `WebCurve`.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
const threePath = join(repo, 'web/node_modules/three');

async function loadShimOver(pkgDir) {
  const pkgUrl = pathToFileURL(join(pkgDir, 'threers.js')).href;
  const pkg = await import(pkgUrl);
  if (typeof pkg.WebCurve !== 'function') return null;
  pkg.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-curves-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threePath, 'build/three.module.js'));
const T = ready ? await loadShimOver(pkgDir) : null;
const THREE = ready ? await import(pathToFileURL(join(threePath, 'build/three.module.js')).href) : null;
const skip = !T && `no wasm with WebCurve in ${pkgDir}`;

// f64 results agree to 1e-12. Rust's libm and V8's sin / cos / acos / pow
// can round an ulp apart (about one call in a hundred for sin / cos), and
// three.js's tangent is a difference quotient over t ± 1e-4, which scales an
// ulp in a point by ~1e4; tangents and the Frenet frames built from them are
// held to 1e-12 × 1e4. Straight curves (two-point Catmull-Rom) are excluded
// from frame comparisons: their frames are `acos` of a dot product within an
// ulp of 1, where three.js's own result is rounding noise.
const TOL = 1e-12;
const TANGENT_TOL = 1e-8;
const CLOSED_TWIST_TOL = 3e-8;
/// The largest disagreement seen per kind, printed at the end.
const worst = {};
process.on('exit', () => { if (Object.keys(worst).length) console.log('# worst |ours - three.js|:', JSON.stringify(worst)); });

/// Mulberry32: the same random inputs on every run.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const flat = (v) => {
  if (v === null || v === undefined) return [v];
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return v.flatMap(flat);
  if (v.isVector3) return [v.x, v.y, v.z];
  if (v.isVector2) return [v.x, v.y];
  if (v.tangents) return [...flat(v.tangents), ...flat(v.normals), ...flat(v.binormals)];
  throw new Error(`flat: ${v}`);
};

function close(a, b, msg, tol = TOL, cat = 'points') {
  const x = flat(a), y = flat(b);
  assert.equal(x.length, y.length, `${msg}: length ${x.length} vs three.js ${y.length}`);
  for (let i = 0; i < y.length; i++) {
    if (Object.is(x[i], y[i])) continue;
    if (Number.isNaN(x[i]) && Number.isNaN(y[i])) continue;
    const d = Math.abs(x[i] - y[i]) / Math.max(1, Math.abs(y[i]));
    worst[cat] = Math.max(worst[cat] ?? 0, d);
    if (!(d <= tol)) {
      assert.fail(`${msg}[${i}]: ${x[i]} vs three.js ${y[i]}`);
    }
  }
  // Vector2 against Vector2, Vector3 against Vector3.
  const kind = (v) => (v?.isVector2 ? 2 : v?.isVector3 ? 3 : typeof v);
  const ka = Array.isArray(a) ? a.map(kind) : [kind(a)], kb = Array.isArray(b) ? b.map(kind) : [kind(b)];
  assert.deepEqual(ka, kb, `${msg}: vector types`);
}

/// Every public method of three.js's Curve on one curve, ours vs theirs.
function exercise(a, b, name, { frames = true } = {}) {
  const ts = [0, 1, 0.5, 0.25, 0.999, 1e-5, 0.123456789, 0.87654321];
  for (const t of ts) {
    close(a.getPoint(t), b.getPoint(t), `${name}.getPoint(${t})`);
    close(a.getPointAt(t), b.getPointAt(t), `${name}.getPointAt(${t})`);
    close(a.getTangent(t), b.getTangent(t), `${name}.getTangent(${t})`, TANGENT_TOL, 'tangents');
    close(a.getTangentAt(t), b.getTangentAt(t), `${name}.getTangentAt(${t})`, TANGENT_TOL, 'tangents');
    close(a.getUtoTmapping(t), b.getUtoTmapping(t), `${name}.getUtoTmapping(${t})`);
  }
  close(a.getUtoTmapping(0, a.getLength() * 0.3), b.getUtoTmapping(0, b.getLength() * 0.3), `${name}.getUtoTmapping(distance)`);
  // A target passed in is filled and returned.
  const target = a.getPoint(0).isVector2 ? new T.Vector2() : new T.Vector3();
  assert.equal(a.getPointAt(0.4, target), target, `${name}: optionalTarget`);
  close(a.getPoints(), b.getPoints(), `${name}.getPoints()`);
  close(a.getPoints(7), b.getPoints(7), `${name}.getPoints(7)`);
  close(a.getSpacedPoints(), b.getSpacedPoints(), `${name}.getSpacedPoints()`);
  close(a.getSpacedPoints(9), b.getSpacedPoints(9), `${name}.getSpacedPoints(9)`);
  close(a.getLength(), b.getLength(), `${name}.getLength()`);
  close(a.getLengths(), b.getLengths(), `${name}.getLengths()`);
  close(a.getLengths(17), b.getLengths(17), `${name}.getLengths(17)`);
  if (frames) {
    for (const closed of [false, true]) close(a.computeFrenetFrames(12, closed), b.computeFrenetFrames(12, closed), `${name}.computeFrenetFrames(12, ${closed})`, TANGENT_TOL, 'frames');
  }
  // Fewer arc-length divisions change every arc-length result.
  a.arcLengthDivisions = b.arcLengthDivisions = 13;
  a.updateArcLengths(); b.updateArcLengths();
  close(a.getPointAt(0.37), b.getPointAt(0.37), `${name}.getPointAt with arcLengthDivisions 13`);
  close(a.getLength(), b.getLength(), `${name}.getLength with arcLengthDivisions 13`);
}

const V2 = (L, r) => new L.Vector2(r() * 8 - 4, r() * 8 - 4);
const V3 = (L, r) => new L.Vector3(r() * 8 - 4, r() * 8 - 4, r() * 8 - 4);

/// [name, build(L, r)] for every built-in curve.
const CURVES = [
  ['LineCurve', (L, r) => new L.LineCurve(V2(L, r), V2(L, r))],
  ['QuadraticBezierCurve', (L, r) => new L.QuadraticBezierCurve(V2(L, r), V2(L, r), V2(L, r))],
  ['CubicBezierCurve', (L, r) => new L.CubicBezierCurve(V2(L, r), V2(L, r), V2(L, r), V2(L, r))],
  ['SplineCurve', (L, r) => new L.SplineCurve(Array.from({ length: 6 }, () => V2(L, r)))],
  ['EllipseCurve', (L, r) => new L.EllipseCurve(r(), r(), 1 + r(), 0.5 + r(), r() * 6, r() * 6, false, 0)],
  ['EllipseCurve cw rotated', (L, r) => new L.EllipseCurve(r(), r(), 1 + r(), 0.5 + r(), r() * 6, -r() * 6, true, r() * 3)],
  ['EllipseCurve full', (L) => new L.EllipseCurve(0, 0, 2, 1, 0, Math.PI * 2, true, 0.3)],
  ['ArcCurve', (L, r) => new L.ArcCurve(r(), r(), 1 + r(), r() * 3, 3 + r() * 3, true)],
  ['LineCurve3', (L, r) => new L.LineCurve3(V3(L, r), V3(L, r))],
  ['QuadraticBezierCurve3', (L, r) => new L.QuadraticBezierCurve3(V3(L, r), V3(L, r), V3(L, r))],
  ['CubicBezierCurve3', (L, r) => new L.CubicBezierCurve3(V3(L, r), V3(L, r), V3(L, r), V3(L, r))],
  ['CatmullRomCurve3 centripetal', (L, r) => new L.CatmullRomCurve3(Array.from({ length: 7 }, () => V3(L, r)))],
  ['CatmullRomCurve3 chordal closed', (L, r) => new L.CatmullRomCurve3(Array.from({ length: 5 }, () => V3(L, r)), true, 'chordal')],
  ['CatmullRomCurve3 catmullrom', (L, r) => new L.CatmullRomCurve3(Array.from({ length: 6 }, () => V3(L, r)), false, 'catmullrom', 0.3)],
  ['CatmullRomCurve3 catmullrom closed', (L, r) => new L.CatmullRomCurve3(Array.from({ length: 4 }, () => V3(L, r)), true, 'catmullrom', 0.8)],
  ['CatmullRomCurve3 two points', (L, r) => new L.CatmullRomCurve3([V3(L, r), V3(L, r)])],
  ['CatmullRomCurve3 repeated points', (L) => new L.CatmullRomCurve3([new L.Vector3(0, 0, 0), new L.Vector3(0, 0, 0), new L.Vector3(1, 2, 3), new L.Vector3(1, 2, 3), new L.Vector3(4, 0, 1)])],
];

/// A three.js-docs style custom curve, defined identically for both libraries.
function customSin(L) {
  return class CustomSinCurve extends L.Curve {
    constructor(scale = 1) { super(); this.scale = scale; }
    getPoint(t, optionalTarget = new L.Vector3()) {
      const tx = t * 3 - 1.5, ty = Math.sin(2 * Math.PI * t), tz = Math.cos(Math.PI * t) * 0.5;
      return optionalTarget.set(tx, ty, tz).multiplyScalar(this.scale);
    }
  };
}
function custom2D(L) {
  return class Heart extends L.Curve {
    getPoint(t, target = new L.Vector2()) {
      const a = t * Math.PI * 2;
      return target.set(16 * Math.sin(a) ** 3 / 10, (13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) / 10);
    }
  };
}
/// Overrides getTangent too, through `super`.
function customTangent(L) {
  const Sin = customSin(L);
  return class Flipped extends Sin {
    getTangent(t, target) { return super.getTangent(t, target).multiplyScalar(-1); }
  };
}
/// A subclass of a built-in that bends its getPoint.
function wavyLine(L) {
  return class WavyLine extends L.LineCurve3 {
    getPoint(t, target = new L.Vector3()) {
      super.getPoint(t, target);
      target.y += Math.sin(t * Math.PI * 4) * 0.2;
      return target;
    }
  };
}

/// f32 buffers: equal, or one f32 ulp apart where an f64 a libm ulp away
/// rounds the other way, or (closed tubes) within CLOSED_TWIST_TOL — counted, and
/// reported at the end.
const f32ulp = (v) => { const f = new Float32Array([Math.abs(v)]); const i = new Uint32Array(f.buffer); i[0]++; return f[0] - Math.abs(Math.fround(v)); };
const flips = { exact: 0, oneUlp: 0, near: 0 };
process.on('exit', () => console.log(`# f32 buffer values: ${flips.exact} identical, ${flips.oneUlp} one ulp apart, ${flips.near} within the closed-twist bound`));
function sameGeometry(ours, theirs, name) {
  for (const a of ['position', 'normal', 'uv']) {
    const x = ours.attributes[a]?.array, y = theirs.attributes[a].array;
    assert.ok(x, `${name}: no ${a}`);
    assert.equal(x.length, y.length, `${name}.${a} length`);
    for (let i = 0; i < y.length; i++) {
      if (Object.is(Math.fround(x[i]), y[i])) { flips.exact++; continue; }
      if (Math.abs(Math.fround(x[i]) - y[i]) <= f32ulp(y[i]) * 1.0000001) { flips.oneUlp++; continue; }
      // A closed tube's twist is `acos` of n0 · nN, which for a tube whose
      // ends already line up is within an ulp of 1: there acos turns one ulp
      // into sqrt(ulp) ≈ 1.5e-8, and three.js's own twist is that noise.
      if (Math.abs(x[i] - y[i]) <= CLOSED_TWIST_TOL) { flips.near++; continue; }
      assert.fail(`${name}.${a}[${i}]: ${x[i]} vs three.js ${y[i]}`);
    }
  }
  assert.deepEqual(ours.index ? Array.from(ours.index.array) : null, theirs.index ? Array.from(theirs.index.array) : null, `${name}: index`);
}

describe('shim curves over wasm vs three.js r165', { skip }, () => {
  for (const [name, build] of CURVES) {
    it(name, () => {
      for (let seed = 1; seed <= 4; seed++) {
        const a = build(T, rng(seed)), b = build(THREE, rng(seed));
        assert.equal(a.type, b.type);
        exercise(a, b, `${name} #${seed}`, { frames: !a.getPoint(0).isVector2 && !name.endsWith('two points') });
      }
    });
  }

  it('toJSON / fromJSON / clone / copy', () => {
    for (const [name, build] of CURVES) {
      const a = build(T, rng(7)), b = build(THREE, rng(7));
      a.arcLengthDivisions = b.arcLengthDivisions = 50;
      assert.deepEqual(a.toJSON(), b.toJSON(), `${name}.toJSON`);
      const back = new T[b.type]().fromJSON(b.toJSON());
      assert.deepEqual(back.toJSON(), b.toJSON(), `${name}.fromJSON`);
      exercise(back, b, `${name} fromJSON`, { frames: false });
      const c = a.clone();
      assert.ok(c instanceof T[b.type] && c !== a);
      assert.deepEqual(c.toJSON(), a.toJSON(), `${name}.clone`);
    }
  });

  it('arc lengths are cached until needsUpdate, as three.js caches them', () => {
    const a = new T.CatmullRomCurve3([new T.Vector3(0, 0, 0), new T.Vector3(1, 1, 0), new T.Vector3(2, 0, 1)]);
    const b = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 0), new THREE.Vector3(2, 0, 1)]);
    close(a.getLength(), b.getLength(), 'first');
    for (const c of [a, b]) c.points[2].set(5, 5, 5);
    // Stale: the table was not updated.
    close(a.getLength(), b.getLength(), 'stale');
    close(a.getPointAt(0.5), b.getPointAt(0.5), 'stale getPointAt');
    close(a.cacheArcLengths, b.cacheArcLengths, 'cacheArcLengths');
    a.needsUpdate = b.needsUpdate = true;
    close(a.getLength(), b.getLength(), 'after needsUpdate');
    for (const c of [a, b]) { c.points.push(c.points[0].clone().setY(-2)); c.updateArcLengths(); }
    close(a.getLength(), b.getLength(), 'after updateArcLengths');
    assert.equal(a.needsUpdate, b.needsUpdate);
  });

  const pathOf = (L, r) => {
    const p = new L.CurvePath();
    const a = V3(L, r), b = V3(L, r), c = V3(L, r), d = V3(L, r);
    p.add(new L.LineCurve3(a, b));
    p.add(new L.QuadraticBezierCurve3(b, V3(L, r), c));
    p.add(new L.CatmullRomCurve3([c, V3(L, r), V3(L, r), d]));
    p.add(new L.CubicBezierCurve3(d, V3(L, r), V3(L, r), V3(L, r)));
    return p;
  };
  const path2Of = (L, r) => {
    const p = new L.CurvePath();
    const a = V2(L, r), b = V2(L, r);
    p.add(new L.LineCurve(a, b));
    p.add(new L.EllipseCurve(b.x, b.y, 1, 0.5, 0, 2, false, 0.2));
    p.add(new L.SplineCurve([V2(L, r), V2(L, r), V2(L, r)]));
    p.add(new L.QuadraticBezierCurve(V2(L, r), V2(L, r), a));
    return p;
  };

  it('CurvePath (3D): walk, lengths, points, closePath, autoClose', () => {
    for (let seed = 1; seed <= 3; seed++) {
      const a = pathOf(T, rng(seed)), b = pathOf(THREE, rng(seed));
      close(a.getCurveLengths(), b.getCurveLengths(), 'getCurveLengths');
      exercise(a, b, `CurvePath #${seed}`);
      a.closePath(); b.closePath();
      assert.equal(a.curves.length, b.curves.length, 'closePath');
      assert.equal(a.curves.at(-1).type, b.curves.at(-1).type);
      a.updateArcLengths(); b.updateArcLengths();
      exercise(a, b, `closed CurvePath #${seed}`);
      a.autoClose = b.autoClose = true;
      close(a.getPoints(5), b.getPoints(5), 'autoClose getPoints');
      close(a.getSpacedPoints(6), b.getSpacedPoints(6), 'autoClose getSpacedPoints');
      assert.deepEqual(a.toJSON(), b.toJSON(), 'CurvePath.toJSON');
      const back = new T.CurvePath().fromJSON(b.toJSON());
      close(back.getPoints(4), b.getPoints(4), 'CurvePath.fromJSON');
      close(a.clone().getSpacedPoints(8), b.clone().getSpacedPoints(8), 'CurvePath.clone');
    }
  });

  it('CurvePath (2D): per-curve resolutions', () => {
    for (let seed = 1; seed <= 3; seed++) {
      const a = path2Of(T, rng(seed)), b = path2Of(THREE, rng(seed));
      exercise(a, b, `2D CurvePath #${seed}`, { frames: false });
      close(a.getPoints(3), b.getPoints(3), 'getPoints(3)');
    }
  });

  it('Path / Shape are CurvePaths with the base-class methods', () => {
    const build = (L) => {
      const s = new L.Shape();
      s.moveTo(0, 0); s.lineTo(2, 0); s.quadraticCurveTo(3, 0.5, 2.5, 1.5);
      s.absarc(1, 1.5, 1, 0, Math.PI, false);
      s.splineThru([new L.Vector2(-0.5, 1), new L.Vector2(-0.2, 0.4)]);
      s.closePath();
      return s;
    };
    const a = build(T), b = build(THREE);
    assert.ok(a instanceof T.CurvePath && a instanceof T.Curve);
    for (const u of [0, 0.3, 0.77, 1]) close(a.getUtoTmapping(u), b.getUtoTmapping(u), `Shape.getUtoTmapping(${u})`);
    assert.deepEqual(a.curves.map((c) => c.type), b.curves.map((c) => c.type));
    close(a.curves[1].getPoints(4), b.curves[1].getPoints(4), 'Path.curves are real curves');
    const ja = a.toJSON(), jb = b.toJSON();
    delete ja.uuid; delete jb.uuid;
    assert.deepEqual(ja, jb, 'Shape.toJSON');
  });

  it('user Curve subclasses run the base-class machinery', () => {
    for (const [name, make, frames] of [['CustomSinCurve', customSin, true], ['2D Heart', custom2D, false], ['getTangent override via super', customTangent, true], ['LineCurve3 subclass', wavyLine, true]]) {
      const A = make(T), B = make(THREE);
      const args = name === 'LineCurve3 subclass' ? [[new T.Vector3(0, 0, 0), new T.Vector3(3, 1, -1)], [new THREE.Vector3(0, 0, 0), new THREE.Vector3(3, 1, -1)]] : [[2], [2]];
      const a = new A(...args[0]), b = new B(...args[1]);
      assert.ok(a instanceof T.Curve);
      exercise(a, b, name, { frames });
    }
  });

  it('user curves inside a CurvePath', () => {
    const build = (L) => {
      const p = new L.CurvePath();
      const S = customSin(L);
      p.add(new S(1));
      p.add(new L.LineCurve3(new L.Vector3(1.5, 0, -0.5), new L.Vector3(3, 1, 0)));
      return p;
    };
    exercise(build(T), build(THREE), 'CurvePath of a user curve');
  });

  it('a throwing user getPoint surfaces its exception', () => {
    class Bad extends T.Curve { getPoint(t) { if (t > 0.5) throw new Error('nope'); return new T.Vector3(t, 0, 0); } }
    const c = new Bad();
    assert.throws(() => c.getLength(), /nope/);
    assert.throws(() => c.getLength(), /nope/);
  });

  it('TubeGeometry over built-in and user curves, frames included', () => {
    const cases = [
      ['user curve', (L) => new (customSin(L))(3), 40, 0.4, 8, false],
      ['user curve closed', (L) => new (customSin(L))(2), 20, 0.3, 5, true],
      ['CatmullRom closed', (L) => new L.CatmullRomCurve3([new L.Vector3(-2, 0, 0), new L.Vector3(0, 2, 1), new L.Vector3(2, 0, -1), new L.Vector3(0, -2, 0.5)], true), 64, 0.25, 12, true],
      ['default', null],
      ['CurvePath', (L) => pathOf(L, rng(3)), 30, 0.2, 6, false],
      ['subclassed line', (L) => new (wavyLine(L))(new L.Vector3(0, 0, 0), new L.Vector3(3, 1, -1)), 16, 0.1, 4, false],
    ];
    for (const [name, path, ...rest] of cases) {
      const a = path ? new T.TubeGeometry(path(T), ...rest) : new T.TubeGeometry();
      const b = path ? new THREE.TubeGeometry(path(THREE), ...rest) : new THREE.TubeGeometry();
      sameGeometry(a, b, `Tube ${name}`);
      const ftol = rest[3] ? CLOSED_TWIST_TOL : TANGENT_TOL;
      close(a.tangents, b.tangents, `Tube ${name} tangents`, ftol, 'frames');
      close(a.normals, b.normals, `Tube ${name} normals`, ftol, 'frames');
      close(a.binormals, b.binormals, `Tube ${name} binormals`, ftol, 'frames');
    }
  });

  it('ExtrudeGeometry along built-in and user curves', () => {
    const shape = (L) => new L.Shape([new L.Vector2(0, 0.3), new L.Vector2(-0.3, -0.3), new L.Vector2(0.3, -0.3)]);
    const paths = [
      ['user curve', (L) => new (customSin(L))(2)],
      ['CatmullRom', (L) => new L.CatmullRomCurve3([new L.Vector3(-2, 0, 0), new L.Vector3(-1, 1, 0.5), new L.Vector3(1, -0.5, 1), new L.Vector3(2, 0.5, -0.5)])],
      ['CubicBezier3', (L) => new L.CubicBezierCurve3(new L.Vector3(0, 0, 0), new L.Vector3(1, 2, 0), new L.Vector3(2, -1, 1), new L.Vector3(3, 0, 0))],
      ['CurvePath', (L) => pathOf(L, rng(5))],
    ];
    for (const [name, path] of paths) {
      for (const steps of [1, 9, 40]) {
        sameGeometry(new T.ExtrudeGeometry(shape(T), { steps, extrudePath: path(T) }),
          new THREE.ExtrudeGeometry(shape(THREE), { steps, extrudePath: path(THREE) }), `Extrude ${name} steps ${steps}`);
      }
    }
  });
});
