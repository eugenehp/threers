// The shim's `Color` and `ColorManagement` against three.js r165 itself
// (web/node_modules/three), on seeded random inputs: every setter, getter
// and operator, with the receiver and the result agreeing to 1e-12 —
// hex and CSS parsing (named colours, #rgb, #rrggbb, rgb()/rgba() in bytes
// and percent, hsl()/hsla(), whitespace, invalid strings), the colour-space
// arguments (sRGB, linear, Display P3 and its linear form), HSL round trips,
// lerpHSL / offsetHSL, buffer attributes, iteration and the NAMES table.
// Also: a material's `color` mutated in place after it is drawn reaches the
// wasm material at the next render, with no Color ever holding wasm state.
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
  const file = join(mkdtempSync(join(tmpdir(), 'threers-color-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threePath, 'build/three.module.js'));
const S = ready ? await loadShimOver(pkgDir) : null;
const T = ready ? await import(pathToFileURL(join(threePath, 'build/three.module.js')).href) : null;
const skip = !ready && `no wasm in ${pkgDir} or no three.js in ${threePath}`;

const TOL = 1e-12;
function close(a, b) {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (a === b) return true;
  return Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b));
}
function same(got, want, path) {
  if (typeof want === 'number') {
    if (typeof got !== 'number' || !close(got, want)) throw new Error(`${path}: ${got} !== ${want}`);
  } else if (want && typeof want === 'object') {
    if (want.isColor) {
      if (!got?.isColor) throw new Error(`${path}: not a Color`);
      for (const k of ['r', 'g', 'b']) same(got[k], want[k], `${path}.${k}`);
    } else {
      for (const k of Object.keys(want)) same(got[k], want[k], `${path}.${k}`);
    }
  } else if (got !== want) {
    throw new Error(`${path}: ${JSON.stringify(got)} !== ${JSON.stringify(want)}`);
  }
}

// mulberry32
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(0xc0102);
const f = (lo = -0.25, hi = 1.25) => lo + (hi - lo) * rand();
const int = (n) => Math.floor(rand() * n);
const hex = () => int(0x1000000);
const SPACES = ['srgb', 'srgb-linear', 'display-p3', 'display-p3-linear'];
const space = () => SPACES[int(SPACES.length)];

/// Warnings both libraries print for bad input, counted rather than shown.
function quiet(fn) {
  const warn = console.warn;
  const seen = [];
  console.warn = (...a) => seen.push(a.join(' '));
  try { return [fn(), seen]; } finally { console.warn = warn; }
}

/// Run `op` on a pair of colours built the same way in each library and
/// compare receiver, return value and warnings.
function check(label, init, op) {
  const a = init(S), b = init(T);
  const [ra, wa] = quiet(() => op(a, S));
  const [rb, wb] = quiet(() => op(b, T));
  same(a, b, `${label} receiver`);
  if (rb === b) assert.equal(ra, a, `${label} returns this`);
  else same(ra, rb, `${label} result`);
  assert.deepEqual(wa, wb, `${label} warnings`);
}
const rgbOf = (r, g, bl) => (L) => new L.Color(r, g, bl);

function cssStrings() {
  const out = [];
  const names = Object.keys(T.Color.NAMES);
  for (let i = 0; i < 60; i++) {
    const n = names[int(names.length)];
    out.push(n, n.toUpperCase(), n[0].toUpperCase() + n.slice(1));
  }
  const sp = () => [' ', '', '  ', '\t'][int(4)];
  const pct = () => `${int(130)}%`;
  for (let i = 0; i < 120; i++) {
    const h = hex().toString(16).padStart(6, '0');
    out.push(`#${h}`, `#${h.toUpperCase()}`, `#${h[0]}${h[2]}${h[4]}`);
    out.push(`rgb(${int(300)},${int(256)},${int(256)})`);
    out.push(`rgb(${sp()}${int(256)}${sp()},${sp()}${int(256)},${int(256)}${sp()})`);
    out.push(`rgba(${int(256)}, ${int(256)}, ${int(256)}, ${rand().toFixed(2)})`);
    out.push(`rgba(${int(256)},${int(256)},${int(256)},1)`);
    out.push(`rgb(${pct()},${pct()},${pct()})`);
    out.push(`rgba(${pct()}, ${pct()}, ${pct()}, .${int(10)})`);
    out.push(`hsl(${(rand() * 720).toFixed(int(3))},${(rand() * 100).toFixed(1)}%,${(rand() * 100).toFixed(int(2))}%)`);
    out.push(`hsla(${int(360)}, ${int(101)}%, ${int(101)}%, ${rand().toFixed(3)})`);
    out.push(`hsl(${sp()}.${int(10)}${sp()},${sp()}${int(100)}%${sp()},${int(100)}%)`);
  }
  out.push('', 'notacolour', '#12', '#12345', '#1234567', 'rgb(1,2)', 'rgb(1.5,2,3)', 'hsl(10,20,30)',
    'cmyk(1,2,3,4)', 'rgb( 255 , 0 , 0 )', 'RGB(1,2,3)', 'transparent', 'rebeccapurple');
  return out;
}

describe('Color matches three.js r165', { skip }, () => {
  it('constructs from hex, CSS, components, Colors and nothing', () => {
    for (let i = 0; i < 300; i++) {
      const h = hex();
      check('new Color(hex)', (L) => new L.Color(h), (c) => c);
      const [r, g, b] = [f(), f(), f()];
      check('new Color(r,g,b)', (L) => new L.Color(r, g, b), (c) => c);
    }
    check('new Color()', (L) => new L.Color(), (c) => c);
    check('new Color(Color)', (L) => new L.Color(new L.Color(0x123456)), (c) => c);
    for (const s of cssStrings()) check(`new Color(${JSON.stringify(s)})`, (L) => new L.Color(s), (c) => c);
    assert.equal(new S.Color().isColor, true);
    assert.deepEqual(Object.keys(S.Color.NAMES).sort(), Object.keys(T.Color.NAMES).sort());
    for (const k of Object.keys(T.Color.NAMES)) assert.equal(S.Color.NAMES[k], T.Color.NAMES[k], k);
  });

  it('setHex / getHex / getHexString / getStyle in every colour space', () => {
    for (let i = 0; i < 400; i++) {
      const h = hex() + (rand() < 0.2 ? rand() : 0);
      const cs = space(), out = space();
      const init = rgbOf(f(), f(), f());
      check('setHex', init, (c) => c.setHex(h));
      check('setHex(cs)', init, (c) => c.setHex(h, cs));
      check('getHex', init, (c) => c.getHex());
      check('getHex(cs)', init, (c) => c.getHex(out));
      check('getHexString(cs)', init, (c) => c.getHexString(out));
      check('getStyle', init, (c) => c.getStyle());
      check('getStyle(cs)', init, (c) => c.getStyle(out));
      check('toJSON', init, (c) => c.toJSON());
    }
  });

  it('setRGB / setScalar / set / getRGB with colour spaces', () => {
    for (let i = 0; i < 400; i++) {
      const [r, g, b] = [f(), f(), f()];
      const cs = space(), out = space();
      const init = rgbOf(f(), f(), f());
      check('setRGB', init, (c) => c.setRGB(r, g, b));
      check('setRGB(cs)', init, (c) => c.setRGB(r, g, b, cs));
      check('setScalar', init, (c) => c.setScalar(r));
      check('set(r,g,b)', init, (c) => c.set(r, g, b));
      const h2 = hex();
      check('set(hex)', init, (c) => c.set(h2));
      check('set(Color)', init, (c, L) => c.set(new L.Color(r, g, b)));
      check('getRGB', init, (c) => c.getRGB({ r: 0, g: 0, b: 0 }, out));
      check('getRGB default', init, (c) => c.getRGB({ r: 0, g: 0, b: 0 }));
    }
  });

  it('setHSL / getHSL round trips, lerpHSL, offsetHSL', () => {
    for (let i = 0; i < 400; i++) {
      const [h, s, l] = [f(-2, 3), f(-0.2, 1.2), f(-0.2, 1.2)];
      const cs = space(), out = space();
      const init = rgbOf(f(0, 1), f(0, 1), f(0, 1));
      check('setHSL', init, (c) => c.setHSL(h, s, l));
      check('setHSL(cs)', init, (c) => c.setHSL(h, s, l, cs));
      check('getHSL', init, (c) => c.getHSL({ h: 0, s: 0, l: 0 }));
      check('getHSL(cs)', init, (c) => c.getHSL({ h: 0, s: 0, l: 0 }, out));
      check('HSL round trip', init, (c) => {
        const t = c.getHSL({ h: 0, s: 0, l: 0 }, cs);
        return c.setHSL(t.h, t.s, t.l, cs).getHSL({ h: 0, s: 0, l: 0 }, cs);
      });
      const [dh, ds, dl] = [f(-1, 1), f(-0.5, 0.5), f(-0.5, 0.5)];
      check('offsetHSL', init, (c) => c.offsetHSL(dh, ds, dl));
      const [r2, g2, b2, a] = [f(0, 1), f(0, 1), f(0, 1), f(-0.5, 1.5)];
      check('lerpHSL', init, (c, L) => c.lerpHSL(new L.Color(r2, g2, b2), a));
    }
    // Greys, primaries and hue wrap-around, where the branch choices matter.
    for (const [r, g, b] of [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0.2, 0.2, 0.9]]) {
      check('getHSL exact', rgbOf(r, g, b), (c) => c.getHSL({ h: 0, s: 0, l: 0 }));
      check('lerpHSL exact', rgbOf(r, g, b), (c, L) => c.lerpHSL(new L.Color(1 - r, b, g), 0.3));
    }
  });

  it('setStyle and setColorName on CSS strings', () => {
    for (const s of cssStrings()) {
      const init = rgbOf(0.3, 0.6, 0.9);
      check(`setStyle(${JSON.stringify(s)})`, init, (c) => c.setStyle(s));
      check(`setStyle(${JSON.stringify(s)}, linear)`, init, (c) => c.setStyle(s, 'srgb-linear'));
      check(`setStyle(${JSON.stringify(s)}, p3)`, init, (c) => c.setStyle(s, 'display-p3'));
    }
    for (const n of Object.keys(T.Color.NAMES)) {
      const cs = space();
      check(`setColorName(${n})`, rgbOf(0, 0, 0), (c) => c.setColorName(n, cs));
    }
    check('setColorName(unknown)', rgbOf(0, 0, 0), (c) => c.setColorName('nope'));
  });

  it('setStyle goes further than three.js only where three.js gives up', () => {
    // A model three.js warns is unknown: threers parses it (CSS Color 4).
    const [c, warned] = quiet(() => new S.Color(0, 0, 0).setStyle('color(srgb 1 0 0)'));
    assert.deepEqual([c.r, c.g, c.b, warned.length], [1, 0, 0, 0]);
    // three.js's own models keep three.js's behaviour, misses included.
    const [, threeWarned] = quiet(() => new T.Color().setStyle('color(srgb 1 0 0)'));
    assert.equal(threeWarned.length, 1);
  });

  it('arithmetic, lerp, conversions, equality, arrays, attributes, iteration', () => {
    for (let i = 0; i < 400; i++) {
      const p = [f(), f(), f()], q = [f(), f(), f()], s = f(-2, 2);
      const init = rgbOf(...p);
      const other = (L) => new L.Color(...q);
      check('add', init, (c, L) => c.add(other(L)));
      check('addColors', init, (c, L) => c.addColors(other(L), new L.Color(...p)));
      check('addScalar', init, (c) => c.addScalar(s));
      check('sub', init, (c, L) => c.sub(other(L)));
      check('multiply', init, (c, L) => c.multiply(other(L)));
      check('multiplyScalar', init, (c) => c.multiplyScalar(s));
      check('lerp', init, (c, L) => c.lerp(other(L), s));
      check('lerpColors', init, (c, L) => c.lerpColors(other(L), new L.Color(...p).multiplyScalar(0.5), s));
      check('copy', init, (c, L) => c.copy(other(L)));
      check('clone', init, (c) => c.clone());
      check('convertSRGBToLinear', init, (c) => c.convertSRGBToLinear());
      check('convertLinearToSRGB', init, (c) => c.convertLinearToSRGB());
      check('copySRGBToLinear', init, (c, L) => c.copySRGBToLinear(other(L)));
      check('copyLinearToSRGB', init, (c, L) => c.copyLinearToSRGB(other(L)));
      check('setFromVector3', init, (c, L) => c.setFromVector3(new L.Vector3(...q)));
      const m = Array.from({ length: 9 }, () => f(-1, 1));
      check('applyMatrix3', init, (c, L) => c.applyMatrix3(new L.Matrix3().fromArray(m)));
      check('equals', init, (c, L) => c.equals(new L.Color(...p)));
      check('equals other', init, (c, L) => c.equals(other(L)));
      const arr = [f(), f(), f(), f(), f()];
      const off = int(3);
      check('fromArray', init, (c) => c.fromArray(arr, off));
      check('toArray', init, (c) => c.toArray([9, 9, 9, 9, 9], off));
      check('toArray()', init, (c) => c.toArray());
      check('iterator', init, (c) => [...c]);
      const data = new Float32Array(Array.from({ length: 12 }, () => f()));
      const idx = int(4);
      check('fromBufferAttribute', init, (c, L) => c.fromBufferAttribute(new L.BufferAttribute(data, 3), idx));
    }
  });

  it('ColorManagement converts between colour spaces as three.js does', () => {
    for (const k of ['enabled', 'workingColorSpace']) assert.equal(S.ColorManagement[k], T.ColorManagement[k], k);
    for (const cs of SPACES) {
      assert.equal(S.ColorManagement.getPrimaries(cs), T.ColorManagement.getPrimaries(cs), `primaries ${cs}`);
      assert.equal(S.ColorManagement.getTransfer(cs), T.ColorManagement.getTransfer(cs), `transfer ${cs}`);
    }
    assert.equal(S.ColorManagement.getTransfer(''), T.ColorManagement.getTransfer(''));
    for (let i = 0; i < 200; i++) {
      const [a, b] = [space(), space()];
      check(`convert ${a}->${b}`, rgbOf(f(), f(), f()), (c, L) => L.ColorManagement.convert(c, a, b));
      check(`fromWorking ${b}`, rgbOf(f(), f(), f()), (c, L) => L.ColorManagement.fromWorkingColorSpace(c, b));
      check(`toWorking ${a}`, rgbOf(f(), f(), f()), (c, L) => L.ColorManagement.toWorkingColorSpace(c, a));
    }
    assert.throws(() => { S.ColorManagement.workingColorSpace = 'srgb'; }, /Unsupported working color space/);
    // A Display P3 working space, then back.
    for (const L of [S, T]) L.ColorManagement.workingColorSpace = 'display-p3-linear';
    try {
      for (let i = 0; i < 100; i++) {
        const h = hex();
        check('setHex in P3 working space', rgbOf(0, 0, 0), (c) => c.setHex(h));
        check('getStyle in P3 working space', rgbOf(f(), f(), f()), (c) => c.getStyle());
      }
    } finally {
      for (const L of [S, T]) L.ColorManagement.workingColorSpace = 'srgb-linear';
    }
    // Management off: no conversion at all.
    for (const L of [S, T]) L.ColorManagement.enabled = false;
    try {
      for (let i = 0; i < 50; i++) {
        const h = hex();
        check('setHex unmanaged', rgbOf(0, 0, 0), (c) => c.setHex(h));
      }
    } finally {
      for (const L of [S, T]) L.ColorManagement.enabled = true;
    }
  });

  it('holds no wasm state', () => {
    const c = new S.Color(0x336699).setStyle('hsl(30, 50%, 50%)').lerp(new S.Color('red'), 0.5);
    assert.deepEqual(Object.keys(c).sort(), ['b', 'g', 'isColor', 'r']);
  });
});

describe('material colours reach wasm when mutated in place', { skip }, () => {
  it('_syncMaterialColors pushes exactly when a colour changes', () => {
    const m = new S.MeshStandardMaterial({ color: 0x808080, emissive: 0x000000 });
    const pushed = [];
    const real = m._w;
    m._w = new Proxy(real, {
      get(t, k) {
        if (k === 'setColor' || k === 'setEmissive') return (wc) => { pushed.push(k); return t[k](wc); };
        const v = t[k];
        return typeof v === 'function' ? v.bind(t) : v;
      },
    });
    const scene = new S.Scene();
    const mesh = new S.Mesh(new S.BoxGeometry(), m);
    scene.add(mesh);
    const prep = () => S.WebGLRenderer.prototype._prepareRender.call({ _inRender: false, _syncClearBackground() {} }, scene, new S.PerspectiveCamera());
    prep();
    assert.deepEqual(pushed, [], 'nothing changed, nothing pushed');
    m.color.setHex(0xff0000);
    prep();
    assert.deepEqual(pushed, ['setColor']);
    m.color.r = 0.25;
    prep();
    assert.deepEqual(pushed, ['setColor', 'setColor']);
    prep();
    assert.equal(pushed.length, 2, 'unchanged since the last push');
    m.emissive = new S.Color(0.1, 0.2, 0.3);
    prep();
    assert.deepEqual(pushed, ['setColor', 'setColor', 'setEmissive']);
    m.color = new S.Color(0.25, m.color.g, m.color.b);
    prep();
    assert.equal(pushed.at(-1), 'setColor', 'a new Color object is pushed even with equal values');
  });
});

describe('scene fog and background colours are re-read each render', { skip }, () => {
  it('fog.color / fog.near / background mutated in place are pushed once', () => {
    const scene = new S.Scene();
    const calls = [];
    const real = scene._w;
    scene._w = new Proxy(real, {
      get(t, k) {
        if (k === 'setFog') return (...a) => { calls.push(['fog', a[1], a[2], a[4]]); return t.setFog(...a); };
        const v = t[k];
        return typeof v === 'function' ? v.bind(t) : v;
      },
      set(t, k, v) { if (k === 'background') calls.push(['background']); t[k] = v; return true; },
    });
    scene._syncColors();
    assert.deepEqual(calls, [], 'no fog, no colour background: nothing to push');
    scene.fog = new S.Fog(0x336699, 2, 30);
    scene.background = new S.Color(0x101820);
    calls.length = 0;
    scene._syncColors();
    assert.deepEqual(calls, [], 'unchanged');
    scene.fog.color.setHex(0xff0000);
    scene._syncColors();
    scene.fog.near = 5;
    scene._syncColors();
    scene.background.lerp(new S.Color(1, 1, 1), 0.5);
    scene._syncColors();
    scene._syncColors();
    assert.deepEqual(calls, [['fog', 2, 30, 1], ['fog', 5, 30, 1], ['background']]);
    scene.fog = null;
    calls.length = 0;
    scene._syncColors();
    assert.deepEqual(calls, []);
  });
});
