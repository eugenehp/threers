// The shim's `WebGLRenderer` sizing / viewport / clear-colour surface, tested
// without a wasm build: the classes are lifted out of the shim source and run
// against stubbed wasm handles, as in svg-renderer.test.mjs. What the wasm side
// then does with the calls is covered by the browser suites (tests/webgl.rs,
// tests/webgpu.rs). The polygon-aware `WireframeGeometry` / `EdgesGeometry` are
// Rust now, so they run over a real wasm build (`THREERS_PKG_DIR`, else
// web/pkg), and are skipped when there is none.

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const shim = readFileSync(join(repo, 'web/threejs-shim.js'), 'utf8');

/// Source of a top-level `export class` / `class` / `function` / `const`,
/// with the `export` dropped.
function extract(name) {
  // `Color` is three.js r165's own code and leans on the ColorManagement
  // helpers declared beside it, so it comes as that whole section.
  if (name === 'Color') {
    const from = shim.indexOf('// ---- Color + ColorManagement');
    const to = shim.indexOf('// ---- Math value types');
    assert.ok(from !== -1 && to > from, 'the Color section is missing from the shim');
    return shim.slice(from, to).replace(/^export /gm, '');
  }
  const heads = [`export class ${name} `, `class ${name} `, `export function ${name}(`,
    `function ${name}(`, `const ${name} =`];
  let start = -1;
  for (const h of heads) {
    const i = shim.indexOf(`\n${h}`);
    if (i !== -1) { start = i + 1; break; }
  }
  assert.notEqual(start, -1, `${name} is missing from the shim`);
  if (shim.startsWith('const ', start)) {
    return shim.slice(start, shim.indexOf(';\n', start) + 1);
  }
  let depth = 0;
  for (let i = shim.indexOf('{', start); i < shim.length; i++) {
    if (shim[i] === '{') depth++;
    else if (shim[i] === '}' && --depth === 0) {
      return shim.slice(start, i + 1).replace(/^export /, '');
    }
  }
  throw new Error(`unterminated ${name}`);
}

// ---- stand-ins for the wasm handles ----
let calls = [];
const log = (name) => (...args) => { calls.push([name, ...args]); };
// The shim hands wasm linear floats (`_color` decodes a hex through the
// three.js `Color`); `hex` re-encodes them so the assertions read in sRGB.
const _enc = (c) => Math.round(255 * Math.min(1, Math.max(0,
  c < 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 0.41666) - 0.055)));
class WebColor {
  constructor(r, g, b) { this.r = r; this.g = g; this.b = b; }
  static fromHex(hex) { const c = new WebColor(0, 0, 0); c._hex = hex; return c; }
  get hex() { return this._hex ?? ((_enc(this.r) << 16) | (_enc(this.g) << 8) | _enc(this.b)); }
}
class WebRenderer {
  static async new(canvas) { const r = new WebRenderer(); r.canvas = canvas; return r; }
}
for (const m of ['setSize', 'setPixelRatio', 'setViewport', 'resetViewport', 'setScissor',
  'setScissorTest', 'setAutoClear', 'setMsaa', 'setToneMapping', 'render',
  'setSpecularAntialiasing', 'setReflectiveTransparency']) {
  WebRenderer.prototype[m] = log(m);
}
WebRenderer.prototype.clear = function (color, alpha) { calls.push(['clear', color.hex, alpha]); };
class WebVector2 {} class WebVector3 {} class WebVector4 {}
class WebBufferAttribute {} class WebBufferGeometry {}
const _wasmReady = true;
const initThreers = async () => {};
const _optional = {};

const MathUtils = { generateUUID: () => 'uuid' };
// `render` draws a bare Object3D through the scene mirror (three.js renders any
// root); the scenes here are plain stand-ins, so they take the Scene path.
class Object3D {}
const src = [
  'Vector2', 'Vector3', 'Vector4', 'Color', '_color', '_DEFAULT_SCENE_BACKGROUND',
  '_clearColorHex', 'WebGLRenderer', '_D_GEOMETRY', '_notifyUsers', '_DrawRange', '_attrPacked', 'BufferAttribute', 'BufferGeometry',
].map(extract).join('\n');
const mod = new Function(
  'WebColor', 'WebRenderer', 'WebVector2', 'WebVector3', 'WebVector4', 'WebBufferAttribute',
  'WebBufferGeometry', '_wasmReady', 'initThreers', '_optional', 'MathUtils', 'Object3D',
  `${src}\nreturn { WebGLRenderer, BufferGeometry, BufferAttribute, Color, Vector2, Vector4 };`,
)(WebColor, WebRenderer, WebVector2, WebVector3, WebVector4, WebBufferAttribute,
  WebBufferGeometry, _wasmReady, initThreers, _optional, MathUtils, Object3D);
const { WebGLRenderer, BufferGeometry, BufferAttribute, Color, Vector2, Vector4 } = mod;

const canvas = () => ({ width: 300, height: 150, style: {} });
const sawCalls = (name) => calls.filter((c) => c[0] === name);

describe('shim WebGLRenderer antialias (three.js parameters)', () => {
  it('is off by default, as in three.js', async () => {
    calls = [];
    await WebGLRenderer.create(canvas());
    assert.deepEqual(sawCalls('setMsaa'), [['setMsaa', 1]]);
  });
  it('antialias: true turns on 4x MSAA, given beside the canvas or with it', async () => {
    calls = [];
    await WebGLRenderer.create(canvas(), { antialias: true });
    const c = canvas();
    const r = await WebGLRenderer.create({ canvas: c, antialias: true });
    assert.deepEqual(sawCalls('setMsaa'), [['setMsaa', 4], ['setMsaa', 4]]);
    assert.equal(r.domElement, c);
  });
});

describe('shim WebGLRenderer threers extras', () => {
  it('specularAntialiasing and reflectiveTransparency start off and reach wasm', async () => {
    const r = await WebGLRenderer.create(canvas());
    calls = [];
    assert.equal(r.specularAntialiasing, false);
    assert.equal(r.reflectiveTransparency, false);
    r.specularAntialiasing = true;
    r.reflectiveTransparency = 1;
    assert.equal(r.specularAntialiasing, true);
    assert.equal(r.reflectiveTransparency, true);
    assert.deepEqual(sawCalls('setSpecularAntialiasing'), [['setSpecularAntialiasing', true]]);
    assert.deepEqual(sawCalls('setReflectiveTransparency'), [['setReflectiveTransparency', true]]);
  });
});

describe('shim WebGLRenderer sizing (three.js r165 semantics)', () => {
  let r, c;
  beforeEach(async () => {
    c = canvas();
    r = await WebGLRenderer.create(c);
    calls = [];
  });

  it('starts at the canvas size, pixel ratio 1', () => {
    assert.equal(r.getPixelRatio(), 1);
    assert.deepEqual([...Object.values(r.getSize(new Vector2()))], [300, 150]);
    assert.deepEqual(r.getViewport(new Vector4()), new Vector4(0, 0, 300, 150));
    assert.equal(r.getScissorTest(), false);
  });

  it('setSize takes CSS pixels and sizes the buffer by the pixel ratio', () => {
    r.setPixelRatio(2);
    calls = [];
    r.setSize(400, 300);
    assert.equal(c.width, 800);
    assert.equal(c.height, 600);
    assert.equal(c.style.width, '400px');
    assert.equal(c.style.height, '300px');
    assert.deepEqual(sawCalls('setSize'), [['setSize', 800, 600]]);
    const s = r.getSize(new Vector2());
    assert.deepEqual([s.x, s.y], [400, 300]);
    const d = r.getDrawingBufferSize(new Vector2());
    assert.deepEqual([d.x, d.y], [800, 600]);
  });

  it('setPixelRatio reaches wasm and resizes without touching CSS', () => {
    r.setSize(200, 100);
    c.style = {};
    calls = [];
    r.setPixelRatio(1.5);
    assert.deepEqual(sawCalls('setPixelRatio'), [['setPixelRatio', 1.5]]);
    assert.equal(c.width, 300);
    assert.equal(c.height, 150);
    assert.deepEqual(c.style, {}, 'updateStyle is false when the ratio changes');
    r.setPixelRatio(undefined);
    assert.equal(r.getPixelRatio(), 1.5, 'undefined is ignored, as in three.js');
  });

  // Pages that size the canvas in device pixels and never set a ratio must
  // see exactly what they saw when setPixelRatio was a no-op.
  it('without setPixelRatio, setSize still means device pixels', () => {
    r.setSize(640, 480, false);
    assert.equal(c.width, 640);
    assert.equal(c.height, 480);
    assert.deepEqual(c.style, {}, 'updateStyle false leaves CSS alone');
  });

  // An explicit viewport is what switches the WebGL2 backend to composing
  // through its back buffer; a resize must not look like one.
  it('a resize resets the viewport instead of setting one', () => {
    r.setSize(10, 10);
    assert.equal(sawCalls('setViewport').length, 0);
    assert.equal(sawCalls('resetViewport').length, 1);
    assert.deepEqual(r.getViewport(new Vector4()), new Vector4(0, 0, 10, 10));
  });

  it('records viewport, scissor and scissor test, Vector4 or numbers', () => {
    r.setViewport(new Vector4(1, 2, 3, 4));
    r.setScissor(5, 6, 7, 8);
    r.setScissorTest(true);
    assert.deepEqual(r.getViewport(new Vector4()), new Vector4(1, 2, 3, 4));
    assert.deepEqual(r.getScissor(new Vector4()), new Vector4(5, 6, 7, 8));
    assert.equal(r.getScissorTest(), true);
    assert.deepEqual(calls, [
      ['setViewport', 1, 2, 3, 4], ['setScissor', 5, 6, 7, 8], ['setScissorTest', true],
    ]);
  });
});

describe('shim WebGLRenderer clear colour', () => {
  let r;
  beforeEach(async () => {
    r = await WebGLRenderer.create(canvas());
    calls = [];
  });

  it('defaults to opaque black, as three.js', () => {
    assert.equal(r.getClearColor(new Color()).getHex(), 0x000000);
    assert.equal(r.getClearAlpha(), 1);
  });

  it('accepts hex, CSS strings and Colors', () => {
    r.setClearColor(0x336699, 0.5);
    assert.equal(r.getClearColor(new Color()).getHex(), 0x336699);
    assert.equal(r.getClearAlpha(), 0.5);
    r.setClearColor('#102030');
    assert.equal(r.getClearColor(new Color()).getHex(), 0x102030);
    assert.equal(r.getClearAlpha(), 1);
    r.setClearColor(new Color(0xabcdef));
    assert.equal(r.getClearColor(new Color()).getHex(), 0xabcdef);
    r.setClearAlpha(0.25);
    assert.equal(r.getClearAlpha(), 0.25);
  });

  // The hex is decoded from sRGB by the shim's `Color` — the same
  // path `scene.background = 0x…` takes, so the two agree on the canvas.
  it('clear() fills with the clear colour, by hex, and honours the colour flag', () => {
    r.setClearColor(0x204060, 0.75);
    r.clear();
    r.clear(false);
    r.clearColor();
    r.clearDepth();
    r.clearStencil();
    assert.deepEqual(sawCalls('clear'), [['clear', 0x204060, 0.75], ['clear', 0x204060, 0.75]]);
  });

  it('runs the canonical multi-view idiom as three.js does', () => {
    r.autoClear = false;
    r.setScissorTest(true);
    const scene = { _w: {}, _background: new Color(0x111111), _objects: [],
      _syncTransforms() {} };
    const camera = { _w: {}, _sync() {} };
    for (const [x, hex] of [[0, 0xff0000], [150, 0x00ff00]]) {
      r.setViewport(x, 0, 150, 150);
      r.setScissor(x, 0, 150, 150);
      r.setClearColor(hex);
      r.clear();
      r.render(scene, camera);
    }
    assert.deepEqual(calls.map((c) => c[0]), [
      'setAutoClear', 'setScissorTest',
      'setViewport', 'setScissor', 'clear', 'render',
      'setViewport', 'setScissor', 'clear', 'render',
    ]);
    assert.deepEqual(sawCalls('clear').map((c) => c[1]), [0xff0000, 0x00ff00]);
    assert.deepEqual(sawCalls('setAutoClear'), [['setAutoClear', false]]);
  });

  it('becomes the background of a null-background scene only once set', () => {
    const scene = { _w: { setBackgroundAlpha: log('setBackgroundAlpha') }, _background: null,
      _backgroundAlpha: 1, _objects: [], _syncTransforms() {} };
    const camera = { _w: {}, _sync() {} };
    r.render(scene, camera);
    assert.equal(scene._w.background, undefined, 'no clear colour set: scene untouched');
    r.setClearColor(0x445566, 0.5);
    r.render(scene, camera);
    assert.equal(scene._w.background.hex, 0x445566);
    assert.deepEqual(sawCalls('setBackgroundAlpha').at(-1), ['setBackgroundAlpha', 0.5]);
    // A scene with its own background keeps it.
    const own = { _w: {}, _background: new Color(0x010203), _objects: [], _syncTransforms() {} };
    r.render(own, camera);
    assert.equal(own._w.background, undefined);
    // The same scene drawn by a renderer that never set one gets the
    // wasm default back, not the other renderer's colour.
    return WebGLRenderer.create(canvas()).then((plain) => {
      plain.render(scene, camera);
      assert.equal(scene._w.background.hex, 0x111111);
    });
  });
});

/// A unit quad: two triangles over corners 0..3, split along 0–2.
function quad({ indexed = true, polygons = true } = {}) {
  const g = new BufferGeometry();
  const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
  if (indexed) {
    g.setAttribute('position', new BufferAttribute(new Float32Array(corners.flat()), 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
  } else {
    const tris = [0, 1, 2, 0, 2, 3].flatMap((i) => corners[i]);
    g.setAttribute('position', new BufferAttribute(new Float32Array(tris), 3));
  }
  if (polygons) g.setPolygons([2]);
  return g;
}
/// Segments of a line geometry as sorted "x,y,z-x,y,z" strings.
function segments(g) {
  const a = g.attributes.position.array;
  const out = [];
  for (let i = 0; i < a.length; i += 6) {
    const p = `${a[i]},${a[i + 1]},${a[i + 2]}`;
    const q = `${a[i + 3]},${a[i + 4]},${a[i + 5]}`;
    out.push(p < q ? `${p}-${q}` : `${q}-${p}`);
  }
  return out.sort();
}
const OUTLINE = ['0,0,0-0,1,0', '0,0,0-1,0,0', '0,1,0-1,1,0', '1,0,0-1,1,0'];

describe('shim BufferGeometry polygons', () => {
  it('validates the grouping against the triangles', () => {
    const g = quad({ polygons: false });
    assert.throws(() => g.setPolygons([3]), /sum to 3 triangles but the geometry draws 2/);
    assert.throws(() => g.setPolygons([2, 0]), /zero triangles/);
    g.setPolygons([1, 1]);
    assert.deepEqual([...g.polygons], [1, 1]);
    g.setPolygons(null);
    assert.equal(g.polygons, null);
  });

  it('drops it on a new index or a new vertex count, keeps it on a deformation', () => {
    const g = quad();
    g.translate(1, 0, 0);
    assert.deepEqual([...g.polygons], [2], 'same vertex count: same topology');
    g.setIndex([0, 1, 2, 0, 2, 3]);
    assert.equal(g.polygons, null);
    const h = quad();
    h.setAttribute('position', new BufferAttribute(new Float32Array(15), 3));
    assert.equal(h.polygons, null);
  });

  it('survives clone and toNonIndexed', () => {
    assert.deepEqual([...quad().clone().polygons], [2]);
    const flat = quad().toNonIndexed();
    assert.equal(flat.index, null);
    assert.deepEqual([...flat.polygons], [2]);
  });
});

// ---- WireframeGeometry / EdgesGeometry over the real wasm ----
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
let real = null;
if (existsSync(join(pkgDir, 'threers.js')) && existsSync(join(pkgDir, 'threers_bg.wasm'))) {
  const pkg = await import(pathToFileURL(join(pkgDir, 'threers.js')).href);
  pkg.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  if (typeof pkg.WebGeometry?.wireframe === 'function') real = pkg;
}
/// The shim's geometry classes over `pkg`'s real wasm handles.
function loadReal(pkg) {
  const names = ['Vector2', 'Vector3', '_D_GEOMETRY', '_notifyUsers', '_DrawRange', '_attrPacked', 'BufferAttribute', 'BufferGeometry', '_primitive', 'BoxGeometry',
    '_fillFromWasm', '_wasmSourceOf', '_edgesFromWasm', 'EdgesGeometry', 'WireframeGeometry'];
  return new Function('WebBufferAttribute', 'WebBufferGeometry', 'WebGeometry', 'WebVector2', 'WebVector3', 'MathUtils',
    `${names.map(extract).join('\n')}\nreturn { BufferGeometry, BufferAttribute, BoxGeometry, EdgesGeometry, WireframeGeometry };`,
  )(pkg.WebBufferAttribute, pkg.WebBufferGeometry, pkg.WebGeometry, pkg.WebVector2, pkg.WebVector3, MathUtils);
}

describe('shim WireframeGeometry / EdgesGeometry with polygons (real wasm)',
  { skip: !real && `no wasm build with WebGeometry.wireframe in ${pkgDir}` }, () => {
  const w = real && loadReal(real);
  /// `quad()` rebuilt with the real-wasm classes.
  const rquad = (o = {}) => {
    const q = quad(o);
    const g = new w.BufferGeometry();
    g.setAttribute('position', new w.BufferAttribute(new Float32Array(q.attributes.position.array), 3));
    if (q.index) g.setIndex(Array.from(q.index.array));
    if (q.polygons) g.setPolygons(q.polygons);
    return g;
  };

  it('WireframeGeometry outlines a polygon without its diagonal', () => {
    assert.deepEqual(segments(new w.WireframeGeometry(rquad())), OUTLINE);
    assert.deepEqual(segments(new w.WireframeGeometry(rquad({ indexed: false }))), OUTLINE,
      'unindexed: corners match by position');
  });

  it('WireframeGeometry without polygons is unchanged (three.js)', () => {
    const s = segments(new w.WireframeGeometry(rquad({ polygons: false })));
    assert.deepEqual(s, [...OUTLINE, '0,0,0-1,1,0'].sort());
  });

  it('EdgesGeometry never draws an edge inside a polygon, however bent', () => {
    // Fold the quad along its diagonal by lifting corner 3 well past 1°.
    const bend = (g) => {
      const a = g.attributes.position.array;
      for (let i = 0; i < a.length; i += 3) if (a[i] === 0 && a[i + 1] === 1) a[i + 2] = 1;
      return g;
    };
    const folded = segments(new w.EdgesGeometry(bend(rquad({ polygons: false })), 1));
    assert.ok(folded.includes('0,0,0-1,1,0'), `the fold is a crease without polygons: ${folded}`);
    const grouped = segments(new w.EdgesGeometry(bend(rquad()), 1));
    assert.ok(!grouped.includes('0,0,0-1,1,0'), `a polygon's own diagonal drew: ${grouped}`);
    assert.equal(grouped.length, 4, 'the border still draws');
  });

  it('a stale grouping is ignored rather than trusted', () => {
    const g = rquad();
    g._polygons = new Uint32Array([5]);
    const warn = console.warn;
    console.warn = () => {};
    try {
      const s = segments(new w.WireframeGeometry(g));
      assert.equal(s.length, 5, 'falls back to every triangle edge');
    } finally {
      console.warn = warn;
    }
  });

  it('a BoxGeometry reads as three.js builds it: 18 wireframe edges, 12 edges', () => {
    const box = new w.BoxGeometry(1, 1, 1);
    assert.equal(new w.WireframeGeometry(box).attributes.position.count, 36);
    assert.equal(new w.EdgesGeometry(box).attributes.position.count, 24);
  });
});
