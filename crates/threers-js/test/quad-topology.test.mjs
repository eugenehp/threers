// The shim's quad-topology extensions — `BufferGeometry.inferQuads` /
// `subdivide`, tested without a wasm build (the geometry classes are lifted
// out of the shim and run against stubbed wasm handles, as in
// webgl-renderer.test.mjs; the pairing itself is Rust,
// `BufferGeometry::infer_quads`, and tested there) — and `polygonIndex` on
// raycast hits and the polygon-aware exporters, over the shim module and a
// real wasm build (the Rust raycaster reports it).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const shim = readFileSync(join(repo, 'web/threejs-shim.js'), 'utf8');

/// Source of a top-level `export class` / `class` / `function` / `const`,
/// `export` dropped.
function extract(source, name) {
  const heads = [`export class ${name} `, `class ${name} `, `export function ${name}(`,
    `function ${name}(`, `const ${name} =`];
  let start = -1;
  for (const h of heads) {
    const i = source.indexOf(`\n${h}`);
    if (i !== -1) { start = i + 1; break; }
  }
  assert.notEqual(start, -1, `${name} is missing`);
  if (source.startsWith('const ', start)) return source.slice(start, source.indexOf(';\n', start) + 1);
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}' && --depth === 0) {
      return source.slice(start, i + 1).replace(/^export /, '');
    }
  }
  throw new Error(`unterminated ${name}`);
}

// ---- stand-ins for the wasm handles ----
class WebBufferAttribute { constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; } }
let wasmCalls = [];
class WebBufferGeometry {
  constructor() { this.attrs = {}; this.index = null; this.polygons = null; }
  setAttribute(name, a) { this.attrs[name] = a; }
  setIndex(i) { this.index = Uint32Array.from(i); this.polygons = null; }
  setPolygons(c) { this.polygons = Uint32Array.from(c); }
  // Pretend every consecutive triangle pair after a reversal was a quad.
  inferQuads(face, shape) {
    wasmCalls.push(['inferQuads', face, shape]);
    const tris = this.index.length / 3;
    const out = [];
    for (let t = tris - 1; t >= 0; t--) out.push(...this.index.slice(t * 3, t * 3 + 3));
    this.index = Uint32Array.from(out);
    this.polygons = new Uint32Array(tris / 2).fill(2);
    return tris / 2;
  }
  getIndexArray() { return this.index; }
  polygonCounts() { return this.polygons ?? undefined; }
  getAttributeArray(name) { return this.attrs[name]?.array; }
  getAttributeItemSize(name) { return this.attrs[name]?.itemSize; }
  // A canned "result": one quad, its two triangles and grouping.
  subdivide(levels, sharpBoundary) {
    wasmCalls.push(['subdivide', levels, sharpBoundary]);
    const out = new WebBufferGeometry();
    out.setAttribute('position', new WebBufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 3));
    out.setAttribute('normal', new WebBufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), 3));
    out.setIndex([0, 1, 2, 0, 2, 3]);
    out.setPolygons([2]);
    return out;
  }
  // `BufferGeometry::faces` for fans: a polygon of n triangles fanned from
  // its first corner is the loop a, b, c, d, …
  faces() {
    const out = [];
    let t = 0;
    for (const n of this.polygons ?? []) {
      const tri = (k) => this.index.slice((t + k) * 3, (t + k) * 3 + 3);
      out.push(n + 2, ...tri(0));
      for (let k = 1; k < n; k++) out.push(tri(k)[2]);
      t += n;
    }
    return Uint32Array.from(out);
  }
}
const MathUtils = { generateUUID: () => 'uuid' };
/// The shim's geometry classes over the stub wasm handles.
const shimSrc = ['_D_GEOMETRY', '_notifyUsers', '_DrawRange', '_attrPacked', 'BufferAttribute', 'BufferGeometry']
  .map((n) => extract(shim, n)).join('\n');
const loadStubbed = (wasm) => new Function('WebBufferAttribute', 'WebBufferGeometry', 'MathUtils',
  `${shimSrc}\nreturn { BufferAttribute, BufferGeometry };`)(wasm.WebBufferAttribute, wasm.WebBufferGeometry, MathUtils);
const { BufferAttribute, BufferGeometry } = loadStubbed({ WebBufferAttribute, WebBufferGeometry });

/// Three unit quads in a row along x, each two triangles fanned from its
/// first corner.
function strip({ grouped }, M = { BufferGeometry, BufferAttribute }) {
  const g = new M.BufferGeometry();
  const p = [];
  for (let y = 0; y <= 1; y++) for (let x = 0; x <= 3; x++) p.push(x, y, 0);
  g.setAttribute('position', new M.BufferAttribute(new Float32Array(p), 3));
  const idx = [];
  for (let x = 0; x < 3; x++) {
    const [a, b, c, d] = [x, x + 1, x + 5, x + 4];
    idx.push(a, b, c, a, c, d);
  }
  g.setIndex(new Uint32Array(idx));
  if (grouped) g.setPolygons([2, 2, 2]);
  return g;
}

describe('shim BufferGeometry.inferQuads', () => {
  it('sends the geometry and angles to wasm and takes back index + polygons', () => {
    wasmCalls = [];
    const g = strip({ grouped: false });
    const before = Array.from(g.index.array);
    const quads = g.inferQuads();
    assert.equal(quads, 3);
    assert.deepEqual(wasmCalls, [['inferQuads', 40 * Math.PI / 180, 40 * Math.PI / 180]]);
    assert.deepEqual(Array.from(g.polygons), [2, 2, 2]);
    // The stub reversed the triangle order; the JS index follows it.
    const after = Array.from(g.index.array);
    assert.deepEqual(after.slice(0, 3), before.slice(15, 18));
    g.inferQuads(0.1, 0.2);
    assert.deepEqual(wasmCalls[1], ['inferQuads', 0.1, 0.2]);
  });

  it('leaves the geometry alone when nothing pairs', () => {
    const g = strip({ grouped: false });
    const w = WebBufferGeometry.prototype.inferQuads;
    WebBufferGeometry.prototype.inferQuads = () => 0;
    try {
      const before = Array.from(g.index.array);
      assert.equal(g.inferQuads(), 0);
      assert.deepEqual(Array.from(g.index.array), before);
      assert.equal(g.polygons, null);
    } finally {
      WebBufferGeometry.prototype.inferQuads = w;
    }
  });
});

describe('shim BufferGeometry.subdivide', () => {
  it('sends levels and the boundary rule to wasm and returns a new geometry', () => {
    wasmCalls = [];
    const g = strip({ grouped: true });
    const before = Array.from(g.index.array);
    const out = g.subdivide();
    assert.deepEqual(wasmCalls, [['subdivide', 1, true]]);
    assert.ok(out instanceof BufferGeometry);
    assert.notEqual(out, g);
    assert.deepEqual(Array.from(out.index.array), [0, 1, 2, 0, 2, 3]);
    assert.deepEqual(Array.from(out.polygons), [2]);
    assert.equal(out.attributes.position.array.length, 12);
    assert.equal(out.attributes.normal.array.length, 12);
    // The source keeps its own triangles and grouping.
    assert.deepEqual(Array.from(g.index.array), before);
    assert.deepEqual(Array.from(g.polygons), [2, 2, 2]);
    g.subdivide(3, { sharpBoundary: false });
    assert.deepEqual(wasmCalls[1], ['subdivide', 3, false]);
  });

  it('throws on a wasm build without subdivision', () => {
    const w = WebBufferGeometry.prototype.subdivide;
    delete WebBufferGeometry.prototype.subdivide;
    try {
      assert.throws(() => strip({ grouped: false }).subdivide(), /no subdivision/);
    } finally {
      WebBufferGeometry.prototype.subdivide = w;
    }
  });
});

// OBJExporter / PLYExporter `{ polygons: true }` are Rust now (`WebSceneExporter`)
// and tested over the real wasm in exporters.test.mjs, and below.

// The shim module over a real wasm build (`THREERS_PKG_DIR`, else web/pkg), so
// the Catmull–Clark counts are the Rust ones and hits come from the Rust
// raycaster. Skipped when the build is missing or predates the scene-graph
// mirror.
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  if (typeof m.WebDragControls !== 'function' || typeof m.WebBufferGeometry?.prototype?.subdivide !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-quads-')), 'shim.mjs');
  writeFileSync(file, src);
  return { pkg: m, S: await import(pathToFileURL(file).href) };
}
const real = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const S = real?.S;
const skipReal = !real && `${pkgDir} missing, or predates subdivide / the scene-graph mirror`;

describe('shim Raycaster: polygonIndex', { skip: skipReal }, () => {
  const cast = (x, y, geometry) => new S.Raycaster(new S.Vector3(x, y, 1), new S.Vector3(0, 0, -1))
    .intersectObject(new S.Mesh(geometry, new S.MeshBasicMaterial()));
  it('reports the quad, for either of its triangles', () => {
    for (const [x, y, quad] of [[0.7, 0.2, 0], [0.2, 0.7, 0], [1.7, 0.2, 1], [1.2, 0.9, 1], [2.5, 0.4, 2]]) {
      const hits = cast(x, y, strip({ grouped: true }, S));
      assert.equal(hits.length, 1, `${x},${y}`);
      assert.equal(hits[0].polygonIndex, quad, `${x},${y}`);
      assert.equal(Math.floor(hits[0].faceIndex / 2), quad);
    }
  });
  it('is null without a grouping', () => {
    const hits = cast(1.7, 0.2, strip({ grouped: false }, S));
    assert.equal(hits.length, 1);
    assert.equal(hits[0].polygonIndex, null);
    assert.equal(hits[0].faceIndex, 2);
  });
});

describe('BufferGeometry.subdivide over the real wasm', { skip: skipReal }, () => {
  /// The shim's cube: the box generator's arrays as a JS BufferGeometry, its
  /// six quads declared.
  function cube() {
    const box = real.pkg.WebGeometry.box(2, 2, 2).toBufferGeometry();
    const g = new S.BufferGeometry();
    g.setAttribute('position', new S.BufferAttribute(new Float32Array(box.getAttributeArray('position')), 3));
    g.setIndex(new Uint32Array(box.getIndexArray()));
    g.setPolygons(box.polygonCounts());
    return g;
  }
  it('turns the 6-quad cube into 24 quads, then 96, all grouped', () => {
    const g = cube();
    assert.deepEqual(Array.from(g.polygons), [2, 2, 2, 2, 2, 2]);
    const one = g.subdivide(1);
    assert.equal(one.polygons.length, 24);
    assert.ok(one.polygons.every((n) => n === 2));
    assert.equal(one.index.array.length, 24 * 6);
    assert.equal(one.attributes.normal.array.length, one.attributes.position.array.length);
    const two = g.subdivide(2);
    assert.equal(two.polygons.length, 96);
    assert.ok(two.polygons.every((n) => n === 2));
    // Subdividing the subdivided keeps going from its quads.
    assert.equal(one.subdivide(1).polygons.length, 96);
    // The source is untouched.
    assert.equal(g.polygons.length, 6);
    assert.throws(() => g.subdivide(7), /levels must be 0..=6/);
  });
  it('without a grouping the cage is the 12 triangles: 36 quads', () => {
    const g = cube();
    g.setPolygons(null);
    assert.equal(g.subdivide(1).polygons.length, 36);
  });
  it('exports the subdivided quads as quad faces', () => {
    const one = cube().subdivide(1);
    const faces = new S.OBJExporter().parse(new S.Mesh(one), { polygons: true })
      .split('\n').filter((l) => l.startsWith('f '));
    assert.equal(faces.length, 24);
    assert.ok(faces.every((l) => l.split(' ').length === 5), faces.join('\n'));
    const ply = new S.PLYExporter().parse(new S.Mesh(one), null, { polygons: true });
    assert.match(ply, /element face 24\n/);
  });
});
