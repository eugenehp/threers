// The shim's `SVGRenderer`, tested without a wasm build.
//
// Everything here is the JavaScript half — `domElement`, `render`, `setSize`,
// the option forwarding — so it runs against a stubbed handle and a small DOM
// rather than waiting on `wasm-pack`. The Rust half has its own tests.
//
// The class is lifted out of the shim source instead of imported because
// importing the shim pulls in the wasm module, which is the thing being avoided.

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const shimSource = readFileSync(join(repo, 'web/threejs-shim.js'), 'utf8');

function extractClass(source, name) {
  const start = source.indexOf(`export class ${name}`);
  assert.notEqual(start, -1, `${name} is missing from the shim`);
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}' && --depth === 0) {
      return source.slice(start, i + 1).replace(`export class ${name}`, 'class');
    }
  }
  throw new Error(`unterminated class ${name}`);
}

// --- the smallest DOM these methods need ---
class El {
  constructor(nodeName) {
    this.nodeName = nodeName;
    this.attrs = {};
    this.childNodes = [];
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  get attributes() {
    return Object.entries(this.attrs).map(([name, value]) => ({ name, value }));
  }
  appendChild(child) {
    if (child.parent) child.parent.childNodes.splice(child.parent.childNodes.indexOf(child), 1);
    child.parent = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    this.childNodes.splice(this.childNodes.indexOf(child), 1);
    child.parent = null;
    return child;
  }
  get firstChild() { return this.childNodes[0] ?? null; }
}

globalThis.document = { createElementNS: (_ns, nodeName) => new El(nodeName) };
globalThis.DOMParser = class {
  parseFromString(markup) {
    const root = new El('svg');
    for (const [, k, v] of markup.matchAll(/(\w[\w-]*)="([^"]*)"/g)) {
      if (!['d', 'fill', 'stroke'].includes(k)) root.setAttribute(k, v);
    }
    for (const m of markup.matchAll(/<(path|rect|line|circle)\b/g)) root.appendChild(new El(m[1]));
    return { documentElement: root };
  }
};

let calls = [];
globalThis.WebSvgRenderer = class {
  constructor(w, h) { this.w = w; this.h = h; }
  setSize(w, h) { calls.push(['setSize', w, h]); this.w = w; this.h = h; }
  setPrecision(p) { calls.push(['setPrecision', p]); }
  setCurveTolerance(p) { calls.push(['setCurveTolerance', p]); }
  setSeamStroke(p) { calls.push(['setSeamStroke', p]); }
  setClearColor(hex, a) { calls.push(['setClearColor', hex, a]); }
  setShading(m) { calls.push(['setShading', m]); }
  setToneMapping(m, e) { calls.push(['setToneMapping', m, e]); }
  setBackground(b) { calls.push(['setBackground', b]); }
  setCullBackfaces(b) { calls.push(['setCullBackfaces', b]); }
  setDepthSplit(t) { calls.push(['setDepthSplit', t]); }
  setSort(b) { calls.push(['setSort', b]); }
  renderToString() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.w}" height="${this.h}" `
      + `viewBox="0 0 ${this.w} ${this.h}"><rect/><path d="M0 0"/><path d="M1 1"/></svg>`;
  }
};

const SVGRenderer = eval(`(${extractClass(shimSource, 'SVGRenderer')})`);
const sawCall = (name) => calls.find((c) => c[0] === name);

describe('shim SVGRenderer', () => {
  let r;
  beforeEach(() => {
    calls = [];
    r = new SVGRenderer();
  });

  // Without these two a page cannot turn the renderer on at all: three.js
  // enables a renderer by appending its `domElement` and calling `render`.
  it('owns an <svg> element sized to the canvas', () => {
    assert.equal(r.domElement.nodeName, 'svg');
    assert.equal(r.domElement.getAttribute('viewBox'), '0 0 800 600');
  });

  // The renderer draws whatever pose was last pushed into wasm. Skip the sync
  // and moving a mesh or the camera from JavaScript changes nothing, which
  // looks like a frozen picture rather than like a missing call.
  it('pushes the scene and camera into wasm before drawing', () => {
    const synced = [];
    const scene = { _w: {}, _syncTransforms: (cam) => synced.push(['scene', cam]) };
    const camera = { _w: {}, _sync: () => synced.push(['camera']) };
    r.render(scene, camera);
    assert.equal(synced.length, 2, 'both the scene and the camera must be synced');
    assert.equal(synced[0][0], 'scene');
    assert.equal(synced[0][1], camera, 'the scene sync needs the camera for layers');
    assert.equal(synced[1][0], 'camera');
  });

  it('leaves an orbit-controlled camera alone', () => {
    const synced = [];
    const scene = { _w: {}, _syncTransforms: () => synced.push('scene') };
    const camera = { _w: {}, _orbitControlled: true, _sync: () => synced.push('camera') };
    r.render(scene, camera);
    assert.deepEqual(synced, ['scene'], 'controls own the wasm camera pose');
  });

  it('draws into that element and reports what it drew', () => {
    const markup = r.render({ _w: {} }, { _w: {} });
    assert.match(markup, /^<svg/);
    assert.equal(r.domElement.childNodes.length, 3);
    assert.equal(r.domElement.getAttribute('width'), '800');
    assert.equal(r.info.render.faces, 2);
  });

  // Rebuilding the handle on resize would silently drop every option set
  // before it, which surfaces much later as "my SVG lost its background".
  it('resizes in place rather than rebuilding', () => {
    r.setSize(400, 300);
    assert.deepEqual(sawCall('setSize'), ['setSize', 400, 300]);
    assert.equal(r.domElement.getAttribute('viewBox'), '0 0 400 300');
  });

  it('clears between frames unless told not to', () => {
    r.render({ _w: {} }, { _w: {} });
    r.render({ _w: {} }, { _w: {} });
    assert.equal(r.domElement.childNodes.length, 3, 'autoClear should replace, not append');

    r.autoClear = false;
    r.render({ _w: {} }, { _w: {} });
    assert.equal(r.domElement.childNodes.length, 6);

    r.clear();
    assert.equal(r.domElement.childNodes.length, 0);
  });

  // `Scene.background` is stored unconverted for WebGL's sake, and this
  // renderer encodes to sRGB on the way out — so handing those numbers over
  // raw renders #101820 as a much lighter #475663.
  it('renders scene.background as the colour it was authored as', () => {
    const scene = { _w: {}, _background: { getHex: () => 0x101820 }, _backgroundAlpha: 1 };
    r.render(scene, { _w: {} });
    assert.deepEqual(sawCall('setClearColor'), ['setClearColor', 0x101820, 1]);
  });

  it('lets an explicit clear colour win over the scene', () => {
    r.setClearColor(0xff0000, 1);
    calls = [];
    r.render({ _w: {}, _background: { getHex: () => 0x101820 } }, { _w: {} });
    assert.equal(sawCall('setClearColor'), undefined, 'scene must not override an explicit colour');
  });

  it('takes a clear colour the three ways three.js does', () => {
    r.setClearColor(0x112233, 0.5);
    assert.deepEqual(sawCall('setClearColor'), ['setClearColor', 0x112233, 0.5]);

    calls = [];
    r.setClearColor({ getHex: () => 0x445566 });
    assert.equal(sawCall('setClearColor')[1], 0x445566);

    calls = [];
    r.setClearColor('#778899');
    assert.equal(sawCall('setClearColor')[1], 0x778899);
  });

  it('maps setQuality onto what actually costs bytes', () => {
    r.setQuality('low');
    assert.equal(sawCall('setCurveTolerance')[1], 0);
    assert.equal(sawCall('setSeamStroke')[1], 0);

    calls = [];
    r.setQuality('high');
    assert.ok(sawCall('setCurveTolerance')[1] > 0);
    assert.ok(sawCall('setSeamStroke')[1] > 0);
  });

  it('forwards the threers-only options', () => {
    r.setShading(2);
    r.setToneMapping(4, 1.2);
    r.setBackground(false);
    r.setCullBackfaces(false);
    r.setCurveTolerance(0.3);
    r.setDepthSplit(0.02);
    r.setSort(false);
    assert.deepEqual(sawCall('setShading'), ['setShading', 2]);
    assert.deepEqual(sawCall('setToneMapping'), ['setToneMapping', 4, 1.2]);
    assert.deepEqual(sawCall('setBackground'), ['setBackground', false]);
    assert.deepEqual(sawCall('setCullBackfaces'), ['setCullBackfaces', false]);
    assert.deepEqual(sawCall('setCurveTolerance'), ['setCurveTolerance', 0.3]);
    assert.deepEqual(sawCall('setDepthSplit'), ['setDepthSplit', 0.02]);
    assert.deepEqual(sawCall('setSort'), ['setSort', false]);
  });

  it('restores the default precision on null, and tolerates setPixelRatio', () => {
    r.setPrecision(null);
    assert.equal(sawCall('setPrecision')[1], 2);
    r.setPixelRatio(2); // an SVG has no pixels; must not throw
  });

  // A shim method calling a binding that does not exist fails only at runtime,
  // in a browser, on the one line that uses it.
  it('only calls methods the wasm binding actually exposes', () => {
    const cls = extractClass(shimSource, 'SVGRenderer');
    const used = [...cls.matchAll(/this\._w\.(\w+)\(/g)].map((m) => m[1]);
    const rust = readFileSync(join(repo, 'src/wasm.rs'), 'utf8');
    const start = rust.indexOf('pub struct WebSvgRenderer');
    const block = rust.slice(start, rust.indexOf('/// Wrap an RGBA frame', start));
    const exposed = new Set([...block.matchAll(/js_name = (\w+)/g)].map((m) => m[1]));
    for (const name of new Set(used)) {
      assert.ok(exposed.has(name), `shim calls _w.${name}(), which WebSvgRenderer does not expose`);
    }
  });
});
