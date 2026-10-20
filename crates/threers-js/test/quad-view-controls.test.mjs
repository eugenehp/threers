// The shim's `QuadView.connect` (pointer events → the wasm `WebQuadView`
// interaction bindings) and `renderer.shadowMap.autoUpdate / needsUpdate`,
// tested without a wasm build: the classes are lifted out of the shim source
// and run against stubbed wasm handles and a stubbed DOM element, as in
// webgl-renderer.test.mjs. The logic behind the bindings is unit-tested in
// Rust (`helpers::quad_view_controls`) and in the browser suites.

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const shim = readFileSync(join(repo, 'web/threejs-shim.js'), 'utf8');

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
class WebQuadView {
  constructor() { this.zoom = 1; this.shareShadows = true; }
  pointerDown(cam, x, y, button, w, h) { calls.push(['pointerDown', x, y, button, w, h]); return x < 0 ? undefined : 'top'; }
  pointerMove(cam, x, y) { calls.push(['pointerMove', x, y]); return true; }
  pointerUp() { calls.push(['pointerUp']); }
  wheel(cam, x, y, delta, w, h) { calls.push(['wheel', x, y, delta, w, h]); return true; }
  dblclick(x, y, w, h) { calls.push(['dblclick', x, y, w, h]); return 'side'; }
  panBy(x, y, z) { calls.push(['panBy', x, y, z]); }
}
class WebColor { static fromHex(hex) { const c = new WebColor(); c.hex = hex; return c; } }
class WebRenderer {
  constructor() { this.shadowMapAutoUpdate = true; this.shadowMapNeedsUpdate = false; }
  static async new(canvas) { const r = new WebRenderer(); r.canvas = canvas; return r; }
  setToneMapping() {} setSize() {} setPixelRatio() {}
}
class WebVector2 {} class WebVector3 {} class WebVector4 {} class WebBox3 {}
class WebBufferAttribute {} class WebBufferGeometry {}
const _wasmReady = true;
const initThreers = async () => {};
const _optional = { WebQuadView };

const src = ['Vector2', 'Vector3', 'Vector4', 'Color', '_color', '_DEFAULT_SCENE_BACKGROUND',
  '_clearColorHex', 'WebGLRenderer', 'QuadView'].map(extract).join('\n');
const mod = new Function(
  'WebColor', 'WebRenderer', 'WebVector2', 'WebVector3', 'WebVector4', 'WebBox3',
  'WebBufferAttribute', 'WebBufferGeometry', '_wasmReady', 'initThreers', '_optional',
  `${src}\nreturn { QuadView, WebGLRenderer, Vector3 };`,
)(WebColor, WebRenderer, WebVector2, WebVector3, WebVector4, WebBox3, WebBufferAttribute,
  WebBufferGeometry, _wasmReady, initThreers, _optional);
const { QuadView, WebGLRenderer, Vector3 } = mod;

/// An element with listeners, a 200×100 CSS box at (10, 20), and a document.
function element() {
  const on = {}, docOn = {};
  const doc = {
    addEventListener: (t, f) => { docOn[t] = f; },
    removeEventListener: (t, f) => { if (docOn[t] === f) delete docOn[t]; },
  };
  return {
    on, docOn, style: {}, clientWidth: 200, clientHeight: 100, ownerDocument: doc,
    captured: null,
    addEventListener(t, f) { on[t] = f; },
    removeEventListener(t, f) { if (on[t] === f) delete on[t]; },
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }),
    setPointerCapture(id) { this.captured = id; },
    releasePointerCapture() { this.captured = null; },
  };
}

function camera() {
  return {
    _w: { readPosition: () => ({ x: 1, y: 2, z: 3 }), readTarget: () => ({ x: 4, y: 5, z: 6 }) },
    position: new Vector3(), _lookAt: new Vector3(), synced: 0,
    _sync() { this.synced++; },
    // The shim's cameras aim with `lookAt` (which records the target).
    lookAt(x, y, z) { this._lookAt.set(x, y, z); },
  };
}

const ev = (o) => ({ preventDefault() { this.prevented = true; }, pointerId: 7, button: 0, ...o });

describe('shim QuadView.connect', () => {
  beforeEach(() => { calls = []; });

  it('routes a drag to the wasm bindings in canvas coordinates and syncs the camera back', () => {
    const q = new QuadView();
    const el = element();
    const cam = camera();
    let changes = 0;
    q.onChange = () => changes++;
    q.connect(el, cam);
    assert.ok(el.on.pointerdown && el.on.wheel && el.on.dblclick && el.on.contextmenu);
    assert.equal(el.style.touchAction, 'none');

    const down = ev({ clientX: 60, clientY: 45 });
    el.on.pointerdown(down);
    // Top-left origin, relative to the element: (50, 25) on a 200×100 view.
    assert.deepEqual(calls[0], ['pointerDown', 50, 25, 0, 200, 100]);
    assert.equal(cam.synced, 1, 'the JS camera is pushed before the gesture');
    assert.ok(down.prevented);
    assert.equal(el.captured, 7);
    assert.ok(el.docOn.pointermove, 'moves are followed on the document');

    el.docOn.pointermove(ev({ clientX: 70, clientY: 40 }));
    assert.deepEqual(calls[1], ['pointerMove', 60, 20]);
    assert.deepEqual([cam.position.x, cam.position.y, cam.position.z], [1, 2, 3]);
    assert.deepEqual([cam._lookAt.x, cam._lookAt.y, cam._lookAt.z], [4, 5, 6]);
    assert.equal(changes, 1);
    // Another pointer's move is not this gesture's.
    el.docOn.pointermove(ev({ pointerId: 9, clientX: 0, clientY: 0 }));
    assert.equal(calls.length, 2);

    el.docOn.pointerup(ev({}));
    assert.deepEqual(calls[2], ['pointerUp']);
    assert.equal(el.docOn.pointermove, undefined);
  });

  it('scales CSS pixels to the layout size, and forwards wheel and double-click', () => {
    const q = new QuadView();
    const el = element();
    // The renderer lays the panes out at 400×200 while the element is 200×100.
    q._renderer = { getSize: (v) => { v.x = 400; v.y = 200; return v; } };
    q.connect(el, camera());
    const wheel = ev({ clientX: 110, clientY: 70, deltaY: 120 });
    el.on.wheel(wheel);
    assert.deepEqual(calls[0], ['wheel', 200, 100, 120, 400, 200]);
    assert.ok(wheel.prevented);
    el.on.dblclick(ev({ clientX: 10, clientY: 20 }));
    assert.deepEqual(calls[1], ['dblclick', 0, 0, 400, 200]);
  });

  it('a press over a gap starts nothing, and disconnect removes the listeners', () => {
    const q = new QuadView();
    const el = element();
    q.connect(el, camera());
    el.on.pointerdown(ev({ clientX: 0, clientY: 20 })); // x = −10: the stub's "gap"
    assert.equal(el.docOn.pointermove, undefined);
    q.disconnect();
    assert.deepEqual(Object.keys(el.on), []);
  });

  it('exposes zoom, panBy and shareShadows', () => {
    const q = new QuadView();
    q.zoom = 3;
    assert.equal(q._w.zoom, 3);
    q.panBy(1, 2, 3);
    assert.deepEqual(calls.at(-1), ['panBy', 1, 2, 3]);
    q.shareShadows = false;
    assert.equal(q.shareShadows, false);
  });
});

describe('shim renderer.shadowMap', () => {
  it('forwards autoUpdate and needsUpdate to wasm', async () => {
    const r = await WebGLRenderer.create({ width: 300, height: 150, style: {} });
    assert.equal(r.shadowMap.enabled, true);
    assert.equal(r.shadowMap.autoUpdate, true);
    r.shadowMap.autoUpdate = false;
    r.shadowMap.needsUpdate = true;
    assert.equal(r._w.shadowMapAutoUpdate, false);
    assert.equal(r._w.shadowMapNeedsUpdate, true);
    assert.equal(r.shadowMap.needsUpdate, true);
  });
});
