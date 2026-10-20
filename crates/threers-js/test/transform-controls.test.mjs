// The shim's TransformControls, FlyControls and MapControls against three.js
// r165's own (web/node_modules/three/examples/jsm/controls), driven by the
// same DOM pointer and key events through a small DOM stub.
//
// TransformControls is three.js's math in Rust (`controls/transform`), in
// f64 and in three.js's order of operations (its sin / cos / atan2 from
// JavaScript's Math), so the attached object's position, quaternion and
// scale after a drag must be three.js's bit for bit — in every
// mode, both spaces, on every axis and plane handle, with snapping, under a
// rotated and scaled parent, through a perspective and an orthographic
// camera. The picker cast is the shim's (Rust, f32) raycaster, so drags start
// from pointers three.js picks the handle at with margin to spare. The event
// sequence (`axis-changed`, `dragging-changed`, `rotationAngle-changed`,
// `change`, `mouseDown`, `objectChange`, `mouseUp`) must match too, and the
// gizmo's handle geometry must be three.js's, buffer for buffer.
//
// FlyControls is f64 Rust too (to 1e-12). MapControls runs on the Rust orbit
// controls, which keep their pose in f32: compared to 1e-4.
//
// The wasm comes from `THREERS_PKG_DIR` (a wasm-bindgen `--target web` output
// directory) or web/pkg; the suite skips when that build predates the bindings.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');
const threeDir = join(repo, 'web/node_modules/three');

async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const pkg = await import(pkgUrl);
  if (typeof pkg.WebTransformControls !== 'function') return null;
  pkg.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-tc-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const ready = existsSync(join(pkgDir, 'threers_bg.wasm')) && existsSync(join(threeDir, 'build/three.module.js'));
const S = ready ? await loadShimOver(pkgDir) : null;
const T = ready ? await import(pathToFileURL(join(threeDir, 'build/three.module.js')).href) : null;
const jsm = (p) => import(pathToFileURL(join(threeDir, 'examples/jsm/controls', p)).href);
const T_TC = S ? (await jsm('TransformControls.js')).TransformControls : null;
const T_FLY = S ? (await jsm('FlyControls.js')).FlyControls : null;
const T_MAP = S ? (await jsm('MapControls.js')).MapControls : null;
const skip = !ready ? 'three@0.165 or a wasm build is missing'
  : !S ? `the wasm build in ${pkgDir} predates the TransformControls bindings` : false;

// ---- DOM stub ----

function target() {
  const listeners = new Map();
  return {
    addEventListener(type, fn) { (listeners.get(type) ?? listeners.set(type, []).get(type)).push(fn); },
    removeEventListener(type, fn) {
      const l = listeners.get(type);
      if (l) listeners.set(type, l.filter((f) => f !== fn));
    },
    _fire(type, event) { for (const fn of (listeners.get(type) ?? []).slice()) fn(event); },
  };
}
if (S) {
  globalThis.document ??= Object.assign(target(), { pointerLockElement: null });
  globalThis.window ??= Object.assign(target(), { innerWidth: 800, innerHeight: 600 });
}
function element() {
  const doc = globalThis.document;
  const el = Object.assign(target(), {
    style: {},
    clientWidth: 800, clientHeight: 600, offsetWidth: 800, offsetHeight: 600, offsetLeft: 0, offsetTop: 0,
    ownerDocument: doc,
    getRootNode: () => doc,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600 }),
    setPointerCapture() {}, releasePointerCapture() {}, focus() {},
  });
  return el;
}
/// A pointer event at NDC `(x, y)` on the 800×600 element (also seen by the
/// document, where the shim's orbit controls listen during a drag).
function fire(el, type, x, y, button, extra = {}) {
  const clientX = (x + 1) / 2 * 800, clientY = (1 - y) / 2 * 600;
  const e = { type, clientX, clientY, pageX: clientX, pageY: clientY, button, pointerId: 1,
    pointerType: 'mouse', ctrlKey: false, metaKey: false, shiftKey: false, altKey: false,
    preventDefault() {}, stopPropagation() {}, ...extra };
  el._fire(type, e);
  if (el.ownerDocument && el.ownerDocument !== el) el.ownerDocument._fire(type, e);
}

// ---- scenes ----

/// The same scene in either library: a rotated, scaled parent holding the
/// object, which has its own transform.
function build(L, { ortho = false } = {}) {
  const scene = new L.Scene();
  const camera = ortho
    ? new L.OrthographicCamera(-4, 4, 3, -3, 0.1, 100)
    : new L.PerspectiveCamera(50, 800 / 600, 0.1, 100);
  camera.position.set(3.2, 4.1, 5.3);
  camera.lookAt(0.2, 0.1, -0.1);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  const parent = new L.Group();
  parent.position.set(0.3, -0.2, 0.1);
  parent.rotation.set(0.2, -0.35, 0.15);
  parent.scale.set(1.2, 0.9, 1.1);
  scene.add(parent);
  const object = new L.Mesh(new L.BoxGeometry(0.5, 0.5, 0.5), new L.MeshBasicMaterial());
  object.position.set(0.25, 0.4, -0.3);
  object.rotation.set(-0.4, 0.6, 0.25);
  object.scale.set(0.8, 1.3, 1.05);
  parent.add(object);
  return { scene, camera, parent, object };
}

function setup(L, TC, opts = {}) {
  const w = build(L, opts);
  const el = element();
  const tc = new TC(w.camera, el);
  tc.attach(w.object);
  w.scene.add(tc);
  if (opts.mode) tc.setMode(opts.mode);
  if (opts.space) tc.setSpace(opts.space);
  if (opts.size) tc.setSize(opts.size);
  if (opts.translationSnap) tc.setTranslationSnap(opts.translationSnap);
  if (opts.rotationSnap) tc.setRotationSnap(opts.rotationSnap);
  if (opts.scaleSnap) tc.setScaleSnap(opts.scaleSnap);
  if (opts.showX === false) tc.showX = false;
  w.scene.updateMatrixWorld();
  const events = [];
  for (const type of ['change', 'objectChange', 'mouseDown', 'mouseUp', 'axis-changed', 'dragging-changed',
    'rotationAngle-changed']) {
    tc.addEventListener(type, (e) => events.push(e.value !== undefined ? `${type}:${e.value}` : e.mode ? `${type}:${e.mode}` : type));
  }
  return { ...w, el, tc, events };
}

/// A pointer three.js picks `axis` at, with the same pick a little way off in
/// each direction (so an f32 cast agrees).
function findPointer(three, axis) {
  const { tc } = three;
  const hit = (x, y) => { tc.pointerHover({ x, y, button: -1 }); return tc.axis === axis; };
  const found = [];
  for (let y = -0.99; y <= 0.99; y += 0.01) {
    for (let x = -0.99; x <= 0.99; x += 0.01) {
      if (!hit(x, y)) continue;
      const d = 0.006;
      if (hit(x + d, y) && hit(x - d, y) && hit(x, y + d) && hit(x, y - d)) found.push([x, y]);
    }
  }
  tc.pointerHover({ x: 5, y: 5, button: -1 });
  assert.ok(found.length, `three.js picks ${axis} nowhere`);
  return found[Math.floor(found.length / 2)];
}

const near = (a, b, tol, what) => {
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) <= tol, `${what}[${i}]: ${a[i]} vs ${b[i]} (|Δ| ${Math.abs(a[i] - b[i])})`);
  }
};
const pose = (o) => [o.position.x, o.position.y, o.position.z, o.quaternion.x, o.quaternion.y,
  o.quaternion.z, o.quaternion.w, o.scale.x, o.scale.y, o.scale.z];

/// Hover, press, drag through `moves` (NDC offsets), release — the same events
/// to both, the scene updated between them as a render would.
function drag(opts, axis, moves) {
  const a = setup(T, T_TC, opts);
  const b = setup(S, S.TransformControls, opts);
  const [x, y] = findPointer(a, axis);
  a.events.length = 0;
  const all = [a, b];
  for (const w of all) { fire(w.el, 'pointermove', x, y, -1); w.scene.updateMatrixWorld(); }
  for (const w of all) { fire(w.el, 'pointerdown', x, y, 0); w.scene.updateMatrixWorld(); }
  for (const [dx, dy] of moves) {
    for (const w of all) { fire(w.el, 'pointermove', x + dx, y + dy, -1); w.scene.updateMatrixWorld(); }
  }
  const [lx, ly] = moves[moves.length - 1];
  for (const w of all) {
    assert.equal(w.tc.dragging, true, `${w === a ? 'three' : 'shim'} dragging`);
    assert.equal(w.tc.axis, axis);
  }
  near(pose(b.object), pose(a.object), 0, `${opts.mode ?? 'translate'}/${opts.space ?? 'world'}/${axis} pose`);
  near(b.tc.pointEnd.toArray(), a.tc.pointEnd.toArray(), 0, 'pointEnd');
  for (const w of all) fire(w.el, 'pointerup', x + lx, y + ly, 0);
  assert.deepEqual(b.events, a.events, 'events');
  assert.equal(b.tc.dragging, false);
  assert.equal(b.tc.axis, null);
  return { a, b };
}

const MOVES = [[0.04, 0.03], [0.11, -0.05], [-0.07, 0.12]];

describe('TransformControls vs three.js r165', { skip }, () => {
  for (const space of ['world', 'local']) {
    for (const axis of ['X', 'Y', 'Z', 'XY', 'YZ', 'XZ', 'XYZ']) {
      it(`translate ${space} ${axis}`, () => { drag({ mode: 'translate', space }, axis, MOVES); });
    }
    for (const axis of ['X', 'Y', 'Z', 'E', 'XYZE']) {
      it(`rotate ${space} ${axis}`, () => { drag({ mode: 'rotate', space }, axis, MOVES); });
    }
    for (const axis of ['X', 'Y', 'Z', 'XY', 'YZ', 'XZ', 'XYZ']) {
      it(`scale ${space} ${axis}`, () => { drag({ mode: 'scale', space }, axis, MOVES); });
    }
  }
  it('translation snap, world and local', () => {
    drag({ mode: 'translate', space: 'world', translationSnap: 0.25 }, 'XZ', MOVES);
    drag({ mode: 'translate', space: 'local', translationSnap: 0.25 }, 'X', MOVES);
  });
  it('rotation and scale snap', () => {
    drag({ mode: 'rotate', space: 'local', rotationSnap: Math.PI / 12 }, 'Y', MOVES);
    drag({ mode: 'rotate', space: 'world', rotationSnap: Math.PI / 12 }, 'E', MOVES);
    drag({ mode: 'scale', scaleSnap: 0.1 }, 'Y', MOVES);
  });
  it('size changes where the handles are', () => {
    drag({ mode: 'translate', size: 1.7 }, 'Y', MOVES);
  });
  it('orthographic camera', () => {
    drag({ ortho: true, mode: 'translate' }, 'XY', MOVES);
    drag({ ortho: true, mode: 'rotate', space: 'local' }, 'Z', MOVES);
    drag({ ortho: true, mode: 'scale' }, 'XYZ', MOVES);
  });
  it('showX hides the X handles from the picker', () => {
    const b = setup(S, S.TransformControls, { showX: false });
    const a = setup(T, T_TC, { showX: false });
    const [x, y] = findPointer(a, 'Y');
    fire(b.el, 'pointermove', x, y, -1);
    assert.equal(b.tc.axis, 'Y');
    const xHandles = b.tc._gizmo.picker.translate.children.filter((h) => h.name.includes('X'));
    assert.ok(xHandles.length && xHandles.every((h) => !h.visible));
  });
  it('reset puts the object back mid-drag', () => {
    const { a, b } = { a: setup(T, T_TC, {}), b: setup(S, S.TransformControls, {}) };
    const [x, y] = findPointer(a, 'X');
    for (const w of [a, b]) {
      fire(w.el, 'pointermove', x, y, -1); w.scene.updateMatrixWorld();
      fire(w.el, 'pointerdown', x, y, 0); w.scene.updateMatrixWorld();
      fire(w.el, 'pointermove', x + 0.1, y, -1);
      w.tc.reset();
    }
    near(pose(b.object), pose(a.object), 0, 'reset pose');
  });
  it('handles, materials and plane are three.js\'s', () => {
    const a = setup(T, T_TC, {});
    const b = setup(S, S.TransformControls, {});
    for (const group of ['gizmo', 'picker', 'helper']) {
      for (const mode of ['translate', 'rotate', 'scale']) {
        const ha = a.tc._gizmo[group][mode].children, hb = b.tc._gizmo[group][mode].children;
        assert.equal(hb.length, ha.length, `${group}.${mode} count`);
        for (let i = 0; i < ha.length; i++) {
          const where = `${group}.${mode}[${i}] ${ha[i].name}`;
          assert.equal(hb[i].name, ha[i].name, where);
          assert.equal(hb[i].tag, ha[i].tag, where);
          assert.equal(hb[i].isLine === true, ha[i].isLine === true, `${where} line`);
          assert.equal(hb[i].material.opacity, ha[i].material.opacity, `${where} opacity`);
          assert.equal(hb[i].material.depthTest, false);
          for (const name of ['position', 'normal', 'uv']) {
            const ga = ha[i].geometry.attributes[name], gb = hb[i].geometry.attributes[name];
            assert.equal(!!gb, !!ga, `${where}.${name} present`);
            if (ga) assert.deepEqual(Array.from(gb.array), Array.from(ga.array), `${where}.${name}`);
          }
          const ia = ha[i].geometry.index, ib = hb[i].geometry.index;
          assert.deepEqual(ib ? Array.from(ib.array) : null, ia ? Array.from(ia.array) : null, `${where} index`);
          near(pose(hb[i]), pose(ha[i]), 0, `${where} placement`);
          assert.equal(hb[i].visible, ha[i].visible, `${where} visible`);
        }
      }
    }
    // Shared materials stay shared.
    const red = b.tc._gizmo.gizmo.translate.children.filter((h) => h.name === 'X').map((h) => h.material);
    assert.ok(red.length === 3 && red.every((m) => m === red[0]));
    const pa = a.tc._plane.geometry.attributes.position.array, pb = b.tc._plane.geometry.attributes.position.array;
    assert.deepEqual(Array.from(pb), Array.from(pa));
    // The plane follows the hovered handle. (Before any hover three.js
    // aims it with the `_dirVector` its module last left behind — shared by
    // every instance — so it is only comparable once an axis has set it.)
    for (const axis of ['XY', 'Y', 'XZ']) {
      const [x, y] = findPointer(a, axis);
      for (const w of [a, b]) { fire(w.el, 'pointermove', x, y, -1); w.scene.updateMatrixWorld(); }
      near(pose(b.tc._plane), pose(a.tc._plane), 0, `plane for ${axis}`);
    }
  });
  it('highlights the hovered axis as three.js does', () => {
    const a = setup(T, T_TC, { mode: 'rotate' });
    const b = setup(S, S.TransformControls, { mode: 'rotate' });
    const [x, y] = findPointer(a, 'Z');
    for (const w of [a, b]) { fire(w.el, 'pointermove', x, y, -1); w.scene.updateMatrixWorld(); }
    const ha = a.tc._gizmo.gizmo.rotate.children, hb = b.tc._gizmo.gizmo.rotate.children;
    for (let i = 0; i < ha.length; i++) {
      assert.equal(hb[i].material.color.getHex(), ha[i].material.color.getHex(), `${ha[i].name} colour`);
      assert.equal(hb[i].material.opacity, ha[i].material.opacity, `${ha[i].name} opacity`);
      near(pose(hb[i]), pose(ha[i]), 0, `${ha[i].name} placement`);
    }
  });
  it('mode / space / size setters dispatch as three.js', () => {
    const got = [];
    const b = setup(S, S.TransformControls, {});
    const a = setup(T, T_TC, {});
    const log = (tc, out) => ['mode-changed', 'space-changed', 'size-changed', 'change']
      .forEach((t) => tc.addEventListener(t, (e) => out.push(`${t}:${e.value}`)));
    const want = [];
    log(a.tc, want); log(b.tc, got);
    for (const tc of [a.tc, b.tc]) {
      tc.setMode('scale'); tc.setMode('scale'); tc.setSpace('local'); tc.setSize(2);
      assert.equal(tc.getMode(), 'scale');
    }
    assert.deepEqual(got, want);
  });
});

describe('FlyControls vs three.js r165', { skip }, () => {
  it('keys, pointer look and update(delta)', () => {
    const make = (L, Fly) => {
      const camera = new L.PerspectiveCamera(50, 4 / 3, 0.1, 100);
      camera.position.set(1, 2, 3);
      camera.rotation.set(0.1, -0.2, 0.05);
      const el = element();
      const c = new Fly(camera, el);
      c.movementSpeed = 3; c.rollSpeed = 0.4;
      let changes = 0;
      c.addEventListener('change', () => changes++);
      return { camera, el, c, changes: () => changes };
    };
    const a = make(T, T_FLY), b = make(S, S.FlyControls);
    const key = (type, code, extra = {}) => window._fire(type, { code, altKey: false, ...extra });
    const step = (dt) => { a.c.update(dt); b.c.update(dt); near(pose(b.camera), pose(a.camera), 1e-12, 'fly pose'); };
    key('keydown', 'KeyW'); step(0.1); step(0.05);
    key('keydown', 'KeyQ'); key('keydown', 'ArrowLeft'); step(0.1);
    for (const w of [a, b]) fire(w.el, 'pointermove', 0.3, -0.2, -1);
    step(0.2);
    for (const w of [a, b]) fire(w.el, 'pointerdown', 0.3, -0.2, 2);
    step(0.1);
    key('keyup', 'KeyW'); key('keyup', 'KeyQ'); key('keyup', 'ArrowLeft');
    key('keydown', 'KeyE', { altKey: true }); // ignored
    for (const w of [a, b]) fire(w.el, 'pointerup', 0.3, -0.2, 2);
    step(0.1);
    assert.deepEqual(b.c.moveVector.toArray(), a.c.moveVector.toArray());
    assert.deepEqual(b.c.rotationVector.toArray(), a.c.rotationVector.toArray());
    a.c.dragToLook = true; b.c.dragToLook = true;
    for (const w of [a, b]) { fire(w.el, 'pointerdown', 0, 0, 0); fire(w.el, 'pointermove', -0.5, 0.4, -1); }
    step(0.1);
    for (const w of [a, b]) fire(w.el, 'pointerup', -0.5, 0.4, 0);
    step(0.1); step(0.1);
    assert.equal(b.changes(), a.changes());
    for (const w of [a, b]) w.c.dispose();
  });
});

describe('MapControls vs three.js r165', { skip }, () => {
  const make = (L, Map) => {
    const camera = new L.PerspectiveCamera(50, 800 / 600, 0.1, 100);
    camera.position.set(0, 6, 4);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const el = element();
    const c = new Map(camera, el);
    c.update();
    return { camera, el, c };
  };
  const dragWith = (button, path) => {
    const a = make(T, T_MAP), b = make(S, S.MapControls);
    for (const w of [a, b]) {
      fire(w.el, 'pointerdown', path[0][0], path[0][1], button);
      for (const [x, y] of path.slice(1)) fire(w.el, 'pointermove', x, y, -1);
      fire(w.el, 'pointerup', path[path.length - 1][0], path[path.length - 1][1], button);
    }
    return { a, b };
  };
  it('left drag pans across the ground (screenSpacePanning off)', () => {
    const { a, b } = dragWith(0, [[0, 0], [0.05, -0.03], [0.12, -0.1]]);
    near(b.camera.position.toArray(), a.camera.position.toArray(), 1e-4, 'pan position');
    near(b.c.target.toArray(), a.c.target.toArray(), 1e-4, 'pan target');
    assert.equal(Math.abs(b.camera.position.y - 6) < 1e-4, true, 'height kept');
    assert.equal(b.c.screenSpacePanning, false);
  });
  it('right drag rotates', () => {
    const { a, b } = dragWith(2, [[0, 0], [0.1, 0.02], [0.2, 0.05]]);
    near(b.camera.position.toArray(), a.camera.position.toArray(), 1e-4, 'rotate position');
  });
});
