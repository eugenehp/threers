// The shim's scene graph against its wasm mirror, and both against three.js
// r165: one wasm node per Object3D with the same parent, the same children
// in the same order and the same local state, whatever the JS side did to
// get there — nested and re-parented groups, a mesh's own children, lights,
// lines, points and sprites anywhere in the tree, objects animated after
// they were added, `matrixAutoUpdate` off with a sheared matrix, every Euler
// order, multi-material meshes, morph targets, draw ranges, InstancedMesh's
// needsUpdate rules, skinning with a bind matrix, cameras in the graph. The
// wasm side is read back through `WebScene.nodeInfo` / `getWorldMatrix` /
// `getBoneMatrices`, and world matrices are compared with three.js's for the
// same scene. Also: three.js's DragControls driven by the same pointer events
// on both sides.
//
// The shim runs as a module over a real wasm build (`THREERS_PKG_DIR`, else
// web/pkg); three.js is web/node_modules/three (0.165). Rust composes in
// f32, so matrices agree to 64 f32 epsilons relative to max(1, |value|).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const threeDir = join(repo, 'web/node_modules/three');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  if (typeof m.WebDragControls !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-mirror-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const haveThree = existsSync(join(threeDir, 'build/three.module.js'));
const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const THREE = haveThree ? await import(pathToFileURL(join(threeDir, 'build/three.module.js')).href) : null;
const ThreeDrag = haveThree
  ? (await import(pathToFileURL(join(threeDir, 'examples/jsm/controls/DragControls.js')).href)).DragControls : null;
const skip = (!S && `no wasm build with the scene-graph mirror in ${pkgDir}`) || (!haveThree && 'three missing');

const TOL = 64 * 2 ** -24;
function nearArray(a, b, what, scale = 1) {
  assert.equal(a.length, b.length, `${what}: length`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) <= TOL * Math.max(scale, Math.abs(a[i]), Math.abs(b[i])),
      `${what}[${i}]: ${a[i]} vs ${b[i]}`);
  }
}
/// The node record of a shim object after a sync of its graph.
const info = (o) => { o._syncNow(); return o._m.graph.w.nodeInfo(o._m.handle); };
const worldOf = (o) => { o._syncNow(); return Array.from(o._m.graph.w.getWorldMatrix(o._m.handle)); };

/// Every object under `root`: its node's parent is its parent's node, its
/// node's children are its children's nodes, in order.
function assertIsomorphic(root) {
  root._syncNow();
  const visit = (o) => {
    const m = o._m;
    assert.ok(m, `${o.name || o.type} is mirrored`);
    const n = m.graph.w.nodeInfo(m.handle);
    assert.ok(n, `${o.name || o.type} has a live node`);
    if (o.parent) assert.equal(n.parent, o.parent._m.id, `${o.name || o.type}: parent`);
    assert.deepEqual(n.children, o.children.map((c) => c._m.id), `${o.name || o.type}: children`);
    assert.equal(n.visible, o.visible, `${o.name || o.type}: visible`);
    assert.equal(n.layers >>> 0, o.layers.mask >>> 0, `${o.name || o.type}: layers`);
    for (const c of o.children) visit(c);
  };
  visit(root);
}

/// Build the same tree in both libraries.
function tree(T) {
  const scene = new T.Scene();
  const geom = () => new T.BoxGeometry(1, 1, 1);
  const outer = new T.Group(); outer.name = 'outer';
  outer.position.set(1, 2, 3); outer.rotation.set(0.3, -0.2, 0.5); outer.scale.set(2, 0.5, 1.5);
  const inner = new T.Object3D(); inner.name = 'inner';
  inner.position.set(-0.5, 0.25, 1); inner.rotation.set(0.1, 0.7, 0); inner.scale.set(1.2, 1.2, 0.8);
  const mesh = new T.Mesh(geom(), new T.MeshBasicMaterial()); mesh.name = 'mesh';
  mesh.position.set(0.3, 0, 0); mesh.rotation.set(0, 0, 0.4);
  const meshChild = new T.Mesh(geom(), new T.MeshBasicMaterial()); meshChild.name = 'meshChild';
  meshChild.position.set(0, 1, 0); meshChild.scale.set(0.5, 0.5, 0.5);
  const light = new T.PointLight(0xffffff, 1); light.name = 'light'; light.position.set(0.2, 0.3, 0.4);
  const lineGeom = new T.BufferGeometry();
  lineGeom.setAttribute('position', new T.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1, 2, 0, 1]), 3));
  const line = new T.Line(lineGeom, new T.LineBasicMaterial()); line.name = 'line'; line.position.x = 0.7;
  const points = new T.Points(lineGeom, new T.PointsMaterial()); points.name = 'points'; points.rotation.y = 1;
  const sprite = new T.Sprite(new T.SpriteMaterial()); sprite.name = 'sprite'; sprite.position.set(0, 2, 0);
  mesh.add(meshChild);
  inner.add(mesh, light, line, points);
  outer.add(inner);
  outer.add(sprite);
  scene.add(outer);
  return { scene, outer, inner, mesh, meshChild, light, line, points, sprite };
}

function compareWorlds(t, s, names) {
  t.scene.updateMatrixWorld(true);
  for (const n of names) nearArray(worldOf(s[n]), Array.from(t[n].matrixWorld.elements), `${n} world`, 10);
}

describe('the wasm scene graph mirrors the JS one', { skip }, () => {
  it('one node per object, with its parent and children in order (nested groups, a mesh\'s child, lights, lines, points, sprites)', () => {
    const s = tree(S);
    assertIsomorphic(s.scene);
    assert.equal(info(s.mesh).kind, 'mesh');
    assert.equal(info(s.light).kind, 'light');
    assert.equal(info(s.line).kind, 'line');
    assert.equal(info(s.line).lineMode, 1, 'a Line is a strip');
    assert.equal(info(s.points).kind, 'points');
    assert.equal(info(s.sprite).kind, 'sprite');
    assert.equal(info(s.outer).isGroup, true);
    assert.equal(info(s.inner).isGroup, false, 'an Object3D is not a Group');
  });
  it('nested rotation and non-uniform scale compose as in three.js', () => {
    const t = tree(THREE), s = tree(S);
    compareWorlds(t, s, ['outer', 'inner', 'mesh', 'meshChild', 'light', 'line', 'points', 'sprite']);
  });
  it('a mesh in a Group animated after it was added follows', () => {
    const t = tree(THREE), s = tree(S);
    s.scene._syncNow();
    for (const T of [t, s]) {
      T.mesh.position.set(-1, 0.5, 2);
      T.mesh.rotation.y += 0.9;
      T.mesh.scale.x = 3;
      T.outer.quaternion.setFromAxisAngle(new (T === t ? THREE : S).Vector3(0, 1, 0), 1.1);
      T.meshChild.rotateX(0.4);
    }
    compareWorlds(t, s, ['outer', 'mesh', 'meshChild']);
  });
  it('Group.add / remove / attach after the group is in the scene', () => {
    const t = tree(THREE), s = tree(S);
    s.scene._syncNow();
    for (const [T, L] of [[t, THREE], [s, S]]) {
      const extra = new L.Mesh(new L.BoxGeometry(1, 1, 1), new L.MeshBasicMaterial()); extra.name = 'extra';
      extra.position.set(1, 1, 1);
      T.inner.add(extra);
      T.extra = extra;
      T.inner.remove(T.line);
      T.outer.add(T.line);
      T.scene.attach(T.meshChild);
    }
    assertIsomorphic(s.scene);
    compareWorlds(t, s, ['extra', 'line', 'meshChild']);
    const node = s.line._m.handle;
    s.outer.remove(s.line);
    assertIsomorphic(s.scene);
    assert.equal(s.scene._w.nodeInfo(node).parent, undefined, 'removed: detached in wasm');
  });
  it('matrixAutoUpdate off: the matrix as written, shear included', () => {
    const t = tree(THREE), s = tree(S);
    for (const [T, L] of [[t, THREE], [s, S]]) {
      T.mesh.matrixAutoUpdate = false;
      T.mesh.matrix.makeShear(0.3, 0, 0.2, 0, 0, 0.5).setPosition(0.5, -0.5, 0.25);
    }
    compareWorlds(t, s, ['mesh', 'meshChild']);
    assert.equal(info(s.mesh).matrixAutoUpdate, false);
    // A matrix written in place is picked up at the next sync.
    for (const T of [t, s]) T.mesh.matrix.elements[12] = 4;
    compareWorlds(t, s, ['mesh', 'meshChild']);
    for (const T of [t, s]) { T.mesh.matrixAutoUpdate = true; T.mesh.position.set(0, 1, 0); }
    compareWorlds(t, s, ['mesh']);
  });
  it('every Euler order, and quaternion writes, as three.js composes them', () => {
    for (const order of ['XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY']) {
      const t = tree(THREE), s = tree(S);
      for (const T of [t, s]) { T.inner.rotation.order = order; T.inner.rotation.set(0.9, -0.6, 1.3); }
      compareWorlds(t, s, ['inner', 'mesh']);
      assert.ok(Math.abs(s.inner.quaternion.x - t.inner.quaternion.x) < 1e-12, `${order}: quaternion`);
    }
    const t = tree(THREE), s = tree(S);
    for (const T of [t, s]) { T.mesh.quaternion.set(0.1, 0.7, -0.2, 0.67).normalize(); }
    compareWorlds(t, s, ['mesh']);
    assert.ok(Math.abs(s.mesh.rotation.y - t.mesh.rotation.y) < 1e-12, 'rotation follows the quaternion');
  });
  it('visibility and layers per node (an invisible group hides its subtree in Rust\'s traversal)', () => {
    const s = tree(S);
    s.inner.visible = false;
    s.mesh.layers.set(3);
    s.light.layers.enable(5);
    assertIsomorphic(s.scene);
    assert.equal(info(s.inner).visible, false);
    assert.equal(info(s.mesh).visible, true, 'children keep their own flag');
    assert.equal(info(s.mesh).layers, 8);
    assert.equal(info(s.light).layers, 33);
    s.renderOrder = 0;
    s.mesh.renderOrder = 7; s.mesh.castShadow = true;
    assert.equal(info(s.mesh).renderOrder, 7);
    assert.equal(info(s.mesh).castShadow, true);
  });
  it('a multi-material mesh is one node with groups and a material array', () => {
    const g = new S.BufferGeometry();
    g.setAttribute('position', new S.BufferAttribute(new Float32Array(18), 3));
    g.addGroup(0, 3, 0); g.addGroup(3, 3, 1);
    const m = new S.Mesh(g, [new S.MeshBasicMaterial(), new S.MeshBasicMaterial()]);
    const scene = new S.Scene(); scene.add(m);
    assert.equal(info(m).materials, 2);
    assert.equal(info(m).groups, 2);
    assert.deepEqual(info(scene).children, [m._m.id], 'no sub-meshes');
    m.material = new S.MeshBasicMaterial();
    assert.equal(info(m).materials, 0, 'back to one material');
    const box = new S.Mesh(new S.BoxGeometry(1, 1, 1), Array.from({ length: 6 }, () => new S.MeshBasicMaterial()));
    scene.add(box);
    assert.equal(info(box).groups, 6, 'BoxGeometry has three.js\'s six groups');
  });
  it('morph influences and the draw range reach the node; no blending in JS', () => {
    const g = new S.BufferGeometry();
    const base = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    g.setAttribute('position', new S.BufferAttribute(base.slice(), 3));
    g.morphAttributes = { position: [new S.BufferAttribute(new Float32Array(9).fill(1), 3)] };
    const m = new S.Mesh(g, new S.MeshBasicMaterial());
    const scene = new S.Scene(); scene.add(m);
    assert.deepEqual(info(m).morphInfluences, [0]);
    assert.equal(info(m).morphTargets, 1);
    m.morphTargetInfluences[0] = 0.25;
    assert.deepEqual(info(m).morphInfluences, [0.25]);
    assert.deepEqual(Array.from(g.attributes.position.array), Array.from(base), 'positions untouched');
    g.setDrawRange(3, 6);
    assert.deepEqual(info(m).drawRange, [3, 6]);
    g.drawRange.count = Infinity;
    assert.deepEqual(info(m).drawRange, [3, Infinity]);
  });
  it('InstancedMesh: matrices and colours on needsUpdate, count live', () => {
    const im = new S.InstancedMesh(new S.BoxGeometry(1, 1, 1), new S.MeshBasicMaterial(), 3);
    const scene = new S.Scene(); scene.add(im);
    im.position.set(5, 0, 0);
    assert.equal(info(im).instances, 3);
    const read = () => { im._syncNow(); return Array.from(scene._w.getInstanceMatrices(im._m.handle)); };
    const m = new S.Matrix4().makeTranslation(1, 2, 3);
    im.setMatrixAt(1, m);
    assert.equal(read()[16 + 12], 0, 'not uploaded without needsUpdate (three.js)');
    im.instanceMatrix.needsUpdate = true;
    assert.equal(read()[16 + 12], 1, 'uploaded on needsUpdate');
    im.count = 2;
    assert.equal(info(im).instances, 2);
    im.setColorAt(0, new S.Color(1, 0, 0));
    im.instanceColor.needsUpdate = true;
    assert.equal(info(im).instanceColors, 2);
    nearArray(worldOf(im), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1], 'own transform');
  });
  it('a light\'s position, colour and intensity are live', () => {
    const s = tree(S);
    s.scene._syncNow();
    s.light.position.x = 5;
    const t = tree(THREE); t.light.position.x = 5;
    compareWorlds(t, s, ['light']);
    s.light.intensity = 3;
    s.light.color.setHex(0xff0000);
    s.scene._syncNow();
    assert.deepEqual(s.light._m.light.slice(0, 4), [1, 0, 0, 3]);
  });
  it('skinning: bone palette from the mirrored bones, with the bind matrix (attached and detached)', () => {
    const build = (L, mode) => {
      const g = new L.BufferGeometry();
      g.setAttribute('position', new L.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
      g.setAttribute('skinIndex', new L.BufferAttribute(new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4));
      g.setAttribute('skinWeight', new L.BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4));
      const mesh = new L.SkinnedMesh(g, new L.MeshBasicMaterial());
      const b0 = new L.Bone(), b1 = new L.Bone();
      b0.position.set(0.2, -1, 0); b1.position.set(0, 1, 0.3);
      b0.add(b1); mesh.add(b0);
      const scene = new L.Scene(); scene.add(mesh);
      mesh.position.set(1, 0.5, -0.5); mesh.rotation.set(0.2, 0.3, 0);
      mesh.bind(new L.Skeleton([b0, b1]));
      if (mode === 'detached') mesh.bindMode = 'detached';
      mesh.position.set(-0.3, 0.2, 0.4);
      b1.rotation.set(0.5, 0, 0.6);
      return { scene, mesh };
    };
    for (const mode of ['attached', 'detached']) {
      const t = build(THREE, mode), s = build(S, mode);
      t.scene.updateMatrixWorld(true);
      t.mesh.skeleton.update();
      s.mesh._syncNow();
      const got = Array.from(s.scene._w.getBoneMatrices(s.mesh._m.handle));
      // three.js: modelMatrix · bindMatrixInverse · boneMatrix · bindMatrix.
      const pre = mode === 'attached' ? new THREE.Matrix4()
        : new THREE.Matrix4().multiplyMatrices(t.mesh.matrixWorld, t.mesh.bindMatrixInverse);
      for (let i = 0; i < 2; i++) {
        const bone = new THREE.Matrix4().fromArray(t.mesh.skeleton.boneMatrices, i * 16);
        const want = new THREE.Matrix4().multiplyMatrices(pre, bone).multiply(t.mesh.bindMatrix);
        nearArray(got.slice(i * 16, i * 16 + 16), Array.from(want.elements), `${mode} bone ${i}`, 10);
      }
    }
  });
  it('a camera is a node: parented, and posed from its world matrix', () => {
    const t = tree(THREE), s = tree(S);
    for (const [T, L] of [[t, THREE], [s, S]]) {
      const cam = new L.PerspectiveCamera(50, 1.5, 0.1, 100);
      cam.position.set(0, 0.5, 6);
      cam.lookAt(0, 0, 0);
      T.outer.add(cam);
      T.cam = cam;
    }
    assertIsomorphic(s.scene);
    t.scene.updateMatrixWorld(true);
    s.cam._sync();
    const view = Array.from(s.cam._w.viewMatrix());
    nearArray(view, Array.from(t.cam.matrixWorldInverse.elements), 'view matrix', 10);
  });
  it('an object outside any Scene is mirrored on demand, and moves into a Scene it joins', () => {
    const lone = new S.Group();
    const m = new S.Mesh(new S.BoxGeometry(1, 1, 1), new S.MeshBasicMaterial());
    lone.add(m);
    // Off the face's diagonal (a ray along a shared edge hits both triangles,
    // in three.js as here).
    const r = new S.Raycaster(new S.Vector3(0.1, 0.2, 5), new S.Vector3(0, 0, -1));
    assert.equal(r.intersectObject(lone).length, 1);
    assert.equal(m._m.graph.scene, null, 'the hidden graph');
    const scene = new S.Scene();
    scene.add(lone);
    assertIsomorphic(scene);
    assert.equal(m._m.graph, scene._graph);
    lone.position.x = 10;
    // three.js casts against matrixWorld as it stands: still hit until updated.
    assert.equal(r.intersectObject(scene).length, 1);
    scene.updateMatrixWorld();
    assert.equal(r.intersectObject(scene).length, 0);
  });
  it('BatchedMesh draws are internal nodes, not scene children', () => {
    const b = new S.BatchedMesh(4, 100, 100, new S.MeshBasicMaterial());
    const scene = new S.Scene(); scene.add(b);
    const id = b.addGeometry(new S.BoxGeometry(1, 1, 1));
    b.setMatrixAt(id, new S.Matrix4().makeTranslation(2, 0, 0));
    assert.deepEqual(scene.children, [b]);
    assert.equal(info(b).children.length, 1);
    const r = new S.Raycaster(new S.Vector3(2.1, 0.2, 5), new S.Vector3(0, 0, -1));
    const hits = r.intersectObject(scene);
    assert.equal(hits.length, 1);
    assert.equal(hits[0].object, b);
    assert.equal(hits[0].batchId, id);
  });
});

// ---- DragControls ----

/// A canvas-like element both libraries' DragControls listen on.
function element() {
  const on = {};
  return {
    on, style: {},
    addEventListener(type, fn) { (on[type] ??= []).push(fn); },
    removeEventListener(type, fn) { on[type] = (on[type] ?? []).filter((f) => f !== fn); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 400, height: 300 }),
    fire(type, e) { for (const f of on[type] ?? []) f({ pointerType: 'mouse', button: 0, ...e }); },
  };
}

function dragScene(L) {
  const scene = new L.Scene();
  const group = new L.Group();
  group.position.set(0.5, 0.2, 0); group.scale.set(2, 1.5, 1); group.rotation.set(0, 0, 0.3);
  const box = new L.Mesh(new L.BoxGeometry(0.6, 0.6, 0.6), new L.MeshBasicMaterial());
  box.name = 'box'; box.position.set(0.1, 0.1, 0);
  group.add(box);
  scene.add(group);
  const camera = new L.PerspectiveCamera(50, 4 / 3, 0.1, 100);
  camera.position.set(0.4, 0.3, 5);
  camera.lookAt(0.4, 0.3, 0);
  camera.updateMatrixWorld();
  scene.updateMatrixWorld(true);
  return { scene, group, box, camera };
}

describe('DragControls (Rust) matches three.js r165', { skip }, () => {
  for (const transformGroup of [false, true]) {
    it(`drags a parented object along the camera plane${transformGroup ? ' (transformGroup)' : ''}`, () => {
      const run = (L, Controls) => {
        const d = dragScene(L), el = element(), events = [];
        const dc = new Controls([d.box], d.camera, el);
        dc.transformGroup = transformGroup;
        for (const type of ['hoveron', 'hoveroff', 'dragstart', 'drag', 'dragend']) {
          dc.addEventListener(type, (e) => events.push(`${type}:${e.object.name || e.object.type}`));
        }
        // The box's centre on screen, then a drag.
        const p = new L.Vector3().setFromMatrixPosition(d.box.matrixWorld).project(d.camera);
        const px = (p.x + 1) / 2 * 400, py = (1 - p.y) / 2 * 300;
        el.fire('pointermove', { clientX: px, clientY: py });
        el.fire('pointerdown', { clientX: px, clientY: py });
        for (let k = 1; k <= 3; k++) {
          el.fire('pointermove', { clientX: px + 25 * k, clientY: py - 10 * k });
          d.scene.updateMatrixWorld(true);
        }
        el.fire('pointerup', {});
        el.fire('pointermove', { clientX: 5, clientY: 5 });
        return { d, events, cursor: el.style.cursor };
      };
      const t = run(THREE, ThreeDrag), s = run(S, S.DragControls);
      assert.deepEqual(s.events, t.events);
      assert.equal(s.cursor, t.cursor);
      const moved = transformGroup ? 'group' : 'box';
      const a = s.d[moved].position, b = t.d[moved].position;
      nearArray([a.x, a.y, a.z], [b.x, b.y, b.z], `${moved} position`, 10);
    });
  }
});
