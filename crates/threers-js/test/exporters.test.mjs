// The shim's OBJExporter / STLExporter / PLYExporter / GLTFExporter — thin
// wrappers over the Rust writers (`WebSceneExporter`) — against three.js
// r165's own exporters on the same scene: a Scene holding a rotated, scaled
// Group with an indexed box and a point cloud, an unindexed triangle soup
// mirrored by a negative scale, a line and line segments.
//
// The shim side is the shim module itself over a real wasm build
// (`THREERS_PKG_DIR`, else web/pkg). Skipped when that build predates
// `WebSceneExporter` or the scene-graph mirror.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const threeDir = join(repo, 'web/node_modules/three');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

/// The shim as a module whose relative imports point at `dir` and web/.
async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  if (typeof m.WebSceneExporter !== 'function' || typeof m.WebDragControls !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-exporters-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const haveThree = existsSync(join(threeDir, 'build/three.module.js'));
const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const skip = (!S && `${pkgDir} has no WebSceneExporter or predates the scene-graph mirror`)
  || (!haveThree && 'web/node_modules/three missing');

/// The same scene twice: once from three.js's classes, once from the shim's
/// (stand-ins over the shim's BufferGeometry and the real wasm).
function buildScene(T, M) {
  const geom = (attrs, index) => {
    const g = new M.BufferGeometry();
    for (const [name, [array, size]] of Object.entries(attrs)) {
      g.setAttribute(name, new M.BufferAttribute(new Float32Array(array), size));
    }
    if (index) g.setIndex(T.isThree ? Array.from(index) : new Uint32Array(index));
    return g;
  };
  const scene = new T.Scene();
  const group = new T.Group(); group.name = 'G';
  group.position.x = 1; group.position.y = 2; group.position.z = 3;
  set3(group.rotation, 0.3, 0.5, 0.7); set3(group.scale, 2, 0.5, 1.5);
  scene.add(group);
  const box = T.boxArrays;
  const boxMesh = new T.Mesh(geom({ position: [box.position, 3], normal: [box.normal, 3], uv: [box.uv, 2] }, box.index),
    new T.MeshStandardMaterial({ name: 'boxmat', metalness: 0.25, roughness: 0.5 }));
  boxMesh.name = 'boxy';
  set3(boxMesh.position, -1, 0.25, 0); set3(boxMesh.rotation, -0.2, 0.1, 1.1); set3(boxMesh.scale, 1, 1.25, 0.75);
  group.add(boxMesh);
  const soup = new T.Mesh(geom({
    position: [[0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.3, 0.2, 0.1, 1.7, -0.4, 0.25, 0.1, 0.9, -0.6], 3],
    normal: [[0, 0, 1, 0, 0.6, 0.8, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0.6, 0, 0.8], 3],
    color: [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.5, 0.5, 0.001, 0.2, 0.9, 1, 1, 1], 3],
  }), new T.MeshBasicMaterial({ transparent: true, opacity: 0.5 }));
  soup.name = 'soup'; set3(soup.scale, 1, -1, 1); set3(soup.position, 0.1, 0.2, 0.3);
  scene.add(soup);
  const pts = new T.Points(geom({ position: [[0.5, 0.5, 0.5, -1, 2, 3, 0.1, 0.2, 0.3], 3],
    color: [[0.2, 0.4, 0.6, 1, 0.5, 0, 0, 0, 0], 3] }));
  pts.name = 'pts'; set3(pts.position, 0, 1, 0);
  group.add(pts);
  const linePos = [[0, 0, 0, 1, 1, 0, 2, 0, 1, 3, 1, 1], 3];
  const line = new T.Line(geom({ position: linePos })); line.name = 'ln';
  const segs = new T.LineSegments(geom({ position: linePos })); segs.name = 'segs'; set3(segs.position, 0.5, 0, 0);
  scene.add(line); scene.add(segs);
  const hidden = new T.Mesh(geom({ position: [[0, 0, 0, 1, 0, 0, 0, 1, 0], 3] }), new T.MeshStandardMaterial());
  hidden.name = 'hidden'; hidden.visible = false;
  scene.add(hidden);
  if (T.isThree) scene.updateMatrixWorld(true);
  return { scene, group, boxMesh, soup };
}
function set3(v, x, y, z) { v.x = x; v.y = y; v.z = z; }

let THREE = null, ref = null, ours = null;
if (!skip) {
  THREE = await import(pathToFileURL(join(threeDir, 'build/three.module.js')).href);
  const ex = async (n) => import(pathToFileURL(join(threeDir, `examples/jsm/exporters/${n}.js`)).href);
  ref = {
    OBJExporter: (await ex('OBJExporter')).OBJExporter, STLExporter: (await ex('STLExporter')).STLExporter,
    PLYExporter: (await ex('PLYExporter')).PLYExporter, GLTFExporter: (await ex('GLTFExporter')).GLTFExporter,
  };
  const b = new THREE.BoxGeometry(1, 2, 3);
  const boxArrays = Object.fromEntries(['position', 'normal', 'uv'].map((n) => [n, Array.from(b.getAttribute(n).array)]));
  boxArrays.index = Array.from(b.index.array);
  const threeSide = { ...THREE, isThree: true, boxArrays };
  ref.built = buildScene(threeSide, THREE);
  ours = { ...S };
  ours.built = buildScene({ ...S, boxArrays }, S);
}
globalThis.requestAnimationFrame ??= (f) => setTimeout(f, 0);
// three.js's GLTFExporter reads its Blobs back through FileReader, which node lacks.
globalThis.FileReader ??= class {
  readAsDataURL(blob) {
    blob.arrayBuffer().then((b) => {
      this.result = `data:${blob.type || 'application/octet-stream'};base64,${Buffer.from(b).toString('base64')}`;
      this.onloadend?.();
    });
  }
  readAsArrayBuffer(blob) { blob.arrayBuffer().then((b) => { this.result = b; this.onloadend?.(); }); }
};

/// Text equal, or the first differing line.
function sameText(got, want, what) {
  if (got === want) return;
  const a = got.split('\n'), b = want.split('\n');
  for (let i = 0; i < Math.max(a.length, b.length); i++) assert.equal(a[i], b[i], `${what}: line ${i + 1}`);
}
const bytes = (v) => new Uint8Array(v instanceof DataView ? v.buffer : v);

describe('exporters match three.js r165', { skip }, () => {
  it('OBJ, byte for byte', () => {
    sameText(new ours.OBJExporter().parse(ours.built.scene), new ref.OBJExporter().parse(ref.built.scene), 'obj');
    sameText(new ours.OBJExporter().parse(ours.built.group), new ref.OBJExporter().parse(ref.built.group), 'obj subtree');
  });
  it('STL ASCII and binary, byte for byte', () => {
    sameText(new ours.STLExporter().parse(ours.built.scene), new ref.STLExporter().parse(ref.built.scene), 'stl');
    const got = new ours.STLExporter().parse(ours.built.scene, { binary: true });
    assert.ok(got instanceof DataView, 'binary STL is a DataView, as in three.js');
    assert.deepEqual(bytes(got), bytes(new ref.STLExporter().parse(ref.built.scene, { binary: true })));
  });
  it('PLY ASCII and binary, both byte orders, byte for byte', () => {
    for (const root of ['scene', 'group', 'soup', 'boxMesh']) {
      const opts = [{}, { binary: true }, { binary: true, littleEndian: true },
        { excludeAttributes: ['index', 'color'] }, { excludeAttributes: ['normal', 'uv'], binary: true }];
      for (const o of opts) {
        const got = new ours.PLYExporter().parse(ours.built[root], null, o);
        const want = new ref.PLYExporter().parse(ref.built[root], null, o);
        if (typeof want === 'string') sameText(got, want, `ply ${root} ${JSON.stringify(o)}`);
        else assert.deepEqual(bytes(got), bytes(want), `ply ${root} ${JSON.stringify(o)}`);
      }
    }
  });
  it('PLY calls onDone with the result', async () => {
    const result = await new Promise((resolve) => new ours.PLYExporter().parse(ours.built.soup, resolve));
    assert.match(result, /^ply\nformat ascii 1.0\n/);
  });
  it('glTF has three.js\'s structure', async () => {
    const want = await new ref.GLTFExporter().parseAsync(ref.built.scene, {});
    const got = await new ours.GLTFExporter().parseAsync(ours.built.scene, {});
    assert.equal(got.asset.generator, want.asset.generator);
    // Nodes by name: same transforms (as matrices), same mesh/children shape.
    const byName = (j) => Object.fromEntries(j.nodes.map((n) => [n.name, n]));
    const [g, w] = [byName(got), byName(want)];
    assert.deepEqual(Object.keys(g).sort(), ['G', 'boxy', 'segs', 'soup', 'ln', 'pts'].sort().filter((n) => n in w).sort());
    assert.ok(!('hidden' in g), 'onlyVisible drops the hidden mesh');
    for (const name of ['G', 'boxy', 'soup']) {
      const [a, b] = [g[name].matrix ?? [], w[name].matrix ?? []];
      assert.equal(a.length, b.length, `${name} matrix`);
      a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-6, `${name} matrix[${i}] ${v} vs ${b[i]}`));
    }
    for (const name of ['boxy', 'soup']) {
      const [pa, pb] = [got.meshes[g[name].mesh].primitives[0], want.meshes[w[name].mesh].primitives[0]];
      assert.deepEqual(Object.keys(pa.attributes).sort(), Object.keys(pb.attributes).sort(), `${name} attributes`);
      for (const k of Object.keys(pa.attributes)) {
        const [x, y] = [got.accessors[pa.attributes[k]], want.accessors[pb.attributes[k]]];
        assert.equal(x.count, y.count, `${name} ${k} count`);
        assert.equal(x.type, y.type, `${name} ${k} type`);
        if (y.min) y.min.forEach((v, i) => assert.ok(Math.abs(v - x.min[i]) < 1e-6));
      }
      assert.equal(pa.indices === undefined, pb.indices === undefined, `${name} indexed`);
      const [ma, mb] = [got.materials[pa.material], want.materials[pb.material]];
      assert.equal(ma.name, mb.name, `${name} material name`);
      assert.equal(ma.alphaMode, mb.alphaMode, `${name} alphaMode`);
      assert.deepEqual(Object.keys(ma.extensions ?? {}), Object.keys(mb.extensions ?? {}), `${name} extensions`);
      for (const k of ['metallicFactor', 'roughnessFactor']) {
        assert.ok(Math.abs(ma.pbrMetallicRoughness[k] - mb.pbrMetallicRoughness[k]) < 1e-6, `${name} ${k}`);
      }
      const [ca, cb] = [ma.pbrMetallicRoughness.baseColorFactor, mb.pbrMetallicRoughness.baseColorFactor];
      assert.equal(ca === undefined, cb === undefined, `${name} baseColorFactor presence`);
    }
    assert.deepEqual(got.extensionsUsed ?? [], want.extensionsUsed ?? []);
    assert.match(got.buffers[0].uri, /^data:application\/octet-stream;base64,/);
  });
  it('glTF binary is a GLB', async () => {
    const glb = await new ours.GLTFExporter().parseAsync(ours.built.scene, { binary: true });
    assert.ok(glb instanceof ArrayBuffer);
    const dv = new DataView(glb);
    assert.equal(dv.getUint32(0, true), 0x46546c67, 'glTF magic');
    assert.equal(dv.getUint32(8, true), glb.byteLength);
  });
  it('polygons: one face per quad when asked', () => {
    const g = new ours.BufferGeometry();
    const p = [];
    for (let y = 0; y <= 1; y++) for (let x = 0; x <= 3; x++) p.push(x, y, 0);
    g.setAttribute('position', new ours.BufferAttribute(new Float32Array(p), 3));
    const idx = [];
    for (let x = 0; x < 3; x++) idx.push(x, x + 1, x + 5, x, x + 5, x + 4);
    g.setIndex(new Uint32Array(idx));
    g.setPolygons([2, 2, 2]);
    const m = new ours.Mesh(g);
    const faces = (s) => s.split('\n').filter((l) => l.startsWith('f '));
    assert.equal(faces(new ours.OBJExporter().parse(m)).length, 6);
    assert.deepEqual(faces(new ours.OBJExporter().parse(m, { polygons: true })), ['f 1 2 6 5', 'f 2 3 7 6', 'f 3 4 8 7']);
    const ply = new ours.PLYExporter().parse(m, null, { polygons: true });
    assert.match(ply, /element face 3\n/);
    assert.match(ply, /\n4 0 1 5 4\n4 1 2 6 5\n4 2 3 7 6\n\n$/);
  });
  it('Object3D traversal walks children, honours visibility and finds by name', () => {
    const root = ours.built.scene;
    const seen = [];
    root.traverse((o) => seen.push(o.name));
    assert.deepEqual(seen, ['', 'G', 'boxy', 'pts', 'soup', 'ln', 'segs', 'hidden']);
    const visible = [];
    root.traverseVisible((o) => visible.push(o.name));
    assert.ok(!visible.includes('hidden'));
    assert.equal(root.getObjectByName('pts'), ours.built.group.children[1]);
    const up = [];
    ours.built.boxMesh.traverseAncestors((o) => up.push(o.name));
    assert.deepEqual(up, ['G', '']);
  });
});
