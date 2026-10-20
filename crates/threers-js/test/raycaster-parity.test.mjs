// The shim's Raycaster against three.js r165 itself: the same scenes built
// in both, the same rays cast, and the hit lists compared field by field —
// object, distance, point, face, faceIndex, uv, uv1, normal, instanceId,
// index, distanceToRay — plus which fields are present. three.js is the
// `three` devDependency of web/ (web/node_modules/three, pinned 0.165); the
// suite skips when it, or a wasm build, is missing.
//
// The shim runs as it does in a page: the module itself over a real wasm
// build (`THREERS_PKG_DIR`, else web/pkg), its objects mirrored into the Rust
// scene graph and cast by the Rust raycaster. Also covered: three-mesh-bvh's
// `acceleratedRaycast` (web/mesh-bvh-impl.js) over the wasm BVH, with and
// without `firstHitOnly`.
//
// Tolerance. Rust casts in f32, three.js in f64, so values are compared to a
// bound derived from f32 rather than to 1e-9: a float rounds with relative
// error ε = 2^-24 ≈ 6e-8, and a hit is the end of a short chain — compose
// the world matrix (a few products per element), invert it, carry the ray
// into the object's space, intersect (cross products, a division),
// interpolate, carry the point back. Each step's rounding is ε relative to
// the magnitudes it works on, which here (scenes a few units across, rays
// starting 4–6 units out) are O(10); the chain adds some tens of such
// errors. So the bound is 64·ε ≈ 4e-6 relative to max(1, |value|, scene
// scale), with the scene scale 10 — about 4e-5 absolute on these scenes.
// What a cast can decide differently in f32 (a ray grazing an edge, a point
// right at a threshold) is far less likely than one per suite at 300 rays a
// scene; a mismatch in the hit list is still a failure.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const threePath = join(repo, 'web/node_modules/three/build/three.module.js');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

/// The shim and web/mesh-bvh-impl.js as modules over `pkgDir`'s wasm (their
/// imports pointed at it and at web/).
async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const pkg = await import(pkgUrl);
  if (typeof pkg.WebDragControls !== 'function') return null;
  pkg.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const tmp = mkdtempSync(join(tmpdir(), 'threers-raycast-'));
  const shimFile = join(tmp, 'shim.mjs');
  writeFileSync(shimFile, readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`));
  const bvhFile = join(tmp, 'mesh-bvh.mjs');
  writeFileSync(bvhFile, readFileSync(join(web, 'mesh-bvh-impl.js'), 'utf8')
    .replaceAll("'/web/threejs-shim.js'", JSON.stringify(pathToFileURL(shimFile).href))
    .replaceAll("'/web/pkg/threers.js'", JSON.stringify(pkgUrl))
    .replaceAll("'/web/features.js'", JSON.stringify(pathToFileURL(join(web, 'features.js')).href)));
  const shim = await import(pathToFileURL(shimFile).href);
  const bvh = typeof pkg.WebMeshBvh === 'function' ? await import(pathToFileURL(bvhFile).href) : null;
  return { shim, bvh };
}

const ready = existsSync(threePath) && existsSync(join(pkgDir, 'threers_bg.wasm'));
const loaded = ready ? await loadShimOver(pkgDir) : null;
const S = loaded?.shim ?? null;
const BVH = loaded?.bvh ?? null;
const THREE = ready ? await import(pathToFileURL(threePath).href) : null;
const skip = !ready ? 'three@0.165 or a wasm build is missing'
  : !S ? `the wasm build in ${pkgDir} predates the scene-graph mirror (no WebDragControls)` : false;

// ---- scenes, built the same way in both libraries ----

const isShim = (T) => T === S;
function material(T, kind, opts = {}) {
  const C = { mesh: T.MeshBasicMaterial, line: T.LineBasicMaterial, points: T.PointsMaterial,
    sprite: T.SpriteMaterial }[kind];
  return new C(opts);
}
function attr(T, array, itemSize) {
  return new T.BufferAttribute(new Float32Array(array), itemSize);
}
function index(T, g, idx) {
  if (isShim(T)) g.setIndex(new Uint32Array(idx));
  else g.setIndex(idx);
}
const root = (T) => new T.Scene();
function named(obj, name) { obj.name = name; return obj; }
function place(obj, { p = [0, 0, 0], r = [0, 0, 0], s = [1, 1, 1] } = {}) {
  obj.position.set(...p);
  obj.rotation.set(...r);
  obj.scale.set(...s);
  return obj;
}

/// A 3×2-cell grid in z = 0 spanning [-1.5, 1.5] × [-1, 1], with normal, uv
/// and a uv1 that differs from uv.
function grid(T, { indexed = true } = {}) {
  const nx = 3, ny = 2, pos = [], nor = [], uv = [], uv1 = [], idx = [];
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) {
      pos.push(-1.5 + i, -1 + j, 0.15 * Math.sin(i + 2 * j));
      const n = [0.1 * i, 0.05 * j, 1]; const l = Math.hypot(...n);
      nor.push(n[0] / l, n[1] / l, n[2] / l);
      uv.push(i / nx, j / ny);
      uv1.push(1 - j / ny, 0.5 * i / nx);
    }
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 2, d = a + nx + 1;
      idx.push(a, b, c, a, c, d);
    }
  }
  const g = new T.BufferGeometry();
  if (indexed) {
    g.setAttribute('position', attr(T, pos, 3));
    g.setAttribute('normal', attr(T, nor, 3));
    g.setAttribute('uv', attr(T, uv, 2));
    g.setAttribute('uv1', attr(T, uv1, 2));
    index(T, g, idx);
  } else {
    const flat = (src, n) => idx.flatMap((v) => src.slice(v * n, v * n + n));
    g.setAttribute('position', attr(T, flat(pos, 3), 3));
    g.setAttribute('normal', attr(T, flat(nor, 3), 3));
    g.setAttribute('uv', attr(T, flat(uv, 2), 2));
  }
  return g;
}

/// A strip along y, two bones: the upper half bends. `bindAt` places the mesh
/// before binding without a bind matrix (three.js then binds at its world
/// matrix) — with `p`, the mesh is moved after binding too.
function skinnedStrip(T, { bindAt = null } = {}) {
  const pos = [], skinIndex = [], skinWeight = [], idx = [];
  for (let j = 0; j <= 4; j++) {
    for (let i = 0; i <= 1; i++) {
      pos.push(i - 0.5, j * 0.5 - 1, 0);
      const w = Math.min(1, Math.max(0, (j - 1) / 2));
      skinIndex.push(0, 1, 0, 0);
      skinWeight.push(1 - w, w, 0, 0);
    }
  }
  for (let j = 0; j < 4; j++) { const a = j * 2; idx.push(a, a + 1, a + 3, a, a + 3, a + 2); }
  const g = new T.BufferGeometry();
  g.setAttribute('position', attr(T, pos, 3));
  g.setAttribute('skinIndex', attr(T, skinIndex, 4));
  g.setAttribute('skinWeight', attr(T, skinWeight, 4));
  index(T, g, idx);
  const mesh = named(new T.SkinnedMesh(g, material(T, 'mesh', { side: 2 })), 'skinned');
  const b0 = new T.Bone(), b1 = new T.Bone();
  b0.position.set(0, -1, 0);
  b1.position.set(0, 1, 0);
  b0.add(b1);
  mesh.add(b0);
  if (bindAt) {
    place(mesh, bindAt);
    mesh.bind(new T.Skeleton([b0, b1]));
  } else {
    const inverses = [new T.Matrix4().makeTranslation(0, 1, 0), new T.Matrix4().makeTranslation(0, 0, 0)];
    mesh.bind(new T.Skeleton([b0, b1], inverses), new T.Matrix4());
  }
  b1.rotation.set(0, 0.3, 0.6);
  return mesh;
}

const scenes = {
  sides(T) {
    const s = root(T);
    s.add(named(place(new T.Mesh(grid(T), material(T, 'mesh', { side: 0 })), { p: [0, 0, 0.5], r: [0.3, 0.2, 0] }), 'front'));
    s.add(named(place(new T.Mesh(grid(T), material(T, 'mesh', { side: 1 })), { p: [0, 0.2, -0.5], r: [-0.2, 0.4, 0.1] }), 'back'));
    s.add(named(place(new T.Mesh(grid(T, { indexed: false }), material(T, 'mesh', { side: 2 })), { p: [0.1, 0, 0], r: [0, -0.3, 0.5] }), 'double'));
    return s;
  },
  groups(T) {
    const s = root(T);
    const g = grid(T);
    g.addGroup(0, 6, 0);
    g.addGroup(6, 12, 1);
    g.addGroup(18, 18, 2);
    const mats = [material(T, 'mesh', { side: 2 }), material(T, 'mesh', { side: 1 }), material(T, 'mesh', { side: 0 })];
    s.add(named(place(new T.Mesh(g, mats), { r: [0.1, 0.5, 0] }), 'multi'));
    const ranged = grid(T);
    ranged.setDrawRange(3, 21);
    s.add(named(place(new T.Mesh(ranged, material(T, 'mesh', { side: 2 })), { p: [0, 0, -1] }), 'ranged'));
    return s;
  },
  morph(T) {
    const s = root(T);
    for (const [name, relative, z] of [['relative', true, 0.4], ['absolute', false, -0.4]]) {
      const g = grid(T, { indexed: false });
      const base = g.attributes.position.array;
      const target = Array.from(base, (v, i) => (relative ? (i % 3 === 2 ? 0.8 * Math.cos(i) : 0.1) : v + (i % 3 === 2 ? 0.6 : 0)));
      g.morphAttributes = { position: [attr(T, target, 3)] };
      g.morphTargetsRelative = relative;
      const m = place(new T.Mesh(g, material(T, 'mesh', { side: 2 })), { p: [0, 0, z], r: [0.2, 0, 0] });
      m.morphTargetInfluences[0] = 0.7;
      s.add(named(m, name));
    }
    return s;
  },
  nested(T) {
    const s = root(T);
    const outer = place(new T.Group(), { p: [0.3, -0.2, 0.1], r: [0.4, -0.2, 0.3], s: [1.5, 0.7, 1.2] });
    const child = named(place(new T.Mesh(grid(T), material(T, 'mesh', { side: 2 })), { p: [0, 0.5, 0.3], r: [0, 0.6, 0], s: [0.8, 1.3, 1] }), 'child');
    const grandchild = named(place(new T.Mesh(grid(T), material(T, 'mesh', { side: 0 })), { p: [0.2, 0, -1], r: [0.9, 0, 0], s: [0.5, 0.5, 2] }), 'grandchild');
    child.add(grandchild);
    outer.add(child);
    s.add(outer);
    return s;
  },
  /// Groups three deep, each rotated (in another Euler order) and scaled.
  deep(T) {
    const s = root(T);
    let parent = s;
    const orders = ['ZYX', 'YXZ', 'XZY'];
    for (let k = 0; k < 3; k++) {
      const g = place(new T.Group(), { p: [0.2 * k, -0.1, 0.15], r: [0.3 + 0.1 * k, -0.4, 0.2 * k], s: [1.2, 0.9 - 0.1 * k, 1.1] });
      g.rotation.order = orders[k];
      g.rotation.set(0.3 + 0.1 * k, -0.4, 0.2 * k);
      parent.add(g);
      parent = g;
    }
    parent.add(named(new T.Mesh(grid(T), material(T, 'mesh', { side: 2 })), 'deepest'));
    return s;
  },
  instanced(T) {
    const s = root(T);
    const im = named(place(new T.InstancedMesh(grid(T), material(T, 'mesh', { side: 2 }), 5), { p: [0, 0.1, 0], r: [0, 0.2, 0] }), 'instanced');
    for (let i = 0; i < 5; i++) {
      const m = new T.Matrix4().makeRotationY(0.4 * i);
      m.elements[12] = (i - 2) * 0.6; m.elements[13] = 0.3 * i - 0.6; m.elements[14] = 0.5 * (i % 3) - 0.5;
      im.setMatrixAt(i, m);
    }
    s.add(im);
    return s;
  },
  skinned(T) {
    const s = root(T);
    s.add(skinnedStrip(T));
    return s;
  },
  /// Bound where it stood (three.js: bindMatrix = matrixWorld), then moved.
  skinnedBound(T) {
    const s = root(T);
    const mesh = skinnedStrip(T, { bindAt: { p: [0.3, 0.2, -0.4], r: [0.1, 0.4, 0], s: [1.2, 1, 1] } });
    s.add(mesh);
    mesh.position.set(-0.2, 0.1, 0.3);
    return s;
  },
  lines(T) {
    const s = root(T);
    const pts = [];
    for (let i = 0; i < 8; i++) pts.push(Math.cos(i) * 1.2, Math.sin(i * 1.3), 0.3 * Math.sin(i * 0.7));
    const geo = () => { const g = new T.BufferGeometry(); g.setAttribute('position', attr(T, pts, 3)); return g; };
    s.add(named(place(new T.LineSegments(geo(), material(T, 'line')), { r: [0.2, 0, 0] }), 'segments'));
    const indexed = geo();
    index(T, indexed, [0, 3, 3, 5, 1, 6, 2, 7]);
    s.add(named(place(new T.LineSegments(indexed, material(T, 'line')), { p: [0, 0, 0.5], s: [1.4, 1.4, 1.4] }), 'indexed'));
    s.add(named(place(new T.Line(geo(), material(T, 'line')), { p: [0.2, 0.1, -0.5], r: [0, 0.4, 0] }), 'strip'));
    s.add(named(place(new T.LineLoop(geo(), material(T, 'line')), { p: [-0.2, -0.3, 0.2], s: [0.8, 0.8, 0.8] }), 'loop'));
    const ranged = geo();
    ranged.setDrawRange(2, 5);
    s.add(named(place(new T.Line(ranged, material(T, 'line')), { p: [0.1, 0.4, 0.3] }), 'rangedStrip'));
    return s;
  },
  points(T) {
    const s = root(T);
    const pts = [];
    for (let i = 0; i < 40; i++) pts.push(Math.sin(i * 1.7) * 1.4, Math.cos(i * 2.3), Math.sin(i * 0.9) * 0.8);
    const g = new T.BufferGeometry();
    g.setAttribute('position', attr(T, pts, 3));
    s.add(named(place(new T.Points(g, material(T, 'points')), { r: [0.3, 0.1, 0], s: [1.2, 1.2, 1.2] }), 'points'));
    const g2 = new T.BufferGeometry();
    g2.setAttribute('position', attr(T, pts, 3));
    index(T, g2, [3, 7, 11, 2, 30, 31, 5, 9]);
    g2.setDrawRange(1, 6);
    s.add(named(place(new T.Points(g2, material(T, 'points')), { p: [0, 0, 0.6] }), 'indexed'));
    return s;
  },
  sprites(T) {
    const s = root(T);
    const add = (name, opts, p, sc, center) => {
      const sp = named(place(new T.Sprite(material(T, 'sprite', opts)), { p, s: sc }), name);
      if (center) sp.center = new T.Vector2(...center);
      s.add(sp);
    };
    add('plain', {}, [0, 0, 0], [1, 1, 1]);
    add('rotated', { rotation: 0.7 }, [0.9, 0.4, -0.5], [1.5, 0.6, 1]);
    add('anchored', {}, [-0.8, -0.3, 0.4], [0.7, 0.9, 1], [0, 1]);
    add('fixed-size', { sizeAttenuation: false }, [0.2, -0.6, -1], [0.1, 0.1, 1]);
    return s;
  },
  builtin(T) {
    const s = root(T);
    s.add(named(place(new T.Mesh(new T.BoxGeometry(1.2, 0.8, 1.6), material(T, 'mesh')), { r: [0.3, 0.6, 0.1] }), 'box'));
    return s;
  },
  /// Invisible objects are hit (three.js tests layers only).
  hidden(T) {
    const s = scenes.sides(T);
    s.children[0].visible = false;
    const g = new T.Group(); g.visible = false;
    g.add(named(place(new T.Mesh(grid(T), material(T, 'mesh', { side: 2 })), { p: [0, 0, -1.2] }), 'underHidden'));
    s.add(g);
    return s;
  },
};

/// Deterministic rays from all around, aimed into the scene.
function rays(seed, n) {
  let a = seed >>> 0;
  const rand = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = [];
  for (let i = 0; i < n; i++) {
    const th = rand() * Math.PI * 2, ph = Math.acos(2 * rand() - 1), r = 4 + 2 * rand();
    const o = [r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph)];
    const t = [3 * rand() - 1.5, 2.4 * rand() - 1.2, 2 * rand() - 1];
    const d = [t[0] - o[0], t[1] - o[1], t[2] - o[2]];
    const l = Math.hypot(...d);
    out.push({ o, d: d.map((v) => v / l) });
  }
  return out;
}

const FIELDS = ['distance', 'point', 'object', 'face', 'faceIndex', 'uv', 'uv1', 'normal', 'instanceId', 'index', 'distanceToRay'];
const EXTENSIONS = ['polygonIndex', 'barycoord'];
/// See the header: 64 f32 epsilons, relative to the scene's scale (10).
const F32_EPS = 2 ** -24;
const TOL = 64 * F32_EPS;
const SCENE_SCALE = 10;

/// The largest error seen, in units of the bound (`RAYCAST_TOL_REPORT=1`
/// prints it): how much of the f32 budget the casts use.
let worst = 0;
process.on('exit', () => { if (process.env.RAYCAST_TOL_REPORT) console.log(`worst error: ${worst.toFixed(4)} of the bound`); });
/// `kappa` scales the bound for an ill-conditioned quantity (see `lineKappa`).
function near(a, b, what, kappa = 1) {
  if (a === null || a === undefined || typeof a === 'number') {
    if (typeof a === 'number' && typeof b === 'number') {
      worst = Math.max(worst, Math.abs(a - b) / (kappa * TOL * Math.max(SCENE_SCALE, Math.abs(a))));
      assert.ok(Math.abs(a - b) <= kappa * TOL * Math.max(SCENE_SCALE, Math.abs(a)), `${what}: ${a} vs ${b} (|Δ| ${Math.abs(a - b)}, κ ${kappa})`);
    } else {
      assert.equal(b, a, what);
    }
    return;
  }
  for (const k of ['x', 'y', 'z']) if (k in a) near(a[k], b?.[k], `${what}.${k}`, kappa);
}

/// The condition number of a line hit's closest point on its segment,
/// 1 / sin²θ for the angle θ between the ray and the segment: as the two turn
/// parallel the closest point slides along the segment for any rounding of
/// either, so its f32 error grows by this factor (`Ray.distanceSqToSegment`
/// divides by 1 − (d·s)²). Found from three.js's hit: the segment starting at
/// its `index`, in the order three.js's `Line.raycast` walks.
function lineKappa(hit, ray) {
  const obj = hit.object, g = obj.geometry;
  const pos = g.attributes.position.array, idx = g.index?.array;
  const count = idx ? idx.length : pos.length / 3;
  const at = (i) => (idx ? idx[i] : i);
  const start = Math.max(0, g.drawRange.start), end = Math.min(count, g.drawRange.start + g.drawRange.count);
  const step = obj.isLineSegments ? 2 : 1;
  let b = -1;
  for (let i = start; i < end - 1; i += step) if (at(i) === hit.index) { b = at(i + 1); break; }
  if (b < 0 && obj.isLineLoop && at(end - 1) === hit.index) b = at(start);
  if (b < 0) return 1;
  const w = (v) => new THREE.Vector3().fromArray(pos, v * 3).applyMatrix4(obj.matrixWorld);
  const s = w(b).sub(w(hit.index)).normalize();
  const c = s.dot(ray.direction);
  return 1 / Math.max(1 - c * c, 1e-6);
}

/// Which hit a record is, whatever its distance.
const hitKey = (h) => `${h.object.name}|${h.faceIndex ?? ''}|${h.index ?? ''}|${h.instanceId ?? ''}|${h.face?.materialIndex ?? ''}`;
/// `shimHits` with each run of hits three.js puts at the same distance (to
/// the f32 tolerance) in three.js's order. three.js sorts stably, so equal
/// distances — two segments meeting where the ray passes, a ray through a
/// shared edge — keep the order they were found in; in f32 the two distances
/// differ in the last bits, either way round, so the order within such a run
/// is not something f32 can reproduce, and is matched by identity instead.
function alignTies(threeHits, shimHits) {
  const out = shimHits.slice();
  for (let i = 0; i < threeHits.length;) {
    let j = i + 1;
    const d = threeHits[i].distance;
    while (j < threeHits.length && Math.abs(threeHits[j].distance - d) <= TOL * Math.max(SCENE_SCALE, Math.abs(d))) j++;
    if (j - i > 1) {
      const run = out.slice(i, j);
      threeHits.slice(i, j).forEach((t, k) => {
        const at = run.findIndex((s) => s && hitKey(s) === hitKey(t));
        if (at >= 0) { out[i + k] = run[at]; run[at] = null; }
      });
    }
    i = j;
  }
  return out;
}

/// three.js's hit list against the shim's, for one cast.
function compare(threeHits, shimHits, what, ray = null) {
  assert.equal(shimHits.length, threeHits.length,
    `${what}: ${shimHits.length} hits, three.js ${threeHits.length}\n` +
    `  three: ${threeHits.map((h) => `${h.object.name}@${h.distance.toFixed(6)}`).join(' ')}\n` +
    `  shim:  ${shimHits.map((h) => `${h.object.name}@${h.distance.toFixed(6)}`).join(' ')}`);
  shimHits = alignTies(threeHits, shimHits);
  threeHits.forEach((t, i) => {
    const s = shimHits[i], at = `${what} hit ${i} (${t.object.name})`;
    const keys = Object.keys(t).filter((k) => FIELDS.includes(k)).sort();
    assert.deepEqual(Object.keys(s).filter((k) => !EXTENSIONS.includes(k)).sort(), keys, `${at}: fields`);
    assert.equal(s.object.name, t.object.name, `${at}: object`);
    const kappa = t.object.isLine && ray ? lineKappa(t, ray) : 1;
    near(t.distance, s.distance, `${at}.distance`, kappa);
    near(t.point, s.point, `${at}.point`, kappa);
    for (const k of ['faceIndex', 'instanceId', 'index']) if (k in t) assert.equal(s[k], t[k], `${at}.${k}`);
    for (const k of ['uv', 'uv1', 'normal', 'distanceToRay']) if (k in t) near(t[k], s[k], `${at}.${k}`);
    if (t.face) {
      for (const k of ['a', 'b', 'c', 'materialIndex']) assert.equal(s.face[k], t.face[k], `${at}.face.${k}`);
      near(t.face.normal, s.face.normal, `${at}.face.normal`);
    } else {
      assert.equal(s.face, t.face, `${at}.face`);
    }
  });
}

/// Cast every ray at both builds of a scene; returns how many hits matched.
function castBoth(name, { seed = 1, count = 300, setup = () => {}, recursive = true } = {}) {
  const t = scenes[name](THREE), s = scenes[name](S);
  // three.js casts against matrixWorld as it stands, so both sides update.
  t.updateMatrixWorld(true); s.updateMatrixWorld(true);
  const tr = new THREE.Raycaster(), sr = new S.Raycaster();
  setup(tr, t); setup(sr, s);
  let hits = 0;
  for (const [i, { o, d }] of rays(seed, count).entries()) {
    tr.set(new THREE.Vector3(...o), new THREE.Vector3(...d));
    sr.set(new S.Vector3(...o), new S.Vector3(...d));
    const th = tr.intersectObject(t, recursive), sh = sr.intersectObject(s, recursive);
    compare(th, sh, `${name} ray ${i}`, tr.ray);
    hits += th.length;
  }
  return hits;
}

describe('shim Raycaster (Rust) matches three.js r165', { skip }, () => {
  for (const name of ['sides', 'groups', 'morph', 'nested', 'deep', 'instanced', 'skinned', 'skinnedBound', 'builtin', 'hidden']) {
    it(`${name}: identical hit lists`, () => {
      assert.ok(castBoth(name) > 20, 'the rays should hit something');
    });
  }
  it('mesh hits carry the threers extensions', () => {
    const s = scenes.sides(S);
    const sr = new S.Raycaster(new S.Vector3(0.2, 0.1, 5), new S.Vector3(0, 0, -1));
    const hits = sr.intersectObject(s);
    assert.ok(hits.length >= 2);
    for (const h of hits) {
      assert.equal(h.polygonIndex, null, 'no polygon grouping');
      const w = h.barycoord;
      assert.ok(Math.abs(w.x + w.y + w.z - 1) < 8 * F32_EPS);
      // The barycentric weights reproduce the interpolated uv.
      const uv = h.object.geometry.attributes.uv.array;
      const u = uv[h.face.a * 2] * w.x + uv[h.face.b * 2] * w.y + uv[h.face.c * 2] * w.z;
      assert.ok(Math.abs(u - h.uv.x) < 8 * F32_EPS);
    }
  });
  it('bounds follow geometry edited in place once flagged needsUpdate', () => {
    const g = grid(S);
    const mesh = new S.Mesh(g, material(S, 'mesh', { side: 2 }));
    const r = new S.Raycaster(new S.Vector3(0.2, 0.1, 5), new S.Vector3(0, 0, -1));
    assert.equal(r.intersectObject(mesh).length, 1);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.array.length; i += 3) pos.array[i] += 10;
    pos.needsUpdate = true;
    assert.equal(r.intersectObject(mesh).length, 0, 'moved away');
    r.set(new S.Vector3(10.2, 0.1, 5), new S.Vector3(0, 0, -1));
    assert.equal(r.intersectObject(mesh).length, 1, 'found where it went');
  });
  it('lines: identical at the default and a tight threshold', () => {
    assert.ok(castBoth('lines') > 50);
    assert.ok(castBoth('lines', { seed: 7, setup: (r) => { r.params.Line.threshold = 0.08; } }) > 5);
  });
  it('points: identical at the default and a tight threshold', () => {
    assert.ok(castBoth('points') > 50);
    assert.ok(castBoth('points', { seed: 9, setup: (r) => { r.params.Points.threshold = 0.15; } }) > 5);
  });
  it('near / far, layers and recursive = false', () => {
    castBoth('nested', { seed: 3, setup: (r) => { r.near = 4.2; r.far = 5.5; } });
    castBoth('sides', { seed: 4, setup: (r, scene) => {
      scene.children[1].layers.set(3);
      r.layers.set(3);
    } });
    assert.equal(castBoth('nested', { seed: 5, recursive: false }), 0, 'the root has no geometry');
  });
  for (const cam of ['perspective', 'orthographic']) {
    it(`setFromCamera (${cam}): same ray, same sprite hits`, () => {
      const make = (T) => {
        const c = cam === 'perspective'
          ? new T.PerspectiveCamera(55, 1.6, 0.5, 40)
          : new T.OrthographicCamera(-3, 3, 2, -2, 0.1, 40);
        if (cam === 'orthographic') { c.zoom = 1.3; c.updateProjectionMatrix(); }
        c.position.set(1.5, 1.1, 5);
        c.lookAt(new T.Vector3(0.1, -0.2, 0));
        c.updateMatrixWorld();
        return c;
      };
      const tc = make(THREE), sc = make(S);
      const t = scenes.sprites(THREE), s = scenes.sprites(S);
      t.updateMatrixWorld(true); s.updateMatrixWorld(true);
      const tr = new THREE.Raycaster(), sr = new S.Raycaster();
      let hits = 0;
      for (let i = 0; i < 400; i++) {
        const ndc = [((i * 37) % 41) / 20 - 1, ((i * 13) % 29) / 14 - 1];
        tr.setFromCamera(new THREE.Vector2(...ndc), tc);
        sr.setFromCamera(new S.Vector2(...ndc), sc);
        near(tr.ray.origin, sr.ray.origin, `ray origin @${ndc}`);
        near(tr.ray.direction, sr.ray.direction, `ray direction @${ndc}`);
        assert.equal(sr.camera, sc);
        const th = tr.intersectObject(t), sh = sr.intersectObject(s);
        compare(th, sh, `sprites @${ndc}`);
        hits += th.length;
      }
      assert.ok(hits > 10, `${hits} sprite hits`);
    });
  }
  it('an object outside any Scene is cast (mirrored on demand)', () => {
    const t = scenes.nested(THREE).children[0], s = scenes.nested(S).children[0];
    t.removeFromParent(); s.removeFromParent();
    t.updateMatrixWorld(true); s.updateMatrixWorld(true);
    assert.equal(s.parent, null);
    let n = 0;
    for (const { o, d } of rays(21, 200)) {
      const tr = new THREE.Raycaster(new THREE.Vector3(...o), new THREE.Vector3(...d));
      const sr = new S.Raycaster(new S.Vector3(...o), new S.Vector3(...d));
      const th = tr.intersectObject(t), sh = sr.intersectObject(s);
      compare(th, sh, 'detached');
      n += th.length;
    }
    assert.ok(n > 10);
  });
  it('an object\'s own raycast is called, and false stops the descent (three.js)', () => {
    for (const T of [THREE, S]) {
      const s = scenes.nested(T);
      const outer = s.children[0], child = outer.children[0];
      const calls = [];
      outer.raycast = function (raycaster, intersects) {
        calls.push('outer');
        intersects.push({ distance: 0.5, point: new T.Vector3(), object: this, custom: true });
      };
      const r = new T.Raycaster(new T.Vector3(0.3, 0.2, 5), new T.Vector3(0, 0, -1));
      const hits = r.intersectObject(s);
      assert.deepEqual(calls, ['outer']);
      assert.equal(hits[0].custom, true, 'sorted in with the rest');
      assert.ok(hits.some((h) => h.object === child), 'children still cast');
      child.raycast = () => false;
      const after = r.intersectObject(s);
      assert.ok(!after.some((h) => h.object.name === 'grandchild'), 'false stops the descent');
    }
  });
});

// ---- three-mesh-bvh's acceleratedRaycast over the wasm BVH ----

describe('mesh-bvh acceleratedRaycast', { skip: skip || (!BVH && 'no mesh-bvh in this wasm build') }, () => {
  const scene = () => {
    const s = scenes.sides(S);
    scenes.groups(S).children.slice().forEach((c) => s.add(c));
    // three.js casts against matrixWorld as it stands.
    s.updateMatrixWorld(true);
    return s;
  };
  const cast = (s, { firstHitOnly = false, accelerated = false } = {}) => {
    const r = new S.Raycaster();
    r.firstHitOnly = firstHitOnly;
    const out = [];
    for (const { o, d } of rays(11, 300)) {
      r.set(new S.Vector3(...o), new S.Vector3(...d));
      const hits = [];
      for (const m of s.children) (accelerated ? BVH.acceleratedRaycast : S.Mesh.prototype.raycast).call(m, r, hits);
      hits.sort((a, b) => a.distance - b.distance);
      out.push(hits);
    }
    return out;
  };
  const withTrees = (s) => {
    for (const m of s.children) m.geometry.boundsTree = new BVH.MeshBVH(m.geometry, { maxLeafTris: 2 });
    return s;
  };
  it('reports the plain Mesh.raycast hits', () => {
    const s = scene();
    const plain = cast(s);
    withTrees(s);
    const fast = cast(s, { accelerated: true });
    let n = 0;
    plain.forEach((p, i) => {
      assert.equal(fast[i].length, p.length, `ray ${i}`);
      p.forEach((h, j) => {
        const f = fast[i][j];
        assert.equal(f.object, h.object);
        assert.equal(f.faceIndex, h.faceIndex);
        assert.equal(f.face.materialIndex, h.face.materialIndex);
        assert.equal(f.polygonIndex, h.polygonIndex);
        near(h.distance, f.distance, `ray ${i} hit ${j} distance`);
        near(h.uv, f.uv, `ray ${i} hit ${j} uv`);
        n++;
      });
    });
    assert.ok(n > 100, `${n} hits`);
  });
  it('firstHitOnly: each BVH mesh reports its nearest hit only (three-mesh-bvh)', () => {
    const s = scene();
    const plain = cast(s);
    withTrees(s);
    const first = cast(s, { firstHitOnly: true, accelerated: true });
    plain.forEach((p, i) => {
      const nearest = s.children.map((m) => p.find((h) => h.object === m)).filter(Boolean)
        .sort((a, b) => a.distance - b.distance);
      assert.deepEqual(first[i].map((h) => [h.object.name, h.faceIndex]),
        nearest.map((h) => [h.object.name, h.faceIndex]), `ray ${i}`);
    });
  });
  it('a mesh without a boundsTree keeps every hit', () => {
    const s = scenes.sides(S);
    s.updateMatrixWorld(true);
    const r = new S.Raycaster(new S.Vector3(0.1, 0.1, 5), new S.Vector3(0, 0, -1));
    r.firstHitOnly = true;
    const plain = [], fast = [];
    for (const m of s.children) { S.Mesh.prototype.raycast.call(m, r, plain); BVH.acceleratedRaycast.call(m, r, fast); }
    assert.equal(fast.length, plain.length);
    assert.ok(plain.length >= 2);
  });
});
