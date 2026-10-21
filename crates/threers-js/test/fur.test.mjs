// The shim's `Fur` (a threers extension): it mirrors into the wasm scene as a
// fur object — not the empty group a missing binding falls back to — with
// every option passed through, and the scene's fur steps without error.
//
// Runs over a real wasm build (`THREERS_PKG_DIR`, else web/pkg); skips when
// that build predates `WebFurMaterial`.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

async function loadShimOver(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  if (!m.WebFurMaterial) return null;
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-fur-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const skip = !S && 'wasm build missing or older than WebFurMaterial';

describe('shim Fur', { skip }, () => {
  it('mirrors as fur, with its options, and steps', () => {
    const scene = new S.Scene();
    const fur = new S.Fur(new S.SphereGeometry(1, 24, 12), {
      length: 0.2,
      density: 60,
      rootColor: 0x221100,
      tipColor: 'oklch(80% 0.1 70)',
      gravity: new S.Vector3(0, -0.5, 0),
      wind: [0.4, 0, 0],
      flutter: 0.1,
      stiffness: 50,
      placement: 'surface',
      cell: 0.04,
    });
    scene.add(fur);
    scene._syncNow();
    assert.equal(fur._m.kind, 'fur');
    assert.equal(fur.castShadow, true, 'fur casts by default');
    assert.equal(fur.receiveShadow, true, 'and receives');
    assert.equal(new S.Fur(new S.SphereGeometry(1, 8, 4), { shadowHeight: 0 }).castShadow, false);
    assert.ok(S.default.Fur === S.Fur, 'THREE.Fur');
    // Animate: move it and step, as the renderer does each frame.
    for (let i = 1; i <= 5; i++) {
      fur.position.x = i * 0.1;
      scene._syncNow();
      scene._w.stepFur(i / 60, 1 / 60);
    }
  });

  it('Hair mirrors as hair and steps', () => {
    const scene = new S.Scene();
    const hair = new S.Hair(new S.SphereGeometry(1, 24, 12), {
      count: 500,
      segments: 6,
      width: 0.01,
      length: 0.6,
      gravity: [0, -1.5, 0],
      curl: 0.4,
      clumping: 0.5,
      wind: [0.3, 0, 0],
    });
    scene.add(hair);
    scene._syncNow();
    assert.equal(hair._m.kind, 'hair');
    assert.ok(S.default.Hair === S.Hair, 'THREE.Hair');
    assert.ok(hair.isFur && hair.isHair);
    assert.equal(hair.castShadow, true, 'hair casts by default');
    assert.equal(new S.Hair(new S.SphereGeometry(1, 8, 4), { shadowStrands: 0 }).castShadow, false);
    for (let i = 1; i <= 3; i++) {
      hair.position.y = i * 0.1;
      scene._syncNow();
      scene._w.stepFur(i / 60, 1 / 60);
    }
  });

  it('Hair on a SkinnedMesh grows on it, as its child, and follows its bone', () => {
    const scene = new S.Scene();
    const geometry = new S.SphereGeometry(0.6, 24, 12);
    const n = geometry.attributes.position.count;
    geometry.setAttribute('skinIndex', new S.Uint16BufferAttribute(new Uint16Array(n * 4), 4));
    const weights = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) weights[i * 4] = 1;
    geometry.setAttribute('skinWeight', new S.Float32BufferAttribute(weights, 4));
    const mesh = new S.SkinnedMesh(geometry, new S.MeshBasicMaterial());
    const bone = new S.Bone();
    mesh.add(bone);
    mesh.bind(new S.Skeleton([bone]));
    scene.add(mesh);
    const hair = new S.Hair(mesh, { count: 400, segments: 5, length: 0.3, stiffness: 30 });
    assert.equal(hair.parent, mesh, 'added as the mesh\'s child');
    assert.equal(hair.geometry, null);
    // Grown on the mesh's own node — not the group a missing rig falls back to.
    const proto = Object.getPrototypeOf(scene._w);
    const create = proto.createSkinnedHair;
    const calls = [];
    proto.createSkinnedHair = function (skinned, ...rest) {
      const h = create.call(this, skinned, ...rest);
      calls.push({ skinned: skinned.id, hair: h.id });
      return h;
    };
    try {
      scene._syncNow();
    } finally {
      proto.createSkinnedHair = create;
    }
    assert.equal(hair._m.kind, 'hair');
    assert.deepEqual(calls, [{ skinned: mesh._m.id, hair: hair._m.id }]);
    // Animated with the bone, without error.
    for (let i = 1; i <= 5; i++) {
      bone.position.x = i * 0.1;
      scene.updateMatrixWorld(true);
      scene._syncNow();
      scene._w.stepFur(i / 60, 1 / 60);
    }
  });

  it('takes lod, shadowHeight and a shader material\'s alphaToCoverage', () => {
    const scene = new S.Scene();
    const fur = new S.Fur(new S.SphereGeometry(1, 16, 8), { lod: false, shadowHeight: 0.4 });
    scene.add(fur);
    scene._syncNow();
    assert.equal(fur._m.kind, 'fur');
    const m = new S.ShaderMaterial({
      fragment: '@fragment fn fs_main(in : VsOut) -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, coverage_alpha(1.0)); }',
      alphaToCoverage: true,
    });
    const mesh = new S.Mesh(new S.BoxGeometry(1, 1, 1), m);
    scene.add(mesh);
    scene._syncNow();
    assert.equal(mesh._m.kind, 'mesh');
  });

  it('without a geometry it is an empty group, not an error', () => {
    const scene = new S.Scene();
    const fur = new S.Fur(null);
    scene.add(fur);
    scene._syncNow();
    assert.equal(fur._m.kind, 'group');
  });
});
