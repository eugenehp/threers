// A texture changed after it was bound into a material reaches that material,
// as in three.js: `texture.colorSpace = SRGBColorSpace` set after binding,
// `needsUpdate = true`, an image that arrives later (TextureLoader swaps the
// wasm texture), in any map slot. The shim runs as a module over a real wasm
// build (`THREERS_PKG_DIR`, else web/pkg); the wasm calls are observed by
// wrapping the material's and texture's methods.

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
  if (typeof m.WebMaterial?.prototype?.setPhysical !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-tex-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const skip = !S && `no wasm build with WebMaterial.setPhysical in ${pkgDir}`;
// The three.js constants are on the shim's default `THREE` object.
const SRGB = S?.default?.SRGBColorSpace;

/// Record the calls `obj[name]` receives, still making them.
function spy(obj, name) {
  const calls = [];
  const orig = obj[name].bind(obj);
  obj[name] = (...a) => { calls.push(a); return orig(...a); };
  return calls;
}
const texel = () => new S.DataTexture(new Uint8Array([200, 100, 50, 255]), 1, 1);

describe('textures re-bind into the materials holding them', { skip }, () => {
  it('colorSpace set after binding decodes the map, and re-binds it', () => {
    const tex = texel();
    const mat = new S.MeshLambertMaterial({ map: tex });
    const binds = spy(mat._w, 'setMapData');
    const srgb = spy(tex._w, 'setSrgb');
    tex.colorSpace = SRGB;
    assert.deepEqual(srgb.at(-1), [true], 'the texture is switched to sRGB');
    assert.equal(binds.length, 1, 'and bound into the material again');
    tex.colorSpace = SRGB;
    assert.equal(binds.length, 1, 'setting the same colour space again changes nothing');
  });

  it('needsUpdate re-binds in every material that holds the texture', () => {
    const tex = texel();
    const a = new S.MeshBasicMaterial({ map: tex });
    const b = new S.MeshStandardMaterial({ map: tex });
    const ca = spy(a._w, 'setMapData');
    const cb = spy(b._w, 'setMapData');
    tex.needsUpdate = true;
    assert.equal(ca.length, 1);
    assert.equal(cb.length, 1);
    assert.equal(tex.version, 1, 'version counts needsUpdate, as in three.js');
  });

  it('an image that arrives later reaches the material (the loader swaps the wasm texture)', () => {
    const tex = new S.Texture();
    const mat = new S.MeshPhongMaterial({ map: tex });
    const binds = spy(mat._w, 'setMapData');
    tex._w = new S.DataTexture(new Uint8Array([10, 20, 30, 255]), 1, 1)._w;
    tex.needsUpdate = true;
    assert.equal(binds.length, 1);
    assert.equal(binds[0][0], tex._w, 'the new wasm texture is the one bound');
  });

  it('a material that moved on to another texture is left alone', () => {
    const first = texel();
    const mat = new S.MeshToonMaterial({ map: first });
    mat.map = texel();
    const binds = spy(mat._w, 'setMapData');
    first.needsUpdate = true;
    assert.equal(binds.length, 0);
  });

  it('every map slot re-binds, not only the colour map', () => {
    const ao = texel();
    const mat = new S.MeshPhongMaterial({ aoMap: ao });
    const binds = spy(mat._w, 'setAoMapData');
    ao.needsUpdate = true;
    assert.equal(binds.length, 1);
  });
});

const round = (rows) => rows.map((v) => Math.round(v * 1e6) / 1e6 + 0);

describe('a texture carries three.js\'s whole uv matrix and sampler state', { skip }, () => {
  it('rotation and center compose as Matrix3.setUvTransform, and re-bind', () => {
    const tex = texel();
    const mat = new S.MeshLambertMaterial({ map: tex });
    const binds = spy(mat._w, 'setMapData');
    const pushed = spy(tex._w, 'setUvMatrix');
    tex.rotation = Math.PI / 2;
    tex.center.set(0.5, 0.5);
    tex.needsUpdate = true;
    const want = new S.Matrix3().setUvTransform(0, 0, 1, 1, Math.PI / 2, 0.5, 0.5).elements;
    assert.deepEqual(round(pushed.at(-1)), round([want[0], want[3], want[6], want[1], want[4], want[7]]));
    assert.equal(binds.length, 1);
  });

  it('a hand-set matrix is used as it is with matrixAutoUpdate off', () => {
    const tex = texel();
    const pushed = spy(tex._w, 'setUvMatrix');
    tex.matrixAutoUpdate = false;
    tex.matrix.set(2, 0, 0.5, 0, 3, 0.25, 0, 0, 1);
    tex.needsUpdate = true;
    assert.deepEqual(round(pushed.at(-1)), [2, 0, 0.5, 0, 3, 0.25]);
  });

  it('a plain { x, y } becomes a Vector2', () => {
    const tex = texel();
    tex.repeat = { x: 4, y: 2 };
    assert.ok(tex.repeat instanceof S.Vector2);
    assert.deepEqual([tex.repeat.x, tex.repeat.y], [4, 2]);
  });

  it('a canvas texture reads its top-first pixels the right way up through flipY', () => {
    const tex = new S.CanvasTexture(null);
    assert.deepEqual([tex.offset.x, tex.offset.y, tex.repeat.x, tex.repeat.y], [0, 0, 1, 1],
      'three.js\'s defaults, not a flip folded into them');
    assert.deepEqual(round(tex._uvSynced.rows), [1, 0, 0, 0, -1, 1]);
  });

  it('filter constants are three.js r165\'s, and needsUpdate applies them', () => {
    const T = S.default;
    assert.equal(T.NearestMipmapLinearFilter, 1005);
    assert.equal(T.LinearMipmapNearestFilter, 1007);
    const tex = texel();
    const filters = spy(tex._w, 'setFilters');
    tex.magFilter = T.LinearFilter;
    tex.minFilter = T.NearestMipmapLinearFilter;
    tex.wrapS = T.MirroredRepeatWrapping;
    tex.wrapT = T.RepeatWrapping;
    tex.needsUpdate = true;
    assert.deepEqual(filters.at(-1), [1006, 1005, 1002, 1000], 'three.js\'s own constants reach wasm');
    tex.minFilter = T.LinearMipmapNearestFilter;
    tex.needsUpdate = true;
    assert.deepEqual(filters.at(-1), [1006, 1007, 1002, 1000], 'LinearMipmapNearest, unfolded');
  });
});

describe('alphaMap, and emptying a slot', { skip }, () => {
  it('alphaMap binds on every material three.js gives one', () => {
    for (const Kind of [S.MeshBasicMaterial, S.MeshLambertMaterial, S.MeshPhongMaterial,
      S.MeshToonMaterial, S.MeshStandardMaterial, S.MeshPhysicalMaterial,
      S.MeshMatcapMaterial, S.SpriteMaterial]) {
      const mat = new Kind();
      const binds = spy(mat._w, 'setAlphaMapData');
      mat.alphaMap = texel();
      assert.equal(binds.length, 1, Kind.name);
    }
  });

  it('a slot set to null is emptied, as in three.js', () => {
    const mat = new S.MeshStandardMaterial({ map: texel(), alphaMap: texel() });
    const cleared = spy(mat._w, 'clearMap');
    mat.alphaMap = null;
    mat.map = null;
    assert.deepEqual(cleared.map((c) => c[0]), ['alphaMap', 'map']);
    assert.equal(mat.map, null);
    mat.map = null;
    assert.equal(cleared.length, 2, 'an empty slot set to null again is left alone');
  });
});

describe('lightMap, bumpMap and specularMap', { skip }, () => {
  it('bind, with their scalars live, on the materials three.js gives them', () => {
    const lambert = new S.MeshLambertMaterial({ lightMapIntensity: 2 });
    const light = spy(lambert._w, 'setLightMapData');
    const bump = spy(lambert._w, 'setBumpMapData');
    const spec = spy(lambert._w, 'setSpecularMapData');
    const scale = spy(lambert._w, 'setBumpScale');
    lambert.lightMap = texel();
    lambert.bumpMap = texel();
    lambert.specularMap = texel();
    lambert.bumpScale = 3;
    assert.deepEqual([light.length, bump.length, spec.length], [1, 1, 1]);
    assert.deepEqual(scale.at(-1), [3]);
    assert.equal(lambert.lightMapIntensity, 2);
    assert.equal('specularMap' in new S.MeshToonMaterial(), false, 'toon has no specularMap in three.js');
    assert.equal('bumpMap' in new S.MeshBasicMaterial(), false, 'nor basic a bumpMap');
    const std = new S.MeshStandardMaterial();
    const intensity = spy(std._w, 'setLightMapIntensity');
    std.lightMapIntensity = 0.5;
    assert.deepEqual(intensity.at(-1), [0.5]);
  });
});

describe('render targets carry three.js\'s configuration', { skip }, () => {
  it('colorSpace from the options, kept by clone', () => {
    const rt = new S.WebGLRenderTarget(4, 4, { colorSpace: SRGB });
    assert.equal(rt.texture.colorSpace, SRGB);
    assert.equal(rt.clone().texture.colorSpace, SRGB);
    assert.equal(new S.WebGLRenderTarget(4, 4).texture.colorSpace, '', 'NoColorSpace by default');
  });

  it('setSize resizes the wasm target in place, keeping its id', () => {
    const rt = new S.WebGLRenderTarget(4, 4);
    const calls = [];
    rt._w = { id: 7, setSize: (w, h) => calls.push([w, h]) };
    rt.setSize(8.7, 2);
    assert.deepEqual(calls, [[8, 2]]);
    assert.equal(rt._w.id, 7, 'the same target, so its texture follows');
  });

  it('read-back follows the type and the colour space', async () => {
    const r = Object.create(S.WebGLRenderer.prototype);
    const seen = [];
    r._w = {
      readRenderTargetPixels: async () => new Uint8Array([1, 2, 3, 4]),
      readRenderTargetF16: async () => new Uint8Array([0, 60, 0, 56]),
    };
    const byte = new S.WebGLRenderTarget(1, 1);
    byte._w = { id: 1, setSrgbColorSpace: (v) => seen.push(v) };
    await r.readRenderTargetPixels(byte, 0, 0, 1, 1, new Uint8Array(4));
    byte.texture.colorSpace = SRGB;
    await r.readRenderTargetPixels(byte, 0, 0, 1, 1, new Uint8Array(4));
    assert.deepEqual(seen, [false, true], 'linear for NoColorSpace, encoded for SRGBColorSpace');
    const half = new S.WebGLRenderTarget(1, 1, { type: S.default.HalfFloatType });
    half._w = { id: 2 };
    const out = new Uint16Array(2);
    await r.readRenderTargetPixels(half, 0, 0, 1, 1, out);
    assert.deepEqual([...out], [0x3c00, 0x3800], 'half floats, 1.0 and 0.5');
  });
});

describe('envMap and CubeTexture', { skip }, () => {
  const face = (v) => ({ data: new Uint8Array([v, v, v, 255]), width: 1, height: 1 });
  const faces = () => [0, 1, 2, 3, 4, 5].map((i) => face(i * 40));

  it('a CubeTexture uploads its six images, and again on needsUpdate', () => {
    const tex = new S.CubeTexture();
    assert.equal(tex._w, null, 'nothing to upload yet');
    tex.images = faces();
    tex.needsUpdate = true;
    assert.ok(tex._w, 'uploaded once all six are there');
    assert.equal(tex._w.size, 1);
  });

  it('envMap binds on Basic, Lambert and Phong, and follows a late upload', () => {
    for (const Kind of [S.MeshBasicMaterial, S.MeshLambertMaterial, S.MeshPhongMaterial]) {
      const tex = new S.CubeTexture();
      const mat = new Kind({ envMap: tex, combine: S.default.MixOperation, reflectivity: 0.5 });
      const binds = spy(mat._w, 'setEnvMap');
      tex.images = faces();
      tex.needsUpdate = true;
      assert.equal(binds.length, 1, `${Kind.name}: bound when the images arrive`);
      assert.equal(mat.combine, 1);
      assert.equal(mat.reflectivity, 0.5);
    }
    assert.equal('envMap' in new S.MeshToonMaterial(), false, 'toon has no envMap in three.js');
  });

  it('mapping, envMapRotation and null reach wasm', () => {
    const tex = new S.CubeTexture(faces());
    const mat = new S.MeshPhongMaterial({ envMap: tex });
    const refract = spy(tex._w, 'setRefraction');
    tex.mapping = S.default.CubeRefractionMapping;
    assert.deepEqual(refract.at(-1), [true]);
    const rot = spy(mat._w, 'setEnvMapRotation');
    mat.envMapRotation.y = 1.5;
    _syncFor(mat);
    assert.deepEqual(rot.at(-1), [0, 1.5, 0, 'XYZ']);
    const cleared = spy(mat._w, 'clearMap');
    mat.envMap = null;
    assert.deepEqual(cleared.at(-1), ['envMap']);
  });

  it('scene.environment = null clears a cube environment', () => {
    const scene = new S.Scene();
    scene.environment = new S.CubeTexture(faces());
    const clear = spy(scene._w, 'clearEnvironment');
    scene.environment = null;
    assert.equal(clear.length, 1);
  });
});

/// What a render does to one material before drawing: three.js reads
/// `envMapRotation` each frame, so the shim pushes it then.
function _syncFor(mat) {
  const scene = new S.Scene();
  scene.add(new S.Mesh(new S.PlaneGeometry(1, 1), mat));
  const r = Object.create(S.WebGLRenderer.prototype);
  r._prepareRender?.call({ _syncClearBackground() {}, _inRender: false }, scene, null);
}

describe('environment on Standard and Physical, and on the scene', { skip }, () => {
  const face = (v) => ({ data: new Uint8Array([v, v, v, 255]), width: 1, height: 1 });
  const cube = () => new S.CubeTexture([0, 1, 2, 3, 4, 5].map((i) => face(i * 40)));

  it('envMap, envMapIntensity and envMapRotation reach wasm', () => {
    for (const Kind of [S.MeshStandardMaterial, S.MeshPhysicalMaterial]) {
      const env = cube();
      const mat = new Kind({ envMap: env, envMapIntensity: 0.5 });
      const binds = spy(mat._w, 'setEnvMap');
      const intensity = spy(mat._w, 'setEnvMapIntensity');
      env.needsUpdate = true;
      assert.equal(binds.length, 1, `${Kind.name}: re-bound when the cube changes`);
      assert.equal(mat.envMapIntensity, 0.5);
      mat.envMapIntensity = 2;
      assert.deepEqual(intensity.at(-1), [2]);
      const rot = spy(mat._w, 'setEnvMapRotation');
      mat.envMapRotation.x = 0.25;
      _syncFor(mat);
      assert.deepEqual(rot.at(-1), [0.25, 0, 0, 'XYZ']);
    }
  });

  it('scene.environmentIntensity and environmentRotation reach wasm', () => {
    const scene = new S.Scene();
    const intensity = spy(scene._w, 'setEnvironmentIntensity');
    scene.environmentIntensity = 0.3;
    assert.deepEqual(intensity.at(-1), [0.3]);
    const rot = spy(scene._w, 'setEnvironmentRotation');
    scene.environmentRotation.y = 1;
    scene._syncEnvironmentRotation();
    assert.deepEqual(rot.at(-1), [0, 1, 0, 'XYZ']);
  });
});

describe('MeshPhysicalMaterial\'s own properties and maps', { skip }, () => {
  it('takes three.js\'s options and defaults, and every scalar is live', () => {
    const mat = new S.MeshPhysicalMaterial({ ior: 2.0, sheen: 1, iridescenceThicknessRange: [50, 300] });
    assert.equal(mat.ior, 2.0);
    assert.equal(mat.specularIntensity, 1);
    assert.deepEqual(mat.iridescenceThicknessRange, [50, 300]);
    const pushed = spy(mat._w, 'setPhysical');
    mat.anisotropy = 0.5;
    const last = pushed.at(-1);
    assert.equal(last[10], 0.5, 'anisotropy');
    assert.equal(last[2], 2.0, 'ior travels with it');
    assert.deepEqual([last[8], last[9]], [50, 300], 'the thickness range');
  });

  it('reflectivity is three.js\'s other view of ior', () => {
    const mat = new S.MeshPhysicalMaterial({ ior: 1.5 });
    assert.ok(Math.abs(mat.reflectivity - 0.5) < 1e-9);
    mat.reflectivity = 1;
    assert.ok(Math.abs(mat.ior - (1.4 / 0.6)) < 1e-9);
  });

  it('every physical map binds by its three.js name, and null empties it', () => {
    const mat = new S.MeshPhysicalMaterial();
    const named = spy(mat._w, 'setNamedMapData');
    const cleared = spy(mat._w, 'clearMap');
    for (const key of ['specularIntensityMap', 'specularColorMap', 'clearcoatMap', 'clearcoatRoughnessMap',
      'clearcoatNormalMap', 'iridescenceMap', 'iridescenceThicknessMap', 'sheenColorMap', 'sheenRoughnessMap',
      'transmissionMap', 'thicknessMap', 'anisotropyMap']) {
      mat[key] = texel();
      assert.equal(named.at(-1)[0], key);
      assert.equal(named.at(-1).length, 2);
    }
    mat.sheenColorMap = null;
    assert.deepEqual(cleared.at(-1), ['sheenColorMap']);
  });
});

describe('MeshToonMaterial.gradientMap', { skip }, () => {
  it('binds by its three.js name, wasm takes it, and null empties it', () => {
    const mat = new S.MeshToonMaterial({ gradientMap: texel() });
    const took = [];
    const orig = mat._w.setNamedMapData.bind(mat._w);
    mat._w.setNamedMapData = (...a) => { const ok = orig(...a); took.push([a[0], ok]); return ok; };
    const cleared = spy(mat._w, 'clearMap');
    assert.ok(mat.gradientMap, 'the option is kept');
    mat.gradientMap = texel();
    assert.deepEqual(took.at(-1), ['gradientMap', true]);
    mat.gradientMap = null;
    assert.deepEqual(cleared.at(-1), ['gradientMap']);
    assert.equal(mat.gradientMap, null);
  });
});

describe('texture.channel', { skip }, () => {
  it('names the uv set, reaches wasm and re-binds', () => {
    const tex = texel();
    const mat = new S.MeshStandardMaterial({ aoMap: tex });
    const binds = spy(mat._w, 'setAoMapData');
    const set = spy(tex._w, 'setChannel');
    assert.equal(tex.channel, 0);
    tex.channel = 1;
    assert.deepEqual(set.at(-1), [1]);
    assert.equal(binds.length, 1);
  });
});
