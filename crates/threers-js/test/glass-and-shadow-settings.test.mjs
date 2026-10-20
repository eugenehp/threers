// The shim hands the renderer what three.js keeps on the JavaScript side:
// `light.shadow.bias` / `.normalBias` / `.radius` / `.mapSize`, at
// construction and when they change later, and threers's own
// `MeshPhysicalMaterial.transparencyMode`. Read back from the wasm objects
// after a sync (`WebLight.shadowSettings`, `WebMaterial.transparencyMode`).
//
// The shim runs as a module over a real wasm build (`THREERS_PKG_DIR`, else
// web/pkg).

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
  if (!('shadowSettings' in m.WebLight.prototype)) return null;
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-settings-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShimOver(pkgDir) : null;
const skip = !S && 'wasm build missing or older than WebLight.shadowSettings';
const close = (a, b) => Math.abs(a - b) < 1e-6;

describe('shim settings reach the renderer', { skip }, () => {
  it('light.shadow bias, normalBias, radius and mapSize, then and later', () => {
    const scene = new S.Scene();
    const sun = new S.DirectionalLight(0xffffff, 2);
    sun.castShadow = true;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 3;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(sun);
    scene._syncNow();
    let [bias, normalBias, radius, mapSize] = sun._w.shadowSettings;
    // three.js adds a negative bias; the renderer subtracts a positive one.
    assert.ok(close(bias, 0.0005), `bias ${bias}`);
    assert.ok(close(normalBias, 0.02), `normalBias ${normalBias}`);
    assert.equal(radius, 3);
    assert.equal(mapSize, 1024);

    // Changed after the light is in the scene: picked up at the next sync.
    sun.shadow.radius = 0;
    sun.shadow.mapSize.width = 2048;
    scene._syncNow();
    [bias, normalBias, radius, mapSize] = sun._w.shadowSettings;
    assert.equal(radius, 0);
    assert.equal(mapSize, 2048);
  });

  it('MeshPhysicalMaterial.transparencyMode, in the constructor and set later', () => {
    const glass = new S.MeshPhysicalMaterial({ transmission: 1, transparencyMode: 'glass' });
    assert.equal(glass.transparencyMode, 'glass');
    assert.equal(glass._w.transparencyMode, 1);
    for (const [mode, n] of [['oit', 2], ['refract', 3], ['blend', 0]]) {
      glass.transparencyMode = mode;
      assert.equal(glass._w.transparencyMode, n, mode);
    }
    // Not a mode: ignored, with a warning, rather than reset to the default.
    const warn = console.warn;
    console.warn = () => {};
    glass.transparencyMode = 'frosted';
    console.warn = warn;
    assert.equal(glass.transparencyMode, 'blend');
    assert.equal(new S.MeshPhysicalMaterial({ transmission: 1 }).transparencyMode, 'blend');
  });

  it('Color.setStyle takes the rest of CSS Color 4, through the renderer', () => {
    const near = (c, hex) => {
      const want = new S.Color(hex);
      return ['r', 'g', 'b'].every((k) => Math.abs(c[k] - want[k]) < 5e-3);
    };
    // What three.js parses itself, as three.js does.
    assert.ok(near(new S.Color().setStyle('rgb(255,0,0)'), 0xff0000));
    // And the rest: CSS Color 4's worked example, in each of its spaces.
    for (const css of ['lab(29.2345% 39.3825 20.0664)', 'oklch(40.101% 0.12332 21.555)',
      'lch(29.2345% 44.2 27)', 'color(srgb 0.49 0.137 0.161)']) {
      const c = new S.Color().setStyle(css);
      assert.ok(near(c, 0x7d2329), `${css} -> ${c.getHexString()}`);
    }
    assert.ok(near(new S.Color().setStyle('#ff000080'), 0xff0000), '#rrggbbaa');
    assert.ok(near(new S.Color().setStyle('hwb(120 0% 0%)'), 0x00ff00), 'hwb()');
  });
});
