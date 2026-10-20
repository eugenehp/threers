// The shim's post-processing over a real wasm build: the passes and the
// composer are Rust (`crate::postprocessing`), so these run the wasm itself
// (`THREERS_PKG_DIR`, else web/pkg) — skipped, and reported as skipped, when
// that build predates `WebPass`. What needs a GPU — the passes' pixels, and
// the same bytes on WebGL2 and WebGPU — is in the browser suites
// (tests/web_postfx) and the native ones (src/postprocessing/tests.rs).
//
// Here: the seeded random stream against three.js's GlitchPass and SSAOPass
// draw orders (transcribed from r165), the wrapper classes' forwarding of
// properties and uniforms, and the composer's bookkeeping.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const shim = readFileSync(join(repo, 'web/threejs-shim.js'), 'utf8');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

let pkg = null;
if (existsSync(join(pkgDir, 'threers.js')) && existsSync(join(pkgDir, 'threers_bg.wasm'))) {
  const mod = await import(pathToFileURL(join(pkgDir, 'threers.js')).href);
  mod.initSync({ module: readFileSync(join(pkgDir, 'threers_bg.wasm')) });
  if (typeof mod.WebPass?.glitchPass === 'function') pkg = mod;
}
const skip = !pkg && `no wasm build with WebPass in ${pkgDir} (build one and set THREERS_PKG_DIR)`;

/// Source of a top-level `export class` / `class` / `function` / `const`,
/// with the `export` dropped.
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

/// The parity scenes' `Math.random`.
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
const randFloat = (r, lo, hi) => lo + r() * (hi - lo);
const randInt = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

describe('the seeded random stream', { skip }, () => {
  it('is the parity LCG, shared with JavaScript', () => {
    pkg.seedRandom(42);
    const r = lcg(42);
    for (let i = 0; i < 100; i++) assert.equal(pkg.random(), r());
    pkg.unseedRandom();
  });

  it('GlitchPass draws three.js r165\'s numbers in its order, frame after frame', () => {
    // three.js GlitchPass, transcribed: the heightmap (randFloat per texel),
    // four UUIDs (DataTexture, its Source, the ShaderMaterial, the quad's
    // Mesh), the trigger; then per render the seed and, by frame, the glitch.
    const r = lcg(42);
    const dt = 16;
    const height = Array.from({ length: dt * dt }, () => randFloat(r, 0, 1));
    for (let i = 0; i < 16; i++) r();
    let randX = randInt(r, 120, 240);
    let curF = 0;
    const u = { byp: 0, amount: 0.08, angle: 0.02, seed: 0.02, seed_x: 0.02, seed_y: 0.02,
      distortion_x: 0.5, distortion_y: 0.6 };
    const frame = () => {
      u.seed = r();
      u.byp = 0;
      if (curF % randX === 0) {
        u.amount = r() / 30; u.angle = randFloat(r, -Math.PI, Math.PI);
        u.seed_x = randFloat(r, -1, 1); u.seed_y = randFloat(r, -1, 1);
        u.distortion_x = randFloat(r, 0, 1); u.distortion_y = randFloat(r, 0, 1);
        curF = 0; randX = randInt(r, 120, 240);
      } else if (curF % randX < randX / 5) {
        u.amount = r() / 90; u.angle = randFloat(r, -Math.PI, Math.PI);
        u.distortion_x = randFloat(r, 0, 1); u.distortion_y = randFloat(r, 0, 1);
        u.seed_x = randFloat(r, -0.3, 0.3); u.seed_y = randFloat(r, -0.3, 0.3);
      } else {
        u.byp = 1;
      }
      curF++;
    };

    pkg.seedRandom(42);
    const p = pkg.WebPass.glitchPass(dt);
    assert.deepEqual(Array.from(p.glitchHeightMap()), height.map(Math.fround));
    assert.equal(p.get('randX'), randX);
    for (let f = 0; f < 300; f++) {
      frame();
      p.glitchStep();
      for (const k in u) {
        assert.equal(p.get(k), Math.fround(u[k]), `frame ${f}: ${k}`);
      }
      assert.equal(p.get('curF'), curF, `frame ${f}: curF`);
    }
    pkg.unseedRandom();
  });

  it('SSAOPass builds three.js r165\'s kernel from the stream', () => {
    const r = lcg(0x5a00426);
    const n = 32;
    const kernel = [];
    for (let i = 0; i < n; i++) {
      let x = r() * 2 - 1, y = r() * 2 - 1, z = r();
      const len = Math.hypot(x, y, z);
      x /= len; y /= len; z /= len;
      let scale = i / n;
      scale = 0.1 + (1 - 0.1) * scale * scale;
      kernel.push(x * scale, y * scale, z * scale);
    }
    pkg.seedRandom(0x5a00426);
    const p = pkg.WebPass.ssaoPass(64, 64, n);
    assert.deepEqual(Array.from(p.ssaoKernel()), kernel.map(Math.fround));
    // Then SimplexNoise's permutation (256 draws) and 16 (x, y) pairs: the
    // stream is 256 + 32 draws further on.
    for (let i = 0; i < 256 + 32; i++) r();
    assert.equal(pkg.random(), r());
    pkg.unseedRandom();
  });
});

describe('pass properties and uniforms forward to the Rust pass', { skip }, () => {
  // The shim's forwarding over the real wasm, with its own vector and colour
  // classes.
  const lib = skip ? null : new Function('WebColor', 'WebPass',
    `${['Vector2', 'Vector3', 'Vector4', 'Color', '_ppMakeLive', '_ppToArray', '_ppObjectIds',
      '_ppTextureHandle', '_ppForward', '_ppSync', '_Pass'].map(extract).join('\n')}
     class UnrealBloomPass extends _Pass {
       constructor(res, s, r, t) { super(WebPass.unrealBloomPass(res.x, res.y, s, r, t)); }
     }
     class DotScreenPass extends _Pass {
       constructor(c, a, s) { super(WebPass.dotScreenPass(c.x, c.y, a, s)); }
     }
     class FilmPass extends _Pass {
       constructor(i, g) { super(WebPass.filmPass(i, g)); }
     }
     class OutlinePass extends _Pass {
       constructor(res) { super(WebPass.outlinePass(res.x, res.y)); }
     }
     return { Vector2, Color, UnrealBloomPass, DotScreenPass, FilmPass, OutlinePass, _ppSync };`,
  )(pkg.WebColor, pkg.WebPass);

  it('a property assignment reaches the pass at once', () => {
    const b = new lib.UnrealBloomPass(new lib.Vector2(800, 600), 1.5, 0.4, 0.85);
    assert.equal(b.strength, 1.5);
    b.strength = 2.25;
    assert.equal(b._w.get('strength'), 2.25);
    b.threshold = 0.5;
    assert.equal(b.threshold, 0.5);
    b.enabled = false;
    assert.equal(b._w.enabled, false);
  });

  it('`uniforms.X.value` forwards, and a vector mutated in place is pushed', () => {
    const d = new lib.DotScreenPass(new lib.Vector2(0.5, 0.5), 1.57, 1.0);
    d.uniforms.scale.value = 3;
    assert.equal(d._w.get('scale'), 3);
    d.uniforms.center.value.set(0.25, 0.75);
    lib._ppSync(d);
    assert.deepEqual(Array.from(d._w.get('center')), [0.25, 0.75]);
    const f = new lib.FilmPass(0.35, false);
    f.uniforms.grayscale.value = true;
    assert.equal(f._w.get('grayscale'), true);
  });

  it('a colour property forwards its linear value', () => {
    const o = new lib.OutlinePass(new lib.Vector2(64, 64));
    o.visibleEdgeColor.setHex(0xff0000);
    lib._ppSync(o);
    assert.deepEqual(Array.from(o._w.get('visibleEdgeColor')), [1, 0, 0]);
    o.hiddenEdgeColor = new lib.Color(0.5, 0.25, 0);
    assert.deepEqual(Array.from(o._w.get('hiddenEdgeColor')).map((v) => +v.toFixed(4)), [0.5, 0.25, 0]);
  });

  it('three.js\'s defaults', () => {
    const o = pkg.WebPass.outlinePass(64, 64);
    assert.equal(o.get('edgeStrength'), 3);
    assert.equal(o.get('edgeThickness'), 1);
    assert.equal(o.get('downSampleRatio'), 2);
    const s = pkg.WebPass.ssaoPass(64, 64, 32);
    assert.equal(s.get('kernelRadius'), 8);
    assert.equal(Math.fround(s.get('minDistance')), Math.fround(0.005));
    const sao = pkg.WebPass.saoPass(64, 64);
    assert.equal(Math.fround(sao.get('saoIntensity')), Math.fround(0.18));
    assert.equal(sao.get('saoKernelRadius'), 100);
    const ssr = pkg.WebPass.ssrPass(64, 64);
    assert.equal(ssr.get('maxDistance'), 180);
    const fxaa = pkg.WebPass.shaderPass('FXAAShader');
    assert.deepEqual(Array.from(fxaa.get('resolution')).map(Math.fround), [1 / 1024, 1 / 512].map(Math.fround));
    assert.equal(pkg.WebPass.isKnownShader('MyOwnShader'), false);
  });

  it('UnrealBloomPass\'s mips are three.js\'s sizes', () => {
    const b = pkg.WebPass.unrealBloomPass(801, 599, 1, 0, 0);
    // Math.round(801 / 2) = 401, then halving with rounding.
    const sizes = [];
    let x = Math.round(801 / 2), y = Math.round(599 / 2);
    for (let i = 0; i < 5; i++) { sizes.push(x, y); x = Math.round(x / 2); y = Math.round(y / 2); }
    assert.deepEqual(Array.from(b.bloomMipSizes()), sizes);
  });
});

describe('WebEffectComposer', { skip }, () => {
  it('adds, inserts and removes passes, and sizes them as it adds them', () => {
    const c = new pkg.WebEffectComposer(400, 300, 2);
    assert.deepEqual(Array.from(c.drawingSize()), [800, 600]);
    const render = pkg.WebPass.renderPass();
    const bloom = pkg.WebPass.unrealBloomPass(10, 10, 1, 0, 0);
    const copy = pkg.WebPass.shaderPass('CopyShader');
    c.addPass(render);
    c.addPass(copy);
    c.insertPass(bloom, 1);
    assert.deepEqual(Array.from(c.passNames()), ['RenderPass', 'UnrealBloomPass', 'ShaderPass']);
    // three.js's addPass calls setSize with the composer's device size.
    assert.deepEqual(Array.from(bloom.bloomMipSizes()).slice(0, 2), [400, 300]);
    c.removePass(bloom);
    assert.deepEqual(Array.from(c.passNames()), ['RenderPass', 'ShaderPass']);
    c.setPixelRatio(1);
    assert.deepEqual(Array.from(c.drawingSize()), [400, 300]);
    assert.equal(c.renderToScreen, true);
    c.renderToScreen = false;
    assert.equal(c.renderToScreen, false);
  });

  it('a pass shared with its JavaScript object is the one the composer holds', () => {
    const c = new pkg.WebEffectComposer(64, 64, 1);
    const film = pkg.WebPass.filmPass(0.5, false);
    c.addPass(film);
    film.set('intensity', 0.9);
    film.enabled = false;
    // Still one pass, the same one: the change is visible through the
    // composer's own handle on it.
    assert.deepEqual(Array.from(c.passNames()), ['FilmPass']);
    assert.equal(Math.fround(film.get('intensity')), Math.fround(0.9));
  });
});

describe('the shim\'s EffectComposer', () => {
  it('passes the clock\'s delta when render() is given none, as three.js does', () => {
    const src = extract('EffectComposer');
    assert.match(src, /if \(deltaTime === undefined\) deltaTime = this\.clock\.getDelta\(\);/);
  });

  it('runs the chain in Rust: it calls one composer render and no pass\'s render', () => {
    const src = extract('EffectComposer');
    assert.match(src, /this\._w\.render[012]\(/);
    assert.doesNotMatch(src, /\.render\(renderer, this\.writeBuffer/);
  });

  it('has no CPU or GLSL post-fx fallback left', () => {
    for (const gone of ['_cpuCopyToCanvas', '_webglFxaaToCanvas', '_glslFxaaOutputBake',
      '_cpuDotscreenOutput', '_cpuDotscreenPattern', '_glslGlitchDispBake', '_cpuGlitchOutput',
      '_f16HalfToFloat', '_f16BytesToRgba8', '_flipRowsBytes', '_sampleF16Linear',
      '__FXAA_FORCE', '__GLITCH_FORCE', 'getContext(\'webgl2\')']) {
      assert.equal(shim.includes(gone), false, `${gone} is back in the shim`);
    }
  });
});
