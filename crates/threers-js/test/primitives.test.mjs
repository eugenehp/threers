// The shim's primitives are three r165's: BufferGeometry subclasses with
// three.js's constructor, `type`, `parameters`, index, attributes and groups —
// the buffers bit for bit, against three.js's own output
// (tests/parity/scenes/rust/primitives-upstream.txt, from
// tests/parity/export-primitives-upstream.mjs). Runs the shim over a real wasm
// build (`THREERS_PKG_DIR`, else web/pkg).

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const pkgDir = process.env.THREERS_PKG_DIR || join(repo, 'web/pkg');

async function loadShim(dir) {
  const pkgUrl = pathToFileURL(join(dir, 'threers.js')).href;
  const m = await import(pkgUrl);
  if (typeof m.WebGeometry?.threePrimitive !== 'function') return null;
  m.initSync({ module: readFileSync(join(dir, 'threers_bg.wasm')) });
  const web = join(repo, 'web');
  const src = readFileSync(join(web, 'threejs-shim.js'), 'utf8')
    .replaceAll("'./pkg/threers.js'", JSON.stringify(pkgUrl))
    .replace(/from '\.\/([^']+)'/g, (_, p) => `from ${JSON.stringify(pathToFileURL(join(web, p)).href)}`);
  const file = join(mkdtempSync(join(tmpdir(), 'threers-prim-')), 'shim.mjs');
  writeFileSync(file, src);
  return import(pathToFileURL(file).href);
}

const S = existsSync(join(pkgDir, 'threers_bg.wasm')) ? await loadShim(pkgDir) : null;
const skip = !S && `no wasm build with WebGeometry.threePrimitive in ${pkgDir}`;

function fixture() {
  const text = readFileSync(join(repo, 'tests/parity/scenes/rust/primitives-upstream.txt'), 'utf8');
  const cases = [];
  let c = null;
  for (const line of text.split('\n')) {
    const [key, ...rest] = line.split(' ');
    if (key === 'geom') c = { name: rest[0], args: JSON.parse(rest.slice(1).join(' ')), attrs: {}, groups: [] };
    else if (key === 'index') c.index = rest[0] === '-' ? null : new Uint32Array(new Uint8Array(Buffer.from(rest[0], 'base64')).buffer);
    else if (key === 'attr') c.attrs[rest[0]] = { itemSize: Number(rest[1]), bits: new Uint32Array(new Uint8Array(Buffer.from(rest[2], 'base64')).buffer) };
    else if (key === 'group') c.groups.push(rest.map(Number));
    else if (key === 'end') cases.push(c);
  }
  return cases;
}

describe('three.js primitives in the shim', { skip }, () => {
  for (const c of S ? fixture() : []) {
    if (c.name === 'LatheGeometry') continue; // its own class, already a BufferGeometry
    it(`${c.name}(${JSON.stringify(c.args)}) is three.js's to the bit`, () => {
      const g = new S[c.name](...c.args);
      assert.ok(g.isBufferGeometry, 'a BufferGeometry');
      assert.equal(g.type, c.name);
      assert.ok(g.parameters, 'parameters');
      if (c.index) assert.deepEqual(Array.from(g.index.array), Array.from(c.index), 'index');
      for (const [name, want] of Object.entries(c.attrs)) {
        const a = g.attributes[name];
        assert.ok(a, `no ${name}`);
        assert.equal(a.itemSize, want.itemSize, `${name} itemSize`);
        const got = new Uint32Array(Float32Array.from(a.array).buffer);
        assert.equal(got.length, want.bits.length, `${name} length`);
        const differ = got.filter((v, i) => v !== want.bits[i]).length;
        assert.equal(differ, 0, `${name}: ${differ} values differ`);
      }
      assert.deepEqual(g.groups.map(gr => [gr.start, gr.count, gr.materialIndex]), c.groups, 'groups');
    });
  }

  it('a primitive works as three.js code uses one', () => {
    const g = new S.BoxGeometry(2, 1, 1);
    g.translate(1, 0, 0);
    g.computeBoundingBox();
    assert.equal(g.boundingBox.min.x, 0);
    assert.equal(g.boundingBox.max.x, 2);
    assert.equal(g.parameters.width, 2);
  });
});
