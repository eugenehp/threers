import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

describe('threers-js package', () => {
  it('declares mini + full ESM exports', () => {
    const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    assert.equal(pkg.type, 'module');
    assert.equal(pkg.name, 'threers');
    assert.ok(pkg.exports['.']);
    assert.ok(pkg.exports['./mini']);
    assert.ok(pkg.exports['./full']);
  });

  it('default entry points at mini', () => {
    const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    const def = pkg.exports['.'].import;
    assert.match(def, /mini/);
  });

  it('has dist/mini and dist/full after build (skip if not built)', {
    skip: !existsSync(join(root, 'dist/mini/threejs-shim.js')),
  }, () => {
    assert.ok(existsSync(join(root, 'dist/mini/pkg/threers_bg.wasm')));
    assert.ok(existsSync(join(root, 'dist/full/pkg/threers_bg.wasm')));
    const miniShim = readFileSync(join(root, 'dist/mini/threejs-shim.js'), 'utf8');
    assert.match(miniShim, /export (async )?function initThreers|export \{[^}]*initThreers/);
    const miniBytes = readFileSync(join(root, 'dist/mini/pkg/threers_bg.wasm')).length;
    const fullBytes = readFileSync(join(root, 'dist/full/pkg/threers_bg.wasm')).length;
    assert.ok(fullBytes > miniBytes, `full (${fullBytes}) should exceed mini (${miniBytes})`);
  });
});
