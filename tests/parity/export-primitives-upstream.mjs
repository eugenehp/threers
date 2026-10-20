#!/usr/bin/env node
// The primitives oracle: three r165's own geometry classes (pinned in this
// package), built with a spread of parameters and written out exactly — the
// index, every attribute as its raw float bits, and the groups.
// `tests/primitives_upstream.rs` builds the same through threers and requires
// the same bits: a vertex a hair off an axis decides which side of a ray or a
// cutting plane it falls on, so near is not enough.
//
//   node tests/parity/export-primitives-upstream.mjs
//   → tests/parity/scenes/rust/primitives-upstream.txt
//
// One block per case:
//   geom <name> <params as JSON, no spaces>
//   index <base64 little-endian u32> | index -
//   attr <name> <itemSize> <base64 little-endian f32>
//   group <start> <count> <materialIndex>
//   end

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as THREE from 'three';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(__dirname, 'scenes', 'rust', 'primitives-upstream.txt');
const PI = Math.PI;

/** `[class name, constructor arguments]`, each as three.js takes them. */
export const CASES = [
    ['BoxGeometry', [1, 1, 1, 1, 1, 1]],
    ['BoxGeometry', [0.9, 6, 0.9, 2, 3, 1]],
    ['BoxGeometry', [1.2, 1.2, 1.2, 1, 1, 1]],
    ['SphereGeometry', [1, 32, 16, 0, 2 * PI, 0, PI]],
    ['SphereGeometry', [0.55, 24, 12, 0, 2 * PI, 0, PI]],
    ['SphereGeometry', [1, 12, 8, 0.3, 4, 0.2, 2]],
    ['CylinderGeometry', [1, 1, 1, 32, 1, false, 0, 2 * PI]],
    ['CylinderGeometry', [0.5, 0.7, 2.5, 24, 2, false, 0, 2 * PI]],
    ['CylinderGeometry', [0.5, 0.5, 2.5, 24, 1, true, 0.5, 4]],
    ['ConeGeometry', [1, 1, 32, 1, false, 0, 2 * PI]],
    ['ConeGeometry', [0.6, 1.4, 16, 2, false, 0, 2 * PI]],
    ['PlaneGeometry', [1, 1, 1, 1]],
    ['PlaneGeometry', [2, 3, 4, 5]],
    ['CircleGeometry', [1, 32, 0, 2 * PI]],
    ['CircleGeometry', [1, 24, 0.4, 3]],
    ['RingGeometry', [0.5, 1, 32, 1, 0, 2 * PI]],
    ['RingGeometry', [0.5, 1, 24, 3, 0.4, 3]],
    ['TorusGeometry', [1, 0.4, 12, 48, 2 * PI]],
    ['TorusGeometry', [1, 0.3, 12, 24, 4]],
    ['TorusKnotGeometry', [1, 0.4, 64, 8, 2, 3]],
    ['TorusKnotGeometry', [0.8, 0.25, 64, 8, 2, 3]],
    ['TetrahedronGeometry', [1, 0]],
    ['TetrahedronGeometry', [1, 1]],
    ['OctahedronGeometry', [1, 0]],
    ['OctahedronGeometry', [1, 1]],
    ['IcosahedronGeometry', [1, 0]],
    ['IcosahedronGeometry', [1, 1]],
    ['DodecahedronGeometry', [1, 0]],
    ['DodecahedronGeometry', [1, 1]],
    ['CapsuleGeometry', [1, 1, 4, 8]],
    ['CapsuleGeometry', [0.5, 1, 4, 8]],
    ['LatheGeometry', [[[0, -0.5], [0.5, 0], [0, 0.5]], 12, 0, 2 * PI]],
    ['LatheGeometry', [[[0.1, -1], [0.6, -0.4], [0.3, 0.2], [0.8, 1]], 16, 0.5, 4]],
];

function b64(typed) {
    return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64');
}

function build(name, args) {
    if (name === 'LatheGeometry') {
        const [points, ...rest] = args;
        return new THREE.LatheGeometry(points.map(([x, y]) => new THREE.Vector2(x, y)), ...rest);
    }
    return new THREE[name](...args);
}

const lines = [`# three r${THREE.REVISION} primitives — tests/parity/export-primitives-upstream.mjs`];
for (const [name, args] of CASES) {
    const g = build(name, args);
    lines.push(`geom ${name} ${JSON.stringify(args)}`);
    lines.push(g.index ? `index ${b64(Uint32Array.from(g.index.array))}` : 'index -');
    for (const key of ['position', 'normal', 'uv']) {
        const a = g.attributes[key];
        if (a) lines.push(`attr ${key} ${a.itemSize} ${b64(Float32Array.from(a.array))}`);
    }
    for (const gr of g.groups) lines.push(`group ${gr.start} ${gr.count} ${gr.materialIndex}`);
    lines.push('end');
}
fs.writeFileSync(out, lines.join('\n') + '\n');
console.log(`wrote ${path.relative(process.cwd(), out)}: ${CASES.length} geometries`);
