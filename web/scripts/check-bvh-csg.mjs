#!/usr/bin/env node
/** CI gate: bvh-csg addon + core exports when BVH_CSG=1. */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(__dirname, '..');

// The boolean runs in wasm (`csgEvaluate`, src/wasm_csg.rs); bvh-csg-impl.js
// keeps three-bvh-csg's classes and bookkeeping around it.
const required = [
    'bvh-csg-addon.js',
    'bvh-csg-impl.js',
];

let ok = true;
for (const f of required) {
    if (!fs.existsSync(path.join(webDir, f))) {
        console.error(`missing: web/${f}`);
        ok = false;
    }
}

const feat = fs.readFileSync(path.join(webDir, 'features.js'), 'utf8');
if (!feat.includes('bvhCsg: true')) {
    console.error('features.bvhCsg is not true — rebuild with BVH_CSG=1 web/build.sh');
    ok = false;
}
if (!feat.includes('meshBvh: true')) {
    console.error('bvh-csg requires mesh-bvh — rebuild with BVH_CSG=1 web/build.sh');
    ok = false;
}
const pkg = fs.readFileSync(path.join(webDir, 'pkg', 'threers.js'), 'utf8');
if (!pkg.includes('WebMeshBvh')) {
    console.error('wasm missing WebMeshBvh — rebuild with BVH_CSG=1 web/build.sh');
    ok = false;
}
if (!pkg.includes('csgEvaluate')) {
    console.error('wasm missing csgEvaluate — rebuild with BVH_CSG=1 web/build.sh');
    ok = false;
}

const addon = fs.readFileSync(path.join(webDir, 'bvh-csg-addon.js'), 'utf8');
if (!addon.includes('bvh-csg-impl')) {
    console.error('bvh-csg-addon.js does not point to impl');
    ok = false;
}

if (!ok) process.exit(1);
console.log('check-bvh-csg: OK');
