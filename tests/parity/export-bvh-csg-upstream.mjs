#!/usr/bin/env node
// The CSG oracle: three-bvh-csg@0.0.16 itself (with three@0.165.0 and
// three-mesh-bvh@0.7.6, pinned in this package), run in Node on the bvh-csg
// parity scenes. Every `Evaluator.evaluate` call the scenes make is recorded —
// both operands as they went in (non-indexed positions, groups, f64
// `matrixWorld`), the operations and settings, and what upstream produced:
// triangle count, surface area, enclosed volume, and area per output group.
//
// `tests/csg_upstream.rs` replays each call through the Rust port and checks
// it against these numbers. Area and volume do not depend on how a surface is
// triangulated, so they hold the port to the same solid even where the two
// split triangles in a different order.
//
//   node tests/parity/export-bvh-csg-upstream.mjs
//   → tests/parity/scenes/rust/bvh-csg-upstream.txt
//
// The file is line-based so the test needs no JSON crate:
//   call <scene> <ops,…> <useGroups 0|1> <attributes,…>
//   brush            (twice: a, then b)
//   matrix <16 column-major numbers>
//   group <start> <count, -1 for all> <materialIndex>   (any number)
//   attr <name> <itemSize> <base64 little-endian f32>   (any number)
//   result <triangles> <area> <volume>                  (one per operation)
//   rgroup <materialIndex> <triangles> <area>           (that result's groups)
//   end

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as THREE from 'three';
import {
    Brush, Evaluator, Operation, OperationGroup,
    ADDITION, SUBTRACTION, INTERSECTION, DIFFERENCE, REVERSE_SUBTRACTION, HOLLOW_INTERSECTION,
} from 'three-bvh-csg';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(__dirname, 'scenes', 'rust', 'bvh-csg-upstream.txt');

// The shim's `geometryToBufferGeometry`: three's primitive as a triangle soup.
const soup = g => g.toNonIndexed();

function b64(typed) {
    return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64');
}

function describe(brush) {
    const g = brush.geometry.index ? brush.geometry.toNonIndexed() : brush.geometry;
    // `toNonIndexed` keeps the groups in index units, which are now vertex units.
    return {
        position: b64(Float32Array.from(g.attributes.position.array)),
        normal: g.attributes.normal ? b64(Float32Array.from(g.attributes.normal.array)) : null,
        uv: g.attributes.uv ? b64(Float32Array.from(g.attributes.uv.array)) : null,
        groups: brush.geometry.groups.map(({ start, count, materialIndex }) => ({
            start, count: Number.isFinite(count) ? count : -1, materialIndex,
        })),
        matrixWorld: Array.from(brush.matrixWorld.elements),
    };
}

// Area, signed volume and per-group area of the drawn triangles.
function metrics(geometry) {
    const pos = geometry.attributes.position;
    const count = Math.min(geometry.drawRange.count, pos.count);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const ab = new THREE.Vector3(), ac = new THREE.Vector3();
    const triArea = i => {
        a.fromBufferAttribute(pos, i); b.fromBufferAttribute(pos, i + 1); c.fromBufferAttribute(pos, i + 2);
        return ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() / 2;
    };
    let area = 0, volume = 0;
    for (let i = 0; i + 2 < count; i += 3) {
        area += triArea(i);
        a.fromBufferAttribute(pos, i); b.fromBufferAttribute(pos, i + 1); c.fromBufferAttribute(pos, i + 2);
        volume += a.dot(b.clone().cross(c)) / 6;
    }
    const groups = geometry.groups.map(g => {
        let ga = 0;
        for (let i = g.start; i + 2 < g.start + g.count; i += 3) ga += triArea(i);
        return { materialIndex: g.materialIndex, triangles: g.count / 3, area: ga };
    });
    return { triangles: Math.floor(count / 3), area, volume, groups };
}

const calls = [];
let scene = '';
const evaluate = Evaluator.prototype.evaluate;
Evaluator.prototype.evaluate = function (a, b, operations, targets) {
    const record = {
        scene,
        operations: Array.isArray(operations) ? [...operations] : [operations],
        useGroups: this.useGroups,
        attributes: [...this.attributes],
        a: describe(a),
        b: describe(b),
    };
    const result = evaluate.call(this, a, b, operations, targets);
    const results = Array.isArray(result) ? result : [result];
    record.results = results.map(r => metrics(r.geometry));
    calls.push(record);
    return result;
};

// ---- the scenes, as tests/parity/scenes/threers-bvh-csg-*.html build them ----

function twoBrushes() {
    const brush1 = new Brush(soup(new THREE.SphereGeometry(1, 32, 16)));
    const brush2 = new Brush(soup(new THREE.BoxGeometry(1.2, 1.2, 1.2)));
    brush2.position.set(0.5, 0, 0);
    brush1.updateMatrixWorld(true);
    brush2.updateMatrixWorld(true);
    return [brush1, brush2];
}

const SCENES = {
    'bvh-csg-union': () => { const [x, y] = twoBrushes(); new Evaluator().evaluate(x, y, ADDITION, new Brush()); },
    'bvh-csg-subtract': () => { const [x, y] = twoBrushes(); new Evaluator().evaluate(x, y, SUBTRACTION, new Brush()); },
    'bvh-csg-intersect': () => { const [x, y] = twoBrushes(); new Evaluator().evaluate(x, y, INTERSECTION, new Brush()); },
    'bvh-csg-difference': () => { const [x, y] = twoBrushes(); new Evaluator().evaluate(x, y, DIFFERENCE, new Brush()); },
    'bvh-csg-multimaterial': () => {
        const red = new THREE.MeshStandardMaterial(), green = new THREE.MeshStandardMaterial(), blue = new THREE.MeshStandardMaterial();
        const evaluator = new Evaluator();
        const c1 = new Brush(soup(new THREE.BoxGeometry(0.9, 6, 0.9)), blue);
        const c2 = new Brush(soup(new THREE.BoxGeometry(6, 0.9, 0.9)), blue);
        const c3 = new Brush(soup(new THREE.BoxGeometry(0.9, 0.9, 6)), blue);
        const sphere = new Brush(soup(new THREE.SphereGeometry(1, 32, 16)), green);
        const box = new Brush(soup(new THREE.BoxGeometry(1.5, 1.5, 1.5)), red);
        [c1, c2, c3, sphere, box].forEach(b => b.updateMatrixWorld(true));
        let result = evaluator.evaluate(c1, c2, ADDITION);
        result = evaluator.evaluate(result, c3, ADDITION);
        result = evaluator.evaluate(sphere, result, SUBTRACTION);
        evaluator.evaluate(box, result, INTERSECTION);
    },
    'bvh-csg-hollow': () => {
        const triangles = 40;
        const positions = new Float32Array(triangles * 9);
        const normals = new Float32Array(triangles * 9);
        const n = 200, n2 = n / 2, d = 80, d2 = d / 2;
        const pA = new THREE.Vector3(), pB = new THREE.Vector3(), pC = new THREE.Vector3();
        const cb = new THREE.Vector3(), ab = new THREE.Vector3();
        for (let t = 0; t < triangles; t++) {
            const i = t * 9;
            const x = Math.sin(t * 0.31) * n - n2, y = Math.cos(t * 0.47) * n - n2, z = Math.sin(t * 0.19) * n - n2;
            const ax = x + Math.sin(t * 1.1) * d - d2, ay = y + Math.cos(t * 1.3) * d - d2, az = z + Math.sin(t * 0.9) * d - d2;
            const bx = x + Math.cos(t * 0.7) * d - d2, by = y + Math.sin(t * 1.7) * d - d2, bz = z + Math.cos(t * 1.1) * d - d2;
            const cx = x + Math.sin(t * 0.5) * d - d2, cy = y + Math.cos(t * 0.3) * d - d2, cz = z + Math.sin(t * 1.5) * d - d2;
            positions.set([ax, ay, az, bx, by, bz, cx, cy, cz], i);
            pA.set(ax, ay, az); pB.set(bx, by, bz); pC.set(cx, cy, cz);
            cb.subVectors(pC, pB); ab.subVectors(pA, pB); cb.cross(ab).normalize();
            normals.set([cb.x, cb.y, cb.z, cb.x, cb.y, cb.z, cb.x, cb.y, cb.z], i);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        geometry.scale(0.006, 0.006, 0.006);
        const evaluator = new Evaluator();
        evaluator.attributes = ['position', 'normal'];
        evaluator.useGroups = false;
        const brushA = new Brush(geometry);
        const brushB = new Brush(soup(new THREE.SphereGeometry(1, 32, 16)));
        brushB.position.set(0, 0.8, 0);
        brushB.scale.setScalar(1.6);
        brushA.updateMatrixWorld(true);
        brushB.updateMatrixWorld(true);
        evaluator.evaluate(brushA, brushB, HOLLOW_INTERSECTION, new Brush());
    },
    'bvh-csg-multiop': () => {
        const evaluator = new Evaluator();
        evaluator.attributes = ['position', 'normal'];
        const brush1 = new Brush(soup(new THREE.IcosahedronGeometry(1, 1)), new THREE.MeshStandardMaterial());
        const brush2 = new Brush(soup(new THREE.CylinderGeometry(0.5, 0.5, 2.5, 24)), new THREE.MeshStandardMaterial());
        brush1.rotation.set(0.35, 0.55, 0.2);
        brush2.rotation.set(-0.25, -0.4, -0.65);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        evaluator.evaluate(brush1, brush2, [SUBTRACTION, INTERSECTION, ADDITION, REVERSE_SUBTRACTION],
            [new Brush(), new Brush(), new Brush(), new Brush()]);
    },
    'bvh-csg-complex': () => {
        const evaluator = new Evaluator();
        const knot = new Brush(soup(new THREE.TorusKnotGeometry(0.8, 0.25, 64, 8)), new THREE.MeshStandardMaterial());
        knot.position.y = -0.2;
        const sphereMat = new THREE.MeshStandardMaterial();
        let merged = null;
        for (const s of [[0.35, 0.55, 0.2, 0.18], [-0.45, 0.15, -0.25, 0.14], [0.1, -0.35, 0.4, 0.16]]) {
            const b = new Brush(soup(new THREE.SphereGeometry(1, 20, 12)), sphereMat);
            b.position.set(s[0], s[1], s[2]);
            b.scale.setScalar(s[3]);
            b.updateMatrixWorld(true);
            merged = merged ? evaluator.evaluate(merged, b, ADDITION) : b;
        }
        knot.updateMatrixWorld(true);
        evaluator.evaluate(knot, merged, SUBTRACTION, new Brush());
    },
    'bvh-csg-hierarchy': () => {
        const mat = new THREE.MeshStandardMaterial();
        const evaluator = new Evaluator();
        evaluator.useGroups = false;
        const root = new Operation(soup(new THREE.BoxGeometry(4, 2.5, 2.5)), mat);
        root.operation = ADDITION;
        const cut = new Operation(soup(new THREE.BoxGeometry(3.6, 2.1, 2.1)), mat);
        cut.operation = SUBTRACTION;
        const sphere = new Operation(soup(new THREE.SphereGeometry(0.55, 24, 12)), mat);
        sphere.operation = ADDITION;
        sphere.position.set(-1.1, 0.2, 1.35);
        const windowGroup = new OperationGroup();
        const winCut = new Operation(soup(new THREE.BoxGeometry(1.2, 1.0, 0.5)), mat);
        winCut.operation = SUBTRACTION;
        const winFrame = new Operation(soup(new THREE.BoxGeometry(1.2, 1.0, 0.12)), mat);
        winFrame.operation = ADDITION;
        windowGroup.add(winCut, winFrame);
        windowGroup.position.set(0.8, 0.15, 1.35);
        root.add(cut, sphere, windowGroup);
        root.updateMatrixWorld(true);
        evaluator.evaluateHierarchy(root, new Brush());
    },
};

for (const [name, build] of Object.entries(SCENES)) {
    scene = name;
    build();
}

const num = v => (Number.isInteger(v) ? String(v) : v.toPrecision(17));
const lines = [
    `# three-bvh-csg 0.0.16, three-mesh-bvh 0.7.6, three r${THREE.REVISION} — tests/parity/export-bvh-csg-upstream.mjs`,
];
for (const c of calls) {
    lines.push(`call ${c.scene} ${c.operations.join(',')} ${c.useGroups ? 1 : 0} ${c.attributes.join(',')}`);
    for (const brush of [c.a, c.b]) {
        lines.push('brush');
        lines.push(`matrix ${brush.matrixWorld.map(num).join(' ')}`);
        for (const g of brush.groups) lines.push(`group ${g.start} ${g.count} ${g.materialIndex}`);
        lines.push(`attr position 3 ${brush.position}`);
        if (brush.normal) lines.push(`attr normal 3 ${brush.normal}`);
        if (brush.uv) lines.push(`attr uv 2 ${brush.uv}`);
    }
    for (const r of c.results) {
        lines.push(`result ${r.triangles} ${num(r.area)} ${num(r.volume)}`);
        for (const g of r.groups) lines.push(`rgroup ${g.materialIndex} ${g.triangles} ${num(g.area)}`);
    }
    lines.push('end');
}
fs.writeFileSync(out, lines.join('\n') + '\n');
console.log(`wrote ${path.relative(process.cwd(), out)}: ${calls.length} evaluations`);
for (const c of calls) {
    console.log(`  ${c.scene}: ops ${c.operations.join(',')} → ${c.results.map(r => `${r.triangles} tris, area ${r.area.toFixed(4)}, volume ${r.volume.toFixed(4)}`).join(' | ')}`);
}
