#!/usr/bin/env node
/** Generate threers/threejs bvh-csg parity scene HTML pairs. */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = __dirname;

const IMPORT_MAP = `<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.165.0/build/three.module.js","three-mesh-bvh":"https://unpkg.com/three-mesh-bvh@0.7.6/build/index.module.js","three-bvh-csg":"https://unpkg.com/three-bvh-csg@0.0.16/build/index.module.js"}}</script>`;

const STYLE = `<style>body{margin:0;background:#101418}canvas{display:block;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}#err{position:absolute;top:0;left:0;padding:12px;color:#fdd;background:#500;font-family:monospace;white-space:pre-wrap}</style>`;

const THREERS_ORBIT = `
        const { installOrbitSyncBridge } = await import('/tests/parity/parity-orbit-sync.js');
        installOrbitSyncBridge();
        const controls = new THREE.OrbitControls(cam, canvas);
        window.__parityOrbit = controls;
        window.__parityOrbitCtx = { side: 'threers', orbit: controls, cam, render: () => renderer.render(scene, cam) };
        if (typeof window.__parityPrimeOrbitCtx === 'function') window.__parityPrimeOrbitCtx(window.__parityOrbitCtx);
        (function orbitLoop() {
            // When sync is on and this pane is not being dragged, keep the camera
            // pose from the peer — do not re-derive it from stale local spherical.
            const syncPassive = window.__parityOrbitSyncEnabled && !window.__parityOrbitInteracting;
            if (!window.__parityCaptureLock && !syncPassive) controls.update();
            renderer.render(scene, cam);
            if (window.__parityOrbitSyncHook) window.__parityOrbitSyncHook();
            requestAnimationFrame(orbitLoop);
        })();`;

const THREEJS_ORBIT = `
        const { installOrbitSyncBridge } = await import('/tests/parity/parity-orbit-sync.js');
        installOrbitSyncBridge();
        const { OrbitControls } = await import('https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js');
        const controls = new OrbitControls(cam, renderer.domElement);
        window.__parityOrbit = controls;
        window.__parityOrbitCtx = { side: 'three', orbit: controls, cam, render: () => renderer.render(scene, cam) };
        if (typeof window.__parityPrimeOrbitCtx === 'function') window.__parityPrimeOrbitCtx(window.__parityOrbitCtx);
        (function orbitLoop() {
            const syncPassive = window.__parityOrbitSyncEnabled && !window.__parityOrbitInteracting;
            if (!window.__parityCaptureLock && !syncPassive) controls.update();
            renderer.render(scene, cam);
            if (window.__parityOrbitSyncHook) window.__parityOrbitSyncHook();
            requestAnimationFrame(orbitLoop);
        })();`;

const TRIANGLE_SOUP_FN = `
function makeTriangleSoupGeometry(THREE) {
    const triangles = 40;
    const positions = new Float32Array(triangles * 9);
    const normals = new Float32Array(triangles * 9);
    const n = 200, n2 = n / 2, d = 80, d2 = d / 2;
    const pA = new THREE.Vector3(), pB = new THREE.Vector3(), pC = new THREE.Vector3();
    const cb = new THREE.Vector3(), ab = new THREE.Vector3();
    for (let t = 0; t < triangles; t++) {
        const i = t * 9;
        const x = Math.sin(t * 0.31) * n - n2;
        const y = Math.cos(t * 0.47) * n - n2;
        const z = Math.sin(t * 0.19) * n - n2;
        const ax = x + Math.sin(t * 1.1) * d - d2;
        const ay = y + Math.cos(t * 1.3) * d - d2;
        const az = z + Math.sin(t * 0.9) * d - d2;
        const bx = x + Math.cos(t * 0.7) * d - d2;
        const by = y + Math.sin(t * 1.7) * d - d2;
        const bz = z + Math.cos(t * 1.1) * d - d2;
        const cx = x + Math.sin(t * 0.5) * d - d2;
        const cy = y + Math.cos(t * 0.3) * d - d2;
        const cz = z + Math.sin(t * 1.5) * d - d2;
        positions[i]=ax; positions[i+1]=ay; positions[i+2]=az;
        positions[i+3]=bx; positions[i+4]=by; positions[i+5]=bz;
        positions[i+6]=cx; positions[i+7]=cy; positions[i+8]=cz;
        pA.set(ax,ay,az); pB.set(bx,by,bz); pC.set(cx,cy,cz);
        cb.subVectors(pC,pB); ab.subVectors(pA,pB); cb.cross(ab).normalize();
        normals[i]=normals[i+3]=normals[i+6]=cb.x;
        normals[i+1]=normals[i+4]=normals[i+7]=cb.y;
        normals[i+2]=normals[i+5]=normals[i+8]=cb.z;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geometry.scale(0.006, 0.006, 0.006);
    return geometry;
}`;

const SCENES = {
    'bvh-csg-union': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, ADDITION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, ADDITION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const brush1 = new Brush(${g('SphereGeometry(1, 32, 16)')}, mat);
        const brush2 = new Brush(${g('BoxGeometry(1.2, 1.2, 1.2)')}, mat);
        brush2.position.set(0.5, 0, 0);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        const evaluator = new Evaluator();
        const result = new Brush();
        evaluator.evaluate(brush1, brush2, ADDITION, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, mat));`;
        },
    },
    'bvh-csg-subtract': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, SUBTRACTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, SUBTRACTION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const brush1 = new Brush(${g('SphereGeometry(1, 32, 16)')}, mat);
        const brush2 = new Brush(${g('BoxGeometry(1.2, 1.2, 1.2)')}, mat);
        brush2.position.set(0.5, 0, 0);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        const evaluator = new Evaluator();
        const result = new Brush();
        evaluator.evaluate(brush1, brush2, SUBTRACTION, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, mat));`;
        },
    },
    'bvh-csg-intersect': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, INTERSECTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, INTERSECTION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const brush1 = new Brush(${g('SphereGeometry(1, 32, 16)')}, mat);
        const brush2 = new Brush(${g('BoxGeometry(1.2, 1.2, 1.2)')}, mat);
        brush2.position.set(0.5, 0, 0);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        const evaluator = new Evaluator();
        const result = new Brush();
        evaluator.evaluate(brush1, brush2, INTERSECTION, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, mat));`;
        },
    },
    'bvh-csg-difference': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, DIFFERENCE } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, DIFFERENCE } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const brush1 = new Brush(${g('SphereGeometry(1, 32, 16)')}, mat);
        const brush2 = new Brush(${g('BoxGeometry(1.2, 1.2, 1.2)')}, mat);
        brush2.position.set(0.5, 0, 0);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        const evaluator = new Evaluator();
        const result = new Brush();
        evaluator.evaluate(brush1, brush2, DIFFERENCE, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, mat));`;
        },
    },

    'bvh-csg-hollow': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, HOLLOW_INTERSECTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, HOLLOW_INTERSECTION } from 'three-bvh-csg';`,
        body: ({ threers }) => `
        ${TRIANGLE_SOUP_FN}
        const evaluator = new Evaluator();
        evaluator.attributes = ['position', 'normal'];
        evaluator.useGroups = false;
        const soupMat = new THREE.MeshStandardMaterial({ color: 0x88aacc, side: THREE.DoubleSide, roughness: 0.3 });
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const brushA = new Brush(${threers ? 'makeTriangleSoupGeometry(THREE)' : 'makeTriangleSoupGeometry(THREE).toNonIndexed()'}, soupMat);
        const brushB = new Brush(${threers ? 'geometryToBufferGeometry(new THREE.SphereGeometry(1, 32, 16))' : 'new THREE.SphereGeometry(1, 32, 16).toNonIndexed()'}, mat);
        brushB.position.set(0, 0.8, 0);
        brushB.scale.setScalar(1.6);
        brushA.updateMatrixWorld(true);
        brushB.updateMatrixWorld(true);
        const result = new Brush();
        evaluator.evaluate(brushA, brushB, HOLLOW_INTERSECTION, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, soupMat));`,
    },

    'bvh-csg-multimaterial': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, ADDITION, SUBTRACTION, INTERSECTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, ADDITION, SUBTRACTION, INTERSECTION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const red = new THREE.MeshStandardMaterial({ color: 0xff1744, roughness: 0.25 });
        const green = new THREE.MeshStandardMaterial({ color: 0x76ff03, roughness: 0.25 });
        const blue = new THREE.MeshStandardMaterial({ color: 0x2979ff, roughness: 0.25 });
        const evaluator = new Evaluator();
        const c1 = new Brush(${g('BoxGeometry(0.9, 6, 0.9)')}, blue);
        const c2 = new Brush(${g('BoxGeometry(6, 0.9, 0.9)')}, blue);
        const c3 = new Brush(${g('BoxGeometry(0.9, 0.9, 6)')}, blue);
        const sphere = new Brush(${g('SphereGeometry(1, 32, 16)')}, green);
        const box = new Brush(${g('BoxGeometry(1.5, 1.5, 1.5)')}, red);
        [c1,c2,c3,sphere,box].forEach(b => b.updateMatrixWorld(true));
        let result = evaluator.evaluate(c1, c2, ADDITION);
        result = evaluator.evaluate(result, c3, ADDITION);
        result = evaluator.evaluate(sphere, result, SUBTRACTION);
        result = evaluator.evaluate(box, result, INTERSECTION);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, result.material));`;
        },
    },

    'bvh-csg-multiop': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, SUBTRACTION, INTERSECTION, ADDITION, REVERSE_SUBTRACTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, SUBTRACTION, INTERSECTION, ADDITION, REVERSE_SUBTRACTION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const evaluator = new Evaluator();
        evaluator.attributes = ['position', 'normal'];
        const mat1 = new THREE.MeshStandardMaterial({ color: 0xfff8e1, roughness: 0.9, side: THREE.DoubleSide });
        const mat2 = new THREE.MeshStandardMaterial({ color: 0xff9800, roughness: 0.9, side: THREE.DoubleSide });
        const brush1 = new Brush(${g('IcosahedronGeometry(1, 1)')}, mat1);
        const brush2 = new Brush(${g('CylinderGeometry(0.5, 0.5, 2.5, 24)')}, mat2);
        brush1.rotation.set(0.35, 0.55, 0.2);
        brush2.rotation.set(-0.25, -0.4, -0.65);
        brush1.updateMatrixWorld(true);
        brush2.updateMatrixWorld(true);
        const r1 = new Brush(), r2 = new Brush(), r3 = new Brush(), r4 = new Brush();
        evaluator.evaluate(brush1, brush2, [SUBTRACTION, INTERSECTION, ADDITION, REVERSE_SUBTRACTION], [r1, r2, r3, r4]);
        const slots = [[-1.8,0,1.8],[1.8,0,1.8],[-1.8,0,-1.8],[1.8,0,-1.8]];
        let triCount = 0;
        for (let i = 0; i < 4; i++) {
            const r = [r1,r2,r3,r4][i];
            r.geometry.computeVertexNormals();
            const m = new THREE.Mesh(r.geometry, r.material);
            m.position.set(slots[i][0], slots[i][1], slots[i][2]);
            scene.add(m);
            const dr = r.geometry.drawRange;
            triCount += ((dr.count !== Infinity ? dr.count : r.geometry.attributes.position.count) / 3) | 0;
        }
        var result = r1;`;
        },
    },

    'bvh-csg-complex': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, ADDITION, SUBTRACTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, ADDITION, SUBTRACTION } from 'three-bvh-csg';`,
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const evaluator = new Evaluator();
        const knotMat = new THREE.MeshStandardMaterial({ color: 0x4dd0e1, roughness: 0.45, metalness: 0.1 });
        const sphereMat = new THREE.MeshStandardMaterial({ color: 0xffab40, roughness: 0.45, metalness: 0.1 });
        const knot = new Brush(${g('TorusKnotGeometry(0.8, 0.25, 64, 8)')}, knotMat);
        knot.position.y = -0.2;
        const specs = [[0.35,0.55,0.2,0.18],[-0.45,0.15,-0.25,0.14],[0.1,-0.35,0.4,0.16]];
        let merged = null;
        for (const s of specs) {
            const b = new Brush(${g('SphereGeometry(1, 20, 12)')}, sphereMat);
            b.position.set(s[0], s[1], s[2]);
            b.scale.setScalar(s[3]);
            b.updateMatrixWorld(true);
            merged = merged ? evaluator.evaluate(merged, b, ADDITION) : b;
        }
        knot.updateMatrixWorld(true);
        const result = new Brush();
        evaluator.evaluate(knot, merged, SUBTRACTION, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, knotMat));`;
        },
    },

    'bvh-csg-hierarchy': {
        threersImports: `import { installBvhCsg, Brush, Evaluator, Operation, OperationGroup, ADDITION, SUBTRACTION } from '/web/bvh-csg-addon.js';`,
        threeImports: `import { Brush, Evaluator, Operation, OperationGroup, ADDITION, SUBTRACTION } from 'three-bvh-csg';`,
        camera: { pos: [-3.8, 2.9, 1.25], target: [0, 0, 0] },
        body: ({ threers }) => {
            const g = (geo) => threers ? `geometryToBufferGeometry(new THREE.${geo})` : `new THREE.${geo}.toNonIndexed()`;
            return `
        const mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });
        const evaluator = new Evaluator();
        evaluator.useGroups = false;
        const root = new Operation(${g('BoxGeometry(4, 2.5, 2.5)')}, mat);
        root.operation = ADDITION;
        const cut = new Operation(${g('BoxGeometry(3.6, 2.1, 2.1)')}, mat);
        cut.operation = SUBTRACTION;
        const sphere = new Operation(${g('SphereGeometry(0.55, 24, 12)')}, mat);
        sphere.operation = ADDITION;
        sphere.position.set(-1.1, 0.2, 1.35);
        const windowGroup = new OperationGroup();
        const winCut = new Operation(${g('BoxGeometry(1.2, 1.0, 0.5)')}, mat);
        winCut.operation = SUBTRACTION;
        const winFrame = new Operation(${g('BoxGeometry(1.2, 1.0, 0.12)')}, mat);
        winFrame.operation = ADDITION;
        windowGroup.add(winCut, winFrame);
        windowGroup.position.set(0.8, 0.15, 1.35);
        root.add(cut, sphere, windowGroup);
        root.updateMatrixWorld(true);
        const result = new Brush();
        evaluator.evaluateHierarchy(root, result);
        result.geometry.computeVertexNormals();
        scene.add(new THREE.Mesh(result.geometry, mat));`;
        },
    },
};

function shell(side, slug, scene) {
    const threers = side === 'threers';
    const body = scene.body({ threers }).trim();
    const cam = scene.camera || { pos: [0, 0, 5], target: [0, 0, 0] };
    const camPos = cam.pos.join(', ');
    const camTarget = cam.target.join(', ');
    const triCountExpr = slug === 'bvh-csg-multiop'
        ? 'String(triCount)'
        : `String(((result.geometry.drawRange.count !== Infinity ? result.geometry.drawRange.count : result.geometry.attributes.position.count) / 3) | 0)`;

    if (threers) {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
${STYLE}
</head>
<body>
<canvas id="c" width="800" height="600"></canvas>
<div id="err" style="display:none"></div>
<script type="module">
import THREE, { initThreers, geometryToBufferGeometry } from '/web/threejs-shim.js';
${scene.threersImports}

const errEl = document.getElementById('err');
(async () => {
    try {
        await initThreers('/web/pkg/threers_bg.wasm');
        installBvhCsg(THREE);
        const canvas = document.getElementById('c');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101418);
        scene.add(new THREE.AmbientLight(0xffffff, 0.45));
        const dl = new THREE.DirectionalLight(0xffffff, 1);
        dl.position.set(3, 5, 2);
        scene.add(dl);
        ${body}
        const cam = new THREE.PerspectiveCamera(50, 800/600, 0.1, 100);
        cam.position.set(${camPos});
        cam.lookAt(${camTarget});
        cam.updateMatrixWorld(true);
        // As the three.js twin asks.
        const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
        renderer.setSize(800, 600);
        renderer.render(scene, cam);
        await new Promise(r => requestAnimationFrame(r));
        document.body.dataset.ready = 'true';
        document.body.dataset.triCount = ${triCountExpr};
${THREERS_ORBIT}
    } catch (e) {
        errEl.style.display = 'block';
        errEl.textContent = e?.stack || String(e);
        document.body.dataset.ready = 'error';
    }
})();
</script>
</body>
</html>`;
    }

    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
${STYLE}
${IMPORT_MAP}
</head>
<body>
<canvas id="c" width="800" height="600"></canvas>
<div id="err" style="display:none"></div>
<script type="module">
import * as THREE from 'three';
import { acceleratedRaycast } from 'three-mesh-bvh';
${scene.threeImports}
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const errEl = document.getElementById('err');
(async () => {
    try {
        const canvas = document.getElementById('c');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101418);
        scene.add(new THREE.AmbientLight(0xffffff, 0.45));
        const dl = new THREE.DirectionalLight(0xffffff, 1);
        dl.position.set(3, 5, 2);
        scene.add(dl);
        ${body}
        const cam = new THREE.PerspectiveCamera(50, 800/600, 0.1, 100);
        cam.position.set(${camPos});
        cam.lookAt(${camTarget});
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(800, 600, false);
        renderer.render(scene, cam);
        await new Promise(r => requestAnimationFrame(r));
        document.body.dataset.ready = 'true';
        document.body.dataset.triCount = ${triCountExpr};
${THREEJS_ORBIT}
    } catch (e) {
        errEl.style.display = 'block';
        errEl.textContent = e?.stack || String(e);
        document.body.dataset.ready = 'error';
    }
})();
</script>
</body>
</html>`;
}

for (const slug of Object.keys(SCENES)) {
    const scene = SCENES[slug];
    fs.writeFileSync(path.join(outDir, `threers-${slug}.html`), shell('threers', slug, scene));
    fs.writeFileSync(path.join(outDir, `threejs-${slug}.html`), shell('threejs', slug, scene));
    console.log('wrote', slug);
}
