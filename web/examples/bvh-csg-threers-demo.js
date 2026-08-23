import THREE, { initThreers, geometryToBufferGeometry } from '/web/threejs-shim.js';
import {
    isBvhCsgEnabled,
    installBvhCsg,
} from '/web/bvh-csg-addon.js';
import {
    SCENARIOS,
    getOpsForScenario,
    evaluateScenario,
} from '/web/examples/bvh-csg-scenes.js';

const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');
const scenariosEl = document.getElementById('scenarios');
const opsEl = document.getElementById('ops');

let currentScenario = 'simple';
let currentOp = 0;
let resultMeshes = [];
/** @type {import('/web/threejs-shim.js').Scene | null} */
let scene = null;
/** @type {import('/web/threejs-shim.js').MeshStandardMaterial | null} */
let mat = null;

function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
}

function showError(e) {
    if (errEl) {
        errEl.style.display = 'block';
        errEl.textContent = e?.stack || String(e);
    }
    console.error(e);
}

function clearResults(scene) {
    for (const m of resultMeshes) scene.remove(m);
    resultMeshes = [];
}

function showResult(scene, mat, payload) {
    clearResults(scene);
    if (payload.results) {
        for (const { mesh, position } of payload.results) {
            const m = new THREE.Mesh(mesh.geometry, mesh.material || mat);
            m.position.set(position[0], position[1], position[2]);
            scene.add(m);
            resultMeshes.push(m);
        }
    } else {
        const m = new THREE.Mesh(payload.result.geometry, payload.result.material || mat);
        scene.add(m);
        resultMeshes.push(m);
    }
    setStatus(`${payload.label} · ${payload.triCount} triangles`);
}

function rebuildOpButtons() {
    if (!opsEl) return;
    opsEl.innerHTML = '';
    const ops = getOpsForScenario(currentScenario);
    if (!ops.length) {
        const hint = document.createElement('span');
        hint.style.color = 'var(--muted)';
        hint.style.fontSize = '13px';
        hint.textContent = 'fixed scenario';
        opsEl.appendChild(hint);
        return;
    }
    for (const op of ops) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = op.label;
        btn.dataset.op = op.id;
        btn.addEventListener('click', () => {
            currentOp = op.value;
            for (const b of opsEl.querySelectorAll('button')) {
                b.classList.toggle('active', b === btn);
            }
            runScenario();
        });
        opsEl.appendChild(btn);
    }
    opsEl.querySelector('button')?.classList.add('active');
}

function runScenario() {
    if (!scene || !mat) return;
    const payload = evaluateScenario(THREE, currentScenario, currentOp);
    showResult(scene, mat, payload);
}

// Wrapped in an async IIFE rather than using top-level `await`, which is
// ES2022 module syntax and needs Safari 15. A syntax error is not recoverable —
// the module never loads and the page reports one error a long way from the
// cause — so the cost of the wrapper is worth the floor it buys.
(async () => {
    try {
        if (!isBvhCsgEnabled()) {
            throw new Error('bvh-csg is disabled. Rebuild with: BVH_CSG=1 web/build.sh');
        }

        await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
        installBvhCsg(THREE);

        const canvas = document.getElementById('canvas');
        const preview = canvas.parentElement;

        function resizeCanvas() {
            const dpr = Math.min(devicePixelRatio || 1, 2);
            const w = Math.floor(preview.clientWidth * dpr);
            const h = Math.floor(preview.clientHeight * dpr);
            canvas.width = w;
            canvas.height = h;
            return { w, h };
        }

        let { w, h } = resizeCanvas();
        const renderer = await THREE.WebGLRenderer.create(canvas);
        renderer.setSize(w, h, false);

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101418);
        scene.add(new THREE.AmbientLight(0xffffff, 0.45));
        const dl = new THREE.DirectionalLight(0xffffff, 1);
        dl.position.set(3, 5, 2);
        scene.add(dl);

        mat = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.45, metalness: 0.1 });

        if (scenariosEl) {
            for (const sc of SCENARIOS) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = sc.label;
                btn.dataset.scenario = sc.id;
                btn.addEventListener('click', () => {
                    currentScenario = sc.id;
                    const ops = getOpsForScenario(currentScenario);
                    currentOp = ops[0]?.value ?? 0;
                    for (const b of scenariosEl.querySelectorAll('button')) {
                        b.classList.toggle('active', b === btn);
                    }
                    rebuildOpButtons();
                    runScenario();
                });
                scenariosEl.appendChild(btn);
            }
            scenariosEl.querySelector('[data-scenario="simple"]')?.classList.add('active');
        }

        rebuildOpButtons();

        const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
        camera.position.set(2.5, 2, 3.5);
        camera.lookAt(0, 0, 0);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        addEventListener('resize', () => {
            ({ w, h } = resizeCanvas());
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });

        runScenario();

        function frame() {
            requestAnimationFrame(frame);
            controls.update();
            renderer.render(scene, camera);
        }
        frame();
    } catch (e) {
        setStatus('error');
        showError(e);
    }
})();
