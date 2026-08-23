// Live Kirigami Expanded Miura viewer — wasm geometry, wgpu renderer.

import THREE, { initThreers } from '../threejs-shim.js';
import { WebGeometry } from '../pkg/threers.js';

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');
const netEl = document.getElementById('net');
const controlsEl = document.getElementById('controls');

/** UI variant index (slot 5 = joined developed net). */
const VARIANTS = [
    { v: 0, label: 'planar', nx: 5, ny: 6 },
    { v: 1, label: 'steep', nx: 5, ny: 6 },
    { v: 2, label: 'cylinder', nx: 5, ny: 8 },
    { v: 3, label: 'saddle', nx: 5, ny: 8 },
    { v: 4, label: 'cells', nx: 5, ny: 6 },
    { v: 5, label: 'net', nx: 4, ny: 6, isNet: true },
    { v: 6, label: 'shallow', nx: 5, ny: 6 },
    { v: 7, label: 'arch', nx: 5, ny: 8 },
    { v: 8, label: 'dome', nx: 5, ny: 8 },
    { v: 9, label: 'ripple', nx: 5, ny: 8 },
    { v: 10, label: 'wing', nx: 5, ny: 8 },
    { v: 11, label: 'wide', nx: 5, ny: 6 },
    { v: 12, label: 'fine', nx: 5, ny: 6 },
    { v: 13, label: 'valley', nx: 5, ny: 8 },
    { v: 14, label: 'tiled', nx: 3, ny: 6 },
    { v: 15, label: 'stacked', nx: 4, ny: 6 },
    { v: 16, label: 'gyroid', nx: 4, ny: 6 },
    { v: 17, label: 'octet', nx: 5, ny: 8 },
    { v: 18, label: 'honeycomb', nx: 4, ny: 6 },
    { v: 19, label: 'helix', nx: 5, ny: 8 },
    { v: 20, label: 'gradient', nx: 5, ny: 8 },
];

function wrapGeom(webGeom) {
    return { _w: webGeom };
}

function buildControls(onSelect) {
    for (const item of VARIANTS) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.v = String(item.v);
        btn.textContent = item.label;
        btn.addEventListener('click', () => onSelect(item.v));
        controlsEl.appendChild(btn);
    }
}

async function main() {
    await initThreers({ module_or_path: new URL('../pkg/threers_bg.wasm', import.meta.url) });
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    renderer.setSize(canvas.width, canvas.height, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1018);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xfff4e6, 2.4);
    key.position.set(-2, 4, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x88a0ff, 0.8);
    fill.position.set(3, 1, 2);
    scene.add(fill);

    const camera = new THREE.PerspectiveCamera(32, canvas.width / canvas.height, 1, 4000);
    camera.position.set(0, 90, 420);
    camera.lookAt(0, 0, 0);

    const group = new THREE.Group();
    scene.add(group);

    const mat = new THREE.MeshStandardMaterial({
        color: 0xc8d4c4,
        metalness: 0.28,
        roughness: 0.46,
        side: 2,
    });

    let mesh = null;
    let building = false;

    function show(variant) {
        if (building) return;
        const spec = VARIANTS.find((x) => x.v === variant) ?? VARIANTS[0];
        document.querySelectorAll('#controls button').forEach((b) => {
            b.classList.toggle('active', Number(b.dataset.v) === variant);
        });
        const slow = spec.v === 16 || spec.v === 17 || spec.v === 18;
        statusEl.textContent = slow ? `building ${spec.label} (lattice core)…` : `loading ${spec.label}…`;

        building = true;
        requestAnimationFrame(() => {
            try {
                const geom = wrapGeom(WebGeometry.kirigami(variant, spec.nx, spec.ny, 1.2));
                if (mesh) scene.remove(mesh);
                mesh = new THREE.Mesh(geom, mat);
                mesh.rotation.x = spec.isNet ? -1.05 : -0.55;
                mesh.rotation.y = spec.isNet ? 0.1 : 0.62;
                group.add(mesh);
                const svg = WebGeometry.kirigamiNetSvg(variant, spec.nx, spec.ny);
                netEl.innerHTML = svg;
                statusEl.textContent = `${spec.label} · wasm ready`;
            } catch (e) {
                errEl.textContent = e && e.stack ? e.stack : String(e);
                statusEl.textContent = 'failed';
            } finally {
                building = false;
            }
        });
    }

    buildControls(show);
    show(0);

    function frame(t) {
        group.rotation.y = t * 0.00022;
        renderer.render(scene, camera);
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = e && e.stack ? e.stack : String(e);
    statusEl.textContent = 'failed';
});
