// Cuboct metamaterial viewer — continuum lattice + discrete assembly (wasm).

import THREE, { initThreers } from '../threejs-shim.js';
import { WebGeometry } from '../pkg/threers.js';

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');
const netEl = document.getElementById('net');
const variantEl = document.getElementById('variant');
const coloredEl = document.getElementById('colored');

function wrapGeom(webGeom) {
    return { _w: webGeom };
}

// Assembly is fast and always has positions; continuum lattice is heavier.
let mode = 'assembly';

async function main() {
    await initThreers({ module_or_path: new URL('../pkg/threers_bg.wasm', import.meta.url) });
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    renderer.setSize(canvas.width, canvas.height, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1018);
    scene.add(new THREE.AmbientLight(0xffffff, 0.42));
    const key = new THREE.DirectionalLight(0xfff4e6, 2.4);
    key.position.set(-2, 4, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x88a0ff, 0.75);
    fill.position.set(3, 1, 2);
    scene.add(fill);

    const camera = new THREE.PerspectiveCamera(32, canvas.width / canvas.height, 1, 4000);
    camera.position.set(0, 60, 380);
    camera.lookAt(0, 0, 0);

    const group = new THREE.Group();
    scene.add(group);

    const mat = new THREE.MeshStandardMaterial({
        color: 0xd4955a,
        metalness: 0.14,
        roughness: 0.44,
    });

    let mesh = null;
    let building = false;

    function removeMesh() {
        if (!mesh) return;
        scene.remove(mesh);
        group.remove(mesh);
        mesh = null;
    }

    function show() {
        if (building) return;
        const v = Number(variantEl.value);
        const colored = coloredEl.checked;
        const label = mode === 'lattice' ? 'continuum' : 'assembly';
        statusEl.textContent = `building ${label}…`;
        building = true;

        requestAnimationFrame(() => {
            try {
                let geom;
                if (mode === 'lattice') {
                    const shape = v === 0 ? 0 : 0.18;
                    geom = wrapGeom(WebGeometry.cuboctLattice(v, 16, 2, 18, shape));
                    mat.color.setHex(0xc8a878);
                } else {
                    geom = wrapGeom(WebGeometry.cuboctAssembly(v, 10, 2, 0.28, colored));
                    mat.color.setHex(colored ? 0xffffff : 0xd4955a);
                }

                removeMesh();
                mesh = new THREE.Mesh(geom, mat);
                if (!mesh._w) {
                    throw new Error('geometry has no GPU mesh — build may have returned empty');
                }
                mesh.rotation.x = -0.45;
                mesh.rotation.y = 0.55;
                group.add(mesh);

                const svg = WebGeometry.cuboctFaceSvg(v, 75, v === 0 ? 0 : 0.18);
                netEl.innerHTML = svg;
                errEl.textContent = '';
                statusEl.textContent = `${label} · wasm ready`;
            } catch (e) {
                errEl.textContent = e && e.stack ? e.stack : String(e);
                statusEl.textContent = 'failed';
            } finally {
                building = false;
            }
        });
    }

    document.querySelectorAll('[data-mode]').forEach((btn) => {
        btn.addEventListener('click', () => {
            mode = btn.dataset.mode;
            document.querySelectorAll('[data-mode]').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            show();
        });
    });
    variantEl.addEventListener('change', show);
    coloredEl.addEventListener('change', show);

    // Sync button highlight with default mode.
    document.querySelectorAll('[data-mode]').forEach((b) => {
        b.classList.toggle('active', b.dataset.mode === mode);
    });
    show();

    function frame(t) {
        group.rotation.y = t * 0.00022;
        try {
            renderer.render(scene, camera);
        } catch (e) {
            errEl.textContent = e && e.stack ? e.stack : String(e);
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    statusEl.textContent = 'error';
    errEl.textContent = e && e.stack ? e.stack : String(e);
});
