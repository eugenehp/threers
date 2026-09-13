// The SVG renderer, driven the way three.js drives a renderer: build a scene,
// append `renderer.domElement`, call `render` in a loop.
//
// Worth noticing what is *not* here — there is no canvas, no adapter request
// and no `await WebGLRenderer.create(...)`. The SVG renderer runs on the CPU in
// wasm, so this page works where WebGPU is unavailable, and the frames it
// produces are resolution-independent.

// Relative path so the page works under any static server root.
import THREE, { initThreers, SVGRenderer } from '../threejs-shim.js';

const stage = document.getElementById('stage');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const errEl = document.getElementById('err');

const WIDTH = 800;
const HEIGHT = 600;

/** A scene chosen to show what vectors buy: a smooth sphere, a self-occluding
 *  knot whose faces have to sort against each other, and a two-triangle floor —
 *  the case a painter's algorithm classically gets wrong. */
function buildScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101820);

    const knot = new THREE.Mesh(
        new THREE.TorusKnotGeometry(1.1, 0.36, 140, 20, 2, 3),
        new THREE.MeshLambertMaterial({ color: 0xd4553c }),
    );
    knot.position.set(0, 0.55, 0);
    scene.add(knot);

    const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.85, 36, 22),
        new THREE.MeshLambertMaterial({ color: 0x2f6f8f }),
    );
    ball.position.set(2.3, -0.45, -1.1);
    scene.add(ball);

    const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(14, 14),
        new THREE.MeshLambertMaterial({ color: 0x2a3340 }),
    );
    floor.position.set(0, -1.35, 0);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    scene.add(new THREE.AmbientLight(0xb9c6d6, 0.35));
    scene.add(new THREE.HemisphereLight(0xcfe3f5, 0x4a4036, 0.4));

    // A directional light points from its position at the origin, so placing it
    // is how you aim it.
    const key = new THREE.DirectionalLight(0xfff2dd, 3.2);
    key.position.set(5, 7, 4);
    scene.add(key);

    const rim = new THREE.DirectionalLight(0x9ec5ff, 1.4);
    rim.position.set(-6, 2, -5);
    scene.add(rim);

    return { scene, knot };
}

async function main() {
    await initThreers();
    statusEl.textContent = 'ready';

    const { scene, knot } = buildScene();

    const camera = new THREE.PerspectiveCamera(45, WIDTH / HEIGHT, 0.1, 100);
    camera.position.set(4, 3, 6);
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    const renderer = new SVGRenderer();
    renderer.setSize(WIDTH, HEIGHT);
    stage.appendChild(renderer.domElement);

    let lastMarkup = '';
    const draw = () => {
        const t0 = performance.now();
        lastMarkup = renderer.render(scene, camera);
        const ms = performance.now() - t0;
        const kb = (lastMarkup.length / 1024).toFixed(0);
        statsEl.textContent =
            `${renderer.info.render.faces} paths · ${kb} KB · ${ms.toFixed(1)} ms `
            + `· ${(lastMarkup.match(/[CQ]/g) || []).length} curve segments`;
    };

    // ---- controls ----
    const $ = (id) => document.getElementById(id);

    $('shading').addEventListener('change', (e) => {
        renderer.setShading(Number(e.target.value));
        draw();
    });
    $('quality').addEventListener('change', (e) => {
        renderer.setQuality(e.target.value);
        // setQuality owns the curve tolerance, so keep the checkbox honest.
        $('curves').checked = e.target.value === 'high';
        draw();
    });
    $('curves').addEventListener('change', (e) => {
        renderer.setCurveTolerance(e.target.checked ? 0.15 : 0);
        draw();
    });
    $('bg').addEventListener('change', (e) => {
        renderer.setBackground(e.target.checked);
        draw();
    });
    $('clear').addEventListener('input', (e) => {
        renderer.setClearColor(e.target.value, 1);
        draw();
    });
    $('precision').addEventListener('input', (e) => {
        renderer.setPrecision(Number(e.target.value));
        draw();
    });

    $('download').addEventListener('click', () => {
        // `lastMarkup` is a complete standalone document — the same bytes the
        // native `SvgRenderer::render_to_file` writes.
        const blob = new Blob([lastMarkup], { type: 'image/svg+xml' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'threers-scene.svg';
        a.click();
        URL.revokeObjectURL(a.href);
    });

    let spinning = true;
    $('spin').addEventListener('click', (e) => {
        spinning = !spinning;
        e.target.textContent = spinning ? 'Pause' : 'Play';
        if (spinning) requestAnimationFrame(loop);
    });

    function loop() {
        if (!spinning) return;
        knot.rotation.y += 0.006;
        draw();
        requestAnimationFrame(loop);
    }

    draw();
    requestAnimationFrame(loop);
}

main().catch((e) => {
    statusEl.textContent = 'failed';
    errEl.textContent = String(e?.stack || e);
});
