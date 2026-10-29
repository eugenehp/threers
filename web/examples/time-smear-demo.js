/**
 * Time smear: render a short clip with threers, then smear it in wasm
 * (`threers::time_smear`) and play it back, re-smearing as the controls move.
 * Same scene as `cargo run --example time_smear`.
 *
 * `?gl=1` forces WebGL2. `?beacon=1` reports checks to `/__status` for
 * headless testing.
 */
import THREE, { initThreers } from '/web/threejs-shim.js';

const W = 480;
const H = 270;
const FRAMES = 160;
const FPS = 30;
const PULSE = [70, 86];

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const errEl = $('err');
const statusEl = $('status');

function beacon(data) {
    if (params.get('beacon') !== '1') return;
    fetch('/__status?' + new URLSearchParams(data)).catch(() => {});
}
function fail(e) {
    console.error(e);
    errEl.textContent += `${e?.message || e}\n`;
    statusEl.textContent = 'error';
    beacon({ error: String(e?.message || e) });
}
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));

function buildScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0.10, 0.11, 0.15);
    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(0.3, 0.8, 0.5);
    scene.add(sun);

    const wall = new THREE.Mesh(
        new THREE.PlaneGeometry(16, 9),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(0.28, 0.30, 0.36) }),
    );
    wall.position.set(0, 0, -2);
    scene.add(wall);
    const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 48, 24),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(0.95, 0.75, 0.35) }),
    );
    const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 24, 12),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(0.6, 0.95, 1.0) }),
    );
    const block = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, 0.9, 0.9),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(1.0, 0.35, 0.45) }),
    );
    block.position.set(-2.4, 1.2, 0);
    scene.add(ball, dot, block);

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
    camera.position.set(0, 0, 8);
    camera.lookAt(0, 0, 0);

    const pose = (i) => {
        const u = i / FRAMES;
        const tau = Math.PI * 2;
        ball.position.set(-4.5 + 9 * u, -0.6 + 0.5 * Math.sin(tau * u), 0);
        dot.position.set(2.6 + 0.8 * Math.cos(2 * tau * u), 1.3 + 0.8 * Math.sin(2 * tau * u), 0.5);
        block.visible = i >= PULSE[0] && i < PULSE[1];
    };
    return { scene, camera, pose };
}

/** Pixel of a frame, as [r, g, b]. */
const px = (rgba, x, y) => Array.from(rgba.subarray((y * W + x) * 4, (y * W + x) * 4 + 3));

async function main() {
    if (params.get('gl') === '1' && 'gpu' in navigator) {
        Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
    }
    const backend = 'gpu' in navigator && navigator.gpu ? 'WebGPU' : 'WebGL2';
    $('backend').textContent = `wasm · ${backend}`;
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

    const gpuCanvas = $('gpu');
    const renderer = await THREE.WebGLRenderer.create(gpuCanvas, { antialias: true });
    renderer.setSize(W, H, false);
    const { scene, camera, pose } = buildScene();

    statusEl.textContent = 'rendering…';
    const t0 = performance.now();
    const smear = await THREE.TimeSmear.capture(renderer, scene, camera, {
        width: W,
        height: H,
        frames: FRAMES,
        fps: FPS,
        update: (i) => pose(i),
        onFrame: (i) => {
            statusEl.textContent = `rendering ${i + 1}/${FRAMES}`;
            if (i % 20 === 0) beacon({ progress: i, ms: (performance.now() - t0).toFixed(0) });
        },
    });
    const renderMs = performance.now() - t0;
    let smearMs = 0;

    for (const c of [$('source'), $('out')]) { c.width = W; c.height = H; }
    const srcCtx = $('source').getContext('2d');
    const outCtx = $('out').getContext('2d');
    let frame = 0;
    let playing = params.get('frame') == null;
    if (!playing) frame = Number(params.get('frame'));

    const look = () => $('look').value;
    const gain = () => (Number($('gain').value) > 0 ? Number($('gain').value) : undefined);
    const draw = () => {
        srcCtx.putImageData(smear.imageData(frame, 'source'), 0, 0);
        outCtx.putImageData(smear.imageData(frame, look(), gain()), 0, 0);
        $('frame').value = frame;
        $('frame-out').textContent = frame;
    };
    const showStatus = () => {
        statusEl.textContent =
            `${FRAMES} frames ${W}×${H} · render ${renderMs.toFixed(0)} ms · smear ${smearMs.toFixed(0)} ms`;
    };
    const resmear = () => {
        smearMs = smear.set({
            strength: Number($('strength').value),
            reach: Number($('reach').value),
            rotation: $('rotation').value,
            workWidth: Number($('work').value),
            adaptive: $('adaptive').checked,
        });
        showStatus();
        draw();
    };

    for (const id of ['strength', 'reach', 'gain']) {
        const input = $(id);
        const out = input.nextElementSibling;
        const sync = () => { out.textContent = id === 'gain' && input.value === '0' ? 'auto' : input.value; };
        sync();
        input.addEventListener('input', () => { sync(); if (id === 'gain') draw(); });
        if (id !== 'gain') input.addEventListener('change', resmear);
    }
    for (const id of ['rotation', 'work', 'adaptive']) $(id).addEventListener('change', resmear);
    $('look').addEventListener('change', () => {
        $('out-cap').textContent = `time smear · ${$('look').selectedOptions[0].text}`;
        draw();
    });
    $('split').addEventListener('change', () => { $('source-fig').hidden = !$('split').checked; });
    $('play').addEventListener('click', () => {
        playing = !playing;
        $('play').textContent = playing ? 'pause' : 'play';
    });
    $('frame').addEventListener('input', () => {
        playing = false;
        $('play').textContent = 'play';
        frame = Number($('frame').value);
        draw();
    });
    $('out-cap').textContent = `time smear · ${$('look').selectedOptions[0].text}`;
    $('play').textContent = playing ? 'pause' : 'play';

    resmear();
    let last = performance.now();
    const tick = (now) => {
        if (playing && now - last >= 1000 / FPS) {
            last = now;
            frame = (frame + 1) % FRAMES;
            draw();
        }
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    // Checks for headless runs: the frames are real, the still backdrop is
    // untouched, and the box echoes before it appears.
    const corner = [12, H - 12];
    const boxAt = [Math.round(W * 0.296), Math.round(H * 0.319)];
    const source = (i) => smear.frame(i, 'source');
    const faithful = smear.frame(40, 'faithful');
    const echo = smear.frame(60, 'echo');
    const mean = source(40).reduce((a, v) => a + v, 0) / source(40).length;
    beacon({
        done: 1,
        backend,
        renderMs: renderMs.toFixed(0),
        smearMs: smearMs.toFixed(0),
        mean: mean.toFixed(1),
        corner: px(source(40), ...corner).join(','),
        cornerSmeared: px(faithful, ...corner).join(','),
        box78: px(source(78), ...boxAt).join(','),
        box60: px(source(60), ...boxAt).join(','),
        echo60: px(echo, ...boxAt).join(','),
    });
}

main().catch(fail);
