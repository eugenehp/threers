// Dunes through the three.js shim: a sand bed grown into a dune field
// (THREE.SandBed.dunes), shaded after Journey's sand, under a desert sky at
// four times of day, sand blowing off the crests (THREE.Spindrift).
import THREE, { initThreers } from '../threejs-shim.js';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const errEl = document.getElementById('err');

function beacon(p) {
    fetch('/__status?' + new URLSearchParams(p)).catch(() => {});
}
window.addEventListener('error', (e) => {
    errEl.textContent += `${e.message}\n`;
    beacon({ error: e.message });
});

document.getElementById('intro').innerHTML = `Nine hundred metres of dunes grown in the browser: a network of curving
    crests, drawn out across the wind, then <b>worked by the wind</b> — sand lifted off the windward slopes and dropped
    in the lee — so every dune has a long gentle back and a <b>slip face at 34°</b>. The sand is shaded after
    <i>Journey</i>'s: grains, ripples across the wind and along the slopes, a sheen stretched along the wind, glitter,
    sand streaking up the windward faces; and sand smokes off the crests. The haze over it meets the sky's horizon
    without a seam. Drag to look around; it flies on its own when left alone.`;

async function main() {
    if (params.get('gl') === '1' && 'gpu' in navigator) {
        Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
    }
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
    const canvas = document.getElementById('c');
    const dpr = Number(params.get('dpr') ?? Math.min(window.devicePixelRatio || 1, 2));
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientWidth * dpr * 9 / 16);
    const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(42, canvas.width / canvas.height, 0.5, 3000);
    const sky = new THREE.DesertSky('day', 2200);
    scene.add(sky);
    const hemi = new THREE.HemisphereLight(0xffffff, 0xffffff, 1);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -160, right: 160, top: 160, bottom: -160, near: 1, far: 1500 });
    sun.shadow.normalBias = 0.3;
    scene.add(sun);
    scene.add(sun.target);

    // The dunes: 900 m square on a flat base (castFrom 0: no rays cast).
    const bed = new THREE.SandBed({ size: [900, 900], center: [0, 0], resolution: Number(params.get('res') ?? 300), castFrom: 0 });
    bed.dunes({ wavelength: 110, height: 14, stretch: 2.2, shaping: Number(params.get('shaping') ?? 12) });
    scene.add(bed);
    const drift = new THREE.Spindrift(bed, { wind: [9, 0, 1.5], rate: 2500 });
    scene.add(drift);

    let current = params.get('time') ?? 'sunset';
    const times = ['dawn', 'day', 'sunset', 'night'];
    const timesEl = document.getElementById('times');
    const rgb = (c) => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.LinearSRGBColorSpace);
    function setTime(name) {
        current = name;
        const t = THREE.desertTime(name);
        sky.setTime(name);
        scene.background = rgb(t.horizon);
        scene.fog = new THREE.FogExp2(rgb(t.horizon), t.haze);
        hemi.color = rgb(t.skyLight);
        hemi.groundColor = rgb(t.groundLight);
        hemi.intensity = t.skyIntensity;
        sun.color = rgb(t.sunColor);
        sun.intensity = t.sunIntensity;
        sun.userData.toward = t.sun;
        bed.look({
            color: t.sand, shadowColor: t.shadow, ripples: 0.6, rippleWavelength: 0.35,
            streaks: 1, stylize: 0.25, fogGlow: t.glow,
        });
        for (const b of timesEl.children) b.classList.toggle('on', b.textContent === name);
    }
    for (const name of times) {
        const b = document.createElement('button');
        b.textContent = name;
        b.addEventListener('click', () => setTime(name));
        timesEl.appendChild(b);
    }
    setTime(current);

    // Flying slowly downwind over the crests, weaving; dragging takes over.
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    const flyEl = document.getElementById('fly');
    let lastTouch = -1e9;
    canvas.addEventListener('pointerdown', () => { lastTouch = performance.now(); flyEl.checked = false; });
    document.getElementById('drift').addEventListener('change', (e) => { drift.visible = e.target.checked; });
    document.getElementById('shadows').addEventListener('change', (e) => { sun.castShadow = e.target.checked; });
    const path = (t) => {
        // Round and round the repeating field: x wraps every 900 m.
        const x = ((t * 6) % 900) - 450;
        return new THREE.Vector3(x, 30 + 4 * Math.sin(t * 0.13), 60 * Math.sin(t * 0.021));
    };
    let flyTime = Number(params.get('t') ?? 40);
    const p0 = path(flyTime);
    camera.position.copy(p0);
    controls.target.copy(path(flyTime + 12)).setY(10);
    controls.update();

    globalThis.__dunes = { scene, renderer, camera, setTime };
    statusEl.textContent = String(renderer._w.backend?.() ?? renderer._w.backend ?? 'running');
    const stopAfter = Number(params.get('frames') ?? 0);
    const start = performance.now();
    let frames = 0, last = start, drawn = 0, prev = start;
    function frame(now) {
        const dt = Math.min((now - prev) / 1000, 0.1);
        prev = now;
        if (flyEl.checked) {
            flyTime += dt;
            camera.position.copy(path(flyTime));
            controls.target.copy(path(flyTime + 12)).setY(10);
        }
        controls.update();
        // The sky and the sun's shadow box go with the camera.
        sky.position.copy(camera.position);
        const s = sun.userData.toward ?? [0, 1, 0];
        const focus = controls.target;
        sun.target.position.copy(focus);
        sun.position.set(focus.x + s[0] * 600, focus.y + s[1] * 600, focus.z + s[2] * 600);
        renderer.render(scene, camera);
        frames++;
        drawn++;
        if (drawn === 1) beacon({ first: Math.round(now - start), backend: statusEl.textContent });
        if (stopAfter && drawn >= stopAfter) {
            beacon({ done: drawn, backend: statusEl.textContent });
            return;
        }
        if (now - last > 1000) {
            const fps = (frames * 1000) / (now - last);
            frames = 0;
            last = now;
            infoEl.textContent = `${fps.toFixed(0)} fps · ${current}`;
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = String(e?.stack ?? e);
    beacon({ error: String(e?.message ?? e) });
});
