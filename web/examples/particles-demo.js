// Particles, snow and sand through the three.js shim: THREE.Particles,
// THREE.SnowCover and THREE.SandBed (threers extensions), stepped by the
// renderer at every render.
import THREE, { initThreers } from '../threejs-shim.js';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const errEl = document.getElementById('err');
const desert = params.get('scene') === 'desert';

// Reports to the dev server, if it is listening.
function beacon(p) {
    fetch('/__status?' + new URLSearchParams(p)).catch(() => {});
}
window.addEventListener('error', (e) => {
    errEl.textContent += `${e.message}\n`;
    beacon({ error: e.message });
});

document.getElementById('intro').innerHTML = desert
    ? `Sand poured from a nozzle falls as grains at a few metres a second and piles into a cone that
       stands at sand's <b>angle of repose</b>, 34°, however big it grows. Beyond it, a sheet of sand
       blown by a steady wind organises itself into <b>dunes</b> (Werner's slab model) that creep
       downwind. <b>Click the ground</b> for a puff of dust. Drag to orbit, scroll to zoom.`
    : `Snow falls at about a metre a second, carried by every eddy of the wind, and <b>settles</b>
       where it lands — on the crates, the log and the ground, found by casting rays into the scene —
       slumping off edges steeper than snow holds. A campfire burns beside it: flame and smoke rise
       because they are lighter than the air, sparks fly in arcs and bounce. <b>Click the ground</b>
       for a burst of sparks there. Drag to orbit, scroll to zoom.`;

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

    const sun = new THREE.DirectionalLight(0xfff1e0, desert ? 2.8 : 1.3);
    sun.position.set(-5, 8, 4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9, near: 0.5, far: 40 });
    scene.add(sun);

    const solid = (geometry, color, at, roughness = 0.85) => {
        const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness }));
        m.position.set(...at);
        m.castShadow = true;
        m.receiveShadow = true;
        scene.add(m);
        return m;
    };
    const ground = solid(new THREE.PlaneGeometry(80, 80), desert ? 0x9c7c55 : 0x4a5040, [0, 0, 0], 1);
    ground.rotation.x = -Math.PI / 2;

    const windEl = document.getElementById('wind');
    const rateEl = document.getElementById('rate');
    const wind = () => [Number(windEl.value), 0, Number(windEl.value) * 0.3];
    let weather = null;
    let falling = null;
    let fire = null;
    let smoke = null;
    let sparks = null;

    if (!desert) {
        scene.background = new THREE.Color(0x9aa7b8);
        scene.fog = new THREE.FogExp2(0x9aa7b8, 0.045);
        scene.add(new THREE.HemisphereLight(0xdfe8f5, 0x6a6f78, 1.0));
        solid(new THREE.BoxGeometry(1.4, 0.9, 1.0), 0x7a4a32, [-2.4, 0.45, -1.2]);
        solid(new THREE.BoxGeometry(0.9, 0.6, 0.9), 0x5a6b78, [-1.2, 0.3, -2.4]);
        const log = solid(new THREE.CylinderGeometry(0.22, 0.22, 2.2, 24), 0x5b4330, [2.2, 0.22, -1.4]);
        log.rotation.z = Math.PI / 2;
        log.rotation.y = 0.5;
        // Stones round the fire.
        for (let k = 0; k < 9; k++) {
            const a = (k / 9) * Math.PI * 2;
            solid(new THREE.SphereGeometry(0.12, 12, 8), 0x6b6b6b, [Math.cos(a) * 0.45, 0.06, 1 + Math.sin(a) * 0.45]);
        }
        const cover = new THREE.SnowCover({ size: [20, 20], resolution: 300, castFrom: 20 });
        // The wind drifts what lies, too: scoured from the open, banked in
        // the lee of the crates.
        cover.drift('+x', 30000);
        scene.add(cover);
        // What you see falling: flakes placed by the GPU round the camera,
        // free on the CPU however many.
        falling = new THREE.Weather('snow', Number(params.get('flakes') ?? 60000), {
            halfWidth: 14, halfDepth: 14, height: 8, wind: wind(),
        });
        scene.add(falling);
        // What lands: fewer flakes, simulated, each carrying more snow.
        weather = new THREE.Particles('snowfall', {
            size: [22, 22], height: 6, rate: Number(rateEl.value), depthPerSecond: 0.0015,
            wind: wind(), maxParticles: 40000, settleOn: cover,
        });
        // Upwind of the yard by the drift on the way down.
        weather.position.set(-wind()[0] * 6, 0, -wind()[2] * 6);
        scene.add(weather);
        fire = new THREE.Particles('fire', { radius: 0.28 });
        fire.position.set(0, 0.05, 1);
        scene.add(fire);
        const embers = new THREE.Particles('embers', { radius: 0.22 });
        embers.position.set(0, 0.05, 1);
        scene.add(embers);
        smoke = new THREE.Particles('smoke', { radius: 0.22, wind: wind() });
        smoke.position.set(0, 0.9, 1);
        scene.add(smoke);
        sparks = new THREE.Particles('sparks');
        sparks.position.set(0, 0.25, 1);
        scene.add(sparks);
        const haze = new THREE.Particles('heatHaze', { radius: 0.2 });
        haze.position.set(0, 0.5, 1);
        scene.add(haze);
        // The fire lights the camp, flickering with its flames.
        const glow = new THREE.PointLight(0xff8a3c, 6, 10, 2);
        glow.position.set(0, 0.6, 1);
        scene.add(glow);
        fire.lights(glow, 7);
    } else {
        document.getElementById('ratel').style.display = 'none';
        scene.background = new THREE.Color(0xa9c3dd);
        scene.fog = new THREE.FogExp2(0xc8b89c, 0.02);
        scene.add(new THREE.HemisphereLight(0xbcd2ff, 0x6b5a44, 0.6));
        const bed = new THREE.SandBed({ size: [4, 4], center: [0, 2.5], resolution: 220, castFrom: 10, ripples: 0 });
        scene.add(bed);
        weather = new THREE.Particles('sandStream', { rate: 5000, flow: 0.002, settleOn: bed });
        weather.position.set(0, 1.8, 2.5);
        scene.add(weather);
        solid(new THREE.CylinderGeometry(0.06, 0.035, 0.14, 24, 1, true), 0x8a8f99, [0, 1.87, 2.5], 0.35);
        const dunes = new THREE.SandBed({ size: [24, 24], center: [2, -12], resolution: 128, castFrom: 10 });
        dunes.spread(0.12, 0.12);
        // Formed already, then still creeping downwind.
        dunes.shape('+x', 4000000);
        dunes.blow('+x', 30000);
        scene.add(dunes);
    }

    // For timing from the console (and the test driver).
    globalThis.__particles = { scene, renderer, sun };
    const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 200);
    camera.position.set(...(desert ? [5.5, 4.5, 7.0] : [4.5, 3.2, 7.5]));
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.set(...(desert ? [0, 0.4, -3] : [-0.5, 0.6, -0.5]));
    controls.enableDamping = true;
    controls.update();

    // Click the ground: sparks (winter) or dust (desert) where it lands.
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => {
        if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
        const r = canvas.getBoundingClientRect();
        ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        raycaster.setFromCamera(ndc, camera);
        const hit = raycaster.intersectObject(ground, false)[0];
        if (!hit) return;
        const puff = new THREE.Particles(desert ? 'dust' : 'sparks', desert ? { count: 160 } : {});
        puff.position.copy(hit.point);
        scene.add(puff);
        if (!desert) {
            // A burst rather than a stream, then gone.
            setTimeout(() => puff.setRate(0), 250);
        }
        setTimeout(() => scene.remove(puff), 5000);
    });

    windEl.addEventListener('input', () => {
        document.getElementById('windv').textContent = `${windEl.value} m/s`;
        weather?.setWind(...wind());
        falling?.setWind(...wind());
        smoke?.setWind(...wind());
    });
    rateEl.addEventListener('input', () => {
        document.getElementById('ratev').textContent = `${rateEl.value}/s`;
        weather?.setRate(Number(rateEl.value));
    });
    const shadowsEl = document.getElementById('shadows');
    shadowsEl.addEventListener('change', () => { sun.castShadow = shadowsEl.checked; });

    statusEl.textContent = String(renderer._w.backend?.() ?? renderer._w.backend ?? 'running');
    const stopAfter = Number(params.get('frames') ?? 0);
    const start = performance.now();
    let frames = 0, last = start, drawn = 0;
    function frame(now) {
        controls.update();
        renderer.render(scene, camera);
        frames++;
        drawn++;
        if (drawn === 1) beacon({ first: Math.round(now - start), backend: statusEl.textContent });
        if (stopAfter && drawn >= stopAfter) {
            const still = new Image();
            still.src = canvas.toDataURL('image/png');
            still.style.cssText = 'width:100%;display:block;border-radius:10px';
            canvas.replaceWith(still);
            still.onload = () => beacon({ done: drawn, backend: statusEl.textContent, particles: weather?.count ?? 0 });
            return;
        }
        if (now - last > 1000) {
            const fps = (frames * 1000) / (now - last);
            frames = 0;
            last = now;
            const n = (weather?.count ?? 0) + (fire?.count ?? 0) + (smoke?.count ?? 0) + (sparks?.count ?? 0);
            const gpu = falling ? ` + ${falling.options.count.toLocaleString()} on the GPU` : '';
            infoEl.textContent = `${fps.toFixed(0)} fps · ${n.toLocaleString()} particles${gpu}`;
            const t = (now - start) / 1000;
            if (t > 3 && t < 60 && Math.floor(t) % 5 === 0) {
                beacon({ ok: 1, t: t.toFixed(0), fps: fps.toFixed(1), particles: n, backend: statusEl.textContent });
            }
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = String(e?.stack ?? e);
    beacon({ error: String(e?.message ?? e) });
});
