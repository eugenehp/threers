// Fur and hair through the three.js shim: THREE.Fur and THREE.Hair (threers
// extensions), animated by the renderer at every render.
import THREE, { initThreers } from '../threejs-shim.js';

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const errEl = document.getElementById('err');

// Reports to the dev server, if it is listening (see the verification notes).
function beacon(params) {
    fetch('/__status?' + new URLSearchParams(params)).catch(() => {});
}
window.addEventListener('error', (e) => {
    errEl.textContent += `${e.message}\n`;
    beacon({ error: e.message });
});

// `?quality=low`: a small canvas and fewer strands, for a software renderer.
const low = new URLSearchParams(location.search).get('quality') === 'low';
const q = (full, lite) => (low ? lite : full);
// `?frames=N`: draw N frames and stop (to capture a still), saying so.
const params = new URLSearchParams(location.search);
const stopAfter = Number(params.get('frames') ?? 0);
// `?parts=fur,hair,ring`: draw only these (all by default).
const parts = new Set((params.get('parts') ?? 'fur,hair,ring').split(','));

async function main() {
    // `?gl=1`: WebGL2 even where WebGPU exists, to compare.
    if (params.get('gl') === '1' && 'gpu' in navigator) {
        Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
    }
    const marks = { start: performance.now() };
    const mark = (k) => { marks[k] = Math.round(performance.now() - marks.start); };
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
    mark('wasm');
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    // Twice the displayed size (or the display's own pixel ratio, if more):
    // the browser's downscale then averages each pixel's 2×2, which is what
    // keeps strands finer than a pixel from beating into moiré. (`?dpr=1`
    // for one sample a CSS pixel.)
    const dpr = Number(params.get('dpr') ?? Math.max(2, Math.min(window.devicePixelRatio || 1, 2)));
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientWidth * dpr * 9 / 16);
    if (low) {
        canvas.width = Number(params.get('w') ?? 640);
        canvas.height = Math.round(canvas.width * 9 / 16);
    }
    canvas.addEventListener('webglcontextlost', () => beacon({ error: 'webgl context lost' }));
    const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
    renderer.shadowMap.enabled = true;
    mark('renderer');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14161d);

    scene.add(new THREE.HemisphereLight(0xbcd0ff, 0x2a2018, 0.9));
    const sun = new THREE.DirectionalLight(0xfff1dc, 2.6);
    sun.position.set(4, 8, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 30 });
    scene.add(sun);

    const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(30, 30),
        new THREE.MeshStandardMaterial({ color: 0x3a3f4a, roughness: 0.95 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Built from the controls; rebuilt when the wind changes (a fur's look
    // is fixed when it is made).
    let objects = [];
    let swinging = null;
    // What the pointer can stroke: an invisible shape over each coat (drawn
    // with no colour and no depth, so only the ray sees it) and its fur.
    let targets = [];
    const proxyMaterial = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
    const proxy = (geometry, parent, at) => {
        const m = new THREE.Mesh(geometry, proxyMaterial);
        if (at) m.position.copy(at);
        parent.add(m);
        return m;
    };
    function build(wind) {
        for (const o of objects) scene.remove(o);
        objects = [];
        targets = [];

        // A swinging ball of short, dense fur.
        if (parts.has('fur')) {
        const pivot = new THREE.Object3D();
        pivot.position.set(-3, 3.6, 0);
        const ball = new THREE.Fur(new THREE.SphereGeometry(0.75, 96, 48), {
            length: 0.28, density: 140, thickness: 0.3, clumping: 0.45,
            gravity: [0, -0.5, 0], rootColor: 0x2a1a0e, tipColor: 0xe0b070,
            wind: [wind, 0, wind * 0.3], windFrequency: 0.6, flutter: 0.08,
            stiffness: 40, damping: 0.35, shells: q(32, Number(params.get('shells') ?? 16)),
        });
        ball.position.set(0, -2.1, 0);
        pivot.add(ball);
        targets.push({ fur: ball, proxy: proxy(new THREE.SphereGeometry(1.0, 24, 12), pivot, ball.position) });
        scene.add(pivot);
        objects.push(pivot);
        swinging = pivot;
        }

        // A head of long hair: the scalp, and strands hanging from its upper part.
        if (parts.has('hair')) {
        const scalpGeometry = new THREE.SphereGeometry(0.7, 64, 32);
        const pos = scalpGeometry.attributes.position;
        const paint = new Float32Array(pos.count * 4);
        for (let i = 0; i < pos.count; i++) {
            const y = pos.getY(i) / 0.7;
            paint.set([1, 1, 1, Math.min(1, Math.max(0, (y + 0.15) * 3))], i * 4);
        }
        scalpGeometry.setAttribute('color', new THREE.BufferAttribute(paint, 4));
        const head = new THREE.Mesh(scalpGeometry,
            new THREE.MeshStandardMaterial({ color: 0x8a6450, roughness: 0.7 }));
        head.position.set(0, 1.9, 0);
        head.castShadow = true;
        head.receiveShadow = true;
        scene.add(head);
        objects.push(head);
        const hair = new THREE.Hair(scalpGeometry, {
            count: q(16000, 5000), segments: q(12, 6), width: q(0.004, 0.008),
            length: 1.0, gravity: [0, -3, 0], taper: 0.3, curl: 0.15,
            clumping: 0.35, clumpSize: 3, lengthVariation: 0.2,
            rootColor: 0x1e1008, tipColor: 0x9a6638,
            wind: [wind, 0, wind * 0.3], windFrequency: 0.4, flutter: 0.05,
        });
        hair.position.copy(head.position);
        scene.add(hair);
        objects.push(hair);
        // The hair hangs below the scalp: a taller shape over it.
        const hairProxy = proxy(new THREE.SphereGeometry(1.0, 24, 12), scene,
            new THREE.Vector3(head.position.x, head.position.y - 0.35, head.position.z));
        hairProxy.scale.set(1, 1.35, 1);
        objects.push(hairProxy);
        targets.push({ fur: hair, proxy: hairProxy });
        }

        // Curly, clumped fur on a torus.
        if (parts.has('ring')) {
        const ring = new THREE.Fur(new THREE.TorusGeometry(0.75, 0.32, 48, 128), {
            length: 0.4, density: 90, thickness: 0.35, taper: 0.05, curl: 1.5,
            clumping: 0.7, gravity: [0, -0.4, 0], shells: q(32, 16),
            rootColor: 0x1c1c2a, tipColor: 0x9fb4e8,
            wind: [wind, 0, wind * 0.3], windFrequency: 0.6, flutter: 0.1,
        });
        ring.position.set(3, 1.2, 0);
        ring.rotation.x = 1.1;
        scene.add(ring);
        objects.push(ring);
        const ringProxy = proxy(new THREE.TorusGeometry(0.75, 0.7, 16, 48), scene, ring.position);
        ringProxy.rotation.x = 1.1;
        objects.push(ringProxy);
        targets.push({ fur: ring, proxy: ringProxy });
        }
    }

    const camera = new THREE.PerspectiveCamera(42, canvas.width / canvas.height, 0.1, 100);
    camera.position.set(0, 3.6, 8.5);
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.set(0, 1.4, 0);
    controls.enableDamping = true;
    controls.update();

    // Stroke the fur: move the pointer over a coat and the strands under it
    // are pushed along with it and parted, then spring back. (Drag orbits.)
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let lastHit = null; // { fur, point, time }
    // On the window, in the capture phase: the orbit controls take the
    // canvas's own pointer events.
    window.addEventListener('pointermove', (e) => {
        if (e.target !== canvas) {
            return;
        }
        if (e.buttons !== 0) {
            lastHit = null;
            return;
        }
        const r = canvas.getBoundingClientRect();
        ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        raycaster.setFromCamera(ndc, camera);
        const hit = raycaster.intersectObjects(targets.map((t) => t.proxy), false)[0];
        const target = hit && targets.find((t) => t.proxy === hit.object);
        const now = performance.now();
        if (!target) {
            lastHit = null;
            return;
        }
        // The hand's speed over the coat pushes; its presence parts.
        const push = new THREE.Vector3();
        if (lastHit && lastHit.fur === target.fur && now - lastHit.time < 120) {
            push.subVectors(hit.point, lastHit.point).multiplyScalar(1000 / Math.max(now - lastHit.time, 4) * 0.6);
            const l = push.length();
            if (l > 4) push.multiplyScalar(4 / l);
        }
        target.fur.stroke(hit.point, push, 0.5, 1.6);
        lastHit = { fur: target.fur, point: hit.point.clone(), time: now };
        canvas.style.cursor = 'grab';
    }, { capture: true });
    canvas.addEventListener('pointerleave', () => { lastHit = null; canvas.style.cursor = ''; });

    const windEl = document.getElementById('wind');
    const msaaEl = document.getElementById('msaa');
    const shadowsEl = document.getElementById('shadows');
    const swingEl = document.getElementById('swing');
    // Off unless asked for: Safari's WebGPU shades a multisampled fragment
    // that discards or takes derivatives once per sample — every fur and
    // hair fragment does both — which costs it ten times over.
    msaaEl.checked = params.get('msaa') === '1';
    if (params.get('wind') != null) windEl.value = params.get('wind');
    if (params.get('shadows') === '0') shadowsEl.checked = false;
    renderer._w.setMsaa?.(msaaEl.checked ? 4 : 1);
    sun.castShadow = shadowsEl.checked;
    build(Number(windEl.value));
    mark('built');
    // Into the wasm scene now — growing the strands, building the casters —
    // so the first frame is only the GPU's work.
    scene._syncNow?.();
    mark('synced');
    windEl.addEventListener('input', () => { document.getElementById('windv').textContent = `${windEl.value} m/s`; });
    windEl.addEventListener('change', () => build(Number(windEl.value)));
    msaaEl.addEventListener('change', () => renderer._w.setMsaa?.(msaaEl.checked ? 4 : 1));
    shadowsEl.addEventListener('change', () => { sun.castShadow = shadowsEl.checked; });

    statusEl.textContent = String(renderer._w.backend?.() ?? renderer._w.backend ?? 'running');
    const start = performance.now();
    let frames = 0, last = start, fps = 0, drawn = 0;
    function frame(now) {
        const t = (now - start) / 1000;
        if (swingEl.checked && swinging) {
            // A pendulum that pauses at each end: the fur trails, then catches up.
            swinging.rotation.z = 0.55 * Math.sin(t * 1.6) * Math.min(1, t);
        }
        controls.update();
        renderer.render(scene, camera);
        if (drawn === 0) {
            mark('first');
            const nav = performance.getEntriesByType('navigation')[0];
            const wasmRes = performance.getEntriesByType('resource').find((r) => r.name.endsWith('.wasm'));
            beacon({
                load: 1, wasm: marks.wasm, renderer: marks.renderer, built: marks.built, synced: marks.synced, first: marks.first,
                page: Math.round(marks.start), wasm_fetch: Math.round(wasmRes?.duration ?? -1),
                wasm_bytes: wasmRes?.transferSize ?? -1, dom: Math.round(nav?.domContentLoadedEventEnd ?? -1),
            });
        }
        frames++;
        drawn++;
        if (stopAfter && drawn >= stopAfter) {
            // Keep the last frame: copied out in the task that drew it, as a
            // WebGL2 canvas does not keep its drawing buffer once composited.
            const still = new Image();
            still.src = canvas.toDataURL('image/png');
            still.style.cssText = 'width:100%;display:block;border-radius:10px';
            canvas.replaceWith(still);
            still.onload = () => beacon({ done: drawn, backend: statusEl.textContent });
            return;
        }
        if (now - last > 1000) {
            fps = (frames * 1000) / (now - last);
            frames = 0; last = now;
            infoEl.textContent = `${fps.toFixed(0)} fps · ${msaaEl.checked ? 'MSAA 4×' : '1 sample'}`;
            // Report the rate now and then (the first seconds compile shaders).
            if (t > 3 && t < 40 && Math.floor(t) % 5 === 0) {
                beacon({ ok: 1, t: t.toFixed(0), fps: fps.toFixed(1), backend: statusEl.textContent });
            }
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    errEl.textContent = String(e?.stack ?? e);
    statusEl.textContent = 'failed';
    beacon({ error: String(e?.message ?? e) });
});
