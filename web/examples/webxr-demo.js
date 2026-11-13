// WebXR: a table of shapes to point at, recolour and pick up, in a headset
// (VR or AR) or on the desktop. The session, both eyes and the copy to the
// headset are threers' `WebXr` in wasm; this is a three.js page.
import THREE, { initThreers } from '../threejs-shim.js';

const status = document.getElementById('status');
const err = document.getElementById('err');

try {
    await initThreers({ module_or_path: new URL('../pkg/threers_bg.wasm', import.meta.url) });
    await main();
} catch (e) {
    console.error(e);
    status.textContent = 'failed';
    err.textContent = String(e?.stack ?? e);
}

async function main() {
    const canvas = document.getElementById('view');
    const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const fit = () => renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    fit();
    addEventListener('resize', () => { if (!renderer.xr.isPresenting) { fit(); camera.aspect = canvas.clientWidth / canvas.clientHeight; camera.updateProjectionMatrix(); } });
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.xr.enabled = true;

    const scene = new THREE.Scene();
    const sky = new THREE.Color(0x1a1f2b);
    scene.background = sky;
    scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30281f, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(2, 4, 1.5);
    sun.castShadow = true;
    scene.add(sun);

    // The floor, and a table at standing height a step ahead.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshStandardMaterial({ color: 0x3a4150, roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(8, 16, 0x5a6478, 0x485062);
    grid.position.y = 0.002;
    scene.add(grid);
    const table = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.05, 0.8), new THREE.MeshStandardMaterial({ color: 0x8a6a4a, roughness: 0.6 }));
    table.position.set(0, 0.85, -1.2);
    table.castShadow = table.receiveShadow = true;
    scene.add(table);
    // Candles at the table's front corners (threers' own): the realistic
    // volumetric flame on the right, the simple one on the left. They
    // flicker on the renderer's clock.
    if (THREE.Candle) {
        for (const [x, realistic] of [[-0.6, false], [0.6, true]]) {
            const candle = new THREE.Candle({ realistic, intensity: 0.4 });
            candle.position.set(x, 0.875, -0.92);
            scene.add(candle);
        }
    }
    for (const [x, z] of [[-0.65, -0.35], [0.65, -0.35], [-0.65, 0.35], [0.65, 0.35]]) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.85, 12), table.material);
        leg.position.set(x, -0.425, z);
        table.add(leg);
    }

    // The shapes: what the controllers point at.
    const palette = [0xe8664a, 0xf2b33d, 0x5cc98a, 0x4fa3e8, 0xa77de8, 0xe86fa8];
    const geometries = [
        new THREE.BoxGeometry(0.16, 0.16, 0.16),
        new THREE.SphereGeometry(0.09, 32, 16),
        new THREE.TorusKnotGeometry(0.06, 0.022, 96, 12),
        new THREE.CylinderGeometry(0.07, 0.07, 0.18, 32),
        new THREE.IcosahedronGeometry(0.1, 0),
        new THREE.TorusGeometry(0.075, 0.028, 16, 48),
    ];
    const shapes = new THREE.Group();
    scene.add(shapes);
    geometries.forEach((g, i) => {
        const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: palette[i], roughness: 0.35, metalness: 0.1 }));
        m.position.set(-0.5 + 0.2 * i, 0.98, -1.2 + (i % 2 ? 0.15 : -0.12));
        m.castShadow = true;
        m.userData.hue = i;
        shapes.add(m);
    });

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.05, 50);
    camera.position.set(0, 1.6, 0.6);
    // Moving the rig moves the viewer; the headset moves the camera inside it.
    const rig = new THREE.Group();
    rig.add(camera);
    scene.add(rig);
    const orbit = new THREE.OrbitControls(camera, canvas);
    orbit.target.set(0, 1.0, -1.2);
    orbit.update();

    // Controllers: a ray each, which lights what it points at; the trigger
    // recolours, the grip carries.
    const raycaster = new THREE.Raycaster();
    const rayGeometry = new THREE.BufferGeometry();
    rayGeometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, -1], 3));
    const hands = [];
    for (let i = 0; i < 2; i++) {
        const controller = renderer.xr.getController(i);
        const ray = new THREE.Line(rayGeometry, new THREE.LineBasicMaterial({ color: 0xffffff }));
        ray.scale.z = 2;
        controller.add(ray);
        rig.add(controller);
        const state = { controller, ray, hovered: null, held: null };
        controller.addEventListener('connected', (e) => { ray.visible = e.data.targetRayMode !== 'gaze'; });
        const recolour = (s) => {
            s.userData.hue = (s.userData.hue + 1) % palette.length;
            s.material.color.setHex(palette[s.userData.hue]);
        };
        controller.addEventListener('select', (e) => {
            if (e.data.targetRayMode !== 'transient-pointer' && state.hovered) recolour(state.hovered);
        });
        // Apple Vision Pro: no grip and no hover, only look-and-pinch. A
        // pinch held carries what it lands on; a quick one recolours it.
        controller.addEventListener('selectstart', (e) => {
            if (e.data.targetRayMode !== 'transient-pointer') return;
            point(state);
            if (!state.hovered) return;
            state.held = state.hovered;
            state.pinched = performance.now();
            controller.attach(state.held);
        });
        controller.addEventListener('selectend', (e) => {
            if (e.data.targetRayMode !== 'transient-pointer' || !state.held) return;
            const s = state.held;
            shapes.attach(s);
            state.held = null;
            if (performance.now() - state.pinched < 300) recolour(s);
        });
        controller.addEventListener('squeezestart', () => {
            if (state.hovered) { state.held = state.hovered; controller.attach(state.held); }
        });
        controller.addEventListener('squeezeend', () => {
            if (state.held) { shapes.attach(state.held); state.held = null; }
        });
        hands.push(state);
        // Hands: a small ball on every tracked joint.
        const hand = renderer.xr.getHand(i);
        rig.add(hand);
        state.hand = hand;
    }
    const jointBall = new THREE.SphereGeometry(1, 10, 6);
    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xf0d6c0, roughness: 0.7 });

    function point(state) {
        const { controller, ray } = state;
        if (state.hovered && state.hovered !== state.held) state.hovered.material.emissive.setHex(0x000000);
        state.hovered = null;
        if (!controller.visible || state.held) { ray.scale.z = 2; return; }
        raycaster.setFromXRController(controller);
        const hit = raycaster.intersectObjects(shapes.children, false)[0];
        ray.scale.z = hit ? hit.distance : 2;
        if (hit) {
            state.hovered = hit.object;
            hit.object.material.emissive.setHex(0x333333);
        }
    }

    function dressHand(hand) {
        for (const joint of Object.values(hand.joints)) {
            if (joint.userData.ball) continue;
            const ball = new THREE.Mesh(jointBall, jointMaterial);
            joint.add(ball);
            joint.userData.ball = ball;
        }
        for (const joint of Object.values(hand.joints)) joint.userData.ball.scale.setScalar(joint.jointRadius ?? 0.008);
    }

    // AR: the room shows through, so no sky and no floor.
    renderer.xr.addEventListener('sessionstart', () => {
        const ar = renderer.xr.getEnvironmentBlendMode() !== 'opaque';
        orbit.enabled = false;
        camera.position.set(0, 0, 0);
        scene.background = ar ? null : sky;
        renderer.setClearColor(0x000000, ar ? 0 : 1);
        floor.visible = grid.visible = !ar;
        status.textContent = ar ? 'in AR' : 'in VR';
    });
    renderer.xr.addEventListener('sessionend', () => {
        orbit.enabled = true;
        camera.position.set(0, 1.6, 0.6);
        camera.quaternion.identity();
        camera.scale.set(1, 1, 1);
        orbit.update();
        scene.background = sky;
        floor.visible = grid.visible = true;
        fit();
        status.textContent = 'ready';
    });

    const buttons = document.getElementById('buttons');
    buttons.append(THREE.VRButton.createButton(renderer), THREE.ARButton.createButton(renderer));
    const has = 'xr' in navigator;
    status.textContent = !window.isSecureContext ? 'ready · WebXR needs https' : has ? 'ready' : 'ready · this browser has no WebXR';

    const clock = new THREE.Clock();
    renderer.setAnimationLoop(() => {
        const t = clock.getElapsedTime();
        shapes.children.forEach((s, i) => {
            if (hands.some((h) => h.held === s)) return;
            s.rotation.y = t * (0.3 + 0.1 * i);
        });
        if (renderer.xr.isPresenting) {
            for (const h of hands) {
                point(h);
                if (h.hand.visible) dressHand(h.hand);
            }
        } else {
            orbit.update();
        }
        renderer.render(scene, camera);
    });
}
