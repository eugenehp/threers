// Shot timeline: hold → ramp fly → path crane, applied via wasm setView.
import THREE, { initThreers } from '../threejs-shim.js';

const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');
const timeEl = document.getElementById('time');
const shotEl = document.getElementById('shot');
const seekEl = document.getElementById('seek');
const playBtn = document.getElementById('play');
const restartBtn = document.getElementById('restart');
const snippetEl = document.getElementById('snippet');

snippetEl.textContent = `const timeline = new THREE.ShotTimeline();
timeline.hold('establishing', { position: [4, 2.5, 5], target: [0, 0.5, 0], duration: 2 });
timeline.fly('push-in', {
  from: { position: [4, 2.5, 5], target: [0, 0.5, 0], fov: 55 },
  to:   { position: [1.8, 1.2, 2.4], target: [0, 0.6, 0], fov: 40 },
  duration: 3, ramp: 'inOut', easeIn: 0.25, easeOut: 0.3,
  transition: { type: 'blend', duration: 0.6 },
});
timeline.path('crane', {
  path: THREE.CameraPath.catmullRom([
    [1.8, 1.2, 2.4], [0.5, 2.5, 3], [-2, 1.5, 2], [-1.5, 0.8, -1],
  ], { fixedTarget: [0, 0.5, 0], ramp: 'smoother' }),
  duration: 5,
  transition: { type: 'blend', duration: 0.8, ramp: 'cubicInOut' },
});
timeline.apply(camera);`;

async function main() {
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101018);

    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.1);
    sun.position.set(4, 8, 2);
    scene.add(sun);

    const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(12, 12),
        new THREE.MeshStandardMaterial({ color: 0x2a3140, roughness: 0.9, metalness: 0.05 }),
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const subject = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({ color: 0xff6633, roughness: 0.35, metalness: 0.15 }),
    );
    subject.position.y = 0.5;
    scene.add(subject);

    const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.35, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0x6ea8fe, roughness: 0.3, metalness: 0.2 }),
    );
    ball.position.set(-1.4, 0.35, 0.8);
    scene.add(ball);

    const pillar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.25, 0.35, 1.2, 20),
        new THREE.MeshStandardMaterial({ color: 0x7dcf8a, roughness: 0.45, metalness: 0.1 }),
    );
    pillar.position.set(1.2, 0.6, -0.6);
    scene.add(pillar);

    const camera = new THREE.PerspectiveCamera(55, canvas.width / canvas.height, 0.1, 100);
    camera.position.set(4, 2.5, 5);
    camera.lookAt(0, 0.5, 0);

    const timeline = new THREE.ShotTimeline();
    timeline.looping = true;
    timeline.hold('establishing', {
        position: [4, 2.5, 5],
        target: [0, 0.5, 0],
        fov: 55,
        duration: 2,
    });
    timeline.fly('push-in', {
        from: { position: [4, 2.5, 5], target: [0, 0.5, 0], fov: 55 },
        to: { position: [1.8, 1.2, 2.4], target: [0, 0.6, 0], fov: 40 },
        duration: 3,
        ramp: 'inOut',
        easeIn: 0.25,
        easeOut: 0.3,
        transition: { type: 'blend', duration: 0.6, ramp: 'cubicInOut' },
    });
    timeline.path('crane', {
        path: THREE.CameraPath.catmullRom(
            [
                [1.8, 1.2, 2.4],
                [0.5, 2.5, 3.0],
                [-2.0, 1.5, 2.0],
                [-1.5, 0.8, -1.0],
            ],
            { fixedTarget: [0, 0.5, 0], ramp: 'smoother', fov: 45 },
        ),
        duration: 5,
        transition: { type: 'blend', duration: 0.8, ramp: 'cubicInOut' },
    });

    const total = timeline.duration();
    seekEl.max = String(total);

    let playing = true;
    let last = performance.now();

    playBtn.addEventListener('click', () => {
        playing = !playing;
        playBtn.textContent = playing ? 'Pause' : 'Play';
        last = performance.now();
    });
    restartBtn.addEventListener('click', () => {
        timeline.seek(0);
        seekEl.value = '0';
        last = performance.now();
    });
    seekEl.addEventListener('input', () => {
        timeline.seek(Number(seekEl.value));
        last = performance.now();
    });

    addEventListener('resize', () => {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const dpr = Math.min(devicePixelRatio || 1, 2);
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
        renderer.setSize(canvas.width, canvas.height);
        camera.aspect = canvas.width / canvas.height;
        camera.updateProjectionMatrix();
    });

    function frame(now) {
        requestAnimationFrame(frame);
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        if (playing) timeline.update(dt);
        const pose = timeline.apply(camera);
        subject.rotation.y += dt * 0.35;
        timeEl.textContent = `${timeline.time.toFixed(1)} s`;
        shotEl.textContent = pose.shotName ? `shot · ${pose.shotName}` : '—';
        if (!seekEl.matches(':active')) seekEl.value = String(timeline.time);
        renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    statusEl.textContent = 'error';
    errEl.textContent = e?.stack || String(e);
    console.error(e);
});
