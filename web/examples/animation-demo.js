// Mixer cross-fade + Tween + ObjectSpring + PoseBlend demo.
import THREE, { initThreers } from '../threejs-shim.js';

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const errEl = document.getElementById('err');

async function main() {
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
    statusEl.textContent = 'wasm ready';

    const canvas = document.getElementById('c');
    const renderer = await THREE.WebGLRenderer.create(canvas);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101018);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.0);
    sun.position.set(3, 6, 2);
    scene.add(sun);

    const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(14, 14),
        new THREE.MeshStandardMaterial({ color: 0x222833, roughness: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const cube = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({ color: 0xff6633, roughness: 0.35, metalness: 0.1 }),
    );
    cube.name = 'Cube';
    cube.position.set(-1.5, 0.5, 0);
    scene.add(cube);

    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.4, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0x6ea8fe, roughness: 0.3, metalness: 0.15 }),
    );
    sphere.name = 'Sphere';
    sphere.position.set(1.5, 0.4, 0);
    scene.add(sphere);

    const follower = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 16, 12),
        new THREE.MeshStandardMaterial({ color: 0x7dcf8a, roughness: 0.4 }),
    );
    follower.position.copy(sphere.position);
    scene.add(follower);

    const camera = new THREE.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 100);
    camera.position.set(0, 3.2, 7);
    camera.lookAt(0, 0.5, 0);

    // Idle ↔ spin clips on the cube via AnimationMixer + PropertyBinding.
    const idle = new THREE.AnimationClip('idle', 2, [
        new THREE.VectorKeyframeTrack('Cube.position', [0, 1, 2], [
            -1.5, 0.5, 0,
            -1.5, 0.85, 0,
            -1.5, 0.5, 0,
        ]),
    ]);
    const spin = new THREE.AnimationClip('spin', 2, [
        new THREE.QuaternionKeyframeTrack('Cube.quaternion', [0, 1, 2], [
            0, 0, 0, 1,
            0, 0.7071, 0, 0.7071,
            0, 0, 0, 1,
        ]),
        new THREE.VectorKeyframeTrack('Cube.position', [0, 2], [
            -1.5, 0.5, 0,
            -1.5, 0.5, 0,
        ]),
    ]);
    spin.blendMode = THREE.AdditiveAnimationBlendMode;

    const mixer = new THREE.AnimationMixer(scene);
    const idleAction = mixer.clipAction(idle).play();
    const spinAction = mixer.clipAction(spin);

    // Tween the sphere across X.
    const tween = new THREE.Tween(
        new THREE.Vector3(1.5, 0.4, 0),
        new THREE.Vector3(2.4, 1.2, -0.5),
        1.6,
    ).setEasing('cubicInOut').setRepeat(Infinity, true);

    const lag = new THREE.ObjectSpring(follower, { frequency: 5, damping: 0.85 });

    // Pose blend between animated sphere and a "fallen" pose.
    const fallen = { position: new THREE.Vector3(1.5, 0.4, 0), quaternion: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2) };
    const blendTarget = {
        position: sphere.position.clone(),
        quaternion: new THREE.Quaternion(),
    };
    const poseBlend = new THREE.PoseBlend(blendTarget, 0.4);

    let which = 'idle';
    document.getElementById('crossfade').onclick = () => {
        if (which === 'idle') {
            spinAction.reset().play();
            spinAction.crossFadeFrom(idleAction, 0.6);
            which = 'spin';
        } else {
            idleAction.reset().play();
            idleAction.crossFadeFrom(spinAction, 0.6);
            which = 'idle';
        }
    };
    document.getElementById('limp').onclick = () => poseBlend.goLimp();
    document.getElementById('getup').onclick = () => poseBlend.getUp();

    let last = performance.now();
    function frame(now) {
        requestAnimationFrame(frame);
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;

        mixer.update(dt);
        const pos = tween.update(dt);
        sphere.position.copy(pos);
        lag.setTarget(sphere).update(dt);

        const animated = { position: pos, quaternion: new THREE.Quaternion() };
        poseBlend.update(dt, animated, fallen);
        // Visualize blend on sphere when not fully animated:
        if (poseBlend.state !== 'animated') {
            sphere.position.copy(blendTarget.position);
            if (sphere.quaternion) sphere.quaternion.copy(blendTarget.quaternion);
        }

        infoEl.textContent = `clip · ${which} · spring settled ${lag.spring.isSettled()} · blend ${poseBlend.state}`;
        renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
}

main().catch((e) => {
    statusEl.textContent = 'error';
    errEl.textContent = e?.stack || String(e);
    console.error(e);
});
