import THREE, { initThreers } from '/web/threejs-shim.js';

const errEl = document.getElementById('err');
const modeEl = document.getElementById('mode');

// Wrapped in an async IIFE rather than using top-level `await`, which is
// ES2022 module syntax and needs Safari 15. A syntax error is not recoverable —
// the module never loads and the page reports one error a long way from the
// cause — so the cost of the wrapper is worth the floor it buys.
(async () => {
    try {
        await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

        const canvas = document.getElementById('canvas');
        const preview = canvas.parentElement;

        function resizeCanvas() {
            const dpr = Math.min(devicePixelRatio || 1, 2);
            const w = Math.floor(preview.clientWidth * dpr);
            const h = Math.floor(preview.clientHeight * dpr);
            canvas.width = w;
            canvas.height = h;
            return { w, h };
        }

        let { w, h } = resizeCanvas();

        const renderer = await THREE.WebGLRenderer.create(canvas);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x202030);
        scene.add(new THREE.AmbientLight(0xffffff, 0.35));
        scene.add(new THREE.DirectionalLight(0xffffff, 1));

        scene.add(new THREE.Mesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshStandardMaterial({ color: 0xff6633, roughness: 0.4, metalness: 0.1 }),
        ));

        const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
        camera.position.set(2.5, 2, 3.5);
        camera.lookAt(0, 0, 0);

        const controls = new THREE.TrackballControls(camera, renderer.domElement);
        if (modeEl) modeEl.textContent = 'TrackballControls';

        addEventListener('resize', () => {
            ({ w, h } = resizeCanvas());
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });

        function frame() {
            requestAnimationFrame(frame);
            renderer.render(scene, camera);
        }
        frame();
        void controls;
    } catch (e) {
        if (errEl) {
            errEl.style.display = 'block';
            errEl.textContent = e?.stack || String(e);
        }
        console.error(e);
    }
})();
