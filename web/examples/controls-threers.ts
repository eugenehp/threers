/**
 * threers controls demo (TypeScript reference — same logic as controls-threers-demo.js).
 */
import THREE, {
    initThreers,
    type PerspectiveCamera,
    type WebGLRenderer,
} from '../threejs-shim.js';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const preview = canvas.parentElement as HTMLElement;

function resizeCanvas(): { w: number; h: number } {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(preview.clientWidth * dpr);
    const h = Math.floor(preview.clientHeight * dpr);
    canvas.width = w;
    canvas.height = h;
    return { w, h };
}

await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });

let { w, h } = resizeCanvas();
const renderer: WebGLRenderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202030);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
scene.add(new THREE.DirectionalLight(0xffffff, 1));
scene.add(new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0xff6633, roughness: 0.4, metalness: 0.1 }),
));

const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100) as PerspectiveCamera;
camera.position.set(2.5, 2, 3.5);
camera.lookAt(0, 0, 0);

const controls = new THREE.TrackballControls(camera, renderer.domElement);

window.addEventListener('resize', () => {
    ({ w, h } = resizeCanvas());
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
});

function frame(): void {
    requestAnimationFrame(frame);
    renderer.render(scene, camera);
}
frame();

void controls;
