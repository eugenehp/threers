/**
 * Browser video / animated-image export demo — VideoExporter API.
 */
import THREE, {
  initThreers,
  VideoExporter,
  assertVideoExportAvailable,
  parseVideoFormat,
} from '/web/threejs-shim.js';

const canvas = document.getElementById('canvas');
const errEl = document.getElementById('err');
const statusEl = document.getElementById('status');
const exportBtn = document.getElementById('export');
const fmtSel = document.getElementById('fmt');
const framesSel = document.getElementById('frames');

const W = 320;
const H = 240;
const FPS = 15;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function fail(e) {
  console.error(e);
  errEl.textContent = String(e?.message || e);
  setStatus('error');
  exportBtn.disabled = false;
}

async function main() {
  await initThreers('/web/pkg/threers_bg.wasm');
  assertVideoExportAvailable();

  const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
  renderer.setSize(W, H, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0.05, 0.06, 0.09);
  scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(-0.4, 0.8, 0.5);
  scene.add(key);

  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 1.6, 1.6),
    new THREE.MeshStandardMaterial({ color: 0x349eef, metalness: 0.15, roughness: 0.35 }),
  );
  scene.add(cube);

  const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
  camera.position.set(0, 1.4, 4.2);
  camera.lookAt(0, 0, 0);

  let t0 = performance.now();
  function tick(now) {
    const t = (now - t0) / 1000;
    cube.rotation.x = 0.5;
    cube.rotation.y = t;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  exportBtn.addEventListener('click', async () => {
    exportBtn.disabled = true;
    errEl.textContent = '';
    const n = Number(framesSel.value) || 30;
    const { transparent } = parseVideoFormat(fmtSel.value);

    try {
      const exporter = VideoExporter.from(renderer, scene, camera)
        .format(fmtSel.value)
        .size(W, H)
        .fps(FPS)
        .transparent(transparent)
        .gifColors(64)
        .parallel(3)
        .frames(n)
        .transparentCornerPunch(transparent)
        .update((i, total) => {
          cube.rotation.x = 0.5;
          cube.rotation.y = (i / total) * Math.PI * 2;
        })
        .on('progress', (e) => setStatus(e.message || e.detail?.message || ''))
        .on('complete', ({ detail }) => {
          console.log('export complete', detail.result.summary);
        })
        .on('error', ({ detail }) => console.error(detail.error));
      const result = await exporter.download('cube');
      setStatus(result.summary);
    } catch (e) {
      fail(e);
      return;
    }
    exportBtn.disabled = false;
  });

  setStatus('ready');
}

main().catch(fail);
