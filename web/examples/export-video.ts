/**
 * TypeScript example: export a short spinning-cube clip from the browser.
 *
 * ```bash
 * NATIVE_CODEC=1 web/build.sh
 * # then open /web/examples/export-video.html
 * ```
 */
import THREE, {
  initThreers,
  VideoExporter,
  VideoFormat,
  assertVideoExportAvailable,
  type VideoFormatName,
  type VideoExportResult,
} from '/web/threejs-shim.js';

const W = 320;
const H = 240;

export async function runExportDemo(
  canvas: HTMLCanvasElement,
  format: VideoFormatName = VideoFormat.Gif,
): Promise<VideoExportResult> {
  await initThreers('/web/pkg/threers_bg.wasm');
  assertVideoExportAvailable();

  const renderer = await THREE.WebGLRenderer.create(canvas);
  renderer.setSize(W, H, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0f17);
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

  return VideoExporter.from(renderer, scene, camera)
    .format(format)
    .size(W, H)
    .fps(15)
    .transparent(format === VideoFormat.Gif || format === VideoFormat.Apng)
    .gifColors(64)
    .parallel(3)
    .frames(30)
    .update((i: number, total: number) => {
      cube.rotation.x = 0.5;
      cube.rotation.y = (i / total) * Math.PI * 2;
    })
    .download('cube');
}
