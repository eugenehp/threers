// Lights. Run after `npm run build`: `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { Scene, PerspectiveCamera, HeadlessRenderer } = require('..')

const W = 64
const H = 64

function renderer(t) {
  try {
    return new HeadlessRenderer(W, H)
  } catch (e) {
    t.skip(`no GPU adapter: ${e.message}`)
    return null
  }
}

const px = (rgba, x, y) => [...rgba.subarray((y * W + x) * 4, (y * W + x) * 4 + 3)]
const sum = (p) => p[0] + p[1] + p[2]

// A unit box at the origin seen from (3, 0, 3): its +x face fills the right
// of the image and its +z face the left.
function boxScene() {
  const scene = new Scene()
  scene.setBackground(0x204060)
  scene.addBox(1, 1, 1, 0xff6633)
  const camera = new PerspectiveCamera(50, 1, 0.1, 100)
  camera.setPosition(3, 0, 3)
  camera.lookAt(0, 0, 0)
  return { scene, camera }
}
const RIGHT = [W / 2 + 8, H / 2] // +x face
const LEFT = [W / 2 - 8, H / 2] // +z face

test('without lights a Standard box renders black (three.js semantics)', (t) => {
  const r = renderer(t)
  if (!r) return
  const { scene, camera } = boxScene()
  const rgba = r.renderRgba(scene, camera)
  assert.ok(sum(px(rgba, ...RIGHT)) < 6, `box is black: ${px(rgba, ...RIGHT)}`)
  assert.ok(sum(px(rgba, 1, 1)) > 60, `background is not: ${px(rgba, 1, 1)}`)
})

test('a directional light from +x lights the +x face, not the +z face', (t) => {
  const r = renderer(t)
  if (!r) return
  const { scene, camera } = boxScene()
  assert.equal(scene.addDirectionalLight(0xffffff, 3, { x: 5, y: 0, z: 0 }), 1)
  const rgba = r.renderRgba(scene, camera)
  const lit = px(rgba, ...RIGHT)
  const unlit = px(rgba, ...LEFT)
  assert.ok(lit[0] > 100, `lit face is bright: ${lit}`)
  assert.ok(lit[0] > lit[2], `and keeps the box's orange: ${lit}`)
  assert.ok(sum(lit) > sum(unlit) + 100, `lit ${lit} vs unlit ${unlit}`)
})

test('ambient light alone shades every face the same', (t) => {
  const r = renderer(t)
  if (!r) return
  const { scene, camera } = boxScene()
  scene.addAmbientLight(0xffffff, 1)
  const rgba = r.renderRgba(scene, camera)
  const a = px(rgba, ...RIGHT)
  const b = px(rgba, ...LEFT)
  assert.ok(sum(a) > 30, `ambient lights the box: ${a}`)
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(a[i] - b[i]) <= 2, `uniform: ${a} vs ${b}`)
})

test('point, spot and hemisphere lights light the box', (t) => {
  const r = renderer(t)
  if (!r) return
  for (const add of [
    (s) => s.addPointLight(0xffffff, 40, { x: 3, y: 0, z: 0 }),
    (s) => s.addSpotLight(0xffffff, 40, { x: 3, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, 0.6, 0.2),
    (s) => s.addHemisphereLight(0xffffff, 0x444444, 2),
  ]) {
    const { scene, camera } = boxScene()
    add(scene)
    const rgba = r.renderRgba(scene, camera)
    assert.ok(sum(px(rgba, ...RIGHT)) > 60, `${add}: ${px(rgba, ...RIGHT)}`)
  }
})

test('light count increments and lights share the handle space', () => {
  const scene = new Scene()
  assert.equal(scene.lightCount(), 0)
  assert.equal(scene.addBox(), 0)
  assert.equal(scene.addAmbientLight(), 1)
  assert.equal(scene.addDirectionalLight(0xffffff, 2, { x: 1, y: 2, z: 3 }, true), 2)
  assert.equal(scene.addPointLight(0xff0000, 5, { x: 0, y: 2, z: 0 }, 10, 1), 3)
  assert.equal(scene.addSpotLight(), 4)
  assert.equal(scene.addHemisphereLight(0x87ceeb, 0x362907, 0.5), 5)
  assert.equal(scene.lightCount(), 5)
  assert.equal(scene.objectCount(), 6)
  // Lights are positioned through the same handle methods as meshes.
  scene.setPosition(3, 0, 5, 0)
  const p = scene.getPosition(3)
  assert.deepEqual([p.x, p.y, p.z], [0, 5, 0])
  scene.remove(3)
  assert.equal(scene.lightCount(), 4)
})

test('lights reject invalid input', () => {
  const scene = new Scene()
  assert.throws(() => scene.addAmbientLight(0x1000000), /0xRRGGBB/)
  assert.throws(() => scene.addAmbientLight(0xffffff, -1), /intensity/)
  assert.throws(() => scene.addDirectionalLight(0xffffff, NaN), /intensity/)
  assert.throws(() => scene.addDirectionalLight(0xffffff, 1, { x: 0, y: 0, z: 0 }), /origin/)
  assert.throws(() => scene.addDirectionalLight(0xffffff, 1, { x: Infinity, y: 0, z: 0 }), /finite/)
  assert.throws(() => scene.addPointLight(0xffffff, 1, undefined, -2), /distance/)
  assert.throws(() => scene.addPointLight(0xffffff, 1, undefined, 0, -1), /decay/)
  assert.throws(() => scene.addSpotLight(0xffffff, 1, { x: 0, y: 0, z: 0 }), /differ/)
  assert.throws(() => scene.addSpotLight(0xffffff, 1, undefined, undefined, 2), /angle/)
  assert.throws(() => scene.addSpotLight(0xffffff, 1, undefined, undefined, 0.5, 2), /penumbra/)
  assert.throws(() => scene.addHemisphereLight(0xffffff, 0x1000000), /groundColor/)
  assert.equal(scene.lightCount(), 0)
  assert.equal(scene.objectCount(), 0)
})
