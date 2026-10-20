// Geometry, materials, handles, cameras, loaders and renderer settings.
// Run after `npm run build`: `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const {
  Scene,
  Geometry,
  Material,
  PerspectiveCamera,
  OrthographicCamera,
  HeadlessRenderer,
} = require('..')

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const W = 64
const H = 64

function renderer(t, w = W, h = H) {
  try {
    return new HeadlessRenderer(w, h)
  } catch (e) {
    t.skip(`no GPU adapter: ${e.message}`)
    return null
  }
}
const px = (rgba, x, y, w = W) => [...rgba.subarray((y * w + x) * 4, (y * w + x) * 4 + 3)]
const sum = (p) => p[0] + p[1] + p[2]
const BG = 0x204060
const BG_PX = [32, 64, 96]
const near = (a, b, tol = 3) => a.every((v, i) => Math.abs(v - b[i]) <= tol)

function camera() {
  const c = new PerspectiveCamera(50, 1, 0.1, 100)
  c.setPosition(0, 0, 4)
  c.lookAt(0, 0, 0)
  return c
}
function scene() {
  const s = new Scene()
  s.setBackground(BG)
  return s
}

// ---- geometry --------------------------------------------------------------

test('every Geometry factory builds triangles', () => {
  const geoms = {
    box: Geometry.box(1, 2, 3),
    plane: Geometry.plane(1, 1, 2, 3),
    sphere: Geometry.sphere(),
    cylinder: Geometry.cylinder(0.5, 1, 2, 16, 2, true),
    cone: Geometry.cone(),
    torus: Geometry.torus(1, 0.3, 8, 24, Math.PI),
    torusKnot: Geometry.torusKnot(),
    ring: Geometry.ring(),
    circle: Geometry.circle(1, 12),
    capsule: Geometry.capsule(0.5, 1),
    lathe: Geometry.lathe([0, -1, 1, 0, 0, 1], 16),
    latheObjects: Geometry.lathe([{ x: 0, y: -1 }, { x: 1, y: 0 }]),
    tube: Geometry.tube([0, 0, 0, 1, 1, 0, 2, 0, 0], 32, 0.1),
    icosahedron: Geometry.icosahedron(1, 1),
    octahedron: Geometry.octahedron(),
    tetrahedron: Geometry.tetrahedron(),
    dodecahedron: Geometry.dodecahedron(),
    quad: Geometry.quad([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 2, 2),
    fromArrays: Geometry.fromArrays([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  }
  for (const [name, g] of Object.entries(geoms)) {
    assert.ok(g.vertexCount() > 0, `${name} has vertices`)
    assert.ok(g.triangleCount() > 0, `${name} has triangles`)
  }
  assert.equal(geoms.box.triangleCount(), 12)
  assert.equal(geoms.plane.triangleCount(), 2 * 2 * 3)
  assert.equal(geoms.fromArrays.triangleCount(), 1)
  assert.equal(Geometry.fromArrays([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], [0, 1, 2, 2, 1, 3]).triangleCount(), 2)
})

test('Geometry factories reject bad input', () => {
  assert.throws(() => Geometry.box(-1), /width/)
  assert.throws(() => Geometry.sphere(1, 2), /widthSegments/)
  assert.throws(() => Geometry.plane(1, 1, 0), /widthSegments/)
  assert.throws(() => Geometry.plane(1, 1, 5000), /widthSegments/)
  assert.throws(() => Geometry.torus(1, 0.4, 12, 48, 0), /arc/)
  assert.throws(() => Geometry.torusKnot(1, 0.4, 64, 8, 0), /p and q/)
  assert.throws(() => Geometry.ring(2, 1), /innerRadius/)
  assert.throws(() => Geometry.lathe([0, 0]), /at least 2/)
  assert.throws(() => Geometry.lathe([0, 0, 1]), /whole number/)
  assert.throws(() => Geometry.tube([0, 0, 0, NaN, 1, 1]), /non-finite/)
  assert.throws(() => Geometry.icosahedron(1, 99), /detail/)
  assert.throws(() => Geometry.fromArrays([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 5]), /out of range/)
  assert.throws(() => Geometry.fromArrays([0, 0]), /multiple of 3/)
  assert.throws(() => Geometry.fromArrays([0, 0, 0, 1, 0, 0, 0, 1, 0], undefined, [0, 0, 1]), /normals/)
})

// ---- materials -------------------------------------------------------------

test('Material factories set kind and validate options', () => {
  for (const k of ['basic', 'standard', 'physical', 'lambert', 'phong', 'normal']) {
    assert.equal(Material[k]().kind, k)
  }
  Material.standard({ color: 0x336699, roughness: 0.3, metalness: 1, emissive: 0x110000, side: 'double' })
  Material.physical({ clearcoat: 1, transmission: 0.5, ior: 1.33, thickness: 0.2 })
  Material.phong({ specular: 0xffffff, shininess: 80 })
  Material.basic({ opacity: 0.5, transparent: true, wireframe: true })
  assert.throws(() => Material.basic({ roughness: 0.5 }), /roughness' does not apply/)
  assert.throws(() => Material.lambert({ shininess: 1 }), /shininess' does not apply/)
  assert.throws(() => Material.normal({ color: 0xffffff }), /color' does not apply/)
  assert.throws(() => Material.standard({ roughness: 2 }), /roughness must be in \[0, 1\]/)
  assert.throws(() => Material.standard({ color: 0x1000000 }), /0xRRGGBB/)
  assert.throws(() => Material.standard({ side: 'both' }), /side/)
  assert.throws(() => Material.physical({ ior: 0.5 }), /ior/)
})

test('addMesh renders each material kind', (t) => {
  const r = renderer(t)
  if (!r) return
  const cam = camera()
  const centre = (mat, light) => {
    const s = scene()
    s.addMesh(Geometry.sphere(1), mat)
    if (light) s.addDirectionalLight(0xffffff, 3, { x: 0, y: 0, z: 5 })
    return px(r.renderRgba(s, cam), W / 2, H / 2)
  }
  const basic = centre(Material.basic({ color: 0xff0000 }))
  assert.ok(basic[0] > 240 && basic[1] < 10 && basic[2] < 10, `basic red: ${basic}`)
  const normal = centre(Material.normal())
  assert.ok(normal[2] > 200, `normal facing +z is blue: ${normal}`)
  for (const k of ['standard', 'physical', 'lambert', 'phong']) {
    assert.ok(sum(centre(Material[k]({ color: 0x00ff00 }))) < 6, `${k} unlit is black`)
    const lit = centre(Material[k]({ color: 0x00ff00 }), true)
    assert.ok(lit[1] > 100 && lit[1] > lit[0] + 50, `${k} lit is green: ${lit}`)
  }
})

// ---- handles ---------------------------------------------------------------

test('handles move, hide, re-material and remove objects', (t) => {
  const r = renderer(t)
  if (!r) return
  const cam = camera()
  const s = scene()
  const h = s.addMesh(Geometry.box(1.5, 1.5, 1.5), Material.basic({ color: 0xffffff }))
  const centre = () => px(r.renderRgba(s, cam), W / 2, H / 2)
  assert.ok(sum(centre()) > 700)
  s.setPosition(h, 10, 0, 0)
  assert.ok(near(centre(), BG_PX), 'moved out of view')
  s.setPosition(h, 0, 0, 0)
  s.setScale(h, 0.01, 0.01, 0.01)
  assert.ok(near(centre(), BG_PX), 'scaled to nothing')
  s.setScale(h, 1, 1, 1)
  s.setRotation(h, 0, Math.PI / 4, 0)
  s.setVisible(h, false)
  assert.ok(near(centre(), BG_PX), 'hidden')
  s.setVisible(h, true)
  s.setMaterial(h, Material.basic({ color: 0x0000ff }))
  const blue = centre()
  assert.ok(blue[2] > 240 && blue[0] < 10, `re-materialed: ${blue}`)
  const p = s.getPosition(h)
  assert.deepEqual([p.x, p.y, p.z], [0, 0, 0])

  // Groups carry their children.
  const g = s.addGroup()
  s.setParent(h, g)
  s.setPosition(g, 10, 0, 0)
  assert.ok(near(centre(), BG_PX), 'moved with its group')
  assert.throws(() => s.setParent(g, h), /cycle/)
  s.setParent(h)
  assert.ok(sum(centre()) > 0 && !near(centre(), BG_PX), 'back at the root')

  assert.equal(s.objectCount(), 2)
  s.remove(h)
  assert.equal(s.objectCount(), 1)
  assert.ok(near(centre(), BG_PX), 'removed')
  assert.throws(() => s.setPosition(h, 0, 0, 0), /removed/)
  assert.throws(() => s.setVisible(99, true), /bad object handle/)
  assert.throws(() => s.setRotation(g, NaN, 0, 0), /finite/)
  assert.throws(() => s.setMaterial(g, Material.basic()), /no mesh/)
  // Handles stay stable after a removal.
  assert.equal(s.addGroup(), 2)
})

// ---- cameras ---------------------------------------------------------------

test('OrthographicCamera renders and validates its frustum', (t) => {
  assert.throws(() => new OrthographicCamera(1, 1), /left != right/)
  assert.throws(() => new OrthographicCamera(-1, 1, 1, -1, 5, 1), /near < far/)
  const r = renderer(t)
  if (!r) return
  const s = scene()
  s.addMesh(Geometry.box(1, 1, 1), Material.basic({ color: 0xffffff }))
  const cam = new OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
  cam.setPosition(0, 0, 5)
  cam.lookAt(0, 0, 0)
  const rgba = r.renderRgba(s, cam)
  // A unit box in a 4-unit view covers the middle quarter: x 24..40.
  assert.ok(sum(px(rgba, 32, 32)) > 700)
  assert.ok(sum(px(rgba, 26, 32)) > 700, 'inside the box edge')
  assert.ok(near(px(rgba, 20, 32), BG_PX), 'outside it — no perspective growth')
  cam.setFrustum(-0.25, 0.25, 0.25, -0.25)
  assert.ok(sum(px(r.renderRgba(s, cam), 2, 2)) > 700, 'zoomed in, the box fills the view')
  assert.ok(r.renderPng(s, cam).length > 0)
})

// ---- renderer --------------------------------------------------------------

test('setSize, tone mapping and the shadow switch', (t) => {
  const r = renderer(t)
  if (!r) return
  const s = scene()
  s.addMesh(Geometry.sphere(1), Material.standard({ color: 0xffffff }))
  s.addDirectionalLight(0xffffff, 1, { x: 0, y: 0, z: 5 })
  const cam = camera()

  r.setToneMapping('aces', 1.5)
  assert.deepEqual(r.toneMapping(), { mode: 'aces', exposure: 1.5 })
  r.setSize(48, 32)
  assert.deepEqual(r.size(), [48, 32])
  assert.equal(r.renderRgba(s, cam).length, 48 * 32 * 4)
  assert.deepEqual(r.toneMapping(), { mode: 'aces', exposure: 1.5 }, 'kept across setSize')
  r.setSize(W, H)
  r.setToneMapping('none')
  const plain = sum(px(r.renderRgba(s, cam), W / 2, H / 2))
  r.setToneMapping('linear', 0.25)
  const dim = sum(px(r.renderRgba(s, cam), W / 2, H / 2))
  assert.ok(dim < plain * 0.6, `exposure 0.25 darkens: ${dim} vs ${plain}`)
  r.setToneMapping('none')
  assert.throws(() => r.setToneMapping('filmic'), /unknown tone mapping/)
  assert.throws(() => r.setToneMapping('linear', -1), /exposure/)
  assert.throws(() => r.setSize(0, 10), /setSize/)

  // A box casting onto a ground plane, light from straight above.
  const sh = scene()
  const ground = sh.addMesh(Geometry.plane(10, 10), Material.standard({ color: 0xffffff }))
  sh.setRotation(ground, -Math.PI / 2, 0, 0)
  sh.setReceiveShadow(ground, true)
  const box = sh.addMesh(Geometry.box(1, 1, 1), Material.standard({ color: 0xffffff }), 0, 1, 0)
  sh.setCastShadow(box, true)
  sh.addDirectionalLight(0xffffff, 2, { x: 5, y: 10, z: 0 }, true)
  const top = new PerspectiveCamera(50, 1, 0.1, 100)
  top.setPosition(0, 8, 2)
  top.lookAt(0, 0, 0)
  assert.equal(r.shadowsEnabled(), true)
  const on = r.renderRgba(sh, top)
  r.setShadowsEnabled(false)
  const off = r.renderRgba(sh, top)
  r.setShadowsEnabled(true)
  // Shadows only ever darken, and the box's shadow darkens the ground a lot.
  let most = 0
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d = sum(px(off, x, y)) - sum(px(on, x, y))
      assert.ok(d >= -6, `pixel ${x},${y} brighter with shadows`)
      most = Math.max(most, d)
    }
  }
  assert.ok(most > 150, `shadow darkens the ground by ${most}`)
})

// ---- loaders ---------------------------------------------------------------

function litScene() {
  const s = scene()
  s.addAmbientLight(0xffffff, 0.5)
  s.addDirectionalLight(0xffffff, 2, { x: 1, y: 2, z: 3 })
  return s
}
const tmp = mkdtempSync(join(tmpdir(), 'threers-node-'))
const OBJ = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n'
const STL = 'solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n'
const PLY = [
  'ply', 'format ascii 1.0', 'element vertex 3', 'property float x', 'property float y',
  'property float z', 'element face 1', 'property list uchar int vertex_indices', 'end_header',
  '0 0 0', '1 0 0', '0 1 0', '3 0 1 2', '',
].join('\n')

test('load reads glb, obj, stl, ply and usd files', (t) => {
  const s = litScene()
  const glb = s.load(join(REPO, 'src/loaders/testdata/sphere_plain.glb'))
  for (const [name, text] of [['tri.obj', OBJ], ['tri.stl', STL], ['tri.ply', PLY]]) {
    writeFileSync(join(tmp, name), text)
    s.load(join(tmp, name), Material.basic({ color: 0xffffff }))
  }
  const usd = s.load(join(REPO, 'src/loaders/usd/testdata/apple.usda'))
  assert.equal(s.objectCount(), 2 + 5)
  assert.equal(usd, glb + 4)

  const r = renderer(t)
  if (!r) return
  // The glTF sphere, alone and centred, is lit and visible.
  const only = litScene()
  only.load(join(REPO, 'src/loaders/testdata/sphere_plain.glb'))
  const c = px(r.renderRgba(only, camera()), W / 2, H / 2)
  assert.ok(!near(c, BG_PX) && sum(c) > 30, `glb sphere visible: ${c}`)
})

test('loadBytes and external glTF buffers', () => {
  const s = litScene()
  const before = s.objectCount()
  s.loadBytes(readFileSync(join(REPO, 'src/loaders/testdata/sphere_plain.glb')), 'glb')
  s.loadBytes(Buffer.from(OBJ), 'obj')
  s.loadBytes(Buffer.from(STL), 'STL')
  s.loadBytes(Buffer.from(PLY), 'ply')
  s.loadBytes(readFileSync(join(REPO, 'src/loaders/usd/testdata/apple.usdc')), 'usd')
  assert.equal(s.objectCount(), before + 5)

  // A .gltf whose triangle lives in a separate .bin next to it.
  const bin = Buffer.alloc(36)
  ;[0, 0, 0, 1, 0, 0, 0, 1, 0].forEach((v, i) => bin.writeFloatLE(v, i * 4))
  writeFileSync(join(tmp, 'tri data.bin'), bin)
  const gltf = JSON.stringify({
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }],
    bufferViews: [{ buffer: 0, byteLength: 36 }],
    buffers: [{ byteLength: 36, uri: 'tri%20data.bin' }],
  })
  writeFileSync(join(tmp, 'tri.gltf'), gltf)
  s.load(join(tmp, 'tri.gltf'))
  assert.throws(() => s.loadBytes(Buffer.from(gltf), 'gltf'), /external buffer/)
})

test('loaders reject bad input', () => {
  const s = new Scene()
  assert.throws(() => s.load(join(tmp, 'missing.glb')), /missing\.glb/)
  assert.throws(() => s.load(join(tmp, 'model.dwg')), /unknown model format 'dwg'/)
  assert.throws(() => s.load(join(tmp, 'noext')), /no extension/)
  assert.throws(() => s.loadBytes(Buffer.from('nope'), 'glb'), /glTF/)
  assert.throws(() => s.loadBytes(Buffer.from('nope'), 'usd'), /USD/)
  assert.throws(() => s.loadBytes(Buffer.from(''), 'obj'), /no geometry/)
  assert.throws(() => s.loadBytes(Buffer.from(OBJ), 'dwg'), /unknown model format/)
  assert.equal(s.objectCount(), 0)
})
