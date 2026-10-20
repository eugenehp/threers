// Textures, map slots and MeshPhysicalMaterial's options.
// Run after `npm run build`: `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { deflateSync, crc32 } from 'node:zlib'

const require = createRequire(import.meta.url)
const { Scene, Geometry, Material, Texture, OrthographicCamera, HeadlessRenderer } = require('..')

const W = 32
const H = 32

function renderer(t) {
  try {
    return new HeadlessRenderer(W, H)
  } catch (e) {
    t.skip(`no GPU adapter: ${e.message}`)
    return null
  }
}

/** The pixel at (x, y) of a camera-facing plane filling the view. */
function centre(r, mat, { x = W / 2, y = H / 2, light = false } = {}) {
  const s = new Scene()
  s.setBackground(0x000000)
  if (light) s.addDirectionalLight(0xffffff, 3, { x: 0, y: 0, z: 1 })
  s.addMesh(Geometry.plane(2, 2), mat)
  const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
  cam.setPosition(0, 0, 5)
  cam.lookAt(0, 0, 0)
  const rgba = r.renderRgba(s, cam)
  const i = (y * W + x) * 4
  return [rgba[i], rgba[i + 1], rgba[i + 2]]
}

const texel = (r, g, b, a = 255, cs = 'srgb') => new Texture(1, 1, Uint8Array.of(r, g, b, a), cs)

function png(w, h, rgba) {
  const rows = []
  for (let y = 0; y < h; y++) rows.push(Buffer.from([0]), rgba.subarray(y * w * 4, (y + 1) * w * 4))
  const chunk = (kind, data) => {
    const body = Buffer.concat([Buffer.from(kind), data])
    const len = Buffer.alloc(4)
    len.writeUInt32BE(data.length)
    const crc = Buffer.alloc(4)
    crc.writeUInt32BE(crc32(body))
    return Buffer.concat([len, body, crc])
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr.set([8, 6, 0, 0, 0], 8)
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

// ---- Texture ---------------------------------------------------------------

test('Texture defaults follow three.js', () => {
  const t = new Texture(2, 1, new Uint8Array(8))
  assert.deepEqual([t.width, t.height, t.colorSpace], [2, 1, 'srgb'])
  assert.deepEqual([t.wrapS, t.wrapT], ['clamp', 'clamp'])
  assert.deepEqual([t.magFilter, t.minFilter, t.mipmapFilter], ['linear', 'linear', 'linear'])
  assert.deepEqual([t.offset, t.repeat, t.rotation, t.center], [[0, 0], [1, 1], 0, [0, 0]])
  assert.equal(t.channel, 0)
  assert.equal(t.flipY, true)
})

test('Texture properties round-trip', () => {
  const t = new Texture(1, 1, new Uint8Array(4), 'linear')
  t.wrapS = 'repeat'
  t.wrapT = 'mirror'
  t.magFilter = 'nearest'
  t.minFilter = 'nearest'
  t.mipmapFilter = null
  t.offset = [0.25, 0.5]
  t.repeat = [2, 3]
  t.rotation = 0.5
  t.center = [0.5, 0.5]
  t.channel = 2
  t.colorSpace = 'srgb'
  assert.deepEqual([t.wrapS, t.wrapT], ['repeat', 'mirror'])
  assert.deepEqual([t.magFilter, t.minFilter, t.mipmapFilter], ['nearest', 'nearest', null])
  assert.deepEqual([t.offset, t.repeat, t.center], [[0.25, 0.5], [2, 3], [0.5, 0.5]])
  assert.ok(Math.abs(t.rotation - 0.5) < 1e-6)
  assert.deepEqual([t.channel, t.colorSpace], [2, 'srgb'])
})

test('Texture rejects bad input', () => {
  assert.throws(() => new Texture(2, 2, new Uint8Array(15)), /takes 16 bytes/)
  assert.throws(() => new Texture(0, 1, new Uint8Array(0)), /takes 0 bytes/)
  assert.throws(() => new Texture(1, 1, new Uint8Array(4), 'rec709'), /colorSpace/)
  const t = new Texture(1, 1, new Uint8Array(4))
  assert.throws(() => { t.wrapS = 'border' }, /wrap/)
  assert.throws(() => { t.minFilter = 'cubic' }, /filter/)
  assert.throws(() => { t.channel = 4 }, /channel/)
  assert.throws(() => { t.offset = [1] }, /offset/)
})

test('Texture.fromFile reads a PNG', () => {
  const dir = mkdtempSync(join(tmpdir(), 'threers-tex-'))
  const p = join(dir, 't.png')
  writeFileSync(p, png(2, 3, Uint8Array.from({ length: 24 }, (_, i) => i)))
  const t = Texture.fromFile(p, 'linear')
  assert.ok(t instanceof Texture)
  assert.deepEqual([t.width, t.height, t.colorSpace], [2, 3, 'linear'])
  assert.throws(() => Texture.fromFile(join(dir, 'missing.png')))
  writeFileSync(join(dir, 'bad.png'), 'not a png')
  assert.throws(() => Texture.fromFile(join(dir, 'bad.png')), /PNG/)
})

// ---- map slots -------------------------------------------------------------

test('setMap fills and empties a slot', () => {
  const m = Material.physical()
  assert.equal(m.getMap('sheenColorMap'), null)
  m.setMap('sheenColorMap', new Texture(3, 2, new Uint8Array(24)))
  const got = m.getMap('sheenColorMap')
  assert.ok(got instanceof Texture)
  assert.deepEqual([got.width, got.height], [3, 2])
  m.setMap('sheenColorMap', null)
  assert.equal(m.getMap('sheenColorMap'), null)
})

test('every Physical slot is settable', () => {
  const m = Material.physical()
  for (const slot of [
    'map', 'alphaMap', 'normalMap', 'aoMap', 'emissiveMap', 'roughnessMap',
    'metalnessMap', 'lightMap', 'bumpMap', 'displacementMap',
    'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
    'iridescenceMap', 'iridescenceThicknessMap', 'sheenColorMap',
    'sheenRoughnessMap', 'specularIntensityMap', 'specularColorMap',
    'transmissionMap', 'thicknessMap', 'anisotropyMap',
  ]) {
    m.setMap(slot, texel(255, 255, 255))
    assert.notEqual(m.getMap(slot), null, slot)
  }
})

test('setMap rejects a slot the material lacks, and a non-texture', () => {
  assert.throws(() => Material.basic().setMap('roughnessMap', texel(0, 0, 0)), /no 'roughnessMap'/)
  assert.throws(() => Material.standard().setMap('clearcoatMap', null), /no 'clearcoatMap'/)
  assert.throws(() => Material.basic().setMap('map', Material.basic()), /expected a Texture/)
})

test('a texture edit after setMap does not leak into the material', () => {
  const t = texel(255, 0, 0)
  const m = Material.basic()
  m.setMap('map', t)
  t.wrapS = 'repeat'
  assert.equal(m.getMap('map').wrapS, 'clamp')
})

// ---- rendering -------------------------------------------------------------

test('a map tints the surface', (t) => {
  const r = renderer(t)
  if (!r) return
  const m = Material.basic({ color: 0xffffff })
  m.setMap('map', texel(255, 0, 0))
  assert.deepEqual(centre(r, m), [255, 0, 0])
})

test('colorSpace decides how the texel is read', (t) => {
  const r = renderer(t)
  if (!r) return
  const srgb = Material.basic({ color: 0xffffff })
  srgb.setMap('map', texel(128, 128, 128, 255, 'srgb'))
  const linear = Material.basic({ color: 0xffffff })
  linear.setMap('map', texel(128, 128, 128, 255, 'linear'))
  // sRGB 128 decodes and re-encodes to itself; linear 0.502 encodes to 188.
  assert.ok(Math.abs(centre(r, srgb)[0] - 128) <= 1)
  assert.ok(Math.abs(centre(r, linear)[0] - 188) <= 1)
})

test('offset and wrap move the image', (t) => {
  const r = renderer(t)
  if (!r) return
  const tex = new Texture(2, 1, Uint8Array.of(255, 0, 0, 255, 0, 0, 255, 255))
  tex.magFilter = 'nearest'
  tex.minFilter = 'nearest'
  const m = Material.basic({ color: 0xffffff })
  m.setMap('map', tex)
  const left = { x: W / 4 }
  const right = { x: (3 * W) / 4 }
  assert.deepEqual(centre(r, m, left), [255, 0, 0])
  assert.deepEqual(centre(r, m, right), [0, 0, 255])
  tex.offset = [0.5, 0]
  tex.wrapS = 'repeat'
  m.setMap('map', tex)
  assert.deepEqual(centre(r, m, left), [0, 0, 255])
  assert.deepEqual(centre(r, m, right), [255, 0, 0])
})

// ---- MeshPhysicalMaterial --------------------------------------------------

test('physical takes every three.js scalar', () => {
  const m = Material.physical({
    specularIntensity: 0.5,
    specularColor: 0xff8040,
    sheen: 1,
    sheenColor: 0x808080,
    sheenRoughness: 0.3,
    iridescence: 1,
    iridescenceIOR: 1.8,
    iridescenceThicknessRange: [200, 800],
    anisotropy: 0.6,
    anisotropyRotation: 1.2,
    attenuationColor: 0x40a0ff,
    attenuationDistance: 0.5,
    dispersion: 0.2,
    transmission: 1,
    thickness: 0.4,
  })
  assert.equal(m.kind, 'physical')
})

test('physical validates its scalars', () => {
  assert.throws(() => Material.physical({ sheen: 2 }), /sheen/)
  assert.throws(() => Material.physical({ iridescenceIOR: 3 }), /iridescenceIOR/)
  assert.throws(() => Material.physical({ iridescenceThicknessRange: [-1, 400] }), /iridescenceThicknessRange/)
  assert.throws(() => Material.physical({ iridescenceThicknessRange: [1, 2, 3] }), /\[min, max\]/)
  assert.throws(() => Material.physical({ attenuationDistance: -1 }), /attenuationDistance/)
  assert.throws(() => Material.physical({ specularColor: 0x1000000 }), /specularColor/)
  assert.throws(() => Material.standard({ sheen: 1 }), /'sheen' does not apply/)
})

test('specularIntensity scales the highlight', (t) => {
  const r = renderer(t)
  if (!r) return
  // A black dielectric lit straight on shows only its specular lobe, which
  // three.js scales (F0 and F90) by specularIntensity.
  const lit = (specularIntensity) =>
    centre(r, Material.physical({ color: 0x000000, roughness: 0.5, specularIntensity }), { light: true })[0]
  const [full, half, off] = [lit(1), lit(0.5), lit(0)]
  assert.ok(full > half && half > off, `${full} > ${half} > ${off}`)
  assert.equal(off, 0)
})

// ---- MeshToonMaterial ------------------------------------------------------

test('toon gradientMap tints each light', (t) => {
  const r = renderer(t)
  if (!r) return
  // A light of intensity π straight on, through a one-texel red ramp: three.js
  // gives irradiance × ramp ÷ π = pure red on a white toon surface.
  const lit = (gradient) => {
    const m = Material.toon({ color: 0xffffff })
    assert.equal(m.kind, 'toon')
    m.setMap('gradientMap', gradient)
    const s = new Scene()
    s.setBackground(0x000000)
    s.addDirectionalLight(0xffffff, Math.PI, { x: 0, y: 0, z: 1 })
    s.addMesh(Geometry.plane(2, 2), m)
    const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
    cam.setPosition(0, 0, 5)
    cam.lookAt(0, 0, 0)
    const rgba = r.renderRgba(s, cam)
    const i = ((H / 2) * W + W / 2) * 4
    return [rgba[i], rgba[i + 1], rgba[i + 2]]
  }
  const red = texel(255, 0, 0, 255, 'linear')
  red.magFilter = 'nearest'
  red.minFilter = 'nearest'
  assert.deepEqual(lit(red), [255, 0, 0])
  assert.deepEqual(lit(null), [255, 255, 255])
})

test('toon takes Lambert-like options only', () => {
  Material.toon({ color: 0x123456, emissive: 0x010203, opacity: 0.5, side: 'double' })
  assert.throws(() => Material.toon({ roughness: 0.5 }), /'roughness' does not apply/)
})
