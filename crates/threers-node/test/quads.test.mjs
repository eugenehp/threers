// Quads: polygon access, quad-aware wireframes and edges, loaders that keep
// quads, and the quad view / viewport compositing controls.
// Run after `npm run build`: `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const { Scene, Geometry, Material, OrthographicCamera, PerspectiveCamera, HeadlessRenderer } = require('..')

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const W = 64
const H = 64
const BG = 0x204060
const BG_PX = [32, 64, 96]
const px = (rgba, x, y, w = W) => [...rgba.subarray((y * w + x) * 4, (y * w + x) * 4 + 3)]
const near = (a, b, tol = 3) => a.every((v, i) => Math.abs(v - b[i]) <= tol)

function renderer(t, w = W, h = H) {
  try {
    return new HeadlessRenderer(w, h)
  } catch (e) {
    t.skip(`no GPU adapter: ${e.message}`)
    return null
  }
}

// Two triangles over the square -1..1, with no polygon grouping.
const SQUARE = () => Geometry.fromArrays([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], [0, 1, 2, 0, 2, 3])

test('grid geometries carry their quads', () => {
  const plane = Geometry.plane(2, 2, 3, 2)
  assert.deepEqual(plane.polygonCounts(), [2, 2, 2, 2, 2, 2])
  assert.equal(plane.polygonCount(), 6)
  assert.equal(plane.quadCount(), 6)
  for (const loop of plane.polygonLoops()) assert.equal(loop.length, 4)
  // 3 x 2 cells: 3 * 3 horizontal + 4 * 2 vertical edges, no diagonals.
  assert.equal(plane.polygonEdges().length, 17)
  assert.ok(plane.polygonEdges().every((e) => e.length === 2))

  assert.equal(Geometry.box(1, 1, 1).quadCount(), 6)
  assert.equal(Geometry.quad([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 4, 2).quadCount(), 8)
  const sphere = Geometry.sphere(1, 8, 4)
  assert.ok(sphere.quadCount() > 0 && sphere.quadCount() < sphere.polygonCount(), 'quads, plus pole triangles')

  const positions = plane.positions()
  assert.equal(positions.length, plane.vertexCount() * 3)
  const loop = plane.polygonLoops()[0]
  assert.ok(loop.every((i) => i < plane.vertexCount()))
  assert.equal(plane.index().length, plane.triangleCount() * 3)
})

test('setPolygons groups triangles into quads', () => {
  const sq = SQUARE()
  assert.equal(sq.polygonCounts(), null)
  assert.equal(sq.polygonCount(), 2)
  assert.equal(sq.quadCount(), 0)
  sq.setPolygons([2])
  assert.deepEqual(sq.polygonCounts(), [2])
  assert.equal(sq.quadCount(), 1)
  assert.deepEqual([...sq.polygonLoops()[0]].sort(), [0, 1, 2, 3])
  assert.equal(sq.polygonEdges().length, 4)
  assert.throws(() => sq.setPolygons([3]), /sum to 3 triangles, the geometry has 2/)
  assert.throws(() => sq.setPolygons([0, 2]), /at least 1/)
  assert.deepEqual(sq.polygonCounts(), [2], 'unchanged after a rejected call')
})

test('wireframe and edges follow polygon outlines', () => {
  // Each line segment is a vertex pair.
  const segs = (g) => g.vertexCount() / 2
  assert.equal(segs(Geometry.wireframe(Geometry.plane(1, 1))), 4, 'quad outline, no diagonal')
  assert.equal(segs(Geometry.wireframe(SQUARE())), 5, 'two triangles share the diagonal')
  assert.equal(segs(Geometry.wireframe(Geometry.plane(1, 1, 3, 2))), 17)
  assert.equal(segs(Geometry.edges(Geometry.box(1, 1, 1))), 12)
  assert.equal(segs(Geometry.edges(Geometry.sphere(1, 16, 8), 90)), 0, 'nothing that sharp on a sphere')
  assert.throws(() => Geometry.edges(Geometry.box(), 200), /thresholdAngle/)
  assert.throws(() => Geometry.wireframe(Material.basic()), /expected a Geometry/)
})

test('line segments and quad wireframes render without diagonals', (t) => {
  const r = renderer(t)
  if (!r) return
  const cam = new OrthographicCamera(-1.25, 1.25, 1.25, -1.25, 0.1, 10)
  cam.setPosition(0, 0, 5)
  cam.lookAt(0, 0, 0)
  // Anything other than background in the 5x5 block at the centre?
  const centreHit = (s) => {
    const rgba = r.renderRgba(s, cam)
    for (let y = 30; y <= 34; y++) for (let x = 30; x <= 34; x++) if (!near(px(rgba, x, y), BG_PX)) return true
    return false
  }
  const edgeHit = (s) => {
    const rgba = r.renderRgba(s, cam)
    // The square's left edge sits at x = -1, pixel 6.4; scan around it.
    for (let x = 4; x <= 9; x++) if (!near(px(rgba, x, 32), BG_PX)) return true
    return false
  }
  const scene = (build) => {
    const s = new Scene()
    s.setBackground(BG)
    build(s)
    return s
  }
  const red = Material.line({ color: 0xff0000 })
  const wire = Material.basic({ color: 0xffffff, wireframe: true })

  const quadLines = scene((s) => s.addLineSegments(Geometry.wireframe(Geometry.plane(2, 2)), red))
  assert.ok(edgeHit(quadLines) && !centreHit(quadLines), 'quad outline only')
  const triLines = scene((s) => s.addLineSegments(Geometry.wireframe(SQUARE()), red))
  assert.ok(centreHit(triLines), 'triangle wireframe has the diagonal')

  const quadWire = scene((s) => s.addMesh(Geometry.plane(2, 2), wire))
  assert.ok(edgeHit(quadWire) && !centreHit(quadWire), 'wireframe material on a quad: no diagonal')
  const triWire = scene((s) => s.addMesh(SQUARE(), wire))
  assert.ok(centreHit(triWire), 'wireframe material on two triangles: diagonal')
  const grouped = SQUARE()
  grouped.setPolygons([2])
  assert.ok(!centreHit(scene((s) => s.addMesh(grouped, wire))), 'setPolygons removes it')

  const lineRgba = r.renderRgba(quadLines, cam)
  let red_seen = false
  for (let x = 4; x <= 9; x++) {
    const p = px(lineRgba, x, 32)
    if (p[0] > 200 && p[1] < 40) red_seen = true
  }
  assert.ok(red_seen, 'Material.line colour')
})

test('line segments validate their inputs; getGeometry and meshes', () => {
  const s = new Scene()
  assert.throws(() => s.addLineSegments(Geometry.wireframe(Geometry.box()), Material.basic()), /Material\.line/)
  assert.throws(() => s.addMesh(Geometry.box(), Material.line()), /addLineSegments/)
  assert.throws(() => s.addLineSegments(Geometry.fromArrays([0, 0, 0, 1, 0, 0, 0, 1, 0]), Material.line()), /even vertex count/)
  assert.throws(() => Material.line({ roughness: 0.5 }), /does not apply/)
  assert.equal(Material.line({ color: 0x00ff00, opacity: 0.5, depthTest: false }).kind, 'line')
  const l = s.addLineSegments(Geometry.edges(Geometry.box()), Material.line())
  assert.equal(s.getGeometry(l).vertexCount(), 24)
  assert.throws(() => s.setMaterial(l, Material.basic()), /Material\.line/)
  s.setMaterial(l, Material.line({ color: 0xff00ff }))
  const m = s.addMesh(Geometry.plane(1, 1, 2, 2), Material.basic())
  assert.throws(() => s.setMaterial(m, Material.line()), /only applies to line segments/)
  assert.equal(s.getGeometry(m).quadCount(), 4)
  assert.deepEqual(s.meshes(m), [m])
  const g = s.addGroup()
  assert.throws(() => s.getGeometry(g), /no geometry/)
  assert.deepEqual(s.meshes(g), [])
})

const tmp = mkdtempSync(join(tmpdir(), 'threers-node-quads-'))

test('loaders keep quads: OBJ, PLY, FBX, USD', () => {
  const s = new Scene()
  writeFileSync(join(tmp, 'quad.obj'), 'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n')
  const obj = s.load(join(tmp, 'quad.obj'))
  assert.deepEqual(s.getGeometry(obj).polygonCounts(), [2])
  assert.equal(s.getGeometry(obj).quadCount(), 1)

  const ply = [
    'ply', 'format ascii 1.0', 'element vertex 4', 'property float x', 'property float y',
    'property float z', 'element face 1', 'property list uchar int vertex_indices', 'end_header',
    '0 0 0', '1 0 0', '1 1 0', '0 1 0', '4 0 1 2 3', '',
  ].join('\n')
  const p = s.loadBytes(Buffer.from(ply), 'ply')
  assert.equal(s.getGeometry(p).quadCount(), 1)

  const fbx = s.load(join(REPO, 'src/loaders/fbx/testdata/blender.fbx'))
  const parts = s.meshes(fbx)
  assert.equal(parts.length, 2, 'both meshes of the file')
  for (const part of parts) assert.equal(s.getGeometry(part).quadCount(), 6, 'cube faces stay quads')
  assert.deepEqual(s.meshes(fbx), parts, 'the same handles again')
  s.setPosition(parts[0], 0, 3, 0)

  const usd = s.load(join(REPO, 'src/loaders/usd/testdata/polygons.usda'))
  const [faces] = s.meshes(usd)
  const geo = s.getGeometry(faces)
  assert.deepEqual(geo.polygonCounts(), [2, 1, 3], 'a quad, a triangle and a pentagon')
  assert.deepEqual(geo.polygonLoops().map((l) => l.length), [4, 3, 5])
})

test('renderQuadView takes layout, border, per-pane backgrounds and margin', (t) => {
  const r = renderer(t, 200, 160)
  if (!r) return
  const s = new Scene()
  s.setBackground(0x000000)
  s.addMesh(Geometry.box(1, 1, 1), Material.normal())
  const cam = new PerspectiveCamera(45, 1, 0.1, 100)
  cam.setPosition(3, 2.5, 4)
  cam.lookAt(0, 0, 0)
  const backgrounds = { top: 0xff0000, front: 0x00ff00, side: 0x0000ff, perspective: 0xffff00 }
  const rgba = r.renderQuadView(s, cam, {
    format: 'rgba',
    layout: ['side', 'front', 'perspective', 'top'],
    gap: 4,
    border: 6,
    gapColor: 0xff00ff,
    backgrounds,
    margin: 0.5,
  })
  const p = (x, y) => px(rgba, x, y, 200)
  assert.ok(near(p(2, 2), [255, 0, 255]), `border is the gap colour: ${p(2, 2)}`)
  // Corners of each pane, inside the 6 px border: side, front / perspective, top.
  assert.ok(near(p(10, 10), [0, 0, 255]), `top-left is side: ${p(10, 10)}`)
  assert.ok(near(p(190, 10), [0, 255, 0]), `top-right is front: ${p(190, 10)}`)
  assert.ok(near(p(10, 150), [255, 255, 0]), `bottom-left is perspective: ${p(10, 150)}`)
  assert.ok(near(p(190, 150), [255, 0, 0]), `bottom-right is top: ${p(190, 150)}`)

  assert.throws(() => r.renderQuadView(s, cam, { layout: ['top', 'front', 'side'] }), /4 pane names/)
  assert.throws(() => r.renderQuadView(s, cam, { layout: ['top', 'top', 'side', 'front'] }), /twice/)
  assert.throws(() => r.renderQuadView(s, cam, { layout: ['top', 'front', 'side', 'iso'] }), /unknown pane/)
  assert.throws(() => r.renderQuadView(s, cam, { border: -1 }), /border/)
  assert.throws(() => r.renderQuadView(s, cam, { backgrounds: { side: 0x1000000 } }), /backgrounds\.side/)
})

test('autoClear off + clear composes viewports', (t) => {
  const r = renderer(t)
  if (!r) return
  const s = new Scene()
  s.setBackground(BG)
  s.addMesh(Geometry.box(4, 4, 1), Material.basic({ color: 0xff0000 }))
  const cam = new OrthographicCamera()
  cam.setPosition(0, 0, 5)
  assert.equal(r.autoClear(), true)
  r.setAutoClear(false)
  r.clear(0x00ff00)
  r.setViewport(0, 0, W / 2, H) // left half, bottom-left origin
  const rgba = r.renderRgba(s, cam)
  // As in three.js, with autoClear off the background colour is not drawn
  // either; only the mesh lands over the cleared green.
  assert.ok(near(px(rgba, 10, 32), [255, 0, 0]), `left half is the scene: ${px(rgba, 10, 32)}`)
  assert.ok(near(px(rgba, 54, 32), [0, 255, 0]), `right half keeps the clear colour: ${px(rgba, 54, 32)}`)
  r.setAutoClear(true)
  r.resetViewport()
  assert.ok(near(px(r.renderRgba(s, cam), 54, 32), [255, 0, 0]), 'whole frame again')
  assert.throws(() => r.clear(0x1000000), /clear: color/)
  assert.throws(() => r.clear(0, 2), /alpha/)
})

test('inferQuads pairs triangles back into quads', () => {
  const sq = SQUARE()
  assert.equal(sq.inferQuads(), 1)
  assert.equal(sq.quadCount(), 1)
  assert.equal(sq.inferQuads(), 0, 'nothing left to pair')
  // A 3 x 3 grid stripped of its grouping comes back as 9 quads.
  const plane = Geometry.plane(3, 3, 3, 3)
  const soup = Geometry.fromArrays(plane.positions(), plane.index())
  assert.equal(soup.quadCount(), 0)
  assert.equal(soup.inferQuads(), 9)
  assert.equal(soup.quadCount(), 9)
  assert.throws(() => sq.inferQuads(200), /maxFaceAngle/)
  assert.throws(() => sq.inferQuads(40, -1), /maxShapeAngle/)
})

test('raycast reports the quad that was hit', () => {
  const s = new Scene()
  const h = s.addMesh(Geometry.plane(2, 2, 2, 2), Material.basic())
  const down = { x: 0, y: 0, z: -1 }
  const hit = (x, y) => {
    const hits = s.raycast({ x, y, z: 5 }, down)
    assert.equal(hits.length, 1)
    return hits[0]
  }
  const first = hit(0.3, 0.6) // off both diagonals of its cell
  assert.equal(first.handle, h)
  assert.ok(Math.abs(first.distance - 5) < 1e-4)
  assert.ok(Math.abs(first.point.x - 0.3) < 1e-4 && Math.abs(first.point.z) < 1e-4)
  // Points near each edge of one cell straddle its diagonal: two triangles,
  // one polygon.
  const cell = [[0.15, 0.5], [0.5, 0.15], [0.85, 0.5], [0.5, 0.85]].map(([x, y]) => hit(x, y))
  assert.ok(new Set(cell.map((c) => c.faceIndex)).size === 2, 'both triangles were hit')
  assert.ok(cell.every((c) => c.polygonIndex === first.polygonIndex), 'all in one quad')
  assert.notEqual(hit(-0.3, -0.6).polygonIndex, first.polygonIndex)
  assert.deepEqual(s.raycast({ x: 5, y: 5, z: 5 }, down), [])

  s.setPosition(h, 0, 0, -1)
  assert.ok(Math.abs(hit(0.3, 0.6).distance - 6) < 1e-4, 'transforms are applied')
  const plain = new Scene()
  plain.addMesh(SQUARE(), Material.basic())
  assert.equal(plain.raycast({ x: 0.2, y: 0.6, z: 5 }, down)[0].polygonIndex, undefined, 'no grouping')

  const cam = new OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
  cam.setPosition(0, 0, 10)
  cam.lookAt(0, 0, 0)
  const picked = s.raycastFromCamera(cam, 0.15, 0.3)
  assert.equal(picked.length, 1)
  assert.ok(Math.abs(picked[0].point.x - 0.3) < 1e-3 && Math.abs(picked[0].point.y - 0.6) < 1e-3)
  const persp = new PerspectiveCamera(50, 1, 0.1, 100)
  persp.setPosition(0, 0, 5)
  persp.lookAt(0, 0, 0)
  assert.equal(s.raycastFromCamera(persp, 0, 0)[0].handle, h)

  assert.throws(() => s.raycast({ x: 0, y: 0, z: 5 }, { x: 0, y: 0, z: 0 }), /direction/)
  assert.throws(() => s.raycast({ x: 0, y: 0, z: 5 }, down, 2, 1), /far must be > near/)
  assert.throws(() => s.raycastFromCamera(persp, NaN, 0), /finite/)
})

test('subdivide refines polygons into quads', () => {
  const box = Geometry.box(1, 1, 1)
  const once = box.subdivide()
  assert.equal(once.quadCount(), 24, 'each of 6 quads → 4')
  assert.equal(box.subdivide(2).quadCount(), 96)
  assert.equal(box.quadCount(), 6, 'the source is untouched')
  assert.equal(box.subdivide(0).quadCount(), 6)
  // A cube cage shrinks toward a sphere: corners pull in.
  const maxAbs = (g) => Math.max(...g.positions().map(Math.abs))
  assert.ok(maxAbs(box.subdivide(2)) < 0.5)
  // A triangle → 3 quads; an open quad keeps its border with sharpBoundary.
  assert.equal(SQUARE().subdivide().quadCount(), 6)
  // An open quad keeps its border (sharpBoundary, the default); the flag is
  // passed through to the core's boundary rule.
  assert.ok(Math.abs(maxAbs(Geometry.plane(2, 2).subdivide(1)) - 1) < 1e-5, 'border held')
  assert.equal(Geometry.plane(2, 2).subdivide(1, false).quadCount(), 4)
  assert.throws(() => box.subdivide(7), /levels/)
})
