// Scene.raycast / raycastFromCamera: three.js's intersection fields, material
// sides, line thresholds, handle lists, recursion, near/far and firstHitOnly.
// Run after `npm run build`: `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { Scene, Geometry, Material, OrthographicCamera, PerspectiveCamera } = require('..')

const down = { x: 0, y: 0, z: -1 }
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps

test('a mesh hit carries face, faceIndex, normal, uv and barycoord', () => {
  const s = new Scene()
  const h = s.addMesh(Geometry.plane(2, 2), Material.basic())
  const hits = s.raycast({ x: 0.5, y: -0.25, z: 5 }, down)
  assert.equal(hits.length, 1)
  const hit = hits[0]
  assert.equal(hit.handle, h)
  assert.ok(close(hit.distance, 5))
  assert.ok([0, 1].includes(hit.faceIndex))
  const { a, b, c, normal, materialIndex } = hit.face
  assert.ok([a, b, c].every((v) => Number.isInteger(v) && v >= 0 && v < 4))
  assert.equal(materialIndex, 0)
  assert.ok(close(normal.z, 1), 'geometric normal')
  assert.ok(close(hit.normal.z, 1), 'normal attribute, facing the ray')
  assert.ok(close(hit.uv.x, 0.75) && close(hit.uv.y, 0.375), JSON.stringify(hit.uv))
  const w = hit.barycoord
  assert.ok(close(w.x + w.y + w.z, 1))
  assert.equal(hit.polygonIndex, 0)
  for (const k of ['uv1', 'instanceId', 'index', 'distanceToRay']) assert.equal(hit[k], undefined, k)
})

test('material sides cull as three.js does', () => {
  const s = new Scene()
  const front = s.addMesh(Geometry.plane(1, 1), Material.basic())
  const back = s.addMesh(Geometry.plane(1, 1), Material.basic({ side: 'back' }))
  const both = s.addMesh(Geometry.plane(1, 1), Material.basic({ side: 'double' }))
  // Off the plane's diagonal: a ray along a shared edge hits both triangles.
  const fromFront = s.raycast({ x: 0.1, y: 0.3, z: 5 }, down).map((h) => h.handle).sort()
  const fromBack = s.raycast({ x: 0.1, y: 0.3, z: -5 }, { x: 0, y: 0, z: 1 }).map((h) => h.handle).sort()
  assert.deepEqual(fromFront, [front, both].sort())
  assert.deepEqual(fromBack, [back, both].sort())
  // The interpolated normal always faces the ray.
  const behind = s.raycast({ x: 0.1, y: 0.3, z: -5 }, { x: 0, y: 0, z: 1 }, undefined, undefined, { objects: [both] })
  assert.ok(close(behind[0].normal.z, -1))
})

test('line segments hit within lineThreshold, reporting index', () => {
  const s = new Scene()
  // The outline of a 2 × 0.6 rectangle: edges at y = ±0.3, x = ±1.
  const line = s.addLineSegments(Geometry.edges(Geometry.plane(2, 0.6)), Material.line())
  const at = { x: 0.2, y: 0.35, z: 5 }
  const hits = s.raycast(at, down, undefined, undefined, { lineThreshold: 0.1 })
  assert.equal(hits.length, 1, 'only the top edge is that close')
  const [hit] = hits
  assert.equal(hit.handle, line)
  assert.ok(Number.isInteger(hit.index), 'the segment\'s first vertex')
  assert.ok(close(hit.point.y, 0.3), 'the point is on the segment')
  for (const k of ['face', 'faceIndex', 'uv', 'normal', 'barycoord']) assert.equal(hit[k], undefined, k)
  assert.equal(s.raycast(at, down, undefined, undefined, { lineThreshold: 0.01 }).length, 0)
  // three.js's default threshold is 1 world unit.
  assert.equal(s.raycast({ x: 0.2, y: 1.2, z: 5 }, down).length, 1)
  assert.throws(() => s.raycast(at, down, undefined, undefined, { lineThreshold: -1 }), /lineThreshold/)
})

test('objects, recursive, near/far and firstHitOnly', () => {
  const s = new Scene()
  const group = s.addGroup()
  const planes = [0, -1, -2].map((z) => {
    const h = s.addMesh(Geometry.plane(1, 1), Material.basic({ side: 'double' }), 0, 0, z)
    s.setParent(h, group)
    return h
  })
  const other = s.addMesh(Geometry.plane(1, 1), Material.basic(), 0, 0, -3)
  const o = { x: 0.1, y: 0.3, z: 5 }
  assert.deepEqual(s.raycast(o, down).map((h) => h.handle), [...planes, other], 'nearest first')
  assert.deepEqual(s.raycast(o, down, undefined, undefined, { objects: [group] }).map((h) => h.handle), planes)
  assert.deepEqual(s.raycast(o, down, undefined, undefined, { objects: [group], recursive: false }), [])
  assert.deepEqual(s.raycast(o, down, undefined, undefined, { objects: [other, planes[2]] }).map((h) => h.handle),
    [planes[2], other], 'a list, sorted together')
  assert.deepEqual(s.raycast(o, down, 5.5, 6.5).map((h) => h.handle), [planes[1]])
  assert.deepEqual(s.raycast(o, down, undefined, undefined, { near: 5.5, far: 6.5 }).map((h) => h.handle), [planes[1]])
  const first = s.raycast(o, down, undefined, undefined, { firstHitOnly: true })
  assert.equal(first.length, 1)
  assert.equal(first[0].handle, planes[0])
  assert.throws(() => s.raycast(o, down, undefined, undefined, { objects: [999] }), /handle/)
})

test('raycastFromCamera takes either camera kind and the same options', () => {
  const s = new Scene()
  const near = s.addMesh(Geometry.plane(2, 2), Material.basic(), 0, 0, 0)
  const far = s.addMesh(Geometry.plane(2, 2), Material.basic(), 0, 0, -2)
  const persp = new PerspectiveCamera(50, 1, 0.1, 100)
  persp.setPosition(0, 0, 5)
  persp.lookAt(0, 0, 0)
  const hits = s.raycastFromCamera(persp, 0.1, 0.2)
  assert.deepEqual(hits.map((h) => h.handle), [near, far])
  assert.ok(hits[0].uv && hits[0].face, 'full hit records')
  assert.deepEqual(s.raycastFromCamera(persp, 0.1, 0.2, { firstHitOnly: true }).map((h) => h.handle), [near])
  assert.deepEqual(s.raycastFromCamera(persp, 0.1, 0.2, { objects: [far] }).map((h) => h.handle), [far])
  const ortho = new OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
  ortho.setPosition(0, 0, 10)
  ortho.lookAt(0, 0, 0)
  const [hit] = s.raycastFromCamera(ortho, 0.25, -0.5)
  assert.equal(hit.handle, near)
  assert.ok(close(hit.point.x, 0.5, 1e-3) && close(hit.point.y, -1, 1e-3))
  // Distances depend on where the ortho ray starts (three.js: the camera
  // plane); only their difference is fixed.
  const both = s.raycastFromCamera(ortho, 0.25, -0.5)
  assert.ok(close(both[1].distance - both[0].distance, 2, 1e-3))
  assert.throws(() => s.raycastFromCamera(persp, 0, 0, { far: -1 }), /far must be > near/)
})
