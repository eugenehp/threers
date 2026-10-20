// Run after `npm run build`: `npm test` (node --test).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { inflateSync } from 'node:zlib'

const require = createRequire(import.meta.url)
const { Scene, PerspectiveCamera, HeadlessRenderer, Vector3 } = require('..')

const CORNERS = [-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1]

test('addQuad adds an object from 12 numbers or 4 points', () => {
  const scene = new Scene()
  assert.equal(scene.addQuad(CORNERS, 0x33aa55), 0)
  const pts = [new Vector3(0, 0, 0), new Vector3(1, 0, 0), { x: 1, y: 1, z: 0 }, { x: 0, y: 1, z: 0 }]
  assert.equal(scene.addQuad(pts, 0xffffff, 4, 2), 1)
  assert.equal(scene.objectCount(), 2)
})

test('addQuad rejects bad input', () => {
  const scene = new Scene()
  assert.throws(() => scene.addQuad([1, 2, 3]), /12 numbers or 4/)
  assert.throws(() => scene.addQuad([...CORNERS.slice(0, 11), NaN]), /non-finite/)
  assert.throws(() => scene.addQuad(new Array(12).fill(0)), /degenerate/)
  assert.throws(() => scene.addQuad(CORNERS, 0, 0), /widthSegments/)
  assert.throws(() => scene.addQuad([1, 2, 3, { x: 0, y: 0, z: 0 }]), /mixes/)
  assert.equal(scene.objectCount(), 0)
})

// Minimal PNG decoder for 8-bit RGBA, non-interlaced (what encode_png writes).
function decodePng(buf) {
  assert.deepEqual([...buf.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10])
  let off = 8
  let width = 0
  let height = 0
  const idat = []
  while (off < buf.length) {
    const len = buf.readUInt32BE(off)
    const type = buf.toString('ascii', off + 4, off + 8)
    const data = buf.subarray(off + 8, off + 8 + len)
    if (type === 'IHDR') {
      width = data.readUInt32BE(0)
      height = data.readUInt32BE(4)
      assert.equal(data[8], 8, 'bit depth')
      assert.equal(data[9], 6, 'RGBA colour type')
    } else if (type === 'IDAT') idat.push(data)
    off += 12 + len
  }
  const raw = inflateSync(Buffer.concat(idat))
  const stride = width * 4
  const out = Buffer.alloc(stride * height)
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)]
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1))
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null
    for (let i = 0; i < stride; i++) {
      const a = i >= 4 ? out[y * stride + i - 4] : 0
      const b = prev ? prev[i] : 0
      const c = prev && i >= 4 ? prev[i - 4] : 0
      let v = line[i]
      if (ft === 1) v += a
      else if (ft === 2) v += b
      else if (ft === 3) v += (a + b) >> 1
      else if (ft === 4) {
        const p = a + b - c
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c)
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c
      }
      out[y * stride + i] = v & 0xff
    }
  }
  return { width, height, rgba: out }
}

function quadScene() {
  const scene = new Scene()
  scene.setBackground(0x2050a0)
  scene.addBox(1, 1, 1, 0xff6633, 0, 0, 0)
  const camera = new PerspectiveCamera(45, 1, 0.1, 100)
  camera.setPosition(3, 2.5, 4)
  camera.lookAt(0, 0, 0)
  return { scene, camera }
}

test('renderQuadView returns a PNG of the right size with 4 panes', (t) => {
  let r
  try {
    r = new HeadlessRenderer(200, 160)
  } catch (e) {
    t.skip(`no GPU adapter: ${e.message}`)
    return
  }
  const { scene, camera } = quadScene()
  const gapColor = 0xff00ff
  const png = r.renderQuadView(scene, camera, { gap: 4, gapColor })
  const { width, height, rgba } = decodePng(png)
  assert.equal(width, 200)
  assert.equal(height, 160)
  const px = (x, y) => [...rgba.subarray((y * width + x) * 4, (y * width + x) * 4 + 3)]
  // Panes are 98 x 78 with a 4 px gap at x 98..101 and y 78..81 (top-left origin).
  const gap = px(99, 79)
  assert.ok(gap[0] > 200 && gap[1] < 60 && gap[2] > 200, `gap is magenta: ${gap}`)
  // Each pane's corner shows the scene background, not the gap colour.
  for (const [x, y] of [[5, 5], [195, 5], [5, 155], [195, 155]]) {
    const p = px(x, y)
    assert.notDeepEqual(p, gap, `pane at ${x},${y}`)
    assert.ok(p[2] > p[0], `pane at ${x},${y} is the blue background: ${p}`)
  }
  // Something was drawn in every pane: the box at its centre differs from
  // the same pane's background.
  for (const [[x, y], [cx, cy]] of [
    [[5, 5], [49, 39]],
    [[195, 5], [151, 39]],
    [[5, 155], [49, 121]],
    [[195, 155], [151, 121]],
  ]) {
    assert.notDeepEqual(px(cx, cy), px(x, y), `box visible in pane centred at ${cx},${cy}`)
  }

  const raw = r.renderQuadView(scene, camera, { format: 'rgba', maximized: 'front' })
  assert.equal(raw.length, 200 * 160 * 4)
  // Maximised: no gap down the middle any more.
  const mid = [...raw.subarray((79 * 200 + 99) * 4, (79 * 200 + 99) * 4 + 3)]
  assert.notDeepEqual(mid, gap)
  assert.throws(() => r.renderQuadView(scene, camera, { maximized: 'diagonal' }), /unknown pane/)
})
