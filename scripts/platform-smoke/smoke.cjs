// Load a threers-node addon and draw one red sphere: node smoke.cjs <file.node>
// Exits non-zero unless the pixels come back right. Used by test-linux-builds.sh.
const m = require(require('path').resolve(process.argv[2]))
const { Scene, Geometry, Material, PerspectiveCamera, HeadlessRenderer } = m
console.log(`node addon loaded on ${process.platform}-${process.arch}, node ${process.version}`)
const r = new HeadlessRenderer(64, 64)
const s = new Scene()
s.setBackground(0x204060)
s.addMesh(Geometry.sphere(1), Material.basic({ color: 0xff0000 }))
const c = new PerspectiveCamera(50, 1, 0.1, 100)
c.setPosition(0, 0, 4)
c.lookAt(0, 0, 0)
const px = r.renderRgba(s, c)
const i = (32 * 64 + 32) * 4
const centre = [px[i], px[i + 1], px[i + 2]]
const corner = [px[0], px[1], px[2]]
const ok = centre[0] > 240 && centre[1] < 10 && centre[2] < 10 && corner[2] > 80
console.log(`node render: centre ${centre}, corner ${corner}: ${ok ? 'ok' : 'WRONG'}`)
process.exit(ok ? 0 : 1)
