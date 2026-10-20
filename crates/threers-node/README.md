# threers-node (napi-rs)

Native **Node.js** addon for [threers](https://github.com/eugenehp/threers) via
[napi-rs](https://napi.rs) (`#[napi]` macros + generated types).

| Runtime | Package | Notes |
|---------|---------|--------|
| Browser / Deno / Node ESM | [`threers`](../threers-js) | wasm + THREE shim — **not** this crate |
| Node native (arm64 / x64) | **`threers-node`** | this package — per-arch `.node` binaries |
| Python | [`threers`](../threers-py) | PyO3 / maturin |

Neon is intentionally **not** used: napi-rs matches the PyO3-style surface, emits
`.d.ts`, and publishes multi-arch optional dependencies. Deno stays on the wasm
ESM package.

## Install

```bash
npm i threers-node
```

Prebuilt binaries ship as optional deps (`@threers/node-darwin-arm64`, …
`@threers/node-win32-x64-msvc`). Unsupported hosts can build from source:

```bash
cd crates/threers-node
npm i
npm run build
```

## Quick start

**Prefer the THREE.* drop-in** (`threers` npm wasm) when you want the same code
as three.js — see [`../threers-js/examples/cube.html`](../threers-js/examples/cube.html)
and [`examples/cube_three.mjs`](examples/cube_three.mjs).

**`threers-node`** is for native headless PNG / host GPU without wasm:

```js
const {
  Scene,
  PerspectiveCamera,
  HeadlessRenderer,
} = require('threers-node')

const scene = new Scene()
scene.setBackground(0x202030)
scene.addBox(1, 1, 1, 0xff6633, 0, 0.5, 0)
// addBox uses MeshStandardMaterial: as in three.js it renders black until lit.
scene.addAmbientLight(0xffffff, 0.4)
scene.addDirectionalLight(0xffffff, 2.5, { x: 3, y: 5, z: 4 })

const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 100)
camera.setPosition(3, 2, 5)
camera.lookAt(0, 0.5, 0)

const renderer = new HeadlessRenderer(1280, 720)
const png = renderer.renderPng(scene, camera)
require('node:fs').writeFileSync('out.png', png)
```

```bash
node examples/cube_headless.mjs
```

### Quads

Grid geometries carry their quads (each two-triangle cell is one polygon), so
wireframes, `Geometry.edges` and picking work on quads, not triangle halves:

```js
const { Geometry, Material } = require('threers-node')

const plane = Geometry.plane(2, 2, 4, 4)        // three.js PlaneGeometry
plane.quadCount()                                // 16
plane.polygonCounts()                            // [2, 2, …] triangles per polygon
plane.polygonLoops()[0]                          // [a, b, c, d] vertex indices
plane.polygonEdges()                             // [[a, b], …] quad outline edges
plane.positions(); plane.index()                 // the vertex data they index

const soup = Geometry.fromArrays(positions, indices)
soup.inferQuads()                                // pair triangles back into quads (40°/40°)
soup.setPolygons([2, 2, 1])                      // or group them yourself
const smooth = plane.subdivide(2)                // Catmull–Clark → a new all-quad Geometry

// Quad wireframes without diagonals: as a material, or as line segments.
scene.addMesh(plane, Material.basic({ color: 0xffffff, wireframe: true }))
scene.addLineSegments(Geometry.edges(Geometry.box(1, 1, 1)), Material.line({ color: 0x000000 }))
scene.addLineSegments(Geometry.wireframe(plane), Material.line({ color: 0x00ff00 }))

// 12 numbers, or 4 points ({x, y, z} objects or Vector3s), in winding order.
scene.addQuad([-2, 0, -2, 2, 0, -2, 2, 0, 2, -2, 0, 2], 0x33aa55, 8, 8)
Geometry.quad([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 8, 8)

// Which quad is under the cursor? (NDC -1..1, +y up)
const [hit] = scene.raycastFromCamera(camera, 0.1, -0.2)
hit && console.log(hit.handle, hit.polygonIndex, hit.faceIndex, hit.point, hit.uv, hit.face.normal)

// three.js Raycaster settings: firstHitOnly, Line/Points thresholds, near/far,
// and intersectObjects over a list of handles (recursive by default).
scene.raycast({ x: 0, y: 0, z: 5 }, { x: 0, y: 0, z: -1 }, undefined, undefined,
  { firstHitOnly: true, lineThreshold: 0.05, objects: [group], recursive: true })
```

A hit has three.js's intersection fields — `distance`, `point`, `faceIndex`,
`face {a, b, c, normal, materialIndex}`, `normal`, `uv`, `uv1`, `instanceId`,
`index` (lines), `distanceToRay` (points) — plus `handle` and the threers
extensions `polygonIndex` and `barycoord`; fields that do not apply to what
was hit are `undefined`.

OBJ, PLY, FBX and USD keep their quads (and n-gons) on load; `getGeometry`
returns a mesh's geometry and `meshes` the handles of a loaded model's parts:

```js
const model = scene.load('part.obj')             // .glb .gltf .obj .stl .ply .fbx .usd[a|c|z]
scene.getGeometry(model).quadCount()
const parts = scene.meshes(scene.load('assembly.fbx'))
```

### Textures

`Texture` is three.js's `Texture`: RGBA8 pixels (rows top first) or a PNG, with
three.js's sampling state. `setMap` puts it in any slot three.js names:

```js
const { Texture } = require('threers-node')

const wood = Texture.fromFile('wood.png')                    // sRGB colour
const rough = Texture.fromFile('wood_rough.png', 'linear')   // data
wood.wrapS = wood.wrapT = 'repeat'                          // 'repeat' | 'mirror' | 'clamp'
wood.repeat = [4, 4]
wood.rotation = 0.3
wood.center = [0.5, 0.5]
wood.mipmapFilter = 'nearest'                               // null = level 0 only
rough.channel = 1                                           // read uv1

const floor = Material.physical({ sheen: 0.4, sheenColor: 0xffe0c0 })
floor.setMap('map', wood)
floor.setMap('roughnessMap', rough)
floor.setMap('clearcoatMap', new Texture(1, 1, Uint8Array.of(128, 0, 0, 255), 'linear'))
floor.setMap('roughnessMap', null)                          // empty the slot
```

A material keeps the texture state it was given: edit the `Texture` and call
`setMap` again to change it.

### Quad view

A CAD-style quad view — top / front / side orthographic panes plus the
perspective camera, framed on the scene's meshes:

```js
const sheet = renderer.renderQuadView(scene, camera)                  // PNG
const front = renderer.renderQuadView(scene, camera, { maximized: 'front' })
const rgba = renderer.renderQuadView(scene, camera, {
  format: 'rgba',
  layout: ['top', 'perspective', 'front', 'side'],  // grid order, TL TR BL BR
  gap: 4, border: 8, gapColor: 0x303030, margin: 0.1,
  backgrounds: { perspective: 0x1d2330 },           // per-pane, over scene.background
})
```

`setViewport(x, y, w, h)`, `setScissor(x, y, w, h)` and `setScissorTest(on)`
restrict `renderPng` / `renderRgba` to a rectangle (output pixels, bottom-left
origin, as in three.js); `resetViewport()` undoes it. With `setAutoClear(false)`
each render draws over the last, so `clear(color)` once and render several
viewports to compose one image.

### Scene API

| Area | Methods |
|------|---------|
| Meshes | `addMesh(geometry, material, x?, y?, z?)`, `addLineSegments(geometry, lineMaterial)`, `addGroup()`, `setParent(child, parent?)`, `addBox` / `addSphere` / `addQuad` |
| Geometry | `Geometry.box` `plane` `sphere` `cylinder` `cone` `torus` `torusKnot` `ring` `circle` `capsule` `lathe` `tube` `icosahedron` `octahedron` `tetrahedron` `dodecahedron` `quad` `fromArrays` `edges` `wireframe` (three.js parameters and defaults) |
| Materials | `Material.basic` `standard` `physical` `lambert` `phong` `toon` `normal` `line` — `{ color, emissive, emissiveIntensity, roughness, metalness, opacity, transparent, wireframe, side, specular, shininess, clearcoat, clearcoatRoughness, transmission, ior, thickness, depthTest }`, and MeshPhysicalMaterial's `{ specularIntensity, specularColor, sheen, sheenColor, sheenRoughness, iridescence, iridescenceIOR, iridescenceThicknessRange, anisotropy, anisotropyRotation, attenuationColor, attenuationDistance, dispersion }`; options a material does not use throw. `material.setMap(slot, texture)` / `getMap(slot)` fill three.js's map slots |
| Lights | `addAmbientLight(color, intensity)`, `addDirectionalLight(color, intensity, position, castShadow)`, `addPointLight(color, intensity, position, distance, decay, castShadow)`, `addSpotLight(color, intensity, position, target, angle, penumbra, distance, decay, castShadow)`, `addHemisphereLight(sky, ground, intensity)`, `lightCount()` — three.js r155+ physical units, defaults intensity 1, decay 2 |
| Handles | `setPosition` `setRotation` (radians, XYZ) `setScale` `setVisible` `setCastShadow` `setReceiveShadow` `setMaterial` `getPosition` `getGeometry` `meshes` `remove` `objectCount` |
| Loading | `load(path, material?)`, `loadBytes(bytes, format, material?)` |
| Picking | `raycast(origin, direction, near?, far?, options?)`, `raycastFromCamera(camera, x, y, options?)` — options `{ near, far, firstHitOnly, lineThreshold, pointsThreshold, recursive, objects }` |
| Cameras | `PerspectiveCamera`, `OrthographicCamera(left, right, top, bottom, near, far)` + `setFrustum` |
| Renderer | `renderPng` / `renderRgba` (either camera), `renderQuadView`, `setSize`, `setToneMapping('none' \| 'linear' \| 'aces' \| 'neutral' \| 'agx', exposure)`, `setShadowsEnabled`, viewport / scissor / `setAutoClear` / `clear` |

Invalid arguments throw with `code: 'InvalidArg'`.

## Tests

```bash
npm run build && npm test
```

## Architectures

| Triple | npm optional package |
|--------|----------------------|
| `aarch64-apple-darwin` | `@threers/node-darwin-arm64` |
| `x86_64-apple-darwin` | `@threers/node-darwin-x64` |
| `x86_64-unknown-linux-gnu` | `@threers/node-linux-x64-gnu` |
| `aarch64-unknown-linux-gnu` | `@threers/node-linux-arm64-gnu` |
| `x86_64-pc-windows-msvc` | `@threers/node-win32-x64-msvc` |

See [docs/bindings.md](../../docs/bindings.md) and the release workflow
`.github/workflows/release-language-packages.yml`.

## Cargo features

Same names as `threers-py`: `headless` (default), `usd` (default; `.usd*` in
`load`), `physics`, `animation`.
