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

Same names as `threers-py`: `headless` (default), `physics`, `animation`.
