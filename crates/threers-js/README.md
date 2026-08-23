# threers (npm / Deno / Node ESM)

Published JavaScript package for [threers](https://github.com/eugenehp/threers):
the `THREE.*` shim plus **two** WebAssembly builds so browsers only download
what they need.

| Entry | Install / import | Wasm contents | Typical use |
|-------|------------------|---------------|-------------|
| **`threers`** / **`threers/mini`** | `import … from 'threers'` | Core renderer (`--no-default-features`) | Default — smallest download |
| **`threers/full`** | `import … from 'threers/full'` | `wasm-full`: OpenSCAD/CSG, NURBS, native codecs, planet, path tracer | CAD, video encode, GI |

| Runtime | Install | Import |
|---------|---------|--------|
| **npm / Node 18+** | `npm i threers` | `import THREE, { initThreers } from 'threers'` |
| **Deno** | — | `import THREE, { initThreers } from 'npm:threers'` |
| **Browser ESM** | via CDN or bundler | same named exports |

For **native** Node (host GPU / headless without wasm), use
[`threers-node`](../threers-node) (napi-rs). Neon is not used; Deno stays on this
wasm package. Architecture matrix: [`docs/bindings.md`](../../docs/bindings.md).

> WebGPU is required for rendering (Chrome 113+, or Firefox with the flag).
>
> **Arch:** wasm is arch-independent (same bytes on arm64 and x64 hosts).

## Quick start (mini — default)

```js
import THREE, { initThreers } from 'threers';
// or: from 'threers/mini'

await initThreers(); // loads dist/mini/pkg/threers_bg.wasm

const renderer = await THREE.WebGLRenderer.create(canvas);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
```

## Full (CAD / codecs / path tracer)

```js
import THREE, { initThreers } from 'threers/full';

await initThreers(); // loads dist/full/pkg/threers_bg.wasm
```

Build flags for `wasm-full`: `captions`, `openscad` (→ CSG/BVH), `native-codec`,
`nurbs`, `planet`, `raytrace`.

## Deno

```ts
import THREE, { initThreers } from "npm:threers/mini";
// import THREE, { initThreers } from "npm:threers/full";

await initThreers();
```

## Build this package from the monorepo

```bash
# both variants → crates/threers-js/dist/{mini,full}/
./crates/threers-js/build.sh

# local web/pkg only:
VARIANT=mini web/build.sh
VARIANT=full web/build.sh

cd crates/threers-js
npm pack
npm publish
```

## Layout

```text
crates/threers-js/
  package.json     exports `.` / `./mini` / `./full`
  deno.json
  build.sh         builds + stages both variants
  dist/mini/       small wasm + shim
  dist/full/       wasm-full + shim + video-export
  src/lib.rs       workspace marker
```

The working tree under `web/` remains the day-to-day development surface;
this crate is the **publish root** for registries.
