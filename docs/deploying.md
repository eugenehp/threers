# Deploying to the web

Part of the [threers](../README.md) documentation.

A threers page is static files: HTML, ES modules and one `.wasm`. Any CDN
serves it. What decides how fast it loads is three things the host has to get
right — the wasm's content type, its caching, and when its download starts —
and `web/scripts/stage-site.mjs` sets all three up.

## Staging a site

```bash
VARIANT=mini ./web/build.sh             # or full; see "Which wasm" below
node web/scripts/stage-site.mjs         # → site/ (OUT=… to put it elsewhere)
```

`site/` is ready to upload as it is:

| Host | How |
|---|---|
| Vercel | Project root `site/`, no build command; `vercel.json` is in it |
| GitHub Pages | Publish `site/` (`.nojekyll` is in it) |
| Netlify, Cloudflare Pages | Publish directory `site/`; `_headers` and `_redirects` are in it |
| nginx, Caddy, S3 + CloudFront | Upload `site/`; set the headers below yourself |

What the staging does to its copy (`web/` is not changed):

- **Content-hashed wasm.** `threers_bg.wasm` becomes `threers_bg.<hash>.wasm`,
  and every reference to it is rewritten — the default URL in the
  wasm-bindgen glue and explicit ones such as
  `initThreers('/web/pkg/threers_bg.wasm')`. A new build is a new file name,
  so the wasm can be cached for a year and never goes stale.
- **Preload hints** in every page that loads the shim: `modulepreload` for the
  shim and the glue, and `<link rel="preload" as="fetch" type="application/wasm"
  crossorigin>` for the wasm. The wasm download starts with the HTML instead of
  after two module round trips, and the page's own fetch reuses it (checked in
  Chrome: one request for the wasm, started by the preload).
- **Minified JavaScript**, with source maps beside it (esbuild, fetched through
  npx; `MINIFY=0` skips it). The shim goes from 0.70 MB to 0.37 MB, 0.09 MB
  over brotli.
- **Headers** for the hosts that take a file of them.
- **A size check**: it fails if any file is over 25 MB (`MAX_FILE_MB`),
  Cloudflare Pages' per-file limit and the tightest of the hosts above.

Left out: `node_modules`, TypeScript sources and declarations, build scripts,
and the 6 GB of fetched Earth imagery (`web/assets/earth/`, whose `CREDITS` is
kept — host the tiles separately for the Earth demo). Pre-compressed copies are
left out too, because every host above compresses on the fly with brotli;
`PRECOMPRESS=1` writes `.br` and `.gz` beside each file for a server that
serves those (nginx `brotli_static` / `gzip_static`, Caddy `precompressed`).

## The headers, and why

| Files | Header | Why |
|---|---|---|
| `*.wasm` | `Content-Type: application/wasm` | Without it the browser cannot compile while downloading (`WebAssembly.instantiateStreaming`); wasm-bindgen falls back to downloading everything first and logs a warning |
| `*.wasm` | `Cache-Control: public, max-age=31536000, immutable` | Safe only because the names are hashed |
| `*.js`, `*.html` | `Cache-Control: public, max-age=0, must-revalidate` | Unhashed names; a repeat visit costs a 304 per file, not a download |

GitHub Pages takes no headers. It serves `.wasm` as `application/wasm` and
caches everything for ten minutes, and with hashed wasm names that is still
correct — only the HTML and JS revalidate.

None of threers needs cross-origin isolation (`COOP`/`COEP`): it uses no
`SharedArrayBuffer`.

## Which wasm

Measured on 0.0.6, after `wasm-opt -Oz`:

| Build | Raw | Brotli | What is in it |
|---|---|---|---|
| `VARIANT=mini` | 2.53 MB | 0.86 MB | The renderer, scene graph, materials, loaders, animation and post-processing |
| `VARIANT=full` | 4.65 MB | 1.59 MB | Plus OpenSCAD and CSG, NURBS, codecs, planet, path tracer, fluid, USD |

Pick the smallest that has what the page uses. `WEBGL=1` adds the WebGL2
fallback for browsers without WebGPU, at a price: the GL backend brings glow and
naga's GLSL translator, and `web/build.sh` measured mini at 1.64 MiB without it
and 5.41 MiB with it (before `wasm-opt`).

## From a package CDN

The npm package works straight from jsDelivr, unpkg or esm.sh. Their URLs
carry the version, so they are immutable already and cached as such:

```html
<link rel="modulepreload" href="https://cdn.jsdelivr.net/npm/threers@0.0.7/dist/mini/threejs-shim.js">
<link rel="preload" href="https://cdn.jsdelivr.net/npm/threers@0.0.7/dist/mini/pkg/threers_bg.wasm"
      as="fetch" type="application/wasm" crossorigin>
<script type="module">
  import THREE, { initThreers } from 'https://cdn.jsdelivr.net/npm/threers@0.0.7/dist/mini/index.js';
  await initThreers();
</script>
```

Pin the version — `threers@0.0.7`, not `threers` or a range — or the CDN
cannot cache the files for long and a release changes the page under you. The
package ships the plain `.wasm` only; the CDN compresses it.

## In your own bundle

Vite, webpack and esbuild copy `threers_bg.wasm` as an asset and give it a
hashed name themselves (it is referenced through `new URL(…, import.meta.url)`,
which all three understand). Serve their output with the headers above.

## Size in the repository

The repository keeps build output out: `web/pkg*/`, `site/`, the Python
extension module and the path tracer's autosave are ignored. Two of those were
committed before the ignore rules existed — untrack them once:

```bash
git rm --cached viewport_path_trace.rtfc crates/threers-py/python/threers/_native.abi3.so
```
