# Parity compare UI

Part of the [threers](../README.md) documentation.

## Parity compare UI

Side-by-side **three.js r165 vs threers wasm** for 113 regression scenes:

```bash
cd tests/parity
npm install                    # puppeteer, pixelmatch (first time)
node server.js                 # http://localhost:8087
```

Open **http://localhost:8087/** — navigate scenes, toggle light/dark theme, view source (JS / TS / Rust tabs).

```bash
cd tests/parity
node run.js                    # core suite → out/compare-results.json
node run-mesh-bvh.js           # mesh-bvh scenes (MESH_BVH=1 build)
node run-bvh-csg.js            # CSG scenes (BVH_CSG=1 build)
```

CI helpers: `scripts/ci-mesh-bvh.sh`, `scripts/ci-bvh-csg.sh`.

After changing Rust or wasm, rebuild and hard-refresh the browser (⌘⇧R) or click **↻** in the compare UI so cached wasm/shim are busted.
