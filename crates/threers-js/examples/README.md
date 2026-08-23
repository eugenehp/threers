# Examples

| File | Description |
|------|-------------|
| [cube.html](examples/cube.html) | **THREE.* drop-in** — spinning cube in the browser |
| [cube.mjs](examples/cube.mjs) | Same scene for bundlers (`import … from 'threers'`) |
| [cube.deno.ts](examples/cube.deno.ts) | Deno — identical THREE setup |

```bash
npm run build
python3 -m http.server -d . 8765
# open http://localhost:8765/examples/cube.html
```

More runtimes: [../examples/README.md](../examples/README.md).
