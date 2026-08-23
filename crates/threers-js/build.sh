#!/usr/bin/env bash
# Stage the npm/Deno ESM package into crates/threers-js/dist/.
#
# Builds two browser download variants:
#   dist/mini/  — smallest wasm (core THREE.* renderer)
#   dist/full/  — wasm-full (openscad, codecs, nurbs, planet, raytrace)
#
# Default package entry (`.`) is **mini** so casual `import 'threers'` stays small.
# Opt into the kitchen sink with `import … from 'threers/full'`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$(cd "$(dirname "$0")" && pwd)"
WEB="$ROOT/web"
DIST="$PKG/dist"

PROFILE="${PROFILE:-release}"

stage_variant() {
  local variant="$1"
  local dest="$DIST/$variant"

  echo "==> Building wasm variant=${variant}..."
  PROFILE="$PROFILE" VARIANT="$variant" bash "$WEB/build.sh"

  echo "==> Staging dist/${variant}/..."
  rm -rf "$dest"
  mkdir -p "$dest/pkg"

  cp "$WEB/threejs-shim.js" "$dest/threejs-shim.js"
  cp "$WEB/threejs-shim.d.ts" "$dest/threejs-shim.d.ts"

  # Feature flags + addons match the wasm that was just built.
  [[ -f "$WEB/features.js" ]] && cp "$WEB/features.js" "$dest/features.js"
  for addon in mesh-bvh-addon.js bvh-csg-addon.js nurbs-addon.js \
               mesh-bvh-impl.js mesh-bvh-stub.js \
               bvh-csg-impl.js bvh-csg-stub.js \
               nurbs-impl.js nurbs-stub.js; do
    [[ -f "$WEB/$addon" ]] && cp "$WEB/$addon" "$dest/$addon"
  done

  # Shim always imports video-export.js; mini builds throw at encode-time if
  # native-codec was not compiled in.
  cp "$WEB/video-export.js" "$dest/video-export.js"
  [[ -f "$WEB/video-export.d.ts" ]] && cp "$WEB/video-export.d.ts" "$dest/video-export.d.ts"
  [[ -f "$WEB/video-export-worker.js" ]] && cp "$WEB/video-export-worker.js" "$dest/video-export-worker.js"

  cp -R "$WEB/pkg/." "$dest/pkg/"

  if [[ -d "$WEB/deps" ]]; then
    mkdir -p "$dest/deps"
    cp -R "$WEB/deps/." "$dest/deps/"
  fi

  cat > "$dest/index.js" <<EOF
/** @typedef {import('./threejs-shim.js').default} THREE */
/** threers browser package — variant: ${variant} */
export { default, initThreers } from './threejs-shim.js';
export * from './threejs-shim.js';
EOF

  cat > "$dest/mod.ts" <<EOF
/**
 * threers (${variant}) — Deno / ESM entry.
 *
 * \`\`\`ts
 * import THREE, { initThreers } from "npm:threers/${variant}";
 * await initThreers();
 * \`\`\`
 */
export { default, initThreers } from "./threejs-shim.js";
export * from "./threejs-shim.js";
EOF

  local wasm_kb
  wasm_kb=$(du -k "$dest/pkg/threers_bg.wasm" | awk '{print $1}')
  echo "    $variant wasm ≈ ${wasm_kb} KiB"
}

echo "==> Cleaning dist/"
rm -rf "$DIST"
mkdir -p "$DIST"

stage_variant mini
stage_variant full

# Root convenience re-exports → mini (smaller default download).
cat > "$DIST/index.js" <<'EOF'
/** Default entry is the **mini** browser build. Use `threers/full` for the kitchen sink. */
export { default, initThreers } from './mini/index.js';
export * from './mini/index.js';
EOF

cat > "$DIST/mod.ts" <<'EOF'
/** Default Deno entry — mini. Prefer `npm:threers/full` when you need CAD/codecs. */
export { default, initThreers } from "./mini/mod.ts";
export * from "./mini/mod.ts";
EOF

[[ -f "$ROOT/LICENSE" ]] && cp "$ROOT/LICENSE" "$PKG/LICENSE"
[[ -f "$ROOT/LICENSE-MIT" ]] && cp "$ROOT/LICENSE-MIT" "$PKG/LICENSE-MIT"

# Size summary for pack/CI logs.
echo "==> dist ready"
du -sh "$DIST" "$DIST/mini" "$DIST/full" "$DIST/mini/pkg/threers_bg.wasm" "$DIST/full/pkg/threers_bg.wasm" 2>/dev/null || true
echo "    npm pack / npm publish from $PKG"
echo "    import from 'threers' | 'threers/mini' (small) or 'threers/full'"
