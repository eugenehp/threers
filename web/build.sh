#!/usr/bin/env bash
set -euo pipefail

# Build the threers wasm package and emit JS glue into web/pkg.
# Requires: rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli wasm-pack

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

PROFILE="${PROFILE:-release}"
PROFILE_FLAG=""
TARGET_DIR="target/wasm32-unknown-unknown/debug"
if [ "$PROFILE" = "release" ]; then
    PROFILE_FLAG="--release"
    TARGET_DIR="target/wasm32-unknown-unknown/release"
fi

echo "==> Building threers ($PROFILE) for wasm32-unknown-unknown..."
FEATURES=""
FEATURE_LIST=()
if [ "${BVH_CSG:-}" = "1" ]; then
    FEATURE_LIST+=("bvh-csg")
    echo "    (bvh-csg feature enabled — includes mesh-bvh)"
elif [ "${MESH_BVH:-}" = "1" ]; then
    FEATURE_LIST+=("mesh-bvh")
    echo "    (mesh-bvh feature enabled)"
fi
if [ "${NATIVE_CODEC:-}" = "1" ]; then
    FEATURE_LIST+=("native-codec")
    echo "    (native-codec feature enabled — GIF/APNG/WebM browser export)"
fi
if [ "${OPENSCAD:-}" = "1" ]; then
    FEATURE_LIST+=("openscad")
    echo "    (openscad feature enabled — scad_geometry/scadExport, pulls bvh-csg)"
fi
if [ "${#FEATURE_LIST[@]}" -gt 0 ]; then
    FEATURES="--features $(IFS=,; echo "${FEATURE_LIST[*]}")"
fi
cargo build --target wasm32-unknown-unknown --lib $PROFILE_FLAG $FEATURES

echo "==> Generating JS bindings into web/pkg..."
wasm-bindgen \
    "$TARGET_DIR/threers.wasm" \
    --target web \
    --out-dir web/pkg \
    --no-typescript

echo "==> Patching WebGPU device limits for browser compatibility..."
MESH_BVH="${MESH_BVH:-}" "$WEB/post-build.sh"

echo "==> Done. Open web/index.html with a static server (e.g. python3 -m http.server -d web)."
echo "    Browser must support WebGPU (Chrome 113+ or Firefox Nightly with flag)."
