#!/usr/bin/env bash
set -euo pipefail

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

echo "==> Building threers-probe ($PROFILE) for wasm32-unknown-unknown..."
cargo build --target wasm32-unknown-unknown -p threers-probe --lib $PROFILE_FLAG

echo "==> Generating JS bindings into web/pkg-probe..."
wasm-bindgen \
    "$TARGET_DIR/threers_probe.wasm" \
    --target web \
    --out-dir web/pkg-probe \
    --no-typescript

echo "==> Done. Serve from repo root, then open web/examples/probe.html"
echo "    python3 -m http.server 8080"
