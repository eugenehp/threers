#!/usr/bin/env bash
set -euo pipefail
# Build the ocean demo for the browser into web/pkg-ocean.
# Requires: rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-release}"
FLAG=""; DIR="target/wasm32-unknown-unknown/debug"
if [ "$PROFILE" = "release" ]; then FLAG="--release"; DIR="target/wasm32-unknown-unknown/release"; fi

echo "==> Building threers-ocean ($PROFILE) for wasm32-unknown-unknown..."
cargo build --target wasm32-unknown-unknown -p threers-ocean --lib $FLAG

echo "==> Generating JS bindings into web/pkg-ocean..."
wasm-bindgen "$DIR/threers_ocean.wasm" --target web --out-dir web/pkg-ocean --no-typescript

# Same browser-compat fix web/post-build.sh applies to the main package: wgpu
# 0.20 asks for `maxInterStageShaderComponents`, which current WebGPU removed and
# which Safari rejects outright.
echo "==> Patching requestDevice limits..."
node -e '
const fs = require("fs");
const p = "web/pkg-ocean/threers_ocean.js";
let s = fs.readFileSync(p, "utf8");
const re = /(__wbg_requestDevice_[a-f0-9]+: function\(arg0, arg1\) \{)[\s\S]*?const ret = arg0\.requestDevice\(arg1\);/;
if (re.test(s)) {
  s = s.replace(re, `$1
            if (arg1) {
                const desc = { ...arg1 };
                if (desc.requiredLimits) {
                    const _rl = {};
                    for (const [_k, _v] of Object.entries(desc.requiredLimits)) {
                        if (_k !== "maxInterStageShaderComponents") _rl[_k] = _v;
                    }
                    if (Object.keys(_rl).length) desc.requiredLimits = _rl;
                    else delete desc.requiredLimits;
                }
                delete desc.maxInterStageShaderComponents;
                arg1 = desc;
            }
            const ret = arg0.requestDevice(arg1);`);
  fs.writeFileSync(p, s);
  console.log("    patched");
} else { console.log("    already patched or pattern absent"); }
'
echo "==> Done. Serve with:  python3 -m http.server -d web 8080"
echo "    then open http://localhost:8080/ocean.html"
