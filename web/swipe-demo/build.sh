#!/bin/zsh
# Builds the swipe and next-word demo for the browser into
# web/pkg-swipe-demo/, and with SERVE=1 serves web/ on 0.0.0.0 (PORT,
# default 8787), uncached:
#
#   web/swipe-demo/build.sh
#   SERVE=1 web/swipe-demo/build.sh   # then open http://<host>:8787/swipe-demo/
#
# The page loads its models from web/swipe-demo/models/ (see the README
# there for how each was made).
set -eu
here=${0:A:h}
root=${here:h:h}
. "$root/scripts/wasm-bindgen-cli.sh"
use_matching_wasm_bindgen "$root"
target=${CARGO_TARGET_DIR:-$root/target}
out="$root/web/pkg-swipe-demo"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
cargo build -p threers-swipe-web --target wasm32-unknown-unknown --release
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  "$target/wasm32-unknown-unknown/release/threers_swipe_web.wasm"
echo "built web/pkg-swipe-demo/"
if [[ -n "${SERVE:-}" ]]; then
  exec python3 "$root/web/ui-demo/serve.py" "$root/web" "${PORT:-8787}"
fi
