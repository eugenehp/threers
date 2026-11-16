#!/usr/bin/env bash
# Build the JavaScript bindings and drop them where the web demo expects them.
#
#   crates/threers-fem/build-web.sh
#   python3 -m http.server --directory web 8080
#   open http://localhost:8080/fem/
#
# Plain `--release` (opt-level 3), not the size-tuned `release-wasm` profile:
# this is all float loops, which `opt-level = "z"` leaves several times slower.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$root/web/pkg-fem"
target="${CARGO_TARGET_DIR:-$root/target}"

. "$root/scripts/wasm-bindgen-cli.sh"
use_matching_wasm_bindgen "$root"

cargo build \
  --manifest-path "$root/crates/threers-fem/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --release \
  --features wasm

wasm-bindgen \
  "$target/wasm32-unknown-unknown/release/threers_fem.wasm" \
  --target web \
  --out-dir "$out" \
  --out-name threers_fem

echo
echo "Built -> web/pkg-fem/  ($(du -h "$out/threers_fem_bg.wasm" | cut -f1))"
echo "Serve:  python3 -m http.server --directory $root/web 8080"
echo "Open:   http://localhost:8080/fem/"
