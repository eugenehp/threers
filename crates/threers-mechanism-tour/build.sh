#!/usr/bin/env bash
# Build the browser tour and drop it where the page expects it.
#
#   crates/threers-mechanism-tour/build.sh
#   python3 -m http.server --directory web 8080
#   open http://localhost:8080/mechanism-tour/
#
# Uses `wasm-bindgen` directly rather than `wasm-pack` — the CLI comes with the
# `wasm-bindgen-cli` crate, which the toolchain needs anyway.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$root/web/pkg-mechanism-tour"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen is required: cargo install wasm-bindgen-cli" >&2
  exit 1
fi

# The CLI and the crate must be the same version or the generated glue will not
# match the module's exports, and the failure shows up in the browser rather
# than here.
crate_version="$(grep -A1 '^name = "wasm-bindgen"$' "$root/Cargo.lock" | grep '^version' | head -1 | cut -d'"' -f2)"
cli_version="$(wasm-bindgen --version | awk '{print $2}')"
if [ "$crate_version" != "$cli_version" ]; then
  echo "version mismatch: Cargo.lock has wasm-bindgen $crate_version, CLI is $cli_version" >&2
  echo "  cargo install -f wasm-bindgen-cli --version $crate_version" >&2
  exit 1
fi

# wasm32 has no threads, so the SCAD evaluator and the CSG kernel run on the
# main stack rather than on the 1 GB worker they get natively. The default is
# 1 MB, which a deep boolean will walk straight off the end of; 32 MB costs
# nothing until it is used and covers everything the native path handles.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-zstack-size=33554432"

cargo build \
  --manifest-path "$root/crates/threers-mechanism-tour/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --release

wasm-bindgen \
  "$root/target/wasm32-unknown-unknown/release/threers_mechanism_tour.wasm" \
  --target web \
  --out-dir "$out" \
  --out-name threers_mechanism_tour

echo
echo "Built -> web/pkg-mechanism-tour/  ($(du -h "$out/threers_mechanism_tour_bg.wasm" | cut -f1))"
echo "Serve:  python3 -m http.server --directory $root/web 8080"
echo "Open:   http://localhost:8080/mechanism-tour/"
