#!/bin/zsh
# Builds the threers-ui demo for the browser into web/pkg-ui-demo/, and with
# SERVE=1 serves web/ on 0.0.0.0 (PORT, default 8787), uncached; with CERT
# and KEY (a certificate and its key), over https:
#
#   web/ui-demo/build.sh
#   SERVE=1 web/ui-demo/build.sh    # then open http://<host>:8787/ui-demo/
set -eu
here=${0:A:h}
root=${here:h:h}
. "$root/scripts/wasm-bindgen-cli.sh"
use_matching_wasm_bindgen "$root"
target=${CARGO_TARGET_DIR:-$root/target}
# The demo draws through threers' GPU UI pass, but not its JS bindings; and
# wasm SIMD speeds up layout, paint and the software 3D.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg threers_no_js_bindings -C target-feature=+simd128"
cargo build -p threers-ui-demo --target wasm32-unknown-unknown --release
wasm-bindgen --target web --no-typescript --out-dir "$root/web/pkg-ui-demo" \
  "$target/wasm32-unknown-unknown/release/threers_ui_demo.wasm"
# The GPU renderer's face: a browser has no font files to read.
cp "$root/src/openscad/DejaVuSans.ttf" "$here/DejaVuSans-Bold.ttf" "$root/web/pkg-ui-demo/"
# The page shows this, to tell which build a device is running.
date "+%Y-%m-%d %H:%M:%S" > "$root/web/pkg-ui-demo/build.txt"
echo "built web/pkg-ui-demo/"
if [[ -n "${SERVE:-}" ]]; then
  exec python3 "$here/serve.py" "$root/web" "${PORT:-8787}" ${CERT:+"$CERT"} ${KEY:+"$KEY"}
fi
