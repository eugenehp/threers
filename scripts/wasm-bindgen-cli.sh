# shellcheck shell=bash
# Sourced by the wasm build and test scripts (bash or zsh); not run on its own.
#
#   . "$ROOT/scripts/wasm-bindgen-cli.sh" && use_matching_wasm_bindgen "$ROOT"
#
# Puts a wasm-bindgen CLI (and its test runner) matching the `wasm-bindgen`
# crate in Cargo.lock first on PATH. The two have to agree exactly, and a
# mismatch only shows after the whole cargo build, as a schema-version error.
# The CLI in ~/.cargo/bin is shared with every other project on the machine, so
# reinstalling it to suit this lockfile breaks the next project that wants a
# different one, and theirs breaks this. Instead, when the one on PATH does not
# match, the matching version is installed once into target/tools/ and used
# from there; the global one is left alone.
#
# `--locked` builds it from the Cargo.lock wasm-bindgen-cli shipped with, as CI
# does. That file can name a dependency yanked since (0.2.129 pins yoke-derive
# 0.8.3), and cargo warns about it; the warning is about their lockfile, not
# ours, and a yanked version still builds.
use_matching_wasm_bindgen() {
  local root="$1" want have dir
  want=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")
  [ -n "$want" ] || return 0
  have=$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }')
  [ "$have" = "$want" ] && return 0
  dir="$root/target/tools/wasm-bindgen-$want"
  if [ ! -x "$dir/bin/wasm-bindgen" ]; then
    echo "==> wasm-bindgen CLI on PATH is ${have:-missing}, Cargo.lock has $want;" \
         "installing $want into target/tools/ (once)" >&2
    if ! cargo install wasm-bindgen-cli --version "$want" --locked --root "$dir" >&2; then
      echo "error: could not install wasm-bindgen-cli $want" >&2
      return 1
    fi
  fi
  PATH="$dir/bin:$PATH"
  export PATH
}
