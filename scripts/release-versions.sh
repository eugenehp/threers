#!/usr/bin/env bash
# Print versions across publish roots — bump these together before a release.
#
# `CHECK=1 ./scripts/release-versions.sh` exits non-zero when any root that
# tracks the crate version has drifted from it. The separately versioned
# crates (threers-physics, threers-probe) are reported but not checked.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

v_cargo() {
  python3 - "$1" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
print(data["package"]["version"])
PY
}

v_json() {
  python3 - "$1" <<'PY'
import sys, json, pathlib
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["version"])
PY
}

v_pyproject() {
  python3 - "$1" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
print(data["project"]["version"])
PY
}

CRATE=$(v_cargo "$ROOT/Cargo.toml")
drift=0

# Roots that track the crate version.
report() { # label version
  printf "%-32s %s" "$1" "$2"
  if [[ "$2" != "$CRATE" ]]; then
    printf "   <- expected %s" "$CRATE"
    drift=1
  fi
  printf "\n"
}

report "crates.io threers"           "$CRATE"
report "npm threers (ESM)"           "$(v_json "$ROOT/crates/threers-js/package.json")"
report "jsr threers (deno.json)"     "$(v_json "$ROOT/crates/threers-js/deno.json")"
report "npm threers-node"            "$(v_json "$ROOT/crates/threers-node/package.json")"
report "npm threers-node (lock)"     "$(v_json "$ROOT/crates/threers-node/package-lock.json")"
report "PyPI threers"                "$(v_pyproject "$ROOT/crates/threers-py/pyproject.toml")"
report "web shim (package.json)"     "$(v_json "$ROOT/web/package.json")"
report "web shim (lock)"             "$(v_json "$ROOT/web/package-lock.json")"

for f in "$ROOT/crates/threers-node/npm"/*/package.json; do
  [[ -f "$f" ]] || continue
  pkg=$(python3 -c "import json; print(json.load(open('$f'))['name'])")
  report "$pkg" "$(v_json "$f")"
done

# Separately versioned — they do not track the crate version, but the bindings
# pin them by version as well as by path, and cargo refuses to resolve a path
# dependency whose version does not satisfy the requirement. Check that pairing.
PHYSICS=$(v_cargo "$ROOT/crates/threers-physics/Cargo.toml")
PROBE=$(v_cargo "$ROOT/crates/threers-probe/Cargo.toml")
ANIM=$(v_cargo "$ROOT/crates/threers-animation/Cargo.toml")
printf "\nindependently versioned:\n"
printf "%-32s %s\n" "crates.io threers-physics"  "$PHYSICS"
printf "%-32s %s\n" "crates.io threers-animation" "$ANIM"
printf "%-32s %s\n" "crates.io threers-probe"    "$PROBE"

for f in "$ROOT/crates/threers-py/Cargo.toml" "$ROOT/crates/threers-node/Cargo.toml"; do
  pinned=$(sed -n 's/^threers-physics = .*version = "\([^"]*\)".*/\1/p' "$f")
  [[ -n "$pinned" ]] || continue
  printf "%-32s %s" "  ${f#"$ROOT/crates/"} pin" "$pinned"
  if [[ "$pinned" != "$PHYSICS" ]]; then
    printf "   <- expected %s" "$PHYSICS"
    drift=1
  fi
  printf "\n"
done

# These two are published separately and their 0.0.1 is already on crates.io, so
# a release that changes them has to bump them too — `cargo publish` cannot
# replace a version. Check against the index before publishing:
#   curl -s https://crates.io/api/v1/crates/threers-physics | python3 -c \
#     "import json,sys; print(json.load(sys.stdin)['crate']['max_version'])"

# The newest changelog entry should name the crate version.
latest=$(grep -m1 -o '^## \[[0-9][^]]*\]' "$ROOT/CHANGELOG.md" | tr -d '#[] ')
printf "\n%-32s %s" "CHANGELOG newest entry" "$latest"
if [[ "$latest" != "$CRATE" ]]; then
  printf "   <- expected %s" "$CRATE"
  drift=1
fi
printf "\n"

if (( drift )); then
  printf "\nversions have drifted\n" >&2
  [[ "${CHECK:-0}" == "1" ]] && exit 1
fi
exit 0
