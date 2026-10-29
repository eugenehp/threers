#!/usr/bin/env bash
# Print versions across publish roots — bump these together before a release.
#
# `CHECK=1 ./scripts/release-versions.sh` exits non-zero when any root that
# tracks the crate version has drifted from it. The separately versioned
# crates (threers-animation, threers-probe) are reported but not checked.
# With CHECK=1 it also asks crates.io, and fails if threers or threers-physics
# is not ahead of what is already on the index.
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
report "crates.io threers-physics"   "$(v_cargo "$ROOT/crates/threers-physics/Cargo.toml")"
report "crates.io threers-codec"     "$(v_cargo "$ROOT/crates/threers-codec/Cargo.toml")"
report "crates.io threers-softbody"  "$(v_cargo "$ROOT/crates/threers-softbody/Cargo.toml")"
report "web shim (package.json)"     "$(v_json "$ROOT/web/package.json")"
report "web shim (lock)"             "$(v_json "$ROOT/web/package-lock.json")"

for f in "$ROOT/crates/threers-node/npm"/*/package.json; do
  [[ -f "$f" ]] || continue
  pkg=$(python3 -c "import json; print(json.load(open('$f'))['name'])")
  report "$pkg" "$(v_json "$f")"
done

# threers-physics tracks the crate version (checked above). The bindings pin it
# by version as well as by path, and cargo refuses to resolve a path dependency
# whose version does not satisfy the requirement. Check that pairing.
PHYSICS=$(v_cargo "$ROOT/crates/threers-physics/Cargo.toml")
PROBE=$(v_cargo "$ROOT/crates/threers-probe/Cargo.toml")
ANIM=$(v_cargo "$ROOT/crates/threers-animation/Cargo.toml")
printf "\nthreers-physics pins:\n"
for f in "$ROOT/crates/threers-py/Cargo.toml" "$ROOT/crates/threers-node/Cargo.toml" \
         "$ROOT/crates/threers-animation/Cargo.toml"; do
  pinned=$(sed -n 's/^threers-physics = .*version = "\([^"]*\)".*/\1/p' "$f")
  [[ -n "$pinned" ]] || continue
  printf "%-32s %s" "  ${f#"$ROOT/crates/"}" "$pinned"
  if [[ "$pinned" != "$PHYSICS" ]]; then
    printf "   <- expected %s" "$PHYSICS"
    drift=1
  fi
  printf "\n"
done

printf "\nindependently versioned:\n"
printf "%-32s %s\n" "crates.io threers-animation" "$ANIM"
printf "%-32s %s\n" "crates.io threers-probe"    "$PROBE"


# `cargo publish` cannot replace a version, so a version that is already on the
# index fails at the upload and not before. Under CHECK=1, ask crates.io.
if [[ "${CHECK:-0}" == "1" ]]; then
  printf "\ncrates.io index:\n"
  for pair in "threers:$CRATE" "threers-physics:$PHYSICS"; do
    name=${pair%%:*} local_v=${pair#*:}
    published=$(curl -sf -H 'User-Agent: threers-release' \
        "https://crates.io/api/v1/crates/$name" 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['crate']['max_version'])" \
        2>/dev/null) || published=""
    printf "%-32s %s" "  $name published" "${published:-unknown}"
    if [[ -z "$published" ]]; then
      printf "   (could not reach crates.io)"
    elif ! python3 -c "import sys; t=lambda v: tuple(map(int, v.split('.'))); sys.exit(t(sys.argv[1]) <= t(sys.argv[2]))" \
        "$local_v" "$published"; then
      printf "   <- %s is not ahead of it" "$local_v"
      drift=1
    fi
    printf "\n"
  done
fi

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
