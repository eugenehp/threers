#!/usr/bin/env bash
# Print versions across publish roots — bump these together before a release.
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

printf "%-28s %s\n" "crates.io threers" "$(v_cargo "$ROOT/Cargo.toml")"
printf "%-28s %s\n" "crates.io threers-physics" "$(v_cargo "$ROOT/crates/threers-physics/Cargo.toml")"
printf "%-28s %s\n" "crates.io threers-probe" "$(v_cargo "$ROOT/crates/threers-probe/Cargo.toml")"
printf "%-28s %s\n" "npm threers (ESM)" "$(v_json "$ROOT/crates/threers-js/package.json")"
printf "%-28s %s\n" "npm threers-node" "$(v_json "$ROOT/crates/threers-node/package.json")"
printf "%-28s %s\n" "PyPI threers" "$(v_pyproject "$ROOT/crates/threers-py/pyproject.toml")"

for f in "$ROOT/crates/threers-node/npm"/*/package.json; do
  [[ -f "$f" ]] || continue
  pkg=$(python3 -c "import json; print(json.load(open('$f'))['name'])")
  ver=$(python3 -c "import json; print(json.load(open('$f'))['version'])")
  printf "%-28s %s\n" "$pkg" "$ver"
done
