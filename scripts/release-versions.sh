#!/usr/bin/env bash
# Print versions across publish roots — bump these together before a release.
#
# Every crate, package and version pin in the repo carries one version, set in
# `[workspace.package]` in the root Cargo.toml.
# `CHECK=1 ./scripts/release-versions.sh` exits non-zero when any of them has
# drifted from it.
# With CHECK=1 it also asks crates.io, and fails if any published crate is
# behind the index, or if every one of them is already there at its version.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# A crate's version, following `version.workspace = true` to the root.
v_cargo() {
  python3 - "$1" "$ROOT/Cargo.toml" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
v = data["package"]["version"]
if isinstance(v, dict):
    v = tomllib.loads(pathlib.Path(sys.argv[2]).read_text())["workspace"]["package"]["version"]
print(v)
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

# Every crate in crates/, workspace member or not.
printf "\ncrates:\n"
for f in "$ROOT"/crates/*/Cargo.toml; do
  d=${f%/Cargo.toml}
  report "  ${d#"$ROOT/crates/"}" "$(v_cargo "$f")"
done

# The crates pin each other by version as well as by path, and cargo refuses to
# resolve a path dependency whose version does not satisfy the requirement.
printf "\nversion pins between crates:\n"
while read -r where pinned; do
  report "  $where" "$pinned"
done < <(python3 - "$ROOT" <<'PY'
import sys, tomllib, pathlib
root = pathlib.Path(sys.argv[1])
for f in [root / "Cargo.toml", *sorted(root.glob("crates/*/Cargo.toml"))]:
    data = tomllib.loads(f.read_text())
    tables = [data.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
    for t in data.get("target", {}).values():
        tables += [t.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
    for t in tables:
        for name, spec in t.items():
            if name.startswith("threers") and isinstance(spec, dict) and "version" in spec:
                print(f"{f.parent.name}->{name}", spec["version"])
PY
)

PHYSICS=$(v_cargo "$ROOT/crates/threers-physics/Cargo.toml")
PROBE=$(v_cargo "$ROOT/crates/threers-probe/Cargo.toml")
ANIM=$(v_cargo "$ROOT/crates/threers-animation/Cargo.toml")

# `cargo publish` cannot replace a version, so a version that is already on the
# index fails at the upload and not before. Under CHECK=1, ask crates.io about
# every crate the release publishes. A version behind the index is an error. One
# equal to it was already published and release-all.sh skips it, so a publish
# that stopped halfway can be rerun; but if all of them are equal, nothing would
# go out, which almost always means the bump was forgotten.
if [[ "${CHECK:-0}" == "1" ]]; then
  printf "\ncrates.io index:\n"
  pending=0 checked=0
  for pair in "threers:$CRATE" "threers-codec:$(v_cargo "$ROOT/crates/threers-codec/Cargo.toml")" \
              "threers-softbody:$(v_cargo "$ROOT/crates/threers-softbody/Cargo.toml")" \
              "threers-physics:$PHYSICS" "threers-animation:$ANIM" "threers-probe:$PROBE"; do
    name=${pair%%:*} local_v=${pair#*:}
    published=$(curl -sf -H 'User-Agent: threers-release' \
        "https://crates.io/api/v1/crates/$name" 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['crate']['max_version'])" \
        2>/dev/null) || published=""
    printf "%-32s %-8s local %s" "  $name published" "${published:-none}" "$local_v"
    if [[ -z "$published" ]]; then
      printf "   (not on crates.io, or could not reach it)"
      pending=1
    else
      checked=1
      cmp=$(python3 -c "import sys; t=lambda v: tuple(map(int, v.split('.'))); a,b=t(sys.argv[1]),t(sys.argv[2]); print((a>b)-(a<b))" \
          "$local_v" "$published")
      case "$cmp" in
        1)  pending=1 ;;
        0)  printf "   (already published, will be skipped)" ;;
        *)  printf "   <- %s is behind it" "$local_v"; drift=1 ;;
      esac
    fi
    printf "\n"
  done
  # Only fatal when this release publishes crates (release-all.sh sets
  # CHECK_CRATES); an npm- or PyPI-only release leaves the index alone.
  if (( checked && ! pending )); then
    printf "  every crate is already on crates.io at its local version — bump first\n"
    [[ "${CHECK_CRATES:-1}" == "1" ]] && drift=1
  fi
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
