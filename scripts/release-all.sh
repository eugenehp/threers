#!/usr/bin/env bash
# Build, pack, or publish threers artifacts across registries.
#
# Usage:
#   ./scripts/release-all.sh              # build everything (default)
#   ./scripts/release-all.sh verify       # the pre-release checks (see below)
#   ./scripts/release-all.sh pack         # build + npm pack + local wheels
#   ./scripts/release-all.sh publish      # upload (needs PUBLISH=1)
#
# `verify` runs the local half of docs/releasing.md: versions, the workspace
# tests (all features, then the physics defaults), clippy native and wasm,
# every feature as a library on its own, the MSRV, docs, and `cargo package`
# of every unpublished crate. It runs each step even after one fails and
# prints a table; non-zero exit if any failed. Hardware decoders, conformance
# vectors and the browser are separate (FULL=1 adds the WebGL2 browser test
# and scripts/browser-smoke.sh). It builds in $CARGO_TARGET_DIR, so point that
# somewhere no other cargo run is using.
#
# Targets (env TARGETS, comma-separated, or default "all"):
#   language  — npm ESM (mini+full), npm native (host), PyPI wheel (host)
#   crates    — crates.io: every workspace crate not marked `publish = false`
#               (today threers-math, -captions, -audio, -codec, -softbody, threers,
#               -physics, -animation, -probe and -video); versions
#               already on the index are skipped, so a partial publish can be rerun
#   npm-esm   — @threers/threers on npm (wasm mini + full)
#   npm-node  — threers-node + @threers/node-* (host .node only locally)
#   pypi      — threers PyPI wheel (host triple)
#   all       — crates + language (the default)
#
# Publishing is safe to repeat: anything whose version is already on its
# registry is skipped, before it is built, and the run ends with a table of
# what was published, skipped and left pending. threers-node is pending until
# CI has published every @threers/node-* package (push the v* tag).
#
# Examples:
#   ./scripts/release-all.sh pack
#   TARGETS=npm-esm,npm-node ./scripts/release-all.sh build
#   PUBLISH=1 ./scripts/release-all.sh publish
#   ./scripts/release-all.sh publish --dry-run
#
# Multi-arch npm-node + PyPI wheels: push a v* tag (CI matrix) or use
#   gh workflow run release-language-packages.yml -f publish=true
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-build}"
shift || true
# Flags alone (`--help`, `--dry-run`) mean the default action.
if [[ "$ACTION" == -* ]]; then
  set -- "$ACTION" "$@"
  ACTION=build
fi

DRY_RUN="${DRY_RUN:-0}"
PUBLISH="${PUBLISH:-0}"
TARGETS="${TARGETS:-all}"
PROFILE="${PROFILE:-release}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) PUBLISH=1 ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

if [[ "$ACTION" == "publish" && "$PUBLISH" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Refusing to publish without PUBLISH=1 or --yes (or use --dry-run)." >&2
  echo "  PUBLISH=1 ./scripts/release-all.sh publish" >&2
  exit 1
fi

# Whole names only: "npm-node" must not also select a target called "node".
has_target() {
  [[ ",$TARGETS," == *",$1,"* || ",$TARGETS," == *",all,"* ]]
}

log() { echo "==> $*"; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
  else
    echo "+ $*"
    "$@"
  fi
}

version_from_cargo() {
  python3 -c "import tomllib, pathlib; p=tomllib.loads(pathlib.Path('$ROOT/Cargo.toml').read_text()); print(p['workspace']['package']['version'])"
}

VERSION="$(version_from_cargo)"
log "threers release helper — action=$ACTION targets=$TARGETS version=$VERSION profile=$PROFILE"

# What a publish did, for the table at the end: "state  what".
SUMMARY=()
note() {
  local state="$1"
  [[ "$DRY_RUN" == "1" && "$state" == published ]] && state="would-pub"
  SUMMARY+=("$(printf '%-10s %s' "$state" "$2")")
}

# Publish the npm package in directory $1 unless it is already up, and note
# which of the two happened (scripts/npm-publish-once.sh decides).
npm_publish() {
  local dir="$1" label="$2" status_file
  status_file=$(mktemp)
  (cd "$dir" && NPM_ONCE_STATUS="$status_file" run "$ROOT/scripts/npm-publish-once.sh")
  if [[ "$(cat "$status_file")" == skipped ]]; then
    note skipped "npm $label $VERSION (already up)"
  else
    note published "npm $label $VERSION"
  fi
  rm -f "$status_file"
}

# Is name@VERSION on npm? Asks the registry with your login, which sees a
# package minutes before the public CDN does.
npm_has() {
  [[ "$(npm view "$1@$VERSION" version 2>/dev/null)" == "$VERSION" ]]
}

# ---------------------------------------------------------------------------
# crates.io
# ---------------------------------------------------------------------------
# Every workspace member cargo would publish (`publish` not false), with its
# version and whether that version is already on crates.io: "name version state",
# state one of published / unpublished / unknown (index unreachable).
crate_publish_state() {
  cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c '
import json, sys, urllib.request, urllib.error
for p in json.load(sys.stdin)["packages"]:
    if p.get("publish") == []:
        continue
    name, ver = p["name"], p["version"]
    req = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{name}/{ver}",
        headers={"User-Agent": "threers-release"})
    try:
        urllib.request.urlopen(req, timeout=20)
        state = "published"
    except urllib.error.HTTPError as e:
        state = "unpublished" if e.code == 404 else "unknown"
    except Exception:
        state = "unknown"
    print(name, ver, state)
'
}

# Packaging several crates at once unpacks them into a local registry under
# ~/.cargo/registry/src/-<hash>/, and a later run at the same version reuses
# what it finds there: after a code change without a bump, the tarballs are
# then built against the old sources and pass. Drop our crates' copies first.
clear_stale_local_registry() {
  local reg="${CARGO_HOME:-$HOME/.cargo}/registry/src" d
  for d in "$reg"/-*/threers*-"$VERSION"; do
    [[ -d "$d" ]] || continue
    run rm -rf "$d"
  done
}

release_crates() {
  # The set comes from the workspace, not a list here, so a new publishable
  # crate is picked up without editing this script. Given several `-p`, cargo
  # orders them by dependency and waits for each to reach the index before
  # packaging the next. Versions already on crates.io are left out, so a publish
  # that stopped halfway can simply be run again. (`--workspace --exclude` would
  # be shorter, but `cargo package --workspace` does not skip `publish = false`.)
  log "crates.io: publishable workspace crates"
  local pkgs=() name ver state
  while read -r name ver state; do
    printf "    %-22s %-8s %s\n" "$name" "$ver" "$state"
    if [[ "$state" == "published" ]]; then
      [[ "$ACTION" == "publish" ]] && note skipped "crates.io $name $ver (already up)"
    else
      pkgs+=(-p "$name")
    fi
  done < <(crate_publish_state)

  if (( ${#pkgs[@]} == 0 )); then
    log "every publishable crate is already on crates.io at its current version — nothing to do"
    return
  fi
  if [[ "$ACTION" == "pack" || "$ACTION" == "verify" ]]; then
    clear_stale_local_registry
    run cargo package "${pkgs[@]}" --allow-dirty
  elif [[ "$ACTION" == "publish" ]]; then
    # cargo orders these by dependency and waits for each to reach the index.
    # If one fails, rerunning picks up from it: the ones before are skipped.
    run cargo publish "${pkgs[@]}"
    local p
    for p in "${pkgs[@]}"; do
      [[ "$p" == -p ]] || note published "crates.io $p $VERSION"
    done
  fi
}

# ---------------------------------------------------------------------------
# npm ESM (wasm mini + full)
# ---------------------------------------------------------------------------
release_npm_esm() {
  log "npm ESM: @threers/threers (dist/mini + dist/full)"
  if [[ "$ACTION" == "publish" ]] && npm_has @threers/threers; then
    echo "    @threers/threers@$VERSION is already on npm; not rebuilding it"
    note skipped "npm @threers/threers $VERSION (already up)"
    return
  fi
  run bash "$ROOT/crates/threers-js/build.sh"
  if [[ "$ACTION" == "pack" || "$ACTION" == "publish" ]]; then
    (cd "$ROOT/crates/threers-js" && run npm pack)
  fi
  if [[ "$ACTION" == "publish" ]]; then
    npm_publish "$ROOT/crates/threers-js" @threers/threers
  fi
  if [[ -f "$ROOT/crates/threers-js/dist/mini/pkg/threers_bg.wasm" ]]; then
    du -h "$ROOT/crates/threers-js/dist/mini/pkg/threers_bg.wasm" \
          "$ROOT/crates/threers-js/dist/full/pkg/threers_bg.wasm" 2>/dev/null || true
  fi
}

# ---------------------------------------------------------------------------
# npm native (napi-rs) — host triple locally; full matrix in CI
# ---------------------------------------------------------------------------
release_npm_node() {
  log "npm native: threers-node (host triple; CI builds all arches on tag)"
  if [[ "$ACTION" == "publish" ]] && npm_has threers-node; then
    echo "    threers-node@$VERSION is already on npm, platform packages first; nothing to do"
    note skipped "npm threers-node and @threers/node-* $VERSION (already up)"
    return
  fi
  if [[ ! -d "$ROOT/crates/threers-node/node_modules" ]]; then
    (cd "$ROOT/crates/threers-node" && run npm install)
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    (cd "$ROOT/crates/threers-node" && npm run build)
  else
    echo "[dry-run] (cd crates/threers-node && npm run build)"
  fi

  # Copy host .node into the matching platform stub if present.
  local node_file=""
  for f in "$ROOT/crates/threers-node"/*.node; do
    [[ -f "$f" ]] || continue
    node_file="$f"
    break
  done
  if [[ -n "$node_file" ]]; then
    local base dest_dir
    base=$(basename "$node_file")
    case "$base" in
      *darwin-arm64*) dest_dir=npm/darwin-arm64 ;;
      *darwin-x64*) dest_dir=npm/darwin-x64 ;;
      *linux-x64-gnu*) dest_dir=npm/linux-x64-gnu ;;
      *linux-arm64-gnu*) dest_dir=npm/linux-arm64-gnu ;;
      *win32-x64-msvc*) dest_dir=npm/win32-x64-msvc ;;
      *) dest_dir="" ;;
    esac
    if [[ -n "$dest_dir" && -d "$ROOT/crates/threers-node/$dest_dir" ]]; then
      log "staging $base → $dest_dir/"
      run cp "$node_file" "$ROOT/crates/threers-node/$dest_dir/"
    fi
  fi

  if [[ "$ACTION" == "pack" || "$ACTION" == "publish" ]]; then
    (cd "$ROOT/crates/threers-node" && run npm pack)
    for d in "$ROOT/crates/threers-node/npm"/*; do
      [[ -d "$d" ]] || continue
      if compgen -G "$d/*.node" > /dev/null; then
        (cd "$d" && run npm pack)
      fi
    done
  fi

  if [[ "$ACTION" == "publish" ]]; then
    local missing=() name
    for d in "$ROOT/crates/threers-node/npm"/*; do
      [[ -d "$d" ]] || continue
      name=$(node -p "require('$d/package.json').name")
      if npm_has "$name"; then
        note skipped "npm $name $VERSION (already up)"
      elif compgen -G "$d/*.node" > /dev/null; then
        npm_publish "$d" "$name"
      else
        # Built on CI only. The parent waits for it: published without one,
        # it installs on that platform and then cannot load.
        echo "    $name: no .node here — CI builds it" >&2
        missing+=("$name")
      fi
    done
    if (( ${#missing[@]} )); then
      echo "    not publishing threers-node yet: ${missing[*]} @ $VERSION not on npm" >&2
      note pending "npm threers-node $VERSION — waits for ${#missing[@]} platform package(s); push the v$VERSION tag"
      return 0
    fi
    npm_publish "$ROOT/crates/threers-node" threers-node
  fi
}

# ---------------------------------------------------------------------------
# PyPI (maturin) — host wheel locally; full matrix in CI
# ---------------------------------------------------------------------------
# Does PyPI have a threers $VERSION wheel for this OS and CPU already?
pypi_has_host_wheel() {
  local json
  json=$(curl -fsS "https://pypi.org/pypi/threers/$VERSION/json" 2>/dev/null) || return 1
  python3 -c '
import json, platform, sys
osname = {"Darwin": "macosx", "Linux": "linux", "Windows": "win"}[platform.system()]
cpu = {"arm64": ("arm64", "aarch64"), "aarch64": ("aarch64", "arm64"),
       "x86_64": ("x86_64", "amd64"), "AMD64": ("amd64", "x86_64")}[platform.machine()]
files = [u["filename"].lower() for u in json.load(sys.stdin)["urls"]]
sys.exit(0 if any(f.endswith(".whl") and osname in f and any(c in f for c in cpu) for f in files) else 1)
' <<<"$json"
}

release_pypi() {
  log "PyPI: threers (maturin, host triple)"
  if ! command -v maturin >/dev/null 2>&1; then
    echo "maturin not found — pip install maturin" >&2
    exit 1
  fi
  # shellcheck disable=SC2054  # one cargo feature list, not three elements
  local flags=(--release --features headless,physics,animation -o dist)
  mkdir -p "$ROOT/crates/threers-py/dist"
  if [[ "$ACTION" == "publish" ]]; then
    if pypi_has_host_wheel; then
      echo "    threers $VERSION already has a wheel for this platform on PyPI; not rebuilding it"
      note skipped "PyPI threers $VERSION, $(uname -s) $(uname -m) wheel (already up)"
      return
    fi
    # --skip-existing too: a rerun after a partial upload is not an error.
    (cd "$ROOT/crates/threers-py" && run maturin publish --skip-existing --non-interactive "${flags[@]}")
    note published "PyPI threers $VERSION, $(uname -s) $(uname -m) wheel"
    note pending "PyPI wheels for the other platforms — CI builds them on the v$VERSION tag"
  else
    (cd "$ROOT/crates/threers-py" && run maturin build "${flags[@]}")
    ls -la "$ROOT/crates/threers-py/dist/" 2>/dev/null || true
  fi
}

# ---------------------------------------------------------------------------
# verify — the pre-release checks of docs/releasing.md that need no hardware
# ---------------------------------------------------------------------------
VERIFY_RESULTS=()

# step "label" cmd...: run it, record ok/FAILED, carry on either way.
step() {
  local label="$1"; shift
  log "verify: $label"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
    VERIFY_RESULTS+=("skip    $label")
  elif "$@"; then
    VERIFY_RESULTS+=("ok      $label")
  elif [[ "${ADVISORY:-0}" == "1" ]]; then
    VERIFY_RESULTS+=("warn    $label")
  else
    VERIFY_RESULTS+=("FAILED  $label")
  fi
}

# `-p` for every workspace member that publishes, or with "unpublished", every
# one that does not.
crate_args() {
  cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c '
import json, sys
want_published = sys.argv[1] == "published"
for p in json.load(sys.stdin)["packages"]:
    if (p.get("publish") != []) == want_published:
        print("-p", p["name"])
' "$1"
}

# Each feature as a library on its own: `--all-features` and dev-dependencies
# both hide a feature that forgot to declare a dependency.
THREERS_FEATURES=(render audio captions softbody mesh-bvh raytrace bvh-csg openscad manifold
  planet native-decode native-codec metal videotoolbox parallel async rlx rlx-geo nurbs brep
  brep-csg brep-kernel step assembly-check learned-denoise)
PHYSICS_FEATURES=(parallel async gpu mechanism assembly openscad)

features_alone() {
  local pkg="$1" f; shift
  for f in "$@"; do
    echo "  $pkg --features $f"
    cargo build -q -p "$pkg" --no-default-features --features "$f" --lib || return 1
  done
}

msrv_check() {
  local msrv c
  msrv=$(python3 -c "
import tomllib
d = tomllib.load(open('$ROOT/Cargo.toml', 'rb'))
print(d.get('package', {}).get('rust-version') or d['workspace']['package']['rust-version'])")
  [[ -n "$msrv" ]] || { echo "no rust-version in Cargo.toml" >&2; return 1; }
  rustup run "$msrv" rustc --version >/dev/null 2>&1 \
    || rustup toolchain install "$msrv" --profile minimal || return 1
  for c in $(publishable_crates); do
    echo "  $c on $msrv"
    cargo "+$msrv" check -q -p "$c" --all-targets --all-features || return 1
  done
}

publishable_crates() {
  cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
    | python3 -c 'import json,sys; print(" ".join(p["name"] for p in json.load(sys.stdin)["packages"] if p.get("publish") != []))'
}

docs_check() {
  # Docs warn rather than fail on a broken intra-doc link; make them fail.
  RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --all-features -p threers -p threers-physics -p threers-probe \
    && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --all-features -p threers-codec \
    && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps -p threers-codec --features vaapi,nvidia,gif,parallel \
         --target x86_64-unknown-linux-gnu \
    && cargo clean -q -p threers   # the narrower doc build leaves an rlib the tests must not link
}

release_verify() {
  step "versions agree"           env CHECK=1 "$ROOT/scripts/release-versions.sh"
  # The crates that ship gate the release. The rest (demos, benches, crates
  # held back) still run, and report `warn` rather than failing it.
  local published unpublished
  read -ra published <<<"$(crate_args published | tr '\n' ' ')"
  read -ra unpublished <<<"$(crate_args unpublished | tr '\n' ' ')"
  step "tests, published crates"  cargo test "${published[@]}" --all-features --no-fail-fast
  ADVISORY=1 step "tests, unpublished crates (advisory)" \
                                  cargo test "${unpublished[@]}" --all-features --no-fail-fast
  step "physics, default features" cargo test -p threers-physics --no-fail-fast
  step "clippy, all features"     cargo clippy --workspace --all-targets --all-features -- -D warnings
  step "clippy, default features" cargo clippy --workspace --all-targets -- -D warnings
  step "threers features alone"   features_alone threers "${THREERS_FEATURES[@]}"
  step "physics features alone"   features_alone threers-physics "${PHYSICS_FEATURES[@]}"
  step "clippy wasm"              cargo clippy --target wasm32-unknown-unknown --lib -- -D warnings
  step "clippy wasm openscad"     cargo clippy --target wasm32-unknown-unknown --lib --features openscad -- -D warnings
  step "clippy wasm webgl"        cargo clippy --target wasm32-unknown-unknown --lib --features webgl -- -D warnings
  step "codec, no default features" cargo test -p threers-codec --release --no-default-features
  step "MSRV"                     msrv_check
  step "docs"                     docs_check
  step "cargo package"            release_crates
  if [[ "${FULL:-0}" == "1" ]]; then
    step "WebGL2 in Chrome"       env FEATURES=webgl TEST=webgl "$ROOT/scripts/wasm-browser-test.sh"
    step "browser smoke"          "$ROOT/scripts/browser-smoke.sh"
  fi

  echo ""
  log "verify summary (v$VERSION)"
  local r failed=0
  for r in "${VERIFY_RESULTS[@]}"; do
    echo "  $r"
    [[ "$r" == FAILED* ]] && failed=1
  done
  echo ""
  echo "Not covered here: hardware decoders, conformance vectors (scripts/test-no-skips.py),"
  echo "and the GPU parity suites — see docs/releasing.md steps 2c, 3d and 3e."
  return "$failed"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$ACTION" in
  verify)
    release_verify
    exit $?
    ;;
  build|pack|publish) ;;
  *)
    echo "unknown action: $ACTION (use build, verify, pack, or publish)" >&2
    exit 1
    ;;
esac

# A publish that goes out with one root still on the previous version is not
# fixable afterwards — npm and crates.io both refuse to replace a version. Check
# before anything is uploaded.
if [[ "$ACTION" == "publish" ]]; then
  log "checking versions across publish roots"
  crates_targeted=0
  has_target crates && crates_targeted=1
  CHECK=1 CHECK_CRATES=$crates_targeted "$(dirname "$0")/release-versions.sh"
fi

if has_target crates || [[ "$TARGETS" == "all" ]]; then
  release_crates
fi

if has_target language || has_target npm-esm || [[ "$TARGETS" == "all" ]]; then
  release_npm_esm
fi

if has_target language || has_target npm-node || [[ "$TARGETS" == "all" ]]; then
  release_npm_node
fi

if has_target language || has_target pypi || [[ "$TARGETS" == "all" ]]; then
  release_pypi
fi

if [[ "$ACTION" == "publish" ]] && (( ${#SUMMARY[@]} )); then
  echo ""
  log "publish summary (v$VERSION)"
  for line in "${SUMMARY[@]}"; do echo "  $line"; done
  echo ""
fi

log "done ($ACTION). Artifacts:"
echo "  crates/threers-js/dist/{mini,full}/   — npm ESM (@threers/threers)"
echo "  crates/threers-node/*.tgz             — npm native (threers-node)"
echo "  crates/threers-py/dist/*.whl          — PyPI"
if [[ "$ACTION" != "publish" ]]; then
  echo ""
  echo "Publish later (bump first if v$VERSION has already shipped: ./scripts/release-versions.sh bump X.Y.Z):"
  echo "  PUBLISH=1 ./scripts/release-all.sh publish"
  echo "  git tag v$VERSION && git push --tags   # after publishing; CI: .github/workflows/release-language-packages.yml"
fi
