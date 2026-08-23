#!/usr/bin/env bash
# Build, pack, or publish threers artifacts across registries.
#
# Usage:
#   ./scripts/release-all.sh              # build everything (default)
#   ./scripts/release-all.sh pack         # build + npm pack + local wheels
#   ./scripts/release-all.sh publish      # upload (needs PUBLISH=1)
#
# Targets (env TARGETS, comma-separated, or default "language"):
#   language  — npm ESM (mini+full), npm native (host), PyPI wheel (host)
#   crates    — crates.io: threers, threers-physics, threers-probe
#   npm-esm   — threers npm (wasm mini + full)
#   npm-node  — threers-node + @threers/node-* (host .node only locally)
#   pypi      — threers PyPI wheel (host triple)
#   all       — crates + language
#
# Examples:
#   ./scripts/release-all.sh pack
#   TARGETS=npm-esm,npm-node ./scripts/release-all.sh build
#   PUBLISH=1 ./scripts/release-all.sh publish
#   ./scripts/release-all.sh publish --dry-run
#
# Multi-arch npm-node + PyPI wheels: push a v* tag (CI matrix) or use
#   gh workflow run release-language-packages.yml
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-build}"
shift || true

DRY_RUN="${DRY_RUN:-0}"
PUBLISH="${PUBLISH:-0}"
TARGETS="${TARGETS:-language}"
PROFILE="${PROFILE:-release}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) PUBLISH=1 ;;
    -h|--help)
      sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
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

has_target() {
  local t="$1"
  [[ "$TARGETS" == *"$t"* ]] || [[ "$TARGETS" == "all" ]]
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
  python3 -c "import tomllib, pathlib; p=tomllib.loads(pathlib.Path('$ROOT/Cargo.toml').read_text()); print(p['package']['version'])"
}

VERSION="$(version_from_cargo)"
log "threers release helper — action=$ACTION targets=$TARGETS version=$VERSION profile=$PROFILE"

# ---------------------------------------------------------------------------
# crates.io
# ---------------------------------------------------------------------------
release_crates() {
  log "crates.io (order: threers → threers-physics → threers-probe)"
  for pkg in threers threers-physics threers-probe; do
    if [[ "$ACTION" == "pack" || "$ACTION" == "publish" ]]; then
      log "packaging $pkg..."
      run cargo package -p "$pkg" --allow-dirty
    fi
    if [[ "$ACTION" == "publish" ]]; then
      log "publishing $pkg to crates.io..."
      run cargo publish -p "$pkg"
    fi
  done
}

# ---------------------------------------------------------------------------
# npm ESM (wasm mini + full)
# ---------------------------------------------------------------------------
release_npm_esm() {
  log "npm ESM: threers (dist/mini + dist/full)"
  run bash "$ROOT/crates/threers-js/build.sh"
  if [[ "$ACTION" == "pack" || "$ACTION" == "publish" ]]; then
    (cd "$ROOT/crates/threers-js" && run npm pack)
  fi
  if [[ "$ACTION" == "publish" ]]; then
    (cd "$ROOT/crates/threers-js" && run npm publish --access public)
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
    for d in "$ROOT/crates/threers-node/npm"/*; do
      [[ -d "$d" ]] || continue
      if compgen -G "$d/*.node" > /dev/null; then
        (cd "$d" && run npm publish --access public)
      else
        echo "skip npm publish $(basename "$d") (no .node — build on CI or cross-compile)" >&2
      fi
    done
    (cd "$ROOT/crates/threers-node" && run npm publish --access public)
  fi
}

# ---------------------------------------------------------------------------
# PyPI (maturin) — host wheel locally; full matrix in CI
# ---------------------------------------------------------------------------
release_pypi() {
  log "PyPI: threers (maturin, host triple)"
  if ! command -v maturin >/dev/null 2>&1; then
    echo "maturin not found — pip install maturin" >&2
    exit 1
  fi
  local flags=(--release --features headless,physics,animation -o dist)
  mkdir -p "$ROOT/crates/threers-py/dist"
  if [[ "$ACTION" == "publish" ]]; then
    (cd "$ROOT/crates/threers-py" && run maturin publish "${flags[@]}")
  else
    (cd "$ROOT/crates/threers-py" && run maturin build "${flags[@]}")
    ls -la "$ROOT/crates/threers-py/dist/" 2>/dev/null || true
  fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$ACTION" in
  build|pack|publish) ;;
  *)
    echo "unknown action: $ACTION (use build, pack, or publish)" >&2
    exit 1
    ;;
esac

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

log "done ($ACTION). Artifacts:"
echo "  crates/threers-js/dist/{mini,full}/   — npm ESM (threers)"
echo "  crates/threers-node/*.tgz             — npm native (threers-node)"
echo "  crates/threers-py/dist/*.whl          — PyPI"
if [[ "$ACTION" != "publish" ]]; then
  echo ""
  echo "Publish later:"
  echo "  PUBLISH=1 ./scripts/release-all.sh publish"
  echo "  git tag v$VERSION && git push --tags   # CI: .github/workflows/release-language-packages.yml"
fi
