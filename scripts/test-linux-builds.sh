#!/usr/bin/env bash
# Run the Linux builds of the native packages on this machine, in Docker.
#
# The addons and wheels for Linux are cross-compiled (scripts/release-all.sh,
# PLATFORMS=all), so nothing here has run them. This loads each one inside a
# Linux container of its own architecture — arm64 natively on Apple silicon,
# x64 under emulation — and renders a frame through Mesa's software Vulkan,
# checking the pixels. Windows builds cannot be run this way.
#
# Usage: scripts/test-linux-builds.sh [arm64|amd64 ...]   (default: both)
#   ADDONS=dir  where threers-node.linux-*.node are  (default crates/threers-node)
#   WHEELS=dir  where the threers *.whl files are     (default crates/threers-py/dist)
# Needs a running Docker (Docker Desktop on macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE="$ROOT/scripts/platform-smoke"
ADDONS="${ADDONS:-$ROOT/crates/threers-node}"
WHEELS="${WHEELS:-$ROOT/crates/threers-py/dist}"
VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('$ROOT/Cargo.toml','rb'))['workspace']['package']['version'])")

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running (on macOS: open -a Docker)" >&2
  exit 1
fi

(( $# )) || set -- arm64 amd64
failed=0
ran=0
for arch in "$@"; do
  case "$arch" in
    arm64) node_plat=linux-arm64-gnu; wheel_cpu=aarch64 ;;
    amd64) node_plat=linux-x64-gnu; wheel_cpu=x86_64 ;;
    *) echo "unknown architecture: $arch (arm64 or amd64)" >&2; exit 1 ;;
  esac
  echo "==> linux/$arch"
  docker build -q --platform "linux/$arch" -t "threers-smoke:$arch" "$SMOKE" >/dev/null \
    || { echo "    could not build the linux/$arch image" >&2; failed=1; continue; }

  addon="$ADDONS/threers-node.$node_plat.node"
  wheel=$(find "$WHEELS" -maxdepth 1 -name "threers-$VERSION-*linux*_$wheel_cpu.whl" 2>/dev/null | head -1 || true)
  cmds=()
  mounts=(-v "$SMOKE:/smoke:ro")
  if [[ -f "$addon" ]]; then
    mounts+=(-v "$addon:/work/$(basename "$addon"):ro")
    cmds+=("node /smoke/smoke.cjs /work/$(basename "$addon")")
  else
    echo "    no $(basename "$addon"); skipping the Node addon"
  fi
  if [[ -n "$wheel" ]]; then
    mounts+=(-v "$wheel:/work/$(basename "$wheel"):ro")
    cmds+=("pip install -q --break-system-packages --root-user-action=ignore /work/$(basename "$wheel") && cd /tmp && python3 /smoke/smoke.py")
  else
    echo "    no threers $VERSION wheel for $wheel_cpu in $WHEELS; skipping Python"
  fi
  (( ${#cmds[@]} )) || continue

  script=$(printf '%s && ' "${cmds[@]}")
  if docker run --rm --platform "linux/$arch" "${mounts[@]}" "threers-smoke:$arch" sh -c "${script% && }"; then
    ran=$((ran + 1))
  else
    echo "    linux/$arch FAILED" >&2
    failed=1
  fi
done

if (( ran == 0 && failed == 0 )); then
  echo "nothing to test: build the Linux addons and wheels first (PLATFORMS=all)" >&2
  exit 1
fi
exit "$failed"
