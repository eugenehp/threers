#!/usr/bin/env bash
# M3 — OpenSCAD/CGAL oracle harness.
#
# For each corpus model, generate a reference STL with the real OpenSCAD (CGAL
# F6 render) and our kernel's STL, then compare with the four-metric gate
# (watertight · volume · Euler · Hausdorff) via the `openscad_compare` example.
#
# The shipped crate stays pure Rust: OpenSCAD is used only here, as a dev-time
# test oracle. If `openscad` is not installed, the harness runs a self-consistency
# smoke test (our STL vs itself) so the plumbing is still exercised in CI.
#
# Usage: scripts/ci-openscad.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="$ROOT/tests/openscad-corpus"
OUT="$(mktemp -d)"
FEAT="--features openscad"
VOL_TOL="${VOL_TOL:-0.02}"
HAUS_TOL="${HAUS_TOL:-0.05}"

# Models the exact kernel resolves to watertight output — these gate the build.
# `plate_2d` is the round-hole-in-plate built via the 2D subsystem: the
# hole-bridging cap triangulation yields a closed manifold directly (no CSG),
# so it reaches CGAL parity via the 2D subsystem. `plate_with_hole` reaches the
# same volume via the exact 3D-difference kernel: the per-face CDT + Delaunay
# arrangement now makes curved booleans (a cylindrical through-hole) watertight.
MUST_PASS=(stacked_boxes box_notch box_slot plate_2d resize_box rounded_2d proj_cut mink_box booleans_2d sheared default_cyl mink2d surf_dat plate_with_hole)
# Known limitations: the exact kernel still falls back to (non-manifold) float
# here, so they're reported but don't fail CI. Move them up as the kernel grows.
KNOWN_LIMITATION=()

# Locate an OpenSCAD (CGAL) renderer: a local binary, else the Docker image
# (pulled if needed). Falls back to a self-consistency smoke test if neither.
OSCAD_IMAGE="${OSCAD_IMAGE:-openscad/openscad}"
OSCAD=none
if command -v openscad >/dev/null 2>&1; then
  OSCAD=local
elif command -v docker >/dev/null 2>&1; then
  if docker image inspect "$OSCAD_IMAGE" >/dev/null 2>&1 || docker pull "$OSCAD_IMAGE" >/dev/null 2>&1; then
    OSCAD=docker
  fi
fi
echo "openscad: $OSCAD"

echo "building examples..."
cargo build $FEAT --example openscad_corpus --example openscad_compare >/dev/null 2>&1

render_ref() {  # $1 = model → writes $OUT/$1.ref.stl (CGAL F6 render)
  local m="$1"
  case "$OSCAD" in
    local)  openscad -o "$OUT/$m.ref.stl" "$CORPUS/$m.scad" >/dev/null 2>&1 ;;
    docker) cp "$CORPUS"/* "$OUT/" 2>/dev/null || true  # .scad + any aux (.dat, include libs)
            docker run --rm -v "$OUT:/work" "$OSCAD_IMAGE" \
              openscad -o "/work/$m.ref.stl" "/work/$m.scad" >/dev/null 2>&1 ;;
  esac
}

compare_model() {  # $1 = model name  → returns compare's exit code
  local m="$1" ours="$OUT/$1.ours.stl" ref
  cargo run -q $FEAT --example openscad_corpus -- "$m" "$ours" >/dev/null
  if [ "$OSCAD" != none ]; then
    render_ref "$m"
    ref="$OUT/$m.ref.stl"
    echo -n "[$m] vs OpenSCAD/CGAL:  "
  else
    ref="$ours"  # smoke test: exercises the plumbing only
    echo -n "[$m] self-consistency (no openscad):  "
  fi
  cargo run -q $FEAT --example openscad_compare -- "$ours" "$ref" "$VOL_TOL" "$HAUS_TOL"
}

fail=0
for m in "${MUST_PASS[@]}"; do
  compare_model "$m" || fail=1
done
for m in ${KNOWN_LIMITATION[@]+"${KNOWN_LIMITATION[@]}"}; do
  compare_model "$m" || echo "  (known limitation — not gating)"
done

rm -rf "$OUT"
[ "$fail" = 0 ] && echo "ci-openscad: PASS" || { echo "ci-openscad: FAIL"; exit 1; }
