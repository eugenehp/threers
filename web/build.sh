#!/usr/bin/env bash
set -euo pipefail

# Build the threers wasm package and emit JS glue into web/pkg.
# Requires: rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli
# (wasm-pack is not used — this drives cargo and wasm-bindgen directly.)
#
# Variants (browser download sizes):
#   VARIANT=mini  — core renderer only (`--no-default-features`); smallest wasm
#   VARIANT=full  — `--features wasm-full` (openscad, codecs, nurbs, planet, raytrace)
#   (unset)       — opt-in via MESH_BVH / BVH_CSG / OPENSCAD / NATIVE_CODEC / … env flags
#
# Output always lands in web/pkg/ (and regenerated features.js). The npm package
# stages mini + full under crates/threers-js/dist/{mini,full}/.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

PROFILE="${PROFILE:-release}"
PROFILE_FLAG=""
TARGET_DIR="target/wasm32-unknown-unknown/debug"
if [ "$PROFILE" = "release" ]; then
    PROFILE_FLAG="--release"
    TARGET_DIR="target/wasm32-unknown-unknown/release"
fi

VARIANT="${VARIANT:-}"
NO_DEFAULT_FEATURES=""
FEATURE_LIST=()

if [ "$VARIANT" = "mini" ]; then
    echo "==> Variant: mini (core renderer, --no-default-features)"
    NO_DEFAULT_FEATURES="--no-default-features"
    # Clear additive feature envs so a leftover OPENSCAD=1 cannot inflate mini.
    unset MESH_BVH BVH_CSG NATIVE_CODEC OPENSCAD NURBS BREP BREP_CSG BREP_KERNEL STEP RLX RLX_GEO || true
elif [ "$VARIANT" = "full" ]; then
    echo "==> Variant: full (Cargo feature wasm-full)"
    FEATURE_LIST+=("wasm-full")
    # Mirror flags for generate-features-addon.mjs / post-build.
    export OPENSCAD=1
    export NATIVE_CODEC=1
    export NURBS=1
    export BVH_CSG=1
    export MESH_BVH=1
elif [ -n "$VARIANT" ]; then
    echo "ERROR: unknown VARIANT='$VARIANT' (use mini, full, or leave unset)"
    exit 1
fi

echo "==> Building threers ($PROFILE) for wasm32-unknown-unknown..."
if [ -z "$VARIANT" ]; then
    if [ "${BVH_CSG:-}" = "1" ]; then
        FEATURE_LIST+=("bvh-csg")
        echo "    (bvh-csg feature enabled — includes mesh-bvh)"
    elif [ "${MESH_BVH:-}" = "1" ]; then
        FEATURE_LIST+=("mesh-bvh")
        echo "    (mesh-bvh feature enabled)"
    fi
    if [ "${NATIVE_CODEC:-}" = "1" ]; then
        FEATURE_LIST+=("native-codec")
        echo "    (native-codec feature enabled — GIF/APNG/WebM/MP4 browser export)"
    fi
    if [ "${OPENSCAD:-}" = "1" ]; then
        FEATURE_LIST+=("openscad")
        echo "    (openscad feature enabled — scad_geometry/scadExport, pulls bvh-csg)"
    fi
    # B-rep chain — collapse implications so only the outermost flag is passed
    # (cargo resolves the rest). See docs/brep-nurbs-plan.md.
    if [ "${STEP:-}" = "1" ]; then
        FEATURE_LIST+=("step")
        echo "    (step feature enabled — pulls brep-kernel → brep-csg → brep → nurbs)"
    elif [ "${BREP_KERNEL:-}" = "1" ]; then
        FEATURE_LIST+=("brep-kernel")
        echo "    (brep-kernel feature enabled — pulls brep-csg → brep → nurbs)"
    elif [ "${BREP_CSG:-}" = "1" ]; then
        FEATURE_LIST+=("brep-csg")
        echo "    (brep-csg feature enabled — pulls brep → nurbs, and openscad)"
    elif [ "${BREP:-}" = "1" ]; then
        FEATURE_LIST+=("brep")
        echo "    (brep feature enabled — pulls nurbs)"
    elif [ "${NURBS:-}" = "1" ]; then
        FEATURE_LIST+=("nurbs")
        echo "    (nurbs feature enabled — NurbsCurve/NurbsSurface/NurbsGeometry)"
    fi
    if [ "${RLX:-}" = "1" ]; then
        FEATURE_LIST+=("rlx")
        echo "    (rlx feature enabled — rlxConvolve/rlxSmoothMesh/rlxFitGrade/rlxPalette)"
    fi
    if [ "${RLX_GEO:-}" = "1" ]; then
        FEATURE_LIST+=("rlx-geo")
        echo "    (rlx-geo feature enabled — geoDelaunay/geoHeightfield/geoVoronoiLabels)"
    fi
fi

FEATURES=""
if [ "${#FEATURE_LIST[@]}" -gt 0 ]; then
    FEATURES="--features $(IFS=,; echo "${FEATURE_LIST[*]}")"
fi

# shellcheck disable=SC2086
cargo build --target wasm32-unknown-unknown --lib $PROFILE_FLAG $NO_DEFAULT_FEATURES $FEATURES

echo "==> Generating JS bindings into web/pkg..."
wasm-bindgen \
    "$TARGET_DIR/threers.wasm" \
    --target web \
    --out-dir web/pkg \
    --no-typescript

# Optional size pass (install: cargo install wasm-opt / binaryen).
if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" = "release" ]; then
    echo "==> wasm-opt -Os…"
    wasm-opt -Os -o web/pkg/threers_bg.wasm web/pkg/threers_bg.wasm
fi

echo "==> Web JS deps (optional three peer for typecheck)..."
if [ -f "$WEB/package.json" ]; then
    (cd "$WEB" && npm install --no-audit --no-fund --silent) || true
fi

echo "==> Patching WebGPU device limits for browser compatibility..."
MESH_BVH="${MESH_BVH:-}" BVH_CSG="${BVH_CSG:-}" NURBS="${NURBS:-}" \
  "$WEB/post-build.sh"

# Record which variant this pkg/ is so staging scripts can assert.
echo "${VARIANT:-custom}" > web/pkg/variant.txt
echo "==> Done (variant=${VARIANT:-custom}). Open with a static server under web/."
echo "    Browser must support WebGPU (Chrome 113+ or Firefox Nightly with flag)."
