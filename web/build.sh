#!/usr/bin/env bash
set -euo pipefail

# Build the threers wasm package and emit JS glue into web/pkg.
# Requires: rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli
# at the exact wasm-bindgen version in Cargo.lock (checked below).
# (wasm-pack is not used — this drives cargo and wasm-bindgen directly.)
#
# Variants (browser download sizes):
#   VARIANT=mini  — core renderer only (`--no-default-features --features render,audio`); smallest wasm
#   VARIANT=full  — `--features wasm-full` (openscad, codecs, nurbs, planet, raytrace, fluid)
#   (unset)       — opt-in via MESH_BVH / BVH_CSG / OPENSCAD / NATIVE_CODEC / NATIVE_DECODE / FLUID / … flags
#
# Tools: wasm-bindgen (required), wasm-opt (used by default; taken from PATH or
# fetched pinned through npx), brotli and gzip (used if present, to write
# pre-compressed copies). WASM_OPT=0 skips the wasm-opt pass.
#
#   WEBGL=1       — adds the WebGL2 fallback to any of the above, for browsers
#                   with no WebGPU. Off by default, and the reason is the size:
#                   measured on mini, after wasm-bindgen, 1.64 MiB becomes
#                   5.41 MiB. The GL backend brings glow and naga's GLSL
#                   translator, which is most of that — so a WebGPU-only
#                   deployment should not pay it, and mini in particular exists
#                   to be small.
#
#   THREADS=1     — threads in the browser: frames, tiles and loop filters of
#                   `decodeVideo` run in parallel. Off by default: it needs a
#                   nightly toolchain, `initThreadPool(n)` called once, blocking
#                   calls made from a Worker, and a page served with COOP and
#                   COEP headers.
#
#   SIMD=1        — compiles with wasm SIMD (`-C target-feature=+simd128`), so
#                   LLVM can auto-vectorise. Off by default: a module with any
#                   SIMD instruction fails to load on engines without it (Safari
#                   before 16.4), and there is no in-module feature detection.
#
# Output always lands in web/pkg/ (and regenerated features.js). The npm package
# stages mini + full under crates/threers-js/dist/{mini,full}/.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

# wasm-bindgen's CLI must match the crate in Cargo.lock exactly; a mismatch
# only shows after the whole cargo build, as a schema-version error. Settle it
# first, with a repo-local install if the one on PATH is another version.
. "$ROOT/scripts/wasm-bindgen-cli.sh"
use_matching_wasm_bindgen "$ROOT"

PROFILE="${PROFILE:-release}"
PROFILE_FLAG=""
# Honour CARGO_TARGET_DIR: cargo writes where it is told, and hardcoding
# `target/` here only meant wasm-bindgen then failed to find the artifact cargo
# had just built.
CARGO_OUT="${CARGO_TARGET_DIR:-target}"
TARGET_DIR="$CARGO_OUT/wasm32-unknown-unknown/debug"
if [ "$PROFILE" = "release" ]; then
    # `release-wasm`, not `release`: same optimisations plus the size-tuned
    # settings, which belong only to this target. See the profile's note in
    # Cargo.toml for the measured difference.
    PROFILE_FLAG="--profile release-wasm"
    TARGET_DIR="$CARGO_OUT/wasm32-unknown-unknown/release-wasm"
fi

VARIANT="${VARIANT:-}"
NO_DEFAULT_FEATURES=""
FEATURE_LIST=()

if [ "$VARIANT" = "mini" ]; then
    echo "==> Variant: mini (core renderer, --no-default-features --features render,audio)"
    NO_DEFAULT_FEATURES="--no-default-features"
    # The renderer is itself a feature now; mini is the renderer and nothing
    # else, plus audio, which the shim imports by name (`AudioListener`).
    FEATURE_LIST+=("render" "audio")
    # Clear additive feature envs so a leftover OPENSCAD=1 cannot inflate mini.
    unset MESH_BVH BVH_CSG NATIVE_CODEC NATIVE_DECODE OPENSCAD NURBS BREP BREP_CSG BREP_KERNEL STEP RLX RLX_GEO FLUID || true
elif [ "$VARIANT" = "full" ]; then
    echo "==> Variant: full (Cargo feature wasm-full)"
    FEATURE_LIST+=("wasm-full")
    # Mirror flags for generate-features-addon.mjs / post-build.
    export OPENSCAD=1
    export NATIVE_CODEC=1
    export NURBS=1
    export BVH_CSG=1
    export MESH_BVH=1
    export FLUID=1
elif [ -n "$VARIANT" ]; then
    echo "ERROR: unknown VARIANT='$VARIANT' (use mini, full, or leave unset)"
    exit 1
fi

# WebGL2 fallback, composable with any variant: WEBGL=1 VARIANT=full ./web/build.sh
#
# Off by default because of what it costs: on mini, after wasm-bindgen, 1.64 MiB
# becomes 5.41 MiB — the GL backend pulls in glow and naga's GLSL translator, and
# a WebGPU-only deployment never executes any of it. Turn it on to serve browsers
# without WebGPU, where the alternative is a blank canvas. WebGPU is still
# preferred wherever it exists, so this only widens what runs.
if [ "${WEBGL:-}" = "1" ]; then
    FEATURE_LIST+=("webgl")
    echo "    (webgl feature enabled — WebGL2 fallback for browsers without WebGPU)"
fi

# Wasm SIMD, composable with any variant: SIMD=1 VARIANT=full ./web/build.sh
#
# A target feature, not a Cargo feature: it changes codegen for every crate, so
# it goes through RUSTFLAGS (appended, so a caller's flags survive). Changing
# RUSTFLAGS invalidates cargo's cache, so switching this rebuilds everything.
WASM_OPT_SIMD=""
if [ "${SIMD:-}" = "1" ]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
    WASM_OPT_SIMD="--enable-simd"
    echo "    (simd128 enabled — needs Chrome 91+, Firefox 89+, Safari 16.4+)"
fi

# Threads, composable like SIMD: THREADS=1 SIMD=1 NATIVE_DECODE=1 ./web/build.sh
#
# Rust's prebuilt wasm std has no atomics, so this rebuilds std on a pinned
# nightly. The linker flags are not optional: without them rustc emits an
# unshared memory, and workers fail with "Memory could not be cloned".
CARGO_CMD="cargo"
BUILD_STD=""
WASM_OPT_THREADS=""
if [ "${THREADS:-}" = "1" ]; then
    THREADS_TOOLCHAIN="${THREADS_TOOLCHAIN:-nightly-2026-10-30}"
    FEATURE_LIST+=("wasm-threads")
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
    CARGO_CMD="cargo +$THREADS_TOOLCHAIN"
    BUILD_STD="-Z build-std=panic_abort,std"
    WASM_OPT_THREADS="--enable-threads"
    echo "    (threads enabled — call initThreadPool(n), decode in a Worker, serve with COOP/COEP)"
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
    if [ "${FLUID:-}" = "1" ]; then
        FEATURE_LIST+=("fluid")
        echo "    (fluid feature enabled — fluidStart() for web/fluid.html)"
    fi
    if [ "${NATIVE_CODEC:-}" = "1" ]; then
        FEATURE_LIST+=("native-codec")
        echo "    (native-codec feature enabled — GIF/APNG/WebM/MP4 browser export)"
    fi
    if [ "${NATIVE_DECODE:-}" = "1" ]; then
        FEATURE_LIST+=("native-decode")
        echo "    (native-decode feature enabled — decodeVideo(): H.264/HEVC/VP9/AV1 to RGBA)"
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
$CARGO_CMD build --target wasm32-unknown-unknown --lib $PROFILE_FLAG $NO_DEFAULT_FEATURES $FEATURES $BUILD_STD

echo "==> Generating JS bindings into web/pkg..."
# The profile strips debug info only, so wasm-bindgen can still read the
# `target_features` section; the symbol names and producers go here instead.
wasm-bindgen \
    "$TARGET_DIR/threers.wasm" \
    --target web \
    --out-dir web/pkg \
    --no-typescript \
    --remove-name-section \
    --remove-producers-section

# --- wasm-opt -------------------------------------------------------------
#
# Part of the toolchain rather than something that happens to be installed: it
# is taken from PATH if present, and otherwise fetched at a pinned version
# through npx, which needs no admin rights and installs nothing system-wide —
# the same route `scripts/wasm-browser-test.sh` uses for Chrome. Override the
# binary with WASM_OPT_BIN, or skip the pass with WASM_OPT=0.
#
# What it buys, measured on mini+webgl built with the size profile:
#
#                       raw        brotli      compile (V8)
#   without          2.86 MiB    0.785 MiB        6.7 ms
#   with -Oz         2.61 MiB    0.812 MiB        5.1 ms
#
# So it is a trade, not a free win: nine per cent off the file and a quarter off
# the browser's compile time, against twenty-seven kilobytes more to download.
# Which side wins depends on the connection — on a slow link the extra transfer
# costs more than the compile saves. It runs by default because the file and the
# compile are the parts this project controls; set WASM_OPT=0 when serving
# compressed and optimising for transfer above all.
#
# The `--enable-*` flags are not optional: wasm-bindgen's output uses features
# wasm-opt will not assume, and without them it refuses the input with a
# validation error about bulk memory.
BINARYEN_VERSION="132"
WASM_OPT_BIN="${WASM_OPT_BIN:-}"
if [ "${WASM_OPT:-1}" != "0" ]; then
    if [ -z "$WASM_OPT_BIN" ] && command -v wasm-opt >/dev/null 2>&1; then
        WASM_OPT_BIN="wasm-opt"
    fi
    if [ -z "$WASM_OPT_BIN" ] && command -v npx >/dev/null 2>&1; then
        WASM_OPT_BIN="npx --yes -p binaryen@$BINARYEN_VERSION wasm-opt"
        echo "    (wasm-opt via npx binaryen@$BINARYEN_VERSION — install binaryen to skip the fetch)"
    fi
    if [ -n "$WASM_OPT_BIN" ]; then
        before=$(wc -c < web/pkg/threers_bg.wasm | tr -d ' ')
        echo "==> wasm-opt -Oz…"
        $WASM_OPT_BIN \
            --enable-bulk-memory --enable-bulk-memory-opt --enable-reference-types \
            --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
            --enable-multivalue $WASM_OPT_SIMD $WASM_OPT_THREADS -Oz \
            -o web/pkg/threers_bg.wasm.opt web/pkg/threers_bg.wasm
        mv web/pkg/threers_bg.wasm.opt web/pkg/threers_bg.wasm
        after=$(wc -c < web/pkg/threers_bg.wasm | tr -d ' ')
        echo "    $before -> $after bytes"
    else
        echo "    (no wasm-opt and no npx — skipping; brew install binaryen)"
    fi
else
    echo "==> wasm-opt skipped (WASM_OPT=0)"
fi

# Pre-compressed copies, so a static host can serve them with
# `Content-Encoding` instead of compressing per request. Brotli is what to serve:
# measured against gzip -9 and zstd -22 on this artifact, brotli -q 11 wins
# (0.785 MiB against 1.001 and 0.830). A larger brotli window buys nothing here —
# the file is under 16 MiB, so the default already spans it.
if command -v brotli >/dev/null 2>&1; then
    brotli -q 11 -f -o web/pkg/threers_bg.wasm.br web/pkg/threers_bg.wasm
    echo "==> wrote threers_bg.wasm.br ($(wc -c < web/pkg/threers_bg.wasm.br | tr -d ' ') bytes)"
fi
if command -v gzip >/dev/null 2>&1; then
    gzip -9 -c web/pkg/threers_bg.wasm > web/pkg/threers_bg.wasm.gz
    echo "==> wrote threers_bg.wasm.gz ($(wc -c < web/pkg/threers_bg.wasm.gz | tr -d ' ') bytes)"
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
