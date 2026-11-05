#!/bin/zsh
#
# The renderer in real Chrome on both of its web backends. Native tests cannot
# stand in for either: WebGL2 goes through naga's GLSL and ANGLE, and on WebGPU
# Tint rejects WGSL that naga accepts, which fails silently as a black canvas.
#
#   scripts/browser-smoke.sh
#
# Both suites run even if the first fails. tests/webgpu.rs fails rather than
# passes if Chrome hands it anything but a WebGPU adapter.
set -u

here=${0:A:h}
failed=0
FEATURES=webgl TEST=webgl "$here/wasm-browser-test.sh" "$@" || failed=1
WEBGPU=1 FEATURES=captions TEST=webgpu "$here/wasm-browser-test.sh" "$@" || failed=1
exit $failed
