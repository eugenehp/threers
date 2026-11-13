#!/bin/zsh
# Builds and runs the demo with the shell every threers Vision Pro app has
# (templates/visionos-app/app): see its build.sh for the settings —
# `device`, CONSOLE, SHOT, RUN, SIM, CARGO_TARGET_DIR. The demo adds:
#
#   SHOW=candles        open on the 3D page's candles
#   SCRIPT="2:tap:97,129; 4:drag:640,600,640,250; 6:poke:97,96"
#                       pinches and fingertip pokes to play (src/script.rs):
#                       UI pixels on the panel, seconds from launch
set -eu
here=${0:A:h}
root=${here:h:h:h}
export CRATE=${here:h}
export APP_NAME=${APP_NAME:-threers}
export BUNDLE_ID=${BUNDLE_ID:-com.threers.uidemo}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target-visionos}
[[ -n ${SHOW:-} ]] && export APP_ENV_THREERS_DEMO_SHOW=$SHOW
[[ -n ${SCRIPT:-} ]] && export APP_ENV_THREERS_XR_SCRIPT=$SCRIPT
exec "$root/templates/visionos-app/app/build.sh" "$@"
