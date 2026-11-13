#!/bin/zsh
# Builds this crate as an Apple Vision Pro app — no Xcode project: the Rust
# library with cargo, the SwiftUI shell (App.swift) with swiftc, an
# Info.plist and a signature — and runs it.
#
#   app/build.sh               the simulator: build, install, launch
#   app/build.sh device        a headset: SIGN_IDENTITY="Apple Development: …"
#                              and PROFILE=path/to.mobileprovision
#
# Settings (environment): APP_NAME (shown under the icon; default the
# package name), BUNDLE_ID (default com.example.<package>), MIN_OS (default
# 26.0), SIM (a simulator's UDID; default the first Apple Vision Pro),
# RUN=0 to build only, CONSOLE=file for the app's output, SHOT=secs:file.png
# for a screenshot that long after launch, CARGO_TARGET_DIR,
# CRATE (another crate to build with this shell), and APP_ENV_<NAME>=value
# for each setting the app reads as <NAME> in the simulator.
set -eu
here=${0:A:h}
# The crate: this folder's parent, or CRATE (a crate that shares this shell).
crate=${CRATE:-${here:h}}
kind=${1:-sim}
package=$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$crate/Cargo.toml" | head -1)
lib=${package//-/_}
app_name=${APP_NAME:-$package}
bundle_id=${BUNDLE_ID:-com.example.$lib}
min_os=${MIN_OS:-26.0}
version=$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$crate/Cargo.toml" | head -1)
version=${version:-1.0}  # a workspace's crate gives none of its own
target_dir=${CARGO_TARGET_DIR:-$crate/target}

if [[ $kind == sim ]]; then
  rust_target=aarch64-apple-visionos-sim swift_target=arm64-apple-xros$min_os-simulator sdk=xrsimulator platform=XRSimulator
else
  rust_target=aarch64-apple-visionos swift_target=arm64-apple-xros$min_os sdk=xros platform=XROS
fi

echo "==> $package for $rust_target"
XROS_DEPLOYMENT_TARGET=$min_os CARGO_TARGET_DIR=$target_dir \
  cargo build --manifest-path "$crate/Cargo.toml" --lib --target $rust_target --release

app=$target_dir/visionos-app/$kind/$lib.app
rm -rf "$app"
mkdir -p "$app"
xcrun --sdk $sdk swiftc -O -parse-as-library -target $swift_target \
  "$here/App.swift" "$target_dir/$rust_target/release/lib$lib.a" \
  -framework CompositorServices -framework ARKit -framework Metal -framework QuartzCore \
  -framework CoreGraphics -framework CoreFoundation -framework Foundation -lobjc -liconv \
  -o "$app/$lib"
sed -e "s/APP_NAME/$app_name/" -e "s/APP_EXECUTABLE/$lib/g" -e "s/APP_VERSION/$version/" \
    -e "s/BUNDLE_ID/$bundle_id/" -e "s/PLATFORM/$platform/" -e "s/MIN_OS/$min_os/" \
    "$here/Info.plist" > "$app/Info.plist"
printf 'APPL????' > "$app/PkgInfo"
if [[ $kind == sim ]]; then
  codesign --force --sign - "$app"
else
  [[ -n ${PROFILE:-} ]] && cp "$PROFILE" "$app/embedded.mobileprovision"
  codesign --force --sign "${SIGN_IDENTITY:?set SIGN_IDENTITY to your signing identity}" "$app"
fi
echo "==> built $app"

if [[ $kind == sim && ${RUN:-1} == 1 ]]; then
  sim=${SIM:-$(xcrun simctl list devices available | grep -i 'Apple Vision Pro' | head -1 | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/')}
  [[ -z $sim ]] && { echo "no Apple Vision Pro simulator: Xcode → Settings → Components"; exit 1; }
  xcrun simctl boot "$sim" 2>/dev/null || true
  open -a Simulator 2>/dev/null || true
  xcrun simctl install "$sim" "$app"
  # APP_ENV_<NAME> reaches the app as <NAME> (simctl's SIMCTL_CHILD_ prefix).
  for name in ${(k)parameters[(I)APP_ENV_*]}; do
    export "SIMCTL_CHILD_${name#APP_ENV_}=${(P)name}"
  done
  if [[ -n ${CONSOLE:-} ]]; then
    xcrun simctl launch --console-pty --terminate-running-process "$sim" "$bundle_id" > "$CONSOLE" 2>&1 &
  else
    xcrun simctl launch --terminate-running-process "$sim" "$bundle_id"
  fi
  echo "==> running in the simulator ($sim)"
  if [[ -n ${SHOT:-} ]]; then
    sleep "${SHOT%%:*}"
    xcrun simctl io "$sim" screenshot "${SHOT#*:}" > /dev/null
    echo "==> screenshot ${SHOT#*:}"
  fi
fi
