#!/bin/zsh
# Builds a threers-ios app — a Rust crate with `threers_ios::app!(MyApp)` —
# into an iPhone/iPad or Apple Vision Pro app, with no Xcode project: the
# crate as a static library with cargo (whatever its `crate-type`), the
# shared shell (Harness.swift) with swiftc, an Info.plist and a signature.
# Then runs it.
#
#   app/build.sh path/to/crate                 iPhone simulator
#   app/build.sh path/to/crate vision          Apple Vision Pro simulator: a volume
#                                              (RealityKit) and the immersive space
#   app/build.sh path/to/crate vision-window   the same with a flat Metal window
#   app/build.sh path/to/crate ios-device      } SIGN_IDENTITY="Apple Development: …"
#   app/build.sh path/to/crate vision-device   } and PROFILE=path/to.mobileprovision
#
# Environment: APP_NAME (under the icon; default the package name),
# BUNDLE_ID (default com.example.<lib>), SIM (a simulator's UDID; default
# the first iPhone / Apple Vision Pro), RUN=0 to build only, CONSOLE=file
# for the app's output, SHOT=secs:file.png for a screenshot that long after
# launch, STRIP=0 to keep the symbols, FEATURES for the crate's cargo
# features (NO_DEFAULT_FEATURES=1 without its default ones),
# CARGO_TARGET_DIR. Variables named
# APP_ENV_<NAME> reach the app in the simulator as <NAME>.
# APP_LINK adds linker words (frameworks or libraries the crate's own
# dependencies need, e.g. "-framework Accelerate" for BLAS).
set -eu
here=${0:A:h}
crate=${1:?usage: app/build.sh path/to/crate [ios|ios-device|vision|vision-device]}
crate=${crate:A}
kind=${2:-ios}
package=$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$crate/Cargo.toml" | head -1)
lib=${package//-/_}
app_name=${APP_NAME:-$package}
bundle_id=${BUNDLE_ID:-com.example.$lib}
version=$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$crate/Cargo.toml" | head -1)
target_dir=${CARGO_TARGET_DIR:-$crate/target}

case $kind in
  ios)           rust_target=aarch64-apple-ios-sim      min_os=17.0 swift_target=arm64-apple-ios17.0-simulator  sdk=iphonesimulator platform=iPhoneSimulator ;;
  ios-device)    rust_target=aarch64-apple-ios          min_os=17.0 swift_target=arm64-apple-ios17.0            sdk=iphoneos        platform=iPhoneOS ;;
  vision|vision-window) rust_target=aarch64-apple-visionos-sim min_os=26.0 swift_target=arm64-apple-xros26.0-simulator sdk=xrsimulator     platform=XRSimulator ;;
  vision-device) rust_target=aarch64-apple-visionos     min_os=26.0 swift_target=arm64-apple-xros26.0           sdk=xros            platform=XROS ;;
  *) echo "usage: app/build.sh path/to/crate [ios|ios-device|vision|vision-window|vision-device]"; exit 1 ;;
esac
if [[ $kind == ios* ]]; then
  frameworks=(-framework ARKit -framework SceneKit) families='[1,2]'
else
  frameworks=(-framework CompositorServices -framework ARKit -framework RealityKit) families='[7]'
fi

echo "==> $package for $rust_target"
IPHONEOS_DEPLOYMENT_TARGET=17.0 XROS_DEPLOYMENT_TARGET=26.0 CARGO_TARGET_DIR=$target_dir \
  cargo rustc --manifest-path "$crate/Cargo.toml" --lib --crate-type staticlib --target $rust_target --release \
  ${FEATURES:+--features=$FEATURES} ${NO_DEFAULT_FEATURES:+--no-default-features}

app=$target_dir/threers-ios/$kind/$lib.app
rm -rf "$app"
mkdir -p "$app"
flags=()
[[ $kind == vision-window ]] && flags=(-D THREERS_WINDOW)
xcrun --sdk $sdk swiftc -O -parse-as-library -target $swift_target $flags \
  "$here/Harness.swift" "$target_dir/$rust_target/release/lib$lib.a" \
  $frameworks -framework UIKit -framework SwiftUI -framework Metal -framework QuartzCore \
  -framework CoreGraphics -framework CoreFoundation -framework Foundation -framework Security \
  -lobjc -liconv -lc++ ${=APP_LINK:-} -Xlinker -dead_strip -o "$app/$lib"
# the symbols are for the debugger, not the device
[[ ${STRIP:-1} == 1 ]] && strip "$app/$lib"
plist=$app/Info.plist
cp "$here/Info.plist" "$plist"
plutil -replace CFBundleExecutable -string "$lib" "$plist"
plutil -replace CFBundleName -string "$lib" "$plist"
plutil -replace CFBundleDisplayName -string "$app_name" "$plist"
plutil -replace CFBundleIdentifier -string "$bundle_id" "$plist"
plutil -replace CFBundleShortVersionString -string "$version" "$plist"
plutil -replace MinimumOSVersion -string "$min_os" "$plist"
plutil -replace CFBundleSupportedPlatforms -json "[\"$platform\"]" "$plist"
plutil -replace UIDeviceFamily -json "$families" "$plist"
# a volume opens first: say so, or visionOS hands the app a window scene it has no window for
[[ $kind == vision || $kind == vision-device ]] && \
  plutil -replace UIApplicationSceneManifest.UIApplicationPreferredDefaultSceneSessionRole -string UIWindowSceneSessionRoleVolumetricApplication "$plist"
# the app's icons (and any images): its asset catalog, app/Assets.xcassets
# (`AppIcon`: an .appiconset for iOS, a layered .solidimagestack for
# visionOS), compiled for this platform, its keys merged into the plist
if [[ -d "$crate/app/Assets.xcassets" ]]; then
  if [[ $kind == ios* ]]; then devices=(--target-device iphone --target-device ipad); else devices=(--target-device vision); fi
  partial=$target_dir/threers-ios/$kind/assets.plist
  xcrun actool "$crate/app/Assets.xcassets" --compile "$app" --platform $sdk --minimum-deployment-target $min_os \
    --app-icon AppIcon $devices --output-partial-info-plist "$partial" --errors --notices > /dev/null
  /usr/libexec/PlistBuddy -c "Merge $partial" "$plist" > /dev/null
fi
printf 'APPL????' > "$app/PkgInfo"
[[ $kind == vision* ]] && cp "$here/VertexColor.usda" "$app/"
if [[ $kind == *-device ]]; then
  [[ -n ${PROFILE:-} ]] && cp "$PROFILE" "$app/embedded.mobileprovision"
  codesign --force --sign "${SIGN_IDENTITY:?set SIGN_IDENTITY to your signing identity}" "$app"
else
  codesign --force --sign - "$app"
fi
echo "==> built $app"

[[ $kind == *-device || ${RUN:-1} != 1 ]] && exit 0
if [[ $kind == ios ]]; then pattern='iPhone'; else pattern='Apple Vision Pro'; fi
sim=${SIM:-$(xcrun simctl list devices available | grep -i "$pattern" | head -1 | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/')}
[[ -z $sim ]] && { echo "no $pattern simulator: Xcode → Settings → Components"; exit 1; }
xcrun simctl boot "$sim" 2>/dev/null || true
open -a Simulator 2>/dev/null || true
xcrun simctl install "$sim" "$app"
for v in ${(k)parameters[(I)APP_ENV_*]}; do export "SIMCTL_CHILD_${v#APP_ENV_}=${(P)v}"; done
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
