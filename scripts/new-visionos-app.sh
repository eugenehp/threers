#!/bin/zsh
# Creates a threers app for Apple Vision Pro from templates/visionos-app.
#
#   scripts/new-visionos-app.sh path/to/my-app [package-name]
#
# The package name defaults to the folder's name. LOCAL=1 depends on this
# checkout of threers instead of the git repository. Then:
#
#   cd path/to/my-app && app/build.sh      # runs in the Vision Pro simulator
set -eu
root=${0:A:h:h}
dest=${1:?usage: scripts/new-visionos-app.sh path/to/my-app [package-name]}
[[ -e $dest ]] && { echo "$dest already exists"; exit 1; }
name=${2:-${dest:t}}
name=${name:l}
name=${name// /-}
mkdir -p "$dest"
cp -R "$root/templates/visionos-app/." "$dest/"
sed -i '' "s/^name = \"my-vision-app\"/name = \"$name\"/" "$dest/Cargo.toml"
if [[ ${LOCAL:-} == 1 ]]; then
  sed -i '' "s#git = \"https://github.com/eugenehp/threers\"#path = \"$root\"#g" "$dest/Cargo.toml"
fi
chmod +x "$dest/app/build.sh"
missing=()
for t in aarch64-apple-visionos aarch64-apple-visionos-sim; do
  rustup target list --installed 2>/dev/null | grep -qx $t || missing+=($t)
done
echo "created $dest ($name)"
(( ${#missing} )) && echo "first: rustup target add ${missing[*]}"
xcrun --sdk xrsimulator --show-sdk-path > /dev/null 2>&1 || echo "first: install the visionOS platform in Xcode → Settings → Components"
echo "then:  cd $dest && app/build.sh"
