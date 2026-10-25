#!/bin/zsh
#
# Run the wasm tests in a real browser.
#
# `wasm-bindgen-test` drives a browser over WebDriver, which needs a driver
# binary. On this machine neither of the obvious routes works: Homebrew's
# `chromedriver` cask is disabled for failing Gatekeeper, and Safari refuses a
# WebDriver session until someone runs `sudo safaridriver --enable`. So this
# fetches Chrome for Testing and a version-matched chromedriver through
# `@puppeteer/browsers`, which needs no admin rights and installs nothing
# system-wide.
#
#   scripts/wasm-browser-test.sh [cargo test args...]
#   FEATURES=webgl TEST=webgl scripts/wasm-browser-test.sh
#   WEBGPU=1 FEATURES=captions TEST=webgpu scripts/wasm-browser-test.sh
#
# The download happens once and is cached; pass a different directory in
# `BROWSER_CACHE` to put it somewhere other than the default.
#
# By default Chrome runs with no WebGPU adapter and WebGL2 from SwiftShader —
# exactly the right environment for testing the WebGL one. `WEBGPU=1` instead
# launches it with `--enable-unsafe-webgpu`, which in headless Chrome on macOS
# gives a real WebGPU adapter (Dawn on Metal) for `tests/webgpu.rs`.
set -eu

here=${0:A:h}
root=${here:h}
cache=${BROWSER_CACHE:-${TMPDIR:-/tmp}/threers-browsers}

command -v npx >/dev/null || { echo "npx is needed to fetch the browser" >&2; exit 2 }
# The runner has to be the version of the `wasm-bindgen` crate in Cargo.lock,
# or it refuses the wasm. The one on PATH if it is; else one wasm-pack has
# cached for that version.
want=$(awk '/^name = "wasm-bindgen"$/ {getline; gsub(/version = |"/, ""); print; exit}' "${0:A:h:h}/Cargo.lock")
runner=""
if command -v wasm-bindgen-test-runner >/dev/null \
   && [[ "$(wasm-bindgen-test-runner --version 2>/dev/null | awk '{print $2}')" == "$want" ]]; then
  runner=wasm-bindgen-test-runner
else
  for dir in "$HOME/Library/Caches/.wasm-pack" "$HOME/.cache/.wasm-pack"; do
    c="$dir/wasm-bindgen-cargo-install-$want/wasm-bindgen-test-runner"
    [[ -x "$c" ]] && { runner="$c"; break }
  done
fi
[[ -n "$runner" ]] \
  || { echo "need wasm-bindgen-test-runner $want: cargo install -f wasm-bindgen-cli --version $want" >&2; exit 2 }

mkdir -p "$cache"
# The two have to be the same version or chromedriver refuses to drive it.
driver=$(npx --yes @puppeteer/browsers install chromedriver@stable --path "$cache" 2>/dev/null | tail -1 | sed 's/^[^ ]* //')
chrome=$(npx --yes @puppeteer/browsers install chrome@stable --path "$cache" 2>/dev/null | tail -1 | sed 's/^[^ ]* //')
[[ -x "$driver" ]] || { echo "no chromedriver at '$driver'" >&2; exit 1 }
[[ -x "$chrome" ]] || { echo "no chrome at '$chrome'" >&2; exit 1 }
# Downloaded binaries are quarantined, and a quarantined chromedriver dies with
# SIGKILL the moment it is exec'd — which reads as a driver crash, not a
# permission problem.
xattr -dr com.apple.quarantine "$cache" 2>/dev/null || true

# `wasm-bindgen-test-runner` reads capabilities from `webdriver.json` in the
# crate root. It holds an absolute path to a downloaded browser, so it is
# written for the run and taken away after rather than committed.
config="$root/webdriver.json"
cleanup() { rm -f "$config" }
trap cleanup EXIT INT TERM
python3 - "$chrome" "$config" "${WEBGPU:-0}" <<'PY'
import json, sys
# `WEBGPU=1`: a Chrome that offers a WebGPU adapter, for `tests/webgpu.rs`.
# Headless Chrome has none unless told it is acceptable; on macOS Dawn then runs
# on Metal. The SwiftShader flags below would instead leave WebGL2 as the only
# adapter, which is what the WebGL suite wants and this one must not get.
if sys.argv[3] not in ("", "0"):
    json.dump({
        "goog:chromeOptions": {
            "binary": sys.argv[1],
            "args": [
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPU",
                "--ignore-gpu-blocklist",
                "--use-angle=metal",
            ],
        }
    }, open(sys.argv[2], "w"), indent=2)
    sys.exit(0)
json.dump({
    "goog:chromeOptions": {
        "binary": sys.argv[1],
        # `--disable-gpu` used to be here. It switches off WebGL as well as
        # WebGPU, so a GL adapter could not be obtained at all — which looks
        # from the test like "this browser has no WebGL2" rather than like a
        # flag. ANGLE-over-SwiftShader gives a real WebGL2 context in software,
        # and recent Chrome will not use SwiftShader for WebGL without being
        # told it is acceptable. None of this affects tests that never ask for
        # an adapter.
        "args": [
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            # `--enable-unsafe-swiftshader` also lets newer Chrome for Testing
            # builds hand out a SwiftShader *WebGPU* adapter, and the renderer
            # prefers WebGPU whenever `navigator.gpu` answers — so the WebGL2
            # suite silently ran on WebGPU and failed its own backend checks.
            # Switching the feature off leaves WebGL2 as the only way to draw.
            "--disable-features=WebGPU",
        ],
    }
}, open(sys.argv[2], "w"), indent=2)
PY

echo "chromedriver: $("$driver" --version | head -1)"
cd "$root"
# Which suite to run. The USD tests are the default because they were the first
# ones here; `FEATURES` and `TEST` pick another without editing this file:
#
#   FEATURES=webgl TEST=webgl scripts/wasm-browser-test.sh
#
# Anything after `--` still goes to cargo, so a filter or `--nocapture` works as
# usual. Note that appending `--features x --test y` instead would *add* a second
# suite rather than replace this one, which is what used to happen.
features=${FEATURES:-usd}
test_name=${TEST:-usd_wasm}

echo "suite: --features $features --test $test_name"
# `wasm-bindgen-test` gives each test twenty seconds by default, and these draw
# through SwiftShader — software WebGL2 — which is fast enough when the machine
# is idle and not when a native `cargo test` is running beside it. A timeout here
# reads as "Failed to detect test as having been run", which looks like a hang
# rather than a deadline. Raised well past what the suite needs (the WebGL2
# suite takes about 225 s on an idle M4 Pro); override with
# WASM_BINDGEN_TEST_TIMEOUT if a test really is stuck.
CHROMEDRIVER="$driver" \
WASM_BINDGEN_TEST_TIMEOUT="${WASM_BINDGEN_TEST_TIMEOUT:-480}" \
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER="$runner" \
  cargo test --target wasm32-unknown-unknown --features "$features" --test "$test_name" "$@"
