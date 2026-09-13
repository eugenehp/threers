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
#
# The download happens once and is cached; pass a different directory in
# `BROWSER_CACHE` to put it somewhere other than the default.
set -eu

here=${0:A:h}
root=${here:h}
cache=${BROWSER_CACHE:-${TMPDIR:-/tmp}/threers-browsers}

command -v npx >/dev/null || { echo "npx is needed to fetch the browser" >&2; exit 2 }
command -v wasm-bindgen-test-runner >/dev/null \
  || { echo "cargo install wasm-bindgen-cli --version \$(cargo pkgid wasm-bindgen | sed 's/.*#//')" >&2; exit 2 }

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
python3 - "$chrome" "$config" <<'PY'
import json, sys
json.dump({
    "goog:chromeOptions": {
        "binary": sys.argv[1],
        "args": ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    }
}, open(sys.argv[2], "w"), indent=2)
PY

echo "chromedriver: $("$driver" --version | head -1)"
cd "$root"
CHROMEDRIVER="$driver" \
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test --target wasm32-unknown-unknown --features usd --test usd_wasm "$@"
