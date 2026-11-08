# Releasing

Seven crates publish to **crates.io**. Language packages publish to npm and
PyPI (not crates.io). See also [`docs/bindings.md`](bindings.md).

| Artifact | Version | Registry | On the index (2026-11-06) |
|---|---|---|---|
| `threers-codec` | 0.0.12 | crates.io | 0.0.10 |
| `threers-softbody` | 0.0.12 | crates.io | 0.0.10; the soft-body lattice and solver (`gpu` feature: WebGPU compute) |
| `threers` | 0.0.12 | crates.io | 0.0.10 |
| `threers-physics` | 0.0.12 | crates.io | 0.0.10 |
| `threers-animation` | 0.0.12 | crates.io | 0.0.1 |
| `threers-probe` | 0.0.12 | crates.io | 0.0.2 |
| `threers` (npm ESM) | 0.0.12 | npm / Deno | not yet; **mini** (default) + **full** wasm entries ([`crates/threers-js`](../crates/threers-js)) |
| `threers-node` | 0.0.12 | npm | not yet; per-arch `.node` via `@threers/node-*` ([`crates/threers-node`](../crates/threers-node)) |
| `threers` (PyPI) | 0.0.12 | PyPI | not yet; per-platform wheels ([`crates/threers-py`](../crates/threers-py)) |

Everything else under `crates/` is a demo, a bench, a binding or not ready
for the index, and says `publish = false`: `threers-ocean`,
`threers-robot-arm`, `threers-physics-bench`, `threers-connectome`,
`threers-mechanism-tour`, `threers-hop`, `threers-fluid-ios`, the three
binding crates (published as language packages instead), `threers-ecs`
(not on crates.io yet) and `threers-continuum` (held back: a cable under
steady tension does not yet hold the rod at a steady bend). All of them are workspace members, so
`--all-targets` builds their tests and examples too.

## The order is not optional

`threers` depends on `threers-codec` and `threers-softbody` by **path and
version**, so those two go up first. Both companions depend on `threers` the
same way:

```toml
# crates/threers-physics/Cargo.toml
threers = { path = "../..", version = "0.0.12", features = ["mesh-bvh"] }
```

A path dependency is what the workspace builds against; the `version` is what the
published crate carries. So `threers` has to be on crates.io *before* either
companion can even be packaged — until it is, you get

```text
failed to select a version for the requirement `threers = "^0.0.8"`
candidate versions found which didn't match: 0.0.7, 0.0.6, 0.0.5, 0.0.4, 0.0.3, 0.0.2, 0.0.1
```

which is not a fault, it is the ordering telling you about itself.

## Checklist

```bash
# 0. Versions first, because a publish cannot be taken back: npm and crates.io
#    both refuse to replace a version that is already up. `CHECK=1` exits
#    non-zero when any root that tracks the crate version has drifted, and
#    `release-all.sh publish` runs it before it uploads anything.
CHECK=1 ./scripts/release-versions.sh

# 1. Everything builds and passes.
#    `--no-fail-fast` is not optional here. `cargo test` stops at the first
#    failing *binary*, and each crate has many: a lib target, one binary per
#    file in `tests/`, and the doctests last. One failure in the lib therefore
#    hides every integration test and every doctest behind it, and the summary
#    it prints looks like a complete run. That is not hypothetical — it is how
#    22 broken physics tests and every doctest in the workspace stayed invisible
#    through a whole round of "the suite is green".
cargo build --workspace --all-features
cargo test --workspace --all-features --no-fail-fast

# 1b. And with default features, not just all of them. An example with no
#     `required-features`, or a test using a feature-gated API without a `cfg`,
#     only fails in the configuration nobody runs.
cargo test -p threers-physics --no-fail-fast

# 2. Clean at every feature combination that ships, not just the default one.
#    `--all-targets` matters: it is the only thing that compiles the examples,
#    and an example that stopped compiling is invisible to `cargo build`.
cargo clippy --workspace --all-targets --all-features

# 2b. Each feature on its own, as a library. `--all-features` hides a feature
#     that forgot to declare a dependency, because some other feature pulls it
#     in; `--all-targets` hides it too, because dev-dependencies are present for
#     tests and examples but not for the consumer who just adds the crate. That
#     combination is exactly how `gpu` shipped needing `pollster` without
#     depending on it: every check here passed, and `cargo add threers-physics
#     --features gpu` would not have compiled.
for f in captions mesh-bvh raytrace bvh-csg openscad manifold planet video \
         native-codec metal videotoolbox parallel async rlx rlx-geo nurbs brep \
         brep-csg brep-kernel step assembly-check learned-denoise; do
  cargo build -p threers --no-default-features --features "$f" --lib || break
done
for f in parallel async gpu mechanism assembly openscad; do
  cargo build -p threers-physics --no-default-features --features "$f" --lib || break
done

# 3. wasm still builds — it is half the point of the project.
cargo clippy --target wasm32-unknown-unknown --lib
cargo clippy --target wasm32-unknown-unknown --lib --features openscad
cargo clippy --target wasm32-unknown-unknown --lib --features webgl

# 2c. The soft-body solver still matches its JavaScript original bit for bit,
#     and the GPU solver still matches the CPU one (both need a GPU adapter;
#     they skip without one).
cargo test --release --test softbody_parity --test softbody_gpu -- --test-threads 1

# 3b. And the WebGL2 backend actually draws, which only a browser can answer.
#     Needs a chromedriver on PATH; see docs/webgl.md.
FEATURES=webgl TEST=webgl ./scripts/wasm-browser-test.sh

# 3c. The codec crate on its own: every feature combination, its declared
#     minimum Rust (1.87), and the targets its hardware backends build for.
cargo test -p threers-codec --release --all-features
cargo test -p threers-codec --release --no-default-features
cargo +1.87 check -p threers-codec --all-targets --all-features
cargo clippy -p threers-codec --all-targets --features vaapi,nvidia,parallel --target x86_64-unknown-linux-gnu
cargo clippy -p threers-codec --all-targets --features nvidia --target x86_64-pc-windows-msvc

# 3d. The hardware backends, on machines that have the hardware. Each suite
#     skips without it, so require it, or a green run can mean nothing ran.
#     NVIDIA (Linux; RUST_TEST_THREADS=1, see docs/codecs.md):
THREERS_REQUIRE_NVIDIA=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features nvidia,parallel
#     A VA driver: AMD (Mesa) and Intel (iHD) both, they read different fields.
THREERS_REQUIRE_VAAPI=1 RUST_TEST_THREADS=1 cargo test -p threers-codec --release --features vaapi,parallel
#     VideoToolbox (macOS):
cargo test -p threers-codec --release --features videotoolbox --test hw

# 3e. Everything that skips itself, with its inputs in place. A skip reports
#     `ok`; this fails on any skip or empty test target not listed in
#     scripts/allowed-skips.txt.
THREERS_H264_CONFORMANCE=~/tools/h264-conformance \
THREERS_HEVC_CONFORMANCE=~/tools/hevc-conformance \
THREERS_HM_DECODER=~/tools/hm/bin/umake/clang-21.0/x86_64/release/TAppDecoder \
  scripts/test-no-skips.py -p threers-codec --release --features parallel,videotoolbox

# 3f. The renderer in real Chrome, on WebGL2 and on WebGPU.
scripts/browser-smoke.sh

# 4. Docs, including the intra-doc links between the crates.
#    Keep this *after* the tests, and run `cargo clean -p threers` before going
#    back to them. `-p threers -p threers-physics -p threers-probe` is a narrower
#    package set than `--workspace`, so `--all-features` resolves to a narrower
#    feature set, and the `threers` rlib it leaves behind is missing features the
#    full build has. Doctests link against whatever rlib is there, so the next
#    `cargo test --workspace` reports things like "could not find `assembly` in
#    `threers`" — 41 failures in a crate whose doctests all pass when built on
#    their own. Nothing is wrong with the code; the artifact is just the wrong
#    one, and it persists until it is cleaned.
cargo doc --no-deps --all-features -p threers -p threers-physics -p threers-probe
cargo doc --no-deps --all-features -p threers-codec
# docs.rs builds the codec crate for Linux, where its NVIDIA and VAAPI modules
# exist and the VideoToolbox ones do not; links must hold there too.
cargo doc --no-deps -p threers-codec --features vaapi,nvidia,gif,parallel --target x86_64-unknown-linux-gnu

# 5. What would actually ship.
cargo package -p threers --list | wc -l
cargo package -p threers                     # size, and it must compile from the tarball
cargo package -p threers-softbody --features gpu   # and the new crates, from theirs
cargo package -p threers-codec

# 6. Then, in this order — the two threers depends on first.
cargo publish -p threers-codec
cargo publish -p threers-softbody
cargo publish -p threers
cargo publish -p threers-physics             # only after the first is on the index
cargo publish -p threers-probe               # likewise
cargo publish -p threers-animation           # after threers-physics, which it depends on
git tag -a v0.0.12 -m 'threers 0.0.12' && git push --tags

# 7. Language packages — one script locally; CI on tag push.
./scripts/release-versions.sh                    # versions aligned?
./scripts/release-all.sh pack                    # build + npm pack + wheels
PUBLISH=1 ./scripts/release-all.sh publish     # upload (or push tag → CI)

# Multi-arch npm-node + PyPI: push v* tag → .github/workflows/release-language-packages.yml
git tag -a v0.0.12 -m 'threers 0.0.12' && git push --tags
```

## One command (`scripts/release-all.sh`)

| Command | What it does |
|---------|----------------|
| `./scripts/release-all.sh` | Build npm ESM (mini+full), npm-node (host), PyPI wheel (host) |
| `./scripts/release-all.sh pack` | Above + `npm pack` tarballs |
| `PUBLISH=1 ./scripts/release-all.sh publish` | Upload language packages |
| `TARGETS=crates ./scripts/release-all.sh publish` | crates.io only |
| `TARGETS=all PUBLISH=1 ./scripts/release-all.sh publish` | crates.io + language |

```bash
./scripts/release-versions.sh   # print versions from every manifest
./scripts/release-all.sh pack
PUBLISH=1 TARGETS=npm-esm ./scripts/release-all.sh publish   # just wasm npm
```

**Multi-arch** (`@threers/node-*`, PyPI arm64/x64 matrix): push a `v*` tag or run
`gh workflow run release-language-packages.yml` after secrets are set (`NPM_TOKEN`,
PyPI trusted publishing).

```bash
# After crates.io publish + version bump:
git tag v0.0.12 && git push origin v0.0.12
```

## npm ESM (`crates/threers-js`)

Publish root for browsers, Deno (`npm:threers`), and Node ESM. Day-to-day JS
still lives under `web/`; `build.sh` stages **two** wasm variants:

| Entry | Cargo | Role |
|-------|-------|------|
| `threers` / `threers/mini` | `--no-default-features` | Smallest browser download (default) |
| `threers/full` | `--features wasm-full` | OpenSCAD/CSG, NURBS, codecs, planet, raytrace |

```bash
./crates/threers-js/build.sh          # builds mini then full into dist/
cd crates/threers-js && npm pack && npm publish

# local web/pkg only:
VARIANT=mini web/build.sh
VARIANT=full web/build.sh
```

## npm native (`crates/threers-node`)

napi-rs addon (`#[napi]` — **not** Neon). Parent package `threers-node` plus
optional platform packages:

| Triple | Package |
|--------|---------|
| `aarch64-apple-darwin` | `@threers/node-darwin-arm64` |
| `x86_64-apple-darwin` | `@threers/node-darwin-x64` |
| `x86_64-unknown-linux-gnu` | `@threers/node-linux-x64-gnu` |
| `aarch64-unknown-linux-gnu` | `@threers/node-linux-arm64-gnu` |
| `x86_64-pc-windows-msvc` | `@threers/node-win32-x64-msvc` |

```bash
cd crates/threers-node
npm i
npm run build          # local host triple
npx napi prepublish -t npm
npm publish
```

Deno / browser → use `threers` (wasm), not `threers-node`.

## PyPI (`crates/threers-py`)

PyO3 extension module `threers._native` plus a thin `python/threers/` package.
Build wheels per arch with [maturin](https://www.maturin.rs) (see the release
workflow for the full arm64/x64 matrix).

```bash
cd crates/threers-py
maturin develop --features headless,physics,animation   # local editable
maturin build --release --features headless,physics,animation
maturin publish --features headless,physics,animation
```

Targeted wheels (examples):

```bash
maturin build --release --features headless,physics,animation --target aarch64-apple-darwin
maturin build --release --features headless,physics,animation --target x86_64-unknown-linux-gnu
```

## What to check in the package

`cargo package --list` is worth reading rather than skimming. Two things have
gone wrong before:

- **Built wasm.** `.gitignore` matched `/web/pkg/*` while the per-demo bundles
  live in `/web/pkg-mechanism-tour/`, `/web/pkg-ocean/` and `/web/pkg-robot-arm/`
  — so megabytes of build output were being both committed and published. The
  ignore now covers `/web/pkg-*/`.
- **Test fixtures.** `tests/parity/scenes` is 22 MB of reference scene dumps that
  only the parity harness can read. `exclude` in `Cargo.toml` keeps them out; the
  limit is 10 MB and it applies to the *compressed* figure.

Also worth a glance, because none of it belongs in a crate: `out/` (example
render output), `papers/`, and `web/assets/earth/` (6 GB of NASA imagery, fetched
rather than stored — see `.gitignore`).

## Versions

crates.io is the source of truth, not the tags:

```bash
curl -s -H 'User-Agent: threers-release' https://crates.io/api/v1/crates/threers \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['crate']['max_version'])"
```

It reads 0.0.10, and 0.0.12 is what ships: 0.0.11 was prepared but never
published or tagged, so the index goes straight from 0.0.10. Note that 0.0.10 was published
but never tagged (the tags stop at `v0.0.9`), and 0.0.9 was tagged but never
published: tag after publishing, so the two agree.

Every published crate now carries the one workspace version, so they all go
up together; `release-versions.sh` treats each as an aligned root. Several
skip versions on the index because of that: `threers-animation` goes from
0.0.1 to 0.0.12 and `threers-probe` from 0.0.2. `CHECK=1
./scripts/release-versions.sh` asks crates.io and fails if a published crate
is not ahead of its version there — `cargo publish` cannot replace a version,
so a stale one would otherwise fail at the upload and not before. Check any
of them by hand the same way:

```bash
for c in threers threers-codec threers-softbody threers-physics threers-animation threers-probe; do
  printf '%-18s ' "$c"
  curl -s -H 'User-Agent: threers-release' "https://crates.io/api/v1/crates/$c" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('crate', {}).get('max_version', 'not published'))"
done
```

The bindings pin `threers-physics` by version as well as by path, so bumping it
means bumping `crates/threers-py/Cargo.toml`, `crates/threers-node/Cargo.toml`
and `crates/threers-animation/Cargo.toml` with it or nothing resolves. `release-versions.sh` checks that pairing.
