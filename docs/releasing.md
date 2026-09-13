# Releasing

Three crates publish to **crates.io**. Language packages publish to npm and
PyPI (not crates.io). See also [`docs/bindings.md`](bindings.md).

| Artifact | Version | Registry | Arch notes |
|---|---|---|---|
| `threers` | 0.0.6 | crates.io | source; consumers compile |
| `threers-physics` | 0.0.1 | crates.io | first release |
| `threers-probe` | 0.0.1 | crates.io | first release |
| `threers` (npm ESM) | 0.0.6 | npm / Deno | **mini** (default) + **full** wasm entries ([`crates/threers-js`](../crates/threers-js)) |
| `threers-node` | 0.0.6 | npm | per-arch `.node` via `@threers/node-*` ([`crates/threers-node`](../crates/threers-node)) |
| `threers` (PyPI) | 0.0.6 | PyPI | per-platform wheels ([`crates/threers-py`](../crates/threers-py)) |

`crates/` also holds seven demos and benches. None of them publish
(`publish = false`), and they fall into two groups:

- **In the workspace and building**: `threers-ocean`, `threers-animation`,
  `threers-robot-arm`, `threers-physics-bench`, `threers-connectome`.
- **`exclude`d**: `threers-continuum` and `threers-mechanism-tour`. Both build as
  libraries, but their own tests and examples call an older shape of their API —
  builder methods that became fields, argument lists a version behind. They are
  out of the workspace so `--all-targets` stays clean; reconcile them and move
  them back into `members`.

## The order is not optional

Both companions depend on `threers` by **path and version**:

```toml
# crates/threers-physics/Cargo.toml
threers = { path = "../..", version = "0.0.6", features = ["mesh-bvh"] }
```

A path dependency is what the workspace builds against; the `version` is what the
published crate carries. So `threers` has to be on crates.io *before* either
companion can even be packaged — until it is, you get

```text
failed to select a version for the requirement `threers = "^0.0.5"`
candidate versions found which didn't match: 0.0.4, 0.0.3, 0.0.2, 0.0.1
```

which is not a fault, it is the ordering telling you about itself.

## Checklist

```bash
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

# 5. What would actually ship.
cargo package -p threers --list | wc -l
cargo package -p threers                     # size, and it must compile from the tarball

# 6. Then, in this order.
cargo publish -p threers
cargo publish -p threers-physics             # only after the first is on the index
cargo publish -p threers-probe               # likewise
git tag -a v0.0.6 -m 'threers 0.0.6' && git push --tags

# 7. Language packages — one script locally; CI on tag push.
./scripts/release-versions.sh                    # versions aligned?
./scripts/release-all.sh pack                    # build + npm pack + wheels
PUBLISH=1 ./scripts/release-all.sh publish     # upload (or push tag → CI)

# Multi-arch npm-node + PyPI: push v* tag → .github/workflows/release-language-packages.yml
git tag -a v0.0.6 -m 'threers 0.0.6' && git push --tags
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
git tag v0.0.5 && git push origin v0.0.5
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

It reads 0.0.4, so 0.0.5 is what ships. `threers-physics` and `threers-probe`
have never been published, so their 0.0.1 is a first release for each — check
the name is not taken before counting on it.
