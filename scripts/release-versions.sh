#!/usr/bin/env bash
# Print versions across publish roots — bump these together before a release.
#
# Every crate, package and version pin in the repo carries one version, set in
# `[workspace.package]` in the root Cargo.toml.
# `CHECK=1 ./scripts/release-versions.sh` exits non-zero when any of them has
# drifted from it.
# With CHECK=1 it also asks crates.io, and fails if any published crate is
# behind the index, or if every one of them is already there at its version.
#
# `./scripts/release-versions.sh bump 0.0.13` moves every one of them at once:
# the manifests and the pins between crates, the npm and PyPI packages and
# their lockfiles, the workspace's own entries in Cargo.lock, the install
# snippets in the READMEs and docs, and the CHANGELOG. If the current version
# never shipped (no tag and not on crates.io), its CHANGELOG section is renamed
# to the new one; otherwise `[Unreleased]` becomes it. Then it prints the
# report, so what it did is visible before anything is committed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${1:-}" == "bump" ]]; then
  NEW="${2:-}"
  if [[ ! "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: $0 bump X.Y.Z" >&2
    exit 1
  fi
  python3 - "$ROOT" "$NEW" <<'PY'
import datetime, json, pathlib, re, subprocess, sys, tomllib, urllib.error, urllib.request

root, new = pathlib.Path(sys.argv[1]), sys.argv[2]
old = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
key = lambda v: tuple(map(int, v.split(".")))
if key(new) <= key(old):
    sys.exit(f"{new} is not ahead of the current {old}")
changed = []

# CHANGELOG first, because it is the one step that can refuse, and a refusal
# has to come before anything is written. A version that never went out has
# nothing to preserve, so its section simply takes the new number; one that
# did keeps its section and `[Unreleased]` becomes the new one.
def shipped(v):
    tags = subprocess.run(["git", "-C", str(root), "tag", "--list", f"v{v}"], capture_output=True, text=True).stdout
    if tags.strip():
        return True
    req = urllib.request.Request(f"https://crates.io/api/v1/crates/threers/{v}", headers={"User-Agent": "threers-release"})
    try:
        urllib.request.urlopen(req, timeout=20)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise SystemExit(f"crates.io answered {e.code}; cannot tell whether {v} shipped")
    except OSError as e:
        raise SystemExit(f"could not reach crates.io ({e}); cannot tell whether {v} shipped")

cl = root / "CHANGELOG.md"
changelog = cl.read_text()
today = datetime.date.today().isoformat()
heading = re.search(r'(?m)^## \[(\d+\.\d+\.\d+)\][^\n]*$', changelog)
url = "https://github.com/eugenehp/threers/compare"
if heading and heading.group(1) == old and not shipped(old):
    changelog = changelog.replace(heading.group(0), f"## [{new}] — {today}", 1)
    changelog = changelog.replace(f"[Unreleased]: {url}/v{old}...HEAD", f"[Unreleased]: {url}/v{new}...HEAD")
    changelog = re.sub(rf'(?m)^\[{re.escape(old)}\]: ({re.escape(url)}/v[\d.]+)\.\.\.v{re.escape(old)}$',
                  rf'[{new}]: \g<1>...v{new}', changelog)
    note = f"CHANGELOG: {old} never shipped, so its section is now {new}"
else:
    unreleased = re.search(r'(?ms)^## \[Unreleased\]\n(.*?)(?=^## \[)', changelog)
    if not unreleased or not unreleased.group(1).strip():
        sys.exit(f"{old} shipped and [Unreleased] is empty: write the changelog before bumping")
    changelog = changelog.replace("## [Unreleased]\n", f"## [Unreleased]\n\n## [{new}] — {today}\n", 1)
    changelog = changelog.replace(f"[Unreleased]: {url}/v{old}...HEAD",
                        f"[Unreleased]: {url}/v{new}...HEAD\n[{new}]: {url}/v{old}...v{new}")
    note = f"CHANGELOG: [Unreleased] is now {new}"

def rewrite(path, text):
    if text != path.read_text():
        path.write_text(text)
        changed.append(str(path.relative_to(root)))

# Cargo manifests: a package's own `version = "OLD"` and the version of every
# `threers*` dependency pin. Nothing else in them is touched.
q = re.escape(f'"{old}"')
for f in [root / "Cargo.toml", *sorted(root.glob("crates/*/Cargo.toml"))]:
    text = re.sub(rf'(?m)^(version\s*=\s*){q}', rf'\g<1>"{new}"', f.read_text())
    text = re.sub(rf'(?m)^(threers[\w-]*\s*=\s*\{{[^\n}}]*\bversion\s*=\s*){q}', rf'\g<1>"{new}"', text)
    rewrite(f, text)

# npm and Deno: parsed, so a third-party package that happens to share the
# version number is never caught. Only files that write back byte for byte are
# edited; one that does not is reported rather than reformatted.
json_files = [root / "crates/threers-js/package.json", root / "crates/threers-js/deno.json",
              root / "crates/threers-node/package.json", root / "crates/threers-node/package-lock.json",
              *sorted(root.glob("crates/threers-node/npm/*/package.json")),
              root / "web/package.json", root / "web/package-lock.json"]
for f in json_files:
    if not f.exists():
        continue
    raw = f.read_text()
    data = json.loads(raw)
    dump = lambda d: json.dumps(d, indent=2, ensure_ascii=False) + "\n"
    if dump(data) != raw:
        sys.exit(f"{f.relative_to(root)} would be reformatted by a JSON round trip; bump it by hand")
    def bump_json(d):
        if d.get("version") == old:
            d["version"] = new
        for deps in ("dependencies", "optionalDependencies", "devDependencies"):
            for name, v in d.get(deps, {}).items():
                if name.startswith("@threers/") and v == old:
                    d[deps][name] = new
    bump_json(data)
    for name, pkg in data.get("packages", {}).items():
        if name == "" or name.startswith("node_modules/@threers/"):
            bump_json(pkg)
    rewrite(f, dump(data))

py = root / "crates/threers-py/pyproject.toml"
rewrite(py, re.sub(rf'(?m)^(version\s*=\s*){q}', rf'\g<1>"{new}"', py.read_text()))

# Cargo.lock: the workspace's own packages, which carry no `source` line.
# Editing it here rather than letting cargo regenerate it keeps every other
# entry exactly as it was.
lock = root / "Cargo.lock"
rewrite(lock, re.sub(rf'(name = "threers[\w-]*"\nversion = ){q}(?!\nsource)', rf'\g<1>"{new}"', lock.read_text()))

# Install snippets in the READMEs and docs: `threers-x = "OLD"`,
# `threers-x = {{ version = "OLD", ... }}` and `threers@OLD` on a CDN. The
# releasing guide also shows the tag and a pin at the current version.
docs = [root / "README.md", *sorted(root.glob("crates/*/README.md")), *sorted(root.glob("docs/*.md"))]
for f in docs:
    if not f.exists():
        continue
    text = f.read_text()
    text = re.sub(rf'(threers[\w-]*\s*=\s*){q}', rf'\g<1>"{new}"', text)
    text = re.sub(rf'(threers[\w-]*\s*=\s*\{{[^\n}}]*\bversion\s*=\s*){q}', rf'\g<1>"{new}"', text)
    text = re.sub(rf'(threers@){re.escape(old)}\b', rf'\g<1>{new}', text)
    if f.name == "releasing.md":
        text = re.sub(rf'\bv{re.escape(old)}\b', f"v{new}", text)
        text = re.sub(rf"'threers {re.escape(old)}'", f"'threers {new}'", text)
        text = re.sub(rf'\| {re.escape(old)} \|', f"| {new} |", text)
    rewrite(f, text)

rewrite(cl, changelog)

print(f"bumped {old} -> {new} in {len(changed)} files")
for c in changed:
    print("  " + c)
print(note)
PY
  printf "\n"
fi

# A crate's version, following `version.workspace = true` to the root.
v_cargo() {
  python3 - "$1" "$ROOT/Cargo.toml" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
v = data["package"]["version"]
if isinstance(v, dict):
    v = tomllib.loads(pathlib.Path(sys.argv[2]).read_text())["workspace"]["package"]["version"]
print(v)
PY
}

v_json() {
  python3 - "$1" <<'PY'
import sys, json, pathlib
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["version"])
PY
}

v_pyproject() {
  python3 - "$1" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
print(data["project"]["version"])
PY
}

CRATE=$(v_cargo "$ROOT/Cargo.toml")
drift=0

# Roots that track the crate version.
report() { # label version
  printf "%-32s %s" "$1" "$2"
  if [[ "$2" != "$CRATE" ]]; then
    printf "   <- expected %s" "$CRATE"
    drift=1
  fi
  printf "\n"
}

report "crates.io threers"           "$CRATE"
report "npm @threers/threers (ESM)"  "$(v_json "$ROOT/crates/threers-js/package.json")"
report "jsr @threers/threers (deno)"  "$(v_json "$ROOT/crates/threers-js/deno.json")"
report "npm threers-node"            "$(v_json "$ROOT/crates/threers-node/package.json")"
report "npm threers-node (lock)"     "$(v_json "$ROOT/crates/threers-node/package-lock.json")"
report "PyPI threers"                "$(v_pyproject "$ROOT/crates/threers-py/pyproject.toml")"
report "web shim (package.json)"     "$(v_json "$ROOT/web/package.json")"
report "web shim (lock)"             "$(v_json "$ROOT/web/package-lock.json")"

for f in "$ROOT/crates/threers-node/npm"/*/package.json; do
  [[ -f "$f" ]] || continue
  pkg=$(python3 -c "import json; print(json.load(open('$f'))['name'])")
  report "$pkg" "$(v_json "$f")"
done

# Every crate in crates/, workspace member or not.
printf "\ncrates:\n"
for f in "$ROOT"/crates/*/Cargo.toml; do
  d=${f%/Cargo.toml}
  report "  ${d#"$ROOT/crates/"}" "$(v_cargo "$f")"
done

# The crates pin each other by version as well as by path, and cargo refuses to
# resolve a path dependency whose version does not satisfy the requirement.
printf "\nversion pins between crates:\n"
while read -r where pinned; do
  report "  $where" "$pinned"
done < <(python3 - "$ROOT" <<'PY'
import sys, tomllib, pathlib
root = pathlib.Path(sys.argv[1])
for f in [root / "Cargo.toml", *sorted(root.glob("crates/*/Cargo.toml"))]:
    data = tomllib.loads(f.read_text())
    tables = [data.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
    for t in data.get("target", {}).values():
        tables += [t.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
    for t in tables:
        for name, spec in t.items():
            if name.startswith("threers") and isinstance(spec, dict) and "version" in spec:
                print(f"{f.parent.name}->{name}", spec["version"])
PY
)

# What the READMEs and docs tell people to put in their Cargo.toml or load from
# a CDN. These are what a reader copies, so they follow the release too.
printf "\ninstall snippets in the docs:\n"
while read -r where pinned; do
  report "  $where" "$pinned"
done < <(python3 - "$ROOT" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
pattern = re.compile(r'threers[\w-]*\s*=\s*(?:\{[^\n}]*\bversion\s*=\s*)?"(\d+\.\d+\.\d+)"|threers@(\d+\.\d+\.\d+)\b')
for f in [root / "README.md", *sorted(root.glob("crates/*/README.md")), *sorted(root.glob("docs/*.md"))]:
    if not f.exists():
        continue
    for n, line in enumerate(f.read_text().splitlines(), 1):
        # A path dependency in a repo-internal example is a pin, not a snippet;
        # the pins between crates are checked above.
        if "path =" in line:
            continue
        for m in pattern.finditer(line):
            print(f"{f.relative_to(root)}:{n}", m.group(1) or m.group(2))
PY
)

# `cargo publish` cannot replace a version, so a version that is already on the
# index fails at the upload and not before. Under CHECK=1, ask crates.io about
# every crate the release publishes. A version behind the index is an error. One
# equal to it was already published and release-all.sh skips it, so a publish
# that stopped halfway can be rerun; but if all of them are equal, nothing would
# go out, which almost always means the bump was forgotten.
if [[ "${CHECK:-0}" == "1" ]]; then
  printf "\ncrates.io index:\n"
  pending=0 checked=0
  # The set comes from the workspace (`publish` not false), as in release-all.sh.
  for pair in $(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" \
      | python3 -c '
import json, sys
for p in json.load(sys.stdin)["packages"]:
    if p.get("publish") != []:
        print(p["name"] + ":" + p["version"])'); do
    name=${pair%%:*} local_v=${pair#*:}
    published=$(curl -sf -H 'User-Agent: threers-release' \
        "https://crates.io/api/v1/crates/$name" 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['crate']['max_version'])" \
        2>/dev/null) || published=""
    printf "%-32s %-8s local %s" "  $name published" "${published:-none}" "$local_v"
    if [[ -z "$published" ]]; then
      printf "   (not on crates.io, or could not reach it)"
      pending=1
    else
      checked=1
      cmp=$(python3 -c "import sys; t=lambda v: tuple(map(int, v.split('.'))); a,b=t(sys.argv[1]),t(sys.argv[2]); print((a>b)-(a<b))" \
          "$local_v" "$published")
      case "$cmp" in
        1)  pending=1 ;;
        0)  printf "   (already published, will be skipped)" ;;
        *)  printf "   <- %s is behind it" "$local_v"; drift=1 ;;
      esac
    fi
    printf "\n"
  done
  # Only fatal when this release publishes crates (release-all.sh sets
  # CHECK_CRATES); an npm- or PyPI-only release leaves the index alone.
  if (( checked && ! pending )); then
    printf "  every crate is already on crates.io at its local version — bump first\n"
    [[ "${CHECK_CRATES:-1}" == "1" ]] && drift=1
  fi
fi

# The newest changelog entry should name the crate version.
latest=$(grep -m1 -o '^## \[[0-9][^]]*\]' "$ROOT/CHANGELOG.md" | tr -d '#[] ')
printf "\n%-32s %s" "CHANGELOG newest entry" "$latest"
if [[ "$latest" != "$CRATE" ]]; then
  printf "   <- expected %s" "$CRATE"
  drift=1
fi
printf "\n"

if (( drift )); then
  printf "\nversions have drifted\n" >&2
  [[ "${CHECK:-0}" == "1" ]] && exit 1
fi
exit 0
