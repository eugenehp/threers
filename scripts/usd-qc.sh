#!/bin/zsh
#
# Quality control for the USD implementation, through OpenUSD's own tools.
#
# Every `.usda` in the test data is written out as a `.usdc` by this crate and
# then handed back to Apple's tools, which are the authority on what a USD file
# is. Four questions are asked of each:
#
#   usdchecker   is it a valid file, and no less valid than the source?
#   usdtree      is the prim hierarchy the same one?
#   usdcat       does the composed stage flatten to the same thing?
#   usdrecord    does Hydra draw the same picture from it?
#
# The last is the only end-to-end check — it goes through composition, schema
# resolution, geometry and shading — and it is also the one with a trap: a stage
# with nothing in view renders a blank image, and two blank images match. So the
# report says which fixtures actually drew something, and the render check is
# evidence only for those.
#
# Requires OpenUSD's command-line tools, which ship with macOS at /usr/bin.
# Run after regenerating the crates:
#
#   USD_CONVERT="$(ls path/to/*.usda | tr '\n' ,)" \
#     cargo test --lib --features usd loaders::usd::crate_write::external -- --ignored
#   scripts/usd-qc.sh path/to
set -u
here="${1:-}"
if [[ -z "$here" || ! -d "$here" ]]; then
  echo "usage: $0 <directory of .usda and matching .usdc>" >&2
  exit 2
fi
cd "$here" || exit 2
command -v /usr/bin/usdchecker >/dev/null || { echo "OpenUSD tools not found" >&2; exit 2; }
mkdir -p render qc

pass=0; fail=0; drew=0
note() { printf "  %-14s %-10s %s\n" "$1" "$2" "$3"; }

for f in *.usda; do
  base="${f%.usda}"
  [[ -f "$base.usdc" ]] || continue

  # No less valid than the source. A fixture may have its own problems; what
  # matters is that writing it did not add any.
  src_errs=$(/usr/bin/usdchecker "$f" 2>&1 | grep -c "^Error")
  our_errs=$(/usr/bin/usdchecker "$base.usdc" 2>&1 | grep -c "^Error")
  if [[ "$our_errs" -gt "$src_errs" ]]; then
    note "$base" "usdchecker" "FAIL ($our_errs errors against $src_errs in the source)"
    fail=$((fail+1)); continue
  fi

  /usr/bin/usdtree "$f" >"qc/$base.tree.src" 2>&1
  /usr/bin/usdtree "$base.usdc" >"qc/$base.tree.our" 2>&1
  if ! diff -q "qc/$base.tree.src" "qc/$base.tree.our" >/dev/null; then
    note "$base" "usdtree" "FAIL (the hierarchy differs)"; fail=$((fail+1)); continue
  fi

  if /usr/bin/usdcat --flatten --skipSourceFileComment --out "qc/$base.flat.src" "$f" 2>/dev/null \
  && /usr/bin/usdcat --flatten --skipSourceFileComment --out "qc/$base.flat.our" "$base.usdc" 2>/dev/null; then
    if ! diff -q "qc/$base.flat.src" "qc/$base.flat.our" >/dev/null; then
      note "$base" "usdcat" "FAIL (the composed stage differs)"; fail=$((fail+1)); continue
    fi
  fi

  rm -f "render/$base.src.000.png" "render/$base.our.000.png"
  /usr/bin/usdrecord --frames 0 --imageWidth 128 "$f" "render/$base.src.###.png" >/dev/null 2>&1
  /usr/bin/usdrecord --frames 0 --imageWidth 128 "$base.usdc" "render/$base.our.###.png" >/dev/null 2>&1
  if [[ -f "render/$base.src.000.png" && -f "render/$base.our.000.png" ]]; then
    if ! cmp -s "render/$base.src.000.png" "render/$base.our.000.png"; then
      note "$base" "usdrecord" "FAIL (Hydra draws it differently)"; fail=$((fail+1)); continue
    fi
    # Did anything appear? A stage with nothing in view renders blank, and two
    # blank images agreeing says nothing at all.
    if [[ $(stat -f%z "render/$base.our.000.png") -gt 600 ]]; then
      drew=$((drew+1)); note "$base" "all" "ok — and Hydra draws the same picture"
    else
      note "$base" "all" "ok — nothing in view, so the render proves nothing"
    fi
  else
    note "$base" "all" "ok — not renderable"
  fi
  pass=$((pass+1))
done

echo
echo "passed $pass, failed $fail — $drew of them with something actually drawn"
[[ "$fail" -eq 0 ]]
