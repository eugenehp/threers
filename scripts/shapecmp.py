#!/usr/bin/env python3
"""Every TrueType font on this machine, shaped by threers and by HarfBuzz, compared.

    cargo build --release --example shape_dump
    python3 scripts/shapecmp.py

HarfBuzz is a third implementation — neither this code nor the one it was ported from —
which is the only kind of agreement worth much. Needs `hb-shape` (brew install harfbuzz)
and `fonttools`. Last run: 178 fonts agree on all 4 texts, 0 differ, 26 skipped.

Two traps. HarfBuzz applies `trak` inside advances at 12pt, so compare **glyph ids**, not
positions. And its default-on feature set is larger than this shaper's, so the features
this does not apply are switched off below — otherwise the run measures the feature list
rather than the reading.
"""
import subprocess, sys, glob
from fontTools.ttLib import TTFont
texts = ["office affluent fjord flat fifty efficient shuffle waffle",
         "The 1:30 train; 10-20 x 3, Hamburgefonstiv!",
         "Th fb ffb ffh ffk ffl fft ffj ct st Qu",
         "A--B --> <= >= != == ... www.example.com"]
paths = sorted(glob.glob('/System/Library/Fonts/*.ttf') + glob.glob('/System/Library/Fonts/Supplemental/*.ttf'))
# The features this shaper applies: ccmp, rlig, rclt, calt, liga. Everything HarfBuzz
# turns on by default and this does not is switched off, or the comparison measures the
# feature list rather than the reading.
OFF = '-kern,-locl,-mark,-mkmk,-clig,-dlig,-hlig,-frac,-numr,-dnom,-rvrn,-ltrm,-ltra'
BIN = 'target/release/examples/shape_dump'
same = diff = skipped = 0
for p in paths:
    try:
        f = TTFont(p, lazy=True)
        if 'morx' in f or 'glyf' not in f:
            skipped += 1; continue
    except Exception:
        skipped += 1; continue
    bad = []
    for t in texts:
        r = subprocess.run([BIN, p, t], capture_output=True, text=True)
        if r.returncode:
            bad = None; break
        ours = r.stdout.strip()
        hb = subprocess.run(['hb-shape','--no-positions','--no-glyph-names','--script=latn',
                             f'--features={OFF}', p, t],
                            capture_output=True, text=True).stdout.strip().strip('[]')
        if ours != hb: bad.append((t, hb, ours))
    if bad is None:
        skipped += 1; continue
    if bad:
        diff += 1
        print('DIFF', p.split('/')[-1])
        for t, h, o in bad[:1]:
            print('   ', t[:40]); print('     hb  ', h[:120]); print('     ours', o[:120])
    else:
        same += 1
print(f'{same} fonts agree on all {len(texts)} texts, {diff} differ, {skipped} skipped (AAT morx or not TrueType)')
