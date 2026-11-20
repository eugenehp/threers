#!/usr/bin/env python3
"""WinAnsi (codes 32-255) advance widths of the PDF standard fonts, from the
URW base-35 AFMs that ship with Ghostscript (metric-compatible with Adobe's),
as src/exporters/pdf_fonts.rs.

    scripts/afm_widths.py /opt/homebrew/share/ghostscript/fonts > src/exporters/pdf_fonts.rs
"""
import sys, os
d = sys.argv[1]
ascii_names = ("space exclam quotedbl numbersign dollar percent ampersand quotesingle parenleft parenright asterisk plus comma hyphen period slash "
  "zero one two three four five six seven eight nine colon semicolon less equal greater question at "
  + " ".join(chr(c) for c in range(65, 91)) + " bracketleft backslash bracketright asciicircum underscore grave "
  + " ".join(chr(c) for c in range(97, 123)) + " braceleft bar braceright asciitilde").split()
assert len(ascii_names) == 95
win = {128:'Euro',130:'quotesinglbase',131:'florin',132:'quotedblbase',133:'ellipsis',134:'dagger',135:'daggerdbl',136:'circumflex',
  137:'perthousand',138:'Scaron',139:'guilsinglleft',140:'OE',142:'Zcaron',145:'quoteleft',146:'quoteright',147:'quotedblleft',
  148:'quotedblright',149:'bullet',150:'endash',151:'emdash',152:'tilde',153:'trademark',154:'scaron',155:'guilsinglright',156:'oe',
  158:'zcaron',159:'Ydieresis'}
latin1 = ("space exclamdown cent sterling currency yen brokenbar section dieresis copyright ordfeminine guillemotleft logicalnot hyphen "
  "registered macron degree plusminus twosuperior threesuperior acute mu paragraph periodcentered cedilla onesuperior ordmasculine "
  "guillemotright onequarter onehalf threequarters questiondown Agrave Aacute Acircumflex Atilde Adieresis Aring AE Ccedilla Egrave "
  "Eacute Ecircumflex Edieresis Igrave Iacute Icircumflex Idieresis Eth Ntilde Ograve Oacute Ocircumflex Otilde Odieresis multiply "
  "Oslash Ugrave Uacute Ucircumflex Udieresis Yacute Thorn germandbls agrave aacute acircumflex atilde adieresis aring ae ccedilla "
  "egrave eacute ecircumflex edieresis igrave iacute icircumflex idieresis eth ntilde ograve oacute ocircumflex otilde odieresis "
  "divide oslash ugrave uacute ucircumflex udieresis yacute thorn ydieresis").split()
assert len(latin1) == 96
names = {32 + i: n for i, n in enumerate(ascii_names)}
names.update(win)
names.update({160 + i: n for i, n in enumerate(latin1)})
fonts = [("HELVETICA", "Helvetica", "n019003l"), ("TIMES", "Times-Roman", "n021003l"),
         ("COURIER", "Courier", "n022003l")]
print("// Generated from the URW base-35 AFMs that ship with Ghostscript 10.07")
print("// (n019003l, n021003l and n022003l.afm), whose advance widths equal those")
print("// of the Adobe core fonts they stand in for, by scripts/afm_widths.py. Do not edit.")
print()
missing_all = []
for const, base, f in fonts:
    w = {}
    for line in open(os.path.join(d, f + ".afm"), encoding="latin-1"):
        if line.startswith("C "):
            parts = dict(p.strip().split(" ", 1) for p in line.split(";") if p.strip())
            w[parts["N"]] = int(float(parts["WX"]))
    row = []
    for code in range(32, 256):
        n = names.get(code)
        if n is None:
            row.append(0)
        elif n in w:
            row.append(w[n])
        else:
            missing_all.append((base, code, n)); row.append(0)
    print(f"/// {base} ({f}.afm): WinAnsi codes 32 to 255, 1/1000 em; 0 where the code is unassigned.")
    print(f"pub(super) const {const}: [u16; 224] = [")
    for k in range(0, 224, 16):
        print("    " + ", ".join(str(x) for x in row[k:k + 16]) + ",")
    print("];")
    print()
sys.stderr.write(f"missing: {missing_all}\n")
