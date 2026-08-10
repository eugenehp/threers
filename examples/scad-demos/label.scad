// text() with the bundled DejaVu Sans outline font: smooth, watertight glyphs
// that render correctly from any angle. Pass font="…ttf" to use another typeface
// (e.g. OpenSCAD's Liberation Sans) — the geometry pipeline is font-agnostic.
linear_extrude(2) text("R3", size = 12, halign = "center", valign = "center");
