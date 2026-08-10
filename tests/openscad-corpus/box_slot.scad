// Corpus model: a cube with a square through-hole (a box subtracted, poking
// through in z) — the "hole" case, axis-aligned.
difference() {
    cube(10);
    translate([3, 3, -1]) cube([4, 4, 12]);
}
