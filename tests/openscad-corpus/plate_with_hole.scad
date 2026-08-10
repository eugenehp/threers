// Corpus model: a plate with a cylindrical through-hole.
difference() {
    cube(10);
    translate([5, 5, -1]) cylinder(h = 12, r = 2, $fn = 32);
}
