// Corpus model: two stacked unit-ish cubes sharing a face (coplanar union).
union() {
    cube(10);
    translate([0, 0, 10]) cube(10);
}
