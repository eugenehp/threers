// Corpus model: a box with an overlapping box subtracted (axis-aligned,
// volume-overlapping difference — exercises coplanar handling + T-junction heal).
difference() {
    cube(10);
    translate([5, 0, 0]) cube(10);
}
