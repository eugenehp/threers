// Rounded mounting plate with two bolt holes — 2D offset + boolean holes,
// then extruded. Circle resolution comes from $fa/$fs (no explicit $fn).
linear_extrude(3)
difference() {
    offset(r = 3) square([40, 20]);
    translate([9, 10])  circle(2.5);
    translate([31, 10]) circle(2.5);
}
