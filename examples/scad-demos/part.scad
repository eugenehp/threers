// A rounded bar with two through-holes. The holes are subtracted in 2D (the new
// polygon-boolean kernel), then extruded — so it never touches the 3D CSG path.
use <hardware.scad>
linear_extrude(6)
difference() {
    rounded_bar(40, 12, 3);
    translate([9, 6])  hole(2);
    translate([31, 6]) hole(2);
}
