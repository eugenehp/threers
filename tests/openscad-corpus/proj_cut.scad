// projection(cut=true): the z=0 cross-section of a tilted cube, re-extruded.
linear_extrude(3) projection(cut = true) translate([0, 0, -5]) rotate([20, 0, 0]) cube(10);
