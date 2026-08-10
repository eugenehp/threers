// 3D minkowski rounding a (convex) box with a sphere — exact for convex operands.
minkowski() { cube([16, 10, 4], center = true); sphere(2, $fn = 16); }
