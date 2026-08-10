// offset(r): round a rectangle's corners, then extrude (2D subsystem end-to-end).
linear_extrude(4) offset(r = 2, $fn = 32) square([20, 10]);
