// 2D union/difference/intersection combined, then extruded.
linear_extrude(4)
difference() {
    union() {
        square([30, 12], center = true);
        rotate(90) square([30, 12], center = true);   // a plus sign
    }
    circle(4);                                          // hole through the middle
}
