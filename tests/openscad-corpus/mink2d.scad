// Non-convex 2D Minkowski: an L-shape rounded by a circle, then extruded.
linear_extrude(2) minkowski() {
    polygon([[0,0],[10,0],[10,4],[4,4],[4,10],[0,10]]);
    circle(2, $fn = 32);
}
