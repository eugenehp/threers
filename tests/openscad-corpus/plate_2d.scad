// Corpus model: a plate with a round through-hole, built via the 2D subsystem
// (square − circle, then linear-extruded). Exercises the parser's 2D primitives,
// the 2D `difference` → hole path, and the hole-bridging cap triangulation — the
// result is a closed manifold built directly, with no CSG differencing.
linear_extrude(10) difference() {
    square([10, 10]);
    translate([5, 5]) circle(2, $fn = 32);
}
