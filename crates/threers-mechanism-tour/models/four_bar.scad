// A four-bar crank-rocker: the linkage behind every windscreen wiper.
//
// Four hinges in a *closed* loop, which is the thing a chain of joints cannot
// do. The crank goes all the way round; the rocker cannot, and swings back and
// forth between two limits nobody wrote down. Where it turns round is decided
// by the four lengths, and by nothing else.
//
// Grashof's condition says which of the four can rotate fully: shortest plus
// longest must be no more than the other two. Here 15 + 60 = 75 against
// 55 + 40 = 95, and the shortest link is the one next to the ground, which
// makes it a crank-rocker rather than a drag-link or a double-rocker.
//
// The mobility count comes out negative. A planar loop counted in three
// dimensions always does — the count assumes the four hinge constraints are
// independent and, being parallel, they are not. It is over-constrained on
// paper and moves perfectly well.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque, speed) { }

STEEL = 0.0078;
PIN   = [0, 1, 0];        // every hinge turns about y, so the linkage is planar

// The four pin centres, in the plane the linkage works in.
A = [20,  0, 26];         // crank on ground
D = [80,  0, 26];         // rocker on ground        — ground link, 60
B = [35,  0, 26];         // crank to coupler        — crank,       15
C = [73.333, 0, 65.443];  // coupler to rocker       — coupler, 55; rocker, 40

// A link is the convex hull of its two bearings, which is both what one looks
// like and exactly what its collider will be.
module bar(p, q, r = 6, t = 7) {
    hull() {
        translate(p) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 28, center = true);
        translate(q) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 28, center = true);
    }
}

// The ground link is a real part like the others — it is the one that is fixed,
// which is all that makes it "the frame". Set behind the linkage in y so the
// bars swing past it.
part("frame", fixed = true)
    color("slategray") translate([0, -16, 0]) {
        cube([100, 8, 12]);
        translate([A[0] - 7, 0, 0]) cube([14, 8, A[2]]);
        translate([D[0] - 7, 0, 0]) cube([14, 8, D[2]]);
    }

part("crank", density = STEEL)   color("indianred") bar(A, B, 7);
part("coupler", density = STEEL) color("goldenrod") bar(B, C, 5);
part("rocker", density = STEEL)  color("steelblue") bar(D, C, 6);

hinge("crank_pin",  parts = ["crank",   "frame"],   at = A, axis = PIN);
hinge("coupler_pin", parts = ["coupler", "crank"],  at = B, axis = PIN);
hinge("rocker_pin",  parts = ["rocker",  "coupler"], at = C, axis = PIN);
hinge("rocker_ground", parts = ["rocker", "frame"], at = D, axis = PIN);

// One input. Everything else follows from the geometry.
drive("crank_pin", speed = 150, torque = 200000);
