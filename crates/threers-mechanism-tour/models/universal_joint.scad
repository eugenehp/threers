// A Hooke joint: constant in, and emphatically not constant out.
//
// Two shafts at an angle, a cross between them, and a yoke on each shaft holding
// one of the cross's two pins. It is the only mechanism in the tour that is not
// planar — every other closed loop here has all its axes parallel, and this one
// cannot, because the whole point of it is that the two shafts are not.
//
// Turn the input at a constant rate and the output does *not* turn at a constant
// rate. It runs ahead, falls behind, runs ahead and falls behind again, twice per
// revolution, and the relation is exact:
//
//     tan θ_out = cos β · tan θ_in
//
// which makes the output's speed swing between cos β and 1/cos β of the input's.
// At the 30° drawn here that is 0.866 to 1.155 — ±15%, from a joint whose entire
// job is to transmit rotation unchanged. Nobody expects it, every driveshaft has
// it, and the cure is the disease applied twice: a second joint, phased a quarter
// turn against the first and at the same angle, has an error equal and opposite,
// which is why a propshaft has a U-joint at each end rather than one in the
// middle.
//
// Nothing in this file mentions any of that. It is four parts and four joints,
// and the wobble is what the geometry leaves them.
//
// 30° is also about the practical limit, and for a reason you can see: the two
// yokes are always exactly a quarter turn apart — that is what the cross is
// for — and it is their arms passing each other that runs out of room first.
//
// ---------------------------------------------------------------------------
// It is drawn about the origin, and that is not a stylistic choice — it is the
// only model in the tour that has to be.
//
// A spherical four-bar is over-constrained: four hinges, three moving bodies,
// 6·3 − 5·4 = −2 by the spatial count, exactly as the planar four-bar is
// over-constrained at −2. Both move perfectly well, because the extra
// constraints are redundant rather than contradictory. The difference is *what*
// makes them redundant. For the planar loop it is that the axes are parallel,
// which stays true wherever the model is drawn. For this one it is that all four
// axes meet at a single point, and that is a knife edge: miss it and the four
// constraints stop being redundant and start being inconsistent, and the solver
// pulls the loop apart trying to satisfy all of them at once.
//
// Drawn with its centre at [70, 0, 70] this model blew up after 36 frames — and
// more solver substeps made it blow up *sooner*, which is the signature of
// exactly that. It was converging harder onto something that cannot be
// satisfied. The anchors are single precision, so a hundred units out the four
// axes miss each other by something like 1e-5, and that was enough. At the origin
// they meet, and it runs.
//
// Which is worth knowing generally: an over-constrained loop is only as good as
// the arithmetic that makes its constraints redundant.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque, speed) { }

STEEL = 0.0078;         // g/mm³

BETA  = 30;             // the angle between the shafts
YOKE  = 18;             // half the span between a yoke's two arms
ARM   = 6;              // and how thick an arm is
SH    = 8;              // shaft radius
PR    = 4;              // the cross's pins
REACH = 26;             // how far a yoke's bar stands off the joint centre
BEARING = 48;           // and how far along each shaft its ground bearing sits

BED = -52;              // the top of the bed, well under everything

// The input's pin lies along y — across the plane the two shafts make — and the
// output's lies in that plane. Perpendicular to each other and each
// perpendicular to its own shaft, which is the whole definition of the cross.
IN_AXIS   = [0, 1, 0];
OUT_AXIS  = [sin(BETA), 0, cos(BETA)];
OUT_SHAFT = [cos(BETA), 0, -sin(BETA)];
OUT_AT    = [BEARING * OUT_SHAFT[0], 0, BEARING * OUT_SHAFT[2]];

// ---- one yoke, drawn once --------------------------------------------------
// Canonically: the shaft runs along +x, the pin it holds runs along +z, and the
// joint centre is the origin. Both yokes are this, placed differently — which is
// the only honest way to draw them, since a Hooke joint's two halves *are* the
// same part turned.
//
// No two of its solids share a face. That matters more here than anywhere else
// in the tour: the output's copy is rotated 30°, so what were exact coincidences
// become *near* coincidences in irrational coordinates, which is the worst thing
// an exact kernel can be handed. Butted rather than overlapped, this fork cost
// 844 triangles as the input and 18,942 as the output — 4.4 seconds for one part.
module fork(reach, shaft_len) {
    translate([reach, 0, 0]) rotate([0, 90, 0])
        cylinder(h = shaft_len, r = SH, $fn = 28);
    translate([reach - 8, -7, -YOKE - ARM]) cube([9, 14, 2 * (YOKE + ARM)]);
    for (s = [-1, 1]) {
        translate([-2, -5, s * YOKE - ARM / 2]) cube([reach - 2, 10, ARM]);
        translate([0, 0, s * YOKE - ARM / 2 - 0.5])
            cylinder(h = ARM + 1, r = 9, $fn = 24);
    }
}

// A housing on a pillar, standing on the bed and reaching up to `at`.
module pedestal(at) {
    translate([at[0] - 6, -6, BED - 1]) cube([12, 12, at[2] - BED - 8]);
    translate([at[0] - 11, -13, at[2] - 11]) cube([22, 26, 22]);
}

// ---- the frame -------------------------------------------------------------
// A bed and two housings, one at the far end of each shaft. They are at the ends
// on purpose: the joint itself is the subject, and nothing structural belongs
// between it and the camera, which sits at -y.
part("frame", fixed = true)
    color("slategray") {
        translate([-70, -30, BED - 10]) cube([135, 60, 11]);
        color("dimgray") {
            pedestal([-BEARING, 0, 0]);
            pedestal(OUT_AT);
        }
    }

// ---- the input shaft and its yoke ------------------------------------------
// `rotate([-90, 180, 0])` turns the canonical fork round to face the other way
// and stands its pin along y: the shaft runs back towards -x, and the arms
// straddle the cross at |y| >= 15 while the output's straddle it in the plane at
// right angles. That is how two yokes a quarter turn apart share one centre
// without ever being in the same place.
//
// `collider = "decompose"` is not optional here, and this is the model that
// shows why. A yoke is a *fork*, and a moving part's collider is its convex hull
// by default — which fills the fork's gap back in with the solid it was cut out
// of. Two forks interlocked at right angles both have hulls containing the joint
// centre, so the check reported "input into output by 20.928" before anything had
// moved, and the solver spent the run shoving apart two blobs that in the machine
// never touch. Split into convex pieces, each yoke keeps the gap that lets the
// other one through — at about a voxel of accuracy, which is why the arms are
// given 10 mm of clearance rather than 1.
part("input", density = STEEL, collider = "decompose")
    color("indianred") rotate([-90, 180, 0]) fork(REACH, 32);
hinge("input_bearing", parts = ["input", "frame"],
      at = [-BEARING, 0, 0], axis = [1, 0, 0]);

// ---- the cross -------------------------------------------------------------
// Two pins through one centre, at right angles. It is the smallest part in the
// tour and the only reason the other three can be where they are.
part("cross", density = STEEL)
    color("goldenrod") {
        rotate([-90, 0, 0]) cylinder(h = 48, r = PR, $fn = 20, center = true);
        rotate([0, BETA, 0]) cylinder(h = 48, r = PR, $fn = 20, center = true);
    }
hinge("cross_in", parts = ["cross", "input"], at = [0, 0, 0], axis = IN_AXIS);

// ---- the output shaft and its yoke -----------------------------------------
// The same fork, turned to sit on the other pin: `rotate([0, BETA, 0])` takes x
// to the output shaft's direction and z to the pin's, which is exactly the
// relationship the two have to be in.
part("output", density = STEEL, collider = "decompose")
    color("steelblue") rotate([0, BETA, 0]) fork(REACH, 32);
hinge("cross_out", parts = ["output", "cross"], at = [0, 0, 0], axis = OUT_AXIS);
hinge("output_bearing", parts = ["output", "frame"], at = OUT_AT, axis = OUT_SHAFT);

// One input, turning steadily. Everything after it is what the geometry allows.
drive("input_bearing", speed = 180, torque = 8000000);
