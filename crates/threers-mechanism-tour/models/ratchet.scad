// A ratchet: free one way, locked the other.
//
// No joint in this model knows about "one way". The carriage is on an ordinary
// slider and the pawl is on an ordinary hinge, and the direction comes entirely
// out of a tooth shape — a shallow ramp the pawl can ride up, and a face it
// cannot. The carriage is driven forward and goes; it is then driven back, with
// the same force and the same distance asked for, and does not move. Nothing
// had to be told to stop it.
//
// The teeth belong to the frame, which is what makes this cheap. A *fixed* part
// keeps every triangle it was drawn with, so the teeth are exactly the teeth. A
// moving toothed part could not: the convex hull of a ratchet wheel is a plain
// disc, and the teeth — the whole mechanism — are exactly the concavities a
// hull fills in. That is what `collider = "decompose"` is for, and it is worth
// knowing what it costs before reaching for it: a voxel decomposition returns a
// collider about one voxel *larger* than the part it came from. The rotary
// version of this model was drawn with the pawl 0.8 mm clear of the wheel, and
// the check said "wheel into pawl by 2.069" before anything was simulated.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module slider(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque, speed) { }

PLASTIC = 0.0012;
PIN     = [0, 1, 0];

DECK  = 10;              // top of the base, and the floor of every valley
PITCH = 12;
TALL  = 6;               // how far a tooth stands above the valley
FIRST = 9.6;             // where the toothed section starts
LEAN  = 3.05;            // how far a tooth's face leans back over its height
TEETH = 10;

// The rail the carriage runs on: a stalk down the back of the deck with a head
// on top, and the carriage wrapping that head. Behind the teeth and above them,
// so it carries the carriage without standing between the teeth and the camera.
STALK = [20, 28];        // the stalk's near and far faces in y
HEAD  = [14, 34];        // the head's, overhanging the stalk both ways
RAIL_Z = 26;             // the underside of the head

P    = [49, 0, RAIL_Z];  // the pawl's pivot, carried by the carriage
NOSE = [35, 0, 15.2];    // and its nose, sitting in a valley

// A saw tooth: a ramp up towards +x, then a face back down. Which of the two
// is the ramp is the entire mechanism.
//
// The face leans back, 3 over 6, and the teeth start where they do, for one
// reason: the nose has to be able to *fall in*. It swings on an arc about the
// pivot, and dropping from a crest it passes the tip of the tooth it is
// dropping behind. With the face straight up, that tip stood a millimetre
// inside the arc the nose sweeps, so the nose came to rest perched on it,
// lifted 14°, and the first pull back slid it a whole tooth before it caught.
// Leant back, the tip is clear of the arc and the nose lands 0.6 from the face.
// It still locks: a push on a face leaning 27° turns the pawl down onto its
// stop as long as the lean is under 37°, which is where a pawl this shape
// would start being lifted out instead.
module rack() {
    translate([0, 9, 0]) rotate([90, 0, 0])
        linear_extrude(height = 18)
            for (i = [0 : TEETH - 1])
                polygon([[FIRST + i * PITCH, DECK],
                         [FIRST + (i + 1) * PITCH - LEAN, DECK + TALL],
                         [FIRST + (i + 1) * PITCH, DECK]]);
}

// Exact, because it never moves. `collider = "mesh"` is already the default for
// a fixed part; it is written out here because it is the point.
part("frame", fixed = true, collider = "mesh", friction = 0.2)
    color("slategray") {
        translate([0, -18, 0]) cube([148, 36 + HEAD[1] - 18, DECK]);
        rack();
        // The rail: the stalk, and the head on top of it.
        translate([0, STALK[0], DECK]) cube([148, STALK[1] - STALK[0], RAIL_Z - DECK]);
        translate([0, HEAD[0], RAIL_Z]) cube([148, HEAD[1] - HEAD[0], 4]);
    }

// The block over the teeth that carries the pawl, an arm back over the rail's
// head resting on it, and a lip down the back of the head, so it wraps it.
part("carriage", density = PLASTIC, friction = 0.2)
    color("steelblue") translate([40, 0, 0]) {
        translate([0, -8, 28]) cube([36, 16, 16]);
        translate([0, 8, RAIL_Z + 4]) cube([36, HEAD[1] - 8, 6]);
        translate([0, HEAD[1], RAIL_Z - 2]) cube([36, 3, 12]);
    }
slider("feed", parts = ["carriage", "frame"], at = [40, 0, 28],
       axis = [1, 0, 0], range = [-4, 64]);

module bar(p, q, r, t) {
    hull() {
        translate(p) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 24, center = true);
        translate(q) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 24, center = true);
    }
}

// Hung off the carriage rather than off the frame, so it travels with it.
part("pawl", density = PLASTIC, friction = 0.2, bounce = 0)
    color("goldenrod") bar(P, NOSE, 3, 8);
// It may lift, and it may not drop below where it is drawn. That stop is what a
// tooth face pushes against on the way back, and it is the whole of the lock.
hinge("pawl_pivot", parts = ["pawl", "carriage"], at = P, axis = PIN,
      range = [0, 30]);
// The spring: enough to drop the nose into the next valley, not enough to stop
// a ramp lifting it out of the last one.
drive("pawl_pivot", to = 0, over = 0.2, at = 0, torque = 200000);

// Forward, then back — the same mate, the same force, the same 60 mm asked for.
drive("feed", to = 60, over = 1.8, at = 0.3, torque = 2000000);
drive("feed", to = 0,  over = 1.5, at = 2.6, torque = 2000000);
