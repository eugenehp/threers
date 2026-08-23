// A leadscrew stage, driven past both of its end stops.
//
// One mate does the whole job. A screw() holds the carriage on the rail *and*
// ties turning to advancing, so driving its travel turns it — and when the
// carriage runs into a stop, the thread cannot advance, so it cannot turn
// either. The motor stalls because the geometry says so.
//
// The stops are the point of this model. They are part of the rail, which is
// how anybody would draw them and how a mate quietly loses them: two parts a
// mate joins do not also touch, because a hinge lives inside both the door and
// the frame and leaving that contact on makes the contact and the constraint
// fight. The carriage is mated to the very part carrying its stops, so by
// default it slides straight through them — in the interference check and the
// travel sweep as well as in the simulation, silently, all three agreeing with
// each other and all three wrong about the machine.
//
// `collide = true` is the answer, and it is worth knowing the alternative: put
// the stops on a part of their own, which is what the first bench does with its
// lid and its stop.
//
// Asked for 60 mm it reaches 40, and asked for -40 it reaches -14. Neither
// number is written down anywhere. Both are where the metal is.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module screw(name, parts, at, axis, pitch, range, collide) { }
module drive(name, to, over, at, torque, speed) { }

STEEL   = 0.0078;
PLASTIC = 0.0012;

BED   = 10;              // top of the rail bed
RIDE  = 11;              // and the underside of the carriage, a hair above it
STOP_A = 26;             // the near stop's inner face
STOP_B = 110;            // the far one's

part("frame", fixed = true)
    color("slategray") {
        translate([0, -20, 0]) cube([140, 40, BED]);
        // Overlapping the bed rather than sitting exactly on it: two solids
        // with exactly coplanar touching faces are the worst case an exact
        // boolean kernel has, and it is free to avoid.
        translate([STOP_A - 6, -20, BED - 2]) cube([6, 40, 22]);
        translate([STOP_B, -20, BED - 2]) cube([6, 40, 22]);
        // Two gibs down the length of the bed, half a millimetre off the
        // carriage either side. Flat ways and a pair of gibs: the bed takes the
        // weight, the gibs take everything sideways, and gravity closes the
        // joint. That is what a machine slide is, and without them a slider()
        // naming this part was the only thing keeping the carriage on its line.
        //
        // It is also the *one* shape of guide this model can have. Every other
        // carriage in the tour wraps its rail — over the head, under it, either
        // side — because capture is what a prismatic joint wants. Capture needs
        // a concave carriage, and a moving part's collider is its convex hull,
        // which fills the channel back in. Everywhere else that costs nothing,
        // because the pair is mated and never collides. Here it is mated *and*
        // colliding, and a wrapped carriage would spend the whole run trying to
        // push its own hull out of the rail it is threaded on.
        color("dimgray") {
            translate([0, -20, BED - 2]) cube([140, 4.5, 8]);
            translate([0, 15.5, BED - 2]) cube([140, 4.5, 8]);
        }
    }

part("carriage", density = PLASTIC, friction = 0.1)
    color("steelblue") translate([40, -15, RIDE]) cube([30, 30, 18]);

// One mate: a helix. `collide = true` is what lets it notice the stops that are
// drawn on the very part it is mated to.
screw("lead", parts = ["carriage", "frame"], at = [40, 0, RIDE],
      axis = [1, 0, 0], pitch = 2.5, range = [-40, 60], collide = true);

// Out until it runs out of rail, then back until it runs out the other way.
drive("lead", to = 60,  over = 2.0, at = 0.3, torque = 900000);
drive("lead", to = -40, over = 2.0, at = 3.0, torque = 900000);
