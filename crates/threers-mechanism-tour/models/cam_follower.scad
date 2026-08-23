// A cam and a roller follower.
//
// Nothing here declares how the follower moves. It is a part on a slider, and
// the only thing under it is the cam. The lift you see plotted is not a curve
// anybody wrote — it is what the two shapes leave room for, and it comes out of
// the contact.
//
// The follower's nose is a roller on its own pin, which is what a cam this fast
// would really have. That is not only about wear: a flat face on a curved cam
// is a bad contact to solve as well as to build. The contact point wanders
// across the face, the pair is a flat against a polygon, and a follower pressed
// onto it by a spring works its way in. Curve against curve is one point,
// wherever the cam happens to be.
//
// The cam is a disc turning about a point 8 mm off its own centre — the
// cheapest cam there is, and the only one that stays convex. A profiled cam
// would need `collider = "decompose"`, the way the ratchet wheel does.
//
// Watch the shaft torque. Lifting the follower asks for about 1.2e6; the shaft
// is given 3e7 and needs most of it, because an eccentric cam is an unbalanced
// flywheel and 178 g of steel hung 8 mm off the axis is 1.4e7 all by itself.
// Balance it with a counterweight and the motor could be a tenth of the size.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module slider(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque, speed) { }

STEEL   = 0.0078;
PLASTIC = 0.0012;
PIN     = [0, 1, 0];

R  = 20;                    // cam radius
E  = 8;                     // how far its centre is from the shaft
RR = 6;                     // roller radius
O  = [42, 0, 40];           // the shaft, high enough that the cam clears the deck

// Where the roller's centre sits at the drawn angle: on the line above the
// shaft, one roller and one cam radius away from the cam's own centre.
LIFT = O[2] + sqrt((R + RR) * (R + RR) - E * E);

module barrel(centre, r, h, sides) {
    translate([centre[0], centre[1] + h / 2, centre[2]])
        rotate([90, 0, 0]) cylinder(h = h, r = r, $fn = sides);
}

part("frame", fixed = true)
    color("slategray") {
        translate([0, -18, 0]) cube([96, 36, 10]);
        translate([O[0] - 9, -26, 10]) cube([18, 8, O[2] - 10]);   // shaft pedestal
    }

part("cam", density = STEEL, friction = 0.06)
    color("indianred") barrel([O[0] + E, 0, O[2]], R, 18, 64);
hinge("cam_shaft", parts = ["cam", "frame"], at = O, axis = PIN);
drive("cam_shaft", speed = 150, torque = 30000000);

// The follower: a stem on a slider, with a roller pinned to the bottom of it.
part("follower", density = PLASTIC, friction = 0.1, bounce = 0)
    color("olivedrab")
        translate([O[0] - 4, -5, LIFT]) cube([8, 10, 38]);
slider("lift", parts = ["follower", "frame"], at = [O[0], 0, LIFT],
       axis = [0, 0, 1], range = [-16, 16]);

part("roller", density = STEEL, friction = 0.06, bounce = 0)
    color("goldenrod") barrel([O[0], 0, LIFT], RR, 14, 32);
hinge("roller_pin", parts = ["roller", "follower"], at = [O[0], 0, LIFT], axis = PIN);

// The return spring: hold it down, but only this hard. Without one the follower
// rattles over the nose, where the cam drops away from under it.
drive("lift", to = -16, over = 0.4, at = 0, torque = 60000);
