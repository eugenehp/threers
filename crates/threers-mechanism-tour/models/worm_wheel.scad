// A worm and wheel: forty to one, in one mesh.
//
// The gear train elsewhere in this tour needs three shafts and two meshes to
// reach 7.5:1. This reaches 40:1 with two parts, because a worm is a screw and
// its wheel counts threads rather than teeth: one turn of the worm advances the
// wheel by exactly one tooth, so the ratio *is* the tooth count. Nothing else in
// mechanics gives that much reduction in one step, and it is why anything that
// has to hold its position — a lathe's indexing head, a lock gate, a rudder
// quadrant, the elevation drive of a big telescope — is likely to be one.
//
// It is also the only pair in the tour whose two shafts are not parallel, and
// that took a change to the front end to say. A `gear()` mate had one `axis` for
// both parts, which is right for a spur pair on parallel shafts and cannot
// describe this at all. `axis_b` gives the second-named part its own; the joint
// underneath had always taken two.
//
// ---------------------------------------------------------------------------
// What this model does *not* show is the self-locking, and it is worth being
// straight about why.
//
// A real worm of this size is self-locking: drive the worm and the wheel turns,
// push the wheel and nothing moves at all. That is not a mechanism, it is
// friction. The thread is an inclined plane wrapped round a cylinder, and its
// lead angle here is
//
//     λ = atan(lead / (π·d)) = atan(7.07 / (π·24)) = 5.4°
//
// against a steel-on-bronze friction angle of about 6°. Below the friction angle
// the load cannot push itself back down the ramp — the same reason a shallow
// wedge stays put and a steep one does not.
//
// A `gear()` mate is an ideal constraint with no friction in it, so this model
// back-drives freely: turn the wheel and the worm spins. The kinematics are
// exact and the statics are not there at all. Worth knowing before trusting a
// simulated worm drive to hold anything up.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module gear(name, parts, at, axis, axis_b, ratio) { }
module drive(name, to, over, at, torque, speed) { }

STEEL  = 0.0078;        // g/mm³
BRONZE = 0.0088;        // the wheel, as a worm wheel is
DECK   = 10;

TEETH = 40;             // and so, single-start, the ratio
PITCH_R = 45;           // the wheel's pitch radius
M = 2 * PITCH_R / TEETH;                 // module: 2.25
LEAD = 3.14159265 * M;                   // the worm's axial pitch = the wheel's
                                         // circular pitch. 7.069 mm.

WORM_R = 12;            // the worm's pitch radius
THREAD = 3;             // and how far its thread stands proud
TURNS = 4;

// The wheel lies flat and turns about z; the worm lies across the back of it and
// turns about x. Their axes are perpendicular and offset by the sum of the two
// pitch radii, which is the only spacing at which a thread meets a tooth.
WHEEL = [60, 0, DECK + 30];
WORM  = [60, PITCH_R + WORM_R, DECK + 30];
MESH  = [WHEEL[0], PITCH_R, WHEEL[2]];

// ---- the wheel -------------------------------------------------------------
// Straight-cut, which a cheap worm wheel is. A proper one is *throated* — its
// rim hollowed to the worm's radius so the thread beds into it over an arc
// rather than touching at a point — and that is a boolean this model does not
// need, since the mesh here is a constraint rather than a contact.
function tooth_ring(n) =
    let (p = 360 / n, tip = PITCH_R + M, root = PITCH_R - 1.25 * M)
    [for (i = [0 : n - 1]) each [
        [root * cos((i - 0.25) * p), root * sin((i - 0.25) * p)],
        [tip  * cos((i - 0.09) * p), tip  * sin((i - 0.09) * p)],
        [tip  * cos((i + 0.09) * p), tip  * sin((i + 0.09) * p)],
        [root * cos((i + 0.25) * p), root * sin((i + 0.25) * p)],
    ]];

part("wheel", density = BRONZE)
    color("olivedrab") translate(WHEEL) {
        translate([0, 0, -7]) linear_extrude(height = 14)
            polygon(points = tooth_ring(TEETH));
        // A hub down onto its shaft, overlapping the wheel rather than butting
        // against it.
        color("darkolivegreen") translate([0, 0, -8]) cylinder(h = 10, r = 14, $fn = 32);
    }
hinge("wheel_shaft", parts = ["wheel", "frame"], at = WHEEL, axis = [0, 0, 1]);

// ---- the worm --------------------------------------------------------------
// A real helix, not a suggestion of one: the profile below is a core circle with
// one lobe on it, extruded along the axis while twisting a full turn every LEAD
// of length. That is what a single-start thread *is*, and it is why one turn of
// the worm moves the wheel by one tooth.
//
// `slices` is the cost. A helix is a swept curve and the extruder can only
// approximate it with flat rings; 20 a turn is enough to look like metal and
// costs about 3,000 triangles, where 60 a turn costs 9,000 and looks the same.
module worm_thread() {
    linear_extrude(height = TURNS * LEAD, twist = -TURNS * 360, slices = TURNS * 20)
        union() {
            circle(r = WORM_R - THREAD / 2, $fn = 24);
            polygon([[WORM_R - THREAD - 1, -1.6], [WORM_R + THREAD, -0.9],
                     [WORM_R + THREAD, 0.9], [WORM_R - THREAD - 1, 1.6]]);
        }
}

part("worm", density = STEEL)
    color("indianred") translate(WORM) rotate([0, 90, 0]) {
        translate([0, 0, -TURNS * LEAD / 2]) worm_thread();
        // Journals out either end, into the frame's two housings.
        color("firebrick") {
            translate([0, 0, -TURNS * LEAD / 2 - 26]) cylinder(h = 28, r = 7, $fn = 24);
            translate([0, 0, TURNS * LEAD / 2 - 2]) cylinder(h = 28, r = 7, $fn = 24);
        }
    }
hinge("worm_shaft", parts = ["worm", "frame"], at = WORM, axis = [1, 0, 0]);

// One turn of the worm, one tooth of the wheel. Negative for the same reason a
// spur mesh is: the two turn opposite ways about their own axes.
gear("mesh", parts = ["worm", "wheel"], at = MESH,
     axis = [1, 0, 0], axis_b = [0, 0, 1], ratio = -TEETH);

// ---- the frame -------------------------------------------------------------
// A deck, a shaft for the wheel, and two housings carrying the worm — one at
// each end, so nothing stands between the mesh and the camera at -y.
part("frame", fixed = true)
    color("slategray") {
        translate([0, -40, 0]) cube([120, 110, DECK]);
        color("dimgray") {
            translate([WHEEL[0], WHEEL[1], DECK - 1])
                cylinder(h = WHEEL[2] - DECK + 9, r = 9, $fn = 24);
            for (s = [-1, 1]) {
                translate([WORM[0] + s * 42 - 9, WORM[1] - 11, DECK - 1])
                    cube([18, 22, WORM[2] - DECK - 10]);
                translate([WORM[0] + s * 42 - 9, WORM[1] - 13, WORM[2] - 12])
                    cube([18, 26, 24]);
            }
        }
    }

// 90°/s on the worm is 2.25°/s at the wheel — nine degrees in the four seconds
// this runs, which is what a 40:1 reduction looks like and the point of it.
drive("worm_shaft", speed = 90, torque = 300000);
