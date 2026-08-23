// A Geneva wheel: continuous in, four steps and four stops out.
//
// The driver turns steadily and never stops. The wheel turns a quarter of a turn,
// stops dead, waits three quarters of the driver's revolution, and does it again.
// Nothing here says "index" or "dwell" or "90 degrees". There is a pin on a
// crank, four slots in a disc, and the arithmetic that decides where they go.
//
// That arithmetic is the whole model. For an n-slot wheel at centre distance C,
//
//     crank radius   a = C·sin(180/n)
//     wheel radius   b = C·cos(180/n)
//
// and a² + b² = C² is not a coincidence — it is the condition that the pin's
// circular path crosses the wheel's rim at exactly the angle the slot points.
// Get it right and the pin enters the slot *along* it rather than across it,
// and — this is the part worth watching — the wheel's speed is zero at the moment
// of entry and zero again at exit. It is accelerated from rest and brought back
// to rest by the geometry, which is why a Geneva can be run fast without
// hammering itself apart, and why it needs no brake to hold still between steps.
// It arrives stopped.
//
// The four-slot version is the one that ran cinema projectors: 24 frames a
// second, each yanked into the gate and held perfectly still while it was lit.
//
// ---------------------------------------------------------------------------
// It is built out of convex pieces, and that is the engineering of this model
// rather than a detail of it.
//
// A moving part's collider is its convex hull. A slotted disc's hull is a disc —
// the slots fill straight back in, the pin rides round the outside, and there is
// no mechanism at all. `collider = "decompose"` exists for exactly this, and was
// the first thing tried: it splits a concave part into convex pieces at a cost of
// about a voxel of accuracy. Here that voxel *is* the mechanism. Every clearance
// in a Geneva is a millimetre or two — the fit of the pin in its slot, the depth
// the slot is cut to, the air under the crank — and a collider a voxel oversize
// closes all of them at once. It jammed three separate ways: after 0.9° of the
// first revolution, then after 39°, then after 55°, each time with clear air
// where the drawing said there was clearance and the interference check agreeing
// there was.
//
// So the wheel is a hub with four petals bolted to it, each petal convex, and the
// slots are the gaps between them. Eight parts and six welds against one part and
// none, and every surface is exactly where it was drawn.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module weld(name, parts, at, axis) { }
module drive(name, to, over, at, torque, speed) { }

STEEL   = 0.0078;       // g/mm³
PLASTIC = 0.0012;
DECK    = 10;

SLOTS = 4;
C = 70;                                 // centre distance
A = C * sin(180 / SLOTS);               // the crank's radius: 49.50
B = C * cos(180 / SLOTS);               // and the wheel's:    49.50
PIN = 4;                                // the pin
FIT = 1.5;                              // and its clearance in the slot
HALF = PIN + FIT;                       // so the slot comes out 11 wide

// How deep the slot is cut. The pin's own path bottoms out at C − a = 20.5 from
// the wheel's centre, so its inner face never comes nearer than 16.5; cutting to
// 12 leaves four and a half millimetres it will never need.
SLOT_IN = 12;
HUB = 13;                               // and the hub fills what is left

DRIVE_AT = [30, 0, DECK];
WHEEL_AT = [DRIVE_AT[0] + C, 0, DECK];

// The wheel sits *above* the crank, and only the pin comes up into its plane.
// That is not a detail either — it is the only way the two can overlap at all.
// The crank sweeps a circle 120 mm across about a centre 70 from the wheel, and
// the wheel is 99 across: in one plane they would foul for most of every
// revolution. In two, with a pin bridging them, they meet only where they should.
CRANK_T = 6;
CLEAR   = 12;
WHEEL_Z = DECK + CRANK_T + CLEAR;
THICK   = 12;

// Drawn 15° before the pin reaches the rim, so the first index starts at once and
// there is room for two. Entry is at exactly −45°: cos φ = C / 2a, and with
// a = C/√2 that is √2/2.
START = -60;

// ---- the driver ------------------------------------------------------------
// The crank, and the pin as a separate part bolted to it — which looks like
// over-engineering and is the other thing that makes this mechanism run.
//
// Drawn as one part, the hull of a flat crank with a tall thin pin at one end of
// it is neither: it is the tapered wedge *between* them, swelling from the pin's
// 4 mm at the top to the crank's 10 at the bottom. That wedge fills exactly the
// gap the crank was dropped out of the wheel's plane to make, and the two touch
// through it. The symptom was a crank that turned to the same angle whatever it
// started from — from −60°, from −90°, from −180°, always to −56° — and stopped
// dead there, with 4.6 mm of clear air between the pin and the rim.
part("driver", density = STEEL)
    color("indianred") translate(DRIVE_AT) rotate([0, 0, START])
        hull() {
            cylinder(h = CRANK_T, r = 14, $fn = 32);
            translate([A, 0, 0]) cylinder(h = CRANK_T, r = 10, $fn = 24);
        }
hinge("crank", parts = ["driver", "frame"], at = DRIVE_AT, axis = [0, 0, 1]);
drive("crank", speed = 180, torque = 4000000);

PIN_AT = [DRIVE_AT[0] + A * cos(START), A * sin(START), DRIVE_AT[2] + CRANK_T - 1];
part("pin", density = STEEL)
    color("firebrick") translate(PIN_AT)
        cylinder(h = CLEAR + THICK + 4, r = PIN, $fn = 24);
weld("pin_bolts", parts = ["pin", "driver"], at = PIN_AT, axis = [0, 0, 1]);

// ---- the wheel -------------------------------------------------------------
// A hub, and four petals bolted to it. Each petal is bounded by two straight slot
// walls, an arc of the rim, and a chord across its inside — convex, so its hull
// is its shape and the slot beside it is exactly as wide as it was drawn.
function petal(i, steps) =
    let (s0 = 180 / SLOTS + i * 360 / SLOTS,
         s1 = s0 + 360 / SLOTS,
         d  = asin(HALF / B))
    concat(
        [[SLOT_IN * cos(s0) - HALF * sin(s0), SLOT_IN * sin(s0) + HALF * cos(s0)]],
        [for (k = [0 : steps])
            let (t = (s0 + d) + ((s1 - d) - (s0 + d)) * k / steps)
            [B * cos(t), B * sin(t)]],
        [[SLOT_IN * cos(s1) + HALF * sin(s1), SLOT_IN * sin(s1) - HALF * cos(s1)]]);

module petal_at(i) {
    translate([WHEEL_AT[0], WHEEL_AT[1], WHEEL_Z])
        linear_extrude(height = THICK) polygon(points = petal(i, 10));
}
// Where a petal's inner chord crosses the hub, which is where its bolts go.
function spoke(i) =
    let (t = 90 + i * 90)
    [WHEEL_AT[0] + 12.6 * cos(t), WHEEL_AT[1] + 12.6 * sin(t), WHEEL_Z + THICK / 2];

part("hub", density = PLASTIC, friction = 0.05)
    color("steelblue") translate([WHEEL_AT[0], WHEEL_AT[1], WHEEL_Z])
        cylinder(h = THICK, r = HUB, $fn = 40);
hinge("index", parts = ["hub", "frame"], at = WHEEL_AT, axis = [0, 0, 1]);

// Drawn one at a time rather than in a loop, because each is its own part and a
// part needs a name of its own.
part("petal_a", density = PLASTIC, friction = 0.05) color("steelblue") petal_at(0);
weld("spoke_a", parts = ["petal_a", "hub"], at = spoke(0), axis = [0, 0, 1]);

part("petal_b", density = PLASTIC, friction = 0.05) color("steelblue") petal_at(1);
weld("spoke_b", parts = ["petal_b", "hub"], at = spoke(1), axis = [0, 0, 1]);

part("petal_c", density = PLASTIC, friction = 0.05) color("steelblue") petal_at(2);
weld("spoke_c", parts = ["petal_c", "hub"], at = spoke(2), axis = [0, 0, 1]);

part("petal_d", density = PLASTIC, friction = 0.05) color("steelblue") petal_at(3);
weld("spoke_d", parts = ["petal_d", "hub"], at = spoke(3), axis = [0, 0, 1]);

// ---- the frame -------------------------------------------------------------
// A deck and two shafts, each thinner than the hub it carries.
part("frame", fixed = true)
    color("slategray") {
        translate([-30, -60, 0]) cube([190, 120, DECK]);
        color("dimgray") {
            translate(DRIVE_AT) translate([0, 0, -1]) cylinder(h = 12, r = 8, $fn = 24);
            translate(WHEEL_AT) translate([0, 0, -1])
                cylinder(h = WHEEL_Z + THICK - DECK + 3, r = 6, $fn = 24);
        }
    }
