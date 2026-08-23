// A small bench that exercises every kind of joint, drive and check.
//
//   cargo run -p threers-physics --features assembly,openscad --example mechanism_tour
//
// Drawn in millimetres, as CAD is. The shims below are what let OpenSCAD open
// this file: it has never heard of these modules and would otherwise drop each
// part() along with its children. threers overrides them with the real thing.
//
// Every joint here is *built* as well as declared, and that distinction is the
// one this bench exists to make. A mate is a sentence about two parts. It puts
// no pin in a bore, no rail under a carriage and no teeth on a gear, and it
// hides the omission twice over: the pair it names is excluded from the
// collision solver and from the interference check, because a real pin does live
// inside both halves of a real joint. So a shaft threaded on nothing, or a pair
// of gears 6 mm apart, simulates perfectly and reads perfectly. Only asking the
// geometry finds it — see the tour's `probe` example.

module part(name, fixed = false, density = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module slider(name, parts, at, axis, range) { }
module screw(name, parts, at, axis, pitch) { }
module gear(name, parts, at, axis, ratio) { }
module drive(name, to, over, at, torque, speed) { }

STEEL   = 0.0078;   // g/mm³
PLASTIC = 0.0012;
DECK    = 10;       // top of the bench

// ---- the gear pair, sized from its teeth ----------------------------------
// One module shared between them is what makes two gears a pair: it fixes each
// pitch radius at M·n/2, and so the centre distance at the sum of the two. The
// ratio is then the tooth counts and nothing else. Drawn the other way round —
// two plain discs at a spacing somebody liked, with a gear() mate asserting 2:1
// — the simulation is identical and the machine is a fiction.
M       = 2;
N_CRANK = 12;
N_WHEEL = 24;
function pitch_r(n) = M * n / 2;                 // 12 and 24
CENTRES = pitch_r(N_CRANK) + pitch_r(N_WHEEL);   // so, 36

CRANK = [35, 27, DECK];
WHEEL = [CRANK[0] + CENTRES, 27, DECK];

// One tooth is four points: out to the tip, across it, and back down to the
// root. Tooth centres land on multiples of the pitch angle, so an unphased gear
// has a tooth pointing along +x.
function gear_2d(n) =
    let (p = 360 / n, r = pitch_r(n), tip = r + M, root = r - 1.25 * M)
    [for (i = [0 : n - 1]) each [
        [root * cos((i - 0.25) * p), root * sin((i - 0.25) * p)],
        [tip  * cos((i - 0.07) * p), tip  * sin((i - 0.07) * p)],
        [tip  * cos((i + 0.07) * p), tip  * sin((i + 0.07) * p)],
        [root * cos((i + 0.25) * p), root * sin((i + 0.25) * p)],
    ]];

module wheel(n, h, phase = 0) {
    rotate([0, 0, phase]) linear_extrude(height = h) polygon(points = gear_2d(n));
}

BOLT  = [160, 20, DECK];
PIVOT = [10, 86, 13];      // the lid's hinge line
RAIL  = 133;               // the carriage rail's centreline in x

// ---------------------------------------------------------------- the bench
// The deck, and everything the moving parts actually run on: two shafts, a
// carriage rail, a boss for the bolt to thread into, and a pair of hinge lugs.
// None of it is decoration — each one is the thing its mate would otherwise be
// asserting into thin air.
part("base", fixed = true)
    color("slategray") {
        cube([180, 90, DECK]);

        color("dimgray") {
            // Shafts, each thinner than the root circle of the wheel on it.
            translate(CRANK) translate([0, 0, -DECK]) cylinder(h = 20, r = 4, $fn = 24);
            translate(WHEEL) translate([0, 0, -DECK]) cylinder(h = 20, r = 4, $fn = 24);

            // The carriage's rail: a stalk with a head, which the carriage hooks
            // under. A block set on a flat deck is held down by gravity and by
            // the slider mate, and by nothing you could machine.
            translate([RAIL - 4, 0, DECK - 1]) cube([8, 60, 5]);
            translate([RAIL - 10, 0, DECK + 4]) cube([20, 60, 4]);

            // A boss for the bolt to run into. A screw mate ties turning to
            // advancing; the thread it is named after has to be somewhere.
            translate(BOLT) cylinder(h = 6, r = 9, $fn = 32);

            // Two lugs for the lid's hinge, and the pin through them. The lid's
            // own knuckle goes between them.
            translate([6, 82, DECK]) cube([8, 8, 9]);
            translate([66, 82, DECK]) cube([8, 8, 9]);
            translate([4, PIVOT[1], PIVOT[2]]) rotate([0, 90, 0])
                cylinder(h = 72, r = 2.5, $fn = 20);
        }
    }

// ---- a crank that turns for as long as the run lasts ----------------------
// A shaft making full turns cannot be driven to a position: a hinge angle
// wraps at ±360°. `speed` is the tool for it.
part("crank", density = STEEL)
    color("indianred")
    translate(CRANK) {
        wheel(N_CRANK, 6);
        // A hub, inside the root circle so the collider is still the gear.
        color("firebrick") translate([-7, -3, 0]) cube([14, 6, 6]);
    }
hinge("crank_axle", parts = ["crank", "base"], at = CRANK, axis = [0, 0, 1]);
drive("crank_axle", speed = 120, torque = 40000);

// ---- geared to a wheel, 2:1 and counter-rotating --------------------------
// A gear constrains a *rate*, so both parts still need the hinges that hold
// them. Negative because external gears mesh the other way round.
//
// Phased so a tooth of one arrives at a gap of the other: the crank has a tooth
// pointing straight at the wheel, and the wheel — 24 teeth, so one every 15°,
// and 180 is a multiple of 15 — would meet it with a tooth of its own. Half a
// pitch round fixes that, and it is derived rather than dialled in.
PHASE = 180 / N_WHEEL;
part("wheel", density = STEEL)
    color("steelblue") translate(WHEEL) wheel(N_WHEEL, 6, PHASE);
hinge("wheel_axle", parts = ["wheel", "base"], at = WHEEL, axis = [0, 0, 1]);
gear("mesh", parts = ["crank", "wheel"],
     at = [CRANK[0] + pitch_r(N_CRANK), CRANK[1], DECK + 3], axis = [0, 0, 1],
     ratio = -N_WHEEL / N_CRANK);

// ---- a carriage on a rail, driven to a position --------------------------
// Drawn as what it is: a body over the rail, two walls down its sides and two
// hooks reaching in under the head. Over, under and either side of the head is
// five of the six freedoms gone, which is what a prismatic joint *is* — and it
// is the difference between a carriage and a brick.
part("carriage", density = PLASTIC)
    color("olivedrab") translate([0, 8, 0]) {
        translate([RAIL - 12, 0, DECK + 8.2]) cube([24, 16, 12]);   // body
        translate([RAIL - 12, 0, DECK - 0.2]) cube([1.8, 16, 8.4]); // near wall
        translate([RAIL + 10.2, 0, DECK - 0.2]) cube([1.8, 16, 8.4]);
        translate([RAIL - 12, 0, DECK - 0.2]) cube([7.5, 16, 4]);   // hooks
        translate([RAIL + 4.5, 0, DECK - 0.2]) cube([7.5, 16, 4]);
    }
slider("feed", parts = ["carriage", "base"],
       at = [RAIL, 8, DECK + 6], axis = [0, 1, 0], range = [0, 26]);
drive("feed", to = 26, over = 1.2, at = 0.4);

// ---- a bolt that turns itself into the deck -------------------------------
// One mate, not two: a thread holds the pair on an axis *and* ties turning to
// advancing, so driving its travel turns it. The head is what you watch: the
// shank is inside the boss, which is where a bolt's shank belongs.
part("bolt", density = STEEL)
    color("goldenrod")
    translate(BOLT) {
        cylinder(h = 20, r = 4, $fn = 24);
        translate([0, 0, 16]) cylinder(h = 5, r = 7, $fn = 6);
    }
screw("thread", parts = ["bolt", "base"], at = BOLT, axis = [0, 0, 1],
      pitch = 1.25, range = [-5, 0]);
// Degrees a second, as on any other rotation — a screw's *coordinate* is its
// travel, but what a motor on one drives is the shaft. 500°/s down a 1.25 mm
// lead is 1.7 mm a second, so it bottoms out about three seconds in.
//
// The torque has to *hold* the bolt as well as turn it, and a fine lead is an
// enormous mechanical advantage in both directions: 76,000 of bolt weight comes
// back to the shaft as 76,000 x 1.25 / 2pi = 15,000 of torque. A motor given
// 8,000 cannot brake its own bolt, and the thread runs away under it.
drive("thread", speed = 500, torque = 60000);

// ---- a lid that is asked for more than it can have ------------------------
// The axis points along −x so a positive angle *lifts* it. By the right-hand
// rule about +x the lid would swing down through the machinery instead, which
// the simulation would report as a stalled crank rather than as a mistake.
//
// Its knuckle runs between the base's two lugs, on the base's pin — three
// knuckles and a pin, which is a hinge. A flat plate laid on a deck with a
// hinge() naming its back edge is a plate laid on a deck.
part("lid", density = PLASTIC)
    color("sienna") {
        translate([16, 51, 13]) cube([48, 35, 3]);
        translate([16, PIVOT[1], PIVOT[2]]) rotate([0, 90, 0])
            cylinder(h = 48, r = 3, $fn = 24);
    }
hinge("lid_pivot", parts = ["lid", "base"], at = PIVOT, axis = [-1, 0, 0], range = [0, 105]);
drive("lid_pivot", to = 105, over = 1.0, at = 0.6);

// The stop it will actually reach, well short of the 105 asked for.
part("stop", fixed = true)
    color("dimgray")
    translate([5, 58, 36]) cube([70, 16, 6]);

// ---- and one part attached to nothing at all ------------------------------
// Renders, has mass, interferes with nothing, and passes every overlap check
// ever written. Only a contact check across poses notices.
part("spare", density = PLASTIC)
    color("orchid")
    translate([150, 62, DECK]) cube([16, 16, 16]);
