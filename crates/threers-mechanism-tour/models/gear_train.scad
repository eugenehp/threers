// A compound gear train — 7.5 turns of the motor per turn of the output.
//
// A gear() constrains a *rate*, not a place, so every shaft still needs the
// hinge that holds it: the gear says how two shafts are related, the hinges say
// where they are. Stacking two gears on one shaft is what makes a train
// compound — the middle part carries both, so the second stage starts from the
// first stage's output rather than from the input again.
//
// The teeth are real, and they have to be: a gear train drawn as three plain
// discs turning near each other is not a picture of a gear train. They are also
// free. A tooth profile is a *polygon*, and extruding one costs no boolean at
// all — where drilling four lightening holes through each of these wheels cost
// 33 ms a wheel natively and two minutes in a browser.
//
// Two things make teeth mesh rather than merely coexist. Every gear here shares
// one module — the tooth size — which is what fixes each pitch radius at M·n/2
// and each centre distance at the sum of two of them. And each gear is *phased*
// so a tooth of one arrives at a gap of the other: PH_C and PH_OUT below are
// derived from that requirement rather than dialled in by eye.
//
// The teeth overlap in the annulus between the two pitch circles, as meshing
// teeth must, and nothing objects: a gear() is a mate, and two parts a mate
// joins do not also collide.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module gear(name, parts, at, axis, ratio) { }
module drive(name, to, over, at, torque, speed) { }

STEEL = 0.0078;     // g/mm³
DECK  = 8;
M     = 1.5;        // module — the one number every gear here has in common

N_IN = 12; N_BIG = 36; N_SMALL = 16; N_OUT = 40;
function pitch_r(n) = M * n / 2;          // 9, 27, 12 and 30

// Shafts, at the only spacings that mesh: the sum of the two pitch radii.
A = [26, 40, DECK];
B = [A[0] + pitch_r(N_IN) + pitch_r(N_BIG), 40, DECK];
C = [B[0] + pitch_r(N_SMALL) + pitch_r(N_OUT), 40, DECK];

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
    rotate([0, 0, phase])
        linear_extrude(height = h)
            polygon(points = gear_2d(n));
}

// The input arrives at the line of centres with a tooth, so the compound has to
// meet it with a gap: half a pitch round.
PH_C = 180 / N_BIG;
// The compound's small gear is carried round by that same phase, so the output
// has to allow for it — converted to the output's own angular scale, since the
// two turn through different angles for the same arc — and then offset half a
// pitch again to put its gap where that tooth will arrive.
PH_OUT = 180 - PH_C * N_SMALL / N_OUT - 180 / N_OUT;

// The deck, and the three shafts standing on it.
//
// A hinge() says a wheel turns about a line. It does not put anything on that
// line, and a gear train whose wheels are threaded on nothing is a row of discs
// hanging in the air at the height they were drawn. Worse, the one place it
// shows is the one place a mate hides: a wheel is mated to the frame, so the
// pair is excluded from both the collision solver and the interference check,
// and all three agree that a wheel supported by nothing is fine.
//
// Each shaft is thinner than the root circle of every wheel on it — 4 against
// 7.1 at the worst, the little input pinion — so there is metal all the way
// round the bore, and clear of the wheel meshing with it: the big wheel's teeth
// reach to within 3.5 of the input's shaft.
SHAFT = 4;
part("frame", fixed = true)
    color("slategray") {
        cube([145, 80, DECK]);
        color("dimgray") {
            translate([A[0], A[1], 0]) cylinder(h = 20, r = SHAFT, $fn = 24);
            translate([B[0], B[1], 0]) cylinder(h = 27, r = SHAFT, $fn = 24);
            translate([C[0], C[1], 0]) cylinder(h = 29, r = SHAFT, $fn = 24);
        }
    }

part("input", density = STEEL)
    color("indianred") translate(A) wheel(N_IN, 8);
hinge("input_shaft", parts = ["input", "frame"], at = A, axis = [0, 0, 1]);
drive("input_shaft", speed = 180, torque = 90000);

// One part, two wheels, one shaft. The small one overlaps the big one by 0.4
// rather than sitting exactly on top of it: two solids whose faces are exactly
// coplanar and exactly touching are the worst case an exact boolean kernel has,
// and here it is free to avoid.
OVERLAP = 0.4;
part("compound", density = STEEL)
    color("steelblue") translate(B) {
        wheel(N_BIG, 8, PH_C);
        translate([0, 0, 8 - OVERLAP]) wheel(N_SMALL, 8 + OVERLAP, PH_C);
    }
hinge("layshaft", parts = ["compound", "frame"], at = B, axis = [0, 0, 1]);

part("output", density = STEEL)
    color("olivedrab") translate([C[0], C[1], C[2] + 8]) wheel(N_OUT, 8, PH_OUT);
hinge("output_shaft", parts = ["output", "frame"], at = C, axis = [0, 0, 1]);

// Negative because external gears counter-rotate, and equal to the tooth counts
// because that is what a gear ratio *is*. Both together are +7.5, so the output
// turns the same way as the input and a seven-and-a-half-th as fast.
gear("stage_one", parts = ["input", "compound"],
     at = [A[0] + pitch_r(N_IN), 40, DECK + 4], axis = [0, 0, 1],
     ratio = -N_BIG / N_IN);
gear("stage_two", parts = ["compound", "output"],
     at = [B[0] + pitch_r(N_SMALL), 40, DECK + 12], axis = [0, 0, 1],
     ratio = -N_OUT / N_SMALL);
