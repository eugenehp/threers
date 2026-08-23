// A slider-crank: the linkage in every engine ever built.
//
// Three revolutes and a prismatic, closing a loop through a *sliding* joint
// rather than a fourth hinge. That is the whole difference from the four-bar,
// and it is what turns going round into going back and forth.
//
// It is also the tour's second answer to the same question the cam asks, reached
// the other way. The cam finds its follower's motion through a *contact* — two
// shapes and whatever room they leave each other. This finds the piston's motion
// through a *constraint* — a rod of a fixed length with its ends pinned. Both
// have a closed form, neither has it written down anywhere, and the two can be
// read against each other:
//
//     x(θ) = O + R·cosθ + √(L² − R²·sin²θ)
//
// What makes it worth plotting is the square root. Take it away — pretend the
// rod is infinitely long — and the piston is a pure cosine, symmetric, with the
// same time either side of mid-stroke. The real thing is not: the second term
// adds a harmonic at twice crank speed, so the piston spends *less* time in the
// outer half of its stroke than the inner, and is going faster at 90° after top
// centre than a cosine would have it. With R/L = 1/4 that is 5 mm on a 40 mm
// stroke, which is 12% and is the reason engines need balance shafts.
//
// Nothing here says any of that. It is four parts, four joints and a motor.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module slider(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque, speed) { }

STEEL = 0.0078;         // g/mm³
ALLOY = 0.0027;         // the piston, as a piston is
PIN   = [0, 1, 0];      // everything turns about y, so the engine is planar

DECK = 10;
O = [40, 0, 60];        // the crankshaft's axis
R = 20;                 // the throw — half the stroke
L = 80;                 // and the rod, centre to centre

// Drawn at 90° after top centre, which is mid-*angle* and emphatically not
// mid-stroke: the piston is 17.5 mm from one end of its travel and 22.5 from the
// other, and that asymmetry is the model.
CX  = O[0] + sqrt(L * L - R * R);
B   = [O[0], 0, O[2] + R];       // the crank pin, straight up at this angle
C   = [CX, 0, O[2]];             // and the gudgeon pin, on the bore's axis
TDC = O[0] + R + L;
BDC = O[0] - R + L;

BORE  = 19;             // the cylinder's inside radius
PISTON = 18.5;          // and the piston, half a millimetre under it
CYL = [86, 155];        // how far the cylinder runs in x

module along_x(at, r, len, sides = 32) {
    translate(at) rotate([0, 90, 0]) cylinder(h = len, r = r, $fn = sides);
}
module along_y(at, r, from, len, sides = 24) {
    translate([at[0], from, at[2]]) rotate([-90, 0, 0])
        cylinder(h = len, r = r, $fn = sides);
}

// ---- the frame -------------------------------------------------------------
// A bed, a main bearing for the crankshaft, and the cylinder — which is drawn
// sectioned, with the front of its wall cut away.
//
// That is not a liberty. A cylinder is a bore, and a bore that goes all the way
// round is opaque: the piston inside it is the one thing this model is about and
// a solid tube hides it completely. Cutting the front open is how every engine
// has ever been drawn, and it leaves the joint intact — the piston still runs in
// a bore, with a half-millimetre fit measurable all the way round the back.
//
// Nothing overlaps by nothing. Each solid sinks a millimetre into the one it
// stands on: exactly coplanar touching faces are the worst case an exact boolean
// kernel has, and this frame has two booleans in it already.
part("frame", fixed = true)
    color("slategray") {
        translate([0, -26, 0]) cube([160, 64, DECK]);

        color("dimgray") {
            // The main bearing, behind the crank's own web. +y is the far side
            // from the tour's camera, and everything structural lives there.
            translate([28, 26, DECK - 1]) cube([24, 12, 63]);

            difference() {
                along_x([CYL[0], 0, O[2]], BORE + 6, CYL[1] - CYL[0]);
                along_x([CYL[0] - 2, 0, O[2]], BORE, CYL[1] - CYL[0] + 4);
                // the section cut, taking the near wall away
                translate([CYL[0] - 2, -32, O[2] - 26])
                    cube([CYL[1] - CYL[0] + 4, 19, 52]);
            }
            // and two feet under it
            translate([92, 12, DECK - 1]) cube([12, 14, 32]);
            translate([138, 12, DECK - 1]) cube([12, 14, 32]);
        }
    }

// ---- the crankshaft --------------------------------------------------------
// A journal running back into that bearing, a web out to the throw, the crank
// pin itself, and a counterweight opposite. The counterweight is the one part
// here that does no work at all and is not decoration either: it is what an
// unbalanced 100 g of steel swinging on a 20 mm arm costs, and leaving it off
// shows up in the main bearing rather than in the motion.
part("crank", density = STEEL)
    color("indianred") {
        hull() {
            along_y(O, 11, 6, 8);
            along_y(B, 8, 6, 8);
        }
        hull() {
            along_y(O, 11, 6, 8);
            along_y([O[0], 0, O[2] - R * 0.55], 13, 6, 8);
        }
        color("firebrick") {
            along_y(O, 8, 13, 21);          // the journal, into its bearing
            along_y(B, 5, -7, 14);          // and the crank pin, through the rod
        }
    }
hinge("crank_pin", parts = ["crank", "frame"], at = O, axis = PIN);

// ---- the connecting rod ----------------------------------------------------
// Fat at the big end and thin at the little one, which is what a rod is and also
// exactly what its collider will be: the convex hull of two circles.
module rod(p, q, rp, rq, t) {
    hull() {
        translate([p[0], -t / 2, p[2]]) rotate([-90, 0, 0])
            cylinder(h = t, r = rp, $fn = 28);
        translate([q[0], -t / 2, q[2]]) rotate([-90, 0, 0])
            cylinder(h = t, r = rq, $fn = 28);
    }
}
part("conrod", density = STEEL)
    color("goldenrod") rod(B, C, 7, 5, 10);
hinge("big_end", parts = ["conrod", "crank"], at = B, axis = PIN);

// ---- the piston ------------------------------------------------------------
// Slotted up its back so the rod can reach the gudgeon pin. Drawn solid, the
// rod's little end is simply *inside* it — two solids in one place, which is the
// fault this tour keeps finding and which a mate hides every time, since the
// pair is joined and so excluded from the interference check.
//
// The slot costs nothing where it matters: a moving part's collider is its
// convex hull, so what runs in the bore is still the full cylinder.
part("piston", density = ALLOY, friction = 0.05)
    color("steelblue") {
        difference() {
            along_x([C[0] - 11, 0, O[2]], PISTON, 22);
            translate([C[0] - 12, -5.5, O[2] - 15]) cube([13, 11, 30]);
        }
        color("cadetblue") along_y(C, 4, -9, 18);
    }
hinge("little_end", parts = ["piston", "conrod"], at = C, axis = PIN);
slider("bore", parts = ["piston", "frame"], at = C, axis = [1, 0, 0],
       range = [BDC - CX, TDC - CX]);

// One input, and everything else is a consequence.
//
// The torque is not a guess. Nothing here is lifted very far, but the rod's big
// end carries 86 g of steel round a 20 mm circle and the piston is 54 g being
// turned round twice a revolution; the weight alone is about 17,000,000 at the
// crank, and a motor that cannot hold that stops being an input and starts being
// a thing the linkage swings. Twenty million holds 180°/s to within a degree.
drive("crank_pin", speed = 180, torque = 20000000);
