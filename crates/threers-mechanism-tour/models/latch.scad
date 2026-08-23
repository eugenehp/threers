// A gate, a spring latch, and a drive that cannot open it.
//
// This is the whole argument in one model. The gate is driven to a position by
// a servo with a force ceiling, four times over: shut it, open it, give up,
// and — once the latch has been released — open it. The second of those is the
// interesting one. Nothing here says "the gate is locked". The hook drops
// behind the catch because it is heavy and sprung, the catch runs into it
// because it is in the way, and the drive fails to reach the number it was
// given because a bounded force cannot beat a hinge stop.
//
// A drive that could always reach its target would teleport the gate through
// the hook, and the animation would look fine and be a lie.
//
// The pawl is two parts bolted together rather than one, and the reason is the
// collider. Drawn as a single L, its convex hull is the *triangle* between the
// arm and the hook — which fills in exactly the gap the catch has to pass
// through, and the check duly reported it fouling before anything moved. An arm
// and a hook are each convex on their own. `collider = "decompose"` would also
// work and costs a voxel of accuracy; two parts and a weld cost nothing.
//
// Also here: a rack and pinion used as a *readout*, the gate turning the pinion
// rather than the other way about — the same constraint read from the far end.
//
// Every joint in it is built as well as declared, and that took more metal than
// the mechanism itself. Four of the seven were sentences and nothing else: an
// arm hinged to a post 14 mm away from it, a plunger on a slider with no guide
// within 14 mm, a pinion 9 mm off its own shaft, and a rack and pinion 50 mm
// apart with no teeth on either. All four simulated perfectly, because a mate
// both supplies the constraint and excludes the pair from every check that
// could have noticed there was nothing there.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module slider(name, parts, at, axis, range) { }
module rack(name, parts, at, axis, rack_axis, radius) { }
module weld(name, parts, at, axis) { }
module drive(name, to, over, at, torque, speed) { }

STEEL   = 0.0078;
PLASTIC = 0.0012;
PIN     = [0, 1, 0];

DECK     = 8;
GATE_Z   = 9;                 // riding just clear of the deck
GATE_TOP = 41;
// Six millimetres proud, not ten. The hook rides over the catch and then falls
// off the back of it, and what it falls is what it arrives with: a ten
// millimetre drop landed it five millimetres deep in the catch, a six
// millimetre one lands it a fraction of that. Six is still more bite than the
// gate can ever pull through.
CATCH_TOP = 47;
TRAVEL   = 48;                // how far the gate has to go to be shut

// Where the catch's back face is drawn. It has to end up *past* the hook when
// the gate is shut, or the hook rides on top of the catch instead of dropping
// in behind it — which is not a latch, and which shows up as the hook being
// crushed into the catch rather than sitting beside it.
REAR = 31;
RAMP = 24;                    // 6 up over 24 along: about 14°

PIVOT = [46, 0, 58];          // the pawl swings here
ARM_A = [26, 0, 58];          // tail end, where the release pushes down
ARM_B = [74, 0, 58];          // nose end, where the hook hangs
HOOK  = [73, 0, 45];          // and the bottom of the hook
PINION = [96, 0, 46];         // the readout shaft
PLUNGE = [27, 0, 66];         // and the release plunger's axis

// Everything structural stands *behind* the machine and reaches forward over the
// top of it. The two posts used to be in front, at y = -26, which is the side
// the tour's camera sits on: a pair of grey columns across the one thing the
// model is about. Behind is +y here, as it is in every model in the tour.
//
// And it is one post, not two, carrying one bar the width of the plunger it
// holds. Built as a portal — two columns and a beam across their tops — it was
// structurally excellent and a grey wall through the middle of the picture. The
// release has to be supported from behind, because the gate sweeps through every
// square millimetre of deck under it, but only what the plunger needs has to
// come forward with it.
BACK   = [30, 40];            // the post's near and far faces in y
BEAM_Z = 90;                  // the cantilever the release hangs from

module bar(p, q, r, t) {
    hull() {
        translate(p) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 24, center = true);
        translate(q) rotate([-90, 0, 0]) cylinder(h = t, r = r, $fn = 24, center = true);
    }
}

// ---- the frame ------------------------------------------------------------
// The deck, the rail the gate runs on, two columns, the beam across their tops,
// and — hanging from that beam — the pawl's yoke and the plunger's guide.
//
// The rail is a stalk with a head on it, and the gate wraps that head. A gate
// drawn as a block resting on a flat deck is held up by gravity and held on its
// line by the slider() mate, which is to say by a sentence.
STALK = [14, 20];             // the rail stalk's faces in y
HEAD  = [10.5, 24];           // the head overhangs it both ways
// No two of the twenty-odd solids below share a face. Every one of them
// overlaps its neighbour by a millimetre or sinks a millimetre into the deck,
// and none of that is tidiness: exactly coplanar touching faces are the worst
// case an exact boolean kernel has, and this frame is where that stopped being
// a footnote. Butted together the way they were first drawn — post on deck, bar
// on post, web between two prongs, shaft ending flush in its boss — the union
// came out as 49,224 triangles and took 2.6 seconds to evaluate, because every
// coincident pair splits both faces along the other's edges and the splits
// compound. Given a millimetre each it is 3,000 triangles and 40 milliseconds.
//
// Sixty times faster and sixteen times smaller, for a millimetre nobody can see.
part("frame", fixed = true)
    color("slategray") {
        translate([0, -22, 0]) cube([150, BACK[1] + 26, DECK]);

        color("dimgray") {
            // the gate's rail: a stalk sunk into the deck, and a head over it
            translate([0, STALK[0], DECK - 1]) cube([150, STALK[1] - STALK[0], 12]);
            translate([0, HEAD[0], 18]) cube([150, HEAD[1] - HEAD[0], 6]);

            // one post, and the bar it cantilevers forward over the release
            translate([38, BACK[0] - 1, DECK - 1])
                cube([16, BACK[1] - BACK[0] + 1, BEAM_Z + 4 - DECK]);
            translate([17, -8, BEAM_Z - 2]) cube([24, BACK[1] + 6, 8]);

            // The pawl's yoke: two prongs down either side of the arm, a web
            // back to the post and a bridge over the top tying them together.
            // They start above the catch, which sweeps through at z 41..47 — the
            // arm is mated to this part and so passes through it unchallenged,
            // but the catch is not and would be caught.
            translate([42, -10, 50]) cube([12, 5.5, 20]);
            translate([42, 4.5, 50]) cube([12, 3, 20]);
            translate([43, 6, 52]) cube([10, BACK[0] - 5, 13]);
            translate([43, -9, 63]) cube([10, BACK[0] + 11, 6]);
            translate([PIVOT[0], -10.5, PIVOT[2]]) rotate([-90, 0, 0])
                cylinder(h = 18.5, r = 3, $fn = 20);

            // The plunger's guide: a yoke it runs in, hanging from that bar,
            // because there is nowhere on the deck to stand one — the gate
            // sweeps through all of it. Open down the middle at the front so
            // you can watch the plunger, gibbed at the corners so it is still a
            // guide: over, under and either side is what makes one.
            translate([18, -8, 64]) cube([1.8, 16, BEAM_Z - 63]);
            translate([34.2, -8, 64]) cube([1.8, 16, BEAM_Z - 63]);
            translate([18.5, 6.2, 64]) cube([17, 2, BEAM_Z - 63]);
            translate([19, -8.2, 64]) cube([4, 2, BEAM_Z - 63]);
            translate([31, -8.2, 64]) cube([4, 2, BEAM_Z - 63]);

            // the pinion's post, and its shaft ending inside it rather than
            // flush with its back face
            translate([88, BACK[0] + 3, DECK - 1]) cube([16, 8, PINION[2] - DECK + 7]);
            translate([PINION[0], 16, PINION[2]]) rotate([-90, 0, 0])
                cylinder(h = BACK[1] - 18, r = 4, $fn = 24);
        }
    }

// ---- the gate, and the things bolted to it --------------------------------
// The body, the wrap that holds it on the rail, and the arm carrying its rack.
RACK_M  = 2;                                   // the readout's tooth size
RACK_P  = 3.14159265 * RACK_M;                 // so, its pitch: 6.283
RACK_R  = 12;                                  // the pinion's pitch radius
RACK_Z  = PINION[2] - RACK_R;                  // and the rack's pitch line
// The pinion has a tooth pointing straight down — twelve teeth is one every 30°,
// and 270 is a multiple of 30 — so the rack has to meet it with a gap. Half a
// pitch off a tooth centre is a gap centre, and this is where it has to land.
RACK_X0 = PINION[0] - 8.5 * RACK_P;
RACK_N  = 10;

module rack_teeth(n, m, x0, z0) {
    p = 3.14159265 * m;
    for (i = [0 : n - 1])
        polygon([[x0 + (i - 0.25) * p, z0 - 1.25 * m],
                 [x0 + (i - 0.07) * p, z0 + m],
                 [x0 + (i + 0.07) * p, z0 + m],
                 [x0 + (i + 0.25) * p, z0 - 1.25 * m]]);
}

part("gate", density = PLASTIC, friction = 0.2)
    color("steelblue") {
        translate([10, -10, GATE_Z]) cube([48, 20, GATE_TOP - GATE_Z]);
        // the wrap: over the head, under it either side of the stalk, and the
        // wall down the back tying the two together
        translate([10, 8, 24.2]) cube([48, 18, 5.8]);
        translate([10, 8, 12]) cube([48, 5.5, 5.8]);
        translate([10, 20.5, 12]) cube([48, 5.5, 5.8]);
        translate([10, 24.2, 12]) cube([48, 3.8, 18]);
        // The rack arm, reaching back over the rail to the pinion and staying
        // under it through the whole travel — 48 mm of it. A rack that only
        // meets its pinion at one end of the stroke is a rack for that instant.
        translate([58, 12, 24.2]) cube([44, 12, 7.8]);
        color("cadetblue")
            translate([0, 24, 0]) rotate([90, 0, 0])
                linear_extrude(height = 12)
                    rack_teeth(RACK_N, RACK_M, RACK_X0, RACK_Z);
    }
slider("travel", parts = ["gate", "frame"], at = [10, 17, 21],
       axis = [1, 0, 0], range = [0, TRAVEL]);

// A ramp the hook can ride up and a face it cannot. Drawn as its cross-section
// and extruded across the gate, which is how you would cut it.
//
// The ramp rises 6 over 24, which is 14°. At 45° — ten over ten, which is what
// this was — lifting the pawl costs as much force sideways as it does upwards,
// and the gate has to shove that hard through a single line of contact. The
// hook buried itself 1.1 mm in the catch every time it climbed. A latch ramp is
// shallow for exactly this reason, and it is the same reason a wedge works.
part("catch", density = STEEL, friction = 0.15)
    color("orangered")
        translate([0, 8, 0]) rotate([90, 0, 0])
            linear_extrude(height = 16)
                polygon([[REAR + 8 + RAMP, GATE_TOP], [REAR + 8, CATCH_TOP],
                         [REAR, CATCH_TOP], [REAR, GATE_TOP]]);
weld("catch_bolts", parts = ["catch", "gate"], at = [40, 0, 46], axis = [0, 0, 1]);

// ---- the latch ------------------------------------------------------------
// One straight bar through the pivot: nose end to the right, tail to the left
// for the release to push on. It stays clear above the catch for its whole
// length, which is the point of drawing it this way. It hangs on the frame's
// pin, between the frame's two prongs.
part("arm", density = STEEL, friction = 0.25, bounce = 0)
    color("goldenrod") bar(ARM_A, ARM_B, 4, 8);
// Lifting is negative, and the pawl may not drop below where it is drawn. That
// stop is what the catch pushes against when the gate tries to leave.
hinge("pawl_pivot", parts = ["arm", "frame"], at = PIVOT, axis = PIN,
      range = [-30, 0]);
// The spring: enough to drop the hook back down, not enough to stop a ramp
// lifting it.
drive("pawl_pivot", to = 0, over = 0.2, at = 0, torque = 1400000);

part("hook", density = STEEL, friction = 0.25, bounce = 0)
    color("darkgoldenrod") bar([ARM_B[0] - 1, 0, ARM_B[2]], HOOK, 3, 8);
weld("hook_bolts", parts = ["hook", "arm"], at = ARM_B, axis = [0, 0, 1]);

// ---- the release actuator -------------------------------------------------
// Pushes *down* on the tail, which lifts the nose. A lever is the cheapest way
// to turn a plunger you can put somewhere into a motion you need elsewhere.
part("plunger", density = STEEL, friction = 0.2)
    color("orchid") translate([20, -6, 66]) cube([14, 12, 20]);
slider("release", parts = ["plunger", "frame"], at = PLUNGE,
       axis = [0, 0, 1], range = [-14, 2]);

// ---- a rack and pinion, read backwards ------------------------------------
// Teeth, because a readout drawn as a smooth disc near a smooth block is a
// readout of nothing. The bore is real too: the frame's shaft runs through it.
module wheel(n, m, h) {
    difference() {
        linear_extrude(height = h)
            polygon(points = [for (i = [0 : n - 1]) each
                let (p = 360 / n, r = m * n / 2, tip = r + m, root = r - 1.25 * m) [
                    [root * cos((i - 0.25) * p), root * sin((i - 0.25) * p)],
                    [tip  * cos((i - 0.07) * p), tip  * sin((i - 0.07) * p)],
                    [tip  * cos((i + 0.07) * p), tip  * sin((i + 0.07) * p)],
                    [root * cos((i + 0.25) * p), root * sin((i + 0.25) * p)],
                ]]);
        translate([0, 0, -1]) cylinder(h = h + 2, r = 4.4, $fn = 24);
        for (i = [0 : 2])
            rotate([0, 0, i * 120])
                translate([m * n / 4, 0, -1]) cylinder(h = h + 2, r = m, $fn = 18);
    }
}
part("pinion", density = STEEL)
    color("indianred")
        translate([PINION[0], 34, PINION[2]]) rotate([90, 0, 0])
            wheel(RACK_R * 2 / RACK_M, RACK_M, 18);
hinge("pinion_shaft", parts = ["pinion", "frame"], at = PINION, axis = PIN);
rack("readout", parts = ["pinion", "gate"], at = PINION,
     axis = [0, 1, 0], rack_axis = [1, 0, 0], radius = RACK_R);

// ---- what it is told to do ------------------------------------------------
// Four moves on one mate. They queue: each starts from wherever the one in
// front actually got to, which is not always where it was aimed.
drive("travel", to = TRAVEL, over = 1.1, at = 0.3, torque = 700000);   // shut it
drive("travel", to = 0,      over = 1.0, at = 2.2, torque = 700000);   // open it
drive("travel", to = TRAVEL, over = 0.3, at = 3.3, torque = 700000);   // give up
drive("travel", to = 0,      over = 1.2, at = 4.6, torque = 700000);   // now open it

// Firm, and slow. Firm because a plunger that cannot hold the pawl up against
// its spring bounces on and off it, hammering the hinge dozens of times instead
// of once. Slow because the arrival is what shakes the joint: a revolute joint
// carrying a load has real angular compliance, and taking 0.8 s over this move
// rather than 0.5 took the axis read back out of the geometry from 1.7° off the
// declared one to inside the check's tolerance. The verification is sensitive to
// exactly the thing a machine designer would also care about.
drive("release", to = 0,   over = 0.1, at = 0.0, torque = 1400000);     // held up
drive("release", to = -10, over = 0.8, at = 3.7, torque = 1400000);     // lift the pawl
drive("release", to = 0,   over = 0.4, at = 6.0, torque = 1400000);     // let it drop
