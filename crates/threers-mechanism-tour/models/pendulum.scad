// One hinge, one arm, and nothing driving it.
//
// Every other mechanism here is told what to do by a drive. This one is told
// nothing at all: it is drawn 30° off vertical and let go. What it does after
// that is not a declaration, an animation curve or a tuned number — it is the
// only thing a hinge, a mass and gravity permit.
//
// Which makes it the one model whose answer can be written down in advance. A
// body swinging on a hinge is a compound pendulum, and its period is
//
//     T = 2π · √( I / (m·g·d) ) · (1 + θ₀²/16 + …)
//
// where `d` is how far the centre of mass sits from the hinge line and `I` is
// the moment of inertia about that line — for this arm, a 50 × 8 bar hung by one
// end, `m(L² + w²)/12 + m·d²`. That comes to 0.374 s, and nothing in this file
// says so. The tour's test measures the swing and checks it.
//
// The arm is hung from a bracket that overhangs the post, because a pendulum
// pivoted on the face of its own column swings straight into it. Nothing here
// would complain — the arm is mated to the frame, and two parts a mate joins do
// not collide — which is exactly why it is worth drawing properly.

module part(name, fixed = false, density = undef, mass = undef,
            collider = undef, friction = undef, bounce = undef) { children(); }
module hinge(name, parts, at, axis, range) { }

STEEL = 0.0078;         // g/mm³
PIN   = [0, 1, 0];      // the arm swings in the xz plane

L     = 50;             // arm length, hinge to tip
W     = 8;              // and its thickness in the swing plane
START = 30;             // where it is let go

PIVOT = [56, 0, 68];

part("frame", fixed = true)
    color("slategray") {
        translate([0, -16, 0]) cube([100, 32, 8]);
        translate([8, -8, 8]) cube([16, 16, 60]);          // the post
        translate([8, -8, 68]) cube([54, 16, 8]);          // and its bracket
    }

// Drawn where it is released. A model's drawn pose *is* its initial condition,
// so tilting it here is the whole of "and then let go".
part("arm", density = STEEL, friction = 0.2, bounce = 0)
    color("indianred")
        translate(PIVOT) rotate([0, -START, 0])
            translate([-W / 2, -8, -L]) cube([W, 16, L]);

hinge("swing", parts = ["arm", "frame"], at = PIVOT, axis = PIN);
