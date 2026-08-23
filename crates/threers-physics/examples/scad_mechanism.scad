// A hinged box that describes its own mechanism.
//
// Simulated with:
//   cargo run -p threers-physics --features assembly,openscad \
//       --example scad_mechanism
//
// The four modules below are shims. threers overrides them with the real
// thing when it reads this file as a mechanism; OpenSCAD, which has never
// heard of them, uses these instead and renders the model as an ordinary
// static assembly. Without them OpenSCAD would drop every part() *along with
// its children* and show nothing at all.
module part(name, fixed = false, density = undef) { children(); }
module hinge(name, parts, at, axis, range) { }
module drive(name, to, over, at, torque) { }

// ---- dimensions (metres) ---------------------------------------------------
W    = 0.30;   // across the hinge
D    = 0.20;   // front to back
H    = 0.16;   // box height
T    = 0.012;  // lid thickness
OPEN = 105;    // degrees the lid is specified to open

// A shelf hangs over the box. It is scenery, but it is *declared* scenery —
// wrapped in a fixed part() so the mechanism knows it is there and the lid can
// hit it.
// The lid's far edge swings on a D radius from a hinge at z = H, so a shelf
// this high catches it around 45° — well short of the 105° asked for.
SHELF_Z = 0.30;

// ---- the model -------------------------------------------------------------
part("box", fixed = true)
    color("burlywood")
    difference() {
        cube([W, D, H]);
        translate([0.006, 0.006, 0.006]) cube([W - 0.012, D - 0.012, H]);
    }

part("lid", density = 1200)
    color("sienna")
    translate([0, 0, H]) cube([W, D, T]);

// A beam rather than a full slab, so it catches the lid without hiding it.
part("shelf", fixed = true)
    color("slategray")
    translate([-0.05, 0.10, SHELF_Z]) cube([W + 0.10, 0.10, 0.02]);

// The pivot is written once, in the coordinates the model is drawn in. Both
// parts share it, because both parts are drawn in the same space.
hinge("lid_pivot",
      parts = ["lid", "box"],
      at    = [W / 2, 0, H],
      axis  = [1, 0, 0],
      range = [0, OPEN]);

// Not a pose — a request. The lid opens as far as it can, which is not as far
// as this asks: the shelf is in the way.
drive("lid_pivot", to = OPEN, over = 1.5);
