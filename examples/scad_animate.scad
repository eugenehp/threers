// A four-jaw chuck that opens and closes over one loop of $t — the kind of
// motion OpenSCAD's animation mode exists for, and what `scad_animate` renders.
//
// Open it in OpenSCAD and press Animate, or render it here:
//   cargo run --release --example scad_animate --features "openscad,video,native-codec"

$fn = 56;

// One out-and-back sweep per loop, eased so the ends do not snap.
stroke  = (1 - cos(360 * $t)) / 2;   // 0 → 1 → 0
opening = 10 + 10 * stroke;          // jaw face distance from the axis
spin    = 360 * $t;                  // the whole chuck turns once

body_r  = 34;    // outer radius
body_h  = 14;    // disc thickness
bore_r  = 9;
jaw_w   = 13;
jaw_h   = 20;
jaw_l   = 22;
slot_d  = 7;     // how deep the jaw channels are cut into the top face

// A jaw: a block on a tongue, with a stepped gripping face and a lightening
// pocket. It slides radially with `opening`.
module jaw() {
    color("gainsboro")
    translate([opening, 0, body_h - slot_d]) {
        difference() {
            union() {
                translate([0, -jaw_w / 2, 0]) cube([jaw_l, jaw_w, slot_d + jaw_h]);
                // Tongue that stays captive in the channel.
                translate([-8, -jaw_w / 2 + 2, 0]) cube([8, jaw_w - 4, slot_d]);
            }
            // Stepped gripping face.
            translate([-1, -jaw_w / 2 - 1, slot_d + jaw_h - 7]) cube([6, jaw_w + 2, 8]);
            translate([-1, -jaw_w / 2 - 1, slot_d + jaw_h - 14]) cube([3, jaw_w + 2, 8]);
            // Pocket.
            translate([jaw_l - 14, -jaw_w / 2 + 3, slot_d + 4])
                cube([9, jaw_w - 6, jaw_h - 10]);
        }
    }
}

// The body: a disc with a bore, four radial channels in its top face, and a
// bolt circle.
module body() {
    color("steelblue")
    difference() {
        cylinder(h = body_h, r = body_r);
        translate([0, 0, -1]) cylinder(h = body_h + 2, r = bore_r);
        for (a = [0 : 90 : 359])
            rotate([0, 0, a])
                translate([bore_r - 3, -jaw_w / 2 - 0.4, body_h - slot_d])
                    cube([body_r - bore_r + 1, jaw_w + 0.8, slot_d + 1]);
        for (a = [45 : 90 : 359])
            rotate([0, 0, a])
                translate([body_r - 8, 0, -1]) cylinder(h = body_h + 2, r = 2.6);
    }
}

// A held workpiece, present only while the jaws are closed on it.
module blank() {
    if (opening < 14)
        color([0.85, 0.45, 0.2, 0.85])
            translate([0, 0, body_h - slot_d]) cylinder(h = 46, r = opening - 0.2);
}

rotate([0, 0, spin]) {
    body();
    blank();
    for (a = [0 : 90 : 359]) rotate([0, 0, a]) jaw();
}
