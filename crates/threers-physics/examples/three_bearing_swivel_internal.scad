// A Three-Bearing Swivel Module, drawn once and simulated from the same file.
//
//   cargo run -p threers-physics --features assembly,openscad --release \
//       --example three_bearing_swivel
//
// The 3BSM is the exhaust nozzle of a STOVL fighter: three rotary bearings
// whose mating faces are cut obliquely through the duct. In cruise every
// bearing sits at zero and the module is a straight pipe. Turn them and the
// oblique cuts fold the duct over until the jet points 95 degrees down, which
// is what holds the aircraft up in the hover.
//
// Nothing here is a pose. The bearing angles are what the solver produces from
// the servo targets the controller asks for, and the controller gets those
// from inverse kinematics on the chain these hinges describe.
//
// ---------------------------------------------------------------------------
// This file is the *internal* arrangement: each pinion runs inside a ring gear
// whose teeth face in, carried on a web out past the pinion, and the pair turns
// the same way rather than counter-rotating. `three_bearing_swivel.scad` is the
// same mechanism with the pinion outside the ring; everything both of them share
// lives in the body file, which is where to read the model.
//
// Worth knowing before reading further: this is *not* the compact one. An
// internal pair usually is, because the pinion tucks inside the annulus. Here
// the annulus already has a duct in it, so the pinion has to sit between the two
// and the ring is pushed out to a 0.768 m pitch radius against the external
// arrangement's 0.600. What it buys is a bigger radius to take the load at, and
// a mesh that does not reverse the drive.
include <three_bearing_swivel_body.scad>

swivel_module(internal = true);
