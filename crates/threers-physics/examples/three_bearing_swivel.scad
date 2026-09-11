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
// This file is the *external* arrangement: each pinion sits beside its ring gear
// and the pair counter-rotates. `three_bearing_swivel_internal.scad` is the same
// mechanism with the pinion inside the ring instead; everything both of them
// share lives in the body file, which is where to read the model.
include <three_bearing_swivel_body.scad>

swivel_module(internal = false);
