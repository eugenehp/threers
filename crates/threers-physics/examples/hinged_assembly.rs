//! Mechanical assembly: a hinged lid, checked and then driven through the solver.
//!
//! ```text
//! cargo run -p threers-physics --features assembly --example hinged_assembly
//! ```
//!
//! Four things happen here that the raw joint API cannot do on its own:
//!
//! 1. The hinge is declared on each part in *the part's own coordinates*, and
//!    the assembly moves the parts until those two lines are the same line.
//! 2. The check reports whether anything overlaps and how many degrees of
//!    freedom the mechanism has.
//! 3. The sweep drives the lid through its declared travel and finds where it
//!    fouls — before a single step is simulated.
//! 4. The move is authored as a target, not a pose, so the lid stops when it
//!    reaches its stop, and stops early when something is in the way.

use threers::core::BufferGeometry;
use threers::geometries::BoxGeometry;
use threers_physics::assembly::deg;
use threers_physics::prelude::*;

/// Box outside dimensions, in metres.
const BOX: [f32; 3] = [0.30, 0.16, 0.20];
const LID: [f32; 3] = [0.30, 0.012, 0.20];
/// How far the lid is meant to open.
const OPEN: f32 = 105.0;

fn main() {
    let (mut asm, hinge, lid) = build();

    // ---- 1. put it together ------------------------------------------------
    //
    // The lid starts nowhere near its hinge; the solve is what brings it in.
    asm.place_at(lid, [0.0, 0.40, 0.10]);
    match asm.solve() {
        Ok(()) => println!("-- assembled --"),
        Err(e) => {
            println!("could not assemble: {e}");
            return;
        }
    }
    let placed = asm.part_of(lid).unwrap().transform.translation;
    println!(
        "  the lid moved from (0.00, 0.40, 0.10) to ({:.3}, {:.3}, {:.3})",
        placed.x, placed.y, placed.z
    );

    // ---- 2. check it -------------------------------------------------------
    let report = asm.check();
    println!("-- check --");
    println!("  mobility: {} degree(s) of freedom", report.mobility);
    println!("  interferences: {}", report.interferences.len());
    for hit in &report.interferences {
        println!(
            "    {} into {} by {:.4} m",
            asm.part_of(hit.a).unwrap().name,
            asm.part_of(hit.b).unwrap().name,
            hit.depth
        );
    }
    if !report.unchecked.is_empty() {
        println!("  {} pair(s) could not be tested", report.unchecked.len());
    }
    println!("  clear: {}", report.clear());

    // ---- 3. sweep it -------------------------------------------------------
    //
    // The lid is specified to open 105°. Whether it *can* is a different
    // question, and this is the one place to ask it cheaply.
    println!("-- travel --");
    report_sweep(&asm, hinge, "on its own");

    // Now hang a shelf over the box and ask again. The lid's tip swings on a
    // 0.20 radius from a hinge at y = 0.08, so a shelf whose underside sits at
    // 0.21 catches it around 40°.
    let shelf = asm.part_from_geometry(
        "shelf",
        block([0.60, 0.02, 0.40]),
        PartPhysics::fixed().fit(ColliderFit::Box),
    );
    asm.place_at(shelf, [0.0, 0.22, 0.0]);
    report_sweep(&asm, hinge, "under a shelf");

    // ---- 4. drive it -------------------------------------------------------
    let mut world = World::new();
    asm.build(&mut world).expect("the assembly builds");

    println!("-- opening --");
    asm.drive(hinge).to(deg(OPEN)).over(1.5);

    for frame in 0..300 {
        asm.update(1.0 / 60.0, &mut world);
        world.step(1.0 / 60.0);

        if frame % 30 == 29 {
            let angle = asm.coordinate(hinge, &world).unwrap_or(0.0);
            println!(
                "  {:5.2}s  {:6.1}°",
                (frame + 1) as f32 / 60.0,
                angle.to_degrees()
            );
        }
    }

    // A drive that has run its time and not arrived is a drive that met
    // something. An animation curve cannot tell you this, because it would have
    // arrived regardless.
    let angle = asm.coordinate(hinge, &world).unwrap_or(0.0);
    let shortfall = deg(OPEN) - angle;
    if shortfall > deg(1.0) {
        println!(
            "  asked for {OPEN:.0}°, got {:.1}° — {:.1}° short, held off by the shelf",
            angle.to_degrees(),
            shortfall.to_degrees()
        );
    } else {
        println!("  reached {:.1}°, resting on its own stop", angle.to_degrees());
    }

    // ---- and it falls shut on its own --------------------------------------
    println!("-- closing --");
    asm.drive(hinge).to(0.0).over(1.0);
    for _ in 0..240 {
        asm.update(1.0 / 60.0, &mut world);
        world.step(1.0 / 60.0);
    }
    println!(
        "  back to {:.1}°, resting on the stop",
        asm.coordinate(hinge, &world).unwrap_or(0.0).to_degrees()
    );
}

/// A box and its lid, hinged along the back top edge.
fn build() -> (Assembly, MateId, PartId) {
    let mut asm = Assembly::new();

    let body = asm.part_from_geometry("box", block(BOX), PartPhysics::fixed());
    let lid = asm.part_from_geometry(
        "lid",
        block(LID),
        // Printed plastic, near enough.
        PartPhysics::dynamic().density(1200.0).friction(0.8),
    );

    // The pivot on each part, in that part's own coordinates. The box's is on
    // its back top edge; the lid's is on its own back underside. Neither refers
    // to the other, and neither is a world coordinate.
    let back = -BOX[2] * 0.5;
    let hinge = asm.mate(
        Mate::hinge(
            Feature::axis(lid, Axis::new([0.0, -LID[1] * 0.5, back], [-1.0, 0.0, 0.0])),
            Feature::axis(body, Axis::new([0.0, BOX[1] * 0.5, back], [-1.0, 0.0, 0.0])),
        )
        // Closed at 0, and the stop at 105° is the hinge's own, not a limit of
        // the animation.
        .limits(0.0, deg(OPEN))
        .named("lid hinge"),
    );

    (asm, hinge, lid)
}

fn report_sweep(asm: &Assembly, hinge: MateId, label: &str) {
    match asm.sweep(hinge, 64) {
        Some(sweep) => {
            print!(
                "  {label}: opens to {:.0}° of the {:.0}° specified",
                sweep.clear.1.to_degrees(),
                sweep.declared.1.to_degrees()
            );
            match sweep.blocker {
                Some((_, on)) => println!(" — fouls on {}", asm.part_of(on).unwrap().name),
                None => println!(" — clear"),
            }
        }
        None => println!("  {label}: not a mate that can be swept"),
    }
}

fn block(size: [f32; 3]) -> BufferGeometry {
    BoxGeometry::new(size[0], size[1], size[2])
}
