//! A `.scad` model that describes its own mechanism, simulated.
//!
//! ```text
//! cargo run -p threers-physics --features assembly,openscad --example scad_mechanism
//! ```
//!
//! The model is `scad_mechanism.scad` beside this file. It declares its parts
//! and its hinge in the same coordinates it is drawn in, and asks the lid to
//! open 105°. It does not get 105°, because there is a shelf over the box — and
//! the interesting part is that the *static* check and the *simulation* agree
//! about how far it does get, having been asked in completely different ways.
//!
//! Nothing here sets a transform. The lid's pose in every frame is one the
//! solver produced.

use threers_physics::assembly::Assembly;
use threers_physics::mechanism::ScadMechanism;
use threers_physics::verify::Verify;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/scad_mechanism.scad");

    // ---- 1. what the model says about itself -------------------------------
    let spec = match threers::parse_scad_mechanism_file(path) {
        Ok(spec) => spec,
        Err(e) => {
            println!("could not read the model: {e}");
            return;
        }
    };

    println!("-- declared --");
    for part in &spec.parts {
        println!(
            "  part {:<8} {}",
            part.name,
            if part.fixed { "fixed" } else { "moving" }
        );
    }
    for mate in &spec.mates {
        println!(
            "  {:?} {:<10} {} on {} about {:?}",
            mate.kind, mate.name, mate.parts[0], mate.parts[1], mate.axis
        );
    }
    for drive in &spec.drives {
        match drive.to {
            Some(to) => println!(
                "  drive {:<10} to {to}° over {}s",
                drive.mate,
                drive.over.unwrap_or(0.0)
            ),
            None => println!(
                "  drive {:<10} at {}°/s, continuously",
                drive.mate,
                drive.max_speed.unwrap_or(0.0)
            ),
        }
    }
    if !spec.dangling_parts().is_empty() {
        println!("  !! mates name parts that do not exist: {:?}", spec.dangling_parts());
    }

    // ---- 2. check it, before simulating anything ---------------------------
    let mut asm = match Assembly::from_scad(&spec) {
        Ok(asm) => asm,
        Err(e) => {
            println!("the declarations do not line up: {e}");
            return;
        }
    };
    if let Err(e) = asm.solve() {
        println!("the model and its declarations disagree: {e}");
        return;
    }

    let report = asm.check();
    println!("-- check --");
    println!("  mobility: {} degree(s) of freedom", report.mobility);
    println!("  interferences: {}", report.interferences.len());
    for hit in &report.interferences {
        println!(
            "    {} into {} by {:.4}",
            asm.part_of(hit.a).unwrap().name,
            asm.part_of(hit.b).unwrap().name,
            hit.depth
        );
    }

    let hinge = asm.mate_named("lid_pivot").expect("the model declares it");
    let declared = spec.mate("lid_pivot").unwrap().range.unwrap()[1];
    let predicted = match asm.sweep(hinge, 96) {
        Some(sweep) => {
            let reach = sweep.clear.1.to_degrees();
            match sweep.blocker {
                Some((_, on)) => println!(
                    "-- travel --\n  {reach:.0}° of the {declared:.0}° specified — fouls on {}",
                    asm.part_of(on).unwrap().name
                ),
                None => println!("-- travel --\n  the full {declared:.0}°, clear"),
            }
            reach
        }
        None => {
            println!("-- travel --\n  not a mate that can be swept");
            declared
        }
    };

    // ---- 3. simulate it ----------------------------------------------------
    let mut mech = match ScadMechanism::from_file(path) {
        Ok(m) => m.frames(240).fps(60),
        Err(e) => {
            println!("could not build the mechanism: {e}");
            return;
        }
    };

    println!("-- simulating {} frames --", 240);
    let frames = mech.simulate();
    let hinge = mech.assembly.mate_named("lid_pivot").unwrap();
    let reached = mech
        .assembly
        .coordinate(hinge, &mech.world)
        .unwrap_or(0.0)
        .to_degrees();

    println!("  frames produced: {}", frames.len());
    println!(
        "  triangles per frame: {} (unchanged across the run)",
        frames[0].triangle_count()
    );
    println!("  asked for:  {declared:.0}°");
    println!("  swept to:   {predicted:.0}°   (static check, before any simulation)");
    println!("  reached:    {reached:.0}°   (the solver, 240 steps later)");
    println!(
        "  the two agree to within {:.1}°",
        (predicted - reached).abs()
    );

    // ---- 4. ask the geometry what it did -----------------------------------
    //
    // Everything above trusts the declaration. This does not: it reads the axis
    // back out of the triangles that moved and compares. Against a simulation
    // that confirms the pipeline rather than the model — the joints were built
    // from the declaration, so they hold the parts where it said. What it does
    // find on its own is anything about contact: a joint that comes apart, or a
    // part mated to nothing at all.
    println!("-- verify --");
    let mut checked = match ScadMechanism::from_file(path) {
        Ok(m) => m.frames(240).fps(60),
        Err(e) => {
            println!("  could not rebuild the mechanism: {e}");
            return;
        }
    };
    let report = Verify::new().run(&mut checked);
    println!(
        "  {} mate(s) confirmed over {} poses",
        report.confirmed, report.poses
    );
    if report.agrees() {
        println!("  the geometry agrees with the declaration");
    }
    for finding in &report.findings {
        println!("  {finding}");
    }

    // ---- 5. render it ------------------------------------------------------
    //
    // The frames are ordinary `ScadFrame`s, so the render path is the one that
    // already existed — `render_evaluated` cannot tell that these came from a
    // solver rather than from re-evaluating the model per frame.
    if std::env::args().any(|a| a == "--render") {
        render(&frames);
    } else {
        println!("-- render --");
        println!("  pass --render to write a PNG sequence to out/scad_mechanism/");
    }
}

/// Render the simulated frames through the ordinary SCAD render path.
fn render(frames: &[threers::openscad::animate::ScadFrame]) {
    use threers::openscad::animate::{ScadCamera, ScadRender};

    println!("-- render --");
    // A fixed three-quarter view, not a turntable: the subject here is the
    // mechanism moving, and an orbiting camera makes it hard to tell which of
    // the two is doing the moving.
    let renderer = ScadRender::new(960, 540).camera(ScadCamera::Auto {
        yaw: 35.0,
        pitch: 18.0,
        zoom: 0.85,
    });

    // Every eighth frame: enough to see the lid stop against the shelf without
    // writing 240 files.
    let sampled: Vec<_> = frames.iter().step_by(8).cloned().collect();
    let images = match renderer.render_evaluated(&sampled) {
        Ok(images) => images,
        Err(e) => {
            println!("  no renderer available here: {e}");
            return;
        }
    };

    let dir = std::path::Path::new("out/scad_mechanism");
    if let Err(e) = std::fs::create_dir_all(dir) {
        println!("  cannot create {}: {e}", dir.display());
        return;
    }
    for (i, rgba) in images.iter().enumerate() {
        let png = threers::utils::png::encode_png(960, 540, rgba);
        let path = dir.join(format!("frame{i:03}.png"));
        if let Err(e) = std::fs::write(&path, png) {
            println!("  cannot write {}: {e}", path.display());
            return;
        }
    }
    println!("  wrote {} frames to {}", images.len(), dir.display());
}
