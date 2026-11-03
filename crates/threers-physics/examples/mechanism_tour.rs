//! Everything the mechanism and assembly layers do, on one small bench.
//!
//! ```text
//! cargo run -p threers-physics --features assembly,openscad --example mechanism_tour
//! cargo run -p threers-physics --features assembly,openscad --example mechanism_tour -- --render
//! ```
//!
//! The model is `mechanism_tour.scad` beside this file: a base, a crank geared
//! to a wheel, a carriage on a rail, a bolt in a thread, a lid that is asked for
//! more travel than it has, and one part attached to nothing at all. It declares
//! all of that itself, in the coordinates it is drawn in.
//!
//! Seven acts, in the order you would actually do them:
//!
//! 1. read what the model says about itself
//! 2. put it together and check it, before simulating anything
//! 3. ask how far the lid can really open
//! 4. run it, and read the joints back
//! 5. record the run as poses, and reduce it to keyframes
//! 6. ask the geometry whether it agrees with the declaration
//! 7. measure the parts with the generic predicates

use threers::assembly as geo;
use threers::materials::{BasicMaterial, Material};
use threers::math::Color;
use threers_physics::assembly::Assembly;
use threers_physics::mechanism::{ScadMechanism, Tolerance};
use threers_physics::verify::Verify;

const FRAMES: usize = 240;
const FPS: u32 = 60;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mechanism_tour.scad");

    // ---- 1. what the model says about itself -------------------------------
    let spec = match threers::parse_scad_mechanism_file(path) {
        Ok(spec) => spec,
        Err(e) => return println!("could not read the model: {e}"),
    };

    println!("== declared ==");
    println!("  {} parts, {} mates, {} drives", spec.parts.len(), spec.mates.len(), spec.drives.len());
    for mate in &spec.mates {
        println!(
            "    {:<12} {:?}  {} on {}",
            mate.name, mate.kind, mate.parts[0], mate.parts[1]
        );
    }
    for drive in &spec.drives {
        match drive.to {
            Some(to) => println!(
                "    drive {:<12} to {to} over {}s, starting at {}s",
                drive.mate,
                drive.over.unwrap_or(0.0),
                drive.start
            ),
            // No target: a motor. The only thing that works for a shaft making
            // full turns, since a hinge angle wraps at ±360°.
            None => println!(
                "    drive {:<12} at {}°/s, continuously",
                drive.mate,
                drive.max_speed.unwrap_or(0.0)
            ),
        }
    }
    // A mate naming a part that does not exist is the commonest way to get a
    // mechanism wrong, and otherwise shows up much later as a joint that does
    // nothing.
    if !spec.dangling_parts().is_empty() {
        println!("  !! mates name parts that do not exist: {:?}", spec.dangling_parts());
    }

    // ---- 2. put it together, and check it ----------------------------------
    let mut asm = match Assembly::from_scad(&spec) {
        Ok(asm) => asm,
        Err(e) => return println!("the declarations do not line up: {e}"),
    };
    // The model drew its parts in place, so this normally moves nothing — and
    // says so loudly when the drawing and the declarations disagree.
    if let Err(e) = asm.solve() {
        return println!("the model and its declarations disagree: {e}");
    }

    let report = asm.check();
    println!("== check ==");
    println!("  mobility: {} degree(s) of freedom", report.mobility);
    println!("  interferences: {}", report.interferences.len());
    for hit in &report.interferences {
        println!(
            "    {} into {} by {:.3}",
            asm.part_of(hit.a).unwrap().name,
            asm.part_of(hit.b).unwrap().name,
            hit.depth
        );
    }
    // Two triangle meshes have no volume between them to measure, so the pair is
    // reported rather than passed. A check that could not run is not a check
    // that passed.
    for (a, b) in &report.unchecked {
        println!(
            "    untestable: {} / {} — both are surfaces",
            asm.part_of(*a).unwrap().name,
            asm.part_of(*b).unwrap().name
        );
    }
    if !report.ignored_limits.is_empty() {
        println!("    {} mate(s) given a range they cannot use", report.ignored_limits.len());
    }

    // ---- 3. how far can the lid really open? -------------------------------
    let hinge = asm.mate_named("lid_pivot").expect("the model declares it");
    let asked = spec.mate("lid_pivot").unwrap().range.unwrap()[1];
    let predicted = match asm.sweep(hinge, 96) {
        Some(sweep) => {
            let reach = sweep.clear.1.to_degrees();
            match sweep.blocker {
                Some((_, on)) => println!(
                    "== travel ==\n  {reach:.0}° of the {asked:.0}° specified — fouls on {}",
                    asm.part_of(on).unwrap().name
                ),
                None => println!("== travel ==\n  the full {asked:.0}°, clear"),
            }
            reach
        }
        None => {
            println!("== travel ==\n  not a mate that can be swept");
            asked
        }
    };

    // ---- 4. run it ---------------------------------------------------------
    let mut mech = match ScadMechanism::from_file(path) {
        // `millimetres` is not just gravity: the solver's slop, recovery speed
        // and rest thresholds are lengths too, tuned for a scene in metres.
        Ok(m) => m.frames(FRAMES).fps(FPS).millimetres(),
        Err(e) => return println!("could not build the mechanism: {e}"),
    };
    // A fine thread is a stiff constraint — a millimetre of travel demands five
    // radians of turn — and stiff constraints want a shorter step.
    mech.world.substeps = 8;

    // A pose per part per frame, not a copy of the model per frame.
    let track = mech.record();

    println!("== after {:.1}s ==", FRAMES as f32 / FPS as f32);
    for (name, unit) in [
        ("crank_axle", "rad"),
        ("wheel_axle", "rad"),
        ("feed", "mm"),
        ("thread", "mm"),
        ("lid_pivot", "rad"),
    ] {
        let mate = mech.assembly.mate_named(name).unwrap();
        println!(
            "  {name:<11} {:>9.3} {unit:<4} at {:>8.3} {unit}/s",
            mech.assembly.coordinate(mate, &mech.world).unwrap_or(0.0),
            mech.assembly.speed(mate, &mech.world).unwrap_or(0.0),
        );
    }
    let crank = mech.assembly.mate_named("crank_axle").unwrap();
    let wheel = mech.assembly.mate_named("wheel_axle").unwrap();
    let (fast, slow) = (
        mech.assembly.speed(crank, &mech.world).unwrap_or(0.0),
        mech.assembly.speed(wheel, &mech.world).unwrap_or(0.0),
    );
    println!("  the gear holds {:.2}:1", if slow != 0.0 { fast / slow } else { 0.0 });

    let reached = mech
        .assembly
        .coordinate(hinge, &mech.world)
        .unwrap_or(0.0)
        .to_degrees();
    println!(
        "  the lid was asked for {asked:.0}°, swept to {predicted:.0}°, and reached {reached:.0}°"
    );

    // ---- 5. what the run costs, and what it reduces to ---------------------
    let mut arena = threers::core::ObjectArena::new();
    let nodes = track.spawn(&mut arena, None, grey());
    let keys = |c: &threers::animation::AnimationClip| -> usize {
        c.tracks.iter().map(|t| t.times.len()).sum()
    };
    let all = keys(&track.to_clip(&nodes, "run"));
    let thin = keys(&track.to_clip_simplified(&nodes, "run", Tolerance::default()));
    println!("== the run ==");
    println!(
        "  {} frames as {:.1} kB of poses, or {:.1} MB had every frame been baked",
        track.frames(),
        track.parts().iter().map(|p| p.poses.len() * 7 * 4).sum::<usize>() as f64 / 1e3,
        track.baked_bytes() as f64 / 1e6
    );
    println!(
        "  as an AnimationClip: {all} keys, {thin} once reduced ({:.0}× fewer)",
        all as f64 / thin.max(1) as f64
    );
    // Played back through the ordinary mixer, with no physics anywhere.
    println!("  replayed through the mixer: {}", replay(&track, &nodes));

    // ---- 6. does the geometry agree with the declaration? ------------------
    println!("== verify ==");
    let mut checked = match ScadMechanism::from_file(path) {
        Ok(m) => m.frames(FRAMES).fps(FPS).millimetres(),
        Err(e) => return println!("  could not rebuild: {e}"),
    };
    checked.world.substeps = 8;
    let found = Verify::new().run(&mut checked);
    println!("  {} mate(s) confirmed over {} poses", found.confirmed, found.poses);
    for finding in &found.findings {
        println!("  {finding}");
    }
    if found.agrees() {
        println!("  the geometry agrees with the declaration");
    }

    // The check that finds real drift: the declaration against a model's own
    // `$t` animation, with no solver in between.
    drift_demo();

    // ---- 7. measure the parts themselves -----------------------------------
    println!("== the parts, measured ==");
    for name in ["base", "crank", "lid"] {
        let Some(part) = mech.assembly.part_named(name).and_then(|p| mech.assembly.part_of(p))
        else {
            continue;
        };
        let Some(mesh) = part.geometry.as_ref() else { continue };
        let tris = geo::from_geometry(mesh);
        let key = geo::rigid_key(&tris);
        println!(
            "  {name:<7} {:>5} triangles  {:>8.0} mm² surface  {:>9.0} mm³  {} shell(s)  hand {:>2}",
            tris.len(),
            geo::area(&tris),
            geo::volume(&tris),
            geo::shells(&tris, 0.0).len(),
            key.handedness
        );
    }
    // The identity survives being moved, which is what makes it usable for
    // finding the same body again in another pose.
    if let Some(part) = mech.assembly.part_named("crank").and_then(|p| mech.assembly.part_of(p)) {
        if let Some(mesh) = part.geometry.as_ref() {
            let here = geo::from_geometry(mesh);
            let moved = geo::from_geometry_at(mesh, &shifted());
            println!(
                "  the crank's identity survives being moved and turned: {}",
                geo::rigid_key(&here).matches(&geo::rigid_key(&moved))
            );
        }
    }

    // ---- render ------------------------------------------------------------
    if std::env::args().any(|a| a == "--render") {
        render(&track);
    } else {
        println!("== render ==");
        println!("  pass --render to write a PNG sequence to out/mechanism_tour/");
    }
}

/// Play the reduced clip through the ordinary mixer and report where it lands.
fn replay(
    track: &threers_physics::mechanism::PoseTrack,
    _nodes: &[threers::core::ObjectId],
) -> String {
    use threers::animation::{AnimationMixer, LoopMode};

    let mut scene = threers::scene::Scene::new();
    let root = scene.root;
    // Spawned into the scene's own arena: a clip targets nodes, and they have to
    // be the nodes the mixer will find.
    let nodes = track.spawn(&mut scene.arena, Some(root), grey());
    let clip = track.to_clip_simplified(&nodes, "run", Tolerance::default());

    let mut mixer = AnimationMixer::new();
    let action = mixer.clip_action(clip);
    // A run is a one-shot: it opens the lid, it does not open it forever.
    mixer.actions[action].loop_mode = LoopMode::Once;
    for _ in 0..track.frames() {
        mixer.update(&mut scene, 1.0 / FPS as f32);
    }

    let Some(lid) = track.parts().iter().position(|p| p.name == "lid") else {
        return "no lid".into();
    };
    let played = scene.arena.get(nodes[lid]).map(|o| o.quaternion);
    let simulated = track.pose(lid, track.frames() - 1).map(|p| p.rotation);
    match (played, simulated) {
        (Some(a), Some(b)) if a.dot(b).abs() > 0.999 => {
            "ends where the simulation ended".into()
        }
        _ => "ends somewhere else".into(),
    }
}

/// A model whose `rotate()` and whose `hinge()` disagree, and the check that
/// notices.
fn drift_demo() {
    let model = |pivot_z: f32| {
        format!(
            r#"
            part("box", fixed = true) cube([30, 20, 16]);
            part("lid") translate([0, 0, 16]) rotate([0, 90 * $t, 0]) cube([30, 20, 1.2]);
            hinge("lid_pivot", parts = ["lid", "box"],
                  at = [0, 0, {pivot_z}], axis = [0, 1, 0], range = [0, 90]);
            "#
        )
    };
    // The drawing swings the lid about z = 16. Told the truth, the check passes;
    // told 14, it reports the two-unit difference. Neither model looks wrong on
    // its own, and nothing else in either one notices.
    let checker = Verify::new().axis_tolerance(1e-4);
    for (label, z) in [("declared truthfully", 16.0), ("declared at z = 14", 14.0)] {
        match checker.against_model(&model(z), 0.0, 1.0) {
            Ok(report) if report.agrees() => println!("  {label}: agrees"),
            Ok(report) => {
                for f in report.faults() {
                    println!("  {label}: {f}");
                }
            }
            Err(e) => println!("  {label}: {e}"),
        }
    }
}

fn grey() -> Material {
    Material::Basic(BasicMaterial::new(Color::from_hex(0xcccccc)))
}

/// Some rigid motion, for showing that an identity survives one.
fn shifted() -> threers::math::Matrix4 {
    let iso = threers_physics::math::Isometry::new(
        threers::math::Vector3::new(500.0, -200.0, 90.0),
        threers::math::Quaternion::from_axis_angle(
            threers::math::Vector3::new(0.3, 0.9, 0.2).normalize(),
            1.1,
        ),
    );
    iso.to_matrix4()
}

/// Bake a frame at a time and render, so the whole sequence is never in memory.
fn render(track: &threers_physics::mechanism::PoseTrack) {
    use threers::openscad::animate::{ScadCamera, ScadRender};

    println!("== render ==");
    let renderer = ScadRender::new(960, 540).camera(ScadCamera::Auto {
        yaw: 38.0,
        pitch: 26.0,
        zoom: 0.9,
    });
    let sampled: Vec<_> = (0..track.frames()).step_by(8).map(|i| track.frame(i)).collect();
    let images = match renderer.render_evaluated(&sampled) {
        Ok(images) => images,
        Err(e) => return println!("  no renderer available here: {e}"),
    };

    let dir = std::path::Path::new("out/mechanism_tour");
    if let Err(e) = std::fs::create_dir_all(dir) {
        return println!("  cannot create {}: {e}", dir.display());
    }
    for (i, rgba) in images.iter().enumerate() {
        let png = threers::utils::png::encode_png(960, 540, rgba);
        if let Err(e) = std::fs::write(dir.join(format!("frame{i:03}.png")), png) {
            return println!("  cannot write: {e}");
        }
    }
    println!("  wrote {} frames to {}", images.len(), dir.display());
}
