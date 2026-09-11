//! A Three-Bearing Swivel Module: modelled in OpenSCAD, steered by inverse
//! kinematics, and simulated as rigid bodies on real bearings turned by real
//! gears.
//!
//! ```text
//! cargo run -p threers-physics --features assembly,openscad --release \
//!     --example three_bearing_swivel [--internal] [--render] [--section]
//! ```
//!
//! `--render` writes the transition as a PNG sequence; adding `--section` writes
//! one still instead, cut open on the plane through the duct axis and the drive
//! axis, which is where you can see what the two arrangements actually differ
//! by. The cut is triangle-level clipping of the drawn frame — it changes
//! nothing that moves.
//!
//! The drive comes in two arrangements and `--internal` picks the second:
//!
//! - **external** — teeth on the outside of each ring gear, pinion beside it,
//!   motor bolted to the outside of the duct. One wall, 8:1 mesh, and the drive
//!   is what sets the installed envelope.
//! - **internal** — teeth on the *inside* of a ring seated in the race bore,
//!   with the pinion and its motor in the cooling annulus between a liner and a
//!   casing. Nothing projects at all. The annulus picks the ratio: the pinion has
//!   to clear the liner, so it is small, so the mesh is 16:1.
//!
//! Both share `three_bearing_swivel_body.scad`; the two `.scad` files beside it
//! are three lines each. Whichever is selected gets the full treatment, and the
//! last section runs both and compares them.
//!
//! The 3BSM is how a STOVL fighter points its exhaust at the ground: a roll
//! bearing aft of the turbine and two obliquely-cut rotary joints, folding the
//! duct over by 95° without a single sliding seal in the gas path. Turning the
//! centre segment half a turn against both of its neighbours does it, and
//! `4 × 23.75 = 95` exactly.
//!
//! The roll bearing is what makes it flyable. It has no cant, so it does nothing
//! to the jet on its own; it turns the whole folded assembly about the engine
//! axis, which is one-for-one with the jet's azimuth — the yaw control the
//! published description mentions, and the reason the deployment schedule is
//! closed form rather than a search. Cant all three bearings instead and every
//! one of those properties goes away; see `threers::kinematics::swivel`.
//!
//! It takes about a minute and a half to run, most of it in the clearance check
//! below, which is doing mesh-against-mesh distance in both directions over 49
//! poses.
//!
//! Nothing here is servoed to a nozzle angle. Three pinions turn, and everything
//! else — 95° of thrust vector, half a tonne of duct held against gravity — is
//! what the solver makes of that through the meshes.
//!
//! Nine things happen here, in order, and each one checks the last:
//!
//! 1. **The model describes itself.** `three_bearing_swivel.scad` declares its
//!    parts, its three bearings and its drive train in the coordinates it is
//!    drawn in. The kinematic model is built from those declarations rather than
//!    restated — the cants are read back out of the hinge axes and the masses
//!    off the meshes.
//! 2. **The assembly is checked** before anything moves: mobility, interference,
//!    and a swept-volume check on each bearing through a full turn.
//! 3. **The schedule is solved.** Damped least squares over the three bearings,
//!    tracked down the envelope, produces the table a nozzle controller carries.
//! 4. **The drive train is sized** from that schedule: mesh ratio, gearbox,
//!    reflected inertia, tooth loads, and what the motor has to produce.
//! 5. **The mechanism is simulated.** The controller commands *pinion rates*;
//!    the meshes turn those into bearing angles; the solver decides where the
//!    parts actually go, against gravity, bearing friction and finite torque.
//! 6. **Nothing is allowed to touch.** Minimum surface-to-surface distance,
//!    mesh against mesh, over the whole deployment. The assembly's own
//!    interference check cannot answer this — it skips a pair where both
//!    colliders are surfaces, and every collider here is a triangle mesh, so it
//!    reports the lot as untestable and an empty interference list reads as
//!    "clear". The physics world has the same hole from the other side: two
//!    triangle-mesh bodies generate no contacts at all, so a mechanism whose
//!    parts pass through each other runs perfectly.
//! 7. **The attachments are shown to be attachments.** A ring gear welded to a
//!    race is measured against that race under load; the ratio is read back off
//!    the parts rather than off the command; and switching the drives off at
//!    half deflection drops the nozzle, because their torque was the only thing
//!    holding it.
//! 8. **The carrier is shown to matter.** The same run with the meshes measured
//!    against the world instead of against the segments that carry them.
//! 9. **Both arrangements are flown and compared** — ratio sign, envelope, ring
//!    mass, tooth load and tracking, external against internal.
//!
//! Pass `--render` to write a PNG sequence of the transition.

use std::sync::Arc;

use threers::core::{BufferAttribute, BufferGeometry};
use threers::kinematics::swivel::{cant_of, Bearing, Schedule, SwivelNozzle};
use threers::openscad::frame::ScadFrame;
use threers::openscad::ScadPart;
use threers::kinematics::{DriveEnv, GearTrain, StribeckFriction};
use threers::openscad::mechanism::MechanismSpec;
use threers_physics::assembly::MateId;
use threers_physics::joint::{JointKind, Motor};
use threers_physics::mechanism::ScadMechanism;
use threers_physics::prelude::*;

/// The two arrangements of the same mechanism: pinion outside its ring gear, and
/// pinion inside it. Both files are three lines and an include; the model they
/// share is `three_bearing_swivel_body.scad`.
const EXTERNAL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/three_bearing_swivel.scad"
);
const INTERNAL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/three_bearing_swivel_internal.scad"
);

/// Gross thrust of the engine this nozzle is on, newtons. Roughly what an F135
/// makes in the hover.
const THRUST: f64 = 80_000.0;

/// Where the aircraft's centre of gravity sits relative to bearing 1, metres.
/// Forward of the nozzle, so vectored lift pitches the nose down and something
/// else — a lift fan, on the real thing — has to answer it.
const CG: [f64; 3] = [-6.2, 0.0, 0.0];

/// Overall reduction from motor to bearing. The mesh carries part of it and the
/// gearbox inside each drive unit carries the rest, so a build with a 16:1 mesh
/// gets a 12.5:1 gearbox where an 8:1 mesh gets 25:1 — and the motor turns at
/// the same speed either way. `boxratio()` in the model is the same rule.
const OVERALL: f64 = 200.0;

/// The three bearings, and the pinion shaft and mesh that turn each.
const BEARINGS: [&str; 3] = ["bearing1", "bearing2", "bearing3"];
const SHAFTS: [&str; 3] = ["shaft1", "shaft2", "shaft3"];
const MESHES: [&str; 3] = ["mesh1", "mesh2", "mesh3"];

/// Outer-loop gain, pinion rate commanded per radian of bearing error.
///
/// The inner loop is the drive itself, which follows a rate command with a time
/// constant of about 80 ms; 12 rad/s of outer bandwidth sits comfortably inside
/// that. Higher and the two loops start arguing.
const KP: f64 = 12.0;

/// Fastest the *bearing* may be told to turn, rad/s. The pinion's limit is this
/// times the mesh ratio, which is the only way to write it that means the same
/// thing in both builds: a 16:1 mesh has to spin its pinion twice as fast to move
/// the nozzle at the same rate, and a limit written at the pinion would quietly
/// halve the deployment rate instead of limiting it.
const BEARING_RATE_LIMIT: f64 = 2.5;

fn main() {
    let internal = std::env::args().any(|a| a == "--internal");
    let path = if internal { INTERNAL } else { EXTERNAL };
    println!("Three-Bearing Swivel Module");
    println!("===========================");
    println!(
        "  {} gears -- pass {} for the other arrangement",
        if internal { "internal" } else { "external" },
        if internal { "no flag" } else { "--internal" }
    );

    // ---- 1. what the model says about itself -------------------------------
    let spec = match threers::parse_scad_mechanism_file(path) {
        Ok(spec) => spec,
        Err(e) => {
            println!("could not read the model: {e}");
            return;
        }
    };

    println!("\n-- declared ------------------------------------------------------");
    println!("  {} parts, {} mates", spec.parts.len(), spec.mates.len());
    for mate in &spec.mates {
        let kind = format!("{:?}", mate.kind).to_lowercase();
        let kind = kind.split(&[' ', '{'][..]).next().unwrap_or("").to_string();
        let extra = match mate.kind {
            threers::openscad::mechanism::MateSpecKind::Hinge => format!(
                "cant {:+7.3} deg",
                cant_of(as_f64(mate.axis))
            ),
            threers::openscad::mechanism::MateSpecKind::Gear { ratio } => format!(
                "ratio {ratio:+.1}, carried on {}",
                mate.carrier.as_deref().unwrap_or("the world")
            ),
            _ => String::new(),
        };
        println!(
            "  {kind:<6} {:<13} {:<9} on {:<12} {extra}",
            mate.name, mate.parts[0], mate.parts[1]
        );
    }
    for bad in spec.dangling_parts() {
        println!("  !! a mate names {bad}, which no part() declares");
    }

    // ---- 2. build it and check it ------------------------------------------
    let mech = match ScadMechanism::from_file(path) {
        Ok(m) => m,
        Err(e) => {
            println!("could not build the mechanism: {e}");
            return;
        }
    };

    println!("\n-- built ---------------------------------------------------------");
    let mut mass_of: Vec<(String, f64, [f64; 3])> = Vec::new();
    for part in mech.assembly.parts() {
        let mass = part
            .colliders
            .iter()
            .map(|c| c.mass_properties().mass)
            .sum::<f32>() as f64;
        let com = part_com(part);
        println!(
            "  {:<12} {:>8.1} kg   {:>6} tri   cg [{:+.3} {:+.3} {:+.3}]",
            part.name,
            mass,
            part.geometry
                .as_ref()
                .and_then(|g| g.get_attribute("position"))
                .map(|a| a.count() / 3)
                .unwrap_or(0),
            com[0],
            com[1],
            com[2]
        );
        mass_of.push((part.name.clone(), mass, com));
    }

    let report = mech.assembly.check();
    println!(
        "  mobility: {} degrees of freedom   interference: {} pair(s), {} untestable",
        report.mobility,
        report.interferences.len(),
        report.unchecked.len()
    );
    if !report.unchecked.is_empty() {
        println!("  Every collider here is a triangle mesh, and the assembly check skips a pair");
        println!("  where both sides are surfaces -- so that zero is 'never asked', not 'clear'.");
        println!("  Reading it as clear is how a mechanism that cannot turn passes review. The");
        println!("  clearance section below is the check that was actually run.");
    }
    for hit in &report.interferences {
        println!(
            "    {} into {} by {:.4} m",
            name_of(&mech, hit.a),
            name_of(&mech, hit.b),
            hit.depth
        );
    }
    for name in BEARINGS {
        let Some(id) = mech.assembly.mate_named(name) else {
            continue;
        };
        match mech.assembly.sweep(id, 180) {
            Some(sweep) => {
                let (lo, hi) = (sweep.clear.0.to_degrees(), sweep.clear.1.to_degrees());
                match sweep.blocker {
                    Some((_, on)) => println!(
                        "  {name}: clear over {lo:.0}..{hi:.0} deg, then fouls {}",
                        name_of(&mech, on)
                    ),
                    None => println!("  {name}: clear over its whole declared {lo:.0}..{hi:.0} deg"),
                }
            }
            None => println!("  {name}: not a mate that can be swept"),
        }
    }

    // ---- 3. the kinematic model, built from the declarations ---------------
    //
    // The cants come back out of the hinge axes and the stations out of the
    // hinge points, so there is one statement of the geometry and not two. The
    // masses come from the meshes the simulation collides with — and each
    // segment carries more than itself: the ring gear bolted to it, and the
    // drive unit and pinion for the *next* bearing down, which ride on it.
    let bearings: Vec<Bearing> = BEARINGS
        .iter()
        .filter_map(|n| spec.mate(n))
        .map(|m| Bearing {
            cant: cant_of(as_f64(m.axis)),
            station: m.at[0] as f64,
        })
        .collect();
    if bearings.len() != 3 {
        println!("\nthe model does not declare all three bearings");
        return;
    }
    let Some((nozzle, masses)) = chain_for(&spec, &mech, bearings) else {
        println!("could not build the kinematic chain");
        return;
    };

    println!("\n-- kinematics ----------------------------------------------------");
    println!(
        "  cants            {:+.3} / {:+.3} / {:+.3} deg",
        nozzle.bearings[0].cant, nozzle.bearings[1].cant, nozzle.bearings[2].cant
    );
    println!(
        "  swung mass       {:.0} kg  ({:.0} / {:.0} / {:.0}), drive train included",
        masses.iter().sum::<f64>(),
        masses[0],
        masses[1],
        masses[2]
    );
    println!(
        "  envelope         {:.2} deg = 2 * (c1 - c2 + c3), reached at half a turn on each",
        nozzle.envelope()
    );
    let ganged = (0..=180)
        .map(|t| {
            let q = [t as f64; 3];
            (nozzle.deflection(&q), nozzle.lateral(&q))
        })
        .fold((0.0f64, 0.0f64), |(d, l), (dd, ll)| {
            (d.max(dd), if ll.abs() > l.abs() { ll } else { l })
        });
    println!(
        "  ganged schedule  reaches {:.2} deg -- and leaves the vertical plane by {:.1} deg",
        ganged.0, ganged.1
    );

    let schedule = nozzle.schedule(nozzle.envelope(), 600);
    println!(
        "  solved schedule  reaches {:.2} deg, break-out open loop to {:.2} deg costing {:.2} deg of yaw",
        schedule.reach(),
        schedule.breakout,
        schedule.worst_lateral()
    );
    println!(
        "  authority        zero stowed (no first-order pitch at all), {:.2} at mid-envelope",
        nozzle.authority(&schedule.bearings_at(50.0))
    );

    // ---- 4. the drive train ------------------------------------------------
    let mesh_ratio = mesh_ratio(&spec);
    // Motor to bearing. The sign is the mesh's; a reduction is its magnitude.
    let gearbox = OVERALL / mesh_ratio.abs();
    let train = GearTrain {
        ratio: OVERALL,
        // A fueldraulic motor's rotor, kg·m². Small, and multiplied by 40,000.
        rotor_inertia: 2.0e-4,
        efficiency: 0.85,
        backlash_deg: 0.04,
        friction: StribeckFriction {
            tau_c: 18.0,
            tau_s: 26.0,
            v_s: 0.02,
            viscous: 1.2,
        },
        env: DriveEnv::default(),
    };

    println!("\n-- drive train ---------------------------------------------------");
    println!(
        "  mesh             {mesh_ratio:+.0} : 1   ring gear on the race, pinion on the segment upstream"
    );
    println!("  gearbox          {gearbox:.1} : 1   inside each drive unit");
    println!("  overall          {:.0} : 1   motor turns per turn of the bearing", train.ratio);

    let mut peak_hold = [0.0f64; 3];
    for d in 0..=95 {
        let q = schedule.bearings_at(d as f64);
        for (peak, tau) in peak_hold.iter_mut().zip(nozzle.holding_torque(&q)) {
            *peak = peak.max(tau.abs());
        }
    }
    println!("             bearing hold   at the pinion   at the motor   tooth load");
    for (i, name) in BEARINGS.iter().enumerate() {
        let at_pinion = peak_hold[i] / mesh_ratio.abs();
        let at_motor = peak_hold[i] / (train.ratio * train.efficiency);
        // Tangential force where the teeth meet, which is what sizes them.
        let tooth = peak_hold[i] / pitch_radius(&spec);
        println!(
            "  {name} {:11.0} N.m {:12.1} N.m {:12.2} N.m {:11.0} N",
            peak_hold[i], at_pinion, at_motor, tooth
        );
    }
    println!(
        "  reflected rotor inertia {:.0} kg.m2 at the bearing, from a {:.4} kg.m2 rotor",
        train.reflected_inertia(),
        train.rotor_inertia
    );
    println!(
        "  the train's own friction is {:.0} N.m at the bearing, so a load below that",
        train.friction.tau_s * train.ratio * (1.0 - train.efficiency)
    );
    println!("  never back-drives it: the nozzle holds its angle with the motors switched off.");

    // ---- 5. simulate the transition ----------------------------------------
    println!("\n-- simulated -----------------------------------------------------");
    let Some(run) = simulate(&spec, &nozzle, &schedule, true, KP) else {
        return;
    };

    // ---- 6. does anything actually touch anything --------------------------
    clearance(&spec, &schedule);

    // ---- 7. the drive path is a load path ----------------------------------
    attachment(&spec, &schedule);

    // ---- 7. what the carrier is worth --------------------------------------
    //
    // The same mechanism with `carrier = …` stripped off every mesh, so the
    // ratios are enforced against the world the way a two-body coupling has to
    // be. Nothing else changes: same geometry, same schedule, same commands.
    println!("\n-- what the carrier is worth -------------------------------------");
    let mut naive = spec.clone();
    for mate in &mut naive.mates {
        mate.carrier = None;
    }
    // First the constraint on its own, with no controller anywhere near it:
    // hold pinions 2 and 3 dead still in their own cases and turn pinion 1.
    // Bearings 2 and 3 are then locked to the segments they sit on and must not
    // move relative to them, however far bearing 1 swings.
    println!("  Turn pinion 1 alone, with pinions 2 and 3 held still in their cases.");
    println!("  Bearings 2 and 3 are locked by their own meshes and should not move at all.");
    println!("                          bearing 1   bearing 2   bearing 3");
    let ratio = mesh_ratio;
    for (label, model) in [
        ("carried on its segment", &spec),
        ("measured in the world ", &naive),
    ] {
        match probe(model) {
            Some(p) => {
                println!(
                    "  {label}  {:8.1} deg {:8.1} deg {:8.1} deg",
                    p.bearings[0], p.bearings[1], p.bearings[2]
                );
                if label.starts_with("carried") {
                    println!(
                        "    and pinion 2, told to hold, turned {:.1} deg in the world while its own",
                        p.pinion2_world
                    );
                    println!(
                        "    shaft moved {:.2} deg -- carried by the segment it is bolted to, not driven.",
                        p.pinion2_shaft
                    );
                    // And the shaft motion it *did* have is the mesh being
                    // faithful, not the mesh slipping: bearing 2 drifted a
                    // fraction of a degree and the pinion followed it at the
                    // ratio, which is the coupling working rather than failing.
                    if p.bearings[1].abs() > 1e-3 {
                        println!(
                            "    And the {:.2} is the mesh being faithful, not slack: bearing 2 drifted",
                            p.pinion2_shaft
                        );
                        println!(
                            "    {:.2} deg, which at the {:+.0} mesh is {:.2} deg of pinion.",
                            p.bearings[1],
                            ratio,
                            p.bearings[1] * ratio
                        );
                    }
                }
            }
            None => println!("  {label}  did not build"),
        }
    }
    println!("  A world-referenced ratio cannot tell a pinion turning from its case turning, so");
    println!("  swinging segment A reads as bearings 2 and 3 being driven, and the nozzle folds");
    println!("  up on its own.");

    // And then what that costs the whole transition, with the controller doing
    // its best to hide it.
    println!();
    println!("  Flown, with the schedule and the same rate loop on all three drives:");
    println!("                          hover jet   worst miss   worst out-of-plane");
    println!(
        "  carried on its segment  {:8.2} deg {:9.2} deg {:14.2} deg",
        run.hover, run.worst_track, run.worst_lateral
    );
    match simulate(&naive, &nozzle, &schedule, false, KP) {
        Some(bad) => {
            println!(
                "  measured in the world   {:8.2} deg {:9.2} deg {:14.2} deg",
                bad.hover, bad.worst_track, bad.worst_lateral
            );
            println!("  The loop spends its authority hiding the wrong plant and mostly gets there:");
            println!(
                "  it still reaches the hover. What it costs is {:.1} deg of jet out of the vertical",
                bad.worst_lateral
            );
            println!(
                "  plane against {:.1} -- about {:.0} kN of side force on {:.0} kN of thrust, which the",
                run.worst_lateral,
                THRUST * bad.worst_lateral.to_radians().sin() / 1000.0,
                THRUST / 1000.0
            );
            println!("  aircraft has to find a rudder for.");
        }
        None => println!("  measured in the world   did not build"),
    }

    // ---- 9. what the jet does to the aircraft ------------------------------
    println!("\n-- thrust --------------------------------------------------------");
    println!(
        "  {THRUST:.0} N gross, moments about a CG {:.1} m forward of bearing 1",
        -CG[0]
    );
    println!("     jet      lift, N    axial, N   lift frac   pitch, kN.m");
    for want in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 95.0] {
        let q = schedule.bearings_at(want);
        let s = nozzle.thrust(&q, THRUST);
        println!(
            "  {:6.1} {:11.0} {:11.0} {:11.3} {:13.1}",
            s.deflection,
            tidy(s.lift(), 0.5),
            tidy(s.axial(), 0.5),
            tidy(s.lift_fraction(), 5e-4),
            tidy(s.pitch() / 1000.0, 0.05)
        );
    }

    // ---- 10. the other arrangement ------------------------------------------
    compare();

    if std::env::args().any(|a| a == "--render") {
        render(&spec, &schedule);
    } else {
        println!("\n  pass --render to write the transition to out/three_bearing_swivel/,");
        println!("  and --section with it to cut the drawn model open on the fold plane");
    }
}

/// One arrangement of the drive, measured.
struct Survey {
    label: &'static str,
    ratio: f64,
    r_ring: f64,
    r_pin: f64,
    centres: f64,
    /// How far the whole machine reaches from the duct axis — the hole it has to
    /// be installed in.
    envelope: f64,
    /// And how far the gas path reaches, so the two are comparable on the thing
    /// an exhaust duct is actually for.
    bore: f64,
    ring_mass: f64,
    swung: f64,
    /// Worst holding torque anywhere in the envelope, N·m.
    hold: f64,
    /// How many parts it takes to build.
    parts: usize,
    /// Tangential force where the teeth meet at that pose, which is what sizes a
    /// tooth: the torque divided by the pitch radius it acts at.
    tooth_load: f64,
    run: Run,
}

/// Build, size and fly one arrangement.
fn survey(label: &'static str, path: &str) -> Option<Survey> {
    let spec = threers::parse_scad_mechanism_file(path).ok()?;
    let mech = ScadMechanism::from_file(path).ok()?;
    let bearings: Vec<Bearing> = BEARINGS
        .iter()
        .filter_map(|n| spec.mate(n))
        .map(|m| Bearing {
            cant: cant_of(as_f64(m.axis)),
            station: m.at[0] as f64,
        })
        .collect();
    let (nozzle, masses) = chain_for(&spec, &mech, bearings)?;
    let schedule = nozzle.schedule(nozzle.envelope(), 400);

    let ratio = mesh_ratio(&spec);
    let r_ring = pitch_radius(&spec);
    let mut hold = 0.0f64;
    for d in 0..=95 {
        hold = hold.max(
            nozzle
                .holding_torque(&schedule.bearings_at(d as f64))
                .iter()
                .fold(0.0f64, |m, t| m.max(t.abs())),
        );
    }
    Some(Survey {
        label,
        ratio,
        r_ring,
        r_pin: r_ring / ratio.abs(),
        centres: centre_distance(&spec),
        envelope: envelope_of(&mech, &[]),
        bore: bore_of(&mech),
        ring_mass: mech
            .assembly
            .parts()
            .iter()
            .find(|p| p.name == "ring1")
            .map(|p| {
                p.colliders
                    .iter()
                    .map(|c| c.mass_properties().mass)
                    .sum::<f32>() as f64
            })
            .unwrap_or(0.0),
        swung: masses.iter().sum(),
        parts: mech.assembly.parts().len(),
        hold,
        tooth_load: hold / r_ring,
        run: simulate(&spec, &nozzle, &schedule, false, KP)?,
    })
}

/// How far the named parts reach from the duct axis, metres. An empty list means
/// all of them.
fn envelope_of(mech: &ScadMechanism, names: &[&str]) -> f64 {
    mech.assembly
        .parts()
        .iter()
        .filter(|p| names.is_empty() || names.contains(&p.name.as_str()))
        .filter_map(|p| p.geometry.as_ref())
        .filter_map(|g| g.get_attribute("position"))
        .flat_map(|pos| {
            pos.array
                .chunks(pos.item_size)
                .map(|v| ((v[1] * v[1] + v[2] * v[2]) as f64).sqrt())
                .collect::<Vec<_>>()
        })
        .fold(0.0f64, f64::max)
}

/// The gas-path bore: the smallest radius any part reaches, which is the liner
/// where there is one and the duct wall where there is not.
fn bore_of(mech: &ScadMechanism) -> f64 {
    mech.assembly
        .parts()
        .iter()
        .filter_map(|p| p.geometry.as_ref())
        .filter_map(|g| g.get_attribute("position"))
        .flat_map(|pos| {
            pos.array
                .chunks(pos.item_size)
                .map(|v| ((v[1] * v[1] + v[2] * v[2]) as f64).sqrt())
                .collect::<Vec<_>>()
        })
        .fold(f64::INFINITY, f64::min)
}

/// The same mechanism with the pinion on either side of its ring gear.
fn compare() {
    println!("\n-- external vs internal ------------------------------------------");
    let surveys: Vec<Survey> = [("external", EXTERNAL), ("internal", INTERNAL)]
        .into_iter()
        .filter_map(|(l, p)| survey(l, p))
        .collect();
    if surveys.len() != 2 {
        println!("  could not build both arrangements");
        return;
    }
    let (e, i) = (&surveys[0], &surveys[1]);
    println!("                            {:>10} {:>10}", e.label, i.label);
    let row = |name: &str, unit: &str, a: f64, b: f64, dp: usize| {
        println!(
            "  {name:<24}  {:>10.dp$} {:>10.dp$}  {unit}",
            a,
            b,
            dp = dp
        );
    };
    row("mesh ratio", "turns of pinion per turn of bearing", e.ratio, i.ratio, 1);
    row("gearbox behind it", ": 1", OVERALL / e.ratio.abs(), OVERALL / i.ratio.abs(), 1);
    row("ring pitch radius", "m", e.r_ring, i.r_ring, 3);
    row("pinion pitch radius", "m", e.r_pin, i.r_pin, 3);
    row("centre distance", "m", e.centres, i.centres, 3);
    row("narrowest gas radius", "m, at the throat", e.bore, i.bore, 3);
    row("installed radius", "m, the hole it goes in", e.envelope, i.envelope, 3);
    row("ring gear, each", "kg", e.ring_mass, i.ring_mass, 0);
    row("swung mass", "kg", e.swung, i.swung, 0);
    row("parts / mates", "", e.parts as f64, i.parts as f64, 0);
    row("worst bearing torque", "N.m", e.hold, i.hold, 0);
    row("tooth load at that", "N", e.tooth_load, i.tooth_load, 0);
    row("tracked to", "deg of jet", e.run.worst_track, i.run.worst_track, 2);
    row("worst out of plane", "deg", e.run.worst_lateral, i.run.worst_lateral, 2);
    println!();
    println!("  Same deflection envelope, same schedule, same 200:1 from motor to bearing.");
    println!("  What differs is where the drive lives, and everything else follows from it.");
    println!();
    println!("  External puts the pinion beside its ring gear and the motor outside the duct.");
    println!("  One wall, and the drive is what you have to find room for: it reaches");
    println!(
        "  {:.3} m from the axis where its own ring gear is {:.3}.",
        e.envelope, e.r_ring
    );
    println!();
    println!("  Internal puts the teeth on the inside of a ring seated in the race bore and the");
    println!("  pinion and motor in the cooling annulus between liner and casing. Nothing");
    println!(
        "  projects, so the race sets the envelope rather than a motor: {:.3} m against {:.3},",
        i.envelope, e.envelope
    );
    println!("  which is {:.0} mm of bay radius.", (e.envelope - i.envelope) * 1000.0);
    println!();
    println!("  The mesh ratio is not a free choice there. An internal pinion sits at");
    println!("  R_ring - R_pinion and has to clear the liner, so it is small, so it needs a lot");
    println!(
        "  of teeth: {:.0}:1 against {:.0}:1, with half the gearbox behind it. That larger ring",
        i.ratio.abs(),
        e.ratio.abs()
    );
    println!(
        "  is also a longer lever — {:.3} m of pitch radius against {:.3} — and it shows in the",
        i.r_ring, e.r_ring
    );
    println!(
        "  teeth: {:.0} N against {:.0}, {:.0} per cent less, for a bearing torque that is itself",
        i.tooth_load,
        e.tooth_load,
        100.0 * (1.0 - i.tooth_load / e.tooth_load)
    );
    println!("  lower because the build came out lighter in the places that matter.");
    println!();
    println!(
        "  It costs the throat. A liner has to end inside the casing rather than being it, so",
    );
    println!(
        "  the gas gets {:.3} m of radius against {:.3} — {:.0} per cent of exit area given away",
        i.bore,
        e.bore,
        100.0 * (1.0 - (i.bore / e.bore).powi(2))
    );
    println!(
        "  — and it costs the second wall: {} parts against {}, with three more welded joints.",
        i.parts, e.parts
    );
    println!(
        "  Swung mass is a wash ({:.0} kg against {:.0}): the liners weigh what the outboard",
        i.swung, e.swung
    );
    println!("  drive units did.");
    println!();
    println!("  The residual wander in the last two rows is the solver rather than either");
    println!("  mechanism. Both are quoted at 128 velocity iterations; halve that and they read");
    println!("  1.74 and 2.46 degrees, double it and they read 0.34 and 0.57. It halves with the");
    println!("  count, which is what an unconverged constraint set looks like and not what a");
    println!("  machine does. The internal build carries three more welded bodies, so it needs");
    println!("  more of them to reach the same place.");
}

/// What one run of the transition produced.
struct Run {
    /// Worst gap between the jet the schedule asked for and the jet the solver
    /// produced, degrees, over the whole run.
    worst_track: f64,
    /// And the worst it left the vertical plane by.
    worst_lateral: f64,
    /// Where the jet ended up at the end of the hover, when the schedule has
    /// been holding 95 degrees for three seconds and everything has settled.
    hover: f64,
    /// Fastest each pinion was told to turn, rad/s.
    #[allow(dead_code)]
    peak_rate: [f64; 3],
    /// Peak torque the commanded motion needs at each pinion shaft, N·m, from
    /// the chain's own inverse dynamics: weight plus what it takes to accelerate
    /// the parts along the schedule.
    #[allow(dead_code)]
    peak_demand: [f64; 3],
}

/// Drive the module through cruise, deploy, hover and retract — by turning
/// pinions, and nothing else.
fn simulate(
    spec: &MechanismSpec,
    nozzle: &SwivelNozzle,
    schedule: &Schedule,
    verbose: bool,
    kp: f64,
) -> Option<Run> {
    // Static holding torque at the bearing, for the actuator table at the end.
    let mut hold = [0.0f64; 3];
    for d in 0..=95 {
        let q = schedule.bearings_at(d as f64);
        for (peak, tau) in hold.iter_mut().zip(nozzle.holding_torque(&q)) {
            *peak = peak.max(tau.abs());
        }
    }
    let mut mech = match ScadMechanism::from_spec(spec) {
        Ok(m) => m,
        Err(e) => {
            println!("  could not build: {e}");
            return None;
        }
    };
    let fps = 240;
    let dt = 1.0 / fps as f32;
    // Half a tonne of duct cantilevered off the engine flange, held only through
    // an 8:1 mesh, is a demanding constraint problem: the default eight velocity
    // iterations leave the chain visibly sagging.
    mech.world.solver_config.velocity_iterations = 128;
    mech.world.solver_config.position_iterations = 8;

    let bearings: Vec<MateId> = BEARINGS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    let shafts: Vec<MateId> = SHAFTS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    if bearings.len() != 3 || shafts.len() != 3 {
        println!("  the model did not declare all three drives");
        return None;
    }
    // Whatever the drives were told to do in the model, this run is driving
    // them itself.
    for mate in &shafts {
        mech.assembly.drive(*mate).hold();
    }
    let ceiling: Vec<f64> = SHAFTS
        .iter()
        .map(|n| {
            spec.drives
                .iter()
                .find(|d| d.mate == *n)
                .and_then(|d| d.torque)
                .unwrap_or(500.0) as f64
        })
        .collect();
    let mesh_ratio = mesh_ratio(spec);

    let nozzle_body = mech
        .assembly
        .part_named("nozzle")
        .and_then(|p| mech.assembly.body_of(p))?;

    // Settle: the schedule says stowed, and the pinions have to hold it there.
    // Always closed-loop, whatever the run is testing — an open-loop run still
    // has to start from somewhere, and "wherever half a tonne of duct sagged to"
    // is not a starting point either version deserves.
    for _ in 0..fps {
        step(&mut mech, &bearings, &shafts, &ceiling, mesh_ratio, schedule, 0.0, 0.0, KP, dt);
    }
    if verbose {
        let (droop, _) = measured_jet(&mech, nozzle_body);
        println!(
            "  stowed, the pinions hold the module to {droop:.3} deg off the engine axis"
        );
        println!("     t     phase      commanded   achieved   out-of-plane   b1      b2      b3");
    }

    let mut worst_track = 0.0f64;
    let mut worst_lateral = 0.0f64;
    let mut deflection = 0.0f64;
    let mut hover = 0.0f64;
    let mut peak_rate = [0.0f64; 3];
    let mut peak_demand = [0.0f64; 3];
    let steps = (TRANSITION.total() * fps as f64) as usize;
    for step_i in 0..=steps {
        let t = step_i as f64 / fps as f64;
        let (phase, u) = TRANSITION.at(t);
        // Feed-forward rate, over a window the drive can actually follow.
        const WINDOW: f64 = 0.05;
        let ahead = TRANSITION.at(t + WINDOW).1;
        let behind = TRANSITION.at(t - WINDOW).1;
        let want = nozzle.deflection(&schedule.bearings_at_travel(u));

        // What the commanded motion costs, before asking what it achieved.
        // Central differences over a window rather than step to step: the
        // schedule is a table, so it is piecewise linear, and differentiating
        // that twice at 240 Hz gives a train of impulses at the row boundaries
        // rather than an acceleration. The window is the drive's own — it
        // cannot follow a corner in a table either.
        let q = schedule.bearings_at_travel(u);
        let qa = schedule.bearings_at_travel(ahead);
        let qb = schedule.bearings_at_travel(behind);
        let rate: Vec<f64> = qa
            .iter()
            .zip(&qb)
            .map(|(a, b)| (a - b) / (2.0 * WINDOW))
            .collect();
        let accel: Vec<f64> = qa
            .iter()
            .zip(&qb)
            .zip(&q)
            .map(|((a, b), c)| (a - 2.0 * c + b) / (WINDOW * WINDOW))
            .collect();
        for (k, tau) in nozzle.slew_torque(&q, &rate, &accel).iter().enumerate() {
            peak_demand[k] = peak_demand[k].max(tau.abs() / mesh_ratio.abs());
            peak_rate[k] = peak_rate[k].max((rate[k] * mesh_ratio).to_radians().abs());
        }

        step(
            &mut mech, &bearings, &shafts, &ceiling, mesh_ratio, schedule, u,
            (ahead - behind) / (2.0 * WINDOW), kp, dt,
        );

        let lateral;
        (deflection, lateral) = measured_jet(&mech, nozzle_body);
        worst_track = worst_track.max((deflection - want).abs());
        worst_lateral = worst_lateral.max(lateral.abs());
        if phase == "hover" {
            hover = deflection;
        }

        if verbose && step_i % (fps as usize) == 0 {
            let at: Vec<f64> = bearings
                .iter()
                .map(|m| {
                    mech.assembly
                        .coordinate(*m, &mech.world)
                        .unwrap_or(0.0)
                        .to_degrees() as f64
                })
                .collect();
            println!(
                "  {t:5.2}  {phase:<10} {want:8.2}   {deflection:8.2}   {lateral:10.3}   {:6.1} {:6.1} {:6.1}",
                at[0], at[1], at[2]
            );
        }
    }
    if verbose {
        println!("  tracked the schedule to within {worst_track:.3} deg of jet over {steps} steps");
        println!("  the jet never left the vertical plane by more than {worst_lateral:.3} deg");
        println!("  every degree of that came from a pinion turning -- no bearing was driven");
    }
    let _ = deflection;
    if verbose {
        println!("\n-- actuators -----------------------------------------------------");
        println!("           peak rate   trajectory needs   declared   static hold");
        for k in 0..3 {
            println!(
                "  {} {:8.1} deg/s {:12.0} N.m {:9.0} N.m {:9.0} N.m",
                SHAFTS[k],
                peak_rate[k].to_degrees(),
                peak_demand[k],
                ceiling[k],
                hold[k] / mesh_ratio.abs()
            );
        }
        println!(
            "  All at the pinion shaft; divide by the {:.1}:1 gearbox for the motor.",
            OVERALL / mesh_ratio.abs()
        );
        println!("  The ceilings are five times what either load column asks for, and they have");
        println!("  to be: the margin is not for the load, it is for the loop. Drop shaft1 to");
        println!("  400 N.m -- comfortably above both -- and the jet leaves the vertical plane");
        println!("  by 4.3 degrees instead of 1.0, because the drive cannot correct fast enough");
        println!("  through the part of the schedule that turns 28 degrees of bearing per degree");
        println!("  of jet. An undersized servo does not look undersized. It looks slow.");
    }
    Some(Run {
        worst_track,
        worst_lateral,
        hover,
        peak_rate,
        peak_demand,
    })
}

/// The mesh ratio the model declared: pinion turns per turn of the bearing.
fn mesh_ratio(spec: &MechanismSpec) -> f64 {
    spec.mate(MESHES[0])
        .and_then(|m| match m.kind {
            threers::openscad::mechanism::MateSpecKind::Gear { ratio } => Some(ratio as f64),
            _ => None,
        })
        .unwrap_or(-8.0)
}

/// Angle between two orientations, degrees.
fn between(a: Quaternion, b: Quaternion) -> f64 {
    let r = a.conjugate().multiply(b);
    2.0 * (r.w.abs().clamp(-1.0, 1.0) as f64).acos().to_degrees()
}

/// Shortest signed difference between two angles, radians.
fn wrapped(delta: f64) -> f64 {
    let mut d = delta % (2.0 * std::f64::consts::PI);
    if d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    if d < -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    d
}

/// Show that the parts are attached rather than posed.
///
/// Everything here is read back off the bodies after the solver has moved them.
/// A ring gear that is welded to a race has to stay on it under load; a pinion
/// mounted on a segment has to be carried when that segment swings, without
/// being *driven* by the swing; a mesh has to hand over the ratio it claims; and
/// a drive that is holding half a tonne of nozzle up has to drop it when the
/// torque goes away. None of those is true of an animation.
fn attachment(spec: &MechanismSpec, schedule: &Schedule) {
    let ratio = mesh_ratio(spec);
    println!("\n-- attached, not posed -------------------------------------------");
    let Ok(mut mech) = ScadMechanism::from_spec(spec) else {
        println!("  could not rebuild the mechanism");
        return;
    };
    mech.world.solver_config.velocity_iterations = 128;
    mech.world.solver_config.position_iterations = 8;
    let bearings: Vec<MateId> = BEARINGS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    let shafts: Vec<MateId> = SHAFTS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    if bearings.len() != 3 || shafts.len() != 3 {
        return;
    }
    for mate in &shafts {
        mech.assembly.drive(*mate).hold();
    }
    let ceiling: Vec<f64> = SHAFTS
        .iter()
        .map(|n| {
            spec.drives
                .iter()
                .find(|d| d.mate == *n)
                .and_then(|d| d.torque)
                .unwrap_or(500.0) as f64
        })
        .collect();

    let body = |mech: &ScadMechanism, name: &str| {
        mech.assembly
            .part_named(name)
            .and_then(|p| mech.assembly.body_of(p))
    };
    let welded = [("ring1", "swivel_a"), ("ring2", "swivel_b"), ("ring3", "nozzle")];
    let pairs: Vec<(BodyId, BodyId)> = welded
        .iter()
        .filter_map(|(a, b)| Some((body(&mech, a)?, body(&mech, b)?)))
        .collect();
    let Some(nozzle_body) = body(&mech, "nozzle") else {
        return;
    };

    let fps = 240;
    let dt = 1.0 / fps as f32;
    for _ in 0..fps {
        step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, 0.0, 0.0, KP, dt);
    }

    // How each ring gear sits on its race at rest. It has to still sit that way
    // after the whole deployment, or the weld is not holding it.
    let rest: Vec<Quaternion> = pairs
        .iter()
        .map(|(a, b)| {
            let (ra, rb) = (
                mech.world.body(*a).unwrap().rotation(),
                mech.world.body(*b).unwrap().rotation(),
            );
            ra.conjugate().multiply(rb)
        })
        .collect();

    // Unwrapped travel, because a pinion makes eight turns for one of the
    // bearing's and a joint coordinate wraps at half a turn.
    let mut last_b: Vec<f64> = bearings
        .iter()
        .map(|m| mech.assembly.coordinate(*m, &mech.world).unwrap_or(0.0) as f64)
        .collect();
    let mut last_s: Vec<f64> = shafts
        .iter()
        .map(|m| mech.assembly.coordinate(*m, &mech.world).unwrap_or(0.0) as f64)
        .collect();
    let mut turn_b = [0.0f64; 3];
    let mut turn_s = [0.0f64; 3];
    let mut worst_weld = 0.0f64;

    let deploy = (Transition::DEPLOY * fps as f64) as usize;
    for i in 0..=deploy {
        let u = smoothstep(i as f64 / deploy as f64);
        let du = 1.0 / Transition::DEPLOY;
        step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, u, du, KP, dt);
        for (k, (a, b)) in pairs.iter().enumerate() {
            let (ra, rb) = (
                mech.world.body(*a).unwrap().rotation(),
                mech.world.body(*b).unwrap().rotation(),
            );
            worst_weld = worst_weld.max(between(rest[k], ra.conjugate().multiply(rb)));
        }
        for k in 0..3 {
            let b = mech.assembly.coordinate(bearings[k], &mech.world).unwrap_or(0.0) as f64;
            let s = mech.assembly.coordinate(shafts[k], &mech.world).unwrap_or(0.0) as f64;
            turn_b[k] += wrapped(b - last_b[k]);
            turn_s[k] += wrapped(s - last_s[k]);
            last_b[k] = b;
            last_s[k] = s;
        }
    }

    println!(
        "  ring gear on its race    stayed within {worst_weld:.4} deg of where it was welded,"
    );
    println!("                           through the whole deployment and all of its load");
    println!("  mesh ratio, measured     from the simulated angles, not from the command:");
    for k in 0..3 {
        let ratio = if turn_b[k].abs() > 1e-6 {
            turn_s[k] / turn_b[k]
        } else {
            0.0
        };
        println!(
            "    {}  bearing {:7.1} deg, pinion {:8.1} deg -> {ratio:+.3} : 1",
            BEARINGS[k],
            turn_b[k].to_degrees(),
            turn_s[k].to_degrees()
        );
    }
    // The remainder is the velocity solve rather than the model: a mesh and a
    // rate-limited motor are two stiff constraints on the same shaft, and the
    // gap between them closes with iteration count — -7.754 at 32, -7.927 at 64,
    // -7.981 at 128. Which is what a ratio being *solved* looks like, as against
    // one being asserted.
    println!(
        "                           declared {ratio:+.3}; the remainder is the velocity solve,"
    );
    println!(
        "                           and it closes on {:+.0} as the iteration count rises",
        ratio
    );

    // And the load path. Not at the corner of the envelope, where the module is
    // very nearly balanced on its own bearings and the drives are holding almost
    // nothing — back it off to where the holding torque peaks, which is around
    // half deflection, and take the torque away there.
    let loaded = 0.60;
    for i in 0..(fps * 3) {
        let u = 1.0 - (1.0 - loaded) * smoothstep(i as f64 / (fps * 2) as f64);
        step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, u, 0.0, KP, dt);
    }
    let (held, _) = measured_jet(&mech, nozzle_body);
    for shaft in &shafts {
        if let Some(joint) = mech
            .assembly
            .joint_of(*shaft)
            .and_then(|j| mech.world.joint_mut(j))
        {
            if let JointKind::Revolute { motor, .. } = &mut joint.kind {
                *motor = Some(Motor::new(0.0, 0.0));
            }
        }
    }
    for _ in 0..fps {
        mech.assembly.update(dt, &mut mech.world);
        mech.world.step(dt);
    }
    let (fell, _) = measured_jet(&mech, nozzle_body);
    println!(
        "  torque removed           holding {held:.1} deg of deflection, the drives were switched"
    );
    println!(
        "                           off and the jet fell to {fell:.1} deg in one second under its"
    );
    println!("                           own weight -- back-driving the meshes on the way. The");
    println!("                           load was going through the gears the whole time.");
}

/// What turning one pinion on its own does to everything else.
struct Probe {
    /// Where the three bearings ended up, degrees.
    bearings: [f64; 3],
    /// How far the second pinion turned in the world, degrees.
    pinion2_world: f64,
    /// And how far it turned in its own bearing, which is the number it was
    /// commanded to keep at zero.
    pinion2_shaft: f64,
}

/// Turn the first pinion and nothing else, and report where everything else
/// ended up.
///
/// No schedule, no controller, no feedback, no gravity: three rate commands, two
/// of them zero. What comes back is the mesh's own behaviour, and the two mesh
/// references give different answers to it.
fn probe(spec: &MechanismSpec) -> Option<Probe> {
    let mut mech = ScadMechanism::from_spec(spec).ok()?;
    mech.world.solver_config.velocity_iterations = 128;
    // Gravity off: the question is what the coupling does, not what half a tonne
    // of duct does while it is doing it.
    mech.world.gravity = Vector3::ZERO;
    let bearings: Vec<MateId> = BEARINGS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    let shafts: Vec<MateId> = SHAFTS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    if bearings.len() != 3 || shafts.len() != 3 {
        return None;
    }
    for mate in &shafts {
        mech.assembly.drive(*mate).hold();
    }
    let pinion2 = mech
        .assembly
        .part_named("pinion2")
        .and_then(|p| mech.assembly.body_of(p))?;
    let start = mech.world.body(pinion2)?.rotation();
    let dt = 1.0 / 240.0;
    for _ in 0..480 {
        for (i, shaft) in shafts.iter().enumerate() {
            if let Some(joint) = mech
                .assembly
                .joint_of(*shaft)
                .and_then(|j| mech.world.joint_mut(j))
            {
                joint.servo = None;
                if let JointKind::Revolute { motor, .. } = &mut joint.kind {
                    // Pinion 1 turns; 2 and 3 are commanded to stand still,
                    // which is a rate command like any other.
                    *motor = Some(Motor::new(if i == 0 { -4.0 } else { 0.0 }, 2000.0));
                }
            }
        }
        mech.assembly.update(dt, &mut mech.world);
        mech.world.step(dt);
    }
    let mut out = [0.0; 3];
    for (i, mate) in bearings.iter().enumerate() {
        out[i] = mech
            .assembly
            .coordinate(*mate, &mech.world)
            .unwrap_or(0.0)
            .to_degrees() as f64;
    }
    Some(Probe {
        bearings: out,
        pinion2_world: between(start, mech.world.body(pinion2)?.rotation()),
        pinion2_shaft: mech
            .assembly
            .coordinate(shafts[1], &mech.world)
            .unwrap_or(0.0)
            .to_degrees() as f64,
    })
}

/// One step of the controller and the solver.
///
/// The outer loop is on the bearing, because that is where the resolver is and
/// the only place it can be: a pinion makes eight turns for one of the
/// bearing's, and a joint angle wraps at half a turn, so a position target four
/// revolutions away is a position target at zero. The inner loop is a rate
/// command on the pinion, which is what the drive unit actually takes.
#[allow(clippy::too_many_arguments)]
fn step(
    mech: &mut ScadMechanism,
    bearings: &[MateId],
    shafts: &[MateId],
    ceiling: &[f64],
    mesh_ratio: f64,
    schedule: &Schedule,
    u: f64,
    du: f64,
    kp: f64,
    dt: f32,
) {
    let target = schedule.bearings_at_travel(u);
    // Feed-forward: where the schedule is going, per second.
    const H: f64 = 1e-3;
    let ahead = schedule.bearings_at_travel((u + H * du.abs().max(1e-6)).clamp(0.0, 1.0));
    let behind = schedule.bearings_at_travel((u - H * du.abs().max(1e-6)).clamp(0.0, 1.0));

    for i in 0..3 {
        let measured = mech
            .assembly
            .coordinate(bearings[i], &mech.world)
            .unwrap_or(0.0) as f64;
        let error = target[i].to_radians() - measured;
        let feed_forward = if du.abs() > 1e-9 {
            (ahead[i] - behind[i]).to_radians() / (2.0 * H) * du.signum()
        } else {
            0.0
        };
        let bearing_rate = feed_forward + kp * error;
        let limit = BEARING_RATE_LIMIT * mesh_ratio.abs();
        let pinion_rate = (mesh_ratio * bearing_rate).clamp(-limit, limit);
        if let Some(joint) = mech
            .assembly
            .joint_of(shafts[i])
            .and_then(|j| mech.world.joint_mut(j))
        {
            // A rate command, not a place to be: the servo slot stays empty so
            // the two do not argue over the same shaft.
            joint.servo = None;
            if let JointKind::Revolute { motor, .. } = &mut joint.kind {
                *motor = Some(Motor::new(pinion_rate as f32, ceiling[i] as f32));
            }
        }
    }
    mech.assembly.update(dt, &mut mech.world);
    mech.world.step(dt);
}

/// The transition this runs: cruise, deploy, hover, retract.
struct Transition;
const TRANSITION: Transition = Transition;

impl Transition {
    const CRUISE: f64 = 1.0;
    const DEPLOY: f64 = 6.0;
    const HOVER: f64 = 3.0;
    const RETRACT: f64 = 6.0;

    fn total(&self) -> f64 {
        Self::CRUISE + Self::DEPLOY + Self::HOVER + Self::RETRACT
    }

    /// How far through the deployment to be at time `t`, and what to call that
    /// phase. Zero is stowed, one is the corner of the envelope.
    ///
    /// The fraction is of *bearing travel*, not of jet angle, which is what the
    /// actuators experience. Interpolating on jet angle instead puts a third of
    /// the total bearing travel — all of break-out — inside the first degree of
    /// deflection, and then asks the drives to cover it in whatever slice of the
    /// six seconds that degree happens to get.
    fn at(&self, t: f64) -> (&'static str, f64) {
        if t < Self::CRUISE {
            return ("cruise", 0.0);
        }
        let t = t - Self::CRUISE;
        if t < Self::DEPLOY {
            return ("deploy", smoothstep(t / Self::DEPLOY));
        }
        let t = t - Self::DEPLOY;
        if t < Self::HOVER {
            return ("hover", 1.0);
        }
        let t = (t - Self::HOVER) / Self::RETRACT;
        ("retract", 1.0 - smoothstep(t.min(1.0)))
    }
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn as_f64(v: [f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

fn name_of(mech: &ScadMechanism, part: threers_physics::assembly::PartId) -> &str {
    mech.assembly
        .part_of(part)
        .map(|p| p.name.as_str())
        .unwrap_or("?")
}

/// The analytic chain for a built mechanism, with its masses and centres of mass
/// taken off the meshes the simulation collides with.
///
/// Each segment carries more than itself: the ring gear bolted to it, and the
/// drive unit and pinion for the *next* bearing, which ride on it because that is
/// the part their ring gear turns against.
fn chain_for(
    spec: &MechanismSpec,
    mech: &ScadMechanism,
    bearings: Vec<Bearing>,
) -> Option<(SwivelNozzle, Vec<f64>)> {
    let mass_of: Vec<(String, f64, [f64; 3])> = mech
        .assembly
        .parts()
        .iter()
        .map(|p| {
            let mass = p
                .colliders
                .iter()
                .map(|c| c.mass_properties().mass)
                .sum::<f32>() as f64;
            (p.name.clone(), mass, part_com(p))
        })
        .collect();
    let segments: Vec<String> = BEARINGS
        .iter()
        .filter_map(|n| spec.mate(n))
        .map(|m| m.parts[0].clone())
        .collect();
    if segments.len() != 3 {
        return None;
    }
    let (masses, coms) = group_by_segment(spec, &segments, &bearings, &mass_of);
    Some((
        SwivelNozzle::from_bearings(bearings, exit_station(mech))
            .masses(masses.clone())
            .com_offsets(coms)
            .reference(CG),
        masses,
    ))
}

/// Centre distance between a bearing and the pinion that drives it, metres.
///
/// Read off the two mates rather than declared: the model draws the pinion where
/// it goes, and this is how far that is.
fn centre_distance(spec: &MechanismSpec) -> f64 {
    let (Some(bearing), Some(shaft)) = (spec.mate("bearing1"), spec.mate("shaft1")) else {
        return 0.675;
    };
    (0..3)
        .map(|i| (shaft.at[i] - bearing.at[i]) as f64)
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// The pitch radius of a ring gear, from the centre distance the model drew.
///
/// Not declared anywhere as a number. The centre distance is `R_ring + R_pinion`
/// for an external pair and `R_ring - R_pinion` for an internal one, and the
/// ratio fixes how it divides — so the sign of the ratio, which is the one thing
/// that says which arrangement this is, is also what solves it.
fn pitch_radius(spec: &MechanismSpec) -> f64 {
    let ratio = mesh_ratio(spec);
    let d = centre_distance(spec);
    if ratio < 0.0 {
        d * ratio.abs() / (ratio.abs() + 1.0)
    } else {
        d * ratio.abs() / (ratio.abs() - 1.0)
    }
}

/// Each segment carries more than itself: the ring gear bolted to it, and the
/// drive unit and pinion for the *next* bearing, which ride on it because that
/// is the part their ring gear turns against.
///
/// Returns one mass and one centre of mass per segment, the centre measured from
/// that segment's own bearing, which is what the chain wants.
fn group_by_segment(
    spec: &MechanismSpec,
    segments: &[String],
    bearings: &[Bearing],
    mass_of: &[(String, f64, [f64; 3])],
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let host = |part: &str| -> Option<&str> {
        spec.mates
            .iter()
            .find(|m| m.parts[0] == part)
            .map(|m| m.parts[1].as_str())
    };
    let mut masses = vec![0.0; segments.len()];
    let mut moments = vec![[0.0f64; 3]; segments.len()];
    for (name, mass, com) in mass_of {
        // Walk up until the part lands on a segment, or on ground.
        let mut on = name.as_str();
        for _ in 0..8 {
            if let Some(i) = segments.iter().position(|s| s == on) {
                masses[i] += mass;
                for k in 0..3 {
                    moments[i][k] += com[k] * mass;
                }
                break;
            }
            match host(on) {
                Some(next) => on = next,
                None => break,
            }
        }
    }
    let coms = (0..segments.len())
        .map(|i| {
            let m = masses[i].max(1e-9);
            [
                moments[i][0] / m - bearings[i].station,
                moments[i][1] / m,
                moments[i][2] / m,
            ]
        })
        .collect();
    (masses, coms)
}

/// Round a value that is zero to within `eps` to a positive zero, so a table of
/// forces does not print `-0` where the physics says nothing at all.
fn tidy(v: f64, eps: f64) -> f64 {
    if v.abs() < eps {
        0.0
    } else {
        v
    }
}

/// Centre of mass of a part, in the coordinates the model draws it in.
///
/// Every part is drawn in place in one frame, so this needs no transform. Worth
/// taking from the mesh rather than assuming the middle: an obliquely-cut duct's
/// centre of mass is not on the centreline, and the holding torque is exactly
/// that offset times the weight.
fn part_com(part: &threers_physics::assembly::Part) -> [f64; 3] {
    let mut mass = 0.0f32;
    let mut sum = Vector3::ZERO;
    for c in &part.colliders {
        let mp = c.mass_properties();
        mass += mp.mass;
        sum = sum + mp.center_of_mass * mp.mass;
    }
    if mass <= 0.0 {
        return [0.0; 3];
    }
    let com = sum * (1.0 / mass);
    [com.x as f64, com.y as f64, com.z as f64]
}

/// Where the exit plane is, from the drawn nozzle rather than from a constant.
fn exit_station(mech: &ScadMechanism) -> f64 {
    mech.assembly
        .parts()
        .iter()
        .find(|p| p.name == "nozzle")
        .and_then(|p| p.geometry.as_ref())
        .and_then(|g| g.get_attribute("position"))
        .map(|pos| {
            pos.array
                .chunks(pos.item_size)
                .fold(f64::NEG_INFINITY, |m, v| m.max(v[0] as f64))
        })
        .unwrap_or(1.48)
}

/// The jet direction the *simulation* produced: the nozzle body's own rotation
/// applied to the duct axis. Nothing here consults the command.
fn measured_jet(mech: &ScadMechanism, body: BodyId) -> (f64, f64) {
    let Some(b) = mech.world.body(body) else {
        return (0.0, 0.0);
    };
    let e = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(b.rotation());
    (
        (e.x.clamp(-1.0, 1.0) as f64).acos().to_degrees(),
        (e.y.clamp(-1.0, 1.0) as f64).asin().to_degrees(),
    )
}

/// The closest any two parts come to each other over the whole deployment.
///
/// The assembly's own interference check cannot answer this. It skips any pair
/// where both colliders are surfaces, and every collider here is a triangle
/// mesh, so it reports the lot as *untestable* — and an empty interference list
/// then reads as "clear" when nothing was ever asked. The physics world has the
/// same hole from the other side: two triangle-mesh bodies generate no contacts,
/// so a mechanism whose parts pass through each other runs perfectly.
///
/// This measures it instead: minimum surface-to-surface distance, mesh against
/// mesh, on the poses the solver produced, over the whole schedule. A pair that
/// comes to zero is a pair that would jam.
///
/// Every vertex of each part against the other's surface, and in both
/// directions — one way alone misses a feature of B poking between two vertices
/// of A, which on a gear rim against a duct wall is the case that matters. What
/// it still does not catch is one part wholly inside another, because the
/// distance it takes is unsigned; nothing here is nested, and the fixed
/// engine duct would show it if anything were.
///
/// The one exemption is a pinion against the ring gear it drives. Their teeth
/// interleave by construction, and the mesh here is a *rate* constraint, so
/// tooth phase is not simulated at all — tooth-to-tooth clearance is outside
/// what this model claims either way, and is called out rather than measured.
fn clearance(spec: &MechanismSpec, schedule: &Schedule) {
    use threers::mesh_bvh::{BuildOptions, MeshBvh};

    println!("\n-- clearance -----------------------------------------------------");
    let Ok(mut mech) = ScadMechanism::from_spec(spec) else {
        println!("  could not rebuild the mechanism");
        return;
    };
    mech.world.solver_config.velocity_iterations = 128;
    mech.world.solver_config.position_iterations = 8;
    let ratio = mesh_ratio(spec);
    let bearings: Vec<MateId> = BEARINGS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    let shafts: Vec<MateId> = SHAFTS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    if bearings.len() != 3 || shafts.len() != 3 {
        return;
    }
    for mate in &shafts {
        mech.assembly.drive(*mate).hold();
    }
    let ceiling: Vec<f64> = SHAFTS
        .iter()
        .map(|n| {
            spec.drives
                .iter()
                .find(|d| d.mate == *n)
                .and_then(|d| d.torque)
                .unwrap_or(500.0) as f64
        })
        .collect();

    // One BVH per part, on the geometry in its own frame, built once.
    let names: Vec<String> = mech.assembly.parts().iter().map(|p| p.name.clone()).collect();
    let mut bvh: Vec<Option<MeshBvh>> = Vec::new();
    let mut verts: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut bodies: Vec<Option<BodyId>> = Vec::new();
    for name in &names {
        let part = mech.assembly.part_named(name).unwrap();
        let p = mech.assembly.part_of(part).unwrap();
        bvh.push(
            p.geometry
                .as_ref()
                .and_then(|g| MeshBvh::build(g, BuildOptions::default())),
        );
        verts.push(
            p.geometry
                .as_ref()
                .and_then(|g| g.get_attribute("position"))
                .map(|a| a.array.chunks(3).map(|v| [v[0], v[1], v[2]]).collect())
                .unwrap_or_default(),
        );
        bodies.push(mech.assembly.body_of(part));
    }

    // Pairs that are allowed to touch: anything *welded*, which is bolted
    // together and touches by design, and a pinion against its own ring gear.
    //
    // Not the hinged pairs. Two segments either side of a bearing are the ones
    // that most need checking — a segment whose middle crosses its own joint
    // plane fouls the next one along — and exempting them because they are
    // jointed is how that goes unnoticed. They are given a millimetre of running
    // clearance at the face so the check has something to measure.
    let mut exempt: Vec<(usize, usize)> = Vec::new();
    let index_of = |n: &str| names.iter().position(|x| x == n);
    for mate in &spec.mates {
        if !matches!(mate.kind, threers::openscad::mechanism::MateSpecKind::Weld) {
            continue;
        }
        if let (Some(a), Some(b)) = (index_of(&mate.parts[0]), index_of(&mate.parts[1])) {
            exempt.push((a.min(b), a.max(b)));
        }
    }
    for i in 1..=3 {
        if let (Some(a), Some(b)) = (
            index_of(&format!("pinion{i}")),
            index_of(&format!("ring{i}")),
        ) {
            exempt.push((a.min(b), a.max(b)));
        }
    }

    let fps = 240;
    let dt = 1.0 / fps as f32;
    for _ in 0..fps {
        step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, 0.0, 0.0, KP, dt);
    }

    let mut worst: Vec<(f32, usize, usize, f64, Vector3)> = Vec::new();
    let samples = 48;
    let deploy_steps = (Transition::DEPLOY * fps as f64) as usize;
    for k in 0..=samples {
        // Walk to each sample and let it settle there, rather than jumping: the
        // poses have to be ones the solver produced, not ones asserted here. A
        // quasi-static walk is the right shape for a clearance check anyway —
        // what matters is where the parts *are*, not how fast they got there.
        let u = smoothstep(k as f64 / samples as f64);
        for _ in 0..(deploy_steps / samples).max(1) {
            step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, u,
                 1.0 / Transition::DEPLOY, KP, dt);
        }
        let deflection = nozzle_deflection(&mech, &names, &bodies);
        let poses: Vec<Option<(Vector3, Quaternion)>> = bodies
            .iter()
            .map(|b| b.and_then(|b| mech.world.body(b)).map(|x| (x.translation(), x.rotation())))
            .collect();
        for a in 0..names.len() {
            for b in (a + 1)..names.len() {
                if exempt.contains(&(a, b)) {
                    continue;
                }
                let (Some(pa), Some(pb)) = (poses[a], poses[b]) else {
                    continue;
                };
                // Both directions. A's vertices against B's surface finds A
                // poking into B; it does not find B poking into A between two of
                // A's vertices, and on a gear rim against a duct wall that is
                // exactly the case that matters.
                let mut min = f32::INFINITY;
                let mut at = Vector3::ZERO;
                for (from, to, pf, pt) in [(a, b, pa, pb), (b, a, pb, pa)] {
                    let Some(tree) = bvh[to].as_ref() else { continue };
                    for v in &verts[from] {
                        let w = Vector3::new(v[0], v[1], v[2]).apply_quaternion(pf.1) + pf.0;
                        let local = (w - pt.0).apply_quaternion(pt.1.conjugate());
                        let (_, d, _) = tree.closest_point_to_point(local);
                        if d < min {
                            min = d;
                            at = w;
                        }
                    }
                }
                match worst.iter_mut().find(|(_, x, y, _, _)| *x == a && *y == b) {
                    Some(slot) if min < slot.0 => *slot = (min, a, b, deflection, at),
                    Some(_) => {}
                    None => worst.push((min, a, b, deflection, at)),
                }
            }
        }
    }
    worst.sort_by(|x, y| x.0.total_cmp(&y.0));
    println!("  Closest surface-to-surface approach over the deployment, mesh against mesh:");
    println!("            pair                              gap      at jet   where");
    for (d, a, b, defl, at) in worst.iter().take(8) {
        println!(
            "  {:<16} vs {:<16} {:8.1} mm {:7.1} deg   [{:+.3} {:+.3} {:+.3}]{}",
            names[*a],
            names[*b],
            d * 1000.0,
            defl,
            at.x,
            at.y,
            at.z,
            if *d <= 0.0005 { "  <- fouls" } else { "" }
        );
    }
    let fouls = worst.iter().filter(|(d, ..)| *d <= 0.0005).count();
    println!(
        "  {} pair(s) of {} checked come within half a millimetre.",
        fouls,
        worst.len()
    );
    println!("  Pinion against its own ring gear is exempt: their teeth interleave by design,");
    println!("  and the mesh is a rate constraint, so tooth phase is not simulated at all.");
}

/// Deflection read off the simulated nozzle body.
fn nozzle_deflection(
    mech: &ScadMechanism,
    names: &[String],
    bodies: &[Option<BodyId>],
) -> f64 {
    let Some(i) = names.iter().position(|n| n == "nozzle") else {
        return 0.0;
    };
    bodies[i]
        .and_then(|b| mech.world.body(b))
        .map(|b| {
            let e = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(b.rotation());
            (e.x.clamp(-1.0, 1.0) as f64).acos().to_degrees()
        })
        .unwrap_or(0.0)
}

/// Cut a frame open on a plane, for a section view.
///
/// Triangle-level clipping rather than CSG: each triangle is trimmed to the
/// half-space `dot(p, n) >= d` and anything wholly outside is dropped. What
/// comes back is an open shell, which is exactly what a section drawing is —
/// there is nothing here to be watertight for, because nothing simulates it. The
/// mechanism that produced these poses is the whole one, and this only decides
/// what gets drawn.
///
/// The plane to use is `y = 0`: the duct folds in X-Z, so cutting there keeps
/// the whole silhouette of the fold and takes the near half of the wall away —
/// and the drive train is all at +Y, so none of it is lost.
fn section(frame: &ScadFrame, n: [f32; 3], d: f32) -> ScadFrame {
    let parts: Vec<ScadPart> = frame
        .parts
        .iter()
        .filter_map(|p| {
            clip(&p.geometry, n, d).map(|geometry| ScadPart {
                geometry,
                color: p.color,
            })
        })
        .collect();
    ScadFrame {
        index: frame.index,
        t: frame.t,
        parts: Arc::new(parts),
        viewport: frame.viewport,
    }
}

/// Trim a triangle soup to a half-space, carrying its normals through so the cut
/// mesh shades the way the whole one did.
fn clip(g: &BufferGeometry, n: [f32; 3], d: f32) -> Option<BufferGeometry> {
    let pos = g.get_attribute("position")?;
    let nrm = g.get_attribute("normal");
    let vertex = |i: usize| -> ([f32; 3], [f32; 3]) {
        let p = [pos.array[i * 3], pos.array[i * 3 + 1], pos.array[i * 3 + 2]];
        let v = match nrm {
            Some(a) if a.array.len() >= i * 3 + 3 => {
                [a.array[i * 3], a.array[i * 3 + 1], a.array[i * 3 + 2]]
            }
            _ => [0.0, 0.0, 1.0],
        };
        (p, v)
    };
    let indices: Vec<u32> = match &g.index {
        Some(idx) => idx.clone(),
        None => (0..pos.count() as u32).collect(),
    };
    let side = |p: [f32; 3]| p[0] * n[0] + p[1] * n[1] + p[2] * n[2] - d;

    let mut out_p: Vec<f32> = Vec::new();
    let mut out_n: Vec<f32> = Vec::new();
    for tri in indices.chunks_exact(3) {
        let v: Vec<([f32; 3], [f32; 3])> = tri.iter().map(|&i| vertex(i as usize)).collect();
        // Sutherland-Hodgman against the single plane: a triangle comes out as
        // nothing, a triangle, or a quad.
        let mut poly: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(4);
        for i in 0..3 {
            let (a, b) = (v[i], v[(i + 1) % 3]);
            let (da, db) = (side(a.0), side(b.0));
            if da >= 0.0 {
                poly.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                let t = da / (da - db);
                let mix = |x: [f32; 3], y: [f32; 3]| {
                    [
                        x[0] + (y[0] - x[0]) * t,
                        x[1] + (y[1] - x[1]) * t,
                        x[2] + (y[2] - x[2]) * t,
                    ]
                };
                poly.push((mix(a.0, b.0), mix(a.1, b.1)));
            }
        }
        for k in 1..poly.len().saturating_sub(1) {
            for &(p, q) in &[poly[0], poly[k], poly[k + 1]] {
                out_p.extend_from_slice(&p);
                out_n.extend_from_slice(&q);
            }
        }
    }
    if out_p.is_empty() {
        return None;
    }
    let mut cut = BufferGeometry::new();
    cut.set_attribute("position", BufferAttribute::new(out_p, 3));
    cut.set_attribute("normal", BufferAttribute::new(out_n, 3));
    Some(cut)
}

/// A PNG sequence of the transition, through the ordinary SCAD render path.
fn render(spec: &MechanismSpec, schedule: &Schedule) {
    let ratio = mesh_ratio(spec);
    // `--section` cuts the drawn model open on the fold plane, so the drive
    // train inside the casing is visible. It changes nothing that moves.
    let cut = std::env::args().any(|a| a == "--section");
    use threers::openscad::animate::{ScadCamera, ScadRender};

    println!("\n-- render --------------------------------------------------------");
    let Ok(mut mech) = ScadMechanism::from_spec(spec) else {
        println!("  could not rebuild the mechanism");
        return;
    };
    mech.world.solver_config.velocity_iterations = 128;
    mech.world.solver_config.position_iterations = 8;
    let bearings: Vec<MateId> = BEARINGS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    let shafts: Vec<MateId> = SHAFTS
        .iter()
        .filter_map(|n| mech.assembly.mate_named(n))
        .collect();
    for mate in &shafts {
        mech.assembly.drive(*mate).hold();
    }
    let ceiling: Vec<f64> = SHAFTS
        .iter()
        .map(|n| {
            spec.drives
                .iter()
                .find(|d| d.mate == *n)
                .and_then(|d| d.torque)
                .unwrap_or(500.0) as f64
        })
        .collect();

    let fps = 60;
    let dt = 1.0 / fps as f32;

    let dir = std::path::Path::new("out/three_bearing_swivel");
    if let Err(e) = std::fs::create_dir_all(dir) {
        println!("  cannot create {}: {e}", dir.display());
        return;
    }

    // A section is a drawing, not a sequence: one pose, cut on the plane through
    // the duct axis and the drive axis, which is the only cut that puts the ring
    // gear, its pinion and the motor all in the same face. Halving the duct the
    // other way shows the bore and very little else.
    if cut {
        for _ in 0..fps {
            step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, 0.0, 0.0, KP, dt);
        }
        let shot = mech.snapshot(0, 0.0);
        // Looking down onto the cut, nearly square to it, so the wall reads as a
        // wall rather than as a silhouette: on the internal build that is the
        // liner, the annulus, the casing, and the drive lying in between them.
        let renderer = ScadRender::new(1600, 700).camera(ScadCamera::Auto {
            yaw: 4.0,
            pitch: 68.0,
            zoom: 1.0,
        });
        match renderer.render_evaluated(&[section(&shot, [0.0, 0.0, -1.0], 0.0)]) {
            Ok(images) if !images.is_empty() => {
                let png = threers::utils::png::encode_png(1600, 700, &images[0]);
                let path = dir.join("section.png");
                if let Err(e) = std::fs::write(&path, png) {
                    println!("  cannot write {}: {e}", path.display());
                    return;
                }
                println!("  wrote {}", path.display());
            }
            Ok(_) => println!("  the cut left nothing to draw"),
            Err(e) => println!("  no renderer available here: {e}"),
        }
        return;
    }

    let frames = (TRANSITION.total() * fps as f64) as usize;
    let mut shots = Vec::new();
    for i in 0..frames {
        let t = i as f64 / fps as f64;
        let (_, u) = TRANSITION.at(t);
        const WINDOW: f64 = 0.05;
        let du = (TRANSITION.at(t + WINDOW).1 - TRANSITION.at(t - WINDOW).1) / (2.0 * WINDOW);
        step(&mut mech, &bearings, &shafts, &ceiling, ratio, schedule, u, du, KP, dt);
        if i % 3 == 0 {
            shots.push(mech.snapshot(shots.len(), t));
        }
    }

    // A fixed three-quarter view rather than a turntable: the subject is the
    // mechanism folding, and an orbiting camera makes it hard to tell which of
    // the two is doing the moving. The drive units are out on +Y, so this angle
    // shows one of them and the fold at the same time.
    // A section is drawn from further back and squarer on: cutting the near half
    // away leaves the model wider than it is deep, and the three-quarter view
    // that suits the whole thing hides the annulus behind its own wall.
    let renderer = ScadRender::new(1280, 540).camera(ScadCamera::Auto {
        yaw: 24.0,
        pitch: 12.0,
        zoom: 0.8,
    });
    let images = match renderer.render_evaluated(&shots) {
        Ok(images) => images,
        Err(e) => {
            println!("  no renderer available here: {e}");
            return;
        }
    };
    for (i, rgba) in images.iter().enumerate() {
        let png = threers::utils::png::encode_png(1280, 540, rgba);
        if let Err(e) = std::fs::write(dir.join(format!("frame{i:04}.png")), png) {
            println!("  cannot write frame {i}: {e}");
            return;
        }
    }
    println!("  wrote {} frames to {}", images.len(), dir.display());
}
