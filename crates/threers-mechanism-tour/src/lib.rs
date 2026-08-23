//! The mechanism tour, in a browser.
//!
//! `threers-physics/examples/mechanism_tour.rs` walks the whole assembly stack
//! and prints it to a terminal. This is the same seven acts with the model
//! editable and the run on screen: the parts are drawn where the solver put
//! them, every joint's reading is plotted against time, and the transcript is
//! the one the example prints.
//!
//! Nothing here is browser-specific — [`Tour::run`] is ordinary Rust, and
//! `examples/console.rs` runs it natively. [`web`] is the thin part: flat
//! buffers over wasm memory, because a frame is seven floats per part and the
//! player asks for a different one sixty times a second.

use std::fmt::Write as _;

use threers::assembly as geo;
use threers::openscad::mechanism::{MateSpecKind, MechanismSpec};
use threers::openscad::ScadPart;
use threers_physics::assembly::Assembly;
use threers_physics::mechanism::{PoseTrack, ScadMechanism, Tolerance};
use threers_physics::verify::Verify;

pub mod models;
pub use models::{model, Model, MODELS};

#[cfg(target_arch = "wasm32")]
pub mod web;

/// The bench the example uses, so the page opens on something that works.
pub const MODEL: &str = include_str!("../../threers-physics/examples/mechanism_tour.scad");

/// One section of the transcript.
pub struct Act {
    pub title: String,
    pub body: String,
}

/// A drawable: one `color()` group of one part, as a triangle soup.
///
/// De-indexed with a normal per face rather than per vertex. A solid out of a
/// CSG kernel is faceted — a cylinder really is 32 flat strips — and shading it
/// smooth draws a curve that the geometry does not have.
pub struct Piece {
    /// Which part moves it.
    pub owner: usize,
    pub color: [f32; 4],
    /// Three vertices per triangle, in the part's own frame.
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
}

impl Piece {
    fn from_scad(piece: &ScadPart, owner: usize) -> Piece {
        let tris = geo::from_geometry(&piece.geometry);
        let mut positions = Vec::with_capacity(tris.len() * 9);
        let mut normals = Vec::with_capacity(tris.len() * 9);
        for t in &tris {
            let e1 = [
                t[1][0] - t[0][0],
                t[1][1] - t[0][1],
                t[1][2] - t[0][2],
            ];
            let e2 = [
                t[2][0] - t[0][0],
                t[2][1] - t[0][1],
                t[2][2] - t[0][2],
            ];
            let mut n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            // A degenerate triangle has no normal to give. Leave it pointing up
            // rather than at NaN, which would take the whole draw call with it.
            if len > 0.0 {
                n = [n[0] / len, n[1] / len, n[2] / len];
            } else {
                n = [0.0, 1.0, 0.0];
            }
            for v in t {
                positions.extend_from_slice(&[v[0] as f32, v[1] as f32, v[2] as f32]);
                normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);
            }
        }
        Piece {
            owner,
            color: piece.rgba_or([0.80, 0.81, 0.85, 1.0]),
            positions,
            normals,
        }
    }

    pub fn triangles(&self) -> usize {
        self.positions.len() / 9
    }
}

/// One joint's reading over the whole run, in the units it is declared in.
pub struct JointTrace {
    pub name: String,
    /// `Hinge`, `Screw { pitch: 1.25 }` — the declaration, as written.
    pub kind: String,
    /// `°` or the model's own length unit.
    pub unit: String,
    pub values: Vec<f32>,
    /// The travel the model declared, if it declared any.
    pub limit: Option<[f32; 2]>,
}

/// Whether a declared joint was actually *built*.
///
/// A mate is a sentence about two parts. It is not a pin, a bore, a rail or a
/// bearing, and it draws none of them — the solver will hold two parts in a
/// perfect hinge relationship whether or not there is anything between them to
/// do the holding. Worse, a mate *suppresses* the two checks that would notice:
/// its pair is excluded from collision and from the interference audit, on
/// purpose, because a real pin does live inside both halves of a real joint.
///
/// So a mechanism can pass every other check here while its links hang in the
/// air a millimetre apart. See [`joints_built`] for how this is measured.
pub struct JointBuild {
    pub name: String,
    /// `Hinge`, `Slider`, `Gear` — the declaration's own word.
    pub kind: String,
    pub parts: Vec<String>,
    /// The gap between the two parts near the joint's axis, in model units:
    /// nought where a pin fills a bore, a clearance where a stem runs in a
    /// guide, and the whole empty distance where nothing was ever built.
    pub gap: f32,
    /// How far off the axis that closest approach was found — nought for a pin
    /// in a bore, out at the stem's own radius for a slider in a guide.
    pub off_axis: f32,
    /// What counts as a clearance fit on a model this size.
    pub fit: f32,
}

impl JointBuild {
    /// Is there metal between these two parts where the joint says there is?
    pub fn built(&self) -> bool {
        self.gap <= self.fit
    }
}

/// The numbers the page puts in its header, rather than in the transcript.
#[derive(Default, Clone, Copy)]
pub struct Facts {
    pub parts: usize,
    pub mates: usize,
    pub drives: usize,
    pub triangles: usize,
    pub mobility: i32,
    pub interferences: usize,
    pub unchecked: usize,
    pub ungrounded: bool,
    /// Bytes of poses, against bytes had every frame been baked.
    pub pose_bytes: usize,
    pub baked_bytes: usize,
    pub keys: usize,
    pub reduced_keys: usize,
    pub confirmed: usize,
    pub agrees: bool,
    pub replayed: bool,
    /// Milliseconds spent in the solver, for the run that was recorded.
    pub simulated_ms: f64,
}

/// Everything one run of the tour produced.
pub struct Tour {
    pub acts: Vec<Act>,
    pub parts: Vec<String>,
    pub pieces: Vec<Piece>,
    /// Every frame end to end: `frame * parts * 7`, as position then quaternion.
    pub poses: Vec<f32>,
    pub stride: usize,
    pub frames: usize,
    pub fps: u32,
    pub joints: Vec<JointTrace>,
    /// Over the whole run, so a camera framed on it never has to re-frame.
    pub bounds: [f32; 6],
    pub facts: Facts,
}

impl Tour {
    /// Read the model, assemble it, check it, run it, and ask the geometry
    /// whether it agrees — the seven acts, in the order you would do them.
    ///
    /// `units_per_metre` is what the model is drawn in: 1000 for millimetres,
    /// 1 for metres. It scales gravity *and* the solver's length constants,
    /// which are tuned for a scene in metres and mean something a thousand times
    /// tighter in a millimetre one.
    pub fn run(
        source: &str,
        frames: usize,
        fps: u32,
        units_per_metre: f32,
        substeps: usize,
    ) -> Result<Tour, String> {
        let mut acts = Vec::new();
        let mut facts = Facts::default();

        // ---- 1. what the model says about itself ---------------------------
        let spec = threers::parse_scad_mechanism(source)
            .map_err(|e| format!("could not read the model: {e}"))?;
        acts.push(Act {
            title: "declared".into(),
            body: declarations(&spec, &mut facts),
        });

        // ---- 2. put it together, and check it ------------------------------
        let mut asm = Assembly::from_scad(&spec)
            .map_err(|e| format!("the declarations do not line up: {e}"))?;
        // The model drew its parts in place, so this normally moves nothing —
        // and says so loudly when the drawing and the declarations disagree.
        asm.solve()
            .map_err(|e| format!("the model and its declarations disagree: {e}"))?;
        acts.push(Act {
            title: "check".into(),
            body: check(&asm, &mut facts),
        });

        // ---- 3. how far can each mate really travel? -----------------------
        acts.push(Act {
            title: "travel".into(),
            body: travel(&mut asm, &spec),
        });

        // ---- 4. run it -----------------------------------------------------
        let mut mech = build(&spec, frames, fps, units_per_metre, substeps)?;
        let started = now_ms();
        // A pose per part per frame, not a copy of the model per frame.
        let track = mech.record();
        facts.simulated_ms = now_ms() - started;
        acts.push(Act {
            title: format!("after {:.1}s", frames as f32 / fps as f32),
            body: readout(&mech, &spec),
        });

        // Joint readings frame by frame, which the terminal version only shows
        // at the end. A second pass rather than a hook into `record`: the run is
        // milliseconds and the recording API stays the one thing to reach for.
        let joints = trace(&spec, frames, fps, units_per_metre, substeps)?;

        // ---- 5. what the run costs, and what it reduces to -----------------
        acts.push(Act {
            title: "the run".into(),
            body: cost(&track, fps, &mut facts),
        });

        // ---- 6. does the geometry agree with the declaration? --------------
        let mut checked = build(&spec, frames, fps, units_per_metre, substeps)?;
        acts.push(Act {
            title: "verify".into(),
            body: verify(&mut checked, &mut facts),
        });

        // ---- 7. measure the parts themselves -------------------------------
        acts.push(Act {
            title: "the parts, measured".into(),
            body: measure(&mech, &spec, &mut facts),
        });

        // ---- and everything the page has to draw ---------------------------
        let parts: Vec<String> = track.parts().iter().map(|p| p.name.clone()).collect();
        let mut pieces = Vec::new();
        for (i, part) in track.parts().iter().enumerate() {
            for piece in &part.pieces {
                pieces.push(Piece::from_scad(piece, i));
            }
        }
        facts.triangles = pieces.iter().map(Piece::triangles).sum();

        let stride = parts.len() * 7;
        let mut poses = Vec::with_capacity(stride * track.frames());
        for frame in 0..track.frames() {
            for part in 0..parts.len() {
                let p = track
                    .pose(part, frame)
                    .unwrap_or(threers_physics::math::Isometry::IDENTITY);
                poses.extend_from_slice(&[
                    p.translation.x,
                    p.translation.y,
                    p.translation.z,
                    p.rotation.x,
                    p.rotation.y,
                    p.rotation.z,
                    p.rotation.w,
                ]);
            }
        }

        let bounds = match track.bounds() {
            Some((lo, hi)) => [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]],
            None => [0.0; 6],
        };

        Ok(Tour {
            acts,
            parts,
            pieces,
            poses,
            stride,
            frames: track.frames(),
            fps: track.fps(),
            joints,
            bounds,
            facts,
        })
    }

    /// The seven acts as the example prints them.
    pub fn transcript(&self) -> String {
        let mut out = String::new();
        for act in &self.acts {
            let _ = writeln!(out, "== {} ==", act.title);
            out.push_str(&act.body);
        }
        out
    }
}

fn build(
    spec: &MechanismSpec,
    frames: usize,
    fps: u32,
    units_per_metre: f32,
    substeps: usize,
) -> Result<ScadMechanism, String> {
    let mut mech = ScadMechanism::from_spec(spec)
        .map_err(|e| format!("could not build the mechanism: {e}"))?
        .frames(frames)
        .fps(fps)
        .units(units_per_metre);
    // A fine thread is a stiff constraint — a millimetre of travel demands five
    // radians of turn — and stiff constraints want a shorter step. This is the
    // *cap*: `World::substeps` reports how many actually ran.
    mech.world.max_substeps = substeps.max(1);
    Ok(mech)
}

fn declarations(spec: &MechanismSpec, facts: &mut Facts) -> String {
    facts.parts = spec.parts.len();
    facts.mates = spec.mates.len();
    facts.drives = spec.drives.len();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "  {} parts, {} mates, {} drives",
        spec.parts.len(),
        spec.mates.len(),
        spec.drives.len()
    );
    for mate in &spec.mates {
        let _ = writeln!(
            out,
            "    {:<12} {:?}  {} on {}",
            mate.name, mate.kind, mate.parts[0], mate.parts[1]
        );
    }
    for drive in &spec.drives {
        match drive.to {
            Some(to) => {
                let _ = writeln!(
                    out,
                    "    drive {:<12} to {to} over {}s, starting at {}s",
                    drive.mate,
                    drive.over.unwrap_or(0.0),
                    drive.start
                );
            }
            // No target: a motor. The only thing that works for a shaft making
            // full turns, since a hinge angle wraps at ±360°.
            None => {
                let _ = writeln!(
                    out,
                    "    drive {:<12} at {}°/s, continuously",
                    drive.mate,
                    drive.max_speed.unwrap_or(0.0)
                );
            }
        }
    }
    // A mate naming a part that does not exist is the commonest way to get a
    // mechanism wrong, and otherwise shows up much later as a joint that does
    // nothing.
    let dangling = spec.dangling_parts();
    if !dangling.is_empty() {
        let _ = writeln!(out, "  !! mates name parts that do not exist: {dangling:?}");
    }
    let loose = spec.dangling_drives();
    if !loose.is_empty() {
        let _ = writeln!(out, "  !! drives name mates that do not exist: {loose:?}");
    }
    out
}

fn check(asm: &Assembly, facts: &mut Facts) -> String {
    let report = asm.check();
    facts.mobility = report.mobility;
    facts.interferences = report.interferences.len();
    facts.unchecked = report.unchecked.len();
    facts.ungrounded = report.ungrounded;

    let name = |id| asm.part_of(id).map(|p| p.name.as_str()).unwrap_or("?");
    let mut out = String::new();
    let _ = writeln!(out, "  mobility: {} degree(s) of freedom", report.mobility);
    if report.ungrounded {
        let _ = writeln!(out, "  !! nothing is fixed — the whole assembly can drift");
    }
    let _ = writeln!(out, "  interferences: {}", report.interferences.len());
    for hit in &report.interferences {
        let _ = writeln!(
            out,
            "    {} into {} by {:.3}",
            name(hit.a),
            name(hit.b),
            hit.depth
        );
    }
    // Two triangle meshes have no volume between them to measure, so the pair is
    // reported rather than passed. A check that could not run is not a check
    // that passed.
    for (a, b) in &report.unchecked {
        let _ = writeln!(
            out,
            "    untestable: {} / {} — both are surfaces",
            name(*a),
            name(*b)
        );
    }
    if !report.ignored_limits.is_empty() {
        let _ = writeln!(
            out,
            "    {} mate(s) given a range they cannot use",
            report.ignored_limits.len()
        );
    }
    out
}

/// Sweep every mate the model gave a range to, and say what stops it short.
fn travel(asm: &mut Assembly, spec: &MechanismSpec) -> String {
    let mut out = String::new();
    let mut any = false;
    for declared in &spec.mates {
        if declared.range.is_none() {
            continue;
        }
        let Some(mate) = asm.mate_named(&declared.name) else {
            continue;
        };
        let Some(sweep) = asm.sweep(mate, 96) else {
            continue;
        };
        any = true;
        // Both ends, because a mate is as likely to be stopped short on the way
        // back as on the way out — and a thread declared `[-5, 0]` reads as
        // travelling nowhere if only the upper end is printed.
        let angular = declared.kind.is_angular();
        let unit = if angular { "°" } else { "" };
        let show = |v: f32| if angular { v.to_degrees() } else { v };
        let _ = write!(
            out,
            "  {:<12} declared {:.0}..{:.0}{unit}, clear {:.0}..{:.0}{unit}",
            declared.name,
            show(sweep.declared.0),
            show(sweep.declared.1),
            show(sweep.clear.0),
            show(sweep.clear.1),
        );
        match sweep.blocker {
            Some((_, on)) => {
                let _ = writeln!(
                    out,
                    " — fouls on {}",
                    asm.part_of(on).map(|p| p.name.as_str()).unwrap_or("?")
                );
            }
            None => out.push_str(" — all of it\n"),
        }
    }
    if !any {
        out.push_str("  no mate declares a range to sweep\n");
    }
    out
}

fn readout(mech: &ScadMechanism, spec: &MechanismSpec) -> String {
    let mut out = String::new();
    for declared in &spec.mates {
        let Some(mate) = mech.assembly.mate_named(&declared.name) else {
            continue;
        };
        let Some(value) = mech.assembly.coordinate(mate, &mech.world) else {
            continue;
        };
        let unit = if declared.kind.is_angular() {
            "rad"
        } else {
            "units"
        };
        let _ = writeln!(
            out,
            "  {:<12} {value:>9.3} {unit:<6} at {:>8.3} {unit}/s",
            declared.name,
            mech.assembly.speed(mate, &mech.world).unwrap_or(0.0),
        );
    }
    // A gear constrains a rate. The thing worth reading back is whether it still
    // holds after four seconds of being driven through everything else.
    for declared in &spec.mates {
        let MateSpecKind::Gear { ratio } = declared.kind else {
            continue;
        };
        let speed = |part: &str| -> Option<f32> {
            let axle = spec
                .mates
                .iter()
                .find(|m| m.kind == MateSpecKind::Hinge && m.parts[0] == part)?;
            let id = mech.assembly.mate_named(&axle.name)?;
            mech.assembly.speed(id, &mech.world)
        };
        if let (Some(fast), Some(slow)) = (speed(&declared.parts[0]), speed(&declared.parts[1])) {
            let _ = writeln!(
                out,
                "  {:<12} declared {ratio:.2}:1, holds {:.2}:1",
                declared.name,
                if slow != 0.0 { fast / slow } else { 0.0 }
            );
        }
    }
    out
}

fn cost(track: &PoseTrack, fps: u32, facts: &mut Facts) -> String {
    use threers::materials::{BasicMaterial, Material};
    use threers::math::Color;

    let grey = || Material::Basic(BasicMaterial::new(Color::from_hex(0xcccccc)));
    let keys = |c: &threers::animation::AnimationClip| -> usize {
        c.tracks.iter().map(|t| t.times.len()).sum()
    };

    let mut arena = threers::core::ObjectArena::new();
    let nodes = track.spawn(&mut arena, None, grey());
    facts.keys = keys(&track.to_clip(&nodes, "run"));
    facts.reduced_keys = keys(&track.to_clip_simplified(&nodes, "run", Tolerance::default()));
    facts.pose_bytes = track.parts().iter().map(|p| p.poses.len() * 7 * 4).sum();
    facts.baked_bytes = track.baked_bytes();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "  {} frames as {:.1} kB of poses, or {:.1} MB had every frame been baked",
        track.frames(),
        facts.pose_bytes as f64 / 1e3,
        facts.baked_bytes as f64 / 1e6
    );
    let _ = writeln!(
        out,
        "  as an AnimationClip: {} keys, {} once reduced ({:.0}× fewer)",
        facts.keys,
        facts.reduced_keys,
        facts.keys as f64 / facts.reduced_keys.max(1) as f64
    );
    // Played back through the ordinary mixer, with no physics anywhere.
    let (played, note) = replay(track, fps);
    facts.replayed = played;
    let _ = writeln!(out, "  replayed through the mixer: {note}");
    out
}

/// Play the reduced clip through the ordinary mixer and report where it lands.
fn replay(track: &PoseTrack, fps: u32) -> (bool, String) {
    use threers::animation::{AnimationMixer, LoopMode};

    let mut scene = threers::scene::Scene::new();
    let root = scene.root;
    // Spawned into the scene's own arena: a clip targets nodes, and they have to
    // be the nodes the mixer will find.
    let nodes = track.spawn(
        &mut scene.arena,
        Some(root),
        threers::materials::Material::Basic(threers::materials::BasicMaterial::new(
            threers::math::Color::from_hex(0xcccccc),
        )),
    );
    let clip = track.to_clip_simplified(&nodes, "run", Tolerance::default());

    let mut mixer = AnimationMixer::new();
    let action = mixer.clip_action(clip);
    // A run is a one-shot: it opens the lid, it does not open it forever.
    mixer.actions[action].loop_mode = LoopMode::Once;
    for _ in 0..track.frames() {
        mixer.update(&mut scene, 1.0 / fps as f32);
    }

    // Whichever part moved most is the one worth comparing.
    let moved = track
        .parts()
        .iter()
        .enumerate()
        .max_by(|a, b| turned(a.1).total_cmp(&turned(b.1)))
        .map(|(i, _)| i);
    let Some(part) = moved else {
        return (false, "nothing to replay".into());
    };
    let played = scene.arena.get(nodes[part]).map(|o| o.quaternion);
    let simulated = track.pose(part, track.frames() - 1).map(|p| p.rotation);
    match (played, simulated) {
        (Some(a), Some(b)) if a.dot(b).abs() > 0.999 => {
            (true, "ends where the simulation ended".into())
        }
        _ => (false, "ends somewhere else".into()),
    }
}

/// How far a part turned over the run, for picking the one worth checking.
fn turned(part: &threers_physics::mechanism::TrackedPart) -> f32 {
    let (Some(first), Some(last)) = (part.poses.first(), part.poses.last()) else {
        return 0.0;
    };
    1.0 - first.rotation.dot(last.rotation).abs()
}

fn verify(mech: &mut ScadMechanism, facts: &mut Facts) -> String {
    let found = Verify::new().run(mech);
    facts.confirmed = found.confirmed;
    facts.agrees = found.agrees();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "  {} mate(s) confirmed over {} poses",
        found.confirmed, found.poses
    );
    for finding in &found.findings {
        let _ = writeln!(out, "  {finding}");
    }
    if found.agrees() {
        out.push_str("  the geometry agrees with the declaration\n");
    }
    out.push_str(&drift());
    out
}

/// A model whose `rotate()` and whose `hinge()` disagree, and the check that
/// notices — the one that finds real drift, with no solver in between.
fn drift() -> String {
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
    let mut out = String::new();
    for (label, z) in [("declared truthfully", 16.0), ("declared at z = 14", 14.0)] {
        match checker.against_model(&model(z), 0.0, 1.0) {
            Ok(report) if report.agrees() => {
                let _ = writeln!(out, "  {label}: agrees");
            }
            Ok(report) => {
                for f in report.faults() {
                    let _ = writeln!(out, "  {label}: {f}");
                }
            }
            Err(e) => {
                let _ = writeln!(out, "  {label}: {e}");
            }
        }
    }
    out
}

fn measure(mech: &ScadMechanism, spec: &MechanismSpec, _facts: &mut Facts) -> String {
    let mut out = String::new();
    for declared in &spec.parts {
        let Some(part) = mech
            .assembly
            .part_named(&declared.name)
            .and_then(|p| mech.assembly.part_of(p))
        else {
            continue;
        };
        let Some(mesh) = part.geometry.as_ref() else {
            continue;
        };
        let tris = geo::from_geometry(mesh);
        let key = geo::rigid_key(&tris);
        let _ = writeln!(
            out,
            "  {:<9} {:>5} triangles  {:>8.0} u² surface  {:>9.0} u³  {} shell(s)  hand {:>2}",
            declared.name,
            tris.len(),
            geo::area(&tris),
            geo::volume(&tris),
            geo::shells(&tris, 0.0).len(),
            key.handedness
        );
    }
    // The identity survives being moved, which is what makes it usable for
    // finding the same body again in another pose.
    if let Some(mesh) = spec
        .parts
        .first()
        .and_then(|p| mech.assembly.part_named(&p.name))
        .and_then(|p| mech.assembly.part_of(p))
        .and_then(|p| p.geometry.as_ref())
    {
        let here = geo::from_geometry(mesh);
        let moved = geo::from_geometry_at(mesh, &shifted());
        let _ = writeln!(
            out,
            "  the first part's identity survives being moved and turned: {}",
            geo::rigid_key(&here).matches(&geo::rigid_key(&moved))
        );
    }
    out
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

/// Every joint's reading, frame by frame, in the units it was declared in.
fn trace(
    spec: &MechanismSpec,
    frames: usize,
    fps: u32,
    units_per_metre: f32,
    substeps: usize,
) -> Result<Vec<JointTrace>, String> {
    let mut mech = build(spec, frames, fps, units_per_metre, substeps)?;
    let mut traces: Vec<JointTrace> = spec
        .mates
        .iter()
        .map(|m| JointTrace {
            name: m.name.clone(),
            kind: format!("{:?}", m.kind),
            unit: if m.kind.is_angular() { "°" } else { "u" }.into(),
            values: Vec::with_capacity(frames),
            limit: m.range,
        })
        .collect();

    for _ in 0..frames {
        for (trace, declared) in traces.iter_mut().zip(&spec.mates) {
            let value = mech
                .assembly
                .mate_named(&declared.name)
                .and_then(|id| mech.assembly.coordinate(id, &mech.world))
                .unwrap_or(f32::NAN);
            trace.values.push(if declared.kind.is_angular() {
                value.to_degrees()
            } else {
                value
            });
        }
        mech.step();
    }
    // A mate with no coordinate — a gear, a weld — has nothing to plot.
    traces.retain(|t| t.values.iter().any(|v| v.is_finite()));
    Ok(traces)
}

/// A monotonic millisecond clock, where there is one.
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1e3)
        .unwrap_or(0.0)
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web::now_ms()
}
