//! What each mechanism actually looks like, from where the page looks at it.
//!
//! ```text
//! cargo run --release -p threers-mechanism-tour --example look
//! cargo run --release -p threers-mechanism-tour --example look -- latch 0.5
//! ```
//!
//! Writes one PNG per model to `out/tour-look/`. The joint readings can be
//! right, the interference check clean and every test green while the picture is
//! wrong — a post standing between the camera and the mechanism, a plate over
//! the one thing the model is about. That is not a number, so it needs an image.
//!
//! The camera matches `web/mechanism-tour/index.html`: yaw −0.9 rad and pitch
//! 0.55, which puts the eye in front of the machine and above it. Front is −y
//! in every model here, and that is the whole reason it is worth stating.

use threers::openscad::animate::{ScadCamera, ScadRender};
use threers_mechanism_tour::{model, Tour, MODELS};

/// The page's own default view, in degrees.
const YAW: f32 = -51.6;
const PITCH: f32 = 31.5;

fn main() {
    let only = std::env::args().nth(1);
    let when: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let dir = std::path::Path::new("out/tour-look");
    if let Err(e) = std::fs::create_dir_all(dir) {
        return println!("cannot create {}: {e}", dir.display());
    }

    for m in MODELS {
        if only.as_deref().is_some_and(|w| w != m.id) {
            continue;
        }
        let Some(m) = model(m.id) else { continue };
        let tour = match Tour::run(m.source, m.settings()) {
            Ok(t) => t,
            Err(e) => {
                println!("{}: {e}", m.id);
                continue;
            }
        };
        // The pose track is what the solver produced; re-run it here only to
        // pick the one frame we want to look at.
        let frame = ((when * tour.fps as f32) as usize).min(tour.frames - 1);
        let evaluated = match rebuild(m.source, m.settings(), frame) {
            Ok(parts) => parts,
            Err(e) => {
                println!("{}: {e}", m.id);
                continue;
            }
        };
        let renderer = ScadRender::new(1100, 700).camera(ScadCamera::Auto {
            yaw: YAW,
            pitch: PITCH,
            zoom: 0.92,
        });
        match renderer.render_evaluated(&[evaluated]) {
            Ok(images) if !images.is_empty() => {
                let png = threers::utils::png::encode_png(1100, 700, &images[0]);
                let path = dir.join(format!("{}.png", m.id));
                match std::fs::write(&path, png) {
                    Ok(()) => println!("{:<12} frame {frame} -> {}", m.id, path.display()),
                    Err(e) => println!("{}: {e}", m.id),
                }
            }
            Ok(_) => println!("{}: nothing rendered", m.id),
            Err(e) => println!("{}: no renderer here: {e}", m.id),
        }
    }
}

/// One frame of the run, as placed solids the renderer can take.
fn rebuild(
    source: &str,
    set: threers_mechanism_tour::Settings,
    frame: usize,
) -> Result<Vec<threers::openscad::ScadPart>, String> {
    let spec = threers::parse_scad_mechanism(source)?;
    let mut mech = threers_physics::mechanism::ScadMechanism::from_spec(&spec)
        .map_err(|e| format!("{e}"))?
        .frames(set.frames)
        .fps(set.fps)
        .units(set.units_per_metre);
    if let Some(rate) = set.rate {
        mech = mech.physics_rate(rate);
    }
    mech.world.substeps = set.substeps;
    let track = mech.record();
    Ok(track.frame(frame.min(track.frames().saturating_sub(1))))
}
