//! The mechanisms the tour ships with.
//!
//! Each one is a complete `.scad` file that declares its own parts, joints and
//! drives — no includes, so the browser can hand any of them straight to the
//! evaluator, and so any of them can be pasted into OpenSCAD and opened.
//!
//! They are ordered the way they build on each other: gears, then a closed
//! loop, then motion that comes out of contact rather than out of a
//! declaration, then a machine that puts all of it together.

/// One bundled mechanism, with the run it wants.
pub struct Model {
    /// Stable identifier — the thing a URL or a command line names.
    pub id: &'static str,
    pub title: &'static str,
    /// One line, for a picker.
    pub blurb: &'static str,
    pub source: &'static str,
    pub frames: usize,
    pub fps: u32,
    /// Model units per metre: 1000 for a model drawn in millimetres.
    pub units_per_metre: f32,
    /// Substeps the solver wants. Stiff couplings — a fine thread, a gear train
    /// with a large reduction — need a shorter step than loose parts do.
    pub substeps: usize,
}

pub const BENCH: &str = include_str!("../../threers-physics/examples/mechanism_tour.scad");
pub const GEAR_TRAIN: &str = include_str!("../models/gear_train.scad");
pub const FOUR_BAR: &str = include_str!("../models/four_bar.scad");
pub const CAM_FOLLOWER: &str = include_str!("../models/cam_follower.scad");
pub const RATCHET: &str = include_str!("../models/ratchet.scad");
pub const LATCH: &str = include_str!("../models/latch.scad");

pub const MODELS: &[Model] = &[
    Model {
        id: "bench",
        title: "the bench",
        blurb: "one of everything: hinge, gear, slider, thread, and a lid that is asked for more travel than it has",
        source: BENCH,
        frames: 240,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 8,
    },
    Model {
        id: "gear-train",
        title: "compound gear train",
        blurb: "three shafts, two meshes, 7.5:1 — and the reduction is a consequence, not a number anyone typed",
        source: GEAR_TRAIN,
        frames: 240,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 8,
    },
    Model {
        id: "four-bar",
        title: "four-bar crank-rocker",
        blurb: "a closed loop: the crank goes round, the rocker cannot, and the geometry decides where it turns back",
        source: FOUR_BAR,
        frames: 240,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 8,
    },
    Model {
        id: "cam",
        title: "cam and follower",
        blurb: "nothing declares how the follower moves — it rides on the cam, and the contact is the mechanism",
        source: CAM_FOLLOWER,
        frames: 240,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 12,
    },
    Model {
        id: "ratchet",
        title: "ratchet and pawl",
        blurb: "turns one way and locks the other, out of a tooth shape and a spring — with the teeth kept concave",
        source: RATCHET,
        frames: 300,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 12,
    },
    Model {
        id: "latch",
        title: "latched gate",
        blurb: "rack and pinion, a spring latch that catches, and a drive that then fails to open it until it is released",
        source: LATCH,
        frames: 420,
        fps: 60,
        units_per_metre: 1000.0,
        substeps: 12,
    },
];

/// Look one up by `id`.
pub fn model(id: &str) -> Option<&'static Model> {
    MODELS.iter().find(|m| m.id == id)
}
