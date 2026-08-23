//! The discretised rod against SoRoSim's Cosserat-rod solutions.
//!
//! `wL⁴/8EI` in `rod.rs` checks the small-deflection cantilever, which is the
//! easy half. This checks rods bent through tens of degrees by randomly
//! oriented gravity plus a wrench at mid-span and another at the tip — 500
//! independent equilibrium problems per material, of which these run a handful.
//!
//! Every test here **skips** when the reference data is absent, because it is
//! not vendored: see `threers_continuum::sorosim` for where to get it. Run the
//! full sweep with
//!
//! ```bash
//! cargo run --release -p threers-continuum --example sorosim_statics -- 100
//! ```
//!
//! and note `--release`: a hundred shapes at sixty-four iterations is minutes
//! of debug build and seconds of an optimised one.

use threers_continuum::sorosim::{
    evaluate_statics, Material, Statics, StaticsRun,
};

/// Small enough to run in a debug build without anyone noticing.
const SHAPES: usize = 3;

/// `None` and a printed note rather than a failure: the data is somebody
/// else's and may not be there.
fn load(material: Material) -> Option<Statics> {
    let set = Statics::load(material);
    if set.is_none() {
        eprintln!(
            "skipping: no SoRoSim reference data (see threers_continuum::sorosim)"
        );
    }
    set
}

fn run() -> StaticsRun {
    StaticsRun::default().links(10).seconds(8.0)
}

#[test]
fn the_reference_set_reads_as_a_rod() {
    let Some(set) = load(Material::Tpu) else { return };
    assert_eq!(set.shapes.len(), 500, "the set is 500 shapes");
    assert_eq!(set.arc.len(), 14, "14 measurement stations");
    assert!(
        set.arc.windows(2).all(|w| w[1] >= w[0]),
        "arc positions must be non-decreasing to interpolate on: {:?}",
        set.arc
    );
    assert_eq!((set.arc[0], set.arc[13]), (0.0, 1.0), "root to tip");

    let length = Material::Tpu.rod(1).length;
    for shape in set.shapes.iter().take(20) {
        // The rod is clamped at the origin and inextensible, so every shape
        // starts there and its polyline is the rod's own length.
        assert!(shape.stations[0].length() < 1e-6, "not clamped at the origin");
        let polyline: f32 = shape
            .stations
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum();
        assert!(
            (polyline - length).abs() < 0.01 * length,
            "polyline {polyline} for a {length} m rod"
        );
        assert!(
            (shape.gravity.length() - 9.81).abs() < 0.05,
            "gravity is 9.81 pointed somewhere: {:?}",
            shape.gravity
        );
    }
}

#[test]
fn a_soft_rod_matches_the_cosserat_solution() {
    // The headline number. TPU at ten links, against a solver that models the
    // rod as a continuum rather than as ten sticks.
    let Some(set) = load(Material::Tpu) else { return };
    let length = Material::Tpu.rod(1).length;
    let report = evaluate_statics(&set, run(), Some(SHAPES));

    assert_eq!(report.diverged_count(), 0, "the solver lost the chain");
    let shape = report.mean_relative_shape_error(length);
    let tip = report.mean_relative_tip_error(length);
    assert!(
        shape < 0.02,
        "mean shape error {:.2}% of rod length, wanted under 2%",
        100.0 * shape
    );
    assert!(
        tip < 0.04,
        "mean tip error {:.2}% of rod length, wanted under 4%",
        100.0 * tip
    );
}

#[test]
fn chopping_the_rod_finer_matches_the_continuum_better() {
    // The property that makes `links` a fidelity knob: it is not that ten links
    // is accurate, it is that twenty would be more so.
    let Some(set) = load(Material::Tpu) else { return };
    let length = Material::Tpu.rod(1).length;
    let coarse = evaluate_statics(&set, run().links(5), Some(SHAPES))
        .mean_relative_shape_error(length);
    let fine = evaluate_statics(&set, run().links(10), Some(SHAPES))
        .mean_relative_shape_error(length);
    assert!(
        fine < coarse,
        "five links matched at {:.2}% and ten at {:.2}% — finer should be closer",
        100.0 * coarse,
        100.0 * fine
    );
}

#[test]
fn a_rod_that_cannot_twist_cannot_match_a_three_axis_wrench() {
    // Torsion is off by default in `Rod`, because a planar bend never uses it
    // and it costs a constraint row per link. Under the reference wrenches —
    // which carry moment on all three axes — it is not optional: the rod
    // refuses the component along its own axis and settles somewhere else
    // entirely. Measured, this is the difference between a few percent and
    // tens of them, and it is the single largest term in the comparison.
    let Some(set) = load(Material::Tpu) else { return };
    let length = Material::Tpu.rod(1).length;
    let twisting = evaluate_statics(&set, run().twist(true), Some(SHAPES))
        .mean_relative_shape_error(length);
    let locked = evaluate_statics(&set, run().twist(false), Some(SHAPES))
        .mean_relative_shape_error(length);
    assert!(
        locked > 2.0 * twisting,
        "locking twist cost only {:.2}% -> {:.2}%; if torsion has stopped \
         mattering, this test and `Material::rod`'s docs should say so",
        100.0 * twisting,
        100.0 * locked
    );
}

#[test]
fn a_stiff_thin_rod_is_the_hard_case_and_we_say_how_hard() {
    // Spring steel is 3000x the modulus of the TPU rod on a tenth the radius:
    // very stiff, very light, and the worst case for a chain solver. Where the
    // soft rod is indifferent to the solver budget, this one needs four times
    // it — at 16 substeps of 64 iterations the same comparison reads 4.6% and
    // loses a shape in six.
    let Some(set) = load(Material::SpringSteel) else { return };
    let length = Material::SpringSteel.rod(1).length;
    let report = evaluate_statics(&set, run().links(5).budget(32, 128), Some(SHAPES));
    let shape = report.mean_relative_shape_error(length);

    assert_eq!(report.diverged_count(), 0, "a steel shape diverged");
    assert!(
        shape < 0.03,
        "steel matched at {:.2}% of rod length, which is worse than recorded",
        100.0 * shape
    );

    // And the cheaper budget really is the difference, rather than this being
    // a rod that is simply easier than it looks.
    let lean = evaluate_statics(&set, run().links(5), Some(SHAPES));
    assert!(
        lean.mean_relative_shape_error(length) > shape,
        "the budget no longer matters for steel ({:.2}% lean vs {:.2}% rich) — \
         if the solver's chain conditioning improved, say so in the docs",
        100.0 * lean.mean_relative_shape_error(length),
        100.0 * shape
    );
}
