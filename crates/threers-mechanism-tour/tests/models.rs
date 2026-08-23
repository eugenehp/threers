//! Every bundled mechanism, run end to end.
//!
//! These are slow for unit tests and they are worth it: each one is a whole
//! pipeline — parse, assemble, check, sweep, simulate, record, reduce, verify —
//! and the things that break them are the things that are hard to catch any
//! other way. A collider that quietly changed shape, a solver constant that
//! stopped being a fraction of the model, a drive queue that ran out of order:
//! none of those fail a unit test, and all of them fail a latch.

use threers_mechanism_tour::{model, Tour, MODELS};

fn run(id: &str) -> Tour {
    let m = model(id).unwrap_or_else(|| panic!("no bundled model {id:?}"));
    Tour::run(m.source, m.frames, m.fps, m.units_per_metre, m.substeps)
        .unwrap_or_else(|e| panic!("{id}: {e}"))
}

/// The joint reading at a given second.
fn at(tour: &Tour, joint: &str, seconds: f32) -> f32 {
    let j = tour
        .joints
        .iter()
        .find(|j| j.name == joint)
        .unwrap_or_else(|| panic!("no joint {joint:?}"));
    let frame = ((seconds * tour.fps as f32) as usize).min(j.values.len() - 1);
    j.values[frame]
}

#[test]
fn every_model_runs_and_the_geometry_agrees() {
    assert!(!MODELS.is_empty());
    for m in MODELS {
        let tour = run(m.id);
        assert!(
            tour.facts.agrees,
            "{}: the geometry disagrees with the declaration:\n{}",
            m.id,
            tour.acts
                .iter()
                .find(|a| a.title == "verify")
                .map(|a| a.body.as_str())
                .unwrap_or("")
        );
        assert_eq!(tour.facts.interferences, 0, "{}: parts overlap", m.id);
        assert!(tour.frames > 0 && !tour.pieces.is_empty(), "{}", m.id);
        assert!(tour.facts.triangles > 0, "{}: nothing to draw", m.id);
        // A run that reduces to nothing is a run where nothing moved.
        assert!(tour.facts.reduced_keys > 0, "{}", m.id);
        assert!(
            tour.facts.reduced_keys < tour.facts.keys,
            "{}: reduction gained nothing",
            m.id
        );
    }
}

/// The same source twice gives the same answer.
///
/// It did not always: a decomposed collider was built from a `HashSet`'s
/// iteration order, which is seeded per process, so a ratchet came out
/// differently on every run.
#[test]
fn a_run_is_reproducible() {
    let a = run("ratchet");
    let b = run("ratchet");
    assert_eq!(a.poses.len(), b.poses.len());
    assert_eq!(a.poses, b.poses, "the same model gave two different runs");
}

/// Forward it rides over the teeth; backward it does not move at all — and the
/// drive that fails is asking for the same 60 mm with the same force.
#[test]
fn the_ratchet_locks_one_way() {
    let tour = run("ratchet");
    let out = at(&tour, "feed", 2.4);
    assert!(out > 55.0, "the carriage should have run out: {out}");
    // Driven back from 2.6 to 4.1 and still there.
    let back = at(&tour, "feed", 4.2);
    assert!(
        (back - out).abs() < 1.0,
        "the pawl let it back: {out} -> {back}"
    );
    // And the pawl really did climb over teeth on the way out rather than the
    // carriage passing through them.
    let lifted = tour
        .joints
        .iter()
        .find(|j| j.name == "pawl_pivot")
        .map(|j| j.values.iter().cloned().fold(f32::MIN, f32::max))
        .unwrap_or(0.0);
    assert!(lifted > 5.0, "the pawl never lifted: {lifted}°");
}

/// Shut, then a drive that cannot open it, then a release, then it opens.
#[test]
fn the_latch_holds_until_it_is_released() {
    let tour = run("latch");
    let shut = at(&tour, "travel", 2.0);
    assert!(shut > 46.0, "the gate never shut: {shut}");

    // Driven to 0 between 2.2 and 3.2 and it is still shut.
    let blocked = at(&tour, "travel", 3.2);
    assert!(
        blocked > 43.0,
        "the drive walked through the latch: {blocked}"
    );

    // The plunger lifts the pawl, and then the same drive works.
    let open = at(&tour, "travel", 6.5);
    assert!(open < 2.0, "the gate never opened once released: {open}");

    // The rack and pinion read the gate's travel all the way through.
    let turns = at(&tour, "pinion_shaft", 2.0);
    assert!(turns.abs() > 180.0, "the pinion did not follow: {turns}°");
}

/// Two gear mates in series, and the ratio is a consequence rather than a number
/// anybody typed.
#[test]
fn the_gear_train_reduces_by_seven_and_a_half() {
    let tour = run("gear-train");
    let readout = tour
        .acts
        .iter()
        .find(|a| a.title.starts_with("after"))
        .map(|a| a.body.clone())
        .unwrap_or_default();
    assert!(
        readout.contains("declared -3.00:1, holds -3.00:1"),
        "stage one slipped:\n{readout}"
    );
    assert!(
        readout.contains("declared -2.50:1, holds -2.50:1"),
        "stage two slipped:\n{readout}"
    );
}

/// A closed loop: the crank goes all the way round and the rocker cannot.
#[test]
fn the_four_bar_is_a_crank_rocker() {
    let tour = run("four-bar");
    let span = |name: &str| {
        let j = tour.joints.iter().find(|j| j.name == name).unwrap();
        let lo = j.values.iter().cloned().fold(f32::MAX, f32::min);
        let hi = j.values.iter().cloned().fold(f32::MIN, f32::max);
        hi - lo
    };
    // The crank's reading wraps at ±360, so a full turn shows as most of it.
    assert!(span("crank_pin") > 180.0, "the crank did not turn round");
    let rocker = span("rocker_ground");
    assert!(
        (10.0..90.0).contains(&rocker),
        "the rocker should swing and not rotate: {rocker}°"
    );
    // Grübler counts a planar loop in three dimensions as over-constrained. It
    // is, on paper, and it moves anyway.
    assert!(tour.facts.mobility < 0);
}

/// Nothing declares the follower's motion; it comes out of the contact.
#[test]
fn the_cam_lifts_the_follower_without_being_told_to() {
    let tour = run("cam");
    let lift = tour.joints.iter().find(|j| j.name == "lift").unwrap();
    let lo = lift.values.iter().cloned().fold(f32::MAX, f32::min);
    let hi = lift.values.iter().cloned().fold(f32::MIN, f32::max);
    // The cam is 8 mm off centre, so the throw is about 16 either way — and the
    // follower must not sink through the cam to get there.
    assert!(
        (12.0..20.0).contains(&(hi - lo)),
        "the throw should be about 16: {lo} .. {hi}"
    );
}
