//! A boolean that cannot be verified watertight must not grow in number.
//!
//! When the exact arrangement declines, the caller falls back to the float
//! evaluator; when *that* is not closed either and cannot be healed, the mesh is
//! returned anyway. For a preview that is a reasonable trade. For anything
//! downstream it is not: a mesh with cracks cannot be printed, cannot have mass
//! properties computed, and its defects compound through the next boolean.
//!
//! That path used to be a `log::warn` and nothing else, which is why it went
//! unnoticed on the reference model until someone measured. It is counted now,
//! and this gate pins the count.
//!
//! **The baseline below is a known defect, not a target.** It is here so the
//! number cannot quietly grow while someone works on the kernel, and so that
//! driving it to zero shows up as a failing test asking to be updated.

#![cfg(feature = "openscad")]

use threers::exact_csg::{reset_unverified_booleans, unverified_booleans};
use threers::openscad::animate::ScadAnimation;

/// Booleans on the reference model that the kernel cannot verify.
///
/// Traced as far as `cdt_face` discarding real surface on four faces, which
/// opens a mesh that arrived closed — see the known-issues entry in CHANGELOG.md
/// for the measurements and for the fixes that were tried and did not work.
#[cfg(not(feature = "manifold"))]
const KNOWN_UNVERIFIED: usize = 4;

/// With the Manifold backend there is nothing to fall back from — it guarantees
/// manifold output by construction, so the count must be zero and stay there.
#[cfg(feature = "manifold")]
const KNOWN_UNVERIFIED: usize = 0;

// `KNOWN_UNVERIFIED` is 0 under `manifold`, which makes the `n <= KNOWN_UNVERIFIED`
// below degenerate into `n <= 0` for that one configuration. The comparison is
// still the one we want — "did not grow" — in every other build.
#[allow(clippy::absurd_extreme_comparisons)]
#[test]
fn unverified_boolean_count_does_not_grow() {
    let src = match std::fs::read_to_string("examples/scad_animate.scad") {
        Ok(s) => s,
        Err(_) => {
            eprintln!("skipping: reference model not present");
            return;
        }
    };
    reset_unverified_booleans();
    let mut anim = ScadAnimation::from_source(&src).frames(1);
    anim.evaluate().expect("model evaluates");
    let n = unverified_booleans();

    assert!(
        n <= KNOWN_UNVERIFIED,
        "unverified booleans grew: {n} > {KNOWN_UNVERIFIED}. A boolean that was \
         watertight is not any more — bisect the kernel change rather than \
         raising this number."
    );
    #[cfg(not(feature = "manifold"))]
    if n < KNOWN_UNVERIFIED {
        panic!(
            "unverified booleans fell to {n} (from {KNOWN_UNVERIFIED}) — good. \
             Lower KNOWN_UNVERIFIED to {n} to lock the improvement in."
        );
    }
}

/// The counter has to actually count, or the gate above is decorative.
#[test]
fn the_counter_is_wired_up() {
    reset_unverified_booleans();
    assert_eq!(unverified_booleans(), 0);
    threers::exact_csg::note_unverified_boolean_for_test();
    assert_eq!(unverified_booleans(), 1);
    reset_unverified_booleans();
    assert_eq!(unverified_booleans(), 0);
}
