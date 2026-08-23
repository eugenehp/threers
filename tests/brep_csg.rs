//! Stage 2 acceptance suite for the `brep-csg` feature.
//!
//! What Stage 2 delivers is an *accelerator* inside the exact-CSG kernel: when
//! both operands carry surface provenance, a candidate triangle pair is resolved
//! by asking what its surfaces do rather than by intersecting the triangles.
//!
//! The suite is organised around the three things that have to hold:
//!
//! 1. **The closed forms are right.** Sampled curves lie on both surfaces, and
//!    `Disjoint` is only ever reported when it is provable.
//! 2. **The accelerator actually fires** on real booleans — otherwise (3) is
//!    trivially true and means nothing.
//! 3. **It never makes an answer worse.** A tagged boolean is never worse than
//!    the same boolean with its provenance stripped. This is the safety
//!    property in its testable form, and it has already earned its keep: it
//!    caught a seam-projection hook that turned two overlapping spheres from
//!    `Exact` into `NeedsArrangement`.

#![cfg(feature = "brep-csg")]

use threers::brep::{ssi, Curve3d, SsiResult, Surface};
use threers::core::BufferGeometry;
use threers::exact_csg::{analytic_report, boolean, BooleanOutcome, Op};
use threers::{cube, cylinder, sphere};

const OPS: [(Op, &str); 3] = [
    (Op::Union, "union"),
    (Op::Difference, "difference"),
    (Op::Intersection, "intersection"),
];

/// The models the kernel is exercised on. Each is a pair of solids plus a name.
fn models() -> Vec<(&'static str, BufferGeometry, BufferGeometry)> {
    vec![
        (
            "coaxial cylinders",
            cylinder(4.0, 2.0).to_geometry(),
            cylinder(4.0, 2.0).translate([0.0, 2.0, 0.0]).to_geometry(),
        ),
        (
            "concentric cylinders",
            cylinder(4.0, 2.0).to_geometry(),
            cylinder(6.0, 1.0).to_geometry(),
        ),
        (
            "crossed cylinders",
            cylinder(6.0, 1.5).to_geometry(),
            cylinder(6.0, 1.5).rotate([90.0, 0.0, 0.0]).to_geometry(),
        ),
        (
            "translated cubes",
            cube([2.0, 2.0, 2.0]).to_geometry(),
            cube([2.0, 2.0, 2.0])
                .translate([1.0, 0.0, 0.0])
                .to_geometry(),
        ),
        (
            "overlapping spheres",
            sphere(2.0).to_geometry(),
            sphere(2.0).translate([2.0, 0.0, 0.0]).to_geometry(),
        ),
        (
            "near-identical spheres",
            sphere(2.0).to_geometry(),
            sphere(2.0).translate([0.5, 0.0, 0.0]).to_geometry(),
        ),
        (
            "thin-walled tube",
            cylinder(4.0, 2.0).to_geometry(),
            cylinder(4.0, 1.9).to_geometry(),
        ),
        (
            "cylinder through a cube",
            cube([4.0, 4.0, 4.0]).to_geometry(),
            cylinder(8.0, 1.0).to_geometry(),
        ),
        (
            "sphere in a cube",
            cube([3.0, 3.0, 3.0]).to_geometry(),
            sphere(2.0).to_geometry(),
        ),
    ]
}

fn stripped(g: &BufferGeometry) -> BufferGeometry {
    let mut c = g.clone();
    c.surfaces = None;
    c
}

fn is_exact(o: &BooleanOutcome) -> bool {
    matches!(o, BooleanOutcome::Exact(_))
}

fn vertex_count(o: &BooleanOutcome) -> Option<usize> {
    match o {
        BooleanOutcome::Exact(g) => g.get_attribute("position").map(|a| a.count()),
        BooleanOutcome::NeedsArrangement => None,
    }
}

// ---------------------------------------------------------------------------
// 1. the closed forms are right
// ---------------------------------------------------------------------------

#[test]
fn the_kernels_named_degeneracies_are_resolved_analytically() {
    // `src/exact_csg/mod.rs` names two remaining failure modes, the first being
    // "two identical primitives translated along an axis". As *surfaces* that is
    // a struct comparison, not a numeric problem.
    let a = Surface::cylinder([0.0; 3], [0.0, 1.0, 0.0], 2.0);
    let b = Surface::cylinder([0.0, 2.0, 0.0], [0.0, 1.0, 0.0], 2.0);
    assert_eq!(ssi(&a, &b), SsiResult::Coincident { opposite: false });

    // And the Steinmetz seam, which the mesh kernel handles worst of all, is two
    // exact ellipses.
    let x = Surface::cylinder([0.0; 3], [0.0, 1.0, 0.0], 1.5);
    let y = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.5);
    let SsiResult::Curves(c) = ssi(&x, &y) else {
        panic!("crossed cylinders should resolve to curves");
    };
    assert_eq!(c.len(), 2);
    assert!(c.iter().all(|k| k.kind() == "ellipse"));
}

#[test]
fn reported_curves_lie_on_both_surfaces() {
    // A closed form that is subtly wrong produces a plausible curve in the wrong
    // place. Only measuring against *both* surfaces catches that.
    let pairs = [
        (
            Surface::sphere([0.0; 3], 3.0),
            Surface::sphere([4.0, 0.0, 0.0], 3.0),
        ),
        (
            Surface::plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
        ),
        (
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
            Surface::cylinder([0.0; 3], [1.0, 0.0, 0.0], 2.0),
        ),
        (
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
            Surface::cone([0.0; 3], [0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_4),
        ),
        (
            Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
            Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0),
        ),
    ];
    for (a, b) in &pairs {
        let SsiResult::Curves(curves) = ssi(a, b) else {
            panic!("{} vs {}: expected curves", a.kind(), b.kind());
        };
        for c in &curves {
            let samples: Vec<[f64; 3]> = match c {
                Curve3d::Point(p) => vec![*p],
                Curve3d::Line { .. } => (-10..=10).map(|i| c.point(i as f64)).collect(),
                _ => (0..48)
                    .map(|i| c.point(std::f64::consts::TAU * i as f64 / 48.0))
                    .collect(),
            };
            for p in samples {
                assert!(a.distance(p) < 1e-9, "{} vs {}: off A", a.kind(), b.kind());
                assert!(b.distance(p) < 1e-9, "{} vs {}: off B", a.kind(), b.kind());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. the accelerator fires
// ---------------------------------------------------------------------------

#[test]
fn provenance_reaches_the_kernel_through_transforms() {
    // Every interesting boolean has a transform on at least one operand, so if
    // `bake_matrix` dropped the table the accelerator would never see anything.
    // This is the wiring that makes Stage 2 reachable at all.
    for (name, a, b) in models() {
        assert!(a.surface_table().is_some(), "{name}: operand A untagged");
        assert!(b.surface_table().is_some(), "{name}: operand B untagged");
    }
}

#[test]
fn the_accelerator_resolves_pairs_on_the_models_it_should() {
    // Without this, "never worse" would be satisfied by an accelerator that does
    // nothing at all.
    let mut fired = 0usize;
    for (name, a, b) in models() {
        let r = analytic_report(&a, &b);
        assert!(r.active, "{name}: provenance not seen");
        assert!(r.candidates > 0, "{name}: no candidate pairs");
        assert_eq!(
            r.skipped + r.same_surface + r.curves + r.numeric,
            r.candidates,
            "{name}: decisions do not account for every candidate"
        );
        if r.skipped + r.same_surface + r.curves > 0 {
            fired += 1;
        }
    }
    assert!(
        fired >= 5,
        "the accelerator resolved something on only {fired} of {} models",
        models().len()
    );
}

#[test]
fn coaxial_cylinders_are_recognised_as_the_same_surface_by_the_kernel() {
    let a = cylinder(4.0, 2.0).to_geometry();
    let b = cylinder(4.0, 2.0).translate([0.0, 2.0, 0.0]).to_geometry();
    let r = analytic_report(&a, &b);
    assert!(
        r.same_surface > 0,
        "the shared cylinder was not recognised: {r:?}"
    );
}

#[test]
fn provably_disjoint_surface_pairs_are_skipped() {
    // `Disjoint` pays in a narrower band than it first appears, and the measured
    // shape of that band is worth recording.
    //
    // The AABB broad phase is cheaper and already eliminates anything far apart:
    // two *concentric* spheres a whole unit apart produce **zero** candidate
    // pairs, so the closed form is never even consulted. What `Disjoint` catches
    // is near-miss geometry, where the boxes overlap but the surfaces still
    // cannot meet — a thin-walled tube being the everyday case.
    let a = cylinder(4.0, 2.0).to_geometry();
    let b = cylinder(4.0, 1.9).to_geometry();
    let r = analytic_report(&a, &b);
    assert!(
        r.skipped > 0,
        "a 0.1-thick wall should be proven apart: {r:?}"
    );
    // Same model, and the coincident cap planes are recognised too.
    assert!(
        r.same_surface > 0,
        "the shared cap planes were missed: {r:?}"
    );

    // The broad phase really does pre-empt the easy case.
    let far = analytic_report(&sphere(2.0).to_geometry(), &sphere(1.0).to_geometry());
    assert_eq!(
        far.candidates, 0,
        "concentric spheres should not even reach the narrow phase"
    );

    // And every skipped pair really is disjoint — a false `Disjoint` is the one
    // way this layer could produce a *worse* answer than the numeric path, by
    // skipping a real intersection.
    let sa = Surface::cylinder([0.0; 3], [0.0, 1.0, 0.0], 2.0);
    let sb = Surface::cylinder([0.0; 3], [0.0, 1.0, 0.0], 1.0);
    assert_eq!(ssi(&sa, &sb), SsiResult::Disjoint);
    for i in 0..64 {
        for j in 0..16 {
            let u = std::f64::consts::TAU * i as f64 / 64.0;
            let v = -4.0 + 8.0 * j as f64 / 16.0;
            assert!(
                sb.distance(sa.point(u, v)) > 0.5,
                "the two cylinders come closer than their radii differ"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. it never makes an answer worse
// ---------------------------------------------------------------------------

#[test]
fn a_tagged_boolean_is_never_worse_than_an_untagged_one() {
    // The safety property, in the strongest form that is testable inside one
    // build. `Exact` must never become `NeedsArrangement`.
    for (name, a, b) in models() {
        let (sa, sb) = (stripped(&a), stripped(&b));
        for (op, opname) in OPS {
            let with = boolean(&a, &b, op);
            let without = boolean(&sa, &sb, op);
            assert!(
                is_exact(&with) || !is_exact(&without),
                "{name} {opname}: provenance turned Exact into NeedsArrangement"
            );
        }
    }
}

#[test]
fn a_tagged_result_is_still_watertight() {
    // Not merely "still Exact" — the mesh the kernel hands back must still pass
    // its own closed-manifold gate, which `repair_geometry` reports by declining
    // to change anything that is already closed.
    use threers::exact_csg::repair_geometry;
    for (name, a, b) in models() {
        for (op, opname) in OPS {
            if let BooleanOutcome::Exact(g) = boolean(&a, &b, op) {
                assert!(
                    repair_geometry(&g).is_none(),
                    "{name} {opname}: the result needed repair, so it was not closed"
                );
            }
        }
    }
}

#[test]
fn results_that_were_already_exact_stay_exact_and_the_same_size() {
    // A stronger no-regression check on the cases the kernel already handled:
    // the accelerator may change *which* segments are generated, but on models
    // where it changes nothing the output must be identical.
    for (name, a, b) in models() {
        let (sa, sb) = (stripped(&a), stripped(&b));
        let r = analytic_report(&a, &b);
        // Only assert identity where the accelerator made no decision at all.
        if r.skipped + r.same_surface > 0 {
            continue;
        }
        for (op, opname) in OPS {
            let with = boolean(&a, &b, op);
            let without = boolean(&sa, &sb, op);
            assert_eq!(
                vertex_count(&with),
                vertex_count(&without),
                "{name} {opname}: output changed although the accelerator resolved nothing"
            );
        }
    }
}

#[test]
fn stripping_provenance_is_what_forces_the_numeric_path() {
    let a = cylinder(4.0, 2.0).to_geometry();
    let b = cylinder(4.0, 2.0).translate([0.0, 2.0, 0.0]).to_geometry();
    let live = analytic_report(&a, &b);
    let dead = analytic_report(&stripped(&a), &stripped(&b));

    assert!(live.active && live.same_surface > 0);
    assert!(!dead.active, "stripped operands must not be seen as tagged");
    assert_eq!(dead.skipped + dead.same_surface + dead.curves, 0);
    assert_eq!(
        dead.numeric, dead.candidates,
        "every pair must fall through"
    );
}

// ---------------------------------------------------------------------------
// what is not yet delivered
// ---------------------------------------------------------------------------

#[test]
fn coaxial_cylinders_now_resolve_where_they_used_to_defer() {
    // The kernel's first named degeneracy — "two identical primitives translated
    // along an axis" — now returns `Exact` for all three ops.
    //
    // The fix was *not* in `corefine`. The accelerator already recognised the
    // shared cylinder there; what failed was the classifier, which gates on
    // `coplanar()` — an exact `orient3d == 0` test that refined vertices, coming
    // out of the CDT a few ULPs off their plane, cannot pass. Sub-triangles then
    // fell through to a ray-parity test whose ray starts *on* the other solid's
    // boundary, and the coin flip landed both-keep on some facets and
    // neither-keep on others: duplicate triangles in one place, holes in
    // another, 34 malformed edges, all on the two rim circles.
    //
    // Surface identity is not fragile that way, and supplying it as a fallback
    // is the whole fix.
    let a = cylinder(4.0, 2.0).to_geometry();
    let b = cylinder(4.0, 2.0).translate([0.0, 2.0, 0.0]).to_geometry();

    for (op, name) in OPS {
        assert!(
            is_exact(&boolean(&a, &b, op)),
            "coaxial cylinders {name}: still deferring"
        );
    }

    // And it is provenance that does it — strip the tables and it defers again.
    let (sa, sb) = (stripped(&a), stripped(&b));
    assert!(
        !is_exact(&boolean(&sa, &sb, Op::Union)),
        "untagged coaxial cylinders should still defer; if not, the credit \
         belongs elsewhere and this test is measuring nothing"
    );
}

#[test]
fn seam_points_are_sharpened_along_their_own_edge_not_relocated() {
    // The right way to make a seam exact. A `tri_tri_segment` endpoint lies on a
    // mesh edge and has to stay there; sliding it *along* that edge until it
    // sits on the other surface is a bracketed root find on a signed distance.
    //
    // (Projecting onto the surfaces' intersection curve instead moves it off the
    // edge and breaks the refinement — measured, then removed.)
    let sphere_surf = Surface::sphere([2.0, 0.0, 0.0], 2.0);

    // An edge that crosses the sphere, with an endpoint deliberately off by the
    // ~1e-6 the numeric path carries.
    let (e0, e1) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let exact = sphere_surf
        .edge_crossing(e0, e1, [0.0; 3])
        .expect("the edge crosses the sphere");
    assert!(
        sphere_surf.distance(exact) < 1e-15,
        "the root find left the point {} off the surface",
        sphere_surf.distance(exact)
    );
    // It stayed on the edge.
    assert!(exact[1].abs() < 1e-15 && exact[2].abs() < 1e-15);

    // An edge that does not cross is left alone rather than guessed at.
    assert!(sphere_surf
        .edge_crossing([5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0; 3])
        .is_none());
}

#[test]
fn the_crossed_cylinder_seam_still_defers_and_this_test_records_that() {
    // Honest accounting. Two cylinders crossing at 90° remain `NeedsArrangement`
    // at this radius and tessellation. This is *not* the coincident-face problem
    // above — their surfaces genuinely cross, the accelerator correctly reports
    // two exact ellipses, and the failure is sliver triangles out of the CDT on
    // a genuinely curved∧curved seam.
    //
    // Note it is geometry-specific rather than a categorical gap: the corpus's
    // own `cyl⊥cyl cross` case (`cylinder(4, 0.8)`) resolves. Closing this needs
    // work on the CDT's sliver handling, not on the analytic layer.
    //
    // Asserted so that closing it breaks the test rather than letting the claim
    // drift.
    let a = cylinder(6.0, 1.5).to_geometry();
    let b = cylinder(6.0, 1.5).rotate([90.0, 0.0, 0.0]).to_geometry();
    assert!(
        !is_exact(&boolean(&a, &b, Op::Union)),
        "crossed cylinders now resolve — good; update this test and the plan"
    );
}
