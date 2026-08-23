//! What the B-rep boolean resolves, and what it declines.
//!
//! The property under test is not "it works" — it is that **a result is never
//! worse than a decline**. A boolean that returns a solid with a hole in it is
//! worse than one that returns nothing, because the caller cannot tell. So every
//! case here asserts one of exactly two outcomes: a watertight solid of the
//! right volume, or a `Declined` naming the reason.
//!
//! The declines are asserted too. They are the map of what is not implemented,
//! and a decline turning into a *result* is as much a change worth noticing as
//! the other way round — it should come with a volume to check.

#![cfg(feature = "brep-kernel")]

use std::f64::consts::PI;
use threers::brep::intersect::march;
use threers::brep::{Body, BooleanOp, Declined, Defect, Surface};
use threers::nurbs::NurbsSurface;

/// Volume of a body's tessellation, by the divergence theorem. Asserts closure
/// on the way past, since an open mesh has no volume to speak of.
fn volume(body: &Body, tolerance: f64) -> f64 {
    let mut b = body.clone();
    b.refine_edges(tolerance);
    let (mesh, report) = b.tessellate(tolerance);
    assert!(
        report.is_closed() && report.carried_through == 0,
        "{} open edges, {} faces unfilled",
        report.boundary_edges,
        report.carried_through
    );
    let pos = &mesh.get_attribute("position").unwrap().array;
    let idx = mesh.index.as_ref().unwrap();
    let v = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    idx.chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>()
        .abs()
}

fn check(a: &Body, b: &Body, op: BooleanOp, expected: f64) {
    let r = a.boolean(b, op, 1e-3).expect("this pair has a closed form");
    let got = volume(&r, 1e-3);
    assert!(
        (got - expected).abs() / expected < 0.01,
        "volume {got}, expected {expected}"
    );
}

fn plate() -> Body {
    Body::cuboid([10.0, 8.0, 2.0])
}

fn through_drill() -> Body {
    Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0)
}

#[test]
fn a_through_hole() {
    let (a, b) = (plate(), through_drill());
    let bore = PI * 4.0 * 2.0;
    check(&a, &b, BooleanOp::Difference, 160.0 - bore);
    check(&a, &b, BooleanOp::Union, 160.0 + PI * 4.0 * 6.0 - bore);
    check(&a, &b, BooleanOp::Intersection, bore);
}

#[test]
fn a_blind_hole() {
    // The drill starts inside the material, so the bore is capped — the cut
    // face is the drill's own end, not a face of the plate.
    let a = Body::cuboid([10.0, 8.0, 4.0]);
    let b = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 6.0);
    let bore = PI * 4.0 * 2.0;
    check(&a, &b, BooleanOp::Difference, 320.0 - bore);
    check(&a, &b, BooleanOp::Union, 320.0 + PI * 4.0 * 6.0 - bore);
    check(&a, &b, BooleanOp::Intersection, bore);
}

#[test]
fn a_cross_hole() {
    // Drilled across rather than down, so the bore's seams land on the *side*
    // walls. Nothing about the algorithm should care, and this is the test that
    // says so.
    let a = Body::cuboid([10.0, 8.0, 4.0]);
    let b = Body::cylinder([-9.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 18.0);
    let bore = PI * 10.0;
    check(&a, &b, BooleanOp::Difference, 320.0 - bore);
    check(&a, &b, BooleanOp::Union, 320.0 + PI * 18.0 - bore);
    check(&a, &b, BooleanOp::Intersection, bore);
}

#[test]
fn a_curved_face_split_by_a_curved_one() {
    // A sphere bored through: the seam is not a plane cut, and the remainder has
    // the exact volume of the "napkin ring", 4/3·π·(R²−a²)^{3/2}.
    let ball = Body::sphere([0.0; 3], 3.0);
    let drill = Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0);
    let h = (9.0f64 - 1.0).sqrt();
    check(
        &ball,
        &drill,
        BooleanOp::Difference,
        4.0 / 3.0 * PI * h.powi(3),
    );
}

#[test]
fn a_curved_face_split_by_a_flat_one() {
    // A sphere sitting in the top of a box: the box's top face comes back with a
    // circular hole, and the sphere's lower half becomes a dimple.
    let box_ = Body::cuboid([6.0, 6.0, 6.0]);
    let ball = Body::sphere([0.0, 0.0, 3.0], 2.0);
    let cap = 2.0 / 3.0 * PI * 8.0; // the half below z = 3
    check(&box_, &ball, BooleanOp::Difference, 216.0 - cap);
    check(
        &box_,
        &ball,
        BooleanOp::Union,
        216.0 + 4.0 / 3.0 * PI * 8.0 - cap,
    );
    check(&box_, &ball, BooleanOp::Intersection, cap);
}

#[test]
fn a_curve_is_found_on_a_face_whose_seam_has_moved() {
    // A face cut open at a seam runs its `u` from wherever that seam fell — a
    // ball with a square column bored through it has a sphere face going from
    // 3.069 to 9.352 — while `Surface::invert` answers in the surface's own
    // terms. Testing the raw parameter against that face's loops rejects
    // everything on the far side of the seam, so a later cut through such a
    // face found its intersection curve and was told the curve did not reach
    // the face it was on.
    let ball = Body::sphere([0.0; 3], 3.0);
    let column = Body::cuboid([2.0, 2.0, 20.0]);
    let bored = ball
        .boolean(&column, BooleanOp::Difference, 1e-3)
        .expect("closed form");

    // The sphere face is the one whose seam moved.
    let sphere = bored
        .faces()
        .iter()
        .find(|f| bored.surfaces()[f.surface].kind() == "sphere")
        .expect("the ball is still a sphere");
    assert!(
        sphere.u_range.0 > 1.0,
        "this test needs a shifted seam, got {:?}",
        sphere.u_range
    );

    // A drill through the wall of the bore, at a `u` on the other side of that
    // seam. Whatever the boolean makes of it, it must not silently miss the
    // curve: the tell is that it does *not* come back unchanged.
    let drill = Body::cylinder([1.3, 0.7, -30.0], [0.0, 0.0, 1.0], 0.35, 60.0);
    match bored.boolean(&drill, BooleanOp::Difference, 1e-3) {
        Ok(cut) => assert!(
            cut.faces().len() > bored.faces().len(),
            "the drill left no trace: {} faces before, {} after",
            bored.faces().len(),
            cut.faces().len()
        ),
        // Declining is allowed; quietly ignoring the drill is not.
        Err(Declined::NeedsArrangement { .. }) => {}
        Err(other) => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn a_result_can_be_cut_again() {
    // The property `what_comes_out_is_a_solid_that_can_go_back_in` makes
    // possible, actually exercised: take a result and cut it.
    //
    // Not every one survives — a rod with a cross-hole, a bored ball and a
    // rounded edge all decline the second cut, for reasons the plan records —
    // so this pins the ones that do rather than claiming the general case. It
    // exists because until now that was measured only by scratch tests written
    // and deleted each time, which is how a revert once went unnoticed for two
    // iterations.
    // Each with somewhere its own drill can bite: the plate's existing bore is
    // two wide, so a drill inside that is no test of anything.
    let at = |x: f64, y: f64| Body::cylinder([x, y, -30.0], [0.0, 0.0, 1.0], 0.35, 60.0);
    let cases: [(&str, Body, Body, Body); 2] = [
        (
            "two boxes overlapping at a corner",
            Body::cuboid([4.0; 3]),
            Body::cuboid([4.0; 3]).translated([2.0, 2.0, 2.0]).unwrap(),
            at(1.3, 0.7),
        ),
        ("a bored plate", plate(), through_drill(), at(3.5, 2.5)),
    ];
    for (name, a, b, drill) in &cases {
        for op in [BooleanOp::Difference, BooleanOp::Union] {
            let once = a.boolean(b, op, 1e-3).expect("the first cut resolves");
            let twice = once
                .boolean(drill, BooleanOp::Difference, 1e-3)
                .unwrap_or_else(|e| panic!("{name} {op:?} could not be cut again: {e:?}"));
            assert!(
                twice.is_valid_solid(1e-3),
                "{name} {op:?}: the second cut is not a solid"
            );
            // A face count says the drill was *mentioned*, not that it cut. A
            // bored ball passed this line for as long as it has existed while
            // taking 0.0138 out of a body it should have taken 2.01 out of —
            // closed, valid, one face richer, and drilled to a hundredth of a
            // percent. Every case here is at least two thick where its drill
            // goes in, so a bit of radius 0.35 has to remove `pi·r²·2`, and the
            // margin below that is wide enough to be about the drilling and not
            // about the mesh.
            let bit = PI * 0.35 * 0.35 * 2.0;
            let gone = volume(&once, 1e-3) - volume(&twice, 1e-3);
            assert!(
                gone > bit * 0.5,
                "{name} {op:?}: the drill removed {gone:.4}, and cannot have removed less than {:.4}",
                bit * 0.5
            );
        }
    }

    // A bored ball drilled off-axis, which has a history worth keeping. It used
    // to resolve on a trace that found one of the drill's two rings, so the hole
    // had an entry and no exit: the body came back closed, valid and one face
    // richer having removed 0.0138 of the 1.997 it should have. Seeding by the
    // curve's own scale found both rings and turned that into a decline; reading
    // a ring's *turn* rather than its parameter span made it an answer.
    //
    // Checked by weight, because weight is what was wrong with it.
    let ball = Body::sphere([0.0; 3], 3.0);
    let bore = Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0);
    let bored = ball
        .boolean(&bore, BooleanOp::Difference, 1e-3)
        .expect("a ball bores");
    let drilled = bored
        .boolean(&at(1.3, 0.7), BooleanOp::Difference, 1e-3)
        .expect("a bored ball takes an off-axis drill");
    let gone = volume(&bored, 1e-3) - volume(&drilled, 1e-3);
    // The drill's footprint on the sphere, by quadrature: the exact figure is
    // 1.997, and a sphere tessellated at 1e-3 comes in about two per cent light.
    assert!(
        (gone - 1.997).abs() / 1.997 < 0.03,
        "the off-axis drill removed {gone:.4}, and 1.997 is right"
    );
}

#[test]
fn what_comes_out_is_a_solid_that_can_go_back_in() {
    // A boolean's result being *closed* is not the same as its being a solid.
    // A face's boundary comes from its trim loops, so the tessellation shuts
    // whatever the edge list says; `is_valid_solid` reads the edge list, and
    // that is what the next boolean checks. Results used to fail it — a rod with
    // a cross-hole could not be touched again, and a corner overlap's union came
    // back as *two* closed shells, which the kernel could not tell from one
    // solid and would happily accept as an input.
    let cases: [(&str, Body, Body); 6] = [
        (
            "corner overlap",
            Body::cuboid([4.0; 3]),
            Body::cuboid([4.0; 3]).translated([2.0, 2.0, 2.0]).unwrap(),
        ),
        (
            "bore across a rod",
            Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0),
            Body::cylinder([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 10.0),
        ),
        ("plate and drill", plate(), through_drill()),
        (
            "sphere and drill",
            Body::sphere([0.0; 3], 3.0),
            Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0),
        ),
        (
            "column in a ball",
            Body::sphere([0.0; 3], 3.0),
            Body::cuboid([2.0, 2.0, 20.0]),
        ),
        (
            "round an edge",
            Body::cuboid([10.0, 8.0, 4.0]),
            Body::cylinder([0.0, -6.0, 0.0], [0.0, 1.0, 0.0], 3.0, 12.0)
                .translated([5.0, 0.0, 2.0])
                .unwrap(),
        ),
    ];
    for (name, a, b) in &cases {
        for op in [BooleanOp::Difference, BooleanOp::Union] {
            let r = a.boolean(b, op, 1e-3).expect("all of these resolve");
            assert!(
                r.defects(1e-3).is_empty(),
                "{name} {op:?}: {:?}",
                r.defects(1e-3)
            );
            let shells = r.shells();
            assert_eq!(shells.len(), 1, "{name} {op:?}: {} shells", shells.len());
            assert!(shells[0].closed, "{name} {op:?}: the shell does not close");
            assert!(r.is_valid_solid(1e-3), "{name} {op:?} is not a solid");
        }
    }
}

#[test]
fn solids_that_do_not_meet_still_have_all_three_answers() {
    // A boolean whose inputs never touch has answers that need no intersection
    // at all, and getting them wrong is easy: the union of two disjoint solids
    // is one body with *two shells*, and a kernel that assumes one shell either
    // drops a piece or returns something open.
    let here = Body::cuboid([2.0; 3]);
    let far = Body::cuboid([2.0; 3]).translated([10.0, 0.0, 0.0]).unwrap();
    check(&here, &far, BooleanOp::Difference, 8.0);
    check(&here, &far, BooleanOp::Union, 16.0);
    let empty = here
        .boolean(&far, BooleanOp::Intersection, 1e-3)
        .expect("nothing in common is an answer, not a failure");
    assert!(empty.faces().is_empty(), "{} faces", empty.faces().len());

    // Two shells, twelve faces, and closed.
    let both = here.boolean(&far, BooleanOp::Union, 1e-3).unwrap();
    assert_eq!(both.faces().len(), 12);

    // And one solid wholly inside another, which meets nowhere either.
    let outer = Body::cuboid([6.0; 3]);
    let inner = Body::cuboid([2.0; 3]);
    check(&outer, &inner, BooleanOp::Difference, 216.0 - 8.0);
    check(&outer, &inner, BooleanOp::Intersection, 8.0);
    check(&outer, &inner, BooleanOp::Union, 216.0);
}

#[test]
fn a_nurbs_surface_is_traced_like_any_other() {
    // The NURBS path through the boolean had never been exercised, and it is
    // not a declining one: `ssi` has no case for `Surface::Nurbs`, so such a
    // pair falls to `Unknown` and is *marched* — the same numeric tracer a
    // cross-drilled hole's quartic seam uses. An untested path that produces
    // something is worth more attention than one that refuses to.
    //
    // A rational quadratic arc of three control points, weighted `1, √½, 1`,
    // is exactly a quarter circle; extruded, exactly a quarter cylinder. So the
    // answer is known: cut it at `z = 2` and the seam is an arc of radius 3.
    let (r, h) = (3.0f64, 4.0f64);
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let grid = [
        [r, 0.0, 0.0],
        [r, 0.0, h],
        [r, r, 0.0],
        [r, r, h],
        [0.0, r, 0.0],
        [0.0, r, h],
    ];
    let weights = [1.0, 1.0, w, w, 1.0, 1.0];
    let patch = NurbsSurface::new(
        2,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        3,
        2,
        &grid,
        Some(&weights),
    )
    .expect("a 3 by 2 grid");
    let surface = Surface::nurbs(patch);

    // It really is the cylinder, or the rest of the test proves nothing.
    for (u, v) in [(0.0, 0.0), (0.25, 0.5), (0.5, 0.5), (1.0, 1.0)] {
        let p = surface.point(u, v);
        assert!(
            (p[0].hypot(p[1]) - r).abs() < 1e-12,
            "({u},{v}) is not on it"
        );
    }

    let plane = Surface::plane([0.0, 0.0, h * 0.5], [0.0, 0.0, 1.0]);
    let curves = march(
        &surface,
        &plane,
        ([-1.0, -1.0, -1.0], [r + 1.0, r + 1.0, h + 1.0]),
        1e-3,
    );
    assert_eq!(curves.len(), 1, "one arc, not {}", curves.len());

    for i in 0..=200 {
        let p = curves[0].point(i as f64 / 200.0);
        assert!(
            (p[0].hypot(p[1]) - r).abs() <= 1e-3,
            "off the patch at {p:?}"
        );
        assert!((p[2] - h * 0.5).abs() <= 1e-9, "off the plane at {p:?}");
    }
}

#[test]
fn touching_is_told_apart_from_crossing() {
    // A cylinder of radius 4 about `x = 1` passes through a torus of `R = 4,
    // r = 1`, and at `(-3, 0, 0)` it is exactly 4 from its own axis and exactly
    // on the torus's inner equator. The two surfaces touch there rather than
    // crossing, so the intersection pinches to a point and two of its branches
    // meet — which the tracer cannot follow, because `settle` has no direction
    // to step in when the normals are parallel.
    //
    // What matters is that this is said, not that it is solved. Left to run it
    // came back `NotWatertight`, which reads as a fault in the kernel and is
    // not one.
    let torus = Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
    let grazing = Body::cylinder([1.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0);
    for op in [
        BooleanOp::Difference,
        BooleanOp::Union,
        BooleanOp::Intersection,
    ] {
        assert!(
            matches!(
                torus.boolean(&grazing, op, 1e-3),
                Err(Declined::TangentialContact { .. })
            ),
            "{op:?} should say the two touch"
        );
    }

    // And the same pair on-axis, which crosses cleanly, still resolves.
    let square = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0);
    assert!(torus.boolean(&square, BooleanOp::Difference, 1e-3).is_ok());

    // The other shape of the same thing: an intersection that is a single
    // *point*. Two spheres exactly a diameter apart meet at one, and there is
    // no curve to split a face along. This used to come back
    // `NeedsArrangement`, which is not what is wrong with it — nothing is
    // arranged, the two graze.
    let ball = Body::sphere([0.0; 3], 1.0);
    let kissing = Body::sphere([2.0, 0.0, 0.0], 1.0);
    for op in [
        BooleanOp::Difference,
        BooleanOp::Union,
        BooleanOp::Intersection,
    ] {
        assert!(
            matches!(
                ball.boolean(&kissing, op, 1e-3),
                Err(Declined::TangentialContact { .. })
            ),
            "{op:?} should say the two touch"
        );
    }

    // A hair further apart and they do not touch at all, which is an answer.
    let apart = Body::sphere([2.01, 0.0, 0.0], 1.0);
    assert!(ball.boolean(&apart, BooleanOp::Union, 1e-3).is_ok());
}

#[test]
fn a_trimmed_curved_face_holds_its_tolerance() {
    // The 1% the other volume checks allow is not tight enough to see this.
    //
    // Ear clipping fans, and on a cylinder every point of a rim shares that
    // rim's `v`, so a chord from one rim vertex to another lies along the
    // boundary however long it is. It cannot be split — the new vertex would
    // land on the boundary and the face across the seam has no such point — so
    // it stayed, a flat sheet across a curved wall. The worst spanned 85 degrees
    // of a 90-degree quarter and sagged 0.68 off a radius of 3.
    //
    // What it cost was half a percent of volume that no tolerance moved: the
    // error *grew* as the tolerance fell, 0.23 to 0.31, because a finer boundary
    // makes a longer fan. This asserts the thing that failed — that asking for
    // less error gets less error.
    let box_ = Body::cuboid([10.0, 8.0, 4.0]);
    let round = Body::cylinder([0.0, -6.0, 0.0], [0.0, 1.0, 0.0], 3.0, 12.0)
        .translated([5.0, 0.0, 2.0])
        .unwrap();
    let quarter = PI * 9.0 / 4.0 * 8.0;

    let mut last = f64::MAX;
    for tolerance in [1e-3, 1e-4] {
        let cut = box_
            .boolean(&round, BooleanOp::Intersection, tolerance)
            .expect("closed form");
        let err = (volume(&cut, tolerance) - quarter).abs();
        assert!(
            err < quarter * 1e-3,
            "at {tolerance} the volume is out by {err}, which is more than a tenth of a percent"
        );
        assert!(
            err < last,
            "{tolerance} is no better than the tolerance before it"
        );
        last = err;
    }
}

#[test]
fn a_cylinder_parked_on_an_edge() {
    // Rounding an edge: the cylinder sits *on* the box's edge, so two of the
    // box's faces pass through its axis and cut it in straight lines.
    //
    // Two things had to hold. A straight seam arrives as a curve of exactly two
    // points, and the seam-origin search used to skip anything shorter than
    // three — so a face cut only by straight seams looked like a face wanting no
    // seam at all. And the four curves close a cycle *inside* the cylinder's
    // parameter rectangle, which the arrangement was handing to itself as a
    // hole.
    let box_ = Body::cuboid([10.0, 8.0, 4.0]);
    let round = Body::cylinder([0.0, -6.0, 0.0], [0.0, 1.0, 0.0], 3.0, 12.0)
        .translated([5.0, 0.0, 2.0])
        .unwrap();
    // The axis lies along the edge, so exactly a quarter of the cylinder is in
    // the box, over the box's 8 of length.
    let quarter = PI * 9.0 / 4.0 * 8.0;
    let whole = PI * 9.0 * 12.0;
    check(&box_, &round, BooleanOp::Difference, 320.0 - quarter);
    check(&box_, &round, BooleanOp::Intersection, quarter);
    check(&box_, &round, BooleanOp::Union, 320.0 + whole - quarter);
}

#[test]
fn a_square_column_bored_through_a_ball() {
    // The case that forces the seam *onto* a curve.
    //
    // The column meets the sphere in two closed loops of four arcs each, and
    // neither loop leaves a gap in `u` — so wherever the sphere's parameter
    // rectangle is cut open, the cut lands in the middle of an arc. Two things
    // have to hold for that to close: the chord has to be split where it
    // crosses, and the seam has to be in the *same* place when the vertex is
    // inserted as when the face is cut.
    //
    // 23.0865 is the column's share, ∫∫ 2√(9−x²−y²) over [−1,1]², to four
    // places; there is no closed form for it.
    let ball = Body::sphere([0.0; 3], 3.0);
    let column = Body::cuboid([2.0, 2.0, 20.0]);
    let shared = 23.0865;
    let sphere = 4.0 / 3.0 * PI * 27.0;
    check(&ball, &column, BooleanOp::Difference, sphere - shared);
    check(&ball, &column, BooleanOp::Intersection, shared);
    check(&ball, &column, BooleanOp::Union, sphere + 80.0 - shared);
}

#[test]
fn a_curved_rim_survives_refinement() {
    // A tube: an outer cylinder bored coaxially, so both the rim and the bore
    // are circles.
    //
    // This is the case that catches a face keeping a *fixed* boundary polyline
    // next to an edge that refines to tolerance. The two describe the same
    // circle and come apart the moment it is subdivided — 16 points against
    // 145 — while every straight-edged case passes, because refining a straight
    // edge changes nothing.
    let outer = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 3.0, 5.0);
    let bore = Body::cylinder([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0, 7.0);
    let overlap = PI * 5.0;
    check(
        &outer,
        &bore,
        BooleanOp::Difference,
        PI * 9.0 * 5.0 - overlap,
    );
    check(
        &outer,
        &bore,
        BooleanOp::Union,
        PI * 9.0 * 5.0 + PI * 7.0 - overlap,
    );
    check(&outer, &bore, BooleanOp::Intersection, overlap);

    // And at a coarser tolerance, where the rim is subdivided differently.
    let r = outer.boolean(&bore, BooleanOp::Difference, 1e-2).unwrap();
    let v = volume(&r, 1e-2);
    assert!((v - (PI * 45.0 - overlap)).abs() / v < 0.02, "{v}");
}

#[test]
fn a_cone_bored_through() {
    // A conical face split by a cylindrical one: the seam is a circle on both,
    // but at different parameter rates.
    let cone = Body::cone([0.0; 3], [0.0, 0.0, 1.0], 3.0, 5.0);
    let drill = Body::cylinder([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0, 8.0);
    let cone_volume = PI / 3.0 * 9.0 * 5.0;
    // The drill's radius reaches the cone's at z = 10/3, so the overlap is a
    // cylinder below that and the cone's own tip above it.
    let overlap = PI * (10.0 / 3.0) + PI / 3.0 * (5.0 - 10.0 / 3.0);
    check(&cone, &drill, BooleanOp::Intersection, overlap);
    check(&cone, &drill, BooleanOp::Difference, cone_volume - overlap);
    check(
        &cone,
        &drill,
        BooleanOp::Union,
        cone_volume + PI * 8.0 - overlap,
    );
}

#[test]
fn bores_can_be_drilled_one_after_another() {
    // The test that says a result is a *model* and not just an answer. Each
    // bore is cut into the output of the last, so anything the boolean gets
    // subtly wrong compounds instead of cancelling.
    //
    // This is also what caught the hole-bridging defect: with one hole a face
    // triangulates by bridging to its outer boundary, which is always
    // reachable. With two, the second bridge may have to run to the first hole,
    // and whether that segment is clear is a question one hole never asks.
    let mut part = Body::cuboid([40.0, 24.0, 4.0]);
    let mut expected = 40.0 * 24.0 * 4.0;
    for x in [-14.0f64, 0.0, 14.0] {
        let drill = Body::cylinder([x, 0.0, -4.0], [0.0, 0.0, 1.0], 2.5, 12.0);
        part = part
            .boolean(&drill, BooleanOp::Difference, 1e-3)
            .unwrap_or_else(|e| panic!("bore at x={x}: {e:?}"));
        expected -= PI * 2.5 * 2.5 * 4.0;
        let got = volume(&part, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.01,
            "after the bore at x={x}: volume {got}, expected {expected}"
        );
    }
    assert_eq!(part.faces().len(), 9, "four walls, two faces, three bores");
    assert_eq!(
        part.surfaces()
            .iter()
            .filter(|s| s.kind() == "cylinder")
            .count(),
        3,
        "every bore is still a cylinder, not a band of triangles"
    );
}

#[test]
fn a_torus_bored_coaxially() {
    // Two surfaces of revolution about the *same* axis meet in circles, which
    // makes a case with no closed form in general position exact here. This is
    // the shape of an O-ring groove, and of a bore through a doughnut.
    //
    // Checked against Pappus: a solid of revolution's volume is its profile
    // area times the distance its centroid travels. Half a tube of radius 1
    // has area π/2 and its centroid sits 4r/3π from the tube's centre, so the
    // outer half is π²·(4 + 4/3π) and the inner half π²·(4 − 4/3π).
    let torus = Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
    let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0);
    let shift = 4.0 / (3.0 * PI);
    let outer = PI * PI * (4.0 + shift);
    let inner = PI * PI * (4.0 - shift);

    check(&torus, &drill, BooleanOp::Difference, outer);
    check(
        &torus,
        &drill,
        BooleanOp::Union,
        2.0 * PI * PI * 4.0 + PI * 16.0 * 6.0 - inner,
    );
}

#[test]
fn a_wrapping_parameter_is_cut_into_a_ring_of_bands() {
    // A torus wraps in *both* parameters, so cuts across its tube divide a
    // circle, not an interval: the stretch from the highest cut back round to
    // the lowest is one band, not two with a seam down the middle.
    //
    // Splitting it as an interval leaves those two halves with nothing shared
    // between them, which showed up only on the intersection — the operation
    // that keeps the band the seam runs through.
    let torus = Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
    let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0);
    let inner = PI * PI * (4.0 - 4.0 / (3.0 * PI));

    let meet = torus
        .boolean(&drill, BooleanOp::Intersection, 1e-3)
        .expect("coaxial, so there is a closed form");
    assert_eq!(meet.faces().len(), 2, "one tube band and one cylinder band");
    check(&torus, &drill, BooleanOp::Intersection, inner);

    // And with a sphere, where the cuts sit elsewhere on the tube.
    let ball = Body::sphere([0.0; 3], 4.0);
    let a = volume(
        &torus.boolean(&ball, BooleanOp::Difference, 1e-3).unwrap(),
        1e-3,
    );
    let b = volume(
        &torus.boolean(&ball, BooleanOp::Intersection, 1e-3).unwrap(),
        1e-3,
    );
    let whole = 2.0 * PI * PI * 4.0;
    assert!(
        (a + b - whole).abs() / whole < 0.01,
        "the pieces should add back to the torus: {a} + {b} vs {whole}"
    );
}

#[test]
fn a_coaxial_pair_that_misses_has_an_empty_intersection() {
    // The drill passes clean through the doughnut's hole. Saying so — an empty
    // solid — is a *result*, and knowing it takes the same closed form.
    let torus = Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
    let clear = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 1.0, 6.0);
    let meet = torus
        .boolean(&clear, BooleanOp::Intersection, 1e-3)
        .expect("coaxial surfaces of revolution have a closed form");
    assert!(meet.faces().is_empty(), "{} faces", meet.faces().len());

    // And the difference leaves the doughnut untouched.
    let cut = torus.boolean(&clear, BooleanOp::Difference, 1e-3).unwrap();
    check(&torus, &clear, BooleanOp::Difference, 2.0 * PI * PI * 4.0);
    assert_eq!(cut.faces().len(), 1);
}

#[test]
fn two_boxes_overlapping_at_a_corner() {
    // The case the subdivision exists for, and the commonest one there is.
    //
    // Every face between the two solids is crossed by *open* curves — cuts that
    // enter and leave through its boundary rather than closing on it — and on
    // the three faces nearest the shared corner two of those cuts meet at the
    // corner itself. Neither reaches the boundary at both ends; together they
    // cross the face. No containment test describes that.
    let a = Body::cuboid([4.0, 4.0, 4.0]);
    let b = Body::cuboid([4.0, 4.0, 4.0])
        .translated([2.0, 2.0, 2.0])
        .unwrap();
    // The overlap is a 2 × 2 × 2 corner.
    check(&a, &b, BooleanOp::Intersection, 8.0);
    check(&a, &b, BooleanOp::Difference, 64.0 - 8.0);
    check(&a, &b, BooleanOp::Union, 64.0 + 64.0 - 8.0);
}

#[test]
fn a_corner_overlap_keeps_every_face_analytic() {
    // Nine faces out of six: the three the other box cuts each come back in
    // two pieces, and all of them are still planes.
    let a = Body::cuboid([4.0, 4.0, 4.0]);
    let b = Body::cuboid([4.0, 4.0, 4.0])
        .translated([2.0, 2.0, 2.0])
        .unwrap();
    let cut = a.boolean(&b, BooleanOp::Difference, 1e-3).unwrap();
    assert_eq!(cut.faces().len(), 9);
    assert!(cut.surfaces().iter().all(|s| s.kind() == "plane"));
}

#[test]
fn a_quartic_seam_is_traced_rather_than_declined() {
    // Two cylinders crossing at *different* radii meet in a quartic — there is
    // no conic to name it with. Tracing walks the curve along `n₁ × n₂`, driving
    // every point back onto both surfaces, so the seam is known to the tolerance
    // it claims rather than fitted to something plausible.
    use threers::brep::{march, Curve3d, Surface};
    let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
    let b = Surface::cylinder([0.0; 3], [1.0, 0.0, 0.0], 1.0);
    let curves = march(&a, &b, ([-6.0; 3], [6.0; 3]), 1e-3);

    // A thin bore right through a thick rod leaves the wall in two places.
    assert_eq!(curves.len(), 2);
    for c in &curves {
        let Curve3d::Sampled { points, closed } = c else {
            panic!("a quartic has no closed form");
        };
        assert!(closed, "each loop closes on itself");
        assert!(points.len() > 16, "sampled to hold the tolerance");
        for p in points {
            assert!(a.distance(*p) <= 1e-6, "{} off the first", a.distance(*p));
            assert!(b.distance(*p) <= 1e-6, "{} off the second", b.distance(*p));
        }
    }
}

#[test]
fn a_trimmed_curved_face_follows_its_surface() {
    // Ear clipping fills a boundary; on a curved surface that leaves the
    // interior spanned by flat sheets. A bore through a sphere gives a face
    // trimmed to an outline no parameter rectangle covers, and its volume is
    // the check that the interior followed the sphere rather than cutting
    // across it.
    let ball = Body::sphere([0.0; 3], 3.0);
    let drill = Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0);
    let h = (9.0f64 - 1.0).sqrt();
    check(
        &ball,
        &drill,
        BooleanOp::Difference,
        4.0 / 3.0 * PI * h.powi(3),
    );
}

#[test]
fn two_solids_sharing_a_wall() {
    // A wall the two share is on the boundary of *both*, so ray parity has
    // nothing to say about it: it is neither inside nor outside. A rule decides
    // instead, and which way the two faces point is what it turns on.
    //
    // Facing each other, as here — a block sitting on a plate — the wall is
    // *between* them: interior to a union, of no thickness in an intersection,
    // and untouched by a difference, since nothing was taken from that side.
    let plate = Body::cuboid([4.0, 4.0, 4.0]);
    let block = Body::cuboid([2.0, 2.0, 2.0])
        .translated([0.0, 0.0, 3.0])
        .unwrap();
    check(&plate, &block, BooleanOp::Difference, 64.0);
    check(&plate, &block, BooleanOp::Union, 64.0 + 8.0);
    let meet = plate
        .boolean(&block, BooleanOp::Intersection, 1e-3)
        .expect("a shared wall is resolvable");
    assert!(meet.faces().is_empty(), "touching solids do not overlap");
}

#[test]
fn a_shared_wall_that_only_partly_overlaps() {
    // Neither face contains the other, so the overlap has to be *computed* —
    // the other face's outline clipped to this one, which is a polygon boolean
    // rather than a cut. This is the case that made the rule alone useless.
    let a = Body::cuboid([4.0, 4.0, 4.0]);
    let b = Body::cuboid([4.0, 4.0, 4.0])
        .translated([2.0, 2.0, 0.0])
        .unwrap();
    // They share the whole of their top and bottom planes over a 2 × 2 patch,
    // and overlap in a 2 × 2 × 4 column.
    check(&a, &b, BooleanOp::Intersection, 16.0);
    check(&a, &b, BooleanOp::Difference, 64.0 - 16.0);
    check(&a, &b, BooleanOp::Union, 64.0 + 64.0 - 16.0);
}

#[test]
fn a_curve_along_a_face_boundary_does_not_cut_it() {
    // Two solids sharing a wall also meet *edge-on* along every adjoining face,
    // and there the intersection runs exactly down the boundary of both. Winding
    // number decides a point on a boundary by whichever way the arithmetic
    // falls, so such a curve reads as a scatter of in and out — and every
    // consumer of that answer gets confused.
    //
    // It does not cut the face: the boundary already describes it. Requiring
    // points to be strictly inside is what makes the case above resolvable at
    // all.
    let a = Body::cuboid([4.0, 4.0, 4.0]);
    let b = Body::cuboid([4.0, 4.0, 4.0])
        .translated([4.0, 0.0, 0.0])
        .unwrap();
    // Face to face, touching on the whole wall and edge-on all round it.
    check(&a, &b, BooleanOp::Union, 128.0);
    check(&a, &b, BooleanOp::Difference, 64.0);
}

#[test]
fn curves_meeting_at_a_corner_share_one_vertex() {
    // Three surfaces through a point give three curves that meet there, and each
    // arrives with its own endpoint a few nanometres from its neighbours'.
    //
    // Welding by position downstream hides that for the faces which rely on it,
    // and cannot help the ones that need to name a point by the curve it came
    // from. The curves are welded to each other first instead, so such a corner
    // is one vertex and naming it is safe.
    let a = Body::cuboid([4.0; 3]);
    let b = Body::cuboid([4.0; 3]).translated([2.0, 2.0, 2.0]).unwrap();
    check(&a, &b, BooleanOp::Intersection, 8.0);
    check(&a, &b, BooleanOp::Difference, 64.0 - 8.0);
    check(&a, &b, BooleanOp::Union, 64.0 + 64.0 - 8.0);

    // Nine faces, every one still a plane: the corner did not fracture into
    // near-duplicate vertices that then had to be papered over.
    let cut = a.boolean(&b, BooleanOp::Difference, 1e-3).unwrap();
    assert_eq!(cut.faces().len(), 9);
    assert!(cut.surfaces().iter().all(|s| s.kind() == "plane"));
}

#[test]
fn a_bore_across_a_rod() {
    // Two cylinders crossing at different radii meet in a quartic — no conic
    // names it, so the seam is traced. The wall's own boundary is then a rim,
    // which is a *straight line* in parameter space: every ear along it is
    // exactly flat, and clipping those without emitting deleted eleven of the
    // wall's sixteen rim vertices while the cap beyond still held edges to them.
    let rod = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0);
    let bore = Body::cylinder([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 10.0);

    // Checked by the identity |A| + |B| = |A∪B| + |A∩B| rather than against a
    // closed form: the overlap of two perpendicular cylinders of unequal radius
    // is an elliptic integral, and a hand-computed expectation would be the less
    // trustworthy half of the comparison.
    let (rod_volume, bore_volume) = (PI * 4.0 * 6.0, PI * 10.0);
    let cut = rod
        .boolean(&bore, BooleanOp::Difference, 1e-3)
        .expect("a traced seam is still a seam");
    let joined = rod.boolean(&bore, BooleanOp::Union, 1e-3).unwrap();

    let overlap = rod_volume - volume(&cut, 1e-3);
    assert!(overlap > 0.0, "the bore passes through the rod");
    let expected = rod_volume + bore_volume - overlap;
    let got = volume(&joined, 1e-3);
    assert!(
        (got - expected).abs() / expected < 0.01,
        "union {got}, expected {expected}"
    );

    // And the intersection, which is the pair of windows on the rod's wall plus
    // the length of bore between them — an *island* on a curved surface, which
    // has to keep its trim ring: the parameter-rectangle fill that would
    // otherwise take it wants rims at constant `v`, and a traced seam is not one.
    let meet = rod
        .boolean(&bore, BooleanOp::Intersection, 1e-3)
        .expect("the windows and the bore between them");
    let both = volume(&meet, 1e-3);
    assert!(
        ((rod_volume + bore_volume) - (got + both)).abs() / (rod_volume + bore_volume) < 0.01,
        "|A|+|B| = {} but |A∪B|+|A∩B| = {}",
        rod_volume + bore_volume,
        got + both
    );

    // Four faces: the rod's wall with two windows in it, its two ends, and the
    // bore's wall. All still analytic.
    assert_eq!(cut.faces().len(), 4);
    assert_eq!(
        cut.surfaces()
            .iter()
            .filter(|s| s.kind() == "cylinder")
            .count(),
        2
    );
}

#[test]
fn a_sphere_clipping_a_corner() {
    // Three walls cut at once, and each cut is an *arc*: the plane of a wall
    // meets the sphere in a whole circle, but only the part crossing that wall
    // is a seam. A circle crossing a wall can also be on it in two separate
    // stretches, so a face's clip is a set of intervals rather than one.
    let block = Body::cuboid([10.0, 8.0, 4.0]);
    let ball = Body::sphere([5.0, 4.0, 2.0], 3.0);

    // The sphere is centred on the corner, so an eighth of it is inside.
    let octant = 4.0 / 3.0 * PI * 27.0 / 8.0;
    check(&block, &ball, BooleanOp::Intersection, octant);
    check(&block, &ball, BooleanOp::Difference, 320.0 - octant);
    check(
        &block,
        &ball,
        BooleanOp::Union,
        320.0 + 4.0 / 3.0 * PI * 27.0 - octant,
    );

    let cut = block.boolean(&ball, BooleanOp::Difference, 1e-3).unwrap();
    assert_eq!(
        cut.faces().len(),
        7,
        "six walls, three of them cut, and the dimple"
    );
    assert_eq!(
        cut.surfaces()
            .iter()
            .filter(|s| s.kind() == "sphere")
            .count(),
        1,
        "the dimple is still a sphere"
    );
}

#[test]
fn the_three_operations_stay_consistent_with_each_other() {
    // |A| + |B| = |A ∪ B| + |A ∩ B|, which holds whatever the shapes are and
    // needs no hand-computed expectation to check.
    let (a, b) = (plate(), through_drill());
    let va = volume(&a, 1e-3);
    let vb = volume(&b, 1e-3);
    let u = volume(&a.boolean(&b, BooleanOp::Union, 1e-3).unwrap(), 1e-3);
    let i = volume(&a.boolean(&b, BooleanOp::Intersection, 1e-3).unwrap(), 1e-3);
    let d = volume(&a.boolean(&b, BooleanOp::Difference, 1e-3).unwrap(), 1e-3);
    assert!(((va + vb) - (u + i)).abs() / (va + vb) < 0.01);
    assert!((d - (va - i)).abs() / va < 0.01);
}

#[test]
fn a_result_is_always_watertight_or_declined_never_neither() {
    // The property the whole layer rests on. Every pair below either resolves to
    // a closed solid or says why not — and `volume` asserts the closure, so a
    // result that is not watertight fails here rather than reaching a caller.
    let cases: Vec<(&str, Body, Body)> = vec![
        (
            "box/box overlapping",
            Body::cuboid([4.0; 3]),
            Body::cuboid([4.0; 3]).translated([2.0, 2.0, 2.0]).unwrap(),
        ),
        // A cylinder parked on an edge, to round it: two of the box's faces
        // pass through its axis and cut it in straight lines.
        (
            "round an edge",
            Body::cuboid([10.0, 8.0, 4.0]),
            Body::cylinder([0.0, -6.0, 0.0], [0.0, 1.0, 0.0], 3.0, 12.0)
                .translated([5.0, 0.0, 2.0])
                .unwrap(),
        ),
        (
            "box/box flush",
            Body::cuboid([4.0; 3]),
            Body::cuboid([2.0; 3]).translated([0.0, 0.0, 3.0]).unwrap(),
        ),
        ("box/through drill", plate(), through_drill()),
        (
            "box/blind drill",
            Body::cuboid([10.0, 8.0, 4.0]),
            Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 6.0),
        ),
        (
            "sphere/drill",
            Body::sphere([0.0; 3], 3.0),
            Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0),
        ),
        (
            "sphere/box",
            Body::sphere([0.0; 3], 3.0),
            Body::cuboid([2.0, 2.0, 20.0]),
        ),
        (
            "coaxial cylinders",
            Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 3.0, 5.0),
            Body::cylinder([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0, 7.0),
        ),
        (
            "crossing cylinders",
            through_drill(),
            Body::cylinder([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 10.0),
        ),
        (
            "cone/cylinder",
            Body::cone([0.0; 3], [0.0, 0.0, 1.0], 3.0, 5.0),
            Body::cylinder([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0, 8.0),
        ),
        // Coaxial, and passing through the tube rather than clear of it.
        (
            "torus/cylinder",
            Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0),
            Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0),
        ),
        // Off-axis, so the cylinder grazes the torus's inner equator at a
        // single point: the intersection pinches there and declines as a
        // tangential contact rather than as a fault.
        (
            "torus/cylinder off-axis",
            Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0),
            Body::cylinder([1.0, 0.0, -3.0], [0.0, 0.0, 1.0], 4.0, 6.0),
        ),
        (
            "box/sphere",
            Body::cuboid([6.0; 3]),
            Body::sphere([0.0, 0.0, 3.0], 2.0),
        ),
        // A sphere clipping a box's corner: three walls cut at once, each by an
        // arc rather than a whole circle.
        (
            "sphere at a corner",
            Body::cuboid([10.0, 8.0, 4.0]),
            Body::sphere([5.0, 4.0, 2.0], 3.0),
        ),
    ];

    let mut resolved = 0;
    for (name, a, b) in &cases {
        for op in [
            BooleanOp::Difference,
            BooleanOp::Union,
            BooleanOp::Intersection,
        ] {
            match a.boolean(b, op, 1e-3) {
                Ok(r) => {
                    // Panics unless it is closed and every face was filled. An
                    // *empty* result is legitimate — two disjoint solids have an
                    // empty intersection — and is not counted as coverage.
                    let v = volume(&r, 1e-3);
                    if r.faces().is_empty() {
                        assert_eq!(v, 0.0, "{name}: no faces but a volume");
                    } else {
                        assert!(v > 0.0, "{name}: a closed result with no volume");
                        // And a *solid*, not merely a closed mesh: one shell,
                        // every edge on exactly two faces. That is what decides
                        // whether it can be the input to the next boolean, and
                        // for most of this layer's life it was not true.
                        assert!(
                            r.defects(1e-3).is_empty(),
                            "{name} {op:?}: {:?}",
                            r.defects(1e-3)
                        );
                        assert!(
                            r.is_valid_solid(1e-3),
                            "{name} {op:?} is closed but not a solid"
                        );
                        resolved += 1;
                    }
                }
                Err(
                    Declined::NoClosedForm { .. }
                    | Declined::NeedsArrangement { .. }
                    | Declined::UnclassifiablePiece { .. }
                    | Declined::TangentialContact { .. }
                    | Declined::NotWatertight { .. },
                ) => {}
                Err(Declined::NotASolid) => panic!("{name}: both inputs are solids"),
            }
        }
    }
    // Not a target — a record. If this drops, coverage regressed; if it rises,
    // the new cases want volumes asserted above rather than just counting.
    //
    // The rest are the map of what is not implemented: `NeedsArrangement` where
    // an intersection curve crosses a face boundary (box/box), and
    // `NoClosedForm` where a surface pair genuinely has none.
    //
    // `CoincidentFaces` used to be on that list and is gone: two bodies sharing
    // a surface are *handled* now — `SsiResult::Coincident` sends the pair to
    // the keep/drop rules rather than refusing it — so the variant could not be
    // returned by anything and named a refusal the kernel no longer makes.
    assert_eq!(resolved, 38, "of {} operations", cases.len() * 3);
}

#[test]
fn a_rim_cut_a_second_time_is_shared_by_two_faces_at_a_time() {
    // Cut a bore into a ball, then a square column down the same axis. The
    // column is wider than the bore across its diagonal and narrower across its
    // flats, so it takes the bore's wall apart into five pieces — and each piece
    // used to claim the *whole* rim circle where the wall meets the sphere. Six
    // faces on one edge: a result that tessellates closed and is not a solid,
    // because an edge belongs to two faces or the topology is not a surface.
    //
    // The rim is cut instead. Every vertex of it is labelled by the pair of
    // faces that reach it, the corners the column took away carry the label
    // before them, and each stretch of one label becomes an edge those two
    // faces share.
    let ball = Body::sphere([0.0; 3], 3.0);
    let bore = Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0);
    let bored = ball
        .boolean(&bore, BooleanOp::Difference, 1e-3)
        .expect("a sphere and a cylinder have a closed form");
    let twice = bored
        .boolean(&Body::cuboid([1.5, 1.5, 8.0]), BooleanOp::Difference, 1e-3)
        .expect("and so do a sphere and a box");

    let bad: Vec<_> = twice
        .defects(1e-3)
        .into_iter()
        .filter(|d| matches!(d, Defect::EdgeFaceCount { .. }))
        .collect();
    assert!(
        bad.is_empty(),
        "an edge is not a boundary between two faces: {bad:?}"
    );

    assert!(twice.defects(1e-3).is_empty(), "{:?}", twice.defects(1e-3));

    // And the curve where the column's plane cuts the bore's wall is an edge.
    //
    // It was not, and for a reason with nothing to do with this pair: the
    // arrangement records which curve a boundary came from on that curve's
    // *interior* points, and a plane meets a cylinder in a straight line, which
    // two samples describe exactly. With no interior point there was no
    // provenance, so no edge was made — every piece of the wall came back
    // bounded by rim arcs alone, with four free ends, and the shell could not
    // close. A two-point curve is given a midpoint now.
    let wall_cuts = twice
        .edges()
        .iter()
        .filter(|e| {
            let (a, b) = (
                twice.surfaces()[e.surfaces.0].kind(),
                twice.surfaces()[e.surfaces.1].kind(),
            );
            (a, b) == ("plane", "cylinder") || (a, b) == ("cylinder", "plane")
        })
        .count();
    assert_eq!(
        wall_cuts, 8,
        "the four column planes cut the bore's wall in two lines each"
    );

    // And no edge runs from a vertex to itself. A rim that closes repeats its
    // first vertex last, and a stretch cut across that repeat named the same
    // vertex twice — an edge of no length, which the stretches either side then
    // had nothing to meet at.
    let degenerate = twice
        .edges()
        .iter()
        .filter(|e| {
            e.vertices
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                < 2
        })
        .count();
    assert_eq!(
        degenerate, 0,
        "an edge from a vertex to itself is not an edge"
    );

    // And it is a solid: one closed shell, every edge between exactly two
    // faces, so it can be the input to a third cut.
    //
    // Four separate faults stood between here and that, each found by measuring
    // rather than guessing and none of them the last: a rim six faces claimed,
    // a whole class of edge never made because a straight curve leaves no trace
    // where provenance lives, an edge from a vertex to itself across a closed
    // rim's repeated end, and a stretch that changed hands at a junction the
    // labelling carried straight through.
    assert!(
        twice.is_valid_solid(1e-3),
        "defects {:?}, shells closed {}",
        twice.defects(1e-3),
        twice.shells().iter().all(|s| s.closed)
    );
}

#[test]
fn a_curve_is_clipped_by_where_it_runs_not_by_where_its_chords_sag() {
    // A traced curve is a polyline, and asking for a point between two of its
    // samples gives a point on the *chord* — which on a curved surface is not on
    // the surface. Clipping asked exactly that question, with the surface's own
    // tolerance for an answer, so it read "on the face" near every sample and
    // "off it" in between: sixty-five intervals along a sixty-five point curve,
    // crossed with the other face's twenty-seven, giving sixty-four three-point
    // fragments where there was one curve. The arrangement could not be built
    // from that and the whole operation declined `NeedsArrangement`.
    //
    // The curve does not leave the surface between samples. Only the straight
    // line drawn through it does.
    let big = Body::sphere([0.0; 3], 3.0);
    let small = Body::sphere([2.0, 0.0, 0.0], 2.0);
    let cut = big
        .boolean(&small, BooleanOp::Difference, 1e-3)
        .expect("two spheres have a closed form");

    // The bore misses the first sphere's axis by nothing and the second's by
    // two, so one pair is coaxial and exact and the other is traced.
    let bore = Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 0.8, 12.0);
    if let Err(why) = cut.boolean(&bore, BooleanOp::Difference, 1e-3) {
        assert!(
            !matches!(why, Declined::NeedsArrangement { .. }),
            "the arrangement is being handed fragments again: {why:?}"
        );
    }
}

#[test]
fn a_crossing_sits_on_both_of_the_surfaces_whose_rim_it_cuts() {
    // A rim is the curve two surfaces share, and a crossing spliced into one is
    // computed on the surface being cut and left there — so it lands *off* the
    // rim. Measured at 2.6e-3 on a 1e-3 model, with the two faces that own the
    // vertex disagreeing by 2.8e-3 about where it is.
    //
    // Nothing downstream saw it, because a fill takes a ring vertex's position
    // from the body vertex rather than from its own `uv`. It is a crack held
    // shut by the fill: every change to how a curve is sampled has failed here
    // by prying it open, each face minting its own copy of a point that is on
    // neither curve.
    //
    // The test is not "the boolean succeeded" — these cases already did. It is
    // that no ring point sits *near* a surface it is not on. Close to another
    // surface and not on it is the signature of a point that was meant to be
    // shared and is not; either it is on that surface, or it has no business
    // being within a thousandth of it.
    for (name, tool) in [
        ("ball", Body::sphere([0.0; 3], 3.0)),
        (
            "rod",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
        ),
    ] {
        let cross = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0);
        let cut = tool
            .boolean(&cross, BooleanOp::Difference, 1e-3)
            .expect("a bore across a solid resolves");
        let surfaces = cut.surfaces();
        for (fi, face) in cut.faces().iter().enumerate() {
            let Some(loops) = face.loops.as_ref() else {
                continue;
            };
            let own = &surfaces[face.surface];
            for (li, ring) in loops.iter().enumerate() {
                for (k, uv) in ring.uv.iter().enumerate() {
                    let p = own.point(uv[0], uv[1]);
                    let near = surfaces
                        .iter()
                        .enumerate()
                        .filter(|(si, _)| *si != face.surface)
                        .map(|(_, other)| other.distance(p).abs())
                        .fold(f64::MAX, f64::min);
                    assert!(
                        !(1e-6..1e-2).contains(&near),
                        "{name} f{fi} l{li} #{k} (vertex {}) is {near:.2e} from a \
                         surface it is not on",
                        ring.vertices[k]
                    );
                }
            }
        }
    }
}

#[test]
fn a_rod_can_be_added_to_a_ball_that_has_already_been_bored() {
    // The union that a single off-rim crossing used to cost: one open edge, in a
    // result whose topology was otherwise sound.
    let ball = Body::sphere([0.0; 3], 3.0);
    let bore = Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0);
    let rod = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0);

    let bored = ball
        .boolean(&bore, BooleanOp::Difference, 1e-3)
        .expect("a bore down the axis is closed form");
    let with_rod = bored
        .boolean(&rod, BooleanOp::Union, 1e-3)
        .expect("adding a rod through a bored ball");

    assert!(
        with_rod.is_valid_solid(1e-3),
        "defects {:?}",
        with_rod.defects(1e-3)
    );
    // The rod is longer than the ball is wide, so the union is strictly larger.
    assert!(volume(&with_rod, 1e-3) > volume(&bored, 1e-3));
}

#[test]
fn two_solids_with_no_vertices_are_not_assumed_to_miss_each_other() {
    // A whole torus is one face with no edges and no vertices, and so is a whole
    // sphere. The trace box was read off the vertices, so a body with none
    // contributed nothing to it — and two of them handed the marcher a box whose
    // low corner was above its high one. It found no intersection curve, the
    // boolean concluded the two do not meet, and `torus - ball` came back as the
    // torus: untouched, watertight, `is_valid_solid`, and wrong. That is the one
    // answer this crate is not allowed to give, and nothing downstream could
    // have told.
    let torus = Body::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
    let ball = Body::sphere([4.0, 0.0, 0.0], 1.5);
    assert!(
        torus.vertices().is_empty() && ball.vertices().is_empty(),
        "the case only bites when neither body has a vertex to be found by"
    );

    // The ball is centred on the tube's own centre circle and is half again as
    // wide as the tube, so it swallows a stretch of it whole. Any honest
    // difference is smaller than the torus; a decline is fine, a full torus is
    // not.
    let whole = volume(&torus, 1e-3);
    if let Ok(cut) = torus.boolean(&ball, BooleanOp::Difference, 1e-3) {
        let left = volume(&cut, 1e-3);
        assert!(
            left < whole - 1.0,
            "the ball takes a bite out of the tube, but the difference kept \
             {left:.3} of {whole:.3}"
        );
    }
}

#[test]
fn the_same_cut_asked_for_five_ways_is_either_right_or_declined() {
    // Tolerance is the caller's dial, and nothing says a solid that resolves at
    // one setting resolves at every finer one — the rings are sampled from it,
    // so the whole downstream shape of the problem changes with it. What must
    // hold at every setting is the rule this file is about: a result, or a
    // reason. Never a solid that is merely plausible.
    //
    // Three of the five resolve, and five did before the trim loops stopped
    // being pinned to the sampling they arrived with. That pin was holding a
    // *wrong answer* in place: `rod - column` came back as a sixteen-gon prism,
    // 82.2870 where the intersection says 84.8106, because a cap kept the
    // primitive's sixteen-point rim and the wall had to follow it. Unpinning
    // fixes the size and costs the two finest tolerances here.
    //
    // That is the trade this file's header asks for in as many words: a result
    // is never worse than a decline. Two honest declines are worth more than one
    // solid of the wrong size, and the count is written down rather than
    // asserted away, so a change that recovers them is visible.
    //
    // The sizes are asserted too, not only closure — a watertight solid of the
    // wrong size is the failure this file exists to catch.
    let mut sizes: Vec<f64> = Vec::new();
    let mut resolved = 0;
    for tol in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5] {
        let ball = Body::sphere([0.0; 3], 3.0);
        let bore = Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0);
        let cross = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0);

        let Ok(bored) = ball.boolean(&bore, BooleanOp::Difference, tol) else {
            continue;
        };
        if let Ok(twice) = bored.boolean(&cross, BooleanOp::Difference, tol) {
            assert!(
                twice.is_valid_solid(tol),
                "at {tol:e} the second cut returned a solid it should have \
                 declined: {:?}",
                twice.defects(tol)
            );
            sizes.push(volume(&twice, tol));
            resolved += 1;
        }
    }
    assert_eq!(
        resolved, 5,
        "all five tolerances resolved when this was written; {resolved} do now"
    );
    let lo = sizes.iter().cloned().fold(f64::MAX, f64::min);
    let hi = sizes.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        hi - lo < lo * 0.01,
        "the five agree to a per cent or they are not the same solid: {sizes:?}"
    );
}

#[test]
fn a_seam_is_reached_from_both_sides_at_the_same_place() {
    // Both sides of a seam are the same line. Where a ring runs *along* it on
    // both sides — a vertical run in parameter space, two or more points at one
    // `u` — the two runs describe that one line, and they have to end together.
    //
    // Where one stops a point short, the ring leaves the seam diagonally for the
    // next vertex along, and the two legs of that step bound a wedge no triangle
    // covers. On a bored ball at 2e-4 that is three of the four edges the fill
    // leaves open; here, at 1e-3, the ring is wrong the same way and the fill
    // happens to survive it. A crack held shut is still a crack, and this is the
    // one case in the corpus where it can be caught in a result that resolves.
    //
    // The run is what makes the claim safe. A ring may span a whole period
    // without ever lying along the seam, and then its extreme points are single
    // and their `v` has no reason to agree at all — asserting it there invents
    // points on faces that were right, and five other tests say so.
    let ball = Body::sphere([0.0; 3], 3.0);
    let cross = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0);
    let cut = ball
        .boolean(&cross, BooleanOp::Difference, 1e-3)
        .expect("a bore across a ball resolves");

    let surfaces = cut.surfaces();
    let mut checked = 0;
    for (fi, face) in cut.faces().iter().enumerate() {
        if !surfaces[face.surface].periodic().0 {
            continue;
        }
        let Some(loops) = face.loops.as_ref() else {
            continue;
        };
        for (li, ring) in loops.iter().enumerate() {
            let us: Vec<f64> = ring.uv.iter().map(|c| c[0]).collect();
            let lo = us.iter().cloned().fold(f64::MAX, f64::min);
            let hi = us.iter().cloned().fold(f64::MIN, f64::max);
            if (hi - lo - 2.0 * PI).abs() > 1e-6 {
                continue;
            }
            let side = |at: f64| -> Vec<f64> {
                ring.uv
                    .iter()
                    .filter(|c| (c[0] - at).abs() < 1e-9)
                    .map(|c| c[1])
                    .collect()
            };
            let (left, right) = (side(lo), side(hi));
            if left.len() < 2 || right.len() < 2 {
                continue;
            }
            let ends = |v: &[f64]| {
                (
                    v.iter().cloned().fold(f64::MAX, f64::min),
                    v.iter().cloned().fold(f64::MIN, f64::max),
                )
            };
            let (la, lb) = ends(&left);
            let (ra, rb) = ends(&right);
            assert!(
                (la - ra).abs() < 1e-9 && (lb - rb).abs() < 1e-9,
                "f{fi} l{li}: the seam is run from {la:.6}..{lb:.6} on one side \
                 and {ra:.6}..{rb:.6} on the other"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no ring ran along a seam; the case has changed"
    );
}

#[test]
fn every_step_of_a_ring_is_a_step_of_an_edge() {
    // A ring is meant to be this face's edges in an order. Where it is, the ring
    // can be rebuilt when the edges move, refined with them, and kept across a
    // second boolean; where it is not, the ring is a fixed polyline that drifts
    // from the edges beside it, and `split_face` drops it rather than let that
    // happen — which is how a face ends up with no boundary at all.
    //
    // A swept face's seam is the part that used to be backed by nothing. It is
    // not sampled, so it is a single step from one rim to the other, and both
    // its ends are rim vertices that edges *do* name — so the run-finder in
    // `materialise_seams`, which looks for vertices no edge names, never starts.
    // Measured on the rod below: of a 36-step ring, two steps, `v90 -> v107` and
    // `v107 -> v90` — one line, walked up one side of the parameter domain and
    // down the other.
    //
    // Faces that run to a point are exempt, and deliberately: a ring crossing a
    // pole steps from a vertex to itself, and a seam ending at one is
    // pole-to-pole, which this crate's STEP reader cannot place.
    let cases: [(&str, Body, Body); 2] = [
        (
            "a rod bored across",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
            Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0),
        ),
        (
            "a rod bored across, off its axis",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
            Body::cylinder([-6.0, 0.6, 0.0], [1.0, 0.0, 0.0], 0.7, 12.0),
        ),
    ];
    for (name, solid, tool) in cases {
        let cut = solid
            .boolean(&tool, BooleanOp::Difference, 1e-3)
            .expect("a bore through a solid resolves");

        let mut steps: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for edge in cut.edges() {
            for w in edge.vertices.windows(2) {
                steps.insert((w[0], w[1]));
                steps.insert((w[1], w[0]));
            }
        }

        let mut checked = 0;
        for (fi, face) in cut.faces().iter().enumerate() {
            let Some(loops) = face.loops.as_ref() else {
                continue;
            };
            let pole = loops.iter().any(|r| {
                let n = r.vertices.len();
                n >= 3 && (0..n).any(|i| r.vertices[i] == r.vertices[(i + 1) % n])
            });
            if pole {
                continue;
            }
            for (li, ring) in loops.iter().enumerate() {
                let n = ring.vertices.len();
                if n < 3 || ring.uv.len() != n {
                    continue;
                }
                for i in 0..n {
                    let step = (ring.vertices[i], ring.vertices[(i + 1) % n]);
                    assert!(
                        steps.contains(&step),
                        "{name}: f{fi} l{li} step #{i} (v{} to v{}) has no edge \
                         behind it",
                        step.0,
                        step.1
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "{name}: no face stated a ring");
    }
}

#[test]
fn what_a_cut_removes_is_what_the_intersection_holds() {
    // The check this file was missing. Every other test here asks whether a
    // result is *watertight*, and a watertight solid of the wrong size passes
    // all of them — `is_valid_solid` cannot see wrongness. A chaining metric
    // built on it scored a rod bored twice down the same axis, which came back
    // holding 67.03 of the 75.40 it should, as a success.
    //
    // Inclusion-exclusion needs no closed form and no expected volume:
    //
    //     |A| - |A n B| = |A - B|          and    |A| + |B| - |A n B| = |A u B|
    //
    // so any three of the four measure each other. It is the same identity
    // `a_bore_across_a_rod` uses for one pair, over every pair that resolves.
    //
    // Measured at 5e-3 rather than 1e-3. The identity holds at any tolerance —
    // it is about the operations agreeing with each other, not about either
    // being exact — and the same twenty-six pairs measure each other either way:
    //
    //     1e-2   worst 1.40e-2   too coarse for the one per cent asked below
    //     5e-3   worst 3.91e-3   24s
    //     1e-3   worst 2.14e-3   74s
    //
    // Three times faster with two and a half times the margin still to spare. A
    // guard nobody wants to run guards nothing.
    let tol = 5e-3;
    let bases: [(&str, Body); 4] = [
        ("ball", Body::sphere([0.0; 3], 3.0)),
        (
            "rod",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
        ),
        ("cube", Body::cuboid([4.0; 3])),
        ("plate", Body::cuboid([10.0, 8.0, 4.0])),
    ];
    let tools: [(&str, Body); 4] = [
        (
            "bore",
            Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0),
        ),
        (
            "cross",
            Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0),
        ),
        ("column", Body::cuboid([1.4, 1.4, 14.0])),
        ("ball2", Body::sphere([2.0, 0.0, 0.0], 2.0)),
    ];
    let closed = |b: &Body| -> Option<f64> {
        let mut c = b.clone();
        c.refine_edges(tol);
        let (_, r) = c.tessellate(tol);
        (r.is_closed() && r.carried_through == 0).then(|| volume(b, tol))
    };

    let mut checked = 0;
    for (an, a) in &bases {
        for (bn, b) in &tools {
            let (Some(va), Some(vb)) = (closed(a), closed(b)) else {
                continue;
            };
            let got = |op| a.boolean(b, op, tol).ok().and_then(|r| closed(&r));
            let (d, u, i) = (
                got(BooleanOp::Difference),
                got(BooleanOp::Union),
                got(BooleanOp::Intersection),
            );
            if let (Some(d), Some(i)) = (d, i) {
                checked += 1;
                let want = va - i;
                assert!(
                    (want - d).abs() / va.max(1.0) < 0.01,
                    "{an} - {bn}: the intersection says {want:.4} should be left, \
                     the difference gives {d:.4}"
                );
            }
            if let (Some(u), Some(i)) = (u, i) {
                checked += 1;
                let want = va + vb - i;
                assert!(
                    (want - u).abs() / (va + vb).max(1.0) < 0.01,
                    "{an} u {bn}: the parts say {want:.4}, the union gives {u:.4}"
                );
            }
        }
    }
    assert!(checked >= 20, "only {checked} pairs measured each other");
}

#[test]
fn cutting_twice_with_one_tool_takes_nothing_the_second_time() {
    // `(A - B) - B` is `A - B`. Both of the second cuts this corpus can make
    // used to break it, and both by about an eighth of the solid:
    //
    //     ball - bore  again   94.6184 became 82.7862
    //     ball - ball2 again   88.2798 became 77.6824
    //
    // Watertight, `is_valid_solid`, and wrong — which is why nothing else here
    // saw them. A second cut meets a wall of its own making, coincident with the
    // tool's, and has to keep exactly one copy. It kept both, three times over
    // for one reason each:
    //
    //   * `faces_overlap` compared the two faces' first rings, and a whole
    //     cylinder wall's rings are its *rims* — lines with no area;
    //   * `point_on_face` chose its ring the same way, so a face did not contain
    //     its own middle;
    //   * and it tested the point in the canonical period while the ring lived
    //     an unwrapped turn away, so a sphere's cavity wall missed every point
    //     of itself by 2pi.
    //
    // Both now decline instead, which is what this file asks for: a result is
    // never worse than a decline. Declining is what the `continue`s below
    // permit; returning a smaller solid is what they do not.
    //
    // Ignored rather than deleted: it is a specification of what the kernel owes
    // and a one-command reproduction of what it does instead. Run it with
    // `cargo test --features brep-kernel -- --ignored`.
    let tol = 1e-3;
    let cases: [(&str, Body, Body); 3] = [
        (
            "ball bored",
            Body::sphere([0.0; 3], 3.0),
            Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0),
        ),
        (
            "ball cut by a ball",
            Body::sphere([0.0; 3], 3.0),
            Body::sphere([2.0, 0.0, 0.0], 2.0),
        ),
        (
            "rod bored",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
            Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0),
        ),
    ];
    for (name, a, b) in cases {
        let Ok(once) = a.boolean(&b, BooleanOp::Difference, tol) else {
            continue;
        };
        let Ok(twice) = once.boolean(&b, BooleanOp::Difference, tol) else {
            continue; // declining the second cut is allowed; lying about it is not
        };
        let (v1, v2) = (volume(&once, tol), volume(&twice, tol));
        assert!(
            (v1 - v2).abs() / v1.max(1.0) < 0.01,
            "{name}, cut again by the same tool: {v1:.4} became {v2:.4}"
        );
    }
}

#[test]
fn no_two_vertices_of_a_result_are_the_same_point() {
    // Two points closer than the tolerance are the same point — that is what a
    // tolerance means. A result that keeps both is carrying a crack: every later
    // stage then has to decide which one it meant, and this run has watched that
    // go wrong repeatedly, most expensively when one crossing was minted twice
    // 8.9e-4 apart and the two faces either side of it stopped agreeing.
    //
    // Two things were wrong. The weld ran at *half* the tolerance, which left
    // pairs the definition says are one; widening it removed four of the
    // thirty-four a chaining corpus carried.
    //
    // The other thirty were not really duplicates at all. Assembly mints a
    // vertex whenever a piece names a point, and some of those end up on no edge
    // and in no ring — a discarded piece, a superseded crossing. Invisible in
    // the solid, and not free: they read as duplicates of the points that
    // survived, one pair 7.1e-16 apart. A bored ball carried twelve such pairs
    // among a hundred and fifty-seven vertices, and has a hundred and
    // thirty-seven with none once they are dropped.
    let tol = 1e-3;
    let bases: [(&str, Body); 3] = [
        ("ball", Body::sphere([0.0; 3], 3.0)),
        (
            "rod",
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0),
        ),
        ("cube", Body::cuboid([4.0; 3])),
    ];
    let tools: [(&str, Body); 3] = [
        (
            "bore",
            Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0),
        ),
        (
            "cross",
            Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0),
        ),
        ("column", Body::cuboid([1.4, 1.4, 14.0])),
    ];
    for (bn, b) in &bases {
        for (tn, t) in &tools {
            let Ok(cut) = b.boolean(t, BooleanOp::Difference, tol) else {
                continue;
            };
            let v = cut.vertices();
            for i in 0..v.len() {
                for j in i + 1..v.len() {
                    let d = ((v[i][0] - v[j][0]).powi(2)
                        + (v[i][1] - v[j][1]).powi(2)
                        + (v[i][2] - v[j][2]).powi(2))
                    .sqrt();
                    assert!(
                        d >= tol,
                        "{bn} - {tn}: vertices {i} and {j} are {d:.3e} apart, \
                         which the tolerance says is one point"
                    );
                }
            }
        }
    }
}

/// The same shape at four sizes gets the same answer.
///
/// Two equal cylinders crossing at right angles graze: their surfaces are
/// tangent at two points whatever the radius, so the operation cannot be
/// carried out and the honest reply is `TangentialContact` — at every size.
///
/// It was not. The graze was looked for by asking each *sample* of the
/// intersection whether the normals there were near-parallel, and near the
/// graze `|n1 x n2|` runs about `sqrt(2)·x/r`, so the threshold does scale with
/// the model — but whether a sample lands inside the window does not scale with
/// anything. At `r = 1` one did and the graze was reported; at `r = 2` none did
/// and the same shape twice as large came back `NotWatertight` with 82 open
/// edges. A kernel that answers by the size of the part is worse than one that
/// declines, because nothing downstream can tell which reply it got.
#[test]
fn a_graze_is_a_graze_at_any_size() {
    for r in [0.5, 1.0, 2.0, 4.0] {
        let a = Body::cylinder([0.0, 0.0, -4.0 * r], [0.0, 0.0, 1.0], r, 8.0 * r);
        let b = Body::cylinder([-6.0 * r, 0.0, 0.0], [1.0, 0.0, 0.0], r, 12.0 * r);
        for op in [
            BooleanOp::Union,
            BooleanOp::Intersection,
            BooleanOp::Difference,
        ] {
            let got = a.boolean(&b, op, 1e-3 * r);
            assert!(
                matches!(got, Err(Declined::TangentialContact { .. })),
                "two equal cylinders of radius {r} graze, so {op:?} should say so: {:?}",
                got.map(|b| b.faces().len())
            );
        }
    }
}

/// A bore across a rod is right or declined, at every ratio of the two.
///
/// The interesting range is where the tool is a large fraction of the rod, and
/// there the outcome used to alternate — 0.45 refused, 0.50 answered, 0.55
/// refused, 0.65 answered — which is not how geometry behaves and was the clue
/// that something discrete was deciding it. What decided it was a rim: two rims
/// cut a bore's wall into three bands, and where the walk closed a ring after
/// one turn around the cylinder instead of coming back along the other rim, the
/// band that should have been inside the rod was never built.
///
/// One of those came back watertight, in two shells, with seven faces and a
/// volume of 23.53 where 20.46 is right — 15% too much material, and nothing
/// structural to show for it. That is the failure this whole kernel is arranged
/// to prevent, so it is checked by size and not by shape.
#[test]
fn a_bore_across_a_rod_is_right_or_declined_at_every_ratio() {
    // Volume shared by two perpendicular cylinders, `a` about x and `b` about
    // z. For each y within the narrower one, the two give independent extents
    // in z and in x, so the section is a rectangle and the whole is a single
    // integral — no closed form is needed for a number that only has to be
    // right.
    let shared = |a: f64, b: f64| -> f64 {
        let n = 4000;
        let (lo, hi) = (-a, a);
        let h = (hi - lo) / n as f64;
        let f = |y: f64| 4.0 * ((a * a - y * y).max(0.0) * (b * b - y * y).max(0.0)).sqrt();
        let mut sum = f(lo) + f(hi);
        for i in 1..n {
            sum += f(lo + h * i as f64) * if i % 2 == 0 { 2.0 } else { 4.0 };
        }
        sum * h / 3.0
    };

    let mut resolved = 0;
    for i in 0..=10 {
        let ratio = 0.45 + 0.05 * i as f64;
        let rod = Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 1.0, 8.0);
        let tool = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], ratio, 12.0);
        let overlap = shared(ratio, 1.0);
        let whole_rod = PI * 8.0;
        let whole_tool = PI * ratio * ratio * 12.0;

        for (name, cut, expected) in [
            (
                "rod less tool",
                rod.boolean(&tool, BooleanOp::Difference, 1e-3),
                whole_rod - overlap,
            ),
            (
                "tool less rod",
                tool.boolean(&rod, BooleanOp::Difference, 1e-3),
                whole_tool - overlap,
            ),
        ] {
            let Ok(body) = cut else { continue };
            resolved += 1;
            let got = volume(&body, 1e-3);
            assert!(
                (got - expected).abs() / expected < 0.01,
                "{name} at ratio {ratio:.2}: volume {got:.4}, expected {expected:.4}"
            );
        }
    }
    assert!(
        resolved == 22,
        "all twenty-two resolved when this was written; {resolved} do now"
    );
}
