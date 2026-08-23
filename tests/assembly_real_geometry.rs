//! The assembly predicates, on geometry a boolean kernel actually produced.
//!
//! Every unit test in `src/assembly` builds its meshes by hand, which means
//! every one of them is a mesh with no degenerate triangles, no near-duplicate
//! vertices and exactly the connectivity it was written to have. Real CSG output
//! has all three, and the predicates that survive a hand-built cube are not
//! thereby known to survive a `difference()`.

#![cfg(all(feature = "assembly-check", feature = "openscad"))]

use threers::assembly::{aabb, from_geometry, inside, interfere, nearest, rigid_key, shells, Tri};
use threers::parse_scad;

fn tris(source: &str) -> Vec<Tri> {
    let solid = parse_scad(source).expect("the model parses");
    from_geometry(&solid.to_geometry())
}

#[test]
fn a_real_boolean_result_is_one_body() {
    // An open-topped box: the cavity reaches the top face, so the inner and
    // outer surfaces join and the whole thing is one shell.
    let open_box = tris(
        "difference() {
            cube([30, 20, 16]);
            translate([2, 2, 2]) cube([26, 16, 20]);
        }",
    );
    assert!(
        open_box.len() > 20,
        "the kernel produced {}",
        open_box.len()
    );
    let bodies = shells(&open_box, 0.0);
    assert_eq!(
        bodies.len(),
        1,
        "an open box came apart into {} bodies",
        bodies.len()
    );
    assert_eq!(
        bodies[0].len(),
        open_box.len(),
        "triangles were lost or duplicated"
    );
}

#[test]
fn a_sealed_cavity_is_two_surfaces_and_says_so() {
    // A void fully enclosed in solid has an inner surface that touches nothing.
    // Two shells is the truthful answer, not a failure: they are two disjoint
    // surfaces, and anything that reported one would be guessing.
    let sealed = tris(
        "difference() {
            cube([30, 20, 16]);
            translate([8, 6, 5]) cube([10, 8, 6]);
        }",
    );
    let bodies = shells(&sealed, 0.0);
    assert_eq!(
        bodies.len(),
        2,
        "a sealed void gave {} surfaces",
        bodies.len()
    );
    // The outer one is the bigger box.
    let outer = bodies.iter().max_by(|a, b| {
        let ((alo, ahi), (blo, bhi)) = (aabb(a), aabb(b));
        (ahi[0] - alo[0])
            .partial_cmp(&(bhi[0] - blo[0]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (lo, hi) = aabb(outer.unwrap());
    assert!(
        (hi[0] - lo[0] - 30.0).abs() < 1e-6,
        "outer spans {:?}",
        hi[0] - lo[0]
    );
}

#[test]
fn parts_drawn_apart_separate() {
    let apart = tris(
        "cube([30, 20, 16]);
         translate([0, 0, 30]) cube([30, 20, 1.2]);
         translate([-5, 5, 60]) cube([40, 10, 2]);",
    );
    assert_eq!(shells(&apart, 0.0).len(), 3);
}

#[test]
fn parts_drawn_touching_come_back_as_one_body() {
    // The trap. A lid resting on a box shares that face's corners exactly, so
    // connectivity cannot tell them apart — and quietly getting one body where
    // there are two is what makes a later cross-pose check compare the wrong
    // things. Pinned here so it stays a documented property rather than a
    // surprise.
    let unioned = tris("cube([30, 20, 16]); translate([0, 0, 16]) cube([30, 20, 1.2]);");
    assert_eq!(shells(&unioned, 0.0).len(), 1, "the union merged them");

    // And keeping the meshes separate does not help: the vertices still
    // coincide, so the weld joins what the boolean did not.
    let concatenated = tris(
        "assembly() {
            cube([30, 20, 16]);
            translate([0, 0, 16]) cube([30, 20, 1.2]);
        }",
    );
    assert!(
        concatenated.len() > unioned.len(),
        "assembly() should keep both meshes whole"
    );
    assert_eq!(
        shells(&concatenated, 0.0).len(),
        1,
        "coincident faces welded, as they must"
    );

    // Drawn with the clearance a real lid has, they separate.
    let with_clearance = tris(
        "assembly() {
            cube([30, 20, 16]);
            translate([0, 0, 16.2]) cube([30, 20, 1.2]);
        }",
    );
    assert_eq!(shells(&with_clearance, 0.0).len(), 2);
}

#[test]
fn a_curved_boolean_survives_welding() {
    // A bore through a block: the cylinder wall is a fan of narrow triangles
    // meeting the block's faces, which is where a weld tolerance either holds a
    // body together or does not.
    let drilled = tris(
        "difference() {
            cube([40, 20, 8]);
            translate([15, 10, -1]) cylinder(h = 20, r = 3, $fn = 48);
        }",
    );
    let bodies = shells(&drilled, 0.0);
    assert_eq!(
        bodies.len(),
        1,
        "a drilled block split into {} bodies",
        bodies.len()
    );
}

#[test]
fn a_real_part_keeps_its_identity_when_moved() {
    // The same solid, drawn in two places. A key that survives this survives
    // the thing it exists for.
    let here = tris(
        "difference() {
            cube([40, 20, 8]);
            translate([15, 10, -1]) cylinder(h = 20, r = 3, $fn = 32);
        }",
    );
    let there = tris(
        "translate([100, -60, 25]) rotate([0, 0, 37])
         difference() {
            cube([40, 20, 8]);
            translate([15, 10, -1]) cylinder(h = 20, r = 3, $fn = 32);
        }",
    );
    let (a, b) = (rigid_key(&here), rigid_key(&there));
    assert!(a.matches(&b), "{a:?}\nvs\n{b:?}");
}

#[test]
fn two_real_parts_that_are_not_the_same_do_not_match() {
    let plain = tris("cube([40, 20, 8]);");
    let drilled = tris(
        "difference() {
            cube([40, 20, 8]);
            translate([15, 10, -1]) cylinder(h = 20, r = 3, $fn = 32);
        }",
    );
    assert!(
        !rigid_key(&plain).matches(&rigid_key(&drilled)),
        "a bore made no difference to the identity"
    );
}

#[test]
fn closest_approach_between_real_parts_is_the_drawn_clearance() {
    // Two blocks 0.4 apart, as drawn.
    let soup = tris(
        "cube([20, 20, 8]);
         translate([20.4, 0, 0]) cube([20, 20, 8]);",
    );
    let bodies = shells(&soup, 0.0);
    assert_eq!(bodies.len(), 2);
    let gap = nearest(&bodies[0], &bodies[1]);
    assert!((gap - 0.4).abs() < 1e-4, "measured {gap}");
    assert!(!interfere(&bodies[0], &bodies[1]));
}

#[test]
fn a_pin_in_a_bore_reads_as_clearance_and_not_as_interference() {
    // The case a CAD assembly is full of: a pin drawn 0.2 under the bore it
    // sits in. They must read as close, and must not read as overlapping.
    let block = tris(
        "difference() {
            cube([40, 20, 8]);
            translate([20, 10, -1]) cylinder(h = 20, r = 3, $fn = 64);
        }",
    );
    let pin = tris("translate([20, 10, 0]) cylinder(h = 8, r = 2.8, $fn = 64);");
    let gap = nearest(&block, &pin);
    assert!((gap - 0.2).abs() < 0.02, "the clearance measured {gap}");
    assert!(
        !interfere(&block, &pin),
        "a pin with clearance read as interfering"
    );

    // And an oversized pin does interfere.
    let tight = tris("translate([20, 10, 0]) cylinder(h = 8, r = 3.4, $fn = 64);");
    assert!(nearest(&block, &tight) == 0.0);
    assert!(
        interfere(&block, &tight),
        "an interference fit read as clear"
    );
}

#[test]
fn containment_works_on_a_real_shell() {
    let block = tris(
        "difference() {
            cube([40, 20, 8]);
            translate([20, 10, -1]) cylinder(h = 20, r = 3, $fn = 48);
        }",
    );
    assert!(inside(&block, [5.0, 10.0, 4.0]), "a point in the material");
    assert!(!inside(&block, [20.0, 10.0, 4.0]), "a point in the bore");
    assert!(!inside(&block, [-5.0, 10.0, 4.0]), "a point outside");
}
