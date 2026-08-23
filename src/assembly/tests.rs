use super::*;

/// A closed, outward-wound box from `o` to `o + s`.
fn cube(o: [f64; 3], s: f64) -> Vec<Tri> {
    let p = |x: f64, y: f64, z: f64| [o[0] + x * s, o[1] + y * s, o[2] + z * s];
    let (a, b, c, d) = (p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 0.), p(0., 1., 0.));
    let (e, f, g, h) = (p(0., 0., 1.), p(1., 0., 1.), p(1., 1., 1.), p(0., 1., 1.));
    vec![
        [a, c, b],
        [a, d, c], // -z
        [e, f, g],
        [e, g, h], // +z
        [a, b, f],
        [a, f, e], // -y
        [d, h, g],
        [d, g, c], // +y
        [a, e, h],
        [a, h, d], // -x
        [b, c, g],
        [b, g, f], // +x
    ]
}

/// A closed, outward-wound box spanning `lo` to `hi`.
fn boxed(lo: [f64; 3], hi: [f64; 3]) -> Vec<Tri> {
    let mut c = cube([0.0; 3], 1.0);
    for t in c.iter_mut() {
        for v in t.iter_mut() {
            for k in 0..3 {
                v[k] = lo[k] + v[k] * (hi[k] - lo[k]);
            }
        }
    }
    c
}

fn spin_z(tris: &[Tri], angle: f64, about: [f64; 3]) -> Vec<Tri> {
    let (s, c) = angle.sin_cos();
    let map = |p: &[f64; 3]| {
        let (x, y) = (p[0] - about[0], p[1] - about[1]);
        [about[0] + c * x - s * y, about[1] + s * x + c * y, p[2]]
    };
    tris.iter()
        .map(|t| [map(&t[0]), map(&t[1]), map(&t[2])])
        .collect()
}

fn shift(tris: &[Tri], by: [f64; 3]) -> Vec<Tri> {
    let map = |p: &[f64; 3]| [p[0] + by[0], p[1] + by[1], p[2] + by[2]];
    tris.iter()
        .map(|t| [map(&t[0]), map(&t[1]), map(&t[2])])
        .collect()
}

/// Cut every triangle into three about its centroid. Same surface, three times
/// the triangles — what a boolean kernel run twice is entitled to hand back.
fn subdivide(tris: &[Tri]) -> Vec<Tri> {
    let mut out = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        let m = [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
            (t[0][2] + t[1][2] + t[2][2]) / 3.0,
        ];
        out.push([t[0], t[1], m]);
        out.push([t[1], t[2], m]);
        out.push([t[2], t[0], m]);
    }
    out
}

// ---- predicates -----------------------------------------------------------

#[test]
fn a_soup_splits_into_the_bodies_it_contains() {
    let mut soup = cube([0.0, 0.0, 0.0], 1.0);
    soup.extend(cube([10.0, 0.0, 0.0], 1.0));
    soup.extend(cube([0.0, 10.0, 0.0], 2.0));
    let bodies = shells(&soup, 0.0);
    assert_eq!(bodies.len(), 3);
    assert!(bodies.iter().all(|b| b.len() == 12));
    // Every triangle is accounted for, exactly once.
    assert_eq!(bodies.iter().map(|b| b.len()).sum::<usize>(), soup.len());
}

#[test]
fn one_body_stays_one_body() {
    let bodies = shells(&cube([0.0, 0.0, 0.0], 1.0), 0.0);
    assert_eq!(bodies.len(), 1);
}

#[test]
fn welding_survives_the_rounding_a_boolean_leaves() {
    // The same cube, with one face's vertices nudged by less than the weld
    // tolerance — which is what a kernel that recomputed a shared face leaves.
    let mut soup = cube([0.0, 0.0, 0.0], 1.0);
    let nudge = 1e-9;
    for t in soup.iter_mut().take(2) {
        for v in t.iter_mut() {
            v[0] += nudge;
        }
    }
    assert_eq!(
        shells(&soup, 0.0).len(),
        1,
        "rounding split one body in two"
    );
}

#[test]
fn containment_is_answered_by_winding() {
    let c = cube([0.0, 0.0, 0.0], 2.0);
    assert!(inside(&c, [1.0, 1.0, 1.0]));
    assert!(!inside(&c, [3.0, 1.0, 1.0]));
    assert!((winding(&c, [1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
    assert!(winding(&c, [5.0, 5.0, 5.0]).abs() < 1e-6);
}

#[test]
fn closest_approach_is_the_gap_between_the_faces() {
    let a = cube([0.0, 0.0, 0.0], 1.0);
    let b = cube([3.0, 0.0, 0.0], 1.0);
    assert!((nearest(&a, &b) - 2.0).abs() < 1e-9, "{}", nearest(&a, &b));

    // Touching face to face reads as zero, not as a small positive number.
    let touching = cube([1.0, 0.0, 0.0], 1.0);
    assert!(nearest(&a, &touching) < 1e-9);
}

#[test]
fn shared_volume_with_no_crossing_at_all_is_still_interference() {
    // Two boxes of the same height and depth, overlapped along their length.
    // Every place their surfaces meet is face-on-face or edge-on-edge: nothing
    // passes through anything, and half of one is inside the other. A crossing
    // test alone reports these as clear.
    let a = cube([0.0, 0.0, 0.0], 1.0);
    let b = cube([0.5, 0.0, 0.0], 1.0);
    assert!(
        !a.iter().any(|x| b.iter().any(|y| tri_tri_intersect(x, y))),
        "the test needs a configuration where nothing crosses"
    );
    assert!(interfere(&a, &b), "shared volume missed");
}

#[test]
fn a_shallow_overlap_in_a_big_mesh_is_found_from_either_side() {
    // No crossings anywhere, a mesh with enough faces that a blind sample would
    // be thin, and an overlap that is a sliver of both bodies. The probes have
    // to land on the few faces that reach the other body, from whichever side
    // the question is asked.
    let big = subdivide(&subdivide(&boxed([0.0; 3], [100.0; 3])));
    let slab = boxed([0.0, 0.0, -1.0], [100.0, 100.0, 0.5]);
    assert!(
        big.len() > 100,
        "the test needs a mesh a blind sample would thin"
    );
    assert!(
        !big.iter()
            .any(|x| slab.iter().any(|y| tri_tri_intersect(x, y))),
        "the test needs a configuration where nothing crosses"
    );

    assert!(interfere(&big, &slab), "the overlap was probed past");
    assert!(interfere(&slab, &big), "and the other way round");

    // And moving the slab clear of it goes back to reporting nothing.
    let clear = boxed([0.0, 0.0, -2.0], [100.0, 100.0, -0.5]);
    assert!(!interfere(&big, &clear));
}

#[test]
fn interference_is_shared_volume_and_not_shared_surface() {
    let a = cube([0.0, 0.0, 0.0], 1.0);
    // Offset in every axis, so the surfaces genuinely cross.
    assert!(
        interfere(&a, &cube([0.5, 0.3, 0.2], 1.0)),
        "crossing overlap missed"
    );
    assert!(interfere(&a, &cube([0.5, 0.0, 0.0], 1.0)), "overlap missed");
    assert!(
        !interfere(&a, &cube([3.0, 0.0, 0.0], 1.0)),
        "false positive"
    );
    // Resting contact shares a plane and no volume.
    assert!(
        !interfere(&a, &cube([1.0, 0.0, 0.0], 1.0)),
        "touching is not interfering"
    );
    // One swallowed by the other crosses no surface at all.
    let big = cube([-1.0, -1.0, -1.0], 4.0);
    assert!(interfere(&big, &a), "containment missed");
}

/// A scalene tetrahedron: no symmetry plane, so it is genuinely chiral.
fn chiral_tetra() -> Vec<Tri> {
    let p0 = [0.0, 0.0, 0.0];
    let p1 = [2.0, 0.0, 0.0];
    let p2 = [0.0, 3.0, 0.0];
    let p3 = [0.3, 0.5, 1.7];
    vec![[p0, p2, p1], [p0, p1, p3], [p1, p2, p3], [p2, p0, p3]]
}

/// Reflect through `x = 0`, reversing winding so the faces still point out —
/// which is what a modelling kernel does with `mirror()`.
fn mirror_x(tris: &[Tri]) -> Vec<Tri> {
    tris.iter()
        .map(|t| {
            let f = |p: &[f64; 3]| [-p[0], p[1], p[2]];
            [f(&t[0]), f(&t[2]), f(&t[1])]
        })
        .collect()
}

// ---- identity -------------------------------------------------------------

#[test]
fn a_part_and_its_mirror_are_not_the_same_part() {
    let right = chiral_tetra();
    let left = mirror_x(&right);

    let kr = rigid_key(&right);
    let kl = rigid_key(&left);

    // Every length-valued invariant agrees — which is exactly why handedness
    // has to exist.
    assert!((kr.area_scale - kl.area_scale).abs() < 1e-9);
    assert!((kr.volume_scale - kl.volume_scale).abs() < 1e-9);
    assert!((0..3).all(|k| (kr.moments[k] - kl.moments[k]).abs() < 1e-9));

    assert_ne!(kr.handedness, 0, "the test shape is not chiral");
    assert_eq!(kr.handedness, -kl.handedness, "handedness did not reverse");
    assert!(!kr.matches(&kl), "a part matched its own mirror image");
}

#[test]
fn handedness_survives_rigid_motion_and_retriangulation() {
    let a = chiral_tetra();
    let moved = shift(&spin_z(&a, 1.1, [0.4, -0.3, 0.0]), [5.0, 2.0, -1.0]);
    assert_eq!(rigid_key(&a).handedness, rigid_key(&moved).handedness);
    assert!(rigid_key(&a).matches(&rigid_key(&moved)));

    let rebuilt = subdivide(&subdivide(&a));
    assert_eq!(rigid_key(&a).handedness, rigid_key(&rebuilt).handedness);
    assert!(rigid_key(&a).matches(&rigid_key(&rebuilt)));
}

#[test]
fn a_symmetric_body_reports_no_handedness_and_matches_either() {
    // A cube has three equal moments: no determined frame, and it is its own
    // mirror image anyway.
    let c = cube([0.0, 0.0, 0.0], 1.0);
    assert_eq!(rigid_key(&c).handedness, 0);
    assert!(rigid_key(&c).matches(&rigid_key(&mirror_x(&c))));

    // A box with three different sides has a determined frame, but a mirror
    // plane across every axis — so still no hand.
    let mut brick = cube([0.0, 0.0, 0.0], 1.0);
    for t in brick.iter_mut() {
        for v in t.iter_mut() {
            v[0] *= 3.0;
            v[1] *= 2.0;
        }
    }
    assert_eq!(rigid_key(&brick).handedness, 0, "a brick has no handedness");
    assert!(rigid_key(&brick).matches(&rigid_key(&mirror_x(&brick))));
}

#[test]
fn correspondence_does_not_pair_a_left_hand_part_with_a_right_hand_one() {
    let right = chiral_tetra();
    let left = shift(&mirror_x(&right), [10.0, 0.0, 0.0]);
    // Both poses hold both parts; the rebuild lists them the other way round.
    let before = vec![right.clone(), left.clone()];
    let after = vec![
        shift(&left, [0.1, 0.0, 0.0]),
        shift(&right, [0.1, 0.0, 0.0]),
    ];
    let pairs = correspond(&before, &after);
    assert!(
        pairs.contains(&(0, 1)) && pairs.contains(&(1, 0)),
        "the hands were swapped: {pairs:?}"
    );
}

#[test]
fn a_key_survives_rigid_motion() {
    let a = cube([0.0, 0.0, 0.0], 1.7);
    let moved = shift(&spin_z(&a, 0.9, [0.3, -0.2, 0.0]), [12.0, -4.0, 3.0]);
    assert!(
        rigid_key(&a).matches(&rigid_key(&moved)),
        "{:?} vs {:?}",
        rigid_key(&a),
        rigid_key(&moved)
    );
}

#[test]
fn a_key_survives_retriangulation() {
    // The failure the old key had: it counted triangles, so the same solid
    // rebuilt with a different triangulation was a different body.
    let a = cube([0.0, 0.0, 0.0], 1.0);
    let rebuilt = subdivide(&a);
    assert_ne!(
        a.len(),
        rebuilt.len(),
        "the test needs the counts to differ"
    );
    assert!(
        rigid_key(&a).matches(&rigid_key(&rebuilt)),
        "retriangulation changed the identity"
    );
}

#[test]
fn a_key_tells_different_bodies_apart() {
    let a = cube([0.0, 0.0, 0.0], 1.0);
    let bigger = cube([0.0, 0.0, 0.0], 1.05);
    assert!(!rigid_key(&a).matches(&rigid_key(&bigger)), "5% missed");

    // Same volume, different shape: a slab against a cube.
    let mut slab = cube([0.0, 0.0, 0.0], 1.0);
    for t in slab.iter_mut() {
        for v in t.iter_mut() {
            v[0] *= 4.0;
            v[2] *= 0.25;
        }
    }
    assert!(!rigid_key(&a).matches(&rigid_key(&slab)), "shape missed");
}

#[test]
fn a_key_does_not_depend_on_what_units_the_model_is_drawn_in() {
    // The old key quantised a length to a fixed step, so whether two bodies
    // shared an identity depended on whether the model was in metres or
    // millimetres. Every field here is a length compared relatively.
    for scale in [1e-3f64, 1.0, 1e3] {
        let a = cube([0.0, 0.0, 0.0], scale);
        let same = shift(&a, [7.0 * scale, 0.0, 0.0]);
        let other = cube([0.0, 0.0, 0.0], scale * 1.05);
        assert!(rigid_key(&a).matches(&rigid_key(&same)), "scale {scale}");
        assert!(!rigid_key(&a).matches(&rigid_key(&other)), "scale {scale}");
    }
}

#[test]
fn correspondence_finds_bodies_that_moved_and_were_reordered() {
    let small = cube([0.0, 0.0, 0.0], 1.0);
    let large = cube([5.0, 0.0, 0.0], 2.0);
    let before = vec![small.clone(), large.clone()];
    // Rebuilt: different order, both moved a little.
    let after = vec![
        shift(&large, [0.1, 0.0, 0.0]),
        shift(&small, [0.0, 0.1, 0.0]),
    ];
    let pairs = correspond(&before, &after);
    assert_eq!(pairs.len(), 2);
    assert!(pairs.contains(&(0, 1)), "{pairs:?}");
    assert!(pairs.contains(&(1, 0)), "{pairs:?}");
}

#[test]
fn identical_parts_are_separated_by_which_one_is_nearest() {
    let a = cube([0.0, 0.0, 0.0], 1.0);
    let b = cube([10.0, 0.0, 0.0], 1.0);
    let before = vec![a.clone(), b.clone()];
    let after = vec![shift(&b, [0.2, 0.0, 0.0]), shift(&a, [0.2, 0.0, 0.0])];
    let pairs = correspond(&before, &after);
    assert!(
        pairs.contains(&(0, 1)) && pairs.contains(&(1, 0)),
        "{pairs:?}"
    );
}

// ---- motion ---------------------------------------------------------------

#[test]
fn an_axis_is_recovered_from_two_poses() {
    let before = cube([2.0, 0.0, 0.0], 1.0);
    let pivot = [0.0, 0.0, 0.0];
    let after = spin_z(&before, 0.7, pivot);
    let (point, angle) = recover_axis(&before, &after, [0.0, 0.0, 1.0]).unwrap();
    assert!((angle - 0.7).abs() < 1e-6, "angle was {angle}");
    // The axis passes through the pivot; anywhere along z is the same axis.
    assert!(
        point[0].abs() < 1e-6 && point[1].abs() < 1e-6,
        "point was {point:?}"
    );
}

#[test]
fn a_rebuilt_mesh_is_refused_rather_than_fitted() {
    // Same triangle count, vertices no longer corresponding — which used to
    // produce an axis derived from unrelated points.
    let before = cube([2.0, 0.0, 0.0], 1.0);
    let mut after = spin_z(&before, 0.7, [0.0, 0.0, 0.0]);
    after.reverse();
    assert!(
        recover_axis(&before, &after, [0.0, 0.0, 1.0]).is_none(),
        "a bogus correspondence was fitted anyway"
    );
}

#[test]
fn a_body_that_barely_turned_locates_no_axis() {
    let before = cube([2.0, 0.0, 0.0], 1.0);
    let after = spin_z(&before, 1e-5, [0.0, 0.0, 0.0]);
    assert!(recover_axis(&before, &after, [0.0, 0.0, 1.0]).is_none());
}

#[test]
fn a_screw_recovers_its_own_direction_and_slide() {
    let before = cube([2.0, 0.0, 0.0], 1.0);
    let turned = spin_z(&before, 0.6, [0.0, 0.0, 0.0]);
    let after = shift(&turned, [0.0, 0.0, 1.5]);

    let s = recover_screw(&before, &after).unwrap();
    assert!(
        s.direction[2].abs() > 0.999,
        "direction was {:?}",
        s.direction
    );
    // The direction may come back either way round; the angle follows it.
    let flip = s.direction[2].signum();
    assert!((s.angle * flip - 0.6).abs() < 1e-6, "angle was {}", s.angle);
    assert!((s.slide * flip - 1.5).abs() < 1e-6, "slide was {}", s.slide);
    assert!(s.residual < 1e-9, "residual was {}", s.residual);
    assert!(
        !s.is_revolute(1e-6) && !s.is_prismatic(1e-6),
        "it is helical"
    );
}

#[test]
fn a_pure_slide_has_a_direction_and_no_located_axis() {
    let before = cube([0.0, 0.0, 0.0], 1.0);
    let after = shift(&before, [0.0, 3.0, 0.0]);
    let s = recover_screw(&before, &after).unwrap();
    assert!(s.angle.abs() < 1e-9);
    assert!((s.slide - 3.0).abs() < 1e-9);
    assert!(s.direction[1].abs() > 0.999);
    assert!(s.is_prismatic(1e-6));
}

#[test]
fn a_hinge_reads_as_a_hinge() {
    let before = cube([2.0, 0.0, 0.0], 1.0);
    let after = spin_z(&before, 0.5, [0.0, 0.0, 0.0]);
    let s = recover_screw(&before, &after).unwrap();
    assert!(s.is_revolute(1e-6), "{s:?}");
    assert!(s.lead().abs() < 1e-6);
}

// ---- across poses ---------------------------------------------------------

/// A base, a lid resting on it, and a loose part off to one side.
fn assembly_poses(lid_lift: &[f64]) -> Vec<Vec<Body>> {
    let base = cube([0.0, 0.0, 0.0], 1.0);
    let loose = cube([20.0, 0.0, 0.0], 0.5);
    lid_lift
        .iter()
        .map(|&z| {
            vec![
                base.clone(),
                shift(&cube([0.0, 0.0, 1.0], 1.0), [0.0, 0.0, z]),
                loose.clone(),
            ]
        })
        .collect()
}

#[test]
fn alignment_tracks_bodies_through_a_sweep() {
    let poses = assembly_poses(&[0.0, 0.1, 0.2, 0.3]);
    let a = align(&poses);
    assert_eq!(a.len(), 4);
    for pose in &a {
        assert_eq!(pose, &vec![Some(0), Some(1), Some(2)]);
    }
}

#[test]
fn a_pair_that_stays_together_is_persistent() {
    // The lid never lifts: it rests on the base in every pose.
    let poses = assembly_poses(&[0.0, 0.0, 0.0]);
    let e = engagement(&poses, 1e-6);
    let pair = e
        .iter()
        .find(|p| (p.a, p.b) == (0, 1))
        .expect("base and lid touch");
    assert!(pair.persistent(), "{pair:?}");
    assert_eq!(pair.touching, 3);
}

#[test]
fn a_joint_coming_apart_is_reported_as_coming_apart() {
    // This is the case an interference test cannot see: nothing overlaps in
    // any pose, and the lid has left.
    let poses = assembly_poses(&[0.0, 0.0, 2.0]);
    let e = engagement(&poses, 1e-6);
    let pair = e
        .iter()
        .find(|p| (p.a, p.b) == (0, 1))
        .expect("they touch at first");
    assert!(!pair.persistent(), "the separation was not noticed");
    assert!(pair.came_apart(), "it started together and ended apart");
    assert!(!pair.came_together());
    assert_eq!(pair.touching, 2);
    assert!(
        (pair.widest - 2.0).abs() < 1e-6,
        "widest was {}",
        pair.widest
    );
}

#[test]
fn coming_into_contact_is_not_coming_apart() {
    // A lid swinging onto a stop touches partway through. It is not persistent
    // and nothing has failed — telling those two apart is the whole point of
    // asking about the first and last pose rather than counting.
    let poses = assembly_poses(&[3.0, 1.5, 0.0]);
    let e = engagement(&poses, 1e-6);
    let pair = e
        .iter()
        .find(|p| (p.a, p.b) == (0, 1))
        .expect("they end up touching");
    assert!(!pair.persistent());
    assert!(pair.came_together(), "it arrived rather than left");
    assert!(
        !pair.came_apart(),
        "an arrival was reported as a separation"
    );
}

#[test]
fn an_unjoined_body_is_found() {
    let poses = assembly_poses(&[0.0, 0.1]);
    let loose = floating(&poses, 1e-6);
    assert_eq!(loose, vec![2], "the part that touches nothing was missed");
}

#[test]
fn nothing_floats_when_everything_touches() {
    let base = cube([0.0, 0.0, 0.0], 1.0);
    let lid = cube([0.0, 0.0, 1.0], 1.0);
    let poses = vec![vec![base.clone(), lid.clone()], vec![base, lid]];
    assert!(floating(&poses, 1e-6).is_empty());
}

// ---- free volume ----------------------------------------------------------

#[test]
fn a_free_band_is_found_beside_a_turning_part() {
    // A part hugging the axis, swept. Everything beyond it is free.
    let arm = cube([0.0, -0.25, 0.0], 0.5);
    let poses: Vec<Vec<Tri>> = (0..8)
        .map(|i| spin_z(&arm, i as f64 * 0.5, [0.0, 0.0, 0.0]))
        .collect();
    let bands = free_annuli(&poses, [0.0; 3], [0.0, 0.0, 1.0], 4.0, 40, 0.0, 4, 0.25);
    assert!(!bands.is_empty(), "no free band beside a small swept part");
    // The widest free run reaches the outer radius.
    assert!(
        bands.iter().any(|(_, _, hi)| *hi > 3.0),
        "the free band stopped short: {bands:?}"
    );
}

#[test]
fn degenerate_bin_counts_return_nothing_rather_than_panicking() {
    let c = vec![cube([0.0, 0.0, 0.0], 1.0)];
    let axis = [0.0, 0.0, 1.0];
    assert!(free_annuli(&c, [0.0; 3], axis, 4.0, 0, 0.0, 4, 0.25).is_empty());
    assert!(free_annuli(&c, [0.0; 3], axis, 4.0, 40, 0.0, 0, 0.25).is_empty());
    assert!(free_annuli(&c, [0.0; 3], axis, 4.0, 40, 0.0, 4, 0.0).is_empty());
    assert!(free_annuli(&c, [0.0; 3], axis, 0.0, 40, 0.0, 4, 0.25).is_empty());
    assert!(free_annuli(&c, [0.0; 3], [0.0; 3], 4.0, 40, 0.0, 4, 0.25).is_empty());
}

#[test]
fn empty_input_is_answered_rather_than_crashed() {
    let empty: Vec<Tri> = Vec::new();
    assert!(shells(&empty, 0.0).is_empty());
    assert!(tri_bbs(&empty).is_empty());
    assert_eq!(nearest(&empty, &cube([0.0; 3], 1.0)), f64::INFINITY);
    assert!(!interfere(&empty, &cube([0.0; 3], 1.0)));
    assert!(recover_screw(&empty, &empty).is_none());
    assert!(engagement(&[], 1e-6).is_empty());
    assert!(floating(&[], 1e-6).is_empty());
    assert!(align(&[]).is_empty());
}

// ---- measurements ---------------------------------------------------------

#[test]
fn area_and_volume_are_what_they_should_be() {
    let c = cube([3.0, -2.0, 1.0], 2.0);
    assert!((area(&c) - 24.0).abs() < 1e-9, "area {}", area(&c));
    assert!((volume(&c) - 8.0).abs() < 1e-9, "volume {}", volume(&c));
    // And both survive being cut up.
    let fine = subdivide(&c);
    assert!((area(&fine) - 24.0).abs() < 1e-9);
    assert!((volume(&fine) - 8.0).abs() < 1e-9);
}

#[test]
fn the_area_centroid_survives_retriangulation_and_the_vertex_one_does_not() {
    // A cube with one face cut finely: the surface is unchanged, so the area
    // centroid is unchanged, and the vertex mean is dragged toward the cut.
    let c = cube([0.0, 0.0, 0.0], 1.0);
    let mut uneven = c.clone();
    let face: Vec<Tri> = uneven.drain(..2).collect();
    uneven.extend(subdivide(&subdivide(&face)));

    let before = centroid_area(&c);
    let after = centroid_area(&uneven);
    assert!(
        (0..3).all(|k| (before[k] - after[k]).abs() < 1e-9),
        "area centroid moved: {before:?} -> {after:?}"
    );

    let vb = centroid(&c);
    let va = centroid(&uneven);
    assert!(
        (0..3).any(|k| (vb[k] - va[k]).abs() > 1e-3),
        "the test assumes the vertex mean is the fragile one"
    );
}
