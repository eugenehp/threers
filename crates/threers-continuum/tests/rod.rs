//! The rod against closed-form beam theory, and cables against the rod.
//!
//! These run with a raised solver budget, which is not incidental — see
//! `a_long_chain_needs_a_bigger_solver_budget` at the bottom for the measured
//! reason, and the crate docs for how to pick one.

use threers_continuum::prelude::*;
use threers_physics::prelude::*;

const G: f32 = 9.81;

/// A world tuned for chains: sequential impulses carry information about one
/// constraint per iteration, and a rod is nothing but a long row of them.
fn rod_world() -> World {
    let mut world = World::new();
    world.substeps = 16;
    world.solver_config.velocity_iterations = 64;
    world
}

/// A 300 mm nylon-ish rod, 2 mm radius: floppy enough to droop measurably and
/// stiff enough that the droop is still small-deflection.
fn test_rod(links: usize) -> Rod {
    Rod::new(0.3, links)
        .radius(0.002)
        .material(2.0e9, 0.35, 1200.0)
        .damped(0.05)
}

fn settle(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step_fixed();
    }
}

/// `δ = wL⁴/8EI`, the tip deflection of a uniformly loaded cantilever.
fn beam_droop(rod: &Rod) -> f32 {
    let ei = rod.youngs * rod.second_moment();
    let w = rod.mass() * G / rod.length;
    w * rod.length.powi(4) / (8.0 * ei)
}

/// Build a horizontal cantilever, let it settle, and report the tip droop and
/// the arc length it settled at.
fn cantilever(links: usize, steps: usize) -> (f32, f32) {
    let rod = test_rod(links);
    let mut world = rod_world();
    let arm = Continuum::build(
        &mut world,
        rod,
        None,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    settle(&mut world, steps);
    (-arm.tip(&world).y, arm.arc_length(&world))
}

#[test]
fn a_cantilever_droops_by_the_beam_formula() {
    // The whole model in one number: hinge stiffnesses derived from `n·EI/L`,
    // dropped into a solver that has never heard of beams, reproducing the
    // closed-form deflection of the beam they came from.
    let rod = test_rod(10);
    let (droop, arc) = cantilever(10, 1200);
    let predicted = beam_droop(&rod);
    assert!(
        (droop - predicted).abs() < 0.1 * predicted,
        "10-link cantilever drooped {droop:.5} m, wL⁴/8EI says {predicted:.5} m"
    );
    assert!(
        (arc - rod.length).abs() < 0.002,
        "the rod stretched: {arc} vs {}",
        rod.length
    );
}

#[test]
fn convergence_is_in_the_shape_not_the_stiffness() {
    // `n·EI/L` makes each station stiffen as the rod is chopped finer, so the
    // rod as a whole bends the same and merely bends it more smoothly. Both
    // discretisations land on the beam answer; the finer one lands closer.
    let predicted = beam_droop(&test_rod(5));
    let coarse = (cantilever(5, 1200).0 - predicted).abs();
    let fine = (cantilever(10, 1200).0 - predicted).abs();
    assert!(
        coarse < 0.15 * predicted && fine < 0.1 * predicted,
        "coarse off by {coarse:.6}, fine off by {fine:.6}, beam {predicted:.6}"
    );
    assert!(
        fine <= coarse,
        "chopping finer made it worse: {coarse:.6} -> {fine:.6}"
    );
}

/// A rod standing along +Y with three cables at 120°, the first toward +X.
fn tendon_rig(links: usize) -> (World, Continuum) {
    let mut world = rod_world();
    world.gravity = Vector3::ZERO; // the cable's doing, not gravity's
    let mut arm = Continuum::build(
        &mut world,
        test_rod(links),
        None,
        Vector3::ZERO,
        Vector3::UP,
    );
    arm.add_tendon_ring(&mut world, 3, 0.0015, 0.0, 0, 0.0);
    (world, arm)
}

#[test]
fn pulling_a_cable_bends_the_rod_toward_it() {
    let (mut world, arm) = tendon_rig(10);
    arm.set_pull(&mut world, 0, 0.0015, 20.0);
    settle(&mut world, 900);

    let tip = arm.tip(&world);
    assert!(
        tip.x > 0.01,
        "pulling the +x cable should curl the rod that way; tip at {tip:?}"
    );
    assert!(
        tip.z.abs() < 0.5 * tip.x.abs(),
        "it should stay near the plane of the cable it was pulled by: {tip:?}"
    );
    assert!(
        tip.y < 0.3,
        "a bent rod is shorter end to end than a straight one: {tip:?}"
    );
}

#[test]
fn the_cable_bends_the_rod_without_stretching_it() {
    let (mut world, arm) = tendon_rig(10);
    arm.set_pull(&mut world, 0, 0.0015, 20.0);
    settle(&mut world, 900);
    let arc = arm.arc_length(&world);
    assert!(
        (arc - 0.3).abs() < 0.005,
        "the backbone is inextensible; it measured {arc}"
    );
}

#[test]
fn opposite_cables_bend_it_opposite_ways() {
    let mut tips = Vec::new();
    for cable in [0usize, 1] {
        let (mut world, arm) = tendon_rig(10);
        arm.set_pull(&mut world, cable, 0.0015, 20.0);
        settle(&mut world, 900);
        tips.push(arm.tip(&world));
    }
    // Cable 0 sits at 0°, cable 1 at 120°, so the two tips must end up on
    // opposite sides of the rod — a negative dot product in the cross-section.
    let (a, b) = (tips[0], tips[1]);
    let flat = |v: Vector3| Vector3::new(v.x, 0.0, v.z);
    assert!(
        flat(a).dot(flat(b)) < 0.0,
        "cables 120° apart bent the rod to {a:?} and {b:?}"
    );
}

#[test]
fn clark_coordinates_steer_the_bend_around_the_rod() {
    // The controller's coordinate system, checked against the physics: a Clark
    // vector must produce a tip on the far side from the cable it reels in,
    // because the reeled cable is the inside of the bend.
    let clark = Clark::new(3, 0.0015, 0.0);
    let (mut world, arm) = tendon_rig(10);
    let command = clark.from_bend(0.6, 0.0);
    arm.set_pulls(&mut world, &clark.to_pulls(command), 20.0);
    settle(&mut world, 900);

    let tip = arm.tip(&world);
    assert!(
        tip.x > 0.005,
        "a bend commanded toward +x went to {tip:?} instead"
    );
    let (angle, direction) = clark.bend(command);
    assert!((angle - 0.6).abs() < 1e-5 && direction.abs() < 1e-5);
}

#[test]
fn a_long_chain_needs_a_bigger_solver_budget() {
    // Not a property of this crate but of the solver under it, pinned here
    // because it is what decides how many links are usable. A rod is a chain of
    // constraints and sequential impulses carry information about roughly one
    // per iteration, so past a handful of links the default budget does not
    // merely sag — it diverges. A plain row of `Joint::fixed` welds does the
    // same, which is how you can tell it is not the springs.
    let rod = test_rod(10);
    let predicted = beam_droop(&rod);

    let mut lean = World::new(); // the defaults: 4 substeps, 4 iterations
    let arm = Continuum::build(
        &mut lean,
        rod.clone(),
        None,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    settle(&mut lean, 600);
    let sloppy = -arm.tip(&lean).y;

    assert!(
        (sloppy - predicted).abs() > 0.1 * predicted,
        "the default budget now holds a 10-link rod ({sloppy:.5} vs {predicted:.5}) — \
         good news, and this test and the crate docs should be updated to say so"
    );
    let (tuned, _) = cantilever(10, 1200);
    assert!(
        (tuned - predicted).abs() < 0.1 * predicted,
        "raising the budget should fix it: {tuned:.5} vs {predicted:.5}"
    );
}
