//! The engine's account of what it did to the scene it was given.

use threers_physics::diagnostics::{Timings, Warning};
use threers_physics::prelude::*;

fn ball_world() -> (World, BodyId) {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));
    let ball = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.5))
            .translation(Vector3::new(0.0, 4.0, 0.0)),
    );
    (world, ball)
}

#[test]
fn a_sane_scene_reports_nothing() {
    let (mut world, _) = ball_world();
    for _ in 0..240 {
        world.step(1.0 / 60.0);
    }
    let d = world.diagnostics();
    assert!(
        d.is_clean(),
        "unexpected: {:?}",
        d.warnings().collect::<Vec<_>>()
    );
    assert_eq!(d.warnings().count(), 0);
}

#[test]
fn a_nan_written_straight_into_a_body_is_recorded_not_hidden() {
    // The velocity fields are public on purpose, so no guard can cover this
    // assignment — the engine repairs it during the step and says so.
    let (mut world, ball) = ball_world();
    world.bodies_mut().get_mut(ball).unwrap().linear_velocity =
        Vector3::new(f32::NAN, 0.0, 0.0);
    world.step_fixed();

    let record = world.diagnostics().warning(Warning::NonFiniteVelocity);
    assert_eq!(record.count, 1);
    assert_eq!(record.first_step, 0);
    // And the repair actually happened.
    let v = world.bodies().get(ball).unwrap().linear_velocity;
    assert!(v.x.is_finite());
}

#[test]
fn an_absurd_but_finite_velocity_is_recorded_when_it_is_clamped() {
    let (mut world, ball) = ball_world();
    world.max_velocity = 100.0;
    world.bodies_mut().get_mut(ball).unwrap().linear_velocity = Vector3::new(1e6, 0.0, 0.0);
    world.step_fixed();

    // Once per substep: gravity pushes it back over the limit each time, so the
    // count is of occurrences rather than of steps.
    assert_eq!(
        world.diagnostics().count(Warning::VelocityClamped),
        world.substeps as u64
    );
    assert!(world.bodies().get(ball).unwrap().linear_velocity.x <= 100.0);
    // The one that was clamped is not also reported as non-finite.
    assert_eq!(world.diagnostics().count(Warning::NonFiniteVelocity), 0);
}

#[test]
fn a_timestep_that_is_not_a_timestep_is_recorded_rather_than_ignored() {
    let (mut world, _) = ball_world();
    world.step(f32::NAN);
    world.step(-1.0);
    world.step(0.0);
    assert_eq!(world.diagnostics().count(Warning::InvalidTimestep), 3);
    assert_eq!(world.diagnostics().steps(), 0, "nothing was simulated");
}

#[test]
fn falling_behind_real_time_is_recorded_where_the_time_is_dropped() {
    let (mut world, _) = ball_world();
    world.timestep = 1.0 / 60.0;
    world.max_substeps = 2;

    // A whole second at once: far more than two steps can consume, so most of
    // it is discarded. Left uncounted, the simulation simply runs slow and
    // nothing says why.
    world.step(1.0);
    assert_eq!(world.diagnostics().count(Warning::TimeDiscarded), 1);
    assert_eq!(world.diagnostics().steps(), 2, "only the budget was run");

    // A frame it can keep up with says nothing.
    world.clear_warnings();
    world.step(1.0 / 60.0);
    assert!(world.diagnostics().is_clean());
}

#[test]
fn a_warning_says_when_it_last_fired_not_only_that_it_did() {
    let (mut world, ball) = ball_world();
    world.bodies_mut().get_mut(ball).unwrap().linear_velocity =
        Vector3::new(f32::NAN, 0.0, 0.0);
    world.step_fixed();
    for _ in 0..10 {
        world.step_fixed();
    }

    // Still counted, but visibly historical — the difference between a scene
    // that hiccuped once and one that is broken right now.
    let record = world.diagnostics().warning(Warning::NonFiniteVelocity);
    assert_eq!(record.count, 1);
    assert_eq!(record.last_step, 0);
    assert_eq!(world.diagnostics().current().count(), 0);
    assert_eq!(world.diagnostics().steps(), 11);
}

#[test]
fn counters_describe_the_step_that_just_ran() {
    let mut world = World::new();
    world.add_body(RigidBody::fixed().shape(Shape::ground()));
    for i in 0..5 {
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(i as f32 * 4.0, 0.49, 0.0)),
        );
    }
    world.step_fixed();

    let c = world.diagnostics().counters();
    assert_eq!(c.bodies, 6);
    assert_eq!(c.awake_bodies, 5, "the ground is not dynamic");
    assert_eq!(c.manifolds, 5, "each box rests on the ground");
    assert!(c.contact_points >= 5);
    assert_eq!(c.islands, 5, "the boxes are spread out and touch nothing else");
    assert_eq!(c.substeps, world.substeps as u32);
}

#[test]
fn sleeping_shows_up_as_bodies_the_step_stops_paying_for() {
    let (mut world, _) = ball_world();
    for _ in 0..2000 {
        world.step_fixed();
    }
    assert_eq!(
        world.diagnostics().counters().awake_bodies,
        0,
        "the ball should have settled and fallen asleep"
    );
}

#[test]
fn timings_are_off_until_asked_for() {
    let (mut world, _) = ball_world();
    world.step_fixed();
    assert_eq!(world.diagnostics().timings(), Timings::default());
    assert!(!world.diagnostics().profiling());

    world.set_profiling(true);
    assert_eq!(world.diagnostics().profiling(), Timings::supported());
    world.step_fixed();

    if !Timings::supported() {
        // wasm has no clock; the contract is that it stays zero and says so.
        assert_eq!(world.diagnostics().timings(), Timings::default());
        return;
    }

    let t = world.diagnostics().timings();
    assert!(t.step > std::time::Duration::ZERO, "the step took no time");
    assert!(
        t.accounted() <= t.step,
        "stages {:?} exceed the whole step {:?}",
        t.accounted(),
        t.step
    );
    assert!(t.stages().iter().any(|(_, d)| *d > std::time::Duration::ZERO));

    // And turning it off puts the numbers back rather than leaving stale ones.
    world.set_profiling(false);
    assert_eq!(world.diagnostics().timings(), Timings::default());
}

#[test]
fn energy_falls_as_a_ball_bounces_and_never_rises() {
    let (mut world, _) = ball_world();
    world.step_fixed();

    let mut previous = world.energy();
    let start = previous;
    for _ in 0..600 {
        world.step_fixed();
        let now = world.energy();
        assert!(
            now <= previous + 0.05 * start.abs().max(1.0),
            "energy rose from {previous} to {now}"
        );
        previous = now;
    }
    assert!(previous < start, "a bouncing ball should lose energy");
}

#[test]
fn energy_is_conserved_by_a_free_body_with_gravity_off() {
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let body = world.add_body(RigidBody::dynamic().shape(Shape::cuboid(0.5, 0.5, 0.5)));
    world.bodies_mut().get_mut(body).unwrap().linear_velocity = Vector3::new(1.0, 2.0, -0.5);

    let start = world.energy();
    for _ in 0..600 {
        world.step_fixed();
    }
    let end = world.energy();
    assert!(
        (end - start).abs() < 1e-3 * start,
        "energy drifted from {start} to {end}"
    );
}
