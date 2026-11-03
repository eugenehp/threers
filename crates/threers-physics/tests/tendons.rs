//! Cables routed through more than two points.
//!
//! The property under test throughout is that the constraint is on the *path* —
//! its total length — rather than on any one segment. A rope over a guide pulls
//! its load toward the guide, not toward the anchor, and shortening it anywhere
//! shortens it everywhere.

use threers_physics::prelude::*;

const G: f32 = 9.81;

fn settle(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step_fixed();
    }
}

fn hanging_world(load_at: Vector3, mass: f32) -> (World, BodyId, BodyId) {
    let mut world = World::new();
    let anchor = world.add_body(RigidBody::fixed().shape(Shape::ball(0.02)));
    let load = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.05))
            .mass(mass)
            .translation(load_at)
            .linear_damping(1.5)
            .angular_damping(1.5)
            .can_sleep(false),
    );
    (world, anchor, load)
}

#[test]
fn a_rope_holds_a_load_at_its_length() {
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), 5.0);
    let rope = world.add_tendon(Tendon::rope(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        1.0,
    ));
    settle(&mut world, 600);

    let length = world.tendon_length(rope).unwrap();
    assert!(
        (length - 1.0).abs() < 0.02,
        "a 1 m rope under 5 kg measured {length} m"
    );
}

#[test]
fn a_slack_rope_does_nothing() {
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), 1.0);
    world.add_tendon(Tendon::rope(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        5.0,
    ));
    settle(&mut world, 120);
    let y = world.body(load).unwrap().translation().y;
    assert!(y < -1.2, "the load should be falling freely, it is at {y}");
}

#[test]
fn the_path_runs_through_the_guides_not_past_them() {
    // Anchor at the origin, a fixed guide out at (1, -1), the load below it.
    // A 2 m rope leaves |guide→load| = 2 − √2 ≈ 0.586 once it is taut, so the
    // load hangs that far *under the guide* — pulled sideways to a place the
    // straight line from the anchor never passes through.
    let mut world = World::new();
    let anchor = world.add_body(RigidBody::fixed().shape(Shape::ball(0.02)));
    let guide = world.add_body(
        RigidBody::fixed()
            .shape(Shape::ball(0.02))
            .translation(Vector3::new(1.0, -1.0, 0.0)),
    );
    let load = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.03))
            .mass(1.0)
            .translation(Vector3::new(1.0, -1.4, 0.0))
            .linear_damping(2.0)
            .can_sleep(false),
    );
    let rope = world.add_tendon(Tendon::rope(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(guide, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        2.0,
    ));
    settle(&mut world, 1200);

    let expected = Vector3::new(1.0, -1.0 - (2.0 - 2.0f32.sqrt()), 0.0);
    let got = world.body(load).unwrap().translation();
    assert!(
        (got - expected).length() < 0.05,
        "load settled at {got:?}, the routed path puts it at {expected:?}"
    );
    let length = world.tendon_length(rope).unwrap();
    assert!(
        (length - 2.0).abs() < 0.03,
        "path measured {length}, the rope is 2 m"
    );
}

#[test]
fn tension_reads_the_weight_it_is_carrying() {
    let mass = 3.0;
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), mass);
    let rope = world.add_tendon(Tendon::rope(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        1.0,
    ));
    settle(&mut world, 900);

    let tension = world.tendon_tension(rope).unwrap();
    let weight = mass * G;
    assert!(
        (tension - weight).abs() < 0.1 * weight,
        "a {mass} kg load hanging still reads {tension:.2} N, weighs {weight:.2} N"
    );
}

#[test]
fn a_constant_pull_lifts_what_it_can_and_no_more() {
    // 40 N on 2 kg goes up; 10 N on 2 kg does not.
    for (tension, rises) in [(40.0f32, true), (10.0, false)] {
        let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), 2.0);
        world.add_tendon(Tendon::pulled(
            vec![
                TendonPoint::new(anchor, Vector3::ZERO),
                TendonPoint::new(load, Vector3::ZERO),
            ],
            tension,
        ));
        let start = world.body(load).unwrap().translation().y;
        settle(&mut world, 120);
        let end = world.body(load).unwrap().translation().y;
        assert_eq!(
            end > start,
            rises,
            "{tension} N on 2 kg went {start} -> {end}"
        );
    }
}

#[test]
fn an_elastic_cord_stretches_by_hookes_law() {
    let (mass, stiffness) = (2.0f32, 500.0f32);
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), mass);
    let cord = world.add_tendon(Tendon::sprung(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        1.0,
        stiffness,
        20.0,
    ));
    settle(&mut world, 1200);

    let stretch = world.tendon_length(cord).unwrap() - 1.0;
    let expected = mass * G / stiffness;
    assert!(
        (stretch - expected).abs() < 0.15 * expected,
        "a {stiffness} N/m cord under {mass} kg stretched {stretch:.4} m, Hooke says {expected:.4} m"
    );
}

#[test]
fn a_winch_reels_to_its_target_length() {
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -2.0, 0.0), 1.0);
    let winch = world.add_tendon(Tendon::new(vec![
        TendonPoint::new(anchor, Vector3::ZERO),
        TendonPoint::new(load, Vector3::ZERO),
    ]));
    world.tendon_mut(winch).unwrap().kind = TendonKind::winch(0.5, 200.0);
    settle(&mut world, 900);

    let length = world.tendon_length(winch).unwrap();
    assert!(
        (length - 0.5).abs() < 0.03,
        "the winch was told 0.5 m and reeled to {length}"
    );
}

#[test]
fn a_winch_that_cannot_lift_the_load_stalls_instead_of_dragging_it() {
    // The force ceiling is what makes a drive beatable: 5 N of winch against
    // 10 kg of load reels in nothing at all.
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -2.0, 0.0), 10.0);
    let winch = world.add_tendon(Tendon::new(vec![
        TendonPoint::new(anchor, Vector3::ZERO),
        TendonPoint::new(load, Vector3::ZERO),
    ]));
    world.tendon_mut(winch).unwrap().kind = TendonKind::winch(0.5, 5.0);
    settle(&mut world, 600);

    let y = world.body(load).unwrap().translation().y;
    assert!(
        y < -2.0,
        "a 5 N winch hauled a 10 kg load from -2.0 up to {y}"
    );
}

#[test]
fn a_path_that_lies_entirely_on_one_body_cannot_be_pulled() {
    // Its length is a fact about the body, not about the world, so no amount of
    // tension can change it and none may be applied. If the two ends of a
    // within-body segment did not cancel in the Jacobian, the constraint would
    // be trying to shorten a rigid part of its own path — and the body would
    // shoot off sideways for no reason anyone could name.
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let bar = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.5, 0.02, 0.02))
            .mass(1.0)
            .can_sleep(false),
    );
    let rope = world.add_tendon(Tendon::pulled(
        vec![
            TendonPoint::new(bar, Vector3::new(-0.5, 0.0, 0.0)),
            TendonPoint::new(bar, Vector3::new(0.5, 0.0, 0.0)),
        ],
        1000.0,
    ));
    let before = world.tendon_length(rope).unwrap();
    settle(&mut world, 120);

    let body = world.body(bar).unwrap();
    assert!(
        body.linear_velocity.length() < 1e-4 && body.angular_velocity.length() < 1e-4,
        "1000 N through a rigid path moved the bar: v={:?} w={:?}",
        body.linear_velocity,
        body.angular_velocity
    );
    let after = world.tendon_length(rope).unwrap();
    assert!(
        (after - before).abs() < 1e-5,
        "a rigid path changed length: {before} -> {after}"
    );
}

#[test]
fn a_guide_on_a_moving_body_carries_its_share() {
    // The middle of the route on a *dynamic* body: pulling the cable has to
    // load the guide too, and from the direction the cable actually leaves it.
    let mut world = World::new();
    world.gravity = Vector3::ZERO;
    let anchor = world.add_body(RigidBody::fixed().shape(Shape::ball(0.02)));
    let guide = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.05))
            .mass(1.0)
            .translation(Vector3::new(1.0, 0.0, 0.0))
            .can_sleep(false),
    );
    let end = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.05))
            .mass(1.0)
            .translation(Vector3::new(1.0, -1.0, 0.0))
            .can_sleep(false),
    );
    world.add_tendon(Tendon::pulled(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(guide, Vector3::ZERO),
            TendonPoint::new(end, Vector3::ZERO),
        ],
        10.0,
    ));
    // Two steps only: this is about the direction the load goes on, and after a
    // second of 10 N on 1 kg the end has overtaken the guide and the geometry
    // that was being tested is gone.
    settle(&mut world, 2);

    // The guide is pulled toward the anchor (−X) *and* toward the end (−Y);
    // the end is pulled straight at the guide (+Y).
    let g = world.body(guide).unwrap().linear_velocity;
    let e = world.body(end).unwrap().linear_velocity;
    assert!(g.x < -0.1, "the guide was not pulled toward the anchor: {g:?}");
    assert!(g.y < -0.1, "the guide was not pulled toward the end: {g:?}");
    assert!(
        e.y > 0.1 && e.x.abs() < 0.05,
        "the end should be pulled straight up its own segment: {e:?}"
    );
}

#[test]
fn removing_a_guide_removes_the_cable_through_it() {
    let (mut world, anchor, load) = hanging_world(Vector3::new(0.0, -1.0, 0.0), 1.0);
    let rope = world.add_tendon(Tendon::rope(
        vec![
            TendonPoint::new(anchor, Vector3::ZERO),
            TendonPoint::new(load, Vector3::ZERO),
        ],
        1.0,
    ));
    assert!(world.tendon(rope).is_some());
    world.remove_body(load);
    assert!(
        world.tendon(rope).is_none(),
        "a cable outlived one of its guides"
    );
    settle(&mut world, 10);
}
