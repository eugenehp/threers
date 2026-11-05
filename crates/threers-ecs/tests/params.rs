//! Systems as functions of what they use. Run under Miri as well: the
//! parameters hand out borrows of one world at once.

use std::panic::{catch_unwind, AssertUnwindSafe};
use threers_ecs::prelude::*;

#[derive(Debug, PartialEq)]
struct Velocity(f32);
impl Component for Velocity {}
struct Health(f32);
impl Component for Health {}

#[derive(Default)]
struct Score(u32);
#[derive(Clone, Debug, PartialEq)]
struct Scored(u32);

fn movement(mut moving: Query<(Node, &Velocity)>, time: Res<Time>) {
    moving.each(|(node, v)| node.position.x += v.0 * time.delta);
}

fn scoring(mut score: ResMut<Score>, mut out: EventWriter<Scored>, healthy: Query<&Health>) {
    let alive = healthy.iter().filter(|h| h.0 > 0.0).count() as u32;
    score.0 += alive;
    out.send(Scored(alive));
}

fn listening(mut heard: EventReader<Scored>, mut log: ResMut<Vec<u32>>) {
    heard.read(|e| log.push(e.0));
}

/// The plain kind, ordered against the new kind by its function's name.
fn after_scoring(w: &mut World) {
    let last = *w.resource::<Vec<u32>>().last().unwrap_or(&0);
    w.resource_mut::<Vec<u32>>().push(last + 100);
}

fn world_with_things() -> App {
    let mut app = App::new();
    app.world.insert_resource(Score::default());
    app.world.insert_resource(Vec::<u32>::new());
    for i in 0..3 {
        let e = app.world.spawn(Object3D::group());
        app.world.insert(
            e,
            (Velocity(i as f32), Health(if i == 0 { 0.0 } else { 5.0 })),
        );
    }
    app
}

#[test]
fn parameter_systems_read_write_and_mix_with_world_systems() {
    let mut app = world_with_things();
    app.fixed_system(movement)
        .fixed_system(scoring)
        .fixed_system(System::new(listening).after(scoring))
        .fixed_system(System::new(after_scoring).after(listening));
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Score>().0, 4);
    assert_eq!(*app.world.resource::<Vec<u32>>(), [2, 102, 2, 102]);
    let xs: Vec<f32> = app
        .world
        .iter::<(NodeRef, &Velocity)>()
        .map(|(n, _)| n.position.x)
        .collect();
    assert!(
        xs.iter().any(|&x| (x - 2.0 * 2.0 / 60.0).abs() < 1e-5),
        "{xs:?}"
    );
}

#[test]
fn get_single_and_commands_work_from_parameters() {
    fn spawner(commands: Commands, health: Query<(Entity, &Health)>) {
        if let Some((id, _)) = health.iter().find(|(_, h)| h.0 == 0.0) {
            commands.despawn(id);
        }
    }
    fn healer(mut health: Query<(Entity, &mut Health)>, mut log: ResMut<Vec<u32>>) {
        let mut ids = Vec::new();
        health.each(|(id, _)| ids.push(id));
        for &id in &ids {
            health.get_mut(id, |(_, h)| h.0 += 1.0);
        }
        log.push(ids.len() as u32);
    }
    let mut app = world_with_things();
    app.system(spawner)
        .system(System::new(healer).after(spawner));
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.count::<&Health>(), 2, "the dead one went");
    assert!(app.world.iter::<&Health>().all(|h| h.0 >= 6.0));
}

fn message(f: impl FnOnce()) -> String {
    let e = catch_unwind(AssertUnwindSafe(f)).expect_err("refused");
    e.downcast_ref::<String>().cloned().unwrap_or_default()
}

#[test]
fn aliasing_parameters_are_refused_by_name() {
    fn two_views(_: Query<&mut Health>, _: Query<&Health>) {}
    fn two_scores(_: ResMut<Score>, _: Res<Score>) {}
    fn node_twice(_: Query<Node>, _: Query<(NodeRef, &Health)>) {}
    fn events_twice(_: EventReader<Scored>, _: EventWriter<Scored>) {}
    let t = message(|| {
        App::new().system(two_views);
    });
    assert!(t.contains("Health") && t.contains("two_views"), "{t}");
    let t = message(|| {
        App::new().system(two_scores);
    });
    assert!(t.contains("resource") && t.contains("Score"), "{t}");
    assert!(message(|| {
        App::new().system(node_twice);
    })
    .contains("Node"));
    assert!(message(|| {
        App::new().system(events_twice);
    })
    .contains("Scored"));
    // Reading twice is fine.
    fn two_reads(_: Query<&Health>, _: Query<(&Health, &Velocity)>, _: Res<Score>, _: Res<Score>) {}
    App::new().system(two_reads);
}

#[test]
fn a_missing_resource_parameter_says_what_to_add() {
    fn needs_physics(_: Res<Score>) {}
    let mut app = App::new();
    app.system(needs_physics);
    let t = message(|| app.frame(0.016));
    assert!(t.contains("no resource") && t.contains("Score"), "{t}");
}
