//! Events, states, pausing and change tracking.

use threers::core::Object3D;
use threers_ecs::prelude::*;

#[derive(Clone, Debug, PartialEq)]
struct Ping(u32);

#[derive(Default)]
struct Got(Vec<u32>);

fn ping(w: &mut World) {
    let n = w.resource::<Time>().frame as u32;
    w.send(Ping(n));
}

fn hear(w: &mut World) {
    let pings = w.events::<Ping>();
    w.resource_mut::<Got>().0.extend(pings.iter().map(|p| p.0));
}

fn got(app: &App) -> &[u32] {
    &app.world.resource::<Got>().0
}

#[test]
fn every_reader_sees_an_event_once() {
    fn second(w: &mut World) {
        let n = w.events::<Ping>().len() as u32;
        *w.resource_mut::<u32>() += n;
    }
    let mut app = App::new();
    app.resource(Got::default()).resource(0_u32);
    app.system(ping).system(hear).system(second);
    for _ in 0..5 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(got(&app), [0, 1, 2, 3, 4]);
    assert_eq!(*app.world.resource::<u32>(), 5);
}

#[test]
fn a_reader_earlier_in_the_frame_gets_it_next_time() {
    let mut app = App::new();
    app.resource(Got::default());
    app.system(hear).system(ping);
    app.frame(1.0 / 60.0);
    assert!(got(&app).is_empty());
    app.frame(1.0 / 60.0);
    assert_eq!(got(&app), [0]);
}

#[test]
fn fixed_and_variable_systems_lose_and_repeat_nothing() {
    for fps in [15.0, 60.0, 240.0] {
        // Sent once a frame, read at the fixed rate.
        let mut app = App::new();
        app.resource(Got::default());
        app.system(ping);
        app.fixed_system(hear);
        for _ in 0..240 {
            app.frame(1.0 / fps);
        }
        let mut seen = got(&app).to_vec();
        seen.dedup();
        assert_eq!(seen.len(), got(&app).len(), "repeated at {fps} fps");
        assert!(seen.len() >= 235, "lost at {fps} fps: {}", seen.len());
    }
}

#[test]
fn nobody_listening_does_not_grow_without_bound() {
    let mut app = App::new();
    app.system(ping);
    for _ in 0..(EVENT_FRAMES * 3) {
        app.frame(1.0 / 60.0);
    }
    assert!(app.world.events::<Ping>().len() as u32 <= EVENT_FRAMES);
}

#[test]
fn code_outside_a_system_reads_too() {
    let mut w = World::new();
    w.send(Ping(1));
    assert_eq!(w.events::<Ping>(), [Ping(1)]);
    assert!(w.events::<Ping>().is_empty());
    assert!(w.events::<u8>().is_empty());
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Screen {
    Menu,
    Playing,
    Over,
}
impl GameState for Screen {}

#[derive(Default)]
struct Trail(Vec<&'static str>);

fn trail(w: &mut World, step: &'static str) {
    w.resource_mut::<Trail>().0.push(step);
}

fn screens() -> App {
    let mut app = App::new();
    app.resource(Trail::default());
    app.state(Screen::Menu);
    app.on_enter(Screen::Menu, |w: &mut World| trail(w, "enter menu"));
    app.on_exit(Screen::Menu, |w: &mut World| trail(w, "exit menu"));
    app.on_enter(Screen::Playing, |w: &mut World| trail(w, "enter playing"));
    app.system(System::new(|w: &mut World| trail(w, "menu")).in_state(Screen::Menu));
    app.system(System::new(|w: &mut World| trail(w, "play")).in_state(Screen::Playing));
    app
}

#[test]
fn states_gate_systems_and_run_enter_and_exit_once() {
    let mut app = screens();
    app.frame(1.0 / 60.0);
    app.world.set_state(Screen::Playing);
    // Still the menu until the next frame starts.
    assert_eq!(app.world.state::<Screen>(), Screen::Menu);
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    assert_eq!(
        app.world.resource::<Trail>().0,
        [
            "enter menu",
            "menu",
            "exit menu",
            "enter playing",
            "play",
            "play"
        ]
    );
}

#[test]
fn setting_the_current_state_is_not_a_transition() {
    let mut app = screens();
    app.frame(1.0 / 60.0);
    app.world.set_state(Screen::Menu);
    app.frame(1.0 / 60.0);
    assert_eq!(
        app.world.resource::<Trail>().0,
        ["enter menu", "menu", "menu"]
    );
}

#[test]
fn an_enter_system_can_move_on_at_once() {
    let mut app = screens();
    app.on_enter(Screen::Over, |w: &mut World| w.set_state(Screen::Playing));
    app.frame(1.0 / 60.0);
    app.world.set_state(Screen::Over);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.state::<Screen>(), Screen::Playing);
}

#[test]
#[should_panic(expected = "call App::state")]
fn a_state_nobody_added_says_so() {
    World::new().state::<Screen>();
}

#[test]
fn run_if_skips_without_touching_anything_else() {
    let mut app = App::new();
    app.resource(Got::default()).resource(false);
    app.system(ping);
    app.system(System::new(hear).run_if(|w| *w.resource::<bool>()));
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    assert!(got(&app).is_empty());
    app.world.insert_resource(true);
    app.frame(1.0 / 60.0);
    // It catches up on what it slept through.
    assert_eq!(got(&app), [0, 1, 2]);
}

#[derive(Default)]
struct Steps(u32);

fn count_step(w: &mut World) {
    w.resource_mut::<Steps>().0 += 1;
}

fn counting() -> App {
    let mut app = App::new();
    app.resource(Steps::default());
    app.fixed_system(count_step);
    app
}

#[test]
fn a_pause_stops_game_time_but_not_the_frame() {
    let mut app = counting();
    app.frame(1.0 / 60.0);
    app.world.resource_mut::<Time>().paused = true;
    let elapsed = app.world.resource::<Time>().elapsed;
    for _ in 0..30 {
        app.frame(1.0 / 60.0);
    }
    let time = *app.world.resource::<Time>();
    assert_eq!(app.world.resource::<Steps>().0, 1);
    assert_eq!(time.delta, 0.0);
    assert_eq!(time.elapsed, elapsed);
    assert!(time.real_delta > 0.0);
    assert_eq!(time.frame, 31);

    app.world.resource_mut::<Time>().paused = false;
    app.frame(1.0 / 60.0);
    // No burst of steps for the time spent paused.
    assert_eq!(app.world.resource::<Steps>().0, 2);
}

#[test]
fn single_steps_run_one_per_frame_while_paused() {
    let mut app = counting();
    app.world.resource_mut::<Time>().paused = true;
    app.world.resource_mut::<Time>().step_once = 2;
    for _ in 0..5 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(app.world.resource::<Steps>().0, 2);
    assert_eq!(app.world.resource::<Time>().step_once, 0);
}

#[test]
fn time_scale_slows_the_steps() {
    let mut app = counting();
    app.world.resource_mut::<Time>().scale = 0.5;
    for _ in 0..60 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(app.world.resource::<Steps>().0, 30);
}

#[test]
fn a_key_tapped_in_a_pause_does_not_fire_on_resume() {
    let mut app = App::new();
    app.world.resource_mut::<Input>().bind("jump", Key::Space);
    app.resource(Steps::default());
    app.fixed_system(|w: &mut World| {
        if w.resource::<Input>().just_pressed("jump") {
            w.resource_mut::<Steps>().0 += 1;
        }
    });
    app.world.resource_mut::<Time>().paused = true;
    let input = app.world.resource_mut::<Input>();
    input.press(Key::Space);
    input.release(Key::Space);
    app.frame(1.0 / 60.0);
    app.world.resource_mut::<Time>().paused = false;
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Steps>().0, 0);
}

#[derive(Debug, PartialEq)]
struct Health(i32);
impl Component for Health {}

#[derive(Default)]
struct Noticed {
    added: Vec<Entity>,
    changed: Vec<Entity>,
}

fn notice(w: &mut World) {
    let added = w.entities::<Added<Health>>();
    let changed = w.entities::<Changed<Health>>();
    let n = w.resource_mut::<Noticed>();
    n.added = added;
    n.changed = changed;
}

fn tracked() -> (App, Entity) {
    let mut app = App::new();
    app.resource(Noticed::default());
    app.system(notice);
    let id = app.world.spawn(Object3D::group());
    (app, id)
}

#[test]
fn added_and_changed_fire_once_per_change() {
    let (mut app, id) = tracked();
    app.world.insert(id, Health(1));
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Noticed>().added, [id]);
    assert_eq!(app.world.resource::<Noticed>().changed, [id]);

    app.frame(1.0 / 60.0);
    assert!(app.world.resource::<Noticed>().added.is_empty());
    assert!(app.world.resource::<Noticed>().changed.is_empty());

    app.world.get_mut::<Health>(id).unwrap().0 = 2;
    app.frame(1.0 / 60.0);
    assert!(app.world.resource::<Noticed>().added.is_empty());
    assert_eq!(app.world.resource::<Noticed>().changed, [id]);
}

#[test]
fn a_mutable_query_counts_and_a_shared_one_does_not() {
    let (mut app, id) = tracked();
    app.world.insert(id, Health(1));
    app.frame(1.0 / 60.0);

    app.world.each::<&Health>(|_| {});
    app.frame(1.0 / 60.0);
    assert!(app.world.resource::<Noticed>().changed.is_empty());

    app.world.each::<&mut Health>(|_| {});
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Noticed>().changed, [id]);
}

#[test]
fn replacing_a_component_is_a_change_not_an_addition() {
    let (mut app, id) = tracked();
    app.world.insert(id, Health(1));
    app.frame(1.0 / 60.0);
    app.world.insert(id, Health(2));
    app.frame(1.0 / 60.0);
    assert!(app.world.resource::<Noticed>().added.is_empty());
    assert_eq!(app.world.resource::<Noticed>().changed, [id]);
}

#[test]
fn a_system_does_not_see_its_own_changes_next_time() {
    fn heal(w: &mut World) {
        let changed = w.count::<Changed<Health>>() as u32;
        *w.resource_mut::<u32>() += changed;
        w.each::<&mut Health>(|health| health.0 += 1);
    }
    let mut app = App::new();
    app.resource(0_u32);
    app.system(heal);
    let id = app.world.spawn(Object3D::group());
    app.world.insert(id, Health(1));
    for _ in 0..5 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(*app.world.resource::<u32>(), 1);
}

#[test]
fn tracking_outside_a_system_starts_from_clear_trackers() {
    let mut w = World::new();
    let id = w.spawn(Object3D::group());
    w.insert(id, Health(1));
    assert_eq!(w.count::<Added<Health>>(), 1);
    w.clear_trackers();
    assert_eq!(w.count::<Added<Health>>(), 0);
}

#[test]
fn read_only_queries_need_only_a_shared_world() {
    let mut w = World::new();
    let id = w.spawn(Object3D::group());
    w.scene.get_mut(id).unwrap().position.x = 3.0;
    w.insert(id, Health(5));
    let shared = &w;
    let mut seen = Vec::new();
    shared.each_ref::<(Entity, NodeRef, &Health, Option<&Health2>)>(|(e, node, h, other)| {
        seen.push((e, node.position.x, h.0, other.is_some()));
    });
    assert_eq!(seen, [(id, 3.0, 5, false)]);
    assert_eq!(shared.count::<NodeRef>(), 2);
    assert_eq!(shared.count::<(NodeRef, With<Health>)>(), 1);
}

struct Health2;
impl Component for Health2 {}

#[test]
fn queries_can_nest_when_they_only_read() {
    let mut w = World::new();
    for i in 0..3 {
        let id = w.spawn(Object3D::group());
        w.insert(id, Health(i));
    }
    let mut pairs = 0;
    w.each_ref::<&Health>(|a| {
        w.each_ref::<&Health>(|b| {
            pairs += (a.0 != b.0) as u32;
        });
    });
    assert_eq!(pairs, 6);
}

#[test]
fn despawn_many_handles_overlap_and_the_dead() {
    let mut w = World::new();
    let parent = w.spawn(Object3D::group());
    let child = w.spawn_under(parent, Object3D::group());
    let other = w.spawn(Object3D::group());
    let dead = w.spawn(Object3D::group());
    w.despawn(dead);
    let root = w.scene.root;

    assert_eq!(
        w.despawn_many(&[child, parent, other, other, dead, root]),
        3
    );

    assert!(!w.is_alive(parent) && !w.is_alive(child) && !w.is_alive(other));
    assert!(w.is_alive(root));
    assert!(w.scene.get(root).unwrap().children.is_empty());
}

#[test]
fn a_later_despawn_waits_its_turn() {
    let mut w = World::new();
    let first = w.spawn(Object3D::group());
    let second = w.spawn(Object3D::group());
    let commands = w.commands();
    commands.despawn(first);
    commands.add(move |w| {
        assert!(!w.is_alive(first));
        assert!(w.is_alive(second), "ran ahead of the queue");
    });
    commands.despawn(second);
    w.flush();
    assert!(!w.is_alive(second));
}

#[test]
fn queued_despawns_keep_their_place_among_other_commands() {
    let mut w = World::new();
    let a = w.spawn(Object3D::group());
    let b = w.spawn(Object3D::group());
    let c = w.spawn(Object3D::group());
    let commands = w.commands();
    commands.insert(a, Health(1));
    commands.despawn(a);
    commands.despawn(b);
    commands.insert(a, Health(2));
    commands.insert(c, Health(3));
    commands.despawn(c);
    w.flush();
    assert_eq!(w.count::<&Health>(), 0);
    assert!(!w.is_alive(a) && !w.is_alive(b) && !w.is_alive(c));
}
