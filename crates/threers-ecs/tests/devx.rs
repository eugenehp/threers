//! What makes a game quicker to write: messages that say what to do, numbers
//! tuned while it runs, lines to see with, timers, an inspector that edits,
//! recordings that become tests, levels with collision, and a starting point.
#![recursion_limit = "512"]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use threers_ecs::prelude::*;
use threers_ecs::tuning::Retuned;

fn message(f: impl FnOnce()) -> String {
    let error = catch_unwind(AssertUnwindSafe(f)).expect_err("it should panic");
    error
        .downcast_ref::<String>()
        .cloned()
        .or(error.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("threers_devx_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---- messages ---------------------------------------------------------------

#[test]
fn a_missing_plugin_resource_names_the_plugin_and_what_there_is() {
    let app = App::new();
    let text = message(|| {
        app.world.resource::<Physics>();
    });
    assert!(text.contains("PhysicsPlugin::default()"), "{text}");
    assert!(
        text.contains("The world has: ") && text.contains("Input"),
        "{text}"
    );
}

#[test]
fn a_game_type_gets_the_general_advice() {
    struct Score;
    let text = message(|| {
        App::new().world.resource::<Score>();
    });
    assert!(text.contains("insert it first"), "{text}");
}

#[test]
fn a_lent_resource_says_it_is_lent() {
    struct Score;
    let mut app = App::new();
    app.world.insert_resource(Score);
    let text = message(|| {
        app.world.with_resource::<Score, _>(|w, _| {
            w.resource::<Score>();
        })
    });
    assert!(
        text.contains("lent out") && text.contains("with_resource::<Score>"),
        "{text}"
    );
}

#[test]
fn a_misspelt_action_lists_the_bound_ones() {
    let mut app = App::new();
    app.world
        .resource_mut::<Input>()
        .bind("jump", Key::Space)
        .bind_axis("steer", Key::KeyA, Key::KeyD);
    let text = message(|| {
        app.world.resource::<Input>().pressed("jmup");
    });
    assert!(text.contains("\"jmup\"") && text.contains("jump"), "{text}");
    let text = message(|| {
        app.world.resource::<Input>().axis("stear");
    });
    assert!(text.contains("steer"), "{text}");
}

// ---- tuning -----------------------------------------------------------------

#[derive(Default, Debug, PartialEq)]
struct Handling {
    grip: f32,
    drift: bool,
}
reflect!(Handling { grip, drift });

fn wait(app: &mut App, seconds: f64) {
    let start = std::time::Instant::now();
    while start.elapsed().as_secs_f64() < seconds {
        app.frame(0.02);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn a_missing_tuning_file_is_written_with_the_defaults() {
    let path = temp("defaults").join("tuning/handling.json");
    let mut app = App::new();
    app.tune::<Handling>(&path);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("\"grip\": 0") && text.contains("\"drift\": false"),
        "{text}"
    );
}

#[test]
fn editing_the_file_retunes_the_running_game() {
    let path = temp("live").join("handling.json");
    std::fs::write(&path, r#"{"grip": 1.5}"#).unwrap();
    let mut app = App::new();
    app.tune::<Handling>(&path);
    assert_eq!(app.world.resource::<Handling>().grip, 1.5);
    wait(&mut app, 0.3);
    std::fs::write(&path, r#"{"grip": 2.5, "drift": true}"#).unwrap();
    wait(&mut app, 0.6);
    assert_eq!(
        *app.world.resource::<Handling>(),
        Handling {
            grip: 2.5,
            drift: true
        }
    );
    let news = app.world.events::<Retuned>();
    assert!(
        news.iter()
            .any(|r| r.name == "Handling" && r.problem.is_none()),
        "{news:?}"
    );
}

#[test]
fn a_mistake_in_the_file_keeps_the_rest_and_says_what_it_was() {
    let path = temp("typo").join("handling.json");
    std::fs::write(&path, r#"{"grip": 2.0}"#).unwrap();
    let mut app = App::new();
    app.tune::<Handling>(&path);
    wait(&mut app, 0.3);
    std::fs::write(&path, r#"{"grip": 3.0, "gripp": 9, "drift": "yes"}"#).unwrap();
    wait(&mut app, 0.6);
    assert_eq!(app.world.resource::<Handling>().grip, 3.0);
    let problem = app
        .world
        .events::<Retuned>()
        .into_iter()
        .find_map(|r| r.problem)
        .unwrap();
    assert!(
        problem.contains("no field gripp") && problem.contains("drift has the wrong kind"),
        "{problem}"
    );
    // Half-saved JSON is reported, and nothing changes.
    std::fs::write(&path, r#"{"grip": 4.0,"#).unwrap();
    wait(&mut app, 0.6);
    assert_eq!(app.world.resource::<Handling>().grip, 3.0);
}

// ---- gizmos -----------------------------------------------------------------

fn gizmo_lines(app: &App) -> usize {
    app.world
        .scene
        .arena
        .nodes
        .values()
        .filter(|n| n.name == "gizmos" && n.visible)
        .map(|n| match &n.kind {
            ObjectKind::LineSegments(l) => {
                l.geometry.get_attribute("position").unwrap().count() / 2
            }
            _ => 0,
        })
        .sum()
}

#[test]
fn frame_lines_last_a_frame_and_each_colour_gets_one_node() {
    fn draw(w: &mut World) {
        if w.resource::<Time>().frame == 0 {
            let g = w.resource_mut::<Gizmos>();
            g.line(Vector3::ZERO, Vector3::UP, Color::from_hex(0xff0000));
            g.cube(Vector3::ZERO, Vector3::ONE, Color::from_hex(0x00ff00));
        }
    }
    let mut app = App::new();
    app.system(draw);
    app.frame(1.0 / 60.0);
    assert_eq!(gizmo_lines(&app), 1 + 12);
    let nodes = app
        .world
        .scene
        .arena
        .nodes
        .values()
        .filter(|n| n.name == "gizmos")
        .count();
    assert_eq!(nodes, 2);
    app.frame(1.0 / 60.0);
    assert_eq!(gizmo_lines(&app), 0);
}

/// A frame with no fixed step must still show what the last step drew.
#[test]
fn fixed_lines_stay_up_until_the_next_step() {
    fn draw(w: &mut World) {
        w.resource_mut::<Gizmos>()
            .point(Vector3::ZERO, 1.0, Color::WHITE);
    }
    let mut app = App::new();
    app.fixed_system(draw);
    app.frame(1.0 / 60.0);
    assert_eq!(gizmo_lines(&app), 3);
    app.frame(0.001); // too short for a step
    assert_eq!(gizmo_lines(&app), 3);
    app.frame(1.0 / 60.0);
    assert_eq!(gizmo_lines(&app), 3, "a new step replaces, not adds");
}

#[test]
fn turned_off_gizmos_draw_nothing() {
    fn draw(w: &mut World) {
        w.resource_mut::<Gizmos>()
            .sphere(Vector3::ZERO, 1.0, Color::WHITE);
    }
    let mut app = App::new();
    app.world.resource_mut::<Gizmos>().enabled = false;
    app.system(draw);
    app.frame(1.0 / 60.0);
    assert_eq!(gizmo_lines(&app), 0);
}

// ---- timers -----------------------------------------------------------------

#[test]
fn a_repeating_timer_keeps_time_and_counts_every_firing() {
    let mut t = Timer::repeating(0.1);
    assert!(!t.tick(0.05));
    assert!(t.tick(0.3));
    assert_eq!(t.times_fired(), 3);
    assert!((t.fraction() - 0.5).abs() < 1e-4);
    assert!(!t.finished());
}

#[test]
fn delays_run_in_order_on_game_time_and_wait_out_a_pause() {
    let mut app = App::new();
    app.world.insert_resource(Vec::<u32>::new());
    app.world
        .after(0.3, |w| w.resource_mut::<Vec<u32>>().push(2));
    app.world
        .after(0.1, |w| w.resource_mut::<Vec<u32>>().push(1));
    app.frame(0.05);
    assert!(app.world.resource::<Vec<u32>>().is_empty());
    app.world.resource_mut::<Time>().paused = true;
    for _ in 0..30 {
        app.frame(0.05);
    }
    assert!(
        app.world.resource::<Vec<u32>>().is_empty(),
        "a pause stops the clock"
    );
    app.world.resource_mut::<Time>().paused = false;
    for _ in 0..10 {
        app.frame(0.05);
    }
    assert_eq!(*app.world.resource::<Vec<u32>>(), [1, 2]);
}

#[test]
fn a_delay_can_set_up_another() {
    let mut app = App::new();
    app.world.insert_resource(0u32);
    app.world.after(0.1, |w| {
        *w.resource_mut::<u32>() += 1;
        w.after(0.1, |w| *w.resource_mut::<u32>() += 10);
    });
    for _ in 0..20 {
        app.frame(0.05);
    }
    assert_eq!(*app.world.resource::<u32>(), 11);
}

#[test]
fn a_tween_arrives_reports_and_leaves() {
    let mut app = App::new();
    let door = app
        .world
        .object(Object3D::group())
        .at(Vector3::new(1.0, 0.0, 0.0))
        .id();
    app.world.insert(
        door,
        Tween::to_position(Vector3::new(1.0, 4.0, 0.0), 0.5).easing(Easing::Linear),
    );
    for _ in 0..15 {
        app.frame(1.0 / 60.0);
    }
    let y = app.world.node(door).unwrap().position.y;
    assert!((y - 2.0).abs() < 0.15, "half way at {y}");
    for _ in 0..30 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(
        app.world.node(door).unwrap().position,
        Vector3::new(1.0, 4.0, 0.0)
    );
    assert!(!app.world.has::<Tween>(door));
    assert_eq!(
        app.world.events::<TweenFinished>(),
        vec![TweenFinished(door)]
    );
}

// ---- inspector --------------------------------------------------------------

#[derive(Default)]
struct Health {
    hp: f32,
    alive: bool,
}
impl Component for Health {}
reflect!(Health { hp, alive });

#[test]
fn set_field_changes_one_field_by_name() {
    let mut w = World::new();
    w.register::<Health>();
    let id = w
        .object(Object3D::group())
        .with(Health {
            hp: 3.0,
            alive: true,
        })
        .id();
    assert!(w.set_field(id, "Health", "hp", &threers_ecs::FieldValue::Number(7.0)));
    assert!(!w.set_field(id, "Health", "mana", &threers_ecs::FieldValue::Number(7.0)));
    assert!(!w.set_field(id, "Nothing", "hp", &threers_ecs::FieldValue::Number(7.0)));
    assert_eq!(w.get::<Health>(id).unwrap().hp, 7.0);
    assert!(w.get::<Health>(id).unwrap().alive);
}

#[test]
fn the_dev_panel_edits_the_selected_entity_from_the_keys() {
    let mut game = TestGame::new(|app| {
        app.plugin(threers_ecs::dev::DevTools);
        app.world.register::<Health>();
        app.world.object(Object3D::group()).with(Health {
            hp: 40.0,
            alive: true,
        });
    });
    game.tap(Key::F7).frames(1).tap(Key::F10).frames(1);
    let (health,) = game.world().single::<(&Health,)>().unwrap();
    assert_eq!(health.hp, 41.0, "a step of a tenth of 40's leading place");
    game.tap(Key::F8).frames(1).tap(Key::F9).frames(1);
    assert!(!game.world().single::<(&Health,)>().unwrap().0.alive);
}

// ---- recordings -------------------------------------------------------------

#[test]
fn a_recording_survives_text_and_keeps_a_tap_inside_one_frame() {
    let mut app = App::new();
    app.start_recording();
    let input = app.world.resource_mut::<Input>();
    input.press(Key::Space);
    input.release(Key::Space);
    input.set_analog(PadAxis::LeftX, -0.25);
    input.move_mouse(Vector2::new(10.0, 20.0));
    input.press(PadButton::South);
    input.touch_start(3, Vector2::new(1.5, 2.5));
    input.touch_end(3);
    input.scroll += 2.0;
    app.frame(0.016);
    app.frame(0.033);
    let take = app.stop_recording().unwrap();
    assert_eq!(take.frames.len(), 2);
    assert_eq!(take.frames[0].1.len(), 8);
    let back = Recording::from_text(&take.to_text()).unwrap();
    assert_eq!(back, take);
    assert!(Recording::from_text("hello").is_err());
    let error = Recording::from_text("threers-recording 1\n0.016 +key:Nope\n").unwrap_err();
    assert!(
        error.contains("line 2") && error.contains("key:Nope"),
        "{error}"
    );
}

// ---- levels -----------------------------------------------------------------

/// A level as an authoring tool would write one: a mesh, and custom properties.
fn level_glb(extras: &str) -> Vec<u8> {
    let mut scene = Scene::new();
    let floor = scene.add_mesh(
        BoxGeometry::new(20.0, 1.0, 20.0),
        StandardMaterial::new(Color::WHITE).into(),
    );
    scene.get_mut(floor).unwrap().name = "floor".into();
    let glb = threers::loaders::scene_to_glb(&scene.arena, &[floor]);
    // Into the JSON chunk, on the one node.
    let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let json = std::str::from_utf8(&glb[20..20 + json_len])
        .unwrap()
        .trim_end();
    let patched = json.replacen(
        "\"name\":\"floor\"",
        &format!("\"name\":\"floor\",\"extras\":{extras}"),
        1,
    );
    assert_ne!(patched, json, "the writer's node layout changed: {json}");
    let mut chunk = patched.into_bytes();
    while chunk.len() % 4 != 0 {
        chunk.push(b' ');
    }
    let rest = &glb[20 + json_len..];
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((12 + 8 + chunk.len() + rest.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&chunk);
    out.extend_from_slice(rest);
    out
}

#[test]
fn a_solid_level_holds_a_falling_box_up() {
    use threers_ecs::assets::{AssetsPlugin, SceneReady};
    let dir = temp("level");
    std::fs::write(
        dir.join("level.glb"),
        level_glb(r#"{"solid":true,"Health":{"hp":3}}"#),
    )
    .unwrap();
    let mut game = TestGame::new(|app| {
        app.plugin(AssetsPlugin::new(&dir))
            .plugin(PhysicsPlugin::default());
        app.world.register::<Health>();
        app.world.spawn_scene("level.glb");
    });
    let start = std::time::Instant::now();
    while game.world().events::<SceneReady>().is_empty() {
        assert!(start.elapsed().as_secs() < 10, "the level never loaded");
        game.frames(1);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let floor = game
        .world()
        .iter::<(Entity, NodeRef)>()
        .find(|(_, n)| n.name == "floor")
        .unwrap()
        .0;
    assert!(game.world().has::<Body>(floor), "solid gave it a body");
    assert_eq!(game.world().get::<Health>(floor).unwrap().hp, 3.0);
    let crate_ = game
        .world()
        .object(Object3D::group())
        .at(Vector3::new(0.0, 4.0, 0.0))
        .body(RigidBody::dynamic().shape(Shape::cuboid(0.5, 0.5, 0.5)))
        .id();
    game.seconds(2.0);
    let y = game.world().node(crate_).unwrap().position.y;
    assert!(
        (0.8..1.3).contains(&y),
        "rests on the floor's top at 0.5: {y}"
    );
}

// ---- the starting point ----------------------------------------------------

#[path = "../examples/minimal.rs"]
#[allow(dead_code)]
mod minimal;

#[test]
fn the_template_is_the_minimal_example() {
    let example = include_str!("../examples/minimal.rs");
    let template = include_str!("../../../templates/game/src/main.rs");
    assert_eq!(
        example, template,
        "copy examples/minimal.rs over templates/game/src/main.rs"
    );
}

#[test]
fn the_minimal_game_drives() {
    let dir = temp("minimal");
    let here = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let mut game = TestGame::new(minimal::build);
    std::env::set_current_dir(here).unwrap();
    game.hold(Key::KeyW, 60);
    let (node, _) = game
        .world()
        .single::<(NodeRef, With<CameraTarget>)>()
        .unwrap();
    assert!(node.position.z < -3.0, "{:?}", node.position);
}
