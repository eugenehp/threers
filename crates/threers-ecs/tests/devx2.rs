//! The second round of tools: profiling, surviving a panic, and the rest.
#![recursion_limit = "512"]

use threers_ecs::prelude::*;
use threers_ecs::profile::{Failures, Profile, SystemFailed};

fn slow(_: &mut World) {
    let start = std::time::Instant::now();
    while start.elapsed().as_micros() < 800 {}
}
fn quick(w: &mut World) {
    *w.resource_mut::<u32>() += 1;
}

#[test]
fn every_system_is_timed_by_name() {
    let mut app = App::new();
    app.world.insert_resource(0u32);
    app.system(slow).system(quick);
    for _ in 0..10 {
        app.frame(1.0 / 60.0);
    }
    let profile = app.world.resource::<Profile>();
    let (first, time) = profile.slowest()[0];
    assert_eq!(threers_ecs::profile::short(first), "slow");
    assert!(time.ms > 0.5 && time.runs == 10, "{time:?}");
    assert!(profile.get("quick").unwrap().ms < 0.3);
}

fn breaks_on_the_third(w: &mut World) {
    *w.resource_mut::<u32>() += 1;
    if *w.resource::<u32>() == 3 {
        panic!("third time");
    }
}

#[test]
fn a_panicking_system_is_switched_off_and_the_rest_go_on() {
    let mut app = App::new();
    app.catch_panics(true);
    app.world.insert_resource(0u32);
    app.world.insert_resource(Vec::<u32>::new());
    app.system(breaks_on_the_third);
    app.system(|w: &mut World| w.resource_mut::<Vec<u32>>().push(1));
    for _ in 0..6 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(
        *app.world.resource::<u32>(),
        3,
        "not run again after it failed"
    );
    assert_eq!(
        app.world.resource::<Vec<u32>>().len(),
        6,
        "the other ran every frame"
    );
    let failures = &app.world.resource::<Failures>().0;
    assert_eq!(failures.len(), 1);
    assert!(
        failures[0].system.ends_with("breaks_on_the_third") && failures[0].message == "third time"
    );
    let _: Vec<SystemFailed> = app.world.events();
}

struct Score(u32);

/// The resource it had borrowed is back afterwards.
#[test]
fn a_panic_inside_with_resource_gives_the_resource_back() {
    let mut app = App::new();
    app.catch_panics(true);
    app.world.insert_resource(Score(5));
    app.system(|w: &mut World| {
        w.with_resource::<Score, _>(|_, score| {
            score.0 += 1;
            panic!("mid-borrow");
        });
    });
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Score>().0, 6);
    // And it can be lent again.
    app.world.with_resource::<Score, _>(|_, s| s.0 += 1);
    assert_eq!(app.world.resource::<Score>().0, 7);
}

#[test]
#[should_panic(expected = "third time")]
fn without_catching_a_panic_is_a_panic() {
    let mut app = App::new();
    app.world.insert_resource(0u32);
    app.system(breaks_on_the_third);
    for _ in 0..4 {
        app.frame(1.0 / 60.0);
    }
}

/// A picture test that fails says where to look: the frame it got and a diff.
#[test]
fn a_failing_picture_test_writes_what_it_saw_and_where_it_differs() {
    let dir = std::env::temp_dir().join(format!("threers_frames_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let reference = dir.join("scene.png");
    let build = |colour: u32| {
        move |app: &mut App| {
            app.world.insert_resource(MainCamera(PerspectiveCamera::new(
                50.0,
                16.0 / 9.0,
                0.1,
                100.0,
            )));
            let camera = &mut app.world.resource_mut::<MainCamera>().0;
            camera.position = Vector3::new(0.0, 0.0, 5.0);
            camera.look_at(Vector3::ZERO);
            app.world.scene.background = Color::from_hex(0x000000);
            app.world.mesh(
                BoxGeometry::new(2.0, 2.0, 2.0),
                BasicMaterial::new(Color::from_hex(colour)),
            );
        }
    };
    let mut first = TestGame::new(build(0xff0000));
    let rgba = first.render(320, 180);
    if rgba.is_empty() {
        return;
    }
    std::fs::write(&reference, threers::encode_png(320, 180, &rgba)).unwrap();
    first.assert_frame_matches(&reference, 1.0);

    let mut second = TestGame::new(build(0x00ff00));
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        second.assert_frame_matches(&reference, 1.0)
    }))
    .expect_err("a green box is not a red one");
    let text = text.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(
        text.contains("scene.actual.png") && text.contains("scene.diff.png"),
        "{text}"
    );
    let diff = threers::decode_png(&std::fs::read(dir.join("scene.diff.png")).unwrap()).unwrap();
    let centre = ((90 * 320 + 160) * 4) as usize;
    assert!(
        diff.rgba[centre] > 200 && diff.rgba[centre + 1] == 0,
        "the box is marked changed"
    );
    assert!(
        diff.rgba[0] == diff.rgba[1],
        "the corner, unchanged, is grey"
    );
}

/// Not `Clone`: read by reference, each system once, as `events` reads.
struct Big(Vec<u8>);

#[test]
fn events_are_lent_once_to_each_reader() {
    fn reader(w: &mut World) {
        let mut seen = 0;
        w.read_events::<Big>(|big| seen += big.0.len());
        *w.resource_mut::<usize>() += seen;
    }
    let mut app = App::new();
    app.world.insert_resource(0usize);
    app.system(reader);
    app.world.send(Big(vec![0; 1000]));
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    assert_eq!(*app.world.resource::<usize>(), 1000, "once, not each frame");
}

/// A body shaped like what is drawn, fitted to a group of meshes.
#[test]
fn a_body_fits_the_meshes_it_is_drawn_with() {
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default());
    let w = &mut app.world;
    w.mesh(
        BoxGeometry::new(100.0, 1.0, 100.0),
        StandardMaterial::new(Color::WHITE),
    )
    .at(Vector3::new(0.0, -0.5, 0.0))
    .solid_as(ColliderFit::Box);
    // A long bar, drawn as a child, its middle off the parent's origin.
    let parent = w
        .object(Object3D::group())
        .at(Vector3::new(0.0, 5.0, 0.0))
        .id();
    let bar = w.spawn_under(
        parent,
        Object3D::mesh(Mesh::new(
            BoxGeometry::new(6.0, 0.5, 0.5),
            StandardMaterial::new(Color::WHITE).into(),
        )),
    );
    w.node_mut(bar).unwrap().position = Vector3::new(2.0, 0.0, 0.0);
    let parent_body = w
        .entity(parent)
        .body_from_mesh(RigidBody::dynamic().mass(10.0), ColliderFit::Box)
        .id();
    for _ in 0..180 {
        app.frame(1.0 / 60.0);
    }
    // It lies on the floor at the bar's half height, not hanging off a point.
    let y = app.world.node(parent_body).unwrap().position.y;
    assert!((y - 0.25).abs() < 0.15, "resting at {y}");
    // And the hit box reaches the bar's far end, four units out.
    let physics = app.world.resource::<Physics>();
    let ray = threers::math::Ray::new(Vector3::new(4.8, 3.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
    let hit = physics.raycast(&ray, 10.0, QueryFilter::default().only_dynamic());
    assert!(hit.is_some(), "the bar's end is solid");
}

#[derive(Default)]
struct Crate {
    weight: f32,
}
impl Component for Crate {}
reflect!(Crate { weight });

fn looked_at() -> TestGame {
    TestGame::new(|app| {
        app.plugin(threers_ecs::dev::DevTools);
        app.world.insert_resource(Hud::new(320, 180));
        let mut camera = PerspectiveCamera::new(50.0, 320.0 / 180.0, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.0, 10.0);
        camera.look_at(Vector3::ZERO);
        app.world.insert_resource(MainCamera(camera));
        app.world.register::<Crate>();
        // A crate made of a body and a lid: clicking the lid picks the crate.
        let id = app
            .world
            .object(Object3D::group())
            .named("crate")
            .with(Crate { weight: 3.0 })
            .id();
        let lid = app.world.spawn_under(
            id,
            Object3D::mesh(Mesh::new(
                BoxGeometry::new(2.0, 2.0, 2.0),
                BasicMaterial::new(Color::WHITE).into(),
            )),
        );
        let _ = lid;
        app.world
            .mesh(
                BoxGeometry::new(1.0, 1.0, 1.0),
                BasicMaterial::new(Color::WHITE),
            )
            .at(Vector3::new(6.0, 0.0, 0.0));
    })
}

#[test]
fn clicking_with_the_panel_up_selects_what_is_under_the_pointer() {
    let mut game = looked_at();
    game.tap(Key::F3).frames(1);
    game.input().move_mouse(Vector2::new(160.0, 90.0));
    game.tap(threers_ecs::MouseButton::Left).frames(1);
    let selected = game.resource::<threers_ecs::dev::DevState>().selected;
    let crate_id = game
        .world()
        .iter::<(Entity, With<Crate>)>()
        .next()
        .unwrap()
        .0;
    assert_eq!(selected, Some(crate_id), "the lid's crate");
    // Over nothing, the selection stays.
    game.input().move_mouse(Vector2::new(5.0, 5.0));
    game.tap(threers_ecs::MouseButton::Left).frames(1);
    assert_eq!(
        game.resource::<threers_ecs::dev::DevState>().selected,
        Some(crate_id)
    );
}

#[test]
fn the_free_camera_flies_while_the_game_waits_and_hands_back() {
    let mut game = looked_at();
    let before = game.resource::<MainCamera>().0.position;
    game.tap(Key::F2).frames(1);
    assert!(game.resource::<Time>().paused);
    game.hold(Key::KeyW, 30);
    let flown = game.resource::<MainCamera>().0.position;
    assert!(flown.z < before.z - 2.0, "flew forward: {flown:?}");
    game.tap(Key::F2).frames(1);
    assert_eq!(game.resource::<MainCamera>().0.position, before);
    assert!(!game.resource::<Time>().paused);
}

#[test]
fn with_colliders_shown_each_car_draws_its_tyres_and_speed() {
    let mut game = TestGame::new(|app| {
        app.plugin(PhysicsPlugin::default())
            .plugin(threers_ecs::dev::DevTools);
        let w = &mut app.world;
        w.mesh(
            BoxGeometry::new(100.0, 1.0, 100.0),
            StandardMaterial::new(Color::WHITE),
        )
        .at(Vector3::new(0.0, -0.5, 0.0))
        .solid();
        w.object(Object3D::group())
            .at(Vector3::new(0.0, 1.0, 0.0))
            .car(CarSpec::default());
    });
    game.seconds(1.0);
    let lines = |game: &mut TestGame| {
        game.world()
            .iter::<NodeRef>()
            .filter(|n| n.name == "gizmos" && n.visible)
            .map(|n| match &n.kind {
                ObjectKind::LineSegments(l) => {
                    l.geometry.get_attribute("position").unwrap().count() / 2
                }
                _ => 0,
            })
            .sum::<usize>()
    };
    assert_eq!(lines(&mut game), 0);
    game.tap(Key::F4).frames(1);
    // Four tyres (a stalk, a point of three and a strut each) and the arrow.
    assert!(lines(&mut game) >= 4 * 5 + 5, "{}", lines(&mut game));
}

fn console_game() -> TestGame {
    TestGame::new(|app| {
        app.plugin(threers_ecs::dev::DevTools);
        app.world.register::<Crate>();
        app.world
            .object(Object3D::group())
            .named("box")
            .with(Crate { weight: 3.0 });
        app.world
            .resource_mut::<Input>()
            .bind_axis("drive", Key::KeyS, Key::KeyW);
        app.world
            .console_command("double", "double: double every crate's weight", |w, _| {
                let mut n = 0;
                w.each::<&mut Crate>(|c| {
                    c.weight *= 2.0;
                    n += 1;
                });
                Ok(format!("{n} doubled"))
            });
    })
}

fn type_line(game: &mut TestGame, line: &str) {
    game.input().type_text(line);
    game.frames(1);
    game.tap(Key::Enter).frames(1);
}

#[test]
fn the_console_opens_takes_typing_and_runs_commands() {
    let mut game = console_game();
    game.tap(Key::Backquote).frames(1);
    assert!(game.resource::<threers_ecs::console::ConsoleState>().open);

    // Typing W into the console does not drive.
    game.input().press(Key::KeyW);
    game.frames(1);
    assert_eq!(game.resource::<Input>().axis("drive"), 0.0);
    game.input().release(Key::KeyW);

    type_line(&mut game, "select box");
    type_line(&mut game, "set Crate.weight 7.5");
    let (c, _) = game.world().single::<(&Crate, With<Crate>)>().unwrap();
    assert_eq!(c.weight, 7.5);
    type_line(&mut game, "double");
    assert_eq!(game.world().single::<(&Crate,)>().unwrap().0.weight, 15.0);
    type_line(&mut game, "speed 0.5");
    assert_eq!(game.resource::<Time>().scale, 0.5);
    type_line(&mut game, "tp 1 2 3");
    let (node, _) = game.world().single::<(NodeRef, With<Crate>)>().unwrap();
    assert_eq!(node.position, Vector3::new(1.0, 2.0, 3.0));

    // Escape closes it, and the keys drive again.
    game.tap(Key::Escape).frames(1);
    assert!(!game.resource::<threers_ecs::console::ConsoleState>().open);
    game.input().press(Key::KeyW);
    game.frames(1);
    assert_eq!(game.resource::<Input>().axis("drive"), 1.0);
}

#[derive(Default)]
struct Paused(u32, u32);

#[test]
fn escape_closing_the_console_does_not_reach_the_game() {
    let mut game = TestGame::new(|app| {
        app.plugin(threers_ecs::dev::DevTools);
        app.world.resource_mut::<Input>().bind("menu", Key::Escape);
        app.resource(Paused::default());
        // One of each kind, both after the console in the order.
        app.system(|w: &mut World| {
            if w.resource::<Input>().just_pressed("menu") {
                w.resource_mut::<Paused>().0 += 1;
            }
        });
        app.fixed_system(|w: &mut World| {
            if w.resource::<Input>().just_pressed("menu") {
                w.resource_mut::<Paused>().1 += 1;
            }
        });
    });
    game.tap(Key::Backquote).frames(1);
    game.tap(Key::Escape).frames(3);
    assert!(!game.resource::<threers_ecs::console::ConsoleState>().open);
    assert_eq!(
        (game.resource::<Paused>().0, game.resource::<Paused>().1),
        (0, 0)
    );
    // With it shut, Escape is the game's again.
    game.tap(Key::Escape).frames(1);
    assert_eq!(
        (game.resource::<Paused>().0, game.resource::<Paused>().1),
        (1, 1)
    );
}

#[test]
fn console_mistakes_say_what_there_is() {
    let mut game = console_game();
    let w = game.world();
    let unknown = threers_ecs::console::run(w, "fly").unwrap_err();
    assert!(
        unknown.contains("no command \"fly\"")
            && unknown.contains("double")
            && unknown.contains("help"),
        "{unknown}"
    );
    assert!(threers_ecs::console::run(w, "set Crate.weight 1")
        .unwrap_err()
        .contains("select something first"));
    threers_ecs::console::run(w, "select box").unwrap();
    assert!(threers_ecs::console::run(w, "set Crate.colour 1")
        .unwrap_err()
        .contains("no Crate.colour"));
    assert!(threers_ecs::console::run(w, "speed fast")
        .unwrap_err()
        .contains("needs a number"));
    assert!(threers_ecs::console::run(w, "help")
        .unwrap()
        .contains("double: double"));
}

#[test]
fn typed_text_survives_a_recording() {
    let mut app = App::new();
    app.start_recording();
    app.world.resource_mut::<Input>().type_text("tp 1 2%3 é");
    app.frame(1.0 / 60.0);
    let take = app.stop_recording().unwrap();
    let back = Recording::from_text(&take.to_text()).unwrap();
    assert_eq!(back, take);
    assert!(
        take.to_text()
            .lines()
            .nth(1)
            .unwrap()
            .split_whitespace()
            .count()
            == 2,
        "one word on the line"
    );
}

#[derive(Default)]
struct Progress {
    coins: f32,
}
reflect!(Progress { coins });

/// What one session saved, the next picks up.
#[test]
fn hot_reload_saves_and_the_next_build_resumes() {
    use threers_ecs::hot::{HotReload, HotState};
    let path = std::env::temp_dir().join(format!("threers_hot_{}/game.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let build = |app: &mut App| {
        app.world.register_resource::<Progress>();
        app.world.insert_resource(Progress { coins: 0.0 });
        app.world.register::<Crate>();
        app.world
            .object(Object3D::group())
            .named("box")
            .with(Crate { weight: 1.0 });
    };
    let mut first = App::new();
    build(&mut first);
    first.plugin(HotReload::new(&path).every(0.05));
    first.world.resource_mut::<Progress>().coins = 7.0;
    let id = first
        .world
        .iter::<(Entity, With<Crate>)>()
        .next()
        .unwrap()
        .0;
    first.world.get_mut::<Crate>(id).unwrap().weight = 4.5;
    first.world.node_mut(id).unwrap().position = Vector3::new(1.0, 2.0, 3.0);
    for _ in 0..3 {
        first.frame(0.1);
        std::thread::sleep(std::time::Duration::from_millis(60));
    }
    assert!(path.exists());

    let mut second = App::new();
    build(&mut second);
    second.plugin(HotReload::new(&path).resume(true));
    assert_eq!(second.world.resource::<HotState>().resumed, Ok(true));
    assert_eq!(second.world.resource::<Progress>().coins, 7.0);
    let (node, c) = second.world.single::<(NodeRef, &Crate)>().unwrap();
    assert_eq!(
        (node.position, c.weight),
        (Vector3::new(1.0, 2.0, 3.0), 4.5)
    );

    // Not asked to resume: a fresh game.
    let mut fresh = App::new();
    build(&mut fresh);
    fresh.plugin(HotReload::new(&path).resume(false));
    assert_eq!(fresh.world.resource::<Progress>().coins, 0.0);
}

#[test]
fn baking_merges_like_with_like_and_keeps_the_rest() {
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default());
    let w = &mut app.world;
    let grey = || StandardMaterial::new(Color::from_hex(0x888888));
    let parent = w
        .object(Object3D::group())
        .at(Vector3::new(0.0, 10.0, 0.0))
        .id();
    for i in 0..20 {
        w.mesh(BoxGeometry::new(1.0, 1.0, 1.0), grey())
            .at(Vector3::new(i as f32 * 2.0, 0.0, 0.0))
            .static_mesh();
    }
    // A second colour, a shadow setting of its own, one under a moved parent,
    // one with a body: four more groups' worth of cases.
    for i in 0..5 {
        w.mesh(
            BoxGeometry::new(1.0, 1.0, 1.0),
            StandardMaterial::new(Color::from_hex(0xff0000)),
        )
        .at(Vector3::new(i as f32, 5.0, 0.0))
        .static_mesh();
    }
    w.mesh(BoxGeometry::new(1.0, 1.0, 1.0), grey())
        .shadows(true, false)
        .static_mesh();
    let child = w.spawn_under(
        parent,
        Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), grey().into())),
    );
    w.insert(child, Static);
    let solid = w
        .mesh(BoxGeometry::new(1.0, 1.0, 1.0), grey())
        .at(Vector3::new(0.0, 0.0, 9.0))
        .static_mesh()
        .solid()
        .id();
    let loose = w
        .mesh(BoxGeometry::new(1.0, 1.0, 1.0), grey())
        .at(Vector3::new(0.0, 0.0, -9.0))
        .id();

    let baked = w.bake_static();
    assert_eq!(
        baked.meshes.len(),
        2,
        "grey and red; the lone shadow-caster stays"
    );
    assert_eq!(baked.merged, 20 + 1 + 1 + 5);
    assert!(w.node(loose).unwrap().visible, "not static, not touched");
    assert!(!w.node(child).unwrap().visible);
    // The child's box is where its parent put it.
    let grey_mesh = baked
        .meshes
        .iter()
        .copied()
        .find(|&m| match &w.node(m).unwrap().kind {
            ObjectKind::Mesh(mesh) => mesh.geometry.triangle_count() == 22 * 12,
            _ => false,
        });
    let grey_mesh = grey_mesh.expect("the grey bake");
    let ObjectKind::Mesh(mesh) = &w.node(grey_mesh).unwrap().kind else {
        unreachable!()
    };
    let top = mesh
        .geometry
        .positions()
        .unwrap()
        .map(|p| p.y)
        .fold(f32::MIN, f32::max);
    assert!((top - 10.5).abs() < 1e-4, "{top}");
    // The hidden original still collides.
    assert!(w.has::<Body>(solid));
}

#[path = "../examples/tutorial.rs"]
#[allow(dead_code)]
mod tutorial;

/// The tutorial's game plays: steering by the keys reaches the orbs, and
/// taking them all wins.
#[test]
fn the_tutorial_game_can_be_won_by_the_keys() {
    let dir = std::env::temp_dir().join(format!("threers_tutorial_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let here = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let mut game = TestGame::new(tutorial::build);
    std::env::set_current_dir(here).unwrap();
    for _ in 0..60 * 50 {
        // Head for the nearest orb, as a player would.
        let (ball, _) = game
            .world()
            .single::<(NodeRef, With<CameraTarget>)>()
            .map(|(n, c)| (n.position, c))
            .unwrap();
        let orb = game
            .world()
            .iter::<(NodeRef, Option<&Body>)>()
            .filter(|(n, b)| {
                b.is_some() && n.name.is_empty() && n.position.distance_to(ball) > 0.01
            })
            .map(|(n, _)| n.position)
            .filter(|p| (p.y - 1.0).abs() < 0.5)
            .min_by(|a, b| a.distance_to(ball).total_cmp(&b.distance_to(ball)));
        let Some(orb) = orb else { break };
        let to = orb - ball;
        let input = game.input();
        for (key, on) in [
            (Key::KeyD, to.x > 0.3),
            (Key::KeyA, to.x < -0.3),
            (Key::KeyS, to.z > 0.3),
            (Key::KeyW, to.z < -0.3),
        ] {
            if on {
                input.press(key)
            } else {
                input.release(key)
            }
        }
        game.frames(1);
        if game.world().state::<tutorial::Phase>() != tutorial::Phase::Playing {
            break;
        }
    }
    game.frames(2);
    assert_eq!(game.resource::<tutorial::Round>().collected, tutorial::ORBS);
    assert_eq!(
        game.world().state::<tutorial::Phase>(),
        tutorial::Phase::Won
    );
    assert!(
        game.resource::<tutorial::Round>().clock.remaining() > 0.0,
        "with time to spare"
    );
    eprintln!(
        "won with {:.1} s left, after {:.1} s",
        game.resource::<tutorial::Round>().clock.remaining(),
        game.resource::<Time>().elapsed
    );
}

/// The tutorial's step 9 says this; hold it to it.
#[test]
fn the_tutorial_test_snippet_holds() {
    let dir = std::env::temp_dir().join(format!("threers_tutorial9_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let here = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let mut game = TestGame::new(tutorial::build);
    std::env::set_current_dir(here).unwrap();
    game.hold(Key::KeyW, 60);
    assert!(
        game.world()
            .single::<(NodeRef, With<CameraTarget>)>()
            .unwrap()
            .0
            .position
            .z
            < 0.0
    );
}
