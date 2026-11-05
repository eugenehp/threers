//! Touch, save slots, packs, fonts, the loading screen, pointer lock and rumble.
#![recursion_limit = "512"]

use threers::math::{Color, Vector2};
use threers_ecs::assets::*;
use threers_ecs::hud::{with_hud, Hud};
use threers_ecs::prelude::*;
use threers_ecs::reflect;
use threers_ecs::saves::SaveSlots;
use threers_ecs::testing::TestGame;
use threers_ecs::touch::{TouchButton, TouchControls};

fn touch_game() -> TestGame {
    TestGame::new(|app| {
        app.resource(Hud::new(640, 360));
        app.plugin(
            TouchControls::new()
                .stick("x", "y")
                .button(TouchButton::new("jump", "A", 0.88, 0.75)),
        );
    })
}

fn at(x: f32, y: f32) -> Vector2 {
    Vector2::new(x, y)
}

#[test]
fn a_thumb_drags_the_stick_from_wherever_it_landed() {
    let mut g = touch_game();
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(1);
    assert_eq!(g.input().axis("x"), 0.0);
    g.input().touch_move(1, at(135.0, 165.0));
    g.frames(1);
    let v = g.input().axis2("x", "y");
    assert!((v.x - 0.5).abs() < 1e-3, "{v:?}");
    assert!((v.y - 0.5).abs() < 1e-3, "up is positive: {v:?}");
    g.input().touch_end(1);
    g.frames(1);
    assert_eq!(g.input().axis("x"), 0.0);
}

#[test]
fn a_full_push_stops_at_one() {
    let mut g = touch_game();
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(1);
    g.input().touch_move(1, at(400.0, 200.0));
    g.frames(1);
    assert_eq!(g.input().axis("x"), 1.0);
}

#[test]
fn a_touch_on_the_right_is_not_the_stick() {
    let mut g = touch_game();
    g.input().touch_start(1, at(400.0, 100.0));
    g.frames(1);
    g.input().touch_move(1, at(450.0, 100.0));
    g.frames(1);
    assert_eq!(g.input().axis("x"), 0.0);
}

#[test]
fn a_button_works_as_its_action_at_the_same_time_as_the_stick() {
    let mut g = touch_game();
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(1);
    g.input().touch_move(1, at(170.0, 200.0));
    g.input().touch_start(2, at(0.88 * 640.0, 0.75 * 360.0));
    g.frames(1);
    assert!(g.input().pressed("jump"));
    assert_eq!(g.input().axis("x"), 1.0);
    g.input().touch_end(2);
    g.frames(1);
    assert!(!g.input().pressed("jump"));
    assert_eq!(g.input().axis("x"), 1.0, "the stick keeps its own finger");
}

#[test]
fn a_gamepad_host_cannot_wipe_the_stick() {
    let mut g = touch_game();
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(1);
    g.input().touch_move(1, at(170.0, 200.0));
    g.frames(1);
    PadTracker::default().sync(g.input());
    assert_eq!(g.input().axis("x"), 1.0);
}

#[test]
fn nothing_is_drawn_until_someone_touches_the_screen() {
    let painted = |g: &TestGame| {
        let hud = g.resource::<Hud>();
        hud.pixels().chunks(4).filter(|p| p[3] != 0).count()
    };
    let mut g = touch_game();
    g.frames(2);
    assert_eq!(painted(&g), 0);
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(2);
    assert!(painted(&g) > 1000);
}

#[test]
fn losing_focus_lifts_every_finger() {
    let mut g = touch_game();
    g.input().touch_start(1, at(100.0, 200.0));
    g.frames(1);
    g.input().touch_move(1, at(170.0, 200.0));
    g.frames(1);
    g.input().release_all();
    assert_eq!(g.input().axis("x"), 0.0);
    assert!(g.input().touches.is_empty());
}

#[derive(Default, Debug, PartialEq)]
struct Points {
    value: u32,
}
reflect!(Points { value });

fn slots(name: &str) -> (SaveSlots, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("threers-saves-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    (SaveSlots::in_dir("test", &dir), dir)
}

fn world(points: u32) -> World {
    let mut w = World::new();
    w.register_resource::<Points>();
    w.insert_resource(Points { value: points });
    w
}

#[test]
fn a_slot_survives_to_a_fresh_world() {
    let (slots, _) = slots("roundtrip");
    slots.write(&world(7), "one").unwrap();
    assert!(slots.exists("one"));
    let mut fresh = world(0);
    slots.read(&mut fresh, "one").unwrap();
    assert_eq!(fresh.resource::<Points>().value, 7);
}

#[test]
fn slots_list_replace_and_delete() {
    let (slots, dir) = slots("list");
    assert!(slots.list().is_empty());
    slots.write(&world(1), "b").unwrap();
    slots.write(&world(2), "a").unwrap();
    slots.write(&world(3), "a").unwrap();
    assert_eq!(slots.list(), ["a", "b"]);
    let mut fresh = world(0);
    slots.read(&mut fresh, "a").unwrap();
    assert_eq!(fresh.resource::<Points>().value, 3);
    slots.delete("a").unwrap();
    slots.delete("a").unwrap();
    assert_eq!(slots.list(), ["b"]);
    let leftovers = std::fs::read_dir(dir).unwrap().flatten().count();
    assert_eq!(leftovers, 1, "no temporary files left behind");
}

#[test]
fn a_missing_slot_is_an_error_and_a_bad_name_never_reaches_the_disk() {
    let (slots, dir) = slots("names");
    let mut w = world(0);
    assert!(slots
        .read(&mut w, "nothing")
        .unwrap_err()
        .contains("no save"));
    for bad in ["../escape", "a/b", "", "a.b", "..\\x"] {
        assert!(slots.write(&w, bad).is_err(), "{bad}");
        assert!(!slots.exists(bad));
        assert!(slots.delete(bad).is_err());
    }
    assert!(!dir.exists());
}

#[test]
fn the_default_place_is_per_game() {
    assert_eq!(SaveSlots::new("mine").game(), "mine");
}

fn assets_game(dir: &std::path::Path) -> TestGame {
    let dir = dir.to_path_buf();
    TestGame::new(move |app| {
        app.plugin(AssetsPlugin::new(dir));
    })
}

fn wait(g: &mut TestGame, mut done: impl FnMut(&mut TestGame) -> bool) {
    let start = std::time::Instant::now();
    while !done(g) {
        assert!(start.elapsed().as_secs() < 10, "timed out");
        g.frames(1);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("threers-pack-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_pack_round_trips_and_rejects_damage() {
    let bytes = Pack::build([("a", &b"one"[..]), ("dir/b", &b"two!"[..])]);
    assert!(Pack::parse(&bytes).is_ok());
    assert!(Pack::parse(b"nope").is_err());
    for cut in 0..bytes.len() {
        assert!(Pack::parse(&bytes[..cut]).is_err(), "cut at {cut}");
    }
}

#[test]
fn assets_come_from_a_pack_even_when_asked_for_before_it_arrives() {
    let dir = temp("from");
    let pack = Pack::build([("a.bin", &b"packed"[..]), ("later.bin", &b"later"[..])]);
    std::fs::write(dir.join("level.pack"), pack).unwrap();
    std::fs::write(dir.join("loose.bin"), b"loose").unwrap();
    let mut g = assets_game(&dir);
    let assets = g.world().resource_mut::<Assets>();
    assets.load_pack("level.pack");
    let inside = assets.load::<Bytes>("a.bin");
    let outside = assets.load::<Bytes>("loose.bin");
    wait(&mut g, |g| !g.resource::<Assets>().is_loading());
    assert_eq!(g.resource::<Assets>().get(&inside).unwrap().0, b"packed");
    assert_eq!(g.resource::<Assets>().get(&outside).unwrap().0, b"loose");

    // And for anyone who asks afterwards, even with a file of that name on disk.
    std::fs::write(dir.join("later.bin"), b"not this one").unwrap();
    let late = g
        .world()
        .resource_mut::<Assets>()
        .load::<Bytes>("later.bin");
    wait(&mut g, |g| g.resource::<Assets>().get(&late).is_some());
    assert_eq!(g.resource::<Assets>().get(&late).unwrap().0, b"later");
}

#[test]
#[should_panic(expected = "already loaded as another type")]
fn one_path_cannot_be_two_types() {
    let mut assets = Assets::new("nowhere");
    assets.load::<Bytes>("a");
    assets.load::<SceneAsset>("a");
}

#[test]
fn a_missing_pack_does_not_strand_what_was_waiting() {
    let dir = temp("gone");
    std::fs::write(dir.join("loose.bin"), b"loose").unwrap();
    let mut g = assets_game(&dir);
    let assets = g.world().resource_mut::<Assets>();
    assets.load_pack("gone.pack");
    let loose = assets.load::<Bytes>("loose.bin");
    wait(&mut g, |g| g.resource::<Assets>().get(&loose).is_some());
}

#[test]
fn progress_counts_up() {
    let dir = temp("progress");
    std::fs::write(dir.join("a"), b"1").unwrap();
    let mut g = assets_game(&dir);
    assert_eq!(g.resource::<Assets>().progress(), (0, 0));
    g.world().resource_mut::<Assets>().load::<Bytes>("a");
    g.world().resource_mut::<Assets>().load::<Bytes>("missing");
    assert_eq!(g.resource::<Assets>().progress(), (0, 2));
    wait(&mut g, |g| g.resource::<Assets>().progress() == (2, 2));
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Screen {
    Loading,
    Playing,
}
impl GameState for Screen {}

#[test]
fn the_loading_screen_draws_a_bar_and_moves_on_when_done() {
    let dir = temp("loading");
    std::fs::write(dir.join("a"), b"1").unwrap();
    let mut g = {
        let dir = dir.clone();
        TestGame::new(move |app| {
            app.resource(Hud::new(320, 180));
            app.plugin(AssetsPlugin::new(dir))
                .state(Screen::Loading)
                .plugin(LoadingScreen {
                    during: Screen::Loading,
                    then: Screen::Playing,
                });
        })
    };
    g.world().resource_mut::<Assets>().load::<Bytes>("a");
    g.frames(1);
    assert!(g.resource::<Hud>().pixels().chunks(4).any(|p| p[3] != 0));
    wait(&mut g, |g| g.world().state::<Screen>() == Screen::Playing);
}

fn grey_pixels(hud: &Hud) -> usize {
    hud.pixels()
        .chunks(4)
        .filter(|p| p[3] > 20 && p[3] < 235)
        .count()
}

#[test]
fn a_real_font_draws_smooth_edges() {
    let dir = temp("font");
    std::fs::write(
        dir.join("sans.ttf"),
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../src/openscad/DejaVuSans.ttf"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut g = {
        let dir = dir.clone();
        TestGame::new(move |app| {
            app.resource(Hud::new(320, 180));
            app.plugin(AssetsPlugin::new(dir));
            app.system(|w: &mut World| {
                with_hud(w, |hud| {
                    hud.fill(Color::WHITE);
                    hud.text_size(28.0);
                    hud.text(10.0, 40.0, "coins 3 / 4");
                });
            });
        })
    };
    use_font(g.world(), "sans.ttf");
    wait(&mut g, |g| {
        let w = g.world();
        let font = w.resource_mut::<Assets>().load::<FontAsset>("sans.ttf");
        w.resource::<Assets>().get(&font).is_some()
    });
    g.frames(3);
    assert!(grey_pixels(g.resource::<Hud>()) > 100, "antialiased");
    assert!(FontAsset::parse(b"not a font").is_err());
}

#[test]
fn a_game_can_ask_for_the_pointer_and_for_rumble() {
    let mut input = Input::default();
    assert!(input.take_rumble().is_none());
    input.rumble(0.3, 0.1);
    input.rumble(0.8, 0.2);
    input.rumble(0.5, 0.3);
    let rumble = input.take_rumble().unwrap();
    assert_eq!((rumble.strength, rumble.seconds), (0.8, 0.2));
    assert!(input.take_rumble().is_none());

    input.rumble(9.0, 1.0);
    assert_eq!(input.take_rumble().unwrap().strength, 1.0);
}

#[test]
fn locked_motion_adds_up_with_no_position() {
    let mut app = App::new();
    let input = app.world.resource_mut::<Input>();
    input.add_mouse_motion(at(5.0, -2.0));
    input.add_mouse_motion(at(5.0, -2.0));
    assert_eq!((input.mouse_delta.x, input.mouse_delta.y), (10.0, -4.0));
    assert_eq!(input.mouse_position.x, 0.0);
    app.frame(0.0);
    assert_eq!(app.world.resource::<Input>().mouse_delta.x, 0.0);
}
