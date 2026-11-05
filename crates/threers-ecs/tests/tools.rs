//! `TestGame`, the HUD and the dev tools.

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::BoxGeometry;
use threers::lights::AmbientLight;
use threers::materials::{Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers_ecs::dev::{DevState, DevTools};
use threers_ecs::hud::{with_hud, Hud};
use threers_ecs::prelude::*;
use threers_ecs::reflect;
use threers_ecs::testing::TestGame;

#[derive(Default)]
struct Health {
    hp: f32,
}
impl Component for Health {}
reflect!(Health { hp });

fn painted(hud: &Hud) -> usize {
    hud.pixels().chunks(4).filter(|p| p[3] != 0).count()
}

#[test]
fn a_game_can_run_at_any_frame_rate_and_time_adds_up() {
    for fps in [30.0, 144.0] {
        let mut game = TestGame::new(|_| {}).at_fps(fps);
        game.seconds(2.0);
        assert!(
            (game.resource::<Time>().elapsed - 2.0).abs() < 0.02,
            "{fps}"
        );
    }
}

#[test]
fn hold_and_tap_drive_input() {
    let mut game = TestGame::new(|app| {
        app.world.resource_mut::<Input>().bind("go", Key::KeyG);
    });
    game.press(Key::KeyG).frames(2);
    assert!(game.input().pressed("go"));
    game.release(Key::KeyG);
    assert!(!game.input().pressed("go"));
    game.hold(Key::KeyG, 3);
    assert!(!game.input().pressed("go"));
}

fn scene(color: u32) -> TestGame {
    TestGame::new(|app| {
        let mesh = Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(StandardMaterial::new(Color::from_hex(color))),
        );
        app.world.spawn(Object3D::mesh(mesh));
        app.world
            .scene
            .add_light(AmbientLight::new(Color::WHITE, 2.0));
        let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
        camera.position = Vector3::new(2.0, 2.0, 3.0);
        camera.look_at(Vector3::ZERO);
        app.resource(MainCamera(camera));
    })
}

#[test]
fn frames_compare_through_the_real_renderer() {
    let dir = std::env::temp_dir().join(format!("threers-ecs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("frame.png");

    // Nothing to compare with is a failure, not a pass.
    let missing =
        std::panic::catch_unwind(|| scene(0xff0000).assert_frame_matches("/nonexistent.png", 1.0));
    assert!(missing.is_err());

    let rgba = scene(0xff0000).render(320, 180);
    std::fs::write(&path, threers::encode_png(320, 180, &rgba)).unwrap();
    scene(0xff0000).assert_frame_matches(&path, 0.5);

    let path_again = path.clone();
    let different =
        std::panic::catch_unwind(move || scene(0x00ff00).assert_frame_matches(&path_again, 0.5));
    assert!(different.is_err(), "a green box matched a red one");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_hud_starts_each_frame_clear() {
    let mut game = TestGame::new(|app| {
        app.resource(Hud::new(320, 180));
        app.resource(true);
        app.system(|w: &mut World| {
            if *w.resource::<bool>() {
                with_hud(w, |hud| {
                    hud.fill(Color::WHITE);
                    hud.rect(10.0, 10.0, 50.0, 50.0);
                });
            }
        });
    });
    game.frames(1);
    assert!(painted(game.resource::<Hud>()) >= 2500);
    game.world().insert_resource(false);
    game.frames(1);
    assert_eq!(painted(game.resource::<Hud>()), 0);
}

#[test]
fn drawing_with_no_hud_is_nothing_at_all() {
    let mut w = World::new();
    with_hud(&mut w, |_| panic!("there is no hud"));
}

fn tools() -> TestGame {
    TestGame::new(|app| {
        app.plugin(DevTools).resource(Hud::new(640, 360));
        app.world.register::<Health>();
        for hp in [1.0, 2.0] {
            app.world.spawn_with(Object3D::group(), Health { hp });
        }
    })
}

#[test]
fn f3_shows_a_panel_only_while_it_is_on() {
    let mut game = tools();
    game.frames(1);
    assert_eq!(painted(game.resource::<Hud>()), 0);
    game.tap(Key::F3);
    assert!(game.resource::<DevState>().panel);
    assert!(painted(game.resource::<Hud>()) > 1000);
    game.tap(Key::F3);
    assert_eq!(painted(game.resource::<Hud>()), 0);
}

#[test]
fn f5_pauses_and_f6_steps() {
    let mut game = tools();
    game.tap(Key::F5);
    assert!(game.resource::<Time>().paused);
    let steps = game.resource::<Time>().fixed_steps;
    game.frames(10);
    assert_eq!(game.resource::<Time>().fixed_steps, steps);
    game.tap(Key::F6);
    game.frames(1);
    assert_eq!(game.resource::<Time>().fixed_steps, steps + 1);
    game.tap(Key::F5);
    assert!(!game.resource::<Time>().paused);
}

#[test]
fn f7_walks_the_inspectable_entities() {
    let mut game = tools();
    let ids = game.world().inspectable();
    game.tap(Key::F7);
    assert_eq!(game.resource::<DevState>().selected, Some(ids[0]));
    game.tap(Key::F7);
    assert_eq!(game.resource::<DevState>().selected, Some(ids[1]));
    game.tap(Key::F7);
    assert_eq!(game.resource::<DevState>().selected, Some(ids[0]));
}

#[test]
fn frame_time_is_measured() {
    let mut game = tools();
    game.frames(30);
    let ms = game.resource::<DevState>().frame_ms();
    assert!((ms - 16.67).abs() < 0.1, "{ms}");
}
