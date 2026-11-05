//! Plays the `hop` example from start to finish, through the same input a window
//! would feed it. Nothing here touches the game's state directly.

#[path = "../examples/hop.rs"]
mod hop;

use hop::{Coin, Phase, Player, Score, PLATFORMS};
use threers_ecs::prelude::*;

fn game() -> App {
    let mut app = App::new();
    hop::build(&mut app);
    // Settle onto the first platform.
    advance(&mut app, 30);
    app
}

fn advance(app: &mut App, frames: u32) {
    for _ in 0..frames {
        app.frame(1.0 / 60.0);
    }
}

fn player(app: &mut App) -> Vector3 {
    let mut position = None;
    app.world
        .each::<(Node, With<Player>)>(|(node, _)| position = Some(node.position));
    position.unwrap()
}

fn grounded(app: &mut App) -> bool {
    let mut grounded = false;
    app.world
        .each::<(&Character, With<Player>)>(|(character, _)| grounded = character.grounded);
    grounded
}

fn score(app: &App) -> (u32, u32) {
    let score = app.world.resource::<Score>();
    (score.coins, score.falls)
}

fn input(app: &mut App) -> &mut Input {
    app.world.resource_mut::<Input>()
}

fn press_for(app: &mut App, key: Key, frames: u32) {
    input(app).press(key);
    advance(app, frames);
    input(app).release(key);
}

/// Walk to a point, one axis at a time being enough for this level, jumping
/// first if asked. Steers by the same keys a player would hold.
fn hop_to(app: &mut App, x: f32, z: f32, jump: bool) {
    if jump {
        input(app).press(Key::Space);
        advance(app, 1);
        input(app).release(Key::Space);
    }
    for _ in 0..300 {
        let at = player(app);
        let (dx, dz) = (x - at.x, z - at.z);
        for (key, on) in [
            (Key::KeyD, dx > 0.15),
            (Key::KeyA, dx < -0.15),
            (Key::KeyS, dz > 0.15),
            (Key::KeyW, dz < -0.15),
        ] {
            if on {
                input(app).press(key);
            } else {
                input(app).release(key);
            }
        }
        advance(app, 1);
        if dx.abs() <= 0.15 && dz.abs() <= 0.15 {
            break;
        }
    }
    input(app).release_all();
    for _ in 0..180 {
        if grounded(app) {
            break;
        }
        advance(app, 1);
    }
}

#[test]
fn the_player_lands_and_stands_on_the_first_platform() {
    let mut app = game();
    assert!(grounded(&mut app));
    // Top of the platform, plus half the capsule.
    let y = player(&mut app).y;
    assert!((y - 1.3).abs() < 0.05, "{y}");
}

#[test]
fn no_coin_can_be_had_without_jumping() {
    let mut app = game();
    press_for(&mut app, Key::KeyW, 25);
    advance(&mut app, 30);
    assert_eq!(score(&app).0, 0);
    assert_eq!(app.world.entities::<With<Coin>>().len(), PLATFORMS.len());
}

#[test]
fn walking_moves_at_the_set_speed_and_a_diagonal_is_no_faster() {
    let mut app = game();
    let start = player(&mut app);
    press_for(&mut app, Key::KeyD, 30);
    let moved = player(&mut app).x - start.x;
    assert!((moved - hop::SPEED * 0.5).abs() < 0.2, "{moved}");
}

#[test]
fn a_jump_rises_and_comes_back_down_only_when_pressed_on_the_ground() {
    let mut app = game();
    let ground = player(&mut app).y;
    press_for(&mut app, Key::Space, 1);
    assert!(!grounded(&mut app));
    let mut peak = ground;
    for _ in 0..120 {
        advance(&mut app, 1);
        peak = peak.max(player(&mut app).y);
    }
    let expected = hop::JUMP * hop::JUMP / (2.0 * 9.81);
    assert!(
        (peak - ground - expected).abs() < 0.2,
        "rose {}",
        peak - ground
    );
    assert!(grounded(&mut app));

    // A second press in the air does nothing.
    press_for(&mut app, Key::Space, 10);
    let rising = player(&mut app).y;
    press_for(&mut app, Key::Space, 10);
    assert!(player(&mut app).y < rising + 1.0);
}

#[test]
fn a_jump_works_the_same_at_any_frame_rate() {
    let run = |fps: u32| {
        let mut app = App::new();
        hop::build(&mut app);
        for _ in 0..fps / 2 {
            app.frame(1.0 / fps as f64);
        }
        input(&mut app).press(Key::Space);
        let mut peak = f32::MIN;
        for _ in 0..fps {
            app.frame(1.0 / fps as f64);
            input(&mut app).release(Key::Space);
            peak = peak.max(player(&mut app).y);
        }
        peak
    };
    let peak = run(60);
    for fps in [30, 120, 240] {
        assert!((run(fps) - peak).abs() < 0.2, "{fps} fps");
    }
}

#[test]
fn walking_off_the_edge_falls_and_comes_back_to_the_start() {
    let mut app = game();
    // Long enough to clear the edge, short enough not to walk off again.
    press_for(&mut app, Key::KeyD, 60);
    advance(&mut app, 180);
    assert_eq!(score(&app).1, 1);
    assert!(grounded(&mut app));
    let at = player(&mut app);
    assert!(at.x.abs() < 0.1 && at.z.abs() < 0.1, "{at:?}");
}

#[test]
fn a_focus_loss_stops_the_player() {
    let mut app = game();
    input(&mut app).press(Key::KeyD);
    advance(&mut app, 5);
    input(&mut app).release_all();
    advance(&mut app, 2);
    let at = player(&mut app).x;
    advance(&mut app, 10);
    assert_eq!(player(&mut app).x, at);
}

#[test]
fn the_camera_follows_the_player() {
    let mut app = game();
    press_for(&mut app, Key::KeyD, 20);
    advance(&mut app, 1);
    let at = player(&mut app);
    let camera = &app.world.resource::<MainCamera>().0;
    assert_eq!(camera.position.x, at.x);
}

#[test]
fn escape_asks_the_host_to_stop() {
    let mut app = game();
    assert!(app.world.get_resource::<Exit>().is_none());
    press_for(&mut app, Key::Escape, 1);
    assert!(app.world.get_resource::<Exit>().is_some());
}

#[test]
fn jumping_and_collecting_make_sounds() {
    use threers_ecs::audio::{AudioOut, AudioPlugin};
    let mut app = App::new();
    hop::build(&mut app);
    app.plugin(AudioPlugin::Offline);
    advance(&mut app, 30);
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(0));
    press_for(&mut app, Key::Space, 1);
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(1));
    // Straight up into the first coin.
    advance(&mut app, 60);
    assert_eq!(score(&app).0, 1);
    assert_eq!(app.world.resource::<AudioOut>().offline_voices(), Some(2));
}

#[test]
fn a_gamepad_plays_the_game_too() {
    let mut app = game();
    let mut pad = PadTracker::default();
    let start = player(&mut app);

    let mut send = |app: &mut App, event| {
        pad.update(0, event);
        pad.sync(input(app));
    };

    // Stick up is forward, which in this level is towards -z.
    send(&mut app, PadEvent::Axis(PadAxis::LeftY, 1.0));
    advance(&mut app, 12);
    send(&mut app, PadEvent::Axis(PadAxis::LeftY, 0.0));
    advance(&mut app, 2);
    let at = player(&mut app);
    assert!(at.z < start.z - 0.5, "{at:?}");

    send(&mut app, PadEvent::Pressed(PadButton::South));
    send(&mut app, PadEvent::Released(PadButton::South));
    advance(&mut app, 5);
    assert!(player(&mut app).y > at.y + 0.3);
}

#[test]
fn the_whole_level_can_be_played_through() {
    let mut app = game();
    assert_eq!(score(&app), (0, 0), "nothing is free");

    // The first coin floats above the start: jump straight up into it.
    hop_to(&mut app, 0.0, 0.0, true);
    assert_eq!(score(&app), (1, 0));

    for (index, [x, _, z]) in PLATFORMS.iter().enumerate().skip(1) {
        hop_to(&mut app, *x, *z, true);
        // Up into the coin overhead, if the arc over didn't already take it.
        hop_to(&mut app, *x, *z, true);
        assert_eq!(score(&app).0, index as u32 + 1);
    }

    assert_eq!(score(&app), (4, 0));
    assert!(app.world.entities::<With<Coin>>().is_empty());

    // The win takes effect, and stops the controls.
    advance(&mut app, 1);
    assert_eq!(app.world.state::<Phase>(), Phase::Won);
    let at = player(&mut app);
    press_for(&mut app, Key::KeyD, 20);
    assert_eq!(player(&mut app).x, at.x);
}

#[test]
fn a_save_taken_mid_game_restores_coins_score_and_place() {
    let mut app = game();
    hop_to(&mut app, 0.0, 0.0, true);
    press_for(&mut app, Key::KeyD, 10);
    advance(&mut app, 5);
    let at = player(&mut app);
    let save = app.world.save();
    assert_eq!(score(&app).0, 1);

    // A fresh game, loaded, is the same game.
    let mut other = game();
    assert_eq!(other.world.entities::<With<Coin>>().len(), 4);
    other.world.load(&save).expect("load");
    assert_eq!(score(&other).0, 1);
    assert_eq!(other.world.entities::<With<Coin>>().len(), 3);
    advance(&mut other, 2);
    let there = player(&mut other);
    assert!(
        (there.x - at.x).abs() < 1e-3 && (there.y - at.y).abs() < 0.02,
        "{there:?} {at:?}"
    );

    // And it still plays: the body came with it, so the character stands and walks.
    assert!(grounded(&mut other));
    press_for(&mut other, Key::KeyA, 10);
    assert!(player(&mut other).x < there.x - 0.3);
}
