//! Actions, and how presses reach fixed and variable systems.

use threers::math::Vector2;
use threers_ecs::prelude::*;

fn bound() -> Input {
    let mut input = Input::default();
    input
        .bind("jump", Key::Space)
        .bind("jump", PadButton::South);
    input.bind_axis("x", Key::KeyA, Key::KeyD);
    input.bind_axis("y", Key::KeyS, Key::KeyW);
    input.bind_analog("x", PadAxis::LeftX);
    input
}

#[test]
fn an_action_follows_any_of_its_buttons() {
    let mut input = bound();
    assert!(!input.pressed("jump"));
    input.press(PadButton::South);
    assert!(input.pressed("jump"));
    input.press(Key::Space);
    input.release(PadButton::South);
    assert!(input.pressed("jump"));
    input.release(Key::Space);
    assert!(!input.pressed("jump"));
}

#[test]
#[should_panic(expected = "no action named \"jmup\"")]
fn a_misspelt_action_says_so() {
    bound().pressed("jmup");
}

#[test]
#[should_panic(expected = "no axis named \"z\"")]
fn a_misspelt_axis_says_so() {
    bound().axis("z");
}

#[test]
fn opposite_keys_cancel() {
    let mut input = bound();
    input.press(Key::KeyA);
    assert_eq!(input.axis("x"), -1.0);
    input.press(Key::KeyD);
    assert_eq!(input.axis("x"), 0.0);
}

#[test]
fn a_diagonal_is_no_faster_than_straight_ahead() {
    let mut input = bound();
    input.press(Key::KeyD);
    input.press(Key::KeyW);
    let v = input.axis2("x", "y");
    assert!((v.x * v.x + v.y * v.y - 1.0).abs() < 1e-6);
    assert!((v.x - v.y).abs() < 1e-6);
}

#[test]
fn a_partial_stick_stays_partial_on_two_axes() {
    let mut input = Input::default();
    input.deadzone = 0.0;
    input
        .bind_analog("x", PadAxis::LeftX)
        .bind_analog("y", PadAxis::LeftY);
    input.set_analog(PadAxis::LeftX, 0.3);
    input.set_analog(PadAxis::LeftY, 0.4);
    let v = input.axis2("x", "y");
    assert!((v.x - 0.3).abs() < 1e-6 && (v.y - 0.4).abs() < 1e-6);
}

#[test]
fn the_deadzone_hides_drift_and_full_travel_still_reaches_one() {
    let mut input = bound();
    input.set_analog(PadAxis::LeftX, 0.1);
    assert_eq!(input.axis("x"), 0.0);
    input.set_analog(PadAxis::LeftX, -1.0);
    assert_eq!(input.axis("x"), -1.0);
    // Just past the edge starts from zero rather than jumping to 0.15.
    input.set_analog(PadAxis::LeftX, 0.16);
    assert!(input.axis("x") > 0.0 && input.axis("x") < 0.02);
}

#[test]
fn a_resting_stick_does_not_cancel_a_held_key() {
    let mut input = bound();
    input.set_analog(PadAxis::LeftX, 0.0);
    input.press(Key::KeyD);
    assert_eq!(input.axis("x"), 1.0);
}

#[test]
fn losing_focus_lets_everything_go() {
    let mut input = bound();
    input.press(Key::Space);
    input.press(Key::KeyD);
    input.set_analog(PadAxis::LeftX, 1.0);
    input.release_all();
    assert!(!input.pressed("jump"));
    assert_eq!(input.axis("x"), 0.0);
    assert!(input.just_released("jump"));
}

#[test]
fn mouse_movement_adds_up_within_a_frame_and_then_clears() {
    let mut app = App::new();
    let input = app.world.resource_mut::<Input>();
    input.move_mouse(Vector2::new(10.0, 0.0));
    input.move_mouse(Vector2::new(15.0, 5.0));
    assert_eq!(input.mouse_delta.x, 15.0);
    assert_eq!(input.mouse_delta.y, 5.0);
    input.scroll = 2.0;
    app.frame(0.0);
    let input = app.world.resource::<Input>();
    assert_eq!(input.mouse_delta.x, 0.0);
    assert_eq!(input.scroll, 0.0);
    assert_eq!(input.mouse_position.x, 15.0);
}

#[derive(Default)]
struct Seen {
    fixed: u32,
    variable: u32,
    fixed_releases: u32,
}

fn watch_fixed(w: &mut World) {
    let input = w.resource::<Input>();
    let (pressed, released) = (input.just_pressed("jump"), input.just_released("jump"));
    let seen = w.resource_mut::<Seen>();
    seen.fixed += pressed as u32;
    seen.fixed_releases += released as u32;
}

fn watch_variable(w: &mut World) {
    let pressed = w.resource::<Input>().just_pressed("jump");
    w.resource_mut::<Seen>().variable += pressed as u32;
}

fn app() -> App {
    let mut app = App::new();
    app.world.resource_mut::<Input>().bind("jump", Key::Space);
    app.resource(Seen::default());
    app.fixed_system(watch_fixed);
    app.system(watch_variable);
    app
}

fn seen(app: &App) -> (u32, u32) {
    let seen = app.world.resource::<Seen>();
    (seen.fixed, seen.variable)
}

fn tap(app: &mut App) {
    let input = app.world.resource_mut::<Input>();
    input.press(Key::Space);
    input.release(Key::Space);
}

#[test]
fn a_press_on_a_frame_with_no_step_waits_for_the_next_step() {
    let mut app = app();
    app.world.resource_mut::<Input>().press(Key::Space);
    app.frame(0.25 / 60.0);
    assert_eq!(seen(&app), (0, 1));
    app.frame(0.25 / 60.0);
    assert_eq!(seen(&app), (0, 1));
    app.frame(0.6 / 60.0);
    assert_eq!(seen(&app), (1, 1));
    app.frame(2.0 / 60.0);
    assert_eq!(seen(&app), (1, 1));
}

#[test]
fn a_press_on_a_frame_with_two_steps_counts_once() {
    let mut app = app();
    app.world.resource_mut::<Input>().press(Key::Space);
    app.frame(2.0 / 60.0);
    assert_eq!(app.world.resource::<Time>().fixed_steps, 2);
    assert_eq!(seen(&app), (1, 1));
}

#[test]
fn a_tap_shorter_than_a_frame_is_still_seen() {
    let mut app = app();
    tap(&mut app);
    app.frame(1.0 / 60.0);
    assert_eq!(seen(&app), (1, 1));
    assert!(!app.world.resource::<Input>().pressed("jump"));
    assert_eq!(app.world.resource::<Seen>().fixed_releases, 1);
}

#[test]
fn every_tap_counts_at_any_frame_rate() {
    for fps in [30.0, 60.0, 144.0, 240.0] {
        let mut app = app();
        for _ in 0..50 {
            tap(&mut app);
            // Long enough that a step always falls between taps.
            for _ in 0..(fps / 15.0_f64).ceil() as u32 {
                app.frame(1.0 / fps);
            }
        }
        assert_eq!(seen(&app), (50, 50), "{fps} fps");
    }
}

#[test]
fn a_held_key_repeating_is_not_a_new_press() {
    let mut app = app();
    app.world.resource_mut::<Input>().press(Key::Space);
    app.frame(1.0 / 60.0);
    app.world.resource_mut::<Input>().press(Key::Space);
    app.frame(1.0 / 60.0);
    assert_eq!(seen(&app), (1, 1));
}

#[test]
fn variable_systems_read_their_own_edges_after_a_frame_of_steps() {
    let mut app = app();
    app.world.resource_mut::<Input>().press(Key::Space);
    app.frame(3.0 / 60.0);
    // The fixed steps used theirs up; the frame's is untouched.
    assert_eq!(seen(&app), (1, 1));
}

#[test]
fn key_codes_match_the_browsers_names() {
    assert_eq!(Key::from_code("KeyW"), Some(Key::KeyW));
    assert_eq!(Key::from_code("ArrowLeft"), Some(Key::ArrowLeft));
    assert_eq!(Key::from_code("Nonsense"), None);
}

fn pads() -> (PadTracker, Input) {
    let mut input = Input::default();
    input.bind("jump", PadButton::South);
    input.bind_analog("x", PadAxis::LeftX);
    (PadTracker::default(), input)
}

fn apply(tracker: &mut PadTracker, input: &mut Input, pad: usize, event: PadEvent) {
    tracker.update(pad, event);
    tracker.sync(input);
}

#[test]
fn a_pad_drives_actions_and_axes() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    assert!(input.pressed("jump") && input.just_pressed("jump"));
    apply(&mut t, &mut input, 0, PadEvent::Axis(PadAxis::LeftX, -1.0));
    assert_eq!(input.axis("x"), -1.0);
    apply(&mut t, &mut input, 0, PadEvent::Released(PadButton::South));
    assert!(!input.pressed("jump"));
}

#[test]
fn one_pad_letting_go_does_not_release_what_another_holds() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 1, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 0, PadEvent::Released(PadButton::South));
    assert!(input.pressed("jump"));
    apply(&mut t, &mut input, 1, PadEvent::Released(PadButton::South));
    assert!(!input.pressed("jump"));
}

#[test]
fn it_does_not_matter_which_pad_lets_go_first() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 1, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 1, PadEvent::Released(PadButton::South));
    assert!(input.pressed("jump"));
    apply(&mut t, &mut input, 1, PadEvent::Disconnected);
    assert!(input.pressed("jump"));
}

#[test]
fn a_resting_pad_does_not_cancel_a_pushed_one() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Axis(PadAxis::LeftX, 0.0));
    apply(&mut t, &mut input, 1, PadEvent::Axis(PadAxis::LeftX, -1.0));
    assert_eq!(input.axis("x"), -1.0);
    apply(&mut t, &mut input, 2, PadEvent::Axis(PadAxis::LeftX, 0.5));
    assert_eq!(input.axis("x"), -1.0);
}

#[test]
fn unplugging_a_pad_lets_go_of_everything_it_held() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 0, PadEvent::Axis(PadAxis::LeftX, 1.0));
    apply(&mut t, &mut input, 0, PadEvent::Disconnected);
    assert!(!input.pressed("jump"));
    assert_eq!(input.axis("x"), 0.0);
    assert!(input.just_released("jump"));
}

#[test]
fn unplugging_one_pad_leaves_the_other() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 1, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 0, PadEvent::Disconnected);
    assert!(input.pressed("jump"));
}

#[test]
fn syncing_again_is_not_another_press() {
    let mut app = App::new();
    app.world
        .resource_mut::<Input>()
        .bind("jump", PadButton::South);
    let mut t = PadTracker::default();
    t.update(0, PadEvent::Pressed(PadButton::South));
    t.sync(app.world.resource_mut::<Input>());
    app.frame(1.0 / 60.0);
    t.sync(app.world.resource_mut::<Input>());
    assert!(!app.world.resource::<Input>().just_pressed("jump"));
    assert!(app.world.resource::<Input>().pressed("jump"));
}

#[test]
fn pads_leave_the_keyboard_alone() {
    let (mut t, mut input) = pads();
    input.bind("jump", Key::Space);
    input.press(Key::Space);
    apply(&mut t, &mut input, 0, PadEvent::Pressed(PadButton::South));
    apply(&mut t, &mut input, 0, PadEvent::Released(PadButton::South));
    assert!(input.pressed("jump"));
}

#[test]
fn a_pad_still_held_after_losing_focus_comes_back_with_it() {
    let (mut t, mut input) = pads();
    apply(&mut t, &mut input, 0, PadEvent::Axis(PadAxis::LeftX, 1.0));
    input.release_all();
    assert_eq!(input.axis("x"), 0.0);
    t.sync(&mut input);
    assert_eq!(input.axis("x"), 1.0);
}
