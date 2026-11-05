//! The gilrs side. A real pad cannot be pressed from a test, so this covers what
//! can be checked without one: the mapping, and that the service starts and polls.

use gilrs::{Axis, Button};
use threers_ecs::gamepad::{pad_axis, pad_button, trigger, Gamepads};
use threers_ecs::prelude::*;

#[test]
fn every_pad_button_has_a_gilrs_source() {
    let all = [
        Button::South,
        Button::East,
        Button::West,
        Button::North,
        Button::LeftTrigger,
        Button::RightTrigger,
        Button::LeftTrigger2,
        Button::RightTrigger2,
        Button::LeftThumb,
        Button::RightThumb,
        Button::Start,
        Button::Select,
        Button::DPadUp,
        Button::DPadDown,
        Button::DPadLeft,
        Button::DPadRight,
    ];
    let mapped: Vec<_> = all.iter().filter_map(|&b| pad_button(b)).collect();
    for button in PadButton::ALL {
        assert_eq!(
            mapped.iter().filter(|&&m| m == button).count(),
            1,
            "{button:?}"
        );
    }
}

#[test]
fn gilrs_calls_the_bumper_a_trigger() {
    assert_eq!(
        pad_button(Button::LeftTrigger),
        Some(PadButton::LeftShoulder)
    );
    assert_eq!(
        pad_button(Button::RightTrigger),
        Some(PadButton::RightShoulder)
    );
    assert_eq!(
        pad_button(Button::LeftTrigger2),
        Some(PadButton::LeftTrigger)
    );
    assert_eq!(
        pad_button(Button::RightTrigger2),
        Some(PadButton::RightTrigger)
    );
}

#[test]
fn only_real_triggers_have_analog_travel() {
    assert_eq!(trigger(Button::LeftTrigger2), Some(PadAxis::LeftTrigger));
    assert_eq!(trigger(Button::RightTrigger2), Some(PadAxis::RightTrigger));
    assert_eq!(trigger(Button::LeftTrigger), None);
    assert_eq!(trigger(Button::South), None);
}

#[test]
fn sticks_map_and_unknown_input_is_dropped() {
    assert_eq!(pad_axis(Axis::LeftStickX), Some(PadAxis::LeftX));
    assert_eq!(pad_axis(Axis::LeftStickY), Some(PadAxis::LeftY));
    assert_eq!(pad_axis(Axis::RightStickX), Some(PadAxis::RightX));
    assert_eq!(pad_axis(Axis::RightStickY), Some(PadAxis::RightY));
    assert_eq!(pad_axis(Axis::Unknown), None);
    assert_eq!(pad_button(Button::Unknown), None);
}

#[test]
fn the_service_starts_and_polls() {
    let mut pads = Gamepads::new().expect("gamepad service");
    let mut input = Input::default();
    for _ in 0..3 {
        pads.poll(&mut input, true);
    }
    println!("{} connected", pads.connected());
}
