//! Pads and hardware keyboards on iOS, through Apple's GameController
//! framework, where gilrs has no backend.
//!
//! Polled each frame like gilrs: every extended gamepad (Xbox, PlayStation,
//! MFi) is read whole and the changes go through the same [`PadTracker`],
//! so a game sees the same [`PadButton`]s and [`PadAxis`] values on either.
//! A hardware keyboard (an iPad's Magic Keyboard, any Bluetooth one) comes as
//! the same [`Key`]s a desktop window gives, by position.

use crate::input::{Input, Key, PadAxis, PadButton, PadEvent, PadTracker, Rumble};
use objc2::rc::Retained;
use objc2_game_controller::{
    GCController, GCControllerButtonInput, GCControllerDirectionPad, GCExtendedGamepad,
    GCKeyboard,
};
use std::collections::{HashMap, HashSet};

/// Why there are no pads: never, on iOS, where the framework is always there.
#[derive(Debug)]
pub struct NoPads;

impl std::fmt::Display for NoPads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GameController is unavailable")
    }
}

impl std::error::Error for NoPads {}

/// What a pad was holding at the last poll.
#[derive(Default)]
struct Held {
    buttons: HashSet<PadButton>,
    axes: HashMap<PadAxis, f32>,
}

/// Every connected gamepad and the hardware keyboard, feeding one [`Input`].
pub struct Gamepads {
    tracker: PadTracker,
    /// By the controller's place in the framework's list.
    pads: HashMap<usize, Held>,
    keys: HashSet<Key>,
}

impl Gamepads {
    /// # Errors
    ///
    /// Never on iOS; the signature matches the gilrs one.
    pub fn new() -> Result<Self, Box<NoPads>> {
        Ok(Self {
            tracker: PadTracker::default(),
            pads: HashMap::new(),
            keys: HashSet::new(),
        })
    }

    /// Bring `input` up to date: the pads' changes, and the keyboard's keys.
    /// While `focused` is false nothing reaches the game.
    pub fn poll(&mut self, input: &mut Input, focused: bool) {
        let mut seen = HashSet::new();
        // SAFETY: GameController's class methods and getters, on the main
        // thread the window runs on.
        let controllers = unsafe { GCController::controllers() };
        for (index, controller) in controllers.iter().enumerate() {
            let Some(pad) = (unsafe { controller.extendedGamepad() }) else {
                continue;
            };
            seen.insert(index);
            let now = read(&pad);
            let held = self.pads.entry(index).or_default();
            for button in now.buttons.difference(&held.buttons) {
                self.tracker.update(index, PadEvent::Pressed(*button));
            }
            for button in held.buttons.difference(&now.buttons) {
                self.tracker.update(index, PadEvent::Released(*button));
            }
            for (axis, value) in &now.axes {
                if held.axes.get(axis) != Some(value) {
                    self.tracker.update(index, PadEvent::Axis(*axis, *value));
                }
            }
            *held = now;
        }
        let gone: Vec<usize> = self.pads.keys().filter(|i| !seen.contains(*i)).copied().collect();
        for index in gone {
            self.pads.remove(&index);
            self.tracker.update(index, PadEvent::Disconnected);
        }

        let down = keyboard();
        if focused {
            self.tracker.sync(input);
            for key in down.difference(&self.keys) {
                input.press(*key);
            }
            for key in self.keys.difference(&down) {
                input.release(*key);
            }
        }
        self.keys = down;
        // No rumble through the framework here; take it so it does not pile up.
        let _: Option<Rumble> = input.take_rumble();
    }

    /// How many pads are connected.
    pub fn connected(&self) -> usize {
        self.pads.len()
    }
}

fn pressed(button: &GCControllerButtonInput) -> bool {
    unsafe { button.isPressed() }
}

/// A pad's buttons and axes, now.
fn read(pad: &Retained<GCExtendedGamepad>) -> Held {
    let mut held = Held::default();
    let mut button = |on: bool, which: PadButton| {
        if on {
            held.buttons.insert(which);
        }
    };
    unsafe {
        button(pressed(&pad.buttonA()), PadButton::South);
        button(pressed(&pad.buttonB()), PadButton::East);
        button(pressed(&pad.buttonX()), PadButton::West);
        button(pressed(&pad.buttonY()), PadButton::North);
        button(pressed(&pad.leftShoulder()), PadButton::LeftShoulder);
        button(pressed(&pad.rightShoulder()), PadButton::RightShoulder);
        button(pressed(&pad.leftTrigger()), PadButton::LeftTrigger);
        button(pressed(&pad.rightTrigger()), PadButton::RightTrigger);
        button(pressed(&pad.buttonMenu()), PadButton::Start);
        if let Some(options) = pad.buttonOptions() {
            button(pressed(&options), PadButton::Select);
        }
        if let Some(thumb) = pad.leftThumbstickButton() {
            button(pressed(&thumb), PadButton::LeftThumb);
        }
        if let Some(thumb) = pad.rightThumbstickButton() {
            button(pressed(&thumb), PadButton::RightThumb);
        }
        let dpad: Retained<GCControllerDirectionPad> = pad.dpad();
        button(pressed(&dpad.up()), PadButton::DPadUp);
        button(pressed(&dpad.down()), PadButton::DPadDown);
        button(pressed(&dpad.left()), PadButton::DPadLeft);
        button(pressed(&dpad.right()), PadButton::DPadRight);

        // Up and right are positive, as gilrs gives them.
        let (left, right) = (pad.leftThumbstick(), pad.rightThumbstick());
        held.axes.insert(PadAxis::LeftX, left.xAxis().value());
        held.axes.insert(PadAxis::LeftY, left.yAxis().value());
        held.axes.insert(PadAxis::RightX, right.xAxis().value());
        held.axes.insert(PadAxis::RightY, right.yAxis().value());
        held.axes.insert(PadAxis::LeftTrigger, pad.leftTrigger().value());
        held.axes.insert(PadAxis::RightTrigger, pad.rightTrigger().value());
    }
    held
}

/// The hardware keyboard's keys held now, if there is one.
fn keyboard() -> HashSet<Key> {
    let mut down = HashSet::new();
    // SAFETY: as above.
    let Some(keyboard) = (unsafe { GCKeyboard::coalescedKeyboard() }) else {
        return down;
    };
    let Some(keys) = (unsafe { keyboard.keyboardInput() }) else {
        return down;
    };
    for &(code, key) in KEYS {
        if let Some(button) = unsafe { keys.buttonForKeyCode(code) }
            && pressed(&button)
        {
            down.insert(key);
        }
    }
    down
}

/// Each key by its USB HID usage, which is what a `GCKeyCode` is.
const KEYS: &[(isize, Key)] = &[
    (0x04, Key::KeyA),
    (0x05, Key::KeyB),
    (0x06, Key::KeyC),
    (0x07, Key::KeyD),
    (0x08, Key::KeyE),
    (0x09, Key::KeyF),
    (0x0A, Key::KeyG),
    (0x0B, Key::KeyH),
    (0x0C, Key::KeyI),
    (0x0D, Key::KeyJ),
    (0x0E, Key::KeyK),
    (0x0F, Key::KeyL),
    (0x10, Key::KeyM),
    (0x11, Key::KeyN),
    (0x12, Key::KeyO),
    (0x13, Key::KeyP),
    (0x14, Key::KeyQ),
    (0x15, Key::KeyR),
    (0x16, Key::KeyS),
    (0x17, Key::KeyT),
    (0x18, Key::KeyU),
    (0x19, Key::KeyV),
    (0x1A, Key::KeyW),
    (0x1B, Key::KeyX),
    (0x1C, Key::KeyY),
    (0x1D, Key::KeyZ),
    (0x1E, Key::Digit1),
    (0x1F, Key::Digit2),
    (0x20, Key::Digit3),
    (0x21, Key::Digit4),
    (0x22, Key::Digit5),
    (0x23, Key::Digit6),
    (0x24, Key::Digit7),
    (0x25, Key::Digit8),
    (0x26, Key::Digit9),
    (0x27, Key::Digit0),
    (0x28, Key::Enter),
    (0x29, Key::Escape),
    (0x2A, Key::Backspace),
    (0x2B, Key::Tab),
    (0x2C, Key::Space),
    (0x2D, Key::Minus),
    (0x2E, Key::Equal),
    (0x2F, Key::BracketLeft),
    (0x30, Key::BracketRight),
    (0x31, Key::Backslash),
    (0x33, Key::Semicolon),
    (0x34, Key::Quote),
    (0x35, Key::Backquote),
    (0x36, Key::Comma),
    (0x37, Key::Period),
    (0x38, Key::Slash),
    (0x3A, Key::F1),
    (0x3B, Key::F2),
    (0x3C, Key::F3),
    (0x3D, Key::F4),
    (0x3E, Key::F5),
    (0x3F, Key::F6),
    (0x40, Key::F7),
    (0x41, Key::F8),
    (0x42, Key::F9),
    (0x43, Key::F10),
    (0x44, Key::F11),
    (0x45, Key::F12),
    (0x4C, Key::Delete),
    (0x4F, Key::ArrowRight),
    (0x50, Key::ArrowLeft),
    (0x51, Key::ArrowDown),
    (0x52, Key::ArrowUp),
    (0xE0, Key::ControlLeft),
    (0xE1, Key::ShiftLeft),
    (0xE2, Key::AltLeft),
    (0xE4, Key::ControlRight),
    (0xE5, Key::ShiftRight),
    (0xE6, Key::AltRight),
];
