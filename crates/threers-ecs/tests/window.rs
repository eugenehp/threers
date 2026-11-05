//! The part of the window that can run without one.

use threers_ecs::prelude::*;
use threers_ecs::window::apply_key;
use winit::keyboard::KeyCode;

#[test]
fn winit_key_names_reach_the_same_keys() {
    for (code, key) in [
        (KeyCode::KeyW, Key::KeyW),
        (KeyCode::Space, Key::Space),
        (KeyCode::ArrowUp, Key::ArrowUp),
        (KeyCode::ShiftLeft, Key::ShiftLeft),
        (KeyCode::Digit1, Key::Digit1),
        (KeyCode::F12, Key::F12),
        (KeyCode::Escape, Key::Escape),
    ] {
        let mut input = Input::default();
        apply_key(&mut input, code, true);
        assert!(input.is_down(key), "{code:?}");
        apply_key(&mut input, code, false);
        assert!(!input.is_down(key), "{code:?}");
    }
}

#[test]
fn a_key_with_no_name_here_is_ignored() {
    let mut input = Input::default();
    apply_key(&mut input, KeyCode::MediaPlayPause, true);
}
