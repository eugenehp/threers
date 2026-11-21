//! How a key press shows, sounds and feels.
//!
//! Shown: the letter large over the finger ([`KeyFeedback::preview`]), and
//! the key itself pressed in ([`PressStyle`]). Heard: Apple's three clicks —
//! a letter's, Backspace's, and the other keys' — made here as samples
//! ([`KeySound::samples`]), played by
//! [`VirtualKeyboard::play_clicks`](super::VirtualKeyboard::play_clicks)
//! with the `sound` feature or by your own audio. Felt: a light tap
//! ([`Haptic`]) for a platform's haptics to give. Each press is reported
//! to [`VirtualKeyboard::on_feedback`](super::VirtualKeyboard::on_feedback)
//! as a [`KeyPress`] saying which sound and which tap it wants.
//!
//! ```
//! use threers_ui::virtual_keyboard::{KeyFeedback, KeySound, PressStyle};
//!
//! let quiet = KeyFeedback { sound: false, press: PressStyle::Dim, ..KeyFeedback::default() };
//! assert!(quiet.preview);
//! let click = KeySound::Letter.samples(48_000);
//! assert!(click.len() < 48_000 / 10, "short");
//! ```

use super::config::KeyAction;

/// How the pressed key looks while it is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PressStyle {
    /// Unchanged (the preview shows the press).
    None,
    /// Faded.
    Dim,
    /// In the other keys' colour, as Apple's: a letter goes grey, a grey
    /// key goes white.
    #[default]
    Swap,
}

/// Which of Apple's clicks a key makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeySound {
    /// A letter, digit or symbol.
    Letter,
    /// Backspace.
    Delete,
    /// Shift, space, return, the page and language keys.
    Modifier,
}

/// A tap to feel, for a platform's haptics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Haptic {
    /// A key: the lightest tap (`UIImpactFeedbackGenerator(style: .light)`).
    Light,
    /// Holding a key brought its alternates up, or locked Shift.
    Medium,
}

/// How presses show, sound and feel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyFeedback {
    /// The letter pressed shown large over the finger.
    pub preview: bool,
    /// How a pressed key looks (keys with a preview keep their look).
    pub press: PressStyle,
    /// Clicks, asked for in each [`KeyPress`].
    pub sound: bool,
    /// Their loudness, 0 to 1.
    pub volume: f32,
    /// Taps, asked for in each [`KeyPress`].
    pub haptics: bool,
}

impl Default for KeyFeedback {
    fn default() -> Self {
        Self {
            preview: true,
            press: PressStyle::Swap,
            sound: true,
            volume: 0.6,
            haptics: true,
        }
    }
}

impl KeyFeedback {
    /// No preview, no press look, no sound, no taps.
    pub fn none() -> Self {
        Self {
            preview: false,
            press: PressStyle::None,
            sound: false,
            volume: 0.0,
            haptics: false,
        }
    }
}

/// A key pressed, as [`VirtualKeyboard::on_feedback`](super::VirtualKeyboard::on_feedback)
/// is told: what to play and what to feel.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyPress {
    /// The key's name (its label, or its accessible name).
    pub name: String,
    pub action: KeyAction,
    /// The click to play, if sound is on.
    pub sound: Option<KeySound>,
    pub volume: f32,
    /// The tap to give, if haptics are on.
    pub haptic: Option<Haptic>,
    /// Backspace again, held.
    pub repeat: bool,
}

impl KeySound {
    /// The click `action` makes.
    pub fn of(action: &KeyAction) -> Option<Self> {
        Some(match action {
            KeyAction::Text(t) if t == " " => KeySound::Modifier,
            KeyAction::Text(_) => KeySound::Letter,
            KeyAction::Named(n) if n == "Backspace" => KeySound::Delete,
            KeyAction::None | KeyAction::Letters(_) => return None,
            _ => KeySound::Modifier,
        })
    }

    /// The click, mono, at `rate` samples a second, peaking at about 0.5:
    /// a decaying tone with a little noise, as Apple's tock — highest for
    /// a letter, lower for Backspace, lowest for the rest.
    pub fn samples(self, rate: u32) -> Vec<f32> {
        let (freq, noise, decay) = match self {
            KeySound::Letter => (2100.0, 0.55, 0.0055),
            KeySound::Delete => (1500.0, 0.5, 0.0065),
            KeySound::Modifier => (950.0, 0.4, 0.0085),
        };
        let rate = rate.max(8000) as f32;
        let n = (rate * decay * 7.0) as usize;
        // A fixed seed: the same click every time.
        let mut seed = 0x2545_f491u32;
        let mut last = 0.0f32;
        (0..n)
            .map(|i| {
                let t = i as f32 / rate;
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let white = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
                // High-passed, for a dry tick rather than a hiss.
                let hiss = white - last;
                last = white;
                let attack = (t / 0.0004).min(1.0);
                let env = attack * (-t / decay).exp();
                let tone = (std::f32::consts::TAU * freq * t).sin();
                0.5 * env * (tone * (1.0 - noise) + hiss * 0.5 * noise)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clicks_are_short_bounded_and_die_away() {
        for s in [KeySound::Letter, KeySound::Delete, KeySound::Modifier] {
            let c = s.samples(48_000);
            assert!(c.len() > 48 && c.len() < 4_800, "{s:?}: {}", c.len());
            let peak = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak > 0.2 && peak <= 0.6, "{s:?}: {peak}");
            let tail = c[c.len() * 9 / 10..]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(tail < peak * 0.05, "{s:?} dies away: {tail}");
        }
        assert_ne!(
            KeySound::Letter.samples(44_100),
            KeySound::Modifier.samples(44_100)
        );
    }

    #[test]
    fn each_key_makes_apple_s_click() {
        let of = |a: KeyAction| KeySound::of(&a);
        assert_eq!(of(KeyAction::Text("a".into())), Some(KeySound::Letter));
        assert_eq!(of(KeyAction::Text(" ".into())), Some(KeySound::Modifier));
        assert_eq!(
            of(KeyAction::Named("Backspace".into())),
            Some(KeySound::Delete)
        );
        assert_eq!(of(KeyAction::Shift), Some(KeySound::Modifier));
        assert_eq!(of(KeyAction::None), None);
    }
}
