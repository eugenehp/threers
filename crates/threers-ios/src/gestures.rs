//! The shell's input between frames: what comes faster than frames do is
//! merged here, and a pinch kept track of, apart from the view so it can be
//! tested without one.

use threers::ui::input::GesturePhase;

use crate::view::Key;
use crate::KeyModifiers;

/// One input as the shell says it: a [`Key`] and its three numbers.
pub(crate) type Input = (u32, f32, f32, f32);

/// The inputs since the last frame, in order. A pinch's steps that come
/// together are one step, a pan's one move; nothing else is merged.
#[derive(Debug, Default)]
pub(crate) struct Inputs(Vec<Input>);

impl Inputs {
    /// Whether `key` waits for the frame (the rest are the view's size and
    /// the like, taken at once).
    pub(crate) fn waits(key: u32) -> bool {
        [Key::Touch, Key::Pinch, Key::Pan, Key::Keyboard, Key::Active, Key::KeyUp].iter().any(|&k| k as u32 == key)
    }

    pub(crate) fn push(&mut self, next: Input) {
        let (key, a, b, c) = next;
        match self.0.last_mut() {
            Some(last) if key == Key::Pinch as u32 && last.0 == key && is_step(last.1) && is_step(a) => {
                last.1 *= a;
                // where the fingers are now, if said
                if said((b, c)) {
                    (last.2, last.3) = (b, c);
                }
            }
            Some(last) if key == Key::Pan as u32 && last.0 == key => {
                last.1 += a;
                last.2 += b;
            }
            _ => self.0.push(next),
        }
    }

    /// Them all, for the frame.
    pub(crate) fn take(&mut self) -> Vec<Input> {
        std::mem::take(&mut self.0)
    }
}

/// A pinch's step, not its end (which comes after its last step, and before
/// the next pinch's first: merged with either, one of them is lost).
pub(crate) fn is_step(scale: f32) -> bool {
    scale.is_finite() && scale > 0.0
}

/// Whether a place was said (0, 0 is not).
pub(crate) fn said((x, y): (f32, f32)) -> bool {
    x.is_finite() && y.is_finite() && (x > 0.0 || y > 0.0)
}

/// The pinch going on: its scale so far, where it is, and whether the app
/// took it.
#[derive(Debug, Default)]
pub(crate) struct Pinch(Option<(f32, (f32, f32), bool)>);

impl Pinch {
    /// A step of it at `at` in points (not said: where it was, else
    /// `middle`), to the app; whether the app has it.
    pub(crate) fn step(
        &mut self,
        step: f32,
        at: (f32, f32),
        middle: (f32, f32),
        app: impl FnOnce(f32, f32, f32, GesturePhase) -> bool,
    ) -> bool {
        let (x, y) = match self.0 {
            _ if said(at) => at,
            Some((_, was, _)) => was,
            None => middle,
        };
        let (scale, phase) = match self.0 {
            Some((_, _, false)) => return false,
            Some((s, _, true)) => (s * step, GesturePhase::Changed),
            None => (step, GesturePhase::Began),
        };
        let took = app(scale, x, y, phase);
        let mine = took || phase == GesturePhase::Changed;
        self.0 = Some((scale, (x, y), mine));
        mine
    }

    /// The fingers lifted.
    pub(crate) fn over(&mut self, app: impl FnOnce(f32, f32, f32, GesturePhase) -> bool) {
        if let Some((scale, (x, y), true)) = self.0.take() {
            app(scale, x, y, GesturePhase::Ended);
        }
    }
}

/// Seconds a key is held before it repeats, and between its repeats (a
/// Mac's, as it comes).
const REPEAT_AFTER: f64 = 0.5;
const REPEAT_EVERY: f64 = 1.0 / 30.0;

/// The key held down, which repeats.
#[derive(Debug, Default)]
pub(crate) struct KeyHeld(Option<(String, KeyModifiers, f64)>);

impl KeyHeld {
    /// A key came down at `now`: the one that repeats, if it does.
    pub(crate) fn down(&mut self, name: &str, held: KeyModifiers, now: f64) {
        let repeats = !held.meta && !held.control && name != "Escape";
        self.0 = repeats.then(|| (name.to_string(), held, now + REPEAT_AFTER));
    }

    pub(crate) fn up(&mut self) {
        self.0 = None;
    }

    /// The key to repeat now, if it is time (once a frame: none are made up
    /// for after a slow one).
    pub(crate) fn due(&mut self, now: f64) -> Option<(String, KeyModifiers)> {
        let (name, held, at) = self.0.as_mut().filter(|k| now >= k.2)?;
        *at = (*at + REPEAT_EVERY).max(now);
        Some((name.clone(), *held))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOUCH: u32 = Key::Touch as u32;
    const PINCH: u32 = Key::Pinch as u32;
    const PAN: u32 = Key::Pan as u32;
    const KEYBOARD: u32 = Key::Keyboard as u32;

    fn merged(inputs: &[Input]) -> Vec<Input> {
        let mut q = Inputs::default();
        inputs.iter().for_each(|&i| q.push(i));
        let out = q.take();
        assert!(q.take().is_empty(), "taken once");
        out
    }

    /// What an app that takes pinches on its page (from x 10) is told of
    /// frames of inputs.
    fn told(frames: &[&[Input]]) -> Vec<(f32, f32, f32, GesturePhase)> {
        let (mut q, mut pinch, mut told) = (Inputs::default(), Pinch::default(), Vec::new());
        for frame in frames {
            frame.iter().for_each(|&i| q.push(i));
            for (_, a, b, c) in q.take() {
                let app = |s, x, y, p| {
                    told.push((s, x, y, p));
                    x >= 10.0
                };
                if a == 0.0 {
                    pinch.over(app);
                } else {
                    pinch.step(a, (b, c), (400.0, 300.0), app);
                }
            }
        }
        told
    }

    #[test]
    fn steps_in_one_frame_are_one_where_the_fingers_are_now() {
        assert_eq!(merged(&[(PINCH, 1.5, 40.0, 50.0), (PINCH, 2.0, 44.0, 52.0)]), [(PINCH, 3.0, 44.0, 52.0)]);
        // a place not said leaves the one that was
        assert_eq!(merged(&[(PINCH, 1.5, 40.0, 50.0), (PINCH, 2.0, 0.0, 0.0)]), [(PINCH, 3.0, 40.0, 50.0)]);
    }

    #[test]
    fn a_held_key_repeats_till_it_is_let_go() {
        let mut k = KeyHeld::default();
        let none = KeyModifiers::default();
        let count = |k: &mut KeyHeld, from: f64, frames: usize| (0..frames).filter(|i| k.due(from + *i as f64 / 60.0).is_some()).count();
        assert_eq!(count(&mut k, 0.0, 60), 0);
        k.down("ArrowLeft", KeyModifiers { shift: true, ..none }, 0.0);
        assert_eq!(count(&mut k, 0.0, 29), 0, "not at once");
        assert!((28..=32).contains(&count(&mut k, 0.51, 60)));
        assert_eq!(k.due(2.0), Some(("ArrowLeft".into(), KeyModifiers { shift: true, ..none })));
        // a slow frame's are not made up for
        assert!(k.due(9.0).is_some() && k.due(9.01).is_none() || count(&mut k, 9.01, 2) <= 1);
        // another key takes over; let go, none
        k.down("a", none, 10.0);
        assert!(k.due(10.3).is_none() && k.due(10.6).is_some());
        k.up();
        assert_eq!(count(&mut k, 11.0, 60), 0);
        // a shortcut's do not, and stop what did
        for (name, held) in [("v", KeyModifiers { meta: true, ..none }), ("Escape", none)] {
            k.down("a", none, 0.0);
            k.down(name, held, 0.1);
            assert_eq!(count(&mut k, 1.0, 60), 0);
        }
    }

    #[test]
    fn a_pinchs_end_is_not_merged_away() {
        let fast = [(PINCH, 1.5, 40.0, 50.0), (PINCH, 0.0, 0.0, 0.0), (PINCH, 0.5, 60.0, 70.0), (PINCH, 0.0, 0.0, 0.0)];
        assert_eq!(merged(&fast), fast);
    }

    #[test]
    fn a_pan_sums_and_the_rest_keep_their_order() {
        assert_eq!(merged(&[(PAN, 1.0, 2.0, 0.0), (PAN, 3.0, 4.0, 0.0)]), [(PAN, 4.0, 6.0, 0.0)]);
        let mixed = [
            (PINCH, 1.5, 4.0, 5.0),
            (PAN, 1.0, 2.0, 0.0),
            (PINCH, 1.5, 4.0, 5.0),
            (TOUCH, 0.0, 1.0, 1.0),
            (TOUCH, 2.0, 1.0, 1.0),
            (KEYBOARD, 97.0, 0.0, 0.0),
            (KEYBOARD, 97.0, 0.0, 0.0),
        ];
        assert_eq!(merged(&mixed), mixed);
    }

    #[test]
    fn a_bad_scale_spoils_no_good_one() {
        let out = merged(&[(PINCH, 1.5, 4.0, 5.0), (PINCH, f32::NAN, 4.0, 5.0), (PINCH, 2.0, 4.0, 5.0)]);
        assert_eq!((out[0].1, out[2].1), (1.5, 2.0));
    }

    #[test]
    fn a_fast_pinch_begins_on_the_page_and_ends() {
        use GesturePhase::*;
        // all of it in one frame, and the next pinch begun in it too
        let frame = [
            (PINCH, 1.2, 441.0, 417.0),
            (PINCH, 1.25, 442.0, 417.0),
            (PINCH, 0.0, 0.0, 0.0),
            (PINCH, 0.8, 440.0, 416.0),
        ];
        assert_eq!(
            told(&[&frame, &[(PINCH, 0.5, 440.0, 416.0), (PINCH, 0.0, 0.0, 0.0)]]),
            [
                (1.5, 442.0, 417.0, Began),
                (1.5, 442.0, 417.0, Ended),
                (0.8, 440.0, 416.0, Began),
                (0.4, 440.0, 416.0, Changed),
                (0.4, 440.0, 416.0, Ended),
            ]
        );
    }

    #[test]
    fn a_pinch_not_said_where_is_where_it_was_or_the_middle() {
        use GesturePhase::*;
        assert_eq!(
            told(&[&[(PINCH, 2.0, 0.0, 0.0)], &[(PINCH, 1.5, 50.0, 60.0)], &[(PINCH, 1.0, 0.0, 0.0)]]),
            [(2.0, 400.0, 300.0, Began), (3.0, 50.0, 60.0, Changed), (3.0, 50.0, 60.0, Changed)]
        );
    }

    #[test]
    fn a_pinch_the_app_leaves_is_left_till_the_next() {
        use GesturePhase::*;
        let off = [(PINCH, 2.0, 5.0, 60.0)];
        assert_eq!(
            told(&[&off, &off, &[(PINCH, 0.0, 0.0, 0.0)], &[(PINCH, 2.0, 50.0, 60.0)]]),
            [(2.0, 5.0, 60.0, Began), (2.0, 50.0, 60.0, Began)]
        );
    }

    #[test]
    fn the_views_own_settings_do_not_wait() {
        for k in [Key::Resize, Key::Scale, Key::Insets] {
            assert!(!Inputs::waits(k as u32));
        }
        for k in [TOUCH, PINCH, PAN, KEYBOARD] {
            assert!(Inputs::waits(k));
        }
    }
}
