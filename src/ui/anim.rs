//! Values that move to their targets over time, for interfaces that are rebuilt every frame.
//!
//! A [`Ui`](super::Ui) built each frame has no widget to hold "how far along is this fade". An [`Animator`] is
//! that memory, kept beside the application's state: name each thing that animates (a hover, a panel opening,
//! a tab fading in) with any key, say where it should be with [`Animator::set`], advance it with
//! [`Animator::step`] once a frame, and read [`Animator::get`] while building. A key that has never been set
//! reads as its resting value, so nothing needs registering.
//!
//! Each move is a tween along an [`Easing`](crate::easing::Easing) over a duration: retargeting in the middle
//! of a move starts from wherever the value is, so a hover that ends early eases back from where it got to.
//!
//! ```
//! use threers::easing::Easing;
//! use threers::ui::anim::Animator;
//!
//! let mut fades: Animator<&str> = Animator::new(0.2, Easing::QuadOut);
//! assert_eq!(fades.get(&"hover"), 0.0);
//! fades.set("hover", 1.0);
//! fades.step(0.1);
//! let halfway = fades.get(&"hover");
//! assert!(halfway > 0.5 && halfway < 1.0); // eased out: past the middle at half the time
//! fades.step(0.2);
//! assert_eq!(fades.get(&"hover"), 1.0);
//! assert!(fades.settled());
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use crate::easing::Easing;

#[derive(Clone, Copy, Debug)]
struct Track {
    from: f32,
    to: f32,
    /// 0 at the start of the move, 1 when it has arrived.
    progress: f32,
    /// Seconds the move takes.
    duration: f32,
    easing: Easing,
}

impl Track {
    fn value(&self) -> f32 {
        self.from + (self.to - self.from) * self.easing.apply(self.progress.clamp(0.0, 1.0))
    }

    fn arrived(&self) -> bool {
        self.progress >= 1.0
    }
}

/// Named values moving to targets. See the [module documentation](self).
#[derive(Clone, Debug)]
pub struct Animator<K: Eq + Hash> {
    tracks: HashMap<K, Track>,
    duration: f32,
    easing: Easing,
    /// What a key that was never set reads as.
    rest: f32,
}

impl<K: Eq + Hash + Clone> Animator<K> {
    /// An animator whose moves take `duration` seconds along `easing`, with 0 as the resting value.
    pub fn new(duration: f32, easing: Easing) -> Self {
        Self { tracks: HashMap::new(), duration: duration.max(1e-4), easing, rest: 0.0 }
    }

    /// Sets what an unset key reads as (default 0).
    pub fn with_rest(mut self, rest: f32) -> Self {
        self.rest = rest;
        self
    }

    /// The value now.
    pub fn get(&self, key: &K) -> f32 {
        self.tracks.get(key).map_or(self.rest, Track::value)
    }

    /// Where the value is going (its value now if it isn't moving).
    pub fn target(&self, key: &K) -> f32 {
        self.tracks.get(key).map_or(self.rest, |t| t.to)
    }

    /// Starts moving `key` to `target` from where it is now, at the animator's speed. A key already heading
    /// there is left alone, so this can be called every frame.
    pub fn set(&mut self, key: K, target: f32) {
        self.set_over(key, target, self.duration, self.easing);
    }

    /// Like [`set`](Self::set), with its own duration and easing.
    pub fn set_over(&mut self, key: K, target: f32, duration: f32, easing: Easing) {
        let rest = self.rest;
        let track = self.tracks.entry(key).or_insert(Track { from: rest, to: rest, progress: 1.0, duration, easing });
        if track.to == target {
            return;
        }
        *track = Track { from: track.value(), to: target, progress: 0.0, duration: duration.max(1e-4), easing };
    }

    /// Puts `key` at `value` at once, with no move.
    pub fn jump(&mut self, key: K, value: f32) {
        self.tracks.insert(key, Track { from: value, to: value, progress: 1.0, duration: self.duration, easing: self.easing });
    }

    /// Moves every track on by `dt` seconds. Tracks that have arrived and rest where unset keys do are
    /// forgotten, so the table doesn't grow with every hover. Returns whether anything is still moving.
    pub fn step(&mut self, dt: f32) -> bool {
        for t in self.tracks.values_mut() {
            if !t.arrived() {
                t.progress = (t.progress + dt.max(0.0) / t.duration).min(1.0);
            }
        }
        let rest = self.rest;
        self.tracks.retain(|_, t| !(t.arrived() && t.to == rest));
        !self.settled()
    }

    /// Whether nothing is moving.
    pub fn settled(&self) -> bool {
        self.tracks.values().all(Track::arrived)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear() -> Animator<u32> {
        Animator::new(1.0, Easing::Linear)
    }

    #[test]
    fn unset_keys_rest_and_a_move_arrives_exactly() {
        let mut a = linear();
        assert_eq!(a.get(&7), 0.0);
        a.set(7, 2.0);
        assert!(!a.settled());
        a.step(0.5);
        assert!((a.get(&7) - 1.0).abs() < 1e-6);
        assert!(a.step(0.25));
        assert!(!a.step(0.5));
        assert_eq!(a.get(&7), 2.0);
        assert!(a.settled());
    }

    #[test]
    fn retargeting_starts_from_where_the_value_is() {
        let mut a = linear();
        a.set(1, 1.0);
        a.step(0.5);
        // Turned back at the halfway point: it eases back from 0.5, not from 1.
        a.set(1, 0.0);
        assert!((a.get(&1) - 0.5).abs() < 1e-6);
        a.step(0.5);
        assert!((a.get(&1) - 0.25).abs() < 1e-6);
        a.step(1.0);
        assert_eq!(a.get(&1), 0.0);
    }

    #[test]
    fn setting_the_same_target_every_frame_does_not_restart_the_move() {
        let mut a = linear();
        for _ in 0..4 {
            a.set(1, 1.0);
            a.step(0.2);
        }
        assert!((a.get(&1) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn a_track_that_is_back_at_rest_is_forgotten() {
        let mut a = linear();
        a.set(1, 1.0);
        a.step(1.0);
        a.set(1, 0.0);
        a.step(1.0);
        assert!(a.tracks.is_empty());
        let mut b = Animator::<u32>::new(1.0, Easing::Linear).with_rest(1.0);
        assert_eq!(b.get(&3), 1.0);
        b.set(3, 0.0);
        b.step(1.0);
        assert_eq!(b.get(&3), 0.0);
    }

    #[test]
    fn jump_has_no_move_and_easing_shapes_it() {
        let mut a = Animator::<u32>::new(1.0, Easing::QuadIn);
        a.jump(1, 0.8);
        assert_eq!(a.get(&1), 0.8);
        assert!(a.settled());
        a.set(2, 1.0);
        a.step(0.5);
        assert!(a.get(&2) < 0.5, "ease-in is slow at first");
    }
}
