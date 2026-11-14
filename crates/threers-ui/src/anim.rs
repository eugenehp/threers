//! Values that move to their targets over time, for interfaces that are rebuilt every frame.
//!
//! A [`Ui`](super::Ui) built each frame has no widget to hold "how far along is this fade". An [`Animator`] is
//! that memory, kept beside the application's state: name each thing that animates (a hover, a panel opening,
//! a tab fading in) with any key, say where it should be with [`Animator::set`], advance it with
//! [`Animator::step`] once a frame, and read [`Animator::get`] while building. A key that has never been set
//! reads as its resting value, so nothing needs registering.
//!
//! Each move is a tween along an [`threers_math::easing::Easing`] over a duration: retargeting in the middle
//! of a move starts from wherever the value is, so a hover that ends early eases back from where it got to.
//!
//! ```
//! use threers_math::easing::Easing;
//! use threers_ui::anim::Animator;
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

use threers_math::easing::Easing;

/// A damped spring, as the web's motion libraries animate with:
/// a move that starts at the speed the value already has, so retargeting
/// half way through bends the path instead of restarting it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// Pull towards the target per unit away from it.
    pub stiffness: f32,
    /// Drag per unit of speed. `2 √(stiffness × mass)` is critical: the
    /// quickest arrival that does not overshoot.
    pub damping: f32,
    pub mass: f32,
}

impl Spring {
    /// `stiffness`, `damping` and `mass` as given (each made positive).
    pub fn new(stiffness: f32, damping: f32, mass: f32) -> Self {
        Spring {
            stiffness: stiffness.max(1e-3),
            damping: damping.max(0.0),
            mass: mass.max(1e-3),
        }
    }

    /// Critically damped, within 0.1% of its target after `seconds` (and
    /// fully at rest a little later).
    pub fn settling_in(seconds: f32) -> Self {
        // A critically damped spring is within 0.1% of its target after
        // about 9.2 / ω.
        let omega = 9.2 / seconds.max(1e-3);
        Spring::new(omega * omega, 2.0 * omega, 1.0)
    }

    /// Quick, with a little overshoot.
    pub fn snappy() -> Self {
        Spring::new(400.0, 28.0, 1.0)
    }

    /// Slow and soft, no overshoot.
    pub fn gentle() -> Self {
        Spring::settling_in(0.6)
    }

    /// The offset from the target and the speed after `dt` seconds,
    /// solved exactly rather than integrated, so a long frame neither
    /// overshoots nor blows up.
    pub fn step(self, x: f32, v: f32, dt: f32) -> (f32, f32) {
        let (x, v, t) = (x as f64, v as f64, dt.max(0.0) as f64);
        let (k, c, m) = (self.stiffness as f64, self.damping as f64, self.mass as f64);
        let w0 = (k / m).sqrt();
        let zeta = c / (2.0 * (k * m).sqrt());
        let (nx, nv) = if (zeta - 1.0).abs() < 1e-6 {
            let a = x;
            let b = v + w0 * x;
            let e = (-w0 * t).exp();
            ((a + b * t) * e, (b - w0 * (a + b * t)) * e)
        } else if zeta < 1.0 {
            let wd = w0 * (1.0 - zeta * zeta).sqrt();
            let a = x;
            let b = (v + zeta * w0 * x) / wd;
            let e = (-zeta * w0 * t).exp();
            let (sn, cs) = (wd * t).sin_cos();
            let pos = e * (a * cs + b * sn);
            let vel = e * ((b * wd - zeta * w0 * a) * cs - (a * wd + zeta * w0 * b) * sn);
            (pos, vel)
        } else {
            let s = (zeta * zeta - 1.0).sqrt();
            let (r1, r2) = (-w0 * (zeta - s), -w0 * (zeta + s));
            let c2 = (v - r1 * x) / (r2 - r1);
            let c1 = x - c2;
            let (e1, e2) = ((r1 * t).exp(), (r2 * t).exp());
            (c1 * e1 + c2 * e2, c1 * r1 * e1 + c2 * r2 * e2)
        };
        (nx as f32, nv as f32)
    }
}

/// Closer than this to the target, and slower than this, a spring has
/// arrived (in the value's own units: a pixel's hundredth, or a 0..1
/// fraction's ten-thousandth, are both well below what shows).
const SPRING_REST: f32 = 1e-4;

#[derive(Clone, Copy, Debug)]
enum Motion {
    Tween {
        from: f32,
        /// 0 at the start of the move, 1 when it has arrived.
        progress: f32,
        /// Seconds the move takes.
        duration: f32,
        easing: Easing,
    },
    Spring {
        value: f32,
        velocity: f32,
        spring: Spring,
    },
}

#[derive(Clone, Copy, Debug)]
struct Track {
    to: f32,
    motion: Motion,
}

impl Track {
    fn value(&self) -> f32 {
        match self.motion {
            Motion::Tween {
                from,
                progress,
                easing,
                ..
            } => from + (self.to - from) * easing.apply(progress.clamp(0.0, 1.0)),
            Motion::Spring { value, .. } => value,
        }
    }

    fn velocity(&self) -> f32 {
        match self.motion {
            Motion::Spring { velocity, .. } => velocity,
            Motion::Tween { .. } => 0.0,
        }
    }

    fn arrived(&self) -> bool {
        match self.motion {
            Motion::Tween { progress, .. } => progress >= 1.0,
            Motion::Spring { value, .. } => value == self.to,
        }
    }

    fn step(&mut self, dt: f32) {
        match &mut self.motion {
            Motion::Tween {
                progress, duration, ..
            } => {
                if *progress < 1.0 {
                    *progress = (*progress + dt.max(0.0) / *duration).min(1.0);
                }
            }
            Motion::Spring {
                value,
                velocity,
                spring,
            } => {
                if *value == self.to {
                    return;
                }
                let (x, v) = spring.step(*value - self.to, *velocity, dt);
                let scale = self.to.abs().max(1.0);
                if x.abs() < SPRING_REST * scale && v.abs() < SPRING_REST * scale {
                    *value = self.to;
                    *velocity = 0.0;
                } else {
                    *value = self.to + x;
                    *velocity = v;
                }
            }
        }
    }

    fn at_rest(value: f32, easing: Easing) -> Self {
        Track {
            to: value,
            motion: Motion::Tween {
                from: value,
                progress: 1.0,
                duration: 1.0,
                easing,
            },
        }
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
    /// [`set`](Self::set) moves by this spring instead of the duration and
    /// easing.
    spring: Option<Spring>,
}

impl<K: Eq + Hash + Clone> Animator<K> {
    /// An animator whose moves take `duration` seconds along `easing`, with 0 as the resting value.
    pub fn new(duration: f32, easing: Easing) -> Self {
        Self {
            tracks: HashMap::new(),
            duration: duration.max(1e-4),
            easing,
            rest: 0.0,
            spring: None,
        }
    }

    /// An animator whose moves are `spring`'s.
    pub fn springy(spring: Spring) -> Self {
        Self {
            spring: Some(spring),
            ..Self::new(0.2, Easing::Linear)
        }
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

    /// How fast the value is moving, in units per second (springs only;
    /// a tween reports 0).
    pub fn velocity(&self, key: &K) -> f32 {
        self.tracks.get(key).map_or(0.0, Track::velocity)
    }

    /// Starts moving `key` to `target` from where it is now, at the animator's speed. A key already heading
    /// there is left alone, so this can be called every frame.
    pub fn set(&mut self, key: K, target: f32) {
        match self.spring {
            Some(spring) => self.set_spring(key, target, spring),
            None => self.set_over(key, target, self.duration, self.easing),
        }
    }

    /// Moves `key` to `target` by `spring`, from where it is and at the
    /// speed it already has.
    pub fn set_spring(&mut self, key: K, target: f32, spring: Spring) {
        let (rest, easing) = (self.rest, self.easing);
        let track = self
            .tracks
            .entry(key)
            .or_insert(Track::at_rest(rest, easing));
        if track.to == target && matches!(track.motion, Motion::Spring { .. }) {
            return;
        }
        let (value, velocity) = (track.value(), track.velocity());
        *track = Track {
            to: target,
            motion: Motion::Spring {
                value,
                velocity,
                spring,
            },
        };
    }

    /// Like [`set`](Self::set), with its own duration and easing.
    pub fn set_over(&mut self, key: K, target: f32, duration: f32, easing: Easing) {
        let rest = self.rest;
        let track = self
            .tracks
            .entry(key)
            .or_insert(Track::at_rest(rest, easing));
        if track.to == target && matches!(track.motion, Motion::Tween { .. }) {
            return;
        }
        *track = Track {
            to: target,
            motion: Motion::Tween {
                from: track.value(),
                progress: 0.0,
                duration: duration.max(1e-4),
                easing,
            },
        };
    }

    /// Puts `key` at `value` at once, with no move.
    pub fn jump(&mut self, key: K, value: f32) {
        self.tracks.insert(key, Track::at_rest(value, self.easing));
    }

    /// Moves every track on by `dt` seconds. Tracks that have arrived and rest where unset keys do are
    /// forgotten, so the table doesn't grow with every hover. Returns whether anything is still moving.
    pub fn step(&mut self, dt: f32) -> bool {
        for t in self.tracks.values_mut() {
            t.step(dt);
        }
        let rest = self.rest;
        self.tracks.retain(|_, t| !(t.arrived() && t.to == rest));
        !self.settled()
    }

    /// Whether nothing is moving.
    pub fn settled(&self) -> bool {
        self.tracks.values().all(Track::arrived)
    }

    /// Whether the app should draw another frame for this animator, asked
    /// rather than told. An idle UI that
    /// stops drawing when this is false draws nothing between inputs.
    pub fn needs_frame(&self) -> bool {
        !self.settled()
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

    #[test]
    fn a_spring_arrives_exactly_and_keeps_its_speed_when_retargeted() {
        let mut a: Animator<u32> = Animator::springy(Spring::settling_in(0.5));
        a.set(1, 100.0);
        let mut t = 0.0;
        while a.step(1.0 / 60.0) {
            t += 1.0 / 60.0;
            assert!(a.get(&1) <= 100.0 + 1e-3, "critically damped: no overshoot");
            assert!(t < 2.0, "never settled");
        }
        assert_eq!(a.get(&1), 100.0);
        // Within 0.1% at 0.5 s; at rest (0.01%, and still) by 0.8 s.
        assert!((0.4..=0.85).contains(&t), "settled after {t} s");
        assert!(!a.needs_frame());

        // Half way back, turned round: it carries on the way it was going
        // for a moment rather than stopping dead.
        a.set(1, 0.0);
        for _ in 0..10 {
            a.step(1.0 / 60.0);
        }
        let (at, speed) = (a.get(&1), a.velocity(&1));
        assert!(speed < 0.0);
        a.set(1, 100.0);
        a.step(1.0 / 60.0);
        assert!(a.get(&1) < at, "momentum is kept through the retarget");
    }

    #[test]
    fn a_spring_step_is_exact_whatever_the_frame_length() {
        let s = Spring::snappy();
        // One long step lands where many short ones do.
        let (x1, v1) = s.step(1.0, 0.0, 0.3);
        let (mut x, mut v) = (1.0, 0.0);
        for _ in 0..300 {
            (x, v) = s.step(x, v, 0.001);
        }
        assert!(
            (x - x1).abs() < 1e-4 && (v - v1).abs() < 1e-3,
            "{x} {x1} {v} {v1}"
        );
        // An underdamped spring overshoots; an overdamped one does not.
        let (under, _) = Spring::new(400.0, 5.0, 1.0).step(1.0, 0.0, 0.15);
        assert!(under < 0.0);
        let mut over = (1.0f32, 0.0f32);
        for _ in 0..100 {
            over = Spring::new(100.0, 60.0, 1.0).step(over.0, over.1, 0.02);
            assert!(over.0 >= 0.0);
        }
    }
}
