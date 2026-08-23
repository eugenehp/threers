//! Tweens: move a value from A to B over time.

use crate::animatable::Animatable;
use crate::easing::Easing;

/// What a tween does when it reaches the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Repeat {
    /// Stop at the end value.
    #[default]
    Once,
    /// Start over, a fixed number of extra times.
    Times(u32),
    /// Never stop.
    Forever,
}

/// Interpolates a value over a duration.
///
/// ```
/// use threers_animation::prelude::*;
///
/// let mut fade = Tween::new(0.0f32, 1.0, 2.0).easing(Easing::CubicOut);
///
/// fade.update(1.0);                       // halfway through
/// assert!(fade.value() > 0.5, "an ease-out is past halfway at the midpoint");
/// assert!(!fade.is_finished());
///
/// fade.update(1.0);
/// assert_eq!(fade.value(), 1.0);
/// assert!(fade.is_finished());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tween<T> {
    from: T,
    to: T,
    duration: f32,
    delay: f32,
    easing: Easing,
    repeat: Repeat,
    yoyo: bool,

    elapsed: f32,
    cycles_done: u32,
    current: T,
    finished: bool,
}

impl<T: Animatable> Tween<T> {
    /// A tween from `from` to `to` over `duration` seconds.
    ///
    /// A zero or negative duration completes on the first update rather than
    /// dividing by zero.
    pub fn new(from: T, to: T, duration: f32) -> Self {
        Self {
            from,
            to,
            duration: duration.max(0.0),
            delay: 0.0,
            easing: Easing::Linear,
            repeat: Repeat::Once,
            yoyo: false,
            elapsed: 0.0,
            cycles_done: 0,
            current: from,
            finished: false,
        }
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    /// Wait this long before starting. The value stays at `from` meanwhile.
    pub fn delay(mut self, seconds: f32) -> Self {
        self.delay = seconds.max(0.0);
        self
    }

    pub fn repeat(mut self, repeat: Repeat) -> Self {
        self.repeat = repeat;
        self
    }

    /// Reverse on every other cycle, so the value ping-pongs instead of
    /// snapping back to the start. Only meaningful with [`Repeat`].
    pub fn yoyo(mut self, yoyo: bool) -> Self {
        self.yoyo = yoyo;
        self
    }

    /// Advance by `dt` seconds and return the new value.
    pub fn update(&mut self, dt: f32) -> T {
        if self.finished || !dt.is_finite() {
            return self.current;
        }
        self.elapsed += dt.max(0.0);

        let active = self.elapsed - self.delay;
        if active < 0.0 {
            self.current = self.from;
            return self.current;
        }

        // A zero-length tween is a jump, not a division by zero.
        if self.duration <= 0.0 {
            self.current = self.to;
            self.finished = true;
            return self.current;
        }

        let mut cycle = (active / self.duration) as u32;
        let mut t = (active % self.duration) / self.duration;

        let total_cycles = match self.repeat {
            Repeat::Once => Some(1),
            Repeat::Times(n) => Some(n + 1),
            Repeat::Forever => None,
        };

        if let Some(limit) = total_cycles {
            if cycle >= limit {
                cycle = limit - 1;
                t = 1.0;
                self.finished = true;
            }
        }
        self.cycles_done = cycle;

        // On a yoyo, odd cycles run backwards. Reversing the *eased* value
        // rather than the raw time keeps the curve's shape mirrored, which is
        // what "ping-pong" is expected to look like.
        let reversed = self.yoyo && cycle % 2 == 1;
        let shaped = self.easing.apply(if reversed { 1.0 - t } else { t });

        self.current = self.from.lerp(self.to, shaped);
        self.current
    }

    /// The value as of the last [`Self::update`], without advancing.
    pub fn value(&self) -> T {
        self.current
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Progress through the current cycle, `0..1`, ignoring easing.
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        let active = (self.elapsed - self.delay).max(0.0);
        ((active % self.duration) / self.duration).clamp(0.0, 1.0)
    }

    /// Completed cycles so far.
    pub fn cycles(&self) -> u32 {
        self.cycles_done
    }

    /// Back to the beginning, keeping the configuration.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.cycles_done = 0;
        self.finished = false;
        self.current = self.from;
    }

    /// Retarget without restarting: the tween continues from where it is toward
    /// a new destination.
    ///
    /// This is what you want when the target moves mid-flight — restarting
    /// instead produces a visible hitch.
    pub fn retarget(&mut self, to: T) {
        self.from = self.current;
        self.to = to;
        self.elapsed = self.delay;
        self.cycles_done = 0;
        self.finished = false;
    }

    pub fn from(&self) -> T {
        self.from
    }

    pub fn to(&self) -> T {
        self.to
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn a_linear_tween_moves_at_a_constant_rate() {
        let mut t = Tween::new(0.0f32, 10.0, 1.0);
        for i in 1..=10 {
            t.update(0.1);
            let want = i as f32;
            assert!((t.value() - want).abs() < 1e-3, "step {i}: {} vs {want}", t.value());
        }
        assert!(t.is_finished());
    }

    #[test]
    fn it_lands_exactly_on_the_end_value() {
        // Floating-point accumulation must not leave it a hair short.
        let mut t = Tween::new(0.0f32, 1.0, 1.0).easing(Easing::CubicInOut);
        for _ in 0..1000 {
            t.update(0.003);
        }
        assert_eq!(t.value(), 1.0);
        assert!(t.is_finished());
    }

    #[test]
    fn a_delay_holds_the_start_value() {
        let mut t = Tween::new(5.0f32, 10.0, 1.0).delay(0.5);
        t.update(0.25);
        assert_eq!(t.value(), 5.0);
        t.update(0.25);
        assert_eq!(t.value(), 5.0);
        t.update(0.5);
        assert!(t.value() > 5.0 && t.value() < 10.0, "{}", t.value());
    }

    #[test]
    fn a_zero_duration_tween_jumps_rather_than_dividing_by_zero() {
        let mut t = Tween::new(0.0f32, 7.0, 0.0);
        assert_eq!(t.update(0.016), 7.0);
        assert!(t.is_finished());
        assert!(t.progress().is_finite());
    }

    #[test]
    fn repeat_restarts_the_requested_number_of_times() {
        let mut t = Tween::new(0.0f32, 1.0, 1.0).repeat(Repeat::Times(2));
        // Three cycles in total.
        for _ in 0..29 {
            t.update(0.1);
        }
        assert!(!t.is_finished(), "should still be in the third cycle");
        t.update(0.2);
        assert!(t.is_finished());
        assert_eq!(t.value(), 1.0);
    }

    #[test]
    fn forever_never_finishes() {
        let mut t = Tween::new(0.0f32, 1.0, 0.5).repeat(Repeat::Forever);
        for _ in 0..1000 {
            t.update(0.05);
            assert!(!t.is_finished());
            assert!(t.value() >= -1e-4 && t.value() <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn a_yoyo_comes_back_instead_of_snapping() {
        let mut t = Tween::new(0.0f32, 1.0, 1.0)
            .repeat(Repeat::Forever)
            .yoyo(true);
        // End of the first cycle: at the far end.
        for _ in 0..10 {
            t.update(0.1);
        }
        assert!(t.value() > 0.9, "{}", t.value());
        // Halfway through the second: on the way back.
        for _ in 0..5 {
            t.update(0.1);
        }
        assert!(t.value() < 0.6, "yoyo did not reverse: {}", t.value());
        // End of the second cycle: back at the start.
        for _ in 0..5 {
            t.update(0.1);
        }
        assert!(t.value() < 0.1, "yoyo did not return: {}", t.value());
    }

    #[test]
    fn retarget_continues_from_where_it_is() {
        let mut t = Tween::new(0.0f32, 10.0, 1.0);
        t.update(0.5);
        let midpoint = t.value();
        assert!((midpoint - 5.0).abs() < 1e-3);

        t.retarget(-10.0);
        // No jump: the next value is still near where it was.
        t.update(0.01);
        assert!((t.value() - midpoint).abs() < 0.3, "retarget jumped to {}", t.value());
        // And it now heads for the new destination.
        for _ in 0..100 {
            t.update(0.02);
        }
        assert_eq!(t.value(), -10.0);
    }

    #[test]
    fn reset_returns_it_to_the_start() {
        let mut t = Tween::new(2.0f32, 8.0, 1.0);
        t.update(2.0);
        assert!(t.is_finished());
        t.reset();
        assert_eq!(t.value(), 2.0);
        assert!(!t.is_finished());
        assert_eq!(t.progress(), 0.0);
    }

    #[test]
    fn it_works_for_vectors_and_survives_nonsense_time() {
        let mut t = Tween::new(Vector3::ZERO, Vector3::new(3.0, 6.0, 9.0), 1.0);
        t.update(0.5);
        assert!((t.value() - Vector3::new(1.5, 3.0, 4.5)).length() < 1e-3);

        // Garbage frame times must not corrupt the tween.
        t.update(f32::NAN);
        t.update(f32::INFINITY);
        t.update(-1.0);
        assert!(t.value().x.is_finite() && t.value().y.is_finite());
    }

    #[test]
    fn an_overshooting_curve_really_overshoots_a_scalar() {
        let mut t = Tween::new(0.0f32, 1.0, 1.0).easing(Easing::BackOut);
        let mut peak: f32 = 0.0;
        for _ in 0..100 {
            peak = peak.max(t.update(0.01));
        }
        assert!(peak > 1.0, "BackOut should overshoot, peaked at {peak}");
        assert_eq!(t.value(), 1.0, "but it must still land exactly");
    }
}
