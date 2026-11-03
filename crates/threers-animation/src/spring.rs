//! Springs: motion defined by a destination rather than a duration.
//!
//! A tween needs to know how long it will take. A spring does not — you move the
//! target and it chases, and if you move the target again mid-flight it simply
//! adjusts. That makes springs the right tool for anything that reacts to input:
//! camera follow, UI that responds to a cursor, a turret tracking a target.
//!
//! The motion is a damped harmonic oscillator integrated semi-implicitly, which
//! is stable at any step size — unlike the explicit form, which explodes when
//! stiffness times step exceeds a threshold.

use crate::animatable::Animatable;
use threers::math::Vector3;

/// A value that can also be scaled and added, which springs need in order to
/// accumulate velocity. [`Animatable`] alone is not enough — a rotation can be
/// interpolated but has no meaningful sum.
pub trait SpringValue: Animatable {
    fn zero() -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn scale(self, factor: f32) -> Self;
    /// Magnitude, for the settled test.
    fn magnitude(self) -> f32;
}

impl SpringValue for f32 {
    fn zero() -> Self {
        0.0
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn scale(self, factor: f32) -> Self {
        self * factor
    }
    fn magnitude(self) -> f32 {
        self.abs()
    }
}

impl SpringValue for Vector3 {
    fn zero() -> Self {
        Vector3::ZERO
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn scale(self, factor: f32) -> Self {
        self * factor
    }
    fn magnitude(self) -> f32 {
        self.length()
    }
}

/// A damped spring chasing a target.
///
/// ```
/// use threers_animation::prelude::*;
///
/// // Reaches its target in about a third of a second, without overshooting.
/// let mut follow = Spring::critically_damped(0.0f32, 3.0);
/// follow.set_target(10.0);
///
/// for _ in 0..60 {
///     follow.update(1.0 / 60.0);
/// }
/// assert!((follow.value() - 10.0).abs() < 0.1);
/// assert!(follow.is_settled());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring<T> {
    value: T,
    velocity: T,
    target: T,
    /// Undamped angular frequency. Higher is snappier.
    pub angular_frequency: f32,
    /// 1.0 is critically damped — the fastest approach with no overshoot.
    /// Below 1 oscillates, above 1 crawls in.
    pub damping_ratio: f32,
    /// Distance below which the spring counts as arrived.
    pub rest_threshold: f32,
}

impl<T: SpringValue> Spring<T> {
    /// A spring with an explicit frequency and damping ratio.
    pub fn new(value: T, angular_frequency: f32, damping_ratio: f32) -> Self {
        Self {
            value,
            velocity: T::zero(),
            target: value,
            angular_frequency: angular_frequency.max(0.0),
            damping_ratio: damping_ratio.max(0.0),
            rest_threshold: 1e-3,
        }
    }

    /// The common case: arrives as fast as possible without ever overshooting.
    ///
    /// `frequency` is in cycles per second — roughly, `1 / frequency` seconds to
    /// arrive. 3 to 6 suits most UI; 1 to 2 suits a lazy camera.
    pub fn critically_damped(value: T, frequency: f32) -> Self {
        Self::new(value, frequency * std::f32::consts::TAU, 1.0)
    }

    /// Springy, with visible overshoot. `bounciness` in `0..1`, higher wobbles more.
    pub fn bouncy(value: T, frequency: f32, bounciness: f32) -> Self {
        Self::new(
            value,
            frequency * std::f32::consts::TAU,
            (1.0 - bounciness.clamp(0.0, 0.99)).max(0.05),
        )
    }

    pub fn set_target(&mut self, target: T) {
        self.target = target;
    }

    pub fn target(&self) -> T {
        self.target
    }

    pub fn value(&self) -> T {
        self.value
    }

    pub fn velocity(&self) -> T {
        self.velocity
    }

    /// Jump to a value, cancelling any motion.
    pub fn reset_to(&mut self, value: T) {
        self.value = value;
        self.target = value;
        self.velocity = T::zero();
    }

    /// Kick the spring — an impulse, for hit reactions and the like.
    pub fn nudge(&mut self, velocity: T) {
        self.velocity = self.velocity.add(velocity);
    }

    /// Advance and return the new value.
    pub fn update(&mut self, dt: f32) -> T {
        if !dt.is_finite() || dt <= 0.0 {
            return self.value;
        }
        // Long frames are split, because a spring integrated in one huge step is
        // inaccurate even when it is stable.
        let max_step = 1.0 / 60.0;
        let mut remaining = dt.min(1.0);
        while remaining > 0.0 {
            let step = remaining.min(max_step);
            self.integrate(step);
            remaining -= step;
        }
        self.value
    }

    fn integrate(&mut self, dt: f32) {
        let w = self.angular_frequency;
        let z = self.damping_ratio;

        // Backward Euler on *both* terms.
        //
        // Taking the damping implicitly and the spring explicitly is the usual
        // shortcut and it is half a scheme: the restoring force is then computed
        // from where the spring *was*, so it keeps pushing after the target is
        // reached and a critically damped spring — which by definition must not
        // — overshoots. Solving
        //
        //     v' = v + dt(-w^2 x' - 2zw v'),   x' = x + dt v'
        //
        // for v' puts the extra `w^2 dt^2` in the denominator, and that term is
        // the whole difference. It costs one multiply.
        let displacement = self.value.sub(self.target);
        let acceleration = displacement.scale(-w * w);
        let numerator = self.velocity.add(acceleration.scale(dt));
        let denominator = 1.0 + 2.0 * z * w * dt + w * w * dt * dt;

        self.velocity = numerator.scale(1.0 / denominator);
        self.value = self.value.add(self.velocity.scale(dt));
    }

    /// Whether the spring has effectively arrived and stopped.
    pub fn is_settled(&self) -> bool {
        self.value.sub(self.target).magnitude() < self.rest_threshold
            && self.velocity.magnitude() < self.rest_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_critically_damped_spring_never_overshoots() {
        let mut s = Spring::critically_damped(0.0f32, 4.0);
        s.set_target(1.0);
        let mut peak: f32 = 0.0;
        for _ in 0..300 {
            peak = peak.max(s.update(1.0 / 60.0));
        }
        assert!(peak <= 1.0 + 1e-3, "overshot to {peak}");
        assert!((s.value() - 1.0).abs() < 1e-2);
        assert!(s.is_settled());
    }

    #[test]
    fn a_bouncy_spring_does_overshoot_then_settles() {
        let mut s = Spring::bouncy(0.0f32, 3.0, 0.8);
        s.set_target(1.0);
        let mut peak: f32 = 0.0;
        for _ in 0..600 {
            peak = peak.max(s.update(1.0 / 60.0));
        }
        assert!(peak > 1.05, "a bouncy spring should overshoot, peaked at {peak}");
        assert!((s.value() - 1.0).abs() < 1e-2, "but it must settle: {}", s.value());
    }

    #[test]
    fn it_is_stable_at_absurd_stiffness_and_step_sizes() {
        // The explicit form blows up here; the semi-implicit one must not.
        for frequency in [1.0f32, 10.0, 100.0, 1000.0] {
            for dt in [1.0f32 / 240.0, 1.0 / 60.0, 1.0 / 10.0, 0.5] {
                let mut s = Spring::critically_damped(0.0f32, frequency);
                s.set_target(1.0);
                for _ in 0..200 {
                    s.update(dt);
                }
                assert!(
                    s.value().is_finite() && s.value().abs() < 10.0,
                    "frequency {frequency} at dt {dt} diverged to {}",
                    s.value()
                );
            }
        }
    }

    #[test]
    fn retargeting_mid_flight_is_smooth() {
        let mut s = Spring::critically_damped(0.0f32, 3.0);
        s.set_target(10.0);
        for _ in 0..20 {
            s.update(1.0 / 60.0);
        }
        let before = s.value();
        s.set_target(-5.0);
        let after = s.update(1.0 / 60.0);
        // No jump: it turns around from where it was, rather than teleporting
        // toward the new target. It does *move* — at 3 Hz a 15-unit reversal is
        // an acceleration of w^2 * 15 ~= 5300 units/s^2, so a frame of it is
        // most of a unit — but that is velocity times dt, which is the spring
        // working, not a discontinuity. What would fail here is a spring that
        // restarted from the new target instead of from its own state.
        assert!(
            (after - before).abs() < 1.0,
            "jumped from {before} to {after}"
        );

        for _ in 0..600 {
            s.update(1.0 / 60.0);
        }
        assert!((s.value() + 5.0).abs() < 1e-2);
    }

    #[test]
    fn a_nudge_moves_it_without_moving_the_target() {
        let mut s = Spring::critically_damped(0.0f32, 2.0);
        s.nudge(5.0);
        assert_eq!(s.target(), 0.0);
        let mut peak: f32 = 0.0;
        for _ in 0..300 {
            peak = peak.max(s.update(1.0 / 60.0));
        }
        assert!(peak > 0.1, "the nudge did nothing");
        assert!(s.value().abs() < 1e-2, "it should return to the target");
    }

    #[test]
    fn vectors_work_the_same_way() {
        use threers::math::Vector3;
        let mut s = Spring::critically_damped(Vector3::ZERO, 4.0);
        s.set_target(Vector3::new(1.0, 2.0, 3.0));
        for _ in 0..300 {
            s.update(1.0 / 60.0);
        }
        assert!((s.value() - Vector3::new(1.0, 2.0, 3.0)).length() < 1e-2);
        assert!(s.is_settled());
    }

    #[test]
    fn nonsense_time_is_ignored() {
        let mut s = Spring::critically_damped(1.0f32, 3.0);
        s.set_target(2.0);
        s.update(f32::NAN);
        s.update(f32::INFINITY);
        s.update(-1.0);
        s.update(0.0);
        assert_eq!(s.value(), 1.0);
    }

    #[test]
    fn reset_cancels_motion_entirely() {
        let mut s = Spring::critically_damped(0.0f32, 3.0);
        s.set_target(100.0);
        for _ in 0..10 {
            s.update(1.0 / 60.0);
        }
        assert!(s.velocity().abs() > 0.0);
        s.reset_to(7.0);
        assert_eq!(s.value(), 7.0);
        assert_eq!(s.velocity(), 0.0);
        assert!(s.is_settled());
    }
}
