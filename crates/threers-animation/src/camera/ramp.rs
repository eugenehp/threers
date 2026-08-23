//! Speed remapping along a normalised parameter.

use crate::easing::Easing;

/// Remap a linear `0..1` time parameter for cinematic speed control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedRamp {
    /// Pass through unchanged (constant parameter speed).
    Linear,
    /// Named ease on the parameter itself.
    Ease(Easing),
    /// Ease-in then ease-out with independent weights (`0..1` each).
    /// `ease_in = 0.2, ease_out = 0.3` spends 20% accelerating, 50% cruise,
    /// 30% decelerating — the classic camera move shape.
    InOut { ease_in: f32, ease_out: f32 },
    /// Smoothstep (Hermite) — soft ramps with zero derivative at ends.
    Smooth,
    /// Smootherstep (Ken Perlin) — even softer end derivatives.
    Smoother,
}

impl Default for SpeedRamp {
    fn default() -> Self {
        Self::Ease(Easing::CubicInOut)
    }
}

impl SpeedRamp {
    /// Map linear `t ∈ 0..1` → shaped parameter still in `0..1`.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::Ease(e) => e.apply(t),
            Self::InOut { ease_in, ease_out } => in_out_cruise(t, ease_in, ease_out),
            Self::Smooth => t * t * (3.0 - 2.0 * t),
            Self::Smoother => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

fn in_out_cruise(t: f32, ease_in: f32, ease_out: f32) -> f32 {
    let a = ease_in.clamp(0.0, 1.0);
    let b = ease_out.clamp(0.0, 1.0 - a);
    let mid = 1.0 - a - b;
    if t <= a {
        if a <= 1e-8 {
            return 0.0;
        }
        // Ease-in over [0, a] covering progress [0, a] of the move's "distance
        // budget" proportional to segment length — use cubic ease-in so we
        // leave rest with zero velocity.
        let u = t / a;
        return a * (u * u * (3.0 - 2.0 * u));
    }
    if t >= 1.0 - b {
        if b <= 1e-8 {
            return 1.0;
        }
        let u = (t - (1.0 - b)) / b;
        let eased = u * u * (3.0 - 2.0 * u);
        return a + mid + b * eased;
    }
    // Constant-speed cruise.
    let u = (t - a) / mid.max(1e-8);
    a + mid * u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_preserved() {
        for ramp in [
            SpeedRamp::Linear,
            SpeedRamp::Ease(Easing::CubicInOut),
            SpeedRamp::InOut {
                ease_in: 0.25,
                ease_out: 0.25,
            },
            SpeedRamp::Smooth,
            SpeedRamp::Smoother,
        ] {
            assert!((ramp.apply(0.0)).abs() < 1e-5);
            assert!((ramp.apply(1.0) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn cruise_is_linear_in_the_middle() {
        let ramp = SpeedRamp::InOut {
            ease_in: 0.2,
            ease_out: 0.2,
        };
        let a = ramp.apply(0.4);
        let b = ramp.apply(0.6);
        // Mid segment should advance roughly linearly with t.
        assert!((b - a - 0.2).abs() < 0.05, "a={a} b={b}");
    }
}
