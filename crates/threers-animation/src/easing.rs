//! Easing curves.
//!
//! Every curve maps `0..1` to `0..1`. What differs is the *shape* between those
//! endpoints, and that shape is most of what makes motion read as deliberate
//! rather than mechanical.
//!
//! The names follow the Penner set that CSS, three.js and every tween library
//! since have used, so a curve chosen in a design tool transfers directly.

/// A named easing curve.
///
/// ```
/// use threers_animation::prelude::*;
///
/// // Every curve passes through both ends unchanged.
/// for easing in Easing::ALL {
///     assert!((easing.apply(0.0) - 0.0).abs() < 1e-5, "{easing:?} at 0");
///     assert!((easing.apply(1.0) - 1.0).abs() < 1e-5, "{easing:?} at 1");
/// }
///
/// // Ease-out starts fast: it is already past halfway at the halfway point.
/// assert!(Easing::QuadOut.apply(0.5) > 0.5);
/// assert!(Easing::QuadIn.apply(0.5) < 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// No easing. Constant speed, and the only curve that looks wrong on almost
    /// everything physical.
    #[default]
    Linear,

    QuadIn,
    QuadOut,
    QuadInOut,

    CubicIn,
    CubicOut,
    CubicInOut,

    QuartIn,
    QuartOut,
    QuartInOut,

    QuintIn,
    QuintOut,
    QuintInOut,

    SineIn,
    SineOut,
    SineInOut,

    ExpoIn,
    ExpoOut,
    ExpoInOut,

    CircIn,
    CircOut,
    CircInOut,

    /// Overshoots slightly before settling — anticipation.
    BackIn,
    BackOut,
    BackInOut,

    /// Overshoots and oscillates. Springy, and easy to overuse.
    ElasticIn,
    ElasticOut,
    ElasticInOut,

    /// Bounces on arrival, like something dropped.
    BounceIn,
    BounceOut,
    BounceInOut,

    /// Snaps to the end immediately.
    StepStart,
    /// Holds until the very end, then snaps.
    StepEnd,
}

impl Easing {
    /// Every curve, for iteration and property tests.
    pub const ALL: [Easing; 33] = [
        Easing::Linear,
        Easing::QuadIn,
        Easing::QuadOut,
        Easing::QuadInOut,
        Easing::CubicIn,
        Easing::CubicOut,
        Easing::CubicInOut,
        Easing::QuartIn,
        Easing::QuartOut,
        Easing::QuartInOut,
        Easing::QuintIn,
        Easing::QuintOut,
        Easing::QuintInOut,
        Easing::SineIn,
        Easing::SineOut,
        Easing::SineInOut,
        Easing::ExpoIn,
        Easing::ExpoOut,
        Easing::ExpoInOut,
        Easing::CircIn,
        Easing::CircOut,
        Easing::CircInOut,
        Easing::BackIn,
        Easing::BackOut,
        Easing::BackInOut,
        Easing::ElasticIn,
        Easing::ElasticOut,
        Easing::ElasticInOut,
        Easing::BounceIn,
        Easing::BounceOut,
        Easing::BounceInOut,
        Easing::StepStart,
        Easing::StepEnd,
    ];

    /// Shape a normalised time. `t` is clamped to `0..1`.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        use std::f32::consts::PI;
        match self {
            Self::Linear => t,

            Self::QuadIn => t * t,
            Self::QuadOut => t * (2.0 - t),
            Self::QuadInOut => in_out(t, |x| x * x),

            Self::CubicIn => t * t * t,
            Self::CubicOut => 1.0 + (t - 1.0).powi(3),
            Self::CubicInOut => in_out(t, |x| x * x * x),

            Self::QuartIn => t.powi(4),
            Self::QuartOut => 1.0 - (t - 1.0).powi(4),
            Self::QuartInOut => in_out(t, |x| x.powi(4)),

            Self::QuintIn => t.powi(5),
            Self::QuintOut => 1.0 + (t - 1.0).powi(5),
            Self::QuintInOut => in_out(t, |x| x.powi(5)),

            Self::SineIn => 1.0 - (t * PI * 0.5).cos(),
            Self::SineOut => (t * PI * 0.5).sin(),
            Self::SineInOut => -((PI * t).cos() - 1.0) * 0.5,

            Self::ExpoIn => {
                if t == 0.0 {
                    0.0
                } else {
                    (2.0f32).powf(10.0 * (t - 1.0))
                }
            }
            Self::ExpoOut => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - (2.0f32).powf(-10.0 * t)
                }
            }
            Self::ExpoInOut => in_out(t, |x| {
                if x == 0.0 {
                    0.0
                } else {
                    (2.0f32).powf(10.0 * (x - 1.0))
                }
            }),

            Self::CircIn => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
            Self::CircOut => (1.0 - (t - 1.0) * (t - 1.0)).max(0.0).sqrt(),
            Self::CircInOut => in_out(t, |x| 1.0 - (1.0 - x * x).max(0.0).sqrt()),

            Self::BackIn => back_in(t),
            Self::BackOut => 1.0 - back_in(1.0 - t),
            Self::BackInOut => in_out(t, back_in),

            Self::ElasticIn => 1.0 - elastic_out(1.0 - t),
            Self::ElasticOut => elastic_out(t),
            Self::ElasticInOut => in_out(t, |x| 1.0 - elastic_out(1.0 - x)),

            Self::BounceIn => 1.0 - bounce_out(1.0 - t),
            Self::BounceOut => bounce_out(t),
            Self::BounceInOut => in_out(t, |x| 1.0 - bounce_out(1.0 - x)),

            Self::StepStart => {
                if t > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::StepEnd => {
                if t >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Whether the curve leaves `0..1` on the way. Worth knowing before easing
    /// something that must not overshoot — a scale that dips below zero flips
    /// the mesh inside out.
    pub fn overshoots(self) -> bool {
        matches!(
            self,
            Self::BackIn
                | Self::BackOut
                | Self::BackInOut
                | Self::ElasticIn
                | Self::ElasticOut
                | Self::ElasticInOut
        )
    }
}

/// Mirror an ease-in curve into a symmetric ease-in-out.
fn in_out(t: f32, f: impl Fn(f32) -> f32) -> f32 {
    if t < 0.5 {
        f(t * 2.0) * 0.5
    } else {
        1.0 - f((1.0 - t) * 2.0) * 0.5
    }
}

fn back_in(t: f32) -> f32 {
    // 1.70158 is Penner's constant: a 10% overshoot.
    const C: f32 = 1.70158;
    t * t * ((C + 1.0) * t - C)
}

fn elastic_out(t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    const PERIOD: f32 = 0.3;
    let s = PERIOD / 4.0;
    (2.0f32).powf(-10.0 * t) * ((t - s) * (2.0 * std::f32::consts::PI) / PERIOD).sin() + 1.0
}

fn bounce_out(t: f32) -> f32 {
    const N: f32 = 7.5625;
    const D: f32 = 2.75;
    if t < 1.0 / D {
        N * t * t
    } else if t < 2.0 / D {
        let t = t - 1.5 / D;
        N * t * t + 0.75
    } else if t < 2.5 / D {
        let t = t - 2.25 / D;
        N * t * t + 0.9375
    } else {
        let t = t - 2.625 / D;
        N * t * t + 0.984375
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_curve_hits_both_endpoints() {
        for e in Easing::ALL {
            assert!((e.apply(0.0)).abs() < 1e-5, "{e:?} at t=0 gave {}", e.apply(0.0));
            assert!((e.apply(1.0) - 1.0).abs() < 1e-5, "{e:?} at t=1 gave {}", e.apply(1.0));
        }
    }

    #[test]
    fn every_curve_is_finite_across_the_range() {
        for e in Easing::ALL {
            for i in 0..=100 {
                let v = e.apply(i as f32 / 100.0);
                assert!(v.is_finite(), "{e:?} produced {v}");
            }
        }
    }

    #[test]
    fn input_is_clamped_rather_than_extrapolated() {
        for e in Easing::ALL {
            assert_eq!(e.apply(-5.0), e.apply(0.0), "{e:?} below range");
            assert_eq!(e.apply(5.0), e.apply(1.0), "{e:?} above range");
        }
    }

    #[test]
    fn non_overshooting_curves_stay_inside_the_unit_range() {
        for e in Easing::ALL.iter().filter(|e| !e.overshoots()) {
            for i in 0..=100 {
                let v = e.apply(i as f32 / 100.0);
                assert!(
                    (-1e-4..=1.0 + 1e-4).contains(&v),
                    "{e:?} left the range at t={}: {v}",
                    i as f32 / 100.0
                );
            }
        }
    }

    #[test]
    fn overshooting_curves_really_do_overshoot() {
        // Otherwise `overshoots()` is lying, and someone will trust it.
        for e in Easing::ALL.iter().filter(|e| e.overshoots()) {
            let escaped = (0..=100)
                .map(|i| e.apply(i as f32 / 100.0))
                .any(|v| !(-1e-3..=1.0 + 1e-3).contains(&v));
            assert!(escaped, "{e:?} claims to overshoot but never does");
        }
    }

    #[test]
    fn in_and_out_variants_lean_the_right_way() {
        // Ease-in starts slow, ease-out starts fast. At the midpoint that means
        // in < 0.5 < out for every family.
        for (i, o) in [
            (Easing::QuadIn, Easing::QuadOut),
            (Easing::CubicIn, Easing::CubicOut),
            (Easing::QuartIn, Easing::QuartOut),
            (Easing::QuintIn, Easing::QuintOut),
            (Easing::SineIn, Easing::SineOut),
            (Easing::ExpoIn, Easing::ExpoOut),
            (Easing::CircIn, Easing::CircOut),
        ] {
            assert!(i.apply(0.5) < 0.5, "{i:?} does not ease in");
            assert!(o.apply(0.5) > 0.5, "{o:?} does not ease out");
        }
    }

    #[test]
    fn in_out_variants_are_symmetric_about_the_midpoint() {
        for e in [
            Easing::QuadInOut,
            Easing::CubicInOut,
            Easing::QuartInOut,
            Easing::QuintInOut,
            Easing::SineInOut,
            Easing::CircInOut,
        ] {
            assert!((e.apply(0.5) - 0.5).abs() < 1e-4, "{e:?} is not centred");
            for i in 0..=50 {
                let t = i as f32 / 100.0;
                let a = e.apply(t);
                let b = 1.0 - e.apply(1.0 - t);
                assert!((a - b).abs() < 1e-3, "{e:?} asymmetric at {t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn monotonic_curves_never_go_backwards() {
        for e in Easing::ALL
            .iter()
            .filter(|e| !e.overshoots() && !matches!(e, Easing::BounceIn | Easing::BounceOut | Easing::BounceInOut))
        {
            let mut previous = e.apply(0.0);
            for i in 1..=200 {
                let v = e.apply(i as f32 / 200.0);
                assert!(v >= previous - 1e-4, "{e:?} went backwards at {i}");
                previous = v;
            }
        }
    }

    #[test]
    fn steps_snap_where_their_names_say() {
        assert_eq!(Easing::StepStart.apply(0.0), 0.0);
        assert_eq!(Easing::StepStart.apply(0.01), 1.0);
        assert_eq!(Easing::StepEnd.apply(0.99), 0.0);
        assert_eq!(Easing::StepEnd.apply(1.0), 1.0);
    }
}
