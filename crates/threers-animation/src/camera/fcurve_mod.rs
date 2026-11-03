//! F-curve modifiers (Blender graph-editor modifier stack).

use super::curve::FCurve;

/// Keyframe semantic kind (Blender keyframe type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyframeKind {
    #[default]
    Keyframe,
    Breakdown,
    MovingHold,
    Extreme,
    Jitter,
}

/// One modifier in the stack, applied after base curve evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum FCurveModifier {
    /// Additive noise.
    Noise {
        strength: f32,
        scale: f32,
        offset: f32,
    },
    /// Repeat the curve (same as Extrapolation::Cycle, as a modifier).
    Cycles {
        before: u32,
        after: u32,
    },
    /// Envelope: scale+offset with min/max clamps.
    Envelope {
        influence: f32,
        min: f32,
        max: f32,
    },
    /// y = amplitude * sin(phase + freq * t) + offset
    BuiltInFunction {
        amplitude: f32,
        frequency: f32,
        phase: f32,
        offset: f32,
    },
    /// Polynomial generator: y = c0 + c1*t + c2*t^2 + c3*t^3
    Generator {
        coeffs: [f32; 4],
        additive: bool,
    },
    /// Clamp output.
    Limit {
        min: f32,
        max: f32,
    },
    /// Quantise to steps.
    Stepped {
        step_size: f32,
    },
}

impl FCurveModifier {
    pub fn apply(&self, t: f32, value: f32) -> f32 {
        match self {
            Self::Noise {
                strength,
                scale,
                offset,
            } => {
                let n = pseudo_noise(t * scale.max(1e-6) + offset);
                value + n * strength
            }
            Self::Cycles { .. } => value, // handled at evaluate level when wrapped
            Self::Envelope {
                influence,
                min,
                max,
            } => {
                let v = value * influence;
                v.clamp(*min, *max)
            }
            Self::BuiltInFunction {
                amplitude,
                frequency,
                phase,
                offset,
            } => value + amplitude * (phase + frequency * t).sin() + offset,
            Self::Generator { coeffs, additive } => {
                let g = coeffs[0]
                    + coeffs[1] * t
                    + coeffs[2] * t * t
                    + coeffs[3] * t * t * t;
                if *additive {
                    value + g
                } else {
                    g
                }
            }
            Self::Limit { min, max } => value.clamp(*min, *max),
            Self::Stepped { step_size } => {
                let s = step_size.abs().max(1e-8);
                (value / s).floor() * s
            }
        }
    }
}

fn pseudo_noise(x: f32) -> f32 {
    // Cheap deterministic hash → [-1, 1]
    let s = (x * 12.9898).sin() * 43758.547;
    s.fract() * 2.0 - 1.0
}

/// F-curve plus a modifier stack.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModifiedFCurve {
    pub curve: FCurve,
    pub modifiers: Vec<FCurveModifier>,
    pub muted: bool,
}

impl ModifiedFCurve {
    pub fn new(curve: FCurve) -> Self {
        Self {
            curve,
            modifiers: Vec::new(),
            muted: false,
        }
    }

    pub fn push_modifier(&mut self, m: FCurveModifier) {
        self.modifiers.push(m);
    }

    pub fn evaluate(&self, t: f32) -> f32 {
        if self.muted {
            return self.curve.evaluate(t);
        }
        let mut v = self.curve.evaluate(t);
        for m in &self.modifiers {
            v = m.apply(t, v);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::curve::CurveKey;

    #[test]
    fn limit_clamps() {
        let mut m = ModifiedFCurve::new(FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 10.0),
        ]));
        m.push_modifier(FCurveModifier::Limit { min: 2.0, max: 4.0 });
        assert!((m.evaluate(0.5) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn stepped_quantises() {
        let mut m = ModifiedFCurve::new(FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 1.0),
        ]));
        m.push_modifier(FCurveModifier::Stepped { step_size: 0.25 });
        let v = m.evaluate(0.6);
        assert!((v - 0.5).abs() < 1e-4, "{v}");
    }
}
