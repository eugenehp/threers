//! Rolling shutter / shutter curve for motion blur.

use super::curve::{CurveKey, FCurve};
use super::pose::CameraPose;
use crate::animatable::Animatable;

/// How exposure is distributed across the frame (and rolling shutter).
#[derive(Debug, Clone, PartialEq)]
pub struct ShutterCurve {
    /// Fraction of the frame the shutter is open (`0..1`). Blender default 0.5.
    pub shutter: f32,
    /// Optional custom openness curve over the shutter interval (`0..1` → weight).
    pub curve: FCurve,
    /// Rolling shutter scan time as a fraction of the frame (`0` = global shutter).
    pub rolling: f32,
}

impl Default for ShutterCurve {
    fn default() -> Self {
        Self {
            shutter: 0.5,
            curve: FCurve::new(),
            rolling: 0.0,
        }
    }
}

impl ShutterCurve {
    pub fn new(shutter: f32) -> Self {
        Self {
            shutter: shutter.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    pub fn with_rolling(mut self, rolling: f32) -> Self {
        self.rolling = rolling.clamp(0.0, 1.0);
        self
    }

    /// Openness weight at normalised shutter phase `u ∈ 0..1`.
    pub fn weight(&self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        if self.curve.is_empty() {
            1.0
        } else {
            self.curve.evaluate(u).clamp(0.0, 1.0)
        }
    }

    /// Interpolate between shutter-open (`a`) and shutter-close (`b`) poses.
    /// `scanline_v` is `0` at the top of the frame, `1` at the bottom.
    pub fn interpolate(&self, a: CameraPose, b: CameraPose, scanline_v: f32) -> CameraPose {
        let roll = self.rolling * scanline_v.clamp(0.0, 1.0);
        // Centre the exposure window, then bias by rolling scan.
        let u = (0.5 + (roll - 0.5) * self.shutter).clamp(0.0, 1.0);
        let b = a.unwrap_azimuth_toward(b);
        a.lerp(b, u)
    }
}

/// Triangular shutter openness curve.
pub fn triangular_shutter() -> FCurve {
    FCurve::from_keys(vec![
        CurveKey::linear(0.0, 0.0),
        CurveKey::linear(0.5, 1.0),
        CurveKey::linear(1.0, 0.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn rolling_biases_sample() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 1.0, 1.2, 5.0);
        let shutter = ShutterCurve::new(0.5).with_rolling(1.0);
        let top = shutter.interpolate(a, b, 0.0);
        let bottom = shutter.interpolate(a, b, 1.0);
        assert!(bottom.azimuth >= top.azimuth - 1e-4);
    }
}
