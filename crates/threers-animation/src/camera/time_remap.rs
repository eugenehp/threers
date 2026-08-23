//! Scene time remapping / speed curves.

use super::curve::{CurveKey, FCurve};

/// Maps scene time → evaluation time (Blender time remapping / speed curve).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeRemap {
    /// Curve of evaluation-time vs scene-time. Empty → identity.
    pub curve: FCurve,
    /// Global scale (Blender “Time Remapping old/new” as a ratio).
    pub scale: f32,
    pub offset: f32,
}

impl Default for TimeRemap {
    fn default() -> Self {
        Self {
            curve: FCurve::new(),
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl TimeRemap {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn scaled(scale: f32) -> Self {
        Self {
            scale: scale.max(1e-6),
            ..Self::default()
        }
    }

    /// Linear remap from (old_start, old_end) → (new_start, new_end).
    pub fn linear_range(old_start: f32, old_end: f32, new_start: f32, new_end: f32) -> Self {
        let mut curve = FCurve::new();
        curve.insert(CurveKey::linear(old_start, new_start));
        curve.insert(CurveKey::linear(old_end, new_end));
        Self {
            curve,
            scale: 1.0,
            offset: 0.0,
        }
    }

    pub fn remap(&self, scene_time: f32) -> f32 {
        let t = if self.curve.is_empty() {
            scene_time
        } else {
            self.curve.evaluate(scene_time)
        };
        t * self.scale + self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_speed() {
        let remap = TimeRemap::scaled(0.5);
        assert!((remap.remap(2.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_range_maps_ends() {
        let remap = TimeRemap::linear_range(0.0, 10.0, 0.0, 5.0);
        assert!((remap.remap(10.0) - 5.0).abs() < 1e-3);
    }
}
