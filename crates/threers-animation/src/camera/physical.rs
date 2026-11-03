//! Physical camera: sensor size ↔ focal length ↔ vertical FOV.

use std::f32::consts::PI;

/// Common film / sensor sizes (width × height in millimetres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmBack {
    pub width_mm: f32,
    pub height_mm: f32,
}

impl FilmBack {
    pub const FULL_FRAME: Self = Self {
        width_mm: 36.0,
        height_mm: 24.0,
    };
    pub const APS_C: Self = Self {
        width_mm: 23.6,
        height_mm: 15.6,
    };
    pub const MICRO_FOUR_THIRDS: Self = Self {
        width_mm: 17.3,
        height_mm: 13.0,
    };
    pub const SUPER_35: Self = Self {
        width_mm: 24.89,
        height_mm: 18.66,
    };
    pub const IMAX_15_70: Self = Self {
        width_mm: 70.41,
        height_mm: 52.63,
    };

    pub fn aspect(self) -> f32 {
        self.width_mm / self.height_mm.max(1e-6)
    }

    /// Vertical FOV (radians) for a focal length in millimetres.
    pub fn fov_vertical(self, focal_length_mm: f32) -> f32 {
        let f = focal_length_mm.max(1e-3);
        2.0 * (self.height_mm / (2.0 * f)).atan()
    }

    /// Horizontal FOV (radians).
    pub fn fov_horizontal(self, focal_length_mm: f32) -> f32 {
        let f = focal_length_mm.max(1e-3);
        2.0 * (self.width_mm / (2.0 * f)).atan()
    }

    /// Focal length (mm) that yields a vertical FOV.
    pub fn focal_length_for_fov(self, fov_vertical: f32) -> f32 {
        let half = (fov_vertical * 0.5).clamp(1e-4, PI * 0.49);
        self.height_mm / (2.0 * half.tan())
    }
}

/// Anamorphic squeeze: horizontal FOV is widened by `squeeze` (e.g. 2.0 for 2x).
pub fn anamorphic_horizontal_fov(vertical_fov: f32, aspect: f32, squeeze: f32) -> f32 {
    let half_v = vertical_fov * 0.5;
    let half_h = (half_v.tan() * aspect * squeeze.max(1e-3)).atan();
    half_h * 2.0
}

/// Resolve which FOV axis Blender-style sensor fit uses.
pub fn fov_for_sensor_fit(
    back: FilmBack,
    focal_length_mm: f32,
    fit: threers::cameras::SensorFit,
    viewport_aspect: f32,
) -> f32 {
    use threers::cameras::SensorFit;
    let fit = match fit {
        SensorFit::Auto => {
            if viewport_aspect >= back.aspect() {
                SensorFit::Horizontal
            } else {
                SensorFit::Vertical
            }
        }
        other => other,
    };
    match fit {
        SensorFit::Horizontal => {
            let h = back.fov_horizontal(focal_length_mm);
            // Convert horizontal → vertical for our camera (stores vertical FOV).
            let half_h = h * 0.5;
            2.0 * (half_h.tan() / viewport_aspect.max(1e-6)).atan()
        }
        SensorFit::Vertical | SensorFit::Auto => back.fov_vertical(focal_length_mm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifty_mm_on_full_frame_is_about_27_degrees_vertical() {
        let fov = FilmBack::FULL_FRAME.fov_vertical(50.0);
        // 2*atan(12/50) ≈ 27°
        assert!((fov.to_degrees() - 27.0).abs() < 1.0, "{}", fov.to_degrees());
    }

    #[test]
    fn focal_length_roundtrips() {
        let back = FilmBack::SUPER_35;
        let fov = back.fov_vertical(35.0);
        let f = back.focal_length_for_fov(fov);
        assert!((f - 35.0).abs() < 1e-3);
    }
}
