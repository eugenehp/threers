use crate::math::Color;

/// Flat-color line material. Mirrors three.js's `LineBasicMaterial`.
#[derive(Debug, Clone, Copy)]
pub struct LineBasicMaterial {
    pub color: Color,
    pub opacity: f32,
    pub line_width: f32,
    pub dashed: bool,
    pub dash_scale: f32,
    pub dash_size: f32,
    pub gap_size: f32,
    /// When false, lines draw on top of the depth buffer (x-ray / overlay).
    /// Default true — matches three.js `depthTest`.
    pub depth_test: bool,
}

impl Default for LineBasicMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            opacity: 1.0,
            line_width: 1.0,
            dashed: false,
            dash_scale: 1.0,
            dash_size: 0.0,
            gap_size: 0.0,
            depth_test: true,
        }
    }
}

impl LineBasicMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    /// Draw through occluders (active highlights, HUD strokes).
    pub fn with_depth_test(mut self, enabled: bool) -> Self {
        self.depth_test = enabled;
        self
    }
}
