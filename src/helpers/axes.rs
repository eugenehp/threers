use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D};
use crate::materials::LineBasicMaterial;
use crate::math::Color;

/// X (red), Y (green), Z (blue) axes meeting at the origin.
/// Mirrors three.js's `AxesHelper`.
pub struct AxesHelper;

impl AxesHelper {
    pub fn new(size: f32) -> Object3D {
        let positions = vec![
            0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            size,
        ];
        let colors = vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0,
        ];
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("color", BufferAttribute::new(colors, 3));
        // three.js helpers are `toneMapped: false`.
        let mat = LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(Color::WHITE) };
        Object3D::line_segments(LineSegments::new(g, mat.into()))
    }
}
