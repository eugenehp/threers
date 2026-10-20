use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D};
use crate::materials::LineBasicMaterial;
use crate::math::Color;

/// Grid in the XZ plane. Mirrors three.js's `GridHelper`.
pub struct GridHelper;

impl GridHelper {
    pub fn new(size: f32, divisions: usize, color1: Color, color2: Color) -> Object3D {
        let n = divisions.max(1);
        let step = size / n as f32;
        let half = size * 0.5;
        let center = n / 2;
        // three.js stores sRGB components in the vertex-color attribute.
        let c1 = color1.to_array();
        let c2 = color2.to_array();
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        for i in 0..=n {
            let k = -half + step * i as f32;
            let c = if i == center { c1 } else { c2 };
            // Line along Z at fixed X.
            positions.extend_from_slice(&[k, 0.0, -half, k, 0.0, half]);
            colors.extend_from_slice(&[c[0], c[1], c[2], c[0], c[1], c[2]]);
            // Line along X at fixed Z.
            positions.extend_from_slice(&[-half, 0.0, k, half, 0.0, k]);
            colors.extend_from_slice(&[c[0], c[1], c[2], c[0], c[1], c[2]]);
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("color", BufferAttribute::new(colors, 3));
        // three.js helpers are `toneMapped: false`.
        let mat = LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(Color::WHITE) };
        Object3D::line_segments(LineSegments::new(g, mat.into()))
    }

    pub fn new_from_hex(size: f32, divisions: usize, color1_hex: u32, color2_hex: u32) -> Object3D {
        Self::new(
            size,
            divisions,
            Color::from_hex(color1_hex),
            Color::from_hex(color2_hex),
        )
    }

    pub fn default_(size: f32, divisions: usize) -> Object3D {
        Self::new(
            size,
            divisions,
            Color::from_hex(0x888888),
            Color::from_hex(0x444444),
        )
    }
}
