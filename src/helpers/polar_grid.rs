use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D};
use crate::materials::LineBasicMaterial;
use crate::math::Color;

/// Radial grid: `radial_segments` spokes from the center, `circles` concentric
/// circles up to `radius`. Mirrors three.js's `PolarGridHelper`.
pub struct PolarGridHelper;

impl PolarGridHelper {
    pub fn new(
        radius: f32,
        radial_segments: usize,
        circles: usize,
        divisions: usize,
        color1: Color,
        color2: Color,
    ) -> Object3D {
        let _ = color2;
        let mut positions = Vec::new();

        // Spokes.
        for s in 0..radial_segments {
            let t = s as f32 / radial_segments as f32 * std::f32::consts::PI * 2.0;
            let (sx, cz) = (t.sin(), t.cos());
            positions.extend_from_slice(&[0.0, 0.0, 0.0, radius * sx, 0.0, radius * cz]);
        }

        // Circles.
        for c in 1..=circles {
            let r = radius * c as f32 / circles as f32;
            let n = divisions.max(8);
            for i in 0..n {
                let t0 = i as f32 / n as f32 * std::f32::consts::PI * 2.0;
                let t1 = (i + 1) as f32 / n as f32 * std::f32::consts::PI * 2.0;
                let (s0, c0) = (t0.sin(), t0.cos());
                let (s1, c1) = (t1.sin(), t1.cos());
                positions.extend_from_slice(&[r * s0, 0.0, r * c0, r * s1, 0.0, r * c1]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        // three.js helpers are `toneMapped: false`.
        let mat = LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(color1) };
        Object3D::line_segments(LineSegments::new(g, mat.into()))
    }

    pub fn default_(radius: f32, radial_segments: usize, circles: usize) -> Object3D {
        Self::new(
            radius,
            radial_segments,
            circles,
            32,
            Color::from_hex(0x888888),
            Color::from_hex(0x444444),
        )
    }
}
