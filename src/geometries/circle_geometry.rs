use crate::core::{BufferAttribute, BufferGeometry};
use std::f32::consts::PI;

pub struct CircleGeometry;

impl CircleGeometry {
    /// Filled disk in the XY plane (normal +Z). `segments` controls the rim
    /// vertex count. `theta_start` and `theta_length` are radians.
    pub fn new(
        radius: f32,
        segments: usize,
        theta_start: f32,
        theta_length: f32,
    ) -> BufferGeometry {
        let n = segments.max(3);

        let mut positions = Vec::with_capacity((n + 2) * 3);
        let mut normals = Vec::with_capacity((n + 2) * 3);
        let mut uvs = Vec::with_capacity((n + 2) * 2);
        let mut indices = Vec::with_capacity(n * 3);

        // Center vertex.
        positions.extend_from_slice(&[0.0, 0.0, 0.0]);
        normals.extend_from_slice(&[0.0, 0.0, 1.0]);
        uvs.extend_from_slice(&[0.5, 0.5]);

        for i in 0..=n {
            let t = theta_start + (i as f32 / n as f32) * theta_length;
            let (s, c) = t.sin_cos();
            let x = radius * c;
            let y = radius * s;
            positions.extend_from_slice(&[x, y, 0.0]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            uvs.extend_from_slice(&[(x / radius + 1.0) * 0.5, (y / radius + 1.0) * 0.5]);
        }

        for i in 1..=n as u32 {
            indices.extend_from_slice(&[i, i + 1, 0]);
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_xy_plane(&mut geom);
        geom
    }

    pub fn default_(radius: f32) -> BufferGeometry {
        Self::new(radius, 32, 0.0, PI * 2.0)
    }
}
