use crate::core::{BufferAttribute, BufferGeometry};
use std::f32::consts::PI;

pub struct RingGeometry;

impl RingGeometry {
    /// Annular disk in the XY plane (normal +Z).
    /// `theta_segments` is around-circumference vertices; `phi_segments` is
    /// radial subdivisions between `inner_radius` and `outer_radius`.
    pub fn new(
        inner_radius: f32,
        outer_radius: f32,
        theta_segments: usize,
        phi_segments: usize,
        theta_start: f32,
        theta_length: f32,
    ) -> BufferGeometry {
        let t_seg = theta_segments.max(3);
        let p_seg = phi_segments.max(1);

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        for j in 0..=p_seg {
            let radius = inner_radius + (j as f32 / p_seg as f32) * (outer_radius - inner_radius);
            for i in 0..=t_seg {
                let t = theta_start + (i as f32 / t_seg as f32) * theta_length;
                let (s, c) = t.sin_cos();
                let x = radius * c;
                let y = radius * s;
                positions.extend_from_slice(&[x, y, 0.0]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                uvs.extend_from_slice(&[
                    (x / outer_radius + 1.0) * 0.5,
                    (y / outer_radius + 1.0) * 0.5,
                ]);
            }
        }

        let row = (t_seg + 1) as u32;
        for j in 0..p_seg as u32 {
            for i in 0..t_seg as u32 {
                let a = j * row + i;
                let b = a + row;
                let c = b + 1;
                let d = a + 1;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
            }
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

    pub fn default_(inner_radius: f32, outer_radius: f32) -> BufferGeometry {
        Self::new(inner_radius, outer_radius, 32, 1, 0.0, PI * 2.0)
    }
}
