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
        Self::from_f64(
            super::js_number(inner_radius),
            super::js_number(outer_radius),
            theta_segments,
            phi_segments,
            super::js_angle(theta_start),
            super::js_angle(theta_length),
        )
    }

    /// [`new`](Self::new) with JavaScript's numbers, as three.js takes them.
    pub fn from_f64(
        inner_radius: f64,
        outer_radius: f64,
        theta_segments: usize,
        phi_segments: usize,
        theta_start: f64,
        theta_length: f64,
    ) -> BufferGeometry {
        let t_seg = theta_segments.max(3);
        let p_seg = phi_segments.max(1);

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        // In f64 from the caller's numbers, rounded once, as three.js does —
        // including its running radius, `radius += radiusStep` row by row.
        let (inner, outer) = (inner_radius, outer_radius);
        let (ts, tl) = (theta_start, theta_length);
        let radius_step = (outer - inner) / p_seg as f64;
        let mut radius = inner;
        for _ in 0..=p_seg {
            for i in 0..=t_seg {
                let t = ts + i as f64 / t_seg as f64 * tl;
                let x = radius * t.cos();
                let y = radius * t.sin();
                positions.extend_from_slice(&[x as f32, y as f32, 0.0]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                uvs.extend_from_slice(&[
                    ((x / outer + 1.0) / 2.0) as f32,
                    ((y / outer + 1.0) / 2.0) as f32,
                ]);
            }
            radius += radius_step;
        }

        let row = (t_seg + 1) as u32;
        let mut polygons = super::polygons::CellPolygons::new();
        for j in 0..p_seg as u32 {
            for i in 0..t_seg as u32 {
                let a = j * row + i;
                let b = a + row;
                let c = b + 1;
                let d = a + 1;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
                polygons.quad(&positions, [a, b, c, d]);
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        polygons.apply(&mut geom);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_xy_plane(&mut geom);
        geom
    }

    pub fn default_(inner_radius: f32, outer_radius: f32) -> BufferGeometry {
        Self::new(inner_radius, outer_radius, 32, 1, 0.0, PI * 2.0)
    }
}
