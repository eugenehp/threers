use crate::core::{BufferAttribute, BufferGeometry};
use std::f32::consts::PI;

pub struct SphereGeometry;

impl SphereGeometry {
    /// UV sphere matching three.js `SphereGeometry` defaults (`phiStart=0`,
    /// `phiLength=2π`, `thetaStart=0`, `thetaLength=π`).
    pub fn new(radius: f32, width_segments: usize, height_segments: usize) -> BufferGeometry {
        Self::with_range(
            radius,
            width_segments,
            height_segments,
            0.0,
            PI * 2.0,
            0.0,
            PI,
        )
    }

    pub fn with_range(
        radius: f32,
        width_segments: usize,
        height_segments: usize,
        phi_start: f32,
        phi_length: f32,
        theta_start: f32,
        theta_length: f32,
    ) -> BufferGeometry {
        let w = width_segments.max(3);
        let h = height_segments.max(2);
        let theta_end = (theta_start + theta_length).min(PI);

        let mut positions = Vec::with_capacity((w + 1) * (h + 1) * 3);
        let mut normals = Vec::with_capacity((w + 1) * (h + 1) * 3);
        let mut uvs = Vec::with_capacity((w + 1) * (h + 1) * 2);
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity(h + 1);

        for iy in 0..=h {
            let v = iy as f32 / h as f32;
            let mut row = Vec::with_capacity(w + 1);
            let mut u_offset = 0.0;
            if iy == 0 && theta_start == 0.0 {
                u_offset = 0.5 / w as f32;
            } else if iy == h && theta_end == PI {
                u_offset = -0.5 / w as f32;
            }

            for ix in 0..=w {
                let u = ix as f32 / w as f32;
                let phi = phi_start + u * phi_length;
                let theta = theta_start + v * theta_length;
                let (sp, cp) = theta.sin_cos();
                let cp_phi = phi.cos();
                let sp_phi = phi.sin();
                let x = -radius * cp_phi * sp;
                let y = radius * cp;
                let z = radius * sp_phi * sp;
                positions.extend_from_slice(&[x, y, z]);
                let n_len = (x * x + y * y + z * z).sqrt().max(1e-8);
                normals.extend_from_slice(&[x / n_len, y / n_len, z / n_len]);
                uvs.extend_from_slice(&[u + u_offset, 1.0 - v]);
                row.push((positions.len() / 3 - 1) as u32);
            }
            grid.push(row);
        }

        let mut indices = Vec::new();
        for iy in 0..h {
            for ix in 0..w {
                let a = grid[iy][ix + 1];
                let b = grid[iy][ix];
                let c = grid[iy + 1][ix];
                let d = grid[iy + 1][ix + 1];
                if iy != 0 || theta_start > 0.0 {
                    indices.extend_from_slice(&[a, b, d]);
                }
                if iy != h - 1 || theta_end < PI {
                    indices.extend_from_slice(&[b, c, d]);
                }
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_sphere(&mut geom, radius);
        geom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_1x32x16_matches_threejs() {
        let g = SphereGeometry::new(1.0, 32, 16);
        let pos = g.get_attribute("position").expect("position");
        assert_eq!(pos.count(), 561);
        let idx = g.index.as_ref().expect("index");
        assert_eq!(idx.len(), 2880);
        let data = &pos.array;
        assert!((data[0] - 0.0).abs() < 1e-4);
        assert!((data[1] - 1.0).abs() < 1e-4);
        assert!((data[2] - 0.0).abs() < 1e-4);
        assert_eq!(idx[0..6], [0, 33, 34, 1, 34, 35]);
        let uv = g.get_attribute("uv").expect("uv");
        assert!((uv.array[0] - 0.015625).abs() < 1e-4);
        assert!((uv.array[1] - 1.0).abs() < 1e-4);
    }
}
