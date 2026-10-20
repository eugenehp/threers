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
        use super::{js_angle, js_number};
        Self::from_f64(
            js_number(radius),
            width_segments,
            height_segments,
            js_angle(phi_start),
            js_angle(phi_length),
            js_angle(theta_start),
            js_angle(theta_length),
        )
    }

    /// [`with_range`](Self::with_range) with JavaScript's numbers, as
    /// three.js takes them.
    pub fn from_f64(
        radius: f64,
        width_segments: usize,
        height_segments: usize,
        phi_start: f64,
        phi_length: f64,
        theta_start: f64,
        theta_length: f64,
    ) -> BufferGeometry {
        use std::f64::consts::PI;
        let w = width_segments.max(3);
        let h = height_segments.max(2);
        let theta_end = (theta_start + theta_length).min(PI);

        let mut positions = Vec::with_capacity((w + 1) * (h + 1) * 3);
        let mut normals = Vec::with_capacity((w + 1) * (h + 1) * 3);
        let mut uvs = Vec::with_capacity((w + 1) * (h + 1) * 2);
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity(h + 1);

        // In f64, rounded once into the attributes, as three.js does.
        let (r, ps, pl, ts, tl) = (radius, phi_start, phi_length, theta_start, theta_length);
        for iy in 0..=h {
            let v = iy as f64 / h as f64;
            let mut row = Vec::with_capacity(w + 1);
            let mut u_offset = 0.0;
            if iy == 0 && theta_start == 0.0 {
                u_offset = 0.5 / w as f64;
            } else if iy == h && theta_end == PI {
                u_offset = -0.5 / w as f64;
            }

            for ix in 0..=w {
                let u = ix as f64 / w as f64;
                let phi = ps + u * pl;
                let theta = ts + v * tl;
                let x = -r * phi.cos() * theta.sin();
                let y = r * theta.cos();
                let z = r * phi.sin() * theta.sin();
                positions.extend_from_slice(&[x as f32, y as f32, z as f32]);
                // three.js `normal.copy(vertex).normalize()`.
                let len = (x * x + y * y + z * z).sqrt();
                let inv = if len > 0.0 { 1.0 / len } else { 0.0 };
                normals.extend_from_slice(&[(x * inv) as f32, (y * inv) as f32, (z * inv) as f32]);
                uvs.extend_from_slice(&[(u + u_offset) as f32, (1.0 - v) as f32]);
                row.push((positions.len() / 3 - 1) as u32);
            }
            grid.push(row);
        }

        let mut indices = Vec::new();
        // Quads, except where three.js drops a pole cell's degenerate half.
        let mut polygons = super::polygons::CellPolygons::new();
        for iy in 0..h {
            for ix in 0..w {
                let a = grid[iy][ix + 1];
                let b = grid[iy][ix];
                let c = grid[iy + 1][ix];
                let d = grid[iy + 1][ix + 1];
                let first = iy != 0 || theta_start > 0.0;
                let second = iy != h - 1 || theta_end < PI;
                if first {
                    indices.extend_from_slice(&[a, b, d]);
                }
                if second {
                    indices.extend_from_slice(&[b, c, d]);
                }
                if first && second {
                    polygons.quad(&positions, [a, b, c, d]);
                } else {
                    polygons.triangles(first as usize + second as usize);
                }
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        polygons.apply(&mut geom);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_sphere(&mut geom, radius as f32);
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
