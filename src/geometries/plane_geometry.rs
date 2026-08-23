use crate::core::{BufferAttribute, BufferGeometry};

pub struct PlaneGeometry;

impl PlaneGeometry {
    /// XY-plane quad centered at origin, normal +Z (matches three.js `PlaneGeometry`).
    pub fn new(width: f32, height: f32) -> BufferGeometry {
        Self::with_segments(width, height, 1, 1)
    }

    pub fn with_segments(
        width: f32,
        height: f32,
        width_segments: usize,
        height_segments: usize,
    ) -> BufferGeometry {
        let width_half = width * 0.5;
        let height_half = height * 0.5;
        let grid_x = width_segments.max(1);
        let grid_y = height_segments.max(1);
        let grid_x1 = grid_x + 1;
        let grid_y1 = grid_y + 1;
        let segment_width = width / grid_x as f32;
        let segment_height = height / grid_y as f32;

        let mut positions = Vec::with_capacity(grid_x1 * grid_y1 * 3);
        let mut normals = Vec::with_capacity(grid_x1 * grid_y1 * 3);
        let mut uvs = Vec::with_capacity(grid_x1 * grid_y1 * 2);

        for iy in 0..grid_y1 {
            let y = iy as f32 * segment_height - height_half;
            for ix in 0..grid_x1 {
                let x = ix as f32 * segment_width - width_half;
                // three.js pushes (x, -y, 0) — flips grid Y into world +Y.
                positions.extend_from_slice(&[x, -y, 0.0]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                uvs.push(ix as f32 / grid_x as f32);
                uvs.push(1.0 - (iy as f32 / grid_y as f32));
            }
        }

        let mut indices = Vec::with_capacity(grid_x * grid_y * 6);
        for iy in 0..grid_y {
            for ix in 0..grid_x {
                let a = (ix + grid_x1 * iy) as u32;
                let b = (ix + grid_x1 * (iy + 1)) as u32;
                let c = (ix + 1 + grid_x1 * (iy + 1)) as u32;
                let d = (ix + 1 + grid_x1 * iy) as u32;
                indices.extend_from_slice(&[a, b, d, b, c, d]);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_3x2_2_matches_threejs_default_segments() {
        let g = PlaneGeometry::new(3.0, 2.2);
        let pos = g.get_attribute("position").expect("position");
        assert_eq!(pos.count(), 4);
        let data = &pos.array;
        let expected: [f32; 12] = [
            -1.5, 1.1, 0.0, 1.5, 1.1, 0.0, -1.5, -1.1, 0.0, 1.5, -1.1, 0.0,
        ];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 0.001, "pos[{i}] = {a}, expected {b}");
        }
        let idx = g.index.as_ref().expect("index");
        assert_eq!(idx, &[0, 2, 1, 2, 3, 1]);
    }
}
