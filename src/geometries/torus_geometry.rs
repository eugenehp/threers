use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;
use std::f32::consts::PI;

pub struct TorusGeometry;

impl TorusGeometry {
    /// Standard torus. `radius` is the donut radius (center of tube to center
    /// of torus); `tube` is the tube radius; `radial_segments` is around the
    /// tube; `tubular_segments` is around the torus.
    pub fn new(
        radius: f32,
        tube: f32,
        radial_segments: usize,
        tubular_segments: usize,
        arc: f32,
    ) -> BufferGeometry {
        let rs = radial_segments.max(3);
        let ts = tubular_segments.max(3);

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        for j in 0..=rs {
            let v = j as f32 / rs as f32 * PI * 2.0;
            let (sv, cv) = v.sin_cos();
            for i in 0..=ts {
                let u = i as f32 / ts as f32 * arc;
                let (su, cu) = u.sin_cos();
                let x = (radius + tube * cv) * cu;
                let y = (radius + tube * cv) * su;
                let z = tube * sv;
                positions.extend_from_slice(&[x, y, z]);

                let center = Vector3::new(radius * cu, radius * su, 0.0);
                let n = (Vector3::new(x, y, z) - center).normalize();
                normals.extend_from_slice(&[n.x, n.y, n.z]);

                uvs.extend_from_slice(&[i as f32 / ts as f32, j as f32 / rs as f32]);
            }
        }

        let row = (ts + 1) as u32;
        for j in 1..=rs as u32 {
            for i in 1..=ts as u32 {
                let a = (row * j) + i - 1;
                let b = (row * (j - 1)) + i - 1;
                let c = (row * (j - 1)) + i;
                let d = (row * j) + i;
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
        crate::brep::primitives::tag_torus(&mut geom, radius, tube);
        geom
    }

    pub fn default_(radius: f32, tube: f32) -> BufferGeometry {
        Self::new(radius, tube, 12, 48, PI * 2.0)
    }
}
