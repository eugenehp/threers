use crate::core::{BufferAttribute, BufferGeometry};
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
        Self::from_f64(
            super::js_number(radius),
            super::js_number(tube),
            radial_segments,
            tubular_segments,
            super::js_angle(arc),
        )
    }

    /// [`new`](Self::new) with JavaScript's numbers, as three.js takes them.
    pub fn from_f64(
        radius: f64,
        tube: f64,
        radial_segments: usize,
        tubular_segments: usize,
        arc: f64,
    ) -> BufferGeometry {
        // three.js floors the counts and clamps neither.
        let rs = radial_segments.max(1);
        let ts = tubular_segments.max(1);

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        // In f64 from the caller's numbers, rounded once, as three.js does.
        let r = radius;
        for j in 0..=rs {
            let v = j as f64 / rs as f64 * std::f64::consts::PI * 2.0;
            for i in 0..=ts {
                let u = i as f64 / ts as f64 * arc;
                let x = (r + tube * v.cos()) * u.cos();
                let y = (r + tube * v.cos()) * u.sin();
                let z = tube * v.sin();
                positions.extend_from_slice(&[x as f32, y as f32, z as f32]);
                // `normal.subVectors(vertex, center).normalize()`.
                let (nx, ny, nz) = (x - r * u.cos(), y - r * u.sin(), z);
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let inv = 1.0 / if len == 0.0 { 1.0 } else { len };
                normals.extend_from_slice(&[
                    (nx * inv) as f32,
                    (ny * inv) as f32,
                    (nz * inv) as f32,
                ]);
                uvs.extend_from_slice(&[
                    (i as f64 / ts as f64) as f32,
                    (j as f64 / rs as f64) as f32,
                ]);
            }
        }

        let row = (ts + 1) as u32;
        let mut polygons = super::polygons::CellPolygons::new();
        for j in 1..=rs as u32 {
            for i in 1..=ts as u32 {
                let a = (row * j) + i - 1;
                let b = (row * (j - 1)) + i - 1;
                let c = (row * (j - 1)) + i;
                let d = (row * j) + i;
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
        crate::brep::primitives::tag_torus(&mut geom, radius as f32, tube as f32);
        geom
    }

    pub fn default_(radius: f32, tube: f32) -> BufferGeometry {
        Self::new(radius, tube, 12, 48, PI * 2.0)
    }
}
