use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector2;
use std::f32::consts::PI;

pub struct LatheGeometry;

impl LatheGeometry {
    /// Revolve a 2D profile around the Y axis. Matches three.js `LatheGeometry`.
    pub fn new(
        points: &[Vector2],
        segments: usize,
        phi_start: f32,
        phi_length: f32,
    ) -> BufferGeometry {
        let pts: Vec<[f64; 2]> = points
            .iter()
            .map(|p| [super::js_number(p.x), super::js_number(p.y)])
            .collect();
        Self::with_f64(
            &pts,
            segments,
            super::js_angle(phi_start),
            super::js_angle(phi_length),
        )
    }

    /// [`new`](Self::new) with the profile and angles in `f64`, computed as
    /// three.js computes them in JavaScript, so the `f32` buffers come out
    /// bit for bit as r165's.
    pub fn with_f64(
        points: &[[f64; 2]],
        segments: usize,
        phi_start: f64,
        phi_length: f64,
    ) -> BufferGeometry {
        if points.len() < 2 {
            return BufferGeometry::new();
        }
        let segments = segments.max(1);
        let phi_length = phi_length.clamp(0.0, std::f64::consts::PI * 2.0);
        let pn = points.len();
        let inverse_segments = 1.0 / segments as f64;
        // `Vector3.normalize`: scale by the reciprocal of `length() || 1`.
        let normalize = |v: [f64; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
            [v[0] * inv, v[1] * inv, v[2] * inv]
        };

        // The initial meridian's normals, as three.js computes them — the
        // last one is the previous segment's normal *unnormalised*, as r165
        // has it.
        let mut init_normals: Vec<[f64; 3]> = Vec::with_capacity(pn);
        let mut prev_normal = [0.0f64; 3];
        for j in 0..pn {
            if j == 0 {
                let dx = points[1][0] - points[0][0];
                let dy = points[1][1] - points[0][1];
                let normal = [dy * 1.0, -dx, dy * 0.0];
                prev_normal = normal;
                init_normals.push(normalize(normal));
            } else if j == pn - 1 {
                init_normals.push(prev_normal);
            } else {
                let dx = points[j + 1][0] - points[j][0];
                let dy = points[j + 1][1] - points[j][1];
                let cur = [dy * 1.0, -dx, dy * 0.0];
                let normal = [
                    cur[0] + prev_normal[0],
                    cur[1] + prev_normal[1],
                    cur[2] + prev_normal[2],
                ];
                init_normals.push(normalize(normal));
                prev_normal = cur;
            }
        }

        let mut positions = Vec::with_capacity((segments + 1) * pn * 3);
        let mut normals = Vec::with_capacity((segments + 1) * pn * 3);
        let mut uvs = Vec::with_capacity((segments + 1) * pn * 2);
        let mut indices = Vec::new();

        for i in 0..=segments {
            let phi = phi_start + i as f64 * inverse_segments * phi_length;
            let (sin, cos) = (phi.sin(), phi.cos());
            for j in 0..pn {
                let p = points[j];
                positions.extend_from_slice(&[
                    (p[0] * sin) as f32,
                    p[1] as f32,
                    (p[0] * cos) as f32,
                ]);
                uvs.extend_from_slice(&[
                    (i as f64 / segments as f64) as f32,
                    (j as f64 / (pn - 1) as f64) as f32,
                ]);
                let n = init_normals[j];
                normals.extend_from_slice(&[(n[0] * sin) as f32, n[1] as f32, (n[0] * cos) as f32]);
            }
        }

        let mut polygons = super::polygons::CellPolygons::new();
        for i in 0..segments {
            for j in 0..(pn - 1) {
                let base = j + i * pn;
                let a = base as u32;
                let b = (base + pn) as u32;
                let c = (base + pn + 1) as u32;
                let d = (base + 1) as u32;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[c, d, b]);
                polygons.quad(&positions, [a, b, c, d]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        polygons.apply(&mut g);
        #[cfg(feature = "brep")]
        {
            let pts: Vec<Vector2> = points
                .iter()
                .map(|p| Vector2::new(p[0] as f32, p[1] as f32))
                .collect();
            crate::brep::primitives::tag_lathe(&mut g, &pts, phi_start as f32, phi_length as f32);
        }
        g
    }

    pub fn default_(points: &[Vector2]) -> BufferGeometry {
        Self::new(points, 12, 0.0, PI * 2.0)
    }
}
