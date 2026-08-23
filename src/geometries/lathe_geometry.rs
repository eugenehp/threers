use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{Vector2, Vector3};
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
        let segments = segments.max(1);
        let phi_length = phi_length.clamp(0.0, PI * 2.0);
        let pn = points.len().max(2);
        let inverse_segments = 1.0 / segments as f32;

        let mut init_normals = Vec::with_capacity(pn * 3);
        let mut prev_normal = Vector3::ZERO;
        for j in 0..pn {
            match j {
                0 => {
                    let dx = points[1].x - points[0].x;
                    let dy = points[1].y - points[0].y;
                    let mut normal = Vector3::new(dy, -dx, 0.0);
                    prev_normal = normal;
                    normal = normal.normalize();
                    init_normals.extend_from_slice(&[normal.x, normal.y, normal.z]);
                }
                j if j == pn - 1 => {
                    init_normals.extend_from_slice(&[prev_normal.x, prev_normal.y, prev_normal.z]);
                }
                _ => {
                    let dx = points[j + 1].x - points[j].x;
                    let dy = points[j + 1].y - points[j].y;
                    let cur = Vector3::new(dy, -dx, 0.0);
                    let normal = Vector3::new(
                        cur.x + prev_normal.x,
                        cur.y + prev_normal.y,
                        cur.z + prev_normal.z,
                    )
                    .normalize();
                    init_normals.extend_from_slice(&[normal.x, normal.y, normal.z]);
                    prev_normal = cur;
                }
            }
        }

        let mut positions = Vec::with_capacity((segments + 1) * pn * 3);
        let mut normals = Vec::with_capacity((segments + 1) * pn * 3);
        let mut uvs = Vec::with_capacity((segments + 1) * pn * 2);
        let mut indices = Vec::new();

        for i in 0..=segments {
            let phi = phi_start + i as f32 * inverse_segments * phi_length;
            let (sin, cos) = phi.sin_cos();
            for j in 0..pn {
                let p = points[j];
                positions.extend_from_slice(&[p.x * sin, p.y, p.x * cos]);
                let nx = init_normals[j * 3] * sin;
                let ny = init_normals[j * 3 + 1];
                let nz = init_normals[j * 3] * cos;
                normals.extend_from_slice(&[nx, ny, nz]);
                uvs.extend_from_slice(&[i as f32 / segments as f32, j as f32 / (pn - 1) as f32]);
            }
        }

        for i in 0..segments {
            for j in 0..(pn - 1) {
                let base = j + i * pn;
                let a = base as u32;
                let b = (base + pn) as u32;
                let c = (base + pn + 1) as u32;
                let d = (base + 1) as u32;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[c, d, b]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_lathe(&mut g, points, phi_start, phi_length);
        g
    }

    pub fn default_(points: &[Vector2]) -> BufferGeometry {
        Self::new(points, 12, 0.0, PI * 2.0)
    }
}
