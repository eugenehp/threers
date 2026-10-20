use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

pub struct ParametricGeometry;

impl ParametricGeometry {
    /// Sample a surface defined by `f(u, v) -> Vector3` on a `slices × stacks`
    /// grid. Normals are estimated by finite differences. Matches three.js's
    /// `ParametricGeometry`.
    pub fn new<F>(f: F, slices: usize, stacks: usize) -> BufferGeometry
    where
        F: Fn(f32, f32) -> Vector3,
    {
        let s = slices.max(1);
        let t = stacks.max(1);
        let mut positions = Vec::with_capacity((s + 1) * (t + 1) * 3);
        let mut normals = Vec::with_capacity((s + 1) * (t + 1) * 3);
        let mut uvs = Vec::with_capacity((s + 1) * (t + 1) * 2);
        let mut indices = Vec::new();

        let eps = 1e-4f32;
        for j in 0..=t {
            let v = j as f32 / t as f32;
            for i in 0..=s {
                let u = i as f32 / s as f32;
                let p = f(u, v);
                let du = f((u + eps).min(1.0), v) - f((u - eps).max(0.0), v);
                let dv = f(u, (v + eps).min(1.0)) - f(u, (v - eps).max(0.0));
                let n = du.cross(dv).normalize();
                positions.extend_from_slice(&[p.x, p.y, p.z]);
                normals.extend_from_slice(&[n.x, n.y, n.z]);
                uvs.extend_from_slice(&[u, v]);
            }
        }

        let row = (s + 1) as u32;
        let mut polygons = super::polygons::CellPolygons::new();
        for j in 0..t as u32 {
            for i in 0..s as u32 {
                let a = j * row + i;
                let b = (j + 1) * row + i;
                let c = (j + 1) * row + i + 1;
                let d = j * row + i + 1;
                indices.extend_from_slice(&[a, b, d, b, c, d]);
                polygons.quad(&positions, [a, b, c, d]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        polygons.apply(&mut g);
        g
    }
}
