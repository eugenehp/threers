use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::{earcut, Curve2, Shape};

pub struct ExtrudeGeometry;

impl ExtrudeGeometry {
    /// Simplified extrusion: take the outline of `shape`, sample it into a
    /// 2D contour, then sweep it along Z by `depth`. The cap faces are
    /// triangulated with a fan from the first vertex (assumes a convex
    /// outline — three.js's earcut-based triangulation lands later).
    pub fn new(shape: &Shape, depth: f32, samples_per_segment: usize) -> BufferGeometry {
        let pts2 = shape
            .outline
            .curve_path
            .get_points(samples_per_segment.max(2));
        let n = pts2.len();
        if n < 3 {
            return BufferGeometry::new();
        }
        let mut positions = Vec::with_capacity(2 * n * 3 + 6 * n);
        let mut normals = Vec::with_capacity(2 * n * 3 + 6 * n);
        let mut uvs = Vec::with_capacity(2 * n * 2 + 4 * n);
        let mut indices = Vec::new();

        // Bottom cap (z = 0).
        for p in &pts2 {
            positions.extend_from_slice(&[p.x, p.y, 0.0]);
            normals.extend_from_slice(&[0.0, 0.0, -1.0]);
            uvs.extend_from_slice(&[p.x, p.y]);
        }
        // Top cap (z = depth).
        for p in &pts2 {
            positions.extend_from_slice(&[p.x, p.y, depth]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            uvs.extend_from_slice(&[p.x, p.y]);
        }
        // Earcut caps (handles concave outlines).
        let tris = earcut::earcut(&pts2, &[]);
        // Read only by the `brep` tagging below; without that feature nothing
        // consumes it, and it is not worth cfg-ing the arithmetic itself.
        #[cfg_attr(not(feature = "brep"), allow(unused_variables))]
        let cap_triangles = tris.len() / 3;
        for tri in tris.chunks_exact(3) {
            indices.extend_from_slice(&[tri[0], tri[2], tri[1]]);
            let off = n as u32;
            indices.extend_from_slice(&[off + tri[0], off + tri[1], off + tri[2]]);
        }

        // Sides — each contour edge becomes a quad (4 unique vertices for flat shading).
        let side_base = positions.len() / 3;
        let _ = side_base;
        for i in 0..n {
            let a = pts2[i];
            let b = pts2[(i + 1) % n];
            // Per-quad normal (outward in XY plane).
            let edge_x = b.x - a.x;
            let edge_y = b.y - a.y;
            let nx_raw = edge_y;
            let ny_raw = -edge_x;
            let nlen = (nx_raw * nx_raw + ny_raw * ny_raw).sqrt().max(1e-8);
            let nx = nx_raw / nlen;
            let ny = ny_raw / nlen;
            let base = (positions.len() / 3) as u32;
            positions.extend_from_slice(&[a.x, a.y, 0.0]);
            positions.extend_from_slice(&[b.x, b.y, 0.0]);
            positions.extend_from_slice(&[b.x, b.y, depth]);
            positions.extend_from_slice(&[a.x, a.y, depth]);
            for _ in 0..4 {
                normals.extend_from_slice(&[nx, ny, 0.0]);
            }
            uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_extrude(&mut g, &pts2, depth, cap_triangles);
        g
    }
}
