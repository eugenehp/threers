use crate::core::{BufferAttribute, BufferGeometry};

pub struct BoxGeometry;

impl BoxGeometry {
    /// Create an axis-aligned box centered at the origin with the given dimensions.
    ///
    /// Emits position (vec3), normal (vec3), and uv (vec2) attributes, with
    /// 4 unique vertices per face so each face gets a flat normal — matching
    /// three.js's `BoxGeometry` behavior with default segment counts.
    pub fn new(width: f32, height: f32, depth: f32) -> BufferGeometry {
        let hx = width * 0.5;
        let hy = height * 0.5;
        let hz = depth * 0.5;

        // Per-face: [p0, p1, p2, p3], normal, then uv mapped to [(0,0),(1,0),(1,1),(0,1)].
        let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
            // +X (right)
            (
                [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]],
                [1.0, 0.0, 0.0],
            ),
            // -X (left)
            (
                [
                    [-hx, -hy, -hz],
                    [-hx, -hy, hz],
                    [-hx, hy, hz],
                    [-hx, hy, -hz],
                ],
                [-1.0, 0.0, 0.0],
            ),
            // +Y (top)
            (
                [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]],
                [0.0, 1.0, 0.0],
            ),
            // -Y (bottom)
            (
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [hx, -hy, hz],
                    [-hx, -hy, hz],
                ],
                [0.0, -1.0, 0.0],
            ),
            // +Z (front)
            (
                [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
                [0.0, 0.0, 1.0],
            ),
            // -Z (back)
            (
                [
                    [hx, -hy, -hz],
                    [-hx, -hy, -hz],
                    [-hx, hy, -hz],
                    [hx, hy, -hz],
                ],
                [0.0, 0.0, -1.0],
            ),
        ];

        let mut positions = Vec::with_capacity(6 * 4 * 3);
        let mut normals = Vec::with_capacity(6 * 4 * 3);
        let mut uvs = Vec::with_capacity(6 * 4 * 2);
        let mut indices = Vec::with_capacity(6 * 6);

        for (face_idx, (verts, n)) in faces.iter().enumerate() {
            let base = (face_idx * 4) as u32;
            for v in verts {
                positions.extend_from_slice(v);
                normals.extend_from_slice(n);
            }
            uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_box(&mut geom, width, height, depth);
        geom
    }
}
