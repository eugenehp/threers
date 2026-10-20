use crate::core::{BufferAttribute, BufferGeometry};

pub struct BoxGeometry;

impl BoxGeometry {
    /// Create an axis-aligned box centered at the origin with the given dimensions.
    ///
    /// three.js r165's `BoxGeometry(width, height, depth)` exactly: the same
    /// vertices in the same order (4 per face, each face a flat normal), the
    /// same index (each face split `a b d`, `b c d`), uvs, and six groups —
    /// `+x, -x, +y, -y, +z, -z`, material indices 0–5 — so a raycast reports
    /// the face and `faceIndex` three.js does and a six-material box draws.
    pub fn new(width: f32, height: f32, depth: f32) -> BufferGeometry {
        Self::with_segments(width, height, depth, 1, 1, 1)
    }

    /// three.js `BoxGeometry(width, height, depth, widthSegments,
    /// heightSegments, depthSegments)`: each face a grid of quads.
    pub fn with_segments(
        width: f32,
        height: f32,
        depth: f32,
        width_segments: u32,
        height_segments: u32,
        depth_segments: u32,
    ) -> BufferGeometry {
        use super::js_number;
        Self::from_f64(
            js_number(width),
            js_number(height),
            js_number(depth),
            width_segments,
            height_segments,
            depth_segments,
        )
    }

    /// [`with_segments`](Self::with_segments) with JavaScript's numbers, as
    /// three.js takes them.
    pub fn from_f64(
        width: f64,
        height: f64,
        depth: f64,
        width_segments: u32,
        height_segments: u32,
        depth_segments: u32,
    ) -> BufferGeometry {
        let (ws, hs, ds) = (
            width_segments.max(1),
            height_segments.max(1),
            depth_segments.max(1),
        );
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut groups = Vec::new();
        let mut cells = 0usize;
        // three.js `buildPlane(u, v, w, udir, vdir, width, height, depth,
        // gridX, gridY, materialIndex)`, axes as indices 0 = x, 1 = y, 2 = z.
        // In f64, rounded once into the attributes, as three.js does.
        let mut build_plane = |u: usize,
                               v: usize,
                               w: usize,
                               udir: f64,
                               vdir: f64,
                               pw: f64,
                               ph: f64,
                               pd: f64,
                               grid_x: u32,
                               grid_y: u32,
                               material_index: u32| {
            let segment_width = pw / grid_x as f64;
            let segment_height = ph / grid_y as f64;
            let (width_half, height_half, depth_half) = (pw / 2.0, ph / 2.0, pd / 2.0);
            let (grid_x1, grid_y1) = (grid_x + 1, grid_y + 1);
            let base = (positions.len() / 3) as u32;
            let group_start = indices.len() as u32;
            for iy in 0..grid_y1 {
                let y = iy as f64 * segment_height - height_half;
                for ix in 0..grid_x1 {
                    let x = ix as f64 * segment_width - width_half;
                    let mut p = [0.0f64; 3];
                    p[u] = x * udir;
                    p[v] = y * vdir;
                    p[w] = depth_half;
                    positions.extend_from_slice(&p.map(|c| c as f32));
                    let mut n = [0.0f32; 3];
                    n[w] = if pd > 0.0 { 1.0 } else { -1.0 };
                    normals.extend_from_slice(&n);
                    uvs.push((ix as f64 / grid_x as f64) as f32);
                    uvs.push((1.0 - iy as f64 / grid_y as f64) as f32);
                }
            }
            for iy in 0..grid_y {
                for ix in 0..grid_x {
                    let a = base + ix + grid_x1 * iy;
                    let b = base + ix + grid_x1 * (iy + 1);
                    let c = base + (ix + 1) + grid_x1 * (iy + 1);
                    let d = base + (ix + 1) + grid_x1 * iy;
                    indices.extend_from_slice(&[a, b, d, b, c, d]);
                    cells += 1;
                }
            }
            groups.push((
                group_start,
                indices.len() as u32 - group_start,
                material_index,
            ));
        };
        build_plane(2, 1, 0, -1.0, -1.0, depth, height, width, ds, hs, 0); // px
        build_plane(2, 1, 0, 1.0, -1.0, depth, height, -width, ds, hs, 1); // nx
        build_plane(0, 2, 1, 1.0, 1.0, width, depth, height, ws, ds, 2); // py
        build_plane(0, 2, 1, 1.0, -1.0, width, depth, -height, ws, ds, 3); // ny
        build_plane(0, 1, 2, 1.0, -1.0, width, height, depth, ws, hs, 4); // pz
        build_plane(0, 1, 2, -1.0, -1.0, width, height, -depth, ws, hs, 5); // nz

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        for (start, count, material_index) in groups {
            geom.add_group(start, count, material_index);
        }
        // One quad a grid cell, as two consecutive triangles.
        geom.set_polygons(vec![2; cells]);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_box(&mut geom, width as f32, height as f32, depth as f32);
        geom
    }
}
