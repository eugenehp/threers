use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::shapes::{shape_utils, Shape};

/// three.js r165 `ShapeGeometry`: the shapes' faces, flat in the XY plane,
/// with world-space UVs (`uv = (x, y)`) and +Z normals.
pub struct ShapeGeometry;

impl ShapeGeometry {
    /// `new ShapeGeometry(shape, curveSegments)` — one shape, no groups.
    pub fn new(shape: &Shape, curve_segments: usize) -> BufferGeometry {
        Self::build(std::slice::from_ref(shape), curve_segments, false)
    }

    /// `new ShapeGeometry([shapes…], curveSegments)` — group `i` (material
    /// index `i`) covers shape `i`'s triangles, as three.js does for an array.
    pub fn from_shapes(shapes: &[Shape], curve_segments: usize) -> BufferGeometry {
        Self::build(shapes, curve_segments, true)
    }

    /// Either form: `grouped` is whether three.js was handed an array.
    pub fn build(shapes: &[Shape], curve_segments: usize, grouped: bool) -> BufferGeometry {
        let mut indices: Vec<u32> = Vec::new();
        let mut vertices: Vec<f64> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut uvs: Vec<f64> = Vec::new();
        let mut groups = Vec::new();
        let mut group_start = 0u32;

        for (i, shape) in shapes.iter().enumerate() {
            let index_offset = (vertices.len() / 3) as u32;
            let points = shape.extract_points(curve_segments);
            let mut shape_vertices = points.shape;
            let mut holes = points.holes;
            if !shape_utils::is_clock_wise(&shape_vertices) {
                shape_vertices.reverse();
            }
            for h in holes.iter_mut() {
                if shape_utils::is_clock_wise(h) {
                    h.reverse();
                }
            }
            let faces = shape_utils::triangulate_shape(&mut shape_vertices, &mut holes);
            for h in &holes {
                shape_vertices.extend_from_slice(h);
            }
            for v in &shape_vertices {
                vertices.extend_from_slice(&[v.x, v.y, 0.0]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                uvs.extend_from_slice(&[v.x, v.y]);
            }
            let mut group_count = 0u32;
            for f in &faces {
                indices.extend_from_slice(&[
                    f[0] + index_offset,
                    f[1] + index_offset,
                    f[2] + index_offset,
                ]);
                group_count += 3;
            }
            if grouped {
                groups.push((group_start, group_count, i as u32));
                group_start += group_count;
            }
        }

        let mut g = BufferGeometry::new();
        g.set_index(indices);
        g.set_attribute(
            "position",
            BufferAttribute::new(vertices.iter().map(|&v| v as f32).collect(), 3),
        );
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute(
            "uv",
            BufferAttribute::new(uvs.iter().map(|&v| v as f32).collect(), 2),
        );
        for (s, c, m) in groups {
            g.add_group(s, c, m);
        }
        g
    }
}
