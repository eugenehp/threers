use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D};
use crate::materials::LineBasicMaterial;
use crate::math::Color;

/// Line-segments visualization of each vertex's normal vector.
/// Mirrors three.js's `VertexNormalsHelper`.
pub struct VertexNormalsHelper;

impl VertexNormalsHelper {
    /// three.js's `VertexNormalsHelper` for a geometry at the origin: every
    /// vertex's normal, normalised, `length` long.
    pub fn new(geom: &BufferGeometry, length: f32, color: Color) -> Object3D {
        let (Some(pos), Some(nrm)) = (geom.get_attribute("position"), geom.get_attribute("normal"))
        else {
            return empty_line(color);
        };
        let g = super::three_lines::vertex_normals(
            &super::three_lines::IDENTITY,
            &pos.array,
            &nrm.array,
            length as f64,
        );
        super::light_helpers::line_object(&g, color)
    }
}

/// Visualization of each vertex's tangent vector. Requires the geometry to
/// carry a `tangent` (vec4) attribute. Mirrors three.js's `VertexTangentsHelper`.
pub struct VertexTangentsHelper;

impl VertexTangentsHelper {
    pub fn new(geom: &BufferGeometry, length: f32, color: Color) -> Object3D {
        let pos = match geom.get_attribute("position") {
            Some(p) => &p.array,
            None => return empty_line(color),
        };
        let tan = match geom.get_attribute("tangent") {
            Some(t) => &t.array,
            None => return empty_line(color),
        };
        let count = pos.len() / 3;
        let mut positions = Vec::with_capacity(count * 6);
        for i in 0..count {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];
            let tx = tan[i * 4];
            let ty = tan[i * 4 + 1];
            let tz = tan[i * 4 + 2];
            positions.extend_from_slice(&[
                px,
                py,
                pz,
                px + tx * length,
                py + ty * length,
                pz + tz * length,
            ]);
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        Object3D::line_segments(LineSegments::new(
        g,
        LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(color) }.into(),
    ))
    }
}

fn empty_line(color: Color) -> Object3D {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(Vec::new(), 3));
    Object3D::line_segments(LineSegments::new(
        g,
        LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(color) }.into(),
    ))
}
