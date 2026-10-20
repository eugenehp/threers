use super::light_helpers::line_object;
use super::three_lines;
use crate::core::Object3D;
use crate::math::{Box3, Color};

/// Wireframe box around an AABB. three.js's `BoxHelper`, taking the box
/// directly rather than an object's bounding box.
pub struct BoxHelper;

impl BoxHelper {
    pub fn new(bb: &Box3, color: Color) -> Object3D {
        let v = |p: crate::math::Vector3| [p.x as f64, p.y as f64, p.z as f64];
        line_object(&three_lines::box_lines(v(bb.min), v(bb.max)), color)
    }
}
