use super::light_helpers::line_object;
use super::three_lines;
use crate::cameras::Camera;
use crate::core::Object3D;
use crate::math::{Color, Matrix4};

/// three.js's `CameraHelper`: frustum, cone, up triangle, line of sight and
/// cross, in three.js's colours — except the frustum, drawn in `color`.
pub struct CameraHelper;

impl CameraHelper {
    pub fn new(camera: &dyn Camera, color: Color) -> Object3D {
        let f64m = |m: Matrix4| -> three_lines::Mat4 { m.elements.map(|v| v as f64) };
        let projection_inverse = three_lines::invert(&f64m(camera.projection_matrix()));
        let world = three_lines::invert(&f64m(camera.view_matrix()));
        let mut colors = three_lines::camera_default_colors();
        colors[0] = [color.r, color.g, color.b];
        line_object(
            &three_lines::camera(&projection_inverse, &world, colors),
            Color::WHITE,
        )
    }
}
