use super::light_helpers::line_object;
use super::three_lines::{self, HelperGeometry};
use crate::core::Object3D;
use crate::math::{Color, Vector3};

/// three.js's `ArrowHelper` with the default head (a fifth of the length,
/// a fifth of that wide). Line objects only, so the cone is drawn as its
/// wireframe; the filled cone is in [`three_lines::arrow`]'s `mesh`.
pub struct ArrowHelper;

impl ArrowHelper {
    pub fn new(direction: Vector3, origin: Vector3, length: f32, color: Color) -> Object3D {
        let d = direction.normalize();
        let q = three_lines::arrow_quaternion([d.x as f64, d.y as f64, d.z as f64]);
        let world = three_lines::compose(
            [origin.x as f64, origin.y as f64, origin.z as f64],
            q,
            [1.0; 3],
        );
        let head = length as f64 * 0.2;
        let g = three_lines::arrow(&world, length as f64, head, head * 0.2);
        let mut lines = g.lines;
        for t in g.mesh.chunks_exact(9) {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                lines.extend_from_slice(&t[a * 3..a * 3 + 3]);
                lines.extend_from_slice(&t[b * 3..b * 3 + 3]);
            }
        }
        line_object(
            &HelperGeometry {
                lines,
                ..Default::default()
            },
            color,
        )
    }
}
