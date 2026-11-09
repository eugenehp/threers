use super::{Box3, Matrix4, Plane, Sphere, Vector3};

/// View frustum as six planes (right, left, bottom, top, far, near).
/// Mirrors three.js's `Frustum`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    pub const fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Extract frustum planes from a combined projection * view matrix, for the
    /// **WebGPU** clip space this crate projects into: x and y in -1..1, but
    /// depth in 0..1.
    ///
    /// The four side planes are the usual `row3 ± rowN`. The NEAR plane is not:
    /// at depth 0..1 it is `row2` alone, where the OpenGL form would be
    /// `row3 + row2`. Using the OpenGL form put the extracted near plane at the
    /// midpoint of the frustum, so culling discarded everything in the closer
    /// half — which was consistent with `Matrix4::perspective` while that was
    /// also OpenGL, and wrong for `orthographic`, which has always been 0..1.
    pub fn from_projection_matrix(m: &Matrix4) -> Self {
        let me = &m.elements;
        let p0 = Plane::new(
            Vector3::new(me[3] - me[0], me[7] - me[4], me[11] - me[8]),
            me[15] - me[12],
        )
        .normalize();
        let p1 = Plane::new(
            Vector3::new(me[3] + me[0], me[7] + me[4], me[11] + me[8]),
            me[15] + me[12],
        )
        .normalize();
        let p2 = Plane::new(
            Vector3::new(me[3] + me[1], me[7] + me[5], me[11] + me[9]),
            me[15] + me[13],
        )
        .normalize();
        let p3 = Plane::new(
            Vector3::new(me[3] - me[1], me[7] - me[5], me[11] - me[9]),
            me[15] - me[13],
        )
        .normalize();
        let p4 = Plane::new(
            Vector3::new(me[3] - me[2], me[7] - me[6], me[11] - me[10]),
            me[15] - me[14],
        )
        .normalize();
        let p5 = Plane::new(Vector3::new(me[2], me[6], me[10]), me[14]).normalize();
        Self {
            planes: [p0, p1, p2, p3, p4, p5],
        }
    }

    pub fn contains_point(&self, p: Vector3) -> bool {
        self.planes.iter().all(|pl| pl.distance_to_point(p) >= 0.0)
    }

    pub fn intersects_sphere(&self, s: &Sphere) -> bool {
        let neg_r = -s.radius;
        self.planes
            .iter()
            .all(|p| p.distance_to_point(s.center) >= neg_r)
    }

    pub fn intersects_box(&self, b: &Box3) -> bool {
        for plane in &self.planes {
            let p = Vector3::new(
                if plane.normal.x > 0.0 {
                    b.max.x
                } else {
                    b.min.x
                },
                if plane.normal.y > 0.0 {
                    b.max.y
                } else {
                    b.min.y
                },
                if plane.normal.z > 0.0 {
                    b.max.z
                } else {
                    b.min.z
                },
            );
            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_from_perspective_contains_origin_offset() {
        let proj = Matrix4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let f = Frustum::from_projection_matrix(&proj);
        // A point in front of the camera (negative Z, right-handed) should be inside.
        assert!(f.contains_point(Vector3::new(0.0, 0.0, -5.0)));
        // A point behind the camera should be outside.
        assert!(!f.contains_point(Vector3::new(0.0, 0.0, 5.0)));
    }
}
