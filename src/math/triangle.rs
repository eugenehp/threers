use super::{Plane, Vector3};

/// Mirrors three.js's `Triangle`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Vector3,
    pub b: Vector3,
    pub c: Vector3,
}

impl Default for Triangle {
    fn default() -> Self {
        Self {
            a: Vector3::ZERO,
            b: Vector3::ZERO,
            c: Vector3::ZERO,
        }
    }
}

impl Triangle {
    pub const fn new(a: Vector3, b: Vector3, c: Vector3) -> Self {
        Self { a, b, c }
    }

    pub fn area(&self) -> f32 {
        (self.b - self.a).cross(self.c - self.a).length() * 0.5
    }

    pub fn midpoint(&self) -> Vector3 {
        (self.a + self.b + self.c) * (1.0 / 3.0)
    }

    pub fn normal(&self) -> Vector3 {
        let n = (self.c - self.b).cross(self.a - self.b);
        let len_sq = n.length_sq();
        if len_sq > 0.0 {
            n * (1.0 / len_sq.sqrt())
        } else {
            Vector3::ZERO
        }
    }

    pub fn plane(&self) -> Plane {
        Plane::from_coplanar_points(self.a, self.b, self.c)
    }

    /// Barycentric coordinates of `p` in this triangle. Returns `(u, v, w)` with `u+v+w == 1`
    /// when `p` lies in the triangle's plane. Mirrors three.js's `getBarycoord`.
    pub fn barycoord(&self, p: Vector3) -> Vector3 {
        let v0 = self.c - self.a;
        let v1 = self.b - self.a;
        let v2 = p - self.a;
        let dot00 = v0.dot(v0);
        let dot01 = v0.dot(v1);
        let dot02 = v0.dot(v2);
        let dot11 = v1.dot(v1);
        let dot12 = v1.dot(v2);
        let denom = dot00 * dot11 - dot01 * dot01;
        if denom == 0.0 {
            return Vector3::new(-2.0, -1.0, -1.0);
        }
        let inv = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv;
        let v = (dot00 * dot12 - dot01 * dot02) * inv;
        Vector3::new(1.0 - u - v, v, u)
    }

    pub fn contains_point(&self, p: Vector3) -> bool {
        let bc = self.barycoord(p);
        bc.x >= 0.0 && bc.y >= 0.0 && bc.z >= 0.0
    }

    /// Triangle–triangle intersection (including coplanar cases).
    ///
    /// Available with the `bvh-csg` feature; delegates to the f64 ExtendedTriangle
    /// path used by the CSG port.
    #[cfg(feature = "bvh-csg")]
    pub fn intersects_triangle(&self, other: &Triangle) -> bool {
        self.intersects_triangle_ext(other, true)
    }

    /// three-mesh-bvh's `ExtendedTriangle.intersectsTriangle`, in f64 as the
    /// CSG port runs it. The flag is three-mesh-bvh's third argument,
    /// `suppressLog`, which only silences its coplanar warning; it does not
    /// change the answer.
    #[cfg(feature = "bvh-csg")]
    pub fn intersects_triangle_ext(&self, other: &Triangle, _suppress_log: bool) -> bool {
        use crate::csg::js_topology::JsTriangle;
        JsTriangle::from_triangle(*self).intersects_triangle_js(JsTriangle::from_triangle(*other))
    }

    /// True if `direction` points against the triangle's normal.
    pub fn is_front_facing(&self, direction: Vector3) -> bool {
        (self.b - self.a).cross(self.c - self.a).dot(direction) < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_of_unit_right_triangle() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((t.area() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normal_of_xy_triangle_points_z() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let n = t.normal();
        assert!((n.z - 1.0).abs() < 1e-5, "got {:?}", n);
    }

    #[test]
    fn contains_centroid() {
        let t = Triangle::new(
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!(t.contains_point(t.midpoint()));
    }
}
