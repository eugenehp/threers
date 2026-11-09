use super::{Box3, Matrix3, Matrix4, Sphere, Vector3};

/// Plane in Hessian-normal form: dot(normal, p) + constant == 0.
/// Mirrors three.js's `Plane`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vector3,
    pub constant: f32,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            normal: Vector3::new(1.0, 0.0, 0.0),
            constant: 0.0,
        }
    }
}

impl Plane {
    pub const fn new(normal: Vector3, constant: f32) -> Self {
        Self { normal, constant }
    }

    pub fn from_normal_and_coplanar_point(normal: Vector3, point: Vector3) -> Self {
        Self {
            normal,
            constant: -point.dot(normal),
        }
    }

    pub fn from_coplanar_points(a: Vector3, b: Vector3, c: Vector3) -> Self {
        let normal = (c - b).cross(a - b).normalize();
        Self::from_normal_and_coplanar_point(normal, a)
    }

    pub fn normalize(&self) -> Self {
        let inv = 1.0 / self.normal.length();
        Self {
            normal: self.normal * inv,
            constant: self.constant * inv,
        }
    }

    pub fn negate(&self) -> Self {
        Self {
            normal: -self.normal,
            constant: -self.constant,
        }
    }

    pub fn distance_to_point(&self, p: Vector3) -> f32 {
        self.normal.dot(p) + self.constant
    }

    /// Intersection with a line segment; returns the hit point when the line crosses the plane.
    pub fn intersect_line(&self, line: &crate::Line3) -> Option<Vector3> {
        const EPS: f32 = 1e-10;
        let d1 = self.distance_to_point(line.start);
        let d2 = self.distance_to_point(line.end);
        if d1 * d2 < 0.0 {
            let t = d1 / (d1 - d2);
            return Some(line.start + (line.end - line.start) * t);
        }
        if d1.abs() < EPS {
            return Some(line.start);
        }
        None
    }

    pub fn distance_to_sphere(&self, s: &Sphere) -> f32 {
        self.distance_to_point(s.center) - s.radius
    }

    pub fn project_point(&self, p: Vector3) -> Vector3 {
        p - self.normal * self.distance_to_point(p)
    }

    pub fn coplanar_point(&self) -> Vector3 {
        self.normal * -self.constant
    }

    pub fn intersects_sphere(&self, s: &Sphere) -> bool {
        self.distance_to_point(s.center).abs() <= s.radius
    }

    pub fn intersects_box(&self, b: &Box3) -> bool {
        let (min_p, max_p) = if self.normal.x > 0.0 {
            (b.min.x, b.max.x)
        } else {
            (b.max.x, b.min.x)
        };
        let (min_y, max_y) = if self.normal.y > 0.0 {
            (b.min.y, b.max.y)
        } else {
            (b.max.y, b.min.y)
        };
        let (min_z, max_z) = if self.normal.z > 0.0 {
            (b.min.z, b.max.z)
        } else {
            (b.max.z, b.min.z)
        };
        let min_p = Vector3::new(min_p, min_y, min_z);
        let max_p = Vector3::new(max_p, max_y, max_z);
        self.distance_to_point(min_p) <= 0.0 && self.distance_to_point(max_p) >= 0.0
    }

    pub fn translate(&self, offset: Vector3) -> Self {
        Self {
            normal: self.normal,
            constant: self.constant - offset.dot(self.normal),
        }
    }

    /// `normal_matrix` is `Matrix3::normal_matrix(m)` — caller is expected to precompute it
    /// when transforming many planes by the same matrix (matches three.js's API contract).
    pub fn apply_matrix4(&self, m: &Matrix4, normal_matrix: Option<&Matrix3>) -> Self {
        let owned;
        let nm = match normal_matrix {
            Some(n) => n,
            None => {
                owned = Matrix3::normal_matrix(m);
                &owned
            }
        };
        let ref_point = transform_point(m, self.coplanar_point());
        let normal = transform_normal(nm, self.normal).normalize();
        Self {
            normal,
            constant: -ref_point.dot(normal),
        }
    }
}

fn transform_point(m: &Matrix4, p: Vector3) -> Vector3 {
    let e = &m.elements;
    let w = e[3] * p.x + e[7] * p.y + e[11] * p.z + e[15];
    let inv_w = if w == 0.0 { 1.0 } else { 1.0 / w };
    Vector3::new(
        (e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12]) * inv_w,
        (e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13]) * inv_w,
        (e[2] * p.x + e[6] * p.y + e[10] * p.z + e[14]) * inv_w,
    )
}

fn transform_normal(m: &Matrix3, n: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * n.x + e[3] * n.y + e[6] * n.z,
        e[1] * n.x + e[4] * n.y + e[7] * n.z,
        e[2] * n.x + e[5] * n.y + e[8] * n.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_to_origin_for_x_plane_at_5() {
        let p = Plane::new(Vector3::new(1.0, 0.0, 0.0), -5.0);
        assert!((p.distance_to_point(Vector3::ZERO) - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn project_point_onto_xy_plane() {
        let p = Plane::new(Vector3::new(0.0, 0.0, 1.0), 0.0);
        let q = p.project_point(Vector3::new(2.0, 3.0, 7.0));
        assert!((q.z - 0.0).abs() < 1e-6);
        assert!((q.x - 2.0).abs() < 1e-6);
        assert!((q.y - 3.0).abs() < 1e-6);
    }
}
