use super::{Box3, Matrix4, Vector3};

/// Bounding sphere. Mirrors three.js's `math::Sphere` (distinct from `SphereGeometry`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vector3,
    pub radius: f32,
}

impl Default for Sphere {
    fn default() -> Self {
        Self::empty()
    }
}

impl Sphere {
    pub const fn new(center: Vector3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// An empty sphere has negative radius so any expand-by-point produces a valid sphere.
    pub const fn empty() -> Self {
        Self {
            center: Vector3::ZERO,
            radius: -1.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.radius < 0.0
    }

    /// Welzl-free approximation: center on the centroid, radius = max distance to any point.
    /// Matches three.js's `Sphere.setFromPoints` (without optional `optionalCenter`).
    pub fn from_points(points: &[Vector3]) -> Self {
        if points.is_empty() {
            return Self::empty();
        }
        let bb = Box3::from_points(points);
        let center = bb.center();
        let mut max_r2 = 0.0f32;
        for p in points {
            let d2 = (*p - center).length_sq();
            if d2 > max_r2 {
                max_r2 = d2;
            }
        }
        Self {
            center,
            radius: max_r2.sqrt(),
        }
    }

    pub fn from_box3(b: &Box3) -> Self {
        if b.is_empty() {
            return Self::empty();
        }
        let center = b.center();
        let half = b.size() * 0.5;
        Self {
            center,
            radius: half.length(),
        }
    }

    pub fn expand_by_point(&mut self, p: Vector3) -> &mut Self {
        if self.is_empty() {
            self.center = p;
            self.radius = 0.0;
            return self;
        }
        let d = (p - self.center).length();
        if d > self.radius {
            self.radius = d;
        }
        self
    }

    pub fn contains_point(&self, p: Vector3) -> bool {
        (p - self.center).length_sq() <= self.radius * self.radius
    }

    pub fn distance_to_point(&self, p: Vector3) -> f32 {
        (p - self.center).length() - self.radius
    }

    pub fn intersects_sphere(&self, other: &Self) -> bool {
        let r = self.radius + other.radius;
        (other.center - self.center).length_sq() <= r * r
    }

    pub fn intersects_box(&self, b: &Box3) -> bool {
        b.distance_to_point(self.center) <= self.radius
    }

    pub fn clamp_point(&self, p: Vector3) -> Vector3 {
        let dir = p - self.center;
        let d2 = dir.length_sq();
        if d2 <= self.radius * self.radius {
            return p;
        }
        self.center + dir.normalize() * self.radius
    }

    pub fn translate(&self, offset: Vector3) -> Self {
        Self {
            center: self.center + offset,
            radius: self.radius,
        }
    }

    /// Apply a Matrix4 transform. Radius scales by the largest axis scale of `m`.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        let e = &m.elements;
        let w = e[3] * self.center.x + e[7] * self.center.y + e[11] * self.center.z + e[15];
        let inv_w = if w == 0.0 { 1.0 } else { 1.0 / w };
        let center = Vector3::new(
            (e[0] * self.center.x + e[4] * self.center.y + e[8] * self.center.z + e[12]) * inv_w,
            (e[1] * self.center.x + e[5] * self.center.y + e[9] * self.center.z + e[13]) * inv_w,
            (e[2] * self.center.x + e[6] * self.center.y + e[10] * self.center.z + e[14]) * inv_w,
        );
        let sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();
        let s = sx.max(sy).max(sz);
        Self {
            center,
            radius: self.radius * s,
        }
    }

    pub fn bounding_box(&self) -> Box3 {
        if self.is_empty() {
            return Box3::empty();
        }
        let r = Vector3::new(self.radius, self.radius, self.radius);
        Box3::new(self.center - r, self.center + r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points_encloses_all() {
        let pts = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ];
        let s = Sphere::from_points(&pts);
        for p in pts {
            assert!(s.contains_point(p), "should contain {:?}", p);
        }
    }

    #[test]
    fn intersects_sphere_when_overlapping() {
        let a = Sphere::new(Vector3::ZERO, 1.0);
        let b = Sphere::new(Vector3::new(1.5, 0.0, 0.0), 1.0);
        assert!(a.intersects_sphere(&b));
        let c = Sphere::new(Vector3::new(3.0, 0.0, 0.0), 1.0);
        assert!(!a.intersects_sphere(&c));
    }
}
