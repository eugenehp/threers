use super::{Matrix4, Vector3};

/// Axis-aligned bounding box in 3D. Mirrors three.js's `Box3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box3 {
    pub min: Vector3,
    pub max: Vector3,
}

impl Default for Box3 {
    fn default() -> Self {
        Self::empty()
    }
}

impl Box3 {
    pub const fn new(min: Vector3, max: Vector3) -> Self {
        Self { min, max }
    }

    /// An empty box: min = +inf, max = -inf, so any expand_by_point gives the right result.
    pub fn empty() -> Self {
        Self {
            min: Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y || self.max.z < self.min.z
    }

    pub fn from_points(points: &[Vector3]) -> Self {
        let mut b = Self::empty();
        for p in points {
            b.expand_by_point(*p);
        }
        b
    }

    pub fn from_center_and_size(center: Vector3, size: Vector3) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn expand_by_point(&mut self, p: Vector3) -> &mut Self {
        self.min = Vector3::new(
            self.min.x.min(p.x),
            self.min.y.min(p.y),
            self.min.z.min(p.z),
        );
        self.max = Vector3::new(
            self.max.x.max(p.x),
            self.max.y.max(p.y),
            self.max.z.max(p.z),
        );
        self
    }

    pub fn expand_by_vector(&mut self, v: Vector3) -> &mut Self {
        self.min = self.min - v;
        self.max = self.max + v;
        self
    }

    pub fn expand_by_scalar(&mut self, s: f32) -> &mut Self {
        self.expand_by_vector(Vector3::new(s, s, s))
    }

    pub fn center(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    pub fn size(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::ZERO
        } else {
            self.max - self.min
        }
    }

    pub fn contains_point(&self, p: Vector3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn contains_box(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && other.max.x <= self.max.x
            && self.min.y <= other.min.y
            && other.max.y <= self.max.y
            && self.min.z <= other.min.z
            && other.max.z <= self.max.z
    }

    pub fn intersects_box(&self, other: &Self) -> bool {
        !(other.max.x < self.min.x
            || other.min.x > self.max.x
            || other.max.y < self.min.y
            || other.min.y > self.max.y
            || other.max.z < self.min.z
            || other.min.z > self.max.z)
    }

    pub fn clamp_point(&self, p: Vector3) -> Vector3 {
        Vector3::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
            p.z.clamp(self.min.z, self.max.z),
        )
    }

    pub fn distance_to_point(&self, p: Vector3) -> f32 {
        self.clamp_point(p).distance_to(p)
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            min: Vector3::new(
                self.min.x.max(other.min.x),
                self.min.y.max(other.min.y),
                self.min.z.max(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.min(other.max.x),
                self.max.y.min(other.max.y),
                self.max.z.min(other.max.z),
            ),
        }
    }

    pub fn translate(&self, offset: Vector3) -> Self {
        Self {
            min: self.min + offset,
            max: self.max + offset,
        }
    }

    /// Transform the box by `m`, returning the AABB enclosing the transformed corners.
    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        if self.is_empty() {
            return *self;
        }
        let corners = [
            Vector3::new(self.min.x, self.min.y, self.min.z),
            Vector3::new(self.min.x, self.min.y, self.max.z),
            Vector3::new(self.min.x, self.max.y, self.min.z),
            Vector3::new(self.min.x, self.max.y, self.max.z),
            Vector3::new(self.max.x, self.min.y, self.min.z),
            Vector3::new(self.max.x, self.min.y, self.max.z),
            Vector3::new(self.max.x, self.max.y, self.min.z),
            Vector3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut out = Self::empty();
        for c in corners {
            out.expand_by_point(transform_point(m, c));
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_box_is_empty() {
        assert!(Box3::empty().is_empty());
    }

    #[test]
    fn from_points_encloses_all() {
        let pts = [
            Vector3::new(-1.0, 0.0, 2.0),
            Vector3::new(3.0, -2.0, 1.0),
            Vector3::new(0.0, 5.0, -1.0),
        ];
        let b = Box3::from_points(&pts);
        for p in pts {
            assert!(b.contains_point(p));
        }
        assert_eq!(b.min, Vector3::new(-1.0, -2.0, -1.0));
        assert_eq!(b.max, Vector3::new(3.0, 5.0, 2.0));
    }

    #[test]
    fn intersects_box_overlap() {
        let a = Box3::new(Vector3::ZERO, Vector3::ONE);
        let b = Box3::new(Vector3::new(0.5, 0.5, 0.5), Vector3::new(2.0, 2.0, 2.0));
        assert!(a.intersects_box(&b));
        let c = Box3::new(Vector3::new(2.0, 2.0, 2.0), Vector3::new(3.0, 3.0, 3.0));
        assert!(!a.intersects_box(&c));
    }
}
