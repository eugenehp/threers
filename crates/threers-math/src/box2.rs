use super::Vector2;

/// Axis-aligned bounding box in 2D. Mirrors three.js's `Box2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box2 {
    pub min: Vector2,
    pub max: Vector2,
}

impl Default for Box2 {
    fn default() -> Self {
        Self::empty()
    }
}

impl Box2 {
    pub const fn new(min: Vector2, max: Vector2) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Vector2::new(f32::INFINITY, f32::INFINITY),
            max: Vector2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y
    }

    pub fn from_points(points: &[Vector2]) -> Self {
        let mut b = Self::empty();
        for p in points {
            b.expand_by_point(*p);
        }
        b
    }

    pub fn from_center_and_size(center: Vector2, size: Vector2) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn expand_by_point(&mut self, p: Vector2) -> &mut Self {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
        self
    }

    pub fn expand_by_vector(&mut self, v: Vector2) -> &mut Self {
        self.min = self.min - v;
        self.max = self.max + v;
        self
    }

    pub fn expand_by_scalar(&mut self, s: f32) -> &mut Self {
        self.expand_by_vector(Vector2::new(s, s))
    }

    pub fn center(&self) -> Vector2 {
        if self.is_empty() {
            Vector2::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    pub fn size(&self) -> Vector2 {
        if self.is_empty() {
            Vector2::ZERO
        } else {
            self.max - self.min
        }
    }

    pub fn contains_point(&self, p: Vector2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn contains_box(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && other.max.x <= self.max.x
            && self.min.y <= other.min.y
            && other.max.y <= self.max.y
    }

    pub fn intersects_box(&self, other: &Self) -> bool {
        !(other.max.x < self.min.x
            || other.min.x > self.max.x
            || other.max.y < self.min.y
            || other.min.y > self.max.y)
    }

    pub fn clamp_point(&self, p: Vector2) -> Vector2 {
        p.clamp(self.min, self.max)
    }

    pub fn distance_to_point(&self, p: Vector2) -> f32 {
        self.clamp_point(p).distance_to(p)
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    pub fn translate(&self, offset: Vector2) -> Self {
        Self {
            min: self.min + offset,
            max: self.max + offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points_encloses_all() {
        let pts = [Vector2::new(-1.0, 2.0), Vector2::new(3.0, -2.0)];
        let b = Box2::from_points(&pts);
        assert_eq!(b.min, Vector2::new(-1.0, -2.0));
        assert_eq!(b.max, Vector2::new(3.0, 2.0));
    }
}
