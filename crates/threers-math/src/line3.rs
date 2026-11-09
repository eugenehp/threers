use super::{Matrix4, Vector3};

/// Line segment in 3D. Mirrors three.js's `Line3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3 {
    pub start: Vector3,
    pub end: Vector3,
}

impl Default for Line3 {
    fn default() -> Self {
        Self {
            start: Vector3::ZERO,
            end: Vector3::ZERO,
        }
    }
}

impl Line3 {
    pub const fn new(start: Vector3, end: Vector3) -> Self {
        Self { start, end }
    }

    pub fn center(&self) -> Vector3 {
        (self.start + self.end) * 0.5
    }

    pub fn delta(&self) -> Vector3 {
        self.end - self.start
    }

    pub fn distance(&self) -> f32 {
        self.delta().length()
    }

    pub fn distance_sq(&self) -> f32 {
        self.delta().length_sq()
    }

    /// Point along the segment at parameter `t` (0..=1 for points on the segment).
    pub fn at(&self, t: f32) -> Vector3 {
        self.start + self.delta() * t
    }

    /// `t` of the closest point on the (possibly extended) line to `p`.
    /// If `clamp_to_line`, the parameter is clamped to `[0, 1]`.
    pub fn closest_point_to_point_parameter(&self, p: Vector3, clamp_to_line: bool) -> f32 {
        let d = self.delta();
        let denom = d.dot(d);
        if denom == 0.0 {
            return 0.0;
        }
        let t = (p - self.start).dot(d) / denom;
        if clamp_to_line {
            t.clamp(0.0, 1.0)
        } else {
            t
        }
    }

    pub fn closest_point_to_point(&self, p: Vector3, clamp_to_line: bool) -> Vector3 {
        self.at(self.closest_point_to_point_parameter(p, clamp_to_line))
    }

    pub fn apply_matrix4(&self, m: &Matrix4) -> Self {
        Self {
            start: transform_point(m, self.start),
            end: transform_point(m, self.end),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint() {
        let l = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
        assert_eq!(l.center(), Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn closest_clamped_to_endpoint() {
        let l = Line3::new(Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0));
        let p = Vector3::new(2.0, 1.0, 0.0);
        let c = l.closest_point_to_point(p, true);
        assert_eq!(c, Vector3::new(1.0, 0.0, 0.0));
    }
}
