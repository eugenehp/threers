//! Follow Path with per-point tilt and radius (Blender curve follow).

use super::pose::CameraPose;
use threers::math::Vector3;

/// One control point on a tilted path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TiltedPoint {
    pub position: Vector3,
    /// Bank angle around the tangent (radians).
    pub tilt: f32,
    /// Path radius / scale at this point (dolly offset from curve).
    pub radius: f32,
}

impl TiltedPoint {
    pub fn new(position: Vector3) -> Self {
        Self {
            position,
            tilt: 0.0,
            radius: 0.0,
        }
    }

    pub fn with_tilt(mut self, tilt: f32) -> Self {
        self.tilt = tilt;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }
}

/// Polyline / Catmull-style path carrying tilt + radius.
#[derive(Debug, Clone, PartialEq)]
pub struct TiltedPath {
    pub points: Vec<TiltedPoint>,
    /// Look-at target; `None` = look along tangent.
    pub fixed_target: Option<Vector3>,
    pub look_ahead: f32,
}

impl TiltedPath {
    pub fn new(points: Vec<TiltedPoint>) -> Self {
        Self {
            points,
            fixed_target: None,
            look_ahead: 1.0,
        }
    }

    pub fn with_fixed_target(mut self, target: Vector3) -> Self {
        self.fixed_target = Some(target);
        self.look_ahead = 0.0;
        self
    }

    fn sample_point(&self, u: f32) -> (Vector3, f32, f32, Vector3) {
        let u = u.clamp(0.0, 1.0);
        if self.points.is_empty() {
            return (Vector3::ZERO, 0.0, 0.0, Vector3::new(0.0, 0.0, -1.0));
        }
        if self.points.len() == 1 {
            let p = self.points[0];
            return (p.position, p.tilt, p.radius, Vector3::new(0.0, 0.0, -1.0));
        }
        let n = self.points.len() - 1;
        let f = u * n as f32;
        let i = (f.floor() as usize).min(n - 1);
        let t = f - i as f32;
        let a = self.points[i];
        let b = self.points[i + 1];
        let pos = a.position.lerp(b.position, t);
        let tilt = a.tilt + (b.tilt - a.tilt) * t;
        let radius = a.radius + (b.radius - a.radius) * t;
        let tangent = (b.position - a.position).normalize();
        let tangent = if tangent.length_sq() < 1e-10 {
            Vector3::new(0.0, 0.0, -1.0)
        } else {
            tangent
        };
        (pos, tilt, radius, tangent)
    }

    pub fn sample_pose(&self, u: f32, template: CameraPose, follow_curve: bool) -> CameraPose {
        let (pos, tilt, radius, tangent) = self.sample_point(u);
        // Offset along curve normal (up × tangent) by radius.
        let up = Vector3::UP;
        let side = {
            let s = tangent.cross(up);
            if s.length_sq() < 1e-10 {
                Vector3::RIGHT
            } else {
                s.normalize()
            }
        };
        let eye = pos + side * radius;
        let target = if let Some(fixed) = self.fixed_target {
            fixed
        } else if self.look_ahead > 0.0 {
            let (ahead, _, _, _) = self.sample_point((u + 0.05).min(1.0));
            ahead
        } else {
            eye + tangent
        };
        let mut pose = CameraPose::from_look_at(eye, target, template.fov);
        if follow_curve {
            pose.roll = tilt;
        } else {
            pose.roll = template.roll + tilt;
        }
        pose.focus_distance = template.focus_distance;
        pose.aperture = template.aperture;
        pose.shift_x = template.shift_x;
        pose.shift_y = template.shift_y;
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilt_applies_roll() {
        let path = TiltedPath::new(vec![
            TiltedPoint::new(Vector3::new(0.0, 0.0, 0.0)).with_tilt(0.0),
            TiltedPoint::new(Vector3::new(10.0, 0.0, 0.0)).with_tilt(0.5),
        ])
        .with_fixed_target(Vector3::new(5.0, 0.0, -5.0));
        let pose = path.sample_pose(1.0, CameraPose::default(), true);
        assert!((pose.roll - 0.5).abs() < 1e-4);
    }
}
