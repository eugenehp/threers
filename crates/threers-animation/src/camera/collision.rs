//! Collision / volume avoidance for fly-through cameras.

use threers::math::Vector3;

/// Simple spherical keep-out volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionVolume {
    pub center: Vector3,
    pub radius: f32,
}

impl CollisionVolume {
    pub fn new(center: Vector3, radius: f32) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
        }
    }
}

/// Axis-aligned box keep-out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionBox {
    pub min: Vector3,
    pub max: Vector3,
}

impl CollisionBox {
    pub fn from_center_size(center: Vector3, size: Vector3) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }
}

/// Push a camera eye out of keep-out volumes along the look ray.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraCollision {
    pub spheres: Vec<CollisionVolume>,
    pub boxes: Vec<CollisionBox>,
    /// Extra clearance beyond volume surfaces.
    pub padding: f32,
    /// When the eye is pulled in, never get closer than this to the target.
    pub min_distance: f32,
}

impl CameraCollision {
    pub fn new() -> Self {
        Self {
            padding: 0.1,
            min_distance: 0.2,
            ..Self::default()
        }
    }

    pub fn push_sphere(&mut self, volume: CollisionVolume) {
        self.spheres.push(volume);
    }

    pub fn push_box(&mut self, volume: CollisionBox) {
        self.boxes.push(volume);
    }

    /// Resolve `eye` so it sits outside all volumes, sliding along `target - eye`.
    pub fn resolve_eye(&self, mut eye: Vector3, target: Vector3) -> Vector3 {
        let padding = self.padding.max(0.0);
        let min_dist = self.min_distance.max(1e-4);

        // Sphere push-outs.
        for s in &self.spheres {
            let r = s.radius + padding;
            let offset = eye - s.center;
            let d = offset.length();
            if d < r && d > 1e-8 {
                eye = s.center + offset * (r / d);
            } else if d <= 1e-8 {
                // Degenerate — push along look axis.
                let away = (eye - target).normalize();
                let dir = if away.length_sq() < 1e-10 {
                    Vector3::UP
                } else {
                    away
                };
                eye = s.center + dir * r;
            }
        }

        // AABB push-outs (nearest-face).
        for b in &self.boxes {
            let min = Vector3::new(b.min.x - padding, b.min.y - padding, b.min.z - padding);
            let max = Vector3::new(b.max.x + padding, b.max.y + padding, b.max.z + padding);
            if eye.x >= min.x
                && eye.x <= max.x
                && eye.y >= min.y
                && eye.y <= max.y
                && eye.z >= min.z
                && eye.z <= max.z
            {
                let dx = (eye.x - min.x).min(max.x - eye.x);
                let dy = (eye.y - min.y).min(max.y - eye.y);
                let dz = (eye.z - min.z).min(max.z - eye.z);
                if dx <= dy && dx <= dz {
                    eye.x = if eye.x - min.x < max.x - eye.x {
                        min.x
                    } else {
                        max.x
                    };
                } else if dy <= dz {
                    eye.y = if eye.y - min.y < max.y - eye.y {
                        min.y
                    } else {
                        max.y
                    };
                } else {
                    eye.z = if eye.z - min.z < max.z - eye.z {
                        min.z
                    } else {
                        max.z
                    };
                }
            }
        }

        // Never collapse onto the target.
        let to_eye = eye - target;
        let dist = to_eye.length();
        if dist < min_dist {
            let dir = if dist < 1e-8 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                to_eye * (1.0 / dist)
            };
            eye = target + dir * min_dist;
        }
        eye
    }

    /// Resolve a full pose (rebuilds spherical from corrected eye + target).
    pub fn resolve_pose(&self, pose: super::pose::CameraPose) -> super::pose::CameraPose {
        let eye = self.resolve_eye(pose.eye(), pose.target);
        let mut out = super::pose::CameraPose::from_look_at(eye, pose.target, pose.fov);
        out.roll = pose.roll;
        out.focus_distance = pose.focus_distance;
        out.aperture = pose.aperture;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_pushes_eye_out() {
        let mut col = CameraCollision::new();
        col.padding = 0.0;
        col.push_sphere(CollisionVolume::new(Vector3::ZERO, 2.0));
        let eye = col.resolve_eye(Vector3::new(0.5, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0));
        assert!(eye.length() >= 2.0 - 1e-3, "{eye:?}");
    }

    #[test]
    fn min_distance_enforced() {
        let col = CameraCollision {
            min_distance: 1.0,
            ..CameraCollision::new()
        };
        let eye = col.resolve_eye(Vector3::new(0.1, 0.0, 0.0), Vector3::ZERO);
        assert!((eye.length() - 1.0).abs() < 1e-3);
    }
}
