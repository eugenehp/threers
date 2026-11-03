//! Auto-framing: pack a subject AABB into the view.

use threers::math::Vector3;

use super::pose::CameraPose;

/// Axis-aligned subject bounds used for framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramingBounds {
    pub min: Vector3,
    pub max: Vector3,
}

impl FramingBounds {
    pub fn from_center_size(center: Vector3, size: Vector3) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn center(self) -> Vector3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(self) -> Vector3 {
        self.max - self.min
    }

    /// Sphere that contains the box — conservative framing radius basis.
    pub fn bounding_radius(self) -> f32 {
        (self.size() * 0.5).length()
    }
}

/// How tightly to pack the subject in frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramingStyle {
    /// Fraction of the shorter view edge the subject should fill (`0..1`).
    pub fill: f32,
    /// Preferred elevation (φ). Default ≈ 55°.
    pub elevation: f32,
    /// Preferred azimuth (θ).
    pub azimuth: f32,
    /// Padding multiplier on computed radius (>1 = looser).
    pub padding: f32,
}

impl Default for FramingStyle {
    fn default() -> Self {
        Self {
            fill: 0.55,
            elevation: 1.0,
            azimuth: 0.6,
            padding: 1.15,
        }
    }
}

/// Compute a pose that frames `bounds` for the given FOV and aspect.
pub fn frame_bounds(bounds: FramingBounds, fov: f32, aspect: f32, style: FramingStyle) -> CameraPose {
    let fill = style.fill.clamp(0.05, 0.95);
    let radius_subject = bounds.bounding_radius().max(1e-4);
    let half_v = (fov * 0.5).tan().max(1e-6);
    let half_h = half_v * aspect.max(1e-6);
    let half = half_v.min(half_h);
    // Subject projected height ≈ 2 r; we want that to occupy `fill` of the view.
    let dist = (radius_subject / (half * fill)) * style.padding.max(1.0);
    CameraPose::new(bounds.center(), style.azimuth, style.elevation, dist).with_fov(fov)
}

/// Keep `subject` near a normalised screen point (`nx, ny` in -1..1 NDC-ish,
/// y up). Returns a new target offset suggestion — callers blend toward it.
pub fn screen_offset_for(
    eye: Vector3,
    subject: Vector3,
    fov: f32,
    aspect: f32,
    nx: f32,
    ny: f32,
) -> Vector3 {
    let forward = (subject - eye).normalize();
    if forward.length_sq() < 1e-10 {
        return subject;
    }
    let right = {
        let r = forward.cross(Vector3::UP);
        if r.length_sq() < 1e-10 {
            Vector3::RIGHT
        } else {
            r.normalize()
        }
    };
    let up = right.cross(forward).normalize();
    let dist = (subject - eye).length();
    let half_v = (fov * 0.5).tan();
    let half_h = half_v * aspect;
    // Shift target so subject lands at (nx, ny) in the view plane.
    subject - right * (nx * half_h * dist) - up * (ny * half_v * dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_puts_camera_outside_the_box() {
        let b = FramingBounds::from_center_size(Vector3::ZERO, Vector3::new(2.0, 2.0, 2.0));
        let pose = frame_bounds(b, 50.0_f32.to_radians(), 16.0 / 9.0, FramingStyle::default());
        assert!(pose.radius > 1.0);
        assert!((pose.target - Vector3::ZERO).length() < 1e-4);
    }
}
