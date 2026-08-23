//! Stereo camera pairs with interpupillary distance (IPD).

use super::pose::CameraPose;
use threers::math::Vector3;

/// Left / right eye poses for stereoscopic rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPair {
    pub left: CameraPose,
    pub right: CameraPose,
}

/// Where the stereo pivot sits (Blender stereo pivot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StereoPivot {
    #[default]
    Center,
    Left,
    Right,
}

/// Stereo rig: offsets a base pose along its local right axis by ±IPD/2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoRig {
    /// Interpupillary distance in world units (e.g. 0.065 for metres).
    pub ipd: f32,
    /// When true, both eyes look at the base target (toe-in). When false,
    /// eyes stay parallel (shift-sensor style).
    pub converge: bool,
    pub pivot: StereoPivot,
    /// Spherical stereo: eyes sit on a sphere around the target (VR180-style).
    pub spherical: bool,
}

impl Default for StereoRig {
    fn default() -> Self {
        Self {
            ipd: 0.065,
            converge: true,
            pivot: StereoPivot::Center,
            spherical: false,
        }
    }
}

impl StereoRig {
    pub fn new(ipd: f32) -> Self {
        Self {
            ipd: ipd.max(0.0),
            ..Self::default()
        }
    }

    pub fn parallel(ipd: f32) -> Self {
        Self {
            ipd: ipd.max(0.0),
            converge: false,
            ..Self::default()
        }
    }

    pub fn with_pivot(mut self, pivot: StereoPivot) -> Self {
        self.pivot = pivot;
        self
    }

    pub fn spherical(mut self, enabled: bool) -> Self {
        self.spherical = enabled;
        self
    }

    fn basis(pose: &CameraPose) -> (Vector3, Vector3, Vector3) {
        let eye = pose.eye();
        let forward = (pose.target - eye).normalize();
        let forward = if forward.length_sq() < 1e-10 {
            Vector3::new(0.0, 0.0, -1.0)
        } else {
            forward
        };
        let right = {
            let r = forward.cross(pose.up());
            if r.length_sq() < 1e-10 {
                Vector3::RIGHT
            } else {
                r.normalize()
            }
        };
        let up = right.cross(forward).normalize();
        (right, up, forward)
    }

    pub fn pair(&self, pose: CameraPose) -> StereoPair {
        let (right, _up, forward) = Self::basis(&pose);
        let half = self.ipd * 0.5;
        let eye = pose.eye();

        let (left_off, right_off) = match self.pivot {
            StereoPivot::Center => (-half, half),
            StereoPivot::Left => (0.0, self.ipd),
            StereoPivot::Right => (-self.ipd, 0.0),
        };

        let (left_eye, right_eye) = if self.spherical {
            let radius = (eye - pose.target).length().max(1e-4);
            let angle = (self.ipd * 0.5) / radius;
            let rot = |sign: f32| {
                let c = (sign * angle).cos();
                let s = (sign * angle).sin();
                let offset = eye - pose.target;
                let flat = Vector3::new(offset.x, 0.0, offset.z);
                let len = flat.length().max(1e-8);
                let nx = flat.x / len * c - flat.z / len * s;
                let nz = flat.x / len * s + flat.z / len * c;
                pose.target + Vector3::new(nx * len, offset.y, nz * len)
            };
            let _ = (forward, right);
            (rot(-1.0), rot(1.0))
        } else {
            (eye + right * left_off, eye + right * right_off)
        };

        let (left_target, right_target) = if self.converge {
            (pose.target, pose.target)
        } else {
            (
                pose.target + right * left_off,
                pose.target + right * right_off,
            )
        };

        let mut left = CameraPose::from_look_at(left_eye, left_target, pose.fov);
        let mut right_p = CameraPose::from_look_at(right_eye, right_target, pose.fov);
        for p in [&mut left, &mut right_p] {
            p.roll = pose.roll;
            p.focus_distance = pose.focus_distance;
            p.aperture = pose.aperture;
            p.shift_x = pose.shift_x;
            p.shift_y = pose.shift_y;
            p.f_stop = pose.f_stop;
            p.aperture_blades = pose.aperture_blades;
            p.anamorphic_ratio = pose.anamorphic_ratio;
            p.projection = pose.projection;
            p.sensor_fit = pose.sensor_fit;
        }
        StereoPair {
            left,
            right: right_p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipd_separates_eyes() {
        let pose = CameraPose::new(Vector3::ZERO, 0.0, std::f32::consts::FRAC_PI_2, 5.0);
        let pair = StereoRig::new(0.06).pair(pose);
        let sep = (pair.right.eye() - pair.left.eye()).length();
        assert!((sep - 0.06).abs() < 1e-3, "{sep}");
    }

    #[test]
    fn left_pivot_keeps_left_eye() {
        let pose = CameraPose::new(Vector3::ZERO, 0.0, std::f32::consts::FRAC_PI_2, 5.0);
        let eye = pose.eye();
        let pair = StereoRig::new(0.06)
            .with_pivot(StereoPivot::Left)
            .pair(pose);
        assert!((pair.left.eye() - eye).length() < 1e-3);
    }

    #[test]
    fn converge_shares_target() {
        let pose = CameraPose::new(Vector3::ZERO, 0.5, 1.2, 4.0);
        let pair = StereoRig::new(0.06).pair(pose);
        assert!((pair.left.target - pair.right.target).length() < 1e-5);
    }
}
