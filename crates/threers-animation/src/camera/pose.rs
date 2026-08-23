//! Spherical camera pose — the unit everything else animates.

use crate::animatable::Animatable;
use threers::cameras::{PerspectiveCamera, ProjectionKind, SensorFit};
use threers::math::{Spherical, Vector3};

/// Full cinematic camera state in spherical form around a look-at target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    pub target: Vector3,
    pub azimuth: f32,
    pub elevation: f32,
    pub radius: f32,
    pub roll: f32,
    pub fov: f32,
    pub focus_distance: f32,
    pub aperture: f32,
    pub f_stop: f32,
    pub aperture_blades: u32,
    pub anamorphic_ratio: f32,
    pub shift_x: f32,
    pub shift_y: f32,
    pub sensor_fit: SensorFit,
    pub projection: ProjectionKind,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            target: Vector3::ZERO,
            azimuth: 0.0,
            elevation: std::f32::consts::FRAC_PI_2,
            radius: 5.0,
            roll: 0.0,
            fov: 50.0_f32.to_radians(),
            focus_distance: 0.0,
            aperture: 0.0,
            f_stop: 0.0,
            aperture_blades: 0,
            anamorphic_ratio: 1.0,
            shift_x: 0.0,
            shift_y: 0.0,
            sensor_fit: SensorFit::Auto,
            projection: ProjectionKind::Perspective,
        }
    }
}

impl CameraPose {
    pub fn new(target: Vector3, azimuth: f32, elevation: f32, radius: f32) -> Self {
        Self {
            target,
            azimuth,
            elevation,
            radius: radius.max(1e-4),
            ..Self::default()
        }
    }

    pub fn from_camera(camera: &PerspectiveCamera) -> Self {
        let offset = camera.position - camera.target;
        let s = Spherical::from_vector3(offset).make_safe();
        let eye = camera.position;
        let forward = (camera.target - eye).normalize();
        let world_up = Vector3::UP;
        let right = forward.cross(world_up);
        let roll = if right.length_sq() < 1e-10 {
            0.0
        } else {
            let right = right.normalize();
            let expected_up = right.cross(forward).normalize();
            let actual_up = camera.up.normalize();
            let sin = expected_up.cross(actual_up).dot(forward);
            let cos = expected_up.dot(actual_up);
            sin.atan2(cos)
        };
        Self {
            target: camera.target,
            azimuth: s.theta,
            elevation: s.phi,
            radius: s.radius.max(1e-4),
            roll,
            fov: camera.fov,
            focus_distance: camera.focus_distance,
            aperture: camera.aperture,
            f_stop: camera.f_stop,
            aperture_blades: camera.aperture_blades,
            anamorphic_ratio: camera.anamorphic_ratio,
            shift_x: camera.shift_x,
            shift_y: camera.shift_y,
            sensor_fit: camera.sensor_fit,
            projection: camera.projection,
        }
    }

    pub fn eye(&self) -> Vector3 {
        self.target + self.spherical().to_vector3()
    }

    pub fn spherical(&self) -> Spherical {
        Spherical::new(self.radius.max(1e-4), self.elevation, self.azimuth)
    }

    pub fn up(&self) -> Vector3 {
        let eye = self.eye();
        let forward = (self.target - eye).normalize();
        let world_up = Vector3::UP;
        let right = {
            let r = forward.cross(world_up);
            if r.length_sq() < 1e-10 {
                Vector3::RIGHT
            } else {
                r.normalize()
            }
        };
        let up = right.cross(forward).normalize();
        if self.roll.abs() < 1e-8 {
            return up;
        }
        let c = self.roll.cos();
        let s = self.roll.sin();
        (up * c + right * s).normalize()
    }

    pub fn apply_to(&self, camera: &mut PerspectiveCamera) {
        camera.position = self.eye();
        camera.target = self.target;
        camera.up = self.up();
        camera.fov = self.fov.max(1e-3);
        camera.focus_distance = self.focus_distance.max(0.0);
        camera.aperture = self.aperture.max(0.0);
        camera.f_stop = self.f_stop.max(0.0);
        camera.aperture_blades = self.aperture_blades;
        camera.anamorphic_ratio = self.anamorphic_ratio.max(1e-3);
        camera.shift_x = self.shift_x;
        camera.shift_y = self.shift_y;
        camera.sensor_fit = self.sensor_fit;
        camera.projection = self.projection;
    }

    pub fn from_look_at(eye: Vector3, target: Vector3, fov: f32) -> Self {
        let offset = eye - target;
        let s = Spherical::from_vector3(offset).make_safe();
        Self {
            target,
            azimuth: s.theta,
            elevation: s.phi,
            radius: s.radius.max(1e-4),
            fov: fov.max(1e-3),
            ..Self::default()
        }
    }

    pub fn with_fov(mut self, fov: f32) -> Self {
        self.fov = fov.max(1e-3);
        self
    }

    pub fn with_vertigo_fov(mut self, new_fov: f32) -> Self {
        let new_fov = new_fov.max(1e-3);
        let old_half = (self.fov * 0.5).tan().max(1e-6);
        let new_half = (new_fov * 0.5).tan().max(1e-6);
        self.radius = (self.radius * old_half / new_half).max(1e-4);
        self.fov = new_fov;
        self
    }

    pub fn unwrap_azimuth_toward(self, other: Self) -> Self {
        let mut a = other;
        let mut d = a.azimuth - self.azimuth;
        while d > std::f32::consts::PI {
            a.azimuth -= std::f32::consts::TAU;
            d = a.azimuth - self.azimuth;
        }
        while d < -std::f32::consts::PI {
            a.azimuth += std::f32::consts::TAU;
            d = a.azimuth - self.azimuth;
        }
        a
    }
}

impl Animatable for CameraPose {
    fn lerp(self, other: Self, t: f32) -> Self {
        let other = self.unwrap_azimuth_toward(other);
        Self {
            target: self.target.lerp(other.target, t),
            azimuth: self.azimuth + (other.azimuth - self.azimuth) * t,
            elevation: self.elevation + (other.elevation - self.elevation) * t,
            radius: (self.radius + (other.radius - self.radius) * t).max(1e-4),
            roll: self.roll + (other.roll - self.roll) * t,
            fov: (self.fov + (other.fov - self.fov) * t).max(1e-3),
            focus_distance: self.focus_distance + (other.focus_distance - self.focus_distance) * t,
            aperture: self.aperture + (other.aperture - self.aperture) * t,
            f_stop: self.f_stop + (other.f_stop - self.f_stop) * t,
            aperture_blades: if t < 0.5 {
                self.aperture_blades
            } else {
                other.aperture_blades
            },
            anamorphic_ratio: self.anamorphic_ratio
                + (other.anamorphic_ratio - self.anamorphic_ratio) * t,
            shift_x: self.shift_x + (other.shift_x - self.shift_x) * t,
            shift_y: self.shift_y + (other.shift_y - self.shift_y) * t,
            sensor_fit: if t < 0.5 {
                self.sensor_fit
            } else {
                other.sensor_fit
            },
            projection: if t < 0.5 {
                self.projection
            } else {
                other.projection
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_camera() {
        let mut cam = PerspectiveCamera::new(60.0, 16.0 / 9.0, 0.1, 100.0);
        cam.position = Vector3::new(3.0, 2.0, 4.0);
        cam.look_at(Vector3::new(0.0, 0.5, 0.0));
        cam.shift_x = 0.1;
        let pose = CameraPose::from_camera(&cam);
        let mut out = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        pose.apply_to(&mut out);
        assert!((out.position - cam.position).length() < 1e-3);
        assert!((out.shift_x - 0.1).abs() < 1e-5);
    }

    #[test]
    fn lerp_takes_short_azimuth_arc() {
        let a = CameraPose::new(Vector3::ZERO, 3.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, -3.0, 1.2, 5.0);
        let m = a.lerp(b, 0.5);
        assert!(m.azimuth.abs() > 2.5, "took the long way: {}", m.azimuth);
    }

    #[test]
    fn vertigo_keeps_framed_height() {
        let p = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 10.0);
        let half = (p.fov * 0.5).tan() * p.radius;
        let q = p.with_vertigo_fov(30.0_f32.to_radians());
        let half2 = (q.fov * 0.5).tan() * q.radius;
        assert!((half - half2).abs() < 1e-3);
    }
}
