use super::PointerEvent;
use crate::cameras::PerspectiveCamera;
use crate::math::{Spherical, Vector3};

/// Spherical orbit camera controls — rotate, pan, zoom around `target`.
/// Mirrors three.js's `OrbitControls`.
#[derive(Debug, Clone, Copy)]
pub struct OrbitControls {
    pub target: Vector3,
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_polar_angle: f32,
    pub max_polar_angle: f32,
    /// Azimuth limits (radians). When `min > max` wrapping is allowed (default).
    pub min_azimuth_angle: f32,
    pub max_azimuth_angle: f32,
    pub rotate_speed: f32,
    pub zoom_speed: f32,
    pub pan_speed: f32,
    /// Inertia factor in `0..1`. `0` = no damping (input applied immediately).
    /// Typical three.js value is `0.05` with `enable_damping = true`.
    pub damping: f32,
    /// When true, residual spherical / pan deltas decay each frame. Call
    /// [`update`](Self::update) / [`update_dt`](Self::update_dt) every frame.
    pub enable_damping: bool,
    /// Continuous azimuth spin when idle.
    pub auto_rotate: bool,
    /// Radians per second when `auto_rotate` is on. three.js default ≈ 2π / 30.
    pub auto_rotate_speed: f32,
    spherical: Spherical,
    /// Residual rotate deltas (theta, phi) for damping.
    spherical_delta: (f32, f32),
    pan_offset: Vector3,
    zoom_scale: f32,
}

impl OrbitControls {
    pub fn new(camera: &PerspectiveCamera) -> Self {
        let offset = camera.position - camera.target;
        let spherical = Spherical::from_vector3(offset);
        Self {
            target: camera.target,
            min_distance: 0.1,
            max_distance: 1000.0,
            min_polar_angle: 0.0,
            max_polar_angle: std::f32::consts::PI,
            min_azimuth_angle: f32::NEG_INFINITY,
            max_azimuth_angle: f32::INFINITY,
            rotate_speed: 1.0,
            zoom_speed: 1.0,
            pan_speed: 1.0,
            damping: 0.05,
            enable_damping: false,
            auto_rotate: false,
            auto_rotate_speed: std::f32::consts::TAU / 30.0,
            spherical,
            spherical_delta: (0.0, 0.0),
            pan_offset: Vector3::ZERO,
            zoom_scale: 1.0,
        }
    }

    /// Rebuild spherical state from the camera's current pose (parity orbit sync).
    pub fn reseed_from_camera(&mut self, camera: &PerspectiveCamera) {
        self.target = camera.target;
        self.spherical = Spherical::from_vector3(camera.position - camera.target);
        self.pan_offset = Vector3::ZERO;
        self.spherical_delta = (0.0, 0.0);
        self.zoom_scale = 1.0;
    }

    /// Apply one frame of input at a presumed 60 Hz step. Prefer
    /// [`update_dt`](Self::update_dt) when you have a real delta.
    pub fn update(
        &mut self,
        ev: PointerEvent,
        camera: &mut PerspectiveCamera,
        viewport: (f32, f32),
    ) {
        self.update_dt(ev, camera, viewport, 1.0 / 60.0);
    }

    /// Apply input + damping / auto-rotate for `dt` seconds.
    pub fn update_dt(
        &mut self,
        ev: PointerEvent,
        camera: &mut PerspectiveCamera,
        viewport: (f32, f32),
        dt: f32,
    ) {
        let (w, h) = viewport;
        let half_fov = camera.fov * 0.5;
        let dist = self.spherical.radius;
        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            1.0 / 60.0
        };

        if ev.rotating {
            let dtheta = -ev.dx / h * std::f32::consts::PI * 2.0 * self.rotate_speed;
            let dphi = -ev.dy / h * std::f32::consts::PI * 2.0 * self.rotate_speed;
            self.spherical_delta.0 += dtheta;
            self.spherical_delta.1 += dphi;
        }
        if ev.panning {
            let world_per_pixel = 2.0 * dist * half_fov.tan() / h;
            let offset = self.spherical.to_vector3();
            let forward = (-offset).normalize();
            let world_up = Vector3::UP;
            let right = forward.cross(world_up).normalize();
            let up = right.cross(forward).normalize();
            self.pan_offset = self.pan_offset
                + right * (-ev.dx * world_per_pixel * self.pan_speed)
                + up * (ev.dy * world_per_pixel * self.pan_speed);
        }
        if ev.wheel != 0.0 {
            let factor = (1.0 - ev.wheel * 0.001 * self.zoom_speed).clamp(0.1, 10.0);
            if self.enable_damping && self.damping > 0.0 {
                self.zoom_scale *= factor;
            } else {
                self.spherical.radius =
                    (self.spherical.radius * factor).clamp(self.min_distance, self.max_distance);
            }
        }

        if self.auto_rotate && !ev.rotating {
            self.spherical_delta.0 += self.auto_rotate_speed * dt;
        }

        let has_residual = self.spherical_delta.0.abs() > 1e-8
            || self.spherical_delta.1.abs() > 1e-8
            || self.pan_offset.length_sq() > 1e-12
            || (self.zoom_scale - 1.0).abs() > 1e-8;
        let has_input = ev.rotating || ev.panning || ev.wheel != 0.0;
        let needs_tick = has_input || has_residual || self.auto_rotate;

        // Idle frames must not rewrite the camera — external sync may have set
        // the wasm pose without updating our spherical state yet.
        if !needs_tick {
            return;
        }

        if self.enable_damping && self.damping > 0.0 {
            let damp = (1.0 - self.damping.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            self.spherical.theta += self.spherical_delta.0;
            self.spherical.phi += self.spherical_delta.1;
            self.spherical_delta.0 *= damp;
            self.spherical_delta.1 *= damp;
            self.target = self.target + self.pan_offset;
            self.pan_offset = self.pan_offset * damp;
            if (self.zoom_scale - 1.0).abs() > 1e-8 {
                self.spherical.radius = (self.spherical.radius * self.zoom_scale)
                    .clamp(self.min_distance, self.max_distance);
                // Ease zoom_scale back toward 1.
                self.zoom_scale = 1.0 + (self.zoom_scale - 1.0) * damp;
            }
            // Snap residuals to rest so damping does not hiss forever as
            // sub-pixel orbit jitter.
            const REST: f32 = 1e-6;
            if self.spherical_delta.0.abs() < REST {
                self.spherical_delta.0 = 0.0;
            }
            if self.spherical_delta.1.abs() < REST {
                self.spherical_delta.1 = 0.0;
            }
            if self.pan_offset.length_sq() < REST * REST {
                self.pan_offset = Vector3::ZERO;
            }
            if (self.zoom_scale - 1.0).abs() < REST {
                self.zoom_scale = 1.0;
            }
        } else {
            self.spherical.theta += self.spherical_delta.0;
            self.spherical.phi += self.spherical_delta.1;
            self.spherical_delta = (0.0, 0.0);
            self.target = self.target + self.pan_offset;
            self.pan_offset = Vector3::ZERO;
            if (self.zoom_scale - 1.0).abs() > 1e-8 {
                self.spherical.radius = (self.spherical.radius * self.zoom_scale)
                    .clamp(self.min_distance, self.max_distance);
                self.zoom_scale = 1.0;
            }
        }

        const EPS: f32 = 1e-4;
        self.spherical.phi = self
            .spherical
            .phi
            .clamp(self.min_polar_angle, self.max_polar_angle)
            .clamp(EPS, std::f32::consts::PI - EPS);
        if self.min_azimuth_angle <= self.max_azimuth_angle {
            self.spherical.theta = self
                .spherical
                .theta
                .clamp(self.min_azimuth_angle, self.max_azimuth_angle);
        }
        self.spherical.radius = self
            .spherical
            .radius
            .clamp(self.min_distance, self.max_distance);

        let offset = self.spherical.to_vector3();
        camera.target = self.target;
        camera.position = self.target + offset;
        let _ = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::PointerEvent;

    #[test]
    fn idle_update_preserves_external_camera_pose() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orbit = OrbitControls::new(&cam);
        // External sync moves the camera without reseeding orbit.
        cam.position = Vector3::new(-3.8, 2.9, 1.25);
        cam.target = Vector3::ZERO;
        let idle = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        orbit.update(idle, &mut cam, (800.0, 600.0));
        assert!((cam.position.x + 3.8).abs() < 1e-4);
        assert!((cam.position.y - 2.9).abs() < 1e-4);
        assert!((cam.position.z - 1.25).abs() < 1e-4);
    }

    #[test]
    fn reseed_from_camera_refreshes_spherical() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orbit = OrbitControls::new(&cam);
        cam.position = Vector3::new(-3.8, 2.9, 1.25);
        orbit.reseed_from_camera(&cam);
        let idle = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        orbit.update(idle, &mut cam, (800.0, 600.0));
        assert!((cam.position.x + 3.8).abs() < 1e-4);
        let moved = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        orbit.update(moved, &mut cam, (800.0, 600.0));
        assert!((cam.position.x + 3.8).abs() < 1e-4);
    }

    #[test]
    fn damping_keeps_residual_motion() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orbit = OrbitControls::new(&cam);
        orbit.enable_damping = true;
        orbit.damping = 0.1;
        let flick = PointerEvent {
            dx: 40.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: true,
            panning: false,
        };
        orbit.update_dt(flick, &mut cam, (800.0, 600.0), 1.0 / 60.0);
        let after_flick = cam.position;
        let idle = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        orbit.update_dt(idle, &mut cam, (800.0, 600.0), 1.0 / 60.0);
        assert!((cam.position - after_flick).length() > 1e-4);
    }

    #[test]
    fn damping_settles_without_endless_residual() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orbit = OrbitControls::new(&cam);
        orbit.enable_damping = true;
        orbit.damping = 0.15;
        let flick = PointerEvent {
            dx: 80.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: true,
            panning: false,
        };
        orbit.update_dt(flick, &mut cam, (800.0, 600.0), 1.0 / 60.0);
        let idle = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        for _ in 0..300 {
            orbit.update_dt(idle, &mut cam, (800.0, 600.0), 1.0 / 60.0);
        }
        let settled = cam.position;
        orbit.update_dt(idle, &mut cam, (800.0, 600.0), 1.0 / 60.0);
        assert_eq!(
            cam.position, settled,
            "idle update after settle must not rewrite the camera"
        );
        assert_eq!(orbit.spherical_delta, (0.0, 0.0));
    }

    #[test]
    fn auto_rotate_spins_when_idle() {
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orbit = OrbitControls::new(&cam);
        orbit.auto_rotate = true;
        orbit.auto_rotate_speed = 1.0;
        let start_theta = orbit.spherical.theta;
        let idle = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
        };
        orbit.update_dt(idle, &mut cam, (800.0, 600.0), 0.5);
        assert!((orbit.spherical.theta - start_theta).abs() > 0.1);
    }
}
