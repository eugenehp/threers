//! First-class camera animator: fly / orbit / look / FOV / focus / paths.

use crate::animatable::Animatable;
use crate::easing::Easing;
use threers::cameras::PerspectiveCamera;
use threers::math::Vector3;

use super::follow::CameraFollow;
use super::path::CameraPath;
use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use super::shake::CameraShake;

#[derive(Debug, Clone)]
enum Motion {
    Idle,
    Transition {
        from: CameraPose,
        to: CameraPose,
        duration: f32,
        elapsed: f32,
        ramp: SpeedRamp,
    },
    Path {
        path: CameraPath,
        duration: f32,
        elapsed: f32,
        template: CameraPose,
    },
}

/// Drives a [`CameraPose`] over time with cinematic moves, paths, shake, and follow.
///
/// ```
/// use threers_animation::prelude::*;
/// use threers::math::Vector3;
///
/// let start = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
/// let mut cam = CameraAnimator::new(start);
/// cam.fly_to(
///     CameraPose::new(Vector3::new(1.0, 0.0, 0.0), 1.0, 1.0, 3.0),
///     2.0,
///     Easing::CubicInOut,
/// );
/// cam.update(1.0);
/// assert!(!cam.is_idle());
/// ```
#[derive(Debug, Clone)]
pub struct CameraAnimator {
    pose: CameraPose,
    motion: Motion,
    pub shake: CameraShake,
    follow: Option<CameraFollow>,
    /// When true, spring-follow output softens the scripted pose each frame.
    pub follow_enabled: bool,
}

impl CameraAnimator {
    pub fn new(pose: CameraPose) -> Self {
        Self {
            pose,
            motion: Motion::Idle,
            shake: CameraShake::default(),
            follow: None,
            follow_enabled: false,
        }
    }

    pub fn from_camera(camera: &PerspectiveCamera) -> Self {
        Self::new(CameraPose::from_camera(camera))
    }

    pub fn pose(&self) -> CameraPose {
        self.pose
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.motion, Motion::Idle)
            && (!self.follow_enabled || self.follow.as_ref().is_some_and(|f| f.is_settled()))
    }

    /// Snap immediately and cancel motion.
    pub fn set_pose(&mut self, pose: CameraPose) {
        self.pose = pose;
        self.motion = Motion::Idle;
        if let Some(f) = &mut self.follow {
            f.set_goal(pose);
        }
    }

    /// Fly between two full poses with a named ease.
    pub fn fly_to(&mut self, to: CameraPose, duration: f32, easing: Easing) {
        self.fly_to_ramped(to, duration, SpeedRamp::Ease(easing));
    }

    /// Fly with an explicit speed ramp (ease / cruise / smootherstep / …).
    pub fn fly_to_ramped(&mut self, to: CameraPose, duration: f32, ramp: SpeedRamp) {
        let to = self.pose.unwrap_azimuth_toward(to);
        self.motion = Motion::Transition {
            from: self.pose,
            to,
            duration: duration.max(0.0),
            elapsed: 0.0,
            ramp,
        };
    }

    /// Orbit in spherical space by deltas.
    pub fn orbit_by(
        &mut self,
        d_azimuth: f32,
        d_elevation: f32,
        d_radius: f32,
        duration: f32,
        easing: Easing,
    ) {
        let mut to = self.pose;
        to.azimuth += d_azimuth;
        to.elevation = (to.elevation + d_elevation).clamp(1e-3, std::f32::consts::PI - 1e-3);
        to.radius = (to.radius + d_radius).max(1e-4);
        self.fly_to(to, duration, easing);
    }

    /// Orbit to absolute spherical coordinates, keeping the current target.
    pub fn orbit_to(
        &mut self,
        azimuth: f32,
        elevation: f32,
        radius: f32,
        duration: f32,
        easing: Easing,
    ) {
        let mut to = self.pose;
        to.azimuth = azimuth;
        to.elevation = elevation.clamp(1e-3, std::f32::consts::PI - 1e-3);
        to.radius = radius.max(1e-4);
        self.fly_to(to, duration, easing);
    }

    /// Retarget look-at. When `hold_eye`, the eye stays put and spherical is rebuilt.
    pub fn look_at(&mut self, target: Vector3, duration: f32, easing: Easing, hold_eye: bool) {
        let mut to = self.pose;
        if hold_eye {
            let eye = self.pose.eye();
            to = CameraPose::from_look_at(eye, target, self.pose.fov);
            to.roll = self.pose.roll;
            to.focus_distance = self.pose.focus_distance;
            to.aperture = self.pose.aperture;
        } else {
            to.target = target;
        }
        self.fly_to(to, duration, easing);
    }

    /// Animate focus distance + aperture (DoF pull).
    pub fn focus_pull(&mut self, focus_distance: f32, aperture: f32, duration: f32, easing: Easing) {
        let mut to = self.pose;
        to.focus_distance = focus_distance.max(0.0);
        to.aperture = aperture.max(0.0);
        self.fly_to(to, duration, easing);
    }

    /// Animate FOV. `vertigo` keeps subject scale via a compensating dolly.
    pub fn fov_to(&mut self, fov: f32, duration: f32, easing: Easing, vertigo: bool) {
        let mut to = self.pose;
        if vertigo {
            to = to.with_vertigo_fov(fov);
        } else {
            to.fov = fov.max(1e-3);
        }
        self.fly_to(to, duration, easing);
    }

    /// Play a world-space path over `duration` seconds.
    pub fn play_path(&mut self, path: CameraPath, duration: f32) {
        self.motion = Motion::Path {
            path,
            duration: duration.max(1e-4),
            elapsed: 0.0,
            template: self.pose,
        };
    }

    /// Soften scripted motion with a spring at `frequency` Hz.
    pub fn enable_follow(&mut self, frequency: f32) {
        self.follow = Some(CameraFollow::new(self.pose, frequency.max(0.1)));
        self.follow_enabled = true;
    }

    pub fn disable_follow(&mut self) {
        self.follow_enabled = false;
    }

    /// Advance simulation; returns the pose to apply this frame (includes shake).
    pub fn update(&mut self, dt: f32) -> CameraPose {
        if dt.is_finite() && dt > 0.0 {
            self.step_motion(dt);
            if self.shake.is_active() {
                self.shake.update(dt);
            }
            if self.follow_enabled {
                if let Some(f) = &mut self.follow {
                    f.set_goal(self.pose);
                    self.pose = f.update(dt);
                }
            }
        }
        // Identity when shake is inactive — bit-stable, no look-at rebuild.
        self.shake.apply(self.pose)
    }

    fn step_motion(&mut self, dt: f32) {
        let mut done = false;
        match &mut self.motion {
            Motion::Idle => {}
            Motion::Transition {
                from,
                to,
                duration,
                elapsed,
                ramp,
            } => {
                *elapsed += dt;
                if *duration <= 0.0 {
                    self.pose = *to;
                    done = true;
                } else {
                    *elapsed = (*elapsed).min(*duration);
                    let t = (*elapsed / *duration).clamp(0.0, 1.0);
                    let u = ramp.apply(t);
                    // Snap endpoints exactly so ease curves cannot leave a
                    // residual that reads as idle jitter.
                    self.pose = if u <= 0.0 {
                        *from
                    } else if u >= 1.0 {
                        *to
                    } else {
                        from.lerp(*to, u)
                    };
                    done = *elapsed >= *duration - 1e-6;
                }
            }
            Motion::Path {
                path,
                duration,
                elapsed,
                template,
            } => {
                *elapsed += dt;
                *elapsed = (*elapsed).min(*duration);
                let t = (*elapsed / *duration).clamp(0.0, 1.0);
                self.pose = path.sample_pose(t, *template);
                done = *elapsed >= *duration - 1e-6;
            }
        }
        if done {
            self.motion = Motion::Idle;
        }
    }

    /// Write the latest posed frame onto a perspective camera.
    pub fn apply_to(&mut self, camera: &mut PerspectiveCamera) {
        self.shake.apply(self.pose).apply_to(camera);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_to_reaches_destination() {
        let start = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let goal = CameraPose::new(Vector3::new(1.0, 0.0, 0.0), 1.0, 1.0, 3.0);
        let mut anim = CameraAnimator::new(start);
        anim.fly_to(goal, 1.0, Easing::Linear);
        for _ in 0..60 {
            anim.update(1.0 / 60.0);
        }
        assert!(anim.is_idle());
        assert!((anim.pose().target - goal.target).length() < 1e-3);
        assert!((anim.pose().radius - goal.radius).abs() < 1e-3);
    }

    #[test]
    fn orbit_by_changes_azimuth() {
        let mut anim = CameraAnimator::new(CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0));
        anim.orbit_by(1.0, 0.0, 0.0, 0.5, Easing::Linear);
        for _ in 0..30 {
            anim.update(1.0 / 60.0);
        }
        assert!((anim.pose().azimuth - 1.0).abs() < 1e-3);
    }
}
