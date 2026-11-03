//! Spring-lag follow: chase a moving subject without a fixed duration.

use crate::spring::Spring;
use threers::math::Vector3;

use super::pose::CameraPose;

/// Lazily follow a target pose with independent springs on eye, look-at, and FOV.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraFollow {
    eye: Spring<Vector3>,
    target: Spring<Vector3>,
    fov: Spring<f32>,
    focus: Spring<f32>,
    pub roll: f32,
    pub aperture: f32,
}

impl CameraFollow {
    /// `frequency` in Hz — 1..=2 feels cinematic; 4..=6 feels snappy.
    pub fn new(pose: CameraPose, frequency: f32) -> Self {
        Self {
            eye: Spring::critically_damped(pose.eye(), frequency),
            target: Spring::critically_damped(pose.target, frequency * 1.15),
            fov: Spring::critically_damped(pose.fov, frequency),
            focus: Spring::critically_damped(pose.focus_distance, frequency),
            roll: pose.roll,
            aperture: pose.aperture,
        }
    }

    pub fn set_goal(&mut self, pose: CameraPose) {
        self.eye.set_target(pose.eye());
        self.target.set_target(pose.target);
        self.fov.set_target(pose.fov);
        self.focus.set_target(pose.focus_distance);
        self.roll = pose.roll;
        self.aperture = pose.aperture;
    }

    /// Chase a world-space subject while keeping current framing radius / angles.
    pub fn set_subject(&mut self, subject: Vector3, pose_template: CameraPose) {
        let mut goal = pose_template;
        goal.target = subject;
        self.set_goal(goal);
    }

    pub fn nudge(&mut self, eye_velocity: Vector3) {
        self.eye.nudge(eye_velocity);
    }

    pub fn update(&mut self, dt: f32) -> CameraPose {
        self.eye.update(dt);
        self.target.update(dt);
        self.fov.update(dt);
        self.focus.update(dt);
        // Snap when settled so spring noise cannot hiss as sub-pixel jitter.
        if self.eye.is_settled() {
            self.eye.reset_to(self.eye.target());
        }
        if self.target.is_settled() {
            self.target.reset_to(self.target.target());
        }
        if self.fov.is_settled() {
            self.fov.reset_to(self.fov.target());
        }
        if self.focus.is_settled() {
            self.focus.reset_to(self.focus.target());
        }
        self.pose()
    }

    pub fn is_settled(&self) -> bool {
        self.eye.is_settled() && self.target.is_settled() && self.fov.is_settled()
    }

    pub fn pose(&self) -> CameraPose {
        let mut pose = CameraPose::from_look_at(self.eye.value(), self.target.value(), self.fov.value());
        pose.roll = self.roll;
        pose.focus_distance = self.focus.value();
        pose.aperture = self.aperture;
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follow_reaches_goal() {
        let start = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let mut follow = CameraFollow::new(start, 3.0);
        let goal = CameraPose::new(Vector3::new(2.0, 0.0, 0.0), 1.0, 1.0, 3.0);
        follow.set_goal(goal);
        for _ in 0..300 {
            follow.update(1.0 / 60.0);
        }
        assert!(follow.is_settled());
        let p = follow.pose();
        assert!((p.target - goal.target).length() < 0.05);
    }
}
