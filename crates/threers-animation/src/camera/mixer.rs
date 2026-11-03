//! Camera mixer — keyframed clips independent of the scene-graph AnimationMixer.

use super::curve::{CameraCurves, CurveKey, FCurve};
use super::pose::CameraPose;
use crate::tween::Repeat;

/// A reusable camera clip: F-curves over an absolute time range.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraClip {
    pub name: String,
    pub duration: f32,
    pub curves: CameraCurves,
}

impl CameraClip {
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration: duration.max(0.0),
            curves: CameraCurves::default(),
        }
    }

    /// Convenience: two-pose fly encoded as linear keys on every channel.
    pub fn from_poses(name: impl Into<String>, from: CameraPose, to: CameraPose, duration: f32) -> Self {
        let duration = duration.max(0.0);
        let to = from.unwrap_azimuth_toward(to);
        let mut clip = Self::new(name, duration);
        fn pair(c: &mut FCurve, t0: f32, v0: f32, t1: f32, v1: f32) {
            c.insert(CurveKey::linear(t0, v0));
            c.insert(CurveKey::linear(t1, v1));
        }
        pair(&mut clip.curves.target_x, 0.0, from.target.x, duration, to.target.x);
        pair(&mut clip.curves.target_y, 0.0, from.target.y, duration, to.target.y);
        pair(&mut clip.curves.target_z, 0.0, from.target.z, duration, to.target.z);
        pair(&mut clip.curves.azimuth, 0.0, from.azimuth, duration, to.azimuth);
        pair(
            &mut clip.curves.elevation,
            0.0,
            from.elevation,
            duration,
            to.elevation,
        );
        pair(&mut clip.curves.radius, 0.0, from.radius, duration, to.radius);
        pair(&mut clip.curves.roll, 0.0, from.roll, duration, to.roll);
        pair(&mut clip.curves.fov, 0.0, from.fov, duration, to.fov);
        pair(
            &mut clip.curves.focus_distance,
            0.0,
            from.focus_distance,
            duration,
            to.focus_distance,
        );
        pair(&mut clip.curves.aperture, 0.0, from.aperture, duration, to.aperture);
        clip
    }

    pub fn sample(&self, t: f32, template: CameraPose) -> CameraPose {
        self.curves.evaluate(t.clamp(0.0, self.duration.max(0.0)), template)
    }
}

/// Playing instance of a [`CameraClip`].
#[derive(Debug, Clone)]
pub struct CameraAction {
    pub clip: CameraClip,
    pub time: f32,
    pub weight: f32,
    pub speed: f32,
    pub repeat: Repeat,
    pub enabled: bool,
    cycles: u32,
}

impl CameraAction {
    pub fn new(clip: CameraClip) -> Self {
        Self {
            clip,
            time: 0.0,
            weight: 1.0,
            speed: 1.0,
            repeat: Repeat::Once,
            enabled: true,
            cycles: 0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.enabled || !dt.is_finite() {
            return;
        }
        self.time += dt * self.speed;
        let dur = self.clip.duration.max(1e-8);
        match self.repeat {
            Repeat::Once => {
                if self.time > dur {
                    self.time = dur;
                    self.enabled = false;
                }
            }
            Repeat::Times(n) => {
                while self.time > dur && self.cycles <= n {
                    self.time -= dur;
                    self.cycles += 1;
                }
                if self.cycles > n {
                    self.time = dur;
                    self.enabled = false;
                }
            }
            Repeat::Forever => {
                self.time = self.time.rem_euclid(dur);
            }
        }
    }

    pub fn sample(&self, template: CameraPose) -> CameraPose {
        self.clip.sample(self.time, template)
    }
}

/// Blend one or more camera actions by weight (normalised).
#[derive(Debug, Clone, Default)]
pub struct CameraMixer {
    actions: Vec<CameraAction>,
    pub template: CameraPose,
}

impl CameraMixer {
    pub fn new(template: CameraPose) -> Self {
        Self {
            actions: Vec::new(),
            template,
        }
    }

    pub fn play(&mut self, clip: CameraClip) -> usize {
        self.actions.push(CameraAction::new(clip));
        self.actions.len() - 1
    }

    pub fn action_mut(&mut self, index: usize) -> Option<&mut CameraAction> {
        self.actions.get_mut(index)
    }

    pub fn update(&mut self, dt: f32) {
        for action in &mut self.actions {
            action.update(dt);
        }
    }

    /// Weighted blend of enabled actions. Falls back to `template` when none play.
    pub fn pose(&self) -> CameraPose {
        use crate::animatable::Animatable;
        let mut acc_w = 0.0;
        let mut result = self.template;
        let mut first = true;
        for action in &self.actions {
            if !action.enabled || action.weight <= 0.0 {
                continue;
            }
            let sample = action.sample(self.template);
            if first {
                result = sample;
                acc_w = action.weight;
                first = false;
            } else {
                let w = action.weight;
                let t = w / (acc_w + w).max(1e-8);
                result = result.unwrap_azimuth_toward(sample);
                result = result.lerp(sample, t);
                acc_w += w;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn clip_from_poses_hits_end() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::new(1.0, 0.0, 0.0), 1.0, 1.0, 2.0);
        let clip = CameraClip::from_poses("fly", a, b, 1.0);
        let end = clip.sample(1.0, a);
        assert!((end.radius - 2.0).abs() < 1e-3);
        assert!((end.target.x - 1.0).abs() < 1e-3);
    }

    #[test]
    fn mixer_plays_clip() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 1.0);
        let mut mixer = CameraMixer::new(a);
        mixer.play(CameraClip::from_poses("x", a, b, 1.0));
        for _ in 0..60 {
            mixer.update(1.0 / 60.0);
        }
        assert!((mixer.pose().radius - 1.0).abs() < 1e-2);
    }
}
