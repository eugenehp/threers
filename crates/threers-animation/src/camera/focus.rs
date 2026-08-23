//! Physical DoF focus tracking — keep focus distance locked to a subject.

use super::pose::CameraPose;
use crate::spring::Spring;
use threers::math::Vector3;

/// Tracks a world-space subject and writes focus distance + optical DoF params.
#[derive(Debug, Clone, PartialEq)]
pub struct FocusTracker {
    /// Subject position to keep sharp (Blender focus object).
    pub subject: Vector3,
    /// Aperture radius (`None` = leave pose alone).
    pub aperture: Option<f32>,
    /// F-stop (`None` = leave). When set, also derives a rough aperture.
    pub f_stop: Option<f32>,
    /// Diaphragm blades.
    pub blades: Option<u32>,
    /// Anamorphic bokeh ratio.
    pub anamorphic_ratio: Option<f32>,
    /// Soft follow of focus distance (Hz). `None` = hard lock each frame.
    soft: Option<Spring<f32>>,
}

impl FocusTracker {
    pub fn new(subject: Vector3) -> Self {
        Self {
            subject,
            aperture: None,
            f_stop: None,
            blades: None,
            anamorphic_ratio: None,
            soft: None,
        }
    }

    pub fn with_aperture(mut self, aperture: f32) -> Self {
        self.aperture = Some(aperture.max(0.0));
        self
    }

    pub fn with_f_stop(mut self, f_stop: f32) -> Self {
        self.f_stop = Some(f_stop.max(0.0));
        self
    }

    pub fn with_blades(mut self, blades: u32) -> Self {
        self.blades = Some(blades);
        self
    }

    pub fn with_anamorphic(mut self, ratio: f32) -> Self {
        self.anamorphic_ratio = Some(ratio.max(1e-3));
        self
    }

    pub fn with_soft_follow(mut self, frequency: f32, initial_distance: f32) -> Self {
        self.soft = Some(Spring::critically_damped(
            initial_distance.max(0.0),
            frequency.max(0.1),
        ));
        self
    }

    pub fn set_subject(&mut self, subject: Vector3) {
        self.subject = subject;
    }

    pub fn desired_distance(&self, pose: &CameraPose) -> f32 {
        (pose.eye() - self.subject).length().max(0.0)
    }

    pub fn apply(&mut self, pose: &mut CameraPose, dt: f32) {
        let desired = self.desired_distance(pose);
        let distance = if let Some(spring) = &mut self.soft {
            spring.set_target(desired);
            spring.update(dt.max(0.0));
            if spring.is_settled() {
                spring.reset_to(desired);
            }
            spring.value()
        } else {
            desired
        };
        pose.focus_distance = distance;
        if let Some(a) = self.aperture {
            pose.aperture = a;
        }
        if let Some(f) = self.f_stop {
            pose.f_stop = f;
            // Rough thin-lens: aperture ≈ focal_length / f_stop; without film
            // back we keep a unit focal length scale.
            if self.aperture.is_none() && f > 1e-6 {
                pose.aperture = (1.0 / f).max(0.0);
            }
        }
        if let Some(b) = self.blades {
            pose.aperture_blades = b;
        }
        if let Some(r) = self.anamorphic_ratio {
            pose.anamorphic_ratio = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_lock_matches_eye_distance() {
        let mut pose = CameraPose::new(Vector3::ZERO, 0.0, std::f32::consts::FRAC_PI_2, 4.0);
        let mut tracker = FocusTracker::new(Vector3::ZERO)
            .with_aperture(0.05)
            .with_blades(6)
            .with_f_stop(2.8);
        tracker.apply(&mut pose, 1.0 / 60.0);
        assert!((pose.focus_distance - 4.0).abs() < 1e-3);
        assert_eq!(pose.aperture_blades, 6);
        assert!((pose.f_stop - 2.8).abs() < 1e-5);
    }
}
