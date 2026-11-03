//! Handheld shake and impulse kick.

use threers::math::Vector3;

use super::pose::CameraPose;

/// Procedural camera shake (position + roll) with optional decaying impulses.
///
/// Inactive by default (`amplitude = 0`). When inactive, [`apply`](Self::apply)
/// is a pure identity — it does **not** rebuild the pose through look-at, which
/// would introduce per-frame spherical quantization jitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraShake {
    /// Positional amplitude in world units.
    pub amplitude: f32,
    /// Roll amplitude in radians.
    pub roll_amplitude: f32,
    /// Dominant frequency in Hz.
    pub frequency: f32,
    /// Secondary frequency multiplier for roughness.
    pub roughness: f32,
    /// How fast an impulse decays (1/e time constant inverse).
    pub impulse_decay: f32,
    /// Master enable. When false, [`apply`](Self::apply) is always identity.
    pub enabled: bool,
    time: f32,
    impulse: f32,
    impulse_dir: Vector3,
}

impl Default for CameraShake {
    fn default() -> Self {
        Self {
            amplitude: 0.0,
            roll_amplitude: 0.0,
            frequency: 8.0,
            roughness: 2.3,
            impulse_decay: 4.0,
            enabled: true,
            time: 0.0,
            impulse: 0.0,
            impulse_dir: Vector3::UP,
        }
    }
}

impl CameraShake {
    pub fn new(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude: amplitude.max(0.0),
            frequency: frequency.max(0.0),
            ..Self::default()
        }
    }

    /// Whether any offset would be applied this frame.
    pub fn is_active(&self) -> bool {
        self.enabled
            && (self.amplitude > 1e-8 || self.roll_amplitude > 1e-8 || self.impulse > 1e-8)
    }

    /// Kick the camera — impact reaction. `strength` scales amplitude.
    pub fn impulse(&mut self, dir: Vector3, strength: f32) {
        let n = dir.normalize();
        self.impulse_dir = if n.length_sq() < 1e-10 {
            Vector3::UP
        } else {
            n
        };
        self.impulse = strength.max(0.0);
    }

    /// Current impulse magnitude (for envelope gating).
    pub fn impulse_strength(&self) -> f32 {
        self.impulse
    }

    pub fn update(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 || !self.enabled {
            return;
        }
        if self.is_active() {
            self.time += dt;
        }
        if self.impulse > 0.0 {
            self.impulse *= (-self.impulse_decay * dt).exp();
            if self.impulse < 1e-5 {
                self.impulse = 0.0;
            }
        }
    }

    /// Offset applied on top of a clean pose.
    pub fn offset(&self) -> (Vector3, f32) {
        if !self.is_active() {
            return (Vector3::ZERO, 0.0);
        }
        let w1 = self.frequency * std::f32::consts::TAU;
        let w2 = w1 * self.roughness.max(1.0);
        let t = self.time;
        let noise = Vector3::new(
            (t * w1).sin() * 0.6 + (t * w2 + 1.7).sin() * 0.4,
            (t * w1 * 1.1 + 0.4).cos() * 0.6 + (t * w2 * 0.9 + 2.1).sin() * 0.4,
            (t * w1 * 0.85 + 1.1).sin() * 0.6 + (t * w2 * 1.2 + 0.6).cos() * 0.4,
        );
        let pos = noise * self.amplitude + self.impulse_dir * self.impulse;
        let roll = if self.roll_amplitude > 1e-8 {
            (t * w1 * 0.7 + 0.3).sin() * self.roll_amplitude
                + (t * w2 + 1.0).sin() * self.roll_amplitude * 0.35
        } else {
            0.0
        };
        (pos, roll)
    }

    /// Apply shake. Inactive → bit-identical identity (no look-at rebuild).
    pub fn apply(&self, pose: CameraPose) -> CameraPose {
        let (pos, roll) = self.offset();
        if pos.length_sq() < 1e-20 && roll.abs() < 1e-10 {
            return pose;
        }
        let eye = pose.eye() + pos;
        let target = pose.target + pos * 0.25;
        let mut out = CameraPose::from_look_at(eye, target, pose.fov);
        out.roll = pose.roll + roll;
        out.focus_distance = pose.focus_distance;
        out.aperture = pose.aperture;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_amplitude_is_bit_exact_identity() {
        let shake = CameraShake::default();
        let pose = CameraPose::new(Vector3::new(0.1, 0.2, 0.3), 0.5, 1.2, 4.0);
        let out = shake.apply(pose);
        assert_eq!(out, pose);
    }

    #[test]
    fn inactive_apply_does_not_drift_over_frames() {
        let shake = CameraShake::default();
        let mut pose = CameraPose::new(Vector3::ZERO, 0.7, 1.1, 5.0);
        let start = pose;
        for _ in 0..120 {
            pose = shake.apply(pose);
        }
        assert_eq!(pose, start);
    }

    #[test]
    fn impulse_decays() {
        let mut shake = CameraShake::default();
        shake.impulse(Vector3::new(1.0, 0.0, 0.0), 2.0);
        assert!(shake.impulse > 1.0);
        for _ in 0..120 {
            shake.update(1.0 / 60.0);
        }
        assert!(shake.impulse < 0.05);
    }
}
