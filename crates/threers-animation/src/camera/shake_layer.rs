//! Layered shake with keyed amplitude envelopes.

use super::curve::FCurve;
use super::pose::CameraPose;
use super::shake::CameraShake;
use threers::math::Vector3;

/// One shake layer: procedural shake scaled by a keyed envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct ShakeLayer {
    pub name: String,
    pub shake: CameraShake,
    /// Amplitude multiplier over time (`1` = full). Empty curve → constant 1.
    pub envelope: FCurve,
    pub enabled: bool,
}

impl ShakeLayer {
    pub fn new(name: impl Into<String>, shake: CameraShake) -> Self {
        Self {
            name: name.into(),
            shake,
            envelope: FCurve::new(),
            enabled: true,
        }
    }

    pub fn with_envelope(mut self, envelope: FCurve) -> Self {
        self.envelope = envelope;
        self
    }
}

/// Stack of shake layers evaluated at a shared clock.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShakeStack {
    layers: Vec<ShakeLayer>,
    time: f32,
}

impl ShakeStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, layer: ShakeLayer) -> usize {
        self.layers.push(layer);
        self.layers.len() - 1
    }

    pub fn layers_mut(&mut self) -> &mut [ShakeLayer] {
        &mut self.layers
    }

    pub fn seek(&mut self, time: f32) {
        if time.is_finite() {
            self.time = time.max(0.0);
        }
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn update(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.time += dt;
        for layer in &mut self.layers {
            if layer.enabled {
                layer.shake.update(dt);
            }
        }
    }

    /// Impulse on a named layer (no-op if missing).
    pub fn impulse(&mut self, layer: usize, dir: Vector3, strength: f32) {
        if let Some(l) = self.layers.get_mut(layer) {
            l.shake.impulse(dir, strength);
        }
    }

    pub fn apply(&self, pose: CameraPose) -> CameraPose {
        let mut out = pose;
        for layer in &self.layers {
            if !layer.enabled {
                continue;
            }
            let env = if layer.envelope.is_empty() {
                1.0
            } else {
                layer.envelope.evaluate(self.time).max(0.0)
            };
            if env <= 1e-8 && layer.shake.impulse_strength() <= 1e-8 {
                continue;
            }
            // Temporarily scale amplitudes via a scaled copy.
            let mut shake = layer.shake;
            shake.amplitude *= env;
            shake.roll_amplitude *= env;
            out = shake.apply(out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::curve::CurveKey;

    #[test]
    fn envelope_gates_amplitude() {
        let mut shake = CameraShake::new(1.0, 8.0);
        shake.enabled = true;
        let env = FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 1.0),
        ]);
        let mut stack = ShakeStack::new();
        stack.push(ShakeLayer::new("hand", shake).with_envelope(env));
        let pose = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        stack.seek(0.0);
        assert_eq!(stack.apply(pose), pose);
        stack.seek(1.0);
        let kicked = stack.apply(pose);
        assert!((kicked.eye() - pose.eye()).length() > 1e-3);
    }
}
