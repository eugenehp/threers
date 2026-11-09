use super::{Audio, AudioListener};
use threers_math::{Matrix4, Vector3};

/// How gain falls with distance: the three models of Web Audio's `PannerNode`,
/// which three.js's `PositionalAudio` hands them to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceModel {
    Linear,
    #[default]
    Inverse,
    Exponential,
}

/// How a source sounds from a listener.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialMix {
    /// Linear gain: volume, master volume, distance and cone together.
    pub gain: f32,
    /// -1 is fully left of the listener, 1 fully right.
    pub pan: f32,
}

/// Spatial audio source. Inherits all `Audio` parameters plus a distance model.
/// Attach one with `Object3D::audio` in threers; it points along
/// the node's +Z, as in three.js.
#[derive(Debug, Clone)]
pub struct PositionalAudio {
    pub audio: Audio,
    pub distance_model: DistanceModel,
    pub ref_distance: f32,
    pub max_distance: f32,
    pub rolloff_factor: f32,
    /// Outer cone angle in radians (or `f32::INFINITY` for omni).
    pub cone_outer_angle: f32,
    pub cone_inner_angle: f32,
    pub cone_outer_gain: f32,
}

impl Default for PositionalAudio {
    fn default() -> Self {
        Self {
            audio: Audio::default(),
            distance_model: DistanceModel::default(),
            ref_distance: 1.0,
            max_distance: 10_000.0,
            rolloff_factor: 1.0,
            cone_outer_angle: f32::INFINITY,
            cone_inner_angle: f32::INFINITY,
            cone_outer_gain: 0.0,
        }
    }
}

impl PositionalAudio {
    /// Gain at `distance`, by the `PannerNode` formulas. Only `Linear` stops at
    /// `max_distance`; the other two keep falling, as they do there.
    pub fn distance_gain(&self, distance: f32) -> f32 {
        let reference = self.ref_distance;
        match self.distance_model {
            DistanceModel::Linear => {
                let d = distance.clamp(reference, self.max_distance.max(reference));
                let span = self.max_distance - reference;
                if span <= 0.0 {
                    return 1.0;
                }
                1.0 - self.rolloff_factor.min(1.0) * (d - reference) / span
            }
            DistanceModel::Inverse => {
                let d = distance.max(reference);
                let denominator = reference + self.rolloff_factor * (d - reference);
                if denominator <= 0.0 {
                    return 1.0;
                }
                reference / denominator
            }
            DistanceModel::Exponential => {
                if reference <= 0.0 {
                    return 1.0;
                }
                (distance.max(reference) / reference).powf(-self.rolloff_factor)
            }
        }
    }

    /// Gain `angle` radians off the source's axis. The cone angles are full
    /// widths, so the edge of a cone is at half of one.
    pub fn cone_gain(&self, angle: f32) -> f32 {
        let tau = std::f32::consts::TAU;
        if self.cone_inner_angle >= tau && self.cone_outer_angle >= tau {
            return 1.0;
        }
        let (inner, outer) = (self.cone_inner_angle / 2.0, self.cone_outer_angle / 2.0);
        if angle <= inner {
            1.0
        } else if angle >= outer {
            self.cone_outer_gain
        } else {
            let x = (angle - inner) / (outer - inner);
            (1.0 - x) + self.cone_outer_gain * x
        }
    }

    /// This source, placed by `world`, as `listener` hears it.
    pub fn mix(&self, world: &Matrix4, listener: &AudioListener) -> SpatialMix {
        let e = &world.elements;
        let position = Vector3::new(e[12], e[13], e[14]);
        let axis = Vector3::new(e[8], e[9], e[10]).normalize();
        let to_listener = listener.position - position;
        let distance = to_listener.length();

        // A listener on the source has no direction: dead centre, on axis.
        let (angle, pan) = if distance > 0.0 {
            let toward = to_listener * (1.0 / distance);
            let right = listener.forward.cross(listener.up).normalize();
            (
                axis.dot(toward).clamp(-1.0, 1.0).acos(),
                (-toward).dot(right).clamp(-1.0, 1.0),
            )
        } else {
            (0.0, 0.0)
        };
        SpatialMix {
            gain: self.audio.volume
                * listener.master_volume
                * self.distance_gain(distance)
                * self.cone_gain(angle),
            pan,
        }
    }
}
