//! Clips from the crate's other animation sources: the glTF parser's
//! three.js-shaped tracks, and the older node-id-targeted
//! [`crate::animation::AnimationClip`] (glTF's one-call loader, USD, the
//! mechanism recorder).
//!
//! Tracks name their nodes by uuid, `@<id>` ([`SceneHost::uuid_of`]), so a
//! clip drives exactly the nodes it was loaded or recorded for.

use super::scene_host::SceneHost;
use super::track::{Interpolation, KeyframeTrack, TrackValues, ValueType};
use super::AnimationClip;
use crate::animation as legacy;
use crate::core::{ObjectArena, ObjectKind};
use crate::loaders::gltf_parser::{GltfClip, GltfObject, TrackInterpolation, TrackKind};

impl AnimationClip {
    /// A glTF animation as three.js's `GLTFLoader` builds it — the same
    /// tracks (`position`, `quaternion`, `scale`, `morphTargetInfluences`),
    /// the same interpolation, cubic splines through
    /// `GLTFCubicSplineInterpolant` — with each track naming its node by uuid.
    pub fn from_gltf(clip: &GltfClip, objects: &[GltfObject]) -> AnimationClip {
        let tracks = clip
            .tracks
            .iter()
            .filter_map(|t| {
                let id = objects.get(t.target)?.id;
                let value_type = match t.kind {
                    TrackKind::Number => ValueType::Number,
                    TrackKind::Vector => ValueType::Vector,
                    TrackKind::Quaternion => ValueType::Quaternion,
                };
                let interpolation = match t.interpolation {
                    TrackInterpolation::Step => Interpolation::Discrete,
                    TrackInterpolation::Linear => Interpolation::Linear,
                    TrackInterpolation::CubicSpline => Interpolation::GltfCubicSpline,
                };
                let mut track = KeyframeTrack::new(
                    value_type,
                    format!("{}.{}", SceneHost::uuid_of(id), t.property),
                    t.times.to_vec(),
                    TrackValues::Numeric(t.values.to_vec()),
                    None,
                )
                .ok()?;
                track.set_interpolation(interpolation);
                Some(track)
            })
            .collect();
        AnimationClip::new(clip.name.clone(), -1.0, tracks)
    }

    /// A node-id-targeted clip as a three.js one. Colour tracks bind a
    /// light's colour or a mesh's material colour, whichever the node is;
    /// visibility binds `visible` (positive values show the node, as before);
    /// free scalar tracks have nothing to bind and are left out.
    pub fn from_legacy(clip: &legacy::AnimationClip, arena: &ObjectArena) -> AnimationClip {
        let tracks = clip
            .tracks
            .iter()
            .filter_map(|t| {
                let node = SceneHost::uuid_of(t.object);
                let is_light = matches!(
                    arena.get(t.object).map(|o| &o.kind),
                    Some(ObjectKind::Light(_))
                );
                let (value_type, property) = match t.target {
                    legacy::TrackTarget::Position => (ValueType::Vector, "position".to_string()),
                    legacy::TrackTarget::Quaternion => {
                        (ValueType::Quaternion, "quaternion".to_string())
                    }
                    legacy::TrackTarget::Scale => (ValueType::Vector, "scale".to_string()),
                    legacy::TrackTarget::Color if is_light => {
                        (ValueType::Color, "color".to_string())
                    }
                    legacy::TrackTarget::Color => (ValueType::Color, "material.color".to_string()),
                    legacy::TrackTarget::MorphWeight { index } => {
                        (ValueType::Number, format!("morphTargetInfluences[{index}]"))
                    }
                    legacy::TrackTarget::Intensity => (ValueType::Number, "intensity".to_string()),
                    legacy::TrackTarget::Visibility => (ValueType::Number, "visible".to_string()),
                    legacy::TrackTarget::Scalar => return None,
                };
                let values: Vec<f32> = match &t.values {
                    legacy::TrackValues::Vector(v) => {
                        v.iter().flat_map(|v| [v.x, v.y, v.z]).collect()
                    }
                    legacy::TrackValues::Quaternion(q) => {
                        q.iter().flat_map(|q| [q.x, q.y, q.z, q.w]).collect()
                    }
                    legacy::TrackValues::Color(c) => {
                        c.iter().flat_map(|c| [c.r, c.g, c.b]).collect()
                    }
                    legacy::TrackValues::Scalar(s) => {
                        if matches!(t.target, legacy::TrackTarget::Visibility) {
                            s.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect()
                        } else {
                            s.clone()
                        }
                    }
                };
                let interpolation = match t.interpolation {
                    legacy::Interpolation::Step => Interpolation::Discrete,
                    // The old mixer sampled its cubic tracks linearly.
                    legacy::Interpolation::Linear | legacy::Interpolation::Cubic => {
                        Interpolation::Linear
                    }
                };
                let interpolation = if matches!(t.target, legacy::TrackTarget::Visibility) {
                    Interpolation::Discrete
                } else {
                    interpolation
                };
                KeyframeTrack::new(
                    value_type,
                    format!("{node}.{property}"),
                    t.times.clone(),
                    TrackValues::Numeric(values),
                    Some(interpolation),
                )
                .ok()
            })
            .collect();
        AnimationClip::new(clip.name.clone(), clip.duration as f64, tracks)
    }
}
