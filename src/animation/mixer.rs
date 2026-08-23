use super::track::TrackTarget;
use super::{AnimationAction, AnimationClip};
use crate::core::ObjectArena;
use crate::materials::Material;
use crate::math::Quaternion;
use crate::scene::Scene;

/// How an action combines with others (three.js blend modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Additive,
}

/// Plays animation clips against a scene. Holds zero or more `AnimationAction`s.
pub struct AnimationMixer {
    pub actions: Vec<AnimationAction>,
    pub time: f32,
}

impl Default for AnimationMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            time: 0.0,
        }
    }

    pub fn clip_action(&mut self, clip: AnimationClip) -> usize {
        let idx = self.actions.len();
        self.actions.push(AnimationAction::new(clip));
        idx
    }

    /// Cross-fade `to` in while fading `from` out over `duration` seconds.
    pub fn cross_fade(&mut self, from: usize, to: usize, duration: f32) {
        if let Some(a) = self.actions.get_mut(from) {
            a.fade_out(duration);
        }
        if let Some(a) = self.actions.get_mut(to) {
            a.fade_in(duration);
            a.play();
        }
    }

    /// Advance time and apply every active action's tracks to the scene.
    pub fn update(&mut self, scene: &mut Scene, delta: f32) {
        self.time += delta;
        // Two-pass: sample into scratch, then weighted-write. For simplicity we
        // apply sequentially with weight lerp (good enough for layered clips).
        for action in &mut self.actions {
            action.tick_fade(delta);
            if !action.enabled || action.effective_weight() <= 1e-6 {
                continue;
            }
            action.advance(delta);
            apply_action(&mut scene.arena, action);
        }
    }
}

fn apply_action(arena: &mut ObjectArena, action: &AnimationAction) {
    let t = action.current_time();
    let w = action.effective_weight().clamp(0.0, 1.0);
    let additive = matches!(action.blend_mode, BlendMode::Additive);
    for tr in &action.clip.tracks {
        let Some(obj) = arena.nodes.get_mut(tr.object) else {
            continue;
        };
        match tr.target {
            TrackTarget::Position => {
                if let Some(v) = tr.sample_vector(t) {
                    if additive {
                        obj.position = obj.position + v * w;
                    } else if w >= 1.0 - 1e-6 {
                        obj.position = v;
                    } else {
                        obj.position = obj.position.lerp(v, w);
                    }
                }
            }
            TrackTarget::Quaternion => {
                if let Some(q) = tr.sample_quaternion(t) {
                    if additive {
                        // Approximate additive rotation: slerp from identity.
                        let add = Quaternion::identity().slerp(q, w);
                        obj.quaternion = (obj.quaternion * add).normalize();
                    } else if w >= 1.0 - 1e-6 {
                        obj.quaternion = q;
                    } else {
                        obj.quaternion = obj.quaternion.slerp(q, w);
                    }
                }
            }
            TrackTarget::Scale => {
                if let Some(v) = tr.sample_vector(t) {
                    if additive {
                        obj.scale = obj.scale + v * w;
                    } else if w >= 1.0 - 1e-6 {
                        obj.scale = v;
                    } else {
                        obj.scale = obj.scale.lerp(v, w);
                    }
                }
            }
            TrackTarget::Color => {
                if let Some(c) = tr.sample_color(t) {
                    if let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind {
                        if let Some(mat) = std::sync::Arc::get_mut(&mut mesh.material) {
                            apply_color(mat, c);
                        }
                    }
                }
            }
            TrackTarget::MorphWeight { index } => {
                if let Some(s) = tr.sample_scalar(t) {
                    if let crate::core::ObjectKind::Mesh(mesh) = &mut obj.kind {
                        ensure_morph_len(&mut mesh.morph_influences, index + 1);
                        let cur = mesh.morph_influences[index];
                        mesh.morph_influences[index] = if additive {
                            cur + s * w
                        } else {
                            cur + (s - cur) * w
                        };
                    }
                }
            }
            TrackTarget::Scalar => { /* user-driven */ }
        }
    }
}

fn ensure_morph_len(weights: &mut Vec<f32>, len: usize) {
    if weights.len() < len {
        weights.resize(len, 0.0);
    }
}

fn apply_color(mat: &mut Material, c: crate::math::Color) {
    match mat {
        Material::Basic(m) => m.color = c,
        Material::Lambert(m) => m.color = c,
        Material::Phong(m) => m.color = c,
        Material::Standard(m) => m.color = c,
        Material::Physical(m) => m.color = c,
        Material::Toon(m) => m.color = c,
        Material::Matcap(m) => m.color = c,
        Material::Line(m) => m.color = c,
        Material::Points(m) => m.color = c,
        Material::Sprite(m) => m.color = c,
        Material::Mirror(m) => m.color = c,
        Material::Atmosphere(m) => m.color = c,
        Material::Normal(_) | Material::Depth(_) | Material::Distance(_) | Material::Sky(_) => {}
        Material::Shader(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{Interpolation, KeyframeTrack, TrackTarget};
    use crate::core::{Object3D, ObjectArena};
    use crate::math::Vector3;

    #[test]
    fn fade_out_reaches_zero_weight() {
        let mut action = AnimationAction::new(AnimationClip::new("t", 1.0, vec![]));
        action.fade_out(0.5);
        action.tick_fade(0.25);
        assert!((action.effective_weight() - 0.5).abs() < 1e-4);
        action.tick_fade(0.25);
        assert!(action.effective_weight() <= 1e-5);
        assert!(!action.enabled);
    }

    #[test]
    fn weighted_position_lerp() {
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::group());
        arena.nodes.get_mut(id).unwrap().position = Vector3::ZERO;
        let track = KeyframeTrack::vector(
            id,
            TrackTarget::Position,
            vec![0.0, 1.0],
            vec![Vector3::ZERO, Vector3::new(2.0, 0.0, 0.0)],
        );
        let mut track = track;
        track.interpolation = Interpolation::Linear;
        let clip = AnimationClip::new("move", 1.0, vec![track]);
        let mut action = AnimationAction::new(clip);
        action.set_effective_weight(0.5);
        action.advance(1.0); // end of clip
        apply_action(&mut arena, &action);
        let p = arena.nodes.get(id).unwrap().position;
        assert!((p.x - 1.0).abs() < 1e-3, "{p:?}");
    }
}
