use std::sync::Arc;

use super::action::Request;
use super::three::{self, ActionId, MixerRoot, SceneHost, SceneKind};
use super::{AnimationAction, AnimationClip, LoopMode};
use crate::scene::Scene;

/// How an action combines with others (three.js blend modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Additive,
}

/// Plays node-id-targeted clips against a scene. Holds zero or more
/// `AnimationAction`s.
///
/// This is the older, index-based front of the animation system: each
/// action is played by a [`three::AnimationMixer`] — three.js r165's mixer,
/// with its blending, fades, loop modes and bindings — over the scene passed
/// to [`update`](Self::update). Weights below one blend toward the value a
/// property had when its first action started, as three.js does; `LoopMode::Once`
/// holds the last frame (three.js's `clampWhenFinished`).
pub struct AnimationMixer {
    pub actions: Vec<AnimationAction>,
    pub time: f32,
    engine: Option<three::AnimationMixer<SceneKind>>,
    ids: Vec<Option<ActionId>>,
}

impl Default for AnimationMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AnimationMixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationMixer")
            .field("actions", &self.actions.len())
            .field("time", &self.time)
            .finish()
    }
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            time: 0.0,
            engine: None,
            ids: Vec::new(),
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

    /// The three.js mixer playing the actions (created at the first update).
    pub fn engine(&self) -> Option<&three::AnimationMixer<SceneKind>> {
        self.engine.as_ref()
    }

    /// Advance time and apply every active action's tracks to the scene.
    pub fn update(&mut self, scene: &mut Scene, delta: f32) {
        self.time += delta;
        let root = scene.root;
        let engine = self
            .engine
            .get_or_insert_with(|| three::AnimationMixer::new(MixerRoot::Node(root)));
        let mut host = SceneHost::new(&mut scene.arena);
        self.ids.resize(self.actions.len(), None);
        for (i, a) in self.actions.iter_mut().enumerate() {
            let id = match self.ids[i] {
                Some(id) => id,
                None => {
                    let clip = Arc::new(three::AnimationClip::from_legacy(&a.clip, host.arena));
                    let blend = match a.blend_mode {
                        BlendMode::Normal => three::NORMAL_ANIMATION_BLEND_MODE,
                        BlendMode::Additive => three::ADDITIVE_ANIMATION_BLEND_MODE,
                    };
                    let Ok(Some(id)) = engine.clip_action(&mut host, clip, None, Some(blend))
                    else {
                        continue;
                    };
                    self.ids[i] = Some(id);
                    id
                }
            };
            if let Some(ea) = engine.action_mut(id) {
                ea.loop_mode = match a.loop_mode {
                    LoopMode::Once => three::LOOP_ONCE,
                    LoopMode::Repeat => three::LOOP_REPEAT,
                    LoopMode::PingPong => three::LOOP_PING_PONG,
                };
                // A one-shot keeps its last frame, as it always has here.
                ea.clamp_when_finished = a.loop_mode == LoopMode::Once;
                ea.time_scale = a.time_scale as f64;
                ea.blend_mode = match a.blend_mode {
                    BlendMode::Normal => three::NORMAL_ANIMATION_BLEND_MODE,
                    BlendMode::Additive => three::ADDITIVE_ANIMATION_BLEND_MODE,
                };
            }
            let mut faded = false;
            for req in std::mem::take(&mut a.requests) {
                match req {
                    Request::Play => {
                        engine.play(&mut host, id);
                        if let Some(ea) = engine.action_mut(id) {
                            ea.paused = false;
                            ea.enabled = true;
                        }
                    }
                    Request::Stop => engine.stop(&mut host, id),
                    Request::Pause => {
                        if let Some(ea) = engine.action_mut(id) {
                            ea.paused = true;
                        }
                    }
                    Request::FadeIn(d) => {
                        engine.fade_in(id, d as f64);
                        engine.play(&mut host, id);
                        faded = true;
                    }
                    Request::FadeOut(d) => {
                        engine.fade_out(id, d as f64);
                        faded = true;
                    }
                    Request::Weight(w) => engine.set_effective_weight(id, w as f64),
                }
            }
            if !faded {
                if let Some(ea) = engine.action_mut(id) {
                    ea.weight = a.weight as f64;
                }
            }
            if a.enabled {
                if !engine.is_scheduled(id) {
                    engine.play(&mut host, id);
                }
                if let Some(ea) = engine.action_mut(id) {
                    if ea.paused && a.loop_mode != LoopMode::Once {
                        ea.paused = false;
                    }
                }
            } else if engine.is_scheduled(id) {
                if let Some(ea) = engine.action_mut(id) {
                    if ea.enabled {
                        ea.paused = true;
                    }
                }
            }
        }
        engine.update(&mut host, delta as f64);
        for (i, a) in self.actions.iter_mut().enumerate() {
            if let Some(ea) = self
                .ids
                .get(i)
                .copied()
                .flatten()
                .and_then(|id| engine.action(id))
            {
                let running = ea.enabled && !(ea.paused && a.loop_mode == LoopMode::Once);
                a.sync_from(ea.time as f32, ea.effective_weight() as f32, running);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{KeyframeTrack, TrackTarget};
    use crate::core::Object3D;
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
    fn weighted_position_blends_toward_the_original() {
        let mut scene = Scene::new();
        let id = scene.add(Object3D::group());
        let track = KeyframeTrack::vector(
            id,
            TrackTarget::Position,
            vec![0.0, 1.0],
            vec![Vector3::ZERO, Vector3::new(2.0, 0.0, 0.0)],
        );
        let clip = AnimationClip::new("move", 1.0, vec![track]);
        let mut mixer = AnimationMixer::new();
        let a = mixer.clip_action(clip);
        mixer.actions[a].loop_mode = LoopMode::Once;
        mixer.actions[a].set_effective_weight(0.5);
        mixer.update(&mut scene, 1.0); // end of clip
        let p = scene.arena.get(id).unwrap().position;
        assert!((p.x - 1.0).abs() < 1e-3, "{p:?}");
    }

    #[test]
    fn once_holds_the_last_frame_and_repeat_wraps() {
        let mut scene = Scene::new();
        let id = scene.add(Object3D::group());
        let track = KeyframeTrack::vector(
            id,
            TrackTarget::Position,
            vec![0.0, 1.0],
            vec![Vector3::ZERO, Vector3::new(4.0, 0.0, 0.0)],
        );
        let mut mixer = AnimationMixer::new();
        let a = mixer.clip_action(AnimationClip::new("once", 1.0, vec![track.clone()]));
        mixer.actions[a].loop_mode = LoopMode::Once;
        for _ in 0..30 {
            mixer.update(&mut scene, 0.1);
        }
        assert!((scene.arena.get(id).unwrap().position.x - 4.0).abs() < 1e-5);
        assert!(!mixer.actions[a].enabled, "finished");

        let mut scene2 = Scene::new();
        let id2 = scene2.add(Object3D::group());
        let mut t2 = track;
        t2.object = id2;
        let mut mixer = AnimationMixer::new();
        mixer.clip_action(AnimationClip::new("loop", 1.0, vec![t2]));
        for _ in 0..5 {
            mixer.update(&mut scene2, 0.25); // 1.25 → 0.25 into the second loop
        }
        assert!((scene2.arena.get(id2).unwrap().position.x - 1.0).abs() < 1e-5);
    }
}
