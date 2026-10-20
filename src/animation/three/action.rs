//! three.js r165 `AnimationAction`'s state and the parts of its update that
//! need nothing but the action (`_updateTime`, `_setEndings`). Everything
//! that goes through the mixer — play, stop, fades, warps, the weight and
//! time-scale curves — is on [`AnimationMixer`](super::AnimationMixer),
//! addressed by [`ActionId`](super::ActionId).

use std::sync::Arc;

use super::clip::AnimationClip;
use super::interpolant::{
    Interpolant, InterpolantSettings, WRAP_AROUND_ENDING, ZERO_CURVATURE_ENDING, ZERO_SLOPE_ENDING,
};
use super::mixer::{BindingId, ControlId, MixerRoot};
use super::{HostKind, LOOP_ONCE, LOOP_PING_PONG, LOOP_REPEAT};

/// A `'finished'` or `'loop'` event, before it is tied to an action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionEvent {
    /// `{ type: 'finished', direction }`.
    Finished { direction: f64 },
    /// `{ type: 'loop', loopDelta }`.
    Loop { loop_delta: f64 },
}

/// three.js `AnimationAction`.
#[derive(Debug)]
pub struct AnimationAction<K: HostKind> {
    pub(crate) clip: Arc<AnimationClip>,
    pub(crate) local_root: Option<MixerRoot<K>>,
    /// `NormalAnimationBlendMode` or `AdditiveAnimationBlendMode`.
    pub blend_mode: u32,
    pub(crate) interpolants: Vec<Interpolant>,
    pub(crate) interpolant_settings: InterpolantSettings,
    pub(crate) property_bindings: Vec<Option<BindingId>>,
    pub(crate) cache_index: Option<usize>,
    pub(crate) by_clip_cache_index: Option<usize>,
    pub(crate) time_scale_interpolant: Option<ControlId>,
    pub(crate) weight_interpolant: Option<ControlId>,
    /// `LoopOnce` (2200), `LoopRepeat` (2201) or `LoopPingPong` (2202).
    pub loop_mode: u32,
    pub(crate) loop_count: f64,
    pub(crate) start_time: Option<f64>,
    /// Local time, seconds.
    pub time: f64,
    pub time_scale: f64,
    pub(crate) effective_time_scale: f64,
    pub weight: f64,
    pub(crate) effective_weight: f64,
    /// Repetitions when looping (`Infinity` by default).
    pub repetitions: f64,
    /// `true`: zero effective time scale.
    pub paused: bool,
    /// `false`: zero effective weight.
    pub enabled: bool,
    pub clamp_when_finished: bool,
    pub zero_slope_at_start: bool,
    pub zero_slope_at_end: bool,
}

impl<K: HostKind> AnimationAction<K> {
    pub(crate) fn new(
        clip: Arc<AnimationClip>,
        local_root: Option<MixerRoot<K>>,
        blend_mode: u32,
    ) -> Self {
        let interpolants = clip.tracks.iter().map(|t| t.create_interpolant()).collect();
        let n = clip.tracks.len();
        Self {
            clip,
            local_root,
            blend_mode,
            interpolants,
            interpolant_settings: InterpolantSettings {
                ending_start: ZERO_CURVATURE_ENDING,
                ending_end: ZERO_CURVATURE_ENDING,
            },
            property_bindings: vec![None; n],
            cache_index: None,
            by_clip_cache_index: None,
            time_scale_interpolant: None,
            weight_interpolant: None,
            loop_mode: LOOP_REPEAT,
            loop_count: -1.0,
            start_time: None,
            time: 0.0,
            time_scale: 1.0,
            effective_time_scale: 1.0,
            weight: 1.0,
            effective_weight: 1.0,
            repetitions: f64::INFINITY,
            paused: false,
            enabled: true,
            clamp_when_finished: false,
            zero_slope_at_start: true,
            zero_slope_at_end: true,
        }
    }

    /// `getClip()`.
    pub fn clip(&self) -> &Arc<AnimationClip> {
        &self.clip
    }
    /// The action's own root (`_localRoot`), if it was given one.
    pub fn local_root(&self) -> Option<&MixerRoot<K>> {
        self.local_root.as_ref()
    }
    /// `getEffectiveWeight()`.
    pub fn effective_weight(&self) -> f64 {
        self.effective_weight
    }
    /// `getEffectiveTimeScale()`.
    pub fn effective_time_scale(&self) -> f64 {
        self.effective_time_scale
    }
    /// `_loopCount` (-1 before the first update).
    pub fn loop_count(&self) -> f64 {
        self.loop_count
    }
    /// `_startTime` (from `startAt`, until reached).
    pub fn start_time(&self) -> Option<f64> {
        self.start_time
    }
    /// The interpolants' shared ending settings.
    pub fn interpolant_settings(&self) -> InterpolantSettings {
        self.interpolant_settings
    }

    /// `startAt(time)`.
    pub fn start_at(&mut self, time: f64) -> &mut Self {
        self.start_time = Some(time);
        self
    }

    /// `setLoop(mode, repetitions)`.
    pub fn set_loop(&mut self, mode: u32, repetitions: f64) -> &mut Self {
        self.loop_mode = mode;
        self.repetitions = repetitions;
        self
    }

    /// `_updateTime(deltaTime)`: advance and wrap the local time; returns
    /// the clip time to sample and the event to dispatch, if any.
    pub(crate) fn update_time(&mut self, delta_time: f64) -> (f64, Option<ActionEvent>) {
        let duration = self.clip.duration;
        let loop_mode = self.loop_mode;
        let mut time = self.time + delta_time;
        let mut loop_count = self.loop_count;
        let ping_pong = loop_mode == LOOP_PING_PONG;
        if delta_time == 0.0 {
            if loop_count == -1.0 {
                return (time, None);
            }
            return (
                if ping_pong && (to_int32(loop_count) & 1) == 1 {
                    duration - time
                } else {
                    time
                },
                None,
            );
        }
        let mut event = None;
        if loop_mode == LOOP_ONCE {
            if loop_count == -1.0 {
                // just started
                self.loop_count = 0.0;
                self.set_endings(true, true, false);
            }
            'handle_stop: {
                if time >= duration {
                    time = duration;
                } else if time < 0.0 {
                    time = 0.0;
                } else {
                    self.time = time;
                    break 'handle_stop;
                }
                if self.clamp_when_finished {
                    self.paused = true;
                } else {
                    self.enabled = false;
                }
                self.time = time;
                event = Some(ActionEvent::Finished {
                    direction: if delta_time < 0.0 { -1.0 } else { 1.0 },
                });
            }
        } else {
            // repetitive Repeat or PingPong
            if loop_count == -1.0 {
                // just started
                if delta_time >= 0.0 {
                    loop_count = 0.0;
                    self.set_endings(true, self.repetitions == 0.0, ping_pong);
                } else {
                    // when looping in reverse direction, the initial
                    // transition through zero counts as a repetition,
                    // so leave loopCount at -1
                    self.set_endings(self.repetitions == 0.0, true, ping_pong);
                }
            }
            if time >= duration || time < 0.0 {
                // wrap around
                let loop_delta = (time / duration).floor(); // signed
                time -= duration * loop_delta;
                loop_count += loop_delta.abs();
                let pending = self.repetitions - loop_count;
                if pending <= 0.0 {
                    // have to stop (switch state, clamp time, fire event)
                    if self.clamp_when_finished {
                        self.paused = true;
                    } else {
                        self.enabled = false;
                    }
                    time = if delta_time > 0.0 { duration } else { 0.0 };
                    self.time = time;
                    event = Some(ActionEvent::Finished {
                        direction: if delta_time > 0.0 { 1.0 } else { -1.0 },
                    });
                } else {
                    // keep running
                    if pending == 1.0 {
                        // entering the last round
                        let at_start = delta_time < 0.0;
                        self.set_endings(at_start, !at_start, ping_pong);
                    } else {
                        self.set_endings(false, false, ping_pong);
                    }
                    self.loop_count = loop_count;
                    self.time = time;
                    event = Some(ActionEvent::Loop { loop_delta });
                }
            } else {
                self.time = time;
            }
            if ping_pong && (to_int32(loop_count) & 1) == 1 {
                // invert time for the "pong round"
                return (duration - time, event);
            }
        }
        (time, event)
    }

    /// `_setEndings(atStart, atEnd, pingPong)`.
    pub(crate) fn set_endings(&mut self, at_start: bool, at_end: bool, ping_pong: bool) {
        let settings = &mut self.interpolant_settings;
        if ping_pong {
            settings.ending_start = ZERO_SLOPE_ENDING;
            settings.ending_end = ZERO_SLOPE_ENDING;
        } else {
            // assuming for LoopOnce atStart == atEnd == true
            settings.ending_start = if at_start {
                if self.zero_slope_at_start {
                    ZERO_SLOPE_ENDING
                } else {
                    ZERO_CURVATURE_ENDING
                }
            } else {
                WRAP_AROUND_ENDING
            };
            settings.ending_end = if at_end {
                if self.zero_slope_at_end {
                    ZERO_SLOPE_ENDING
                } else {
                    ZERO_CURVATURE_ENDING
                }
            } else {
                WRAP_AROUND_ENDING
            };
        }
    }
}

/// JS `ToInt32`, for `loopCount & 1`.
pub(crate) fn to_int32(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    let m = v.trunc().rem_euclid(4294967296.0);
    (if m >= 2147483648.0 {
        m - 4294967296.0
    } else {
        m
    }) as i32
}
