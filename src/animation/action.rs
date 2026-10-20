use super::mixer::BlendMode;
use super::AnimationClip;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    Once,
    Repeat,
    PingPong,
}

#[derive(Debug, Clone)]
struct Fade {
    from: f32,
    to: f32,
    duration: f32,
    elapsed: f32,
    stop_when_done: bool,
}

/// What a caller asked of an action since the mixer last ran it; the mixer
/// replays these on its three.js action, in order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Request {
    Play,
    Stop,
    Pause,
    FadeIn(f32),
    FadeOut(f32),
    Weight(f32),
}

/// One clip on an [`AnimationMixer`](super::AnimationMixer). The mixer
/// plays it as a three.js `AnimationAction`
/// ([`three::AnimationMixer`](super::three::AnimationMixer)): the fields
/// here are read each update, and `play` / `stop` / `pause` / fades are
/// passed on in the order they were called. Used on its own,
/// [`tick_fade`](Self::tick_fade) and [`advance`](Self::advance) still step
/// it by hand.
#[derive(Debug, Clone)]
pub struct AnimationAction {
    pub clip: AnimationClip,
    pub enabled: bool,
    pub weight: f32,
    pub time_scale: f32,
    pub loop_mode: LoopMode,
    pub blend_mode: BlendMode,
    time: f32,
    direction: f32,
    effective_weight: f32,
    fade: Option<Fade>,
    pub(crate) requests: Vec<Request>,
}

impl AnimationAction {
    pub fn new(clip: AnimationClip) -> Self {
        Self {
            clip,
            enabled: true,
            weight: 1.0,
            time_scale: 1.0,
            loop_mode: LoopMode::Repeat,
            blend_mode: BlendMode::Normal,
            time: 0.0,
            direction: 1.0,
            effective_weight: 1.0,
            fade: None,
            requests: Vec::new(),
        }
    }

    pub fn play(&mut self) {
        self.enabled = true;
        self.requests.push(Request::Play);
    }
    pub fn stop(&mut self) {
        self.enabled = false;
        self.time = 0.0;
        self.fade = None;
        self.effective_weight = 0.0;
        self.requests.push(Request::Stop);
    }
    pub fn pause(&mut self) {
        self.enabled = false;
        self.requests.push(Request::Pause);
    }

    pub fn current_time(&self) -> f32 {
        self.time
    }

    /// What the mixer's three.js action reported after an update.
    pub(crate) fn sync_from(&mut self, time: f32, effective_weight: f32, enabled: bool) {
        self.time = time;
        self.effective_weight = effective_weight;
        if !enabled {
            self.enabled = false;
        }
    }

    pub fn effective_weight(&self) -> f32 {
        self.effective_weight
    }

    pub fn set_effective_weight(&mut self, w: f32) {
        self.weight = w;
        self.effective_weight = w;
        self.fade = None;
        self.requests.push(Request::Weight(w));
    }

    pub fn fade_in(&mut self, duration: f32) {
        self.fade = Some(Fade {
            from: 0.0,
            to: self.weight,
            duration: duration.max(0.0),
            elapsed: 0.0,
            stop_when_done: false,
        });
        self.effective_weight = 0.0;
        self.enabled = true;
        self.requests.push(Request::FadeIn(duration));
    }

    pub fn fade_out(&mut self, duration: f32) {
        self.fade = Some(Fade {
            from: self.effective_weight,
            to: 0.0,
            duration: duration.max(0.0),
            elapsed: 0.0,
            stop_when_done: true,
        });
        self.requests.push(Request::FadeOut(duration));
    }

    pub fn tick_fade(&mut self, dt: f32) {
        let Some(fade) = self.fade.as_mut() else {
            return;
        };
        fade.elapsed += dt.max(0.0);
        let u = if fade.duration <= 0.0 {
            1.0
        } else {
            (fade.elapsed / fade.duration).clamp(0.0, 1.0)
        };
        self.effective_weight = fade.from + (fade.to - fade.from) * u;
        if u >= 1.0 {
            let stop = fade.stop_when_done;
            self.fade = None;
            if stop {
                self.enabled = false;
                self.time = 0.0;
            }
        }
    }

    pub fn advance(&mut self, delta: f32) {
        if self.clip.duration <= 0.0 {
            return;
        }
        let scaled = delta * self.time_scale * self.direction;
        self.time += scaled;
        match self.loop_mode {
            LoopMode::Once => {
                if self.time > self.clip.duration {
                    self.time = self.clip.duration;
                    self.enabled = false;
                }
            }
            LoopMode::Repeat => {
                while self.time > self.clip.duration {
                    self.time -= self.clip.duration;
                }
                while self.time < 0.0 {
                    self.time += self.clip.duration;
                }
            }
            LoopMode::PingPong => {
                if self.time > self.clip.duration {
                    self.time = self.clip.duration - (self.time - self.clip.duration);
                    self.direction = -1.0;
                }
                if self.time < 0.0 {
                    self.time = -self.time;
                    self.direction = 1.0;
                }
            }
        }
    }
}
