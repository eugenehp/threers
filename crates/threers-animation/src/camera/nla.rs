//! NLA-style strips for camera clips.

use super::mixer::CameraClip;
use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use crate::animatable::Animatable;
use crate::easing::Easing;

/// One NLA strip on the camera track.
#[derive(Debug, Clone)]
pub struct NlaStrip {
    pub name: String,
    pub clip: CameraClip,
    /// Strip placement on the timeline.
    pub frame_start: f32,
    pub frame_end: f32,
    /// Range of the action to play (mapped into the strip).
    pub action_start: f32,
    pub action_end: f32,
    pub blend_in: f32,
    pub blend_out: f32,
    pub mute: bool,
    pub influence: f32,
    /// Playback speed inside the strip (`1` = normal).
    pub time_scale: f32,
    /// How many times to repeat the action inside the strip.
    pub repeat: f32,
    pub blend_ramp: SpeedRamp,
}

impl NlaStrip {
    pub fn new(name: impl Into<String>, clip: CameraClip, frame_start: f32, frame_end: f32) -> Self {
        let duration = clip.duration.max(0.0);
        Self {
            name: name.into(),
            clip,
            frame_start,
            frame_end: frame_end.max(frame_start),
            action_start: 0.0,
            action_end: duration,
            blend_in: 0.0,
            blend_out: 0.0,
            mute: false,
            influence: 1.0,
            time_scale: 1.0,
            repeat: 1.0,
            blend_ramp: SpeedRamp::Ease(Easing::Linear),
        }
    }

    pub fn length(&self) -> f32 {
        (self.frame_end - self.frame_start).max(0.0)
    }

    fn strip_weight(&self, time: f32) -> f32 {
        if self.mute || time < self.frame_start || time > self.frame_end {
            return 0.0;
        }
        let local = time - self.frame_start;
        let len = self.length().max(1e-8);
        let mut w = self.influence.clamp(0.0, 1.0);
        if self.blend_in > 0.0 && local < self.blend_in {
            w *= self.blend_ramp.apply((local / self.blend_in).clamp(0.0, 1.0));
        }
        if self.blend_out > 0.0 && local > len - self.blend_out {
            let u = ((len - local) / self.blend_out).clamp(0.0, 1.0);
            w *= self.blend_ramp.apply(u);
        }
        w
    }

    fn action_time(&self, time: f32) -> f32 {
        let local = ((time - self.frame_start) * self.time_scale.max(1e-6)).max(0.0);
        let action_len = (self.action_end - self.action_start).max(1e-8);
        let spanned = action_len * self.repeat.max(1e-6);
        let mut u = local % spanned;
        // ping within one action length
        let cycles = (u / action_len).floor();
        u -= cycles * action_len;
        self.action_start + u.min(action_len)
    }

    pub fn sample(&self, time: f32, template: CameraPose) -> Option<(CameraPose, f32)> {
        let w = self.strip_weight(time);
        if w <= 1e-8 {
            return None;
        }
        let at = self.action_time(time);
        Some((self.clip.sample(at, template), w))
    }
}

/// Stack of NLA strips (bottom → top).
#[derive(Debug, Clone, Default)]
pub struct NlaTrack {
    pub strips: Vec<NlaStrip>,
}

impl NlaTrack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, strip: NlaStrip) {
        self.strips.push(strip);
    }

    pub fn evaluate(&self, time: f32, template: CameraPose) -> CameraPose {
        let mut result = template;
        let mut acc_w = 0.0;
        let mut first = true;
        for strip in &self.strips {
            if let Some((sample, w)) = strip.sample(time, template) {
                if first {
                    result = sample;
                    acc_w = w;
                    first = false;
                } else {
                    let t = w / (acc_w + w).max(1e-8);
                    result = result.unwrap_azimuth_toward(sample);
                    result = result.lerp(sample, t);
                    acc_w += w;
                }
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
    fn strip_plays_clip() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 2.0);
        let clip = CameraClip::from_poses("x", a, b, 1.0);
        let mut track = NlaTrack::new();
        track.push(NlaStrip::new("s", clip, 0.0, 1.0));
        let mid = track.evaluate(0.5, a);
        assert!(mid.radius > 2.0 && mid.radius < 5.0);
    }

    #[test]
    fn mute_skips() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 1.0);
        let clip = CameraClip::from_poses("x", a, b, 1.0);
        let mut strip = NlaStrip::new("s", clip, 0.0, 1.0);
        strip.mute = true;
        let mut track = NlaTrack::new();
        track.push(strip);
        assert!((track.evaluate(0.5, a).radius - 5.0).abs() < 1e-3);
    }
}
