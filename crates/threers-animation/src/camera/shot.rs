//! Multi-shot editorial timeline — cuts and blended transitions between cameras.

use crate::animatable::Animatable;
use crate::easing::Easing;

use super::curve::CameraCurves;
use super::path::CameraPath;
use super::pose::CameraPose;
use super::ramp::SpeedRamp;

/// How one shot hands off to the next.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ShotTransition {
    /// Hard cut at the boundary.
    #[default]
    Cut,
    /// Blend poses across `duration` seconds using `ramp`.
    Blend { duration: f32, ramp: SpeedRamp },
}

impl ShotTransition {
    pub fn ease(duration: f32, easing: Easing) -> Self {
        Self::Blend {
            duration: duration.max(0.0),
            ramp: SpeedRamp::Ease(easing),
        }
    }
}

/// What drives a single shot.
#[derive(Debug, Clone)]
pub enum ShotSource {
    /// Hold a static pose.
    Hold(CameraPose),
    /// Tween from `from` to `to` across the shot duration.
    Fly {
        from: CameraPose,
        to: CameraPose,
        ramp: SpeedRamp,
    },
    /// Follow a path across the shot duration.
    Path(CameraPath),
    /// Evaluate F-curves (times relative to shot start).
    Curves(CameraCurves),
}

/// One editorial shot.
#[derive(Debug, Clone)]
pub struct Shot {
    pub name: String,
    pub duration: f32,
    pub source: ShotSource,
    /// Transition *into* this shot from the previous one.
    pub transition: ShotTransition,
}

impl Shot {
    pub fn hold(name: impl Into<String>, pose: CameraPose, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration: duration.max(0.0),
            source: ShotSource::Hold(pose),
            transition: ShotTransition::Cut,
        }
    }

    pub fn fly(
        name: impl Into<String>,
        from: CameraPose,
        to: CameraPose,
        duration: f32,
        ramp: SpeedRamp,
    ) -> Self {
        Self {
            name: name.into(),
            duration: duration.max(0.0),
            source: ShotSource::Fly { from, to, ramp },
            transition: ShotTransition::Cut,
        }
    }

    pub fn path(name: impl Into<String>, path: CameraPath, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration: duration.max(0.0),
            source: ShotSource::Path(path),
            transition: ShotTransition::Cut,
        }
    }

    pub fn with_transition(mut self, transition: ShotTransition) -> Self {
        self.transition = transition;
        self
    }

    fn sample(&self, local_t: f32) -> CameraPose {
        let u = if self.duration <= 0.0 {
            1.0
        } else {
            (local_t / self.duration).clamp(0.0, 1.0)
        };
        match &self.source {
            ShotSource::Hold(p) => *p,
            ShotSource::Fly { from, to, ramp } => {
                let to = from.unwrap_azimuth_toward(*to);
                from.lerp(to, ramp.apply(u))
            }
            ShotSource::Path(path) => path.sample_pose(u, CameraPose::default()),
            ShotSource::Curves(curves) => curves.evaluate(local_t, CameraPose::default()),
        }
    }
}

/// Ordered list of shots sharing one clock.
#[derive(Debug, Clone, Default)]
pub struct ShotTimeline {
    shots: Vec<Shot>,
    time: f32,
    pub looping: bool,
}

impl ShotTimeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, shot: Shot) -> usize {
        self.shots.push(shot);
        self.shots.len() - 1
    }

    pub fn shots(&self) -> &[Shot] {
        &self.shots
    }

    pub fn duration(&self) -> f32 {
        self.shots.iter().map(|s| s.duration).sum()
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn seek(&mut self, time: f32) {
        let total = self.duration();
        if total <= 0.0 {
            self.time = 0.0;
            return;
        }
        let mut t = time.max(0.0);
        if self.looping {
            t %= total;
        } else {
            t = t.min(total);
        }
        self.time = t;
    }

    pub fn update(&mut self, dt: f32) {
        if dt.is_finite() {
            self.seek(self.time + dt);
        }
    }

    /// Sample the composed pose at the current time (handles blend transitions).
    pub fn pose(&self) -> CameraPose {
        if self.shots.is_empty() {
            return CameraPose::default();
        }
        let (idx, local) = self.shot_at(self.time);
        let current = self.shots[idx].sample(local);

        // Blend from previous shot if we're inside the incoming transition window.
        if idx == 0 {
            return current;
        }
        let prev = &self.shots[idx - 1];
        let cur = &self.shots[idx];
        match cur.transition {
            ShotTransition::Cut => current,
            ShotTransition::Blend { duration, ramp } => {
                let duration = duration.min(cur.duration.max(0.0));
                if duration <= 0.0 || local >= duration {
                    return current;
                }
                let prev_pose = prev.sample(prev.duration);
                let u = ramp.apply((local / duration).clamp(0.0, 1.0));
                let current = prev_pose.unwrap_azimuth_toward(current);
                prev_pose.lerp(current, u)
            }
        }
    }

    fn shot_at(&self, time: f32) -> (usize, f32) {
        let mut acc = 0.0;
        for (i, shot) in self.shots.iter().enumerate() {
            let end = acc + shot.duration;
            if time < end || i + 1 == self.shots.len() {
                return (i, (time - acc).max(0.0));
            }
            acc = end;
        }
        (0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn hard_cut_switches_pose() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::new(1.0, 0.0, 0.0), 1.0, 1.0, 3.0);
        let mut tl = ShotTimeline::new();
        tl.push(Shot::hold("a", a, 1.0));
        tl.push(Shot::hold("b", b, 1.0));
        tl.seek(0.5);
        assert!((tl.pose().radius - 5.0).abs() < 1e-3);
        tl.seek(1.5);
        assert!((tl.pose().radius - 3.0).abs() < 1e-3);
    }

    #[test]
    fn blend_mixes_across_boundary() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 1.0);
        let mut tl = ShotTimeline::new();
        tl.push(Shot::hold("a", a, 1.0));
        tl.push(
            Shot::hold("b", b, 1.0)
                .with_transition(ShotTransition::ease(0.5, Easing::Linear)),
        );
        tl.seek(1.25); // halfway through blend
        let r = tl.pose().radius;
        assert!(r > 1.0 && r < 5.0, "expected blend, got {r}");
    }
}
