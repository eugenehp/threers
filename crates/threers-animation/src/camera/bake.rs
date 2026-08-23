//! Bake recorded poses / orbit samples into a reusable [`CameraPath`].

use super::path::{CameraPath, PathKind};
use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use threers::math::Vector3;

/// One recorded sample from an interactive orbit (or any scripted pose).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitSample {
    pub time: f32,
    pub pose: CameraPose,
}

/// Recorder that captures poses over time, then bakes them to a path.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OrbitRecorder {
    samples: Vec<OrbitSample>,
    /// Minimum seconds between accepted samples (dedupe).
    pub min_interval: f32,
    /// Minimum eye travel before accepting a new sample.
    pub min_eye_delta: f32,
}

impl OrbitRecorder {
    pub fn new() -> Self {
        Self {
            min_interval: 1.0 / 30.0,
            min_eye_delta: 1e-3,
            ..Self::default()
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn samples(&self) -> &[OrbitSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Push a sample if it clears the dedupe thresholds.
    pub fn push(&mut self, time: f32, pose: CameraPose) {
        if !time.is_finite() {
            return;
        }
        if let Some(last) = self.samples.last() {
            if time - last.time < self.min_interval
                && (pose.eye() - last.pose.eye()).length() < self.min_eye_delta
            {
                return;
            }
        }
        self.samples.push(OrbitSample { time, pose });
    }

    /// Bake eye + interest curves. Returns `None` if fewer than 2 samples.
    pub fn bake(&self) -> Option<CameraPath> {
        bake_poses(self.samples.iter().map(|s| s.pose))
    }

    /// Total recorded duration.
    pub fn duration(&self) -> f32 {
        match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => (b.time - a.time).max(0.0),
            _ => 0.0,
        }
    }
}

/// Bake an iterator of poses into a Catmull-Rom eye path + interest path.
pub fn bake_poses(poses: impl IntoIterator<Item = CameraPose>) -> Option<CameraPath> {
    let mut eyes = Vec::new();
    let mut interests = Vec::new();
    for p in poses {
        eyes.push(p.eye());
        interests.push(p.target);
    }
    if eyes.len() < 2 {
        return None;
    }
    let mut path = CameraPath::catmull_rom(eyes);
    path.interest = interests;
    path.interest_kind = PathKind::CatmullRom;
    path.look_ahead = 0.0;
    path.fixed_target = None;
    path.ramp = SpeedRamp::Linear;
    path.rebuild_arc_length();
    Some(path)
}

/// Bake raw eye / look-at point lists.
pub fn bake_eye_target(eyes: Vec<Vector3>, targets: Vec<Vector3>) -> Option<CameraPath> {
    if eyes.len() < 2 {
        return None;
    }
    let mut path = CameraPath::catmull_rom(eyes);
    if targets.len() == path.points.len() {
        path.interest = targets;
        path.interest_kind = PathKind::CatmullRom;
        path.look_ahead = 0.0;
        path.fixed_target = None;
    } else if let Some(t) = targets.first().copied() {
        path.fixed_target = Some(t);
        path.look_ahead = 0.0;
    }
    path.ramp = SpeedRamp::Linear;
    path.rebuild_arc_length();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_bakes_path() {
        let mut rec = OrbitRecorder::new();
        rec.min_interval = 0.0;
        rec.min_eye_delta = 0.0;
        for i in 0..5 {
            let t = i as f32 * 0.1;
            let pose = CameraPose::new(Vector3::ZERO, t, 1.2, 5.0);
            rec.push(t, pose);
        }
        let path = rec.bake().expect("path");
        assert!(path.total_length() > 0.0);
        let a = path.sample_eye(0.0);
        let b = path.sample_eye(1.0);
        assert!((a - rec.samples[0].pose.eye()).length() < 0.1);
        assert!((b - rec.samples.last().unwrap().pose.eye()).length() < 0.15);
    }
}
