//! Matchmove / tracking data — consume solved cameras and 2D tracks.
//!
//! Full CV solving lives outside this crate; these types hold the *results*
//! (Blender Camera Solver / Follow Track equivalents).

use super::pose::CameraPose;
use threers::math::Vector3;

/// One solved camera key from a matchmove.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolvedCameraKey {
    pub time: f32,
    pub pose: CameraPose,
}

/// Discrete solved camera animation (Camera Solver output).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolvedCameraClip {
    pub keys: Vec<SolvedCameraKey>,
}

impl SolvedCameraClip {
    pub fn new(keys: Vec<SolvedCameraKey>) -> Self {
        let mut keys = keys;
        keys.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Self { keys }
    }

    pub fn sample(&self, t: f32) -> Option<CameraPose> {
        if self.keys.is_empty() {
            return None;
        }
        if t <= self.keys[0].time {
            return Some(self.keys[0].pose);
        }
        let last = self.keys.len() - 1;
        if t >= self.keys[last].time {
            return Some(self.keys[last].pose);
        }
        let mut i = 0;
        while i + 1 < self.keys.len() && self.keys[i + 1].time < t {
            i += 1;
        }
        let a = self.keys[i];
        let b = self.keys[i + 1];
        let dt = (b.time - a.time).max(1e-8);
        let u = ((t - a.time) / dt).clamp(0.0, 1.0);
        use crate::animatable::Animatable;
        Some(a.pose.lerp(a.pose.unwrap_azimuth_toward(b.pose), u))
    }
}

/// A 2D track marker over time (normalised film coords, centre origin, y up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackMarker2d {
    pub time: f32,
    pub x: f32,
    pub y: f32,
}

/// Follow Track: either a reconstructed 3D point or a screen-space look hint.
#[derive(Debug, Clone, PartialEq)]
pub struct FollowTrackData {
    /// Optional reconstructed 3D point path (world).
    pub point3d: Vec<(f32, Vector3)>,
    /// Optional 2D markers (used when no 3D point — offsets look-at in view plane).
    pub markers2d: Vec<TrackMarker2d>,
    /// Depth used when only 2D markers are available.
    pub depth: f32,
}

impl Default for FollowTrackData {
    fn default() -> Self {
        Self {
            point3d: Vec::new(),
            markers2d: Vec::new(),
            depth: 5.0,
        }
    }
}

impl FollowTrackData {
    pub fn from_point3d(points: Vec<(f32, Vector3)>) -> Self {
        Self {
            point3d: points,
            ..Self::default()
        }
    }

    pub fn apply(&self, base: CameraPose, t: f32) -> CameraPose {
        if let Some(p) = self.sample_point3d(t) {
            let mut out = CameraPose::from_look_at(base.eye(), p, base.fov);
            out.roll = base.roll;
            out.focus_distance = (base.eye() - p).length();
            out.aperture = base.aperture;
            return out;
        }
        if let Some(m) = self.sample_2d(t) {
            let eye = base.eye();
            let forward = (base.target - eye).normalize();
            let right = {
                let r = forward.cross(Vector3::UP);
                if r.length_sq() < 1e-10 {
                    Vector3::RIGHT
                } else {
                    r.normalize()
                }
            };
            let up = right.cross(forward).normalize();
            let depth = self.depth.max(1e-3);
            let half_v = (base.fov * 0.5).tan();
            let target = eye + forward * depth + right * (m.x * half_v * depth)
                + up * (m.y * half_v * depth);
            let mut out = CameraPose::from_look_at(eye, target, base.fov);
            out.roll = base.roll;
            out.focus_distance = depth;
            out.aperture = base.aperture;
            return out;
        }
        base
    }

    fn sample_point3d(&self, t: f32) -> Option<Vector3> {
        if self.point3d.is_empty() {
            return None;
        }
        if t <= self.point3d[0].0 {
            return Some(self.point3d[0].1);
        }
        let last = self.point3d.len() - 1;
        if t >= self.point3d[last].0 {
            return Some(self.point3d[last].1);
        }
        let mut i = 0;
        while i + 1 < self.point3d.len() && self.point3d[i + 1].0 < t {
            i += 1;
        }
        let (t0, a) = self.point3d[i];
        let (t1, b) = self.point3d[i + 1];
        let u = ((t - t0) / (t1 - t0).max(1e-8)).clamp(0.0, 1.0);
        Some(a.lerp(b, u))
    }

    fn sample_2d(&self, t: f32) -> Option<TrackMarker2d> {
        if self.markers2d.is_empty() {
            return None;
        }
        if t <= self.markers2d[0].time {
            return Some(self.markers2d[0]);
        }
        let last = self.markers2d.len() - 1;
        if t >= self.markers2d[last].time {
            return Some(self.markers2d[last]);
        }
        let mut i = 0;
        while i + 1 < self.markers2d.len() && self.markers2d[i + 1].time < t {
            i += 1;
        }
        let a = self.markers2d[i];
        let b = self.markers2d[i + 1];
        let u = ((t - a.time) / (b.time - a.time).max(1e-8)).clamp(0.0, 1.0);
        Some(TrackMarker2d {
            time: t,
            x: a.x + (b.x - a.x) * u,
            y: a.y + (b.y - a.y) * u,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solved_clip_lerps() {
        let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let b = CameraPose::new(Vector3::ZERO, 1.0, 1.0, 3.0);
        let clip = SolvedCameraClip::new(vec![
            SolvedCameraKey { time: 0.0, pose: a },
            SolvedCameraKey { time: 1.0, pose: b },
        ]);
        let m = clip.sample(0.5).unwrap();
        assert!(m.radius > 3.0 && m.radius < 5.0);
    }

    #[test]
    fn follow_track_point3d() {
        let data = FollowTrackData::from_point3d(vec![
            (0.0, Vector3::ZERO),
            (1.0, Vector3::new(2.0, 0.0, 0.0)),
        ]);
        let base = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let out = data.apply(base, 1.0);
        assert!((out.target.x - 2.0).abs() < 1e-3);
    }
}
