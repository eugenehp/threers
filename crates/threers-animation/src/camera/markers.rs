//! Bind cameras to timeline markers (Blender “Bind Camera to Markers”).

use super::multicam::{CamTake, GateMask};
use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use crate::animatable::Animatable;
use crate::easing::Easing;

/// A named marker on the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineMarker {
    pub time: f32,
    pub name: String,
    /// Index into the bound camera list.
    pub camera: Option<usize>,
}

impl TimelineMarker {
    pub fn new(time: f32, name: impl Into<String>) -> Self {
        Self {
            time,
            name: name.into(),
            camera: None,
        }
    }

    pub fn with_camera(mut self, index: usize) -> Self {
        self.camera = Some(index);
        self
    }
}

/// Marker-driven multi-cam binder.
#[derive(Debug, Clone, Default)]
pub struct MarkerCameraBind {
    pub markers: Vec<TimelineMarker>,
    pub cameras: Vec<CamTake>,
    /// Soft cut duration when switching (0 = hard).
    pub dissolve: f32,
    pub dissolve_ramp: SpeedRamp,
}

impl MarkerCameraBind {
    pub fn new() -> Self {
        Self {
            dissolve_ramp: SpeedRamp::Ease(Easing::Linear),
            ..Self::default()
        }
    }

    pub fn push_camera(&mut self, take: CamTake) -> usize {
        self.cameras.push(take);
        self.cameras.len() - 1
    }

    pub fn push_marker(&mut self, marker: TimelineMarker) {
        self.markers.push(marker);
        self.markers
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Active camera index at `time` (last marker at or before time).
    pub fn active_index(&self, time: f32) -> Option<usize> {
        let mut current = None;
        for m in &self.markers {
            if m.time <= time {
                if let Some(i) = m.camera {
                    current = Some(i);
                }
            } else {
                break;
            }
        }
        current.or(if self.cameras.is_empty() { None } else { Some(0) })
    }

    pub fn pose(&self, time: f32) -> CameraPose {
        let Some(idx) = self.active_index(time) else {
            return CameraPose::default();
        };
        let Some(take) = self.cameras.get(idx) else {
            return CameraPose::default();
        };
        if self.dissolve <= 0.0 {
            return take.pose;
        }
        // Find previous marker camera for dissolve.
        let mut prev_idx = None;
        let mut switch_time = 0.0;
        for m in &self.markers {
            if m.time <= time {
                if let Some(i) = m.camera {
                    if Some(i) != prev_idx {
                        switch_time = m.time;
                    }
                    prev_idx = Some(i);
                }
            }
        }
        let Some(prev) = prev_idx else {
            return take.pose;
        };
        if prev == idx || time - switch_time >= self.dissolve {
            return take.pose;
        }
        let from = self.cameras.get(prev).map(|c| c.pose).unwrap_or(take.pose);
        let u = self
            .dissolve_ramp
            .apply(((time - switch_time) / self.dissolve).clamp(0.0, 1.0));
        let to = from.unwrap_azimuth_toward(take.pose);
        from.lerp(to, u)
    }

    pub fn gate(&self, time: f32) -> Option<GateMask> {
        self.active_index(time)
            .and_then(|i| self.cameras.get(i))
            .and_then(|t| t.gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn marker_switches_camera() {
        let mut bind = MarkerCameraBind::new();
        let a = bind.push_camera(CamTake::new(
            "a",
            CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0),
        ));
        let b = bind.push_camera(CamTake::new(
            "b",
            CameraPose::new(Vector3::ZERO, 0.0, 1.2, 2.0),
        ));
        bind.push_marker(TimelineMarker::new(0.0, "start").with_camera(a));
        bind.push_marker(TimelineMarker::new(1.0, "cut").with_camera(b));
        assert!((bind.pose(0.5).radius - 5.0).abs() < 1e-3);
        assert!((bind.pose(1.5).radius - 2.0).abs() < 1e-3);
    }
}
