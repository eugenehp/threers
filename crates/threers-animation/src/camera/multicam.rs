//! Multi-cam switching with letterbox / gate masks.

use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use crate::animatable::Animatable;
use crate::easing::Easing;

/// Film / sensor gate overlay (aspect crop guides).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateMask {
    /// Display aspect to crop to (e.g. 2.39 for scope).
    pub aspect: f32,
    /// Opacity of the masked bars `0..1`.
    pub opacity: f32,
}

impl GateMask {
    pub const SCOPE_239: Self = Self {
        aspect: 2.39,
        opacity: 0.85,
    };
    pub const FLAT_185: Self = Self {
        aspect: 1.85,
        opacity: 0.85,
    };
    pub const SQUARE: Self = Self {
        aspect: 1.0,
        opacity: 0.85,
    };

    /// Height of letterbox bars as a fraction of frame height for a viewport aspect.
    pub fn bar_fraction(self, viewport_aspect: f32) -> f32 {
        let va = viewport_aspect.max(1e-6);
        let ga = self.aspect.max(1e-6);
        if va <= ga {
            // Pillarbox — vertical bars; report side fraction instead.
            0.0
        } else {
            (1.0 - ga / va).clamp(0.0, 1.0) * 0.5
        }
    }

    pub fn side_fraction(self, viewport_aspect: f32) -> f32 {
        let va = viewport_aspect.max(1e-6);
        let ga = self.aspect.max(1e-6);
        if ga <= va {
            0.0
        } else {
            (1.0 - va / ga).clamp(0.0, 1.0) * 0.5
        }
    }
}

/// Letterbox / pillarbox presentation state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub gate: GateMask,
    pub viewport_aspect: f32,
}

impl Letterbox {
    pub fn new(gate: GateMask, viewport_aspect: f32) -> Self {
        Self {
            gate,
            viewport_aspect: viewport_aspect.max(1e-6),
        }
    }

    pub fn top_bottom(&self) -> f32 {
        self.gate.bar_fraction(self.viewport_aspect)
    }

    pub fn left_right(&self) -> f32 {
        self.gate.side_fraction(self.viewport_aspect)
    }
}

/// One camera take in a multi-cam setup.
#[derive(Debug, Clone, PartialEq)]
pub struct CamTake {
    pub name: String,
    pub pose: CameraPose,
    pub gate: Option<GateMask>,
}

impl CamTake {
    pub fn new(name: impl Into<String>, pose: CameraPose) -> Self {
        Self {
            name: name.into(),
            pose,
            gate: None,
        }
    }

    pub fn with_gate(mut self, gate: GateMask) -> Self {
        self.gate = Some(gate);
        self
    }
}

/// Live multi-cam switcher with optional blended cuts.
#[derive(Debug, Clone, Default)]
pub struct MultiCam {
    takes: Vec<CamTake>,
    active: usize,
    /// Soft cut in progress.
    blend_from: Option<usize>,
    blend_elapsed: f32,
    blend_duration: f32,
    blend_ramp: SpeedRamp,
}

impl MultiCam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, take: CamTake) -> usize {
        self.takes.push(take);
        self.takes.len() - 1
    }

    pub fn takes(&self) -> &[CamTake] {
        &self.takes
    }

    pub fn active_index(&self) -> usize {
        self.active
    }

    pub fn active(&self) -> Option<&CamTake> {
        self.takes.get(self.active)
    }

    /// Hard cut.
    pub fn cut_to(&mut self, index: usize) {
        if index < self.takes.len() {
            self.active = index;
            self.blend_from = None;
            self.blend_elapsed = 0.0;
        }
    }

    /// Soft cut over `duration` seconds.
    pub fn dissolve_to(&mut self, index: usize, duration: f32, easing: Easing) {
        if index >= self.takes.len() || index == self.active {
            return;
        }
        self.blend_from = Some(self.active);
        self.active = index;
        self.blend_duration = duration.max(0.0);
        self.blend_elapsed = 0.0;
        self.blend_ramp = SpeedRamp::Ease(easing);
    }

    pub fn update(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        if self.blend_from.is_some() {
            self.blend_elapsed += dt;
            if self.blend_elapsed >= self.blend_duration {
                self.blend_from = None;
            }
        }
    }

    /// Current composed pose (blended during dissolves).
    pub fn pose(&self) -> CameraPose {
        let Some(to) = self.takes.get(self.active) else {
            return CameraPose::default();
        };
        let Some(from_idx) = self.blend_from else {
            return to.pose;
        };
        let Some(from) = self.takes.get(from_idx) else {
            return to.pose;
        };
        if self.blend_duration <= 0.0 {
            return to.pose;
        }
        let u = self
            .blend_ramp
            .apply((self.blend_elapsed / self.blend_duration).clamp(0.0, 1.0));
        let to_pose = from.pose.unwrap_azimuth_toward(to.pose);
        from.pose.lerp(to_pose, u)
    }

    /// Active gate (destination take during a dissolve).
    pub fn gate(&self) -> Option<GateMask> {
        self.takes.get(self.active).and_then(|t| t.gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::math::Vector3;

    #[test]
    fn hard_cut_switches() {
        let a = CamTake::new("a", CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0));
        let b = CamTake::new("b", CameraPose::new(Vector3::ZERO, 1.0, 1.0, 2.0));
        let mut mc = MultiCam::new();
        mc.push(a);
        mc.push(b);
        assert!((mc.pose().radius - 5.0).abs() < 1e-4);
        mc.cut_to(1);
        assert!((mc.pose().radius - 2.0).abs() < 1e-4);
    }

    #[test]
    fn scope_bars_on_wide_viewport() {
        // Viewport wider than 2.39 → horizontal letterbox bars.
        let lb = Letterbox::new(GateMask::SCOPE_239, 2.8);
        assert!(lb.top_bottom() > 0.05);
        assert_eq!(lb.left_right(), 0.0);
    }

    #[test]
    fn scope_pillars_on_16_9() {
        // 16:9 is narrower than 2.39 → pillarbox.
        let lb = Letterbox::new(GateMask::SCOPE_239, 16.0 / 9.0);
        assert!(lb.left_right() > 0.05);
        assert_eq!(lb.top_bottom(), 0.0);
    }
}
