//! Camera paths: Catmull-Rom / cubic Bezier through eye (and optional interest).

use super::pose::CameraPose;
use super::ramp::SpeedRamp;
use threers::math::Vector3;

/// How control points are interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathKind {
    /// Centripetal Catmull-Rom through the points (no overshoot on sharp corners).
    #[default]
    CatmullRom,
    /// Cubic Bezier: points come in groups of 4 (p0, c0, c1, p1, …).
    Bezier,
    /// Straight segments between consecutive points.
    Linear,
}

/// A world-space curve the camera eye (and optionally look-at) follows.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraPath {
    pub points: Vec<Vector3>,
    pub kind: PathKind,
    /// Optional separate interest / look-at curve. When empty, look-at is fixed
    /// or derived from path tangent.
    pub interest: Vec<Vector3>,
    pub interest_kind: PathKind,
    /// When no interest path, look this far ahead along the tangent.
    pub look_ahead: f32,
    /// Fixed look-at when `interest` is empty and `look_ahead <= 0`.
    pub fixed_target: Option<Vector3>,
    pub ramp: SpeedRamp,
    /// FOV along the path (`None` = leave pose FOV alone when sampling into a pose).
    pub fov: Option<f32>,
    /// Arc-length table for constant-speed travel. Built lazily.
    lengths: Vec<f32>,
    total_length: f32,
}

impl Default for CameraPath {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            kind: PathKind::CatmullRom,
            interest: Vec::new(),
            interest_kind: PathKind::CatmullRom,
            look_ahead: 1.0,
            fixed_target: None,
            ramp: SpeedRamp::Smoother,
            fov: None,
            lengths: Vec::new(),
            total_length: 0.0,
        }
    }
}

impl CameraPath {
    pub fn catmull_rom(points: Vec<Vector3>) -> Self {
        let mut p = Self {
            points,
            kind: PathKind::CatmullRom,
            ..Self::default()
        };
        p.rebuild_arc_length();
        p
    }

    pub fn bezier(points: Vec<Vector3>) -> Self {
        let mut p = Self {
            points,
            kind: PathKind::Bezier,
            ..Self::default()
        };
        p.rebuild_arc_length();
        p
    }

    pub fn linear(points: Vec<Vector3>) -> Self {
        let mut p = Self {
            points,
            kind: PathKind::Linear,
            ..Self::default()
        };
        p.rebuild_arc_length();
        p
    }

    pub fn with_interest(mut self, points: Vec<Vector3>) -> Self {
        self.interest = points;
        self
    }

    pub fn with_ramp(mut self, ramp: SpeedRamp) -> Self {
        self.ramp = ramp;
        self
    }

    pub fn with_fov(mut self, fov: f32) -> Self {
        self.fov = Some(fov.max(1e-3));
        self
    }

    pub fn with_fixed_target(mut self, target: Vector3) -> Self {
        self.fixed_target = Some(target);
        self.look_ahead = 0.0;
        self
    }

    pub fn total_length(&self) -> f32 {
        self.total_length
    }

    pub fn rebuild_arc_length(&mut self) {
        const SAMPLES: usize = 64;
        self.lengths.clear();
        self.lengths.push(0.0);
        if self.points.len() < 2 {
            self.total_length = 0.0;
            return;
        }
        let mut prev = self.sample_raw(0.0);
        let mut acc = 0.0;
        for i in 1..=SAMPLES {
            let u = i as f32 / SAMPLES as f32;
            let p = self.sample_raw(u);
            acc += (p - prev).length();
            self.lengths.push(acc);
            prev = p;
        }
        self.total_length = acc;
    }

    /// Sample eye position at normalised `t ∈ 0..1` with speed ramp + arc-length.
    pub fn sample_eye(&self, t: f32) -> Vector3 {
        let u = self.arc_parameter(self.ramp.apply(t.clamp(0.0, 1.0)));
        self.sample_raw(u)
    }

    /// Sample look-at at normalised `t`.
    pub fn sample_target(&self, t: f32) -> Vector3 {
        let t = t.clamp(0.0, 1.0);
        let u = self.arc_parameter(self.ramp.apply(t));
        if self.interest.len() >= 2 {
            return sample_curve(&self.interest, self.interest_kind, u);
        }
        if let Some(fixed) = self.fixed_target {
            return fixed;
        }
        if self.look_ahead > 0.0 && self.total_length > 1e-6 {
            let ahead = (u + self.look_ahead / self.total_length).min(1.0);
            return self.sample_raw(ahead);
        }
        // Fall back to path tangent.
        let eye = self.sample_raw(u);
        let next = self.sample_raw((u + 0.01).min(1.0));
        let dir = (next - eye).normalize();
        if dir.length_sq() < 1e-10 {
            eye + Vector3::new(0.0, 0.0, -1.0)
        } else {
            eye + dir
        }
    }

    /// Sample a full pose (roll = 0, focus/aperture left at defaults unless FOV set).
    pub fn sample_pose(&self, t: f32, template: CameraPose) -> CameraPose {
        let eye = self.sample_eye(t);
        let target = self.sample_target(t);
        let mut pose = CameraPose::from_look_at(eye, target, template.fov);
        if let Some(fov) = self.fov {
            pose.fov = fov;
        }
        pose.roll = template.roll;
        pose.focus_distance = template.focus_distance;
        pose.aperture = template.aperture;
        pose
    }

    fn arc_parameter(&self, t: f32) -> f32 {
        if self.lengths.len() < 2 || self.total_length <= 1e-8 {
            return t;
        }
        let want = t * self.total_length;
        // Binary search the cumulative length table.
        let mut lo = 0usize;
        let mut hi = self.lengths.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.lengths[mid] < want {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let l0 = self.lengths[lo];
        let l1 = self.lengths[hi];
        let seg = (l1 - l0).max(1e-8);
        let local = (want - l0) / seg;
        let u0 = lo as f32 / (self.lengths.len() - 1) as f32;
        let u1 = hi as f32 / (self.lengths.len() - 1) as f32;
        u0 + (u1 - u0) * local
    }

    fn sample_raw(&self, u: f32) -> Vector3 {
        sample_curve(&self.points, self.kind, u.clamp(0.0, 1.0))
    }
}

fn sample_curve(points: &[Vector3], kind: PathKind, u: f32) -> Vector3 {
    if points.is_empty() {
        return Vector3::ZERO;
    }
    if points.len() == 1 {
        return points[0];
    }
    match kind {
        PathKind::Linear => sample_linear(points, u),
        PathKind::Bezier => sample_bezier(points, u),
        PathKind::CatmullRom => sample_catmull(points, u),
    }
}

fn sample_linear(points: &[Vector3], u: f32) -> Vector3 {
    let n = points.len() - 1;
    let f = u * n as f32;
    let i = (f.floor() as usize).min(n - 1);
    let t = f - i as f32;
    points[i].lerp(points[i + 1], t)
}

fn sample_bezier(points: &[Vector3], u: f32) -> Vector3 {
    // Groups of 4: p0,c0,c1,p1. Overlapping end = next start.
    let segments = (points.len().saturating_sub(1)) / 3;
    if segments == 0 {
        return sample_linear(points, u);
    }
    let f = u * segments as f32;
    let s = (f.floor() as usize).min(segments - 1);
    let t = f - s as f32;
    let i = s * 3;
    let p0 = points[i];
    let c0 = points[i + 1];
    let c1 = points[i + 2];
    let p1 = points[i + 3];
    let omt = 1.0 - t;
    p0 * (omt * omt * omt) + c0 * (3.0 * omt * omt * t) + c1 * (3.0 * omt * t * t) + p1 * (t * t * t)
}

fn sample_catmull(points: &[Vector3], u: f32) -> Vector3 {
    let n = points.len() - 1;
    let f = u * n as f32;
    let i = (f.floor() as usize).min(n - 1);
    let t = f - i as f32;
    let p0 = points[i.saturating_sub(1)];
    let p1 = points[i];
    let p2 = points[(i + 1).min(points.len() - 1)];
    let p3 = points[(i + 2).min(points.len() - 1)];
    // Uniform Catmull-Rom.
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * (-0.5 * t3 + t2 - 0.5 * t)
        + p1 * (1.5 * t3 - 2.5 * t2 + 1.0)
        + p2 * (-1.5 * t3 + 2.0 * t2 + 0.5 * t)
        + p3 * (0.5 * t3 - 0.5 * t2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_endpoints() {
        let path = CameraPath::catmull_rom(vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 1.0, 0.0),
            Vector3::new(3.0, 1.0, 0.0),
        ]);
        let a = path.sample_eye(0.0);
        let b = path.sample_eye(1.0);
        assert!(a.length() < 1e-3, "{a:?}");
        assert!((b - Vector3::new(3.0, 1.0, 0.0)).length() < 1e-2, "{b:?}");
    }

    #[test]
    fn fixed_target_is_honoured() {
        let path = CameraPath::linear(vec![Vector3::new(0.0, 0.0, 5.0), Vector3::new(5.0, 0.0, 5.0)])
            .with_fixed_target(Vector3::ZERO);
        let t = path.sample_target(0.5);
        assert!(t.length() < 1e-5);
    }
}
