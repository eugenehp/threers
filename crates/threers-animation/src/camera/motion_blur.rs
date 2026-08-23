//! Motion-blur–aware path reparameterisation (constant screen-space speed).

use super::path::CameraPath;
use super::pose::CameraPose;
use threers::math::Vector3;

/// How path parameter maps to motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathSpeedMode {
    /// Constant arc-length speed (default path behaviour).
    #[default]
    ArcLength,
    /// Constant approximate screen-space velocity (steadier motion blur).
    ScreenSpace,
}

/// Build a lookup that remaps linear `t ∈ 0..1` → path `u` so angular /
/// screen-space motion is roughly constant — useful when you want even motion
/// blur along a cinematic move.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenSpaceParam {
    /// Cumulative screen-space "cost" samples, normalised to `0..1` parameter.
    table: Vec<f32>,
}

impl ScreenSpaceParam {
    /// Analyse `path` at `samples` steps using `template` FOV / framing.
    pub fn build(path: &CameraPath, template: CameraPose, samples: usize) -> Self {
        let n = samples.max(2);
        let mut costs = Vec::with_capacity(n);
        costs.push(0.0);
        let mut prev_eye = path.sample_eye(0.0);
        let mut prev_dir = dir_of(prev_eye, path.sample_target(0.0));
        let half_fov = (template.fov * 0.5).tan().max(1e-6);
        let mut acc = 0.0;
        for i in 1..n {
            let u = i as f32 / (n - 1) as f32;
            let eye = path.sample_eye(u);
            let target = path.sample_target(u);
            let dir = dir_of(eye, target);
            // Translational cost in view-scaled units + angular cost.
            let depth = (eye - target).length().max(1e-3);
            let translate = (eye - prev_eye).length() / (depth * half_fov);
            let angular = angle_between(prev_dir, dir) / template.fov.max(1e-3);
            acc += translate + angular * 2.0;
            costs.push(acc);
            prev_eye = eye;
            prev_dir = dir;
        }
        // Normalise to 0..1 cumulative.
        let total = acc.max(1e-8);
        for c in &mut costs {
            *c /= total;
        }
        Self { table: costs }
    }

    /// Map linear time `t` → path parameter with even screen-space speed.
    pub fn remap(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        if self.table.len() < 2 {
            return t;
        }
        // Find segment where cumulative cost crosses t.
        let mut lo = 0usize;
        let mut hi = self.table.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.table[mid] < t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let c0 = self.table[lo];
        let c1 = self.table[hi];
        let u0 = lo as f32 / (self.table.len() - 1) as f32;
        let u1 = hi as f32 / (self.table.len() - 1) as f32;
        if (c1 - c0).abs() < 1e-8 {
            return u0;
        }
        let local = (t - c0) / (c1 - c0);
        u0 + (u1 - u0) * local
    }

    /// Sample a pose with screen-space-constant motion.
    pub fn sample_pose(&self, path: &CameraPath, t: f32, template: CameraPose) -> CameraPose {
        path.sample_pose(self.remap(t), template)
    }
}

fn dir_of(eye: Vector3, target: Vector3) -> Vector3 {
    let d = (target - eye).normalize();
    if d.length_sq() < 1e-10 {
        Vector3::new(0.0, 0.0, -1.0)
    } else {
        d
    }
}

fn angle_between(a: Vector3, b: Vector3) -> f32 {
    let d = a.dot(b).clamp(-1.0, 1.0);
    d.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_preserved() {
        let path = CameraPath::linear(vec![
            Vector3::new(0.0, 0.0, 5.0),
            Vector3::new(5.0, 0.0, 5.0),
            Vector3::new(5.0, 0.0, 0.0),
        ])
        .with_fixed_target(Vector3::ZERO);
        let ssp = ScreenSpaceParam::build(&path, CameraPose::default(), 32);
        assert!(ssp.remap(0.0).abs() < 1e-4);
        assert!((ssp.remap(1.0) - 1.0).abs() < 1e-3);
    }
}
