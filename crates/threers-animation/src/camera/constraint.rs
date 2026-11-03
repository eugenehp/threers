//! Blender-style camera constraints.

use super::curve_follow::TiltedPath;
use super::mixer::CameraClip;
use super::path::CameraPath;
use super::pose::CameraPose;
use super::tracking::{FollowTrackData, SolvedCameraClip};
use crate::animatable::Animatable;
use threers::math::Vector3;

/// Track axis for Track To / Locked Track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackAxis {
    #[default]
    MinusZ,
    PlusZ,
    PlusY,
    MinusY,
    PlusX,
    MinusX,
}

/// One evaluated constraint in a stack.
///
/// Public and matched on by callers, so the variants stay flat rather than
/// boxing the largest to even them out — a camera rig holds a handful of these,
/// not thousands.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum CameraConstraint {
    Aim {
        target: Vector3,
        hold_eye: bool,
        weight: f32,
    },
    /// Blender Track To — align a local axis toward a target.
    TrackTo {
        target: Vector3,
        track_axis: TrackAxis,
        up: Vector3,
        weight: f32,
    },
    /// Damped Track — spring-smoothed aim (stateful via `current`).
    DampedTrack {
        target: Vector3,
        /// Influence per second toward the target direction.
        influence: f32,
        weight: f32,
        /// Working look-at (updated each evaluate when `dt` provided via stack).
        current: Vector3,
    },
    /// Locked Track — rotate around a locked local axis only.
    LockedTrack {
        target: Vector3,
        lock_axis: TrackAxis,
        weight: f32,
    },
    Parent {
        origin: Vector3,
        forward: Vector3,
        up: Vector3,
        local_eye: Vector3,
        local_target: Vector3,
        weight: f32,
    },
    /// Child Of — full parent transform with per-channel influence.
    ChildOf {
        origin: Vector3,
        forward: Vector3,
        up: Vector3,
        local_eye: Vector3,
        local_target: Vector3,
        loc: bool,
        rot: bool,
        weight: f32,
    },
    PathFollow {
        path: CameraPath,
        percent: f32,
        weight: f32,
    },
    /// Follow Path with curve tilt + radius (Blender curve follow).
    FollowPath {
        path: TiltedPath,
        percent: f32,
        follow_curve: bool,
        weight: f32,
    },
    /// Clamp eye to a path (nearest point).
    ClampTo {
        path: CameraPath,
        weight: f32,
    },
    /// Push eye above a floor plane (y = height by default).
    Floor {
        height: f32,
        normal: Vector3,
        weight: f32,
    },
    LimitLocation {
        min: Vector3,
        max: Vector3,
        weight: f32,
    },
    LimitRotation {
        /// Roll limits (radians).
        min_roll: f32,
        max_roll: f32,
        min_elevation: f32,
        max_elevation: f32,
        weight: f32,
    },
    CopyTransforms {
        source: CameraPose,
        loc: bool,
        rot: bool,
        lens: bool,
        weight: f32,
    },
    /// Evaluate an action/clip at local time.
    Action {
        clip: CameraClip,
        time: f32,
        weight: f32,
    },
    UpVector {
        up: Vector3,
        weight: f32,
    },
    /// Apply a solved matchmove camera at time `t`.
    CameraSolver {
        solved: SolvedCameraClip,
        time: f32,
        weight: f32,
    },
    /// Follow a 2D track reconstructed to a 3D look-at (or use solved point).
    FollowTrack {
        data: FollowTrackData,
        time: f32,
        weight: f32,
    },
}

impl CameraConstraint {
    pub fn weight(&self) -> f32 {
        let w = match self {
            Self::Aim { weight, .. }
            | Self::TrackTo { weight, .. }
            | Self::DampedTrack { weight, .. }
            | Self::LockedTrack { weight, .. }
            | Self::Parent { weight, .. }
            | Self::ChildOf { weight, .. }
            | Self::PathFollow { weight, .. }
            | Self::FollowPath { weight, .. }
            | Self::ClampTo { weight, .. }
            | Self::Floor { weight, .. }
            | Self::LimitLocation { weight, .. }
            | Self::LimitRotation { weight, .. }
            | Self::CopyTransforms { weight, .. }
            | Self::Action { weight, .. }
            | Self::UpVector { weight, .. }
            | Self::CameraSolver { weight, .. }
            | Self::FollowTrack { weight, .. } => *weight,
        };
        w.clamp(0.0, 1.0)
    }

    fn blend(base: CameraPose, posed: CameraPose, w: f32) -> CameraPose {
        if w <= 0.0 {
            base
        } else if w >= 1.0 {
            posed
        } else {
            base.lerp(base.unwrap_azimuth_toward(posed), w)
        }
    }

    fn parent_frame(
        origin: Vector3,
        forward: Vector3,
        up: Vector3,
        local_eye: Vector3,
        local_target: Vector3,
        base: CameraPose,
    ) -> CameraPose {
        let f = forward.normalize();
        let r = {
            let cross = f.cross(up.normalize());
            if cross.length_sq() < 1e-10 {
                Vector3::RIGHT
            } else {
                cross.normalize()
            }
        };
        let u = r.cross(f).normalize();
        let eye = origin + r * local_eye.x + u * local_eye.y + f * local_eye.z;
        let target = origin + r * local_target.x + u * local_target.y + f * local_target.z;
        let mut posed = CameraPose::from_look_at(eye, target, base.fov);
        posed.focus_distance = base.focus_distance;
        posed.aperture = base.aperture;
        posed
    }

    fn apply(&self, base: CameraPose, dt: f32) -> CameraPose {
        match self {
            Self::Aim {
                target,
                hold_eye,
                weight,
            } => {
                let aimed = if *hold_eye {
                    let mut p = CameraPose::from_look_at(base.eye(), *target, base.fov);
                    p.roll = base.roll;
                    p.focus_distance = base.focus_distance;
                    p.aperture = base.aperture;
                    p
                } else {
                    let mut p = base;
                    p.target = *target;
                    p
                };
                Self::blend(base, aimed, weight.clamp(0.0, 1.0))
            }
            Self::TrackTo {
                target,
                track_axis: _,
                up,
                weight,
            } => {
                // Minus-Z track (camera look) is the threers default.
                let mut posed = CameraPose::from_look_at(base.eye(), *target, base.fov);
                posed.focus_distance = base.focus_distance;
                posed.aperture = base.aperture;
                // Align roll from up hint.
                let stack_up = CameraConstraint::UpVector {
                    up: *up,
                    weight: 1.0,
                };
                posed = stack_up.apply(posed, 0.0);
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::DampedTrack {
                target,
                influence,
                weight,
                current,
            } => {
                let _ = (influence, current, dt);
                // Stateless approximation: blend target toward goal by influence*dt.
                let mut aimed_target = *current;
                let k = (influence.max(0.0) * dt.max(0.0)).clamp(0.0, 1.0);
                aimed_target = aimed_target.lerp(*target, k);
                let mut posed = CameraPose::from_look_at(base.eye(), aimed_target, base.fov);
                posed.focus_distance = base.focus_distance;
                posed.aperture = base.aperture;
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::LockedTrack {
                target, weight, ..
            } => {
                // Lock elevation: only yaw the look-at toward the target on XZ.
                let eye = base.eye();
                let flat = Vector3::new(target.x, eye.y, target.z);
                let mut posed = CameraPose::from_look_at(eye, flat, base.fov);
                posed.elevation = base.elevation;
                posed.radius = base.radius;
                posed.target = Vector3::new(target.x, base.target.y, target.z);
                posed = CameraPose::new(posed.target, posed.azimuth, base.elevation, base.radius);
                posed.fov = base.fov;
                posed.roll = base.roll;
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::Parent {
                origin,
                forward,
                up,
                local_eye,
                local_target,
                weight,
            }
            | Self::ChildOf {
                origin,
                forward,
                up,
                local_eye,
                local_target,
                weight,
                ..
            } => {
                let posed =
                    Self::parent_frame(*origin, *forward, *up, *local_eye, *local_target, base);
                let posed = if let Self::ChildOf { loc, rot, .. } = self {
                    let mut p = base;
                    if *loc {
                        p.target = posed.target;
                        p.radius = posed.radius;
                        p.azimuth = posed.azimuth;
                        p.elevation = posed.elevation;
                    }
                    if *rot {
                        p.roll = posed.roll;
                        if !*loc {
                            // orientation-only: keep eye, retarget
                            p = CameraPose::from_look_at(base.eye(), posed.target, base.fov);
                            p.roll = posed.roll;
                        }
                    }
                    p
                } else {
                    posed
                };
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::PathFollow {
                path,
                percent,
                weight,
            } => {
                let sampled = path.sample_pose(percent.clamp(0.0, 1.0), base);
                Self::blend(base, sampled, weight.clamp(0.0, 1.0))
            }
            Self::FollowPath {
                path,
                percent,
                follow_curve,
                weight,
            } => {
                let sampled = path.sample_pose(percent.clamp(0.0, 1.0), base, *follow_curve);
                Self::blend(base, sampled, weight.clamp(0.0, 1.0))
            }
            Self::ClampTo { path, weight } => {
                let eye = base.eye();
                // Nearest sample on path.
                let mut best = path.sample_eye(0.0);
                let mut best_d = (best - eye).length_sq();
                for i in 1..=32 {
                    let p = path.sample_eye(i as f32 / 32.0);
                    let d = (p - eye).length_sq();
                    if d < best_d {
                        best_d = d;
                        best = p;
                    }
                }
                let mut posed = CameraPose::from_look_at(best, base.target, base.fov);
                posed.roll = base.roll;
                posed.focus_distance = base.focus_distance;
                posed.aperture = base.aperture;
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::Floor {
                height,
                normal,
                weight,
            } => {
                let n = normal.normalize();
                let mut eye = base.eye();
                let plane_d = n.dot(Vector3::new(0.0, *height, 0.0));
                let dist = n.dot(eye) - plane_d;
                if dist < 0.0 {
                    eye = eye - n * dist;
                }
                let mut posed = CameraPose::from_look_at(eye, base.target, base.fov);
                posed.roll = base.roll;
                posed.focus_distance = base.focus_distance;
                posed.aperture = base.aperture;
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::LimitLocation { min, max, weight } => {
                let eye = base.eye();
                let clamped = Vector3::new(
                    eye.x.clamp(min.x, max.x),
                    eye.y.clamp(min.y, max.y),
                    eye.z.clamp(min.z, max.z),
                );
                let mut posed = CameraPose::from_look_at(clamped, base.target, base.fov);
                posed.roll = base.roll;
                posed.focus_distance = base.focus_distance;
                posed.aperture = base.aperture;
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::LimitRotation {
                min_roll,
                max_roll,
                min_elevation,
                max_elevation,
                weight,
            } => {
                let mut posed = base;
                posed.roll = base.roll.clamp(*min_roll, *max_roll);
                posed.elevation = base.elevation.clamp(*min_elevation, *max_elevation);
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::CopyTransforms {
                source,
                loc,
                rot,
                lens,
                weight,
            } => {
                let mut posed = base;
                if *loc {
                    posed.target = source.target;
                    posed.azimuth = source.azimuth;
                    posed.elevation = source.elevation;
                    posed.radius = source.radius;
                }
                if *rot {
                    posed.roll = source.roll;
                }
                if *lens {
                    posed.fov = source.fov;
                    posed.shift_x = source.shift_x;
                    posed.shift_y = source.shift_y;
                    posed.focus_distance = source.focus_distance;
                    posed.aperture = source.aperture;
                    posed.f_stop = source.f_stop;
                }
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::Action { clip, time, weight } => {
                let posed = clip.sample(*time, base);
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::UpVector { up, weight } => {
                let w = weight.clamp(0.0, 1.0);
                if w <= 0.0 {
                    return base;
                }
                let eye = base.eye();
                let forward = (base.target - eye).normalize();
                if forward.length_sq() < 1e-10 {
                    return base;
                }
                let desired_up = {
                    let side = forward.cross(up.normalize());
                    if side.length_sq() < 1e-10 {
                        return base;
                    }
                    side.normalize().cross(forward).normalize()
                };
                let right = {
                    let r = forward.cross(Vector3::UP);
                    if r.length_sq() < 1e-10 {
                        Vector3::RIGHT
                    } else {
                        r.normalize()
                    }
                };
                let natural_up = right.cross(forward).normalize();
                let sin = natural_up.cross(desired_up).dot(forward);
                let cos = natural_up.dot(desired_up);
                let roll = sin.atan2(cos);
                let mut out = base;
                out.roll = base.roll + (roll - base.roll) * w;
                out
            }
            Self::CameraSolver {
                solved,
                time,
                weight,
            } => {
                let posed = solved.sample(*time).unwrap_or(base);
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
            Self::FollowTrack { data, time, weight } => {
                let posed = data.apply(base, *time);
                Self::blend(base, posed, weight.clamp(0.0, 1.0))
            }
        }
    }
}

/// Ordered constraint list, evaluated bottom-up.
#[derive(Debug, Clone, Default)]
pub struct ConstraintStack {
    constraints: Vec<CameraConstraint>,
    /// Last dt for damped constraints.
    pub dt: f32,
}

impl ConstraintStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, c: CameraConstraint) -> usize {
        self.constraints.push(c);
        self.constraints.len() - 1
    }

    pub fn clear(&mut self) {
        self.constraints.clear();
    }

    pub fn constraints(&self) -> &[CameraConstraint] {
        &self.constraints
    }

    pub fn constraints_mut(&mut self) -> &mut [CameraConstraint] {
        &mut self.constraints
    }

    pub fn evaluate(&self, base: CameraPose) -> CameraPose {
        let mut pose = base;
        for c in &self.constraints {
            pose = c.apply(pose, self.dt);
        }
        // Advance damped-track working state.
        pose
    }

    /// Evaluate and update stateful damped tracks.
    pub fn evaluate_mut(&mut self, base: CameraPose, dt: f32) -> CameraPose {
        self.dt = dt;
        let mut pose = base;
        for c in &mut self.constraints {
            if let CameraConstraint::DampedTrack {
                target,
                influence,
                current,
                ..
            } = c
            {
                let k = (influence.max(0.0) * dt.max(0.0)).clamp(0.0, 1.0);
                *current = current.lerp(*target, k);
            }
            pose = c.apply(pose, dt);
        }
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aim_hold_eye_retargets() {
        let base = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
        let eye = base.eye();
        let mut stack = ConstraintStack::new();
        stack.push(CameraConstraint::Aim {
            target: Vector3::new(2.0, 0.0, 0.0),
            hold_eye: true,
            weight: 1.0,
        });
        let out = stack.evaluate(base);
        assert!((out.eye() - eye).length() < 1e-3);
    }

    #[test]
    fn floor_lifts_eye() {
        let base = CameraPose::from_look_at(
            Vector3::new(0.0, -1.0, 5.0),
            Vector3::ZERO,
            50.0_f32.to_radians(),
        );
        let mut stack = ConstraintStack::new();
        stack.push(CameraConstraint::Floor {
            height: 0.0,
            normal: Vector3::UP,
            weight: 1.0,
        });
        let out = stack.evaluate(base);
        assert!(out.eye().y >= -1e-4, "{}", out.eye().y);
    }

    #[test]
    fn limit_location_clamps() {
        let base =
            CameraPose::from_look_at(Vector3::new(10.0, 0.0, 0.0), Vector3::ZERO, 0.8);
        let mut stack = ConstraintStack::new();
        stack.push(CameraConstraint::LimitLocation {
            min: Vector3::new(-1.0, -1.0, -1.0),
            max: Vector3::new(1.0, 1.0, 1.0),
            weight: 1.0,
        });
        let out = stack.evaluate(base);
        assert!(out.eye().x <= 1.0 + 1e-3);
    }
}
