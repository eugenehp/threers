use super::interpolant::{find_segment, Interpolation};
use crate::core::ObjectId;
use crate::math::{Color, Quaternion, Vector3};

/// Which property of an Object3D this track drives. Mirrors three.js's
/// property bindings (`position`, `quaternion`, `scale`, plus material color).
#[derive(Debug, Clone, Copy)]
pub enum TrackTarget {
    Position,
    Quaternion,
    Scale,
    /// Mesh material color. No-op if the targeted object isn't a Mesh or its
    /// material doesn't carry a base color.
    Color,
    /// Morph target weight at `index`.
    MorphWeight { index: usize },
    /// Free scalar — caller decides what to do with it.
    Scalar,
}

#[derive(Debug, Clone)]
pub enum TrackValues {
    Vector(Vec<Vector3>),
    Quaternion(Vec<Quaternion>),
    Color(Vec<Color>),
    Scalar(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct KeyframeTrack {
    pub object: ObjectId,
    pub target: TrackTarget,
    pub times: Vec<f32>,
    pub values: TrackValues,
    pub interpolation: Interpolation,
}

impl KeyframeTrack {
    pub fn vector(
        object: ObjectId,
        target: TrackTarget,
        times: Vec<f32>,
        values: Vec<Vector3>,
    ) -> Self {
        Self {
            object,
            target,
            times,
            values: TrackValues::Vector(values),
            interpolation: Interpolation::Linear,
        }
    }
    pub fn quaternion(
        object: ObjectId,
        target: TrackTarget,
        times: Vec<f32>,
        values: Vec<Quaternion>,
    ) -> Self {
        Self {
            object,
            target,
            times,
            values: TrackValues::Quaternion(values),
            interpolation: Interpolation::Linear,
        }
    }
    pub fn color(
        object: ObjectId,
        target: TrackTarget,
        times: Vec<f32>,
        values: Vec<Color>,
    ) -> Self {
        Self {
            object,
            target,
            times,
            values: TrackValues::Color(values),
            interpolation: Interpolation::Linear,
        }
    }
    pub fn scalar(
        object: ObjectId,
        target: TrackTarget,
        times: Vec<f32>,
        values: Vec<f32>,
    ) -> Self {
        Self {
            object,
            target,
            times,
            values: TrackValues::Scalar(values),
            interpolation: Interpolation::Linear,
        }
    }

    pub fn sample_vector(&self, t: f32) -> Option<Vector3> {
        let TrackValues::Vector(vals) = &self.values else {
            return None;
        };
        let (i, a) = find_segment(&self.times, t);
        if a == 0.0 || matches!(self.interpolation, Interpolation::Step) {
            return Some(vals[i.min(vals.len() - 1)]);
        }
        Some(vals[i].lerp(vals[(i + 1).min(vals.len() - 1)], a))
    }

    pub fn sample_quaternion(&self, t: f32) -> Option<Quaternion> {
        let TrackValues::Quaternion(vals) = &self.values else {
            return None;
        };
        let (i, a) = find_segment(&self.times, t);
        if a == 0.0 || matches!(self.interpolation, Interpolation::Step) {
            return Some(vals[i.min(vals.len() - 1)]);
        }
        Some(vals[i].slerp(vals[(i + 1).min(vals.len() - 1)], a))
    }

    pub fn sample_color(&self, t: f32) -> Option<Color> {
        let TrackValues::Color(vals) = &self.values else {
            return None;
        };
        let (i, a) = find_segment(&self.times, t);
        if a == 0.0 || matches!(self.interpolation, Interpolation::Step) {
            return Some(vals[i.min(vals.len() - 1)]);
        }
        let a_c = vals[i];
        let b_c = vals[(i + 1).min(vals.len() - 1)];
        Some(Color::new(
            a_c.r + (b_c.r - a_c.r) * a,
            a_c.g + (b_c.g - a_c.g) * a,
            a_c.b + (b_c.b - a_c.b) * a,
        ))
    }

    pub fn sample_scalar(&self, t: f32) -> Option<f32> {
        let TrackValues::Scalar(vals) = &self.values else {
            return None;
        };
        let (i, a) = find_segment(&self.times, t);
        if a == 0.0 || matches!(self.interpolation, Interpolation::Step) {
            return Some(vals[i.min(vals.len() - 1)]);
        }
        let a_v = vals[i];
        let b_v = vals[(i + 1).min(vals.len() - 1)];
        Some(a_v + (b_v - a_v) * a)
    }
}
