//! three.js r165 `KeyframeTrack` and its typed subclasses (`NumberKeyframeTrack`,
//! `VectorKeyframeTrack`, `QuaternionKeyframeTrack`, `ColorKeyframeTrack`,
//! `BooleanKeyframeTrack`, `StringKeyframeTrack`), as one type tagged with its
//! [`ValueType`].
//!
//! Times and numeric values are `f32`, as three.js's `Float32Array`s are, and
//! every in-place edit (`shift`, `scale`) rounds through `f32` the same way a
//! typed-array store does. Boolean and string tracks keep their values as
//! three.js keeps them in plain arrays.

use super::interpolant::{Interpolant, InterpolantKind};

/// `InterpolateDiscrete`.
pub const INTERPOLATE_DISCRETE: u32 = 2300;
/// `InterpolateLinear`.
pub const INTERPOLATE_LINEAR: u32 = 2301;
/// `InterpolateSmooth`.
pub const INTERPOLATE_SMOOTH: u32 = 2302;

/// A track's `ValueTypeName` — which subclass it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// `NumberKeyframeTrack` (`'number'`).
    Number,
    /// `VectorKeyframeTrack` (`'vector'`).
    Vector,
    /// `QuaternionKeyframeTrack` (`'quaternion'`).
    Quaternion,
    /// `ColorKeyframeTrack` (`'color'`).
    Color,
    /// `BooleanKeyframeTrack` (`'bool'`).
    Bool,
    /// `StringKeyframeTrack` (`'string'`).
    String,
}

impl ValueType {
    /// three.js `ValueTypeName`.
    pub fn name(self) -> &'static str {
        match self {
            ValueType::Number => "number",
            ValueType::Vector => "vector",
            ValueType::Quaternion => "quaternion",
            ValueType::Color => "color",
            ValueType::Bool => "bool",
            ValueType::String => "string",
        }
    }

    /// `getTrackTypeForValueTypeName` (case-insensitive, with its aliases).
    pub fn from_name(name: &str) -> Option<ValueType> {
        Some(match name.to_ascii_lowercase().as_str() {
            "scalar" | "double" | "float" | "number" | "integer" => ValueType::Number,
            "vector" | "vector2" | "vector3" | "vector4" => ValueType::Vector,
            "color" => ValueType::Color,
            "quaternion" => ValueType::Quaternion,
            "bool" | "boolean" => ValueType::Bool,
            "string" => ValueType::String,
            _ => return None,
        })
    }

    /// `DefaultInterpolation`.
    pub fn default_interpolation(self) -> Interpolation {
        match self {
            ValueType::Bool | ValueType::String => Interpolation::Discrete,
            _ => Interpolation::Linear,
        }
    }

    /// Whether the subclass has a factory method for `interpolation`
    /// (Boolean and String tracks only interpolate discretely; quaternions
    /// have no smooth interpolant).
    pub fn supports(self, interpolation: Interpolation) -> bool {
        match (self, interpolation) {
            (ValueType::Bool | ValueType::String, Interpolation::Discrete) => true,
            (ValueType::Bool | ValueType::String, _) => false,
            (ValueType::Quaternion, Interpolation::Smooth) => false,
            _ => true,
        }
    }

    /// Whether the values are numbers (as opposed to booleans or strings).
    pub fn is_numeric(self) -> bool {
        !matches!(self, ValueType::Bool | ValueType::String)
    }
}

/// How a track interpolates: three.js's three modes, plus GLTFLoader's
/// cubic spline (a replaced `createInterpolant` whose function carries
/// `isInterpolantFactoryMethodGLTFCubicSpline`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interpolation {
    Discrete,
    Linear,
    Smooth,
    GltfCubicSpline,
}

impl Interpolation {
    /// The three.js constant (`InterpolateDiscrete` …); glTF's cubic spline
    /// has none, and reports `undefined` from `getInterpolation()`.
    pub fn constant(self) -> Option<u32> {
        match self {
            Interpolation::Discrete => Some(INTERPOLATE_DISCRETE),
            Interpolation::Linear => Some(INTERPOLATE_LINEAR),
            Interpolation::Smooth => Some(INTERPOLATE_SMOOTH),
            Interpolation::GltfCubicSpline => None,
        }
    }

    pub fn from_constant(c: u32) -> Option<Interpolation> {
        match c {
            INTERPOLATE_DISCRETE => Some(Interpolation::Discrete),
            INTERPOLATE_LINEAR => Some(Interpolation::Linear),
            INTERPOLATE_SMOOTH => Some(Interpolation::Smooth),
            _ => None,
        }
    }
}

/// A track's values, typed as three.js stores them.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackValues {
    /// A `Float32Array`.
    Numeric(Vec<f32>),
    /// `BooleanKeyframeTrack`'s plain array.
    Bool(Vec<bool>),
    /// `StringKeyframeTrack`'s plain array.
    String(Vec<String>),
}

impl TrackValues {
    pub fn len(&self) -> usize {
        match self {
            TrackValues::Numeric(v) => v.len(),
            TrackValues::Bool(v) => v.len(),
            TrackValues::String(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The numeric values (booleans as 0/1, strings as none).
    pub fn numeric(&self) -> Option<&[f32]> {
        match self {
            TrackValues::Numeric(v) => Some(v),
            _ => None,
        }
    }
    fn slice(&self, from: usize, to: usize) -> TrackValues {
        let to = to.min(self.len());
        let from = from.min(to);
        match self {
            TrackValues::Numeric(v) => TrackValues::Numeric(v[from..to].to_vec()),
            TrackValues::Bool(v) => TrackValues::Bool(v[from..to].to_vec()),
            TrackValues::String(v) => TrackValues::String(v[from..to].to_vec()),
        }
    }
    fn same(&self, a: usize, b: usize) -> bool {
        match self {
            // `!==` on numbers: NaN differs from itself, 0 equals -0.
            TrackValues::Numeric(v) => v.get(a) == v.get(b) && v.get(a).is_some(),
            TrackValues::Bool(v) => v.get(a) == v.get(b),
            TrackValues::String(v) => v.get(a) == v.get(b),
        }
    }
    fn copy_within(&mut self, src: usize, dst: usize) {
        match self {
            TrackValues::Numeric(v) => v[dst] = v[src],
            TrackValues::Bool(v) => v[dst] = v[src],
            TrackValues::String(v) => v[dst] = v[src].clone(),
        }
    }
}

/// Why [`KeyframeTrack::new`] refused a track (three.js throws).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackError {
    /// `THREE.KeyframeTrack: no keyframes in track named …`
    NoKeyframes(String),
    /// Values of the wrong kind for the track type (numbers on a string track).
    ValueKind(String),
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::NoKeyframes(n) => {
                write!(f, "THREE.KeyframeTrack: no keyframes in track named {n}")
            }
            TrackError::ValueKind(n) => write!(
                f,
                "THREE.KeyframeTrack: values of the wrong kind for track {n}"
            ),
        }
    }
}

impl std::error::Error for TrackError {}

/// A procedural modifier on a track, Blender's F-curve modifiers — not a
/// three.js feature, but the shim's `THREE.TrackModifier` (`track.modifiers`),
/// applied when the mixer samples the track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackModifier {
    /// Hash noise added to every value.
    Noise {
        amplitude: f64,
        frequency: f64,
        seed: f64,
    },
    /// Repeat the keys outside their range (0: hold the end instead).
    Cycles { before: f64, after: f64 },
    /// Sample at whole multiples of `step_size`.
    Stepped { step_size: f64 },
    /// Clamp every value.
    Limit { min: f64, max: f64 },
}

impl TrackModifier {
    /// The time a track with `modifiers` and keys up to `duration` samples at `t`.
    pub fn remap_time(modifiers: &[TrackModifier], t: f64, duration: f64) -> f64 {
        let mut time = t;
        for m in modifiers {
            match *m {
                TrackModifier::Cycles { before, after } if duration > 0.0 => {
                    if time < 0.0 {
                        time = if before == 0.0 {
                            0.0
                        } else {
                            ((time % duration) + duration) % duration
                        };
                    } else if time > duration {
                        time = if after == 0.0 {
                            duration
                        } else {
                            time % duration
                        };
                    }
                }
                TrackModifier::Stepped { step_size } => {
                    let s = step_size.max(1e-6);
                    time = (time / s).floor() * s;
                }
                _ => {}
            }
        }
        time
    }

    /// Apply the value modifiers to a sample taken at (remapped) time `t`.
    pub fn apply_value(modifiers: &[TrackModifier], values: &mut [f64], t: f64) {
        for m in modifiers {
            match *m {
                TrackModifier::Noise {
                    amplitude,
                    frequency,
                    seed,
                } => {
                    for (i, v) in values.iter_mut().enumerate() {
                        let n = ((t * frequency + seed + i as f64 * 19.19) * 12.9898).sin()
                            * 43758.5453;
                        let f = n - n.floor();
                        *v += (f * 2.0 - 1.0) * amplitude;
                    }
                }
                TrackModifier::Limit { min, max } => {
                    for v in values.iter_mut() {
                        *v = max.min(min.max(*v));
                    }
                }
                _ => {}
            }
        }
    }
}

/// three.js `KeyframeTrack`.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeTrack {
    /// The binding path, `nodeName.property` (see `PropertyBinding`).
    pub name: String,
    pub value_type: ValueType,
    pub times: Vec<f32>,
    pub values: TrackValues,
    interpolation: Interpolation,
    /// Procedural modifiers applied when sampled (none in three.js).
    pub modifiers: Vec<TrackModifier>,
}

impl KeyframeTrack {
    /// `new XxxKeyframeTrack(name, times, values, interpolation)`.
    /// `interpolation` `None` is the type's default.
    pub fn new(
        value_type: ValueType,
        name: impl Into<String>,
        times: Vec<f32>,
        values: TrackValues,
        interpolation: Option<Interpolation>,
    ) -> Result<Self, TrackError> {
        let name = name.into();
        if times.is_empty() {
            return Err(TrackError::NoKeyframes(name));
        }
        let ok = match (&values, value_type) {
            (TrackValues::Bool(_), ValueType::Bool) => true,
            (TrackValues::String(_), ValueType::String) => true,
            (TrackValues::Numeric(_), t) => t.is_numeric(),
            _ => false,
        };
        if !ok {
            return Err(TrackError::ValueKind(name));
        }
        let mut track = Self {
            name,
            value_type,
            times,
            values,
            interpolation: value_type.default_interpolation(),
            modifiers: Vec::new(),
        };
        // `setInterpolation( interpolation || this.DefaultInterpolation )`;
        // on construction an unsupported mode falls back to the default.
        let wanted = interpolation.unwrap_or(value_type.default_interpolation());
        if value_type.supports(wanted) {
            track.interpolation = wanted;
        } else {
            log::warn!(
                "THREE.KeyframeTrack: unsupported interpolation for {} keyframe track named {}",
                value_type.name(),
                track.name
            );
        }
        Ok(track)
    }

    /// `new NumberKeyframeTrack(name, times, values)`.
    pub fn number(name: impl Into<String>, times: Vec<f32>, values: Vec<f32>) -> Self {
        Self::numeric(ValueType::Number, name, times, values)
    }
    /// `new VectorKeyframeTrack(name, times, values)`.
    pub fn vector(name: impl Into<String>, times: Vec<f32>, values: Vec<f32>) -> Self {
        Self::numeric(ValueType::Vector, name, times, values)
    }
    /// `new QuaternionKeyframeTrack(name, times, values)`.
    pub fn quaternion(name: impl Into<String>, times: Vec<f32>, values: Vec<f32>) -> Self {
        Self::numeric(ValueType::Quaternion, name, times, values)
    }
    /// `new ColorKeyframeTrack(name, times, values)`.
    pub fn color(name: impl Into<String>, times: Vec<f32>, values: Vec<f32>) -> Self {
        Self::numeric(ValueType::Color, name, times, values)
    }
    /// `new BooleanKeyframeTrack(name, times, values)`.
    pub fn boolean(name: impl Into<String>, times: Vec<f32>, values: Vec<bool>) -> Self {
        Self {
            name: name.into(),
            value_type: ValueType::Bool,
            times,
            values: TrackValues::Bool(values),
            interpolation: Interpolation::Discrete,
            modifiers: Vec::new(),
        }
    }
    /// `new StringKeyframeTrack(name, times, values)`.
    pub fn string(name: impl Into<String>, times: Vec<f32>, values: Vec<String>) -> Self {
        Self {
            name: name.into(),
            value_type: ValueType::String,
            times,
            values: TrackValues::String(values),
            interpolation: Interpolation::Discrete,
            modifiers: Vec::new(),
        }
    }

    fn numeric(
        value_type: ValueType,
        name: impl Into<String>,
        times: Vec<f32>,
        values: Vec<f32>,
    ) -> Self {
        Self {
            name: name.into(),
            value_type,
            times,
            values: TrackValues::Numeric(values),
            interpolation: Interpolation::Linear,
            modifiers: Vec::new(),
        }
    }

    /// Builder: `setInterpolation(mode)`.
    pub fn with_interpolation(mut self, interpolation: Interpolation) -> Self {
        self.set_interpolation(interpolation);
        self
    }

    /// `setInterpolation(mode)`: an unsupported mode warns and leaves the
    /// track as it was (the constructor is the one place it falls back).
    pub fn set_interpolation(&mut self, interpolation: Interpolation) -> &mut Self {
        if self.value_type.supports(interpolation)
            || interpolation == Interpolation::GltfCubicSpline
        {
            self.interpolation = interpolation;
        } else {
            log::warn!(
                "THREE.KeyframeTrack: unsupported interpolation for {} keyframe track named {}",
                self.value_type.name(),
                self.name
            );
        }
        self
    }

    /// `getInterpolation()`.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// `getValueSize()`: values per keyframe (three's is a float; this one
    /// rounds down, and [`validate`](Self::validate) reports a remainder).
    pub fn value_size(&self) -> usize {
        if self.times.is_empty() {
            0
        } else {
            self.values.len() / self.times.len()
        }
    }

    /// `getValueSize()` as three.js computes it (possibly fractional).
    pub fn value_size_f64(&self) -> f64 {
        self.values.len() as f64 / self.times.len() as f64
    }

    /// The interpolant `createInterpolant()` would build for this track.
    pub fn create_interpolant(&self) -> Interpolant {
        let size = self.value_size();
        let kind = match self.interpolation {
            Interpolation::Discrete => InterpolantKind::Discrete,
            Interpolation::Linear => {
                if self.value_type == ValueType::Quaternion {
                    InterpolantKind::QuaternionLinear
                } else {
                    InterpolantKind::Linear
                }
            }
            Interpolation::Smooth => InterpolantKind::Cubic,
            Interpolation::GltfCubicSpline => {
                return Interpolant::new(
                    if self.value_type == ValueType::Quaternion {
                        InterpolantKind::GltfCubicSplineQuaternion
                    } else {
                        InterpolantKind::GltfCubicSpline
                    },
                    size / 3,
                );
            }
        };
        Interpolant::new(kind, size)
    }

    /// `shift(timeOffset)`: move every key (stored back as `f32`).
    pub fn shift(&mut self, time_offset: f64) -> &mut Self {
        if time_offset != 0.0 {
            for t in &mut self.times {
                *t = (*t as f64 + time_offset) as f32;
            }
        }
        self
    }

    /// `scale(timeScale)`: multiply every key time (stored back as `f32`).
    pub fn scale(&mut self, time_scale: f64) -> &mut Self {
        if time_scale != 1.0 {
            for t in &mut self.times {
                *t = (*t as f64 * time_scale) as f32;
            }
        }
        self
    }

    /// `trim(startTime, endTime)`: drop keys outside the range, keeping at
    /// least one.
    pub fn trim(&mut self, start_time: f64, end_time: f64) -> &mut Self {
        let n_keys = self.times.len() as isize;
        let mut from: isize = 0;
        let mut to: isize = n_keys - 1;
        while from != n_keys && (self.times[from as usize] as f64) < start_time {
            from += 1;
        }
        while to != -1 && (self.times[to as usize] as f64) > end_time {
            to -= 1;
        }
        to += 1; // inclusive -> exclusive bound
        if from != 0 || to != n_keys {
            // empty tracks are forbidden, so keep at least one keyframe
            if from >= to {
                to = to.max(1);
                from = to - 1;
            }
            let stride = self.value_size();
            let (f, t) = (from as usize, to as usize);
            self.times = self.times[f.min(self.times.len())..t.min(self.times.len())].to_vec();
            self.values = self.values.slice(f * stride, t * stride);
        }
        self
    }

    /// `validate()`: whole value size, keys present, times in order and
    /// numbers, no NaN values. Problems are logged as three.js logs them.
    pub fn validate(&self) -> bool {
        let mut valid = true;
        let value_size = self.value_size_f64();
        if value_size - value_size.floor() != 0.0 {
            log::error!(
                "THREE.KeyframeTrack: Invalid value size in track. {}",
                self.name
            );
            valid = false;
        }
        let n_keys = self.times.len();
        if n_keys == 0 {
            log::error!("THREE.KeyframeTrack: Track is empty. {}", self.name);
            valid = false;
        }
        let mut prev_time: Option<f32> = None;
        for (i, &curr_time) in self.times.iter().enumerate() {
            if curr_time.is_nan() {
                log::error!(
                    "THREE.KeyframeTrack: Time is not a valid number. {} {i}",
                    self.name
                );
                valid = false;
                break;
            }
            if let Some(p) = prev_time {
                if p > curr_time {
                    log::error!("THREE.KeyframeTrack: Out of order keys. {} {i}", self.name);
                    valid = false;
                    break;
                }
            }
            prev_time = Some(curr_time);
        }
        if let TrackValues::Numeric(values) = &self.values {
            for (i, v) in values.iter().enumerate() {
                if v.is_nan() {
                    log::error!(
                        "THREE.KeyframeTrack: Value is not a valid number. {} {i}",
                        self.name
                    );
                    valid = false;
                    break;
                }
            }
        }
        valid
    }

    /// `optimize()`: drop keys equal to both neighbours (and keys at the same
    /// time as the next), as morph-target sequences are full of.
    pub fn optimize(&mut self) -> &mut Self {
        if self.times.is_empty() {
            return self;
        }
        let mut times = self.times.clone();
        let mut values = self.values.clone();
        let stride = self.value_size();
        let smooth = self.interpolation == Interpolation::Smooth;
        let last_index = times.len() as isize - 1;
        let mut write_index: usize = 1;
        let mut i: isize = 1;
        while i < last_index {
            let iu = i as usize;
            let mut keep = false;
            let time = times[iu];
            let time_next = times[iu + 1];
            // remove adjacent keyframes scheduled at the same time
            if time != time_next && (iu != 1 || time != times[0]) {
                if !smooth {
                    // remove unnecessary keyframes same as their neighbors
                    let offset = iu * stride;
                    let offset_p = offset - stride;
                    let offset_n = offset + stride;
                    for j in 0..stride {
                        if !values.same(offset + j, offset_p + j)
                            || !values.same(offset + j, offset_n + j)
                        {
                            keep = true;
                            break;
                        }
                    }
                } else {
                    keep = true;
                }
            }
            // in-place compaction
            if keep {
                if iu != write_index {
                    times[write_index] = times[iu];
                    let read_offset = iu * stride;
                    let write_offset = write_index * stride;
                    for j in 0..stride {
                        values.copy_within(read_offset + j, write_offset + j);
                    }
                }
                write_index += 1;
            }
            i += 1;
        }
        // flush last keyframe (compaction looks ahead)
        if last_index > 0 {
            let lu = last_index as usize;
            times[write_index] = times[lu];
            let read_offset = lu * stride;
            let write_offset = write_index * stride;
            for j in 0..stride {
                values.copy_within(read_offset + j, write_offset + j);
            }
            write_index += 1;
        }
        if write_index != times.len() {
            self.times = times[..write_index].to_vec();
            self.values = values.slice(0, write_index * stride);
        } else {
            self.times = times;
            self.values = values;
        }
        self
    }

    /// The value at key `i` (numeric tracks), `value_size` numbers.
    pub fn value_at(&self, i: usize) -> Option<&[f32]> {
        let s = self.value_size();
        self.values
            .numeric()
            .and_then(|v| v.get(i * s..(i + 1) * s))
    }
}
