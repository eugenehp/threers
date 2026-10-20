//! three.js r165 `Interpolant` and its subclasses — `DiscreteInterpolant`,
//! `LinearInterpolant`, `QuaternionLinearInterpolant`, `CubicInterpolant` —
//! plus GLTFLoader's `GLTFCubicSplineInterpolant` (and its quaternion
//! variant), ported line for line.
//!
//! The seek is three.js's: an O(1) linear scan from the cached interval,
//! falling back to binary search, with `undefined` past either end of the
//! times read as "no sample". The cached index and the cubic interpolant's
//! slope weights are state, exactly as in three.js, which is why an action's
//! interpolants live as long as the action and why the shared ending settings
//! only take effect when the interval changes.
//!
//! Data is not owned here: [`Interpolant::evaluate`] takes the parameter
//! positions (`f32`, a track's `times`) and sample values (`f32`) and writes
//! into any [`ResultBuffer`] — the `f64` mixer buffer when an action drives
//! it, a `f32` buffer when used on its own (three.js allocates
//! `new sampleValues.constructor(size)`, a `Float32Array`).

/// `ZeroCurvatureEnding`: f''(t) = 0 at the ends (a natural spline).
pub const ZERO_CURVATURE_ENDING: u32 = 2400;
/// `ZeroSlopeEnding`: f'(t) = 0 at the ends.
pub const ZERO_SLOPE_ENDING: u32 = 2401;
/// `WrapAroundEnding`: the curve continues from the other end.
pub const WRAP_AROUND_ENDING: u32 = 2402;

/// What an interpolant does between two samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolantKind {
    /// `DiscreteInterpolant`: the sample at or before `t`.
    Discrete,
    /// `LinearInterpolant`.
    Linear,
    /// `QuaternionLinearInterpolant`: `Quaternion.slerpFlat` per 4 values.
    QuaternionLinear,
    /// `CubicInterpolant`: Hermite with slopes from the neighbours.
    Cubic,
    /// GLTFLoader's `GLTFCubicSplineInterpolant`: samples are
    /// `[inTangent, value, outTangent]` triples.
    GltfCubicSpline,
    /// `GLTFCubicSplineQuaternionInterpolant`: the above, normalised.
    GltfCubicSplineQuaternion,
}

/// three.js `interpolant.settings` (`endingStart` / `endingEnd`). An action
/// shares one between all its interpolants and flips it as it loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpolantSettings {
    pub ending_start: u32,
    pub ending_end: u32,
}

impl Default for InterpolantSettings {
    fn default() -> Self {
        Self {
            ending_start: ZERO_CURVATURE_ENDING,
            ending_end: ZERO_CURVATURE_ENDING,
        }
    }
}

/// Where an interpolant writes: three.js's `resultBuffer`, any numeric array.
pub trait ResultBuffer {
    fn get(&self, i: usize) -> f64;
    fn set(&mut self, i: usize, v: f64);
}

impl ResultBuffer for [f64] {
    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.get(i).copied().unwrap_or(f64::NAN)
    }
    #[inline]
    fn set(&mut self, i: usize, v: f64) {
        if let Some(s) = self.get_mut(i) {
            *s = v;
        }
    }
}

impl ResultBuffer for [f32] {
    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.get(i).map(|&v| v as f64).unwrap_or(f64::NAN)
    }
    #[inline]
    fn set(&mut self, i: usize, v: f64) {
        if let Some(s) = self.get_mut(i) {
            *s = v as f32;
        }
    }
}

/// The outcome of a seek: copy one sample, or interpolate inside an interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seek {
    /// `copySampleValue_(index)`.
    Copy(usize),
    /// `interpolate_(i1, t0, t, t1)`, after `intervalChanged_` when `changed`.
    Interpolate {
        i1: usize,
        t0: f64,
        t1: f64,
        changed: bool,
    },
}

/// One three.js interpolant's state (the data is passed to each call).
#[derive(Debug, Clone)]
pub struct Interpolant {
    pub kind: InterpolantKind,
    /// `valueSize`: values per sample (a third of a glTF cubic track's).
    pub value_size: usize,
    cached_index: usize,
    weight_prev: f64,
    offset_prev: usize,
    weight_next: f64,
    offset_next: usize,
}

#[inline]
fn at(pp: &[f32], i: isize) -> Option<f64> {
    if i >= 0 {
        pp.get(i as usize).map(|&v| v as f64)
    } else {
        None
    }
}
/// JS `t < x` where `x` may be `undefined` (false).
#[inline]
fn lt(t: f64, x: Option<f64>) -> bool {
    matches!(x, Some(v) if t < v)
}
/// JS `t >= x` where `x` may be `undefined` (false).
#[inline]
fn ge(t: f64, x: Option<f64>) -> bool {
    matches!(x, Some(v) if t >= v)
}

impl Interpolant {
    pub fn new(kind: InterpolantKind, value_size: usize) -> Self {
        Self {
            kind,
            value_size,
            cached_index: 0,
            weight_prev: -0.0,
            offset_prev: 0,
            weight_next: -0.0,
            offset_next: 0,
        }
    }

    /// three.js's settings when none are given (`DefaultSettings_`); only the
    /// cubic interpolant reads them.
    pub fn default_settings() -> InterpolantSettings {
        InterpolantSettings::default()
    }

    /// The `_cachedIndex` the next seek starts from.
    pub fn cached_index(&self) -> usize {
        self.cached_index
    }

    /// The interval seek of `Interpolant.evaluate`, updating the cached index.
    pub fn seek(&mut self, pp: &[f32], t: f64) -> Seek {
        let n = pp.len() as isize;
        let mut i1 = self.cached_index as isize;
        let mut t1 = at(pp, i1);
        let mut t0 = at(pp, i1 - 1);

        'validate_interval: {
            'seek: {
                let right: isize;
                'linear_scan: {
                    'forward_scan: {
                        if !lt(t, t1) {
                            let give_up_at = i1 + 2;
                            loop {
                                if t1.is_none() {
                                    if lt(t, t0) {
                                        break 'forward_scan;
                                    }
                                    // after end
                                    i1 = n;
                                    self.cached_index = i1 as usize;
                                    return Seek::Copy((i1 - 1).max(0) as usize);
                                }
                                if i1 == give_up_at {
                                    break; // this loop
                                }
                                t0 = t1;
                                i1 += 1;
                                t1 = at(pp, i1);
                                if lt(t, t1) {
                                    // we have arrived at the sought interval
                                    break 'seek;
                                }
                            }
                            // prepare binary search on the right side of the index
                            right = n;
                            break 'linear_scan;
                        }
                    }
                    if !ge(t, t0) {
                        // looping?
                        let t1global = at(pp, 1);
                        if lt(t, t1global) {
                            i1 = 2; // + 1, using the scan for the details
                            t0 = t1global;
                        }
                        // linear reverse scan
                        let give_up_at = i1 - 2;
                        loop {
                            if t0.is_none() {
                                // before start
                                self.cached_index = 0;
                                return Seek::Copy(0);
                            }
                            if i1 == give_up_at {
                                break; // this loop
                            }
                            t1 = t0;
                            i1 -= 1;
                            t0 = at(pp, i1 - 1);
                            if ge(t, t0) {
                                // we have arrived at the sought interval
                                break 'seek;
                            }
                        }
                        // prepare binary search on the left side of the index
                        right = i1;
                        i1 = 0;
                        break 'linear_scan;
                    }
                    // the interval is valid
                    break 'validate_interval;
                } // linear scan

                // binary search
                let mut right = right;
                while i1 < right {
                    let mid = ((i1 + right) as usize >> 1) as isize;
                    if lt(t, at(pp, mid)) {
                        right = mid;
                    } else {
                        i1 = mid + 1;
                    }
                }
                t1 = at(pp, i1);
                t0 = at(pp, i1 - 1);
                // check boundary cases, again
                if t0.is_none() {
                    self.cached_index = 0;
                    return Seek::Copy(0);
                }
                if t1.is_none() {
                    i1 = n;
                    self.cached_index = i1 as usize;
                    return Seek::Copy((i1 - 1).max(0) as usize);
                }
            } // seek
            self.cached_index = i1 as usize;
            return Seek::Interpolate {
                i1: i1 as usize,
                t0: t0.unwrap_or(f64::NAN),
                t1: t1.unwrap_or(f64::NAN),
                changed: true,
            };
        } // validate_interval
        Seek::Interpolate {
            i1: i1 as usize,
            t0: t0.unwrap_or(f64::NAN),
            t1: t1.unwrap_or(f64::NAN),
            changed: false,
        }
    }

    /// `Interpolant.evaluate(t)`: seek, then copy or interpolate into
    /// `result`. `settings` is `interpolant.settings` (`None`: the defaults).
    pub fn evaluate<R: ResultBuffer + ?Sized>(
        &mut self,
        pp: &[f32],
        values: &[f32],
        t: f64,
        settings: Option<&InterpolantSettings>,
        result: &mut R,
    ) {
        match self.seek(pp, t) {
            Seek::Copy(index) => self.copy_sample_value(values, index, result),
            Seek::Interpolate {
                i1,
                t0,
                t1,
                changed,
            } => {
                if changed {
                    self.interval_changed(pp, i1, t0, t1, settings);
                }
                self.interpolate(values, i1, t0, t, t1, result);
            }
        }
    }

    /// `copySampleValue_(index)`.
    pub fn copy_sample_value<R: ResultBuffer + ?Sized>(
        &self,
        values: &[f32],
        index: usize,
        result: &mut R,
    ) {
        let stride = self.value_size;
        let offset = match self.kind {
            InterpolantKind::GltfCubicSpline | InterpolantKind::GltfCubicSplineQuaternion => {
                index * stride * 3 + stride
            }
            _ => index * stride,
        };
        for i in 0..stride {
            result.set(i, sample(values, offset + i));
        }
    }

    /// `intervalChanged_(i1, t0, t1)` (only the cubic interpolant has one).
    pub fn interval_changed(
        &mut self,
        pp: &[f32],
        i1: usize,
        t0: f64,
        t1: f64,
        settings: Option<&InterpolantSettings>,
    ) {
        if self.kind != InterpolantKind::Cubic {
            return;
        }
        let settings = settings.copied().unwrap_or_default();
        let i1 = i1 as isize;
        let mut i_prev = i1 - 2;
        let mut i_next = i1 + 1;
        let mut t_prev = at(pp, i_prev);
        let mut t_next = at(pp, i_next);

        if t_prev.is_none() {
            match settings.ending_start {
                ZERO_SLOPE_ENDING => {
                    // f'(t0) = 0
                    i_prev = i1;
                    t_prev = Some(2.0 * t0 - t1);
                }
                WRAP_AROUND_ENDING => {
                    // use the other end of the curve
                    i_prev = pp.len() as isize - 2;
                    t_prev = Some(
                        t0 + at(pp, i_prev).unwrap_or(f64::NAN)
                            - at(pp, i_prev + 1).unwrap_or(f64::NAN),
                    );
                }
                _ => {
                    // f''(t0) = 0 a.k.a. Natural Spline
                    i_prev = i1;
                    t_prev = Some(t1);
                }
            }
        }
        if t_next.is_none() {
            match settings.ending_end {
                ZERO_SLOPE_ENDING => {
                    // f'(tN) = 0
                    i_next = i1;
                    t_next = Some(2.0 * t1 - t0);
                }
                WRAP_AROUND_ENDING => {
                    // use the other end of the curve
                    i_next = 1;
                    t_next =
                        Some(t1 + at(pp, 1).unwrap_or(f64::NAN) - at(pp, 0).unwrap_or(f64::NAN));
                }
                _ => {
                    // f''(tN) = 0, a.k.a. Natural Spline
                    i_next = i1 - 1;
                    t_next = Some(t0);
                }
            }
        }
        let half_dt = (t1 - t0) * 0.5;
        let stride = self.value_size as isize;
        self.weight_prev = half_dt / (t0 - t_prev.unwrap_or(f64::NAN));
        self.weight_next = half_dt / (t_next.unwrap_or(f64::NAN) - t1);
        // A negative offset (a wrap on a one-sample track) reads nothing,
        // as `values[-n]` is `undefined` in JS.
        self.offset_prev = if i_prev * stride >= 0 {
            (i_prev * stride) as usize
        } else {
            usize::MAX / 2
        };
        self.offset_next = if i_next * stride >= 0 {
            (i_next * stride) as usize
        } else {
            usize::MAX / 2
        };
    }

    /// `interpolate_(i1, t0, t, t1)`.
    pub fn interpolate<R: ResultBuffer + ?Sized>(
        &self,
        values: &[f32],
        i1: usize,
        t0: f64,
        t: f64,
        t1: f64,
        result: &mut R,
    ) {
        let stride = self.value_size;
        match self.kind {
            InterpolantKind::Discrete => {
                self.copy_sample_value(values, i1.wrapping_sub(1), result);
            }
            InterpolantKind::Linear => {
                let offset1 = i1 * stride;
                let offset0 = offset1.wrapping_sub(stride);
                let weight1 = (t - t0) / (t1 - t0);
                let weight0 = 1.0 - weight1;
                for i in 0..stride {
                    result.set(
                        i,
                        sample(values, offset0.wrapping_add(i)) * weight0
                            + sample(values, offset1 + i) * weight1,
                    );
                }
            }
            InterpolantKind::QuaternionLinear => {
                let alpha = (t - t0) / (t1 - t0);
                let mut offset = i1 * stride;
                let end = offset + stride;
                // three.js writes every quaternion to offset 0 of the result.
                while offset < end {
                    let a = [
                        sample(values, offset.wrapping_sub(stride)),
                        sample(values, offset.wrapping_sub(stride).wrapping_add(1)),
                        sample(values, offset.wrapping_sub(stride).wrapping_add(2)),
                        sample(values, offset.wrapping_sub(stride).wrapping_add(3)),
                    ];
                    let b = [
                        sample(values, offset),
                        sample(values, offset + 1),
                        sample(values, offset + 2),
                        sample(values, offset + 3),
                    ];
                    let q = slerp_flat(a, b, alpha);
                    for (k, v) in q.iter().enumerate() {
                        result.set(k, *v);
                    }
                    offset += 4;
                }
            }
            InterpolantKind::Cubic => {
                let o1 = i1 * stride;
                let o0 = o1.wrapping_sub(stride);
                let op = self.offset_prev;
                let on = self.offset_next;
                let wp = self.weight_prev;
                let wn = self.weight_next;
                let p = (t - t0) / (t1 - t0);
                let pp = p * p;
                let ppp = pp * p;
                // evaluate polynomials
                let sp = -wp * ppp + 2.0 * wp * pp - wp * p;
                let s0 = (1.0 + wp) * ppp + (-1.5 - 2.0 * wp) * pp + (-0.5 + wp) * p + 1.0;
                let s1 = (-1.0 - wn) * ppp + (1.5 + wn) * pp + 0.5 * p;
                let sn = wn * ppp - wn * pp;
                // combine data linearly
                for i in 0..stride {
                    result.set(
                        i,
                        sp * sample(values, op.wrapping_add(i))
                            + s0 * sample(values, o0.wrapping_add(i))
                            + s1 * sample(values, o1 + i)
                            + sn * sample(values, on.wrapping_add(i)),
                    );
                }
            }
            InterpolantKind::GltfCubicSpline | InterpolantKind::GltfCubicSplineQuaternion => {
                let stride2 = stride * 2;
                let stride3 = stride * 3;
                let td = t1 - t0;
                let p = (t - t0) / td;
                let pp = p * p;
                let ppp = pp * p;
                let offset1 = i1 * stride3;
                let offset0 = offset1.wrapping_sub(stride3);
                let s2 = -2.0 * ppp + 3.0 * pp;
                let s3 = ppp - pp;
                let s0 = 1.0 - s2;
                let s1 = s3 - pp + p;
                // [ inTangent_1, splineVertex_1, outTangent_1, inTangent_2, ... ]
                for i in 0..stride {
                    let p0 = sample(values, offset0.wrapping_add(i + stride));
                    let m0 = sample(values, offset0.wrapping_add(i + stride2)) * td;
                    let p1 = sample(values, offset1 + i + stride);
                    let m1 = sample(values, offset1 + i) * td;
                    result.set(i, s0 * p0 + s1 * m0 + s2 * p1 + s3 * m1);
                }
                if self.kind == InterpolantKind::GltfCubicSplineQuaternion {
                    // _q.fromArray( result ).normalize().toArray( result )
                    let q = [result.get(0), result.get(1), result.get(2), result.get(3)];
                    let q = quaternion_normalize(q);
                    for (k, v) in q.iter().enumerate() {
                        result.set(k, *v);
                    }
                }
            }
        }
    }
}

/// `values[i]`, `NaN` where JS would read `undefined`.
#[inline]
fn sample(values: &[f32], i: usize) -> f64 {
    values.get(i).map(|&v| v as f64).unwrap_or(f64::NAN)
}

/// three.js `Quaternion.normalize()` on `[x, y, z, w]`.
pub fn quaternion_normalize(q: [f64; 4]) -> [f64; 4] {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        let l = 1.0 / l;
        [q[0] * l, q[1] * l, q[2] * l, q[3] * l]
    }
}

/// three.js `Quaternion.slerpFlat`, from `[x, y, z, w]` `a` to `b`.
pub fn slerp_flat(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let [mut x0, mut y0, mut z0, mut w0] = a;
    let [x1, y1, z1, w1] = b;
    if t == 0.0 {
        return [x0, y0, z0, w0];
    }
    if t == 1.0 {
        return [x1, y1, z1, w1];
    }
    if w0 != w1 || x0 != x1 || y0 != y1 || z0 != z1 {
        let mut s = 1.0 - t;
        let mut t = t;
        let cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
        let dir = if cos >= 0.0 { 1.0 } else { -1.0 };
        let sqr_sin = 1.0 - cos * cos;
        // Skip the Slerp for tiny steps to avoid numeric problems:
        if sqr_sin > f64::EPSILON {
            let sin = sqr_sin.sqrt();
            let len = sin.atan2(cos * dir);
            s = (s * len).sin() / sin;
            t = (t * len).sin() / sin;
        }
        let t_dir = t * dir;
        x0 = x0 * s + x1 * t_dir;
        y0 = y0 * s + y1 * t_dir;
        z0 = z0 * s + z1 * t_dir;
        w0 = w0 * s + w1 * t_dir;
        // Normalize in case we just did a lerp:
        if s == 1.0 - t {
            let f = 1.0 / (x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0).sqrt();
            x0 *= f;
            y0 *= f;
            z0 *= f;
            w0 *= f;
        }
    }
    [x0, y0, z0, w0]
}

/// three.js `Quaternion.multiplyQuaternionsFlat`: `a * b`.
pub fn multiply_quaternions_flat(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [x0, y0, z0, w0] = a;
    let [x1, y1, z1, w1] = b;
    [
        x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1,
        y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1,
        z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ]
}

/// An interpolant with its own data and result, as three.js users build one
/// (`new LinearInterpolant(times, values, size, result)`).
#[derive(Debug, Clone)]
pub struct OwnedInterpolant {
    pub interpolant: Interpolant,
    pub parameter_positions: Vec<f32>,
    pub sample_values: Vec<f32>,
    pub result: Vec<f32>,
    pub settings: Option<InterpolantSettings>,
}

impl OwnedInterpolant {
    pub fn new(
        kind: InterpolantKind,
        parameter_positions: Vec<f32>,
        sample_values: Vec<f32>,
        sample_size: usize,
    ) -> Self {
        Self {
            interpolant: Interpolant::new(kind, sample_size),
            parameter_positions,
            sample_values,
            result: vec![0.0; sample_size],
            settings: None,
        }
    }

    /// `evaluate(t)`, returning the result buffer.
    pub fn evaluate(&mut self, t: f64) -> &[f32] {
        self.interpolant.evaluate(
            &self.parameter_positions,
            &self.sample_values,
            t,
            self.settings.as_ref(),
            self.result.as_mut_slice(),
        );
        &self.result
    }
}
