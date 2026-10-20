//! three.js r165 `PropertyMixer`: the per-property accumulator that blends
//! every action's sample by weight (lerp, quaternion slerp, or a select for
//! booleans and strings), adds the additive layer, mixes toward the saved
//! original for weights under one, and writes the binding only when the
//! result changed from the previous frame.
//!
//! Buffer layout, as three.js's: `[ incoming | accu0 | accu1 | orig | addAccu | (work) ]`,
//! `f64` like its `Float64Array`.

use super::binding::{AnimationHost, HostKind, PropertyBinding};
use super::group::GroupRef;
use super::interpolant::{multiply_quaternions_flat, slerp_flat};
use super::track::ValueType;

/// What a mixer writes through: one binding, or a group's per-object ones
/// (three.js `PropertyBinding.Composite`).
pub enum Binding<K: HostKind> {
    Single(PropertyBinding<K>),
    Composite { group: GroupRef<K>, path: String },
}

impl<K: HostKind> std::fmt::Debug for Binding<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Binding::Single(b) => f.debug_tuple("Single").field(&b.path).finish(),
            Binding::Composite { path, .. } => f.debug_tuple("Composite").field(path).finish(),
        }
    }
}

impl<K: HostKind> Binding<K> {
    pub fn path(&self) -> &str {
        match self {
            Binding::Single(b) => &b.path,
            Binding::Composite { path, .. } => path,
        }
    }
    pub fn get_value<H: AnimationHost<K>>(&mut self, host: &mut H, out: &mut [f64]) {
        match self {
            Binding::Single(b) => b.get_value(host, out),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.get_value(host, path, out)
                }
            }
        }
    }
    pub fn set_value<H: AnimationHost<K>>(&mut self, host: &mut H, values: &[f64]) {
        match self {
            Binding::Single(b) => b.set_value(host, values),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.set_value(host, path, values)
                }
            }
        }
    }
    pub fn get_strings<H: AnimationHost<K>>(&mut self, host: &mut H, out: &mut [Option<String>]) {
        match self {
            Binding::Single(b) => b.get_strings(host, out),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.get_strings(host, path, out)
                }
            }
        }
    }
    pub fn set_strings<H: AnimationHost<K>>(&mut self, host: &mut H, values: &[Option<String>]) {
        match self {
            Binding::Single(b) => b.set_strings(host, values),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.set_strings(host, path, values)
                }
            }
        }
    }
    /// `bind()`.
    pub fn bind<H: AnimationHost<K>>(&mut self, host: &mut H) {
        match self {
            Binding::Single(b) => b.bind(host),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.bind_path(host, path)
                }
            }
        }
    }
    /// `unbind()`.
    pub fn unbind<H: AnimationHost<K>>(&mut self, host: &mut H) {
        match self {
            Binding::Single(b) => b.unbind(host),
            Binding::Composite { group, path } => {
                if let Ok(mut g) = group.lock() {
                    g.unbind_path(host, path)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mix {
    Lerp,
    Slerp,
    Select,
}

/// three.js `PropertyMixer`.
#[derive(Debug)]
pub struct PropertyMixer<K: HostKind> {
    pub binding: Binding<K>,
    pub value_size: usize,
    mix: Mix,
    strings: bool,
    /// The numeric buffer (booleans as 0/1).
    pub buffer: Vec<f64>,
    /// A string track's buffer.
    pub string_buffer: Vec<Option<String>>,
    work_index: usize,
    orig_index: usize,
    add_index: usize,
    pub cumulative_weight: f64,
    pub cumulative_weight_additive: f64,
    pub use_count: usize,
    pub reference_count: usize,
    /// Position in the mixer's binding list (`_cacheIndex`).
    pub(crate) cache_index: Option<usize>,
    /// The root uuid and track name it is cached under.
    pub(crate) root_uuid: String,
    pub(crate) track_name: String,
    /// Actions holding it (Rust's lifetime count; three.js has the GC).
    pub(crate) holders: usize,
}

impl<K: HostKind> PropertyMixer<K> {
    /// `new PropertyMixer(binding, typeName, valueSize)`.
    pub fn new(binding: Binding<K>, value_type: ValueType, value_size: usize) -> Self {
        let (mix, len, strings) = match value_type {
            ValueType::Quaternion => (Mix::Slerp, value_size * 6, false),
            ValueType::String => (Mix::Select, value_size * 5, true),
            ValueType::Bool => (Mix::Select, value_size * 5, false),
            _ => (Mix::Lerp, value_size * 5, false),
        };
        let fill = if mix == Mix::Select { f64::NAN } else { 0.0 };
        Self {
            binding,
            value_size,
            mix,
            strings,
            buffer: if strings { Vec::new() } else { vec![fill; len] },
            string_buffer: if strings { vec![None; len] } else { Vec::new() },
            work_index: 5,
            orig_index: 3,
            add_index: 4,
            cumulative_weight: 0.0,
            cumulative_weight_additive: 0.0,
            use_count: 0,
            reference_count: 0,
            cache_index: None,
            root_uuid: String::new(),
            track_name: String::new(),
            holders: 0,
        }
    }

    /// Whether this mixes strings.
    pub fn is_string(&self) -> bool {
        self.strings
    }

    /// `accumulate(accuIndex, weight)`: fold `incoming` into `accu<i>`.
    pub fn accumulate(&mut self, accu_index: usize, weight: f64) {
        let stride = self.value_size;
        let offset = accu_index * stride + stride;
        let mut current = self.cumulative_weight;
        if current == 0.0 {
            // accuN := incoming * weight
            if self.strings {
                for i in 0..stride {
                    self.string_buffer[offset + i] = self.string_buffer[i].clone();
                }
            } else {
                for i in 0..stride {
                    self.buffer[offset + i] = self.buffer[i];
                }
            }
            current = weight;
        } else {
            // accuN := accuN + incoming * weight
            current += weight;
            let mix = weight / current;
            self.mix_region(offset, 0, mix, stride);
        }
        self.cumulative_weight = current;
    }

    /// `accumulateAdditive(weight)`: fold `incoming` into `addAccu`.
    pub fn accumulate_additive(&mut self, weight: f64) {
        let stride = self.value_size;
        let offset = stride * self.add_index;
        if self.cumulative_weight_additive == 0.0 {
            // add = identity
            self.set_identity();
        }
        // add := add + incoming * weight
        self.mix_region_additive(offset, 0, weight, stride);
        self.cumulative_weight_additive += weight;
    }

    /// `apply(accuIndex)`: finish the frame and write the binding if the
    /// result differs from the last frame's.
    pub fn apply<H: AnimationHost<K>>(&mut self, host: &mut H, accu_index: usize) {
        let stride = self.value_size;
        let offset = accu_index * stride + stride;
        let weight = self.cumulative_weight;
        let weight_additive = self.cumulative_weight_additive;
        self.cumulative_weight = 0.0;
        self.cumulative_weight_additive = 0.0;
        if weight < 1.0 {
            // accuN := accuN + original * ( 1 - cumulativeWeight )
            let original = stride * self.orig_index;
            self.mix_region(offset, original, 1.0 - weight, stride);
        }
        if weight_additive > 0.0 {
            // accuN := accuN + additive accuN
            self.mix_region_additive(offset, self.add_index * stride, 1.0, stride);
        }
        for i in stride..stride + stride {
            let changed = if self.strings {
                self.string_buffer[i] != self.string_buffer[i + stride]
            } else {
                let (a, b) = (self.buffer[i], self.buffer[i + stride]);
                // `!==`; an undefined (NaN) boolean slot equals itself
                if self.mix == Mix::Select {
                    !(a == b || (a.is_nan() && b.is_nan()))
                } else {
                    a != b
                }
            };
            if changed {
                // value has changed -> update scene graph
                if self.strings {
                    self.binding
                        .set_strings(host, &self.string_buffer[offset..]);
                } else {
                    self.binding.set_value(host, &self.buffer[offset..]);
                }
                break;
            }
        }
    }

    /// `saveOriginalState()`: read the property, copy it to both accus.
    pub fn save_original_state<H: AnimationHost<K>>(&mut self, host: &mut H) {
        let stride = self.value_size;
        let original = stride * self.orig_index;
        if self.strings {
            self.binding
                .get_strings(host, &mut self.string_buffer[original..]);
            for i in stride..original {
                self.string_buffer[i] = self.string_buffer[original + (i % stride)].clone();
            }
        } else {
            self.binding.get_value(host, &mut self.buffer[original..]);
            // accu[0..1] := orig -- initially detect changes against the original
            for i in stride..original {
                self.buffer[i] = self.buffer[original + (i % stride)];
            }
        }
        // Add to identity for additive
        self.set_identity();
        self.cumulative_weight = 0.0;
        self.cumulative_weight_additive = 0.0;
    }

    /// `restoreOriginalState()`.
    pub fn restore_original_state<H: AnimationHost<K>>(&mut self, host: &mut H) {
        let original = self.value_size * 3;
        if self.strings {
            self.binding
                .set_strings(host, &self.string_buffer[original..]);
        } else {
            self.binding.set_value(host, &self.buffer[original..]);
        }
    }

    fn set_identity(&mut self) {
        let start = self.add_index * self.value_size;
        let end = start + self.value_size;
        match self.mix {
            Mix::Lerp => self.buffer[start..end].iter_mut().for_each(|v| *v = 0.0),
            Mix::Slerp => {
                self.buffer[start..end].iter_mut().for_each(|v| *v = 0.0);
                self.buffer[start + 3] = 1.0;
            }
            Mix::Select => {
                // _setAdditiveIdentityOther: add := orig
                let orig = self.orig_index * self.value_size;
                for i in 0..self.value_size {
                    if self.strings {
                        self.string_buffer[start + i] = self.string_buffer[orig + i].clone();
                    } else {
                        self.buffer[start + i] = self.buffer[orig + i];
                    }
                }
            }
        }
    }

    fn mix_region(&mut self, dst: usize, src: usize, t: f64, stride: usize) {
        match self.mix {
            Mix::Select => self.select(dst, src, t, stride),
            Mix::Slerp => self.slerp(dst, src, t),
            Mix::Lerp => {
                let b = &mut self.buffer;
                let s = 1.0 - t;
                for i in 0..stride {
                    b[dst + i] = b[dst + i] * s + b[src + i] * t;
                }
            }
        }
    }

    fn mix_region_additive(&mut self, dst: usize, src: usize, t: f64, stride: usize) {
        match self.mix {
            // additive is not relevant for non-numeric types
            Mix::Select => self.select(dst, src, t, stride),
            Mix::Slerp => {
                let work = self.work_index * stride;
                let b = &mut self.buffer;
                let q = multiply_quaternions_flat(quat(b, dst), quat(b, src));
                put_quat(b, work, q);
                let r = slerp_flat(quat(b, dst), quat(b, work), t);
                put_quat(b, dst, r);
            }
            Mix::Lerp => {
                let b = &mut self.buffer;
                for i in 0..stride {
                    b[dst + i] += b[src + i] * t;
                }
            }
        }
    }

    fn select(&mut self, dst: usize, src: usize, t: f64, stride: usize) {
        if t >= 0.5 {
            for i in 0..stride {
                if self.strings {
                    self.string_buffer[dst + i] = self.string_buffer[src + i].clone();
                } else {
                    self.buffer[dst + i] = self.buffer[src + i];
                }
            }
        }
    }

    fn slerp(&mut self, dst: usize, src: usize, t: f64) {
        let b = &mut self.buffer;
        let r = slerp_flat(quat(b, dst), quat(b, src), t);
        put_quat(b, dst, r);
    }
}

/// Four values from `o` (`NaN` past the end, as JS reads `undefined`).
#[inline]
fn quat(b: &[f64], o: usize) -> [f64; 4] {
    [0, 1, 2, 3].map(|k| b.get(o + k).copied().unwrap_or(f64::NAN))
}

/// Store four values at `o` (dropping any past the end, as a typed array does).
#[inline]
fn put_quat(b: &mut [f64], o: usize, q: [f64; 4]) {
    for (k, v) in q.into_iter().enumerate() {
        if let Some(s) = b.get_mut(o + k) {
            *s = v;
        }
    }
}
