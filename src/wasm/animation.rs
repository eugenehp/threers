//! Bindings for the animation system (`crate::animation::three`, three.js
//! r165's ported): clips, keyframe tracks, interpolants, the mixer and its
//! actions, object groups, and the scene-graph lookups `PropertyBinding`
//! makes. `web/threejs-shim.js` wraps these as `THREE.AnimationClip`,
//! `THREE.AnimationMixer` and the rest.
//!
//! The mixer runs here, over JS objects: a node is a number the shim gives
//! each object it animates, and a bound property a number this side gives
//! each binding. Three callbacks the shim installs answer for the JS side —
//! `findNode` (which resolves names against the Rust scene graph through
//! [`WebScene::find_animation_node`]), `bind` (the property-level half of
//! `PropertyBinding.bind()`, on the JS object three.js code expects to see
//! animated) and `read` (a property's current value, for
//! `saveOriginalState`). Values the mixer writes are queued, one flat
//! `Float64Array` per call ([`take_animation_writes`]), and the shim stores
//! them into the JS objects in one pass — whose setters bring the Rust scene
//! graph's nodes up to date at the next sync, as for any other change.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use crate::animation::three::{
    self, AnimationClip, AnimationHost, AnimationMixer, AnimationObjectGroup, ClipRef, GroupRef,
    HostKind, Interpolant, InterpolantKind, InterpolantSettings, Interpolation, KeyframeTrack,
    MixerRoot, NodeTree, ParsedPath, TrackValues, ValueType,
};

/// The JS host's types: nodes and bound properties are numbers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JsKind;

impl HostKind for JsKind {
    type Node = u32;
    type Target = u32;
}

#[derive(Default)]
struct AnimState {
    uuids: HashMap<u32, String>,
    find: Option<js_sys::Function>,
    bind: Option<js_sys::Function>,
    read: Option<js_sys::Function>,
    next_target: u32,
    widths: HashMap<u32, usize>,
    writes: Vec<f64>,
    string_writes: Vec<(u32, Vec<Option<String>>)>,
    released: Vec<u32>,
    /// Object groups by the number the shim gave them (a mixer root may be one).
    groups: HashMap<u32, GroupRef<JsKind>>,
}

thread_local! {
    static ANIM: RefCell<AnimState> = RefCell::new(AnimState::default());
}

/// The JS host: state is thread-local (shared by every mixer, as groups and
/// bound properties may be), callbacks are called with no borrow held.
pub(crate) struct JsHost;

fn callback(pick: fn(&AnimState) -> &Option<js_sys::Function>) -> Option<js_sys::Function> {
    ANIM.with(|s| pick(&s.borrow()).clone())
}

fn opt_js(s: &Option<String>) -> JsValue {
    match s {
        Some(s) => JsValue::from_str(s),
        None => JsValue::UNDEFINED,
    }
}

impl AnimationHost<JsKind> for JsHost {
    fn node_uuid(&self, node: &u32) -> String {
        ANIM.with(|s| s.borrow().uuids.get(node).cloned())
            .unwrap_or_else(|| format!("#{node}"))
    }

    fn find_node(&mut self, root: &u32, node_name: Option<&str>) -> Option<u32> {
        let f = callback(|s| &s.find)?;
        let name = node_name
            .map(JsValue::from_str)
            .unwrap_or(JsValue::UNDEFINED);
        let r = f.call2(&JsValue::NULL, &JsValue::from(*root), &name).ok()?;
        r.as_f64()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| v as u32)
    }

    fn bind(
        &mut self,
        node: &u32,
        p: &ParsedPath,
        path: &str,
        ty: ValueType,
        size: usize,
    ) -> Option<u32> {
        let f = callback(|s| &s.bind)?;
        let target = ANIM.with(|s| {
            let mut s = s.borrow_mut();
            s.next_target = s.next_target.wrapping_add(1).max(1);
            s.next_target
        });
        let args = js_sys::Array::new();
        args.push(&JsValue::from(target));
        args.push(&JsValue::from(*node));
        args.push(&opt_js(&p.object_name));
        args.push(&opt_js(&p.object_index));
        args.push(&JsValue::from_str(&p.property_name));
        args.push(&opt_js(&p.property_index));
        args.push(&JsValue::from_str(ty.name()));
        args.push(&JsValue::from(size as u32));
        args.push(&JsValue::from_str(path));
        args.push(&opt_js(&p.node_name));
        let width = f.apply(&JsValue::NULL, &args).ok()?.as_f64().unwrap_or(0.0);
        if width >= 1.0 {
            ANIM.with(|s| s.borrow_mut().widths.insert(target, width as usize));
            Some(target)
        } else {
            None
        }
    }

    fn get_value(&mut self, target: &u32, out: &mut [f64]) {
        let Some(f) = callback(|s| &s.read) else {
            return;
        };
        let Ok(v) = f.call1(&JsValue::NULL, &JsValue::from(*target)) else {
            return;
        };
        if let Some(n) = v.as_f64() {
            if let Some(o) = out.first_mut() {
                *o = n;
            }
            return;
        }
        if let Some(b) = v.as_bool() {
            if let Some(o) = out.first_mut() {
                *o = b as u8 as f64;
            }
            return;
        }
        if let Some(a) = v.dyn_ref::<js_sys::Float64Array>() {
            let n = (a.length() as usize).min(out.len());
            a.subarray(0, n as u32).copy_to(&mut out[..n]);
            return;
        }
        if let Some(a) = v.dyn_ref::<js_sys::Array>() {
            for (i, x) in a.iter().enumerate().take(out.len()) {
                out[i] = x
                    .as_f64()
                    .or_else(|| x.as_bool().map(|b| b as u8 as f64))
                    .unwrap_or(f64::NAN);
            }
        }
    }

    fn set_value(&mut self, target: &u32, values: &[f64]) {
        ANIM.with(|s| {
            let mut s = s.borrow_mut();
            let width = s.widths.get(target).copied().unwrap_or(0).min(values.len());
            s.writes.push(*target as f64);
            s.writes.push(width as f64);
            s.writes.extend_from_slice(&values[..width]);
        });
    }

    fn get_strings(&mut self, target: &u32, out: &mut [Option<String>]) {
        let Some(f) = callback(|s| &s.read) else {
            return;
        };
        let Ok(v) = f.call1(&JsValue::NULL, &JsValue::from(*target)) else {
            return;
        };
        if let Some(s) = v.as_string() {
            if let Some(o) = out.first_mut() {
                *o = Some(s);
            }
        } else if let Some(a) = v.dyn_ref::<js_sys::Array>() {
            for (i, x) in a.iter().enumerate().take(out.len()) {
                out[i] = x.as_string();
            }
        }
    }

    fn set_strings(&mut self, target: &u32, values: &[Option<String>]) {
        ANIM.with(|s| {
            let mut s = s.borrow_mut();
            let width = s.widths.get(target).copied().unwrap_or(0).min(values.len());
            s.string_writes.push((*target, values[..width].to_vec()));
        });
    }

    fn release(&mut self, target: u32) {
        ANIM.with(|s| {
            let mut s = s.borrow_mut();
            s.widths.remove(&target);
            s.released.push(target);
        });
    }
}

/// Install the shim's callbacks: `findNode(rootKey, nodeName) → key`,
/// `bind(target, nodeKey, objectName, objectIndex, propertyName,
/// propertyIndex, valueTypeName, valueSize, path, nodeName) → width` (0:
/// unavailable) and `read(target) → value(s)`.
#[wasm_bindgen(js_name = setAnimationCallbacks)]
pub fn set_animation_callbacks(
    find: js_sys::Function,
    bind: js_sys::Function,
    read: js_sys::Function,
) {
    ANIM.with(|s| {
        let mut s = s.borrow_mut();
        s.find = Some(find);
        s.bind = Some(bind);
        s.read = Some(read);
    });
}

/// The uuid three.js keys the mixer's caches by, for an object the shim
/// numbered `key` (roots and group members).
#[wasm_bindgen(js_name = setAnimationNodeUuid)]
pub fn set_animation_node_uuid(key: u32, uuid: &str) {
    ANIM.with(|s| s.borrow_mut().uuids.insert(key, uuid.to_string()));
}

/// Forget a numbered object.
#[wasm_bindgen(js_name = dropAnimationNode)]
pub fn drop_animation_node(key: u32) {
    ANIM.with(|s| s.borrow_mut().uuids.remove(&key));
}

/// Values written since the last call, `[target, n, v0 … vn-1, target, …]`.
#[wasm_bindgen(js_name = takeAnimationWrites)]
pub fn take_animation_writes() -> Vec<f64> {
    ANIM.with(|s| std::mem::take(&mut s.borrow_mut().writes))
}

/// Whether there are writes (numeric or string) or releases waiting.
#[wasm_bindgen(js_name = animationWritesPending)]
pub fn animation_writes_pending() -> bool {
    ANIM.with(|s| {
        let s = s.borrow();
        !s.writes.is_empty() || !s.string_writes.is_empty() || !s.released.is_empty()
    })
}

/// String-track writes since the last call: `[target, [values…], …]`.
#[wasm_bindgen(js_name = takeAnimationStringWrites)]
pub fn take_animation_string_writes() -> js_sys::Array {
    let w = ANIM.with(|s| std::mem::take(&mut s.borrow_mut().string_writes));
    let out = js_sys::Array::new();
    for (t, vals) in w {
        out.push(&JsValue::from(t));
        let a = js_sys::Array::new();
        for v in vals {
            a.push(&opt_js(&v));
        }
        out.push(&a);
    }
    out
}

/// Bound properties no binding uses any more.
#[wasm_bindgen(js_name = takeAnimationReleased)]
pub fn take_animation_released() -> Vec<u32> {
    ANIM.with(|s| std::mem::take(&mut s.borrow_mut().released))
}

// ---- the scene graph's half of PropertyBinding.findNode ----

struct GraphTree<'a> {
    scene: &'a WebScene,
}

impl NodeTree for GraphTree<'_> {
    type Id = ObjectId;
    fn name_is(&self, n: &ObjectId, name: &str) -> bool {
        self.scene
            .inner
            .arena
            .get(*n)
            .map(|o| o.name == name)
            .unwrap_or(false)
    }
    fn uuid_is(&self, n: &ObjectId, uuid: &str) -> bool {
        self.scene
            .node_uuids
            .get(n)
            .map(|u| u == uuid)
            .unwrap_or(false)
    }
    fn skeleton_bones(&self, n: &ObjectId) -> Option<Vec<ObjectId>> {
        match &self.scene.inner.arena.get(*n)?.kind {
            crate::core::ObjectKind::SkinnedMesh(sm) => {
                Some(sm.skeleton.bones.iter().map(|b| b.node).collect())
            }
            _ => None,
        }
    }
    fn children(&self, n: &ObjectId) -> Vec<ObjectId> {
        self.scene
            .inner
            .arena
            .get(*n)
            .map(|o| o.children.clone())
            .unwrap_or_default()
    }
}

#[wasm_bindgen]
impl WebScene {
    /// three.js `PropertyBinding.findNode(root, nodeName)` on this graph:
    /// the node's id, or `undefined`. Names and uuids are the mirrored ones.
    #[wasm_bindgen(js_name = findAnimationNode)]
    pub fn find_animation_node(&self, root: f64, node_name: Option<String>) -> Option<f64> {
        let root = object_id_from_f64(root)?;
        self.inner.arena.get(root)?;
        three::find_node(&GraphTree { scene: self }, &root, node_name.as_deref())
            .map(object_id_to_f64)
    }

    /// Mirror an object's `name` and `uuid` (what tracks name nodes by).
    #[wasm_bindgen(js_name = setNameUuid)]
    pub fn set_name_uuid(&mut self, node: &WebObjectHandle, name: &str, uuid: &str) {
        if let Some(o) = self.inner.arena.get_mut(node.id) {
            if o.name != name {
                o.name = name.to_string();
            }
            self.node_uuids.insert(node.id, uuid.to_string());
        }
    }
}

// ---- tracks and clips ----

fn value_type(name: &str) -> Result<ValueType, JsValue> {
    ValueType::from_name(name).ok_or_else(|| {
        JsValue::from_str(&format!(
            "THREE.KeyframeTrack: Unsupported typeName: {name}"
        ))
    })
}

/// three.js `KeyframeTrack` data (the shim's track classes hold the arrays
/// three.js code reads; this is where their methods run).
#[wasm_bindgen]
pub struct WebAnimTrack {
    pub(crate) inner: KeyframeTrack,
}

#[wasm_bindgen]
impl WebAnimTrack {
    /// `type`: `ValueTypeName`. Values: `values` for numeric tracks, `bools`
    /// (0/1) for boolean ones, `strings` for string ones. `interpolation`:
    /// a three.js constant, 0 for the type's default, 1 for glTF's cubic
    /// spline.
    #[wasm_bindgen(constructor)]
    pub fn new(
        type_name: &str,
        name: &str,
        times: Vec<f32>,
        values: Option<Vec<f32>>,
        bools: Option<Vec<u8>>,
        strings: Option<Vec<String>>,
        interpolation: u32,
    ) -> Result<WebAnimTrack, JsValue> {
        let ty = value_type(type_name)?;
        let vals = match ty {
            ValueType::Bool => TrackValues::Bool(
                bools
                    .unwrap_or_default()
                    .into_iter()
                    .map(|b| b != 0)
                    .collect(),
            ),
            ValueType::String => TrackValues::String(strings.unwrap_or_default()),
            _ => TrackValues::Numeric(values.unwrap_or_default()),
        };
        let interp = match interpolation {
            0 | 1 => None,
            c => Interpolation::from_constant(c),
        };
        let mut inner = KeyframeTrack::new(ty, name, times, vals, interp)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        if interpolation == 1 {
            inner.set_interpolation(Interpolation::GltfCubicSpline);
        }
        Ok(WebAnimTrack { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn times(&self) -> Vec<f32> {
        self.inner.times.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.inner
            .values
            .numeric()
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }
    #[wasm_bindgen(getter)]
    pub fn bools(&self) -> Vec<u8> {
        match &self.inner.values {
            TrackValues::Bool(b) => b.iter().map(|&x| x as u8).collect(),
            _ => Vec::new(),
        }
    }
    #[wasm_bindgen(getter)]
    pub fn strings(&self) -> Vec<String> {
        match &self.inner.values {
            TrackValues::String(s) => s.clone(),
            _ => Vec::new(),
        }
    }
    /// The three.js constant (0 for none: glTF's cubic spline).
    #[wasm_bindgen(getter)]
    pub fn interpolation(&self) -> u32 {
        self.inner.interpolation().constant().unwrap_or(0)
    }
    /// `setInterpolation(mode)`; returns whether the type supports it.
    #[wasm_bindgen(js_name = setInterpolation)]
    pub fn set_interpolation(&mut self, constant: u32) -> bool {
        match Interpolation::from_constant(constant) {
            Some(i) if self.inner.value_type.supports(i) => {
                self.inner.set_interpolation(i);
                true
            }
            _ => false,
        }
    }
    #[wasm_bindgen(js_name = valueSize)]
    pub fn value_size(&self) -> f64 {
        self.inner.value_size_f64()
    }
    pub fn optimize(&mut self) {
        self.inner.optimize();
    }
    pub fn trim(&mut self, start: f64, end: f64) {
        self.inner.trim(start, end);
    }
    pub fn shift(&mut self, offset: f64) {
        self.inner.shift(offset);
    }
    pub fn scale(&mut self, factor: f64) {
        self.inner.scale(factor);
    }
    pub fn validate(&self) -> bool {
        self.inner.validate()
    }
    /// `KeyframeTrack.toJSON(track)`, as JSON text.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        three::track_to_json_string(&self.inner)
    }
    /// The shim's `track.modifiers` (see [`modifiers_of`]).
    #[wasm_bindgen(js_name = setModifiers)]
    pub fn set_modifiers(&mut self, kinds: Vec<u8>, params: Vec<f64>) {
        self.inner.modifiers = modifiers_of(&kinds, &params);
    }
}

/// `THREE.TrackModifier` descriptors as kinds (0 noise, 1 cycles, 2 stepped,
/// 3 limit) with three parameters each: noise `amplitude, frequency, seed`;
/// cycles `before, after`; stepped `stepSize`; limit `min, max`.
fn modifiers_of(kinds: &[u8], params: &[f64]) -> Vec<three::TrackModifier> {
    kinds
        .iter()
        .enumerate()
        .filter_map(|(i, k)| {
            let p = |j: usize| params.get(i * 3 + j).copied().unwrap_or(0.0);
            Some(match k {
                0 => three::TrackModifier::Noise {
                    amplitude: p(0),
                    frequency: p(1),
                    seed: p(2),
                },
                1 => three::TrackModifier::Cycles {
                    before: p(0),
                    after: p(1),
                },
                2 => three::TrackModifier::Stepped { step_size: p(0) },
                3 => three::TrackModifier::Limit {
                    min: p(0),
                    max: p(1),
                },
                _ => return None,
            })
        })
        .collect()
}

/// `TrackModifier.remapTime(modifiers, t, track)` (`duration`: the track's
/// last key time).
#[wasm_bindgen(js_name = trackModifierRemapTime)]
pub fn track_modifier_remap_time(kinds: Vec<u8>, params: Vec<f64>, t: f64, duration: f64) -> f64 {
    three::TrackModifier::remap_time(&modifiers_of(&kinds, &params), t, duration)
}

/// `TrackModifier.applyValue(modifiers, values, t)`.
#[wasm_bindgen(js_name = trackModifierApplyValue)]
pub fn track_modifier_apply_value(
    kinds: Vec<u8>,
    params: Vec<f64>,
    mut values: Vec<f64>,
    t: f64,
) -> Vec<f64> {
    three::TrackModifier::apply_value(&modifiers_of(&kinds, &params), &mut values, t);
    values
}

/// three.js `AnimationClip` data and its operations.
#[wasm_bindgen]
pub struct WebAnimClip {
    pub(crate) inner: Arc<AnimationClip>,
}

impl WebAnimClip {
    fn wrap(c: AnimationClip) -> WebAnimClip {
        WebAnimClip { inner: Arc::new(c) }
    }
}

#[wasm_bindgen]
impl WebAnimClip {
    /// An empty clip; add tracks, then `resetDuration` if the duration was
    /// negative (three.js's constructor does it with its tracks in hand).
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, duration: f64, blend_mode: u32, uuid: &str) -> WebAnimClip {
        let mut c = AnimationClip::with_blend_mode(name, 0.0, Vec::new(), blend_mode);
        c.duration = duration;
        c.uuid = uuid.to_string();
        WebAnimClip::wrap(c)
    }
    #[wasm_bindgen(js_name = addTrack)]
    pub fn add_track(&mut self, track: &WebAnimTrack) {
        Arc::make_mut(&mut self.inner)
            .tracks
            .push(track.inner.clone());
    }
    #[wasm_bindgen(js_name = resetDuration)]
    pub fn reset_duration(&mut self) -> f64 {
        Arc::make_mut(&mut self.inner).reset_duration().duration
    }
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.inner.duration
    }
    #[wasm_bindgen(getter)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.clone()
    }
    #[wasm_bindgen(getter, js_name = blendMode)]
    pub fn blend_mode(&self) -> u32 {
        self.inner.blend_mode
    }
    #[wasm_bindgen(getter, js_name = trackCount)]
    pub fn track_count(&self) -> usize {
        self.inner.tracks.len()
    }
    /// A copy of track `i`.
    pub fn track(&self, i: usize) -> Option<WebAnimTrack> {
        self.inner
            .tracks
            .get(i)
            .map(|t| WebAnimTrack { inner: t.clone() })
    }
    /// The type name of track `i`.
    #[wasm_bindgen(js_name = trackType)]
    pub fn track_type(&self, i: usize) -> Option<String> {
        self.inner
            .tracks
            .get(i)
            .map(|t| t.value_type.name().to_string())
    }
    /// `AnimationClip.toJSON(clip)`, as JSON text.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        self.inner.to_json_string()
    }
    /// `AnimationClip.parse(json)`, from JSON text.
    pub fn parse(json: &str) -> Result<WebAnimClip, JsValue> {
        AnimationClip::parse_json_str(json)
            .map(WebAnimClip::wrap)
            .map_err(|e| JsValue::from_str(&e))
    }
    /// `AnimationClip.parseAnimation(animation, bones)` (the JSON hierarchy
    /// format; `bone_names` per hierarchy entry).
    #[wasm_bindgen(js_name = parseAnimation)]
    pub fn parse_animation(json: &str, bone_names: Vec<String>) -> Option<WebAnimClip> {
        let v = crate::loaders::json::parse(json).ok()?;
        AnimationClip::parse_animation(&v, &bone_names).map(WebAnimClip::wrap)
    }
    /// `AnimationClip.CreateFromMorphTargetSequence(name, targets, fps, noLoop)`.
    #[wasm_bindgen(js_name = createFromMorphTargetSequence)]
    pub fn create_from_morph_target_sequence(
        name: &str,
        targets: Vec<String>,
        fps: f64,
        no_loop: bool,
    ) -> WebAnimClip {
        WebAnimClip::wrap(AnimationClip::create_from_morph_target_sequence(
            name, &targets, fps, no_loop,
        ))
    }
    /// `AnimationClip.CreateClipsFromMorphTargetSequences(targets, fps, noLoop)`.
    #[wasm_bindgen(js_name = createClipsFromMorphTargetSequences)]
    pub fn create_clips_from_morph_target_sequences(
        targets: Vec<String>,
        fps: f64,
        no_loop: bool,
    ) -> js_sys::Array {
        AnimationClip::create_clips_from_morph_target_sequences(&targets, fps, no_loop)
            .into_iter()
            .map(|c| JsValue::from(WebAnimClip::wrap(c)))
            .collect()
    }
    /// `AnimationUtils.subclip(clip, name, startFrame, endFrame, fps)`.
    pub fn subclip(&self, name: &str, start_frame: f64, end_frame: f64, fps: f64) -> WebAnimClip {
        WebAnimClip::wrap(three::utils::subclip(
            &self.inner,
            name,
            start_frame,
            end_frame,
            fps,
        ))
    }
    /// `AnimationUtils.makeClipAdditive(this, referenceFrame, this, fps)` in place.
    #[wasm_bindgen(js_name = makeClipAdditive)]
    pub fn make_clip_additive(&mut self, reference_frame: f64, fps: f64) {
        three::utils::make_clip_additive(
            Arc::make_mut(&mut self.inner),
            reference_frame,
            None,
            fps,
        );
    }
    /// `AnimationUtils.makeClipAdditive(this, referenceFrame, reference, fps)`.
    #[wasm_bindgen(js_name = makeClipAdditiveFrom)]
    pub fn make_clip_additive_from(
        &mut self,
        reference_frame: f64,
        reference: &WebAnimClip,
        fps: f64,
    ) {
        let reference = (*reference.inner).clone();
        three::utils::make_clip_additive(
            Arc::make_mut(&mut self.inner),
            reference_frame,
            Some(&reference),
            fps,
        );
    }
}

// ---- interpolants ----

/// A standalone three.js interpolant (`new LinearInterpolant(times, values,
/// size)` …): kind 0 discrete, 1 linear, 2 cubic, 3 quaternion linear,
/// 4 glTF cubic spline, 5 its quaternion variant.
#[wasm_bindgen]
pub struct WebInterpolant {
    inner: Interpolant,
    times: Vec<f32>,
    values: Vec<f32>,
    settings: Option<InterpolantSettings>,
}

#[wasm_bindgen]
impl WebInterpolant {
    #[wasm_bindgen(constructor)]
    pub fn new(kind: u32, times: Vec<f32>, values: Vec<f32>, sample_size: usize) -> WebInterpolant {
        let kind = match kind {
            0 => InterpolantKind::Discrete,
            2 => InterpolantKind::Cubic,
            3 => InterpolantKind::QuaternionLinear,
            4 => InterpolantKind::GltfCubicSpline,
            5 => InterpolantKind::GltfCubicSplineQuaternion,
            _ => InterpolantKind::Linear,
        };
        WebInterpolant {
            inner: Interpolant::new(kind, sample_size),
            times,
            values,
            settings: None,
        }
    }
    /// Replace the samples (`parameterPositions` / `sampleValues` changed).
    #[wasm_bindgen(js_name = setData)]
    pub fn set_data(&mut self, times: Vec<f32>, values: Vec<f32>) {
        self.times = times;
        self.values = values;
    }
    /// `interpolant.settings = { endingStart, endingEnd }`.
    #[wasm_bindgen(js_name = setSettings)]
    pub fn set_settings(&mut self, ending_start: u32, ending_end: u32) {
        self.settings = Some(InterpolantSettings {
            ending_start,
            ending_end,
        });
    }
    #[wasm_bindgen(js_name = clearSettings)]
    pub fn clear_settings(&mut self) {
        self.settings = None;
    }
    /// `evaluate(t)` into a `Float32Array`-like result (`f32` storage, as
    /// three.js's default result buffer).
    #[wasm_bindgen(js_name = evaluate32)]
    pub fn evaluate32(&mut self, t: f64) -> Vec<f32> {
        let mut r = vec![0.0f32; self.inner.value_size];
        self.inner.evaluate(
            &self.times,
            &self.values,
            t,
            self.settings.as_ref(),
            r.as_mut_slice(),
        );
        r
    }
    /// The seek alone, for a JS subclass with its own `interpolate_`:
    /// `[0, index, -, -, cachedIndex]` to copy sample `index`, or
    /// `[1 | 2, i1, t0, t1, cachedIndex]` to interpolate (2: the interval
    /// changed, call `intervalChanged_` first).
    pub fn seek(&mut self, t: f64) -> Vec<f64> {
        match self.inner.seek(&self.times, t) {
            three::Seek::Copy(i) => vec![
                0.0,
                i as f64,
                f64::NAN,
                f64::NAN,
                self.inner.cached_index() as f64,
            ],
            three::Seek::Interpolate {
                i1,
                t0,
                t1,
                changed,
            } => vec![
                if changed { 2.0 } else { 1.0 },
                i1 as f64,
                t0,
                t1,
                self.inner.cached_index() as f64,
            ],
        }
    }
    /// `evaluate(t)` into a `Float64Array` result.
    #[wasm_bindgen(js_name = evaluate64)]
    pub fn evaluate64(&mut self, t: f64) -> Vec<f64> {
        let mut r = vec![0.0f64; self.inner.value_size];
        self.inner.evaluate(
            &self.times,
            &self.values,
            t,
            self.settings.as_ref(),
            r.as_mut_slice(),
        );
        r
    }
}

// ---- a property mixer on its own ----

/// three.js `PropertyMixer` constructed by hand: binding `path` under root
/// `key` (an object's number or a group's). The shim passes its `buffer`
/// over before each call and takes it back after.
#[wasm_bindgen]
pub struct WebPropertyMixer {
    inner: three::PropertyMixer<JsKind>,
}

#[wasm_bindgen]
impl WebPropertyMixer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        key: u32,
        path: &str,
        type_name: &str,
        value_size: usize,
    ) -> Result<WebPropertyMixer, JsValue> {
        let ty = value_type(type_name)?;
        let binding = match root_of(key as f64) {
            Some(MixerRoot::Group(g)) => {
                let parsed = three::parse_track_name(path)
                    .map_err(|e| js_sys::Error::new(&e.to_string()))?;
                if let Ok(mut gl) = g.lock() {
                    gl.subscribe(&mut JsHost, path, &parsed, ty, value_size);
                }
                three::Binding::Composite {
                    group: g,
                    path: path.to_string(),
                }
            }
            _ => three::Binding::Single(
                three::PropertyBinding::new(&mut JsHost, key, path, None, ty, value_size)
                    .map_err(|e| js_sys::Error::new(&e.to_string()))?,
            ),
        };
        Ok(WebPropertyMixer {
            inner: three::PropertyMixer::new(binding, ty, value_size),
        })
    }
    #[wasm_bindgen(js_name = setBuffer)]
    pub fn set_buffer(&mut self, values: &[f64]) {
        let n = values.len().min(self.inner.buffer.len());
        self.inner.buffer[..n].copy_from_slice(&values[..n]);
    }
    pub fn buffer(&self) -> Vec<f64> {
        self.inner.buffer.clone()
    }
    pub fn accumulate(&mut self, accu_index: usize, weight: f64) {
        self.inner.accumulate(accu_index, weight);
    }
    #[wasm_bindgen(js_name = accumulateAdditive)]
    pub fn accumulate_additive(&mut self, weight: f64) {
        self.inner.accumulate_additive(weight);
    }
    pub fn apply(&mut self, accu_index: usize) {
        self.inner.apply(&mut JsHost, accu_index);
    }
    #[wasm_bindgen(js_name = saveOriginalState)]
    pub fn save_original_state(&mut self) {
        self.inner.save_original_state(&mut JsHost);
    }
    #[wasm_bindgen(js_name = restoreOriginalState)]
    pub fn restore_original_state(&mut self) {
        self.inner.restore_original_state(&mut JsHost);
    }
    #[wasm_bindgen(getter, js_name = cumulativeWeight)]
    pub fn cumulative_weight(&self) -> f64 {
        self.inner.cumulative_weight
    }
    #[wasm_bindgen(getter, js_name = cumulativeWeightAdditive)]
    pub fn cumulative_weight_additive(&self) -> f64 {
        self.inner.cumulative_weight_additive
    }
    #[wasm_bindgen(getter, js_name = useCount)]
    pub fn use_count(&self) -> usize {
        self.inner.use_count
    }
    #[wasm_bindgen(setter, js_name = useCount)]
    pub fn set_use_count(&mut self, v: usize) {
        self.inner.use_count = v;
    }
    #[wasm_bindgen(getter, js_name = referenceCount)]
    pub fn reference_count(&self) -> usize {
        self.inner.reference_count
    }
    #[wasm_bindgen(setter, js_name = referenceCount)]
    pub fn set_reference_count(&mut self, v: usize) {
        self.inner.reference_count = v;
    }
}

// ---- object groups ----

/// three.js `AnimationObjectGroup` over numbered JS objects.
#[wasm_bindgen]
pub struct WebAnimationObjectGroup {
    key: u32,
    inner: GroupRef<JsKind>,
}

impl Drop for WebAnimationObjectGroup {
    fn drop(&mut self) {
        ANIM.with(|s| s.borrow_mut().groups.remove(&self.key));
    }
}

#[wasm_bindgen]
impl WebAnimationObjectGroup {
    /// A group the mixer will know as root `key` (a number from the same
    /// series as the objects').
    #[wasm_bindgen(constructor)]
    pub fn new(key: u32, uuid: &str, objects: Vec<u32>) -> WebAnimationObjectGroup {
        let mut g = AnimationObjectGroup::new(&JsHost, objects);
        g.uuid = uuid.to_string();
        let inner = g.into_ref();
        ANIM.with(|s| s.borrow_mut().groups.insert(key, inner.clone()));
        WebAnimationObjectGroup { key, inner }
    }
    pub fn add(&self, keys: Vec<u32>) {
        if let Ok(mut g) = self.inner.lock() {
            g.add(&mut JsHost, &keys);
        }
    }
    pub fn remove(&self, keys: Vec<u32>) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(&JsHost, &keys);
        }
    }
    pub fn uncache(&self, keys: Vec<u32>) {
        if let Ok(mut g) = self.inner.lock() {
            g.uncache(&mut JsHost, &keys);
        }
    }
    #[wasm_bindgen(getter)]
    pub fn uuid(&self) -> String {
        self.inner
            .lock()
            .map(|g| g.uuid.clone())
            .unwrap_or_default()
    }
    #[wasm_bindgen(setter)]
    pub fn set_uuid(&self, uuid: String) {
        if let Ok(mut g) = self.inner.lock() {
            g.uuid = uuid;
        }
    }
    /// `[objects.total, objects.inUse, bindingsPerObject]`.
    pub fn stats(&self) -> Vec<f64> {
        self.inner
            .lock()
            .map(|g| {
                vec![
                    g.total_objects() as f64,
                    g.objects_in_use() as f64,
                    g.bindings_per_object() as f64,
                ]
            })
            .unwrap_or_default()
    }
    /// The object keys, cached (removed) ones first, and how many are cached.
    pub fn objects(&self) -> Vec<u32> {
        self.inner
            .lock()
            .map(|g| g.objects().to_vec())
            .unwrap_or_default()
    }
    #[wasm_bindgen(getter, js_name = nCachedObjects)]
    pub fn n_cached_objects(&self) -> usize {
        self.inner.lock().map(|g| g.n_cached_objects()).unwrap_or(0)
    }
}

// ---- the mixer ----

fn action_id(v: f64) -> Option<three::ActionId> {
    if !(v.is_finite() && v >= 0.0) {
        return None;
    }
    Some(slotmap::KeyData::from_ffi(v as u64).into())
}

fn action_f64(id: three::ActionId) -> f64 {
    use slotmap::Key;
    id.data().as_ffi() as f64
}

/// A root from its number: a registered group, else an object. NaN: none.
fn root_of(key: f64) -> Option<MixerRoot<JsKind>> {
    if !(key.is_finite() && key >= 0.0) {
        return None;
    }
    let key = key as u32;
    match ANIM.with(|s| s.borrow().groups.get(&key).cloned()) {
        Some(g) => Some(MixerRoot::Group(g)),
        None => Some(MixerRoot::Node(key)),
    }
}

/// three.js `AnimationMixer` over numbered JS objects; actions are numbers.
#[wasm_bindgen]
pub struct WebAnimationMixer {
    inner: AnimationMixer<JsKind>,
}

/// Action fields for [`WebAnimationMixer::action_get`] / `action_set`.
const F_TIME: u32 = 0;
const F_TIME_SCALE: u32 = 1;
const F_WEIGHT: u32 = 2;
const F_ENABLED: u32 = 3;
const F_PAUSED: u32 = 4;
const F_LOOP: u32 = 5;
const F_REPETITIONS: u32 = 6;
const F_CLAMP: u32 = 7;
const F_ZERO_START: u32 = 8;
const F_ZERO_END: u32 = 9;
const F_BLEND: u32 = 10;
const F_EFFECTIVE_WEIGHT: u32 = 11;
const F_EFFECTIVE_TIME_SCALE: u32 = 12;
const F_RUNNING: u32 = 13;
const F_SCHEDULED: u32 = 14;

#[wasm_bindgen]
impl WebAnimationMixer {
    /// A mixer for root `key` (an object's number, or a group's).
    #[wasm_bindgen(constructor)]
    pub fn new(key: f64) -> WebAnimationMixer {
        let root = root_of(key).unwrap_or(MixerRoot::Node(0));
        WebAnimationMixer {
            inner: AnimationMixer::new(root),
        }
    }

    /// `clipAction(clip, optionalRoot, blendMode)`: the action's number, or
    /// -1 for `null`. `root_key` NaN: the mixer's root. `blend_mode` NaN:
    /// the clip's.
    #[wasm_bindgen(js_name = clipAction)]
    pub fn clip_action(
        &mut self,
        clip: &WebAnimClip,
        root_key: f64,
        blend_mode: f64,
    ) -> Result<f64, JsValue> {
        self.clip_action_ref(ClipRef::Clip(clip.inner.clone()), root_key, blend_mode)
    }

    /// `clipAction(uuid, …)` for a clip the mixer already knows by uuid
    /// (three.js's string form, when the root has no clip of that name).
    #[wasm_bindgen(js_name = clipActionByUuid)]
    pub fn clip_action_by_uuid(
        &mut self,
        clip_uuid: &str,
        root_key: f64,
        blend_mode: f64,
    ) -> Result<f64, JsValue> {
        self.clip_action_ref(ClipRef::Uuid(clip_uuid.to_string()), root_key, blend_mode)
    }

    fn clip_action_ref(
        &mut self,
        clip: ClipRef,
        root_key: f64,
        blend_mode: f64,
    ) -> Result<f64, JsValue> {
        let root = root_of(root_key);
        let blend = blend_mode.is_finite().then_some(blend_mode as u32);
        match self.inner.clip_action(&mut JsHost, clip, root, blend) {
            Ok(Some(id)) => Ok(action_f64(id)),
            Ok(None) => Ok(-1.0),
            Err(e) => Err(js_sys::Error::new(&e.to_string()).into()),
        }
    }

    /// `existingAction(clip, optionalRoot)`: the action's number or -1.
    #[wasm_bindgen(js_name = existingAction)]
    pub fn existing_action(&self, clip_uuid: &str, root_key: f64) -> f64 {
        let root = root_of(root_key);
        self.inner
            .existing_action(&JsHost, clip_uuid, root.as_ref())
            .map(action_f64)
            .unwrap_or(-1.0)
    }

    pub fn play(&mut self, id: f64) {
        if let Some(id) = action_id(id) {
            self.inner.play(&mut JsHost, id);
        }
    }
    pub fn stop(&mut self, id: f64) {
        if let Some(id) = action_id(id) {
            self.inner.stop(&mut JsHost, id);
        }
    }
    pub fn reset(&mut self, id: f64) {
        if let Some(id) = action_id(id) {
            self.inner.reset(id);
        }
    }
    #[wasm_bindgen(js_name = fadeIn)]
    pub fn fade_in(&mut self, id: f64, duration: f64) {
        if let Some(id) = action_id(id) {
            self.inner.fade_in(id, duration);
        }
    }
    #[wasm_bindgen(js_name = fadeOut)]
    pub fn fade_out(&mut self, id: f64, duration: f64) {
        if let Some(id) = action_id(id) {
            self.inner.fade_out(id, duration);
        }
    }
    #[wasm_bindgen(js_name = crossFadeFrom)]
    pub fn cross_fade_from(&mut self, id: f64, other: f64, duration: f64, warp: bool) {
        if let (Some(id), Some(other)) = (action_id(id), action_id(other)) {
            self.inner.cross_fade_from(id, other, duration, warp);
        }
    }
    #[wasm_bindgen(js_name = stopFading)]
    pub fn stop_fading(&mut self, id: f64) {
        if let Some(id) = action_id(id) {
            self.inner.stop_fading(id);
        }
    }
    #[wasm_bindgen(js_name = setEffectiveWeight)]
    pub fn set_effective_weight(&mut self, id: f64, w: f64) {
        if let Some(id) = action_id(id) {
            self.inner.set_effective_weight(id, w);
        }
    }
    #[wasm_bindgen(js_name = setEffectiveTimeScale)]
    pub fn set_effective_time_scale(&mut self, id: f64, s: f64) {
        if let Some(id) = action_id(id) {
            self.inner.set_effective_time_scale(id, s);
        }
    }
    #[wasm_bindgen(js_name = setDuration)]
    pub fn set_duration(&mut self, id: f64, d: f64) {
        if let Some(id) = action_id(id) {
            self.inner.set_duration(id, d);
        }
    }
    #[wasm_bindgen(js_name = syncWith)]
    pub fn sync_with(&mut self, id: f64, other: f64) {
        if let (Some(id), Some(other)) = (action_id(id), action_id(other)) {
            self.inner.sync_with(id, other);
        }
    }
    pub fn halt(&mut self, id: f64, d: f64) {
        if let Some(id) = action_id(id) {
            self.inner.halt(id, d);
        }
    }
    pub fn warp(&mut self, id: f64, start: f64, end: f64, d: f64) {
        if let Some(id) = action_id(id) {
            self.inner.warp(id, start, end, d);
        }
    }
    #[wasm_bindgen(js_name = stopWarping)]
    pub fn stop_warping(&mut self, id: f64) {
        if let Some(id) = action_id(id) {
            self.inner.stop_warping(id);
        }
    }
    #[wasm_bindgen(js_name = startAt)]
    pub fn start_at(&mut self, id: f64, t: f64) {
        if let Some(a) = action_id(id).and_then(|id| self.inner.action_mut(id)) {
            a.start_at(t);
        }
    }
    /// `_startTime`, NaN for none.
    #[wasm_bindgen(js_name = startTime)]
    pub fn start_time(&self, id: f64) -> f64 {
        action_id(id)
            .and_then(|id| self.inner.action(id))
            .and_then(|a| a.start_time())
            .unwrap_or(f64::NAN)
    }

    /// An action's field: 0 time, 1 timeScale, 2 weight, 3 enabled,
    /// 4 paused, 5 loop, 6 repetitions, 7 clampWhenFinished,
    /// 8 zeroSlopeAtStart, 9 zeroSlopeAtEnd, 10 blendMode, 11 effective
    /// weight, 12 effective time scale, 13 isRunning, 14 isScheduled.
    #[wasm_bindgen(js_name = actionGet)]
    pub fn action_get(&self, id: f64, field: u32) -> f64 {
        let Some(id) = action_id(id) else {
            return f64::NAN;
        };
        let Some(a) = self.inner.action(id) else {
            return f64::NAN;
        };
        let b = |v: bool| v as u8 as f64;
        match field {
            F_TIME => a.time,
            F_TIME_SCALE => a.time_scale,
            F_WEIGHT => a.weight,
            F_ENABLED => b(a.enabled),
            F_PAUSED => b(a.paused),
            F_LOOP => a.loop_mode as f64,
            F_REPETITIONS => a.repetitions,
            F_CLAMP => b(a.clamp_when_finished),
            F_ZERO_START => b(a.zero_slope_at_start),
            F_ZERO_END => b(a.zero_slope_at_end),
            F_BLEND => a.blend_mode as f64,
            F_EFFECTIVE_WEIGHT => a.effective_weight(),
            F_EFFECTIVE_TIME_SCALE => a.effective_time_scale(),
            F_RUNNING => b(self.inner.is_running(id)),
            F_SCHEDULED => b(self.inner.is_scheduled(id)),
            _ => f64::NAN,
        }
    }

    /// Set an action's field (see [`action_get`](Self::action_get)).
    #[wasm_bindgen(js_name = actionSet)]
    pub fn action_set(&mut self, id: f64, field: u32, value: f64) {
        let Some(a) = action_id(id).and_then(|id| self.inner.action_mut(id)) else {
            return;
        };
        let t = value != 0.0 && !value.is_nan();
        match field {
            F_TIME => a.time = value,
            F_TIME_SCALE => a.time_scale = value,
            F_WEIGHT => a.weight = value,
            F_ENABLED => a.enabled = t,
            F_PAUSED => a.paused = t,
            F_LOOP => a.loop_mode = value as u32,
            F_REPETITIONS => a.repetitions = value,
            F_CLAMP => a.clamp_when_finished = t,
            F_ZERO_START => a.zero_slope_at_start = t,
            F_ZERO_END => a.zero_slope_at_end = t,
            F_BLEND => a.blend_mode = value as u32,
            _ => {}
        }
    }

    /// Start `update(deltaTime)`; then call `resume` until it reports no event.
    #[wasm_bindgen(js_name = beginUpdate)]
    pub fn begin_update(&mut self, delta: f64) {
        self.inner.begin_update(delta);
    }
    /// Start `setTime(t)`.
    #[wasm_bindgen(js_name = beginSetTime)]
    pub fn begin_set_time(&mut self, t: f64) {
        self.inner.begin_set_time(t);
    }
    /// Run to the next event: `[kind, action, value]` — kind 1
    /// `'finished'` (value: direction), 2 `'loop'` (value: loopDelta), 0
    /// done (the bindings were applied).
    pub fn resume(&mut self) -> Vec<f64> {
        match self.inner.resume(&mut JsHost) {
            None => vec![0.0, -1.0, 0.0],
            Some(e) => match e.kind {
                three::ActionEvent::Finished { direction } => {
                    vec![1.0, action_f64(e.action), direction]
                }
                three::ActionEvent::Loop { loop_delta } => {
                    vec![2.0, action_f64(e.action), loop_delta]
                }
            },
        }
    }
    /// `update(deltaTime)` when no listener is waiting: runs to the end and
    /// returns the events, three numbers each.
    #[wasm_bindgen(js_name = updateAll)]
    pub fn update_all(&mut self, delta: f64) -> Vec<f64> {
        let events = self.inner.update(&mut JsHost, delta);
        let mut out = Vec::with_capacity(events.len() * 3);
        for e in events {
            match e.kind {
                three::ActionEvent::Finished { direction } => {
                    out.extend([1.0, action_f64(e.action), direction])
                }
                three::ActionEvent::Loop { loop_delta } => {
                    out.extend([2.0, action_f64(e.action), loop_delta])
                }
            }
        }
        out
    }

    #[wasm_bindgen(js_name = stopAllAction)]
    pub fn stop_all_action(&mut self) {
        self.inner.stop_all_action(&mut JsHost);
    }
    #[wasm_bindgen(js_name = uncacheClip)]
    pub fn uncache_clip(&mut self, clip_uuid: &str) {
        self.inner.uncache_clip(&mut JsHost, clip_uuid);
    }
    #[wasm_bindgen(js_name = uncacheRoot)]
    pub fn uncache_root(&mut self, root_uuid: &str) {
        self.inner.uncache_root(&mut JsHost, root_uuid);
    }
    #[wasm_bindgen(js_name = uncacheAction)]
    pub fn uncache_action(&mut self, clip_uuid: &str, root_key: f64) {
        let root = root_of(root_key);
        self.inner
            .uncache_action(&mut JsHost, clip_uuid, root.as_ref());
    }
    /// Drop an action JS no longer holds (no-op while cached).
    #[wasm_bindgen(js_name = forgetAction)]
    pub fn forget_action(&mut self, id: f64) -> bool {
        action_id(id)
            .map(|id| self.inner.forget_action(&mut JsHost, id))
            .unwrap_or(false)
    }
    /// The cached actions, active first.
    #[wasm_bindgen(js_name = cachedActions)]
    pub fn cached_actions(&self) -> Vec<f64> {
        self.inner
            .cached_actions()
            .iter()
            .map(|&id| action_f64(id))
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time
    }
    #[wasm_bindgen(setter)]
    pub fn set_time(&mut self, t: f64) {
        self.inner.time = t;
    }
    #[wasm_bindgen(getter, js_name = timeScale)]
    pub fn time_scale(&self) -> f64 {
        self.inner.time_scale
    }
    #[wasm_bindgen(setter, js_name = timeScale)]
    pub fn set_time_scale(&mut self, s: f64) {
        self.inner.time_scale = s;
    }
    /// `[actions.total, actions.inUse, bindings.total, bindings.inUse,
    /// controlInterpolants.total, controlInterpolants.inUse]`.
    pub fn stats(&self) -> Vec<f64> {
        let s = self.inner.stats();
        [
            s.actions_total,
            s.actions_in_use,
            s.bindings_total,
            s.bindings_in_use,
            s.control_interpolants_total,
            s.control_interpolants_in_use,
        ]
        .map(|v| v as f64)
        .to_vec()
    }
}

/// `PropertyBinding.parseTrackName(name)`: `[directoryName, nodeName,
/// objectName, objectIndex, propertyName, propertyIndex]` (absent parts
/// `undefined`), or an error to throw.
#[wasm_bindgen(js_name = parseTrackName)]
pub fn parse_track_name(name: &str) -> Result<js_sys::Array, JsValue> {
    let p = three::parse_track_name(name).map_err(|e| js_sys::Error::new(&e.to_string()))?;
    let a = js_sys::Array::new();
    a.push(&JsValue::from_str(&p.directory_name));
    a.push(&opt_js(&p.node_name));
    a.push(&opt_js(&p.object_name));
    a.push(&opt_js(&p.object_index));
    a.push(&JsValue::from_str(&p.property_name));
    a.push(&opt_js(&p.property_index));
    Ok(a)
}

/// `PropertyBinding.sanitizeNodeName(name)`.
#[wasm_bindgen(js_name = sanitizeNodeName)]
pub fn sanitize_node_name(name: &str) -> String {
    three::sanitize_node_name(name)
}

/// `AnimationUtils.getKeyframeOrder(times)`.
#[wasm_bindgen(js_name = getKeyframeOrder)]
pub fn get_keyframe_order(times: Vec<f64>) -> Vec<u32> {
    three::utils::get_keyframe_order(&times)
        .into_iter()
        .map(|i| i as u32)
        .collect()
}
