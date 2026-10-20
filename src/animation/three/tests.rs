//! The port against three.js r165 itself: `tests/animation/gen-three-truth.mjs`
//! runs three.js on track names, interpolant sweeps, track and clip edits,
//! `AnimationUtils`, and whole mixer scenarios (`scenarios.mjs`), and these
//! tests replay the same inputs here and compare — values to 1e-6 relative,
//! events and their order exactly.
//!
//! The scenarios run on [`JsHost`], a model of the JS objects three.js binds
//! to (numbers, booleans, strings, arrays, objects with `fromArray` /
//! `toArray`, materials, skeletons, morph dictionaries), resolving paths the
//! way `PropertyBinding.bind()` does.

use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use crate::loaders::json::{self, Value};

const TRUTH: &str = include_str!("../../../tests/animation/three-r165-truth.json");

fn truth() -> Value {
    json::parse(TRUTH).expect("three-r165-truth.json parses")
}

// ---- JSON helpers ----

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => *n,
        Value::Null => f64::NAN,
        Value::Bool(b) => *b as u8 as f64,
        Value::Object(o) => match o.get("$num").and_then(|s| s.as_str()) {
            Some("Infinity") => f64::INFINITY,
            Some("-Infinity") => f64::NEG_INFINITY,
            _ => f64::NAN,
        },
        _ => f64::NAN,
    }
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().map(num).collect())
        .unwrap_or_default()
}
fn opt_str(v: Option<&Value>) -> Option<String> {
    v.and_then(|v| v.as_str()).map(str::to_string)
}
fn arr(v: &Value) -> &Vec<Value> {
    v.as_array().expect("array")
}
fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing key {k}"))
}

fn near(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= f64::max(1e-9, 1e-6 * a.abs().max(b.abs()))
}

fn assert_near_all(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            near(*x, *y),
            "{what}[{i}]: rust {x} vs three {y} (all: {a:?} vs {b:?})"
        );
    }
}

/// Compare two JSON values structurally, numbers to 1e-6 relative.
fn assert_json_near(ours: &Value, theirs: &Value, path: &str) {
    match (ours, theirs) {
        (Value::Object(a), Value::Object(b)) => {
            let mut ka: Vec<_> = a.keys().collect();
            let mut kb: Vec<_> = b.keys().collect();
            ka.sort();
            kb.sort();
            assert_eq!(ka, kb, "{path}: keys");
            for k in ka {
                if k == "uuid" {
                    continue;
                }
                assert_json_near(&a[k], &b[k], &format!("{path}.{k}"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: length");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_json_near(x, y, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(_), _) | (_, Value::Number(_)) | (Value::Null, _) | (_, Value::Null) => {
            assert!(
                near(num(ours), num(theirs)),
                "{path}: rust {ours:?} vs three {theirs:?}"
            );
        }
        (a, b) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{path}"),
    }
}

fn clip_json(c: &AnimationClip) -> Value {
    json::parse(&c.to_json_string()).expect("our JSON parses")
}

fn track_from_json(t: &Value) -> KeyframeTrack {
    let ty = ValueType::from_name(get(t, "type").as_str().unwrap()).unwrap();
    let name = get(t, "name").as_str().unwrap().to_string();
    let times: Vec<f32> = nums(get(t, "times")).iter().map(|&x| x as f32).collect();
    let vals = get(t, "values");
    let values = match ty {
        ValueType::Bool => TrackValues::Bool(
            arr(vals)
                .iter()
                .map(|v| v.as_bool().unwrap_or(false))
                .collect(),
        ),
        ValueType::String => TrackValues::String(
            arr(vals)
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
        ),
        _ => TrackValues::Numeric(nums(vals).iter().map(|&x| x as f32).collect()),
    };
    let interp = t
        .get("interpolation")
        .filter(|v| !matches!(v, Value::Null))
        .map(num)
        .and_then(|c| Interpolation::from_constant(c as u32));
    let mut track = KeyframeTrack::new(ty, name, times, values, interp).expect("valid track");
    if t.get("gltfCubic").and_then(|v| v.as_bool()) == Some(true) {
        track.set_interpolation(Interpolation::GltfCubicSpline);
    }
    track
}

// ---- unit cases ----

#[test]
fn parse_track_name_matches_three() {
    let t = truth();
    for case in arr(get(&t, "parse")) {
        let name = get(case, "name").as_str().unwrap();
        let ours = parse_track_name(name);
        if case.get("error").is_some() {
            assert!(ours.is_err(), "{name}: three.js throws, we parsed {ours:?}");
            continue;
        }
        let p = ours.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            p.node_name,
            opt_str(case.get("nodeName")),
            "{name}: nodeName"
        );
        assert_eq!(
            p.object_name,
            opt_str(case.get("objectName")),
            "{name}: objectName"
        );
        assert_eq!(
            p.object_index,
            opt_str(case.get("objectIndex")),
            "{name}: objectIndex"
        );
        assert_eq!(
            p.property_name,
            get(case, "propertyName").as_str().unwrap(),
            "{name}: propertyName"
        );
        assert_eq!(
            p.property_index,
            opt_str(case.get("propertyIndex")),
            "{name}: propertyIndex"
        );
    }
    for case in arr(get(&t, "sanitize")) {
        let c = arr(case);
        assert_eq!(
            sanitize_node_name(c[0].as_str().unwrap()),
            c[1].as_str().unwrap()
        );
    }
}

#[test]
fn interpolants_match_three() {
    let t = truth();
    for case in arr(get(&t, "interpolants")) {
        let kind = get(case, "kind").as_str().unwrap();
        let size = num(get(case, "size")) as usize;
        let times: Vec<f32> = nums(get(case, "times")).iter().map(|&x| x as f32).collect();
        let values: Vec<f32> = nums(get(case, "values"))
            .iter()
            .map(|&x| x as f32)
            .collect();
        let settings = match get(case, "settings") {
            Value::Array(s) => Some(InterpolantSettings {
                ending_start: num(&s[0]) as u32,
                ending_end: num(&s[1]) as u32,
            }),
            _ => None,
        };
        let k = match kind {
            "linear" => InterpolantKind::Linear,
            "discrete" => InterpolantKind::Discrete,
            "cubic" => InterpolantKind::Cubic,
            "quaternion" => InterpolantKind::QuaternionLinear,
            "gltf" => InterpolantKind::GltfCubicSpline,
            "gltfQuat" => InterpolantKind::GltfCubicSplineQuaternion,
            _ => unreachable!(),
        };
        let mut owned = OwnedInterpolant::new(k, times, values, size);
        owned.settings = settings;
        let ts = nums(get(case, "ts"));
        for (i, (tt, expected)) in ts.iter().zip(arr(get(case, "results"))).enumerate() {
            let r: Vec<f64> = owned.evaluate(*tt).iter().map(|&v| v as f64).collect();
            assert_near_all(
                &r,
                &nums(expected),
                &format!("{kind} {settings:?} sample {i} t={tt}"),
            );
        }
    }
}

#[test]
fn keyframe_track_edits_match_three() {
    let t = truth();
    for case in arr(get(&t, "tracks")) {
        let input = get(case, "input");
        let base = || track_from_json(input);
        let name = get(input, "name").as_str().unwrap();
        let tr = base();
        let gi = get(case, "getInterpolation");
        assert_eq!(
            tr.interpolation().constant().map(|c| c as f64),
            if matches!(gi, Value::Null) {
                None
            } else {
                Some(num(gi))
            },
            "{name}: getInterpolation"
        );
        assert!(
            near(tr.value_size_f64(), num(get(case, "valueSize"))),
            "{name}: value size"
        );
        assert_eq!(
            tr.validate(),
            get(case, "validate").as_bool().unwrap(),
            "{name}: validate"
        );
        let j = |t: &KeyframeTrack| json::parse(&track_to_json_string(t)).unwrap();
        assert_json_near(&j(&tr), get(case, "json"), &format!("{name} json"));
        let Some(opt) = case.get("optimize") else {
            continue;
        };
        assert_json_near(&j(base().optimize()), opt, &format!("{name} optimize"));
        for trim in arr(get(case, "trim")) {
            let a = nums(get(trim, "args"));
            assert_json_near(
                &j(base().trim(a[0], a[1])),
                get(trim, "out"),
                &format!("{name} trim {a:?}"),
            );
        }
        assert_json_near(
            &j(base().shift(0.123456789)),
            get(case, "shift"),
            &format!("{name} shift"),
        );
        assert_json_near(
            &j(base().scale(1.0 / 30.0)),
            get(case, "scale"),
            &format!("{name} scale"),
        );
    }
}

fn clip_from_json(c: &Value) -> AnimationClip {
    let tracks = arr(get(c, "tracks")).iter().map(track_from_json).collect();
    let mut clip = AnimationClip::with_blend_mode(
        get(c, "name").as_str().unwrap_or(""),
        num(get(c, "duration")),
        tracks,
        c.get("blendMode")
            .map(num)
            .map(|b| b as u32)
            .unwrap_or(NORMAL_ANIMATION_BLEND_MODE),
    );
    clip.uuid = opt_str(c.get("uuid")).unwrap_or_default();
    clip
}

#[test]
fn clips_and_utils_match_three() {
    let t = truth();
    let c = get(&t, "clips");
    let tracks = || {
        vec![
            KeyframeTrack::vector(
                "a.position",
                vec![0.0, 0.5, 2.0],
                vec![0., 0., 0., 1., 2., 3., 4., 5., 6.],
            ),
            KeyframeTrack::quaternion(
                "a.quaternion",
                vec![0.25, 1.5],
                vec![0., 0., 0., 1., 0., 0.6, 0., 0.8],
            ),
            KeyframeTrack::number("b.x", vec![0.0, 0.1, 0.2, 3.0], vec![1., 1., 1., 2.])
                .with_interpolation(Interpolation::Smooth),
        ]
    };
    let walk = AnimationClip::new("walk", -1.0, tracks());
    assert!(near(walk.duration, num(get(c, "resetDuration"))));
    assert_json_near(&clip_json(&walk), get(c, "toJSON"), "toJSON");
    assert_json_near(
        &clip_json(AnimationClip::new("t", 1.0, tracks()).trim()),
        get(c, "trim"),
        "trim",
    );
    assert_json_near(
        &clip_json(AnimationClip::new("o", -1.0, tracks()).optimize()),
        get(c, "optimize"),
        "optimize",
    );
    assert_eq!(
        AnimationClip::new("v", -1.0, tracks()).validate(),
        get(c, "validate").as_bool().unwrap()
    );

    let parsed = AnimationClip::parse(get(c, "parseIn")).unwrap();
    assert_eq!(parsed.uuid, "FIXED-UUID");
    assert_json_near(&clip_json(&parsed), get(c, "parse"), "parse");
    let nd = AnimationClip::parse_json_str(
        r#"{"name":"nd","tracks":[{"name":".x","type":"number","times":[0,4],"values":[0,1]}]}"#,
    )
    .unwrap();
    assert_json_near(
        &clip_json(&nd),
        get(c, "parseNoDuration"),
        "parse without duration",
    );

    let names: Vec<String> = arr(get(c, "morphNames"))
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    for (i, no_loop) in [false, true].into_iter().enumerate() {
        let seq =
            AnimationClip::create_from_morph_target_sequence("seq", &names[..4], 24.0, no_loop);
        assert_json_near(
            &clip_json(&seq),
            &arr(get(c, "morphSeq"))[i],
            &format!("morph sequence noLoop={no_loop}"),
        );
    }
    let clips = AnimationClip::create_clips_from_morph_target_sequences(&names, 30.0, false);
    let expected = arr(get(c, "morphClips"));
    assert_eq!(clips.len(), expected.len(), "morph clip count");
    for (ours, theirs) in clips.iter().zip(expected) {
        assert_json_near(
            &clip_json(ours),
            theirs,
            &format!("morph clip {}", ours.name),
        );
    }

    let src = clip_from_json(get(c, "subclipSrc"));
    for case in arr(get(c, "subclip")) {
        let a = nums(get(case, "args"));
        let sub = utils::subclip(&src, "sub", a[0], a[1], a[2]);
        assert_json_near(
            &clip_json(&sub),
            get(case, "out"),
            &format!("subclip {a:?}"),
        );
    }

    for case in arr(get(c, "makeClipAdditive")) {
        let a = arr(get(case, "args"));
        let (frame, fps, use_ref) = (num(&a[0]), num(&a[1]), a[2].as_bool().unwrap());
        let gltf = case.get("gltf").is_some();
        let mut target = if gltf {
            let mut vals = Vec::new();
            for k in 0..3 {
                for p in 0..3 {
                    for j in 0..3 {
                        vals.push((k as f64 + p as f64 * 0.1 + j as f64 * 0.01) as f32);
                    }
                }
            }
            let quats = [[0.0, 0.6, 0.0, 0.8f32], [0.5, 0.5, 0.5, 0.5]];
            let mut qv = Vec::new();
            for q in quats {
                for p in 0..3 {
                    if p == 1 {
                        qv.extend(q);
                    } else {
                        qv.extend([0.01, 0.02, 0.03, 0.04]);
                    }
                }
            }
            AnimationClip::new(
                "gltf",
                -1.0,
                vec![
                    KeyframeTrack::vector("a.position", vec![0.0, 0.25, 0.5], vals)
                        .with_interpolation(Interpolation::GltfCubicSpline),
                    KeyframeTrack::quaternion("a.quaternion", vec![0.0, 0.5], qv)
                        .with_interpolation(Interpolation::GltfCubicSpline),
                ],
            )
        } else {
            AnimationClip::new(
                "tgt",
                -1.0,
                vec![
                    KeyframeTrack::vector(
                        "a.position",
                        vec![0.0, 0.25, 0.5],
                        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                    ),
                    KeyframeTrack::quaternion(
                        "a.quaternion",
                        vec![0.0, 0.5],
                        vec![0., 0., 0., 1., 0., 0.6, 0., 0.8],
                    ),
                    KeyframeTrack::boolean("a.visible", vec![0.0], vec![true]),
                    KeyframeTrack::number("a.s", vec![0.0, 0.1, 0.2], vec![1., 2., 4.])
                        .with_interpolation(Interpolation::Smooth),
                ],
            )
        };
        let reference = use_ref.then(|| {
            AnimationClip::new(
                "ref",
                -1.0,
                vec![
                    KeyframeTrack::vector(
                        "a.position",
                        vec![0.0, 1.0],
                        vec![10., 10., 10., 20., 20., 20.],
                    ),
                    KeyframeTrack::quaternion(
                        "a.quaternion",
                        vec![0.0, 1.0],
                        vec![0., 0., 0.6, 0.8, 0., 0., 0., 1.],
                    ),
                ],
            )
        });
        utils::make_clip_additive(&mut target, frame, reference.as_ref(), fps);
        assert_json_near(
            &clip_json(&target),
            get(case, "out"),
            &format!("makeClipAdditive {frame} {fps} {use_ref} gltf={gltf}"),
        );
    }
    let order = utils::get_keyframe_order(&[3.0, 1.0, 2.0, 1.0, 0.5, f64::NAN, 2.0]);
    assert_eq!(
        order.iter().map(|&i| i as f64).collect::<Vec<_>>(),
        nums(get(c, "keyframeOrder"))
    );
    let sorted = utils::sorted_array(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], 2, &[2, 0, 1]);
    assert_eq!(sorted, nums(get(c, "sortedArray")));
}

// ---- a JS object model to bind to ----

#[derive(Debug, Clone, PartialEq)]
enum JsVal {
    Num(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<f64>),
    /// An object with `fromArray` / `toArray` (Vector3, Quaternion, Color).
    Math(Vec<f64>),
    Obj(usize),
}

#[derive(Debug, Default, Clone)]
struct JsObj {
    name: String,
    uuid: String,
    props: HashMap<String, JsVal>,
    children: Vec<usize>,
    skeleton: Option<Vec<usize>>,
    morph_dictionary: HashMap<String, usize>,
    has_morph_geometry: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct JsKind;

#[derive(Debug, Clone, PartialEq)]
enum Access {
    Direct,
    EntireArray,
    ArrayElement(usize),
    FromToArray,
}

#[derive(Debug, Clone, PartialEq)]
struct JsTarget {
    obj: usize,
    prop: String,
    access: Access,
    bool_track: bool,
}

impl HostKind for JsKind {
    type Node = usize;
    type Target = JsTarget;
}

#[derive(Default)]
struct JsHost {
    objs: Vec<JsObj>,
}

impl NodeTree for JsHost {
    type Id = usize;
    fn name_is(&self, n: &usize, name: &str) -> bool {
        self.objs[*n].name == name
    }
    fn uuid_is(&self, n: &usize, uuid: &str) -> bool {
        self.objs[*n].uuid == uuid
    }
    fn skeleton_bones(&self, n: &usize) -> Option<Vec<usize>> {
        self.objs[*n].skeleton.clone()
    }
    fn children(&self, n: &usize) -> Vec<usize> {
        self.objs[*n].children.clone()
    }
}

impl AnimationHost<JsKind> for JsHost {
    fn node_uuid(&self, node: &usize) -> String {
        self.objs[*node].uuid.clone()
    }
    fn find_node(&mut self, root: &usize, name: Option<&str>) -> Option<usize> {
        find_node(&*self, root, name)
    }
    fn bind(
        &mut self,
        node: &usize,
        p: &ParsedPath,
        _path: &str,
        ty: ValueType,
        _size: usize,
    ) -> Option<JsTarget> {
        let mut target = *node;
        if let Some(object_name) = &p.object_name {
            let mut object_index = p.object_index.clone();
            match object_name.as_str() {
                "bones" => {
                    let bones = self.objs[target].skeleton.clone()?;
                    let idx = object_index.clone().unwrap_or_default();
                    let found = bones
                        .iter()
                        .copied()
                        .find(|b| self.objs[*b].name == idx)
                        .or_else(|| {
                            idx.parse::<usize>()
                                .ok()
                                .and_then(|i| bones.get(i).copied())
                        })?;
                    target = found;
                    object_index = None;
                }
                "materials" | "map" => return None,
                other => match self.objs[target].props.get(other) {
                    Some(JsVal::Obj(o)) => target = *o,
                    _ => return None,
                },
            }
            if let Some(i) = object_index {
                match self.objs[target].props.get(&i) {
                    Some(JsVal::Obj(o)) => target = *o,
                    _ => return None,
                }
            }
        }
        let prop = p.property_name.clone();
        let value = self.objs[target].props.get(&prop)?.clone();
        let access = if let Some(pi) = &p.property_index {
            let mut index = pi.clone();
            if prop == "morphTargetInfluences" {
                if !self.objs[target].has_morph_geometry {
                    return None;
                }
                if let Some(i) = self.objs[target].morph_dictionary.get(pi) {
                    index = i.to_string();
                }
            }
            Access::ArrayElement(index.parse().ok()?)
        } else {
            match value {
                JsVal::Math(_) => Access::FromToArray,
                JsVal::Arr(_) => Access::EntireArray,
                _ => Access::Direct,
            }
        };
        Some(JsTarget {
            obj: target,
            prop,
            access,
            bool_track: ty == ValueType::Bool,
        })
    }
    fn get_value(&mut self, t: &JsTarget, out: &mut [f64]) {
        let v = &self.objs[t.obj].props[&t.prop];
        let mut put = |vals: &[f64]| {
            for (o, v) in out.iter_mut().zip(vals) {
                *o = *v;
            }
        };
        match (&t.access, v) {
            (Access::Direct, JsVal::Num(n)) => put(&[*n]),
            (Access::Direct, JsVal::Bool(b)) => put(&[*b as u8 as f64]),
            (Access::EntireArray, JsVal::Arr(a)) | (Access::FromToArray, JsVal::Math(a)) => put(a),
            (Access::ArrayElement(i), JsVal::Arr(a)) => {
                put(&[a.get(*i).copied().unwrap_or(f64::NAN)])
            }
            _ => {}
        }
    }
    fn set_value(&mut self, t: &JsTarget, values: &[f64]) {
        let bool_track = t.bool_track;
        let v = self.objs[t.obj].props.get_mut(&t.prop).unwrap();
        match (&t.access, v) {
            (Access::Direct, v @ (JsVal::Num(_) | JsVal::Bool(_))) => {
                *v = if bool_track {
                    JsVal::Bool(values[0] != 0.0)
                } else {
                    JsVal::Num(values[0])
                };
            }
            (Access::EntireArray, JsVal::Arr(a)) | (Access::FromToArray, JsVal::Math(a)) => {
                for (i, d) in a.iter_mut().enumerate() {
                    *d = values.get(i).copied().unwrap_or(f64::NAN);
                }
            }
            (Access::ArrayElement(i), JsVal::Arr(a)) => {
                if let Some(d) = a.get_mut(*i) {
                    *d = values[0];
                }
            }
            _ => {}
        }
    }
    fn get_strings(&mut self, t: &JsTarget, out: &mut [Option<String>]) {
        if let Some(JsVal::Str(s)) = self.objs[t.obj].props.get(&t.prop) {
            out[0] = Some(s.clone());
        }
    }
    fn set_strings(&mut self, t: &JsTarget, values: &[Option<String>]) {
        if let Some(v) = self.objs[t.obj].props.get_mut(&t.prop) {
            *v = JsVal::Str(values[0].clone().unwrap_or_default());
        }
    }
}

impl JsHost {
    fn add(&mut self, o: JsObj) -> usize {
        self.objs.push(o);
        self.objs.len() - 1
    }

    /// `buildTree` in scenarios.mjs.
    fn build(
        &mut self,
        spec: &Value,
        paths: &mut HashMap<String, usize>,
        parent: &str,
        skins: &mut Vec<(usize, Vec<String>)>,
    ) -> usize {
        let name = get(spec, "name").as_str().unwrap().to_string();
        let mut o = JsObj {
            name: name.clone(),
            uuid: opt_str(spec.get("uuid")).unwrap_or_else(generate_uuid),
            ..Default::default()
        };
        let v3 = |k: &str, d: [f64; 3]| spec.get(k).map(nums).unwrap_or(d.to_vec());
        o.props
            .insert("position".into(), JsVal::Math(v3("position", [0.0; 3])));
        o.props
            .insert("scale".into(), JsVal::Math(v3("scale", [1.0; 3])));
        o.props.insert(
            "quaternion".into(),
            JsVal::Math(
                spec.get("quaternion")
                    .map(nums)
                    .unwrap_or(vec![0.0, 0.0, 0.0, 1.0]),
            ),
        );
        o.props.insert("visible".into(), JsVal::Bool(true));
        let kind = opt_str(spec.get("kind"));
        if matches!(kind.as_deref(), Some("mesh") | Some("skinned")) {
            let m = spec.get("material");
            let color = m.map(|m| nums(get(m, "color"))).unwrap_or(vec![1.0; 3]);
            let opacity = m.map(|m| num(get(m, "opacity"))).unwrap_or(1.0);
            let mut mat = JsObj::default();
            mat.props.insert("color".into(), JsVal::Math(color));
            mat.props.insert("opacity".into(), JsVal::Num(opacity));
            let mi = self.add(mat);
            o.props.insert("material".into(), JsVal::Obj(mi));
            if let Some(targets) = spec.get("morphTargets") {
                let names: Vec<String> = arr(targets)
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                o.props.insert(
                    "morphTargetInfluences".into(),
                    JsVal::Arr(vec![0.0; names.len()]),
                );
                o.morph_dictionary = names.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
                o.has_morph_geometry = true;
            }
        }
        if let Some(Value::Object(custom)) = spec.get("custom") {
            for (k, v) in custom {
                let val = match v {
                    Value::Number(n) => JsVal::Num(*n),
                    Value::Object(m) => {
                        if let Some(b) = m.get("bool") {
                            JsVal::Bool(b.as_bool().unwrap())
                        } else if let Some(s) = m.get("string") {
                            JsVal::Str(s.as_str().unwrap().to_string())
                        } else if let Some(a) = m.get("array") {
                            JsVal::Arr(nums(a))
                        } else if let Some(c) = m.get("color").or(m.get("vector3")) {
                            JsVal::Math(nums(c))
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };
                o.props.insert(k.clone(), val);
            }
        }
        let id = self.add(o);
        let path = if paths.contains_key(&name) {
            format!("{parent}/{name}")
        } else {
            name.clone()
        };
        paths.insert(path, id);
        if kind.as_deref() == Some("skinned") {
            skins.push((
                id,
                arr(get(spec, "bones"))
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect(),
            ));
        }
        for c in spec
            .get("children")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let ci = self.build(&c, paths, &name, skins);
            self.objs[id].children.push(ci);
        }
        id
    }

    fn read(&self, node: usize, path: &str) -> Value {
        let mut o = node;
        let parts: Vec<&str> = path.split('.').collect();
        for p in &parts[..parts.len() - 1] {
            match self.objs[o].props.get(*p) {
                Some(JsVal::Obj(x)) => o = *x,
                _ => return Value::Null,
            }
        }
        let n = |v: &[f64]| Value::Array(v.iter().map(|&x| Value::Number(x)).collect());
        match self.objs[o].props.get(parts[parts.len() - 1]) {
            Some(JsVal::Num(x)) => n(&[*x]),
            Some(JsVal::Bool(b)) => n(&[*b as u8 as f64]),
            Some(JsVal::Str(s)) => Value::String(s.clone()),
            Some(JsVal::Arr(a)) | Some(JsVal::Math(a)) => n(a),
            _ => Value::Null,
        }
    }
}

// ---- scenario replay ----

struct Replay {
    host: JsHost,
    nodes: HashMap<String, usize>,
    root: usize,
    clips: Vec<Arc<AnimationClip>>,
    mixer: Option<AnimationMixer<JsKind>>,
    group: Option<GroupRef<JsKind>>,
    vars: HashMap<String, Option<ActionId>>,
    order: Vec<String>,
    frames: Vec<Value>,
    pending_events: Vec<Value>,
    listeners: HashMap<String, Vec<Value>>,
    depth: usize,
}

impl Replay {
    fn var_of(&self, id: ActionId) -> Value {
        for name in &self.order {
            if self.vars.get(name) == Some(&Some(id)) {
                return Value::String(name.clone());
            }
        }
        Value::Null
    }

    fn bind_var(&mut self, name: &str, id: Option<ActionId>) {
        self.vars.insert(name.to_string(), id);
        if !self.order.iter().any(|n| n == name) {
            self.order.push(name.to_string());
        }
    }

    fn record(&mut self, label: &str, record: &[Value]) {
        let props: Vec<Value> = record
            .iter()
            .map(|r| {
                let r = arr(r);
                self.host
                    .read(self.nodes[r[0].as_str().unwrap()], r[1].as_str().unwrap())
            })
            .collect();
        let mixer = self.mixer.as_ref().unwrap();
        let n = |v: f64| Value::Number(v);
        let b = |v: bool| Value::Number(v as u8 as f64);
        let actions: Vec<Value> = self
            .order
            .iter()
            .map(|name| match self.vars[name] {
                None => Value::Null,
                Some(id) => {
                    let a = mixer.action(id).unwrap();
                    Value::Array(vec![
                        n(a.time),
                        n(a.effective_weight()),
                        n(a.effective_time_scale()),
                        b(a.enabled),
                        b(a.paused),
                        b(mixer.is_running(id)),
                        b(mixer.is_scheduled(id)),
                        n(a.time_scale),
                        n(a.weight),
                    ])
                }
            })
            .collect();
        let s = mixer.stats();
        let stats = [
            s.actions_total,
            s.actions_in_use,
            s.bindings_total,
            s.bindings_in_use,
            s.control_interpolants_total,
            s.control_interpolants_in_use,
        ]
        .map(|v| n(v as f64))
        .to_vec();
        let mut frame = HashMap::new();
        frame.insert("op".to_string(), Value::String(label.to_string()));
        frame.insert("time".to_string(), n(mixer.time));
        frame.insert("props".to_string(), Value::Array(props));
        frame.insert("actions".to_string(), Value::Array(actions));
        frame.insert("stats".to_string(), Value::Array(stats));
        frame.insert(
            "events".to_string(),
            Value::Array(std::mem::take(&mut self.pending_events)),
        );
        self.frames.push(Value::Object(frame));
    }

    fn drive(&mut self, record: &[Value]) {
        loop {
            let ev = {
                let mixer = self.mixer.as_mut().unwrap();
                mixer.resume(&mut self.host)
            };
            let Some(ev) = ev else { break };
            let (kind, extra) = match ev.kind {
                ActionEvent::Finished { direction } => ("finished", direction),
                ActionEvent::Loop { loop_delta } => ("loop", loop_delta),
            };
            self.pending_events.push(Value::Array(vec![
                Value::String(kind.into()),
                self.var_of(ev.action),
                Value::Number(extra),
            ]));
            if self.depth > 4 {
                continue;
            }
            self.depth += 1;
            for op in self.listeners.get(kind).cloned().unwrap_or_default() {
                self.exec(&op, Some(ev.action), record);
            }
            self.depth -= 1;
        }
    }

    fn act(&self, v: &Value, ev: Option<ActionId>) -> Option<ActionId> {
        match v.as_str() {
            Some("$event") => ev,
            Some(name) => self.vars.get(name).copied().flatten(),
            None => None,
        }
    }

    fn root_arg(&self, v: Option<&Value>) -> Option<MixerRoot<JsKind>> {
        v.and_then(|v| v.as_str())
            .map(|n| MixerRoot::Node(self.nodes[n]))
    }

    fn exec(&mut self, op: &Value, ev: Option<ActionId>, record: &[Value]) {
        let op = arr(op);
        let kind = op[0].as_str().unwrap();
        let a = &op[1..];
        let h = &mut self.host;
        macro_rules! m {
            () => {
                self.mixer.as_mut().unwrap()
            };
        }
        match kind {
            "mixer" => {
                self.mixer = Some(AnimationMixer::new(MixerRoot::Node(
                    self.nodes[a[0].as_str().unwrap()],
                )))
            }
            "group" => {
                let objs: Vec<usize> = arr(&a[0])
                    .iter()
                    .map(|n| self.nodes[n.as_str().unwrap()])
                    .collect();
                self.group = Some(AnimationObjectGroup::new(h, objs).into_ref());
            }
            "mixerGroup" => {
                self.mixer = Some(AnimationMixer::new(MixerRoot::Group(
                    self.group.clone().unwrap(),
                )))
            }
            "groupAdd" | "groupRemove" | "groupUncache" => {
                let objs: Vec<usize> = arr(&a[0])
                    .iter()
                    .map(|n| self.nodes[n.as_str().unwrap()])
                    .collect();
                let g = self.group.clone().unwrap();
                let mut g = g.lock().unwrap();
                match kind {
                    "groupAdd" => g.add(h, &objs),
                    "groupRemove" => g.remove(h, &objs),
                    _ => g.uncache(h, &objs),
                }
            }
            "clipAction" => {
                let clip = self.clips[num(&a[1]) as usize].clone();
                let root = self.root_arg(a.get(2));
                let blend = a
                    .get(3)
                    .filter(|v| !matches!(v, Value::Null))
                    .map(|v| num(v) as u32);
                let id = m!().clip_action(&mut self.host, clip, root, blend).unwrap();
                self.bind_var(a[0].as_str().unwrap(), id);
            }
            "clipActionByName" => {
                let name = a[1].as_str().unwrap();
                let clip = AnimationClip::find_by_name(
                    &self.clips.iter().map(|c| (**c).clone()).collect::<Vec<_>>(),
                    name,
                )
                .map(|c| c.uuid.clone());
                let clip = clip.and_then(|u| self.clips.iter().find(|c| c.uuid == u).cloned());
                let id = match clip {
                    Some(c) => m!().clip_action(&mut self.host, c, None, None).unwrap(),
                    None => m!()
                        .clip_action(&mut self.host, ClipRef::Uuid(name.into()), None, None)
                        .unwrap(),
                };
                self.bind_var(a[0].as_str().unwrap(), id);
            }
            "existingAction" => {
                let uuid = self.clips[num(&a[1]) as usize].uuid.clone();
                let root = self.root_arg(a.get(2));
                let id =
                    self.mixer
                        .as_ref()
                        .unwrap()
                        .existing_action(&self.host, &uuid, root.as_ref());
                self.bind_var(a[0].as_str().unwrap(), id);
            }
            "play" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().play(&mut self.host, id);
            }
            "stop" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().stop(&mut self.host, id);
                self.record(kind, record);
            }
            "reset" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().reset(id);
            }
            "fadeIn" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().fade_in(id, num(&a[1]));
            }
            "fadeOut" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().fade_out(id, num(&a[1]));
            }
            "crossFadeTo" | "crossFadeFrom" => {
                let id = self.act(&a[0], ev).unwrap();
                let other = self.act(&a[1], ev).unwrap();
                let warp = a[3].as_bool().unwrap_or(false);
                if kind == "crossFadeTo" {
                    m!().cross_fade_to(id, other, num(&a[2]), warp);
                } else {
                    m!().cross_fade_from(id, other, num(&a[2]), warp);
                }
            }
            "warp" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().warp(id, num(&a[1]), num(&a[2]), num(&a[3]));
            }
            "halt" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().halt(id, num(&a[1]));
            }
            "startAt" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().action_mut(id).unwrap().start_at(num(&a[1]));
            }
            "setLoop" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().action_mut(id)
                    .unwrap()
                    .set_loop(num(&a[1]) as u32, num(&a[2]));
            }
            "setEffectiveWeight" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().set_effective_weight(id, num(&a[1]));
            }
            "setEffectiveTimeScale" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().set_effective_time_scale(id, num(&a[1]));
            }
            "setDuration" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().set_duration(id, num(&a[1]));
            }
            "syncWith" => {
                let id = self.act(&a[0], ev).unwrap();
                let other = self.act(&a[1], ev).unwrap();
                m!().sync_with(id, other);
            }
            "stopFading" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().stop_fading(id);
            }
            "stopWarping" => {
                let id = self.act(&a[0], ev).unwrap();
                m!().stop_warping(id);
            }
            "set" => {
                let id = self.act(&a[0], ev).unwrap();
                let act = m!().action_mut(id).unwrap();
                let v = &a[2];
                match a[1].as_str().unwrap() {
                    "time" => act.time = num(v),
                    "timeScale" => act.time_scale = num(v),
                    "weight" => act.weight = num(v),
                    "enabled" => act.enabled = v.as_bool().unwrap(),
                    "paused" => act.paused = v.as_bool().unwrap(),
                    "loop" => act.loop_mode = num(v) as u32,
                    "repetitions" => act.repetitions = num(v),
                    "clampWhenFinished" => act.clamp_when_finished = v.as_bool().unwrap(),
                    "zeroSlopeAtStart" => act.zero_slope_at_start = v.as_bool().unwrap(),
                    "zeroSlopeAtEnd" => act.zero_slope_at_end = v.as_bool().unwrap(),
                    "blendMode" => act.blend_mode = num(v) as u32,
                    f => panic!("set {f}"),
                }
            }
            "mixerTimeScale" => m!().time_scale = num(&a[0]),
            "setProp" => {
                let o = self.nodes[a[0].as_str().unwrap()];
                let p = a[1].as_str().unwrap().to_string();
                let v = match &a[2] {
                    Value::Array(_) => JsVal::Math(nums(&a[2])),
                    Value::Bool(b) => JsVal::Bool(*b),
                    other => JsVal::Num(num(other)),
                };
                self.host.objs[o].props.insert(p, v);
            }
            "update" => {
                m!().begin_update(num(&a[0]));
                self.drive(record);
                self.record(kind, record);
            }
            "setTime" => {
                m!().begin_set_time(num(&a[0]));
                self.drive(record);
                self.record(kind, record);
            }
            "stopAllAction" => {
                m!().stop_all_action(&mut self.host);
                self.record(kind, record);
            }
            "uncacheClip" => {
                let uuid = self.clips[num(&a[0]) as usize].uuid.clone();
                m!().uncache_clip(&mut self.host, &uuid);
            }
            "uncacheAction" => {
                let uuid = self.clips[num(&a[0]) as usize].uuid.clone();
                let root = self.root_arg(a.get(1));
                m!().uncache_action(&mut self.host, &uuid, root.as_ref());
            }
            "uncacheRoot" => {
                let uuid = self.host.objs[self.nodes[a[0].as_str().unwrap()]]
                    .uuid
                    .clone();
                m!().uncache_root(&mut self.host, &uuid);
            }
            "makeClipAdditive" => {
                let ti = num(&a[0]) as usize;
                let reference = match &a[2] {
                    Value::Null => None,
                    r => Some((*self.clips[num(r) as usize]).clone()),
                };
                let target = Arc::make_mut(&mut self.clips[ti]);
                utils::make_clip_additive(target, num(&a[1]), reference.as_ref(), num(&a[3]));
            }
            other => panic!("unknown op {other}"),
        }
        let _ = self.root;
    }
}

fn frame_mismatch(name: &str, i: usize, ours: &Value, theirs: &Value) -> String {
    format!(
        "{name}: frame {i} differs\n  rust:  {}\n  three: {}",
        ours.to_json(),
        theirs.to_json()
    )
}

fn compare_frame(name: &str, i: usize, ours: &Value, theirs: &Value) {
    let check = |ok: bool, what: &str| {
        assert!(ok, "{} ({what})", frame_mismatch(name, i, ours, theirs));
    };
    check(get(ours, "op").as_str() == get(theirs, "op").as_str(), "op");
    check(
        near(num(get(ours, "time")), num(get(theirs, "time"))),
        "mixer time",
    );
    for key in ["props", "actions"] {
        let (a, b) = (arr(get(ours, key)), arr(get(theirs, key)));
        check(a.len() == b.len(), key);
        for (x, y) in a.iter().zip(b) {
            match (x, y) {
                (Value::Array(_), Value::Array(_)) => {
                    let (x, y) = (nums(x), nums(y));
                    check(
                        x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| near(*p, *q)),
                        key,
                    );
                }
                (Value::String(p), Value::String(q)) => check(p == q, key),
                (Value::Null, Value::Null) => {}
                _ => check(false, key),
            }
        }
    }
    check(
        nums(get(ours, "stats")) == nums(get(theirs, "stats")),
        "stats",
    );
    let (ea, eb) = (arr(get(ours, "events")), arr(get(theirs, "events")));
    check(ea.len() == eb.len(), "event count");
    for (x, y) in ea.iter().zip(eb) {
        let (x, y) = (arr(x), arr(y));
        check(
            x[0].as_str() == y[0].as_str()
                && x[1].as_str() == y[1].as_str()
                && near(num(&x[2]), num(&y[2])),
            "events",
        );
    }
}

#[test]
fn mixer_scenarios_match_three() {
    let t = truth();
    for entry in arr(get(&t, "scenarios")) {
        let sc = get(entry, "scenario");
        let name = get(sc, "name").as_str().unwrap();
        let mut host = JsHost::default();
        let mut nodes = HashMap::new();
        let mut skins = Vec::new();
        let root = host.build(get(sc, "tree"), &mut nodes, "", &mut skins);
        for (mesh, bones) in skins {
            host.objs[mesh].skeleton = Some(bones.iter().map(|b| nodes[b]).collect());
        }
        let clips: Vec<Arc<AnimationClip>> = arr(get(sc, "clips"))
            .iter()
            .map(|c| {
                let tracks = arr(get(c, "tracks")).iter().map(track_from_json).collect();
                Arc::new(AnimationClip::with_blend_mode(
                    get(c, "name").as_str().unwrap(),
                    num(get(c, "duration")),
                    tracks,
                    c.get("blendMode")
                        .map(num)
                        .map(|b| b as u32)
                        .unwrap_or(NORMAL_ANIMATION_BLEND_MODE),
                ))
            })
            .collect();
        let mut listeners = HashMap::new();
        if let Some(Value::Object(l)) = sc.get("listeners") {
            for (k, v) in l {
                listeners.insert(k.clone(), arr(v).clone());
            }
        }
        let mut r = Replay {
            host,
            nodes,
            root,
            clips,
            mixer: None,
            group: None,
            vars: HashMap::new(),
            order: Vec::new(),
            frames: Vec::new(),
            pending_events: Vec::new(),
            listeners,
            depth: 0,
        };
        let record = arr(get(sc, "record")).clone();
        for op in arr(get(sc, "ops")) {
            r.exec(op, None, &record);
        }
        let expected = arr(get(entry, "frames"));
        assert_eq!(r.frames.len(), expected.len(), "{name}: frame count");
        for (i, (ours, theirs)) in r.frames.iter().zip(expected).enumerate() {
            compare_frame(name, i, ours, theirs);
        }
        if let Value::Array(clips_out) = get(entry, "clips") {
            for (ours, theirs) in r.clips.iter().zip(clips_out) {
                assert_json_near(
                    &clip_json(ours),
                    theirs,
                    &format!("{name}: clip {}", ours.name),
                );
            }
        }
    }
}

// ---- the native scene host ----

fn named(
    arena: &crate::core::ObjectArena,
    root: crate::core::ObjectId,
    name: &str,
) -> Option<crate::core::ObjectId> {
    find_node(&NameTree(arena), &root, Some(name))
}

struct NameTree<'a>(&'a crate::core::ObjectArena);
impl NodeTree for NameTree<'_> {
    type Id = crate::core::ObjectId;
    fn name_is(&self, n: &Self::Id, name: &str) -> bool {
        self.0.get(*n).map(|o| o.name == name).unwrap_or(false)
    }
    fn uuid_is(&self, _: &Self::Id, _: &str) -> bool {
        false
    }
    fn skeleton_bones(&self, _: &Self::Id) -> Option<Vec<Self::Id>> {
        None
    }
    fn children(&self, n: &Self::Id) -> Vec<Self::Id> {
        self.0
            .get(*n)
            .map(|o| o.children.clone())
            .unwrap_or_default()
    }
}

/// features.gltf through the glTF parser and this mixer, against the same
/// file through three.js's GLTFLoader and AnimationMixer: every animated
/// node, 40 frames, a cubic-spline track and a ping-pong clip among them.
#[test]
fn gltf_animation_on_the_native_scene_matches_three() {
    use crate::loaders::gltf_parser::{Document, GltfOptions, GltfParser};
    let t = truth();
    let g = get(&t, "gltf");
    let doc =
        Document::from_bytes(include_bytes!("../../loaders/testdata/gltf/features.gltf")).unwrap();
    let mut parser = GltfParser::new(
        doc,
        GltfOptions {
            refuse_unknown_required: false,
            ..Default::default()
        },
    )
    .unwrap();
    crate::loaders::gltf_parser::supply_images(&mut parser, &Default::default()).unwrap();
    let mut arena = crate::core::ObjectArena::new();
    parser.parse(&mut arena).unwrap();
    let root = parser.objects[parser.default_scene().unwrap()].id;
    let clips: Vec<Arc<AnimationClip>> = parser
        .animations
        .iter()
        .flatten()
        .map(|&c| Arc::new(AnimationClip::from_gltf(&parser.clips[c], &parser.objects)))
        .collect();
    let expected_clips = arr(get(g, "clips"));
    assert_eq!(clips.len(), expected_clips.len());
    for (c, e) in clips.iter().zip(expected_clips) {
        let e = arr(e);
        assert_eq!(c.name, e[0].as_str().unwrap());
        assert!(
            near(c.duration, num(&e[1])),
            "{} duration {} vs {}",
            c.name,
            c.duration,
            num(&e[1])
        );
        assert_eq!(c.tracks.len(), arr(&e[2]).len());
    }
    let mut mixer = AnimationMixer::<SceneKind>::new(MixerRoot::Node(root));
    let mut host = SceneHost::new(&mut arena);
    let a = mixer
        .clip_action(&mut host, clips[0].clone(), None, None)
        .unwrap()
        .unwrap();
    mixer.play(&mut host, a);
    let b = mixer
        .clip_action(&mut host, clips[1].clone(), None, None)
        .unwrap()
        .unwrap();
    mixer
        .action_mut(b)
        .unwrap()
        .set_loop(LOOP_PING_PONG, f64::INFINITY);
    mixer.play(&mut host, b);
    let names: Vec<String> = arr(get(g, "nodes"))
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    for (i, frame) in arr(get(g, "frames")).iter().enumerate() {
        mixer.update(
            &mut host,
            if i == 0 {
                0.0
            } else {
                0.07 + (i % 3) as f64 * 0.01
            },
        );
        assert!(near(mixer.time, num(get(frame, "time"))));
        for (name, expected) in names.iter().zip(arr(get(frame, "nodes"))) {
            let node = named(host.arena, root, name);
            let Value::Array(e) = expected else {
                assert!(
                    node.is_none(),
                    "{name}: three.js has no such node under the scene"
                );
                continue;
            };
            let o = host
                .arena
                .get(node.unwrap_or_else(|| panic!("{name} found")))
                .unwrap();
            let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
            let what = format!("frame {i} {name}");
            assert_near_all(
                &f(&[o.position.x, o.position.y, o.position.z]),
                &nums(&e[0]),
                &format!("{what} position"),
            );
            let q = o.quaternion;
            assert_near_all(
                &f(&[q.x, q.y, q.z, q.w]),
                &nums(&e[1]),
                &format!("{what} quaternion"),
            );
            assert_near_all(
                &f(&[o.scale.x, o.scale.y, o.scale.z]),
                &nums(&e[2]),
                &format!("{what} scale"),
            );
            if let Value::Array(_) = &e[3] {
                let inf = match &o.kind {
                    crate::core::ObjectKind::Mesh(m) => f(&m.morph_influences),
                    crate::core::ObjectKind::SkinnedMesh(m) => f(&m.morph_influences),
                    _ => Vec::new(),
                };
                assert_near_all(&inf, &nums(&e[3]), &format!("{what} morphTargetInfluences"));
            }
        }
    }
}

/// Bones by name under a skinned mesh, morph targets by name on a skinned
/// mesh, a shared material's colour, a light's intensity, and a property the
/// scene has no field for (a slot).
#[test]
fn native_bindings_reach_the_scene() {
    use crate::core::{BufferGeometry, Object3D, ObjectKind, Skeleton, SkinnedMesh};
    use crate::materials::{BasicMaterial, Material};
    use crate::math::{Color, Matrix4};
    let mut scene = crate::scene::Scene::new();
    let mut hip = Object3D::group();
    hip.name = "hip".into();
    let hip = scene.add(hip);
    let mat = Arc::new(Material::Basic(BasicMaterial::new(Color::new(
        1.0, 0.0, 0.0,
    ))));
    let geom = BufferGeometry {
        morph_target_names: vec!["smile".into(), "frown".into()],
        ..Default::default()
    };
    let mut sm = SkinnedMesh::from_arc(
        Arc::new(geom),
        mat.clone(),
        Arc::new(Skeleton::new(vec![crate::core::Bone {
            node: hip,
            inverse_bind: Matrix4::identity(),
        }])),
    );
    sm.morph_influences = vec![0.0, 0.0];
    let mut body = Object3D::skinned_mesh(sm);
    body.name = "body".into();
    let body = scene.add(body);
    let other = scene.add(Object3D::mesh(crate::core::Mesh::from_arc(
        Arc::new(BufferGeometry::default()),
        mat.clone(),
    )));
    let mut lamp = Object3D::light(crate::lights::PointLight::new(
        Color::new(1.0, 1.0, 1.0),
        1.0,
    ));
    lamp.name = "lamp".into();
    let lamp = scene.add(lamp);
    let clip = Arc::new(AnimationClip::new(
        "all",
        -1.0,
        vec![
            KeyframeTrack::vector(
                "body.bones[hip].position",
                vec![0.0, 1.0],
                vec![0., 0., 0., 2., 4., 6.],
            ),
            KeyframeTrack::number(
                "body.morphTargetInfluences[frown]",
                vec![0.0, 1.0],
                vec![0., 1.],
            ),
            KeyframeTrack::color(
                "body.material.color",
                vec![0.0, 1.0],
                vec![1., 0., 0., 0., 0., 1.],
            ),
            KeyframeTrack::number("lamp.intensity", vec![0.0, 1.0], vec![1., 3.]),
            KeyframeTrack::number(
                "lamp.userData[flicker].amount",
                vec![0.0, 1.0],
                vec![0., 10.],
            ),
            KeyframeTrack::boolean("body.visible", vec![0.0, 0.4], vec![true, false]),
        ],
    ));
    let mut slots = PropertySlots::new();
    let mut mixer = AnimationMixer::<SceneKind>::new(MixerRoot::Node(scene.root));
    {
        let mut host = SceneHost::with_slots(&mut scene.arena, &mut slots);
        let a = mixer
            .clip_action(&mut host, clip, None, None)
            .unwrap()
            .unwrap();
        mixer.play(&mut host, a);
        mixer.update(&mut host, 0.5);
    }
    let arena = &scene.arena;
    let p = arena.get(hip).unwrap().position;
    assert!(
        (p.x - 1.0).abs() < 1e-6 && (p.y - 2.0).abs() < 1e-6 && (p.z - 3.0).abs() < 1e-6,
        "{p:?}"
    );
    let ObjectKind::SkinnedMesh(sm) = &arena.get(body).unwrap().kind else {
        unreachable!()
    };
    assert_eq!(sm.morph_influences, vec![0.0, 0.5]);
    let c = sm.material.color();
    assert!(
        (c.r - 0.5).abs() < 1e-6 && (c.b - 0.5).abs() < 1e-6,
        "{c:?}"
    );
    // the other mesh shares the material, so it changed too
    let ObjectKind::Mesh(m) = &arena.get(other).unwrap().kind else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(&m.material, &sm.material));
    let ObjectKind::Light(l) = &arena.get(lamp).unwrap().kind else {
        unreachable!()
    };
    assert!((l.intensity() - 2.0).abs() < 1e-6);
    assert!(!arena.get(body).unwrap().visible);
    assert_eq!(
        slots.get(lamp, "userData[flicker].amount"),
        Some(&[5.0][..])
    );
}

/// A 100-bone chain with position, quaternion and scale tracks on every bone
/// (300 tracks), on the native scene: what an update costs.
#[test]
fn hundred_bone_character_update_cost() {
    use crate::core::Object3D;
    let mut scene = crate::scene::Scene::new();
    let mut parent = scene.root;
    let mut bones = Vec::new();
    for i in 0..100 {
        let mut b = Object3D::group();
        b.name = format!("bone{i}");
        let id = scene.arena.insert(b);
        scene.arena.add_child(parent, id);
        bones.push(id);
        parent = id;
    }
    let times: Vec<f32> = (0..31).map(|k| k as f32 / 30.0).collect();
    let mut tracks = Vec::new();
    for j in 0..100 {
        let (mut q, mut p, mut s) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..31 {
            let a = ((k as f32) / 5.0 + j as f32).sin() * 0.3;
            q.extend([0.0, (a / 2.0).sin(), 0.0, (a / 2.0).cos()]);
            p.extend([0.0, 0.1 + 0.01 * (k as f32).sin(), 0.0]);
            s.extend([1.0, 1.0, 1.0]);
        }
        tracks.push(KeyframeTrack::quaternion(
            format!("bone{j}.quaternion"),
            times.clone(),
            q,
        ));
        tracks.push(KeyframeTrack::vector(
            format!("bone{j}.position"),
            times.clone(),
            p,
        ));
        tracks.push(KeyframeTrack::vector(
            format!("bone{j}.scale"),
            times.clone(),
            s,
        ));
    }
    let clip = Arc::new(AnimationClip::new("walk", -1.0, tracks));
    let mut mixer = AnimationMixer::<SceneKind>::new(MixerRoot::Node(scene.root));
    let mut host = SceneHost::new(&mut scene.arena);
    let a = mixer
        .clip_action(&mut host, clip, None, None)
        .unwrap()
        .unwrap();
    mixer.play(&mut host, a);
    for _ in 0..30 {
        mixer.update(&mut host, 1.0 / 60.0);
    }
    let frames = 2000;
    let t0 = web_time::Instant::now();
    for _ in 0..frames {
        mixer.update(&mut host, 1.0 / 60.0);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    eprintln!("100 bones x 3 tracks, native mixer update: {ms:.4} ms");
    assert!(ms < 16.7, "{ms} ms per update");
}
