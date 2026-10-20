//! three.js r165 `AnimationClip`: a named, reusable set of keyframe tracks.

use super::track::{Interpolation, KeyframeTrack, TrackValues, ValueType};
use super::utils;
use super::NORMAL_ANIMATION_BLEND_MODE;
use crate::loaders::json::Value;

/// three.js `AnimationClip`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClip {
    pub name: String,
    pub tracks: Vec<KeyframeTrack>,
    /// Seconds. Negative at construction means "from the tracks".
    pub duration: f64,
    /// `NormalAnimationBlendMode` (2500) or `AdditiveAnimationBlendMode` (2501).
    pub blend_mode: u32,
    pub uuid: String,
}

impl AnimationClip {
    /// `new AnimationClip(name, duration, tracks, blendMode)`; a negative
    /// duration is computed from the tracks (`resetDuration`).
    pub fn new(name: impl Into<String>, duration: f64, tracks: Vec<KeyframeTrack>) -> Self {
        Self::with_blend_mode(name, duration, tracks, NORMAL_ANIMATION_BLEND_MODE)
    }

    pub fn with_blend_mode(
        name: impl Into<String>,
        duration: f64,
        tracks: Vec<KeyframeTrack>,
        blend_mode: u32,
    ) -> Self {
        let mut clip = Self {
            name: name.into(),
            tracks,
            duration,
            blend_mode,
            uuid: generate_uuid(),
        };
        // this means it should figure out its duration by scanning the tracks
        if clip.duration < 0.0 {
            clip.reset_duration();
        }
        clip
    }

    /// `resetDuration()`: the last key time of the longest track.
    pub fn reset_duration(&mut self) -> &mut Self {
        let mut duration = 0.0f64;
        for track in &self.tracks {
            let last = track.times.last().map(|&t| t as f64).unwrap_or(f64::NAN);
            // Math.max: NaN wins.
            duration = if duration.is_nan() || last.is_nan() {
                f64::NAN
            } else {
                duration.max(last)
            };
        }
        self.duration = duration;
        self
    }

    /// `trim()`: every track trimmed to `[0, duration]`.
    pub fn trim(&mut self) -> &mut Self {
        let d = self.duration;
        for t in &mut self.tracks {
            t.trim(0.0, d);
        }
        self
    }

    /// `validate()`: every track valid (stops at the first invalid one).
    pub fn validate(&self) -> bool {
        let mut valid = true;
        for t in &self.tracks {
            valid = valid && t.validate();
        }
        valid
    }

    /// `optimize()`: every track optimised.
    pub fn optimize(&mut self) -> &mut Self {
        for t in &mut self.tracks {
            t.optimize();
        }
        self
    }

    /// three.js `clip.clone()`: the same tracks and settings under a new uuid
    /// (`Clone` keeps the uuid).
    pub fn clone_clip(&self) -> Self {
        Self::with_blend_mode(
            self.name.clone(),
            self.duration,
            self.tracks.clone(),
            self.blend_mode,
        )
    }

    /// `AnimationClip.findByName(clips, name)`.
    pub fn find_by_name<'a>(clips: &'a [AnimationClip], name: &str) -> Option<&'a AnimationClip> {
        clips.iter().find(|c| c.name == name)
    }

    /// `AnimationClip.CreateFromMorphTargetSequence(name, morphTargets, fps,
    /// noLoop)`, from the morph targets' names.
    pub fn create_from_morph_target_sequence(
        name: impl Into<String>,
        morph_target_names: &[String],
        fps: f64,
        no_loop: bool,
    ) -> Self {
        let num = morph_target_names.len();
        let mut tracks = Vec::with_capacity(num);
        for (i, target) in morph_target_names.iter().enumerate() {
            let times = [
                ((i + num - 1) % num) as f64,
                i as f64,
                ((i + 1) % num) as f64,
            ];
            let values = [0.0f64, 1.0, 0.0];
            let order = utils::get_keyframe_order(&times);
            let mut times = utils::sorted_array(&times, 1, &order);
            let mut values = utils::sorted_array(&values, 1, &order);
            // if there is a key at the first frame, duplicate it as the
            // last frame as well for perfect loop.
            if !no_loop && times[0] == 0.0 {
                times.push(num as f64);
                values.push(values[0]);
            }
            let mut track = KeyframeTrack::number(
                format!(".morphTargetInfluences[{target}]"),
                times.iter().map(|&t| t as f32).collect(),
                values.iter().map(|&v| v as f32).collect(),
            );
            track.scale(1.0 / fps);
            tracks.push(track);
        }
        Self::new(name, -1.0, tracks)
    }

    /// `AnimationClip.CreateClipsFromMorphTargetSequences(morphTargets, fps,
    /// noLoop)`: one clip per `name_000`-style prefix, in first-seen order.
    pub fn create_clips_from_morph_target_sequences(
        morph_target_names: &[String],
        fps: f64,
        no_loop: bool,
    ) -> Vec<Self> {
        let mut groups: Vec<(String, Vec<String>)> = Vec::new();
        for name in morph_target_names {
            // /^([\w-]*?)([\d]+)$/
            let Some(prefix) = morph_sequence_prefix(name) else {
                continue;
            };
            match groups.iter_mut().find(|(p, _)| *p == prefix) {
                Some((_, list)) => list.push(name.clone()),
                None => groups.push((prefix, vec![name.clone()])),
            }
        }
        // JS `for in` visits integer-like keys first, in ascending order.
        let mut ordered: Vec<(String, Vec<String>)> = Vec::new();
        let mut ints: Vec<(u64, (String, Vec<String>))> = groups
            .iter()
            .filter_map(|g| js_array_index(&g.0).map(|i| (i, g.clone())))
            .collect();
        ints.sort_by_key(|(i, _)| *i);
        ordered.extend(ints.into_iter().map(|(_, g)| g));
        ordered.extend(
            groups
                .into_iter()
                .filter(|g| js_array_index(&g.0).is_none()),
        );
        ordered
            .into_iter()
            .map(|(name, targets)| {
                Self::create_from_morph_target_sequence(name, &targets, fps, no_loop)
            })
            .collect()
    }

    /// `AnimationClip.parse(json)`: tracks scaled by `1 / fps`, uuid kept.
    pub fn parse(json: &Value) -> Result<Self, String> {
        let tracks_json = json
            .get("tracks")
            .and_then(|t| t.as_array())
            .ok_or_else(|| "AnimationClip.parse: no tracks".to_string())?;
        let frame_time = 1.0
            / json
                .get("fps")
                .and_then(|v| v.as_number())
                .filter(|f| *f != 0.0)
                .unwrap_or(1.0);
        let mut tracks = Vec::with_capacity(tracks_json.len());
        for t in tracks_json {
            let mut track = parse_keyframe_track(t)?;
            track.scale(frame_time);
            tracks.push(track);
        }
        let name = json
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // `new AnimationClip(json.name, json.duration, …)`: an absent duration
        // is the constructor's default, -1.
        let duration = match json.get("duration") {
            Some(Value::Number(d)) => *d,
            _ => -1.0,
        };
        let blend_mode = json
            .get("blendMode")
            .and_then(|v| v.as_number())
            .map(|v| v as u32)
            .unwrap_or(NORMAL_ANIMATION_BLEND_MODE);
        let mut clip = Self::with_blend_mode(name, duration, tracks, blend_mode);
        clip.uuid = json
            .get("uuid")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(clip)
    }

    /// `AnimationClip.parseAnimation(animation, bones)`: the old JSONLoader
    /// `hierarchy` format, one bone name per hierarchy entry.
    pub fn parse_animation(animation: &Value, bone_names: &[String]) -> Option<Self> {
        let mut tracks = Vec::new();
        let clip_name = animation
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("default")
            .to_string();
        let blend_mode = animation
            .get("blendMode")
            .and_then(|v| v.as_number())
            .map(|v| v as u32)
            .unwrap_or(NORMAL_ANIMATION_BLEND_MODE);
        let mut duration = animation
            .get("length")
            .and_then(|v| v.as_number())
            .filter(|l| *l != 0.0)
            .unwrap_or(-1.0);
        let empty = Vec::new();
        let hierarchy = animation
            .get("hierarchy")
            .and_then(|h| h.as_array())
            .unwrap_or(&empty);
        for (h, entry) in hierarchy.iter().enumerate() {
            let Some(keys) = entry.get("keys").and_then(|k| k.as_array()) else {
                continue;
            };
            if keys.is_empty() {
                continue;
            }
            if keys[0].get("morphTargets").is_some() {
                // three.js's morph branch reads `animationKeys[k]` with k past
                // the end and throws; there is nothing faithful to do but skip.
                log::warn!("THREE.AnimationClip.parseAnimation: morph target hierarchies are not supported");
                duration = f64::NAN;
                continue;
            }
            let bone = format!(
                ".bones[{}]",
                bone_names.get(h).map(String::as_str).unwrap_or("")
            );
            let mut add = |ty: ValueType, suffix: &str, prop: &str| {
                let mut times = Vec::new();
                let mut values = Vec::new();
                utils::flatten_json(keys, &mut times, &mut values, prop);
                if !times.is_empty() {
                    if let Ok(t) = KeyframeTrack::new(
                        ty,
                        format!("{bone}{suffix}"),
                        times.iter().map(|&t| t as f32).collect(),
                        TrackValues::Numeric(values.iter().map(|&v| v as f32).collect()),
                        None,
                    ) {
                        tracks.push(t);
                    }
                }
            };
            add(ValueType::Vector, ".position", "pos");
            add(ValueType::Quaternion, ".quaternion", "rot");
            add(ValueType::Vector, ".scale", "scl");
        }
        if tracks.is_empty() {
            return None;
        }
        Some(Self::with_blend_mode(
            clip_name, duration, tracks, blend_mode,
        ))
    }

    /// `AnimationClip.toJSON(clip)` as a JSON string, keys in three.js's
    /// order (`JSON.stringify(clip.toJSON())`).
    pub fn to_json_string(&self) -> String {
        let mut s = String::new();
        s.push_str("{\"name\":");
        push_json_string(&mut s, &self.name);
        s.push_str(",\"duration\":");
        s.push_str(&js_number_json(self.duration));
        s.push_str(",\"tracks\":[");
        for (i, t) in self.tracks.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&track_to_json_string(t));
        }
        s.push(']');
        // `uuid: undefined` (a parsed clip without one) is left out.
        if !self.uuid.is_empty() {
            s.push_str(",\"uuid\":");
            push_json_string(&mut s, &self.uuid);
        }
        s.push_str(",\"blendMode\":");
        s.push_str(&js_number_json(self.blend_mode as f64));
        s.push('}');
        s
    }

    /// `AnimationClip.parse` from a JSON string.
    pub fn parse_json_str(src: &str) -> Result<Self, String> {
        let v = crate::loaders::json::parse(src).map_err(|e| e.to_string())?;
        Self::parse(&v)
    }
}

/// `KeyframeTrack.toJSON(track)` as a JSON string.
pub fn track_to_json_string(t: &KeyframeTrack) -> String {
    let mut s = String::new();
    s.push_str("{\"name\":");
    push_json_string(&mut s, &t.name);
    s.push_str(",\"times\":[");
    for (i, v) in t.times.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&js_number_json(*v as f64));
    }
    s.push_str("],\"values\":[");
    match &t.values {
        TrackValues::Numeric(v) => {
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&js_number_json(*x as f64));
            }
        }
        TrackValues::Bool(v) => {
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(if *x { "true" } else { "false" });
            }
        }
        TrackValues::String(v) => {
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                push_json_string(&mut s, x);
            }
        }
    }
    s.push(']');
    let interp = t.interpolation();
    if interp != t.value_type.default_interpolation() {
        s.push_str(",\"interpolation\":");
        match interp.constant() {
            Some(c) => s.push_str(&c.to_string()),
            // `undefined` is left out by JSON.stringify.
            None => {
                s.truncate(s.len() - ",\"interpolation\":".len());
            }
        }
    }
    s.push_str(",\"type\":");
    push_json_string(&mut s, t.value_type.name());
    s.push('}');
    s
}

/// `parseKeyframeTrack(json)`.
pub fn parse_keyframe_track(json: &Value) -> Result<KeyframeTrack, String> {
    let ty = json
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "THREE.KeyframeTrack: track type undefined, can not parse".to_string())?;
    let value_type = ValueType::from_name(ty)
        .ok_or_else(|| format!("THREE.KeyframeTrack: Unsupported typeName: {ty}"))?;
    let name = json
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("undefined")
        .to_string();
    let interpolation = json
        .get("interpolation")
        .and_then(|v| v.as_number())
        .and_then(|c| Interpolation::from_constant(c as u32));
    let (times, values): (Vec<f64>, Vec<Value>) = match json.get("times").and_then(|v| v.as_array())
    {
        Some(times) => (
            times
                .iter()
                .map(|t| t.as_number().unwrap_or(f64::NAN))
                .collect(),
            json.get("values")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
        ),
        None => {
            let empty = Vec::new();
            let keys = json
                .get("keys")
                .and_then(|k| k.as_array())
                .unwrap_or(&empty);
            let mut times = Vec::new();
            let mut values = Vec::new();
            utils::flatten_json_values(keys, &mut times, &mut values, "value");
            (times, values)
        }
    };
    let values = match value_type {
        ValueType::Bool => TrackValues::Bool(values.iter().map(js_truthy).collect()),
        ValueType::String => TrackValues::String(
            values
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => js_number_json(*n),
                    Value::Bool(b) => b.to_string(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => TrackValues::Numeric(
            values
                .iter()
                .map(|v| v.as_number().unwrap_or(f64::NAN) as f32)
                .collect(),
        ),
    };
    KeyframeTrack::new(
        value_type,
        name,
        times.iter().map(|&t| t as f32).collect(),
        values,
        interpolation,
    )
    .map_err(|e| e.to_string())
}

fn js_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => *n != 0.0 && !n.is_nan(),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

/// `/^([\w-]*?)([\d]+)$/` → the lazy prefix.
fn morph_sequence_prefix(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'-';
    if !bytes.iter().all(|&c| is_word(c)) {
        return None;
    }
    // Trailing digits (at least one); the lazy prefix takes as few characters
    // as it can, so every trailing digit goes to the second group.
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end == bytes.len() {
        return None;
    }
    Some(name[..end].to_string())
}

/// A JS array-index property key (canonical non-negative integer string).
fn js_array_index(s: &str) -> Option<u64> {
    if s.is_empty() || (s.len() > 1 && s.starts_with('0')) || !s.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    s.parse::<u64>().ok().filter(|&n| n < u32::MAX as u64)
}

fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A number as `JSON.stringify` writes it (`null` for NaN and infinities).
pub fn js_number_json(v: f64) -> String {
    if !v.is_finite() {
        return "null".into();
    }
    js_number_to_string(v)
}

/// JS `Number.prototype.toString()` for a finite number.
pub fn js_number_to_string(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if v < 0.0 {
        return format!("-{}", js_number_to_string(-v));
    }
    // Shortest round-trip digits, as `d.ddde±x`.
    let e = format!("{v:e}");
    let (mantissa, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let k = digits.len() as i32;
    let n = exp + 1;
    if k <= n && n <= 21 {
        let mut s = digits.clone();
        s.extend(std::iter::repeat_n('0', (n - k) as usize));
        s
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{}", "0".repeat((-n) as usize), digits)
    } else {
        let sign = if n - 1 < 0 { '-' } else { '+' };
        if k == 1 {
            format!("{}e{}{}", digits, sign, (n - 1).abs())
        } else {
            format!(
                "{}.{}e{}{}",
                &digits[..1],
                &digits[1..],
                sign,
                (n - 1).abs()
            )
        }
    }
}

/// three.js `MathUtils.generateUUID()`-shaped (upper-case, version 4).
pub fn generate_uuid() -> String {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            let seed = web_time::SystemTime::now()
                .duration_since(web_time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15);
            x = seed ^ (s as *const _ as u64).rotate_left(17) ^ 0x2545_F491_4F6C_DD1D;
            if x == 0 {
                x = 0x2545_F491_4F6C_DD1D;
            }
        }
        let mut next = || {
            // xorshift64*
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let a = next();
        let b = next();
        s.set(x);
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&a.to_le_bytes());
        bytes[8..].copy_from_slice(&b.to_le_bytes());
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let h: Vec<String> = bytes.iter().map(|b| format!("{b:02X}")).collect();
        format!(
            "{}-{}-{}-{}-{}",
            h[0..4].concat(),
            h[4..6].concat(),
            h[6..8].concat(),
            h[8..10].concat(),
            h[10..16].concat()
        )
    })
}
