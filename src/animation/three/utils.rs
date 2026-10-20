//! three.js r165 `AnimationUtils`: keyframe ordering, the AOS key flattener,
//! `subclip` and `makeClipAdditive`. (`convertArray` and `isTypedArray` are
//! about JS array types and have no Rust counterpart.)

use super::clip::AnimationClip;
use super::interpolant::{multiply_quaternions_flat, quaternion_normalize};
use super::track::{Interpolation, TrackValues, ValueType};
use super::ADDITIVE_ANIMATION_BLEND_MODE;
use crate::loaders::json::Value;

/// `getKeyframeOrder(times)`: the indices that sort `times` (stably).
pub fn get_keyframe_order(times: &[f64]) -> Vec<usize> {
    let mut result: Vec<usize> = (0..times.len()).collect();
    // compareTime returns times[i] - times[j]; a NaN difference counts as 0.
    result.sort_by(|&i, &j| {
        let d = times[i] - times[j];
        if d < 0.0 {
            std::cmp::Ordering::Less
        } else if d > 0.0 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    result
}

/// `sortedArray(values, stride, order)`.
pub fn sorted_array<T: Copy>(values: &[T], stride: usize, order: &[usize]) -> Vec<T> {
    let mut result = Vec::with_capacity(values.len());
    let mut i = 0;
    while result.len() < values.len() {
        let src = order[i] * stride;
        for j in 0..stride {
            result.push(values[src + j]);
        }
        i += 1;
    }
    result
}

/// `flattenJSON(jsonKeys, times, values, valuePropertyName)` for numeric
/// values (arrays are spread, scalars pushed).
pub fn flatten_json(keys: &[Value], times: &mut Vec<f64>, values: &mut Vec<f64>, prop: &str) {
    let mut vals = Vec::new();
    flatten_json_values(keys, times, &mut vals, prop);
    values.extend(vals.iter().map(|v| match v {
        Value::Number(n) => *n,
        Value::Bool(b) => *b as u8 as f64,
        _ => f64::NAN,
    }));
}

/// `flattenJSON` keeping the JSON values (for boolean and string tracks).
pub fn flatten_json_values(
    keys: &[Value],
    times: &mut Vec<f64>,
    values: &mut Vec<Value>,
    prop: &str,
) {
    let mut i = 0;
    // the first key with a value
    while i < keys.len() && keys[i].get(prop).is_none() {
        i += 1;
    }
    let Some(first) = keys.get(i).and_then(|k| k.get(prop)) else {
        return; // no data
    };
    let spread = matches!(first, Value::Array(_));
    for key in &keys[i..] {
        let Some(value) = key.get(prop) else {
            continue;
        };
        times.push(
            key.get("time")
                .and_then(|t| t.as_number())
                .unwrap_or(f64::NAN),
        );
        if spread {
            match value {
                Value::Array(a) => values.extend(a.iter().cloned()),
                other => values.push(other.clone()),
            }
        } else {
            values.push(value.clone());
        }
    }
}

/// `AnimationUtils.subclip(sourceClip, name, startFrame, endFrame, fps)`:
/// the keys whose frame is in `[startFrame, endFrame)`, shifted to start at 0.
pub fn subclip(
    source: &AnimationClip,
    name: &str,
    start_frame: f64,
    end_frame: f64,
    fps: f64,
) -> AnimationClip {
    let mut clip = source.clone_clip();
    clip.name = name.to_string();
    let mut tracks = Vec::new();
    for mut track in std::mem::take(&mut clip.tracks) {
        let value_size = track.value_size();
        let mut times = Vec::new();
        let mut keep = Vec::new();
        for (j, &t) in track.times.iter().enumerate() {
            let frame = t as f64 * fps;
            if frame < start_frame || frame >= end_frame {
                continue;
            }
            times.push(t);
            keep.push(j);
        }
        if times.is_empty() {
            continue;
        }
        track.values = match &track.values {
            TrackValues::Numeric(v) => TrackValues::Numeric(
                keep.iter()
                    .flat_map(|&j| (0..value_size).map(move |k| v[j * value_size + k]))
                    .collect(),
            ),
            TrackValues::Bool(v) => TrackValues::Bool(
                keep.iter()
                    .flat_map(|&j| (0..value_size).map(move |k| v[j * value_size + k]))
                    .collect(),
            ),
            TrackValues::String(v) => TrackValues::String(
                keep.iter()
                    .flat_map(|&j| (0..value_size).map(move |k| v[j * value_size + k].clone()))
                    .collect(),
            ),
        };
        track.times = times;
        tracks.push(track);
    }
    clip.tracks = tracks;
    // find minimum .times value across all tracks in the trimmed clip
    let mut min_start = f64::INFINITY;
    for t in &clip.tracks {
        if min_start > t.times[0] as f64 {
            min_start = t.times[0] as f64;
        }
    }
    // shift all tracks such that clip begins at t=0
    for t in &mut clip.tracks {
        t.shift(-min_start);
    }
    clip.reset_duration();
    clip
}

/// `AnimationUtils.makeClipAdditive(targetClip, referenceFrame,
/// referenceClip, fps)`: each numeric track made relative to its value at
/// the reference frame (quaternions by the conjugate), and the clip marked
/// additive. `reference` `None` is the target itself.
pub fn make_clip_additive(
    target: &mut AnimationClip,
    reference_frame: f64,
    reference: Option<&AnimationClip>,
    fps: f64,
) {
    let fps = if fps <= 0.0 { 30.0 } else { fps };
    let reference_time = reference_frame / fps;
    let num_tracks = reference
        .map(|r| r.tracks.len())
        .unwrap_or(target.tracks.len());
    for i in 0..num_tracks {
        let reference_track = match reference {
            Some(r) => r.tracks[i].clone(),
            None => target.tracks[i].clone(),
        };
        let ref_type = reference_track.value_type;
        // Skip this track if it's non-numeric
        if !ref_type.is_numeric() {
            continue;
        }
        // the target track whose name and type match the reference track
        let Some(ti) = target
            .tracks
            .iter()
            .position(|t| t.name == reference_track.name && t.value_type == ref_type)
        else {
            continue;
        };
        let ref_values = reference_track.values.numeric().unwrap_or(&[]);
        let ref_value_size = reference_track.value_size();
        let ref_offset = if reference_track.interpolation() == Interpolation::GltfCubicSpline {
            ref_value_size / 3
        } else {
            0
        };
        let target_value_size = target.tracks[ti].value_size();
        let target_offset = if target.tracks[ti].interpolation() == Interpolation::GltfCubicSpline {
            target_value_size / 3
        } else {
            0
        };
        let last_index = reference_track.times.len() - 1;
        // The value to subtract out of the track — a Float32Array in three.js.
        let slice = |from: usize, to: usize| -> Vec<f32> {
            if from >= to {
                Vec::new()
            } else {
                ref_values[from.min(ref_values.len())..to.min(ref_values.len())].to_vec()
            }
        };
        let mut reference_value: Vec<f32> = if reference_time <= reference_track.times[0] as f64 {
            // earlier than the first keyframe: the first keyframe
            slice(ref_offset, ref_value_size - ref_offset)
        } else if reference_time >= reference_track.times[last_index] as f64 {
            // after the last keyframe: the last keyframe
            let start = last_index * ref_value_size + ref_offset;
            slice(start, start + ref_value_size - ref_offset)
        } else {
            // Interpolate to the reference value, into the interpolant's own
            // Float32Array (a glTF track's is a third of the value size).
            let mut interpolant = reference_track.create_interpolant();
            let mut result = vec![0.0f32; interpolant.value_size];
            interpolant.evaluate(
                &reference_track.times,
                ref_values,
                reference_time,
                None,
                result.as_mut_slice(),
            );
            let (from, to) = (ref_offset, ref_value_size - ref_offset);
            if from >= to || from >= result.len() {
                Vec::new()
            } else {
                result[from..to.min(result.len())].to_vec()
            }
        };
        // Conjugate the quaternion
        if ref_type == ValueType::Quaternion {
            let q = [0, 1, 2, 3].map(|k| {
                reference_value
                    .get(k)
                    .map(|&v| v as f64)
                    .unwrap_or(f64::NAN)
            });
            let q = quaternion_normalize(q);
            let conj = [-q[0], -q[1], -q[2], q[3]];
            // toArray into the Float32Array
            if reference_value.len() < 4 {
                reference_value.resize(4, 0.0);
            }
            for k in 0..4 {
                reference_value[k] = conj[k] as f32;
            }
        }
        // Subtract the reference value from all of the track values
        let track = &mut target.tracks[ti];
        let num_times = track.times.len();
        let TrackValues::Numeric(values) = &mut track.values else {
            continue;
        };
        let rv = |k: usize| {
            reference_value
                .get(k)
                .map(|&v| v as f64)
                .unwrap_or(f64::NAN)
        };
        for j in 0..num_times {
            let start = j * target_value_size + target_offset;
            if ref_type == ValueType::Quaternion {
                // multiplyQuaternionsFlat(values, start, ref, 0, values, start)
                let get = |v: &Vec<f32>, k: usize| v.get(k).map(|&x| x as f64).unwrap_or(f64::NAN);
                let tq = [
                    get(values, start),
                    get(values, start + 1),
                    get(values, start + 2),
                    get(values, start + 3),
                ];
                let r = multiply_quaternions_flat([rv(0), rv(1), rv(2), rv(3)], tq);
                for (k, v) in r.iter().enumerate() {
                    if let Some(s) = values.get_mut(start + k) {
                        *s = *v as f32;
                    }
                }
            } else {
                let value_end = target_value_size.saturating_sub(target_offset * 2);
                for k in 0..value_end {
                    if let Some(s) = values.get_mut(start + k) {
                        *s = (*s as f64 - rv(k)) as f32;
                    }
                }
            }
        }
    }
    target.blend_mode = ADDITIVE_ANIMATION_BLEND_MODE;
}
