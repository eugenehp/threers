//! three.js r165's animation system, ported: [`AnimationClip`],
//! [`KeyframeTrack`] (number, vector, quaternion, colour, boolean and string
//! values; discrete, linear, smooth and glTF cubic-spline interpolation),
//! the [`Interpolant`]s, [`PropertyBinding`] with three.js's track-name
//! syntax, [`PropertyMixer`] (weighted accumulation, additive layers,
//! quaternion slerp), [`AnimationMixer`] with its actions (loop modes,
//! fades, cross-fades, warps, scheduled starts, `'loop'` / `'finished'`
//! events), [`AnimationObjectGroup`] and the [`utils`] (`subclip`,
//! `makeClipAdditive` …).
//!
//! The arithmetic is three.js's, operation for operation: track times and
//! values are `f32` (its `Float32Array`s), the mixer's buffers `f64` (its
//! `Float64Array`s), fades and warps run through `f32` control curves, and
//! actions update and bindings apply in the same order — so the same calls
//! give the same numbers. Tests compare against three.js r165 run in node.
//!
//! The scene is an [`AnimationHost`]: [`SceneHost`] animates a native
//! [`ObjectArena`](crate::core::ObjectArena) (transforms, visibility, morph
//! influences on meshes and skinned meshes, material and light parameters,
//! bones, and named scalar slots for anything else); the wasm build binds
//! the JS objects three.js code expects to see animated.

mod action;
mod binding;
mod clip;
mod convert;
mod group;
mod interpolant;
mod mixer;
mod property_mixer;
mod scene_host;
mod track;
pub mod utils;

#[cfg(test)]
mod tests;

pub use action::{ActionEvent, AnimationAction};
pub use binding::{
    find_node, parse_track_name, sanitize_node_name, AnimationHost, HostKind, NodeTree, ParseError,
    ParsedPath, PropertyBinding,
};
pub use clip::{
    generate_uuid, js_number_to_string, parse_keyframe_track, track_to_json_string, AnimationClip,
};
pub use group::{AnimationObjectGroup, GroupRef};
pub use interpolant::{
    multiply_quaternions_flat, quaternion_normalize, slerp_flat, Interpolant, InterpolantKind,
    InterpolantSettings, OwnedInterpolant, ResultBuffer, Seek, WRAP_AROUND_ENDING,
    ZERO_CURVATURE_ENDING, ZERO_SLOPE_ENDING,
};
pub use mixer::{ActionId, AnimationMixer, BindingId, ClipRef, MixerEvent, MixerRoot, MixerStats};
pub use property_mixer::{Binding, PropertyMixer};
pub use scene_host::{MaterialProperty, PropertySlots, SceneHost, SceneKind, SceneTarget};
pub use track::{
    Interpolation, KeyframeTrack, TrackError, TrackModifier, TrackValues, ValueType,
    INTERPOLATE_DISCRETE, INTERPOLATE_LINEAR, INTERPOLATE_SMOOTH,
};

/// `LoopOnce`.
pub const LOOP_ONCE: u32 = 2200;
/// `LoopRepeat`.
pub const LOOP_REPEAT: u32 = 2201;
/// `LoopPingPong`.
pub const LOOP_PING_PONG: u32 = 2202;
/// `NormalAnimationBlendMode`.
pub const NORMAL_ANIMATION_BLEND_MODE: u32 = 2500;
/// `AdditiveAnimationBlendMode`.
pub const ADDITIVE_ANIMATION_BLEND_MODE: u32 = 2501;
