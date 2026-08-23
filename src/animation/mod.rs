//! Keyframe animation. Mirrors three.js's `AnimationClip` / `AnimationMixer`
//! / `AnimationAction` / `KeyframeTrack` model.
//!
//! A `Clip` is a named set of `Track`s. Each `Track` targets a node's
//! property by an `ObjectId` + `TrackTarget` enum (position/quaternion/scale/etc.).
//! A `Mixer` advances time and applies the sampled values back to its `Scene`.
//! `Action`s are play/pause handles over individual clips.

mod action;
mod clip;
mod interpolant;
mod mixer;
mod track;

pub use action::{AnimationAction, LoopMode};
pub use clip::AnimationClip;
pub use interpolant::Interpolation;
pub use mixer::{AnimationMixer, BlendMode};
pub use track::{KeyframeTrack, TrackTarget};
