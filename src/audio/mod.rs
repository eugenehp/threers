//! Audio data types. Modeled on three.js's `Audio`/`AudioListener`/
//! `PositionalAudio`/`AudioAnalyser`. The actual playback backend is left to
//! the user (cpal natively, Web Audio in wasm) — this crate only owns the
//! data shape so other subsystems (scene graph, animation) can reference it.

mod analyser;
mod backend;
mod listener;
mod positional;
mod source;

pub use analyser::AudioAnalyser;
pub use backend::{AudioBackend, NoopBackend};
pub use listener::AudioListener;
pub use positional::{DistanceModel, PositionalAudio, SpatialMix};
pub use source::Audio;
