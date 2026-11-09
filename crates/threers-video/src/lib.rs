//! Video for [threers](https://docs.rs/threers):
//!
//! - [`export`] — rendered frames to MP4 (H.264), MOV (HEVC), WebM (VP9), GIF
//!   and APNG, with soft or burned-in captions; the pure-Rust encoders, or
//!   ffmpeg where it is installed. Native only.
//! - [`sequence`] — render a sequence of stills, a video, or both from one pass.
//! - [`player`] — [`VideoPlayer`]: video on a mesh, with its sound, decoded in
//!   pure Rust on every core, in hardware (VideoToolbox, NVDEC, VAAPI) or, in a
//!   browser, by WebCodecs.
//! - `ScadVideo` (the `openscad` feature) — `export_video` on an OpenSCAD
//!   animation.
//!
//! It sits on top of `threers` (with `render`, `audio` and the codecs) and
//! `threers-codec`; the core does not depend on it.

#[cfg(not(target_arch = "wasm32"))]
pub mod export;
pub mod player;
#[cfg(all(feature = "openscad", not(target_arch = "wasm32")))]
mod scad;
pub mod sequence;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use export::{
    export_video, export_video_with_progress, format_video_progress, CaptionExport, CaptionMode,
    VideoCodec, VideoError, VideoExportEvent, VideoExportPhase, VideoExportProgress, VideoExporter,
    VideoOptions, VideoQuality,
};
pub use player::VideoPlayer;
#[cfg(all(feature = "openscad", not(target_arch = "wasm32")))]
pub use scad::ScadVideo;
pub use sequence::{
    render_sequence, video_available, SequenceError, SequenceOptions, SequenceReport,
};
