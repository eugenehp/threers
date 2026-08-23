//! Native, from-scratch H.264 / AVC encoder (no ffmpeg, no C bindings).
//!
//! Baseline-profile IDR slices with CAVLC `I_PCM` macroblocks (lossless in the
//! YUV domain) and an MP4 mux path via [`encode_mp4`]. Builds on native and
//! `wasm32`. Width and height must be even; alpha is not supported.

pub mod avcc;
pub mod encoder;
pub mod nal;
pub mod params;
pub mod pps;
pub mod slice;

pub use avcc::build_avcc;
pub use encoder::{encode_mp4, encode_mp4_with_captions, H264Encoder};
pub use nal::{nal_unit, nal_unit_base, push_annexb, split_annexb, NalUnitType, START_CODE};
pub use params::{write_sps, H264Config, MB_SIZE};
