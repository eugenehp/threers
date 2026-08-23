//! Native, dependency-free media codecs (behind the `native-codec` feature).
//!
//! Unlike the optional `video` module (which streams frames to a system
//! `ffmpeg` process), everything here is pure Rust with no OS, process,
//! thread, or filesystem dependency — so it builds and runs on `wasm32` as
//! well as native. Encoders return `Vec<u8>` (or write via [`GifWriter`]); the
//! caller owns I/O.
//!
//! - [`crate::codec::bitstream`] — MSB-first bit writer, Exp-Golomb, RBSP/emulation-prevention.
//! - [`crate::codec::h264`] — from-scratch H.264 / AVC encoder (see its module docs).
//! - [`crate::codec::hevc`] — from-scratch HEVC / H.265 encoder (see its module docs).
//! - [`crate::codec::vp9`] — from-scratch VP9 encoder (intra + inter; alpha via WebM).
//! - [`crate::codec::mp4`] / [`crate::codec::webm`] — ISOBMFF and Matroska/WebM container
//!   muxers (H.264 `avc1`, HEVC `hvc1`, VP9 alpha via `BlockAdditional`).
//! - [`crate::codec::apng`] / [`crate::codec::gif`] — animated PNG and GIF89a encode
//!   (GIF also decodes).

pub mod animation;
pub mod apng;
pub mod bitstream;
pub mod gif;
pub mod h264;
pub mod hevc;
pub mod mp4;
pub mod vp9;
pub mod webm;

pub use animation::{
    encode_animation_rgba, encode_animation_rgba_with_progress, format_animation_progress,
    AnimationEncodeError, AnimationEncodeOptions, AnimationExportPhase, AnimationExportProgress,
    BrowserCodec,
};
pub use apng::ApngEncoder;
pub use bitstream::{emulation_prevention, rbsp_trailing_bits, BitWriter};
pub use gif::{
    decode_gif, encode_gif, DecodedFrame, DisposalMethod, DisposalMode, GifDecoder, GifEncoder,
    GifError, GifFrameMeta, GifInfo, GifOptions, GifVersion, GifWriter, LzwClearMode, PaletteMode,
    QuantizerKind,
};
pub use vp9::{encode_intra_frame, encode_intra_gray, Reconstruction};
pub use webm::{encode_gray_webm, encode_webm, mux_webm, WebmCodec, WebmFrame, WebmParams};
