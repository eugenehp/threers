//! Native, from-scratch H.264 / AVC encoder (no ffmpeg, no C bindings).
//!
//! Baseline-profile IDR slices with CAVLC `I_PCM` macroblocks (lossless in the
//! YUV domain) and an MP4 mux path via [`encode_mp4`]. Builds on native and
//! `wasm32`. Width and height must be even; alpha is not supported.

pub mod avcc;
pub mod cabac;
pub mod cabac_tables;
pub mod cavlc;
pub mod deblock;
pub mod deblock_tables;
pub mod compress;
pub mod encoder;
pub mod intra;
pub mod nal;
pub mod params;
pub mod pps;
pub mod slice;
pub mod transform;

pub use avcc::build_avcc;
pub use compress::{
    encode_mp4 as encode_compressed_mp4, encode_mp4_from_iter as encode_compressed_mp4_from_iter,
    write_mp4_from_iter as write_compressed_mp4, write_mp4_quality as write_compressed_mp4_quality,
    CompressedEncoder,
};
pub use encoder::{
    encode_mp4, encode_mp4_from_iter, encode_mp4_streaming, encode_mp4_with_captions, write_mp4_from_iter as write_mp4, H264Encoder,
};
pub use nal::{nal_unit, nal_unit_base, push_annexb, split_annexb, NalUnitType, START_CODE};
pub use params::{write_sps, EntropyMode, H264Config, MB_SIZE};
