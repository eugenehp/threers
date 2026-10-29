//! ffmpeg-compatible behaviour, without ffmpeg: the values and pictures the
//! ffmpeg n7.1 CLI produces from an MP4/MOV, computed from the file.
//!
//! - [`mov`]: what the mov demuxer reports for the video track (banner fps
//!   and duration, `r_frame_rate`, frame pts and durations after
//!   `mov_fix_index`).
//! - [`mkv`]: the same for a VP9 WebM/Matroska file (`matroskadec` and
//!   `find_stream_info`'s frame-rate estimate).
//! - [`pipe`]: which decoded frames the CLI writes to a raw pipe (CFR/VSCFR
//!   frame sync, accurate seeks).
//! - [`swscale`]: libswscale's C `yuv2rgb` for 8-bit 4:2:0 / 4:2:2 → rgb24
//!   (even heights), and [`sws_generic`] — the filtered scaler path for the
//!   rest (odd heights, 9-16 bits, 4:4:4).
//!
//! Decoding itself is [`crate::decode`] (H.264/HEVC, bit-identical to
//! ffmpeg's decoders on the tested streams); audio is
//! [`crate::aac`] + [`crate::swr`].

pub mod mkv;
pub mod mov;
pub mod pipe;
pub mod swscale;
pub mod sws_generic;
#[cfg(feature = "decode")]
pub mod reader;
