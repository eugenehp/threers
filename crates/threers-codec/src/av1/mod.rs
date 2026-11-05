//! A native AV1 encoder: key frames only.
//!
//! Every picture is intra coded, so a stream is a sequence of independently
//! decodable frames and compresses like Motion JPEG rather than like video.
//! It is a real AV1 bitstream nonetheless, read by any conforming decoder:
//! blocks of 8, 16 and 32 pixels chosen by their activity, DC prediction, the
//! DCT, quantisation and the full coefficient syntax. What it leaves out to
//! stay small is everything a decoder can be told it will not get:
//!
//! - no inter frames, so no motion search, references or rate control;
//! - probabilities stay at their defaults (`disable_cdf_update`), which costs
//!   compression and means nothing has to mirror the decoder's adaptation;
//! - no loop filters, CDEF or restoration, so low quantisers show block edges.
//!
//! The context derivations, scans and inverse transform are the decoder's
//! own, in [`decode::av1`](crate::decode::av1).

mod bitwriter;
mod encoder;
mod range;

pub use encoder::Av1Encoder;
