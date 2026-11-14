//! A native AV1 encoder: key frames only.
//!
//! Every picture is intra coded, so a stream is a sequence of independently
//! decodable frames and compresses like Motion JPEG rather than like video.
//! It is a real AV1 bitstream nonetheless, read by any conforming decoder:
//! blocks of 8, 16 and 32 pixels chosen by their activity, DC prediction, the
//! DCT, quantisation and the full coefficient syntax, with probabilities that
//! adapt as the frame is coded, exactly as the decoder's do (12–19% smaller
//! than the defaults alone, at the same picture). What it leaves out to stay
//! small is everything a decoder can be told it will not get:
//!
//! - no inter frames, so no motion search, references or rate control;
//! - no loop filters, CDEF or restoration, so low quantisers show block edges.
//!
//! At `qidx` 0 it is lossless: AV1's lossless mode, the 4×4 Walsh–Hadamard
//! transform with no quantisation, every block predicted 4×4 by 4×4, with DC
//! or Paeth (per block, for luma and chroma, whichever leaves less residual).
//!
//! The context derivations, scans and inverse transform are the decoder's
//! own, in [`decode::av1`](crate::decode::av1).

mod bitwriter;
mod encoder;
mod range;

pub use encoder::Av1Encoder;
