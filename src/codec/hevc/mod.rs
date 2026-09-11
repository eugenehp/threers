//! Native, from-scratch HEVC / H.265 encoder (no ffmpeg, no C bindings).
//!
//! Pure Rust so it runs everywhere threers does — native **and** `wasm32`. The
//! encoder produces bytes; the caller decides where they go (write a file on
//! native, trigger a download / feed MediaSource in the browser).
//!
//! # Why from scratch
//!
//! The existing [`crate::video`] path shells out to the system `ffmpeg`. That
//! can't work in the browser and forces an external dependency on native. A
//! native encoder removes both constraints — at the cost of reimplementing a
//! genuinely large codec. HEVC is CABAC-only (no CAVLC fallback), so even a
//! minimal conformant stream needs the arithmetic coder, parameter sets, and a
//! slice layer.
//!
//! # Build order (milestones)
//!
//! Each layer is independently testable; higher layers are gated behind the ones
//! below being bit-exact.
//!
//! 1. **Foundation** — *done*: [`crate::codec::bitstream`] (bit writer,
//!    Exp-Golomb, RBSP/emulation-prevention), [`cabac`] (the arithmetic engine,
//!    verified by an encode↔decode roundtrip), and [`nal`] (NAL units +
//!    Annex-B).
//! 2. **Parameter sets** — *done*: VPS / SPS / PPS writers ([`params`]).
//! 3. **End-to-end playable** — *done & externally verified*: an IDR I-slice
//!    ([`mod@crate::codec::hevc::slice`]) of `I_PCM` CUs, assembled by [`HevcEncoder`]. No compression,
//!    but ffmpeg decodes it **losslessly** in the YUV domain (see
//!    `tests/hevc_ffmpeg.rs`), which validates the whole pipeline (params →
//!    slice header → CABAC control bins → PCM raw bytes → conformance-window
//!    crop → NAL/Annex-B).
//! 4. **Transparency** — *done & verified*. A full Apple-compatible transparent
//!    video: base color layer + an `AUX_ALPHA` auxiliary layer, the
//!    [`alpha_channel_info`](alpha::alpha_channel_info_sei) SEI, and an
//!    Apple-style MP4 (`hvcC` + `almo`), all via [`TransparentEncoder`]. The
//!    multi-layer `VPS` is Apple's reference NAL, which is resolution-independent
//!    for decode (self-contained per-layer SPSs). **AVFoundation** — the
//!    QuickTime/Safari decoder — decodes the result with **lossless alpha** at
//!    multiple sizes (see `tests/hevc_alpha.rs`). Data path is alpha-aware via
//!    [`Yuv420Frame::alpha`](encoder::Yuv420Frame).
//! 5. **Compression** — [`compress`]'s [`CompressedEncoder`] transform-codes
//!    intra frames (DCT-4/8/16 + DST-4, [`quant`], [`intra`] with the DC/H/V
//!    boundary filters, MPM mode coding, reference smoothing, reconstruction
//!    loop). Two modes:
//!    - *Prediction-only* (default): every `cbf = 0`. **Externally verified
//!      conformant** — ffmpeg decodes it to *exactly* the encoder's
//!      reconstruction for high-contrast content at multiple sizes (~240×
//!      smaller than raw; lossy). See `tests/hevc_compress.rs`.
//!    - *Residual* (`.residual(true)`, experimental): also codes quantized
//!      coefficients ([`residual`], `residual_coding` §7.3.8.11). The luma path
//!      and smooth / low-frequency chroma are **externally conformant** — ffmpeg
//!      decodes them to exactly the encoder's reconstruction (see
//!      `compressed_residual_smooth_is_conformant` in `tests/hevc_compress.rs`).
//!      One known gap remains: dense 8×8 chroma blocks whose last significant
//!      coefficient falls in last-position group 4 (coordinate 4 or 5) desync a
//!      conformant decoder, so residual stays off by default. See
//!      [`residual`] for the investigation notes.
//! 6. **Muxing & inter** — the MP4 muxer ([`crate::codec::mp4`]) exists for the
//!    alpha path; generalize it for the plain color path, then P/B pictures with
//!    motion estimation.
//! 7. **Web transparency** — *done (image path)*: [`crate::codec::apng`] and
//!    [`crate::codec::gif`] emit universally-web-supported transparent animation
//!    (APNG lossless-alpha; GIF via median-cut + reserved transparent index).
//!    Native VP9/WebM with residual + `BlockAdditional` alpha is also in place
//!    ([`crate::codec::vp9`] + [`crate::codec::webm`]); inter (P-frames) remains.

// These modules transcribe libvpx / libde265 reference code, and the point of
// doing that is that a reader can lay the two side by side. So the shape of the
// original survives here: loops that index by hand because the spec numbers its
// arrays, argument lists as long as the C function's, branches left distinct
// where the spec distinguishes cases that happen to compute the same thing, and
// constants grouped the way the bitstream tables print them. Idiomatic Rust
// would read better and would no longer be checkable against the reference.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::if_same_then_else,
    clippy::unusual_byte_groupings
)]

pub mod alpha;
pub mod cabac;
pub mod compress;
pub mod deblock;
pub mod deblock_tables;
pub mod encoder;
pub mod hvcc;
pub mod intra;
pub mod nal;
pub mod params;
pub mod quant;
pub mod residual;
pub mod slice;
pub mod tables;
pub mod transform;
pub mod transparent;

pub use alpha::{alpha_channel_info_sei, AlphaChannelInfo};
pub use cabac::{CabacEncoder, CtxModel};
pub use compress::{
    encode_mp4 as encode_compressed_mp4,
    encode_mp4_from_iter as encode_compressed_mp4_from_iter,
    encode_mp4_streaming as encode_compressed_mp4_streaming,
    encode_mp4_with_captions as encode_compressed_mp4_with_captions, CompressedEncoder,
    Reconstruction,
};
pub use encoder::{HevcEncoder, Yuv420Frame};
pub use hvcc::{build_hvcc, HvccArray, HvccProfile};
pub use nal::{nal_unit, nal_unit_base, push_annexb, NalUnitType, START_CODE};
pub use params::{HevcConfig, CTB_SIZE};
pub use transparent::TransparentEncoder;
