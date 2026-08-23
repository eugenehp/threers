//! Native, from-scratch VP9 encoder (no libvpx, no C bindings) — the codec that
//! rides inside [`crate::codec::webm`] for a fully dependency-free `.webm`.
//!
//! Pure Rust so it runs everywhere threers does, native **and** `wasm32`. Like
//! the [`hevc`](crate::codec::hevc) effort, VP9 is built as a ladder of
//! independently-testable layers, each gated behind the one below being
//! bit-exact.
//!
//! # Why VP9 (and WebM)
//!
//! WebM is the only broadly browser-native *video* container that carries an
//! alpha channel (VP9 + a second VP9 stream in `BlockAdditional`). The existing
//! [`crate::video`] path reaches it through the system `ffmpeg`/`libvpx`, which
//! can't run in the browser. A native encoder removes that dependency — at the
//! cost of reimplementing a large codec.
//!
//! # Build order (milestones)
//!
//! 1. **Boolean entropy coder** — *done*: [`bool_coder`], the binary range coder
//!    every compressed syntax element uses (RFC 6386 §7), verified by an
//!    encode↔decode roundtrip. This is VP9's analog of HEVC's CABAC engine.
//! 2. **Container** — *done*: [`crate::codec::webm`] muxes coded frames into a
//!    playable `.webm` (validated end-to-end against ffmpeg in `tests/webm.rs`).
//! 3. **Frame headers + gray keyframe** — *done & externally verified*:
//!    [`keyframe`] emits a profile-0 intra keyframe (uncompressed + compressed
//!    header, one 64×64 `PARTITION_NONE` / `skip` / `DC_PRED` superblock). With
//!    no neighbors DC is mid-gray (`128`); ffmpeg decodes it losslessly in the
//!    YUV domain (see `tests/webm.rs`).
//! 4. **Muxing** — *done*: [`crate::codec::webm::encode_gray_webm`] pairs the
//!    keyframe encoder with the WebM muxer for a dependency-free `.webm`.
//! 5. **Real intra (prediction-only)** — *done*: [`encoder`] partitions to 8×8
//!    with SAD-picked DC/V/H modes (skip-only path retained for zero residual).
//! 6. **Residual** — *done & externally verified*: 4×4 hybrid DCT/ADST + quant +
//!    coefficient tokens ([`transform`], [`quant`], [`tokens`]). DC/V/H modes
//!    with matching TX types; frame sizes that are multiples of 8 (partial edge
//!    superblocks included). ffmpeg decodes to our recon bit-for-bit
//!    (see `tests/webm.rs`).
//! 7. **Alpha** — *done & externally verified*: WebM `AlphaMode` + alpha VP9 in
//!    `BlockAdditional` ([`crate::codec::webm`]). ffmpeg/`libvpx-vp9` decodes
//!    colour + alpha to our recons (see `tests/webm.rs`).
//! 8. **Inter** — *in progress*: [`inter`] motion search, LAST/GOLDEN/ALTREF refs,
//!    compound LAST/GOLDEN+ALTREF, square + HORZ/VERT partitions through 64×64,
//!    switchable EIGHTTAP/SMOOTH/SHARP MC ([`mc`]), SAD ME ([`me`]),
//!    `TX_MODE_SELECT` with max/max−1 TX pick, multi-column tiles (≥512 px),
//!    segmentation (map + `SEG_LVL_ALT_Q` on the right half), and in-loop
//!    deblocking ([`loopfilter`], level 10 with default mode/ref deltas).
//!
//! Layers 1–7 are complete; layer 8 covers multi-ref motion (including compound),
//! rectangular partitions, selectable transforms up to 32×32, switchable
//! interpolation, tile columns, segmentation, and loop filter (ffmpeg-verified
//! in `tests/webm.rs`).

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

pub mod bool_coder;
pub mod encoder;
pub mod inter;
pub mod intra;
pub mod keyframe;
pub mod loopfilter;
pub mod mc;
pub mod me;
pub mod mv;
pub mod quant;
pub mod tables;
pub mod tokens;
pub mod transform;

pub use bool_coder::BoolEncoder;
pub use encoder::{encode_intra_frame, Reconstruction};
pub use inter::{
    encode_inter_frame, encode_inter_frame_altref, encode_inter_frame_compound,
    encode_inter_frame_golden, encode_inter_frame_refresh, encode_inter_newmv_residual,
    encode_inter_newmv_skip, encode_inter_residual, encode_inter_zeromv_skip,
};
pub use keyframe::encode_intra_gray;

/// VP9 forbids a compressed frame from ending with `0b110xxxxx` — that pattern
/// marks a superframe index. Append a padding `0` when needed (libvpx / spec).
pub(crate) fn finalize_frame(mut bits: Vec<u8>) -> Vec<u8> {
    if bits.last().is_some_and(|&b| b & 0xe0 == 0xc0) {
        bits.push(0);
    }
    bits
}
