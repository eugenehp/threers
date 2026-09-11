//! HEVC deblocking thresholds (§8.7.2, Table 8-12).
//!
//! `β'` and `t'C` indexed by the quantizer, in 8-bit form (the spec scales them
//! by the bit depth). Both are flat zero below `Q = 16` and `Q = 18`
//! respectively, which is why the filter does nothing at high quality.
//!
//! Recalled and then checked byte-for-byte against three independent decoders
//! shipped on this machine — libde265, x265 and ffmpeg's libavcodec all carry
//! the pair adjacent in their data segments, and all three agree.

/// `β'` per `Q`, `Q` in `0..=51`.
pub const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, //
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// `t'C` per `Q`, `Q` in `0..=53` — two wider than `β'`, because the boundary
/// strength shifts the lookup along.
pub const TC: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    1, 1, 1, 1, 1, 1, 1, 1, 1, //
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, //
    7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];
