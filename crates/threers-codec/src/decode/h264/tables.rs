//! H.264 constant tables the decoder needs beyond the encoder's.
//!
//! Each was checked byte-for-byte against the copy compiled into a shipped
//! libavcodec — see `codec::decode` module docs.

/// 4×4 zig-zag: scan position → raster index (Table 8-13).
pub const ZIGZAG4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// 8×8 zig-zag (frame) scan: scan position → raster index (Table 8-14).
pub const ZIGZAG8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// 4×4 field scan: scan position → raster index (Table 8-13).
pub const FIELD4: [usize; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// 8×8 field scan (Table 8-14), as ffmpeg's `field_scan8x8`.
pub const FIELD8: [usize; 64] = [
    0, 8, 16, 1, 9, 24, 32, 17, 2, 25, 40, 48, 56, 33, 10, 3, 18, 41, 49, 57, 26, 11, 4, 19, 34,
    42, 50, 58, 27, 12, 5, 20, 35, 43, 51, 59, 28, 13, 6, 21, 36, 44, 52, 60, 29, 14, 22, 37, 45,
    53, 61, 30, 7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

/// `Default_4x4_Intra` and `Default_4x4_Inter` (Table 7-3), raster order.
pub const DEFAULT_4X4: [[u8; 16]; 2] = [
    [6, 13, 20, 28, 13, 20, 28, 32, 20, 28, 32, 37, 28, 32, 37, 42],
    [10, 14, 20, 24, 14, 20, 24, 27, 20, 24, 27, 30, 24, 27, 30, 34],
];

/// `Default_8x8_Intra` and `Default_8x8_Inter` (Table 7-4), raster order.
pub const DEFAULT_8X8: [[u8; 64]; 2] = [
    [
        6, 10, 13, 16, 18, 23, 25, 27, 10, 11, 16, 18, 23, 25, 27, 29, 13, 16, 18, 23, 25, 27, 29,
        31, 16, 18, 23, 25, 27, 29, 31, 33, 18, 23, 25, 27, 29, 31, 33, 36, 23, 25, 27, 29, 31, 33,
        36, 38, 25, 27, 29, 31, 33, 36, 38, 40, 27, 29, 31, 33, 36, 38, 40, 42,
    ],
    [
        9, 13, 15, 17, 19, 21, 22, 24, 13, 13, 17, 19, 21, 22, 24, 25, 15, 17, 19, 21, 22, 24, 25,
        27, 17, 19, 21, 22, 24, 25, 27, 28, 19, 21, 22, 24, 25, 27, 28, 30, 21, 22, 24, 25, 27, 28,
        30, 32, 22, 24, 25, 27, 28, 30, 32, 33, 24, 25, 27, 28, 30, 32, 33, 35,
    ],
];

/// `coded_block_pattern` from its `me(v)` code number, 4:2:0: `[intra, inter]`
/// (Table 9-4).
pub const CBP_420: [[u8; 48]; 2] = [
    [
        47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26,
        28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
    ],
    [
        0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33,
        34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
    ],
];

/// The same for monochrome, where there is no chroma part.
pub const CBP_400: [[u8; 16]; 2] = [
    [15, 0, 7, 11, 13, 14, 3, 5, 10, 12, 1, 2, 4, 8, 6, 9],
    [0, 1, 2, 4, 8, 3, 5, 10, 12, 15, 7, 11, 13, 14, 6, 9],
];

/// `QPC` for `qPI >= 30` (Table 8-15); below 30 it is `qPI`.
pub const CHROMA_QP: [u8; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

pub fn chroma_qp(qpi: i32) -> i32 {
    if qpi < 30 {
        qpi
    } else {
        CHROMA_QP[(qpi - 30).min(21) as usize] as i32
    }
}

/// `normAdjust4x4` values `v` (8-315).
const V4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// `normAdjust8x8` values `v` (8-318).
const V8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

pub fn norm_adjust4(m: usize, i: usize, j: usize) -> i32 {
    if i.is_multiple_of(2) && j.is_multiple_of(2) {
        V4[m][0]
    } else if i % 2 == 1 && j % 2 == 1 {
        V4[m][1]
    } else {
        V4[m][2]
    }
}

pub fn norm_adjust8(m: usize, i: usize, j: usize) -> i32 {
    if i.is_multiple_of(4) && j.is_multiple_of(4) {
        V8[m][0]
    } else if i % 2 == 1 && j % 2 == 1 {
        V8[m][1]
    } else if i % 4 == 2 && j % 4 == 2 {
        V8[m][2]
    } else if (i.is_multiple_of(4) && j % 2 == 1) || (i % 2 == 1 && j.is_multiple_of(4)) {
        V8[m][3]
    } else if (i.is_multiple_of(4) && j % 4 == 2) || (i % 4 == 2 && j.is_multiple_of(4)) {
        V8[m][4]
    } else {
        V8[m][5]
    }
}

/// `significant_coeff_flag` context increment for 8×8 blocks, frame coded
/// (Table 9-43), by scan position.
pub const SIG8: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8,
    7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12,
    13, 11, 14, 10, 12,
];

/// `significant_coeff_flag` context increment for 8×8 blocks, field coded.
pub const SIG8_FIELD: [u8; 63] = [
    0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5, 6, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8,
    11, 12, 11, 9, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8, 13, 13, 9, 9, 10, 10, 8, 13, 13, 9,
    9, 10, 10, 14, 14, 14, 14, 14,
];

/// `last_significant_coeff_flag` context increment for 8×8 blocks.
pub const LAST8: [u8; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8,
    8,
];

/// Block index `blkIdx` (decoding order) → `(x, y)` of the 4×4 block, in
/// 4-sample units within the macroblock.
pub const BLK_XY: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// The inverse of [`BLK_XY`]: raster 4×4 index → `blkIdx`.
pub const RASTER_TO_BLK: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
