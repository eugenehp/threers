//! HEVC constant tables. Each was checked against the copy compiled into a
//! shipped libavcodec (the deblocking pair is the encoder's, which was
//! checked the same way).

#[cfg(feature = "video")] // the deblocking filter
pub use crate::codec::hevc::deblock_tables::{BETA, TC};

/// `intraPredAngle` for modes 2..=34 (Table 8-4), indexed by mode − 2.
pub const INTRA_ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9,
    -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/// `invAngle` for modes 11..=25 (Table 8-5), indexed by mode − 11.
pub const INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096,
];

/// Luma interpolation filter taps for quarter positions 1..=3 (Table 8-11).
pub const LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma interpolation filter taps for eighth positions 1..=7 (Table 8-12).
pub const CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// `QpC` as a function of `qPi` for 4:2:0 (Table 8-10).
pub fn chroma_qp(qpi: i32) -> i32 {
    const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
    if qpi < 30 {
        qpi
    } else if qpi <= 43 {
        T[(qpi - 30) as usize]
    } else {
        qpi - 6
    }
}

/// Default 8×8 scaling lists (Table 7-6), in up-right diagonal scan order:
/// intra, inter.
pub const DEFAULT_SCALING: [[u8; 64]; 2] = [
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19,
        20, 21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35,
        35, 31, 29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
    ],
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20,
        20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28,
        28, 28, 28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
    ],
];

/// `ctxIdxMap` for `sig_coeff_flag` in 4×4 blocks (Table 9-50).
pub const CTX_IDX_MAP: [u8; 15] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8];

/// Up-right diagonal scan of an `n × n` block (§6.5.3): scan position →
/// `(x, y)`.
pub const fn diag_scan<const N: usize>(n: usize) -> [(u8, u8); N] {
    let mut out = [(0u8, 0u8); N];
    let mut i = 0;
    let mut x: isize = 0;
    let mut y: isize = 0;
    while i < n * n {
        while y >= 0 {
            if (x as usize) < n && (y as usize) < n {
                out[i] = (x as u8, y as u8);
                i += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    out
}

const fn horiz_scan<const N: usize>(n: usize) -> [(u8, u8); N] {
    let mut out = [(0u8, 0u8); N];
    let mut i = 0;
    while i < n * n {
        out[i] = ((i % n) as u8, (i / n) as u8);
        i += 1;
    }
    out
}

const fn vert_scan<const N: usize>(n: usize) -> [(u8, u8); N] {
    let mut out = [(0u8, 0u8); N];
    let mut i = 0;
    while i < n * n {
        out[i] = ((i / n) as u8, (i % n) as u8);
        i += 1;
    }
    out
}

/// Scans within a 4×4 sub-block, by `scanIdx` (diagonal, horizontal,
/// vertical).
pub const SCAN4: [[(u8, u8); 16]; 3] = [diag_scan::<16>(4), horiz_scan::<16>(4), vert_scan::<16>(4)];

/// Sub-block scans for 1×1, 2×2, 4×4 and 8×8 grids of sub-blocks.
pub const SCAN_SB1: [[(u8, u8); 1]; 3] = [[(0, 0)]; 3];
pub const SCAN_SB2: [[(u8, u8); 4]; 3] = [diag_scan::<4>(2), horiz_scan::<4>(2), vert_scan::<4>(2)];
pub const SCAN_SB4: [(u8, u8); 16] = diag_scan::<16>(4);
pub const SCAN_SB8: [(u8, u8); 64] = diag_scan::<64>(8);

/// The 8×8 diagonal scan, for scaling-list layout.
pub const SCAN8X8: [(u8, u8); 64] = diag_scan::<64>(8);

/// The 32×32 transform matrix (§8.6.4.2). Smaller transforms take every
/// `32 / n`-th row.
pub const DCT32: [[i8; 32]; 32] = {
    const T: [i32; 32] = [
        64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46,
        43, 38, 36, 31, 25, 22, 18, 13, 9, 4,
    ];
    let mut m = [[0i8; 32]; 32];
    let mut k = 0;
    while k < 32 {
        let mut n = 0;
        while n < 32 {
            let v = if k == 0 {
                64
            } else {
                let mut a = ((2 * n + 1) * k) % 128;
                if a > 64 {
                    a = 128 - a;
                }
                if a > 32 {
                    -T[64 - a]
                } else if a == 32 {
                    0
                } else {
                    T[a]
                }
            };
            m[k][n] = v as i8;
            n += 1;
        }
        k += 1;
    }
    m
};

/// The 4×4 DST-VII used for intra luma 4×4 blocks.
pub const DST4: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];
