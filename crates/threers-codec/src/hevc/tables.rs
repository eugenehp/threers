//! Constant tables for the HEVC CABAC engine (H.265 §9.3.4.3).
//!
//! These are normative and bit-exact — a single wrong entry silently corrupts
//! every arithmetic-coded bin, so they are transcribed verbatim from the spec.
//!
//! **The round-trip test in [`super::cabac`] cannot check them.** It encodes and
//! decodes with these same constants, so a wrong entry is self-consistent and
//! passes: both sides agree with each other and neither agrees with the
//! standard. Two entries were wrong for exactly this reason (`transIdxLps[28]`
//! and `rangeTabLps[31][0]`), and only an external decoder caught it — see
//! `tests/hevc_residual_conformance.rs`, which decodes with ffmpeg.
//!
//! [`tables_match_the_spec`](self) checks the structural invariants that hold
//! independently of the values, which is what would have caught one of the two.

/// `rangeTabLps[pStateIdx][qRangeIdx]` (Table 9-46): the LPS sub-range for each
/// probability state and the 2-bit quantized current range.
pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// `transIdxLps[pStateIdx]` (Table 9-47): next probability state after coding an
/// LPS.
pub const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// `transIdxMps[pStateIdx]` (Table 9-47): next probability state after coding an
/// MPS.
pub const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Structural invariants of the normative tables. These hold regardless of
    /// the exact probability model, so they catch a transcription slip without
    /// needing a reference decoder — `transIdxLps[28] = 23` was found this way.
    #[test]
    fn tables_match_the_spec() {
        // transIdxLps is non-decreasing: a less-probable state never becomes
        // *more* probable after coding an LPS.
        for i in 1..63 {
            assert!(
                TRANS_IDX_LPS[i] >= TRANS_IDX_LPS[i - 1],
                "transIdxLps decreases at {i}: {} -> {}",
                TRANS_IDX_LPS[i - 1],
                TRANS_IDX_LPS[i]
            );
        }
        assert_eq!(TRANS_IDX_LPS[63], 63, "state 63 is the LPS fixed point");

        // transIdxMps walks up one state per MPS and saturates at 62, with 63
        // (the terminating state) mapping to itself.
        for i in 0..63 {
            assert_eq!(TRANS_IDX_MPS[i], (i as u8 + 1).min(62), "transIdxMps[{i}]");
        }
        assert_eq!(TRANS_IDX_MPS[63], 63);

        // rangeTabLps shrinks as the LPS gets less probable (down a column) and
        // grows with the quantized range (across a row).
        for i in 0..63 {
            for q in 0..4 {
                if i > 0 {
                    assert!(
                        RANGE_TAB_LPS[i][q] <= RANGE_TAB_LPS[i - 1][q],
                        "rangeTabLps column {q} increases at state {i}"
                    );
                }
                if q > 0 {
                    assert!(
                        RANGE_TAB_LPS[i][q] > RANGE_TAB_LPS[i][q - 1],
                        "rangeTabLps row {i} not increasing at q={q}"
                    );
                }
            }
        }
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
    }
}
