//! Fixed VP9 probability tables and token trees (from the VP9 spec / libvpx
//! `vp9_entropymode.c`). Keyframe partition/mode probabilities, default skip
//! probabilities, and the tree structures used to binarize tree-coded symbols.

// `INTER_OFFSET(mode)` is `mode - NEARESTMV` in libvpx, so the zero offset is
// spelled `NEARESTMV - NEARESTMV`. Writing the 0 directly would hide which
// mode the tree leaf belongs to.
#![allow(clippy::eq_op)]

// ---- intra prediction modes (VP9 `PREDICTION_MODE`) ----
pub const DC_PRED: i8 = 0;
pub const V_PRED: i8 = 1;
pub const H_PRED: i8 = 2;
pub const D45_PRED: i8 = 3;
pub const D135_PRED: i8 = 4;
pub const D117_PRED: i8 = 5;
pub const D153_PRED: i8 = 6;
pub const D207_PRED: i8 = 7;
pub const D63_PRED: i8 = 8;
pub const TM_PRED: i8 = 9;

/// Modes the prediction-only encoder selects among.
pub const PRED_MODES: [i8; 3] = [DC_PRED, V_PRED, H_PRED];

/// `vp9_intra_mode_tree` — binarizes one of the 10 intra modes. A negative entry
/// is a leaf (`-mode`); a positive entry is the index of a child node pair.
pub const INTRA_MODE_TREE: [i8; 18] = [
    -DC_PRED, 2, // 0
    -TM_PRED, 4, // 2
    -V_PRED, 6, // 4
    8, 12, // 6
    -H_PRED, 10, // 8
    -D135_PRED, -D117_PRED, // 10
    -D45_PRED, 14, // 12
    -D63_PRED, 16, // 14
    -D153_PRED, -D207_PRED, // 16
];

// ---- partitions (VP9 `PARTITION_TYPE`) ----
pub const PARTITION_NONE: i8 = 0;
pub const PARTITION_HORZ: i8 = 1;
pub const PARTITION_VERT: i8 = 2;
pub const PARTITION_SPLIT: i8 = 3;

/// `vp9_partition_tree` — binarizes one of the 4 partition types.
pub const PARTITION_TREE: [i8; 6] = [
    -PARTITION_NONE,
    2,
    -PARTITION_HORZ,
    4,
    -PARTITION_VERT,
    -PARTITION_SPLIT,
];

/// `vp9_kf_partition_probs[PARTITION_CONTEXTS][PARTITION_TYPES - 1]` — the fixed
/// keyframe partition probabilities, grouped by block size (8×8, 16×16, 32×32,
/// 64×64), each with four above/left-split contexts.
pub const KF_PARTITION_PROBS: [[u8; 3]; 16] = [
    [158, 97, 94],
    [93, 24, 99],
    [85, 119, 44],
    [62, 59, 67],
    [149, 53, 53],
    [94, 20, 48],
    [83, 53, 24],
    [52, 18, 18],
    [150, 40, 39],
    [78, 12, 26],
    [67, 33, 11],
    [24, 7, 5],
    [174, 35, 49],
    [68, 11, 27],
    [57, 15, 9],
    [12, 3, 3],
];

/// `default_skip_probs[SKIP_CONTEXTS]`.
pub const DEFAULT_SKIP_PROBS: [u8; 3] = [192, 128, 64];

/// `default_partition_probs` — inter / past-independence partition probs.
pub const DEFAULT_PARTITION_PROBS: [[u8; 3]; 16] = [
    [199, 122, 141],
    [147, 63, 159],
    [148, 133, 118],
    [121, 104, 114],
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

/// Inter modes as `PREDICTION_MODE` values (libvpx).
pub const NEARESTMV: i8 = 10;
pub const NEARMV: i8 = 11;
pub const ZEROMV: i8 = 12;
pub const NEWMV: i8 = 13;

/// VP9 `MV_REFERENCE_FRAME` values.
pub const INTRA_FRAME: i8 = 0;
pub const LAST_FRAME: i8 = 1;
pub const GOLDEN_FRAME: i8 = 2;
pub const ALTREF_FRAME: i8 = 3;

/// `INTER_OFFSET(mode) = mode - NEARESTMV` — token values for [`INTER_MODE_TREE`].
pub const INTER_OFFSET_NEARESTMV: i8 = NEARESTMV - NEARESTMV; // 0
pub const INTER_OFFSET_NEARMV: i8 = NEARMV - NEARESTMV; // 1
pub const INTER_OFFSET_ZEROMV: i8 = ZEROMV - NEARESTMV; // 2
pub const INTER_OFFSET_NEWMV: i8 = NEWMV - NEARESTMV; // 3

/// `vp9_inter_mode_tree` — leaves are `-INTER_OFFSET(mode)`.
pub const INTER_MODE_TREE: [i8; 6] = [
    -INTER_OFFSET_ZEROMV,
    2,
    -(NEARESTMV - NEARESTMV), // 0
    4,
    -(NEARMV - NEARESTMV),
    -(NEWMV - NEARESTMV),
];

/// `default_inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1]`.
pub const DEFAULT_INTER_MODE_PROBS: [[u8; 3]; 7] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];

/// `default_intra_inter_p[INTRA_INTER_CONTEXTS]`.
pub const DEFAULT_INTRA_INTER_PROBS: [u8; 4] = [9, 102, 187, 225];

/// `default_single_ref_p[REF_CONTEXTS][2]`.
pub const DEFAULT_SINGLE_REF_PROBS: [[u8; 2]; 5] =
    [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]];

/// `default_comp_inter_p[COMP_INTER_CONTEXTS]` — compound vs single.
pub const DEFAULT_COMP_INTER_PROBS: [u8; 5] = [239, 183, 119, 96, 41];

/// `default_comp_ref_p[REF_CONTEXTS]` — variable ref LAST vs GOLDEN.
pub const DEFAULT_COMP_REF_PROBS: [u8; 5] = [50, 126, 123, 221, 226];

/// `default_tx_probs` — used when `tx_mode == TX_MODE_SELECT`.
/// `p32x32[TX_SIZE_CONTEXTS][TX_SIZES-1]`, `p16x16[...][TX_SIZES-2]`, `p8x8[...][TX_SIZES-3]`.
pub const DEFAULT_TX_PROBS_32: [[u8; 3]; 2] = [[3, 136, 37], [5, 52, 13]];
pub const DEFAULT_TX_PROBS_16: [[u8; 2]; 2] = [[20, 152], [15, 101]];
pub const DEFAULT_TX_PROBS_8: [[u8; 1]; 2] = [[100], [66]];

/// `default_switchable_interp_prob[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS-1]`.
pub const DEFAULT_SWITCHABLE_INTERP_PROBS: [[u8; 2]; 4] =
    [[235, 162], [36, 255], [34, 3], [149, 144]];

/// `vp9_switchable_interp_tree` — `{ -EIGHTTAP, 2, -SMOOTH, -SHARP }`.
pub const SWITCHABLE_INTERP_TREE: [i8; 4] = [0, 2, -1, -2];

/// TX size enum values.
pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;

/// `DIFF_UPDATE_PROB` / `MV_UPDATE_PROB` — probability of a probability update.
pub const DIFF_UPDATE_PROB: u8 = 252;

/// `mode_2_counter[MB_MODE_COUNT]` entries used for inter-mode context.
pub fn mode_2_counter(mode: i8) -> i32 {
    match mode {
        NEARESTMV | NEARMV => 0,
        ZEROMV => 3,
        NEWMV => 1,
        _ => 9, // intra
    }
}

/// `counter_to_context` → `motion_vector_context` index into inter-mode probs.
pub fn counter_to_inter_mode_ctx(counter: i32) -> usize {
    // Mirrors libvpx `counter_to_context` / `motion_vector_context`.
    const BOTH_ZERO: usize = 0;
    const ZERO_PLUS_PREDICTED: usize = 1;
    const BOTH_PREDICTED: usize = 2;
    const NEW_PLUS_NON_INTRA: usize = 3;
    const BOTH_NEW: usize = 4;
    const INTRA_PLUS_NON_INTRA: usize = 5;
    const BOTH_INTRA: usize = 6;
    match counter {
        0 => BOTH_PREDICTED,
        1 | 4 => NEW_PLUS_NON_INTRA,
        2 => BOTH_NEW,
        3 => ZERO_PLUS_PREDICTED,
        6 => BOTH_ZERO,
        9..=10 | 12 => INTRA_PLUS_NON_INTRA,
        18 => BOTH_INTRA,
        _ => BOTH_PREDICTED, // unused / invalid counters
    }
}

/// `vp9_kf_y_mode_prob[above][left][9]` — keyframe Y-mode tree probabilities.
pub const KF_Y_MODE_PROB: [[[u8; 9]; 10]; 10] = [
    [
        [137, 30, 42, 148, 151, 207, 70, 52, 91],
        [92, 45, 102, 136, 116, 180, 74, 90, 100],
        [73, 32, 19, 187, 222, 215, 46, 34, 100],
        [91, 30, 32, 116, 121, 186, 93, 86, 94],
        [72, 35, 36, 149, 68, 206, 68, 63, 105],
        [73, 31, 28, 138, 57, 124, 55, 122, 151],
        [67, 23, 21, 140, 126, 197, 40, 37, 171],
        [86, 27, 28, 128, 154, 212, 45, 43, 53],
        [74, 32, 27, 107, 86, 160, 63, 134, 102],
        [59, 67, 44, 140, 161, 202, 78, 67, 119],
    ],
    [
        [63, 36, 126, 146, 123, 158, 60, 90, 96],
        [43, 46, 168, 134, 107, 128, 69, 142, 92],
        [44, 29, 68, 159, 201, 177, 50, 57, 77],
        [58, 38, 76, 114, 97, 172, 78, 133, 92],
        [46, 41, 76, 140, 63, 184, 69, 112, 57],
        [38, 32, 85, 140, 46, 112, 54, 151, 133],
        [39, 27, 61, 131, 110, 175, 44, 75, 136],
        [52, 30, 74, 113, 130, 175, 51, 64, 58],
        [47, 35, 80, 100, 74, 143, 64, 163, 74],
        [36, 61, 116, 114, 128, 162, 80, 125, 82],
    ],
    [
        [82, 26, 26, 171, 208, 204, 44, 32, 105],
        [55, 44, 68, 166, 179, 192, 57, 57, 108],
        [42, 26, 11, 199, 241, 228, 23, 15, 85],
        [68, 42, 19, 131, 160, 199, 55, 52, 83],
        [58, 50, 25, 139, 115, 232, 39, 52, 118],
        [50, 35, 33, 153, 104, 162, 64, 59, 131],
        [44, 24, 16, 150, 177, 202, 33, 19, 156],
        [55, 27, 12, 153, 203, 218, 26, 27, 49],
        [53, 49, 21, 110, 116, 168, 59, 80, 76],
        [38, 72, 19, 168, 203, 212, 50, 50, 107],
    ],
    [
        [103, 26, 36, 129, 132, 201, 83, 80, 93],
        [59, 38, 83, 112, 103, 162, 98, 136, 90],
        [62, 30, 23, 158, 200, 207, 59, 57, 50],
        [67, 30, 29, 84, 86, 191, 102, 91, 59],
        [60, 32, 33, 112, 71, 220, 64, 89, 104],
        [53, 26, 34, 130, 56, 149, 84, 120, 103],
        [53, 21, 23, 133, 109, 210, 56, 77, 172],
        [77, 19, 29, 112, 142, 228, 55, 66, 36],
        [61, 29, 29, 93, 97, 165, 83, 175, 162],
        [47, 47, 43, 114, 137, 181, 100, 99, 95],
    ],
    [
        [69, 23, 29, 128, 83, 199, 46, 44, 101],
        [53, 40, 55, 139, 69, 183, 61, 80, 110],
        [40, 29, 19, 161, 180, 207, 43, 24, 91],
        [60, 34, 19, 105, 61, 198, 53, 64, 89],
        [52, 31, 22, 158, 40, 209, 58, 62, 89],
        [44, 31, 29, 147, 46, 158, 56, 102, 198],
        [35, 19, 12, 135, 87, 209, 41, 45, 167],
        [55, 25, 21, 118, 95, 215, 38, 39, 66],
        [51, 38, 25, 113, 58, 164, 70, 93, 97],
        [47, 54, 34, 146, 108, 203, 72, 103, 151],
    ],
    [
        [64, 19, 37, 156, 66, 138, 49, 95, 133],
        [46, 27, 80, 150, 55, 124, 55, 121, 135],
        [36, 23, 27, 165, 149, 166, 54, 64, 118],
        [53, 21, 36, 131, 63, 163, 60, 109, 81],
        [40, 26, 35, 154, 40, 185, 51, 97, 123],
        [35, 19, 34, 179, 19, 97, 48, 129, 124],
        [36, 20, 26, 136, 62, 164, 33, 77, 154],
        [45, 18, 32, 130, 90, 157, 40, 79, 91],
        [45, 26, 28, 129, 45, 129, 49, 147, 123],
        [38, 44, 51, 136, 74, 162, 57, 97, 121],
    ],
    [
        [75, 17, 22, 136, 138, 185, 32, 34, 166],
        [56, 39, 58, 133, 117, 173, 48, 53, 187],
        [35, 21, 12, 161, 212, 207, 20, 23, 145],
        [56, 29, 19, 117, 109, 181, 55, 68, 112],
        [47, 29, 17, 153, 64, 220, 59, 51, 114],
        [46, 16, 24, 136, 76, 147, 41, 64, 172],
        [34, 17, 11, 108, 152, 187, 13, 15, 209],
        [51, 24, 14, 115, 133, 209, 32, 26, 104],
        [55, 30, 18, 122, 79, 179, 44, 88, 116],
        [37, 49, 25, 129, 168, 164, 41, 54, 148],
    ],
    [
        [82, 22, 32, 127, 143, 213, 39, 41, 70],
        [62, 44, 61, 123, 105, 189, 48, 57, 64],
        [47, 25, 17, 175, 222, 220, 24, 30, 86],
        [68, 36, 17, 106, 102, 206, 59, 74, 74],
        [57, 39, 23, 151, 68, 216, 55, 63, 58],
        [49, 30, 35, 141, 70, 168, 82, 40, 115],
        [51, 25, 15, 136, 129, 202, 38, 35, 139],
        [68, 26, 16, 111, 141, 215, 29, 28, 28],
        [59, 39, 19, 114, 75, 180, 77, 104, 42],
        [40, 61, 26, 126, 152, 206, 61, 59, 93],
    ],
    [
        [78, 23, 39, 111, 117, 170, 74, 124, 94],
        [48, 34, 86, 101, 92, 146, 78, 179, 134],
        [47, 22, 24, 138, 187, 178, 68, 69, 59],
        [56, 25, 33, 105, 112, 187, 95, 177, 129],
        [48, 31, 27, 114, 63, 183, 82, 116, 56],
        [43, 28, 37, 121, 63, 123, 61, 192, 169],
        [42, 17, 24, 109, 97, 177, 56, 76, 122],
        [58, 18, 28, 105, 139, 182, 70, 92, 63],
        [46, 23, 32, 74, 86, 150, 67, 183, 88],
        [36, 38, 48, 92, 122, 165, 88, 137, 91],
    ],
    [
        [65, 70, 60, 155, 159, 199, 61, 60, 81],
        [44, 78, 115, 132, 119, 173, 71, 112, 93],
        [39, 38, 21, 184, 227, 206, 42, 32, 64],
        [58, 47, 36, 124, 137, 193, 80, 82, 78],
        [49, 50, 35, 144, 95, 205, 63, 78, 59],
        [41, 53, 52, 148, 71, 142, 65, 128, 51],
        [40, 36, 28, 143, 143, 202, 40, 55, 137],
        [52, 34, 29, 129, 183, 227, 42, 35, 43],
        [42, 44, 44, 104, 105, 164, 64, 130, 80],
        [43, 81, 53, 140, 169, 204, 68, 84, 72],
    ],
];

/// `vp9_kf_uv_mode_prob[y_mode][9]`.
pub const KF_UV_MODE_PROB: [[u8; 9]; 10] = [
    [144, 11, 54, 157, 195, 130, 46, 58, 108],
    [118, 15, 123, 148, 131, 101, 44, 93, 131],
    [113, 12, 23, 188, 226, 142, 26, 32, 125],
    [120, 11, 50, 123, 163, 135, 64, 77, 103],
    [113, 9, 36, 155, 111, 157, 32, 44, 161],
    [116, 9, 55, 176, 76, 96, 37, 61, 149],
    [115, 9, 28, 141, 161, 167, 21, 25, 193],
    [120, 12, 32, 145, 195, 142, 32, 38, 86],
    [116, 12, 64, 120, 140, 125, 49, 115, 121],
    [102, 19, 66, 162, 182, 122, 35, 59, 128],
];

/// Backward-compat aliases used by the gray keyframe path.
pub const KF_Y_MODE_PROB_DC_DC: [u8; 9] = KF_Y_MODE_PROB[DC_PRED as usize][DC_PRED as usize];
pub const KF_UV_MODE_PROB_DC: [u8; 9] = KF_UV_MODE_PROB[DC_PRED as usize];

/// `partition_context_lookup[BLOCK_SIZES].{above,left}` (libvpx).
/// Indexed by square block: 0=4×4 … 3=8×8 … 6=16×16 … 9=32×32 … 12=64×64.
pub const PARTITION_CONTEXT_LOOKUP: [(u8, u8); 13] = [
    (15, 15), // 4×4
    (15, 14), // 4×8
    (14, 15), // 8×4
    (14, 14), // 8×8
    (14, 12), // 8×16
    (12, 14), // 16×8
    (12, 12), // 16×16
    (12, 8),  // 16×32
    (8, 12),  // 32×16
    (8, 8),   // 32×32
    (8, 0),   // 32×64
    (0, 8),   // 64×32
    (0, 0),   // 64×64
];

/// Square block sizes we use (pixels on a side): 8, 16, 32, 64.
pub fn bsl_from_px(px: u32) -> usize {
    match px {
        8 => 0,
        16 => 1,
        32 => 2,
        64 => 3,
        _ => panic!("unsupported block size {px}"),
    }
}

pub fn partition_ctx_index(px: u32) -> usize {
    // 8→3, 16→6, 32→9, 64→12 in PARTITION_CONTEXT_LOOKUP
    match px {
        8 => 3,
        16 => 6,
        32 => 9,
        64 => 12,
        _ => panic!("unsupported block size {px}"),
    }
}

/// `partition_context_lookup` index for a (possibly rectangular) block.
pub fn partition_ctx_index_wh(bw: u32, bh: u32) -> usize {
    match (bw, bh) {
        (8, 8) => 3,
        (8, 16) => 4,
        (16, 8) => 5,
        (16, 16) => 6,
        (16, 32) => 7,
        (32, 16) => 8,
        (32, 32) => 9,
        (32, 64) => 10,
        (64, 32) => 11,
        (64, 64) => 12,
        _ => panic!("unsupported block {bw}x{bh}"),
    }
}

/// `max_txsize_lookup` for square or rectangular blocks.
pub fn max_tx_size_for_bsize(bsize_px: u32) -> u8 {
    max_tx_size_wh(bsize_px, bsize_px)
}

pub fn max_tx_size_wh(bw: u32, bh: u32) -> u8 {
    match bw.min(bh) {
        8 => TX_8X8,
        16 => TX_16X16,
        _ => TX_32X32, // 32 or 64 on the short edge
    }
}

/// Pixel edge length for a TX size enum value.
pub fn tx_size_px(tx: u8) -> usize {
    4 << tx
}

/// `uv_txsize_lookup` for 4:2:0 square or rectangular blocks.
pub fn uv_tx_size_420(bsize_px: u32, y_tx: u8) -> u8 {
    uv_tx_size_wh(bsize_px, bsize_px, y_tx)
}

pub fn uv_tx_size_wh(bw: u32, bh: u32, y_tx: u8) -> u8 {
    // Indexed like libvpx `uv_txsize_lookup[bsize][y_tx][1][1]` (4:2:0).
    match (bw, bh) {
        (8, 8) => TX_4X4,
        (8, 16) | (16, 8) => TX_4X4, // Y max is TX_8 → UV TX_4
        (16, 16) => {
            if y_tx == TX_4X4 {
                TX_4X4
            } else {
                TX_8X8
            }
        }
        (16, 32) | (32, 16) => match y_tx {
            TX_4X4 => TX_4X4,
            TX_8X8 => TX_8X8,
            _ => TX_8X8, // Y TX_16 → UV TX_8 for these (420)
        },
        (32, 32) => match y_tx {
            TX_4X4 => TX_4X4,
            TX_8X8 => TX_8X8,
            _ => TX_16X16,
        },
        (32, 64) | (64, 32) => match y_tx {
            TX_4X4 => TX_4X4,
            TX_8X8 => TX_8X8,
            TX_16X16 => TX_16X16,
            _ => TX_16X16,
        },
        (64, 64) => y_tx,
        _ => y_tx,
    }
}
