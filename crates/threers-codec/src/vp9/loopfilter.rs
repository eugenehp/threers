//! VP9 8-bit in-loop deblocking filter — a from-scratch port of libvpx
//! v1.14.1's `vpx_dsp/loopfilter.c` (`filter4`/`filter8`/`filter16`,
//! `vpx_lpf_*_c`) and `vp9/common/vp9_loopfilter.c` (`update_sharpness`,
//! `vp9_loop_filter_frame_init`, `vp9_build_mask`, `vp9_adjust_mask`,
//! `vp9_filter_block_plane_ss00`/`ss11`).
//!
//! This module intentionally supports a *reduced* surface compared to
//! libvpx: no highbitdepth, no SIMD "dual" batching (the `_dual_c` reference
//! functions are themselves just two calls to the non-dual function, so
//! skipping them is bit-exact). Segmentation may alter per-block Q via
//! `SEG_LVL_ALT_Q` in the encoder; the loop filter itself still uses the
//! frame-level filter level with default mode/ref deltas (no `SEG_LVL_ALT_L`).
//! Per-ref/mode filter levels are resolved by the caller into [`LfMi::level`].
//! Square and rectangular coding blocks (8×8 through 64×64, including
//! HORZ/VERT partitions) are handled via the full `BLOCK_SIZES` mask tables.
//!
//! Filtering order follows libvpx exactly: superblocks are visited in raster
//! order, and within each 64×64 superblock the luma plane is filtered
//! (vertical edges, then horizontal edges) before the two chroma planes —
//! each of which is itself filtered vertical-then-horizontal. This ordering
//! matters for bit-exactness because a superblock's left/top edges read
//! pixels already rewritten by its already-processed neighbours.
//!
//! When mode/ref deltas are enabled (libvpx defaults after a keyframe), each
//! MI's effective level is `clamp(base + ref_δ·scale + mode_δ·scale)` with
//! `scale = 1 << (base >> 5)`. Pass the resolved level in [`LfMi::level`].

/// Per-MI info needed for mask building.
pub struct LfMi {
    pub skip: bool,
    pub is_inter: bool,
    /// 0=4x4, 1=8x8, 2=16x16, 3=32x32.
    pub tx_size_y: u8,
    /// Block width/height in MI units (8×8).
    pub bw_mi: u8,
    pub bh_mi: u8,
    /// `BLOCK_SIZES` index into the LF mask tables (3=8×8 … 12=64×64).
    pub bsize_idx: usize,
    /// True only for the top-left MI of a coding block.
    pub block_origin: bool,
    /// Effective loop-filter level for this MI (`0` skips filtering its edges).
    pub level: u8,
}

/// Default `ref_deltas` after `vp9_setup_past_independence` / `set_default_lf_deltas`.
/// Indexed by `INTRA_FRAME`..=`ALTREF_FRAME`.
pub const DEFAULT_LF_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];

/// Default `mode_deltas`: `[ZEROMV, other inter]`.
pub const DEFAULT_LF_MODE_DELTAS: [i8; 2] = [0, 0];

/// `mode_lf_lut[mode]` — 0 for ZEROMV (and intra), 1 for other inter modes.
pub fn mode_lf_delta_idx(mode: i8) -> usize {
    match mode {
        // NEARESTMV=10, NEARMV=11, ZEROMV=12, NEWMV=13
        12 => 0, // ZEROMV
        10 | 11 | 13 => 1,
        _ => 0, // intra
    }
}

/// Effective filter level for one MI (`vp9_loop_filter_frame_init` + `get_filter_level`).
pub fn lf_level_for_mi(base: u8, ref_frame: i8, mode: i8) -> u8 {
    if base == 0 {
        return 0;
    }
    let scale = 1i32 << (base >> 5);
    let rf = ref_frame.clamp(0, 3) as usize;
    let md = mode_lf_delta_idx(mode);
    let lvl = i32::from(base)
        + i32::from(DEFAULT_LF_REF_DELTAS[rf]) * scale
        + i32::from(DEFAULT_LF_MODE_DELTAS[md]) * scale;
    lvl.clamp(0, 63) as u8
}

const MI_BLOCK_SIZE: usize = 8; // MI units (8x8 each) per 64x64 superblock side.
const MAX_LOOP_FILTER: usize = 63;

// ---------------------------------------------------------------------------
// Sharpness -> per-level thresholds (vp9_loopfilter.c: update_sharpness,
// vp9_loop_filter_init).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct LoopFilterThresh {
    mblim: u8,
    lim: u8,
    hev_thr: u8,
}

fn build_thresholds(sharpness: u8) -> [LoopFilterThresh; MAX_LOOP_FILTER + 1] {
    let mut out = [LoopFilterThresh::default(); MAX_LOOP_FILTER + 1];
    let sharpness = sharpness as i32;
    for lvl in 0..=MAX_LOOP_FILTER {
        let lvl_i = lvl as i32;
        let mut block_inside_limit = lvl_i >> ((sharpness > 0) as i32 + (sharpness > 4) as i32);
        if sharpness > 0 && block_inside_limit > 9 - sharpness {
            block_inside_limit = 9 - sharpness;
        }
        if block_inside_limit < 1 {
            block_inside_limit = 1;
        }
        out[lvl].lim = block_inside_limit as u8;
        out[lvl].mblim = (2 * (lvl_i + 2) + block_inside_limit) as u8;
        out[lvl].hev_thr = (lvl_i >> 4) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// Mask-building tables (vp9_loopfilter.c) — full `BLOCK_SIZES` (indices 0..=12).
// We only emit blocks at indices 3..=12 (8×8 and up).
// ---------------------------------------------------------------------------

const LEFT_PRED_MASK_Y: [u64; 13] = [
    0x0000_0000_0000_0001, // 4×4
    0x0000_0000_0000_0001, // 4×8
    0x0000_0000_0000_0001, // 8×4
    0x0000_0000_0000_0001, // 8×8
    0x0000_0000_0000_0101, // 8×16
    0x0000_0000_0000_0001, // 16×8
    0x0000_0000_0000_0101, // 16×16
    0x0000_0000_0101_0101, // 16×32
    0x0000_0000_0000_0101, // 32×16
    0x0000_0000_0101_0101, // 32×32
    0x0101_0101_0101_0101, // 32×64
    0x0000_0000_0101_0101, // 64×32
    0x0101_0101_0101_0101, // 64×64
];
const ABOVE_PRED_MASK_Y: [u64; 13] = [
    0x1, 0x1, 0x1, 0x1, 0x1, 0x3, 0x3, 0x3, 0xf, 0xf, 0xf, 0xff, 0xff,
];
const SIZE_MASK_Y: [u64; 13] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001, // 8×8
    0x0000_0000_0000_0101, // 8×16
    0x0000_0000_0000_0003, // 16×8
    0x0000_0000_0000_0303, // 16×16
    0x0000_0000_0303_0303, // 16×32
    0x0000_0000_0000_0f0f, // 32×16
    0x0000_0000_0f0f_0f0f, // 32×32
    0x0f0f_0f0f_0f0f_0f0f, // 32×64
    0x0000_0000_ffff_ffff, // 64×32
    0xffff_ffff_ffff_ffff, // 64×64
];

const LEFT_PRED_MASK_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0011, 0x0001, 0x0011, 0x1111, 0x0011,
    0x1111,
];
const ABOVE_PRED_MASK_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0003, 0x0003, 0x0003, 0x000f,
    0x000f,
];
const SIZE_MASK_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0011, 0x0003, 0x0033, 0x3333, 0x00ff,
    0xffff,
];

// Indexed by TX_SIZE (0=4x4,1=8x8,2=16x16,3=32x32).
const LEFT_TXFORM_MASK_Y: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x5555_5555_5555_5555,
    0x1111_1111_1111_1111,
];
const ABOVE_TXFORM_MASK_Y: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x00ff_00ff_00ff_00ff,
    0x0000_00ff_0000_00ff,
];
const LEFT_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x5555, 0x1111];
const ABOVE_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x0f0f, 0x000f];

const LEFT_BORDER_Y: u64 = 0x1111_1111_1111_1111;
const ABOVE_BORDER_Y: u64 = 0x0000_00ff_0000_00ff;
const LEFT_BORDER_UV: u16 = 0x1111;
const ABOVE_BORDER_UV: u16 = 0x000f;

/// `first_block_in_16x16[8][8]`: true where both `row` and `col` (mod 8) are even.
#[inline]
fn first_block_in_16x16(row_in_sb: usize, col_in_sb: usize) -> bool {
    row_in_sb.is_multiple_of(2) && col_in_sb.is_multiple_of(2)
}

/// Per-superblock loop filter mask state (`LOOP_FILTER_MASK`).
#[derive(Clone)]
struct LoopFilterMask {
    left_y: [u64; 4],
    above_y: [u64; 4],
    int_4x4_y: u64,
    lfl_y: [u8; 64],
    left_uv: [u16; 4],
    above_uv: [u16; 4],
    int_4x4_uv: u16,
}

impl Default for LoopFilterMask {
    fn default() -> Self {
        LoopFilterMask {
            left_y: [0; 4],
            above_y: [0; 4],
            int_4x4_y: 0,
            lfl_y: [0; 64],
            left_uv: [0; 4],
            above_uv: [0; 4],
            int_4x4_uv: 0,
        }
    }
}

/// `vp9_build_mask` for any `BLOCK_SIZES` entry. `level == 0` skips the block.
#[allow(clippy::too_many_arguments)]
fn build_mask(
    level: u8,
    is_inter: bool,
    skip: bool,
    tx_size_y: u8,
    bsize_idx: usize,
    bw_mi: usize,
    bh_mi: usize,
    row_in_sb: usize,
    col_in_sb: usize,
    lfm: &mut LoopFilterMask,
) {
    if level == 0 {
        return;
    }
    // TX UV capped by the shorter edge in MI units (≈ square log2).
    let tx_cap = (bw_mi.min(bh_mi).trailing_zeros()) as u8;
    let tx_size_uv = tx_size_y.min(tx_cap);

    let shift_y = col_in_sb + (row_in_sb << 3);
    let shift_uv = (col_in_sb >> 1) + ((row_in_sb >> 1) << 2);
    let build_uv = first_block_in_16x16(row_in_sb, col_in_sb);

    let mut index = shift_y;
    for _ in 0..bh_mi {
        for k in 0..bw_mi {
            lfm.lfl_y[index + k] = level;
        }
        index += 8;
    }

    lfm.above_y[tx_size_y as usize] |= ABOVE_PRED_MASK_Y[bsize_idx] << shift_y;
    lfm.left_y[tx_size_y as usize] |= LEFT_PRED_MASK_Y[bsize_idx] << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv as usize] |= ABOVE_PRED_MASK_UV[bsize_idx] << shift_uv;
        lfm.left_uv[tx_size_uv as usize] |= LEFT_PRED_MASK_UV[bsize_idx] << shift_uv;
    }

    if skip && is_inter {
        return;
    }

    lfm.above_y[tx_size_y as usize] |=
        (SIZE_MASK_Y[bsize_idx] & ABOVE_TXFORM_MASK_Y[tx_size_y as usize]) << shift_y;
    lfm.left_y[tx_size_y as usize] |=
        (SIZE_MASK_Y[bsize_idx] & LEFT_TXFORM_MASK_Y[tx_size_y as usize]) << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv as usize] |=
            (SIZE_MASK_UV[bsize_idx] & ABOVE_TXFORM_MASK_UV[tx_size_uv as usize]) << shift_uv;
        lfm.left_uv[tx_size_uv as usize] |=
            (SIZE_MASK_UV[bsize_idx] & LEFT_TXFORM_MASK_UV[tx_size_uv as usize]) << shift_uv;
    }

    if tx_size_y == 0 {
        lfm.int_4x4_y |= SIZE_MASK_Y[bsize_idx] << shift_y;
    }
    if build_uv && tx_size_uv == 0 {
        lfm.int_4x4_uv |= SIZE_MASK_UV[bsize_idx] << shift_uv;
    }
}

/// `vp9_adjust_mask`.
fn adjust_mask(
    mi_rows: usize,
    mi_cols: usize,
    sb_row: usize,
    sb_col: usize,
    lfm: &mut LoopFilterMask,
) {
    lfm.left_y[2] |= lfm.left_y[3];
    lfm.above_y[2] |= lfm.above_y[3];
    lfm.left_uv[2] |= lfm.left_uv[3];
    lfm.above_uv[2] |= lfm.above_uv[3];

    lfm.left_y[1] |= lfm.left_y[0] & LEFT_BORDER_Y;
    lfm.left_y[0] &= !LEFT_BORDER_Y;
    lfm.above_y[1] |= lfm.above_y[0] & ABOVE_BORDER_Y;
    lfm.above_y[0] &= !ABOVE_BORDER_Y;
    lfm.left_uv[1] |= lfm.left_uv[0] & LEFT_BORDER_UV;
    lfm.left_uv[0] &= !LEFT_BORDER_UV;
    lfm.above_uv[1] |= lfm.above_uv[0] & ABOVE_BORDER_UV;
    lfm.above_uv[0] &= !ABOVE_BORDER_UV;

    if sb_row + MI_BLOCK_SIZE > mi_rows {
        let rows = (mi_rows - sb_row) as u32;
        let mask_y: u64 = (1u64 << (rows * 8)) - 1;
        let mask_uv: u16 = (((1u32 << (((rows + 1) >> 1) * 4)) - 1) & 0xffff) as u16;

        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv;

        if rows == 1 {
            lfm.above_uv[1] |= lfm.above_uv[2];
            lfm.above_uv[2] = 0;
        }
        if rows == 5 {
            lfm.above_uv[1] |= lfm.above_uv[2] & 0xff00;
            lfm.above_uv[2] &= !0xff00u16;
        }
    }

    if sb_col + MI_BLOCK_SIZE > mi_cols {
        let columns = (mi_cols - sb_col) as u32;
        let mask_y: u64 = ((1u64 << columns) - 1).wrapping_mul(0x0101_0101_0101_0101);
        let mask_uv: u16 =
            (((1u32 << ((columns + 1) >> 1)) - 1).wrapping_mul(0x1111) & 0xffff) as u16;
        let mask_uv_int: u16 =
            (((1u32 << (columns >> 1)) - 1).wrapping_mul(0x1111) & 0xffff) as u16;

        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv_int;

        if columns == 1 {
            lfm.left_uv[1] |= lfm.left_uv[2];
            lfm.left_uv[2] = 0;
        }
        if columns == 5 {
            lfm.left_uv[1] |= lfm.left_uv[2] & 0xcccc;
            lfm.left_uv[2] &= !0xccccu16;
        }
    }

    if sb_col == 0 {
        for i in 0..3 {
            lfm.left_y[i] &= 0xfefe_fefe_fefe_fefe;
            lfm.left_uv[i] &= 0xeeee;
        }
    }
}

// ---------------------------------------------------------------------------
// Core pixel filters (vpx_dsp/loopfilter.c: filter4/filter8/filter16).
// ---------------------------------------------------------------------------

#[inline]
fn signed_char_clamp(t: i32) -> i32 {
    t.clamp(-128, 127)
}

#[inline]
fn round_pow2_i(v: i32, n: u32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

#[inline]
fn round_pow2_u(v: u32, n: u32) -> u32 {
    (v + (1 << (n - 1))) >> n
}

#[inline]
fn filter_mask(
    limit: u8,
    blimit: u8,
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
) -> bool {
    let limit = limit as i32;
    let blimit = blimit as i32;
    let d = |a: u8, b: u8| (a as i32 - b as i32).abs();
    d(p3, p2) <= limit
        && d(p2, p1) <= limit
        && d(p1, p0) <= limit
        && d(q1, q0) <= limit
        && d(q2, q1) <= limit
        && d(q3, q2) <= limit
        && d(p0, q0) * 2 + d(p1, q1) / 2 <= blimit
}

#[inline]
fn flat_mask4(thresh: u8, p3: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8, q3: u8) -> bool {
    let thresh = thresh as i32;
    let d = |a: u8, b: u8| (a as i32 - b as i32).abs();
    d(p1, p0) <= thresh
        && d(q1, q0) <= thresh
        && d(p2, p0) <= thresh
        && d(q2, q0) <= thresh
        && d(p3, p0) <= thresh
        && d(q3, q0) <= thresh
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn flat_mask5(
    thresh: u8,
    p4: u8,
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
    q4: u8,
) -> bool {
    let thresh_i = thresh as i32;
    flat_mask4(thresh, p3, p2, p1, p0, q0, q1, q2, q3)
        && (p4 as i32 - p0 as i32).abs() <= thresh_i
        && (q4 as i32 - q0 as i32).abs() <= thresh_i
}

#[inline]
fn hev_mask(thresh: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> bool {
    let thresh = thresh as i32;
    (p1 as i32 - p0 as i32).abs() > thresh || (q1 as i32 - q0 as i32).abs() > thresh
}

/// `filter4`.
fn filter4(mask: bool, thresh: u8, buf: &mut [u8], op1: usize, op0: usize, oq0: usize, oq1: usize) {
    let (p1, p0, q0, q1) = (buf[op1], buf[op0], buf[oq0], buf[oq1]);
    let (ps1, ps0, qs0, qs1) = (
        p1 as i32 - 128,
        p0 as i32 - 128,
        q0 as i32 - 128,
        q1 as i32 - 128,
    );
    let hev = hev_mask(thresh, p1, p0, q0, q1);

    let mut filter = signed_char_clamp(ps1 - qs1);
    filter = if hev { filter } else { 0 };

    filter = signed_char_clamp(filter + 3 * (qs0 - ps0));
    filter = if mask { filter } else { 0 };

    let filter1 = signed_char_clamp(filter + 4) >> 3;
    let filter2 = signed_char_clamp(filter + 3) >> 3;

    buf[oq0] = (signed_char_clamp(qs0 - filter1) + 128) as u8;
    buf[op0] = (signed_char_clamp(ps0 + filter2) + 128) as u8;

    let outer = round_pow2_i(filter1, 1);
    let outer = if hev { 0 } else { outer };

    buf[oq1] = (signed_char_clamp(qs1 - outer) + 128) as u8;
    buf[op1] = (signed_char_clamp(ps1 + outer) + 128) as u8;
}

/// `filter8`.
#[allow(clippy::too_many_arguments)]
fn filter8(
    mask: bool,
    thresh: u8,
    flat: bool,
    buf: &mut [u8],
    op3: usize,
    op2: usize,
    op1: usize,
    op0: usize,
    oq0: usize,
    oq1: usize,
    oq2: usize,
    oq3: usize,
) {
    if flat && mask {
        let p3 = buf[op3] as u32;
        let p2 = buf[op2] as u32;
        let p1 = buf[op1] as u32;
        let p0 = buf[op0] as u32;
        let q0 = buf[oq0] as u32;
        let q1 = buf[oq1] as u32;
        let q2 = buf[oq2] as u32;
        let q3 = buf[oq3] as u32;

        buf[op2] = round_pow2_u(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0, 3) as u8;
        buf[op1] = round_pow2_u(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1, 3) as u8;
        buf[op0] = round_pow2_u(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2, 3) as u8;
        buf[oq0] = round_pow2_u(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3, 3) as u8;
        buf[oq1] = round_pow2_u(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3, 3) as u8;
        buf[oq2] = round_pow2_u(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3, 3) as u8;
    } else {
        filter4(mask, thresh, buf, op1, op0, oq0, oq1);
    }
}

/// `filter16`.
#[allow(clippy::too_many_arguments)]
fn filter16(
    mask: bool,
    thresh: u8,
    flat: bool,
    flat2: bool,
    buf: &mut [u8],
    idx: [usize; 16], // op7,op6,op5,op4,op3,op2,op1,op0,oq0,oq1,oq2,oq3,oq4,oq5,oq6,oq7
) {
    let [op7, op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5, oq6, oq7] = idx;
    if flat2 && flat && mask {
        let p7 = buf[op7] as u32;
        let p6 = buf[op6] as u32;
        let p5 = buf[op5] as u32;
        let p4 = buf[op4] as u32;
        let p3 = buf[op3] as u32;
        let p2 = buf[op2] as u32;
        let p1 = buf[op1] as u32;
        let p0 = buf[op0] as u32;
        let q0 = buf[oq0] as u32;
        let q1 = buf[oq1] as u32;
        let q2 = buf[oq2] as u32;
        let q3 = buf[oq3] as u32;
        let q4 = buf[oq4] as u32;
        let q5 = buf[oq5] as u32;
        let q6 = buf[oq6] as u32;
        let q7 = buf[oq7] as u32;

        buf[op6] = round_pow2_u(p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0, 4) as u8;
        buf[op5] = round_pow2_u(p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1, 4) as u8;
        buf[op4] = round_pow2_u(
            p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2,
            4,
        ) as u8;
        buf[op3] = round_pow2_u(
            p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3,
            4,
        ) as u8;
        buf[op2] = round_pow2_u(
            p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4,
            4,
        ) as u8;
        buf[op1] = round_pow2_u(
            p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5,
            4,
        ) as u8;
        buf[op0] = round_pow2_u(
            p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6,
            4,
        ) as u8;
        buf[oq0] = round_pow2_u(
            p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7,
            4,
        ) as u8;
        buf[oq1] = round_pow2_u(
            p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2,
            4,
        ) as u8;
        buf[oq2] = round_pow2_u(
            p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3,
            4,
        ) as u8;
        buf[oq3] = round_pow2_u(
            p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4,
            4,
        ) as u8;
        buf[oq4] = round_pow2_u(
            p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5,
            4,
        ) as u8;
        buf[oq5] = round_pow2_u(p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6, 4) as u8;
        buf[oq6] = round_pow2_u(p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7, 4) as u8;
    } else {
        filter8(
            mask, thresh, flat, buf, op3, op2, op1, op0, oq0, oq1, oq2, oq3,
        );
    }
}

// ---------------------------------------------------------------------------
// Edge sweeps (`vpx_lpf_{vertical,horizontal}_{4,8,16}_c`), non-dual only.
// `pos` is the index of `q0` (first sample past the edge). `pitch` is the
// step to walk from `q0` to `q1` (1 for a vertical edge, `stride` for a
// horizontal edge); the sweep itself always advances by `stride` for
// vertical edges (down the column) or by `1` for horizontal edges (across
// the row), matching the C reference's inner loops.
// ---------------------------------------------------------------------------

fn lpf_vertical_4(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i * stride;
        let (p3, p2, p1, p0) = (buf[s - 4], buf[s - 3], buf[s - 2], buf[s - 1]);
        let (q0, q1, q2, q3) = (buf[s], buf[s + 1], buf[s + 2], buf[s + 3]);
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        filter4(mask, thresh, buf, s - 2, s - 1, s, s + 1);
    }
}

fn lpf_vertical_8(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i * stride;
        let (p3, p2, p1, p0) = (buf[s - 4], buf[s - 3], buf[s - 2], buf[s - 1]);
        let (q0, q1, q2, q3) = (buf[s], buf[s + 1], buf[s + 2], buf[s + 3]);
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        filter8(
            mask,
            thresh,
            flat,
            buf,
            s - 4,
            s - 3,
            s - 2,
            s - 1,
            s,
            s + 1,
            s + 2,
            s + 3,
        );
    }
}

fn lpf_vertical_16(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i * stride;
        let (p3, p2, p1, p0) = (buf[s - 4], buf[s - 3], buf[s - 2], buf[s - 1]);
        let (q0, q1, q2, q3) = (buf[s], buf[s + 1], buf[s + 2], buf[s + 3]);
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat2 = flat_mask5(
            1,
            buf[s - 8],
            buf[s - 7],
            buf[s - 6],
            buf[s - 5],
            p0,
            q0,
            buf[s + 4],
            buf[s + 5],
            buf[s + 6],
            buf[s + 7],
        );
        filter16(
            mask,
            thresh,
            flat,
            flat2,
            buf,
            [
                s - 8,
                s - 7,
                s - 6,
                s - 5,
                s - 4,
                s - 3,
                s - 2,
                s - 1,
                s,
                s + 1,
                s + 2,
                s + 3,
                s + 4,
                s + 5,
                s + 6,
                s + 7,
            ],
        );
    }
}

fn lpf_horizontal_4(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i;
        let (p3, p2, p1, p0) = (
            buf[s - 4 * stride],
            buf[s - 3 * stride],
            buf[s - 2 * stride],
            buf[s - stride],
        );
        let (q0, q1, q2, q3) = (
            buf[s],
            buf[s + stride],
            buf[s + 2 * stride],
            buf[s + 3 * stride],
        );
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        filter4(mask, thresh, buf, s - 2 * stride, s - stride, s, s + stride);
    }
}

fn lpf_horizontal_8(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i;
        let (p3, p2, p1, p0) = (
            buf[s - 4 * stride],
            buf[s - 3 * stride],
            buf[s - 2 * stride],
            buf[s - stride],
        );
        let (q0, q1, q2, q3) = (
            buf[s],
            buf[s + stride],
            buf[s + 2 * stride],
            buf[s + 3 * stride],
        );
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        filter8(
            mask,
            thresh,
            flat,
            buf,
            s - 4 * stride,
            s - 3 * stride,
            s - 2 * stride,
            s - stride,
            s,
            s + stride,
            s + 2 * stride,
            s + 3 * stride,
        );
    }
}

fn lpf_horizontal_16(buf: &mut [u8], stride: usize, pos: usize, blimit: u8, limit: u8, thresh: u8) {
    for i in 0..8 {
        let s = pos + i;
        let (p3, p2, p1, p0) = (
            buf[s - 4 * stride],
            buf[s - 3 * stride],
            buf[s - 2 * stride],
            buf[s - stride],
        );
        let (q0, q1, q2, q3) = (
            buf[s],
            buf[s + stride],
            buf[s + 2 * stride],
            buf[s + 3 * stride],
        );
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat = flat_mask4(1, p3, p2, p1, p0, q0, q1, q2, q3);
        let flat2 = flat_mask5(
            1,
            buf[s - 8 * stride],
            buf[s - 7 * stride],
            buf[s - 6 * stride],
            buf[s - 5 * stride],
            p0,
            q0,
            buf[s + 4 * stride],
            buf[s + 5 * stride],
            buf[s + 6 * stride],
            buf[s + 7 * stride],
        );
        filter16(
            mask,
            thresh,
            flat,
            flat2,
            buf,
            [
                s - 8 * stride,
                s - 7 * stride,
                s - 6 * stride,
                s - 5 * stride,
                s - 4 * stride,
                s - 3 * stride,
                s - 2 * stride,
                s - stride,
                s,
                s + stride,
                s + 2 * stride,
                s + 3 * stride,
                s + 4 * stride,
                s + 5 * stride,
                s + 6 * stride,
                s + 7 * stride,
            ],
        );
    }
}

// ---------------------------------------------------------------------------
// Per-row-band mask application (`filter_selectively_vert`/`_horiz`,
// non-dual). `num_cols` is 8 for luma, 4 for 4:2:0 chroma; each mask bit
// covers one 8-pixel-wide column within the band.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert(
    buf: &mut [u8],
    stride: usize,
    pos: usize,
    mask16: u32,
    mask8: u32,
    mask4: u32,
    mask4i: u32,
    thr: &[LoopFilterThresh; MAX_LOOP_FILTER + 1],
    lfl: &[u8],
    num_cols: usize,
) {
    let (mut m16, mut m8, mut m4, mut m4i) = (mask16, mask8, mask4, mask4i);
    for c in 0..num_cols {
        let p = pos + c * 8;
        let t = &thr[lfl[c] as usize];
        if m16 & 1 != 0 {
            lpf_vertical_16(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        } else if m8 & 1 != 0 {
            lpf_vertical_8(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        } else if m4 & 1 != 0 {
            lpf_vertical_4(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        }
        if m4i & 1 != 0 {
            lpf_vertical_4(buf, stride, p + 4, t.mblim, t.lim, t.hev_thr);
        }
        m16 >>= 1;
        m8 >>= 1;
        m4 >>= 1;
        m4i >>= 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_selectively_horiz(
    buf: &mut [u8],
    stride: usize,
    pos: usize,
    mask16: u32,
    mask8: u32,
    mask4: u32,
    mask4i: u32,
    thr: &[LoopFilterThresh; MAX_LOOP_FILTER + 1],
    lfl: &[u8],
    num_cols: usize,
) {
    let (mut m16, mut m8, mut m4, mut m4i) = (mask16, mask8, mask4, mask4i);
    for c in 0..num_cols {
        let p = pos + c * 8;
        let t = &thr[lfl[c] as usize];
        if m16 & 1 != 0 {
            lpf_horizontal_16(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        } else if m8 & 1 != 0 {
            lpf_horizontal_8(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        } else if m4 & 1 != 0 {
            lpf_horizontal_4(buf, stride, p, t.mblim, t.lim, t.hev_thr);
        }
        if m4i & 1 != 0 {
            lpf_horizontal_4(buf, stride, p + 4 * stride, t.mblim, t.lim, t.hev_thr);
        }
        m16 >>= 1;
        m8 >>= 1;
        m4 >>= 1;
        m4i >>= 1;
    }
}

// ---------------------------------------------------------------------------
// Plane filtering (`vp9_filter_block_plane_ss00`/`ss11`).
// ---------------------------------------------------------------------------

/// Luma (no subsampling): `vp9_filter_block_plane_ss00`.
fn filter_plane_ss00(
    buf: &mut [u8],
    stride: usize,
    base: usize,
    sb_row: usize,
    mi_rows: usize,
    lfm: &LoopFilterMask,
    thr: &[LoopFilterThresh; MAX_LOOP_FILTER + 1],
) {
    for r in 0..MI_BLOCK_SIZE {
        if sb_row + r >= mi_rows {
            break;
        }
        let m16 = ((lfm.left_y[2] >> (r * 8)) & 0xff) as u32;
        let m8 = ((lfm.left_y[1] >> (r * 8)) & 0xff) as u32;
        let m4 = ((lfm.left_y[0] >> (r * 8)) & 0xff) as u32;
        let m4i = ((lfm.int_4x4_y >> (r * 8)) & 0xff) as u32;
        let pos = base + r * 8 * stride;
        filter_selectively_vert(
            buf,
            stride,
            pos,
            m16,
            m8,
            m4,
            m4i,
            thr,
            &lfm.lfl_y[r * 8..r * 8 + 8],
            8,
        );
    }

    for r in 0..MI_BLOCK_SIZE {
        if sb_row + r >= mi_rows {
            break;
        }
        let (m16, m8, m4) = if sb_row + r == 0 {
            (0, 0, 0)
        } else {
            (
                ((lfm.above_y[2] >> (r * 8)) & 0xff) as u32,
                ((lfm.above_y[1] >> (r * 8)) & 0xff) as u32,
                ((lfm.above_y[0] >> (r * 8)) & 0xff) as u32,
            )
        };
        let m4i = ((lfm.int_4x4_y >> (r * 8)) & 0xff) as u32;
        let pos = base + r * 8 * stride;
        filter_selectively_horiz(
            buf,
            stride,
            pos,
            m16,
            m8,
            m4,
            m4i,
            thr,
            &lfm.lfl_y[r * 8..r * 8 + 8],
            8,
        );
    }
}

/// 4:2:0 chroma: `vp9_filter_block_plane_ss11`.
fn filter_plane_ss11(
    buf: &mut [u8],
    stride: usize,
    base: usize,
    sb_row: usize,
    mi_rows: usize,
    lfm: &LoopFilterMask,
    thr: &[LoopFilterThresh; MAX_LOOP_FILTER + 1],
) {
    for r_uv in 0..4 {
        let mi_r = r_uv * 2;
        if sb_row + mi_r >= mi_rows {
            break;
        }
        let m16 = ((lfm.left_uv[2] >> (r_uv * 4)) & 0xf) as u32;
        let m8 = ((lfm.left_uv[1] >> (r_uv * 4)) & 0xf) as u32;
        let m4 = ((lfm.left_uv[0] >> (r_uv * 4)) & 0xf) as u32;
        let m4i = ((lfm.int_4x4_uv >> (r_uv * 4)) & 0xf) as u32;
        let mut lfl_uv = [0u8; 4];
        for c in 0..4 {
            lfl_uv[c] = lfm.lfl_y[mi_r * 8 + c * 2];
        }
        let pos = base + r_uv * 8 * stride;
        filter_selectively_vert(buf, stride, pos, m16, m8, m4, m4i, thr, &lfl_uv, 4);
    }

    for r_uv in 0..4 {
        let mi_r = r_uv * 2;
        if sb_row + mi_r >= mi_rows {
            break;
        }
        let skip_border = sb_row + mi_r == mi_rows - 1;
        let m4i_full = ((lfm.int_4x4_uv >> (r_uv * 4)) & 0xf) as u32;
        let m4i = if skip_border { 0 } else { m4i_full };
        let (m16, m8, m4) = if sb_row + mi_r == 0 {
            (0, 0, 0)
        } else {
            (
                ((lfm.above_uv[2] >> (r_uv * 4)) & 0xf) as u32,
                ((lfm.above_uv[1] >> (r_uv * 4)) & 0xf) as u32,
                ((lfm.above_uv[0] >> (r_uv * 4)) & 0xf) as u32,
            )
        };
        let mut lfl_uv = [0u8; 4];
        for c in 0..4 {
            lfl_uv[c] = lfm.lfl_y[mi_r * 8 + c * 2];
        }
        let pos = base + r_uv * 8 * stride;
        filter_selectively_horiz(buf, stride, pos, m16, m8, m4, m4i, thr, &lfl_uv, 4);
    }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Apply in-loop deblocking to a YUV420 frame in place.
///
/// `mi` is row-major, length `mi_rows * mi_cols`, one entry per 8×8 MI.
///
/// Like libvpx, this filters whole 64×64 superblocks, so the very last
/// (possibly partial) superblock in each row/column is still filtered up to
/// its full 64×64/32×32 (luma/chroma) extent even if `mi_rows`/`mi_cols` end
/// mid-superblock — mirroring how VP9 always reconstructs full superblocks
/// and only crops to the visible frame for display. Callers must therefore
/// pad `y`/`u`/`v` (and their strides) up to the next multiple of 64 luma /
/// 32 chroma pixels in both dimensions, not just to `mi_rows*8`/`mi_cols*8`.
///
/// `filter_level` is the frame base level in `0..=63` (`0` is a no-op for the
/// whole frame). Per-MI strength comes from [`LfMi::level`] (already including
/// mode/ref deltas). `sharpness` in `0..=7`.
pub fn loop_filter_frame(
    y: &mut [u8],
    y_stride: usize,
    u: &mut [u8],
    v: &mut [u8],
    uv_stride: usize,
    _width: usize,
    _height: usize,
    mi: &[LfMi],
    mi_rows: usize,
    mi_cols: usize,
    filter_level: u8,
    sharpness: u8,
) {
    if filter_level == 0 || mi_rows == 0 || mi_cols == 0 {
        return;
    }
    debug_assert_eq!(mi.len(), mi_rows * mi_cols);

    let thresholds = build_thresholds(sharpness);

    let mut sb_row = 0;
    while sb_row < mi_rows {
        let mut sb_col = 0;
        while sb_col < mi_cols {
            let mut lfm = LoopFilterMask::default();

            for r in 0..MI_BLOCK_SIZE {
                let row = sb_row + r;
                if row >= mi_rows {
                    break;
                }
                for c in 0..MI_BLOCK_SIZE {
                    let col = sb_col + c;
                    if col >= mi_cols {
                        break;
                    }
                    let m = &mi[row * mi_cols + col];
                    if !m.block_origin {
                        continue;
                    }
                    build_mask(
                        m.level,
                        m.is_inter,
                        m.skip,
                        m.tx_size_y,
                        m.bsize_idx,
                        m.bw_mi as usize,
                        m.bh_mi as usize,
                        r,
                        c,
                        &mut lfm,
                    );
                }
            }

            adjust_mask(mi_rows, mi_cols, sb_row, sb_col, &mut lfm);

            let y_base = sb_row * 8 * y_stride + sb_col * 8;
            filter_plane_ss00(y, y_stride, y_base, sb_row, mi_rows, &lfm, &thresholds);

            let uv_base = sb_row * 4 * uv_stride + sb_col * 4;
            filter_plane_ss11(u, uv_stride, uv_base, sb_row, mi_rows, &lfm, &thresholds);
            filter_plane_ss11(v, uv_stride, uv_base, sb_row, mi_rows, &lfm, &thresholds);

            sb_col += MI_BLOCK_SIZE;
        }
        sb_row += MI_BLOCK_SIZE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Buffers are padded to the enclosing 64x64-superblock grid, per
    /// [`loop_filter_frame`]'s buffer contract; `w`/`h` here are that padded
    /// (superblock-aligned) size.
    fn make_step_frame(w: usize, h: usize) -> (Vec<u8>, usize) {
        let stride = w;
        let mut y = vec![0u8; stride * h];
        for row in 0..h {
            for col in 0..w {
                y[row * stride + col] = if col < w / 2 { 100 } else { 150 };
            }
        }
        (y, stride)
    }

    fn flat_mi(mi_rows: usize, mi_cols: usize, level: u8) -> Vec<LfMi> {
        (0..mi_rows * mi_cols)
            .map(|_| LfMi {
                skip: false,
                is_inter: false,
                tx_size_y: 0,
                bw_mi: 1,
                bh_mi: 1,
                bsize_idx: 3,       // BLOCK_8X8
                block_origin: true, // every 8×8 is its own block
                level,
            })
            .collect()
    }

    fn sb_align(n: usize) -> usize {
        n.div_ceil(64) * 64
    }

    #[test]
    fn filter_level_zero_is_noop() {
        let (w, h) = (64, 64);
        let (mut y, y_stride) = make_step_frame(w, h);
        let y_orig = y.clone();
        let mut u = vec![128u8; (w / 2) * (h / 2)];
        let mut v = vec![128u8; (w / 2) * (h / 2)];
        let uv_stride = w / 2;
        let mi_rows = h / 8;
        let mi_cols = w / 8;
        let mi = flat_mi(mi_rows, mi_cols, 10);

        loop_filter_frame(
            &mut y, y_stride, &mut u, &mut v, uv_stride, w, h, &mi, mi_rows, mi_cols, 0, 0,
        );

        assert_eq!(y, y_orig, "filter_level=0 must not modify the luma plane");
    }

    #[test]
    fn synthetic_edge_is_modified_when_level_positive() {
        let (w, h) = (64, 64);
        let (mut y, y_stride) = make_step_frame(w, h);
        let y_orig = y.clone();
        let mut u = vec![128u8; (w / 2) * (h / 2)];
        let mut v = vec![128u8; (w / 2) * (h / 2)];
        let uv_stride = w / 2;
        let mi_rows = h / 8;
        let mi_cols = w / 8;
        let mi = flat_mi(mi_rows, mi_cols, 63);

        loop_filter_frame(
            &mut y, y_stride, &mut u, &mut v, uv_stride, w, h, &mi, mi_rows, mi_cols, 63, 0,
        );

        assert_ne!(
            y, y_orig,
            "filter_level>0 should modify pixels near the synthetic edge"
        );

        // Columns 0 and w-1 sit on the frame border and are excluded by the
        // "no filtering on the outermost edge" rule; interior edge columns
        // around the step (col = w/2) should change.
        let mid_row = h / 2;
        let mut changed = false;
        for col in (w / 2 - 4)..(w / 2 + 4) {
            if y[mid_row * y_stride + col] != y_orig[mid_row * y_stride + col] {
                changed = true;
            }
        }
        assert!(
            changed,
            "expected pixels around the synthetic step edge to change"
        );
    }

    #[test]
    fn no_panic_on_partial_superblock_frame() {
        // Visible content is 40x24 (not a superblock multiple), exercising the
        // adjust_mask edge-clip paths, but per the buffer contract the actual
        // Y/U/V allocations are padded up to the enclosing 64x64/32x32 superblock.
        let (w, h) = (40usize, 24usize);
        let mi_rows = h.div_ceil(8);
        let mi_cols = w.div_ceil(8);
        let mi = flat_mi(mi_rows, mi_cols, 30);

        let (pad_w, pad_h) = (sb_align(w), sb_align(h));
        let (mut y, y_stride) = make_step_frame(pad_w, pad_h);
        let uv_stride = pad_w / 2;
        let mut u = vec![128u8; uv_stride * (pad_h / 2)];
        let mut v = vec![128u8; uv_stride * (pad_h / 2)];

        loop_filter_frame(
            &mut y, y_stride, &mut u, &mut v, uv_stride, w, h, &mi, mi_rows, mi_cols, 30, 2,
        );
    }

    #[test]
    fn lf_level_applies_default_ref_deltas() {
        // base 10, scale 1: LAST→10, GOLDEN/ALTREF→9, INTRA→11.
        assert_eq!(lf_level_for_mi(10, 1, 12), 10); // LAST + ZEROMV
        assert_eq!(lf_level_for_mi(10, 2, 10), 9); // GOLDEN + NEARESTMV
        assert_eq!(lf_level_for_mi(10, 3, 13), 9); // ALTREF + NEWMV
        assert_eq!(lf_level_for_mi(10, 0, 0), 11); // INTRA
        assert_eq!(lf_level_for_mi(0, 1, 12), 0);
    }
}
