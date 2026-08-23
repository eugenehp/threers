//! VP9 inter (P-frame) encoder.
//!
//! - [`encode_inter_zeromv_skip`]: every 8×8 is LAST / ZEROMV / skip (copy ref).
//! - [`encode_inter_newmv_skip`]: every 8×8 is LAST / NEWMV / skip with a constant
//!   MV in 1/8-pel luma units (even values; `allow_hp=0`).
//! - [`encode_inter_residual`]: ZEROMV + 4×4 DCT residual against LAST.
//! - [`encode_inter_newmv_residual`]: NEWMV + residual (EIGHTTAP MC + inter coeffs).
//! - [`encode_inter_frame`]: per-block SAD search with NEARESTMV/NEARMV/ZEROMV/NEWMV
//!   and 16/32/64 `PARTITION_NONE` when a single MV wins.
//! - [`encode_inter_frame_golden`]: same, also choosing LAST vs GOLDEN per block.
//! - [`encode_inter_frame_altref`]: LAST / GOLDEN / ALTREF per block, plus
//!   LAST/GOLDEN+ALTREF compound (avg of two refs) when it wins SAD.
//! - [`encode_inter_frame_refresh`]: LAST-only ME with a custom `refresh_frame_flags`.

use crate::codec::bitstream::BitWriter;
use crate::codec::hevc::Yuv420Frame;
use crate::codec::vp9::bool_coder::BoolEncoder;
use crate::codec::vp9::encoder::Reconstruction;
use crate::codec::vp9::loopfilter::{lf_level_for_mi, loop_filter_frame, LfMi};
use crate::codec::vp9::mc::{predict_inter, InterpFilter};
use crate::codec::vp9::me::{sad_mv, search_mv, DEFAULT_RANGE_PEL};
use crate::codec::vp9::mv::{write_mv, Mv};
use crate::codec::vp9::quant::{
    ac_quant, dc_quant, quantize, quantize_16x16, quantize_32x32, quantize_8x8,
};
use crate::codec::vp9::tables::{
    bsl_from_px, counter_to_inter_mode_ctx, max_tx_size_wh, mode_2_counter, partition_ctx_index,
    partition_ctx_index_wh, tx_size_px, uv_tx_size_wh, ALTREF_FRAME, DEFAULT_COMP_INTER_PROBS,
    DEFAULT_COMP_REF_PROBS, DEFAULT_INTER_MODE_PROBS, DEFAULT_INTRA_INTER_PROBS,
    DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROBS, DEFAULT_SKIP_PROBS,
    DEFAULT_SWITCHABLE_INTERP_PROBS, DEFAULT_TX_PROBS_16, DEFAULT_TX_PROBS_32, DEFAULT_TX_PROBS_8,
    DIFF_UPDATE_PROB, GOLDEN_FRAME, INTER_MODE_TREE, INTER_OFFSET_NEARESTMV, INTER_OFFSET_NEARMV,
    INTER_OFFSET_NEWMV, INTER_OFFSET_ZEROMV, INTRA_FRAME, LAST_FRAME, NEARESTMV, NEARMV, NEWMV,
    PARTITION_CONTEXT_LOOKUP, PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_TREE,
    PARTITION_VERT, SWITCHABLE_INTERP_TREE, TX_16X16, TX_4X4, TX_8X8, ZEROMV,
};

/// Frame-level `REFERENCE_MODE` (libvpx). Compound allowed once ALTREF sign-bias differs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefMode {
    /// Only single-ref blocks; one bit `0` in the compressed header.
    Single,
    /// Only compound blocks; bits `1,0`.
    Compound,
    /// Per-block select; bits `1,1`.
    Select,
}

/// With LAST/GOLDEN bias 0 and ALTREF bias 1 → fixed=ALTREF, var=[LAST, GOLDEN].
const COMP_FIXED_REF: i8 = ALTREF_FRAME;
const COMP_VAR_REF: [i8; 2] = [LAST_FRAME, GOLDEN_FRAME];

#[inline]
fn ref_sign_bias(rf: i8) -> bool {
    rf == ALTREF_FRAME
}

#[inline]
fn scale_mv_for_ref(mv: Mv, from_ref: i8, to_ref: i8) -> Mv {
    if ref_sign_bias(from_ref) != ref_sign_bias(to_ref) {
        Mv {
            row: -mv.row,
            col: -mv.col,
        }
    } else {
        mv
    }
}

#[inline]
fn avg_bytes(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..out.len() {
        out[i] = ((u16::from(a[i]) + u16::from(b[i]) + 1) >> 1) as u8;
    }
}
use crate::codec::vp9::tokens::{
    write_coefs, write_coefs_16x16, write_coefs_32x32, write_coefs_8x8,
};
use crate::codec::vp9::transform::{
    fdct16x16, fdct32x32, fdct8x8, fht4x4, idct16x16_add, idct32x32_add, idct8x8_add, iht4x4_add,
    TxType,
};

const MAX_TILE_WIDTH_SB: u32 = 64;
const MIN_TILE_WIDTH_SB: u32 = 4;
const QINDEX: u32 = 128;
const QINDEX_USIZE: usize = 128;
/// Segment 1 uses `SEG_LVL_ALT_Q` with this delta (abs_delta=0).
const SEG1_ALT_Q_DELTA: i32 = 32;
const MAX_SEGMENTS: usize = 8;
const SEG_TREE_PROBS: usize = 7;
const SEG_LVL_ALT_Q: usize = 0;
const SEG_LVL_MAX: usize = 4;
const MAX_PROB: u8 = 255;
const MAXQ: i32 = 255;
/// `vp9_segment_tree` — walked with `vp9_write_tree(..., segment_id, 3, 0)`.
const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
/// Frame loop-filter level (sharpness 0, no mode/ref deltas).
const FILTER_LEVEL: u8 = 10;
const FILTER_SHARPNESS: u8 = 0;

#[derive(Clone, Copy)]
enum EntPlane {
    Y,
    U,
    V,
}

/// Forward transform + quantize for a residual tile (`$tx_px` ∈ {4,8,16,32}).
macro_rules! fwd_quant_tx {
    (4, $res:expr, $q:expr) => {{
        let mut coeff = [0i32; 16];
        fht4x4($res, &mut coeff, TxType::DctDct);
        quantize(&coeff, $q)
    }};
    (8, $res:expr, $q:expr) => {{
        let mut coeff = [0i32; 64];
        fdct8x8($res, &mut coeff);
        quantize_8x8(&coeff, $q)
    }};
    (16, $res:expr, $q:expr) => {{
        let mut coeff = [0i32; 256];
        fdct16x16($res, &mut coeff);
        quantize_16x16(&coeff, $q)
    }};
    (32, $res:expr, $q:expr) => {{
        let mut coeff = [0i32; 1024];
        fdct32x32($res, &mut coeff);
        quantize_32x32(&coeff, $q)
    }};
}

macro_rules! idct_add_tx {
    (4, $dq:expr, $block:expr) => {
        iht4x4_add($dq, $block, 4, TxType::DctDct)
    };
    (8, $dq:expr, $block:expr) => {
        idct8x8_add($dq, $block, 8)
    };
    (16, $dq:expr, $block:expr) => {
        idct16x16_add($dq, $block, 16)
    };
    (32, $dq:expr, $block:expr) => {
        idct32x32_add($dq, $block, 32)
    };
}

macro_rules! write_coefs_tx {
    (4, $e:expr, $q:expr, $is_y:expr, $ctx:expr, $dc:expr, $ac:expr) => {
        write_coefs($e, $q, $is_y, true, $ctx, TxType::DctDct, $dc, $ac)
    };
    (8, $e:expr, $q:expr, $is_y:expr, $ctx:expr, $dc:expr, $ac:expr) => {
        write_coefs_8x8($e, $q, $is_y, $ctx)
    };
    (16, $e:expr, $q:expr, $is_y:expr, $ctx:expr, $dc:expr, $ac:expr) => {
        write_coefs_16x16($e, $q, $is_y, $ctx)
    };
    (32, $e:expr, $q:expr, $is_y:expr, $ctx:expr, $dc:expr, $ac:expr) => {
        write_coefs_32x32($e, $q, $is_y, $ctx)
    };
}

/// Score all TX tiles of one plane for a fixed `$tx_px`.
macro_rules! score_plane_tx_px {
    ($tx_px:tt, $ncoef:tt, $pred:expr, $bw:expr, $bh:expr, $src:expr, $stride:expr, $ox:expr, $oy:expr, $qindex:expr) => {{
        let n_w = $bw / $tx_px;
        let n_h = $bh / $tx_px;
        let mut cost = 0u32;
        let mut any = false;
        for ty in 0..n_h {
            for txc in 0..n_w {
                let mut residual = [0i16; $ncoef];
                for r in 0..$tx_px {
                    for c in 0..$tx_px {
                        let pi = (ty * $tx_px + r) * $bw + txc * $tx_px + c;
                        let s = $src[($oy + ty * $tx_px + r) * $stride + ($ox + txc * $tx_px + c)]
                            as i16;
                        residual[r * $tx_px + c] = s - $pred[pi] as i16;
                    }
                }
                let (q, _) = fwd_quant_tx!($tx_px, &residual, $qindex);
                for &c in &q {
                    cost = cost.saturating_add(c.unsigned_abs());
                }
                any |= q.iter().any(|&c| c != 0);
            }
        }
        (cost, any)
    }};
}

fn score_plane_tx(
    pred: &[u8],
    bw: usize,
    bh: usize,
    src: &[u8],
    stride: usize,
    ox: usize,
    oy: usize,
    tx: u8,
    qindex: usize,
) -> (u32, bool) {
    match tx {
        TX_4X4 => score_plane_tx_px!(4, 16, pred, bw, bh, src, stride, ox, oy, qindex),
        TX_8X8 => score_plane_tx_px!(8, 64, pred, bw, bh, src, stride, ox, oy, qindex),
        TX_16X16 => score_plane_tx_px!(16, 256, pred, bw, bh, src, stride, ox, oy, qindex),
        _ => score_plane_tx_px!(32, 1024, pred, bw, bh, src, stride, ox, oy, qindex),
    }
}

fn store_plane(
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    src: &[u8],
) {
    for r in 0..bh {
        dst[(y + r) * stride + x..(y + r) * stride + x + bw]
            .copy_from_slice(&src[r * bw..r * bw + bw]);
    }
}

struct TxTile {
    /// Absolute plane origin (entropy contexts).
    x: usize,
    y: usize,
    /// Block-local origin into the recon buffer.
    lx: usize,
    ly: usize,
    q: Vec<i32>,
    dq: Vec<i32>,
    pred: Vec<u8>,
}

macro_rules! prepare_plane_tx_px {
    ($tx_px:tt, $ncoef:tt, $pred:expr, $bw:expr, $bh:expr, $src:expr, $stride:expr, $ox:expr, $oy:expr, $qindex:expr) => {{
        let n_w = $bw / $tx_px;
        let n_h = $bh / $tx_px;
        let mut tiles = Vec::with_capacity(n_w * n_h);
        for ty in 0..n_h {
            for txc in 0..n_w {
                let x = $ox + txc * $tx_px;
                let y = $oy + ty * $tx_px;
                let mut residual = [0i16; $ncoef];
                let mut pred_tile = [0u8; $ncoef];
                for r in 0..$tx_px {
                    for c in 0..$tx_px {
                        let pi = (ty * $tx_px + r) * $bw + txc * $tx_px + c;
                        pred_tile[r * $tx_px + c] = $pred[pi];
                        residual[r * $tx_px + c] = $src[(y + r) * $stride + (x + c)] as i16
                            - pred_tile[r * $tx_px + c] as i16;
                    }
                }
                let (q, dq) = fwd_quant_tx!($tx_px, &residual, $qindex);
                tiles.push(TxTile {
                    x,
                    y,
                    lx: txc * $tx_px,
                    ly: ty * $tx_px,
                    q: q.to_vec(),
                    dq: dq.to_vec(),
                    pred: pred_tile.to_vec(),
                });
            }
        }
        tiles
    }};
}

fn prepare_plane_tx(
    pred: &[u8],
    bw: usize,
    bh: usize,
    src: &[u8],
    stride: usize,
    ox: usize,
    oy: usize,
    tx: u8,
    qindex: usize,
) -> Vec<TxTile> {
    match tx {
        TX_4X4 => prepare_plane_tx_px!(4, 16, pred, bw, bh, src, stride, ox, oy, qindex),
        TX_8X8 => prepare_plane_tx_px!(8, 64, pred, bw, bh, src, stride, ox, oy, qindex),
        TX_16X16 => prepare_plane_tx_px!(16, 256, pred, bw, bh, src, stride, ox, oy, qindex),
        _ => prepare_plane_tx_px!(32, 1024, pred, bw, bh, src, stride, ox, oy, qindex),
    }
}

/// Encode a showable P-frame that copies `reference` (ZEROMV + skip on LAST).
pub fn encode_inter_zeromv_skip(reference: &Reconstruction) -> (Vec<u8>, Reconstruction) {
    encode_inter_skip(reference, Mv { row: 0, col: 0 }, false)
}

/// Encode a ZEROMV P-frame with 4×4 DCT residual against LAST.
///
/// Prediction is a copy of `reference` at the same position; the residual of
/// `src − pred` is transform-coded with inter coefficient probabilities.
/// `src` and `reference` must share dimensions (multiples of 8).
pub fn encode_inter_residual(
    src: &Yuv420Frame,
    reference: &Reconstruction,
) -> (Vec<u8>, Reconstruction) {
    encode_inter_residual_mv(
        src,
        reference,
        None,
        None,
        Mv::default(),
        false,
        false,
        0x01,
        RefMode::Single,
    )
}

/// Encode a NEWMV P-frame with 4×4 DCT residual against LAST.
///
/// `mv_row_q3` / `mv_col_q3` are luma **1/8-pel** units and must be even
/// (`allow_hp=0`). Prediction uses EIGHTTAP MC; residual uses inter coef probs.
pub fn encode_inter_newmv_residual(
    src: &Yuv420Frame,
    reference: &Reconstruction,
    mv_row_q3: i16,
    mv_col_q3: i16,
) -> (Vec<u8>, Reconstruction) {
    assert!(
        mv_row_q3 % 2 == 0 && mv_col_q3 % 2 == 0,
        "NEWMV residual needs even Q3 MVs with allow_hp=0 (got {}, {})",
        mv_row_q3,
        mv_col_q3
    );
    assert!(
        mv_row_q3 != 0 || mv_col_q3 != 0,
        "use encode_inter_residual for a zero MV"
    );
    encode_inter_residual_mv(
        src,
        reference,
        None,
        None,
        Mv {
            row: mv_row_q3,
            col: mv_col_q3,
        },
        true,
        false,
        0x01,
        RefMode::Single,
    )
}

/// Encode a P-frame with **per-block** SAD motion search + 4×4 DCT residual.
///
/// Each block picks an even-Q3 MV via SAD search, then codes NEARESTMV / NEARMV /
/// ZEROMV / NEWMV plus residual against EIGHTTAP prediction from LAST.
pub fn encode_inter_frame(src: &Yuv420Frame, last: &Reconstruction) -> (Vec<u8>, Reconstruction) {
    encode_inter_residual_mv(
        src,
        last,
        None,
        None,
        Mv::default(),
        false,
        true,
        0x01,
        RefMode::Single,
    )
}

/// Like [`encode_inter_frame`], but each block may also predict from `golden`.
///
/// Frame buffers: LAST→slot 0, GOLDEN→slot 1, ALTREF→slot 2 (keyframe fills all
/// slots; this path refreshes only slot 0). Pass the previous showable recon as
/// `last` and an older still-valid recon (often the keyframe) as `golden`.
pub fn encode_inter_frame_golden(
    src: &Yuv420Frame,
    last: &Reconstruction,
    golden: &Reconstruction,
) -> (Vec<u8>, Reconstruction) {
    assert_eq!(last.width, golden.width);
    assert_eq!(last.height, golden.height);
    encode_inter_residual_mv(
        src,
        last,
        Some(golden),
        None,
        Mv::default(),
        false,
        true,
        0x01,
        RefMode::Single,
    )
}

/// Like [`encode_inter_frame_golden`], also allowing ALTREF per block.
///
/// Pass distinct recons for slots that still hold those pictures. Use
/// [`encode_inter_frame_refresh`] with `0x04` earlier in the GOP if ALTREF
/// should differ from GOLDEN (keyframe alone leaves both equal to the key).
pub fn encode_inter_frame_altref(
    src: &Yuv420Frame,
    last: &Reconstruction,
    golden: &Reconstruction,
    altref: &Reconstruction,
) -> (Vec<u8>, Reconstruction) {
    assert_eq!(last.width, golden.width);
    assert_eq!(last.height, golden.height);
    assert_eq!(last.width, altref.width);
    assert_eq!(last.height, altref.height);
    encode_inter_residual_mv(
        src,
        last,
        Some(golden),
        Some(altref),
        Mv::default(),
        false,
        true,
        0x01,
        RefMode::Select,
    )
}

/// Like [`encode_inter_frame_altref`], but every block is LAST+ALTREF compound
/// (`REFERENCE_MODE` = compound; prediction is the average of the two refs).
pub fn encode_inter_frame_compound(
    src: &Yuv420Frame,
    last: &Reconstruction,
    golden: &Reconstruction,
    altref: &Reconstruction,
) -> (Vec<u8>, Reconstruction) {
    assert_eq!(last.width, golden.width);
    assert_eq!(last.height, golden.height);
    assert_eq!(last.width, altref.width);
    assert_eq!(last.height, altref.height);
    encode_inter_residual_mv(
        src,
        last,
        Some(golden),
        Some(altref),
        Mv::default(),
        false,
        true,
        0x01,
        RefMode::Compound,
    )
}

/// LAST-only ME P-frame with a custom `refresh_frame_flags` bitmask.
///
/// Bit `i` writes this recon into reference slot `i`. Common values: `0x01`
/// (LAST/slot 0, default), `0x04` (ALTREF/slot 2 only — leaves LAST/GOLDEN).
pub fn encode_inter_frame_refresh(
    src: &Yuv420Frame,
    last: &Reconstruction,
    refresh_frame_flags: u8,
) -> (Vec<u8>, Reconstruction) {
    encode_inter_residual_mv(
        src,
        last,
        None,
        None,
        Mv::default(),
        false,
        true,
        refresh_frame_flags,
        RefMode::Single,
    )
}

fn encode_inter_residual_mv(
    src: &Yuv420Frame,
    last: &Reconstruction,
    golden: Option<&Reconstruction>,
    altref: Option<&Reconstruction>,
    mv: Mv,
    newmv: bool,
    search: bool,
    refresh_frame_flags: u8,
    ref_mode: RefMode,
) -> (Vec<u8>, Reconstruction) {
    assert_eq!(src.width, last.width);
    assert_eq!(src.height, last.height);
    assert!(
        src.width >= 8 && src.height >= 8 && src.width.is_multiple_of(8) && src.height.is_multiple_of(8),
        "VP9 inter residual needs dims that are multiples of 8 (got {}×{})",
        src.width,
        src.height
    );

    let mut enc = ResidualInterEnc::new(src, last, golden, altref, mv, newmv, search, ref_mode);
    let compressed = compressed_header_inter(ref_mode);
    let log2_tile_cols = choose_log2_tile_cols(src.width);
    let tile = enc.encode_tiles(log2_tile_cols);

    let mut wb = BitWriter::new();
    write_uncompressed_inter_header(
        &mut wb,
        src.width,
        src.height,
        compressed.len() as u32,
        refresh_frame_flags,
        log2_tile_cols,
    );
    let mut out = wb.finish();
    out.extend_from_slice(&compressed);
    out.extend_from_slice(&tile);

    let mut recon = Reconstruction {
        width: src.width,
        height: src.height,
        y: enc.recon_y,
        u: enc.recon_u,
        v: enc.recon_v,
    };
    let lf_mi = lf_mi_from_residual(&enc.mi);
    apply_loop_filter(&mut recon, &lf_mi, enc.mi_rows, enc.mi_cols);
    (crate::codec::vp9::finalize_frame(out), recon)
}

/// Encode a showable P-frame: NEWMV + skip with a constant MV in **1/8-pel luma**.
///
/// Values must be even (`allow_hp=0`). Reconstruction is EIGHTTAP motion-
/// compensation of `reference` (full- or sub-pel) with edge clamping.
pub fn encode_inter_newmv_skip(
    reference: &Reconstruction,
    mv_row_q3: i16,
    mv_col_q3: i16,
) -> (Vec<u8>, Reconstruction) {
    assert!(
        mv_row_q3 % 2 == 0 && mv_col_q3 % 2 == 0,
        "NEWMV skip needs even Q3 MVs with allow_hp=0 (got {}, {})",
        mv_row_q3,
        mv_col_q3
    );
    assert!(
        mv_row_q3 != 0 || mv_col_q3 != 0,
        "use encode_inter_zeromv_skip for a zero MV"
    );
    encode_inter_skip(
        reference,
        Mv {
            row: mv_row_q3,
            col: mv_col_q3,
        },
        true,
    )
}

fn encode_inter_skip(reference: &Reconstruction, mv: Mv, newmv: bool) -> (Vec<u8>, Reconstruction) {
    assert!(
        reference.width >= 8
            && reference.height >= 8
            && reference.width.is_multiple_of(8)
            && reference.height.is_multiple_of(8),
        "VP9 inter needs dims that are multiples of 8 (got {}×{})",
        reference.width,
        reference.height
    );

    let mut enc = InterEnc::new(reference, mv, newmv);
    let compressed = compressed_header_inter(RefMode::Single);
    let log2_tile_cols = choose_log2_tile_cols(reference.width);
    let tile = enc.encode_tiles(log2_tile_cols);

    let mut wb = BitWriter::new();
    write_uncompressed_inter_header(
        &mut wb,
        reference.width,
        reference.height,
        compressed.len() as u32,
        0x01,
        log2_tile_cols,
    );
    let mut out = wb.finish();
    out.extend_from_slice(&compressed);
    out.extend_from_slice(&tile);

    let mut recon = motion_compensate(reference, mv);
    let lf_mi: Vec<LfMi> = enc
        .mi
        .iter()
        .map(|m| LfMi {
            skip: m.skip,
            is_inter: m.is_inter,
            tx_size_y: 1, // BLOCK_8X8 → TX_8X8
            bw_mi: 1,
            bh_mi: 1,
            bsize_idx: partition_ctx_index_wh(8, 8),
            block_origin: true,
            level: lf_level_for_mi(FILTER_LEVEL, m.ref_frame, m.mode),
        })
        .collect();
    apply_loop_filter(&mut recon, &lf_mi, enc.mi_rows, enc.mi_cols);
    (crate::codec::vp9::finalize_frame(out), recon)
}

fn motion_compensate(reference: &Reconstruction, mv: Mv) -> Reconstruction {
    let (w, h) = (reference.width as usize, reference.height as usize);
    let mut y = vec![0u8; w * h];
    predict_inter(
        &reference.y,
        w,
        w,
        h,
        0,
        0,
        w,
        h,
        mv.row,
        mv.col,
        false,
        InterpFilter::EightTap,
        &mut y,
    );
    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    predict_inter(
        &reference.u,
        cw,
        cw,
        ch,
        0,
        0,
        cw,
        ch,
        mv.row,
        mv.col,
        true,
        InterpFilter::EightTap,
        &mut u,
    );
    predict_inter(
        &reference.v,
        cw,
        cw,
        ch,
        0,
        0,
        cw,
        ch,
        mv.row,
        mv.col,
        true,
        InterpFilter::EightTap,
        &mut v,
    );
    Reconstruction {
        width: reference.width,
        height: reference.height,
        y,
        u,
        v,
    }
}

struct MiInfo {
    mode: i8,
    skip: bool,
    is_inter: bool,
    ref_frame: i8,
    mv: Mv,
    interp_filter: InterpFilter,
}

struct InterEnc<'a> {
    reference: &'a Reconstruction,
    mv: Mv,
    newmv: bool,
    mi_cols: usize,
    mi_rows: usize,
    tile_col_start: usize,
    tile_col_end: usize,
    mi: Vec<MiInfo>,
    above_ctx: Vec<u8>,
    left_ctx: [u8; 8],
    bool: BoolEncoder,
}

impl<'a> InterEnc<'a> {
    fn new(reference: &'a Reconstruction, mv: Mv, newmv: bool) -> Self {
        let mi_cols = reference.width as usize / 8;
        let mi_rows = reference.height as usize / 8;
        let mode = if newmv { NEWMV } else { ZEROMV };
        Self {
            reference,
            mv,
            newmv,
            mi_cols,
            mi_rows,
            tile_col_start: 0,
            tile_col_end: mi_cols,
            mi: (0..mi_cols * mi_rows)
                .map(|_| MiInfo {
                    mode,
                    skip: true,
                    is_inter: true,
                    ref_frame: LAST_FRAME,
                    mv,
                    interp_filter: InterpFilter::EightTap,
                })
                .collect(),
            above_ctx: vec![0; mi_cols],
            left_ctx: [0; 8],
            bool: BoolEncoder::new(),
        }
    }

    fn encode_tiles(&mut self, log2_tile_cols: u32) -> Vec<u8> {
        let tile_cols = 1usize << log2_tile_cols;
        let mut tiles = Vec::with_capacity(tile_cols);
        for tile_col in 0..tile_cols {
            let (start, end) = tile_mi_col_range(self.mi_cols, log2_tile_cols, tile_col as u32);
            self.tile_col_start = start;
            self.tile_col_end = end;
            tiles.push(self.encode_one_tile());
        }
        pack_tile_data(&tiles)
    }

    fn encode_one_tile(&mut self) -> Vec<u8> {
        let mut mi_row = 0;
        while mi_row < self.mi_rows {
            self.left_ctx = [0; 8];
            let mut mi_col = self.tile_col_start;
            while mi_col < self.tile_col_end {
                self.encode_partition(mi_row, mi_col, 64);
                mi_col += 8;
            }
            mi_row += 8;
        }
        std::mem::take(&mut self.bool).finish()
    }

    fn encode_partition(&mut self, mi_row: usize, mi_col: usize, bsize_px: u32) {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return;
        }
        let mi_half = (bsize_px as usize / 8) / 2;
        let has_rows = mi_row + mi_half < self.mi_rows;
        let has_cols = mi_col + mi_half < self.mi_cols;
        let partition = if bsize_px == 8 {
            PARTITION_NONE
        } else {
            PARTITION_SPLIT
        };

        self.write_partition(mi_row, mi_col, bsize_px, has_rows, has_cols, partition);

        if bsize_px == 8 {
            self.encode_block(mi_row, mi_col);
            self.update_partition_context(mi_row, mi_col, bsize_px, bsize_px);
            return;
        }

        let sub_px = bsize_px / 2;
        self.encode_partition(mi_row, mi_col, sub_px);
        self.encode_partition(mi_row, mi_col + mi_half, sub_px);
        self.encode_partition(mi_row + mi_half, mi_col, sub_px);
        self.encode_partition(mi_row + mi_half, mi_col + mi_half, sub_px);
    }

    fn write_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize_px: u32,
        has_rows: bool,
        has_cols: bool,
        partition: i8,
    ) {
        let ctx = self.partition_context(mi_row, mi_col, bsize_px);
        let probs = &DEFAULT_PARTITION_PROBS[ctx];
        if has_rows && has_cols {
            write_tree(&mut self.bool, &PARTITION_TREE, probs, partition);
        } else if !has_rows && has_cols {
            self.bool.put_bool(probs[1], partition == PARTITION_SPLIT);
        } else if has_rows && !has_cols {
            self.bool.put_bool(probs[2], partition == PARTITION_SPLIT);
        }
    }

    fn partition_context(&self, mi_row: usize, mi_col: usize, bsize_px: u32) -> usize {
        let bsl = bsl_from_px(bsize_px);
        let above = (self.above_ctx[mi_col] >> bsl) & 1;
        let left = (self.left_ctx[mi_row & 7] >> bsl) & 1;
        bsl * 4 + (left as usize) * 2 + above as usize
    }

    fn update_partition_context(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        sub_px: u32,
        bsize_px: u32,
    ) {
        let bs = bsize_px as usize / 8;
        let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[partition_ctx_index(sub_px)];
        for i in 0..bs {
            if mi_col + i < self.mi_cols {
                self.above_ctx[mi_col + i] = above_v;
            }
            self.left_ctx[(mi_row + i) & 7] = left_v;
        }
    }

    fn encode_block(&mut self, mi_row: usize, mi_col: usize) {
        let skip = true;
        let is_inter = true;
        write_segment_id(&mut self.bool, segment_id_for_mi_col(mi_col, self.mi_cols));
        let skip_ctx = self.skip_context(mi_row, mi_col);
        self.bool.put_bool(DEFAULT_SKIP_PROBS[skip_ctx], skip);

        let ii_ctx = self.intra_inter_context(mi_row, mi_col);
        self.bool
            .put_bool(DEFAULT_INTRA_INTER_PROBS[ii_ctx], is_inter);

        let ref_ctx = self.single_ref_p1_context(mi_row, mi_col);
        self.bool
            .put_bool(DEFAULT_SINGLE_REF_PROBS[ref_ctx][0], false);

        let mode_ctx = self.inter_mode_context(mi_row, mi_col);
        if self.newmv {
            write_tree(
                &mut self.bool,
                &INTER_MODE_TREE,
                &DEFAULT_INTER_MODE_PROBS[mode_ctx],
                INTER_OFFSET_NEWMV,
            );
        } else {
            write_tree(
                &mut self.bool,
                &INTER_MODE_TREE,
                &DEFAULT_INTER_MODE_PROBS[mode_ctx],
                INTER_OFFSET_ZEROMV,
            );
        }

        let filt_ctx = self.switchable_interp_context(mi_row, mi_col);
        write_tree(
            &mut self.bool,
            &SWITCHABLE_INTERP_TREE,
            &DEFAULT_SWITCHABLE_INTERP_PROBS[filt_ctx],
            InterpFilter::EightTap as i8,
        );

        if self.newmv {
            let ref_mv = self.nearest_ref_mv(mi_row, mi_col);
            write_mv(&mut self.bool, self.mv, ref_mv, false);
        }

        self.mi[mi_row * self.mi_cols + mi_col] = MiInfo {
            mode: if self.newmv { NEWMV } else { ZEROMV },
            skip,
            is_inter,
            ref_frame: LAST_FRAME,
            mv: self.mv,
            interp_filter: InterpFilter::EightTap,
        };
        let _ = self.reference;
    }

    /// NEWMV nearest ref: first unique LAST MV from neighbors (above, then left)
    /// — matches `mv_ref_blocks[BLOCK_8X8]` / `find_mv_refs_idx`.
    fn nearest_ref_mv(&self, mi_row: usize, mi_col: usize) -> Mv {
        const NBRS: [(i32, i32); 8] = [
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-2, -1),
            (-1, -2),
            (-2, -2),
        ];
        for &(dr, dc) in &NBRS {
            let r = mi_row as i32 + dr;
            let c = mi_col as i32 + dc;
            if !self.mi_in_tile(r, c) {
                continue;
            }
            let n = &self.mi[r as usize * self.mi_cols + c as usize];
            if n.is_inter && n.ref_frame == LAST_FRAME {
                return clamp_mv_ref(n.mv, mi_row, mi_col, self.mi_rows, self.mi_cols, 1, 1);
            }
        }
        Mv::default()
    }

    fn mi_in_tile(&self, r: i32, c: i32) -> bool {
        r >= 0
            && (r as usize) < self.mi_rows
            && c >= self.tile_col_start as i32
            && (c as usize) < self.tile_col_end
    }

    fn left_available(&self, mi_col: usize) -> bool {
        mi_col > self.tile_col_start
    }

    fn switchable_interp_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let missing = InterpFilter::SWITCHABLE_FILTERS as usize;
        let left = if self.left_available(mi_col) {
            self.mi[mi_row * self.mi_cols + mi_col - 1].interp_filter as usize
        } else {
            missing
        };
        let above = if mi_row > 0 {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].interp_filter as usize
        } else {
            missing
        };
        if left == above {
            left
        } else if left == missing {
            above
        } else if above == missing {
            left
        } else {
            missing
        }
    }

    fn skip_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].skip as usize
        } else {
            0
        };
        let left = if self.left_available(mi_col) {
            self.mi[mi_row * self.mi_cols + (mi_col - 1)].skip as usize
        } else {
            0
        };
        above + left
    }

    fn intra_inter_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if li && ai {
                    3
                } else if li || ai {
                    1
                } else {
                    0
                }
            }
            (Some(e), None) | (None, Some(e)) => 2 * (!e.is_inter as usize),
            (None, None) => 0,
        }
    }

    fn inter_mode_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let mut counter = 0i32;
        if self.left_available(mi_col) {
            counter += mode_2_counter(self.mi[mi_row * self.mi_cols + (mi_col - 1)].mode);
        }
        if mi_row > 0 {
            counter += mode_2_counter(self.mi[(mi_row - 1) * self.mi_cols + mi_col].mode);
        }
        counter_to_inter_mode_ctx(counter)
    }

    fn single_ref_p1_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if ai && li {
                    2
                } else if ai || li {
                    let edge = if ai { l } else { a };
                    4 * (edge.ref_frame == LAST_FRAME) as usize
                } else {
                    2 * (a.ref_frame == LAST_FRAME) as usize
                        + 2 * (l.ref_frame == LAST_FRAME) as usize
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter {
                    2
                } else {
                    4 * (e.ref_frame == LAST_FRAME) as usize
                }
            }
            (None, None) => 2,
        }
    }
}

// ---- ZEROMV + residual -------------------------------------------------------

#[derive(Clone, Copy)]
struct ResidualMi {
    skip: bool,
    is_inter: bool,
    ref_frame: i8,
    /// Second ref for compound; [`INTRA_FRAME`] means single-ref.
    second_ref: i8,
    mode: i8,
    mv: Mv,
    mv1: Mv,
    bw_px: u32,
    bh_px: u32,
    /// True only on the top-left MI of this coding block.
    block_origin: bool,
    /// Coded (or implied-on-skip) transform size for TX_MODE_SELECT / LF.
    tx_size: u8,
    interp_filter: InterpFilter,
}

impl ResidualMi {
    fn is_compound(self) -> bool {
        self.second_ref > INTRA_FRAME
    }
}

#[derive(Clone, Copy)]
struct InterPick {
    mv: Mv,
    mv1: Mv,
    mode: i8,
    ref_frame: i8,
    second_ref: i8,
}

struct ResidualInterEnc<'a> {
    src: &'a Yuv420Frame,
    reference: &'a Reconstruction,
    golden: Option<&'a Reconstruction>,
    altref: Option<&'a Reconstruction>,
    mv: Mv,
    newmv: bool,
    search: bool,
    ref_mode: RefMode,
    mi_cols: usize,
    mi_rows: usize,
    tile_col_start: usize,
    tile_col_end: usize,
    recon_y: Vec<u8>,
    recon_u: Vec<u8>,
    recon_v: Vec<u8>,
    mi: Vec<ResidualMi>,
    above_ctx: Vec<u8>,
    left_ctx: [u8; 8],
    above_ent_y: Vec<u8>,
    left_ent_y: [u8; 16],
    above_ent_u: Vec<u8>,
    left_ent_u: [u8; 8],
    above_ent_v: Vec<u8>,
    left_ent_v: [u8; 8],
    bool: BoolEncoder,
}

impl<'a> ResidualInterEnc<'a> {
    fn new(
        src: &'a Yuv420Frame,
        reference: &'a Reconstruction,
        golden: Option<&'a Reconstruction>,
        altref: Option<&'a Reconstruction>,
        mv: Mv,
        newmv: bool,
        search: bool,
        ref_mode: RefMode,
    ) -> Self {
        let (w, h) = (src.width as usize, src.height as usize);
        let mi_cols = w / 8;
        let mi_rows = h / 8;
        let (cw, ch) = (w / 2, h / 2);
        let mode = if newmv { NEWMV } else { ZEROMV };
        Self {
            src,
            reference,
            golden,
            altref,
            mv,
            newmv,
            search,
            ref_mode,
            mi_cols,
            mi_rows,
            tile_col_start: 0,
            tile_col_end: mi_cols,
            recon_y: vec![0; w * h],
            recon_u: vec![128; cw * ch],
            recon_v: vec![128; cw * ch],
            mi: (0..mi_cols * mi_rows)
                .map(|_| ResidualMi {
                    skip: true,
                    is_inter: true,
                    ref_frame: LAST_FRAME,
                    second_ref: INTRA_FRAME,
                    mode,
                    mv,
                    mv1: Mv::default(),
                    bw_px: 8,
                    bh_px: 8,
                    block_origin: true,
                    tx_size: TX_8X8,
                    interp_filter: InterpFilter::EightTap,
                })
                .collect(),
            above_ctx: vec![0; mi_cols],
            left_ctx: [0; 8],
            above_ent_y: vec![0; mi_cols * 2],
            left_ent_y: [0; 16],
            above_ent_u: vec![0; mi_cols],
            left_ent_u: [0; 8],
            above_ent_v: vec![0; mi_cols],
            left_ent_v: [0; 8],
            bool: BoolEncoder::new(),
        }
    }

    fn encode_tiles(&mut self, log2_tile_cols: u32) -> Vec<u8> {
        let tile_cols = 1usize << log2_tile_cols;
        let mut tiles = Vec::with_capacity(tile_cols);
        for tile_col in 0..tile_cols {
            let (start, end) = tile_mi_col_range(self.mi_cols, log2_tile_cols, tile_col as u32);
            self.tile_col_start = start;
            self.tile_col_end = end;
            tiles.push(self.encode_one_tile());
        }
        pack_tile_data(&tiles)
    }

    fn encode_one_tile(&mut self) -> Vec<u8> {
        let mut mi_row = 0;
        while mi_row < self.mi_rows {
            self.left_ctx = [0; 8];
            self.left_ent_y = [0; 16];
            self.left_ent_u = [0; 8];
            self.left_ent_v = [0; 8];
            let mut mi_col = self.tile_col_start;
            while mi_col < self.tile_col_end {
                self.encode_partition(mi_row, mi_col, 64);
                mi_col += 8;
            }
            mi_row += 8;
        }
        std::mem::take(&mut self.bool).finish()
    }

    fn mi_in_tile(&self, r: i32, c: i32) -> bool {
        r >= 0
            && (r as usize) < self.mi_rows
            && c >= self.tile_col_start as i32
            && (c as usize) < self.tile_col_end
    }

    fn left_available(&self, mi_col: usize) -> bool {
        mi_col > self.tile_col_start
    }

    fn encode_partition(&mut self, mi_row: usize, mi_col: usize, bsize_px: u32) {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return;
        }
        let mi_half = (bsize_px as usize / 8) / 2;
        let has_rows = mi_row + mi_half < self.mi_rows;
        let has_cols = mi_col + mi_half < self.mi_cols;
        let bs_mi = (bsize_px / 8) as usize;
        let block_fits = mi_row + bs_mi <= self.mi_rows && mi_col + bs_mi <= self.mi_cols;
        let partition = if bsize_px == 8 {
            PARTITION_NONE
        } else if matches!(bsize_px, 16 | 32 | 64) && block_fits && self.search {
            self.pick_partition(mi_row, mi_col, bsize_px)
        } else {
            PARTITION_SPLIT
        };
        self.write_partition(mi_row, mi_col, bsize_px, has_rows, has_cols, partition);
        match partition {
            PARTITION_NONE => {
                self.encode_block(mi_row, mi_col, bsize_px, bsize_px);
                self.update_partition_context(
                    mi_row, mi_col, bsize_px, bsize_px, bsize_px, bsize_px,
                );
            }
            PARTITION_HORZ => {
                let sub_h = bsize_px / 2;
                self.encode_block(mi_row, mi_col, bsize_px, sub_h);
                self.encode_block(mi_row + mi_half, mi_col, bsize_px, sub_h);
                self.update_partition_context(mi_row, mi_col, bsize_px, sub_h, bsize_px, bsize_px);
            }
            PARTITION_VERT => {
                let sub_w = bsize_px / 2;
                self.encode_block(mi_row, mi_col, sub_w, bsize_px);
                self.encode_block(mi_row, mi_col + mi_half, sub_w, bsize_px);
                self.update_partition_context(mi_row, mi_col, sub_w, bsize_px, bsize_px, bsize_px);
            }
            _ => {
                let sub_px = bsize_px / 2;
                self.encode_partition(mi_row, mi_col, sub_px);
                self.encode_partition(mi_row, mi_col + mi_half, sub_px);
                self.encode_partition(mi_row + mi_half, mi_col, sub_px);
                self.encode_partition(mi_row + mi_half, mi_col + mi_half, sub_px);
            }
        }
    }

    /// Pick NONE/HORZ/VERT/SPLIT by luma SAD, biased toward fewer coding blocks.
    fn pick_partition(&self, mi_row: usize, mi_col: usize, bsize_px: u32) -> i8 {
        let w = self.src.width as usize;
        let h = self.src.height as usize;
        let px = mi_col * 8;
        let py = mi_row * 8;
        let bw = bsize_px as usize;
        let hw = bw / 2;
        let score = |dx: usize, dy: usize, sw: usize, sh: usize| {
            let mv = search_mv(
                &self.src.y,
                w,
                &self.reference.y,
                w,
                w,
                h,
                px + dx,
                py + dy,
                sw,
                sh,
                DEFAULT_RANGE_PEL,
            );
            sad_mv(
                &self.src.y,
                w,
                &self.reference.y,
                w,
                w,
                h,
                px + dx,
                py + dy,
                sw,
                sh,
                mv,
            )
        };
        let sad_none = score(0, 0, bw, bw);
        let sad_horz = score(0, 0, bw, hw).saturating_add(score(0, hw, bw, hw));
        let sad_vert = score(0, 0, hw, bw).saturating_add(score(hw, 0, hw, bw));
        let sad_split = score(0, 0, hw, hw)
            .saturating_add(score(hw, 0, hw, hw))
            .saturating_add(score(0, hw, hw, hw))
            .saturating_add(score(hw, hw, hw, hw));
        // Approximate the mode/MV/tree cost saved by using fewer blocks.
        let bias = match bsize_px {
            16 => 256,
            32 => 1024,
            64 => 4096,
            _ => 256,
        };
        let choices = [
            (sad_none, PARTITION_NONE),
            (sad_horz.saturating_add(bias / 2), PARTITION_HORZ),
            (sad_vert.saturating_add(bias / 2), PARTITION_VERT),
            (sad_split.saturating_add(bias), PARTITION_SPLIT),
        ];
        choices
            .into_iter()
            .min_by_key(|&(cost, partition)| (cost, partition))
            .unwrap()
            .1
    }

    fn write_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize_px: u32,
        has_rows: bool,
        has_cols: bool,
        partition: i8,
    ) {
        let bsl = bsl_from_px(bsize_px);
        let above = (self.above_ctx[mi_col] >> bsl) & 1;
        let left = (self.left_ctx[mi_row & 7] >> bsl) & 1;
        let ctx = bsl * 4 + (left as usize) * 2 + above as usize;
        let probs = &DEFAULT_PARTITION_PROBS[ctx];
        if has_rows && has_cols {
            write_tree(&mut self.bool, &PARTITION_TREE, probs, partition);
        } else if !has_rows && has_cols {
            self.bool.put_bool(probs[1], partition == PARTITION_SPLIT);
        } else if has_rows && !has_cols {
            self.bool.put_bool(probs[2], partition == PARTITION_SPLIT);
        }
    }

    fn update_partition_context(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        sub_bw: u32,
        sub_bh: u32,
        bw: u32,
        bh: u32,
    ) {
        let bs_w = bw as usize / 8;
        let bs_h = bh as usize / 8;
        let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[partition_ctx_index_wh(sub_bw, sub_bh)];
        for i in 0..bs_w {
            if mi_col + i < self.mi_cols {
                self.above_ctx[mi_col + i] = above_v;
            }
        }
        for i in 0..bs_h {
            self.left_ctx[(mi_row + i) & 7] = left_v;
        }
    }

    fn encode_block(&mut self, mi_row: usize, mi_col: usize, bw_px: u32, bh_px: u32) {
        debug_assert!(matches!(
            (bw_px, bh_px),
            (8, 8)
                | (8, 16)
                | (16, 8)
                | (16, 16)
                | (16, 32)
                | (32, 16)
                | (32, 32)
                | (32, 64)
                | (64, 32)
                | (64, 64)
        ));
        let bw_mi = (bw_px / 8) as usize;
        let bh_mi = (bh_px / 8) as usize;
        let (px, py) = (mi_col * 8, mi_row * 8);
        let w = self.src.width as usize;
        let h = self.src.height as usize;
        let (cpx, cpy) = (px / 2, py / 2);
        let cw = w / 2;
        let ch = h / 2;
        let bw = bw_px as usize;
        let bh = bh_px as usize;
        let cw_blk = bw / 2;
        let ch_blk = bh / 2;

        let pick = if self.search {
            self.pick_ref_and_mode(mi_row, mi_col, bw_px, bh_px, px, py, bw, bh, w, h)
        } else if self.newmv {
            InterPick {
                mv: self.mv,
                mv1: Mv::default(),
                mode: NEWMV,
                ref_frame: LAST_FRAME,
                second_ref: INTRA_FRAME,
            }
        } else {
            InterPick {
                mv: Mv::default(),
                mv1: Mv::default(),
                mode: ZEROMV,
                ref_frame: LAST_FRAME,
                second_ref: INTRA_FRAME,
            }
        };
        let (mv, mv1, mode, ref_frame, second_ref) = (
            pick.mv,
            pick.mv1,
            pick.mode,
            pick.ref_frame,
            pick.second_ref,
        );
        let ref_recon = self.recon_for(ref_frame);
        let compound = second_ref > INTRA_FRAME;
        let interp_filter =
            self.pick_interp_filter(px, py, bw, bh, w, h, mv, mv1, ref_frame, second_ref);

        let mut y_pred = vec![0u8; bw * bh];
        let mut u_pred = vec![0u8; cw_blk * ch_blk];
        let mut v_pred = vec![0u8; cw_blk * ch_blk];
        if compound {
            let r1 = self.recon_for(second_ref);
            let mut ya = vec![0u8; bw * bh];
            let mut yb = vec![0u8; bw * bh];
            predict_inter(
                &ref_recon.y,
                w,
                w,
                h,
                px,
                py,
                bw,
                bh,
                mv.row,
                mv.col,
                false,
                interp_filter,
                &mut ya,
            );
            predict_inter(
                &r1.y,
                w,
                w,
                h,
                px,
                py,
                bw,
                bh,
                mv1.row,
                mv1.col,
                false,
                interp_filter,
                &mut yb,
            );
            avg_bytes(&ya, &yb, &mut y_pred);
            let mut ua = vec![0u8; cw_blk * ch_blk];
            let mut ub = vec![0u8; cw_blk * ch_blk];
            let mut va = vec![0u8; cw_blk * ch_blk];
            let mut vb = vec![0u8; cw_blk * ch_blk];
            predict_inter(
                &ref_recon.u,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv.row,
                mv.col,
                true,
                interp_filter,
                &mut ua,
            );
            predict_inter(
                &r1.u,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv1.row,
                mv1.col,
                true,
                interp_filter,
                &mut ub,
            );
            predict_inter(
                &ref_recon.v,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv.row,
                mv.col,
                true,
                interp_filter,
                &mut va,
            );
            predict_inter(
                &r1.v,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv1.row,
                mv1.col,
                true,
                interp_filter,
                &mut vb,
            );
            avg_bytes(&ua, &ub, &mut u_pred);
            avg_bytes(&va, &vb, &mut v_pred);
        } else {
            predict_inter(
                &ref_recon.y,
                w,
                w,
                h,
                px,
                py,
                bw,
                bh,
                mv.row,
                mv.col,
                false,
                interp_filter,
                &mut y_pred,
            );
            predict_inter(
                &ref_recon.u,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv.row,
                mv.col,
                true,
                interp_filter,
                &mut u_pred,
            );
            predict_inter(
                &ref_recon.v,
                cw,
                cw,
                ch,
                cpx,
                cpy,
                cw_blk,
                ch_blk,
                mv.row,
                mv.col,
                true,
                interp_filter,
                &mut v_pred,
            );
        }

        let segment_id = segment_id_for_mi_col(mi_col, self.mi_cols);
        let qindex = qindex_for_segment(segment_id);

        let max_tx = max_tx_size_wh(bw_px, bh_px);
        let cand = if max_tx == TX_4X4 {
            [max_tx, max_tx]
        } else {
            [max_tx, max_tx - 1]
        };
        let cand = if max_tx == TX_4X4 {
            &cand[..1]
        } else {
            &cand[..]
        };

        let mut best_tx = max_tx;
        let mut best_cost = u32::MAX;
        let mut best_any = true;
        for &tx in cand {
            let (cost, any) = self.score_block_tx(
                bw_px, bh_px, tx, &y_pred, &u_pred, &v_pred, w, cw, px, py, cpx, cpy, qindex,
            );
            // Prefer all-zero (skip); else lower coef cost; prefer smaller TX on ties.
            let better = (!any && best_any)
                || (any == best_any && (cost < best_cost || (cost == best_cost && tx < best_tx)));
            if better {
                best_tx = tx;
                best_cost = cost;
                best_any = any;
            }
        }

        let skip = !best_any;
        let tx_size = if skip { max_tx } else { best_tx };
        self.write_inter_mode_bits(
            mi_row,
            mi_col,
            bw_px,
            bh_px,
            skip,
            tx_size,
            mode,
            ref_frame,
            second_ref,
            mv,
            mv1,
            interp_filter,
            segment_id,
        );

        let mut y_recon = y_pred.clone();
        let mut u_recon = u_pred.clone();
        let mut v_recon = v_pred.clone();
        if !skip {
            let y_tiles =
                prepare_plane_tx(&y_pred, bw, bh, &self.src.y, w, px, py, tx_size, qindex);
            let uv_tx = uv_tx_size_wh(bw_px, bh_px, tx_size);
            let u_tiles = prepare_plane_tx(
                &u_pred,
                cw_blk,
                ch_blk,
                &self.src.u,
                cw,
                cpx,
                cpy,
                uv_tx,
                qindex,
            );
            let v_tiles = prepare_plane_tx(
                &v_pred,
                cw_blk,
                ch_blk,
                &self.src.v,
                cw,
                cpx,
                cpy,
                uv_tx,
                qindex,
            );
            self.emit_plane_tx(EntPlane::Y, &y_tiles, &mut y_recon, bw, tx_size, qindex);
            self.emit_plane_tx(EntPlane::U, &u_tiles, &mut u_recon, cw_blk, uv_tx, qindex);
            self.emit_plane_tx(EntPlane::V, &v_tiles, &mut v_recon, cw_blk, uv_tx, qindex);
        } else {
            self.clear_ent_block(mi_row, mi_col, bw_mi, bh_mi);
        }

        store_plane(&mut self.recon_y, w, px, py, bw, bh, &y_recon);
        store_plane(&mut self.recon_u, cw, cpx, cpy, cw_blk, ch_blk, &u_recon);
        store_plane(&mut self.recon_v, cw, cpx, cpy, cw_blk, ch_blk, &v_recon);

        self.store_mi(
            mi_row,
            mi_col,
            bw_mi,
            bh_mi,
            ResidualMi {
                skip,
                is_inter: true,
                ref_frame,
                second_ref,
                mode,
                mv,
                mv1,
                bw_px,
                bh_px,
                block_origin: true,
                tx_size,
                interp_filter,
            },
        );
    }

    /// Coef-cost probe for one TX size (no bitstream writes).
    fn score_block_tx(
        &self,
        bw_px: u32,
        bh_px: u32,
        y_tx: u8,
        y_pred: &[u8],
        u_pred: &[u8],
        v_pred: &[u8],
        w: usize,
        cw: usize,
        px: usize,
        py: usize,
        cpx: usize,
        cpy: usize,
        qindex: usize,
    ) -> (u32, bool) {
        let bw = bw_px as usize;
        let bh = bh_px as usize;
        let uv_tx = uv_tx_size_wh(bw_px, bh_px, y_tx);
        let (mut cost, mut any) =
            score_plane_tx(y_pred, bw, bh, &self.src.y, w, px, py, y_tx, qindex);
        let cw_blk = bw / 2;
        let ch_blk = bh / 2;
        let (cu, au) = score_plane_tx(
            u_pred,
            cw_blk,
            ch_blk,
            &self.src.u,
            cw,
            cpx,
            cpy,
            uv_tx,
            qindex,
        );
        let (cv, av) = score_plane_tx(
            v_pred,
            cw_blk,
            ch_blk,
            &self.src.v,
            cw,
            cpx,
            cpy,
            uv_tx,
            qindex,
        );
        cost = cost.saturating_add(cu).saturating_add(cv);
        any |= au | av;
        (cost, any)
    }

    fn emit_plane_tx(
        &mut self,
        ent: EntPlane,
        tiles: &[TxTile],
        recon: &mut [u8],
        bw: usize,
        tx: u8,
        qindex: usize,
    ) {
        let tx_px = tx_size_px(tx);
        let n4 = tx_px / 4;
        let is_y = matches!(ent, EntPlane::Y);
        let dc_q = dc_quant(qindex);
        let ac_q = ac_quant(qindex);
        for t in tiles {
            let ctx = self.ent_ctx(ent, t.x, t.y, n4);
            let eob = match tx {
                TX_4X4 => {
                    let mut q = [0i32; 16];
                    q.copy_from_slice(&t.q);
                    write_coefs_tx!(4, &mut self.bool, &q, is_y, ctx, dc_q, ac_q)
                }
                TX_8X8 => {
                    let mut q = [0i32; 64];
                    q.copy_from_slice(&t.q);
                    write_coefs_tx!(8, &mut self.bool, &q, is_y, ctx, dc_q, ac_q)
                }
                TX_16X16 => {
                    let mut q = [0i32; 256];
                    q.copy_from_slice(&t.q);
                    write_coefs_tx!(16, &mut self.bool, &q, is_y, ctx, dc_q, ac_q)
                }
                _ => {
                    let mut q = [0i32; 1024];
                    q.copy_from_slice(&t.q);
                    write_coefs_tx!(32, &mut self.bool, &q, is_y, ctx, dc_q, ac_q)
                }
            };
            self.set_ent(ent, t.x, t.y, n4, (eob > 0) as u8);
            match tx {
                TX_4X4 => {
                    let mut b = [0u8; 16];
                    b.copy_from_slice(&t.pred);
                    let mut dq = [0i32; 16];
                    dq.copy_from_slice(&t.dq);
                    idct_add_tx!(4, &dq, &mut b);
                    for r in 0..4 {
                        for c in 0..4 {
                            recon[(t.ly + r) * bw + t.lx + c] = b[r * 4 + c];
                        }
                    }
                }
                TX_8X8 => {
                    let mut b = [0u8; 64];
                    b.copy_from_slice(&t.pred);
                    let mut dq = [0i32; 64];
                    dq.copy_from_slice(&t.dq);
                    idct_add_tx!(8, &dq, &mut b);
                    for r in 0..8 {
                        for c in 0..8 {
                            recon[(t.ly + r) * bw + t.lx + c] = b[r * 8 + c];
                        }
                    }
                }
                TX_16X16 => {
                    let mut b = [0u8; 256];
                    b.copy_from_slice(&t.pred);
                    let mut dq = [0i32; 256];
                    dq.copy_from_slice(&t.dq);
                    idct_add_tx!(16, &dq, &mut b);
                    for r in 0..16 {
                        for c in 0..16 {
                            recon[(t.ly + r) * bw + t.lx + c] = b[r * 16 + c];
                        }
                    }
                }
                _ => {
                    let mut b = [0u8; 1024];
                    b.copy_from_slice(&t.pred);
                    let mut dq = [0i32; 1024];
                    dq.copy_from_slice(&t.dq);
                    idct_add_tx!(32, &dq, &mut b);
                    for r in 0..32 {
                        for c in 0..32 {
                            recon[(t.ly + r) * bw + t.lx + c] = b[r * 32 + c];
                        }
                    }
                }
            }
        }
    }

    fn ent_ctx(&self, plane: EntPlane, x: usize, y: usize, n4: usize) -> usize {
        match plane {
            EntPlane::Y => {
                let a = (0..n4).any(|i| self.above_ent_y[x / 4 + i] != 0);
                let l = (0..n4).any(|i| self.left_ent_y[(y / 4 + i) & 15] != 0);
                a as usize + l as usize
            }
            EntPlane::U => {
                let a = (0..n4).any(|i| self.above_ent_u[x / 4 + i] != 0);
                let l = (0..n4).any(|i| self.left_ent_u[(y / 4 + i) & 7] != 0);
                a as usize + l as usize
            }
            EntPlane::V => {
                let a = (0..n4).any(|i| self.above_ent_v[x / 4 + i] != 0);
                let l = (0..n4).any(|i| self.left_ent_v[(y / 4 + i) & 7] != 0);
                a as usize + l as usize
            }
        }
    }

    fn set_ent(&mut self, plane: EntPlane, x: usize, y: usize, n4: usize, has: u8) {
        match plane {
            EntPlane::Y => {
                for i in 0..n4 {
                    self.above_ent_y[x / 4 + i] = has;
                    self.left_ent_y[(y / 4 + i) & 15] = has;
                }
            }
            EntPlane::U => {
                for i in 0..n4 {
                    self.above_ent_u[x / 4 + i] = has;
                    self.left_ent_u[(y / 4 + i) & 7] = has;
                }
            }
            EntPlane::V => {
                for i in 0..n4 {
                    self.above_ent_v[x / 4 + i] = has;
                    self.left_ent_v[(y / 4 + i) & 7] = has;
                }
            }
        }
    }

    fn clear_ent_block(&mut self, mi_row: usize, mi_col: usize, bw_mi: usize, bh_mi: usize) {
        for c in 0..bw_mi {
            if mi_col + c >= self.mi_cols {
                break;
            }
            for t in 0..2 {
                self.above_ent_y[(mi_col + c) * 2 + t] = 0;
            }
            self.above_ent_u[mi_col + c] = 0;
            self.above_ent_v[mi_col + c] = 0;
        }
        for r in 0..bh_mi {
            if mi_row + r >= self.mi_rows {
                break;
            }
            for t in 0..2 {
                self.left_ent_y[((mi_row + r) & 7) * 2 + t] = 0;
            }
            self.left_ent_u[(mi_row + r) & 7] = 0;
            self.left_ent_v[(mi_row + r) & 7] = 0;
        }
    }

    fn write_inter_mode_bits(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw_px: u32,
        bh_px: u32,
        skip: bool,
        tx_size: u8,
        mode: i8,
        ref_frame: i8,
        second_ref: i8,
        mv: Mv,
        mv1: Mv,
        interp_filter: InterpFilter,
        segment_id: u8,
    ) {
        write_segment_id(&mut self.bool, segment_id);
        let skip_ctx = self.skip_context(mi_row, mi_col);
        self.bool.put_bool(DEFAULT_SKIP_PROBS[skip_ctx], skip);

        let ii_ctx = self.intra_inter_context(mi_row, mi_col);
        self.bool.put_bool(DEFAULT_INTRA_INTER_PROBS[ii_ctx], true);

        // TX_MODE_SELECT: size is coded for inter blocks that are not skip.
        if !skip {
            self.write_selected_tx_size(mi_row, mi_col, bw_px, bh_px, tx_size);
        }

        let compound = second_ref > INTRA_FRAME;
        match self.ref_mode {
            RefMode::Select => {
                let ctx = self.comp_inter_context(mi_row, mi_col);
                self.bool.put_bool(DEFAULT_COMP_INTER_PROBS[ctx], compound);
            }
            RefMode::Compound => debug_assert!(compound),
            RefMode::Single => debug_assert!(!compound),
        }

        if compound {
            // idx = sign_bias[fixed]=1 → variable is ref_frame[0], fixed is ref_frame[1].
            let bit = ref_frame == COMP_VAR_REF[1];
            let ctx = self.comp_ref_context(mi_row, mi_col);
            self.bool.put_bool(DEFAULT_COMP_REF_PROBS[ctx], bit);
        } else {
            let ref_ctx = self.single_ref_p1_context(mi_row, mi_col);
            if ref_frame == LAST_FRAME {
                self.bool
                    .put_bool(DEFAULT_SINGLE_REF_PROBS[ref_ctx][0], false);
            } else {
                self.bool
                    .put_bool(DEFAULT_SINGLE_REF_PROBS[ref_ctx][0], true);
                let ctx2 = self.single_ref_p2_context(mi_row, mi_col);
                self.bool
                    .put_bool(DEFAULT_SINGLE_REF_PROBS[ctx2][1], ref_frame != GOLDEN_FRAME);
            }
        }

        let mode_ctx = self.inter_mode_context(mi_row, mi_col, bw_px, bh_px);
        write_tree(
            &mut self.bool,
            &INTER_MODE_TREE,
            &DEFAULT_INTER_MODE_PROBS[mode_ctx],
            inter_mode_token(mode),
        );

        let filt_ctx = self.switchable_interp_context(mi_row, mi_col);
        write_tree(
            &mut self.bool,
            &SWITCHABLE_INTERP_TREE,
            &DEFAULT_SWITCHABLE_INTERP_PROBS[filt_ctx],
            interp_filter as i8,
        );

        if mode == NEWMV {
            let (nearest0, _) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, ref_frame);
            write_mv(&mut self.bool, mv, nearest0, false);
            if compound {
                let (nearest1, _) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, second_ref);
                write_mv(&mut self.bool, mv1, nearest1, false);
            }
        }
    }

    fn switchable_interp_context(&self, mi_row: usize, mi_col: usize) -> usize {
        // `get_pred_context_switchable_interp` — missing neighbors use SWITCHABLE_FILTERS.
        let missing = InterpFilter::SWITCHABLE_FILTERS as usize;
        let left = if self.left_available(mi_col) {
            self.mi[mi_row * self.mi_cols + mi_col - 1].interp_filter as usize
        } else {
            missing
        };
        let above = if mi_row > 0 {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].interp_filter as usize
        } else {
            missing
        };
        if left == above {
            left
        } else if left == missing {
            above
        } else if above == missing {
            left
        } else {
            missing
        }
    }

    fn pick_interp_filter(
        &self,
        px: usize,
        py: usize,
        bw: usize,
        bh: usize,
        w: usize,
        h: usize,
        mv: Mv,
        mv1: Mv,
        ref_frame: i8,
        second_ref: i8,
    ) -> InterpFilter {
        let subpel = (mv.row & 7) != 0
            || (mv.col & 7) != 0
            || (second_ref > INTRA_FRAME && ((mv1.row & 7) != 0 || (mv1.col & 7) != 0));
        if !subpel {
            return InterpFilter::EightTap;
        }
        let compound = second_ref > INTRA_FRAME;
        let ref0 = self.recon_for(ref_frame);
        let mut best = InterpFilter::EightTap;
        let mut best_sad = u32::MAX;
        for &f in &[
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
        ] {
            let mut pred = vec![0u8; bw * bh];
            if compound {
                let ref1 = self.recon_for(second_ref);
                let mut a = vec![0u8; bw * bh];
                let mut b = vec![0u8; bw * bh];
                predict_inter(
                    &ref0.y, w, w, h, px, py, bw, bh, mv.row, mv.col, false, f, &mut a,
                );
                predict_inter(
                    &ref1.y, w, w, h, px, py, bw, bh, mv1.row, mv1.col, false, f, &mut b,
                );
                avg_bytes(&a, &b, &mut pred);
            } else {
                predict_inter(
                    &ref0.y, w, w, h, px, py, bw, bh, mv.row, mv.col, false, f, &mut pred,
                );
            }
            let mut sad = 0u32;
            for r in 0..bh {
                for c in 0..bw {
                    let s = self.src.y[(py + r) * w + px + c];
                    sad += u32::from(s.abs_diff(pred[r * bw + c]));
                }
            }
            if sad < best_sad || (sad == best_sad && (f as u8) < (best as u8)) {
                best_sad = sad;
                best = f;
            }
        }
        best
    }

    fn write_selected_tx_size(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw_px: u32,
        bh_px: u32,
        tx_size: u8,
    ) {
        let max_tx = max_tx_size_wh(bw_px, bh_px);
        let ctx = self.tx_size_context(mi_row, mi_col, max_tx);
        match max_tx {
            TX_8X8 => {
                self.bool
                    .put_bool(DEFAULT_TX_PROBS_8[ctx][0], tx_size != TX_4X4);
            }
            TX_16X16 => {
                let probs = &DEFAULT_TX_PROBS_16[ctx];
                self.bool.put_bool(probs[0], tx_size != TX_4X4);
                if tx_size != TX_4X4 {
                    self.bool.put_bool(probs[1], tx_size != TX_8X8);
                }
            }
            _ => {
                let probs = &DEFAULT_TX_PROBS_32[ctx];
                self.bool.put_bool(probs[0], tx_size != TX_4X4);
                if tx_size != TX_4X4 {
                    self.bool.put_bool(probs[1], tx_size != TX_8X8);
                    if tx_size != TX_8X8 {
                        self.bool.put_bool(probs[2], tx_size != TX_16X16);
                    }
                }
            }
        }
    }

    /// `get_tx_size_context` — `(above + left) > max_tx_size`.
    fn tx_size_context(&self, mi_row: usize, mi_col: usize, max_tx: u8) -> usize {
        let has_above = mi_row > 0;
        let has_left = self.left_available(mi_col);
        let mut above_ctx = if has_above {
            let m = &self.mi[(mi_row - 1) * self.mi_cols + mi_col];
            if !m.skip {
                m.tx_size as i32
            } else {
                max_tx as i32
            }
        } else {
            max_tx as i32
        };
        let mut left_ctx = if has_left {
            let m = &self.mi[mi_row * self.mi_cols + mi_col - 1];
            if !m.skip {
                m.tx_size as i32
            } else {
                max_tx as i32
            }
        } else {
            max_tx as i32
        };
        if !has_left {
            left_ctx = above_ctx;
        }
        if !has_above {
            above_ctx = left_ctx;
        }
        ((above_ctx + left_ctx) > max_tx as i32) as usize
    }

    fn store_mi(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw_mi: usize,
        bh_mi: usize,
        info: ResidualMi,
    ) {
        for r in 0..bh_mi {
            for c in 0..bw_mi {
                if mi_row + r < self.mi_rows && mi_col + c < self.mi_cols {
                    self.mi[(mi_row + r) * self.mi_cols + mi_col + c] = ResidualMi {
                        block_origin: r == 0 && c == 0,
                        ..info
                    };
                }
            }
        }
    }

    /// Pick LAST / GOLDEN / ALTREF (single) or LAST/GOLDEN+ALTREF compound by SAD.
    fn pick_ref_and_mode(
        &self,
        mi_row: usize,
        mi_col: usize,
        bw_px: u32,
        bh_px: u32,
        px: usize,
        py: usize,
        bw: usize,
        bh: usize,
        w: usize,
        h: usize,
    ) -> InterPick {
        let try_ref = |recon: &Reconstruction, ref_frame: i8| {
            let (nearest, near) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, ref_frame);
            let searched = search_mv(
                &self.src.y,
                w,
                &recon.y,
                w,
                w,
                h,
                px,
                py,
                bw,
                bh,
                DEFAULT_RANGE_PEL,
            );
            let (mv, mode) = pick_inter_mode(searched, nearest, near);
            let sad = sad_mv(&self.src.y, w, &recon.y, w, w, h, px, py, bw, bh, mv);
            (
                InterPick {
                    mv,
                    mv1: Mv::default(),
                    mode,
                    ref_frame,
                    second_ref: INTRA_FRAME,
                },
                sad as u64,
            )
        };

        let mut best: Option<(InterPick, u64)> = None;
        if self.ref_mode != RefMode::Compound {
            best = Some(try_ref(self.reference, LAST_FRAME));
            if let Some(golden) = self.golden {
                let gold = try_ref(golden, GOLDEN_FRAME);
                if gold.1 <= best.as_ref().unwrap().1 {
                    best = Some(gold);
                }
            }
            if let Some(altref) = self.altref {
                let alt = try_ref(altref, ALTREF_FRAME);
                if alt.1 <= best.as_ref().unwrap().1 {
                    best = Some(alt);
                }
            }
        }

        // Compound: fixed=ALTREF + var∈{LAST, GOLDEN}.
        if self.ref_mode == RefMode::Compound {
            if let Some(alt) = self.altref {
                let try_comp = |var_recon: &Reconstruction, var_ref: i8| {
                    let (n0, near0) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, var_ref);
                    let (n1, near1) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, ALTREF_FRAME);
                    let s0 = search_mv(
                        &self.src.y,
                        w,
                        &var_recon.y,
                        w,
                        w,
                        h,
                        px,
                        py,
                        bw,
                        bh,
                        DEFAULT_RANGE_PEL,
                    );
                    let s1 = search_mv(
                        &self.src.y,
                        w,
                        &alt.y,
                        w,
                        w,
                        h,
                        px,
                        py,
                        bw,
                        bh,
                        DEFAULT_RANGE_PEL,
                    );
                    // Independent searches are not jointly optimal — score the
                    // standard mode candidates (incl. colocated ZEROMV).
                    let cands = [
                        (Mv::default(), Mv::default(), ZEROMV),
                        (n0, n1, NEARESTMV),
                        (near0, near1, NEARMV),
                        (s0, s1, NEWMV),
                    ];
                    let mut best_c = None;
                    for &(mv0, mv1, mode) in &cands {
                        let sad = sad_compound(
                            &self.src.y,
                            w,
                            &var_recon.y,
                            w,
                            &alt.y,
                            w,
                            w,
                            h,
                            px,
                            py,
                            bw,
                            bh,
                            mv0,
                            mv1,
                        );
                        let pick = InterPick {
                            mv: mv0,
                            mv1,
                            mode,
                            ref_frame: var_ref,
                            second_ref: ALTREF_FRAME,
                        };
                        match best_c {
                            None => best_c = Some((pick, sad)),
                            Some((_, s)) if sad < s => best_c = Some((pick, sad)),
                            _ => {}
                        }
                    }
                    best_c.unwrap()
                };
                let mut best_c = try_comp(self.reference, LAST_FRAME);
                if let Some(golden) = self.golden {
                    let gold_c = try_comp(golden, GOLDEN_FRAME);
                    if gold_c.1 < best_c.1 {
                        best_c = gold_c;
                    }
                }
                return best_c.0;
            }
        }
        // SELECT: also try compound when it strictly beats single.
        if self.ref_mode == RefMode::Select {
            if let Some(alt) = self.altref {
                let try_comp = |var_recon: &Reconstruction, var_ref: i8| {
                    let (n0, near0) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, var_ref);
                    let (n1, near1) = self.ref_mvs(mi_row, mi_col, bw_px, bh_px, ALTREF_FRAME);
                    let s0 = search_mv(
                        &self.src.y,
                        w,
                        &var_recon.y,
                        w,
                        w,
                        h,
                        px,
                        py,
                        bw,
                        bh,
                        DEFAULT_RANGE_PEL,
                    );
                    let s1 = search_mv(
                        &self.src.y,
                        w,
                        &alt.y,
                        w,
                        w,
                        h,
                        px,
                        py,
                        bw,
                        bh,
                        DEFAULT_RANGE_PEL,
                    );
                    let cands = [
                        (Mv::default(), Mv::default(), ZEROMV),
                        (n0, n1, NEARESTMV),
                        (near0, near1, NEARMV),
                        (s0, s1, NEWMV),
                    ];
                    let mut best_c = None;
                    for &(mv0, mv1, mode) in &cands {
                        let sad = sad_compound(
                            &self.src.y,
                            w,
                            &var_recon.y,
                            w,
                            &alt.y,
                            w,
                            w,
                            h,
                            px,
                            py,
                            bw,
                            bh,
                            mv0,
                            mv1,
                        );
                        let pick = InterPick {
                            mv: mv0,
                            mv1,
                            mode,
                            ref_frame: var_ref,
                            second_ref: ALTREF_FRAME,
                        };
                        match best_c {
                            None => best_c = Some((pick, sad)),
                            Some((_, s)) if sad < s => best_c = Some((pick, sad)),
                            _ => {}
                        }
                    }
                    best_c.unwrap()
                };
                let last_c = try_comp(self.reference, LAST_FRAME);
                if last_c.1 < best.as_ref().unwrap().1 {
                    best = Some(last_c);
                }
                if let Some(golden) = self.golden {
                    let gold_c = try_comp(golden, GOLDEN_FRAME);
                    if gold_c.1 < best.as_ref().unwrap().1 {
                        best = Some(gold_c);
                    }
                }
            }
        }
        best.expect("no inter pick").0
    }

    fn recon_for(&self, ref_frame: i8) -> &'a Reconstruction {
        match ref_frame {
            GOLDEN_FRAME => self.golden.unwrap_or(self.reference),
            ALTREF_FRAME => self.altref.unwrap_or(self.reference),
            _ => self.reference,
        }
    }

    /// Neighbor offsets for `mv_ref_blocks` / mode-context first two slots.
    fn mv_ref_neighbors(bw_px: u32, bh_px: u32) -> &'static [(i32, i32)] {
        match (bw_px, bh_px) {
            (8, 16) => &[
                (0, -1),
                (-1, 0),
                (1, -1),
                (-1, -1),
                (0, -2),
                (-2, 0),
                (-2, -1),
                (-1, -2),
            ],
            (16, 8) => &[
                (-1, 0),
                (0, -1),
                (-1, 1),
                (-1, -1),
                (-2, 0),
                (0, -2),
                (-1, -2),
                (-2, -1),
            ],
            (16, 32) => &[
                (0, -1),
                (-1, 0),
                (2, -1),
                (-1, -1),
                (-1, 1),
                (0, -3),
                (-3, 0),
                (-3, -3),
            ],
            (32, 16) => &[
                (-1, 0),
                (0, -1),
                (-1, 2),
                (-1, -1),
                (1, -1),
                (-3, 0),
                (0, -3),
                (-3, -3),
            ],
            (32, 64) => &[
                (0, -1),
                (-1, 0),
                (4, -1),
                (-1, 2),
                (-1, -1),
                (0, -3),
                (-3, 0),
                (2, -1),
            ],
            (64, 32) => &[
                (-1, 0),
                (0, -1),
                (-1, 4),
                (2, -1),
                (-1, -1),
                (-3, 0),
                (0, -3),
                (-1, 2),
            ],
            (64, 64) => &[
                (-1, 3),
                (3, -1),
                (-1, 4),
                (4, -1),
                (-1, -1),
                (-1, 0),
                (0, -1),
                (-1, 6),
            ],
            (32, 32) => &[
                (-1, 1),
                (1, -1),
                (-1, 2),
                (2, -1),
                (-1, -1),
                (-3, 0),
                (0, -3),
                (-3, -3),
            ],
            (16, 16) => &[
                (-1, 0),
                (0, -1),
                (-1, 1),
                (1, -1),
                (-1, -1),
                (-3, 0),
                (0, -3),
                (-3, -3),
            ],
            _ => &[
                (-1, 0),
                (0, -1),
                (-1, -1),
                (-2, 0),
                (0, -2),
                (-2, -1),
                (-1, -2),
                (-2, -2),
            ],
        }
    }

    /// Nearest + near MVs for `ref_frame` (`find_mv_refs_idx`).
    fn ref_mvs(
        &self,
        mi_row: usize,
        mi_col: usize,
        bw_px: u32,
        bh_px: u32,
        ref_frame: i8,
    ) -> (Mv, Mv) {
        let nbrs = Self::mv_ref_neighbors(bw_px, bh_px);
        let bw_mi = (bw_px / 8) as usize;
        let bh_mi = (bh_px / 8) as usize;
        let mut list = [Mv::default(); 2];
        let mut count = 0usize;
        let clamp = |cand: Mv| {
            clamp_mv_ref(
                cand,
                mi_row,
                mi_col,
                self.mi_rows,
                self.mi_cols,
                bw_mi,
                bh_mi,
            )
        };
        let add = |list: &mut [Mv; 2], count: &mut usize, cand: Mv| -> bool {
            let cand = clamp(cand);
            if *count == 0 {
                list[0] = cand;
                *count = 1;
                false
            } else if cand != list[0] {
                list[1] = cand;
                true
            } else {
                false
            }
        };
        let mut different_ref_found = false;
        for &(dr, dc) in nbrs {
            let r = mi_row as i32 + dr;
            let c = mi_col as i32 + dc;
            if !self.mi_in_tile(r, c) {
                continue;
            }
            different_ref_found = true;
            let n = &self.mi[r as usize * self.mi_cols + c as usize];
            if n.is_inter && n.ref_frame == ref_frame {
                if add(&mut list, &mut count, n.mv) {
                    return (list[0], list[1]);
                }
            } else if n.is_inter && n.second_ref == ref_frame
                && add(&mut list, &mut count, n.mv1) {
                    return (list[0], list[1]);
                }
        }
        // `IF_DIFF_REF_FRAME_ADD_MV` — negate when sign biases differ.
        if different_ref_found && count < 2 {
            for &(dr, dc) in nbrs {
                let r = mi_row as i32 + dr;
                let c = mi_col as i32 + dc;
                if !self.mi_in_tile(r, c) {
                    continue;
                }
                let n = &self.mi[r as usize * self.mi_cols + c as usize];
                if !n.is_inter {
                    continue;
                }
                if n.ref_frame != ref_frame
                    && add(
                        &mut list,
                        &mut count,
                        scale_mv_for_ref(n.mv, n.ref_frame, ref_frame),
                    ) {
                        return (list[0], list[1]);
                    }
                if n.second_ref > INTRA_FRAME && n.second_ref != ref_frame && n.mv1 != n.mv
                    && add(
                        &mut list,
                        &mut count,
                        scale_mv_for_ref(n.mv1, n.second_ref, ref_frame),
                    ) {
                        return (list[0], list[1]);
                    }
            }
        }
        (list[0], list[1])
    }

    fn skip_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].skip as usize
        } else {
            0
        };
        let left = if self.left_available(mi_col) {
            self.mi[mi_row * self.mi_cols + (mi_col - 1)].skip as usize
        } else {
            0
        };
        above + left
    }

    fn intra_inter_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if li && ai {
                    3
                } else if li || ai {
                    1
                } else {
                    0
                }
            }
            (Some(e), None) | (None, Some(e)) => 2 * (!e.is_inter as usize),
            (None, None) => 0,
        }
    }

    /// Inter-mode entropy context from the first two `mv_ref_blocks` neighbors
    /// (`find_mv_refs_idx` / `mode_context[ref_frame]`).
    fn inter_mode_context(&self, mi_row: usize, mi_col: usize, bw_px: u32, bh_px: u32) -> usize {
        let nbrs = Self::mv_ref_neighbors(bw_px, bh_px);
        let mut counter = 0i32;
        for &(dr, dc) in nbrs.iter().take(2) {
            let r = mi_row as i32 + dr;
            let c = mi_col as i32 + dc;
            if !self.mi_in_tile(r, c) {
                continue;
            }
            counter += mode_2_counter(self.mi[r as usize * self.mi_cols + c as usize].mode);
        }
        counter_to_inter_mode_ctx(counter)
    }

    fn single_ref_p1_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if ai && li {
                    2
                } else if ai || li {
                    let edge = if ai { l } else { a };
                    if !edge.is_compound() {
                        4 * (edge.ref_frame == LAST_FRAME) as usize
                    } else {
                        1 + (edge.ref_frame == LAST_FRAME || edge.second_ref == LAST_FRAME) as usize
                    }
                } else {
                    let a2 = a.is_compound();
                    let l2 = l.is_compound();
                    if a2 && l2 {
                        1 + (a.ref_frame == LAST_FRAME
                            || a.second_ref == LAST_FRAME
                            || l.ref_frame == LAST_FRAME
                            || l.second_ref == LAST_FRAME) as usize
                    } else if a2 || l2 {
                        let rfs = if !a2 { a.ref_frame } else { l.ref_frame };
                        let (crf1, crf2) = if a2 {
                            (a.ref_frame, a.second_ref)
                        } else {
                            (l.ref_frame, l.second_ref)
                        };
                        if rfs == LAST_FRAME {
                            3 + (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                        } else {
                            (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                        }
                    } else {
                        2 * (a.ref_frame == LAST_FRAME) as usize
                            + 2 * (l.ref_frame == LAST_FRAME) as usize
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter {
                    2
                } else if !e.is_compound() {
                    4 * (e.ref_frame == LAST_FRAME) as usize
                } else {
                    1 + (e.ref_frame == LAST_FRAME || e.second_ref == LAST_FRAME) as usize
                }
            }
            (None, None) => 2,
        }
    }

    /// `vp9_get_pred_context_single_ref_p2` (handles compound neighbors).
    fn single_ref_p2_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if ai && li {
                    2
                } else if ai || li {
                    let edge = if ai { l } else { a };
                    if !edge.is_compound() {
                        if edge.ref_frame == LAST_FRAME {
                            3
                        } else {
                            4 * (edge.ref_frame == GOLDEN_FRAME) as usize
                        }
                    } else {
                        1 + 2
                            * (edge.ref_frame == GOLDEN_FRAME || edge.second_ref == GOLDEN_FRAME)
                                as usize
                    }
                } else {
                    let a2 = a.is_compound();
                    let l2 = l.is_compound();
                    if a2 && l2 {
                        if a.ref_frame == l.ref_frame && a.second_ref == l.second_ref {
                            3 * (a.ref_frame == GOLDEN_FRAME
                                || a.second_ref == GOLDEN_FRAME
                                || l.ref_frame == GOLDEN_FRAME
                                || l.second_ref == GOLDEN_FRAME)
                                as usize
                        } else {
                            2
                        }
                    } else if a2 || l2 {
                        let rfs = if !a2 { a.ref_frame } else { l.ref_frame };
                        let (crf1, crf2) = if a2 {
                            (a.ref_frame, a.second_ref)
                        } else {
                            (l.ref_frame, l.second_ref)
                        };
                        if rfs == GOLDEN_FRAME {
                            3 + (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                        } else if rfs == ALTREF_FRAME {
                            (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                        } else {
                            1 + 2 * (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                        }
                    } else if a.ref_frame == LAST_FRAME && l.ref_frame == LAST_FRAME {
                        3
                    } else if a.ref_frame == LAST_FRAME || l.ref_frame == LAST_FRAME {
                        let edge0 = if a.ref_frame == LAST_FRAME {
                            l.ref_frame
                        } else {
                            a.ref_frame
                        };
                        4 * (edge0 == GOLDEN_FRAME) as usize
                    } else {
                        2 * (a.ref_frame == GOLDEN_FRAME) as usize
                            + 2 * (l.ref_frame == GOLDEN_FRAME) as usize
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter || (e.ref_frame == LAST_FRAME && !e.is_compound()) {
                    2
                } else if !e.is_compound() {
                    4 * (e.ref_frame == GOLDEN_FRAME) as usize
                } else {
                    3 * (e.ref_frame == GOLDEN_FRAME || e.second_ref == GOLDEN_FRAME) as usize
                }
            }
            (None, None) => 2,
        }
    }

    /// `vp9_get_reference_mode_context` — compound vs single bit.
    fn comp_inter_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                if !a.is_compound() && !l.is_compound() {
                    (a.ref_frame == COMP_FIXED_REF) as usize
                        ^ (l.ref_frame == COMP_FIXED_REF) as usize
                } else if !a.is_compound() {
                    2 + (a.ref_frame == COMP_FIXED_REF || !a.is_inter) as usize
                } else if !l.is_compound() {
                    2 + (l.ref_frame == COMP_FIXED_REF || !l.is_inter) as usize
                } else {
                    4
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_compound() {
                    (e.ref_frame == COMP_FIXED_REF) as usize
                } else {
                    3
                }
            }
            (None, None) => 1,
        }
    }

    /// `vp9_get_pred_context_comp_ref_p` — variable ref LAST vs GOLDEN.
    fn comp_ref_context(&self, mi_row: usize, mi_col: usize) -> usize {
        // fix_ref_idx = sign_bias[ALTREF] = 1 → var_ref_idx = 0 (variable in ref_frame[0]).
        let above = if mi_row > 0 {
            Some(&self.mi[(mi_row - 1) * self.mi_cols + mi_col])
        } else {
            None
        };
        let left = if self.left_available(mi_col) {
            Some(&self.mi[mi_row * self.mi_cols + (mi_col - 1)])
        } else {
            None
        };
        let var_of = |m: &ResidualMi| -> i8 {
            if m.is_compound() {
                m.ref_frame // var at [0]
            } else {
                m.ref_frame
            }
        };
        match (above, left) {
            (Some(a), Some(l)) => {
                let ai = !a.is_inter;
                let li = !l.is_inter;
                if ai && li {
                    2
                } else if ai || li {
                    let edge = if ai { l } else { a };
                    if !edge.is_compound() {
                        1 + 2 * (edge.ref_frame != COMP_VAR_REF[1]) as usize
                    } else {
                        1 + 2 * (edge.ref_frame != COMP_VAR_REF[1]) as usize
                    }
                } else {
                    let a_sg = !a.is_compound();
                    let l_sg = !l.is_compound();
                    let vrfa = var_of(a);
                    let vrfl = var_of(l);
                    if vrfa == vrfl && COMP_VAR_REF[1] == vrfa {
                        0
                    } else if l_sg && a_sg {
                        if (vrfa == COMP_FIXED_REF && vrfl == COMP_VAR_REF[0])
                            || (vrfl == COMP_FIXED_REF && vrfa == COMP_VAR_REF[0])
                        {
                            4
                        } else if vrfa == vrfl {
                            3
                        } else {
                            1
                        }
                    } else if l_sg || a_sg {
                        let vrfc = if l_sg { vrfa } else { vrfl };
                        let rfs = if a_sg { vrfa } else { vrfl };
                        if vrfc == COMP_VAR_REF[1] && rfs != COMP_VAR_REF[1] {
                            1
                        } else if rfs == COMP_VAR_REF[1] && vrfc != COMP_VAR_REF[1] {
                            2
                        } else {
                            4
                        }
                    } else if vrfa == vrfl {
                        4
                    } else {
                        2
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter {
                    2
                } else if e.is_compound() {
                    4 * (e.ref_frame != COMP_VAR_REF[1]) as usize
                } else {
                    3 * (e.ref_frame != COMP_VAR_REF[1]) as usize
                }
            }
            (None, None) => 2,
        }
    }
}

/// Map inter mode to [`INTER_MODE_TREE`] leaf token (`INTER_OFFSET`).
fn inter_mode_token(mode: i8) -> i8 {
    match mode {
        NEARESTMV => INTER_OFFSET_NEARESTMV,
        NEARMV => INTER_OFFSET_NEARMV,
        ZEROMV => INTER_OFFSET_ZEROMV,
        NEWMV => INTER_OFFSET_NEWMV,
        _ => panic!("not an inter mode: {mode}"),
    }
}

/// SAD of source vs average of two EIGHTTAP predictions.
fn sad_compound(
    src: &[u8],
    src_stride: usize,
    ref0: &[u8],
    ref0_stride: usize,
    ref1: &[u8],
    ref1_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    mv0: Mv,
    mv1: Mv,
) -> u64 {
    let mut a = vec![0u8; bw * bh];
    let mut b = vec![0u8; bw * bh];
    predict_inter(
        ref0,
        ref0_stride,
        ref_w,
        ref_h,
        x,
        y,
        bw,
        bh,
        mv0.row,
        mv0.col,
        false,
        InterpFilter::EightTap,
        &mut a,
    );
    predict_inter(
        ref1,
        ref1_stride,
        ref_w,
        ref_h,
        x,
        y,
        bw,
        bh,
        mv1.row,
        mv1.col,
        false,
        InterpFilter::EightTap,
        &mut b,
    );
    let mut sad = 0u64;
    for r in 0..bh {
        for c in 0..bw {
            let pred = ((u16::from(a[r * bw + c]) + u16::from(b[r * bw + c]) + 1) >> 1) as u8;
            let s = src[(y + r) * src_stride + (x + c)];
            sad += u64::from(s.abs_diff(pred));
        }
    }
    sad
}

/// Prefer predicted modes when the searched MV matches a ref candidate.
fn pick_inter_mode(best: Mv, nearest: Mv, near: Mv) -> (Mv, i8) {
    if best == Mv::default() {
        (Mv::default(), ZEROMV)
    } else if best == nearest {
        (nearest, NEARESTMV)
    } else if best == near {
        (near, NEARMV)
    } else {
        (best, NEWMV)
    }
}

/// `clamp_mv_ref` (`MV_BORDER = 16 << 3`); `bw_mi`/`bh_mi` are block size in MI units.
fn clamp_mv_ref(
    mv: Mv,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    mi_cols: usize,
    bw_mi: usize,
    bh_mi: usize,
) -> Mv {
    const MV_BORDER: i32 = 16 << 3;
    const MI_SIZE: i32 = 8;
    let left = -(mi_col as i32 * MI_SIZE * 8) - MV_BORDER;
    let right = ((mi_cols as i32) - bw_mi as i32 - mi_col as i32) * MI_SIZE * 8 + MV_BORDER;
    let top = -(mi_row as i32 * MI_SIZE * 8) - MV_BORDER;
    let bottom = ((mi_rows as i32) - bh_mi as i32 - mi_row as i32) * MI_SIZE * 8 + MV_BORDER;
    Mv {
        row: (i32::from(mv.row)).clamp(top, bottom) as i16,
        col: (i32::from(mv.col)).clamp(left, right) as i16,
    }
}

fn compressed_header_inter(ref_mode: RefMode) -> Vec<u8> {
    let mut e = BoolEncoder::new();
    e.put_literal(3, 2); // tx_mode base = ALLOW_32X32
    e.put_bit(true); // TX_MODE_SELECT
                     // tx_probs: p8x8 (2×1) + p16x16 (2×2) + p32x32 (2×3) = 12 no-updates
    for _ in 0..(2 + 2 * 2 + 2 * 3) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    e.put_bit(false); // TX_4X4 coef probs: no update
    e.put_bit(false); // TX_8X8 coef probs: no update
    e.put_bit(false); // TX_16X16 coef probs: no update
    e.put_bit(false); // TX_32X32 coef probs: no update
    for _ in 0..3 {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    for _ in 0..(7 * 3) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    // switchable_interp_probs: 4 contexts × 2
    for _ in 0..(4 * 2) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    for _ in 0..4 {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    // reference_mode (compound allowed: ALTREF sign-bias differs)
    match ref_mode {
        RefMode::Single => {
            e.put_bit(false);
            for _ in 0..(5 * 2) {
                e.put_bool(DIFF_UPDATE_PROB, false);
            }
        }
        RefMode::Compound => {
            e.put_bit(true);
            e.put_bit(false);
            for _ in 0..5 {
                e.put_bool(DIFF_UPDATE_PROB, false);
            }
        }
        RefMode::Select => {
            e.put_bit(true);
            e.put_bit(true);
            for _ in 0..5 {
                e.put_bool(DIFF_UPDATE_PROB, false);
            }
            for _ in 0..(5 * 2) {
                e.put_bool(DIFF_UPDATE_PROB, false);
            }
            for _ in 0..5 {
                e.put_bool(DIFF_UPDATE_PROB, false);
            }
        }
    }
    for _ in 0..(4 * 9) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    for _ in 0..(16 * 3) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    for _ in 0..nmv_update_count(false) {
        e.put_bool(DIFF_UPDATE_PROB, false);
    }
    e.finish()
}

fn nmv_update_count(allow_hp: bool) -> usize {
    let mut n = 3 + 2 * (1 + 10 + 1 + 10) + 2 * (2 * 3 + 3);
    if allow_hp {
        n += 2 * (1 + 1);
    }
    n
}

fn apply_loop_filter(recon: &mut Reconstruction, lf_mi: &[LfMi], mi_rows: usize, mi_cols: usize) {
    if FILTER_LEVEL == 0 {
        return;
    }
    let w = recon.width as usize;
    let h = recon.height as usize;
    let sb_w = w.div_ceil(64) * 64;
    let sb_h = h.div_ceil(64) * 64;
    let (cw, ch) = (w / 2, h / 2);
    let (sb_cw, sb_ch) = (sb_w / 2, sb_h / 2);

    let mut y = vec![0u8; sb_w * sb_h];
    let mut u = vec![0u8; sb_cw * sb_ch];
    let mut v = vec![0u8; sb_cw * sb_ch];
    for row in 0..h {
        y[row * sb_w..row * sb_w + w].copy_from_slice(&recon.y[row * w..row * w + w]);
        for col in w..sb_w {
            y[row * sb_w + col] = recon.y[row * w + w - 1];
        }
    }
    for row in h..sb_h {
        y.copy_within((h - 1) * sb_w..(h - 1) * sb_w + sb_w, row * sb_w);
    }
    for row in 0..ch {
        u[row * sb_cw..row * sb_cw + cw].copy_from_slice(&recon.u[row * cw..row * cw + cw]);
        v[row * sb_cw..row * sb_cw + cw].copy_from_slice(&recon.v[row * cw..row * cw + cw]);
        for col in cw..sb_cw {
            u[row * sb_cw + col] = recon.u[row * cw + cw - 1];
            v[row * sb_cw + col] = recon.v[row * cw + cw - 1];
        }
    }
    for row in ch..sb_ch {
        u.copy_within((ch - 1) * sb_cw..(ch - 1) * sb_cw + sb_cw, row * sb_cw);
        v.copy_within((ch - 1) * sb_cw..(ch - 1) * sb_cw + sb_cw, row * sb_cw);
    }

    loop_filter_frame(
        &mut y,
        sb_w,
        &mut u,
        &mut v,
        sb_cw,
        w,
        h,
        lf_mi,
        mi_rows,
        mi_cols,
        FILTER_LEVEL,
        FILTER_SHARPNESS,
    );

    for row in 0..h {
        recon.y[row * w..row * w + w].copy_from_slice(&y[row * sb_w..row * sb_w + w]);
    }
    for row in 0..ch {
        recon.u[row * cw..row * cw + cw].copy_from_slice(&u[row * sb_cw..row * sb_cw + cw]);
        recon.v[row * cw..row * cw + cw].copy_from_slice(&v[row * sb_cw..row * sb_cw + cw]);
    }
}

fn lf_mi_from_residual(mi: &[ResidualMi]) -> Vec<LfMi> {
    mi.iter()
        .map(|m| LfMi {
            skip: m.skip,
            is_inter: m.is_inter,
            tx_size_y: m.tx_size,
            bw_mi: (m.bw_px / 8) as u8,
            bh_mi: (m.bh_px / 8) as u8,
            bsize_idx: partition_ctx_index_wh(m.bw_px, m.bh_px),
            block_origin: m.block_origin,
            // Compound uses ref_frame[0] (variable ref) for LF level lookup.
            level: lf_level_for_mi(FILTER_LEVEL, m.ref_frame, m.mode),
        })
        .collect()
}

fn segment_id_for_mi_col(mi_col: usize, mi_cols: usize) -> u8 {
    // Right half of the frame → segment 1 (ALT_Q); left half → segment 0.
    if mi_col >= mi_cols / 2 {
        1
    } else {
        0
    }
}

fn qindex_for_segment(segment_id: u8) -> usize {
    if segment_id == 1 {
        (QINDEX as i32 + SEG1_ALT_Q_DELTA).clamp(0, MAXQ) as usize
    } else {
        QINDEX_USIZE
    }
}

/// `vp9_write_tree` with `len = 3` — segment_id is the 3-bit path.
fn write_segment_id(e: &mut BoolEncoder, segment_id: u8) {
    let mut i: i8 = 0;
    let mut len = 3i32;
    let bits = segment_id as i32;
    while len > 0 {
        len -= 1;
        let bit = ((bits >> len) & 1) != 0;
        e.put_bool(MAX_PROB, bit);
        i = SEGMENT_TREE[i as usize + usize::from(bit)];
    }
}

/// Uncompressed `segmentation_params`: map + ALT_Q on segment 1.
fn write_segmentation_params(w: &mut BitWriter) {
    w.write_bit(1); // segmentation_enabled
    w.write_bit(1); // update_map
    for _ in 0..SEG_TREE_PROBS {
        w.write_bit(0); // tree_probs → MAX_PROB
    }
    w.write_bit(0); // temporal_update = 0
    w.write_bit(1); // update_data
    w.write_bit(0); // abs_delta = 0 (delta coding)
    for seg in 0..MAX_SEGMENTS {
        for feat in 0..SEG_LVL_MAX {
            let active = seg == 1 && feat == SEG_LVL_ALT_Q;
            w.write_bit(u32::from(active));
            if active {
                // encode_unsigned_max(|delta|, MAXQ) → 8 bits, then sign.
                w.write_bits(SEG1_ALT_Q_DELTA.unsigned_abs(), 8);
                w.write_bit(0); // positive
            }
        }
    }
}

fn write_uncompressed_inter_header(
    w: &mut BitWriter,
    width: u32,
    _height: u32,
    header_size: u32,
    refresh_frame_flags: u8,
    log2_tile_cols: u32,
) {
    w.write_bits(2, 2);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(1);
    w.write_bit(1);
    w.write_bit(0);

    w.write_bits(0, 2);
    w.write_bits(refresh_frame_flags as u32, 8);

    for i in 0..3u32 {
        w.write_bits(i, 3); // LAST→buf0, GOLDEN→buf1, ALTREF→buf2
                            // Differing ALTREF sign-bias enables compound (fixed=ALTREF, var=LAST/GOLDEN).
        w.write_bit(if i == 2 { 1 } else { 0 });
    }

    w.write_bit(1); // use LAST for frame size
    w.write_bit(0); // render_size same

    w.write_bit(0); // allow_high_precision_mv
    w.write_bit(1); // interp_filter = SWITCHABLE

    w.write_bit(0); // refresh_frame_context
    w.write_bit(1); // frame_parallel_decoding_mode
    w.write_bits(0, 2); // frame_context_idx

    w.write_bits(FILTER_LEVEL as u32, 6);
    w.write_bits(FILTER_SHARPNESS as u32, 3);
    // Mode/ref deltas enabled; no update — decoder keeps defaults from the
    // preceding keyframe's `vp9_setup_past_independence` ({1,0,-1,-1}/{0,0}).
    w.write_bit(1); // loop_filter_delta_enabled
    w.write_bit(0); // loop_filter_delta_update

    w.write_bits(QINDEX, 8);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0);

    write_segmentation_params(w);

    write_tile_info(w, width, log2_tile_cols);
    w.write_bits(header_size, 16);
}

/// libvpx `vp9_get_tile_n_bits` — returns (min_log2, max_log2) for tile columns.
fn tile_log2_bounds(width: u32) -> (u32, u32) {
    let mi_cols = width / 8;
    let sb_cols = mi_cols.div_ceil(8);
    let mut min_log2 = 0u32;
    while (MAX_TILE_WIDTH_SB << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb_cols >> max_log2) >= MIN_TILE_WIDTH_SB {
        max_log2 += 1;
    }
    max_log2 -= 1;
    (min_log2, max_log2)
}

/// Prefer two tile columns when the frame is wide enough (`≥ 512` px → ≥8 SB cols).
fn choose_log2_tile_cols(width: u32) -> u32 {
    let (min_log2, max_log2) = tile_log2_bounds(width);
    if max_log2 >= 1 {
        1u32.max(min_log2).min(max_log2)
    } else {
        min_log2
    }
}

/// libvpx `get_tile_offset` / `vp9_tile_set_col` — MI column range for one tile.
fn tile_mi_col_range(mi_cols: usize, log2_tile_cols: u32, tile_col: u32) -> (usize, usize) {
    let sb_cols = mi_cols.div_ceil(8);
    let start_sb = (tile_col as usize * sb_cols) >> log2_tile_cols;
    let end_sb = ((tile_col as usize + 1) * sb_cols) >> log2_tile_cols;
    let start = (start_sb * 8).min(mi_cols);
    let end = (end_sb * 8).min(mi_cols);
    (start, end)
}

fn write_tile_info(w: &mut BitWriter, width: u32, log2_tile_cols: u32) {
    let (min_log2, max_log2) = tile_log2_bounds(width);
    debug_assert!(log2_tile_cols >= min_log2 && log2_tile_cols <= max_log2);
    // columns: `ones = log2 - min_log2` leading 1s, then a 0 if log2 < max.
    let mut ones = log2_tile_cols - min_log2;
    while ones > 0 {
        w.write_bit(1);
        ones -= 1;
    }
    if log2_tile_cols < max_log2 {
        w.write_bit(0);
    }
    w.write_bit(0); // log2_tile_rows = 0
}

/// Pack independent tile bitstreams: BE32 size prefix on every tile but the last.
fn pack_tile_data(tiles: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, tile) in tiles.iter().enumerate() {
        if i + 1 < tiles.len() {
            out.extend_from_slice(&(tile.len() as u32).to_be_bytes());
        }
        out.extend_from_slice(tile);
    }
    out
}

fn write_tree(e: &mut BoolEncoder, tree: &[i8], probs: &[u8], token: i8) {
    let path = tree_path(tree, 0, token).expect("token not in tree");
    for (node, bit) in path {
        e.put_bool(probs[node >> 1], bit);
    }
}

fn tree_path(tree: &[i8], node: usize, token: i8) -> Option<Vec<(usize, bool)>> {
    for bit in 0..2 {
        let child = tree[node + bit];
        if child <= 0 {
            if -child == token {
                return Some(vec![(node, bit == 1)]);
            }
        } else if let Some(mut rest) = tree_path(tree, child as usize, token) {
            rest.insert(0, (node, bit == 1));
            return Some(rest);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hevc::Yuv420Frame;
    use crate::codec::vp9::encoder::encode_intra_frame;

    #[test]
    fn tile_col_range_splits_512_evenly() {
        // 512px → 64 MI cols → 8 SB cols → log2=1 → tiles [0,32) and [32,64).
        assert_eq!(choose_log2_tile_cols(512), 1);
        assert_eq!(tile_mi_col_range(64, 1, 0), (0, 32));
        assert_eq!(tile_mi_col_range(64, 1, 1), (32, 64));
        assert_eq!(choose_log2_tile_cols(64), 0);
        assert_eq!(choose_log2_tile_cols(80), 0);
    }

    #[test]
    fn pack_tile_data_prefixes_non_last() {
        let packed = pack_tile_data(&[vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(&packed[..4], &3u32.to_be_bytes());
        assert_eq!(&packed[4..], &[1, 2, 3, 4, 5]);
        assert_eq!(pack_tile_data(&[vec![9, 9]]), vec![9, 9]);
    }

    #[test]
    fn segmentation_alt_q_raises_right_half_qindex() {
        assert_eq!(segment_id_for_mi_col(0, 8), 0);
        assert_eq!(segment_id_for_mi_col(3, 8), 0);
        assert_eq!(segment_id_for_mi_col(4, 8), 1);
        assert_eq!(qindex_for_segment(0), QINDEX_USIZE);
        assert_eq!(
            qindex_for_segment(1),
            (QINDEX as i32 + SEG1_ALT_Q_DELTA) as usize
        );
    }

    #[test]
    fn compound_zeromv_recon_is_avg() {
        use crate::codec::vp9::encoder::encode_intra_frame;
        let w = 64u32;
        let h = 64u32;
        let mut key_src = Yuv420Frame::new(w, h);
        for j in 0..h {
            for i in 0..w {
                key_src.y[(j * w + i) as usize] =
                    ((i.wrapping_mul(7) + j.wrapping_mul(11)) & 0xFF) as u8;
            }
        }
        let (cw, ch) = (w / 2, h / 2);
        for j in 0..ch {
            for i in 0..cw {
                key_src.u[(j * cw + i) as usize] = 128;
                key_src.v[(j * cw + i) as usize] = 128;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key_src);
        let mut p1_src = Yuv420Frame::new(w, h);
        p1_src.u.copy_from_slice(&key_recon.u);
        p1_src.v.copy_from_slice(&key_recon.v);
        for j in 0..h as usize {
            for i in 0..w as usize {
                let sx = (i + 4).min(w as usize - 1);
                p1_src.y[j * w as usize + i] = key_recon.y[j * w as usize + sx];
            }
        }
        let (_, alt) = encode_inter_frame_refresh(&p1_src, &key_recon, 0x04);
        let mut p2_src = Yuv420Frame::new(w, h);
        for i in 0..key_recon.y.len() {
            p2_src.y[i] = ((u16::from(key_recon.y[i]) + u16::from(alt.y[i]) + 1) >> 1) as u8;
        }
        p2_src.u.copy_from_slice(&key_recon.u);
        p2_src.v.copy_from_slice(&key_recon.v);
        let (_, p2) = encode_inter_frame_compound(&p2_src, &key_recon, &key_recon, &alt);
        eprintln!(
            "key0={} alt0={} avg0={} recon0={}",
            key_recon.y[0], alt.y[0], p2_src.y[0], p2.y[0]
        );
        // Without LF, compound ZEROMV skip should equal avg. With LF, edges change.
        // Count how many mi are compound via a second encode peek — just check most pixels match avg before LF corners.
        let mut match_avg = 0usize;
        for i in 0..p2.y.len() {
            if p2.y[i] == p2_src.y[i] {
                match_avg += 1;
            }
        }
        eprintln!("pixels matching pre-LF avg src: {match_avg}/{}", p2.y.len());
        assert!(match_avg > p2.y.len() / 2);
    }

    #[test]
    fn zeromv_skip_recon_equals_reference() {
        let mut f = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                f.y[(j * 64 + i) as usize] = ((i + j * 3) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&f);
        let (bits, p_recon) = encode_inter_zeromv_skip(&key_recon);
        assert!(bits.len() > 12);
        // Post-LF recon: ZEROMV/skip still deblocks prediction edges.
        let mut expected = key_recon.clone();
        let lf_mi: Vec<LfMi> = (0..64)
            .map(|_| LfMi {
                skip: true,
                is_inter: true,
                tx_size_y: 1,
                bw_mi: 1,
                bh_mi: 1,
                bsize_idx: partition_ctx_index_wh(8, 8),
                block_origin: true,
                level: lf_level_for_mi(FILTER_LEVEL, LAST_FRAME, ZEROMV),
            })
            .collect();
        apply_loop_filter(&mut expected, &lf_mi, 8, 8);
        assert_eq!(p_recon.y, expected.y);
    }

    #[test]
    fn newmv_skip_shifts_recon() {
        let mut f = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                f.y[(j * 64 + i) as usize] = ((i * 3 + j) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&f);
        let (bits, p_recon) = encode_inter_newmv_skip(&key_recon, 0, 16);
        assert!(bits.len() > 12);
        let mut expected = motion_compensate(&key_recon, Mv { row: 0, col: 16 });
        let lf_mi: Vec<LfMi> = (0..64)
            .map(|_| LfMi {
                skip: true,
                is_inter: true,
                tx_size_y: 1,
                bw_mi: 1,
                bh_mi: 1,
                bsize_idx: partition_ctx_index_wh(8, 8),
                block_origin: true,
                level: lf_level_for_mi(FILTER_LEVEL, LAST_FRAME, NEWMV),
            })
            .collect();
        apply_loop_filter(&mut expected, &lf_mi, 8, 8);
        assert_eq!(p_recon.y, expected.y);
    }

    #[test]
    fn newmv_halfpel_skip_uses_filter() {
        let mut f = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                f.y[(j * 64 + i) as usize] = (i * 4) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&f);
        let (bits, p_recon) = encode_inter_newmv_skip(&key_recon, 0, 4); // half-pel
        assert!(bits.len() > 12);
        // Pre-LF half-pel samples sit between neighbors; LF may nudge edges.
        let mc = motion_compensate(&key_recon, Mv { row: 0, col: 4 });
        let mut between = 0usize;
        let mut total = 0usize;
        for j in 8..56usize {
            for i in 8..55usize {
                let a = key_recon.y[j * 64 + i];
                let b = key_recon.y[j * 64 + i + 1];
                let p = mc.y[j * 64 + i];
                if a != b {
                    total += 1;
                    if p > a.min(b) && p < a.max(b) {
                        between += 1;
                    }
                }
            }
        }
        assert!(
            between * 2 > total,
            "half-pel MC should interpolate: {between}/{total}"
        );
        assert_ne!(p_recon.y, key_recon.y);
    }

    #[test]
    fn residual_against_different_src_beats_skip_copy() {
        let mut key = Yuv420Frame::new(64, 64);
        key.y.fill(128);
        key.u.fill(128);
        key.v.fill(128);
        let (_, key_recon) = encode_intra_frame(&key);

        let mut src = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                src.y[(j * 64 + i) as usize] = ((i * 5 + j * 3) & 0xFF) as u8;
            }
        }
        let (bits, recon) = encode_inter_residual(&src, &key_recon);
        assert!(
            bits.len() > 50,
            "expected residual bitstream, got {}",
            bits.len()
        );
        let mut err = 0u64;
        for (a, b) in src.y.iter().zip(&recon.y) {
            err += a.abs_diff(*b) as u64;
        }
        let copy_err: u64 = src.y.iter().map(|&x| x.abs_diff(128) as u64).sum();
        assert!(
            err < copy_err / 2,
            "residual err={err} should beat copy={copy_err}"
        );
    }

    #[test]
    fn newmv_residual_beats_zeromv_on_shifted_src() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);

        // Source ≈ reference shifted by +2 horizontally.
        let mut src = Yuv420Frame::new(64, 64);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        for j in 0..32usize {
            for i in 0..32usize {
                let sx = i.saturating_add(1).min(31);
                src.u[j * 32 + i] = key_recon.u[j * 32 + sx];
                src.v[j * 32 + i] = key_recon.v[j * 32 + sx];
            }
        }

        let (_, z_recon) = encode_inter_residual(&src, &key_recon);
        let (bits, n_recon) = encode_inter_newmv_residual(&src, &key_recon, 0, 16);
        assert!(bits.len() > 20);

        let z_err: u64 = src
            .y
            .iter()
            .zip(&z_recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        let n_err: u64 = src
            .y
            .iter()
            .zip(&n_recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(
            n_err <= z_err,
            "NEWMV residual err={n_err} should beat ZEROMV err={z_err}"
        );
        assert!(
            n_err < 5000,
            "shifted content should nearly skip: err={n_err}"
        );
    }

    #[test]
    fn me_frame_finds_mixed_block_mvs() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);

        // Left half shifted +2 pel, right half unshifted.
        let mut src = Yuv420Frame::new(64, 64);
        src.u.copy_from_slice(&key_recon.u);
        src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = if i < 32 {
                    i.saturating_add(2).min(63)
                } else {
                    i
                };
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }

        let (_, z_recon) = encode_inter_residual(&src, &key_recon);
        let (bits, me_recon) = encode_inter_frame(&src, &key_recon);
        assert!(bits.len() > 12, "P-frame too small: {}", bits.len());

        let z_err: u64 = src
            .y
            .iter()
            .zip(&z_recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        let me_err: u64 = src
            .y
            .iter()
            .zip(&me_recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(
            me_err < z_err / 2,
            "ME residual err={me_err} should beat global ZEROMV err={z_err}"
        );
        assert!(
            me_err < 800,
            "mixed-shift content should nearly skip: err={me_err}"
        );
    }

    #[test]
    fn uniform_shift_uses_nearestmv() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);
        let mut src = Yuv420Frame::new(64, 64);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        for j in 0..32usize {
            for i in 0..32usize {
                let sx = i.saturating_add(1).min(31);
                src.u[j * 32 + i] = key_recon.u[j * 32 + sx];
                src.v[j * 32 + i] = key_recon.v[j * 32 + sx];
            }
        }

        let (new_bits, _) = encode_inter_newmv_residual(&src, &key_recon, 0, 16);
        let (me_bits, me_recon) = encode_inter_frame(&src, &key_recon);
        // Predicted modes should beat coding NEWMV on every block.
        assert!(
            me_bits.len() < new_bits.len(),
            "NEARESTMV path {} bytes should beat all-NEWMV {} bytes",
            me_bits.len(),
            new_bits.len()
        );
        let err: u64 = src
            .y
            .iter()
            .zip(&me_recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(err < 500, "uniform shift should nearly skip: err={err}");
    }

    #[test]
    fn uniform_shift_prefers_16x16_partition() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);
        let mut src = Yuv420Frame::new(64, 64);
        src.u.copy_from_slice(&key_recon.u);
        src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        let (bits, recon) = encode_inter_frame(&src, &key_recon);
        // With 16×16 NONE, a near-skip uniform P-frame should stay compact.
        assert!(
            bits.len() < 120,
            "expected compact 16×16-friendly bitstream, got {}",
            bits.len()
        );
        let err: u64 = src
            .y
            .iter()
            .zip(&recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(err < 500, "err={err}");
    }

    #[test]
    fn uniform_shift_prefers_32x32_partition() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);
        let mut src = Yuv420Frame::new(64, 64);
        src.u.copy_from_slice(&key_recon.u);
        src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        let (bits, recon) = encode_inter_frame(&src, &key_recon);
        // One/two 32×32 NONE blocks should stay very compact on uniform motion.
        assert!(
            bits.len() < 80,
            "expected compact 32×32-friendly bitstream, got {}",
            bits.len()
        );
        let err: u64 = src
            .y
            .iter()
            .zip(&recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(err < 500, "err={err}");
    }

    #[test]
    fn uniform_shift_prefers_64x64_partition() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);
        let mut src = Yuv420Frame::new(64, 64);
        src.u.copy_from_slice(&key_recon.u);
        src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        let (bits, recon) = encode_inter_frame(&src, &key_recon);
        // A single 64×64 NONE + NEWMV/skip should be tiny.
        assert!(
            bits.len() < 50,
            "expected compact 64×64 NONE bitstream, got {}",
            bits.len()
        );
        let err: u64 = src
            .y
            .iter()
            .zip(&recon.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(err < 500, "err={err}");
    }

    #[test]
    fn golden_beats_last_when_src_matches_key() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);

        // P1: shift content → LAST becomes shifted recon.
        let mut p1_src = Yuv420Frame::new(64, 64);
        p1_src.u.copy_from_slice(&key_recon.u);
        p1_src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                p1_src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        let (_, last) = encode_inter_frame(&p1_src, &key_recon);

        // P2: source ≈ key again → GOLDEN (key) should beat LAST (shifted).
        let mut p2_src = Yuv420Frame::new(64, 64);
        p2_src.y.copy_from_slice(&key_recon.y);
        p2_src.u.copy_from_slice(&key_recon.u);
        p2_src.v.copy_from_slice(&key_recon.v);

        let (_, only_last) = encode_inter_frame(&p2_src, &last);
        let (bits, with_golden) = encode_inter_frame_golden(&p2_src, &last, &key_recon);
        assert!(bits.len() > 12);

        let last_err: u64 = p2_src
            .y
            .iter()
            .zip(&only_last.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        let gold_err: u64 = p2_src
            .y
            .iter()
            .zip(&with_golden.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(
            gold_err <= last_err,
            "golden err={gold_err} should beat last-only err={last_err}"
        );
        assert!(
            gold_err < 500,
            "key-matched src should nearly skip via GOLDEN: err={gold_err}"
        );
    }

    #[test]
    fn altref_beats_last_when_src_matches_alt() {
        let mut key = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                key.y[(j * 64 + i) as usize] = ((i * 3 + j * 5) & 0xFF) as u8;
            }
        }
        let (_, key_recon) = encode_intra_frame(&key);

        // P1 into ALTREF only (0x04): LAST/GOLDEN stay at key, ALTREF = shift.
        let mut p1_src = Yuv420Frame::new(64, 64);
        p1_src.u.copy_from_slice(&key_recon.u);
        p1_src.v.copy_from_slice(&key_recon.v);
        for j in 0..64usize {
            for i in 0..64usize {
                let sx = i.saturating_add(2).min(63);
                p1_src.y[j * 64 + i] = key_recon.y[j * 64 + sx];
            }
        }
        let (_, alt) = encode_inter_frame_refresh(&p1_src, &key_recon, 0x04);

        // P2: source ≈ alt → ALTREF should beat LAST (= key).
        let mut p2_src = Yuv420Frame::new(64, 64);
        p2_src.y.copy_from_slice(&alt.y);
        p2_src.u.copy_from_slice(&alt.u);
        p2_src.v.copy_from_slice(&alt.v);

        let (_, only_last) = encode_inter_frame(&p2_src, &key_recon);
        let (bits, with_alt) = encode_inter_frame_altref(&p2_src, &key_recon, &key_recon, &alt);
        assert!(bits.len() > 12);

        let last_err: u64 = p2_src
            .y
            .iter()
            .zip(&only_last.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        let alt_err: u64 = p2_src
            .y
            .iter()
            .zip(&with_alt.y)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum();
        assert!(
            alt_err <= last_err,
            "altref err={alt_err} should beat last-only err={last_err}"
        );
        assert!(
            alt_err < 500,
            "alt-matched src should nearly skip via ALTREF: err={alt_err}"
        );
    }

    #[test]
    fn nmv_update_count_no_hp() {
        assert_eq!(nmv_update_count(false), 65);
        assert_eq!(nmv_update_count(true), 69);
    }

    #[test]
    fn inter_mode_tree_paths() {
        assert_eq!(
            tree_path(&INTER_MODE_TREE, 0, INTER_OFFSET_ZEROMV).unwrap(),
            vec![(0, false)]
        );
        assert_eq!(
            tree_path(&INTER_MODE_TREE, 0, INTER_OFFSET_NEARESTMV).unwrap(),
            vec![(0, true), (2, false)]
        );
        assert_eq!(
            tree_path(&INTER_MODE_TREE, 0, INTER_OFFSET_NEARMV).unwrap(),
            vec![(0, true), (2, true), (4, false)]
        );
        assert_eq!(
            tree_path(&INTER_MODE_TREE, 0, INTER_OFFSET_NEWMV).unwrap(),
            vec![(0, true), (2, true), (4, true)]
        );
    }

    #[test]
    fn pick_mode_prefers_predicted() {
        let nearest = Mv { row: 0, col: 16 };
        let near = Mv { row: 8, col: 0 };
        assert_eq!(pick_inter_mode(Mv::default(), nearest, near).1, ZEROMV);
        assert_eq!(pick_inter_mode(nearest, nearest, near).1, NEARESTMV);
        assert_eq!(pick_inter_mode(near, nearest, near).1, NEARMV);
        assert_eq!(
            pick_inter_mode(Mv { row: 0, col: 24 }, nearest, near).1,
            NEWMV
        );
    }
}
