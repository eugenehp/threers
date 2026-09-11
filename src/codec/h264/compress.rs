//! Compressed intra H.264: `I_NxN` macroblocks with CAVLC residuals.
//!
//! This is the replacement for the `I_PCM` path in [`super::slice`], which codes
//! raw samples and so produces files the size of the source. Here every 4×4 luma
//! block picks one of nine intra modes, the residual goes through the core
//! transform and quantizer, and the coefficients are CAVLC-coded.
//!
//! Prediction is per 4×4 block from already-reconstructed neighbours, so the
//! reconstruction has to happen *during* coding, in decoding order — the same
//! constraint the HEVC path has, for the same reason.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};
use crate::codec::h264::nal::{nal_unit_base, push_annexb, NalUnitType};
use crate::codec::h264::params::write_sps;
use crate::codec::h264::avcc::build_avcc;
use crate::codec::h264::encoder::H264Encoder;
use crate::codec::h264::pps::write_pps_deblock_control;
use crate::codec::h264::params::EntropyMode;
use crate::codec::mp4::H264Mp4Params;
use crate::codec::rate::{Quality, Reach, RateControl, PROBE_QP};
use crate::captions::CaptionTrack;
use crate::codec::hevc::encoder::{pad_plane, Yuv420Frame};
use crate::codec::h264::cabac::{self, Cabac};
use crate::codec::h264::cavlc::{self, BlockKind, ZIGZAG4};
use crate::codec::h264::intra;
use crate::codec::h264::params::{H264Config, MB_SIZE};
use crate::codec::h264::transform::{
    chroma_dc_forward, chroma_dc_inverse, dequant4, fit_luma_dc_levels, forward4, inverse4,
    luma_dc_hadamard, luma_dc_inverse, quant4,
};

/// `mb_type` for `I_NxN` in an I slice (Table 7-11).
const MB_TYPE_I_NXN: u32 = 0;

/// Position of each 4×4 luma block within the macroblock, in 4-sample units,
/// in the order the bitstream visits them (Table 6-3): z-order over the four
/// 8×8 quadrants, and z-order again inside each.
const BLK_POS: [(usize, usize); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 2), (3, 2), (2, 3), (3, 3),
];

/// Inverse of [`BLK_POS`]: `(x4, y4)` within a macroblock → block index.
fn blk_index(x4: usize, y4: usize) -> usize {
    BLK_POS.iter().position(|&p| p == (x4, y4)).expect("in-range")
}

/// `coded_block_pattern` code numbers for intra macroblocks (Table 9-4).
/// Index is the CBP value; the entry is the `me(v)` code number.
const CBP_INTRA_CODE: [u32; 48] = {
    // Table 9-4 lists codeNum → CBP; invert it once at compile time.
    const FROM_CODE: [usize; 48] = [
        47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19,
        21, 26, 28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36,
        40, 38, 41,
    ];
    let mut out = [0u32; 48];
    let mut i = 0;
    while i < 48 {
        out[FROM_CODE[i]] = i as u32;
        i += 1;
    }
    out
};

/// Chroma QP from luma QP (Table 8-15) with `chroma_qp_index_offset = 0`.
fn chroma_qp(qp: i32) -> i32 {
    const T: [i32; 22] = [
        29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ];
    let qpi = qp.clamp(0, 51);
    if qpi < 30 {
        qpi
    } else {
        T[(qpi - 30) as usize]
    }
}

/// A reconstructed picture the decoder must reproduce.
pub struct Reconstruction {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub coded_width: u32,
    pub coded_height: u32,
}

/// Per-picture state the neighbour derivations read.
struct Ctx {
    /// Reconstructed planes at coded (MB-aligned) size.
    ry: Vec<u8>,
    ru: Vec<u8>,
    rv: Vec<u8>,
    cw: u32,
    ch: u32,
    cwc: u32,
    chc: u32,
    mbw: usize,
    /// Non-zero coefficient counts per 4×4 luma block, indexed by picture 4×4
    /// grid — this is what `nC` is derived from.
    nnz_y: Vec<usize>,
    /// Same for the two chroma planes, on the half-resolution 4×4 grid.
    nnz_c: [Vec<usize>; 2],
    /// Intra 4×4 prediction mode per luma block, for the MPM derivation.
    /// `None` where no intra block has been coded.
    modes: Vec<Option<u8>>,
    /// Per-macroblock summary the CABAC context derivations read. Unused by the
    /// CAVLC path, which needs only coefficient counts.
    mb: Vec<MbNeighbor>,
    /// Whether the previous macroblock in decoding order sent a non-zero
    /// `mb_qp_delta` — the only thing that element's first bin's context asks.
    prev_qp_delta_nonzero: bool,
}

impl Ctx {
    /// Whether the sample at `(px, py)` is already reconstructed when coding the
    /// block at `(bx, by)`.
    ///
    /// Macroblocks go in raster order and 4×4 blocks in z-order inside them, so
    /// comparing `(macroblock index, block index)` lexicographically *is*
    /// decoding order. This is what makes the above-right neighbour come out
    /// right: available from the macroblock above-right, unavailable from a
    /// later block in this one.
    fn avail_luma(&self, px: i32, py: i32, bx: u32, by: u32) -> bool {
        if px < 0 || py < 0 || px >= self.cw as i32 || py >= self.ch as i32 {
            return false;
        }
        self.order_luma(px as u32, py as u32) < self.order_luma(bx, by)
    }

    fn order_luma(&self, px: u32, py: u32) -> (usize, usize) {
        let (mbx, mby) = ((px / MB_SIZE) as usize, (py / MB_SIZE) as usize);
        let (x4, y4) = (((px % MB_SIZE) / 4) as usize, ((py % MB_SIZE) / 4) as usize);
        (mby * self.mbw + mbx, blk_index(x4, y4))
    }

    /// Chroma availability is macroblock-granular: the whole 8×8 is predicted at
    /// once, so only whole neighbouring macroblocks matter.
    fn avail_chroma(&self, px: i32, py: i32, mbx: usize, mby: usize) -> bool {
        if px < 0 || py < 0 || px >= self.cwc as i32 || py >= self.chc as i32 {
            return false;
        }
        let (nx, ny) = ((px as u32 / 8) as usize, (py as u32 / 8) as usize);
        ny * self.mbw + nx < mby * self.mbw + mbx
    }
}

/// Encode one picture as an IDR I-slice of `I_NxN` macroblocks.
pub fn compressed_slice_rbsp(
    cfg: &H264Config,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    qp: i32,
) -> (Vec<u8>, Reconstruction) {
    let (cw, ch) = (cfg.coded_width, cfg.coded_height);
    let (cwc, chc) = (cw / 2, ch / 2);
    let mbw = (cw / MB_SIZE) as usize;
    let mbh = (ch / MB_SIZE) as usize;
    let mb_qp = adaptive_qp(y, cw, mbw, mbh, qp, cfg.aq_strength);

    let mut w = BitWriter::new();
    let mut header: Vec<u8> = Vec::new();
    // ---- slice_header ----
    w.write_ue(0); // first_mb_in_slice
    w.write_ue(7); // slice_type 7 → I (all slices in the picture are I)
    w.write_ue(0); // pic_parameter_set_id
    w.write_bits(0, 4); // frame_num
    w.write_ue(0); // idr_pic_id
    w.flag(false); // no_output_of_prior_pics_flag
    w.flag(false); // long_term_reference_flag
    w.write_se(qp - 26); // slice_qp_delta (pic_init_qp_minus26 = 0)
    if cfg.deblock {
        w.write_ue(0); // disable_deblocking_filter_idc — filter on
        w.write_se(0); // slice_alpha_c0_offset_div2
        w.write_se(0); // slice_beta_offset_div2
    } else {
        w.write_ue(1); // disable_deblocking_filter_idc — filter off
    }

    // CABAC starts byte-aligned, and the padding is ones so a decoder that has
    // mis-parsed the header runs into a syntax error rather than silently
    // decoding noise (§7.3.4).
    let mut entropy = match cfg.entropy {
        EntropyMode::Cavlc => Entropy::Cavlc(w),
        EntropyMode::Cabac => {
            while !w.is_byte_aligned() {
                w.write_bit(1); // cabac_alignment_one_bit
            }
            header = w.finish();
            Entropy::Cabac(Box::new(Cabac::new(qp)))
        }
    };

    let mut ctx = Ctx {
        ry: vec![128; (cw * ch) as usize],
        ru: vec![128; (cwc * chc) as usize],
        rv: vec![128; (cwc * chc) as usize],
        cw,
        ch,
        cwc,
        chc,
        mbw,
        nnz_y: vec![0; (cw / 4 * (ch / 4)) as usize],
        nnz_c: [
            vec![0; (cwc / 4 * (chc / 4)) as usize],
            vec![0; (cwc / 4 * (chc / 4)) as usize],
        ],
        modes: vec![None; (cw / 4 * (ch / 4)) as usize],
        mb: vec![MbNeighbor::default(); mbw * mbh],
        prev_qp_delta_nonzero: false,
    };

    let last = mbw * mbh - 1;
    // `QP_Y,PRED` — the QP of the previous macroblock in decoding order, which
    // is what `mb_qp_delta` is relative to (§7.4.5). It only moves when a
    // macroblock actually sends a delta.
    let mut qp_prev = qp;
    // The QP a decoder ends up with per macroblock, which is what the
    // deblocking thresholds are derived from — not the QP the encoder
    // quantized with, since a macroblock that codes nothing sends no delta.
    let mut decoded_qp = vec![qp; mbw * mbh];
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let qp_mb = mb_qp[mby * mbw + mbx];
            let sent = encode_macroblock(
                &mut entropy,
                &mut ctx,
                y,
                u,
                v,
                mbx,
                mby,
                qp_mb,
                chroma_qp(qp_mb),
                qp_mb - qp_prev,
            );
            if sent {
                qp_prev = qp_mb;
            }
            decoded_qp[mby * mbw + mbx] = qp_prev;
            if let Entropy::Cabac(c) = &mut entropy {
                c.terminate(u32::from(mby * mbw + mbx == last));
            }
        }
    }

    let rbsp = match entropy {
        Entropy::Cavlc(mut w) => {
            rbsp_trailing_bits(&mut w);
            w.finish()
        }
        // The flush's final bit is the `rbsp_stop_one_bit`, so the slice needs
        // no trailing bits of its own (§9.3.4.3.5).
        Entropy::Cabac(c) => {
            let mut out = header;
            out.extend_from_slice(&c.finish());
            out
        }
    };
    // Intra prediction reads unfiltered samples, so the filter runs once over
    // the finished picture rather than inside the reconstruction loop.
    let (mut ry, mut ru, mut rv) = (ctx.ry, ctx.ru, ctx.rv);
    if cfg.deblock {
        crate::codec::h264::deblock::deblock(
            &mut ry,
            &mut ru,
            &mut rv,
            cw,
            ch,
            &decoded_qp,
            chroma_qp,
        );
    }
    (
        rbsp,
        Reconstruction {
            y: ry,
            u: ru,
            v: rv,
            coded_width: cw,
            coded_height: ch,
        },
    )
}

/// The two intra macroblock shapes this encoder can emit.
///
/// `I_NxN` predicts each 4×4 block from its immediate neighbours, which suits
/// detail. `I_16x16` predicts the whole macroblock at once and sends the sixteen
/// block DCs through a second Hadamard, which suits flat areas and gradients:
/// there it costs a couple of dozen bits where `I_NxN` pays sixteen mode
/// signals plus sixteen `coeff_token`s just to say "nothing here".
enum LumaCoding {
    NxN(NxNLuma),
    I16(I16Luma),
}

struct NxNLuma {
    modes: [u8; 16],
    /// Zig-zag scanned levels per 4×4 block, in coding order.
    levels: [[i32; 16]; 16],
    cbp_luma: u32,
}

struct I16Luma {
    pred_mode: u8,
    /// Zig-zag scanned Hadamard DC levels (`Intra16x16DCLevel`), always coded.
    dc: [i32; 16],
    /// The 15 AC levels of each 4×4 block, in coding order.
    ac: [[i32; 15]; 16],
    /// 0 or 15 — `I_16x16` codes every 8×8 or none of them.
    cbp_luma: u32,
}

/// Chroma is predicted and coded the same way whichever luma shape wins, so it
/// is built once and shared between the candidates.
struct ChromaData {
    mode: u8,
    /// Quantized 2×2 chroma DC per plane, in the raster order the block uses.
    dc: [[i32; 4]; 2],
    /// 15 AC levels per 4×4 chroma block, per plane.
    ac: [[[i32; 15]; 4]; 2],
    cbp: u32,
}

/// A luma candidate, with the distortion it would leave behind.
struct Candidate {
    coding: LumaCoding,
    ssd: u64,
}

/// The macroblock's share of the state the *next* macroblock reads, so a
/// candidate can be built into `ctx`, measured, and then undone.
struct LumaState {
    rec: [u8; 256],
    nnz: [usize; 16],
    modes: [Option<u8>; 16],
}

#[allow(clippy::too_many_arguments)]
fn encode_macroblock(
    w: &mut Entropy,
    ctx: &mut Ctx,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    mbx: usize,
    mby: usize,
    qp: i32,
    qp_c: i32,
    qp_delta: i32,
) -> bool {
    // Chroma first: it depends only on neighbouring macroblocks, never on this
    // one's luma, so both candidates can be scored against the same chroma.
    let chroma = encode_chroma(ctx, u, v, mbx, mby, qp_c);

    let nxn = build_nxn(ctx, y, mbx, mby, qp);
    let nxn_state = save_luma(ctx, mbx, mby);
    let nxn_cost = rd_cost(nxn.ssd, trial_bits(w, ctx, &nxn.coding, &chroma, mbx, mby, qp_delta), qp);

    // `I_16x16` predicts from outside the macroblock only, so it does not care
    // that `build_nxn` has already overwritten the inside.
    let i16 = build_i16(ctx, y, mbx, mby, qp);
    let i16_cost = rd_cost(i16.ssd, trial_bits(w, ctx, &i16.coding, &chroma, mbx, mby, qp_delta), qp);

    let winner = if i16_cost < nxn_cost {
        &i16.coding
    } else {
        restore_luma(ctx, mbx, mby, &nxn_state);
        &nxn.coding
    };
    w.write_mb(ctx, winner, &chroma, mbx, mby, qp_delta);

    // Record what the next macroblocks' CABAC contexts will ask about this one.
    ctx.mb[mby * ctx.mbw + mbx] = MbNeighbor {
        avail: true,
        i16: matches!(winner, LumaCoding::I16(_)),
        cbp_luma: match winner {
            LumaCoding::NxN(l) => l.cbp_luma,
            LumaCoding::I16(l) => l.cbp_luma,
        },
        cbp_chroma: chroma.cbp,
        chroma_mode: chroma.mode,
        dc_cbf: match winner {
            LumaCoding::I16(l) => l.dc.iter().any(|&v| v != 0),
            LumaCoding::NxN(_) => false,
        },
        chroma_dc_cbf: [
            chroma.dc[0].iter().any(|&v| v != 0),
            chroma.dc[1].iter().any(|&v| v != 0),
        ],
    };

    // A macroblock with nothing to code carries no `mb_qp_delta`, so the
    // predictor does not move past it — and neither does the reconstruction,
    // since there is nothing to dequantize.
    let sent = sends_qp_delta(winner, &chroma);
    ctx.prev_qp_delta_nonzero = sent && qp_delta != 0;
    sent
}

/// Whether this macroblock carries `mb_qp_delta` (§7.3.5).
fn sends_qp_delta(luma: &LumaCoding, chroma: &ChromaData) -> bool {
    match luma {
        LumaCoding::I16(_) => true,
        LumaCoding::NxN(l) => l.cbp_luma != 0 || chroma.cbp != 0,
    }
}

/// Lagrangian cost. λ is the usual `0.85 · 2^((QP−12)/3)` for an SSD metric —
/// it doubles every three QP steps, which is how fast the quantizer's own
/// distortion grows.
fn rd_cost(ssd: u64, bits: usize, qp: i32) -> f64 {
    let lambda = 0.85 * (f64::from(qp - 12) / 3.0).exp2();
    ssd as f64 + lambda * bits as f64
}

/// Exact size of the macroblock as it would be written, in bits.
///
/// CAVLC carries no adaptive state between macroblocks — `nC` comes from
/// neighbour coefficient counts, which `ctx` already holds — so a trial encode
/// is not an estimate. It is the number of bits this choice actually costs.
#[allow(clippy::too_many_arguments)]
fn trial_bits(
    w: &Entropy,
    ctx: &Ctx,
    luma: &LumaCoding,
    chroma: &ChromaData,
    mbx: usize,
    mby: usize,
    qp_delta: i32,
) -> usize {
    let mut t = w.probe();
    t.write_mb(ctx, luma, chroma, mbx, mby, qp_delta);
    t.bit_len()
}

fn save_luma(ctx: &Ctx, mbx: usize, mby: usize) -> LumaState {
    let (ox, oy) = ((mbx * MB_SIZE as usize) as u32, (mby * MB_SIZE as usize) as u32);
    let mut rec = [0u8; 256];
    for j in 0..16u32 {
        let row = ((oy + j) * ctx.cw + ox) as usize;
        let dst = (j * 16) as usize;
        rec[dst..dst + 16].copy_from_slice(&ctx.ry[row..row + 16]);
    }
    let mut nnz = [0usize; 16];
    let mut modes = [None; 16];
    for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
        let gi = grid_index(ctx.cw, ox + bx4 as u32 * 4, oy + by4 as u32 * 4);
        nnz[blk] = ctx.nnz_y[gi];
        modes[blk] = ctx.modes[gi];
    }
    LumaState { rec, nnz, modes }
}

fn restore_luma(ctx: &mut Ctx, mbx: usize, mby: usize, s: &LumaState) {
    let (ox, oy) = ((mbx * MB_SIZE as usize) as u32, (mby * MB_SIZE as usize) as u32);
    for j in 0..16u32 {
        let row = ((oy + j) * ctx.cw + ox) as usize;
        let src = (j * 16) as usize;
        ctx.ry[row..row + 16].copy_from_slice(&s.rec[src..src + 16]);
    }
    for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
        let gi = grid_index(ctx.cw, ox + bx4 as u32 * 4, oy + by4 as u32 * 4);
        ctx.nnz_y[gi] = s.nnz[blk];
        ctx.modes[gi] = s.modes[blk];
    }
}

/// Build the `I_NxN` candidate, reconstructing into `ctx` as it goes — each
/// block predicts from the one before it, so there is no other order to do it in.
fn build_nxn(ctx: &mut Ctx, y: &[u8], mbx: usize, mby: usize, qp: i32) -> Candidate {
    let mut l = NxNLuma {
        modes: [intra::DC; 16],
        levels: [[0; 16]; 16],
        cbp_luma: 0,
    };
    let mut ssd = 0u64;

    for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
        let px = (mbx * MB_SIZE as usize + bx4 * 4) as u32;
        let py = (mby * MB_SIZE as usize + by4 * 4) as u32;

        let n = luma_neighbors(ctx, px, py);
        let orig = gather(y, ctx.cw, px, py, 4);
        let mode = best_luma_mode(&orig, &n);
        let pred = intra::predict4(mode, &n);

        let mut resid = [0i32; 16];
        for i in 0..16 {
            resid[i] = orig[i] - pred[i];
        }
        let levels = quant4(&forward4(&resid), qp, true);

        // Reconstruct with exactly what the decoder will see.
        let res = inverse4(&dequant4(&levels, qp));
        let mut rec = [0i32; 16];
        for i in 0..16 {
            rec[i] = (pred[i] + res[i]).clamp(0, 255);
            ssd += ((orig[i] - rec[i]) * (orig[i] - rec[i])) as u64;
        }
        store(&mut ctx.ry, ctx.cw, px, py, 4, &rec);

        // Coefficients travel in zig-zag scan order.
        let mut scan = [0i32; 16];
        for (s, &r) in ZIGZAG4.iter().enumerate() {
            scan[s] = levels[r];
        }
        let nnz = scan.iter().filter(|&&c| c != 0).count();
        l.modes[blk] = mode;
        l.levels[blk] = scan;
        if nnz > 0 {
            l.cbp_luma |= 1 << (blk / 4);
        }

        let gi = grid_index(ctx.cw, px, py);
        ctx.nnz_y[gi] = nnz;
        ctx.modes[gi] = Some(mode);
    }

    Candidate {
        coding: LumaCoding::NxN(l),
        ssd,
    }
}

/// Build the `I_16x16` candidate: one prediction for all 256 samples, the
/// sixteen block DCs through a 4×4 Hadamard, and the AC left in place.
fn build_i16(ctx: &mut Ctx, y: &[u8], mbx: usize, mby: usize, qp: i32) -> Candidate {
    let (ox, oy) = ((mbx * MB_SIZE as usize) as u32, (mby * MB_SIZE as usize) as u32);
    let n = luma16_neighbors(ctx, mbx, mby);
    let orig = gather(y, ctx.cw, ox, oy, 16);
    let pred_mode = best_luma16_mode(&orig, &n);
    let pred = intra::predict16(pred_mode, &n);

    // Forward transform every 4×4 block, then lift the DCs out into a raster
    // 4×4 of their own — block `blk` sits at row `by4`, column `bx4`.
    let mut coeff = [[0i32; 16]; 16];
    let mut dc = [0i32; 16];
    for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
        let mut r = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                let k = (by4 * 4 + j) * 16 + bx4 * 4 + i;
                r[j * 4 + i] = orig[k] - pred[k];
            }
        }
        coeff[blk] = forward4(&r);
        dc[by4 * 4 + bx4] = coeff[blk][0];
    }

    // The Hadamard scales by 4 in each direction and the decoder's inverse
    // undoes only a quarter of that (§8.5.10 shifts by 6, not 4), so the
    // encoder's quantizer runs two bits coarser than for AC.
    let hdc = luma_dc_hadamard(&dc);
    let mf = quant_mf0((qp % 6) as usize) as i64;
    let qbits = 15 + qp / 6 + 2;
    let f = (1i64 << qbits) / 3;
    let mut dc_levels = [0i32; 16];
    for k in 0..16 {
        let l = ((hdc[k].abs() as i64) * mf + f) >> qbits;
        dc_levels[k] = if hdc[k] < 0 { -(l as i32) } else { l as i32 };
    }
    fit_luma_dc_levels(&mut dc_levels, qp);
    let dc_rec = luma_dc_inverse(&dc_levels, qp);

    let mut l = I16Luma {
        pred_mode,
        dc: [0; 16],
        ac: [[0; 15]; 16],
        cbp_luma: 0,
    };
    for (s, &r) in ZIGZAG4.iter().enumerate() {
        l.dc[s] = dc_levels[r];
    }

    let mut ssd = 0u64;
    for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
        let q = quant4(&coeff[blk], qp, true);
        let mut scan = [0i32; 15];
        for s in 1..16 {
            scan[s - 1] = q[ZIGZAG4[s]];
        }
        let nnz = scan.iter().filter(|&&c| c != 0).count();
        l.ac[blk] = scan;
        if nnz > 0 {
            l.cbp_luma = 15;
        }

        // Dequantized AC with the Hadamard's DC dropped in at position 0 — the
        // DC never goes through the ordinary dequantizer.
        let mut deq = dequant4(&q, qp);
        deq[0] = dc_rec[by4 * 4 + bx4];
        let res = inverse4(&deq);
        let mut rec = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                let k = (by4 * 4 + j) * 16 + bx4 * 4 + i;
                rec[j * 4 + i] = (pred[k] + res[j * 4 + i]).clamp(0, 255);
                let e = orig[k] - rec[j * 4 + i];
                ssd += (e * e) as u64;
            }
        }
        let (px, py) = (ox + bx4 as u32 * 4, oy + by4 as u32 * 4);
        store(&mut ctx.ry, ctx.cw, px, py, 4, &rec);

        let gi = grid_index(ctx.cw, px, py);
        ctx.nnz_y[gi] = nnz;
        // Not an `Intra_4x4` block: neighbours predicting against it must treat
        // its mode as DC (§8.3.1.1), which is what `None` means here.
        ctx.modes[gi] = None;
    }

    Candidate {
        coding: LumaCoding::I16(l),
        ssd,
    }
}

/// Predict, transform, quantize and reconstruct both chroma planes.
fn encode_chroma(
    ctx: &mut Ctx,
    u: &[u8],
    v: &[u8],
    mbx: usize,
    mby: usize,
    qp_c: i32,
) -> ChromaData {
    let cnb = chroma_neighbors(ctx, mbx, mby);
    let mut d = ChromaData {
        mode: best_chroma_mode(ctx, u, v, mbx, mby, &cnb),
        dc: [[0; 4]; 2],
        ac: [[[0; 15]; 4]; 2],
        cbp: 0,
    };

    for (plane, src) in [(0usize, u), (1usize, v)] {
        let cn = chroma_neighbors_plane(ctx, mbx, mby, plane);
        let pred = intra::predict_chroma(d.mode, &cn);
        let (cx, cy) = ((mbx * 8) as u32, (mby * 8) as u32);
        let orig = gather(src, ctx.cwc, cx, cy, 8);

        // Transform each 4×4, split DC from AC.
        let mut dc = [0i32; 4];
        let mut ac_coeff = [[0i32; 16]; 4];
        for b in 0..4 {
            let (sx, sy) = ((b % 2) * 4, (b / 2) * 4);
            let mut r = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let k = (sy + j) * 8 + sx + i;
                    r[j * 4 + i] = orig[k] - pred[k];
                }
            }
            let c = forward4(&r);
            dc[b] = c[0];
            ac_coeff[b] = c;
        }

        // DC goes through the 2×2 Hadamard; its quantizer runs one bit coarser
        // because the Hadamard doubles the scale in each dimension.
        let hdc = chroma_dc_forward(&dc);
        let m = (qp_c % 6) as usize;
        let qbits = 15 + qp_c / 6 + 1;
        let f = (1i64 << qbits) / 3;
        let mut dc_levels = [0i32; 4];
        for b in 0..4 {
            let mf = quant_mf0(m) as i64;
            let l = ((hdc[b].abs() as i64) * mf + f) >> qbits;
            dc_levels[b] = if hdc[b] < 0 { -(l as i32) } else { l as i32 };
        }
        crate::codec::h264::transform::fit_chroma_dc_levels(&mut dc_levels, qp_c);
        let dc_rec = chroma_dc_inverse(&dc_levels, qp_c);

        let mut any_ac = false;
        for b in 0..4 {
            let q = quant4(&ac_coeff[b], qp_c, true);
            let mut scan = [0i32; 15];
            for s in 1..16 {
                scan[s - 1] = q[ZIGZAG4[s]];
            }
            let nnz = scan.iter().filter(|&&c| c != 0).count();
            d.ac[plane][b] = scan;
            any_ac |= nnz > 0;

            // Reconstruct: dequantized AC with the Hadamard DC dropped in at
            // position 0, which is why the DC is not dequantized twice.
            let mut deq = dequant4(&q, qp_c);
            deq[0] = dc_rec[b];
            let res = inverse4(&deq);
            let (sx, sy) = ((b % 2) * 4, (b / 2) * 4);
            let mut rec = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let k = (sy + j) * 8 + sx + i;
                    rec[j * 4 + i] = (pred[k] + res[j * 4 + i]).clamp(0, 255);
                }
            }
            let rp = if plane == 0 { &mut ctx.ru } else { &mut ctx.rv };
            store(rp, ctx.cwc, cx + sx as u32, cy + sy as u32, 4, &rec);

            let gi = grid_index(ctx.cwc, cx + sx as u32, cy + sy as u32);
            ctx.nnz_c[plane][gi] = nnz;
        }

        d.dc[plane] = dc_levels;
        if any_ac {
            d.cbp = 2;
        } else if dc_levels.iter().any(|&c| c != 0) && d.cbp < 1 {
            d.cbp = 1;
        }
    }
    d
}

/// Per-macroblock quantizer offsets from local variance (adaptive quantization).
///
/// A flat macroblock and a busy one at the same QP do not look equally good:
/// quantization error is far more visible against a smooth background than
/// against texture, which masks it. Adaptive quantization spends the bits where
/// the eye can see them — lower QP on flat areas, higher on detail.
///
/// The offsets are centred on the picture's own mean log-variance, so the
/// average QP is unchanged and the file size stays roughly where it was. This
/// redistributes bits rather than adding them, which is also what makes the
/// change measurable: compare at matched size, on a perceptual metric.
///
/// `strength` is in QP steps per octave of variance. Zero disables it.
fn adaptive_qp(y: &[u8], cw: u32, mbw: usize, mbh: usize, base: i32, strength: f32) -> Vec<i32> {
    if strength == 0.0 {
        return vec![base; mbw * mbh];
    }
    let mut energy = Vec::with_capacity(mbw * mbh);
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let (ox, oy) = ((mbx * MB_SIZE as usize) as u32, (mby * MB_SIZE as usize) as u32);
            let (mut sum, mut sq) = (0i64, 0i64);
            for j in 0..MB_SIZE {
                let row = ((oy + j) * cw + ox) as usize;
                for &p in &y[row..row + MB_SIZE as usize] {
                    sum += i64::from(p);
                    sq += i64::from(p) * i64::from(p);
                }
            }
            let n = i64::from(MB_SIZE * MB_SIZE);
            let var = (sq - sum * sum / n).max(0) as f32 / n as f32;
            energy.push((var + 1.0).log2());
        }
    }
    let mean = energy.iter().sum::<f32>() / energy.len() as f32;
    energy
        .iter()
        .map(|e| {
            // Clamp the swing: beyond a few QP steps the flat blocks stop
            // improving and the busy ones start visibly falling apart.
            let adj = (strength * (e - mean)).round().clamp(-5.0, 5.0) as i32;
            (base + adj).clamp(0, 51)
        })
        .collect()
}

/// The quantizer multiplier for position (0,0) — the only one a DC block needs.
fn quant_mf0(m: usize) -> i32 {
    const MF0: [i32; 6] = [13107, 11916, 10082, 9362, 8192, 7282];
    MF0[m]
}

#[allow(clippy::too_many_arguments)]
fn write_macroblock(
    w: &mut BitWriter,
    ctx: &Ctx,
    luma: &LumaCoding,
    chroma: &ChromaData,
    mbx: usize,
    mby: usize,
    qp_delta: i32,
) {
    match luma {
        LumaCoding::NxN(l) => {
            w.write_ue(MB_TYPE_I_NXN);
            // Per-block prediction modes, each against the smaller of its two
            // neighbours' modes (§8.3.1.1).
            for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
                let px = (mbx * MB_SIZE as usize + bx4 * 4) as u32;
                let py = (mby * MB_SIZE as usize + by4 * 4) as u32;
                let pred = mpm(ctx, px, py);
                let mode = l.modes[blk];
                if mode == pred {
                    w.write_bit(1);
                } else {
                    w.write_bit(0);
                    let rem = if mode > pred { mode - 1 } else { mode };
                    w.write_bits(rem as u32, 3);
                }
            }
            w.write_ue(chroma.mode as u32);
            let cbp = l.cbp_luma | (chroma.cbp << 4);
            w.write_ue(CBP_INTRA_CODE[cbp as usize]);
            if cbp > 0 {
                w.write_se(qp_delta);
            }
        }
        LumaCoding::I16(l) => {
            // Table 7-11 packs the prediction mode and both halves of the coded
            // block pattern into `mb_type` itself, so no `coded_block_pattern`
            // is sent — and `mb_qp_delta` is unconditional.
            let mb_type = 1
                + u32::from(l.pred_mode)
                + 4 * chroma.cbp
                + 12 * u32::from(l.cbp_luma != 0);
            w.write_ue(mb_type);
            w.write_ue(chroma.mode as u32);
            w.write_se(qp_delta);
        }
    }

    // ---- residual (§7.3.5.3) ----
    match luma {
        LumaCoding::NxN(l) => {
            for i8 in 0..4usize {
                if l.cbp_luma & (1 << i8) == 0 {
                    continue;
                }
                for i4 in 0..4usize {
                    let blk = i8 * 4 + i4;
                    let (px, py) = blk_origin(mbx, mby, blk);
                    let nc = luma_nc(ctx, px, py);
                    cavlc::encode_block(w, &l.levels[blk], BlockKind::Luma4x4, nc);
                }
            }
        }
        LumaCoding::I16(l) => {
            // The DC block is always present, and takes its `nC` from the
            // neighbours of luma block 0 — both of which lie outside this
            // macroblock, so this is well defined before anything else is coded.
            let (px0, py0) = blk_origin(mbx, mby, 0);
            cavlc::encode_block(w, &l.dc, BlockKind::Luma16x16Dc, luma_nc(ctx, px0, py0));
            if l.cbp_luma != 0 {
                for blk in 0..16usize {
                    let (px, py) = blk_origin(mbx, mby, blk);
                    let nc = luma_nc(ctx, px, py);
                    cavlc::encode_block(w, &l.ac[blk], BlockKind::Luma16x16Ac, nc);
                }
            }
        }
    }
    if chroma.cbp & 3 != 0 {
        for plane in 0..2 {
            cavlc::encode_block(w, &chroma.dc[plane], BlockKind::ChromaDc, -1);
        }
    }
    if chroma.cbp & 2 != 0 {
        for plane in 0..2 {
            for b in 0..4usize {
                let cx = (mbx * 8 + (b % 2) * 4) as u32;
                let cy = (mby * 8 + (b / 2) * 4) as u32;
                let nc = chroma_nc(ctx, plane, cx, cy);
                cavlc::encode_block(w, &chroma.ac[plane][b], BlockKind::ChromaAc, nc);
            }
        }
    }
}

/// Which entropy coder the slice is using.
///
/// The macroblock *decisions* — modes, transform levels, `I_NxN` versus
/// `I_16x16` — do not depend on the entropy coder, only their cost does. So the
/// build path is shared and only the emitter differs, and the rate-distortion
/// comparison asks whichever one is in use what a candidate would cost.
enum Entropy {
    Cavlc(BitWriter),
    Cabac(Box<Cabac>),
}

impl Entropy {
    fn bit_len(&self) -> usize {
        match self {
            Entropy::Cavlc(w) => w.bit_len(),
            Entropy::Cabac(c) => c.bit_len(),
        }
    }

    /// A copy that starts with an empty buffer, for costing a candidate.
    fn probe(&self) -> Entropy {
        match self {
            Entropy::Cavlc(_) => Entropy::Cavlc(BitWriter::new()),
            Entropy::Cabac(c) => Entropy::Cabac(Box::new(c.probe())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_mb(
        &mut self,
        ctx: &Ctx,
        luma: &LumaCoding,
        chroma: &ChromaData,
        mbx: usize,
        mby: usize,
        qp_delta: i32,
    ) {
        match self {
            Entropy::Cavlc(w) => write_macroblock(w, ctx, luma, chroma, mbx, mby, qp_delta),
            Entropy::Cabac(c) => {
                write_macroblock_cabac(c, ctx, luma, chroma, mbx, mby, qp_delta)
            }
        }
    }
}

/// Everything the CABAC context derivations need to know about a neighbouring
/// macroblock, gathered once (§9.3.3.1.1).
#[derive(Clone, Copy, Default)]
struct MbNeighbor {
    avail: bool,
    i16: bool,
    cbp_luma: u32,
    cbp_chroma: u32,
    chroma_mode: u8,
    /// `coded_block_flag` of the `Intra16x16DCLevel` block, if this macroblock
    /// had one.
    dc_cbf: bool,
    /// `coded_block_flag` of each chroma DC block. Distinct from
    /// `cbp_chroma != 0`, which only says the block was *present*.
    chroma_dc_cbf: [bool; 2],
}

impl Ctx {
    fn neighbor(&self, mbx: usize, mby: usize, dx: isize, dy: isize) -> MbNeighbor {
        let (x, y) = (mbx as isize + dx, mby as isize + dy);
        if x < 0 || y < 0 || x >= self.mbw as isize {
            return MbNeighbor::default();
        }
        let i = y as usize * self.mbw + x as usize;
        if i >= self.mb.len() {
            return MbNeighbor::default();
        }
        self.mb[i]
    }
}

#[allow(clippy::too_many_arguments)]
fn write_macroblock_cabac(
    c: &mut Cabac,
    ctx: &Ctx,
    luma: &LumaCoding,
    chroma: &ChromaData,
    mbx: usize,
    mby: usize,
    qp_delta: i32,
) {
    let a = ctx.neighbor(mbx, mby, -1, 0);
    let b = ctx.neighbor(mbx, mby, 0, -1);

    // ---- mb_type ----
    // The bin-0 context counts neighbours that are *not* I_NxN, which on intra
    // pictures means "neighbours that were flat enough for I_16x16".
    let (a_not_nxn, b_not_nxn) = (a.avail && a.i16, b.avail && b.i16);
    let cbp_luma = match luma {
        LumaCoding::NxN(l) => l.cbp_luma,
        LumaCoding::I16(l) => l.cbp_luma,
    };
    match luma {
        LumaCoding::NxN(l) => {
            c.mb_type_nxn(a_not_nxn, b_not_nxn);
            for (blk, &(bx4, by4)) in BLK_POS.iter().enumerate() {
                let px = (mbx * MB_SIZE as usize + bx4 * 4) as u32;
                let py = (mby * MB_SIZE as usize + by4 * 4) as u32;
                c.intra4x4_mode(l.modes[blk], mpm(ctx, px, py));
            }
        }
        LumaCoding::I16(l) => {
            c.mb_type_i16(
                l.pred_mode,
                chroma.cbp,
                l.cbp_luma != 0,
                a_not_nxn,
                b_not_nxn,
            );
        }
    }

    // ---- intra_chroma_pred_mode ----
    c.chroma_pred_mode(
        chroma.mode,
        a.avail && a.chroma_mode != 0,
        b.avail && b.chroma_mode != 0,
    );

    // ---- coded_block_pattern (I_NxN only; I_16x16 packs it into mb_type) ----
    if matches!(luma, LumaCoding::NxN(_)) {
        let luma_inc = std::array::from_fn(|b8| cbp_luma_ctx_inc(ctx, mbx, mby, b8, cbp_luma));
        let cin0 = usize::from(a.avail && a.cbp_chroma != 0)
            + 2 * usize::from(b.avail && b.cbp_chroma != 0);
        let cin1 = usize::from(a.avail && a.cbp_chroma == 2)
            + 2 * usize::from(b.avail && b.cbp_chroma == 2);
        c.coded_block_pattern(cbp_luma, chroma.cbp, luma_inc, cin0, cin1);
    }
    if cbp_luma != 0 || chroma.cbp != 0 || matches!(luma, LumaCoding::I16(_)) {
        c.mb_qp_delta(qp_delta, ctx.prev_qp_delta_nonzero);
    }

    // ---- residual ----
    match luma {
        LumaCoding::I16(l) => {
            let (px0, py0) = blk_origin(mbx, mby, 0);
            let dc_cbf = l.dc.iter().any(|&v| v != 0);
            let inc = dc_cbf_inc(ctx, mbx, mby);
            c.coded_block_flag(cabac::CAT_LUMA_DC, inc, dc_cbf);
            if dc_cbf {
                c.residual_block(&l.dc, cabac::CAT_LUMA_DC);
            }
            let _ = (px0, py0);
            if l.cbp_luma != 0 {
                for blk in 0..16usize {
                    let (px, py) = blk_origin(mbx, mby, blk);
                    let cbf = l.ac[blk].iter().any(|&v| v != 0);
                    let inc = luma_cbf_inc(ctx, px, py);
                    c.coded_block_flag(cabac::CAT_LUMA_AC, inc, cbf);
                    if cbf {
                        c.residual_block(&l.ac[blk], cabac::CAT_LUMA_AC);
                    }
                }
            }
        }
        LumaCoding::NxN(l) => {
            for i8 in 0..4usize {
                if l.cbp_luma & (1 << i8) == 0 {
                    continue;
                }
                for i4 in 0..4usize {
                    let blk = i8 * 4 + i4;
                    let (px, py) = blk_origin(mbx, mby, blk);
                    let cbf = l.levels[blk].iter().any(|&v| v != 0);
                    let inc = luma_cbf_inc(ctx, px, py);
                    c.coded_block_flag(cabac::CAT_LUMA_4X4, inc, cbf);
                    if cbf {
                        c.residual_block(&l.levels[blk], cabac::CAT_LUMA_4X4);
                    }
                }
            }
        }
    }
    if chroma.cbp & 3 != 0 {
        for plane in 0..2 {
            let cbf = chroma.dc[plane].iter().any(|&v| v != 0);
            let inc = chroma_dc_cbf_inc(ctx, mbx, mby, plane);
            c.coded_block_flag(cabac::CAT_CHROMA_DC, inc, cbf);
            if cbf {
                c.residual_block(&chroma.dc[plane], cabac::CAT_CHROMA_DC);
            }
        }
    }
    if chroma.cbp & 2 != 0 {
        for plane in 0..2 {
            for blk in 0..4usize {
                let cx = (mbx * 8 + (blk % 2) * 4) as u32;
                let cy = (mby * 8 + (blk / 2) * 4) as u32;
                let cbf = chroma.ac[plane][blk].iter().any(|&v| v != 0);
                let inc = chroma_ac_cbf_inc(ctx, plane, cx, cy, mbx, mby, chroma.cbp);
                c.coded_block_flag(cabac::CAT_CHROMA_AC, inc, cbf);
                if cbf {
                    c.residual_block(&chroma.ac[plane][blk], cabac::CAT_CHROMA_AC);
                }
            }
        }
    }
}

/// `ctxIdxInc` for `coded_block_pattern`'s luma bins (§9.3.3.1.1.4).
///
/// The sense is inverted: a neighbouring 8×8 that *has* coefficients scores
/// zero, and so does one that does not exist. Only a neighbour that exists and
/// is empty pushes the context towards "this one is probably empty too".
fn cbp_luma_ctx_inc(ctx: &Ctx, mbx: usize, mby: usize, b8: usize, cbp_here: u32) -> usize {
    // The 8×8 to the left and above, which for the odd/lower blocks lie inside
    // this macroblock and are therefore already decided.
    let left = if b8 % 2 == 1 {
        Some((cbp_here, b8 - 1))
    } else {
        let n = ctx.neighbor(mbx, mby, -1, 0);
        n.avail.then_some((n.cbp_luma, b8 + 1))
    };
    let above = if b8 >= 2 {
        Some((cbp_here, b8 - 2))
    } else {
        let n = ctx.neighbor(mbx, mby, 0, -1);
        n.avail.then_some((n.cbp_luma, b8 + 2))
    };
    let flag = |v: Option<(u32, usize)>| match v {
        Some((cbp, idx)) => usize::from((cbp >> idx) & 1 == 0),
        None => 0,
    };
    flag(left) + 2 * flag(above)
}

/// `ctxIdxInc` for a luma 4×4 block's `coded_block_flag` (§9.3.3.1.1.9).
///
/// An unavailable neighbour scores 1 here, not 0 — for an intra macroblock the
/// absence of a neighbour is read as "expect coefficients", which is the
/// opposite of the coded-block-pattern rule above.
fn luma_cbf_inc(ctx: &Ctx, px: u32, py: u32) -> usize {
    let f = |nx: i32, ny: i32| -> usize {
        if !ctx.avail_luma(nx, ny, px, py) {
            return 1;
        }
        usize::from(ctx.nnz_y[grid_index(ctx.cw, nx as u32, ny as u32)] > 0)
    };
    f(px as i32 - 1, py as i32) + 2 * f(px as i32, py as i32 - 1)
}

/// `ctxIdxInc` for the `I_16x16` luma DC block.
///
/// The neighbouring block only exists if that macroblock was itself `I_16x16`;
/// against an `I_NxN` neighbour there is no DC block to consult, which scores
/// zero rather than the "missing neighbour" one.
fn dc_cbf_inc(ctx: &Ctx, mbx: usize, mby: usize) -> usize {
    let f = |n: MbNeighbor| -> usize {
        if !n.avail {
            return 1;
        }
        usize::from(n.i16 && n.dc_cbf)
    };
    f(ctx.neighbor(mbx, mby, -1, 0)) + 2 * f(ctx.neighbor(mbx, mby, 0, -1))
}

/// `ctxIdxInc` for a chroma DC block's `coded_block_flag`.
///
/// `cbp_chroma != 0` only says the neighbour *had* a DC block; what the context
/// wants is whether that block actually carried coefficients. The two differ
/// whenever chroma AC is coded over a zero DC, which is exactly what smooth
/// content produces — and never what noise does, which is why a randomised
/// sweep passed while a gradient did not.
fn chroma_dc_cbf_inc(ctx: &Ctx, mbx: usize, mby: usize, plane: usize) -> usize {
    let f = |n: MbNeighbor| -> usize {
        if !n.avail {
            return 1;
        }
        if n.cbp_chroma == 0 {
            return 0; // no DC block there to consult
        }
        usize::from(n.chroma_dc_cbf[plane])
    };
    f(ctx.neighbor(mbx, mby, -1, 0)) + 2 * f(ctx.neighbor(mbx, mby, 0, -1))
}

/// `ctxIdxInc` for a chroma AC block's `coded_block_flag`.
///
/// `cur_cbp_chroma` is this macroblock's own value, and it has to be passed in:
/// three of the four chroma blocks have a neighbour inside the *current*
/// macroblock, whose entry in `ctx.mb` is not written until after the whole
/// macroblock is emitted.
fn chroma_ac_cbf_inc(
    ctx: &Ctx,
    plane: usize,
    cx: u32,
    cy: u32,
    mbx: usize,
    mby: usize,
    cur_cbp_chroma: u32,
) -> usize {
    let f = |nx: i32, ny: i32| -> usize {
        if nx < 0 || ny < 0 {
            return 1;
        }
        let (nxu, nyu) = (nx as u32, ny as u32);
        if !chroma_block_done(ctx, nxu, nyu, cx, cy) {
            return 1;
        }
        // The block is only coded when its macroblock carries chroma AC.
        let (mx, my) = ((nxu / 8) as usize, (nyu / 8) as usize);
        let cbp = if (mx, my) == (mbx, mby) {
            cur_cbp_chroma
        } else {
            ctx.mb[my * ctx.mbw + mx].cbp_chroma
        };
        if cbp != 2 {
            return 0;
        }
        usize::from(ctx.nnz_c[plane][grid_index(ctx.cwc, nxu, nyu)] > 0)
    };
    f(cx as i32 - 1, cy as i32) + 2 * f(cx as i32, cy as i32 - 1)
}

/// Picture-space origin of a macroblock's 4×4 luma block, by coding index.
fn blk_origin(mbx: usize, mby: usize, blk: usize) -> (u32, u32) {
    let (bx4, by4) = BLK_POS[blk];
    (
        (mbx * MB_SIZE as usize + bx4 * 4) as u32,
        (mby * MB_SIZE as usize + by4 * 4) as u32,
    )
}

/// Reference samples along the top and left edges of a whole macroblock.
fn luma16_neighbors(ctx: &Ctx, mbx: usize, mby: usize) -> intra::Neighbors16 {
    let s = |x: i32, y: i32| ctx.ry[(y as u32 * ctx.cw + x as u32) as usize] as i32;
    let (xi, yi) = ((mbx * MB_SIZE as usize) as i32, (mby * MB_SIZE as usize) as i32);
    let (bx, by) = (xi as u32, yi as u32);
    let above = (0..16)
        .all(|k| ctx.avail_luma(xi + k, yi - 1, bx, by))
        .then(|| std::array::from_fn(|k| s(xi + k as i32, yi - 1)));
    let left = (0..16)
        .all(|k| ctx.avail_luma(xi - 1, yi + k, bx, by))
        .then(|| std::array::from_fn(|k| s(xi - 1, yi + k as i32)));
    let corner = ctx
        .avail_luma(xi - 1, yi - 1, bx, by)
        .then(|| s(xi - 1, yi - 1));
    intra::Neighbors16 {
        above,
        left,
        corner,
    }
}

/// Pick the 16×16 mode by SATD rather than SAD.
///
/// At this size the residual goes through a transform before it is coded, and
/// SAD cannot tell a smooth residual from a noisy one of the same magnitude —
/// it picks modes that cost more bits. The Hadamard is a cheap stand-in for
/// what the transform will actually do.
fn best_luma16_mode(orig: &[i32], n: &intra::Neighbors16) -> u8 {
    let mut best = (u64::MAX, intra::I16_DC);
    for mode in 0..=3u8 {
        if !intra::allowed16(mode, n) {
            continue;
        }
        let p = intra::predict16(mode, n);
        let mut satd = 0u64;
        for by in 0..4 {
            for bx in 0..4 {
                let mut blk = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        let k = (by * 4 + j) * 16 + bx * 4 + i;
                        blk[j * 4 + i] = orig[k] - p[k];
                    }
                }
                satd += hadamard_sad(&blk);
            }
        }
        if satd < best.0 {
            best = (satd, mode);
        }
    }
    best.1
}

/// Sum of absolute 4×4 Hadamard coefficients, normalised back to SAD scale.
fn hadamard_sad(b: &[i32; 16]) -> u64 {
    let t = luma_dc_hadamard(b);
    let s: u64 = t.iter().map(|v| v.unsigned_abs() as u64).sum();
    s / 4
}

/// Most-probable mode: the smaller of the left and above blocks' modes, DC when
/// either is missing (§8.3.1.1).
fn mpm(ctx: &Ctx, px: u32, py: u32) -> u8 {
    // A neighbour that exists but was coded as `I_16x16` has no 4×4 mode; the
    // spec substitutes DC for it, which is *not* the same as it being missing —
    // a missing neighbour forces the prediction to DC outright, while this one
    // still loses to a smaller mode on the other side.
    let left = ctx
        .avail_luma(px as i32 - 1, py as i32, px, py)
        .then(|| ctx.modes[grid_index(ctx.cw, px - 4, py)].unwrap_or(intra::DC));
    let above = ctx
        .avail_luma(px as i32, py as i32 - 1, px, py)
        .then(|| ctx.modes[grid_index(ctx.cw, px, py - 4)].unwrap_or(intra::DC));
    match (left, above) {
        (Some(a), Some(b)) => a.min(b),
        _ => intra::DC,
    }
}

fn luma_nc(ctx: &Ctx, px: u32, py: u32) -> i32 {
    let left = ctx
        .avail_luma(px as i32 - 1, py as i32, px, py)
        .then(|| ctx.nnz_y[grid_index(ctx.cw, px - 4, py)]);
    let above = ctx
        .avail_luma(px as i32, py as i32 - 1, px, py)
        .then(|| ctx.nnz_y[grid_index(ctx.cw, px, py - 4)]);
    cavlc::derive_nc(left, above)
}

fn chroma_nc(ctx: &Ctx, plane: usize, cx: u32, cy: u32) -> i32 {
    let inside = |x: i32, y: i32| x >= 0 && y >= 0;
    let left = (inside(cx as i32 - 1, cy as i32)
        && chroma_block_done(ctx, cx - 4, cy, cx, cy))
    .then(|| ctx.nnz_c[plane][grid_index(ctx.cwc, cx - 4, cy)]);
    let above = (inside(cx as i32, cy as i32 - 1)
        && chroma_block_done(ctx, cx, cy - 4, cx, cy))
    .then(|| ctx.nnz_c[plane][grid_index(ctx.cwc, cx, cy - 4)]);
    cavlc::derive_nc(left, above)
}

/// Chroma 4×4 blocks are coded in raster order inside a macroblock, so ordering
/// is `(macroblock, raster index)`.
fn chroma_block_done(ctx: &Ctx, nx: u32, ny: u32, cx: u32, cy: u32) -> bool {
    let key = |x: u32, y: u32| {
        let (mx, my) = ((x / 8) as usize, (y / 8) as usize);
        let (bx, by) = (((x % 8) / 4) as usize, ((y % 8) / 4) as usize);
        (my * ctx.mbw + mx, by * 2 + bx)
    };
    key(nx, ny) < key(cx, cy)
}

fn luma_neighbors(ctx: &Ctx, px: u32, py: u32) -> intra::Neighbors4 {
    let s = |x: i32, y: i32| ctx.ry[(y as u32 * ctx.cw + x as u32) as usize] as i32;
    let (xi, yi) = (px as i32, py as i32);

    let above = (0..4)
        .all(|k| ctx.avail_luma(xi + k, yi - 1, px, py))
        .then(|| [s(xi, yi - 1), s(xi + 1, yi - 1), s(xi + 2, yi - 1), s(xi + 3, yi - 1)]);
    let above_right = (0..4)
        .all(|k| ctx.avail_luma(xi + 4 + k, yi - 1, px, py))
        .then(|| {
            [
                s(xi + 4, yi - 1),
                s(xi + 5, yi - 1),
                s(xi + 6, yi - 1),
                s(xi + 7, yi - 1),
            ]
        });
    let left = (0..4)
        .all(|k| ctx.avail_luma(xi - 1, yi + k, px, py))
        .then(|| [s(xi - 1, yi), s(xi - 1, yi + 1), s(xi - 1, yi + 2), s(xi - 1, yi + 3)]);
    let corner = ctx
        .avail_luma(xi - 1, yi - 1, px, py)
        .then(|| s(xi - 1, yi - 1));
    intra::Neighbors4::new(above, above_right, left, corner)
}

fn chroma_neighbors_plane(ctx: &Ctx, mbx: usize, mby: usize, plane: usize) -> intra::NeighborsC {
    let rp: &[u8] = if plane == 0 { &ctx.ru } else { &ctx.rv };
    let s = |x: i32, y: i32| rp[(y as u32 * ctx.cwc + x as u32) as usize] as i32;
    let (xi, yi) = ((mbx * 8) as i32, (mby * 8) as i32);
    let above = (0..8)
        .all(|k| ctx.avail_chroma(xi + k, yi - 1, mbx, mby))
        .then(|| std::array::from_fn(|k| s(xi + k as i32, yi - 1)));
    let left = (0..8)
        .all(|k| ctx.avail_chroma(xi - 1, yi + k, mbx, mby))
        .then(|| std::array::from_fn(|k| s(xi - 1, yi + k as i32)));
    let corner = ctx
        .avail_chroma(xi - 1, yi - 1, mbx, mby)
        .then(|| s(xi - 1, yi - 1));
    intra::NeighborsC {
        above,
        left,
        corner,
    }
}

/// A stand-in used only to choose the shared chroma mode; both planes are
/// scored together because the bitstream signals one mode for both.
fn chroma_neighbors(ctx: &Ctx, mbx: usize, mby: usize) -> intra::NeighborsC {
    chroma_neighbors_plane(ctx, mbx, mby, 0)
}

fn best_luma_mode(orig: &[i32], n: &intra::Neighbors4) -> u8 {
    let mut best = (u32::MAX, intra::DC);
    for mode in 0..=8u8 {
        if !intra::allowed(mode, n) {
            continue;
        }
        let p = intra::predict4(mode, n);
        let sad: u32 = orig
            .iter()
            .zip(&p)
            .map(|(&o, &q)| (o - q).unsigned_abs())
            .sum();
        if sad < best.0 {
            best = (sad, mode);
        }
    }
    best.1
}

fn best_chroma_mode(
    ctx: &Ctx,
    u: &[u8],
    v: &[u8],
    mbx: usize,
    mby: usize,
    _hint: &intra::NeighborsC,
) -> u8 {
    let (cx, cy) = ((mbx * 8) as u32, (mby * 8) as u32);
    let mut best = (u32::MAX, intra::C_DC);
    for mode in 0..=3u8 {
        let mut sad = 0u32;
        let mut ok = true;
        for (plane, src) in [(0usize, u), (1usize, v)] {
            let n = chroma_neighbors_plane(ctx, mbx, mby, plane);
            if !intra::chroma_allowed(mode, &n) {
                ok = false;
                break;
            }
            let p = intra::predict_chroma(mode, &n);
            let o = gather(src, ctx.cwc, cx, cy, 8);
            sad += o
                .iter()
                .zip(&p)
                .map(|(&a, &b)| (a - b).unsigned_abs())
                .sum::<u32>();
        }
        if ok && sad < best.0 {
            best = (sad, mode);
        }
    }
    best.1
}

fn grid_index(width: u32, px: u32, py: u32) -> usize {
    ((py / 4) * (width / 4) + px / 4) as usize
}

fn gather(plane: &[u8], w: u32, x: u32, y: u32, n: u32) -> Vec<i32> {
    let mut b = vec![0i32; (n * n) as usize];
    for yy in 0..n {
        for xx in 0..n {
            b[(yy * n + xx) as usize] = plane[((y + yy) * w + x + xx) as usize] as i32;
        }
    }
    b
}

fn store(plane: &mut [u8], w: u32, x: u32, y: u32, n: u32, block: &[i32]) {
    for yy in 0..n {
        for xx in 0..n {
            plane[((y + yy) * w + x + xx) as usize] =
                block[(yy * n + xx) as usize].clamp(0, 255) as u8;
        }
    }
}

impl CompressedEncoder {
    /// New encoder for `width × height` at quantization `qp` (roughly 18–40
    /// useful, lower is bigger and better).
    pub fn new(width: u32, height: u32, qp: i32) -> Self {
        let mut cfg = H264Config::new(width, height);
        cfg.qp = qp;
        cfg.entropy = EntropyMode::Cabac;
        cfg.deblock = true;
        cfg.aq_strength = 0.0;
        let sps = nal_unit_base(NalUnitType::Sps, &write_sps(&cfg));
        let pps = nal_unit_base(
            NalUnitType::Pps,
            &write_pps_deblock_control(cfg.entropy == EntropyMode::Cabac),
        );
        Self {
            cfg,
            qp,
            headers_emitted: false,
            sps,
            pps,
        }
    }

    /// Turn the in-loop deblocking filter on or off. On by default.
    pub fn deblock(mut self, on: bool) -> Self {
        self.cfg.deblock = on;
        self
    }

    /// Choose the entropy coder. CABAC by default; CAVLC produces a
    /// Baseline-profile stream, which is larger but reaches older hardware.
    pub fn entropy(mut self, mode: EntropyMode) -> Self {
        self.cfg.entropy = mode;
        self.sps = nal_unit_base(NalUnitType::Sps, &write_sps(&self.cfg));
        self.pps = nal_unit_base(
            NalUnitType::Pps,
            &write_pps_deblock_control(mode == EntropyMode::Cabac),
        );
        self
    }

    /// Set the adaptive-quantization strength (see
    /// [`H264Config::aq_strength`]). Off by default.
    pub fn aq(mut self, strength: f32) -> Self {
        self.cfg.aq_strength = strength;
        self
    }

    pub fn config(&self) -> &H264Config {
        &self.cfg
    }

    pub fn sps(&self) -> &[u8] {
        &self.sps
    }

    pub fn pps(&self) -> &[u8] {
        &self.pps
    }

    /// Encode one 4:2:0 frame → Annex-B access unit, plus the reconstruction a
    /// conformant decoder must reproduce. The first call prepends SPS/PPS.
    pub fn encode_frame(&mut self, frame: &Yuv420Frame) -> (Vec<u8>, Reconstruction) {
        assert_eq!(
            (frame.width, frame.height),
            (self.cfg.width, self.cfg.height),
            "frame size differs from encoder size"
        );
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);

        let (slice, recon) = compressed_slice_rbsp(&self.cfg, &y, &u, &v, self.qp);
        let mut au = Vec::new();
        if !self.headers_emitted {
            push_annexb(&mut au, &self.sps);
            push_annexb(&mut au, &self.pps);
            self.headers_emitted = true;
        }
        push_annexb(&mut au, &nal_unit_base(NalUnitType::Idr, &slice));
        (au, recon)
    }

    /// One frame's slice NAL, in Annex-B form, without SPS or PPS.
    ///
    /// Takes `&self`. An all-intra stream carries nothing between frames —
    /// `compressed_slice_rbsp` reads the config and the pixels and nothing else
    /// — so frames can be encoded on separate threads and the result is
    /// byte-identical to encoding them in order.
    pub fn encode_slice_au(&self, frame: &Yuv420Frame) -> Vec<u8> {
        self.encode_slice_au_at(frame, self.qp)
    }

    /// [`encode_slice_au`](Self::encode_slice_au) at a quantizer chosen per
    /// frame, which is what rate control needs.
    ///
    /// Only the slice header moves: `pic_init_qp_minus26` stays zero in the PPS
    /// and the frame's quantizer rides in `slice_qp_delta`, so one SPS and PPS
    /// serve every frame however the rate controller steers.
    pub fn encode_slice_au_at(&self, frame: &Yuv420Frame, qp: i32) -> Vec<u8> {
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let (slice, _recon) = compressed_slice_rbsp(&self.cfg, &y, &u, &v, qp.clamp(0, 51));
        let mut au = Vec::new();
        push_annexb(&mut au, &nal_unit_base(NalUnitType::Idr, &slice));
        au
    }
}


/// Encode `frames` as a compressed intra H.264 `.mp4` (`avc1`) at quantization
/// `qp`, entirely in-process — no ffmpeg, and it builds for `wasm32`.
///
/// This is the compressed counterpart to [`crate::codec::h264::encode_mp4`],
/// which codes `I_PCM` and so writes raw samples at roughly 1.5 bytes per pixel
/// per frame. Same codec, same playback reach; two orders of magnitude smaller.
///
/// `qp` is the H.264 quantization parameter: lower is better and bigger, the
/// usable range is roughly 18-40, and 26 is a reasonable default.
pub fn encode_mp4(width: u32, height: u32, fps: u32, qp: i32, frames: &[Yuv420Frame]) -> Vec<u8> {
    encode_mp4_from_iter(width, height, fps, qp, None, frames.iter().cloned())
}

/// [`encode_mp4`] over an iterator, so the caller can generate frames lazily and
/// never hold more than one — see [`crate::codec::h264::encode_mp4_from_iter`]
/// for why that matters at high resolutions.
pub fn encode_mp4_from_iter<I>(
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    captions: Option<&CaptionTrack>,
    frames: I,
) -> Vec<u8>
where
    I: IntoIterator<Item = Yuv420Frame>,
{
    let mut out = Vec::new();
    write_mp4_from_iter(&mut out, width, height, fps, qp, captions, frames)
        .expect("writing to a Vec cannot fail");
    out
}

/// [`encode_mp4_from_iter`] written straight to a sink, so neither the clip nor
/// the finished file is ever held whole.
pub fn write_mp4_from_iter<W, I>(
    out: &mut W,
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    captions: Option<&CaptionTrack>,
    frames: I,
) -> std::io::Result<()>
where
    W: std::io::Write,
    I: IntoIterator<Item = Yuv420Frame>,
{
    write_mp4_quality(out, width, height, fps, Quality::Qp(qp), captions, frames, None)
}

/// [`write_mp4_from_iter`] with a quality target rather than a fixed quantizer.
///
/// `total` is the frame count when the caller knows it, which lets average
/// bitrate control aim each frame at the average of what is left rather than
/// spreading corrections over a guessed horizon.
#[allow(clippy::too_many_arguments)]
pub fn write_mp4_quality<W, I>(
    out: &mut W,
    width: u32,
    height: u32,
    fps: u32,
    quality: Quality,
    captions: Option<&CaptionTrack>,
    frames: I,
    total: Option<u32>,
) -> std::io::Result<()>
where
    W: std::io::Write,
    I: IntoIterator<Item = Yuv420Frame>,
{
    let fps = fps.max(1);
    let timescale = 600u32;
    let base_qp = match quality {
        Quality::Qp(q) => q,
        Quality::Bitrate(_) => PROBE_QP,
    };
    let enc = CompressedEncoder::new(width, height, base_qp);
    let avcc = build_avcc(enc.sps(), enc.pps());

    // Frames go out in batches so they can be encoded concurrently. Nothing
    // carries between them, so the batching cannot show up in the output — it
    // only decides how many cores are busy and how much is held at once.
    let mut batch = crate::codec::frames_in_flight(width, height);
    let mut samples = Vec::new();
    let mut frames = frames.into_iter();
    let mut rc = match quality {
        Quality::Qp(_) => None,
        Quality::Bitrate(bps) => Some(RateControl::new(bps, fps, total)),
    };
    if let Some(r) = rc.as_ref() {
        // The controller only learns between groups, so a clip that fits in one
        // never gets to correct itself. Sized from the clip and the memory
        // budget alone — the cores available must not change the output.
        batch = r.group_size(crate::codec::frames_by_memory(width, height));
    }

    // The controller has to start somewhere. One frame coded at the probe
    // quantizer tells it what this content costs, which is worth far more than
    // any prior: content varies by orders of magnitude and quantizers do not.
    let mut pending: Vec<Yuv420Frame> = Vec::with_capacity(batch);
    // `if let (Some(r), Some(f)) = (rc.as_mut(), frames.next())` would pull a
    // frame whether or not there is a controller to feed it to — the tuple is
    // built before the pattern is tried — and a fixed-quantizer encode would
    // silently lose its first frame.
    if let Some(r) = rc.as_mut() {
        if let Some(first) = frames.next() {
            let probe = enc.encode_slice_au_at(&first, PROBE_QP);
            r.seed(PROBE_QP, probe.len());
            // If the answer looks far from the probe, measure there too: one
            // extra frame buys a local slope instead of a chord across the
            // curve's bend.
            if let Some(q2) = r.probe_again() {
                let again = enc.encode_slice_au_at(&first, q2);
                r.seed_pair(q2, again.len());
            }
            pending.push(first);
        }
    }

    loop {
        while pending.len() < batch {
            match frames.next() {
                Some(f) => pending.push(f),
                None => break,
            }
        }
        if pending.is_empty() {
            break;
        }
        // One quantizer per batch: the controller is sequential and the encoder
        // is not, and a batch is small next to a clip.
        let qp = rc.as_ref().map_or(base_qp, |r| r.next_qp());
        let aus = crate::codec::map_frames(&pending, |f| enc.encode_slice_au_at(f, qp));
        pending.clear();
        for au in &aus {
            if let Some(r) = rc.as_mut() {
                r.observe(qp, au.len());
            }
            samples.push(H264Encoder::sample_from_au(au));
        }
    }
    match rc.as_ref().and_then(|r| r.outcome()) {
        Some(Reach::TooComplex { times }) => eprintln!(
            "threers: this content will not fit that bitrate — even at the coarsest \
             quantizer it is about {times:.1}x larger. The file is as small as the \
             encoder can make it."
        ),
        Some(Reach::TooSimple { times }) => eprintln!(
            "threers: this content cannot fill that bitrate — even losslessly quantized \
             it is about {times:.1}x smaller, so the file is smaller than asked for. \
             That is the content, not a failure."
        ),
        None => {}
    }
    let params = H264Mp4Params {
        width,
        height,
        timescale,
        frame_duration: timescale / fps,
        avcc_payload: &avcc,
        samples: &samples,
    };
    crate::codec::mp4::write_h264(out, &params, captions.filter(|t| !t.is_empty()))
}




/// A compressed intra H.264 encoder: pixels in, Annex-B out.
///
/// Every frame is an IDR whose macroblocks are `I_NxN` or `I_16x16`, whichever
/// costs less. This is the counterpart to
/// [`super::encoder::H264Encoder`], which codes `I_PCM` and so does not
/// compress at all.
pub struct CompressedEncoder {
    cfg: H264Config,
    qp: i32,
    headers_emitted: bool,
    sps: Vec<u8>,
    pps: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 9-4 maps code numbers onto CBP values one-to-one; the inverse this
    /// module builds must therefore be a permutation of 0..48.
    #[test]
    fn cbp_table_is_a_permutation() {
        let mut seen = [false; 48];
        for &c in CBP_INTRA_CODE.iter() {
            assert!(c < 48, "code number {c} out of range");
            assert!(!seen[c as usize], "code number {c} used twice");
            seen[c as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "some code number is unused");
    }

    #[test]
    fn block_positions_are_a_permutation() {
        let mut seen = [[false; 4]; 4];
        for &(x, y) in BLK_POS.iter() {
            assert!(!seen[y][x], "block position ({x},{y}) repeats");
            seen[y][x] = true;
        }
        assert!(seen.iter().flatten().all(|&s| s));
        // Spot-check the z-order: block 3 sits at (1,1), block 4 starts the next
        // 8x8 quadrant at (2,0).
        assert_eq!(BLK_POS[3], (1, 1));
        assert_eq!(BLK_POS[4], (2, 0));
        assert_eq!(blk_index(2, 0), 4);
    }

    #[test]
    fn chroma_qp_follows_table_8_15() {
        assert_eq!(chroma_qp(0), 0);
        assert_eq!(chroma_qp(29), 29);
        assert_eq!(chroma_qp(30), 29);
        assert_eq!(chroma_qp(39), 35);
        assert_eq!(chroma_qp(51), 39);
        // Monotonic and never above the luma QP past the split.
        for q in 0..=51 {
            assert!(chroma_qp(q) <= q.max(0));
            if q > 0 {
                assert!(chroma_qp(q) >= chroma_qp(q - 1));
            }
        }
    }
}
