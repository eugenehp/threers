//! CABAC entropy coding for H.264 intra slices (§9.3).
//!
//! CABAC replaces CAVLC's fixed code tables with an adaptive arithmetic coder.
//! The tables are gone; what remains is a probability state per *syntax element
//! in a context*, updated after every bin. On intra pictures it is worth roughly
//! 10-15% over CAVLC, and more of that at high rates where CAVLC's minimum of
//! one bit per symbol bites hardest.
//!
//! The arithmetic engine is [`CabacEncoder`], shared with the HEVC path.
//! H.264 §9.3.4.3 and H.265 §9.3.4.3 specify the same coder — the same
//! `rangeTabLPS`, the same transition tables, the same renormalisation and
//! flush. Only the *initialisation* differs, and that is what
//! [`super::cabac_tables`] carries.
//!
//! Everything here is I-slice syntax: `mb_type` for `I_NxN` and `I_16x16`, the
//! intra prediction modes, `coded_block_pattern`, and `residual_block_cabac`
//! for all five `ctxBlockCat` values a 4:2:0 intra macroblock can produce.

use crate::codec::h264::cabac_tables::{init_state, INIT_MN};
use crate::codec::hevc::cabac::{CabacEncoder, CtxModel};

/// `ctxIdx` bases (§9.3.3.1, Table 9-11).
const MB_TYPE_I: usize = 3;
const MB_QP_DELTA: usize = 60;
const CHROMA_PRED: usize = 64;
const PREV_INTRA_FLAG: usize = 68;
const REM_INTRA_MODE: usize = 69;
const CBP_LUMA: usize = 73;
const CBP_CHROMA: usize = 77;
const CBF: usize = 85;
const SIG: usize = 105;
const LAST_SIG: usize = 166;
const ABS_LEVEL: usize = 227;

/// Per-`ctxBlockCat` offsets within each element's context block (Table 9-40).
const CBF_CAT: [usize; 5] = [0, 4, 8, 12, 16];
const SIG_CAT: [usize; 5] = [0, 15, 29, 44, 47];
const ABS_CAT: [usize; 5] = [0, 10, 20, 30, 39];

/// `ctxBlockCat` values (Table 9-42). 4:2:0 intra reaches all five.
pub const CAT_LUMA_DC: usize = 0;
pub const CAT_LUMA_AC: usize = 1;
pub const CAT_LUMA_4X4: usize = 2;
pub const CAT_CHROMA_DC: usize = 3;
pub const CAT_CHROMA_AC: usize = 4;

/// The arithmetic engine plus one probability state per `ctxIdx`.
#[derive(Clone)]
pub struct Cabac {
    enc: CabacEncoder,
    ctx: [CtxModel; 276],
}

impl Cabac {
    pub fn new(slice_qp: i32) -> Self {
        Self {
            enc: CabacEncoder::new(),
            ctx: std::array::from_fn(|i| {
                let (m, n) = INIT_MN[i];
                let (state, mps) = init_state(m, n, slice_qp);
                CtxModel::from_state(state, mps)
            }),
        }
    }

    /// A copy with an empty output buffer, for costing a candidate. See
    /// [`CabacEncoder::probe`].
    pub fn probe(&self) -> Self {
        Self {
            enc: self.enc.probe(),
            ctx: self.ctx,
        }
    }

    pub fn bit_len(&self) -> usize {
        self.enc.bit_len()
    }

    /// Emit a context-coded bin.
    fn bin(&mut self, ctx_idx: usize, b: u32) {
        debug_assert!(
            ctx_idx <= 10 || (60..276).contains(&ctx_idx),
            "ctxIdx {ctx_idx} is not initialised for an I slice"
        );
        self.enc.encode_bin(&mut self.ctx[ctx_idx], b);
    }

    fn bypass(&mut self, b: u32) {
        self.enc.encode_bypass(b);
    }

    /// The `end_of_slice_flag`, and the `I_PCM` discriminator in `mb_type`,
    /// which share `ctxIdx 276` and are coded by the terminating routine
    /// rather than an adaptive context (§9.3.3.2.4).
    pub fn terminate(&mut self, b: u32) {
        self.enc.encode_terminate(b);
    }

    /// Finish the arithmetic coder and return the coded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.enc.finish()
    }

    // ---- macroblock-level syntax ----

    /// `mb_type` for an `I_NxN` macroblock: a single zero bin.
    pub fn mb_type_nxn(&mut self, left_not_nxn: bool, above_not_nxn: bool) {
        let inc = usize::from(left_not_nxn) + usize::from(above_not_nxn);
        self.bin(MB_TYPE_I + inc, 0);
    }

    /// `mb_type` for an `I_16x16` macroblock (Table 9-36).
    ///
    /// The bin string is not the `ue(v)` code number: it spells out the
    /// prediction mode and both halves of the coded block pattern directly,
    /// which is how a flat macroblock ends up costing so little.
    pub fn mb_type_i16(
        &mut self,
        pred_mode: u8,
        cbp_chroma: u32,
        cbp_luma_nonzero: bool,
        left_not_nxn: bool,
        above_not_nxn: bool,
    ) {
        let inc = usize::from(left_not_nxn) + usize::from(above_not_nxn);
        self.bin(MB_TYPE_I + inc, 1);
        self.terminate(0); // not I_PCM
        self.bin(MB_TYPE_I + 3, u32::from(cbp_luma_nonzero));
        self.bin(MB_TYPE_I + 4, u32::from(cbp_chroma != 0));
        // The context of the remaining bins depends on whether the chroma bin
        // above was set, because that decides how many bins are still to come.
        let (hi, lo) = if cbp_chroma != 0 {
            self.bin(MB_TYPE_I + 5, u32::from(cbp_chroma == 2));
            (MB_TYPE_I + 6, MB_TYPE_I + 7)
        } else {
            (MB_TYPE_I + 6, MB_TYPE_I + 7)
        };
        self.bin(hi, u32::from(pred_mode >> 1));
        self.bin(lo, u32::from(pred_mode & 1));
    }

    /// `prev_intra4x4_pred_mode_flag` and `rem_intra4x4_pred_mode`.
    pub fn intra4x4_mode(&mut self, mode: u8, predicted: u8) {
        if mode == predicted {
            self.bin(PREV_INTRA_FLAG, 1);
        } else {
            self.bin(PREV_INTRA_FLAG, 0);
            let rem = if mode > predicted { mode - 1 } else { mode };
            for k in 0..3 {
                self.bin(REM_INTRA_MODE, u32::from((rem >> k) & 1));
            }
        }
    }

    /// `intra_chroma_pred_mode` — truncated unary, `cMax = 3`.
    pub fn chroma_pred_mode(&mut self, mode: u8, left_nonzero: bool, above_nonzero: bool) {
        let inc = usize::from(left_nonzero) + usize::from(above_nonzero);
        if mode == 0 {
            self.bin(CHROMA_PRED + inc, 0);
            return;
        }
        self.bin(CHROMA_PRED + inc, 1);
        if mode == 1 {
            self.bin(CHROMA_PRED + 3, 0);
            return;
        }
        self.bin(CHROMA_PRED + 3, 1);
        if mode == 2 {
            self.bin(CHROMA_PRED + 3, 0);
        } else {
            self.bin(CHROMA_PRED + 3, 1); // cMax reached, no terminating zero
        }
    }

    /// `coded_block_pattern`: four luma bins then two chroma bins.
    ///
    /// `luma_inc[b]` and the chroma increments come from the neighbouring
    /// blocks — note the luma sense is inverted relative to everything else:
    /// a neighbour that *has* coefficients contributes zero.
    pub fn coded_block_pattern(
        &mut self,
        cbp_luma: u32,
        cbp_chroma: u32,
        luma_inc: [usize; 4],
        chroma_inc0: usize,
        chroma_inc1: usize,
    ) {
        for (b, &inc) in luma_inc.iter().enumerate() {
            self.bin(CBP_LUMA + inc, (cbp_luma >> b) & 1);
        }
        self.bin(CBP_CHROMA + chroma_inc0, u32::from(cbp_chroma != 0));
        if cbp_chroma != 0 {
            self.bin(CBP_CHROMA + 4 + chroma_inc1, u32::from(cbp_chroma == 2));
        }
    }

    /// `mb_qp_delta`, unary over the signed-to-unsigned mapping (§9.3.2.7).
    ///
    /// The first bin's context asks only whether the *previous* macroblock
    /// changed the quantizer — a picture that adapts everywhere and one that
    /// never does are cheap for opposite reasons.
    pub fn mb_qp_delta(&mut self, delta: i32, prev_nonzero: bool) {
        let code = if delta <= 0 {
            (-2 * delta) as u32
        } else {
            (2 * delta - 1) as u32
        };
        for k in 0..code {
            let inc = match k {
                0 => usize::from(prev_nonzero),
                1 => 2,
                _ => 3,
            };
            self.bin(MB_QP_DELTA + inc, 1);
        }
        let inc = match code {
            0 => usize::from(prev_nonzero),
            1 => 2,
            _ => 3,
        };
        self.bin(MB_QP_DELTA + inc, 0);
    }

    /// `coded_block_flag` for one residual block.
    pub fn coded_block_flag(&mut self, cat: usize, inc: usize, cbf: bool) {
        self.bin(CBF + CBF_CAT[cat] + inc, u32::from(cbf));
    }

    /// `residual_block_cabac` (§7.3.5.3.3) for a block already known to have
    /// coefficients. `levels` are in scan order.
    ///
    /// Significance runs forward and stops at the last non-zero; the levels
    /// then run *backwards* from it, because the context for each magnitude
    /// depends on how many ones and how many larger values have been seen so
    /// far — information that only accumulates in that direction.
    pub fn residual_block(&mut self, levels: &[i32], cat: usize) {
        let max = levels.len();
        let last = levels
            .iter()
            .rposition(|&c| c != 0)
            .expect("caller checked coded_block_flag");

        for (i, &c) in levels.iter().enumerate().take(last + 1) {
            if i == max - 1 {
                break; // the final position's significance is implied
            }
            let inc = sig_inc(cat, i);
            self.bin(SIG + SIG_CAT[cat] + inc, u32::from(c != 0));
            if c != 0 {
                self.bin(LAST_SIG + SIG_CAT[cat] + inc, u32::from(i == last));
                if i == last {
                    break;
                }
            }
        }

        let mut num_eq1 = 0u32;
        let mut num_gt1 = 0u32;
        for &c in levels[..=last].iter().rev() {
            if c == 0 {
                continue;
            }
            let abs = c.unsigned_abs();
            let inc0 = if num_gt1 != 0 {
                0
            } else {
                4.min(1 + num_eq1) as usize
            };
            let inc_rest = 5 + (if cat == CAT_CHROMA_DC { 3 } else { 4 }).min(num_gt1) as usize;
            self.abs_level_minus1(cat, abs - 1, inc0, inc_rest);
            self.bypass(u32::from(c < 0));
            if abs == 1 {
                num_eq1 += 1;
            } else {
                num_gt1 += 1;
            }
        }
    }

    /// `coeff_abs_level_minus1`: UEG0 with `uCoff = 14` — a context-coded
    /// truncated-unary prefix, then a bypass Exp-Golomb tail for the rare
    /// large magnitudes.
    fn abs_level_minus1(&mut self, cat: usize, v: u32, inc0: usize, inc_rest: usize) {
        let base = ABS_LEVEL + ABS_CAT[cat];
        let prefix = v.min(14);
        for k in 0..prefix {
            self.bin(base + if k == 0 { inc0 } else { inc_rest }, 1);
        }
        if prefix < 14 {
            self.bin(base + if prefix == 0 { inc0 } else { inc_rest }, 0);
            return;
        }
        // Exp-Golomb order 0 of the excess, all bypass.
        let mut rest = v - 14;
        let mut k = 0u32;
        while rest >= (1 << k) {
            self.bypass(1);
            rest -= 1 << k;
            k += 1;
        }
        self.bypass(0);
        for b in (0..k).rev() {
            self.bypass((rest >> b) & 1);
        }
    }
}

/// `ctxIdxInc` for `significant_coeff_flag` / `last_significant_coeff_flag`.
///
/// Every category keys off the scan position directly except chroma DC, whose
/// four coefficients share three contexts.
fn sig_inc(cat: usize, scan_pos: usize) -> usize {
    if cat == CAT_CHROMA_DC {
        scan_pos.min(2)
    } else {
        scan_pos
    }
}

/// Compile-time layout checks.
///
/// An I slice must never index a context the initialisation table leaves
/// undefined — `ctxIdx` 11..=59 are P/B-only and hold `(0, 0)` here. A stray
/// derivation would still produce a decodable-looking stream, because the
/// arithmetic coder does not object to a wrong probability; it would just
/// decode to the wrong thing. So the check has to be on the index.
const _: () = assert!(MB_TYPE_I + 7 <= 10, "I-slice mb_type stays inside 3..=10");
const _: () = assert!(MB_QP_DELTA >= 60 && CHROMA_PRED >= 60);
const _: () = assert!(PREV_INTRA_FLAG >= 60 && REM_INTRA_MODE >= 60);
const _: () = assert!(CBP_LUMA >= 60 && CBP_CHROMA >= 60 && CBF >= 60);
const _: () = assert!(SIG >= 60 && LAST_SIG >= 60 && ABS_LEVEL >= 60);

/// The three context blocks tile end to end; one slipped offset would silently
/// share a probability state between two unrelated block types.
const _: () = assert!(CBF + CBF_CAT[4] + 3 == 104, "coded_block_flag ends at 104");
const _: () = assert!(SIG + SIG_CAT[4] + 13 == 165, "significant_coeff_flag ends at 165");
const _: () = assert!(LAST_SIG + SIG_CAT[4] + 13 == 226, "last_significant ends at 226");
const _: () = assert!(ABS_LEVEL + ABS_CAT[4] + 9 == 275, "coeff_abs_level ends at 275");

#[cfg(test)]
mod tests {
    use super::*;

    /// Every significance category needs one context per scan position it can
    /// report, and chroma DC needs three.
    #[test]
    fn significance_categories_get_the_span_they_need() {
        for (cat, span) in [(0usize, 15usize), (1, 14), (2, 15), (3, 3), (4, 14)] {
            let next = SIG_CAT.get(cat + 1).copied().unwrap_or(SIG_CAT[4] + 14);
            assert_eq!(next - SIG_CAT[cat], span, "category {cat} span");
        }
    }

    /// `sig_inc` must stay inside its category's span for every position the
    /// category can produce.
    #[test]
    fn significance_context_stays_in_its_category() {
        for (cat, max_coeff) in [(0usize, 16usize), (1, 15), (2, 16), (3, 4), (4, 15)] {
            let limit = SIG_CAT.get(cat + 1).copied().unwrap_or(SIG_CAT[4] + 14);
            for pos in 0..max_coeff - 1 {
                let idx = SIG_CAT[cat] + sig_inc(cat, pos);
                assert!(idx < limit, "category {cat} position {pos} -> {idx}");
            }
        }
    }
}
