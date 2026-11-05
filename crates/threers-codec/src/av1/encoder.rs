//! Key frames: headers, block decisions and the coefficient syntax.

use std::f32::consts::PI;

use super::bitwriter::{obu, BitWriter};
use super::range::RangeEncoder;
use crate::decode::av1::cdf::CdfContext;
use crate::decode::av1::env::{gather_left_partition_prob, gather_top_partition_prob};
use crate::decode::av1::itx::itxfm_add;
use crate::decode::av1::levels::*;
use crate::decode::av1::obu::Layout;
use crate::decode::av1::recon::{get_dc_sign_ctx, get_lo_ctx, get_skip_ctx};
use crate::decode::av1::tables::{
    AL_PART_CTX, BLOCK_SIZES, DQ_TBL, LO_CTX_OFFSETS, PARTITION_TYPE_COUNT, SCANS, TXFM_DIMENSIONS,
    TX_TYPES_PER_SET,
};

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME: u8 = 6;

/// An AV1 encoder for 8-bit 4:2:0 pictures whose sides are multiples of 8, so
/// that every block lies inside the frame. Each picture becomes one temporal
/// unit, independently decodable; see [the module docs](super).
pub struct Av1Encoder {
    width: usize,
    height: usize,
    qidx: u8,
    cdf: Box<CdfContext>,
    /// Orthonormal DCT rows, by log2 of the size.
    bases: [Vec<f32>; 6],
    wrote_sequence: bool,
}

/// One frame's state: the reconstruction, and what later blocks read of
/// earlier ones.
struct Frame<'a> {
    src: [&'a [u8]; 3],
    rec: [Vec<u16>; 3],
    strides: [usize; 3],
    enc: RangeEncoder,
    /// Per 8 luma pixels, across the frame and down the superblock.
    above_part: Vec<u8>,
    left_part: [u8; 16],
    /// Per 4 luma pixels.
    above_skip: Vec<u8>,
    left_skip: Vec<u8>,
    /// Coefficient contexts, luma then the two chromas, per 4 samples.
    above_coef: [Vec<u8>; 3],
    left_coef: [Vec<u8>; 3],
    w4: usize,
    h4: usize,
}

impl Av1Encoder {
    /// An encoder at quantiser `qidx`, 1 (nearly lossless) to 255.
    pub fn new(width: usize, height: usize, qidx: u8) -> Result<Self, String> {
        if width == 0 || height == 0 || width % 8 != 0 || height % 8 != 0 {
            return Err(format!(
                "AV1 encoder needs sides that are multiples of 8, not {width}x{height}"
            ));
        }
        if width > 65536 || height > 65536 {
            return Err("AV1 frame is too large".into());
        }
        if qidx == 0 {
            return Err("qidx 0 would be lossless, which this encoder does not do".into());
        }
        let bases = std::array::from_fn(|log2| {
            let n = 1usize << log2;
            let mut m = vec![0f32; n * n];
            for k in 0..n {
                let scale = if k == 0 {
                    (1.0 / n as f32).sqrt()
                } else {
                    (2.0 / n as f32).sqrt()
                };
                for i in 0..n {
                    m[k * n + i] =
                        scale * (PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
                }
            }
            m
        });
        Ok(Av1Encoder {
            width,
            height,
            qidx,
            cdf: CdfContext::default_for(qidx),
            bases,
            wrote_sequence: false,
        })
    }

    /// One picture as planes with no padding between rows, chroma at half
    /// size each way. Returns a temporal unit, with the sequence header on
    /// the first.
    pub fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<u8>, String> {
        let (w, h) = (self.width, self.height);
        if y.len() != w * h || u.len() != w * h / 4 || v.len() != w * h / 4 {
            return Err("AV1 encoder: plane sizes do not match the frame".into());
        }
        let mut out = Vec::new();
        obu(OBU_TEMPORAL_DELIMITER, &[], &mut out);
        if !self.wrote_sequence {
            obu(OBU_SEQUENCE_HEADER, &self.sequence_header(), &mut out);
            self.wrote_sequence = true;
        }
        let mut payload = self.frame_header();
        payload.extend(self.tile(y, u, v));
        obu(OBU_FRAME, &payload, &mut out);
        Ok(out)
    }

    fn sequence_header(&self) -> Vec<u8> {
        let mut b = BitWriter::default();
        b.put(0, 3); // seq_profile: Main
        b.flag(false); // still_picture
        b.flag(false); // reduced_still_picture_header
        b.flag(false); // timing_info_present_flag
        b.flag(false); // initial_display_delay_present_flag
        b.put(0, 5); // operating_points_cnt_minus_1
        b.put(0, 12); // operating_point_idc
        b.put(31, 5); // seq_level_idx: no level claimed
        b.flag(false); // seq_tier
        b.put(15, 4); // frame_width_bits_minus_1
        b.put(15, 4); // frame_height_bits_minus_1
        b.put(self.width as u32 - 1, 16);
        b.put(self.height as u32 - 1, 16);
        b.flag(false); // frame_id_numbers_present_flag
        b.flag(false); // use_128x128_superblock
        b.flag(false); // enable_filter_intra
        b.flag(false); // enable_intra_edge_filter
        b.flag(false); // enable_interintra_compound
        b.flag(false); // enable_masked_compound
        b.flag(false); // enable_warped_motion
        b.flag(false); // enable_dual_filter
        b.flag(false); // enable_order_hint
        b.flag(false); // seq_choose_screen_content_tools
        b.flag(false); // seq_force_screen_content_tools
        b.flag(false); // enable_superres
        b.flag(false); // enable_cdef
        b.flag(false); // enable_restoration
        b.flag(false); // high_bitdepth
        b.flag(false); // mono_chrome
        b.flag(false); // color_description_present_flag
        b.flag(false); // color_range: studio
        b.put(0, 2); // chroma_sample_position
        b.flag(false); // separate_uv_delta_q
        b.flag(false); // film_grain_params_present
        b.trailing()
    }

    fn frame_header(&self) -> Vec<u8> {
        let mut b = BitWriter::default();
        b.flag(false); // show_existing_frame
        b.put(0, 2); // frame_type: KEY_FRAME
        b.flag(true); // show_frame
        b.flag(true); // disable_cdf_update
        b.flag(false); // frame_size_override_flag
        b.flag(false); // render_and_frame_size_different
                       // tile_info: one tile.
        b.flag(true); // uniform_tile_spacing_flag
        let tile_log2 = |n: usize| {
            let mut k = 0;
            while (1usize << k) < n {
                k += 1;
            }
            k
        };
        if tile_log2(self.width.div_ceil(64).min(64)) > 0 {
            b.flag(false); // increment_tile_cols_log2
        }
        if tile_log2(self.height.div_ceil(64).min(64)) > 0 {
            b.flag(false); // increment_tile_rows_log2
        }
        // quantization_params
        b.put(self.qidx as u32, 8);
        b.flag(false); // DeltaQYDc
        b.flag(false); // DeltaQUDc
        b.flag(false); // DeltaQUAc
        b.flag(false); // using_qmatrix
        b.flag(false); // segmentation_enabled
        b.flag(false); // delta_q_present
                       // loop_filter_params: off.
        b.put(0, 6);
        b.put(0, 6);
        b.put(0, 3); // loop_filter_sharpness
        b.flag(false); // loop_filter_delta_enabled
        b.flag(false); // tx_mode_select: TX_MODE_LARGEST
        b.flag(true); // reduced_tx_set
        b.aligned()
    }

    fn tile(&self, y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
        let (w, h) = (self.width, self.height);
        let (w4, h4) = (w / 4, h / 4);
        let mut f = Frame {
            src: [y, u, v],
            rec: [vec![0; w * h], vec![0; w * h / 4], vec![0; w * h / 4]],
            strides: [w, w / 2, w / 2],
            enc: RangeEncoder::new(),
            above_part: vec![0; w4 / 2 + 16],
            left_part: [0; 16],
            above_skip: vec![0; w4 + 32],
            left_skip: vec![0; 32],
            above_coef: [
                vec![0x40; w4 + 32],
                vec![0x40; w4 + 32],
                vec![0x40; w4 + 32],
            ],
            left_coef: [vec![0x40; 32], vec![0x40; 32], vec![0x40; 32]],
            w4,
            h4,
        };
        for by in (0..h4).step_by(16) {
            f.left_part = [0; 16];
            f.left_skip.fill(0);
            for l in &mut f.left_coef {
                l.fill(0x40);
            }
            for bx in (0..w4).step_by(16) {
                self.partition(&mut f, BL_64X64, bx, by);
            }
        }
        f.enc.finish()
    }

    /// `decode_sb`, with blocks of 64 always split and any crossing the frame
    /// edge split until they do not.
    fn partition(&self, f: &mut Frame, bl: usize, bx: usize, by: usize) {
        let hsz = 16usize >> bl;
        let have_h = f.w4 > bx + hsz;
        let have_v = f.h4 > by + hsz;
        if !have_h && !have_v {
            return self.partition(f, bl + 1, bx, by);
        }
        let by8 = (by & 31) >> 1;
        let shift = 4 - bl;
        let ctx = ((f.above_part[bx >> 1] >> shift) & 1) as usize
            + ((((f.left_part[by8] >> shift) & 1) as usize) << 1);
        let cdf = &self.cdf.m.partition[bl][ctx];
        let bp;
        if have_h && have_v {
            let fits = bx + 2 * hsz <= f.w4 && by + 2 * hsz <= f.h4;
            let split = bl < BL_8X8 && (bl < BL_32X32 || !fits || self.busy(f, bl, bx, by));
            bp = if split {
                PARTITION_SPLIT
            } else {
                PARTITION_NONE
            };
            f.enc.symbol(cdf, PARTITION_TYPE_COUNT[bl] as usize, bp);
            if split {
                self.partition(f, bl + 1, bx, by);
                self.partition(f, bl + 1, bx + hsz, by);
                self.partition(f, bl + 1, bx, by + hsz);
                self.partition(f, bl + 1, bx + hsz, by + hsz);
            } else {
                self.block(f, bl, bx, by);
            }
        } else if have_h {
            f.enc.bool(gather_top_partition_prob(cdf, bl), true);
            bp = PARTITION_SPLIT;
            self.partition(f, bl + 1, bx, by);
            self.partition(f, bl + 1, bx + hsz, by);
        } else {
            f.enc.bool(gather_left_partition_prob(cdf, bl), true);
            bp = PARTITION_SPLIT;
            self.partition(f, bl + 1, bx, by);
            self.partition(f, bl + 1, bx, by + hsz);
        }
        if bp != PARTITION_SPLIT || bl == BL_8X8 {
            let n = hsz.min(16);
            f.above_part[bx >> 1..(bx >> 1) + n].fill(AL_PART_CTX[0][bl][bp]);
            f.left_part[by8..(by8 + n).min(16)].fill(AL_PART_CTX[1][bl][bp]);
        }
    }

    /// Whether a block has more detail than one prediction and transform suit:
    /// its halves differ, measured as luma variance against the quantiser.
    fn busy(&self, f: &Frame, bl: usize, bx: usize, by: usize) -> bool {
        let n = 8usize << (BL_8X8 - bl);
        let stride = f.strides[0];
        let (mut sum, mut sq) = (0u64, 0u64);
        for y in 0..n {
            let row = &f.src[0][(by * 4 + y) * stride + bx * 4..][..n];
            for &p in row {
                sum += p as u64;
                sq += p as u64 * p as u64;
            }
        }
        let count = (n * n) as u64;
        let variance = (sq - sum * sum / count) / count;
        let q = DQ_TBL[0][self.qidx as usize][1] as u64 / 8;
        variance > (4 * q * q).max(16)
    }

    /// One block: the mode syntax, then each plane's single transform.
    fn block(&self, f: &mut Frame, bl: usize, bx: usize, by: usize) {
        let bs = BLOCK_SIZES[bl][PARTITION_NONE][0] as usize;
        let log2 = 3 + BL_8X8 - bl;
        let txs = [log2 - 2, log2 - 3, log2 - 3];
        let mut coded = Vec::with_capacity(3);
        for plane in 0..3 {
            coded.push(self.transform(f, plane, txs[plane], bx, by));
        }
        let skip = coded.iter().all(|c| c.is_none());

        let n4 = 1usize << (log2 - 2);
        let ly = by & 31;
        let sctx = (f.above_skip[bx] + f.left_skip[ly]) as usize;
        f.enc.bool(self.cdf.m.skip[sctx][0] as u32, skip);
        f.above_skip[bx..bx + n4].fill(skip as u8);
        f.left_skip[ly..ly + n4].fill(skip as u8);
        f.enc
            .symbol(&self.cdf.kfym[0][0], N_INTRA_PRED_MODES - 1, DC_PRED);
        f.enc.symbol(
            &self.cdf.m.uv_mode[1][DC_PRED],
            N_UV_INTRA_PRED_MODES - 1,
            DC_PRED,
        );

        if skip {
            for plane in 0..3 {
                let (x, y, n) = if plane == 0 {
                    (bx, ly, n4)
                } else {
                    (bx / 2, ly / 2, n4 / 2)
                };
                f.above_coef[plane][x..x + n.max(1)].fill(0x40);
                f.left_coef[plane][y..y + n.max(1)].fill(0x40);
            }
            return;
        }
        for (plane, levels) in coded.into_iter().enumerate() {
            self.coefficients(f, plane, txs[plane], bs, bx, by, levels.as_deref());
        }
    }

    /// Predict a plane of the block and reconstruct it: DC prediction, forward
    /// DCT, quantisation and the decoder's own inverse. Returns the levels,
    /// indexed as the decoder's `cf`, or `None` if all are zero.
    fn transform(
        &self,
        f: &mut Frame,
        plane: usize,
        tx: usize,
        bx: usize,
        by: usize,
    ) -> Option<Vec<i32>> {
        let n = 4usize << tx;
        let shift_c = (plane != 0) as usize;
        let (x0, y0) = ((bx * 4) >> shift_c, (by * 4) >> shift_c);
        let stride = f.strides[plane];
        let rec = &mut f.rec[plane];

        let mut sum = 0u32;
        if y0 > 0 {
            sum += rec[(y0 - 1) * stride + x0..][..n]
                .iter()
                .map(|&v| v as u32)
                .sum::<u32>();
        }
        if x0 > 0 {
            sum += (0..n)
                .map(|i| rec[(y0 + i) * stride + x0 - 1] as u32)
                .sum::<u32>();
        }
        let count = n as u32 * ((y0 > 0) as u32 + (x0 > 0) as u32);
        let dc = if count == 0 {
            128
        } else {
            (sum + count / 2) / count
        };

        // Rows then columns of the orthonormal DCT.
        let log2 = tx + 2;
        let base = &self.bases[log2];
        let src = f.src[plane];
        let residual: Vec<f32> = (0..n * n)
            .map(|i| src[(y0 + i / n) * stride + x0 + i % n] as f32 - dc as f32)
            .collect();
        let mut rows = vec![0f32; n * n];
        for y in 0..n {
            for k in 0..n {
                rows[y * n + k] = (0..n).map(|x| residual[y * n + x] * base[k * n + x]).sum();
            }
        }
        let t_dim = TXFM_DIMENSIONS[tx];
        let dq_shift = (t_dim.ctx as i32 - 2).max(0);
        let dq = DQ_TBL[0][self.qidx as usize];
        // What the inverse needs for an orthonormal coefficient of 1.
        let gain = if tx == TX_32X32 { 4.0 } else { 8.0 };
        let mut levels = vec![0i32; n * n];
        let mut cf = vec![0i32; n * n];
        let mut any = false;
        let mut last = 0;
        for x in 0..n {
            for yf in 0..n {
                let c: f32 = (0..n).map(|y| rows[y * n + x] * base[yf * n + y]).sum();
                let q = dq[(x + yf != 0) as usize] as i32;
                let target = c * gain * (1 << dq_shift) as f32;
                let level = (target.abs() / q as f32 + 0.4) as i32;
                if level != 0 {
                    let rc = x * n + yf;
                    let value = (q * level) >> dq_shift;
                    levels[rc] = if c < 0.0 { -level } else { level };
                    cf[rc] = if c < 0.0 { -value } else { value };
                    any = true;
                }
            }
        }
        let scan = SCANS[tx];
        for (i, &rc) in scan.iter().enumerate() {
            if levels[rc as usize] != 0 {
                last = i as i32;
            }
        }
        for y in 0..n {
            rec[(y0 + y) * stride + x0..][..n].fill(dc as u16);
        }
        if !any {
            return None;
        }
        itxfm_add(
            rec,
            y0 * stride + x0,
            stride,
            &mut cf,
            last,
            tx,
            DCT_DCT,
            255,
        );
        Some(levels)
    }

    /// `decode_coefs` in reverse. Writes one transform's syntax and leaves its
    /// context behind.
    #[allow(clippy::too_many_arguments)]
    fn coefficients(
        &self,
        f: &mut Frame,
        plane: usize,
        tx: usize,
        bs: usize,
        bx: usize,
        by: usize,
        levels: Option<&[i32]>,
    ) {
        let chroma = plane != 0;
        let ci = chroma as usize;
        let t_dim = TXFM_DIMENSIONS[tx];
        let tctx = t_dim.ctx as usize;
        let (ax, ly) = if chroma {
            (bx / 2, (by & 31) / 2)
        } else {
            (bx, by & 31)
        };
        let mut ab = [0x40u8; 16];
        let mut lb = [0x40u8; 16];
        let w = t_dim.w as usize;
        let h = t_dim.h as usize;
        ab[..w].copy_from_slice(&f.above_coef[plane][ax..ax + w]);
        lb[..h].copy_from_slice(&f.left_coef[plane][ly..ly + h]);
        let sctx = get_skip_ctx(&t_dim, bs, &ab, &lb, chroma, Layout::I420);
        let coef = &self.cdf.coef;
        let enc = &mut f.enc;
        enc.bool(coef.skip[tctx][sctx][0] as u32, levels.is_none());
        let Some(levels) = levels else {
            f.above_coef[plane][ax..ax + w].fill(0x40);
            f.left_coef[plane][ly..ly + h].fill(0x40);
            return;
        };

        // Only luma below 32 pixels says which transform it used.
        if !chroma && (t_dim.max as usize) + 1 < TX_64X64 {
            let idx = TX_TYPES_PER_SET.iter().position(|&t| t == DCT_DCT).unwrap();
            enc.symbol(&self.cdf.m.txtp_intra2[t_dim.min as usize][DC_PRED], 4, idx);
        }

        let slw = (t_dim.lw as usize).min(TX_32X32);
        let slh = (t_dim.lh as usize).min(TX_32X32);
        let tx2dszctx = slw + slh;
        let scan = SCANS[tx];
        let eob = scan
            .iter()
            .rposition(|&rc| levels[rc as usize] != 0)
            .unwrap() as u32;

        let bin = if eob < 2 {
            eob
        } else {
            32 - eob.leading_zeros()
        };
        let ns = 4 + tx2dszctx;
        let eob_cdf: &[u16] = match tx2dszctx {
            0 => &coef.eob_bin_16[ci][0],
            1 => &coef.eob_bin_32[ci][0],
            2 => &coef.eob_bin_64[ci][0],
            3 => &coef.eob_bin_128[ci][0],
            4 => &coef.eob_bin_256[ci][0],
            5 => &coef.eob_bin_512[ci],
            _ => &coef.eob_bin_1024[ci],
        };
        enc.symbol(eob_cdf, ns, bin as usize);
        if bin > 1 {
            let hi = (eob >> (bin - 2)) & 1;
            enc.bool(coef.eob_hi_bit[tctx][ci][bin as usize][0] as u32, hi != 0);
            enc.bools(eob & ((1 << (bin - 2)) - 1), bin - 2);
        }

        let hi_ctx = tctx.min(3);
        let shift = slh as u32 + 2;
        let mask = (4u32 << slh) - 1;
        let stride = 4usize << slh;
        let offsets = &LO_CTX_OFFSETS[0];
        let token = |level: i32| (level.unsigned_abs()).min(15);
        let mut grid = vec![0u8; stride * ((4 << slw) + 2) + 64];
        let mut mag = 0u32;
        let at = |i: u32| {
            let rc = scan[i as usize] as u32;
            (rc as usize, rc >> shift, rc & mask)
        };
        if eob != 0 {
            let ctx0 = 1 + (eob > 2 << tx2dszctx) as usize + (eob > 4 << tx2dszctx) as usize;
            let (rc, x, y) = at(eob);
            let tok = token(levels[rc]);
            enc.symbol(
                &coef.eob_base_tok[tctx][ci][ctx0],
                2,
                tok.min(3) as usize - 1,
            );
            let mut level_tok = tok * 0x41;
            if tok >= 3 {
                let c = if (x | y) > 1 { 14 } else { 7 };
                enc.hi_tok(&coef.br_tok[hi_ctx][ci][c], tok);
                level_tok = tok + (3 << 6);
            }
            grid[x as usize * stride + y as usize] = level_tok as u8;
            for i in (1..eob).rev() {
                let (rc, x, mut y) = at(i);
                let lo = x as usize * stride + y as usize;
                let ctx = get_lo_ctx(
                    &grid,
                    lo,
                    TX_CLASS_2D,
                    &mut mag,
                    Some(offsets),
                    x,
                    y,
                    stride,
                ) as usize;
                y |= x;
                let tok = token(levels[rc]);
                enc.symbol(&coef.base_tok[tctx][ci][ctx], 3, tok.min(3) as usize);
                if tok >= 3 {
                    mag &= 63;
                    let c = (if y > 1 { 14 } else { 7 })
                        + if mag > 12 {
                            6
                        } else {
                            ((mag + 1) >> 1) as usize
                        };
                    enc.hi_tok(&coef.br_tok[hi_ctx][ci][c], tok);
                    grid[lo] = (tok + (3 << 6)) as u8;
                } else {
                    grid[lo] = (tok * 0x41) as u8;
                }
            }
            let tok = token(levels[0]);
            enc.symbol(&coef.base_tok[tctx][ci][0], 3, tok.min(3) as usize);
            if tok >= 3 {
                let m = (grid[1] as u32 + grid[stride] as u32 + grid[stride + 1] as u32) & 63;
                let c = if m > 12 { 6 } else { ((m + 1) >> 1) as usize };
                enc.hi_tok(&coef.br_tok[hi_ctx][ci][c], tok);
            }
        } else {
            let tok = token(levels[0]);
            enc.symbol(&coef.eob_base_tok[tctx][ci][0], 2, tok.min(3) as usize - 1);
            if tok >= 3 {
                enc.hi_tok(&coef.br_tok[hi_ctx][ci][0], tok);
            }
        }

        // Signs and what did not fit, DC first, then the rest in scan order.
        let mut cul = 0u32;
        let mut sign_level = 1u32 << 6;
        let dc = levels[0];
        if dc != 0 {
            let dsc = get_dc_sign_ctx(&t_dim, &ab, &lb);
            let negative = dc < 0;
            enc.bool(coef.dc_sign[ci][dsc][0] as u32, negative);
            sign_level = if negative { 0 } else { 2 << 6 };
            let m = dc.unsigned_abs();
            if m >= 15 {
                enc.golomb(m - 15);
            }
            cul = m;
        }
        for i in 1..=eob {
            let level = levels[scan[i as usize] as usize];
            if level != 0 {
                enc.bool_equi(level < 0);
                let m = level.unsigned_abs();
                if m >= 15 {
                    enc.golomb(m - 15);
                }
                cul = cul.saturating_add(m);
            }
        }
        let ctx = (cul.min(63) | sign_level) as u8;
        f.above_coef[plane][ax..ax + w].fill(ctx);
        f.left_coef[plane][ly..ly + h].fill(ctx);
    }
}
