//! IDR I-slice with `I_PCM` coding units (H.265 §7.3.6, §7.3.8).
//!
//! This is the milestone-3 slice layer: every CTB is a single `PART_2Nx2N`
//! intra CU whose `pcm_flag` is 1, so the CU carries its samples verbatim. No
//! intra prediction, transform, quantization, or residual coding is exercised —
//! the point is a *conformant, decodable* stream that proves the whole pipeline
//! (params → slice header → CABAC control bins → PCM raw bytes → NAL) before
//! real compression lands.
//!
//! The only CABAC-modeled bin per CTB is `part_mode`; `pcm_flag` and
//! `end_of_slice_segment_flag` are termination decisions. After each PCM CU the
//! arithmetic engine is flushed, the stream byte-aligned, raw samples written,
//! and the engine re-initialized — exactly the dance a decoder mirrors.

use crate::bitstream::BitWriter;
use crate::hevc::cabac::{CabacEncoder, CtxModel};
use crate::hevc::params::{HevcConfig, CTB_SIZE};

/// `initValue` for the I-slice `part_mode` context (H.265 Table 9-11, initType 0).
const PART_MODE_INIT_VALUE: u8 = 184;

/// A planar picture already padded to the coded (CTB-aligned) size.
///
/// `y` is `coded_width × coded_height`. For 4:2:0, `u`/`v` are each half that in
/// both dimensions; for monochrome (per [`HevcConfig::has_chroma`]) they are
/// unused and may be empty.
pub struct PaddedYuv<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub coded_width: u32,
    pub coded_height: u32,
}

/// Encode the whole picture as one IDR I-slice referencing PPS 0.
pub fn slice_segment_rbsp(cfg: &HevcConfig, yuv: &PaddedYuv) -> Vec<u8> {
    slice_segment_rbsp_pps(cfg, yuv, 0)
}

/// Encode the whole picture as one IDR I-slice referencing `pps_id`, returning
/// the slice-segment-layer RBSP (byte-aligned; ready to wrap in a NAL unit).
pub fn slice_segment_rbsp_pps(cfg: &HevcConfig, yuv: &PaddedYuv, pps_id: u32) -> Vec<u8> {
    // ---- slice_segment_header() (§7.3.6.1) ----
    let mut h = BitWriter::new();
    h.flag(true); // first_slice_segment_in_pic_flag
    h.flag(false); // no_output_of_prior_pics_flag (IDR is IRAP)
    h.write_ue(pps_id); // slice_pic_parameter_set_id
                        // first slice → no dependent_slice_segment_flag / slice_segment_address.
    h.write_ue(2); // slice_type = I
                   // output_flag_present_flag = 0, SAO off, I-slice → nothing else until QP.
    h.write_se(0); // slice_qp_delta → SliceQpY = 26
                   // Deblocking disabled at PPS and no override; loop-filter-across-slices off.
                   // byte_alignment(): a 1 bit then zero-fill to the next byte.
    h.write_bit(1); // alignment_bit_equal_to_one
    let mut rbsp = h.finish();

    // ---- slice_segment_data() (§7.3.8.1), CABAC ----
    let cw = yuv.coded_width;
    let ch = yuv.coded_height;
    let cwc = cw / 2; // chroma stride
    let half = CTB_SIZE / 2;

    let mut cabac = CabacEncoder::new();
    let mut part_ctx = CtxModel::init(PART_MODE_INIT_VALUE, cfg.qp);
    let (nx, ny) = cfg.ctbs();
    let total = nx * ny;
    let mut coded = 0u32;

    for cy in (0..ch).step_by(CTB_SIZE as usize) {
        for cx in (0..cw).step_by(CTB_SIZE as usize) {
            // coding_quadtree(): split_cu_flag is absent (log2CbSize == MinCbLog2SizeY).
            // coding_unit(): I-slice ⇒ intra; part_mode "1" = PART_2Nx2N.
            cabac.encode_bin(&mut part_ctx, 1);
            // pcm_flag = 1 — a termination decision that flushes the engine.
            cabac.encode_terminate(1);

            // pcm_alignment_zero_bit* then pcm_sample() as raw bytes. pcm_sample()
            // codes chroma only when ChromaArrayType != 0 (§7.3.8.7).
            let w = cabac.writer_mut();
            w.align_zero();
            emit_block(w, yuv.y, cw, cx, cy, CTB_SIZE); // luma 16×16
            if cfg.has_chroma() {
                let (ccx, ccy) = (cx / 2, cy / 2);
                emit_block(w, yuv.u, cwc, ccx, ccy, half); // Cb 8×8
                emit_block(w, yuv.v, cwc, ccx, ccy, half); // Cr 8×8
            }

            // Re-initialize the arithmetic engine after the PCM interruption.
            cabac.reinit();

            // end_of_slice_segment_flag (termination): 1 only on the last CTB.
            coded += 1;
            cabac.encode_terminate((coded == total) as u32);
        }
    }

    rbsp.extend_from_slice(&cabac.finish());
    rbsp
}

/// Write a `size × size` block of `plane` (raster order, stride `stride`) at
/// luma/chroma origin `(x0, y0)` as raw bytes.
#[inline]
fn emit_block(w: &mut BitWriter, plane: &[u8], stride: u32, x0: u32, y0: u32, size: u32) {
    for yy in 0..size {
        let row = ((y0 + yy) * stride + x0) as usize;
        for xx in 0..size as usize {
            w.write_byte(plane[row + xx]);
        }
    }
}
