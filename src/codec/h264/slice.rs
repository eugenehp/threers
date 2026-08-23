//! IDR I-slice with `I_PCM` macroblocks (ISO/IEC 14496-10 §7.3.3, §7.3.4).
//!
//! The paired PPS selects CAVLC (`entropy_coding_mode_flag = 0`). Each
//! macroblock signals `mb_type = I_PCM` via `ue(25)`, aligns to a byte boundary,
//! and writes raw PCM samples — no CABAC engine required.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};
use crate::codec::h264::params::{H264Config, MB_SIZE};

/// A planar picture padded to the coded (MB-aligned) size.
pub struct PaddedYuv<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub coded_width: u32,
    pub coded_height: u32,
}

/// `mb_type` for I_PCM in an I slice (Table 7-11).
const MB_TYPE_I_PCM: u32 = 25;

/// Encode the whole picture as one IDR slice referencing PPS 0.
pub fn idr_slice_rbsp(cfg: &H264Config, yuv: &PaddedYuv) -> Vec<u8> {
    let mut w = BitWriter::new();
    // ---- slice_header (Exp-Golomb) ----
    w.write_ue(0); // first_mb_in_slice
    w.write_ue(7); // slice_type coded as 7 → I (2)
    w.write_ue(0); // pic_parameter_set_id
    w.write_bits(0, 4); // frame_num (log2_max_frame_num_minus4 = 4)
    w.write_ue(0); // idr_pic_id
                   // dec_ref_pic_marking() — required when nal_ref_idc != 0 (IDR)
    w.flag(false); // no_output_of_prior_pics_flag
    w.flag(false); // long_term_reference_flag
                   // ref_pic_list_modification() absent for I slices (slice_type % 5 == 2)
    w.write_se(0); // slice_qp_delta → SliceQPY = 26 + pic_init_qp_minus26

    // ---- slice_data (CAVLC) ----
    let cw = yuv.coded_width;
    let ch = yuv.coded_height;
    let cwc = cw / 2;
    let half = MB_SIZE / 2;
    let (nx, ny) = (cw / MB_SIZE, ch / MB_SIZE);

    for mb_y in 0..ny {
        for mb_x in 0..nx {
            let cx = mb_x * MB_SIZE;
            let cy = mb_y * MB_SIZE;
            w.write_ue(MB_TYPE_I_PCM);
            w.align_zero(); // pcm_alignment_zero_bit*
            emit_block(&mut w, yuv.y, cw, cx, cy, MB_SIZE);
            let (ccx, ccy) = (cx / 2, cy / 2);
            emit_block(&mut w, yuv.u, cwc, ccx, ccy, half);
            emit_block(&mut w, yuv.v, cwc, ccx, ccy, half);
        }
    }

    let _ = cfg;
    rbsp_trailing_bits(&mut w);
    w.finish()
}

#[inline]
fn emit_block(w: &mut BitWriter, plane: &[u8], stride: u32, x0: u32, y0: u32, size: u32) {
    for yy in 0..size {
        let row = ((y0 + yy) * stride + x0) as usize;
        for xx in 0..size as usize {
            w.write_byte(plane[row + xx]);
        }
    }
}
