//! Write a PPS RBSP compatible with decoders (Baseline profile, CAVLC for I_PCM).
//!
//! With `entropy_coding_mode_flag = 0`, each macroblock's `mb_type` is coded as
//! Exp-Golomb — `ue(25)` for `I_PCM` — and raw PCM bytes follow without CABAC.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};

/// RBSP payload for PPS 0 (without NAL header).
pub fn write_pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(0); // pic_parameter_set_id
    w.write_ue(0); // seq_parameter_set_id
    w.flag(false); // entropy_coding_mode_flag = 0 (CAVLC)
    w.flag(false); // bottom_field_pic_order_in_frame_present_flag
    w.write_ue(0); // num_slice_groups_minus1
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.flag(false); // weighted_pred_flag
    w.write_bits(0, 2); // weighted_bipred_idc
    w.write_se(0); // pic_init_qp_minus26 → SliceQPY = 26
    w.write_se(0); // pic_init_qs_minus26
    w.write_se(0); // chroma_qp_index_offset
    w.flag(false); // deblocking_filter_control_present_flag
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // redundant_pic_cnt_present_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}
