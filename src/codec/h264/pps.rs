//! Write a PPS RBSP compatible with decoders (Baseline profile, CAVLC for I_PCM).
//!
//! With `entropy_coding_mode_flag = 0`, each macroblock's `mb_type` is coded as
//! Exp-Golomb — `ue(25)` for `I_PCM` — and raw PCM bytes follow without CABAC.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};

/// RBSP payload for PPS 0 (without NAL header).
///
/// The `I_PCM` path uses this. Its macroblocks are coded at `qP = 0`, where the
/// deblocking thresholds are zero and the filter is a no-op, so it has no reason
/// to spend bits signalling control over it.
pub fn write_pps() -> Vec<u8> {
    write_pps_inner(false, false)
}

/// As [`write_pps`], but with `deblocking_filter_control_present_flag` set so a
/// slice can turn the in-loop deblocking filter off.
///
/// The compressed path needs this. Deblocking is on by default and is *in-loop*:
/// the decoder filters block edges before the result becomes reference data, so
/// an encoder that does not reproduce it bit-exactly drifts from the decoder at
/// every edge — and more so at high QP, where the filter is stronger. Until it is
/// implemented, those slices disable it rather than silently disagree.
pub fn write_pps_deblock_control(cabac: bool) -> Vec<u8> {
    write_pps_inner(true, cabac)
}

fn write_pps_inner(deblock_control: bool, cabac: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(0); // pic_parameter_set_id
    w.write_ue(0); // seq_parameter_set_id
    w.flag(cabac); // entropy_coding_mode_flag
    w.flag(false); // bottom_field_pic_order_in_frame_present_flag
    w.write_ue(0); // num_slice_groups_minus1
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.flag(false); // weighted_pred_flag
    w.write_bits(0, 2); // weighted_bipred_idc
    w.write_se(0); // pic_init_qp_minus26 → SliceQPY = 26
    w.write_se(0); // pic_init_qs_minus26
    w.write_se(0); // chroma_qp_index_offset
    w.flag(deblock_control); // deblocking_filter_control_present_flag
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // redundant_pic_cnt_present_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}
