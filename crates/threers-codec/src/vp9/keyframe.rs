//! A minimal, decodable VP9 intra keyframe (profile 0, 8-bit 4:2:0).
//!
//! This is the VP9 analog of the HEVC path's first `I_PCM` frame: it doesn't
//! compress anything interesting yet, but it drives the *entire* pipeline —
//! uncompressed header bit-packing, the bool-coded compressed header, and the
//! bool-coded tile/partition/mode syntax — end to end, so a real decoder (ffmpeg
//! / libvpx) accepts it. The frame is a single 64×64 superblock coded as one
//! `PARTITION_NONE` block with `skip = 1` and `DC_PRED`: with no neighbors DC
//! prediction is `128`, the loop filter is off, and there is no residual, so the
//! decoded picture is solid mid-gray. Prediction-only multi-block coding
//! ([`crate::vp9::encoder`]) builds on this scaffolding; residual
//! transform tokens are the next layer.

use crate::bitstream::BitWriter;
use crate::vp9::bool_coder::BoolEncoder;
use crate::vp9::tables::{
    DC_PRED, DEFAULT_SKIP_PROBS, INTRA_MODE_TREE, KF_PARTITION_PROBS, KF_UV_MODE_PROB_DC,
    KF_Y_MODE_PROB_DC_DC, PARTITION_NONE, PARTITION_TREE,
};

const FRAME_SYNC_CODE: u32 = 0x0049_8342;
const CS_BT_601: u32 = 1;

/// Encode a solid-gray VP9 intra keyframe of `width × height` (currently only
/// 64×64 — a single superblock — is supported).
///
/// Returns the raw VP9 frame bitstream; pair it with [`crate::webm`] (or
/// any VP9 container) to produce a playable file.
pub fn encode_intra_gray(width: u32, height: u32) -> Vec<u8> {
    assert!(
        width == 64 && height == 64,
        "encode_intra_gray currently supports a single 64x64 superblock only"
    );

    let compressed = compressed_header();
    let tile = tile_data();

    let mut wb = BitWriter::new();
    write_uncompressed_header(&mut wb, width, height, compressed.len() as u32);
    let mut frame = wb.finish(); // byte-aligned

    frame.extend_from_slice(&compressed);
    frame.extend_from_slice(&tile);
    crate::vp9::finalize_frame(frame)
}

/// The uncompressed frame header (VP9 spec §6.2), bit-packed, keyframe / profile
/// 0 / 8-bit 4:2:0. Ends byte-aligned (the `BitWriter` zero-pads on `finish`).
fn write_uncompressed_header(w: &mut BitWriter, width: u32, height: u32, header_size: u32) {
    w.write_bits(2, 2); // frame_marker
    w.write_bit(0); // profile_low_bit
    w.write_bit(0); // profile_high_bit  -> Profile 0
    w.write_bit(0); // show_existing_frame
    w.write_bit(0); // frame_type = KEY_FRAME
    w.write_bit(1); // show_frame
    w.write_bit(0); // error_resilient_mode

    w.write_bits(FRAME_SYNC_CODE, 24); // frame_sync_code

    // color_config (profile 0 -> 8-bit, subsampling 4:2:0 implied)
    w.write_bits(CS_BT_601, 3); // color_space
    w.write_bit(0); // color_range = studio

    // frame_size
    w.write_bits(width - 1, 16);
    w.write_bits(height - 1, 16);
    // render_size
    w.write_bit(0); // render_and_frame_size_different

    // (keyframe: refresh_frame_flags = 0xFF, implied)
    // error_resilient_mode == 0:
    w.write_bit(0); // refresh_frame_context
    w.write_bit(1); // frame_parallel_decoding_mode
    w.write_bits(0, 2); // frame_context_idx (reset to 0 for intra)

    // loop_filter_params (filter off)
    w.write_bits(0, 6); // loop_filter_level
    w.write_bits(0, 3); // loop_filter_sharpness
    w.write_bit(0); // loop_filter_delta_enabled

    // quantization_params
    w.write_bits(128, 8); // base_q_idx
    w.write_bit(0); // delta_q_y_dc coded
    w.write_bit(0); // delta_q_uv_dc coded
    w.write_bit(0); // delta_q_uv_ac coded

    // segmentation_params
    w.write_bit(0); // segmentation_enabled

    // tile_info: for a 1-superblock frame min == max log2 tile cols == 0, so no
    // column bits are coded; a single tile-rows bit remains.
    w.write_bit(0); // tile_rows_log2

    w.write_bits(header_size, 16); // header_size_in_bytes
}

/// The compressed header (VP9 spec §6.3) for an intra keyframe: `tx_mode =
/// ONLY_4X4`, no coefficient-probability updates, no skip-probability updates.
/// All the inter-frame syntax is skipped because `FrameIsIntra == 1`.
fn compressed_header() -> Vec<u8> {
    let mut e = BoolEncoder::new();
    e.put_literal(0, 2); // read_tx_mode: tx_mode = ONLY_4X4
    e.put_bit(false); // read_coef_probs: TX_4X4 update_probs = 0
    for _ in 0..3 {
        e.put_bool(252, false); // read_skip_prob: diff_update_prob (no update)
    }
    e.finish()
}

/// The single tile's bool-coded data: one 64×64 `PARTITION_NONE` block with
/// `skip = 1` and `DC_PRED` luma/chroma modes.
fn tile_data() -> Vec<u8> {
    let mut e = BoolEncoder::new();

    // decode_partition(0, 0, BLOCK_64X64): both halves fit -> full partition
    // tree; ctx 12 is the 64x64 "neither neighbor split" context.
    write_tree(
        &mut e,
        &PARTITION_TREE,
        &KF_PARTITION_PROBS[12],
        PARTITION_NONE,
    );

    // intra_frame_mode_info for the 64x64 block:
    e.put_bool(DEFAULT_SKIP_PROBS[0], true); // read_skip: skip = 1 (ctx 0)
                                             // read_tx_size: not coded (tx_mode == ONLY_4X4).
    write_tree(&mut e, &INTRA_MODE_TREE, &KF_Y_MODE_PROB_DC_DC, DC_PRED); // y mode
    write_tree(&mut e, &INTRA_MODE_TREE, &KF_UV_MODE_PROB_DC, DC_PRED); // uv mode

    e.finish()
}

/// Encode `token` through a VP9 token tree: walk from the root to the leaf
/// `-token`, emitting one bool per node with that node's probability.
fn write_tree(e: &mut BoolEncoder, tree: &[i8], probs: &[u8], token: i8) {
    let path = tree_path(tree, 0, token).expect("token not in tree");
    for (node, bit) in path {
        e.put_bool(probs[node >> 1], bit);
    }
}

/// The root→leaf path to `token` as `(node_index, bit)` steps, or `None` if the
/// subtree rooted at `node` does not contain it.
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
    use crate::vp9::tables::{TM_PRED, V_PRED};

    #[test]
    fn tree_paths_are_correct() {
        // DC_PRED is the first leaf: a single `0` at the root.
        assert_eq!(
            tree_path(&INTRA_MODE_TREE, 0, DC_PRED).unwrap(),
            vec![(0, false)]
        );
        // TM_PRED is the other child of the root: `1` then `0`.
        assert_eq!(
            tree_path(&INTRA_MODE_TREE, 0, TM_PRED).unwrap(),
            vec![(0, true), (2, false)]
        );
        // V_PRED: `1, 1, 0`.
        assert_eq!(
            tree_path(&INTRA_MODE_TREE, 0, V_PRED).unwrap(),
            vec![(0, true), (2, true), (4, false)]
        );
        assert_eq!(
            tree_path(&PARTITION_TREE, 0, PARTITION_NONE).unwrap(),
            vec![(0, false)]
        );
    }

    #[test]
    fn frame_has_plausible_shape() {
        let frame = encode_intra_gray(64, 64);
        // Frame marker (top two bits of byte 0) == 0b10.
        assert_eq!(frame[0] >> 6, 0b10);
        // Header, compressed header, and tile data are all present.
        assert!(frame.len() > 12, "frame unexpectedly tiny: {}", frame.len());
    }
}
