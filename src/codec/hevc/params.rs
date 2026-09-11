//! HEVC high-level syntax: `profile_tier_level`, VPS, SPS, PPS (H.265 §7.3.2).
//!
//! Everything here writes the *minimal conformant* configuration the milestone-3
//! `I_PCM` encoder needs: Main profile, 8-bit 4:2:0, one layer, no temporal
//! scalability, in-loop filters (deblocking + SAO) disabled, and PCM enabled
//! with the CTB size pinned to the min CB size so no `split_cu_flag` is ever
//! coded (every CTB is a single CU).
//!
//! These are pure bit-writers over [`BitWriter`]; the structural round-trip test
//! parses them back, and the end-to-end test decodes the assembled stream with
//! an external decoder to confirm conformance.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};

/// Fixed configuration for the minimal PCM encoder. Sizes are luma samples.
#[derive(Clone, Copy, Debug)]
pub struct HevcConfig {
    /// Displayed picture width (any positive value; coded size is padded to a
    /// multiple of the CTB size and cropped back via the conformance window).
    pub width: u32,
    /// Displayed picture height.
    pub height: u32,
    /// Coded width, padded up to a multiple of [`CTB_SIZE`].
    pub coded_width: u32,
    /// Coded height, padded up to a multiple of [`CTB_SIZE`].
    pub coded_height: u32,
    /// `chroma_format_idc`: `1` = 4:2:0 (color), `0` = monochrome (4:0:0). The
    /// alpha auxiliary layer of a transparent video is monochrome.
    pub chroma_format_idc: u8,
    /// `general_level_idc` (level × 30). `180` = 6.0 (roomy default); the
    /// transparent path matches Apple's reference with `30` (level 1.0).
    pub level_idc: u8,
    /// Signal `video_full_range_flag = 1` via a minimal VUI. The alpha auxiliary
    /// layer needs this so its 0..255 luma is read as full-range opacity rather
    /// than expanded from limited `[16,235]`.
    pub full_range: bool,
    /// Enable I_PCM in the SPS. The lossless PCM path needs it; the compressed
    /// (transform-coded) path disables it.
    pub pcm_enabled: bool,
    /// Slice QP: CABAC context init, and (compressed path) the quantization step.
    pub qp: i32,
    /// `max_transform_hierarchy_depth_intra`: how many times an intra CU's
    /// transform tree may split. `0` pins one transform block per CU; `1` allows
    /// 16x16 to split into four 8x8.
    ///
    /// This is not just a transform-size knob. HEVC does intra prediction *per
    /// transform block* from already-reconstructed neighbours, so splitting also
    /// shortens the prediction distance — an 8x8 block predicts from the 8x8
    /// next to it rather than from the CTB edge 16 samples away.
    pub max_transform_hierarchy_depth_intra: u32,
    /// `log2` of the coding-tree-block edge.
    pub ctb_log2: u32,
    /// `log2` of the smallest coding block the quadtree may reach. Equal to
    /// [`HevcConfig::ctb_log2`] means no quadtree: one coding unit per CTB.
    pub min_cb_log2: u32,
    /// `log2` of the largest transform block. A coding block bigger than this
    /// has its transform tree split without signalling it.
    pub max_tb_log2: u32,
    /// Tile columns and rows. `(1, 1)` means the picture is one tile, which is
    /// to say untiled.
    ///
    /// Tiles break intra prediction and reset the entropy coder at their edges,
    /// so each one can be coded independently — and therefore concurrently.
    /// That costs bitrate in proportion to how much tile boundary the picture
    /// gains: measured on photographic content, 2x2 at 8K costs 0.7%, 2x2 at 4K
    /// costs 1.0%, and 4x4 at 4K costs 5.1%. Prefer few large tiles.
    pub tile_grid: (u32, u32),
}

/// A tile grid chosen from the picture size alone.
///
/// Tiles cost bitrate in proportion to how much boundary the picture gains, so
/// what matters is keeping each tile large. Measured on photographic content:
/// tiles of four megapixels or more cost under a tenth of a percent, a 2x2 split
/// of 1080p costs 0.43%, and splitting 360p four ways costs 5%.
///
/// Halving the longer side until a tile is at most four megapixels keeps every
/// tile close to that, which makes the grid `1x1` up to 1440p, `2x1` at 4K and
/// `4x2` at 8K. Deriving it from the resolution rather than from the core count
/// matters: the same input has to encode to the same bytes on every machine.
pub fn auto_tile_grid(width: u32, height: u32) -> (u32, u32) {
    const TARGET: u64 = 4_200_000; // ~4 megapixels per tile
    let (mut c, mut r) = (1u32, 1u32);
    while u64::from(width / c) * u64::from(height / r) > TARGET && c <= 8 && r <= 8 {
        if width / c >= height / r {
            c *= 2;
        } else {
            r *= 2;
        }
    }
    (c, r)
}

/// Default coding-tree-block edge in luma samples. The `I_PCM` path pins the
/// CTB to the min CB size so the quadtree never splits and each CTB carries
/// exactly one PCM CU; the compressed path overrides all three sizes via
/// [`HevcConfig::with_coding_tree`].
pub const CTB_SIZE: u32 = 16;
/// `log2` of [`CTB_SIZE`].
pub const CTB_LOG2: u32 = 4;

impl HevcConfig {
    /// Build a 4:2:0 (color) config for a `width × height` picture, padding the
    /// coded size up to whole CTBs.
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_chroma(width, height, 1)
    }

    /// Build a monochrome (4:0:0) config — used for the alpha auxiliary layer.
    pub fn new_monochrome(width: u32, height: u32) -> Self {
        Self::with_chroma(width, height, 0)
    }

    fn with_chroma(width: u32, height: u32, chroma_format_idc: u8) -> Self {
        let coded_width = width.div_ceil(CTB_SIZE) * CTB_SIZE;
        let coded_height = height.div_ceil(CTB_SIZE) * CTB_SIZE;
        Self {
            width,
            height,
            coded_width,
            coded_height,
            ctb_log2: CTB_LOG2,
            min_cb_log2: CTB_LOG2,
            max_tb_log2: CTB_LOG2,
            tile_grid: (1, 1),
            chroma_format_idc,
            level_idc: 180,
            full_range: false,
            pcm_enabled: true,
            qp: 26,
            max_transform_hierarchy_depth_intra: 0,
        }
    }

    /// Enable/disable I_PCM in the SPS (disable for the compressed path).
    pub fn with_pcm(mut self, enabled: bool) -> Self {
        self.pcm_enabled = enabled;
        self
    }

    /// Set `max_transform_hierarchy_depth_intra` (see the field).
    pub fn with_transform_depth(mut self, depth: u32) -> Self {
        self.max_transform_hierarchy_depth_intra = depth;
        self
    }

    /// Set `general_level_idc` (level × 30).
    pub fn with_level(mut self, level_idc: u8) -> Self {
        self.level_idc = level_idc;
        self
    }

    /// Signal full-range luma/chroma (needed for the alpha auxiliary layer).
    pub fn with_full_range(mut self, full_range: bool) -> Self {
        self.full_range = full_range;
        self
    }

    /// Whether this picture carries chroma planes.
    pub fn has_chroma(&self) -> bool {
        self.chroma_format_idc != 0
    }

    /// `(SubWidthC, SubHeightC)` — chroma subsampling (1,1 for monochrome; 2,2 for 4:2:0).
    pub fn sub_wh(&self) -> (u32, u32) {
        if self.has_chroma() {
            (2, 2)
        } else {
            (1, 1)
        }
    }

    /// Total tile count.
    pub fn tiles(&self) -> u32 {
        self.tile_grid.0 * self.tile_grid.1
    }

    /// Split the picture into `cols` x `rows` tiles.
    pub fn with_tiles(mut self, cols: u32, rows: u32) -> Self {
        self.tile_grid = (cols.max(1), rows.max(1));
        self
    }

    /// Tile the picture only if it is large enough for the tiles to be free.
    pub fn with_auto_tiles(self) -> Self {
        let (c, r) = auto_tile_grid(self.coded_width, self.coded_height);
        self.with_tiles(c, r)
    }

    /// Number of CTBs across / down the coded picture. The last row and column
    /// may hang off the edge; the quadtree splits them down to blocks that fit.
    pub fn ctbs(&self) -> (u32, u32) {
        let ctb = 1 << self.ctb_log2;
        (self.coded_width.div_ceil(ctb), self.coded_height.div_ceil(ctb))
    }

    /// Set the coding-tree sizes and re-pad the coded picture to whole minimum
    /// coding blocks — which is all HEVC requires, so a bigger CTB costs no
    /// extra padding.
    pub fn with_coding_tree(mut self, ctb_log2: u32, min_cb_log2: u32, max_tb_log2: u32) -> Self {
        assert!(min_cb_log2 >= 3 && min_cb_log2 <= ctb_log2, "MinCb in [8, CTB]");
        assert!((2..=5).contains(&max_tb_log2), "max TB in [4, 32]");
        self.ctb_log2 = ctb_log2;
        self.min_cb_log2 = min_cb_log2;
        self.max_tb_log2 = max_tb_log2;
        let mcb = 1 << min_cb_log2;
        self.coded_width = self.width.div_ceil(mcb) * mcb;
        self.coded_height = self.height.div_ceil(mcb) * mcb;
        self
    }

    /// Whether a conformance window is needed to crop padding back to display size.
    fn needs_conf_window(&self) -> bool {
        self.coded_width != self.width || self.coded_height != self.height
    }
}

/// `profile_tier_level(1, 0)` — one layer, level 6.0, no sub-layers (H.265 §7.3.3).
///
/// 96 bits total: 88 bits of profile info + an 8-bit level. `monochrome` selects
/// the Range-Extensions **Monochrome** profile (`profile_idc = 4` with the
/// 4:0:0 constraint flags) instead of **Main** (`profile_idc = 1`).
fn profile_tier_level(w: &mut BitWriter, monochrome: bool, level_idc: u8) {
    w.write_bits(0, 2); // general_profile_space
    w.flag(false); // general_tier_flag (Main tier)
    let profile_idc: u32 = if monochrome { 4 } else { 1 };
    w.write_bits(profile_idc, 5); // general_profile_idc

    // general_profile_compatibility_flag[32]: the bit for this profile_idc.
    for j in 0..32u32 {
        w.flag(j == profile_idc);
    }

    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(false); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag

    if monochrome {
        // Range-Extensions constraint flags for profile_idc 4 (§7.3.3).
        w.flag(true); // general_max_12bit_constraint_flag
        w.flag(true); // general_max_10bit_constraint_flag
        w.flag(true); // general_max_8bit_constraint_flag
        w.flag(true); // general_max_422chroma_constraint_flag
        w.flag(true); // general_max_420chroma_constraint_flag
        w.flag(true); // general_max_monochrome_constraint_flag
        w.flag(false); // general_intra_constraint_flag
        w.flag(false); // general_one_picture_only_constraint_flag
        w.flag(true); // general_lower_bit_rate_constraint_flag
        w.write_bits(0, 34 - 32); // general_reserved_zero_34bits (upper 2)
        w.write_bits(0, 32); //                                  (lower 32)
        w.flag(false); // general_inbld_flag
    } else {
        // Main profile: 43 reserved-zero bits + 1 reserved/inbld bit.
        w.write_bits(0, 43 - 32);
        w.write_bits(0, 32);
        w.flag(false); // general_inbld_flag
    }

    w.write_bits(level_idc as u32, 8); // general_level_idc
}

/// Sub-layer ordering info (single sub-layer): DPB sizing (H.265 §7.3.2.2/2.1).
fn sub_layer_ordering(w: &mut BitWriter) {
    w.flag(false); // *_sub_layer_ordering_info_present_flag = 0
                   // One iteration (i = max_sub_layers_minus1 = 0).
    w.write_ue(1); // max_dec_pic_buffering_minus1 = 1 (DPB holds 2)
    w.write_ue(0); // max_num_reorder_pics = 0
    w.write_ue(0); // max_latency_increase_plus1 = 0
}

/// Video Parameter Set RBSP (H.265 §7.3.2.1). Byte-aligned, includes trailing bits.
///
/// `monochrome` selects the profile advertised in the VPS's `profile_tier_level`
/// (kept consistent with the SPS).
pub fn write_vps(monochrome: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 4); // vps_video_parameter_set_id
    w.flag(true); // vps_base_layer_internal_flag
    w.flag(true); // vps_base_layer_available_flag
    w.write_bits(0, 6); // vps_max_layers_minus1
    w.write_bits(0, 3); // vps_max_sub_layers_minus1
    w.flag(true); // vps_temporal_id_nesting_flag
    w.write_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    profile_tier_level(&mut w, monochrome, 180);
    sub_layer_ordering(&mut w);
    w.write_bits(0, 6); // vps_max_layer_id
    w.write_ue(0); // vps_num_layer_sets_minus1
    w.flag(false); // vps_timing_info_present_flag
    w.flag(false); // vps_extension_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}

/// Sequence Parameter Set RBSP with `sps_seq_parameter_set_id = 0` (H.265 §7.3.2.2).
pub fn write_sps(cfg: &HevcConfig) -> Vec<u8> {
    write_sps_id(cfg, 0)
}

/// Sequence Parameter Set RBSP with an explicit `sps_seq_parameter_set_id` — the
/// alpha auxiliary layer uses id 1 alongside the base layer's id 0.
pub fn write_sps_id(cfg: &HevcConfig, sps_id: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 4); // sps_video_parameter_set_id
    w.write_bits(0, 3); // sps_max_sub_layers_minus1
    w.flag(true); // sps_temporal_id_nesting_flag
    profile_tier_level(&mut w, !cfg.has_chroma(), cfg.level_idc);
    w.write_ue(sps_id); // sps_seq_parameter_set_id
    w.write_ue(cfg.chroma_format_idc as u32); // chroma_format_idc (0 mono, 1 4:2:0)
    w.write_ue(cfg.coded_width); // pic_width_in_luma_samples
    w.write_ue(cfg.coded_height); // pic_height_in_luma_samples

    if cfg.needs_conf_window() {
        w.flag(true); // conformance_window_flag
                      // Offsets are in units of SubWidthC/SubHeightC (1 mono, 2 for 4:2:0).
        let (subw, subh) = cfg.sub_wh();
        w.write_ue(0); // conf_win_left_offset
        w.write_ue((cfg.coded_width - cfg.width) / subw); // conf_win_right_offset
        w.write_ue(0); // conf_win_top_offset
        w.write_ue((cfg.coded_height - cfg.height) / subh); // conf_win_bottom_offset
    } else {
        w.flag(false); // conformance_window_flag
    }

    w.write_ue(0); // bit_depth_luma_minus8
    w.write_ue(0); // bit_depth_chroma_minus8
    w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4 (POC lsb = 4 bits)
    sub_layer_ordering(&mut w);

    w.write_ue(cfg.min_cb_log2 - 3); // log2_min_luma_coding_block_size_minus3
    w.write_ue(cfg.ctb_log2 - cfg.min_cb_log2); // log2_diff_max_min_luma_coding_block_size
    w.write_ue(0); // log2_min_luma_transform_block_size_minus2 (minTU = 4)
    w.write_ue(cfg.max_tb_log2 - 2); // log2_diff_max_min_luma_transform_block_size
    w.write_ue(0); // max_transform_hierarchy_depth_inter
    w.write_ue(cfg.max_transform_hierarchy_depth_intra); // max_transform_hierarchy_depth_intra
    w.flag(false); // scaling_list_enabled_flag
    w.flag(false); // amp_enabled_flag
    w.flag(false); // sample_adaptive_offset_enabled_flag

    w.flag(cfg.pcm_enabled); // pcm_enabled_flag
    if cfg.pcm_enabled {
        w.write_bits(7, 4); // pcm_sample_bit_depth_luma_minus1 = 7 (8-bit)
        w.write_bits(7, 4); // pcm_sample_bit_depth_chroma_minus1 = 7
        w.write_ue(cfg.min_cb_log2 - 3); // log2_min_pcm_luma_coding_block_size_minus3
        w.write_ue(0); // log2_diff_max_min_pcm_luma_coding_block_size (IPCM max = min)
        w.flag(true); // pcm_loop_filter_disabled_flag
    }

    w.write_ue(0); // num_short_term_ref_pic_sets
    w.flag(false); // long_term_ref_pics_present_flag
    w.flag(false); // sps_temporal_mvp_enabled_flag
    w.flag(false); // strong_intra_smoothing_enabled_flag
    if cfg.full_range {
        w.flag(true); // vui_parameters_present_flag
                      // Minimal VUI: only signal the full-range flag (§E.2.1).
        w.flag(false); // aspect_ratio_info_present_flag
        w.flag(false); // overscan_info_present_flag
        w.flag(true); // video_signal_type_present_flag
        w.write_bits(5, 3); // video_format = 5 (unspecified)
        w.flag(true); // video_full_range_flag
        w.flag(false); // colour_description_present_flag
        w.flag(false); // chroma_loc_info_present_flag
        w.flag(false); // neutral_chroma_indication_flag
        w.flag(false); // field_seq_flag
        w.flag(false); // frame_field_info_present_flag
        w.flag(false); // default_display_window_flag
        w.flag(false); // vui_timing_info_present_flag
        w.flag(false); // bitstream_restriction_flag
    } else {
        w.flag(false); // vui_parameters_present_flag
    }
    w.flag(false); // sps_extension_present_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}

/// Picture Parameter Set RBSP (H.265 §7.3.2.3). Byte-aligned, includes trailing bits.
///
/// Deblocking is disabled here (via `deblocking_filter_control_present_flag = 1`,
/// `pps_deblocking_filter_disabled_flag = 1`) and SAO is off in the SPS, so PCM
/// samples reach the output untouched — losslessly.
pub fn write_pps() -> Vec<u8> {
    write_pps_ids(0, 0)
}

/// PPS 0 with `cu_qp_delta_enabled_flag` set and a quantization group the size
/// of a whole coding tree block (`diff_cu_qp_delta_depth = 0`).
///
/// At that granularity both quantizer-prediction neighbours always fall in the
/// *previous* CTB, so §8.6.1's `qPY_A`/`qPY_B` both collapse to `qPY_PREV` and
/// the predictor is simply the last coded QP. It is also the granularity a rate
/// controller wants, which is the other thing per-block QP is for.
pub fn write_pps_cu_qp_delta() -> Vec<u8> {
    write_pps_inner(0, 0, true, true)
}

/// As [`write_pps_cu_qp_delta`], with the deblocking filter switched off.
pub fn write_pps_cu_qp_delta_no_deblock() -> Vec<u8> {
    write_pps_inner(0, 0, true, false)
}

/// Picture Parameter Set RBSP with explicit ids (H.265 §7.3.2.3). The alpha
/// auxiliary layer uses `pps_id = 1`, `sps_id = 1`.
pub fn write_pps_ids(pps_id: u32, sps_id: u32) -> Vec<u8> {
    write_pps_inner(pps_id, sps_id, false, false)
}

fn write_pps_inner(pps_id: u32, sps_id: u32, cu_qp_delta: bool, deblock: bool) -> Vec<u8> {
    write_pps_full(pps_id, sps_id, cu_qp_delta, deblock, (1, 1))
}

/// PPS 0 with `cu_qp_delta`, deblocking, and a tile grid.
pub fn write_pps_tiled(tile_grid: (u32, u32)) -> Vec<u8> {
    write_pps_full(0, 0, true, true, tile_grid)
}

fn write_pps_full(
    pps_id: u32,
    sps_id: u32,
    cu_qp_delta: bool,
    deblock: bool,
    tile_grid: (u32, u32),
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(pps_id); // pps_pic_parameter_set_id
    w.write_ue(sps_id); // pps_seq_parameter_set_id
    w.flag(false); // dependent_slice_segments_enabled_flag
    w.flag(false); // output_flag_present_flag
    w.write_bits(0, 3); // num_extra_slice_header_bits
    w.flag(false); // sign_data_hiding_enabled_flag
    w.flag(false); // cabac_init_present_flag
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.write_se(0); // init_qp_minus26
    w.flag(false); // constrained_intra_pred_flag
    w.flag(false); // transform_skip_enabled_flag
    w.flag(cu_qp_delta); // cu_qp_delta_enabled_flag
    if cu_qp_delta {
        w.write_ue(0); // diff_cu_qp_delta_depth — quantization group = CTB
    }
    w.write_se(0); // pps_cb_qp_offset
    w.write_se(0); // pps_cr_qp_offset
    w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
    w.flag(false); // weighted_pred_flag
    w.flag(false); // weighted_bipred_flag
    w.flag(false); // transquant_bypass_enabled_flag
    let tiled = tile_grid.0 * tile_grid.1 > 1;
    w.flag(tiled); // tiles_enabled_flag
    w.flag(false); // entropy_coding_sync_enabled_flag
    if tiled {
        w.write_ue(tile_grid.0 - 1); // num_tile_columns_minus1
        w.write_ue(tile_grid.1 - 1); // num_tile_rows_minus1
        w.flag(true); // uniform_spacing_flag
        // Deblocking still runs across tile edges. The filter is a picture-level
        // pass here — intra prediction reads unfiltered samples — so letting it
        // cross costs nothing and removes the seams tiling would otherwise leave.
        w.flag(true); // loop_filter_across_tiles_enabled_flag
    }
    w.flag(false); // pps_loop_filter_across_slices_enabled_flag
    w.flag(true); // deblocking_filter_control_present_flag
    w.flag(false); // deblocking_filter_override_enabled_flag
    w.flag(!deblock); // pps_deblocking_filter_disabled_flag
    if deblock {
        w.write_se(0); // pps_beta_offset_div2
        w.write_se(0); // pps_tc_offset_div2
    }
    w.flag(false); // pps_scaling_list_data_present_flag
    w.flag(false); // lists_modification_present_flag
    w.write_ue(0); // log2_parallel_merge_level_minus2
    w.flag(false); // slice_segment_header_extension_present_flag
    w.flag(false); // pps_extension_present_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_pads_to_ctb() {
        let c = HevcConfig::new(1920, 1080);
        assert_eq!(c.coded_width, 1920); // already aligned
        assert_eq!(c.coded_height, 1088); // 1080 → 1088
        assert_eq!(c.ctbs(), (120, 68));
        assert!(c.needs_conf_window());

        let a = HevcConfig::new(64, 64);
        assert_eq!((a.coded_width, a.coded_height), (64, 64));
        assert!(!a.needs_conf_window());
    }

    #[test]
    fn param_sets_are_byte_aligned_and_nonempty() {
        assert!(!write_vps(false).is_empty());
        assert!(!write_sps(&HevcConfig::new(64, 64)).is_empty());
        assert!(!write_pps().is_empty());
    }

    #[test]
    fn monochrome_config_and_sps() {
        let c = HevcConfig::new_monochrome(64, 64);
        assert_eq!(c.chroma_format_idc, 0);
        assert!(!c.has_chroma());
        assert_eq!(c.sub_wh(), (1, 1));
        // SPS still 12-byte PTL + fields; just check it builds non-empty.
        assert!(!write_sps(&c).is_empty());
        assert!(!write_vps(true).is_empty());
    }
}
