//! H.264 high-level syntax: SPS and PPS for a minimal Baseline-profile I_PCM encoder.
//!
//! Baseline profile with CAVLC (`entropy_coding_mode_flag = 0`) codes every
//! macroblock as `I_PCM` via `ue(25)` — no CABAC engine required. Parameter sets
//! describe 8-bit 4:2:0 with optional frame cropping for non-MB-aligned sizes.

use crate::codec::bitstream::{rbsp_trailing_bits, BitWriter};

/// Luma macroblock edge in samples (16×16).
pub const MB_SIZE: u32 = 16;

/// Fixed configuration for the PCM encoder.
#[derive(Clone, Copy, Debug)]
pub struct H264Config {
    pub width: u32,
    pub height: u32,
    pub coded_width: u32,
    pub coded_height: u32,
    pub level_idc: u8,
    pub qp: i32,
}

impl H264Config {
    pub fn new(width: u32, height: u32) -> Self {
        let coded_width = width.div_ceil(MB_SIZE) * MB_SIZE;
        let coded_height = height.div_ceil(MB_SIZE) * MB_SIZE;
        Self {
            width,
            height,
            coded_width,
            coded_height,
            level_idc: 10, // 1.0 — matches common decoders / x264 defaults
            qp: 26,
        }
    }

    pub fn mbs(&self) -> (u32, u32) {
        (self.coded_width / MB_SIZE, self.coded_height / MB_SIZE)
    }

    fn needs_crop(&self) -> bool {
        self.coded_width != self.width || self.coded_height != self.height
    }

    /// Frame-crop offsets in crop units (CropUnitX = CropUnitY = 2 for 4:2:0).
    /// Padding is on the right/bottom only — same convention as the HEVC
    /// conformance window (`left`/`top` = 0).
    fn crop_offsets(&self) -> (u32, u32, u32, u32) {
        let dx = self.coded_width - self.width;
        let dy = self.coded_height - self.height;
        (0, dx / 2, 0, dy / 2)
    }
}

/// Write an SPS RBSP for Baseline profile 4:2:0 8-bit with I_PCM enabled.
pub fn write_sps(cfg: &H264Config) -> Vec<u8> {
    let (mbs_w, mbs_h) = cfg.mbs();
    let mut w = BitWriter::new();
    w.write_byte(66); // profile_idc = Baseline
    w.flag(false); // constraint_set0_flag
    w.flag(false); // constraint_set1_flag
    w.flag(false); // constraint_set2_flag
    w.flag(false); // constraint_set3_flag
    w.flag(false); // constraint_set4_flag
    w.flag(false); // constraint_set5_flag
    w.write_bits(0, 2); // reserved_zero_2bits
    w.write_byte(cfg.level_idc);
    w.write_ue(0); // seq_parameter_set_id
    w.write_ue(0); // log2_max_frame_num_minus4 → 4-bit frame_num
    w.write_ue(2); // pic_order_cnt_type (type 2 → no POC lsb in SPS)
    w.write_ue(0); // max_num_ref_frames
    w.flag(false); // gaps_in_frame_num_value_allowed_flag
    w.write_ue(mbs_w - 1); // pic_width_in_mbs_minus1
    w.write_ue(mbs_h - 1); // pic_height_in_map_units_minus1
    w.flag(true); // frame_mbs_only_flag
    w.flag(true); // direct_8x8_inference_flag
    w.flag(cfg.needs_crop()); // frame_cropping_flag
    if cfg.needs_crop() {
        let (l, r, t, b) = cfg.crop_offsets();
        w.write_ue(l);
        w.write_ue(r);
        w.write_ue(t);
        w.write_ue(b);
    }
    w.flag(false); // vui_parameters_present_flag
    rbsp_trailing_bits(&mut w);
    w.finish()
}
