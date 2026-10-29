//! AV1 OBU headers (dav1d `src/obu.c`, `src/getbits.c`): the sequence
//! header, the uncompressed frame header and tile group headers.

use std::sync::Arc;

use super::msac::{inv_recenter, ulog2};

pub type Res<T> = Result<T, String>;

fn err<T>(m: &str) -> Res<T> {
    Err(m.to_string())
}

// ---------------------------------------------------------------- bits

pub struct GetBits<'a> {
    data: &'a [u8],
    pub pos: usize, // bit position
    pub end: usize, // bit end
    pub error: bool,
}

impl<'a> GetBits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        GetBits { data, pos: 0, end: data.len() * 8, error: false }
    }

    pub fn bit(&mut self) -> u32 {
        if self.pos >= self.end {
            self.error = true;
            return 0;
        }
        let b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        b as u32
    }

    pub fn bits(&mut self, n: u32) -> u32 {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.bit() as u64;
        }
        v as u32
    }

    pub fn sbits(&mut self, n: u32) -> i32 {
        let v = self.bits(n) as i64;
        ((v << (64 - n)) >> (64 - n)) as i32
    }

    pub fn flag(&mut self) -> bool {
        self.bit() != 0
    }

    pub fn uleb128(&mut self) -> u32 {
        let mut val = 0u64;
        let mut i = 0;
        loop {
            let v = self.bits(8);
            val |= ((v & 0x7f) as u64) << i;
            i += 7;
            if v & 0x80 == 0 {
                break;
            }
            if i >= 56 {
                self.error = true;
                return 0;
            }
        }
        if val > u32::MAX as u64 {
            self.error = true;
            return 0;
        }
        val as u32
    }

    pub fn uniform(&mut self, max: u32) -> u32 {
        let l = ulog2(max) + 1;
        let m = (1u32 << l) - max;
        let v = self.bits(l as u32 - 1);
        if v < m {
            v
        } else {
            (v << 1) - m + self.bit()
        }
    }

    pub fn vlc(&mut self) -> u32 {
        if self.bit() != 0 {
            return 0;
        }
        let mut n_bits = 0;
        loop {
            n_bits += 1;
            if n_bits == 32 {
                return u32::MAX;
            }
            if self.bit() != 0 {
                break;
            }
        }
        ((1u32 << n_bits) - 1) + self.bits(n_bits)
    }

    fn subexp_u(&mut self, r: u32, n: u32) -> u32 {
        let mut v = 0u32;
        let mut i = 0;
        loop {
            let b = if i != 0 { 3 + i - 1 } else { 3 };
            if n < v + 3 * (1 << b) {
                v += self.uniform(n - v + 1);
                break;
            }
            if self.bit() == 0 {
                v += self.bits(b);
                break;
            }
            v += 1 << b;
            i += 1;
        }
        if r * 2 <= n {
            inv_recenter(r, v)
        } else {
            n - inv_recenter(n - r, v)
        }
    }

    pub fn bits_subexp(&mut self, r: i32, n: u32) -> i32 {
        self.subexp_u((r + (1 << n)) as u32, 2 << n) as i32 - (1 << n)
    }

    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    pub fn byte_pos(&self) -> usize {
        self.pos >> 3
    }
}

// ---------------------------------------------------------------- types

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Layout {
    I400,
    #[default]
    I420,
    I422,
    I444,
}

pub const FRAME_KEY: u8 = 0;
pub const FRAME_INTER: u8 = 1;
pub const FRAME_INTRA: u8 = 2;
pub const FRAME_SWITCH: u8 = 3;

pub const ADAPTIVE: u8 = 2;
pub const PRIMARY_REF_NONE: u8 = 7;

pub const TX_4X4_ONLY: u8 = 0;
pub const TX_LARGEST: u8 = 1;
pub const TX_SWITCHABLE: u8 = 2;

pub const WM_IDENTITY: u8 = 0;
pub const WM_TRANSLATION: u8 = 1;
pub const WM_ROT_ZOOM: u8 = 2;
pub const WM_AFFINE: u8 = 3;

pub const RESTORATION_NONE: u8 = 0;
pub const RESTORATION_SWITCHABLE: u8 = 1;
pub const RESTORATION_WIENER: u8 = 2;
pub const RESTORATION_SGRPROJ: u8 = 3;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct OperatingPoint {
    pub major_level: u8,
    pub minor_level: u8,
    pub initial_display_delay: u8,
    pub idc: u16,
    pub tier: u8,
    pub decoder_model_param_present: bool,
    pub display_model_param_present: bool,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct OperatingParameterInfo {
    pub decoder_buffer_delay: u32,
    pub encoder_buffer_delay: u32,
    pub low_delay_mode: bool,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct SeqHeader {
    pub profile: u8,
    pub max_width: i32,
    pub max_height: i32,
    pub layout: Layout,
    pub pri: u8,
    pub trc: u8,
    pub mtrx: u8,
    pub chr: u8,
    pub hbd: u8,
    pub color_range: bool,
    pub num_operating_points: u8,
    pub operating_points: Vec<OperatingPoint>,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present: bool,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture: u32,
    pub decoder_model_info_present: bool,
    pub encoder_decoder_buffer_delay_length: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_delay_length: u8,
    pub frame_presentation_delay_length: u8,
    pub display_model_info_present: bool,
    pub width_n_bits: u8,
    pub height_n_bits: u8,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_n_bits: u8,
    pub frame_id_n_bits: u8,
    pub sb128: bool,
    pub filter_intra: bool,
    pub intra_edge_filter: bool,
    pub inter_intra: bool,
    pub masked_compound: bool,
    pub warped_motion: bool,
    pub dual_filter: bool,
    pub order_hint: bool,
    pub jnt_comp: bool,
    pub ref_frame_mvs: bool,
    pub screen_content_tools: u8,
    pub force_integer_mv: u8,
    pub order_hint_n_bits: u8,
    pub super_res: bool,
    pub cdef: bool,
    pub restoration: bool,
    pub ss_hor: u8,
    pub ss_ver: u8,
    pub monochrome: bool,
    pub color_description_present: bool,
    pub separate_uv_delta_q: bool,
    pub film_grain_present: bool,
    pub operating_parameter_info: Vec<OperatingParameterInfo>,
}

impl SeqHeader {
    /// Everything but `operating_parameter_info` (dav1d's memcmp).
    pub fn same_sequence(&self, o: &SeqHeader) -> bool {
        let mut a = self.clone();
        a.operating_parameter_info = o.operating_parameter_info.clone();
        a == *o
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpedMotionParams {
    pub typ: u8,
    pub matrix: [i32; 6],
    /// alpha, beta, gamma, delta.
    pub abcd: [i16; 4],
}

pub const DEFAULT_WM_PARAMS: WarpedMotionParams =
    WarpedMotionParams { typ: WM_IDENTITY, matrix: [0, 0, 1 << 16, 0, 0, 1 << 16], abcd: [0; 4] };

impl Default for WarpedMotionParams {
    fn default() -> Self {
        DEFAULT_WM_PARAMS
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SegmentationData {
    pub delta_q: i16,
    pub delta_lf_y_v: i8,
    pub delta_lf_y_h: i8,
    pub delta_lf_u: i8,
    pub delta_lf_v: i8,
    pub r#ref: i8,
    pub skip: bool,
    pub globalmv: bool,
    /// Which features the segment enables, `FeatureEnabled[i][j]` as bit `j`:
    /// 0 alt Q, 1–4 the loop filter deltas, 5 reference, 6 skip, 7 global MV.
    /// Values alone cannot say: a feature can be enabled with value 0.
    pub enabled: u8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SegmentationDataSet {
    pub d: [SegmentationData; 8],
    pub preskip: bool,
    pub last_active_segid: i8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct LoopfilterModeRefDeltas {
    pub mode_delta: [i8; 2],
    pub ref_delta: [i8; 8],
}

pub const DEFAULT_MODE_REF_DELTAS: LoopfilterModeRefDeltas =
    LoopfilterModeRefDeltas { mode_delta: [0, 0], ref_delta: [1, 0, 0, 0, -1, 0, -1, -1] };

#[derive(Clone, Copy, Default, Debug)]
pub struct FilmGrainData {
    pub seed: u32,
    pub num_y_points: i32,
    pub y_points: [[u8; 2]; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_uv_points: [i32; 2],
    pub uv_points: [[[u8; 2]; 10]; 2],
    pub scaling_shift: i32,
    pub ar_coeff_lag: i32,
    pub ar_coeffs_y: [i8; 24],
    pub ar_coeffs_uv: [[i8; 28]; 2],
    pub ar_coeff_shift: u64,
    pub grain_scale_shift: i32,
    pub uv_mult: [i32; 2],
    pub uv_luma_mult: [i32; 2],
    pub uv_offset: [i32; 2],
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

#[derive(Clone, Debug)]
pub struct Tiling {
    pub uniform: bool,
    pub n_bytes: u8,
    pub min_log2_cols: u8,
    pub max_log2_cols: u8,
    pub log2_cols: u8,
    pub cols: u8,
    pub min_log2_rows: u8,
    pub max_log2_rows: u8,
    pub log2_rows: u8,
    pub rows: u8,
    pub col_start_sb: [u16; 65],
    pub row_start_sb: [u16; 65],
    pub update: u16,
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling {
            uniform: false,
            n_bytes: 0,
            min_log2_cols: 0,
            max_log2_cols: 0,
            log2_cols: 0,
            cols: 0,
            min_log2_rows: 0,
            max_log2_rows: 0,
            log2_rows: 0,
            rows: 0,
            col_start_sb: [0; 65],
            row_start_sb: [0; 65],
            update: 0,
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Quant {
    pub yac: u8,
    pub ydc_delta: i8,
    pub udc_delta: i8,
    pub uac_delta: i8,
    pub vdc_delta: i8,
    pub vac_delta: i8,
    pub qm: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal: bool,
    pub update_data: bool,
    pub seg_data: SegmentationDataSet,
    pub lossless: [bool; 8],
    pub qidx: [u8; 8],
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Delta {
    pub q_present: bool,
    pub q_res_log2: u8,
    pub lf_present: bool,
    pub lf_res_log2: u8,
    pub lf_multi: bool,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Loopfilter {
    pub level_y: [u8; 2],
    pub level_u: u8,
    pub level_v: u8,
    pub mode_ref_delta_enabled: bool,
    pub mode_ref_delta_update: bool,
    pub mode_ref_deltas: LoopfilterModeRefDeltas,
    pub sharpness: u8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Cdef {
    pub damping: u8,
    pub n_bits: u8,
    pub y_strength: [u8; 8],
    pub uv_strength: [u8; 8],
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Restoration {
    pub typ: [u8; 3],
    pub unit_size: [u8; 2],
}

#[derive(Clone, Default, Debug)]
pub struct FrameHeader {
    pub film_grain: FilmGrainData,
    pub film_grain_present: bool,
    pub film_grain_update: bool,
    pub frame_type: u8,
    /// coded width, superres-upscaled width.
    pub width: [i32; 2],
    pub height: i32,
    pub frame_offset: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub show_existing_frame: bool,
    pub existing_frame_idx: u8,
    pub frame_id: u32,
    pub frame_presentation_delay: u32,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub frame_size_override: bool,
    pub primary_ref_frame: u8,
    pub buffer_removal_time_present: bool,
    pub refresh_frame_flags: u8,
    pub render_width: i32,
    pub render_height: i32,
    pub super_res_width_scale_denominator: u8,
    pub super_res_enabled: bool,
    pub have_render_size: bool,
    pub allow_intrabc: bool,
    pub frame_ref_short_signaling: bool,
    pub refidx: [i8; 7],
    pub hp: bool,
    pub subpel_filter_mode: u8,
    pub switchable_motion_mode: bool,
    pub use_ref_frame_mvs: bool,
    pub refresh_context: bool,
    pub tiling: Tiling,
    pub quant: Quant,
    pub segmentation: Segmentation,
    pub delta: Delta,
    pub all_lossless: bool,
    pub loopfilter: Loopfilter,
    pub cdef: Cdef,
    pub restoration: Restoration,
    pub txfm_mode: u8,
    pub switchable_comp_refs: bool,
    pub skip_mode_allowed: bool,
    pub skip_mode_enabled: bool,
    pub skip_mode_refs: [i8; 2],
    pub warp_motion: bool,
    pub reduced_txtp_set: bool,
    pub gmv: [WarpedMotionParams; 7],
}

impl FrameHeader {
    pub fn is_key_or_intra(&self) -> bool {
        self.frame_type & 1 == 0
    }
    pub fn is_inter_or_switch(&self) -> bool {
        self.frame_type & 1 != 0
    }
}

pub fn get_poc_diff(order_hint_n_bits: u8, poc0: i32, poc1: i32) -> i32 {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1 << (order_hint_n_bits - 1);
    let diff = poc0 - poc1;
    (diff & (mask - 1)) - (diff & mask)
}

fn check_trailing_bits(gb: &mut GetBits) -> Res<()> {
    gb.bit();
    if gb.error {
        return err("trailing bits overrun");
    }
    Ok(())
}

// ---------------------------------------------------------------- sequence header

pub fn parse_seq_hdr(gb: &mut GetBits) -> Res<SeqHeader> {
    let mut h = SeqHeader {
        operating_points: vec![OperatingPoint::default(); 32],
        operating_parameter_info: vec![OperatingParameterInfo::default(); 32],
        ..Default::default()
    };
    h.profile = gb.bits(3) as u8;
    if h.profile > 2 {
        return err("bad profile");
    }
    h.still_picture = gb.flag();
    h.reduced_still_picture_header = gb.flag();
    if h.reduced_still_picture_header && !h.still_picture {
        return err("bad still picture flags");
    }
    if h.reduced_still_picture_header {
        h.num_operating_points = 1;
        h.operating_points[0].major_level = gb.bits(3) as u8;
        h.operating_points[0].minor_level = gb.bits(2) as u8;
        h.operating_points[0].initial_display_delay = 10;
    } else {
        h.timing_info_present = gb.flag();
        if h.timing_info_present {
            h.num_units_in_tick = gb.bits(32);
            h.time_scale = gb.bits(32);
            h.equal_picture_interval = gb.flag();
            if h.equal_picture_interval {
                let n = gb.vlc();
                if n == u32::MAX {
                    return err("bad num_ticks_per_picture");
                }
                h.num_ticks_per_picture = n + 1;
            }
            h.decoder_model_info_present = gb.flag();
            if h.decoder_model_info_present {
                h.encoder_decoder_buffer_delay_length = gb.bits(5) as u8 + 1;
                h.num_units_in_decoding_tick = gb.bits(32);
                h.buffer_removal_delay_length = gb.bits(5) as u8 + 1;
                h.frame_presentation_delay_length = gb.bits(5) as u8 + 1;
            }
        }
        h.display_model_info_present = gb.flag();
        h.num_operating_points = gb.bits(5) as u8 + 1;
        for i in 0..h.num_operating_points as usize {
            let mut op = OperatingPoint { idc: gb.bits(12) as u16, ..Default::default() };
            if op.idc != 0 && (op.idc & 0xff == 0 || op.idc & 0xf00 == 0) {
                return err("bad operating point idc");
            }
            op.major_level = 2 + gb.bits(3) as u8;
            op.minor_level = gb.bits(2) as u8;
            if op.major_level > 3 {
                op.tier = gb.bit() as u8;
            }
            if h.decoder_model_info_present {
                op.decoder_model_param_present = gb.flag();
                if op.decoder_model_param_present {
                    let n = h.encoder_decoder_buffer_delay_length as u32;
                    let opi = &mut h.operating_parameter_info[i];
                    opi.decoder_buffer_delay = gb.bits(n);
                    opi.encoder_buffer_delay = gb.bits(n);
                    opi.low_delay_mode = gb.flag();
                }
            }
            if h.display_model_info_present {
                op.display_model_param_present = gb.flag();
            }
            op.initial_display_delay = if op.display_model_param_present { gb.bits(4) as u8 + 1 } else { 10 };
            h.operating_points[i] = op;
        }
    }
    h.width_n_bits = gb.bits(4) as u8 + 1;
    h.height_n_bits = gb.bits(4) as u8 + 1;
    h.max_width = gb.bits(h.width_n_bits as u32) as i32 + 1;
    h.max_height = gb.bits(h.height_n_bits as u32) as i32 + 1;
    if !h.reduced_still_picture_header {
        h.frame_id_numbers_present = gb.flag();
        if h.frame_id_numbers_present {
            h.delta_frame_id_n_bits = gb.bits(4) as u8 + 2;
            h.frame_id_n_bits = gb.bits(3) as u8 + h.delta_frame_id_n_bits + 1;
        }
    }
    h.sb128 = gb.flag();
    h.filter_intra = gb.flag();
    h.intra_edge_filter = gb.flag();
    if h.reduced_still_picture_header {
        h.screen_content_tools = ADAPTIVE;
        h.force_integer_mv = ADAPTIVE;
    } else {
        h.inter_intra = gb.flag();
        h.masked_compound = gb.flag();
        h.warped_motion = gb.flag();
        h.dual_filter = gb.flag();
        h.order_hint = gb.flag();
        if h.order_hint {
            h.jnt_comp = gb.flag();
            h.ref_frame_mvs = gb.flag();
        }
        h.screen_content_tools = if gb.flag() { ADAPTIVE } else { gb.bit() as u8 };
        h.force_integer_mv =
            if h.screen_content_tools != 0 { if gb.flag() { ADAPTIVE } else { gb.bit() as u8 } } else { 2 };
        if h.order_hint {
            h.order_hint_n_bits = gb.bits(3) as u8 + 1;
        }
    }
    h.super_res = gb.flag();
    h.cdef = gb.flag();
    h.restoration = gb.flag();

    h.hbd = gb.bit() as u8;
    if h.profile == 2 && h.hbd != 0 {
        h.hbd += gb.bit() as u8;
    }
    if h.profile != 1 {
        h.monochrome = gb.flag();
    }
    h.color_description_present = gb.flag();
    if h.color_description_present {
        h.pri = gb.bits(8) as u8;
        h.trc = gb.bits(8) as u8;
        h.mtrx = gb.bits(8) as u8;
    } else {
        h.pri = 2;
        h.trc = 2;
        h.mtrx = 2;
    }
    if h.monochrome {
        h.color_range = gb.flag();
        h.layout = Layout::I400;
        h.ss_hor = 1;
        h.ss_ver = 1;
        h.chr = 0;
    } else if h.pri == 1 && h.trc == 13 && h.mtrx == 0 {
        h.layout = Layout::I444;
        h.color_range = true;
        if h.profile != 1 && !(h.profile == 2 && h.hbd == 2) {
            return err("sRGB needs 4:4:4 profile");
        }
    } else {
        h.color_range = gb.flag();
        match h.profile {
            0 => {
                h.layout = Layout::I420;
                h.ss_hor = 1;
                h.ss_ver = 1;
            }
            1 => h.layout = Layout::I444,
            _ => {
                if h.hbd == 2 {
                    h.ss_hor = gb.bit() as u8;
                    if h.ss_hor != 0 {
                        h.ss_ver = gb.bit() as u8;
                    }
                } else {
                    h.ss_hor = 1;
                }
                h.layout = if h.ss_hor != 0 {
                    if h.ss_ver != 0 {
                        Layout::I420
                    } else {
                        Layout::I422
                    }
                } else {
                    Layout::I444
                };
            }
        }
        h.chr = if h.ss_hor & h.ss_ver != 0 { gb.bits(2) as u8 } else { 0 };
    }
    if !h.monochrome {
        h.separate_uv_delta_q = gb.flag();
    }
    h.film_grain_present = gb.flag();
    check_trailing_bits(gb)?;
    Ok(h)
}

// ---------------------------------------------------------------- frame header

/// The reference slots' headers, as the parser needs them.
pub type RefHeaders = [Option<Arc<FrameHeader>>; 8];

fn tile_log2(sz: i32, tgt: i32) -> u8 {
    let mut k = 0;
    while (sz << k) < tgt {
        k += 1;
    }
    k
}

fn read_frame_size(seq: &SeqHeader, refs: &RefHeaders, h: &mut FrameHeader, gb: &mut GetBits, use_ref: bool) -> Res<()> {
    if use_ref {
        for i in 0..7 {
            if gb.flag() {
                let r = refs[h.refidx[i] as usize].as_ref().ok_or("missing reference")?;
                h.width[1] = r.width[1];
                h.height = r.height;
                h.render_width = r.render_width;
                h.render_height = r.render_height;
                h.super_res_enabled = seq.super_res && gb.flag();
                if h.super_res_enabled {
                    let d = 9 + gb.bits(3) as i32;
                    h.super_res_width_scale_denominator = d as u8;
                    h.width[0] = ((h.width[1] * 8 + (d >> 1)) / d).max(16.min(h.width[1]));
                } else {
                    h.super_res_width_scale_denominator = 8;
                    h.width[0] = h.width[1];
                }
                return Ok(());
            }
        }
    }
    if h.frame_size_override {
        h.width[1] = gb.bits(seq.width_n_bits as u32) as i32 + 1;
        h.height = gb.bits(seq.height_n_bits as u32) as i32 + 1;
    } else {
        h.width[1] = seq.max_width;
        h.height = seq.max_height;
    }
    h.super_res_enabled = seq.super_res && gb.flag();
    if h.super_res_enabled {
        let d = 9 + gb.bits(3) as i32;
        h.super_res_width_scale_denominator = d as u8;
        h.width[0] = ((h.width[1] * 8 + (d >> 1)) / d).max(16.min(h.width[1]));
    } else {
        h.super_res_width_scale_denominator = 8;
        h.width[0] = h.width[1];
    }
    h.have_render_size = gb.flag();
    if h.have_render_size {
        h.render_width = gb.bits(16) as i32 + 1;
        h.render_height = gb.bits(16) as i32 + 1;
    } else {
        h.render_width = h.width[1];
        h.render_height = h.height;
    }
    Ok(())
}

pub fn parse_frame_hdr(seq: &SeqHeader, refs: &RefHeaders, gb: &mut GetBits, temporal_id: u8, spatial_id: u8) -> Res<FrameHeader> {
    let mut h = FrameHeader { temporal_id, spatial_id, ..Default::default() };
    let rf = |i: usize| -> Res<&Arc<FrameHeader>> { refs[i].as_ref().ok_or_else(|| "missing reference".to_string()) };

    h.show_existing_frame = !seq.reduced_still_picture_header && gb.flag();
    if h.show_existing_frame {
        h.existing_frame_idx = gb.bits(3) as u8;
        if seq.decoder_model_info_present && !seq.equal_picture_interval {
            h.frame_presentation_delay = gb.bits(seq.frame_presentation_delay_length as u32);
        }
        if seq.frame_id_numbers_present {
            h.frame_id = gb.bits(seq.frame_id_n_bits as u32);
            let r = rf(h.existing_frame_idx as usize)?;
            if r.frame_id != h.frame_id {
                return err("frame id mismatch");
            }
        }
        return Ok(h);
    }

    h.frame_type = if seq.reduced_still_picture_header { FRAME_KEY } else { gb.bits(2) as u8 };
    h.show_frame = seq.reduced_still_picture_header || gb.flag();
    if h.show_frame {
        if seq.decoder_model_info_present && !seq.equal_picture_interval {
            h.frame_presentation_delay = gb.bits(seq.frame_presentation_delay_length as u32);
        }
        h.showable_frame = h.frame_type != FRAME_KEY;
    } else {
        h.showable_frame = gb.flag();
    }
    h.error_resilient_mode = (h.frame_type == FRAME_KEY && h.show_frame)
        || h.frame_type == FRAME_SWITCH
        || seq.reduced_still_picture_header
        || gb.flag();
    h.disable_cdf_update = gb.flag();
    h.allow_screen_content_tools =
        if seq.screen_content_tools == ADAPTIVE { gb.flag() } else { seq.screen_content_tools != 0 };
    h.force_integer_mv = if h.allow_screen_content_tools {
        if seq.force_integer_mv == ADAPTIVE {
            gb.flag()
        } else {
            seq.force_integer_mv != 0
        }
    } else {
        false
    };
    if h.is_key_or_intra() {
        h.force_integer_mv = true;
    }
    if seq.frame_id_numbers_present {
        h.frame_id = gb.bits(seq.frame_id_n_bits as u32);
    }
    h.frame_size_override = if seq.reduced_still_picture_header {
        false
    } else if h.frame_type == FRAME_SWITCH {
        true
    } else {
        gb.flag()
    };
    h.frame_offset = if seq.order_hint { gb.bits(seq.order_hint_n_bits as u32) as u8 } else { 0 };
    h.primary_ref_frame =
        if !h.error_resilient_mode && h.is_inter_or_switch() { gb.bits(3) as u8 } else { PRIMARY_REF_NONE };

    if seq.decoder_model_info_present {
        h.buffer_removal_time_present = gb.flag();
        if h.buffer_removal_time_present {
            for i in 0..seq.num_operating_points as usize {
                let sop = &seq.operating_points[i];
                if sop.decoder_model_param_present {
                    let in_t = (sop.idc >> h.temporal_id) & 1;
                    let in_s = (sop.idc >> (h.spatial_id + 8)) & 1;
                    if sop.idc == 0 || (in_t != 0 && in_s != 0) {
                        gb.bits(seq.buffer_removal_delay_length as u32);
                    }
                }
            }
        }
    }

    if h.is_key_or_intra() {
        h.refresh_frame_flags = if h.frame_type == FRAME_KEY && h.show_frame { 0xff } else { gb.bits(8) as u8 };
        if h.refresh_frame_flags != 0xff && h.error_resilient_mode && seq.order_hint {
            for _ in 0..8 {
                gb.bits(seq.order_hint_n_bits as u32);
            }
        }
        read_frame_size(seq, refs, &mut h, gb, false)?;
        h.allow_intrabc = h.allow_screen_content_tools && !h.super_res_enabled && gb.flag();
        h.use_ref_frame_mvs = false;
    } else {
        h.allow_intrabc = false;
        h.refresh_frame_flags = if h.frame_type == FRAME_SWITCH { 0xff } else { gb.bits(8) as u8 };
        if h.error_resilient_mode && seq.order_hint {
            for _ in 0..8 {
                gb.bits(seq.order_hint_n_bits as u32);
            }
        }
        h.frame_ref_short_signaling = seq.order_hint && gb.flag();
        if h.frame_ref_short_signaling {
            h.refidx[0] = gb.bits(3) as i8;
            h.refidx[1] = -1;
            h.refidx[2] = -1;
            h.refidx[3] = gb.bits(3) as i8;
            h.refidx[4] = -1;
            h.refidx[5] = -1;
            h.refidx[6] = -1;
            let mut shifted = [0i32; 8];
            let cur = 1 << (seq.order_hint_n_bits - 1);
            for (i, s) in shifted.iter_mut().enumerate() {
                let r = rf(i)?;
                *s = cur + get_poc_diff(seq.order_hint_n_bits, r.frame_offset as i32, h.frame_offset as i32);
            }
            let mut used = [false; 8];
            used[h.refidx[0] as usize] = true;
            used[h.refidx[3] as usize] = true;
            let mut latest = -1;
            for (i, &hint) in shifted.iter().enumerate() {
                if !used[i] && hint >= cur && hint >= latest {
                    h.refidx[6] = i as i8;
                    latest = hint;
                }
            }
            if latest != -1 {
                used[h.refidx[6] as usize] = true;
            }
            let mut earliest = i32::MAX;
            for (i, &hint) in shifted.iter().enumerate() {
                if !used[i] && hint >= cur && hint < earliest {
                    h.refidx[4] = i as i8;
                    earliest = hint;
                }
            }
            if earliest != i32::MAX {
                used[h.refidx[4] as usize] = true;
            }
            earliest = i32::MAX;
            for (i, &hint) in shifted.iter().enumerate() {
                if !used[i] && hint >= cur && hint < earliest {
                    h.refidx[5] = i as i8;
                    earliest = hint;
                }
            }
            if earliest != i32::MAX {
                used[h.refidx[5] as usize] = true;
            }
            for i in 1..7 {
                if h.refidx[i] < 0 {
                    latest = -1;
                    for (j, &hint) in shifted.iter().enumerate() {
                        if !used[j] && hint < cur && hint >= latest {
                            h.refidx[i] = j as i8;
                            latest = hint;
                        }
                    }
                    if latest != -1 {
                        used[h.refidx[i] as usize] = true;
                    }
                }
            }
            earliest = i32::MAX;
            let mut r = -1i8;
            for (i, &hint) in shifted.iter().enumerate() {
                if hint < earliest {
                    r = i as i8;
                    earliest = hint;
                }
            }
            for x in h.refidx.iter_mut() {
                if *x < 0 {
                    *x = r;
                }
            }
        }
        for i in 0..7 {
            if !h.frame_ref_short_signaling {
                h.refidx[i] = gb.bits(3) as i8;
            }
            if seq.frame_id_numbers_present {
                let delta = gb.bits(seq.delta_frame_id_n_bits as u32) + 1;
                let n = seq.frame_id_n_bits as u32;
                let ref_id = (h.frame_id + (1 << n) - delta) & ((1 << n) - 1);
                if rf(h.refidx[i] as usize)?.frame_id != ref_id {
                    return err("reference frame id mismatch");
                }
            }
        }
        let use_ref = !h.error_resilient_mode && h.frame_size_override;
        read_frame_size(seq, refs, &mut h, gb, use_ref)?;
        h.hp = !h.force_integer_mv && gb.flag();
        h.subpel_filter_mode = if gb.flag() { super::levels::FILTER_SWITCHABLE } else { gb.bits(2) as u8 };
        h.switchable_motion_mode = gb.flag();
        h.use_ref_frame_mvs =
            !h.error_resilient_mode && seq.ref_frame_mvs && seq.order_hint && h.is_inter_or_switch() && gb.flag();
    }

    h.refresh_context = !seq.reduced_still_picture_header && !h.disable_cdf_update && !gb.flag();

    // Tiling.
    let t = &mut h.tiling;
    t.uniform = gb.flag();
    let sbsz_min1 = (64 << seq.sb128 as i32) - 1;
    let sbsz_log2 = 6 + seq.sb128 as i32;
    let sbw = (h.width[0] + sbsz_min1) >> sbsz_log2;
    let sbh = (h.height + sbsz_min1) >> sbsz_log2;
    let max_tile_width_sb = 4096 >> sbsz_log2;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2);
    t.min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    t.max_log2_cols = tile_log2(1, sbw.min(64));
    t.max_log2_rows = tile_log2(1, sbh.min(64));
    let min_log2_tiles = (tile_log2(max_tile_area_sb, sbw * sbh) as i32).max(t.min_log2_cols as i32);
    if t.uniform {
        t.log2_cols = t.min_log2_cols;
        while t.log2_cols < t.max_log2_cols && gb.flag() {
            t.log2_cols += 1;
        }
        let tile_w = 1 + ((sbw - 1) >> t.log2_cols);
        t.cols = 0;
        let mut sbx = 0;
        while sbx < sbw {
            t.col_start_sb[t.cols as usize] = sbx as u16;
            sbx += tile_w;
            t.cols += 1;
        }
        t.min_log2_rows = (min_log2_tiles - t.log2_cols as i32).max(0) as u8;
        t.log2_rows = t.min_log2_rows;
        while t.log2_rows < t.max_log2_rows && gb.flag() {
            t.log2_rows += 1;
        }
        let tile_h = 1 + ((sbh - 1) >> t.log2_rows);
        t.rows = 0;
        let mut sby = 0;
        while sby < sbh {
            t.row_start_sb[t.rows as usize] = sby as u16;
            sby += tile_h;
            t.rows += 1;
        }
    } else {
        t.cols = 0;
        let mut widest = 0;
        let mut max_area = sbw * sbh;
        let mut sbx = 0;
        while sbx < sbw && t.cols < 64 {
            let tw_sb = (sbw - sbx).min(max_tile_width_sb);
            let tile_w = if tw_sb > 1 { 1 + gb.uniform(tw_sb as u32) as i32 } else { 1 };
            t.col_start_sb[t.cols as usize] = sbx as u16;
            sbx += tile_w;
            widest = widest.max(tile_w);
            t.cols += 1;
        }
        t.log2_cols = tile_log2(1, t.cols as i32);
        if min_log2_tiles != 0 {
            max_area >>= min_log2_tiles + 1;
        }
        let max_tile_height_sb = (max_area / widest).max(1);
        t.rows = 0;
        let mut sby = 0;
        while sby < sbh && t.rows < 64 {
            let th_sb = (sbh - sby).min(max_tile_height_sb);
            let tile_h = if th_sb > 1 { 1 + gb.uniform(th_sb as u32) as i32 } else { 1 };
            t.row_start_sb[t.rows as usize] = sby as u16;
            sby += tile_h;
            t.rows += 1;
        }
        t.log2_rows = tile_log2(1, t.rows as i32);
    }
    t.col_start_sb[t.cols as usize] = sbw as u16;
    t.row_start_sb[t.rows as usize] = sbh as u16;
    if t.log2_cols != 0 || t.log2_rows != 0 {
        t.update = gb.bits((t.log2_cols + t.log2_rows) as u32) as u16;
        if t.update as i32 >= t.cols as i32 * t.rows as i32 {
            return err("bad context_update_tile_id");
        }
        t.n_bytes = gb.bits(2) as u8 + 1;
    } else {
        t.n_bytes = 0;
        t.update = 0;
    }

    // Quantisation.
    let q = &mut h.quant;
    q.yac = gb.bits(8) as u8;
    q.ydc_delta = if gb.flag() { gb.sbits(7) as i8 } else { 0 };
    if !seq.monochrome {
        let diff_uv = seq.separate_uv_delta_q && gb.flag();
        q.udc_delta = if gb.flag() { gb.sbits(7) as i8 } else { 0 };
        q.uac_delta = if gb.flag() { gb.sbits(7) as i8 } else { 0 };
        if diff_uv {
            q.vdc_delta = if gb.flag() { gb.sbits(7) as i8 } else { 0 };
            q.vac_delta = if gb.flag() { gb.sbits(7) as i8 } else { 0 };
        } else {
            q.vdc_delta = q.udc_delta;
            q.vac_delta = q.uac_delta;
        }
    }
    q.qm = gb.flag();
    if q.qm {
        q.qm_y = gb.bits(4) as u8;
        q.qm_u = gb.bits(4) as u8;
        q.qm_v = if seq.separate_uv_delta_q { gb.bits(4) as u8 } else { q.qm_u };
    }

    // Segmentation.
    let s = &mut h.segmentation;
    s.enabled = gb.flag();
    if s.enabled {
        if h.primary_ref_frame == PRIMARY_REF_NONE {
            s.update_map = true;
            s.temporal = false;
            s.update_data = true;
        } else {
            s.update_map = gb.flag();
            s.temporal = s.update_map && gb.flag();
            s.update_data = gb.flag();
        }
        if s.update_data {
            s.seg_data.preskip = false;
            s.seg_data.last_active_segid = -1;
            for i in 0..8 {
                let seg = &mut s.seg_data.d[i];
                seg.enabled = 0;
                if gb.flag() {
                    seg.enabled |= 1;
                    seg.delta_q = gb.sbits(9) as i16;
                    s.seg_data.last_active_segid = i as i8;
                } else {
                    seg.delta_q = 0;
                }
                for k in 0..4 {
                    let v = if gb.flag() {
                        seg.enabled |= 2 << k;
                        s.seg_data.last_active_segid = i as i8;
                        gb.sbits(7) as i8
                    } else {
                        0
                    };
                    match k {
                        0 => seg.delta_lf_y_v = v,
                        1 => seg.delta_lf_y_h = v,
                        2 => seg.delta_lf_u = v,
                        _ => seg.delta_lf_v = v,
                    }
                }
                if gb.flag() {
                    seg.enabled |= 1 << 5;
                    seg.r#ref = gb.bits(3) as i8;
                    s.seg_data.last_active_segid = i as i8;
                    s.seg_data.preskip = true;
                } else {
                    seg.r#ref = -1;
                }
                seg.skip = gb.flag();
                seg.enabled |= (seg.skip as u8) << 6;
                if seg.skip {
                    s.seg_data.last_active_segid = i as i8;
                    s.seg_data.preskip = true;
                }
                seg.globalmv = gb.flag();
                seg.enabled |= (seg.globalmv as u8) << 7;
                if seg.globalmv {
                    s.seg_data.last_active_segid = i as i8;
                    s.seg_data.preskip = true;
                }
            }
        } else {
            let pri = h.refidx[h.primary_ref_frame as usize] as usize;
            s.seg_data = rf(pri)?.segmentation.seg_data;
        }
    } else {
        s.seg_data = SegmentationDataSet::default();
        for d in s.seg_data.d.iter_mut() {
            d.r#ref = -1;
        }
    }

    // Delta q / lf.
    h.delta.q_present = h.quant.yac != 0 && gb.flag();
    h.delta.q_res_log2 = if h.delta.q_present { gb.bits(2) as u8 } else { 0 };
    h.delta.lf_present = h.delta.q_present && !h.allow_intrabc && gb.flag();
    h.delta.lf_res_log2 = if h.delta.lf_present { gb.bits(2) as u8 } else { 0 };
    h.delta.lf_multi = h.delta.lf_present && gb.flag();

    let q = &h.quant;
    let delta_lossless = q.ydc_delta == 0 && q.udc_delta == 0 && q.uac_delta == 0 && q.vdc_delta == 0 && q.vac_delta == 0;
    h.all_lossless = true;
    for i in 0..8 {
        let s = &mut h.segmentation;
        s.qidx[i] = if s.enabled {
            (q.yac as i32 + s.seg_data.d[i].delta_q as i32).clamp(0, 255) as u8
        } else {
            q.yac
        };
        s.lossless[i] = s.qidx[i] == 0 && delta_lossless;
        h.all_lossless &= s.lossless[i];
    }

    // Loop filter.
    let lf = &mut h.loopfilter;
    if h.all_lossless || h.allow_intrabc {
        lf.level_y = [0, 0];
        lf.level_u = 0;
        lf.level_v = 0;
        lf.sharpness = 0;
        lf.mode_ref_delta_enabled = true;
        lf.mode_ref_delta_update = true;
        lf.mode_ref_deltas = DEFAULT_MODE_REF_DELTAS;
    } else {
        lf.level_y[0] = gb.bits(6) as u8;
        lf.level_y[1] = gb.bits(6) as u8;
        if !seq.monochrome && (lf.level_y[0] != 0 || lf.level_y[1] != 0) {
            lf.level_u = gb.bits(6) as u8;
            lf.level_v = gb.bits(6) as u8;
        }
        lf.sharpness = gb.bits(3) as u8;
        lf.mode_ref_deltas = if h.primary_ref_frame == PRIMARY_REF_NONE {
            DEFAULT_MODE_REF_DELTAS
        } else {
            let r = h.refidx[h.primary_ref_frame as usize] as usize;
            refs[r].as_ref().ok_or("missing reference")?.loopfilter.mode_ref_deltas
        };
        lf.mode_ref_delta_enabled = gb.flag();
        if lf.mode_ref_delta_enabled {
            lf.mode_ref_delta_update = gb.flag();
            if lf.mode_ref_delta_update {
                for i in 0..8 {
                    if gb.flag() {
                        lf.mode_ref_deltas.ref_delta[i] = gb.sbits(7) as i8;
                    }
                }
                for i in 0..2 {
                    if gb.flag() {
                        lf.mode_ref_deltas.mode_delta[i] = gb.sbits(7) as i8;
                    }
                }
            }
        }
    }

    // CDEF.
    if !h.all_lossless && seq.cdef && !h.allow_intrabc {
        h.cdef.damping = gb.bits(2) as u8 + 3;
        h.cdef.n_bits = gb.bits(2) as u8;
        for i in 0..(1usize << h.cdef.n_bits) {
            h.cdef.y_strength[i] = gb.bits(6) as u8;
            if !seq.monochrome {
                h.cdef.uv_strength[i] = gb.bits(6) as u8;
            }
        }
    } else {
        h.cdef.n_bits = 0;
        h.cdef.y_strength[0] = 0;
        h.cdef.uv_strength[0] = 0;
    }

    // Loop restoration.
    let r = &mut h.restoration;
    if (!h.all_lossless || h.super_res_enabled) && seq.restoration && !h.allow_intrabc {
        r.typ[0] = gb.bits(2) as u8;
        if !seq.monochrome {
            r.typ[1] = gb.bits(2) as u8;
            r.typ[2] = gb.bits(2) as u8;
        }
        if r.typ[0] != 0 || r.typ[1] != 0 || r.typ[2] != 0 {
            r.unit_size[0] = 6 + seq.sb128 as u8;
            if gb.flag() {
                r.unit_size[0] += 1;
                if !seq.sb128 {
                    r.unit_size[0] += gb.bit() as u8;
                }
            }
            r.unit_size[1] = r.unit_size[0];
            if (r.typ[1] != 0 || r.typ[2] != 0) && seq.ss_hor == 1 && seq.ss_ver == 1 {
                r.unit_size[1] -= gb.bit() as u8;
            }
        } else {
            r.unit_size[0] = 8;
        }
    } else {
        r.typ = [RESTORATION_NONE; 3];
    }

    h.txfm_mode = if h.all_lossless {
        TX_4X4_ONLY
    } else if gb.flag() {
        TX_SWITCHABLE
    } else {
        TX_LARGEST
    };
    h.switchable_comp_refs = h.is_inter_or_switch() && gb.flag();
    h.skip_mode_allowed = false;
    if h.switchable_comp_refs && h.is_inter_or_switch() && seq.order_hint {
        let nb = seq.order_hint_n_bits;
        let poc = h.frame_offset as i32;
        let mut off_before: u32 = u32::MAX;
        let mut off_after: i32 = -1;
        let (mut before_idx, mut after_idx) = (0i8, 0i8);
        for i in 0..7 {
            let refpoc = rf(h.refidx[i] as usize)?.frame_offset as i32;
            let diff = get_poc_diff(nb, refpoc, poc);
            if diff > 0 {
                if off_after == -1 || get_poc_diff(nb, off_after, refpoc) > 0 {
                    off_after = refpoc;
                    after_idx = i as i8;
                }
            } else if diff < 0 && (off_before == u32::MAX || get_poc_diff(nb, refpoc, off_before as i32) > 0) {
                off_before = refpoc as u32;
                before_idx = i as i8;
            }
        }
        if off_before != u32::MAX && off_after != -1 {
            h.skip_mode_refs = [before_idx.min(after_idx), before_idx.max(after_idx)];
            h.skip_mode_allowed = true;
        } else if off_before != u32::MAX {
            let mut off_before2: u32 = u32::MAX;
            let mut before2_idx = 0i8;
            for i in 0..7 {
                let refpoc = rf(h.refidx[i] as usize)?.frame_offset as i32;
                if get_poc_diff(nb, refpoc, off_before as i32) < 0
                    && (off_before2 == u32::MAX || get_poc_diff(nb, refpoc, off_before2 as i32) > 0)
                {
                    off_before2 = refpoc as u32;
                    before2_idx = i as i8;
                }
            }
            if off_before2 != u32::MAX {
                h.skip_mode_refs = [before_idx.min(before2_idx), before_idx.max(before2_idx)];
                h.skip_mode_allowed = true;
            }
        }
    }
    h.skip_mode_enabled = h.skip_mode_allowed && gb.flag();
    h.warp_motion = !h.error_resilient_mode && h.is_inter_or_switch() && seq.warped_motion && gb.flag();
    h.reduced_txtp_set = gb.flag();

    h.gmv = [DEFAULT_WM_PARAMS; 7];
    if h.is_inter_or_switch() {
        for i in 0..7 {
            h.gmv[i].typ = if !gb.flag() {
                WM_IDENTITY
            } else if gb.flag() {
                WM_ROT_ZOOM
            } else if gb.flag() {
                WM_TRANSLATION
            } else {
                WM_AFFINE
            };
            if h.gmv[i].typ == WM_IDENTITY {
                continue;
            }
            let ref_mat = if h.primary_ref_frame == PRIMARY_REF_NONE {
                DEFAULT_WM_PARAMS.matrix
            } else {
                rf(h.refidx[h.primary_ref_frame as usize] as usize)?.gmv[i].matrix
            };
            let typ = h.gmv[i].typ;
            let mat = &mut h.gmv[i].matrix;
            let (bits, shift);
            if typ >= WM_ROT_ZOOM {
                mat[2] = (1 << 16) + 2 * gb.bits_subexp((ref_mat[2] - (1 << 16)) >> 1, 12);
                mat[3] = 2 * gb.bits_subexp(ref_mat[3] >> 1, 12);
                bits = 12;
                shift = 10;
            } else {
                bits = 9 - !h.hp as u32;
                shift = 13 + !h.hp as u32;
            }
            if typ == WM_AFFINE {
                mat[4] = 2 * gb.bits_subexp(ref_mat[4] >> 1, 12);
                mat[5] = (1 << 16) + 2 * gb.bits_subexp((ref_mat[5] - (1 << 16)) >> 1, 12);
            } else {
                mat[4] = -mat[3];
                mat[5] = mat[2];
            }
            mat[0] = gb.bits_subexp(ref_mat[0] >> shift, bits) * (1 << shift);
            mat[1] = gb.bits_subexp(ref_mat[1] >> shift, bits) * (1 << shift);
        }
    }

    h.film_grain_present = seq.film_grain_present && (h.show_frame || h.showable_frame) && gb.flag();
    if h.film_grain_present {
        let seed = gb.bits(16);
        h.film_grain_update = h.frame_type != FRAME_INTER || gb.flag();
        if !h.film_grain_update {
            let refidx = gb.bits(3) as i8;
            if !h.refidx.contains(&refidx) {
                return err("bad film grain reference");
            }
            h.film_grain = rf(refidx as usize)?.film_grain;
            h.film_grain.seed = seed;
        } else {
            let f = &mut h.film_grain;
            f.seed = seed;
            f.num_y_points = gb.bits(4) as i32;
            if f.num_y_points > 14 {
                return err("bad film grain");
            }
            for i in 0..f.num_y_points as usize {
                f.y_points[i][0] = gb.bits(8) as u8;
                if i > 0 && f.y_points[i - 1][0] >= f.y_points[i][0] {
                    return err("bad film grain");
                }
                f.y_points[i][1] = gb.bits(8) as u8;
            }
            f.chroma_scaling_from_luma = !seq.monochrome && gb.flag();
            if seq.monochrome || f.chroma_scaling_from_luma || (seq.ss_ver == 1 && seq.ss_hor == 1 && f.num_y_points == 0) {
                f.num_uv_points = [0, 0];
            } else {
                for pl in 0..2 {
                    f.num_uv_points[pl] = gb.bits(4) as i32;
                    if f.num_uv_points[pl] > 10 {
                        return err("bad film grain");
                    }
                    for i in 0..f.num_uv_points[pl] as usize {
                        f.uv_points[pl][i][0] = gb.bits(8) as u8;
                        if i > 0 && f.uv_points[pl][i - 1][0] >= f.uv_points[pl][i][0] {
                            return err("bad film grain");
                        }
                        f.uv_points[pl][i][1] = gb.bits(8) as u8;
                    }
                }
            }
            if seq.ss_hor == 1 && seq.ss_ver == 1 && (f.num_uv_points[0] != 0) != (f.num_uv_points[1] != 0) {
                return err("bad film grain");
            }
            f.scaling_shift = gb.bits(2) as i32 + 8;
            f.ar_coeff_lag = gb.bits(2) as i32;
            let num_y_pos = (2 * f.ar_coeff_lag * (f.ar_coeff_lag + 1)) as usize;
            if f.num_y_points != 0 {
                for i in 0..num_y_pos {
                    f.ar_coeffs_y[i] = (gb.bits(8) as i32 - 128) as i8;
                }
            }
            for pl in 0..2 {
                if f.num_uv_points[pl] != 0 || f.chroma_scaling_from_luma {
                    let num_uv_pos = num_y_pos + (f.num_y_points != 0) as usize;
                    for i in 0..num_uv_pos {
                        f.ar_coeffs_uv[pl][i] = (gb.bits(8) as i32 - 128) as i8;
                    }
                    if f.num_y_points == 0 {
                        f.ar_coeffs_uv[pl][num_uv_pos] = 0;
                    }
                }
            }
            f.ar_coeff_shift = gb.bits(2) as u64 + 6;
            f.grain_scale_shift = gb.bits(2) as i32;
            for pl in 0..2 {
                if f.num_uv_points[pl] != 0 {
                    f.uv_mult[pl] = gb.bits(8) as i32 - 128;
                    f.uv_luma_mult[pl] = gb.bits(8) as i32 - 128;
                    f.uv_offset[pl] = gb.bits(9) as i32 - 256;
                }
            }
            f.overlap_flag = gb.flag();
            f.clip_to_restricted_range = gb.flag();
        }
    } else {
        h.film_grain = FilmGrainData::default();
    }
    if gb.error {
        return err("frame header overrun");
    }
    Ok(h)
}

/// `parse_tile_hdr`: (start, end) tile indices of a tile group.
pub fn parse_tile_hdr(h: &FrameHeader, gb: &mut GetBits) -> (u32, u32) {
    let n_tiles = h.tiling.cols as u32 * h.tiling.rows as u32;
    let have_tile_pos = n_tiles > 1 && gb.flag();
    if have_tile_pos {
        let n = (h.tiling.log2_cols + h.tiling.log2_rows) as u32;
        let s = gb.bits(n);
        let e = gb.bits(n);
        (s, e)
    } else {
        (0, n_tiles - 1)
    }
}
