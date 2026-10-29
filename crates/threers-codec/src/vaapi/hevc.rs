//! HEVC on VAAPI: the native decoder's slice segments, reference picture
//! sets and lists translated into `VAPictureParameterBufferHEVC`,
//! `VAIQMatrixBufferHEVC` and `VASliceParameterBufferHEVC` (libva's
//! `va_dec_hevc.h`).

use super::{Device, Id, Session, BUF_IQ_MATRIX, BUF_PICTURE_PARAMETER, BUF_SLICE_DATA, BUF_SLICE_PARAMETER, INVALID_ID, PROFILE_HEVC_MAIN, PROFILE_HEVC_MAIN10, RT_FORMAT_YUV420, RT_FORMAT_YUV420_10};
use crate::decode::hevc::params::{ScalingFactors, SliceType, Sps};
use crate::decode::hevc::{AccelSlice, HevcAccel, Result};
use crate::decode::{ColorInfo, DecodedFrame};

const PIC_INVALID: u32 = 0x01;
const PIC_LONG_TERM: u32 = 0x08;
const PIC_ST_CURR_BEFORE: u32 = 0x10;
const PIC_ST_CURR_AFTER: u32 = 0x20;
const PIC_LT_CURR: u32 = 0x40;

/// `VAPictureHEVC`.
#[repr(C)]
#[derive(Clone, Copy)]
struct VaPicture {
    picture_id: Id,
    pic_order_cnt: i32,
    flags: u32,
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<VaPicture>() == 28);

impl VaPicture {
    const INVALID: VaPicture = VaPicture { picture_id: INVALID_ID, pic_order_cnt: 0, flags: PIC_INVALID, reserved: [0; 4] };
}

/// `VAPictureParameterBufferHEVC`.
#[repr(C)]
struct PictureParams {
    curr_pic: VaPicture,
    reference_frames: [VaPicture; 15],
    pic_width_in_luma_samples: u16,
    pic_height_in_luma_samples: u16,
    pic_fields: u32,
    sps_max_dec_pic_buffering_minus1: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    pcm_sample_bit_depth_luma_minus1: u8,
    pcm_sample_bit_depth_chroma_minus1: u8,
    log2_min_luma_coding_block_size_minus3: u8,
    log2_diff_max_min_luma_coding_block_size: u8,
    log2_min_transform_block_size_minus2: u8,
    log2_diff_max_min_transform_block_size: u8,
    log2_min_pcm_luma_coding_block_size_minus3: u8,
    log2_diff_max_min_pcm_luma_coding_block_size: u8,
    max_transform_hierarchy_depth_intra: u8,
    max_transform_hierarchy_depth_inter: u8,
    init_qp_minus26: i8,
    diff_cu_qp_delta_depth: u8,
    pps_cb_qp_offset: i8,
    pps_cr_qp_offset: i8,
    log2_parallel_merge_level_minus2: u8,
    num_tile_columns_minus1: u8,
    num_tile_rows_minus1: u8,
    column_width_minus1: [u16; 19],
    row_height_minus1: [u16; 21],
    slice_parsing_fields: u32,
    log2_max_pic_order_cnt_lsb_minus4: u8,
    num_short_term_ref_pic_sets: u8,
    num_long_term_ref_pic_sps: u8,
    num_ref_idx_l0_default_active_minus1: u8,
    num_ref_idx_l1_default_active_minus1: u8,
    pps_beta_offset_div2: i8,
    pps_tc_offset_div2: i8,
    num_extra_slice_header_bits: u8,
    st_rps_bits: u32,
    reserved: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<PictureParams>() == 604);
const _: () = assert!(std::mem::offset_of!(PictureParams, pic_width_in_luma_samples) == 448);
const _: () = assert!(std::mem::offset_of!(PictureParams, num_tile_columns_minus1) == 474);
const _: () = assert!(std::mem::offset_of!(PictureParams, column_width_minus1) == 476);
const _: () = assert!(std::mem::offset_of!(PictureParams, row_height_minus1) == 514);
const _: () = assert!(std::mem::offset_of!(PictureParams, slice_parsing_fields) == 556);
const _: () = assert!(std::mem::offset_of!(PictureParams, st_rps_bits) == 568);

/// `VAIQMatrixBufferHEVC`.
#[repr(C)]
struct IqMatrix {
    scaling_list_4x4: [[u8; 16]; 6],
    scaling_list_8x8: [[u8; 64]; 6],
    scaling_list_16x16: [[u8; 64]; 6],
    scaling_list_32x32: [[u8; 64]; 2],
    scaling_list_dc_16x16: [u8; 6],
    scaling_list_dc_32x32: [u8; 2],
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<IqMatrix>() == 1016);
const _: () = assert!(std::mem::offset_of!(IqMatrix, scaling_list_dc_16x16) == 992);

/// `VASliceParameterBufferHEVC`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SliceParams {
    slice_data_size: u32,
    slice_data_offset: u32,
    slice_data_flag: u32,
    slice_data_byte_offset: u32,
    slice_segment_address: u32,
    ref_pic_list: [[u8; 15]; 2],
    long_slice_flags: u32,
    collocated_ref_idx: u8,
    num_ref_idx_l0_active_minus1: u8,
    num_ref_idx_l1_active_minus1: u8,
    slice_qp_delta: i8,
    slice_cb_qp_offset: i8,
    slice_cr_qp_offset: i8,
    slice_beta_offset_div2: i8,
    slice_tc_offset_div2: i8,
    luma_log2_weight_denom: u8,
    delta_chroma_log2_weight_denom: i8,
    /// Per list, in C order: luma weights, luma offsets, chroma weights,
    /// chroma offsets (list 0's four arrays, then list 1's).
    weights: [ListWeights; 2],
    five_minus_max_num_merge_cand: u8,
    num_entry_point_offsets: u16,
    entry_offset_to_subset_array: u16,
    slice_data_num_emu_prevn_bytes: u16,
    reserved: [u32; 2],
}
const _: () = assert!(std::mem::size_of::<SliceParams>() == 264);
// `delta_luma_weight_l0` at 66, `ChromaOffsetL1` at 216.
const _: () = assert!(std::mem::offset_of!(SliceParams, weights) == 66);
const _: () = assert!(std::mem::offset_of!(SliceParams, weights) + std::mem::size_of::<ListWeights>() + std::mem::offset_of!(ListWeights, chroma_offset) == 216);

/// One list's prediction weights in `VASliceParameterBufferHEVC`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ListWeights {
    delta_luma_weight: [i8; 15],
    luma_offset: [i8; 15],
    delta_chroma_weight: [[i8; 2]; 15],
    chroma_offset: [[i8; 2]; 15],
}
const _: () = assert!(std::mem::offset_of!(SliceParams, ref_pic_list) == 20);
const _: () = assert!(std::mem::offset_of!(SliceParams, long_slice_flags) == 52);
const _: () = assert!(std::mem::offset_of!(SliceParams, collocated_ref_idx) == 56);
const _: () = assert!(std::mem::offset_of!(SliceParams, luma_log2_weight_denom) == 64);
const _: () = assert!(std::mem::offset_of!(SliceParams, five_minus_max_num_merge_cand) == 246);
const _: () = assert!(std::mem::offset_of!(SliceParams, num_entry_point_offsets) == 248);
const _: () = assert!(std::mem::offset_of!(SliceParams, slice_data_num_emu_prevn_bytes) == 252);

/// The [`HevcAccel`] that drives a VA context.
pub(crate) struct VaHevc {
    dev: Device,
    session: Option<Session>,
    target: Id,
    /// `dpb` index → `ReferenceFrames` index for the current picture.
    rf_index: Vec<u8>,
    /// The last slice segment, held until it is known whether it is the
    /// picture's last (`LastSliceOfPic`).
    pending: Option<(SliceParams, Vec<u8>)>,
}

impl VaHevc {
    pub(crate) fn new(dev: Device) -> Self {
        VaHevc { dev, session: None, target: INVALID_ID, rf_index: Vec::new(), pending: None }
    }

    fn session(&mut self, sps: &Sps) -> Result<()> {
        if sps.chroma_format_idc != 1 || sps.bit_depth_y != sps.bit_depth_c || !(8..=10).contains(&sps.bit_depth_y) {
            return Err("the VAAPI HEVC path decodes 8- and 10-bit 4:2:0 only".into());
        }
        let deep = sps.bit_depth_y > 8;
        let (profile, rt) = if deep { (PROFILE_HEVC_MAIN10, RT_FORMAT_YUV420_10) } else { (PROFILE_HEVC_MAIN, RT_FORMAT_YUV420) };
        let key = (sps.width, sps.height, profile, rt);
        if self.session.as_ref().is_some_and(|s| s.key == key) {
            return Ok(());
        }
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
        let n = (sps.max_dec_pic_buffering as usize + 3).clamp(6, 20);
        self.session = Some(Session::new(&self.dev, profile, rt, sps.width, sps.height, n)?);
        Ok(())
    }

    fn submit(&mut self, mut sp: SliceParams, data: &[u8], last: bool) -> Result<()> {
        sp.long_slice_flags |= last as u32;
        let (dev, ses) = (&self.dev, self.session.as_mut().ok_or("no VA context")?);
        let mut ids = [ses.buffer(dev, BUF_SLICE_PARAMETER, &sp)?, ses.raw_buffer(dev, BUF_SLICE_DATA, data.len(), data.as_ptr().cast())?];
        ses.render(dev, &mut ids)
    }

    fn begin(&mut self, s: &AccelSlice<'_>) -> Result<()> {
        let (sps, pps, sh) = (s.sps, s.pps, s.sh);
        self.session(sps)?;
        let dpb = s.dpb;
        let ses = self.session.as_mut().unwrap();
        self.target = ses.claim(s.poc as i64, |k| dpb.iter().any(|p| p.poc as i64 == k))?;

        let mut pp: PictureParams = unsafe { std::mem::zeroed() };
        pp.curr_pic = VaPicture { picture_id: self.target, pic_order_cnt: s.poc, flags: 0, reserved: [0; 4] };
        pp.reference_frames = [VaPicture::INVALID; 15];
        self.rf_index = vec![0xff; dpb.len()];
        let mut n = 0;
        for (i, p) in dpb.iter().enumerate() {
            let Some(id) = ses.surface_of(p.poc as i64) else { continue };
            if n == 15 {
                break;
            }
            let mut flags = if p.long_term { PIC_LONG_TERM } else { 0 };
            if s.st_before.contains(&i) {
                flags |= PIC_ST_CURR_BEFORE;
            } else if s.st_after.contains(&i) {
                flags |= PIC_ST_CURR_AFTER;
            } else if s.lt_curr.contains(&i) {
                flags |= PIC_LT_CURR;
            }
            pp.reference_frames[n] = VaPicture { picture_id: id, pic_order_cnt: p.poc, flags, reserved: [0; 4] };
            self.rf_index[i] = n as u8;
            n += 1;
        }
        pp.pic_width_in_luma_samples = sps.width as u16;
        pp.pic_height_in_luma_samples = sps.height as u16;
        let nal = s.nal_type;
        let irap = (16..=23).contains(&nal);
        pp.pic_fields = sps.chroma_format_idc
            | (sps.pcm as u32) << 3
            | (sps.scaling.is_some() as u32) << 4
            | (pps.transform_skip as u32) << 5
            | (sps.amp as u32) << 6
            | (sps.strong_intra_smoothing as u32) << 7
            | (pps.sign_data_hiding as u32) << 8
            | (pps.constrained_intra_pred as u32) << 9
            | (pps.cu_qp_delta as u32) << 10
            | (pps.weighted_pred as u32) << 11
            | (pps.weighted_bipred as u32) << 12
            | (pps.transquant_bypass as u32) << 13
            | (pps.tiles as u32) << 14
            | (pps.entropy_sync as u32) << 15
            | (pps.loop_filter_across_slices as u32) << 16
            | (pps.loop_filter_across_tiles as u32) << 17
            | (sps.pcm_loop_filter_disabled as u32) << 18
            | ((sps.max_num_reorder == 0) as u32) << 19;
        pp.sps_max_dec_pic_buffering_minus1 = sps.max_dec_pic_buffering.saturating_sub(1) as u8;
        pp.bit_depth_luma_minus8 = (sps.bit_depth_y - 8) as u8;
        pp.bit_depth_chroma_minus8 = (sps.bit_depth_c - 8) as u8;
        if sps.pcm {
            pp.pcm_sample_bit_depth_luma_minus1 = (sps.pcm_bit_depth_y - 1) as u8;
            pp.pcm_sample_bit_depth_chroma_minus1 = (sps.pcm_bit_depth_c - 1) as u8;
            pp.log2_min_pcm_luma_coding_block_size_minus3 = (sps.log2_min_pcm - 3) as u8;
            pp.log2_diff_max_min_pcm_luma_coding_block_size = (sps.log2_max_pcm - sps.log2_min_pcm) as u8;
        }
        pp.log2_min_luma_coding_block_size_minus3 = (sps.log2_min_cb - 3) as u8;
        pp.log2_diff_max_min_luma_coding_block_size = (sps.log2_ctb - sps.log2_min_cb) as u8;
        pp.log2_min_transform_block_size_minus2 = (sps.log2_min_tb - 2) as u8;
        pp.log2_diff_max_min_transform_block_size = (sps.log2_max_tb - sps.log2_min_tb) as u8;
        pp.max_transform_hierarchy_depth_intra = sps.max_th_depth_intra as u8;
        pp.max_transform_hierarchy_depth_inter = sps.max_th_depth_inter as u8;
        pp.init_qp_minus26 = (pps.init_qp - 26) as i8;
        pp.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth as u8;
        pp.pps_cb_qp_offset = pps.cb_qp_offset as i8;
        pp.pps_cr_qp_offset = pps.cr_qp_offset as i8;
        pp.log2_parallel_merge_level_minus2 = (pps.log2_parallel_merge_level - 2) as u8;
        if pps.tiles {
            let cols = pps.tile_cols.boundaries(sps.width_ctbs());
            let rows = pps.tile_rows.boundaries(sps.height_ctbs());
            pp.num_tile_columns_minus1 = (cols.len() - 2) as u8;
            pp.num_tile_rows_minus1 = (rows.len() - 2) as u8;
            for (i, w) in cols.windows(2).take(19).enumerate() {
                pp.column_width_minus1[i] = (w[1] - w[0]).saturating_sub(1) as u16;
            }
            for (i, h) in rows.windows(2).take(21).enumerate() {
                pp.row_height_minus1[i] = (h[1] - h[0]).saturating_sub(1) as u16;
            }
        }
        pp.slice_parsing_fields = pps.lists_modification_present as u32
            | (sps.long_term_present as u32) << 1
            | (sps.temporal_mvp as u32) << 2
            | (pps.cabac_init_present as u32) << 3
            | (pps.output_flag_present as u32) << 4
            | (pps.dependent_slices as u32) << 5
            | (pps.slice_chroma_qp_offsets_present as u32) << 6
            | (sps.sao as u32) << 7
            | (pps.deblocking_override_enabled as u32) << 8
            | (pps.deblocking_disabled as u32) << 9
            | (pps.slice_header_extension_present as u32) << 10
            | (irap as u32) << 11
            | ((nal == 19 || nal == 20) as u32) << 12
            // IntraPicFlag, as the reference clients set it: the picture is IRAP.
            | (irap as u32) << 13;
        pp.log2_max_pic_order_cnt_lsb_minus4 = (sps.log2_max_poc_lsb - 4) as u8;
        pp.num_short_term_ref_pic_sets = sps.st_rps.len() as u8;
        pp.num_long_term_ref_pic_sps = sps.lt_ref_pics.len() as u8;
        pp.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_default[0].saturating_sub(1) as u8;
        pp.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_default[1].saturating_sub(1) as u8;
        pp.pps_beta_offset_div2 = (pps.beta_offset / 2) as i8;
        pp.pps_tc_offset_div2 = (pps.tc_offset / 2) as i8;
        pp.num_extra_slice_header_bits = pps.extra_slice_header_bits as u8;
        pp.st_rps_bits = sh.st_rps_bits;

        let (dev, ses) = (&self.dev, self.session.as_mut().unwrap());
        ses.begin(dev, self.target)?;
        let mut ids = vec![ses.buffer(dev, BUF_PICTURE_PARAMETER, &pp)?];
        if let Some(f) = pps.scaling.as_ref().or(sps.scaling.as_ref()) {
            ids.push(ses.buffer(dev, BUF_IQ_MATRIX, &iq_matrix(f))?);
        }
        ses.render(dev, &mut ids)
    }
}

/// The coded lists back out of the expanded factors: 4×4 as they are, the
/// larger sizes as the 8×8 grid they were upsampled from, with the DC that
/// replaced their first entry.
fn iq_matrix(f: &ScalingFactors) -> IqMatrix {
    let mut iq: IqMatrix = unsafe { std::mem::zeroed() };
    let coded = |m: &[u8], n: usize| -> [u8; 64] {
        let rep = n / 8;
        std::array::from_fn(|i| {
            let (y, x) = (i / 8, i % 8);
            // (0, 0) of a 16×16 or 32×32 matrix holds the DC; its neighbour
            // carries the coded value.
            if i == 0 && n > 8 { m[1] } else { m[y * rep * n + x * rep] }
        })
    };
    for m in 0..6 {
        iq.scaling_list_4x4[m].copy_from_slice(&f.m[0][m][..16]);
        iq.scaling_list_8x8[m] = coded(&f.m[1][m], 8);
        iq.scaling_list_16x16[m] = coded(&f.m[2][m], 16);
        iq.scaling_list_dc_16x16[m] = f.m[2][m][0];
    }
    for (i, m) in [0, 3].into_iter().enumerate() {
        iq.scaling_list_32x32[i] = coded(&f.m[3][m], 32);
        iq.scaling_list_dc_32x32[i] = f.m[3][m][0];
    }
    iq
}

impl HevcAccel for VaHevc {
    fn slice(&mut self, s: &AccelSlice<'_>) -> Result<()> {
        if s.first {
            self.begin(s)?;
        }
        if let Some((sp, data)) = self.pending.take() {
            self.submit(sp, &data, false)?;
        }
        let (sps, sh) = (s.sps, s.sh);
        let mut sp: SliceParams = unsafe { std::mem::zeroed() };
        sp.slice_data_size = s.nal.len() as u32;
        // The convention drivers are built against (ffmpeg's): the offset
        // counts the header without its emulation prevention bytes, and the
        // bytes it leaves out are given separately. Mesa ignores both; Intel
        // adds the two to find the slice data.
        let (_, epb) = super::coded_offset(s.nal, 2, sh.data_offset);
        sp.slice_data_byte_offset = (2 + sh.data_offset) as u32;
        sp.slice_data_num_emu_prevn_bytes = epb as u16;
        sp.slice_segment_address = sh.segment_address;
        sp.ref_pic_list = [[0xff; 15]; 2];
        for l in 0..2 {
            for (i, e) in s.refs[l].iter().take(15).enumerate() {
                sp.ref_pic_list[l][i] = self.rf_index.get(e.dpb).copied().unwrap_or(0xff);
            }
        }
        let slice_type = match sh.slice_type {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        };
        sp.long_slice_flags = (sh.dependent as u32) << 1
            | slice_type << 2
            | (sh.sao_luma as u32) << 6
            | (sh.sao_chroma as u32) << 7
            | (sh.mvd_l1_zero as u32) << 8
            | (sh.cabac_init_flag as u32) << 9
            | (sh.temporal_mvp as u32) << 10
            | (sh.deblocking_disabled as u32) << 11
            | (sh.collocated_from_l0 as u32) << 12
            | (sh.loop_filter_across_slices as u32) << 13;
        sp.collocated_ref_idx = if sh.temporal_mvp { sh.collocated_ref_idx as u8 } else { 0xff };
        sp.num_ref_idx_l0_active_minus1 = sh.num_ref_idx[0].saturating_sub(1) as u8;
        sp.num_ref_idx_l1_active_minus1 = sh.num_ref_idx[1].saturating_sub(1) as u8;
        sp.slice_qp_delta = sh.qp_delta as i8;
        sp.slice_cb_qp_offset = sh.cb_qp_offset as i8;
        sp.slice_cr_qp_offset = sh.cr_qp_offset as i8;
        sp.slice_beta_offset_div2 = (sh.beta_offset / 2) as i8;
        sp.slice_tc_offset_div2 = (sh.tc_offset / 2) as i8;
        if let Some(w) = &sh.weights {
            let (ld, cd) = (w.luma_log2_denom as i32, w.chroma_log2_denom as i32);
            let (sy, sc) = (sps.bit_depth_y as i32 - 8, sps.bit_depth_c as i32 - 8);
            sp.luma_log2_weight_denom = ld as u8;
            sp.delta_chroma_log2_weight_denom = (cd - ld) as i8;
            for l in 0..2 {
                for (i, e) in w.w[l].iter().take(15).enumerate() {
                    let lw = &mut sp.weights[l];
                    lw.delta_luma_weight[i] = (e[0].0 - (1 << ld)) as i8;
                    lw.luma_offset[i] = (e[0].1 >> sy) as i8;
                    for c in 0..2 {
                        lw.delta_chroma_weight[i][c] = (e[1 + c].0 - (1 << cd)) as i8;
                        lw.chroma_offset[i][c] = (e[1 + c].1 >> sc) as i8;
                    }
                }
            }
        }
        sp.five_minus_max_num_merge_cand = (5 - sh.max_merge_cand) as u8;
        // Left at 0, as ffmpeg leaves it: without the offsets themselves
        // (a separate subset buffer) the count is no use to a driver.
        sp.num_entry_point_offsets = 0;
        self.pending = Some((sp, s.nal.to_vec()));
        Ok(())
    }

    fn end_picture(&mut self) -> Result<()> {
        if let Some((sp, data)) = self.pending.take() {
            self.submit(sp, &data, true)?;
        }
        self.session.as_mut().ok_or("no VA context")?.end(&self.dev)
    }

    fn frame(&mut self, poc: i32, sps: &Sps) -> Result<DecodedFrame> {
        let ses = self.session.as_ref().ok_or("no VA context")?;
        let surface = ses.surface_of(poc as i64).ok_or("no VA surface holds that picture")?;
        let (w, h, ..) = ses.key;
        let deep = sps.bit_depth_y > 8;
        let crop = sps.crop.map(|v| v as usize);
        let color = ColorInfo::from_vui(sps.matrix_coefficients, sps.full_range, sps.height - sps.crop[2] - sps.crop[3]);
        self.dev.read_frame(surface, w as usize, h as usize, deep, crop, color)
    }
}

impl Drop for VaHevc {
    fn drop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
    }
}
