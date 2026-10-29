//! H.264 on VAAPI: the native decoder's slices, reference lists and DPB
//! translated into `VAPictureParameterBufferH264`, `VAIQMatrixBufferH264`
//! and `VASliceParameterBufferH264` (the layout of libva's `va.h`).

use super::{Device, Id, Session, BUF_IQ_MATRIX, BUF_PICTURE_PARAMETER, BUF_SLICE_DATA, BUF_SLICE_PARAMETER, INVALID_ID, PROFILE_H264_HIGH, RT_FORMAT_YUV420};
use crate::decode::h264::params::{ScalingLists, SliceType, Sps};
use crate::decode::h264::types::RefPic;
use crate::decode::h264::{AccelSlice, H264Accel, Result};
use crate::decode::{ColorInfo, DecodedFrame};

const PIC_INVALID: u32 = 0x01;
const PIC_TOP_FIELD: u32 = 0x02;
const PIC_BOTTOM_FIELD: u32 = 0x04;
const PIC_SHORT_TERM: u32 = 0x08;
const PIC_LONG_TERM: u32 = 0x10;

/// `VAPictureH264`.
#[repr(C)]
#[derive(Clone, Copy)]
struct VaPicture {
    picture_id: Id,
    frame_idx: u32,
    flags: u32,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<VaPicture>() == 36);

impl VaPicture {
    const INVALID: VaPicture = VaPicture {
        picture_id: INVALID_ID,
        frame_idx: 0,
        flags: PIC_INVALID,
        top_field_order_cnt: 0,
        bottom_field_order_cnt: 0,
        reserved: [0; 4],
    };
}

/// `VAPictureParameterBufferH264`.
#[repr(C)]
struct PictureParams {
    curr_pic: VaPicture,
    reference_frames: [VaPicture; 16],
    picture_width_in_mbs_minus1: u16,
    picture_height_in_mbs_minus1: u16,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    num_ref_frames: u8,
    seq_fields: u32,
    num_slice_groups_minus1: u8,
    slice_group_map_type: u8,
    slice_group_change_rate_minus1: u16,
    pic_init_qp_minus26: i8,
    pic_init_qs_minus26: i8,
    chroma_qp_index_offset: i8,
    second_chroma_qp_index_offset: i8,
    pic_fields: u32,
    frame_num: u16,
    reserved: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<PictureParams>() == 672);
const _: () = assert!(std::mem::offset_of!(PictureParams, seq_fields) == 620);
const _: () = assert!(std::mem::offset_of!(PictureParams, pic_fields) == 632);
const _: () = assert!(std::mem::offset_of!(PictureParams, frame_num) == 636);

/// `VAIQMatrixBufferH264`.
#[repr(C)]
struct IqMatrix {
    scaling_list_4x4: [[u8; 16]; 6],
    scaling_list_8x8: [[u8; 64]; 2],
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<IqMatrix>() == 240);

/// `VASliceParameterBufferH264`.
#[repr(C)]
struct SliceParams {
    slice_data_size: u32,
    slice_data_offset: u32,
    slice_data_flag: u32,
    slice_data_bit_offset: u16,
    first_mb_in_slice: u16,
    slice_type: u8,
    direct_spatial_mv_pred_flag: u8,
    num_ref_idx_l0_active_minus1: u8,
    num_ref_idx_l1_active_minus1: u8,
    cabac_init_idc: u8,
    slice_qp_delta: i8,
    disable_deblocking_filter_idc: u8,
    slice_alpha_c0_offset_div2: i8,
    slice_beta_offset_div2: i8,
    ref_pic_list: [[VaPicture; 32]; 2],
    luma_log2_weight_denom: u8,
    chroma_log2_weight_denom: u8,
    luma_weight_l0_flag: u8,
    luma_weight_l0: [i16; 32],
    luma_offset_l0: [i16; 32],
    chroma_weight_l0_flag: u8,
    chroma_weight_l0: [[i16; 2]; 32],
    chroma_offset_l0: [[i16; 2]; 32],
    luma_weight_l1_flag: u8,
    luma_weight_l1: [i16; 32],
    luma_offset_l1: [i16; 32],
    chroma_weight_l1_flag: u8,
    chroma_weight_l1: [[i16; 2]; 32],
    chroma_offset_l1: [[i16; 2]; 32],
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<SliceParams>() == 3128);
const _: () = assert!(std::mem::offset_of!(SliceParams, ref_pic_list) == 28);
const _: () = assert!(std::mem::offset_of!(SliceParams, luma_weight_l0) == 2336);
const _: () = assert!(std::mem::offset_of!(SliceParams, chroma_weight_l0) == 2466);
const _: () = assert!(std::mem::offset_of!(SliceParams, luma_weight_l1_flag) == 2722);
const _: () = assert!(std::mem::offset_of!(SliceParams, chroma_offset_l1) == 2982);

/// The [`H264Accel`] that drives a VA context.
pub(crate) struct VaH264 {
    dev: Device,
    session: Option<Session>,
    /// Surface of the picture being decoded.
    target: Id,
}

impl VaH264 {
    pub(crate) fn new(dev: Device) -> Self {
        VaH264 { dev, session: None, target: INVALID_ID }
    }

    /// A context for the SPS's coded size, created (or recreated) on demand.
    fn session(&mut self, sps: &Sps) -> Result<()> {
        if sps.chroma_format_idc != 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err("the VAAPI H.264 path decodes 8-bit 4:2:0 only".into());
        }
        let (w, h) = (sps.width_mbs * 16, sps.height_mbs * 16);
        let key = (w, h, PROFILE_H264_HIGH, RT_FORMAT_YUV420);
        if self.session.as_ref().is_some_and(|s| s.key == key) {
            return Ok(());
        }
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
        // References plus the picture being decoded and a field waiting for
        // its pair, with room for the driver to pipeline.
        let n = (sps.max_num_ref_frames as usize + 4).clamp(6, 24);
        self.session = Some(Session::new(&self.dev, PROFILE_H264_HIGH, RT_FORMAT_YUV420, w, h, n)?);
        Ok(())
    }

    fn surface_of(&self, store_id: u32) -> Option<Id> {
        self.session.as_ref()?.surface_of(store_id as i64)
    }

    /// A DPB entry, or a list entry of the given structure.
    fn picture(&self, p: &RefPic, structure: u8, long: bool, reference: bool) -> VaPicture {
        let Some(id) = self.surface_of(p.id) else {
            return VaPicture::INVALID;
        };
        let mut flags = match structure {
            1 => PIC_TOP_FIELD,
            2 => PIC_BOTTOM_FIELD,
            _ => 0,
        };
        if reference {
            flags |= if long { PIC_LONG_TERM } else { PIC_SHORT_TERM };
        }
        VaPicture {
            picture_id: id,
            frame_idx: if long { p.lt_idx } else { p.frame_num },
            flags,
            top_field_order_cnt: if structure & 1 != 0 { p.field_poc[0] } else { 0 },
            bottom_field_order_cnt: if structure & 2 != 0 { p.field_poc[1] } else { 0 },
            reserved: [0; 4],
        }
    }

    fn begin(&mut self, s: &AccelSlice<'_>) -> Result<()> {
        let (sps, pps, hdr) = (s.sps, s.pps, s.hdr);
        self.session(sps)?;
        let dpb = s.dpb;
        self.target = self.session.as_mut().unwrap().claim(s.store_id as i64, |k| dpb.iter().any(|p| p.id as i64 == k))?;
        let mut pp: PictureParams = unsafe { std::mem::zeroed() };
        let long_cur = hdr.long_term_reference;
        pp.curr_pic = VaPicture {
            picture_id: self.target,
            frame_idx: hdr.frame_num,
            flags: match s.structure {
                1 => PIC_TOP_FIELD,
                2 => PIC_BOTTOM_FIELD,
                _ => 0,
            } | if hdr.nal_ref_idc != 0 { if long_cur { PIC_LONG_TERM } else { PIC_SHORT_TERM } } else { 0 },
            top_field_order_cnt: if s.structure & 1 != 0 { s.field_poc[0] } else { 0 },
            bottom_field_order_cnt: if s.structure & 2 != 0 { s.field_poc[1] } else { 0 },
            reserved: [0; 4],
        };
        pp.reference_frames = [VaPicture::INVALID; 16];
        let mut n = 0;
        for p in s.dpb.iter().filter(|p| p.is_ref()) {
            if n == 16 {
                break;
            }
            let long = p.long != 0;
            let marked = if long { p.long } else { p.short };
            let v = self.picture(p, marked, long, true);
            if v.picture_id != INVALID_ID {
                pp.reference_frames[n] = v;
                n += 1;
            }
        }
        pp.picture_width_in_mbs_minus1 = (sps.width_mbs - 1) as u16;
        // The frame's height in macroblocks, field-coded or not, as the
        // drivers read it (ffmpeg's convention; Intel depends on it).
        pp.picture_height_in_mbs_minus1 = (sps.height_mbs - 1) as u16;
        pp.bit_depth_luma_minus8 = (sps.bit_depth_luma - 8) as u8;
        pp.bit_depth_chroma_minus8 = (sps.bit_depth_chroma - 8) as u8;
        pp.num_ref_frames = sps.max_num_ref_frames as u8;
        pp.seq_fields = sps.chroma_format_idc
            | (sps.gaps_allowed as u32) << 3
            | (sps.frame_mbs_only as u32) << 4
            | (sps.mbaff as u32) << 5
            | (sps.direct_8x8_inference as u32) << 6
            | (sps.log2_max_frame_num - 4) << 8
            | sps.poc_type << 12
            | (if sps.poc_type == 0 { sps.log2_max_poc_lsb - 4 } else { 0 }) << 14
            | (sps.delta_pic_order_always_zero as u32) << 18;
        pp.pic_init_qp_minus26 = (pps.pic_init_qp - 26) as i8;
        pp.chroma_qp_index_offset = pps.chroma_qp_offset[0] as i8;
        pp.second_chroma_qp_index_offset = pps.chroma_qp_offset[1] as i8;
        pp.pic_fields = pps.cabac as u32
            | (pps.weighted_pred as u32) << 1
            | pps.weighted_bipred_idc << 2
            | (pps.transform_8x8_mode as u32) << 4
            | (hdr.field_pic as u32) << 5
            | (pps.constrained_intra_pred as u32) << 6
            | (pps.bottom_field_pic_order_in_frame_present as u32) << 7
            | (pps.deblocking_filter_control_present as u32) << 8
            | (pps.redundant_pic_cnt_present as u32) << 9
            | ((hdr.nal_ref_idc != 0) as u32) << 10;
        pp.frame_num = hdr.frame_num as u16;

        let flat = ScalingLists::flat();
        let lists = pps.scaling.as_ref().or(sps.scaling.as_ref()).unwrap_or(&flat);
        let iq = IqMatrix {
            scaling_list_4x4: lists.m4,
            scaling_list_8x8: [lists.m8[0], lists.m8[1]],
            reserved: [0; 4],
        };

        let (dev, ses) = (&self.dev, self.session.as_mut().unwrap());
        ses.begin(dev, self.target)?;
        let mut ids = [ses.buffer(dev, BUF_PICTURE_PARAMETER, &pp)?, ses.buffer(dev, BUF_IQ_MATRIX, &iq)?];
        ses.render(dev, &mut ids)
    }
}

/// Bit offset of RBSP bit `rbsp_bits` (counted after the header byte) within
/// the coded NAL unit, emulation prevention bytes included.
fn coded_bit_offset(nal: &[u8], rbsp_bits: usize) -> usize {
    super::coded_offset(nal, 1, rbsp_bits / 8).0 * 8 + rbsp_bits % 8
}

impl H264Accel for VaH264 {
    fn slice(&mut self, s: &AccelSlice<'_>) -> Result<()> {
        if s.first {
            self.begin(s)?;
        }
        let hdr = s.hdr;
        let mut sp: SliceParams = unsafe { std::mem::zeroed() };
        sp.slice_data_size = s.nal.len() as u32;
        sp.slice_data_bit_offset = coded_bit_offset(s.nal, hdr.data_bit_offset) as u16;
        sp.first_mb_in_slice = hdr.first_mb as u16;
        sp.slice_type = match hdr.slice_type {
            SliceType::P => 0,
            SliceType::B => 1,
            SliceType::I => 2,
        };
        sp.direct_spatial_mv_pred_flag = hdr.direct_spatial as u8;
        sp.num_ref_idx_l0_active_minus1 = hdr.num_ref_idx[0].saturating_sub(1) as u8;
        sp.num_ref_idx_l1_active_minus1 = hdr.num_ref_idx[1].saturating_sub(1) as u8;
        sp.cabac_init_idc = hdr.cabac_init_idc as u8;
        sp.slice_qp_delta = (hdr.qp - s.pps.pic_init_qp) as i8;
        sp.disable_deblocking_filter_idc = hdr.disable_deblocking as u8;
        sp.slice_alpha_c0_offset_div2 = (hdr.alpha_offset / 2) as i8;
        sp.slice_beta_offset_div2 = (hdr.beta_offset / 2) as i8;
        sp.ref_pic_list = [[VaPicture::INVALID; 32]; 2];
        for l in 0..2 {
            for (i, e) in s.refs[l].iter().take(32).enumerate() {
                sp.ref_pic_list[l][i] = self.picture(&s.dpb[e.dpb], e.structure, e.long_term, true);
            }
        }
        // Default tables where the slice codes none, as other clients send
        // them: weight 1 << denom, offset 0, denominator 5 for implicit
        // bi-prediction. The flags stay 0, so they change nothing.
        if hdr.weights.is_none() {
            let implicit = hdr.slice_type == SliceType::B && s.pps.weighted_bipred_idc == 2;
            let denom = if implicit { 5 } else { 0 };
            sp.luma_log2_weight_denom = denom;
            sp.chroma_log2_weight_denom = denom;
            let one = 1i16 << denom;
            for i in 0..32 {
                sp.luma_weight_l0[i] = one;
                sp.luma_weight_l1[i] = one;
                sp.chroma_weight_l0[i] = [one; 2];
                sp.chroma_weight_l1[i] = [one; 2];
            }
        }
        if let Some(w) = &hdr.weights {
            sp.luma_log2_weight_denom = w.luma_log2_denom as u8;
            sp.chroma_log2_weight_denom = w.chroma_log2_denom as u8;
            let lists = if hdr.slice_type == SliceType::B { 2 } else { 1 };
            for l in 0..lists {
                for (i, wc) in w.weights[l].iter().take(32).enumerate() {
                    let (luma_w, luma_o) = wc[0];
                    let chroma = [[wc[1].0 as i16, wc[2].0 as i16], [wc[1].1 as i16, wc[2].1 as i16]];
                    if l == 0 {
                        sp.luma_weight_l0_flag = 1;
                        sp.chroma_weight_l0_flag = 1;
                        sp.luma_weight_l0[i] = luma_w as i16;
                        sp.luma_offset_l0[i] = luma_o as i16;
                        sp.chroma_weight_l0[i] = chroma[0];
                        sp.chroma_offset_l0[i] = chroma[1];
                    } else {
                        sp.luma_weight_l1_flag = 1;
                        sp.chroma_weight_l1_flag = 1;
                        sp.luma_weight_l1[i] = luma_w as i16;
                        sp.luma_offset_l1[i] = luma_o as i16;
                        sp.chroma_weight_l1[i] = chroma[0];
                        sp.chroma_offset_l1[i] = chroma[1];
                    }
                }
            }
        }
        let (dev, ses) = (&self.dev, self.session.as_mut().ok_or("no VA context")?);
        let mut ids = [
            ses.buffer(dev, BUF_SLICE_PARAMETER, &sp)?,
            ses.raw_buffer(dev, BUF_SLICE_DATA, s.nal.len(), s.nal.as_ptr().cast())?,
        ];
        ses.render(dev, &mut ids)
    }

    fn end_picture(&mut self) -> Result<()> {
        self.session.as_mut().ok_or("no VA context")?.end(&self.dev)
    }

    fn frame(&mut self, store_id: u32, sps: &Sps) -> Result<DecodedFrame> {
        let surface = self.surface_of(store_id).ok_or("no VA surface holds that picture")?;
        let (w, h, ..) = self.session.as_ref().unwrap().key;
        let crop = sps.crop.map(|v| v as usize);
        let color = ColorInfo::from_vui(sps.matrix_coefficients, sps.full_range, h - sps.crop[2] - sps.crop[3]);
        self.dev.read_frame(surface, w as usize, h as usize, false, crop, color)
    }
}

impl Drop for VaH264 {
    fn drop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
    }
}
