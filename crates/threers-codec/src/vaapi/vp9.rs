//! VP9 on VAAPI: the native decoder's uncompressed header translated into
//! `VADecPictureParameterBufferVP9` and `VASliceParameterBufferVP9`
//! (libva's `va_dec_vp9.h`). The driver parses the compressed header and
//! adapts probabilities itself.

use std::sync::Arc;

use super::{Device, Id, Session, BUF_PICTURE_PARAMETER, BUF_SLICE_DATA, BUF_SLICE_PARAMETER, INVALID_ID, RT_FORMAT_YUV420, RT_FORMAT_YUV420_10};
use crate::decode::vp9::{AccelFrame, Frame, Vp9Accel};
use crate::decode::{ColorInfo, DecodedFrame};

pub(crate) const PROFILE_VP9_0: i32 = 19;
const PROFILE_VP9_2: i32 = 21;

/// `VADecPictureParameterBufferVP9`.
#[repr(C)]
struct PictureParams {
    frame_width: u16,
    frame_height: u16,
    reference_frames: [Id; 8],
    pic_fields: u32,
    filter_level: u8,
    sharpness_level: u8,
    log2_tile_rows: u8,
    log2_tile_columns: u8,
    frame_header_length_in_bytes: u8,
    first_partition_size: u16,
    mb_segment_tree_probs: [u8; 7],
    segment_pred_probs: [u8; 3],
    profile: u8,
    bit_depth: u8,
    reserved: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<PictureParams>() == 92);
const _: () = assert!(std::mem::offset_of!(PictureParams, pic_fields) == 36);
const _: () = assert!(std::mem::offset_of!(PictureParams, first_partition_size) == 46);
const _: () = assert!(std::mem::offset_of!(PictureParams, segment_pred_probs) == 55);
const _: () = assert!(std::mem::offset_of!(PictureParams, bit_depth) == 59);

/// `VASegmentParameterVP9`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SegmentParams {
    segment_flags: u16,
    filter_level: [[u8; 2]; 4],
    luma_ac_quant_scale: i16,
    luma_dc_quant_scale: i16,
    chroma_ac_quant_scale: i16,
    chroma_dc_quant_scale: i16,
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<SegmentParams>() == 36);
const _: () = assert!(std::mem::offset_of!(SegmentParams, luma_ac_quant_scale) == 10);

/// `VASliceParameterBufferVP9`.
#[repr(C)]
struct SliceParams {
    slice_data_size: u32,
    slice_data_offset: u32,
    slice_data_flag: u32,
    seg_param: [SegmentParams; 8],
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<SliceParams>() == 316);

fn key(f: &Frame) -> i64 {
    f as *const Frame as usize as i64
}

/// The [`Vp9Accel`] that drives a VA context.
pub(crate) struct VaVp9 {
    dev: Device,
    session: Option<Session>,
    /// Shown frames not yet read back: their surfaces must not be reused.
    pinned: Vec<Arc<Frame>>,
}

impl VaVp9 {
    pub(crate) fn new(dev: Device) -> Self {
        VaVp9 { dev, session: None, pinned: Vec::new() }
    }

    fn session(&mut self, f: &Frame) -> Result<(), String> {
        if (f.ss_h, f.ss_v) != (1, 1) || !(f.bpp == 8 || f.bpp == 10) {
            return Err("the VAAPI VP9 path decodes 8- and 10-bit 4:2:0 only".into());
        }
        let (profile, rt) = if f.bpp > 8 { (PROFILE_VP9_2, RT_FORMAT_YUV420_10) } else { (PROFILE_VP9_0, RT_FORMAT_YUV420) };
        let key = (f.width, f.height, profile, rt);
        if self.session.as_ref().is_some_and(|s| s.key == key) {
            return Ok(());
        }
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
        // Eight reference slots, the frame being decoded, and shown frames
        // waiting to be read.
        self.session = Some(Session::new(&self.dev, profile, rt, f.width, f.height, 12)?);
        Ok(())
    }
}

impl Vp9Accel for VaVp9 {
    fn frame(&mut self, a: &AccelFrame<'_>) -> Result<(), String> {
        let (h, cur) = (a.h, a.cur);
        self.session(cur)?;
        let live: Vec<i64> = a.refs.iter().flatten().map(|f| key(f)).chain(self.pinned.iter().map(|f| key(f))).collect();
        let ses = self.session.as_mut().unwrap();
        let target = ses.claim(key(cur), |k| live.contains(&k))?;

        let mut pp: PictureParams = unsafe { std::mem::zeroed() };
        pp.frame_width = cur.width as u16;
        pp.frame_height = cur.height as u16;
        for (i, r) in a.refs.iter().enumerate() {
            pp.reference_frames[i] = r.as_ref().and_then(|f| ses.surface_of(key(f))).unwrap_or(INVALID_ID);
        }
        let fm = h.filtermode as u32;
        let seg = &h.segmentation;
        pp.pic_fields = cur.ss_h as u32
            | (cur.ss_v as u32) << 1
            | (!h.keyframe as u32) << 2
            | (!h.invisible as u32) << 3
            | (h.errorres as u32) << 4
            | (h.intraonly as u32) << 5
            | ((!h.keyframe && h.highprecisionmvs) as u32) << 6
            // VA numbers the filters regular, smooth, sharp, bilinear.
            | (fm ^ (fm <= 1) as u32) << 7
            | (h.parallelmode as u32) << 10
            | (h.resetctx as u32) << 11
            | (h.refreshctx as u32) << 13
            | (h.framectxid as u32) << 14
            | (seg.enabled as u32) << 16
            | (seg.temporal as u32) << 17
            | (seg.update_map as u32) << 18
            | (h.refidx[0] as u32) << 19
            | (h.signbias[0] as u32) << 22
            | (h.refidx[1] as u32) << 23
            | (h.signbias[1] as u32) << 26
            | (h.refidx[2] as u32) << 27
            | (h.signbias[2] as u32) << 30
            | (h.lossless as u32) << 31;
        pp.filter_level = h.filter_level;
        pp.sharpness_level = h.sharpness as u8;
        pp.log2_tile_rows = h.log2_tile_rows as u8;
        pp.log2_tile_columns = h.log2_tile_cols as u8;
        pp.frame_header_length_in_bytes = a.uncompressed_size as u8;
        pp.first_partition_size = a.compressed_size as u16;
        pp.mb_segment_tree_probs = seg.prob;
        pp.segment_pred_probs = if seg.temporal { seg.pred_prob } else { [255; 3] };
        pp.profile = h.profile;
        pp.bit_depth = h.bpp;

        let mut sp: SliceParams = unsafe { std::mem::zeroed() };
        sp.slice_data_size = a.data.len() as u32;
        for (i, f) in seg.feat.iter().enumerate() {
            sp.seg_param[i] = SegmentParams {
                segment_flags: f.ref_enabled as u16 | (f.ref_val as u16) << 1 | (f.skip_enabled as u16) << 3,
                filter_level: f.lflvl,
                luma_ac_quant_scale: f.qmul[0][1],
                luma_dc_quant_scale: f.qmul[0][0],
                chroma_ac_quant_scale: f.qmul[1][1],
                chroma_dc_quant_scale: f.qmul[1][0],
                reserved: [0; 4],
            };
        }

        let dev = &self.dev;
        ses.begin(dev, target)?;
        let mut ids = [
            ses.buffer(dev, BUF_PICTURE_PARAMETER, &pp)?,
            ses.buffer(dev, BUF_SLICE_PARAMETER, &sp)?,
            ses.raw_buffer(dev, BUF_SLICE_DATA, a.data.len(), a.data.as_ptr().cast())?,
        ];
        let r = ses.render(dev, &mut ids);
        let e = ses.end(dev);
        r.and(e)?;
        if !h.invisible {
            self.pinned.push(cur.clone());
        }
        Ok(())
    }

    fn read(&mut self, f: &Frame) -> Result<DecodedFrame, String> {
        self.pinned.retain(|p| key(p) != key(f));
        let ses = self.session.as_ref().ok_or("no VA context")?;
        let surface = ses.surface_of(key(f)).ok_or("no VA surface holds that frame")?;
        // VP9's colour_space, as the matrix number the native decoder reports.
        const SPC: [u32; 8] = [2, 5, 1, 6, 7, 9, 2, 0];
        let color = ColorInfo::from_vui(Some(SPC[f.color_space as usize & 7]), f.full_range, f.height);
        self.dev.read_frame(surface, f.width as usize, f.height as usize, f.bpp > 8, [0; 4], color)
    }
}

impl Drop for VaVp9 {
    fn drop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
    }
}
