//! AV1 on VAAPI: the native decoder's sequence and frame headers translated
//! into `VADecPictureParameterBufferAV1`, one `VASliceParameterBufferAV1`
//! per tile and the tile groups' data (libva's `va_dec_av1.h`, written as a
//! zeroed block with fields at the offsets the C compiler reported,
//! [`super::av1_layout`]). Film grain is applied by the native decoder to
//! what is read back.

use std::sync::Arc;

use super::av1_layout::*;
use super::{Device, Session, BUF_PICTURE_PARAMETER, BUF_SLICE_DATA, BUF_SLICE_PARAMETER, INVALID_ID, RT_FORMAT_YUV420, RT_FORMAT_YUV420_10};
use crate::decode::av1::obu::Layout;
use crate::decode::av1::{AccelFrame, Av1Accel, Picture};

/// A zeroed C struct of a known size with fields written at known offsets.
struct Blob(Vec<u64>);

impl Blob {
    fn new(size: usize) -> Blob {
        Blob(vec![0; size.div_ceil(8)])
    }
    fn bytes(&mut self) -> &mut [u8] {
        let n = self.0.len() * 8;
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), n) }
    }
    fn u8(&mut self, at: usize, v: u8) {
        self.bytes()[at] = v;
    }
    fn u16(&mut self, at: usize, v: u16) {
        self.bytes()[at..at + 2].copy_from_slice(&v.to_ne_bytes());
    }
    fn u32(&mut self, at: usize, v: u32) {
        self.bytes()[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    }
    fn ptr(&self) -> *const std::ffi::c_void {
        self.0.as_ptr().cast()
    }
}

fn key(p: &Picture) -> i64 {
    p as *const Picture as usize as i64
}

/// The [`Av1Accel`] that drives a VA context.
pub(crate) struct VaAv1 {
    dev: Device,
    session: Option<Session>,
    /// Shown pictures not yet read back: their surfaces must not be reused.
    pinned: Vec<Arc<Picture>>,
}

impl VaAv1 {
    pub(crate) fn new(dev: Device) -> Self {
        VaAv1 { dev, session: None, pinned: Vec::new() }
    }

    fn session(&mut self, p: &Picture, max: (i32, i32)) -> Result<(), String> {
        if p.layout != Layout::I420 || !(p.bpc == 8 || p.bpc == 10) {
            return Err("the VAAPI AV1 path decodes 8- and 10-bit 4:2:0 only".into());
        }
        let rt = if p.bpc > 8 { RT_FORMAT_YUV420_10 } else { RT_FORMAT_YUV420 };
        // Surfaces at the sequence's maximum size, so frames of any size up to
        // it (and references of other sizes) share one context.
        let key = (max.0 as u32, max.1 as u32, PROFILE_AV1_0, rt);
        if self.session.as_ref().is_some_and(|s| s.key == key) {
            return Ok(());
        }
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
        self.session = Some(Session::new(&self.dev, PROFILE_AV1_0, rt, key.0, key.1, 12)?);
        Ok(())
    }
}

/// `FrameRestorationType` from the coded `lr_type` (none, switchable,
/// Wiener, self-guided).
const LR_TYPE: [u32; 4] = [0, 3, 1, 2];

impl Av1Accel for VaAv1 {
    fn frame(&mut self, a: &AccelFrame<'_>) -> Result<(), String> {
        let (seq, h, cur) = (a.seq, a.hdr, a.cur);
        self.session(cur, (seq.max_width, seq.max_height))?;
        let live: Vec<i64> = a.refs.iter().flatten().map(|s| key(&s.pic)).chain(self.pinned.iter().map(|p| key(p))).collect();
        let ses = self.session.as_mut().unwrap();
        let target = ses.claim(key(cur), |k| live.contains(&k))?;

        let mut p = Blob::new(P_SIZE);
        p.u8(P_profile, seq.profile);
        p.u8(P_order_hint_bits_minus_1, seq.order_hint_n_bits.saturating_sub(1));
        p.u8(P_bit_depth_idx, seq.hbd);
        p.u8(P_matrix_coefficients, seq.mtrx);
        p.u32(
            P_seq_info_fields,
            (seq.still_picture as u32) << B_STILL
                | (seq.sb128 as u32) << B_SB128
                | (seq.filter_intra as u32) << B_FILTER_INTRA
                | (seq.intra_edge_filter as u32) << B_INTRA_EDGE
                | (seq.inter_intra as u32) << B_INTERINTRA
                | (seq.masked_compound as u32) << B_MASKED
                | (seq.dual_filter as u32) << B_DUAL_FILTER
                | (seq.order_hint as u32) << B_ORDER_HINT
                | (seq.jnt_comp as u32) << B_JNT
                | (seq.cdef as u32) << B_CDEF
                | (seq.monochrome as u32) << B_MONO
                | (seq.color_range as u32) << B_RANGE
                | (seq.ss_hor as u32) << B_SSX
                | (seq.ss_ver as u32) << B_SSY
                | (seq.film_grain_present as u32) << B_GRAIN_PRESENT,
        );
        p.u32(P_current_frame, target);
        // Film grain is the native decoder's: display the decoded picture.
        p.u32(P_current_display_picture, target);
        p.u16(P_frame_width_minus1, (h.width[1] - 1) as u16);
        p.u16(P_frame_height_minus1, (h.height - 1) as u16);
        for (i, r) in a.refs.iter().enumerate() {
            let s = r.as_ref().and_then(|r| ses.surface_of(key(&r.pic))).unwrap_or(INVALID_ID);
            p.u32(P_ref_frame_map + 4 * i, s);
        }
        for i in 0..7 {
            p.u8(P_ref_frame_idx + i, h.refidx[i].max(0) as u8);
        }
        p.u8(P_primary_ref_frame, h.primary_ref_frame);
        p.u8(P_order_hint, h.frame_offset);

        let seg = &h.segmentation;
        let s0 = P_seg_info;
        p.u32(
            s0 + SEG_FIELDS,
            (seg.enabled as u32) << B_SEG_ENABLED
                | (seg.update_map as u32) << B_SEG_UPDATE_MAP
                | (seg.temporal as u32) << B_SEG_TEMPORAL
                | (seg.update_data as u32) << B_SEG_UPDATE_DATA,
        );
        if seg.enabled {
            for (i, d) in seg.seg_data.d.iter().enumerate() {
                let values = [d.delta_q, d.delta_lf_y_v as i16, d.delta_lf_y_h as i16, d.delta_lf_u as i16, d.delta_lf_v as i16, d.r#ref.max(0) as i16, 0, 0];
                for (j, v) in values.into_iter().enumerate() {
                    p.u16(s0 + SEG_FEATURE_DATA + 2 * (8 * i + j), v as u16);
                }
                p.u8(s0 + SEG_FEATURE_MASK + i, d.enabled);
            }
        }

        let t = &h.tiling;
        p.u8(P_tile_cols, t.cols);
        p.u8(P_tile_rows, t.rows);
        for i in 0..t.cols as usize {
            p.u16(P_width_in_sbs_minus_1 + 2 * i, t.col_start_sb[i + 1] - t.col_start_sb[i] - 1);
        }
        for i in 0..t.rows as usize {
            p.u16(P_height_in_sbs_minus_1 + 2 * i, t.row_start_sb[i + 1] - t.row_start_sb[i] - 1);
        }
        p.u16(P_tile_count_minus_1, (t.cols as u16 * t.rows as u16).saturating_sub(1));
        p.u16(P_context_update_tile_id, t.update);
        p.u32(
            P_pic_info_fields,
            (h.frame_type as u32) << B_FRAME_TYPE
                | (h.show_frame as u32) << B_SHOW
                | (h.showable_frame as u32) << B_SHOWABLE
                | (h.error_resilient_mode as u32) << B_ERR_RES
                | (h.disable_cdf_update as u32) << B_NO_CDF_UPDATE
                | (h.allow_screen_content_tools as u32) << B_SCT
                | (h.force_integer_mv as u32) << B_INT_MV
                | (h.allow_intrabc as u32) << B_INTRABC
                | (h.super_res_enabled as u32) << B_SUPERRES
                | (h.hp as u32) << B_HP
                | (h.switchable_motion_mode as u32) << B_MOTION_SWITCH
                | (h.use_ref_frame_mvs as u32) << B_REF_MVS
                | (!h.refresh_context as u32) << B_NO_END_CDF
                | (t.uniform as u32) << B_UNIFORM_TILES
                | (h.warp_motion as u32) << B_WARP,
        );
        p.u8(P_superres_scale_denominator, h.super_res_width_scale_denominator);
        p.u8(P_interp_filter, h.subpel_filter_mode);
        let lf = &h.loopfilter;
        p.u8(P_filter_level, lf.level_y[0]);
        p.u8(P_filter_level + 1, lf.level_y[1]);
        p.u8(P_filter_level_u, lf.level_u);
        p.u8(P_filter_level_v, lf.level_v);
        p.u8(
            P_loop_filter_info_fields,
            (lf.sharpness << B_SHARP) | (lf.mode_ref_delta_enabled as u8) << B_MRD_ENABLED | (lf.mode_ref_delta_update as u8) << B_MRD_UPDATE,
        );
        for i in 0..8 {
            p.u8(P_ref_deltas + i, lf.mode_ref_deltas.ref_delta[i] as u8);
        }
        for i in 0..2 {
            p.u8(P_mode_deltas + i, lf.mode_ref_deltas.mode_delta[i] as u8);
        }
        let q = &h.quant;
        p.u8(P_base_qindex, q.yac);
        p.u8(P_y_dc_delta_q, q.ydc_delta as u8);
        p.u8(P_u_dc_delta_q, q.udc_delta as u8);
        p.u8(P_u_ac_delta_q, q.uac_delta as u8);
        p.u8(P_v_dc_delta_q, q.vdc_delta as u8);
        p.u8(P_v_ac_delta_q, q.vac_delta as u8);
        p.u16(
            P_qmatrix_fields,
            ((q.qm as u32) << B_QM | (q.qm_y as u32) << B_QM_Y | (q.qm_u as u32) << B_QM_U | (q.qm_v as u32) << B_QM_V) as u16,
        );
        let d = &h.delta;
        p.u32(
            P_mode_control_fields,
            (d.q_present as u32) << B_DQ
                | (d.q_res_log2 as u32) << B_DQ_RES
                | (d.lf_present as u32) << B_DLF
                | (d.lf_res_log2 as u32) << B_DLF_RES
                | (d.lf_multi as u32) << B_DLF_MULTI
                | (h.txfm_mode as u32) << B_TX_MODE
                | (h.switchable_comp_refs as u32) << B_REF_SELECT
                | (h.reduced_txtp_set as u32) << B_REDUCED_TX
                | (h.skip_mode_enabled as u32) << B_SKIP_MODE,
        );
        p.u8(P_cdef_damping_minus_3, h.cdef.damping.saturating_sub(3));
        p.u8(P_cdef_bits, h.cdef.n_bits);
        for i in 0..8 {
            p.u8(P_cdef_y_strengths + i, h.cdef.y_strength[i]);
            p.u8(P_cdef_uv_strengths + i, h.cdef.uv_strength[i]);
        }
        let r = &h.restoration;
        if r.typ.iter().any(|&t| t != 0) {
            let shift = r.unit_size[0].saturating_sub(6) as u32;
            let uv_shift = r.unit_size[0].saturating_sub(r.unit_size[1]) as u32;
            p.u16(
                P_loop_restoration_fields,
                (LR_TYPE[r.typ[0] as usize & 3] << B_LR_Y
                    | LR_TYPE[r.typ[1] as usize & 3] << B_LR_CB
                    | LR_TYPE[r.typ[2] as usize & 3] << B_LR_CR
                    | shift << B_LR_SHIFT
                    | uv_shift << B_LR_UV_SHIFT) as u16,
            );
        }
        for (i, g) in h.gmv.iter().enumerate() {
            let w0 = P_wm + i * WM_SIZE;
            p.u32(w0 + WM_TYPE, g.typ as u32);
            for (k, &m) in g.matrix.iter().enumerate() {
                p.u32(w0 + WM_MAT + 4 * k, m as u32);
            }
        }

        let dev = &self.dev;
        ses.begin(dev, target)?;
        let mut ids = vec![ses.raw_buffer(dev, BUF_PICTURE_PARAMETER, P_SIZE, p.ptr())?];
        // One parameter block per tile, then the tile group's data.
        let nb = t.n_bytes as usize;
        for tg in a.tiles {
            let n = (tg.end - tg.start + 1) as usize;
            let mut params = Blob::new(S_SIZE * n);
            let mut at = 0usize;
            for (k, j) in (tg.start..=tg.end).enumerate() {
                let size = if j == tg.end {
                    tg.data.len().checked_sub(at).ok_or("tile data overrun")?
                } else {
                    let raw = tg.data.get(at..at + nb).ok_or("tile size overrun")?;
                    at += nb;
                    raw.iter().enumerate().fold(0usize, |s, (i, &b)| s | (b as usize) << (8 * i)) + 1
                };
                let s = k * S_SIZE;
                params.u32(s + S_DATA_SIZE, size as u32);
                params.u32(s + S_DATA_OFFSET, at as u32);
                params.u16(s + S_TILE_ROW, (j / t.cols as u32) as u16);
                params.u16(s + S_TILE_COL, (j % t.cols as u32) as u16);
                params.u16(s + S_TG_START, tg.start as u16);
                params.u16(s + S_TG_END, tg.end as u16);
                at += size;
            }
            ids.push(ses.raw_buffer_n(dev, BUF_SLICE_PARAMETER, S_SIZE, n, params.ptr())?);
            ids.push(ses.raw_buffer(dev, BUF_SLICE_DATA, tg.data.len(), tg.data.as_ptr().cast())?);
        }
        let r = ses.render(dev, &mut ids);
        let e = ses.end(dev);
        r.and(e)?;
        if h.show_frame {
            self.pinned.push(cur.clone());
        }
        Ok(())
    }

    fn read(&mut self, pic: &Picture) -> Result<Picture, String> {
        self.pinned.retain(|p| key(p) != key(pic));
        let ses = self.session.as_ref().ok_or("no VA context")?;
        let surface = ses.surface_of(key(pic)).ok_or("no VA surface holds that picture")?;
        let color = crate::decode::ColorInfo::from_vui(None, false, pic.h as u32);
        let f = self.dev.read_frame(surface, pic.w as usize, pic.h as usize, pic.bpc > 8, [0; 4], color)?;
        // Into a picture laid out as the native decoder's, for film grain and
        // `Av1Frame::to_decoded`.
        let mut out = Picture::new(pic.w, pic.h, pic.layout, pic.bpc);
        for (pl, src) in [&f.y, &f.u, &f.v].into_iter().enumerate() {
            let (sx, sy) = out.ss(pl);
            let (w, h) = (((pic.w + sx) >> sx) as usize, ((pic.h + sy) >> sy) as usize);
            let stride = out.pstride(pl);
            for y in 0..h {
                out.planes[pl][y * stride..y * stride + w].copy_from_slice(&src[y * w..y * w + w]);
            }
        }
        Ok(out)
    }
}

impl Drop for VaAv1 {
    fn drop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.destroy(&self.dev);
        }
    }
}
