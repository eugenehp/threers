//! Frame set-up and the tile/superblock-row loop (dav1d `dav1d_submit_frame`,
//! `dav1d_decode_frame_init`, `setup_tile`, `dav1d_decode_tile_sbrow`,
//! `dav1d_decode_frame_main`).

use std::sync::Arc;

use super::cdf::CdfContext;
use super::decode::*;
use super::env::BlockContext;
use super::intra_edge;
use super::lf_mask::{self, Av1Filter, Av1FilterLut, Av1Restoration};
use super::msac::Msac;
use super::obu::*;
use super::picture::Picture;
use super::refmvs::{RefmvsFrame, RefmvsTile, TemporalBlock};
use super::tables::SGR_PARAMS;
use super::warpmv;

/// What a reference slot keeps (dav1d `c->refs[i]` + `c->cdf[i]`).
#[derive(Clone)]
pub struct RefSlot {
    pub pic: Arc<Picture>,
    pub hdr: Arc<FrameHeader>,
    pub seq: Arc<SeqHeader>,
    pub cdf: Arc<CdfContext>,
    pub segmap: Option<Arc<Vec<u8>>>,
    pub refmvs: Option<Arc<Vec<TemporalBlock>>>,
    pub refpoc: [u32; 7],
    pub showable: bool,
}

pub struct TileData {
    pub data: Vec<u8>,
    pub start: u32,
    pub end: u32,
}

fn scale_fac(r: i32, t: i32) -> i32 {
    ((r << 14) + (t >> 1)) / t
}

fn get_upscale_x0(in_w: i32, out_w: i32, step: i32) -> i32 {
    let err = out_w * step - (in_w << 14);
    let x0 = (-((out_w - in_w) << 13) + (out_w >> 1)) / out_w + 128 - (err / 2);
    x0 & 0x3fff
}


impl Fd {
    /// Allocate and initialise everything for one frame.
    pub fn new(seq: Arc<SeqHeader>, hdr: Arc<FrameHeader>, slots: &[Option<RefSlot>; 8]) -> Result<(Fd, Arc<CdfContext>), String> {
        let bpc = 8 + 2 * seq.hbd as u32;
        let layout = seq.layout;
        let ss_hor = (layout == Layout::I420 || layout == Layout::I422) as i32;
        let ss_ver = (layout == Layout::I420) as i32;
        let mut refp: [Option<RefFrame>; 7] = Default::default();
        let mut svc = [[Svc::default(); 2]; 7];
        let mut gmv_warp_allowed = [false; 7];
        let mut ref_coded_width = [0i32; 7];
        if hdr.is_inter_or_switch() {
            if hdr.primary_ref_frame != PRIMARY_REF_NONE && slots[hdr.refidx[hdr.primary_ref_frame as usize] as usize].is_none() {
                return Err("missing primary reference".into());
            }
            for i in 0..7 {
                let s = slots[hdr.refidx[i] as usize].as_ref().ok_or("missing reference")?;
                let (rw, rh) = (s.pic.w, s.pic.h);
                if hdr.width[0] * 2 < rw || hdr.height * 2 < rh || hdr.width[0] > rw * 16 || hdr.height > rh * 16 || s.pic.layout != layout || s.pic.bpc != bpc {
                    return Err("invalid reference".into());
                }
                ref_coded_width[i] = s.hdr.width[0];
                if hdr.width[0] != rw || hdr.height != rh {
                    svc[i][0].scale = scale_fac(rw, hdr.width[0]);
                    svc[i][1].scale = scale_fac(rh, hdr.height);
                    svc[i][0].step = (svc[i][0].scale + 8) >> 4;
                    svc[i][1].step = (svc[i][1].scale + 8) >> 4;
                }
                let mut g = hdr.gmv[i];
                gmv_warp_allowed[i] = hdr.gmv[i].typ > WM_TRANSLATION && !hdr.force_integer_mv && !warpmv::get_shear_params(&mut g) && svc[i][0].scale == 0;
                refp[i] = Some(RefFrame { pic: s.pic.clone(), hdr: s.hdr.clone() });
            }
        }
        // The shear parameters of the global motion models are used by MC.
        let mut hdr_m = (*hdr).clone();
        for g in hdr_m.gmv.iter_mut() {
            let _ = warpmv::get_shear_params(g);
        }
        let hdr = Arc::new(hdr_m);

        let in_cdf = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            Arc::new(*CdfContext::default_for(hdr.quant.yac))
        } else {
            slots[hdr.refidx[hdr.primary_ref_frame as usize] as usize].as_ref().unwrap().cdf.clone()
        };

        let cur = Picture::new(hdr.width[0], hdr.height, layout, bpc);
        let sr_cur = if hdr.width[0] != hdr.width[1] { Some(Picture::new(hdr.width[1], hdr.height, layout, bpc)) } else { None };
        let mut resize_step = [0; 2];
        let mut resize_start = [0; 2];
        if let Some(sr) = &sr_cur {
            resize_step[0] = scale_fac(cur.w, sr.w);
            let in_cw = (cur.w + ss_hor) >> ss_hor;
            let out_cw = (sr.w + ss_hor) >> ss_hor;
            resize_step[1] = scale_fac(in_cw, out_cw);
            resize_start[0] = get_upscale_x0(cur.w, sr.w, resize_step[0]);
            resize_start[1] = get_upscale_x0(in_cw, out_cw, resize_step[1]);
        }

        let w4 = (hdr.width[0] + 3) >> 2;
        let h4 = (hdr.height + 3) >> 2;
        let bw = ((hdr.width[0] + 7) >> 3) << 1;
        let bh = ((hdr.height + 7) >> 3) << 1;
        let sb128w = (bw + 31) >> 5;
        let sb128h = (bh + 31) >> 5;
        let sb_shift = 4 + seq.sb128 as i32;
        let sb_step = 16 << seq.sb128 as i32;
        let sbh = (bh + sb_step - 1) >> sb_shift;
        let b4_stride = (bw + 31) & !31;
        let sr_w = sr_cur.as_ref().map(|p| p.w).unwrap_or(cur.w);
        let sr_sb128w = (sr_w + 127) >> 7;

        // Reference MVs.
        let mut rf = None;
        let mut refpoc = [0u32; 7];
        if hdr.is_inter_or_switch() || hdr.allow_intrabc {
            if !hdr.allow_intrabc {
                for i in 0..7 {
                    refpoc[i] = refp[i].as_ref().unwrap().hdr.frame_offset as u32;
                }
            }
            let mut rp_ref: [Option<Arc<Vec<TemporalBlock>>>; 7] = Default::default();
            let mut refrefpoc = [[0u32; 7]; 7];
            if hdr.use_ref_frame_mvs {
                for i in 0..7 {
                    let s = slots[hdr.refidx[i] as usize].as_ref().unwrap();
                    let ref_w = ((ref_coded_width[i] + 7) >> 3) << 1;
                    let ref_h = ((s.pic.h + 7) >> 3) << 1;
                    if s.refmvs.is_some() && ref_w == bw && ref_h == bh {
                        rp_ref[i] = s.refmvs.clone();
                    }
                    refrefpoc[i] = s.refpoc;
                }
            }
            rf = Some(RefmvsFrame::new(&seq, &hdr, &refpoc, &refrefpoc, rp_ref));
        }

        // Segmentation maps.
        let mut prev_segmap = None;
        let mut cur_segmap = None;
        let segmap_size = b4_stride as usize * 32 * sb128h as usize;
        if hdr.segmentation.enabled {
            if hdr.segmentation.temporal || !hdr.segmentation.update_map {
                let pri = hdr.primary_ref_frame as usize;
                let ref_w = ((ref_coded_width[pri] + 7) >> 3) << 1;
                let ref_h = ((refp[pri].as_ref().unwrap().pic.h + 7) >> 3) << 1;
                if ref_w == bw && ref_h == bh {
                    prev_segmap = slots[hdr.refidx[pri] as usize].as_ref().unwrap().segmap.clone();
                }
            }
            cur_segmap = Some(if hdr.segmentation.update_map {
                vec![0u8; segmap_size]
            } else if let Some(p) = &prev_segmap {
                p.as_ref().clone()
            } else {
                vec![0u8; segmap_size]
            });
        }

        // Loop filter state.
        let num_sb128 = (sb128w * sb128h) as usize;
        let mut lim_lut = Av1FilterLut::default();
        lf_mask::calc_eih(&mut lim_lut, hdr.loopfilter.sharpness as i32);
        let mut lvl = [[[[0u8; 2]; 8]; 4]; 8];
        lf_mask::calc_lf_values(&mut lvl, &hdr, [0; 4]);
        let restore_planes = (hdr.restoration.typ[0] != RESTORATION_NONE) as u32
            | (((hdr.restoration.typ[1] != RESTORATION_NONE) as u32) << 1)
            | (((hdr.restoration.typ[2] != RESTORATION_NONE) as u32) << 2);
        let mut start_of_tile_row = vec![0u8; sbh as usize];
        let mut sby = 0usize;
        for tile_row in 0..hdr.tiling.rows as usize {
            start_of_tile_row[sby] = tile_row as u8;
            sby += 1;
            while sby < hdr.tiling.row_start_sb[tile_row + 1] as usize {
                start_of_tile_row[sby] = 0;
                sby += 1;
            }
        }
        let re_sz = (sb128h * hdr.tiling.cols as i32) as usize;
        let lf = LfState {
            level: vec![[0u8; 4]; num_sb128 * 32 * 32 + 3],
            mask: vec![Av1Filter::default(); num_sb128],
            lr_mask: vec![Av1Restoration::default(); (sr_sb128w * sb128h) as usize],
            lim_lut,
            lvl,
            tx_lpf_right_edge: [vec![0u8; re_sz * 32], vec![0u8; re_sz * 32]],
            start_of_tile_row,
            restore_planes,
        };

        let mut dq = [[[0u16; 2]; 3]; 8];
        init_quant_tables(&seq, &hdr, hdr.quant.yac as i32, &mut dq);
        let mut qm = [[None; 3]; 19];
        if hdr.quant.qm {
            for row in qm.iter_mut() {
                *row = [
                    if hdr.quant.qm_y < 15 { Some((hdr.quant.qm_y as usize, 0)) } else { None },
                    if hdr.quant.qm_u < 15 { Some((hdr.quant.qm_u as usize, 1)) } else { None },
                    if hdr.quant.qm_v < 15 { Some((hdr.quant.qm_v as usize, 1)) } else { None },
                ];
            }
        }

        let mut jnt_weights = [[0u8; 7]; 7];
        if hdr.switchable_comp_refs {
            const QDW: [[i32; 2]; 3] = [[2, 3], [2, 5], [2, 7]];
            const QDL: [[u8; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];
            let nb = seq.order_hint_n_bits;
            for i in 0..7 {
                let r0 = refp[i].as_ref().unwrap().hdr.frame_offset as i32;
                for j in i + 1..7 {
                    let r1 = refp[j].as_ref().unwrap().hdr.frame_offset as i32;
                    let d1 = get_poc_diff(nb, r0, hdr.frame_offset as i32).abs().min(31);
                    let d0 = get_poc_diff(nb, r1, hdr.frame_offset as i32).abs().min(31);
                    let order = (d0 <= d1) as usize;
                    let mut k = 0;
                    while k < 3 {
                        let c0 = QDW[k][order];
                        let c1 = QDW[k][1 - order];
                        let d0c0 = d0 * c0;
                        let d1c1 = d1 * c1;
                        if (d0 > d1 && d0c0 < d1c1) || (d0 <= d1 && d0c0 > d1c1) {
                            break;
                        }
                        k += 1;
                    }
                    jnt_weights[i][j] = QDL[k][order];
                }
            }
        }

        let n_ts = hdr.tiling.cols as usize * hdr.tiling.rows as usize;
        let ipred_sz = (sbh * sb128w) as usize * 128;
        let fd = Fd {
            seq: seq.clone(),
            hdr: hdr.clone(),
            layout,
            ss_hor,
            ss_ver,
            bpc,
            bitdepth_max: (1 << bpc) - 1,
            cur,
            sr_cur,
            refp,
            svc,
            gmv_warp_allowed,
            w4,
            h4,
            bw,
            bh,
            sb128w,
            sb_shift,
            sb_step,
            sbh,
            sr_sb128w,
            b4_stride,
            resize_step,
            resize_start,
            dq,
            qm,
            a: vec![BlockContext::default(); (sb128w * hdr.tiling.rows as i32) as usize],
            ai: 0,
            rf,
            jnt_weights,
            lf,
            ipred_edge: [vec![0; ipred_sz], vec![0; ipred_sz], vec![0; ipred_sz]],
            cur_segmap,
            prev_segmap,
            ts: Vec::with_capacity(n_ts),
            tsi: 0,
            bx: 0,
            by: 0,
            l: BlockContext::default(),
            rt: RefmvsTile { col_start: 0, col_end: 0, row_start: 0, row_end: 0 },
            lf_mask: 0,
            cdef_idx: 0,
            warpmv: DEFAULT_WM_PARAMS,
            tl_4x4_filter: 0,
            pal: [[0; 8]; 3],
            al_pal: [[[[0; 8]; 3]; 32]; 2],
            pal_sz_uv: [[0; 32]; 2],
            pal_idx_y: vec![0; 64 * 64],
            pal_idx_uv: vec![0; 64 * 64],
            txtp_map: [0; 32 * 32],
            cf: vec![0; 32 * 32],
            levels: vec![0; 32 * 34],
            refpoc,
            mc_buf: Vec::new(),
        };
        Ok((fd, in_cdf))
    }

    fn setup_tile(&mut self, in_cdf: &CdfContext, data: &[u8], tile_row: usize, tile_col: usize) {
        let hdr = self.hdr.clone();
        let col_sb_start = hdr.tiling.col_start_sb[tile_col] as i32;
        let col_sb128_start = col_sb_start >> (!self.seq.sb128) as i32;
        let col_sb_end = hdr.tiling.col_start_sb[tile_col + 1] as i32;
        let row_sb_start = hdr.tiling.row_start_sb[tile_row] as i32;
        let row_sb_end = hdr.tiling.row_start_sb[tile_row + 1] as i32;
        let sb_shift = self.sb_shift;
        let tiling = TilePos {
            row: tile_row,
            col: tile_col,
            col_start: col_sb_start << sb_shift,
            col_end: (col_sb_end << sb_shift).min(self.bw),
            row_start: row_sb_start << sb_shift,
            row_end: (row_sb_end << sb_shift).min(self.bh),
        };
        let mut lr_ref = [None; 3];
        let (sb_idx, unit_idx);
        if hdr.width[0] != hdr.width[1] {
            sb_idx = (tiling.row_start >> 5) * self.sr_sb128w;
            unit_idx = (tiling.row_start & 16) >> 3;
        } else {
            sb_idx = (tiling.row_start >> 5) * self.sb128w + col_sb128_start;
            unit_idx = ((tiling.row_start & 16) >> 3) + ((tiling.col_start & 16) >> 4);
        }
        for (p, lr) in lr_ref.iter_mut().enumerate() {
            if (self.lf.restore_planes >> p) & 1 == 0 {
                continue;
            }
            let (si, ui);
            if hdr.width[0] != hdr.width[1] {
                let ss_hor = (p != 0 && self.layout != Layout::I444) as i32;
                let d = hdr.super_res_width_scale_denominator as i32;
                let unit_size_log2 = hdr.restoration.unit_size[(p != 0) as usize] as i32;
                let rnd = (8 << unit_size_log2) - 1;
                let shift = unit_size_log2 + 3;
                let x = ((4 * tiling.col_start * d >> ss_hor) + rnd) >> shift;
                let px_x = x << (unit_size_log2 + ss_hor);
                let u_idx = unit_idx + ((px_x & 64) >> 6);
                let sb128x = px_x >> 7;
                if sb128x >= self.sr_sb128w {
                    continue;
                }
                si = (sb_idx + sb128x) as usize;
                ui = u_idx as usize;
            } else {
                si = sb_idx as usize;
                ui = unit_idx as usize;
            }
            let u = &mut self.lf.lr_mask[si].lr[p][ui];
            u.filter_v = [3, -7, 15];
            u.filter_h = [3, -7, 15];
            u.sgr_weights = [-32, 31];
            *lr = Some((si, ui));
        }
        self.ts.push(TileState {
            cdf: Box::new(*in_cdf),
            msac: Msac::new(data, hdr.disable_cdf_update),
            tiling,
            dqmem: [[[0; 2]; 3]; 8],
            dq_frame: true,
            last_qidx: hdr.quant.yac as i32,
            last_delta_lf: [0; 4],
            lflvlmem: [[[[0; 2]; 8]; 4]; 8],
            lflvl_frame: true,
            lr_ref,
        });
    }

    fn read_restoration_info(&mut self, si: usize, ui: usize, p: usize, frame_type: u8) {
        let mut lr = self.lf.lr_mask[si].lr[p][ui];
        let (rsi, rui) = self.ts[self.tsi].lr_ref[p].unwrap();
        let rr = self.lf.lr_mask[rsi].lr[p][rui];
        if frame_type == RESTORATION_SWITCHABLE {
            let filter = sa!(self, 2, m.restore_switchable) as u8;
            lr.typ = filter + (filter != 0) as u8;
        } else {
            let t = if frame_type == RESTORATION_WIENER { ba!(self, m.restore_wiener) } else { ba!(self, m.restore_sgrproj) };
            lr.typ = if t { frame_type } else { RESTORATION_NONE };
        }
        if lr.typ == RESTORATION_WIENER {
            let ms = self.msac();
            lr.filter_v[0] = if p != 0 { 0 } else { (ms.subexp(rr.filter_v[0] as i32 + 5, 16, 1) - 5) as i8 };
            lr.filter_v[1] = (ms.subexp(rr.filter_v[1] as i32 + 23, 32, 2) - 23) as i8;
            lr.filter_v[2] = (ms.subexp(rr.filter_v[2] as i32 + 17, 64, 3) - 17) as i8;
            lr.filter_h[0] = if p != 0 { 0 } else { (ms.subexp(rr.filter_h[0] as i32 + 5, 16, 1) - 5) as i8 };
            lr.filter_h[1] = (ms.subexp(rr.filter_h[1] as i32 + 23, 32, 2) - 23) as i8;
            lr.filter_h[2] = (ms.subexp(rr.filter_h[2] as i32 + 17, 64, 3) - 17) as i8;
            lr.sgr_weights = rr.sgr_weights;
            self.ts[self.tsi].lr_ref[p] = Some((si, ui));
        } else if lr.typ == RESTORATION_SGRPROJ {
            let ms = self.msac();
            let idx = ms.bools(4) as usize;
            let sgr = SGR_PARAMS[idx];
            lr.typ += idx as u8;
            lr.sgr_weights[0] = if sgr[0] != 0 { (ms.subexp(rr.sgr_weights[0] as i32 + 96, 128, 4) - 96) as i8 } else { 0 };
            lr.sgr_weights[1] = if sgr[1] != 0 { (ms.subexp(rr.sgr_weights[1] as i32 + 32, 128, 4) - 32) as i8 } else { 95 };
            lr.filter_v = rr.filter_v;
            lr.filter_h = rr.filter_h;
            self.ts[self.tsi].lr_ref[p] = Some((si, ui));
        }
        self.lf.lr_mask[si].lr[p][ui] = lr;
    }

    fn decode_tile_sbrow(&mut self) -> DResult {
        let hdr = self.hdr.clone();
        let seq = self.seq.clone();
        let root_bl = if seq.sb128 { 0 } else { 1 };
        let sb_step = self.sb_step;
        let (tile_row, tile_col) = (self.ts[self.tsi].tiling.row, self.ts[self.tsi].tiling.col);
        let col_sb_start = hdr.tiling.col_start_sb[tile_col] as i32;
        let col_sb128_start = col_sb_start >> (!seq.sb128) as i32;
        let t = &self.ts[self.tsi].tiling;
        if hdr.is_inter_or_switch() || hdr.allow_intrabc {
            let rf = self.rf.as_ref().unwrap();
            self.rt = RefmvsTile { col_start: t.col_start, col_end: t.col_end.min(rf.iw4), row_start: t.row_start, row_end: t.row_end.min(rf.ih4) };
        }
        reset_context(&mut self.l, hdr.is_key_or_intra());
        self.pal_sz_uv[1] = [0; 32];
        let sb128y = self.by >> 5;
        let (cs, ce) = (t.col_start, t.col_end);
        self.ai = (col_sb128_start + tile_row as i32 * self.sb128w) as usize;
        self.lf_mask = (sb128y * self.sb128w + col_sb128_start) as usize;
        let tree = intra_edge::tree();
        self.bx = cs;
        while self.bx < ce {
            if root_bl == 0 {
                self.cdef_idx = 0;
                self.lf.mask[self.lf_mask].cdef_idx = [-1; 4];
            } else {
                self.cdef_idx = (((self.bx & 16) >> 4) + ((self.by & 16) >> 3)) as usize;
                self.lf.mask[self.lf_mask].cdef_idx[self.cdef_idx] = -1;
            }
            // Restoration units.
            for p in 0..3 {
                if (self.lf.restore_planes >> p) & 1 == 0 {
                    continue;
                }
                let ss_ver = (p != 0 && self.layout == Layout::I420) as i32;
                let ss_hor = (p != 0 && self.layout != Layout::I444) as i32;
                let unit_size_log2 = hdr.restoration.unit_size[(p != 0) as usize] as i32;
                let y = self.by * 4 >> ss_ver;
                let h = (self.cur.h + ss_ver) >> ss_ver;
                let unit_size = 1 << unit_size_log2;
                let mask = unit_size - 1;
                if y & mask != 0 {
                    continue;
                }
                let half_unit = unit_size >> 1;
                if y != 0 && y + half_unit > h {
                    continue;
                }
                let frame_type = hdr.restoration.typ[p];
                if hdr.width[0] != hdr.width[1] {
                    let w = (self.sr_cur.as_ref().unwrap().w + ss_hor) >> ss_hor;
                    let n_units = 1.max((w + half_unit) >> unit_size_log2);
                    let d = hdr.super_res_width_scale_denominator as i32;
                    let rnd = unit_size * 8 - 1;
                    let shift = unit_size_log2 + 3;
                    let x0 = ((4 * self.bx * d >> ss_hor) + rnd) >> shift;
                    let x1 = ((4 * (self.bx + sb_step) * d >> ss_hor) + rnd) >> shift;
                    for x in x0..x1.min(n_units) {
                        let px_x = x << (unit_size_log2 + ss_hor);
                        let sb_idx = (self.by >> 5) * self.sr_sb128w + (px_x >> 7);
                        let unit_idx = ((self.by & 16) >> 3) + ((px_x & 64) >> 6);
                        self.read_restoration_info(sb_idx as usize, unit_idx as usize, p, frame_type);
                    }
                } else {
                    let x = 4 * self.bx >> ss_hor;
                    if x & mask != 0 {
                        continue;
                    }
                    let w = (self.cur.w + ss_hor) >> ss_hor;
                    if x != 0 && x + half_unit > w {
                        continue;
                    }
                    let sb_idx = (self.by >> 5) * self.sr_sb128w + (self.bx >> 5);
                    let unit_idx = ((self.by & 16) >> 3) + ((self.bx & 16) >> 4);
                    self.read_restoration_info(sb_idx as usize, unit_idx as usize, p, frame_type);
                }
            }
            self.decode_sb(root_bl, tree.root[root_bl])?;
            if self.bx & 16 != 0 || seq.sb128 {
                self.ai += 1;
                self.lf_mask += 1;
            }
            self.bx += sb_step;
        }

        self.backup_ipred_edge();

        // Back up the right-edge tx_lpf contexts for the loop filter.
        let align_h = ((self.bh + 31) & !31) as usize;
        let by = self.by as usize;
        let o0 = align_h * tile_col + by;
        let src0 = (self.by & 16) as usize;
        let n = sb_step as usize;
        self.lf.tx_lpf_right_edge[0][o0..o0 + n].copy_from_slice(&self.l.tx_lpf_y[src0..src0 + n]);
        let ss_ver = self.ss_ver as usize;
        let align_h = align_h >> ss_ver;
        let o1 = align_h * tile_col + (by >> ss_ver);
        let src1 = ((self.by & 16) >> ss_ver) as usize;
        let n1 = n >> ss_ver;
        self.lf.tx_lpf_right_edge[1][o1..o1 + n1].copy_from_slice(&self.l.tx_lpf_uv[src1..src1 + n1]);

        if self.ts[self.tsi].msac.overread() {
            return Err("symbol decoder overread".into());
        }
        Ok(())
    }

    fn backup_ipred_edge(&mut self) {
        let ts = &self.ts[self.tsi];
        let sby = self.by >> self.sb_shift;
        let sby_off = (self.sb128w * 128 * sby) as usize;
        let x_off = ts.tiling.col_start as usize;
        let n = 4 * (ts.tiling.col_end as usize - x_off);
        let ys = self.cur.stride[0];
        let row = ((self.by + self.sb_step) * 4 - 1) as usize;
        let src = row * ys + x_off * 4;
        self.ipred_edge[0][sby_off + x_off * 4..sby_off + x_off * 4 + n].copy_from_slice(&self.cur.planes[0][src..src + n]);
        if self.layout != Layout::I400 {
            let (ss_hor, ss_ver) = (self.ss_hor as usize, self.ss_ver as usize);
            let uvs = self.cur.stride[1];
            let crow = (((self.by + self.sb_step) * 4) as usize >> ss_ver) - 1;
            let uv_off = (x_off * 4 >> ss_hor) + crow * uvs;
            let nc = n >> ss_hor;
            for pl in 1..=2 {
                let d = sby_off + (x_off * 4 >> ss_hor);
                self.ipred_edge[pl][d..d + nc].copy_from_slice(&self.cur.planes[pl][uv_off..uv_off + nc]);
            }
        }
    }

    /// Decode the frame's tiles with in-loop filtering (`dav1d_decode_frame`).
    pub fn decode_tiles(&mut self, in_cdf: &CdfContext, tiles: &[TileData]) -> Result<Option<CdfContext>, String> {
        let hdr = self.hdr.clone();
        let (mut tile_row, mut tile_col) = (0usize, 0usize);
        let mut update_set = false;
        for tg in tiles {
            let mut data: &[u8] = &tg.data;
            for j in tg.start..=tg.end {
                let tile_sz;
                if j == tg.end {
                    tile_sz = data.len();
                } else {
                    let nb = hdr.tiling.n_bytes as usize;
                    if nb > data.len() {
                        return Err("tile size overrun".into());
                    }
                    let mut s = 0usize;
                    for k in 0..nb {
                        s |= (data[k] as usize) << (k * 8);
                    }
                    data = &data[nb..];
                    tile_sz = s + 1;
                    if tile_sz > data.len() {
                        return Err("tile size overrun".into());
                    }
                }
                self.setup_tile(in_cdf, &data[..tile_sz], tile_row, tile_col);
                tile_col += 1;
                if tile_col == hdr.tiling.cols as usize {
                    tile_col = 0;
                    tile_row += 1;
                }
                if j == hdr.tiling.update as u32 && hdr.refresh_context {
                    update_set = true;
                }
                data = &data[tile_sz..];
            }
        }

        for a in self.a.iter_mut() {
            reset_context(a, hdr.is_key_or_intra());
        }
        for tile_row in 0..hdr.tiling.rows as usize {
            let sbh_end = (hdr.tiling.row_start_sb[tile_row + 1] as i32).min(self.sbh);
            for sby in hdr.tiling.row_start_sb[tile_row] as i32..sbh_end {
                self.by = sby << (4 + self.seq.sb128 as i32);
                let by_end = (self.by + self.sb_step) >> 1;
                if hdr.use_ref_frame_mvs {
                    let (bw, by) = (self.bw, self.by);
                    self.rf.as_mut().unwrap().load_tmvs(0, bw >> 1, by >> 1, by_end);
                }
                for tile_col in 0..hdr.tiling.cols as usize {
                    self.tsi = tile_row * hdr.tiling.cols as usize + tile_col;
                    self.decode_tile_sbrow()?;
                }
                if hdr.is_inter_or_switch() {
                    let (bw, by) = (self.bw, self.by);
                    self.rf.as_mut().unwrap().save_tmvs(0, bw >> 1, by >> 1, by_end);
                }
            }
        }
        self.apply_filters();
        if hdr.refresh_context && update_set {
            let mut out = *in_cdf;
            out.update_from(&self.ts[hdr.tiling.update as usize].cdf, hdr.is_key_or_intra());
            return Ok(Some(out));
        }
        Ok(None)
    }
}
