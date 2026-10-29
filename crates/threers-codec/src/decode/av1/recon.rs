//! Coefficients and reconstruction (dav1d `src/recon_tmpl.c`, pass 0).

use std::sync::Arc;

use super::decode::*;
use super::env::Mv;
use super::intra_edge::*;
use super::ipred::{self, Top, ANGLE_SMOOTH_EDGE_FLAG, EDGE_LEN, TL};
use super::itx;
use super::levels::*;
use super::mc::{self, Bd, Dst, Src};
use super::obu::*;
use super::picture::Picture;
use super::qm;
use super::tables::*;
use super::wedge;

fn read_golomb(ms: &mut super::msac::Msac) -> u32 {
    let mut len = 0;
    let mut val = 1u32;
    while !ms.bool_equi() && len < 32 {
        len += 1;
    }
    for _ in 0..len {
        val = (val << 1) + ms.bool_equi() as u32;
    }
    val.wrapping_sub(1)
}

fn get_skip_ctx(t_dim: &TxfmInfo, bs: usize, a: &[u8], l: &[u8], chroma: bool, layout: Layout) -> usize {
    let b_dim = BLOCK_DIMENSIONS[bs];
    if chroma {
        let ss_ver = (layout == Layout::I420) as i32;
        let ss_hor = (layout != Layout::I444) as i32;
        let not_one_blk = (b_dim[2] as i32 - ((b_dim[2] != 0) as i32 & ss_hor)) > t_dim.lw as i32
            || (b_dim[3] as i32 - ((b_dim[3] != 0) as i32 & ss_ver)) > t_dim.lh as i32;
        let ca = a[..1 << t_dim.lw].iter().any(|&v| v != 0x40) as usize;
        let cl = l[..1 << t_dim.lh].iter().any(|&v| v != 0x40) as usize;
        7 + not_one_blk as usize * 3 + ca + cl
    } else if b_dim[2] == t_dim.lw && b_dim[3] == t_dim.lh {
        0
    } else {
        let la = a[..1 << t_dim.lw].iter().fold(0u8, |s, &v| s | v);
        let ll = l[..1 << t_dim.lh].iter().fold(0u8, |s, &v| s | v);
        SKIP_CTX[(la & 0x3f).min(4) as usize][(ll & 0x3f).min(4) as usize] as usize
    }
}

fn get_dc_sign_ctx(t_dim: &TxfmInfo, a: &[u8], l: &[u8]) -> usize {
    let (w, h) = (t_dim.w as usize, t_dim.h as usize);
    let mut s = 0i32;
    for &v in &a[..w] {
        s += (v >> 6) as i32;
    }
    for &v in &l[..h] {
        s += (v >> 6) as i32;
    }
    s -= (w + h) as i32;
    (s != 0) as usize + (s > 0) as usize
}

fn get_lo_ctx(levels: &[u8], o: usize, tx_class: u8, hi_mag: &mut u32, ctx_offsets: Option<&[[u8; 5]; 5]>, x: u32, y: u32, stride: usize) -> u32 {
    let lv = |k: usize| levels[o + k] as u32;
    let mut mag = lv(1) + lv(stride);
    
    let offset = if tx_class == TX_CLASS_2D {
        mag += lv(stride + 1);
        *hi_mag = mag;
        mag += lv(2) + lv(2 * stride);
        ctx_offsets.unwrap()[y.min(4) as usize][x.min(4) as usize] as u32
    } else {
        mag += lv(2);
        *hi_mag = mag;
        mag += lv(3) + lv(4);
        26 + if y > 1 { 10 } else { y * 5 }
    };
    offset + if mag > 512 { 4 } else { (mag + 64) >> 7 }
}

pub enum Ctx {
    Luma,
    Chroma(usize),
}

impl Fd {
    fn ctx_bytes(&self, which: &Ctx, ax: usize, ly: usize) -> ([u8; 16], [u8; 16]) {
        let (a, l) = match which {
            Ctx::Luma => (&self.a[self.ai].lcoef, &self.l.lcoef),
            Ctx::Chroma(p) => (&self.a[self.ai].ccoef[*p], &self.l.ccoef[*p]),
        };
        let mut ab = [0u8; 16];
        let mut lb = [0u8; 16];
        let na = 16.min(32 - ax);
        let nl = 16.min(32 - ly);
        ab[..na].copy_from_slice(&a[ax..ax + na]);
        lb[..nl].copy_from_slice(&l[ly..ly + nl]);
        (ab, lb)
    }

    fn set_coef_ctx(&mut self, which: &Ctx, ax: usize, ly: usize, nw: usize, nh: usize, v: u8) {
        let ai = self.ai;
        let (a, l) = match which {
            Ctx::Luma => (&mut self.a[ai].lcoef, &mut self.l.lcoef),
            Ctx::Chroma(p) => (&mut self.a[ai].ccoef[*p], &mut self.l.ccoef[*p]),
        };
        for x in &mut a[ax..ax + nw] {
            *x = v;
        }
        for x in &mut l[ly..ly + nh] {
            *x = v;
        }
    }

    /// `decode_coefs`: returns (eob, txtp, res_ctx); coefficients in `self.cf`.
    #[allow(clippy::too_many_arguments)]
    fn decode_coefs(&mut self, which: &Ctx, ax: usize, ly: usize, tx: usize, bs: usize, b: &Av1Block, intra: bool, plane: usize, txtp_in: u8) -> (i32, u8, u8) {
        let hdr = self.hdr.clone();
        let chroma = plane != 0;
        let lossless = hdr.segmentation.lossless[b.seg_id as usize];
        let t_dim = TXFM_DIMENSIONS[tx];
        let (ab, lb) = self.ctx_bytes(which, ax, ly);
        let sctx = get_skip_ctx(&t_dim, bs, &ab, &lb, chroma, self.layout);
        let tctx = t_dim.ctx as usize;
        let all_skip = ba!(self, coef.skip[tctx][sctx]);
        if all_skip {
            return (-1, if lossless { WHT_WHT } else { DCT_DCT }, 0x40);
        }
        let txtp;
        if lossless {
            txtp = WHT_WHT;
        } else if t_dim.max as usize + intra as usize >= TX_64X64 {
            txtp = DCT_DCT;
        } else if chroma {
            txtp = if intra { TXTP_FROM_UVMODE[b.uv_mode as usize] } else { super::env::get_uv_inter_txtp(&t_dim, txtp_in) };
        } else if hdr.segmentation.qidx[b.seg_id as usize] == 0 {
            txtp = DCT_DCT;
        } else if intra {
            let ymn = if b.y_mode as usize == FILTER_PRED { FILTER_MODE_TO_Y_MODE[b.y_angle as usize] as usize } else { b.y_mode as usize };
            let mn = t_dim.min as usize;
            if hdr.reduced_txtp_set || mn == TX_16X16 {
                let idx = sa!(self, 4, m.txtp_intra2[mn][ymn]);
                txtp = TX_TYPES_PER_SET[idx];
            } else {
                let idx = sa!(self, 6, m.txtp_intra1[mn][ymn]);
                txtp = TX_TYPES_PER_SET[idx + 5];
            }
        } else {
            let mn = t_dim.min as usize;
            if hdr.reduced_txtp_set || t_dim.max as usize == TX_32X32 {
                let idx = ba!(self, m.txtp_inter3[mn]) as u8;
                txtp = idx.wrapping_sub(1) & IDTX;
            } else if mn == TX_16X16 {
                let idx = sa!(self, 11, m.txtp_inter2);
                txtp = TX_TYPES_PER_SET[idx + 12];
            } else {
                let idx = sa!(self, 15, m.txtp_inter1[mn]);
                txtp = TX_TYPES_PER_SET[idx + 24];
            }
        }

        let slw = (t_dim.lw as usize).min(TX_32X32);
        let slh = (t_dim.lh as usize).min(TX_32X32);
        let tx2dszctx = slw + slh;
        let tx_class = TX_TYPE_CLASS[txtp as usize];
        let is_1d = (tx_class != TX_CLASS_2D) as usize;
        let ci = chroma as usize;
        let ns = 4 + tx2dszctx;
        let eob_bin = match tx2dszctx {
            0 => sa!(self, ns, coef.eob_bin_16[ci][is_1d]),
            1 => sa!(self, ns, coef.eob_bin_32[ci][is_1d]),
            2 => sa!(self, ns, coef.eob_bin_64[ci][is_1d]),
            3 => sa!(self, ns, coef.eob_bin_128[ci][is_1d]),
            4 => sa!(self, ns, coef.eob_bin_256[ci][is_1d]),
            5 => sa!(self, ns, coef.eob_bin_512[ci]),
            _ => sa!(self, ns, coef.eob_bin_1024[ci]),
        } as i32;
        let eob = if eob_bin > 1 {
            let hi = ba!(self, coef.eob_hi_bit[tctx][ci][eob_bin as usize]) as i32;
            ((hi | 2) << (eob_bin - 2)) | self.msac().bools(eob_bin as u32 - 2) as i32
        } else {
            eob_bin
        };

        let hi_ctx_t = tctx.min(3);
        let mut rc: u32;
        let mut dc_tok: u32;
        let cf = &mut self.cf;
        let _ = cf;
        if eob != 0 {
            let mut levels = std::mem::take(&mut self.levels);
            let mut cf = std::mem::take(&mut self.cf);
            let ctx0 = 1 + (eob > 2 << tx2dszctx) as usize + (eob > 4 << tx2dszctx) as usize;
            let eob_tok = sa!(self, 2, coef.eob_base_tok[tctx][ci][ctx0]) as u32;
            let mut tok = eob_tok + 1;
            let mut level_tok = tok * 0x41;
            let mut mag = 0u32;
            let (stride, shift, shift2, mask, scan, ctx_offsets): (usize, u32, u32, u32, &[u16], Option<&[[u8; 5]; 5]>);
            match tx_class {
                TX_CLASS_2D => {
                    let nonsquare = (tx >= RTX_4X8) as usize;
                    ctx_offsets = Some(&LO_CTX_OFFSETS[nonsquare + (tx & nonsquare)]);
                    scan = SCANS[tx];
                    stride = 4 << slh;
                    shift = slh as u32 + 2;
                    shift2 = 0;
                    mask = (4 << slh) - 1;
                    for v in &mut levels[..stride * ((4 << slw) + 2)] {
                        *v = 0;
                    }
                }
                TX_CLASS_H => {
                    ctx_offsets = None;
                    scan = &[];
                    stride = 16;
                    shift = slh as u32 + 2;
                    shift2 = 0;
                    mask = (4 << slh) - 1;
                    for v in &mut levels[..stride * ((4 << slh) + 2)] {
                        *v = 0;
                    }
                }
                _ => {
                    ctx_offsets = None;
                    scan = &[];
                    stride = 16;
                    shift = slw as u32 + 2;
                    shift2 = slh as u32 + 2;
                    mask = (4 << slw) - 1;
                    for v in &mut levels[..stride * ((4 << slw) + 2)] {
                        *v = 0;
                    }
                }
            }
            let pos = |i: u32| -> (u32, u32, u32) {
                match tx_class {
                    TX_CLASS_2D => {
                        let rc = scan[i as usize] as u32;
                        (rc, rc >> shift, rc & mask)
                    }
                    TX_CLASS_H => (i, i & mask, i >> shift),
                    _ => {
                        let (x, y) = (i & mask, i >> shift);
                        ((x << shift2) | y, x, y)
                    }
                }
            };
            let (rc0, x, y) = pos(eob as u32);
            rc = rc0;
            if eob_tok == 2 {
                let c = if if tx_class == TX_CLASS_2D { (x | y) > 1 } else { y != 0 } { 14 } else { 7 };
                tok = {
                    let ts = &mut self.ts[self.tsi];
                    ts.msac.hi_tok(&mut ts.cdf.coef.br_tok[hi_ctx_t][ci][c])
                };
                level_tok = tok + (3 << 6);
            }
            cf[rc as usize] = (tok << 11) as i32;
            levels[x as usize * stride + y as usize] = level_tok as u8;
            let mut i = eob - 1;
            while i > 0 {
                let (rc_i, x, mut y) = pos(i as u32);
                let lo = x as usize * stride + y as usize;
                let ctx = get_lo_ctx(&levels, lo, tx_class, &mut mag, ctx_offsets, x, y, stride) as usize;
                if tx_class == TX_CLASS_2D {
                    y |= x;
                }
                let mut tk = sa!(self, 3, coef.base_tok[tctx][ci][ctx]) as u32;
                if tk == 3 {
                    mag &= 63;
                    let c = (if y > (tx_class == TX_CLASS_2D) as u32 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 } as usize;
                    tk = {
                        let ts = &mut self.ts[self.tsi];
                        ts.msac.hi_tok(&mut ts.cdf.coef.br_tok[hi_ctx_t][ci][c])
                    };
                    levels[lo] = (tk + (3 << 6)) as u8;
                    cf[rc_i as usize] = ((tk << 11) | rc) as i32;
                    rc = rc_i;
                } else {
                    levels[lo] = (tk * 0x41) as u8;
                    let v = if tk != 0 { (tk << 11) | rc } else { 0 };
                    if tk != 0 {
                        rc = rc_i;
                    }
                    cf[rc_i as usize] = v as i32;
                }
                i -= 1;
            }
            // dc
            let ctx = if tx_class == TX_CLASS_2D { 0 } else { get_lo_ctx(&levels, 0, tx_class, &mut mag, ctx_offsets, 0, 0, stride) as usize };
            dc_tok = sa!(self, 3, coef.base_tok[tctx][ci][ctx]) as u32;
            if dc_tok == 3 {
                if tx_class == TX_CLASS_2D {
                    mag = levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32;
                }
                mag &= 63;
                let c = if mag > 12 { 6 } else { ((mag + 1) >> 1) as usize };
                dc_tok = {
                    let ts = &mut self.ts[self.tsi];
                    ts.msac.hi_tok(&mut ts.cdf.coef.br_tok[hi_ctx_t][ci][c])
                };
            }
            self.levels = levels;
            self.cf = cf;
        } else {
            let tok_br = sa!(self, 2, coef.eob_base_tok[tctx][ci][0]) as u32;
            dc_tok = 1 + tok_br;
            if tok_br == 2 {
                dc_tok = {
                    let ts = &mut self.ts[self.tsi];
                    ts.msac.hi_tok(&mut ts.cdf.coef.br_tok[hi_ctx_t][ci][0])
                };
            }
            rc = 0;
        }

        // residual and sign
        let seg = b.seg_id as usize;
        let ts_dq = if self.ts[self.tsi].dq_frame { self.dq[seg][plane] } else { self.ts[self.tsi].dqmem[seg][plane] };
        let qm_tbl: Option<&[u8]> = if txtp < IDTX { self.qm[tx][plane].map(|(lv, c)| qm::table(lv, c, tx)) } else { None };
        let dq_shift = (tctx as i32 - 2).max(0);
        let cf_max = !(!127u32 << if self.bpc == 8 { 8 } else { self.bpc }) as i32;
        let mut cul_level: u32;
        let dc_sign_level: u32;
        let mut cf = std::mem::take(&mut self.cf);
        let mut walk_ac = rc != 0;
        if dc_tok == 0 {
            cul_level = 0;
            dc_sign_level = 1 << 6;
            walk_ac = true;
        } else {
            let dsc = get_dc_sign_ctx(&t_dim, &ab, &lb);
            let dc_sign = ba!(self, coef.dc_sign[ci][dsc]) as i32;
            let mut dc_dq = ts_dq[0] as i32;
            dc_sign_level = ((dc_sign - 1) & (2 << 6)) as u32;
            if let Some(q) = qm_tbl {
                dc_dq = (dc_dq * q[0] as i32 + 16) >> 5;
                if dc_tok == 15 {
                    dc_tok = read_golomb(self.msac()).wrapping_add(15);
                    dc_tok &= 0xfffff;
                    dc_dq = ((dc_dq as u32).wrapping_mul(dc_tok) & 0xffffff) as i32;
                } else {
                    dc_dq *= dc_tok as i32;
                }
                cul_level = dc_tok;
                dc_dq >>= dq_shift;
                dc_dq = (dc_dq as u32).min((cf_max + dc_sign) as u32) as i32;
            } else if dc_tok == 15 {
                dc_tok = read_golomb(self.msac()).wrapping_add(15);
                dc_tok &= 0xfffff;
                dc_dq = (((dc_dq as u32).wrapping_mul(dc_tok) & 0xffffff) >> dq_shift) as i32;
                dc_dq = (dc_dq as u32).min((cf_max + dc_sign) as u32) as i32;
                cul_level = dc_tok;
            } else {
                dc_dq = (dc_dq * dc_tok as i32) >> dq_shift;
                cul_level = dc_tok;
            }
            cf[0] = if dc_sign != 0 { -dc_dq } else { dc_dq };
        }
        if walk_ac && rc != 0 {
            let ac_dq = ts_dq[1] as u32;
            loop {
                let sign = self.msac().bool_equi() as i32;
                let rc_tok = cf[rc as usize] as u32;
                let mut tok;
                let dq_sat: u32;
                match qm_tbl {
                    Some(q) => {
                        let mut dq = (ac_dq * q[rc as usize] as u32 + 16) >> 5;
                        if rc_tok >= (15 << 11) {
                            tok = read_golomb(self.msac()).wrapping_add(15);
                            tok &= 0xfffff;
                            dq = dq.wrapping_mul(tok) & 0xffffff;
                        } else {
                            tok = rc_tok >> 11;
                            dq *= tok;
                        }
                        cul_level = cul_level.wrapping_add(tok);
                        dq >>= dq_shift;
                        dq_sat = dq.min((cf_max + sign) as u32);
                    }
                    None => {
                        if rc_tok >= (15 << 11) {
                            tok = read_golomb(self.msac()).wrapping_add(15);
                            tok &= 0xfffff;
                            let dq = (ac_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift;
                            dq_sat = dq.min((cf_max + sign) as u32);
                        } else {
                            tok = rc_tok >> 11;
                            dq_sat = (ac_dq * tok) >> dq_shift;
                        }
                        cul_level = cul_level.wrapping_add(tok);
                    }
                }
                cf[rc as usize] = if sign != 0 { -(dq_sat as i32) } else { dq_sat as i32 };
                rc = rc_tok & 0x3ff;
                if rc == 0 {
                    break;
                }
            }
        }
        self.cf = cf;
        (eob, txtp, (cul_level.min(63) | dc_sign_level) as u8)
    }

    fn itx(&mut self, pl: usize, off: usize, tx: usize, txtp: u8, eob: i32) {
        let stride = self.cur.pstride(pl);
        let bdm = self.bitdepth_max;
        itx::itxfm_add(&mut self.cur.planes[pl], off, stride, &mut self.cf, eob, tx, txtp, bdm);
    }

    #[allow(clippy::too_many_arguments)]
    fn read_coef_tree(&mut self, bs: usize, b: &Av1Block, ytx: usize, depth: usize, tx_split: [u16; 2], x_off: usize, y_off: usize, dst: usize) {
        let t_dim = TXFM_DIMENSIONS[ytx];
        let (txw, txh) = (t_dim.w as i32, t_dim.h as i32);
        let ys = self.cur.stride[0];
        if depth < 2 && tx_split[depth] != 0 && tx_split[depth] & (1 << (y_off * 4 + x_off)) != 0 {
            let sub = t_dim.sub as usize;
            let st = TXFM_DIMENSIONS[sub];
            let (txsw, txsh) = (st.w as i32, st.h as i32);
            self.read_coef_tree(bs, b, sub, depth + 1, tx_split, x_off * 2, y_off * 2, dst);
            self.bx += txsw;
            if txw >= txh && self.bx < self.bw {
                self.read_coef_tree(bs, b, sub, depth + 1, tx_split, x_off * 2 + 1, y_off * 2, dst + 4 * txsw as usize);
            }
            self.bx -= txsw;
            self.by += txsh;
            if txh >= txw && self.by < self.bh {
                let d2 = dst + 4 * txsh as usize * ys;
                self.read_coef_tree(bs, b, sub, depth + 1, tx_split, x_off * 2, y_off * 2 + 1, d2);
                self.bx += txsw;
                if txw >= txh && self.bx < self.bw {
                    self.read_coef_tree(bs, b, sub, depth + 1, tx_split, x_off * 2 + 1, y_off * 2 + 1, d2 + 4 * txsw as usize);
                }
                self.bx -= txsw;
            }
            self.by -= txsh;
        } else {
            let bx4 = (self.bx & 31) as usize;
            let by4 = (self.by & 31) as usize;
            let (eob, txtp, cf_ctx) = self.decode_coefs(&Ctx::Luma, bx4, by4, ytx, bs, b, false, 0, 0);
            let nw = (txw.min(self.bw - self.bx)) as usize;
            let nh = (txh.min(self.bh - self.by)) as usize;
            self.set_coef_ctx(&Ctx::Luma, bx4, by4, nw, nh, cf_ctx);
            let n = 1usize << t_dim.lw;
            for y in 0..txh as usize {
                for x in 0..n {
                    self.txtp_map[(by4 + y) * 32 + bx4 + x] = txtp;
                }
            }
            if eob >= 0 {
                self.itx(0, dst, ytx, txtp, eob);
            }
        }
    }

    fn top_sb(&self, pl: usize, by_luma: i32, y_aligned: i32) -> Option<usize> {
        let sby = by_luma >> self.sb_shift;
        if y_aligned & (self.sb_step - 1) == 0 && sby > 0 {
            Some((self.sb128w * 128 * (sby - 1)) as usize)
        } else {
            let _ = pl;
            None
        }
    }

    pub fn recon_b_intra(&mut self, bs: usize, intra_edge_flags: u8, b: &Av1Block) {
        let tiling = (self.ts[self.tsi].tiling.col_start, self.ts[self.tsi].tiling.col_end, self.ts[self.tsi].tiling.row_start, self.ts[self.tsi].tiling.row_end);
        let bx4 = (self.bx & 31) as usize;
        let by4 = (self.by & 31) as usize;
        let ss_ver = self.ss_ver;
        let ss_hor = self.ss_hor;
        let cbx4 = bx4 >> ss_hor;
        let cby4 = by4 >> ss_ver;
        let b_dim = BLOCK_DIMENSIONS[bs];
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let cw4 = (w4 + ss_hor) >> ss_hor;
        let ch4 = (h4 + ss_ver) >> ss_ver;
        let has_chroma = self.layout != Layout::I400 && (bw4 > ss_hor || self.bx & 1 != 0) && (bh4 > ss_ver || self.by & 1 != 0);
        let t_dim = TXFM_DIMENSIONS[b.tx as usize];
        let uv_t_dim = TXFM_DIMENSIONS[b.uvtx as usize];
        let cbw4 = (bw4 + ss_hor) >> ss_hor;
        let cbh4 = (bh4 + ss_ver) >> ss_ver;
        let ief_flag = (self.seq.intra_edge_filter as i32) << 10;
        let bitdepth = self.bpc;
        let bdm = self.bitdepth_max;
        let mut edge = vec![0u16; EDGE_LEN];
        let ys = self.cur.stride[0];
        let uvs = self.cur.stride[1];

        let mut init_y = 0;
        while init_y < h4 {
            let sub_h4 = h4.min(16 + init_y);
            let sub_ch4 = ch4.min((init_y + 16) >> ss_ver);
            let mut init_x = 0;
            while init_x < w4 {
                if b.pal_sz[0] != 0 {
                    let off = 4 * (self.by as usize * ys + self.bx as usize);
                    let pal = self.pal[0];
                    ipred::pal_pred(&mut self.cur.planes[0], off, ys, &pal, &self.pal_idx_y, (bw4 * 4) as usize, (bh4 * 4) as usize);
                }
                let intra_flags = sm_flag(&self.a[self.ai], bx4) | sm_flag(&self.l, by4) | ief_flag;
                let sb_has_tr = if init_x + 16 < w4 {
                    true
                } else if init_y != 0 {
                    false
                } else {
                    intra_edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0
                };
                let sb_has_bl = if init_x != 0 {
                    false
                } else if init_y + 16 < h4 {
                    true
                } else {
                    intra_edge_flags & EDGE_I444_LEFT_HAS_BOTTOM != 0
                };
                let sub_w4 = w4.min(init_x + 16);
                let mut y = init_y;
                self.by += init_y;
                while y < sub_h4 {
                    let mut dst = 4 * (self.by as usize * ys + (self.bx + init_x) as usize);
                    let mut x = init_x;
                    self.bx += init_x;
                    while x < sub_w4 {
                        if b.pal_sz[0] == 0 {
                            let mut angle = b.y_angle as i32;
                            let edge_flags = (if (y > init_y || !sb_has_tr) && x + t_dim.w as i32 >= sub_w4 { 0 } else { EDGE_I444_TOP_HAS_RIGHT })
                                | (if x > init_x || (!sb_has_bl && y + t_dim.h as i32 >= sub_h4) { 0 } else { EDGE_I444_LEFT_HAS_BOTTOM });
                            let tsb = self.top_sb(0, self.by, self.by);
                            let top = match tsb {
                                Some(o) => Top::Sb(&self.ipred_edge[0][o..]),
                                None => Top::Frame,
                            };
                            let m = ipred::prepare_intra_edges(
                                self.bx,
                                self.bx > tiling.0,
                                self.by,
                                self.by > tiling.2,
                                tiling.1,
                                tiling.3,
                                edge_flags,
                                &self.cur.planes[0],
                                dst,
                                ys,
                                top,
                                b.y_mode as usize,
                                &mut angle,
                                t_dim.w as i32,
                                t_dim.h as i32,
                                self.seq.intra_edge_filter,
                                &mut edge,
                                bitdepth,
                            );
                            ipred::intra_pred(
                                m,
                                &mut self.cur.planes[0],
                                dst,
                                ys,
                                &edge,
                                t_dim.w as usize * 4,
                                t_dim.h as usize * 4,
                                angle | intra_flags,
                                4 * self.bw - 4 * self.bx,
                                4 * self.bh - 4 * self.by,
                                bdm,
                            );
                        }
                        if !b.skip {
                            let ax = bx4 + x as usize;
                            let ly = by4 + y as usize;
                            let (eob, txtp, cf_ctx) = self.decode_coefs(&Ctx::Luma, ax, ly, b.tx as usize, bs, b, true, 0, 0);
                            let nw = (t_dim.w as i32).min(self.bw - self.bx) as usize;
                            let nh = (t_dim.h as i32).min(self.bh - self.by) as usize;
                            self.set_coef_ctx(&Ctx::Luma, ax, ly, nw, nh, cf_ctx);
                            if eob >= 0 {
                                self.itx(0, dst, b.tx as usize, txtp, eob);
                            }
                        } else {
                            self.set_coef_ctx(&Ctx::Luma, bx4 + x as usize, by4 + y as usize, 1 << t_dim.lw, 1 << t_dim.lh, 0x40);
                        }
                        dst += 4 * t_dim.w as usize;
                        x += t_dim.w as i32;
                        self.bx += t_dim.w as i32;
                    }
                    self.bx -= x;
                    y += t_dim.h as i32;
                    self.by += t_dim.h as i32;
                }
                self.by -= y;

                if has_chroma {
                    if b.uv_mode as usize == CFL_PRED {
                        let mut ac = vec![0i16; 32 * 32];
                        let y_src = 4 * (self.bx & !ss_hor) as usize + 4 * (self.by & !ss_ver) as usize * ys;
                        let uv_off = 4 * ((self.bx >> ss_hor) as usize + (self.by >> ss_ver) as usize * uvs);
                        let furthest_r = ((cw4 << ss_hor) + t_dim.w as i32 - 1) & !(t_dim.w as i32 - 1);
                        let furthest_b = ((ch4 << ss_ver) + t_dim.h as i32 - 1) & !(t_dim.h as i32 - 1);
                        ipred::cfl_ac(
                            &mut ac,
                            &self.cur.planes[0],
                            y_src,
                            ys,
                            (cbw4 - (furthest_r >> ss_hor)) as usize,
                            (cbh4 - (furthest_b >> ss_ver)) as usize,
                            (cbw4 * 4) as usize,
                            (cbh4 * 4) as usize,
                            ss_hor != 0,
                            ss_ver != 0,
                        );
                        for pl in 0..2 {
                            if b.cfl_alpha[pl] == 0 {
                                continue;
                            }
                            let mut angle = 0;
                            let tsb = self.top_sb(pl + 1, self.by, self.by & !ss_ver);
                            let top = match tsb {
                                Some(o) => Top::Sb(&self.ipred_edge[pl + 1][o..]),
                                None => Top::Frame,
                            };
                            let xpos = self.bx >> ss_hor;
                            let ypos = self.by >> ss_ver;
                            let m = ipred::prepare_intra_edges(
                                xpos,
                                xpos > (tiling.0 >> ss_hor),
                                ypos,
                                ypos > (tiling.2 >> ss_ver),
                                tiling.1 >> ss_hor,
                                tiling.3 >> ss_ver,
                                0,
                                &self.cur.planes[1 + pl],
                                uv_off,
                                uvs,
                                top,
                                DC_PRED,
                                &mut angle,
                                uv_t_dim.w as i32,
                                uv_t_dim.h as i32,
                                false,
                                &mut edge,
                                bitdepth,
                            );
                            ipred::cfl_pred(
                                m,
                                &mut self.cur.planes[1 + pl],
                                uv_off,
                                uvs,
                                &edge,
                                uv_t_dim.w as usize * 4,
                                uv_t_dim.h as usize * 4,
                                &ac,
                                b.cfl_alpha[pl] as i32,
                                bdm,
                            );
                        }
                    } else if b.pal_sz[1] != 0 {
                        let uv_off = 4 * ((self.bx >> ss_hor) as usize + (self.by >> ss_ver) as usize * uvs);
                        for pl in 1..=2 {
                            let pal = self.pal[pl];
                            ipred::pal_pred(&mut self.cur.planes[pl], uv_off, uvs, &pal, &self.pal_idx_uv, (cbw4 * 4) as usize, (cbh4 * 4) as usize);
                        }
                    }

                    let sm_uv_fl = sm_uv_flag(&self.a[self.ai], cbx4) | sm_uv_flag(&self.l, cby4);
                    let lsh = self.layout as u8 - 1;
                    let uv_sb_has_tr = if ((init_x + 16) >> ss_hor) < cw4 {
                        true
                    } else if init_y != 0 {
                        false
                    } else {
                        intra_edge_flags & (EDGE_I420_TOP_HAS_RIGHT >> lsh) != 0
                    };
                    let uv_sb_has_bl = if init_x != 0 {
                        false
                    } else if ((init_y + 16) >> ss_ver) < ch4 {
                        true
                    } else {
                        intra_edge_flags & (EDGE_I420_LEFT_HAS_BOTTOM >> lsh) != 0
                    };
                    let sub_cw4 = cw4.min((init_x + 16) >> ss_hor);
                    for pl in 0..2 {
                        let mut y = init_y >> ss_ver;
                        self.by += init_y;
                        while y < sub_ch4 {
                            let mut dst = 4 * ((self.by >> ss_ver) as usize * uvs + ((self.bx + init_x) >> ss_hor) as usize);
                            let mut x = init_x >> ss_hor;
                            self.bx += init_x;
                            while x < sub_cw4 {
                                if !((b.uv_mode as usize == CFL_PRED && b.cfl_alpha[pl] != 0) || b.pal_sz[1] != 0) {
                                    let mut angle = b.uv_angle as i32;
                                    let edge_flags = (if (y > (init_y >> ss_ver) || !uv_sb_has_tr) && x + uv_t_dim.w as i32 >= sub_cw4 {
                                        0
                                    } else {
                                        EDGE_I444_TOP_HAS_RIGHT
                                    }) | (if x > (init_x >> ss_hor) || (!uv_sb_has_bl && y + uv_t_dim.h as i32 >= sub_ch4) {
                                        0
                                    } else {
                                        EDGE_I444_LEFT_HAS_BOTTOM
                                    });
                                    let tsb = self.top_sb(pl + 1, self.by, self.by & !ss_ver);
                                    let top = match tsb {
                                        Some(o) => Top::Sb(&self.ipred_edge[pl + 1][o..]),
                                        None => Top::Frame,
                                    };
                                    let uv_mode = if b.uv_mode as usize == CFL_PRED { DC_PRED } else { b.uv_mode as usize };
                                    let xpos = self.bx >> ss_hor;
                                    let ypos = self.by >> ss_ver;
                                    let m = ipred::prepare_intra_edges(
                                        xpos,
                                        xpos > (tiling.0 >> ss_hor),
                                        ypos,
                                        ypos > (tiling.2 >> ss_ver),
                                        tiling.1 >> ss_hor,
                                        tiling.3 >> ss_ver,
                                        edge_flags,
                                        &self.cur.planes[1 + pl],
                                        dst,
                                        uvs,
                                        top,
                                        uv_mode,
                                        &mut angle,
                                        uv_t_dim.w as i32,
                                        uv_t_dim.h as i32,
                                        self.seq.intra_edge_filter,
                                        &mut edge,
                                        bitdepth,
                                    );
                                    angle |= ief_flag;
                                    ipred::intra_pred(
                                        m,
                                        &mut self.cur.planes[1 + pl],
                                        dst,
                                        uvs,
                                        &edge,
                                        uv_t_dim.w as usize * 4,
                                        uv_t_dim.h as usize * 4,
                                        angle | sm_uv_fl,
                                        (4 * self.bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor,
                                        (4 * self.bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver,
                                        bdm,
                                    );
                                }
                                let ax = cbx4 + x as usize;
                                let ly = cby4 + y as usize;
                                if !b.skip {
                                    let (eob, txtp, cf_ctx) = self.decode_coefs(&Ctx::Chroma(pl), ax, ly, b.uvtx as usize, bs, b, true, 1 + pl, 0);
                                    let ctw = (uv_t_dim.w as i32).min((self.bw - self.bx + ss_hor) >> ss_hor) as usize;
                                    let cth = (uv_t_dim.h as i32).min((self.bh - self.by + ss_ver) >> ss_ver) as usize;
                                    self.set_coef_ctx(&Ctx::Chroma(pl), ax, ly, ctw, cth, cf_ctx);
                                    if eob >= 0 {
                                        self.itx(1 + pl, dst, b.uvtx as usize, txtp, eob);
                                    }
                                } else {
                                    self.set_coef_ctx(&Ctx::Chroma(pl), ax, ly, 1 << uv_t_dim.lw, 1 << uv_t_dim.lh, 0x40);
                                }
                                dst += uv_t_dim.w as usize * 4;
                                x += uv_t_dim.w as i32;
                                self.bx += (uv_t_dim.w as i32) << ss_hor;
                            }
                            self.bx -= x << ss_hor;
                            y += uv_t_dim.h as i32;
                            self.by += (uv_t_dim.h as i32) << ss_ver;
                        }
                        self.by -= y << ss_ver;
                    }
                }
                init_x += 16;
            }
            init_y += 16;
        }
    }

    /// Reference picture for `refidx`, or the current frame (intrabc).
    fn ref_pic(&self, refidx: Option<usize>) -> Arc<Picture> {
        match refidx {
            Some(i) => self.refp[i].as_ref().unwrap().pic.clone(),
            None => Arc::new(Picture { planes: Default::default(), stride: [0; 2], w: 0, h: 0, layout: self.layout, bpc: self.bpc }),
        }
    }

    /// `mc()`: predict into `out` (pixels in `cur`, an obmc scratch, or
    /// int16 compound intermediates).
    #[allow(clippy::too_many_arguments)]
    fn mc(&mut self, out: McOut, bw4: i32, bh4: i32, bx: i32, by: i32, pl: usize, mv: Mv, refidx: Option<usize>, filter2d: u8) {
        let ss_ver = (pl != 0 && self.layout == Layout::I420) as i32;
        let ss_hor = (pl != 0 && self.layout != Layout::I444) as i32;
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let (mvx, mvy) = (mv.x as i32, mv.y as i32);
        let mx = mvx & (15 >> (ss_hor == 0) as i32);
        let my = mvy & (15 >> (ss_ver == 0) as i32);
        let bd = Bd::new(self.bitdepth_max);
        let rp = self.ref_pic(refidx);
        let (rplane, rstride, rw, rh): (&[u16], usize, i32, i32) = match refidx {
            Some(_) => (&rp.planes[pl], rp.pstride(pl), rp.w, rp.h),
            None => (&self.cur.planes[pl], self.cur.pstride(pl), self.cur.w, self.cur.h),
        };
        let w = bw4 * h_mul;
        let h = bh4 * v_mul;
        let mut buf = std::mem::take(&mut self.mc_buf);
        let (src_stride, sx, sy, scaled);
        // Offset of the block in the reference plane when its whole filter
        // window lies inside the picture: then no padded copy is needed.
        let mut direct = None;
        if rw == self.cur.w && rh == self.cur.h {
            let dx = bx * h_mul + (mvx >> (3 + ss_hor));
            let dy = by * v_mul + (mvy >> (3 + ss_ver));
            let (iw, ih) = if refidx.is_some() {
                ((self.cur.w + ss_hor) >> ss_hor, (self.cur.h + ss_ver) >> ss_ver)
            } else {
                (self.bw * 4 >> ss_hor, self.bh * 4 >> ss_ver)
            };
            if refidx.is_some() && dx >= 3 && dy >= 3 && dx + w + 4 <= iw && dy + h + 4 <= ih {
                direct = Some(dy as usize * rstride + dx as usize);
            } else {
                mc::fetch(&mut buf, rplane, rstride, iw, ih, dx - 3, dy - 3, w + 7, h + 7);
            }
            src_stride = (w + 7) as usize;
            sx = mx << (ss_hor == 0) as i32;
            sy = my << (ss_ver == 0) as i32;
            scaled = None;
        } else {
            let ri = refidx.unwrap();
            let svc = self.svc[ri];
            let orig_pos_y = (by * v_mul << 4) + mvy * (1 << (ss_ver == 0) as i32);
            let orig_pos_x = (bx * h_mul << 4) + mvx * (1 << (ss_hor == 0) as i32);
            let scale_mv = |val: i32, scale: i32| -> i32 {
                let tmp = val as i64 * scale as i64 + (scale as i64 - 0x4000) * 8;
                let v = ((tmp.abs() + 128) >> 8) as i32;
                (if tmp < 0 { -v } else { v }) + 32
            };
            let pos_x = scale_mv(orig_pos_x, svc[0].scale);
            let pos_y = scale_mv(orig_pos_y, svc[1].scale);
            let left = pos_x >> 10;
            let top = pos_y >> 10;
            let right = ((pos_x + (w - 1) * svc[0].step) >> 10) + 1;
            let bottom = ((pos_y + (h - 1) * svc[1].step) >> 10) + 1;
            let iw = (rw + ss_hor) >> ss_hor;
            let ih = (rh + ss_ver) >> ss_ver;
            mc::fetch(&mut buf, rplane, rstride, iw, ih, left - 3, top - 3, right - left + 7, bottom - top + 7);
            src_stride = (right - left + 7) as usize;
            sx = pos_x & 0x3ff;
            sy = pos_y & 0x3ff;
            scaled = Some((svc[0].step, svc[1].step));
        }
        let src = match direct {
            // Only taken with a reference picture (`refidx` is `Some`).
            Some(off) => Src { buf: &rp.planes[pl], off, stride: rstride },
            None => Src { buf: &buf, off: 3 * src_stride + 3, stride: src_stride },
        };
        let f2d = filter2d as usize;
        let run = |dst: &mut Dst| match scaled {
            None => mc::mc(dst, &src, w, h, sx, sy, f2d, bd),
            Some((dxs, dys)) => mc::mc_scaled(dst, &src, w, h, sx, sy, dxs, dys, f2d, bd),
        };
        match out {
            McOut::Cur { off } => {
                let stride = self.cur.pstride(pl);
                let mut d = Dst::Px { buf: &mut self.cur.planes[pl], off, stride };
                run(&mut d);
            }
            McOut::Buf(b, stride) => {
                let mut d = Dst::Px { buf: b, off: 0, stride };
                run(&mut d);
            }
            McOut::Tmp(t) => {
                let mut d = Dst::Tmp { buf: t, stride: w as usize };
                run(&mut d);
            }
        }
        self.mc_buf = buf;
    }

    #[allow(clippy::too_many_arguments)]
    fn obmc(&mut self, dst_off: usize, b_dim: [u8; 4], pl: usize, bx4: usize, by4: usize, w4: i32, h4: i32) {
        let ss_ver = (pl != 0 && self.layout == Layout::I420) as i32;
        let ss_hor = (pl != 0 && self.layout != Layout::I444) as i32;
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let stride = self.cur.pstride(pl);
        let (bx, by) = (self.bx, self.by);
        if self.by > self.ts[self.tsi].tiling.row_start && (pl == 0 || b_dim[0] as i32 * h_mul + b_dim[1] as i32 * v_mul >= 16) {
            let (mut i, mut x) = (0, 0);
            while x < w4 && i < (b_dim[2] as i32).min(4) {
                let a_r = *self.rf.as_ref().unwrap().blk(by - 1, bx + x + 1);
                let a_b_dim = BLOCK_DIMENSIONS[a_r.bs as usize];
                let step4 = (a_b_dim[0] as i32).clamp(2, 16);
                if a_r.r[0] > 0 {
                    let ow4 = step4.min(b_dim[0] as i32);
                    let oh4 = (b_dim[1] as i32).min(16) >> 1;
                    let lw = (ow4 * h_mul) as usize;
                    let lh = (((oh4 * 3 + 3) >> 2) * v_mul) as usize;
                    let mut lap = vec![0u16; lw * lh];
                    let ai = self.ai;
                    let f = FILTER_2D[self.a[ai].filter[1][bx4 + x as usize + 1] as usize][self.a[ai].filter[0][bx4 + x as usize + 1] as usize];
                    self.mc(McOut::Buf(&mut lap, lw), ow4, (oh4 * 3 + 3) >> 2, bx + x, by, pl, a_r.mv[0], Some(a_r.r[0] as usize - 1), f);
                    mc::blend_h(&mut self.cur.planes[pl], dst_off + (x * h_mul) as usize, stride, &lap, lw, (v_mul * oh4) as usize);
                    i += 1;
                }
                x += step4;
            }
        }
        if self.bx > self.ts[self.tsi].tiling.col_start {
            let (mut i, mut y) = (0, 0);
            while y < h4 && i < (b_dim[3] as i32).min(4) {
                let l_r = *self.rf.as_ref().unwrap().blk(by + y + 1, bx - 1);
                let l_b_dim = BLOCK_DIMENSIONS[l_r.bs as usize];
                let step4 = (l_b_dim[1] as i32).clamp(2, 16);
                if l_r.r[0] > 0 {
                    let ow4 = (b_dim[0] as i32).min(16) >> 1;
                    let oh4 = step4.min(b_dim[1] as i32);
                    let lw = (h_mul * ow4) as usize;
                    let lh = (v_mul * oh4) as usize;
                    let mut lap = vec![0u16; lw * lh];
                    let f = FILTER_2D[self.l.filter[1][by4 + y as usize + 1] as usize][self.l.filter[0][by4 + y as usize + 1] as usize];
                    self.mc(McOut::Buf(&mut lap, lw), ow4, oh4, bx, by + y, pl, l_r.mv[0], Some(l_r.r[0] as usize - 1), f);
                    mc::blend_v(&mut self.cur.planes[pl], dst_off + (y * v_mul) as usize * stride, stride, &lap, lw, lh);
                    i += 1;
                }
                y += step4;
            }
        }
    }

    /// `warp_affine`: `out` is a pixel offset in `cur` or a tmp buffer.
    fn warp_affine(&mut self, out: McOut, b_dim: [u8; 4], pl: usize, refidx: usize, wmp: &WarpedMotionParams) {
        let ss_ver = (pl != 0 && self.layout == Layout::I420) as i32;
        let ss_hor = (pl != 0 && self.layout != Layout::I444) as i32;
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let mat = wmp.matrix;
        let rp = self.ref_pic(Some(refidx));
        let width = (rp.w + ss_hor) >> ss_hor;
        let height = (rp.h + ss_ver) >> ss_ver;
        let bd = Bd::new(self.bitdepth_max);
        let rstride = rp.pstride(pl);
        let bw = b_dim[0] as i32 * h_mul;
        let mut buf = Vec::new();
        let mut out = out;
        let mut y = 0;
        while y < b_dim[1] as i32 * v_mul {
            let src_y = self.by * 4 + ((y + 4) << ss_ver);
            let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
            let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
            let mut x = 0;
            while x < bw {
                let src_x = self.bx * 4 + ((x + 4) << ss_hor);
                let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> ss_hor;
                let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> ss_ver;
                let dx = (mvx >> 16) as i32 - 4;
                let mx = ((mvx as i32 & 0xffff) - wmp.abcd[0] as i32 * 4 - wmp.abcd[1] as i32 * 7) & !0x3f;
                let dy = (mvy >> 16) as i32 - 4;
                let my = ((mvy as i32 & 0xffff) - wmp.abcd[2] as i32 * 4 - wmp.abcd[3] as i32 * 4) & !0x3f;
                mc::fetch(&mut buf, &rp.planes[pl], rstride, width, height, dx - 3, dy - 3, 15, 15);
                let src = Src { buf: &buf, off: 3 * 15 + 3, stride: 15 };
                match &mut out {
                    McOut::Cur { off } => {
                        let stride = self.cur.pstride(pl);
                        let mut d = Dst::Px { buf: &mut self.cur.planes[pl], off: *off, stride };
                        mc::warp8x8(&mut d, x, y, &src, &wmp.abcd, mx, my, bd);
                    }
                    McOut::Tmp(t) => {
                        let mut d = Dst::Tmp { buf: t, stride: bw as usize };
                        mc::warp8x8(&mut d, x, y, &src, &wmp.abcd, mx, my, bd);
                    }
                    McOut::Buf(..) => unreachable!(),
                }
                x += 8;
            }
            y += 8;
        }
    }

    pub fn recon_b_inter(&mut self, bs: usize, b: &Av1Block) -> DResult {
        let hdr = self.hdr.clone();
        let bx4 = (self.bx & 31) as usize;
        let by4 = (self.by & 31) as usize;
        let ss_ver = self.ss_ver;
        let ss_hor = self.ss_hor;
        let cbx4 = bx4 >> ss_hor;
        let cby4 = by4 >> ss_ver;
        let b_dim = BLOCK_DIMENSIONS[bs];
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let has_chroma = self.layout != Layout::I400 && (bw4 > ss_hor || self.bx & 1 != 0) && (bh4 > ss_ver || self.by & 1 != 0);
        let chr_layout_idx = if self.layout == Layout::I400 { 0 } else { 3 - self.layout as usize };
        let cbh4 = (bh4 + ss_ver) >> ss_ver;
        let cbw4 = (bw4 + ss_hor) >> ss_hor;
        let ys = self.cur.stride[0];
        let uvs = self.cur.stride[1];
        let dst = 4 * (self.by as usize * ys + self.bx as usize);
        let uvdstoff = 4 * ((self.bx >> ss_hor) as usize + (self.by >> ss_ver) as usize * uvs);
        let bd = Bd::new(self.bitdepth_max);
        let (bx, by) = (self.bx, self.by);

        if hdr.is_key_or_intra() {
            // intrabc
            self.mc(McOut::Cur { off: dst }, bw4, bh4, bx, by, 0, b.mv[0], None, FILTER_2D_BILINEAR as u8);
            if has_chroma {
                for pl in 1..3 {
                    self.mc(
                        McOut::Cur { off: uvdstoff },
                        bw4 << (bw4 == ss_hor) as i32,
                        bh4 << (bh4 == ss_ver) as i32,
                        bx & !ss_hor,
                        by & !ss_ver,
                        pl,
                        b.mv[0],
                        None,
                        FILTER_2D_BILINEAR as u8,
                    );
                }
            }
        } else if b.comp_type == COMP_INTER_NONE {
            let r0 = b.r#ref[0] as usize;
            let filter_2d = b.filter2d;
            let warp = bw4.min(bh4) > 1 && ((b.inter_mode == GLOBALMV && self.gmv_warp_allowed[r0]) || (b.motion_mode == MM_WARP && self.warpmv.typ > WM_TRANSLATION));
            let wmp = if b.motion_mode == MM_WARP { self.warpmv } else { hdr.gmv[r0] };
            if warp {
                self.warp_affine(McOut::Cur { off: dst }, b_dim, 0, r0, &wmp);
            } else {
                self.mc(McOut::Cur { off: dst }, bw4, bh4, bx, by, 0, b.mv[0], Some(r0), filter_2d);
                if b.motion_mode == MM_OBMC {
                    self.obmc(dst, b_dim, 0, bx4, by4, w4, h4);
                }
            }
            if b.interintra_type != 0 {
                let mut m = if b.interintra_mode == II_SMOOTH_PRED { SMOOTH_PRED } else { b.interintra_mode as usize };
                let mut tmp = vec![0u16; 64 * 64];
                let mut angle = 0;
                let mut edge = vec![0u16; EDGE_LEN];
                let tsb = self.top_sb(0, self.by, self.by);
                let t = &self.ts[self.tsi].tiling;
                let top = match tsb {
                    Some(o) => Top::Sb(&self.ipred_edge[0][o..]),
                    None => Top::Frame,
                };
                m = ipred::prepare_intra_edges(
                    bx,
                    bx > t.col_start,
                    by,
                    by > t.row_start,
                    t.col_end,
                    t.row_end,
                    0,
                    &self.cur.planes[0],
                    dst,
                    ys,
                    top,
                    m,
                    &mut angle,
                    bw4,
                    bh4,
                    false,
                    &mut edge,
                    self.bpc,
                );
                let (w, h) = ((bw4 * 4) as usize, (bh4 * 4) as usize);
                ipred::intra_pred(m, &mut tmp, 0, w, &edge, w, h, 0, 0, 0, self.bitdepth_max);
                let mask = wedge::ii_mask(0, bs, b.interintra_type, b.interintra_mode, b.wedge_idx);
                mc::blend(&mut self.cur.planes[0], dst, ys, &tmp, w, h, mask);
            }

            if has_chroma {
                let mut is_sub8x8 = bw4 == ss_hor || bh4 == ss_ver;
                let rf = self.rf.as_ref().unwrap();
                if is_sub8x8 {
                    if bw4 == 1 {
                        is_sub8x8 &= rf.blk(by, bx - 1).r[0] > 0;
                    }
                    if bh4 == ss_ver {
                        is_sub8x8 &= rf.blk(by - 1, bx).r[0] > 0;
                    }
                    if bw4 == 1 && bh4 == ss_ver {
                        is_sub8x8 &= rf.blk(by - 1, bx - 1).r[0] > 0;
                    }
                }
                if is_sub8x8 {
                    let (mut h_off, mut v_off) = (0usize, 0usize);
                    if bw4 == 1 && bh4 == ss_ver {
                        let rr = *self.rf.as_ref().unwrap().blk(by - 1, bx - 1);
                        for pl in 0..2 {
                            let f = self.tl_4x4_filter;
                            self.mc(McOut::Cur { off: uvdstoff }, bw4, bh4, bx - 1, by - 1, 1 + pl, rr.mv[0], Some(rr.r[0] as usize - 1), f);
                        }
                        v_off = 2 * uvs;
                        h_off = 2;
                    }
                    if bw4 == 1 {
                        let lf = FILTER_2D[self.l.filter[1][by4] as usize][self.l.filter[0][by4] as usize];
                        let rr = *self.rf.as_ref().unwrap().blk(by, bx - 1);
                        for pl in 0..2 {
                            self.mc(McOut::Cur { off: uvdstoff + v_off }, bw4, bh4, bx - 1, by, 1 + pl, rr.mv[0], Some(rr.r[0] as usize - 1), lf);
                        }
                        h_off = 2;
                    }
                    if bh4 == ss_ver {
                        let ai = self.ai;
                        let tf = FILTER_2D[self.a[ai].filter[1][bx4] as usize][self.a[ai].filter[0][bx4] as usize];
                        let rr = *self.rf.as_ref().unwrap().blk(by - 1, bx);
                        for pl in 0..2 {
                            self.mc(McOut::Cur { off: uvdstoff + h_off }, bw4, bh4, bx, by - 1, 1 + pl, rr.mv[0], Some(rr.r[0] as usize - 1), tf);
                        }
                        v_off = 2 * uvs;
                    }
                    for pl in 0..2 {
                        self.mc(McOut::Cur { off: uvdstoff + h_off + v_off }, bw4, bh4, bx, by, 1 + pl, b.mv[0], Some(r0), filter_2d);
                    }
                } else {
                    let cwarp = cbw4.min(cbh4) > 1 && ((b.inter_mode == GLOBALMV && self.gmv_warp_allowed[r0]) || (b.motion_mode == MM_WARP && self.warpmv.typ > WM_TRANSLATION));
                    if cwarp {
                        for pl in 0..2 {
                            self.warp_affine(McOut::Cur { off: uvdstoff }, b_dim, 1 + pl, r0, &wmp);
                        }
                    } else {
                        for pl in 0..2 {
                            self.mc(
                                McOut::Cur { off: uvdstoff },
                                bw4 << (bw4 == ss_hor) as i32,
                                bh4 << (bh4 == ss_ver) as i32,
                                bx & !ss_hor,
                                by & !ss_ver,
                                1 + pl,
                                b.mv[0],
                                Some(r0),
                                filter_2d,
                            );
                            if b.motion_mode == MM_OBMC {
                                self.obmc(uvdstoff, b_dim, 1 + pl, bx4, by4, w4, h4);
                            }
                        }
                    }
                    if b.interintra_type != 0 {
                        let ii_mask = wedge::ii_mask(chr_layout_idx, bs, b.interintra_type, b.interintra_mode, b.wedge_idx);
                        for pl in 0..2 {
                            let mut tmp = vec![0u16; 64 * 64];
                            let mut edge = vec![0u16; EDGE_LEN];
                            let mut m = if b.interintra_mode == II_SMOOTH_PRED { SMOOTH_PRED } else { b.interintra_mode as usize };
                            let mut angle = 0;
                            let tsb = self.top_sb(pl + 1, self.by, self.by);
                            let t = &self.ts[self.tsi].tiling;
                            let top = match tsb {
                                Some(o) => Top::Sb(&self.ipred_edge[pl + 1][o..]),
                                None => Top::Frame,
                            };
                            m = ipred::prepare_intra_edges(
                                bx >> ss_hor,
                                (bx >> ss_hor) > (t.col_start >> ss_hor),
                                by >> ss_ver,
                                (by >> ss_ver) > (t.row_start >> ss_ver),
                                t.col_end >> ss_hor,
                                t.row_end >> ss_ver,
                                0,
                                &self.cur.planes[1 + pl],
                                uvdstoff,
                                uvs,
                                top,
                                m,
                                &mut angle,
                                cbw4,
                                cbh4,
                                false,
                                &mut edge,
                                self.bpc,
                            );
                            let (w, h) = ((cbw4 * 4) as usize, (cbh4 * 4) as usize);
                            ipred::intra_pred(m, &mut tmp, 0, w, &edge, w, h, 0, 0, 0, self.bitdepth_max);
                            mc::blend(&mut self.cur.planes[1 + pl], uvdstoff, uvs, &tmp, w, h, ii_mask);
                        }
                    }
                }
            }
            self.tl_4x4_filter = filter_2d;
        } else {
            let filter_2d = b.filter2d;
            let mut tmp = [vec![0i16; 128 * 128], vec![0i16; 128 * 128]];
            let mut seg_mask = vec![0u8; 128 * 128];
            let mut jnt_weight = 0;
            let (bw, bh) = ((bw4 * 4) as usize, (bh4 * 4) as usize);
            for i in 0..2 {
                let ri = b.r#ref[i] as usize;
                let (t0, t1) = tmp.split_at_mut(1);
                let t = if i == 0 { &mut t0[0] } else { &mut t1[0] };
                if b.inter_mode == GLOBALMV_GLOBALMV && self.gmv_warp_allowed[ri] {
                    let g = hdr.gmv[ri];
                    self.warp_affine(McOut::Tmp(t), b_dim, 0, ri, &g);
                } else {
                    self.mc(McOut::Tmp(t), bw4, bh4, bx, by, 0, b.mv[i], Some(ri), filter_2d);
                }
            }
            let ms = b.mask_sign as usize;
            let mut mask_src: Option<&'static [u8]> = None;
            let mut use_seg = false;
            match b.comp_type {
                COMP_INTER_AVG => mc::avg(&mut self.cur.planes[0], dst, ys, &tmp[0], &tmp[1], bw, bh, bd),
                COMP_INTER_WEIGHTED_AVG => {
                    jnt_weight = self.jnt_weights[b.r#ref[0] as usize][b.r#ref[1] as usize] as i32;
                    mc::w_avg(&mut self.cur.planes[0], dst, ys, &tmp[0], &tmp[1], bw, bh, jnt_weight, bd);
                }
                COMP_INTER_SEG => {
                    mc::w_mask(
                        &mut self.cur.planes[0],
                        dst,
                        ys,
                        &tmp[ms],
                        &tmp[1 - ms],
                        bw,
                        bh,
                        &mut seg_mask,
                        ms as i32,
                        chr_layout_idx >= 1,
                        chr_layout_idx == 2,
                        bd,
                    );
                    use_seg = true;
                }
                _ => {
                    let m = wedge::wedge_mask(0, bs, 0, b.wedge_idx as usize);
                    mc::mask(&mut self.cur.planes[0], dst, ys, &tmp[ms], &tmp[1 - ms], bw, bh, m, bd);
                    if has_chroma {
                        mask_src = Some(wedge::wedge_mask(chr_layout_idx, bs, ms, b.wedge_idx as usize));
                    }
                }
            }
            if has_chroma {
                for pl in 0..2 {
                    for i in 0..2 {
                        let ri = b.r#ref[i] as usize;
                        let (t0, t1) = tmp.split_at_mut(1);
                        let t = if i == 0 { &mut t0[0] } else { &mut t1[0] };
                        if b.inter_mode == GLOBALMV_GLOBALMV && cbw4.min(cbh4) > 1 && self.gmv_warp_allowed[ri] {
                            let g = hdr.gmv[ri];
                            self.warp_affine(McOut::Tmp(t), b_dim, 1 + pl, ri, &g);
                        } else {
                            self.mc(McOut::Tmp(t), bw4, bh4, bx, by, 1 + pl, b.mv[i], Some(ri), filter_2d);
                        }
                    }
                    let (cw, ch) = (bw >> ss_hor, bh >> ss_ver);
                    match b.comp_type {
                        COMP_INTER_AVG => mc::avg(&mut self.cur.planes[1 + pl], uvdstoff, uvs, &tmp[0], &tmp[1], cw, ch, bd),
                        COMP_INTER_WEIGHTED_AVG => mc::w_avg(&mut self.cur.planes[1 + pl], uvdstoff, uvs, &tmp[0], &tmp[1], cw, ch, jnt_weight, bd),
                        _ => {
                            let m: &[u8] = if use_seg { &seg_mask } else { mask_src.unwrap() };
                            mc::mask(&mut self.cur.planes[1 + pl], uvdstoff, uvs, &tmp[ms], &tmp[1 - ms], cw, ch, m, bd);
                        }
                    }
                }
            }
        }

        let cw4 = (w4 + ss_hor) >> ss_hor;
        let ch4 = (h4 + ss_ver) >> ss_ver;
        if b.skip {
            self.set_coef_ctx(&Ctx::Luma, bx4, by4, 1 << b_dim[2], 1 << b_dim[3], 0x40);
            if has_chroma {
                let nw = 1 << ulog2(cbw4 as u32);
                let nh = 1 << ulog2(cbh4 as u32);
                self.set_coef_ctx(&Ctx::Chroma(0), cbx4, cby4, nw, nh, 0x40);
                self.set_coef_ctx(&Ctx::Chroma(1), cbx4, cby4, nw, nh, 0x40);
            }
            return Ok(());
        }

        let uvtx = TXFM_DIMENSIONS[b.uvtx as usize];
        let ytx = TXFM_DIMENSIONS[b.max_ytx as usize];
        let tx_split = [b.tx_split0 as u16, b.tx_split1];
        let mut init_y = 0;
        while init_y < bh4 {
            let mut init_x = 0;
            while init_x < bw4 {
                let mut y_off = (init_y != 0) as usize;
                let mut dsty = dst + ys * 4 * init_y as usize;
                let mut y = init_y;
                self.by += init_y;
                while y < h4.min(init_y + 16) {
                    let mut x_off = (init_x != 0) as usize;
                    let mut x = init_x;
                    self.bx += init_x;
                    while x < w4.min(init_x + 16) {
                        self.read_coef_tree(bs, b, b.max_ytx as usize, 0, tx_split, x_off, y_off, dsty + x as usize * 4);
                        self.bx += ytx.w as i32;
                        x += ytx.w as i32;
                        x_off += 1;
                    }
                    dsty += ys * 4 * ytx.h as usize;
                    self.bx -= x;
                    self.by += ytx.h as i32;
                    y += ytx.h as i32;
                    y_off += 1;
                }
                self.by -= y;

                if has_chroma {
                    for pl in 0..2 {
                        let mut uvdst = uvdstoff + (uvs * init_y as usize * 4 >> ss_ver);
                        let mut y = init_y >> ss_ver;
                        self.by += init_y;
                        while y < ch4.min((init_y + 16) >> ss_ver) {
                            let mut x = init_x >> ss_hor;
                            self.bx += init_x;
                            while x < cw4.min((init_x + 16) >> ss_hor) {
                                let txtp_in = self.txtp_map[(by4 + ((y as usize) << ss_ver)) * 32 + bx4 + ((x as usize) << ss_hor)];
                                let ax = cbx4 + x as usize;
                                let ly = cby4 + y as usize;
                                let (eob, txtp, cf_ctx) = self.decode_coefs(&Ctx::Chroma(pl), ax, ly, b.uvtx as usize, bs, b, false, 1 + pl, txtp_in);
                                let ctw = (uvtx.w as i32).min((self.bw - self.bx + ss_hor) >> ss_hor) as usize;
                                let cth = (uvtx.h as i32).min((self.bh - self.by + ss_ver) >> ss_ver) as usize;
                                self.set_coef_ctx(&Ctx::Chroma(pl), ax, ly, ctw, cth, cf_ctx);
                                if eob >= 0 {
                                    self.itx(1 + pl, uvdst + 4 * x as usize, b.uvtx as usize, txtp, eob);
                                }
                                self.bx += (uvtx.w as i32) << ss_hor;
                                x += uvtx.w as i32;
                            }
                            uvdst += uvs * 4 * uvtx.h as usize;
                            self.bx -= x << ss_hor;
                            self.by += (uvtx.h as i32) << ss_ver;
                            y += uvtx.h as i32;
                        }
                        self.by -= y << ss_ver;
                    }
                }
                init_x += 16;
            }
            init_y += 16;
        }
        Ok(())
    }
}

pub enum McOut<'a> {
    Cur { off: usize },
    Buf(&'a mut [u16], usize),
    Tmp(&'a mut [i16]),
}

fn ulog2(v: u32) -> i32 {
    31 - v.leading_zeros() as i32
}

fn sm_flag(b: &super::env::BlockContext, idx: usize) -> i32 {
    if b.intra[idx] == 0 {
        return 0;
    }
    let m = b.mode[idx] as usize;
    if m == SMOOTH_PRED || m == SMOOTH_H_PRED || m == SMOOTH_V_PRED {
        ANGLE_SMOOTH_EDGE_FLAG
    } else {
        0
    }
}

fn sm_uv_flag(b: &super::env::BlockContext, idx: usize) -> i32 {
    let m = b.uvmode[idx] as usize;
    if m == SMOOTH_PRED || m == SMOOTH_H_PRED || m == SMOOTH_V_PRED {
        ANGLE_SMOOTH_EDGE_FLAG
    } else {
        0
    }
}

#[allow(dead_code)]
const _TL: usize = TL;
