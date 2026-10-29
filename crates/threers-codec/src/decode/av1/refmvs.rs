//! Spatial and temporal motion-vector candidates (dav1d `src/refmvs.c`).
//!
//! dav1d keeps 35 rotating rows of 4x4 block info per superblock row; the
//! reads only ever reach the current superblock row and the odd rows -1,
//! -3, -5 above it, so a whole-frame array is equivalent and is used here.

use std::sync::Arc;

use super::env::{fix_mv_precision, get_gmv_2d, Mv};
use super::intra_edge::EDGE_I444_TOP_HAS_RIGHT;
use super::obu::{get_poc_diff, FrameHeader, SeqHeader, WM_TRANSLATION};
use super::tables::BLOCK_DIMENSIONS;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct TemporalBlock {
    pub mv: Mv,
    pub r: i8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RefmvsBlock {
    pub mv: [Mv; 2],
    /// `[0]` = 0: intra; `[1]` = -1: not compound.
    pub r: [i8; 2],
    pub bs: u8,
    /// 1 = globalmv+affine, 2 = newmv.
    pub mf: u8,
}

impl Default for RefmvsBlock {
    fn default() -> Self {
        RefmvsBlock { mv: [Mv::ZERO; 2], r: [0, -1], bs: 0, mf: 0 }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RefmvsCandidate {
    pub mv: [Mv; 2],
    pub weight: i32,
}

pub struct RefmvsFrame {
    pub iw4: i32,
    pub ih4: i32,
    pub iw8: i32,
    pub ih8: i32,
    pub use_ref_frame_mvs: bool,
    pub sign_bias: [u8; 7],
    pub mfmv_sign: [u8; 7],
    pub pocdiff: [i8; 7],
    pub mfmv_ref: [u8; 3],
    pub mfmv_ref2cur: [i32; 3],
    pub mfmv_ref2ref: [[i32; 7]; 3],
    pub n_mfmvs: usize,
    /// This frame's saved temporal MVs (8x8), `rp_stride` wide.
    pub rp: Vec<TemporalBlock>,
    pub rp_ref: [Option<Arc<Vec<TemporalBlock>>>; 7],
    /// Projected temporal MVs for the current frame, whole frame.
    pub rp_proj: Vec<TemporalBlock>,
    pub rp_stride: usize,
    /// 4x4 block info, whole frame (`r_stride` wide).
    pub r: Vec<RefmvsBlock>,
    pub r_stride: usize,
}

pub struct RefmvsTile {
    pub col_start: i32,
    pub col_end: i32,
    pub row_start: i32,
    pub row_end: i32,
}

impl RefmvsFrame {
    /// `dav1d_refmvs_init_frame`.
    pub fn new(seq: &SeqHeader, hdr: &FrameHeader, ref_poc: &[u32; 7], ref_ref_poc: &[[u32; 7]; 7], rp_ref: [Option<Arc<Vec<TemporalBlock>>>; 7]) -> Self {
        let rp_stride = (((hdr.width[0] + 127) & !127) >> 3) as usize;
        let iw8 = (hdr.width[0] + 7) >> 3;
        let ih8 = (hdr.height + 7) >> 3;
        let sb = 16 << seq.sb128 as i32;
        // Rows to the superblock-aligned height (blocks may overhang).
        let r_rows = (((ih8 * 2 + sb - 1) / sb) * sb) as usize;
        let mut rf = RefmvsFrame {
            iw4: iw8 << 1,
            ih4: ih8 << 1,
            iw8,
            ih8,
            use_ref_frame_mvs: false,
            sign_bias: [0; 7],
            mfmv_sign: [0; 7],
            pocdiff: [0; 7],
            mfmv_ref: [0; 3],
            mfmv_ref2cur: [0; 3],
            mfmv_ref2ref: [[0; 7]; 3],
            n_mfmvs: 0,
            rp: vec![TemporalBlock::default(); rp_stride * (r_rows / 2 + 1)],
            rp_ref,
            rp_proj: vec![TemporalBlock::default(); rp_stride * (r_rows / 2 + 1)],
            rp_stride,
            r: vec![RefmvsBlock::default(); rp_stride * 2 * r_rows],
            r_stride: rp_stride * 2,
        };
        let nb = seq.order_hint_n_bits;
        let poc = hdr.frame_offset as i32;
        for i in 0..7 {
            let d = get_poc_diff(nb, ref_poc[i] as i32, poc);
            rf.sign_bias[i] = (d > 0) as u8;
            rf.mfmv_sign[i] = (d < 0) as u8;
            rf.pocdiff[i] = get_poc_diff(nb, poc, ref_poc[i] as i32).clamp(-31, 31) as i8;
        }
        if hdr.use_ref_frame_mvs && nb != 0 {
            let mut total = 2;
            let rr = &rf.rp_ref;
            let mut refs = Vec::new();
            if rr[0].is_some() && ref_ref_poc[0][6] != ref_poc[3] {
                refs.push(0u8);
                total = 3;
            }
            if rr[4].is_some() && get_poc_diff(nb, ref_poc[4] as i32, poc) > 0 {
                refs.push(4);
            }
            if rr[5].is_some() && get_poc_diff(nb, ref_poc[5] as i32, poc) > 0 {
                refs.push(5);
            }
            if refs.len() < total && rr[6].is_some() && get_poc_diff(nb, ref_poc[6] as i32, poc) > 0 {
                refs.push(6);
            }
            if refs.len() < total && rr[1].is_some() {
                refs.push(1);
            }
            rf.n_mfmvs = refs.len();
            for (n, &m) in refs.iter().enumerate() {
                rf.mfmv_ref[n] = m;
                let rpoc = ref_poc[m as usize] as i32;
                let diff1 = get_poc_diff(nb, rpoc, poc);
                if diff1.abs() > 31 {
                    rf.mfmv_ref2cur[n] = i32::MIN;
                } else {
                    rf.mfmv_ref2cur[n] = if m < 4 { -diff1 } else { diff1 };
                    for k in 0..7 {
                        let rrpoc = ref_ref_poc[m as usize][k] as i32;
                        let diff2 = get_poc_diff(nb, rpoc, rrpoc);
                        rf.mfmv_ref2ref[n][k] = if diff2 as u32 > 31 { 0 } else { diff2 };
                    }
                }
            }
        }
        rf.use_ref_frame_mvs = rf.n_mfmvs > 0;
        rf
    }

    #[inline]
    pub fn blk(&self, y4: i32, x4: i32) -> &RefmvsBlock {
        &self.r[y4 as usize * self.r_stride + x4 as usize]
    }

    /// `splat_mv`.
    pub fn splat(&mut self, b: &RefmvsBlock, by4: i32, bx4: i32, bw4: i32, bh4: i32) {
        for y in 0..bh4 {
            let o = (by4 + y) as usize * self.r_stride + bx4 as usize;
            for v in &mut self.r[o..o + bw4 as usize] {
                *v = *b;
            }
        }
    }

    /// `load_tmvs_c` (rows of 8x8 blocks `row_start8..row_end8`).
    pub fn load_tmvs(&mut self, col_start8: i32, col_end8: i32, row_start8: i32, row_end8: i32) {
        let row_end8 = row_end8.min(self.ih8);
        let col_start8i = (col_start8 - 8).max(0);
        let col_end8i = (col_end8 + 8).min(self.iw8);
        let stride = self.rp_stride;
        for y in row_start8..row_end8 {
            for x in col_start8..col_end8 {
                self.rp_proj[y as usize * stride + x as usize].mv = Mv::INVALID;
            }
        }
        for n in 0..self.n_mfmvs {
            let ref2cur = self.mfmv_ref2cur[n];
            if ref2cur == i32::MIN {
                continue;
            }
            let r = self.mfmv_ref[n] as usize;
            let ref_sign = r as i32 - 4;
            let rp_ref = self.rp_ref[r].clone().unwrap();
            for y in row_start8..row_end8 {
                let y_sb_align = y & !7;
                let y_proj_start = y_sb_align.max(row_start8);
                let y_proj_end = (y_sb_align + 8).min(row_end8);
                let row = &rp_ref[y as usize * stride..];
                let mut x = col_start8i;
                while x < col_end8i {
                    let mut rb = x as usize;
                    let b_ref = row[rb].r;
                    if b_ref == 0 {
                        x += 1;
                        continue;
                    }
                    let ref2ref = self.mfmv_ref2ref[n][b_ref as usize - 1];
                    if ref2ref == 0 {
                        x += 1;
                        continue;
                    }
                    let b_mv = row[rb].mv;
                    let off = mv_projection(b_mv, ref2cur, ref2ref);
                    let sgn = |v: i32, s: i32| if s < 0 { -v } else { v };
                    let mut pos_x = x + sgn((off.x as i32).abs() >> 6, off.x as i32 ^ ref_sign);
                    let pos_y = y + sgn((off.y as i32).abs() >> 6, off.y as i32 ^ ref_sign);
                    if pos_y >= y_proj_start && pos_y < y_proj_end {
                        let pos = pos_y as usize * stride;
                        loop {
                            let x_sb_align = x & !7;
                            if pos_x >= (x_sb_align - 8).max(col_start8) && pos_x < (x_sb_align + 16).min(col_end8) {
                                let t = &mut self.rp_proj[pos + pos_x as usize];
                                t.mv = row[rb].mv;
                                t.r = ref2ref as i8;
                            }
                            x += 1;
                            if x >= col_end8i {
                                break;
                            }
                            rb += 1;
                            if row[rb].r != b_ref || row[rb].mv != b_mv {
                                break;
                            }
                            pos_x += 1;
                        }
                    } else {
                        loop {
                            x += 1;
                            if x >= col_end8i {
                                break;
                            }
                            rb += 1;
                            if row[rb].r != b_ref || row[rb].mv != b_mv {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `save_tmvs_c` for 8x8 rows `row_start8..row_end8`, clipped to the frame.
    pub fn save_tmvs(&mut self, col_start8: i32, col_end8: i32, row_start8: i32, row_end8: i32) {
        let row_end8 = row_end8.min(self.ih8);
        let col_end8 = col_end8.min(self.iw8);
        for y in row_start8..row_end8 {
            let b = (y as usize * 2 + 1) * self.r_stride;
            let rp = y as usize * self.rp_stride;
            let mut x = col_start8;
            while x < col_end8 {
                let c = self.r[b + x as usize * 2 + 1];
                let bw8 = (BLOCK_DIMENSIONS[c.bs as usize][0] as i32 + 1) >> 1;
                let ok = |k: usize| {
                    c.r[k] > 0 && self.mfmv_sign[c.r[k] as usize - 1] != 0 && ((c.mv[k].y as i32).abs() | (c.mv[k].x as i32).abs()) < 4096
                };
                let t = if ok(1) {
                    TemporalBlock { mv: c.mv[1], r: c.r[1] }
                } else if ok(0) {
                    TemporalBlock { mv: c.mv[0], r: c.r[0] }
                } else {
                    TemporalBlock { mv: Mv::ZERO, r: 0 }
                };
                for _ in 0..bw8 {
                    self.rp[rp + x as usize] = t;
                    x += 1;
                }
            }
        }
    }
}

fn mv_projection(mv: Mv, num: i32, den: i32) -> Mv {
    const DIV_MULT: [u16; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092, 1024, 963, 910, 862, 819, 780,
        744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    let frac = num * DIV_MULT[den as usize] as i32;
    let y = mv.y as i32 * frac;
    let x = mv.x as i32 * frac;
    Mv { y: ((y + 8192 + (y >> 31)) >> 14).clamp(-0x3fff, 0x3fff) as i16, x: ((x + 8192 + (x >> 31)) >> 14).clamp(-0x3fff, 0x3fff) as i16 }
}

struct Find<'a> {
    stack: [RefmvsCandidate; 9],
    cnt: usize,
    r: [i8; 2],
    gmv: [Mv; 2],
    have_newmv: bool,
    rf: &'a RefmvsFrame,
}

impl Find<'_> {
    fn add_spatial(&mut self, weight: i32, b: &RefmvsBlock, row_match: &mut bool, newmv: bool) {
        if b.mv[0] == Mv::INVALID {
            return;
        }
        let gmv = self.gmv;
        if self.r[1] == -1 {
            for n in 0..2 {
                if b.r[n] == self.r[0] {
                    let cand = if b.mf & 1 != 0 && gmv[0] != Mv::INVALID { gmv[0] } else { b.mv[n] };
                    *row_match = true;
                    if newmv {
                        self.have_newmv |= b.mf >> 1 != 0;
                    }
                    let last = self.cnt;
                    for m in 0..last {
                        if self.stack[m].mv[0] == cand {
                            self.stack[m].weight += weight;
                            return;
                        }
                    }
                    if last < 8 {
                        self.stack[last].mv[0] = cand;
                        self.stack[last].weight = weight;
                        self.cnt = last + 1;
                    }
                    return;
                }
            }
        } else if b.r == self.r {
            let cand = [
                if b.mf & 1 != 0 && gmv[0] != Mv::INVALID { gmv[0] } else { b.mv[0] },
                if b.mf & 1 != 0 && gmv[1] != Mv::INVALID { gmv[1] } else { b.mv[1] },
            ];
            *row_match = true;
            if newmv {
                self.have_newmv |= b.mf >> 1 != 0;
            }
            let last = self.cnt;
            for n in 0..last {
                if self.stack[n].mv == cand {
                    self.stack[n].weight += weight;
                    return;
                }
            }
            if last < 8 {
                self.stack[last].mv = cand;
                self.stack[last].weight = weight;
                self.cnt = last + 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_row(&mut self, y4: i32, x4: i32, bw4: i32, w4: i32, max_rows: i32, step: i32, m: &mut bool, newmv: bool) -> u32 {
        let rf = self.rf;
        let first = *rf.blk(y4, x4);
        let fdim = BLOCK_DIMENSIONS[first.bs as usize];
        let cand_bw4 = fdim[0] as i32;
        let mut len = step.max(bw4.min(cand_bw4));
        if bw4 <= cand_bw4 {
            let weight = if bw4 == 1 { 2 } else { 2.max((2 * max_rows).min(fdim[1] as i32)) };
            self.add_spatial(len * weight, &first, m, newmv);
            return (weight >> 1) as u32;
        }
        let mut x = 0;
        let mut cand = first;
        loop {
            self.add_spatial(len * 2, &cand, m, newmv);
            x += len;
            if x >= w4 {
                return 1;
            }
            cand = *rf.blk(y4, x4 + x);
            len = step.max(BLOCK_DIMENSIONS[cand.bs as usize][0] as i32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_col(&mut self, y4: i32, x4: i32, bh4: i32, h4: i32, max_cols: i32, step: i32, m: &mut bool, newmv: bool) -> u32 {
        let rf = self.rf;
        let first = *rf.blk(y4, x4);
        let fdim = BLOCK_DIMENSIONS[first.bs as usize];
        let cand_bh4 = fdim[1] as i32;
        let mut len = step.max(bh4.min(cand_bh4));
        if bh4 <= cand_bh4 {
            let weight = if bh4 == 1 { 2 } else { 2.max((2 * max_cols).min(fdim[0] as i32)) };
            self.add_spatial(len * weight, &first, m, newmv);
            return (weight >> 1) as u32;
        }
        let mut y = 0;
        let mut cand = first;
        loop {
            self.add_spatial(len * 2, &cand, m, newmv);
            y += len;
            if y >= h4 {
                return 1;
            }
            cand = *rf.blk(y4 + y, x4);
            len = step.max(BLOCK_DIMENSIONS[cand.bs as usize][1] as i32);
        }
    }

    fn add_temporal(&mut self, hdr: &FrameHeader, rb: &TemporalBlock, globalmv_ctx: Option<&mut i32>, tgmv: &[Mv; 2]) {
        if rb.mv == Mv::INVALID {
            return;
        }
        let rf = self.rf;
        let mut mv = mv_projection(rb.mv, rf.pocdiff[self.r[0] as usize - 1] as i32, rb.r as i32);
        fix_mv_precision(hdr, &mut mv);
        let last = self.cnt;
        if self.r[1] == -1 {
            if let Some(g) = globalmv_ctx {
                *g = (((mv.x as i32 - tgmv[0].x as i32).abs() | (mv.y as i32 - tgmv[0].y as i32).abs()) >= 16) as i32;
            }
            for n in 0..last {
                if self.stack[n].mv[0] == mv {
                    self.stack[n].weight += 2;
                    return;
                }
            }
            if last < 8 {
                self.stack[last].mv[0] = mv;
                self.stack[last].weight = 2;
                self.cnt = last + 1;
            }
        } else {
            let mut mv1 = mv_projection(rb.mv, rf.pocdiff[self.r[1] as usize - 1] as i32, rb.r as i32);
            fix_mv_precision(hdr, &mut mv1);
            let p = [mv, mv1];
            for n in 0..last {
                if self.stack[n].mv == p {
                    self.stack[n].weight += 2;
                    return;
                }
            }
            if last < 8 {
                self.stack[last].mv = p;
                self.stack[last].weight = 2;
                self.cnt = last + 1;
            }
        }
    }
}

fn neg(m: Mv) -> Mv {
    Mv { y: m.y.wrapping_neg(), x: m.x.wrapping_neg() }
}

/// `add_compound_extended_candidate`: `same` is stack[base..base+2],
/// `diff` stack[base+2..base+4].
fn add_compound_extended(st: &mut [RefmvsCandidate; 9], base: usize, cnt: &mut [usize; 4], c: &RefmvsBlock, s0: u8, s1: u8, r: [i8; 2], sign_bias: &[u8; 7]) {
    for n in 0..2 {
        let cr = c.r[n];
        if cr <= 0 {
            break;
        }
        let mut m = c.mv[n];
        let sb = sign_bias[cr as usize - 1];
        if cr == r[0] {
            if cnt[0] < 2 {
                st[base + cnt[0]].mv[0] = m;
                cnt[0] += 1;
            }
            if cnt[3] < 2 {
                if s1 ^ sb != 0 {
                    m = neg(m);
                }
                st[base + 2 + cnt[3]].mv[1] = m;
                cnt[3] += 1;
            }
        } else if cr == r[1] {
            if cnt[1] < 2 {
                st[base + cnt[1]].mv[1] = m;
                cnt[1] += 1;
            }
            if cnt[2] < 2 {
                if s0 ^ sb != 0 {
                    m = neg(m);
                }
                st[base + 2 + cnt[2]].mv[0] = m;
                cnt[2] += 1;
            }
        } else {
            let im = neg(m);
            if cnt[2] < 2 {
                st[base + 2 + cnt[2]].mv[0] = if s0 ^ sb != 0 { im } else { m };
                cnt[2] += 1;
            }
            if cnt[3] < 2 {
                st[base + 2 + cnt[3]].mv[1] = if s1 ^ sb != 0 { im } else { m };
                cnt[3] += 1;
            }
        }
    }
}

fn add_single_extended(st: &mut [RefmvsCandidate; 9], cnt: &mut usize, c: &RefmvsBlock, sign: u8, sign_bias: &[u8; 7]) {
    for n in 0..2 {
        let cr = c.r[n];
        if cr <= 0 {
            break;
        }
        let mut m = c.mv[n];
        if sign ^ sign_bias[cr as usize - 1] != 0 {
            m = neg(m);
        }
        let last = *cnt;
        if !(0..last).any(|k| st[k].mv[0] == m) {
            st[last].mv[0] = m;
            st[last].weight = 2;
            *cnt = last + 1;
        }
    }
}

/// `dav1d_refmvs_find`: (candidate stack, count, context).
#[allow(clippy::too_many_arguments)]
pub fn find(rf: &RefmvsFrame, hdr: &FrameHeader, rt: &RefmvsTile, r: [i8; 2], bs: usize, edge_flags: u8, by4: i32, bx4: i32) -> ([RefmvsCandidate; 8], usize, i32) {
    let b_dim = BLOCK_DIMENSIONS[bs];
    let bw4 = b_dim[0] as i32;
    let w4 = bw4.min(16).min(rt.col_end - bx4);
    let bh4 = b_dim[1] as i32;
    let h4 = bh4.min(16).min(rt.row_end - by4);
    let mut gmv = [Mv::INVALID; 2];
    let mut tgmv = [Mv::ZERO; 2];
    if r[0] > 0 {
        let g = &hdr.gmv[r[0] as usize - 1];
        tgmv[0] = get_gmv_2d(g, bx4, by4, bw4, bh4, hdr);
        gmv[0] = if g.typ > WM_TRANSLATION { tgmv[0] } else { Mv::INVALID };
    }
    if r[1] > 0 {
        let g = &hdr.gmv[r[1] as usize - 1];
        tgmv[1] = get_gmv_2d(g, bx4, by4, bw4, bh4, hdr);
        gmv[1] = if g.typ > WM_TRANSLATION { tgmv[1] } else { Mv::INVALID };
    }
    let mut f = Find { stack: [RefmvsCandidate::default(); 9], cnt: 0, r, gmv, have_newmv: false, rf };

    let (mut have_col_mvs, mut have_row_mvs) = (false, false);
    let mut max_rows = 0;
    let mut n_rows = !0u32;
    if by4 > rt.row_start {
        max_rows = ((by4 - rt.row_start + 1) >> 1).min(2 + (bh4 > 1) as i32);
        n_rows = f.scan_row(by4 - 1, bx4, bw4, w4, max_rows, if bw4 >= 16 { 4 } else { 1 }, &mut have_row_mvs, true);
    }
    let mut max_cols = 0;
    let mut n_cols = !0u32;
    if bx4 > rt.col_start {
        max_cols = ((bx4 - rt.col_start + 1) >> 1).min(2 + (bw4 > 1) as i32);
        n_cols = f.scan_col(by4, bx4 - 1, bh4, h4, max_cols, if bh4 >= 16 { 4 } else { 1 }, &mut have_col_mvs, true);
    }
    if n_rows != !0 && edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0 && bw4.max(bh4) <= 16 && bw4 + bx4 < rt.col_end {
        let b = *rf.blk(by4 - 1, bx4 + bw4);
        f.add_spatial(4, &b, &mut have_row_mvs, true);
    }
    let nearest_match = have_col_mvs as i32 + have_row_mvs as i32;
    let nearest_cnt = f.cnt;
    for c in f.stack.iter_mut().take(nearest_cnt) {
        c.weight += 640;
    }

    // Temporal.
    let mut globalmv_ctx = hdr.use_ref_frame_mvs as i32;
    if rf.use_ref_frame_mvs {
        let stride = rf.rp_stride as i32;
        let (by8, bx8) = (by4 >> 1, bx4 >> 1);
        let rbi = by8 * stride + bx8;
        let step_h = if bw4 >= 16 { 2 } else { 1 };
        let step_v = if bh4 >= 16 { 2 } else { 1 };
        let w8 = ((w4 + 1) >> 1).min(8);
        let h8 = ((h4 + 1) >> 1).min(8);
        let mut y = 0;
        while y < h8 {
            let mut x = 0;
            while x < w8 {
                let rb = rf.rp_proj[(rbi + y * stride + x) as usize];
                if x == 0 && y == 0 {
                    f.add_temporal(hdr, &rb, Some(&mut globalmv_ctx), &tgmv);
                } else {
                    f.add_temporal(hdr, &rb, None, &tgmv);
                }
                x += step_h;
            }
            y += step_v;
        }
        if bw4.min(bh4) >= 2 && bw4.max(bh4) < 16 {
            let (bh8, bw8) = (bh4 >> 1, bw4 >> 1);
            let rb = rbi + bh8 * stride;
            let has_bottom = by8 + bh8 < (rt.row_end >> 1).min((by8 & !7) + 8);
            if has_bottom && bx8 > (rt.col_start >> 1).max(bx8 & !7) {
                let t = rf.rp_proj[(rb - 1) as usize];
                f.add_temporal(hdr, &t, None, &tgmv);
            }
            if bx8 + bw8 < (rt.col_end >> 1).min((bx8 & !7) + 8) {
                if has_bottom {
                    let t = rf.rp_proj[(rb + bw8) as usize];
                    f.add_temporal(hdr, &t, None, &tgmv);
                }
                if by8 + bh8 - 1 < (rt.row_end >> 1).min((by8 & !7) + 8) {
                    let t = rf.rp_proj[(rb + bw8 - stride) as usize];
                    f.add_temporal(hdr, &t, None, &tgmv);
                }
            }
        }
    }

    // Top/left.
    let mut dummy = false;
    if n_rows != !0 && n_cols != !0 {
        let b = *rf.blk(by4 - 1, bx4 - 1);
        f.add_spatial(4, &b, &mut have_row_mvs, false);
    }
    let _ = &mut dummy;
    // Secondary top and left edges (8x8 resolution).
    for n in 2..=3u32 {
        if n > n_rows && n as i32 <= max_rows {
            let y4 = by4 - (by4 & 31) + (((by4 & 31) - 2 * n as i32 + 1) | 1);
            n_rows = n_rows.wrapping_add(f.scan_row(y4, bx4 | 1, bw4, w4, 1 + max_rows - n as i32, if bw4 >= 16 { 4 } else { 2 }, &mut have_row_mvs, false));
        }
        if n > n_cols && n as i32 <= max_cols {
            let y4 = by4 - (by4 & 31) + ((by4 & 31) | 1);
            n_cols = n_cols.wrapping_add(f.scan_col(y4, (bx4 - n as i32 * 2 + 1) | 1, bh4, h4, 1 + max_cols - n as i32, if bh4 >= 16 { 4 } else { 2 }, &mut have_col_mvs, false));
        }
    }
    let ref_match_count = have_col_mvs as i32 + have_row_mvs as i32;
    let have_newmv = f.have_newmv as i32;
    let (refmv_ctx, newmv_ctx) = match nearest_match {
        0 => (ref_match_count.min(2), (ref_match_count > 0) as i32),
        1 => ((ref_match_count * 3).min(4), 3 - have_newmv),
        _ => (5, 5 - have_newmv),
    };

    // Sort (nearest, then secondary).
    let mut len = nearest_cnt;
    while len > 0 {
        let mut last = 0;
        for n in 1..len {
            if f.stack[n - 1].weight < f.stack[n].weight {
                f.stack.swap(n - 1, n);
                last = n;
            }
        }
        len = last;
    }
    len = f.cnt;
    while len > nearest_cnt {
        let mut last = nearest_cnt;
        for n in nearest_cnt + 1..len {
            if f.stack[n - 1].weight < f.stack[n].weight {
                f.stack.swap(n - 1, n);
                last = n;
            }
        }
        len = last;
    }

    let mut cnt = f.cnt;
    let mut st = f.stack;
    let clamp_bounds = (
        -(bx4 + bw4 + 4) * 4 * 8,
        (rf.iw4 - bx4 + 4) * 4 * 8,
        -(by4 + bh4 + 4) * 4 * 8,
        (rf.ih4 - by4 + 4) * 4 * 8,
    );
    let clampmv = |m: &mut Mv| {
        let (l, rr, t, b) = clamp_bounds;
        m.x = (m.x as i32).clamp(l, rr) as i16;
        m.y = (m.y as i32).clamp(t, b) as i16;
    };
    let ctx = if r[1] > 0 {
        if cnt < 2 {
            let s0 = rf.sign_bias[r[0] as usize - 1];
            let s1 = rf.sign_bias[r[1] as usize - 1];
            let sz4 = w4.min(h4);
            let base = cnt;
            let mut sc = [0usize; 4];
            if n_rows != !0 {
                let mut x = 0;
                while x < sz4 {
                    let c = *rf.blk(by4 - 1, bx4 + x);
                    add_compound_extended(&mut st, base, &mut sc, &c, s0, s1, r, &rf.sign_bias);
                    x += BLOCK_DIMENSIONS[c.bs as usize][0] as i32;
                }
            }
            if n_cols != !0 {
                let mut y = 0;
                while y < sz4 {
                    let c = *rf.blk(by4 + y, bx4 - 1);
                    add_compound_extended(&mut st, base, &mut sc, &c, s0, s1, r, &rf.sign_bias);
                    y += BLOCK_DIMENSIONS[c.bs as usize][1] as i32;
                }
            }
            for n in 0..2 {
                let mut m = sc[n];
                if m >= 2 {
                    continue;
                }
                let l = sc[2 + n];
                if l != 0 {
                    st[base + m].mv[n] = st[base + 2].mv[n];
                    m += 1;
                    if m == 2 {
                        continue;
                    }
                    if l == 2 {
                        st[base + 1].mv[n] = st[base + 3].mv[n];
                        continue;
                    }
                }
                loop {
                    st[base + m].mv[n] = tgmv[n];
                    m += 1;
                    if m >= 2 {
                        break;
                    }
                }
            }
            let mut n = cnt;
            if n == 1 && st[0].mv == st[base].mv {
                st[1].mv = st[2].mv;
            }
            loop {
                st[n].weight = 2;
                n += 1;
                if n >= 2 {
                    break;
                }
            }
            cnt = 2;
        }
        for c in st.iter_mut().take(cnt) {
            clampmv(&mut c.mv[0]);
            clampmv(&mut c.mv[1]);
        }
        match refmv_ctx >> 1 {
            0 => newmv_ctx.min(1),
            1 => 1 + newmv_ctx.min(3),
            _ => (3 + newmv_ctx).clamp(4, 7),
        }
    } else {
        if cnt < 2 && r[0] > 0 {
            let sign = rf.sign_bias[r[0] as usize - 1];
            let sz4 = w4.min(h4);
            if n_rows != !0 {
                let mut x = 0;
                while x < sz4 && cnt < 2 {
                    let c = *rf.blk(by4 - 1, bx4 + x);
                    add_single_extended(&mut st, &mut cnt, &c, sign, &rf.sign_bias);
                    x += BLOCK_DIMENSIONS[c.bs as usize][0] as i32;
                }
            }
            if n_cols != !0 {
                let mut y = 0;
                while y < sz4 && cnt < 2 {
                    let c = *rf.blk(by4 + y, bx4 - 1);
                    add_single_extended(&mut st, &mut cnt, &c, sign, &rf.sign_bias);
                    y += BLOCK_DIMENSIONS[c.bs as usize][1] as i32;
                }
            }
        }
        for c in st.iter_mut().take(cnt) {
            clampmv(&mut c.mv[0]);
        }
        for c in st.iter_mut().take(2).skip(cnt) {
            c.mv[0] = tgmv[0];
        }
        (refmv_ctx << 4) | (globalmv_ctx << 3) | newmv_ctx
    };
    let mut out = [RefmvsCandidate::default(); 8];
    out.copy_from_slice(&st[..8]);
    (out, cnt, ctx)
}
