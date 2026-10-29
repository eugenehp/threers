//! Block syntax: partitions, modes, motion vectors and the per-tile and
//! per-frame loops (dav1d `src/decode.c`, single-threaded path, pass 0).

use std::sync::Arc;

use super::cdf::CdfContext;
use super::env::*;
use super::intra_edge::{self, EdgeNode, EDGE_ALL_LEFT_HAS_BOTTOM, EDGE_ALL_TOP_HAS_RIGHT, EDGE_ALL_TR_AND_BL, EDGE_I444_TOP_HAS_RIGHT};
use super::levels::*;
use super::lf_mask::{self, Av1Filter, Av1FilterLut, Av1Restoration, LfEdges, LfLevels, RestorationUnit};
use super::msac::{ulog2, Msac};
use super::obu::*;
use super::picture::Picture;
use super::refmvs::{self, RefmvsBlock, RefmvsCandidate, RefmvsFrame, RefmvsTile, TemporalBlock};
use super::tables::*;
use super::warpmv;

pub type DResult = Result<(), String>;

macro_rules! ba {
    ($s:ident, $($p:tt)+) => {{
        let ts = &mut $s.ts[$s.tsi];
        ts.msac.bool_adapt(&mut ts.cdf.$($p)+)
    }};
}
macro_rules! sa {
    ($s:ident, $n:expr, $($p:tt)+) => {{
        let ts = &mut $s.ts[$s.tsi];
        ts.msac.symbol_adapt(&mut ts.cdf.$($p)+, $n) as usize
    }};
}
pub(crate) use ba;
pub(crate) use sa;

#[derive(Clone, Copy, Default, Debug)]
pub struct Av1Block {
    pub bl: u8,
    pub bs: u8,
    pub bp: u8,
    pub intra: bool,
    pub seg_id: u8,
    pub skip_mode: bool,
    pub skip: bool,
    pub uvtx: u8,
    // intra
    pub y_mode: u8,
    pub uv_mode: u8,
    pub tx: u8,
    pub pal_sz: [u8; 2],
    pub y_angle: i8,
    pub uv_angle: i8,
    pub cfl_alpha: [i8; 2],
    // inter
    pub mv: [Mv; 2],
    pub wedge_idx: u8,
    pub mask_sign: u8,
    pub interintra_mode: u8,
    pub comp_type: u8,
    pub inter_mode: u8,
    pub motion_mode: u8,
    pub drl_idx: u8,
    pub r#ref: [i8; 2],
    pub max_ytx: u8,
    pub filter2d: u8,
    pub interintra_type: u8,
    pub tx_split0: u8,
    pub tx_split1: u16,
}

#[derive(Clone, Copy, Default)]
pub struct Svc {
    pub scale: i32,
    pub step: i32,
}

pub struct TilePos {
    pub col_start: i32,
    pub col_end: i32,
    pub row_start: i32,
    pub row_end: i32,
    pub col: usize,
    pub row: usize,
}

pub struct TileState {
    pub cdf: Box<CdfContext>,
    pub msac: Msac,
    pub tiling: TilePos,
    pub dqmem: [[[u16; 2]; 3]; 8],
    pub dq_frame: bool,
    pub last_qidx: i32,
    pub last_delta_lf: [i8; 4],
    pub lflvlmem: LfLevels,
    pub lflvl_frame: bool,
    /// `lr_ref[p]`: (lr_mask index, unit index).
    pub lr_ref: [Option<(usize, usize)>; 3],
}

/// A reference frame as the current frame sees it.
pub struct RefFrame {
    pub pic: Arc<Picture>,
    pub hdr: Arc<FrameHeader>,
}

pub struct LfState {
    pub level: Vec<[u8; 4]>,
    pub mask: Vec<Av1Filter>,
    pub lr_mask: Vec<Av1Restoration>,
    pub lim_lut: Av1FilterLut,
    pub lvl: LfLevels,
    pub tx_lpf_right_edge: [Vec<u8>; 2],
    pub start_of_tile_row: Vec<u8>,
    pub restore_planes: u32,
}

/// The frame decoder: dav1d's frame, tile and task contexts in one.
pub struct Fd {
    pub seq: Arc<SeqHeader>,
    pub hdr: Arc<FrameHeader>,
    pub layout: Layout,
    pub ss_hor: i32,
    pub ss_ver: i32,
    pub bpc: u32,
    pub bitdepth_max: i32,
    pub cur: Picture,
    /// Superres output (when `width[0] != width[1]`).
    pub sr_cur: Option<Picture>,
    pub refp: [Option<RefFrame>; 7],
    pub svc: [[Svc; 2]; 7],
    pub gmv_warp_allowed: [bool; 7],
    pub w4: i32,
    pub h4: i32,
    pub bw: i32,
    pub bh: i32,
    pub sb128w: i32,
    pub sb_shift: i32,
    pub sb_step: i32,
    pub sbh: i32,
    pub sr_sb128w: i32,
    pub b4_stride: i32,
    pub resize_step: [i32; 2],
    pub resize_start: [i32; 2],
    pub dq: [[[u16; 2]; 3]; 8],
    /// `qm[tx][plane]`: (qm level, chroma) or None.
    pub qm: [[Option<(usize, usize)>; 3]; N_RECT_TX_SIZES],
    pub a: Vec<BlockContext>,
    pub ai: usize,
    pub rf: Option<RefmvsFrame>,
    pub jnt_weights: [[u8; 7]; 7],
    pub lf: LfState,
    pub ipred_edge: [Vec<u16>; 3],
    pub cur_segmap: Option<Vec<u8>>,
    pub prev_segmap: Option<Arc<Vec<u8>>>,
    pub ts: Vec<TileState>,
    pub tsi: usize,
    // task
    pub bx: i32,
    pub by: i32,
    pub l: BlockContext,
    pub rt: RefmvsTile,
    pub lf_mask: usize,
    pub cdef_idx: usize,
    pub warpmv: WarpedMotionParams,
    pub tl_4x4_filter: u8,
    pub pal: [[u16; 8]; 3],
    pub al_pal: [[[[u16; 8]; 3]; 32]; 2],
    pub pal_sz_uv: [[u8; 32]; 2],
    pub pal_idx_y: Vec<u8>,
    pub pal_idx_uv: Vec<u8>,
    pub txtp_map: [u8; 32 * 32],
    pub cf: Vec<i32>,
    pub levels: Vec<u8>,
    pub refpoc: [u32; 7],
    /// Edge-padded reference window for `mc`, reused between blocks.
    pub mc_buf: Vec<u16>,
}

#[inline]
fn set<T: Copy>(a: &mut [T], off: usize, n: usize, v: T) {
    for x in &mut a[off..off + n] {
        *x = v;
    }
}

pub fn init_quant_tables(seq: &SeqHeader, hdr: &FrameHeader, qidx: i32, dq: &mut [[[u16; 2]; 3]; 8]) {
    let n = if hdr.segmentation.enabled { 8 } else { 1 };
    let hbd = seq.hbd as usize;
    let q = &hdr.quant;
    for (i, d) in dq.iter_mut().enumerate().take(n) {
        let yac = if hdr.segmentation.enabled { (qidx + hdr.segmentation.seg_data.d[i].delta_q as i32).clamp(0, 255) } else { qidx };
        let c = |dl: i8| (yac + dl as i32).clamp(0, 255) as usize;
        d[0][0] = DQ_TBL[hbd][c(q.ydc_delta)][0];
        d[0][1] = DQ_TBL[hbd][yac as usize][1];
        d[1][0] = DQ_TBL[hbd][c(q.udc_delta)][0];
        d[1][1] = DQ_TBL[hbd][c(q.uac_delta)][1];
        d[2][0] = DQ_TBL[hbd][c(q.vdc_delta)][0];
        d[2][1] = DQ_TBL[hbd][c(q.vac_delta)][1];
    }
}

fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        max - (diff + 1)
    }
}

fn findoddzero(buf: &[u8], len: usize) -> bool {
    (0..len).any(|n| buf[n * 2] == 0)
}

impl Fd {
    #[inline]
    pub fn msac(&mut self) -> &mut Msac {
        &mut self.ts[self.tsi].msac
    }

    fn read_mv_component_diff(&mut self, comp: usize, mv_prec: i32) -> i32 {
        let sign = ba!(self, mv.comp[comp].sign);
        let cl = sa!(self, 10, mv.comp[comp].classes);
        let (up, mut fp, mut hp);
        fp = 3;
        hp = 1;
        if cl == 0 {
            up = ba!(self, mv.comp[comp].class0) as i32;
            if mv_prec >= 0 {
                fp = sa!(self, 3, mv.comp[comp].class0_fp[up as usize]) as i32;
                if mv_prec > 0 {
                    hp = ba!(self, mv.comp[comp].class0_hp) as i32;
                }
            }
        } else {
            let mut u = 1 << cl;
            for n in 0..cl {
                u |= (ba!(self, mv.comp[comp].classN[n]) as i32) << n;
            }
            up = u;
            if mv_prec >= 0 {
                fp = sa!(self, 3, mv.comp[comp].classN_fp) as i32;
                if mv_prec > 0 {
                    hp = ba!(self, mv.comp[comp].classN_hp) as i32;
                }
            }
        }
        let diff = ((up << 3) | (fp << 1) | hp) + 1;
        if sign {
            -diff
        } else {
            diff
        }
    }

    fn read_mv_residual(&mut self, mv: &mut Mv, mv_prec: i32) {
        let joint = sa!(self, N_MV_JOINTS - 1, mv.joint);
        if joint & MV_JOINT_V != 0 {
            mv.y = (mv.y as i32 + self.read_mv_component_diff(0, mv_prec)) as i16;
        }
        if joint & MV_JOINT_H != 0 {
            mv.x = (mv.x as i32 + self.read_mv_component_diff(1, mv_prec)) as i16;
        }
    }

    fn read_tx_tree(&mut self, from: usize, depth: usize, masks: &mut [u16; 2], x_off: usize, y_off: usize) {
        let bx4 = (self.bx & 31) as usize;
        let by4 = (self.by & 31) as usize;
        let t_dim = TXFM_DIMENSIONS[from];
        let (txw, txh) = (t_dim.lw, t_dim.lh);
        let is_split = if depth < 2 && from > TX_4X4 {
            let cat = 2 * (TX_64X64 - t_dim.max as usize) - depth;
            let a = ((self.a[self.ai].tx[bx4] as i32) < txw as i32) as usize;
            let l = ((self.l.tx[by4] as i32) < txh as i32) as usize;
            let s = ba!(self, m.txpart[cat][a + l]);
            if s {
                masks[depth] |= 1 << (y_off * 4 + x_off);
            }
            s
        } else {
            false
        };
        if is_split && t_dim.max as usize > TX_8X8 {
            let sub = t_dim.sub as usize;
            let st = TXFM_DIMENSIONS[sub];
            let (txsw, txsh) = (st.w as i32, st.h as i32);
            self.read_tx_tree(sub, depth + 1, masks, x_off * 2, y_off * 2);
            self.bx += txsw;
            if txw >= txh && self.bx < self.bw {
                self.read_tx_tree(sub, depth + 1, masks, x_off * 2 + 1, y_off * 2);
            }
            self.bx -= txsw;
            self.by += txsh;
            if txh >= txw && self.by < self.bh {
                self.read_tx_tree(sub, depth + 1, masks, x_off * 2, y_off * 2 + 1);
                self.bx += txsw;
                if txw >= txh && self.bx < self.bw {
                    self.read_tx_tree(sub, depth + 1, masks, x_off * 2 + 1, y_off * 2 + 1);
                }
                self.bx -= txsw;
            }
            self.by -= txsh;
        } else {
            let ai = self.ai;
            set(&mut self.a[ai].tx, bx4, 1 << t_dim.lw, if is_split { TX_4X4 as i8 } else { txw as i8 });
            set(&mut self.l.tx, by4, 1 << t_dim.lh, if is_split { TX_4X4 as i8 } else { txh as i8 });
        }
    }

    fn find_matching_ref(&self, intra_edge_flags: u8, bw4: i32, bh4: i32, w4: i32, h4: i32, have_left: bool, have_top: bool, r: i8, masks: &mut [u64; 2]) {
        let rf = self.rf.as_ref().unwrap();
        let (bx, by) = (self.bx, self.by);
        let mut count = 0;
        let mut have_topleft = have_top && have_left;
        let mut have_topright =
            bw4.max(bh4) < 32 && have_top && bx + bw4 < self.ts[self.tsi].tiling.col_end && intra_edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0;
        let matches = |b: &RefmvsBlock| b.r[0] == r + 1 && b.r[1] == -1;
        let bsd = |b: &RefmvsBlock| BLOCK_DIMENSIONS[b.bs as usize];
        if have_top {
            let mut x2 = bx;
            let r2 = rf.blk(by - 1, x2);
            if matches(r2) {
                masks[0] |= 1;
                count = 1;
            }
            let mut aw4 = bsd(r2)[0] as i32;
            if aw4 >= bw4 {
                let off = bx & (aw4 - 1);
                if off != 0 {
                    have_topleft = false;
                }
                if aw4 - off > bw4 {
                    have_topright = false;
                }
            } else {
                let mut mask = 1u32 << aw4;
                let mut x = aw4;
                while x < w4 {
                    x2 += aw4;
                    let r2 = rf.blk(by - 1, x2);
                    if matches(r2) {
                        masks[0] |= mask as u64;
                        count += 1;
                        if count >= 8 {
                            return;
                        }
                    }
                    aw4 = bsd(r2)[0] as i32;
                    mask <<= aw4;
                    x += aw4;
                }
            }
        }
        if have_left {
            let mut y2 = by;
            let r2 = rf.blk(y2, bx - 1);
            if matches(r2) {
                masks[1] |= 1;
                count += 1;
                if count >= 8 {
                    return;
                }
            }
            let mut lh4 = bsd(r2)[1] as i32;
            if lh4 >= bh4 {
                if by & (lh4 - 1) != 0 {
                    have_topleft = false;
                }
            } else {
                let mut mask = 1u32 << lh4;
                let mut y = lh4;
                while y < h4 {
                    y2 += lh4;
                    let r2 = rf.blk(y2, bx - 1);
                    if matches(r2) {
                        masks[1] |= mask as u64;
                        count += 1;
                        if count >= 8 {
                            return;
                        }
                    }
                    lh4 = bsd(r2)[1] as i32;
                    mask <<= lh4;
                    y += lh4;
                }
            }
        }
        if have_topleft && matches(rf.blk(by - 1, bx - 1)) {
            masks[1] |= 1u64 << 32;
            count += 1;
            if count >= 8 {
                return;
            }
        }
        if have_topright && matches(rf.blk(by - 1, bx + bw4)) {
            masks[0] |= 1u64 << 32;
        }
    }

    fn derive_warpmv(&mut self, bw4: i32, bh4: i32, masks: &[u64; 2], mv: Mv) {
        let rf = self.rf.as_ref().unwrap();
        let (bx, by) = (self.bx, self.by);
        let mut pts = [[[0i32; 2]; 2]; 8];
        let mut np = 0usize;
        let bsd = |b: &RefmvsBlock| BLOCK_DIMENSIONS[b.bs as usize];
        let add = |pts: &mut [[[i32; 2]; 2]; 8], np: &mut usize, dx: i32, dy: i32, sx: i32, sy: i32, b: &RefmvsBlock| {
            let d = bsd(b);
            pts[*np][0][0] = 16 * (2 * dx + sx * d[0] as i32) - 8;
            pts[*np][0][1] = 16 * (2 * dy + sy * d[1] as i32) - 8;
            pts[*np][1][0] = pts[*np][0][0] + b.mv[0].x as i32;
            pts[*np][1][1] = pts[*np][0][1] + b.mv[0].y as i32;
            *np += 1;
        };
        if masks[0] as u32 == 1 && masks[1] >> 32 == 0 {
            let b = rf.blk(by - 1, bx);
            let off = bx & (bsd(b)[0] as i32 - 1);
            add(&mut pts, &mut np, -off, 0, 1, -1, b);
        } else {
            let mut off = 0u32;
            let mut xmask = masks[0] as u32;
            while np < 8 && xmask != 0 {
                let tz = xmask.trailing_zeros();
                off += tz;
                xmask >>= tz;
                add(&mut pts, &mut np, off as i32, 0, 1, -1, rf.blk(by - 1, bx + off as i32));
                xmask &= !1;
            }
        }
        if np < 8 && masks[1] == 1 {
            let b = rf.blk(by, bx - 1);
            let off = by & (bsd(b)[1] as i32 - 1);
            add(&mut pts, &mut np, 0, -off, -1, 1, rf.blk(by - off, bx - 1));
        } else {
            let mut off = 0u32;
            let mut ymask = masks[1] as u32;
            while np < 8 && ymask != 0 {
                let tz = ymask.trailing_zeros();
                off += tz;
                ymask >>= tz;
                add(&mut pts, &mut np, 0, off as i32, -1, 1, rf.blk(by + off as i32, bx - 1));
                ymask &= !1;
            }
        }
        if np < 8 && masks[1] >> 32 != 0 {
            add(&mut pts, &mut np, 0, 0, -1, -1, rf.blk(by - 1, bx - 1));
        }
        if np < 8 && masks[0] >> 32 != 0 {
            add(&mut pts, &mut np, bw4, 0, 1, -1, rf.blk(by - 1, bx + bw4));
        }
        let mut mvd = [0i32; 8];
        let mut ret = 0usize;
        let thresh = 4 * bw4.max(bh4).clamp(4, 28);
        for i in 0..np {
            mvd[i] = (pts[i][1][0] - pts[i][0][0] - mv.x as i32).abs() + (pts[i][1][1] - pts[i][0][1] - mv.y as i32).abs();
            if mvd[i] > thresh {
                mvd[i] = -1;
            } else {
                ret += 1;
            }
        }
        if ret == 0 {
            ret = 1;
        } else {
            let (mut i, mut j) = (0usize, np as isize - 1);
            for _ in 0..np - ret {
                while mvd[i] != -1 {
                    i += 1;
                }
                while mvd[j as usize] == -1 {
                    j -= 1;
                }
                if i as isize > j {
                    break;
                }
                mvd[i] = mvd[j as usize];
                pts[i] = pts[j as usize];
                i += 1;
                j -= 1;
            }
        }
        let mut wmp = self.warpmv;
        if !warpmv::find_affine_int(&pts, ret, bw4, bh4, mv, &mut wmp, bx, by) && !warpmv::get_shear_params(&mut wmp) {
            wmp.typ = WM_AFFINE;
        } else {
            wmp.typ = WM_IDENTITY;
        }
        self.warpmv = wmp;
    }

    fn read_vartx_tree(&mut self, b: &mut Av1Block, bs: usize, bx4: usize, by4: usize) {
        let b_dim = BLOCK_DIMENSIONS[bs];
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let mut tx_split = [0u16; 2];
        b.max_ytx = MAX_TXFM_SIZE_FOR_BS[bs][0];
        let lossless = self.hdr.segmentation.lossless[b.seg_id as usize];
        let switchable = self.hdr.txfm_mode == TX_SWITCHABLE;
        let ai = self.ai;
        if !b.skip && (lossless || b.max_ytx as usize == TX_4X4) {
            b.max_ytx = TX_4X4 as u8;
            b.uvtx = TX_4X4 as u8;
            if switchable {
                set(&mut self.a[ai].tx, bx4, 1 << b_dim[2], TX_4X4 as i8);
                set(&mut self.l.tx, by4, 1 << b_dim[3], TX_4X4 as i8);
            }
        } else if !switchable || b.skip {
            if switchable {
                set(&mut self.a[ai].tx, bx4, 1 << b_dim[2], b_dim[2] as i8);
                set(&mut self.l.tx, by4, 1 << b_dim[3], b_dim[3] as i8);
            }
            b.uvtx = MAX_TXFM_SIZE_FOR_BS[bs][self.layout as usize];
        } else {
            let ytx = TXFM_DIMENSIONS[b.max_ytx as usize];
            let (mut y, mut y_off) = (0, 0);
            while y < bh4 {
                let (mut x, mut x_off) = (0, 0);
                while x < bw4 {
                    self.read_tx_tree(b.max_ytx as usize, 0, &mut tx_split, x_off, y_off);
                    self.bx += ytx.w as i32;
                    x += ytx.w as i32;
                    x_off += 1;
                }
                self.bx -= x;
                self.by += ytx.h as i32;
                y += ytx.h as i32;
                y_off += 1;
            }
            self.by -= y;
            b.uvtx = MAX_TXFM_SIZE_FOR_BS[bs][self.layout as usize];
        }
        b.tx_split0 = tx_split[0] as u8;
        b.tx_split1 = tx_split[1];
    }

    fn get_prev_frame_segid(&self, by: i32, bx: i32, w4: i32, h4: i32) -> u8 {
        let map = self.prev_segmap.as_ref().unwrap();
        let stride = self.b4_stride as usize;
        let mut seg_id = 8u8;
        let mut o = by as usize * stride + bx as usize;
        let mut h = h4;
        loop {
            for x in 0..w4 as usize {
                seg_id = seg_id.min(map[o + x]);
            }
            o += stride;
            h -= 1;
            if !(h > 0 && seg_id != 0) {
                break;
            }
        }
        seg_id
    }

    fn splat(&mut self, tmpl: RefmvsBlock, bw4: i32, bh4: i32) {
        let (by, bx) = (self.by, self.bx);
        self.rf.as_mut().unwrap().splat(&tmpl, by, bx, bw4, bh4);
    }

    fn order_palette(pal_idx: &[u8], stride: usize, i: usize, first: usize, last: usize, order: &mut [[u8; 8]; 64], ctx: &mut [u8; 64]) {
        let mut have_top = i > first;
        let mut p = first + (i - first) * stride;
        let mut n = 0;
        let mut j = first as isize;
        while j >= last as isize {
            let have_left = j > 0;
            let mut mask = 0u32;
            let mut o_idx = 0;
            let add = |v: u8, o: &mut [u8; 8], o_idx: &mut usize, mask: &mut u32| {
                o[*o_idx] = v;
                *o_idx += 1;
                *mask |= 1 << v;
            };
            if !have_left {
                ctx[n] = 0;
                add(pal_idx[p - stride], &mut order[n], &mut o_idx, &mut mask);
            } else if !have_top {
                ctx[n] = 0;
                add(pal_idx[p - 1], &mut order[n], &mut o_idx, &mut mask);
            } else {
                let l = pal_idx[p - 1];
                let t = pal_idx[p - stride];
                let tl = pal_idx[p - stride - 1];
                let same_t_l = t == l;
                let same_t_tl = t == tl;
                let same_l_tl = l == tl;
                let same_all = same_t_l & same_t_tl & same_l_tl;
                if same_all {
                    ctx[n] = 4;
                    add(t, &mut order[n], &mut o_idx, &mut mask);
                } else if same_t_l {
                    ctx[n] = 3;
                    add(t, &mut order[n], &mut o_idx, &mut mask);
                    add(tl, &mut order[n], &mut o_idx, &mut mask);
                } else if same_t_tl | same_l_tl {
                    ctx[n] = 2;
                    add(tl, &mut order[n], &mut o_idx, &mut mask);
                    add(if same_t_tl { l } else { t }, &mut order[n], &mut o_idx, &mut mask);
                } else {
                    ctx[n] = 1;
                    add(t.min(l), &mut order[n], &mut o_idx, &mut mask);
                    add(t.max(l), &mut order[n], &mut o_idx, &mut mask);
                    add(tl, &mut order[n], &mut o_idx, &mut mask);
                }
            }
            for bit in 0..8u8 {
                if mask & (1 << bit) == 0 {
                    order[n][o_idx] = bit;
                    o_idx += 1;
                }
            }
            have_top = true;
            j -= 1;
            n += 1;
            p += stride - 1;
        }
    }

    /// `read_pal_indices` + `pal_idx_finish` (indices kept unpacked, one per
    /// byte, `bw4 * 4` wide).
    fn read_pal_indices(&mut self, uv: bool, pal_sz: usize, pl: usize, w4: usize, h4: usize, bw4: usize, bh4: usize) {
        let stride = bw4 * 4;
        let mut tmp = vec![0u8; 64 * 64];
        tmp[0] = self.msac().uniform(pal_sz as u32) as u8;
        let mut order = [[0u8; 8]; 64];
        let mut ctx = [0u8; 64];
        for i in 1..4 * (w4 + h4) - 1 {
            let first = i.min(w4 * 4 - 1);
            let last = (i as isize - h4 as isize * 4 + 1).max(0) as usize;
            Self::order_palette(&tmp, stride, i, first, last, &mut order, &mut ctx);
            let mut m = 0;
            let mut j = first as isize;
            while j >= last as isize {
                let c = ctx[m] as usize;
                let color_idx = sa!(self, pal_sz - 1, m.color_map[pl][pal_sz - 2][c]);
                tmp[(i - j as usize) * stride + j as usize] = order[m][color_idx];
                j -= 1;
                m += 1;
            }
        }
        // pal_idx_finish: fill the invisible right/bottom edges.
        let (w, h, bw, bh) = (w4 * 4, h4 * 4, bw4 * 4, bh4 * 4);
        let dst = if uv { &mut self.pal_idx_uv } else { &mut self.pal_idx_y };
        for y in 0..h {
            dst[y * bw..y * bw + w].copy_from_slice(&tmp[y * stride..y * stride + w]);
            let last = tmp[y * stride + w - 1];
            for v in &mut dst[y * bw + w..y * bw + bw] {
                *v = last;
            }
        }
        for y in h..bh {
            dst.copy_within((h - 1) * bw..h * bw, y * bw);
        }
    }

    fn read_pal_plane(&mut self, b: &mut Av1Block, pl: usize, sz_ctx: usize, bx4: usize, by4: usize) {
        let pal_sz = sa!(self, 6, m.pal_sz[(pl != 0) as usize][sz_ctx]) + 2;
        b.pal_sz[(pl != 0) as usize] = pal_sz as u8;
        let mut cache = [0u16; 16];
        let mut used_cache = [0u16; 8];
        let mut l_cache = if pl != 0 { self.pal_sz_uv[1][by4] } else { self.l.pal_sz[by4] } as usize;
        let mut n_cache = 0;
        let mut a_cache = if by4 & 15 != 0 { if pl != 0 { self.pal_sz_uv[0][bx4] } else { self.a[self.ai].pal_sz[bx4] } } else { 0 } as usize;
        let lp = self.al_pal[1][by4][pl];
        let ap = self.al_pal[0][bx4][pl];
        let (mut li, mut ai) = (0, 0);
        while l_cache != 0 && a_cache != 0 {
            if lp[li] < ap[ai] {
                if n_cache == 0 || cache[n_cache - 1] != lp[li] {
                    cache[n_cache] = lp[li];
                    n_cache += 1;
                }
                li += 1;
                l_cache -= 1;
            } else {
                if ap[ai] == lp[li] {
                    li += 1;
                    l_cache -= 1;
                }
                if n_cache == 0 || cache[n_cache - 1] != ap[ai] {
                    cache[n_cache] = ap[ai];
                    n_cache += 1;
                }
                ai += 1;
                a_cache -= 1;
            }
        }
        if l_cache != 0 {
            loop {
                if n_cache == 0 || cache[n_cache - 1] != lp[li] {
                    cache[n_cache] = lp[li];
                    n_cache += 1;
                }
                li += 1;
                l_cache -= 1;
                if l_cache == 0 {
                    break;
                }
            }
        } else if a_cache != 0 {
            loop {
                if n_cache == 0 || cache[n_cache - 1] != ap[ai] {
                    cache[n_cache] = ap[ai];
                    n_cache += 1;
                }
                ai += 1;
                a_cache -= 1;
                if a_cache == 0 {
                    break;
                }
            }
        }
        let mut i = 0;
        for &c in cache.iter().take(n_cache) {
            if i >= pal_sz {
                break;
            }
            if self.msac().bool_equi() {
                used_cache[i] = c;
                i += 1;
            }
        }
        let n_used = i;
        let mut pal = [0u16; 8];
        if i < pal_sz {
            let bpc = self.bpc;
            let mut prev = self.msac().bools(bpc) as i32;
            pal[i] = prev as u16;
            i += 1;
            if i < pal_sz {
                let mut bits = bpc as i32 - 3 + self.msac().bools(2) as i32;
                let max = (1 << bpc) - 1;
                let npl = (pl == 0) as i32;
                loop {
                    let delta = self.msac().bools(bits as u32) as i32;
                    prev = (prev + delta + npl).min(max);
                    pal[i] = prev as u16;
                    i += 1;
                    if prev + npl >= max {
                        while i < pal_sz {
                            pal[i] = max as u16;
                            i += 1;
                        }
                        break;
                    }
                    bits = bits.min(1 + ulog2((max - prev - npl) as u32));
                    if i >= pal_sz {
                        break;
                    }
                }
            }
            let (mut n, mut m) = (0, n_used);
            let mut out = [0u16; 8];
            for o in out.iter_mut().take(pal_sz) {
                if n < n_used && (m >= pal_sz || used_cache[n] <= pal[m]) {
                    *o = used_cache[n];
                    n += 1;
                } else {
                    *o = pal[m];
                    m += 1;
                }
            }
            // dav1d merges in place: entries are only read at m >= the write
            // position, so reading the pre-merge copy is equivalent.
            pal = out;
        } else {
            pal[..n_used].copy_from_slice(&used_cache[..n_used]);
        }
        self.pal[pl] = pal;
    }

    fn read_pal_uv(&mut self, b: &mut Av1Block, sz_ctx: usize, bx4: usize, by4: usize) {
        self.read_pal_plane(b, 1, sz_ctx, bx4, by4);
        let bpc = self.bpc;
        let n = b.pal_sz[1] as usize;
        if self.msac().bool_equi() {
            let bits = bpc as i32 - 4 + self.msac().bools(2) as i32;
            let max = (1 << bpc) - 1;
            let mut prev = self.msac().bools(bpc) as i32;
            self.pal[2][0] = prev as u16;
            for i in 1..n {
                let mut delta = self.msac().bools(bits as u32) as i32;
                if delta != 0 && self.msac().bool_equi() {
                    delta = -delta;
                }
                prev = (prev + delta) & max;
                self.pal[2][i] = prev as u16;
            }
        } else {
            for i in 0..n {
                self.pal[2][i] = self.msac().bools(bpc) as u16;
            }
        }
    }

    fn copy_pal_block_y(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        for x in 0..bw4 {
            self.al_pal[0][bx4 + x][0] = self.pal[0];
        }
        for y in 0..bh4 {
            self.al_pal[1][by4 + y][0] = self.pal[0];
        }
    }

    fn copy_pal_block_uv(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        for pl in 1..=2 {
            for x in 0..bw4 {
                self.al_pal[0][bx4 + x][pl] = self.pal[pl];
            }
            for y in 0..bh4 {
                self.al_pal[1][by4 + y][pl] = self.pal[pl];
            }
        }
    }

    fn lf_mask_call(&mut self, b: &Av1Block, bs: usize, has_chroma: bool, levels: [u8; 4], inter: Option<(bool, [u16; 2])>, ytx: usize, uvtx: usize) {
        let bx4 = (self.bx & 31) as usize;
        let by4 = (self.by & 31) as usize;
        let cbx4 = bx4 >> self.ss_hor;
        let cby4 = by4 >> self.ss_ver;
        let _ = b;
        let a = &mut self.a[self.ai];
        let l = &mut self.l;
        let (ay, auv) = (&mut a.tx_lpf_y[bx4..], &mut a.tx_lpf_uv[cbx4..]);
        let (ly, luv) = (&mut l.tx_lpf_y[by4..], &mut l.tx_lpf_uv[cby4..]);
        let e = LfEdges { ay, ly, uv: if has_chroma { Some((auv, luv)) } else { None } };
        lf_mask::create_lf_mask(
            &mut self.lf.mask[self.lf_mask],
            &mut self.lf.level,
            self.b4_stride as usize,
            levels,
            self.bx as usize,
            self.by as usize,
            self.w4 as usize,
            self.h4 as usize,
            bs,
            ytx,
            inter,
            uvtx,
            self.layout,
            e,
        );
    }

    fn lflvl(&self, seg: usize) -> &[[[u8; 2]; 8]; 4] {
        let ts = &self.ts[self.tsi];
        if ts.lflvl_frame {
            &self.lf.lvl[seg]
        } else {
            &ts.lflvlmem[seg]
        }
    }

    #[allow(clippy::misrefactored_assign_op)] // intrabc clamping, see there
    pub fn decode_b(&mut self, bl: usize, bs: usize, bp: usize, intra_edge_flags: u8) -> DResult {
        let hdr = self.hdr.clone();
        let seq = self.seq.clone();
        let mut b = Av1Block::default();
        let b_dim = BLOCK_DIMENSIONS[bs];
        let bx4 = (self.bx & 31) as usize;
        let by4 = (self.by & 31) as usize;
        let ss_ver = self.ss_ver as usize;
        let ss_hor = self.ss_hor as usize;
        let cbx4 = bx4 >> ss_hor;
        let cby4 = by4 >> ss_ver;
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let cbw4 = (bw4 + ss_hor as i32) >> ss_hor;
        let cbh4 = (bh4 + ss_ver as i32) >> ss_ver;
        let tiling_cs = self.ts[self.tsi].tiling.col_start;
        let tiling_rs = self.ts[self.tsi].tiling.row_start;
        let have_left = self.bx > tiling_cs;
        let have_top = self.by > tiling_rs;
        let has_chroma = self.layout != Layout::I400 && (bw4 > ss_hor as i32 || self.bx & 1 != 0) && (bh4 > ss_ver as i32 || self.by & 1 != 0);
        let cw4 = (w4 + ss_hor as i32) >> ss_hor;
        let ch4 = (h4 + ss_ver as i32) >> ss_ver;
        let inter_frame = hdr.is_inter_or_switch();
        let ai = self.ai;

        b.bl = bl as u8;
        b.bp = bp as u8;
        b.bs = bs as u8;

        let segmentation = &hdr.segmentation;
        let mut seg: Option<SegmentationData> = None;
        let mut seg_pred = false;
        if segmentation.enabled {
            if !segmentation.update_map {
                b.seg_id = if self.prev_segmap.is_some() {
                    let s = self.get_prev_frame_segid(self.by, self.bx, w4, h4);
                    if s >= 8 {
                        return Err("bad segment id".into());
                    }
                    s
                } else {
                    0
                };
                seg = Some(segmentation.seg_data.d[b.seg_id as usize]);
            } else if segmentation.seg_data.preskip {
                let ctx = (self.a[ai].seg_pred[bx4] + self.l.seg_pred[by4]) as usize;
                if segmentation.temporal && {
                    seg_pred = ba!(self, m.seg_pred[ctx]);
                    seg_pred
                } {
                    b.seg_id = if self.prev_segmap.is_some() {
                        let s = self.get_prev_frame_segid(self.by, self.bx, w4, h4);
                        if s >= 8 {
                            return Err("bad segment id".into());
                        }
                        s
                    } else {
                        0
                    };
                } else {
                    let (pred, seg_ctx) =
                        get_cur_frame_segid(self.by as usize, self.bx as usize, have_top, have_left, self.cur_segmap.as_ref().unwrap(), self.b4_stride as usize);
                    let diff = sa!(self, 7, m.seg_id[seg_ctx]) as i32;
                    let last_active = segmentation.seg_data.last_active_segid as i32;
                    let mut s = neg_deinterleave(diff, pred as i32, last_active + 1);
                    if s > last_active {
                        s = 0;
                    }
                    if !(0..8).contains(&s) {
                        s = 0;
                    }
                    b.seg_id = s as u8;
                }
                seg = Some(segmentation.seg_data.d[b.seg_id as usize]);
            }
        } else {
            b.seg_id = 0;
        }

        // skip_mode
        if seg.map(|s| !s.globalmv && s.r#ref == -1 && !s.skip).unwrap_or(true) && hdr.skip_mode_enabled && bw4.min(bh4) > 1 {
            let smctx = (self.a[ai].skip_mode[bx4] + self.l.skip_mode[by4]) as usize;
            b.skip_mode = ba!(self, m.skip_mode[smctx]);
        }

        // skip
        if b.skip_mode || seg.map(|s| s.skip).unwrap_or(false) {
            b.skip = true;
        } else {
            let sctx = (self.a[ai].skip[bx4] + self.l.skip[by4]) as usize;
            b.skip = ba!(self, m.skip[sctx]);
        }

        // segment_id (post-skip)
        if segmentation.enabled && segmentation.update_map && !segmentation.seg_data.preskip {
            let ctx = (self.a[ai].seg_pred[bx4] + self.l.seg_pred[by4]) as usize;
            if !b.skip && segmentation.temporal && {
                seg_pred = ba!(self, m.seg_pred[ctx]);
                seg_pred
            } {
                b.seg_id = if self.prev_segmap.is_some() {
                    let s = self.get_prev_frame_segid(self.by, self.bx, w4, h4);
                    if s >= 8 {
                        return Err("bad segment id".into());
                    }
                    s
                } else {
                    0
                };
            } else {
                let (pred, seg_ctx) =
                    get_cur_frame_segid(self.by as usize, self.bx as usize, have_top, have_left, self.cur_segmap.as_ref().unwrap(), self.b4_stride as usize);
                let mut s;
                if b.skip {
                    s = pred as i32;
                } else {
                    let diff = sa!(self, 7, m.seg_id[seg_ctx]) as i32;
                    let last_active = segmentation.seg_data.last_active_segid as i32;
                    s = neg_deinterleave(diff, pred as i32, last_active + 1);
                    if s > last_active {
                        s = 0;
                    }
                }
                if !(0..8).contains(&s) {
                    s = 0;
                }
                b.seg_id = s as u8;
            }
            seg = Some(segmentation.seg_data.d[b.seg_id as usize]);
        }

        // cdef index
        if !b.skip {
            let idx = if seq.sb128 { (((self.bx & 16) >> 4) + ((self.by & 16) >> 3)) as usize } else { 0 };
            let m = &mut self.lf.mask[self.lf_mask].cdef_idx;
            let ci = self.cdef_idx;
            if m[ci + idx] == -1 {
                let nb = hdr.cdef.n_bits as u32;
                let v = self.ts[self.tsi].msac.bools(nb) as i8;
                let m = &mut self.lf.mask[self.lf_mask].cdef_idx;
                m[ci + idx] = v;
                if bw4 > 16 {
                    m[ci + idx + 1] = v;
                }
                if bh4 > 16 {
                    m[ci + idx + 2] = v;
                }
                if bw4 == 32 && bh4 == 32 {
                    m[ci + idx + 3] = v;
                }
            }
        }

        // delta q / lf
        let sbm = 31 >> (!seq.sb128) as i32;
        if self.bx & sbm == 0 && self.by & sbm == 0 {
            let prev_qidx = self.ts[self.tsi].last_qidx;
            let have_delta_q = hdr.delta.q_present && (bs != if seq.sb128 { BS_128X128 } else { BS_64X64 } || !b.skip);
            let prev_delta_lf = self.ts[self.tsi].last_delta_lf;
            if have_delta_q {
                let mut delta_q = sa!(self, 3, m.delta_q) as i32;
                if delta_q == 3 {
                    let n_bits = 1 + self.msac().bools(3);
                    delta_q = self.msac().bools(n_bits) as i32 + 1 + (1 << n_bits);
                }
                if delta_q != 0 {
                    if self.msac().bool_equi() {
                        delta_q = -delta_q;
                    }
                    delta_q *= 1 << hdr.delta.q_res_log2;
                }
                let ts = &mut self.ts[self.tsi];
                ts.last_qidx = (ts.last_qidx + delta_q).clamp(1, 255);
                if hdr.delta.lf_present {
                    let n_lfs = if hdr.delta.lf_multi { if self.layout != Layout::I400 { 4 } else { 2 } } else { 1 };
                    for i in 0..n_lfs {
                        let ci = i + hdr.delta.lf_multi as usize;
                        let mut delta_lf = sa!(self, 3, m.delta_lf[ci]) as i32;
                        if delta_lf == 3 {
                            let n_bits = 1 + self.msac().bools(3);
                            delta_lf = self.msac().bools(n_bits) as i32 + 1 + (1 << n_bits);
                        }
                        if delta_lf != 0 {
                            if self.msac().bool_equi() {
                                delta_lf = -delta_lf;
                            }
                            delta_lf *= 1 << hdr.delta.lf_res_log2;
                        }
                        let ts = &mut self.ts[self.tsi];
                        ts.last_delta_lf[i] = (ts.last_delta_lf[i] as i32 + delta_lf).clamp(-63, 63) as i8;
                    }
                }
            }
            let ts = &mut self.ts[self.tsi];
            if ts.last_qidx == hdr.quant.yac as i32 {
                ts.dq_frame = true;
            } else if ts.last_qidx != prev_qidx {
                init_quant_tables(&seq, &hdr, ts.last_qidx, &mut ts.dqmem);
                ts.dq_frame = false;
            }
            if ts.last_delta_lf == [0; 4] {
                ts.lflvl_frame = true;
            } else if ts.last_delta_lf != prev_delta_lf {
                ts.lflvl_frame = false;
                lf_mask::calc_lf_values(&mut ts.lflvlmem, &hdr, ts.last_delta_lf);
            }
        }

        if b.skip_mode {
            b.intra = false;
        } else if inter_frame {
            if let Some(s) = seg.filter(|s| s.r#ref >= 0 || s.globalmv) {
                b.intra = s.r#ref == 0;
            } else {
                let ictx = get_intra_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                b.intra = !ba!(self, m.intra[ictx]);
            }
        } else if hdr.allow_intrabc {
            b.intra = !ba!(self, m.intrabc);
        } else {
            b.intra = true;
        }

        if b.intra {
            // ---------------------------------------------------------- intra
            b.y_mode = if inter_frame {
                let c = YMODE_SIZE_CONTEXT[bs] as usize;
                sa!(self, N_INTRA_PRED_MODES - 1, m.y_mode[c])
            } else {
                let ca = INTRA_MODE_CONTEXT[self.a[ai].mode[bx4] as usize] as usize;
                let cl = INTRA_MODE_CONTEXT[self.l.mode[by4] as usize] as usize;
                sa!(self, N_INTRA_PRED_MODES - 1, kfym[ca][cl])
            } as u8;
            if b_dim[2] + b_dim[3] >= 2 && b.y_mode as usize >= VERT_PRED && b.y_mode as usize <= VERT_LEFT_PRED {
                let k = b.y_mode as usize - VERT_PRED;
                b.y_angle = sa!(self, 6, m.angle_delta[k]) as i8 - 3;
            } else {
                b.y_angle = 0;
            }
            if has_chroma {
                let cfl_allowed = if hdr.segmentation.lossless[b.seg_id as usize] {
                    cbw4 == 1 && cbh4 == 1
                } else {
                    (CFL_ALLOWED_MASK & (1 << bs)) != 0
                };
                let ym = b.y_mode as usize;
                b.uv_mode = sa!(self, N_UV_INTRA_PRED_MODES - 1 - !cfl_allowed as usize, m.uv_mode[cfl_allowed as usize][ym]) as u8;
                b.uv_angle = 0;
                if b.uv_mode as usize == CFL_PRED {
                    let sign = sa!(self, 7, m.cfl_sign) as i32 + 1;
                    let sign_u = sign * 0x56 >> 8;
                    let sign_v = sign - sign_u * 3;
                    if sign_u != 0 {
                        let ctx = ((sign_u == 2) as i32 * 3 + sign_v) as usize;
                        let v = sa!(self, 15, m.cfl_alpha[ctx]) as i8 + 1;
                        b.cfl_alpha[0] = if sign_u == 1 { -v } else { v };
                    } else {
                        b.cfl_alpha[0] = 0;
                    }
                    if sign_v != 0 {
                        let ctx = ((sign_v == 2) as i32 * 3 + sign_u) as usize;
                        let v = sa!(self, 15, m.cfl_alpha[ctx]) as i8 + 1;
                        b.cfl_alpha[1] = if sign_v == 1 { -v } else { v };
                    } else {
                        b.cfl_alpha[1] = 0;
                    }
                } else if b_dim[2] + b_dim[3] >= 2 && b.uv_mode as usize >= VERT_PRED && b.uv_mode as usize <= VERT_LEFT_PRED {
                    let k = b.uv_mode as usize - VERT_PRED;
                    b.uv_angle = sa!(self, 6, m.angle_delta[k]) as i8 - 3;
                }
            }

            b.pal_sz = [0, 0];
            if hdr.allow_screen_content_tools && bw4.max(bh4) <= 16 && bw4 + bh4 >= 4 {
                let sz_ctx = (b_dim[2] + b_dim[3] - 2) as usize;
                if b.y_mode as usize == DC_PRED {
                    let pal_ctx = (self.a[ai].pal_sz[bx4] > 0) as usize + (self.l.pal_sz[by4] > 0) as usize;
                    if ba!(self, m.pal_y[sz_ctx][pal_ctx]) {
                        self.read_pal_plane(&mut b, 0, sz_ctx, bx4, by4);
                    }
                }
                if has_chroma && b.uv_mode as usize == DC_PRED {
                    let pal_ctx = (b.pal_sz[0] > 0) as usize;
                    if ba!(self, m.pal_uv[pal_ctx]) {
                        self.read_pal_uv(&mut b, sz_ctx, bx4, by4);
                    }
                }
            }

            if b.y_mode as usize == DC_PRED && b.pal_sz[0] == 0 && b_dim[2].max(b_dim[3]) <= 3 && seq.filter_intra && ba!(self, m.use_filter_intra[bs]) {
                b.y_mode = FILTER_PRED as u8;
                b.y_angle = sa!(self, 4, m.filter_intra) as i8;
            }

            if b.pal_sz[0] != 0 {
                self.read_pal_indices(false, b.pal_sz[0] as usize, 0, w4 as usize, h4 as usize, bw4 as usize, bh4 as usize);
            }
            if has_chroma && b.pal_sz[1] != 0 {
                self.read_pal_indices(true, b.pal_sz[1] as usize, 1, cw4 as usize, ch4 as usize, cbw4 as usize, cbh4 as usize);
            }

            let mut t_dim;
            if hdr.segmentation.lossless[b.seg_id as usize] {
                b.tx = TX_4X4 as u8;
                b.uvtx = TX_4X4 as u8;
                t_dim = TXFM_DIMENSIONS[TX_4X4];
            } else {
                b.tx = MAX_TXFM_SIZE_FOR_BS[bs][0];
                b.uvtx = MAX_TXFM_SIZE_FOR_BS[bs][self.layout as usize];
                t_dim = TXFM_DIMENSIONS[b.tx as usize];
                if hdr.txfm_mode == TX_SWITCHABLE && t_dim.max as usize > TX_4X4 {
                    let tctx = get_tx_ctx(&self.a[ai], &self.l, &t_dim, by4, bx4);
                    let mx = t_dim.max as usize;
                    let mut depth = sa!(self, mx.min(2), m.txsz[mx - 1][tctx]);
                    while depth > 0 {
                        b.tx = t_dim.sub;
                        t_dim = TXFM_DIMENSIONS[b.tx as usize];
                        depth -= 1;
                    }
                }
            }

            self.recon_b_intra(bs, intra_edge_flags, &b);

            if hdr.loopfilter.level_y[0] != 0 || hdr.loopfilter.level_y[1] != 0 {
                let lv = self.lflvl(b.seg_id as usize);
                let levels = [lv[0][0][0], lv[1][0][0], lv[2][0][0], lv[3][0][0]];
                self.lf_mask_call(&b, bs, has_chroma, levels, None, b.tx as usize, b.uvtx as usize);
            }

            let y_mode_nofilt = if b.y_mode as usize == FILTER_PRED { DC_PRED as u8 } else { b.y_mode };
            for i in 0..2 {
                let off = if i == 0 { bx4 } else { by4 };
                let n = 1usize << b_dim[2 + i];
                let t_lsz = if i == 0 { t_dim.lw } else { t_dim.lh } as i8;
                let pal_uv = if has_chroma { b.pal_sz[1] } else { 0 };
                let e = if i == 0 { &mut self.a[ai] } else { &mut self.l };
                set(&mut e.tx_intra, off, n, t_lsz);
                set(&mut e.tx, off, n, t_lsz);
                set(&mut e.mode, off, n, y_mode_nofilt);
                set(&mut e.pal_sz, off, n, b.pal_sz[0]);
                set(&mut e.seg_pred, off, n, seg_pred as u8);
                set(&mut e.skip_mode, off, n, 0);
                set(&mut e.intra, off, n, 1);
                set(&mut e.skip, off, n, b.skip as u8);
                if inter_frame {
                    set(&mut e.comp_type, off, n, COMP_INTER_NONE);
                    set(&mut e.r#ref[0], off, n, -1);
                    set(&mut e.r#ref[1], off, n, -1);
                    set(&mut e.filter[0], off, n, N_SWITCHABLE_FILTERS);
                    set(&mut e.filter[1], off, n, N_SWITCHABLE_FILTERS);
                }
                set(&mut self.pal_sz_uv[i], off, n, pal_uv);
            }
            if b.pal_sz[0] != 0 {
                self.copy_pal_block_y(bx4, by4, bw4 as usize, bh4 as usize);
            }
            if has_chroma {
                set(&mut self.a[ai].uvmode, cbx4, 1 << ulog2(cbw4 as u32), b.uv_mode);
                set(&mut self.l.uvmode, cby4, 1 << ulog2(cbh4 as u32), b.uv_mode);
                if b.pal_sz[1] != 0 {
                    self.copy_pal_block_uv(bx4, by4, bw4 as usize, bh4 as usize);
                }
            }
            if inter_frame || hdr.allow_intrabc {
                self.splat(RefmvsBlock { mv: [Mv::INVALID, Mv::ZERO], r: [0, -1], bs: bs as u8, mf: 0 }, bw4, bh4);
            }
        } else if hdr.is_key_or_intra() {
            // ------------------------------------------------------- intrabc
            let (stack, _n_mvs, _ctx) =
                refmvs::find(self.rf.as_ref().unwrap(), &hdr, &self.rt, [0, -1], bs, intra_edge_flags, self.by, self.bx);
            if stack[0].mv[0] != Mv::ZERO {
                b.mv[0] = stack[0].mv[0];
            } else if stack[1].mv[0] != Mv::ZERO {
                b.mv[0] = stack[1].mv[0];
            } else if self.by - (16 << seq.sb128 as i32) < tiling_rs {
                b.mv[0] = Mv { y: 0, x: (-(512 << seq.sb128 as i32) - 2048) as i16 };
            } else {
                b.mv[0] = Mv { y: -(512 << seq.sb128 as i32) as i16, x: 0 };
            }
            let mut mv = b.mv[0];
            self.read_mv_residual(&mut mv, -1);
            b.mv[0] = mv;

            let tiling = &self.ts[self.tsi].tiling;
            let mut border_left = tiling.col_start * 4;
            let mut border_top = tiling.row_start * 4;
            if has_chroma {
                if bw4 < 2 && ss_hor != 0 {
                    border_left += 4;
                }
                if bh4 < 2 && ss_ver != 0 {
                    border_top += 4;
                }
            }
            let mut src_left = self.bx * 4 + (b.mv[0].x as i32 >> 3);
            let mut src_top = self.by * 4 + (b.mv[0].y as i32 >> 3);
            let mut src_right = src_left + bw4 * 4;
            let mut src_bottom = src_top + bh4 * 4;
            let border_right = ((tiling.col_end + (bw4 - 1)) & !(bw4 - 1)) * 4;
            // dav1d's own form: both edges move by the same delta, so the
            // second line of each pair reads `x -= x - limit` on purpose.
            if src_left < border_left {
                src_right += border_left - src_left;
                src_left += border_left - src_left;
            } else if src_right > border_right {
                src_left -= src_right - border_right;
                src_right -= src_right - border_right;
            }
            if src_top < border_top {
                src_bottom += border_top - src_top;
                src_top += border_top - src_top;
            }
            let sb = seq.sb128 as i32;
            let sbx = (self.bx >> (4 + sb)) << (6 + sb);
            let sby = (self.by >> (4 + sb)) << (6 + sb);
            let sb_size = 1 << (6 + sb);
            if src_bottom > sby && src_right > sbx {
                if src_top - border_top >= src_bottom - sby {
                    src_top -= src_bottom - sby;
                    src_bottom -= src_bottom - sby;
                } else if src_left - border_left >= src_right - sbx {
                    src_left -= src_right - sbx;
                    src_right -= src_right - sbx;
                }
            }
            if src_bottom > sby + sb_size {
                src_top -= src_bottom - (sby + sb_size);
                src_bottom -= src_bottom - (sby + sb_size);
            }
            if src_bottom > sby && src_right > sbx {
                return Err("intrabc mv overlaps the current superblock".into());
            }
            b.mv[0].x = ((src_left - self.bx * 4) * 8) as i16;
            b.mv[0].y = ((src_top - self.by * 4) * 8) as i16;

            self.read_vartx_tree(&mut b, bs, bx4, by4);
            self.recon_b_inter(bs, &b)?;

            self.splat(RefmvsBlock { mv: [b.mv[0], Mv::ZERO], r: [0, -1], bs: bs as u8, mf: 0 }, bw4, bh4);
            for i in 0..2 {
                let off = if i == 0 { bx4 } else { by4 };
                let n = 1usize << b_dim[2 + i];
                let e = if i == 0 { &mut self.a[ai] } else { &mut self.l };
                set(&mut e.tx_intra, off, n, b_dim[2 + i] as i8);
                set(&mut e.mode, off, n, DC_PRED as u8);
                set(&mut e.pal_sz, off, n, 0);
                set(&mut e.seg_pred, off, n, seg_pred as u8);
                set(&mut e.skip_mode, off, n, 0);
                set(&mut e.intra, off, n, 0);
                set(&mut e.skip, off, n, b.skip as u8);
                set(&mut self.pal_sz_uv[i], off, n, 0);
            }
            if has_chroma {
                set(&mut self.a[ai].uvmode, cbx4, 1 << ulog2(cbw4 as u32), DC_PRED as u8);
                set(&mut self.l.uvmode, cby4, 1 << ulog2(cbh4 as u32), DC_PRED as u8);
            }
        } else {
            // --------------------------------------------------------- inter
            let mut has_subpel_filter;
            let is_comp = if b.skip_mode {
                true
            } else if seg.map(|s| s.r#ref == -1 && !s.globalmv && !s.skip).unwrap_or(true) && hdr.switchable_comp_refs && bw4.min(bh4) > 1 {
                let ctx = get_comp_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                ba!(self, m.comp[ctx])
            } else {
                false
            };

            if b.skip_mode {
                b.r#ref = hdr.skip_mode_refs;
                b.comp_type = COMP_INTER_AVG;
                b.inter_mode = NEARESTMV_NEARESTMV;
                b.drl_idx = NEAREST_DRL;
                has_subpel_filter = false;
                let (stack, _, _) = refmvs::find(
                    self.rf.as_ref().unwrap(),
                    &hdr,
                    &self.rt,
                    [b.r#ref[0] + 1, b.r#ref[1] + 1],
                    bs,
                    intra_edge_flags,
                    self.by,
                    self.bx,
                );
                b.mv = stack[0].mv;
                fix_mv_precision(&hdr, &mut b.mv[0]);
                fix_mv_precision(&hdr, &mut b.mv[1]);
            } else if is_comp {
                let (a, l) = (&self.a[ai], &self.l);
                let dir_ctx = get_comp_dir_ctx(a, l, by4, bx4, have_top, have_left);
                if ba!(self, m.comp_dir[dir_ctx]) {
                    let (a, l) = (&self.a[ai], &self.l);
                    let ctx1 = get_fwd_ref_ctx(a, l, by4, bx4, have_top, have_left);
                    if ba!(self, m.comp_fwd_ref[0][ctx1]) {
                        let ctx2 = get_fwd_ref_2_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        b.r#ref[0] = 2 + ba!(self, m.comp_fwd_ref[2][ctx2]) as i8;
                    } else {
                        let ctx2 = get_fwd_ref_1_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        b.r#ref[0] = ba!(self, m.comp_fwd_ref[1][ctx2]) as i8;
                    }
                    let ctx3 = get_bwd_ref_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                    if ba!(self, m.comp_bwd_ref[0][ctx3]) {
                        b.r#ref[1] = 6;
                    } else {
                        let ctx4 = get_bwd_ref_1_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        b.r#ref[1] = 4 + ba!(self, m.comp_bwd_ref[1][ctx4]) as i8;
                    }
                } else {
                    let uctx_p = get_ref_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                    if ba!(self, m.comp_uni_ref[0][uctx_p]) {
                        b.r#ref = [4, 6];
                    } else {
                        let uctx_p1 = get_uni_p1_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        b.r#ref[0] = 0;
                        b.r#ref[1] = 1 + ba!(self, m.comp_uni_ref[1][uctx_p1]) as i8;
                        if b.r#ref[1] == 2 {
                            let uctx_p2 = get_fwd_ref_2_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                            b.r#ref[1] += ba!(self, m.comp_uni_ref[2][uctx_p2]) as i8;
                        }
                    }
                }

                let (stack, n_mvs, ctx) = refmvs::find(
                    self.rf.as_ref().unwrap(),
                    &hdr,
                    &self.rt,
                    [b.r#ref[0] + 1, b.r#ref[1] + 1],
                    bs,
                    intra_edge_flags,
                    self.by,
                    self.bx,
                );
                b.inter_mode = sa!(self, N_COMP_INTER_PRED_MODES - 1, m.comp_inter_mode[ctx as usize]) as u8;
                let im = COMP_INTER_PRED_MODES[b.inter_mode as usize];
                b.drl_idx = NEAREST_DRL;
                if b.inter_mode == NEWMV_NEWMV {
                    if n_mvs > 1 {
                        let c1 = get_drl_context(&stack, 0);
                        b.drl_idx += ba!(self, m.drl_bit[c1]) as u8;
                        if b.drl_idx == NEARER_DRL && n_mvs > 2 {
                            let c2 = get_drl_context(&stack, 1);
                            b.drl_idx += ba!(self, m.drl_bit[c2]) as u8;
                        }
                    }
                } else if im[0] == NEARMV || im[1] == NEARMV {
                    b.drl_idx = NEARER_DRL;
                    if n_mvs > 2 {
                        let c2 = get_drl_context(&stack, 1);
                        b.drl_idx += ba!(self, m.drl_bit[c2]) as u8;
                        if b.drl_idx == NEAR_DRL && n_mvs > 3 {
                            let c3 = get_drl_context(&stack, 2);
                            b.drl_idx += ba!(self, m.drl_bit[c3]) as u8;
                        }
                    }
                }
                has_subpel_filter = bw4.min(bh4) == 1 || b.inter_mode != GLOBALMV_GLOBALMV;
                for idx in 0..2 {
                    match im[idx] {
                        NEARMV | NEARESTMV => {
                            b.mv[idx] = stack[b.drl_idx as usize].mv[idx];
                            fix_mv_precision(&hdr, &mut b.mv[idx]);
                        }
                        GLOBALMV => {
                            let g = &hdr.gmv[b.r#ref[idx] as usize];
                            has_subpel_filter |= g.typ == WM_TRANSLATION;
                            b.mv[idx] = get_gmv_2d(g, self.bx, self.by, bw4, bh4, &hdr);
                        }
                        _ => {
                            let mut mv = stack[b.drl_idx as usize].mv[idx];
                            let mv_prec = hdr.hp as i32 - hdr.force_integer_mv as i32;
                            self.read_mv_residual(&mut mv, mv_prec);
                            b.mv[idx] = mv;
                        }
                    }
                }

                let mut is_segwedge = false;
                if seq.masked_compound {
                    let mask_ctx = get_mask_comp_ctx(&self.a[ai], &self.l, by4, bx4);
                    is_segwedge = ba!(self, m.mask_comp[mask_ctx]);
                }
                if !is_segwedge {
                    if seq.jnt_comp {
                        let poc = hdr.frame_offset as u32;
                        let r0 = self.refp[b.r#ref[0] as usize].as_ref().unwrap().hdr.frame_offset as u32;
                        let r1 = self.refp[b.r#ref[1] as usize].as_ref().unwrap().hdr.frame_offset as u32;
                        let jnt_ctx = get_jnt_comp_ctx(seq.order_hint_n_bits, poc, r0, r1, &self.a[ai], &self.l, by4, bx4);
                        b.comp_type = COMP_INTER_WEIGHTED_AVG + ba!(self, m.jnt_comp[jnt_ctx]) as u8;
                    } else {
                        b.comp_type = COMP_INTER_AVG;
                    }
                } else {
                    if WEDGE_ALLOWED_MASK & (1 << bs) != 0 {
                        let ctx = WEDGE_CTX_LUT[bs] as usize;
                        b.comp_type = COMP_INTER_WEDGE - ba!(self, m.wedge_comp[ctx]) as u8;
                        if b.comp_type == COMP_INTER_WEDGE {
                            b.wedge_idx = sa!(self, 15, m.wedge_idx[ctx]) as u8;
                        }
                    } else {
                        b.comp_type = COMP_INTER_SEG;
                    }
                    b.mask_sign = self.msac().bool_equi() as u8;
                }
            } else {
                b.comp_type = COMP_INTER_NONE;
                if let Some(s) = seg.filter(|s| s.r#ref > 0) {
                    b.r#ref[0] = s.r#ref - 1;
                } else if seg.map(|s| s.globalmv || s.skip).unwrap_or(false) {
                    b.r#ref[0] = 0;
                } else {
                    let ctx1 = get_ref_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                    if ba!(self, m.r#ref[0][ctx1]) {
                        let ctx2 = get_bwd_ref_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        if ba!(self, m.r#ref[1][ctx2]) {
                            b.r#ref[0] = 6;
                        } else {
                            let ctx3 = get_bwd_ref_1_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                            b.r#ref[0] = 4 + ba!(self, m.r#ref[5][ctx3]) as i8;
                        }
                    } else {
                        let ctx2 = get_fwd_ref_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                        if ba!(self, m.r#ref[2][ctx2]) {
                            let ctx3 = get_fwd_ref_2_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                            b.r#ref[0] = 2 + ba!(self, m.r#ref[4][ctx3]) as i8;
                        } else {
                            let ctx3 = get_fwd_ref_1_ctx(&self.a[ai], &self.l, by4, bx4, have_top, have_left);
                            b.r#ref[0] = ba!(self, m.r#ref[3][ctx3]) as i8;
                        }
                    }
                }
                b.r#ref[1] = -1;

                let (stack, n_mvs, ctx) =
                    refmvs::find(self.rf.as_ref().unwrap(), &hdr, &self.rt, [b.r#ref[0] + 1, -1], bs, intra_edge_flags, self.by, self.bx);
                let segskip = seg.map(|s| s.skip || s.globalmv).unwrap_or(false);
                if segskip || {
                    let c = (ctx & 7) as usize;
                    ba!(self, m.newmv_mode[c])
                } {
                    if segskip || {
                        let c = ((ctx >> 3) & 1) as usize;
                        !ba!(self, m.globalmv_mode[c])
                    } {
                        b.inter_mode = GLOBALMV;
                        let g = &hdr.gmv[b.r#ref[0] as usize];
                        b.mv[0] = get_gmv_2d(g, self.bx, self.by, bw4, bh4, &hdr);
                        has_subpel_filter = bw4.min(bh4) == 1 || g.typ == WM_TRANSLATION;
                    } else {
                        has_subpel_filter = true;
                        let c = ((ctx >> 4) & 15) as usize;
                        if ba!(self, m.refmv_mode[c]) {
                            b.inter_mode = NEARMV;
                            b.drl_idx = NEARER_DRL;
                            if n_mvs > 2 {
                                let c2 = get_drl_context(&stack, 1);
                                b.drl_idx += ba!(self, m.drl_bit[c2]) as u8;
                                if b.drl_idx == NEAR_DRL && n_mvs > 3 {
                                    let c3 = get_drl_context(&stack, 2);
                                    b.drl_idx += ba!(self, m.drl_bit[c3]) as u8;
                                }
                            }
                        } else {
                            b.inter_mode = NEARESTMV;
                            b.drl_idx = NEAREST_DRL;
                        }
                        b.mv[0] = stack[b.drl_idx as usize].mv[0];
                        if b.drl_idx < NEAR_DRL {
                            fix_mv_precision(&hdr, &mut b.mv[0]);
                        }
                    }
                } else {
                    has_subpel_filter = true;
                    b.inter_mode = NEWMV;
                    b.drl_idx = NEAREST_DRL;
                    if n_mvs > 1 {
                        let c1 = get_drl_context(&stack, 0);
                        b.drl_idx += ba!(self, m.drl_bit[c1]) as u8;
                        if b.drl_idx == NEARER_DRL && n_mvs > 2 {
                            let c2 = get_drl_context(&stack, 1);
                            b.drl_idx += ba!(self, m.drl_bit[c2]) as u8;
                        }
                    }
                    let mut mv;
                    if n_mvs > 1 {
                        mv = stack[b.drl_idx as usize].mv[0];
                    } else {
                        mv = stack[0].mv[0];
                        fix_mv_precision(&hdr, &mut mv);
                    }
                    let mv_prec = hdr.hp as i32 - hdr.force_integer_mv as i32;
                    self.read_mv_residual(&mut mv, mv_prec);
                    b.mv[0] = mv;
                }

                // interintra
                let ii_sz_grp = YMODE_SIZE_CONTEXT[bs] as usize;
                if seq.inter_intra && INTERINTRA_ALLOWED_MASK & (1 << bs) != 0 && ba!(self, m.interintra[ii_sz_grp]) {
                    b.interintra_mode = sa!(self, N_INTER_INTRA_PRED_MODES - 1, m.interintra_mode[ii_sz_grp]) as u8;
                    let wedge_ctx = WEDGE_CTX_LUT[bs] as usize;
                    b.interintra_type = INTER_INTRA_BLEND + ba!(self, m.interintra_wedge[wedge_ctx]) as u8;
                    if b.interintra_type == INTER_INTRA_WEDGE {
                        b.wedge_idx = sa!(self, 15, m.wedge_idx[wedge_ctx]) as u8;
                    }
                } else {
                    b.interintra_type = INTER_INTRA_NONE;
                }

                // motion variation
                let r0 = b.r#ref[0] as usize;
                if hdr.switchable_motion_mode
                    && b.interintra_type == INTER_INTRA_NONE
                    && bw4.min(bh4) >= 2
                    && !(!hdr.force_integer_mv && b.inter_mode == GLOBALMV && hdr.gmv[r0].typ > WM_TRANSLATION)
                    && ((have_left && findoddzero(&self.l.intra[by4 + 1..], (h4 >> 1) as usize))
                        || (have_top && findoddzero(&self.a[ai].intra[bx4 + 1..], (w4 >> 1) as usize)))
                {
                    let mut mask = [0u64; 2];
                    self.find_matching_ref(intra_edge_flags, bw4, bh4, w4, h4, have_left, have_top, b.r#ref[0], &mut mask);
                    let allow_warp = self.svc[r0][0].scale == 0 && !hdr.force_integer_mv && hdr.warp_motion && (mask[0] | mask[1]) != 0;
                    b.motion_mode = if allow_warp { sa!(self, 2, m.motion_mode[bs]) as u8 } else { ba!(self, m.obmc[bs]) as u8 };
                    if b.motion_mode == MM_WARP {
                        has_subpel_filter = false;
                        self.derive_warpmv(bw4, bh4, &mask, b.mv[0]);
                    }
                } else {
                    b.motion_mode = MM_TRANSLATION;
                }
            }

            // subpel filter
            let mut filter = [0u8; 2];
            if hdr.subpel_filter_mode == FILTER_SWITCHABLE {
                if has_subpel_filter {
                    let comp = b.comp_type != COMP_INTER_NONE;
                    let ctx1 = get_filter_ctx(&self.a[ai], &self.l, comp, 0, b.r#ref[0], by4, bx4);
                    filter[0] = sa!(self, N_SWITCHABLE_FILTERS as usize - 1, m.filter[0][ctx1]) as u8;
                    if seq.dual_filter {
                        let ctx2 = get_filter_ctx(&self.a[ai], &self.l, comp, 1, b.r#ref[0], by4, bx4);
                        filter[1] = sa!(self, N_SWITCHABLE_FILTERS as usize - 1, m.filter[1][ctx2]) as u8;
                    } else {
                        filter[1] = filter[0];
                    }
                } else {
                    filter = [FILTER_8TAP_REGULAR; 2];
                }
            } else {
                filter = [hdr.subpel_filter_mode; 2];
            }
            b.filter2d = FILTER_2D[filter[1] as usize][filter[0] as usize];

            self.read_vartx_tree(&mut b, bs, bx4, by4);
            self.recon_b_inter(bs, &b)?;

            if hdr.loopfilter.level_y[0] != 0 || hdr.loopfilter.level_y[1] != 0 {
                let is_globalmv = b.inter_mode == if is_comp { GLOBALMV_GLOBALMV } else { GLOBALMV };
                let lv = self.lflvl(b.seg_id as usize);
                let ri = (b.r#ref[0] + 1) as usize;
                let gi = !is_globalmv as usize;
                let levels = [lv[0][ri][gi], lv[1][ri][gi], lv[2][ri][gi], lv[3][ri][gi]];
                let tx_split = [b.tx_split0 as u16, b.tx_split1];
                let (mut ytx, mut uvtx) = (b.max_ytx as usize, b.uvtx as usize);
                if hdr.segmentation.lossless[b.seg_id as usize] {
                    ytx = TX_4X4;
                    uvtx = TX_4X4;
                }
                self.lf_mask_call(&b, bs, has_chroma, levels, Some((b.skip, tx_split)), ytx, uvtx);
            }

            if is_comp {
                let mode = b.inter_mode;
                let mf = (mode == GLOBALMV_GLOBALMV) as u8 | (((1u32 << mode) & 0xbc != 0) as u8) * 2;
                self.splat(RefmvsBlock { mv: b.mv, r: [b.r#ref[0] + 1, b.r#ref[1] + 1], bs: bs as u8, mf }, bw4, bh4);
            } else {
                let mode = b.inter_mode;
                let mf = (mode == GLOBALMV && bw4.min(bh4) >= 2) as u8 | ((mode == NEWMV) as u8 * 2);
                self.splat(
                    RefmvsBlock { mv: [b.mv[0], Mv::ZERO], r: [b.r#ref[0] + 1, if b.interintra_type != 0 { 0 } else { -1 }], bs: bs as u8, mf },
                    bw4,
                    bh4,
                );
            }
            for i in 0..2 {
                let off = if i == 0 { bx4 } else { by4 };
                let n = 1usize << b_dim[2 + i];
                let e = if i == 0 { &mut self.a[ai] } else { &mut self.l };
                set(&mut e.seg_pred, off, n, seg_pred as u8);
                set(&mut e.skip_mode, off, n, b.skip_mode as u8);
                set(&mut e.intra, off, n, 0);
                set(&mut e.skip, off, n, b.skip as u8);
                set(&mut e.pal_sz, off, n, 0);
                set(&mut e.tx_intra, off, n, b_dim[2 + i] as i8);
                set(&mut e.comp_type, off, n, b.comp_type);
                set(&mut e.filter[0], off, n, filter[0]);
                set(&mut e.filter[1], off, n, filter[1]);
                set(&mut e.mode, off, n, b.inter_mode);
                set(&mut e.r#ref[0], off, n, b.r#ref[0]);
                set(&mut e.r#ref[1], off, n, b.r#ref[1]);
                set(&mut self.pal_sz_uv[i], off, n, 0);
            }
            if has_chroma {
                set(&mut self.a[ai].uvmode, cbx4, 1 << ulog2(cbw4 as u32), DC_PRED as u8);
                set(&mut self.l.uvmode, cby4, 1 << ulog2(cbh4 as u32), DC_PRED as u8);
            }
        }

        // segmentation map
        if hdr.segmentation.enabled && hdr.segmentation.update_map {
            let stride = self.b4_stride as usize;
            let map = self.cur_segmap.as_mut().unwrap();
            let n = 1usize << b_dim[2];
            for y in 0..bh4 as usize {
                let o = (self.by as usize + y) * stride + self.bx as usize;
                set(map, o, n, b.seg_id);
            }
        }
        if !b.skip {
            let nm = &mut self.lf.mask[self.lf_mask].noskip_mask;
            let mask = ((!0u32 >> (32 - bw4)) << (bx4 & 15)) as u16;
            let bx_idx = (bx4 & 16) >> 4;
            let mut k = by4 >> 1;
            let mut y = 0;
            while y < bh4 {
                nm[k][bx_idx] |= mask;
                if bw4 == 32 {
                    nm[k][1] |= mask;
                }
                y += 2;
                k += 1;
            }
        }
        Ok(())
    }

    pub fn decode_sb(&mut self, bl: usize, node_idx: usize) -> DResult {
        let tree = intra_edge::tree();
        let node: EdgeNode = tree.nodes[node_idx];
        let hsz = 16 >> bl;
        let have_h_split = self.bw > self.bx + hsz;
        let have_v_split = self.bh > self.by + hsz;
        if !have_h_split && !have_v_split {
            return self.decode_sb(bl + 1, node.split[0]);
        }
        let bx8 = ((self.bx & 31) >> 1) as usize;
        let by8 = ((self.by & 31) >> 1) as usize;
        let ctx = get_partition_ctx(&self.a[self.ai], &self.l, bl, by8, bx8);
        let bp;
        if have_h_split && have_v_split {
            bp = sa!(self, PARTITION_TYPE_COUNT[bl] as usize, m.partition[bl][ctx]);
            if self.layout == Layout::I422 && matches!(bp, PARTITION_V | PARTITION_V4 | PARTITION_T_LEFT_SPLIT | PARTITION_T_RIGHT_SPLIT) {
                return Err("invalid 4:2:2 partition".into());
            }
            let b = BLOCK_SIZES[bl][bp];
            let (b0, b1) = (b[0] as usize, b[1] as usize);
            match bp {
                PARTITION_NONE => self.decode_b(bl, b0, bp, node.o)?,
                PARTITION_H => {
                    self.decode_b(bl, b0, bp, node.h[0])?;
                    self.by += hsz;
                    self.decode_b(bl, b0, bp, node.h[1])?;
                    self.by -= hsz;
                }
                PARTITION_V => {
                    self.decode_b(bl, b0, bp, node.v[0])?;
                    self.bx += hsz;
                    self.decode_b(bl, b0, bp, node.v[1])?;
                    self.bx -= hsz;
                }
                PARTITION_SPLIT => {
                    if bl == BL_8X8 {
                        self.decode_b(bl, BS_4X4, bp, EDGE_ALL_TR_AND_BL)?;
                        let tl_filter = self.tl_4x4_filter;
                        self.bx += 1;
                        self.decode_b(bl, BS_4X4, bp, node.tip_split[0])?;
                        self.bx -= 1;
                        self.by += 1;
                        self.decode_b(bl, BS_4X4, bp, node.tip_split[1])?;
                        self.bx += 1;
                        self.tl_4x4_filter = tl_filter;
                        self.decode_b(bl, BS_4X4, bp, node.tip_split[2])?;
                        self.bx -= 1;
                        self.by -= 1;
                    } else {
                        self.decode_sb(bl + 1, node.split[0])?;
                        self.bx += hsz;
                        self.decode_sb(bl + 1, node.split[1])?;
                        self.bx -= hsz;
                        self.by += hsz;
                        self.decode_sb(bl + 1, node.split[2])?;
                        self.bx += hsz;
                        self.decode_sb(bl + 1, node.split[3])?;
                        self.bx -= hsz;
                        self.by -= hsz;
                    }
                }
                PARTITION_T_TOP_SPLIT => {
                    self.decode_b(bl, b0, bp, EDGE_ALL_TR_AND_BL)?;
                    self.bx += hsz;
                    self.decode_b(bl, b0, bp, node.v[1])?;
                    self.bx -= hsz;
                    self.by += hsz;
                    self.decode_b(bl, b1, bp, node.h[1])?;
                    self.by -= hsz;
                }
                PARTITION_T_BOTTOM_SPLIT => {
                    self.decode_b(bl, b0, bp, node.h[0])?;
                    self.by += hsz;
                    self.decode_b(bl, b1, bp, node.v[0])?;
                    self.bx += hsz;
                    self.decode_b(bl, b1, bp, 0)?;
                    self.bx -= hsz;
                    self.by -= hsz;
                }
                PARTITION_T_LEFT_SPLIT => {
                    self.decode_b(bl, b0, bp, EDGE_ALL_TR_AND_BL)?;
                    self.by += hsz;
                    self.decode_b(bl, b0, bp, node.h[1])?;
                    self.by -= hsz;
                    self.bx += hsz;
                    self.decode_b(bl, b1, bp, node.v[1])?;
                    self.bx -= hsz;
                }
                PARTITION_T_RIGHT_SPLIT => {
                    self.decode_b(bl, b0, bp, node.v[0])?;
                    self.bx += hsz;
                    self.decode_b(bl, b1, bp, node.h[0])?;
                    self.by += hsz;
                    self.decode_b(bl, b1, bp, 0)?;
                    self.by -= hsz;
                    self.bx -= hsz;
                }
                PARTITION_H4 => {
                    self.decode_b(bl, b0, bp, node.h[0])?;
                    self.by += hsz >> 1;
                    self.decode_b(bl, b0, bp, node.h4)?;
                    self.by += hsz >> 1;
                    self.decode_b(bl, b0, bp, EDGE_ALL_LEFT_HAS_BOTTOM)?;
                    self.by += hsz >> 1;
                    if self.by < self.bh {
                        self.decode_b(bl, b0, bp, node.h[1])?;
                    }
                    self.by -= hsz * 3 >> 1;
                }
                _ => {
                    self.decode_b(bl, b0, bp, node.v[0])?;
                    self.bx += hsz >> 1;
                    self.decode_b(bl, b0, bp, node.v4)?;
                    self.bx += hsz >> 1;
                    self.decode_b(bl, b0, bp, EDGE_ALL_TOP_HAS_RIGHT)?;
                    self.bx += hsz >> 1;
                    if self.bx < self.bw {
                        self.decode_b(bl, b0, bp, node.v[1])?;
                    }
                    self.bx -= hsz * 3 >> 1;
                }
            }
        } else if have_h_split {
            let p = {
                let ts = &self.ts[self.tsi];
                gather_top_partition_prob(&ts.cdf.m.partition[bl][ctx], bl)
            };
            let is_split = self.msac().bool(p);
            if is_split {
                bp = PARTITION_SPLIT;
                self.decode_sb(bl + 1, node.split[0])?;
                self.bx += hsz;
                self.decode_sb(bl + 1, node.split[1])?;
                self.bx -= hsz;
            } else {
                bp = PARTITION_H;
                self.decode_b(bl, BLOCK_SIZES[bl][PARTITION_H][0] as usize, PARTITION_H, node.h[0])?;
            }
        } else {
            let p = {
                let ts = &self.ts[self.tsi];
                gather_left_partition_prob(&ts.cdf.m.partition[bl][ctx], bl)
            };
            let is_split = self.msac().bool(p);
            if self.layout == Layout::I422 && !is_split {
                return Err("invalid 4:2:2 partition".into());
            }
            if is_split {
                bp = PARTITION_SPLIT;
                self.decode_sb(bl + 1, node.split[0])?;
                self.by += hsz;
                self.decode_sb(bl + 1, node.split[2])?;
                self.by -= hsz;
            } else {
                bp = PARTITION_V;
                self.decode_b(bl, BLOCK_SIZES[bl][PARTITION_V][0] as usize, PARTITION_V, node.v[0])?;
            }
        }
        if bp != PARTITION_SPLIT || bl == BL_8X8 {
            let n = hsz as usize;
            let ai = self.ai;
            set(&mut self.a[ai].partition, bx8, n.min(16), AL_PART_CTX[0][bl][bp]);
            set(&mut self.l.partition, by8, n.min(16), AL_PART_CTX[1][bl][bp]);
        }
        Ok(())
    }
}

pub fn reset_context(ctx: &mut BlockContext, keyframe: bool) {
    ctx.intra = [keyframe as u8; 32];
    ctx.uvmode = [DC_PRED as u8; 32];
    if keyframe {
        ctx.mode = [DC_PRED as u8; 32];
    }
    ctx.partition = [0; 16];
    ctx.skip = [0; 32];
    ctx.skip_mode = [0; 32];
    ctx.tx_lpf_y = [2; 32];
    ctx.tx_lpf_uv = [1; 32];
    ctx.tx_intra = [-1; 32];
    ctx.tx = [TX_64X64 as i8; 32];
    if !keyframe {
        ctx.r#ref = [[-1; 32]; 2];
        ctx.comp_type = [0; 32];
        ctx.mode = [NEARESTMV; 32];
    }
    ctx.lcoef = [0x40; 32];
    ctx.ccoef = [[0x40; 32]; 2];
    ctx.filter = [[N_SWITCHABLE_FILTERS; 32]; 2];
    ctx.seg_pred = [0; 32];
    ctx.pal_sz = [0; 32];
}

pub const CFL_ALLOWED_MASK: u32 = (1 << BS_32X32)
    | (1 << BS_32X16)
    | (1 << BS_32X8)
    | (1 << BS_16X32)
    | (1 << BS_16X16)
    | (1 << BS_16X8)
    | (1 << BS_16X4)
    | (1 << BS_8X32)
    | (1 << BS_8X16)
    | (1 << BS_8X8)
    | (1 << BS_8X4)
    | (1 << BS_4X16)
    | (1 << BS_4X8)
    | (1 << BS_4X4);
pub const WEDGE_ALLOWED_MASK: u32 = (1 << BS_32X32)
    | (1 << BS_32X16)
    | (1 << BS_32X8)
    | (1 << BS_16X32)
    | (1 << BS_16X16)
    | (1 << BS_16X8)
    | (1 << BS_8X32)
    | (1 << BS_8X16)
    | (1 << BS_8X8);
pub const INTERINTRA_ALLOWED_MASK: u32 =
    (1 << BS_32X32) | (1 << BS_32X16) | (1 << BS_16X32) | (1 << BS_16X16) | (1 << BS_16X8) | (1 << BS_8X16) | (1 << BS_8X8);

// Unused-but-referenced helpers keep the type list honest for later stages.
#[allow(dead_code)]
fn _types(_: &RefmvsCandidate, _: &TemporalBlock, _: &RestorationUnit) {}
