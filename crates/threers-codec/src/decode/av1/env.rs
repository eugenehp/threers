//! Above/left block context and the context derivations that read it
//! (dav1d `src/env.h`).

use super::levels::*;
use super::obu::{get_poc_diff, FrameHeader, WarpedMotionParams, WM_AFFINE, WM_IDENTITY, WM_ROT_ZOOM, WM_TRANSLATION};
use super::refmvs::RefmvsCandidate;
use super::tables::TxfmInfo;

#[derive(Clone)]
#[derive(Default)]
pub struct BlockContext {
    pub mode: [u8; 32],
    pub lcoef: [u8; 32],
    pub ccoef: [[u8; 32]; 2],
    pub seg_pred: [u8; 32],
    pub skip: [u8; 32],
    pub skip_mode: [u8; 32],
    pub intra: [u8; 32],
    pub comp_type: [u8; 32],
    /// -1 means intra.
    pub r#ref: [[i8; 32]; 2],
    /// 3 means unset.
    pub filter: [[u8; 32]; 2],
    pub tx_intra: [i8; 32],
    pub tx: [i8; 32],
    pub tx_lpf_y: [u8; 32],
    pub tx_lpf_uv: [u8; 32],
    pub partition: [u8; 16],
    pub uvmode: [u8; 32],
    pub pal_sz: [u8; 32],
}


type Bc = BlockContext;

pub fn get_intra_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, have_top: bool, have_left: bool) -> usize {
    if have_left {
        if have_top {
            let ctx = (l.intra[yb4] + a.intra[xb4]) as usize;
            ctx + (ctx == 2) as usize
        } else {
            l.intra[yb4] as usize * 2
        }
    } else if have_top {
        a.intra[xb4] as usize * 2
    } else {
        0
    }
}

pub fn get_tx_ctx(a: &Bc, l: &Bc, max_tx: &TxfmInfo, yb4: usize, xb4: usize) -> usize {
    (l.tx_intra[yb4] as i32 >= max_tx.lh as i32) as usize + (a.tx_intra[xb4] as i32 >= max_tx.lw as i32) as usize
}

pub fn get_partition_ctx(a: &Bc, l: &Bc, bl: usize, yb8: usize, xb8: usize) -> usize {
    ((a.partition[xb8] >> (4 - bl)) & 1) as usize + ((((l.partition[yb8] >> (4 - bl)) & 1) as usize) << 1)
}

pub fn gather_left_partition_prob(i: &[u16], bl: usize) -> u32 {
    let i = |k: usize| i[k] as u32;
    let mut out = i(PARTITION_H - 1).wrapping_sub(i(PARTITION_H));
    out = out.wrapping_add(i(PARTITION_SPLIT - 1).wrapping_sub(i(PARTITION_T_LEFT_SPLIT)));
    if bl != BL_128X128 {
        out = out.wrapping_add(i(PARTITION_H4 - 1).wrapping_sub(i(PARTITION_H4)));
    }
    out
}

pub fn gather_top_partition_prob(i: &[u16], bl: usize) -> u32 {
    let i = |k: usize| i[k] as u32;
    let mut out = i(PARTITION_V - 1).wrapping_sub(i(PARTITION_T_TOP_SPLIT));
    out = out.wrapping_add(i(PARTITION_T_LEFT_SPLIT - 1));
    if bl != BL_128X128 {
        out = out.wrapping_add(i(PARTITION_V4 - 1).wrapping_sub(i(PARTITION_T_RIGHT_SPLIT)));
    }
    out
}

pub fn get_uv_inter_txtp(uvt_dim: &TxfmInfo, ytxtp: u8) -> u8 {
    if uvt_dim.max as usize == TX_32X32 {
        return if ytxtp == IDTX { IDTX } else { DCT_DCT };
    }
    if uvt_dim.min as usize == TX_16X16
        && ((1u32 << ytxtp) & ((1 << H_FLIPADST) | (1 << V_FLIPADST) | (1 << H_ADST) | (1 << V_ADST))) != 0
    {
        return DCT_DCT;
    }
    ytxtp
}

pub fn get_filter_ctx(a: &Bc, l: &Bc, comp: bool, dir: usize, r: i8, yb4: usize, xb4: usize) -> usize {
    const N: u8 = 3;
    let a_f = if a.r#ref[0][xb4] == r || a.r#ref[1][xb4] == r { a.filter[dir][xb4] } else { N };
    let l_f = if l.r#ref[0][yb4] == r || l.r#ref[1][yb4] == r { l.filter[dir][yb4] } else { N };
    let c = comp as usize * 4;
    if a_f == l_f {
        c + a_f as usize
    } else if a_f == N {
        c + l_f as usize
    } else if l_f == N {
        c + a_f as usize
    } else {
        c + N as usize
    }
}

pub fn get_comp_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, have_top: bool, have_left: bool) -> usize {
    if have_top {
        if have_left {
            if a.comp_type[xb4] != 0 {
                if l.comp_type[yb4] != 0 {
                    4
                } else {
                    2 + (l.r#ref[0][yb4] as u8 as u32 >= 4) as usize
                }
            } else if l.comp_type[yb4] != 0 {
                2 + (a.r#ref[0][xb4] as u8 as u32 >= 4) as usize
            } else {
                ((l.r#ref[0][yb4] >= 4) ^ (a.r#ref[0][xb4] >= 4)) as usize
            }
        } else if a.comp_type[xb4] != 0 {
            3
        } else {
            (a.r#ref[0][xb4] >= 4) as usize
        }
    } else if have_left {
        if l.comp_type[yb4] != 0 {
            3
        } else {
            (l.r#ref[0][yb4] >= 4) as usize
        }
    } else {
        1
    }
}

#[inline]
fn has_uni_comp(e: &Bc, off: usize) -> bool {
    (e.r#ref[0][off] < 4) == (e.r#ref[1][off] < 4)
}

pub fn get_comp_dir_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, have_top: bool, have_left: bool) -> usize {
    if have_top && have_left {
        let (ai, li) = (a.intra[xb4] != 0, l.intra[yb4] != 0);
        if ai && li {
            return 2;
        }
        if ai || li {
            let (e, off) = if ai { (l, yb4) } else { (a, xb4) };
            if e.comp_type[off] == COMP_INTER_NONE {
                return 2;
            }
            return 1 + 2 * has_uni_comp(e, off) as usize;
        }
        let a_comp = a.comp_type[xb4] != COMP_INTER_NONE;
        let l_comp = l.comp_type[yb4] != COMP_INTER_NONE;
        let (a0, l0) = (a.r#ref[0][xb4], l.r#ref[0][yb4]);
        if !a_comp && !l_comp {
            1 + 2 * ((a0 >= 4) == (l0 >= 4)) as usize
        } else if !a_comp || !l_comp {
            let (e, off) = if a_comp { (a, xb4) } else { (l, yb4) };
            if !has_uni_comp(e, off) {
                return 1;
            }
            3 + ((a0 >= 4) == (l0 >= 4)) as usize
        } else {
            let (au, lu) = (has_uni_comp(a, xb4), has_uni_comp(l, yb4));
            if !au && !lu {
                return 0;
            }
            if !au || !lu {
                return 2;
            }
            3 + ((a0 == 4) == (l0 == 4)) as usize
        }
    } else if have_top || have_left {
        let (e, off) = if have_left { (l, yb4) } else { (a, xb4) };
        if e.intra[off] != 0 {
            return 2;
        }
        if e.comp_type[off] == COMP_INTER_NONE {
            return 2;
        }
        4 * has_uni_comp(e, off) as usize
    } else {
        2
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_jnt_comp_ctx(nb: u8, poc: u32, ref0poc: u32, ref1poc: u32, a: &Bc, l: &Bc, yb4: usize, xb4: usize) -> usize {
    let d0 = get_poc_diff(nb, ref0poc as i32, poc as i32).unsigned_abs();
    let d1 = get_poc_diff(nb, poc as i32, ref1poc as i32).unsigned_abs();
    let offset = (d0 == d1) as usize;
    let a_ctx = (a.comp_type[xb4] >= COMP_INTER_AVG || a.r#ref[0][xb4] == 6) as usize;
    let l_ctx = (l.comp_type[yb4] >= COMP_INTER_AVG || l.r#ref[0][yb4] == 6) as usize;
    3 * offset + a_ctx + l_ctx
}

pub fn get_mask_comp_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize) -> usize {
    let f = |e: &Bc, o: usize| {
        if e.comp_type[o] >= COMP_INTER_SEG {
            1
        } else if e.r#ref[0][o] == 6 {
            3
        } else {
            0
        }
    };
    (f(a, xb4) + f(l, yb4)).min(5)
}

/// Neighbour reference counting shared by the `av1_get_*ref*_ctx` family:
/// `pick(r)` returns the counter slot for reference `r`, if any.
fn count_refs<const N: usize>(a: &Bc, l: &Bc, yb4: usize, xb4: usize, have_top: bool, have_left: bool, pick: impl Fn(i8) -> Option<usize>) -> [i32; N] {
    let mut cnt = [0i32; N];
    let mut add = |e: &Bc, o: usize| {
        if let Some(k) = pick(e.r#ref[0][o]) {
            cnt[k] += 1;
        }
        if e.comp_type[o] != 0 {
            if let Some(k) = pick(e.r#ref[1][o]) {
                cnt[k] += 1;
            }
        }
    };
    if have_top && a.intra[xb4] == 0 {
        add(a, xb4);
    }
    if have_left && l.intra[yb4] == 0 {
        add(l, yb4);
    }
    cnt
}

#[inline]
fn cmp3(a: i32, b: i32) -> usize {
    if a == b {
        1
    } else if a < b {
        0
    } else {
        2
    }
}

pub fn get_ref_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<2>(a, l, yb4, xb4, ht, hl, |r| Some((r >= 4) as usize));
    cmp3(c[0], c[1])
}

pub fn get_fwd_ref_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<4>(a, l, yb4, xb4, ht, hl, |r| if r < 4 { Some(r as usize) } else { None });
    cmp3(c[0] + c[1], c[2] + c[3])
}

pub fn get_fwd_ref_1_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<2>(a, l, yb4, xb4, ht, hl, |r| if r < 2 { Some(r as usize) } else { None });
    cmp3(c[0], c[1])
}

pub fn get_fwd_ref_2_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<2>(a, l, yb4, xb4, ht, hl, |r| if (r as u32 ^ 2) < 2 { Some(r as usize - 2) } else { None });
    cmp3(c[0], c[1])
}

pub fn get_bwd_ref_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<3>(a, l, yb4, xb4, ht, hl, |r| if r >= 4 { Some(r as usize - 4) } else { None });
    cmp3(c[1] + c[0], c[2])
}

pub fn get_bwd_ref_1_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<3>(a, l, yb4, xb4, ht, hl, |r| if r >= 4 { Some(r as usize - 4) } else { None });
    cmp3(c[0], c[1])
}

pub fn get_uni_p1_ctx(a: &Bc, l: &Bc, yb4: usize, xb4: usize, ht: bool, hl: bool) -> usize {
    let c = count_refs::<3>(a, l, yb4, xb4, ht, hl, |r| if (r as i32 as u32).wrapping_sub(1) < 3 { Some(r as usize - 1) } else { None });
    cmp3(c[0], c[1] + c[2])
}

pub fn get_drl_context(stack: &[RefmvsCandidate], idx: usize) -> usize {
    if stack[idx].weight >= 640 {
        (stack[idx + 1].weight < 640) as usize
    } else if stack[idx + 1].weight < 640 {
        2
    } else {
        0
    }
}

/// `get_cur_frame_segid`: (predicted segment id, context).
pub fn get_cur_frame_segid(by: usize, bx: usize, have_top: bool, have_left: bool, map: &[u8], stride: usize) -> (u8, usize) {
    let o = bx + by * stride;
    if have_left && have_top {
        let l = map[o - 1];
        let a = map[o - stride];
        let al = map[o - stride - 1];
        let ctx = if l == a && al == l {
            2
        } else if l == a || al == l || a == al {
            1
        } else {
            0
        };
        (if a == al { a } else { l }, ctx)
    } else {
        (
            if have_left {
                map[o - 1]
            } else if have_top {
                map[o - stride]
            } else {
                0
            },
            0,
        )
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Mv {
    pub y: i16,
    pub x: i16,
}

impl Mv {
    pub const ZERO: Mv = Mv { y: 0, x: 0 };
    pub const INVALID: Mv = Mv { y: i16::MIN, x: i16::MIN };
}

#[inline]
pub fn fix_int_mv_precision(mv: &mut Mv) {
    let f = |v: i16| ((v as i32 - (v as i32 >> 15) + 3) as u32 & !7u32) as i16;
    mv.x = f(mv.x);
    mv.y = f(mv.y);
}

pub fn fix_mv_precision(hdr: &FrameHeader, mv: &mut Mv) {
    if hdr.force_integer_mv {
        fix_int_mv_precision(mv);
    } else if !hdr.hp {
        let f = |v: i16| ((v as i32 - (v as i32 >> 15)) as u32 & !1u32) as i16;
        mv.x = f(mv.x);
        mv.y = f(mv.y);
    }
}

pub fn get_gmv_2d(gmv: &WarpedMotionParams, bx4: i32, by4: i32, bw4: i32, bh4: i32, hdr: &FrameHeader) -> Mv {
    match gmv.typ {
        WM_IDENTITY => Mv::ZERO,
        WM_TRANSLATION => {
            let mut r = Mv { y: (gmv.matrix[0] >> 13) as i16, x: (gmv.matrix[1] >> 13) as i16 };
            if hdr.force_integer_mv {
                fix_int_mv_precision(&mut r);
            }
            r
        }
        _ => {
            debug_assert!(gmv.typ == WM_ROT_ZOOM || gmv.typ == WM_AFFINE);
            let m = &gmv.matrix;
            let x = bx4 * 4 + bw4 * 2 - 1;
            let y = by4 * 4 + bh4 * 2 - 1;
            let xc = (m[2] - (1 << 16)).wrapping_mul(x).wrapping_add(m[3].wrapping_mul(y)).wrapping_add(m[0]);
            let yc = (m[5] - (1 << 16)).wrapping_mul(y).wrapping_add(m[4].wrapping_mul(x)).wrapping_add(m[1]);
            let shift = 16 - (3 - !hdr.hp as i32);
            let round = (1 << shift) >> 1;
            let f = |c: i32| {
                let v = ((c.unsigned_abs() as i32 + round) >> shift) << (!hdr.hp as i32);
                (if c < 0 { -v } else { v }) as i16
            };
            let mut r = Mv { y: f(yc), x: f(xc) };
            if hdr.force_integer_mv {
                fix_int_mv_precision(&mut r);
            }
            r
        }
    }
}
