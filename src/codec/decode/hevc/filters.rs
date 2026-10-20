//! HEVC in-loop filters: deblocking (§8.7.2) and sample adaptive offset
//! (§8.7.3), each run over a whole decoded picture.

use super::params::{Pps, Sps};
use super::pic::{flag, MvField, PicState};
use super::tables::{chroma_qp, BETA, TC};

/// Reference POC of list `l` of a block, through its slice's lists.
fn ref_poc(pic: &PicState, x: usize, y: usize, f: &MvField, l: usize) -> Option<i32> {
    if f.ref_idx[l] < 0 {
        return None;
    }
    let s = pic.ctb_slice[pic.ctb_rs(x, y)];
    pic.slices
        .get(s as usize)?
        .refs[l]
        .get(f.ref_idx[l] as usize)
        .map(|e| e.0)
}

/// Boundary strength across the edge between the 4×4 blocks containing
/// luma samples `p` and `q` (§8.7.2.4).
fn strength(pic: &PicState, p: (usize, usize), q: (usize, usize), tu_edge: bool) -> u8 {
    let ip = pic.idx4(p.0, p.1);
    let iq = pic.idx4(q.0, q.1);
    let (fp, fq) = (pic.flags[ip], pic.flags[iq]);
    if (fp | fq) & flag::INTRA != 0 {
        return 2;
    }
    if tu_edge && (fp | fq) & flag::CBF != 0 {
        return 1;
    }
    let (mp, mq) = (&pic.mv[ip], &pic.mv[iq]);
    let pr = [ref_poc(pic, p.0, p.1, mp, 0), ref_poc(pic, p.0, p.1, mp, 1)];
    let qr = [ref_poc(pic, q.0, q.1, mq, 0), ref_poc(pic, q.0, q.1, mq, 1)];
    let far = |a: [i16; 2], b: [i16; 2]| {
        (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= 4
    };
    let np = pr.iter().flatten().count();
    let nq = qr.iter().flatten().count();
    if np != nq {
        return 1;
    }
    if np == 1 {
        let (rp, vp) = if pr[0].is_some() { (pr[0], mp.mv[0]) } else { (pr[1], mp.mv[1]) };
        let (rq, vq) = if qr[0].is_some() { (qr[0], mq.mv[0]) } else { (qr[1], mq.mv[1]) };
        return (rp != rq || far(vp, vq)) as u8;
    }
    if np == 0 {
        return 0;
    }
    let same = (pr[0] == qr[0] && pr[1] == qr[1]) || (pr[0] == qr[1] && pr[1] == qr[0]);
    if !same {
        return 1;
    }
    let (p0, p1, q0, q1) = (mp.mv[0], mp.mv[1], mq.mv[0], mq.mv[1]);
    if pr[0] != pr[1] {
        if pr[0] == qr[0] {
            (far(p0, q0) || far(p1, q1)) as u8
        } else {
            (far(p0, q1) || far(p1, q0)) as u8
        }
    } else {
        ((far(p0, q0) || far(p1, q1)) && (far(p0, q1) || far(p1, q0))) as u8
    }
}

/// Whether the edge into the block at `q` from `p` may be filtered at all:
/// the slice containing `q` has deblocking on, and slice and tile boundary
/// rules allow it.
fn edge_allowed(pic: &PicState, pps: &Pps, p: (usize, usize), q: (usize, usize)) -> Option<usize> {
    let cq = pic.ctb_rs(q.0, q.1);
    let cp = pic.ctb_rs(p.0, p.1);
    let sq = pic.ctb_slice[cq];
    let sp = pic.ctb_slice[cp];
    if sq == u16::MAX || sp == u16::MAX {
        return None;
    }
    let s = &pic.slices[sq as usize];
    if s.deblocking_disabled {
        return None;
    }
    if pic.slices[sp as usize].addr_rs != s.addr_rs && !s.lf_across_slices {
        return None;
    }
    if !pps.loop_filter_across_tiles && pic.layout.tile[cq] != pic.layout.tile[cp] {
        return None;
    }
    Some(sq as usize)
}

/// Filter one 4-line luma edge segment. `at` indexes `q0` of the first line.
#[allow(clippy::too_many_arguments)]
fn luma_segment(
    d: &mut [u16],
    at: usize,
    across: usize,
    along: usize,
    bs: u8,
    qp_av: i32,
    beta_off: i32,
    tc_off: i32,
    bd: u32,
    no_p: bool,
    no_q: bool,
) {
    let qb = (qp_av + beta_off).clamp(0, 51) as usize;
    let beta = BETA[qb] * (1 << (bd - 8));
    let qt = (qp_av + 2 * (bs as i32 - 1) + tc_off).clamp(0, 53) as usize;
    let tc = TC[qt] * (1 << (bd - 8));
    if tc == 0 && beta == 0 {
        return;
    }
    let max = (1i32 << bd) - 1;
    let px = |line: usize, i: usize| at + line * along - across * (i + 1);
    let qx = |line: usize, i: usize| at + line * along + across * i;
    let s = |d: &[u16], i: usize| d[i] as i32;
    let dp = |d: &[u16], l: usize| (s(d, px(l, 2)) - 2 * s(d, px(l, 1)) + s(d, px(l, 0))).abs();
    let dq = |d: &[u16], l: usize| (s(d, qx(l, 2)) - 2 * s(d, qx(l, 1)) + s(d, qx(l, 0))).abs();
    let (dp0, dp3, dq0, dq3) = (dp(d, 0), dp(d, 3), dq(d, 0), dq(d, 3));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    let (dps, dqs) = (dp0 + dp3, dq0 + dq3);
    if dpq0 + dpq3 >= beta {
        return;
    }
    let strong_line = |d: &[u16], l: usize, dpq: i32| {
        2 * dpq < (beta >> 2)
            && (s(d, px(l, 3)) - s(d, px(l, 0))).abs() + (s(d, qx(l, 0)) - s(d, qx(l, 3))).abs() < (beta >> 3)
            && (s(d, px(l, 0)) - s(d, qx(l, 0))).abs() < ((5 * tc + 1) >> 1)
    };
    let strong = strong_line(d, 0, dpq0) && strong_line(d, 3, dpq3);
    let side = (beta + (beta >> 1)) >> 3;
    let de_p = dps < side;
    let de_q = dqs < side;
    for l in 0..4 {
        let p: [i32; 4] = std::array::from_fn(|i| s(d, px(l, i)));
        let q: [i32; 4] = std::array::from_fn(|i| s(d, qx(l, i)));
        if strong {
            if !no_p {
                d[px(l, 0)] = ((p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3)
                    .clamp(p[0] - 2 * tc, p[0] + 2 * tc) as u16;
                d[px(l, 1)] = ((p[2] + p[1] + p[0] + q[0] + 2) >> 2).clamp(p[1] - 2 * tc, p[1] + 2 * tc) as u16;
                d[px(l, 2)] = ((2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3)
                    .clamp(p[2] - 2 * tc, p[2] + 2 * tc) as u16;
            }
            if !no_q {
                d[qx(l, 0)] = ((p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3)
                    .clamp(q[0] - 2 * tc, q[0] + 2 * tc) as u16;
                d[qx(l, 1)] = ((p[0] + q[0] + q[1] + q[2] + 2) >> 2).clamp(q[1] - 2 * tc, q[1] + 2 * tc) as u16;
                d[qx(l, 2)] = ((p[0] + q[0] + q[1] + 3 * q[2] + 2 * q[3] + 4) >> 3)
                    .clamp(q[2] - 2 * tc, q[2] + 2 * tc) as u16;
            }
        } else {
            let mut delta = (9 * (q[0] - p[0]) - 3 * (q[1] - p[1]) + 8) >> 4;
            if delta.abs() >= tc * 10 {
                continue;
            }
            delta = delta.clamp(-tc, tc);
            if !no_p {
                d[px(l, 0)] = (p[0] + delta).clamp(0, max) as u16;
            }
            if !no_q {
                d[qx(l, 0)] = (q[0] - delta).clamp(0, max) as u16;
            }
            if de_p && !no_p {
                let dpv = ((((p[2] + p[0] + 1) >> 1) - p[1] + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                d[px(l, 1)] = (p[1] + dpv).clamp(0, max) as u16;
            }
            if de_q && !no_q {
                let dqv = ((((q[2] + q[0] + 1) >> 1) - q[1] - delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                d[qx(l, 1)] = (q[1] + dqv).clamp(0, max) as u16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn chroma_segment(d: &mut [u16], at: usize, across: usize, along: usize, tc: i32, bd: u32, no_p: bool, no_q: bool) {
    if tc == 0 {
        return;
    }
    let max = (1i32 << bd) - 1;
    for l in 0..4 {
        let q0i = at + l * along;
        let (p0, p1) = (d[q0i - across] as i32, d[q0i - 2 * across] as i32);
        let (q0, q1) = (d[q0i] as i32, d[q0i + across] as i32);
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        if !no_p {
            d[q0i - across] = (p0 + delta).clamp(0, max) as u16;
        }
        if !no_q {
            d[q0i] = (q0 - delta).clamp(0, max) as u16;
        }
    }
}

pub fn deblock(pic: &mut PicState, sps: &Sps, pps: &Pps) {
    let (w, h) = (pic.width, pic.height);
    let (bdy, bdc) = (sps.bit_depth_y, sps.bit_depth_c);
    let wl = pic.planes[0].width;
    let wc = pic.planes[1].width;
    for dir in 0..2 {
        // dir 0: vertical edges (filtering across x), 1: horizontal.
        let mut y = 0;
        while y < h {
            let mut x = 0;
            while x < w {
                let on_grid = if dir == 0 { x % 8 == 0 && x > 0 } else { y % 8 == 0 && y > 0 };
                if !on_grid {
                    x += 4;
                    continue;
                }
                let q = (x, y);
                let p = if dir == 0 { (x - 1, y) } else { (x, y - 1) };
                let iq = pic.idx4(x, y);
                let (tu_edge, pu_edge) = if dir == 0 {
                    (pic.flags[iq] & flag::EDGE_V != 0, pic.pu_edge[iq] & 1 != 0)
                } else {
                    (pic.flags[iq] & flag::EDGE_H != 0, pic.pu_edge[iq] & 2 != 0)
                };
                if !(tu_edge || pu_edge) {
                    x += 4;
                    continue;
                }
                let Some(si) = edge_allowed(pic, pps, p, q) else {
                    x += 4;
                    continue;
                };
                let bs = strength(pic, p, q, tu_edge);
                if bs == 0 {
                    x += 4;
                    continue;
                }
                let s = &pic.slices[si];
                let (beta_off, tc_off) = (s.beta_offset, s.tc_offset);
                let (cbo, cro) = (pps.cb_qp_offset, pps.cr_qp_offset);
                let ip = pic.idx4(p.0, p.1);
                let qp_av = (pic.qp[ip] as i32 + pic.qp[iq] as i32 + 1) >> 1;
                let no_p = pic.flags[ip] & flag::NO_FILTER != 0;
                let no_q = pic.flags[iq] & flag::NO_FILTER != 0;
                let (at, across, along) = if dir == 0 { (y * wl + x, 1, wl) } else { (y * wl + x, wl, 1) };
                luma_segment(&mut pic.planes[0].data, at, across, along, bs, qp_av, beta_off, tc_off, bdy, no_p, no_q);
                // Chroma edges lie on the 8×8 chroma grid and are filtered
                // in 4-sample segments, each taking the strength and QP at
                // its first sample — so only every other luma segment's.
                let chroma_seg = if dir == 0 { x % 16 == 0 && y % 8 == 0 } else { y % 16 == 0 && x % 8 == 0 };
                if bs == 2 && chroma_seg {
                    let (cx, cy) = (x / 2, y / 2);
                    let (at, across, along) = if dir == 0 { (cy * wc + cx, 1, wc) } else { (cy * wc + cx, wc, 1) };
                    for (cidx, off) in [(1, cbo), (2, cro)] {
                        let qpc = chroma_qp(qp_av + off);
                        let qt = (qpc + 2 + tc_off).clamp(0, 53) as usize;
                        let tc = TC[qt] * (1 << (bdc - 8));
                        chroma_segment(&mut pic.planes[cidx].data, at, across, along, tc, bdc, no_p, no_q);
                    }
                }
                x += 4;
            }
            y += 4;
        }
    }
}

pub fn sao(pic: &mut PicState, sps: &Sps, pps: &Pps) {
    let l = pic.layout.clone();
    let log2 = l.log2_ctb as usize;
    for cidx in 0..3 {
        if !pic.sao.iter().any(|s| s[cidx].kind != 0) {
            continue;
        }
        let sub = if cidx == 0 { 0 } else { 1 };
        let bd = if cidx == 0 { sps.bit_depth_y } else { sps.bit_depth_c };
        let max = (1i32 << bd) - 1;
        let src = pic.planes[cidx].data.clone();
        let (pw, ph) = (pic.planes[cidx].width, pic.planes[cidx].height);
        let ctb = 1usize << (log2 - sub);
        for rs in 0..l.ctb_w * l.ctb_h {
            let sp = pic.sao[rs][cidx];
            if sp.kind == 0 || pic.ctb_slice[rs] == u16::MAX {
                continue;
            }
            let (x0, y0) = ((rs % l.ctb_w) * ctb, (rs / l.ctb_w) * ctb);
            let (x1, y1) = ((x0 + ctb).min(pw), (y0 + ctb).min(ph));
            let cur_slice = &pic.slices[pic.ctb_slice[rs] as usize];
            let mut band_table = [0usize; 32];
            if sp.kind == 1 {
                for k in 0..4 {
                    band_table[(k + sp.band_or_class as usize) & 31] = k + 1;
                }
            }
            let (hp, vp): ([isize; 2], [isize; 2]) = match sp.band_or_class {
                0 => ([-1, 1], [0, 0]),
                1 => ([0, 0], [-1, 1]),
                2 => ([-1, 1], [-1, 1]),
                _ => ([1, -1], [-1, 1]),
            };
            for y in y0..y1 {
                for x in x0..x1 {
                    let (lx, ly) = (x << sub, y << sub);
                    if pic.flags[pic.idx4(lx, ly)] & flag::NO_FILTER != 0 {
                        continue;
                    }
                    let v = src[y * pw + x] as i32;
                    let off = if sp.kind == 1 {
                        let b = band_table[(v >> (bd - 5)) as usize];
                        if b == 0 {
                            continue;
                        }
                        sp.offsets[b - 1] as i32
                    } else {
                        let mut skip = false;
                        let mut nv = [0i32; 2];
                        for k in 0..2 {
                            let (nx, ny) = (x as isize + hp[k], y as isize + vp[k]);
                            if nx < 0 || ny < 0 || nx >= pw as isize || ny >= ph as isize {
                                skip = true;
                                break;
                            }
                            let (nx, ny) = (nx as usize, ny as usize);
                            if nx < x0 || nx >= x1 || ny < y0 || ny >= y1 {
                                // Neighbour in another CTB: slice and tile rules.
                                let (nlx, nly) = (nx << sub, ny << sub);
                                let nrs = pic.ctb_rs(nlx, nly);
                                let ns = pic.ctb_slice[nrs];
                                if ns == u16::MAX {
                                    skip = true;
                                    break;
                                }
                                let n_slice = &pic.slices[ns as usize];
                                if n_slice.addr_rs != cur_slice.addr_rs {
                                    let zn = l.z[(nly >> 2) * l.w4 + (nlx >> 2)];
                                    let zc = l.z[(ly >> 2) * l.w4 + (lx >> 2)];
                                    if (zn < zc && !cur_slice.lf_across_slices)
                                        || (zc < zn && !n_slice.lf_across_slices)
                                    {
                                        skip = true;
                                        break;
                                    }
                                }
                                if !pps.loop_filter_across_tiles && l.tile[nrs] != l.tile[rs] {
                                    skip = true;
                                    break;
                                }
                            }
                            nv[k] = src[ny * pw + nx] as i32;
                        }
                        if skip {
                            continue;
                        }
                        let e = 2 + (v - nv[0]).signum() + (v - nv[1]).signum();
                        match e {
                            0 => sp.offsets[0] as i32,
                            1 => sp.offsets[1] as i32,
                            3 => sp.offsets[2] as i32,
                            4 => sp.offsets[3] as i32,
                            _ => continue,
                        }
                    };
                    pic.planes[cidx].data[y * pw + x] = (v + off).clamp(0, max) as u16;
                }
            }
        }
    }
}
