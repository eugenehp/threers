//! H.264 deblocking (§8.7), run over a whole decoded picture.

use super::slice::CurPic;
use super::types::MbInfo;
use crate::codec::h264::deblock_tables::{ALPHA, BETA, TC0};

/// Boundary strength between 4×4 block `kp` of `p` and `kq` of `q`.
fn strength(mb_edge: bool, p: &MbInfo, kp: usize, q: &MbInfo, kq: usize) -> u8 {
    if p.kind.is_intra() || q.kind.is_intra() {
        return if mb_edge { 4 } else { 3 };
    }
    if p.nz_mask & (1 << kp) != 0 || q.nz_mask & (1 << kq) != 0 {
        return 2;
    }
    let k8 = |k: usize| (k / 8) * 2 + (k % 4) / 2;
    let (p8, q8) = (k8(kp), k8(kq));
    let pr = [p.ref_id[0][p8], p.ref_id[1][p8]];
    let qr = [q.ref_id[0][q8], q.ref_id[1][q8]];
    let pm = [p.mv[0][kp], p.mv[1][kp]];
    let qm = [q.mv[0][kq], q.mv[1][kq]];
    let far = |a: [i16; 2], b: [i16; 2]| {
        (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= 4
    };
    let np = (pr[0] >= 0) as u8 + (pr[1] >= 0) as u8;
    let nq = (qr[0] >= 0) as u8 + (qr[1] >= 0) as u8;
    if np != nq {
        return 1;
    }
    if np == 1 {
        let (rp, mp) = if pr[0] >= 0 { (pr[0], pm[0]) } else { (pr[1], pm[1]) };
        let (rq, mq) = if qr[0] >= 0 { (qr[0], qm[0]) } else { (qr[1], qm[1]) };
        return (rp != rq || far(mp, mq)) as u8;
    }
    if np == 0 {
        return 0;
    }
    // Two vectors each.
    let same_set = (pr[0] == qr[0] && pr[1] == qr[1]) || (pr[0] == qr[1] && pr[1] == qr[0]);
    if !same_set {
        return 1;
    }
    if pr[0] != pr[1] {
        if pr[0] == qr[0] {
            (far(pm[0], qm[0]) || far(pm[1], qm[1])) as u8
        } else {
            (far(pm[0], qm[1]) || far(pm[1], qm[0])) as u8
        }
    } else {
        ((far(pm[0], qm[0]) || far(pm[1], qm[1])) && (far(pm[0], qm[1]) || far(pm[1], qm[0])))
            as u8
    }
}

/// Filter `n` lines across one edge. `at` indexes `q0` of the first line,
/// `across` steps from `p0` to `q0`, `along` from line to line. `bs` holds
/// the strength for each line.
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    data: &mut [u16],
    at: usize,
    across: usize,
    along: usize,
    n: usize,
    bs: &[u8],
    qp_av: i32,
    alpha_off: i32,
    beta_off: i32,
    bd: u32,
    luma: bool,
) {
    let index_a = (qp_av + alpha_off).clamp(0, 51) as usize;
    let index_b = (qp_av + beta_off).clamp(0, 51) as usize;
    let scale = 1 << (bd - 8);
    let alpha = ALPHA[index_a] * scale;
    let beta = BETA[index_b] * scale;
    if alpha == 0 || beta == 0 {
        return;
    }
    let max = (1i32 << bd) - 1;
    for line in 0..n {
        let s = bs[line];
        if s == 0 {
            continue;
        }
        let q0i = at + line * along;
        let px = |i: usize| q0i - across * (i + 1);
        let qx = |i: usize| q0i + across * i;
        let p0 = data[px(0)] as i32;
        let q0 = data[qx(0)] as i32;
        let p1 = data[px(1)] as i32;
        let q1 = data[qx(1)] as i32;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        if luma {
            let p2 = data[px(2)] as i32;
            let q2 = data[qx(2)] as i32;
            let ap = (p2 - p0).abs();
            let aq = (q2 - q0).abs();
            if s < 4 {
                let tc0 = TC0[index_a][s as usize] * scale;
                let tc = tc0 + (ap < beta) as i32 + (aq < beta) as i32;
                let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                data[px(0)] = (p0 + delta).clamp(0, max) as u16;
                data[qx(0)] = (q0 - delta).clamp(0, max) as u16;
                if ap < beta {
                    data[px(1)] =
                        (p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0)) as u16;
                }
                if aq < beta {
                    data[qx(1)] =
                        (q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0)) as u16;
                }
            } else {
                let p3 = data[px(3)] as i32;
                let q3 = data[qx(3)] as i32;
                let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
                if ap < beta && strong {
                    data[px(0)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u16;
                    data[px(1)] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u16;
                    data[px(2)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u16;
                } else {
                    data[px(0)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u16;
                }
                if aq < beta && strong {
                    data[qx(0)] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u16;
                    data[qx(1)] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u16;
                    data[qx(2)] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u16;
                } else {
                    data[qx(0)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u16;
                }
            }
        } else if s < 4 {
            let tc = TC0[index_a][s as usize] * scale + 1;
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            data[px(0)] = (p0 + delta).clamp(0, max) as u16;
            data[qx(0)] = (q0 - delta).clamp(0, max) as u16;
        } else {
            data[px(0)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u16;
            data[qx(0)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u16;
        }
    }
}

/// Deblock the whole picture in macroblock order.
pub fn deblock_picture(pic: &mut CurPic, bd_y: u32, bd_c: u32, chroma: bool) {
    let (w_mbs, h_mbs) = (pic.w_mbs, pic.h_mbs);
    let wl = pic.planes[0].width;
    let wc = pic.planes[1].width;
    for addr in 0..w_mbs * h_mbs {
        let q = pic.mbs[addr];
        if q.slice == u32::MAX {
            continue;
        }
        let sp = pic.slices[q.slice as usize];
        if sp.disable == 1 {
            continue;
        }
        let (mx, my) = (addr % w_mbs, addr / w_mbs);
        let left = (mx > 0).then(|| pic.mbs[addr - 1]).filter(|l| {
            l.slice != u32::MAX && !(sp.disable == 2 && l.slice != q.slice)
        });
        let top = (my > 0).then(|| pic.mbs[addr - w_mbs]).filter(|t| {
            t.slice != u32::MAX && !(sp.disable == 2 && t.slice != q.slice)
        });
        // Strengths: [vertical/horizontal][edge][segment].
        let mut bs = [[[0u8; 4]; 4]; 2];
        for e in 0..4 {
            if q.t8x8 && e % 2 == 1 {
                continue;
            }
            for s in 0..4 {
                // Vertical edge e, segment s.
                let kq = s * 4 + e;
                bs[0][e][s] = if e == 0 {
                    left.map_or(0, |l| strength(true, &l, s * 4 + 3, &q, kq))
                } else {
                    strength(false, &q, kq - 1, &q, kq)
                };
                let kq = e * 4 + s;
                bs[1][e][s] = if e == 0 {
                    top.map_or(0, |t| strength(true, &t, 12 + s, &q, kq))
                } else {
                    strength(false, &q, kq - 4, &q, kq)
                };
            }
        }
        let (a_off, b_off) = (sp.alpha, sp.beta);
        let (x0, y0) = (mx * 16, my * 16);
        // Luma.
        for dir in 0..2 {
            for e in 0..4 {
                let edge_bs = bs[dir][e];
                if edge_bs == [0; 4] {
                    continue;
                }
                let p = if e == 0 {
                    if dir == 0 {
                        left.unwrap()
                    } else {
                        top.unwrap()
                    }
                } else {
                    q
                };
                let qp_av = (p.qp as i32 + q.qp as i32 + 1) >> 1;
                let mut lines = [0u8; 16];
                for (i, l) in lines.iter_mut().enumerate() {
                    *l = edge_bs[i / 4];
                }
                let (at, across, along) = if dir == 0 {
                    (y0 * wl + x0 + e * 4, 1, wl)
                } else {
                    ((y0 + e * 4) * wl + x0, wl, 1)
                };
                filter_edge(
                    &mut pic.planes[0].data,
                    at,
                    across,
                    along,
                    16,
                    &lines,
                    qp_av,
                    a_off,
                    b_off,
                    bd_y,
                    true,
                );
            }
        }
        if !chroma {
            continue;
        }
        let (cx0, cy0) = (mx * 8, my * 8);
        for c in 0..2 {
            for dir in 0..2 {
                for ce in 0..2 {
                    let e = ce * 2;
                    let edge_bs = bs[dir][e];
                    if edge_bs == [0; 4] {
                        continue;
                    }
                    let p = if e == 0 {
                        if dir == 0 {
                            left.unwrap()
                        } else {
                            top.unwrap()
                        }
                    } else {
                        q
                    };
                    let qp_av = (p.qpc[c] as i32 + q.qpc[c] as i32 + 1) >> 1;
                    let mut lines = [0u8; 8];
                    for (i, l) in lines.iter_mut().enumerate() {
                        *l = edge_bs[i / 2];
                    }
                    let (at, across, along) = if dir == 0 {
                        (cy0 * wc + cx0 + ce * 4, 1, wc)
                    } else {
                        ((cy0 + ce * 4) * wc + cx0, wc, 1)
                    };
                    filter_edge(
                        &mut pic.planes[1 + c].data,
                        at,
                        across,
                        along,
                        8,
                        &lines,
                        qp_av,
                        a_off,
                        b_off,
                        bd_c,
                        false,
                    );
                }
            }
        }
    }
}
