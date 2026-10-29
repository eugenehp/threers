//! H.264 deblocking (§8.7), run over a whole decoded picture.

use super::slice::{CurPic, SliceDeblock};
use crate::decode::picture::Plane;
use super::types::MbInfo;
use crate::h264::deblock_tables::{ALPHA, BETA, TC0};

/// Boundary strength between 4×4 block `kp` of `p` and `kq` of `q`.
///
/// `intra_bs` is the strength an intra block gets on this edge (4 on frame
/// MB edges and on vertical MB edges of fields, else 3); `mvy` is the
/// vertical motion threshold (4, or 2 in field units).
fn strength(intra_bs: u8, p: &MbInfo, kp: usize, q: &MbInfo, kq: usize, mvy: i32) -> u8 {
    if p.kind.is_intra() || q.kind.is_intra() {
        return intra_bs;
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
        (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= mvy
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
/// `chroma_format_idc` 0 (no chroma), 1 (4:2:0) or 2 (4:2:2).
/// `field`: a field picture (horizontal intra MB edges take strength 3, and
/// vertical motion differences count from 2, both in field units).
pub fn deblock_picture(pic: &mut CurPic, bd_y: u32, bd_c: u32, chroma_format_idc: u32, field: bool) {
    let n = pic.w_mbs * pic.h_mbs;
    let f = Filter { bd_y, bd_c, chroma_format_idc, field };
    deblock_mbs(&mut pic.planes, &pic.mbs, &pic.slices, pic.w_mbs, 0..n, 2, 0, &f);
}

/// Macroblock row `my` of `pic` after only its own vertical-edge filtering,
/// as `[luma, cb, cr]` planes one macroblock row high.
///
/// ffmpeg intra-predicts from this state in one case: in a field picture
/// whose slice has `disable_deblocking_filter_idc` 2, `xchg_mb_border` looks
/// for the top-left macroblock at `mb_xy - 1 - mb_stride`, which is in the
/// other field (whose `slice_table` entries the field start cleared), so it
/// never swaps the unfiltered top-left samples back in. Horizontal edges do
/// not reach a row's last line before the next row is filtered, so vertical
/// edges are all that line has seen.
pub fn row_after_vertical(pic: &CurPic, my: usize, bd_y: u32, bd_c: u32, chroma_format_idc: u32) -> [Plane; 3] {
    let chh = if chroma_format_idc == 2 { 16 } else { 8 };
    let rows = |p: &Plane, h: usize| Plane {
        data: p.data[my * h * p.width..(my + 1) * h * p.width].to_vec(),
        width: p.width,
        height: h,
    };
    let mut planes = [rows(&pic.planes[0], 16), Plane::new(0, 0, 0), Plane::new(0, 0, 0)];
    if chroma_format_idc != 0 {
        planes[1] = rows(&pic.planes[1], chh);
        planes[2] = rows(&pic.planes[2], chh);
    }
    let f = Filter { bd_y, bd_c, chroma_format_idc, field: true };
    let w = pic.w_mbs;
    deblock_mbs(&mut planes, &pic.mbs, &pic.slices, w, my * w..(my + 1) * w, 1, my, &f);
    planes
}

#[derive(Clone, Copy)]
struct Filter {
    bd_y: u32,
    bd_c: u32,
    chroma_format_idc: u32,
    field: bool,
}

/// Filter the macroblocks in `addrs`, vertical edges only (`dirs` 1) or both
/// directions (2). The planes start at macroblock row `base_row`.
#[allow(clippy::too_many_arguments)]
fn deblock_mbs(
    planes: &mut [Plane; 3],
    mbs: &[MbInfo],
    slices: &[SliceDeblock],
    w_mbs: usize,
    addrs: std::ops::Range<usize>,
    dirs: usize,
    base_row: usize,
    f: &Filter,
) {
    let Filter { bd_y, bd_c, chroma_format_idc, field } = *f;
    let mvy = if field { 2 } else { 4 };
    let top_bs = if field { 3 } else { 4 };
    let chroma = chroma_format_idc != 0;
    let c422 = chroma_format_idc == 2;
    let wl = planes[0].width;
    let wc = planes[1].width;
    for addr in addrs {
        // By reference: an `MbInfo` is a few hundred bytes.
        let q = &mbs[addr];
        if q.slice == u32::MAX {
            continue;
        }
        let sp = slices[q.slice as usize];
        if sp.disable == 1 {
            continue;
        }
        let (mx, my) = (addr % w_mbs, addr / w_mbs);
        let left = (mx > 0).then(|| &mbs[addr - 1]).filter(|l| {
            l.slice != u32::MAX && !(sp.disable == 2 && l.slice != q.slice)
        });
        let top = (my > 0).then(|| &mbs[addr - w_mbs]).filter(|t| {
            t.slice != u32::MAX && !(sp.disable == 2 && t.slice != q.slice)
        });
        // Strengths: [vertical/horizontal][edge][segment].
        let mut bs = [[[0u8; 4]; 4]; 2];
        // With 4:2:2 the internal horizontal chroma edges are filtered even
        // where an 8×8 transform skips the luma edge (ffmpeg filter_mb_dir).
        let mut skip_luma = [[false; 4]; 2];
        for e in 0..4 {
            if q.t8x8 && e % 2 == 1 {
                skip_luma = [[skip_luma[0][0], true, skip_luma[0][2], true], [skip_luma[1][0], true, skip_luma[1][2], true]];
                if !c422 {
                    continue;
                }
            }
            for s in 0..4 {
                // Vertical edge e, segment s.
                let kq = s * 4 + e;
                bs[0][e][s] = if q.t8x8 && e % 2 == 1 {
                    0
                } else if e == 0 {
                    left.map_or(0, |l| strength(4, l, s * 4 + 3, q, kq, mvy))
                } else {
                    strength(3, q, kq - 1, q, kq, mvy)
                };
                let kq = e * 4 + s;
                bs[1][e][s] = if e == 0 {
                    top.map_or(0, |t| strength(top_bs, t, 12 + s, q, kq, mvy))
                } else {
                    strength(3, q, kq - 4, q, kq, mvy)
                };
            }
        }
        let (a_off, b_off) = (sp.alpha, sp.beta);
        let (x0, y0) = (mx * 16, (my - base_row) * 16);
        // Luma.
        for dir in 0..dirs {
            for e in 0..4 {
                let edge_bs = bs[dir][e];
                if edge_bs == [0; 4] || skip_luma[dir][e] {
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
                    &mut planes[0].data,
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
        let chh = if c422 { 16 } else { 8 };
        let (cx0, cy0) = (mx * 8, (my - base_row) * chh);
        for c in 0..2 {
            for dir in 0..dirs {
                // Vertical edges at chroma x 0, 4 (luma edges 0, 2); horizontal
                // ones at chroma y 0, 4 (4:2:0, luma 0, 2) or 0, 4, 8, 12
                // (4:2:2, luma 0..3).
                let n_edges = if c422 && dir == 1 { 4 } else { 2 };
                for ce in 0..n_edges {
                    let e = if n_edges == 4 { ce } else { ce * 2 };
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
                    // A vertical 4:2:2 edge has 16 lines, 4 per strength segment.
                    let len = if dir == 0 { chh } else { 8 };
                    let mut lines = [0u8; 16];
                    for (i, l) in lines.iter_mut().enumerate().take(len) {
                        *l = edge_bs[i * 4 / len];
                    }
                    let (at, across, along) = if dir == 0 {
                        (cy0 * wc + cx0 + ce * 4, 1, wc)
                    } else {
                        ((cy0 + ce * 4) * wc + cx0, wc, 1)
                    };
                    filter_edge(
                        &mut planes[1 + c].data,
                        at,
                        across,
                        along,
                        len,
                        &lines[..len],
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

/// Deblock an MBAFF frame (ffmpeg `ff_h264_filter_mb`), pair by pair in
/// decoding order. A field macroblock filters along its field's rows. Where
/// a pair meets one of the other kind the rules change:
///
/// - left edge: eight strengths and two QPs, one per macroblock of the left
///   pair, applied to alternate lines (frame macroblock) or to halves (field
///   macroblock); as in ffmpeg, if a call's first strength is below 4, lines
///   of strength 4 are skipped (`tc0_table[.][4]` reads the next row's -1),
///   and if it is 4, every line takes the strong filter;
/// - top edge of a frame macroblock under a field pair: filtered twice, once
///   per field, with strength 3 for intra and at least 1 otherwise;
/// - other horizontal edges between the kinds: strength 1 unless intra (3)
///   or coded (2).
pub fn deblock_mbaff(pic: &mut CurPic, bd_y: u32, bd_c: u32, chroma_format_idc: u32) {
    let chroma = chroma_format_idc != 0;
    let c422 = chroma_format_idc == 2;
    let chh = if c422 { 16 } else { 8 };
    let (w, h) = (pic.w_mbs, pic.h_mbs);
    let wl = pic.planes[0].width;
    let wc = pic.planes[1].width;
    let nz = |m: &MbInfo, row: usize, col: usize| (m.nz_mask >> (row * 4 + col)) & 1 != 0;
    for pair in 0..w * h / 2 {
        let (mx, pr) = (pair % w, pair / w);
        for bot in 0..2 {
            let my = 2 * pr + bot;
            let addr = my * w + mx;
            let q = pic.mbs[addr];
            if q.slice == u32::MAX {
                continue;
            }
            let sp = pic.slices[q.slice as usize];
            if sp.disable == 1 {
                continue;
            }
            let (a_off, b_off) = (sp.alpha, sp.beta);
            let avail = |m: &MbInfo| m.slice != u32::MAX && !(sp.disable == 2 && m.slice != q.slice);
            let field = q.field;
            let mvy = if field { 2 } else { 4 };
            // Picture row of the macroblock's view row 0, and the step.
            let (ly0, step) = if field { (32 * pr + bot, 2) } else { (16 * my, 1) };
            let cy0 = if field { 2 * chh * pr + bot } else { chh * my };
            let (x0, cx0) = (mx * 16, mx * 8);
            let ls = step * wl;
            let cs = step * wc;
            let qpc_av = |p: &MbInfo, c: usize| (p.qpc[c] as i32 + q.qpc[c] as i32 + 1) >> 1;
            let qp_av = |p: &MbInfo| (p.qp as i32 + q.qp as i32 + 1) >> 1;

            // ---- Left macroblock edge.
            let (lt, lb) = if mx > 0 {
                (pic.mbs[2 * pr * w + mx - 1], pic.mbs[(2 * pr + 1) * w + mx - 1])
            } else {
                (MbInfo::default(), MbInfo::default())
            };
            let left_ok = mx > 0 && avail(&lb);
            let mut left_bs = [0u8; 4];
            if left_ok && lt.field != field {
                let mut bs = [0u8; 8];
                for (i, b) in bs.iter_mut().enumerate() {
                    let j = if field { i >> 2 } else { i & 1 };
                    let mbn = if j == 0 { &lt } else { &lb };
                    *b = if q.kind.is_intra() || mbn.kind.is_intra() {
                        4
                    } else {
                        let row = if field { i & 3 } else { 2 * bot + (i >> 2) };
                        1 + (nz(&q, i >> 1, 0) || nz(mbn, row, 3)) as u8
                    };
                }
                for half in 0..2 {
                    let mbn = if half == 0 { &lt } else { &lb };
                    // Luma: 8 lines, one strength per 2.
                    let (at, along, first) = if field {
                        ((ly0 + 2 * 8 * half) * wl + x0, 2 * wl, 4 * half)
                    } else {
                        ((ly0 + half) * wl + x0, 2 * wl, half)
                    };
                    let bsi = |l: usize| if field { bs[4 * half + l / 2] } else { bs[half + 2 * (l / 2)] };
                    let mut lines = [0u8; 8];
                    for (l, v) in lines.iter_mut().enumerate() {
                        *v = mbaff_line_bs(bs[first], bsi(l));
                    }
                    filter_edge(&mut pic.planes[0].data, at, 1, along, 8, &lines, qp_av(mbn), a_off, b_off, bd_y, true);
                    if chroma {
                        let n = chh / 2;
                        let (cat, calong) = if field {
                            ((cy0 + 2 * n * half) * wc + cx0, 2 * wc)
                        } else {
                            ((cy0 + half) * wc + cx0, 2 * wc)
                        };
                        let per = n / 4;
                        let mut cl = [0u8; 8];
                        for (l, v) in cl.iter_mut().enumerate().take(n) {
                            let b = if field { bs[4 * half + l / per] } else { bs[half + 2 * (l / per)] };
                            *v = mbaff_line_bs(bs[first], b);
                        }
                        for c in 0..2 {
                            filter_edge(&mut pic.planes[1 + c].data, cat, 1, calong, n, &cl[..n], qpc_av(mbn, c), a_off, b_off, bd_c, false);
                        }
                    }
                }
            } else if left_ok {
                let p = pic.mbs[addr - 1];
                for (s, b) in left_bs.iter_mut().enumerate() {
                    *b = strength(4, &p, s * 4 + 3, &q, s * 4, mvy);
                }
            }

            // ---- Top macroblock edge.
            let mut top_bs = [0u8; 4];
            let mut top_p = None;
            let mut double_top = false;
            if !field && bot == 0 {
                if pr > 0 && avail(&pic.mbs[(my - 1) * w + mx]) {
                    let above_top = pic.mbs[(my - 2) * w + mx];
                    if above_top.field {
                        // Twice, as fields (filtered with the horizontal
                        // edges, after the vertical ones).
                        double_top = true;
                    } else {
                        top_p = Some(pic.mbs[(my - 1) * w + mx]);
                    }
                }
            } else if !field {
                // Bottom frame macroblock: the top one of its pair.
                top_p = Some(pic.mbs[addr - w]).filter(avail);
            } else if pr > 0 {
                // Field macroblock: the same field's row above. A top field
                // macroblock over a frame pair meets that pair's bottom MB.
                let above_top = pic.mbs[(2 * pr - 2) * w + mx];
                let a = if bot == 0 && above_top.field { (2 * pr - 2) * w + mx } else { (2 * pr - 1) * w + mx };
                top_p = Some(pic.mbs[a]).filter(avail);
            }
            if let Some(p) = top_p {
                for (s, b) in top_bs.iter_mut().enumerate() {
                    *b = if p.field != field {
                        if q.kind.is_intra() || p.kind.is_intra() {
                            3
                        } else if nz(&q, 0, s) || nz(&p, 3, s) {
                            2
                        } else {
                            1
                        }
                    } else {
                        strength(if field { 3 } else { 4 }, &p, 12 + s, &q, s, mvy)
                    };
                }
            }

            // ---- Strengths for the internal edges, and filtering.
            let mut bs = [[[0u8; 4]; 4]; 2];
            bs[0][0] = left_bs;
            bs[1][0] = top_bs;
            let mut skip_luma = [[false; 4]; 2];
            for e in 1..4 {
                if q.t8x8 && e % 2 == 1 {
                    skip_luma[0][e] = true;
                    skip_luma[1][e] = true;
                    if !c422 {
                        continue;
                    }
                }
                for s in 0..4 {
                    if !(q.t8x8 && e % 2 == 1) {
                        bs[0][e][s] = strength(3, &q, s * 4 + e - 1, &q, s * 4 + e, mvy);
                    }
                    bs[1][e][s] = strength(3, &q, (e - 1) * 4 + s, &q, e * 4 + s, mvy);
                }
            }
            let left_p = if left_ok && lt.field == field { Some(pic.mbs[addr - 1]) } else { None };
            for dir in 0..2 {
                if dir == 1 && double_top {
                    for j in 0..2 {
                        let mbn = pic.mbs[(my - 2 + j) * w + mx];
                        let mut bs = [0u8; 4];
                        for (i, b) in bs.iter_mut().enumerate() {
                            *b = if q.kind.is_intra() || mbn.kind.is_intra() {
                                3
                            } else {
                                1 + (nz(&q, 0, i) || nz(&mbn, 3, i)) as u8
                            };
                        }
                        let mut lines = [0u8; 16];
                        for (i, l) in lines.iter_mut().enumerate() {
                            *l = bs[i / 4];
                        }
                        let at = (16 * my + j) * wl + x0;
                        filter_edge(&mut pic.planes[0].data, at, 2 * wl, 1, 16, &lines, qp_av(&mbn), a_off, b_off, bd_y, true);
                        if chroma {
                            let mut cl = [0u8; 8];
                            for (i, l) in cl.iter_mut().enumerate() {
                                *l = bs[i / 2];
                            }
                            let cat = (chh * my + j) * wc + cx0;
                            for c in 0..2 {
                                filter_edge(&mut pic.planes[1 + c].data, cat, 2 * wc, 1, 8, &cl, qpc_av(&mbn, c), a_off, b_off, bd_c, false);
                            }
                        }
                    }
                }
                for e in 0..4 {
                    let edge_bs = bs[dir][e];
                    if edge_bs == [0; 4] || skip_luma[dir][e] {
                        continue;
                    }
                    let p = if e == 0 {
                        if dir == 0 { left_p } else { top_p }
                    } else {
                        Some(q)
                    };
                    let Some(p) = p else { continue };
                    let mut lines = [0u8; 16];
                    for (i, l) in lines.iter_mut().enumerate() {
                        *l = edge_bs[i / 4];
                    }
                    let (at, across, along) = if dir == 0 {
                        (ly0 * wl + x0 + e * 4, 1, ls)
                    } else {
                        ((ly0 + step * e * 4) * wl + x0, ls, 1)
                    };
                    filter_edge(&mut pic.planes[0].data, at, across, along, 16, &lines, qp_av(&p), a_off, b_off, bd_y, true);
                }
                if !chroma {
                    continue;
                }
                let n_edges = if c422 && dir == 1 { 4 } else { 2 };
                for ce in 0..n_edges {
                    let e = if n_edges == 4 { ce } else { ce * 2 };
                    let edge_bs = bs[dir][e];
                    if edge_bs == [0; 4] {
                        continue;
                    }
                    let p = if e == 0 {
                        if dir == 0 { left_p } else { top_p }
                    } else {
                        Some(q)
                    };
                    let Some(p) = p else { continue };
                    let len = if dir == 0 { chh } else { 8 };
                    let mut lines = [0u8; 16];
                    for (i, l) in lines.iter_mut().enumerate().take(len) {
                        *l = edge_bs[i * 4 / len];
                    }
                    let (at, across, along) = if dir == 0 {
                        (cy0 * wc + cx0 + ce * 4, 1, cs)
                    } else {
                        ((cy0 + step * ce * 4) * wc + cx0, cs, 1)
                    };
                    for c in 0..2 {
                        filter_edge(&mut pic.planes[1 + c].data, at, across, along, len, &lines[..len], qpc_av(&p, c), a_off, b_off, bd_c, false);
                    }
                }
            }
        }
    }
}

/// A line's strength in one of ffmpeg's MBAFF left-edge calls, whose first
/// strength `first` picks the filter for all eight lines.
fn mbaff_line_bs(first: u8, b: u8) -> u8 {
    if first == 4 {
        4
    } else if b == 4 {
        0
    } else {
        b
    }
}
