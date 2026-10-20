//! HEVC fractional sample interpolation and weighted sample prediction
//! (§8.5.3.3.3, §8.5.3.3.4). Predictions are kept at 14-bit precision until
//! the weighting step.

use super::tables::{CHROMA_FILTER, LUMA_FILTER};
use crate::codec::decode::picture::Plane;

/// Reusable buffers, so interpolation allocates nothing per block.
#[derive(Default)]
pub struct Scratch {
    win: Vec<u16>,
    tmp: Vec<i32>,
}

/// A view of the `ww × wh` window at `(x0, y0)`: straight into the plane
/// when it lies inside, otherwise an edge-clamped copy in `win`.
fn view<'a>(p: &'a Plane, x0: isize, y0: isize, ww: usize, wh: usize, win: &'a mut Vec<u16>) -> (&'a [u16], usize, usize) {
    if x0 >= 0 && y0 >= 0 && x0 as usize + ww <= p.width && y0 as usize + wh <= p.height {
        return (&p.data, p.width, y0 as usize * p.width + x0 as usize);
    }
    win.clear();
    win.reserve(ww * wh);
    for r in 0..wh {
        let y = (y0 + r as isize).clamp(0, p.height as isize - 1) as usize;
        let row = &p.data[y * p.width..(y + 1) * p.width];
        for c in 0..ww {
            win.push(row[(x0 + c as isize).clamp(0, p.width as isize - 1) as usize]);
        }
    }
    (win, ww, 0)
}

#[allow(clippy::too_many_arguments)]
fn interp_n<const T: usize>(
    p: &Plane,
    x: isize,
    y: isize,
    fx: &[i32; T],
    fy: &[i32; T],
    xf: usize,
    yf: usize,
    w: usize,
    h: usize,
    bd: u32,
    out: &mut [i16],
    s: &mut Scratch,
) {
    let half = T / 2 - 1;
    let shift1 = (bd as i32 - 8).min(4);
    let shift3 = (14 - bd as i32).max(2);
    let (ww, wh) = (w + T - 1, h + T - 1);
    let (src, stride, base) = view(p, x - half as isize, y - half as isize, ww, wh, &mut s.win);
    match (xf, yf) {
        (0, 0) => {
            for r in 0..h {
                let row = &src[base + (r + half) * stride + half..][..w];
                for (o, &v) in out[r * w..r * w + w].iter_mut().zip(row) {
                    *o = ((v as i32) << shift3) as i16;
                }
            }
        }
        (_, 0) => {
            let mut acc = [0i32; 64];
            for r in 0..h {
                let row = &src[base + (r + half) * stride..][..ww];
                hfilter(fx, row, &mut acc[..w]);
                for (o, &a) in out[r * w..r * w + w].iter_mut().zip(acc.iter()) {
                    *o = (a >> shift1) as i16;
                }
            }
        }
        (0, _) => {
            let mut acc = [0i32; 64];
            for r in 0..h {
                let acc = &mut acc[..w];
                acc.fill(0);
                for (k, &f) in fy.iter().enumerate() {
                    let row = &src[base + (r + k) * stride + half..][..w];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a += f * v as i32;
                    }
                }
                for (o, &a) in out[r * w..r * w + w].iter_mut().zip(acc.iter()) {
                    *o = (a >> shift1) as i16;
                }
            }
        }
        _ => {
            s.tmp.clear();
            s.tmp.resize(wh * w, 0);
            for r in 0..wh {
                let row = &src[base + r * stride..][..ww];
                let t = &mut s.tmp[r * w..r * w + w];
                hfilter(fx, row, t);
                for v in t.iter_mut() {
                    *v >>= shift1;
                }
            }
            let mut acc = [0i32; 64];
            for r in 0..h {
                let acc = &mut acc[..w];
                acc.fill(0);
                for (k, &f) in fy.iter().enumerate() {
                    let row = &s.tmp[(r + k) * w..(r + k) * w + w];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a += f * v;
                    }
                }
                for (o, &a) in out[r * w..r * w + w].iter_mut().zip(acc.iter()) {
                    *o = (a >> 6) as i16;
                }
            }
        }
    }
}

/// `acc[c] = Σ f[k] · row[c + k]`, tap-major so each pass over `acc` is a
/// plain multiply-add the compiler can vectorise.
#[inline(always)]
fn hfilter<const T: usize>(f: &[i32; T], row: &[u16], acc: &mut [i32]) {
    let w = acc.len();
    acc.fill(0);
    for (k, &fk) in f.iter().enumerate() {
        if fk == 0 {
            continue;
        }
        for (a, &v) in acc.iter_mut().zip(&row[k..k + w]) {
            *a += fk * v as i32;
        }
    }
}

/// Interpolate a `w × h` block whose full-sample position is `(x, y)` and
/// fraction `(xf, yf)` (quarter samples for luma, eighths for chroma).
#[allow(clippy::too_many_arguments)]
pub fn interp(
    p: &Plane,
    x: isize,
    y: isize,
    xf: usize,
    yf: usize,
    w: usize,
    h: usize,
    bd: u32,
    luma: bool,
    out: &mut [i16],
    s: &mut Scratch,
) {
    if luma {
        interp_n(p, x, y, &LUMA_FILTER[xf], &LUMA_FILTER[yf], xf, yf, w, h, bd, out, s);
    } else {
        interp_n(p, x, y, &CHROMA_FILTER[xf], &CHROMA_FILTER[yf], xf, yf, w, h, bd, out, s);
    }
}

/// How two (or one) 14-bit predictions become samples.
#[derive(Clone, Copy, Debug)]
pub enum Weighting {
    Default,
    /// `(w0, o0, w1, o1, log2_denom)`, offsets already at the bit depth.
    Explicit(i32, i32, i32, i32, u32),
}

/// Combine predictions and store the `w × h` block at `(x0, y0)` of `dst`.
#[allow(clippy::too_many_arguments)]
pub fn put(
    dst: &mut Plane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    p0: Option<&[i16]>,
    p1: Option<&[i16]>,
    wt: Weighting,
    bd: u32,
) {
    let max = (1i32 << bd) - 1;
    let shift1 = 14 - bd as i32;
    // Clip to the picture: blocks along the right and bottom edges can
    // extend past it when the size is not a multiple of the minimum CU.
    let wc = w.min(dst.width.saturating_sub(x0));
    let hc = h.min(dst.height.saturating_sub(y0));
    let stride = dst.width;
    let rows = dst.data.chunks_exact_mut(stride).skip(y0).take(hc);
    match (p0, p1, wt) {
        (Some(a), Some(b), Weighting::Default) => {
            let shift2 = 15 - bd as i32;
            let off = 1 << (shift2 - 1);
            for (r, row) in rows.enumerate() {
                let (a, b) = (&a[r * w..][..wc], &b[r * w..][..wc]);
                for ((d, &p), &q) in row[x0..x0 + wc].iter_mut().zip(a).zip(b) {
                    *d = ((p as i32 + q as i32 + off) >> shift2).clamp(0, max) as u16;
                }
            }
        }
        (Some(a), Some(b), Weighting::Explicit(w0, o0, w1, o1, denom)) => {
            let log2wd = denom as i32 + shift1;
            let round = (o0 + o1 + 1) << log2wd;
            for (r, row) in rows.enumerate() {
                let (a, b) = (&a[r * w..][..wc], &b[r * w..][..wc]);
                for ((d, &p), &q) in row[x0..x0 + wc].iter_mut().zip(a).zip(b) {
                    let v = (p as i32 * w0 + q as i32 * w1 + round) >> (log2wd + 1);
                    *d = v.clamp(0, max) as u16;
                }
            }
        }
        (Some(a), None, wt) | (None, Some(a), wt) => {
            let explicit = match (wt, p0.is_some()) {
                (Weighting::Explicit(w0, o0, _, _, d), true) => Some((w0, o0, d)),
                (Weighting::Explicit(_, _, w1, o1, d), false) => Some((w1, o1, d)),
                _ => None,
            };
            match explicit {
                None => {
                    let off = 1 << (shift1 - 1);
                    for (r, row) in rows.enumerate() {
                        for (d, &p) in row[x0..x0 + wc].iter_mut().zip(&a[r * w..][..wc]) {
                            *d = ((p as i32 + off) >> shift1).clamp(0, max) as u16;
                        }
                    }
                }
                Some((wl, ol, d)) => {
                    let log2wd = d as i32 + shift1;
                    for (r, row) in rows.enumerate() {
                        for (d, &p) in row[x0..x0 + wc].iter_mut().zip(&a[r * w..][..wc]) {
                            let p = p as i32;
                            let v = if log2wd >= 1 {
                                ((p * wl + (1 << (log2wd - 1))) >> log2wd) + ol
                            } else {
                                p * wl + ol
                            };
                            *d = v.clamp(0, max) as u16;
                        }
                    }
                }
            }
        }
        _ => {}
    }
}
