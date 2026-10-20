//! H.264 fractional sample interpolation (§8.4.2.2).
//!
//! Each block computes the half-sample planes its fraction needs once —
//! `b` (horizontal half), `h` (vertical half), `j` (centre) — and averages
//! from those, rather than re-filtering per output sample.

use crate::codec::decode::picture::Plane;

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

/// `out[c] = tap(row[c..c + 6])`, tap-major so it vectorises.
#[inline(always)]
fn h6(row: &[u16], out: &mut [i32]) {
    let w = out.len();
    for (c, o) in out.iter_mut().enumerate() {
        *o = row[c] as i32 + row[c + 5] as i32;
    }
    for (k, f) in [(1, -5), (2, 20), (3, 20), (4, -5)] {
        for (o, &v) in out.iter_mut().zip(&row[k..k + w]) {
            *o += f * v as i32;
        }
    }
}

/// `out[c] = tap(rows[0][c], …, rows[5][c])`.
#[inline(always)]
fn v6<T: Copy + Into<i32>>(rows: [&[T]; 6], out: &mut [i32]) {
    const F: [i32; 6] = [1, -5, 20, 20, -5, 1];
    out.fill(0);
    for (r, f) in rows.iter().zip(F) {
        for (o, &v) in out.iter_mut().zip(r.iter()) {
            *o += f * v.into();
        }
    }
}

/// Predict a `w × h` luma block (`w, h <= 16`) whose top-left full-sample
/// position is `(x, y)` with quarter-sample fraction `(xf, yf)`. Writes
/// `out[r * w + c]`.
#[allow(clippy::too_many_arguments)]
pub fn luma(p: &Plane, x: isize, y: isize, xf: usize, yf: usize, w: usize, h: usize, out: &mut [i32], bd: u32, win: &mut Vec<u16>) {
    let (src, stride, base) = view(p, x - 2, y - 2, w + 5, h + 5, win);
    let max = (1 << bd) - 1;
    // Row `r` of the window, `r` from -2; index `c + 2` is column `c`.
    let row = |r: isize| &src[(base as isize + (r + 2) * stride as isize) as usize..][..w + 5];
    let half = |v: i32| ((v + 16) >> 5).clamp(0, max);
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    let g = |r: usize| &row(r as isize)[2..2 + w + 1];

    if xf == 0 && yf == 0 {
        for r in 0..h {
            for (o, &v) in out[r * w..r * w + w].iter_mut().zip(g(r)) {
                *o = v as i32;
            }
        }
        return;
    }
    let need_b = xf != 0 && (yf != 2 || xf == 2);
    let need_h = yf != 0 && (xf != 2 || yf == 2);
    let need_j = (xf == 2 && yf != 0) || (yf == 2 && xf != 0);
    // Unrounded horizontal halves `b1`, rows -2..h+3 (index r + 2).
    let mut b1 = [0i32; 21 * 16];
    if need_b || need_j {
        let (lo, hi) = if need_j { (-2isize, h as isize + 3) } else { (0, h as isize + 1) };
        for r in lo..hi {
            let i = ((r + 2) as usize) * 16;
            h6(row(r), &mut b1[i..i + w]);
        }
    }
    let b = |r: usize, c: usize| half(b1[(r + 2) * 16 + c]);
    // Vertical halves `h`, rows 0..h, columns 0..=w.
    let mut hh = [0i32; 16 * 17];
    if need_h {
        for r in 0..h {
            let ri = r as isize;
            let rows = [-2, -1, 0, 1, 2, 3].map(|k| &row(ri + k)[2..2 + w + 1]);
            let o = &mut hh[r * 17..r * 17 + w + 1];
            v6(rows, o);
            for v in o.iter_mut() {
                *v = half(*v);
            }
        }
    }
    let hv = |r: usize, c: usize| hh[r * 17 + c];
    let mut j = [0i32; 16 * 16];
    if need_j {
        for r in 0..h {
            let rows = [0, 1, 2, 3, 4, 5].map(|k| &b1[(r + k) * 16..(r + k) * 16 + w]);
            let o = &mut j[r * 16..r * 16 + w];
            v6(rows, o);
            for v in o.iter_mut() {
                *v = ((*v + 512) >> 10).clamp(0, max);
            }
        }
    }
    let jv = |r: usize, c: usize| j[r * 16 + c];
    for r in 0..h {
        let o = &mut out[r * w..r * w + w];
        let gr = g(r);
        match (xf, yf) {
            (1, 0) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(gr[c] as i32, b(r, c))),
            (2, 0) => o.iter_mut().enumerate().for_each(|(c, o)| *o = b(r, c)),
            (3, 0) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(b(r, c), gr[c + 1] as i32)),
            (0, 1) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(gr[c] as i32, hv(r, c))),
            (0, 2) => o.iter_mut().enumerate().for_each(|(c, o)| *o = hv(r, c)),
            (0, 3) => {
                let gn = g(r + 1);
                o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(hv(r, c), gn[c] as i32))
            }
            (2, 2) => o.iter_mut().enumerate().for_each(|(c, o)| *o = jv(r, c)),
            (2, 1) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(b(r, c), jv(r, c))),
            (2, 3) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(jv(r, c), b(r + 1, c))),
            (1, 2) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(hv(r, c), jv(r, c))),
            (3, 2) => o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(jv(r, c), hv(r, c + 1))),
            (xf, yf) => {
                let (rb, ch) = ((yf == 3) as usize, (xf == 3) as usize);
                o.iter_mut().enumerate().for_each(|(c, o)| *o = avg(b(r + rb, c), hv(r, c + ch)))
            }
        }
    }
}

/// Predict a `w × h` chroma block at full-sample `(x, y)` with eighth-sample
/// fraction `(xf, yf)`.
#[allow(clippy::too_many_arguments)]
pub fn chroma(p: &Plane, x: isize, y: isize, xf: i32, yf: i32, w: usize, h: usize, out: &mut [i32], win: &mut Vec<u16>) {
    let (src, stride, base) = view(p, x, y, w + 1, h + 1, win);
    let (wa, wb, wc, wd) = ((8 - xf) * (8 - yf), xf * (8 - yf), (8 - xf) * yf, xf * yf);
    for r in 0..h {
        let r0 = &src[base + r * stride..][..w + 1];
        let r1 = &src[base + (r + 1) * stride..][..w + 1];
        for c in 0..w {
            out[r * w + c] = (wa * r0[c] as i32 + wb * r0[c + 1] as i32 + wc * r1[c] as i32 + wd * r1[c + 1] as i32 + 32) >> 6;
        }
    }
}
