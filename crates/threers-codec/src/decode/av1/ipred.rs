//! Intra prediction (dav1d `src/ipred_prepare_tmpl.c`, `src/ipred_tmpl.c`).
//!
//! Edges live in a buffer indexed around `TL` (dav1d's `topleft` pointer):
//! `[TL]` top-left, `[TL+1..]` top then top-right, `[TL-1..]` left (top to
//! bottom) then bottom-left.

use super::intra_edge::{EDGE_I444_LEFT_HAS_BOTTOM, EDGE_I444_TOP_HAS_RIGHT};
use super::levels::*;
use super::tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};

pub const TL: usize = 128;
pub const EDGE_LEN: usize = 257;
pub const ANGLE_SMOOTH_EDGE_FLAG: i32 = 512;

const MODE_TO_ANGLE: [i32; 8] = [90, 180, 45, 135, 113, 157, 203, 67];

/// (needs_left, needs_top, needs_topleft, needs_topright, needs_bottomleft)
fn needs(mode: usize) -> (bool, bool, bool, bool, bool) {
    match mode {
        DC_PRED => (true, true, false, false, false),
        VERT_PRED => (false, true, false, false, false),
        HOR_PRED => (true, false, false, false, false),
        LEFT_DC_PRED => (true, false, false, false, false),
        TOP_DC_PRED => (false, true, false, false, false),
        DC_128_PRED => (false, false, false, false, false),
        Z1_PRED => (false, true, true, true, false),
        Z2_PRED => (true, true, true, false, false),
        Z3_PRED => (true, false, true, false, true),
        SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED => (true, true, false, false, false),
        PAETH_PRED => (true, true, true, false, false),
        FILTER_PRED => (true, true, true, false, false),
        _ => (false, false, false, false, false),
    }
}

/// Where the top edge comes from: the frame row above, or the pre-filter
/// copy of the superblock row above (`ipred_edge`).
pub enum Top<'a> {
    Frame,
    Sb(&'a [u16]),
}

/// `dav1d_prepare_intra_edges`. `x, y, w, h, tw, th` in 4px units of the
/// plane; `dst` is the plane, `off` the block origin.
#[allow(clippy::too_many_arguments)]
pub fn prepare_intra_edges(
    x: i32,
    have_left: bool,
    y: i32,
    have_top: bool,
    w: i32,
    h: i32,
    edge_flags: u8,
    dst: &[u16],
    off: usize,
    stride: usize,
    top_sb: Top,
    mut mode: usize,
    angle: &mut i32,
    tw: i32,
    th: i32,
    filter_edge: bool,
    out: &mut [u16],
    bitdepth: u32,
) -> usize {
    match mode {
        VERT_PRED..=VERT_LEFT_PRED => {
            *angle = MODE_TO_ANGLE[mode - VERT_PRED] + 3 * *angle;
            mode = if *angle <= 90 {
                if *angle < 90 && have_top {
                    Z1_PRED
                } else {
                    VERT_PRED
                }
            } else if *angle < 180 {
                Z2_PRED
            } else if *angle > 180 && have_left {
                Z3_PRED
            } else {
                HOR_PRED
            };
        }
        DC_PRED => {
            mode = match (have_left, have_top) {
                (false, false) => DC_128_PRED,
                (false, true) => TOP_DC_PRED,
                (true, false) => LEFT_DC_PRED,
                (true, true) => DC_PRED,
            }
        }
        PAETH_PRED => {
            mode = match (have_left, have_top) {
                (false, false) => DC_128_PRED,
                (false, true) => VERT_PRED,
                (true, false) => HOR_PRED,
                (true, true) => PAETH_PRED,
            }
        }
        _ => {}
    }
    let (n_left, n_top, n_tl, n_tr, n_bl) = needs(mode);
    // The top row as (slice, index of the block's first top sample).
    let top_src: Option<(&[u16], isize)> = if have_top && (n_top || n_tl || (n_left && !have_left)) {
        Some(match top_sb {
            Top::Sb(e) => (e, x as isize * 4),
            Top::Frame => (dst, off as isize - stride as isize),
        })
    } else {
        None
    };
    let tp = |i: isize| -> u16 {
        let (s, o) = top_src.unwrap();
        s[(o + i) as usize]
    };
    let px = |dx: isize, dy: isize| -> u16 { dst[(off as isize + dy * stride as isize + dx) as usize] };

    if n_left {
        let sz = (th << 2) as usize;
        let left = TL - sz;
        if have_left {
            let px_have = sz.min(((h - y) << 2) as usize);
            for i in 0..px_have {
                out[left + sz - 1 - i] = px(-1, i as isize);
            }
            if px_have < sz {
                let v = out[left + sz - px_have];
                for o in &mut out[left..left + sz - px_have] {
                    *o = v;
                }
            }
        } else {
            let v = if have_top { tp(0) } else { (((1 << bitdepth) >> 1) + 1) as u16 };
            for o in &mut out[left..left + sz] {
                *o = v;
            }
        }
        if n_bl {
            let have_bl = if !have_left || y + th >= h { false } else { edge_flags & EDGE_I444_LEFT_HAS_BOTTOM != 0 };
            if have_bl {
                let px_have = sz.min(((h - y - th) << 2) as usize);
                for i in 0..px_have {
                    out[left - (i + 1)] = px(-1, (sz + i) as isize);
                }
                if px_have < sz {
                    let v = out[left - px_have];
                    for o in &mut out[left - sz..left - px_have] {
                        *o = v;
                    }
                }
            } else {
                let v = out[left];
                for o in &mut out[left - sz..left] {
                    *o = v;
                }
            }
        }
    }
    if n_top {
        let sz = (tw << 2) as usize;
        let top = TL + 1;
        if have_top {
            let px_have = sz.min(((w - x) << 2) as usize);
            for i in 0..px_have {
                out[top + i] = tp(i as isize);
            }
            if px_have < sz {
                let v = out[top + px_have - 1];
                for o in &mut out[top + px_have..top + sz] {
                    *o = v;
                }
            }
        } else {
            let v = if have_left { px(-1, 0) } else { (((1 << bitdepth) >> 1) - 1) as u16 };
            for o in &mut out[top..top + sz] {
                *o = v;
            }
        }
        if n_tr {
            let have_tr = if !have_top || x + tw >= w { false } else { edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0 };
            if have_tr {
                let px_have = sz.min(((w - x - tw) << 2) as usize);
                for i in 0..px_have {
                    out[top + sz + i] = tp((sz + i) as isize);
                }
                if px_have < sz {
                    let v = out[top + sz + px_have - 1];
                    for o in &mut out[top + sz + px_have..top + 2 * sz] {
                        *o = v;
                    }
                }
            } else {
                let v = out[top + sz - 1];
                for o in &mut out[top + sz..top + 2 * sz] {
                    *o = v;
                }
            }
        }
    }
    if n_tl {
        out[TL] = if have_left {
            if have_top {
                tp(-1)
            } else {
                px(-1, 0)
            }
        } else if have_top {
            tp(0)
        } else {
            ((1 << bitdepth) >> 1) as u16
        };
        if mode == Z2_PRED && tw + th >= 6 && filter_edge {
            out[TL] = ((((out[TL - 1] as i32 + out[TL + 1] as i32) * 5) + out[TL] as i32 * 6 + 8) >> 4) as u16;
        }
    }
    mode
}

#[inline]
fn e(edge: &[u16], i: isize) -> i32 {
    edge[(TL as isize + i) as usize] as i32
}

fn splat(dst: &mut [u16], off: usize, stride: usize, w: usize, h: usize, v: u16) {
    for y in 0..h {
        for d in &mut dst[off + y * stride..off + y * stride + w] {
            *d = v;
        }
    }
}

fn dc_gen_top(edge: &[u16], w: usize) -> u32 {
    let mut dc = (w >> 1) as u32;
    for i in 0..w {
        dc += e(edge, 1 + i as isize) as u32;
    }
    dc >> w.trailing_zeros()
}

fn dc_gen_left(edge: &[u16], h: usize) -> u32 {
    let mut dc = (h >> 1) as u32;
    for i in 0..h {
        dc += e(edge, -(1 + i as isize)) as u32;
    }
    dc >> h.trailing_zeros()
}

fn dc_gen(edge: &[u16], w: usize, h: usize, hbd: bool) -> u32 {
    let mut dc = ((w + h) >> 1) as u32;
    for i in 0..w {
        dc += e(edge, i as isize + 1) as u32;
    }
    for i in 0..h {
        dc += e(edge, -(i as isize + 1)) as u32;
    }
    dc >>= (w + h).trailing_zeros();
    if w != h {
        let (m12, m14, sh) = if hbd { (0xAAAB, 0x6667, 17) } else { (0x5556, 0x3334, 16) };
        dc *= if w > h * 2 || h > w * 2 { m14 } else { m12 };
        dc >>= sh;
    }
    dc
}

fn get_filter_strength(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
        return 3;
    }
    0
}

/// `filter_edge`: `inp(i)` reads the input edge at index `i`.
#[allow(clippy::too_many_arguments)]
fn filter_edge(out: &mut [u16], o: usize, sz: i32, lim_from: i32, lim_to: i32, inp: &dyn Fn(i32) -> i32, from: i32, to: i32, strength: i32) {
    const K: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let mut i = 0;
    while i < sz.min(lim_from) {
        out[o + i as usize] = inp(i.clamp(from, to - 1)) as u16;
        i += 1;
    }
    while i < lim_to.min(sz) {
        let mut s = 0;
        for j in 0..5 {
            s += inp((i - 2 + j).clamp(from, to - 1)) * K[strength as usize - 1][j as usize];
        }
        out[o + i as usize] = ((s + 8) >> 4) as u16;
        i += 1;
    }
    while i < sz {
        out[o + i as usize] = inp(i.clamp(from, to - 1)) as u16;
        i += 1;
    }
}

fn get_upsample(wh: i32, angle: i32, is_sm: bool) -> bool {
    angle < 40 && wh <= 16 >> is_sm as i32
}

fn upsample_edge(out: &mut [u16], o: usize, hsz: i32, inp: &dyn Fn(i32) -> i32, from: i32, to: i32, max: i32) {
    const K: [i32; 4] = [-1, 9, 9, -1];
    let mut i = 0;
    while i < hsz - 1 {
        out[o + (i * 2) as usize] = inp(i.clamp(from, to - 1)) as u16;
        let mut s = 0;
        for j in 0..4 {
            s += inp((i + j - 1).clamp(from, to - 1)) * K[j as usize];
        }
        out[o + (i * 2 + 1) as usize] = ((s + 8) >> 4).clamp(0, max) as u16;
        i += 1;
    }
    out[o + (i * 2) as usize] = inp(i.clamp(from, to - 1)) as u16;
}

/// `intra_pred[mode]`.
#[allow(clippy::too_many_arguments)]
pub fn intra_pred(mode: usize, dst: &mut [u16], off: usize, stride: usize, edge: &[u16], w: usize, h: usize, angle: i32, max_w: i32, max_h: i32, bitdepth_max: i32) {
    let hbd = bitdepth_max != 255;
    match mode {
        DC_PRED => splat(dst, off, stride, w, h, dc_gen(edge, w, h, hbd) as u16),
        DC_128_PRED => splat(dst, off, stride, w, h, ((bitdepth_max + 1) >> 1) as u16),
        TOP_DC_PRED => splat(dst, off, stride, w, h, dc_gen_top(edge, w) as u16),
        LEFT_DC_PRED => splat(dst, off, stride, w, h, dc_gen_left(edge, h) as u16),
        HOR_PRED => {
            for y in 0..h {
                let v = e(edge, -(1 + y as isize)) as u16;
                for d in &mut dst[off + y * stride..off + y * stride + w] {
                    *d = v;
                }
            }
        }
        VERT_PRED => {
            for y in 0..h {
                for x in 0..w {
                    dst[off + y * stride + x] = edge[TL + 1 + x];
                }
            }
        }
        PAETH_PRED => {
            let tl = e(edge, 0);
            for y in 0..h {
                let left = e(edge, -(y as isize + 1));
                for x in 0..w {
                    let top = e(edge, 1 + x as isize);
                    let base = left + top - tl;
                    let ld = (left - base).abs();
                    let td = (top - base).abs();
                    let tld = (tl - base).abs();
                    dst[off + y * stride + x] = if ld <= td && ld <= tld {
                        left
                    } else if td <= tld {
                        top
                    } else {
                        tl
                    } as u16;
                }
            }
        }
        SMOOTH_PRED => {
            let wh = &SM_WEIGHTS[w..];
            let wv = &SM_WEIGHTS[h..];
            let right = e(edge, w as isize);
            let bottom = e(edge, -(h as isize));
            for y in 0..h {
                for x in 0..w {
                    let p = wv[y] as i32 * e(edge, 1 + x as isize)
                        + (256 - wv[y] as i32) * bottom
                        + wh[x] as i32 * e(edge, -(1 + y as isize))
                        + (256 - wh[x] as i32) * right;
                    dst[off + y * stride + x] = ((p + 256) >> 9) as u16;
                }
            }
        }
        SMOOTH_V_PRED => {
            let wv = &SM_WEIGHTS[h..];
            let bottom = e(edge, -(h as isize));
            for y in 0..h {
                for x in 0..w {
                    let p = wv[y] as i32 * e(edge, 1 + x as isize) + (256 - wv[y] as i32) * bottom;
                    dst[off + y * stride + x] = ((p + 128) >> 8) as u16;
                }
            }
        }
        SMOOTH_H_PRED => {
            let wh = &SM_WEIGHTS[w..];
            let right = e(edge, w as isize);
            for y in 0..h {
                for x in 0..w {
                    let p = wh[x] as i32 * e(edge, -(y as isize + 1)) + (256 - wh[x] as i32) * right;
                    dst[off + y * stride + x] = ((p + 128) >> 8) as u16;
                }
            }
        }
        Z1_PRED => z1(dst, off, stride, edge, w as i32, h as i32, angle, bitdepth_max),
        Z2_PRED => z2(dst, off, stride, edge, w as i32, h as i32, angle, max_w, max_h, bitdepth_max),
        Z3_PRED => z3(dst, off, stride, edge, w as i32, h as i32, angle, bitdepth_max),
        FILTER_PRED => filter_pred(dst, off, stride, edge, w, h, angle, bitdepth_max),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn z1(dst: &mut [u16], off: usize, stride: usize, edge: &[u16], w: i32, h: i32, mut angle: i32, max: i32) {
    let is_sm = (angle >> 9) & 1 != 0;
    let filt = angle >> 10 != 0;
    angle &= 511;
    let mut dx = DR_INTRA_DERIVATIVE[(angle >> 1) as usize] as i32;
    let mut top_out = [0u16; 128];
    let inp = |i: i32| e(edge, 1 + i as isize);
    let upsample_above = filt && get_upsample(w + h, 90 - angle, is_sm);
    // `top` as (buffer is top_out?, base index into the source).
    let use_out;
    let max_base_x;
    if upsample_above {
        upsample_edge(&mut top_out, 0, w + h, &inp, -1, w + w.min(h), max);
        use_out = true;
        max_base_x = 2 * (w + h) - 2;
        dx <<= 1;
    } else {
        let fs = if filt { get_filter_strength(w + h, 90 - angle, is_sm) } else { 0 };
        if fs != 0 {
            filter_edge(&mut top_out, 0, w + h, 0, w + h, &inp, -1, w + w.min(h), fs);
            use_out = true;
            max_base_x = w + h - 1;
        } else {
            use_out = false;
            max_base_x = w + w.min(h) - 1;
        }
    }
    let top = |i: i32| if use_out { top_out[i as usize] as i32 } else { e(edge, 1 + i as isize) };
    let base_inc = 1 + upsample_above as i32;
    let mut xpos = dx;
    for y in 0..h as usize {
        let frac = xpos & 0x3e;
        let mut base = xpos >> 6;
        for x in 0..w as usize {
            if base < max_base_x {
                let v = top(base) * (64 - frac) + top(base + 1) * frac;
                dst[off + y * stride + x] = ((v + 32) >> 6) as u16;
            } else {
                let v = top(max_base_x) as u16;
                for d in &mut dst[off + y * stride + x..off + y * stride + w as usize] {
                    *d = v;
                }
                break;
            }
            base += base_inc;
        }
        xpos += dx;
    }
}

#[allow(clippy::too_many_arguments)]
fn z2(dst: &mut [u16], off: usize, stride: usize, edge_in: &[u16], w: i32, h: i32, mut angle: i32, max_w: i32, max_h: i32, max: i32) {
    let is_sm = (angle >> 9) & 1 != 0;
    let filt = angle >> 10 != 0;
    angle &= 511;
    let mut dy = DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize] as i32;
    let mut dx = DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize] as i32;
    let upsample_left = filt && get_upsample(w + h, 180 - angle, is_sm);
    let upsample_above = filt && get_upsample(w + h, angle - 90, is_sm);
    let mut edge = [0u16; 64 + 64 + 1];
    const T: usize = 64;
    let inp_top = |i: i32| e(edge_in, i as isize);
    if upsample_above {
        upsample_edge(&mut edge, T, w + 1, &inp_top, 0, w + 1, max);
        dx <<= 1;
    } else {
        let fs = if filt { get_filter_strength(w + h, angle - 90, is_sm) } else { 0 };
        if fs != 0 {
            let inp = |i: i32| e(edge_in, 1 + i as isize);
            filter_edge(&mut edge, T + 1, w, 0, max_w, &inp, -1, w, fs);
        } else {
            for i in 0..w as usize {
                edge[T + 1 + i] = edge_in[TL + 1 + i];
            }
        }
    }
    let inp_left = |i: i32| e(edge_in, i as isize - h as isize);
    if upsample_left {
        upsample_edge(&mut edge, T - (h as usize) * 2, h + 1, &inp_left, 0, h + 1, max);
        dy <<= 1;
    } else {
        let fs = if filt { get_filter_strength(w + h, 180 - angle, is_sm) } else { 0 };
        if fs != 0 {
            filter_edge(&mut edge, T - h as usize, h, h - max_h, h, &inp_left, 0, h + 1, fs);
        } else {
            for i in 0..h as usize {
                edge[T - h as usize + i] = edge_in[TL - h as usize + i];
            }
        }
    }
    edge[T] = edge_in[TL];
    let base_inc_x = 1 + upsample_above as i32;
    let left = T as isize - (1 + upsample_left as isize);
    let mut xpos = ((1 + upsample_above as i32) << 6) - dx;
    for y in 0..h {
        let mut base_x = xpos >> 6;
        let frac_x = xpos & 0x3e;
        let mut ypos = (y << (6 + upsample_left as i32)) - dy;
        for x in 0..w {
            
            let v = if base_x >= 0 {
                edge[T + base_x as usize] as i32 * (64 - frac_x) + edge[T + base_x as usize + 1] as i32 * frac_x
            } else {
                let base_y = ypos >> 6;
                let frac_y = ypos & 0x3e;
                edge[(left - base_y as isize) as usize] as i32 * (64 - frac_y) + edge[(left - (base_y as isize + 1)) as usize] as i32 * frac_y
            };
            dst[off + y as usize * stride + x as usize] = ((v + 32) >> 6) as u16;
            base_x += base_inc_x;
            ypos -= dy;
        }
        xpos -= dx;
    }
}

#[allow(clippy::too_many_arguments)]
fn z3(dst: &mut [u16], off: usize, stride: usize, edge: &[u16], w: i32, h: i32, mut angle: i32, max: i32) {
    let is_sm = (angle >> 9) & 1 != 0;
    let filt = angle >> 10 != 0;
    angle &= 511;
    let mut dy = DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize] as i32;
    let mut left_out = [0u16; 128];
    let inp = |i: i32| e(edge, i as isize - (w + h) as isize);
    let upsample_left = filt && get_upsample(w + h, angle - 180, is_sm);
    // `left[-k]`: either left_out[base - k] or edge[TL - 1 - k].
    let (use_out, lbase, max_base_y);
    if upsample_left {
        upsample_edge(&mut left_out, 0, w + h, &inp, (w - h).max(0), w + h + 1, max);
        use_out = true;
        lbase = 2 * (w + h) - 2;
        max_base_y = 2 * (w + h) - 2;
        dy <<= 1;
    } else {
        let fs = if filt { get_filter_strength(w + h, angle - 180, is_sm) } else { 0 };
        if fs != 0 {
            filter_edge(&mut left_out, 0, w + h, 0, w + h, &inp, (w - h).max(0), w + h + 1, fs);
            use_out = true;
            lbase = w + h - 1;
            max_base_y = w + h - 1;
        } else {
            use_out = false;
            lbase = 0;
            max_base_y = h + w.min(h) - 1;
        }
    }
    let left = |k: i32| if use_out { left_out[(lbase - k) as usize] as i32 } else { e(edge, -1 - k as isize) };
    let base_inc = 1 + upsample_left as i32;
    let mut ypos = dy;
    for x in 0..w as usize {
        let frac = ypos & 0x3e;
        let mut base = ypos >> 6;
        let mut y = 0usize;
        while y < h as usize {
            if base < max_base_y {
                let v = left(base) * (64 - frac) + left(base + 1) * frac;
                dst[off + y * stride + x] = ((v + 32) >> 6) as u16;
            } else {
                let v = left(max_base_y) as u16;
                while y < h as usize {
                    dst[off + y * stride + x] = v;
                    y += 1;
                }
                break;
            }
            y += 1;
            base += base_inc;
        }
        ypos += dy;
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_pred(dst: &mut [u16], off: usize, stride: usize, edge: &[u16], w: usize, h: usize, filt_idx: i32, max: i32) {
    let f = &FILTER_INTRA_TAPS[(filt_idx & 511) as usize];
    // dav1d reads from `topleft_in` for the first row, then from `dst`.
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(4) {
            let (p0, p1, p2, p3, p4, p5, p6);
            let topv = |i: usize| -> i32 {
                if y == 0 {
                    edge[TL + 1 + x + i] as i32
                } else {
                    dst[off + (y - 1) * stride + x + i] as i32
                }
            };
            p1 = topv(0);
            p2 = topv(1);
            p3 = topv(2);
            p4 = topv(3);
            // top-left
            p0 = if y == 0 {
                if x == 0 {
                    edge[TL] as i32
                } else {
                    edge[TL + x] as i32
                }
            } else if x == 0 {
                edge[TL - y] as i32
            } else {
                dst[off + (y - 1) * stride + x - 1] as i32
            };
            // left (two rows)
            let leftv = |r: usize| -> i32 {
                if x == 0 {
                    edge[TL - 1 - y - r] as i32
                } else {
                    dst[off + (y + r) * stride + x - 1] as i32
                }
            };
            p5 = leftv(0);
            p6 = leftv(1);
            let mut k = 0;
            for yy in 0..2 {
                for xx in 0..4 {
                    let acc = f[k] as i32 * p0
                        + f[k + 8] as i32 * p1
                        + f[k + 16] as i32 * p2
                        + f[k + 24] as i32 * p3
                        + f[k + 32] as i32 * p4
                        + f[k + 40] as i32 * p5
                        + f[k + 48] as i32 * p6;
                    dst[off + (y + yy) * stride + x + xx] = ((acc + 8) >> 4).clamp(0, max) as u16;
                    k += 1;
                }
            }
        }
    }
}

/// `cfl_ac_c`.
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac(ac: &mut [i16], ypx: &[u16], yoff: usize, stride: usize, w_pad: usize, h_pad: usize, width: usize, height: usize, ss_hor: bool, ss_ver: bool) {
    let (sh, sv) = (ss_hor as usize, ss_ver as usize);
    let mut y = 0;
    while y < height - 4 * h_pad {
        let row = yoff + y * (stride << sv);
        let mut x = 0;
        while x < width - 4 * w_pad {
            let mut s = ypx[row + (x << sh)] as i32;
            if ss_hor {
                s += ypx[row + x * 2 + 1] as i32;
            }
            if ss_ver {
                s += ypx[row + (x << sh) + stride] as i32;
                if ss_hor {
                    s += ypx[row + x * 2 + 1 + stride] as i32;
                }
            }
            ac[y * width + x] = (s << (1 + !ss_ver as i32 + !ss_hor as i32)) as i16;
            x += 1;
        }
        while x < width {
            ac[y * width + x] = ac[y * width + x - 1];
            x += 1;
        }
        y += 1;
    }
    while y < height {
        ac.copy_within((y - 1) * width..y * width, y * width);
        y += 1;
    }
    let log2sz = width.trailing_zeros() + height.trailing_zeros();
    let mut sum = (1i32 << log2sz) >> 1;
    for v in ac.iter().take(width * height) {
        sum += *v as i32;
    }
    sum >>= log2sz;
    for v in ac.iter_mut().take(width * height) {
        *v = (*v as i32 - sum) as i16;
    }
}

/// `cfl_pred[mode]`.
#[allow(clippy::too_many_arguments)]
pub fn cfl_pred(mode: usize, dst: &mut [u16], off: usize, stride: usize, edge: &[u16], w: usize, h: usize, ac: &[i16], alpha: i32, bitdepth_max: i32) {
    let hbd = bitdepth_max != 255;
    let dc = match mode {
        DC_PRED => dc_gen(edge, w, h, hbd) as i32,
        DC_128_PRED => (bitdepth_max + 1) >> 1,
        TOP_DC_PRED => dc_gen_top(edge, w) as i32,
        _ => dc_gen_left(edge, h) as i32,
    };
    for y in 0..h {
        for x in 0..w {
            let diff = alpha * ac[y * w + x] as i32;
            let v = (diff.abs() + 32) >> 6;
            dst[off + y * stride + x] = (dc + if diff < 0 { -v } else { v }).clamp(0, bitdepth_max) as u16;
        }
    }
}

/// `pal_pred_c` on unpacked indices.
pub fn pal_pred(dst: &mut [u16], off: usize, stride: usize, pal: &[u16; 8], idx: &[u8], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            dst[off + y * stride + x] = pal[idx[y * w + x] as usize];
        }
    }
}
