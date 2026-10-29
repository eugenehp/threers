//! In-loop filters: deblocking (dav1d `lf_apply_tmpl.c`, `loopfilter_tmpl.c`),
//! CDEF (`cdef_apply_tmpl.c`, `cdef_tmpl.c`), superres (`mc.resize`) and loop
//! restoration (`lr_apply_tmpl.c`, `looprestoration_tmpl.c`).
//!
//! dav1d interleaves these per superblock row behind the decoder with line
//! buffers that keep the inputs each stage would see in whole-frame order
//! (decoding never reads filtered pixels of the current frame). Here they
//! run as whole-frame passes after the tiles: deblock all rows, then CDEF
//! from a copy of the deblocked frame, then superres, then restoration per
//! 64-row stripe with the deblocked rows at the stripe edges. The
//! per-pixel arithmetic is dav1d's.

use super::decode::Fd;
use super::lf_mask::{Av1FilterLut, RestorationUnit};
use super::mc::{self, Bd};
use super::obu::*;
use super::picture::Picture;
use super::tables::{CDEF_DIRECTIONS, SGR_PARAMS, SGR_X_BY_X};

// ------------------------------------------------------------------ deblock

#[allow(clippy::too_many_arguments)]
fn loop_filter(d: &mut [u16], off: usize, e: i32, i: i32, h: i32, stridea: isize, strideb: isize, wd: i32, bd: Bd) {
    let bd8 = bd.bits as i32 - 8;
    let f = 1 << bd8;
    let (e, i, h) = (e << bd8, i << bd8, h << bd8);
    let maxp = bd.max;
    let mut o = off as isize;
    for _ in 0..4 {
        let px = |k: isize| d[(o + strideb * k) as usize] as i32;
        let p1 = px(-2);
        let p0 = px(-1);
        let q0 = px(0);
        let q1 = px(1);
        let (mut p2, mut q2, mut p3, mut q3) = (0, 0, 0, 0);
        let mut fm = (p1 - p0).abs() <= i && (q1 - q0).abs() <= i && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;
        if wd > 4 {
            p2 = px(-3);
            q2 = px(2);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;
            if wd > 6 {
                p3 = px(-4);
                q3 = px(3);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            o += stridea;
            continue;
        }
        let (mut p6, mut p5, mut p4, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);
        let mut flat8out = false;
        let mut flat8in = false;
        if wd >= 16 {
            p6 = px(-7);
            p5 = px(-6);
            p4 = px(-5);
            q4 = px(4);
            q5 = px(5);
            q6 = px(6);
            flat8out = (p6 - p0).abs() <= f && (p5 - p0).abs() <= f && (p4 - p0).abs() <= f && (q4 - q0).abs() <= f && (q5 - q0).abs() <= f && (q6 - q0).abs() <= f;
        }
        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f && (p1 - p0).abs() <= f && (q1 - q0).abs() <= f && (q2 - q0).abs() <= f;
        }
        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }
        let mut set = |k: isize, v: i32| d[(o + strideb * k) as usize] = v as u16;
        if wd >= 16 && flat8out && flat8in {
            set(-6, (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
            set(-5, (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
            set(-4, (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
            set(-3, (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
            set(-2, (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
            set(-1, (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
            set(0, (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
            set(1, (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8) >> 4);
            set(2, (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8) >> 4);
            set(3, (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8) >> 4);
            set(4, (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
            set(5, (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
        } else if wd >= 8 && flat8in {
            set(-3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set(-2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set(-1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set(0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set(1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set(2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd == 6 && flat8in {
            set(-2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set(-1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set(0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set(1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;
            let cd = |v: i32| v.clamp(-128 * (1 << bd8), 128 * (1 << bd8) - 1);
            let cp = |v: i32| v.clamp(0, maxp);
            let lim = (128 << bd8) - 1;
            if hev {
                let mut ff = cd(p1 - q1);
                ff = cd(3 * (q0 - p0) + ff);
                let f1 = (ff + 4).min(lim) >> 3;
                let f2 = (ff + 3).min(lim) >> 3;
                set(-1, cp(p0 + f2));
                set(0, cp(q0 - f1));
            } else {
                let ff = cd(3 * (q0 - p0));
                let f1 = (ff + 4).min(lim) >> 3;
                let f2 = (ff + 3).min(lim) >> 3;
                set(-1, cp(p0 + f2));
                set(0, cp(q0 - f1));
                let ff = (f1 + 1) >> 1;
                set(-2, cp(p1 + ff));
                set(1, cp(q1 - ff));
            }
        }
        o += stridea;
    }
}

/// `loop_filter_{h,v}_sb128{y,uv}_c`. `dir` 0 = vertical edges (filter
/// along rows), 1 = horizontal edges. Levels are read at `lvl[li][comp]`
/// stepping `lstep`, the neighbour across the edge at `li + lprev`.
#[allow(clippy::too_many_arguments)]
fn lf_sb(d: &mut [u16], off: usize, stride: usize, vmask: [u32; 3], lvl: &[[u8; 4]], l0: usize, lstep: usize, lprev: isize, comp: usize, lut: &Av1FilterLut, uv: bool, dir: usize, bd: Bd) {
    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut y = 1u32;
    let mut o = off;
    let mut li = l0;
    while vm & !(y.wrapping_sub(1)) != 0 {
        if vm & y != 0 {
            let cur = lvl[li][comp];
            let l = if cur != 0 { cur } else { lvl[(li as isize + lprev) as usize][comp] } as usize;
            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l] as i32;
                let i = lut.i[l] as i32;
                let wd = if uv {
                    4 + 2 * ((vmask[1] & y) != 0) as i32
                } else {
                    let idx = if vmask[2] & y != 0 { 2 } else { (vmask[1] & y != 0) as i32 };
                    4 << idx
                };
                if dir == 0 {
                    loop_filter(d, o, e, i, h, stride as isize, 1, wd, bd);
                } else {
                    loop_filter(d, o, e, i, h, 1, stride as isize, wd, bd);
                }
            }
        }
        if y == 1 << 31 {
            break;
        }
        y <<= 1;
        o += if dir == 0 { 4 * stride } else { 4 };
        li += lstep;
    }
}

// ------------------------------------------------------------------ cdef

fn constrain(diff: i32, threshold: i32, shift: i32) -> i32 {
    let adiff = diff.abs();
    let v = adiff.min(0.max(threshold - (adiff >> shift)));
    if diff < 0 {
        -v
    } else {
        v
    }
}

fn cdef_find_dir(img: &[u16], off: usize, stride: usize, bd8: i32) -> (i32, u32) {
    let mut hv = [[0i32; 8]; 2];
    let mut diag = [[0i32; 15]; 2];
    let mut alt = [[0i32; 11]; 4];
    for y in 0..8 {
        for x in 0..8 {
            let px = (img[off + y * stride + x] as i32 >> bd8) - 128;
            diag[0][y + x] += px;
            alt[0][y + (x >> 1)] += px;
            hv[0][y] += px;
            alt[1][3 + y - (x >> 1)] += px;
            diag[1][7 + y - x] += px;
            alt[2][3 - (y >> 1) + x] += px;
            hv[1][x] += px;
            alt[3][(y >> 1) + x] += px;
        }
    }
    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] = cost[2].wrapping_add((hv[0][n] * hv[0][n]) as u32);
        cost[6] = cost[6].wrapping_add((hv[1][n] * hv[1][n]) as u32);
    }
    cost[2] = cost[2].wrapping_mul(105);
    cost[6] = cost[6].wrapping_mul(105);
    const DIV: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV[n];
        cost[0] = cost[0].wrapping_add(((diag[0][n] * diag[0][n] + diag[0][14 - n] * diag[0][14 - n]) as u32).wrapping_mul(d));
        cost[4] = cost[4].wrapping_add(((diag[1][n] * diag[1][n] + diag[1][14 - n] * diag[1][14 - n]) as u32).wrapping_mul(d));
    }
    cost[0] = cost[0].wrapping_add((diag[0][7] * diag[0][7] * 105) as u32);
    cost[4] = cost[4].wrapping_add((diag[1][7] * diag[1][7] * 105) as u32);
    for n in 0..4 {
        let c = &mut cost[n * 2 + 1];
        for m in 0..5 {
            *c = c.wrapping_add((alt[n][3 + m] * alt[n][3 + m]) as u32);
        }
        *c = c.wrapping_mul(105);
        for m in 0..3 {
            let d = DIV[2 * m + 1];
            *c = c.wrapping_add(((alt[n][m] * alt[n][m] + alt[n][10 - m] * alt[n][10 - m]) as u32).wrapping_mul(d));
        }
    }
    let mut best_dir = 0;
    let mut best = cost[0];
    for n in 1..8 {
        if cost[n] > best {
            best = cost[n];
            best_dir = n;
        }
    }
    (best_dir as i32, best.wrapping_sub(cost[best_dir ^ 4]) >> 10)
}

fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 { (31 - (var >> 6).leading_zeros() as i32).min(12) } else { 0 };
    (strength * (4 + i) + 8) >> 4
}

#[inline]
fn ulog2(v: i32) -> i32 {
    31 - (v as u32).leading_zeros() as i32
}

/// `cdef_filter_block_c` with the padded input `tmp` (12 wide, origin at
/// (2, 2), `i16::MIN` outside the frame).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_block(dst: &mut [u16], off: usize, stride: usize, tmp: &[i32], pri: i32, sec: i32, dir: i32, damping: i32, w: usize, h: usize, bd8: i32) {
    const TS: isize = 12;
    let t = |y: usize, x: usize, o: isize| tmp[((y as isize + 2) * TS + x as isize + 2 + o) as usize];
    let dirs = |k: i32, j: usize| {
        
        CDEF_DIRECTIONS[k as usize][j] as isize
    };
    let umin = |a: i32, b: i32| if (a as u32) < (b as u32) { a } else { b };
    if pri != 0 {
        let pri_tap = 4 - ((pri >> bd8) & 1);
        let pri_shift = 0.max(damping - ulog2(pri));
        if sec != 0 {
            let sec_shift = damping - ulog2(sec);
            for y in 0..h {
                for x in 0..w {
                    let px = dst[off + y * stride + x] as i32;
                    let mut sum = 0;
                    let (mut max, mut min) = (px, px);
                    let mut ptk = pri_tap;
                    for k in 0..2 {
                        let off1 = dirs(dir + 2, k);
                        let p0 = t(y, x, off1);
                        let p1 = t(y, x, -off1);
                        sum += ptk * constrain(p0 - px, pri, pri_shift);
                        sum += ptk * constrain(p1 - px, pri, pri_shift);
                        ptk = (ptk & 3) | 2;
                        min = umin(p0, min);
                        max = max.max(p0);
                        min = umin(p1, min);
                        max = max.max(p1);
                        let off2 = dirs(dir + 4, k);
                        let off3 = dirs(dir, k);
                        let s = [t(y, x, off2), t(y, x, -off2), t(y, x, off3), t(y, x, -off3)];
                        let st = 2 - k as i32;
                        for &sv in &s {
                            sum += st * constrain(sv - px, sec, sec_shift);
                            min = umin(sv, min);
                            max = max.max(sv);
                        }
                    }
                    dst[off + y * stride + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)).clamp(min, max) as u16;
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let px = dst[off + y * stride + x] as i32;
                    let mut sum = 0;
                    let mut ptk = pri_tap;
                    for k in 0..2 {
                        let o = dirs(dir + 2, k);
                        sum += ptk * constrain(t(y, x, o) - px, pri, pri_shift);
                        sum += ptk * constrain(t(y, x, -o) - px, pri, pri_shift);
                        ptk = (ptk & 3) | 2;
                    }
                    dst[off + y * stride + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u16;
                }
            }
        }
    } else {
        let sec_shift = damping - ulog2(sec);
        for y in 0..h {
            for x in 0..w {
                let px = dst[off + y * stride + x] as i32;
                let mut sum = 0;
                for k in 0..2 {
                    let o1 = dirs(dir + 4, k);
                    let o2 = dirs(dir, k);
                    let st = 2 - k as i32;
                    for &sv in &[t(y, x, o1), t(y, x, -o1), t(y, x, o2), t(y, x, -o2)] {
                        sum += st * constrain(sv - px, sec, sec_shift);
                    }
                }
                dst[off + y * stride + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u16;
            }
        }
    }
}

/// `padding()` for CDEF: a 12x12 window (origin (2,2)) of the deblocked
/// frame, `i16::MIN` where an edge is unavailable.
fn fill_cdef_tmp(tmp: &mut [i32], src: &[u16], stride: usize, off: usize, w: usize, h: usize, (have_left, have_right, have_top, have_bottom): (bool, bool, bool, bool)) {
    for v in tmp.iter_mut() {
        *v = i16::MIN as i32;
    }
    let y0 = if have_top { -2 } else { 0 };
    let y1 = h as isize + if have_bottom { 2 } else { 0 };
    let x0 = if have_left { -2 } else { 0 };
    let x1 = w as isize + if have_right { 2 } else { 0 };
    for y in y0..y1 {
        for x in x0..x1 {
            tmp[((y + 2) * 12 + x + 2) as usize] = src[(off as isize + y * stride as isize + x) as usize] as i32;
        }
    }
}

// ------------------------------------------------------------------ restoration

const LR_HAVE_LEFT: u32 = 1;
const LR_HAVE_RIGHT: u32 = 2;
const LR_HAVE_TOP: u32 = 4;
const LR_HAVE_BOTTOM: u32 = 8;

/// A stripe of one restoration unit with its surroundings resolved per the
/// edge flags: `at(x, y)` for x in [-4, w+4), y in [-3, h+3).
struct StripeSrc {
    buf: Vec<i32>,
    w: usize,
}

impl StripeSrc {
    #[inline]
    fn at(&self, x: isize, y: isize) -> i32 {
        self.buf[((y + 3) * (self.w as isize + 8) + x + 4) as usize]
    }
}

/// Resolve the samples a restoration unit's stripe reads: the stripe
/// itself (CDEF output), the two deblocked rows above and below it
/// (dav1d's `lr_lpf_line`), replicating where an edge is missing.
#[allow(clippy::too_many_arguments)]
fn build_stripe(cdef: &[u16], lpf: &[u16], stride: usize, lpf_stride: usize, x0: i32, y0: i32, w: i32, h: i32, plane_w: i32, plane_h: i32, edges: u32) -> StripeSrc {
    let bw = w as usize + 8;
    let mut buf = vec![0i32; bw * (h as usize + 6)];
    for yy in -3..h + 3 {
        let (rows, rstride, ry): (&[u16], usize, i32) = if yy < 0 {
            if edges & LR_HAVE_TOP != 0 {
                (lpf, lpf_stride, y0 + yy.max(-2))
            } else {
                (cdef, stride, y0)
            }
        } else if yy >= h {
            if edges & LR_HAVE_BOTTOM != 0 {
                (lpf, lpf_stride, (y0 + h + (yy - h).min(1)).min(plane_h - 1))
            } else {
                (cdef, stride, y0 + h - 1)
            }
        } else {
            (cdef, stride, y0 + yy)
        };
        for xx in -4..w + 4 {
            let mut ax = x0 + xx;
            if xx < 0 && edges & LR_HAVE_LEFT == 0 {
                ax = x0;
            }
            if xx >= w && edges & LR_HAVE_RIGHT == 0 {
                ax = x0 + w - 1;
            }
            let ax = ax.clamp(0, plane_w + 3);
            buf[(yy + 3) as usize * bw + (xx + 4) as usize] = rows[ry as usize * rstride + ax as usize] as i32;
        }
    }
    StripeSrc { buf, w: w as usize }
}

#[allow(clippy::too_many_arguments)]
fn filter_unit(dst: &mut [u16], stride: usize, x0: usize, y0: usize, w: usize, h: usize, src: &StripeSrc, lr: &RestorationUnit, bitdepth_max: i32) {
    let bitdepth = 32 - (bitdepth_max as u32).leading_zeros() as i32;
    if lr.typ == RESTORATION_WIENER {
        let mut fh = [0i32; 7];
        let mut fv = [0i32; 7];
        for k in 0..3 {
            fh[k] = lr.filter_h[k] as i32;
            fh[6 - k] = fh[k];
            fv[k] = lr.filter_v[k] as i32;
            fv[6 - k] = fv[k];
        }
        // dav1d adds the centre +128 into the tap (16 bpc) or as
        // `src * 128` (8 bpc): the same sum.
        fh[3] = -(fh[0] + fh[1] + fh[2]) * 2 + 128;
        fv[3] = 128 - (fv[0] + fv[1] + fv[2]) * 2;
        let round_bits_h = 3 + (bitdepth == 12) as i32 * 2;
        let rounding_off_h = 1 << (round_bits_h - 1);
        let clip_limit = 1 << (bitdepth + 1 + 7 - round_bits_h);
        let round_bits_v = 11 - (bitdepth == 12) as i32 * 2;
        let rounding_off_v = 1 << (round_bits_v - 1);
        let round_offset = 1 << (bitdepth + (round_bits_v - 1));
        let rows = h + 6;
        let mut hor = vec![0i32; rows * w];
        for r in 0..rows {
            let yy = r as isize - 3;
            for x in 0..w {
                let mut sum = 1 << (bitdepth + 6);
                for i in 0..7 {
                    sum += src.at(x as isize + i as isize - 3, yy) * fh[i];
                }
                hor[r * w + x] = ((sum + rounding_off_h) >> round_bits_h).clamp(0, clip_limit - 1);
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut sum = -round_offset;
                for k in 0..7 {
                    sum += hor[(y + k) * w + x] * fv[k];
                }
                dst[(y0 + y) * stride + x0 + x] = ((sum + rounding_off_v) >> round_bits_v).clamp(0, bitdepth_max) as u16;
            }
        }
        return;
    }
    let idx = (lr.typ - RESTORATION_SGRPROJ) as usize;
    let s0 = SGR_PARAMS[idx][0] as u32;
    let s1 = SGR_PARAMS[idx][1] as u32;
    let w0 = lr.sgr_weights[0] as i32;
    let w1 = 128 - (lr.sgr_weights[0] as i32 + lr.sgr_weights[1] as i32);
    let bd8 = bitdepth - 8;
    let cols = w + 2;
    // (A, B) of `sgr_calc_row_ab` at rows -1..=h, columns -1..=w.
    let ab = |r: isize, n: i32, s: u32, one_by_x: u32, odd_only: bool| -> (Vec<i32>, Vec<i32>) {
        let mut a = vec![0i32; (h + 2) * cols];
        let mut b = vec![0i32; (h + 2) * cols];
        for i in -1..=h as isize {
            if odd_only && i & 1 == 0 {
                continue;
            }
            for j in -1..=w as isize {
                let (mut ss, mut sq) = (0i32, 0i32);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let c = src.at(j + dx, i + dy);
                        ss += c;
                        sq += c * c;
                    }
                }
                let aa = (sq + ((1 << (2 * bd8)) >> 1)) >> (2 * bd8);
                let bb = (ss + ((1 << bd8) >> 1)) >> bd8;
                let p = (aa * n - bb * bb).max(0) as u32;
                let z = (p.wrapping_mul(s).wrapping_add(1 << 19)) >> 20;
                let x = SGR_X_BY_X[z.min(255) as usize] as u32;
                let o = (i + 1) as usize * cols + (j + 1) as usize;
                a[o] = ((x.wrapping_mul(ss as u32).wrapping_mul(one_by_x)).wrapping_add(1 << 11) >> 12) as i32;
                b[o] = x as i32;
            }
        }
        (a, b)
    };
    let at = |v: &[i32], i: isize, j: isize| v[(i + 1) as usize * cols + (j + 1) as usize];
    let mut t5 = vec![0i32; w * h];
    let mut t3 = vec![0i32; w * h];
    if s0 != 0 {
        let (a, b) = ab(2, 25, s0, 164, true);
        for y in 0..h as isize {
            for x in 0..w as isize {
                let px = src.at(x, y);
                let v = if y & 1 == 0 {
                    let six = |p: &[i32]| (at(p, y - 1, x) + at(p, y + 1, x)) * 6 + (at(p, y - 1, x - 1) + at(p, y + 1, x - 1) + at(p, y - 1, x + 1) + at(p, y + 1, x + 1)) * 5;
                    (six(&a) - six(&b) * px + (1 << 8)) >> 9
                } else {
                    let three = |p: &[i32]| at(p, y, x) * 6 + (at(p, y, x - 1) + at(p, y, x + 1)) * 5;
                    (three(&a) - three(&b) * px + (1 << 7)) >> 8
                };
                t5[y as usize * w + x as usize] = v;
            }
        }
    }
    if s1 != 0 {
        let (a, b) = ab(1, 9, s1, 455, false);
        for y in 0..h as isize {
            for x in 0..w as isize {
                let px = src.at(x, y);
                let eight = |p: &[i32]| {
                    (at(p, y, x) + at(p, y, x - 1) + at(p, y, x + 1) + at(p, y - 1, x) + at(p, y + 1, x)) * 4
                        + (at(p, y - 1, x - 1) + at(p, y + 1, x - 1) + at(p, y - 1, x + 1) + at(p, y + 1, x + 1)) * 3
                };
                t3[y as usize * w + x as usize] = (eight(&a) - eight(&b) * px + (1 << 8)) >> 9;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let d = &mut dst[(y0 + y) * stride + x0 + x];
            let v = if s0 != 0 && s1 != 0 {
                w0 * t5[y * w + x] + w1 * t3[y * w + x]
            } else if s0 != 0 {
                w0 * t5[y * w + x]
            } else {
                w1 * t3[y * w + x]
            };
            *d = (*d as i32 + ((v + (1 << 10)) >> 11)).clamp(0, bitdepth_max) as u16;
        }
    }
}

impl Fd {
    fn deblock_sbrow_cols(&mut self, sby: i32) {
        let seq = self.seq.clone();
        let hdr = self.hdr.clone();
        let is_sb64 = !seq.sb128 as i32;
        let starty4 = (sby & is_sb64) << 4;
        let sbsz = 32 >> is_sb64;
        let sbl2 = 5 - is_sb64;
        let halign = ((self.bh + 31) & !31) as usize;
        let ss_ver = self.ss_ver;
        let ss_hor = self.ss_hor;
        let vmask_bits = 16 >> ss_ver;
        let hmask_bits = 16 >> ss_hor;
        let vmax = 1u32 << vmask_bits;
        let hmax = 1u32 << hmask_bits;
        let endy4 = (starty4 + (self.h4 - sby * sbsz).min(sbsz)) as u32;
        let uv_endy4 = (endy4 + ss_ver as u32) >> ss_ver;
        let mask_base = ((sby >> (!seq.sb128) as i32) * self.sb128w) as usize;

        // Tile column boundaries.
        let mut lpf_y = (sby << sbl2) as usize;
        let mut lpf_uv = (sby << (sbl2 - ss_ver)) as usize;
        let mut tile_col = 1;
        loop {
            let mut x = hdr.tiling.col_start_sb[tile_col] as i32;
            if (x << sbl2) >= self.bw {
                break;
            }
            let bx4 = if x & is_sb64 != 0 { 16 } else { 0 };
            let cbx4 = bx4 >> ss_hor;
            x >>= is_sb64;
            let lf = &mut self.lf;
            let m = &mut lf.mask[mask_base + x as usize];
            let mut mask = 1u32 << starty4;
            for y in starty4 as u32..endy4 {
                let sidx = (mask >= 0x10000) as usize;
                let smask = (mask >> (sidx << 4)) as u16;
                let yh = &mut m.filter_y[0][bx4 as usize];
                let idx = 2 * (yh[2][sidx] & smask != 0) as usize + (yh[1][sidx] & smask != 0) as usize;
                yh[2][sidx] &= !smask;
                yh[1][sidx] &= !smask;
                yh[0][sidx] &= !smask;
                let e = lf.tx_lpf_right_edge[0][lpf_y + (y - starty4 as u32) as usize] as usize;
                yh[idx.min(e)][sidx] |= smask;
                mask <<= 1;
            }
            if self.layout != Layout::I400 {
                let mut uv_mask = 1u32 << (starty4 >> ss_ver);
                for y in (starty4 >> ss_ver) as u32..uv_endy4 {
                    let sidx = (uv_mask >= vmax) as usize;
                    let smask = (uv_mask >> (sidx << (4 - ss_ver))) as u16;
                    let uh = &mut m.filter_uv[0][cbx4 as usize];
                    let idx = (uh[1][sidx] & smask != 0) as usize;
                    uh[1][sidx] &= !smask;
                    uh[0][sidx] &= !smask;
                    let e = lf.tx_lpf_right_edge[1][lpf_uv + (y - (starty4 >> ss_ver) as u32) as usize] as usize;
                    uh[idx.min(e)][sidx] |= smask;
                    uv_mask <<= 1;
                }
            }
            lpf_y += halign;
            lpf_uv += halign >> ss_ver;
            tile_col += 1;
        }

        // Tile row boundaries.
        let start_of_tile_row = self.lf.start_of_tile_row[sby as usize] as usize;
        if start_of_tile_row != 0 {
            for x in 0..self.sb128w as usize {
                let a = &self.a[self.sb128w as usize * (start_of_tile_row - 1) + x];
                let m = &mut self.lf.mask[mask_base + x];
                let w = 32.min(self.w4 - ((x as i32) << 5)) as u32;
                let mut mask = 1u32;
                for i in 0..w {
                    let sidx = (mask >= 0x10000) as usize;
                    let smask = (mask >> (sidx << 4)) as u16;
                    let yv = &mut m.filter_y[1][starty4 as usize];
                    let idx = 2 * (yv[2][sidx] & smask != 0) as usize + (yv[1][sidx] & smask != 0) as usize;
                    yv[2][sidx] &= !smask;
                    yv[1][sidx] &= !smask;
                    yv[0][sidx] &= !smask;
                    yv[idx.min(a.tx_lpf_y[i as usize] as usize)][sidx] |= smask;
                    mask <<= 1;
                }
                if self.layout != Layout::I400 {
                    let cw = (w + ss_hor as u32) >> ss_hor;
                    let mut uv_mask = 1u32;
                    for i in 0..cw {
                        let sidx = (uv_mask >= hmax) as usize;
                        let smask = (uv_mask >> (sidx << (4 - ss_hor))) as u16;
                        let uv = &mut m.filter_uv[1][(starty4 >> ss_ver) as usize];
                        let idx = (uv[1][sidx] & smask != 0) as usize;
                        uv[1][sidx] &= !smask;
                        uv[0][sidx] &= !smask;
                        uv[idx.min(a.tx_lpf_uv[i as usize] as usize)][sidx] |= smask;
                        uv_mask <<= 1;
                    }
                }
            }
        }

        let bd = Bd::new(self.bitdepth_max);
        let ys = self.cur.stride[0];
        let b4s = self.b4_stride as usize;
        let row0 = (sby * self.sb_step * 4) as usize;
        let lvl_row = (self.b4_stride * sby * sbsz) as usize;
        for x in 0..self.sb128w as usize {
            let have_left = x > 0;
            let w = 32.min(self.w4 - x as i32 * 32) as usize;
            let m = self.lf.mask[mask_base + x];
            for xx in 0..w {
                if !have_left && xx == 0 {
                    continue;
                }
                let mut hm = [0u32; 3];
                for k in 0..3 {
                    hm[k] = if starty4 == 0 {
                        let mut v = m.filter_y[0][xx][k][0] as u32;
                        if endy4 > 16 {
                            v |= (m.filter_y[0][xx][k][1] as u32) << 16;
                        }
                        v
                    } else {
                        m.filter_y[0][xx][k][1] as u32
                    };
                }
                let off = row0 * ys + x * 128 + xx * 4;
                lf_sb(&mut self.cur.planes[0], off, ys, hm, &self.lf.level, lvl_row + x * 32 + xx, b4s, -1, 0, &self.lf.lim_lut, false, 0, bd);
            }
        }
        if hdr.loopfilter.level_u == 0 && hdr.loopfilter.level_v == 0 {
            return;
        }
        let uvs = self.cur.stride[1];
        let crow0 = row0 >> ss_ver;
        let lvl_row = (self.b4_stride * (sby * sbsz >> ss_ver)) as usize;
        let sy = starty4 >> ss_ver;
        for x in 0..self.sb128w as usize {
            let have_left = x > 0;
            let w = ((32.min(self.w4 - x as i32 * 32) + ss_hor) >> ss_hor) as usize;
            let m = self.lf.mask[mask_base + x];
            for xx in 0..w {
                if !have_left && xx == 0 {
                    continue;
                }
                let mut hm = [0u32; 3];
                for k in 0..2 {
                    hm[k] = if sy == 0 {
                        let mut v = m.filter_uv[0][xx][k][0] as u32;
                        if uv_endy4 > (16 >> ss_ver) as u32 {
                            v |= (m.filter_uv[0][xx][k][1] as u32) << (16 >> ss_ver);
                        }
                        v
                    } else {
                        m.filter_uv[0][xx][k][1] as u32
                    };
                }
                let off = crow0 * uvs + x * (128 >> ss_hor) + xx * 4;
                let l0 = lvl_row + x * (32 >> ss_hor) + xx;
                for (pl, comp) in [(1usize, 2usize), (2, 3)] {
                    lf_sb(&mut self.cur.planes[pl], off, uvs, hm, &self.lf.level, l0, b4s, -1, comp, &self.lf.lim_lut, true, 0, bd);
                }
            }
        }
    }

    fn deblock_sbrow_rows(&mut self, sby: i32) {
        let seq = self.seq.clone();
        let hdr = self.hdr.clone();
        let have_top = sby > 0;
        let is_sb64 = !seq.sb128 as i32;
        let starty4 = (sby & is_sb64) << 4;
        let sbsz = 32 >> is_sb64;
        let ss_ver = self.ss_ver;
        let ss_hor = self.ss_hor;
        let endy4 = starty4 + (self.h4 - sby * sbsz).min(sbsz);
        let uv_endy4 = (endy4 + ss_ver) >> ss_ver;
        let mask_base = ((sby >> (!seq.sb128) as i32) * self.sb128w) as usize;
        let bd = Bd::new(self.bitdepth_max);
        let ys = self.cur.stride[0];
        let b4s = self.b4_stride as usize;
        let row0 = (sby * self.sb_step * 4) as usize;
        let lvl_row = (self.b4_stride * sby * sbsz) as usize;
        for x in 0..self.sb128w as usize {
            let m = self.lf.mask[mask_base + x];
            for (k, y) in (starty4..endy4).enumerate() {
                if !have_top && y == 0 {
                    continue;
                }
                let yy = y as usize;
                let vm = [0, 1, 2].map(|t| m.filter_y[1][yy][t][0] as u32 | ((m.filter_y[1][yy][t][1] as u32) << 16));
                let off = (row0 + k * 4) * ys + x * 128;
                let l0 = lvl_row + k * b4s + x * 32;
                lf_sb(&mut self.cur.planes[0], off, ys, vm, &self.lf.level, l0, 1, -(b4s as isize), 1, &self.lf.lim_lut, false, 1, bd);
            }
        }
        if hdr.loopfilter.level_u == 0 && hdr.loopfilter.level_v == 0 {
            return;
        }
        let uvs = self.cur.stride[1];
        let crow0 = row0 >> ss_ver;
        let lvl_row = (self.b4_stride * (sby * sbsz >> ss_ver)) as usize;
        for x in 0..self.sb128w as usize {
            let m = self.lf.mask[mask_base + x];
            for (k, y) in ((starty4 >> ss_ver)..uv_endy4).enumerate() {
                if !have_top && y == 0 {
                    continue;
                }
                let yy = y as usize;
                let sh = 16 >> ss_hor;
                let vm = [m.filter_uv[1][yy][0][0] as u32 | ((m.filter_uv[1][yy][0][1] as u32) << sh), m.filter_uv[1][yy][1][0] as u32 | ((m.filter_uv[1][yy][1][1] as u32) << sh), 0];
                let off = (crow0 + k * 4) * uvs + x * (128 >> ss_hor);
                let l0 = lvl_row + k * b4s + x * (32 >> ss_hor);
                for (pl, comp) in [(1usize, 2usize), (2, 3)] {
                    lf_sb(&mut self.cur.planes[pl], off, uvs, vm, &self.lf.level, l0, 1, -(b4s as isize), comp, &self.lf.lim_lut, true, 1, bd);
                }
            }
        }
    }

    fn cdef_frame(&mut self, src: &Picture) {
        let hdr = self.hdr.clone();
        let bd8 = self.bpc as i32 - 8;
        let damping = hdr.cdef.damping as i32 + bd8;
        let layout = self.layout;
        let uv_dirs: [[i32; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 0, 2, 4, 5, 6, 6, 6]];
        let uv_dir = uv_dirs[(layout == Layout::I422) as usize];
        let (ss_hor, ss_ver) = (self.ss_hor as usize, self.ss_ver as usize);
        let sb64w = (self.sb128w << 1) as usize;
        let mut tmp = vec![0i32; 12 * 12];
        let mut by = 0;
        while by < self.bh {
            let have_top = by > 0;
            let have_bottom = by + 2 < self.bh;
            for sbx in 0..sb64w {
                let sb128x = sbx >> 1;
                let sb64_idx = (((by & 16) >> 3) as usize) + (sbx & 1);
                let mask_i = (by >> 5) as usize * self.sb128w as usize + sb128x;
                let cdef_idx = self.lf.mask[mask_i].cdef_idx[sb64_idx];
                if cdef_idx == -1 {
                    continue;
                }
                let ci = cdef_idx as usize;
                let y_lvl = hdr.cdef.y_strength[ci] as i32;
                let uv_lvl = hdr.cdef.uv_strength[ci] as i32;
                if y_lvl == 0 && uv_lvl == 0 {
                    continue;
                }
                let by_idx = ((by & 30) >> 1) as usize;
                let nr = self.lf.mask[mask_i].noskip_mask[by_idx];
                let noskip = (nr[1] as u32) << 16 | nr[0] as u32;
                let y_pri = (y_lvl >> 2) << bd8;
                let mut y_sec = y_lvl & 3;
                y_sec += (y_sec == 3) as i32;
                y_sec <<= bd8;
                let uv_pri = (uv_lvl >> 2) << bd8;
                let mut uv_sec = uv_lvl & 3;
                uv_sec += (uv_sec == 3) as i32;
                uv_sec <<= bd8;
                let mut bx = sbx as i32 * 16;
                while bx < ((sbx as i32 + 1) * 16).min(self.bw) {
                    let have_left = bx > 0;
                    let have_right = bx + 2 < self.bw;
                    let bx_mask = 3u32 << (bx & 30);
                    if noskip & bx_mask == 0 {
                        bx += 2;
                        continue;
                    }
                    let ys = src.stride[0];
                    let yoff = (by as usize * 4) * ys + bx as usize * 4;
                    let (dir, var) = if y_pri != 0 || uv_pri != 0 { cdef_find_dir(&src.planes[0], yoff, ys, bd8) } else { (0, 0) };
                    let edges = (have_left, have_right, have_top, have_bottom);
                    if y_pri != 0 {
                        let adj = adjust_strength(y_pri, var);
                        if adj != 0 || y_sec != 0 {
                            fill_cdef_tmp(&mut tmp, &src.planes[0], ys, yoff, 8, 8, edges);
                            cdef_filter_block(&mut self.cur.planes[0], yoff, ys, &tmp, adj, y_sec, dir, damping, 8, 8, bd8);
                        }
                    } else if y_sec != 0 {
                        fill_cdef_tmp(&mut tmp, &src.planes[0], ys, yoff, 8, 8, edges);
                        cdef_filter_block(&mut self.cur.planes[0], yoff, ys, &tmp, 0, y_sec, 0, damping, 8, 8, bd8);
                    }
                    if uv_lvl != 0 && layout != Layout::I400 {
                        let uvdir = if uv_pri != 0 { uv_dir[dir as usize] } else { 0 };
                        let uvs = src.stride[1];
                        let (cw, ch) = (8 >> ss_hor, 8 >> ss_ver);
                        let coff = ((by as usize * 4) >> ss_ver) * uvs + ((bx as usize * 4) >> ss_hor);
                        for pl in 1..3 {
                            fill_cdef_tmp(&mut tmp, &src.planes[pl], uvs, coff, cw, ch, edges);
                            cdef_filter_block(&mut self.cur.planes[pl], coff, uvs, &tmp, uv_pri, uv_sec, uvdir, damping - 1, cw, ch, bd8);
                        }
                    }
                    bx += 2;
                }
            }
            by += 2;
        }
    }

    fn superres(&mut self) {
        let bd = Bd::new(self.bitdepth_max);
        let sr = self.sr_cur.as_mut().unwrap();
        let np = if self.layout == Layout::I400 { 1 } else { 3 };
        for pl in 0..np {
            let (ss_hor, ss_ver) = self.cur.ss(pl);
            let dst_w = ((sr.w + ss_hor) >> ss_hor) as usize;
            let src_w = (4 * self.bw + ss_hor) >> ss_hor;
            let h = ((self.cur.h + ss_ver) >> ss_ver) as usize;
            let (ds, ss) = (sr.pstride(pl), self.cur.pstride(pl));
            let step = self.resize_step[(pl != 0) as usize];
            let start = self.resize_start[(pl != 0) as usize];
            for y in 0..h {
                let (d, s) = (&mut sr.planes[pl][y * ds..y * ds + dst_w], &self.cur.planes[pl][y * ss..]);
                mc::resize_row(d, s, dst_w, src_w, step, start, bd);
            }
        }
    }

    /// Deblocked rows for loop restoration (upscaled with superres).
    fn lr_source_rows(&self, deblocked: &Picture) -> Picture {
        match &self.sr_cur {
            None => deblocked.clone(),
            Some(sr) => {
                let bd = Bd::new(self.bitdepth_max);
                let mut out = Picture::new(sr.w, sr.h, sr.layout, sr.bpc);
                let np = if self.layout == Layout::I400 { 1 } else { 3 };
                for pl in 0..np {
                    let (ss_hor, ss_ver) = deblocked.ss(pl);
                    let dst_w = ((sr.w + ss_hor) >> ss_hor) as usize;
                    let src_w = (4 * self.bw) >> ss_hor;
                    let h = ((deblocked.h + ss_ver) >> ss_ver) as usize;
                    let (ds, ss) = (out.pstride(pl), deblocked.pstride(pl));
                    let step = self.resize_step[ss_hor as usize];
                    let start = self.resize_start[ss_hor as usize];
                    for y in 0..h {
                        let (d, s) = (&mut out.planes[pl][y * ds..y * ds + dst_w], &deblocked.planes[pl][y * ss..]);
                        mc::resize_row(d, s, dst_w, src_w, step, start, bd);
                    }
                }
                out
            }
        }
    }

    fn lr_frame(&mut self, lpf: &Picture) {
        let seq = self.seq.clone();
        let hdr = self.hdr.clone();
        let superres = hdr.width[0] != hdr.width[1];
        let mut pic = if superres { self.sr_cur.take().unwrap() } else { std::mem::replace(&mut self.cur, Picture::new(1, 1, self.layout, self.bpc)) };
        for plane in 0..3 {
            if (self.lf.restore_planes >> plane) & 1 == 0 {
                continue;
            }
            let (ss_hor, ss_ver) = pic.ss(plane);
            let h = (pic.h + ss_ver) >> ss_ver;
            let w = (pic.w + ss_hor) >> ss_hor;
            let stride = pic.pstride(plane);
            // Every unit reads the unrestored CDEF output.
            let cdef_out = pic.planes[plane].clone();
            let unit_size_log2 = hdr.restoration.unit_size[(plane != 0) as usize] as i32;
            let unit_size = 1 << unit_size_log2;
            let half_unit = unit_size >> 1;
            let max_unit = unit_size + half_unit;
            let shift_hor = 7 - ss_hor;
            for sby in 0..self.sbh {
                let not_last = sby + 1 < self.sbh;
                let next_row_y = (sby + 1) << ((6 - ss_ver) + seq.sb128 as i32);
                let row_h = (next_row_y - (8 >> ss_ver) * not_last as i32).min(h);
                let offset = (8 * (sby != 0) as i32) >> ss_ver;
                let y = (sby << ((6 - ss_ver) + seq.sb128 as i32)) - offset;
                let row_y = y + ((8 >> ss_ver) * (y != 0) as i32);
                let mut aligned_unit_pos = row_y & !(unit_size - 1);
                if aligned_unit_pos != 0 && aligned_unit_pos + half_unit > h {
                    aligned_unit_pos -= unit_size;
                }
                aligned_unit_pos <<= ss_ver;
                let sb_idx = ((aligned_unit_pos >> 7) * self.sr_sb128w) as usize;
                let unit_idx = (((aligned_unit_pos >> 6) & 1) << 1) as usize;
                let mut units: Vec<(i32, i32, RestorationUnit, u32)> = Vec::new();
                let mut edges = (if y > 0 { LR_HAVE_TOP } else { 0 }) | LR_HAVE_RIGHT;
                let mut x = 0;
                let mut cur_lr = self.lf.lr_mask[sb_idx].lr[plane][unit_idx];
                while x + max_unit <= w {
                    let next_x = x + unit_size;
                    let next_u_idx = unit_idx + ((next_x >> (shift_hor - 1)) & 1) as usize;
                    let nlr = self.lf.lr_mask[sb_idx + (next_x >> shift_hor) as usize].lr[plane][next_u_idx];
                    if cur_lr.typ != RESTORATION_NONE {
                        units.push((x, unit_size, cur_lr, edges));
                    }
                    edges |= LR_HAVE_LEFT;
                    x = next_x;
                    cur_lr = nlr;
                }
                if cur_lr.typ != RESTORATION_NONE {
                    edges &= !LR_HAVE_RIGHT;
                    units.push((x, w - x, cur_lr, edges));
                }
                for (ux, uw, lr, e) in units {
                    let mut sy = y;
                    let mut stripe_h = ((64 - 8 * (sy == 0) as i32) >> ss_ver).min(row_h - sy);
                    let mut e = e;
                    while sy + stripe_h <= row_h {
                        let have_bottom = sby + 1 != self.sbh || sy + stripe_h != row_h;
                        if have_bottom {
                            e |= LR_HAVE_BOTTOM;
                        } else {
                            e &= !LR_HAVE_BOTTOM;
                        }
                        let src = build_stripe(&cdef_out, &lpf.planes[plane], stride, lpf.pstride(plane), ux, sy, uw, stripe_h, w, h, e);
                        filter_unit(&mut pic.planes[plane], stride, ux as usize, sy as usize, uw as usize, stripe_h as usize, &src, &lr, self.bitdepth_max);
                        sy += stripe_h;
                        e |= LR_HAVE_TOP;
                        stripe_h = (64 >> ss_ver).min(row_h - sy);
                        if stripe_h == 0 {
                            break;
                        }
                    }
                }
            }
        }
        if superres {
            self.sr_cur = Some(pic);
        } else {
            self.cur = pic;
        }
    }

    /// Run the in-loop filters over the decoded frame.
    pub fn apply_filters(&mut self) {
        let hdr = self.hdr.clone();
        let seq = self.seq.clone();
        if hdr.loopfilter.level_y[0] != 0 || hdr.loopfilter.level_y[1] != 0 {
            for sby in 0..self.sbh {
                self.deblock_sbrow_cols(sby);
                self.deblock_sbrow_rows(sby);
            }
        }
        let deblocked = if seq.cdef || self.lf.restore_planes != 0 { Some(self.cur.clone()) } else { None };
        if seq.cdef {
            self.cdef_frame(deblocked.as_ref().unwrap());
        }
        if self.sr_cur.is_some() {
            self.superres();
        }
        if self.lf.restore_planes != 0 {
            let lpf = self.lr_source_rows(deblocked.as_ref().unwrap());
            self.lr_frame(&lpf);
        }
    }

}
