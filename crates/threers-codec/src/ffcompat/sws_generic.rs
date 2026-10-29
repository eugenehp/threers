//! libswscale n7.1's generic (filtered) YUV -> rgb24 path, for what the
//! special `yuv2rgb` converter in [`super::swscale`] does not take: odd
//! output heights, more than 8 bits per sample, unsubsampled chroma. That is
//! what ffmpeg runs for moviepy's `-vf scale=W:H -sws_flags bicubic -pix_fmt
//! rgb24` at the source size: `initFilter` (bicubic, B = 0, C = 0.6) with the
//! aarch64 filter alignment (4 horizontal, 2 vertical), `hScale{8,16}To15`
//! (NEON, integer), and `packed_vscale`'s choice of `yuv2rgb24_{1,2,X}` or
//! their `_full` (full chroma interpolation) variants. Everything is integer
//! arithmetic, so the port is exact by construction where the code matches.

#![allow(clippy::explicit_counter_loop, clippy::needless_range_loop)]

use super::swscale::Yuv2Rgb;

/// `SWS_MAX_REDUCE_CUTOFF`.
const MAX_REDUCE_CUTOFF: f64 = 0.002;
const HEADROOM: i32 = 512; // YUVRGB_TABLE_HEADROOM

/// One direction's filter bank (`initFilter` output).
struct Filter {
    size: usize,
    pos: Vec<i32>,
    coef: Vec<i16>,
}

fn av_log2(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        31 - v.leading_zeros() as i32
    }
}

fn rounded_div(a: i64, b: i64) -> i64 {
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

/// `initFilter` for `SWS_BICUBIC` without src/dst vectors.
fn init_filter(x_inc: i32, src_w: i32, dst_w: i32, filter_align: usize, one: i64, src_pos: i32, dst_pos: i32) -> Filter {
    let fone: i64 = 1i64 << (54 - av_log2((src_w / dst_w) as u32).min(8));
    let dw = dst_w as usize;
    let mut pos = vec![0i32; dw];
    let mut filter_size: usize;
    let mut filter: Vec<i64>;
    if (x_inc - 0x10000).abs() < 10 && src_pos == dst_pos {
        filter_size = 1;
        filter = vec![fone; dw];
        for (i, p) in pos.iter_mut().enumerate() {
            *p = i as i32;
        }
    } else {
        let size_factor = 4i64; // bicubic
        let fs = if x_inc <= 1 << 16 { 1 + size_factor } else { 1 + (size_factor * src_w as i64 + dst_w as i64 - 1) / dst_w as i64 };
        filter_size = fs.min(src_w as i64 - 2).max(1) as usize;
        filter = vec![0i64; dw * filter_size];
        let mut x_dst_in_src: i64 = ((dst_pos as i64 * x_inc as i64) >> 7) - ((src_pos as i64 * 0x10000) >> 7);
        let b: i64 = 0;
        let c: i64 = (0.6f64 * (1u64 << 24) as f64) as i64;
        for i in 0..dw {
            let mut xx = ((x_dst_in_src - (filter_size as i64 - 2) * (1i64 << 16)) / (1 << 17)) as i32;
            pos[i] = xx;
            for j in 0..filter_size {
                let mut d: i64 = ((xx as i64 * (1i64 << 17)) - x_dst_in_src).abs() << 13;
                if x_inc > 1 << 16 {
                    d = d * dst_w as i64 / src_w as i64;
                }
                let mut coeff: i64 = if d >= 1i64 << 31 {
                    0
                } else {
                    let dd = (d * d) >> 30;
                    let ddd = (dd * d) >> 30;
                    if d < 1i64 << 30 {
                        (12 * (1i64 << 24) - 9 * b - 6 * c) * ddd + (-18 * (1i64 << 24) + 12 * b + 6 * c) * dd + (6 * (1i64 << 24) - 2 * b) * (1i64 << 30)
                    } else {
                        (-b - 6 * c) * ddd + (6 * b + 30 * c) * dd + (-12 * b - 48 * c) * d + (8 * b + 24 * c) * (1i64 << 30)
                    }
                };
                coeff /= (1i64 << 54) / fone;
                filter[i * filter_size + j] = coeff;
                xx += 1;
            }
            x_dst_in_src += 2 * x_inc as i64;
        }
    }
    // filter2 == filter (no src/dst filter vectors); filterPos unchanged.
    let f2 = filter_size;
    let mut filter2 = filter;
    // Step 1: find the reduced size, shifting near-zero lead coefficients out.
    let mut min_filter_size = 0usize;
    let cut = MAX_REDUCE_CUTOFF * fone as f64;
    for i in (0..dw).rev() {
        let mut min = f2;
        let mut cutoff: i64 = 0;
        for _ in 0..f2 {
            cutoff += filter2[i * f2].abs();
            if cutoff as f64 > cut {
                break;
            }
            if i < dw - 1 && pos[i] >= pos[i + 1] {
                break;
            }
            for k in 1..f2 {
                filter2[i * f2 + k - 1] = filter2[i * f2 + k];
            }
            filter2[i * f2 + f2 - 1] = 0;
            pos[i] += 1;
        }
        cutoff = 0;
        for j in (1..f2).rev() {
            cutoff += filter2[i * f2 + j].abs();
            if cutoff as f64 > cut {
                break;
            }
            min -= 1;
        }
        if min > min_filter_size {
            min_filter_size = min;
        }
    }
    filter_size = (min_filter_size + filter_align - 1) & !(filter_align - 1);
    let fsz = filter_size;
    let mut filt = vec![0i64; dw * fsz];
    for i in 0..dw {
        for j in 0..fsz {
            filt[i * fsz + j] = if j >= f2 { 0 } else { filter2[i * f2 + j] };
        }
    }
    // Fix borders.
    for i in 0..dw {
        if pos[i] < 0 {
            for j in 1..fsz {
                let left = (j as i32 + pos[i]).max(0) as usize;
                filt[i * fsz + left] += filt[i * fsz + j];
                filt[i * fsz + j] = 0;
            }
            pos[i] = 0;
        }
        if pos[i] + fsz as i32 > src_w {
            let shift = pos[i] + (fsz as i32 - src_w).min(0);
            let mut acc: i64 = 0;
            for j in (0..fsz).rev() {
                if pos[i] + j as i32 >= src_w {
                    acc += filt[i * fsz + j];
                    filt[i * fsz + j] = 0;
                }
            }
            for j in (0..fsz).rev() {
                if (j as i32) < shift {
                    filt[i * fsz + j] = 0;
                } else {
                    filt[i * fsz + j] = filt[i * fsz + j - shift as usize];
                }
            }
            pos[i] -= shift;
            filt[i * fsz + (src_w - 1 - pos[i]) as usize] += acc;
        }
    }
    // Normalise with error diffusion.
    let mut coef = vec![0i16; dw * fsz];
    for i in 0..dw {
        let mut sum: i64 = filt[i * fsz..(i + 1) * fsz].iter().sum();
        sum = (sum + one / 2) / one;
        if sum == 0 {
            sum = 1;
        }
        let mut error = 0i64;
        for j in 0..fsz {
            let v = filt[i * fsz + j] + error;
            let int_v = rounded_div(v, sum);
            coef[i * fsz + j] = int_v as i16;
            error = v - int_v * sum;
        }
    }
    Filter { size: fsz, pos, coef }
}

/// `get_local_pos`.
fn local_pos(chr_subsample: u32, pos: i32) -> i32 {
    let mut p = pos;
    if p == -1 || p <= -513 {
        p = (128 << chr_subsample) - 128;
    }
    p += 128;
    p >> chr_subsample
}

/// `hScale{8,16}To15` (the NEON kernels saturate the narrowed sum; the
/// values never reach the low bound).
fn hscale(dst: &mut [i16], src: &[u16], f: &Filter, sh: u32) {
    for (i, d) in dst.iter_mut().enumerate() {
        let p = f.pos[i] as usize;
        let mut val: i32 = 0;
        for j in 0..f.size {
            val += src[p + j] as i32 * f.coef[f.size * i + j] as i32;
        }
        *d = (val >> sh).clamp(-32768, 32767) as i16;
    }
}

/// A source picture layout: luma size, chroma subsampling shifts, bit depth.
#[derive(Clone, Copy, Debug)]
pub struct SrcFormat {
    /// Luma width.
    pub w: usize,
    /// Luma height.
    pub h: usize,
    /// log2 of the horizontal chroma subsampling.
    pub log2_cw: u32,
    /// log2 of the vertical chroma subsampling.
    pub log2_ch: u32,
    /// Bits per sample.
    pub bit_depth: u32,
}

/// A configured scaler for one source format at the same output size.
pub struct GenericScaler {
    fmt: SrcFormat,
    full_chroma: bool,
    chr_src_w: usize,
    chr_src_h: usize,
    chr_dst_w: usize,
    h_lum: Filter,
    h_chr: Filter,
    v_lum: Filter,
    v_chr: Filter,
}

impl GenericScaler {
    /// `sws_init_context` decisions for `fmt` -> rgb24 at the same size,
    /// `SWS_BICUBIC`, chroma siting as vf_scale sets it (`in_chroma_loc`
    /// unspecified = centre: `src_*_chr_pos` 128 when subsampled).
    pub fn new(fmt: SrcFormat) -> Self {
        let (src_w, src_h) = (fmt.w as i32, fmt.h as i32);
        let (dst_w, dst_h) = (src_w, src_h);
        let mut full_chroma = false;
        if dst_w & 1 != 0 {
            full_chroma = true;
        }
        if fmt.log2_cw == 0 && fmt.log2_ch == 0 {
            full_chroma = true;
        }
        let chr_dst_h_sub = if full_chroma { 0 } else { 1 };
        let chr_src_w = (src_w + (1 << fmt.log2_cw) - 1) >> fmt.log2_cw;
        let chr_src_h = (src_h + (1 << fmt.log2_ch) - 1) >> fmt.log2_ch;
        let chr_dst_w = (dst_w + (1 << chr_dst_h_sub) - 1) >> chr_dst_h_sub;
        let chr_dst_h = dst_h;
        let lum_x_inc = (((src_w as i64) << 16) + (dst_w as i64 >> 1)) / dst_w as i64;
        let lum_y_inc = (((src_h as i64) << 16) + (dst_h as i64 >> 1)) / dst_h as i64;
        let chr_x_inc = (((chr_src_w as i64) << 16) + (chr_dst_w as i64 >> 1)) / chr_dst_w as i64;
        let chr_y_inc = (((chr_src_h as i64) << 16) + (chr_dst_h as i64 >> 1)) / chr_dst_h as i64;
        // vf_scale's calc_chroma_pos (centre siting) for the input; rgb24
        // output has no subsampling (-513).
        let src_h_chr_pos = if fmt.log2_cw > 0 { 128 * ((1 << fmt.log2_cw) - 1) } else { -513 };
        let src_v_chr_pos = if fmt.log2_ch > 0 { 128 * ((1 << fmt.log2_ch) - 1) } else { -513 };
        let h_lum = init_filter(lum_x_inc as i32, src_w, dst_w, 4, 1 << 14, local_pos(0, 0), local_pos(0, 0));
        let h_chr = init_filter(chr_x_inc as i32, chr_src_w, chr_dst_w, 4, 1 << 14, local_pos(fmt.log2_cw, src_h_chr_pos), local_pos(chr_dst_h_sub, -513));
        let v_lum = init_filter(lum_y_inc as i32, src_h, dst_h, 2, 1 << 12, local_pos(0, 0), local_pos(0, 0));
        let v_chr = init_filter(chr_y_inc as i32, chr_src_h, chr_dst_h, 2, 1 << 12, local_pos(fmt.log2_ch, src_v_chr_pos), local_pos(0, -513));
        Self {
            fmt,
            full_chroma,
            chr_src_w: chr_src_w as usize,
            chr_src_h: chr_src_h as usize,
            chr_dst_w: chr_dst_w as usize,
            h_lum,
            h_chr,
            v_lum,
            v_chr,
        }
    }

    /// Convert planes (`u16` samples, tightly packed) to rgb24.
    pub fn convert(&self, t: &Yuv2Rgb, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
        let (w, h) = (self.fmt.w, self.fmt.h);
        let sh = if self.fmt.bit_depth == 8 { 7 } else { self.fmt.bit_depth - 1 };
        let (cw, ch, cdw) = (self.chr_src_w, self.chr_src_h, self.chr_dst_w);
        let mut lum = vec![vec![0i16; w]; h];
        for (r, row) in lum.iter_mut().enumerate() {
            hscale(row, &y[r * w..(r + 1) * w], &self.h_lum, sh);
        }
        let mut cu = vec![vec![0i16; cdw]; ch];
        let mut cv = vec![vec![0i16; cdw]; ch];
        for r in 0..ch {
            hscale(&mut cu[r], &u[r * cw..(r + 1) * cw], &self.h_chr, sh);
            hscale(&mut cv[r], &v[r * cw..(r + 1) * cw], &self.h_chr, sh);
        }
        let mut out = vec![0u8; w * h * 3];
        let (lf, cf) = (&self.v_lum, &self.v_chr);
        for yy in 0..h {
            let dest = &mut out[yy * w * 3..(yy + 1) * w * 3];
            let (lp, cp) = (lf.pos[yy] as usize, cf.pos[yy] as usize);
            let lc = &lf.coef[yy * lf.size..(yy + 1) * lf.size];
            let cc = &cf.coef[yy * cf.size..(yy + 1) * cf.size];
            let ls: Vec<&[i16]> = (0..lf.size).map(|j| lum[lp + j].as_slice()).collect();
            let us: Vec<&[i16]> = (0..cf.size).map(|j| cu[cp + j].as_slice()).collect();
            let vs: Vec<&[i16]> = (0..cf.size).map(|j| cv[cp + j].as_slice()).collect();
            // packed_vscale: uint16 filter views.
            let two = |c: &[i16]| c.len() == 2 && (c[1] as u16 as u32 + c[0] as u16 as u32) == 4096 && (c[1] as u16 as u32) <= 4096;
            if two(lc) && two(cc) {
                let (ya, ca) = (lc[1] as u16 as i32, cc[1] as u16 as i32);
                if self.full_chroma {
                    full_2(t, dest, &ls, &us, &vs, w, ya, ca);
                } else {
                    half_2(t, dest, &ls, &us, &vs, w, ya, ca);
                }
            } else if self.full_chroma {
                full_x(t, dest, lc, &ls, cc, &us, &vs, w);
            } else {
                half_x(t, dest, lc, &ls, cc, &us, &vs, w);
            }
        }
        out
    }
}

#[inline]
fn put_half(t: &Yuv2Rgb, dest: &mut [u8], i: usize, y1: i32, y2: i32, u: i32, v: i32) {
    let (uu, vv) = ((u + HEADROOM) as usize, (v + HEADROOM) as usize);
    let r = t.r_v[vv];
    let g = t.g_u[uu] + t.g_v[vv];
    let b = t.b_u[uu];
    for (k, yv) in [(0, y1), (1, y2)] {
        let o = (2 * i + k) * 3;
        let yv = yv as i64;
        dest[o] = t.y_table[(r + yv) as usize];
        dest[o + 1] = t.y_table[(g + yv) as usize];
        dest[o + 2] = t.y_table[(b + yv) as usize];
    }
}

/// `yuv2rgb24_2_c`.
#[allow(clippy::too_many_arguments)]
fn half_2(t: &Yuv2Rgb, dest: &mut [u8], l: &[&[i16]], u: &[&[i16]], v: &[&[i16]], w: usize, ya: i32, ca: i32) {
    let (ya1, ca1) = (4096 - ya, 4096 - ca);
    for i in 0..w.div_ceil(2) {
        let y1 = (l[0][2 * i] as i32 * ya1 + l[1][2 * i] as i32 * ya) >> 19;
        let y2 = (l[0][2 * i + 1] as i32 * ya1 + l[1][2 * i + 1] as i32 * ya) >> 19;
        let uu = (u[0][i] as i32 * ca1 + u[1][i] as i32 * ca) >> 19;
        let vv = (v[0][i] as i32 * ca1 + v[1][i] as i32 * ca) >> 19;
        put_half(t, dest, i, y1, y2, uu, vv);
    }
}

/// `yuv2rgb24_X_c`.
#[allow(clippy::too_many_arguments)]
fn half_x(t: &Yuv2Rgb, dest: &mut [u8], lf: &[i16], l: &[&[i16]], cf: &[i16], u: &[&[i16]], v: &[&[i16]], w: usize) {
    for i in 0..w.div_ceil(2) {
        let (mut y1, mut y2, mut uu, mut vv) = (1i32 << 18, 1i32 << 18, 1i32 << 18, 1i32 << 18);
        for j in 0..lf.len() {
            y1 += l[j][2 * i] as i32 * lf[j] as i32;
            y2 += l[j][2 * i + 1] as i32 * lf[j] as i32;
        }
        for j in 0..cf.len() {
            uu += u[j][i] as i32 * cf[j] as i32;
            vv += v[j][i] as i32 * cf[j] as i32;
        }
        put_half(t, dest, i, y1 >> 19, y2 >> 19, uu >> 19, vv >> 19);
    }
}

/// `yuv2rgb_write_full` for rgb24.
#[inline]
fn put_full(t: &Yuv2Rgb, dest: &mut [u8], i: usize, y: i32, u: i32, v: i32) {
    let [yc, yo, v2r, v2g, u2g, u2b] = t.full;
    let mut y = y - yo;
    y = y.wrapping_mul(yc);
    y = y.wrapping_add(1 << 21);
    let mut r = (y as u32).wrapping_add((v as u32).wrapping_mul(v2r as u32)) as i32;
    let mut g = (y as u32).wrapping_add((v as u32).wrapping_mul(v2g as u32)).wrapping_add((u as u32).wrapping_mul(u2g as u32)) as i32;
    let mut b = (y as u32).wrapping_add((u as u32).wrapping_mul(u2b as u32)) as i32;
    if (r | g | b) as u32 & 0xC000_0000 != 0 {
        let clip = |a: i32| if a as u32 & !((1u32 << 30) - 1) != 0 { (!a >> 31) & ((1 << 30) - 1) } else { a };
        r = clip(r);
        g = clip(g);
        b = clip(b);
    }
    dest[3 * i] = (r >> 22) as u8;
    dest[3 * i + 1] = (g >> 22) as u8;
    dest[3 * i + 2] = (b >> 22) as u8;
}

/// `yuv2rgb24_full_2_c`.
#[allow(clippy::too_many_arguments)]
fn full_2(t: &Yuv2Rgb, dest: &mut [u8], l: &[&[i16]], u: &[&[i16]], v: &[&[i16]], w: usize, ya: i32, ca: i32) {
    let (ya1, ca1) = (4096 - ya, 4096 - ca);
    for i in 0..w {
        let y = (l[0][i] as i32 * ya1 + l[1][i] as i32 * ya) >> 10;
        let uu = (u[0][i] as i32 * ca1 + u[1][i] as i32 * ca - (128 << 19)) >> 10;
        let vv = (v[0][i] as i32 * ca1 + v[1][i] as i32 * ca - (128 << 19)) >> 10;
        put_full(t, dest, i, y, uu, vv);
    }
}

/// `yuv2rgb24_full_X_c`.
#[allow(clippy::too_many_arguments)]
fn full_x(t: &Yuv2Rgb, dest: &mut [u8], lf: &[i16], l: &[&[i16]], cf: &[i16], u: &[&[i16]], v: &[&[i16]], w: usize) {
    for i in 0..w {
        let mut y: i32 = 1 << 9;
        let mut uu: i32 = (1 << 9) - (128 << 19);
        let mut vv: i32 = (1 << 9) - (128 << 19);
        for j in 0..lf.len() {
            y += l[j][i] as i32 * lf[j] as i32;
        }
        for j in 0..cf.len() {
            uu += u[j][i] as i32 * cf[j] as i32;
            vv += v[j][i] as i32 * cf[j] as i32;
        }
        put_full(t, dest, i, y >> 10, uu >> 10, vv >> 10);
    }
}
