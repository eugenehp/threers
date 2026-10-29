//! Motion compensation DSP (dav1d `src/mc_tmpl.c`, C paths), generic over
//! bit depth on `u16` samples. Sources are blocks fetched with edge
//! clamping (equivalent to dav1d's `emu_edge` + direct reads).

use super::tables::{MC_SUBPEL_FILTERS, MC_WARP_FILTER, OBMC_MASKS, RESIZE_FILTER};

#[derive(Clone, Copy)]
pub struct Bd {
    pub max: i32,
    pub bits: u32,
}

impl Bd {
    pub fn new(bitdepth_max: i32) -> Self {
        Bd { max: bitdepth_max, bits: 32 - (bitdepth_max as u32).leading_zeros() }
    }
    #[inline]
    pub fn ib(&self) -> i32 {
        if self.bits == 8 {
            4
        } else {
            14 - self.bits as i32
        }
    }
    #[inline]
    pub fn bias(&self) -> i32 {
        if self.bits == 8 {
            0
        } else {
            8192
        }
    }
    #[inline]
    pub fn clip(&self, v: i32) -> u16 {
        v.clamp(0, self.max) as u16
    }
}

/// A source block: `buf[y * stride + x]` covers (x0-3.., y0-3..) when built
/// by [`fetch`], with the requested origin at `off`.
pub struct Src<'a> {
    pub buf: &'a [u16],
    pub off: usize,
    pub stride: usize,
}

impl Src<'_> {
    #[inline]
    fn at(&self, x: isize, y: isize) -> i32 {
        self.buf[(self.off as isize + y * self.stride as isize + x) as usize] as i32
    }
}

/// Clamped fetch of `bw x bh` samples at (x, y) from a plane of visible
/// size `iw x ih` (`emu_edge_c`).
#[allow(clippy::too_many_arguments)]
pub fn fetch(out: &mut Vec<u16>, plane: &[u16], stride: usize, iw: i32, ih: i32, x: i32, y: i32, bw: i32, bh: i32) {
    out.clear();
    out.reserve((bw * bh) as usize);
    // Columns left of the picture repeat its first sample, right of it its
    // last; the run between is one copy per row.
    let left = (-x).clamp(0, bw) as usize;
    let right = (x + bw - iw).clamp(0, bw - left as i32) as usize;
    let mid = bw as usize - left - right;
    let mx = x.clamp(0, iw - 1) as usize;
    for j in 0..bh {
        let row = &plane[(y + j).clamp(0, ih - 1) as usize * stride..];
        out.extend(std::iter::repeat_n(row[0], left));
        out.extend_from_slice(&row[mx..mx + mid]);
        out.extend(std::iter::repeat_n(row[iw as usize - 1], right));
    }
}

/// (type_h, type_v) of each `Filter2d`.
const FT: [(usize, usize); 9] = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (1, 0), (1, 1), (1, 2)];

#[inline]
fn h_filter(mx: i32, w: i32, th: usize) -> Option<&'static [i8; 8]> {
    if mx == 0 {
        None
    } else if w > 4 {
        Some(&MC_SUBPEL_FILTERS[th][mx as usize - 1])
    } else {
        Some(&MC_SUBPEL_FILTERS[3 + (th & 1)][mx as usize - 1])
    }
}

#[inline]
fn v_filter(my: i32, h: i32, tv: usize) -> Option<&'static [i8; 8]> {
    if my == 0 {
        None
    } else if h > 4 {
        Some(&MC_SUBPEL_FILTERS[tv][my as usize - 1])
    } else {
        Some(&MC_SUBPEL_FILTERS[3 + (tv & 1)][my as usize - 1])
    }
}

#[inline]
fn f8(s: &Src, x: isize, y: isize, f: &[i8; 8], dx: isize, dy: isize) -> i32 {
    let mut acc = 0;
    for k in 0..8 {
        acc += f[k] as i32 * s.at(x + (k as isize - 3) * dx, y + (k as isize - 3) * dy);
    }
    acc
}

#[inline]
fn f8m(mid: &[i32], o: isize, f: &[i8; 8], stride: isize) -> i32 {
    let mut acc = 0;
    for k in 0..8 {
        acc += f[k] as i32 * mid[(o + (k as isize - 3) * stride) as usize];
    }
    acc
}

#[inline]
fn rnd(v: i32, sh: i32) -> i32 {
    (v + ((1 << sh) >> 1)) >> sh
}

/// Output of a `put` (pixels) or `prep` (int16 intermediates).
pub enum Dst<'a> {
    Px { buf: &'a mut [u16], off: usize, stride: usize },
    Tmp { buf: &'a mut [i16], stride: usize },
}

impl Dst<'_> {
    #[inline]
    fn px(&mut self, x: i32, y: i32, v: u16) {
        if let Dst::Px { buf, off, stride } = self {
            buf[*off + y as usize * *stride + x as usize] = v;
        }
    }
    #[inline]
    fn tmp(&mut self, x: i32, y: i32, v: i32) {
        if let Dst::Tmp { buf, stride } = self {
            buf[y as usize * *stride + x as usize] = v as i16;
        }
    }
    fn is_px(&self) -> bool {
        matches!(self, Dst::Px { .. })
    }
}

/// `put_8tap_c` / `prep_8tap_c` / `put_bilin_c` / `prep_bilin_c`.
#[allow(clippy::too_many_arguments)]
pub fn mc(dst: &mut Dst, src: &Src, w: i32, h: i32, mx: i32, my: i32, filter2d: usize, bd: Bd) {
    let ib = bd.ib();
    let bias = bd.bias();
    let put = dst.is_px();
    if filter2d == 9 {
        // bilinear
        let fb = |s: &Src, x: isize, y: isize, m: i32, dx: isize, dy: isize| 16 * s.at(x, y) + m * (s.at(x + dx, y + dy) - s.at(x, y));
        if mx != 0 {
            if my != 0 {
                let mut mid = vec![0i32; 128 * (h as usize + 1)];
                for y in 0..=h {
                    for x in 0..w {
                        mid[y as usize * 128 + x as usize] = rnd(fb(src, x as isize, y as isize, mx, 1, 0), 4 - ib);
                    }
                }
                for y in 0..h {
                    for x in 0..w {
                        let o = y as usize * 128 + x as usize;
                        let v = 16 * mid[o] + my * (mid[o + 128] - mid[o]);
                        if put {
                            dst.px(x, y, bd.clip(rnd(v, 4 + ib)));
                        } else {
                            dst.tmp(x, y, rnd(v, 4) - bias);
                        }
                    }
                }
            } else {
                let irnd = (1 << ib) >> 1;
                for y in 0..h {
                    for x in 0..w {
                        let px = rnd(fb(src, x as isize, y as isize, mx, 1, 0), 4 - ib);
                        if put {
                            dst.px(x, y, bd.clip((px + irnd) >> ib));
                        } else {
                            dst.tmp(x, y, px - bias);
                        }
                    }
                }
            }
        } else if my != 0 {
            for y in 0..h {
                for x in 0..w {
                    let v = fb(src, x as isize, y as isize, my, 0, 1);
                    if put {
                        dst.px(x, y, bd.clip(rnd(v, 4)));
                    } else {
                        dst.tmp(x, y, rnd(v, 4 - ib) - bias);
                    }
                }
            }
        } else {
            copy(dst, src, w, h, bd);
        }
        return;
    }
    let (th, tv) = FT[filter2d];
    let fh = h_filter(mx, w, th);
    let fv = v_filter(my, h, tv);
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let mut mid = vec![0i32; 128 * (h as usize + 7)];
            for y in 0..h + 7 {
                for x in 0..w {
                    mid[y as usize * 128 + x as usize] = rnd(f8(src, x as isize, y as isize - 3, fh, 1, 0), 6 - ib);
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let o = (y as isize + 3) * 128 + x as isize;
                    let v = f8m(&mid, o, fv, 128);
                    if put {
                        dst.px(x, y, bd.clip(rnd(v, 6 + ib)));
                    } else {
                        dst.tmp(x, y, rnd(v, 6) - bias);
                    }
                }
            }
        }
        (Some(fh), None) => {
            let irnd = 32 + ((1 << (6 - ib)) >> 1);
            for y in 0..h {
                for x in 0..w {
                    let v = f8(src, x as isize, y as isize, fh, 1, 0);
                    if put {
                        dst.px(x, y, bd.clip((v + irnd) >> 6));
                    } else {
                        dst.tmp(x, y, rnd(v, 6 - ib) - bias);
                    }
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let v = f8(src, x as isize, y as isize, fv, 0, 1);
                    if put {
                        dst.px(x, y, bd.clip(rnd(v, 6)));
                    } else {
                        dst.tmp(x, y, rnd(v, 6 - ib) - bias);
                    }
                }
            }
        }
        (None, None) => copy(dst, src, w, h, bd),
    }
}

fn copy(dst: &mut Dst, src: &Src, w: i32, h: i32, bd: Bd) {
    let ib = bd.ib();
    let bias = bd.bias();
    let put = dst.is_px();
    for y in 0..h {
        for x in 0..w {
            let v = src.at(x as isize, y as isize);
            if put {
                dst.px(x, y, v as u16);
            } else {
                dst.tmp(x, y, (v << ib) - bias);
            }
        }
    }
}

/// `put/prep_8tap_scaled_c`, `put/prep_bilin_scaled_c`. `src` is positioned
/// at the block's top-left integer sample.
#[allow(clippy::too_many_arguments)]
pub fn mc_scaled(dst: &mut Dst, src: &Src, w: i32, h: i32, mx: i32, mut my: i32, dx: i32, dy: i32, filter2d: usize, bd: Bd) {
    let ib = bd.ib();
    let bias = bd.bias();
    let put = dst.is_px();
    if filter2d == 9 {
        let mut mid = [[0i32; 128]; 2];
        let mut in_y = -2;
        let mut sy: isize = 0;
        for yy in 0..h {
            let y = my >> 10;
            let dmy = my & 0x3ff;
            while in_y < y {
                let (mut imx, mut ioff) = (mx, 0isize);
                let row = (in_y & 1) as usize;
                for x in 0..w as usize {
                    let m = imx >> 6;
                    let a = src.at(ioff, sy);
                    let b = src.at(ioff + 1, sy);
                    mid[row][x] = rnd(16 * a + m * (b - a), 4 - ib);
                    imx += dx;
                    ioff += (imx >> 10) as isize;
                    imx &= 0x3ff;
                }
                sy += 1;
                in_y += 1;
            }
            let (m1, m2) = ((y & 1) as usize, ((y + 1) & 1) as usize);
            for x in 0..w {
                let a = mid[m1][x as usize];
                let b = mid[m2][x as usize];
                let v = 16 * a + (dmy >> 6) * (b - a);
                if put {
                    dst.px(x, yy, bd.clip(rnd(v, 4 + ib)));
                } else {
                    dst.tmp(x, yy, rnd(v, 4) - bias);
                }
            }
            my += dy;
        }
        return;
    }
    let (th, tv) = FT[filter2d];
    let irnd = (1 << ib) >> 1;
    let mut mid = [[0i32; 128]; 8];
    let mut ptrs = [0usize, 1, 2, 3, 4, 5, 6, 7];
    let mut in_y = -8;
    let mut sy: isize = -3;
    for yy in 0..h {
        let src_y = my >> 10;
        let fv = v_filter((my & 0x3ff) >> 6, h, tv);
        while in_y < src_y {
            let (mut imx, mut ioff) = (mx, 0isize);
            let p0 = ptrs[0];
            ptrs.rotate_left(1);
            ptrs[7] = p0;
            for x in 0..w as usize {
                let fh = h_filter(imx >> 6, w, th);
                mid[p0][x] = match fh {
                    Some(f) => rnd(f8(src, ioff, sy, f, 1, 0), 6 - ib),
                    None => src.at(ioff, sy) << ib,
                };
                imx += dx;
                ioff += (imx >> 10) as isize;
                imx &= 0x3ff;
            }
            sy += 1;
            in_y += 1;
        }
        for x in 0..w as usize {
            let v = match fv {
                Some(f) => {
                    let mut acc = 0;
                    for k in 0..8 {
                        acc += f[k] as i32 * mid[ptrs[k]][x];
                    }
                    Some(acc)
                }
                None => None,
            };
            if put {
                let p = match v {
                    Some(a) => bd.clip(rnd(a, 6 + ib)),
                    None => bd.clip((mid[ptrs[3]][x] + irnd) >> ib),
                };
                dst.px(x as i32, yy, p);
            } else {
                let t = match v {
                    Some(a) => rnd(a, 6),
                    None => mid[ptrs[3]][x],
                };
                dst.tmp(x as i32, yy, t - bias);
            }
        }
        my += dy;
    }
}

pub fn avg(dst: &mut [u16], off: usize, stride: usize, t1: &[i16], t2: &[i16], w: usize, h: usize, bd: Bd) {
    let ib = bd.ib();
    let sh = ib + 1;
    let r = (1 << ib) + bd.bias() * 2;
    for y in 0..h {
        for x in 0..w {
            let v = t1[y * w + x] as i32 + t2[y * w + x] as i32;
            dst[off + y * stride + x] = bd.clip((v + r) >> sh);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn w_avg(dst: &mut [u16], off: usize, stride: usize, t1: &[i16], t2: &[i16], w: usize, h: usize, weight: i32, bd: Bd) {
    let ib = bd.ib();
    let sh = ib + 4;
    let r = (8 << ib) + bd.bias() * 16;
    for y in 0..h {
        for x in 0..w {
            let v = t1[y * w + x] as i32 * weight + t2[y * w + x] as i32 * (16 - weight);
            dst[off + y * stride + x] = bd.clip((v + r) >> sh);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn mask(dst: &mut [u16], off: usize, stride: usize, t1: &[i16], t2: &[i16], w: usize, h: usize, m: &[u8], bd: Bd) {
    let ib = bd.ib();
    let sh = ib + 6;
    let r = (32 << ib) + bd.bias() * 64;
    for y in 0..h {
        for x in 0..w {
            let mm = m[y * w + x] as i32;
            let v = t1[y * w + x] as i32 * mm + t2[y * w + x] as i32 * (64 - mm);
            dst[off + y * stride + x] = bd.clip((v + r) >> sh);
        }
    }
}

#[inline]
fn blend_px(a: i32, b: i32, m: i32) -> u16 {
    (((a * (64 - m) + b * m) + 32) >> 6) as u16
}

pub fn blend(dst: &mut [u16], off: usize, stride: usize, tmp: &[u16], w: usize, h: usize, m: &[u8]) {
    for y in 0..h {
        for x in 0..w {
            let d = &mut dst[off + y * stride + x];
            *d = blend_px(*d as i32, tmp[y * w + x] as i32, m[y * w + x] as i32);
        }
    }
}

pub fn blend_v(dst: &mut [u16], off: usize, stride: usize, tmp: &[u16], w: usize, h: usize) {
    let m = &OBMC_MASKS[w..];
    for y in 0..h {
        for x in 0..(w * 3) >> 2 {
            let d = &mut dst[off + y * stride + x];
            *d = blend_px(*d as i32, tmp[y * w + x] as i32, m[x] as i32);
        }
    }
}

pub fn blend_h(dst: &mut [u16], off: usize, stride: usize, tmp: &[u16], w: usize, h: usize) {
    let m = &OBMC_MASKS[h..];
    for y in 0..(h * 3) >> 2 {
        for x in 0..w {
            let d = &mut dst[off + y * stride + x];
            *d = blend_px(*d as i32, tmp[y * w + x] as i32, m[y] as i32);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn w_mask(dst: &mut [u16], off: usize, stride: usize, t1: &[i16], t2: &[i16], w: usize, h: usize, mask: &mut [u8], sign: i32, ss_hor: bool, ss_ver: bool, bd: Bd) {
    let ib = bd.ib();
    let sh = ib + 6;
    let r = (32 << ib) + bd.bias() * 64;
    let mask_sh = bd.bits as i32 + ib - 4;
    let mask_rnd = 1 << (mask_sh - 5);
    let mut mo = 0usize;
    let mut hh = h as i32;
    for y in 0..h {
        let mut x = 0;
        while x < w {
            let a = t1[y * w + x] as i32;
            let b = t2[y * w + x] as i32;
            let m = (38 + (((a - b).abs() + mask_rnd) >> mask_sh)).min(64);
            dst[off + y * stride + x] = bd.clip((a * m + b * (64 - m) + r) >> sh);
            if ss_hor {
                x += 1;
                let a = t1[y * w + x] as i32;
                let b = t2[y * w + x] as i32;
                let n = (38 + (((a - b).abs() + mask_rnd) >> mask_sh)).min(64);
                dst[off + y * stride + x] = bd.clip((a * n + b * (64 - n) + r) >> sh);
                let mi = mo + (x >> 1);
                if hh & ss_ver as i32 != 0 {
                    mask[mi] = ((m + n + mask[mi] as i32 + 2 - sign) >> 2) as u8;
                } else if ss_ver {
                    mask[mi] = (m + n) as u8;
                } else {
                    mask[mi] = ((m + n + 1 - sign) >> 1) as u8;
                }
            } else {
                mask[mo + x] = m as u8;
            }
            x += 1;
        }
        if !ss_ver || hh & 1 != 0 {
            mo += w >> ss_hor as usize;
        }
        hh -= 1;
    }
}

/// `warp_affine_8x8_c` / `_8x8t_c`; `src` positioned at the 8x8 origin.
pub fn warp8x8(dst: &mut Dst, dx0: i32, dy0: i32, src: &Src, abcd: &[i16; 4], mut mx: i32, mut my: i32, bd: Bd) {
    let ib = bd.ib();
    let mut mid = [0i32; 15 * 8];
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[(64 + ((tmx + 512) >> 10)) as usize];
            let mut acc = 0;
            for k in 0..8 {
                acc += f[k] as i32 * src.at(x as isize + k as isize - 3, y as isize - 3);
            }
            mid[y * 8 + x] = rnd(acc, 7 - ib);
            tmx += abcd[0] as i32;
        }
        mx += abcd[1] as i32;
    }
    let put = dst.is_px();
    for y in 0..8 {
        let mut tmy = my;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[(64 + ((tmy + 512) >> 10)) as usize];
            let mut acc = 0;
            for k in 0..8 {
                acc += f[k] as i32 * mid[(y + k) * 8 + x];
            }
            if put {
                dst.px(dx0 + x as i32, dy0 + y as i32, bd.clip(rnd(acc, 7 + ib)));
            } else {
                dst.tmp(dx0 + x as i32, dy0 + y as i32, rnd(acc, 7) - bd.bias());
            }
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

/// `resize_c` for one row.
#[allow(clippy::too_many_arguments)]
pub fn resize_row(dst: &mut [u16], src: &[u16], dst_w: usize, src_w: i32, dx: i32, mx0: i32, bd: Bd) {
    let mut mx = mx0;
    let mut src_x = -1i32;
    for d in dst.iter_mut().take(dst_w) {
        let f = &RESIZE_FILTER[(mx >> 8) as usize];
        let mut acc = 0;
        for k in 0..8 {
            acc += f[k] as i32 * src[(src_x + k as i32 - 3).clamp(0, src_w - 1) as usize] as i32;
        }
        *d = bd.clip((-acc + 64) >> 7);
        mx += dx;
        src_x += mx >> 14;
        mx &= 0x3fff;
    }
}
