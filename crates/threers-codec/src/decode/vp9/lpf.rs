//! VP9 loop filter driver (ffmpeg `vp9lpf.c`), run superblock by
//! superblock in raster order after the whole frame is reconstructed.

use super::block::FrameDec;
use super::dsp::loop_filter;

struct Lut<'a> {
    lim: &'a [u8; 64],
    mblim: &'a [u8; 64],
    bpp: u32,
}

impl Lut<'_> {
    #[inline]
    fn eih(&self, l: usize) -> (i32, i32, i32) {
        (self.mblim[l] as i32, self.lim[l] as i32, (l >> 4) as i32)
    }
}

/// `loop_filter_8[wd][dir]` on 8 lines at `off`.
#[inline]
fn lf8(d: &mut [u16], off: usize, ls: usize, dir: u32, wd: u32, (e, i, h): (i32, i32, i32), bpp: u32) {
    if dir == 0 {
        loop_filter(d, off, ls as isize, 1, e, i, h, wd, bpp);
    } else {
        loop_filter(d, off, 1, ls as isize, e, i, h, wd, bpp);
    }
}

/// `loop_filter_16[dir]`: two 8-line groups, 16 wide.
#[inline]
fn lf16(d: &mut [u16], off: usize, ls: usize, dir: u32, eih: (i32, i32, i32), bpp: u32) {
    lf8(d, off, ls, dir, 16, eih, bpp);
    let next = if dir == 0 { 8 * ls } else { 8 };
    lf8(d, off + next, ls, dir, 16, eih, bpp);
}

/// `loop_filter_mix2[wd1 is 8][wd2 is 8][dir]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn mix2(d: &mut [u16], off: usize, ls: usize, dir: u32, w1: bool, w2: bool, a: (i32, i32, i32), b: (i32, i32, i32), bpp: u32) {
    lf8(d, off, ls, dir, if w1 { 8 } else { 4 }, a, bpp);
    let next = if dir == 0 { 8 * ls } else { 8 };
    lf8(d, off + next, ls, dir, if w2 { 8 } else { 4 }, b, bpp);
}

/// `filter_plane_cols`.
#[allow(clippy::too_many_arguments)]
fn filter_plane_cols(lut: &Lut, col: usize, ss_h: usize, ss_v: usize, lvl: &[u8; 64], mask: &[[u8; 4]; 8], d: &mut [u16], dst0: usize, ls: usize) {
    let bpp = lut.bpp;
    let mut dst = dst0;
    let mut lv = 0usize;
    let mut y = 0;
    while y < 8 {
        let mut ptr = dst;
        let mut l = lv;
        let hmask1 = mask[y];
        let hmask2 = mask[y + 1 + ss_v];
        let hm1 = (hmask1[0] | hmask1[1] | hmask1[2]) as u32;
        let hm13 = hmask1[3] as u32;
        let hm2 = (hmask2[1] | hmask2[2]) as u32;
        let hm23 = hmask2[3] as u32;
        let hm = hm1 | hm2 | hm13 | hm23;
        let mut x = 1u32;
        while hm & !(x - 1) != 0 {
            if col != 0 || x > 1 {
                if hm1 & x != 0 {
                    let l0 = lvl[l] as usize;
                    let a = lut.eih(l0);
                    if hmask1[0] as u32 & x != 0 {
                        if hmask2[0] as u32 & x != 0 {
                            lf16(d, ptr, ls, 0, a, bpp);
                        } else {
                            lf8(d, ptr, ls, 0, 16, a, bpp);
                        }
                    } else if hm2 & x != 0 {
                        let b = lut.eih(lvl[l + (8 << ss_v)] as usize);
                        mix2(d, ptr, ls, 0, hmask1[1] as u32 & x != 0, hmask2[1] as u32 & x != 0, a, b, bpp);
                    } else {
                        lf8(d, ptr, ls, 0, if hmask1[1] as u32 & x != 0 { 8 } else { 4 }, a, bpp);
                    }
                } else if hm2 & x != 0 {
                    let b = lut.eih(lvl[l + (8 << ss_v)] as usize);
                    lf8(d, ptr + 8 * ls, ls, 0, if hmask2[1] as u32 & x != 0 { 8 } else { 4 }, b, bpp);
                }
            }
            if ss_h != 0 {
                if x & 0xAA != 0 {
                    l += 2;
                }
            } else {
                if hm13 & x != 0 {
                    let a = lut.eih(lvl[l] as usize);
                    if hm23 & x != 0 {
                        let b = lut.eih(lvl[l + (8 << ss_v)] as usize);
                        mix2(d, ptr + 4, ls, 0, false, false, a, b, bpp);
                    } else {
                        lf8(d, ptr + 4, ls, 0, 4, a, bpp);
                    }
                } else if hm23 & x != 0 {
                    let b = lut.eih(lvl[l + (8 << ss_v)] as usize);
                    lf8(d, ptr + 8 * ls + 4, ls, 0, 4, b, bpp);
                }
                l += 1;
            }
            x <<= 1;
            ptr += 8 >> ss_h;
        }
        y += 2 << ss_v;
        dst += 16 * ls;
        lv += 16 << ss_v;
    }
}

/// `filter_plane_rows`.
#[allow(clippy::too_many_arguments)]
fn filter_plane_rows(lut: &Lut, row: usize, ss_h: usize, ss_v: usize, lvl: &[u8; 64], mask: &[[u8; 4]; 8], d: &mut [u16], dst0: usize, ls: usize) {
    let bpp = lut.bpp;
    let mut dst = dst0;
    let mut lv = 0usize;
    for y in 0..8 {
        let mut ptr = dst;
        let mut l = lv;
        let vmask = mask[y];
        let vm = (vmask[0] | vmask[1] | vmask[2]) as u32;
        let vm3 = vmask[3] as u32;
        let mut x = 1u32;
        while vm & !(x - 1) != 0 {
            let xn = x << (1 + ss_h);
            if row != 0 || y != 0 {
                if vm & x != 0 {
                    let a = lut.eih(lvl[l] as usize);
                    if vmask[0] as u32 & x != 0 {
                        if vmask[0] as u32 & xn != 0 {
                            lf16(d, ptr, ls, 1, a, bpp);
                        } else {
                            lf8(d, ptr, ls, 1, 16, a, bpp);
                        }
                    } else if vm & xn != 0 {
                        let b = lut.eih(lvl[l + 1 + ss_h] as usize);
                        mix2(d, ptr, ls, 1, vmask[1] as u32 & x != 0, vmask[1] as u32 & xn != 0, a, b, bpp);
                    } else {
                        lf8(d, ptr, ls, 1, if vmask[1] as u32 & x != 0 { 8 } else { 4 }, a, bpp);
                    }
                } else if vm & xn != 0 {
                    let b = lut.eih(lvl[l + 1 + ss_h] as usize);
                    lf8(d, ptr + 8, ls, 1, if vmask[1] as u32 & xn != 0 { 8 } else { 4 }, b, bpp);
                }
            }
            if ss_v == 0 {
                if vm3 & x != 0 {
                    let a = lut.eih(lvl[l] as usize);
                    if vm3 & xn != 0 {
                        let b = lut.eih(lvl[l + 1 + ss_h] as usize);
                        mix2(d, ptr + ls * 4, ls, 1, false, false, a, b, bpp);
                    } else {
                        lf8(d, ptr + ls * 4, ls, 1, 4, a, bpp);
                    }
                } else if vm3 & xn != 0 {
                    let b = lut.eih(lvl[l + 1 + ss_h] as usize);
                    lf8(d, ptr + ls * 4 + 8, ls, 1, 4, b, bpp);
                }
            }
            x <<= 2 << ss_h;
            ptr += 16;
            l += 2 << ss_h;
        }
        dst += (8 * ls) >> ss_v;
        if ss_v != 0 {
            if y & 1 != 0 {
                lv += 16;
            }
        } else {
            lv += 8;
        }
    }
}

/// Filter the whole frame (`ff_vp9_loopfilter_sb` for every superblock).
pub fn loop_filter_frame(fd: &mut FrameDec, lim: &[u8; 64], mblim: &[u8; 64]) {
    let lut = Lut { lim, mblim, bpp: fd.bpp };
    let (ss_h, ss_v) = (fd.ss_h, fd.ss_v);
    let uvm = (ss_h | ss_v != 0) as usize;
    for sbr in 0..fd.sb_rows {
        for sbc in 0..fd.sb_cols {
            let (row, col) = (sbr * 8, sbc * 8);
            let lf = &fd.lflvl[sbr * fd.sb_cols + sbc];
            let ls = fd.cur.planes[0].width;
            let off = row * 8 * ls + col * 8;
            filter_plane_cols(&lut, col, 0, 0, &lf.level, &lf.mask[0][0], &mut fd.cur.planes[0].data, off, ls);
            filter_plane_rows(&lut, row, 0, 0, &lf.level, &lf.mask[0][1], &mut fd.cur.planes[0].data, off, ls);
            let lsc = fd.cur.planes[1].width;
            let offc = ((row * 8) >> ss_v) * lsc + ((col * 8) >> ss_h);
            for p in 1..3 {
                filter_plane_cols(&lut, col, ss_h, ss_v, &lf.level, &lf.mask[uvm][0], &mut fd.cur.planes[p].data, offc, lsc);
                filter_plane_rows(&lut, row, ss_h, ss_v, &lf.level, &lf.mask[uvm][1], &mut fd.cur.planes[p].data, offc, lsc);
            }
        }
    }
}
