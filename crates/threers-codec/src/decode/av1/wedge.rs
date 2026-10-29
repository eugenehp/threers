//! Wedge and inter-intra blend masks (dav1d `src/wedge.c`).

use std::sync::OnceLock;

use super::levels::*;

const HOR: u8 = 0;
const VER: u8 = 1;
const O27: u8 = 2;
const O63: u8 = 3;
const O117: u8 = 4;
const O153: u8 = 5;

type Cb = [(u8, u8, u8); 16];
const CB_HGTW: Cb = [
    (O27, 4, 4), (O63, 4, 4), (O117, 4, 4), (O153, 4, 4),
    (HOR, 4, 2), (HOR, 4, 4), (HOR, 4, 6), (VER, 4, 4),
    (O27, 4, 2), (O27, 4, 6), (O153, 4, 2), (O153, 4, 6),
    (O63, 2, 4), (O63, 6, 4), (O117, 2, 4), (O117, 6, 4),
];
const CB_HLTW: Cb = [
    (O27, 4, 4), (O63, 4, 4), (O117, 4, 4), (O153, 4, 4),
    (VER, 2, 4), (VER, 4, 4), (VER, 6, 4), (HOR, 4, 4),
    (O27, 4, 2), (O27, 4, 6), (O153, 4, 2), (O153, 4, 6),
    (O63, 2, 4), (O63, 6, 4), (O117, 2, 4), (O117, 6, 4),
];
const CB_HEQW: Cb = [
    (O27, 4, 4), (O63, 4, 4), (O117, 4, 4), (O153, 4, 4),
    (HOR, 4, 2), (HOR, 4, 6), (VER, 2, 4), (VER, 6, 4),
    (O27, 4, 2), (O27, 4, 6), (O153, 4, 2), (O153, 4, 6),
    (O63, 2, 4), (O63, 6, 4), (O117, 2, 4), (O117, 6, 4),
];

pub struct Masks {
    /// `[layout 444/422/420][bs - BS_32X32][sign][wedge_idx]`.
    pub wedge: Vec<Vec<[[Vec<u8>; 16]; 2]>>,
    /// `[layout][bs - BS_32X32][interintra mode]`.
    pub ii: Vec<Vec<[Vec<u8>; 4]>>,
}

fn insert_border(dst: &mut [u8], src: &[u8; 8], ctr: i32) {
    if ctr > 4 {
        dst[..(ctr - 4) as usize].fill(0);
    }
    let d = (ctr.max(4) - 4) as usize;
    let s = (4 - ctr).max(0) as usize;
    let n = (64 - ctr).min(8) as usize;
    dst[d..d + n].copy_from_slice(&src[s..s + n]);
    if ctr < 64 - 4 {
        dst[(ctr + 4) as usize..64].fill(64);
    }
}

fn transpose(src: &[u8]) -> Vec<u8> {
    let mut d = vec![0; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            d[x * 64 + y] = src[y * 64 + x];
        }
    }
    d
}

fn hflip(src: &[u8]) -> Vec<u8> {
    let mut d = vec![0; 64 * 64];
    for y in 0..64 {
        for x in 0..64 {
            d[y * 64 + 63 - x] = src[y * 64 + x];
        }
    }
    d
}

fn init_chroma(luma: &[u8], sign: i32, w: usize, h: usize, ss_ver: usize) -> Vec<u8> {
    let mut c = vec![0u8; (w >> 1) * (h >> ss_ver)];
    let (mut lo, mut co) = (0, 0);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let mut sum = luma[lo + x] as i32 + luma[lo + x + 1] as i32 + 1;
            if ss_ver != 0 {
                sum += luma[lo + w + x] as i32 + luma[lo + w + x + 1] as i32 + 1;
            }
            c[co + (x >> 1)] = ((sum - sign) >> (1 + ss_ver)) as u8;
            x += 2;
        }
        lo += w << ss_ver;
        co += w >> 1;
        y += 1 + ss_ver;
    }
    c
}

fn build_nondc_ii(w: usize, h: usize, step: usize) -> [Vec<u8>; 3] {
    const W1D: [u8; 32] = [60, 52, 45, 39, 34, 30, 26, 22, 19, 17, 15, 13, 11, 10, 8, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1];
    let mut v = vec![0u8; w * h];
    let mut hm = vec![0u8; w * h];
    let mut sm = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = W1D[y * step];
            sm[y * w + x] = W1D[x.min(y) * step];
            hm[y * w + x] = W1D[x * step];
        }
    }
    [v, hm, sm]
}

pub fn masks() -> &'static Masks {
    static M: OnceLock<Masks> = OnceLock::new();
    M.get_or_init(|| {
        let border_odd = [1, 2, 6, 18, 37, 53, 60, 63];
        let border_even = [1, 4, 11, 27, 46, 58, 62, 63];
        let border_vert = [0, 2, 7, 21, 43, 57, 62, 64];
        let mut master: Vec<Vec<u8>> = vec![vec![0u8; 64 * 64]; 6];
        for y in 0..64 {
            insert_border(&mut master[VER as usize][y * 64..y * 64 + 64], &border_vert, 32);
        }
        let mut ctr = 48;
        let mut y = 0;
        while y < 64 {
            insert_border(&mut master[O63 as usize][y * 64..y * 64 + 64], &border_even, ctr);
            insert_border(&mut master[O63 as usize][(y + 1) * 64..(y + 1) * 64 + 64], &border_odd, ctr - 1);
            y += 2;
            ctr -= 1;
        }
        master[O27 as usize] = transpose(&master[O63 as usize]);
        master[HOR as usize] = transpose(&master[VER as usize]);
        master[O117 as usize] = hflip(&master[O63 as usize]);
        master[O153 as usize] = hflip(&master[O27 as usize]);

        let nb = BS_8X8 - BS_32X32 + 1;
        let empty = || -> [[Vec<u8>; 16]; 2] { Default::default() };
        let mut wedge: Vec<Vec<[[Vec<u8>; 16]; 2]>> = (0..3).map(|_| (0..nb).map(|_| empty()).collect()).collect();
        let fills: [(usize, usize, usize, &Cb, u32); 9] = [
            (32, 32, BS_32X32, &CB_HEQW, 0x7bfb),
            (32, 16, BS_32X16, &CB_HLTW, 0x7beb),
            (32, 8, BS_32X8, &CB_HLTW, 0x6beb),
            (16, 32, BS_16X32, &CB_HGTW, 0x7beb),
            (16, 16, BS_16X16, &CB_HEQW, 0x7bfb),
            (16, 8, BS_16X8, &CB_HLTW, 0x7beb),
            (8, 32, BS_8X32, &CB_HGTW, 0x7aeb),
            (8, 16, BS_8X16, &CB_HGTW, 0x7beb),
            (8, 8, BS_8X8, &CB_HEQW, 0x7bfb),
        ];
        for &(w, h, bs, cb, mut signs) in fills.iter() {
            let bi = bs - BS_32X32;
            for n in 0..16 {
                let sign = (signs & 1) as i32;
                let (dir, xo, yo) = cb[n];
                let x_off = 32 - (w * xo as usize >> 3);
                let y_off = 32 - (h * yo as usize >> 3);
                let mut m444 = vec![0u8; w * h];
                let src = &master[dir as usize];
                for yy in 0..h {
                    for xx in 0..w {
                        let v = src[(y_off + yy) * 64 + x_off + xx];
                        m444[yy * w + xx] = if sign != 0 { 64 - v } else { v };
                    }
                }
                // `offsets[c][bs].wedge[r][n]` is the chroma mask built with
                // rounding `r` (the storage slot depends on `sign`, the
                // offset does not).
                wedge[1][bi][0][n] = init_chroma(&m444, 0, w, h, 0);
                wedge[1][bi][1][n] = init_chroma(&m444, 1, w, h, 0);
                wedge[2][bi][0][n] = init_chroma(&m444, 0, w, h, 1);
                wedge[2][bi][1][n] = init_chroma(&m444, 1, w, h, 1);
                wedge[0][bi][0][n] = m444.clone();
                wedge[0][bi][1][n] = m444;
                signs >>= 1;
            }
        }
        let mut ii: Vec<Vec<[Vec<u8>; 4]>> = (0..3).map(|_| (0..nb).map(|_| Default::default()).collect()).collect();
        let dc = vec![32u8; 32 * 32];
        let nd = |w, h, s| build_nondc_ii(w, h, s);
        let n32x32 = nd(32, 32, 1);
        let n16x32 = nd(16, 32, 1);
        let n16x16 = nd(16, 16, 2);
        let n8x32 = nd(8, 32, 1);
        let n8x16 = nd(8, 16, 2);
        let n8x8 = nd(8, 8, 4);
        let n4x16 = nd(4, 16, 2);
        let n4x8 = nd(4, 8, 4);
        let n4x4 = nd(4, 4, 8);
        let assign: [(usize, [&[Vec<u8>; 3]; 3]); 7] = [
            (BS_32X32, [&n32x32, &n16x32, &n16x16]),
            (BS_32X16, [&n32x32, &n16x16, &n16x16]),
            (BS_16X32, [&n16x32, &n8x32, &n8x16]),
            (BS_16X16, [&n16x16, &n8x16, &n8x8]),
            (BS_16X8, [&n16x16, &n8x8, &n8x8]),
            (BS_8X16, [&n8x16, &n4x16, &n4x8]),
            (BS_8X8, [&n8x8, &n4x8, &n4x4]),
        ];
        for (bs, per) in assign.iter() {
            for c in 0..3 {
                let bi = bs - BS_32X32;
                ii[c][bi][II_DC_PRED as usize] = dc.clone();
                for p in 0..3 {
                    ii[c][bi][p + 1] = per[c][p].clone();
                }
            }
        }
        Masks { wedge, ii }
    })
}

/// `WEDGE_MASK(c, bs, sign, idx)`.
pub fn wedge_mask(c: usize, bs: usize, sign: usize, idx: usize) -> &'static [u8] {
    &masks().wedge[c][bs - BS_32X32][sign][idx]
}

/// `II_MASK(c, bs, b)`.
pub fn ii_mask(c: usize, bs: usize, interintra_type: u8, interintra_mode: u8, wedge_idx: u8) -> &'static [u8] {
    if interintra_type == INTER_INTRA_BLEND {
        &masks().ii[c][bs - BS_32X32][interintra_mode as usize]
    } else {
        wedge_mask(c, bs, 0, wedge_idx as usize)
    }
}
