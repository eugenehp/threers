//! Warped-motion model fitting and shear parameters (dav1d `src/warpmv.c`).
//! `DIV_LUT` is extracted from the C source text by a one-off script.

use super::env::Mv;
use super::obu::WarpedMotionParams;

#[rustfmt::skip]
static DIV_LUT: [u16; 257] = [
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768, 15709, 15650, 15592, 15534, 15477,
    15420, 15364, 15308, 15252, 15197, 15142, 15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614,
    14564, 14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843,
    13797, 13752, 13707, 13662, 13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148,
    13107, 13066, 13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633, 12596, 12558, 12520,
    12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228, 12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950,
    11916, 11882, 11848, 11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491, 11460, 11429,
    11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951,
    10923, 10894, 10866, 10838, 10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538, 10512,
    10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255, 10230, 10205, 10180, 10156, 10131, 10107,
    10082, 10058, 10034, 10010, 9986, 9963, 9939, 9916, 9892, 9869, 9846, 9823, 9800, 9777, 9754, 9732,
    9709, 9687, 9664, 9642, 9620, 9598, 9576, 9554, 9533, 9511, 9489, 9468, 9447, 9425, 9404, 9383,
    9362, 9341, 9321, 9300, 9279, 9259, 9239, 9218, 9198, 9178, 9158, 9138, 9118, 9098, 9079, 9059,
    9039, 9020, 9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867, 8849, 8830, 8812, 8793, 8775, 8756,
    8738, 8720, 8702, 8684, 8666, 8648, 8630, 8613, 8595, 8577, 8560, 8542, 8525, 8508, 8490, 8473,
    8456, 8439, 8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306, 8289, 8273, 8257, 8240, 8224, 8208,
    8192,
];

fn iclip_wmp(v: i32) -> i16 {
    let cv = v.clamp(i16::MIN as i32, i16::MAX as i32);
    let r = (cv.abs() + 32) >> 6;
    ((if cv < 0 { -r } else { r }) * (1 << 6)) as i16
}

fn resolve_divisor_32(d: u32) -> (i32, i32) {
    let mut shift = 31 - d.leading_zeros() as i32;
    let e = d as i32 - (1 << shift);
    let f = if shift > 8 { (e + (1 << (shift - 9))) >> (shift - 8) } else { e << (8 - shift) };
    shift += 14;
    (DIV_LUT[f as usize] as i32, shift)
}

#[inline]
fn sign64(v: i32, s: i64) -> i32 {
    if s < 0 { -v } else { v }
}

/// `dav1d_get_shear_params`: true if the model is invalid.
pub fn get_shear_params(wm: &mut WarpedMotionParams) -> bool {
    let mat = wm.matrix;
    if mat[2] <= 0 {
        return true;
    }
    wm.abcd[0] = iclip_wmp(mat[2] - 0x10000);
    wm.abcd[1] = iclip_wmp(mat[3]);
    let (div, shift) = resolve_divisor_32(mat[2].unsigned_abs());
    let y = if mat[2] < 0 { -div } else { div };
    let v1 = (mat[4] as i64 * 0x10000) * y as i64;
    let rnd = (1i64 << shift) >> 1;
    wm.abcd[2] = iclip_wmp(sign64(((v1.abs() + rnd) >> shift) as i32, v1));
    let v2 = (mat[3] as i64 * mat[4] as i64) * y as i64;
    wm.abcd[3] = iclip_wmp(mat[5] - sign64(((v2.abs() + rnd) >> shift) as i32, v2) - 0x10000);
    let a = wm.abcd.map(|v| (v as i32).abs());
    (4 * a[0] + 7 * a[1] >= 0x10000) || (4 * a[2] + 4 * a[3] >= 0x10000)
}

fn resolve_divisor_64(d: u64) -> (i32, i32) {
    let mut shift = 63 - d.leading_zeros() as i32;
    let e = d as i64 - (1i64 << shift);
    let f = if shift > 8 { (e + (1i64 << (shift - 9))) >> (shift - 8) } else { e << (8 - shift) };
    shift += 14;
    (DIV_LUT[f as usize] as i32, shift)
}

fn get_mult_shift(px: i64, idet: i32, shift: i32, lo: i32, hi: i32) -> i32 {
    let v1 = px.wrapping_mul(idet as i64);
    let v2 = sign64(((v1.unsigned_abs() as i64 + ((1i64 << shift) >> 1)) >> shift) as i32, v1);
    v2.clamp(lo, hi)
}

/// `dav1d_find_affine_int`: true on failure.
pub fn find_affine_int(pts: &[[[i32; 2]; 2]], np: usize, bw4: i32, bh4: i32, mv: Mv, wm: &mut WarpedMotionParams, bx4: i32, by4: i32) -> bool {
    let mut a = [[0i32; 2]; 2];
    let mut bx = [0i32; 2];
    let mut by = [0i32; 2];
    let rsuy = 2 * bh4 - 1;
    let rsux = 2 * bw4 - 1;
    let suy = rsuy * 8;
    let sux = rsux * 8;
    let duy = suy + mv.y as i32;
    let dux = sux + mv.x as i32;
    let isuy = by4 * 4 + rsuy;
    let isux = bx4 * 4 + rsux;
    for p in pts.iter().take(np) {
        let dx = p[1][0] - dux;
        let dy = p[1][1] - duy;
        let sx = p[0][0] - sux;
        let sy = p[0][1] - suy;
        if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
            a[0][0] = a[0][0].wrapping_add(((sx * sx) >> 2) + sx * 2 + 8);
            a[0][1] = a[0][1].wrapping_add(((sx * sy) >> 2) + sx + sy + 4);
            a[1][1] = a[1][1].wrapping_add(((sy * sy) >> 2) + sy * 2 + 8);
            bx[0] = bx[0].wrapping_add(((sx * dx) >> 2) + sx + dx + 8);
            bx[1] = bx[1].wrapping_add(((sy * dx) >> 2) + sy + dx + 4);
            by[0] = by[0].wrapping_add(((sx * dy) >> 2) + sx + dy + 4);
            by[1] = by[1].wrapping_add(((sy * dy) >> 2) + sy + dy + 8);
        }
    }
    let det = a[0][0] as i64 * a[1][1] as i64 - a[0][1] as i64 * a[0][1] as i64;
    if det == 0 {
        return true;
    }
    let (d, mut shift) = resolve_divisor_64(det.unsigned_abs());
    let mut idet = if det < 0 { -d } else { d };
    shift -= 16;
    if shift < 0 {
        idet <<= -shift;
        shift = 0;
    }
    let mat = &mut wm.matrix;
    mat[2] = get_mult_shift(a[1][1] as i64 * bx[0] as i64 - a[0][1] as i64 * bx[1] as i64, idet, shift, 0xe001, 0x11fff);
    mat[3] = get_mult_shift(a[0][0] as i64 * bx[1] as i64 - a[0][1] as i64 * bx[0] as i64, idet, shift, -0x1fff, 0x1fff);
    mat[4] = get_mult_shift(a[1][1] as i64 * by[0] as i64 - a[0][1] as i64 * by[1] as i64, idet, shift, -0x1fff, 0x1fff);
    mat[5] = get_mult_shift(a[0][0] as i64 * by[1] as i64 - a[0][1] as i64 * by[0] as i64, idet, shift, 0xe001, 0x11fff);
    mat[0] = (mv.x as i32 * 0x2000 - (isux * (mat[2] - 0x10000) + isuy * mat[3])).clamp(-0x800000, 0x7fffff);
    mat[1] = (mv.y as i32 * 0x2000 - (isux * mat[4] + isuy * (mat[5] - 0x10000))).clamp(-0x800000, 0x7fffff);
    false
}
