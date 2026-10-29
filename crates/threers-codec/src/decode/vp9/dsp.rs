//! VP9 DSP (ffmpeg `vp9dsp_template.c`): intra prediction, inverse
//! transforms, the loop filter kernel, and motion compensation, for every
//! bit depth on `u16` samples.

use super::data::SUBPEL_FILTERS;
use super::*;

#[inline(always)]
fn clip(v: i32, max: i32) -> u16 {
    v.clamp(0, max) as u16
}

/// Intra prediction of an `n × n` block (`n = 4 << tx`) at `off`.
///
/// `top[0]` is the top-left sample and `top[1..]` the row above (`2n`
/// for 4×4 diagonal modes); `left` is the column to the left in ffmpeg's
/// order — bottom to top, except top to bottom for `HOR_UP_PRED`.
#[allow(clippy::too_many_arguments)]
pub fn intra_pred(tx: u8, mode: usize, dst: &mut [u16], off: usize, stride: usize, left: &[i32], top: &[i32], bpp: u32) {
    let n = 4usize << tx;
    let max = (1i32 << bpp) - 1;
    let t = |i: isize| top[(i + 1) as usize];
    let mut put = |x: usize, y: usize, v: i32| dst[off + y * stride + x] = v as u16;
    match mode {
        VERT_PRED => {
            for y in 0..n {
                for x in 0..n {
                    put(x, y, t(x as isize));
                }
            }
        }
        HOR_PRED => {
            for y in 0..n {
                for x in 0..n {
                    put(x, y, left[n - 1 - y]);
                }
            }
        }
        TM_VP8_PRED => {
            let tl = t(-1);
            for y in 0..n {
                let l_m_tl = left[n - 1 - y] - tl;
                for x in 0..n {
                    put(x, y, (t(x as isize) + l_m_tl).clamp(0, max));
                }
            }
        }
        DC_PRED | LEFT_DC_PRED | TOP_DC_PRED | DC_128_PRED | DC_127_PRED | DC_129_PRED => {
            let log = 2 + tx as i32;
            let base = 128 << (bpp - 8);
            let dc = match mode {
                DC_PRED => {
                    let s: i32 = left[..n].iter().sum::<i32>() + (0..n).map(|i| t(i as isize)).sum::<i32>();
                    (s + (1 << log)) >> (log + 1)
                }
                LEFT_DC_PRED => (left[..n].iter().sum::<i32>() + (1 << (log - 1))) >> log,
                TOP_DC_PRED => ((0..n).map(|i| t(i as isize)).sum::<i32>() + (1 << (log - 1))) >> log,
                DC_128_PRED => base,
                DC_127_PRED => base - 1,
                _ => base + 1,
            };
            for y in 0..n {
                for x in 0..n {
                    put(x, y, dc);
                }
            }
        }
        DIAG_DOWN_LEFT_PRED => {
            if n == 4 {
                let a: Vec<i32> = (0..8).map(&t).collect();
                let mut d = [[0i32; 4]; 4];
                let f = |i: usize| (a[i] + a[i + 1] * 2 + a[i + 2] + 2) >> 2;
                for y in 0..4 {
                    for x in 0..4 {
                        d[y][x] = if x + y == 6 { a[7] } else { f(x + y) };
                    }
                }
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let mut v = vec![0i32; n - 1];
                for i in 0..n - 2 {
                    v[i] = (t(i as isize) + t(i as isize + 1) * 2 + t(i as isize + 2) + 2) >> 2;
                }
                v[n - 2] = (t(n as isize - 2) + t(n as isize - 1) * 3 + 2) >> 2;
                for j in 0..n {
                    for x in 0..n {
                        let val = if x < n - 1 - j { v[j + x] } else { t(n as isize - 1) };
                        put(x, j, val);
                    }
                }
            }
        }
        DIAG_DOWN_RIGHT_PRED => {
            if n == 4 {
                let (tl, a0, a1, a2, a3) = (t(-1), t(0), t(1), t(2), t(3));
                let (l0, l1, l2, l3) = (left[3], left[2], left[1], left[0]);
                let mut d = [[0i32; 4]; 4];
                d[3][0] = (l1 + l2 * 2 + l3 + 2) >> 2;
                let v = (l0 + l1 * 2 + l2 + 2) >> 2;
                d[2][0] = v;
                d[3][1] = v;
                let v = (tl + l0 * 2 + l1 + 2) >> 2;
                d[1][0] = v;
                d[2][1] = v;
                d[3][2] = v;
                let v = (l0 + tl * 2 + a0 + 2) >> 2;
                for i in 0..4 {
                    d[i][i] = v;
                }
                let v = (tl + a0 * 2 + a1 + 2) >> 2;
                d[0][1] = v;
                d[1][2] = v;
                d[2][3] = v;
                let v = (a0 + a1 * 2 + a2 + 2) >> 2;
                d[0][2] = v;
                d[1][3] = v;
                d[0][3] = (a1 + a2 * 2 + a3 + 2) >> 2;
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let mut v = vec![0i32; 2 * n - 1];
                for i in 0..n - 2 {
                    v[i] = (left[i] + left[i + 1] * 2 + left[i + 2] + 2) >> 2;
                    v[n + 1 + i] = (t(i as isize) + t(i as isize + 1) * 2 + t(i as isize + 2) + 2) >> 2;
                }
                v[n - 2] = (left[n - 2] + left[n - 1] * 2 + t(-1) + 2) >> 2;
                v[n - 1] = (left[n - 1] + t(-1) * 2 + t(0) + 2) >> 2;
                v[n] = (t(-1) + t(0) * 2 + t(1) + 2) >> 2;
                for j in 0..n {
                    for x in 0..n {
                        put(x, j, v[n - 1 - j + x]);
                    }
                }
            }
        }
        VERT_RIGHT_PRED => {
            if n == 4 {
                let (tl, a0, a1, a2, a3) = (t(-1), t(0), t(1), t(2), t(3));
                let (l0, l1, l2) = (left[3], left[2], left[1]);
                let mut d = [[0i32; 4]; 4];
                d[3][0] = (l0 + l1 * 2 + l2 + 2) >> 2;
                d[2][0] = (tl + l0 * 2 + l1 + 2) >> 2;
                let v = (tl + a0 + 1) >> 1;
                d[0][0] = v;
                d[2][1] = v;
                let v = (l0 + tl * 2 + a0 + 2) >> 2;
                d[1][0] = v;
                d[3][1] = v;
                let v = (a0 + a1 + 1) >> 1;
                d[0][1] = v;
                d[2][2] = v;
                let v = (tl + a0 * 2 + a1 + 2) >> 2;
                d[1][1] = v;
                d[3][2] = v;
                let v = (a1 + a2 + 1) >> 1;
                d[0][2] = v;
                d[2][3] = v;
                let v = (a0 + a1 * 2 + a2 + 2) >> 2;
                d[1][2] = v;
                d[3][3] = v;
                d[0][3] = (a2 + a3 + 1) >> 1;
                d[1][3] = (a1 + a2 * 2 + a3 + 2) >> 2;
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let m = n + n / 2 - 1;
                let mut ve = vec![0i32; m];
                let mut vo = vec![0i32; m];
                for i in 0..n / 2 - 2 {
                    vo[i] = (left[i * 2 + 3] + left[i * 2 + 2] * 2 + left[i * 2 + 1] + 2) >> 2;
                    ve[i] = (left[i * 2 + 4] + left[i * 2 + 3] * 2 + left[i * 2 + 2] + 2) >> 2;
                }
                vo[n / 2 - 2] = (left[n - 1] + left[n - 2] * 2 + left[n - 3] + 2) >> 2;
                ve[n / 2 - 2] = (t(-1) + left[n - 1] * 2 + left[n - 2] + 2) >> 2;
                ve[n / 2 - 1] = (t(-1) + t(0) + 1) >> 1;
                vo[n / 2 - 1] = (left[n - 1] + t(-1) * 2 + t(0) + 2) >> 2;
                for i in 0..n - 1 {
                    ve[n / 2 + i] = (t(i as isize) + t(i as isize + 1) + 1) >> 1;
                    vo[n / 2 + i] = (t(i as isize - 1) + t(i as isize) * 2 + t(i as isize + 1) + 2) >> 2;
                }
                for j in 0..n / 2 {
                    for x in 0..n {
                        put(x, j * 2, ve[n / 2 - 1 - j + x]);
                        put(x, j * 2 + 1, vo[n / 2 - 1 - j + x]);
                    }
                }
            }
        }
        HOR_DOWN_PRED => {
            if n == 4 {
                let (l0, l1, l2, l3) = (left[3], left[2], left[1], left[0]);
                let (tl, a0, a1, a2) = (t(-1), t(0), t(1), t(2));
                let mut d = [[0i32; 4]; 4];
                d[0][2] = (tl + a0 * 2 + a1 + 2) >> 2;
                d[0][3] = (a0 + a1 * 2 + a2 + 2) >> 2;
                let v = (tl + l0 + 1) >> 1;
                d[0][0] = v;
                d[1][2] = v;
                let v = (a0 + tl * 2 + l0 + 2) >> 2;
                d[0][1] = v;
                d[1][3] = v;
                let v = (l0 + l1 + 1) >> 1;
                d[1][0] = v;
                d[2][2] = v;
                let v = (tl + l0 * 2 + l1 + 2) >> 2;
                d[1][1] = v;
                d[2][3] = v;
                let v = (l1 + l2 + 1) >> 1;
                d[2][0] = v;
                d[3][2] = v;
                let v = (l0 + l1 * 2 + l2 + 2) >> 2;
                d[2][1] = v;
                d[3][3] = v;
                d[3][0] = (l2 + l3 + 1) >> 1;
                d[3][1] = (l1 + l2 * 2 + l3 + 2) >> 2;
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let mut v = vec![0i32; n * 3 - 2];
                for i in 0..n - 2 {
                    v[i * 2] = (left[i + 1] + left[i] + 1) >> 1;
                    v[i * 2 + 1] = (left[i + 2] + left[i + 1] * 2 + left[i] + 2) >> 2;
                    v[n * 2 + i] = (t(i as isize - 1) + t(i as isize) * 2 + t(i as isize + 1) + 2) >> 2;
                }
                v[n * 2 - 2] = (t(-1) + left[n - 1] + 1) >> 1;
                v[n * 2 - 4] = (left[n - 1] + left[n - 2] + 1) >> 1;
                v[n * 2 - 1] = (t(0) + t(-1) * 2 + left[n - 1] + 2) >> 2;
                v[n * 2 - 3] = (t(-1) + left[n - 1] * 2 + left[n - 2] + 2) >> 2;
                for j in 0..n {
                    for x in 0..n {
                        put(x, j, v[n * 2 - 2 - j * 2 + x]);
                    }
                }
            }
        }
        VERT_LEFT_PRED => {
            if n == 4 {
                let a: Vec<i32> = (0..7).map(&t).collect();
                let mut d = [[0i32; 4]; 4];
                let avg2 = |i: usize| (a[i] + a[i + 1] + 1) >> 1;
                let avg3 = |i: usize| (a[i] + a[i + 1] * 2 + a[i + 2] + 2) >> 2;
                for x in 0..4 {
                    d[0][x] = avg2(x);
                    d[1][x] = avg3(x);
                    d[2][x] = avg2(x + 1);
                    d[3][x] = avg3(x + 1);
                }
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let mut ve = vec![0i32; n - 1];
                let mut vo = vec![0i32; n - 1];
                for i in 0..n - 2 {
                    ve[i] = (t(i as isize) + t(i as isize + 1) + 1) >> 1;
                    vo[i] = (t(i as isize) + t(i as isize + 1) * 2 + t(i as isize + 2) + 2) >> 2;
                }
                ve[n - 2] = (t(n as isize - 2) + t(n as isize - 1) + 1) >> 1;
                vo[n - 2] = (t(n as isize - 2) + t(n as isize - 1) * 3 + 2) >> 2;
                let last = t(n as isize - 1);
                for j in 0..n / 2 {
                    for x in 0..n {
                        let (e, o) = if x < n - j - 1 { (ve[j + x], vo[j + x]) } else { (last, last) };
                        put(x, j * 2, e);
                        put(x, j * 2 + 1, o);
                    }
                }
            }
        }
        _ => {
            // HOR_UP_PRED: `left` top to bottom.
            if n == 4 {
                let (l0, l1, l2, l3) = (left[0], left[1], left[2], left[3]);
                let mut d = [[l3; 4]; 4];
                d[0][0] = (l0 + l1 + 1) >> 1;
                d[0][1] = (l0 + l1 * 2 + l2 + 2) >> 2;
                let v = (l1 + l2 + 1) >> 1;
                d[1][0] = v;
                d[0][2] = v;
                let v = (l1 + l2 * 2 + l3 + 2) >> 2;
                d[1][1] = v;
                d[0][3] = v;
                let v = (l2 + l3 + 1) >> 1;
                d[2][0] = v;
                d[1][2] = v;
                let v = (l2 + l3 * 3 + 2) >> 2;
                d[2][1] = v;
                d[1][3] = v;
                for y in 0..4 {
                    for x in 0..4 {
                        put(x, y, d[y][x]);
                    }
                }
            } else {
                let mut v = vec![0i32; n * 2 - 2];
                for i in 0..n - 2 {
                    v[i * 2] = (left[i] + left[i + 1] + 1) >> 1;
                    v[i * 2 + 1] = (left[i] + left[i + 1] * 2 + left[i + 2] + 2) >> 2;
                }
                v[n * 2 - 4] = (left[n - 2] + left[n - 1] + 1) >> 1;
                v[n * 2 - 3] = (left[n - 2] + left[n - 1] * 3 + 2) >> 2;
                for j in 0..n {
                    for x in 0..n {
                        let i = j * 2 + x;
                        put(x, j, if i < n * 2 - 2 { v[i] } else { left[n - 1] });
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse transforms
// ---------------------------------------------------------------------------

/// Storage of a `dctcoef`: `int16_t` at 8 bits, `int32_t` above.
#[inline(always)]
fn st(v: i64, eight: bool) -> i64 {
    if eight {
        v as i16 as i64
    } else {
        v as i32 as i64
    }
}

#[inline(always)]
fn r14(v: i64) -> i64 {
    (v + (1 << 13)) >> 14
}

fn idct4(i: &[i64], o: &mut [i64]) {
    let t0 = r14((i[0] + i[2]) * 11585);
    let t1 = r14((i[0] - i[2]) * 11585);
    let t2 = r14(i[1] * 6270 - i[3] * 15137);
    let t3 = r14(i[1] * 15137 + i[3] * 6270);
    o[0] = t0 + t3;
    o[1] = t1 + t2;
    o[2] = t1 - t2;
    o[3] = t0 - t3;
}

fn iadst4(i: &[i64], o: &mut [i64]) {
    let t0 = 5283 * i[0] + 15212 * i[2] + 9929 * i[3];
    let t1 = 9929 * i[0] - 5283 * i[2] - 15212 * i[3];
    let t2 = 13377 * (i[0] - i[2] + i[3]);
    let t3 = 13377 * i[1];
    o[0] = r14(t0 + t3);
    o[1] = r14(t1 + t3);
    o[2] = r14(t2);
    o[3] = r14(t0 + t1 - t3);
}

fn idct8(i: &[i64], o: &mut [i64]) {
    let t0a = r14((i[0] + i[4]) * 11585);
    let t1a = r14((i[0] - i[4]) * 11585);
    let t2a = r14(i[2] * 6270 - i[6] * 15137);
    let t3a = r14(i[2] * 15137 + i[6] * 6270);
    let t4a = r14(i[1] * 3196 - i[7] * 16069);
    let t5a = r14(i[5] * 13623 - i[3] * 9102);
    let t6a = r14(i[5] * 9102 + i[3] * 13623);
    let t7a = r14(i[1] * 16069 + i[7] * 3196);
    let t0 = t0a + t3a;
    let t1 = t1a + t2a;
    let t2 = t1a - t2a;
    let t3 = t0a - t3a;
    let t4 = t4a + t5a;
    let t5a = t4a - t5a;
    let t7 = t7a + t6a;
    let t6a = t7a - t6a;
    let t5 = r14((t6a - t5a) * 11585);
    let t6 = r14((t6a + t5a) * 11585);
    o[0] = t0 + t7;
    o[1] = t1 + t6;
    o[2] = t2 + t5;
    o[3] = t3 + t4;
    o[4] = t3 - t4;
    o[5] = t2 - t5;
    o[6] = t1 - t6;
    o[7] = t0 - t7;
}

fn iadst8(i: &[i64], o: &mut [i64]) {
    let t0a = 16305 * i[7] + 1606 * i[0];
    let t1a = 1606 * i[7] - 16305 * i[0];
    let t2a = 14449 * i[5] + 7723 * i[2];
    let t3a = 7723 * i[5] - 14449 * i[2];
    let t4a = 10394 * i[3] + 12665 * i[4];
    let t5a = 12665 * i[3] - 10394 * i[4];
    let t6a = 4756 * i[1] + 15679 * i[6];
    let t7a = 15679 * i[1] - 4756 * i[6];
    let t0 = r14(t0a + t4a);
    let t1 = r14(t1a + t5a);
    let t2 = r14(t2a + t6a);
    let t3 = r14(t3a + t7a);
    let t4 = r14(t0a - t4a);
    let t5 = r14(t1a - t5a);
    let t6 = r14(t2a - t6a);
    let t7 = r14(t3a - t7a);
    let t4a = 15137 * t4 + 6270 * t5;
    let t5a = 6270 * t4 - 15137 * t5;
    let t6a = 15137 * t7 - 6270 * t6;
    let t7a = 6270 * t7 + 15137 * t6;
    o[0] = t0 + t2;
    o[7] = -(t1 + t3);
    let t2 = t0 - t2;
    let t3 = t1 - t3;
    o[1] = -r14(t4a + t6a);
    o[6] = r14(t5a + t7a);
    let t6 = r14(t4a - t6a);
    let t7 = r14(t5a - t7a);
    o[3] = -r14((t2 + t3) * 11585);
    o[4] = r14((t2 - t3) * 11585);
    o[2] = r14((t6 + t7) * 11585);
    o[5] = -r14((t6 - t7) * 11585);
}

fn idct16(i: &[i64], o: &mut [i64]) {
    let t0a = r14((i[0] + i[8]) * 11585);
    let t1a = r14((i[0] - i[8]) * 11585);
    let t2a = r14(i[4] * 6270 - i[12] * 15137);
    let t3a = r14(i[4] * 15137 + i[12] * 6270);
    let t4a = r14(i[2] * 3196 - i[14] * 16069);
    let t7a = r14(i[2] * 16069 + i[14] * 3196);
    let t5a = r14(i[10] * 13623 - i[6] * 9102);
    let t6a = r14(i[10] * 9102 + i[6] * 13623);
    let t8a = r14(i[1] * 1606 - i[15] * 16305);
    let t15a = r14(i[1] * 16305 + i[15] * 1606);
    let t9a = r14(i[9] * 12665 - i[7] * 10394);
    let t14a = r14(i[9] * 10394 + i[7] * 12665);
    let t10a = r14(i[5] * 7723 - i[11] * 14449);
    let t13a = r14(i[5] * 14449 + i[11] * 7723);
    let t11a = r14(i[13] * 15679 - i[3] * 4756);
    let t12a = r14(i[13] * 4756 + i[3] * 15679);
    let t0 = t0a + t3a;
    let t1 = t1a + t2a;
    let t2 = t1a - t2a;
    let t3 = t0a - t3a;
    let t4 = t4a + t5a;
    let t5 = t4a - t5a;
    let t6 = t7a - t6a;
    let t7 = t7a + t6a;
    let t8 = t8a + t9a;
    let t9 = t8a - t9a;
    let t10 = t11a - t10a;
    let t11 = t11a + t10a;
    let t12 = t12a + t13a;
    let t13 = t12a - t13a;
    let t14 = t15a - t14a;
    let t15 = t15a + t14a;
    let t5a = r14((t6 - t5) * 11585);
    let t6a = r14((t6 + t5) * 11585);
    let t9a = r14(t14 * 6270 - t9 * 15137);
    let t14a = r14(t14 * 15137 + t9 * 6270);
    let t10a = r14(-(t13 * 15137 + t10 * 6270));
    let t13a = r14(t13 * 6270 - t10 * 15137);
    let t0a = t0 + t7;
    let t1a = t1 + t6a;
    let t2a = t2 + t5a;
    let t3a = t3 + t4;
    let t4 = t3 - t4;
    let t5 = t2 - t5a;
    let t6 = t1 - t6a;
    let t7 = t0 - t7;
    let t8a = t8 + t11;
    let t9 = t9a + t10a;
    let t10 = t9a - t10a;
    let t11a = t8 - t11;
    let t12a = t15 - t12;
    let t13 = t14a - t13a;
    let t14 = t14a + t13a;
    let t15a = t15 + t12;
    let t10a = r14((t13 - t10) * 11585);
    let t13a = r14((t13 + t10) * 11585);
    let t11 = r14((t12a - t11a) * 11585);
    let t12 = r14((t12a + t11a) * 11585);
    o[0] = t0a + t15a;
    o[1] = t1a + t14;
    o[2] = t2a + t13a;
    o[3] = t3a + t12;
    o[4] = t4 + t11;
    o[5] = t5 + t10a;
    o[6] = t6 + t9;
    o[7] = t7 + t8a;
    o[8] = t7 - t8a;
    o[9] = t6 - t9;
    o[10] = t5 - t10a;
    o[11] = t4 - t11;
    o[12] = t3a - t12;
    o[13] = t2a - t13a;
    o[14] = t1a - t14;
    o[15] = t0a - t15a;
}

fn iadst16(i: &[i64], o: &mut [i64]) {
    let t0 = i[15] * 16364 + i[0] * 804;
    let t1 = i[15] * 804 - i[0] * 16364;
    let t2 = i[13] * 15893 + i[2] * 3981;
    let t3 = i[13] * 3981 - i[2] * 15893;
    let t4 = i[11] * 14811 + i[4] * 7005;
    let t5 = i[11] * 7005 - i[4] * 14811;
    let t6 = i[9] * 13160 + i[6] * 9760;
    let t7 = i[9] * 9760 - i[6] * 13160;
    let t8 = i[7] * 11003 + i[8] * 12140;
    let t9 = i[7] * 12140 - i[8] * 11003;
    let t10 = i[5] * 8423 + i[10] * 14053;
    let t11 = i[5] * 14053 - i[10] * 8423;
    let t12 = i[3] * 5520 + i[12] * 15426;
    let t13 = i[3] * 15426 - i[12] * 5520;
    let t14 = i[1] * 2404 + i[14] * 16207;
    let t15 = i[1] * 16207 - i[14] * 2404;
    let t0a = r14(t0 + t8);
    let t1a = r14(t1 + t9);
    let t2a = r14(t2 + t10);
    let t3a = r14(t3 + t11);
    let t4a = r14(t4 + t12);
    let t5a = r14(t5 + t13);
    let t6a = r14(t6 + t14);
    let t7a = r14(t7 + t15);
    let t8a = r14(t0 - t8);
    let t9a = r14(t1 - t9);
    let t10a = r14(t2 - t10);
    let t11a = r14(t3 - t11);
    let t12a = r14(t4 - t12);
    let t13a = r14(t5 - t13);
    let t14a = r14(t6 - t14);
    let t15a = r14(t7 - t15);
    let t8 = t8a * 16069 + t9a * 3196;
    let t9 = t8a * 3196 - t9a * 16069;
    let t10 = t10a * 9102 + t11a * 13623;
    let t11 = t10a * 13623 - t11a * 9102;
    let t12 = t13a * 16069 - t12a * 3196;
    let t13 = t13a * 3196 + t12a * 16069;
    let t14 = t15a * 9102 - t14a * 13623;
    let t15 = t15a * 13623 + t14a * 9102;
    let t0 = t0a + t4a;
    let t1 = t1a + t5a;
    let t2 = t2a + t6a;
    let t3 = t3a + t7a;
    let t4 = t0a - t4a;
    let t5 = t1a - t5a;
    let t6 = t2a - t6a;
    let t7 = t3a - t7a;
    let t8a = r14(t8 + t12);
    let t9a = r14(t9 + t13);
    let t10a = r14(t10 + t14);
    let t11a = r14(t11 + t15);
    let t12a = r14(t8 - t12);
    let t13a = r14(t9 - t13);
    let t14a = r14(t10 - t14);
    let t15a = r14(t11 - t15);
    let t4a = t4 * 15137 + t5 * 6270;
    let t5a = t4 * 6270 - t5 * 15137;
    let t6a = t7 * 15137 - t6 * 6270;
    let t7a = t7 * 6270 + t6 * 15137;
    let t12 = t12a * 15137 + t13a * 6270;
    let t13 = t12a * 6270 - t13a * 15137;
    let t14 = t15a * 15137 - t14a * 6270;
    let t15 = t15a * 6270 + t14a * 15137;
    o[0] = t0 + t2;
    o[15] = -(t1 + t3);
    let t2a = t0 - t2;
    let t3a = t1 - t3;
    o[3] = -r14(t4a + t6a);
    o[12] = r14(t5a + t7a);
    let t6 = r14(t4a - t6a);
    let t7 = r14(t5a - t7a);
    o[1] = -(t8a + t10a);
    o[14] = t9a + t11a;
    let t10 = t8a - t10a;
    let t11 = t9a - t11a;
    o[2] = r14(t12 + t14);
    o[13] = -r14(t13 + t15);
    let t14a = r14(t12 - t14);
    let t15a = r14(t13 - t15);
    o[7] = r14(-(t2a + t3a) * 11585);
    o[8] = r14((t2a - t3a) * 11585);
    o[4] = r14((t7 + t6) * 11585);
    o[11] = r14((t7 - t6) * 11585);
    o[6] = r14((t11 + t10) * 11585);
    o[9] = r14((t11 - t10) * 11585);
    o[5] = r14(-(t14a + t15a) * 11585);
    o[10] = r14((t14a - t15a) * 11585);
}

fn idct32(i: &[i64], o: &mut [i64]) {
    let t0a = r14((i[0] + i[16]) * 11585);
    let t1a = r14((i[0] - i[16]) * 11585);
    let t2a = r14(i[8] * 6270 - i[24] * 15137);
    let t3a = r14(i[8] * 15137 + i[24] * 6270);
    let t4a = r14(i[4] * 3196 - i[28] * 16069);
    let t7a = r14(i[4] * 16069 + i[28] * 3196);
    let t5a = r14(i[20] * 13623 - i[12] * 9102);
    let t6a = r14(i[20] * 9102 + i[12] * 13623);
    let t8a = r14(i[2] * 1606 - i[30] * 16305);
    let t15a = r14(i[2] * 16305 + i[30] * 1606);
    let t9a = r14(i[18] * 12665 - i[14] * 10394);
    let t14a = r14(i[18] * 10394 + i[14] * 12665);
    let t10a = r14(i[10] * 7723 - i[22] * 14449);
    let t13a = r14(i[10] * 14449 + i[22] * 7723);
    let t11a = r14(i[26] * 15679 - i[6] * 4756);
    let t12a = r14(i[26] * 4756 + i[6] * 15679);
    let t16a = r14(i[1] * 804 - i[31] * 16364);
    let t31a = r14(i[1] * 16364 + i[31] * 804);
    let t17a = r14(i[17] * 12140 - i[15] * 11003);
    let t30a = r14(i[17] * 11003 + i[15] * 12140);
    let t18a = r14(i[9] * 7005 - i[23] * 14811);
    let t29a = r14(i[9] * 14811 + i[23] * 7005);
    let t19a = r14(i[25] * 15426 - i[7] * 5520);
    let t28a = r14(i[25] * 5520 + i[7] * 15426);
    let t20a = r14(i[5] * 3981 - i[27] * 15893);
    let t27a = r14(i[5] * 15893 + i[27] * 3981);
    let t21a = r14(i[21] * 14053 - i[11] * 8423);
    let t26a = r14(i[21] * 8423 + i[11] * 14053);
    let t22a = r14(i[13] * 9760 - i[19] * 13160);
    let t25a = r14(i[13] * 13160 + i[19] * 9760);
    let t23a = r14(i[29] * 16207 - i[3] * 2404);
    let t24a = r14(i[29] * 2404 + i[3] * 16207);

    let t0 = t0a + t3a;
    let t1 = t1a + t2a;
    let t2 = t1a - t2a;
    let t3 = t0a - t3a;
    let t4 = t4a + t5a;
    let t5 = t4a - t5a;
    let t6 = t7a - t6a;
    let t7 = t7a + t6a;
    let t8 = t8a + t9a;
    let t9 = t8a - t9a;
    let t10 = t11a - t10a;
    let t11 = t11a + t10a;
    let t12 = t12a + t13a;
    let t13 = t12a - t13a;
    let t14 = t15a - t14a;
    let t15 = t15a + t14a;
    let t16 = t16a + t17a;
    let t17 = t16a - t17a;
    let t18 = t19a - t18a;
    let t19 = t19a + t18a;
    let t20 = t20a + t21a;
    let t21 = t20a - t21a;
    let t22 = t23a - t22a;
    let t23 = t23a + t22a;
    let t24 = t24a + t25a;
    let t25 = t24a - t25a;
    let t26 = t27a - t26a;
    let t27 = t27a + t26a;
    let t28 = t28a + t29a;
    let t29 = t28a - t29a;
    let t30 = t31a - t30a;
    let t31 = t31a + t30a;

    let t5a = r14((t6 - t5) * 11585);
    let t6a = r14((t6 + t5) * 11585);
    let t9a = r14(t14 * 6270 - t9 * 15137);
    let t14a = r14(t14 * 15137 + t9 * 6270);
    let t10a = r14(-(t13 * 15137 + t10 * 6270));
    let t13a = r14(t13 * 6270 - t10 * 15137);
    let t17a = r14(t30 * 3196 - t17 * 16069);
    let t30a = r14(t30 * 16069 + t17 * 3196);
    let t18a = r14(-(t29 * 16069 + t18 * 3196));
    let t29a = r14(t29 * 3196 - t18 * 16069);
    let t21a = r14(t26 * 13623 - t21 * 9102);
    let t26a = r14(t26 * 9102 + t21 * 13623);
    let t22a = r14(-(t25 * 9102 + t22 * 13623));
    let t25a = r14(t25 * 13623 - t22 * 9102);

    let t0a = t0 + t7;
    let t1a = t1 + t6a;
    let t2a = t2 + t5a;
    let t3a = t3 + t4;
    let t4a = t3 - t4;
    let t5 = t2 - t5a;
    let t6 = t1 - t6a;
    let t7a = t0 - t7;
    let t8a = t8 + t11;
    let t9 = t9a + t10a;
    let t10 = t9a - t10a;
    let t11a = t8 - t11;
    let t12a = t15 - t12;
    let t13 = t14a - t13a;
    let t14 = t14a + t13a;
    let t15a = t15 + t12;
    let t16a = t16 + t19;
    let t17 = t17a + t18a;
    let t18 = t17a - t18a;
    let t19a = t16 - t19;
    let t20a = t23 - t20;
    let t21 = t22a - t21a;
    let t22 = t22a + t21a;
    let t23a = t23 + t20;
    let t24a = t24 + t27;
    let t25 = t25a + t26a;
    let t26 = t25a - t26a;
    let t27a = t24 - t27;
    let t28a = t31 - t28;
    let t29 = t30a - t29a;
    let t30 = t30a + t29a;
    let t31a = t31 + t28;

    let t10a = r14((t13 - t10) * 11585);
    let t13a = r14((t13 + t10) * 11585);
    let t11 = r14((t12a - t11a) * 11585);
    let t12 = r14((t12a + t11a) * 11585);
    let t18a = r14(t29 * 6270 - t18 * 15137);
    let t29a = r14(t29 * 15137 + t18 * 6270);
    let t19 = r14(t28a * 6270 - t19a * 15137);
    let t28 = r14(t28a * 15137 + t19a * 6270);
    let t20 = r14(-(t27a * 15137 + t20a * 6270));
    let t27 = r14(t27a * 6270 - t20a * 15137);
    let t21a = r14(-(t26 * 15137 + t21 * 6270));
    let t26a = r14(t26 * 6270 - t21 * 15137);

    let t0 = t0a + t15a;
    let t1 = t1a + t14;
    let t2 = t2a + t13a;
    let t3 = t3a + t12;
    let t4 = t4a + t11;
    let t5a = t5 + t10a;
    let t6a = t6 + t9;
    let t7 = t7a + t8a;
    let t8 = t7a - t8a;
    let t9a = t6 - t9;
    let t10 = t5 - t10a;
    let t11a = t4a - t11;
    let t12a = t3a - t12;
    let t13 = t2a - t13a;
    let t14a = t1a - t14;
    let t15 = t0a - t15a;
    let t16 = t16a + t23a;
    let t17a = t17 + t22;
    let t18 = t18a + t21a;
    let t19a = t19 + t20;
    let t20a = t19 - t20;
    let t21 = t18a - t21a;
    let t22a = t17 - t22;
    let t23 = t16a - t23a;
    let t24 = t31a - t24a;
    let t25a = t30 - t25;
    let t26 = t29a - t26a;
    let t27a = t28 - t27;
    let t28a = t28 + t27;
    let t29 = t29a + t26a;
    let t30a = t30 + t25;
    let t31 = t31a + t24a;

    let t20 = r14((t27a - t20a) * 11585);
    let t27 = r14((t27a + t20a) * 11585);
    let t21a = r14((t26 - t21) * 11585);
    let t26a = r14((t26 + t21) * 11585);
    let t22 = r14((t25a - t22a) * 11585);
    let t25 = r14((t25a + t22a) * 11585);
    let t23a = r14((t24 - t23) * 11585);
    let t24a = r14((t24 + t23) * 11585);

    o[0] = t0 + t31;
    o[1] = t1 + t30a;
    o[2] = t2 + t29;
    o[3] = t3 + t28a;
    o[4] = t4 + t27;
    o[5] = t5a + t26a;
    o[6] = t6a + t25;
    o[7] = t7 + t24a;
    o[8] = t8 + t23a;
    o[9] = t9a + t22;
    o[10] = t10 + t21a;
    o[11] = t11a + t20;
    o[12] = t12a + t19a;
    o[13] = t13 + t18;
    o[14] = t14a + t17a;
    o[15] = t15 + t16;
    o[16] = t15 - t16;
    o[17] = t14a - t17a;
    o[18] = t13 - t18;
    o[19] = t12a - t19a;
    o[20] = t11a - t20;
    o[21] = t10 - t21a;
    o[22] = t9a - t22;
    o[23] = t8 - t23a;
    o[24] = t7 - t24a;
    o[25] = t6a - t25;
    o[26] = t5a - t26a;
    o[27] = t4 - t27;
    o[28] = t3 - t28a;
    o[29] = t2 - t29;
    o[30] = t1 - t30a;
    o[31] = t0 - t31;
}

fn iwht4(i: &[i64], o: &mut [i64], pass: u32) {
    let (mut t0, mut t1, mut t2, mut t3) = if pass == 0 { (i[0] >> 2, i[3] >> 2, i[1] >> 2, i[2] >> 2) } else { (i[0], i[3], i[1], i[2]) };
    t0 += t2;
    t3 -= t1;
    let t4 = (t0 - t3) >> 1;
    t1 = t4 - t1;
    t2 = t4 - t2;
    t0 -= t1;
    t3 += t2;
    o[0] = t0;
    o[1] = t1;
    o[2] = t2;
    o[3] = t3;
}

/// `itxfm_add[tx][txtp]`: inverse transform `block` (raster, zeroed
/// afterwards) and add it to the `n × n` samples at `off`. `tx` 4 is the
/// lossless Walsh-Hadamard transform.
#[allow(clippy::too_many_arguments)]
pub fn itxfm_add(tx: usize, txtp: usize, dst: &mut [u16], off: usize, stride: usize, block: &mut [i32], eob: usize, bpp: u32) {
    let eight = bpp == 8;
    let max = (1i32 << bpp) - 1;
    let (sz, bits) = match tx {
        0 => (4usize, 4u32),
        1 => (8, 5),
        2 => (16, 6),
        3 => (32, 6),
        _ => (4, 0),
    };
    let add = |dst: &mut [u16], at: usize, v: i64| {
        let r = if bits != 0 { ((v + (1i64 << (bits - 1))) as i32) >> bits } else { v as i32 };
        dst[at] = clip(dst[at] as i32 + r, max);
    };
    let has_dconly = tx < 4 && txtp == DCT_DCT;
    if has_dconly && eob == 1 {
        let t = r14(r14(block[0] as i64 * 11585) * 11585);
        block[0] = 0;
        for i in 0..sz {
            for j in 0..sz {
                add(dst, off + j * stride + i, t);
            }
        }
        return;
    }
    // type_a / type_b: DCT_DCT = idct idct, DCT_ADST (1) = iadst then idct,
    // ADST_DCT (2) = idct then iadst, ADST_ADST = iadst iadst.
    let (adst_a, adst_b) = match txtp {
        1 => (true, false),
        2 => (false, true),
        3 => (true, true),
        _ => (false, false),
    };
    let tr = |inp: &[i64], out: &mut [i64], adst: bool, pass: u32| match (tx, adst) {
        (0, false) => idct4(inp, out),
        (0, true) => iadst4(inp, out),
        (1, false) => idct8(inp, out),
        (1, true) => iadst8(inp, out),
        (2, false) => idct16(inp, out),
        (2, true) => iadst16(inp, out),
        (3, _) => idct32(inp, out),
        _ => iwht4(inp, out, pass),
    };
    let mut tmp = vec![0i64; sz * sz];
    let mut col = vec![0i64; sz];
    let mut out = vec![0i64; sz];
    for i in 0..sz {
        for k in 0..sz {
            col[k] = block[i + k * sz] as i64;
        }
        tr(&col, &mut out, adst_a, 0);
        for k in 0..sz {
            tmp[i * sz + k] = st(out[k], eight);
        }
    }
    block[..sz * sz].fill(0);
    for i in 0..sz {
        for k in 0..sz {
            col[k] = tmp[i + k * sz];
        }
        tr(&col, &mut out, adst_b, 1);
        for j in 0..sz {
            add(dst, off + j * stride + i, st(out[j], eight));
        }
    }
}

// ---------------------------------------------------------------------------
// Loop filter
// ---------------------------------------------------------------------------

/// ffmpeg `loop_filter` over 8 lines: `stridea` steps along the edge,
/// `strideb` across it; `wd` 4, 8 or 16.
#[allow(clippy::too_many_arguments)]
pub fn loop_filter(dst: &mut [u16], off: usize, stridea: isize, strideb: isize, e: i32, i: i32, h: i32, wd: u32, bpp: u32) {
    let sh = bpp - 8;
    let f = 1i32 << sh;
    let (e, ii, hh) = (e << sh, i << sh, h << sh);
    let max = (1i32 << bpp) - 1;
    let lim = (1i32 << (bpp - 1)) - 1;
    let clip2 = |v: i32| v.clamp(-(1 << (bpp - 1)), (1 << (bpp - 1)) - 1);
    let mut base = off as isize;
    // Taps reach 4 samples either side of the edge, 8 for the 16-wide filter.
    let reach: isize = if wd >= 16 { 8 } else { 4 };
    assert!(strideb > 0);
    for _ in 0..8 {
        // One range check per line covers every tap below.
        assert!(base - reach * strideb >= 0 && ((base + (reach - 1) * strideb) as usize) < dst.len());
        let at = |k: isize| (base + strideb * k) as usize;
        // SAFETY: |k| <= reach on every call, inside the range checked above.
        let g = |d: &[u16], k: isize| unsafe { *d.get_unchecked(at(k)) } as i32;
        let (p3, p2, p1, p0) = (g(dst, -4), g(dst, -3), g(dst, -2), g(dst, -1));
        let (q0, q1, q2, q3) = (g(dst, 0), g(dst, 1), g(dst, 2), g(dst, 3));
        let fm = (p3 - p2).abs() <= ii
            && (p2 - p1).abs() <= ii
            && (p1 - p0).abs() <= ii
            && (q1 - q0).abs() <= ii
            && (q2 - q1).abs() <= ii
            && (q3 - q2).abs() <= ii
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;
        if fm {
            let mut flat8out = false;
            let (mut p7, mut p6, mut p5, mut p4, mut q4, mut q5, mut q6, mut q7) = (0, 0, 0, 0, 0, 0, 0, 0);
            if wd >= 16 {
                p7 = g(dst, -8);
                p6 = g(dst, -7);
                p5 = g(dst, -6);
                p4 = g(dst, -5);
                q4 = g(dst, 4);
                q5 = g(dst, 5);
                q6 = g(dst, 6);
                q7 = g(dst, 7);
                flat8out = (p7 - p0).abs() <= f
                    && (p6 - p0).abs() <= f
                    && (p5 - p0).abs() <= f
                    && (p4 - p0).abs() <= f
                    && (q4 - q0).abs() <= f
                    && (q5 - q0).abs() <= f
                    && (q6 - q0).abs() <= f
                    && (q7 - q0).abs() <= f;
            }
            let flat8in = wd >= 8
                && (p3 - p0).abs() <= f
                && (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f
                && (q3 - q0).abs() <= f;
            // SAFETY: as for `g`.
            let mut set = |k: isize, v: i32| unsafe { *dst.get_unchecked_mut(at(k)) = v as u16 };
            if wd >= 16 && flat8out && flat8in {
                set(-7, (p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
                set(-6, (p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
                set(-5, (p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
                set(-4, (p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
                set(-3, (p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
                set(-2, (p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
                set(-1, (p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
                set(0, (p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7 + 8) >> 4);
                set(1, (p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2 + 8) >> 4);
                set(2, (p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3 + 8) >> 4);
                set(3, (p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4 + 8) >> 4);
                set(4, (p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5 + 8) >> 4);
                set(5, (p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6 + 8) >> 4);
                set(6, (p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7 + 8) >> 4);
            } else if wd >= 8 && flat8in {
                set(-3, (p3 * 3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
                set(-2, (p3 * 2 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
                set(-1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
                set(0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
                set(1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 * 2 + 4) >> 3);
                set(2, (p0 + q0 + q1 + 2 * q2 + q3 * 3 + 4) >> 3);
            } else {
                let hev = (p1 - p0).abs() > hh || (q1 - q0).abs() > hh;
                if hev {
                    let mut fv = clip2(p1 - q1);
                    fv = clip2(3 * (q0 - p0) + fv);
                    let f1 = (fv + 4).min(lim) >> 3;
                    let f2 = (fv + 3).min(lim) >> 3;
                    set(-1, (p0 + f2).clamp(0, max));
                    set(0, (q0 - f1).clamp(0, max));
                } else {
                    let fv = clip2(3 * (q0 - p0));
                    let f1 = (fv + 4).min(lim) >> 3;
                    let f2 = (fv + 3).min(lim) >> 3;
                    set(-1, (p0 + f2).clamp(0, max));
                    set(0, (q0 - f1).clamp(0, max));
                    let fv = (f1 + 1) >> 1;
                    set(-2, (p1 + fv).clamp(0, max));
                    set(1, (q1 - fv).clamp(0, max));
                }
            }
        }
        base += stridea;
    }
}

// ---------------------------------------------------------------------------
// Motion compensation
// ---------------------------------------------------------------------------

/// A reference window: samples of `plane` from `(x0, y0)`, `w × h`, with
/// coordinates clamped to the visible `vis_w × vis_h` (ffmpeg's emulated
/// edge, used exactly when a read would leave the visible picture).
#[allow(clippy::too_many_arguments)]
pub fn fetch(plane: &Plane, x0: isize, y0: isize, w: usize, h: usize, vis_w: usize, vis_h: usize, out: &mut Vec<i32>) {
    out.clear();
    out.reserve(w * h);
    // Columns before the visible area repeat its first sample, after it its
    // last; the run between is one straight copy per row.
    let vw = vis_w as isize;
    let left = (-x0).clamp(0, w as isize) as usize;
    let right = (x0 + w as isize - vw).clamp(0, (w - left) as isize) as usize;
    let mid = w - left - right;
    let mx = x0.clamp(0, vw - 1) as usize;
    for y in 0..h {
        let yy = (y0 + y as isize).clamp(0, vis_h as isize - 1) as usize;
        let row = &plane.data[yy * plane.width..];
        out.extend(std::iter::repeat_n(row[0] as i32, left));
        out.extend(row[mx..mx + mid].iter().map(|&v| v as i32));
        out.extend(std::iter::repeat_n(row[vis_w - 1] as i32, right));
    }
}

#[inline(always)]
fn tap8(s: &[i32], at: usize, stride: usize, f: &[i16; 8], max: i32) -> i32 {
    let mut v = 64i32;
    for k in 0..8 {
        v += f[k] as i32 * s[at + k * stride - 3 * stride];
    }
    (v >> 7).clamp(0, max)
}

#[inline(always)]
fn bilin(s: &[i32], at: usize, stride: usize, m: i32) -> i32 {
    s[at] + ((m * (s[at + stride] - s[at]) + 8) >> 4)
}

/// Unscaled prediction of a `bw × bh` block from a window `win` (stride
/// `ws`, the block's top-left at `(3, 3)`), filter `filter` (`FilterMode`
/// 0-3), subpel `mx`, `my` in 1/16; `avg` averages with what `dst` holds.
#[allow(clippy::too_many_arguments)]
pub fn mc_put(dst: &mut [u16], doff: usize, dstride: usize, win: &[i32], ws: usize, bw: usize, bh: usize, filter: u8, mx: i32, my: i32, avg: bool, bpp: u32) {
    let max = (1i32 << bpp) - 1;
    let store = |dst: &mut [u16], at: usize, v: i32| {
        dst[at] = if avg { ((dst[at] as i32 + v + 1) >> 1) as u16 } else { v as u16 };
    };
    let org = 3 * ws + 3;
    if mx == 0 && my == 0 {
        for y in 0..bh {
            for x in 0..bw {
                store(dst, doff + y * dstride + x, win[org + y * ws + x]);
            }
        }
        return;
    }
    if filter == FilterMode::Bilinear as u8 {
        if my == 0 {
            for y in 0..bh {
                for x in 0..bw {
                    store(dst, doff + y * dstride + x, bilin(win, org + y * ws + x, 1, mx));
                }
            }
        } else if mx == 0 {
            for y in 0..bh {
                for x in 0..bw {
                    store(dst, doff + y * dstride + x, bilin(win, org + y * ws + x, ws, my));
                }
            }
        } else {
            let mut tmp = vec![0i32; 64 * (bh + 1)];
            for y in 0..bh + 1 {
                for x in 0..bw {
                    tmp[y * 64 + x] = bilin(win, org + y * ws + x, 1, mx);
                }
            }
            for y in 0..bh {
                for x in 0..bw {
                    store(dst, doff + y * dstride + x, bilin(&tmp, y * 64 + x, 64, my));
                }
            }
        }
        return;
    }
    let ft = &SUBPEL_FILTERS[filter as usize];
    if my == 0 {
        let f = &ft[mx as usize];
        for y in 0..bh {
            for x in 0..bw {
                let at = org + y * ws + x;
                let mut v = 64i32;
                for k in 0..8 {
                    v += f[k] as i32 * win[at + k - 3];
                }
                store(dst, doff + y * dstride + x, (v >> 7).clamp(0, max));
            }
        }
    } else if mx == 0 {
        let f = &ft[my as usize];
        for y in 0..bh {
            for x in 0..bw {
                store(dst, doff + y * dstride + x, tap8(win, org + y * ws + x, ws, f, max));
            }
        }
    } else {
        let (fx, fy) = (&ft[mx as usize], &ft[my as usize]);
        let mut tmp = vec![0i32; 64 * (bh + 7)];
        for y in 0..bh + 7 {
            for x in 0..bw {
                let at = y * ws + 3 + x;
                let mut v = 64i32;
                for k in 0..8 {
                    v += fx[k] as i32 * win[at + k - 3];
                }
                tmp[y * 64 + x] = (v >> 7).clamp(0, max);
            }
        }
        for y in 0..bh {
            for x in 0..bw {
                store(dst, doff + y * dstride + x, tap8(&tmp, (y + 3) * 64 + x, 64, fy, max));
            }
        }
    }
}

/// Scaled prediction (ffmpeg `do_scaled_8tap_c` / `do_scaled_bilin_c`):
/// `win` holds the reference from three samples above-left of the first
/// integer position; `dx`, `dy` are the 1/16 steps.
#[allow(clippy::too_many_arguments)]
pub fn mc_scaled(dst: &mut [u16], doff: usize, dstride: usize, win: &[i32], ws: usize, bw: usize, bh: usize, filter: u8, mx: i32, my: i32, dx: i32, dy: i32, avg: bool, bpp: u32) {
    let max = (1i32 << bpp) - 1;
    let store = |dst: &mut [u16], at: usize, v: i32| {
        dst[at] = if avg { ((dst[at] as i32 + v + 1) >> 1) as u16 } else { v as u16 };
    };
    if filter == FilterMode::Bilinear as u8 {
        let tmp_h = ((((bh as i32 - 1) * dy + my) >> 4) + 2) as usize;
        let mut tmp = vec![0i32; 64 * tmp_h];
        for y in 0..tmp_h {
            let (mut imx, mut ioff) = (mx, 0usize);
            for x in 0..bw {
                tmp[y * 64 + x] = bilin(win, (y + 3) * ws + 3 + ioff, 1, imx);
                imx += dx;
                ioff += (imx >> 4) as usize;
                imx &= 0xf;
            }
        }
        let mut my = my;
        let mut row = 0usize;
        for y in 0..bh {
            for x in 0..bw {
                store(dst, doff + y * dstride + x, bilin(&tmp, row * 64 + x, 64, my));
            }
            my += dy;
            row += (my >> 4) as usize;
            my &= 0xf;
        }
        return;
    }
    let ft = &SUBPEL_FILTERS[filter as usize];
    let tmp_h = ((((bh as i32 - 1) * dy + my) >> 4) + 8) as usize;
    let mut tmp = vec![0i32; 64 * tmp_h];
    for y in 0..tmp_h {
        let (mut imx, mut ioff) = (mx, 0usize);
        for x in 0..bw {
            let f = &ft[imx as usize];
            let at = y * ws + 3 + ioff;
            let mut v = 64i32;
            for k in 0..8 {
                v += f[k] as i32 * win[at + k - 3];
            }
            tmp[y * 64 + x] = (v >> 7).clamp(0, max);
            imx += dx;
            ioff += (imx >> 4) as usize;
            imx &= 0xf;
        }
    }
    let mut my = my;
    let mut row = 3usize;
    for y in 0..bh {
        let f = &ft[my as usize];
        for x in 0..bw {
            store(dst, doff + y * dstride + x, tap8(&tmp, row * 64 + x, 64, f, max));
        }
        my += dy;
        row += (my >> 4) as usize;
        my &= 0xf;
    }
}
