//! HEVC scaling and inverse transforms (§8.6.2-8.6.4).

use super::tables::{DCT32, DST4};

const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Scale `TransCoeffLevel` in place (§8.6.3). `m` is the scaling factor
/// array for this block (raster), or `None` for the flat 16.
pub fn dequant(c: &mut [i32], n: usize, qp: i32, bd: u32, m: Option<&[u8]>) {
    let log2 = n.trailing_zeros() as i32;
    let bd_shift = bd as i32 + log2 - 5;
    let scale = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
    let add = 1i64 << (bd_shift - 1);
    for (i, v) in c.iter_mut().enumerate().take(n * n) {
        if *v != 0 {
            let f = m.map_or(16, |m| m[i] as i64);
            let d = ((*v as i64 * f * scale as i64) + add) >> bd_shift;
            *v = d.clamp(-32768, 32767) as i32;
        }
    }
}

/// One inverse 1-D transform of length `n` over `src` (stride `ss`) into
/// `dst` (stride `ds`). `last` is one past the last non-zero input.
#[inline]
fn idct_1d(src: &[i32], ss: usize, dst: &mut [i32], ds: usize, n: usize, last: usize) {
    let step = 32 / n;
    for i in 0..n {
        let mut acc = 0i64;
        for j in 0..last {
            let x = src[j * ss];
            if x != 0 {
                acc += DCT32[j * step][i] as i64 * x as i64;
            }
        }
        dst[i * ds] = acc as i32;
    }
}

/// Inverse transform of an `n × n` block of scaled coefficients (raster) into
/// residuals, in place (§8.6.4.2).
pub fn inverse(c: &mut [i32], n: usize, dst: bool, bd: u32) {
    let mut tmp = [0i32; 32 * 32];
    // Columns first.
    for x in 0..n {
        let mut last = 0;
        for y in 0..n {
            if c[y * n + x] != 0 {
                last = y + 1;
            }
        }
        if last == 0 {
            for y in 0..n {
                tmp[y * n + x] = 0;
            }
            continue;
        }
        if dst {
            for i in 0..4 {
                let mut acc = 0i32;
                for j in 0..4 {
                    acc += DST4[j][i] * c[j * 4 + x];
                }
                tmp[i * 4 + x] = acc;
            }
        } else {
            idct_1d(&c[x..], n, &mut tmp[x..], n, n, last);
        }
        for y in 0..n {
            tmp[y * n + x] = ((tmp[y * n + x] + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // Then rows.
    let shift = 20 - bd as i32;
    let add = 1 << (shift - 1);
    for y in 0..n {
        let row = &tmp[y * n..y * n + n];
        let last = row.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1);
        if last == 0 {
            c[y * n..y * n + n].fill(0);
            continue;
        }
        let mut out = [0i32; 32];
        if dst {
            for i in 0..4 {
                let mut acc = 0i32;
                for j in 0..4 {
                    acc += DST4[j][i] * row[j];
                }
                out[i] = acc;
            }
        } else {
            idct_1d(row, 1, &mut out, 1, n, last);
        }
        for i in 0..n {
            c[y * n + i] = (out[i] + add) >> shift;
        }
    }
}

/// Transform skip (§8.6.4.2 with `transform_skip_flag`): residual is the
/// scaled coefficient shifted up then back down by the bit-depth shift.
pub fn transform_skip(c: &mut [i32], n: usize, bd: u32) {
    let shift = 20 - bd as i32;
    let ts_shift = 5 + n.trailing_zeros() as i32;
    let add = 1 << (shift - 1);
    for v in c.iter_mut().take(n * n) {
        *v = ((*v << ts_shift) + add) >> shift;
    }
}
