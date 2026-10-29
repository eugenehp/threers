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
/// `out[..n]`. `last` is one past the last non-zero input.
///
/// Accumulates a whole output row per input coefficient, over a contiguous
/// row of the matrix, so the inner loop vectorises. `i32` is exact: inputs
/// are clipped to 16 bits and |DCT32| <= 90, so 32 terms stay under 2^27.
#[inline(always)]
fn idct_1d(src: &[i32], ss: usize, out: &mut [i32; 32], n: usize, last: usize) {
    let step = 32 / n;
    let out = &mut out[..n];
    out.fill(0);
    for j in 0..last {
        let x = src[j * ss];
        if x == 0 {
            continue;
        }
        for (o, &m) in out.iter_mut().zip(&DCT32[j * step][..n]) {
            *o += m as i32 * x;
        }
    }
}

/// Inverse transform of an `n × n` block of scaled coefficients (raster) into
/// residuals, in place (§8.6.4.2).
pub fn inverse(c: &mut [i32], n: usize, dst: bool, bd: u32) {
    // The intermediate is sized to the block: most blocks are 4×4, and
    // clearing a 32×32 buffer for each was a measurable part of decoding.
    match n {
        4 => inverse_n::<16>(c, n, dst, bd),
        8 => inverse_n::<64>(c, n, dst, bd),
        16 => inverse_n::<256>(c, n, dst, bd),
        _ => inverse_n::<1024>(c, n, dst, bd),
    }
}

fn inverse_n<const NN: usize>(c: &mut [i32], n: usize, dst: bool, bd: u32) {
    let mut tmp = [0i32; NN];
    let mut out = [0i32; 32];
    // Columns first.
    for x in 0..n {
        let last = (0..n).rev().find(|&y| c[y * n + x] != 0).map_or(0, |y| y + 1);
        if last == 0 {
            continue; // `tmp` is zero there already.
        }
        if dst {
            for i in 0..4 {
                let mut acc = 0i32;
                for j in 0..4 {
                    acc += DST4[j][i] * c[j * 4 + x];
                }
                out[i] = acc;
            }
        } else {
            idct_1d(&c[x..], n, &mut out, n, last);
        }
        for y in 0..n {
            tmp[y * n + x] = ((out[y] + 64) >> 7).clamp(-32768, 32767);
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
        if dst {
            for i in 0..4 {
                let mut acc = 0i32;
                for j in 0..4 {
                    acc += DST4[j][i] * row[j];
                }
                out[i] = acc;
            }
        } else {
            idct_1d(row, 1, &mut out, n, last);
        }
        for (d, &o) in c[y * n..y * n + n].iter_mut().zip(&out[..n]) {
            *d = (o + add) >> shift;
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
