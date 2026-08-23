//! 4×4 DCT / ADST, 8×8 DCT, 16×16 DCT, and 32×32 DCT transforms matching libvpx.
//!
//! Forward: `vpx_fdct4x4_c` / `vp9_fht4x4_c` / `vpx_fdct8x8_c` / `vpx_fdct16x16_c` /
//! `vpx_fdct32x32_c`.
//! Inverse: `vpx_idct4x4_16_add_c` / `vp9_iht4x4_16_add_c` / `vpx_idct8x8_64_add_c` /
//! `vpx_idct16x16_256_add_c` / `vpx_idct32x32_1024_add_c`.
//! Constants from `vpx_dsp/txfm_common.h` (libvpx v1.14.1).

// The butterflies index a flat buffer as the spec indexes a 2D block, so the
// row term is written out even when it is `0 * 4` or `1 * 4`. Collapsing those
// to `input[i]` and `input[4 + i]` would save nothing and would stop the code
// reading line-for-line against libvpx, which is the only way to check it.
#![allow(clippy::erasing_op, clippy::identity_op)]

use crate::codec::vp9::tables::{DC_PRED, H_PRED, V_PRED};

const DCT_CONST_BITS: u32 = 14;

const COSPI_1_64: i64 = 16364;
const COSPI_2_64: i64 = 16305;
const COSPI_3_64: i64 = 16207;
const COSPI_4_64: i64 = 16069;
const COSPI_5_64: i64 = 15893;
const COSPI_6_64: i64 = 15679;
const COSPI_7_64: i64 = 15426;
const COSPI_8_64: i64 = 15137;
const COSPI_9_64: i64 = 14811;
const COSPI_10_64: i64 = 14449;
const COSPI_11_64: i64 = 14053;
const COSPI_12_64: i64 = 13623;
const COSPI_13_64: i64 = 13160;
const COSPI_14_64: i64 = 12665;
const COSPI_15_64: i64 = 12140;
const COSPI_16_64: i64 = 11585;
const COSPI_17_64: i64 = 11003;
const COSPI_18_64: i64 = 10394;
const COSPI_19_64: i64 = 9760;
const COSPI_20_64: i64 = 9102;
const COSPI_21_64: i64 = 8423;
const COSPI_22_64: i64 = 7723;
const COSPI_23_64: i64 = 7005;
const COSPI_24_64: i64 = 6270;
const COSPI_25_64: i64 = 5520;
const COSPI_26_64: i64 = 4756;
const COSPI_27_64: i64 = 3981;
const COSPI_28_64: i64 = 3196;
const COSPI_29_64: i64 = 2404;
const COSPI_30_64: i64 = 1606;
const COSPI_31_64: i64 = 804;

const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

/// VP9 `TX_TYPE` for 4×4 hybrid transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TxType {
    DctDct = 0,
    /// ADST vertical, DCT horizontal — used by `V_PRED`.
    AdstDct = 1,
    /// DCT vertical, ADST horizontal — used by `H_PRED`.
    DctAdst = 2,
    AdstAdst = 3,
}

/// Map an intra prediction mode to its TX type (`intra_mode_to_tx_type_lookup`).
pub fn tx_type_from_mode(mode: i8) -> TxType {
    match mode {
        V_PRED => TxType::AdstDct,
        H_PRED => TxType::DctAdst,
        DC_PRED => TxType::DctDct,
        // D135/TM → ADST_ADST; other diagonals unused by this encoder.
        _ => TxType::AdstAdst,
    }
}

/// A 4-point transform: reads one row or column, writes the transformed one.
type Transform4 = fn(&[i32; 4], &mut [i32; 4]);

#[inline]
fn round_power_of_two(value: i64, n: u32) -> i64 {
    (value + (1i64 << (n - 1))) >> n
}

#[inline]
fn fdct_round_shift(input: i64) -> i32 {
    round_power_of_two(input, DCT_CONST_BITS) as i32
}

#[inline]
fn dct_const_round_shift(input: i64) -> i32 {
    round_power_of_two(input, DCT_CONST_BITS) as i32
}

/// Hardware-style wrap to signed 16-bit (`WRAPLOW` with `CONFIG_EMULATE_HARDWARE`).
#[inline]
fn wraplow(x: i32) -> i32 {
    (x as i16) as i32
}

/// 1-D forward DCT4 (libvpx encoder `fdct4`).
fn fdct4(input: &[i32; 4], output: &mut [i32; 4]) {
    let step = [
        input[0] as i64 + input[3] as i64,
        input[1] as i64 + input[2] as i64,
        input[1] as i64 - input[2] as i64,
        input[0] as i64 - input[3] as i64,
    ];
    let temp1 = (step[0] + step[1]) * COSPI_16_64;
    let temp2 = (step[0] - step[1]) * COSPI_16_64;
    output[0] = fdct_round_shift(temp1);
    output[2] = fdct_round_shift(temp2);
    let temp1 = step[2] * COSPI_24_64 + step[3] * COSPI_8_64;
    let temp2 = -step[2] * COSPI_8_64 + step[3] * COSPI_24_64;
    output[1] = fdct_round_shift(temp1);
    output[3] = fdct_round_shift(temp2);
}

/// 1-D forward ADST4 (libvpx encoder `fadst4`).
fn fadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    let x0 = input[0] as i64;
    let x1 = input[1] as i64;
    let x2 = input[2] as i64;
    let x3 = input[3] as i64;
    if (x0 | x1 | x2 | x3) == 0 {
        *output = [0; 4];
        return;
    }
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_4_9 * x0;
    let s2 = SINPI_2_9 * x1;
    let s3 = SINPI_1_9 * x1;
    let s4 = SINPI_3_9 * x2;
    let s5 = SINPI_4_9 * x3;
    let s6 = SINPI_2_9 * x3;
    let s7 = x0 + x1 - x3;

    let x0 = s0 + s2 + s5;
    let x1 = SINPI_3_9 * s7;
    let x2 = s1 - s3 + s6;
    let x3 = s4;

    let s0 = x0 + x3;
    let s1 = x1;
    let s2 = x2 - x3;
    let s3 = x2 - x0 + x3;

    output[0] = wraplow(fdct_round_shift(s0));
    output[1] = wraplow(fdct_round_shift(s1));
    output[2] = wraplow(fdct_round_shift(s2));
    output[3] = wraplow(fdct_round_shift(s3));
}

/// Forward 4×4 DCT. `input` is a row-major residual block (stride 4).
///
/// Matches libvpx `vpx_fdct4x4_c`.
pub fn fdct4x4(input: &[i16; 16], output: &mut [i32; 16]) {
    fht4x4(input, output, TxType::DctDct);
}

/// Forward 4×4 hybrid transform (`vp9_fht4x4_c`).
pub fn fht4x4(input: &[i16; 16], output: &mut [i32; 16], tx_type: TxType) {
    if tx_type == TxType::DctDct {
        // Dedicated path matches `vpx_fdct4x4_c` bit-exact (same math as FHT
        // with DCT/DCT, kept separate for clarity with existing tests).
        let mut intermediate = [0i32; 16];
        {
            let mut out_col = 0usize;
            for i in 0..4 {
                let mut in_high = [
                    input[0 * 4 + i] as i64 * 16,
                    input[1 * 4 + i] as i64 * 16,
                    input[2 * 4 + i] as i64 * 16,
                    input[3 * 4 + i] as i64 * 16,
                ];
                if i == 0 && in_high[0] != 0 {
                    in_high[0] += 1;
                }
                let step = [
                    in_high[0] + in_high[3],
                    in_high[1] + in_high[2],
                    in_high[1] - in_high[2],
                    in_high[0] - in_high[3],
                ];
                let temp1 = (step[0] + step[1]) * COSPI_16_64;
                let temp2 = (step[0] - step[1]) * COSPI_16_64;
                intermediate[out_col] = fdct_round_shift(temp1);
                intermediate[out_col + 2] = fdct_round_shift(temp2);
                let temp1 = step[2] * COSPI_24_64 + step[3] * COSPI_8_64;
                let temp2 = -step[2] * COSPI_8_64 + step[3] * COSPI_24_64;
                intermediate[out_col + 1] = fdct_round_shift(temp1);
                intermediate[out_col + 3] = fdct_round_shift(temp2);
                out_col += 4;
            }
        }
        {
            let mut out_col = 0usize;
            for i in 0..4 {
                let in_high = [
                    intermediate[0 * 4 + i] as i64,
                    intermediate[1 * 4 + i] as i64,
                    intermediate[2 * 4 + i] as i64,
                    intermediate[3 * 4 + i] as i64,
                ];
                let step = [
                    in_high[0] + in_high[3],
                    in_high[1] + in_high[2],
                    in_high[1] - in_high[2],
                    in_high[0] - in_high[3],
                ];
                let temp1 = (step[0] + step[1]) * COSPI_16_64;
                let temp2 = (step[0] - step[1]) * COSPI_16_64;
                output[out_col] = fdct_round_shift(temp1);
                output[out_col + 2] = fdct_round_shift(temp2);
                let temp1 = step[2] * COSPI_24_64 + step[3] * COSPI_8_64;
                let temp2 = -step[2] * COSPI_8_64 + step[3] * COSPI_24_64;
                output[out_col + 1] = fdct_round_shift(temp1);
                output[out_col + 3] = fdct_round_shift(temp2);
                out_col += 4;
            }
        }
        for v in output.iter_mut() {
            *v = (*v + 1) >> 2;
        }
        return;
    }

    let (cols, rows): (Transform4, Transform4) = match tx_type {
        TxType::AdstDct => (fadst4, fdct4),
        TxType::DctAdst => (fdct4, fadst4),
        TxType::AdstAdst => (fadst4, fadst4),
        TxType::DctDct => unreachable!(),
    };

    let mut out = [0i32; 16];
    // Columns
    for i in 0..4 {
        let mut temp_in = [
            input[0 * 4 + i] as i32 * 16,
            input[1 * 4 + i] as i32 * 16,
            input[2 * 4 + i] as i32 * 16,
            input[3 * 4 + i] as i32 * 16,
        ];
        if i == 0 && temp_in[0] != 0 {
            temp_in[0] += 1;
        }
        let mut temp_out = [0i32; 4];
        cols(&temp_in, &mut temp_out);
        for j in 0..4 {
            out[j * 4 + i] = temp_out[j];
        }
    }
    // Rows
    for i in 0..4 {
        let temp_in = [out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]];
        let mut temp_out = [0i32; 4];
        rows(&temp_in, &mut temp_out);
        for j in 0..4 {
            output[i * 4 + j] = (temp_out[j] + 1) >> 2;
        }
    }
}

/// 1-D IDCT4 (libvpx `idct4_c`).
fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    let i0 = input[0] as i16 as i64;
    let i1 = input[1] as i16 as i64;
    let i2 = input[2] as i16 as i64;
    let i3 = input[3] as i16 as i64;

    let temp1 = (i0 + i2) * COSPI_16_64;
    let temp2 = (i0 - i2) * COSPI_16_64;
    let step0 = wraplow(dct_const_round_shift(temp1));
    let step1 = wraplow(dct_const_round_shift(temp2));
    let temp1 = i1 * COSPI_24_64 - i3 * COSPI_8_64;
    let temp2 = i1 * COSPI_8_64 + i3 * COSPI_24_64;
    let step2 = wraplow(dct_const_round_shift(temp1));
    let step3 = wraplow(dct_const_round_shift(temp2));

    output[0] = wraplow(step0 + step3);
    output[1] = wraplow(step1 + step2);
    output[2] = wraplow(step1 - step2);
    output[3] = wraplow(step0 - step3);
}

/// 1-D IADST4 (libvpx `iadst4_c`).
fn iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    let x0 = input[0] as i16 as i32;
    let x1 = input[1] as i16 as i32;
    let x2 = input[2] as i16 as i32;
    let x3 = input[3] as i16 as i32;
    if (x0 | x1 | x2 | x3) == 0 {
        *output = [0; 4];
        return;
    }

    let s0 = SINPI_1_9 * x0 as i64;
    let s1 = SINPI_2_9 * x0 as i64;
    let s2 = SINPI_3_9 * x1 as i64;
    let s3 = SINPI_4_9 * x2 as i64;
    let s4 = SINPI_1_9 * x2 as i64;
    let s5 = SINPI_2_9 * x3 as i64;
    let s6 = SINPI_4_9 * x3 as i64;
    let s7 = wraplow(x0 - x2 + x3) as i64;

    let s0 = s0 + s3 + s5;
    let s1 = s1 - s4 - s6;
    let s3 = s2;
    let s2 = SINPI_3_9 * s7;

    output[0] = wraplow(dct_const_round_shift(s0 + s3));
    output[1] = wraplow(dct_const_round_shift(s1 + s3));
    output[2] = wraplow(dct_const_round_shift(s2));
    output[3] = wraplow(dct_const_round_shift(s0 + s1 - s3));
}

#[inline]
fn clip_pixel(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

#[inline]
fn clip_pixel_add(dest: u8, trans: i32) -> u8 {
    clip_pixel(dest as i32 + wraplow(trans))
}

/// Inverse 4×4 DCT and add into `dest` (libvpx `vpx_idct4x4_16_add_c`).
pub fn idct4x4_add(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    iht4x4_add(input, dest, stride, TxType::DctDct);
}

/// Inverse 4×4 hybrid transform and add (`vp9_iht4x4_16_add_c`).
///
/// libvpx's `transform_2d` is `{ cols, rows }`; IHT applies `.rows` first
/// (horizontal), then `.cols` (vertical). For `ADST_DCT` that is idct-then-iadst.
pub fn iht4x4_add(input: &[i32; 16], dest: &mut [u8], stride: usize, tx_type: TxType) {
    assert!(dest.len() >= 3 * stride + 4);

    // (cols, rows) — same packing as libvpx `transform_2d`.
    let (cols, rows): (Transform4, Transform4) = match tx_type {
        TxType::DctDct => (idct4, idct4),
        TxType::AdstDct => (iadst4, idct4),
        TxType::DctAdst => (idct4, iadst4),
        TxType::AdstAdst => (iadst4, iadst4),
    };

    let mut out = [0i32; 16];
    // Rows (horizontal) first.
    for i in 0..4 {
        let row_in = [
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ];
        let mut row_out = [0i32; 4];
        rows(&row_in, &mut row_out);
        out[i * 4..i * 4 + 4].copy_from_slice(&row_out);
    }
    // Columns (vertical) second.
    for i in 0..4 {
        let col_in = [
            out[0 * 4 + i],
            out[1 * 4 + i],
            out[2 * 4 + i],
            out[3 * 4 + i],
        ];
        let mut col_out = [0i32; 4];
        cols(&col_in, &mut col_out);
        for j in 0..4 {
            let add = round_power_of_two(col_out[j] as i64, 4) as i32;
            let idx = j * stride + i;
            dest[idx] = clip_pixel_add(dest[idx], add);
        }
    }
}

/// Forward 8×8 DCT (`vpx_fdct8x8_c`). `input` is row-major residual (stride 8).
pub fn fdct8x8(input: &[i16; 64], output: &mut [i32; 64]) {
    let mut intermediate = [0i32; 64];
    for pass in 0..2 {
        for i in 0..8 {
            let (s0, s1, s2, s3, s4, s5, s6, s7) = if pass == 0 {
                (
                    (input[0 * 8 + i] as i64 + input[7 * 8 + i] as i64) * 4,
                    (input[1 * 8 + i] as i64 + input[6 * 8 + i] as i64) * 4,
                    (input[2 * 8 + i] as i64 + input[5 * 8 + i] as i64) * 4,
                    (input[3 * 8 + i] as i64 + input[4 * 8 + i] as i64) * 4,
                    (input[3 * 8 + i] as i64 - input[4 * 8 + i] as i64) * 4,
                    (input[2 * 8 + i] as i64 - input[5 * 8 + i] as i64) * 4,
                    (input[1 * 8 + i] as i64 - input[6 * 8 + i] as i64) * 4,
                    (input[0 * 8 + i] as i64 - input[7 * 8 + i] as i64) * 4,
                )
            } else {
                (
                    intermediate[0 * 8 + i] as i64 + intermediate[7 * 8 + i] as i64,
                    intermediate[1 * 8 + i] as i64 + intermediate[6 * 8 + i] as i64,
                    intermediate[2 * 8 + i] as i64 + intermediate[5 * 8 + i] as i64,
                    intermediate[3 * 8 + i] as i64 + intermediate[4 * 8 + i] as i64,
                    intermediate[3 * 8 + i] as i64 - intermediate[4 * 8 + i] as i64,
                    intermediate[2 * 8 + i] as i64 - intermediate[5 * 8 + i] as i64,
                    intermediate[1 * 8 + i] as i64 - intermediate[6 * 8 + i] as i64,
                    intermediate[0 * 8 + i] as i64 - intermediate[7 * 8 + i] as i64,
                )
            };

            let x0 = s0 + s3;
            let x1 = s1 + s2;
            let x2 = s1 - s2;
            let x3 = s0 - s3;
            let t0 = (x0 + x1) * COSPI_16_64;
            let t1 = (x0 - x1) * COSPI_16_64;
            let t2 = x2 * COSPI_24_64 + x3 * COSPI_8_64;
            let t3 = -x2 * COSPI_8_64 + x3 * COSPI_24_64;

            let dest = if pass == 0 {
                &mut intermediate[i * 8..i * 8 + 8]
            } else {
                &mut output[i * 8..i * 8 + 8]
            };
            dest[0] = fdct_round_shift(t0);
            dest[2] = fdct_round_shift(t2);
            dest[4] = fdct_round_shift(t1);
            dest[6] = fdct_round_shift(t3);

            let t0 = (s6 - s5) * COSPI_16_64;
            let t1 = (s6 + s5) * COSPI_16_64;
            let t2 = fdct_round_shift(t0) as i64;
            let t3 = fdct_round_shift(t1) as i64;

            let x0 = s4 + t2;
            let x1 = s4 - t2;
            let x2 = s7 - t3;
            let x3 = s7 + t3;

            let t0 = x0 * COSPI_28_64 + x3 * COSPI_4_64;
            let t1 = x1 * COSPI_12_64 + x2 * COSPI_20_64;
            let t2 = x2 * COSPI_12_64 + x1 * (-COSPI_20_64);
            let t3 = x3 * COSPI_28_64 + x0 * (-COSPI_4_64);
            dest[1] = fdct_round_shift(t0);
            dest[3] = fdct_round_shift(t2);
            dest[5] = fdct_round_shift(t1);
            dest[7] = fdct_round_shift(t3);
        }
    }
    for v in output.iter_mut() {
        *v /= 2;
    }
}

/// One 1-D forward DCT16 pass (libvpx `vpx_fdct16x16_c` inner loop).
fn fdct16_1d(in_high: [i64; 8], step1: [i64; 8], out: &mut [i32; 16]) {
    let mut step2 = [0i64; 8];
    let mut step3 = [0i64; 8];

    // Even part (fdct8 on sums).
    {
        let s0 = in_high[0] + in_high[7];
        let s1 = in_high[1] + in_high[6];
        let s2 = in_high[2] + in_high[5];
        let s3 = in_high[3] + in_high[4];
        let s4 = in_high[3] - in_high[4];
        let s5 = in_high[2] - in_high[5];
        let s6 = in_high[1] - in_high[6];
        let s7 = in_high[0] - in_high[7];

        let x0 = s0 + s3;
        let x1 = s1 + s2;
        let x2 = s1 - s2;
        let x3 = s0 - s3;
        let t0 = (x0 + x1) * COSPI_16_64;
        let t1 = (x0 - x1) * COSPI_16_64;
        let t2 = x3 * COSPI_8_64 + x2 * COSPI_24_64;
        let t3 = x3 * COSPI_24_64 - x2 * COSPI_8_64;
        out[0] = fdct_round_shift(t0);
        out[4] = fdct_round_shift(t2);
        out[8] = fdct_round_shift(t1);
        out[12] = fdct_round_shift(t3);

        let t0 = (s6 - s5) * COSPI_16_64;
        let t1 = (s6 + s5) * COSPI_16_64;
        let t2 = fdct_round_shift(t0) as i64;
        let t3 = fdct_round_shift(t1) as i64;

        let x0 = s4 + t2;
        let x1 = s4 - t2;
        let x2 = s7 - t3;
        let x3 = s7 + t3;

        let t0 = x0 * COSPI_28_64 + x3 * COSPI_4_64;
        let t1 = x1 * COSPI_12_64 + x2 * COSPI_20_64;
        let t2 = x2 * COSPI_12_64 + x1 * (-COSPI_20_64);
        let t3 = x3 * COSPI_28_64 + x0 * (-COSPI_4_64);
        out[2] = fdct_round_shift(t0);
        out[6] = fdct_round_shift(t2);
        out[10] = fdct_round_shift(t1);
        out[14] = fdct_round_shift(t3);
    }

    // Odd part (step1 -> odd_results).
    {
        let temp1 = (step1[5] - step1[2]) * COSPI_16_64;
        let temp2 = (step1[4] - step1[3]) * COSPI_16_64;
        step2[2] = fdct_round_shift(temp1) as i64;
        step2[3] = fdct_round_shift(temp2) as i64;
        let temp1 = (step1[4] + step1[3]) * COSPI_16_64;
        let temp2 = (step1[5] + step1[2]) * COSPI_16_64;
        step2[4] = fdct_round_shift(temp1) as i64;
        step2[5] = fdct_round_shift(temp2) as i64;

        step3[0] = step1[0] + step2[3];
        step3[1] = step1[1] + step2[2];
        step3[2] = step1[1] - step2[2];
        step3[3] = step1[0] - step2[3];
        step3[4] = step1[7] - step2[4];
        step3[5] = step1[6] - step2[5];
        step3[6] = step1[6] + step2[5];
        step3[7] = step1[7] + step2[4];

        let temp1 = step3[1] * (-COSPI_8_64) + step3[6] * COSPI_24_64;
        let temp2 = step3[2] * COSPI_24_64 + step3[5] * COSPI_8_64;
        step2[1] = fdct_round_shift(temp1) as i64;
        step2[2] = fdct_round_shift(temp2) as i64;
        let temp1 = step3[2] * COSPI_8_64 - step3[5] * COSPI_24_64;
        let temp2 = step3[1] * COSPI_24_64 + step3[6] * COSPI_8_64;
        step2[5] = fdct_round_shift(temp1) as i64;
        step2[6] = fdct_round_shift(temp2) as i64;

        let mut s1 = step1;
        s1[0] = step3[0] + step2[1];
        s1[1] = step3[0] - step2[1];
        s1[2] = step3[3] + step2[2];
        s1[3] = step3[3] - step2[2];
        s1[4] = step3[4] - step2[5];
        s1[5] = step3[4] + step2[5];
        s1[6] = step3[7] - step2[6];
        s1[7] = step3[7] + step2[6];

        let temp1 = s1[0] * COSPI_30_64 + s1[7] * COSPI_2_64;
        let temp2 = s1[1] * COSPI_14_64 + s1[6] * COSPI_18_64;
        out[1] = fdct_round_shift(temp1);
        out[9] = fdct_round_shift(temp2);
        let temp1 = s1[2] * COSPI_22_64 + s1[5] * COSPI_10_64;
        let temp2 = s1[3] * COSPI_6_64 + s1[4] * COSPI_26_64;
        out[5] = fdct_round_shift(temp1);
        out[13] = fdct_round_shift(temp2);
        let temp1 = s1[3] * (-COSPI_26_64) + s1[4] * COSPI_6_64;
        let temp2 = s1[2] * (-COSPI_10_64) + s1[5] * COSPI_22_64;
        out[3] = fdct_round_shift(temp1);
        out[11] = fdct_round_shift(temp2);
        let temp1 = s1[1] * (-COSPI_18_64) + s1[6] * COSPI_14_64;
        let temp2 = s1[0] * (-COSPI_2_64) + s1[7] * COSPI_30_64;
        out[7] = fdct_round_shift(temp1);
        out[15] = fdct_round_shift(temp2);
    }
}

/// Forward 16×16 DCT (`vpx_fdct16x16_c`). `input` is row-major residual (stride 16).
pub fn fdct16x16(input: &[i16; 256], output: &mut [i32; 256]) {
    let mut intermediate = [0i32; 256];
    for pass in 0..2 {
        for i in 0..16 {
            let (in_high, step1) = if pass == 0 {
                let col = |row: usize| input[row * 16 + i] as i64;
                (
                    [
                        (col(0) + col(15)) * 4,
                        (col(1) + col(14)) * 4,
                        (col(2) + col(13)) * 4,
                        (col(3) + col(12)) * 4,
                        (col(4) + col(11)) * 4,
                        (col(5) + col(10)) * 4,
                        (col(6) + col(9)) * 4,
                        (col(7) + col(8)) * 4,
                    ],
                    [
                        (col(7) - col(8)) * 4,
                        (col(6) - col(9)) * 4,
                        (col(5) - col(10)) * 4,
                        (col(4) - col(11)) * 4,
                        (col(3) - col(12)) * 4,
                        (col(2) - col(13)) * 4,
                        (col(1) - col(14)) * 4,
                        (col(0) - col(15)) * 4,
                    ],
                )
            } else {
                let col = |row: usize| {
                    let v = intermediate[row * 16 + i];
                    ((v + 1) >> 2) as i64
                };
                (
                    [
                        col(0) + col(15),
                        col(1) + col(14),
                        col(2) + col(13),
                        col(3) + col(12),
                        col(4) + col(11),
                        col(5) + col(10),
                        col(6) + col(9),
                        col(7) + col(8),
                    ],
                    [
                        col(7) - col(8),
                        col(6) - col(9),
                        col(5) - col(10),
                        col(4) - col(11),
                        col(3) - col(12),
                        col(2) - col(13),
                        col(1) - col(14),
                        col(0) - col(15),
                    ],
                )
            };

            let dest = if pass == 0 {
                &mut intermediate[i * 16..i * 16 + 16]
            } else {
                &mut output[i * 16..i * 16 + 16]
            };
            fdct16_1d(in_high, step1, dest.try_into().unwrap());
        }
    }
}

fn idct8(input: &[i32; 8], output: &mut [i32; 8]) {
    let mut step1 = [0i32; 8];
    let mut step2 = [0i32; 8];

    step1[0] = input[0] as i16 as i32;
    step1[2] = input[4] as i16 as i32;
    step1[1] = input[2] as i16 as i32;
    step1[3] = input[6] as i16 as i32;
    let temp1 = input[1] as i16 as i64 * COSPI_28_64 - input[7] as i16 as i64 * COSPI_4_64;
    let temp2 = input[1] as i16 as i64 * COSPI_4_64 + input[7] as i16 as i64 * COSPI_28_64;
    step1[4] = wraplow(dct_const_round_shift(temp1));
    step1[7] = wraplow(dct_const_round_shift(temp2));
    let temp1 = input[5] as i16 as i64 * COSPI_12_64 - input[3] as i16 as i64 * COSPI_20_64;
    let temp2 = input[5] as i16 as i64 * COSPI_20_64 + input[3] as i16 as i64 * COSPI_12_64;
    step1[5] = wraplow(dct_const_round_shift(temp1));
    step1[6] = wraplow(dct_const_round_shift(temp2));

    let temp1 = (step1[0] as i64 + step1[2] as i64) * COSPI_16_64;
    let temp2 = (step1[0] as i64 - step1[2] as i64) * COSPI_16_64;
    step2[0] = wraplow(dct_const_round_shift(temp1));
    step2[1] = wraplow(dct_const_round_shift(temp2));
    let temp1 = step1[1] as i64 * COSPI_24_64 - step1[3] as i64 * COSPI_8_64;
    let temp2 = step1[1] as i64 * COSPI_8_64 + step1[3] as i64 * COSPI_24_64;
    step2[2] = wraplow(dct_const_round_shift(temp1));
    step2[3] = wraplow(dct_const_round_shift(temp2));
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64;
    let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64;
    step1[5] = wraplow(dct_const_round_shift(temp1));
    step1[6] = wraplow(dct_const_round_shift(temp2));
    step1[7] = step2[7];

    output[0] = wraplow(step1[0] + step1[7]);
    output[1] = wraplow(step1[1] + step1[6]);
    output[2] = wraplow(step1[2] + step1[5]);
    output[3] = wraplow(step1[3] + step1[4]);
    output[4] = wraplow(step1[3] - step1[4]);
    output[5] = wraplow(step1[2] - step1[5]);
    output[6] = wraplow(step1[1] - step1[6]);
    output[7] = wraplow(step1[0] - step1[7]);
}

/// 1-D IDCT16 (libvpx `idct16_c`).
fn idct16(input: &[i32; 16], output: &mut [i32; 16]) {
    let mut step1 = [0i32; 16];
    let mut step2 = [0i32; 16];

    // stage 1
    step1[0] = input[0] as i16 as i32;
    step1[1] = input[8] as i16 as i32;
    step1[2] = input[4] as i16 as i32;
    step1[3] = input[12] as i16 as i32;
    step1[4] = input[2] as i16 as i32;
    step1[5] = input[10] as i16 as i32;
    step1[6] = input[6] as i16 as i32;
    step1[7] = input[14] as i16 as i32;
    step1[8] = input[1] as i16 as i32;
    step1[9] = input[9] as i16 as i32;
    step1[10] = input[5] as i16 as i32;
    step1[11] = input[13] as i16 as i32;
    step1[12] = input[3] as i16 as i32;
    step1[13] = input[11] as i16 as i32;
    step1[14] = input[7] as i16 as i32;
    step1[15] = input[15] as i16 as i32;

    // stage 2
    step2[0] = step1[0];
    step2[1] = step1[1];
    step2[2] = step1[2];
    step2[3] = step1[3];
    step2[4] = step1[4];
    step2[5] = step1[5];
    step2[6] = step1[6];
    step2[7] = step1[7];

    let temp1 = step1[8] as i64 * COSPI_30_64 - step1[15] as i64 * COSPI_2_64;
    let temp2 = step1[8] as i64 * COSPI_2_64 + step1[15] as i64 * COSPI_30_64;
    step2[8] = wraplow(dct_const_round_shift(temp1));
    step2[15] = wraplow(dct_const_round_shift(temp2));

    let temp1 = step1[9] as i64 * COSPI_14_64 - step1[14] as i64 * COSPI_18_64;
    let temp2 = step1[9] as i64 * COSPI_18_64 + step1[14] as i64 * COSPI_14_64;
    step2[9] = wraplow(dct_const_round_shift(temp1));
    step2[14] = wraplow(dct_const_round_shift(temp2));

    let temp1 = step1[10] as i64 * COSPI_22_64 - step1[13] as i64 * COSPI_10_64;
    let temp2 = step1[10] as i64 * COSPI_10_64 + step1[13] as i64 * COSPI_22_64;
    step2[10] = wraplow(dct_const_round_shift(temp1));
    step2[13] = wraplow(dct_const_round_shift(temp2));

    let temp1 = step1[11] as i64 * COSPI_6_64 - step1[12] as i64 * COSPI_26_64;
    let temp2 = step1[11] as i64 * COSPI_26_64 + step1[12] as i64 * COSPI_6_64;
    step2[11] = wraplow(dct_const_round_shift(temp1));
    step2[12] = wraplow(dct_const_round_shift(temp2));

    // stage 3
    step1[0] = step2[0];
    step1[1] = step2[1];
    step1[2] = step2[2];
    step1[3] = step2[3];

    let temp1 = step2[4] as i64 * COSPI_28_64 - step2[7] as i64 * COSPI_4_64;
    let temp2 = step2[4] as i64 * COSPI_4_64 + step2[7] as i64 * COSPI_28_64;
    step1[4] = wraplow(dct_const_round_shift(temp1));
    step1[7] = wraplow(dct_const_round_shift(temp2));
    let temp1 = step2[5] as i64 * COSPI_12_64 - step2[6] as i64 * COSPI_20_64;
    let temp2 = step2[5] as i64 * COSPI_20_64 + step2[6] as i64 * COSPI_12_64;
    step1[5] = wraplow(dct_const_round_shift(temp1));
    step1[6] = wraplow(dct_const_round_shift(temp2));

    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);

    // stage 4
    let temp1 = (step1[0] as i64 + step1[1] as i64) * COSPI_16_64;
    let temp2 = (step1[0] as i64 - step1[1] as i64) * COSPI_16_64;
    step2[0] = wraplow(dct_const_round_shift(temp1));
    step2[1] = wraplow(dct_const_round_shift(temp2));
    let temp1 = step1[2] as i64 * COSPI_24_64 - step1[3] as i64 * COSPI_8_64;
    let temp2 = step1[2] as i64 * COSPI_8_64 + step1[3] as i64 * COSPI_24_64;
    step2[2] = wraplow(dct_const_round_shift(temp1));
    step2[3] = wraplow(dct_const_round_shift(temp2));
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step2[8] = step1[8];
    step2[15] = step1[15];
    let temp1 = -step1[9] as i64 * COSPI_8_64 + step1[14] as i64 * COSPI_24_64;
    let temp2 = step1[9] as i64 * COSPI_24_64 + step1[14] as i64 * COSPI_8_64;
    step2[9] = wraplow(dct_const_round_shift(temp1));
    step2[14] = wraplow(dct_const_round_shift(temp2));
    let temp1 = -step1[10] as i64 * COSPI_24_64 - step1[13] as i64 * COSPI_8_64;
    let temp2 = -step1[10] as i64 * COSPI_8_64 + step1[13] as i64 * COSPI_24_64;
    step2[10] = wraplow(dct_const_round_shift(temp1));
    step2[13] = wraplow(dct_const_round_shift(temp2));
    step2[11] = step1[11];
    step2[12] = step1[12];

    // stage 5
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64;
    let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64;
    step1[5] = wraplow(dct_const_round_shift(temp1));
    step1[6] = wraplow(dct_const_round_shift(temp2));
    step1[7] = step2[7];

    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);

    // stage 6
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    let temp1 = (-step1[10] + step1[13]) as i64 * COSPI_16_64;
    let temp2 = (step1[10] + step1[13]) as i64 * COSPI_16_64;
    step2[10] = wraplow(dct_const_round_shift(temp1));
    step2[13] = wraplow(dct_const_round_shift(temp2));
    let temp1 = (-step1[11] + step1[12]) as i64 * COSPI_16_64;
    let temp2 = (step1[11] + step1[12]) as i64 * COSPI_16_64;
    step2[11] = wraplow(dct_const_round_shift(temp1));
    step2[12] = wraplow(dct_const_round_shift(temp2));
    step2[14] = step1[14];
    step2[15] = step1[15];

    // stage 7
    output[0] = wraplow(step2[0] + step2[15]);
    output[1] = wraplow(step2[1] + step2[14]);
    output[2] = wraplow(step2[2] + step2[13]);
    output[3] = wraplow(step2[3] + step2[12]);
    output[4] = wraplow(step2[4] + step2[11]);
    output[5] = wraplow(step2[5] + step2[10]);
    output[6] = wraplow(step2[6] + step2[9]);
    output[7] = wraplow(step2[7] + step2[8]);
    output[8] = wraplow(step2[7] - step2[8]);
    output[9] = wraplow(step2[6] - step2[9]);
    output[10] = wraplow(step2[5] - step2[10]);
    output[11] = wraplow(step2[4] - step2[11]);
    output[12] = wraplow(step2[3] - step2[12]);
    output[13] = wraplow(step2[2] - step2[13]);
    output[14] = wraplow(step2[1] - step2[14]);
    output[15] = wraplow(step2[0] - step2[15]);
}

/// Inverse 8×8 DCT and add into `dest` (`vpx_idct8x8_64_add_c`).
pub fn idct8x8_add(input: &[i32; 64], dest: &mut [u8], stride: usize) {
    assert!(dest.len() >= 7 * stride + 8);
    let mut out = [0i32; 64];
    for i in 0..8 {
        let mut row_in = [0i32; 8];
        row_in.copy_from_slice(&input[i * 8..i * 8 + 8]);
        let mut row_out = [0i32; 8];
        idct8(&row_in, &mut row_out);
        out[i * 8..i * 8 + 8].copy_from_slice(&row_out);
    }
    for i in 0..8 {
        let col_in = [
            out[0 * 8 + i],
            out[1 * 8 + i],
            out[2 * 8 + i],
            out[3 * 8 + i],
            out[4 * 8 + i],
            out[5 * 8 + i],
            out[6 * 8 + i],
            out[7 * 8 + i],
        ];
        let mut col_out = [0i32; 8];
        idct8(&col_in, &mut col_out);
        for j in 0..8 {
            let add = round_power_of_two(col_out[j] as i64, 5) as i32;
            let idx = j * stride + i;
            dest[idx] = clip_pixel_add(dest[idx], add);
        }
    }
}

/// Inverse 16×16 DCT and add into `dest` (`vpx_idct16x16_256_add_c`).
pub fn idct16x16_add(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    assert!(dest.len() >= 15 * stride + 16);
    let mut out = [0i32; 256];
    for i in 0..16 {
        let row_in: [i32; 16] = input[i * 16..i * 16 + 16].try_into().unwrap();
        let mut row_out = [0i32; 16];
        idct16(&row_in, &mut row_out);
        out[i * 16..i * 16 + 16].copy_from_slice(&row_out);
    }
    for i in 0..16 {
        let col_in = [
            out[0 * 16 + i],
            out[1 * 16 + i],
            out[2 * 16 + i],
            out[3 * 16 + i],
            out[4 * 16 + i],
            out[5 * 16 + i],
            out[6 * 16 + i],
            out[7 * 16 + i],
            out[8 * 16 + i],
            out[9 * 16 + i],
            out[10 * 16 + i],
            out[11 * 16 + i],
            out[12 * 16 + i],
            out[13 * 16 + i],
            out[14 * 16 + i],
            out[15 * 16 + i],
        ];
        let mut col_out = [0i32; 16];
        idct16(&col_in, &mut col_out);
        for j in 0..16 {
            let add = round_power_of_two(col_out[j] as i64, 6) as i32;
            let idx = j * stride + i;
            dest[idx] = clip_pixel_add(dest[idx], add);
        }
    }
}

#[inline]
fn dct_32_round(input: i64) -> i32 {
    fdct_round_shift(input)
}

#[inline]
fn half_round_shift(x: i32) -> i32 {
    (x + 1) >> 2
}

/// 1-D forward DCT32 (libvpx `vpx_fdct32`). `round` applies `half_round_shift` after stage 2.
fn fdct32(input: &[i64; 32], output: &mut [i32; 32], round: bool) {
    let mut step = [0i64; 32];

    // Stage 1
    for i in 0..16 {
        step[i] = input[i] + input[31 - i];
    }
    for i in 0..16 {
        step[16 + i] = -input[16 + i] + input[15 - i];
    }

    // Stage 2
    output[0] = (step[0] + step[15]) as i32;
    output[1] = (step[1] + step[14]) as i32;
    output[2] = (step[2] + step[13]) as i32;
    output[3] = (step[3] + step[12]) as i32;
    output[4] = (step[4] + step[11]) as i32;
    output[5] = (step[5] + step[10]) as i32;
    output[6] = (step[6] + step[9]) as i32;
    output[7] = (step[7] + step[8]) as i32;
    output[8] = (-step[8] + step[7]) as i32;
    output[9] = (-step[9] + step[6]) as i32;
    output[10] = (-step[10] + step[5]) as i32;
    output[11] = (-step[11] + step[4]) as i32;
    output[12] = (-step[12] + step[3]) as i32;
    output[13] = (-step[13] + step[2]) as i32;
    output[14] = (-step[14] + step[1]) as i32;
    output[15] = (-step[15] + step[0]) as i32;

    output[16] = step[16] as i32;
    output[17] = step[17] as i32;
    output[18] = step[18] as i32;
    output[19] = step[19] as i32;

    output[20] = dct_32_round((-step[20] + step[27]) * COSPI_16_64);
    output[21] = dct_32_round((-step[21] + step[26]) * COSPI_16_64);
    output[22] = dct_32_round((-step[22] + step[25]) * COSPI_16_64);
    output[23] = dct_32_round((-step[23] + step[24]) * COSPI_16_64);

    output[24] = dct_32_round((step[24] + step[23]) * COSPI_16_64);
    output[25] = dct_32_round((step[25] + step[22]) * COSPI_16_64);
    output[26] = dct_32_round((step[26] + step[21]) * COSPI_16_64);
    output[27] = dct_32_round((step[27] + step[20]) * COSPI_16_64);

    output[28] = step[28] as i32;
    output[29] = step[29] as i32;
    output[30] = step[30] as i32;
    output[31] = step[31] as i32;

    if round {
        for v in output.iter_mut() {
            *v = half_round_shift(*v);
        }
    }

    // Stage 3 — reload step from output
    for i in 0..32 {
        step[i] = output[i] as i64;
    }
    step[0] = output[0] as i64 + output[7] as i64;
    step[1] = output[1] as i64 + output[6] as i64;
    step[2] = output[2] as i64 + output[5] as i64;
    step[3] = output[3] as i64 + output[4] as i64;
    step[4] = -output[4] as i64 + output[3] as i64;
    step[5] = -output[5] as i64 + output[2] as i64;
    step[6] = -output[6] as i64 + output[1] as i64;
    step[7] = -output[7] as i64 + output[0] as i64;
    step[8] = output[8] as i64;
    step[9] = output[9] as i64;
    step[10] = dct_32_round((-output[10] as i64 + output[13] as i64) * COSPI_16_64) as i64;
    step[11] = dct_32_round((-output[11] as i64 + output[12] as i64) * COSPI_16_64) as i64;
    step[12] = dct_32_round((output[12] as i64 + output[11] as i64) * COSPI_16_64) as i64;
    step[13] = dct_32_round((output[13] as i64 + output[10] as i64) * COSPI_16_64) as i64;
    step[14] = output[14] as i64;
    step[15] = output[15] as i64;

    step[16] = output[16] as i64 + output[23] as i64;
    step[17] = output[17] as i64 + output[22] as i64;
    step[18] = output[18] as i64 + output[21] as i64;
    step[19] = output[19] as i64 + output[20] as i64;
    step[20] = -output[20] as i64 + output[19] as i64;
    step[21] = -output[21] as i64 + output[18] as i64;
    step[22] = -output[22] as i64 + output[17] as i64;
    step[23] = -output[23] as i64 + output[16] as i64;
    step[24] = -output[24] as i64 + output[31] as i64;
    step[25] = -output[25] as i64 + output[30] as i64;
    step[26] = -output[26] as i64 + output[29] as i64;
    step[27] = -output[27] as i64 + output[28] as i64;
    step[28] = output[28] as i64 + output[27] as i64;
    step[29] = output[29] as i64 + output[26] as i64;
    step[30] = output[30] as i64 + output[25] as i64;
    step[31] = output[31] as i64 + output[24] as i64;

    // Stage 4
    output[0] = (step[0] + step[3]) as i32;
    output[1] = (step[1] + step[2]) as i32;
    output[2] = (-step[2] + step[1]) as i32;
    output[3] = (-step[3] + step[0]) as i32;
    output[4] = step[4] as i32;
    output[5] = dct_32_round((-step[5] + step[6]) * COSPI_16_64);
    output[6] = dct_32_round((step[6] + step[5]) * COSPI_16_64);
    output[7] = step[7] as i32;
    output[8] = (step[8] + step[11]) as i32;
    output[9] = (step[9] + step[10]) as i32;
    output[10] = (-step[10] + step[9]) as i32;
    output[11] = (-step[11] + step[8]) as i32;
    output[12] = (-step[12] + step[15]) as i32;
    output[13] = (-step[13] + step[14]) as i32;
    output[14] = (step[14] + step[13]) as i32;
    output[15] = (step[15] + step[12]) as i32;

    output[16] = step[16] as i32;
    output[17] = step[17] as i32;
    output[18] = dct_32_round(step[18] * -COSPI_8_64 + step[29] * COSPI_24_64);
    output[19] = dct_32_round(step[19] * -COSPI_8_64 + step[28] * COSPI_24_64);
    output[20] = dct_32_round(step[20] * -COSPI_24_64 + step[27] * -COSPI_8_64);
    output[21] = dct_32_round(step[21] * -COSPI_24_64 + step[26] * -COSPI_8_64);
    output[22] = step[22] as i32;
    output[23] = step[23] as i32;
    output[24] = step[24] as i32;
    output[25] = step[25] as i32;
    output[26] = dct_32_round(step[26] * COSPI_24_64 + step[21] * -COSPI_8_64);
    output[27] = dct_32_round(step[27] * COSPI_24_64 + step[20] * -COSPI_8_64);
    output[28] = dct_32_round(step[28] * COSPI_8_64 + step[19] * COSPI_24_64);
    output[29] = dct_32_round(step[29] * COSPI_8_64 + step[18] * COSPI_24_64);
    output[30] = step[30] as i32;
    output[31] = step[31] as i32;

    // Stage 5 — reload step from output
    for i in 0..32 {
        step[i] = output[i] as i64;
    }
    step[0] = dct_32_round((output[0] as i64 + output[1] as i64) * COSPI_16_64) as i64;
    step[1] = dct_32_round((-output[1] as i64 + output[0] as i64) * COSPI_16_64) as i64;
    step[2] = dct_32_round(output[2] as i64 * COSPI_24_64 + output[3] as i64 * COSPI_8_64) as i64;
    step[3] = dct_32_round(output[3] as i64 * COSPI_24_64 - output[2] as i64 * COSPI_8_64) as i64;
    step[4] = output[4] as i64 + output[5] as i64;
    step[5] = -output[5] as i64 + output[4] as i64;
    step[6] = -output[6] as i64 + output[7] as i64;
    step[7] = output[7] as i64 + output[6] as i64;
    step[8] = output[8] as i64;
    step[9] = dct_32_round(output[9] as i64 * -COSPI_8_64 + output[14] as i64 * COSPI_24_64) as i64;
    step[10] =
        dct_32_round(output[10] as i64 * -COSPI_24_64 + output[13] as i64 * -COSPI_8_64) as i64;
    step[11] = output[11] as i64;
    step[12] = output[12] as i64;
    step[13] =
        dct_32_round(output[13] as i64 * COSPI_24_64 + output[10] as i64 * -COSPI_8_64) as i64;
    step[14] = dct_32_round(output[14] as i64 * COSPI_8_64 + output[9] as i64 * COSPI_24_64) as i64;
    step[15] = output[15] as i64;

    step[16] = output[16] as i64 + output[19] as i64;
    step[17] = output[17] as i64 + output[18] as i64;
    step[18] = -output[18] as i64 + output[17] as i64;
    step[19] = -output[19] as i64 + output[16] as i64;
    step[20] = -output[20] as i64 + output[23] as i64;
    step[21] = -output[21] as i64 + output[22] as i64;
    step[22] = output[22] as i64 + output[21] as i64;
    step[23] = output[23] as i64 + output[20] as i64;
    step[24] = output[24] as i64 + output[27] as i64;
    step[25] = output[25] as i64 + output[26] as i64;
    step[26] = -output[26] as i64 + output[25] as i64;
    step[27] = -output[27] as i64 + output[24] as i64;
    step[28] = -output[28] as i64 + output[31] as i64;
    step[29] = -output[29] as i64 + output[30] as i64;
    step[30] = output[30] as i64 + output[29] as i64;
    step[31] = output[31] as i64 + output[28] as i64;

    // Stage 6
    output[0] = step[0] as i32;
    output[1] = step[1] as i32;
    output[2] = step[2] as i32;
    output[3] = step[3] as i32;
    output[4] = dct_32_round(step[4] * COSPI_28_64 + step[7] * COSPI_4_64);
    output[5] = dct_32_round(step[5] * COSPI_12_64 + step[6] * COSPI_20_64);
    output[6] = dct_32_round(step[6] * COSPI_12_64 + step[5] * -COSPI_20_64);
    output[7] = dct_32_round(step[7] * COSPI_28_64 + step[4] * -COSPI_4_64);
    output[8] = (step[8] + step[9]) as i32;
    output[9] = (-step[9] + step[8]) as i32;
    output[10] = (-step[10] + step[11]) as i32;
    output[11] = (step[11] + step[10]) as i32;
    output[12] = (step[12] + step[13]) as i32;
    output[13] = (-step[13] + step[12]) as i32;
    output[14] = (-step[14] + step[15]) as i32;
    output[15] = (step[15] + step[14]) as i32;

    output[16] = step[16] as i32;
    output[17] = dct_32_round(step[17] * -COSPI_4_64 + step[30] * COSPI_28_64);
    output[18] = dct_32_round(step[18] * -COSPI_28_64 + step[29] * -COSPI_4_64);
    output[19] = step[19] as i32;
    output[20] = step[20] as i32;
    output[21] = dct_32_round(step[21] * -COSPI_20_64 + step[26] * COSPI_12_64);
    output[22] = dct_32_round(step[22] * -COSPI_12_64 + step[25] * -COSPI_20_64);
    output[23] = step[23] as i32;
    output[24] = step[24] as i32;
    output[25] = dct_32_round(step[25] * COSPI_12_64 + step[22] * -COSPI_20_64);
    output[26] = dct_32_round(step[26] * COSPI_20_64 + step[21] * COSPI_12_64);
    output[27] = step[27] as i32;
    output[28] = step[28] as i32;
    output[29] = dct_32_round(step[29] * COSPI_28_64 + step[18] * -COSPI_4_64);
    output[30] = dct_32_round(step[30] * COSPI_4_64 + step[17] * COSPI_28_64);
    output[31] = step[31] as i32;

    // Stage 7 — reload step from output
    for i in 0..32 {
        step[i] = output[i] as i64;
    }
    step[0] = output[0] as i64;
    step[1] = output[1] as i64;
    step[2] = output[2] as i64;
    step[3] = output[3] as i64;
    step[4] = output[4] as i64;
    step[5] = output[5] as i64;
    step[6] = output[6] as i64;
    step[7] = output[7] as i64;
    step[8] = dct_32_round(output[8] as i64 * COSPI_30_64 + output[15] as i64 * COSPI_2_64) as i64;
    step[9] = dct_32_round(output[9] as i64 * COSPI_14_64 + output[14] as i64 * COSPI_18_64) as i64;
    step[10] =
        dct_32_round(output[10] as i64 * COSPI_22_64 + output[13] as i64 * COSPI_10_64) as i64;
    step[11] =
        dct_32_round(output[11] as i64 * COSPI_6_64 + output[12] as i64 * COSPI_26_64) as i64;
    step[12] =
        dct_32_round(output[12] as i64 * COSPI_6_64 + output[11] as i64 * -COSPI_26_64) as i64;
    step[13] =
        dct_32_round(output[13] as i64 * COSPI_22_64 + output[10] as i64 * -COSPI_10_64) as i64;
    step[14] =
        dct_32_round(output[14] as i64 * COSPI_14_64 + output[9] as i64 * -COSPI_18_64) as i64;
    step[15] =
        dct_32_round(output[15] as i64 * COSPI_30_64 + output[8] as i64 * -COSPI_2_64) as i64;

    step[16] = output[16] as i64 + output[17] as i64;
    step[17] = -output[17] as i64 + output[16] as i64;
    step[18] = -output[18] as i64 + output[19] as i64;
    step[19] = output[19] as i64 + output[18] as i64;
    step[20] = output[20] as i64 + output[21] as i64;
    step[21] = -output[21] as i64 + output[20] as i64;
    step[22] = -output[22] as i64 + output[23] as i64;
    step[23] = output[23] as i64 + output[22] as i64;
    step[24] = output[24] as i64 + output[25] as i64;
    step[25] = -output[25] as i64 + output[24] as i64;
    step[26] = -output[26] as i64 + output[27] as i64;
    step[27] = output[27] as i64 + output[26] as i64;
    step[28] = output[28] as i64 + output[29] as i64;
    step[29] = -output[29] as i64 + output[28] as i64;
    step[30] = -output[30] as i64 + output[31] as i64;
    step[31] = output[31] as i64 + output[30] as i64;

    // Final stage — bit-reversed output indices
    output[0] = step[0] as i32;
    output[16] = step[1] as i32;
    output[8] = step[2] as i32;
    output[24] = step[3] as i32;
    output[4] = step[4] as i32;
    output[20] = step[5] as i32;
    output[12] = step[6] as i32;
    output[28] = step[7] as i32;
    output[2] = step[8] as i32;
    output[18] = step[9] as i32;
    output[10] = step[10] as i32;
    output[26] = step[11] as i32;
    output[6] = step[12] as i32;
    output[22] = step[13] as i32;
    output[14] = step[14] as i32;
    output[30] = step[15] as i32;

    output[1] = dct_32_round(step[16] * COSPI_31_64 + step[31] * COSPI_1_64);
    output[17] = dct_32_round(step[17] * COSPI_15_64 + step[30] * COSPI_17_64);
    output[9] = dct_32_round(step[18] * COSPI_23_64 + step[29] * COSPI_9_64);
    output[25] = dct_32_round(step[19] * COSPI_7_64 + step[28] * COSPI_25_64);
    output[5] = dct_32_round(step[20] * COSPI_27_64 + step[27] * COSPI_5_64);
    output[21] = dct_32_round(step[21] * COSPI_11_64 + step[26] * COSPI_21_64);
    output[13] = dct_32_round(step[22] * COSPI_19_64 + step[25] * COSPI_13_64);
    output[29] = dct_32_round(step[23] * COSPI_3_64 + step[24] * COSPI_29_64);
    output[3] = dct_32_round(step[24] * COSPI_3_64 + step[23] * -COSPI_29_64);
    output[19] = dct_32_round(step[25] * COSPI_19_64 + step[22] * -COSPI_13_64);
    output[11] = dct_32_round(step[26] * COSPI_11_64 + step[21] * -COSPI_21_64);
    output[27] = dct_32_round(step[27] * COSPI_27_64 + step[20] * -COSPI_5_64);
    output[7] = dct_32_round(step[28] * COSPI_7_64 + step[19] * -COSPI_25_64);
    output[23] = dct_32_round(step[29] * COSPI_23_64 + step[18] * -COSPI_9_64);
    output[15] = dct_32_round(step[30] * COSPI_15_64 + step[17] * -COSPI_17_64);
    output[31] = dct_32_round(step[31] * COSPI_31_64 + step[16] * -COSPI_1_64);
}
/// 1-D IDCT32 (libvpx `idct32_c`).
fn idct32(input: &[i32; 32], output: &mut [i32; 32]) {
    let mut step1 = [0i32; 32];
    let mut step2 = [0i32; 32];
    // stage 1
    step1[0] = input[0] as i16 as i32;
    step1[1] = input[16] as i16 as i32;
    step1[2] = input[8] as i16 as i32;
    step1[3] = input[24] as i16 as i32;
    step1[4] = input[4] as i16 as i32;
    step1[5] = input[20] as i16 as i32;
    step1[6] = input[12] as i16 as i32;
    step1[7] = input[28] as i16 as i32;
    step1[8] = input[2] as i16 as i32;
    step1[9] = input[18] as i16 as i32;
    step1[10] = input[10] as i16 as i32;
    step1[11] = input[26] as i16 as i32;
    step1[12] = input[6] as i16 as i32;
    step1[13] = input[22] as i16 as i32;
    step1[14] = input[14] as i16 as i32;
    step1[15] = input[30] as i16 as i32;
    {
        let temp1 = input[1] as i16 as i64 * COSPI_31_64 - input[31] as i16 as i64 * COSPI_1_64;
        let temp2 = input[1] as i16 as i64 * COSPI_1_64 + input[31] as i16 as i64 * COSPI_31_64;
        step1[16] = wraplow(dct_const_round_shift(temp1));
        step1[31] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[17] as i16 as i64 * COSPI_15_64 - input[15] as i16 as i64 * COSPI_17_64;
        let temp2 = input[17] as i16 as i64 * COSPI_17_64 + input[15] as i16 as i64 * COSPI_15_64;
        step1[17] = wraplow(dct_const_round_shift(temp1));
        step1[30] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[9] as i16 as i64 * COSPI_23_64 - input[23] as i16 as i64 * COSPI_9_64;
        let temp2 = input[9] as i16 as i64 * COSPI_9_64 + input[23] as i16 as i64 * COSPI_23_64;
        step1[18] = wraplow(dct_const_round_shift(temp1));
        step1[29] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[25] as i16 as i64 * COSPI_7_64 - input[7] as i16 as i64 * COSPI_25_64;
        let temp2 = input[25] as i16 as i64 * COSPI_25_64 + input[7] as i16 as i64 * COSPI_7_64;
        step1[19] = wraplow(dct_const_round_shift(temp1));
        step1[28] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[5] as i16 as i64 * COSPI_27_64 - input[27] as i16 as i64 * COSPI_5_64;
        let temp2 = input[5] as i16 as i64 * COSPI_5_64 + input[27] as i16 as i64 * COSPI_27_64;
        step1[20] = wraplow(dct_const_round_shift(temp1));
        step1[27] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[21] as i16 as i64 * COSPI_11_64 - input[11] as i16 as i64 * COSPI_21_64;
        let temp2 = input[21] as i16 as i64 * COSPI_21_64 + input[11] as i16 as i64 * COSPI_11_64;
        step1[21] = wraplow(dct_const_round_shift(temp1));
        step1[26] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[13] as i16 as i64 * COSPI_19_64 - input[19] as i16 as i64 * COSPI_13_64;
        let temp2 = input[13] as i16 as i64 * COSPI_13_64 + input[19] as i16 as i64 * COSPI_19_64;
        step1[22] = wraplow(dct_const_round_shift(temp1));
        step1[25] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = input[29] as i16 as i64 * COSPI_3_64 - input[3] as i16 as i64 * COSPI_29_64;
        let temp2 = input[29] as i16 as i64 * COSPI_29_64 + input[3] as i16 as i64 * COSPI_3_64;
        step1[23] = wraplow(dct_const_round_shift(temp1));
        step1[24] = wraplow(dct_const_round_shift(temp2));
    }
    // stage 2
    step2[0] = step1[0];
    step2[1] = step1[1];
    step2[2] = step1[2];
    step2[3] = step1[3];
    step2[4] = step1[4];
    step2[5] = step1[5];
    step2[6] = step1[6];
    step2[7] = step1[7];
    {
        let temp1 = step1[8] as i64 * COSPI_30_64 - step1[15] as i64 * COSPI_2_64;
        let temp2 = step1[8] as i64 * COSPI_2_64 + step1[15] as i64 * COSPI_30_64;
        step2[8] = wraplow(dct_const_round_shift(temp1));
        step2[15] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = step1[9] as i64 * COSPI_14_64 - step1[14] as i64 * COSPI_18_64;
        let temp2 = step1[9] as i64 * COSPI_18_64 + step1[14] as i64 * COSPI_14_64;
        step2[9] = wraplow(dct_const_round_shift(temp1));
        step2[14] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = step1[10] as i64 * COSPI_22_64 - step1[13] as i64 * COSPI_10_64;
        let temp2 = step1[10] as i64 * COSPI_10_64 + step1[13] as i64 * COSPI_22_64;
        step2[10] = wraplow(dct_const_round_shift(temp1));
        step2[13] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = step1[11] as i64 * COSPI_6_64 - step1[12] as i64 * COSPI_26_64;
        let temp2 = step1[11] as i64 * COSPI_26_64 + step1[12] as i64 * COSPI_6_64;
        step2[11] = wraplow(dct_const_round_shift(temp1));
        step2[12] = wraplow(dct_const_round_shift(temp2));
    }
    step2[16] = wraplow(step1[16] + step1[17]);
    step2[17] = wraplow(step1[16] - step1[17]);
    step2[18] = wraplow(-step1[18] + step1[19]);
    step2[19] = wraplow(step1[18] + step1[19]);
    step2[20] = wraplow(step1[20] + step1[21]);
    step2[21] = wraplow(step1[20] - step1[21]);
    step2[22] = wraplow(-step1[22] + step1[23]);
    step2[23] = wraplow(step1[22] + step1[23]);
    step2[24] = wraplow(step1[24] + step1[25]);
    step2[25] = wraplow(step1[24] - step1[25]);
    step2[26] = wraplow(-step1[26] + step1[27]);
    step2[27] = wraplow(step1[26] + step1[27]);
    step2[28] = wraplow(step1[28] + step1[29]);
    step2[29] = wraplow(step1[28] - step1[29]);
    step2[30] = wraplow(-step1[30] + step1[31]);
    step2[31] = wraplow(step1[30] + step1[31]);
    // stage 3
    step1[0] = step2[0];
    step1[1] = step2[1];
    step1[2] = step2[2];
    step1[3] = step2[3];
    {
        let temp1 = step2[4] as i64 * COSPI_28_64 - step2[7] as i64 * COSPI_4_64;
        let temp2 = step2[4] as i64 * COSPI_4_64 + step2[7] as i64 * COSPI_28_64;
        step1[4] = wraplow(dct_const_round_shift(temp1));
        step1[7] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = step2[5] as i64 * COSPI_12_64 - step2[6] as i64 * COSPI_20_64;
        let temp2 = step2[5] as i64 * COSPI_20_64 + step2[6] as i64 * COSPI_12_64;
        step1[5] = wraplow(dct_const_round_shift(temp1));
        step1[6] = wraplow(dct_const_round_shift(temp2));
    }
    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);
    step1[16] = step2[16];
    step1[31] = step2[31];
    {
        let temp1 = -step2[17] as i64 * COSPI_4_64 + step2[30] as i64 * COSPI_28_64;
        let temp2 = step2[17] as i64 * COSPI_28_64 + step2[30] as i64 * COSPI_4_64;
        step1[17] = wraplow(dct_const_round_shift(temp1));
        step1[30] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step2[18] as i64 * COSPI_28_64 - step2[29] as i64 * COSPI_4_64;
        let temp2 = -step2[18] as i64 * COSPI_4_64 + step2[29] as i64 * COSPI_28_64;
        step1[18] = wraplow(dct_const_round_shift(temp1));
        step1[29] = wraplow(dct_const_round_shift(temp2));
    }
    step1[19] = step2[19];
    step1[20] = step2[20];
    {
        let temp1 = -step2[21] as i64 * COSPI_20_64 + step2[26] as i64 * COSPI_12_64;
        let temp2 = step2[21] as i64 * COSPI_12_64 + step2[26] as i64 * COSPI_20_64;
        step1[21] = wraplow(dct_const_round_shift(temp1));
        step1[26] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step2[22] as i64 * COSPI_12_64 - step2[25] as i64 * COSPI_20_64;
        let temp2 = -step2[22] as i64 * COSPI_20_64 + step2[25] as i64 * COSPI_12_64;
        step1[22] = wraplow(dct_const_round_shift(temp1));
        step1[25] = wraplow(dct_const_round_shift(temp2));
    }
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[27] = step2[27];
    step1[28] = step2[28];
    // stage 4
    {
        let temp1 = (step1[0] as i64 + step1[1] as i64) * COSPI_16_64;
        let temp2 = (step1[0] as i64 - step1[1] as i64) * COSPI_16_64;
        step2[0] = wraplow(dct_const_round_shift(temp1));
        step2[1] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = step1[2] as i64 * COSPI_24_64 - step1[3] as i64 * COSPI_8_64;
        let temp2 = step1[2] as i64 * COSPI_8_64 + step1[3] as i64 * COSPI_24_64;
        step2[2] = wraplow(dct_const_round_shift(temp1));
        step2[3] = wraplow(dct_const_round_shift(temp2));
    }
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);
    step2[8] = step1[8];
    step2[15] = step1[15];
    {
        let temp1 = -step1[9] as i64 * COSPI_8_64 + step1[14] as i64 * COSPI_24_64;
        let temp2 = step1[9] as i64 * COSPI_24_64 + step1[14] as i64 * COSPI_8_64;
        step2[9] = wraplow(dct_const_round_shift(temp1));
        step2[14] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step1[10] as i64 * COSPI_24_64 - step1[13] as i64 * COSPI_8_64;
        let temp2 = -step1[10] as i64 * COSPI_8_64 + step1[13] as i64 * COSPI_24_64;
        step2[10] = wraplow(dct_const_round_shift(temp1));
        step2[13] = wraplow(dct_const_round_shift(temp2));
    }
    step2[11] = step1[11];
    step2[12] = step1[12];
    step2[16] = wraplow(step1[16] + step1[19]);
    step2[17] = wraplow(step1[17] + step1[18]);
    step2[18] = wraplow(step1[17] - step1[18]);
    step2[19] = wraplow(step1[16] - step1[19]);
    step2[20] = wraplow(-step1[20] + step1[23]);
    step2[21] = wraplow(-step1[21] + step1[22]);
    step2[22] = wraplow(step1[21] + step1[22]);
    step2[23] = wraplow(step1[20] + step1[23]);
    step2[24] = wraplow(step1[24] + step1[27]);
    step2[25] = wraplow(step1[25] + step1[26]);
    step2[26] = wraplow(step1[25] - step1[26]);
    step2[27] = wraplow(step1[24] - step1[27]);
    step2[28] = wraplow(-step1[28] + step1[31]);
    step2[29] = wraplow(-step1[29] + step1[30]);
    step2[30] = wraplow(step1[29] + step1[30]);
    step2[31] = wraplow(step1[28] + step1[31]);
    // stage 5
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    {
        let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64;
        let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64;
        step1[5] = wraplow(dct_const_round_shift(temp1));
        step1[6] = wraplow(dct_const_round_shift(temp2));
    }
    step1[7] = step2[7];
    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);
    step1[16] = step2[16];
    step1[17] = step2[17];
    {
        let temp1 = -step2[18] as i64 * COSPI_8_64 + step2[29] as i64 * COSPI_24_64;
        let temp2 = step2[18] as i64 * COSPI_24_64 + step2[29] as i64 * COSPI_8_64;
        step1[18] = wraplow(dct_const_round_shift(temp1));
        step1[29] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step2[19] as i64 * COSPI_8_64 + step2[28] as i64 * COSPI_24_64;
        let temp2 = step2[19] as i64 * COSPI_24_64 + step2[28] as i64 * COSPI_8_64;
        step1[19] = wraplow(dct_const_round_shift(temp1));
        step1[28] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step2[20] as i64 * COSPI_24_64 - step2[27] as i64 * COSPI_8_64;
        let temp2 = -step2[20] as i64 * COSPI_8_64 + step2[27] as i64 * COSPI_24_64;
        step1[20] = wraplow(dct_const_round_shift(temp1));
        step1[27] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = -step2[21] as i64 * COSPI_24_64 - step2[26] as i64 * COSPI_8_64;
        let temp2 = -step2[21] as i64 * COSPI_8_64 + step2[26] as i64 * COSPI_24_64;
        step1[21] = wraplow(dct_const_round_shift(temp1));
        step1[26] = wraplow(dct_const_round_shift(temp2));
    }
    step1[22] = step2[22];
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[25] = step2[25];
    step1[30] = step2[30];
    step1[31] = step2[31];
    // stage 6
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    {
        let temp1 = (-step1[10] as i64 + step1[13] as i64) * COSPI_16_64;
        let temp2 = (step1[10] as i64 + step1[13] as i64) * COSPI_16_64;
        step2[10] = wraplow(dct_const_round_shift(temp1));
        step2[13] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = (-step1[11] as i64 + step1[12] as i64) * COSPI_16_64;
        let temp2 = (step1[11] as i64 + step1[12] as i64) * COSPI_16_64;
        step2[11] = wraplow(dct_const_round_shift(temp1));
        step2[12] = wraplow(dct_const_round_shift(temp2));
    }
    step2[14] = step1[14];
    step2[15] = step1[15];
    step2[16] = wraplow(step1[16] + step1[23]);
    step2[17] = wraplow(step1[17] + step1[22]);
    step2[18] = wraplow(step1[18] + step1[21]);
    step2[19] = wraplow(step1[19] + step1[20]);
    step2[20] = wraplow(step1[19] - step1[20]);
    step2[21] = wraplow(step1[18] - step1[21]);
    step2[22] = wraplow(step1[17] - step1[22]);
    step2[23] = wraplow(step1[16] - step1[23]);
    step2[24] = wraplow(-step1[24] + step1[31]);
    step2[25] = wraplow(-step1[25] + step1[30]);
    step2[26] = wraplow(-step1[26] + step1[29]);
    step2[27] = wraplow(-step1[27] + step1[28]);
    step2[28] = wraplow(step1[27] + step1[28]);
    step2[29] = wraplow(step1[26] + step1[29]);
    step2[30] = wraplow(step1[25] + step1[30]);
    step2[31] = wraplow(step1[24] + step1[31]);
    // stage 7
    step1[0] = wraplow(step2[0] + step2[15]);
    step1[1] = wraplow(step2[1] + step2[14]);
    step1[2] = wraplow(step2[2] + step2[13]);
    step1[3] = wraplow(step2[3] + step2[12]);
    step1[4] = wraplow(step2[4] + step2[11]);
    step1[5] = wraplow(step2[5] + step2[10]);
    step1[6] = wraplow(step2[6] + step2[9]);
    step1[7] = wraplow(step2[7] + step2[8]);
    step1[8] = wraplow(step2[7] - step2[8]);
    step1[9] = wraplow(step2[6] - step2[9]);
    step1[10] = wraplow(step2[5] - step2[10]);
    step1[11] = wraplow(step2[4] - step2[11]);
    step1[12] = wraplow(step2[3] - step2[12]);
    step1[13] = wraplow(step2[2] - step2[13]);
    step1[14] = wraplow(step2[1] - step2[14]);
    step1[15] = wraplow(step2[0] - step2[15]);
    step1[16] = step2[16];
    step1[17] = step2[17];
    step1[18] = step2[18];
    step1[19] = step2[19];
    {
        let temp1 = (-step2[20] as i64 + step2[27] as i64) * COSPI_16_64;
        let temp2 = (step2[20] as i64 + step2[27] as i64) * COSPI_16_64;
        step1[20] = wraplow(dct_const_round_shift(temp1));
        step1[27] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = (-step2[21] as i64 + step2[26] as i64) * COSPI_16_64;
        let temp2 = (step2[21] as i64 + step2[26] as i64) * COSPI_16_64;
        step1[21] = wraplow(dct_const_round_shift(temp1));
        step1[26] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = (-step2[22] as i64 + step2[25] as i64) * COSPI_16_64;
        let temp2 = (step2[22] as i64 + step2[25] as i64) * COSPI_16_64;
        step1[22] = wraplow(dct_const_round_shift(temp1));
        step1[25] = wraplow(dct_const_round_shift(temp2));
    }
    {
        let temp1 = (-step2[23] as i64 + step2[24] as i64) * COSPI_16_64;
        let temp2 = (step2[23] as i64 + step2[24] as i64) * COSPI_16_64;
        step1[23] = wraplow(dct_const_round_shift(temp1));
        step1[24] = wraplow(dct_const_round_shift(temp2));
    }
    step1[28] = step2[28];
    step1[29] = step2[29];
    step1[30] = step2[30];
    step1[31] = step2[31];
    // final stage
    output[0] = wraplow(step1[0] + step1[31]);
    output[1] = wraplow(step1[1] + step1[30]);
    output[2] = wraplow(step1[2] + step1[29]);
    output[3] = wraplow(step1[3] + step1[28]);
    output[4] = wraplow(step1[4] + step1[27]);
    output[5] = wraplow(step1[5] + step1[26]);
    output[6] = wraplow(step1[6] + step1[25]);
    output[7] = wraplow(step1[7] + step1[24]);
    output[8] = wraplow(step1[8] + step1[23]);
    output[9] = wraplow(step1[9] + step1[22]);
    output[10] = wraplow(step1[10] + step1[21]);
    output[11] = wraplow(step1[11] + step1[20]);
    output[12] = wraplow(step1[12] + step1[19]);
    output[13] = wraplow(step1[13] + step1[18]);
    output[14] = wraplow(step1[14] + step1[17]);
    output[15] = wraplow(step1[15] + step1[16]);
    output[16] = wraplow(step1[15] - step1[16]);
    output[17] = wraplow(step1[14] - step1[17]);
    output[18] = wraplow(step1[13] - step1[18]);
    output[19] = wraplow(step1[12] - step1[19]);
    output[20] = wraplow(step1[11] - step1[20]);
    output[21] = wraplow(step1[10] - step1[21]);
    output[22] = wraplow(step1[9] - step1[22]);
    output[23] = wraplow(step1[8] - step1[23]);
    output[24] = wraplow(step1[7] - step1[24]);
    output[25] = wraplow(step1[6] - step1[25]);
    output[26] = wraplow(step1[5] - step1[26]);
    output[27] = wraplow(step1[4] - step1[27]);
    output[28] = wraplow(step1[3] - step1[28]);
    output[29] = wraplow(step1[2] - step1[29]);
    output[30] = wraplow(step1[1] - step1[30]);
    output[31] = wraplow(step1[0] - step1[31]);
}
/// Forward 32×32 DCT (`vpx_fdct32x32_c`). `input` is row-major residual (stride 32).
pub fn fdct32x32(input: &[i16; 1024], output: &mut [i32; 1024]) {
    let mut intermediate = [0i32; 1024];
    // Columns
    for i in 0..32 {
        let mut temp_in = [0i64; 32];
        for j in 0..32 {
            temp_in[j] = input[j * 32 + i] as i64 * 4;
        }
        let mut temp_out = [0i32; 32];
        fdct32(&temp_in, &mut temp_out, false);
        for j in 0..32 {
            let v = temp_out[j];
            intermediate[j * 32 + i] = (v + 1 + i32::from(v > 0)) >> 2;
        }
    }
    // Rows
    for i in 0..32 {
        let mut temp_in = [0i64; 32];
        for j in 0..32 {
            temp_in[j] = intermediate[j + i * 32] as i64;
        }
        let mut temp_out = [0i32; 32];
        fdct32(&temp_in, &mut temp_out, false);
        for j in 0..32 {
            let v = temp_out[j];
            output[j + i * 32] = (v + 1 + i32::from(v < 0)) >> 2;
        }
    }
}

/// Inverse 32×32 DCT and add into `dest` (`vpx_idct32x32_1024_add_c`).
pub fn idct32x32_add(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    assert!(dest.len() >= 31 * stride + 32);
    let mut out = [0i32; 1024];
    for i in 0..32 {
        let row = &input[i * 32..i * 32 + 32];
        if row.iter().any(|&x| x != 0) {
            let row_in: [i32; 32] = row.try_into().unwrap();
            let mut row_out = [0i32; 32];
            idct32(&row_in, &mut row_out);
            out[i * 32..i * 32 + 32].copy_from_slice(&row_out);
        } else {
            out[i * 32..i * 32 + 32].fill(0);
        }
    }
    for i in 0..32 {
        let mut col_in = [0i32; 32];
        for j in 0..32 {
            col_in[j] = out[j * 32 + i];
        }
        let mut col_out = [0i32; 32];
        idct32(&col_in, &mut col_out);
        for j in 0..32 {
            let add = round_power_of_two(col_out[j] as i64, 6) as i32;
            let idx = j * stride + i;
            dest[idx] = clip_pixel_add(dest[idx], add);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    #[test]
    fn fdct_idct_roundtrip_small_residuals() {
        let mut rng = Rng(0xC0FF_EE42);
        for _ in 0..200 {
            let mut residual = [0i16; 16];
            for r in residual.iter_mut() {
                *r = (rng.next() % 31) as i16 - 15;
            }
            let mut coeffs = [0i32; 16];
            fdct4x4(&residual, &mut coeffs);

            let mut recon = [0u8; 16];
            recon.fill(128);
            idct4x4_add(&coeffs, &mut recon, 4);

            for i in 0..16 {
                let got = recon[i] as i32 - 128;
                let exp = residual[i] as i32;
                assert!(
                    (got - exp).abs() <= 2,
                    "pix {i}: got {got} expected {exp} (residual {:?}, coeffs {:?})",
                    residual,
                    coeffs
                );
            }
        }
    }

    #[test]
    fn hybrid_roundtrip_all_tx_types() {
        let mut rng = Rng(0xAD57_4444);
        for &tx in &[
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            for _ in 0..80 {
                let mut residual = [0i16; 16];
                for r in residual.iter_mut() {
                    *r = (rng.next() % 21) as i16 - 10;
                }
                let mut coeffs = [0i32; 16];
                fht4x4(&residual, &mut coeffs, tx);
                let mut recon = [128u8; 16];
                iht4x4_add(&coeffs, &mut recon, 4, tx);
                for i in 0..16 {
                    let got = recon[i] as i32 - 128;
                    let exp = residual[i] as i32;
                    assert!(
                        (got - exp).abs() <= 3,
                        "tx={tx:?} pix {i}: got {got} expected {exp}"
                    );
                }
            }
        }
    }

    #[test]
    fn mode_to_tx_type_matches_libvpx() {
        assert_eq!(tx_type_from_mode(DC_PRED), TxType::DctDct);
        assert_eq!(tx_type_from_mode(V_PRED), TxType::AdstDct);
        assert_eq!(tx_type_from_mode(H_PRED), TxType::DctAdst);
    }

    #[test]
    fn fdct_zero_is_zero() {
        let input = [0i16; 16];
        let mut out = [1i32; 16];
        fdct4x4(&input, &mut out);
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn fdct8_idct8_roundtrip_small_residuals() {
        let mut residual = [0i16; 64];
        for i in 0..64 {
            residual[i] = ((i as i16 * 3) % 17) - 8;
        }
        let mut coeffs = [0i32; 64];
        fdct8x8(&residual, &mut coeffs);
        let mut dest = [128u8; 64];
        idct8x8_add(&coeffs, &mut dest, 8);
        for i in 0..64 {
            let exp = (128 + residual[i] as i32).clamp(0, 255) as u8;
            assert!(
                dest[i].abs_diff(exp) <= 2,
                "pix {i}: got {} expected {exp}",
                dest[i]
            );
        }
    }

    #[test]
    fn fdct8_zero_is_zero() {
        let input = [0i16; 64];
        let mut out = [1i32; 64];
        fdct8x8(&input, &mut out);
        assert_eq!(out, [0i32; 64]);
    }

    #[test]
    fn fdct16_zero_is_zero() {
        let input = [0i16; 256];
        let mut out = [1i32; 256];
        fdct16x16(&input, &mut out);
        assert_eq!(out, [0i32; 256]);
    }

    #[test]
    fn fdct16_idct16_roundtrip_small_residuals() {
        let mut rng = Rng(0x16_16_BEEF);
        for _ in 0..100 {
            let mut residual = [0i16; 256];
            for r in residual.iter_mut() {
                *r = (rng.next() % 31) as i16 - 15;
            }
            let mut coeffs = [0i32; 256];
            fdct16x16(&residual, &mut coeffs);
            let mut dest = [128u8; 256];
            idct16x16_add(&coeffs, &mut dest, 16);
            for i in 0..256 {
                let exp = (128 + residual[i] as i32).clamp(0, 255) as u8;
                assert!(
                    dest[i].abs_diff(exp) <= 2,
                    "pix {i}: got {} expected {exp}",
                    dest[i]
                );
            }
        }
    }

    #[test]
    fn idct_add_onto_nonzero_base() {
        let residual = [4i16; 16];
        let mut coeffs = [0i32; 16];
        fdct4x4(&residual, &mut coeffs);
        let mut dest = [10u8; 16];
        idct4x4_add(&coeffs, &mut dest, 4);
        for &p in &dest {
            assert!((p as i32 - 14).abs() <= 2, "pixel {p}");
        }
    }

    #[test]
    fn fdct32_zero_is_zero() {
        let input = [0i16; 1024];
        let mut out = [1i32; 1024];
        fdct32x32(&input, &mut out);
        assert_eq!(out, [0i32; 1024]);
    }

    #[test]
    fn fdct32_idct32_roundtrip_small_residuals() {
        let mut rng = Rng(0x32_32_CAFE);
        for _ in 0..100 {
            let mut residual = [0i16; 1024];
            for r in residual.iter_mut() {
                *r = (rng.next() % 21) as i16 - 10;
            }
            let mut coeffs = [0i32; 1024];
            fdct32x32(&residual, &mut coeffs);
            let mut dest = [128u8; 1024];
            idct32x32_add(&coeffs, &mut dest, 32);
            for i in 0..1024 {
                let exp = (128 + residual[i] as i32).clamp(0, 255) as u8;
                assert!(
                    dest[i].abs_diff(exp) <= 2,
                    "pix {i}: got {} expected {exp}",
                    dest[i]
                );
            }
        }
    }

    #[test]
    fn wraplow_sign_extends_16() {
        assert_eq!(wraplow(0x0001_0001), 1);
        assert_eq!(wraplow(0x0000_FFFF_u32 as i32), -1);
        assert_eq!(wraplow(32768), -32768);
    }
}
