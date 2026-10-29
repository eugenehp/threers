//! Quantizer matrices built from the two coded tables (dav1d `src/qm.c`,
//! `dav1d_init_qm_tables`). Stored transposed, like dav1d.

use std::sync::OnceLock;

use super::levels::*;
use super::tables::{QM_TBL_32X16, QM_TBL_32X32_T};

fn subsample(src: &[u8], o: usize, h: usize, hstep: usize, vstep: usize) -> Vec<u8> {
    let mut d = Vec::new();
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < 32 {
            d.push(src[o + y * 32 + x]);
            x += hstep;
        }
        y += vstep;
    }
    d
}

fn transpose(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut d = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            d[x * h + y] = src[y * w + x];
        }
    }
    d
}

fn untriangle(src: &[u8], sz: usize) -> Vec<u8> {
    let mut d = vec![0u8; sz * sz];
    let mut so = 0;
    for y in 0..sz {
        d[y * sz..y * sz + y + 1].copy_from_slice(&src[so..so + y + 1]);
        let mut sp = so + y;
        for x in y + 1..sz {
            sp += x;
            d[y * sz + x] = src[sp];
        }
        so += y + 1;
    }
    d
}

/// `dav1d_qm_tbl[level][chroma][tx]` for level < 15.
pub fn table(level: usize, chroma: usize, tx: usize) -> &'static [u8] {
    static T: OnceLock<Vec<Vec<Vec<Vec<u8>>>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        (0..15)
            .map(|i| {
                (0..2)
                    .map(|j| {
                        let t32x32 = untriangle(&QM_TBL_32X32_T[i][j], 32);
                        let t32x16 = &QM_TBL_32X16[i][j][..];
                        let t4x4 = subsample(&t32x32, 32 * 3 + 3, 32, 8, 8);
                        let t8x4 = subsample(t32x16, 32 + 1, 16, 4, 4);
                        let t8x8 = subsample(&t32x32, 32 + 1, 32, 4, 4);
                        let t16x4 = subsample(t32x16, 32, 16, 2, 4);
                        let t16x8 = subsample(t32x16, 0, 16, 2, 2);
                        let t16x16 = subsample(&t32x32, 0, 32, 2, 2);
                        let t32x8 = subsample(t32x16, 0, 16, 1, 2);
                        let t4x8 = transpose(&t8x4, 8, 4);
                        let t4x16 = transpose(&t16x4, 16, 4);
                        let t8x16 = transpose(&t16x8, 16, 8);
                        let t8x32 = transpose(&t32x8, 32, 8);
                        let t16x32 = transpose(t32x16, 32, 16);
                        let mut v: Vec<Vec<u8>> = vec![Vec::new(); N_RECT_TX_SIZES];
                        // w/h swapped on purpose: coefficients are stored transposed.
                        v[RTX_4X8] = t8x4.clone();
                        v[RTX_8X4] = t4x8;
                        v[RTX_4X16] = t16x4.clone();
                        v[RTX_16X4] = t4x16;
                        v[RTX_8X16] = t16x8.clone();
                        v[RTX_16X8] = t8x16;
                        v[RTX_8X32] = t32x8.clone();
                        v[RTX_32X8] = t8x32;
                        v[RTX_16X32] = t32x16.to_vec();
                        v[RTX_32X16] = t16x32;
                        v[TX_4X4] = t4x4;
                        v[TX_8X8] = t8x8;
                        v[TX_16X16] = t16x16;
                        v[TX_32X32] = t32x32;
                        v[TX_64X64] = v[TX_32X32].clone();
                        v[RTX_64X32] = v[TX_32X32].clone();
                        v[RTX_64X16] = v[RTX_32X16].clone();
                        v[RTX_32X64] = v[TX_32X32].clone();
                        v[RTX_16X64] = v[RTX_16X32].clone();
                        v
                    })
                    .collect()
            })
            .collect()
    });
    &t[level][chroma][tx]
}
