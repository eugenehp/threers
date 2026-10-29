//! Inverse transform + add (dav1d `src/itx_tmpl.c`).

use std::sync::OnceLock;

use super::itx_1d::{tx1d, wht4, IDENTITY, TX1D_TYPES};
use super::levels::*;
use super::tables::{SCANS, TXFM_DIMENSIONS};

/// `itxfm_add[tx][txtp]` calls the function whose internal type is the
/// transpose of `txtp` (dav1d names its functions by first/second pass).
fn fn_type(txtp: u8) -> u8 {
    match txtp {
        DCT_ADST => ADST_DCT,
        ADST_DCT => DCT_ADST,
        ADST_FLIPADST => FLIPADST_ADST,
        FLIPADST_ADST => ADST_FLIPADST,
        DCT_FLIPADST => FLIPADST_DCT,
        FLIPADST_DCT => DCT_FLIPADST,
        H_DCT => V_DCT,
        V_DCT => H_DCT,
        H_FLIPADST => V_FLIPADST,
        V_FLIPADST => H_FLIPADST,
        H_ADST => V_ADST,
        V_ADST => H_ADST,
        t => t,
    }
}

/// Row shift of each `inv_txfm_fn*` instantiation, by `RectTxfmSize`.
fn shift(tx: usize) -> i32 {
    match tx {
        TX_4X4 | RTX_4X8 | RTX_8X4 => 0,
        RTX_4X16 | TX_8X8 | RTX_8X16 | RTX_16X4 | RTX_16X8 | RTX_16X32 | RTX_32X16 | RTX_32X64 | RTX_64X32 => 1,
        _ => 2,
    }
}

/// `dav1d_last_nonzero_col_from_eob[tx]` (scan.c `init_tbl`).
fn last_nonzero_col(tx: usize) -> &'static [u8] {
    static T: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        (0..N_RECT_TX_SIZES)
            .map(|tx| {
                let d = TXFM_DIMENSIONS[tx];
                let w = ((d.w as usize) * 4).min(32);
                let h = ((d.h as usize) * 4).min(32);
                let scan = SCANS[tx];
                let mut out = vec![0u8; w * h];
                let mut max_col = 0usize;
                for (n, o) in out.iter_mut().enumerate() {
                    let rc = scan[n] as usize;
                    max_col = max_col.max(rc & (h - 1));
                    *o = max_col as u8;
                }
                out
            })
            .collect()
    });
    &t[tx]
}

/// `inv_txfm_add_c`: add the inverse transform of `coeff` (which is zeroed)
/// to `dst[off..]`.
#[allow(clippy::too_many_arguments)]
pub fn itxfm_add(dst: &mut [u16], off: usize, stride: usize, coeff: &mut [i32], eob: i32, tx: usize, txtp: u8, bitdepth_max: i32) {
    if txtp == WHT_WHT {
        return wht_add(dst, off, stride, coeff, bitdepth_max);
    }
    let t_dim = TXFM_DIMENSIONS[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let typ = fn_type(txtp);
    let has_dconly = typ == DCT_DCT;
    let is_rect2 = w * 2 == h || h * 2 == w;
    let sh = shift(tx);
    let rnd = (1 << sh) >> 1;
    let clip = |v: i32| v.clamp(0, bitdepth_max) as u16;
    if eob < has_dconly as i32 {
        let mut dc = coeff[0];
        coeff[0] = 0;
        if is_rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc * 181 + 128) >> 8;
        dc = (dc + rnd) >> sh;
        dc = (dc * 181 + 128 + 2048) >> 12;
        for y in 0..h {
            for x in 0..w {
                let d = &mut dst[off + y * stride + x];
                *d = clip(*d as i32 + dc);
            }
        }
        return;
    }
    let txtps = TX1D_TYPES[typ as usize];
    let shh = h.min(32);
    let sw = w.min(32);
    let (row_clip_min, col_clip_min) = if bitdepth_max == 255 {
        (i16::MIN as i32, i16::MIN as i32)
    } else {
        ((!(bitdepth_max as u32) << 7) as i32, (!(bitdepth_max as u32) << 5) as i32)
    };
    let row_clip_max = !row_clip_min;
    let col_clip_max = !col_clip_min;
    let mut tmp = vec![0i32; w * h];
    let last_nonzero_col = if txtps[1] == IDENTITY && txtps[0] != IDENTITY {
        (shh as i32 - 1).min(eob) as usize
    } else if txtps[0] == IDENTITY && txtps[1] != IDENTITY {
        (eob >> (t_dim.lw + 2)) as usize
    } else {
        last_nonzero_col(tx)[eob as usize] as usize
    };
    for y in 0..=last_nonzero_col {
        let c = y * w;
        for x in 0..sw {
            let v = coeff[y + x * shh];
            tmp[c + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        tx1d(t_dim.lw as usize, txtps[0], &mut tmp, c as isize, 1, row_clip_min, row_clip_max);
    }
    for v in coeff.iter_mut().take(sw * shh) {
        *v = 0;
    }
    for v in tmp.iter_mut().take(w * shh) {
        *v = ((*v + rnd) >> sh).clamp(col_clip_min, col_clip_max);
    }
    for x in 0..w {
        tx1d(t_dim.lh as usize, txtps[1], &mut tmp, x as isize, w as isize, col_clip_min, col_clip_max);
    }
    for y in 0..h {
        for x in 0..w {
            let d = &mut dst[off + y * stride + x];
            *d = clip(*d as i32 + ((tmp[y * w + x] + 8) >> 4));
        }
    }
}

fn wht_add(dst: &mut [u16], off: usize, stride: usize, coeff: &mut [i32], bitdepth_max: i32) {
    let mut tmp = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            tmp[y * 4 + x] = coeff[y + x * 4] >> 2;
        }
        wht4(&mut tmp, (y * 4) as isize, 1);
    }
    for v in coeff.iter_mut().take(16) {
        *v = 0;
    }
    for x in 0..4 {
        wht4(&mut tmp, x as isize, 4);
    }
    for y in 0..4 {
        for x in 0..4 {
            let d = &mut dst[off + y * stride + x];
            *d = (*d as i32 + tmp[y * 4 + x]).clamp(0, bitdepth_max) as u16;
        }
    }
}
