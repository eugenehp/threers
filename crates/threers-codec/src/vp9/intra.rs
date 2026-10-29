//! VP9 intra prediction (8-bit): DC / V / H for the prediction-only encoder.
//!
//! Neighbor construction follows the bitstream spec §8.5.1: unavailable above
//! samples are `127`, unavailable left samples are `129`. DC uses the
//! availability flags directly (libvpx `dc_pred[left][above]`).

use crate::vp9::tables::{DC_PRED, H_PRED, V_PRED};

/// Fill `dst` (row-major `bs × bs`) with the intra prediction for `mode`.
///
/// `above` is `bs` samples immediately above the block (or the 127-fill when
/// `have_above` is false). `left` is `bs` samples immediately left (or 129-fill
/// when `have_left` is false).
pub fn predict(
    mode: i8,
    bs: usize,
    have_above: bool,
    have_left: bool,
    above: &[u8],
    left: &[u8],
    dst: &mut [u8],
) {
    assert_eq!(above.len(), bs);
    assert_eq!(left.len(), bs);
    assert_eq!(dst.len(), bs * bs);
    match mode {
        DC_PRED => dc(bs, have_above, have_left, above, left, dst),
        V_PRED => {
            for r in 0..bs {
                dst[r * bs..(r + 1) * bs].copy_from_slice(above);
            }
        }
        H_PRED => {
            for r in 0..bs {
                dst[r * bs..(r + 1) * bs].fill(left[r]);
            }
        }
        _ => panic!("prediction-only encoder supports DC/V/H only, got {mode}"),
    }
}

fn dc(bs: usize, have_above: bool, have_left: bool, above: &[u8], left: &[u8], dst: &mut [u8]) {
    let val = match (have_left, have_above) {
        (true, true) => {
            let sum: u32 = above.iter().map(|&x| x as u32).sum::<u32>()
                + left.iter().map(|&x| x as u32).sum::<u32>();
            let count = (2 * bs) as u32;
            ((sum + (count / 2)) / count) as u8
        }
        (true, false) => {
            let sum: u32 = left.iter().map(|&x| x as u32).sum();
            ((sum + (bs as u32 / 2)) / bs as u32) as u8
        }
        (false, true) => {
            let sum: u32 = above.iter().map(|&x| x as u32).sum();
            ((sum + (bs as u32 / 2)) / bs as u32) as u8
        }
        (false, false) => 128,
    };
    dst.fill(val);
}

/// Sum of absolute differences between `src` (stride `src_stride`) and a
/// tightly packed `bs × bs` prediction in `pred`.
pub fn sad(src: &[u8], src_stride: usize, pred: &[u8], bs: usize) -> u32 {
    let mut s = 0u32;
    for r in 0..bs {
        for c in 0..bs {
            s += src[r * src_stride + c].abs_diff(pred[r * bs + c]) as u32;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_neither_is_mid_gray() {
        let mut dst = [0u8; 64];
        predict(DC_PRED, 8, false, false, &[0; 8], &[0; 8], &mut dst);
        assert!(dst.iter().all(|&x| x == 128));
    }

    #[test]
    fn v_copies_above() {
        let above = [1u8, 2, 3, 4];
        let mut dst = [0u8; 16];
        predict(V_PRED, 4, true, false, &above, &[129; 4], &mut dst);
        assert_eq!(&dst[..4], &above);
        assert_eq!(&dst[4..8], &above);
    }

    #[test]
    fn h_copies_left() {
        let left = [10u8, 20, 30, 40];
        let mut dst = [0u8; 16];
        predict(H_PRED, 4, false, true, &[127; 4], &left, &mut dst);
        assert_eq!(&dst[0..4], &[10, 10, 10, 10]);
        assert_eq!(&dst[12..16], &[40, 40, 40, 40]);
    }
}
