//! Per-block motion estimation for VP9 inter (SAD, even Q3 MVs).
//!
//! Full-pel rect search first, then a half-pel refine around the best candidate.
//! Only even 1/8-pel values are considered (`allow_hp = 0`).

use crate::vp9::mc::{predict_inter, InterpFilter};
use crate::vp9::mv::Mv;

/// Default full-pel search radius (±pels).
pub const DEFAULT_RANGE_PEL: i16 = 8;

/// Pick an even-Q3 MV for an 8×8 luma block.
pub fn search_mv_8x8(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    range_pel: i16,
) -> Mv {
    search_mv(
        src, src_stride, reference, ref_stride, ref_w, ref_h, x, y, 8, 8, range_pel,
    )
}

/// Pick an even-Q3 MV for a `bw×bh` luma block (typically 8 or 16).
pub fn search_mv(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    range_pel: i16,
) -> Mv {
    let mut best = Mv::default();
    let mut best_sad = sad_fullpel(
        src, src_stride, reference, ref_stride, ref_w, ref_h, x, y, bw, bh, 0, 0,
    );

    let range = range_pel.max(0);
    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sad = sad_fullpel(
                src, src_stride, reference, ref_stride, ref_w, ref_h, x, y, bw, bh, dy, dx,
            );
            let cand = Mv {
                row: dy.saturating_mul(8),
                col: dx.saturating_mul(8),
            };
            if sad < best_sad || (sad == best_sad && mv_prefer(cand, best)) {
                best_sad = sad;
                best = cand;
            }
        }
    }

    let base = best;
    for &(dr, dc) in &[
        (0i16, 0i16),
        (0, 4),
        (0, -4),
        (4, 0),
        (-4, 0),
        (4, 4),
        (4, -4),
        (-4, 4),
        (-4, -4),
    ] {
        let cand = Mv {
            row: base.row.saturating_add(dr),
            col: base.col.saturating_add(dc),
        };
        if cand.row % 2 != 0 || cand.col % 2 != 0 || cand == best {
            continue;
        }
        let sad = sad_mv(
            src, src_stride, reference, ref_stride, ref_w, ref_h, x, y, bw, bh, cand,
        );
        if sad < best_sad || (sad == best_sad && mv_prefer(cand, best)) {
            best_sad = sad;
            best = cand;
        }
    }

    best
}

/// SAD of a `bw×bh` block under `mv` (EIGHTTAP when subpel).
pub fn sad_mv(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    mv: Mv,
) -> u32 {
    let mut pred = vec![0u8; bw * bh];
    predict_inter(
        reference,
        ref_stride,
        ref_w,
        ref_h,
        x,
        y,
        bw,
        bh,
        mv.row,
        mv.col,
        false,
        InterpFilter::EightTap,
        &mut pred,
    );
    let mut sad = 0u32;
    for r in 0..bh {
        for c in 0..bw {
            let s = src[(y + r) * src_stride + x + c];
            sad += u32::from(s.abs_diff(pred[r * bw + c]));
        }
    }
    sad
}

fn mv_prefer(a: Mv, b: Mv) -> bool {
    let da = i32::from(a.row).abs() + i32::from(a.col).abs();
    let db = i32::from(b.row).abs() + i32::from(b.col).abs();
    da < db || (da == db && (a.row, a.col) < (b.row, b.col))
}

fn sad_fullpel(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    dy_pel: i16,
    dx_pel: i16,
) -> u32 {
    let mut sad = 0u32;
    for r in 0..bh {
        for c in 0..bw {
            let sy = (y as i32 + r as i32 + i32::from(dy_pel)).clamp(0, ref_h as i32 - 1) as usize;
            let sx = (x as i32 + c as i32 + i32::from(dx_pel)).clamp(0, ref_w as i32 - 1) as usize;
            let s = src[(y + r) * src_stride + x + c];
            let p = reference[sy * ref_stride + sx];
            sad += u32::from(s.abs_diff(p));
        }
    }
    sad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_horizontal_shift() {
        let mut plane = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                plane[j * 64 + i] = ((i * 3 + j) & 0xff) as u8;
            }
        }
        let mut src = plane.clone();
        for j in 16..24 {
            for i in 16..24 {
                src[j * 64 + i] = plane[j * 64 + i + 2];
            }
        }
        let mv = search_mv_8x8(&src, 64, &plane, 64, 64, 64, 16, 16, 4);
        assert_eq!(mv, Mv { row: 0, col: 16 });
    }

    #[test]
    fn finds_16x16_shift() {
        let mut plane = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                plane[j * 64 + i] = ((i * 5 + j * 2) & 0xff) as u8;
            }
        }
        let mut src = plane.clone();
        for j in 8..24 {
            for i in 8..24 {
                src[j * 64 + i] = plane[j * 64 + i + 2];
            }
        }
        let mv = search_mv(&src, 64, &plane, 64, 64, 64, 8, 8, 16, 16, 4);
        assert_eq!(mv, Mv { row: 0, col: 16 });
    }

    #[test]
    fn zero_when_identical() {
        let plane = vec![40u8; 32 * 32];
        let mv = search_mv_8x8(&plane, 32, &plane, 32, 32, 32, 8, 8, 4);
        assert_eq!(mv, Mv::default());
    }
}
