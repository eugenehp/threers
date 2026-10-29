//! VP9 inter prediction (`vpx_convolve8` + switchable 8-tap kernels).
//!
//! Motion vectors are stored in **1/8-pel luma** units (Q3). The convolve path
//! works in Q4 (1/16-pel): luma MVs are doubled; chroma (4:2:0) keeps the Q3
//! value as Q4 of chroma pixels (`clamp_mv_to_umv_border_sb` with `ss=1`).

const FILTER_BITS: i32 = 7;
const SUBPEL_TAPS: usize = 8;
const SUBPEL_BITS: i32 = 4;
const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1;

/// Frame / block interpolation filter (`INTERP_FILTER`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum InterpFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    /// Not used under `SWITCHABLE` (only the three 8-tap filters switch).
    Bilinear = 3,
}

impl InterpFilter {
    pub const SWITCHABLE_FILTERS: u8 = 3;

    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::EightTapSmooth,
            2 => Self::EightTapSharp,
            3 => Self::Bilinear,
            _ => Self::EightTap,
        }
    }
}

/// `sub_pel_filters_8` — EIGHTTAP / Lagrangian.
const EIGHTTAP: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// `sub_pel_filters_8lp` — EIGHTTAP_SMOOTH.
const EIGHTTAP_SMOOTH: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 43, 4, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 18, 59, 51, 9, -4, 0],
    [-1, -4, 16, 57, 53, 12, -4, -1],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [-1, -4, 12, 53, 57, 16, -4, -1],
    [0, -4, 9, 51, 59, 18, -4, -1],
    [0, -4, 7, 49, 60, 21, -3, -2],
    [0, -4, 5, 46, 62, 24, -3, -2],
    [0, -4, 4, 43, 63, 26, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 1, 38, 64, 32, -1, -3],
];

/// `sub_pel_filters_8s` — EIGHTTAP_SHARP.
const EIGHTTAP_SHARP: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
    [-4, 11, -23, 80, 80, -23, 11, -4],
    [-4, 10, -21, 70, 90, -24, 11, -4],
    [-3, 9, -19, 59, 100, -24, 10, -4],
    [-3, 8, -16, 48, 108, -23, 10, -4],
    [-2, 6, -13, 37, 115, -20, 9, -4],
    [-2, 5, -10, 27, 121, -17, 7, -3],
    [-1, 3, -6, 17, 125, -13, 5, -2],
    [0, 1, -3, 8, 127, -7, 3, -1],
];

/// `bilinear_filters`.
const BILINEAR: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

fn kernel(filter: InterpFilter) -> &'static [[i16; 8]; 16] {
    match filter {
        InterpFilter::EightTap => &EIGHTTAP,
        InterpFilter::EightTapSmooth => &EIGHTTAP_SMOOTH,
        InterpFilter::EightTapSharp => &EIGHTTAP_SHARP,
        InterpFilter::Bilinear => &BILINEAR,
    }
}

#[inline]
fn clip_pixel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn round_bits(sum: i32) -> u8 {
    clip_pixel((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS)
}

#[inline]
fn sample(plane: &[u8], stride: usize, width: usize, height: usize, x: i32, y: i32) -> u8 {
    let sx = x.clamp(0, width as i32 - 1) as usize;
    let sy = y.clamp(0, height as i32 - 1) as usize;
    plane[sy * stride + sx]
}

/// Predict an `w×h` block at plane position `(x, y)` from `reference`.
///
/// `mv_row_q3` / `mv_col_q3` are luma 1/8-pel units. Pass `chroma = true` for
/// U/V planes (4:2:0 subsampling of the MV into Q4).
pub fn predict_inter(
    reference: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv_row_q3: i16,
    mv_col_q3: i16,
    chroma: bool,
    filter: InterpFilter,
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), w * h);
    let ss = if chroma { 1 } else { 0 };
    // Q3 → Q4: luma ×2, chroma ×1 (libvpx `clamp_mv_to_umv_border_sb`).
    let mv_q4_row = i32::from(mv_row_q3) << (1 - ss);
    let mv_q4_col = i32::from(mv_col_q3) << (1 - ss);
    let subpel_y = mv_q4_row & SUBPEL_MASK;
    let subpel_x = mv_q4_col & SUBPEL_MASK;
    let int_y = mv_q4_row >> SUBPEL_BITS;
    let int_x = mv_q4_col >> SUBPEL_BITS;
    let base_y = y as i32 + int_y;
    let base_x = x as i32 + int_x;
    let taps = kernel(filter);

    if subpel_x == 0 && subpel_y == 0 {
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = sample(
                    reference,
                    ref_stride,
                    ref_w,
                    ref_h,
                    base_x + c as i32,
                    base_y + r as i32,
                );
            }
        }
        return;
    }

    if subpel_y == 0 {
        let f = &taps[subpel_x as usize];
        for r in 0..h {
            for c in 0..w {
                let mut sum = 0i32;
                for k in 0..SUBPEL_TAPS {
                    let sx = base_x + c as i32 + k as i32 - 3;
                    sum += i32::from(sample(
                        reference,
                        ref_stride,
                        ref_w,
                        ref_h,
                        sx,
                        base_y + r as i32,
                    )) * i32::from(f[k]);
                }
                out[r * w + c] = round_bits(sum);
            }
        }
        return;
    }

    if subpel_x == 0 {
        let f = &taps[subpel_y as usize];
        for r in 0..h {
            for c in 0..w {
                let mut sum = 0i32;
                for k in 0..SUBPEL_TAPS {
                    let sy = base_y + r as i32 + k as i32 - 3;
                    sum += i32::from(sample(
                        reference,
                        ref_stride,
                        ref_w,
                        ref_h,
                        base_x + c as i32,
                        sy,
                    )) * i32::from(f[k]);
                }
                out[r * w + c] = round_bits(sum);
            }
        }
        return;
    }

    // 2D: horizontal into temp, then vertical.
    let x_filter = &taps[subpel_x as usize];
    let y_filter = &taps[subpel_y as usize];
    let tmp_h = h + SUBPEL_TAPS - 1;
    let mut tmp = vec![0u8; tmp_h * w];
    for r in 0..tmp_h {
        let sy = base_y + r as i32 - 3;
        for c in 0..w {
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                let sx = base_x + c as i32 + k as i32 - 3;
                sum += i32::from(sample(reference, ref_stride, ref_w, ref_h, sx, sy))
                    * i32::from(x_filter[k]);
            }
            tmp[r * w + c] = round_bits(sum);
        }
    }
    for r in 0..h {
        for c in 0..w {
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += i32::from(tmp[(r + k) * w + c]) * i32::from(y_filter[k]);
            }
            out[r * w + c] = round_bits(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullpel_is_copy() {
        let mut plane = vec![0u8; 16 * 16];
        for i in 0..plane.len() {
            plane[i] = (i % 251) as u8;
        }
        let mut out = [0u8; 16];
        predict_inter(
            &plane,
            16,
            16,
            16,
            4,
            4,
            4,
            4,
            8,
            16,
            false,
            InterpFilter::EightTap,
            &mut out,
        );
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(out[r * 4 + c], plane[(5 + r) * 16 + (6 + c)]);
            }
        }
    }

    #[test]
    fn phase0_matches_fullpel() {
        let mut plane = vec![40u8; 32 * 32];
        plane[8 * 32 + 8] = 200;
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        predict_inter(
            &plane,
            32,
            32,
            32,
            8,
            8,
            2,
            2,
            0,
            0,
            false,
            InterpFilter::EightTap,
            &mut a,
        );
        predict_inter(
            &plane,
            32,
            32,
            32,
            8,
            8,
            2,
            2,
            0,
            0,
            false,
            InterpFilter::EightTapSmooth,
            &mut b,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn halfpel_is_between_neighbors() {
        let mut plane = vec![0u8; 16 * 16];
        for c in 0..16 {
            plane[4 * 16 + c] = 0;
            plane[5 * 16 + c] = 255;
        }
        let mut out = [0u8; 1];
        // Vertical half-pel (mv_row=4 → Q4 phase 8).
        predict_inter(
            &plane,
            16,
            16,
            16,
            4,
            4,
            1,
            1,
            4,
            0,
            false,
            InterpFilter::EightTap,
            &mut out,
        );
        let between = out[0] as u32;
        assert!(
            between > 32 && between < 224,
            "half-pel MC should interpolate: {between}"
        );
    }

    #[test]
    fn smooth_differs_from_eighttap_at_halfpel() {
        let mut plane = vec![0u8; 16 * 16];
        for r in 0..16 {
            for c in 0..16 {
                plane[r * 16 + c] = if c < 8 { 0 } else { 255 };
            }
        }
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        predict_inter(
            &plane,
            16,
            16,
            16,
            4,
            4,
            2,
            2,
            0,
            4,
            false,
            InterpFilter::EightTap,
            &mut a,
        );
        predict_inter(
            &plane,
            16,
            16,
            16,
            4,
            4,
            2,
            2,
            0,
            4,
            false,
            InterpFilter::EightTapSmooth,
            &mut b,
        );
        assert_ne!(a, b);
    }
}
