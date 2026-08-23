//! HEVC intra prediction (H.265 §8.4.4.2): DC, planar, and the 33 angular modes.
//!
//! Given decoded neighbor samples (top row + left column + corner), predicts an
//! `n×n` block. The encoder tries modes and keeps the cheapest; the decoder runs
//! the chosen mode. Includes the mandatory luma boundary-smoothing post-filters
//! for DC / horizontal / vertical modes (§8.4.4.2) — without them, high-contrast
//! content mismatches a conformant decoder. Reference-sample `[1 2 1]` smoothing
//! is applied by the caller before [`predict`].
//!
//! Reference layout: `above[0]` and `left[0]` are both the corner `p[-1][-1]`;
//! `above[1..=2n]` are the top samples `p[0..][-1]`; `left[1..=2n]` are the left
//! samples `p[-1][0..]`.
//!
//! Pure integer math (wasm-safe).

/// Planar mode index.
pub const PLANAR: u8 = 0;
/// DC mode index.
pub const DC: u8 = 1;
/// Pure-horizontal angular mode.
pub const HORIZONTAL: u8 = 10;
/// Pure-vertical angular mode.
pub const VERTICAL: u8 = 26;

/// `intraPredAngle` per mode (H.265 Table 8-4); modes 0/1 (planar/DC) unused.
const ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
    -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

#[inline]
fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

/// `invAngle` for negative angles (H.265 Table 8-5).
fn inv_angle(angle: i32) -> i32 {
    match angle {
        -2 => 4096,
        -5 => 1638,
        -9 => 910,
        -13 => 630,
        -17 => 482,
        -21 => 390,
        -26 => 315,
        -32 => 256,
        _ => 0,
    }
}

/// Predict an `n×n` block (row-major, `pred[y*n + x]`) for intra `mode` using the
/// neighbor references `above` / `left` (each length `2n + 1`). `luma` enables the
/// boundary-smoothing post-filters for DC / horizontal / vertical modes
/// (§8.4.4.2, luma-only, `nTbS < 32`).
pub fn predict(mode: u8, n: usize, above: &[i32], left: &[i32], luma: bool) -> Vec<i32> {
    debug_assert!(above.len() > 2 * n && left.len() > 2 * n);
    let mut pred = match mode {
        PLANAR => planar(n, above, left),
        DC => dc(n, above, left),
        _ => angular(mode, n, above, left),
    };
    if luma && n < 32 {
        boundary_filter(mode, n, above, left, &mut pred);
    }
    pred
}

#[inline]
fn clip8(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// Post-prediction boundary smoothing (§8.4.4.2.4 DC, §8.4.4.2.6 modes 10/26).
fn boundary_filter(mode: u8, n: usize, above: &[i32], left: &[i32], pred: &mut [i32]) {
    let corner = above[0]; // p[-1][-1]
    match mode {
        DC => {
            let dc_val = pred[0]; // uniform
            pred[0] = (left[1] + 2 * dc_val + above[1] + 2) >> 2;
            for x in 1..n {
                pred[x] = (above[x + 1] + 3 * dc_val + 2) >> 2;
            }
            for y in 1..n {
                pred[y * n] = (left[y + 1] + 3 * dc_val + 2) >> 2;
            }
        }
        HORIZONTAL => {
            for x in 0..n {
                pred[x] = clip8(left[1] + ((above[x + 1] - corner) >> 1));
            }
        }
        VERTICAL => {
            for y in 0..n {
                pred[y * n] = clip8(above[1] + ((left[y + 1] - corner) >> 1));
            }
        }
        _ => {}
    }
}

fn dc(n: usize, above: &[i32], left: &[i32]) -> Vec<i32> {
    let mut sum = 0i32;
    for i in 1..=n {
        sum += above[i] + left[i];
    }
    let dc_val = (sum + n as i32) >> (log2(n) + 1);
    vec![dc_val; n * n]
}

fn planar(n: usize, above: &[i32], left: &[i32]) -> Vec<i32> {
    let ni = n as i32;
    let shift = log2(n) + 1;
    let top_right = above[n + 1]; // p[n][-1]
    let bottom_left = left[n + 1]; // p[-1][n]
    let mut pred = vec![0i32; n * n];
    for y in 0..n {
        for x in 0..n {
            let (xi, yi) = (x as i32, y as i32);
            let h = (ni - 1 - xi) * left[y + 1] + (xi + 1) * top_right;
            let v = (ni - 1 - yi) * above[x + 1] + (yi + 1) * bottom_left;
            pred[y * n + x] = (h + v + ni) >> shift;
        }
    }
    pred
}

fn angular(mode: u8, n: usize, above: &[i32], left: &[i32]) -> Vec<i32> {
    let angle = ANGLE[mode as usize];
    let vertical = mode >= 18;
    // main = reference projected along the mode; side = the orthogonal edge.
    let (main_ref, side_ref) = if vertical {
        (above, left)
    } else {
        (left, above)
    };

    // Build the extended main reference indexed [-n .. 2n], offset by n.
    let off = n as i32;
    let mut rm = vec![0i32; 3 * n + 1];
    for i in 0..=2 * n {
        rm[(i as i32 + off) as usize] = main_ref[i];
    }
    if angle < 0 {
        let inv = inv_angle(angle);
        let lim = (n as i32 * angle) >> 5;
        let mut sum = 128;
        let mut k = -1i32;
        while k > lim {
            sum += inv;
            rm[(k + off) as usize] = side_ref[(sum >> 8) as usize];
            k -= 1;
        }
    }
    let get = |k: i32| rm[(k + off) as usize];

    let mut pred = vec![0i32; n * n];
    for j in 0..n {
        // j is the axis the angle advances along (rows for vertical, cols for horizontal).
        let pos = (j as i32 + 1) * angle;
        let idx = pos >> 5;
        let fract = pos & 31;
        for i in 0..n {
            let ii = i as i32;
            let p = if fract != 0 {
                ((32 - fract) * get(ii + idx + 1) + fract * get(ii + idx + 2) + 16) >> 5
            } else {
                get(ii + idx + 1)
            };
            // i indexes across the block; for vertical i=x, j=y; for horizontal i=y, j=x.
            if vertical {
                pred[j * n + i] = p;
            } else {
                pred[i * n + j] = p;
            }
        }
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs(n: usize, corner: i32, top: &[i32], leftc: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut a = vec![corner; 2 * n + 1];
        let mut l = vec![corner; 2 * n + 1];
        for i in 0..2 * n {
            a[i + 1] = top.get(i).copied().unwrap_or(*top.last().unwrap());
            l[i + 1] = leftc.get(i).copied().unwrap_or(*leftc.last().unwrap());
        }
        (a, l)
    }

    #[test]
    fn dc_is_average() {
        let n = 4;
        let (a, l) = refs(n, 100, &[100; 8], &[100; 8]);
        let p = predict(DC, n, &a, &l, false);
        assert!(p.iter().all(|&v| v == 100), "flat DC = 100");
    }

    #[test]
    fn planar_flat_stays_flat() {
        let n = 4;
        let (a, l) = refs(n, 50, &[50; 8], &[50; 8]);
        let p = predict(PLANAR, n, &a, &l, false);
        assert!(p.iter().all(|&v| v == 50));
    }

    #[test]
    fn vertical_copies_top_row() {
        let n = 4;
        let top = [10, 20, 30, 40, 40, 40, 40, 40];
        let (a, l) = refs(n, 5, &top, &[5; 8]);
        let p = predict(VERTICAL, n, &a, &l, false);
        // Every row equals the top samples above[1..=4].
        for y in 0..n {
            for x in 0..n {
                assert_eq!(p[y * n + x], top[x], "vertical at {x},{y}");
            }
        }
    }

    #[test]
    fn horizontal_copies_left_col() {
        let n = 4;
        let leftc = [11, 22, 33, 44, 44, 44, 44, 44];
        let (a, l) = refs(n, 5, &[5; 8], &leftc);
        let p = predict(HORIZONTAL, n, &a, &l, false);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(p[y * n + x], leftc[y], "horizontal at {x},{y}");
            }
        }
    }

    #[test]
    fn horizontal_boundary_filter() {
        // Luma horizontal: first row = left[1] + ((above[x+1] - corner) >> 1).
        let n = 8;
        let (mut a, mut l) = refs(n, 100, &[100; 16], &[100; 16]);
        a[0] = 90; // corner
        a[1] = 130; // above[1] (p[0][-1])
        l[1] = 100; // left[1] (p[-1][0])
        let p = predict(HORIZONTAL, n, &a, &l, true);
        // pred[0][0] = 100 + ((130 - 90) >> 1) = 100 + 20 = 120.
        assert_eq!(p[0], 120, "filtered top-left");
        // Row 1 is the plain horizontal copy (left[2] = 100).
        assert_eq!(p[n], 100, "unfiltered second row");
    }

    #[test]
    fn dc_boundary_filter_smooths_edges() {
        let n = 8;
        let (a, l) = refs(n, 128, &[200; 16], &[50; 16]);
        let unfiltered = predict(DC, n, &a, &l, false);
        let filtered = predict(DC, n, &a, &l, true);
        // Interior pixels are the plain DC; the top/left edges are smoothed.
        assert_eq!(filtered[n + 1], unfiltered[n + 1], "interior unchanged");
        assert_ne!(filtered[1], unfiltered[1], "top edge smoothed");
        assert_ne!(filtered[n], unfiltered[n], "left edge smoothed");
    }

    #[test]
    fn angular_negative_mode_in_range() {
        // Mode 18 (angle -32) must not panic and must fill the block.
        let n = 8;
        let (a, l) = refs(n, 128, &[128; 16], &[128; 16]);
        let p = predict(18, n, &a, &l, false);
        assert_eq!(p.len(), n * n);
        assert!(p.iter().all(|&v| v == 128), "flat refs → flat prediction");
    }
}
