//! HEVC intra sample prediction (§8.4.4.2).
//!
//! Reference samples live in one linear array running from the bottom of the
//! left column, up through the corner, and along the top row:
//! `arr[2n - 1 - y] = p[-1][y]`, `arr[2n] = p[-1][-1]`,
//! `arr[2n + 1 + x] = p[x][-1]`. Substitution and both smoothing filters are
//! then one-dimensional walks along it, exactly as the specification orders
//! them.

use super::tables::{INTRA_ANGLE, INV_ANGLE};

/// Fill unavailable entries of `arr` (`4n + 1` long) as §8.4.4.2.2 says.
pub fn substitute(arr: &mut [i32], avail: &[bool], bd: u32) {
    let len = arr.len();
    match avail.iter().position(|&a| a) {
        None => arr.fill(1 << (bd - 1)),
        Some(first) => {
            for v in arr.iter_mut().take(first) {
                *v = 0;
            }
            let fv = arr[first];
            for v in arr.iter_mut().take(first) {
                *v = fv;
            }
            for i in first + 1..len {
                if !avail[i] {
                    arr[i] = arr[i - 1];
                }
            }
        }
    }
}

/// Smooth the references (§8.4.4.2.3) when the mode and size call for it.
pub fn filter(arr: &mut [i32], n: usize, mode: u32, luma: bool, strong_enabled: bool, bd: u32) {
    if !luma || mode == 1 || n == 4 {
        return;
    }
    let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
    let thres = match n {
        8 => 7,
        16 => 1,
        _ => 0,
    };
    if min_dist <= thres {
        return;
    }
    let c = 2 * n;
    let last = 4 * n;
    if strong_enabled
        && n == 32
        && (arr[c] + arr[last] - 2 * arr[c + n]).abs() < (1 << (bd - 5))
        && (arr[c] + arr[0] - 2 * arr[c - n]).abs() < (1 << (bd - 5))
    {
        let (bl, corner, tr) = (arr[0], arr[c], arr[last]);
        for i in 0..63 {
            // Top row x = i: between the corner and the far right.
            arr[c + 1 + i] = ((63 - i as i32) * corner + (i as i32 + 1) * tr + 32) >> 6;
            // Left column y = i: between the corner and the bottom.
            arr[c - 1 - i] = ((63 - i as i32) * corner + (i as i32 + 1) * bl + 32) >> 6;
        }
        return;
    }
    let src = arr.to_vec();
    for i in 1..last {
        arr[i] = (src[i - 1] + 2 * src[i] + src[i + 1] + 2) >> 2;
    }
}

/// Predict an `n × n` block from (substituted, filtered) references into
/// `out` (raster). `edge_filters` enables the DC / horizontal / vertical
/// boundary smoothing, which applies to luma below 32×32 only.
pub fn predict(out: &mut [i32], arr: &[i32], n: usize, mode: u32, edge_filters: bool, bd: u32) {
    let c = 2 * n;
    let left = |y: isize| arr[(c as isize - 1 - y) as usize];
    let top = |x: isize| arr[(c as isize + 1 + x) as usize];
    let max = (1 << bd) - 1;
    let ni = n as isize;
    match mode {
        0 => {
            let sh = n.trailing_zeros() + 1;
            for y in 0..ni {
                for x in 0..ni {
                    out[(y * ni + x) as usize] = ((ni - 1 - x) as i32 * left(y)
                        + (x + 1) as i32 * top(ni)
                        + (ni - 1 - y) as i32 * top(x)
                        + (y + 1) as i32 * left(ni)
                        + n as i32)
                        >> sh;
                }
            }
        }
        1 => {
            let mut sum = n as i32;
            for i in 0..ni {
                sum += top(i) + left(i);
            }
            let dc = sum >> (n.trailing_zeros() + 1);
            out[..n * n].fill(dc);
            if edge_filters && n < 32 {
                out[0] = (left(0) + 2 * dc + top(0) + 2) >> 2;
                for x in 1..ni {
                    out[x as usize] = (top(x) + 3 * dc + 2) >> 2;
                }
                for y in 1..ni {
                    out[(y * ni) as usize] = (left(y) + 3 * dc + 2) >> 2;
                }
            }
        }
        _ => {
            let angle = INTRA_ANGLE[mode as usize - 2];
            let vertical = mode >= 18;
            // ref[k] for k in -n..=2n, stored at refs[k + n].
            let mut refs = [0i32; 3 * 64 + 1];
            let base = n as isize;
            let main = |k: isize| if vertical { top(k - 1) } else { left(k - 1) };
            let side = |k: isize| if vertical { left(k - 1) } else { top(k - 1) };
            for k in 0..=ni {
                refs[(k + base) as usize] = main(k);
            }
            if angle < 0 {
                let lim = (ni * angle as isize) >> 5;
                if lim < -1 {
                    let inv = INV_ANGLE[mode as usize - 11] as isize;
                    for k in lim..0 {
                        refs[(k + base) as usize] = side((k * inv + 128) >> 8);
                    }
                }
            } else {
                for k in ni + 1..=2 * ni {
                    refs[(k + base) as usize] = main(k);
                }
            }
            let r = |k: isize| refs[(k + base) as usize];
            for j in 0..ni {
                // j runs along the prediction direction's cross axis.
                let pos = (j + 1) * angle as isize;
                let idx = pos >> 5;
                let fact = (pos & 31) as i32;
                for i in 0..ni {
                    let v = if fact != 0 {
                        ((32 - fact) * r(i + idx + 1) + fact * r(i + idx + 2) + 16) >> 5
                    } else {
                        r(i + idx + 1)
                    };
                    let (x, y) = if vertical { (i, j) } else { (j, i) };
                    out[(y * ni + x) as usize] = v;
                }
            }
            if edge_filters && n < 32 {
                if mode == 26 {
                    for y in 0..ni {
                        out[(y * ni) as usize] = (top(0) + ((left(y) - left(-1)) >> 1)).clamp(0, max);
                    }
                } else if mode == 10 {
                    for x in 0..ni {
                        out[x as usize] = (left(0) + ((top(x) - top(-1)) >> 1)).clamp(0, max);
                    }
                }
            }
        }
    }
}
