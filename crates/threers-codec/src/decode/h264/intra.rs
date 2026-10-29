//! H.264 intra sample prediction (§8.3).
//!
//! Neighbour arrays are gathered by the caller, which knows availability:
//! `top[0]` is `p[-1, -1]` and `top[1 + x]` is `p[x, -1]`; `left[0]` is
//! `p[-1, -1]` and `left[1 + y]` is `p[-1, y]`.

/// Smooth the 8×8 neighbours in place (§8.3.2.2.1). `top` holds 16 samples
/// (the upper 8 already substituted when unavailable).
pub fn filter_8x8_refs(top: &mut [i32; 17], left: &mut [i32; 9], at: bool, al: bool, atl: bool) {
    let t = *top;
    let l = *left;
    if at {
        top[1] = if atl {
            (t[0] + 2 * t[1] + t[2] + 2) >> 2
        } else {
            (3 * t[1] + t[2] + 2) >> 2
        };
        for x in 1..15 {
            top[1 + x] = (t[x] + 2 * t[1 + x] + t[2 + x] + 2) >> 2;
        }
        top[16] = (t[15] + 3 * t[16] + 2) >> 2;
    }
    if atl {
        let c = if at && al {
            (t[1] + 2 * t[0] + l[1] + 2) >> 2
        } else if at {
            (3 * t[0] + t[1] + 2) >> 2
        } else if al {
            (3 * t[0] + l[1] + 2) >> 2
        } else {
            t[0]
        };
        top[0] = c;
        left[0] = c;
    }
    if al {
        left[1] = if atl {
            (l[0] + 2 * l[1] + l[2] + 2) >> 2
        } else {
            (3 * l[1] + l[2] + 2) >> 2
        };
        for y in 1..7 {
            left[1 + y] = (l[y] + 2 * l[1 + y] + l[2 + y] + 2) >> 2;
        }
        left[8] = (l[7] + 3 * l[8] + 2) >> 2;
    }
}

/// Intra 4×4 or 8×8 prediction (`n` = 4 or 8) into `out` (`n * n`, raster).
/// `top` has `2n + 1` entries, `left` `n + 1`.
#[allow(clippy::too_many_arguments)]
pub fn pred_nxn(
    out: &mut [i32],
    n: usize,
    mode: u8,
    top: &[i32],
    left: &[i32],
    at: bool,
    al: bool,
    bd: u32,
) {
    let t = |i: isize| top[(i + 1) as usize];
    let l = |i: isize| left[(i + 1) as usize];
    let ni = n as isize;
    match mode {
        0 => {
            for y in 0..n {
                for x in 0..n {
                    out[y * n + x] = top[1 + x];
                }
            }
        }
        1 => {
            for y in 0..n {
                for x in 0..n {
                    out[y * n + x] = left[1 + y];
                }
            }
        }
        2 => {
            let st: i32 = top[1..=n].iter().sum();
            let sl: i32 = left[1..=n].iter().sum();
            let sh = if n == 4 { 2 } else { 3 };
            let dc = match (at, al) {
                (true, true) => (st + sl + n as i32) >> (sh + 1),
                (true, false) => (st + (n as i32 >> 1)) >> sh,
                (false, true) => (sl + (n as i32 >> 1)) >> sh,
                _ => 1 << (bd - 1),
            };
            out[..n * n].fill(dc);
        }
        3 => {
            for y in 0..ni {
                for x in 0..ni {
                    out[(y * ni + x) as usize] = if x == ni - 1 && y == ni - 1 {
                        (t(2 * ni - 2) + 3 * t(2 * ni - 1) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    };
                }
            }
        }
        4 => {
            // Edge: ... left[1], left[0]=corner, top[0..]
            let e = |k: isize| if k >= 0 { t(k - 1) } else { l(-k - 1) };
            for y in 0..ni {
                for x in 0..ni {
                    let k = x - y;
                    out[(y * ni + x) as usize] = (e(k - 1) + 2 * e(k) + e(k + 1) + 2) >> 2;
                }
            }
        }
        5 => {
            for y in 0..ni {
                for x in 0..ni {
                    let z = 2 * x - y;
                    out[(y * ni + x) as usize] = if z >= 0 && z & 1 == 0 {
                        (t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (t(x - (y >> 1) - 2) + 2 * t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * l(-1) + t(0) + 2) >> 2
                    } else {
                        (l(y - 2 * x - 1) + 2 * l(y - 2 * x - 2) + l(y - 2 * x - 3) + 2) >> 2
                    };
                }
            }
        }
        6 => {
            for y in 0..ni {
                for x in 0..ni {
                    let z = 2 * y - x;
                    out[(y * ni + x) as usize] = if z >= 0 && z & 1 == 0 {
                        (l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (l(y - (x >> 1) - 2) + 2 * l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * l(-1) + t(0) + 2) >> 2
                    } else {
                        (t(x - 2 * y - 1) + 2 * t(x - 2 * y - 2) + t(x - 2 * y - 3) + 2) >> 2
                    };
                }
            }
        }
        7 => {
            for y in 0..ni {
                for x in 0..ni {
                    let b = x + (y >> 1);
                    out[(y * ni + x) as usize] = if y & 1 == 0 {
                        (t(b) + t(b + 1) + 1) >> 1
                    } else {
                        (t(b) + 2 * t(b + 1) + t(b + 2) + 2) >> 2
                    };
                }
            }
        }
        _ => {
            let lim = 2 * ni - 3;
            for y in 0..ni {
                for x in 0..ni {
                    let z = x + 2 * y;
                    let b = y + (x >> 1);
                    out[(y * ni + x) as usize] = if z < lim && z & 1 == 0 {
                        (l(b) + l(b + 1) + 1) >> 1
                    } else if z < lim {
                        (l(b) + 2 * l(b + 1) + l(b + 2) + 2) >> 2
                    } else if z == lim {
                        (l(ni - 2) + 3 * l(ni - 1) + 2) >> 2
                    } else {
                        l(ni - 1)
                    };
                }
            }
        }
    }
}

/// Intra 16×16 prediction into `out` (raster 16×16). `top`/`left` have 17
/// entries.
pub fn pred_16x16(out: &mut [i32; 256], mode: u8, top: &[i32], left: &[i32], at: bool, al: bool, bd: u32) {
    match mode {
        0 => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].copy_from_slice(&top[1..17]);
            }
        }
        1 => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].fill(left[1 + y]);
            }
        }
        2 => {
            let st: i32 = top[1..17].iter().sum();
            let sl: i32 = left[1..17].iter().sum();
            let dc = match (at, al) {
                (true, true) => (st + sl + 16) >> 5,
                (true, false) => (st + 8) >> 4,
                (false, true) => (sl + 8) >> 4,
                _ => 1 << (bd - 1),
            };
            out.fill(dc);
        }
        _ => {
            let max = (1 << bd) - 1;
            let mut h = 0;
            let mut v = 0;
            for i in 0..8 {
                h += (i as i32 + 1) * (top[9 + i] - top[7 - i]);
                v += (i as i32 + 1) * (left[9 + i] - left[7 - i]);
            }
            let a = 16 * (left[16] + top[16]);
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] =
                        ((a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5).clamp(0, max);
                }
            }
        }
    }
}

/// Chroma intra prediction for an 8×`h` macroblock (`h` 8 for 4:2:0, 16 for
/// 4:2:2). `top` has 9 entries, `left` `h + 1` (index 0 is the corner).
#[allow(clippy::too_many_arguments)]
/// `al` is the left availability per 4-row block (they differ only beside
/// an MBAFF pair of the other kind under constrained intra prediction).
pub fn pred_chroma(out: &mut [i32; 128], h: usize, mode: u8, top: &[i32], left: &[i32], at: bool, al: &[bool; 4], bd: u32) {
    match mode {
        0 => {
            for by in 0..h / 4 {
                for bx in 0..2 {
                    let st: i32 = top[1 + bx * 4..5 + bx * 4].iter().sum();
                    let sl: i32 = left[1 + by * 4..5 + by * 4].iter().sum();
                    let both = (st + sl + 4) >> 3;
                    let ot = (st + 2) >> 2;
                    let ol = (sl + 2) >> 2;
                    let def = 1 << (bd - 1);
                    let al = al[by];
                    let dc = if (bx == 0 && by == 0) || (bx > 0 && by > 0) {
                        match (at, al) {
                            (true, true) => both,
                            (true, false) => ot,
                            (false, true) => ol,
                            _ => def,
                        }
                    } else if bx > 0 {
                        // Top-right block prefers the samples above it.
                        if at {
                            ot
                        } else if al {
                            ol
                        } else {
                            def
                        }
                    } else if al {
                        ol
                    } else if at {
                        ot
                    } else {
                        def
                    };
                    for y in 0..4 {
                        for x in 0..4 {
                            out[(by * 4 + y) * 8 + bx * 4 + x] = dc;
                        }
                    }
                }
            }
        }
        1 => {
            for y in 0..h {
                out[y * 8..y * 8 + 8].fill(left[1 + y]);
            }
        }
        2 => {
            for y in 0..h {
                out[y * 8..y * 8 + 8].copy_from_slice(&top[1..9]);
            }
        }
        _ => {
            // §8.3.4.4 with xCF = 0, yCF = 4 * (ChromaArrayType != 1).
            let ycf = if h == 16 { 4 } else { 0 };
            let max = (1 << bd) - 1;
            let mut hh = 0;
            let mut v = 0;
            for i in 0..4usize {
                hh += (i as i32 + 1) * (top[5 + i] - top[3 - i]);
            }
            for i in 0..4 + ycf {
                // p[-1, 4 + yCF + i] - p[-1, 2 + yCF - i] (index -1 is the corner).
                v += (i as i32 + 1) * (left[1 + 4 + ycf + i] - left[(1 + 2 + ycf) - i]);
            }
            let a = 16 * (left[h] + top[8]);
            let b = (34 * hh + 32) >> 6;
            let c = ((if ycf != 0 { 5 } else { 34 }) * v + 32) >> 6;
            for y in 0..h {
                for x in 0..8 {
                    out[y * 8 + x] = ((a + b * (x as i32 - 3) + c * (y as i32 - 3 - ycf as i32) + 16) >> 5).clamp(0, max);
                }
            }
        }
    }
}
