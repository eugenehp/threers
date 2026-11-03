//! H.264 intra prediction (ISO/IEC 14496-10 §8.3).
//!
//! Luma is predicted per 4×4 block from nine directional modes; chroma is
//! predicted a whole 8×8 plane at a time from four. Unlike HEVC there is no
//! reference-sample smoothing filter — the taps are baked into each mode's
//! formula instead.
//!
//! Availability is not cosmetic here: a mode whose reference samples do not
//! exist **may not be signalled at all** (§8.3.1.2.x each say so), so
//! [`allowed`] gates the encoder's search rather than the predictor silently
//! substituting something.

// Each mode below is the spec's formula written out in the spec's own indices,
// so that the two can be read side by side. Iterating the reference slices
// instead would be more idiomatic and would lose that correspondence.
#![allow(clippy::needless_range_loop)]

/// Reference samples around a 4×4 luma block.
///
/// `above` is `p[0..7, −1]` — the row above plus the four above-right. When the
/// above-right samples do not exist but the above row does, the spec substitutes
/// `p[3, −1]` for them; [`Neighbors4::new`] does that, so callers pass what they
/// actually have.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neighbors4 {
    pub above: Option<[i32; 8]>,
    pub left: Option<[i32; 4]>,
    pub corner: Option<i32>,
}

impl Neighbors4 {
    /// Build from the four above samples, the optional four above-right, the
    /// four left samples and the corner.
    pub fn new(
        above: Option<[i32; 4]>,
        above_right: Option<[i32; 4]>,
        left: Option<[i32; 4]>,
        corner: Option<i32>,
    ) -> Self {
        let above8 = above.map(|a| {
            let ar = above_right.unwrap_or([a[3]; 4]); // §8.3.1.2.1 substitution
            [a[0], a[1], a[2], a[3], ar[0], ar[1], ar[2], ar[3]]
        });
        Self {
            above: above8,
            left,
            corner,
        }
    }
}

pub const VERT: u8 = 0;
pub const HORIZ: u8 = 1;
pub const DC: u8 = 2;
pub const DIAG_DOWN_LEFT: u8 = 3;
pub const DIAG_DOWN_RIGHT: u8 = 4;
pub const VERT_RIGHT: u8 = 5;
pub const HORIZ_DOWN: u8 = 6;
pub const VERT_LEFT: u8 = 7;
pub const HORIZ_UP: u8 = 8;

/// Whether `mode` may be signalled given what is available. DC always may be.
pub fn allowed(mode: u8, n: &Neighbors4) -> bool {
    let (a, l, c) = (n.above.is_some(), n.left.is_some(), n.corner.is_some());
    match mode {
        DC => true,
        VERT | DIAG_DOWN_LEFT | VERT_LEFT => a,
        HORIZ | HORIZ_UP => l,
        DIAG_DOWN_RIGHT | VERT_RIGHT | HORIZ_DOWN => a && l && c,
        _ => false,
    }
}

#[inline]
fn clip(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// Predict a 4×4 luma block (§8.3.1.2). Row-major, 16 samples.
///
/// The caller must have checked [`allowed`]; an unavailable mode falls back to
/// DC rather than reading absent samples.
pub fn predict4(mode: u8, n: &Neighbors4) -> [i32; 16] {
    if !allowed(mode, n) {
        return dc4(n);
    }
    let mut p = [0i32; 16];
    let a = n.above.unwrap_or([0; 8]);
    let l = n.left.unwrap_or([0; 4]);
    let c = n.corner.unwrap_or(0);
    let put = |p: &mut [i32; 16], x: usize, y: usize, v: i32| p[y * 4 + x] = v;
    // Several modes reach index -1, which is `p[-1,-1]` — the corner, not a
    // sample off the end of the row. Reading it as `a[-1]` is the classic way to
    // get this wrong.
    let at = |i: i32| if i < 0 { c } else { a[i as usize] };
    let lt = |i: i32| if i < 0 { c } else { l[i as usize] };

    match mode {
        VERT => {
            for y in 0..4 {
                for x in 0..4 {
                    put(&mut p, x, y, a[x]);
                }
            }
        }
        HORIZ => {
            for y in 0..4 {
                for x in 0..4 {
                    put(&mut p, x, y, l[y]);
                }
            }
        }
        DC => return dc4(n),
        DIAG_DOWN_LEFT => {
            for y in 0..4 {
                for x in 0..4 {
                    let v = if x == 3 && y == 3 {
                        (a[6] + 3 * a[7] + 2) >> 2
                    } else {
                        (a[x + y] + 2 * a[x + y + 1] + a[x + y + 2] + 2) >> 2
                    };
                    put(&mut p, x, y, v);
                }
            }
        }
        DIAG_DOWN_RIGHT => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = match x.cmp(&y) {
                        std::cmp::Ordering::Greater => {
                            let k = x - y;
                            (at(k - 2) + 2 * at(k - 1) + at(k) + 2) >> 2
                        }
                        std::cmp::Ordering::Less => {
                            let k = y - x;
                            (lt(k - 2) + 2 * lt(k - 1) + lt(k) + 2) >> 2
                        }
                        std::cmp::Ordering::Equal => (at(0) + 2 * c + lt(0) + 2) >> 2,
                    };
                    put(&mut p, x as usize, y as usize, v);
                }
            }
        }
        VERT_RIGHT => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let k = x - (y >> 1);
                    let v = if z >= 0 && z % 2 == 0 {
                        (at(k - 1) + at(k) + 1) >> 1
                    } else if z >= 0 {
                        (at(k - 2) + 2 * at(k - 1) + at(k) + 2) >> 2
                    } else if z == -1 {
                        (lt(0) + 2 * c + at(0) + 2) >> 2
                    } else {
                        (lt(y - 1) + 2 * lt(y - 2) + lt(y - 3) + 2) >> 2
                    };
                    put(&mut p, x as usize, y as usize, v);
                }
            }
        }
        HORIZ_DOWN => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let k = y - (x >> 1);
                    let v = if z >= 0 && z % 2 == 0 {
                        (lt(k - 1) + lt(k) + 1) >> 1
                    } else if z >= 0 {
                        (lt(k - 2) + 2 * lt(k - 1) + lt(k) + 2) >> 2
                    } else if z == -1 {
                        (lt(0) + 2 * c + at(0) + 2) >> 2
                    } else {
                        (at(x - 1) + 2 * at(x - 2) + at(x - 3) + 2) >> 2
                    };
                    put(&mut p, x as usize, y as usize, v);
                }
            }
        }
        VERT_LEFT => {
            for y in 0..4 {
                for x in 0..4 {
                    let k = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        (a[k] + a[k + 1] + 1) >> 1
                    } else {
                        (a[k] + 2 * a[k + 1] + a[k + 2] + 2) >> 2
                    };
                    put(&mut p, x, y, v);
                }
            }
        }
        HORIZ_UP => {
            for y in 0..4 {
                for x in 0..4 {
                    let z = x + 2 * y;
                    let k = y + (x >> 1);
                    let v = if z < 5 && z % 2 == 0 {
                        (l[k] + l[k + 1] + 1) >> 1
                    } else if z < 5 {
                        (l[k] + 2 * l[k + 1] + l[k + 2] + 2) >> 2
                    } else if z == 5 {
                        (l[2] + 3 * l[3] + 2) >> 2
                    } else {
                        l[3]
                    };
                    put(&mut p, x, y, v);
                }
            }
        }
        _ => return dc4(n),
    }
    p
}

/// DC prediction (§8.3.1.2.3): the mean of whatever is available, 128 if nothing.
fn dc4(n: &Neighbors4) -> [i32; 16] {
    let v = match (n.above, n.left) {
        (Some(a), Some(l)) => (a[..4].iter().sum::<i32>() + l.iter().sum::<i32>() + 4) >> 3,
        (Some(a), None) => (a[..4].iter().sum::<i32>() + 2) >> 2,
        (None, Some(l)) => (l.iter().sum::<i32>() + 2) >> 2,
        (None, None) => 128,
    };
    [v; 16]
}

/// Reference samples along the top and left edges of a whole macroblock.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neighbors16 {
    pub above: Option<[i32; 16]>,
    pub left: Option<[i32; 16]>,
    pub corner: Option<i32>,
}

pub const I16_VERT: u8 = 0;
pub const I16_HORIZ: u8 = 1;
pub const I16_DC: u8 = 2;
pub const I16_PLANE: u8 = 3;

/// Whether a 16×16 mode may be signalled given what is available.
pub fn allowed16(mode: u8, n: &Neighbors16) -> bool {
    match mode {
        I16_DC => true,
        I16_HORIZ => n.left.is_some(),
        I16_VERT => n.above.is_some(),
        I16_PLANE => n.above.is_some() && n.left.is_some() && n.corner.is_some(),
        _ => false,
    }
}

/// Predict a whole 16×16 luma macroblock (§8.3.3). Row-major, 256 samples.
///
/// Unlike the 4×4 modes this reads only samples outside the macroblock, so every
/// candidate can be evaluated before any of the macroblock is reconstructed.
pub fn predict16(mode: u8, n: &Neighbors16) -> [i32; 256] {
    if !allowed16(mode, n) {
        return luma16_dc(n);
    }
    let mut p = [0i32; 256];
    match mode {
        I16_DC => return luma16_dc(n),
        I16_VERT => {
            let a = n.above.unwrap();
            for y in 0..16 {
                p[y * 16..y * 16 + 16].copy_from_slice(&a);
            }
        }
        I16_HORIZ => {
            let l = n.left.unwrap();
            for y in 0..16 {
                p[y * 16..y * 16 + 16].fill(l[y]);
            }
        }
        I16_PLANE => {
            let (a, l, c) = (n.above.unwrap(), n.left.unwrap(), n.corner.unwrap());
            // p[-1,-1] is the x'=7 / y'=7 term of each sum.
            let mut hh = 0i32;
            let mut vv = 0i32;
            for k in 0..8i32 {
                let near_h = if k == 7 { c } else { a[(6 - k) as usize] };
                hh += (k + 1) * (a[(8 + k) as usize] - near_h);
                let near_v = if k == 7 { c } else { l[(6 - k) as usize] };
                vv += (k + 1) * (l[(8 + k) as usize] - near_v);
            }
            let a0 = 16 * (l[15] + a[15]);
            let b = (5 * hh + 32) >> 6;
            let cc = (5 * vv + 32) >> 6;
            for y in 0..16i32 {
                for x in 0..16i32 {
                    p[(y * 16 + x) as usize] =
                        clip((a0 + b * (x - 7) + cc * (y - 7) + 16) >> 5);
                }
            }
        }
        _ => return luma16_dc(n),
    }
    p
}

/// 16×16 DC (§8.3.3.3) — the mean of whichever edges exist, 128 if neither does.
fn luma16_dc(n: &Neighbors16) -> [i32; 256] {
    let v = match (n.above, n.left) {
        (Some(a), Some(l)) => {
            (a.iter().sum::<i32>() + l.iter().sum::<i32>() + 16) >> 5
        }
        (Some(a), None) => (a.iter().sum::<i32>() + 8) >> 4,
        (None, Some(l)) => (l.iter().sum::<i32>() + 8) >> 4,
        (None, None) => 128,
    };
    [v; 256]
}

/// Reference samples around an 8×8 chroma block.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborsC {
    pub above: Option<[i32; 8]>,
    pub left: Option<[i32; 8]>,
    pub corner: Option<i32>,
}

pub const C_DC: u8 = 0;
pub const C_HORIZ: u8 = 1;
pub const C_VERT: u8 = 2;
pub const C_PLANE: u8 = 3;

/// Whether a chroma mode may be signalled.
pub fn chroma_allowed(mode: u8, n: &NeighborsC) -> bool {
    match mode {
        C_DC => true,
        C_HORIZ => n.left.is_some(),
        C_VERT => n.above.is_some(),
        C_PLANE => n.above.is_some() && n.left.is_some() && n.corner.is_some(),
        _ => false,
    }
}

/// Predict an 8×8 chroma block (§8.3.4). Row-major, 64 samples.
pub fn predict_chroma(mode: u8, n: &NeighborsC) -> [i32; 64] {
    if !chroma_allowed(mode, n) {
        return chroma_dc(n);
    }
    let mut p = [0i32; 64];
    match mode {
        C_DC => return chroma_dc(n),
        C_HORIZ => {
            let l = n.left.unwrap();
            for y in 0..8 {
                for x in 0..8 {
                    p[y * 8 + x] = l[y];
                }
            }
        }
        C_VERT => {
            let a = n.above.unwrap();
            for y in 0..8 {
                for x in 0..8 {
                    p[y * 8 + x] = a[x];
                }
            }
        }
        C_PLANE => {
            let (a, l, c) = (n.above.unwrap(), n.left.unwrap(), n.corner.unwrap());
            // p[-1,-1] participates as the x'=3 / y'=3 term.
            let mut hh = 0i32;
            let mut vv = 0i32;
            for k in 0..4i32 {
                let far_h = a[(4 + k) as usize];
                let near_h = if k == 3 { c } else { a[(2 - k) as usize] };
                hh += (k + 1) * (far_h - near_h);
                let far_v = l[(4 + k) as usize];
                let near_v = if k == 3 { c } else { l[(2 - k) as usize] };
                vv += (k + 1) * (far_v - near_v);
            }
            let a0 = 16 * (l[7] + a[7]);
            let b = (34 * hh + 32) >> 6;
            let cc = (34 * vv + 32) >> 6;
            for y in 0..8i32 {
                for x in 0..8i32 {
                    p[(y * 8 + x) as usize] =
                        clip((a0 + b * (x - 3) + cc * (y - 3) + 16) >> 5);
                }
            }
        }
        _ => return chroma_dc(n),
    }
    p
}

/// Chroma DC (§8.3.4.1) — each 4×4 quadrant has its own averaging rule, and they
/// are not the same rule. The right-hand and lower quadrants prefer the
/// neighbour that runs alongside them.
fn chroma_dc(n: &NeighborsC) -> [i32; 64] {
    let mut p = [0i32; 64];
    let sum = |s: &[i32]| s.iter().sum::<i32>();
    for by in 0..2usize {
        for bx in 0..2usize {
            let a = n.above.map(|a| sum(&a[bx * 4..bx * 4 + 4]));
            let l = n.left.map(|l| sum(&l[by * 4..by * 4 + 4]));
            let v = match (bx, by) {
                (0, 0) | (1, 1) => match (a, l) {
                    (Some(a), Some(l)) => (a + l + 4) >> 3,
                    (Some(a), None) => (a + 2) >> 2,
                    (None, Some(l)) => (l + 2) >> 2,
                    (None, None) => 128,
                },
                (1, 0) => match (a, l) {
                    (Some(a), _) => (a + 2) >> 2,
                    (None, Some(l)) => (l + 2) >> 2,
                    (None, None) => 128,
                },
                _ => match (l, a) {
                    (Some(l), _) => (l + 2) >> 2,
                    (None, Some(a)) => (a + 2) >> 2,
                    (None, None) => 128,
                },
            };
            for y in 0..4 {
                for x in 0..4 {
                    p[(by * 4 + y) * 8 + bx * 4 + x] = v;
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n4(above: [i32; 4], ar: [i32; 4], left: [i32; 4], corner: i32) -> Neighbors4 {
        Neighbors4::new(Some(above), Some(ar), Some(left), Some(corner))
    }

    #[test]
    fn nothing_available_predicts_mid_grey() {
        let n = Neighbors4::default();
        assert_eq!(predict4(DC, &n), [128; 16]);
        // An unavailable mode must not read absent samples; it falls back.
        assert_eq!(predict4(VERT_RIGHT, &n), [128; 16]);
        assert!(!allowed(VERT, &n) && allowed(DC, &n));
    }

    #[test]
    fn vertical_and_horizontal_copy_their_edge() {
        let n = n4([10, 20, 30, 40], [40; 4], [1, 2, 3, 4], 99);
        let v = predict4(VERT, &n);
        for y in 0..4 {
            assert_eq!(&v[y * 4..y * 4 + 4], &[10, 20, 30, 40]);
        }
        let h = predict4(HORIZ, &n);
        for y in 0..4 {
            assert!(h[y * 4..y * 4 + 4].iter().all(|&s| s == (y as i32 + 1)));
        }
    }

    #[test]
    fn dc_averages_what_exists() {
        // Both edges: (40 + 40 + 4) >> 3 = 10.
        let n = n4([10; 4], [10; 4], [10; 4], 0);
        assert_eq!(predict4(DC, &n)[0], 10);
        // Above only.
        let n = Neighbors4::new(Some([20; 4]), None, None, None);
        assert_eq!(predict4(DC, &n)[0], 20);
        // Left only.
        let n = Neighbors4::new(None, None, Some([30; 4]), None);
        assert_eq!(predict4(DC, &n)[0], 30);
    }

    /// Every directional mode on a constant field must reproduce that constant —
    /// the taps are weighted averages, so a flat input has to stay flat. This
    /// catches an index slipping outside the block.
    #[test]
    fn flat_input_stays_flat_in_every_mode() {
        let n = n4([77; 4], [77; 4], [77; 4], 77);
        for mode in 0..=8u8 {
            let p = predict4(mode, &n);
            assert!(
                p.iter().all(|&v| v == 77),
                "mode {mode} broke flatness: {p:?}"
            );
        }
    }

    #[test]
    fn availability_gates_modes() {
        let above_only = Neighbors4::new(Some([1; 4]), None, None, None);
        assert!(allowed(VERT, &above_only));
        assert!(allowed(DIAG_DOWN_LEFT, &above_only));
        assert!(!allowed(HORIZ, &above_only));
        assert!(!allowed(DIAG_DOWN_RIGHT, &above_only), "needs left+corner");

        let left_only = Neighbors4::new(None, None, Some([1; 4]), None);
        assert!(allowed(HORIZ, &left_only) && allowed(HORIZ_UP, &left_only));
        assert!(!allowed(VERT, &left_only));
    }

    /// Above-right substitution: with no above-right, `p[4..7,-1]` become
    /// `p[3,-1]`, so down-left prediction saturates to that value.
    #[test]
    fn above_right_substitution() {
        let n = Neighbors4::new(Some([5, 6, 7, 8]), None, None, None);
        let a = n.above.unwrap();
        assert_eq!(&a[4..], &[8, 8, 8, 8]);
    }

    #[test]
    fn chroma_flat_stays_flat() {
        let n = NeighborsC {
            above: Some([64; 8]),
            left: Some([64; 8]),
            corner: Some(64),
        };
        for mode in 0..=3u8 {
            let p = predict_chroma(mode, &n);
            assert!(p.iter().all(|&v| v == 64), "chroma mode {mode}: {:?}", &p[..8]);
        }
    }

    #[test]
    fn chroma_dc_quadrants_use_different_edges() {
        // Above is 100s on the left half, 200s on the right; left edge all 0.
        let n = NeighborsC {
            above: Some([100, 100, 100, 100, 200, 200, 200, 200]),
            left: Some([0; 8]),
            corner: Some(0),
        };
        let p = predict_chroma(C_DC, &n);
        assert_eq!(p[0], (400 + 4) >> 3, "top-left averages both edges (left is all zero)");
        assert_eq!(p[4], (800 + 2) >> 2, "top-right prefers above");
        assert_eq!(p[4 * 8], 0, "bottom-left prefers left, which is all zero");
    }
}
