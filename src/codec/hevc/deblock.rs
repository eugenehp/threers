//! The HEVC in-loop deblocking filter (§8.7.2), for all-intra pictures.
//!
//! Structurally unlike H.264's. It works on an 8×8 grid rather than 4×4, so
//! only every other transform edge is a candidate; it decides per four lines
//! whether to filter at all, and then whether to filter *strongly*, from the
//! second differences either side of the edge; and the whole picture's vertical
//! edges are filtered before any horizontal one, rather than interleaving per
//! block. The net effect is a filter that leaves real detail alone more readily
//! than H.264's does.
//!
//! Scope here: every coding unit intra, one slice, filter offsets zero. That
//! pins the boundary strength at 2 on every filtered edge (§8.7.2.4) and makes
//! the whole inter derivation unreachable. As on the H.264 side, intra
//! prediction reads unfiltered samples, so this is a pass over the finished
//! picture rather than something the reconstruction loop interleaves with.

use crate::codec::hevc::deblock_tables::{BETA, TC};

#[inline]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Which 8×8-grid edges exist, at four-sample granularity.
///
/// An edge is only a candidate where a transform block actually ends. Inside a
/// 32×32 transform there is nothing to smooth at sample 8, 16 or 24, and
/// filtering there would blur real content — so the tree's leaf boundaries have
/// to be recorded as it is built.
pub struct EdgeMap {
    left: Vec<bool>,
    top: Vec<bool>,
    w4: usize,
}

impl EdgeMap {
    pub fn new(cw: u32, ch: u32) -> Self {
        let (w4, h4) = ((cw / 4) as usize, (ch / 4) as usize);
        Self {
            left: vec![false; w4 * h4],
            top: vec![false; w4 * h4],
            w4,
        }
    }

    /// Record the left and top edges of one transform block.
    pub fn mark(&mut self, x: u32, y: u32, size: u32) {
        if x.is_multiple_of(8) && x > 0 {
            for r in (y..y + size).step_by(4) {
                let i = (r / 4) as usize * self.w4 + (x / 4) as usize;
                if i < self.left.len() {
                    self.left[i] = true;
                }
            }
        }
        if y.is_multiple_of(8) && y > 0 {
            for c in (x..x + size).step_by(4) {
                let i = (y / 4) as usize * self.w4 + (c / 4) as usize;
                if i < self.top.len() {
                    self.top[i] = true;
                }
            }
        }
    }

    /// Copy another map's marks into this one at `(x0, y0)`.
    ///
    /// Tiles are encoded as pictures of their own, so each builds an edge map in
    /// its own coordinates; the picture-level filter needs them in one.
    pub fn paste(&mut self, src: &EdgeMap, w: u32, h: u32, x0: u32, y0: u32) {
        for ty in (0..h).step_by(4) {
            for tx in (0..w).step_by(4) {
                let si = (ty / 4) as usize * src.w4 + (tx / 4) as usize;
                let di = ((y0 + ty) / 4) as usize * self.w4 + ((x0 + tx) / 4) as usize;
                if si < src.left.len() && di < self.left.len() {
                    self.left[di] = src.left[si];
                    self.top[di] = src.top[si];
                }
            }
        }
        // A tile's own left and top edges are picture-internal block edges, and
        // the filter is allowed across them
        // (`loop_filter_across_tiles_enabled_flag`), so they have to be marked —
        // the tile could not know that, having been encoded as a picture whose
        // edge they were.
        if x0 > 0 {
            for ty in (0..h).step_by(4) {
                let di = ((y0 + ty) / 4) as usize * self.w4 + (x0 / 4) as usize;
                if di < self.left.len() {
                    self.left[di] = true;
                }
            }
        }
        if y0 > 0 {
            for tx in (0..w).step_by(4) {
                let di = (y0 / 4) as usize * self.w4 + ((x0 + tx) / 4) as usize;
                if di < self.top.len() {
                    self.top[di] = true;
                }
            }
        }
    }

    fn has_left(&self, x: u32, y: u32) -> bool {
        let i = (y / 4) as usize * self.w4 + (x / 4) as usize;
        i < self.left.len() && self.left[i]
    }

    fn has_top(&self, x: u32, y: u32) -> bool {
        let i = (y / 4) as usize * self.w4 + (x / 4) as usize;
        i < self.top.len() && self.top[i]
    }
}

/// Filter the picture in place.
///
/// `qp_at` returns the decoder's `QpY` for a luma position, and `qp_c` maps a
/// luma QP to its chroma counterpart.
pub fn deblock(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    cw: u32,
    ch: u32,
    edges: &EdgeMap,
    qp_at: impl Fn(u32, u32) -> i32,
    qp_c: impl Fn(i32) -> i32,
) {
    // Every vertical edge in the picture, then every horizontal one — the
    // second pass reads what the first wrote (§8.7.2.1).
    for x in (8..cw).step_by(8) {
        for yy in (0..ch).step_by(4) {
            if edges.has_left(x, yy) {
                let qp = (qp_at(x - 1, yy) + qp_at(x, yy) + 1) >> 1;
                luma_segment(y, cw, x, yy, 1, cw as isize, qp);
            }
        }
    }
    for yy in (8..ch).step_by(8) {
        for x in (0..cw).step_by(4) {
            if edges.has_top(x, yy) {
                let qp = (qp_at(x, yy - 1) + qp_at(x, yy) + 1) >> 1;
                luma_segment(y, cw, x, yy, cw as isize, 1, qp);
            }
        }
    }

    // Chroma sits on its own 8×8 grid, which in 4:2:0 is every sixteenth luma
    // sample, and is filtered only where the luma boundary strength is 2 — which
    // for an all-intra picture means wherever a luma edge exists at all.
    let (cwc, chc) = (cw / 2, ch / 2);
    for plane in [&mut *u, &mut *v] {
        for cx in (8..cwc).step_by(8) {
            for cy in (0..chc).step_by(4) {
                let (lx, ly) = (cx * 2, cy * 2);
                if edges.has_left(lx, ly) {
                    let qp = (qp_at(lx - 1, ly) + qp_at(lx, ly) + 1) >> 1;
                    chroma_segment(plane, cwc, cx, cy, 1, cwc as isize, qp_c(qp));
                }
            }
        }
        for cy in (8..chc).step_by(8) {
            for cx in (0..cwc).step_by(4) {
                let (lx, ly) = (cx * 2, cy * 2);
                if edges.has_top(lx, ly) {
                    let qp = (qp_at(lx, ly - 1) + qp_at(lx, ly) + 1) >> 1;
                    chroma_segment(plane, cwc, cx, cy, cwc as isize, 1, qp_c(qp));
                }
            }
        }
    }
}

/// One four-line segment of a luma edge (§8.7.2.5.3 and §8.7.2.5.7).
fn luma_segment(plane: &mut [u8], w: u32, x: u32, y: u32, step: isize, along: isize, qp: i32) {
    let beta = BETA[qp.clamp(0, 51) as usize];
    // Boundary strength 2 shifts the clipping table two places along.
    let tc = TC[(qp + 2).clamp(0, 53) as usize];
    if beta == 0 {
        return;
    }
    let base = (y * w + x) as isize;
    let at = |plane: &[u8], line: isize, i: isize| plane[(base + line * along + i * step) as usize] as i32;

    // The decision reads only the first and last of the four lines: HEVC judges
    // the whole segment on its ends, where H.264 judges every line separately.
    let d2 = |plane: &[u8], line: isize, i: isize| {
        (at(plane, line, i - 3) - 2 * at(plane, line, i - 2) + at(plane, line, i - 1)).abs()
    };
    let dp0 = d2(plane, 0, 0);
    let dq0 = (at(plane, 0, 2) - 2 * at(plane, 0, 1) + at(plane, 0, 0)).abs();
    let dp3 = d2(plane, 3, 0);
    let dq3 = (at(plane, 3, 2) - 2 * at(plane, 3, 1) + at(plane, 3, 0)).abs();
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    let (dp, dq) = (dp0 + dp3, dq0 + dq3);
    if dpq0 + dpq3 >= beta {
        return;
    }

    let strong_line = |plane: &[u8], line: isize, dpq: i32| {
        let (p0, p3) = (at(plane, line, -1), at(plane, line, -4));
        let (q0, q3) = (at(plane, line, 0), at(plane, line, 3));
        2 * dpq < (beta >> 2)
            && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
            && (p0 - q0).abs() < ((5 * tc + 1) >> 1)
    };
    let strong = strong_line(plane, 0, dpq0) && strong_line(plane, 3, dpq3);
    let ep = dp < ((beta + (beta >> 1)) >> 3);
    let eq = dq < ((beta + (beta >> 1)) >> 3);

    for line in 0..4isize {
        let o = base + line * along;
        let s = |plane: &[u8], i: isize| plane[(o + i * step) as usize] as i32;
        let (p0, p1, p2, p3) = (s(plane, -1), s(plane, -2), s(plane, -3), s(plane, -4));
        let (q0, q1, q2, q3) = (s(plane, 0), s(plane, 1), s(plane, 2), s(plane, 3));
        let mut set = |i: isize, val: i32| plane[(o + i * step) as usize] = clip1(val);

        if strong {
            // Up to three samples a side move, each held within ±2tC of where
            // it started so a strong filter can never invent a new edge.
            let c = |orig: i32, val: i32| val.clamp(orig - 2 * tc, orig + 2 * tc);
            set(-1, c(p0, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3));
            set(-2, c(p1, (p2 + p1 + p0 + q0 + 2) >> 2));
            set(-3, c(p2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3));
            set(0, c(q0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3));
            set(1, c(q1, (p0 + q0 + q1 + q2 + 2) >> 2));
            set(2, c(q2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3));
        } else {
            let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() >= tc * 10 {
                continue;
            }
            let d = delta.clamp(-tc, tc);
            set(-1, p0 + d);
            set(0, q0 - d);
            if ep {
                let dp = ((((p2 + p0 + 1) >> 1) - p1 + d) >> 1).clamp(-(tc >> 1), tc >> 1);
                set(-2, p1 + dp);
            }
            if eq {
                let dq = ((((q2 + q0 + 1) >> 1) - q1 - d) >> 1).clamp(-(tc >> 1), tc >> 1);
                set(1, q1 + dq);
            }
        }
    }
}

/// One four-line segment of a chroma edge (§8.7.2.5.5). Chroma never gets the
/// strong filter and never moves more than one sample a side.
fn chroma_segment(plane: &mut [u8], w: u32, x: u32, y: u32, step: isize, along: isize, qp_c: i32) {
    let tc = TC[(qp_c + 2).clamp(0, 53) as usize];
    if tc == 0 {
        return;
    }
    let base = (y * w + x) as isize;
    for line in 0..4isize {
        let o = base + line * along;
        let s = |i: isize| plane[(o + i * step) as usize] as i32;
        let (p0, p1) = (s(-1), s(-2));
        let (q0, q1) = (s(0), s(1));
        let d = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        plane[(o - step) as usize] = clip1(p0 + d);
        plane[o as usize] = clip1(q0 - d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_map(cw: u32, ch: u32, tu: u32) -> EdgeMap {
        let mut m = EdgeMap::new(cw, ch);
        for y in (0..ch).step_by(tu as usize) {
            for x in (0..cw).step_by(tu as usize) {
                m.mark(x, y, tu);
            }
        }
        m
    }

    /// Below `Q = 16` β is zero and the filter cannot fire, whatever the picture.
    #[test]
    fn low_qp_is_a_no_op() {
        let (cw, ch) = (64u32, 32u32);
        let mut y: Vec<u8> = (0..cw * ch).map(|i| ((i * 43) % 256) as u8).collect();
        let mut u = vec![70u8; (cw * ch / 4) as usize];
        let mut v = vec![190u8; (cw * ch / 4) as usize];
        let (y0, u0, v0) = (y.clone(), u.clone(), v.clone());
        deblock(&mut y, &mut u, &mut v, cw, ch, &full_map(cw, ch, 8), |_, _| 10, |q| q);
        assert_eq!(y, y0, "luma untouched");
        assert_eq!((u, v), (u0, v0), "chroma untouched");
    }

    /// Nothing may move where no transform block ends.
    ///
    /// This is the check that a 32×32 transform is not quietly sliced up by the
    /// 8×8 filter grid: with only the 32-sample edges marked, samples around
    /// x = 8, 16 and 24 have to come through untouched however sharp they are.
    #[test]
    fn only_marked_edges_are_filtered() {
        let (cw, ch) = (64u32, 32u32);
        let mut y: Vec<u8> = (0..cw * ch)
            .map(|i| if (i % cw) % 8 < 4 { 60u8 } else { 190u8 })
            .collect();
        let before = y.clone();
        let mut u = vec![128u8; (cw * ch / 4) as usize];
        let mut v = vec![128u8; (cw * ch / 4) as usize];
        deblock(&mut y, &mut u, &mut v, cw, ch, &full_map(cw, ch, 32), |_, _| 40, |q| q);
        for row in 0..ch {
            for col in 0..cw {
                let near_marked = col % 32 < 4 || col % 32 >= 29;
                if !near_marked {
                    let i = (row * cw + col) as usize;
                    assert_eq!(y[i], before[i], "({col},{row}) is not near a marked edge");
                }
            }
        }
    }

    /// A flat picture has no edge to find at any QP.
    #[test]
    fn flat_picture_survives_any_qp() {
        let (cw, ch) = (64u32, 64u32);
        for qp in [20, 30, 40, 51] {
            let mut y = vec![144u8; (cw * ch) as usize];
            let mut u = vec![88u8; (cw * ch / 4) as usize];
            let mut v = vec![201u8; (cw * ch / 4) as usize];
            deblock(&mut y, &mut u, &mut v, cw, ch, &full_map(cw, ch, 8), |_, _| qp, |q| q);
            assert!(y.iter().all(|&s| s == 144), "luma moved at qp {qp}");
            assert!(u.iter().all(|&s| s == 88), "cb moved at qp {qp}");
            assert!(v.iter().all(|&s| s == 201), "cr moved at qp {qp}");
        }
    }
}
