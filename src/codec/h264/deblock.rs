//! The H.264 in-loop deblocking filter (§8.7), for all-intra pictures.
//!
//! Block-transform coding leaves discontinuities at block edges, and at the
//! rates where those become visible they are the most objectionable artefact in
//! the picture. The filter smooths them, with a strength that depends on the
//! quantizer — at low QP the thresholds are zero and nothing happens at all.
//!
//! It is called *in-loop* because a decoder filters before storing a picture as
//! a reference, so an encoder that does not reproduce it bit-exactly drifts
//! from the decoder. That is not a hazard here yet: intra prediction reads the
//! samples *before* the filter (§8.3), and this encoder has no inter
//! prediction, so filtering is a single pass over the finished picture rather
//! than something the reconstruction loop has to interleave with.
//!
//! Scope: one slice per picture, every macroblock intra, filter offsets zero.
//! That fixes the boundary strength at 4 on macroblock edges and 3 inside them
//! (§8.7.2.1) — the whole of the inter derivation, with its motion-vector and
//! reference-index comparisons, is unreachable.

use crate::codec::h264::deblock_tables::{ALPHA, BETA, TC0};
use crate::codec::h264::params::MB_SIZE;

/// Boundary strength on an edge between two intra blocks: 4 across a macroblock
/// edge, 3 within one.
const BS_MB_EDGE: usize = 4;
const BS_INTERNAL: usize = 3;

#[inline]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Filter the whole picture in place.
///
/// `mb_qp` is the decoder's `QP_Y` per macroblock in raster order, and `qp_c`
/// maps a luma QP to its chroma counterpart. Macroblocks are filtered in
/// raster order, and within each one every vertical edge before any horizontal
/// one, each step reading what the previous step wrote — the order is
/// normative, not an implementation choice.
pub fn deblock(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    cw: u32,
    ch: u32,
    mb_qp: &[i32],
    qp_c: impl Fn(i32) -> i32,
) {
    let (mbw, mbh) = ((cw / MB_SIZE) as usize, (ch / MB_SIZE) as usize);
    let cwc = cw / 2;
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let qp_here = mb_qp[mby * mbw + mbx];
            let qp_left = (mbx > 0).then(|| mb_qp[mby * mbw + mbx - 1]);
            let qp_above = (mby > 0).then(|| mb_qp[(mby - 1) * mbw + mbx]);
            let (ox, oy) = ((mbx * MB_SIZE as usize) as u32, (mby * MB_SIZE as usize) as u32);

            // --- luma, vertical edges left to right ---
            for e in 0..4u32 {
                let (bs, qp_p) = match (e, qp_left) {
                    (0, None) => continue, // picture edge
                    (0, Some(q)) => (BS_MB_EDGE, q),
                    _ => (BS_INTERNAL, qp_here),
                };
                let x = ox + e * 4;
                filter_edge(
                    y, cw, x, oy, 1, cw as isize, MB_SIZE, bs, qp_p, qp_here, false,
                );
            }
            // --- luma, horizontal edges top to bottom ---
            for e in 0..4u32 {
                let (bs, qp_p) = match (e, qp_above) {
                    (0, None) => continue,
                    (0, Some(q)) => (BS_MB_EDGE, q),
                    _ => (BS_INTERNAL, qp_here),
                };
                let yy = oy + e * 4;
                filter_edge(
                    y, cw, ox, yy, cw as isize, 1, MB_SIZE, bs, qp_p, qp_here, false,
                );
            }

            // --- chroma, both planes, the same two edges each way ---
            // A chroma edge takes its boundary strength from the luma edge it
            // sits on, so in 4:2:0 only the macroblock edge and the middle one
            // exist, at strengths 4 and 3.
            let (cx, cy) = (ox / 2, oy / 2);
            for plane in [&mut *u, &mut *v] {
                for e in 0..2u32 {
                    let (bs, qp_p) = match (e, qp_left) {
                        (0, None) => continue,
                        (0, Some(q)) => (BS_MB_EDGE, q),
                        _ => (BS_INTERNAL, qp_here),
                    };
                    filter_edge(
                        plane,
                        cwc,
                        cx + e * 4,
                        cy,
                        1,
                        cwc as isize,
                        8,
                        bs,
                        qp_c(qp_p),
                        qp_c(qp_here),
                        true,
                    );
                }
                for e in 0..2u32 {
                    let (bs, qp_p) = match (e, qp_above) {
                        (0, None) => continue,
                        (0, Some(q)) => (BS_MB_EDGE, q),
                        _ => (BS_INTERNAL, qp_here),
                    };
                    filter_edge(
                        plane,
                        cwc,
                        cx,
                        cy + e * 4,
                        cwc as isize,
                        1,
                        8,
                        bs,
                        qp_c(qp_p),
                        qp_c(qp_here),
                        true,
                    );
                }
            }
        }
    }
}

/// Filter `len` sample lines across one edge.
///
/// `step` is the stride *across* the edge — 1 for a vertical edge, the plane
/// width for a horizontal one — and `along` the stride between successive lines
/// of it. `(x, y)` is the position of `q0` on the first line.
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    plane: &mut [u8],
    w: u32,
    x: u32,
    y: u32,
    step: isize,
    along: isize,
    len: u32,
    bs: usize,
    qp_p: i32,
    qp_q: i32,
    chroma: bool,
) {
    // Both thresholds key off the average of the two quantizers, so a hard QP
    // change across a macroblock edge softens the filter on both sides equally.
    let idx = ((qp_p + qp_q + 1) >> 1).clamp(0, 51) as usize;
    let (alpha, beta) = (ALPHA[idx], BETA[idx]);
    if alpha == 0 {
        return;
    }
    let base = (y * w + x) as isize;
    for k in 0..len as isize {
        let o = base + k * along;
        let at = |i: isize| plane[(o + i * step) as usize] as i32;
        let (p0, p1, p2, p3) = (at(-1), at(-2), at(-3), at(-4));
        let (q0, q1, q2, q3) = (at(0), at(1), at(2), at(3));

        // The edge is only filtered where it looks like a coding artefact
        // rather than real structure: a small step across it, flanked by two
        // flat sides (§8.7.2.2).
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let set = |plane: &mut [u8], i: isize, val: u8| plane[(o + i * step) as usize] = val;

        if bs < 4 {
            // §8.7.2.3
            let tc0 = TC0[idx][bs];
            let tc = if chroma {
                tc0 + 1
            } else {
                tc0 + i32::from(ap < beta) + i32::from(aq < beta)
            };
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            set(plane, -1, clip1(p0 + delta));
            set(plane, 0, clip1(q0 - delta));
            if !chroma && ap < beta {
                let d = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
                set(plane, -2, clip1(p1 + d));
            }
            if !chroma && aq < beta {
                let d = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
                set(plane, 1, clip1(q1 + d));
            }
        } else {
            // §8.7.2.4 — across a macroblock edge, where the step is most
            // visible, up to three samples a side may move.
            let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
            if !chroma && ap < beta && strong {
                set(plane, -1, clip1((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3));
                set(plane, -2, clip1((p2 + p1 + p0 + q0 + 2) >> 2));
                set(plane, -3, clip1((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3));
            } else {
                set(plane, -1, clip1((2 * p1 + p0 + q1 + 2) >> 2));
            }
            if !chroma && aq < beta && strong {
                set(plane, 0, clip1((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3));
                set(plane, 1, clip1((q2 + q1 + q0 + p0 + 2) >> 2));
                set(plane, 2, clip1((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3));
            } else {
                set(plane, 0, clip1((2 * q1 + q0 + p1 + 2) >> 2));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Below `indexA = 16` every threshold is zero, so the filter must leave the
    /// picture untouched however blocky it is.
    #[test]
    fn low_qp_is_a_no_op() {
        let (cw, ch) = (32u32, 32u32);
        let mut y: Vec<u8> = (0..cw * ch).map(|i| ((i * 37) % 256) as u8).collect();
        let mut u = vec![100u8; (cw * ch / 4) as usize];
        let mut v = vec![160u8; (cw * ch / 4) as usize];
        let (y0, u0, v0) = (y.clone(), u.clone(), v.clone());
        deblock(&mut y, &mut u, &mut v, cw, ch, &[10; 4], |q| q);
        assert_eq!(y, y0, "luma untouched at qp 10");
        assert_eq!((u, v), (u0, v0), "chroma untouched at qp 10");
    }

    /// A flat picture has no edges to find, so nothing may move at any QP —
    /// this is the check that catches a filter reaching outside its block and
    /// pulling in a neighbour it should not have.
    #[test]
    fn flat_picture_survives_any_qp() {
        let (cw, ch) = (48u32, 32u32);
        for qp in [20, 30, 40, 51] {
            let mut y = vec![137u8; (cw * ch) as usize];
            let mut u = vec![90u8; (cw * ch / 4) as usize];
            let mut v = vec![210u8; (cw * ch / 4) as usize];
            deblock(&mut y, &mut u, &mut v, cw, ch, &[qp; 6], |q| q);
            assert!(y.iter().all(|&s| s == 137), "luma moved at qp {qp}");
            assert!(u.iter().all(|&s| s == 90), "cb moved at qp {qp}");
            assert!(v.iter().all(|&s| s == 210), "cr moved at qp {qp}");
        }
    }

    /// A step at a macroblock edge is what the filter exists for: it must
    /// shrink, and only near the edge.
    #[test]
    fn a_macroblock_edge_step_is_smoothed_locally() {
        let (cw, ch) = (32u32, 16u32);
        let mut y: Vec<u8> = (0..cw * ch)
            .map(|i| if (i % cw) < 16 { 100u8 } else { 130u8 })
            .collect();
        let before = y.clone();
        let mut u = vec![128u8; (cw * ch / 4) as usize];
        let mut v = vec![128u8; (cw * ch / 4) as usize];
        deblock(&mut y, &mut u, &mut v, cw, ch, &[36; 2], |q| q);
        let row = 8 * cw as usize;
        let step_before = (before[row + 16] as i32 - before[row + 15] as i32).abs();
        let step_after = (y[row + 16] as i32 - y[row + 15] as i32).abs();
        assert!(
            step_after < step_before,
            "step {step_before} -> {step_after} at the macroblock edge"
        );
        // Four samples either side is the filter's whole reach.
        for k in 0..11 {
            assert_eq!(y[row + k], before[row + k], "column {k} is out of reach");
        }
        for k in 20..32 {
            assert_eq!(y[row + k], before[row + k], "column {k} is out of reach");
        }
    }
}
