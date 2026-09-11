//! H.264 4×4 integer transform and quantization (ISO/IEC 14496-10 §8.5).
//!
//! H.264's transform is not a DCT approximation in the HEVC sense — it is a
//! deliberately tiny integer butterfly with no multiplies, and the scaling the
//! DCT would have applied is folded into the quantizer's `LevelScale` tables
//! instead. That is why [`dequant`] multiplies by a table rather than shifting:
//! it is undoing the transform's norm at the same time.
//!
//! The **inverse** path must be bit-exact — the decoder runs it, and the encoder
//! mirrors it to reconstruct the neighbours intra prediction reads. The forward
//! path is the encoder's own analysis step and only has to be its faithful
//! partner.
//!
//! Pure integer arithmetic, so native and `wasm32` agree.

/// `v` from Table 8-15 — the three distinct `normAdjust4x4` values per `qP % 6`.
///
/// Position `(i, j)` picks a column: the four "even/even" positions take `v[0]`,
/// the four "odd/odd" take `v[1]`, and the remaining eight take `v[2]`.
const NORM_ADJUST: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Encoder-side quantization multipliers, the customary inverses of
/// [`NORM_ADJUST`] (these are not normative — any encoder may pick its own).
const QUANT_MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// Which of the three position classes `(i, j)` falls into.
#[inline]
fn class(i: usize, j: usize) -> usize {
    match (i % 2, j % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// `LevelScale4x4(m, i, j)` with a flat weight scale (`Flat_4x4_16`).
#[inline]
fn level_scale(m: usize, i: usize, j: usize) -> i32 {
    16 * NORM_ADJUST[m][class(i, j)]
}

/// Forward 4×4 core transform: `Cf · X · Cfᵀ` (§8.5.12.1, the analysis partner).
///
/// `block` is row-major residual; the result is row-major coefficients.
pub fn forward4(block: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    // Rows.
    for r in 0..4 {
        let (a, b, c, d) = (
            block[r * 4],
            block[r * 4 + 1],
            block[r * 4 + 2],
            block[r * 4 + 3],
        );
        let (s0, s1, s2, s3) = (a + d, b + c, b - c, a - d);
        t[r * 4] = s0 + s1;
        t[r * 4 + 1] = 2 * s3 + s2;
        t[r * 4 + 2] = s0 - s1;
        t[r * 4 + 3] = s3 - 2 * s2;
    }
    // Columns.
    let mut o = [0i32; 16];
    for c in 0..4 {
        let (a, b, cc, d) = (t[c], t[4 + c], t[8 + c], t[12 + c]);
        let (s0, s1, s2, s3) = (a + d, b + cc, b - cc, a - d);
        o[c] = s0 + s1;
        o[4 + c] = 2 * s3 + s2;
        o[8 + c] = s0 - s1;
        o[12 + c] = s3 - 2 * s2;
    }
    o
}

/// Inverse 4×4 core transform (§8.5.12.2). Bit-exact to the spec, including the
/// final `(x + 32) >> 6`.
pub fn inverse4(coeff: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    // Rows.
    for r in 0..4 {
        let (d0, d1, d2, d3) = (
            coeff[r * 4],
            coeff[r * 4 + 1],
            coeff[r * 4 + 2],
            coeff[r * 4 + 3],
        );
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        t[r * 4] = e0 + e3;
        t[r * 4 + 1] = e1 + e2;
        t[r * 4 + 2] = e1 - e2;
        t[r * 4 + 3] = e0 - e3;
    }
    // Columns, then the common rounding shift.
    let mut o = [0i32; 16];
    for c in 0..4 {
        let (d0, d1, d2, d3) = (t[c], t[4 + c], t[8 + c], t[12 + c]);
        let e0 = d0 + d2;
        let e1 = d0 - d2;
        let e2 = (d1 >> 1) - d3;
        let e3 = d1 + (d3 >> 1);
        o[c] = (e0 + e3 + 32) >> 6;
        o[4 + c] = (e1 + e2 + 32) >> 6;
        o[8 + c] = (e1 - e2 + 32) >> 6;
        o[12 + c] = (e0 - e3 + 32) >> 6;
    }
    o
}

/// Bound on a dequantized coefficient, set by the *inverse transform*, not by
/// the coefficient's own range.
///
/// [`inverse4`]'s first stage forms sums like `d0 + d2 + d1 + d3/2`, and decoders
/// keep the coefficient block in 16 bits — ffmpeg writes those first-stage
/// results straight back into an `int16_t` array. So the constraint an encoder
/// must respect is that the *intermediates* fit, which with four coefficients per
/// row means roughly a quarter of the range each. A bound on `d` alone is not
/// enough: `|d| = 20800` is a perfectly representable coefficient whose row sum
/// is not, and the reconstruction then diverges from the decoder's by a wrap.
///
/// This binds only on residuals that line up with a high-gain basis function —
/// near-adversarial content — so it costs nothing on anything natural.
const MAX_DEQUANT: i64 = 32767 / 4;

/// Largest level magnitude at `(i, j)` whose dequantized coefficient respects
/// [`MAX_DEQUANT`].
fn max_level(qp: i32, i: usize, j: usize) -> i64 {
    let ls = level_scale((qp % 6) as usize, i, j) as i64;
    let shift = qp / 6;
    if shift >= 4 {
        // d = level * ls << (shift - 4)
        (MAX_DEQUANT >> (shift - 4)) / ls
    } else {
        // d = (level * ls + rounding) >> (4 - shift)
        ((MAX_DEQUANT << (4 - shift)) - (1 << (3 - shift))) / ls
    }
}

/// Quantize a 4×4 coefficient block. `intra` selects the rounding offset — a
/// third of the step for intra, a sixth for inter, which is the usual deadzone.
///
/// Levels are clamped so their dequantized value stays representable; see
/// [`max_level`].
pub fn quant4(coeff: &[i32; 16], qp: i32, intra: bool) -> [i32; 16] {
    let m = (qp % 6) as usize;
    let qbits = 15 + qp / 6;
    let f = (1i64 << qbits) / if intra { 3 } else { 6 };
    let mut out = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let k = i * 4 + j;
            let c = coeff[k] as i64;
            let level =
                ((c.abs() * QUANT_MF[m][class(i, j)] as i64 + f) >> qbits).min(max_level(qp, i, j));
            out[k] = if c < 0 { -(level as i32) } else { level as i32 };
        }
    }
    out
}

/// Dequantize a 4×4 block of levels (§8.5.12.1). Bit-exact — the decoder runs
/// exactly this.
pub fn dequant4(levels: &[i32; 16], qp: i32) -> [i32; 16] {
    let m = (qp % 6) as usize;
    let shift = qp / 6;
    let mut out = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let k = i * 4 + j;
            let ls = level_scale(m, i, j);
            out[k] = if shift >= 4 {
                (levels[k] * ls) << (shift - 4)
            } else {
                let add = 1 << (3 - shift);
                (levels[k] * ls + add) >> (4 - shift)
            };
        }
    }
    out
}

/// Scale a set of four chroma DC levels down until their reconstruction fits.
///
/// A per-level bound is the wrong shape here. The four levels go through an
/// inverse Hadamard before scaling, so what has to stay in range is the *output*
/// — and a single large level is fine where four aligned ones are not. Bounding
/// each level by the four-aligned worst case would clamp a solid colour's DC to
/// a twentieth of what it needs.
///
/// So this checks the actual reconstruction and only scales if it genuinely
/// overflows, which for real content it never does.
pub fn fit_chroma_dc_levels(levels: &mut [i32; 4], qp: i32) {
    for _ in 0..24 {
        let worst = chroma_dc_inverse(levels, qp)
            .iter()
            .map(|v| (*v as i64).abs())
            .max()
            .unwrap_or(0);
        if worst <= MAX_DEQUANT {
            return;
        }
        for l in levels.iter_mut() {
            *l -= l.signum();
        }
    }
}

/// 4×4 Hadamard over the sixteen luma DC coefficients of an `I_16x16`
/// macroblock (§8.5.10). `H` is symmetric and `H·H = 4I`, so the same routine
/// serves both directions.
///
/// `dc[i * 4 + j]` is the DC of the 4×4 block at row `i`, column `j` of the
/// macroblock — raster, not the z-order the blocks are *coded* in.
pub fn luma_dc_hadamard(dc: &[i32; 16]) -> [i32; 16] {
    // The four Walsh rows in the order the spec lists them — sum, low-frequency
    // pair, alternating, high-frequency pair. Reordering them keeps the matrix
    // symmetric and orthogonal, so it survives every round-trip check while
    // still permuting fifteen of the sixteen DCs.
    let bfly = |a: i32, b: i32, c: i32, d: i32| {
        let (s01, d01, s23, d23) = (a + b, a - b, c + d, c - d);
        (s01 + s23, s01 - s23, d01 - d23, d01 + d23)
    };
    let mut t = [0i32; 16];
    for r in 0..4 {
        let (o0, o1, o2, o3) = bfly(dc[r * 4], dc[r * 4 + 1], dc[r * 4 + 2], dc[r * 4 + 3]);
        t[r * 4] = o0;
        t[r * 4 + 1] = o1;
        t[r * 4 + 2] = o2;
        t[r * 4 + 3] = o3;
    }
    let mut o = [0i32; 16];
    for c in 0..4 {
        let (o0, o1, o2, o3) = bfly(t[c], t[4 + c], t[8 + c], t[12 + c]);
        o[c] = o0;
        o[4 + c] = o1;
        o[8 + c] = o2;
        o[12 + c] = o3;
    }
    o
}

/// Inverse luma-DC transform and its scaling (§8.5.10).
///
/// Returns the DC each of the sixteen 4×4 blocks starts from, already scaled —
/// the caller drops it into position 0 before the core inverse, exactly as with
/// chroma DC.
pub fn luma_dc_inverse(levels: &[i32; 16], qp: i32) -> [i32; 16] {
    let f = luma_dc_hadamard(levels);
    let ls = level_scale((qp % 6) as usize, 0, 0);
    let shift = qp / 6;
    let mut out = [0i32; 16];
    for k in 0..16 {
        out[k] = if shift >= 6 {
            (f[k] * ls) << (shift - 6)
        } else {
            (f[k] * ls + (1 << (5 - shift))) >> (6 - shift)
        };
    }
    out
}

/// Scale a set of sixteen luma DC levels down until their reconstruction fits,
/// for the same reason as [`fit_chroma_dc_levels`].
pub fn fit_luma_dc_levels(levels: &mut [i32; 16], qp: i32) {
    for _ in 0..64 {
        let worst = luma_dc_inverse(levels, qp)
            .iter()
            .map(|v| (*v as i64).abs())
            .max()
            .unwrap_or(0);
        if worst <= MAX_DEQUANT {
            return;
        }
        for l in levels.iter_mut() {
            *l -= l.signum();
        }
    }
}

/// 2×2 Hadamard for the chroma DC coefficients (§8.5.11.1, forward direction).
pub fn chroma_dc_forward(dc: &[i32; 4]) -> [i32; 4] {
    let (a, b, c, d) = (dc[0], dc[1], dc[2], dc[3]);
    [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d]
}

/// Inverse 2×2 chroma DC Hadamard plus its scaling (§8.5.11.2).
///
/// Returns the DC value each of the four 4×4 chroma blocks starts from, already
/// scaled — the caller drops it into position 0 before the core inverse.
pub fn chroma_dc_inverse(levels: &[i32; 4], qp: i32) -> [i32; 4] {
    let (a, b, c, d) = (levels[0], levels[1], levels[2], levels[3]);
    let f = [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d];
    let m = (qp % 6) as usize;
    let ls = level_scale(m, 0, 0);
    let mut out = [0i32; 4];
    for k in 0..4 {
        // §8.5.11.2: dcC = ((f * LevelScale(qP%6,0,0)) << (qP/6)) >> 5
        out[k] = ((f[k] * ls) << (qp / 6)) >> 5;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_of_zero_is_zero() {
        assert_eq!(inverse4(&[0; 16]), [0; 16]);
        assert_eq!(quant4(&[0; 16], 26, true), [0; 16]);
        assert_eq!(dequant4(&[0; 16], 26), [0; 16]);
    }

    /// A DC-only block reconstructs to a flat field — the transform's most basic
    /// property, and a check that the `>> 6` normalisation is in the right place.
    #[test]
    fn dc_only_reconstructs_flat() {
        let mut c = [0i32; 16];
        c[0] = 64 * 16; // 16 in each sample after the >>6
        let r = inverse4(&c);
        assert!(r.iter().all(|&v| v == r[0]), "not flat: {r:?}");
        assert_eq!(r[0], 16);
    }

    /// Quantize→dequantize→inverse must land near the original residual. Exactness
    /// is not on offer (that is the point of a quantizer), but a low QP should be
    /// close.
    #[test]
    fn roundtrip_is_close_at_low_qp() {
        let orig: [i32; 16] = [
            12, -7, 3, 0, -5, 9, -2, 1, 4, -1, 6, -3, 0, 2, -4, 8,
        ];
        let levels = quant4(&forward4(&orig), 10, true);
        let back = inverse4(&dequant4(&levels, 10));
        for (a, b) in orig.iter().zip(&back) {
            assert!((a - b).abs() <= 3, "drifted: {orig:?} vs {back:?}");
        }
    }

    #[test]
    fn higher_qp_zeroes_more() {
        let orig: [i32; 16] = [
            40, -18, 9, 2, -14, 22, -6, 3, 11, -4, 15, -8, 1, 5, -9, 20,
        ];
        let c = forward4(&orig);
        let low = quant4(&c, 12, true).iter().filter(|&&v| v != 0).count();
        let high = quant4(&c, 40, true).iter().filter(|&&v| v != 0).count();
        assert!(high <= low, "qp 40 kept {high}, qp 12 kept {low}");
    }

    /// Every level the quantizer can emit must dequantize inside the 16-bit
    /// range, at every QP and every position class.
    #[test]
    fn quantized_levels_never_overflow_dequantization() {
        for qp in 0..=51 {
            // Drive the quantizer with the largest coefficient the forward
            // transform can produce from 8-bit residuals, and then some.
            let coeff = [20000i32; 16];
            for &c in &[coeff, [-20000; 16], [32767; 16]] {
                let levels = quant4(&c, qp, true);
                let d = dequant4(&levels, qp);
                for (k, &v) in d.iter().enumerate() {
                    assert!(
                        (v as i64).abs() <= MAX_DEQUANT,
                        "qp {qp}, position {k}: dequantized to {v}, over the \
                         inverse-transform bound {MAX_DEQUANT}"
                    );
                }
                // The first stage of the inverse transform must fit 16 bits.
                for r in 0..4 {
                    let (d0, d1, d2, d3) = (d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]);
                    for v in [d0 + d2 + d1 + (d3 / 2), (d0 - d2) + ((d1 / 2) - d3)] {
                        assert!(
                            (-32768..=32767).contains(&v),
                            "qp {qp}, row {r}: inverse-transform intermediate {v}"
                        );
                    }
                }
            }
        }
    }

    /// `H·H = 4I` on each side, so applying the 4×4 Hadamard twice scales by 16.
    #[test]
    fn luma_dc_hadamard_is_an_involution_up_to_scale() {
        let dc: [i32; 16] = [
            5, -3, 2, 7, 11, -8, 0, 4, -6, 9, 1, -2, 3, 3, -5, 10,
        ];
        let twice = luma_dc_hadamard(&luma_dc_hadamard(&dc));
        for k in 0..16 {
            assert_eq!(twice[k], 16 * dc[k], "position {k}");
        }
    }

    /// Pin every entry of the matrix, not just its symmetry.
    ///
    /// A Walsh matrix stays symmetric *and* orthogonal when its rows are
    /// permuted, so `H·H = 4I` and "flat stays flat" both hold for the wrong
    /// one. A permuted butterfly passed both and still shifted fifteen of the
    /// sixteen block DCs — visible only to an external decoder. This checks the
    /// transform of every impulse against `o[i][j] = H[i][r]·H[c][j]`, which
    /// admits exactly one matrix.
    #[test]
    fn luma_dc_matrix_is_the_one_in_the_spec() {
        const H: [[i32; 4]; 4] = [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
            [1, -1, 1, -1],
        ];
        for r in 0..4 {
            for c in 0..4 {
                let mut imp = [0i32; 16];
                imp[r * 4 + c] = 1;
                let got = luma_dc_hadamard(&imp);
                for i in 0..4 {
                    for j in 0..4 {
                        assert_eq!(
                            got[i * 4 + j],
                            H[i][r] * H[c][j],
                            "impulse at ({r},{c}), output ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    /// A flat set of DCs must stay flat through the transform and its scaling.
    #[test]
    fn luma_dc_flat_stays_flat() {
        let f = luma_dc_hadamard(&[7; 16]);
        assert_eq!(f[0], 16 * 7, "DC of a flat field");
        assert!(f[1..].iter().all(|&v| v == 0), "no AC from a flat field");
        let rec = luma_dc_inverse(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 28);
        assert!(rec.iter().all(|&v| v == rec[0]), "flat DC level -> flat output");
    }

    #[test]
    fn chroma_dc_hadamard_is_its_own_shape() {
        // The 2x2 Hadamard applied twice scales by 4.
        let dc = [5i32, -3, 2, 7];
        let f = chroma_dc_forward(&dc);
        let again = chroma_dc_forward(&f);
        for k in 0..4 {
            assert_eq!(again[k], 4 * dc[k]);
        }
    }

    /// `LevelScale` must follow the position classes, not a single constant.
    #[test]
    fn level_scale_follows_position_class() {
        for (m, adj) in NORM_ADJUST.iter().enumerate() {
            assert_eq!(level_scale(m, 0, 0), 16 * adj[0]);
            assert_eq!(level_scale(m, 1, 1), 16 * adj[1]);
            assert_eq!(level_scale(m, 0, 1), 16 * adj[2]);
            assert_eq!(level_scale(m, 2, 0), 16 * adj[0]);
        }
    }
}

