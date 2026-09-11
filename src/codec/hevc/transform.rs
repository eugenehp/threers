//! HEVC core integer transforms (H.265 §8.6.4): the DCT-II approximations used
//! for residual coding, plus the alternate DST-VII used for 4×4 intra luma.
//!
//! The **inverse** transform must be bit-exact to the spec (the decoder runs it,
//! and the encoder mirrors it to reconstruct neighbors for intra prediction).
//! The **forward** transform is the encoder's own analysis step; here it is the
//! matching orthogonal partner so `forward → inverse` round-trips.
//!
//! Sizes 4 and 8 are implemented (enough for a first compressed encoder with
//! transform blocks ≤ 8×8); 16/32 follow the same nested-matrix pattern.
//!
//! Pure integer math over slices — identical on native and `wasm32`.

/// 4×4 DCT-II basis (`transMatrix`, H.265 Table in §8.6.4.2).
pub const DCT4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 4×4 DST-VII basis (used for 4×4 intra luma residual).
pub const DST4: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// 16×16 DCT-II basis. Rows `2i` (cols 0..8) reproduce [`DCT8`] (nested design).
pub const DCT16: [[i32; 16]; 16] = [
    [
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
    ],
    [
        89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    ],
    [
        87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
    ],
    [
        83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
    ],
    [
        80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
    ],
    [
        75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    ],
    [
        70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
    ],
    [
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    ],
    [
        57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
    ],
    [
        50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    ],
    [
        43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
    ],
    [
        36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
    ],
    [
        25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
    ],
    [
        18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
    ],
    [
        9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
    ],
];

/// 8×8 DCT-II basis.
pub const DCT8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// Which transform to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    /// DCT-II — all sizes.
    Dct,
    /// DST-VII — 4×4 only (intra luma).
    Dst,
}

/// Row 1 of the 32-point matrix, first half — the only 32-point data taken from
/// the spec by hand.
///
/// Everything else in [`DCT32`] follows from it and from [`DCT16`]: the even
/// rows *are* the 16-point rows (the HEVC matrices nest exactly), and the odd
/// rows are this vector re-indexed and sign-flipped. Sixteen numbers to get
/// right instead of a thousand and twenty-four.
const G32: [i32; 16] = [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4];

/// The 32-point DCT, constructed rather than transcribed. See [`G32`].
pub const DCT32: [[i32; 32]; 32] = build_dct32();

const fn build_dct32() -> [[i32; 32]; 32] {
    let mut m = [[0i32; 32]; 32];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 16 {
            // Even row 2i is DCT16 row i, mirrored about the centre — with a
            // plus sign, because the mirror's sign is (−1) to the power of the
            // *row index*, and 2i is even whatever i is.
            m[2 * i][j] = DCT16[i][j];
            m[2 * i][31 - j] = DCT16[i][j];

            // Odd row 2i+1 samples the generator at angle (2i+1)(2j+1)π/128,
            // folded back into the first quadrant.
            let k = ((2 * i + 1) * (2 * j + 1)) % 128;
            let (s, idx): (i32, usize) = if k < 32 {
                (1, (k - 1) / 2)
            } else if k < 64 {
                (-1, (63 - k) / 2)
            } else if k < 96 {
                (-1, (k - 65) / 2)
            } else {
                (1, (127 - k) / 2)
            };
            m[2 * i + 1][j] = s * G32[idx];
            m[2 * i + 1][31 - j] = -s * G32[idx];
            j += 1;
        }
        i += 1;
    }
    m
}

const DCT32_ROWS: &[&[i32]] = &[
    &DCT32[0],
    &DCT32[1],
    &DCT32[2],
    &DCT32[3],
    &DCT32[4],
    &DCT32[5],
    &DCT32[6],
    &DCT32[7],
    &DCT32[8],
    &DCT32[9],
    &DCT32[10],
    &DCT32[11],
    &DCT32[12],
    &DCT32[13],
    &DCT32[14],
    &DCT32[15],
    &DCT32[16],
    &DCT32[17],
    &DCT32[18],
    &DCT32[19],
    &DCT32[20],
    &DCT32[21],
    &DCT32[22],
    &DCT32[23],
    &DCT32[24],
    &DCT32[25],
    &DCT32[26],
    &DCT32[27],
    &DCT32[28],
    &DCT32[29],
    &DCT32[30],
    &DCT32[31],
];

fn matrix(size: usize, kind: TransformKind) -> &'static [&'static [i32]] {
    // Return rows as slices via small static wrappers.
    match (size, kind) {
        (4, TransformKind::Dst) => DST4_ROWS,
        (4, TransformKind::Dct) => DCT4_ROWS,
        (8, TransformKind::Dct) => DCT8_ROWS,
        (16, TransformKind::Dct) => DCT16_ROWS,
        (32, TransformKind::Dct) => DCT32_ROWS,
        _ => panic!("unsupported transform size {size}/{kind:?}"),
    }
}

const DCT16_ROWS: &[&[i32]] = &[
    &DCT16[0], &DCT16[1], &DCT16[2], &DCT16[3], &DCT16[4], &DCT16[5], &DCT16[6], &DCT16[7],
    &DCT16[8], &DCT16[9], &DCT16[10], &DCT16[11], &DCT16[12], &DCT16[13], &DCT16[14], &DCT16[15],
];

const DCT4_ROWS: &[&[i32]] = &[&DCT4[0], &DCT4[1], &DCT4[2], &DCT4[3]];
const DST4_ROWS: &[&[i32]] = &[&DST4[0], &DST4[1], &DST4[2], &DST4[3]];
const DCT8_ROWS: &[&[i32]] = &[
    &DCT8[0], &DCT8[1], &DCT8[2], &DCT8[3], &DCT8[4], &DCT8[5], &DCT8[6], &DCT8[7],
];

#[inline]
fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

#[inline]
fn clip16(v: i64) -> i64 {
    v.clamp(-32768, 32767)
}

/// Forward 2D transform of an `n×n` residual block (row-major), producing
/// coefficients (row-major). `n` ∈ {4, 8, 16, 32}. Shifts assume 8-bit samples.
pub fn forward(residual: &[i32], n: usize, kind: TransformKind) -> Vec<i32> {
    let mut out = vec![0i32; n * n];
    forward_into(residual, &mut out, n, kind);
    out
}

/// The largest transform this encoder emits, and so the size of the scratch the
/// two passes below need.
pub const MAX_N: usize = 32;

/// [`forward`] writing into a caller-supplied buffer.
///
/// Identical arithmetic — the results are bit-for-bit what `forward` returns —
/// but it allocates nothing and both passes walk memory forwards.
///
/// The second pass consumes *columns* of the first pass's output. Writing that
/// output transposed turns a stride-`n` gather into a contiguous read, which at
/// 32×32 is the difference between touching 32 cache lines per dot product and
/// touching one.
pub fn forward_buf(residual: &[i32], n: usize, kind: TransformKind, out: &mut Vec<i32>) {
    out.clear();
    out.resize(n * n, 0);
    forward_into(residual, out, n, kind);
}

/// [`inverse_into`] with a caller-owned buffer it resizes.
pub fn inverse_buf(coeff: &[i32], n: usize, kind: TransformKind, out: &mut Vec<i32>) {
    out.clear();
    out.resize(n * n, 0);
    inverse_into(coeff, out, n, kind);
}

#[inline]
fn dot(a: &[i32], b: &[i32]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// One 1-D forward pass: `acc[0..n] = M · src`.
///
/// For the DCT this halves the multiplies without changing a single result.
/// The matrix's rows are alternately symmetric and antisymmetric about their
/// centre — `M[2i][n-1-k] = M[2i][k]` and `M[2i+1][n-1-k] = -M[2i+1][k]`, which
/// `dct32_dc_row_is_flat_and_the_matrix_is_centre_symmetric` pins — so
///
/// ```text
/// sum_k M[2i][k]·src[k]   ==  sum_{k<n/2} M[2i][k]·(src[k] + src[n-1-k])
/// sum_k M[2i+1][k]·src[k] ==  sum_{k<n/2} M[2i+1][k]·(src[k] - src[n-1-k])
/// ```
///
/// as an identity over the integers, before any rounding. DST-VII has no such
/// symmetry and takes the direct path; it only ever runs at 4×4.
#[inline]
fn forward_pass(m: &[&[i32]], src: &[i32], n: usize, butterfly: bool, acc: &mut [i64]) {
    if butterfly {
        let half = n / 2;
        let mut e = [0i32; MAX_N / 2];
        let mut o = [0i32; MAX_N / 2];
        for k in 0..half {
            e[k] = src[k] + src[n - 1 - k];
            o[k] = src[k] - src[n - 1 - k];
        }
        for i in 0..half {
            acc[2 * i] = dot(&m[2 * i][..half], &e[..half]);
            acc[2 * i + 1] = dot(&m[2 * i + 1][..half], &o[..half]);
        }
    } else {
        for (i, row) in m.iter().enumerate().take(n) {
            acc[i] = dot(&row[..n], src);
        }
    }
}

pub fn forward_into(residual: &[i32], out: &mut [i32], n: usize, kind: TransformKind) {
    debug_assert!(n <= MAX_N);
    let m = matrix(n, kind);
    let shift1 = log2(n) + 8 - 9; // = log2(n) - 1
    let shift2 = log2(n) + 6;
    let add1 = 1i64 << (shift1.max(1) - 1);
    let add2 = 1i64 << (shift2 - 1);

    let butterfly = matches!(kind, TransformKind::Dct);
    let mut acc = [0i64; MAX_N];

    // Stage 1: transform rows, storing the result transposed.
    //
    // `i32` rather than the `i64` this used to hold: the residual is a
    // difference of 8-bit samples, so before the shift the largest a row sum can
    // reach is 90 · 255 · 32, comfortably inside 32 bits. Narrowing halves the
    // traffic through stage 2.
    let mut tmp = [0i32; MAX_N * MAX_N];
    for r in 0..n {
        forward_pass(m, &residual[r * n..r * n + n], n, butterfly, &mut acc);
        for i in 0..n {
            tmp[i * n + r] = ((acc[i] + add1) >> shift1) as i32;
        }
    }
    // Stage 2: transform columns, which are now rows of `tmp`.
    for c in 0..n {
        forward_pass(m, &tmp[c * n..c * n + n], n, butterfly, &mut acc);
        for i in 0..n {
            out[i * n + c] = ((acc[i] + add2) >> shift2) as i32;
        }
    }
}

/// Inverse 2D transform of an `n×n` coefficient block (row-major) back to a
/// residual (row-major). Bit-exact to H.265 §8.6.4.2 for 8-bit samples
/// (`bdShift = 20 − BitDepth = 12`).
pub fn inverse(coeff: &[i32], n: usize, kind: TransformKind) -> Vec<i32> {
    let mut out = vec![0i32; n * n];
    inverse_into(coeff, &mut out, n, kind);
    out
}

/// [`inverse`] writing into a caller-supplied buffer. See [`forward_into`].
pub fn inverse_into(coeff: &[i32], out: &mut [i32], n: usize, kind: TransformKind) {
    debug_assert!(n <= MAX_N);
    let m = matrix(n, kind);
    let shift1 = 7i64;
    let shift2 = 20 - 8; // 12 for 8-bit
    let add1 = 1i64 << (shift1 - 1);
    let add2 = 1i64 << (shift2 - 1);

    // Stage 1: inverse-transform columns, clipped to 16 bits, stored transposed
    // so stage 2 reads forwards.
    let mut tmp = [0i32; MAX_N * MAX_N];
    for c in 0..n {
        for y in 0..n {
            let mut s = 0i64;
            for (k, row) in m.iter().enumerate().take(n) {
                s += i64::from(row[y]) * i64::from(coeff[k * n + c]);
            }
            tmp[y * n + c] = clip16((s + add1) >> shift1) as i32;
        }
    }
    // Stage 2: inverse-transform rows. `m` is indexed `[k][x]`, so this walks a
    // column of the matrix; the *coefficients* are what stage 1 made contiguous.
    for r in 0..n {
        let t = &tmp[r * n..r * n + n];
        for x in 0..n {
            let mut s = 0i64;
            for (k, row) in m.iter().enumerate().take(n) {
                s += i64::from(row[x]) * i64::from(t[k]);
            }
            out[r * n + x] = ((s + add2) >> shift2) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Rebuild an `n`-point matrix from the `n/2`-point one plus its own first
    /// odd row, the same construction [`build_dct32`] uses.
    fn construct(n: usize, prev: &[&[i32]], g: &[i32]) -> Vec<Vec<i32>> {
        let mut m = vec![vec![0i32; n]; n];
        for i in 0..n / 2 {
            for j in 0..n / 2 {
                m[2 * i][j] = prev[i][j];
                m[2 * i][n - 1 - j] = prev[i][j];
                let k = ((2 * i + 1) * (2 * j + 1)) % (4 * n);
                let (s, idx): (i32, usize) = if k < n {
                    (1, (k - 1) / 2)
                } else if k < 2 * n {
                    (-1, (2 * n - k - 1) / 2)
                } else if k < 3 * n {
                    (-1, (k - 2 * n - 1) / 2)
                } else {
                    (1, (4 * n - k - 1) / 2)
                };
                m[2 * i + 1][j] = s * g[idx];
                m[2 * i + 1][n - 1 - j] = -s * g[idx];
            }
        }
        m
    }

    /// The 32-point matrix is generated, not transcribed, so what needs proving
    /// is the generator — and that can be done against tables an external
    /// decoder has already agreed with.
    ///
    /// Feeding it the 4-point matrix must yield the 8-point one exactly, and the
    /// 8-point must yield the 16-point. Both of those are conformance-tested, so
    /// a generator that reproduces them is sound, and `DCT32` then rests on the
    /// sixteen numbers in `G32` alone.
    #[test]
    fn the_matrix_generator_reproduces_the_known_tables() {
        let d4: Vec<&[i32]> = DCT4.iter().map(|r| &r[..]).collect();
        let g8: Vec<i32> = DCT8[1][..4].to_vec();
        assert_eq!(construct(8, &d4, &g8), rows_of(&DCT8), "8-point from 4-point");

        let d8: Vec<&[i32]> = DCT8.iter().map(|r| &r[..]).collect();
        let g16: Vec<i32> = DCT16[1][..8].to_vec();
        assert_eq!(
            construct(16, &d8, &g16),
            rows_of(&DCT16),
            "16-point from 8-point"
        );

        let d16: Vec<&[i32]> = DCT16.iter().map(|r| &r[..]).collect();
        assert_eq!(
            construct(32, &d16, &G32),
            rows_of(&DCT32),
            "32-point matches the const-built table"
        );
    }

    fn rows_of<const N: usize>(m: &[[i32; N]; N]) -> Vec<Vec<i32>> {
        m.iter().map(|r| r.to_vec()).collect()
    }

    /// The DC row is what every flat block rides on: all 64s, no exceptions.
    #[test]
    fn dct32_dc_row_is_flat_and_the_matrix_is_centre_symmetric() {
        assert!(DCT32[0].iter().all(|&v| v == 64), "row 0 is the DC row");
        for (i, row) in DCT32.iter().enumerate() {
            let sign = if i % 2 == 0 { 1 } else { -1 };
            for j in 0..16 {
                assert_eq!(row[31 - j], sign * row[j], "row {i}, column {j}");
            }
        }
    }

    #[test]
    fn dct4_rows_orthogonal() {
        // Distinct DCT rows are orthogonal; a wrong table entry breaks this.
        for i in 0..4 {
            for j in 0..4 {
                let dot: i32 = (0..4).map(|k| DCT4[i][k] * DCT4[j][k]).sum();
                if i != j {
                    assert_eq!(dot, 0, "DCT4 rows {i},{j} not orthogonal");
                } else {
                    assert!(dot > 0);
                }
            }
        }
    }

    #[test]
    fn dct8_rows_near_orthogonal() {
        // The integer DCT is only *near*-orthogonal: same-parity row pairs can dot
        // to ±50 (a known artifact), but a transcription error would blow far past
        // that. Even↔odd pairs are exactly 0 by symmetry.
        for i in 0..8 {
            for j in 0..8 {
                let dot: i32 = (0..8).map(|k| DCT8[i][k] * DCT8[j][k]).sum();
                if i != j {
                    assert!(dot.abs() <= 64, "DCT8 rows {i},{j} dot={dot} too large");
                    if (i + j) % 2 == 1 {
                        assert_eq!(dot, 0, "opposite-parity DCT8 rows {i},{j} must be exact");
                    }
                } else {
                    assert!(dot > 30000, "DCT8 row {i} norm too small: {dot}");
                }
            }
        }
    }

    fn roundtrip(n: usize, kind: TransformKind, block: &[i32]) {
        let coeff = forward(block, n, kind);
        let recon = inverse(&coeff, n, kind);
        let mut max_err = 0;
        for k in 0..n * n {
            max_err = i32::max(max_err, (recon[k] - block[k]).abs());
        }
        assert!(
            max_err <= 2,
            "roundtrip err {max_err} for size {n} {kind:?}"
        );
    }

    #[test]
    fn dct_roundtrips() {
        // A ramp and an impulse recover through forward∘inverse (± rounding).
        let b4: Vec<i32> = (0..16).map(|i| (i * 7 % 40) - 20).collect();
        roundtrip(4, TransformKind::Dct, &b4);
        roundtrip(4, TransformKind::Dst, &b4);
        let b8: Vec<i32> = (0..64).map(|i| ((i * 13) % 60) - 30).collect();
        roundtrip(8, TransformKind::Dct, &b8);
        let b16: Vec<i32> = (0..256).map(|i| ((i * 17) % 80) - 40).collect();
        roundtrip(16, TransformKind::Dct, &b16);
    }

    #[test]
    fn dct16_nests_dct8() {
        // Nested design: DCT16 even rows (cols 0..8) equal DCT8 rows.
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(DCT16[2 * i][j], DCT8[i][j], "DCT16 nesting at {i},{j}");
            }
        }
    }

    #[test]
    fn dc_only_is_flat() {
        // A constant residual transforms to a single DC coefficient.
        let flat = vec![10i32; 16];
        let coeff = forward(&flat, 4, TransformKind::Dct);
        assert!(coeff[0].abs() > 0, "DC present");
        for (i, &c) in coeff.iter().enumerate().skip(1) {
            assert_eq!(c, 0, "non-DC coeff {i} should be ~0 for flat input");
        }
    }
}
