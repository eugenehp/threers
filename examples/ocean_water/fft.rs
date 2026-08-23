//! A radix-2 FFT, and the butterfly table that drives it on the GPU.
//!
//! The GPU kernel and [`ifft_1d`] here run the *same* loop over the *same*
//! precomputed table. That is the point of this module: a butterfly index off by
//! one produces plausible-looking noise rather than an error, and debugging that
//! through a texture is miserable. So the table is built and exercised on the CPU
//! against a naive DFT, and the shader is a transcription of a function that is
//! already known to be correct.
//!
//! # The transform
//!
//! Iterative Cooley–Tukey, decimation in time. For stage `s` (1-based) and output
//! index `x`, with `m = 1 << s` and `j = x % m`:
//!
//! ```text
//! out[x] = buf[a] + w^j * buf[b],   w = exp(sign * 2*pi*i / m)
//! a = if j < m/2 { x } else { x - m/2 },   b = a + m/2
//! ```
//!
//! Both wings of the butterfly use the *same* expression, because
//! `w^j = -w^(j - m/2)` for `j >= m/2` — the subtraction is already in the
//! twiddle. That is what lets one table entry drive every element.
//!
//! Bit reversal is folded into stage 1's indices rather than run as its own pass.

/// A complex number as `[re, im]`.
#[cfg(test)]
pub type C = [f32; 2];

#[cfg(test)]
#[inline]
pub fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

#[cfg(test)]
#[inline]
pub fn cadd(a: C, b: C) -> C {
    [a[0] + b[0], a[1] + b[1]]
}

pub fn log2_exact(n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n > 1,
        "FFT size must be a power of two"
    );
    n.trailing_zeros() as usize
}

fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// One butterfly table entry: the twiddle, and the two elements to combine.
///
/// Laid out as `[w.re, w.im, a, b]` so it uploads straight to an `Rgba32Float`
/// texture of `log2(n)` rows by `n` columns, which is how the GPU reads it.
pub fn butterfly_table(n: usize, sign: f32) -> Vec<[f32; 4]> {
    let stages = log2_exact(n);
    let mut out = Vec::with_capacity(stages * n);
    for s in 1..=stages {
        let m = 1usize << s;
        let half = m / 2;
        for x in 0..n {
            let j = x % m;
            let (a, b) = if j < half {
                (x, x + half)
            } else {
                (x - half, x)
            };
            // Stage 1 reads the bit-reversed array, so it does the permutation.
            let (a, b) = if s == 1 {
                (reverse_bits(a, stages), reverse_bits(b, stages))
            } else {
                (a, b)
            };
            let theta = sign * std::f32::consts::TAU * j as f32 / m as f32;
            out.push([theta.cos(), theta.sin(), a as f32, b as f32]);
        }
    }
    out
}

/// Run the transform the table describes. `sign` must match the table's.
///
/// This is the reference the GPU kernel is a transcription of. It only ever runs
/// under test: at runtime the table goes to the GPU and this function's job is
/// already done — it is the thing that proved the table right.
#[cfg(test)]
pub fn fft_1d_with_table(buf: &mut [C], table: &[[f32; 4]]) {
    let n = buf.len();
    let stages = log2_exact(n);
    let mut src = buf.to_vec();
    let mut dst = vec![[0.0f32; 2]; n];
    for s in 0..stages {
        let row = &table[s * n..(s + 1) * n];
        for x in 0..n {
            let e = row[x];
            let w = [e[0], e[1]];
            let a = src[e[2] as usize];
            let b = src[e[3] as usize];
            dst[x] = cadd(a, cmul(w, b));
        }
        std::mem::swap(&mut src, &mut dst);
    }
    buf.copy_from_slice(&src);
}

/// Naive DFT, for testing only. `O(n^2)`, and deliberately written straight from
/// the definition so it shares no code with the thing it checks.
#[cfg(test)]
pub fn dft_naive(input: &[C], sign: f32) -> Vec<C> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut acc = [0.0f32; 2];
            for (x, v) in input.iter().enumerate() {
                let th = sign * std::f32::consts::TAU * (k * x % n) as f32 / n as f32;
                acc = cadd(acc, cmul(*v, [th.cos(), th.sin()]));
            }
            acc
        })
        .collect()
}

/// Cross-check against rlx's own FFT, which this project already depends on.
///
/// A naive DFT is an independent oracle but it is *my* independent oracle, and
/// it shares my conventions about sign and normalisation. rlx's is a third-party
/// implementation with its own; agreeing with both is a much stronger statement
/// that the table is right than agreeing with one.
///
/// rlx cannot drive the per-frame ocean, which is why this is a test rather than
/// the implementation — see the module docs on `ocean_fft`.
#[cfg(all(test, feature = "rlx"))]
mod rlx_oracle {
    use super::*;
    use ::rlx::{DType, Device, Graph, Shape};
    // threers's runner, not rlx's own Session: it is the one with the
    // flat-slice entry point, which is all this needs.
    use threers::rlx::GraphRunner;

    /// rlx lays complex data out as `[..., 2N]`: the whole real plane, then the
    /// whole imaginary plane. Ours is interleaved, so the two have to be
    /// transposed across the boundary.
    fn run_rlx_fft(input: &[C], inverse: bool) -> Vec<C> {
        let n = input.len();
        let mut g = Graph::new("fft");
        let x = g.input("x", Shape::new(&[1, 2 * n], DType::F32));
        let y = g.fft(x, inverse);
        g.set_outputs(vec![y]);

        let mut planar: Vec<f32> = input.iter().map(|c| c[0]).collect();
        planar.extend(input.iter().map(|c| c[1]));

        let mut runner = GraphRunner::new(g, Device::Cpu);
        let out = runner.run_flat(&[("x", &planar)]).remove(0);
        (0..n).map(|i| [out[i], out[n + i]]).collect()
    }

    /// Measured, not assumed: can rlx drive the ocean per frame?
    #[test]
    #[ignore = "timing probe, run explicitly"]
    fn rlx_throughput_probe() {
        // One axis of one cascade: 256 rows of 256 complex.
        let n = 256usize;
        let mut g = Graph::new("fft2d_rows");
        let x = g.input("x", Shape::new(&[n, 2 * n], DType::F32));
        let y = g.fft(x, true);
        g.set_outputs(vec![y]);
        let mut runner = GraphRunner::new(g, ::rlx::Device::Cpu);
        let data = vec![0.0f32; n * 2 * n];

        runner.run_flat(&[("x", &data)]);
        let t = std::time::Instant::now();
        const K: u32 = 20;
        for _ in 0..K {
            runner.run_flat(&[("x", &data)]);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / K as f64;
        // A cascade needs two of these (rows then columns) per complex field,
        // two fields, and there are three cascades: twelve passes a frame.
        println!(
            "rlx cpu fft {n}x{n} one axis: {ms:.2} ms  ->  ~{:.1} ms/frame for 3 cascades",
            ms * 12.0
        );
    }

    #[test]
    fn butterfly_table_agrees_with_rlx() {
        for &n in &[8usize, 64, 256] {
            for &inverse in &[false, true] {
                let mut seed = 4242u32;
                let mut next = || {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
                };
                let input: Vec<C> = (0..n).map(|_| [next(), next()]).collect();

                // rlx's inverse is unnormalised, same as ours. Its forward
                // transform uses e^(-i...), so `inverse` maps to sign +1.
                let expect = run_rlx_fft(&input, inverse);
                let sign = if inverse { 1.0 } else { -1.0 };
                let mut got = input.clone();
                fft_1d_with_table(&mut got, &butterfly_table(n, sign));

                for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                    let err = ((g[0] - e[0]).powi(2) + (g[1] - e[1]).powi(2)).sqrt();
                    assert!(
                        err < 1e-3 * n as f32,
                        "n={n} inverse={inverse} i={i} err={err}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    }

    #[test]
    fn table_driven_fft_matches_naive_dft() {
        for &n in &[2usize, 4, 8, 16, 64, 256] {
            for &sign in &[1.0f32, -1.0] {
                let mut seed = 12345u32;
                let input: Vec<C> = (0..n).map(|_| [lcg(&mut seed), lcg(&mut seed)]).collect();

                let expect = dft_naive(&input, sign);
                let mut got = input.clone();
                fft_1d_with_table(&mut got, &butterfly_table(n, sign));

                for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                    let err = ((g[0] - e[0]).powi(2) + (g[1] - e[1]).powi(2)).sqrt();
                    // Error grows with n as the sums get longer; scale the bound.
                    assert!(err < 1e-3 * n as f32, "n={n} sign={sign} i={i} err={err}");
                }
            }
        }
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let n = 64;
        let mut seed = 987u32;
        let input: Vec<C> = (0..n).map(|_| [lcg(&mut seed), lcg(&mut seed)]).collect();

        let mut buf = input.clone();
        fft_1d_with_table(&mut buf, &butterfly_table(n, -1.0));
        fft_1d_with_table(&mut buf, &butterfly_table(n, 1.0));

        // Round trip picks up the usual factor of n.
        for (b, i) in buf.iter().zip(&input) {
            assert!((b[0] / n as f32 - i[0]).abs() < 1e-4);
            assert!((b[1] / n as f32 - i[1]).abs() < 1e-4);
        }
    }

    #[test]
    fn bit_reversal_is_an_involution() {
        for bits in 1..=8 {
            for x in 0..(1usize << bits) {
                assert_eq!(reverse_bits(reverse_bits(x, bits), bits), x);
            }
        }
    }

    #[test]
    fn table_indices_stay_in_range() {
        let n = 128;
        let t = butterfly_table(n, 1.0);
        assert_eq!(t.len(), log2_exact(n) * n);
        for e in &t {
            assert!((e[2] as usize) < n && (e[3] as usize) < n);
        }
    }
}
