//! av_tx's `mdct_pfa_15xM_inv_float_c` (libavutil n7.1 `tx_template.c`):
//! the half-length inverse MDCT CELT runs at 120/240/480/960 coefficients.
//! A 15-point PFA (`fft3` x5 into `fft5_m1..m3`, C, unfused) feeds fifteen
//! power-of-two sub-FFTs, which on aarch64 are the NEON codelets
//! (`fft4_fwd`, `fft8_ns`, `fft16_ns`, `fft32_ns`), emulated bit for bit.

use crate::aac::tx_neon;

pub struct Mdct15 {
    len: usize,
    m: usize,
    /// Doubled Ruritanian input map (`len/2`), embedded PFA map included.
    in_map: Vec<usize>,
    /// CRT output map (`len/2`).
    out_map: Vec<usize>,
    /// The sub-FFT's scatter map (`m`).
    sub_map: Vec<usize>,
    /// `s->exp`: `len/2` pre-twiddles (in input-map order), then `len/2` natural.
    exp: Vec<(f32, f32)>,
    tab: [f32; 12],
}

fn tab53() -> [f32; 12] {
    use std::f64::consts::PI;
    let c = |x: f64| x as f32;
    [
        c((2.0 * PI / 5.0).cos()),
        c((2.0 * PI / 5.0).cos()),
        c((2.0 * PI / 10.0).cos()),
        c((2.0 * PI / 10.0).cos()),
        c((2.0 * PI / 5.0).sin()),
        c((2.0 * PI / 5.0).sin()),
        c((2.0 * PI / 10.0).sin()),
        c((2.0 * PI / 10.0).sin()),
        c((2.0 * PI / 12.0).cos()),
        c((2.0 * PI / 12.0).cos()),
        c((2.0 * PI / 6.0).cos()),
        c((8.0 * PI / 6.0).cos()),
    ]
}

fn mulinv(n: usize, m: usize) -> usize {
    let n = n % m;
    (1..m).find(|x| (n * x) % m == 1).unwrap()
}

#[derive(Clone, Copy, Default)]
struct C {
    re: f32,
    im: f32,
}

impl Mdct15 {
    /// `av_tx_init(AV_TX_FLOAT_MDCT, inv = 1, len, scale)`.
    pub fn new(len: usize, scale: f32) -> Self {
        let half = len / 2;
        let n = 15;
        let m = half / n;
        // ff_tx_gen_compound_mapping(s, NULL opts -> gather, inv = 1, 15, m)
        let (m_inv, n_inv) = (mulinv(m, n), mulinv(n, m));
        let mut in_map = vec![0usize; half];
        let mut out_map = vec![0usize; half];
        for j in 0..m {
            for i in 0..n {
                in_map[j * n + i] = (i * m + j * n) % half;
                out_map[(i * m * m_inv + j * n * n_inv) % half] = i * m + j;
            }
        }
        for i in 0..m {
            let base = i * n + 1;
            for j in 0..(n - 1) / 2 {
                in_map.swap(base + j, base + n - j - 2);
            }
        }
        // TX_EMBED_INPUT_PFA_MAP(map, len/2, 3, 5)
        for k in (0..half).step_by(15) {
            let t: Vec<usize> = in_map[k..k + 15].to_vec();
            for mm in 0..5 {
                for nn in 0..3 {
                    in_map[k + mm * 3 + nn] = t[(mm * 3 + nn * 5) % 15];
                }
            }
        }
        // ff_tx_mdct_gen_exp(s, pre_tab = map)
        let scale_d = scale as f64;
        let theta = if scale_d < 0.0 { half as f64 } else { 0.0 } + 1.0 / 8.0;
        let sc = scale_d.abs().sqrt();
        let natural: Vec<(f32, f32)> = (0..half)
            .map(|i| {
                let alpha = std::f64::consts::FRAC_PI_2 * (i as f64 + theta) / half as f64;
                ((alpha.cos() * sc) as f32, (alpha.sin() * sc) as f32)
            })
            .collect();
        let mut exp: Vec<(f32, f32)> = in_map.iter().map(|&k| natural[k]).collect();
        exp.extend_from_slice(&natural);
        for k in in_map.iter_mut() {
            *k <<= 1;
        }
        let sub_map = tx_neon::split_radix_parity_revtab_scatter(m, true);
        Self { len, m, in_map, out_map, sub_map, exp, tab: tab53() }
    }

    /// `src[k * stride]` are the `len` coefficients; `dst` gets `len` samples.
    pub fn run(&self, dst: &mut [f32], src: &[f32], stride: usize, tmp: &mut Vec<f32>) {
        let t = self.tab;
        let half = self.len / 2;
        let len4 = self.len / 4;
        let m = self.m;
        tmp.clear();
        tmp.resize(self.len, 0.0);
        let in2 = (self.len - 1) * stride;
        let mut fin = [C::default(); 15];
        for (g, i) in (0..half).step_by(15).enumerate() {
            for j in 0..15 {
                let k = self.in_map[i + j];
                let (are, aim) = (src[in2 - k * stride], src[k * stride]);
                let (bre, bim) = self.exp[i + j];
                fin[j] = C { re: are * bre - aim * bim, im: are * bim + aim * bre };
            }
            fft15(tmp, 2 * self.sub_map[g], 2 * m, &fin, &t);
        }
        for i in 0..15 {
            tx_neon::fft_sr_ns(&mut tmp[2 * m * i..2 * m * (i + 1)], m);
        }
        let exp = &self.exp[half..];
        for i in 0..len4 {
            let (i0, i1) = (len4 + i, len4 - i - 1);
            let (s0, s1) = (self.out_map[i0], self.out_map[i1]);
            let src1 = (tmp[2 * s1 + 1], tmp[2 * s1]);
            let src0 = (tmp[2 * s0 + 1], tmp[2 * s0]);
            // CMUL(z[i1].re, z[i0].im, src1.re, src1.im, exp[i1].im, exp[i1].re)
            let (bre, bim) = (exp[i1].1, exp[i1].0);
            dst[2 * i1] = src1.0 * bre - src1.1 * bim;
            dst[2 * i0 + 1] = src1.0 * bim + src1.1 * bre;
            let (bre, bim) = (exp[i0].1, exp[i0].0);
            dst[2 * i0] = src0.0 * bre - src0.1 * bim;
            dst[2 * i1 + 1] = src0.0 * bim + src0.1 * bre;
        }
    }
}

/// `fft3(out, in, stride)`.
#[inline]
fn fft3(inp: &[C], t: &[f32; 12]) -> [C; 3] {
    let t0 = inp[0];
    let mut t1 = C { re: inp[1].im - inp[2].im, im: inp[1].re - inp[2].re };
    let mut t2 = C { re: inp[1].re + inp[2].re, im: inp[1].im + inp[2].im };
    let o0 = C { re: t0.re + t2.re, im: t0.im + t2.im };
    t1.re *= t[8];
    t1.im *= t[9];
    t2.re *= t[10];
    t2.im *= t[10];
    let o1 = C { re: t0.re - t2.re + t1.re, im: t0.im - t2.im - t1.im };
    let o2 = C { re: t0.re - t2.re - t1.re, im: t0.im - t2.im + t1.im };
    [o0, o1, o2]
}

/// `DECL_FFT5(NAME, D0..D4)`: returns the outputs for slots D0..D4.
#[inline]
fn fft5(inp: &[C], t: &[f32; 12]) -> [C; 5] {
    let dc = inp[0];
    let mut t0 = C { re: inp[1].re + inp[4].re, im: inp[1].im + inp[4].im };
    let mut t1 = C { im: inp[1].re - inp[4].re, re: inp[1].im - inp[4].im };
    let t2 = C { re: inp[2].re + inp[3].re, im: inp[2].im + inp[3].im };
    let t3 = C { im: inp[2].re - inp[3].re, re: inp[2].im - inp[3].im };
    let o0 = C { re: dc.re + t0.re + t2.re, im: dc.im + t0.im + t2.im };
    // SMUL(t4.x, t0.x, tab0, tab2, t2.x, t0.x)
    let t4 = C { re: t[0] * t2.re - t[2] * t0.re, im: t[0] * t2.im - t[2] * t0.im };
    t0 = C { re: t[0] * t0.re - t[2] * t2.re, im: t[0] * t0.im - t[2] * t2.im };
    // CMUL(t5.x, t1.x, tab4, tab6, t3.x, t1.x)
    let t5 = C { re: t[4] * t3.re - t[6] * t1.re, im: t[4] * t3.im - t[6] * t1.im };
    t1 = C { re: t[4] * t1.re + t[6] * t3.re, im: t[4] * t1.im + t[6] * t3.im };
    let z0 = C { re: t0.re - t1.re, im: t0.im - t1.im };
    let z3 = C { re: t0.re + t1.re, im: t0.im + t1.im };
    let z2 = C { re: t4.re - t5.re, im: t4.im - t5.im };
    let z1 = C { re: t4.re + t5.re, im: t4.im + t5.im };
    [
        o0,
        C { re: dc.re + z3.re, im: dc.im + z0.im },
        C { re: dc.re + z2.re, im: dc.im + z1.im },
        C { re: dc.re + z1.re, im: dc.im + z2.im },
        C { re: dc.re + z0.re, im: dc.im + z3.im },
    ]
}

/// `fft15(out, in, stride)`, writing complex `q` at `out[base + q * stride]`
/// (both in floats).
fn fft15(out: &mut [f32], base: usize, stride: usize, inp: &[C; 15], t: &[f32; 12]) {
    let mut tmp = [C::default(); 15];
    for i in 0..5 {
        let r = fft3(&inp[i * 3..i * 3 + 3], t);
        tmp[i] = r[0];
        tmp[i + 5] = r[1];
        tmp[i + 10] = r[2];
    }
    const D: [[usize; 5]; 3] = [[0, 6, 12, 3, 9], [10, 1, 7, 13, 4], [5, 11, 2, 8, 14]];
    for (b, d) in D.iter().enumerate() {
        let r = fft5(&tmp[b * 5..b * 5 + 5], t);
        for (k, &slot) in d.iter().enumerate() {
            out[base + slot * stride] = r[k].re;
            out[base + slot * stride + 1] = r[k].im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the direct half-IMDCT definition av_tx implements:
    /// `y[n] = scale * sum_k X[k] cos(pi/N (n + 1/2 + N/2)(k + 1/2))` over
    /// the middle half of the window (N = len).
    #[test]
    fn imdct15_matches_definition() {
        for len in [120usize, 240, 480, 960] {
            let x: Vec<f32> = (0..len).map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5).collect();
            let t = Mdct15::new(len, -1.0 / 32768.0);
            let mut y = vec![0f32; len];
            let mut tmp = Vec::new();
            t.run(&mut y, &x, 1, &mut tmp);
            // Compare with a power-of-two-free O(N^2) reference.
            let n = len as f64;
            let mut worst = 0f64;
            for (i, &yi) in y.iter().enumerate() {
                let nn = i as f64 + n / 2.0;
                let mut s = 0f64;
                for (k, &xk) in x.iter().enumerate() {
                    s += xk as f64 * (std::f64::consts::PI / n * (nn + 0.5 + n / 2.0) * (k as f64 + 0.5)).cos();
                }
                let r = s / 32768.0; // av_tx: -scale times the textbook IMDCT
                worst = worst.max((r - yi as f64).abs());
            }
            let peak = 1.0 / 32768.0 * n.sqrt();
            assert!(worst < peak * 1e-5, "len {len}: worst {worst:e} vs scale {peak:e}");
        }
    }
}
