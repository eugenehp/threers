//! Numeric kernels of ffmpeg n7.1's float AAC decoder.
//!
//! Tables are built at first use exactly as libavcodec builds them at init
//! (`ff_cbrt_tableinit`, `ff_kbd_window_init`, `ff_sine_window_init`,
//! `aac_tableinit`), in the same precision and operation order.
//!
//! The vector kernels follow libavutil's aarch64 NEON `float_dsp` (the
//! reference build): `vector_fmul_window` and `scalarproduct_float` fuse
//! their multiply-adds. The inverse MDCT is av_tx's (`ff_tx_mdct_inv` around
//! the NEON split-radix FFT, [`super::tx_neon`]). C code in the reference
//! build is not FMA-contracted.

// Constants are ffmpeg's, digit for digit.
#![allow(clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop)]

use std::sync::OnceLock;

pub struct Tables {
    /// `ff_cbrt_tab`: `n^(4/3)` as f32 bit patterns.
    pub cbrt: Vec<u32>,
    pub kbd_long: Vec<f32>,
    pub kbd_short: Vec<f32>,
    pub sine_long: Vec<f32>,
    pub sine_short: Vec<f32>,
    /// `ff_aac_pow2sf_tab`: `2^((i - 200) / 4)`.
    pub pow2sf: Vec<f32>,
    pub imdct_long: Imdct,
    pub imdct_short: Imdct,
}

pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        cbrt: cbrt_table(),
        kbd_long: kbd_window(4.0, 1024),
        kbd_short: kbd_window(6.0, 128),
        sine_long: sine_window(1024),
        sine_short: sine_window(128),
        pow2sf: pow2sf_table(),
        // MDCT_INIT(..., len, 1.0/len): scale_float = (1.0/len) / 32768.0f.
        imdct_long: Imdct::new(1024, ((1.0 / 1024.0) / 32768.0f64) as f32),
        imdct_short: Imdct::new(128, ((1.0 / 128.0) / 32768.0f64) as f32),
    })
}

/// `ff_cbrt_tableinit`: products of `p * cbrt(p)` over the prime factors, in
/// double, then rounded to float.
fn cbrt_table() -> Vec<u32> {
    const N: usize = 1 << 13;
    let mut t = vec![1f64; N];
    t[0] = 0.0;
    for i in 2..90 {
        if t[i] == 1.0 {
            let v = i as f64 * (i as f64).cbrt();
            let mut k = i;
            while k < N {
                let mut j = k;
                while j < N {
                    t[j] *= v;
                    j += k;
                }
                k *= i;
            }
        }
    }
    let mut i = 91;
    while i <= 8191 {
        if t[i] == 1.0 {
            let v = i as f64 * (i as f64).cbrt();
            let mut j = i;
            while j < N {
                t[j] *= v;
                j += i;
            }
        }
        i += 2;
    }
    t.iter().map(|&v| (v as f32).to_bits()).collect()
}

fn eval_poly(c: &[f64], x: f64) -> f64 {
    let mut sum = c[c.len() - 1];
    for &ci in c[..c.len() - 1].iter().rev() {
        sum *= x;
        sum += ci;
    }
    sum
}

/// `av_bessel_i0` (Blair & Edwards minimax rationals).
pub(crate) fn bessel_i0(x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.2335582639474375249e+15, -5.5050369673018427753e+14, -3.2940087627407749166e+13,
        -8.4925101247114157499e+11, -1.1912746104985237192e+10, -1.0313066708737980747e+08,
        -5.9545626019847898221e+05, -2.4125195876041896775e+03, -7.0935347449210549190e+00,
        -1.5453977791786851041e-02, -2.5172644670688975051e-05, -3.0517226450451067446e-08,
        -2.6843448573468483278e-11, -1.5982226675653184646e-14, -5.2487866627945699800e-18,
    ];
    const Q1: [f64; 6] = [
        -2.2335582639474375245e+15, 7.8858692566751002988e+12, -1.2207067397808979846e+10,
        1.0377081058062166144e+07, -4.8527560179962773045e+03, 1.0,
    ];
    const P2: [f64; 7] = [
        -2.2210262233306573296e-04, 1.3067392038106924055e-02, -4.4700805721174453923e-01,
        5.5674518371240761397e+00, -2.3517945679239481621e+01, 3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    ];
    const Q2: [f64; 8] = [
        -5.5194330231005480228e-04, 3.2547697594819615062e-02, -1.1151759188741312645e+00,
        1.3982595353892851542e+01, -6.0228002066743340583e+01, 8.5539563258012929600e+01,
        -3.1446690275135491500e+01, 1.0,
    ];
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        x.exp() / x.sqrt() * r
    }
}

/// `kbd_window_init` (float alpha, double accumulation).
fn kbd_window(alpha: f32, n: usize) -> Vec<f32> {
    let a = alpha as f64 * std::f64::consts::PI / n as f64;
    let alpha2 = 4.0 * a * a;
    let mut temp = vec![0f64; n / 2 + 1];
    let mut scale = 0f64;
    for (i, t) in temp.iter_mut().enumerate() {
        *t = bessel_i0(((i * (n - i)) as f64 * alpha2).sqrt());
        scale += *t * (1 + (i != 0 && i < n / 2) as i32) as f64;
    }
    let scale = 1.0 / (scale + 1.0);
    let mut w = vec![0f32; n];
    let mut sum = 0f64;
    for i in 0..n {
        sum += if i <= n / 2 { temp[i] } else { temp[n - i] };
        w[i] = (sum * scale).sqrt() as f32;
    }
    w
}

/// `ff_sine_window_init`: `sinf((i + 0.5) * (M_PI / (2.0 * n)))`.
fn sine_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as f64 + 0.5) * (std::f64::consts::PI / (2.0 * n as f64))) as f32).sin())
        .collect()
}

/// `aac_tableinit`'s `ff_aac_pow2sf_tab` recurrence.
fn pow2sf_table() -> Vec<f32> {
    const EXP2_LUT: [f32; 16] = [
        1.00000000000000000000, 1.04427378242741384032, 1.09050773266525765921,
        1.13878863475669165370, 1.18920711500272106672, 1.24185781207348404859,
        1.29683955465100966593, 1.35425554693689272830, 1.41421356237309504880,
        1.47682614593949931139, 1.54221082540794082361, 1.61049033194925430818,
        1.68179283050742908606, 1.75625216037329948311, 1.83400808640934246349,
        1.91520656139714729387,
    ];
    let mut t1: f32 = 8.8817841970012523233890533447265625e-16;
    let mut prev = 0;
    (0..428)
        .map(|i| {
            let cur = 4 * (i % 4);
            if cur < prev {
                t1 *= 2.0;
            }
            prev = cur;
            t1 * EXP2_LUT[cur]
        })
        .collect()
}

/// av_tx's float inverse MDCT (`ff_tx_mdct_inv`, `libavutil/tx_template.c`)
/// of `n` coefficients without `AV_TX_FULL_IMDCT`: `n` outputs (the middle
/// half of the `2n`-sample IMDCT). Pre-rotation, an `n/2`-point in-place
/// pre-shuffled FFT (the aarch64 NEON codelet, [`super::tx_neon`]), and
/// post-rotation, with av_tx's twiddles and `CMUL` spelling.
pub struct Imdct {
    n: usize,
    /// Doubled sub-FFT input map.
    map: Vec<usize>,
    /// `s->exp`: `n/2` permuted pre-twiddles, then `n/2` natural ones.
    exp: Vec<(f32, f32)>,
}

/// `CMUL(dre, dim, are, aim, bre, bim)`, uncontracted: the reference build
/// rounds every product (measured: plain matches ffmpeg 7.1 bit for bit on
/// all test media; either FMA spelling leaves ~40% of samples 1 ulp off).
#[inline]
fn cmul(are: f32, aim: f32, bre: f32, bim: f32) -> (f32, f32) {
    (are * bre - aim * bim, are * bim + aim * bre)
}

impl Imdct {
    /// `ff_tx_mdct_init` + `ff_tx_mdct_gen_exp` for an inverse transform.
    pub(crate) fn new(n: usize, scale: f32) -> Self {
        let half = n / 2;
        let sub_map = super::tx_neon::split_radix_parity_revtab(half, true);
        let scale_d = scale as f64;
        let theta = if scale_d < 0.0 { half as f64 } else { 0.0 } + 1.0 / 8.0;
        let sc = scale_d.abs().sqrt();
        let natural: Vec<(f32, f32)> = (0..half)
            .map(|i| {
                let alpha = std::f64::consts::FRAC_PI_2 * (i as f64 + theta) / half as f64;
                ((alpha.cos() * sc) as f32, (alpha.sin() * sc) as f32)
            })
            .collect();
        let mut exp: Vec<(f32, f32)> = sub_map.iter().map(|&m| natural[m]).collect();
        exp.extend_from_slice(&natural);
        Self { n, map: sub_map.iter().map(|&m| m << 1).collect(), exp }
    }

    pub fn run(&self, input: &[f32], out: &mut [f32]) {
        let len2 = self.n / 2;
        let len4 = self.n / 4;
        let z = &mut out[..self.n];
        for i in 0..len2 {
            let k = self.map[i];
            let (tre, tim) = (input[self.n - 1 - k], input[k]);
            let (ere, eim) = self.exp[i];
            let (r, m) = cmul(tre, tim, ere, eim);
            z[2 * i] = r;
            z[2 * i + 1] = m;
        }
        super::tx_neon::fft_sr_ns(z, len2);
        let exp = &self.exp[len2..];
        for i in 0..len4 {
            let (i0, i1) = (len4 + i, len4 - i - 1);
            let src1 = (z[2 * i1 + 1], z[2 * i1]);
            let src0 = (z[2 * i0 + 1], z[2 * i0]);
            let (a, b) = cmul(src1.0, src1.1, exp[i1].1, exp[i1].0);
            z[2 * i1] = a;
            z[2 * i0 + 1] = b;
            let (a, b) = cmul(src0.0, src0.1, exp[i0].1, exp[i0].0);
            z[2 * i0] = a;
            z[2 * i1 + 1] = b;
        }
    }
}

/// `ff_vector_fmul_window_neon`:
/// `dst[i] = s0*wj - s1*wi` (product, then fused subtract) and
/// `dst[j] = s1*wj + s0*wi` (fused add onto the rounded `s0*wi`).
pub fn vector_fmul_window(dst: &mut [f32], src0: &[f32], src1: &[f32], win: &[f32], len: usize) {
    for k in 0..len {
        let (i, j) = (k, 2 * len - 1 - k);
        let s0 = src0[k];
        let s1 = src1[len - 1 - k];
        let wi = win[i];
        let wj = win[j];
        dst[i] = (-s1).mul_add(wi, s0 * wj);
        dst[j] = s1.mul_add(wj, s0 * wi);
    }
}

/// `ff_scalarproduct_float_neon`: four fused lanes, then pairwise adds.
pub fn scalarproduct(a: &[f32]) -> f32 {
    let mut l = [0f32; 4];
    for c in a.chunks(4) {
        for (k, &v) in c.iter().enumerate() {
            l[k] = v.mul_add(v, l[k]);
        }
    }
    (l[0] + l[1]) + (l[2] + l[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imdct_matches_direct_definition() {
        for n in [128usize, 1024] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.25).collect();
            let mut out = vec![0f32; n];
            Imdct::new(n, 1.0).run(&x, &mut out);
            // av_tx's sign is minus the textbook full IMDCT x[t] = sum X[k] cos(pi/n (t + 1/2 + n/2)(k + 1/2)); half = t in n/2..3n/2.
            for (j, &o) in out.iter().enumerate() {
                let t = (j + n / 2) as f64;
                let r: f64 = (0..n)
                    .map(|k| {
                        x[k] as f64
                            * (std::f64::consts::PI / n as f64 * (t + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5)).cos()
                    })
                    .sum();
                assert!((o as f64 + r).abs() < 1e-3 * (1.0 + r.abs()), "n={n} j={j}: {o} vs {r}");
            }
        }
    }

    #[test]
    fn pow2sf_is_exact_powers() {
        let t = tables();
        assert_eq!(t.pow2sf[200], 1.0);
        assert_eq!(t.pow2sf[204], 2.0);
        assert_eq!(f32::from_bits(t.cbrt[8]), 16.0);
    }
}
