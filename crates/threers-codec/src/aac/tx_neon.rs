//! libavutil n7.1's aarch64 NEON split-radix FFT (`ff_tx_fft_sr_ns_float_neon`,
//! `libavutil/aarch64/tx_float_neon.S`) transcribed instruction by
//! instruction over an emulated register file, for the in-place
//! pre-shuffled ("ns") transforms av_tx's MDCT runs as its sub-FFT.
//!
//! Every NEON arithmetic instruction is reproduced lane by lane in its own
//! rounding: `fadd`/`fsub`/`fmul` round once, `fmla`/`fmls` are fused. The
//! permutes (`rev64`, `ext`, `zip`, `trn`, `tbl`, ...) only move values.
//! Lengths 64..=1024 (powers of two) are supported.

// Constants are ffmpeg's, digit for digit.
#![allow(clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop)]

type R = usize; // vector register index

pub struct Neon<'a> {
    v: [[f32; 4]; 32],
    /// The transform buffer (interleaved complex), in floats.
    pub buf: &'a mut [f32],
    x1: usize,
    x2: usize,
    // SETUP_SR_RECOMB state: table (by length), re / im cursors in floats.
    tab: &'static [f32],
    x7: usize,
    x8: isize,
    x21: usize,
    x22: usize,
    w19: usize,
}

const SQ: f32 = 0.707106781186547524401;
const C1: f32 = 0.92387950420379638671875;
const C3: f32 = 0.3826834261417388916015625;
const SUBADD: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];

/// `ff_tx_tab_<len>_float`: `cos(2*pi*i/len)` for `i < len/4`, then 0.
fn cos_tab(len: usize) -> &'static [f32] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    let all = T.get_or_init(|| {
        (0..=20)
            .map(|b| {
                let len = 1usize << b;
                if len < 8 {
                    return Vec::new();
                }
                let freq = 2.0 * std::f64::consts::PI / len as f64;
                let mut t: Vec<f32> = (0..len / 4).map(|i| (i as f64 * freq).cos() as f32).collect();
                t.push(0.0);
                t
            })
            .collect()
    });
    &all[len.trailing_zeros() as usize]
}

impl Neon<'_> {
    fn fadd(&mut self, d: R, a: R, b: R) {
        let (a, b) = (self.v[a], self.v[b]);
        self.v[d] = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    }
    fn fsub(&mut self, d: R, a: R, b: R) {
        let (a, b) = (self.v[a], self.v[b]);
        self.v[d] = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    }
    fn fmul(&mut self, d: R, a: R, b: R) {
        let (a, b) = (self.v[a], self.v[b]);
        self.v[d] = [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]];
    }
    /// `d += a * b`, fused.
    fn fmla(&mut self, d: R, a: R, b: R) {
        let (a, b, c) = (self.v[a], self.v[b], self.v[d]);
        self.v[d] = [a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1]), a[2].mul_add(b[2], c[2]), a[3].mul_add(b[3], c[3])];
    }
    /// `d -= a * b`, fused.
    fn fmls(&mut self, d: R, a: R, b: R) {
        let (a, b, c) = (self.v[a], self.v[b], self.v[d]);
        self.v[d] = [(-a[0]).mul_add(b[0], c[0]), (-a[1]).mul_add(b[1], c[1]), (-a[2]).mul_add(b[2], c[2]), (-a[3]).mul_add(b[3], c[3])];
    }
    fn rev64(&mut self, d: R, a: R) {
        let a = self.v[a];
        self.v[d] = [a[1], a[0], a[3], a[2]];
    }
    /// `ext d.16b, a.16b, b.16b, #bytes`.
    fn ext(&mut self, d: R, a: R, b: R, bytes: usize) {
        let k = bytes / 4;
        let (va, vb) = (self.v[a], self.v[b]);
        let at = |i: usize| if i < 4 { va[i] } else { vb[i - 4] };
        self.v[d] = [at(k), at(k + 1), at(k + 2), at(k + 3)];
    }
    fn zip1d(&mut self, d: R, a: R, b: R) {
        let (a, b) = (self.v[a], self.v[b]);
        self.v[d] = [a[0], a[1], b[0], b[1]];
    }
    fn zip2d(&mut self, d: R, a: R, b: R) {
        let (a, b) = (self.v[a], self.v[b]);
        self.v[d] = [a[2], a[3], b[2], b[3]];
    }
    fn trn1s(&mut self, d: R, a: R) {
        let a = self.v[a];
        self.v[d] = [a[0], a[0], a[2], a[2]];
    }
    fn trn2s(&mut self, d: R, a: R) {
        let a = self.v[a];
        self.v[d] = [a[1], a[1], a[3], a[3]];
    }
    fn dupd(&mut self, d: R, a: R, half: usize) {
        let a = self.v[a];
        let (x, y) = (a[2 * half], a[2 * half + 1]);
        self.v[d] = [x, y, x, y];
    }
    /// `mov d.d[dh], a.d[ah]`.
    fn movd(&mut self, d: R, dh: usize, a: R, ah: usize) {
        let a = self.v[a];
        self.v[d][2 * dh] = a[2 * ah];
        self.v[d][2 * dh + 1] = a[2 * ah + 1];
    }
    fn mov(&mut self, d: R, a: R) {
        self.v[d] = self.v[a];
    }
    fn set(&mut self, d: R, x: [f32; 4]) {
        self.v[d] = x;
    }

    // ---- memory (offsets in floats) ----
    fn ldq(&mut self, d: R, at: usize) {
        self.v[d].copy_from_slice(&self.buf[at..at + 4]);
    }
    fn stq(&mut self, s: R, at: usize) {
        let v = self.v[s];
        self.buf[at..at + 4].copy_from_slice(&v);
    }
    /// `ldp qa, qb, [x1 + bytes]`.
    fn ldp(&mut self, a: R, b: R, at: usize) {
        self.ldq(a, at);
        self.ldq(b, at + 4);
    }
    fn stp(&mut self, a: R, b: R, at: usize) {
        self.stq(a, at);
        self.stq(b, at + 4);
    }
    /// `LOAD_INPUT ... no_lut=1`: 16 sequential floats from x2.
    fn load_input(&mut self, d1: R, d2: R, d3: R, d4: R) {
        let p = self.x2;
        self.ldq(d1, p);
        self.ldq(d2, p + 4);
        self.ldq(d3, p + 8);
        self.ldq(d4, p + 12);
        self.x2 += 16;
    }

    fn setup_sr_recomb(&mut self, len: usize) {
        self.tab = cos_tab(len);
        self.x7 = 0;
        self.x8 = (len / 4) as isize - 7;
        if len > 32 {
            self.x21 = len / 2; // 2*len bytes
            self.x22 = 3 * self.x21;
        }
    }
    /// `ld1 {a, b}, [x8], #-32` (im) and `ld1 {c, d}, [x7], #32` (re).
    fn ld_im(&mut self, a: R, b: R) {
        let p = self.x8 as usize;
        let t = self.tab;
        self.v[a] = [t[p], t[p + 1], t[p + 2], t[p + 3]];
        self.v[b] = [t[p + 4], t[p + 5], t[p + 6], t[p + 7]];
        self.x8 -= 8;
    }
    fn ld_re(&mut self, a: R, b: R) {
        let p = self.x7;
        let t = self.tab;
        self.v[a] = [t[p], t[p + 1], t[p + 2], t[p + 3]];
        self.v[b] = [t[p + 4], t[p + 5], t[p + 6], t[p + 7]];
        self.x7 += 8;
    }

    // ---- macros ----
    #[allow(clippy::too_many_arguments)]
    fn fft8(&mut self, e0: R, e1: R, o0: R, o1: R, t0: R, t1: R, t2: R, t3: R, t4: R, t5: R, t6: R) {
        self.fsub(t1, e1, o1);
        self.fadd(o1, e1, o1);
        self.fsub(t0, e0, o0);
        self.fadd(o0, e0, o0);
        self.set(t5, [SQ, -SQ, -SQ, SQ]);
        self.ext(t4, o1, o1, 12);
        self.rev64(t4, t4);
        self.ext(t2, o0, t4, 8);
        self.movd(o0, 1, t4, 1);
        self.fsub(e1, o0, t2);
        self.fadd(t2, o0, t2);
        self.rev64(t6, 31);
        self.dupd(o0, t0, 0);
        self.dupd(o1, t0, 1);
        self.rev64(t4, e1);
        self.rev64(o1, o1);
        self.ext(t6, 31, t6, 8);
        self.zip1d(t3, t2, e1);
        self.zip2d(t2, t2, t4);
        self.fadd(e0, t3, t2);
        self.fsub(e1, t3, t2);
        self.fmul(t1, t1, t5);
        self.fmls(o0, o1, t6);
        self.rev64(t4, t1);
        self.fmla(t1, t4, 31);
        self.rev64(t4, t1);
        self.ext(t4, t4, t4, 8);
        self.fmla(t4, t1, 31);
        self.fadd(o1, o0, t4);
        self.fsub(o0, o0, t4);
    }

    #[allow(clippy::too_many_arguments)]
    fn fft4_x2(&mut self, e0: R, o0: R, e1: R, o1: R, t0: R, t1: R, t2: R, t3: R, t4: R, t5: R, t6: R) {
        self.fadd(t0, e0, o0);
        self.fadd(t2, e1, o1);
        self.fsub(e0, e0, o0);
        self.fsub(e1, e1, o1);
        self.rev64(t4, e0);
        self.rev64(t5, e1);
        self.zip2d(o0, t0, e0);
        self.zip2d(o1, t2, e1);
        // t6 = shuf_4pt_x2 (used by tbl below)
        self.movd(o0, 1, t4, 1);
        self.movd(o1, 1, t5, 1);
        self.zip1d(t1, t0, e0);
        self.zip1d(t3, t2, e1);
        self.fsub(t4, t1, o0);
        self.fadd(t5, t1, o0);
        self.fsub(t2, t3, o1);
        self.fadd(t3, t3, o1);
        // tbl {a, b}: [b[2], a[3], a[2], b[3]]
        let (a, b) = (self.v[t4], self.v[t5]);
        self.v[o0] = [b[2], a[3], a[2], b[3]];
        let (a, b) = (self.v[t2], self.v[t3]);
        self.v[o1] = [b[2], a[3], a[2], b[3]];
        self.zip1d(e0, t5, t4);
        self.zip1d(e1, t3, t2);
        let _ = t6;
    }

    fn fft8_x2(&mut self, e0: R, e1: R, o0: R, o1: R, e2: R, e3: R, o2: R, o3: R) {
        self.fadd(19, e3, o3);
        self.fadd(17, e1, o1);
        self.fadd(18, e2, o2);
        self.fadd(16, e0, o0);
        self.set(23, [SQ, -SQ, -SQ, SQ]);
        self.ext(22, 19, 19, 12);
        self.ext(21, 17, 17, 12);
        self.rev64(22, 22);
        self.rev64(21, 21);
        self.ext(19, 18, 22, 8);
        self.ext(17, 16, 21, 8);
        self.movd(18, 1, 22, 1);
        self.movd(21, 0, 16, 0);
        self.fadd(22, 18, 19);
        self.fsub(19, 18, 19);
        self.fsub(18, 21, 17);
        self.fadd(16, 21, 17);
        self.fsub(e0, e0, o0);
        self.fsub(20, e1, o1);
        self.fsub(e2, e2, o2);
        self.fsub(21, e3, o3);
        self.rev64(24, 31);
        self.zip1d(17, 16, 18);
        self.zip1d(e1, 22, 19);
        self.rev64(18, 18);
        self.rev64(19, 19);
        self.zip2d(16, 16, 18);
        self.zip2d(e3, 22, 19);
        self.dupd(o0, e0, 0);
        self.dupd(o1, e0, 1);
        self.dupd(o2, e2, 0);
        self.dupd(o3, e2, 1);
        self.fadd(e2, e1, e3);
        self.fsub(e3, e1, e3);
        self.fadd(e0, 17, 16);
        self.fsub(e1, 17, 16);
        self.ext(24, 31, 24, 8);
        self.rev64(o1, o1);
        self.rev64(o3, o3);
        self.fmul(19, 20, 23);
        self.fmul(21, 21, 23);
        self.rev64(20, 19);
        self.rev64(18, 21);
        self.fmls(o0, o1, 24);
        self.fmls(o2, o3, 24);
        self.fmla(19, 20, 31);
        self.fmla(21, 18, 31);
        self.rev64(20, 19);
        self.rev64(18, 21);
        self.ext(20, 20, 20, 8);
        self.ext(18, 18, 18, 8);
        self.fmla(20, 19, 31);
        self.fmla(18, 21, 31);
        self.fadd(o1, o0, 20);
        self.fadd(o3, o2, 18);
        self.fsub(o0, o0, 20);
        self.fsub(o2, o2, 18);
    }

    #[allow(clippy::too_many_arguments)]
    fn fft16(&mut self, e0: R, e1: R, e2: R, e3: R, o0: R, o1: R, o2: R, o3: R, t: [R; 7]) {
        let [t0, t1, t2, t3, t4, t5, t6] = t;
        self.fft8(e0, e1, e2, e3, t0, t1, t2, t3, t4, t5, t6);
        self.fft4_x2(o0, o1, o2, o3, t0, t1, t2, t3, t4, t5, t6);
        self.rev64(t0, o0);
        self.rev64(t1, o2);
        self.v[t0][0] = 0.0;
        self.v[t0][1] = 0.0;
        self.v[t1][0] = 0.0;
        self.v[t1][1] = 0.0;
        self.set(t4, [-C1, C1, -C3, C3]);
        self.set(t5, [C3, C3, C1, C1]);
        self.set(t6, [1.0, 1.0, SQ, SQ]);
        self.fmla(o2, t1, 31);
        self.fmls(o0, t0, 31);
        self.fmul(t2, o1, t4);
        self.fmul(t3, o3, t4);
        self.rev64(o3, o3);
        self.rev64(o1, o1);
        self.fmla(t3, o3, t5);
        self.fmls(t2, o1, t5);
        self.fmul(t1, o2, t6);
        self.fmul(t0, o0, t6);
        self.mov(o1, t3);
        self.mov(o2, t1);
        self.fsub(t3, t3, t2);
        self.fsub(t1, t1, t0);
        self.fadd(t2, t2, o1);
        self.rev64(t3, t3);
        self.fadd(t0, t0, o2);
        self.rev64(t1, t1);
        self.fmul(t2, t2, 31);
        self.fmul(t1, t1, 31);
        self.fadd(o3, e3, t3);
        self.fsub(o2, e3, t3);
        self.fsub(o1, e2, t2);
        self.fadd(o0, e2, t2);
        self.fsub(e2, e0, t0);
        self.fadd(e0, e0, t0);
        self.fsub(e3, e1, t1);
        self.fadd(e1, e1, t1);
    }

    /// `SR_COMBINE ... swap_im=0`.
    #[allow(clippy::too_many_arguments)]
    fn sr_combine(&mut self, e: [R; 8], o: [R; 8], t: [R; 12]) {
        let [e0, e1, e2, e3, e4, e5, e6, e7] = e;
        let [o0, o1, o2, o3, o4, o5, o6, o7] = o;
        let [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb] = t;
        self.ld_im(t8, t9);
        self.ld_re(t0, t1);
        self.ext(t2, t8, t8, 8);
        self.ext(t3, t9, t9, 8);
        self.trn1s(t4, t0);
        self.trn2s(t0, t0);
        self.trn1s(t5, t1);
        self.trn2s(t1, t1);
        self.rev64(t6, o0);
        self.rev64(t7, o2);
        self.rev64(t8, o4);
        self.rev64(t9, o6);
        self.fmul(t6, t6, t4);
        self.fmul(t7, t7, t0);
        self.fmul(t8, t8, t4);
        self.fmul(t9, t9, t0);
        self.rev64(ta, o1);
        self.rev64(tb, o3);
        self.rev64(t4, o5);
        self.rev64(t0, o7);
        self.fmul(ta, ta, t5);
        self.fmul(tb, tb, t1);
        self.fmul(t4, t4, t5);
        self.fmul(t0, t0, t1);
        self.trn1s(t5, t3);
        self.trn2s(t3, t3);
        self.trn1s(t1, t2);
        self.trn2s(t2, t2);
        self.fmul(t5, t5, 31);
        self.fmul(t3, t3, 31);
        self.fmul(t1, t1, 31);
        self.fmul(t2, t2, 31);
        self.fmla(t7, o2, t5);
        self.fmls(t9, o6, t5);
        self.fmla(t6, o0, t3);
        self.fmls(t8, o4, t3);
        self.fmla(ta, o1, t2);
        self.fmla(tb, o3, t1);
        self.fmls(t4, o5, t2);
        self.fmls(t0, o7, t1);
        self.fsub(t2, t7, t9);
        self.fsub(t1, t8, t6);
        self.fsub(t3, t4, ta);
        self.fsub(t5, t0, tb);
        self.fadd(t6, t8, t6);
        self.fadd(t7, t9, t7);
        self.fadd(t8, t4, ta);
        self.fadd(t9, t0, tb);
        self.fmul(t1, t1, 31);
        self.fmul(t2, t2, 31);
        self.fmul(t3, t3, 31);
        self.fmul(t5, t5, 31);
        self.rev64(t6, t6);
        self.rev64(t8, t8);
        self.rev64(t7, t7);
        self.rev64(t9, t9);
        self.fsub(o0, e0, t6);
        self.fsub(o1, e1, t8);
        self.fsub(o2, e2, t1);
        self.fsub(o3, e3, t3);
        self.fsub(o4, e4, t7);
        self.fsub(o5, e6, t9);
        self.fadd(o6, e5, t2);
        self.fsub(o7, e7, t5);
        self.fadd(e0, e0, t6);
        self.fadd(e1, e1, t8);
        self.fadd(e2, e2, t1);
        self.fadd(e3, e3, t3);
        self.fadd(e4, e4, t7);
        self.fsub(e5, e5, t2);
        self.fadd(e6, e6, t9);
        self.fadd(e7, e7, t5);
    }

    #[allow(clippy::too_many_arguments)]
    fn sr_combine_half(&mut self, e: [R; 4], o: [R; 4], c: [R; 4], t: [R; 6], part: u8) {
        let [e0, e1, e2, e3] = e;
        let [o0, o1, o2, o3] = o;
        let [c0, c1, c2, c3] = c;
        let [t0, t1, t2, t3, t4, t5] = t;
        if part == 0 {
            self.trn1s(t4, c0);
            self.trn1s(c1, c1);
            self.trn2s(t5, c2);
            self.trn2s(c3, c3);
        } else {
            self.trn2s(t4, c0);
            self.trn2s(c1, c1);
            self.trn1s(t5, c2);
            self.trn1s(c3, c3);
        }
        self.fmul(t5, t5, 31);
        self.fmul(c3, c3, 31);
        self.rev64(t0, o0);
        self.rev64(t1, o2);
        self.rev64(t2, o1);
        self.rev64(t3, o3);
        self.fmul(o0, o0, c3);
        self.fmul(o1, o1, t5);
        self.fmla(o0, t0, t4);
        self.fmla(o1, t2, c1);
        self.fmul(t1, t1, t4);
        self.fmul(t3, t3, c1);
        self.fmls(t1, o2, c3);
        self.fmls(t3, o3, t5);
        self.fsub(t0, t1, o0);
        self.fadd(t1, t1, o0);
        self.fadd(t2, t3, o1);
        self.fsub(t3, t3, o1);
        self.fmul(t0, t0, 31);
        self.fmul(t3, t3, 31);
        self.rev64(t1, t1);
        self.rev64(t2, t2);
        if part == 0 {
            self.fsub(o0, e0, t1);
            self.fsub(o1, e1, t2);
            self.fsub(o2, e2, t0);
            self.fsub(o3, e3, t3);
            self.fadd(e0, e0, t1);
            self.fadd(e1, e1, t2);
            self.fadd(e2, e2, t0);
            self.fadd(e3, e3, t3);
        } else {
            self.fsub(o0, e0, t1);
            self.fadd(o2, e1, t2);
            self.fsub(o1, e2, t0);
            self.fadd(o3, e3, t3);
            self.fadd(e0, e0, t1);
            self.fsub(e1, e1, t2);
            self.fadd(e2, e2, t0);
            self.fsub(e3, e3, t3);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sr_combine_lite(&mut self, e: [R; 4], o: [R; 4], c: [R; 4], t: [R; 3], part: u8) {
        let [e0, e1, e2, e3] = e;
        let [o0, o1, o2, o3] = o;
        let [c0, c1, c2, c3] = c;
        let [t0, t1, t2] = t;
        self.rev64(t0, o0);
        self.rev64(t1, o2);
        if part == 0 { self.trn2s(t2, c3) } else { self.trn1s(t2, c3) }
        self.fmul(t2, t2, 31);
        self.fmul(o2, o2, t2);
        self.fmul(o0, o0, t2);
        if part == 0 { self.trn1s(t2, c0) } else { self.trn2s(t2, c0) }
        self.fmul(t1, t1, t2);
        self.fmla(o0, t0, t2);
        self.fsub(t1, t1, o2);
        self.rev64(t2, o1);
        self.rev64(o2, o3);
        if part == 0 { self.trn2s(t0, c2) } else { self.trn1s(t0, c2) }
        self.fmul(t0, t0, 31);
        self.fmul(o1, o1, t0);
        self.fmul(o3, o3, t0);
        if part == 0 { self.trn1s(t0, c1) } else { self.trn2s(t0, c1) }
        self.fmul(o2, o2, t0);
        self.fmla(o1, t2, t0);
        self.fsub(o2, o2, o3);
        self.fsub(t0, t1, o0);
        self.fadd(o0, t1, o0);
        self.fadd(t2, o2, o1);
        self.fsub(t1, o2, o1);
        self.fmul(t0, t0, 31);
        self.fmul(t1, t1, 31);
        self.rev64(t2, t2);
        self.rev64(o0, o0);
        if part == 0 {
            self.fsub(o1, e1, t2);
            self.fsub(o2, e2, t0);
            self.fsub(o3, e3, t1);
            self.fadd(e1, e1, t2);
            self.fadd(e2, e2, t0);
            self.fadd(e3, e3, t1);
        } else {
            self.fadd(o2, e1, t0);
            self.fsub(o1, e2, t2);
            self.fadd(o3, e3, t1);
            self.fsub(e1, e1, t0);
            self.fadd(e2, e2, t2);
            self.fsub(e3, e3, t1);
        }
        self.mov(t1, o0);
        self.fsub(o0, e0, t1);
        self.fadd(e0, e0, t1);
    }

    const T16: [R; 12] = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

    fn sr_combine_4(&mut self, part: usize, off: usize) {
        let x1 = self.x1;
        let (x10, x11, x12) = (x1 + self.x21, x1 + 2 * self.x21, x1 + self.x22);
        let a = (part * 32 + off) / 4;
        let b = ((2 + part) * 32 + off) / 4;
        self.ldp(0, 1, x1 + a);
        self.ldp(4, 5, x1 + b);
        self.ldp(2, 3, x10 + a);
        self.ldp(6, 7, x10 + b);
        self.ldp(8, 9, x11 + a);
        self.ldp(10, 11, x11 + b);
        self.ldp(12, 13, x12 + a);
        self.ldp(14, 15, x12 + b);
        self.sr_combine([0, 1, 2, 3, 4, 6, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15], Self::T16);
        self.stp(0, 1, x1 + a);
        self.stp(4, 5, x1 + b);
        self.stp(2, 3, x10 + a);
        self.stp(6, 7, x10 + b);
        self.stp(8, 9, x11 + a);
        self.stp(12, 13, x11 + b);
        self.stp(10, 11, x12 + a);
        self.stp(14, 15, x12 + b);
    }

    fn sr_combine_full(&mut self, off: usize) {
        for part in [0, 1, 4, 5] {
            self.sr_combine_4(part, off);
        }
    }

    fn sr_combine_d2(&mut self, part: usize, x14: usize, x15: usize, x16: usize) {
        let o = part * 32 / 4;
        let (x10, x11, x12, x13) = (self.x1 + o, x14 + o, x15 + o, x16 + o);
        self.ldp(0, 1, x10);
        self.ldp(4, 5, x10 + 16);
        self.ldp(2, 3, x11);
        self.ldp(6, 7, x11 + 16);
        self.ldp(8, 9, x12);
        self.ldp(10, 11, x12 + 16);
        self.ldp(12, 13, x13);
        self.ldp(14, 15, x13 + 16);
        self.sr_combine([0, 1, 2, 3, 4, 6, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15], Self::T16);
        self.zip1d(16, 0, 4);
        self.zip2d(17, 0, 4);
        self.zip1d(18, 1, 5);
        self.zip2d(19, 1, 5);
        self.zip1d(20, 2, 6);
        self.zip2d(21, 2, 6);
        self.zip1d(22, 3, 7);
        self.zip2d(23, 3, 7);
        self.ldp(0, 1, x10 + 8);
        self.ldp(4, 5, x10 + 24);
        self.ldp(2, 3, x11 + 8);
        self.ldp(6, 7, x11 + 24);
        for (k, r) in [16, 17, 18, 19].into_iter().enumerate() {
            self.stq(r, x10 + 4 * k);
        }
        for (k, r) in [20, 21, 22, 23].into_iter().enumerate() {
            self.stq(r, x11 + 4 * k);
        }
        self.zip1d(20, 8, 12);
        self.zip2d(21, 8, 12);
        self.zip1d(22, 9, 13);
        self.zip2d(23, 9, 13);
        self.zip1d(24, 10, 14);
        self.zip2d(25, 10, 14);
        self.zip1d(26, 11, 15);
        self.zip2d(27, 11, 15);
        self.ldp(8, 9, x12 + 8);
        self.ldp(10, 11, x12 + 24);
        self.ldp(12, 13, x13 + 8);
        self.ldp(14, 15, x13 + 24);
        for (k, r) in [20, 21, 22, 23].into_iter().enumerate() {
            self.stq(r, x12 + 4 * k);
        }
        for (k, r) in [24, 25, 26, 27].into_iter().enumerate() {
            self.stq(r, x13 + 4 * k);
        }
        let (x10, x11, x12, x13) = (x10 + 16, x11 + 16, x12 + 16, x13 + 16);
        self.sr_combine([0, 1, 2, 3, 4, 6, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15], Self::T16);
        self.zip1d(16, 0, 4);
        self.zip2d(17, 0, 4);
        self.zip1d(18, 1, 5);
        self.zip2d(19, 1, 5);
        for (k, r) in [16, 17, 18, 19].into_iter().enumerate() {
            self.stq(r, x10 + 4 * k);
        }
        self.zip1d(16, 2, 6);
        self.zip2d(17, 2, 6);
        self.zip1d(18, 3, 7);
        self.zip2d(19, 3, 7);
        for (k, r) in [16, 17, 18, 19].into_iter().enumerate() {
            self.stq(r, x11 + 4 * k);
        }
        self.zip1d(20, 8, 12);
        self.zip2d(21, 8, 12);
        self.zip1d(22, 9, 13);
        self.zip2d(23, 9, 13);
        for (k, r) in [20, 21, 22, 23].into_iter().enumerate() {
            self.stq(r, x12 + 4 * k);
        }
        self.zip1d(24, 10, 14);
        self.zip2d(25, 10, 14);
        self.zip1d(26, 11, 15);
        self.zip2d(27, 11, 15);
        for (k, r) in [24, 25, 26, 27].into_iter().enumerate() {
            self.stq(r, x13 + 4 * k);
        }
    }

    /// `ff_tx_fft32_ns_float_neon`: the standalone 32-point codelet (its
    /// register order and zip-store deinterleave differ from the split-radix
    /// base case in [`Self::sr`]).
    fn fft32(&mut self) {
        let x1 = self.x1;
        self.setup_sr_recomb(32);
        self.load_input(0, 1, 2, 3);
        self.load_input(4, 5, 6, 7);
        self.load_input(8, 9, 10, 11);
        self.load_input(12, 13, 14, 15);
        self.fft8_x2(8, 9, 10, 11, 12, 13, 14, 15);
        self.fft16(0, 1, 2, 3, 4, 5, 6, 7, [16, 17, 18, 19, 20, 21, 22]);
        self.sr_combine([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], Self::T16);
        let groups = [[(0, 4), (1, 6)], [(2, 5), (3, 7)], [(8, 12), (9, 13)], [(10, 14), (11, 15)]];
        for (g, pairs) in groups.into_iter().enumerate() {
            let base = 16 + 4 * g;
            for (p, (a, b)) in pairs.into_iter().enumerate() {
                self.zip1d(base + 2 * p, a, b);
                self.zip2d(base + 2 * p + 1, a, b);
            }
            for k in 0..4 {
                self.stq(base + k, x1 + 16 * g + 4 * k);
            }
        }
    }

    /// Label `0:`: the general deinterleave loop (top level only).
    fn deinterleave(&mut self) {
        let mut w19 = self.w19 as isize;
        loop {
            let x14 = self.x1 + self.x21;
            let x15 = self.x1 + 2 * self.x21;
            let x16 = self.x1 + self.x22;
            self.sr_combine_d2(0, x14, x15, x16);
            self.sr_combine_d2(4, x14, x15, x16);
            self.x1 += 32 * 8 / 4;
            w19 -= 32 * 4;
            if w19 <= 0 {
                break;
            }
        }
    }

    /// Label `32:` with local length `w20`: an in-place `w20`-point FFT at
    /// x1, reading pre-shuffled input from x2. Returns `true` when the top
    /// level finished (the deinterleave ran and the function returned).
    fn sr(&mut self, w20: usize) -> bool {
        let x1 = self.x1;
        self.setup_sr_recomb(32);
        self.load_input(0, 1, 2, 3);
        self.load_input(4, 6, 5, 7);
        self.load_input(8, 9, 10, 11);
        self.load_input(12, 13, 14, 15);
        self.fft8_x2(8, 9, 10, 11, 12, 13, 14, 15);
        self.fft16(0, 1, 2, 3, 4, 6, 5, 7, [16, 17, 18, 19, 20, 21, 22]);
        self.sr_combine([0, 1, 2, 3, 4, 6, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15], Self::T16);
        self.stp(2, 3, x1 + 8);
        self.stp(6, 7, x1 + 24);
        self.stp(10, 11, x1 + 40);
        self.stp(14, 15, x1 + 56);
        if w20 <= 32 {
            self.stp(0, 1, x1);
            self.stp(4, 5, x1 + 16);
            self.stp(8, 9, x1 + 32);
            self.stp(12, 13, x1 + 48);
            return false;
        }
        // 64:
        self.setup_sr_recomb(64);
        self.load_input(2, 3, 10, 11);
        self.load_input(6, 14, 7, 15);
        self.fft16(2, 3, 10, 11, 6, 14, 7, 15, [16, 17, 18, 19, 20, 21, 22]);
        self.load_input(16, 17, 18, 19);
        self.load_input(20, 22, 21, 23);
        self.fft16(16, 17, 18, 19, 20, 22, 21, 23, [24, 25, 26, 27, 28, 29, 30]);
        self.ld_im(26, 27);
        self.ld_re(24, 25);
        self.ext(26, 26, 26, 8);
        self.ext(27, 27, 27, 8);
        if self.w19 == 64 {
            self.deinterleave_64();
            return true;
        }
        self.sr_combine_lite([0, 1, 8, 9], [2, 3, 16, 17], [24, 25, 26, 27], [28, 29, 30], 0);
        self.stp(0, 1, x1);
        self.stp(8, 9, x1 + 32);
        self.stp(2, 3, x1 + 64);
        self.stp(16, 17, x1 + 96);
        self.sr_combine_half([4, 5, 12, 13], [6, 7, 20, 21], [24, 25, 26, 27], [28, 29, 30, 0, 1, 8], 1);
        self.stp(4, 20, x1 + 16);
        self.stp(12, 21, x1 + 48);
        self.stp(6, 5, x1 + 80);
        self.stp(7, 13, x1 + 112);
        self.ldp(2, 3, x1 + 8);
        self.ldp(6, 7, x1 + 24);
        self.ldp(12, 13, x1 + 40);
        self.ldp(16, 17, x1 + 56);
        self.sr_combine(
            [2, 3, 12, 13, 6, 16, 7, 17],
            [10, 11, 14, 15, 18, 19, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 8, 0, 1, 4, 5],
        );
        self.stp(2, 3, x1 + 8);
        self.stp(6, 7, x1 + 24);
        self.stp(12, 13, x1 + 40);
        self.stp(16, 17, x1 + 56);
        self.stp(10, 11, x1 + 72);
        self.stp(18, 19, x1 + 88);
        self.stp(14, 15, x1 + 104);
        self.stp(22, 23, x1 + 120);
        if w20 <= 64 {
            return false;
        }
        // 128 and up: two quarter-length sub-transforms, then combine.
        let mut len = 128;
        loop {
            let q = len / 4;
            // x1 offsets as the asm leaves them (bytes): 128: +16*32, then +8*32, back -24*32.
            let (first, second, back) = match len {
                128 => (16 * 32, 8 * 32, 24 * 32),
                256 => (32 * 32, 16 * 32, 48 * 32),
                512 => (64 * 32, 32 * 32, 96 * 32),
                1024 => (96 * 32, 64 * 32, 192 * 32),
                _ => unreachable!(),
            };
            self.x1 += first / 4;
            self.sr(q);
            self.x1 += second / 4;
            self.sr(q);
            self.x1 -= back / 4;
            self.setup_sr_recomb(len);
            if self.w19 == len {
                self.deinterleave();
                return true;
            }
            match len {
                128 => self.sr_combine_full(0),
                256 => {
                    self.sr_combine_full(0);
                    self.sr_combine_full(8 * 32);
                }
                _ => {
                    for _ in 0..len / 128 {
                        self.sr_combine_full(0);
                        self.x1 += 8 * 32 / 4;
                    }
                }
            }
            if w20 <= len {
                return false;
            }
            len *= 2;
            if len > 1024 {
                unreachable!("lengths above 1024 are not transcribed");
            }
        }
    }

    /// Label `2:`: the 64-point top-level deinterleave.
    fn deinterleave_64(&mut self) {
        let x1 = self.x1;
        let saved23 = self.v[23];
        self.sr_combine_lite([0, 1, 8, 9], [2, 3, 16, 17], [24, 25, 26, 27], [28, 29, 30], 0);
        self.sr_combine_half([4, 5, 12, 13], [6, 7, 20, 21], [24, 25, 26, 27], [28, 29, 30, 23, 24, 26], 1);
        self.zip1d(23, 0, 4);
        self.zip2d(24, 0, 4);
        self.zip1d(25, 1, 20);
        self.zip2d(26, 1, 20);
        self.zip1d(27, 8, 12);
        self.zip2d(28, 8, 12);
        self.zip1d(29, 9, 21);
        self.zip2d(30, 9, 21);
        self.mov(20, 5);
        self.mov(21, 7);
        let (x12, x13, x14, x15) = (x1, x1 + 32, x1 + 64, x1 + 96);
        self.zip1d(4, 2, 6);
        self.zip2d(5, 2, 6);
        self.zip1d(6, 3, 20);
        self.zip2d(7, 3, 20);
        self.zip1d(0, 16, 21);
        self.zip2d(1, 16, 21);
        self.zip1d(2, 17, 13);
        self.zip2d(3, 17, 13);
        self.ldp(8, 9, x1 + 8);
        self.ldp(12, 13, x1 + 40);
        for (k, r) in [23, 24, 25, 26].into_iter().enumerate() {
            self.stq(r, x12 + 4 * k);
        }
        for (k, r) in [27, 28, 29, 30].into_iter().enumerate() {
            self.stq(r, x13 + 4 * k);
        }
        for (k, r) in [4, 5, 6, 7].into_iter().enumerate() {
            self.stq(r, x14 + 4 * k);
        }
        for (k, r) in [0, 1, 2, 3].into_iter().enumerate() {
            self.stq(r, x15 + 4 * k);
        }
        let (x12, x13, x14, x15) = (x12 + 16, x13 + 16, x14 + 16, x15 + 16);
        self.v[23] = saved23;
        self.ldp(6, 7, x1 + 24);
        self.ldp(16, 17, x1 + 56);
        self.sr_combine(
            [8, 9, 12, 13, 6, 16, 7, 17],
            [10, 11, 14, 15, 18, 19, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 4, 0, 1, 5, 20],
        );
        type Group = ([(R, R); 2], [(R, R); 2], usize);
        let groups: [Group; 4] = [
            ([(8, 6), (9, 7)], [(0, 1), (2, 3)], x12),
            ([(12, 16), (13, 17)], [(4, 5), (6, 7)], x13),
            ([(10, 18), (11, 19)], [(0, 1), (2, 3)], x14),
            ([(14, 22), (15, 23)], [(4, 5), (6, 7)], x15),
        ];
        for (src, dst, at) in groups {
            for k in 0..2 {
                let (a, b) = src[k];
                let (d1, d2) = dst[k];
                self.zip1d(d1, a, b);
                self.zip2d(d2, a, b);
            }
            let regs = [dst[0].0, dst[0].1, dst[1].0, dst[1].1];
            for (k, r) in regs.into_iter().enumerate() {
                self.stq(r, at + 4 * k);
            }
        }
    }
}

/// `ff_tx_fft_sr_ns_float_neon` on `len` complex values in `z`
/// (interleaved re, im), in place; the input must already be in the
/// codelet's pre-shuffled order.
pub fn fft_sr_ns(z: &mut [f32], len: usize) {
    assert!(len.is_power_of_two() && (4..=1024).contains(&len) && z.len() == 2 * len);
    let mut n = Neon {
        v: [[0.0; 4]; 32],
        buf: z,
        x1: 0,
        x2: 0,
        tab: cos_tab(32),
        x7: 0,
        x8: 0,
        x21: 0,
        x22: 0,
        w19: len,
    };
    n.v[31] = SUBADD;
    match len {
        4 => n.fft4_fn(),
        8 => n.fft8_fn(),
        16 => n.fft16_fn(),
        32 => n.fft32(),
        _ => {
            n.sr(len);
        }
    }
}

impl Neon<'_> {
    /// `FFT4 e0, o0, standalone=1`.
    fn fft4_standalone(&mut self, e0: R, o0: R) {
        self.fadd(16, e0, o0);
        self.fsub(e0, e0, o0);
        self.rev64(18, e0);
        self.zip2d(o0, 16, e0);
        self.zip1d(17, 16, e0);
        self.movd(o0, 1, 18, 1);
        self.fadd(e0, 17, o0);
        self.fsub(16, 17, o0);
        self.mov(o0, 16);
        self.v[o0][3] = self.v[e0][3];
        self.v[e0][3] = self.v[16][3];
    }

    /// `ff_tx_fft4_fwd_float_neon` (in place, also used pre-shuffled).
    fn fft4_fn(&mut self) {
        self.ldp(0, 1, 0);
        self.fft4_standalone(0, 1);
        self.stp(0, 1, 0);
    }

    /// `ff_tx_fft8_ns_float_neon`.
    fn fft8_fn(&mut self) {
        self.load_input(0, 1, 2, 3);
        self.fft8(0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22);
        self.zip1d(16, 0, 2);
        self.zip2d(17, 0, 2);
        self.zip1d(18, 1, 3);
        self.zip2d(19, 1, 3);
        for k in 0..4 {
            self.stq(16 + k, 4 * k);
        }
    }

    /// `ff_tx_fft16_ns_float_neon`.
    fn fft16_fn(&mut self) {
        self.load_input(0, 1, 2, 3);
        self.load_input(4, 5, 6, 7);
        self.fft16(0, 1, 2, 3, 4, 5, 6, 7, [16, 17, 18, 19, 20, 21, 22]);
        let pairs = [(0, 4), (1, 6), (2, 5), (3, 7)];
        for (p, (a, b)) in pairs.into_iter().enumerate() {
            self.zip1d(20 + 2 * p, a, b);
            self.zip2d(21 + 2 * p, a, b);
        }
        for k in 0..8 {
            self.stq(20 + k, 4 * k);
        }
    }
}

/// `parity_revtab_generator` with `basis 8 >> 1`, `dual_stride 0`, gather
/// (`inv_lookup = 1`): the input order the "ns" codelet expects.
pub fn split_radix_parity_revtab(len: usize, inv: bool) -> Vec<usize> {
    fn perm(i: i64, len: i64, inv: bool) -> i64 {
        let len = len >> 1;
        if len <= 1 {
            return i & 1;
        }
        if i & len == 0 {
            return perm(i, len, inv) * 2;
        }
        let len = len >> 1;
        perm(i, len, inv) * 4 + 1 - 2 * (((i & len == 0) as i64) ^ inv as i64)
    }
    #[allow(clippy::too_many_arguments)]
    fn gen(rev: &mut [usize], n: i64, inv: bool, offset: i64, is_dual: bool, dual_high: bool, len: i64, basis: i64) {
        let len = len >> 1;
        if len <= basis {
            // dual_stride = 0: stride 0, no dual layout.
            let _ = (is_dual, dual_high);
            for i in 0..len {
                let k1 = -perm(offset + i * 2, n, inv) & (n - 1);
                let k2 = -perm(offset + i * 2 + 1, n, inv) & (n - 1);
                rev[(offset + i) as usize] = k1 as usize;
                rev[(offset + len + i) as usize] = k2 as usize;
            }
            return;
        }
        gen(rev, n, inv, offset, false, false, len, basis);
        gen(rev, n, inv, offset + len, true, false, len >> 1, basis);
        gen(rev, n, inv, offset + len + (len >> 1), true, true, len >> 1, basis);
    }
    let mut rev = vec![0usize; len];
    gen(&mut rev, len as i64, inv, 0, false, false, len as i64, 4);
    rev
}

/// The same map in scatter form (`FF_TX_MAP_SCATTER`, `inv_lookup = 0`):
/// natural index `i` is stored at position `map[i]`.
pub fn split_radix_parity_revtab_scatter(len: usize, inv: bool) -> Vec<usize> {
    let g = split_radix_parity_revtab(len, inv);
    let mut s = vec![0usize; len];
    for (pos, &k) in g.iter().enumerate() {
        s[k] = pos;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codelet computes a DFT of the un-shuffled input (inverse sign for `inv`).
    #[test]
    fn neon_fft_is_a_dft() {
        for len in [4usize, 8, 16, 32, 64, 128, 256, 512, 1024] {
            for inv in [false, true] {
                let x: Vec<(f64, f64)> = (0..len)
                    .map(|i| (((i * 7919) % 97) as f64 / 97.0 - 0.5, ((i * 104729) % 89) as f64 / 89.0 - 0.5))
                    .collect();
                let map = split_radix_parity_revtab(len, inv);
                let mut z = vec![0f32; 2 * len];
                for i in 0..len {
                    z[2 * i] = x[map[i]].0 as f32;
                    z[2 * i + 1] = x[map[i]].1 as f32;
                }
                fft_sr_ns(&mut z, len);
                let sign = if inv { 1.0 } else { -1.0 };
                let mut worst = 0f64;
                for k in 0..len {
                    let (mut re, mut im) = (0f64, 0f64);
                    for (n, &(a, b)) in x.iter().enumerate() {
                        let t = sign * 2.0 * std::f64::consts::PI * ((n * k) % len) as f64 / len as f64;
                        re += a * t.cos() - b * t.sin();
                        im += a * t.sin() + b * t.cos();
                    }
                    worst = worst.max((z[2 * k] as f64 - re).abs()).max((z[2 * k + 1] as f64 - im).abs());
                }
                assert!(worst < 1e-4, "len {len} inv {inv}: worst {worst}");
            }
        }
    }
}
