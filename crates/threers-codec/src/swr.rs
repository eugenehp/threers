//! What the ffmpeg n7.1 CLI does to decoded float audio for
//! `-f s16le -c:a pcm_s16le -ar RATE -ac 2`: the auto-inserted `aresample`
//! filter's libswresample context (default options, aarch64 NEON build).
//!
//! - **Rematrix** mono -> stereo: both outputs `in * (float)M_SQRT1_2`
//!   (`swr_build_matrix2`, no normalisation below unity gain), before the
//!   resampler or after it as `resample_first` decides
//!   (`out_ch/in_ch - 1 < out_rate/in_rate - 1`, integer left side).
//! - **Resample** (only when the rates differ): `resample_init` with
//!   `filter_size 32`, `phase_shift 10`, `cutoff 0.97`, Kaiser beta 9,
//!   `exact_rational 1`; float filter bank from `build_filter`; the NEON
//!   `resample_common` (8-wide fused dot product, pairwise lane sum, then
//!   an unfused C tail); the start mirrored by `invert_initial_buffer` and the end by
//!   `resample_flush`.
//! - **s16 conversion** per `swr_convert` call: the first `len & !15`
//!   samples by NEON (`fcvtzs #31` then `>> 16`: truncation), the rest by C
//!   `av_clip_int16(lrintf(x * 32768))`. One call per decoded frame, so the
//!   split follows the frame sizes; the flush is one more call.
//!
//! Rate pairs that would need `resample_linear` (a non-zero
//! `dst_incr_mod`) are refused.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use crate::aac::dsp::bessel_i0;

/// Planar float input as the decoder delivered it.
pub struct DecodedFrames<'a> {
    /// One `Vec` per channel.
    pub planes: &'a [Vec<f32>],
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples per delivered frame (after priming was dropped), in order.
    pub frame_lens: &'a [usize],
}

struct Resampler {
    phase_count: i64,
    filter_length: usize,
    filter_alloc: usize,
    bank: Vec<f32>,
    src_incr: i64,
    dst_incr_div: i64,
    dst_incr_mod: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl Resampler {
    fn new(out_rate: i64, in_rate: i64) -> Result<Self, String> {
        Self::with_filter_size(out_rate, in_rate, 32)
    }

    fn with_filter_size(out_rate: i64, in_rate: i64, filter_size: usize) -> Result<Self, String> {
        let cutoff = 0.97f64;
        let factor = (out_rate as f64 * cutoff / in_rate as f64).min(1.0);
        let mut phase_count: i64 = 1 << 10;
        let mut filter_length = ((filter_size as f64 / factor).ceil() as i64).max(1) as usize;
        if filter_length > 1 {
            filter_length = (filter_length + 1) & !1;
        }
        let g = gcd(out_rate, in_rate);
        let exact = out_rate / g;
        if exact <= phase_count {
            phase_count = exact;
        }
        let filter_alloc = (filter_length + 7) & !7;
        let mut bank = vec![0f32; filter_alloc * (phase_count as usize + 1)];
        build_filter(&mut bank, factor, filter_length, filter_alloc, phase_count as usize, 9.0);
        let pc = phase_count as usize;
        let (head, tail) = bank.split_at_mut(filter_alloc * pc + 1);
        tail[..filter_alloc - 1].copy_from_slice(&head[..filter_alloc - 1]);
        bank[filter_alloc * pc] = bank[filter_alloc - 1];
        // av_reduce(out_rate, in_rate * phase_count) then double up to 2^20.
        let (mut src_incr, mut dst_incr) = (out_rate, in_rate * phase_count);
        let g = gcd(src_incr, dst_incr);
        src_incr /= g;
        dst_incr /= g;
        if src_incr > i32::MAX as i64 / 2 || dst_incr > i32::MAX as i64 / 2 {
            return Err("resampling ratio needs av_reduce approximation".into());
        }
        while dst_incr < (1 << 20) && src_incr < (1 << 20) {
            dst_incr *= 2;
            src_incr *= 2;
        }
        let r = Self {
            phase_count,
            filter_length,
            filter_alloc,
            bank,
            src_incr,
            dst_incr_div: dst_incr / src_incr,
            dst_incr_mod: dst_incr % src_incr,
        };
        if r.dst_incr_mod != 0 {
            return Err(format!("{in_rate} -> {out_rate} Hz needs resample_linear"));
        }
        Ok(r)
    }

    /// `ff_resample_common_float_neon` for one output: four fused lanes over
    /// blocks of 8 (or 4 when shorter), the pairwise lane sum, then an
    /// unfused C tail.
    #[inline]
    fn apply(&self, src: &[f32], index: usize) -> f32 {
        let l = self.filter_length;
        let base = self.filter_alloc * index;
        let f = &self.bank[base..base + l];
        let src = &src[..l];
        let x8 = l & !7;
        let x4 = l & !3;
        let mut a = [0f32; 4];
        let (mut val, mut i);
        if x8 >= 8 {
            for (s8, f8) in src[..x8].chunks_exact(8).zip(f[..x8].chunks_exact(8)) {
                for k in 0..4 {
                    a[k] = s8[k].mul_add(f8[k], a[k]);
                }
                for k in 0..4 {
                    a[k] = s8[4 + k].mul_add(f8[4 + k], a[k]);
                }
            }
            val = (a[0] + a[1]) + (a[2] + a[3]);
            i = x8;
        } else if x4 >= 4 {
            for (s4, f4) in src[..x4].chunks_exact(4).zip(f[..x4].chunks_exact(4)) {
                for k in 0..4 {
                    a[k] = s4[k].mul_add(f4[k], a[k]);
                }
            }
            val = (a[0] + a[1]) + (a[2] + a[3]);
            i = x4;
        } else {
            val = 0.0;
            i = 0;
        }
        // The C tail is not contracted (measured: fused leaves 0.2% of
        // 48 -> 44.1 kHz outputs 1 ulp off, with a filter length of 36).
        while i < l {
            val += src[i] * f[i];
            i += 1;
        }
        val
    }
}

/// `build_filter` for `AV_SAMPLE_FMT_FLTP`, Kaiser window.
fn build_filter(bank: &mut [f32], factor: f64, tap_count: usize, alloc: usize, phase_count: usize, kaiser_beta: f64) {
    use std::f64::consts::PI;
    let ph_nb = if phase_count % 2 == 1 { phase_count } else { phase_count / 2 + 1 };
    let center = (tap_count as i64 - 1) / 2;
    let mut tab = vec![0f64; tap_count + 1];
    let mut sin_lut = vec![0f64; ph_nb];
    let factor = factor.min(1.0);
    if factor == 1.0 {
        for (ph, s) in sin_lut.iter_mut().enumerate() {
            *s = (PI * ph as f64 / phase_count as f64).sin() * if center & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
    let mut norm = 0f64;
    for ph in 0..ph_nb {
        let mut s = sin_lut[ph];
        for i in 0..tap_count {
            let x = PI * ((i as i64 - center) as f64 - ph as f64 / phase_count as f64) * factor;
            let mut y = if x == 0.0 {
                1.0
            } else if factor == 1.0 {
                s / x
            } else {
                x.sin() / x
            };
            let w = 2.0 * x / (factor * tap_count as f64 * PI);
            y *= bessel_i0(kaiser_beta * (1.0 - w * w).max(0.0).sqrt());
            tab[i] = y;
            s = -s;
            if ph == 0 {
                norm += y;
            }
        }
        for i in 0..tap_count {
            bank[ph * alloc + i] = (tab[i] * 1.0 / norm) as f32;
        }
        if phase_count % 2 == 1 {
            continue;
        }
        for i in 0..tap_count {
            bank[(phase_count - ph) * alloc + tap_count - 1 - i] = bank[ph * alloc + i];
        }
    }
}

/// The s16 conversion of one `swr_convert` call: the first `len & !15`
/// samples by NEON, the rest by C `av_clip_int16(lrintf(x * 32768))`.
/// Stereo (`conv_fltp_to_s16_2ch_neon`) narrows `fcvtzs #31` with `sri #16`
/// (truncation); mono (`conv_flt_to_s16_neon`, ffmpeg maps 1-channel
/// formats to planar) with `sqrshrn #16` (rounding).
fn to_s16(out: &mut Vec<i16>, planes: &[&[f32]], len: usize) {
    let off = len & !15;
    let mono = planes.len() == 1;
    for i in 0..len {
        for p in planes {
            let x = p[i];
            out.push(if i < off {
                let t = (x as f64 * 2147483648.0).trunc().clamp(i32::MIN as f64, i32::MAX as f64);
                let t = if t.is_nan() { 0 } else { t as i32 as i64 };
                if mono {
                    ((t + 0x8000) >> 16).clamp(-32768, 32767) as i16
                } else {
                    (t >> 16) as i16
                }
            } else {
                // lrintf: round half to even in float.
                let v = (x * 32768.0).round_ties_even();
                v.clamp(-32768.0, 32767.0) as i16
            });
        }
    }
}

/// Interleaved s16 at `out_rate`, stereo, as the ffmpeg 7.1 CLI writes it.
pub fn ffmpeg_cli_s16_stereo(input: &DecodedFrames, out_rate: u32) -> Result<Vec<i16>, String> {
    ffmpeg_cli_s16(input, out_rate, 2)
}

/// Interleaved s16 at `out_rate` with `out_ch` (1 or 2) channels, as
/// `ffmpeg -i IN -f s16le -ar RATE -ac CH` writes it. The rematrix is
/// `swr_build_matrix2`'s: mono -> stereo `x * (float)M_SQRT1_2` per side;
/// stereo -> mono `0.5f*L + 0.5f*R` (`M_SQRT1_2` each, normalised by their
/// sum because s16 output caps the gain at 1).
pub fn ffmpeg_cli_s16(input: &DecodedFrames, out_rate: u32, out_ch: usize) -> Result<Vec<i16>, String> {
    let in_ch = input.planes.len();
    let n: usize = input.frame_lens.iter().sum();
    if input.planes.iter().any(|p| p.len() != n) {
        return Err("frame lengths do not add up to the planes".into());
    }
    if !(1..=2).contains(&in_ch) || !(1..=2).contains(&out_ch) {
        return Err(format!("{in_ch} -> {out_ch} channels"));
    }
    let up = std::f64::consts::FRAC_1_SQRT_2 as f32;
    let down = (std::f64::consts::FRAC_1_SQRT_2 / (std::f64::consts::FRAC_1_SQRT_2 + std::f64::consts::FRAC_1_SQRT_2)) as f32;
    // swri_rematrix on planar float chunks (C, uncontracted).
    let rematrix = |p: &[&[f32]]| -> Vec<Vec<f32>> {
        match (in_ch, out_ch) {
            (1, 2) => {
                let m: Vec<f32> = p[0].iter().map(|&x| up * x).collect();
                vec![m.clone(), m]
            }
            (2, 1) => vec![p[0].iter().zip(p[1]).map(|(&l, &r)| down * l + down * r).collect()],
            _ => p.iter().map(|c| c.to_vec()).collect(),
        }
    };
    let resample = input.sample_rate != out_rate;
    // resample_first: `out_ch/in_ch - 1` (integer) < `out_rate/(float)in_rate - 1.0`.
    let resample_first = resample
        && (((out_ch / in_ch) as i64 - 1) as f64) < (out_rate as f32 / input.sample_rate as f32) as f64 - 1.0;
    let mut out = Vec::new();
    if !resample {
        let mut at = 0;
        for &len in input.frame_lens {
            let chunk: Vec<&[f32]> = input.planes.iter().map(|p| &p[at..at + len]).collect();
            let m = rematrix(&chunk);
            let refs: Vec<&[f32]> = m.iter().map(|v| v.as_slice()).collect();
            to_s16(&mut out, &refs, len);
            at += len;
        }
        return Ok(out);
    }
    // The resampler runs on the channels it sees: the input's when it goes
    // first, else the rematrixed ones.
    let pre: Vec<Vec<f32>> = if resample_first {
        input.planes.to_vec()
    } else {
        let refs: Vec<&[f32]> = input.planes.iter().map(|p| p.as_slice()).collect();
        rematrix(&refs)
    };
    let post = |ys: &[&[f32]]| -> Vec<Vec<f32>> {
        if resample_first {
            rematrix(ys)
        } else {
            ys.iter().map(|c| c.to_vec()).collect()
        }
    };
    let r = Resampler::new(out_rate as i64, input.sample_rate as i64)?;
    let l = r.filter_length;
    let half = (l - 1) / 2;
    if input.frame_lens.first().copied().unwrap_or(0) < l + 1 {
        return Err("first frame shorter than the resampler's filter".into());
    }
    // Extended signal: x[-half..0) mirrored (x[-k] = x[k]), then x, then
    // the flush reflection, sized once it is known.
    let mut ext: Vec<Vec<f32>> = pre
        .iter()
        .map(|p| {
            let mut e: Vec<f32> = (1..=half).rev().map(|k| p[k]).collect();
            e.extend_from_slice(p);
            e
        })
        .collect();
    let (mut s, mut index, mut frac) = (0usize, 0i64, 0i64);
    let nch = pre.len();
    let mut ys: Vec<Vec<f32>> = vec![Vec::new(); nch];
    let emit = |ext: &Vec<Vec<f32>>, avail: usize, ys: &mut Vec<Vec<f32>>, s: &mut usize, index: &mut i64, frac: &mut i64, limit: usize| {
        let mut produced = 0;
        while *s + l <= avail && produced < limit {
            for c in 0..nch {
                ys[c].push(r.apply(&ext[c][*s..], *index as usize));
            }
            produced += 1;
            *frac += r.dst_incr_mod;
            *index += r.dst_incr_div;
            if *frac >= r.src_incr {
                *frac -= r.src_incr;
                *index += 1;
            }
            while *index >= r.phase_count {
                *s += 1;
                *index -= r.phase_count;
            }
        }
        produced
    };
    let mut avail = half;
    for &len in input.frame_lens {
        avail += len;
        let k = emit(&ext, avail, &mut ys, &mut s, &mut index, &mut frac, usize::MAX);
        let start = ys[0].len() - k;
        let chunk: Vec<&[f32]> = ys.iter().map(|y| &y[start..]).collect();
        let m = post(&chunk);
        let refs: Vec<&[f32]> = m.iter().map(|v| v.as_slice()).collect();
        to_s16(&mut out, &refs, k);
    }
    // resample_flush: reflect min(buffered, L) samples, half-sample symmetric.
    let buffered = avail - s;
    let reflection = buffered.min(l).div_ceil(2);
    for e in &mut ext {
        let end = e.len();
        for j in 0..reflection {
            let v = e[end - 1 - j];
            e.push(v);
        }
    }
    avail += reflection;
    loop {
        // flush_frame: swr_convert(NULL) into 4096-sample frames.
        let k = emit(&ext, avail, &mut ys, &mut s, &mut index, &mut frac, 4096);
        if k == 0 {
            break;
        }
        let start = ys[0].len() - k;
        let chunk: Vec<&[f32]> = ys.iter().map(|y| &y[start..]).collect();
        let m = post(&chunk);
        let refs: Vec<&[f32]> = m.iter().map(|v| v.as_slice()).collect();
        to_s16(&mut out, &refs, k);
    }
    Ok(out)
}

/// A planar-float libswresample context driven call by call, as a decoder
/// drives its own `SwrContext` (Opus's SILK resampler: `filter_size 16`, no
/// rematrix, fltp in and out). `convert` is `swr_convert(out, limit, in,
/// n)`: the input is buffered, at most `limit` outputs are produced; `flush`
/// is `swr_convert(out, limit, NULL, 0)` after `resample_flush`'s
/// reflection. The start is mirrored as `invert_initial_buffer` does.
pub struct SwrStream {
    r: Resampler,
    half: usize,
    /// Per channel: `half` mirrored samples (filled once known), then input.
    ext: Vec<Vec<f32>>,
    nin: usize,
    mirrored: bool,
    s: usize,
    index: i64,
    frac: i64,
    flushed: bool,
    /// Input sample rate.
    pub in_rate: u32,
}

impl SwrStream {
    /// A resampler from `in_rate` to `out_rate` with ffmpeg's `filter_size` taps
    /// per phase (32 is swresample's default).
    pub fn new(in_rate: u32, out_rate: u32, channels: usize, filter_size: usize) -> Result<Self, String> {
        let r = Resampler::with_filter_size(out_rate as i64, in_rate as i64, filter_size)?;
        let half = (r.filter_length - 1) / 2;
        Ok(Self {
            r,
            half,
            ext: vec![vec![0.0; half]; channels],
            nin: 0,
            mirrored: false,
            s: 0,
            index: 0,
            frac: 0,
            flushed: false,
            in_rate,
        })
    }

    /// Append one block of planar input, a slice per channel.
    pub fn push(&mut self, input: &[&[f32]]) {
        for (e, x) in self.ext.iter_mut().zip(input) {
            e.extend_from_slice(x);
        }
        self.nin += input.first().map_or(0, |x| x.len());
        if !self.mirrored && self.nin > self.half {
            for e in self.ext.iter_mut() {
                for k in 1..=self.half {
                    e[self.half - k] = e[self.half + k];
                }
            }
            self.mirrored = true;
        }
    }

    fn emit(&mut self, out: &mut [Vec<f32>], limit: usize) -> usize {
        let l = self.r.filter_length;
        let avail = self.ext[0].len();
        let mut produced = 0;
        // invert_initial_buffer waits for filter_length + 1 input samples.
        if !self.flushed && self.nin < l + 1 {
            return 0;
        }
        while self.s + l <= avail && produced < limit {
            for (c, o) in out.iter_mut().enumerate() {
                o.push(self.r.apply(&self.ext[c][self.s..], self.index as usize));
            }
            produced += 1;
            self.frac += self.r.dst_incr_mod;
            self.index += self.r.dst_incr_div;
            if self.frac >= self.r.src_incr {
                self.frac -= self.r.src_incr;
                self.index += 1;
            }
            while self.index >= self.r.phase_count {
                self.s += 1;
                self.index -= self.r.phase_count;
            }
        }
        produced
    }

    /// Buffer `input`, then append up to `limit` outputs per channel to `out`.
    pub fn convert(&mut self, out: &mut [Vec<f32>], limit: usize, input: &[&[f32]]) -> usize {
        self.push(input);
        self.emit(out, limit)
    }

    /// `swr_convert(out, limit, NULL, 0)`.
    pub fn flush(&mut self, out: &mut [Vec<f32>], limit: usize) -> usize {
        if !self.flushed {
            let buffered = self.ext[0].len() - self.s;
            let reflection = buffered.min(self.r.filter_length).div_ceil(2);
            for e in &mut self.ext {
                let end = e.len();
                for j in 0..reflection {
                    let v = e[end - 1 - j];
                    e.push(v);
                }
            }
            self.flushed = true;
        }
        self.emit(out, limit)
    }
}
