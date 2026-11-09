//! AAC-LC encoding: planar `f32` PCM to raw access units, ADTS, or the
//! `AudioSpecificConfig` an MP4 needs. Mono or stereo, any of the standard
//! sample rates, at a target bitrate.
//!
//! What it does, per 1024-sample frame and channel:
//!
//! 1. A sine-windowed MDCT over 2048 samples (the previous frame and this
//!    one), computed through a DCT-IV on a half-size complex FFT.
//! 2. A low-pass at a cutoff that follows the bitrate, so no bits go to
//!    frequencies the rate cannot afford.
//! 3. For stereo, a per-band choice of left/right or mid/side, by which one
//!    needs fewer bits at the same noise.
//! 4. Power-law quantization (`|x|^(3/4)`, rounding offset 0.4054) with one
//!    step for the frame, found by bisection so the frame fits its share of
//!    the bit budget. A reservoir lets quiet frames lend bits to loud ones.
//! 5. Huffman coding with the best of the eleven spectral codebooks per band,
//!    grouped into sections.
//!
//! What it leaves out: window switching (transients can pre-echo), TNS,
//! perceptual noise substitution, intensity stereo, prediction and a
//! psychoacoustic model. The quantization noise is the same everywhere in a
//! frame (apart from mid/side), which minimises the error for the bits spent
//! but does not hide it. Output decodes with any AAC-LC decoder.
//!
//! ```
//! use threers_codec::aac::encoder::{encode_adts, EncoderConfig};
//! let tone: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
//! let adts = encode_adts(&[tone], &EncoderConfig::new(44_100, 1, 64_000)).unwrap();
//! assert_eq!(&adts[..2], &[0xFF, 0xF1]);
//! ```

use super::tables::{BITS, CODES, NUM_SWB_1024, SF_BITS, SF_CODE, SWB_OFFSET_1024};
use crate::bitstream::BitWriter;

/// Samples per frame and channel.
pub const FRAME: usize = 1024;

/// Samples of delay the encoder adds: the first frame's overlap. A decoder's
/// output starts with this much that is not in the input (MP4 files say so
/// with an edit list).
pub const DELAY: usize = FRAME;

/// The sample rates AAC has an index for.
pub const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// What to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    pub sample_rate: u32,
    /// 1 or 2.
    pub channels: usize,
    /// Target, in bits per second, for all channels together.
    pub bitrate: u32,
}

impl EncoderConfig {
    pub fn new(sample_rate: u32, channels: usize, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
        }
    }
}

/// Why encoding was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// AAC has no code for this sample rate.
    SampleRate(u32),
    /// Only mono and stereo are supported.
    Channels(usize),
    /// The bitrate is under 8 kbit/s per channel or over 320 kbit/s.
    Bitrate(u32),
    /// The channels differ in length, or there are none.
    Input(String),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleRate(r) => write!(f, "AAC has no sample-rate code for {r} Hz"),
            Self::Channels(c) => write!(f, "{c} channels: only mono and stereo are supported"),
            Self::Bitrate(b) => {
                write!(f, "{b} bit/s is outside 8 kbit/s per channel to 320 kbit/s")
            }
            Self::Input(why) => write!(f, "bad input: {why}"),
        }
    }
}

impl std::error::Error for EncodeError {}

// ------------------------------------------------------------------ MDCT

/// A power-of-two complex FFT, forward.
struct Fft {
    n: usize,
    twiddle: Vec<(f64, f64)>,
    reversed: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        Self {
            n,
            twiddle: (0..n / 2)
                .map(|k| {
                    let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    (a.cos(), a.sin())
                })
                .collect(),
            reversed: (0..n)
                .map(|i| {
                    if bits == 0 {
                        0
                    } else {
                        i.reverse_bits() >> (usize::BITS - bits)
                    }
                })
                .collect(),
        }
    }

    fn run(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let j = self.reversed[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let (half, step) = (len / 2, n / len);
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = self.twiddle[k * step];
                    let (a, b) = (start + k, start + k + half);
                    let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }
}

/// The MDCT of `2m` samples to `m` coefficients, through a DCT-IV of the
/// folded block and an `m/2`-point FFT.
///
/// `X[k] = sum x[n] cos(pi/m (n + 1/2 + m/2)(k + 1/2))`, the transform AAC
/// defines (up to the factor 2 the encoder applies).
pub struct Mdct {
    m: usize,
    fft: Fft,
    /// `exp(-i pi n / m)` for the pre-twiddle and `exp(-i pi (k + 1/4) / m)` after.
    pre: Vec<(f64, f64)>,
    post: Vec<(f64, f64)>,
}

impl Mdct {
    pub fn new(m: usize) -> Self {
        assert!(m.is_power_of_two() && m >= 4);
        let angle = |x: f64| -std::f64::consts::PI * x / m as f64;
        Self {
            m,
            fft: Fft::new(m / 2),
            pre: (0..m / 2)
                .map(|n| (angle(n as f64).cos(), angle(n as f64).sin()))
                .collect(),
            post: (0..m / 2)
                .map(|k| (angle(k as f64 + 0.25).cos(), angle(k as f64 + 0.25).sin()))
                .collect(),
        }
    }

    /// DCT-IV: `X[k] = sum u[n] cos(pi/m (n + 1/2)(k + 1/2))`.
    pub fn dct4(&self, u: &[f64], out: &mut [f64]) {
        let (m, h) = (self.m, self.m / 2);
        let (mut re, mut im) = (vec![0.0; h], vec![0.0; h]);
        for n in 0..h {
            let (a, b) = (u[2 * n], u[m - 1 - 2 * n]);
            let (c, s) = self.pre[n];
            re[n] = a * c - b * s;
            im[n] = a * s + b * c;
        }
        self.fft.run(&mut re, &mut im);
        for k in 0..h {
            let (c, s) = self.post[k];
            let (yr, yi) = (re[k] * c - im[k] * s, re[k] * s + im[k] * c);
            out[2 * k] = yr;
            out[m - 1 - 2 * k] = -yi;
        }
    }

    /// `x` has `2m` samples (already windowed); `out` gets `m` coefficients.
    pub fn forward(&self, x: &[f64], out: &mut [f64]) {
        let (m, h) = (self.m, self.m / 2);
        let mut u = vec![0.0; m];
        for n in 0..h {
            u[n] = -x[3 * h - 1 - n] - x[3 * h + n];
            u[h + n] = x[n] - x[m - 1 - n];
        }
        self.dct4(&u, out);
    }
}

// ----------------------------------------------------------- bit counting

/// Collects the bits of a unit while counting them.
struct Sink {
    writer: Option<BitWriter>,
    bits: usize,
}

impl Sink {
    fn writing() -> Self {
        Self {
            writer: Some(BitWriter::new()),
            bits: 0,
        }
    }

    fn put(&mut self, value: u32, n: u32) {
        self.bits += n as usize;
        if let Some(w) = &mut self.writer
            && n > 0
        {
            w.write_bits(value, n);
        }
    }
}

// ------------------------------------------------------------ codebooks

/// Largest magnitude each spectral codebook (1 to 11) can carry; 11 escapes.
const CB_MAX: [i32; 11] = [1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 8191];
/// Values per codeword.
const CB_DIM: [usize; 11] = [4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2];
/// Signed codebooks carry the sign in the codeword; the others after it.
const CB_SIGNED: [bool; 11] = [
    true, true, false, false, true, true, false, false, false, false, false,
];

/// The codeword index of a tuple in codebook `cb` (1-based).
fn tuple_index(cb: usize, v: &[i32]) -> usize {
    match cb {
        1 | 2 => v.iter().fold(0, |a, &x| a * 3 + (x + 1) as usize),
        3 | 4 => v.iter().fold(0, |a, &x| a * 3 + x.unsigned_abs() as usize),
        5 | 6 => v.iter().fold(0, |a, &x| a * 9 + (x + 4) as usize),
        7 | 8 => v.iter().fold(0, |a, &x| a * 8 + x.unsigned_abs() as usize),
        9 | 10 => v.iter().fold(0, |a, &x| a * 13 + x.unsigned_abs() as usize),
        _ => v
            .iter()
            .fold(0, |a, &x| a * 17 + x.unsigned_abs().min(16) as usize),
    }
}

/// Bits for the escape sequence after a value of 16 or more.
fn escape_bits(magnitude: i32) -> u32 {
    let n = 31 - (magnitude as u32).leading_zeros(); // floor(log2), at least 4
    (n - 4) + 1 + n
}

/// Bits a tuple costs in a codebook.
fn tuple_bits(cb: usize, v: &[i32]) -> u32 {
    let mut bits = u32::from(BITS[cb - 1][tuple_index(cb, v)]);
    if !CB_SIGNED[cb - 1] {
        bits += v.iter().filter(|&&x| x != 0).count() as u32;
    }
    if cb == 11 {
        bits += v
            .iter()
            .filter(|&&x| x.abs() >= 16)
            .map(|&x| escape_bits(x.abs()))
            .sum::<u32>();
    }
    bits
}

fn put_tuple(out: &mut Sink, cb: usize, v: &[i32]) {
    let idx = tuple_index(cb, v);
    out.put(u32::from(CODES[cb - 1][idx]), u32::from(BITS[cb - 1][idx]));
    if !CB_SIGNED[cb - 1] {
        for &x in v {
            if x != 0 {
                out.put(u32::from(x < 0), 1);
            }
        }
    }
    if cb == 11 {
        for &x in v {
            if x.abs() >= 16 {
                let m = x.unsigned_abs();
                let n = 31 - m.leading_zeros();
                // n - 4 ones, a zero, then the n low bits.
                for _ in 0..n - 4 {
                    out.put(1, 1);
                }
                out.put(0, 1);
                out.put(m & ((1 << n) - 1), n);
            }
        }
    }
}

/// Bits to code the coefficients `q[start..end]` in codebook `cb`; `None` if
/// a value does not fit.
fn band_bits(cb: usize, q: &[i32]) -> Option<u32> {
    let dim = CB_DIM[cb - 1];
    if q.iter().any(|&x| x.abs() > CB_MAX[cb - 1]) {
        return None;
    }
    Some(q.chunks(dim).map(|c| tuple_bits(cb, c)).sum())
}

/// A run of bands sharing a codebook.
#[derive(Clone, Copy, Debug)]
struct Section {
    cb: usize,
    first: usize,
    end: usize,
}

/// How a channel's quantized spectrum is coded.
struct Plan {
    sections: Vec<Section>,
    /// Bits in the section data, the scalefactors and the spectrum.
    bits: u32,
}

/// Picks the cheapest codebook for each band, merges runs into sections and
/// totals the bits (everything after `global_gain`, before the spectrum's
/// own flags).
fn plan(q: &[i32], offsets: &[u16], sf: &[i32], max_sfb: usize) -> Plan {
    let mut cbs = Vec::with_capacity(max_sfb);
    let mut spectral = 0u32;
    for b in 0..max_sfb {
        let band = &q[usize::from(offsets[b])..usize::from(offsets[b + 1])];
        if band.iter().all(|&x| x == 0) {
            cbs.push(0);
            continue;
        }
        let (cb, bits) = (1..=11)
            .filter_map(|cb| band_bits(cb, band).map(|bits| (cb, bits)))
            .min_by_key(|&(_, bits)| bits)
            .expect("book 11 takes any value");
        cbs.push(cb);
        spectral += bits;
    }
    // Sections: equal neighbours merge.
    let mut sections: Vec<Section> = Vec::new();
    for (b, &cb) in cbs.iter().enumerate() {
        match sections.last_mut() {
            Some(s) if s.cb == cb => s.end = b + 1,
            _ => sections.push(Section {
                cb,
                first: b,
                end: b + 1,
            }),
        }
    }
    let mut bits = spectral;
    for s in &sections {
        bits += 4 + 5 * ((s.end - s.first) as u32 / 31 + 1);
    }
    let mut prev = None;
    for (b, &cb) in cbs.iter().enumerate() {
        if cb != 0 {
            let delta = sf[b] - prev.unwrap_or(sf[b]);
            bits += u32::from(SF_BITS[(delta + 60) as usize]);
            prev = Some(sf[b]);
        }
    }
    Plan { sections, bits }
}

/// Writes section data, scalefactors and spectrum for one channel.
fn write_channel_body(out: &mut Sink, q: &[i32], offsets: &[u16], sf: &[i32], plan: &Plan) {
    for s in &plan.sections {
        out.put(s.cb as u32, 4);
        let mut len = s.end - s.first;
        while len >= 31 {
            out.put(31, 5);
            len -= 31;
        }
        out.put(len as u32, 5);
    }
    let mut prev = None;
    for s in &plan.sections {
        if s.cb == 0 {
            continue;
        }
        for b in s.first..s.end {
            let delta = sf[b] - prev.unwrap_or(sf[b]);
            let i = (delta + 60) as usize;
            out.put(SF_CODE[i], u32::from(SF_BITS[i]));
            prev = Some(sf[b]);
        }
    }
    // pulse_data_present, tns_data_present, gain_control_data_present.
    out.put(0, 3);
    for s in &plan.sections {
        if s.cb == 0 {
            continue;
        }
        let dim = CB_DIM[s.cb - 1];
        for b in s.first..s.end {
            for c in q[usize::from(offsets[b])..usize::from(offsets[b + 1])].chunks(dim) {
                put_tuple(out, s.cb, c);
            }
        }
    }
}

// -------------------------------------------------------------- quantizer

/// The rounding offset the AAC reference encoders use.
const ROUNDING: f64 = 0.4054;

/// `2^(3 (sf - 100) / 16)`: the step to the 3/4 power.
fn step34(sf: i32) -> f64 {
    (3.0 * f64::from(sf - 100) / 16.0).exp2()
}

/// The largest quantized magnitude AAC can code: the escape sequence tops out here.
const MAX_QUANTIZED: f64 = 8191.0;

/// Quantizes the bands below `max_sfb`. Returns false if any value is too big
/// to code (the step is too fine), in which case `q` is not usable.
fn quantize(companded: &[f64], offsets: &[u16], sf: &[i32], max_sfb: usize, q: &mut [i32]) -> bool {
    let mut fits = true;
    for b in 0..max_sfb {
        let inv = 1.0 / step34(sf[b]);
        for k in usize::from(offsets[b])..usize::from(offsets[b + 1]) {
            let v = (companded[k] * inv + ROUNDING).floor();
            fits &= v <= MAX_QUANTIZED;
            q[k] = v.min(MAX_QUANTIZED) as i32;
        }
    }
    fits
}

// ---------------------------------------------------------------- encoder

/// Encodes a stream frame by frame.
pub struct AacEncoder {
    cfg: EncoderConfig,
    sf_index: usize,
    offsets: &'static [u16],
    num_swb: usize,
    window: Vec<f64>,
    mdct: Mdct,
    /// The previous frame's samples, per channel, for the overlap.
    previous: Vec<Vec<f64>>,
    /// Samples pushed but not yet in a frame, per channel.
    pending: Vec<Vec<f32>>,
    /// Bands at or above this are never coded (the low-pass).
    cutoff_bands: usize,
    /// Bits saved (or owed) against the average, for the reservoir.
    bank: i64,
    average_bits: i64,
    frames: u64,
    /// Samples pushed in total, for the flush.
    pushed: u64,
}

/// The MDCT scale: the transform's factor 2, and 32768 for the 16-bit range
/// AAC's scalefactors are defined against.
const SCALE: f64 = 2.0 * 32_768.0;

impl AacEncoder {
    pub fn new(cfg: EncoderConfig) -> Result<Self, EncodeError> {
        let sf_index = SAMPLE_RATES
            .iter()
            .position(|&r| r == cfg.sample_rate)
            .ok_or(EncodeError::SampleRate(cfg.sample_rate))?;
        if !(1..=2).contains(&cfg.channels) {
            return Err(EncodeError::Channels(cfg.channels));
        }
        if cfg.bitrate < 8_000 * cfg.channels as u32 || cfg.bitrate > 320_000 {
            return Err(EncodeError::Bitrate(cfg.bitrate));
        }
        let offsets = SWB_OFFSET_1024[sf_index];
        let num_swb = usize::from(NUM_SWB_1024[sf_index]);
        // About a quarter of the per-channel bitrate in Hz: 64 kbit/s a
        // channel keeps 16 kHz, 32 kbit/s keeps 8 kHz.
        let cutoff_hz = (f64::from(cfg.bitrate) / cfg.channels as f64 * 0.25)
            .clamp(4_000.0, f64::from(cfg.sample_rate) / 2.0);
        let cutoff_bin = (cutoff_hz / (f64::from(cfg.sample_rate) / 2.0) * FRAME as f64) as usize;
        let cutoff_bands = (0..num_swb)
            .take_while(|&b| usize::from(offsets[b + 1]) <= cutoff_bin.max(usize::from(offsets[1])))
            .count()
            .max(1);
        Ok(Self {
            cfg,
            sf_index,
            offsets,
            num_swb,
            window: (0..2 * FRAME)
                .map(|n| (std::f64::consts::PI * (n as f64 + 0.5) / (2 * FRAME) as f64).sin())
                .collect(),
            mdct: Mdct::new(FRAME),
            previous: vec![vec![0.0; FRAME]; cfg.channels],
            pending: vec![Vec::new(); cfg.channels],
            cutoff_bands,
            bank: 0,
            average_bits: (f64::from(cfg.bitrate) * FRAME as f64 / f64::from(cfg.sample_rate))
                as i64,
            frames: 0,
            pushed: 0,
        })
    }

    /// The two-byte `AudioSpecificConfig` for an MP4 `esds`.
    pub fn audio_specific_config(&self) -> [u8; 2] {
        // objectType 2 (5 bits), frequency index (4), channels (4), 3 zero bits.
        let v = (2u16 << 11) | ((self.sf_index as u16) << 7) | ((self.cfg.channels as u16) << 3);
        v.to_be_bytes()
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// Adds samples (one slice per channel, equal lengths) and returns the
    /// raw access units for every frame that is now complete.
    pub fn push(&mut self, planar: &[&[f32]]) -> Result<Vec<Vec<u8>>, EncodeError> {
        if planar.len() != self.cfg.channels {
            return Err(EncodeError::Input(format!(
                "{} channels given, {} expected",
                planar.len(),
                self.cfg.channels
            )));
        }
        if planar.iter().any(|c| c.len() != planar[0].len()) {
            return Err(EncodeError::Input("the channels differ in length".into()));
        }
        for (p, c) in self.pending.iter_mut().zip(planar) {
            p.extend_from_slice(c);
        }
        self.pushed += planar[0].len() as u64;
        let mut units = Vec::new();
        while self.pending[0].len() >= FRAME {
            let frame: Vec<Vec<f32>> = self
                .pending
                .iter_mut()
                .map(|p| p.drain(..FRAME).collect())
                .collect();
            units.push(self.encode_frame(&frame));
        }
        Ok(units)
    }

    /// Ends the stream: pads with silence so the last samples (and the
    /// overlap that carries them) come out, and returns those frames.
    pub fn finish(&mut self) -> Vec<Vec<u8>> {
        let mut units = Vec::new();
        let owed = self.pending[0].len();
        // The remainder, padded to a frame, then one frame of silence to
        // flush the overlap.
        let extra = if owed > 0 { 2 } else { 1 };
        for round in 0..extra {
            let frame: Vec<Vec<f32>> = self
                .pending
                .iter_mut()
                .map(|p| {
                    let mut f: Vec<f32> = if round == 0 {
                        std::mem::take(p)
                    } else {
                        Vec::new()
                    };
                    f.resize(FRAME, 0.0);
                    f
                })
                .collect();
            units.push(self.encode_frame(&frame));
        }
        units
    }

    fn encode_frame(&mut self, frame: &[Vec<f32>]) -> Vec<u8> {
        let channels = self.cfg.channels;
        // MDCT of each channel.
        let mut spectra: Vec<Vec<f64>> = Vec::with_capacity(channels);
        for (ch, new) in frame.iter().enumerate() {
            let mut block = Vec::with_capacity(2 * FRAME);
            block.extend_from_slice(&self.previous[ch]);
            block.extend(new.iter().map(|&s| f64::from(s)));
            self.previous[ch] = block[FRAME..].to_vec();
            for (s, w) in block.iter_mut().zip(&self.window) {
                *s *= w;
            }
            let mut spec = vec![0.0; FRAME];
            self.mdct.forward(&block, &mut spec);
            for s in &mut spec {
                *s *= SCALE;
            }
            // The low-pass.
            for s in &mut spec[usize::from(self.offsets[self.cutoff_bands])..] {
                *s = 0.0;
            }
            spectra.push(spec);
        }

        // Mid/side per band, where it needs fewer bits.
        let mut ms = vec![false; self.num_swb];
        if channels == 2 {
            for b in 0..self.cutoff_bands {
                let (lo, hi) = (
                    usize::from(self.offsets[b]),
                    usize::from(self.offsets[b + 1]),
                );
                let (mut lr, mut msc) = (0.0, 0.0);
                for k in lo..hi {
                    let (l, r) = (spectra[0][k], spectra[1][k]);
                    lr += l.abs().powf(0.75) + r.abs().powf(0.75);
                    msc += ((l + r) * 0.5).abs().powf(0.75) + ((l - r) * 0.5).abs().powf(0.75);
                }
                // Mid/side spends a finer step on the same noise: it has to
                // win by more than that costs.
                ms[b] = msc * 1.25 < lr;
            }
            for (b, &on) in ms.iter().enumerate() {
                if on {
                    for k in usize::from(self.offsets[b])..usize::from(self.offsets[b + 1]) {
                        let (l, r) = (spectra[0][k], spectra[1][k]);
                        spectra[0][k] = (l + r) * 0.5;
                        spectra[1][k] = (l - r) * 0.5;
                    }
                }
            }
        }

        let companded: Vec<Vec<f64>> = spectra
            .iter()
            .map(|s| s.iter().map(|x| x.abs().powf(0.75)).collect())
            .collect();
        let signs: Vec<Vec<bool>> = spectra
            .iter()
            .map(|s| s.iter().map(|&x| x < 0.0).collect())
            .collect();

        // The bit budget: the average plus half of whatever is banked.
        // A frame cannot hold more than 6144 bits a channel, whatever was
        // asked for: at a low sample rate a high bitrate asks for more.
        let most = 6144 * channels as i64 - 100;
        let budget =
            (self.average_bits + self.bank / 2).clamp(self.average_bits.min(most) / 2, most) as u32;

        // The finest step (lowest scalefactor) that fits.
        let (mut lo, mut hi) = (0i32, 255i32);
        let mut best: Option<Encoded> = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let e = self.try_step(mid, &companded, &signs, &ms);
            if e.bits <= budget {
                best = Some(e);
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        let e = best.unwrap_or_else(|| self.try_step(255, &companded, &signs, &ms));
        let bytes = self.write_unit(&e, &ms);
        self.bank = (self.bank + self.average_bits - (bytes.len() as i64) * 8)
            .clamp(-self.average_bits * 4, self.average_bits * 4);
        self.frames += 1;
        bytes
    }

    /// Quantizes everything at scalefactor `g` (mid/side bands a little finer)
    /// and plans the coding.
    fn try_step(
        &self,
        g: i32,
        companded: &[Vec<f64>],
        signs: &[Vec<bool>],
        ms: &[bool],
    ) -> Encoded {
        let mut channels = Vec::new();
        let mut total = 0;
        let mut max_sfb_all = 0;
        let mut qs = Vec::new();
        for ch in 0..self.cfg.channels {
            let sf: Vec<i32> = (0..self.num_swb)
                .map(|b| if ms[b] { (g - 2).max(0) } else { g })
                .collect();
            let mut q = vec![0i32; FRAME];
            if !quantize(&companded[ch], self.offsets, &sf, self.cutoff_bands, &mut q) {
                // Too fine a step for the loudest coefficient: not codable.
                return Encoded {
                    channels: Vec::new(),
                    max_sfb: 0,
                    bits: u32::MAX,
                };
            }
            let top = (0..self.cutoff_bands)
                .rev()
                .find(|&b| {
                    q[usize::from(self.offsets[b])..usize::from(self.offsets[b + 1])]
                        .iter()
                        .any(|&x| x != 0)
                })
                .map_or(0, |b| b + 1);
            max_sfb_all = max_sfb_all.max(top);
            qs.push((q, sf));
        }
        for (ch, (q, sf)) in qs.into_iter().enumerate() {
            let mut q = q;
            for (v, &neg) in q.iter_mut().zip(&signs[ch]) {
                if neg {
                    *v = -*v;
                }
            }
            channels.push(ChannelData { q, sf, plan: None });
        }
        for c in &mut channels {
            let p = plan(&c.q, self.offsets, &c.sf, max_sfb_all);
            // global_gain, and the three "present" flags the body ends with.
            total += p.bits + 8 + 3;
            c.plan = Some(p);
        }
        // The element header, its window info, the mid/side flags, the END
        // marker, and on average half a byte of alignment.
        total += if self.cfg.channels == 1 {
            3 + 4 + 11
        } else {
            3 + 4
                + 1
                + 11
                + 2
                + if ms.iter().take(max_sfb_all).any(|&m| m) {
                    max_sfb_all as u32
                } else {
                    0
                }
        };
        total += 3 + 4;
        Encoded {
            channels,
            max_sfb: max_sfb_all,
            bits: total,
        }
    }

    fn write_unit(&self, e: &Encoded, ms: &[bool]) -> Vec<u8> {
        let mut out = Sink::writing();
        let ics_info = |out: &mut Sink| {
            out.put(0, 1); // reserved
            out.put(0, 2); // ONLY_LONG_SEQUENCE
            out.put(0, 1); // sine window
            out.put(e.max_sfb as u32, 6);
            out.put(0, 1); // no prediction
        };
        // The first coded band's scalefactor is the starting point for the
        // differences.
        let global = |c: &ChannelData| -> u32 {
            let p = c.plan.as_ref().expect("planned");
            p.sections
                .iter()
                .find(|s| s.cb != 0)
                .map_or(100, |s| c.sf[s.first]) as u32
        };
        if self.cfg.channels == 1 {
            out.put(0, 3); // SCE
            out.put(0, 4);
            let c = &e.channels[0];
            out.put(global(c), 8);
            ics_info(&mut out);
            write_channel_body(
                &mut out,
                &c.q,
                self.offsets,
                &c.sf,
                c.plan.as_ref().expect("planned"),
            );
        } else {
            out.put(1, 3); // CPE
            out.put(0, 4);
            out.put(1, 1); // common window
            ics_info(&mut out);
            let any = ms.iter().take(e.max_sfb).any(|&m| m);
            out.put(if any { 1 } else { 0 }, 2);
            if any {
                for &m in ms.iter().take(e.max_sfb) {
                    out.put(u32::from(m), 1);
                }
            }
            for c in &e.channels {
                out.put(global(c), 8);
                write_channel_body(
                    &mut out,
                    &c.q,
                    self.offsets,
                    &c.sf,
                    c.plan.as_ref().expect("planned"),
                );
            }
        }
        out.put(7, 3); // END
        let mut w = out.writer.expect("writing");
        w.align_zero();
        w.finish()
    }

    /// Wraps a raw access unit in an ADTS header.
    pub fn adts(&self, unit: &[u8]) -> Vec<u8> {
        adts_frame(unit, self.sf_index, self.cfg.channels)
    }
}

struct ChannelData {
    q: Vec<i32>,
    sf: Vec<i32>,
    plan: Option<Plan>,
}

struct Encoded {
    channels: Vec<ChannelData>,
    max_sfb: usize,
    bits: u32,
}

/// An ADTS header (MPEG-4, LC, no CRC, variable-rate buffer fullness) and
/// the access unit.
pub fn adts_frame(unit: &[u8], sf_index: usize, channels: usize) -> Vec<u8> {
    let len = unit.len() + 7;
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF1; // MPEG-4, layer 0, no CRC
    h[2] = (1 << 6) | ((sf_index as u8) << 2) | ((channels as u8 >> 2) & 1); // profile 1 = LC
    h[3] = (((channels as u8) & 3) << 6) | ((len >> 11) as u8 & 3);
    h[4] = (len >> 3) as u8;
    h[5] = (((len & 7) as u8) << 5) | 0x1F;
    h[6] = 0xFC;
    let mut out = h.to_vec();
    out.extend_from_slice(unit);
    out
}

/// Encodes whole channels to an ADTS stream.
pub fn encode_adts(planar: &[Vec<f32>], cfg: &EncoderConfig) -> Result<Vec<u8>, EncodeError> {
    let mut enc = AacEncoder::new(*cfg)?;
    let views: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let mut out = Vec::new();
    for u in enc.push(&views)?.into_iter().chain(enc.finish()) {
        out.extend(enc.adts(&u));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdct_matches_its_definition() {
        let m = 16;
        let x: Vec<f64> = (0..2 * m)
            .map(|i| ((i * 7 + 3) % 11) as f64 - 5.0)
            .collect();
        let mut fast = vec![0.0; m];
        Mdct::new(m).forward(&x, &mut fast);
        for k in 0..m {
            let direct: f64 = (0..2 * m)
                .map(|n| {
                    x[n] * (std::f64::consts::PI / m as f64
                        * (n as f64 + 0.5 + m as f64 / 2.0)
                        * (k as f64 + 0.5))
                        .cos()
                })
                .sum();
            assert!(
                (fast[k] - direct).abs() < 1e-9,
                "bin {k}: {} vs {direct}",
                fast[k]
            );
        }
    }

    #[test]
    fn the_dct4_matches_its_definition() {
        let m = 32;
        let u: Vec<f64> = (0..m).map(|i| ((i * 5 + 1) % 9) as f64 - 4.0).collect();
        let mut fast = vec![0.0; m];
        Mdct::new(m).dct4(&u, &mut fast);
        for k in 0..m {
            let direct: f64 = (0..m)
                .map(|n| {
                    u[n] * (std::f64::consts::PI / m as f64 * (n as f64 + 0.5) * (k as f64 + 0.5))
                        .cos()
                })
                .sum();
            assert!((fast[k] - direct).abs() < 1e-9, "bin {k}");
        }
    }

    #[test]
    fn the_asc_and_adts_header_are_right() {
        let enc = AacEncoder::new(EncoderConfig::new(44_100, 2, 128_000)).unwrap();
        // AOT 2, index 4, 2 channels: 00010 0100 0010 000.
        assert_eq!(enc.audio_specific_config(), [0x12, 0x10]);
        let f = adts_frame(&[1, 2, 3], 4, 2);
        assert_eq!(&f[..2], &[0xFF, 0xF1]);
        assert_eq!(f.len(), 10);
        let len = (usize::from(f[3] & 3) << 11) | (usize::from(f[4]) << 3) | usize::from(f[5] >> 5);
        assert_eq!(len, 10);
        assert_eq!(f[2] >> 6, 1, "LC");
        assert_eq!((f[2] >> 2) & 15, 4);
        assert_eq!(((f[2] & 1) << 2) | (f[3] >> 6), 2);
    }

    #[test]
    fn bad_settings_are_refused() {
        assert_eq!(
            AacEncoder::new(EncoderConfig::new(12_345, 2, 128_000)).err(),
            Some(EncodeError::SampleRate(12_345))
        );
        assert_eq!(
            AacEncoder::new(EncoderConfig::new(44_100, 3, 128_000)).err(),
            Some(EncodeError::Channels(3))
        );
        assert_eq!(
            AacEncoder::new(EncoderConfig::new(44_100, 2, 8_000)).err(),
            Some(EncodeError::Bitrate(8_000))
        );
        let mut enc = AacEncoder::new(EncoderConfig::new(44_100, 2, 128_000)).unwrap();
        assert!(matches!(
            enc.push(&[&[0.0; 10][..]]),
            Err(EncodeError::Input(_))
        ));
        assert!(matches!(
            enc.push(&[&[0.0; 10][..], &[0.0; 9][..]]),
            Err(EncodeError::Input(_))
        ));
    }

    #[test]
    fn escape_sequences_have_the_right_length() {
        // 16 = 2^4: no ones, a zero, four bits.
        assert_eq!(escape_bits(16), 5);
        assert_eq!(escape_bits(31), 5);
        assert_eq!(escape_bits(32), 7);
        assert_eq!(escape_bits(8191), 8 + 1 + 12);
    }
}
