//! MPEG-1 Layer III encoding: planar `f32` PCM to an `.mp3` file at 32,
//! 44.1 or 48 kHz, mono or stereo, at a constant bitrate.
//!
//! The chain, per granule (576 samples) and channel:
//!
//! 1. The 32-band polyphase analysis filterbank (ISO 11172-3 annex C), whose
//!    prototype is the decoder's synthesis window scaled by 1/32.
//! 2. An 18-point MDCT per subband with the sine window, after the sign flip
//!    that undoes the filterbank's frequency inversion, and the alias-reduction
//!    butterflies between neighbouring subbands.
//! 3. A low-pass that follows the bitrate.
//! 4. For stereo, a per-frame choice of left/right or mid/side.
//! 5. Power-law quantization (`|x|^(3/4)`, rounding offset 0.4054) with one
//!    global step per granule and channel, found by bisection so that the
//!    granule fits its share of the frame.
//! 6. Huffman coding with the best of the 32 big-value tables for each of
//!    three regions (the region split is searched too), and the better of
//!    the two count-1 tables.
//!
//! What it leaves out: block switching (short blocks, so sharp attacks
//! pre-echo), scalefactors (the quantization noise is flat across the
//! spectrum), the bit reservoir (every frame is self-contained), intensity
//! stereo, a psychoacoustic model, and MPEG-2 sample rates. The output
//! decodes with any MP3 decoder. [`encode_mp3`] puts a Xing/LAME `Info` frame
//! first, which carries the encoder delay and padding, so gapless players
//! trim the file to exactly the input's length.
//!
//! ```
//! use threers_codec::mp3::encoder::{encode_mp3, Mp3Config};
//! let tone: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
//! let mp3 = encode_mp3(&[tone], &Mp3Config::new(44_100, 1, 64)).unwrap();
//! assert!(mp3.len() > 1_000);
//! ```

use std::sync::OnceLock;

use super::tables::{BAND_INFO, HT, HTC, INTWINBASE};
use crate::bitstream::BitWriter;

/// Samples per frame and channel.
pub const FRAME: usize = 1152;
const GRANULE: usize = 576;

/// Samples a decoder's output is late by, counting its own 529: where the
/// input's first sample lands when the stream is decoded without the `Info`
/// frame's help. The filterbank takes 481 and the transform's overlap 576.
pub const DELAY: usize = 1057;

/// MPEG-1 Layer III bitrates, kbit/s, by header index 1 to 14.
pub const BITRATES: [u32; 14] = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];

/// The MPEG-1 sample rates, in header order.
pub const SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

/// What to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mp3Config {
    pub sample_rate: u32,
    /// 1 or 2.
    pub channels: usize,
    /// Constant bitrate in kbit/s: one of [`BITRATES`].
    pub kbps: u32,
}

impl Mp3Config {
    pub fn new(sample_rate: u32, channels: usize, kbps: u32) -> Self {
        Self { sample_rate, channels, kbps }
    }
}

/// Why encoding was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp3Error {
    /// Only 32, 44.1 and 48 kHz (MPEG-1) are supported.
    SampleRate(u32),
    Channels(usize),
    /// Not one of [`BITRATES`].
    Bitrate(u32),
    Input(String),
}

impl std::fmt::Display for Mp3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleRate(r) => write!(f, "{r} Hz: MPEG-1 Layer III is 32, 44100 or 48000 Hz"),
            Self::Channels(c) => write!(f, "{c} channels: only mono and stereo are supported"),
            Self::Bitrate(b) => write!(f, "{b} kbit/s is not an MPEG-1 Layer III bitrate"),
            Self::Input(why) => write!(f, "bad input: {why}"),
        }
    }
}

impl std::error::Error for Mp3Error {}

// ------------------------------------------------------------- huffman

/// A big-value table: code lengths and codes for each pair of 0..=15, and
/// how large a value it can carry.
struct PairTable {
    len: [[u8; 16]; 16],
    code: [[u16; 16]; 16],
    linbits: u32,
    /// The largest value the table can code (with its linbits).
    max: i32,
}

/// A count-1 table: code lengths and codes by the quad's four-bit pattern.
struct QuadTable {
    len: [u8; 16],
    code: [u8; 16],
}

struct Tables {
    pairs: Vec<Option<PairTable>>,
    quads: [QuadTable; 2],
    /// The analysis window C[i] = D[i] / 32.
    window: [f64; 512],
    /// cos((2k+1)(i-16)pi/64).
    matrix: Vec<[f64; 64]>,
    /// The 36-point MDCT: window and cosines.
    mdct_window: [f64; 36],
    mdct_cos: Vec<[f64; 36]>,
}

/// Walks a radix-16 table (the decoder's `huff_pair` layout), recording each
/// leaf's code. A non-negative entry is `length << 8 | x << 4 | y` for the
/// bits left in this nibble; a negative one points to a sub-table.
fn walk_pairs(tab: &[i16], base: usize, prefix: u32, prefix_len: u32, len: &mut [[u8; 16]; 16], code: &mut [[u16; 16]; 16]) {
    for idx in 0..16usize {
        let e = tab[base + idx];
        if e < 0 {
            walk_pairs(tab, base + (-e) as usize, (prefix << 4) | idx as u32, prefix_len + 4, len, code);
        } else {
            let l = (e >> 8) as u32;
            // A leaf of length l fills 2^(4-l) entries; take its first.
            if l == 0 || idx & ((1 << (4 - l)) - 1) != 0 {
                continue;
            }
            let (x, y) = (((e >> 4) & 15) as usize, (e & 15) as usize);
            len[x][y] = (prefix_len + l) as u8;
            code[x][y] = ((prefix << l) | (idx as u32 >> (4 - l))) as u16;
        }
    }
}

/// Walks a bit-serial count-1 table.
fn walk_quads(tab: &[i16], at: usize, prefix: u32, depth: u32, out: &mut QuadTable) {
    let a = tab[at];
    if a >= 0 {
        out.len[a as usize] = depth as u8;
        out.code[a as usize] = prefix as u8;
    } else {
        walk_quads(tab, at + 1, prefix << 1, depth + 1, out);
        walk_quads(tab, at + 1 + (-a) as usize, (prefix << 1) | 1, depth + 1, out);
    }
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut pairs = Vec::with_capacity(32);
        for (n, (linbits, tab)) in HT.iter().enumerate() {
            // Tables 4 and 14 do not exist.
            if n == 4 || n == 14 {
                pairs.push(None);
                continue;
            }
            let (mut len, mut code) = ([[0u8; 16]; 16], [[0u16; 16]; 16]);
            walk_pairs(tab, 0, 0, 0, &mut len, &mut code);
            if n == 0 {
                // Table 0 carries only zeros, in no bits.
                pairs.push(Some(PairTable { len, code, linbits: 0, max: 0 }));
                continue;
            }
            let tree_max = (0..16).filter(|&x| (0..16).any(|y| len[x][y] > 0 || len[y][x] > 0)).max().unwrap_or(0) as i32;
            let max = if *linbits > 0 { 15 + (1 << linbits) - 1 } else { tree_max };
            pairs.push(Some(PairTable { len, code, linbits: *linbits, max }));
        }
        let mut quads = [QuadTable { len: [0; 16], code: [0; 16] }, QuadTable { len: [0; 16], code: [0; 16] }];
        for (q, (_, tab)) in quads.iter_mut().zip(HTC.iter()) {
            walk_quads(tab, 0, 0, 0, q);
        }
        // D[i] = +/- base[i] / 65536 with the sign changing every 64 entries;
        // the base is symmetric about 256. C = D / 32.
        let mut window = [0.0; 512];
        for (i, w) in window.iter_mut().enumerate() {
            let base = INTWINBASE[if i <= 256 { i } else { 512 - i }] as f64 / 65_536.0;
            let sign = if (i / 64) % 2 == 0 { 1.0 } else { -1.0 };
            *w = sign * base / 32.0;
        }
        let matrix = (0..32)
            .map(|k| {
                let mut row = [0.0; 64];
                for (i, r) in row.iter_mut().enumerate() {
                    *r = ((2 * k + 1) as f64 * (i as f64 - 16.0) * std::f64::consts::PI / 64.0).cos();
                }
                row
            })
            .collect();
        let mut mdct_window = [0.0; 36];
        for (i, w) in mdct_window.iter_mut().enumerate() {
            *w = (std::f64::consts::PI / 36.0 * (i as f64 + 0.5)).sin();
        }
        let mdct_cos = (0..18)
            .map(|k| {
                let mut row = [0.0; 36];
                for (i, r) in row.iter_mut().enumerate() {
                    *r = (std::f64::consts::PI / 72.0 * (2 * i + 1 + 18) as f64 * (2 * k + 1) as f64).cos();
                }
                row
            })
            .collect();
        Tables { pairs, quads, window, matrix, mdct_window, mdct_cos }
    })
}

/// Bits for one pair of magnitudes in a table; `None` if it cannot code them.
fn pair_bits(t: &PairTable, x: i32, y: i32) -> Option<u32> {
    if x > t.max || y > t.max {
        return None;
    }
    let (cx, cy) = if t.linbits > 0 { (x.min(15), y.min(15)) } else { (x, y) };
    let mut bits = u32::from(t.len[cx as usize][cy as usize]);
    if t.linbits > 0 {
        if x >= 15 {
            bits += t.linbits;
        }
        if y >= 15 {
            bits += t.linbits;
        }
    }
    Some(bits + u32::from(x != 0) + u32::from(y != 0))
}

// -------------------------------------------------------- granule coding

/// A granule of one channel, quantized and ready to write.
#[derive(Clone)]
struct Granule {
    global_gain: u32,
    /// Signed.
    ix: [i32; GRANULE],
    big_values: usize,
    count1_quads: usize,
    count1_table: usize,
    region0_count: usize,
    region1_count: usize,
    tables: [usize; 3],
    bits: u32,
}

impl Granule {
    fn silent() -> Self {
        Self { global_gain: 0, ix: [0; GRANULE], big_values: 0, count1_quads: 0, count1_table: 0, region0_count: 0, region1_count: 0, tables: [0; 3], bits: 0 }
    }
}

/// The sfb edge at `index`, clamped: past the last band is the end.
fn edge(long: &[u16; 23], index: usize) -> usize {
    long.get(index).map_or(GRANULE, |&e| usize::from(e))
}

/// Chooses the coding of magnitudes `ix` (signs ignored here): where the
/// big values end, the count-1 table, the regions and a table per region.
fn code_granule(ix: &[i32; GRANULE], long: &[u16; 23]) -> Option<Granule> {
    let t = tables();
    let mut end = GRANULE;
    while end >= 2 && ix[end - 1] == 0 && ix[end - 2] == 0 {
        end -= 2;
    }
    // Count-1 quads from the top while every value is 0 or 1.
    let mut big = end;
    while big >= 4 && ix[big - 4..big].iter().all(|&v| v <= 1) {
        big -= 4;
    }
    let quads = (end - big) / 4;
    let (mut best_count1, mut count1_table) = (u32::MAX, 0);
    for (n, q) in t.quads.iter().enumerate() {
        let mut bits = 0;
        for c in ix[big..end].chunks(4) {
            let m = (c[0] << 3 | c[1] << 2 | c[2] << 1 | c[3]) as usize;
            bits += u32::from(q.len[m]) + c.iter().filter(|&&v| v != 0).count() as u32;
        }
        if bits < best_count1 {
            best_count1 = bits;
            count1_table = n;
        }
    }
    let pairs = big / 2;
    if ix[..big].iter().any(|&v| v > 15 + (1 << 13) - 1) {
        return None;
    }

    // Per table, running totals of the bits for the pairs, and of the pairs it
    // cannot code, so any region's best table is a subtraction.
    let usable: Vec<usize> = (0..32).filter(|&n| t.pairs[n].is_some()).collect();
    let mut cost = vec![vec![0u32; pairs + 1]; 32];
    let mut bad = vec![vec![0u32; pairs + 1]; 32];
    for &n in &usable {
        let table = t.pairs[n].as_ref().expect("usable");
        for p in 0..pairs {
            let (x, y) = (ix[2 * p], ix[2 * p + 1]);
            let (c, b) = pair_bits(table, x, y).map_or((0, 1), |bits| (bits, 0));
            cost[n][p + 1] = cost[n][p] + c;
            bad[n][p + 1] = bad[n][p] + b;
        }
    }
    let best_table = |from: usize, to: usize| -> (u32, usize) {
        if from >= to {
            return (0, 0);
        }
        usable.iter().filter(|&&n| bad[n][to] == bad[n][from]).map(|&n| (cost[n][to] - cost[n][from], n)).min().unwrap_or((u32::MAX / 4, 0))
    };

    let mut best: Option<(u32, usize, usize, [usize; 3])> = None;
    for r0 in 0..16usize {
        for r1 in 0..8usize {
            // Regions in pairs; they cannot pass the end of the big values.
            let a = (edge(long, r0 + 1) / 2).min(pairs);
            let b = (edge(long, r0 + r1 + 2) / 2).min(pairs).max(a);
            let ((c0, t0), (c1, t1), (c2, t2)) = (best_table(0, a), best_table(a, b), best_table(b, pairs));
            let total = c0.saturating_add(c1).saturating_add(c2);
            if best.is_none_or(|(bits, ..)| total < bits) {
                best = Some((total, r0, r1, [t0, t1, t2]));
            }
        }
    }
    let (big_bits, region0_count, region1_count, tables_chosen) = best?;
    let bits = big_bits.saturating_add(best_count1);
    Some(Granule { global_gain: 0, ix: *ix, big_values: pairs, count1_quads: quads, count1_table, region0_count, region1_count, tables: tables_chosen, bits })
}

fn write_granule(w: &mut BitWriter, g: &Granule, long: &[u16; 23]) {
    let t = tables();
    let a = (edge(long, g.region0_count + 1) / 2).min(g.big_values);
    let b = (edge(long, g.region0_count + g.region1_count + 2) / 2).min(g.big_values).max(a);
    for (region, (from, to)) in [(0, a), (a, b), (b, g.big_values)].into_iter().enumerate() {
        let table = t.pairs[g.tables[region]].as_ref().expect("a chosen table exists");
        for p in from..to {
            let (x, y) = (g.ix[2 * p], g.ix[2 * p + 1]);
            let (ax, ay) = (x.abs(), y.abs());
            if g.tables[region] == 0 {
                continue; // table 0: zeros, no bits
            }
            let (cx, cy) = if table.linbits > 0 { (ax.min(15), ay.min(15)) } else { (ax, ay) };
            w.write_bits(u32::from(table.code[cx as usize][cy as usize]), u32::from(table.len[cx as usize][cy as usize]));
            if table.linbits > 0 && ax >= 15 {
                w.write_bits((ax - 15) as u32, table.linbits);
            }
            if ax != 0 {
                w.write_bits(u32::from(x < 0), 1);
            }
            if table.linbits > 0 && ay >= 15 {
                w.write_bits((ay - 15) as u32, table.linbits);
            }
            if ay != 0 {
                w.write_bits(u32::from(y < 0), 1);
            }
        }
    }
    let q = &t.quads[g.count1_table];
    let start = 2 * g.big_values;
    for c in g.ix[start..start + 4 * g.count1_quads].chunks(4) {
        let m = (c[0].abs() << 3 | c[1].abs() << 2 | c[2].abs() << 1 | c[3].abs()) as usize;
        w.write_bits(u32::from(q.code[m]), u32::from(q.len[m]));
        for &v in c {
            if v != 0 {
                w.write_bits(u32::from(v < 0), 1);
            }
        }
    }
}

// ---------------------------------------------------------------- encoder

/// Per-channel filterbank state.
struct Channel {
    /// The last 512 input samples, newest first.
    x: [f64; 512],
    /// The previous granule's subband samples, `[subband][slot]`.
    previous: [[f64; 18]; 32],
}

/// Encodes a stream frame by frame.
pub struct Mp3Encoder {
    cfg: Mp3Config,
    sr_index: usize,
    bitrate_index: usize,
    long: [u16; 23],
    side_bytes: usize,
    channels: Vec<Channel>,
    pending: Vec<Vec<f32>>,
    /// Bands from this sample on are never coded.
    cutoff: usize,
    /// Fraction of a byte carried between frames, for padding.
    padding_acc: u32,
    frames: u64,
    pushed: u64,
    bytes: u64,
}

/// Input PCM (full scale 1.0) to the spectrum the quantizer works on.
const SCALE: f64 = 0.1122;

impl Mp3Encoder {
    pub fn new(cfg: Mp3Config) -> Result<Self, Mp3Error> {
        let sr_index = SAMPLE_RATES.iter().position(|&r| r == cfg.sample_rate).ok_or(Mp3Error::SampleRate(cfg.sample_rate))?;
        if !(1..=2).contains(&cfg.channels) {
            return Err(Mp3Error::Channels(cfg.channels));
        }
        let bitrate_index = BITRATES.iter().position(|&b| b == cfg.kbps).ok_or(Mp3Error::Bitrate(cfg.kbps))? + 1;
        let per_channel = f64::from(cfg.kbps) * 1000.0 / cfg.channels as f64;
        let cutoff_hz = (per_channel * 0.25).clamp(4_000.0, f64::from(cfg.sample_rate) / 2.0);
        let cutoff = ((cutoff_hz / (f64::from(cfg.sample_rate) / 2.0)) * GRANULE as f64) as usize;
        let _ = tables();
        Ok(Self {
            cfg,
            sr_index,
            bitrate_index,
            long: BAND_INFO[sr_index].long_idx,
            side_bytes: if cfg.channels == 1 { 17 } else { 32 },
            channels: (0..cfg.channels).map(|_| Channel { x: [0.0; 512], previous: [[0.0; 18]; 32] }).collect(),
            pending: vec![Vec::new(); cfg.channels],
            cutoff: cutoff.min(GRANULE),
            padding_acc: 0,
            frames: 0,
            pushed: 0,
            bytes: 0,
        })
    }

    pub fn config(&self) -> &Mp3Config {
        &self.cfg
    }

    /// Frames written so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Adds samples (one slice per channel, equal lengths) and returns the
    /// bytes of every frame that is now complete.
    pub fn push(&mut self, planar: &[&[f32]]) -> Result<Vec<u8>, Mp3Error> {
        if planar.len() != self.cfg.channels {
            return Err(Mp3Error::Input(format!("{} channels given, {} expected", planar.len(), self.cfg.channels)));
        }
        if planar.iter().any(|c| c.len() != planar[0].len()) {
            return Err(Mp3Error::Input("the channels differ in length".into()));
        }
        for (p, c) in self.pending.iter_mut().zip(planar) {
            p.extend_from_slice(c);
        }
        self.pushed += planar[0].len() as u64;
        let mut out = Vec::new();
        while self.pending[0].len() >= FRAME {
            let frame: Vec<Vec<f32>> = self.pending.iter_mut().map(|p| p.drain(..FRAME).collect()).collect();
            out.extend(self.encode_frame(&frame));
        }
        Ok(out)
    }

    /// Ends the stream: pads with silence so the last samples and the delay
    /// behind them come out.
    pub fn finish(&mut self) -> Vec<u8> {
        let total = self.pushed as usize + DELAY;
        let frames_needed = total.div_ceil(FRAME) as u64;
        let mut out = Vec::new();
        while self.frames < frames_needed || !self.pending[0].is_empty() {
            let frame: Vec<Vec<f32>> = self
                .pending
                .iter_mut()
                .map(|p| {
                    let take = p.len().min(FRAME);
                    let mut f: Vec<f32> = p.drain(..take).collect();
                    f.resize(FRAME, 0.0);
                    f
                })
                .collect();
            out.extend(self.encode_frame(&frame));
        }
        out
    }

    /// The `Info` frame that goes first in the file, once everything has been
    /// encoded: it carries the length, the encoder delay and the padding.
    pub fn info_frame(&self) -> Vec<u8> {
        build_info_frame(self, self.frames, self.bytes, self.pushed)
    }

    /// Filters 576 samples of one channel to subbands, transforms them, and
    /// reduces the aliasing: 576 coefficients, subband-major.
    fn analyze(&mut self, ch: usize, samples: &[f64]) -> [f64; GRANULE] {
        let t = tables();
        let state = &mut self.channels[ch];
        // 18 slots of 32 subband samples.
        let mut slots = [[0.0f64; 32]; 18];
        for (slot, out) in slots.iter_mut().enumerate() {
            state.x.copy_within(0..480, 32);
            for j in 0..32 {
                state.x[31 - j] = samples[slot * 32 + j];
            }
            let mut y = [0.0f64; 64];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = (0..8).map(|j| t.window[i + 64 * j] * state.x[i + 64 * j]).sum();
            }
            for (k, o) in out.iter_mut().enumerate() {
                *o = t.matrix[k].iter().zip(&y).map(|(m, v)| m * v).sum();
            }
        }
        let mut xr = [0.0f64; GRANULE];
        for sb in 0..32 {
            // The 36 inputs: the last granule's slots, then this one's, with
            // every odd slot of an odd subband negated to undo the filterbank's
            // frequency inversion.
            let mut z = [0.0f64; 36];
            for slot in 0..18 {
                let flip = if sb % 2 == 1 && slot % 2 == 1 { -1.0 } else { 1.0 };
                z[slot] = state.previous[sb][slot];
                z[18 + slot] = slots[slot][sb] * flip;
            }
            for slot in 0..18 {
                state.previous[sb][slot] = z[18 + slot];
            }
            for k in 0..18 {
                xr[sb * 18 + k] = (0..36).map(|i| t.mdct_window[i] * z[i] * t.mdct_cos[k][i]).sum();
            }
        }
        // Alias reduction between neighbouring subbands.
        const C: [f64; 8] = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037];
        for sb in 1..32 {
            for (i, c) in C.iter().enumerate() {
                let (cs, ca) = (1.0 / (1.0 + c * c).sqrt(), c / (1.0 + c * c).sqrt());
                let (bu, bd) = (xr[sb * 18 - 1 - i], xr[sb * 18 + i]);
                xr[sb * 18 - 1 - i] = bu * cs + bd * ca;
                xr[sb * 18 + i] = bd * cs - bu * ca;
            }
        }
        xr
    }

    fn encode_frame(&mut self, frame: &[Vec<f32>]) -> Vec<u8> {
        let nch = self.cfg.channels;
        // Spectra for both granules and channels.
        let mut xr = vec![vec![[0.0f64; GRANULE]; nch]; 2];
        for gr in 0..2 {
            for ch in 0..nch {
                let s: Vec<f64> = frame[ch][gr * GRANULE..(gr + 1) * GRANULE].iter().map(|&v| f64::from(v) * SCALE).collect();
                let mut spec = self.analyze(ch, &s);
                for v in &mut spec[self.cutoff..] {
                    *v = 0.0;
                }
                xr[gr][ch] = spec;
            }
        }

        // Mid/side for the frame, if it needs fewer bits.
        let mut ms = false;
        if nch == 2 {
            let (mut lr, mut mid_side) = (0.0, 0.0);
            for gr in 0..2 {
                for k in 0..self.cutoff {
                    let (l, r) = (xr[gr][0][k], xr[gr][1][k]);
                    lr += l.abs().powf(0.75) + r.abs().powf(0.75);
                    mid_side += ((l + r) * std::f64::consts::FRAC_1_SQRT_2).abs().powf(0.75) + ((l - r) * std::f64::consts::FRAC_1_SQRT_2).abs().powf(0.75);
                }
            }
            ms = mid_side < lr * 0.9;
            if ms {
                for gr_spec in &mut xr {
                    for k in 0..GRANULE {
                        let (l, r) = (gr_spec[0][k], gr_spec[1][k]);
                        gr_spec[0][k] = (l + r) * std::f64::consts::FRAC_1_SQRT_2;
                        gr_spec[1][k] = (l - r) * std::f64::consts::FRAC_1_SQRT_2;
                    }
                }
            }
        }

        // Frame size, with padding.
        let exact = 144_000 * self.cfg.kbps;
        let rate = self.cfg.sample_rate;
        self.padding_acc += exact % rate;
        let padding = u32::from(self.padding_acc >= rate);
        if padding == 1 {
            self.padding_acc -= rate;
        }
        let size = (exact / rate + padding) as usize;
        let capacity = ((size - 4 - self.side_bytes) * 8) as u32;

        // Each granule-channel takes an equal share of what is left.
        let mut granules: Vec<Granule> = Vec::with_capacity(2 * nch);
        let mut left = capacity;
        let mut remaining = (2 * nch) as u32;
        for gr in 0..2 {
            for ch in 0..nch {
                let share = (left / remaining).min(4095);
                let g = self.fit(&xr[gr][ch], share);
                left -= g.bits.min(left);
                remaining -= 1;
                granules.push(g);
            }
        }

        // Write: header, side info, main data.
        let mut w = BitWriter::with_capacity(size);
        w.write_bits(0x7FF, 11); // sync
        w.write_bits(0b11, 2); // MPEG-1
        w.write_bits(0b01, 2); // Layer III
        w.write_bits(1, 1); // no CRC
        w.write_bits(self.bitrate_index as u32, 4);
        w.write_bits(self.sr_index as u32, 2);
        w.write_bits(padding, 1);
        w.write_bits(0, 1); // private
        w.write_bits(if nch == 1 { 3 } else { 1 }, 2); // mono / joint stereo
        w.write_bits(if ms { 2 } else { 0 }, 2);
        w.write_bits(0, 1); // copyright
        w.write_bits(1, 1); // original
        w.write_bits(0, 2); // emphasis
        w.write_bits(0, 9); // main_data_begin: no reservoir
        w.write_bits(0, if nch == 1 { 5 } else { 3 });
        for _ in 0..nch {
            w.write_bits(0, 4); // scfsi
        }
        for g in &granules {
            w.write_bits(g.bits, 12);
            w.write_bits(g.big_values as u32, 9);
            w.write_bits(g.global_gain, 8);
            w.write_bits(0, 4); // scalefac_compress: no scalefactors
            w.write_bits(0, 1); // window switching off
            for t in g.tables {
                w.write_bits(t as u32, 5);
            }
            w.write_bits(g.region0_count as u32, 4);
            w.write_bits(g.region1_count as u32, 3);
            w.write_bits(0, 1); // preflag
            w.write_bits(0, 1); // scalefac_scale
            w.write_bits(g.count1_table as u32, 1);
        }
        for g in &granules {
            write_granule(&mut w, g, &self.long);
        }
        // Stuffing: zeros to the end of the frame.
        let mut bytes = {
            while w.bit_len() % 8 != 0 {
                w.write_bits(0, 1);
            }
            w.finish()
        };
        bytes.resize(size, 0);
        self.frames += 1;
        self.bytes += bytes.len() as u64;
        bytes
    }

    /// Quantizes one granule at the finest step that fits `budget` bits.
    fn fit(&self, spec: &[f64; GRANULE], budget: u32) -> Granule {
        let pow34: Vec<f64> = spec.iter().map(|x| x.abs().powf(0.75)).collect();
        let try_gain = |gg: u32| -> Option<Granule> {
            let inv = (-0.75 * (f64::from(gg) - 210.0) / 4.0).exp2();
            let mut ix = [0i32; GRANULE];
            for k in 0..GRANULE {
                let v = (pow34[k] * inv + 0.4054).floor();
                if v > 8206.0 {
                    return None;
                }
                ix[k] = v as i32;
            }
            let mut g = code_granule(&ix, &self.long)?;
            for (k, v) in g.ix.iter_mut().enumerate() {
                if spec[k] < 0.0 {
                    *v = -*v;
                }
            }
            g.global_gain = gg;
            Some(g)
        };
        let (mut lo, mut hi) = (0u32, 255u32);
        let mut best: Option<Granule> = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            match try_gain(mid) {
                Some(g) if g.bits <= budget => {
                    best = Some(g);
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
                _ => lo = mid + 1,
            }
        }
        best.unwrap_or_else(Granule::silent)
    }
}

// ----------------------------------------------------------- info frame

fn crc16(data: &[u8]) -> u16 {
    // CRC-16, polynomial 0x8005 reflected (what LAME computes over its tag).
    let mut crc = 0u16;
    for &b in data {
        crc ^= u16::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xA001 } else { crc >> 1 };
        }
    }
    crc
}

/// A valid silent frame carrying the Xing `Info` tag and a LAME-style
/// extension, which is where players read the encoder delay and padding.
///
/// Players decide whether the extension is there by its first four bytes, so
/// they say `LAME`; the next five name the encoder.
fn build_info_frame(enc: &Mp3Encoder, frames: u64, bytes: u64, samples: u64) -> Vec<u8> {
    let cfg = enc.cfg;
    let size = (144_000 * cfg.kbps / cfg.sample_rate) as usize;
    let mut f = vec![0u8; size];
    f[0] = 0xFF;
    f[1] = 0xFB;
    f[2] = ((enc.bitrate_index as u8) << 4) | ((enc.sr_index as u8) << 2);
    f[3] = (if cfg.channels == 1 { 3 } else { 1 }) << 6;
    let at = 4 + enc.side_bytes;
    let file_bytes = (bytes + size as u64) as u32;
    f[at..at + 4].copy_from_slice(b"Info");
    f[at + 4..at + 8].copy_from_slice(&3u32.to_be_bytes()); // frames and bytes present
    f[at + 8..at + 12].copy_from_slice(&(frames as u32).to_be_bytes());
    f[at + 12..at + 16].copy_from_slice(&file_bytes.to_be_bytes());
    // The extension follows the fields the flags announce.
    let l = at + 16;
    f[l..l + 9].copy_from_slice(b"LAMEthrs ");
    f[l + 9] = 0x01; // tag revision 0, constant bitrate
    f[l + 10] = ((f64::from(cfg.kbps) * 1000.0 / cfg.channels as f64 * 0.25).clamp(4_000.0, 20_000.0) / 100.0) as u8; // low-pass, 100 Hz units
    f[l + 20] = cfg.kbps.min(255) as u8;
    // Delay and padding, 12 bits each: the delay a decoder removes besides
    // its own 529 samples, and the silence at the end.
    let delay = DELAY - 529;
    let padding = (frames * FRAME as u64).saturating_sub(delay as u64 + samples) as u32;
    f[l + 21] = (delay >> 4) as u8;
    f[l + 22] = (((delay & 15) << 4) as u8) | ((padding >> 8) as u8 & 15);
    f[l + 23] = padding as u8;
    f[l + 28..l + 32].copy_from_slice(&file_bytes.to_be_bytes()); // music length
    let crc = crc16(&f[..l + 34]);
    f[l + 34..l + 36].copy_from_slice(&crc.to_be_bytes());
    f
}

/// Encodes whole channels to an `.mp3` file, `Info` frame first.
pub fn encode_mp3(planar: &[Vec<f32>], cfg: &Mp3Config) -> Result<Vec<u8>, Mp3Error> {
    let mut enc = Mp3Encoder::new(*cfg)?;
    let views: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let mut body = enc.push(&views)?;
    body.extend(enc.finish());
    let mut out = enc.info_frame();
    out.extend(body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_is_a_prefix_code_that_covers_its_values() {
        let t = tables();
        for (n, table) in t.pairs.iter().enumerate() {
            let Some(table) = table else { continue };
            if n == 0 {
                continue;
            }
            // Kraft: the code lengths sum to at most 1, and nothing is a
            // prefix of anything else.
            let mut codes = Vec::new();
            for x in 0..16 {
                for y in 0..16 {
                    if table.len[x][y] > 0 {
                        codes.push((u32::from(table.code[x][y]), u32::from(table.len[x][y])));
                    }
                }
            }
            let kraft: f64 = codes.iter().map(|&(_, l)| 0.5f64.powi(l as i32)).sum();
            assert!(kraft <= 1.0 + 1e-9, "table {n}: Kraft sum {kraft}");
            for (i, &(a, la)) in codes.iter().enumerate() {
                for &(b, lb) in &codes[i + 1..] {
                    let l = la.min(lb);
                    assert_ne!(a >> (la - l), b >> (lb - l), "table {n}: one code starts another");
                }
            }
            let xmax = if table.linbits > 0 { 15 } else { table.max as usize };
            for x in 0..=xmax {
                for y in 0..=xmax {
                    assert!(table.len[x][y] > 0, "table {n} lacks ({x},{y})");
                }
            }
        }
        for (n, q) in t.quads.iter().enumerate() {
            assert!(q.len.iter().all(|&l| l > 0), "count-1 table {n} covers all 16 quads");
            let kraft: f64 = q.len.iter().map(|&l| 0.5f64.powi(i32::from(l))).sum();
            assert!((kraft - 1.0).abs() < 1e-9, "count-1 table {n}: {kraft}");
        }
        // Table 1's well-known codes.
        let t1 = t.pairs[1].as_ref().unwrap();
        assert_eq!((t1.len[0][0], t1.code[0][0]), (1, 1));
        assert_eq!((t1.len[1][1], t1.code[1][1]), (3, 0));
    }

    #[test]
    fn the_window_is_the_prototype_filter() {
        let w = &tables().window;
        assert!((w[256] - 75_038.0 / 65_536.0 / 32.0).abs() < 1e-12);
        // Symmetric in magnitude about 256, with the sign alternating by 64.
        for i in 1..256 {
            assert!((w[i].abs() - w[512 - i].abs()).abs() < 1e-12, "{i}");
        }
        assert!(w[10] < 0.0 && w[100] < 0.0 || w[100] > 0.0, "signs are set");
    }

    #[test]
    fn bad_settings_are_refused() {
        assert_eq!(Mp3Encoder::new(Mp3Config::new(22_050, 2, 128)).err(), Some(Mp3Error::SampleRate(22_050)));
        assert_eq!(Mp3Encoder::new(Mp3Config::new(44_100, 3, 128)).err(), Some(Mp3Error::Channels(3)));
        assert_eq!(Mp3Encoder::new(Mp3Config::new(44_100, 2, 100)).err(), Some(Mp3Error::Bitrate(100)));
        let mut enc = Mp3Encoder::new(Mp3Config::new(44_100, 2, 128)).unwrap();
        assert!(matches!(enc.push(&[&[0.0; 4][..]]), Err(Mp3Error::Input(_))));
    }
}
