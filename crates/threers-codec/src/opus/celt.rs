//! CELT, as ffmpeg n7.1 decodes it (`opus/dec_celt.c`, `celt.c`, `pvq.c`).
//!
//! Every float expression keeps its C types: `M_PI`/`M_SQRT1_2`/`1.0`
//! literals promote to double where the C does, the postfilter transition
//! runs in double, and the C is not FMA-contracted. The aarch64 reference
//! build's NEON kernels (`vector_fmul_window`, `vector_fmac_scalar`,
//! `opus_postfilter`, `opus_deemphasis`) fuse their multiply-adds as the
//! assembly does.
//!
//! The band quantiser's pointer arithmetic is kept as offsets into one
//! arena (`Mem`) holding both channels' coefficients, the two folding
//! norms and the scratch band.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::mdct15::Mdct15;
use super::rc::{ilog, Rc};
use super::tab::*;

pub const MAX_BANDS: usize = 21;

/// Path counters for coverage checks (`OPUS_STATS=1` in opus_dump).
pub static STATS: [std::sync::atomic::AtomicUsize; 12] = [const { std::sync::atomic::AtomicUsize::new(0) }; 12];
pub const STAT_NAMES: [&str; 12] = ["transient", "anticollapse", "postfilter", "intensity", "dual_stereo", "silence", "tf_select", "split_stereo_n2", "fold_noise", "inv_phase", "redundancy_pre", "redundancy_post"];
#[inline]
pub(crate) fn stat(i: usize) {
    STATS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
const SHORT_BLOCKSIZE: usize = 120;
const OVERLAP: usize = 120;
const MAX_FRAME_SIZE: usize = 960;
const VECTORS: i32 = 11;
const ALLOC_STEPS: i32 = 6;
const FINE_OFFSET: i32 = 21;
const MAX_FINE_BITS: i32 = 8;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const POSTFILTER_MINPERIOD: i32 = 15;
const ENERGY_SILENCE: f32 = -28.0;

const SPREAD_NONE: i32 = 0;
const SPREAD_NORMAL: i32 = 2;
const SPREAD_AGGRESSIVE: i32 = 3;

const M_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const M_SQRT2: f64 = std::f64::consts::SQRT_2;
const M_PI: f64 = std::f64::consts::PI;

// Arena layout (floats).
const X0: usize = 0;
const X1: usize = MAX_FRAME_SIZE;
const NORM1: usize = 2 * MAX_FRAME_SIZE;
const NORM2: usize = NORM1 + 8 * 100;
const SCRATCH: usize = NORM2 + 8 * 100;
const MEM_LEN: usize = SCRATCH + 8 * 22;

type Mem = Vec<f32>;

fn alpha_coef(size: usize) -> f32 {
    [29440.0f32 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 16384.0 / 32768.0][size]
}
fn beta_coef(size: usize) -> f32 {
    [1.0f32 - (30147.0 / 32768.0), 1.0 - (22282.0 / 32768.0), 1.0 - (12124.0 / 32768.0), 1.0 - (6554.0 / 32768.0)][size]
}

/// `ff_opus_deemph_weights`: products of `CELT_EMPH_COEFF` in double.
fn deemph_weights() -> [f32; 16] {
    let c = 0.8500061035f64;
    let p = [c, c * c, c * c * c, c * c * c * c];
    let z = 0.0f64;
    [p[0], p[1], p[2], p[3], z, p[0], p[1], p[2], z, z, p[0], p[1], z, z, z, p[0]].map(|v| v as f32)
}

#[derive(Clone)]
pub struct Block {
    energy: [f32; MAX_BANDS],
    prev_energy: [[f32; MAX_BANDS]; 2],
    collapse_masks: [u8; MAX_BANDS],
    buf: Vec<f32>,
    pf_period_new: i32,
    pf_gains_new: [f32; 3],
    pf_period: i32,
    pf_gains: [f32; 3],
    pf_period_old: i32,
    pf_gains_old: [f32; 3],
    emph_coeff: f32,
}

impl Block {
    fn new() -> Self {
        Block {
            energy: [0.0; MAX_BANDS],
            prev_energy: [[0.0; MAX_BANDS]; 2],
            collapse_masks: [0; MAX_BANDS],
            buf: vec![0.0; 2048],
            pf_period_new: 0,
            pf_gains_new: [0.0; 3],
            pf_period: 0,
            pf_gains: [0.0; 3],
            pf_period_old: 0,
            pf_gains_old: [0.0; 3],
            emph_coeff: 0.0,
        }
    }
}

pub struct Celt {
    tx: [Mdct15; 4],
    block: [Block; 2],
    mem: Mem,
    tmp: Vec<f32>,
    hadamard_tmp: [f32; 256],
    qcoeff: [i32; 256],
    channels: usize,
    output_channels: usize,
    apply_phase_inv: bool,

    size: usize,
    start_band: usize,
    end_band: usize,
    coded_bands: usize,
    transient: bool,
    blocks: usize,
    blocksize: usize,
    silence: bool,
    anticollapse_needed: i32,
    anticollapse: u32,
    intensity_stereo: usize,
    dual_stereo: bool,
    flushed: bool,
    seed: u32,
    spread: i32,
    alloc_trim: i32,

    framebits: i32,
    remaining: i32,
    remaining2: i32,
    caps: [i32; MAX_BANDS],
    fine_bits: [i32; MAX_BANDS],
    fine_priority: [i32; MAX_BANDS],
    pulses: [i32; MAX_BANDS],
    tf_change: [i32; MAX_BANDS],
    weights: [f32; 16],
}

impl Celt {
    /// `ff_celt_init`.
    pub fn new(output_channels: usize, apply_phase_inv: bool) -> Self {
        let scale = -1.0f32 / 32768.0;
        let mut c = Celt {
            tx: [0, 1, 2, 3].map(|i| Mdct15::new(15 << (i + 3), scale)),
            block: [Block::new(), Block::new()],
            mem: vec![0.0; MEM_LEN],
            tmp: Vec::new(),
            hadamard_tmp: [0.0; 256],
            qcoeff: [0; 256],
            channels: 0,
            output_channels,
            apply_phase_inv,
            size: 0,
            start_band: 0,
            end_band: 0,
            coded_bands: 0,
            transient: false,
            blocks: 1,
            blocksize: 0,
            silence: false,
            anticollapse_needed: 0,
            anticollapse: 0,
            intensity_stereo: 0,
            dual_stereo: false,
            flushed: false,
            seed: 0,
            spread: 0,
            alloc_trim: 0,
            framebits: 0,
            remaining: 0,
            remaining2: 0,
            caps: [0; MAX_BANDS],
            fine_bits: [0; MAX_BANDS],
            fine_priority: [0; MAX_BANDS],
            pulses: [0; MAX_BANDS],
            tf_change: [0; MAX_BANDS],
            weights: deemph_weights(),
        };
        c.flush();
        c
    }

    /// `ff_celt_flush`.
    pub fn flush(&mut self) {
        if self.flushed {
            return;
        }
        for b in self.block.iter_mut() {
            b.prev_energy = [[ENERGY_SILENCE; MAX_BANDS]; 2];
            b.energy = [0.0; MAX_BANDS];
            b.buf.iter_mut().for_each(|v| *v = 0.0);
            b.pf_gains = [0.0; 3];
            b.pf_gains_old = [0.0; 3];
            b.pf_gains_new = [0.0; 3];
            b.emph_coeff = 0.0 / self.weights[0];
        }
        self.seed = 0;
        self.flushed = true;
    }

    #[inline]
    fn rng(&mut self) -> u32 {
        self.seed = 1664525u32.wrapping_mul(self.seed).wrapping_add(1013904223);
        self.seed
    }

    fn decode_coarse_energy(&mut self, rc: &mut Rc) {
        let mut prev = [0f32; 2];
        let mut alpha = alpha_coef(self.size);
        let mut beta = beta_coef(self.size);
        let mut model = &CELT_COARSE_ENERGY_DIST[self.size][0];
        if rc.tell() as i32 + 3 <= self.framebits && rc.dec_log(3) != 0 {
            alpha = 0.0;
            beta = 1.0 - (4915.0f32 / 32768.0);
            model = &CELT_COARSE_ENERGY_DIST[self.size][1];
        }
        for i in 0..MAX_BANDS {
            for j in 0..self.channels {
                if i < self.start_band || i >= self.end_band {
                    self.block[j].energy[i] = 0.0;
                    continue;
                }
                let available = self.framebits - rc.tell() as i32;
                let value: f32 = if available >= 15 {
                    let k = i.min(20) << 1;
                    rc.dec_laplace((model[k] as u32) << 7, (model[k + 1] as i32) << 6) as f32
                } else if available >= 2 {
                    let x = rc.dec_cdf(&CELT_MODEL_ENERGY_SMALL) as i32;
                    ((x >> 1) ^ -(x & 1)) as f32
                } else if available >= 1 {
                    -(rc.dec_log(1) as f32)
                } else {
                    -1.0
                };
                let e = &mut self.block[j].energy[i];
                *e = (-9.0f32).max(*e) * alpha + prev[j] + value;
                prev[j] += beta * value;
            }
        }
    }

    fn decode_fine_energy(&mut self, rc: &mut Rc) {
        for i in self.start_band..self.end_band {
            let fb = self.fine_bits[i];
            if fb == 0 {
                continue;
            }
            for j in 0..self.channels {
                let q2 = rc.get_raw(fb as u32);
                let offset = (q2 as f32 + 0.5f32) * (1i32 << (14 - fb)) as f32 / 16384.0f32 - 0.5f32;
                self.block[j].energy[i] += offset;
            }
        }
    }

    fn decode_final_energy(&mut self, rc: &mut Rc) {
        let mut bits_left = self.framebits - rc.tell() as i32;
        for priority in 0..2 {
            let mut i = self.start_band;
            while i < self.end_band && bits_left >= self.channels as i32 {
                if self.fine_priority[i] != priority || self.fine_bits[i] >= MAX_FINE_BITS {
                    i += 1;
                    continue;
                }
                for j in 0..self.channels {
                    let q2 = rc.get_raw(1);
                    let offset = (q2 as f32 - 0.5f32) * (1i32 << (14 - self.fine_bits[i] - 1)) as f32 / 16384.0f32;
                    self.block[j].energy[i] += offset;
                    bits_left -= 1;
                }
                i += 1;
            }
        }
    }

    fn decode_tf_changes(&mut self, rc: &mut Rc) {
        let (mut diff, mut tf_select, mut tf_changed) = (0i32, 0usize, 0i32);
        let mut bits = if self.transient { 2 } else { 4 };
        let mut consumed = rc.tell() as i32;
        let tf_select_bit = (self.size != 0 && consumed + bits < self.framebits) as i32;
        for i in self.start_band..self.end_band {
            if consumed + bits + tf_select_bit <= self.framebits {
                diff ^= rc.dec_log(bits as u32) as i32;
                consumed = rc.tell() as i32;
                tf_changed |= diff;
            }
            self.tf_change[i] = diff;
            bits = if self.transient { 4 } else { 5 };
        }
        let t = &CELT_TF_SELECT[self.size][self.transient as usize];
        if tf_select_bit != 0 && t[0][tf_changed as usize] != t[1][tf_changed as usize] {
            tf_select = rc.dec_log(1) as usize;
            stat(6);
        }
        for i in self.start_band..self.end_band {
            self.tf_change[i] = t[tf_select][self.tf_change[i] as usize] as i32;
        }
    }

    fn denormalize(&mut self, ch: usize) {
        let base = if ch == 0 { X0 } else { X1 };
        for i in self.start_band..self.end_band {
            let dst = base + ((CELT_FREQ_BANDS[i] as usize) << self.size);
            let log_norm = self.block[ch].energy[i] + CELT_MEAN_ENERGY[i];
            let norm = log_norm.min(32.0).exp2();
            for j in 0..(CELT_FREQ_RANGE[i] as usize) << self.size {
                self.mem[dst + j] *= norm;
            }
        }
    }

    fn parse_postfilter(&mut self, rc: &mut Rc, consumed: i32) -> i32 {
        let mut consumed = consumed;
        self.block[0].pf_gains_new = [0.0; 3];
        self.block[1].pf_gains_new = [0.0; 3];
        if self.start_band == 0 && consumed + 16 <= self.framebits {
            if rc.dec_log(1) != 0 {
                stat(2);
                let octave = rc.dec_uint(6);
                let period = (16i32 << octave) + rc.get_raw(4 + octave) as i32 - 1;
                let gain = 0.09375f32 * (rc.get_raw(3) + 1) as f32;
                let tapset = if rc.tell() as i32 + 2 <= self.framebits { rc.dec_cdf(&CELT_MODEL_TAPSET) as usize } else { 0 };
                for b in self.block.iter_mut() {
                    b.pf_period_new = period.max(POSTFILTER_MINPERIOD);
                    for k in 0..3 {
                        b.pf_gains_new[k] = gain * CELT_POSTFILTER_TAPS[tapset][k];
                    }
                }
            }
            consumed = rc.tell() as i32;
        }
        consumed
    }

    fn process_anticollapse(&mut self, ch: usize) {
        let base = if ch == 0 { X0 } else { X1 };
        for i in self.start_band..self.end_band {
            let mut renormalize = false;
            let n = (CELT_FREQ_RANGE[i] as i32) << self.size;
            let depth = (1 + self.pulses[i]) / n;
            let thresh = ((-1.0f64 - (0.125f32 * depth as f32) as f64) as f32).exp2();
            let sqrt_1 = 1.0f32 / (n as f32).sqrt();
            let xptr = base + ((CELT_FREQ_BANDS[i] as usize) << self.size);
            let mut prev = [self.block[ch].prev_energy[0][i], self.block[ch].prev_energy[1][i]];
            if self.channels == 1 {
                prev[0] = prev[0].max(self.block[1].prev_energy[0][i]);
                prev[1] = prev[1].max(self.block[1].prev_energy[1][i]);
            }
            let ediff = (self.block[ch].energy[i] - prev[0].min(prev[1])).max(0.0);
            let mut r = (1.0f32 - ediff).exp2();
            if self.size == 3 {
                r = (r as f64 * M_SQRT2) as f32;
            }
            r = thresh.min(r) * sqrt_1;
            for k in 0..1usize << self.size {
                if self.block[ch].collapse_masks[i] & (1 << k) == 0 {
                    for j in 0..CELT_FREQ_RANGE[i] as usize {
                        let v = if self.rng() & 0x8000 != 0 { r } else { -r };
                        self.mem[xptr + (j << self.size) + k] = v;
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalize_vector(&mut self.mem[xptr..xptr + n as usize], 1.0);
            }
        }
    }

    /// `ff_celt_decode_frame`. `out[c]` receives `frame_size` samples for
    /// each output channel.
    pub fn decode_frame(
        &mut self,
        rc: &mut Rc,
        out: &mut [&mut [f32]],
        channels: usize,
        frame_size: usize,
        start_band: usize,
        end_band: usize,
    ) -> Result<(), String> {
        if channels != 1 && channels != 2 {
            return Err(format!("Invalid number of coded channels: {channels}"));
        }
        if start_band > end_band || end_band > MAX_BANDS {
            return Err(format!("Invalid start/end band: {start_band} {end_band}"));
        }
        self.silence = false;
        self.transient = false;
        self.anticollapse = 0;
        self.flushed = false;
        self.channels = channels;
        self.start_band = start_band;
        self.end_band = end_band;
        self.framebits = rc.rb_bytes() as i32 * 8;
        let size = (frame_size / SHORT_BLOCKSIZE).max(1).ilog2() as usize;
        if size > 3 || frame_size != SHORT_BLOCKSIZE << size {
            return Err(format!("Invalid CELT frame size: {frame_size}"));
        }
        self.size = size;
        if self.output_channels == 0 {
            self.output_channels = channels;
        }
        for i in 0..self.channels {
            let base = if i == 0 { X0 } else { X1 };
            self.mem[base..base + MAX_FRAME_SIZE].iter_mut().for_each(|v| *v = 0.0);
            self.block[i].collapse_masks = [0; MAX_BANDS];
        }
        let mut consumed = rc.tell() as i32;
        if consumed >= self.framebits {
            self.silence = true;
        } else if consumed == 1 {
            self.silence = rc.dec_log(15) != 0;
        }
        if self.silence {
            stat(5);
            consumed = self.framebits;
            rc.total_bits = rc.total_bits.wrapping_add((self.framebits as u32).wrapping_sub(rc.tell()));
        }
        consumed = self.parse_postfilter(rc, consumed);
        if self.size != 0 && consumed + 3 <= self.framebits {
            self.transient = rc.dec_log(3) != 0;
        }
        if self.transient { stat(0); }
        self.blocks = if self.transient { 1 << self.size } else { 1 };
        self.blocksize = frame_size / self.blocks;
        let txi = if self.transient { 0 } else { self.size };

        if channels == 1 {
            for i in 0..MAX_BANDS {
                self.block[0].energy[i] = self.block[0].energy[i].max(self.block[1].energy[i]);
            }
        }

        self.decode_coarse_energy(rc);
        self.decode_tf_changes(rc);
        self.bitalloc(rc);
        self.decode_fine_energy(rc);
        self.quant_bands(rc);
        if self.anticollapse_needed != 0 {
            self.anticollapse = rc.get_raw(1);
        }
        self.decode_final_energy(rc);

        for i in 0..self.channels {
            if self.anticollapse != 0 {
                stat(1);
                self.process_anticollapse(i);
            }
            self.denormalize(i);
        }

        let mut downmix = false;
        if self.output_channels < self.channels {
            // vector_fmac_scalar(coeffs0, coeffs1, 1.0, FFALIGN(frame_size, 16)), fused
            for k in 0..(frame_size + 15) & !15 {
                self.mem[X0 + k] = self.mem[X1 + k].mul_add(1.0, self.mem[X0 + k]);
            }
            downmix = true;
        } else if self.output_channels > self.channels {
            self.mem.copy_within(X0..X0 + frame_size, X1);
        }

        if self.silence {
            for b in self.block.iter_mut() {
                b.energy = [ENERGY_SILENCE; MAX_BANDS];
            }
            self.mem[X0..X0 + 2 * MAX_FRAME_SIZE].iter_mut().for_each(|v| *v = 0.0);
        }

        let window = &CELT_WINDOW_PADDED[8..8 + OVERLAP];
        for i in 0..self.output_channels {
            let base = if i == 0 { X0 } else { X1 };
            for j in 0..self.blocks {
                let dst = 1024 + j * self.blocksize;
                let (tx, mem, tmp, block) = (&self.tx[txi], &self.mem, &mut self.tmp, &mut self.block[i]);
                tx.run(
                    &mut block.buf[dst + OVERLAP / 2..dst + OVERLAP / 2 + self.blocksize],
                    &mem[base + j..],
                    self.blocks,
                    tmp,
                );
                fmul_window_inplace(&mut block.buf[dst..dst + OVERLAP], window, OVERLAP / 2);
            }
            let block = &mut self.block[i];
            if downmix {
                for v in &mut block.buf[1024..1024 + frame_size] {
                    *v *= 0.5f32;
                }
            }
            postfilter(block, self.blocksize * self.blocks);
            block.emph_coeff = deemphasis(&mut out[i][..frame_size], &block.buf[1024 - frame_size..1024], block.emph_coeff, &self.weights);
        }

        if channels == 1 {
            self.block[1].energy = self.block[0].energy;
        }
        for b in self.block.iter_mut() {
            if !self.transient {
                b.prev_energy[1] = b.prev_energy[0];
                b.prev_energy[0] = b.energy;
            } else {
                for j in 0..MAX_BANDS {
                    b.prev_energy[0][j] = b.prev_energy[0][j].min(b.energy[j]);
                }
            }
            for j in (0..self.start_band).chain(self.end_band..MAX_BANDS) {
                b.prev_energy[0][j] = ENERGY_SILENCE;
                b.energy[j] = 0.0;
            }
        }
        self.seed = rc.range;
        Ok(())
    }

    // ---------------------------------------------------------------- celt.c

    fn normc(&self, bits: i32) -> i32 {
        (bits << (self.channels - 1) << self.size) >> 2
    }

    /// `ff_celt_bitalloc(f, rc, 0)`.
    fn bitalloc(&mut self, rc: &mut Rc) {
        let ch = self.channels as i32;
        let size = self.size as i32;
        let (start, end) = (self.start_band, self.end_band);
        let mut skip_startband = start;
        let mut skip_bit = 0;
        let mut intensitystereo_bit = 0;
        let mut dualstereo_bit = 0;
        let mut dynalloc = 6;
        let mut extrabits = 0;
        let mut boost = [0i32; MAX_BANDS];
        let mut trim_offset = [0i32; MAX_BANDS];
        let mut threshold = [0i32; MAX_BANDS];
        let mut bits1 = [0i32; MAX_BANDS];
        let mut bits2 = [0i32; MAX_BANDS];
        let fr = |i: usize| CELT_FREQ_RANGE[i] as i32;
        let fb = |i: usize| CELT_FREQ_BANDS[i] as i32;

        if rc.tell() as i32 + 4 <= self.framebits {
            self.spread = rc.dec_cdf(&CELT_MODEL_SPREAD) as i32;
        } else {
            self.spread = SPREAD_NORMAL;
        }
        for i in 0..MAX_BANDS {
            self.caps[i] = self.normc((CELT_STATIC_CAPS[self.size][self.channels - 1][i] as i32 + 64) * fr(i));
        }
        let mut tbits_8ths = self.framebits << 3;
        for i in start..end {
            let mut quanta = fr(i) << (ch - 1) << size;
            let mut b_dynalloc = dynalloc;
            quanta = (quanta << 3).min((6 << 3).max(quanta));
            while (rc.tell_frac() as i32) + (b_dynalloc << 3) < tbits_8ths && boost[i] < self.caps[i] {
                let is_boost = rc.dec_log(b_dynalloc as u32);
                if is_boost == 0 {
                    break;
                }
                boost[i] += quanta;
                tbits_8ths -= quanta;
                b_dynalloc = 1;
            }
            if boost[i] != 0 {
                dynalloc = (dynalloc - 1).max(2);
            }
        }
        self.alloc_trim = 5;
        if rc.tell_frac() as i32 + (6 << 3) <= tbits_8ths {
            self.alloc_trim = rc.dec_cdf(&CELT_MODEL_ALLOC_TRIM) as i32;
        }
        tbits_8ths = (self.framebits << 3) - rc.tell_frac() as i32 - 1;
        self.anticollapse_needed = 0;
        if self.transient && size >= 2 && tbits_8ths >= ((size + 2) << 3) {
            self.anticollapse_needed = 1 << 3;
        }
        tbits_8ths -= self.anticollapse_needed;
        if tbits_8ths >= 1 << 3 {
            skip_bit = 1 << 3;
        }
        tbits_8ths -= skip_bit;
        if ch == 2 {
            intensitystereo_bit = CELT_LOG2_FRAC[end - start] as i32;
            if intensitystereo_bit <= tbits_8ths {
                tbits_8ths -= intensitystereo_bit;
                if tbits_8ths >= 1 << 3 {
                    dualstereo_bit = 1 << 3;
                    tbits_8ths -= 1 << 3;
                }
            } else {
                intensitystereo_bit = 0;
            }
        }
        for i in start..end {
            let trim = self.alloc_trim - 5 - size;
            let band = fr(i) * (end as i32 - i as i32 - 1);
            let duration = size + 3;
            let scale = duration + ch - 1;
            threshold[i] = ((3 * fr(i)) << duration >> 4).max(ch << 3);
            trim_offset[i] = (trim * (band << scale)) >> 6;
            if fr(i) << size == 1 {
                trim_offset[i] -= ch << 3;
            }
        }
        let mut low = 1;
        let mut high = VECTORS - 1;
        while low <= high {
            let center = (low + high) >> 1;
            let mut done = false;
            let mut total = 0;
            for i in (start..end).rev() {
                let mut bandbits = self.normc(fr(i) * CELT_STATIC_ALLOC[center as usize][i] as i32);
                if bandbits != 0 {
                    bandbits = (bandbits + trim_offset[i]).max(0);
                }
                bandbits += boost[i];
                if bandbits >= threshold[i] || done {
                    done = true;
                    total += bandbits.min(self.caps[i]);
                } else if bandbits >= ch << 3 {
                    total += ch << 3;
                }
            }
            if total > tbits_8ths {
                high = center - 1;
            } else {
                low = center + 1;
            }
        }
        high = low;
        low -= 1;
        for i in start..end {
            bits1[i] = self.normc(fr(i) * CELT_STATIC_ALLOC[low as usize][i] as i32);
            bits2[i] = if high >= VECTORS { self.caps[i] } else { self.normc(fr(i) * CELT_STATIC_ALLOC[high as usize][i] as i32) };
            if bits1[i] != 0 {
                bits1[i] = (bits1[i] + trim_offset[i]).max(0);
            }
            if bits2[i] != 0 {
                bits2[i] = (bits2[i] + trim_offset[i]).max(0);
            }
            if low != 0 {
                bits1[i] += boost[i];
            }
            bits2[i] += boost[i];
            if boost[i] != 0 {
                skip_startband = i;
            }
            bits2[i] = (bits2[i] - bits1[i]).max(0);
        }
        let mut low = 0;
        let mut high = 1 << ALLOC_STEPS;
        for _ in 0..ALLOC_STEPS {
            let center = (low + high) >> 1;
            let mut done = false;
            let mut total = 0;
            for j in (start..end).rev() {
                let bandbits = bits1[j] + ((center * bits2[j]) >> ALLOC_STEPS);
                if bandbits >= threshold[j] || done {
                    done = true;
                    total += bandbits.min(self.caps[j]);
                } else if bandbits >= ch << 3 {
                    total += ch << 3;
                }
            }
            if total > tbits_8ths {
                high = center;
            } else {
                low = center;
            }
        }
        let mut done = false;
        let mut total = 0;
        for i in (start..end).rev() {
            let mut bandbits = bits1[i] + ((low * bits2[i]) >> ALLOC_STEPS);
            if bandbits >= threshold[i] || done {
                done = true;
            } else {
                bandbits = if bandbits >= ch << 3 { ch << 3 } else { 0 };
            }
            bandbits = bandbits.min(self.caps[i]);
            self.pulses[i] = bandbits;
            total += bandbits;
        }
        self.coded_bands = end;
        loop {
            let j = self.coded_bands - 1;
            if j == skip_startband {
                tbits_8ths += skip_bit;
                break;
            }
            let mut remaining = tbits_8ths - total;
            let bandbits = remaining / (fb(j + 1) - fb(start));
            remaining -= bandbits * (fb(j + 1) - fb(start));
            let mut allocation = self.pulses[j] + bandbits * fr(j);
            allocation += (remaining - (fb(j) - fb(start))).max(0);
            if allocation >= threshold[j].max((ch + 1) << 3) {
                if rc.dec_log(1) != 0 {
                    break;
                }
                total += 1 << 3;
                allocation -= 1 << 3;
            }
            total -= self.pulses[j];
            if intensitystereo_bit != 0 {
                total -= intensitystereo_bit;
                intensitystereo_bit = CELT_LOG2_FRAC[j - start] as i32;
                total += intensitystereo_bit;
            }
            self.pulses[j] = if allocation >= ch << 3 { ch << 3 } else { 0 };
            total += self.pulses[j];
            self.coded_bands -= 1;
        }
        self.intensity_stereo = 0;
        self.dual_stereo = false;
        if intensitystereo_bit != 0 {
            self.intensity_stereo = start + rc.dec_uint((self.coded_bands + 1 - start) as u32) as usize;
            if self.intensity_stereo < self.coded_bands { stat(3); }
        }
        if self.intensity_stereo <= start {
            tbits_8ths += dualstereo_bit;
        } else if dualstereo_bit != 0 {
            self.dual_stereo = rc.dec_log(1) != 0;
            if self.dual_stereo { stat(4); }
        }
        let mut remaining = tbits_8ths - total;
        let bandbits = remaining / (fb(self.coded_bands) - fb(start));
        remaining -= bandbits * (fb(self.coded_bands) - fb(start));
        for i in start..self.coded_bands {
            let bits = remaining.min(fr(i));
            self.pulses[i] += bits + bandbits * fr(i);
            remaining -= bits;
        }
        let mut i = start;
        while i < self.coded_bands {
            let n = fr(i) << size;
            let prev_extra = extrabits;
            self.pulses[i] += extrabits;
            if n > 1 {
                extrabits = (self.pulses[i] - self.caps[i]).max(0);
                self.pulses[i] -= extrabits;
                let dof = n * ch + (ch == 2 && n > 2 && !self.dual_stereo && i < self.intensity_stereo) as i32;
                let temp = dof * (CELT_LOG_FREQ_RANGE[i] as i32 + (size << 3));
                let mut offset = (temp >> 1) - dof * FINE_OFFSET;
                if n == 2 {
                    offset += dof << 1;
                }
                if self.pulses[i] + offset < 2 * (dof << 3) {
                    offset += temp >> 2;
                } else if self.pulses[i] + offset < 3 * (dof << 3) {
                    offset += temp >> 3;
                }
                let fine_bits = (self.pulses[i] + offset + (dof << 2)) / (dof << 3);
                let max_bits = ((self.pulses[i] >> 3) >> (ch - 1)).clamp(0, MAX_FINE_BITS);
                self.fine_bits[i] = fine_bits.clamp(0, max_bits);
                self.fine_priority[i] = (self.fine_bits[i] * (dof << 3) >= self.pulses[i] + offset) as i32;
                self.pulses[i] -= self.fine_bits[i] << (ch - 1) << 3;
            } else {
                extrabits = (self.pulses[i] - (ch << 3)).max(0);
                self.pulses[i] -= extrabits;
                self.fine_bits[i] = 0;
                self.fine_priority[i] = 1;
            }
            if extrabits > 0 {
                let mut fineextra = (extrabits >> (ch + 2)).min(MAX_FINE_BITS - self.fine_bits[i]);
                self.fine_bits[i] += fineextra;
                fineextra <<= ch + 2;
                self.fine_priority[i] = (fineextra >= extrabits - prev_extra) as i32;
                extrabits -= fineextra;
            }
            i += 1;
        }
        self.remaining = extrabits;
        while i < end {
            self.fine_bits[i] = self.pulses[i] >> (ch - 1) >> 3;
            self.pulses[i] = 0;
            self.fine_priority[i] = (self.fine_bits[i] < 1) as i32;
            i += 1;
        }
    }

    /// `ff_celt_quant_bands` (decoder).
    fn quant_bands(&mut self, rc: &mut Rc) {
        let totalbits = (self.framebits << 3) - self.anticollapse_needed;
        let mut update_lowband = true;
        let mut lowband_offset = 0usize;
        let (start, end, size) = (self.start_band, self.end_band, self.size);
        let fb = |i: usize| CELT_FREQ_BANDS[i] as usize;
        let fr = |i: usize| CELT_FREQ_RANGE[i] as usize;
        for i in start..end {
            let full = (1u32 << self.blocks) - 1;
            let mut cm = [full, full];
            let band_offset = fb(i) << size;
            let band_size = fr(i) << size;
            let x = X0 + band_offset;
            let y = if self.channels == 2 { Some(X1 + band_offset) } else { None };
            let consumed = rc.tell_frac() as i32;
            let mut effective_lowband: isize = -1;
            let mut b = 0;
            if i != start {
                self.remaining -= consumed;
            }
            self.remaining2 = totalbits - consumed - 1;
            if i < self.coded_bands {
                let curr_balance = self.remaining / 3.min(self.coded_bands as i32 - i as i32);
                b = (self.remaining2 + 1).min(self.pulses[i] + curr_balance).clamp(0, (1 << 14) - 1);
            }
            if (fb(i) as i32 - fr(i) as i32 >= fb(start) as i32 || i == start + 1) && (update_lowband || lowband_offset == 0) {
                lowband_offset = i;
            }
            if i == start + 1 {
                let count = (fr(i) - fr(i - 1)) << size;
                self.mem.copy_within(NORM1 + band_offset - count..NORM1 + band_offset, NORM1 + band_offset);
                if self.channels == 2 {
                    self.mem.copy_within(NORM2 + band_offset - count..NORM2 + band_offset, NORM2 + band_offset);
                }
            }
            if lowband_offset != 0 && (self.spread != SPREAD_AGGRESSIVE || self.blocks > 1 || self.tf_change[i] < 0) {
                effective_lowband = (fb(start) as isize).max(fb(lowband_offset) as isize - fr(i) as isize);
                let mut foldstart = lowband_offset;
                loop {
                    foldstart -= 1;
                    if (fb(foldstart) as isize) <= effective_lowband {
                        break;
                    }
                }
                let mut foldend = lowband_offset - 1;
                loop {
                    foldend += 1;
                    if !(foldend < i && (fb(foldend) as isize) < effective_lowband + fr(i) as isize) {
                        break;
                    }
                }
                cm = [0, 0];
                for j in foldstart..foldend {
                    cm[0] |= self.block[0].collapse_masks[j] as u32;
                    cm[1] |= self.block[self.channels - 1].collapse_masks[j] as u32;
                }
            }
            if self.dual_stereo && i == self.intensity_stereo {
                self.dual_stereo = false;
                for j in fb(start) << size..band_offset {
                    self.mem[NORM1 + j] = (self.mem[NORM1 + j] + self.mem[NORM2 + j]) / 2.0f32;
                }
            }
            let (nl1, nl2) = if effective_lowband != -1 {
                let e = (effective_lowband as usize) << size;
                (Some(NORM1 + e), Some(NORM2 + e))
            } else {
                (None, None)
            };
            if self.dual_stereo {
                cm[0] = self.quant_band(rc, i, x, None, band_size, b >> 1, self.blocks as u32, nl1, size as i32, Some(NORM1 + band_offset), 0, 1.0, Some(SCRATCH), cm[0]);
                cm[1] = self.quant_band(rc, i, y.unwrap(), None, band_size, b >> 1, self.blocks as u32, nl2, size as i32, Some(NORM2 + band_offset), 0, 1.0, Some(SCRATCH), cm[1]);
            } else {
                cm[0] = self.quant_band(rc, i, x, y, band_size, b, self.blocks as u32, nl1, size as i32, Some(NORM1 + band_offset), 0, 1.0, Some(SCRATCH), cm[0] | cm[1]);
                cm[1] = cm[0];
            }
            self.block[0].collapse_masks[i] = cm[0] as u8;
            self.block[self.channels - 1].collapse_masks[i] = cm[1] as u8;
            self.remaining += self.pulses[i] + consumed;
            update_lowband = b > (band_size as i32) << 3;
        }
    }

    // ---------------------------------------------------------------- pvq.c

    fn haar1(&mut self, x: usize, n0: usize, stride: usize) {
        let n0 = n0 >> 1;
        for i in 0..stride {
            for j in 0..n0 {
                let a = x + stride * (2 * j) + i;
                let b = x + stride * (2 * j + 1) + i;
                let (x0, x1) = (self.mem[a], self.mem[b]);
                self.mem[a] = ((x0 + x1) as f64 * M_SQRT1_2) as f32;
                self.mem[b] = ((x0 - x1) as f64 * M_SQRT1_2) as f32;
            }
        }
    }

    fn hadamard_order(stride: usize, hadamard: bool) -> &'static [u8] {
        &CELT_HADAMARD_ORDER[if hadamard { stride - 2 } else { 30 }..]
    }

    fn deinterleave_hadamard(&mut self, x: usize, n0: usize, stride: usize, hadamard: bool) {
        let order = Self::hadamard_order(stride, hadamard);
        let n = n0 * stride;
        for i in 0..stride {
            for j in 0..n0 {
                self.hadamard_tmp[order[i] as usize * n0 + j] = self.mem[x + j * stride + i];
            }
        }
        self.mem[x..x + n].copy_from_slice(&self.hadamard_tmp[..n]);
    }

    fn interleave_hadamard(&mut self, x: usize, n0: usize, stride: usize, hadamard: bool) {
        let order = Self::hadamard_order(stride, hadamard);
        let n = n0 * stride;
        for i in 0..stride {
            for j in 0..n0 {
                self.hadamard_tmp[j * stride + i] = self.mem[x + order[i] as usize * n0 + j];
            }
        }
        self.mem[x..x + n].copy_from_slice(&self.hadamard_tmp[..n]);
    }

    fn exp_rotation(&mut self, x: usize, len: u32, stride: u32, k: u32, spread: i32) {
        if 2 * k >= len || spread == SPREAD_NONE {
            return;
        }
        let gain = len as f32 / (len + (20 - 5 * spread as u32) * k) as f32;
        let theta = (M_PI * gain as f64 * gain as f64 / 4.0) as f32;
        let (s, c) = theta.sin_cos();
        let mut stride2 = 0u32;
        if len >= stride << 3 {
            stride2 = 1;
            while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
                stride2 += 1;
            }
        }
        let len = len / stride;
        for i in 0..stride {
            let base = x + (i * len) as usize;
            if stride2 != 0 {
                rotation_impl(&mut self.mem[base..base + len as usize], stride2 as usize, s, c);
            }
            rotation_impl(&mut self.mem[base..base + len as usize], 1, c, s);
        }
    }

    fn alg_unquant(&mut self, rc: &mut Rc, x: usize, n: u32, k: u32, spread: i32, blocks: u32, gain: f32) -> u32 {
        let norm = decode_pulses(rc, &mut self.qcoeff[..n as usize], n, k);
        let gain = gain / (norm as f32).sqrt();
        for i in 0..n as usize {
            self.mem[x + i] = gain * self.qcoeff[i] as f32;
        }
        self.exp_rotation(x, n, blocks, k, spread);
        extract_collapse_mask(&self.qcoeff, n, blocks)
    }

    fn stereo_merge(&mut self, x: usize, y: usize, mid: f32, n: usize) {
        let (mut xp, mut side) = (0f32, 0f32);
        for i in 0..n {
            xp += self.mem[x + i] * self.mem[y + i];
            side += self.mem[y + i] * self.mem[y + i];
        }
        xp *= mid;
        let mid2 = mid;
        let e0 = mid2 * mid2 + side - 2.0 * xp;
        let e1 = mid2 * mid2 + side + 2.0 * xp;
        if e0 < 6e-4f32 || e1 < 6e-4f32 {
            for i in 0..n {
                self.mem[y + i] = self.mem[x + i];
            }
            return;
        }
        let g0 = 1.0f32 / e0.sqrt();
        let g1 = 1.0f32 / e1.sqrt();
        for i in 0..n {
            let v0 = mid * self.mem[x + i];
            let v1 = self.mem[y + i];
            self.mem[x + i] = g0 * (v0 - v1);
            self.mem[y + i] = g1 * (v0 + v1);
        }
    }

    /// `quant_band_template(..., quant = 0)`.
    fn quant_band(
        &mut self,
        rc: &mut Rc,
        band: usize,
        x: usize,
        y: Option<usize>,
        n: usize,
        b: i32,
        blocks: u32,
        lowband: Option<usize>,
        duration: i32,
        lowband_out: Option<usize>,
        level: i32,
        gain: f32,
        lowband_scratch: Option<usize>,
        fill: u32,
    ) -> u32 {
        let (mut n, mut b, mut blocks, mut lowband, mut duration, mut fill, mut y) = (n, b, blocks, lowband, duration, fill, y);
        let stereo = y.is_some();
        let mut split = stereo;
        let n0 = n;
        let mut n_b = n / blocks as usize;
        let mut n_b0 = n_b;
        let mut b0 = blocks;
        let mut time_divide = 0;
        let mut recombine = 0;
        let mut inv = false;
        let mut mid = 0f32;
        let side;
        let longblocks = b0 == 1;
        let mut cm: u32 = 0;

        if n == 1 {
            let mut px = x;
            for i in 0..=stereo as usize {
                let mut sign = 0;
                if self.remaining2 >= 1 << 3 {
                    sign = rc.get_raw(1);
                    self.remaining2 -= 1 << 3;
                }
                self.mem[px] = 1.0f32 - 2.0f32 * sign as f32;
                if i == 0 {
                    if let Some(yy) = y {
                        px = yy;
                    }
                }
            }
            if let Some(lo) = lowband_out {
                self.mem[lo] = self.mem[x];
            }
            return 1;
        }

        if !stereo && level == 0 {
            let mut tf_change = self.tf_change[band];
            if tf_change > 0 {
                recombine = tf_change;
            }
            if let Some(lb) = lowband {
                if recombine != 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1 {
                    let sc = lowband_scratch.unwrap();
                    self.mem.copy_within(lb..lb + n, sc);
                    lowband = Some(sc);
                }
            }
            for k in 0..recombine {
                if let Some(lb) = lowband {
                    self.haar1(lb, n >> k, 1 << k);
                }
                fill = CELT_BIT_INTERLEAVE[(fill & 0xF) as usize] as u32 | (CELT_BIT_INTERLEAVE[(fill >> 4) as usize] as u32) << 2;
            }
            blocks >>= recombine;
            n_b <<= recombine;
            while (n_b & 1) == 0 && tf_change < 0 {
                if let Some(lb) = lowband {
                    self.haar1(lb, n_b, blocks as usize);
                }
                fill |= fill << blocks;
                blocks <<= 1;
                n_b >>= 1;
                time_divide += 1;
                tf_change += 1;
            }
            b0 = blocks;
            n_b0 = n_b;
            if b0 > 1 {
                if let Some(lb) = lowband {
                    self.deinterleave_hadamard(lb, n_b >> recombine, (b0 << recombine) as usize, longblocks);
                }
            }
        }

        let cache_at = CELT_CACHE_INDEX[(duration + 1) as usize * MAX_BANDS + band] as usize;
        let cache = &CELT_CACHE_BITS[cache_at..];
        if !stereo && duration >= 0 && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
            n >>= 1;
            y = Some(x + n);
            split = true;
            duration -= 1;
            if blocks == 1 {
                fill = (fill & 1) | (fill << 1);
            }
            blocks = (blocks + 1) >> 1;
        }

        if split {
            let mut itheta: i32 = 0;
            let pulse_cap = CELT_LOG_FREQ_RANGE[band] as i32 + duration * 8;
            let offset = (pulse_cap >> 1) - if stereo && n == 2 { QTHETA_OFFSET_TWOPHASE } else { QTHETA_OFFSET };
            let qn = if stereo && band >= self.intensity_stereo { 1 } else { compute_qn(n as i32, b, offset, pulse_cap, stereo) };
            let tell = rc.tell_frac() as i32;
            if qn != 1 {
                itheta = if stereo && n > 2 {
                    rc.dec_uint_step((qn / 2) as u32) as i32
                } else if stereo || b0 > 1 {
                    rc.dec_uint((qn + 1) as u32) as i32
                } else {
                    rc.dec_uint_tri(qn as u32) as i32
                };
                itheta = itheta * 16384 / qn;
            } else if stereo {
                inv = if b > 2 << 3 && self.remaining2 > 2 << 3 { rc.dec_log(2) != 0 } else { false };
                inv = if self.apply_phase_inv { inv } else { false };
                if inv { stat(9); }
                itheta = 0;
            }
            let qalloc = rc.tell_frac() as i32 - tell;
            b -= qalloc;
            let orig_fill = fill;
            let (imid, iside, mut delta);
            if itheta == 0 {
                imid = 32767;
                iside = 0;
                fill &= if blocks >= 32 { u32::MAX } else { (1u32 << blocks) - 1 };
                delta = -16384;
            } else if itheta == 16384 {
                imid = 0;
                iside = 32767;
                fill &= ((1u32 << blocks) - 1) << blocks;
                delta = 16384;
            } else {
                imid = celt_cos(itheta as i16) as i32;
                iside = celt_cos((16384 - itheta) as i16) as i32;
                delta = round_mul16(((n as i32) - 1) << 7, celt_log2tan(iside, imid));
            }
            mid = imid as f32 / 32768.0f32;
            side = iside as f32 / 32768.0f32;

            if n == 2 && stereo {
                stat(7);
                let yy = y.unwrap();
                let mut mbits = b;
                let sbits = if itheta != 0 && itheta != 16384 { 1 << 3 } else { 0 };
                mbits -= sbits;
                let c = itheta > 8192;
                self.remaining2 -= qalloc + sbits;
                let (x2, y2) = if c { (yy, x) } else { (x, yy) };
                let mut sign = 0;
                if sbits != 0 {
                    sign = rc.get_raw(1) as i32;
                }
                let sign = 1 - 2 * sign;
                cm = self.quant_band(rc, band, x2, None, n, mbits, blocks, lowband, duration, lowband_out, level, gain, lowband_scratch, orig_fill);
                self.mem[y2] = -sign as f32 * self.mem[x2 + 1];
                self.mem[y2 + 1] = sign as f32 * self.mem[x2];
                self.mem[x] *= mid;
                self.mem[x + 1] *= mid;
                self.mem[yy] *= side;
                self.mem[yy + 1] *= side;
                let tmp = self.mem[x];
                self.mem[x] = tmp - self.mem[yy];
                self.mem[yy] += tmp;
                let tmp = self.mem[x + 1];
                self.mem[x + 1] = tmp - self.mem[yy + 1];
                self.mem[yy + 1] += tmp;
            } else {
                let yy = y.unwrap();
                let mut next_lowband2 = None;
                let mut next_lowband_out1 = None;
                let mut next_level = 0;
                if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                    if itheta > 8192 {
                        delta -= delta >> (4 - duration);
                    } else {
                        delta = 0.min(delta + (((n as i32) << 3) >> (5 - duration)));
                    }
                }
                let mut mbits = ((b - delta) / 2).clamp(0, b);
                let mut sbits = b - mbits;
                self.remaining2 -= qalloc;
                if let Some(lb) = lowband {
                    if !stereo {
                        next_lowband2 = Some(lb + n);
                    }
                }
                if stereo {
                    next_lowband_out1 = lowband_out;
                } else {
                    next_level = level + 1;
                }
                let mut rebalance = self.remaining2;
                let shift = (b0 >> 1) & (if stereo { 0 } else { u32::MAX });
                if mbits >= sbits {
                    cm = self.quant_band(rc, band, x, None, n, mbits, blocks, lowband, duration, next_lowband_out1, next_level, if stereo { 1.0 } else { gain * mid }, lowband_scratch, fill);
                    rebalance = mbits - (rebalance - self.remaining2);
                    if rebalance > 3 << 3 && itheta != 0 {
                        sbits += rebalance - (3 << 3);
                    }
                    let cmt = self.quant_band(rc, band, yy, None, n, sbits, blocks, next_lowband2, duration, None, next_level, gain * side, None, fill >> blocks);
                    cm |= cmt << shift;
                } else {
                    cm = self.quant_band(rc, band, yy, None, n, sbits, blocks, next_lowband2, duration, None, next_level, gain * side, None, fill >> blocks);
                    cm <<= shift;
                    rebalance = sbits - (rebalance - self.remaining2);
                    if rebalance > 3 << 3 && itheta != 16384 {
                        mbits += rebalance - (3 << 3);
                    }
                    cm |= self.quant_band(rc, band, x, None, n, mbits, blocks, lowband, duration, next_lowband_out1, next_level, if stereo { 1.0 } else { gain * mid }, lowband_scratch, fill);
                }
            }
        } else {
            let mut q = bits2pulses(cache, b);
            let mut curr_bits = pulses2bits(cache, q);
            self.remaining2 -= curr_bits;
            while self.remaining2 < 0 && q > 0 {
                self.remaining2 += curr_bits;
                q -= 1;
                curr_bits = pulses2bits(cache, q);
                self.remaining2 -= curr_bits;
            }
            if q != 0 {
                let k = if q < 8 { q } else { (8 + (q & 7)) << ((q >> 3) - 1) };
                cm = self.alg_unquant(rc, x, n as u32, k as u32, self.spread, blocks, gain);
            } else {
                let cm_mask = (1u32 << blocks) - 1;
                fill &= cm_mask;
                if fill != 0 {
                    match lowband {
                        None => {
                            stat(8);
                            for i in 0..n {
                                self.mem[x + i] = ((self.rng() as i32) >> 20) as f32;
                            }
                            cm = cm_mask;
                        }
                        Some(lb) => {
                            for i in 0..n {
                                let d = if self.rng() & 0x8000 != 0 { 1.0f32 / 256.0 } else { -1.0f32 / 256.0 };
                                self.mem[x + i] = self.mem[lb + i] + d;
                            }
                            cm = fill;
                        }
                    }
                    renormalize_vector(&mut self.mem[x..x + n], gain);
                } else {
                    self.mem[x..x + n].iter_mut().for_each(|v| *v = 0.0);
                }
            }
        }

        if stereo {
            let yy = y.unwrap();
            if n > 2 {
                self.stereo_merge(x, yy, mid, n);
            }
            if inv {
                for i in 0..n {
                    self.mem[yy + i] *= -1.0;
                }
            }
        } else if level == 0 {
            if b0 > 1 {
                self.interleave_hadamard(x, n_b >> recombine, (b0 << recombine) as usize, longblocks);
            }
            n_b = n_b0;
            blocks = b0;
            for _ in 0..time_divide {
                blocks >>= 1;
                n_b <<= 1;
                cm |= cm >> blocks;
                self.haar1(x, n_b, blocks as usize);
            }
            for k in 0..recombine {
                cm = CELT_BIT_DEINTERLEAVE[cm as usize] as u32;
                self.haar1(x, n0 >> k, 1 << k);
            }
            blocks <<= recombine;
            if let Some(lo) = lowband_out {
                let nn = (n0 as f32).sqrt();
                for i in 0..n0 {
                    self.mem[lo + i] = nn * self.mem[x + i];
                }
            }
            cm &= if blocks >= 32 { u32::MAX } else { (1u32 << blocks) - 1 };
        }
        cm
    }
}

fn renormalize_vector(x: &mut [f32], gain: f32) {
    let mut g = 1e-15f32;
    for v in x.iter() {
        g += v * v;
    }
    let g = gain / g.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

fn rotation_impl(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    for i in 0..len - stride {
        let (x1, x2) = (x[i], x[i + stride]);
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    let mut i = len as isize - 2 * stride as isize - 1;
    while i >= 0 {
        let k = i as usize;
        let (x1, x2) = (x[k], x[k + stride]);
        x[k + stride] = c * x2 + s * x1;
        x[k] = c * x1 - s * x2;
        i -= 1;
    }
}

fn extract_collapse_mask(iy: &[i32], n: u32, b: u32) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = (n / b) as usize;
    let mut m = 0u32;
    for i in 0..b as usize {
        for j in 0..n0 {
            m |= ((iy[i * n0 + j] != 0) as u32) << i;
        }
    }
    m
}

#[inline]
fn round_mul16(a: i32, b: i32) -> i32 {
    (a * b + 16384) >> 15
}

fn celt_cos(x: i16) -> i16 {
    let x = (((x as i32) * (x as i32) + 4096) >> 13) as i16 as i32;
    let x = ((32767 - x) + round_mul16(x, -7651 + round_mul16(x, 8277 + round_mul16(-626, x)))) as i16;
    (x as i32 + 1) as i16
}

fn celt_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog(icos as u32) as i32;
    let ls = ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls << 11) - (lc << 11) + round_mul16(isin, round_mul16(isin, -2597) + 7932) - round_mul16(icos, round_mul16(icos, -2597) + 7932)
}

fn bits2pulses(cache: &[u8], bits: i32) -> i32 {
    let mut low = 0i32;
    let mut high = cache[0] as i32;
    let bits = bits - 1;
    for _ in 0..6 {
        let center = (low + high + 1) >> 1;
        if cache[center as usize] as i32 >= bits {
            high = center;
        } else {
            low = center;
        }
    }
    let lowv = if low == 0 { -1 } else { cache[low as usize] as i32 };
    if bits - lowv <= cache[high as usize] as i32 - bits {
        low
    } else {
        high
    }
}

fn pulses2bits(cache: &[u8], pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        cache[pulses as usize] as i32 + 1
    }
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let qb = (b - pulse_cap - (4 << 3)).min((b + n2 * offset) / n2).min(8 << 3);
    if qb < (1 << 3 >> 1) {
        1
    } else {
        (((CELT_QN_EXP2[(qb & 0x7) as usize] as i32 >> (14 - (qb >> 3))) + 1) >> 1) << 1
    }
}

#[inline]
fn pvq_u(n: u32, k: u32) -> u32 {
    let (a, b) = (n.min(k), n.max(k));
    CELT_PVQ_U[PVQ_ROW[a as usize] + b as usize]
}

const PVQ_ROW: [usize; 15] = [0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257];

#[inline]
fn row(n: u32, k: u32) -> u32 {
    CELT_PVQ_U[PVQ_ROW[n as usize] + k as usize]
}

/// `celt_decode_pulses` -> `celt_cwrsi`.
fn decode_pulses(rc: &mut Rc, y: &mut [i32], n: u32, k: u32) -> u64 {
    let idx = rc.dec_uint(pvq_u(n, k) + pvq_u(n, k + 1));
    cwrsi(n, k, idx, y)
}

fn cwrsi(mut n: u32, mut k: u32, mut i: u32, y: &mut [i32]) -> u64 {
    let mut norm: u64 = 0;
    let mut yi = 0;
    let mut push = |v: i32, norm: &mut u64| {
        *norm += (v * v) as u64;
        y[yi] = v;
        yi += 1;
    };
    while n > 2 {
        if k >= n {
            let p = row(n, k + 1);
            let s: i32 = -((i >= p) as i32);
            i -= p & s as u32;
            let k0 = k;
            let q = row(n, n);
            let mut p;
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    p = row(k, n);
                    if p <= i {
                        break;
                    }
                }
            } else {
                p = row(n, k);
                while p > i {
                    k -= 1;
                    p = row(n, k);
                }
            }
            i -= p;
            let val = (k0 as i32 - k as i32 + s) ^ s;
            push(val, &mut norm);
        } else {
            let p = row(k, n);
            let q = row(k + 1, n);
            if p <= i && i < q {
                i -= p;
                push(0, &mut norm);
            } else {
                let s: i32 = -((i >= q) as i32);
                i -= q & s as u32;
                let k0 = k;
                let mut p;
                loop {
                    k -= 1;
                    p = row(k, n);
                    if p <= i {
                        break;
                    }
                }
                i -= p;
                let val = (k0 as i32 - k as i32 + s) ^ s;
                push(val, &mut norm);
            }
        }
        n -= 1;
    }
    // N == 2
    let p = 2 * k + 1;
    let s: i32 = -((i >= p) as i32);
    i -= p & s as u32;
    let k0 = k;
    k = i.div_ceil(2);
    if k != 0 {
        i -= 2 * k - 1;
    }
    let val = (k0 as i32 - k as i32 + s) ^ s;
    push(val, &mut norm);
    // N == 1
    let s = -(i as i32);
    let val = (k as i32 + s) ^ s;
    push(val, &mut norm);
    norm
}

/// `ff_vector_fmul_window_neon(dst, dst, dst + len, win, len)` in place.
fn fmul_window_inplace(d: &mut [f32], win: &[f32], len: usize) {
    for k in 0..len {
        let (i, j) = (k, 2 * len - 1 - k);
        let s0 = d[k];
        let s1 = d[2 * len - 1 - k];
        let (wi, wj) = (win[i], win[j]);
        d[i] = (-s1).mul_add(wi, s0 * wj);
        d[j] = s1.mul_add(wj, s0 * wi);
    }
}

/// `celt_postfilter` (+ `celt_postfilter_apply_transition` in double, and
/// `ff_opus_postfilter_neon`).
fn postfilter(b: &mut Block, len: usize) {
    let filter_len = len as isize - 2 * OVERLAP as isize;
    apply_transition(b, 1024);
    b.pf_period_old = b.pf_period;
    b.pf_gains_old = b.pf_gains;
    b.pf_period = b.pf_period_new;
    b.pf_gains = b.pf_gains_new;
    if len > OVERLAP {
        apply_transition(b, 1024 + OVERLAP);
        if b.pf_gains[0] > f32::EPSILON && filter_len > 0 {
            postfilter_neon(&mut b.buf, 1024 + 2 * OVERLAP, b.pf_period as usize, b.pf_gains, filter_len as usize);
        }
        b.pf_period_old = b.pf_period;
        b.pf_gains_old = b.pf_gains;
    }
    b.buf.copy_within(len..len + 1024 + OVERLAP / 2, 0);
}

fn apply_transition(b: &mut Block, at: usize) {
    let t0 = b.pf_period_old as usize;
    let t1 = b.pf_period as usize;
    if b.pf_gains[0] == 0.0 && b.pf_gains_old[0] == 0.0 {
        return;
    }
    let [g00, g01, g02] = b.pf_gains_old;
    let [g10, g11, g12] = b.pf_gains;
    let d = &mut b.buf;
    let mut x1 = d[at - t1 + 1];
    let mut x2 = d[at - t1];
    let mut x3 = d[at - t1 - 1];
    let mut x4 = d[at - t1 - 2];
    for i in 0..OVERLAP {
        let w = CELT_WINDOW2[i];
        let x0 = d[at + i - t1 + 2];
        let iw = 1.0f64 - w as f64;
        let sum = iw * g00 as f64 * d[at + i - t0] as f64
            + iw * g01 as f64 * (d[at + i - t0 - 1] + d[at + i - t0 + 1]) as f64
            + iw * g02 as f64 * (d[at + i - t0 - 2] + d[at + i - t0 + 2]) as f64
            + (w * g10 * x2) as f64
            + (w * g11 * (x1 + x3)) as f64
            + (w * g12 * (x0 + x4)) as f64;
        d[at + i] = (d[at + i] as f64 + sum) as f32;
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// `ff_opus_postfilter_neon`, four lanes at a time (len is a multiple of 4).
fn postfilter_neon(d: &mut [f32], at: usize, period: usize, g: [f32; 3], len: usize) {
    let mut i = 0;
    while i < len {
        for l in 0..4 {
            let p = at + i + l;
            let x4 = d[p - period - 2];
            let x3 = d[p - period - 1];
            let x2 = d[p - period];
            let x1 = d[p - period + 1];
            let x0 = d[p - period + 2];
            let v3 = x0.mul_add(g[2], x4 * g[2]);
            let v6 = (x1 + x3) * g[1] + v3;
            let v4 = x2.mul_add(g[0], d[p]);
            d[p] = v4 + v6;
        }
        i += 4;
    }
}

/// `ff_opus_deemphasis_neon`, eight samples per step.
fn deemphasis(y: &mut [f32], x: &[f32], coeff: f32, w: &[f32; 16]) -> f32 {
    let (v4, v5, v6, v7) = (&w[0..4], &w[4..8], &w[8..12], &w[12..16]);
    let mut v0 = [0f32; 4];
    for l in 0..4 {
        v0[l] = v4[l] * coeff;
    }
    let mut last = coeff;
    for (yc, xc) in y.chunks_mut(8).zip(x.chunks(8)) {
        let mut v1 = [xc[0], xc[1], xc[2], xc[3]];
        let mut v2 = [xc[4], xc[5], xc[6], xc[7]];
        let mut v3 = [0f32; 4];
        for l in 0..4 {
            v0[l] = v5[l].mul_add(v1[0], v0[l]);
            v3[l] = v7[l] * v2[2];
            v0[l] = v6[l].mul_add(v1[1], v0[l]);
            v3[l] = v6[l].mul_add(v2[1], v3[l]);
            v0[l] = v7[l].mul_add(v1[2], v0[l]);
            v3[l] = v5[l].mul_add(v2[0], v3[l]);
        }
        for l in 0..4 {
            v1[l] += v0[l];
            v2[l] += v3[l];
        }
        for l in 0..4 {
            v2[l] = v4[l].mul_add(v1[3], v2[l]);
        }
        yc[..4].copy_from_slice(&v1);
        yc[4..].copy_from_slice(&v2);
        for l in 0..4 {
            v0[l] = v4[l] * v2[3];
        }
        last = v2[3];
    }
    last
}
