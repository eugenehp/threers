//! SILK, as ffmpeg n7.1 decodes it (`libavcodec/opus/silk.c`): fixed-point
//! LSF/LPC reconstruction with C's integer widths, float synthesis
//! (uncontracted).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::rc::{ilog, Rc};
use super::tab::*;

pub const HISTORY: usize = 322;
const LTP_ORDER: usize = 5;
const MAX_LAG: usize = 288 + LTP_ORDER / 2;

#[derive(Clone)]
struct Frame {
    coded: bool,
    log_gain: i32,
    nlsf: [i16; 16],
    lpc: [f32; 16],
    output: Vec<f32>,
    lpc_history: Vec<f32>,
    primarylag: i32,
    prev_voiced: bool,
}

impl Frame {
    fn new() -> Self {
        Frame {
            coded: false,
            log_gain: 0,
            nlsf: [0; 16],
            lpc: [0.0; 16],
            output: vec![0.0; 2 * HISTORY],
            lpc_history: vec![0.0; 2 * HISTORY],
            primarylag: 0,
            prev_voiced: false,
        }
    }

    fn flush(&mut self) {
        if !self.coded {
            return;
        }
        *self = Frame::new();
    }
}

pub struct Silk {
    output_channels: usize,
    midonly: bool,
    subframes: usize,
    sflength: usize,
    flength: usize,
    nlsf_interp_factor: i32,
    bandwidth: usize,
    wb: bool,
    frame: [Frame; 2],
    prev_stereo_weights: [f32; 2],
    stereo_weights: [f32; 2],
    prev_coded_channels: usize,
}

#[inline]
fn round_mull(a: i64, b: i64, s: u32) -> i64 {
    (((a * b) >> (s - 1)) + 1) >> 1
}
#[inline]
fn mulh(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}
#[inline]
fn mull(a: i32, b: i32, s: u32) -> i32 {
    ((a as i64 * b as i64) >> s) as i32
}

fn stabilize_lsf(nlsf: &mut [i16; 16], order: usize, min_delta: &[u16]) {
    for _pass in 0..20 {
        let mut k = 0;
        let mut min_diff = 0i32;
        for i in 0..order + 1 {
            let low = if i != 0 { nlsf[i - 1] as i32 } else { 0 };
            let high = if i != order { nlsf[i] as i32 } else { 32768 };
            let diff = (high - low) - min_delta[i] as i32;
            if diff < min_diff {
                min_diff = diff;
                k = i;
            }
        }
        if min_diff == 0 {
            return;
        }
        if k == 0 {
            nlsf[0] = min_delta[0] as i16;
        } else if k == order {
            nlsf[order - 1] = (32768 - min_delta[order] as i32) as i16;
        } else {
            let mut min_center = 0i32;
            let mut max_center = 32768i32;
            for i in 0..k {
                min_center += min_delta[i] as i32;
            }
            min_center += min_delta[k] as i32 >> 1;
            let mut i = order;
            while i > k {
                max_center -= min_delta[i] as i32;
                i -= 1;
            }
            max_center -= min_delta[k] as i32 >> 1;
            let mut center_val = nlsf[k - 1] as i32 + nlsf[k] as i32;
            center_val = (center_val >> 1) + (center_val & 1);
            center_val = max_center.min(min_center.max(center_val));
            nlsf[k - 1] = (center_val - (min_delta[k] as i32 >> 1)) as i16;
            nlsf[k] = (nlsf[k - 1] as i32 + min_delta[k] as i32) as i16;
        }
    }
    for i in 1..order {
        let value = nlsf[i];
        let mut j = i as isize - 1;
        while j >= 0 && nlsf[j as usize] > value {
            nlsf[(j + 1) as usize] = nlsf[j as usize];
            j -= 1;
        }
        nlsf[(j + 1) as usize] = value;
    }
    if (nlsf[0] as i32) < min_delta[0] as i32 {
        nlsf[0] = min_delta[0] as i16;
    }
    for i in 1..order {
        nlsf[i] = (nlsf[i] as i32).max((nlsf[i - 1] as i32 + min_delta[i] as i32).min(32767)) as i16;
    }
    if nlsf[order - 1] as i32 > 32768 - min_delta[order] as i32 {
        nlsf[order - 1] = (32768 - min_delta[order] as i32) as i16;
    }
    for i in (0..order - 1).rev() {
        if nlsf[i] as i32 > nlsf[i + 1] as i32 - min_delta[i + 1] as i32 {
            nlsf[i] = (nlsf[i + 1] as i32 - min_delta[i + 1] as i32) as i16;
        }
    }
}

fn is_lpc_stable(lpc: &[i16; 16], order: usize) -> bool {
    let mut lpc32 = [[0i32; 16]; 2];
    let mut dc_resp = 0i32;
    let mut totalinvgain: i32 = 1 << 30;
    let mut cur = 0usize;
    for k in 0..order {
        dc_resp += lpc[k] as i32;
        lpc32[0][k] = lpc[k] as i32 * 4096;
    }
    if dc_resp >= 4096 {
        return false;
    }
    let mut k = order - 1;
    loop {
        let row = lpc32[cur];
        if row[k].abs() > 16773022 {
            return false;
        }
        let rc = -(row[k] * 128);
        let gaindiv = (1i32 << 30) - mulh(rc, rc);
        totalinvgain = mulh(totalinvgain, gaindiv) << 2;
        if k == 0 {
            return totalinvgain >= 107374;
        }
        let fbits = ilog(gaindiv as u32);
        let mut gain = ((1i32 << 29) - 1) / (gaindiv >> (fbits + 1 - 16));
        let error = (1i32 << 29) - mull(gaindiv << (15 + 16 - fbits), gain, 16);
        gain = (gain << 16).wrapping_add(error.wrapping_mul(gain) >> 13);
        let prevrow = row;
        cur = k & 1;
        for j in 0..k {
            let x = (prevrow[j] as i64 - round_mull(prevrow[k - j - 1] as i64, rc as i64, 31)).clamp(i32::MIN as i64, i32::MAX as i64);
            let tmp = round_mull(x, gain as i64, fbits);
            if tmp < i32::MIN as i64 || tmp > i32::MAX as i64 {
                return false;
            }
            lpc32[cur][j] = tmp as i32;
        }
        k -= 1;
    }
}

fn lsp2poly(lsp: &[i32], pol: &mut [i32; 9], half_order: usize) {
    pol[0] = 65536;
    pol[1] = -lsp[0];
    for i in 1..half_order {
        pol[i + 1] = (pol[i - 1] as i64 * 2 - round_mull(lsp[2 * i] as i64, pol[i] as i64, 16)) as i32;
        let mut j = i;
        while j > 1 {
            pol[j] = (pol[j] as i64 + pol[j - 2] as i64 - round_mull(lsp[2 * i] as i64, pol[j - 1] as i64, 16)) as i32;
            j -= 1;
        }
        pol[1] -= lsp[2 * i];
    }
}

fn lsf2lpc(nlsf: &[i16; 16], lpcf: &mut [f32; 16], order: usize) {
    let mut lsp = [0i32; 16];
    let mut p = [0i32; 9];
    let mut q = [0i32; 9];
    let mut lpc32 = [0i32; 16];
    let mut lpc = [0i16; 16];
    for k in 0..order {
        let index = (nlsf[k] >> 8) as usize;
        let offset = (nlsf[k] & 255) as i32;
        let k2 = if order == 10 { SILK_LSF_ORDERING_NBMB[k] } else { SILK_LSF_ORDERING_WB[k] } as usize;
        let mut v = SILK_COSINE[index] as i32 * 256;
        v += (SILK_COSINE[index + 1] as i32 - SILK_COSINE[index] as i32) * offset;
        lsp[k2] = (v + 4) >> 3;
    }
    lsp2poly(&lsp, &mut p, order >> 1);
    lsp2poly(&lsp[1..], &mut q, order >> 1);
    for k in 0..order >> 1 {
        let p_tmp = p[k + 1].wrapping_add(p[k]);
        let q_tmp = q[k + 1].wrapping_sub(q[k]);
        lpc32[k] = (-q_tmp).wrapping_sub(p_tmp);
        lpc32[order - k - 1] = q_tmp.wrapping_sub(p_tmp);
    }
    let mut i = 0;
    while i < 10 {
        // ffmpeg reads lpc32[k] (the running argmax), not lpc32[j].
        let mut maxabs: u32 = 0;
        let mut k = 0usize;
        for j in 0..order {
            let x = lpc32[k].unsigned_abs();
            if x > maxabs {
                maxabs = x;
                k = j;
            }
        }
        maxabs = (maxabs + 16) >> 5;
        if maxabs > 32767 {
            maxabs = maxabs.min(163838);
            let chirp_base = 65470u32.wrapping_sub(((maxabs - 32767) << 14) / ((maxabs * (k as u32 + 1)) >> 2));
            let mut chirp = chirp_base;
            for k in 0..order {
                lpc32[k] = round_mull(lpc32[k] as i64, chirp as i64, 16) as i32;
                chirp = (chirp_base.wrapping_mul(chirp).wrapping_add(32768)) >> 16;
            }
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        for k in 0..order {
            let x = (lpc32[k] + 16) >> 5;
            lpc[k] = x.clamp(-32768, 32767) as i16;
            lpc32[k] = (lpc[k] as i32) << 5;
        }
    } else {
        for k in 0..order {
            lpc[k] = ((lpc32[k] + 16) >> 5) as i16;
        }
    }
    let mut i = 1;
    while i <= 16 && !is_lpc_stable(&lpc, order) {
        let chirp_base = 65536u32 - (1 << i);
        let mut chirp = chirp_base;
        for k in 0..order {
            lpc32[k] = round_mull(lpc32[k] as i64, chirp as i64, 16) as i32;
            lpc[k] = ((lpc32[k] + 16) >> 5) as i16;
            chirp = (chirp_base.wrapping_mul(chirp).wrapping_add(32768)) >> 16;
        }
        i += 1;
    }
    for i in 0..order {
        lpcf[i] = lpc[i] as f32 / 4096.0f32;
    }
}

impl Silk {
    pub fn new(output_channels: usize) -> Self {
        let mut s = Silk {
            output_channels,
            midonly: false,
            subframes: 0,
            sflength: 0,
            flength: 0,
            nlsf_interp_factor: 0,
            bandwidth: 0,
            wb: false,
            frame: [Frame::new(), Frame::new()],
            prev_stereo_weights: [0.0; 2],
            stereo_weights: [0.0; 2],
            prev_coded_channels: 0,
        };
        s.flush();
        s
    }

    pub fn flush(&mut self) {
        self.frame[0].flush();
        self.frame[1].flush();
        self.prev_stereo_weights = [0.0; 2];
    }

    fn decode_lpc(&mut self, ch: usize, rc: &mut Rc, lpc_leadin: &mut [f32; 16], lpc: &mut [f32; 16], voiced: usize) -> (usize, bool) {
        let wb = self.wb;
        let order = if wb { 16 } else { 10 };
        let mut lsf_i2 = [0i8; 16];
        let mut lsf_res = [0i16; 16];
        let mut nlsf = [0i16; 16];
        let lsf_i1 = rc.dec_cdf(&SILK_MODEL_LSF_S1[wb as usize][voiced]) as usize;
        for i in 0..order {
            let index = if wb { SILK_LSF_S2_MODEL_SEL_WB[lsf_i1][i] } else { SILK_LSF_S2_MODEL_SEL_NBMB[lsf_i1][i] } as usize;
            lsf_i2[i] = (rc.dec_cdf(&SILK_MODEL_LSF_S2[index]) as i32 - 4) as i8;
            if lsf_i2[i] == -4 {
                lsf_i2[i] -= rc.dec_cdf(&SILK_MODEL_LSF_S2_EXT) as i8;
            } else if lsf_i2[i] == 4 {
                lsf_i2[i] += rc.dec_cdf(&SILK_MODEL_LSF_S2_EXT) as i8;
            }
        }
        for i in (0..order).rev() {
            let qstep = if wb { 9830 } else { 11796 };
            lsf_res[i] = (lsf_i2[i] as i32 * 1024) as i16;
            if lsf_i2[i] < 0 {
                lsf_res[i] = (lsf_res[i] as i32 + 102) as i16;
            } else if lsf_i2[i] > 0 {
                lsf_res[i] = (lsf_res[i] as i32 - 102) as i16;
            }
            lsf_res[i] = ((lsf_res[i] as i32 * qstep) >> 16) as i16;
            if i + 1 < order {
                let weight = if wb {
                    SILK_LSF_PRED_WEIGHTS_WB[SILK_LSF_WEIGHT_SEL_WB[lsf_i1][i] as usize][i]
                } else {
                    SILK_LSF_PRED_WEIGHTS_NBMB[SILK_LSF_WEIGHT_SEL_NBMB[lsf_i1][i] as usize][i]
                } as i32;
                lsf_res[i] = (lsf_res[i] as i32 + ((lsf_res[i + 1] as i32 * weight) >> 8)) as i16;
            }
        }
        for i in 0..order {
            let codebook: &[u8] = if wb { &SILK_LSF_CODEBOOK_WB[lsf_i1] } else { &SILK_LSF_CODEBOOK_NBMB[lsf_i1] };
            let cur = codebook[i] as i32;
            let prev = if i > 0 { codebook[i - 1] as i32 } else { 0 };
            let next = if i + 1 < order { codebook[i + 1] as i32 } else { 256 };
            let weight_sq = (1024 / (cur - prev) + 1024 / (next - cur)) << 16;
            let ipart = ilog(weight_sq as u32) as i32;
            let fpart = (weight_sq >> (ipart - 8)) & 127;
            let y = (if ipart & 1 != 0 { 32768 } else { 46214 }) >> ((32 - ipart) >> 1);
            let weight = y + ((213 * fpart * y) >> 16);
            let value = cur * 128 + (lsf_res[i] as i32 * 16384) / weight;
            nlsf[i] = value.clamp(0, 32767) as i16;
        }
        stabilize_lsf(&mut nlsf, order, if wb { &SILK_LSF_MIN_SPACING_WB } else { &SILK_LSF_MIN_SPACING_NBMB });
        let frame = &mut self.frame[ch];
        let mut has_lpc_leadin = false;
        if self.subframes == 4 {
            let mut offset = rc.dec_cdf(&SILK_MODEL_LSF_INTERPOLATION_OFFSET) as i32;
            if offset != 4 && frame.coded {
                has_lpc_leadin = true;
                if offset != 0 {
                    let mut leadin = [0i16; 16];
                    for i in 0..order {
                        leadin[i] = (frame.nlsf[i] as i32 + (((nlsf[i] as i32 - frame.nlsf[i] as i32) * offset) >> 2)) as i16;
                    }
                    lsf2lpc(&leadin, lpc_leadin, order);
                } else {
                    *lpc_leadin = frame.lpc;
                }
            } else {
                offset = 4;
            }
            self.nlsf_interp_factor = offset;
            lsf2lpc(&nlsf, lpc, order);
        } else {
            self.nlsf_interp_factor = 4;
            lsf2lpc(&nlsf, lpc, order);
        }
        frame.nlsf[..order].copy_from_slice(&nlsf[..order]);
        frame.lpc[..order].copy_from_slice(&lpc[..order]);
        (order, has_lpc_leadin)
    }

    fn count_children(rc: &mut Rc, model: usize, total: i32, child: &mut [i32]) {
        if total != 0 {
            let off = (((total - 1 + 5) * (total - 1)) >> 1) as usize;
            child[0] = rc.dec_cdf(&SILK_MODEL_PULSE_LOCATION[model][off..]) as i32;
            child[1] = total - child[0];
        } else {
            child[0] = 0;
            child[1] = 0;
        }
    }

    fn decode_excitation(&self, rc: &mut Rc, out: &mut [f32], qoffset_high: usize, active: usize, voiced: usize) {
        let mut pulsecount = [0u8; 20];
        let mut lsbcount = [0u8; 20];
        let mut excitation = [0i32; 320];
        let mut seed = rc.dec_cdf(&SILK_MODEL_LCG_SEED);
        let shellblocks = SILK_SHELL_BLOCKS[self.bandwidth][self.subframes >> 2] as usize;
        let ratelevel = rc.dec_cdf(&SILK_MODEL_EXC_RATE[voiced]) as usize;
        for i in 0..shellblocks {
            pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[ratelevel]) as u8;
            if pulsecount[i] == 17 {
                while pulsecount[i] == 17 {
                    lsbcount[i] += 1;
                    if lsbcount[i] == 10 {
                        break;
                    }
                    pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[9]) as u8;
                }
                if lsbcount[i] == 10 {
                    pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[10]) as u8;
                }
            }
        }
        for i in 0..shellblocks {
            if pulsecount[i] != 0 {
                let mut loc = 16 * i;
                let mut branch = [[0i32; 2]; 4];
                branch[0][0] = pulsecount[i] as i32;
                let (b0, rest) = branch.split_at_mut(1);
                Self::count_children(rc, 0, b0[0][0], &mut rest[0]);
                for b in 0..2 {
                    let t1 = rest[0][b];
                    Self::count_children(rc, 1, t1, &mut rest[1]);
                    for c in 0..2 {
                        let t2 = rest[1][c];
                        Self::count_children(rc, 2, t2, &mut rest[2]);
                        for d in 0..2 {
                            let t3 = rest[2][d];
                            Self::count_children(rc, 3, t3, &mut excitation[loc..loc + 2]);
                            loc += 2;
                        }
                    }
                }
            }
        }
        for i in 0..shellblocks << 4 {
            for _ in 0..lsbcount[i >> 4] {
                excitation[i] = (excitation[i] << 1) | rc.dec_cdf(&SILK_MODEL_EXCITATION_LSB) as i32;
            }
        }
        for i in 0..shellblocks << 4 {
            if excitation[i] != 0 {
                let sign = rc.dec_cdf(&SILK_MODEL_EXCITATION_SIGN[active + voiced][qoffset_high][(pulsecount[i >> 4] as usize).min(6)]);
                if sign == 0 {
                    excitation[i] *= -1;
                }
            }
        }
        for i in 0..shellblocks << 4 {
            let value = excitation[i];
            excitation[i] = value.wrapping_mul(256) | SILK_QUANT_OFFSET[voiced][qoffset_high] as i32;
            if value < 0 {
                excitation[i] += 20;
            } else if value > 0 {
                excitation[i] -= 20;
            }
            seed = 196314165u32.wrapping_mul(seed).wrapping_add(907633515);
            if seed & 0x8000_0000 != 0 {
                excitation[i] *= -1;
            }
            seed = seed.wrapping_add(value as u32);
            out[i] = excitation[i] as f32 / 8388608.0f32;
        }
    }

    fn decode_frame(&mut self, rc: &mut Rc, frame_num: usize, channel: usize, coded_channels: usize, active: usize, active1: usize, redundant: bool) {
        let mut lpc_leadin = [0f32; 16];
        let mut lpc_body = [0f32; 16];
        let mut residual = [0f32; MAX_LAG + HISTORY];
        #[derive(Clone, Copy, Default)]
        struct Sf {
            gain: f32,
            pitchlag: i32,
            ltptaps: [f32; 5],
        }
        let mut sf = [Sf::default(); 4];

        if coded_channels == 2 && channel == 0 {
            let n = rc.dec_cdf(&SILK_MODEL_STEREO_S1) as i32;
            let mut wi = [0i32; 2];
            let mut ws = [0i32; 2];
            wi[0] = rc.dec_cdf(&SILK_MODEL_STEREO_S2) as i32 + 3 * (n / 5);
            ws[0] = rc.dec_cdf(&SILK_MODEL_STEREO_S3) as i32;
            wi[1] = rc.dec_cdf(&SILK_MODEL_STEREO_S2) as i32 + 3 * (n % 5);
            ws[1] = rc.dec_cdf(&SILK_MODEL_STEREO_S3) as i32;
            let mut w = [0i32; 2];
            for i in 0..2 {
                let a = SILK_STEREO_WEIGHTS[wi[i] as usize] as i32;
                let b = SILK_STEREO_WEIGHTS[wi[i] as usize + 1] as i32;
                w[i] = a + (((b - a) * 6554) >> 16) * (ws[i] * 2 + 1);
            }
            self.stereo_weights[0] = ((w[0] - w[1]) as f64 / 8192.0) as f32;
            self.stereo_weights[1] = (w[1] as f64 / 8192.0) as f32;
            self.midonly = if active1 != 0 { false } else { rc.dec_cdf(&SILK_MODEL_MID_ONLY) != 0 };
        }

        let (qoffset_high, voiced);
        if active == 0 {
            qoffset_high = rc.dec_cdf(&SILK_MODEL_FRAME_TYPE_INACTIVE) as usize;
            voiced = 0usize;
        } else {
            let t = rc.dec_cdf(&SILK_MODEL_FRAME_TYPE_ACTIVE) as usize;
            qoffset_high = t & 1;
            voiced = t >> 1;
        }

        for i in 0..self.subframes {
            let frame = &mut self.frame[channel];
            let mut log_gain;
            if i == 0 && (frame_num == 0 || !frame.coded) {
                let x = rc.dec_cdf(&SILK_MODEL_GAIN_HIGHBITS[active + voiced]) as i32;
                log_gain = (x << 3) | rc.dec_cdf(&SILK_MODEL_GAIN_LOWBITS) as i32;
                if frame.coded {
                    log_gain = log_gain.max(frame.log_gain - 16);
                }
            } else {
                let delta_gain = rc.dec_cdf(&SILK_MODEL_GAIN_DELTA) as i32;
                log_gain = ((delta_gain << 1) - 16).max(frame.log_gain + delta_gain - 4).clamp(0, 63);
            }
            frame.log_gain = log_gain;
            let lg = ((log_gain * 0x1D1C71) >> 16) + 2090;
            let ipart = lg >> 7;
            let fpart = lg & 127;
            let lingain = (1i32 << ipart) + (((-174 * fpart * (128 - fpart)) >> 16) + fpart) * ((1i32 << ipart) >> 7);
            sf[i].gain = lingain as f32 / 65536.0f32;
        }

        let (order, has_lpc_leadin) = self.decode_lpc(channel, rc, &mut lpc_leadin, &mut lpc_body, voiced);

        if voiced != 0 {
            let bw = self.bandwidth;
            let frame = &mut self.frame[channel];
            let mut lag_absolute = frame_num == 0 || !frame.prev_voiced;
            let mut primarylag = 0;
            if !lag_absolute {
                let delta = rc.dec_cdf(&SILK_MODEL_PITCH_DELTA) as i32;
                if delta != 0 {
                    primarylag = frame.primarylag + delta - 9;
                } else {
                    lag_absolute = true;
                }
            }
            if lag_absolute {
                let highbits = rc.dec_cdf(&SILK_MODEL_PITCH_HIGHBITS) as i32;
                let lowbits = match bw {
                    0 => rc.dec_cdf(&SILK_MODEL_PITCH_LOWBITS_NB),
                    1 => rc.dec_cdf(&SILK_MODEL_PITCH_LOWBITS_MB),
                    _ => rc.dec_cdf(&SILK_MODEL_PITCH_LOWBITS_WB),
                } as i32;
                primarylag = SILK_PITCH_MIN_LAG[bw] as i32 + highbits * SILK_PITCH_SCALE[bw] as i32 + lowbits;
            }
            frame.primarylag = primarylag;
            let offsets: &[i8] = if self.subframes == 2 {
                if bw == 0 {
                    &SILK_PITCH_OFFSET_NB10MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_NB10MS) as usize]
                } else {
                    &SILK_PITCH_OFFSET_MBWB10MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_MBWB10MS) as usize]
                }
            } else if bw == 0 {
                &SILK_PITCH_OFFSET_NB20MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_NB20MS) as usize]
            } else {
                &SILK_PITCH_OFFSET_MBWB20MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_MBWB20MS) as usize]
            };
            for i in 0..self.subframes {
                sf[i].pitchlag = (primarylag + offsets[i] as i32).clamp(SILK_PITCH_MIN_LAG[bw] as i32, SILK_PITCH_MAX_LAG[bw] as i32);
            }
            let ltpfilter = rc.dec_cdf(&SILK_MODEL_LTP_FILTER) as usize;
            for i in 0..self.subframes {
                let (sel, taps): (&[u16], &[[i8; 5]]) = match ltpfilter {
                    0 => (&SILK_MODEL_LTP_FILTER0_SEL, &SILK_LTP_FILTER0_TAPS),
                    1 => (&SILK_MODEL_LTP_FILTER1_SEL, &SILK_LTP_FILTER1_TAPS),
                    _ => (&SILK_MODEL_LTP_FILTER2_SEL, &SILK_LTP_FILTER2_TAPS),
                };
                let index = rc.dec_cdf(sel) as usize;
                for j in 0..5 {
                    sf[i].ltptaps[j] = taps[index][j] as f32 / 128.0f32;
                }
            }
        }

        let ltpscale = if voiced != 0 && frame_num == 0 {
            SILK_LTP_SCALE_FACTOR[rc.dec_cdf(&SILK_MODEL_LTP_SCALE_INDEX) as usize] as f32 / 16384.0f32
        } else {
            15565.0f32 / 16384.0f32
        };

        self.decode_excitation(rc, &mut residual[MAX_LAG..], qoffset_high, active, voiced);

        if self.output_channels == channel || redundant {
            return;
        }

        let (sflength, flength) = (self.sflength, self.flength);
        let interp4 = self.nlsf_interp_factor == 4;
        let frame = &mut self.frame[channel];
        for i in 0..self.subframes {
            let lpc_coeff = if i < 2 && has_lpc_leadin { &lpc_leadin } else { &lpc_body };
            let dst = HISTORY + i * sflength;
            let resp = MAX_LAG + i * sflength;
            let lpc = HISTORY + i * sflength;
            if voiced != 0 {
                let (out_end, scale) =
                    if i < 2 || interp4 { (-((i * sflength) as isize), ltpscale) } else { (-(((i - 2) * sflength) as isize), 1.0f32) };
                let mut j = -(sf[i].pitchlag as isize) - (LTP_ORDER / 2) as isize;
                while j < out_end {
                    let at = (dst as isize + j) as usize;
                    let mut sum = frame.output[at];
                    // dst[j - k - 1] for k = 0..order, in that order.
                    for (c, x) in lpc_coeff[..order].iter().zip(frame.output[at - order..at].iter().rev()) {
                        sum -= c * x;
                    }
                    residual[(resp as isize + j) as usize] = sum.clamp(-1.0, 1.0) * scale / sf[i].gain;
                    j += 1;
                }
                if out_end != 0 {
                    let rescale = sf[i - 1].gain / sf[i].gain;
                    for j in out_end..0 {
                        residual[(resp as isize + j) as usize] *= rescale;
                    }
                }
                let (lag, taps) = (sf[i].pitchlag as usize, sf[i].ltptaps);
                for j in 0..sflength {
                    let at = resp + j;
                    let mut sum = residual[at];
                    // resptr[j - lag + 2 - k] for k = 0..5, in that order.
                    let w = &residual[at + LTP_ORDER / 2 - lag - (LTP_ORDER - 1)..=at + LTP_ORDER / 2 - lag];
                    for (t, x) in taps.iter().zip(w.iter().rev()) {
                        sum += t * x;
                    }
                    residual[at] = sum;
                }
            }
            let gain = sf[i].gain;
            for j in 0..sflength {
                let at = lpc + j;
                let mut sum = residual[resp + j] * gain;
                // lpc[j - k] for k = 1..=order, in that order.
                for (c, x) in lpc_coeff[..order].iter().zip(frame.lpc_history[at - order..at].iter().rev()) {
                    sum += c * x;
                }
                frame.lpc_history[at] = sum;
                frame.output[dst + j] = sum.clamp(-1.0, 1.0);
            }
        }
        frame.prev_voiced = voiced != 0;
        frame.lpc_history.copy_within(flength..flength + HISTORY, 0);
        frame.output.copy_within(flength..flength + HISTORY, 0);
        frame.coded = true;
    }

    fn unmix_ms(&mut self, l: &mut [f32], r: &mut [f32]) {
        let fl = self.flength;
        let mid = &self.frame[0].output[HISTORY - fl - 2..];
        let side = &self.frame[1].output[HISTORY - fl - 2..];
        // mid[i] -> mid[i + 2] in these slices.
        let (w0p, w1p) = (self.prev_stereo_weights[0], self.prev_stereo_weights[1]);
        let (w0, w1) = (self.stereo_weights[0], self.stereo_weights[1]);
        let n1 = SILK_STEREO_INTERP_LEN[self.bandwidth] as usize;
        let clip = |v: f32| v.clamp(-1.0, 1.0);
        for i in 0..fl {
            let (m2, m1, m0, s1) = (mid[i], mid[i + 1], mid[i + 2], side[i + 1]);
            let p0 = (0.25f64 * (m2 + 2.0 * m1 + m0) as f64) as f32;
            let (i0, i1) = if i < n1 {
                (w0p + i as f32 * (w0 - w0p) / n1 as f32, w1p + i as f32 * (w1 - w1p) / n1 as f32)
            } else {
                (w0, w1)
            };
            l[i] = clip((1.0 + i1) * m1 + s1 + i0 * p0);
            r[i] = clip((1.0 - i1) * m1 - s1 - i0 * p0);
        }
        self.prev_stereo_weights = self.stereo_weights;
    }

    /// `ff_silk_decode_superframe`. Returns the samples per channel written.
    pub fn decode_superframe(&mut self, rc: &mut Rc, out: &mut [Vec<f32>; 2], bandwidth: usize, coded_channels: usize, duration_ms: usize) -> Result<usize, String> {
        if bandwidth > 2 || coded_channels > 2 || duration_ms > 60 {
            return Err("Invalid parameters passed to the SILK decoder.".into());
        }
        let nb_frames = 1 + (duration_ms > 20) as usize + (duration_ms > 40) as usize;
        self.subframes = duration_ms / nb_frames / 5;
        self.sflength = 20 * (bandwidth + 2);
        self.flength = self.sflength * self.subframes;
        self.bandwidth = bandwidth;
        self.wb = bandwidth == 2;
        if coded_channels > self.prev_coded_channels {
            self.frame[1].flush();
        }
        self.prev_coded_channels = coded_channels;
        let mut active = [[0usize; 6]; 2];
        let mut redundancy = [0u32; 2];
        for i in 0..coded_channels {
            for j in 0..nb_frames {
                active[i][j] = rc.dec_log(1) as usize;
            }
            redundancy[i] = rc.dec_log(1);
        }
        for i in 0..coded_channels {
            if redundancy[i] != 0 && duration_ms > 20 {
                redundancy[i] = rc.dec_cdf(if duration_ms == 40 { &SILK_MODEL_LBRR_FLAGS_40 } else { &SILK_MODEL_LBRR_FLAGS_60 });
            }
        }
        for i in 0..nb_frames {
            for j in 0..coded_channels {
                if redundancy[j] & (1 << i) != 0 {
                    let active1 = if j == 0 && redundancy[1] & (1 << i) == 0 { 0 } else { 1 };
                    self.decode_frame(rc, i, j, coded_channels, 1, active1, true);
                }
            }
            self.midonly = false;
        }
        let fl = self.flength;
        for c in out.iter_mut() {
            if c.len() < nb_frames * fl {
                c.resize(nb_frames * fl, 0.0);
            }
        }
        for i in 0..nb_frames {
            let mut j = 0;
            while j < coded_channels && !self.midonly {
                self.decode_frame(rc, i, j, coded_channels, active[j][i], active[1][i], false);
                j += 1;
            }
            if self.midonly && self.frame[1].coded {
                self.frame[1].flush();
            }
            if coded_channels == 1 || self.output_channels == 1 {
                for j in 0..self.output_channels {
                    out[j][i * fl..(i + 1) * fl].copy_from_slice(&self.frame[0].output[HISTORY - fl - 2..HISTORY - 2]);
                }
            } else {
                let (a, b) = out.split_at_mut(1);
                self.unmix_ms(&mut a[0][i * fl..(i + 1) * fl], &mut b[0][i * fl..(i + 1) * fl]);
            }
            self.midonly = false;
        }
        Ok(nb_frames * fl)
    }
}
