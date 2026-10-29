//! Spectral Band Replication: ffmpeg n7.1's `aacsbr.c` / `aacsbr_template.c`
//! (float), with the aarch64 NEON `sbrdsp` kernels and the av_tx 64-point
//! inverse MDCTs of the QMF banks. The C is ported in its own operation
//! order (the reference build does not contract it); the NEON kernels follow
//! their lane layout and fused multiply-adds instruction by instruction.

#![allow(clippy::needless_range_loop, clippy::excessive_precision, clippy::type_complexity)]

use super::bits::BitReader;
use super::dsp::Imdct;
use super::ps::Ps;
use super::sbr_tables as t;

const ENV_OFF: usize = 2; // ENVELOPE_ADJUSTMENT_OFFSET
const NOISE_FLOOR_OFFSET: i32 = 6;
const SYNTH_BUF: usize = (1280 - 128) * 2;

const FIXFIX: u32 = 0;
const FIXVAR: u32 = 1;
const VARFIX: u32 = 2;

pub(crate) const TYPE_SCE: u32 = 0;
pub(crate) const TYPE_CPE: u32 = 1;

#[inline]
fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

// ---- VLCs from code lengths (`ff_vlc_init_tables_from_lengths`) ------------

pub(crate) struct LenVlc {
    /// `(len, code, symbol)` in table order.
    codes: Vec<(u32, u32, i32)>,
    max_len: u32,
}

impl LenVlc {
    pub(crate) fn new(syms: &[u8], lens: &[u8], offset: i32) -> Self {
        let mut code: u64 = 0;
        let mut codes = Vec::with_capacity(syms.len());
        for (&s, &l) in syms.iter().zip(lens) {
            let l = l as u32;
            codes.push((l, (code >> (32 - l)) as u32, s as i32 + offset));
            code += 1u64 << (32 - l);
        }
        let max_len = codes.iter().map(|c| c.0).max().unwrap_or(1);
        Self { codes, max_len }
    }

    /// `get_vlc2`; 0 for an invalid code (ffmpeg would return -1 and the
    /// value checks would then fail; real streams never get there).
    pub(crate) fn read(&self, br: &mut BitReader) -> i32 {
        let bits = br.show(self.max_len);
        for &(l, c, s) in &self.codes {
            if bits >> (self.max_len - l) == c {
                br.skip(l);
                return s;
            }
        }
        0
    }
}

fn sbr_vlcs() -> &'static [LenVlc] {
    static V: std::sync::OnceLock<Vec<LenVlc>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let mut at = 0;
        (0..10)
            .map(|i| {
                let n = t::SBR_HUFF_COUNTS[i] as usize;
                let v = LenVlc::new(&t::SBR_HUFF_SYM[at..at + n], &t::SBR_HUFF_LEN[at..at + n], t::SBR_HUFF_OFFSETS[i] as i32);
                at += n;
                v
            })
            .collect()
    })
}

const T_ENV_1_5: usize = 0;
const F_ENV_1_5: usize = 1;
const T_ENV_BAL_1_5: usize = 2;
const F_ENV_BAL_1_5: usize = 3;
const T_ENV_3_0: usize = 4;
const F_ENV_3_0: usize = 5;
const T_ENV_BAL_3_0: usize = 6;
const F_ENV_BAL_3_0: usize = 7;
const T_NOISE_3_0: usize = 8;
const T_NOISE_BAL_3_0: usize = 9;

// ---- state -------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct SbrData {
    bs_frame_class: u32,
    bs_add_harmonic_flag: bool,
    bs_num_env: usize,
    bs_freq_res: [u8; 9],
    bs_num_noise: usize,
    bs_df_env: [u8; 9],
    bs_df_noise: [u8; 2],
    bs_invf_mode: [[u8; 5]; 2],
    bs_add_harmonic: [u8; 48],
    bs_amp_res: u32,
    synth: Vec<f32>,
    analysis: Vec<f32>,
    synth_offset: usize,
    e_a: [i32; 2],
    bw_array: [f32; 5],
    /// `W[2][32][32][2]`.
    w: Vec<f32>,
    ypos: usize,
    /// `Y[2][38][64][2]`.
    y: Vec<f32>,
    g_temp: Vec<[f32; 48]>,
    q_temp: Vec<[f32; 48]>,
    s_indexmapped: [[u8; 48]; 9],
    env_facs_q: [[u8; 48]; 9],
    env_facs: [[f32; 48]; 9],
    noise_facs_q: [[u8; 5]; 3],
    noise_facs: [[f32; 5]; 3],
    t_env: [u8; 9],
    t_env_num_env_old: u8,
    t_q: [u8; 3],
    f_indexnoise: u32,
    f_indexsine: u32,
}

impl Default for SbrData {
    fn default() -> Self {
        Self {
            bs_frame_class: 0,
            bs_add_harmonic_flag: false,
            bs_num_env: 0,
            bs_freq_res: [0; 9],
            bs_num_noise: 0,
            bs_df_env: [0; 9],
            bs_df_noise: [0; 2],
            bs_invf_mode: [[0; 5]; 2],
            bs_add_harmonic: [0; 48],
            bs_amp_res: 0,
            synth: vec![0.0; SYNTH_BUF],
            analysis: vec![0.0; 1312],
            synth_offset: SYNTH_BUF - (1280 - 128),
            e_a: [0, 0],
            bw_array: [0.0; 5],
            w: vec![0.0; 2 * 32 * 32 * 2],
            ypos: 0,
            y: vec![0.0; 2 * 38 * 64 * 2],
            g_temp: vec![[0.0; 48]; 42],
            q_temp: vec![[0.0; 48]; 42],
            s_indexmapped: [[0; 48]; 9],
            env_facs_q: [[0; 48]; 9],
            env_facs: [[0.0; 48]; 9],
            noise_facs_q: [[0; 5]; 3],
            noise_facs: [[0.0; 5]; 3],
            t_env: [0; 9],
            t_env_num_env_old: 0,
            t_q: [0; 3],
            f_indexnoise: 0,
            f_indexsine: 0,
        }
    }
}

/// `SpectrumParameters`: start, stop, xover, freq_scale, alter_scale,
/// noise_bands.
type Spectrum = [u8; 6];

pub(crate) struct Sbr {
    pub(crate) sample_rate: i32,
    start: bool,
    ready_for_dequant: bool,
    id_aac: u32,
    reset: bool,
    spectrum: Spectrum,
    bs_amp_res_header: u32,
    bs_limiter_bands: u32,
    bs_limiter_gains: u32,
    bs_interpol_freq: u32,
    bs_smoothing_mode: u32,
    bs_coupling: bool,
    k: [i32; 5],
    kx: [i32; 2],
    m: [i32; 2],
    kx_and_m_pushed: bool,
    n_master: i32,
    data: [SbrData; 2],
    pub(crate) ps: Ps,
    n: [i32; 2],
    n_q: i32,
    n_lim: i32,
    f_master: [u16; 49],
    f_tablelow: [u16; 25],
    f_tablehigh: [u16; 49],
    f_tablenoise: [u16; 6],
    f_tablelim: [u16; 30],
    num_patches: i32,
    patch_num_subbands: [u8; 6],
    patch_start_subband: [u8; 6],
    /// `X_low[32][40][2]`.
    x_low: Vec<f32>,
    /// `X_high[64][40][2]`.
    x_high: Vec<f32>,
    /// `X[2][2][38][64]`.
    x: Vec<f32>,
    alpha0: [[f32; 2]; 64],
    alpha1: [[f32; 2]; 64],
    e_origmapped: [[f32; 48]; 8],
    q_mapped: [[f32; 48]; 8],
    s_mapped: [[u8; 48]; 8],
    e_curr: [[f32; 48]; 8],
    q_m: [[f32; 48]; 8],
    s_m: [[f32; 48]; 8],
    gain: [[f32; 48]; 8],
    mdct: Imdct,
    mdct_ana: Imdct,
}

#[inline]
fn xl(k: usize, i: usize) -> usize {
    (k * 40 + i) * 2
}
#[inline]
fn yi(buf: usize, i: usize, k: usize) -> usize {
    ((buf * 38 + i) * 64 + k) * 2
}
#[inline]
fn wi(buf: usize, i: usize, k: usize) -> usize {
    ((buf * 32 + i) * 32 + k) * 2
}
#[inline]
fn xi(ch: usize, ri: usize, i: usize, k: usize) -> usize {
    ((ch * 2 + ri) * 38 + i) * 64 + k
}

/// `exp2fi`: 2^x for integer x, correctly rounded to float.
fn exp2fi(x: i32) -> f32 {
    if (-126..=128).contains(&x) {
        f32::from_bits(((x + 127) as u32) << 23)
    } else if x > 128 {
        f32::INFINITY
    } else if x > -150 {
        f32::from_bits(1u32 << (x + 149))
    } else {
        0.0
    }
}

/// `make_bands` (powf / lrintf in float).
fn make_bands(bands: &mut [i16], start: i32, stop: i32, num_bands: i32) {
    let base = (stop as f32 / start as f32).powf(1.0 / num_bands as f32);
    let mut prod = start as f32;
    let mut previous = start;
    for k in 0..(num_bands - 1) as usize {
        prod *= base;
        let present = prod.round_ties_even() as i32;
        bands[k] = (present - previous) as i16;
        previous = present;
    }
    bands[(num_bands - 1) as usize] = (stop - previous) as i16;
}

fn log2f_lrint(x: f32) -> i32 {
    x.round_ties_even() as i32
}

impl Sbr {
    /// `ff_aac_sbr_ctx_alloc_init` (zeroed context) + `sbr_turnoff`.
    pub(crate) fn new(id_aac: u32) -> Self {
        let mut s = Sbr {
            sample_rate: 0,
            start: false,
            ready_for_dequant: false,
            id_aac,
            reset: false,
            spectrum: [0; 6],
            bs_amp_res_header: 0,
            bs_limiter_bands: 0,
            bs_limiter_gains: 0,
            bs_interpol_freq: 0,
            bs_smoothing_mode: 0,
            bs_coupling: false,
            k: [0; 5],
            kx: [0; 2],
            m: [0; 2],
            kx_and_m_pushed: false,
            n_master: 0,
            data: [SbrData::default(), SbrData::default()],
            ps: Ps::default(),
            n: [0; 2],
            n_q: 0,
            n_lim: 0,
            f_master: [0; 49],
            f_tablelow: [0; 25],
            f_tablehigh: [0; 49],
            f_tablenoise: [0; 6],
            f_tablelim: [0; 30],
            num_patches: 0,
            patch_num_subbands: [0; 6],
            patch_start_subband: [0; 6],
            x_low: vec![0.0; 32 * 40 * 2],
            x_high: vec![0.0; 64 * 40 * 2],
            x: vec![0.0; 2 * 2 * 38 * 64],
            alpha0: [[0.0; 2]; 64],
            alpha1: [[0.0; 2]; 64],
            e_origmapped: [[0.0; 48]; 8],
            q_mapped: [[0.0; 48]; 8],
            s_mapped: [[0; 48]; 8],
            e_curr: [[0.0; 48]; 8],
            q_m: [[0.0; 48]; 8],
            s_m: [[0.0; 48]; 8],
            gain: [[0.0; 48]; 8],
            mdct: Imdct::new(64, 1.0 / (64.0 * 32768.0)),
            mdct_ana: Imdct::new(64, -2.0 * 32768.0),
        };
        s.kx[0] = s.kx[1];
        s.turnoff();
        s
    }

    /// `sbr_turnoff`: pure upsampling.
    fn turnoff(&mut self) {
        self.start = false;
        self.ready_for_dequant = false;
        self.kx[1] = 32;
        self.m[1] = 0;
        self.data[0].e_a[1] = -1;
        self.data[1].e_a[1] = -1;
        self.spectrum = [0xff; 6];
    }

    // ---- frequency tables ------------------------------------------------------

    /// `sbr_make_f_tablelim`.
    fn make_f_tablelim(&mut self) {
        if self.bs_limiter_bands > 0 {
            #[allow(clippy::excessive_precision)]
            const WARPED: [f32; 3] = [1.32715174233856803909, 1.18509277094158210129, 1.11987160404675912501];
            let lim = WARPED[self.bs_limiter_bands as usize - 1];
            let np = self.num_patches as usize;
            let mut borders = [0i16; 7];
            borders[0] = self.kx[1] as i16;
            for k in 1..=np {
                borders[k] = borders[k - 1] + self.patch_num_subbands[k - 1] as i16;
            }
            let n0 = self.n[0] as usize;
            self.f_tablelim[..n0 + 1].copy_from_slice(&self.f_tablelow[..n0 + 1]);
            if np > 1 {
                for k in 0..np - 1 {
                    self.f_tablelim[n0 + 1 + k] = borders[1 + k] as u16;
                }
            }
            self.f_tablelim[..np + n0].sort_by_key(|&v| v as i16);
            self.n_lim = self.n[0] + self.num_patches - 1;
            let in_table = |v: u16| borders[..=np].contains(&(v as i16));
            let (mut out, mut inp) = (0usize, 1usize);
            while out < self.n_lim as usize {
                let (iv, ov) = (self.f_tablelim[inp], self.f_tablelim[out]);
                if iv as f32 >= ov as f32 * lim {
                    out += 1;
                    self.f_tablelim[out] = iv;
                    inp += 1;
                } else if iv == ov || !in_table(iv) {
                    inp += 1;
                    self.n_lim -= 1;
                } else if !in_table(ov) {
                    self.f_tablelim[out] = iv;
                    inp += 1;
                    self.n_lim -= 1;
                } else {
                    out += 1;
                    self.f_tablelim[out] = iv;
                    inp += 1;
                }
            }
        } else {
            self.f_tablelim[0] = self.f_tablelow[0];
            self.f_tablelim[1] = self.f_tablelow[self.n[0] as usize];
            self.n_lim = 1;
        }
    }

    /// `read_sbr_header`.
    fn read_header(&mut self, gb: &mut BitReader) -> u32 {
        let cnt = gb.pos();
        let old_limiter_bands = self.bs_limiter_bands;
        self.start = true;
        self.ready_for_dequant = false;
        let old = self.spectrum;
        self.bs_amp_res_header = gb.bits(1);
        self.spectrum[0] = gb.bits(4) as u8;
        self.spectrum[1] = gb.bits(4) as u8;
        self.spectrum[2] = gb.bits(3) as u8;
        gb.skip(2);
        let extra1 = gb.bit();
        let extra2 = gb.bit();
        if extra1 {
            self.spectrum[3] = gb.bits(2) as u8;
            self.spectrum[4] = gb.bits(1) as u8;
            self.spectrum[5] = gb.bits(2) as u8;
        } else {
            self.spectrum[3] = 2;
            self.spectrum[4] = 1;
            self.spectrum[5] = 2;
        }
        if old != self.spectrum {
            self.reset = true;
        }
        if extra2 {
            self.bs_limiter_bands = gb.bits(2);
            self.bs_limiter_gains = gb.bits(2);
            self.bs_interpol_freq = gb.bits(1);
            self.bs_smoothing_mode = gb.bits(1);
        } else {
            self.bs_limiter_bands = 2;
            self.bs_limiter_gains = 2;
            self.bs_interpol_freq = 1;
            self.bs_smoothing_mode = 1;
        }
        if self.bs_limiter_bands != old_limiter_bands && !self.reset {
            self.make_f_tablelim();
        }
        (gb.pos() - cnt) as u32
    }

    /// `sbr_make_f_master`.
    fn make_f_master(&mut self) -> Result<(), ()> {
        let sp = self.spectrum;
        let off_row = match self.sample_rate {
            16000 => 0,
            22050 => 1,
            24000 => 2,
            32000 => 3,
            44100 | 48000 | 64000 => 4,
            88200 | 96000 | 128000 | 176400 | 192000 => 5,
            _ => return Err(()),
        };
        let offs = &t::SBR_OFFSET[off_row * 16..off_row * 16 + 16];
        let sr = self.sample_rate as u32;
        let temp: u32 = if sr < 32000 { 3000 } else if sr < 64000 { 4000 } else { 5000 };
        let start_min = ((temp << 7) + (sr >> 1)) / sr;
        let stop_min = ((temp << 8) + (sr >> 1)) / sr;
        self.k[0] = start_min as i32 + offs[sp[0] as usize] as i32;
        if sp[1] < 14 {
            self.k[2] = stop_min as i32;
            let mut stop_dk = [0i16; 13];
            make_bands(&mut stop_dk, stop_min as i32, 64, 13);
            stop_dk.sort();
            for k in 0..sp[1] as usize {
                self.k[2] += stop_dk[k] as i32;
            }
        } else if sp[1] == 14 {
            self.k[2] = 2 * self.k[0];
        } else if sp[1] == 15 {
            self.k[2] = 3 * self.k[0];
        } else {
            return Err(());
        }
        self.k[2] = self.k[2].min(64);
        let max_qmf: i32 = if sr <= 32000 {
            48
        } else if sr == 44100 {
            35
        } else if sr >= 48000 {
            32
        } else {
            return Err(());
        };
        if self.k[2] - self.k[0] > max_qmf {
            return Err(());
        }
        let xover = sp[2] as i32;
        let check = |n_master: i32| if n_master <= 0 || xover >= n_master { Err(()) } else { Ok(()) };
        if sp[3] == 0 {
            let dk = sp[4] as i32 + 1;
            self.n_master = ((self.k[2] - self.k[0] + (dk & 2)) >> dk) << 1;
            check(self.n_master)?;
            for k in 1..=self.n_master as usize {
                self.f_master[k] = dk as u16;
            }
            let k2diff = self.k[2] - self.k[0] - self.n_master * dk;
            if k2diff < 0 {
                self.f_master[1] = self.f_master[1].wrapping_sub(1);
                self.f_master[2] = self.f_master[2].wrapping_sub((k2diff < -1) as u16);
            } else if k2diff != 0 {
                let nm = self.n_master as usize;
                self.f_master[nm] += 1;
            }
            self.f_master[0] = self.k[0] as u16;
            for k in 1..=self.n_master as usize {
                self.f_master[k] = self.f_master[k].wrapping_add(self.f_master[k - 1]);
            }
        } else {
            let half_bands = 7 - sp[3] as i32;
            let two_regions;
            if 49 * self.k[2] > 110 * self.k[0] {
                two_regions = true;
                self.k[1] = 2 * self.k[0];
            } else {
                two_regions = false;
                self.k[1] = self.k[2];
            }
            let num_bands_0 = log2f_lrint(half_bands as f32 * (self.k[1] as f32 / self.k[0] as f32).log2()) * 2;
            if num_bands_0 <= 0 {
                return Err(());
            }
            let mut vk0 = [0i16; 49];
            make_bands(&mut vk0[1..], self.k[0], self.k[1], num_bands_0);
            vk0[1..=num_bands_0 as usize].sort();
            let vdk0_max = vk0[num_bands_0 as usize] as i32;
            vk0[0] = self.k[0] as i16;
            for k in 1..=num_bands_0 as usize {
                if vk0[k] <= 0 {
                    return Err(());
                }
                vk0[k] += vk0[k - 1];
            }
            if two_regions {
                let mut vk1 = [0i16; 49];
                #[allow(clippy::excessive_precision)]
                let invwarp: f32 = if sp[4] != 0 { 0.76923076923076923077 } else { 1.0 };
                let num_bands_1 = log2f_lrint(half_bands as f32 * invwarp * (self.k[2] as f32 / self.k[1] as f32).log2()) * 2;
                make_bands(&mut vk1[1..], self.k[1], self.k[2], num_bands_1);
                let nb1 = num_bands_1 as usize;
                let vdk1_min = *vk1[1..=nb1].iter().min().unwrap() as i32;
                if vdk1_min < vdk0_max {
                    vk1[1..=nb1].sort();
                    let change = (vdk0_max - vk1[1] as i32).min((vk1[nb1] as i32 - vk1[1] as i32) >> 1);
                    vk1[1] += change as i16;
                    vk1[nb1] -= change as i16;
                }
                vk1[1..=nb1].sort();
                vk1[0] = self.k[1] as i16;
                for k in 1..=nb1 {
                    if vk1[k] <= 0 {
                        return Err(());
                    }
                    vk1[k] += vk1[k - 1];
                }
                self.n_master = num_bands_0 + num_bands_1;
                check(self.n_master)?;
                for k in 0..=num_bands_0 as usize {
                    self.f_master[k] = vk0[k] as u16;
                }
                for k in 0..nb1 {
                    self.f_master[num_bands_0 as usize + 1 + k] = vk1[1 + k] as u16;
                }
            } else {
                self.n_master = num_bands_0;
                check(self.n_master)?;
                for k in 0..=num_bands_0 as usize {
                    self.f_master[k] = vk0[k] as u16;
                }
            }
        }
        Ok(())
    }

    /// `sbr_hf_calc_npatches`.
    fn calc_npatches(&mut self) -> Result<(), ()> {
        let (mut last_k, mut last_msb, mut sb) = (-1i32, -1i32, 0i32);
        let mut msb = self.k[0];
        let mut usb = self.kx[1];
        let sr = self.sample_rate;
        let goal_sb = ((1000 << 11) + (sr >> 1)) / sr;
        self.num_patches = 0;
        let mut k: i32 = if goal_sb < self.kx[1] + self.m[1] {
            let mut k = 0;
            while (self.f_master[k as usize] as i32) < goal_sb {
                k += 1;
            }
            k
        } else {
            self.n_master
        };
        loop {
            let mut odd = 0;
            if k == last_k && msb == last_msb {
                return Err(());
            }
            last_k = k;
            last_msb = msb;
            let mut i = k;
            while i == k || sb > self.k[0] - 1 + msb - odd {
                sb = self.f_master[i as usize] as i32;
                odd = (sb + self.k[0]) & 1;
                i -= 1;
            }
            if self.num_patches > 5 {
                return Err(());
            }
            let np = self.num_patches as usize;
            self.patch_num_subbands[np] = (sb - usb).max(0) as u8;
            self.patch_start_subband[np] = (self.k[0] - odd - self.patch_num_subbands[np] as i32) as u8;
            if self.patch_num_subbands[np] > 0 {
                usb = sb;
                msb = sb;
                self.num_patches += 1;
            } else {
                msb = self.kx[1];
            }
            if self.f_master[k as usize] as i32 - sb < 3 {
                k = self.n_master;
            }
            if sb == self.kx[1] + self.m[1] {
                break;
            }
        }
        if self.num_patches > 1 && self.patch_num_subbands[self.num_patches as usize - 1] < 3 {
            self.num_patches -= 1;
        }
        Ok(())
    }

    /// `sbr_make_f_derived`.
    fn make_f_derived(&mut self) -> Result<(), ()> {
        let xover = self.spectrum[2] as usize;
        self.n[1] = self.n_master - xover as i32;
        self.n[0] = (self.n[1] + 1) >> 1;
        let n1 = self.n[1] as usize;
        for k in 0..=n1 {
            self.f_tablehigh[k] = self.f_master[xover + k];
        }
        self.m[1] = self.f_tablehigh[n1] as i32 - self.f_tablehigh[0] as i32;
        self.kx[1] = self.f_tablehigh[0] as i32;
        if self.kx[1] + self.m[1] > 64 || self.kx[1] > 32 {
            return Err(());
        }
        self.f_tablelow[0] = self.f_tablehigh[0];
        let temp = (self.n[1] & 1) as usize;
        for k in 1..=self.n[0] as usize {
            self.f_tablelow[k] = self.f_tablehigh[2 * k - temp];
        }
        self.n_q = log2f_lrint(self.spectrum[5] as f32 * (self.k[2] as f32 / self.kx[1] as f32).log2()).max(1);
        if self.n_q > 5 {
            return Err(());
        }
        self.f_tablenoise[0] = self.f_tablelow[0];
        let mut temp = 0i32;
        for k in 1..=self.n_q {
            temp += (self.n[0] - temp) / (self.n_q + 1 - k);
            self.f_tablenoise[k as usize] = self.f_tablelow[temp as usize];
        }
        self.calc_npatches()?;
        self.make_f_tablelim();
        self.data[0].f_indexnoise = 0;
        self.data[1].f_indexnoise = 0;
        Ok(())
    }

    fn sbr_reset(&mut self) {
        if self.make_f_master().and_then(|_| self.make_f_derived()).is_err() {
            self.turnoff();
        }
    }

    // ---- bitstream -----------------------------------------------------------

    /// `read_sbr_grid`.
    fn read_grid(&mut self, gb: &mut BitReader, ch: usize) -> Result<(), ()> {
        const CEIL_LOG2: [u32; 6] = [0, 1, 2, 2, 3, 3];
        let amp_res_header = self.bs_amp_res_header;
        let d = &mut self.data[ch];
        let mut bs_pointer = 0u32;
        let mut abs_bord_trail = 16i32;
        let bs_num_env_old = d.bs_num_env;
        d.bs_freq_res[0] = d.bs_freq_res[d.bs_num_env];
        d.bs_amp_res = amp_res_header;
        d.t_env_num_env_old = d.t_env[bs_num_env_old];
        let bs_frame_class = gb.bits(2);
        match bs_frame_class {
            FIXFIX => {
                let n = 1usize << gb.bits(2);
                if n > 5 {
                    return Err(());
                }
                d.bs_num_env = n;
                let num_rel_lead = n - 1;
                if n == 1 {
                    d.bs_amp_res = 0;
                }
                d.t_env[0] = 0;
                d.t_env[n] = abs_bord_trail as u8;
                abs_bord_trail = (abs_bord_trail + (n as i32 >> 1)) / n as i32;
                for i in 0..num_rel_lead {
                    d.t_env[i + 1] = d.t_env[i] + abs_bord_trail as u8;
                }
                d.bs_freq_res[1] = gb.bits(1) as u8;
                for i in 1..n {
                    d.bs_freq_res[i + 1] = d.bs_freq_res[1];
                }
            }
            FIXVAR => {
                abs_bord_trail += gb.bits(2) as i32;
                let num_rel_trail = gb.bits(2) as usize;
                d.bs_num_env = num_rel_trail + 1;
                let n = d.bs_num_env;
                d.t_env[0] = 0;
                d.t_env[n] = abs_bord_trail as u8;
                for i in 0..num_rel_trail {
                    d.t_env[n - 1 - i] = (d.t_env[n - i] as i32 - 2 * gb.bits(2) as i32 - 2) as u8;
                }
                bs_pointer = gb.bits(CEIL_LOG2[n]);
                for i in 0..n {
                    d.bs_freq_res[n - i] = gb.bits(1) as u8;
                }
            }
            VARFIX => {
                d.t_env[0] = gb.bits(2) as u8;
                let num_rel_lead = gb.bits(2) as usize;
                d.bs_num_env = num_rel_lead + 1;
                let n = d.bs_num_env;
                d.t_env[n] = abs_bord_trail as u8;
                for i in 0..num_rel_lead {
                    d.t_env[i + 1] = d.t_env[i] + 2 * gb.bits(2) as u8 + 2;
                }
                bs_pointer = gb.bits(CEIL_LOG2[n]);
                for i in 0..n {
                    d.bs_freq_res[1 + i] = gb.bits(1) as u8;
                }
            }
            _ => {
                d.t_env[0] = gb.bits(2) as u8;
                abs_bord_trail += gb.bits(2) as i32;
                let num_rel_lead = gb.bits(2) as usize;
                let num_rel_trail = gb.bits(2) as usize;
                let n = num_rel_lead + num_rel_trail + 1;
                if n > 5 {
                    return Err(());
                }
                d.bs_num_env = n;
                d.t_env[n] = abs_bord_trail as u8;
                for i in 0..num_rel_lead {
                    d.t_env[i + 1] = d.t_env[i] + 2 * gb.bits(2) as u8 + 2;
                }
                for i in 0..num_rel_trail {
                    d.t_env[n - 1 - i] = (d.t_env[n - i] as i32 - 2 * gb.bits(2) as i32 - 2) as u8;
                }
                bs_pointer = gb.bits(CEIL_LOG2[n]);
                for i in 0..n {
                    d.bs_freq_res[1 + i] = gb.bits(1) as u8;
                }
            }
        }
        d.bs_frame_class = bs_frame_class;
        let n = d.bs_num_env;
        if bs_pointer as usize > n + 1 {
            return Err(());
        }
        for i in 1..=n {
            if d.t_env[i - 1] >= d.t_env[i] {
                return Err(());
            }
        }
        d.bs_num_noise = (n > 1) as usize + 1;
        d.t_q[0] = d.t_env[0];
        d.t_q[d.bs_num_noise] = d.t_env[n];
        if d.bs_num_noise > 1 {
            let idx = if d.bs_frame_class == FIXFIX {
                n >> 1
            } else if d.bs_frame_class & 1 != 0 {
                n - (bs_pointer as i32 - 1).max(1) as usize
            } else if bs_pointer == 0 {
                1
            } else if bs_pointer == 1 {
                n - 1
            } else {
                bs_pointer as usize - 1
            };
            d.t_q[1] = d.t_env[idx];
        }
        d.e_a[0] = -((d.e_a[1] != bs_num_env_old as i32) as i32);
        d.e_a[1] = -1;
        if d.bs_frame_class & 1 != 0 && bs_pointer != 0 {
            d.e_a[1] = n as i32 + 1 - bs_pointer as i32;
        } else if d.bs_frame_class == 2 && bs_pointer > 1 {
            d.e_a[1] = bs_pointer as i32 - 1;
        }
        Ok(())
    }

    /// `copy_sbr_grid` (channel 0 to channel 1).
    fn copy_grid(&mut self) {
        let src = self.data[0].clone();
        let dst = &mut self.data[1];
        dst.bs_freq_res[0] = dst.bs_freq_res[dst.bs_num_env];
        dst.t_env_num_env_old = dst.t_env[dst.bs_num_env];
        dst.e_a[0] = -((dst.e_a[1] != dst.bs_num_env as i32) as i32);
        dst.bs_freq_res[1..].copy_from_slice(&src.bs_freq_res[1..]);
        dst.t_env = src.t_env;
        dst.t_q = src.t_q;
        dst.bs_num_env = src.bs_num_env;
        dst.bs_amp_res = src.bs_amp_res;
        dst.bs_num_noise = src.bs_num_noise;
        dst.bs_frame_class = src.bs_frame_class;
        dst.e_a[1] = src.e_a[1];
    }

    fn read_dtdf(&mut self, gb: &mut BitReader, ch: usize) {
        let d = &mut self.data[ch];
        for i in 0..d.bs_num_env {
            d.bs_df_env[i] = gb.bits(1) as u8;
        }
        for i in 0..d.bs_num_noise {
            d.bs_df_noise[i] = gb.bits(1) as u8;
        }
    }

    fn read_invf(&mut self, gb: &mut BitReader, ch: usize) {
        let n_q = self.n_q as usize;
        let d = &mut self.data[ch];
        d.bs_invf_mode[1] = d.bs_invf_mode[0];
        for i in 0..n_q {
            d.bs_invf_mode[0][i] = gb.bits(2) as u8;
        }
    }

    /// `read_sbr_envelope`.
    fn read_envelope(&mut self, gb: &mut BitReader, ch: usize) -> Result<(), ()> {
        let coupling = self.bs_coupling;
        let delta = (ch == 1 && coupling) as i32 + 1;
        let odd = (self.n[1] & 1) as usize;
        let nres = self.n;
        let v = sbr_vlcs();
        let d = &mut self.data[ch];
        let (bits, t_huff, f_huff) = if coupling && ch == 1 {
            if d.bs_amp_res != 0 {
                (5, T_ENV_BAL_3_0, F_ENV_BAL_3_0)
            } else {
                (6, T_ENV_BAL_1_5, F_ENV_BAL_1_5)
            }
        } else if d.bs_amp_res != 0 {
            (6, T_ENV_3_0, F_ENV_3_0)
        } else {
            (7, T_ENV_1_5, F_ENV_1_5)
        };
        for i in 0..d.bs_num_env {
            let nb = nres[d.bs_freq_res[i + 1] as usize] as usize;
            if d.bs_df_env[i] != 0 {
                for j in 0..nb {
                    let k = if d.bs_freq_res[i + 1] == d.bs_freq_res[i] {
                        j
                    } else if d.bs_freq_res[i + 1] != 0 {
                        (j + odd) >> 1
                    } else if j != 0 {
                        2 * j - odd
                    } else {
                        0
                    };
                    let val = d.env_facs_q[i][k] as i32 + delta * v[t_huff].read(gb);
                    d.env_facs_q[i + 1][j] = val as u8;
                    if val as u8 > 127 {
                        return Err(());
                    }
                }
            } else {
                d.env_facs_q[i + 1][0] = (delta * gb.bits(bits) as i32) as u8;
                for j in 1..nb {
                    let val = d.env_facs_q[i + 1][j - 1] as i32 + delta * v[f_huff].read(gb);
                    d.env_facs_q[i + 1][j] = val as u8;
                    if val as u8 > 127 {
                        return Err(());
                    }
                }
            }
        }
        d.env_facs_q[0] = d.env_facs_q[d.bs_num_env];
        Ok(())
    }

    /// `read_sbr_noise`.
    fn read_noise(&mut self, gb: &mut BitReader, ch: usize) -> Result<(), ()> {
        let coupling = self.bs_coupling;
        let delta = (ch == 1 && coupling) as i32 + 1;
        let n_q = self.n_q as usize;
        let v = sbr_vlcs();
        let (t_huff, f_huff) = if coupling && ch == 1 { (T_NOISE_BAL_3_0, F_ENV_BAL_3_0) } else { (T_NOISE_3_0, F_ENV_3_0) };
        let d = &mut self.data[ch];
        for i in 0..d.bs_num_noise {
            if d.bs_df_noise[i] != 0 {
                for j in 0..n_q {
                    let val = d.noise_facs_q[i][j] as i32 + delta * v[t_huff].read(gb);
                    d.noise_facs_q[i + 1][j] = val as u8;
                    if val as u8 > 30 {
                        return Err(());
                    }
                }
            } else {
                d.noise_facs_q[i + 1][0] = (delta * gb.bits(5) as i32) as u8;
                for j in 1..n_q {
                    let val = d.noise_facs_q[i + 1][j - 1] as i32 + delta * v[f_huff].read(gb);
                    d.noise_facs_q[i + 1][j] = val as u8;
                    if val as u8 > 30 {
                        return Err(());
                    }
                }
            }
        }
        d.noise_facs_q[0] = d.noise_facs_q[d.bs_num_noise];
        Ok(())
    }

    fn read_harmonic(&mut self, gb: &mut BitReader, ch: usize) {
        let n1 = self.n[1] as usize;
        let d = &mut self.data[ch];
        d.bs_add_harmonic_flag = gb.bit();
        if d.bs_add_harmonic_flag {
            for i in 0..n1 {
                d.bs_add_harmonic[i] = gb.bits(1) as u8;
            }
        }
    }

    fn read_sce(&mut self, gb: &mut BitReader) -> Result<(), ()> {
        if gb.bit() {
            gb.skip(4);
        }
        self.read_grid(gb, 0)?;
        self.read_dtdf(gb, 0);
        self.read_invf(gb, 0);
        self.read_envelope(gb, 0)?;
        self.read_noise(gb, 0)?;
        self.read_harmonic(gb, 0);
        Ok(())
    }

    fn read_cpe(&mut self, gb: &mut BitReader) -> Result<(), ()> {
        if gb.bit() {
            gb.skip(8);
        }
        self.bs_coupling = gb.bit();
        if self.bs_coupling {
            self.read_grid(gb, 0)?;
            self.copy_grid();
            self.read_dtdf(gb, 0);
            self.read_dtdf(gb, 1);
            self.read_invf(gb, 0);
            self.data[1].bs_invf_mode[1] = self.data[1].bs_invf_mode[0];
            self.data[1].bs_invf_mode[0] = self.data[0].bs_invf_mode[0];
            self.read_envelope(gb, 0)?;
            self.read_noise(gb, 0)?;
            self.read_envelope(gb, 1)?;
            self.read_noise(gb, 1)?;
        } else {
            self.read_grid(gb, 0)?;
            self.read_grid(gb, 1)?;
            self.read_dtdf(gb, 0);
            self.read_dtdf(gb, 1);
            self.read_invf(gb, 0);
            self.read_invf(gb, 1);
            self.read_envelope(gb, 0)?;
            self.read_envelope(gb, 1)?;
            self.read_noise(gb, 0)?;
            self.read_noise(gb, 1)?;
        }
        self.read_harmonic(gb, 0);
        self.read_harmonic(gb, 1);
        Ok(())
    }

    /// `read_sbr_data`; `ps_allowed` is `m4ac.ps`.
    fn read_data(&mut self, gb: &mut BitReader, id_aac: u32, ps_allowed: i32) -> u32 {
        let cnt = gb.pos();
        self.id_aac = id_aac;
        self.ready_for_dequant = true;
        let r = match id_aac {
            TYPE_SCE => self.read_sce(gb),
            TYPE_CPE => self.read_cpe(gb),
            _ => Err(()),
        };
        if r.is_err() {
            self.turnoff();
            return (gb.pos() - cnt) as u32;
        }
        if gb.bit() {
            let mut left = gb.bits(4) as i32;
            if left == 15 {
                left += gb.bits(8) as i32;
            }
            left <<= 3;
            while left > 7 {
                left -= 2;
                let id = gb.bits(2);
                if id == 2 && ps_allowed != 0 {
                    left -= self.ps.read_data(gb, left);
                } else {
                    gb.skip_long(left as isize);
                    left = 0;
                }
            }
            if left > 0 {
                gb.skip(left as u32);
            }
        }
        (gb.pos() - cnt) as u32
    }

    /// `ff_aac_sbr_decode_extension`: consumes `cnt` bytes of `gb_host`.
    pub(crate) fn decode_extension(&mut self, gb_host: &mut BitReader, crc: bool, cnt: isize, id_aac: u32, core_rate: u32, ext_rate: &mut u32, ps_allowed: i32) {
        let mut gb = gb_host.clone();
        gb_host.skip_long(cnt * 8 - 4);
        self.reset = false;
        if self.sample_rate == 0 {
            self.sample_rate = 2 * core_rate as i32;
        }
        if *ext_rate == 0 {
            *ext_rate = 2 * core_rate;
        }
        let mut num_sbr_bits = 0u32;
        if crc {
            gb.skip(10);
            num_sbr_bits += 10;
        }
        self.kx[0] = self.kx[1];
        self.m[0] = self.m[1];
        self.kx_and_m_pushed = true;
        num_sbr_bits += 1;
        if gb.bit() {
            num_sbr_bits += self.read_header(&mut gb);
        }
        if self.reset {
            self.sbr_reset();
        }
        if self.start {
            num_sbr_bits += self.read_data(&mut gb, id_aac, ps_allowed);
        }
        let num_align_bits = ((cnt as u32 * 8).wrapping_sub(4).wrapping_sub(num_sbr_bits)) & 7;
        let bytes_read = (num_sbr_bits + num_align_bits + 4) >> 3;
        if bytes_read as isize > cnt {
            self.turnoff();
        }
    }
}

// ---- NEON sbrdsp kernels (aarch64 `sbrdsp_neon.S`) ----------------------------

/// `ff_sbr_sum_square_neon` over `n` complex values: four fused lanes, then
/// two pairwise adds.
fn sum_square(x: &[f32], n: usize) -> f32 {
    let mut v = [0f32; 4];
    let mut p = 0;
    let mut left = n as isize;
    loop {
        for l in 0..4 {
            let a = x.get(p + l).copied().unwrap_or(0.0);
            v[l] = a.mul_add(a, v[l]);
        }
        p += 4;
        left -= 2;
        if left <= 0 {
            break;
        }
    }
    (v[0] + v[1]) + (v[2] + v[3])
}

/// `ff_sbr_autocorrelate_neon`: `phi[3][2][2]` (flat; only the entries the
/// C reads are written).
fn autocorrelate(x: &[f32], phi: &mut [f32; 12]) {
    const FAC: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
    let (mut v1, mut v2, mut v3) = ([0f32; 2], [0f32; 4], [0f32; 4]);
    let mut v4 = [x[0], x[1]];
    let mut v5 = [x[2], x[3], 0.0, 0.0];
    let v16 = [v4[0] * v4[0], v4[1] * v4[1]];
    let v17 = [v5[0] * v4[0], v5[1] * v4[0], 0.0, 0.0];
    let v18 = [v5[0] * v4[1], v5[1] * v4[1], 0.0, 0.0];
    for j in 2..40 {
        v5[2] = x[2 * j];
        v5[3] = x[2 * j + 1];
        v1[0] = v4[0].mul_add(v4[0], v1[0]);
        v1[1] = v4[1].mul_add(v4[1], v1[1]);
        for l in 0..4 {
            v2[l] = v5[l].mul_add(v4[0], v2[l]);
            v3[l] = v5[l].mul_add(v4[1], v3[l]);
        }
        v4 = [v5[0], v5[1]];
        v5[0] = v5[2];
        v5[1] = v5[3];
    }
    let v19 = [v4[0] * v4[0], v4[1] * v4[1]];
    let v20 = [v5[0] * v4[0], v5[1] * v4[0], 0.0, 0.0];
    let v21 = [v5[0] * v4[1], v5[1] * v4[1], 0.0, 0.0];
    let mut v22 = [0f32; 4];
    let mut v23 = [0f32; 4];
    for l in 0..4 {
        v22[l] = (v2[l] + v20[l]) - v17[l];
        v23[l] = (v3[l] + v21[l]) - v18[l];
    }
    let v23 = [v23[1] * FAC[0], v23[0] * FAC[1], v23[3] * FAC[2], v23[2] * FAC[3]];
    for l in 0..4 {
        phi[l] = v22[l] + v23[l];
    }
    let t = [(v1[0] + v19[0]) - v16[0], (v1[1] + v19[1]) - v16[1]];
    phi[4] = t[0] + t[1];
    let v3r = [v3[1] * FAC[0], v3[0] * FAC[1]];
    phi[6] = v2[0] + v3r[0];
    phi[7] = v2[1] + v3r[1];
    phi[10] = v1[0] + v1[1];
}

/// `ff_sbr_hf_gen_neon`: `out[i] = in[i] + alpha1*in[i-2]*bw^2 +
/// alpha0*in[i-1]*bw` for `i` in `start..end`, with its pairwise sums.
/// `out`/`inp` index complex value 0 at the given offsets.
#[allow(clippy::too_many_arguments)]
fn hf_gen(out: &mut [f32], o: usize, inp: &[f32], ib: usize, a0: [f32; 2], a1: [f32; 2], bw: f32, start: usize, end: usize) {
    let s = [bw * bw, bw * bw, bw * 1.0, bw * 1.0];
    let v0 = [a1[0] * s[0], a1[1] * s[1], a0[0] * s[2], a0[1] * s[3]];
    let v1 = [v0[0], -black(v0[1]), v0[2], -black(v0[3])];
    let v0 = [v0[1], v0[0], v0[3], v0[2]];
    let mut i = start;
    loop {
        let b = ib + 2 * i - 4;
        let v2 = [inp[b], inp[b + 1], inp[b + 2], inp[b + 3]];
        let v3 = [inp[b + 4], inp[b + 5]];
        let p4: [f32; 4] = std::array::from_fn(|l| v2[l] * v1[l]);
        let p5: [f32; 4] = std::array::from_fn(|l| v2[l] * v0[l]);
        let re = (p4[0] + p4[1]) + (p4[2] + p4[3]);
        let im = (p5[0] + p5[1]) + (p5[2] + p5[3]);
        out[o + 2 * i] = re + v3[0];
        out[o + 2 * i + 1] = im + v3[1];
        i += 1;
        if i >= end {
            break;
        }
    }
}

/// Keeps LLVM from folding a negation into a neighbouring fused op.
#[inline(always)]
fn black(x: f32) -> f32 {
    std::hint::black_box(x)
}

/// `ff_sbr_hf_apply_noise_{0..3}_neon` on `y` (flat floats from complex
/// value `kx` of the row), pairs of bands per step.
fn apply_noise(y: &mut [f32], yo: usize, s_m: &[f32], q_filt: &[f32], noise: u32, phi: [f32; 4], m_max: usize) {
    let nt = &t::NOISE_TABLE;
    let mut x3 = noise as usize + 1;
    let mut m = 0usize;
    let mut left = m_max as isize;
    loop {
        x3 &= 0x1ff;
        let ntv: [f32; 4] = std::array::from_fn(|l| f(nt[2 * x3 + l]));
        x3 += 2;
        let sv = [s_m[m], s_m[m], s_m.get(m + 1).copied().unwrap_or(0.0), s_m.get(m + 1).copied().unwrap_or(0.0)];
        let qv = [q_filt[m], q_filt[m], q_filt.get(m + 1).copied().unwrap_or(0.0), q_filt.get(m + 1).copied().unwrap_or(0.0)];
        for l in 0..4 {
            let at = yo + 2 * m + l;
            if at >= y.len() {
                continue;
            }
            let v2 = y[at];
            let sine = phi[l].mul_add(sv[l], v2);
            let noise_v = ntv[l].mul_add(qv[l], v2);
            y[at] = if sv[l] == 0.0 { noise_v } else { sine };
        }
        m += 2;
        left -= 2;
        if left <= 0 {
            break;
        }
    }
}

/// av_tx float `vector_fmul_add` (NEON `fmla`): `dst = src0 * src1 + src2`.
fn vector_fmul_add(dst: &mut [f32], src0: &[f32], src1: &[f32], n: usize) {
    for i in 0..n {
        dst[i] = src0[i].mul_add(src1[i], dst[i]);
    }
}

impl Sbr {
    /// `sbr_dequant`.
    fn dequant(&mut self, id_aac: u32) {
        const EXP2_TAB: [f64; 2] = [1.0, std::f64::consts::SQRT_2];
        let n = self.n;
        let n_q = self.n_q as usize;
        if id_aac == TYPE_CPE && self.bs_coupling {
            let pan_offset: i32 = if self.data[0].bs_amp_res != 0 { 12 } else { 24 };
            for e in 1..=self.data[0].bs_num_env {
                for k in 0..n[self.data[0].bs_freq_res[e] as usize] as usize {
                    let q0 = self.data[0].env_facs_q[e][k] as i32;
                    let q1 = self.data[1].env_facs_q[e][k] as i32;
                    let (mut temp1, temp2): (f32, f32) = if self.data[0].bs_amp_res != 0 {
                        (exp2fi(q0 + 7), exp2fi(pan_offset - q1))
                    } else {
                        (
                            (exp2fi((q0 >> 1) + 7) as f64 * EXP2_TAB[(q0 & 1) as usize]) as f32,
                            (exp2fi((pan_offset - q1) >> 1) as f64 * EXP2_TAB[((pan_offset - q1) & 1) as usize]) as f32,
                        )
                    };
                    if temp1 as f64 > 1e20 {
                        temp1 = 1.0;
                    }
                    let fac = temp1 / (1.0f32 + temp2);
                    self.data[0].env_facs[e][k] = fac;
                    self.data[1].env_facs[e][k] = fac * temp2;
                }
            }
            for e in 1..=self.data[0].bs_num_noise {
                for k in 0..n_q {
                    let temp1 = exp2fi(NOISE_FLOOR_OFFSET - self.data[0].noise_facs_q[e][k] as i32 + 1);
                    let temp2 = exp2fi(12 - self.data[1].noise_facs_q[e][k] as i32);
                    let fac = temp1 / (1.0f32 + temp2);
                    self.data[0].noise_facs[e][k] = fac;
                    self.data[1].noise_facs[e][k] = fac * temp2;
                }
            }
        } else {
            for ch in 0..(id_aac == TYPE_CPE) as usize + 1 {
                let d = &mut self.data[ch];
                for e in 1..=d.bs_num_env {
                    for k in 0..n[d.bs_freq_res[e] as usize] as usize {
                        let q = d.env_facs_q[e][k] as i32;
                        let mut v = if d.bs_amp_res != 0 {
                            exp2fi(q + 6)
                        } else {
                            (exp2fi((q >> 1) + 6) as f64 * EXP2_TAB[(q & 1) as usize]) as f32
                        };
                        if v as f64 > 1e20 {
                            v = 1.0;
                        }
                        d.env_facs[e][k] = v;
                    }
                }
                for e in 1..=d.bs_num_noise {
                    for k in 0..n_q {
                        d.noise_facs[e][k] = exp2fi(NOISE_FLOOR_OFFSET - d.noise_facs_q[e][k] as i32);
                    }
                }
            }
        }
    }

    /// `sbr_qmf_analysis` for channel `ch` from 1024 input samples.
    fn qmf_analysis(&mut self, input: &[f32], ch: usize) {
        let d = &mut self.data[ch];
        let x = &mut d.analysis;
        x.copy_within(1024..1024 + 288, 0);
        x[288..288 + 1024].copy_from_slice(&input[..1024]);
        let buf_idx = d.ypos;
        let mut z = [0f32; 320];
        let mut coeffs = [0f32; 64];
        for i in 0..32 {
            let xs = &x[32 * i..32 * i + 320];
            for j in 0..320 {
                z[j] = f(t::QMF_WINDOW_DS[j]) * xs[319 - j];
            }
            for k in 0..64 {
                z[k] = (((z[k] + z[k + 64]) + z[k + 128]) + z[k + 192]) + z[k + 256];
            }
            // qmf_pre_shuffle
            z[64] = z[0];
            z[65] = z[1];
            for k in 1..32 {
                z[64 + 2 * k] = -z[64 - k];
                z[64 + 2 * k + 1] = z[k + 1];
            }
            coeffs.copy_from_slice(&z[64..128]);
            self.mdct_ana.run(&coeffs, &mut z[..64]);
            // qmf_post_shuffle
            let w = &mut d.w[wi(buf_idx, i, 0)..wi(buf_idx, i, 0) + 64];
            for k in 0..32 {
                w[2 * k] = -z[63 - k];
                w[2 * k + 1] = z[k];
            }
        }
    }

    /// `sbr_lf_gen`.
    fn lf_gen(&mut self, ch: usize) {
        self.x_low.fill(0.0);
        let d = &self.data[ch];
        let buf = d.ypos;
        for k in 0..self.kx[1] as usize {
            for i in 8..40 {
                self.x_low[xl(k, i)] = d.w[wi(buf, i - 8, k)];
                self.x_low[xl(k, i) + 1] = d.w[wi(buf, i - 8, k) + 1];
            }
        }
        let buf = 1 - buf;
        for k in 0..self.kx[0] as usize {
            for i in 0..8 {
                self.x_low[xl(k, i)] = d.w[wi(buf, i + 24, k)];
                self.x_low[xl(k, i) + 1] = d.w[wi(buf, i + 24, k) + 1];
            }
        }
    }

    /// `sbr_hf_inverse_filter` (C, uncontracted).
    fn hf_inverse_filter(&mut self) {
        for k in 0..self.k[0].max(0) as usize {
            let mut phi = [0f32; 12];
            autocorrelate(&self.x_low[xl(k, 0)..xl(k, 0) + 80], &mut phi);
            let p = |a: usize, b: usize, c: usize| phi[a * 4 + b * 2 + c];
            let dk = p(2, 1, 0) * p(1, 0, 0) - (p(1, 1, 0) * p(1, 1, 0) + p(1, 1, 1) * p(1, 1, 1)) / 1.000001f32;
            let (a1, a0) = (&mut self.alpha1[k], &mut self.alpha0[k]);
            if dk == 0.0 {
                *a1 = [0.0, 0.0];
            } else {
                let tr = p(0, 0, 0) * p(1, 1, 0) - p(0, 0, 1) * p(1, 1, 1) - p(0, 1, 0) * p(1, 0, 0);
                let ti = p(0, 0, 0) * p(1, 1, 1) + p(0, 0, 1) * p(1, 1, 0) - p(0, 1, 1) * p(1, 0, 0);
                *a1 = [tr / dk, ti / dk];
            }
            if p(1, 0, 0) == 0.0 {
                *a0 = [0.0, 0.0];
            } else {
                let tr = p(0, 0, 0) + a1[0] * p(1, 1, 0) + a1[1] * p(1, 1, 1);
                let ti = p(0, 0, 1) + a1[1] * p(1, 1, 0) - a1[0] * p(1, 1, 1);
                *a0 = [-tr / p(1, 0, 0), -ti / p(1, 0, 0)];
            }
            if a1[0] * a1[0] + a1[1] * a1[1] >= 16.0 || a0[0] * a0[0] + a0[1] * a0[1] >= 16.0 {
                *a1 = [0.0, 0.0];
                *a0 = [0.0, 0.0];
            }
        }
    }

    /// `sbr_chirp`.
    fn chirp(&mut self, ch: usize) {
        const BW_TAB: [f32; 4] = [0.0, 0.75, 0.9, 0.98];
        let n_q = self.n_q as usize;
        let d = &mut self.data[ch];
        for i in 0..n_q {
            let mut new_bw = if d.bs_invf_mode[0][i] + d.bs_invf_mode[1][i] == 1 { 0.6f32 } else { BW_TAB[d.bs_invf_mode[0][i] as usize] };
            if new_bw < d.bw_array[i] {
                new_bw = 0.75f32 * new_bw + 0.25f32 * d.bw_array[i];
            } else {
                new_bw = 0.90625f32 * new_bw + 0.09375f32 * d.bw_array[i];
            }
            d.bw_array[i] = if new_bw < 0.015625 { 0.0 } else { new_bw };
        }
    }

    /// `sbr_hf_gen`.
    fn hf_gen_all(&mut self, ch: usize) -> Result<(), ()> {
        let d = &self.data[ch];
        let (start, end) = (2 * d.t_env[0] as usize, 2 * d.t_env[d.bs_num_env] as usize);
        let mut g: i32 = 0;
        let mut k = self.kx[1] as usize;
        for j in 0..self.num_patches as usize {
            for x in 0..self.patch_num_subbands[j] as usize {
                let p = self.patch_start_subband[j] as usize + x;
                while g <= self.n_q && k >= self.f_tablenoise[g as usize] as usize {
                    g += 1;
                }
                g -= 1;
                if g < 0 {
                    return Err(());
                }
                hf_gen(&mut self.x_high, xl(k, ENV_OFF), &self.x_low, xl(p, ENV_OFF), self.alpha0[p], self.alpha1[p], d.bw_array[g as usize], start, end);
                k += 1;
            }
        }
        let top = (self.m[1] + self.kx[1]) as usize;
        if k < top {
            self.x_high[xl(k, 0)..xl(top, 0)].fill(0.0);
        }
        Ok(())
    }

    /// `sbr_mapping`.
    fn mapping(&mut self, ch: usize) -> Result<(), ()> {
        let kx1 = self.kx[1] as usize;
        let d = &mut self.data[ch];
        for r in 1..8 {
            d.s_indexmapped[r] = [0; 48];
        }
        let e_a = d.e_a;
        for e in 0..d.bs_num_env {
            let res = d.bs_freq_res[e + 1] as usize;
            let ilim = self.n[res] as usize;
            let table: &[u16] = if res != 0 { &self.f_tablehigh } else { &self.f_tablelow };
            if kx1 != table[0] as usize {
                self.start = false;
                self.ready_for_dequant = false;
                self.kx[1] = 32;
                self.m[1] = 0;
                self.data[0].e_a[1] = -1;
                self.data[1].e_a[1] = -1;
                self.spectrum = [0xff; 6];
                return Err(());
            }
            for i in 0..ilim {
                for m in table[i] as usize..table[i + 1] as usize {
                    self.e_origmapped[e][m - kx1] = d.env_facs[e + 1][i];
                }
            }
            let k = (d.bs_num_noise > 1 && d.t_env[e] >= d.t_q[1]) as usize;
            for i in 0..self.n_q as usize {
                for m in self.f_tablenoise[i] as usize..self.f_tablenoise[i + 1] as usize {
                    self.q_mapped[e][m - kx1] = d.noise_facs[k + 1][i];
                }
            }
            for i in 0..self.n[1] as usize {
                if d.bs_add_harmonic_flag {
                    let mid = (self.f_tablehigh[i] as usize + self.f_tablehigh[i + 1] as usize) >> 1;
                    d.s_indexmapped[e + 1][mid - kx1] =
                        d.bs_add_harmonic[i] * (e as i32 >= e_a[1] || d.s_indexmapped[0][mid - kx1] == 1) as u8;
                }
            }
            for i in 0..ilim {
                let (lo, hi) = (table[i] as usize, table[i + 1] as usize);
                let present = (lo..hi).any(|m| d.s_indexmapped[e + 1][m - kx1] != 0) as u8;
                for m in lo..hi {
                    self.s_mapped[e][m - kx1] = present;
                }
            }
        }
        d.s_indexmapped[0] = d.s_indexmapped[d.bs_num_env];
        Ok(())
    }

    /// `sbr_env_estimate`.
    fn env_estimate(&mut self, ch: usize) {
        let kx1 = self.kx[1] as usize;
        let d = &self.data[ch];
        if self.bs_interpol_freq != 0 {
            for e in 0..d.bs_num_env {
                let recip = 0.5f32 / (d.t_env[e + 1] as i32 - d.t_env[e] as i32) as f32;
                let ilb = d.t_env[e] as usize * 2 + ENV_OFF;
                let iub = d.t_env[e + 1] as usize * 2 + ENV_OFF;
                for m in 0..self.m[1] as usize {
                    let sum = sum_square(&self.x_high[xl(m + kx1, ilb)..], iub - ilb);
                    self.e_curr[e][m] = sum * recip;
                }
            }
        } else {
            for e in 0..d.bs_num_env {
                let env_size = 2 * (d.t_env[e + 1] as i32 - d.t_env[e] as i32);
                let ilb = d.t_env[e] as usize * 2 + ENV_OFF;
                let iub = d.t_env[e + 1] as usize * 2 + ENV_OFF;
                let res = d.bs_freq_res[e + 1] as usize;
                let table: &[u16] = if res != 0 { &self.f_tablehigh } else { &self.f_tablelow };
                for p in 0..self.n[res] as usize {
                    let (lo, hi) = (table[p] as usize, table[p + 1] as usize);
                    let den = env_size * (hi as i32 - lo as i32);
                    let mut sum = 0f32;
                    for k in lo..hi {
                        sum += sum_square(&self.x_high[xl(k, ilb)..], iub - ilb);
                    }
                    sum /= den as f32;
                    for k in lo..hi {
                        self.e_curr[e][k - kx1] = sum;
                    }
                }
            }
        }
    }

    /// `sbr_gain_calc`.
    fn gain_calc(&mut self, ch: usize) {
        let limgain: [f32; 4] = [0.70795f64 as f32, 1.0, 1.41254f64 as f32, 10000000000f64 as f32];
        let kx1 = self.kx[1] as usize;
        let d = &self.data[ch];
        let e_a = d.e_a;
        for e in 0..d.bs_num_env {
            let delta = !(e as i32 == e_a[1] || e as i32 == e_a[0]);
            for k in 0..self.n_lim as usize {
                let (lo, hi) = (self.f_tablelim[k] as usize - kx1, self.f_tablelim[k + 1] as usize - kx1);
                for m in lo..hi {
                    let temp = self.e_origmapped[e][m] / (1.0f32 + self.q_mapped[e][m]);
                    self.q_m[e][m] = (temp * self.q_mapped[e][m]).sqrt();
                    self.s_m[e][m] = (temp * d.s_indexmapped[e + 1][m] as f32).sqrt();
                    self.gain[e][m] = if self.s_mapped[e][m] == 0 {
                        (self.e_origmapped[e][m] / ((1.0f32 + self.e_curr[e][m]) * (1.0f32 + self.q_mapped[e][m] * delta as i32 as f32))).sqrt()
                    } else {
                        (self.e_origmapped[e][m] * self.q_mapped[e][m] / ((1.0f32 + self.e_curr[e][m]) * (1.0f32 + self.q_mapped[e][m]))).sqrt()
                    };
                    self.gain[e][m] += f32::MIN_POSITIVE;
                }
                let mut sum = [0f32; 2];
                for m in lo..hi {
                    sum[0] += self.e_origmapped[e][m];
                    sum[1] += self.e_curr[e][m];
                }
                let mut gain_max = limgain[self.bs_limiter_gains as usize] * ((f32::EPSILON + sum[0]) / (f32::EPSILON + sum[1])).sqrt();
                gain_max = if 100000.0f32 > gain_max { gain_max } else { 100000.0 };
                for m in lo..hi {
                    let q_m_max = self.q_m[e][m] * gain_max / self.gain[e][m];
                    self.q_m[e][m] = if self.q_m[e][m] > q_m_max { q_m_max } else { self.q_m[e][m] };
                    self.gain[e][m] = if self.gain[e][m] > gain_max { gain_max } else { self.gain[e][m] };
                }
                sum = [0.0, 0.0];
                for m in lo..hi {
                    sum[0] += self.e_origmapped[e][m];
                    let sel = (delta && self.s_m[e][m] == 0.0) as i32 as f32;
                    sum[1] += self.e_curr[e][m] * self.gain[e][m] * self.gain[e][m] + self.s_m[e][m] * self.s_m[e][m] + sel * self.q_m[e][m] * self.q_m[e][m];
                }
                let mut gain_boost = ((f32::EPSILON + sum[0]) / (f32::EPSILON + sum[1])).sqrt();
                gain_boost = if 1.584893192f32 > gain_boost { gain_boost } else { 1.584893192 };
                for m in lo..hi {
                    self.gain[e][m] *= gain_boost;
                    self.q_m[e][m] *= gain_boost;
                    self.s_m[e][m] *= gain_boost;
                }
            }
        }
    }

    /// `sbr_hf_assemble` into `Y[ypos]`.
    fn hf_assemble(&mut self, ch: usize) {
        let h_smooth: [f32; 5] = [
            0.33333333333333f64 as f32,
            0.30150283239582f64 as f32,
            0.21816949906249f64 as f32,
            0.11516383427084f64 as f32,
            0.03183050093751f64 as f32,
        ];
        let h_sl = 4 * (self.bs_smoothing_mode == 0) as usize;
        let kx = self.kx[1] as usize;
        let m_max = self.m[1] as usize;
        let reset = self.reset;
        let d = &mut self.data[ch];
        let e_a = d.e_a;
        let mut indexnoise = d.f_indexnoise;
        let mut indexsine = d.f_indexsine;
        let t0 = 2 * d.t_env[0] as usize;
        if reset {
            for i in 0..h_sl {
                d.g_temp[i + t0][..m_max].copy_from_slice(&self.gain[0][..m_max]);
                d.q_temp[i + t0][..m_max].copy_from_slice(&self.q_m[0][..m_max]);
            }
        } else if h_sl != 0 {
            let old = 2 * d.t_env_num_env_old as usize;
            for i in 0..4 {
                d.g_temp[i + t0] = d.g_temp[i + old];
                d.q_temp[i + t0] = d.q_temp[i + old];
            }
        }
        for e in 0..d.bs_num_env {
            for i in 2 * d.t_env[e] as usize..2 * d.t_env[e + 1] as usize {
                d.g_temp[h_sl + i][..m_max].copy_from_slice(&self.gain[e][..m_max]);
                d.q_temp[h_sl + i][..m_max].copy_from_slice(&self.q_m[e][..m_max]);
            }
        }
        let yb = d.ypos;
        for e in 0..d.bs_num_env {
            for i in 2 * d.t_env[e] as usize..2 * d.t_env[e + 1] as usize {
                let mut g_filt = [0f32; 48];
                let mut q_filt = [0f32; 48];
                if h_sl != 0 && e as i32 != e_a[0] && e as i32 != e_a[1] {
                    for m in 0..m_max {
                        let idx1 = i + h_sl;
                        g_filt[m] = 0.0;
                        q_filt[m] = 0.0;
                        for j in 0..=h_sl {
                            g_filt[m] += d.g_temp[idx1 - j][m] * h_smooth[j];
                            q_filt[m] += d.q_temp[idx1 - j][m] * h_smooth[j];
                        }
                    }
                } else {
                    g_filt = d.g_temp[i + h_sl];
                    q_filt = d.q_temp[i];
                }
                // hf_g_filt
                let ixh = i + ENV_OFF;
                for m in 0..m_max {
                    let src = xl(kx + m, ixh);
                    let o = yi(yb, i, kx + m);
                    d.y[o] = self.x_high[src] * g_filt[m];
                    d.y[o + 1] = self.x_high[src + 1] * g_filt[m];
                }
                if e as i32 != e_a[0] && e as i32 != e_a[1] {
                    let odd = (kx & 1) != 0;
                    let phi: [f32; 4] = match indexsine {
                        0 => [1.0, 0.0, 1.0, 0.0],
                        1 if !odd => [0.0, 1.0, 0.0, -1.0],
                        1 => [0.0, -1.0, 0.0, 1.0],
                        2 => [-1.0, 0.0, -1.0, 0.0],
                        _ if !odd => [0.0, -1.0, 0.0, 1.0],
                        _ => [0.0, 1.0, 0.0, -1.0],
                    };
                    let yo = yi(yb, i, kx);
                    let s_m = self.s_m[e];
                    apply_noise(&mut d.y, yo, &s_m, &q_filt, indexnoise, phi, m_max);
                } else {
                    let idx = (indexsine & 1) as i32;
                    let a = 1 - ((indexsine as i32 + (kx as i32 & 1)) & 2);
                    let b = (a ^ (-idx)) + idx;
                    let base = yi(yb, i, kx) + idx as usize;
                    let inp = &self.s_m[e];
                    let mut m = 0;
                    while m + 1 < m_max {
                        d.y[base + 2 * m] += inp[m] * a as f32;
                        d.y[base + 2 * m + 2] += inp[m + 1] * b as f32;
                        m += 2;
                    }
                    if m_max & 1 != 0 {
                        d.y[base + 2 * m] += inp[m] * a as f32;
                    }
                }
                indexnoise = (indexnoise + m_max as u32) & 0x1ff;
                indexsine = (indexsine + 1) & 3;
            }
        }
        d.f_indexnoise = indexnoise;
        d.f_indexsine = indexsine;
    }

    /// `sbr_x_gen` into `X[ch]`.
    fn x_gen(&mut self, ch: usize) {
        let d = &self.data[ch];
        let (y0, y1) = (1 - d.ypos, d.ypos);
        let i_temp = (2 * d.t_env_num_env_old as i32 - 32).max(0) as usize;
        let xs = &mut self.x[xi(ch, 0, 0, 0)..xi(ch, 0, 0, 0) + 2 * 38 * 64];
        xs.fill(0.0);
        let at = |ri: usize, i: usize, k: usize| (ri * 38 + i) * 64 + k;
        let mut k = 0;
        while k < self.kx[0] as usize {
            for i in 0..i_temp {
                xs[at(0, i, k)] = self.x_low[xl(k, i + ENV_OFF)];
                xs[at(1, i, k)] = self.x_low[xl(k, i + ENV_OFF) + 1];
            }
            k += 1;
        }
        while k < (self.kx[0] + self.m[0]) as usize {
            for i in 0..i_temp {
                xs[at(0, i, k)] = d.y[yi(y0, i + 32, k)];
                xs[at(1, i, k)] = d.y[yi(y0, i + 32, k) + 1];
            }
            k += 1;
        }
        k = 0;
        while k < self.kx[1] as usize {
            for i in i_temp..38 {
                xs[at(0, i, k)] = self.x_low[xl(k, i + ENV_OFF)];
                xs[at(1, i, k)] = self.x_low[xl(k, i + ENV_OFF) + 1];
            }
            k += 1;
        }
        while k < (self.kx[1] + self.m[1]) as usize {
            for i in i_temp..32 {
                xs[at(0, i, k)] = d.y[yi(y1, i, k)];
                xs[at(1, i, k)] = d.y[yi(y1, i, k) + 1];
            }
            k += 1;
        }
    }

    /// `sbr_qmf_synthesis` of `X[ch]` into `out` (2048 samples, or 1024
    /// downsampled); `state` selects the channel's filterbank buffer.
    fn qmf_synthesis(&mut self, ch: usize, state: usize, out: &mut [f32], div: usize) {
        let window: Vec<f32> = if div != 0 {
            t::QMF_WINDOW_DS.iter().map(|&b| f(b)).collect()
        } else {
            t::QMF_WINDOW_US.iter().map(|&b| f(b)).collect()
        };
        let step = 128 >> div;
        let mut mdct_buf = [[0f32; 64]; 2];
        let mut o = 0;
        for i in 0..32 {
            let d = &mut self.data[state];
            if d.synth_offset < step {
                let saved = (1280 - 128) >> div;
                d.synth.copy_within(0..saved, SYNTH_BUF - saved);
                d.synth_offset = SYNTH_BUF - saved - step;
            } else {
                d.synth_offset -= step;
            }
            let voff = d.synth_offset;
            let re = xi(ch, 0, i, 0);
            let im = xi(ch, 1, i, 0);
            if div != 0 {
                for n in 0..32 {
                    self.x[re + n] = -self.x[re + n];
                    self.x[re + 32 + n] = self.x[im + 31 - n];
                }
                let inp: [f32; 64] = self.x[re..re + 64].try_into().unwrap();
                self.mdct.run(&inp, &mut mdct_buf[0]);
                let v = &mut self.data[state].synth[voff..voff + 64];
                for k in 0..32 {
                    v[k] = mdct_buf[0][63 - 2 * k];
                    v[63 - k] = -mdct_buf[0][63 - 2 * k - 1];
                }
            } else {
                for k in (1..64).step_by(2) {
                    self.x[im + k] = -self.x[im + k];
                }
                let in0: [f32; 64] = self.x[re..re + 64].try_into().unwrap();
                let in1: [f32; 64] = self.x[im..im + 64].try_into().unwrap();
                self.mdct.run(&in0, &mut mdct_buf[0]);
                self.mdct.run(&in1, &mut mdct_buf[1]);
                let v = &mut self.data[state].synth[voff..voff + 128];
                let (s0, s1) = (&mdct_buf[1], &mdct_buf[0]);
                for k in 0..64 {
                    v[k] = s0[k] - s1[63 - k];
                    v[127 - k] = s0[k] + s1[63 - k];
                }
            }
            let v = &self.data[state].synth[voff..];
            let n = 64 >> div;
            let outc = &mut out[o..o + n];
            for k in 0..n {
                outc[k] = v[k] * window[k];
            }
            for (vo, wo) in [(192, 64), (256, 128), (448, 192), (512, 256), (704, 320), (768, 384), (960, 448), (1024, 512), (1216, 576)] {
                vector_fmul_add(outc, &v[vo >> div..], &window[wo >> div..], n);
            }
            o += n;
        }
    }

    /// For [`super::ToolStats`]: SBR active, coupled, PS active, 34-band
    /// PS, IPD/OPD.
    pub(crate) fn coverage(&self) -> (bool, bool, bool, bool, bool) {
        let c = &self.ps.common;
        (self.start, self.bs_coupling, c.start, c.is34bands, c.enable_ipdopd)
    }

    /// `ff_aac_sbr_apply`: `l`/`r` hold 1024 core samples on entry and the
    /// SBR output on return (both used when `nch_out == 2`).
    pub(crate) fn apply(&mut self, id_aac: u32, l: &mut [f32], r: &mut [f32], downsampled: bool, ps_on: bool) {
        if id_aac != self.id_aac {
            self.turnoff();
        }
        if self.start && !self.ready_for_dequant {
            self.turnoff();
        }
        if !self.kx_and_m_pushed {
            self.kx[0] = self.kx[1];
            self.m[0] = self.m[1];
        } else {
            self.kx_and_m_pushed = false;
        }
        if self.start {
            self.dequant(id_aac);
            self.ready_for_dequant = false;
        }
        let mut nch = if id_aac == TYPE_CPE { 2 } else { 1 };
        for ch in 0..nch {
            let input: Vec<f32> = if ch == 0 { l[..1024].to_vec() } else { r[..1024].to_vec() };
            self.qmf_analysis(&input, ch);
            self.lf_gen(ch);
            self.data[ch].ypos ^= 1;
            if self.start {
                self.hf_inverse_filter();
                self.chirp(ch);
                if self.hf_gen_all(ch).is_ok() && self.mapping(ch).is_ok() {
                    self.env_estimate(ch);
                    self.gain_calc(ch);
                    self.hf_assemble(ch);
                }
            }
            self.x_gen(ch);
        }
        if ps_on {
            if self.ps.common.start {
                let top = (self.kx[1] + self.m[1]) as usize;
                let (x0, x1) = self.x.split_at_mut(2 * 38 * 64);
                self.ps.apply(x0, x1, top);
            } else {
                let (x0, x1) = self.x.split_at_mut(2 * 38 * 64);
                x1.copy_from_slice(x0);
            }
            nch = 2;
        }
        let div = downsampled as usize;
        self.qmf_synthesis(0, 0, l, div);
        if nch == 2 {
            self.qmf_synthesis(1, 1, r, div);
        }
    }
}
