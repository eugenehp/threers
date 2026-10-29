//! Parametric Stereo: ffmpeg n7.1's `aacps_common.c` (parsing), `aacps.c`
//! (float) and `aacpsdsp`, with the aarch64 NEON `add_squares`,
//! `mul_pair_single`, `hybrid_analysis` and `stereo_interpolate` kernels in
//! their lane order. Tables come from `ps_tableinit` (see `sbr_tables`).

#![allow(clippy::needless_range_loop, clippy::excessive_precision, clippy::type_complexity)]

use super::bits::BitReader;
use super::sbr::LenVlc;
use super::sbr_tables as t;

const MAX_NUM_ENV: usize = 5;
const MAX_NR_IIDICC: usize = 34;
const QMF_SLOTS: usize = 32;
const MAX_DELAY: usize = 14;
const AP_LINKS: usize = 3;
const MAX_AP_DELAY: usize = 5;
const MAX_SSB: usize = 91;
const MAX_AP_BANDS: usize = 50;

const NR_PAR_BANDS: [usize; 2] = [20, 34];
const NR_IPDOPD_BANDS: [usize; 2] = [11, 17];
const NR_BANDS: [usize; 2] = [71, 91];
const DECAY_CUTOFF: [i32; 2] = [10, 32];
const NR_ALLPASS_BANDS: [usize; 2] = [30, 50];
const SHORT_DELAY_BAND: [usize; 2] = [42, 62];

#[inline]
fn f(b: u32) -> f32 {
    f32::from_bits(b)
}

fn ps_vlcs() -> &'static [LenVlc] {
    static V: std::sync::OnceLock<Vec<LenVlc>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let mut at = 0;
        (0..10)
            .map(|i| {
                let n = t::PS_HUFF_COUNTS[i] as usize;
                let v = LenVlc::new(&t::PS_HUFF_SYM[at..at + n], &t::PS_HUFF_LEN[at..at + n], t::PS_HUFF_OFFSETS[i] as i32);
                at += n;
                v
            })
            .collect()
    })
}

const HUFF_IID_DF1: usize = 0;
const HUFF_IID_DT1: usize = 1;
const HUFF_IID_DF0: usize = 2;
const HUFF_IID_DT0: usize = 3;
const HUFF_ICC_DF: usize = 4;
const HUFF_ICC_DT: usize = 5;
const HUFF_IPD_DF: usize = 6;
const HUFF_IPD_DT: usize = 7;
const HUFF_OPD_DF: usize = 8;
const HUFF_OPD_DT: usize = 9;

type Par = [[i8; MAX_NR_IIDICC]; MAX_NUM_ENV];

#[derive(Clone)]
pub(crate) struct PsCommon {
    pub(crate) start: bool,
    enable_iid: bool,
    iid_quant: usize,
    nr_iid_par: usize,
    nr_ipdopd_par: usize,
    enable_icc: bool,
    icc_mode: u32,
    nr_icc_par: usize,
    enable_ext: bool,
    frame_class: u32,
    num_env_old: usize,
    num_env: usize,
    pub(crate) enable_ipdopd: bool,
    border_position: [i32; MAX_NUM_ENV + 1],
    iid_par: Par,
    icc_par: Par,
    ipd_par: Par,
    opd_par: Par,
    pub(crate) is34bands: bool,
    is34bands_old: bool,
}

pub(crate) struct Ps {
    pub(crate) common: PsCommon,
    in_buf: Vec<[[f32; 2]; 44]>,
    delay: Vec<[[f32; 2]; QMF_SLOTS + MAX_DELAY]>,
    ap_delay: Vec<[[[f32; 2]; QMF_SLOTS + MAX_AP_DELAY]; AP_LINKS]>,
    peak_decay_nrg: [f32; 34],
    power_smooth: [f32; 34],
    peak_decay_diff_smooth: [f32; 34],
    /// `H11/H12/H21/H22[2][MAX_NUM_ENV+1][34]`.
    h: [[[[f32; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2]; 4],
    lbuf: Vec<[[f32; 2]; 32]>,
    rbuf: Vec<[[f32; 2]; 32]>,
    opd_hist: [i8; MAX_NR_IIDICC],
    ipd_hist: [i8; MAX_NR_IIDICC],
}

impl Default for PsCommon {
    fn default() -> Self {
        let z = [[0i8; MAX_NR_IIDICC]; MAX_NUM_ENV];
        Self {
            start: false,
            enable_iid: false,
            iid_quant: 0,
            nr_iid_par: 0,
            nr_ipdopd_par: 0,
            enable_icc: false,
            icc_mode: 0,
            nr_icc_par: 0,
            enable_ext: false,
            frame_class: 0,
            num_env_old: 0,
            num_env: 0,
            enable_ipdopd: false,
            border_position: [0; MAX_NUM_ENV + 1],
            iid_par: z,
            icc_par: z,
            ipd_par: z,
            opd_par: z,
            is34bands: false,
            is34bands_old: false,
        }
    }
}

impl Default for Ps {
    fn default() -> Self {
        Self {
            common: PsCommon::default(),
            in_buf: vec![[[0.0; 2]; 44]; 5],
            delay: vec![[[0.0; 2]; QMF_SLOTS + MAX_DELAY]; MAX_SSB],
            ap_delay: vec![[[[0.0; 2]; QMF_SLOTS + MAX_AP_DELAY]; AP_LINKS]; MAX_AP_BANDS],
            peak_decay_nrg: [0.0; 34],
            power_smooth: [0.0; 34],
            peak_decay_diff_smooth: [0.0; 34],
            h: [[[[0.0; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2]; 4],
            lbuf: vec![[[0.0; 2]; 32]; 91],
            rbuf: vec![[[0.0; 2]; 32]; 91],
            opd_hist: [0; MAX_NR_IIDICC],
            ipd_hist: [0; MAX_NR_IIDICC],
        }
    }
}

const LOG2_TAB: [u32; 5] = [0, 0, 1, 1, 2];

impl PsCommon {
    /// `read_{iid,icc,ipdopd}_data`; `Err` on an illegal value.
    #[allow(clippy::too_many_arguments)]
    fn read_par(gb: &mut BitReader, num: usize, num_env_old: usize, par: &mut Par, table: usize, e: usize, dt: bool, mask: i32, check: &dyn Fn(i32) -> bool) -> Result<(), ()> {
        let vlc = &ps_vlcs()[table];
        if dt {
            let e_prev = if e > 0 { e as i32 - 1 } else { num_env_old as i32 - 1 }.max(0) as usize;
            for b in 0..num {
                let mut val = par[e_prev][b] as i32 + vlc.read(gb);
                if mask != 0 {
                    val &= mask;
                }
                par[e][b] = val as i8;
                if check(par[e][b] as i32) {
                    return Err(());
                }
            }
        } else {
            let mut val = 0i32;
            for b in 0..num {
                val += vlc.read(gb);
                if mask != 0 {
                    val &= mask;
                }
                par[e][b] = val as i8;
                if check(par[e][b] as i32) {
                    return Err(());
                }
            }
        }
        Ok(())
    }
}

impl Ps {
    /// `ff_ps_read_data`: returns the bits consumed from `gb_host`.
    pub(crate) fn read_data(&mut self, gb_host: &mut BitReader, bits_left: i32) -> i32 {
        let start = gb_host.pos();
        let mut gb = gb_host.clone();
        let ps = &mut self.common;
        let r: Result<bool, ()> = (|| {
            let header = gb.bit();
            if header {
                ps.enable_iid = gb.bit();
                if ps.enable_iid {
                    let iid_mode = gb.bits(3) as usize;
                    if iid_mode > 5 {
                        return Err(());
                    }
                    ps.nr_iid_par = [10, 20, 34, 10, 20, 34][iid_mode];
                    ps.iid_quant = (iid_mode > 2) as usize;
                    ps.nr_ipdopd_par = [5, 11, 17, 5, 11, 17][iid_mode];
                }
                ps.enable_icc = gb.bit();
                if ps.enable_icc {
                    ps.icc_mode = gb.bits(3);
                    if ps.icc_mode > 5 {
                        return Err(());
                    }
                    ps.nr_icc_par = [10, 20, 34, 10, 20, 34][ps.icc_mode as usize];
                }
                ps.enable_ext = gb.bit();
            }
            ps.frame_class = gb.bits(1);
            ps.num_env_old = ps.num_env;
            ps.num_env = [[0, 1, 2, 4], [1, 2, 3, 4]][ps.frame_class as usize][gb.bits(2) as usize];
            ps.border_position[0] = -1;
            if ps.frame_class != 0 {
                for e in 1..=ps.num_env {
                    ps.border_position[e] = gb.bits(5) as i32;
                    if ps.border_position[e] < ps.border_position[e - 1] {
                        return Err(());
                    }
                }
            } else {
                for e in 1..=ps.num_env {
                    ps.border_position[e] = ((e * QMF_SLOTS) >> LOG2_TAB[ps.num_env]) as i32 - 1;
                }
            }
            let iq = ps.iid_quant as i32;
            let iid_bad = move |v: i32| v.abs() > 7 + 8 * iq;
            if ps.enable_iid {
                for e in 0..ps.num_env {
                    let dt = gb.bit();
                    let table = [HUFF_IID_DF0, HUFF_IID_DF1, HUFF_IID_DT0, HUFF_IID_DT1][2 * dt as usize + ps.iid_quant];
                    PsCommon::read_par(&mut gb, ps.nr_iid_par, ps.num_env_old, &mut ps.iid_par, table, e, dt, 0, &iid_bad)?;
                }
            } else {
                ps.iid_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
            }
            let icc_bad = |v: i32| v as u32 > 7;
            if ps.enable_icc {
                for e in 0..ps.num_env {
                    let dt = gb.bit();
                    PsCommon::read_par(&mut gb, ps.nr_icc_par, ps.num_env_old, &mut ps.icc_par, if dt { HUFF_ICC_DT } else { HUFF_ICC_DF }, e, dt, 0, &icc_bad)?;
                }
            } else {
                ps.icc_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
            }
            if ps.enable_ext {
                let mut cnt = gb.bits(4) as i32;
                if cnt == 15 {
                    cnt += gb.bits(8) as i32;
                }
                cnt *= 8;
                while cnt > 7 {
                    let id = gb.bits(2);
                    let count = gb.pos();
                    if id == 0 {
                        ps.enable_ipdopd = gb.bit();
                        if ps.enable_ipdopd {
                            let never = |_: i32| false;
                            for e in 0..ps.num_env {
                                let dt = gb.bit();
                                let _ = PsCommon::read_par(&mut gb, ps.nr_ipdopd_par, ps.num_env_old, &mut ps.ipd_par, if dt { HUFF_IPD_DT } else { HUFF_IPD_DF }, e, dt, 7, &never);
                                let dt = gb.bit();
                                let _ = PsCommon::read_par(&mut gb, ps.nr_ipdopd_par, ps.num_env_old, &mut ps.opd_par, if dt { HUFF_OPD_DT } else { HUFF_OPD_DF }, e, dt, 7, &never);
                            }
                        }
                        gb.skip(1);
                    }
                    cnt -= 2 + (gb.pos() - count) as i32;
                }
                if cnt < 0 {
                    return Err(());
                }
                gb.skip(cnt as u32);
            }
            // Fix up envelopes.
            if ps.num_env == 0 || ps.border_position[ps.num_env] < QMF_SLOTS as i32 - 1 {
                let source = if ps.num_env > 0 { ps.num_env as i32 - 1 } else { ps.num_env_old as i32 - 1 };
                if source >= 0 && source as usize != ps.num_env {
                    let (s, d) = (source as usize, ps.num_env);
                    if ps.enable_iid {
                        ps.iid_par[d] = ps.iid_par[s];
                    }
                    if ps.enable_icc {
                        ps.icc_par[d] = ps.icc_par[s];
                    }
                    if ps.enable_ipdopd {
                        ps.ipd_par[d] = ps.ipd_par[s];
                        ps.opd_par[d] = ps.opd_par[s];
                    }
                }
                if ps.enable_iid && (0..ps.nr_iid_par).any(|b| iid_bad(ps.iid_par[ps.num_env][b] as i32)) {
                    return Err(());
                }
                if ps.enable_icc && (0..ps.nr_iid_par).any(|b| ps.icc_par[ps.num_env][b] as u8 > 7) {
                    return Err(());
                }
                ps.num_env += 1;
                ps.border_position[ps.num_env] = QMF_SLOTS as i32 - 1;
            }
            ps.is34bands_old = ps.is34bands;
            if ps.enable_iid || ps.enable_icc {
                ps.is34bands = (ps.enable_iid && ps.nr_iid_par == 34) || (ps.enable_icc && ps.nr_icc_par == 34);
            }
            if !ps.enable_ipdopd {
                ps.ipd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
                ps.opd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
            }
            Ok(header)
        })();
        let consumed = (gb.pos() - start) as i32;
        match r {
            Ok(header) if consumed <= bits_left => {
                if header {
                    ps.start = true;
                }
                gb_host.skip_long(consumed as isize);
                consumed
            }
            other => {
                if let Ok(header) = other {
                    // ffmpeg sets `start` before checking the bit count.
                    if header {
                        ps.start = true;
                    }
                }
                ps.start = false;
                gb_host.skip_long(bits_left as isize);
                ps.iid_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
                ps.icc_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
                ps.ipd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
                ps.opd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
                bits_left
            }
        }
    }
}

// ---- NEON psdsp kernels -------------------------------------------------------

/// `ff_ps_hybrid_analysis_neon`: `n` filters over 13 complex inputs, output
/// every `stride` complex values.
fn hybrid_analysis(out: &mut [[f32; 2]], o: usize, inp: &[[f32; 2]], filter: &[u32], stride: usize, n: usize) {
    let re = |j: usize| inp[j][0];
    let im = |j: usize| inp[j][1];
    let v16: [f32; 4] = std::array::from_fn(|j| re(j) + re(12 - j));
    let v17: [f32; 4] = std::array::from_fn(|j| im(j) + im(12 - j));
    let v18: [f32; 4] = std::array::from_fn(|j| im(j) - im(12 - j));
    let v19: [f32; 4] = std::array::from_fn(|j| re(j) - re(12 - j));
    // v2 = [re4, re5, im4, im5], v4 = [re8, re7, im8, im7],
    // v5 = [im8, im7, re4, re5], v3 = [im4, im5, re8, re7].
    let v22 = [re(4) + re(8), re(5) + re(7), im(4) + im(8), im(5) + im(7)];
    let v23 = [im(8) - im(4), im(7) - im(5), re(4) - re(8), re(5) - re(7)];
    let v20 = [v22[0], v22[1], v23[0], v23[1]];
    let v21 = [v22[2], v22[3], v23[2], v23[3]];
    let v24 = [re(6), im(6)];
    for i in 0..n {
        let fl = |j: usize, c: usize| f(filter[i * 16 + j * 2 + c]);
        let v2: [f32; 4] = std::array::from_fn(|j| fl(j, 0));
        let v3: [f32; 4] = std::array::from_fn(|j| fl(j, 1));
        let v4 = [fl(4, 0), fl(5, 0), fl(4, 1), fl(5, 1)];
        let v6 = [fl(6, 0) * v24[0], fl(6, 0) * v24[1]];
        let mut v0: [f32; 4] = std::array::from_fn(|l| v2[l] * v16[l]);
        let mut v1: [f32; 4] = std::array::from_fn(|l| v2[l] * v17[l]);
        for l in 0..4 {
            v0[l] = (-v3[l]).mul_add(v18[l], v0[l]);
            v1[l] = v3[l].mul_add(v19[l], v1[l]);
            v0[l] = v4[l].mul_add(v20[l], v0[l]);
            v1[l] = v4[l].mul_add(v21[l], v1[l]);
        }
        let p = [v0[0] + v0[1], v0[2] + v0[3], v1[0] + v1[1], v1[2] + v1[3]];
        let s = [p[0] + p[1], p[2] + p[3]];
        out[o + i * stride] = [s[0] + v6[0], s[1] + v6[1]];
    }
}

/// `ff_ps_stereo_interpolate_neon`.
fn stereo_interpolate(l: &mut [[f32; 2]], r: &mut [[f32; 2]], h: &[[f32; 4]; 2], hs: &[[f32; 4]; 2], len: usize) {
    let mut v4 = [h[0][0], h[0][0], h[0][1], h[0][1]];
    let mut v5 = [h[0][2], h[0][2], h[0][3], h[0][3]];
    let v6 = [hs[0][0], hs[0][0], hs[0][1], hs[0][1]];
    let v7 = [hs[0][2], hs[0][2], hs[0][3], hs[0][3]];
    for n in 0..len {
        let v2 = [l[n][0], l[n][1], l[n][0], l[n][1]];
        let v3 = [r[n][0], r[n][1], r[n][0], r[n][1]];
        for k in 0..4 {
            v4[k] += v6[k];
            v5[k] += v7[k];
        }
        let o: [f32; 4] = std::array::from_fn(|k| v3[k].mul_add(v5[k], v2[k] * v4[k]));
        l[n] = [o[0], o[1]];
        r[n] = [o[2], o[3]];
    }
}

/// `ff_ps_stereo_interpolate_ipdopd_neon`.
fn stereo_interpolate_ipdopd(l: &mut [[f32; 2]], r: &mut [[f32; 2]], h: &[[f32; 4]; 2], hs: &[[f32; 4]; 2], len: usize) {
    let neg = |x: f32| -std::hint::black_box(x);
    let mut v16 = [h[0][0], h[0][0], h[0][1], h[0][1]];
    let mut v17 = [h[0][2], h[0][2], h[0][3], h[0][3]];
    let mut v18 = [neg(h[1][0]), h[1][0], neg(h[1][1]), h[1][1]];
    let mut v19 = [neg(h[1][2]), h[1][2], neg(h[1][3]), h[1][3]];
    let v20 = [hs[0][0], hs[0][0], hs[0][1], hs[0][1]];
    let v21 = [hs[0][2], hs[0][2], hs[0][3], hs[0][3]];
    let v22 = [neg(hs[1][0]), hs[1][0], neg(hs[1][1]), hs[1][1]];
    let v23 = [neg(hs[1][2]), hs[1][2], neg(hs[1][3]), hs[1][3]];
    for n in 0..len {
        let v2 = [l[n][0], l[n][1], l[n][0], l[n][1]];
        let v3 = [r[n][0], r[n][1], r[n][0], r[n][1]];
        for k in 0..4 {
            v16[k] += v20[k];
            v17[k] += v21[k];
        }
        let mut v4: [f32; 4] = std::array::from_fn(|k| v3[k].mul_add(v17[k], v2[k] * v16[k]));
        for k in 0..4 {
            v18[k] += v22[k];
            v19[k] += v23[k];
        }
        let v2e = [v2[1], v2[2], v2[3], v2[0]];
        let v3e = [v3[1], v3[2], v3[3], v3[0]];
        for k in 0..4 {
            v4[k] = v2e[k].mul_add(v18[k], v4[k]);
            v4[k] = v3e[k].mul_add(v19[k], v4[k]);
        }
        l[n] = [v4[0], v4[1]];
        r[n] = [v4[2], v4[3]];
    }
}

// ---- parameter remapping --------------------------------------------------------

fn map_idx_10_to_20(out: &mut [i8; 34], par: &[i8; 34], full: bool) {
    let mut b: i32 = if full {
        9
    } else {
        out[10] = 0;
        4
    };
    while b >= 0 {
        let v = par[b as usize];
        out[2 * b as usize + 1] = v;
        out[2 * b as usize] = v;
        b -= 1;
    }
}

fn map_idx_34_to_20(o: &mut [i8; 34], p: &[i8; 34], full: bool) {
    let p = |i: usize| p[i] as i32;
    o[0] = ((2 * p(0) + p(1)) / 3) as i8;
    o[1] = ((p(1) + 2 * p(2)) / 3) as i8;
    o[2] = ((2 * p(3) + p(4)) / 3) as i8;
    o[3] = ((p(4) + 2 * p(5)) / 3) as i8;
    o[4] = ((p(6) + p(7)) / 2) as i8;
    o[5] = ((p(8) + p(9)) / 2) as i8;
    o[6] = p(10) as i8;
    o[7] = p(11) as i8;
    o[8] = ((p(12) + p(13)) / 2) as i8;
    o[9] = ((p(14) + p(15)) / 2) as i8;
    o[10] = p(16) as i8;
    if full {
        o[11] = p(17) as i8;
        o[12] = p(18) as i8;
        o[13] = p(19) as i8;
        o[14] = ((p(20) + p(21)) / 2) as i8;
        o[15] = ((p(22) + p(23)) / 2) as i8;
        o[16] = ((p(24) + p(25)) / 2) as i8;
        o[17] = ((p(26) + p(27)) / 2) as i8;
        o[18] = ((p(28) + p(29) + p(30) + p(31)) / 4) as i8;
        o[19] = ((p(32) + p(33)) / 2) as i8;
    }
}

fn map_idx_10_to_34(o: &mut [i8; 34], p: &[i8; 34], full: bool) {
    if full {
        for (i, s) in [(33, 9), (32, 9), (31, 9), (30, 9), (29, 9), (28, 9), (27, 8), (26, 8), (25, 8), (24, 8), (23, 7), (22, 7), (21, 7), (20, 7), (19, 6), (18, 6), (17, 5), (16, 5)] {
            o[i] = p[s];
        }
    } else {
        o[16] = 0;
    }
    for (i, s) in [(15, 4), (14, 4), (13, 4), (12, 4), (11, 3), (10, 3), (9, 2), (8, 2), (7, 2), (6, 2), (5, 1), (4, 1), (3, 1), (2, 0), (1, 0), (0, 0)] {
        o[i] = p[s];
    }
}

fn map_idx_20_to_34(o: &mut [i8; 34], p: &[i8; 34], full: bool) {
    if full {
        for (i, s) in [(33, 19), (32, 19), (31, 18), (30, 18), (29, 18), (28, 18), (27, 17), (26, 17), (25, 16), (24, 16), (23, 15), (22, 15), (21, 14), (20, 14), (19, 13), (18, 12), (17, 11)] {
            o[i] = p[s];
        }
    }
    for (i, s) in [(16, 10), (15, 9), (14, 9), (13, 8), (12, 8), (11, 7), (10, 6), (9, 5), (8, 5), (7, 4), (6, 4), (5, 3)] {
        o[i] = p[s];
    }
    o[4] = ((p[2] as i32 + p[3] as i32) / 2) as i8;
    o[3] = p[2];
    o[2] = p[1];
    o[1] = ((p[0] as i32 + p[1] as i32) / 2) as i8;
    o[0] = p[0];
}

fn half_sum(a: f32, b: f32) -> f32 {
    (a + b) * 0.5
}

fn map_val_34_to_20(p: &mut [f32; 34]) {
    p[0] = (2.0 * p[0] + p[1]) * 0.33333333f32;
    p[1] = (p[1] + 2.0 * p[2]) * 0.33333333f32;
    p[2] = (2.0 * p[3] + p[4]) * 0.33333333f32;
    p[3] = (p[4] + 2.0 * p[5]) * 0.33333333f32;
    p[4] = half_sum(p[6], p[7]);
    p[5] = half_sum(p[8], p[9]);
    p[6] = p[10];
    p[7] = p[11];
    p[8] = half_sum(p[12], p[13]);
    p[9] = half_sum(p[14], p[15]);
    p[10] = p[16];
    p[11] = p[17];
    p[12] = p[18];
    p[13] = p[19];
    p[14] = half_sum(p[20], p[21]);
    p[15] = half_sum(p[22], p[23]);
    p[16] = half_sum(p[24], p[25]);
    p[17] = half_sum(p[26], p[27]);
    p[18] = (p[28] + p[29] + p[30] + p[31]) * 0.25f32;
    p[19] = half_sum(p[32], p[33]);
}

fn map_val_20_to_34(p: &mut [f32; 34]) {
    for (i, s) in [(33, 19), (32, 19), (31, 18), (30, 18), (29, 18), (28, 18), (27, 17), (26, 17), (25, 16), (24, 16), (23, 15), (22, 15), (21, 14), (20, 14), (19, 13), (18, 12), (17, 11), (16, 10), (15, 9), (14, 9), (13, 8), (12, 8), (11, 7), (10, 6), (9, 5), (8, 5), (7, 4), (6, 4), (5, 3)] {
        p[i] = p[s];
    }
    p[4] = half_sum(p[2], p[3]);
    p[3] = p[2];
    p[2] = p[1];
    p[1] = half_sum(p[0], p[1]);
}

/// `remap34` / `remap20`: the mapped parameters for `num_env` envelopes.
fn remap(par: &Par, num_par: usize, num_env: usize, full: bool, to34: bool) -> Par {
    let mut out = [[0i8; MAX_NR_IIDICC]; MAX_NUM_ENV];
    let f: Option<fn(&mut [i8; 34], &[i8; 34], bool)> = if to34 {
        match num_par {
            20 | 11 => Some(map_idx_20_to_34),
            10 | 5 => Some(map_idx_10_to_34),
            _ => None,
        }
    } else {
        match num_par {
            34 | 17 => Some(map_idx_34_to_20),
            10 | 5 => Some(map_idx_10_to_20),
            _ => None,
        }
    };
    match f {
        Some(f) => {
            for e in 0..num_env {
                f(&mut out[e], &par[e], full);
            }
            out
        }
        None => *par,
    }
}

impl Ps {
    /// `hybrid_analysis` into `lbuf` from `X[0]` (`l`: `[2][38][64]`).
    fn hybrid_analysis(&mut self, l: &[f32], is34: bool, len: usize) {
        let at = |ri: usize, j: usize, i: usize| (ri * 38 + j) * 64 + i;
        for i in 0..5 {
            for j in 0..38 {
                self.in_buf[i][j + 6] = [l[at(0, j, i)], l[at(1, j, i)]];
            }
        }
        let out = &mut self.lbuf;
        if is34 {
            for (band, filt, nf, base) in [(0, &t::F34_0_12[..], 12, 0), (1, &t::F34_1_8[..], 8, 12), (2, &t::F34_2_4[..], 4, 20), (3, &t::F34_2_4[..], 4, 24), (4, &t::F34_2_4[..], 4, 28)] {
                for i in 0..len {
                    let inp = self.in_buf[band][i..i + 13].to_vec();
                    let mut flat: Vec<[f32; 2]> = (0..nf * 32).map(|q| out[base + q / 32][q % 32]).collect();
                    hybrid_analysis(&mut flat, i, &inp, filt, 32, nf);
                    for q in 0..nf {
                        out[base + q][i] = flat[q * 32 + i];
                    }
                }
            }
            // hybrid_analysis_ileave(out + 27, L, 5, len)
            for i in 5..64 {
                for j in 0..len {
                    out[27 + i][j] = [l[at(0, j, i)], l[at(1, j, i)]];
                }
            }
        } else {
            // hybrid6_cx
            for i in 0..len {
                let inp = self.in_buf[0][i..i + 13].to_vec();
                let mut temp = [[0f32; 2]; 8];
                hybrid_analysis(&mut temp, 0, &inp, &t::F20_0_8, 1, 8);
                out[0][i] = temp[6];
                out[1][i] = temp[7];
                out[2][i] = temp[0];
                out[3][i] = temp[1];
                out[4][i] = [temp[2][0] + temp[5][0], temp[2][1] + temp[5][1]];
                out[5][i] = [temp[3][0] + temp[4][0], temp[3][1] + temp[4][1]];
            }
            // hybrid2_re
            const G1_Q2: [f32; 7] = [0.0, 0.01899487526049, 0.0, -0.07293139167538, 0.0, 0.30596630545168, 0.5];
            for (band, base, reverse) in [(1usize, 6usize, 1usize), (2, 8, 0)] {
                for i in 0..len {
                    let inp = &self.in_buf[band][i..i + 13];
                    let re_in = G1_Q2[6] * inp[6][0];
                    let im_in = G1_Q2[6] * inp[6][1];
                    let (mut re_op, mut im_op) = (0f32, 0f32);
                    for j in (0..6).step_by(2) {
                        re_op += G1_Q2[j + 1] * (inp[j + 1][0] + inp[12 - j - 1][0]);
                        im_op += G1_Q2[j + 1] * (inp[j + 1][1] + inp[12 - j - 1][1]);
                    }
                    out[base + reverse][i] = [re_in + re_op, im_in + im_op];
                    out[base + 1 - reverse][i] = [re_in - re_op, im_in - im_op];
                }
            }
            // hybrid_analysis_ileave(out + 7, L, 3, len)
            for i in 3..64 {
                for j in 0..len {
                    out[7 + i][j] = [l[at(0, j, i)], l[at(1, j, i)]];
                }
            }
        }
        for i in 0..5 {
            let tail: [[f32; 2]; 6] = self.in_buf[i][32..38].try_into().unwrap();
            self.in_buf[i][..6].copy_from_slice(&tail);
        }
    }

    /// `hybrid_synthesis` of `src` into `out` (`[2][38][64]`).
    fn hybrid_synthesis(out: &mut [f32], src: &[[[f32; 2]; 32]], is34: bool, len: usize) {
        let at = |ri: usize, n: usize, i: usize| (ri * 38 + n) * 64 + i;
        if is34 {
            for n in 0..len {
                for k in 0..5 {
                    out[at(0, n, k)] = 0.0;
                    out[at(1, n, k)] = 0.0;
                }
                for (dst, lo, hi) in [(0usize, 0usize, 12usize), (1, 12, 20)] {
                    for i in lo..hi {
                        out[at(0, n, dst)] += src[i][n][0];
                        out[at(1, n, dst)] += src[i][n][1];
                    }
                }
                for i in 0..4 {
                    out[at(0, n, 2)] += src[20 + i][n][0];
                    out[at(1, n, 2)] += src[20 + i][n][1];
                    out[at(0, n, 3)] += src[24 + i][n][0];
                    out[at(1, n, 3)] += src[24 + i][n][1];
                    out[at(0, n, 4)] += src[28 + i][n][0];
                    out[at(1, n, 4)] += src[28 + i][n][1];
                }
            }
            // hybrid_synthesis_deint(out, in + 27, 5, len)
            for i in 5..64 {
                for n in 0..len {
                    out[at(0, n, i)] = src[i + 27][n][0];
                    out[at(1, n, i)] = src[i + 27][n][1];
                }
            }
        } else {
            for n in 0..len {
                for c in 0..2 {
                    out[at(c, n, 0)] = src[0][n][c] + src[1][n][c] + src[2][n][c] + src[3][n][c] + src[4][n][c] + src[5][n][c];
                    out[at(c, n, 1)] = src[6][n][c] + src[7][n][c];
                    out[at(c, n, 2)] = src[8][n][c] + src[9][n][c];
                }
            }
            // hybrid_synthesis_deint(out, in + 7, 3, len)
            for i in 3..64 {
                for n in 0..len {
                    out[at(0, n, i)] = src[i + 7][n][0];
                    out[at(1, n, i)] = src[i + 7][n][1];
                }
            }
        }
    }

    /// `decorrelation`: `rbuf` from `lbuf`.
    fn decorrelation(&mut self, is34: bool) {
        let mut power = [[0f32; 32]; 34];
        let mut transient_gain = [[0f32; 32]; 34];
        let k_to_i: &[u8] = if is34 { &t::K_TO_I_34 } else { &t::K_TO_I_20 };
        let ix = is34 as usize;
        let peak_decay_factor = 0.76592833836465f32;
        let (transient_impact, a_smooth) = (1.5f32, 0.25f32);
        if is34 != self.common.is34bands_old {
            self.peak_decay_nrg = [0.0; 34];
            self.power_smooth = [0.0; 34];
            self.peak_decay_diff_smooth = [0.0; 34];
            for d in self.delay.iter_mut() {
                *d = [[0.0; 2]; QMF_SLOTS + MAX_DELAY];
            }
            for d in self.ap_delay.iter_mut() {
                *d = [[[0.0; 2]; QMF_SLOTS + MAX_AP_DELAY]; AP_LINKS];
            }
        }
        let s = &self.lbuf;
        for k in 0..NR_BANDS[ix] {
            let i = k_to_i[k] as usize;
            // ff_ps_add_squares_neon: re*re and im*im rounded, then summed.
            for n in 0..32 {
                let sq = s[k][n][0] * s[k][n][0] + s[k][n][1] * s[k][n][1];
                power[i][n] += sq;
            }
        }
        for i in 0..NR_PAR_BANDS[ix] {
            for n in 0..32 {
                let decayed_peak = peak_decay_factor * self.peak_decay_nrg[i];
                self.peak_decay_nrg[i] = if decayed_peak > power[i][n] { decayed_peak } else { power[i][n] };
                self.power_smooth[i] += a_smooth * (power[i][n] - self.power_smooth[i]);
                self.peak_decay_diff_smooth[i] += a_smooth * (self.peak_decay_nrg[i] - power[i][n] - self.peak_decay_diff_smooth[i]);
                let denom = transient_impact * self.peak_decay_diff_smooth[i];
                transient_gain[i][n] = if denom > self.power_smooth[i] { self.power_smooth[i] / denom } else { 1.0 };
            }
        }
        const A: [f32; 3] = [0.65143905753106, 0.56471812200776, 0.48954165955695];
        let mut k = 0;
        while k < NR_ALLPASS_BANDS[ix] {
            let b = k_to_i[k] as usize;
            let mut g_decay_slope = 1.0f32 - 0.05f32 * (k as i32 - DECAY_CUTOFF[ix]) as f32;
            g_decay_slope = g_decay_slope.clamp(0.0, 1.0);
            let d = &mut self.delay[k];
            d.copy_within(32..32 + MAX_DELAY, 0);
            d[MAX_DELAY..MAX_DELAY + 32].copy_from_slice(&s[k]);
            for m in 0..AP_LINKS {
                self.ap_delay[k][m].copy_within(32..37, 0);
            }
            // ps_decorrelate_c
            let phi = [f(t::PHI_FRACT[(ix * 50 + k) * 2]), f(t::PHI_FRACT[(ix * 50 + k) * 2 + 1])];
            let q = |m: usize, c: usize| f(t::Q_FRACT_ALLPASS[((ix * 50 + k) * 3 + m) * 2 + c]);
            let ag: [f32; 3] = std::array::from_fn(|m| A[m] * g_decay_slope);
            let delay = &self.delay[k][MAX_DELAY - 2..];
            let apd = &mut self.ap_delay[k];
            let out = &mut self.rbuf[k];
            for n in 0..32 {
                let mut in_re = delay[n][0] * phi[0] - delay[n][1] * phi[1];
                let mut in_im = delay[n][0] * phi[1] + delay[n][1] * phi[0];
                for m in 0..AP_LINKS {
                    let a_re = ag[m] * in_re;
                    let a_im = ag[m] * in_im;
                    let ld_re = apd[m][n + 2 - m][0];
                    let ld_im = apd[m][n + 2 - m][1];
                    let (fd_re, fd_im) = (q(m, 0), q(m, 1));
                    let (apd_re, apd_im) = (in_re, in_im);
                    in_re = ld_re * fd_re - ld_im * fd_im;
                    in_re -= a_re;
                    in_im = ld_re * fd_im + ld_im * fd_re;
                    in_im -= a_im;
                    apd[m][n + 5][0] = apd_re + ag[m] * in_re;
                    apd[m][n + 5][1] = apd_im + ag[m] * in_im;
                }
                out[n] = [transient_gain[b][n] * in_re, transient_gain[b][n] * in_im];
            }
            k += 1;
        }
        while k < NR_BANDS[ix] {
            let i = k_to_i[k] as usize;
            let d = &mut self.delay[k];
            d.copy_within(32..32 + MAX_DELAY, 0);
            d[MAX_DELAY..MAX_DELAY + 32].copy_from_slice(&s[k]);
            let off = if k < SHORT_DELAY_BAND[ix] { MAX_DELAY - 14 } else { MAX_DELAY - 1 };
            for n in 0..32 {
                let v = self.delay[k][off + n];
                self.rbuf[k][n] = [v[0] * transient_gain[i][n], v[1] * transient_gain[i][n]];
            }
            k += 1;
        }
    }

    /// `stereo_processing` on `lbuf` / `rbuf`.
    fn stereo_processing(&mut self, is34: bool) {
        let ps2 = self.common.clone();
        let ix = is34 as usize;
        let k_to_i: &[u8] = if is34 { &t::K_TO_I_34 } else { &t::K_TO_I_20 };
        let lut = if ps2.icc_mode < 3 { &t::HA } else { &t::HB };
        if ps2.num_env_old != 0 {
            for hh in self.h.iter_mut() {
                for half in hh.iter_mut() {
                    half[0] = half[ps2.num_env_old];
                }
            }
        }
        let iid_mapped = remap(&ps2.iid_par, ps2.nr_iid_par, ps2.num_env, true, is34);
        let icc_mapped = remap(&ps2.icc_par, ps2.nr_icc_par, ps2.num_env, true, is34);
        let (ipd_mapped, opd_mapped) = if ps2.enable_ipdopd {
            (remap(&ps2.ipd_par, ps2.nr_ipdopd_par, ps2.num_env, false, is34), remap(&ps2.opd_par, ps2.nr_ipdopd_par, ps2.num_env, false, is34))
        } else {
            ([[0i8; 34]; 5], [[0i8; 34]; 5])
        };
        if is34 && !ps2.is34bands_old {
            for hh in self.h.iter_mut() {
                for half in hh.iter_mut() {
                    map_val_20_to_34(&mut half[0]);
                }
            }
            self.ipd_hist = [0; 34];
            self.opd_hist = [0; 34];
        } else if !is34 && ps2.is34bands_old {
            for hh in self.h.iter_mut() {
                for half in hh.iter_mut() {
                    map_val_34_to_20(&mut half[0]);
                }
            }
            self.ipd_hist = [0; 34];
            self.opd_hist = [0; 34];
        }
        for e in 0..ps2.num_env {
            for b in 0..NR_PAR_BANDS[ix] {
                let row = (iid_mapped[e][b] as i32 + 7 + 23 * ps2.iid_quant as i32) as usize;
                let col = icc_mapped[e][b] as usize;
                let hv = |c: usize| f(lut[(row * 8 + col) * 4 + c]);
                let (mut h11, mut h12, mut h21, mut h22) = (hv(0), hv(1), hv(2), hv(3));
                if ps2.enable_ipdopd && b < NR_IPDOPD_BANDS[ix] {
                    let opd_idx = (self.opd_hist[b] as i32 * 8 + opd_mapped[e][b] as i32) as usize;
                    let ipd_idx = (self.ipd_hist[b] as i32 * 8 + ipd_mapped[e][b] as i32) as usize;
                    let (opd_re, opd_im) = (f(t::PD_RE_SMOOTH[opd_idx]), f(t::PD_IM_SMOOTH[opd_idx]));
                    let (ipd_re, ipd_im) = (f(t::PD_RE_SMOOTH[ipd_idx]), f(t::PD_IM_SMOOTH[ipd_idx]));
                    self.opd_hist[b] = (opd_idx & 0x3f) as i8;
                    self.ipd_hist[b] = (ipd_idx & 0x3f) as i8;
                    let ipd_adj_re = opd_re * ipd_re + opd_im * ipd_im;
                    let ipd_adj_im = opd_im * ipd_re - opd_re * ipd_im;
                    let h11i = h11 * opd_im;
                    h11 *= opd_re;
                    let h12i = h12 * ipd_adj_im;
                    h12 *= ipd_adj_re;
                    let h21i = h21 * opd_im;
                    h21 *= opd_re;
                    let h22i = h22 * ipd_adj_im;
                    h22 *= ipd_adj_re;
                    self.h[0][1][e + 1][b] = h11i;
                    self.h[1][1][e + 1][b] = h12i;
                    self.h[2][1][e + 1][b] = h21i;
                    self.h[3][1][e + 1][b] = h22i;
                }
                self.h[0][0][e + 1][b] = h11;
                self.h[1][0][e + 1][b] = h12;
                self.h[2][0][e + 1][b] = h21;
                self.h[3][0][e + 1][b] = h22;
            }
            for k in 0..NR_BANDS[ix] {
                let start = ps2.border_position[e];
                let stop = ps2.border_position[e + 1];
                let width = 1.0f32 / (if stop - start != 0 { stop - start } else { 1 }) as f32;
                let b = k_to_i[k] as usize;
                let mut h = [[0f32; 4]; 2];
                let mut hs = [[0f32; 4]; 2];
                for c in 0..4 {
                    h[0][c] = self.h[c][0][e][b];
                }
                if ps2.enable_ipdopd {
                    let flip = (is34 && (9..=13).contains(&k)) || (!is34 && k <= 1);
                    for c in 0..4 {
                        h[1][c] = if flip { -self.h[c][1][e][b] } else { self.h[c][1][e][b] };
                    }
                }
                for c in 0..4 {
                    hs[0][c] = (self.h[c][0][e + 1][b] - h[0][c]) * width;
                }
                if ps2.enable_ipdopd {
                    for c in 0..4 {
                        hs[1][c] = (self.h[c][1][e + 1][b] - h[1][c]) * width;
                    }
                }
                if stop - start != 0 {
                    let s = (1 + start) as usize;
                    let n = (stop - start) as usize;
                    let (l, r) = (&mut self.lbuf[k][s..s + n], &mut self.rbuf[k][s..s + n]);
                    if ps2.enable_ipdopd {
                        stereo_interpolate_ipdopd(l, r, &h, &hs, n);
                    } else {
                        stereo_interpolate(l, r, &h, &hs, n);
                    }
                }
            }
        }
    }

    /// `ff_ps_apply` on `l` = `X[0]`, `r` = `X[1]` (`[2][38][64]` each).
    pub(crate) fn apply(&mut self, l: &mut [f32], r: &mut [f32], top: usize) {
        let is34 = self.common.is34bands;
        let ix = is34 as usize;
        let top = top + NR_BANDS[ix] - 64;
        for d in &mut self.delay[top.min(MAX_SSB)..NR_BANDS[ix]] {
            *d = [[0.0; 2]; QMF_SLOTS + MAX_DELAY];
        }
        if top < NR_ALLPASS_BANDS[ix] {
            for d in &mut self.ap_delay[top..NR_ALLPASS_BANDS[ix]] {
                *d = [[[0.0; 2]; QMF_SLOTS + MAX_AP_DELAY]; AP_LINKS];
            }
        }
        self.hybrid_analysis(l, is34, 32);
        self.decorrelation(is34);
        self.stereo_processing(is34);
        let lb = std::mem::take(&mut self.lbuf);
        Self::hybrid_synthesis(l, &lb, is34, 32);
        self.lbuf = lb;
        let rb = std::mem::take(&mut self.rbuf);
        Self::hybrid_synthesis(r, &rb, is34, 32);
        self.rbuf = rb;
    }
}
