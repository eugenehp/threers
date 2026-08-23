//! Compressed intra HEVC (transform-coded, not `I_PCM`).
//!
//! Stage 1 (this file, verified): each 16×16 CTB is one intra `PART_2Nx2N` CU
//! with luma mode chosen by SAD and DM chroma; all `cbf = 0`, so the
//! reconstruction is the intra prediction itself (lossy, but *real* compression —
//! far smaller than raw, and decodable by any HEVC decoder). This exercises the
//! whole coding path — CU/transform-tree syntax, MPM mode coding, reference-
//! sample construction, and the reconstruction loop — with only a handful of
//! CABAC contexts. Stage 2 adds residual coefficient coding for quality.
//!
//! Verified by decoding with ffmpeg and comparing to the reconstruction this
//! encoder returns (a conformant decoder must reproduce it bit-for-bit).

use crate::codec::bitstream::BitWriter;
use crate::codec::hevc::cabac::{CabacEncoder, CtxModel};
use crate::codec::hevc::encoder::pad_plane;
use crate::codec::hevc::nal::{nal_unit_base, push_annexb, NalUnitType};
use crate::codec::hevc::params::{write_pps, write_sps, write_vps, HevcConfig, CTB_SIZE};
use crate::codec::hevc::residual::ResidualCtx;
use crate::codec::hevc::transform::{forward, inverse, TransformKind};
use crate::codec::hevc::{intra, quant, residual, Yuv420Frame};

/// A compressed intra HEVC encoder. Transform-codes intra frames to a decodable
/// Annex-B stream far smaller than raw.
///
/// Two modes:
/// - **prediction-only** (default, `residual(false)`): every `cbf = 0`, so the
///   reconstruction is the intra prediction (DC/planar/angular with the
///   boundary-smoothing filters). Lossy, tiny, and **conformant** — ffmpeg
///   decodes it bit-for-bit (see `tests/hevc_compress.rs`).
/// - **residual** (`residual(true)`, *experimental*): also codes quantized
///   transform coefficients ([`crate::codec::hevc::residual`]) for
///   fidelity. The coefficient coder is roundtrip-verified and its context tables
///   match the HM reference; the luma path decodes conformantly, but some chroma
///   / smooth-content cases are still being reconciled with the reference
///   decoder, so it is not yet the default.
pub struct CompressedEncoder {
    cfg: HevcConfig,
    headers_emitted: bool,
    residual: bool,
}

impl CompressedEncoder {
    /// New encoder for `width × height` at quantization `qp`.
    pub fn new(width: u32, height: u32, qp: i32) -> Self {
        let mut cfg = HevcConfig::new(width, height)
            .with_pcm(false)
            .with_level(180);
        cfg.qp = qp;
        Self {
            cfg,
            headers_emitted: false,
            residual: false,
        }
    }

    /// Enable transform-coefficient residual coding (experimental; see the type
    /// docs). Default is the conformant prediction-only path.
    pub fn residual(mut self, on: bool) -> Self {
        self.residual = on;
        self
    }

    /// The active configuration.
    pub fn config(&self) -> &HevcConfig {
        &self.cfg
    }

    /// Encode one 4:2:0 frame → Annex-B access unit, plus the reconstruction a
    /// conformant decoder reproduces. The first call prepends VPS/SPS/PPS.
    pub fn encode_frame(&mut self, frame: &Yuv420Frame) -> (Vec<u8>, Reconstruction) {
        assert_eq!(
            (frame.width, frame.height),
            (self.cfg.width, self.cfg.height),
            "frame size"
        );
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);

        let (slice, recon) = compressed_slice_rbsp(&self.cfg, &y, &u, &v, self.residual);
        let mut au = Vec::new();
        if !self.headers_emitted {
            push_annexb(&mut au, &nal_unit_base(NalUnitType::Vps, &write_vps(false)));
            push_annexb(
                &mut au,
                &nal_unit_base(NalUnitType::Sps, &write_sps(&self.cfg)),
            );
            push_annexb(&mut au, &nal_unit_base(NalUnitType::Pps, &write_pps()));
            self.headers_emitted = true;
        }
        push_annexb(&mut au, &nal_unit_base(NalUnitType::IdrNLp, &slice));
        (au, recon)
    }
}

/// CABAC contexts for the compressed intra path (I-slice init values, §9.3.2.2).
struct Ctx {
    part_mode: CtxModel,
    prev_intra: CtxModel,
    chroma_mode: CtxModel,
    cbf_luma: [CtxModel; 2],
    cbf_chroma: [CtxModel; 4],
    res: ResidualCtx,
}

impl Ctx {
    fn new(qp: i32) -> Self {
        Self {
            part_mode: CtxModel::init(184, qp),
            prev_intra: CtxModel::init(184, qp),
            chroma_mode: CtxModel::init(63, qp),
            cbf_luma: [CtxModel::init(111, qp), CtxModel::init(141, qp)],
            cbf_chroma: [
                CtxModel::init(94, qp),
                CtxModel::init(138, qp),
                CtxModel::init(182, qp),
                CtxModel::init(154, qp),
            ],
            res: ResidualCtx::new(qp),
        }
    }
}

/// Chroma QP from luma QP (4:2:0, offset 0) — the §8.6.1 mapping. Identity below 30.
fn chroma_qp(qp_luma: i32) -> i32 {
    let qpi = qp_luma.clamp(0, 51);
    if qpi < 30 {
        qpi
    } else {
        const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        if qpi <= 43 {
            T[(qpi - 30) as usize]
        } else {
            qpi - 6
        }
    }
}

/// A reconstructed planar picture the decoder must reproduce.
pub struct Reconstruction {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub coded_width: u32,
    pub coded_height: u32,
}

/// Encode the picture as one compressed IDR I-slice; returns the slice RBSP plus
/// the reconstruction (for verification / as neighbor source across frames).
pub fn compressed_slice_rbsp(
    cfg: &HevcConfig,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    residual: bool,
) -> (Vec<u8>, Reconstruction) {
    let cw = cfg.coded_width;
    let ch = cfg.coded_height;
    let (cwc, chc) = (cw / 2, ch / 2);
    let qp_l = cfg.qp;
    let qp_c = chroma_qp(cfg.qp);

    // ---- slice_segment_header() ----
    let mut h = BitWriter::new();
    h.flag(true); // first_slice_segment_in_pic_flag
    h.flag(false); // no_output_of_prior_pics_flag
    h.write_ue(0); // slice_pic_parameter_set_id
    h.write_ue(2); // slice_type = I
    h.write_se(cfg.qp - 26); // slice_qp_delta (init_qp_minus26 = 0)
    h.write_bit(1); // byte_alignment: alignment_bit_equal_to_one
    let mut rbsp = h.finish();

    // Reconstruction buffers (grown as CTBs are coded, read for neighbors).
    let mut ry = vec![0u8; (cw * ch) as usize];
    let mut ru = vec![128u8; (cwc * chc) as usize];
    let mut rv = vec![128u8; (cwc * chc) as usize];

    let mut cabac = CabacEncoder::new();
    let mut ctx = Ctx::new(cfg.qp);
    // Per-CTB chosen luma mode, for MPM derivation of later CTBs.
    let (nx, ny) = cfg.ctbs();
    let mut modes = vec![255u8; (nx * ny) as usize]; // 255 = unavailable/not-intra
    let total = nx * ny;
    let mut coded = 0u32;

    for cby in 0..ny {
        for cbx in 0..nx {
            let (px, py) = (cbx * CTB_SIZE, cby * CTB_SIZE);

            let ln = CTB_SIZE as usize; // luma TU 16
            let cn = (CTB_SIZE / 2) as usize; // chroma TU 8

            // --- luma: choose mode, predict, quantize the residual ---
            let lvals = refs_luma(&ry, cw, ch, px, py, CTB_SIZE, cbx, cby, nx);
            let orig_y = gather(y, cw, px, py, CTB_SIZE);
            let mode = best_luma_mode(&orig_y, ln, &lvals);
            let (la, ll) = extract(&lvals, ln, mode, true);
            let pred_y = intra::predict(mode, ln, &la, &ll, true);
            let resid_y: Vec<i32> = (0..ln * ln).map(|i| orig_y[i] - pred_y[i]).collect();
            let levels_y = quant::quant(&forward(&resid_y, ln, TransformKind::Dct), ln, qp_l, true);
            let cbf_luma = residual && levels_y.iter().any(|&v| v != 0);

            // --- chroma (DM): predict + quantize residual for Cb then Cr ---
            let (cpx, cpy) = (px / 2, py / 2);
            let cmode = chroma_mode_from_luma(mode);
            let mut c_pred = [Vec::new(), Vec::new()];
            let mut c_levels: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
            let mut c_cbf = [false, false];
            for (ci, (plane, rp)) in [(u, &ru), (v, &rv)].iter().enumerate() {
                let cvals = refs_chroma(rp, cwc, chc, cpx, cpy, CTB_SIZE / 2, cbx, cby, nx);
                let (ca, cl) = extract(&cvals, cn, cmode, false);
                let pred = intra::predict(cmode, cn, &ca, &cl, false);
                let orig = gather(plane, cwc, cpx, cpy, CTB_SIZE / 2);
                let resid: Vec<i32> = (0..cn * cn).map(|k| orig[k] - pred[k]).collect();
                let lv = quant::quant(&forward(&resid, cn, TransformKind::Dct), cn, qp_c, true);
                c_cbf[ci] = residual && lv.iter().any(|&v| v != 0);
                c_pred[ci] = pred;
                c_levels[ci] = lv;
            }

            // --- coding_unit + transform_tree syntax ---
            cabac.encode_bin(&mut ctx.part_mode, 1); // part_mode = PART_2Nx2N
            code_luma_mode(&mut cabac, &mut ctx, mode, cbx, cby, nx, &modes);
            cabac.encode_bin(&mut ctx.chroma_mode, 0); // intra_chroma_pred_mode = DM
            cabac.encode_bin(&mut ctx.cbf_chroma[0], c_cbf[0] as u32); // cbf_cb
            cabac.encode_bin(&mut ctx.cbf_chroma[0], c_cbf[1] as u32); // cbf_cr
            cabac.encode_bin(&mut ctx.cbf_luma[1], cbf_luma as u32); // cbf_luma
                                                                     // transform_unit: residual_coding luma, Cb, Cr (each if cbf set).
            if cbf_luma {
                residual::encode(&mut cabac, &mut ctx.res, &levels_y, ln, false, 0);
            }
            if c_cbf[0] {
                residual::encode(&mut cabac, &mut ctx.res, &c_levels[0], cn, true, 0);
            }
            if c_cbf[1] {
                residual::encode(&mut cabac, &mut ctx.res, &c_levels[1], cn, true, 0);
            }

            // --- reconstruct = prediction + dequant→inverse residual ---
            let mut recon_y = pred_y;
            if cbf_luma {
                let res = inverse(&quant::dequant(&levels_y, ln, qp_l), ln, TransformKind::Dct);
                for i in 0..ln * ln {
                    recon_y[i] += res[i];
                }
            }
            store(&mut ry, cw, px, py, CTB_SIZE, &recon_y);
            for ci in 0..2 {
                let mut recon = std::mem::take(&mut c_pred[ci]);
                if c_cbf[ci] {
                    let res = inverse(
                        &quant::dequant(&c_levels[ci], cn, qp_c),
                        cn,
                        TransformKind::Dct,
                    );
                    for k in 0..cn * cn {
                        recon[k] += res[k];
                    }
                }
                let rp = if ci == 0 { &mut ru } else { &mut rv };
                store(rp, cwc, cpx, cpy, CTB_SIZE / 2, &recon);
            }

            modes[(cby * nx + cbx) as usize] = mode;
            coded += 1;
            cabac.encode_terminate((coded == total) as u32);
        }
    }

    rbsp.extend_from_slice(&cabac.finish());
    (
        rbsp,
        Reconstruction {
            y: ry,
            u: ru,
            v: rv,
            coded_width: cw,
            coded_height: ch,
        },
    )
}

/// Chroma mode for DM_CHROMA in 4:2:0 (identity — no 4:2:2 remap needed).
fn chroma_mode_from_luma(luma: u8) -> u8 {
    luma
}

/// Try planar/DC and every angular mode (each with its own smoothing); return the
/// one with least SAD against the original.
fn best_luma_mode(orig: &[i32], n: usize, vals: &[i32]) -> u8 {
    let mut best = (u32::MAX, intra::DC);
    for mode in 0..=34u8 {
        let (above, left) = extract(vals, n, mode, true);
        let pred = intra::predict(mode, n, &above, &left, true);
        let sad: u32 = orig
            .iter()
            .zip(&pred)
            .map(|(&o, &p)| (o - p).unsigned_abs())
            .sum();
        if sad < best.0 {
            best = (sad, mode);
        }
    }
    best.1
}

/// Code the chosen luma mode via the MPM list (§8.4.2, §9.3).
fn code_luma_mode(
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    mode: u8,
    cbx: u32,
    cby: u32,
    nx: u32,
    modes: &[u8],
) {
    let cand_a = neighbor_mode(cbx.checked_sub(1).map(|x| (x, cby)), nx, modes);
    // Above neighbor only counts within the same CTB row region → here, if it's a
    // separate CTB above, spec uses DC. We conservatively use DC for "above".
    let cand_b = 1u8; // DC (above is a different CTB above the current row)

    let list = mpm_list(cand_a, cand_b);
    if let Some(idx) = list.iter().position(|&m| m == mode) {
        cabac.encode_bin(&mut ctx.prev_intra, 1);
        // mpm_idx: truncated unary, cMax 2, bypass.
        match idx {
            0 => cabac.encode_bypass(0),
            1 => {
                cabac.encode_bypass(1);
                cabac.encode_bypass(0);
            }
            _ => {
                cabac.encode_bypass(1);
                cabac.encode_bypass(1);
            }
        }
    } else {
        cabac.encode_bin(&mut ctx.prev_intra, 0);
        let mut sorted = list;
        sorted.sort_unstable();
        let mut rem = mode as i32;
        for &c in sorted.iter().rev() {
            if rem > c as i32 {
                rem -= 1;
            }
        }
        cabac.encode_bypass_bits(rem as u32, 5); // rem_intra_luma_pred_mode, FL(5)
    }
}

fn neighbor_mode(pos: Option<(u32, u32)>, nx: u32, modes: &[u8]) -> u8 {
    match pos {
        Some((x, y)) => {
            let m = modes[(y * nx + x) as usize];
            if m == 255 {
                1
            } else {
                m
            } // DC if unavailable
        }
        None => 1,
    }
}

/// Build the 3-entry most-probable-mode list (§8.4.2).
fn mpm_list(a: u8, b: u8) -> [u8; 3] {
    if a == b {
        if a < 2 {
            [intra::PLANAR, intra::DC, 26]
        } else {
            [
                a,
                2 + ((a as i32 + 29) % 32) as u8,
                2 + ((a as i32 - 2 + 1) % 32) as u8,
            ]
        }
    } else {
        let mut l = [a, b, 0];
        l[2] = if a != intra::PLANAR && b != intra::PLANAR {
            intra::PLANAR
        } else if a != intra::DC && b != intra::DC {
            intra::DC
        } else {
            26
        };
        l
    }
}

// ---- reference samples ----

/// Is luma/chroma sample (px, py) already reconstructed? True iff its CTB precedes
/// the current one in raster order and it is inside the coded picture.
fn available(px: i32, py: i32, w: u32, h: u32, ctb: u32, cbx: u32, cby: u32, nx: u32) -> bool {
    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
        return false;
    }
    let (bx, by) = (px as u32 / ctb, py as u32 / ctb);
    let cur = cby * nx + cbx;
    let nb = by * nx + bx;
    nb < cur
}

/// Build the substituted reference chain `vals[0..=4n]` in the order
/// bottom-left → left → corner → top → top-right (§8.4.4.2.2). Smoothing and
/// above/left extraction happen later (they depend on the chosen mode).
#[allow(clippy::too_many_arguments)]
fn refs_generic(
    recon: &[u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    n: u32,
    ctb: u32,
    cbx: u32,
    cby: u32,
    nx: u32,
) -> Vec<i32> {
    let ni = n as i32;
    let (xi, yi) = (x as i32, y as i32);
    let sample = |px: i32, py: i32| recon[(py as u32 * w + px as u32) as usize] as i32;

    let mut vals = vec![0i32; (4 * n + 1) as usize];
    let mut avail = vec![false; (4 * n + 1) as usize];
    for k in 0..(2 * ni) {
        let py = yi + (2 * ni - 1 - k); // left column p[-1][2n-1-k]
        if available(xi - 1, py, w, h, ctb, cbx, cby, nx) {
            vals[k as usize] = sample(xi - 1, py);
            avail[k as usize] = true;
        }
    }
    if available(xi - 1, yi - 1, w, h, ctb, cbx, cby, nx) {
        vals[(2 * ni) as usize] = sample(xi - 1, yi - 1); // corner
        avail[(2 * ni) as usize] = true;
    }
    for k in 0..(2 * ni) {
        let px = xi + k; // top row p[k][-1]
        if available(px, yi - 1, w, h, ctb, cbx, cby, nx) {
            vals[(2 * ni + 1 + k) as usize] = sample(px, yi - 1);
            avail[(2 * ni + 1 + k) as usize] = true;
        }
    }

    let len = vals.len();
    if avail.iter().all(|&a| !a) {
        vals.iter_mut().for_each(|v| *v = 128);
    } else {
        if !avail[0] {
            let first = (0..len).find(|&i| avail[i]).unwrap();
            vals[0] = vals[first];
        }
        for i in 1..len {
            if !avail[i] {
                vals[i] = vals[i - 1];
            }
        }
    }
    vals
}

#[allow(clippy::too_many_arguments)]
fn refs_luma(
    recon: &[u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    n: u32,
    cbx: u32,
    cby: u32,
    nx: u32,
) -> Vec<i32> {
    refs_generic(recon, w, h, x, y, n, CTB_SIZE, cbx, cby, nx)
}

#[allow(clippy::too_many_arguments)]
fn refs_chroma(
    recon: &[u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    n: u32,
    cbx: u32,
    cby: u32,
    nx: u32,
) -> Vec<i32> {
    refs_generic(recon, w, h, x, y, n, CTB_SIZE / 2, cbx, cby, nx)
}

/// Reference-sample smoothing decision (§8.4.4.2.3): the `[1 2 1]` filter applies
/// for larger blocks and modes far from horizontal/vertical.
fn filter_flag(mode: u8, n: usize) -> bool {
    if mode == intra::DC || n == 4 {
        return false;
    }
    let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
    let thres = match n {
        8 => 7,
        16 => 1,
        32 => 0,
        _ => 8,
    };
    min_dist > thres
}

/// Apply mode-dependent smoothing to the reference chain, then split it into the
/// `above`/`left` arrays the predictor consumes. Reference-sample smoothing is
/// **luma-only** in 4:2:0 (ChromaArrayType != 3, §8.4.4.2.1): chroma reference
/// samples are never filtered, so `luma` gates the `[1 2 1]` filter.
fn extract(vals: &[i32], n: usize, mode: u8, luma: bool) -> (Vec<i32>, Vec<i32>) {
    let corner = 2 * n;
    let smoothed;
    let src: &[i32] = if luma && filter_flag(mode, n) {
        let mut f = vals.to_vec();
        let last = vals.len() - 1;
        for i in 1..last {
            f[i] = (vals[i - 1] + 2 * vals[i] + vals[i + 1] + 2) >> 2;
        }
        smoothed = f;
        &smoothed
    } else {
        vals
    };
    let mut above = vec![0i32; 2 * n + 1];
    let mut left = vec![0i32; 2 * n + 1];
    above[0] = src[corner];
    left[0] = src[corner];
    for k in 1..=2 * n {
        above[k] = src[corner + k];
        left[k] = src[corner - k];
    }
    (above, left)
}

// ---- block helpers ----

fn gather(plane: &[u8], w: u32, x: u32, y: u32, n: u32) -> Vec<i32> {
    let mut b = vec![0i32; (n * n) as usize];
    for yy in 0..n {
        for xx in 0..n {
            b[(yy * n + xx) as usize] = plane[((y + yy) * w + (x + xx)) as usize] as i32;
        }
    }
    b
}

fn store(plane: &mut [u8], w: u32, x: u32, y: u32, n: u32, block: &[i32]) {
    for yy in 0..n {
        for xx in 0..n {
            plane[((y + yy) * w + (x + xx)) as usize] =
                block[(yy * n + xx) as usize].clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpm_list_cases() {
        // Both neighbors DC → planar/DC/vertical fallback.
        assert_eq!(mpm_list(1, 1), [intra::PLANAR, intra::DC, 26]);
        // Distinct non-planar/non-DC → third is planar.
        assert_eq!(mpm_list(10, 26), [10, 26, intra::PLANAR]);
        // Distinct incl. planar → third is DC.
        assert_eq!(mpm_list(intra::PLANAR, 26), [0, 26, intra::DC]);
        // Equal angular → derived neighbors.
        let l = mpm_list(20, 20);
        assert_eq!(l[0], 20);
        assert!(l[1] >= 2 && l[2] >= 2 && l[1] != l[2]);
    }

    #[test]
    fn filter_flag_rules() {
        assert!(!filter_flag(intra::DC, 16)); // DC never filtered
        assert!(!filter_flag(10, 4)); // 4x4 never filtered
        assert!(!filter_flag(26, 16)); // pure vertical: minDist 0 <= 1
        assert!(filter_flag(intra::PLANAR, 16)); // planar minDist 10 > 1
        assert!(filter_flag(18, 16)); // diagonal filtered
    }

    #[test]
    fn rem_mode_roundtrips_through_mpm() {
        // Encoding a non-MPM mode and decoding rem must recover it.
        let list = mpm_list(1, 1);
        let mut sorted = list;
        sorted.sort_unstable();
        for mode in 0..=34u8 {
            if list.contains(&mode) {
                continue;
            }
            let mut rem = mode as i32;
            for &c in sorted.iter().rev() {
                if rem > c as i32 {
                    rem -= 1;
                }
            }
            // Decoder inverse.
            let mut dec = rem;
            for &c in sorted.iter() {
                if dec >= c as i32 {
                    dec += 1;
                }
            }
            assert_eq!(dec as u8, mode, "rem roundtrip for mode {mode}");
        }
    }
}
