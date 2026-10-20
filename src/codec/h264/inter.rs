//! H.264 with P-frames: an IDR picture every so often, and between them
//! pictures predicted from the one before.
//!
//! [`CompressedEncoder`](super::CompressedEncoder) codes every picture on its
//! own, which keeps encoding parallel per frame but spends the same bits on a
//! static background in every frame. Here the pictures between keyframes are
//! P slices: each 16×16 macroblock is either *skipped* (copied from the
//! previous picture along the predicted motion vector) or predicted from it
//! along a searched motion vector, with only the residual coded.
//!
//! * Keyframes are [`super::compress::compressed_slice_rbsp`]
//!   IDR pictures — CABAC, deblocked — under PPS 0.
//! * P pictures are `P_L0_16x16` / `P_Skip` macroblocks, one reference
//!   picture, the deblocking filter off: CABAC under PPS 0 (the default), or
//!   CAVLC under PPS 1.
//! * Motion vectors are in quarter luma samples ([`super::interp`]'s 6-tap
//!   interpolation); the search is whole samples (a diamond),
//!   then half and quarter samples by SATD, on several threads, one band of
//!   macroblock rows each.
//! * A picture that does not match the one before it (a cut) is coded as a
//!   keyframe instead.
//!
//! The encoder's own reconstruction is what a decoder reproduces, sample for
//! sample: `tests/h264_inter.rs` checks it against threers' decoder and ffmpeg.

use crate::codec::bitstream::{emulation_prevention, rbsp_trailing_bits, BitWriter};
use crate::codec::h264::cabac::{self as cabac_syntax, Cabac};
use crate::codec::h264::cavlc::{self, BlockKind, ZIGZAG4};
use crate::codec::h264::compress::{compressed_slice_rbsp, Reconstruction};
use crate::codec::h264::nal::{nal_unit_base, NalUnitType};
use crate::codec::h264::params::{write_sps_with_refs, EntropyMode, H264Config, MB_SIZE};
use crate::codec::h264::pps::write_pps_with_id;
use crate::codec::h264::transform::{chroma_dc_forward, chroma_dc_inverse, dequant4, forward4, inverse4, quant4};
use crate::codec::h264::interp::HalfPlanes;
use crate::codec::hevc::encoder::{pad_plane, Yuv420Frame};

impl Reference {
    fn new(rec: &Reconstruction, mvs: Vec<[i16; 2]>) -> Reference {
        let threads = if cfg!(target_arch = "wasm32") { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()).min(16) };
        let halves = HalfPlanes::new(&rec.y, rec.coded_width as usize, rec.coded_height as usize, threads);
        Reference { y: rec.y.clone(), u: rec.u.clone(), v: rec.v.clone(), halves, mvs }
    }
}

/// Table 9-4, inter column: code number → `coded_block_pattern`.
const CBP_INTER_FROM_CODE: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45,
    46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

const fn cbp_inter_code() -> [u32; 48] {
    let mut out = [0u32; 48];
    let mut i = 0;
    while i < 48 {
        out[CBP_INTER_FROM_CODE[i] as usize] = i as u32;
        i += 1;
    }
    out
}
const CBP_INTER_CODE: [u32; 48] = cbp_inter_code();

/// The 4×4 luma blocks in bitstream order (Table 6-3), in 4-sample units.
const BLK_POS: [(usize, usize); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 2), (3, 2), (2, 3), (3, 3),
];

/// Chroma QP from luma QP (Table 8-15), `chroma_qp_index_offset = 0`.
fn chroma_qp(qp: i32) -> i32 {
    const T: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];
    let q = qp.clamp(0, 51);
    if q < 30 {
        q
    } else {
        T[(q - 30) as usize]
    }
}

/// The SSD Lagrangian the intra encoder uses, and its square root for SAD.
fn lambda(qp: i32) -> f64 {
    0.85 * (f64::from(qp - 12) / 3.0).exp2()
}

/// Bits in a `se(v)` code.
fn se_bits(v: i32) -> u32 {
    let code = if v > 0 { 2 * v as u32 - 1 } else { 2 * (-v) as u32 };
    2 * (32 - (code + 1).leading_zeros() - 1) + 1
}

/// One encoded picture.
#[derive(Debug, Clone)]
pub struct EncodedPicture {
    /// The access unit as an MP4 sample: 4-byte length-prefixed NAL units.
    pub sample: Vec<u8>,
    /// An IDR picture: decoding can start here.
    pub keyframe: bool,
}

/// A reference picture, at coded (macroblock-aligned) size.
struct Reference {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// The luma's half-sample planes, for quarter-sample prediction.
    halves: HalfPlanes,
    /// Each macroblock's motion vector (quarter samples), for the next
    /// picture's search to start from.
    mvs: Vec<[i16; 2]>,
}

/// H.264 with P-frames. See the [module docs](self).
pub struct InterEncoder {
    cfg: H264Config,
    qp: i32,
    sps: Vec<u8>,
    pps_intra: Vec<u8>,
    pps_inter: Vec<u8>,
    keyframe_interval: u32,
    entropy: EntropyMode,
    scene_cut: f64,
    search_range: i32,
    threads: usize,
    reference: Option<Reference>,
    /// Pictures since the last keyframe (the keyframe is 0).
    since_key: u32,
    force_key: bool,
    last: Option<Reconstruction>,
    /// The last P picture's mean absolute difference after motion search.
    last_mad: Option<f64>,
}

impl InterEncoder {
    /// `width × height` at quantizer `qp` (lower is better and bigger; 18–40
    /// is the useful range).
    pub fn new(width: u32, height: u32, qp: i32) -> Self {
        let mut cfg = H264Config::new(width, height);
        cfg.qp = qp.clamp(0, 51);
        cfg.entropy = EntropyMode::Cabac;
        cfg.deblock = true;
        let sps = nal_unit_base(NalUnitType::Sps, &write_sps_with_refs(&cfg, 1));
        let pps_intra = nal_unit_base(NalUnitType::Pps, &write_pps_with_id(0, true));
        let pps_inter = nal_unit_base(NalUnitType::Pps, &write_pps_with_id(1, false));
        let threads = if cfg!(target_arch = "wasm32") {
            1
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get()).min(16)
        };
        InterEncoder {
            qp: cfg.qp,
            cfg,
            sps,
            pps_intra,
            pps_inter,
            keyframe_interval: 50,
            entropy: EntropyMode::Cabac,
            scene_cut: 28.0,
            search_range: 48,
            threads,
            reference: None,
            since_key: 0,
            force_key: false,
            last: None,
            last_mad: None,
        }
    }

    /// A keyframe at least every `n` pictures (default 50: two seconds at
    /// 25 fps). Seeking lands on keyframes, so shorter is faster to seek and
    /// bigger. `1` is all keyframes.
    pub fn keyframe_interval(mut self, n: u32) -> Self {
        self.keyframe_interval = n.max(1);
        self
    }

    /// The P slices' entropy coder: CABAC (the default; Main profile, as the
    /// keyframes already are) or CAVLC, about 10–15 % larger.
    pub fn entropy(mut self, mode: EntropyMode) -> Self {
        self.entropy = mode;
        self
    }

    /// Mean absolute difference per luma sample, after motion search, above
    /// which a picture is taken to be a cut and coded as a keyframe
    /// (default 28). A jump to three times the previous picture's (and over
    /// 12) is a cut too. `f64::INFINITY` turns the absolute test off.
    pub fn scene_cut_threshold(mut self, mad: f64) -> Self {
        self.scene_cut = mad;
        self
    }

    /// Motion search range in luma samples, each way (default 48).
    pub fn search_range(mut self, samples: i32) -> Self {
        self.search_range = samples.clamp(1, 512);
        self
    }

    /// Threads for motion search (default: the machine's cores, up to 16).
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n.max(1);
        self
    }

    /// Make the next picture a keyframe.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    pub fn config(&self) -> &H264Config {
        &self.cfg
    }

    pub fn sps(&self) -> &[u8] {
        &self.sps
    }

    /// Both picture parameter sets: the intra one (id 0), the inter one (id 1).
    pub fn pps(&self) -> [&[u8]; 2] {
        [&self.pps_intra, &self.pps_inter]
    }

    /// The `avcC` decoder configuration, with the SPS and both PPSs.
    pub fn avcc(&self) -> Vec<u8> {
        let mut out = vec![1, self.sps[1], self.sps[2], self.sps[3], 0xff, 0xe1];
        out.extend_from_slice(&(self.sps.len() as u16).to_be_bytes());
        out.extend_from_slice(&self.sps);
        out.push(2);
        for pps in [&self.pps_intra, &self.pps_inter] {
            out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            out.extend_from_slice(pps);
        }
        out
    }

    /// The picture a decoder reconstructs from the last [`encode`](Self::encode),
    /// at coded size.
    pub fn reconstruction(&self) -> Option<&Reconstruction> {
        self.last.as_ref()
    }

    /// Encode the next picture, in display order.
    pub fn encode(&mut self, frame: &Yuv420Frame) -> EncodedPicture {
        assert_eq!((frame.width, frame.height), (self.cfg.width, self.cfg.height), "frame size differs from encoder size");
        let (cw, ch) = (self.cfg.coded_width, self.cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);

        let due = self.force_key || self.reference.is_none() || self.since_key + 1 >= self.keyframe_interval;
        if !due {
            let reference = self.reference.as_ref().unwrap();
            let search = motion_search(&y, reference, cw, ch, self.qp, self.search_range, self.threads);
            let mad = search.iter().map(|m| m.sad as f64).sum::<f64>() / (cw * ch) as f64;
            // A cut: worse than the threshold, or a sudden jump from how
            // well the last pictures matched.
            let jump = self.last_mad.is_some_and(|prev| mad > 12.0 && mad > 3.0 * prev.max(2.0));
            if mad <= self.scene_cut && !jump {
                self.last_mad = Some(mad);
                let frame_num = (self.since_key + 1) % 16;
                let (rbsp, rec, mvs) = p_slice(&self.cfg, self.entropy, self.qp, frame_num, &y, &u, &v, reference, &search);
                self.since_key += 1;
                let nal = nal(1, 2, &rbsp);
                self.reference = Some(Reference::new(&rec, mvs));
                self.last = Some(rec);
                return EncodedPicture { sample: length_prefixed(&[&nal]), keyframe: false };
            }
        }
        self.force_key = false;
        self.since_key = 0;
        self.last_mad = None;
        let (rbsp, rec) = compressed_slice_rbsp(&self.cfg, &y, &u, &v, self.qp);
        let mbs = (cw / MB_SIZE * (ch / MB_SIZE)) as usize;
        self.reference = Some(Reference::new(&rec, vec![[0, 0]; mbs]));
        self.last = Some(rec);
        let nal = nal_unit_base(NalUnitType::Idr, &rbsp);
        EncodedPicture { sample: length_prefixed(&[&nal]), keyframe: true }
    }
}

fn nal(kind: u8, ref_idc: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![(ref_idc << 5) | kind];
    out.extend_from_slice(&emulation_prevention(rbsp));
    out
}

fn length_prefixed(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for n in nals {
        out.extend_from_slice(&(n.len() as u32).to_be_bytes());
        out.extend_from_slice(n);
    }
    out
}

// ── Motion search ─────────────────────────────────────────────────────────────

/// One macroblock's search result: the best whole-sample vector (quarter
/// units) and its luma SAD.
#[derive(Clone, Copy, Debug)]
struct Searched {
    mv: [i16; 2],
    sad: u32,
}

/// 16×16 SAD of `orig` at `(x, y)` against `refp` displaced by whole samples
/// `(dx, dy)`, edges clamped as a decoder does.
fn sad16(orig: &[u8], refp: &[u8], w: u32, h: u32, x: u32, y: u32, dx: i32, dy: i32, limit: u32) -> u32 {
    let (rx, ry) = (x as i32 + dx, y as i32 + dy);
    let inside = rx >= 0 && ry >= 0 && rx + 16 <= w as i32 && ry + 16 <= h as i32;
    let mut sad = 0u32;
    for j in 0..16 {
        let o = &orig[((y + j) * w + x) as usize..][..16];
        if inside {
            let r = &refp[((ry as u32 + j) * w + rx as u32) as usize..][..16];
            sad += o.iter().zip(r).map(|(a, b)| a.abs_diff(*b) as u32).sum::<u32>();
        } else {
            let yy = (ry + j as i32).clamp(0, h as i32 - 1) as u32;
            for (i, a) in o.iter().enumerate() {
                let xx = (rx + i as i32).clamp(0, w as i32 - 1) as u32;
                sad += a.abs_diff(refp[(yy * w + xx) as usize]) as u32;
            }
        }
        if sad >= limit {
            return sad;
        }
    }
    sad
}

/// Sum of absolute 4×4 Hadamard-transformed differences over a 16×16 block.
fn satd16(orig: &[i32; 256], pred: &[i32; 256]) -> u32 {
    let mut total = 0u32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut d = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let k = (by * 4 + j) * 16 + bx * 4 + i;
                    d[j * 4 + i] = orig[k] - pred[k];
                }
            }
            let mut t = [0i32; 16];
            for j in 0..4 {
                let (a, b, c, e) = (d[j * 4], d[j * 4 + 1], d[j * 4 + 2], d[j * 4 + 3]);
                let (s0, s1, d0, d1) = (a + e, b + c, a - e, b - c);
                t[j * 4] = s0 + s1;
                t[j * 4 + 1] = d0 + d1;
                t[j * 4 + 2] = s0 - s1;
                t[j * 4 + 3] = d0 - d1;
            }
            for i in 0..4 {
                let (a, b, c, e) = (t[i], t[4 + i], t[8 + i], t[12 + i]);
                let (s0, s1, d0, d1) = (a + e, b + c, a - e, b - c);
                total += ((s0 + s1).abs() + (d0 + d1).abs() + (s0 - s1).abs() + (d0 - d1).abs()) as u32;
            }
        }
    }
    total / 2
}

/// Search every macroblock against the reference, bands of rows in parallel.
/// The cost adds a rate term for the vector's distance from the same
/// macroblock's vector in the previous picture (the spatial predictor is not
/// known until the picture is coded in order).
fn motion_search(y: &[u8], r: &Reference, cw: u32, ch: u32, qp: i32, range: i32, threads: usize) -> Vec<Searched> {
    let (mbw, mbh) = ((cw / 16) as usize, (ch / 16) as usize);
    let lambda_sad = lambda(qp).sqrt();
    let search_mb = |mbx: usize, mby: usize| -> Searched {
        let (x, y0) = ((mbx * 16) as u32, (mby * 16) as u32);
        let at = |m: [i32; 2], limit: u32| sad16(y, &r.y, cw, ch, x, y0, m[0], m[1], limit);
        let prev = r.mvs[mby * mbw + mbx];
        let pred = [prev[0] as i32 / 4, prev[1] as i32 / 4];
        let cost = |m: [i32; 2], sad: u32| sad as f64 + lambda_sad * (se_bits((m[0] - pred[0]) * 4) + se_bits((m[1] - pred[1]) * 4)) as f64;
        let mut cands = vec![[0, 0], pred];
        for (dx, dy) in [(-1i32, 0i32), (0, -1), (1, 0), (0, 1)] {
            let (nx, ny) = (mbx as i32 + dx, mby as i32 + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < mbw && (ny as usize) < mbh {
                let m = r.mvs[ny as usize * mbw + nx as usize];
                cands.push([m[0] as i32 / 4, m[1] as i32 / 4]);
            }
        }
        let (mut best, mut best_sad) = ([0, 0], u32::MAX);
        let mut best_cost = f64::INFINITY;
        for c in cands {
            let s = at(c, u32::MAX);
            let k = cost(c, s);
            if k < best_cost {
                (best, best_sad, best_cost) = (c, s, k);
            }
        }
        // Diamond search, coarse to fine.
        for step in [8, 4, 2, 1] {
            for _ in 0..12 {
                let mut moved = false;
                for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                    let m = [best[0] + dx, best[1] + dy];
                    if m[0].abs() > range || m[1].abs() > range {
                        continue;
                    }
                    let s = at(m, best_sad.saturating_add(256));
                    let k = cost(m, s);
                    if k < best_cost {
                        (best, best_sad, best_cost) = (m, s, k);
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        // Half then quarter samples around the best whole-sample vector,
        // judged by SATD — which tracks the residual's coded size better
        // than SAD where interpolation smooths the prediction.
        let orig: [i32; 256] = gather(y, cw, x, y0, 16);
        let pred_q = [pred[0] * 4, pred[1] * 4];
        let qcost = |m: [i32; 2], satd: u32| satd as f64 + lambda_sad * (se_bits(m[0] - pred_q[0]) + se_bits(m[1] - pred_q[1])) as f64;
        let mut q = [best[0] * 4, best[1] * 4];
        let mut q_satd = satd16(&orig, &predict_luma(r, cw, ch, x, y0, [q[0] as i16, q[1] as i16]));
        let mut q_cost = qcost(q, q_satd);
        for step in [2, 1] {
            let centre = q;
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let m = [centre[0] + dx * step, centre[1] + dy * step];
                if m[0].abs() > range * 4 + 3 || m[1].abs() > range * 4 + 3 {
                    continue;
                }
                let s = satd16(&orig, &predict_luma(r, cw, ch, x, y0, [m[0] as i16, m[1] as i16]));
                let k = qcost(m, s);
                if k < q_cost {
                    (q, q_satd, q_cost) = (m, s, k);
                }
            }
        }
        let _ = q_satd;
        Searched { mv: [q[0] as i16, q[1] as i16], sad: best_sad }
    };
    let mut out = vec![Searched { mv: [0, 0], sad: 0 }; mbw * mbh];
    let rows_each = mbh.div_ceil(threads.max(1));
    if threads <= 1 || mbh < 2 {
        for (i, o) in out.iter_mut().enumerate() {
            *o = search_mb(i % mbw, i / mbw);
        }
        return out;
    }
    std::thread::scope(|scope| {
        for (band, chunk) in out.chunks_mut(rows_each * mbw).enumerate() {
            let search_mb = &search_mb;
            scope.spawn(move || {
                for (i, o) in chunk.iter_mut().enumerate() {
                    let idx = band * rows_each * mbw + i;
                    *o = search_mb(idx % mbw, idx / mbw);
                }
            });
        }
    });
    out
}

// ── Prediction ────────────────────────────────────────────────────────────────

/// Luma prediction at a quarter-sample vector (§8.4.2.2.1): whole-sample
/// vectors copy, fractional ones are interpolated ([`super::interp`]).
fn predict_luma(r: &Reference, w: u32, h: u32, x: u32, y: u32, mv: [i16; 2]) -> [i32; 256] {
    if mv[0] & 3 != 0 || mv[1] & 3 != 0 {
        let mut out = [0i32; 256];
        let (fx, fy) = ((mv[0] & 3) as usize, (mv[1] & 3) as usize);
        r.halves.block(x as isize + (mv[0] as isize >> 2), y as isize + (mv[1] as isize >> 2), fx, fy, 16, 16, &mut out);
        return out;
    }
    let refp = &r.y;
    let (dx, dy) = (mv[0] as i32 >> 2, mv[1] as i32 >> 2);
    let mut out = [0i32; 256];
    let (rx, ry) = (x as i32 + dx, y as i32 + dy);
    if rx >= 0 && ry >= 0 && rx + 16 <= w as i32 && ry + 16 <= h as i32 {
        for j in 0..16 {
            let row = &refp[((ry as u32 + j) * w + rx as u32) as usize..][..16];
            for (o, &p) in out[(j * 16) as usize..][..16].iter_mut().zip(row) {
                *o = p as i32;
            }
        }
        return out;
    }
    for j in 0..16 {
        let yy = (y as i32 + j + dy).clamp(0, h as i32 - 1) as u32;
        for i in 0..16 {
            let xx = (x as i32 + i + dx).clamp(0, w as i32 - 1) as u32;
            out[(j * 16 + i) as usize] = refp[(yy * w + xx) as usize] as i32;
        }
    }
    out
}

/// Chroma prediction (§8.4.2.2.2): the luma vector is in eighth chroma
/// samples, interpolated bilinearly.
fn predict_chroma(refp: &[u8], w: u32, h: u32, cx: u32, cy: u32, mv: [i16; 2]) -> [i32; 64] {
    let (mx, my) = (mv[0] as i32, mv[1] as i32);
    let (fx, fy) = (mx & 7, my & 7);
    let (ix, iy) = (cx as i32 + (mx >> 3), cy as i32 + (my >> 3));
    let mut out = [0i32; 64];
    if ix >= 0 && iy >= 0 && ix + 9 <= w as i32 && iy + 9 <= h as i32 {
        let (wa, wb, wc, wd) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
        for j in 0..8 {
            let r0 = &refp[((iy + j) as u32 * w + ix as u32) as usize..][..9];
            let r1 = &refp[((iy + j + 1) as u32 * w + ix as u32) as usize..][..9];
            for i in 0..8 {
                out[(j * 8 + i) as usize] =
                    (wa * r0[i as usize] as i32 + wb * r0[i as usize + 1] as i32 + wc * r1[i as usize] as i32 + wd * r1[i as usize + 1] as i32 + 32) >> 6;
            }
        }
        return out;
    }
    let at = |x: i32, y: i32| refp[(y.clamp(0, h as i32 - 1) as u32 * w + x.clamp(0, w as i32 - 1) as u32) as usize] as i32;
    for j in 0..8 {
        for i in 0..8 {
            let (x, y) = (ix + i, iy + j);
            let (a, b, c, d) = (at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1));
            out[(j * 8 + i) as usize] = ((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32) >> 6;
        }
    }
    out
}

// ── The P slice ───────────────────────────────────────────────────────────────

/// What a coded (not skipped) macroblock carries.
struct Coded {
    mv: [i16; 2],
    cbp: u32,
    /// Zig-zag luma levels per block, in bitstream order.
    luma: [[i32; 16]; 16],
    chroma_dc: [[i32; 4]; 2],
    chroma_ac: [[[i32; 15]; 4]; 2],
    rec_y: [i32; 256],
    rec_c: [[i32; 64]; 2],
}

/// Per-picture coding state.
struct State {
    cw: u32,
    mbw: usize,
    mbh: usize,
    nnz_y: Vec<usize>,
    nnz_c: [Vec<usize>; 2],
    /// Motion per macroblock once coded: `None` before it is.
    mv: Vec<Option<[i16; 2]>>,
    // What CABAC's contexts ask of neighbours, per macroblock and block.
    skip: Vec<bool>,
    cbp: Vec<u32>,
    /// |mvd| per component.
    mvd: Vec<[u32; 2]>,
    /// coded_block_flag per 4×4 luma block (picture grid), false where the
    /// block was not coded.
    cbf_y: Vec<bool>,
    cbf_c: [Vec<bool>; 2],
    cbf_dc: Vec<[bool; 2]>,
}

impl State {
    fn nb(&self, mbx: isize, mby: isize) -> Option<[i16; 2]> {
        if mbx < 0 || mby < 0 || mbx as usize >= self.mbw || mby as usize >= self.mbh {
            return None;
        }
        self.mv[mby as usize * self.mbw + mbx as usize]
    }

    /// The 16×16 partition's motion vector predictor (§8.4.1.3), every
    /// available neighbour being an inter macroblock on reference 0.
    fn mvp(&self, mbx: usize, mby: usize) -> [i16; 2] {
        let (x, y) = (mbx as isize, mby as isize);
        let a = self.nb(x - 1, y);
        let b = self.nb(x, y - 1);
        let c = self.nb(x + 1, y - 1).or_else(|| self.nb(x - 1, y - 1));
        if let (Some(a), None, None) = (a, b, c) {
            return a;
        }
        let avail = [a, b, c];
        if avail.iter().filter(|m| m.is_some()).count() == 1 {
            return avail.into_iter().flatten().next().unwrap();
        }
        let (a, b, c) = (a.unwrap_or([0, 0]), b.unwrap_or([0, 0]), c.unwrap_or([0, 0]));
        let med = |p: i16, q: i16, s: i16| p.max(q).min(p.min(q).max(s));
        [med(a[0], b[0], c[0]), med(a[1], b[1], c[1])]
    }

    /// `P_Skip`'s vector (§8.4.1.1).
    fn skip_mv(&self, mbx: usize, mby: usize) -> [i16; 2] {
        let (x, y) = (mbx as isize, mby as isize);
        match (self.nb(x - 1, y), self.nb(x, y - 1)) {
            (None, _) | (_, None) => [0, 0],
            (Some([0, 0]), _) | (_, Some([0, 0])) => [0, 0],
            _ => self.mvp(mbx, mby),
        }
    }

    /// A neighbouring macroblock that has been coded (or skipped).
    fn coded_nb(&self, mbx: isize, mby: isize) -> Option<usize> {
        if mbx < 0 || mby < 0 || mbx as usize >= self.mbw || mby as usize >= self.mbh {
            return None;
        }
        let i = mby as usize * self.mbw + mbx as usize;
        self.mv[i].is_some().then_some(i)
    }

    /// `mb_skip_flag`'s ctxIdxInc: neighbours that exist and are not skipped.
    fn skip_inc(&self, mbx: usize, mby: usize) -> usize {
        let (x, y) = (mbx as isize, mby as isize);
        [self.coded_nb(x - 1, y), self.coded_nb(x, y - 1)].iter().filter(|n| n.is_some_and(|i| !self.skip[i])).count()
    }

    fn commit_skip(&mut self, mbx: usize, mby: usize) {
        let i = mby * self.mbw + mbx;
        (self.skip[i], self.cbp[i], self.mvd[i], self.cbf_dc[i]) = (true, 0, [0, 0], [false; 2]);
        let gw = (self.cw / 4) as usize;
        for (bx, by) in BLK_POS {
            self.cbf_y[(mby * 4 + by) * gw + mbx * 4 + bx] = false;
        }
        let gwc = (self.cw / 8) as usize;
        for p in 0..2 {
            for b in 0..4usize {
                self.cbf_c[p][(mby * 2 + b / 2) * gwc + mbx * 2 + b % 2] = false;
            }
        }
    }

    fn commit_coded(&mut self, c: &Coded, mvp: [i16; 2], mbx: usize, mby: usize) {
        let i = mby * self.mbw + mbx;
        self.skip[i] = false;
        self.cbp[i] = c.cbp;
        self.mvd[i] = [(c.mv[0] - mvp[0]).unsigned_abs() as u32, (c.mv[1] - mvp[1]).unsigned_abs() as u32];
        let chroma = c.cbp >> 4;
        self.cbf_dc[i] = [0, 1].map(|p| chroma > 0 && c.chroma_dc[p].iter().any(|&l| l != 0));
        let gw = (self.cw / 4) as usize;
        for (b, (bx, by)) in BLK_POS.iter().enumerate() {
            self.cbf_y[(mby * 4 + by) * gw + mbx * 4 + bx] = c.cbp & (1 << (b / 4)) != 0 && c.luma[b].iter().any(|&l| l != 0);
        }
        let gwc = (self.cw / 8) as usize;
        for p in 0..2 {
            for b in 0..4usize {
                self.cbf_c[p][(mby * 2 + b / 2) * gwc + mbx * 2 + b % 2] = chroma == 2 && c.chroma_ac[p][b].iter().any(|&l| l != 0);
            }
        }
    }

    /// This macroblock's 16 luma and 2 × 4 chroma coefficient counts.
    fn save_counts(&self, mbx: usize, mby: usize) -> ([usize; 16], [[usize; 4]; 2]) {
        let (gw, gwc) = ((self.cw / 4) as usize, (self.cw / 8) as usize);
        let mut y = [0; 16];
        for (k, (bx, by)) in BLK_POS.iter().enumerate() {
            y[k] = self.nnz_y[(mby * 4 + by) * gw + mbx * 4 + bx];
        }
        let mut c = [[0; 4]; 2];
        for (p, cp) in c.iter_mut().enumerate() {
            for (b, v) in cp.iter_mut().enumerate() {
                *v = self.nnz_c[p][(mby * 2 + b / 2) * gwc + mbx * 2 + b % 2];
            }
        }
        (y, c)
    }

    fn restore_counts(&mut self, mbx: usize, mby: usize, saved: &([usize; 16], [[usize; 4]; 2])) {
        let (gw, gwc) = ((self.cw / 4) as usize, (self.cw / 8) as usize);
        for (k, (bx, by)) in BLK_POS.iter().enumerate() {
            self.nnz_y[(mby * 4 + by) * gw + mbx * 4 + bx] = saved.0[k];
        }
        for p in 0..2 {
            for b in 0..4 {
                self.nnz_c[p][(mby * 2 + b / 2) * gwc + mbx * 2 + b % 2] = saved.1[p][b];
            }
        }
    }

    fn luma_nc(&self, px: u32, py: u32) -> i32 {
        let gw = (self.cw / 4) as usize;
        let at = |x: u32, y: u32| self.nnz_y[(y / 4) as usize * gw + (x / 4) as usize];
        let left = (px >= 4).then(|| at(px - 4, py));
        let above = (py >= 4).then(|| at(px, py - 4));
        cavlc::derive_nc(left, above)
    }

    fn chroma_nc(&self, plane: usize, cx: u32, cy: u32) -> i32 {
        let gw = (self.cw / 8) as usize;
        let at = |x: u32, y: u32| self.nnz_c[plane][(y / 4) as usize * gw + (x / 4) as usize];
        let left = (cx >= 4).then(|| at(cx - 4, cy));
        let above = (cy >= 4).then(|| at(cx, cy - 4));
        cavlc::derive_nc(left, above)
    }
}

/// Transform, quantize and reconstruct a macroblock's residual against a
/// prediction. Inter rounding (a sixth of a step), and 8×8 luma blocks
/// holding a single ±1 are dropped: they cost more bits than they save.
fn code_residual(
    orig_y: &[i32; 256],
    pred_y: &[i32; 256],
    orig_c: &[[i32; 64]; 2],
    pred_c: &[[i32; 64]; 2],
    qp: i32,
    mv: [i16; 2],
) -> Coded {
    let qpc = chroma_qp(qp);
    let mut c = Coded { mv, cbp: 0, luma: [[0; 16]; 16], chroma_dc: [[0; 4]; 2], chroma_ac: [[[0; 15]; 4]; 2], rec_y: *pred_y, rec_c: *pred_c };
    let mut q4 = [[0i32; 16]; 16];
    for (b, &(bx, by)) in BLK_POS.iter().enumerate() {
        let mut r = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                let k = (by * 4 + j) * 16 + bx * 4 + i;
                r[j * 4 + i] = orig_y[k] - pred_y[k];
            }
        }
        q4[b] = quant4(&forward4(&r), qp, false);
    }
    for b8 in 0..4 {
        let blocks = &q4[b8 * 4..b8 * 4 + 4];
        let nonzero: Vec<i32> = blocks.iter().flatten().copied().filter(|&l| l != 0).collect();
        if nonzero.is_empty() || (nonzero.len() == 1 && nonzero[0].abs() == 1) {
            continue;
        }
        c.cbp |= 1 << b8;
        for b in b8 * 4..b8 * 4 + 4 {
            for s in 0..16 {
                c.luma[b][s] = q4[b][ZIGZAG4[s]];
            }
            let res = inverse4(&dequant4(&q4[b], qp));
            let (bx, by) = BLK_POS[b];
            for j in 0..4 {
                for i in 0..4 {
                    let k = (by * 4 + j) * 16 + bx * 4 + i;
                    c.rec_y[k] = (pred_y[k] + res[j * 4 + i]).clamp(0, 255);
                }
            }
        }
    }

    // Chroma: DC through the 2×2 Hadamard, AC per 4×4 — the intra encoder's
    // scheme with inter rounding.
    let mut chroma_cbp = 0;
    let mut qac = [[[0i32; 16]; 4]; 2];
    let mut dc_levels = [[0i32; 4]; 2];
    for p in 0..2 {
        let mut dc = [0i32; 4];
        for b in 0..4 {
            let (sx, sy) = ((b % 2) * 4, (b / 2) * 4);
            let mut r = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let k = (sy + j) * 8 + sx + i;
                    r[j * 4 + i] = orig_c[p][k] - pred_c[p][k];
                }
            }
            let co = forward4(&r);
            dc[b] = co[0];
            qac[p][b] = quant4(&co, qpc, false);
            qac[p][b][0] = 0;
        }
        let hdc = chroma_dc_forward(&dc);
        let m = (qpc % 6) as usize;
        let qbits = 15 + qpc / 6 + 1;
        let f = (1i64 << qbits) / 6;
        const MF0: [i64; 6] = [13107, 11916, 10082, 9362, 8192, 7282];
        for b in 0..4 {
            let l = ((hdc[b].abs() as i64) * MF0[m] + f) >> qbits;
            dc_levels[p][b] = if hdc[b] < 0 { -(l as i32) } else { l as i32 };
        }
        crate::codec::h264::transform::fit_chroma_dc_levels(&mut dc_levels[p], qpc);
        if qac[p].iter().any(|blk| blk.iter().any(|&l| l != 0)) {
            chroma_cbp = 2;
        } else if dc_levels[p].iter().any(|&l| l != 0) {
            chroma_cbp = chroma_cbp.max(1);
        }
    }
    for p in 0..2 {
        if chroma_cbp == 0 {
            break;
        }
        c.chroma_dc[p] = dc_levels[p];
        let dc_rec = chroma_dc_inverse(&dc_levels[p], qpc);
        for b in 0..4 {
            let ac = if chroma_cbp == 2 { qac[p][b] } else { [0; 16] };
            for s in 1..16 {
                c.chroma_ac[p][b][s - 1] = ac[ZIGZAG4[s]];
            }
            let mut deq = dequant4(&ac, qpc);
            deq[0] = dc_rec[b];
            let res = inverse4(&deq);
            let (sx, sy) = ((b % 2) * 4, (b / 2) * 4);
            for j in 0..4 {
                for i in 0..4 {
                    let k = (sy + j) * 8 + sx + i;
                    c.rec_c[p][k] = (pred_c[p][k] + res[j * 4 + i]).clamp(0, 255);
                }
            }
        }
    }
    c.cbp |= chroma_cbp << 4;
    c
}

/// Write a coded macroblock's `macroblock_layer` (after its `mb_skip_run`),
/// updating the coefficient counts the next blocks' contexts read. Returns
/// the bits written.
fn write_coded(w: &mut BitWriter, st: &mut State, c: &Coded, mvp: [i16; 2], mbx: usize, mby: usize) -> usize {
    let start = w.bit_len();
    w.write_ue(0); // mb_type P_L0_16x16
    w.write_se((c.mv[0] - mvp[0]) as i32);
    w.write_se((c.mv[1] - mvp[1]) as i32);
    w.write_ue(CBP_INTER_CODE[c.cbp as usize]);
    if c.cbp != 0 {
        w.write_se(0); // mb_qp_delta
    }
    let gw = (st.cw / 4) as usize;
    let (x0, y0) = ((mbx * 16) as u32, (mby * 16) as u32);
    for (b, &(bx, by)) in BLK_POS.iter().enumerate() {
        let (px, py) = (x0 + bx as u32 * 4, y0 + by as u32 * 4);
        let nnz = if c.cbp & (1 << (b / 4)) != 0 {
            let nc = st.luma_nc(px, py);
            cavlc::encode_block(w, &c.luma[b], BlockKind::Luma4x4, nc)
        } else {
            0
        };
        st.nnz_y[(py / 4) as usize * gw + (px / 4) as usize] = nnz;
    }
    let chroma = c.cbp >> 4;
    if chroma > 0 {
        for p in 0..2 {
            cavlc::encode_block(w, &c.chroma_dc[p], BlockKind::ChromaDc, -1);
        }
    }
    let gwc = (st.cw / 8) as usize;
    let (cx0, cy0) = ((mbx * 8) as u32, (mby * 8) as u32);
    for p in 0..2 {
        for b in 0..4 {
            let (cx, cy) = (cx0 + (b as u32 % 2) * 4, cy0 + (b as u32 / 2) * 4);
            let nnz = if chroma == 2 {
                let nc = st.chroma_nc(p, cx, cy);
                cavlc::encode_block(w, &c.chroma_ac[p][b], BlockKind::ChromaAc, nc)
            } else {
                0
            };
            st.nnz_c[p][(cy / 4) as usize * gwc + (cx / 4) as usize] = nnz;
        }
    }
    w.bit_len() - start
}

/// A coded macroblock's `macroblock_layer` in CABAC (after its
/// `mb_skip_flag`). Reads `st` for the neighbours; the macroblock's own
/// blocks are tracked locally, so a trial write leaves `st` as it was.
fn write_coded_cabac(c: &mut Cabac, st: &State, coded: &Coded, mvp: [i16; 2], mbx: usize, mby: usize) {
    let (x, y) = (mbx as isize, mby as isize);
    let left = st.coded_nb(x - 1, y);
    let above = st.coded_nb(x, y - 1);
    c.mb_type_p_16x16();
    for (comp, (&mv, &pred)) in coded.mv.iter().zip(&mvp).enumerate() {
        let abs = |n: Option<usize>| n.map_or(0, |i| st.mvd[i][comp]);
        c.mvd(comp, (mv - pred) as i32, abs(left) + abs(above));
    }

    // coded_block_pattern (§9.3.3.1.1.4): a luma 8×8's context counts
    // neighbouring 8×8s that exist and are empty (skipped ones included).
    let cbp_l = coded.cbp & 15;
    let chroma = coded.cbp >> 4;
    let luma_inc: [usize; 4] = std::array::from_fn(|b8| {
        let empty = |n: Option<usize>, bit: usize| n.is_some_and(|i| self_empty(st, i, bit));
        let a = if b8 % 2 == 1 { cbp_l & (1 << (b8 - 1)) == 0 } else { empty(left, b8 + 1) };
        let b = if b8 >= 2 { cbp_l & (1 << (b8 - 2)) == 0 } else { empty(above, b8 + 2) };
        usize::from(a) + 2 * usize::from(b)
    });
    let chroma_has = |n: Option<usize>, two: bool| n.is_some_and(|i| !st.skip[i] && if two { st.cbp[i] >> 4 == 2 } else { st.cbp[i] >> 4 != 0 });
    let cin0 = usize::from(chroma_has(left, false)) + 2 * usize::from(chroma_has(above, false));
    let cin1 = usize::from(chroma_has(left, true)) + 2 * usize::from(chroma_has(above, true));
    c.coded_block_pattern(cbp_l, chroma, luma_inc, cin0, cin1);
    if coded.cbp != 0 {
        c.mb_qp_delta(0, false);
    }

    // Luma: coded_block_flag's context from the left and above 4×4 blocks,
    // inside this macroblock or in its neighbours.
    let gw = (st.cw / 4) as usize;
    let mut local = [false; 16];
    let pos_blk = |bx: usize, by: usize| BLK_POS.iter().position(|&p| p == (bx, by)).unwrap();
    for b in 0..16 {
        if cbp_l & (1 << (b / 4)) == 0 {
            continue;
        }
        let (bx, by) = BLK_POS[b];
        let (gx, gy) = (mbx * 4 + bx, mby * 4 + by);
        let nb = |inside: bool, lb: usize, gx: isize, gy: isize| {
            if inside {
                local[lb]
            } else if gx < 0 || gy < 0 {
                false
            } else {
                st.cbf_y[gy as usize * gw + gx as usize]
            }
        };
        let a = nb(bx > 0, if bx > 0 { pos_blk(bx - 1, by) } else { 0 }, gx as isize - 1, gy as isize);
        let bb = nb(by > 0, if by > 0 { pos_blk(bx, by - 1) } else { 0 }, gx as isize, gy as isize - 1);
        let cbf = coded.luma[b].iter().any(|&l| l != 0);
        c.coded_block_flag(cabac_syntax::CAT_LUMA_4X4, usize::from(a) + 2 * usize::from(bb), cbf);
        if cbf {
            c.residual_block(&coded.luma[b], cabac_syntax::CAT_LUMA_4X4);
        }
        local[b] = cbf;
    }
    if chroma > 0 {
        for p in 0..2 {
            let has = |n: Option<usize>| n.is_some_and(|i| st.cbf_dc[i][p]);
            let cbf = coded.chroma_dc[p].iter().any(|&l| l != 0);
            c.coded_block_flag(cabac_syntax::CAT_CHROMA_DC, usize::from(has(left)) + 2 * usize::from(has(above)), cbf);
            if cbf {
                c.residual_block(&coded.chroma_dc[p], cabac_syntax::CAT_CHROMA_DC);
            }
        }
    }
    if chroma == 2 {
        let gwc = (st.cw / 8) as usize;
        for p in 0..2 {
            let mut local = [false; 4];
            for b in 0..4usize {
                let (bx, by) = (b % 2, b / 2);
                let (gx, gy) = ((mbx * 2 + bx) as isize, (mby * 2 + by) as isize);
                let grid = |gx: isize, gy: isize| gx >= 0 && gy >= 0 && st.cbf_c[p][gy as usize * gwc + gx as usize];
                let a = if bx > 0 { local[b - 1] } else { grid(gx - 1, gy) };
                let bb = if by > 0 { local[b - 2] } else { grid(gx, gy - 1) };
                let cbf = coded.chroma_ac[p][b].iter().any(|&l| l != 0);
                c.coded_block_flag(cabac_syntax::CAT_CHROMA_AC, usize::from(a) + 2 * usize::from(bb), cbf);
                if cbf {
                    c.residual_block(&coded.chroma_ac[p][b], cabac_syntax::CAT_CHROMA_AC);
                }
                local[b] = cbf;
            }
        }
    }
}

/// Whether neighbour macroblock `i`'s 8×8 `bit` is empty for
/// `coded_block_pattern`'s context: skipped, or coded without it.
fn self_empty(st: &State, i: usize, bit: usize) -> bool {
    st.skip[i] || st.cbp[i] & (1 << bit) == 0
}

fn ssd<const N: usize>(a: &[i32; N], b: &[i32; N]) -> u64 {
    a.iter().zip(b).map(|(x, y)| ((x - y) * (x - y)) as u64).sum()
}

fn gather<const N: usize>(plane: &[u8], w: u32, x: u32, y: u32, size: u32) -> [i32; N] {
    let mut out = [0i32; N];
    for j in 0..size {
        for i in 0..size {
            out[(j * size + i) as usize] = plane[((y + j) * w + x + i) as usize] as i32;
        }
    }
    out
}

fn store<const N: usize>(plane: &mut [u8], w: u32, x: u32, y: u32, size: u32, block: &[i32; N]) {
    for j in 0..size {
        for i in 0..size {
            plane[((y + j) * w + x + i) as usize] = block[(j * size + i) as usize] as u8;
        }
    }
}

/// Encode one P slice covering the picture. Returns the RBSP, the
/// reconstruction, and the motion vectors per macroblock.
#[allow(clippy::too_many_arguments)]
fn p_slice(
    cfg: &H264Config,
    entropy: EntropyMode,
    qp: i32,
    frame_num: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    r: &Reference,
    search: &[Searched],
) -> (Vec<u8>, Reconstruction, Vec<[i16; 2]>) {
    let (cw, ch) = (cfg.coded_width, cfg.coded_height);
    let (cwc, chc) = (cw / 2, ch / 2);
    let (mbw, mbh) = ((cw / 16) as usize, (ch / 16) as usize);
    let lambda_ssd = lambda(qp);
    let lambda_sad = lambda_ssd.sqrt();

    let cabac = entropy == EntropyMode::Cabac;
    let mut w = BitWriter::new();
    w.write_ue(0); // first_mb_in_slice
    w.write_ue(5); // slice_type: P, every slice of the picture
    w.write_ue(if cabac { 0 } else { 1 }); // pic_parameter_set_id: CABAC 0, CAVLC 1
    w.write_bits(frame_num, 4); // frame_num (log2_max_frame_num = 4)
    w.flag(false); // num_ref_idx_active_override_flag
    w.flag(false); // ref_pic_list_modification_flag_l0
    w.flag(false); // adaptive_ref_pic_marking_mode_flag: sliding window
    if cabac {
        w.write_ue(0); // cabac_init_idc
    }
    w.write_se(qp - 26); // slice_qp_delta
    w.write_ue(1); // disable_deblocking_filter_idc: off
    // CABAC data starts byte-aligned, padded with ones (§7.3.4).
    let mut header = Vec::new();
    let mut cab: Option<Box<Cabac>> = None;
    if cabac {
        while !w.is_byte_aligned() {
            w.write_bit(1);
        }
        header = std::mem::take(&mut w).finish();
        cab = Some(Box::new(Cabac::new_p(qp)));
    }

    let mut st = State {
        cw,
        mbw,
        mbh,
        nnz_y: vec![0; (cw / 4 * (ch / 4)) as usize],
        nnz_c: [vec![0; (cwc / 4 * (chc / 4)) as usize], vec![0; (cwc / 4 * (chc / 4)) as usize]],
        mv: vec![None; mbw * mbh],
        skip: vec![false; mbw * mbh],
        cbp: vec![0; mbw * mbh],
        mvd: vec![[0; 2]; mbw * mbh],
        cbf_y: vec![false; (cw / 4 * (ch / 4)) as usize],
        cbf_c: [vec![false; (cwc / 4 * (chc / 4)) as usize], vec![false; (cwc / 4 * (chc / 4)) as usize]],
        cbf_dc: vec![[false; 2]; mbw * mbh],
    };
    let last_mb = mbw * mbh - 1;
    let (mut ry, mut ru, mut rv) = (vec![0u8; (cw * ch) as usize], vec![0u8; (cwc * chc) as usize], vec![0u8; (cwc * chc) as usize]);
    let mut skip_run = 0u32;

    for mby in 0..mbh {
        for mbx in 0..mbw {
            let i = mby * mbw + mbx;
            let (x, yy) = ((mbx * 16) as u32, (mby * 16) as u32);
            let (cx, cy) = ((mbx * 8) as u32, (mby * 8) as u32);
            let orig_y: [i32; 256] = gather(y, cw, x, yy, 16);
            let orig_c: [[i32; 64]; 2] = [gather(u, cwc, cx, cy, 8), gather(v, cwc, cx, cy, 8)];
            let mvp = st.mvp(mbx, mby);
            let skip_mv = st.skip_mv(mbx, mby);

            // The vector: the search's, the predictor, or the skip vector —
            // whichever is cheapest counting what its difference costs.
            let mut best = (search[i].mv, f64::INFINITY);
            for cand in [search[i].mv, mvp, skip_mv] {
                let p = predict_luma(r, cw, ch, x, yy, cand);
                let sad: u32 = orig_y.iter().zip(&p).map(|(a, b)| (a - b).unsigned_abs()).sum();
                let bits = se_bits((cand[0] - mvp[0]) as i32) + se_bits((cand[1] - mvp[1]) as i32);
                let cost = sad as f64 + lambda_sad * bits as f64;
                if cost < best.1 {
                    best = (cand, cost);
                }
            }
            let mv = best.0;

            // Skip: the prediction at the skip vector, nothing coded.
            let skip_y = predict_luma(r, cw, ch, x, yy, skip_mv);
            let skip_c = [predict_chroma(&r.u, cwc, chc, cx, cy, skip_mv), predict_chroma(&r.v, cwc, chc, cx, cy, skip_mv)];
            let skip_cost = (ssd(&orig_y, &skip_y) + ssd(&orig_c[0], &skip_c[0]) + ssd(&orig_c[1], &skip_c[1])) as f64 + lambda_ssd;

            let pred_y = if mv == skip_mv { skip_y } else { predict_luma(r, cw, ch, x, yy, mv) };
            let pred_c = if mv == skip_mv {
                skip_c
            } else {
                [predict_chroma(&r.u, cwc, chc, cx, cy, mv), predict_chroma(&r.v, cwc, chc, cx, cy, mv)]
            };
            let coded = code_residual(&orig_y, &pred_y, &orig_c, &pred_c, qp, mv);

            let take_skip = if coded.cbp == 0 && mv == skip_mv {
                true
            } else {
                // Exact bits of the coded choice, from a trial write — which
                // updates this macroblock's coefficient counts, so they are
                // put back after.
                let (bits, skip_bits) = match &cab {
                    Some(c) => {
                        let mut t = c.probe();
                        t.mb_skip_flag(false, st.skip_inc(mbx, mby));
                        write_coded_cabac(&mut t, &st, &coded, mvp, mbx, mby);
                        let mut s = c.probe();
                        s.mb_skip_flag(true, st.skip_inc(mbx, mby));
                        (t.bit_len(), s.bit_len())
                    }
                    None => {
                        let saved = st.save_counts(mbx, mby);
                        let mut trial = BitWriter::new();
                        // Coding it also ends the skip run, which costs ue(run).
                        let run_bits = 2 * (32 - (skip_run + 1).leading_zeros() - 1) as usize + 1;
                        let bits = run_bits + write_coded(&mut trial, &mut st, &coded, mvp, mbx, mby);
                        st.restore_counts(mbx, mby, &saved);
                        (bits, 1)
                    }
                };
                let coded_ssd = ssd(&orig_y, &coded.rec_y) + ssd(&orig_c[0], &coded.rec_c[0]) + ssd(&orig_c[1], &coded.rec_c[1]);
                skip_cost - lambda_ssd + lambda_ssd * skip_bits as f64 <= coded_ssd as f64 + lambda_ssd * bits as f64
            };

            if take_skip {
                if let Some(c) = cab.as_mut() {
                    c.mb_skip_flag(true, st.skip_inc(mbx, mby));
                } else {
                    skip_run += 1;
                }
                st.commit_skip(mbx, mby);
                st.mv[i] = Some(skip_mv);
                store(&mut ry, cw, x, yy, 16, &skip_y);
                store(&mut ru, cwc, cx, cy, 8, &skip_c[0]);
                store(&mut rv, cwc, cx, cy, 8, &skip_c[1]);
                // Its coefficient counts are zero, as they started.
                let gw = (cw / 4) as usize;
                for (bx, by) in BLK_POS {
                    st.nnz_y[(yy as usize / 4 + by) * gw + x as usize / 4 + bx] = 0;
                }
                let gwc = (cwc / 4) as usize;
                for p in 0..2 {
                    for b in 0..4usize {
                        st.nnz_c[p][(cy as usize / 4 + b / 2) * gwc + cx as usize / 4 + b % 2] = 0;
                    }
                }
            } else {
                if let Some(c) = cab.as_mut() {
                    c.mb_skip_flag(false, st.skip_inc(mbx, mby));
                    write_coded_cabac(c, &st, &coded, mvp, mbx, mby);
                    st.commit_coded(&coded, mvp, mbx, mby);
                } else {
                    w.write_ue(skip_run); // mb_skip_run
                    skip_run = 0;
                    write_coded(&mut w, &mut st, &coded, mvp, mbx, mby);
                }
                st.mv[i] = Some(coded.mv);
                store(&mut ry, cw, x, yy, 16, &coded.rec_y);
                store(&mut ru, cwc, cx, cy, 8, &coded.rec_c[0]);
                store(&mut rv, cwc, cx, cy, 8, &coded.rec_c[1]);
            }
            if let Some(c) = cab.as_mut() {
                c.terminate(u32::from(i == last_mb)); // end_of_slice_flag
            }
        }
    }
    let mvs = st.mv.iter().map(|m| m.unwrap_or([0, 0])).collect();
    let rec = Reconstruction { y: ry, u: ru, v: rv, coded_width: cw, coded_height: ch };
    if let Some(c) = cab {
        // The coder's flush ends with the rbsp_stop_one_bit (§9.3.4.3.5).
        header.extend_from_slice(&c.finish());
        return (header, rec, mvs);
    }
    if skip_run > 0 {
        w.write_ue(skip_run);
    }
    rbsp_trailing_bits(&mut w);
    (w.finish(), rec, mvs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inter_cbp_table_is_a_permutation() {
        let mut seen = [false; 48];
        for &c in CBP_INTER_FROM_CODE.iter() {
            assert!(!seen[c as usize]);
            seen[c as usize] = true;
        }
        for cbp in 0..48 {
            assert_eq!(CBP_INTER_FROM_CODE[CBP_INTER_CODE[cbp] as usize] as usize, cbp);
        }
    }

    #[test]
    fn se_bits_match_the_writer() {
        for v in -300..300 {
            let mut w = BitWriter::new();
            w.write_se(v);
            assert_eq!(se_bits(v) as usize, w.bit_len(), "{v}");
        }
    }
}
