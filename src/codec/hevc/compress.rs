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

use crate::codec::bitstream::{emulation_prevention, BitWriter};
use crate::codec::hevc::cabac::{CabacEncoder, CtxModel};
use crate::codec::hevc::encoder::pad_plane;
use crate::codec::hevc::nal::{nal_unit_base, push_annexb, NalUnitType};
use crate::codec::hevc::deblock::{deblock, EdgeMap};
use crate::codec::hevc::params::{
    write_pps_cu_qp_delta, write_pps_cu_qp_delta_no_deblock, write_pps_tiled, write_sps,
    write_vps, HevcConfig,
};
use crate::codec::hevc::residual::ResidualCtx;
use crate::codec::hevc::transform::{forward_into, inverse_into, TransformKind, MAX_N};
use crate::codec::hevc::hvcc::{build_hvcc, HvccArray, HvccProfile};
use crate::codec::hevc::{intra, quant, residual, Yuv420Frame};
use crate::codec::mp4::{mux_hevc, mux_hevc_with_captions, Mp4Params};
use crate::codec::rate::{Quality, Reach, RateControl, PROBE_QP};
use crate::captions::CaptionTrack;

/// A compressed intra HEVC encoder. Transform-codes intra frames to a decodable
/// Annex-B stream far smaller than raw.
///
/// Two modes:
/// - **prediction-only** (default, `residual(false)`): every `cbf = 0`, so the
///   reconstruction is the intra prediction (DC/planar/angular with the
///   boundary-smoothing filters). Lossy, tiny, and **conformant** — ffmpeg
///   decodes it bit-for-bit (see `tests/hevc_compress.rs`).
/// - **residual** (`residual(true)`, the default): also codes quantized transform
///   coefficients ([`crate::codec::hevc::residual`]), which is what makes the
///   output actually resemble the input. Externally conformant — ffmpeg decodes
///   it bit-for-bit to the returned reconstruction, on detailed content and at
///   every QP (`tests/hevc_residual_conformance.rs`).
///
/// Prediction-only is kept because it is a useful bisection tool: it exercises
/// the CU/mode/reference-sample path with no transform or coefficient coding, so
/// when a stream misdecodes, whether it also misdecodes here says which half to
/// look in.
pub struct CompressedEncoder {
    cfg: HevcConfig,
    headers_emitted: bool,
    residual: bool,
    aq_strength: f32,
    deblock: bool,
}

impl CompressedEncoder {
    /// New encoder for `width × height` at quantization `qp`.
    pub fn new(width: u32, height: u32, qp: i32) -> Self {
        let mut cfg = HevcConfig::new(width, height)
            .with_pcm(false)
            .with_level(180)
            // 64x64 coding tree blocks, a quadtree down to 8x8, transforms up
            // to 32x32. Every level is a rate-distortion choice, so a flat sky
            // pays one coding unit per 4096 samples while detail still gets an
            // 8x8 unit with its own prediction mode. The transform tree splits
            // again beneath that, down to 4x4 — where luma switches to DST-VII.
            .with_coding_tree(6, 3, 5)
            .with_transform_depth(3)
            // Large pictures get tiled so a single frame can use every core.
            // It engages at 4K and above, where it costs under a quarter of a
            // percent; below that the grid is 1x1 and nothing changes.
            .with_auto_tiles();
        cfg.qp = qp;
        Self {
            cfg,
            headers_emitted: false,
            residual: true,
            aq_strength: 0.0,
            deblock: true,
        }
    }

    /// Turn the in-loop deblocking filter on or off. On by default.
    pub fn deblock(mut self, on: bool) -> Self {
        self.deblock = on;
        self
    }

    /// Set the adaptive-quantization strength, in QP steps per octave of
    /// coding-tree-block variance. Off by default — measured on photographic
    /// content it costs perceptual quality at matched size, the same as on the
    /// H.264 side (see `docs/codec-benchmark.md`). Negative values invert the
    /// assignment, which is the control that measurement needs.
    pub fn aq(mut self, strength: f32) -> Self {
        self.aq_strength = strength;
        self
    }

    /// Enable transform-coefficient residual coding. On by default; pass `false`
    /// for the prediction-only path (see the type docs).
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

        let (slice, recon) =
            compressed_slice_rbsp(&self.cfg, &y, &u, &v, self.residual, self.aq_strength, self.deblock);
        let mut au = Vec::new();
        if !self.headers_emitted {
            push_annexb(&mut au, &nal_unit_base(NalUnitType::Vps, &write_vps(false)));
            push_annexb(
                &mut au,
                &nal_unit_base(NalUnitType::Sps, &write_sps(&self.cfg)),
            );
            push_annexb(
                &mut au,
                &nal_unit_base(
                    NalUnitType::Pps,
                    &self.pps_payload(),
                ),
            );
            self.headers_emitted = true;
        }
        push_annexb(&mut au, &nal_unit_base(NalUnitType::IdrNLp, &slice));
        (au, recon)
    }

    /// One frame's slice NAL, in Annex-B form, without parameter sets.
    ///
    /// Takes `&self`. That is the whole point: an all-intra stream carries
    /// nothing between frames — `compressed_slice_rbsp` reads the config and the
    /// pixels and nothing else — so frames can be encoded on separate threads
    /// and the result is byte-identical to encoding them in order.
    pub fn encode_slice_au(&self, frame: &Yuv420Frame) -> Vec<u8> {
        self.encode_slice_au_at(frame, self.cfg.qp)
    }

    /// [`encode_slice_au`](Self::encode_slice_au) at a quantizer chosen per
    /// frame, which is what rate control needs.
    ///
    /// Only the slice header moves: `init_qp_minus26` stays zero in the PPS and
    /// the frame's quantizer rides in `slice_qp_delta`, so the parameter sets
    /// are the same ones for every frame however the rate controller steers.
    pub fn encode_slice_au_at(&self, frame: &Yuv420Frame, qp: i32) -> Vec<u8> {
        let mut cfg = self.cfg;
        cfg.qp = qp.clamp(0, 51);
        self.encode_with(&cfg, frame)
    }

    fn encode_with(&self, cfg: &HevcConfig, frame: &Yuv420Frame) -> Vec<u8> {
        let (cw, ch) = (cfg.coded_width, cfg.coded_height);
        let y = pad_plane(&frame.y, frame.width, frame.height, cw, ch);
        let u = pad_plane(&frame.u, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let v = pad_plane(&frame.v, frame.width / 2, frame.height / 2, cw / 2, ch / 2);
        let (slice, _recon) =
            compressed_slice_rbsp(cfg, &y, &u, &v, self.residual, self.aq_strength, self.deblock);
        let mut au = Vec::new();
        push_annexb(&mut au, &nal_unit_base(NalUnitType::IdrNLp, &slice));
        au
    }

    fn pps_payload(&self) -> Vec<u8> {
        if self.cfg.tiles() > 1 {
            write_pps_tiled(self.cfg.tile_grid)
        } else if self.deblock {
            write_pps_cu_qp_delta()
        } else {
            write_pps_cu_qp_delta_no_deblock()
        }
    }

    /// Split the picture into `cols` x `rows` tiles, which can then be encoded
    /// concurrently. See [`HevcConfig::tile_grid`] for what it costs.
    pub fn tiles(mut self, cols: u32, rows: u32) -> Self {
        self.cfg = self.cfg.with_tiles(cols, rows);
        self
    }

    /// The parameter sets, as they go into `hvcC`.
    pub fn parameter_sets(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            nal_unit_base(NalUnitType::Vps, &write_vps(false)),
            nal_unit_base(NalUnitType::Sps, &write_sps(&self.cfg)),
            nal_unit_base(NalUnitType::Pps, &self.pps_payload()),
        )
    }
}

/// Encode `frames` as a compressed intra HEVC `.mp4` (`hvc1`) at quantization
/// `qp`, entirely in-process — no ffmpeg, and it builds for `wasm32`.
///
/// This is the compressed counterpart to [`crate::codec::h264::encode_mp4`],
/// which codes `I_PCM` and so writes raw samples. On a 17-frame 1280x1440 render
/// this produces about 0.16 MB at `qp = 27` against 47 MB for the `I_PCM` path,
/// at VMAF 94.9 (`docs/codec-benchmark.md`).
///
/// `qp` is the H.265 quantization parameter: lower is better and bigger, the
/// usable range is roughly 18-40, and 26 is a reasonable default. Every frame is
/// an IDR, so the result is all-intra — fine for short clips and the only option
/// where nothing else can run, but several times larger than an encoder with
/// inter prediction would produce.
pub fn encode_mp4(width: u32, height: u32, fps: u32, qp: i32, frames: &[Yuv420Frame]) -> Vec<u8> {
    encode_mp4_with_captions(width, height, fps, qp, frames, None)
}

/// [`encode_mp4`] with an optional soft-subtitle track.
pub fn encode_mp4_with_captions(
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    frames: &[Yuv420Frame],
    captions: Option<&CaptionTrack>,
) -> Vec<u8> {
    encode_mp4_from_iter(width, height, fps, qp, captions, frames.iter().cloned())
}

/// [`encode_mp4`] over an iterator of frames, so the caller can generate them
/// lazily and never hold more than one.
pub fn encode_mp4_from_iter<I>(
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    captions: Option<&CaptionTrack>,
    frames: I,
) -> Vec<u8>
where
    I: IntoIterator<Item = Yuv420Frame>,
{
    encode_mp4_inner(width, height, fps, qp, captions, &mut frames.into_iter())
}

/// [`encode_mp4`] pulling one frame at a time instead of taking them all up
/// front.
///
/// At high resolutions the slice-taking form is the wrong shape: a 4:2:0 frame
/// is `width * height * 1.5` bytes, so ten seconds of 8K at 24fps is about 12 GB
/// of source frames held live while the encoder walks them. The muxer only needs
/// the *encoded* samples, which are three orders of magnitude smaller, so
/// nothing requires the sources to coexist.
///
/// `next` is called once per frame index in order.
///
/// ```no_run
/// use threers::codec::hevc::{encode_compressed_mp4_streaming, Yuv420Frame};
/// # fn render(_i: usize) -> Vec<u8> { vec![0; 1920 * 1080 * 4] }
/// let bytes = encode_compressed_mp4_streaming(1920, 1080, 30, 26, 300, None, |i| {
///     Yuv420Frame::from_rgba(1920, 1080, &render(i))
/// });
/// ```
pub fn encode_mp4_streaming<F>(
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    frame_count: usize,
    captions: Option<&CaptionTrack>,
    next: F,
) -> Vec<u8>
where
    F: FnMut(usize) -> Yuv420Frame,
{
    encode_mp4_inner(
        width,
        height,
        fps,
        qp,
        captions,
        &mut (0..frame_count).map(next),
    )
}

fn encode_mp4_inner(
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    captions: Option<&CaptionTrack>,
    frames: &mut dyn Iterator<Item = Yuv420Frame>,
) -> Vec<u8> {
    encode_mp4_quality(width, height, fps, Quality::Qp(qp), captions, frames, None)
}

/// Say what a bitrate target could not do, and why it is the request rather
/// than the encoder.
fn report_reach(reach: Reach) {
    match reach {
        Reach::TooComplex { times } => eprintln!(
            "threers: this content will not fit that bitrate — even at the coarsest \
             quantizer it is about {times:.1}x larger. The file is as small as the \
             encoder can make it."
        ),
        Reach::TooSimple { times } => eprintln!(
            "threers: this content cannot fill that bitrate — even losslessly quantized \
             it is about {times:.1}x smaller, so the file is smaller than asked for. \
             That is the content, not a failure."
        ),
    }
}

/// [`encode_mp4_from_iter`] with a quality target rather than a fixed quantizer.
#[allow(clippy::too_many_arguments)]
pub fn encode_mp4_quality(
    width: u32,
    height: u32,
    fps: u32,
    quality: Quality,
    captions: Option<&CaptionTrack>,
    frames: &mut dyn Iterator<Item = Yuv420Frame>,
    total: Option<u32>,
) -> Vec<u8> {
    let enc = CompressedEncoder::new(width, height, PROBE_QP);
    let level_idc = enc.config().level_idc;

    // Parameter sets are hoisted into `hvcC`; only slice NALs become samples.
    let (vps_nal, sps_nal, pps_nal) = enc.parameter_sets();
    let (vps, sps, pps) = (vec![vps_nal], vec![sps_nal], vec![pps_nal]);

    // Frames go out in batches so they can be encoded concurrently. Nothing
    // carries between them, so the batching is invisible in the output — it
    // only decides how many cores are busy and how much is held at once.
    let mut batch = crate::codec::frames_in_flight(width, height);
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut rc = match quality {
        Quality::Qp(_) => None,
        Quality::Bitrate(bps) => Some(RateControl::new(bps, fps.max(1), total)),
    };
    if let Some(r) = rc.as_ref() {
        batch = r.group_size(crate::codec::frames_by_memory(width, height));
    }
    let base_qp = match quality {
        Quality::Qp(q) => q,
        Quality::Bitrate(_) => PROBE_QP,
    };

    // One frame at the probe quantizer tells the controller what this content
    // costs, which is worth more than any prior: content varies by orders of
    // magnitude and quantizers do not.
    let mut pending: Vec<Yuv420Frame> = Vec::with_capacity(batch);
    // `if let (Some(r), Some(f)) = (rc.as_mut(), frames.next())` would pull a
    // frame whether or not there is a controller to feed it to — the tuple is
    // built before the pattern is tried — and a fixed-quantizer encode would
    // silently lose its first frame.
    if let Some(r) = rc.as_mut() {
        if let Some(first) = frames.next() {
            let probe = enc.encode_slice_au_at(&first, PROBE_QP);
            r.seed(PROBE_QP, probe.len());
            // If the answer looks far from the probe, measure there too: one
            // extra frame buys a local slope instead of a chord across the
            // curve's bend.
            if let Some(q2) = r.probe_again() {
                let again = enc.encode_slice_au_at(&first, q2);
                r.seed_pair(q2, again.len());
            }
            pending.push(first);
        }
    }

    loop {
        while pending.len() < batch {
            match frames.next() {
                Some(f) => pending.push(f),
                None => break,
            }
        }
        if pending.is_empty() {
            break;
        }
        let qp = rc.as_ref().map_or(base_qp, |r| r.next_qp());
        let aus = crate::codec::map_frames(&pending, |f| enc.encode_slice_au_at(f, qp));
        pending.clear();
        for au in aus {
            if let Some(r) = rc.as_mut() {
                r.observe(qp, au.len());
            }
            let mut sample = Vec::new();
            for nal in split_annexb(&au) {
                sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                sample.extend_from_slice(&nal);
            }
            samples.push(sample);
        }
    }
    if let Some(reach) = rc.as_ref().and_then(|r| r.outcome()) {
        report_reach(reach);
    }

    let hvcc = build_hvcc(
        &HvccProfile::main_420(level_idc),
        &[
            HvccArray { nal_type: 32, complete: true, nals: &vps },
            HvccArray { nal_type: 33, complete: true, nals: &sps },
            HvccArray { nal_type: 34, complete: true, nals: &pps },
        ],
    );

    let timescale = 600u32;
    let params = Mp4Params {
        width,
        height,
        timescale,
        frame_duration: timescale / fps.max(1),
        hvcc_payload: &hvcc,
        almo_payload: None,
        samples: &samples,
    };
    match captions {
        Some(track) => mux_hevc_with_captions(&params, track),
        None => mux_hevc(&params),
    }
}

/// Split an Annex-B byte stream into NAL units. Start-code scanning is not
/// codec-specific, so this matches the H.264 helper of the same name.
fn split_annexb(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= stream.len()
            && stream[i] == 0
            && stream[i + 1] == 0
            && stream[i + 2] == 0
            && stream[i + 3] == 1
        {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (n, &(off, len)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map(|&(o, _)| o).unwrap_or(stream.len());
        nals.push(stream[off + len..end].to_vec());
    }
    nals
}

/// CABAC contexts for the compressed intra path (I-slice init values, §9.3.2.2).
///
/// `Copy`, and deliberately so: every rate-distortion comparison duplicates the
/// whole set, and at quadtree depths that happens thousands of times a picture.
#[derive(Clone, Copy)]
struct Ctx {
    part_mode: CtxModel,
    prev_intra: CtxModel,
    chroma_mode: CtxModel,
    cbf_luma: [CtxModel; 2],
    cbf_chroma: [CtxModel; 4],
    /// `split_transform_flag`, indexed by `5 - log2TrafoSize`.
    split_tu: [CtxModel; 3],
    /// `split_cu_flag`, indexed by how many neighbours are deeper.
    split_cu: [CtxModel; 3],
    /// `cu_qp_delta_abs` prefix: one context for the first bin, one for the rest.
    cu_qp_delta: [CtxModel; 2],
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
            split_tu: [
                CtxModel::init(153, qp),
                CtxModel::init(138, qp),
                CtxModel::init(138, qp),
            ],
            split_cu: [
                CtxModel::init(139, qp),
                CtxModel::init(141, qp),
                CtxModel::init(157, qp),
            ],
            cu_qp_delta: [CtxModel::init(154, qp), CtxModel::init(154, qp)],
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
    aq_strength: f32,
    deblocking: bool,
) -> (Vec<u8>, Reconstruction) {
    if cfg.tiles() > 1 {
        return tiled_slice_rbsp(cfg, y, u, v, residual, aq_strength, deblocking);
    }
    let mut rbsp = slice_header(cfg, &[]);
    let mut coded = encode_ctbs(cfg, y, u, v, residual, aq_strength, true);
    if deblocking {
        coded.deblock(cfg);
    }
    rbsp.extend_from_slice(&coded.data);
    (rbsp, coded.recon)
}

/// The tile grid in coding tree blocks, with uniform spacing (§6.5.1).
///
/// Column `i` spans `((i+1)·nx)/cols − (i·nx)/cols` CTBs, which distributes the
/// remainder over the leading columns rather than dumping it on the last one.
fn tile_spans(n: u32, parts: u32) -> Vec<(u32, u32)> {
    (0..parts)
        .map(|i| {
            let a = (i * n) / parts;
            let b = ((i + 1) * n) / parts;
            (a, b - a)
        })
        .collect()
}

/// Encode the picture as one slice of independently-coded tiles.
///
/// Tiles break intra prediction *and* the entropy coder at their edges, which
/// is exactly what makes them parallel — and it also means a tile is nothing
/// more than a smaller picture. So each one is encoded by the same
/// `encode_ctbs` that codes a whole frame, on its own plane region, and the
/// substreams are concatenated. Nothing in the coding path needs to know tiles
/// exist.
///
/// Two things are picture-level rather than per-tile: the reconstruction is
/// stitched back together, and the deblocking filter runs over the result, so
/// tile edges get smoothed like any other block edge. That is what
/// `loop_filter_across_tiles_enabled_flag` promises the decoder, and it costs
/// nothing here because the filter was already a separate pass.
fn tiled_slice_rbsp(
    cfg: &HevcConfig,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    residual: bool,
    aq_strength: f32,
    deblocking: bool,
) -> (Vec<u8>, Reconstruction) {
    let (cw, ch) = (cfg.coded_width, cfg.coded_height);
    let (cwc, chc) = (cw / 2, ch / 2);
    let (nx, ny) = cfg.ctbs();
    let ctb = 1u32 << cfg.ctb_log2;
    let cols = tile_spans(nx, cfg.tile_grid.0);
    let rows = tile_spans(ny, cfg.tile_grid.1);

    // Cut the planes into tiles. Tile boundaries fall on CTB boundaries, so
    // every tile but those at the right and bottom edges is a whole number of
    // them; the edge tiles are clipped, which the encoder already handles
    // because a picture is rarely a whole number of CTBs either.
    let mut jobs = Vec::with_capacity(cols.len() * rows.len());
    for (ry0, rh) in &rows {
        for (cx0, cwd) in &cols {
            let x0 = cx0 * ctb;
            let y0 = ry0 * ctb;
            let tw = (cwd * ctb).min(cw - x0);
            let th = (rh * ctb).min(ch - y0);
            let mut tcfg = *cfg;
            tcfg.width = tw;
            tcfg.height = th;
            tcfg.coded_width = tw;
            tcfg.coded_height = th;
            tcfg.tile_grid = (1, 1);
            jobs.push((
                tcfg,
                (x0, y0, tw, th),
                crop(y, cw, x0, y0, tw, th),
                crop(u, cwc, x0 / 2, y0 / 2, tw / 2, th / 2),
                crop(v, cwc, x0 / 2, y0 / 2, tw / 2, th / 2),
            ));
        }
    }

    let last = jobs.len() - 1;
    let coded = crate::codec::map_indexed(&jobs, |i, (tcfg, _, ty, tu, tv)| {
        encode_ctbs(tcfg, ty, tu, tv, residual, aq_strength, i == last)
    });

    // Stitch the reconstruction back together, then filter the whole thing.
    let mut recon = Reconstruction {
        y: vec![0u8; (cw * ch) as usize],
        u: vec![128u8; (cwc * chc) as usize],
        v: vec![128u8; (cwc * chc) as usize],
        coded_width: cw,
        coded_height: ch,
    };
    let mut edges = EdgeMap::new(cw, ch);
    let mut qp = vec![cfg.qp as i8; ((cw >> cfg.min_cb_log2) * (ch >> cfg.min_cb_log2)) as usize];
    let gl = cfg.min_cb_log2;
    let gw = (cw >> gl) as usize;
    for (job, c) in jobs.iter().zip(&coded) {
        let (x0, y0, tw, th) = job.1;
        paste(&mut recon.y, cw, &c.recon.y, tw, x0, y0, tw, th);
        paste(&mut recon.u, cwc, &c.recon.u, tw / 2, x0 / 2, y0 / 2, tw / 2, th / 2);
        paste(&mut recon.v, cwc, &c.recon.v, tw / 2, x0 / 2, y0 / 2, tw / 2, th / 2);
        for ty in (0..th).step_by(1 << gl) {
            for tx in (0..tw).step_by(1 << gl) {
                let src = (ty >> gl) as usize * (tw >> gl) as usize + (tx >> gl) as usize;
                let dst = ((y0 + ty) >> gl) as usize * gw + ((x0 + tx) >> gl) as usize;
                qp[dst] = c.qp[src];
            }
        }
        edges.paste(&c.edges, tw, th, x0, y0);
    }

    if deblocking {
        deblock(
            &mut recon.y,
            &mut recon.u,
            &mut recon.v,
            cw,
            ch,
            &edges,
            |x, y| i32::from(qp[(y >> gl) as usize * gw + (x >> gl) as usize]),
            chroma_qp,
        );
    }

    // The entry points are byte lengths of every substream but the last, counted
    // *after* emulation prevention — the specification counts those bytes. The
    // header always ends with the alignment one-bit, so its last byte is never
    // zero and the escaping of the data cannot depend on it.
    let mut entry = Vec::new();
    for c in &coded[..coded.len() - 1] {
        entry.push(emulation_prevention(&c.data).len());
    }
    let mut rbsp = slice_header(cfg, &entry);
    for c in &coded {
        rbsp.extend_from_slice(&c.data);
    }
    (rbsp, recon)
}

/// Copy a `w x h` rectangle out of a plane.
fn crop(plane: &[u8], stride: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for r in 0..h {
        let o = ((y + r) * stride + x) as usize;
        out.extend_from_slice(&plane[o..o + w as usize]);
    }
    out
}

/// Paste a `w x h` rectangle back into a plane.
#[allow(clippy::too_many_arguments)]
fn paste(dst: &mut [u8], stride: u32, src: &[u8], src_stride: u32, x: u32, y: u32, w: u32, h: u32) {
    for r in 0..h {
        let d = ((y + r) * stride + x) as usize;
        let s = (r * src_stride) as usize;
        dst[d..d + w as usize].copy_from_slice(&src[s..s + w as usize]);
    }
}

/// One coded picture, or one tile of one: the entropy-coded bytes, the
/// reconstruction, and the two things the deblocking filter needs that the
/// reconstruction alone does not carry.
struct Coded {
    data: Vec<u8>,
    recon: Reconstruction,
    edges: EdgeMap,
    /// `QpY` per minimum coding block.
    qp: Vec<i8>,
}

impl Coded {
    fn deblock(&mut self, cfg: &HevcConfig) {
        let (cw, ch) = (cfg.coded_width, cfg.coded_height);
        let gl = cfg.min_cb_log2;
        let gw = (cw >> gl) as usize;
        let qp = &self.qp;
        deblock(
            &mut self.recon.y,
            &mut self.recon.u,
            &mut self.recon.v,
            cw,
            ch,
            &self.edges,
            |x, y| i32::from(qp[(y >> gl) as usize * gw + (x >> gl) as usize]),
            chroma_qp,
        );
    }
}

/// `slice_segment_header()` for the picture's single slice.
///
/// `entry_points` are the byte lengths of every substream but the last, which a
/// decoder needs to find the tiles. Empty when the picture is not tiled.
fn slice_header(cfg: &HevcConfig, entry_points: &[usize]) -> Vec<u8> {
    let mut h = BitWriter::new();
    h.flag(true); // first_slice_segment_in_pic_flag
    h.flag(false); // no_output_of_prior_pics_flag
    h.write_ue(0); // slice_pic_parameter_set_id
    h.write_ue(2); // slice_type = I
    h.write_se(cfg.qp - 26); // slice_qp_delta (init_qp_minus26 = 0)
    if cfg.tiles() > 1 {
        h.write_ue(entry_points.len() as u32);
        if !entry_points.is_empty() {
            let bits = entry_points
                .iter()
                .map(|&o| 32 - (o as u32).leading_zeros())
                .max()
                .unwrap_or(1)
                .max(1);
            h.write_ue(bits - 1); // offset_len_minus1
            for &o in entry_points {
                // The value is the length minus one, so a one-byte subset codes
                // as zero.
                h.write_bits((o - 1) as u32, bits);
            }
        }
    }
    h.write_bit(1); // byte_alignment: alignment_bit_equal_to_one
    h.finish()
}

/// Code every coding tree block of a picture (or of one tile, encoded as a
/// picture of its own) and return the CABAC substream.
///
/// `last_subset` distinguishes the end of the slice from the end of a tile: the
/// former sends `end_of_slice_segment_flag = 1`, the latter sends a zero
/// followed by `end_of_subset_one_bit`, and both flush and byte-align.
fn encode_ctbs(
    cfg: &HevcConfig,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    residual: bool,
    aq_strength: f32,
    last_subset: bool,
) -> Coded {
    let cw = cfg.coded_width;
    let ch = cfg.coded_height;
    let (cwc, chc) = (cw / 2, ch / 2);

    // Reconstruction buffers (grown as CTBs are coded, read for neighbors).
    let mut ry = vec![0u8; (cw * ch) as usize];
    let mut ru = vec![128u8; (cwc * chc) as usize];
    let mut rv = vec![128u8; (cwc * chc) as usize];

    let lim = TreeLimits {
        max_depth: cfg.max_transform_hierarchy_depth_intra,
        max_tb_log2: cfg.max_tb_log2,
    };
    // The usual HEVC intra Lagrangian: cost = SSD + lambda * bits.
    let lambda = 0.57 * 2f64.powf((cfg.qp - 12) as f64 / 3.0);
    let mut cabac = CabacEncoder::new();
    let mut ctx = Ctx::new(cfg.qp);
    let (nx, ny) = cfg.ctbs();
    let ctb = 1u32 << cfg.ctb_log2;
    let g = Geo { cw, ch, ctb, nx };
    let mut grid = Grid::new(cw, ch, cfg.min_cb_log2);
    // One set of working buffers for the whole picture; see `Scratch`.
    let mut scratch = Scratch::default();
    let total = nx * ny;
    let mut coded = 0u32;
    let ctb_qp = adaptive_qp(y, cw, ch, ctb, nx, ny, cfg.qp, aq_strength);
    // `qPY_PREV` (§8.6.1). With the quantization group equal to the CTB, both
    // spatial predictors fall outside it and collapse to this.
    let mut qp_prev = cfg.qp;
    let mut edges = EdgeMap::new(cw, ch);

    for cby in 0..ny {
        for cbx in 0..nx {
            let (px, py) = (cbx * ctb, cby * ctb);
            let qp_ctb = ctb_qp[(cby * nx + cbx) as usize];
            let qp_l = qp_ctb;
            let qp_c = chroma_qp(qp_ctb);
            let delta = qp_ctb - qp_prev;

            // Two passes over the coding tree, for the same reason the transform
            // tree needs two: the search has to try candidates and roll them
            // back, and only the winner may reach the bitstream. The search
            // encodes into a scratch engine so each candidate sees the context
            // state its predecessors leave; `emit_cu` then replays the decided
            // tree into the real one from the same starting grid.
            let saved = grid.save(px, py, cfg.ctb_log2);
            let cu = {
                let mut planes = Planes {
                    y,
                    u,
                    v,
                    ry: &mut ry,
                    ru: &mut ru,
                    rv: &mut rv,
                    cw,
                    ch,
                    cwc,
                    chc,
                    nx,
                    ctb,
                    scratch: &mut scratch,
                    qp_l,
                    qp_c,
                    residual,
                };
                let mut scratch = cabac.probe();
                let mut sctx = ctx;
                let mut qd = QpDelta::new(qp_prev, delta);
                search_cu(
                    &mut planes,
                    &mut scratch,
                    &mut sctx,
                    &mut grid,
                    g,
                    px,
                    py,
                    cfg.ctb_log2,
                    0,
                    lim,
                    lambda,
                    &mut qd,
                )
                .0
            };
            grid.restore(px, py, cfg.ctb_log2, &saved);
            let mut qd = QpDelta::new(qp_prev, delta);
            emit_cu(
                &mut cabac, &mut ctx, &cu, &mut grid, g, px, py, cfg.ctb_log2, 0, lim, &mut qd,
            );
            // A coding tree block with nothing to code carries no delta, so the
            // predictor does not move past it.
            if qd.sent {
                qp_prev = qp_ctb;
            }
            mark_cu_edges(&cu, &mut edges, g, px, py, cfg.ctb_log2);

            coded += 1;
            if coded == total && !last_subset {
                // Not the end of the slice, only of this tile.
                cabac.encode_terminate(0); // end_of_slice_segment_flag
                cabac.encode_terminate(1); // end_of_subset_one_bit
            } else {
                cabac.encode_terminate((coded == total) as u32);
            }
        }
    }

    Coded {
        data: cabac.finish(),
        recon: Reconstruction {
            y: ry,
            u: ru,
            v: rv,
            coded_width: cw,
            coded_height: ch,
        },
        edges,
        qp: grid.into_qp(),
    }
}

/// The transform-tree shape limits taken from the SPS.
#[derive(Clone, Copy)]
struct TreeLimits {
    /// `max_transform_hierarchy_depth_intra`.
    max_depth: u32,
    /// `MaxTbLog2SizeY`. A node above this size splits without signalling it.
    max_tb_log2: u32,
}

/// One node of the transform tree: either a split with four children, or a leaf
/// carrying the quantized levels for its luma block and the two chroma blocks
/// that cover it.
struct Tu {
    log2: u32,
    split: bool,
    children: Vec<Tu>,
    cbf_luma: bool,
    levels_y: Vec<i32>,
    /// Chroma cbf *at this node's depth*. On a split node it is the OR of the
    /// children's, which is what makes the parent flag meaningful.
    cbf_cb: bool,
    cbf_cr: bool,
    levels_cb: Vec<i32>,
    levels_cr: Vec<i32>,
}

/// The planes a transform tree reads and writes, bundled to keep the recursion's
/// argument list survivable.
struct Planes<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    ry: &'a mut [u8],
    ru: &'a mut [u8],
    rv: &'a mut [u8],
    cw: u32,
    ch: u32,
    cwc: u32,
    chc: u32,
    nx: u32,
    /// Coding-tree-block edge, which sets the decoding order neighbours are
    /// judged available in.
    ctb: u32,
    /// Reusable working buffers — see [`Scratch`].
    scratch: &'a mut Scratch,
    /// Luma QP for the coding tree block being coded. With a quantization group
    /// the size of a CTB this is constant across everything below it.
    qp_l: i32,
    qp_c: i32,
    residual: bool,
}

/// Build the transform tree for the node at `(x, y)`, choosing at each level
/// whether to split, and reconstructing as it goes.
///
/// The reconstruction has to happen during the build, in decoding order, because
/// intra prediction reads neighbouring *reconstructed* samples. That is also why
/// splitting helps beyond transform efficiency: four 8x8 blocks each predict
/// from a boundary 8 samples away instead of one 16x16 predicting from the CU
/// edge.
///
/// The decision is Lagrangian — both shapes are built, the reconstruction rolled
/// back between them, and the cheaper `SSD + lambda * bits` kept. Splitting pays
/// on detail and costs flags and transform efficiency on flat areas, so which
/// wins is a property of the block rather than a constant.
#[allow(clippy::too_many_arguments)]
fn build_tu(
    p: &mut Planes,
    cabac: &CabacEncoder,
    ctx: &Ctx,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    mode: u8,
    cmode: u8,
    lambda: f64,
    qd: &QpDelta,
) -> Tu {
    // Above the largest transform size there is no choice: the tree splits and
    // the flag is not coded (§7.3.8.8).
    if log2 > lim.max_tb_log2 {
        return build_split(p, cabac, ctx, x, y, log2, depth, lim, mode, cmode, lambda, qd);
    }
    if !(depth < lim.max_depth && log2 > 2) {
        return build_leaf(p, x, y, log2, mode, cmode);
    }

    let before = save_rect(p, x, y, log2);
    let leaf = build_leaf(p, x, y, log2, mode, cmode);
    let leaf_cost =
        node_cost(p, x, y, log2, &leaf, cabac, ctx, depth, lim, mode, cmode, lambda, qd);
    let leaf_recon = save_rect(p, x, y, log2);

    restore_rect(p, x, y, log2, &before);
    let split = build_split(p, cabac, ctx, x, y, log2, depth, lim, mode, cmode, lambda, qd);
    let split_cost =
        node_cost(p, x, y, log2, &split, cabac, ctx, depth, lim, mode, cmode, lambda, qd);

    if leaf_cost <= split_cost {
        restore_rect(p, x, y, log2, &leaf_recon);
        leaf
    } else {
        split
    }
}

/// A leaf: one luma block, plus the chroma block covering it when this node is
/// large enough to own one.
fn build_leaf(p: &mut Planes, x: u32, y: u32, log2: u32, mode: u8, cmode: u8) -> Tu {
    let n = 1usize << log2;
    let (cbf_luma, levels_y) = code_leaf(
        p.y, p.ry, p.cw, p.ch, x, y, n, p.ctb, p.nx, mode, p.qp_l, true, p.residual, p.scratch,
    );
    let mut tu = Tu {
        log2,
        split: false,
        children: Vec::new(),
        cbf_luma,
        levels_y,
        cbf_cb: false,
        cbf_cr: false,
        levels_cb: Vec::new(),
        levels_cr: Vec::new(),
    };
    if log2 > 2 {
        code_node_chroma(p, &mut tu, x, y, log2, cmode);
    }
    tu
}

/// Split into four quadrants in z-order.
///
/// At `log2 == 3` the split is the 4:2:0 special case: 2x2 chroma does not
/// exist, so the four 4x4 luma blocks share this node's single 4x4 chroma block,
/// which the bitstream carries in the fourth child's transform unit. The chroma
/// is therefore coded *after* the luma children, matching decoding order.
#[allow(clippy::too_many_arguments)]
fn build_split(
    p: &mut Planes,
    cabac: &CabacEncoder,
    ctx: &Ctx,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    mode: u8,
    cmode: u8,
    lambda: f64,
    qd: &QpDelta,
) -> Tu {
    let half = 1u32 << (log2 - 1);
    let mut children = Vec::with_capacity(4);
    for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
        children.push(build_tu(
            p,
            cabac,
            ctx,
            x + dx,
            y + dy,
            log2 - 1,
            depth + 1,
            lim,
            mode,
            cmode,
            lambda,
            qd,
        ));
    }
    let mut tu = Tu {
        log2,
        split: true,
        children,
        cbf_luma: false,
        levels_y: Vec::new(),
        cbf_cb: false,
        cbf_cr: false,
        levels_cb: Vec::new(),
        levels_cr: Vec::new(),
    };
    if log2 == 3 {
        // Children are 4x4 luma and carry no chroma; this node owns it.
        code_node_chroma(p, &mut tu, x, y, log2, cmode);
    } else {
        tu.cbf_cb = tu.children.iter().any(|c| c.cbf_cb);
        tu.cbf_cr = tu.children.iter().any(|c| c.cbf_cr);
    }
    tu
}

/// Code the chroma block covering the `2^log2` luma square at `(x, y)`.
fn code_node_chroma(p: &mut Planes, tu: &mut Tu, x: u32, y: u32, log2: u32, cmode: u8) {
    let cn = 1usize << (log2 - 1);
    let (cx, cy) = (x / 2, y / 2);
    let (cbf_cb, levels_cb) = code_leaf(
        p.u, p.ru, p.cwc, p.chc, cx, cy, cn, p.ctb / 2, p.nx, cmode, p.qp_c, false, p.residual, p.scratch,
    );
    let (cbf_cr, levels_cr) = code_leaf(
        p.v, p.rv, p.cwc, p.chc, cx, cy, cn, p.ctb / 2, p.nx, cmode, p.qp_c, false, p.residual, p.scratch,
    );
    tu.cbf_cb = cbf_cb;
    tu.cbf_cr = cbf_cr;
    tu.levels_cb = levels_cb;
    tu.levels_cr = levels_cr;
}

/// Working buffers for one block, kept alive across calls.
///
/// `code_leaf` used to allocate nine `Vec`s every time it ran, and it runs once
/// per candidate per level of two nested trees — the profile put a quarter of
/// the encoder's time in the allocator. Fixed-size stack arrays would trade that
/// for zeroing four kilobytes to use sixteen of them on a 4x4 block, so these
/// are `Vec`s that keep their capacity instead: one allocation each for the
/// whole picture.
struct Scratch {
    refs: Vec<i32>,
    smooth: Vec<i32>,
    above: Vec<i32>,
    left: Vec<i32>,
    pred: Vec<i32>,
    orig: Vec<i32>,
    resid: Vec<i32>,
    coeff: Vec<i32>,
    deq: Vec<i32>,
    res: Vec<i32>,
}

impl Default for Scratch {
    /// Sized to the largest block once, and sliced from then on. Resizing per
    /// call would zero four kilobytes to use sixteen of them on a 4x4 block,
    /// which is how the first attempt at this came out *slower*.
    fn default() -> Self {
        const N: usize = MAX_N * MAX_N;
        const R: usize = 4 * MAX_N + 1;
        Self {
            refs: vec![0; R],
            smooth: vec![0; R],
            above: vec![0; 2 * MAX_N + 1],
            left: vec![0; 2 * MAX_N + 1],
            pred: vec![0; N],
            orig: vec![0; N],
            resid: vec![0; N],
            coeff: vec![0; N],
            deq: vec![0; N],
            res: vec![0; N],
        }
    }
}

/// Predict, transform, quantize and reconstruct one `n x n` block of one plane.
/// Returns its `cbf` and quantized levels; the reconstruction is written back so
/// the next block in decoding order can predict from it.
#[allow(clippy::too_many_arguments)]
fn code_leaf(
    orig: &[u8],
    recon: &mut [u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    n: usize,
    ctb: u32,
    nx: u32,
    mode: u8,
    qp: i32,
    luma: bool,
    residual: bool,
    sc: &mut Scratch,
) -> (bool, Vec<i32>) {
    let (nn, nr) = (n * n, 4 * n + 1);
    refs_generic_into(recon, w, h, x, y, n as u32, ctb, nx, &mut sc.refs[..nr]);
    let src: &[i32] = if luma && filter_flag(mode, n) {
        smooth_refs_into(&sc.refs[..nr], &mut sc.smooth[..nr]);
        &sc.smooth[..nr]
    } else {
        &sc.refs[..nr]
    };
    split_refs_into(src, n, &mut sc.above[..2 * n + 1], &mut sc.left[..2 * n + 1]);
    intra::predict_into(
        mode,
        n,
        &sc.above[..2 * n + 1],
        &sc.left[..2 * n + 1],
        luma,
        &mut sc.pred[..nn],
    );
    gather_into(orig, w, x, y, n as u32, &mut sc.orig[..nn]);
    for i in 0..nn {
        sc.resid[i] = sc.orig[i] - sc.pred[i];
    }
    // 4x4 intra luma is the one place HEVC uses DST-VII instead of the DCT
    // (§8.6.4.2) — it fits the residual of a block predicted from one edge.
    let kind = if luma && n == 4 {
        TransformKind::Dst
    } else {
        TransformKind::Dct
    };
    forward_into(&sc.resid[..nn], &mut sc.coeff[..nn], n, kind);
    let mut levels = vec![0i32; nn];
    quant::quant_into(&sc.coeff[..nn], n, qp, true, &mut levels);
    let cbf = residual && levels.iter().any(|&c| c != 0);

    if cbf {
        quant::dequant_into(&levels, n, qp, &mut sc.deq[..nn]);
        inverse_into(&sc.deq[..nn], &mut sc.res[..nn], n, kind);
        for i in 0..nn {
            sc.pred[i] += sc.res[i];
        }
    }
    store(recon, w, x, y, n as u32, &sc.pred[..nn]);
    (cbf, levels)
}

/// Emit `transform_tree` / `transform_unit` (H.265 §7.3.8.8, §7.3.8.10).
///
/// Order matters and is not the order the tree was built in: a node's chroma
/// cbf is signalled before its children, because a child only codes its own when
/// the parent's was set.
#[allow(clippy::too_many_arguments)]
fn emit_tu(
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    tu: &Tu,
    depth: u32,
    lim: TreeLimits,
    mode: u8,
    cmode: u8,
    parent_cb: bool,
    parent_cr: bool,
    // Set on the fourth 4x4 child: the chroma of the 8x8 parent it shares, which
    // the bitstream carries here (`blkIdx == 3`) rather than at the parent.
    blk3_chroma: Option<&Tu>,
    qd: &mut QpDelta,
) {
    if tu.log2 <= lim.max_tb_log2 && depth < lim.max_depth && tu.log2 > 2 {
        // ctxInc = 5 - log2TrafoSize.
        cabac.encode_bin(&mut ctx.split_tu[(5 - tu.log2) as usize], tu.split as u32);
    }
    if tu.log2 > 2 {
        if depth == 0 || parent_cb {
            cabac.encode_bin(&mut ctx.cbf_chroma[depth as usize], tu.cbf_cb as u32);
        }
        if depth == 0 || parent_cr {
            cabac.encode_bin(&mut ctx.cbf_chroma[depth as usize], tu.cbf_cr as u32);
        }
    }
    if tu.split {
        for (i, child) in tu.children.iter().enumerate() {
            // 4x4 children share the parent's chroma, delivered at blkIdx 3.
            let carry = (tu.log2 == 3 && i == 3).then_some(tu);
            emit_tu(
                cabac,
                ctx,
                child,
                depth + 1,
                lim,
                mode,
                cmode,
                tu.cbf_cb,
                tu.cbf_cr,
                carry,
                qd,
            );
        }
        return;
    }

    // Leaf. An intra CU always signals cbf_luma; ctxInc is 1 at depth 0, else 0.
    cabac.encode_bin(
        &mut ctx.cbf_luma[usize::from(depth == 0)],
        tu.cbf_luma as u32,
    );
    // `cu_qp_delta` rides in the first transform unit of the group that codes
    // anything, ahead of the residual it applies to (§7.3.8.10).
    //
    // At 4x4 the chroma flags to test are the *parent's*, and for all four
    // children — not only the one that carries the shared chroma residual.
    // A child with no luma coefficients still opens a transform unit if the
    // 8x8 above it has chroma, and that unit is where the delta belongs.
    let cbf_chroma = if tu.log2 > 2 {
        tu.cbf_cb || tu.cbf_cr
    } else {
        parent_cb || parent_cr
    };
    if tu.cbf_luma || cbf_chroma {
        qd.emit(cabac, ctx);
    }
    let n = 1usize << tu.log2;
    if tu.cbf_luma {
        residual::encode(
            cabac,
            &mut ctx.res,
            &tu.levels_y,
            n,
            false,
            scan_idx(mode, tu.log2, false),
        );
    }
    if tu.log2 > 2 {
        let cscan = scan_idx(cmode, tu.log2 - 1, true);
        if tu.cbf_cb {
            residual::encode(cabac, &mut ctx.res, &tu.levels_cb, n / 2, true, cscan);
        }
        if tu.cbf_cr {
            residual::encode(cabac, &mut ctx.res, &tu.levels_cr, n / 2, true, cscan);
        }
    } else if let Some(parent) = blk3_chroma {
        // The shared 4x4 chroma of the 8x8 above, gated by *its* cbf.
        let cscan = scan_idx(cmode, 2, true);
        if parent.cbf_cb {
            residual::encode(cabac, &mut ctx.res, &parent.levels_cb, 4, true, cscan);
        }
        if parent.cbf_cr {
            residual::encode(cabac, &mut ctx.res, &parent.levels_cr, 4, true, cscan);
        }
    }
}

/// Copy the reconstructed samples covered by a node out of all three planes, so
/// a candidate tree shape can be tried and rolled back.
fn save_rect(p: &Planes, x: u32, y: u32, log2: u32) -> Vec<u8> {
    let n = 1u32 << log2;
    let cn = n / 2;
    let mut out = Vec::with_capacity((n * n * 3 / 2) as usize);
    for r in 0..n {
        let o = ((y + r) * p.cw + x) as usize;
        out.extend_from_slice(&p.ry[o..o + n as usize]);
    }
    let (cx, cy) = (x / 2, y / 2);
    for plane in [&*p.ru, &*p.rv] {
        for r in 0..cn {
            let o = ((cy + r) * p.cwc + cx) as usize;
            out.extend_from_slice(&plane[o..o + cn as usize]);
        }
    }
    out
}

fn restore_rect(p: &mut Planes, x: u32, y: u32, log2: u32, saved: &[u8]) {
    let n = 1u32 << log2;
    let cn = n / 2;
    let mut k = 0;
    for r in 0..n {
        let o = ((y + r) * p.cw + x) as usize;
        p.ry[o..o + n as usize].copy_from_slice(&saved[k..k + n as usize]);
        k += n as usize;
    }
    let (cx, cy) = (x / 2, y / 2);
    for ci in 0..2 {
        let plane: &mut [u8] = if ci == 0 { p.ru } else { p.rv };
        for r in 0..cn {
            let o = ((cy + r) * p.cwc + cx) as usize;
            plane[o..o + cn as usize].copy_from_slice(&saved[k..k + cn as usize]);
            k += cn as usize;
        }
    }
}

/// Lagrangian cost of a candidate subtree: squared error of the reconstruction
/// it left behind, plus `lambda` times the bits it actually costs.
///
/// The bit count is exact rather than modelled — the CABAC engine and its
/// contexts are cloned, the candidate coded into the copy, and the copy dropped.
/// That costs an extra encode per candidate per level and is why this encoder is
/// slower than its structure suggests, but it removes the usual guesswork about
/// what a flag is worth in a given context state.
#[allow(clippy::too_many_arguments)]
fn node_cost(
    p: &Planes,
    x: u32,
    y: u32,
    log2: u32,
    tu: &Tu,
    cabac: &CabacEncoder,
    ctx: &Ctx,
    depth: u32,
    lim: TreeLimits,
    mode: u8,
    cmode: u8,
    lambda: f64,
    qd: &QpDelta,
) -> f64 {
    let n = 1u32 << log2;
    let cn = n / 2;
    let mut ssd = 0f64;
    for r in 0..n {
        let o = ((y + r) * p.cw + x) as usize;
        for c in 0..n as usize {
            let d = p.ry[o + c] as f64 - p.y[o + c] as f64;
            ssd += d * d;
        }
    }
    let (cx, cy) = (x / 2, y / 2);
    for (rec, src) in [(&*p.ru, p.u), (&*p.rv, p.v)] {
        for r in 0..cn {
            let o = ((cy + r) * p.cwc + cx) as usize;
            for c in 0..cn as usize {
                let d = rec[o + c] as f64 - src[o + c] as f64;
                ssd += d * d;
            }
        }
    }

    // A probe rather than a clone: the output buffer would otherwise be copied
    // once per candidate per level.
    let mut trial = cabac.probe();
    let mut tctx = *ctx;
    let mut tqd = *qd;
    let before = trial.bit_len();
    emit_tu(
        &mut trial, &mut tctx, tu, depth, lim, mode, cmode, true, true, None, &mut tqd,
    );
    let bits = (trial.bit_len() - before) as f64;

    ssd + lambda * bits
}

/// `cu_qp_delta` state for one quantization group (§7.3.8.10).
///
/// The delta is written in the first transform unit that has anything to code,
/// wherever in the coding tree that turns out to be — so the value travels down
/// the recursion and the flag comes back up. A group that codes nothing sends no
/// delta at all, and then the decoder's `QpY` is the predictor, not the target.
#[derive(Clone, Copy)]
struct QpDelta {
    /// `qPY_PRED` for this group.
    pred: i32,
    value: i32,
    sent: bool,
}

impl QpDelta {
    fn new(pred: i32, value: i32) -> Self {
        Self {
            pred,
            value,
            sent: false,
        }
    }

    /// The `QpY` a coding unit finishing *now* would have.
    ///
    /// Not constant across a quantization group: `CuQpDeltaVal` is zero until
    /// the delta is parsed, so coding units decoded before the first one with
    /// anything to code sit at the predictor and the rest sit at the target.
    /// The deblocking filter reads this per coding unit, which is how the
    /// distinction becomes visible at all.
    fn qp_now(&self) -> i32 {
        self.pred + if self.sent { self.value } else { 0 }
    }

    /// Emit `cu_qp_delta_abs` and its sign if this is the first coded block of
    /// the group. Prefix is truncated unary with `cMax = 5`, two contexts; the
    /// tail beyond that is bypass Exp-Golomb.
    fn emit(&mut self, cabac: &mut CabacEncoder, ctx: &mut Ctx) {
        if self.sent {
            return;
        }
        self.sent = true;
        let abs = self.value.unsigned_abs();
        let prefix = abs.min(5);
        for k in 0..prefix {
            cabac.encode_bin(&mut ctx.cu_qp_delta[usize::from(k > 0)], 1);
        }
        if prefix < 5 {
            cabac.encode_bin(&mut ctx.cu_qp_delta[usize::from(prefix > 0)], 0);
        } else {
            let mut rest = abs - 5;
            let mut k = 0u32;
            while rest >= (1 << k) {
                cabac.encode_bypass(1);
                rest -= 1 << k;
                k += 1;
            }
            cabac.encode_bypass(0);
            for b in (0..k).rev() {
                cabac.encode_bypass((rest >> b) & 1);
            }
        }
        if abs > 0 {
            cabac.encode_bypass(u32::from(self.value < 0));
        }
    }
}

/// Per-coding-tree-block quantizer offsets from local variance.
///
/// See the H.264 counterpart in `codec::h264::compress` for the reasoning and
/// for what it measured. Off by default in both.
#[allow(clippy::too_many_arguments)]
fn adaptive_qp(
    y: &[u8],
    cw: u32,
    ch: u32,
    ctb: u32,
    nx: u32,
    ny: u32,
    base: i32,
    strength: f32,
) -> Vec<i32> {
    let n = (nx * ny) as usize;
    if strength == 0.0 {
        return vec![base; n];
    }
    let mut energy = Vec::with_capacity(n);
    for cby in 0..ny {
        for cbx in 0..nx {
            let (ox, oy) = (cbx * ctb, cby * ctb);
            let (w, h) = (ctb.min(cw - ox), ctb.min(ch - oy));
            let (mut sum, mut sq) = (0i64, 0i64);
            for j in 0..h {
                let row = ((oy + j) * cw + ox) as usize;
                for &p in &y[row..row + w as usize] {
                    sum += i64::from(p);
                    sq += i64::from(p) * i64::from(p);
                }
            }
            let count = i64::from(w * h).max(1);
            let var = (sq - sum * sum / count).max(0) as f32 / count as f32;
            energy.push((var + 1.0).log2());
        }
    }
    let mean = energy.iter().sum::<f32>() / energy.len() as f32;
    energy
        .iter()
        .map(|e| {
            let adj = (strength * (e - mean)).round().clamp(-5.0, 5.0) as i32;
            (base + adj).clamp(0, 51)
        })
        .collect()
}

/// Picture geometry the coding quadtree consults constantly.
#[derive(Clone, Copy)]
struct Geo {
    cw: u32,
    ch: u32,
    ctb: u32,
    nx: u32,
}

/// Per-minimum-coding-block state that later blocks derive their contexts from.
///
/// Both fields are read by *neighbours*, so a candidate that loses a
/// rate-distortion comparison has to leave no trace: [`Grid::save`] and
/// [`Grid::restore`] cover exactly the square a candidate may write.
struct Grid {
    /// Chosen luma prediction mode; `255` where nothing has been coded yet.
    modes: Vec<u8>,
    /// `CtDepth` — the quadtree depth of the coding unit covering each block,
    /// which is what `split_cu_flag`'s context is derived from.
    depth: Vec<u8>,
    /// `QpY` of the coding unit covering each block, for the deblocking filter.
    qp: Vec<i8>,
    log2: u32,
    w: usize,
    h: usize,
}

impl Grid {
    fn new(cw: u32, ch: u32, log2: u32) -> Self {
        let (w, h) = ((cw >> log2) as usize, (ch >> log2) as usize);
        Self {
            modes: vec![255; w * h],
            depth: vec![0; w * h],
            qp: vec![0; w * h],
            log2,
            w,
            h,
        }
    }

    fn idx(&self, x: u32, y: u32) -> usize {
        (y >> self.log2) as usize * self.w + (x >> self.log2) as usize
    }

    /// Prediction mode at a sample position, DC where nothing was coded.
    fn mode_at(&self, x: u32, y: u32) -> u8 {
        match self.modes[self.idx(x, y)] {
            255 => intra::DC,
            m => m,
        }
    }

    fn depth_at(&self, x: u32, y: u32) -> u32 {
        u32::from(self.depth[self.idx(x, y)])
    }

    fn into_qp(self) -> Vec<i8> {
        self.qp
    }

    #[allow(dead_code)]
    fn qp_at(&self, x: u32, y: u32) -> i32 {
        i32::from(self.qp[self.idx(x, y)])
    }

    /// Stamp a coding unit's mode, depth and quantizer across every block it
    /// covers.
    fn set_cu(&mut self, x: u32, y: u32, log2: u32, mode: u8, depth: u32, qp: i32) {
        let n = 1u32 << log2;
        for dy in (0..n).step_by(1 << self.log2) {
            for dx in (0..n).step_by(1 << self.log2) {
                let (gx, gy) = ((x + dx) >> self.log2, (y + dy) >> self.log2);
                if (gx as usize) < self.w && (gy as usize) < self.h {
                    let i = gy as usize * self.w + gx as usize;
                    self.modes[i] = mode;
                    self.depth[i] = depth as u8;
                    self.qp[i] = qp as i8;
                }
            }
        }
    }

    fn save(&self, x: u32, y: u32, log2: u32) -> Vec<(u8, u8, i8)> {
        let n = 1u32 << log2;
        let mut out = Vec::new();
        for dy in (0..n).step_by(1 << self.log2) {
            for dx in (0..n).step_by(1 << self.log2) {
                let (gx, gy) = ((x + dx) >> self.log2, (y + dy) >> self.log2);
                if (gx as usize) < self.w && (gy as usize) < self.h {
                    let i = gy as usize * self.w + gx as usize;
                    out.push((self.modes[i], self.depth[i], self.qp[i]));
                }
            }
        }
        out
    }

    fn restore(&mut self, x: u32, y: u32, log2: u32, saved: &[(u8, u8, i8)]) {
        let n = 1u32 << log2;
        let mut k = 0;
        for dy in (0..n).step_by(1 << self.log2) {
            for dx in (0..n).step_by(1 << self.log2) {
                let (gx, gy) = ((x + dx) >> self.log2, (y + dy) >> self.log2);
                if (gx as usize) < self.w && (gy as usize) < self.h {
                    let i = gy as usize * self.w + gx as usize;
                    (self.modes[i], self.depth[i], self.qp[i]) = saved[k];
                    k += 1;
                }
            }
        }
    }
}

/// One node of the coding quadtree: four children, or a coding unit.
enum Cu {
    /// Fewer than four children where the block hangs off the picture edge —
    /// the ones whose top-left corner is outside are not coded at all.
    Split(Vec<Cu>),
    Leaf(Box<CuLeaf>),
}

struct CuLeaf {
    mode: u8,
    cmode: u8,
    tu: Tu,
}

/// Record every transform-block boundary in a decided coding tree, for the
/// deblocking filter.
///
/// The filter only smooths where a transform actually ends. Walking the decided
/// tree afterwards is the cheapest place to learn that: during the search the
/// answer changes with every candidate, and after it the winner is fixed.
fn mark_cu_edges(cu: &Cu, edges: &mut EdgeMap, g: Geo, x: u32, y: u32, log2: u32) {
    match cu {
        Cu::Split(kids) => {
            let half = 1u32 << (log2 - 1);
            let mut k = 0;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                let (kx, ky) = (x + dx, y + dy);
                if kx >= g.cw || ky >= g.ch {
                    continue;
                }
                mark_cu_edges(&kids[k], edges, g, kx, ky, log2 - 1);
                k += 1;
            }
        }
        Cu::Leaf(l) => mark_tu_edges(&l.tu, edges, x, y),
    }
}

fn mark_tu_edges(tu: &Tu, edges: &mut EdgeMap, x: u32, y: u32) {
    if tu.split {
        let half = 1u32 << (tu.log2 - 1);
        for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)]
            .into_iter()
            .enumerate()
        {
            if let Some(child) = tu.children.get(i) {
                mark_tu_edges(child, edges, x + dx, y + dy);
            }
        }
        return;
    }
    edges.mark(x, y, 1 << tu.log2);
}

/// `split_cu_flag`'s context: how many of the two neighbours sit *deeper* in the
/// quadtree than this node (§9.3.4.2.2). A neighbourhood of small blocks is
/// evidence that this one should be small too.
fn split_ctx(grid: &Grid, g: Geo, x: u32, y: u32, depth: u32) -> usize {
    let (xi, yi) = (x as i32, y as i32);
    let left = available(xi - 1, yi, x, y, g.cw, g.ch, g.ctb, g.nx)
        && grid.depth_at(x - 1, y) > depth;
    let above = available(xi, yi - 1, x, y, g.cw, g.ch, g.ctb, g.nx)
        && grid.depth_at(x, y - 1) > depth;
    usize::from(left) + usize::from(above)
}

/// Squared error the current reconstruction leaves over a block, all three planes.
fn ssd_rect(p: &Planes, x: u32, y: u32, log2: u32) -> f64 {
    let n = (1u32 << log2).min(p.cw - x).min(p.ch - y);
    let mut ssd = 0f64;
    for r in 0..n {
        let o = ((y + r) * p.cw + x) as usize;
        for c in 0..n as usize {
            let d = p.ry[o + c] as f64 - p.y[o + c] as f64;
            ssd += d * d;
        }
    }
    let (cx, cy) = (x / 2, y / 2);
    let cn = n / 2;
    for (rec, src) in [(&*p.ru, p.u), (&*p.rv, p.v)] {
        for r in 0..cn {
            let o = ((cy + r) * p.cwc + cx) as usize;
            for c in 0..cn as usize {
                let d = rec[o + c] as f64 - src[o + c] as f64;
                ssd += d * d;
            }
        }
    }
    ssd
}

/// Choose between coding `(x, y)` as one coding unit and splitting it in four,
/// and return the winner along with its Lagrangian cost.
///
/// `cabac` and `ctx` are a scratch pair: the search encodes into them so that
/// each candidate sees the context state its predecessors actually leave, and
/// the winner's state is what carries forward. The real bitstream is written
/// later by [`emit_cu`], which walks the decided tree.
#[allow(clippy::too_many_arguments)]
fn search_cu(
    p: &mut Planes,
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    grid: &mut Grid,
    g: Geo,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    lambda: f64,
    qd: &mut QpDelta,
) -> (Cu, f64) {
    let size = 1u32 << log2;
    let inside = x + size <= g.cw && y + size <= g.ch;

    // A block hanging off the picture edge is split without a flag: the decoder
    // knows it cannot be coded whole (§7.3.8.4).
    if !inside {
        return split_cu(p, cabac, ctx, grid, g, x, y, log2, depth, lim, lambda, false, qd);
    }
    // At the minimum coding block size there is nothing to decide.
    if log2 == grid.log2 {
        return leaf_cu(p, cabac, ctx, grid, g, x, y, log2, depth, lim, lambda, false, qd);
    }

    let before = save_rect(p, x, y, log2);
    let gbefore = grid.save(x, y, log2);

    let mut cabac_one = cabac.clone();
    let mut ctx_one = *ctx;
    let mut qd_one = *qd;
    let (cu_one, cost_one) = leaf_cu(
        p, &mut cabac_one, &mut ctx_one, grid, g, x, y, log2, depth, lim, lambda, true, &mut qd_one,
    );
    let recon_one = save_rect(p, x, y, log2);
    let grid_one = grid.save(x, y, log2);

    restore_rect(p, x, y, log2, &before);
    grid.restore(x, y, log2, &gbefore);
    let mut cabac_four = cabac.clone();
    let mut ctx_four = *ctx;
    let mut qd_four = *qd;
    let (cu_four, cost_four) = split_cu(
        p, &mut cabac_four, &mut ctx_four, grid, g, x, y, log2, depth, lim, lambda, true,
        &mut qd_four,
    );

    if cost_one <= cost_four {
        restore_rect(p, x, y, log2, &recon_one);
        grid.restore(x, y, log2, &grid_one);
        *cabac = cabac_one;
        *ctx = ctx_one;
        *qd = qd_one;
        (cu_one, cost_one)
    } else {
        *cabac = cabac_four;
        *ctx = ctx_four;
        *qd = qd_four;
        (cu_four, cost_four)
    }
}

/// Code `(x, y)` as a single intra coding unit.
#[allow(clippy::too_many_arguments)]
fn leaf_cu(
    p: &mut Planes,
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    grid: &mut Grid,
    g: Geo,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    lambda: f64,
    flag: bool,
    qd: &mut QpDelta,
) -> (Cu, f64) {
    let start = cabac.bit_len();
    if flag {
        let inc = split_ctx(grid, g, x, y, depth);
        cabac.encode_bin(&mut ctx.split_cu[inc], 0);
    }

    // Pick the mode on the first transform block rather than the whole coding
    // unit: for a block bigger than the largest transform, that is the only part
    // whose reference samples exist yet, and a mode that suits it badly loses
    // the split comparison anyway.
    let mn = 1u32 << log2.min(lim.max_tb_log2);
    let lvals = refs_luma(p.ry, p.cw, p.ch, x, y, mn, p.nx, p.ctb);
    let orig = gather(p.y, p.cw, x, y, mn);
    let mode = best_luma_mode(&orig, mn as usize, &lvals);
    let cmode = chroma_mode_from_luma(mode);

    let tu = build_tu(p, cabac, ctx, x, y, log2, 0, lim, mode, cmode, lambda, qd);

    // ---- coding_unit syntax (§7.3.8.5) ----
    if log2 == grid.log2 {
        // part_mode exists only at the minimum size, where PART_NxN is legal.
        cabac.encode_bin(&mut ctx.part_mode, 1); // PART_2Nx2N
    }
    code_luma_mode(cabac, ctx, mode, grid, g, x, y);
    cabac.encode_bin(&mut ctx.chroma_mode, 0); // intra_chroma_pred_mode = DM
    emit_tu(cabac, ctx, &tu, 0, lim, mode, cmode, true, true, None, qd);
    grid.set_cu(x, y, log2, mode, depth, qd.qp_now());

    let bits = (cabac.bit_len() - start) as f64;
    let cost = ssd_rect(p, x, y, log2) + lambda * bits;
    (Cu::Leaf(Box::new(CuLeaf { mode, cmode, tu })), cost)
}

/// Split `(x, y)` into four quadrants and code each independently.
#[allow(clippy::too_many_arguments)]
fn split_cu(
    p: &mut Planes,
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    grid: &mut Grid,
    g: Geo,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    lambda: f64,
    flag: bool,
    qd: &mut QpDelta,
) -> (Cu, f64) {
    let start = cabac.bit_len();
    if flag {
        let inc = split_ctx(grid, g, x, y, depth);
        cabac.encode_bin(&mut ctx.split_cu[inc], 1);
    }
    let mut cost = lambda * (cabac.bit_len() - start) as f64;

    let half = 1u32 << (log2 - 1);
    let mut kids = Vec::with_capacity(4);
    for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
        let (kx, ky) = (x + dx, y + dy);
        if kx >= g.cw || ky >= g.ch {
            continue; // wholly outside the picture: never coded
        }
        let (kid, kc) = search_cu(
            p,
            cabac,
            ctx,
            grid,
            g,
            kx,
            ky,
            log2 - 1,
            depth + 1,
            lim,
            lambda,
            qd,
        );
        kids.push(kid);
        cost += kc;
    }
    (Cu::Split(kids), cost)
}

/// Write the decided quadtree to the real bitstream.
///
/// Every context this reads — `split_cu_flag`'s neighbour depths, the mode
/// prediction list — comes from `grid`, which the caller has rolled back to its
/// pre-search state, so the derivations here reproduce the search's exactly.
#[allow(clippy::too_many_arguments)]
fn emit_cu(
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    cu: &Cu,
    grid: &mut Grid,
    g: Geo,
    x: u32,
    y: u32,
    log2: u32,
    depth: u32,
    lim: TreeLimits,
    qd: &mut QpDelta,
) {
    let size = 1u32 << log2;
    let codes_flag = x + size <= g.cw && y + size <= g.ch && log2 > grid.log2;
    match cu {
        Cu::Split(kids) => {
            if codes_flag {
                let inc = split_ctx(grid, g, x, y, depth);
                cabac.encode_bin(&mut ctx.split_cu[inc], 1);
            }
            let half = 1u32 << (log2 - 1);
            let mut k = 0;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                let (kx, ky) = (x + dx, y + dy);
                if kx >= g.cw || ky >= g.ch {
                    continue;
                }
                emit_cu(
                    cabac,
                    ctx,
                    &kids[k],
                    grid,
                    g,
                    kx,
                    ky,
                    log2 - 1,
                    depth + 1,
                    lim,
                    qd,
                );
                k += 1;
            }
        }
        Cu::Leaf(l) => {
            if codes_flag {
                let inc = split_ctx(grid, g, x, y, depth);
                cabac.encode_bin(&mut ctx.split_cu[inc], 0);
            }
            if log2 == grid.log2 {
                cabac.encode_bin(&mut ctx.part_mode, 1);
            }
            code_luma_mode(cabac, ctx, l.mode, grid, g, x, y);
            cabac.encode_bin(&mut ctx.chroma_mode, 0);
            emit_tu(cabac, ctx, &l.tu, 0, lim, l.mode, l.cmode, true, true, None, qd);
            grid.set_cu(x, y, log2, l.mode, depth, qd.qp_now());
        }
    }
}

/// Chroma mode for DM_CHROMA in 4:2:0 (identity — no 4:2:2 remap needed).
fn chroma_mode_from_luma(luma: u8) -> u8 {
    luma
}

/// Try planar/DC and every angular mode (each with its own smoothing); return the
/// one with least SAD against the original.
fn best_luma_mode(orig: &[i32], n: usize, vals: &[i32]) -> u8 {
    // The reference samples depend on the mode only through `filter_flag`,
    // which is a boolean — so across all thirty-five modes there are exactly two
    // reference sets, not thirty-five. Building both once turns a hundred and
    // five allocations and thirty-five smoothing passes into three and one.
    let plain = split_refs(vals, n);
    let smooth = split_refs(&smooth_refs(vals), n);
    let mut best = (u32::MAX, intra::DC);
    let mut pred = vec![0i32; n * n];
    for mode in 0..=34u8 {
        let (above, left) = if filter_flag(mode, n) { &smooth } else { &plain };
        intra::predict_into(mode, n, above, left, true, &mut pred[..]);
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
#[allow(clippy::too_many_arguments)]
fn code_luma_mode(
    cabac: &mut CabacEncoder,
    ctx: &mut Ctx,
    mode: u8,
    grid: &Grid,
    g: Geo,
    x: u32,
    y: u32,
) {
    let (xi, yi) = (x as i32, y as i32);
    let cand_a = if available(xi - 1, yi, x, y, g.cw, g.ch, g.ctb, g.nx) {
        grid.mode_at(x - 1, y)
    } else {
        intra::DC
    };
    // The above neighbour only counts inside the current CTB row. Line buffers
    // would otherwise have to hold a whole picture-width of modes, so the spec
    // substitutes DC across the CTB boundary (§8.4.2).
    let ctb_top = ((y >> g.ctb.trailing_zeros()) << g.ctb.trailing_zeros()) as i32;
    let same_ctb_row = yi > ctb_top;
    let cand_b = if same_ctb_row && available(xi, yi - 1, x, y, g.cw, g.ch, g.ctb, g.nx) {
        grid.mode_at(x, y - 1)
    } else {
        intra::DC
    };

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
/// Whether the sample at `(px, py)` has already been reconstructed when coding
/// the block whose origin is `(bx, by)`.
///
/// CTBs are coded in raster order and blocks inside a CTB in z (Morton) order,
/// so a lexicographic compare of `(CTB index, Morton code)` *is* decoding order.
/// Comparing the neighbour's sample against the current block's origin is exact
/// because Morton order is hierarchical: the origin is the smallest code in its
/// block, and every sample of an earlier block sorts below all of them.
fn available(px: i32, py: i32, bx: u32, by: u32, w: u32, h: u32, ctb: u32, nx: u32) -> bool {
    available_at(px, py, decode_order(bx, by, ctb, nx), w, h, ctb, nx)
}

/// [`available`] with the current block's place in decoding order already known.
///
/// Every one of a block's `4n+1` reference samples asks the same question about
/// the same block, so recomputing that half of the comparison per sample made
/// this the hottest thing in the encoder.
#[inline]
fn available_at(px: i32, py: i32, here: (u32, u32), w: u32, h: u32, ctb: u32, nx: u32) -> bool {
    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
        return false;
    }
    decode_order(px as u32, py as u32, ctb, nx) < here
}

#[inline]
fn decode_order(px: u32, py: u32, ctb: u32, nx: u32) -> (u32, u32) {
    // `ctb` is a power of two, but nothing in the type says so and the compiler
    // was emitting real integer divisions — two divisions and two remainders per
    // call, on the encoder's hottest path.
    let sh = ctb.trailing_zeros();
    let mask = ctb - 1;
    ((py >> sh) * nx + (px >> sh), morton(px & mask, py & mask))
}

/// Interleave the bits of `x` and `y` (y in the odd positions) — the z-order
/// curve HEVC scans transform blocks along.
#[inline]
fn morton(x: u32, y: u32) -> u32 {
    spread(x) | (spread(y) << 1)
}

/// Move each of the low 16 bits of `v` into an even position and zero the odd
/// ones: the doubling-and-masking ladder, five steps against the sixteen-round
/// loop it replaces.
#[inline]
fn spread(v: u32) -> u32 {
    let mut v = v & 0xFFFF;
    v = (v | (v << 8)) & 0x00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333;
    v = (v | (v << 1)) & 0x5555_5555;
    v
}

/// `scanIdx` (H.265 §7.4.9.11). 4x4 blocks and 8x8 luma pick their coefficient
/// scan from the prediction mode; everything larger scans diagonally.
fn scan_idx(mode: u8, log2: u32, chroma: bool) -> u8 {
    if !(log2 == 2 || (log2 == 3 && !chroma)) {
        return 0;
    }
    match mode {
        6..=14 => 2,  // near-horizontal prediction -> vertical scan
        22..=30 => 1, // near-vertical prediction -> horizontal scan
        _ => 0,
    }
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
    nx: u32,
) -> Vec<i32> {
    let mut out = vec![0i32; (4 * n + 1) as usize];
    refs_generic_into(recon, w, h, x, y, n, ctb, nx, &mut out);
    out
}

fn refs_generic_into(
    recon: &[u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    n: u32,
    ctb: u32,
    nx: u32,
    out: &mut [i32],
) {
    let ni = n as i32;
    let (xi, yi) = (x as i32, y as i32);
    let sample = |px: i32, py: i32| recon[(py as u32 * w + px as u32) as usize] as i32;
    // Constant for every one of the 4n+1 reference samples below.
    let here = decode_order(x, y, ctb, nx);

    let vals = out;
    vals.fill(0);
    let mut avail = vec![false; (4 * n + 1) as usize];
    for k in 0..(2 * ni) {
        let py = yi + (2 * ni - 1 - k); // left column p[-1][2n-1-k]
        if available_at(xi - 1, py, here, w, h, ctb, nx) {
            vals[k as usize] = sample(xi - 1, py);
            avail[k as usize] = true;
        }
    }
    if available_at(xi - 1, yi - 1, here, w, h, ctb, nx) {
        vals[(2 * ni) as usize] = sample(xi - 1, yi - 1); // corner
        avail[(2 * ni) as usize] = true;
    }
    for k in 0..(2 * ni) {
        let px = xi + k; // top row p[k][-1]
        if available_at(px, yi - 1, here, w, h, ctb, nx) {
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
}


#[allow(clippy::too_many_arguments)]
fn refs_luma(recon: &[u8], w: u32, h: u32, x: u32, y: u32, n: u32, nx: u32, ctb: u32) -> Vec<i32> {
    refs_generic(recon, w, h, x, y, n, ctb, nx)
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
#[allow(dead_code)]
fn extract(vals: &[i32], n: usize, mode: u8, luma: bool) -> (Vec<i32>, Vec<i32>) {
    let (mut a, mut l) = (vec![0i32; 2 * n + 1], vec![0i32; 2 * n + 1]);
    extract_into(vals, n, mode, luma, &mut a, &mut l);
    (a, l)
}

/// [`extract`] into caller-owned buffers.
fn extract_into(vals: &[i32], n: usize, mode: u8, luma: bool, above: &mut [i32], left: &mut [i32]) {
    if luma && filter_flag(mode, n) {
        split_refs_into(&smooth_refs(vals), n, above, left);
    } else {
        split_refs_into(vals, n, above, left);
    }
}

/// The `[1 2 1]` smoothing of §8.4.4.2.3, applied to the whole reference chain.
fn smooth_refs(vals: &[i32]) -> Vec<i32> {
    let mut f = vec![0i32; vals.len()];
    smooth_refs_into(vals, &mut f);
    f
}

fn smooth_refs_into(vals: &[i32], out: &mut [i32]) {
    let last = vals.len() - 1;
    out[0] = vals[0];
    out[last] = vals[last];
    for i in 1..last {
        out[i] = (vals[i - 1] + 2 * vals[i] + vals[i + 1] + 2) >> 2;
    }
}

/// Split the reference chain into the above and left rays, both starting at the
/// corner.
fn split_refs(src: &[i32], n: usize) -> (Vec<i32>, Vec<i32>) {
    let (mut a, mut l) = (vec![0i32; 2 * n + 1], vec![0i32; 2 * n + 1]);
    split_refs_into(src, n, &mut a, &mut l);
    (a, l)
}

fn split_refs_into(src: &[i32], n: usize, above: &mut [i32], left: &mut [i32]) {
    let corner = 2 * n;
    above[0] = src[corner];
    left[0] = src[corner];
    for k in 1..=2 * n {
        above[k] = src[corner + k];
        left[k] = src[corner - k];
    }
}

// ---- block helpers ----

fn gather_into(plane: &[u8], w: u32, x: u32, y: u32, n: u32, out: &mut [i32]) {
    for yy in 0..n {
        let row = ((y + yy) * w + x) as usize;
        let dst = (yy * n) as usize;
        for (o, &v) in out[dst..dst + n as usize]
            .iter_mut()
            .zip(&plane[row..row + n as usize])
        {
            *o = i32::from(v);
        }
    }
}

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
