//! AV1 decoder: a port of dav1d 1.5.1 (single-threaded C paths), the
//! decoder the AV1 conformance suite and libaom agree with bit for bit.
//!
//! [`Av1Decoder::decode`] takes one temporal unit (an IVF/WebM/MP4 sample:
//! a sequence of OBUs) and returns the pictures it shows, film grain
//! applied.

// The arithmetic mirrors dav1d's C expressions term for term.
#![allow(clippy::precedence)]

pub mod cdf;
// Field names follow dav1d's `CdfContext` (`classN`, ...).
#[allow(non_snake_case)]
pub mod cdf_data;
pub mod decode;
pub mod env;
pub mod fg;
pub mod filter;
pub mod frame;
pub mod intra_edge;
pub mod ipred;
pub mod itx;
pub mod itx_1d;
pub mod levels;
pub mod lf_mask;
pub mod mc;
pub mod msac;
pub mod obu;
pub mod picture;
pub mod qm;
pub mod recon;
pub mod refmvs;
mod sched;
#[allow(dead_code)]
pub mod tables;
pub mod warpmv;
pub mod wedge;

use std::sync::Arc;

use cdf_data::CdfContext;
use frame::{Pending, RefSlot, SlotEntry, TileData};
use obu::*;
pub use picture::Picture;

/// A shown picture.
pub struct Av1Frame {
    /// The reconstructed picture, film grain applied.
    pub pic: Arc<Picture>,
    /// The frame header it was decoded with.
    pub hdr: Arc<FrameHeader>,
    /// The sequence header in force.
    pub seq: Arc<SeqHeader>,
}

impl Av1Frame {
    /// As a [`DecodedFrame`](crate::decode::DecodedFrame): planes
    /// tightly packed, the sequence header's `matrix_coefficients` as the
    /// colour matrix (0 with identity = RGB, planes G, B, R).
    pub fn to_decoded(&self) -> crate::decode::DecodedFrame {
        use crate::decode::{ChromaFormat, ColorInfo, DecodedFrame};
        let p = &self.pic;
        let chroma = match p.layout {
            Layout::I400 => ChromaFormat::Monochrome,
            Layout::I420 => ChromaFormat::Yuv420,
            Layout::I422 => ChromaFormat::Yuv422,
            Layout::I444 => ChromaFormat::Yuv444,
        };
        let plane = |pl: usize| -> Vec<u16> {
            if pl > 0 && p.layout == Layout::I400 {
                return Vec::new();
            }
            let (sx, sy) = p.ss(pl);
            let (w, h) = (((p.w + sx) >> sx) as usize, ((p.h + sy) >> sy) as usize);
            let s = p.pstride(pl);
            let mut v = Vec::with_capacity(w * h);
            for y in 0..h {
                v.extend_from_slice(&p.planes[pl][y * s..y * s + w]);
            }
            v
        };
        DecodedFrame {
            width: p.w as u32,
            height: p.h as u32,
            bit_depth: p.bpc as u8,
            chroma,
            y: plane(0),
            u: plane(1),
            v: plane(2),
            color: ColorInfo::from_vui(
                Some(self.seq.mtrx as u32),
                self.seq.color_range,
                p.h as u32,
            ),
        }
    }
}

/// A hardware back end for [`Av1Decoder`]: the decoder parses the OBUs and
/// keeps the reference slots; the accelerator reconstructs each frame. Film
/// grain stays with the decoder, applied to what the accelerator reads back,
/// so the output is the native decoder's either way.
pub trait Av1Accel {
    /// Decode one frame.
    fn frame(&mut self, f: &AccelFrame<'_>) -> Result<(), String>;
    /// The samples of a picture it decoded (before film grain).
    fn read(&mut self, pic: &Picture) -> Result<Picture, String>;
}

/// Everything an [`Av1Accel`] is told about one frame.
pub struct AccelFrame<'a> {
    /// The sequence header in force.
    pub seq: &'a SeqHeader,
    /// The frame header.
    pub hdr: &'a FrameHeader,
    /// The tile groups: their tile data (tile sizes included) and tile range.
    pub tiles: &'a [TileData],
    /// The reference slots as the frame finds them.
    pub refs: &'a [Option<RefSlot>; 8],
    /// The picture being decoded (size and format; no samples).
    pub cur: &'a Arc<Picture>,
}

/// An AV1 decoder (a port of dav1d 1.5.1): feed it temporal units with
/// [`decode`](Self::decode) (Section 5 OBUs, as in IVF, MP4 and Matroska) or
/// [`decode_annexb`](Self::decode_annexb), and it returns the frames they show.
#[derive(Default)]
pub struct Av1Decoder {
    seq: Option<Arc<SeqHeader>>,
    refs: [Option<SlotEntry>; 8],
    frame_hdr: Option<FrameHeader>,
    tiles: Vec<TileData>,
    n_tiles: u32,
    operating_point_idc: u16,
    accel: Option<Box<dyn Av1Accel>>,
    /// Lent to each frame for its tile-parallel decode (`Fd::tile_pool`).
    tile_pool: Vec<decode::TileBufs>,
    /// Takes frames during [`decode_each`](Self::decode_each), to be decoded
    /// concurrently; `None` decodes each as it is parsed.
    sched: Option<Arc<sched::Sched>>,
}

/// A frame parsed and waiting to be decoded: its tile data and the
/// reference slots as they stood before it.
pub(crate) struct FrameJob {
    seq: Arc<SeqHeader>,
    hdr: Arc<FrameHeader>,
    tiles: Vec<TileData>,
    refs: [Option<SlotEntry>; 8],
    done: Arc<Pending>,
}

/// A frame whose tiles are decoded and loop filters are not yet run.
struct Reconstructed {
    fd: decode::Fd,
    in_cdf: Arc<CdfContext>,
    out_cdf: Option<CdfContext>,
}

impl FrameJob {
    /// Which slots this frame reads: the seven it predicts from and the one
    /// its CDFs and segment map come from.
    fn needed_slots(&self) -> [bool; 8] {
        let mut need = [false; 8];
        if self.hdr.is_inter_or_switch() {
            for &i in &self.hdr.refidx {
                need[i as usize] = true;
            }
        }
        if self.hdr.primary_ref_frame != PRIMARY_REF_NONE {
            need[self.hdr.refidx[self.hdr.primary_ref_frame as usize] as usize] = true;
        }
        need
    }

    /// The frames that must finish before this one starts.
    fn dependencies(&self) -> Vec<Arc<Pending>> {
        let need = self.needed_slots();
        (0..8)
            .filter(|&i| need[i])
            .filter_map(|i| self.refs[i].as_ref().map(|e| e.done.clone()))
            .collect()
    }

    /// The slots this frame reads, each waited for.
    fn resolve_refs(&self) -> Result<[Option<RefSlot>; 8], String> {
        let need = self.needed_slots();
        let mut out: [Option<RefSlot>; 8] = Default::default();
        for i in 0..8 {
            if let (true, Some(e)) = (need[i], &self.refs[i]) {
                out[i] = Some(e.done.wait()?);
            }
        }
        Ok(out)
    }

    /// Entropy decoding and reconstruction, with `pool` lent for the tiles.
    fn reconstruct(&self, pool: &mut Vec<decode::TileBufs>) -> Result<Reconstructed, String> {
        let refs = self.resolve_refs()?;
        let (mut fd, in_cdf) = decode::Fd::new(self.seq.clone(), self.hdr.clone(), &refs)?;
        fd.tile_pool = std::mem::take(pool);
        let out_cdf = fd.decode_tiles(&in_cdf, &self.tiles);
        *pool = std::mem::take(&mut fd.tile_pool);
        Ok(Reconstructed {
            fd,
            in_cdf,
            out_cdf: out_cdf?,
        })
    }
}

impl Reconstructed {
    /// Loop filters, then what a reference slot keeps of the frame.
    fn finish(mut self) -> RefSlot {
        self.fd.apply_filters();
        let fd = &mut self.fd;
        let hdr = fd.hdr.clone();
        let cdf = if hdr.refresh_context {
            Arc::new(self.out_cdf.unwrap_or(*self.in_cdf))
        } else {
            self.in_cdf
        };
        let pic = Arc::new(match fd.sr_cur.take() {
            Some(sr) => sr,
            None => std::mem::replace(&mut fd.cur, Picture::new(1, 1, fd.layout, fd.bpc)),
        });
        let segmap = fd.cur_segmap.take().map(Arc::new);
        let refmvs = if (hdr.is_inter_or_switch() || hdr.allow_intrabc) && !hdr.allow_intrabc {
            fd.rf.take().map(|rf| Arc::new(rf.rp))
        } else {
            None
        };
        RefSlot {
            pic,
            hdr: hdr.clone(),
            seq: fd.seq.clone(),
            cdf,
            segmap,
            refmvs,
            refpoc: fd.refpoc,
            showable: hdr.showable_frame,
        }
    }
}

/// `f`, with a panic turned into an error. A frame thread must always fill
/// in its frame's [`Pending`]: later frames are blocked on it, and a corrupt
/// stream that panics the decoder would otherwise hang them for good.
fn caught<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("frame decode panicked".into()))
}

/// A header with its global motion models' shear parameters filled in, as
/// `Fd::new` makes it; the slots keep this form for later frames to parse
/// against.
fn with_shear(hdr: &Arc<FrameHeader>) -> Arc<FrameHeader> {
    let mut h = (**hdr).clone();
    for g in h.gmv.iter_mut() {
        let _ = warpmv::get_shear_params(g);
    }
    Arc::new(h)
}

const OBU_SEQ_HDR: u32 = 1;
const OBU_TD: u32 = 2;
const OBU_FRAME_HDR: u32 = 3;
const OBU_TILE_GRP: u32 = 4;
const OBU_FRAME: u32 = 6;
const OBU_REDUNDANT_FRAME_HDR: u32 = 7;

impl Av1Decoder {
    /// A decoder waiting for a sequence header.
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand reconstruction to `accel` (see [`Av1Accel`]).
    pub fn set_accel(&mut self, accel: Box<dyn Av1Accel>) {
        self.accel = Some(accel);
    }

    /// Shown pictures from the accelerator into memory, then film grain.
    fn finish_output(&mut self, out: &mut [Av1Frame]) -> Result<(), String> {
        for f in out.iter_mut() {
            if let Some(a) = self.accel.as_mut() {
                f.pic = Arc::new(a.read(&f.pic)?);
            }
            if f.hdr.film_grain_present {
                f.pic = Arc::new(fg::apply_grain(&f.pic, &f.hdr.film_grain, &f.seq));
            }
        }
        Ok(())
    }

    fn ref_headers(&self) -> RefHeaders {
        let mut r: RefHeaders = Default::default();
        for (i, s) in self.refs.iter().enumerate() {
            r[i] = s.as_ref().map(|s| s.hdr.clone());
        }
        r
    }

    /// Decode one temporal unit (low-overhead OBU format).
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<Av1Frame>, String> {
        let shown = self.parse_temporal_unit(data)?;
        self.output(shown)
    }

    /// Every OBU of one temporal unit; returns the frames it shows.
    fn parse_temporal_unit(&mut self, data: &[u8]) -> Result<Vec<Arc<Pending>>, String> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            pos += self.parse_obu(&data[pos..], &mut out)?;
        }
        Ok(out)
    }

    /// The pictures a temporal unit shows, once decoded: one per unit, the
    /// last (highest spatial layer), as libaom with output_all_layers=0.
    fn output(&mut self, mut shown: Vec<Arc<Pending>>) -> Result<Vec<Av1Frame>, String> {
        if shown.len() > 1 {
            shown.drain(..shown.len() - 1);
        }
        let mut out = Vec::with_capacity(shown.len());
        for p in shown {
            let s = p.wait()?;
            out.push(Av1Frame {
                pic: s.pic,
                hdr: s.hdr,
                seq: s.seq,
            });
        }
        self.finish_output(&mut out)?;
        Ok(out)
    }

    /// Decode many temporal units (low-overhead OBU format), up to `threads`
    /// frames at once, and give each shown frame to `each` in order, with the
    /// index of its unit. The frames are those calling
    /// [`decode`](Self::decode) on each unit would return.
    ///
    /// A unit that will not decode is skipped, as a player would: real files
    /// have damaged frames, and the rest are still worth having. Returns how
    /// many units gave up a frame that way, or the first error if not one
    /// frame could be shown. An error from `each` stops at once.
    ///
    /// Frames run on the `parallel` feature's thread pool as their references
    /// finish, and only a window of units is in flight, so memory does not
    /// grow with the stream. How much this helps depends on the stream: an
    /// encoder's layered frame structure leaves frames that do not reference
    /// each other, and a strict chain leaves none.
    ///
    /// This call blocks until frames are done, which a browser forbids on its
    /// main thread: from a threaded wasm build, call it in a Worker. It
    /// decodes one frame at a time where no pool can help: without `parallel`
    /// or a second pool thread, from inside the pool, or with an accelerator.
    pub fn decode_each(
        &mut self,
        units: &[&[u8]],
        threads: usize,
        mut each: impl FnMut(usize, Av1Frame) -> Result<(), String>,
    ) -> Result<usize, String> {
        let (mut shown, mut skipped, mut first_error) = (0usize, 0usize, None);
        let mut failed = |e: String, first_error: &mut Option<String>| {
            skipped += 1;
            first_error.get_or_insert(e);
        };
        let concurrent = self.accel.is_none() && threads >= 2 && crate::pool_can_serve_caller();
        if !concurrent {
            for (i, u) in units.iter().enumerate() {
                match self.decode(u) {
                    Ok(frames) => {
                        for f in frames {
                            shown += 1;
                            each(i, f)?;
                        }
                    }
                    Err(e) => failed(e, &mut first_error),
                }
            }
        } else {
            let sched = sched::Sched::new(threads, std::mem::take(&mut self.tile_pool));
            self.sched = Some(sched.clone());
            // Far enough ahead to reach the frames that can run side by side:
            // a unit's shown frame can need several hidden ones decoded first.
            let window = threads * 4;
            let mut waiting = std::collections::VecDeque::new();
            let mut run = || -> Result<(), String> {
                let mut deliver = |this: &mut Self,
                                   (i, unit): (usize, Vec<Arc<Pending>>)|
                 -> Result<(), String> {
                    match this.output(unit) {
                        Ok(frames) => {
                            for f in frames {
                                shown += 1;
                                each(i, f)?;
                            }
                        }
                        Err(e) => failed(e, &mut first_error),
                    }
                    Ok(())
                };
                for (i, u) in units.iter().enumerate() {
                    match self.parse_temporal_unit(u) {
                        Ok(s) => waiting.push_back((i, s)),
                        Err(e) => deliver(self, (i, vec![Pending::ready(Err(e))]))?,
                    }
                    while waiting.len() > window {
                        deliver(self, waiting.pop_front().unwrap())?;
                    }
                }
                while let Some(unit) = waiting.pop_front() {
                    deliver(self, unit)?;
                }
                Ok(())
            };
            let r = run();
            self.sched = None;
            if r.is_err() {
                sched.cancel();
            }
            let (pool, hidden_error) = sched.finish();
            self.tile_pool = pool;
            r?;
            if let Some(e) = hidden_error {
                first_error.get_or_insert(e);
            }
        }
        match first_error {
            Some(e) if shown == 0 => Err(e),
            _ => Ok(skipped),
        }
    }

    /// [`decode_each`](Self::decode_each), collecting the frames.
    pub fn decode_batch(
        &mut self,
        units: &[&[u8]],
        threads: usize,
    ) -> Result<Vec<Av1Frame>, String> {
        let mut out = Vec::new();
        self.decode_each(units, threads, |_, f| {
            out.push(f);
            Ok(())
        })?;
        Ok(out)
    }

    /// Decode one Annex B `temporal_unit()` payload (frame units of
    /// length-prefixed OBUs).
    pub fn decode_annexb(&mut self, tu: &[u8]) -> Result<Vec<Av1Frame>, String> {
        let mut obus = Vec::new();
        let mut p = 0;
        while p < tu.len() {
            let (fsz, n) = leb128(&tu[p..])?;
            p += n;
            let fu = tu.get(p..p + fsz).ok_or("frame unit overrun")?;
            p += fsz;
            let mut q = 0;
            while q < fu.len() {
                let (osz, n) = leb128(&fu[q..])?;
                q += n;
                obus.push(fu.get(q..q + osz).ok_or("OBU overrun")?);
                q += osz;
            }
        }
        let mut shown = Vec::new();
        for o in obus {
            self.parse_obu(o, &mut shown)?;
        }
        self.output(shown)
    }

    /// `dav1d_parse_obus` for one OBU; returns the bytes consumed.
    fn parse_obu(&mut self, data: &[u8], out: &mut Vec<Arc<Pending>>) -> Result<usize, String> {
        let mut gb = GetBits::new(data);
        gb.bit();
        let typ = gb.bits(4);
        let has_extension = gb.flag();
        let has_length_field = gb.flag();
        gb.bit();
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if has_extension {
            temporal_id = gb.bits(3) as u8;
            spatial_id = gb.bits(2) as u8;
            gb.bits(3);
        }
        let start = gb.byte_pos();
        let end = if has_length_field {
            let len = gb.uleb128() as usize;
            let s = gb.byte_pos();
            if len > data.len() - s {
                return Err("OBU length overrun".into());
            }
            s + len
        } else {
            data.len()
        };
        let _ = start;
        let payload_start = gb.byte_pos();
        let payload = &data[payload_start..end];

        if typ != OBU_SEQ_HDR && typ != OBU_TD && has_extension && self.operating_point_idc != 0 {
            let in_t = (self.operating_point_idc >> temporal_id) & 1;
            let in_s = (self.operating_point_idc >> (spatial_id + 8)) & 1;
            if in_t == 0 || in_s == 0 {
                return Ok(end);
            }
        }

        match typ {
            OBU_SEQ_HDR => {
                let mut g = GetBits::new(payload);
                let seq = parse_seq_hdr(&mut g)?;
                self.operating_point_idc = seq.operating_points[0].idc;
                let same = self
                    .seq
                    .as_ref()
                    .map(|s| s.same_sequence(&seq))
                    .unwrap_or(false);
                if !same {
                    if self.seq.is_some() {
                        self.refs = Default::default();
                    }
                    self.frame_hdr = None;
                }
                self.seq = Some(Arc::new(seq));
            }
            OBU_REDUNDANT_FRAME_HDR if self.frame_hdr.is_some() => {}
            OBU_FRAME | OBU_FRAME_HDR | OBU_REDUNDANT_FRAME_HDR => {
                let seq = self.seq.clone().ok_or("frame before sequence header")?;
                let mut g = GetBits::new(payload);
                let hdr =
                    parse_frame_hdr(&seq, &self.ref_headers(), &mut g, temporal_id, spatial_id)?;
                self.tiles.clear();
                self.n_tiles = 0;
                let show_existing = hdr.show_existing_frame;
                self.frame_hdr = Some(hdr);
                if typ == OBU_FRAME {
                    if show_existing {
                        return Err("OBU_FRAME with show_existing_frame".into());
                    }
                    g.byte_align();
                    let rest = &payload[g.byte_pos()..];
                    self.tile_group(rest)?;
                }
            }
            OBU_TILE_GRP => {
                if self.frame_hdr.is_none() {
                    return Err("tile group without frame header".into());
                }
                self.tile_group(payload)?;
            }
            _ => {}
        }

        if let (Some(seq), Some(hdr)) = (self.seq.clone(), self.frame_hdr.as_ref()) {
            if hdr.show_existing_frame {
                let idx = hdr.existing_frame_idx as usize;
                let entry = self.refs[idx]
                    .clone()
                    .ok_or("show_existing_frame of an empty slot")?;
                out.push(entry.done.clone());
                if entry.hdr.frame_type == FRAME_KEY {
                    // Showing a key frame again refreshes every slot with it,
                    // which needs its decoded data: wait for it here.
                    let slot = entry.done.wait()?;
                    let mut s = slot.clone();
                    s.showable = false;
                    for i in 0..8 {
                        let mut c = s.clone();
                        if i != idx {
                            c.refmvs = None;
                        } else {
                            c.refmvs = slot.refmvs.clone();
                        }
                        self.refs[i] = Some(SlotEntry {
                            hdr: entry.hdr.clone(),
                            seq: entry.seq.clone(),
                            done: Pending::ready(Ok(c)),
                        });
                    }
                }
                self.frame_hdr = None;
            } else if self.n_tiles == hdr.tiling.cols as u32 * hdr.tiling.rows as u32 {
                let hdr = Arc::new(self.frame_hdr.take().unwrap());
                let tiles = std::mem::take(&mut self.tiles);
                self.n_tiles = 0;
                self.submit_frame(seq, hdr, tiles, out)?;
            }
        }
        Ok(end)
    }

    fn tile_group(&mut self, data: &[u8]) -> Result<(), String> {
        let hdr = self.frame_hdr.as_ref().unwrap();
        let mut g = GetBits::new(data);
        let (s, e) = parse_tile_hdr(hdr, &mut g);
        g.byte_align();
        if s > e || s != self.n_tiles {
            return Err("tile groups out of order".into());
        }
        self.tiles.push(TileData {
            data: data[g.byte_pos()..].to_vec(),
            start: s,
            end: e,
        });
        self.n_tiles += 1 + e - s;
        Ok(())
    }

    fn submit_frame(
        &mut self,
        seq: Arc<SeqHeader>,
        hdr: Arc<FrameHeader>,
        tiles: Vec<TileData>,
        out: &mut Vec<Arc<Pending>>,
    ) -> Result<(), String> {
        let done = Pending::new();
        let refs = self.refs.clone();
        let entry = SlotEntry {
            hdr: with_shear(&hdr),
            seq: seq.clone(),
            done: done.clone(),
        };
        for i in 0..8 {
            if hdr.refresh_frame_flags & (1 << i) != 0 {
                self.refs[i] = Some(entry.clone());
            }
        }
        if hdr.show_frame {
            out.push(done.clone());
        }
        let job = FrameJob {
            seq,
            hdr,
            tiles,
            refs,
            done,
        };
        if let Some(sched) = &self.sched {
            sched.submit(job);
            return Ok(());
        }
        let r = if let Some(accel) = self.accel.as_mut() {
            let refs =
                std::array::from_fn(|i| job.refs[i].as_ref().and_then(|e| e.done.wait().ok()));
            let (seq, hdr) = (&job.seq, &job.hdr);
            let pic = Arc::new(Picture {
                planes: Default::default(),
                stride: [0; 2],
                w: hdr.width[1],
                h: hdr.height,
                layout: seq.layout,
                bpc: 8 + 2 * seq.hbd as u32,
            });
            accel
                .frame(&AccelFrame {
                    seq,
                    hdr,
                    tiles: &job.tiles,
                    refs: &refs,
                    cur: &pic,
                })
                .map(|()| {
                    let cdf = Arc::new(*CdfContext::default_for(hdr.quant.yac));
                    RefSlot {
                        pic,
                        hdr: hdr.clone(),
                        seq: seq.clone(),
                        cdf,
                        segmap: None,
                        refmvs: None,
                        refpoc: [0; 7],
                        showable: hdr.showable_frame,
                    }
                })
        } else {
            let pool = &mut self.tile_pool;
            caught(|| job.reconstruct(pool).map(Reconstructed::finish))
        };
        job.done.set(r.clone());
        r.map(|_| ())
    }
}

/// Split an Annex B stream into `temporal_unit()` payloads.
pub fn split_annexb(data: &[u8]) -> Result<Vec<&[u8]>, String> {
    let mut v = Vec::new();
    let mut p = 0;
    while p < data.len() {
        let (sz, n) = leb128(&data[p..])?;
        p += n;
        v.push(data.get(p..p + sz).ok_or("temporal unit overrun")?);
        p += sz;
    }
    Ok(v)
}

fn leb128(d: &[u8]) -> Result<(usize, usize), String> {
    let mut v = 0usize;
    for (i, &b) in d.iter().enumerate().take(8) {
        v |= ((b & 0x7f) as usize) << (7 * i);
        if b & 0x80 == 0 {
            return Ok((v, i + 1));
        }
    }
    Err("bad leb128".into())
}
