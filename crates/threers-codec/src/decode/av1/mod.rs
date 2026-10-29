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
#[allow(dead_code)]
pub mod tables;
pub mod warpmv;
pub mod wedge;

use std::sync::Arc;

use cdf_data::CdfContext;
use frame::{RefSlot, TileData};
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
            color: ColorInfo::from_vui(Some(self.seq.mtrx as u32), self.seq.color_range, p.h as u32),
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
    refs: [Option<RefSlot>; 8],
    frame_hdr: Option<FrameHeader>,
    tiles: Vec<TileData>,
    n_tiles: u32,
    operating_point_idc: u16,
    accel: Option<Box<dyn Av1Accel>>,
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
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            pos += self.parse_obu(&data[pos..], &mut out)?;
        }
        // One picture per temporal unit, the last (highest spatial layer)
        // shown, as libaom with output_all_layers=0.
        if out.len() > 1 {
            out.drain(..out.len() - 1);
        }
        self.finish_output(&mut out)?;
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
        let mut out = Vec::new();
        for o in obus {
            self.parse_obu(o, &mut out)?;
        }
        if out.len() > 1 {
            out.drain(..out.len() - 1);
        }
        self.finish_output(&mut out)?;
        Ok(out)
    }

    /// `dav1d_parse_obus` for one OBU; returns the bytes consumed.
    fn parse_obu(&mut self, data: &[u8], out: &mut Vec<Av1Frame>) -> Result<usize, String> {
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
                let same = self.seq.as_ref().map(|s| s.same_sequence(&seq)).unwrap_or(false);
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
                let hdr = parse_frame_hdr(&seq, &self.ref_headers(), &mut g, temporal_id, spatial_id)?;
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
                let slot = self.refs[idx].clone().ok_or("show_existing_frame of an empty slot")?;
                out.push(Av1Frame { pic: slot.pic.clone(), hdr: slot.hdr.clone(), seq: slot.seq.clone() });
                if slot.hdr.frame_type == FRAME_KEY {
                    let mut s = slot.clone();
                    s.showable = false;
                    for i in 0..8 {
                        let mut c = s.clone();
                        if i != idx {
                            c.refmvs = None;
                        } else {
                            c.refmvs = slot.refmvs.clone();
                        }
                        self.refs[i] = Some(c);
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
        self.tiles.push(TileData { data: data[g.byte_pos()..].to_vec(), start: s, end: e });
        self.n_tiles += 1 + e - s;
        Ok(())
    }

    fn submit_frame(&mut self, seq: Arc<SeqHeader>, hdr: Arc<FrameHeader>, tiles: Vec<TileData>, out: &mut Vec<Av1Frame>) -> Result<(), String> {
        if let Some(accel) = self.accel.as_mut() {
            let layout = seq.layout;
            let pic = Arc::new(Picture { planes: Default::default(), stride: [0; 2], w: hdr.width[1], h: hdr.height, layout, bpc: 8 + 2 * seq.hbd as u32 });
            accel.frame(&AccelFrame { seq: &seq, hdr: &hdr, tiles: &tiles, refs: &self.refs, cur: &pic })?;
            let cdf = Arc::new(*CdfContext::default_for(hdr.quant.yac));
            let slot = RefSlot { pic: pic.clone(), hdr: hdr.clone(), seq: seq.clone(), cdf, segmap: None, refmvs: None, refpoc: [0; 7], showable: hdr.showable_frame };
            for i in 0..8 {
                if hdr.refresh_frame_flags & (1 << i) != 0 {
                    self.refs[i] = Some(slot.clone());
                }
            }
            if hdr.show_frame {
                out.push(Av1Frame { pic, hdr, seq });
            }
            return Ok(());
        }
        let (mut fd, in_cdf) = decode::Fd::new(seq.clone(), hdr.clone(), &self.refs)?;
        let hdr = fd.hdr.clone();
        let out_cdf = fd.decode_tiles(&in_cdf, &tiles)?;
        let cdf = if hdr.refresh_context { Arc::new(out_cdf.unwrap_or(*in_cdf)) } else { in_cdf };
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
        let slot = RefSlot { pic: pic.clone(), hdr: hdr.clone(), seq: seq.clone(), cdf, segmap, refmvs, refpoc: fd.refpoc, showable: hdr.showable_frame };
        for i in 0..8 {
            if hdr.refresh_frame_flags & (1 << i) != 0 {
                self.refs[i] = Some(slot.clone());
            }
        }
        if hdr.show_frame {
            out.push(Av1Frame { pic, hdr, seq });
        }
        Ok(())
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
