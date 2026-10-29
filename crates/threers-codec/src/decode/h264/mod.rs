//! A from-scratch H.264 / AVC decoder.
//!
//! Progressive (frame-coded) streams: Baseline, Main and High profiles —
//! CAVLC and CABAC, I/P/B slices, multiple slices, 8×8 transforms, scaling
//! matrices, weighted prediction (explicit and implicit), spatial and
//! temporal direct, long-term references and every memory management
//! operation, 4:2:0, 4:2:2 (High 4:2:2) or monochrome, 8 bits and up. Not
//! supported, and refused with an error rather than decoded wrongly:
//! interlaced coding (field pictures and MBAFF), 4:4:4, slice groups (FMO),
//! data partitioning, and SP/SI slices.
//!
//! Feed it NAL units with [`H264Decoder::decode_nal`]; each finished picture
//! comes out of [`H264Decoder::finish_picture`] (called for you when a new
//! picture starts) tagged with its picture order count.

#[cfg(feature = "decode")]
pub mod cabac_init;
#[cfg(feature = "decode")]
pub mod cavlc;
#[cfg(feature = "decode")]
pub mod deblock;
#[cfg(feature = "decode")]
pub mod inter;
#[cfg(feature = "decode")]
pub mod intra;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "decode"), allow(dead_code))]
pub mod params;
#[cfg(feature = "decode")]
pub mod parse_cabac;
#[cfg(feature = "decode")]
pub mod parse_cavlc;
#[cfg(feature = "decode")]
pub mod slice;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "decode"), allow(dead_code))]
pub mod tables;
#[cfg(feature = "decode")]
pub mod types;

#[cfg(feature = "decode")]
use std::sync::Arc;

#[cfg(feature = "decode")]
use self::cavlc::CavlcTables;
#[cfg(feature = "decode")]
use self::params::{Mmco, Pps, SliceHeader, SliceType, Sps};
#[cfg(feature = "decode")]
use self::slice::{CurPic, SliceDeblock, SliceDec};
#[cfg(feature = "decode")]
use self::types::{MbInfo, RefEntry, RefPic};
#[cfg(feature = "decode")]
use super::bits::unescape;
#[cfg(feature = "decode")]
use super::picture::{ChromaFormat, ColorInfo, DecodedFrame, Plane};

/// Decoding errors are messages naming what was malformed or unsupported.
#[cfg(feature = "decode")]
pub type Result<T> = std::result::Result<T, String>;

#[cfg(feature = "decode")]
/// A decoded picture and where it goes in display order.
#[derive(Debug, Clone)]
pub struct H264Picture {
    /// The picture, cropped to its display size.
    pub frame: DecodedFrame,
    /// `PicOrderCnt`. Resets at every IDR picture (and memory management
    /// operation 5), so it orders pictures only within one such period.
    pub poc: i32,
    /// Whether this picture is an IDR picture, which starts a new POC
    /// period.
    pub idr: bool,
}

#[cfg(feature = "decode")]
/// The picture (frame or field) being decoded, with what marking needs
/// once it is done.
struct Current {
    pic: CurPic,
    sps: Sps,
    /// 1 top field, 2 bottom field, 3 frame.
    structure: u8,
    /// `MbaffFrameFlag`.
    mbaff: bool,
    /// Index of its frame store in the DPB.
    store: usize,
    /// Second field of a frame whose first field is in `store`.
    second_field: bool,
    idr: bool,
    nal_ref_idc: u32,
    frame_num: u32,
    long_term_reference: bool,
    mmcos: Option<Vec<Mmco>>,
    /// For detecting the first slice of the next picture.
    pps_id: u32,
    poc_lsb: u32,
    poc_msb: i32,
    frame_num_offset: i32,
}

/// A hardware back end for [`H264Decoder`].
///
/// The decoder keeps doing everything but reconstruction: it parses the
/// parameter sets and slice headers, numbers and marks references, builds
/// the reference lists and orders output. Each slice is handed over with all
/// of that, the accelerator reconstructs the picture, and when a frame is
/// complete the decoder asks it for the samples. This is the shape of VAAPI
/// and the other "the driver does not parse" APIs.
#[cfg(feature = "decode")]
pub trait H264Accel {
    /// Decode one slice of the current picture.
    fn slice(&mut self, s: &AccelSlice<'_>) -> Result<()>;
    /// The current picture (frame or field) has had all its slices.
    fn end_picture(&mut self) -> Result<()>;
    /// The samples of the frame in store `store_id`, cropped as `sps` says.
    fn frame(&mut self, store_id: u32, sps: &Sps) -> Result<DecodedFrame>;
}

/// Everything an [`H264Accel`] is told about one slice.
#[cfg(feature = "decode")]
pub struct AccelSlice<'a> {
    /// The NAL unit as coded: header byte included, emulation prevention in.
    pub nal: &'a [u8],
    /// The active sequence parameter set.
    pub sps: &'a Sps,
    /// The active picture parameter set.
    pub pps: &'a Pps,
    /// The parsed slice header (`data_bit_offset` is where slice data starts, in RBSP bits after the header byte).
    pub hdr: &'a SliceHeader,
    /// The first slice of a new picture.
    pub first: bool,
    /// The frame store the picture is decoded into (shared by both fields).
    pub store_id: u32,
    /// 1 top field, 2 bottom field, 3 frame.
    pub structure: u8,
    /// `TopFieldOrderCnt`, `BottomFieldOrderCnt` of the picture.
    pub field_poc: [i32; 2],
    /// The frame stores, with their reference marking.
    pub dpb: &'a [RefPic],
    /// `RefPicList0` and `RefPicList1`, entries indexing `dpb`.
    pub refs: &'a [Vec<RefEntry>; 2],
}

/// An H.264 decoder fed one NAL unit at a time ([`decode_nal`](Self::decode_nal)).
///
/// Most callers want [`NativeDecoder`](crate::decode::NativeDecoder), which
/// unpacks MP4 samples and orders the output; this is the layer below it.
#[cfg(feature = "decode")]
pub struct H264Decoder {
    spss: Vec<Option<Sps>>,
    ppss: Vec<Option<Pps>>,
    tables: CavlcTables,
    cur: Option<Current>,
    /// Frame stores: references, plus the frame whose first field waits for
    /// its pair.
    dpb: Vec<RefPic>,
    next_id: u32,
    max_lt_idx: Option<u32>,
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num_offset: i32,
    prev_frame_num: u32,
    prev_ref_frame_num: u32,
    prev_had_mmco5: bool,
    /// A decoded first field waiting for its pair: `(store id, structure,
    /// frame_num, reference)`.
    pending: Option<(u32, u8, u32, bool)>,
    /// Size of the last activated SPS, to notice a resolution change.
    active_size: Option<(u32, u32)>,
    done: Vec<H264Picture>,
    accel: Option<Box<dyn H264Accel>>,
    /// The first error an accelerator reported where nothing could return it.
    accel_error: Option<String>,
    /// The current picture has had a slice handed to the accelerator.
    accel_started: bool,
}

#[cfg(feature = "decode")]
impl Default for H264Decoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "decode")]
fn empty_planes() -> [Plane; 3] {
    [Plane::new(0, 0, 0), Plane::new(0, 0, 0), Plane::new(0, 0, 0)]
}

#[cfg(feature = "decode")]
impl H264Decoder {
    /// A decoder with no parameter sets yet.
    pub fn new() -> Self {
        H264Decoder {
            spss: vec![None; 32],
            ppss: vec![None; 256],
            tables: CavlcTables::new(),
            cur: None,
            dpb: Vec::new(),
            next_id: 1,
            max_lt_idx: None,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            prev_frame_num_offset: 0,
            prev_frame_num: 0,
            prev_ref_frame_num: 0,
            prev_had_mmco5: false,
            pending: None,
            active_size: None,
            done: Vec::new(),
            accel: None,
            accel_error: None,
            accel_started: false,
        }
    }

    /// Hand reconstruction to `accel` (see [`H264Accel`]).
    pub fn set_accel(&mut self, accel: Box<dyn H264Accel>) {
        self.accel = Some(accel);
    }

    /// The first error the accelerator reported from
    /// [`finish_picture`](Self::finish_picture), which cannot return one.
    pub fn take_accel_error(&mut self) -> Option<String> {
        self.accel_error.take()
    }

    /// Decode one NAL unit (without start code or length prefix).
    pub fn decode_nal(&mut self, nal: &[u8]) -> Result<()> {
        if nal.is_empty() {
            return Ok(());
        }
        let nal_ref_idc = ((nal[0] >> 5) & 3) as u32;
        let nal_type = (nal[0] & 31) as u32;
        match nal_type {
            1 | 5 => {
                let rbsp = unescape(&nal[1..]);
                self.decode_slice(nal, &rbsp, nal_type, nal_ref_idc)
            }
            2..=4 => Err("data-partitioned slices are not supported".into()),
            7 => {
                let rbsp = unescape(&nal[1..]);
                let sps = Sps::parse(&rbsp)?;
                let id = sps.id as usize;
                self.spss[id] = Some(sps);
                Ok(())
            }
            8 => {
                let rbsp = unescape(&nal[1..]);
                let pps = Pps::parse(&rbsp, &self.spss)?;
                let id = pps.id as usize;
                self.ppss[id] = Some(pps);
                Ok(())
            }
            9..=11 => {
                // Access unit delimiter / end of sequence / end of stream: the
                // current picture, if any, is complete.
                self.finish_picture();
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Pictures finished so far, in decode order.
    pub fn take_pictures(&mut self) -> Vec<H264Picture> {
        std::mem::take(&mut self.done)
    }

    fn decode_slice(&mut self, nal: &[u8], rbsp: &[u8], nal_type: u32, nal_ref_idc: u32) -> Result<()> {
        let hdr = SliceHeader::parse(rbsp, nal_type, nal_ref_idc, &self.spss, &self.ppss)?;
        if hdr.redundant_pic_cnt > 0 {
            return Ok(());
        }
        let structure = if !hdr.field_pic { 3 } else if hdr.bottom_field { 2 } else { 1 };
        let new_pic = match &self.cur {
            None => true,
            Some(c) => {
                hdr.first_mb == 0
                    || c.frame_num != hdr.frame_num
                    || c.pps_id != hdr.pps_id
                    || c.idr != hdr.idr
                    || (c.nal_ref_idc == 0) != (nal_ref_idc == 0)
                    || c.poc_lsb != hdr.poc_lsb
                    || c.structure != structure
            }
        };
        if new_pic {
            self.finish_picture();
            self.start_picture(&hdr)?;
        }
        let pps = self.ppss[hdr.pps_id as usize].clone().ok_or("missing PPS")?;
        let refs = self.build_lists(&hdr)?;
        if let Some(accel) = self.accel.as_mut() {
            let cur = self.cur.as_ref().unwrap();
            let sps = cur.sps.clone();
            let slice = AccelSlice {
                nal,
                sps: &sps,
                pps: &pps,
                hdr: &hdr,
                first: !self.accel_started,
                store_id: self.dpb[cur.store].id,
                structure,
                field_poc: cur.pic.field_poc,
                dpb: &self.dpb,
                refs: &refs,
            };
            self.accel_started = true;
            return accel.slice(&slice);
        }
        let frefs = if hdr.mbaff { self.field_lists(&refs) } else { Default::default() };
        let cur = self.cur.as_mut().unwrap();
        let slice_num = cur.pic.slices.len() as u32;
        cur.pic.slices.push(SliceDeblock {
            disable: hdr.disable_deblocking,
            alpha: hdr.alpha_offset,
            beta: hdr.beta_offset,
        });
        let mut dec = SliceDec::new(
            &cur.sps,
            &pps,
            &hdr,
            slice_num,
            &mut cur.pic,
            &self.dpb,
            &refs,
            &frefs,
            &self.tables,
        );
        dec.decode(rbsp)
    }

    /// MBAFF field macroblocks' lists (§8.4.2.1): each frame of `refs` as
    /// its two fields, the current macroblock's parity first; `[parity][list]`.
    fn field_lists(&mut self, refs: &[Vec<RefEntry>; 2]) -> [[Vec<RefEntry>; 2]; 2] {
        let mut out: [[Vec<RefEntry>; 2]; 2] = Default::default();
        for (par, lists) in out.iter_mut().enumerate() {
            for (l, list) in lists.iter_mut().enumerate() {
                for e in &refs[l] {
                    for k in 0..2 {
                        let fp = par ^ k;
                        let st = &mut self.dpb[e.dpb];
                        list.push(RefEntry {
                            dpb: e.dpb,
                            id: st.id * 4 + 1 + fp as u32,
                            poc: st.field_poc[fp],
                            long_term: e.long_term,
                            structure: 1 + fp as u8,
                            planes: st.field(fp),
                        });
                    }
                }
            }
        }
        out
    }

    /// Drop frame stores that are neither references nor the frame waiting
    /// for its second field (indices into `dpb` shift).
    fn prune(&mut self) {
        let keep = self.pending.map(|p| p.0);
        self.dpb.retain(|p| p.is_ref() || Some(p.id) == keep);
    }

    fn start_picture(&mut self, hdr: &SliceHeader) -> Result<()> {
        let pps = self.ppss[hdr.pps_id as usize].as_ref().ok_or("missing PPS")?;
        let sps = self.spss[pps.sps_id as usize]
            .clone()
            .ok_or("missing SPS")?;
        if sps.transform_bypass {
            return Err("lossless (transform bypass) coding is not supported".into());
        }
        let structure: u8 = if !hdr.field_pic { 3 } else if hdr.bottom_field { 2 } else { 1 };
        // h264_field_start: pair with a waiting first field of the other
        // parity and the same frame_num; an unmatched one is dropped from
        // display (but stays a reference if marked).
        let mut second_store = None;
        if let Some((id, s, fnum, _)) = self.pending.take() {
            if structure != 3 && structure != s && fnum == hdr.frame_num {
                second_store = self.dpb.iter().position(|p| p.id == id);
            }
            if second_store.is_none() {
                self.prune();
            }
        }
        let second_field = second_store.is_some();
        let size = (sps.width_mbs, sps.height_mbs);
        if self.active_size != Some(size) {
            self.dpb.clear();
            self.active_size = Some(size);
        }
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        if hdr.idr && !second_field {
            self.dpb.clear();
            self.max_lt_idx = None;
        } else if !second_field {
            let expected = (self.prev_ref_frame_num + 1) % max_frame_num;
            if hdr.frame_num != self.prev_ref_frame_num && hdr.frame_num != expected {
                self.fill_frame_num_gap(&sps, hdr.frame_num);
            }
        }

        // Picture order count (§8.2.1); a field sets only its own.
        let mut poc_msb = 0;
        let mut frame_num_offset = 0;
        let (top, bottom) = match sps.poc_type {
            0 => {
                let (prev_msb, prev_lsb) = if hdr.idr {
                    (0, 0)
                } else {
                    (self.prev_poc_msb, self.prev_poc_lsb)
                };
                let max_lsb = 1i32 << sps.log2_max_poc_lsb;
                let lsb = hdr.poc_lsb as i32;
                poc_msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    prev_msb + max_lsb
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    prev_msb - max_lsb
                } else {
                    prev_msb
                };
                let top = poc_msb + lsb;
                (top, if structure == 3 { top + hdr.delta_poc_bottom } else { top })
            }
            _ => {
                let prev_off = if self.prev_had_mmco5 {
                    0
                } else {
                    self.prev_frame_num_offset
                };
                frame_num_offset = if hdr.idr {
                    0
                } else if self.prev_frame_num > hdr.frame_num {
                    prev_off + max_frame_num as i32
                } else {
                    prev_off
                };
                if sps.poc_type == 1 {
                    let n = sps.offset_for_ref_frame.len() as i32;
                    let mut abs = if n != 0 {
                        frame_num_offset + hdr.frame_num as i32
                    } else {
                        0
                    };
                    if hdr.nal_ref_idc == 0 && abs > 0 {
                        abs -= 1;
                    }
                    let mut expected = 0;
                    if abs > 0 {
                        let cycle = (abs - 1) / n;
                        let in_cycle = (abs - 1) % n;
                        let delta: i32 = sps.offset_for_ref_frame.iter().sum();
                        expected = cycle * delta
                            + sps.offset_for_ref_frame[..=in_cycle as usize].iter().sum::<i32>();
                    }
                    if hdr.nal_ref_idc == 0 {
                        expected += sps.offset_for_non_ref_pic;
                    }
                    let top = expected + hdr.delta_poc[0];
                    let mut bottom = top + sps.offset_for_top_to_bottom_field;
                    if structure == 3 {
                        bottom += hdr.delta_poc[1];
                    }
                    (top, bottom)
                } else {
                    let t = if hdr.idr {
                        0
                    } else if hdr.nal_ref_idc == 0 {
                        2 * (frame_num_offset + hdr.frame_num as i32) - 1
                    } else {
                        2 * (frame_num_offset + hdr.frame_num as i32)
                    };
                    (t, t)
                }
            }
        };

        let (w_mbs, h_mbs) = (sps.width_mbs as usize, sps.height_mbs as usize);
        let fill_y = 1u16 << (sps.bit_depth_luma - 1);
        let fill_c = 1u16 << (sps.bit_depth_chroma - 1);
        let ch_rows = if sps.chroma_format_idc == 2 { 16 } else { 8 };
        // A field picture is half the frame's height.
        let pic_h_mbs = if structure == 3 { h_mbs } else { h_mbs / 2 };
        let accel = self.accel.is_some();
        let make = |h: usize| {
            if accel {
                // The accelerator holds the samples.
                empty_planes()
            } else if sps.chroma_format_idc == 0 {
                [Plane::new(w_mbs * 16, h * 16, fill_y), Plane::new(0, 0, 0), Plane::new(0, 0, 0)]
            } else {
                [
                    Plane::new(w_mbs * 16, h * 16, fill_y),
                    Plane::new(w_mbs * 8, h * ch_rows, fill_c),
                    Plane::new(w_mbs * 8, h * ch_rows, fill_c),
                ]
            }
        };
        let store = match second_store {
            Some(i) => i,
            None => {
                let id = self.next_id;
                self.next_id += 1;
                self.dpb.push(RefPic {
                    id,
                    planes: Arc::new(make(h_mbs)),
                    field_planes: [None, None],
                    mbs: Arc::new(Vec::new()),
                    field_mbs: [None, None],
                    field_poc: [i32::MAX, i32::MAX],
                    poc: i32::MAX,
                    frame_num: hdr.frame_num,
                    frame_num_wrap: hdr.frame_num as i32,
                    short: 0,
                    long: 0,
                    lt_idx: 0,
                });
                self.dpb.len() - 1
            }
        };
        {
            let st = &mut self.dpb[store];
            if structure & 1 != 0 {
                st.field_poc[0] = top;
            }
            if structure & 2 != 0 {
                st.field_poc[1] = bottom;
            }
            st.poc = st.field_poc[0].min(st.field_poc[1]);
        }
        let poc = match structure {
            1 => top,
            2 => bottom,
            _ => top.min(bottom),
        };
        self.cur = Some(Current {
            pic: CurPic {
                planes: make(pic_h_mbs),
                mbs: if accel { Vec::new() } else { vec![MbInfo::default(); w_mbs * pic_h_mbs] },
                w_mbs,
                h_mbs: pic_h_mbs,
                poc,
                field_poc: [top, bottom],
                slices: Vec::new(),
                decoded: 0,
            },
            sps,
            structure,
            mbaff: hdr.mbaff,
            store,
            second_field,
            idr: hdr.idr,
            nal_ref_idc: hdr.nal_ref_idc,
            frame_num: hdr.frame_num,
            long_term_reference: hdr.long_term_reference,
            mmcos: hdr.mmcos.clone(),
            pps_id: hdr.pps_id,
            poc_lsb: hdr.poc_lsb,
            poc_msb,
            frame_num_offset,
        });
        Ok(())
    }

    /// Insert "non-existing" frames for skipped `frame_num`s (§8.2.5.2), so
    /// the sliding window and list initialisation see what the encoder did.
    fn fill_frame_num_gap(&mut self, sps: &Sps, frame_num: u32) {
        let max = 1u32 << sps.log2_max_frame_num;
        let mut f = (self.prev_ref_frame_num + 1) % max;
        let mut guard = 0;
        // Content of a non-existing frame is unspecified; reuse the newest
        // reference so a stream that does reference one degrades gracefully.
        let src = self.dpb.iter().filter(|p| p.is_ref()).max_by_key(|p| p.id).map(|p| p.planes.clone());
        let Some(planes) = src else {
            self.prev_ref_frame_num = frame_num.wrapping_sub(1) % max;
            return;
        };
        while f != frame_num && guard < max {
            self.update_wraps(f, max);
            self.sliding_window(sps, false);
            let id = self.next_id;
            self.next_id += 1;
            let mbs = Arc::new(vec![
                MbInfo::default();
                (sps.width_mbs * sps.height_mbs) as usize
            ]);
            self.dpb.push(RefPic {
                id,
                planes: planes.clone(),
                field_planes: [None, None],
                mbs,
                field_mbs: [None, None],
                field_poc: [0, 0],
                poc: 0,
                frame_num: f,
                frame_num_wrap: 0,
                short: 3,
                long: 0,
                lt_idx: 0,
            });
            self.prev_ref_frame_num = f;
            self.prev_frame_num = f;
            f = (f + 1) % max;
            guard += 1;
        }
    }

    /// Recompute `FrameNumWrap` against the current `frame_num`.
    fn update_wraps(&mut self, frame_num: u32, max_frame_num: u32) {
        for p in &mut self.dpb {
            if p.short != 0 {
                p.frame_num_wrap = if p.frame_num > frame_num {
                    p.frame_num as i32 - max_frame_num as i32
                } else {
                    p.frame_num as i32
                };
            }
        }
    }

    fn build_lists(&mut self, hdr: &SliceHeader) -> Result<[Vec<RefEntry>; 2]> {
        let cur = self.cur.as_ref().unwrap();
        let max_frame_num = 1u32 << cur.sps.log2_max_frame_num;
        let structure = cur.structure;
        let field = structure != 3;
        let cur_par = (structure == 2) as usize;
        let cur_poc = cur.pic.poc;
        let cur_store_id = self.dpb[cur.store].id;
        self.update_wraps(hdr.frame_num, max_frame_num);
        let mut lists: [Vec<RefEntry>; 2] = [Vec::new(), Vec::new()];
        if hdr.slice_type == SliceType::I {
            return Ok(lists);
        }
        // Frames take only stores with both fields marked; fields take any.
        let full = |m: u8| if field { m != 0 } else { m == 3 };
        let mut short: Vec<usize> = (0..self.dpb.len()).filter(|&i| full(self.dpb[i].short)).collect();
        let mut long: Vec<usize> = (0..self.dpb.len()).filter(|&i| full(self.dpb[i].long)).collect();
        long.sort_by_key(|&i| self.dpb[i].lt_idx);
        // An ordered list of frame stores -> entries (§8.2.4.2.5 for fields:
        // alternate parities starting with the current one).
        let expand = |dpb: &mut Vec<RefPic>, order: &[usize], long_term: bool| -> Vec<RefEntry> {
            let mark = |p: &RefPic| if long_term { p.long } else { p.short };
            let mut out = Vec::new();
            if !field {
                for &i in order {
                    let p = &dpb[i];
                    out.push(RefEntry { dpb: i, id: p.id * 4 + 3, poc: p.poc, long_term, structure: 3, planes: p.planes.clone() });
                }
                return out;
            }
            let sel = [cur_par, 1 - cur_par];
            let mut idx = [0usize, 0usize];
            loop {
                let mut any = false;
                for (k, &par) in sel.iter().enumerate() {
                    while idx[k] < order.len() && mark(&dpb[order[idx[k]]]) & (1 << par) == 0 {
                        idx[k] += 1;
                    }
                    if idx[k] < order.len() {
                        let i = order[idx[k]];
                        idx[k] += 1;
                        any = true;
                        let planes = dpb[i].field(par);
                        let p = &dpb[i];
                        out.push(RefEntry {
                            dpb: i,
                            id: p.id * 4 + 1 + par as u32,
                            poc: p.field_poc[par],
                            long_term,
                            structure: 1 + par as u8,
                            planes,
                        });
                    }
                }
                if !any {
                    break;
                }
            }
            out
        };
        if hdr.slice_type == SliceType::P {
            short.sort_by_key(|&i| std::cmp::Reverse(self.dpb[i].frame_num_wrap));
            let mut l = expand(&mut self.dpb, &short, false);
            l.extend(expand(&mut self.dpb, &long, true));
            lists[0] = l;
        } else {
            // B: by the frame (or pair) POC around the current POC; equal
            // POCs go before (ffmpeg add_sorted).
            let mut before: Vec<usize> = short.iter().copied().filter(|&i| self.dpb[i].poc <= cur_poc).collect();
            let mut after: Vec<usize> = short.iter().copied().filter(|&i| !before.contains(&i)).collect();
            before.sort_by_key(|&i| std::cmp::Reverse(self.dpb[i].poc));
            after.sort_by_key(|&i| self.dpb[i].poc);
            let o0: Vec<usize> = before.iter().chain(after.iter()).copied().collect();
            let o1: Vec<usize> = after.iter().chain(before.iter()).copied().collect();
            let mut l0 = expand(&mut self.dpb, &o0, false);
            let mut l1 = expand(&mut self.dpb, &o1, false);
            let lt = expand(&mut self.dpb, &long, true);
            l0.extend(lt.iter().cloned());
            l1.extend(lt);
            // Identical lists (compared by frame store, as ffmpeg does):
            // swap the first two of list 1.
            if l1.len() > 1 && l0.len() == l1.len() && l0.iter().zip(&l1).all(|(a, b)| a.dpb == b.dpb) {
                l1.swap(0, 1);
            }
            lists = [l0, l1];
        }
        let _ = cur_store_id;
        let (curr_pic_num, max_pic_num) = if field {
            (2 * hdr.frame_num as i32 + 1, 2 * max_frame_num as i32)
        } else {
            (hdr.frame_num as i32, max_frame_num as i32)
        };
        for l in 0..2 {
            let n = hdr.num_ref_idx[l] as usize;
            lists[l].truncate(n);
            if hdr.list_mods[l].is_empty() {
                continue;
            }
            let mut pred = curr_pic_num;
            let mut idx = 0usize;
            for m in &hdr.list_mods[l] {
                // (store index, parity or None for a frame, long-term)
                let target: Option<(usize, Option<usize>, bool)> = if m.idc < 2 {
                    let abs = m.value as i32 + 1;
                    let mut no_wrap = if m.idc == 0 { pred - abs } else { pred + abs };
                    if no_wrap < 0 {
                        no_wrap += max_pic_num;
                    } else if no_wrap >= max_pic_num {
                        no_wrap -= max_pic_num;
                    }
                    pred = no_wrap;
                    let pic_num = if no_wrap > curr_pic_num { no_wrap - max_pic_num } else { no_wrap };
                    if field {
                        let par = if pic_num & 1 == 1 { cur_par } else { 1 - cur_par };
                        let fnw = pic_num >> 1;
                        self.dpb
                            .iter()
                            .position(|p| p.short & (1 << par) != 0 && p.frame_num_wrap == fnw)
                            .map(|i| (i, Some(par), false))
                    } else {
                        self.dpb.iter().position(|p| p.short == 3 && p.frame_num_wrap == pic_num).map(|i| (i, None, false))
                    }
                } else if field {
                    let par = if m.value & 1 == 1 { cur_par } else { 1 - cur_par };
                    let lt = m.value >> 1;
                    self.dpb
                        .iter()
                        .position(|p| p.long & (1 << par) != 0 && p.lt_idx == lt)
                        .map(|i| (i, Some(par), true))
                } else {
                    self.dpb.iter().position(|p| p.long == 3 && p.lt_idx == m.value).map(|i| (i, None, true))
                };
                let Some((i, par, long_term)) = target else {
                    return Err("ref_pic_list_modification names a missing picture".into());
                };
                let e = match par {
                    None => {
                        let p = &self.dpb[i];
                        RefEntry { dpb: i, id: p.id * 4 + 3, poc: p.poc, long_term, structure: 3, planes: p.planes.clone() }
                    }
                    Some(par) => {
                        let planes = self.dpb[i].field(par);
                        let p = &self.dpb[i];
                        RefEntry { dpb: i, id: p.id * 4 + 1 + par as u32, poc: p.field_poc[par], long_term, structure: 1 + par as u8, planes }
                    }
                };
                if idx > lists[l].len() {
                    idx = lists[l].len();
                }
                let eid = e.id;
                lists[l].insert(idx, e);
                idx += 1;
                let mut k = idx;
                while k < lists[l].len() {
                    if lists[l][k].id == eid {
                        lists[l].remove(k);
                    } else {
                        k += 1;
                    }
                }
            }
            lists[l].truncate(n);
        }
        Ok(lists)
    }

    /// §8.2.5.3: when the DPB is full, unmark the short-term frame (or pair,
    /// or field) with the smallest `FrameNumWrap`. `exclude` is the current
    /// frame store (its first field does not count).
    fn sliding_window(&mut self, sps: &Sps, _exclude: bool) {
        let max = sps.max_num_ref_frames.max(1) as usize;
        let n_ref = self.dpb.iter().filter(|p| p.is_ref()).count();
        let n_short = self.dpb.iter().filter(|p| p.short != 0).count();
        if n_ref >= max && n_short > 0 {
            if let Some(i) = (0..self.dpb.len()).filter(|&i| self.dpb[i].short != 0).min_by_key(|&i| self.dpb[i].frame_num_wrap) {
                self.dpb[i].short = 0;
            }
        }
    }

    /// Reference marking (§8.2.5) for the finished picture `cur`; returns
    /// whether it carried memory_management_control_operation 5.
    fn mark(&mut self, cur: &Current) -> bool {
        let max_frame_num = 1u32 << cur.sps.log2_max_frame_num;
        let structure = cur.structure;
        let field = structure != 3;
        let cur_par = (structure == 2) as usize;
        let cur_bits: u8 = if field { 1 << cur_par } else { 3 };
        let cur_id = self.dpb[cur.store].id;
        self.update_wraps(cur.frame_num, max_frame_num);
        let mut long_term = false;
        let mut lt_idx = 0;
        let mut had_mmco5 = false;
        // Unmark the given fields of store `i`.
        fn unmark(p: &mut RefPic, bits: u8, long: bool) {
            if long {
                p.long &= !bits;
            } else {
                p.short &= !bits;
            }
        }
        // (store, field bits) of a short-term picNumX / long-term pic num.
        let (curr_pic_num, _) = if field { (2 * cur.frame_num as i32 + 1, 0) } else { (cur.frame_num as i32, 0) };
        let find_short = |dpb: &[RefPic], pic_num: i32| -> Option<(usize, u8)> {
            if field {
                let par = if pic_num & 1 == 1 { cur_par } else { 1 - cur_par };
                let fnw = pic_num >> 1;
                dpb.iter().position(|p| p.short & (1 << par) != 0 && p.frame_num_wrap == fnw).map(|i| (i, 1u8 << par))
            } else {
                dpb.iter().position(|p| p.short == 3 && p.frame_num_wrap == pic_num).map(|i| (i, 3))
            }
        };
        let find_long = |dpb: &[RefPic], num: u32| -> Option<(usize, u8)> {
            if field {
                let par = if num & 1 == 1 { cur_par } else { 1 - cur_par };
                dpb.iter().position(|p| p.long & (1 << par) != 0 && p.lt_idx == num >> 1).map(|i| (i, 1u8 << par))
            } else {
                dpb.iter().position(|p| p.long == 3 && p.lt_idx == num).map(|i| (i, 3))
            }
        };
        if cur.idr && !cur.second_field {
            if cur.long_term_reference {
                long_term = true;
                self.max_lt_idx = Some(0);
            } else {
                self.max_lt_idx = None;
            }
        } else if cur.second_field && (self.dpb[cur.store].short != 0 || self.dpb[cur.store].long != 0) && cur.mmcos.is_none() {
            // The first field already holds the frame's marking: the second
            // takes the same (no sliding window, §8.2.5.3).
            long_term = self.dpb[cur.store].long != 0;
            lt_idx = self.dpb[cur.store].lt_idx;
        } else if let Some(ops) = &cur.mmcos {
            for m in ops {
                let pic_num_x = curr_pic_num - (m.diff_pic_nums_minus1 as i32 + 1);
                match m.op {
                    1 => {
                        if let Some((i, b)) = find_short(&self.dpb, pic_num_x) {
                            unmark(&mut self.dpb[i], b, false);
                        }
                    }
                    2 => {
                        if let Some((i, b)) = find_long(&self.dpb, m.long_term_pic_num) {
                            unmark(&mut self.dpb[i], b, true);
                        }
                    }
                    3 => {
                        let idx = m.long_term_frame_idx;
                        if let Some((i, b)) = find_short(&self.dpb, pic_num_x) {
                            let id = self.dpb[i].id;
                            // Free the index unless it holds this field's sibling.
                            for p in self.dpb.iter_mut() {
                                if p.long != 0 && p.lt_idx == idx && (p.id != id || !field) {
                                    p.long = 0;
                                }
                            }
                            let p = &mut self.dpb[i];
                            p.short &= !b;
                            p.long |= b;
                            p.lt_idx = idx;
                        }
                    }
                    4 => {
                        let max = m.max_long_term_frame_idx_plus1;
                        self.max_lt_idx = max.checked_sub(1);
                        for p in self.dpb.iter_mut() {
                            if p.long != 0 && p.lt_idx >= max {
                                p.long = 0;
                            }
                        }
                    }
                    5 => {
                        for p in self.dpb.iter_mut() {
                            p.short = 0;
                            p.long = 0;
                        }
                        self.max_lt_idx = None;
                        had_mmco5 = true;
                    }
                    6 => {
                        let idx = m.long_term_frame_idx;
                        for p in self.dpb.iter_mut() {
                            if p.long != 0 && p.lt_idx == idx && p.id != cur_id {
                                p.long = 0;
                            }
                        }
                        long_term = true;
                        lt_idx = idx;
                    }
                    _ => {}
                }
            }
            if !long_term {
                // A stream that overfills the DPB with adaptive marking
                // is broken; slide rather than grow without bound.
                let max = cur.sps.max_num_ref_frames.max(1) as usize;
                let others = self.dpb.iter().filter(|p| p.is_ref() && p.id != cur_id).count();
                if others >= max {
                    self.sliding_window(&cur.sps, true);
                }
            }
        } else {
            self.sliding_window(&cur.sps, true);
        }
        let st = &mut self.dpb[cur.store];
        if long_term {
            st.long |= cur_bits;
            st.short &= !cur_bits;
            st.lt_idx = lt_idx;
        } else {
            st.short |= cur_bits;
        }
        had_mmco5
    }

    /// Finish the picture in progress, if any: deblock, mark references, and
    /// queue a completed frame for output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        let sps = cur.sps.clone();
        let accel = self.accel.is_some();
        if accel {
            if std::mem::take(&mut self.accel_started) {
                if let Err(e) = self.accel.as_mut().unwrap().end_picture() {
                    self.accel_error.get_or_insert(e);
                }
            }
        } else if cur.mbaff {
            deblock::deblock_mbaff(&mut cur.pic, sps.bit_depth_luma, sps.bit_depth_chroma, sps.chroma_format_idc);
        } else {
            deblock::deblock_picture(
                &mut cur.pic,
                sps.bit_depth_luma,
                sps.bit_depth_chroma,
                sps.chroma_format_idc,
                cur.structure != 3,
            );
        }
        // Move the samples and motion into the frame store.
        let structure = cur.structure;
        let planes = std::mem::replace(&mut cur.pic.planes, empty_planes());
        let mbs = Arc::new(std::mem::take(&mut cur.pic.mbs));
        if !accel {
            let st = &mut self.dpb[cur.store];
            st.field_planes = [None, None];
            if structure == 3 {
                st.planes = Arc::new(planes);
                st.mbs = mbs;
            } else {
                let par = (structure == 2) as usize;
                let dst = Arc::make_mut(&mut st.planes);
                for c in 0..3 {
                    let (fp, d) = (&planes[c], &mut dst[c]);
                    for y in 0..fp.height {
                        let o = (2 * y + par) * d.width;
                        d.data[o..o + d.width].copy_from_slice(&fp.data[y * fp.width..(y + 1) * fp.width]);
                    }
                }
                st.field_mbs[par] = Some(mbs);
            }
        }
        let mut had_mmco5 = false;
        if cur.nal_ref_idc != 0 {
            had_mmco5 = self.mark(&cur);
            let st = &mut self.dpb[cur.store];
            if had_mmco5 {
                // The picture now counts as frame_num 0 with its POC rebased.
                let temp = match structure {
                    1 => st.field_poc[0],
                    2 => st.field_poc[1],
                    _ => st.field_poc[0].min(st.field_poc[1]),
                };
                if structure & 1 != 0 {
                    st.field_poc[0] -= temp;
                }
                if structure & 2 != 0 {
                    st.field_poc[1] -= temp;
                }
                st.poc = st.field_poc[0].min(st.field_poc[1]);
                st.frame_num = 0;
            }
            st.frame_num_wrap = st.frame_num as i32;
            let frame_num = st.frame_num;
            self.prev_ref_frame_num = frame_num;
            if had_mmco5 {
                self.prev_poc_msb = 0;
                self.prev_poc_lsb = if structure == 2 { 0 } else { st.field_poc[0] };
            } else {
                self.prev_poc_msb = cur.poc_msb;
                self.prev_poc_lsb = cur.poc_lsb as i32;
            }
        }
        self.prev_had_mmco5 = had_mmco5;
        self.prev_frame_num = if had_mmco5 { 0 } else { cur.frame_num };
        self.prev_frame_num_offset = if had_mmco5 { 0 } else { cur.frame_num_offset };
        let complete = structure == 3 || cur.second_field;
        let st_id = self.dpb[cur.store].id;
        if !complete {
            self.pending = Some((st_id, structure, cur.frame_num, cur.nal_ref_idc != 0));
        }
        if complete {
            let st = &self.dpb[cur.store];
            let chroma = match sps.chroma_format_idc {
                0 => ChromaFormat::Monochrome,
                2 => ChromaFormat::Yuv422,
                _ => ChromaFormat::Yuv420,
            };
            let crop = sps.crop.map(|v| v as usize);
            let color = ColorInfo::from_vui(
                sps.matrix_coefficients,
                sps.full_range,
                sps.height_mbs * 16 - sps.crop[2] - sps.crop[3],
            );
            let frame = match self.accel.as_mut() {
                Some(a) => match a.frame(st.id, &sps) {
                    Ok(f) => f,
                    Err(e) => {
                        self.accel_error.get_or_insert(e);
                        self.prune();
                        return;
                    }
                },
                None => DecodedFrame::from_planes(&st.planes, chroma, sps.bit_depth_luma as u8, crop, color),
            };
            self.done.push(H264Picture {
                frame,
                poc: st.poc,
                idr: cur.idr || had_mmco5,
            });
        }
        self.prune();
    }
}
