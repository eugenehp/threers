//! A from-scratch HEVC / H.265 decoder.
//!
//! Main, Main 10 and Main Still Picture streams (and the 4:2:0 subset of
//! Main 12): every coding tool those profiles allow — AMP, tiles, wavefront
//! parallel processing, dependent slice segments, PCM, transform skip,
//! transquant bypass, sign data hiding, scaling lists, weighted
//! prediction, long-term references, TMVP, deblocking and SAO — and Main
//! 4:2:2 10 (and its 8-bit subset): 4:2:2 chroma without the range
//! extension coding tools. Not supported, and refused with an error: 4:4:4,
//! the range extension tools, and anything in a layer above the base
//! (multi-layer streams decode their base layer — Apple's alpha layer is
//! skipped).
//!
//! Known gap: with 16×16 CTBs ffmpeg's interleaved per-CTB deblocking/SAO
//! schedule lets SAO read chroma a horizontal deblock has not reached yet,
//! so its output differs from the spec order used here by a few samples
//! (~1 in 15 000); 32×32 and 64×64 CTBs match it exactly.
//!
//! Feed it NAL units with [`HevcDecoder::decode_nal`]; finished pictures
//! come out of [`HevcDecoder::take_pictures`].

#[cfg(feature = "decode")]
pub mod cabac_init;
#[cfg(feature = "decode")]
pub mod ctu;
#[cfg(feature = "decode")]
pub mod filters;
#[cfg(feature = "decode")]
pub mod intra;
#[cfg(feature = "decode")]
pub mod inter;
#[cfg(feature = "decode")]
pub mod mv;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "decode"), allow(dead_code))]
pub mod params;
#[cfg(feature = "decode")]
pub mod pic;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "decode"), allow(dead_code))]
pub mod tables;
#[cfg(feature = "decode")]
pub mod transform;

#[cfg(feature = "decode")]
use std::sync::Arc;

#[cfg(feature = "decode")]
use self::ctu::{Carry, SliceDec};
#[cfg(feature = "decode")]
use self::params::{Pps, SliceHeader, SliceType, Sps};
#[cfg(feature = "decode")]
use self::pic::{Layout, MvField, PicState, SliceInfo};
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
pub struct HevcPicture {
    /// Decode-order number, which [`HevcDecoder::take_output_order`] refers to.
    pub id: u64,
    /// The picture, cropped to its conformance window.
    pub frame: DecodedFrame,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Starts a new coded video sequence (an IRAP picture after which POC
    /// ordering restarts).
    pub new_sequence: bool,
}

#[cfg(feature = "decode")]
/// The motion a picture leaves for later ones to use as collocated.
pub struct Motion {
    /// Motion of each 4×4 block, row-major.
    pub mv: Vec<MvField>,
    /// Width of the picture in 4×4 blocks (the row stride of `mv`).
    pub w4: usize,
    /// Index into `slices` of the slice each CTB belongs to.
    pub ctb_slice: Vec<u16>,
    /// What collocated prediction needs from each slice of the picture.
    pub slices: Vec<SliceInfo>,
    /// log2 of the CTB size.
    pub log2_ctb: usize,
    /// Width of the picture in CTBs.
    pub ctb_w: usize,
}

/// A picture in the decoded picture buffer.
#[cfg(feature = "decode")]
pub struct RefPic {
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Marked "used for long-term reference".
    pub long_term: bool,
    /// The reconstructed samples, before cropping.
    pub planes: Arc<[Plane; 3]>,
    /// The motion it leaves for temporal motion vector prediction.
    pub motion: Arc<Motion>,
}

/// One entry of a reference picture list.
#[cfg(feature = "decode")]
#[derive(Clone, Copy, Debug)]
pub struct RefEntry {
    /// Index of the picture in the decoded picture buffer.
    pub dpb: usize,
    /// Its `PicOrderCntVal`.
    pub poc: i32,
    /// Whether it is a long-term reference.
    pub long_term: bool,
}

#[cfg(feature = "decode")]
struct Current {
    pic: PicState,
    sps: Sps,
    pps: Pps,
    carry: Carry,
    output: bool,
    new_sequence: bool,
    st_before: Vec<usize>,
    st_after: Vec<usize>,
    lt_curr: Vec<usize>,
}

/// A hardware back end for [`HevcDecoder`]: the decoder parses, derives
/// POCs and reference picture sets, builds the lists and orders output; the
/// accelerator reconstructs. See
/// [`H264Accel`](crate::decode::h264::H264Accel) for the shape.
#[cfg(feature = "decode")]
pub trait HevcAccel {
    /// Decode one slice segment of the current picture.
    fn slice(&mut self, s: &AccelSlice<'_>) -> Result<()>;
    /// The current picture has had all its slice segments.
    fn end_picture(&mut self) -> Result<()>;
    /// The samples of the picture with POC `poc`, cropped as `sps` says.
    fn frame(&mut self, poc: i32, sps: &Sps) -> Result<DecodedFrame>;
}

/// Everything an [`HevcAccel`] is told about one slice segment.
#[cfg(feature = "decode")]
pub struct AccelSlice<'a> {
    /// The NAL unit as coded: 2-byte header included, emulation prevention in.
    pub nal: &'a [u8],
    /// `nal_unit_type`.
    pub nal_type: u32,
    /// The active sequence parameter set.
    pub sps: &'a Sps,
    /// The active picture parameter set.
    pub pps: &'a Pps,
    /// The parsed slice segment header.
    pub sh: &'a SliceHeader,
    /// The first segment of a new picture.
    pub first: bool,
    /// `PicOrderCntVal` of the picture.
    pub poc: i32,
    /// The reference pictures; the current one is not among them.
    pub dpb: &'a [RefPic],
    /// `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter`, `RefPicSetLtCurr`
    /// as indices into `dpb`.
    pub st_before: &'a [usize],
    /// See `st_before`.
    pub st_after: &'a [usize],
    /// See `st_before`.
    pub lt_curr: &'a [usize],
    /// `RefPicList0` and `RefPicList1`.
    pub refs: &'a [Vec<RefEntry>; 2],
}

/// An HEVC decoder fed one NAL unit at a time ([`decode_nal`](Self::decode_nal)).
///
/// Most callers want [`NativeDecoder`](crate::decode::NativeDecoder), which
/// unpacks MP4 samples and orders the output; this is the layer below it.
#[cfg(feature = "decode")]
pub struct HevcDecoder {
    spss: Vec<Option<Sps>>,
    ppss: Vec<Option<Pps>>,
    dpb: Vec<RefPic>,
    cur: Option<Current>,
    /// The last independent slice segment header of the current picture.
    last_header: Option<SliceHeader>,
    prev_tid0_poc: i32,
    /// The next IRAP picture starts with `NoRaslOutputFlag` set.
    at_start: bool,
    /// RASL pictures of the current IRAP cannot be decoded.
    skip_rasl: bool,
    /// The picture now arriving is being skipped (a RASL picture).
    skipping: bool,
    layout: Option<(usize, usize, Arc<Layout>)>,
    done: Vec<HevcPicture>,
    /// The DPB output process (§C.5.2): pictures decoded but not yet output,
    /// as `(id, POC, PicLatencyCount)`.
    pending: Vec<(u64, i32, u32)>,
    /// Ids of pictures in the order the output process emits them. Pictures
    /// it discards (`no_output_of_prior_pics`) never appear.
    output: Vec<u64>,
    next_id: u64,
    /// Whether any picture has been decoded — the first needs no flush.
    decoded_any: bool,
    accel: Option<Box<dyn HevcAccel>>,
    accel_error: Option<String>,
    accel_started: bool,
}

#[cfg(feature = "decode")]
impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "decode")]
impl HevcDecoder {
    /// A decoder with no parameter sets yet.
    pub fn new() -> Self {
        HevcDecoder {
            spss: vec![None; 16],
            ppss: vec![None; 64],
            dpb: Vec::new(),
            cur: None,
            last_header: None,
            prev_tid0_poc: 0,
            at_start: true,
            skip_rasl: false,
            skipping: false,
            layout: None,
            done: Vec::new(),
            pending: Vec::new(),
            output: Vec::new(),
            next_id: 0,
            decoded_any: false,
            accel: None,
            accel_error: None,
            accel_started: false,
        }
    }

    /// Hand reconstruction to `accel` (see [`HevcAccel`]).
    pub fn set_accel(&mut self, accel: Box<dyn HevcAccel>) {
        self.accel = Some(accel);
    }

    /// The first error the accelerator reported from
    /// [`finish_picture`](Self::finish_picture), which cannot return one.
    pub fn take_accel_error(&mut self) -> Option<String> {
        self.accel_error.take()
    }

    /// Decode one NAL unit (without start code or length prefix).
    pub fn decode_nal(&mut self, nal: &[u8]) -> Result<()> {
        if nal.len() < 2 {
            return Ok(());
        }
        let nal_type = ((nal[0] >> 1) & 63) as u32;
        let layer = (((nal[0] & 1) << 5) | (nal[1] >> 3)) as u32;
        let tid = (nal[1] & 7).saturating_sub(1) as u32;
        if layer > 0 {
            return Ok(());
        }
        match nal_type {
            0..=9 | 16..=21 => {
                let rbsp = unescape(&nal[2..]);
                self.decode_slice(nal, &rbsp, nal_type, tid)
            }
            33 => {
                let sps = Sps::parse(&unescape(&nal[2..]))?;
                let id = sps.id as usize;
                self.spss[id] = Some(sps);
                // A re-sent set may change under the same id.
                self.layout = None;
                Ok(())
            }
            34 => {
                let pps = Pps::parse(&unescape(&nal[2..]))?;
                let id = pps.id as usize;
                self.ppss[id] = Some(pps);
                // Encoders re-send a PPS under the same id with a new tile
                // grid; the cached layout is only good for the old one.
                self.layout = None;
                Ok(())
            }
            35 => {
                self.finish_picture();
                Ok(())
            }
            36 | 37 => {
                self.finish_picture();
                self.at_start = true;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Pictures finished so far, in decode order.
    /// Ids of pictures (see [`HevcPicture::id`]) in display order, as the
    /// DPB output process of §C.5.2 emits them so far. Pictures still in the
    /// DPB come out with [`flush_output`](Self::flush_output); pictures the
    /// stream asks to be discarded never do.
    pub fn take_output_order(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.output)
    }

    /// End of stream: output everything still waiting, in POC order.
    pub fn flush_output(&mut self) {
        self.finish_picture();
        while self.bump() {}
    }

    /// Output the waiting picture with the smallest POC ("bumping").
    fn bump(&mut self) -> bool {
        let Some(i) = (0..self.pending.len()).min_by_key(|&i| self.pending[i].1) else {
            return false;
        };
        let (id, _, _) = self.pending.remove(i);
        self.output.push(id);
        true
    }

    /// Bump while the reorder or latency limits are exceeded (and, before a
    /// picture is decoded, while the DPB is full).
    fn bump_while_over(&mut self, sps: &Sps, dpb_full: bool) {
        loop {
            let reorder = self.pending.len() > sps.max_num_reorder as usize;
            let latency = sps.max_latency_increase_plus1 != 0 && {
                let max = sps.max_num_reorder + sps.max_latency_increase_plus1 - 1;
                self.pending.iter().any(|e| e.2 >= max)
            };
            let full = dpb_full && {
                let refs = self.dpb.iter().filter(|p| !self.pending.iter().any(|e| e.1 == p.poc)).count();
                refs + self.pending.len() >= sps.max_dec_pic_buffering as usize
            };
            if !(reorder || latency || full) || !self.bump() {
                return;
            }
        }
    }

    /// The pictures finished since the last call, in decode order.
    pub fn take_pictures(&mut self) -> Vec<HevcPicture> {
        std::mem::take(&mut self.done)
    }

    fn decode_slice(&mut self, nal: &[u8], rbsp: &[u8], nal_type: u32, tid: u32) -> Result<()> {
        let first = rbsp.first().is_some_and(|b| b & 0x80 != 0);
        if first {
            self.finish_picture();
            self.skipping = (nal_type == 8 || nal_type == 9) && self.skip_rasl;
            self.last_header = None;
        }
        if self.skipping {
            return Ok(());
        }
        if !first && self.cur.is_none() {
            // The picture's first slice was lost; nothing to attach to.
            return Ok(());
        }
        let prev = self.last_header.as_ref();
        let sh = SliceHeader::parse(rbsp, nal_type, &self.spss, &self.ppss, prev)?;
        if first {
            self.start_picture(&sh, nal_type, tid)?;
        }
        if !sh.dependent {
            self.last_header = Some(sh.clone());
        }
        let refs = self.build_lists(&sh)?;
        if let Some(accel) = self.accel.as_mut() {
            let cur = self.cur.as_ref().unwrap();
            let slice = AccelSlice {
                nal,
                nal_type,
                sps: &cur.sps,
                pps: &cur.pps,
                sh: &sh,
                first: !self.accel_started,
                poc: cur.pic.poc,
                dpb: &self.dpb,
                st_before: &cur.st_before,
                st_after: &cur.st_after,
                lt_curr: &cur.lt_curr,
                refs: &refs,
            };
            self.accel_started = true;
            return accel.slice(&slice);
        }
        let cur = self.cur.as_mut().unwrap();
        // SliceAddrRs: a dependent segment belongs to the slice before it.
        let slice_addr = if sh.dependent {
            cur.pic.slices.last().map_or(0, |s| s.addr_rs)
        } else {
            sh.segment_address
        };
        let slice_idx = cur.pic.slices.len() as u16;
        cur.pic.slices.push(SliceInfo {
            addr_rs: slice_addr,
            deblocking_disabled: sh.deblocking_disabled,
            beta_offset: sh.beta_offset,
            tc_offset: sh.tc_offset,
            lf_across_slices: sh.loop_filter_across_slices,
            cb_qp_offset: sh.cb_qp_offset,
            cr_qp_offset: sh.cr_qp_offset,
            refs: [
                refs[0].iter().map(|e| (e.poc, e.long_term)).collect(),
                refs[1].iter().map(|e| (e.poc, e.long_term)).collect(),
            ],
        });
        let mut dec = SliceDec::new(
            &cur.sps,
            &cur.pps,
            &sh,
            &mut cur.pic,
            &self.dpb,
            &refs,
            &mut cur.carry,
            slice_idx,
            slice_addr,
        );
        dec.decode(rbsp)
    }

    fn start_picture(&mut self, sh: &SliceHeader, nal_type: u32, tid: u32) -> Result<()> {
        let pps = self.ppss[sh.pps_id as usize].clone().ok_or("missing PPS")?;
        let sps = self.spss[pps.sps_id as usize].clone().ok_or("missing SPS")?;
        let layout = match &self.layout {
            Some((s, p, l))
                if *s == pps.sps_id as usize
                    && *p == pps.id as usize
                    && l.ctb_w == sps.width_ctbs() as usize
                    && l.ctb_h == sps.height_ctbs() as usize =>
            {
                l.clone()
            }
            _ => {
                let l = Arc::new(Layout::new(&sps, &pps));
                self.layout = Some((pps.sps_id as usize, pps.id as usize, l.clone()));
                l
            }
        };
        if let Some(p) = self.dpb.first() {
            if p.planes[0].width != sps.width as usize || p.planes[0].height != sps.height as usize {
                self.dpb.clear();
            }
        }
        let irap = (16..=23).contains(&nal_type);
        let idr = nal_type == 19 || nal_type == 20;
        let no_rasl_output = irap && (idr || (16..=18).contains(&nal_type) || self.at_start);
        if irap {
            self.skip_rasl = no_rasl_output && !idr;
            self.at_start = false;
        }
        // POC (§8.3.1).
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let lsb = sh.poc_lsb as i32;
        let msb = if irap && no_rasl_output {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        let sub_layer_non_ref = nal_type < 16 && nal_type.is_multiple_of(2);
        if tid == 0 && !(6..=9).contains(&nal_type) && !sub_layer_non_ref {
            self.prev_tid0_poc = poc;
        }

        // Reference picture set (§8.3.2).
        let mut st_before = Vec::new();
        let mut st_after = Vec::new();
        let mut lt_curr = Vec::new();
        if idr || (irap && no_rasl_output) {
            self.dpb.clear();
        } else {
            let mut keep = vec![false; self.dpb.len()];
            let mut lt_mark = vec![false; self.dpb.len()];
            let mut missing: Vec<(i32, bool, u8)> = Vec::new();
            for &(v, msb_present, used) in &sh.lt {
                let full = if msb_present { v + poc - (poc & (max_lsb - 1)) } else { v };
                let found = self.dpb.iter().position(|p| {
                    if msb_present {
                        p.poc == full
                    } else {
                        p.poc & (max_lsb - 1) == v
                    }
                });
                match found {
                    Some(i) => {
                        keep[i] = true;
                        lt_mark[i] = true;
                        if used {
                            lt_curr.push(i);
                        }
                    }
                    None if used => missing.push((full, true, 2)),
                    None => {}
                }
            }
            for (list, entries) in [(0u8, &sh.st_rps.neg), (1u8, &sh.st_rps.pos)] {
                for &(d, used) in entries {
                    let target = poc + d;
                    let found = self
                        .dpb
                        .iter()
                        .enumerate()
                        .position(|(i, p)| !lt_mark[i] && !p.long_term && p.poc == target);
                    match found {
                        Some(i) => {
                            keep[i] = true;
                            if used {
                                if list == 0 {
                                    st_before.push(i);
                                } else {
                                    st_after.push(i);
                                }
                            }
                        }
                        None if used => missing.push((target, false, list)),
                        None => {}
                    }
                }
            }
            // Drop everything the set does not mention, renumbering the
            // indices collected above.
            let mut remap = vec![usize::MAX; self.dpb.len()];
            let mut kept = Vec::with_capacity(self.dpb.len());
            for (i, p) in std::mem::take(&mut self.dpb).into_iter().enumerate() {
                if keep[i] {
                    remap[i] = kept.len();
                    let mut p = p;
                    if lt_mark[i] {
                        p.long_term = true;
                    }
                    kept.push(p);
                }
            }
            self.dpb = kept;
            for v in [&mut st_before, &mut st_after, &mut lt_curr] {
                for i in v.iter_mut() {
                    *i = remap[*i];
                }
            }
            // References the stream needs and does not have (lost, or a
            // broken stream): stand in a mid-grey picture so decoding goes on.
            for (target, long_term, list) in missing {
                let i = self.dpb.len();
                self.dpb.push(Self::placeholder(&sps, &layout, target, long_term));
                match list {
                    0 => st_before.push(i),
                    1 => st_after.push(i),
                    _ => lt_curr.push(i),
                }
            }
        }

        // Output of prior pictures (§C.5.2.2).
        if irap && no_rasl_output && self.decoded_any {
            // A CRA here always discards; otherwise the stream says.
            if nal_type == 21 || sh.no_output_of_prior_pics {
                self.pending.clear();
            } else {
                while self.bump() {}
            }
        } else {
            self.bump_while_over(&sps, true);
        }
        self.decoded_any = true;

        let mut pic = if self.accel.is_some() { PicState::sized(&sps, layout) } else { PicState::new(&sps, layout) };
        pic.poc = poc;
        self.cur = Some(Current {
            pic,
            sps,
            pps,
            carry: Carry::default(),
            output: sh.pic_output && !((8..=9).contains(&nal_type) && self.skip_rasl),
            new_sequence: irap && no_rasl_output,
            st_before,
            st_after,
            lt_curr,
        });
        Ok(())
    }

    fn placeholder(sps: &Sps, layout: &Layout, poc: i32, long_term: bool) -> RefPic {
        let (w, h) = (sps.width as usize, sps.height as usize);
        let planes = [
            Plane::new(w, h, 1 << (sps.bit_depth_y - 1)),
            Plane::new(w.div_ceil(2), h.div_ceil(2), 1 << (sps.bit_depth_c - 1)),
            Plane::new(w.div_ceil(2), h.div_ceil(2), 1 << (sps.bit_depth_c - 1)),
        ];
        RefPic {
            poc,
            long_term,
            planes: Arc::new(planes),
            motion: Arc::new(Motion {
                mv: vec![MvField::INTRA; layout.w4 * layout.h4],
                w4: layout.w4,
                ctb_slice: vec![u16::MAX; layout.ctb_w * layout.ctb_h],
                slices: Vec::new(),
                log2_ctb: layout.log2_ctb as usize,
                ctb_w: layout.ctb_w,
            }),
        }
    }

    fn build_lists(&self, sh: &SliceHeader) -> Result<[Vec<RefEntry>; 2]> {
        let mut lists: [Vec<RefEntry>; 2] = [Vec::new(), Vec::new()];
        if sh.slice_type == SliceType::I {
            return Ok(lists);
        }
        let cur = self.cur.as_ref().unwrap();
        let total = cur.st_before.len() + cur.st_after.len() + cur.lt_curr.len();
        if total == 0 {
            return Err("inter slice with no reference pictures".into());
        }
        let entry = |i: usize| RefEntry {
            dpb: i,
            poc: self.dpb[i].poc,
            long_term: self.dpb[i].long_term,
        };
        for l in 0..(1 + (sh.slice_type == SliceType::B) as usize) {
            let n = sh.num_ref_idx[l] as usize;
            let order: Vec<usize> = if l == 0 {
                cur.st_before.iter().chain(&cur.st_after).chain(&cur.lt_curr).copied().collect()
            } else {
                cur.st_after.iter().chain(&cur.st_before).chain(&cur.lt_curr).copied().collect()
            };
            let len = n.max(total);
            let temp: Vec<usize> = (0..len).map(|k| order[k % total]).collect();
            lists[l] = (0..n)
                .map(|i| {
                    let k = match &sh.list_entry[l] {
                        Some(e) => e[i] as usize,
                        None => i,
                    };
                    temp.get(k).copied().map(entry).ok_or("list_entry out of range")
                })
                .collect::<std::result::Result<_, _>>()?;
        }
        Ok(lists)
    }

    /// Finish the picture in progress, if any: loop filters, then keep it as
    /// a reference and queue it for output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        if let Some(accel) = self.accel.as_mut() {
            if std::mem::take(&mut self.accel_started) {
                if let Err(e) = accel.end_picture() {
                    self.accel_error.get_or_insert(e);
                }
            }
        } else {
            filters::deblock(&mut cur.pic, &cur.sps, &cur.pps);
            if cur.sps.sao {
                filters::sao(&mut cur.pic, &cur.sps, &cur.pps);
            }
        }
        let sps = &cur.sps;
        let pic = cur.pic;
        if cur.output {
            let color = ColorInfo::from_vui(
                sps.matrix_coefficients,
                sps.full_range,
                sps.height - sps.crop[2] - sps.crop[3],
            );
            let crop = sps.crop.map(|v| v as usize);
            let fmt = if sps.chroma_format_idc == 2 { ChromaFormat::Yuv422 } else { ChromaFormat::Yuv420 };
            let frame = match self.accel.as_mut() {
                Some(a) => match a.frame(pic.poc, sps) {
                    Ok(f) => f,
                    Err(e) => {
                        // The accelerator's error is reported; the picture
                        // still takes its place in output order, mid-grey.
                        self.accel_error.get_or_insert(e);
                        let grey = |p: &Plane, bd: u32| Plane::new(p.width, p.height, 1 << (bd - 1));
                        let planes = [grey(&pic.planes[0], sps.bit_depth_y), grey(&pic.planes[1], sps.bit_depth_c), grey(&pic.planes[2], sps.bit_depth_c)];
                        DecodedFrame::from_planes(&planes, fmt, sps.bit_depth_y as u8, crop, color)
                    }
                },
                None => DecodedFrame::from_planes(&pic.planes, fmt, sps.bit_depth_y as u8, crop, color),
            };
            let id = self.next_id;
            self.next_id += 1;
            self.done.push(HevcPicture {
                id,
                frame,
                poc: pic.poc,
                new_sequence: cur.new_sequence,
            });
            // §C.5.2.3: everything waiting ages by one; then "additional
            // bumping".
            for e in &mut self.pending {
                e.2 += 1;
            }
            self.pending.push((id, pic.poc, 0));
            self.bump_while_over(sps, false);
        }
        let layout = pic.layout.clone();
        self.dpb.push(RefPic {
            poc: pic.poc,
            long_term: false,
            planes: Arc::new(pic.planes),
            motion: Arc::new(Motion {
                mv: pic.mv,
                w4: layout.w4,
                ctb_slice: pic.ctb_slice,
                slices: pic.slices,
                log2_ctb: layout.log2_ctb as usize,
                ctb_w: layout.ctb_w,
            }),
        });
        // A broken stream that never lets go of references must not grow
        // the store without bound.
        let cap = (sps.max_dec_pic_buffering as usize).max(1) + 16;
        while self.dpb.len() > cap {
            self.dpb.remove(0);
        }
    }
}
