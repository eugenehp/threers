//! A from-scratch H.264 / AVC decoder.
//!
//! Progressive (frame-coded) streams: Baseline, Main and High profiles —
//! CAVLC and CABAC, I/P/B slices, multiple slices, 8×8 transforms, scaling
//! matrices, weighted prediction (explicit and implicit), spatial and
//! temporal direct, long-term references and every memory management
//! operation, 4:2:0 or monochrome, 8 bits and up. Not supported, and refused
//! with an error rather than decoded wrongly: interlaced coding (field
//! pictures and MBAFF), 4:2:2/4:4:4, slice groups (FMO), data partitioning,
//! and SP/SI slices.
//!
//! Feed it NAL units with [`H264Decoder::decode_nal`]; each finished picture
//! comes out of [`H264Decoder::finish_picture`] (called for you when a new
//! picture starts) tagged with its picture order count.

#[cfg(feature = "video")]
mod cabac_init;
#[cfg(feature = "video")]
mod cavlc;
#[cfg(feature = "video")]
mod deblock;
#[cfg(feature = "video")]
mod inter;
#[cfg(feature = "video")]
mod intra;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "video"), allow(dead_code))]
pub mod params;
#[cfg(feature = "video")]
mod parse_cabac;
#[cfg(feature = "video")]
mod parse_cavlc;
#[cfg(feature = "video")]
mod slice;
// Parameter sets parse without the decoder: the MP4 muxer reads them.
#[cfg_attr(not(feature = "video"), allow(dead_code))]
mod tables;
#[cfg(feature = "video")]
mod types;

#[cfg(feature = "video")]
use std::sync::Arc;

#[cfg(feature = "video")]
use self::cavlc::CavlcTables;
#[cfg(feature = "video")]
use self::params::{Mmco, Pps, SliceHeader, SliceType, Sps};
#[cfg(feature = "video")]
use self::slice::{CurPic, SliceDeblock, SliceDec};
#[cfg(feature = "video")]
use self::types::{MbInfo, RefEntry, RefPic};
#[cfg(feature = "video")]
use super::bits::unescape;
#[cfg(feature = "video")]
use super::picture::{ChromaFormat, ColorInfo, DecodedFrame, Plane};

#[cfg(feature = "video")]
pub type Result<T> = std::result::Result<T, String>;

#[cfg(feature = "video")]
/// A decoded picture and where it goes in display order.
#[derive(Debug, Clone)]
pub struct H264Picture {
    pub frame: DecodedFrame,
    /// `PicOrderCnt`. Resets at every IDR picture (and memory management
    /// operation 5), so it orders pictures only within one such period.
    pub poc: i32,
    /// Whether this picture is an IDR picture, which starts a new POC
    /// period.
    pub idr: bool,
}

#[cfg(feature = "video")]
/// The picture being decoded, with what marking needs once it is done.
struct Current {
    pic: CurPic,
    sps: Sps,
    idr: bool,
    nal_ref_idc: u32,
    frame_num: u32,
    long_term_reference: bool,
    mmcos: Option<Vec<Mmco>>,
    /// For detecting the first slice of the next picture.
    pps_id: u32,
    poc_lsb: u32,
    poc_msb: i32,
    top_poc: i32,
    bottom_poc: i32,
    frame_num_offset: i32,
}

#[cfg(feature = "video")]
pub struct H264Decoder {
    spss: Vec<Option<Sps>>,
    ppss: Vec<Option<Pps>>,
    tables: CavlcTables,
    cur: Option<Current>,
    dpb: Vec<RefPic>,
    next_id: u32,
    max_lt_idx: Option<u32>,
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num_offset: i32,
    prev_frame_num: u32,
    prev_ref_frame_num: u32,
    prev_had_mmco5: bool,
    /// Size of the last activated SPS, to notice a resolution change.
    active_size: Option<(u32, u32)>,
    done: Vec<H264Picture>,
}

#[cfg(feature = "video")]
impl Default for H264Decoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "video")]
impl H264Decoder {
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
            active_size: None,
            done: Vec::new(),
        }
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
                self.decode_slice(&rbsp, nal_type, nal_ref_idc)
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

    fn decode_slice(&mut self, rbsp: &[u8], nal_type: u32, nal_ref_idc: u32) -> Result<()> {
        let hdr = SliceHeader::parse(rbsp, nal_type, nal_ref_idc, &self.spss, &self.ppss)?;
        if hdr.redundant_pic_cnt > 0 {
            return Ok(());
        }
        let new_pic = match &self.cur {
            None => true,
            Some(c) => {
                hdr.first_mb == 0
                    || c.frame_num != hdr.frame_num
                    || c.pps_id != hdr.pps_id
                    || c.idr != hdr.idr
                    || (c.nal_ref_idc == 0) != (nal_ref_idc == 0)
                    || c.poc_lsb != hdr.poc_lsb
            }
        };
        if new_pic {
            self.finish_picture();
            self.start_picture(&hdr)?;
        }
        let pps = self.ppss[hdr.pps_id as usize].clone().ok_or("missing PPS")?;
        let refs = self.build_lists(&hdr)?;
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
            &self.tables,
        );
        dec.decode(rbsp)
    }

    fn start_picture(&mut self, hdr: &SliceHeader) -> Result<()> {
        let pps = self.ppss[hdr.pps_id as usize].as_ref().ok_or("missing PPS")?;
        let sps = self.spss[pps.sps_id as usize]
            .clone()
            .ok_or("missing SPS")?;
        if sps.transform_bypass {
            return Err("lossless (transform bypass) coding is not supported".into());
        }
        let size = (sps.width_mbs, sps.height_mbs);
        if self.active_size != Some(size) {
            self.dpb.clear();
            self.active_size = Some(size);
        }
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        if hdr.idr {
            self.dpb.clear();
            self.max_lt_idx = None;
        } else {
            if self.dpb.is_empty() && hdr.slice_type != SliceType::I {
                // Joined mid-stream: nothing to predict from. Carry on and let
                // the slice fail on its first reference instead of guessing.
            }
            let expected = (self.prev_ref_frame_num + 1) % max_frame_num;
            if hdr.frame_num != self.prev_ref_frame_num && hdr.frame_num != expected {
                self.fill_frame_num_gap(&sps, hdr.frame_num);
            }
        }

        // Picture order count (§8.2.1).
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
                (top, top + hdr.delta_poc_bottom)
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
                    (top, top + sps.offset_for_top_to_bottom_field + hdr.delta_poc[1])
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
        let planes = if sps.chroma_format_idc == 0 {
            [
                Plane::new(w_mbs * 16, h_mbs * 16, fill_y),
                Plane::new(0, 0, 0),
                Plane::new(0, 0, 0),
            ]
        } else {
            [
                Plane::new(w_mbs * 16, h_mbs * 16, fill_y),
                Plane::new(w_mbs * 8, h_mbs * 8, fill_c),
                Plane::new(w_mbs * 8, h_mbs * 8, fill_c),
            ]
        };
        let id = self.next_id;
        self.next_id += 1;
        let poc = top.min(bottom);
        self.cur = Some(Current {
            pic: CurPic {
                planes,
                mbs: vec![MbInfo::default(); w_mbs * h_mbs],
                w_mbs,
                h_mbs,
                id,
                poc,
                slices: Vec::new(),
                decoded: 0,
            },
            sps,
            idr: hdr.idr,
            nal_ref_idc: hdr.nal_ref_idc,
            frame_num: hdr.frame_num,
            long_term_reference: hdr.long_term_reference,
            mmcos: hdr.mmcos.clone(),
            pps_id: hdr.pps_id,
            poc_lsb: hdr.poc_lsb,
            poc_msb,
            top_poc: top,
            bottom_poc: bottom,
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
        let src = self.dpb.iter().max_by_key(|p| p.id).map(|p| p.planes.clone());
        let Some(planes) = src else {
            self.prev_ref_frame_num = frame_num.wrapping_sub(1) % max;
            return;
        };
        while f != frame_num && guard < max {
            self.sliding_window(sps);
            let id = self.next_id;
            self.next_id += 1;
            let mbs = Arc::new(vec![
                MbInfo::default();
                (sps.width_mbs * sps.height_mbs) as usize
            ]);
            self.dpb.push(RefPic {
                id,
                planes: planes.clone(),
                mbs,
                poc: 0,
                frame_num: f,
                frame_num_wrap: 0,
                long_term: false,
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
            if !p.long_term {
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
        let cur_poc = cur.pic.poc;
        self.update_wraps(hdr.frame_num, max_frame_num);
        let entry = |i: usize, p: &RefPic| RefEntry {
            dpb: i,
            id: p.id,
            poc: p.poc,
            long_term: p.long_term,
        };
        let mut lists: [Vec<RefEntry>; 2] = [Vec::new(), Vec::new()];
        if hdr.slice_type == SliceType::I {
            return Ok(lists);
        }
        let mut short: Vec<(usize, &RefPic)> =
            self.dpb.iter().enumerate().filter(|(_, p)| !p.long_term).collect();
        let mut long: Vec<(usize, &RefPic)> =
            self.dpb.iter().enumerate().filter(|(_, p)| p.long_term).collect();
        long.sort_by_key(|(_, p)| p.lt_idx);
        if hdr.slice_type == SliceType::P {
            short.sort_by_key(|(_, p)| std::cmp::Reverse(p.frame_num_wrap));
            lists[0] = short.iter().chain(long.iter()).map(|&(i, p)| entry(i, p)).collect();
        } else {
            let mut before: Vec<_> = short.iter().filter(|(_, p)| p.poc < cur_poc).collect();
            let mut after: Vec<_> = short.iter().filter(|(_, p)| p.poc > cur_poc).collect();
            before.sort_by_key(|(_, p)| std::cmp::Reverse(p.poc));
            after.sort_by_key(|(_, p)| p.poc);
            lists[0] = before
                .iter()
                .chain(after.iter())
                .map(|&&(i, p)| entry(i, p))
                .chain(long.iter().map(|&(i, p)| entry(i, p)))
                .collect();
            lists[1] = after
                .iter()
                .chain(before.iter())
                .map(|&&(i, p)| entry(i, p))
                .chain(long.iter().map(|&(i, p)| entry(i, p)))
                .collect();
            if lists[1].len() > 1 && lists[0].iter().map(|e| e.id).eq(lists[1].iter().map(|e| e.id)) {
                lists[1].swap(0, 1);
            }
        }
        for l in 0..2 {
            let n = hdr.num_ref_idx[l] as usize;
            lists[l].truncate(n);
            if !hdr.list_mods[l].is_empty() {
                let mut pred = hdr.frame_num as i32;
                let mut idx = 0usize;
                for m in &hdr.list_mods[l] {
                    let target = if m.idc < 2 {
                        let abs = m.value as i32 + 1;
                        let mut no_wrap = if m.idc == 0 { pred - abs } else { pred + abs };
                        if no_wrap < 0 {
                            no_wrap += max_frame_num as i32;
                        } else if no_wrap >= max_frame_num as i32 {
                            no_wrap -= max_frame_num as i32;
                        }
                        pred = no_wrap;
                        let pic_num = if no_wrap > hdr.frame_num as i32 {
                            no_wrap - max_frame_num as i32
                        } else {
                            no_wrap
                        };
                        self.dpb
                            .iter()
                            .enumerate()
                            .find(|(_, p)| !p.long_term && p.frame_num_wrap == pic_num)
                    } else {
                        self.dpb
                            .iter()
                            .enumerate()
                            .find(|(_, p)| p.long_term && p.lt_idx == m.value)
                    };
                    let Some((i, p)) = target else {
                        return Err("ref_pic_list_modification names a missing picture".into());
                    };
                    let e = entry(i, p);
                    if idx > lists[l].len() {
                        idx = lists[l].len();
                    }
                    lists[l].insert(idx, e);
                    idx += 1;
                    let mut k = idx;
                    while k < lists[l].len() {
                        if lists[l][k].id == e.id {
                            lists[l].remove(k);
                        } else {
                            k += 1;
                        }
                    }
                }
                lists[l].truncate(n);
            }
        }
        Ok(lists)
    }

    fn sliding_window(&mut self, sps: &Sps) {
        let max = sps.max_num_ref_frames.max(1) as usize;
        while self.dpb.len() >= max {
            let Some(oldest) = self
                .dpb
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.long_term)
                .min_by_key(|(_, p)| p.frame_num_wrap)
                .map(|(i, _)| i)
            else {
                break;
            };
            self.dpb.remove(oldest);
        }
    }

    /// Finish the picture in progress, if any: deblock, mark references, and
    /// queue it for output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        let sps = &cur.sps;
        deblock::deblock_picture(
            &mut cur.pic,
            sps.bit_depth_luma,
            sps.bit_depth_chroma,
            sps.chroma_format_idc != 0,
        );
        let chroma = if sps.chroma_format_idc == 0 {
            ChromaFormat::Monochrome
        } else {
            ChromaFormat::Yuv420
        };
        let crop = sps.crop.map(|v| v as usize);
        let color = ColorInfo::from_vui(
            sps.matrix_coefficients,
            sps.full_range,
            sps.height_mbs * 16 - sps.crop[2] - sps.crop[3],
        );
        let frame = DecodedFrame::from_planes(
            &cur.pic.planes,
            chroma,
            sps.bit_depth_luma as u8,
            crop,
            color,
        );
        let mut poc = cur.pic.poc;
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        let had_mmco5 = cur
            .mmcos
            .as_ref()
            .is_some_and(|ops| ops.iter().any(|m| m.op == 5));
        if cur.nal_ref_idc != 0 {
            self.update_wraps(cur.frame_num, max_frame_num);
            let mut long_term = false;
            let mut lt_idx = 0;
            if cur.idr {
                if cur.long_term_reference {
                    long_term = true;
                    self.max_lt_idx = Some(0);
                } else {
                    self.max_lt_idx = None;
                }
            } else if let Some(ops) = &cur.mmcos {
                for m in ops {
                    let pic_num_x = cur.frame_num as i32 - (m.diff_pic_nums_minus1 as i32 + 1);
                    match m.op {
                        1 => self
                            .dpb
                            .retain(|p| p.long_term || p.frame_num_wrap != pic_num_x),
                        2 => self
                            .dpb
                            .retain(|p| !p.long_term || p.lt_idx != m.long_term_pic_num),
                        3 => {
                            let idx = m.long_term_frame_idx;
                            if let Some(i) = self
                                .dpb
                                .iter()
                                .position(|p| !p.long_term && p.frame_num_wrap == pic_num_x)
                            {
                                let id = self.dpb[i].id;
                                self.dpb.retain(|p| !(p.long_term && p.lt_idx == idx && p.id != id));
                                if let Some(p) = self.dpb.iter_mut().find(|p| p.id == id) {
                                    p.long_term = true;
                                    p.lt_idx = idx;
                                }
                            }
                        }
                        4 => {
                            let max = m.max_long_term_frame_idx_plus1;
                            self.max_lt_idx = max.checked_sub(1);
                            self.dpb.retain(|p| !p.long_term || p.lt_idx < max);
                        }
                        5 => {
                            self.dpb.clear();
                            self.max_lt_idx = None;
                        }
                        6 => {
                            let idx = m.long_term_frame_idx;
                            self.dpb.retain(|p| !(p.long_term && p.lt_idx == idx));
                            long_term = true;
                            lt_idx = idx;
                        }
                        _ => {}
                    }
                }
                if !long_term {
                    // A stream that overfills the DPB with adaptive marking
                    // is broken; slide rather than grow without bound.
                    self.sliding_window(&cur.sps);
                }
            } else {
                self.sliding_window(&cur.sps);
            }
            if had_mmco5 {
                // The picture now counts as frame_num 0 with its POC rebased.
                let temp = cur.top_poc.min(cur.bottom_poc);
                cur.top_poc -= temp;
                cur.bottom_poc -= temp;
                poc = 0;
            }
            let frame_num = if had_mmco5 { 0 } else { cur.frame_num };
            self.dpb.push(RefPic {
                id: cur.pic.id,
                planes: Arc::new(std::mem::replace(
                    &mut cur.pic.planes,
                    [Plane::new(0, 0, 0), Plane::new(0, 0, 0), Plane::new(0, 0, 0)],
                )),
                mbs: Arc::new(std::mem::take(&mut cur.pic.mbs)),
                poc,
                frame_num,
                frame_num_wrap: frame_num as i32,
                long_term,
                lt_idx,
            });
            self.prev_ref_frame_num = frame_num;
            if had_mmco5 {
                self.prev_poc_msb = 0;
                self.prev_poc_lsb = cur.top_poc;
            } else {
                self.prev_poc_msb = cur.poc_msb;
                self.prev_poc_lsb = cur.poc_lsb as i32;
            }
        }
        self.prev_had_mmco5 = had_mmco5;
        self.prev_frame_num = if had_mmco5 { 0 } else { cur.frame_num };
        self.prev_frame_num_offset = if had_mmco5 { 0 } else { cur.frame_num_offset };
        self.done.push(H264Picture {
            frame,
            poc,
            idr: cur.idr || had_mmco5,
        });
    }
}
