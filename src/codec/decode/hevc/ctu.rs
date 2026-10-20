//! HEVC slice segment data: CTU parsing and reconstruction (§7.3.8, §8.4,
//! §8.6). Motion vector derivation is in `mv`, loop filters in `filters`.

use super::cabac_init as ci;
use super::inter::{self, Weighting};
use super::intra;
use super::params::{Pps, SliceHeader, SliceType, Sps};
use super::pic::{flag, MvField, PicState, Sao};
use super::tables::{chroma_qp, CTX_IDX_MAP, SCAN4, SCAN_SB1, SCAN_SB2, SCAN_SB4, SCAN_SB8};
use super::transform;
use super::{RefEntry, RefPic};
use crate::codec::decode::bits::BitReader;
use crate::codec::decode::cabac::{Cabac, Ctx};

pub type Result<T> = std::result::Result<T, String>;

/// CABAC state carried between the slice segments of one picture.
#[derive(Default)]
pub struct Carry {
    /// Contexts saved after the second CTB of the last WPP row.
    pub wpp: Option<Vec<Ctx>>,
    /// Contexts at the end of the previous slice segment.
    pub ds: Option<Vec<Ctx>>,
    /// `QpY` of the last coding unit — `qPY_PREV` for a dependent segment.
    pub qp_prev: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartMode {
    P2Nx2N,
    P2NxN,
    PNx2N,
    PNxN,
    P2NxnU,
    P2NxnD,
    PnLx2N,
    PnRx2N,
}

impl PartMode {
    /// Prediction blocks as `(x, y, w, h)` within a CU of size `n`.
    pub fn parts(self, n: usize) -> Vec<(usize, usize, usize, usize)> {
        let (h2, q) = (n / 2, n / 4);
        match self {
            PartMode::P2Nx2N => vec![(0, 0, n, n)],
            PartMode::P2NxN => vec![(0, 0, n, h2), (0, h2, n, h2)],
            PartMode::PNx2N => vec![(0, 0, h2, n), (h2, 0, h2, n)],
            PartMode::PNxN => vec![(0, 0, h2, h2), (h2, 0, h2, h2), (0, h2, h2, h2), (h2, h2, h2, h2)],
            PartMode::P2NxnU => vec![(0, 0, n, q), (0, q, n, n - q)],
            PartMode::P2NxnD => vec![(0, 0, n, n - q), (0, n - q, n, q)],
            PartMode::PnLx2N => vec![(0, 0, q, n), (q, 0, n - q, n)],
            PartMode::PnRx2N => vec![(0, 0, n - q, n), (n - q, 0, q, n)],
        }
    }
}

pub struct SliceDec<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub sh: &'a SliceHeader,
    pub pic: &'a mut PicState,
    pub dpb: &'a [RefPic],
    pub refs: &'a [Vec<RefEntry>; 2],
    pub carry: &'a mut Carry,
    pub slice_idx: u16,
    /// `SliceAddrRs`: raster address of the slice's first CTB.
    pub slice_addr: u32,
    pub slice_qp: i32,
    pub ctx: Vec<Ctx>,
    pub ctb_rs: usize,
    qp_y: i32,
    qp_pred: i32,
    last_qp: i32,
    first_qg: bool,
    cu_qp_delta_coded: bool,
    cu_qp_delta: i32,
    cu_bypass: bool,
    cu_intra: bool,
    cu_depth: u32,
    intra_split: bool,
    max_trafo_depth: u32,
    ipm_c: u32,
    pub part_mode: PartMode,
    merge_2nx2n: bool,
    /// Index into `dpb` of the collocated picture.
    pub col: Option<usize>,
    pub no_backward_pred: bool,
    coeff: Vec<i32>,
    pub mc: [Vec<i16>; 2],
    mc_scratch: inter::Scratch,
}

impl<'a> SliceDec<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sps: &'a Sps,
        pps: &'a Pps,
        sh: &'a SliceHeader,
        pic: &'a mut PicState,
        dpb: &'a [RefPic],
        refs: &'a [Vec<RefEntry>; 2],
        carry: &'a mut Carry,
        slice_idx: u16,
        slice_addr: u32,
    ) -> Self {
        let slice_qp = pps.init_qp + sh.qp_delta;
        let col = if sh.temporal_mvp && sh.slice_type != SliceType::I {
            let l = if sh.collocated_from_l0 { 0 } else { 1 };
            refs[l].get(sh.collocated_ref_idx as usize).map(|e| e.dpb)
        } else {
            None
        };
        let poc = pic.poc;
        let no_backward_pred = refs.iter().all(|l| l.iter().all(|e| e.poc <= poc));
        SliceDec {
            sps,
            pps,
            sh,
            pic,
            dpb,
            refs,
            carry,
            slice_idx,
            slice_addr,
            slice_qp,
            ctx: Vec::new(),
            ctb_rs: 0,
            qp_y: slice_qp,
            qp_pred: slice_qp,
            last_qp: slice_qp,
            first_qg: true,
            cu_qp_delta_coded: false,
            cu_qp_delta: 0,
            cu_bypass: false,
            cu_intra: false,
            cu_depth: 0,
            intra_split: false,
            max_trafo_depth: 0,
            ipm_c: 0,
            part_mode: PartMode::P2Nx2N,
            merge_2nx2n: false,
            col,
            no_backward_pred,
            coeff: vec![0; 32 * 32],
            mc: [vec![0; 64 * 64], vec![0; 64 * 64]],
            mc_scratch: inter::Scratch::default(),
        }
    }

    fn init_contexts(&mut self) {
        let init_type = match (self.sh.slice_type, self.sh.cabac_init_flag) {
            (SliceType::I, _) => 0,
            (SliceType::P, false) | (SliceType::B, true) => 1,
            _ => 2,
        };
        self.ctx = ci::INIT[init_type]
            .iter()
            .map(|&v| Ctx::hevc(v, self.slice_qp))
            .collect();
    }

    #[inline(always)]
    fn bin(&mut self, c: &mut Cabac, idx: usize) -> u32 {
        c.decision(&mut self.ctx[idx])
    }

    /// Z-scan availability (§6.4.1) of luma location `(xn, yn)` from
    /// `(xc, yc)`.
    #[inline]
    pub fn avail(&self, xc: isize, yc: isize, xn: isize, yn: isize) -> bool {
        if xn < 0 || yn < 0 || xn >= self.pic.width as isize || yn >= self.pic.height as isize {
            return false;
        }
        let l = &self.pic.layout;
        let zn = l.z[(yn as usize >> 2) * l.w4 + (xn as usize >> 2)];
        let zc = l.z[(yc as usize >> 2) * l.w4 + (xc as usize >> 2)];
        if zn > zc {
            return false;
        }
        let ctb_n = self.pic.ctb_rs(xn as usize, yn as usize);
        let ctb_c = self.pic.ctb_rs(xc as usize, yc as usize);
        let sn = self.pic.ctb_slice[ctb_n];
        if sn == u16::MAX || self.pic.slices[sn as usize].addr_rs != self.slice_addr {
            return false;
        }
        l.tile[ctb_n] == l.tile[ctb_c]
    }

    /// Decode the slice segment's CTUs.
    pub fn decode(&mut self, rbsp: &[u8]) -> Result<()> {
        let layout = self.pic.layout.clone();
        let total = layout.ctb_w * layout.ctb_h;
        let log2 = self.sps.log2_ctb;
        let mut ts = layout.rs2ts[self.sh.segment_address as usize] as usize;
        let first_ts = ts;
        if self.sh.data_offset > rbsp.len() {
            return Err("slice data starts past the NAL".into());
        }
        let mut c = Cabac::new(rbsp, self.sh.data_offset);
        loop {
            let rs = layout.ts2rs[ts] as usize;
            self.ctb_rs = rs;
            let (cx, cy) = (rs % layout.ctb_w, rs / layout.ctb_w);
            let tile_start = layout.tile_start(ts);
            let row_start = self.pps.entropy_sync && layout.tile_col_start(cx as u32) == cx as u32;
            // The CTB is part of this slice before anything asks.
            self.pic.ctb_slice[rs] = self.slice_idx;
            if tile_start {
                self.init_contexts();
                self.first_qg = true;
            } else if row_start {
                let (x0, y0) = ((cx << log2) as isize, (cy << log2) as isize);
                let t = self.avail(x0, y0, x0 + (1 << log2), y0 - (1 << log2));
                match (&self.carry.wpp, t) {
                    (Some(saved), true) => self.ctx = saved.clone(),
                    _ => self.init_contexts(),
                }
                self.first_qg = true;
            } else if ts == first_ts {
                if self.sh.dependent {
                    match &self.carry.ds {
                        Some(saved) => self.ctx = saved.clone(),
                        None => self.init_contexts(),
                    }
                    self.last_qp = self.carry.qp_prev;
                    self.first_qg = false;
                } else {
                    self.init_contexts();
                    self.first_qg = true;
                }
            }
            self.sao_syntax(&mut c, cx, cy)?;
            self.coding_quadtree(&mut c, cx << log2, cy << log2, log2, 0)?;
            let end = c.terminate();
            self.pic.decoded_ctbs += 1;
            if self.pps.entropy_sync
                && (rs % layout.ctb_w == 1 || (rs > 1 && layout.tile[rs] != layout.tile[rs - 2]))
            {
                self.carry.wpp = Some(self.ctx.clone());
            }
            self.carry.qp_prev = self.last_qp;
            if c.overrun() {
                return Err("CABAC slice data overrun".into());
            }
            ts += 1;
            if end == 1 {
                self.carry.ds = Some(self.ctx.clone());
                break;
            }
            if ts >= total {
                return Err("slice runs past the end of the picture".into());
            }
            let rs_next = layout.ts2rs[ts] as usize;
            if (self.pps.tiles && layout.tile_start(ts))
                || (self.pps.entropy_sync
                    && (rs_next.is_multiple_of(layout.ctb_w)
                        || layout.tile[rs_next] != layout.tile[rs_next - 1]))
            {
                if c.terminate() != 1 {
                    return Err("end_of_subset_one_bit missing".into());
                }
                c.restart(c.aligned_byte_pos());
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // SAO syntax (§7.3.8.3)
    // ------------------------------------------------------------------

    fn sao_syntax(&mut self, c: &mut Cabac, cx: usize, cy: usize) -> Result<()> {
        let rs = self.ctb_rs;
        if !self.sh.sao_luma && !self.sh.sao_chroma {
            self.pic.sao[rs] = [Sao::default(); 3];
            return Ok(());
        }
        let l = self.pic.layout.clone();
        let w = l.ctb_w;
        let mut merge_left = false;
        let mut merge_up = false;
        if cx > 0 && rs > self.slice_addr as usize && l.tile[rs] == l.tile[rs - 1] {
            merge_left = self.bin(c, ci::SAO_MERGE) == 1;
        }
        if cy > 0 && !merge_left && rs - w >= self.slice_addr as usize && l.tile[rs] == l.tile[rs - w] {
            merge_up = self.bin(c, ci::SAO_MERGE) == 1;
        }
        if merge_left {
            self.pic.sao[rs] = self.pic.sao[rs - 1];
            return Ok(());
        }
        if merge_up {
            self.pic.sao[rs] = self.pic.sao[rs - w];
            return Ok(());
        }
        let mut out = [Sao::default(); 3];
        for cidx in 0..3 {
            if (cidx == 0 && !self.sh.sao_luma) || (cidx > 0 && !self.sh.sao_chroma) {
                continue;
            }
            let bd = if cidx == 0 { self.sps.bit_depth_y } else { self.sps.bit_depth_c };
            let kind = if cidx == 2 {
                out[1].kind
            } else if self.bin(c, ci::SAO_TYPE) == 0 {
                0
            } else if c.bypass() == 0 {
                1
            } else {
                2
            };
            out[cidx].kind = kind;
            if kind == 0 {
                continue;
            }
            // Offsets are not scaled up above 10 bits: version 1 did, the
            // range extensions replaced that with `log2_sao_offset_scale`
            // (0 unless signalled, and a non-zero one is refused).
            let cmax = (1 << (bd.min(10) - 5)) - 1;
            let mut abs = [0i16; 4];
            for a in abs.iter_mut() {
                let mut v = 0;
                while v < cmax && c.bypass() == 1 {
                    v += 1;
                }
                *a = v as i16;
            }
            if kind == 1 {
                for a in abs.iter_mut() {
                    if *a != 0 && c.bypass() == 1 {
                        *a = -*a;
                    }
                }
                out[cidx].band_or_class = c.bypass_bits(5) as u8;
                out[cidx].offsets[..4].copy_from_slice(&abs[..4]);
            } else {
                out[cidx].band_or_class = if cidx == 2 {
                    out[1].band_or_class
                } else {
                    c.bypass_bits(2) as u8
                };
                out[cidx].offsets = [abs[0], abs[1], -abs[2], -abs[3]];
            }
        }
        self.pic.sao[rs] = out;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Coding quadtree and coding unit
    // ------------------------------------------------------------------

    fn start_qg(&mut self, x: usize, y: usize) {
        self.cu_qp_delta_coded = false;
        self.cu_qp_delta = 0;
        let prev = if self.first_qg { self.slice_qp } else { self.last_qp };
        self.first_qg = false;
        let ctb = self.ctb_rs;
        let nb = |s: &Self, xn: isize, yn: isize| -> i32 {
            if s.avail(x as isize, y as isize, xn, yn)
                && s.pic.ctb_rs(xn as usize, yn as usize) == ctb
            {
                s.pic.qp[s.pic.idx4(xn as usize, yn as usize)] as i32
            } else {
                prev
            }
        };
        let a = nb(self, x as isize - 1, y as isize);
        let b = nb(self, x as isize, y as isize - 1);
        self.qp_pred = (a + b + 1) >> 1;
    }

    fn update_qp_y(&mut self) {
        let off = 6 * (self.sps.bit_depth_y as i32 - 8);
        self.qp_y = ((self.qp_pred + self.cu_qp_delta + 52 + 2 * off) % (52 + off)) - off;
    }

    fn coding_quadtree(&mut self, c: &mut Cabac, x0: usize, y0: usize, log2: u32, depth: u32) -> Result<()> {
        let n = 1usize << log2;
        let (w, h) = (self.pic.width, self.pic.height);
        let split = if x0 + n <= w && y0 + n <= h && log2 > self.sps.log2_min_cb {
            let cond = |s: &Self, xn: isize, yn: isize| {
                (s.avail(x0 as isize, y0 as isize, xn, yn)
                    && s.pic.depth[s.pic.idx4(xn as usize, yn as usize)] as u32 > depth) as usize
            };
            let inc = cond(self, x0 as isize - 1, y0 as isize) + cond(self, x0 as isize, y0 as isize - 1);
            self.bin(c, ci::SPLIT_CU + inc) == 1
        } else {
            log2 > self.sps.log2_min_cb
        };
        // A quantization group is the quadtree node of the minimum delta-QP
        // size, or a coding unit larger than that. Larger nodes reset the
        // delta too, but only the group itself predicts a QP — starting one
        // at every level would spend the "first group in the row" on a
        // node that is not a group.
        let qg = self.sps.log2_ctb.saturating_sub(self.pps.diff_cu_qp_delta_depth);
        if self.pps.cu_qp_delta && (log2 == qg || (log2 > qg && !split)) {
            self.start_qg(x0, y0);
        }
        if split {
            let h2 = n / 2;
            for (dx, dy) in [(0, 0), (h2, 0), (0, h2), (h2, h2)] {
                if x0 + dx < w && y0 + dy < h {
                    self.coding_quadtree(c, x0 + dx, y0 + dy, log2 - 1, depth + 1)?;
                }
            }
            Ok(())
        } else {
            self.coding_unit(c, x0, y0, log2, depth)
        }
    }

    /// Set a per-4×4 array over a block, clipped to the picture.
    fn fill4<T: Copy>(arr: &mut [T], w4: usize, x: usize, y: usize, w: usize, h: usize, pw: usize, ph: usize, v: T) {
        let x1 = (x + w).min(pw);
        let y1 = (y + h).min(ph);
        for yy in (y..y1).step_by(4) {
            for xx in (x..x1).step_by(4) {
                arr[(yy >> 2) * w4 + (xx >> 2)] = v;
            }
        }
    }

    fn set_flags(&mut self, x: usize, y: usize, w: usize, h: usize, set: u8, clear: u8) {
        let (w4, pw, ph) = (self.pic.layout.w4, self.pic.width, self.pic.height);
        for yy in (y..(y + h).min(ph)).step_by(4) {
            for xx in (x..(x + w).min(pw)).step_by(4) {
                let f = &mut self.pic.flags[(yy >> 2) * w4 + (xx >> 2)];
                *f = (*f & !clear) | set;
            }
        }
    }

    /// Mark the left and top edges of a block for deblocking.
    fn mark_edges(&mut self, x: usize, y: usize, w: usize, h: usize, pu: bool) {
        let (w4, pw, ph) = (self.pic.layout.w4, self.pic.width, self.pic.height);
        for yy in (y..(y + h).min(ph)).step_by(4) {
            let i = (yy >> 2) * w4 + (x >> 2);
            if pu {
                self.pic.pu_edge[i] |= 1;
            } else {
                self.pic.flags[i] |= flag::EDGE_V;
            }
        }
        for xx in (x..(x + w).min(pw)).step_by(4) {
            let i = (y >> 2) * w4 + (xx >> 2);
            if pu {
                self.pic.pu_edge[i] |= 2;
            } else {
                self.pic.flags[i] |= flag::EDGE_H;
            }
        }
    }

    fn coding_unit(&mut self, c: &mut Cabac, x0: usize, y0: usize, log2: u32, depth: u32) -> Result<()> {
        let n = 1usize << log2;
        let (w4, pw, ph) = (self.pic.layout.w4, self.pic.width, self.pic.height);
        self.cu_depth = depth;
        self.cu_bypass = self.pps.transquant_bypass && self.bin(c, ci::TRANSQUANT_BYPASS) == 1;
        let skip = if self.sh.slice_type != SliceType::I {
            let cond = |s: &Self, xn: isize, yn: isize| {
                (s.avail(x0 as isize, y0 as isize, xn, yn)
                    && s.pic.flags[s.pic.idx4(xn as usize, yn as usize)] & flag::SKIP != 0) as usize
            };
            let inc = cond(self, x0 as isize - 1, y0 as isize) + cond(self, x0 as isize, y0 as isize - 1);
            self.bin(c, ci::SKIP + inc) == 1
        } else {
            false
        };
        Self::fill4(&mut self.pic.depth, w4, x0, y0, n, n, pw, ph, depth as u8);
        self.set_flags(x0, y0, n, n, if skip { flag::SKIP } else { 0 }, 0xFF);
        if self.cu_bypass {
            self.set_flags(x0, y0, n, n, flag::NO_FILTER, 0);
        }
        if !self.pps.cu_qp_delta {
            self.qp_y = self.slice_qp;
        } else {
            self.update_qp_y();
        }
        // The CU boundary is a transform-block boundary whatever follows.
        self.mark_edges(x0, y0, n, n, false);
        let mut pcm = false;
        if skip {
            self.cu_intra = false;
            self.part_mode = PartMode::P2Nx2N;
            Self::fill4(&mut self.pic.ipm, w4, x0, y0, n, n, pw, ph, 1u8);
            self.prediction_unit(c, x0, y0, n, x0, y0, n, n, 0, true)?;
        } else {
            self.cu_intra = self.sh.slice_type == SliceType::I || self.bin(c, ci::PRED_MODE) == 1;
            self.part_mode = if !self.cu_intra || log2 == self.sps.log2_min_cb {
                self.decode_part_mode(c, log2)
            } else {
                PartMode::P2Nx2N
            };
            self.intra_split = self.cu_intra && self.part_mode == PartMode::PNxN;
            if self.cu_intra {
                self.set_flags(x0, y0, n, n, flag::INTRA, 0);
                Self::fill4(&mut self.pic.mv, w4, x0, y0, n, n, pw, ph, MvField::INTRA);
                if self.part_mode == PartMode::P2Nx2N
                    && self.sps.pcm
                    && log2 >= self.sps.log2_min_pcm
                    && log2 <= self.sps.log2_max_pcm
                {
                    pcm = c.terminate() == 1;
                }
                if pcm {
                    self.pcm_samples(c, x0, y0, n)?;
                } else {
                    self.intra_modes(c, x0, y0, n)?;
                }
            } else {
                Self::fill4(&mut self.pic.ipm, w4, x0, y0, n, n, pw, ph, 1u8);
                self.merge_2nx2n = false;
                for (i, (px, py, w, h)) in self.part_mode.parts(n).into_iter().enumerate() {
                    if (px, py) != (0, 0) {
                        self.mark_edges(x0 + px, y0 + py, w, h, true);
                    }
                    self.prediction_unit(c, x0, y0, n, x0 + px, y0 + py, w, h, i, false)?;
                }
            }
            if !pcm {
                let root_cbf = if !self.cu_intra && !(self.part_mode == PartMode::P2Nx2N && self.merge_2nx2n) {
                    self.bin(c, ci::ROOT_CBF) == 1
                } else {
                    true
                };
                if root_cbf {
                    self.max_trafo_depth = if self.cu_intra {
                        self.sps.max_th_depth_intra + self.intra_split as u32
                    } else {
                        self.sps.max_th_depth_inter
                    };
                    self.transform_tree(c, x0, y0, x0, y0, log2, 0, 0, false, false)?;
                }
            }
        }
        let qp = self.qp_y as i8;
        Self::fill4(&mut self.pic.qp, w4, x0, y0, n, n, pw, ph, qp);
        self.last_qp = self.qp_y;
        Ok(())
    }

    fn decode_part_mode(&mut self, c: &mut Cabac, log2: u32) -> PartMode {
        if self.bin(c, ci::PART_MODE) == 1 {
            return PartMode::P2Nx2N;
        }
        if self.cu_intra {
            return PartMode::PNxN;
        }
        if log2 == self.sps.log2_min_cb {
            if self.bin(c, ci::PART_MODE + 1) == 1 {
                return PartMode::P2NxN;
            }
            if log2 == 3 {
                return PartMode::PNx2N;
            }
            if self.bin(c, ci::PART_MODE + 2) == 1 {
                return PartMode::PNx2N;
            }
            return PartMode::PNxN;
        }
        if !self.sps.amp {
            return if self.bin(c, ci::PART_MODE + 1) == 1 {
                PartMode::P2NxN
            } else {
                PartMode::PNx2N
            };
        }
        if self.bin(c, ci::PART_MODE + 1) == 1 {
            if self.bin(c, ci::PART_MODE + 3) == 1 {
                PartMode::P2NxN
            } else if c.bypass() == 1 {
                PartMode::P2NxnD
            } else {
                PartMode::P2NxnU
            }
        } else if self.bin(c, ci::PART_MODE + 3) == 1 {
            PartMode::PNx2N
        } else if c.bypass() == 1 {
            PartMode::PnRx2N
        } else {
            PartMode::PnLx2N
        }
    }

    fn pcm_samples(&mut self, c: &mut Cabac, x0: usize, y0: usize, n: usize) -> Result<()> {
        let mut r = BitReader::new(c.data());
        r.set_pos(c.aligned_byte_pos() * 8);
        let (bdy, bdc) = (self.sps.bit_depth_y, self.sps.bit_depth_c);
        let (py, pc) = (self.sps.pcm_bit_depth_y, self.sps.pcm_bit_depth_c);
        for y in 0..n {
            for x in 0..n {
                let v = (r.u(py) << (bdy - py)) as u16;
                if x0 + x < self.pic.width && y0 + y < self.pic.height {
                    self.pic.planes[0].set(x0 + x, y0 + y, v);
                }
            }
        }
        for cidx in 1..3 {
            for y in 0..n / 2 {
                for x in 0..n / 2 {
                    let v = (r.u(pc) << (bdc - pc)) as u16;
                    let p = &mut self.pic.planes[cidx];
                    if x0 / 2 + x < p.width && y0 / 2 + y < p.height {
                        p.set(x0 / 2 + x, y0 / 2 + y, v);
                    }
                }
            }
        }
        if r.overrun() {
            return Err("PCM samples run past the slice".into());
        }
        c.restart(r.byte_pos());
        let f = flag::PCM | if self.sps.pcm_loop_filter_disabled { flag::NO_FILTER } else { 0 };
        self.set_flags(x0, y0, n, n, f, 0);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Intra modes (§8.4.2, §8.4.3)
    // ------------------------------------------------------------------

    fn intra_modes(&mut self, c: &mut Cabac, x0: usize, y0: usize, n: usize) -> Result<()> {
        let nparts = if self.intra_split { 4 } else { 1 };
        let pb = if self.intra_split { n / 2 } else { n };
        let mut prev = [false; 4];
        for p in prev.iter_mut().take(nparts) {
            *p = self.bin(c, ci::PREV_INTRA) == 1;
        }
        let (w4, pw, ph) = (self.pic.layout.w4, self.pic.width, self.pic.height);
        let mut first_mode = 0;
        for k in 0..nparts {
            let (xp, yp) = (x0 + (k % 2) * pb, y0 + (k / 2) * pb);
            let (mpm_idx, rem) = if prev[k] {
                let mut i = 0;
                while i < 2 && c.bypass() == 1 {
                    i += 1;
                }
                (Some(i), 0)
            } else {
                (None, c.bypass_bits(5))
            };
            let cand = self.mpm_candidates(xp, yp);
            let mode = match mpm_idx {
                Some(i) => cand[i],
                None => {
                    let mut s = cand;
                    s.sort_unstable();
                    let mut m = rem;
                    for v in s {
                        if m >= v {
                            m += 1;
                        }
                    }
                    m
                }
            };
            Self::fill4(&mut self.pic.ipm, w4, xp, yp, pb, pb, pw, ph, mode as u8);
            if k == 0 {
                first_mode = mode;
            }
        }
        let cm = if self.bin(c, ci::CHROMA_MODE) == 0 {
            4
        } else {
            c.bypass_bits(2)
        };
        self.ipm_c = if cm == 4 {
            first_mode
        } else {
            let m = [0, 26, 10, 1][cm as usize];
            if m == first_mode {
                34
            } else {
                m
            }
        };
        Ok(())
    }

    fn mpm_candidates(&self, xp: usize, yp: usize) -> [u32; 3] {
        let cand_of = |xn: isize, yn: isize, is_b: bool| -> u32 {
            if !self.avail(xp as isize, yp as isize, xn, yn) {
                return 1;
            }
            let i = self.pic.idx4(xn as usize, yn as usize);
            let f = self.pic.flags[i];
            if f & flag::INTRA == 0 || f & flag::PCM != 0 {
                return 1;
            }
            if is_b && (yn as usize) < ((yp >> self.sps.log2_ctb) << self.sps.log2_ctb) {
                return 1;
            }
            self.pic.ipm[i] as u32
        };
        let a = cand_of(xp as isize - 1, yp as isize, false);
        let b = cand_of(xp as isize, yp as isize - 1, true);
        if a == b {
            if a < 2 {
                [0, 1, 26]
            } else {
                [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
            }
        } else {
            let third = if a != 0 && b != 0 {
                0
            } else if a != 1 && b != 1 {
                1
            } else {
                26
            };
            [a, b, third]
        }
    }

    // ------------------------------------------------------------------
    // Prediction units (§7.3.8.6)
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn prediction_unit(
        &mut self,
        c: &mut Cabac,
        xcb: usize,
        ycb: usize,
        ncb: usize,
        xpb: usize,
        ypb: usize,
        w: usize,
        h: usize,
        part: usize,
        skip: bool,
    ) -> Result<()> {
        let merge = skip || self.bin(c, ci::MERGE_FLAG) == 1;
        let mvf = if merge {
            let mut idx = 0;
            if self.sh.max_merge_cand > 1 && self.bin(c, ci::MERGE_IDX) == 1 {
                idx = 1;
                while idx < self.sh.max_merge_cand as usize - 1 && c.bypass() == 1 {
                    idx += 1;
                }
            }
            if part == 0 && self.part_mode == PartMode::P2Nx2N {
                self.merge_2nx2n = true;
            }
            self.merge_motion(xcb, ycb, ncb, xpb, ypb, w, h, part, idx)
        } else {
            let pred = if self.sh.slice_type == SliceType::B {
                if w + h != 12 && self.bin(c, ci::INTER_PRED_IDC + self.cu_depth as usize) == 1 {
                    3
                } else if self.bin(c, ci::INTER_PRED_IDC + 4) == 1 {
                    2
                } else {
                    1
                }
            } else {
                1
            };
            let mut ref_idx = [-1i8; 2];
            let mut mvd = [[0i32; 2]; 2];
            let mut mvp_flag = [0usize; 2];
            for l in 0..2 {
                if pred & (1 << l) == 0 {
                    continue;
                }
                let num = self.sh.num_ref_idx[l] as usize;
                let mut r = 0;
                if num > 1 {
                    while r < num - 1 {
                        let b = if r < 2 {
                            self.bin(c, ci::REF_IDX + r)
                        } else {
                            c.bypass()
                        };
                        if b == 0 {
                            break;
                        }
                        r += 1;
                    }
                }
                ref_idx[l] = r as i8;
                if l == 1 && self.sh.mvd_l1_zero && pred == 3 {
                    mvd[1] = [0, 0];
                } else {
                    mvd[l] = self.mvd_coding(c)?;
                }
                mvp_flag[l] = self.bin(c, ci::MVP_FLAG) as usize;
            }
            let mut m = MvField::INTRA;
            for l in 0..2 {
                if ref_idx[l] < 0 {
                    continue;
                }
                if ref_idx[l] as usize >= self.refs[l].len() {
                    return Err(format!("ref_idx_l{l} {} past the list", ref_idx[l]));
                }
                let mvp = self.amvp(xcb, ycb, ncb, xpb, ypb, w, h, part, l, ref_idx[l], mvp_flag[l]);
                m.ref_idx[l] = ref_idx[l];
                m.mv[l] = [
                    (mvp[0] as i32 + mvd[l][0]) as i16,
                    (mvp[1] as i32 + mvd[l][1]) as i16,
                ];
            }
            m
        };
        let (w4, pw, ph) = (self.pic.layout.w4, self.pic.width, self.pic.height);
        Self::fill4(&mut self.pic.mv, w4, xpb, ypb, w, h, pw, ph, mvf);
        self.motion_compensate(xpb, ypb, w, h, &mvf)
    }

    fn mvd_coding(&mut self, c: &mut Cabac) -> Result<[i32; 2]> {
        let g0 = [self.bin(c, ci::MVD_GT0), self.bin(c, ci::MVD_GT0)];
        let mut g1 = [0; 2];
        for k in 0..2 {
            if g0[k] == 1 {
                g1[k] = self.bin(c, ci::MVD_GT1);
            }
        }
        let mut out = [0i32; 2];
        for k in 0..2 {
            if g0[k] == 0 {
                continue;
            }
            let mut abs = 1i32;
            if g1[k] == 1 {
                let mut kk = 1;
                let mut v = 0i32;
                while c.bypass() == 1 {
                    v += 1 << kk;
                    kk += 1;
                    if kk > 30 {
                        return Err("abs_mvd_minus2 too long".into());
                    }
                }
                v += c.bypass_bits(kk) as i32;
                abs = v + 2;
            }
            out[k] = if c.bypass() == 1 { -abs } else { abs };
        }
        Ok(out)
    }

    /// Inter prediction of one PU from its motion (§8.5.3.3).
    pub fn motion_compensate(&mut self, x: usize, y: usize, w: usize, h: usize, m: &MvField) -> Result<()> {
        let weights = self.sh.weights.as_ref();
        for cidx in 0..3 {
            let (bx, by, bw, bh) = if cidx == 0 { (x, y, w, h) } else { (x / 2, y / 2, w / 2, h / 2) };
            let bd = if cidx == 0 { self.sps.bit_depth_y } else { self.sps.bit_depth_c };
            let mut used = [false; 2];
            for l in 0..2 {
                if m.ref_idx[l] < 0 {
                    continue;
                }
                let e = self.refs[l]
                    .get(m.ref_idx[l] as usize)
                    .ok_or("reference index past the list")?;
                let p = &self.dpb[e.dpb].planes[cidx];
                let mv = m.mv[l];
                let (xi, yi, xf, yf) = if cidx == 0 {
                    (bx as isize + (mv[0] >> 2) as isize, by as isize + (mv[1] >> 2) as isize, (mv[0] & 3) as usize, (mv[1] & 3) as usize)
                } else {
                    (bx as isize + (mv[0] >> 3) as isize, by as isize + (mv[1] >> 3) as isize, (mv[0] & 7) as usize, (mv[1] & 7) as usize)
                };
                inter::interp(p, xi, yi, xf, yf, bw, bh, bd, cidx == 0, &mut self.mc[l], &mut self.mc_scratch);
                used[l] = true;
            }
            let wt = match weights {
                Some(wt) => {
                    let get = |l: usize| -> (i32, i32) {
                        if m.ref_idx[l] < 0 {
                            (0, 0)
                        } else {
                            wt.w[l][m.ref_idx[l] as usize][cidx]
                        }
                    };
                    let (w0, o0) = get(0);
                    let (w1, o1) = get(1);
                    let d = if cidx == 0 { wt.luma_log2_denom } else { wt.chroma_log2_denom };
                    Weighting::Explicit(w0, o0, w1, o1, d)
                }
                None => Weighting::Default,
            };
            let (m0, m1) = self.mc.split_at(1);
            inter::put(
                &mut self.pic.planes[cidx],
                bx,
                by,
                bw,
                bh,
                used[0].then_some(&m0[0][..]),
                used[1].then_some(&m1[0][..]),
                wt,
                bd,
            );
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Transform tree (§7.3.8.8)
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        c: &mut Cabac,
        x0: usize,
        y0: usize,
        xb: usize,
        yb: usize,
        log2: u32,
        depth: u32,
        blk: usize,
        parent_cb: bool,
        parent_cr: bool,
    ) -> Result<()> {
        let inter_split = self.sps.max_th_depth_inter == 0
            && !self.cu_intra
            && self.part_mode != PartMode::P2Nx2N
            && depth == 0;
        let split = if log2 <= self.sps.log2_max_tb
            && log2 > self.sps.log2_min_tb
            && depth < self.max_trafo_depth
            && !(self.intra_split && depth == 0)
        {
            self.bin(c, ci::SPLIT_TRANSFORM + 5 - log2 as usize) == 1
        } else {
            log2 > self.sps.log2_max_tb || (self.intra_split && depth == 0) || inter_split
        };
        let (mut cb, mut cr) = (false, false);
        if log2 > 2 {
            if depth == 0 || parent_cb {
                cb = self.bin(c, ci::CBF_CHROMA + depth as usize) == 1;
            }
            if depth == 0 || parent_cr {
                cr = self.bin(c, ci::CBF_CHROMA + depth as usize) == 1;
            }
        } else {
            cb = parent_cb;
            cr = parent_cr;
        }
        if split {
            let h = 1usize << (log2 - 1);
            for (i, (dx, dy)) in [(0, 0), (h, 0), (0, h), (h, h)].into_iter().enumerate() {
                self.transform_tree(c, x0 + dx, y0 + dy, x0, y0, log2 - 1, depth + 1, i, cb, cr)?;
            }
            return Ok(());
        }
        let cbf_luma = if self.cu_intra || depth != 0 || cb || cr {
            self.bin(c, ci::CBF_LUMA + (depth == 0) as usize) == 1
        } else {
            true
        };
        self.transform_unit(c, x0, y0, xb, yb, log2, blk, cbf_luma, cb, cr)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        c: &mut Cabac,
        x0: usize,
        y0: usize,
        xb: usize,
        yb: usize,
        log2: u32,
        blk: usize,
        cbf_luma: bool,
        cb: bool,
        cr: bool,
    ) -> Result<()> {
        let n = 1usize << log2;
        self.mark_edges(x0, y0, n, n, false);
        if (cbf_luma || cb || cr) && self.pps.cu_qp_delta && !self.cu_qp_delta_coded {
            let mut v = 0i32;
            if self.bin(c, ci::CU_QP_DELTA) == 1 {
                v = 1;
                while v < 5 && self.bin(c, ci::CU_QP_DELTA + 1) == 1 {
                    v += 1;
                }
                if v == 5 {
                    let mut k = 0;
                    let mut s = 0i32;
                    while c.bypass() == 1 {
                        s += 1 << k;
                        k += 1;
                        if k > 30 {
                            return Err("cu_qp_delta_abs too long".into());
                        }
                    }
                    s += c.bypass_bits(k) as i32;
                    v += s;
                }
                if c.bypass() == 1 {
                    v = -v;
                }
            }
            let off = 6 * (self.sps.bit_depth_y as i32 - 8);
            if v < -(26 + off / 2) || v > 25 + off / 2 {
                return Err("CuQpDeltaVal out of range".into());
            }
            self.cu_qp_delta_coded = true;
            self.cu_qp_delta = v;
            self.update_qp_y();
        }
        if self.cu_intra {
            let mode = self.pic.ipm[self.pic.idx4(x0, y0)] as u32;
            self.intra_predict(0, x0, y0, n, mode);
        }
        if cbf_luma {
            self.set_flags(x0, y0, n, n, flag::CBF, 0);
            self.residual(c, x0, y0, log2, 0)?;
        }
        if log2 > 2 {
            for (cidx, cbf) in [(1, cb), (2, cr)] {
                if self.cu_intra {
                    self.intra_predict(cidx, x0 / 2, y0 / 2, n / 2, self.ipm_c);
                }
                if cbf {
                    self.residual(c, x0 / 2, y0 / 2, log2 - 1, cidx)?;
                }
            }
        } else if blk == 3 {
            for (cidx, cbf) in [(1, cb), (2, cr)] {
                if self.cu_intra {
                    self.intra_predict(cidx, xb / 2, yb / 2, 4, self.ipm_c);
                }
                if cbf {
                    self.residual(c, xb / 2, yb / 2, 2, cidx)?;
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Intra prediction driver
    // ------------------------------------------------------------------

    /// Predict the `n × n` block at component location `(x, y)`.
    fn intra_predict(&mut self, cidx: usize, x: usize, y: usize, n: usize, mode: u32) {
        let s = if cidx == 0 { 1 } else { 2 };
        let (xl, yl) = ((x * s) as isize, (y * s) as isize);
        let bd = if cidx == 0 { self.sps.bit_depth_y } else { self.sps.bit_depth_c };
        let len = 4 * n + 1;
        let mut arr = [0i32; 129];
        let mut av = [false; 129];
        let p = &self.pic.planes[cidx];
        let constrained = self.pps.constrained_intra_pred;
        let check = |xn: isize, yn: isize| -> bool {
            // In component samples; availability is decided in luma.
            let (lx, ly) = (xn * s as isize, yn * s as isize);
            if !self.avail(xl, yl, lx, ly) {
                return false;
            }
            !constrained || self.pic.flags[self.pic.idx4(lx as usize, ly as usize)] & flag::INTRA != 0
        };
        let (xi, yi) = (x as isize, y as isize);
        let c2 = 2 * n;
        // Availability changes at most every 4 luma samples; test in steps.
        let step = 4 / s;
        let mut k = 0;
        while k < c2 {
            // Left column, y = k .. k + step.
            let a = check(xi - 1, yi + k as isize);
            for j in k..(k + step).min(c2) {
                let idx = c2 - 1 - j;
                av[idx] = a;
                if a {
                    arr[idx] = p.at(x - 1, y + j) as i32;
                }
            }
            let a = check(xi + k as isize, yi - 1);
            for j in k..(k + step).min(c2) {
                let idx = c2 + 1 + j;
                av[idx] = a;
                if a {
                    arr[idx] = p.at(x + j, y - 1) as i32;
                }
            }
            k += step;
        }
        if check(xi - 1, yi - 1) {
            av[c2] = true;
            arr[c2] = p.at(x - 1, y - 1) as i32;
        }
        intra::substitute(&mut arr[..len], &av[..len], bd);
        intra::filter(&mut arr[..len], n, mode, cidx == 0, self.sps.strong_intra_smoothing, bd);
        let mut out = [0i32; 32 * 32];
        intra::predict(&mut out, &arr[..len], n, mode, cidx == 0, bd);
        let p = &mut self.pic.planes[cidx];
        for yy in 0..n {
            for xx in 0..n {
                if x + xx < p.width && y + yy < p.height {
                    p.set(x + xx, y + yy, out[yy * n + xx] as u16);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Residual coding (§7.3.8.11) and reconstruction
    // ------------------------------------------------------------------

    fn residual(&mut self, c: &mut Cabac, x0: usize, y0: usize, log2: u32, cidx: usize) -> Result<()> {
        let n = 1usize << log2;
        let mut coeff = std::mem::take(&mut self.coeff);
        coeff[..n * n].fill(0);
        let ts = self.residual_coding(c, x0, y0, log2, cidx, &mut coeff)?;
        let bd = if cidx == 0 { self.sps.bit_depth_y } else { self.sps.bit_depth_c };
        if !self.cu_bypass {
            let qp = if cidx == 0 {
                self.qp_y + 6 * (self.sps.bit_depth_y as i32 - 8)
            } else {
                let off_c = 6 * (self.sps.bit_depth_c as i32 - 8);
                let o = if cidx == 1 {
                    self.pps.cb_qp_offset + self.sh.cb_qp_offset
                } else {
                    self.pps.cr_qp_offset + self.sh.cr_qp_offset
                };
                chroma_qp((self.qp_y + o).clamp(-off_c, 57)) + off_c
            };
            let m = match (&self.pps.scaling, &self.sps.scaling) {
                _ if ts && n > 4 => None,
                (Some(s), _) | (None, Some(s)) => {
                    Some(&s.m[log2 as usize - 2][cidx + if self.cu_intra { 0 } else { 3 }][..])
                }
                _ => None,
            };
            // `scaling_list_enabled_flag` gates PPS lists too.
            let m = if self.sps.scaling.is_some() { m } else { None };
            transform::dequant(&mut coeff, n, qp, bd, m);
            if ts {
                transform::transform_skip(&mut coeff, n, bd);
            } else {
                let dst = self.cu_intra && n == 4 && cidx == 0;
                transform::inverse(&mut coeff, n, dst, bd);
            }
        }
        let max = (1i32 << bd) - 1;
        let p = &mut self.pic.planes[cidx];
        for y in 0..n {
            if y0 + y >= p.height {
                break;
            }
            for x in 0..n {
                if x0 + x >= p.width {
                    break;
                }
                let i = (y0 + y) * p.width + x0 + x;
                p.data[i] = (p.data[i] as i32 + coeff[y * n + x]).clamp(0, max) as u16;
            }
        }
        self.coeff = coeff;
        Ok(())
    }

    /// Parse one `residual_coding()` into `out` (raster, `n × n`). Returns
    /// `transform_skip_flag`.
    fn residual_coding(&mut self, c: &mut Cabac, x0: usize, y0: usize, log2: u32, cidx: usize, out: &mut [i32]) -> Result<bool> {
        let n = 1usize << log2;
        let chroma = cidx > 0;
        let ts = self.pps.transform_skip
            && !self.cu_bypass
            && log2 == 2
            && self.bin(c, ci::TRANSFORM_SKIP + chroma as usize) == 1;
        // Last significant position.
        let (off, shift) = if !chroma {
            (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2)
        } else {
            (15, log2 - 2)
        };
        let max = (log2 << 1) - 1;
        let mut xp = 0;
        while xp < max && self.bin(c, ci::LAST_X + (off + (xp >> shift)) as usize) == 1 {
            xp += 1;
        }
        let mut yp = 0;
        while yp < max && self.bin(c, ci::LAST_Y + (off + (yp >> shift)) as usize) == 1 {
            yp += 1;
        }
        let suffix = |c: &mut Cabac, p: u32| -> u32 {
            if p > 3 {
                let nb = (p >> 1) - 1;
                (1 << nb) * (2 + (p & 1)) + c.bypass_bits(nb)
            } else {
                p
            }
        };
        let mut lx = suffix(c, xp) as usize;
        let mut ly = suffix(c, yp) as usize;
        if lx >= n || ly >= n {
            return Err("last significant coefficient outside the block".into());
        }
        // Scan order.
        let scan_idx = if self.cu_intra && (log2 == 2 || (log2 == 3 && cidx == 0)) {
            let m = if cidx == 0 {
                self.pic.ipm[self.pic.idx4(x0, y0)] as u32
            } else {
                self.ipm_c
            };
            if (6..=14).contains(&m) {
                2
            } else if (22..=30).contains(&m) {
                1
            } else {
                0
            }
        } else {
            0
        };
        if scan_idx == 2 {
            std::mem::swap(&mut lx, &mut ly);
        }
        let sbw = 1usize << (log2 - 2);
        let sb_scan: &[(u8, u8)] = match log2 {
            2 => &SCAN_SB1[scan_idx],
            3 => &SCAN_SB2[scan_idx],
            4 => &SCAN_SB4,
            _ => &SCAN_SB8,
        };
        let scan = &SCAN4[scan_idx];
        let last_sb = sb_scan
            .iter()
            .position(|&(x, y)| x as usize == lx >> 2 && y as usize == ly >> 2)
            .ok_or("last sub-block not in scan")?;
        let last_pos = scan
            .iter()
            .position(|&(x, y)| x as usize == lx & 3 && y as usize == ly & 3)
            .unwrap();
        let mut csbf = [0u8; 64];
        let mut prev_c1 = 1u32;
        let sdh = self.pps.sign_data_hiding && !self.cu_bypass;
        for i in (0..=last_sb).rev() {
            let (xs, ys) = (sb_scan[i].0 as usize, sb_scan[i].1 as usize);
            let right = if xs + 1 < sbw { csbf[ys * 8 + xs + 1] } else { 0 };
            let below = if ys + 1 < sbw { csbf[(ys + 1) * 8 + xs] } else { 0 };
            let mut infer_dc = false;
            let coded = if i < last_sb && i > 0 {
                let inc = (right | below).min(1) as usize + if chroma { 2 } else { 0 };
                infer_dc = true;
                self.bin(c, ci::SIG_GROUP + inc) as u8
            } else {
                1
            };
            csbf[ys * 8 + xs] = coded;
            let mut sig = [0u8; 16];
            let mut nsig = 0;
            let start: isize = if i == last_sb {
                sig[0] = last_pos as u8;
                nsig = 1;
                last_pos as isize - 1
            } else {
                15
            };
            if coded == 1 {
                let prev_csbf = right | (below << 1);
                let mut k = start;
                while k >= 0 {
                    let (xp, yp) = (scan[k as usize].0 as usize, scan[k as usize].1 as usize);
                    let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                    if k > 0 || !infer_dc {
                        let sc = if log2 == 2 {
                            CTX_IDX_MAP[(yc << 2) + xc] as usize
                        } else if xc + yc == 0 {
                            0
                        } else {
                            let mut s = match prev_csbf {
                                0 => {
                                    if xp + yp == 0 {
                                        2
                                    } else if xp + yp < 3 {
                                        1
                                    } else {
                                        0
                                    }
                                }
                                1 => {
                                    if yp == 0 {
                                        2
                                    } else if yp == 1 {
                                        1
                                    } else {
                                        0
                                    }
                                }
                                2 => {
                                    if xp == 0 {
                                        2
                                    } else if xp == 1 {
                                        1
                                    } else {
                                        0
                                    }
                                }
                                _ => 2,
                            };
                            if !chroma {
                                if xs + ys > 0 {
                                    s += 3;
                                }
                                s += if log2 == 3 {
                                    if scan_idx == 0 {
                                        9
                                    } else {
                                        15
                                    }
                                } else {
                                    21
                                };
                            } else {
                                s += if log2 == 3 { 9 } else { 12 };
                            }
                            s
                        };
                        let inc = if chroma { 27 + sc } else { sc };
                        if self.bin(c, ci::SIG + inc) == 1 {
                            sig[nsig] = k as u8;
                            nsig += 1;
                            infer_dc = false;
                        }
                    } else {
                        // DC of a coded sub-block nothing else in was coded.
                        sig[nsig] = 0;
                        nsig += 1;
                    }
                    k -= 1;
                }
            }
            if nsig == 0 {
                continue;
            }
            // Levels.
            let mut ctx_set = if i == 0 || chroma { 0 } else { 2 };
            if prev_c1 == 0 {
                ctx_set += 1;
            }
            let mut c1 = 1u32;
            let mut g1 = [0u8; 16];
            let mut first_g1: Option<usize> = None;
            for k in 0..nsig.min(8) {
                let inc = ctx_set * 4 + c1 as usize + if chroma { 16 } else { 0 };
                let f = self.bin(c, ci::GT1 + inc);
                g1[k] = f as u8;
                if f == 1 {
                    c1 = 0;
                    if first_g1.is_none() {
                        first_g1 = Some(k);
                    }
                } else if c1 > 0 && c1 < 3 {
                    c1 += 1;
                }
            }
            prev_c1 = c1;
            let mut g2 = 0;
            if first_g1.is_some() {
                g2 = self.bin(c, ci::GT2 + ctx_set + if chroma { 4 } else { 0 });
            }
            let hidden = sdh && (sig[0] as i32 - sig[nsig - 1] as i32) > 3;
            let nsign = if hidden { nsig - 1 } else { nsig };
            let signs = c.bypass_bits(nsign as u32) << (32 - nsign as u32).min(31);
            let signs = if nsign == 0 { 0 } else { signs };
            let mut rice = 0u32;
            let mut sum_abs = 0i32;
            for k in 0..nsig {
                let base = 1 + g1[k] as i32 + if Some(k) == first_g1 { g2 as i32 } else { 0 };
                let thresh = if k < 8 {
                    if Some(k) == first_g1 {
                        3
                    } else {
                        2
                    }
                } else {
                    1
                };
                let mut abs = base;
                if base == thresh {
                    let mut prefix = 0u32;
                    while prefix < 32 && c.bypass() == 1 {
                        prefix += 1;
                    }
                    let rem = if prefix <= 3 {
                        ((prefix << rice) + c.bypass_bits(rice)) as i32
                    } else {
                        let nb = prefix - 3 + rice;
                        if nb > 31 {
                            return Err("coeff_abs_level_remaining too long".into());
                        }
                        ((((1u32 << (prefix - 3)) + 2) << rice) + c.bypass_bits(nb)) as i32
                    };
                    abs = base + rem;
                    if abs > 3 * (1 << rice) {
                        rice = (rice + 1).min(4);
                    }
                }
                sum_abs += abs;
                let neg = if k < nsign {
                    (signs >> (31 - k as u32)) & 1 == 1
                } else {
                    // Hidden sign: parity of the sub-block's sum.
                    sum_abs & 1 == 1
                };
                let pos = sig[k] as usize;
                let (xc, yc) = (xs * 4 + scan[pos].0 as usize, ys * 4 + scan[pos].1 as usize);
                out[yc * n + xc] = if neg { -abs } else { abs };
            }
        }
        Ok(ts)
    }
}
