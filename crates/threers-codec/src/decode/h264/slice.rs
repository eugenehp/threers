//! Slice data decoding: the macroblock loop, neighbour access, motion vector
//! derivation (§8.4.1), and reconstruction.
//!
//! Parsing lives in `parse_cavlc` / `parse_cabac` as further `impl SliceDec`
//! blocks; both fill [`MbSyn`] and the current [`MbInfo`] and hand over to
//! [`SliceDec::decode_mb`].

use super::cavlc::CavlcTables;
use super::deblock;
use super::inter;
use super::intra;
use super::params::{Pps, SliceHeader, SliceType, Sps};
use super::tables::{chroma_qp, norm_adjust4, norm_adjust8, BLK_XY, RASTER_TO_BLK};
use super::types::*;
use crate::decode::bits::BitReader;
use crate::decode::cabac::{Cabac, Ctx};
use crate::decode::picture::Plane;

pub type Result<T> = std::result::Result<T, String>;

/// Deblocking parameters of one slice.
#[derive(Clone, Copy, Debug)]
pub struct SliceDeblock {
    pub disable: u32,
    pub alpha: i32,
    pub beta: i32,
}

/// The picture being decoded.
pub struct CurPic {
    pub planes: [Plane; 3],
    pub mbs: Vec<MbInfo>,
    pub w_mbs: usize,
    pub h_mbs: usize,
    pub poc: i32,
    /// `TopFieldOrderCnt` / `BottomFieldOrderCnt` (MBAFF field macroblocks).
    pub field_poc: [i32; 2],
    pub slices: Vec<SliceDeblock>,
    /// Macroblocks decoded so far, to tell a complete picture from one with
    /// slices missing.
    pub decoded: usize,
}

/// Neighbour motion as §8.4.1.3.2 sees it: `None` when the partition is not
/// available, otherwise its reference index (`-1` when it does not use the
/// list) and vector.
type NbMotion = Option<(i8, [i16; 2])>;

pub struct SliceDec<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub hdr: &'a SliceHeader,
    pub slice_num: u32,
    pub pic: &'a mut CurPic,
    pub dpb: &'a [RefPic],
    pub refs: &'a [Vec<RefEntry>; 2],
    /// MBAFF: the field lists, `[parity][list]` (see `Decoder::field_lists`).
    pub frefs: &'a [[Vec<RefEntry>; 2]; 2],
    pub tables: &'a CavlcTables,
    pub mb_x: usize,
    /// Macroblock row in the picture (MBAFF: `2 * pair row + bottom`).
    pub mb_y: usize,
    /// Raster index of the macroblock in `pic.mbs`.
    pub addr: usize,
    /// `MbaffFrameFlag`.
    pub mbaff: bool,
    /// The current macroblock is a field macroblock of an MBAFF frame.
    pub mb_field: bool,
    /// Macroblock row in the current macroblock's own sample view: `mb_y`,
    /// or the pair row for an MBAFF field macroblock.
    pub vmb_y: usize,
    /// How that view's rows map onto picture rows.
    pub geo: Geo,
    /// `QPY` of the previous macroblock in decoding order.
    pub qp: i32,
    /// Whether the previous macroblock had a non-zero `mb_qp_delta`.
    pub prev_qp_nz: bool,
    pub ctx: Vec<Ctx>,
    /// `LevelScale4x4[list][qP % 6][raster]`.
    ls4: Box<[[[i32; 16]; 6]; 6]>,
    /// `LevelScale8x8[intra/inter][qP % 6][raster]`.
    ls8: Box<[[[i32; 64]; 6]; 2]>,
    pub bd_y: u32,
    pub bd_c: u32,
    pub chroma: bool,
    /// 4:2:2: chroma macroblocks are 8×16.
    pub c422: bool,
    pub syn: MbSyn,
    /// Edge-clamped reference window, reused by every block.
    mc_win: Vec<u16>,
    /// Prediction scratch for `mc_block`, one block per list; every sample a
    /// block reads is written first, so it is never cleared.
    mc_pred: Option<Box<[[i32; 256]; 2]>>,
    luma_scratch: Box<inter::LumaScratch>,
    /// Macroblock row `.0` after its vertical-edge deblocking, for
    /// [`deblock::row_after_vertical`]'s ffmpeg top-left quirk.
    tl_row: Option<(usize, [Plane; 3])>,
}

impl<'a> SliceDec<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sps: &'a Sps,
        pps: &'a Pps,
        hdr: &'a SliceHeader,
        slice_num: u32,
        pic: &'a mut CurPic,
        dpb: &'a [RefPic],
        refs: &'a [Vec<RefEntry>; 2],
        frefs: &'a [[Vec<RefEntry>; 2]; 2],
        tables: &'a CavlcTables,
    ) -> Self {
        let lists = pps
            .scaling
            .as_ref()
            .or(sps.scaling.as_ref())
            .cloned()
            .unwrap_or_else(super::params::ScalingLists::flat);
        let mut ls4 = Box::new([[[0i32; 16]; 6]; 6]);
        let mut ls8 = Box::new([[[0i32; 64]; 6]; 2]);
        for l in 0..6 {
            for m in 0..6 {
                for k in 0..16 {
                    ls4[l][m][k] = lists.m4[l][k] as i32 * norm_adjust4(m, k / 4, k % 4);
                }
            }
        }
        for l in 0..2 {
            for m in 0..6 {
                for k in 0..64 {
                    ls8[l][m][k] = lists.m8[l][k] as i32 * norm_adjust8(m, k / 8, k % 8);
                }
            }
        }
        SliceDec {
            sps,
            pps,
            hdr,
            slice_num,
            pic,
            dpb,
            refs,
            frefs,
            tables,
            mb_x: 0,
            mb_y: 0,
            addr: 0,
            mbaff: hdr.mbaff,
            mb_field: false,
            vmb_y: 0,
            geo: Geo { step: 1, par: 0 },
            qp: hdr.qp,
            prev_qp_nz: false,
            ctx: Vec::new(),
            ls4,
            ls8,
            bd_y: sps.bit_depth_luma,
            bd_c: sps.bit_depth_chroma,
            chroma: sps.chroma_format_idc != 0,
            c422: sps.chroma_format_idc == 2,
            syn: MbSyn::new(),
            mc_win: Vec::new(),
            mc_pred: None,
            luma_scratch: Box::default(),
            tl_row: None,
        }
    }

    /// Make macroblock `addr` (`CurrMbAddr`, in decoding order) current.
    /// An MBAFF frame decodes pair by pair, top macroblock first.
    fn set_addr(&mut self, addr: usize) {
        let w = self.pic.w_mbs;
        if self.mbaff {
            let pair = addr / 2;
            self.mb_x = pair % w;
            self.mb_y = 2 * (pair / w) + addr % 2;
        } else {
            self.mb_x = addr % w;
            self.mb_y = addr / w;
        }
        self.addr = self.mb_y * w + self.mb_x;
        self.set_field(self.mb_field);
    }

    /// Set the current macroblock's `mb_field_decoding_flag` (MBAFF).
    pub fn set_field(&mut self, field: bool) {
        self.mb_field = self.mbaff && field;
        if self.mb_field {
            self.vmb_y = self.mb_y / 2;
            self.geo = Geo { step: 2, par: self.mb_y % 2 };
        } else {
            self.vmb_y = self.mb_y;
            self.geo = Geo { step: 1, par: 0 };
        }
        self.pic.mbs[self.addr].field = self.mb_field;
    }

    fn begin_mb(&mut self, addr: usize) {
        self.set_addr(addr);
        self.pic.mbs[self.addr] = MbInfo {
            slice: self.slice_num,
            field: self.mb_field,
            ..Default::default()
        };
        self.syn.reset();
    }

    /// Decode the slice's macroblocks.
    pub fn decode(&mut self, rbsp: &[u8]) -> Result<()> {
        let total = self.pic.w_mbs * self.pic.h_mbs;
        let mut addr = self.hdr.first_mb as usize * if self.mbaff { 2 } else { 1 };
        if addr >= total {
            return Err("first_mb_in_slice past the picture".into());
        }
        let mut r = BitReader::new(rbsp);
        r.set_pos(self.hdr.data_bit_offset);
        let first = addr;
        if self.pps.cabac {
            r.byte_align();
            self.init_contexts();
            let mut c = Cabac::new(rbsp, r.pos() / 8);
            // MBAFF: the bottom macroblock's skip flag, read early when the
            // top one is skipped.
            let mut next_skip = false;
            let mut prev_skipped = false;
            loop {
                if addr >= total {
                    return Err("slice runs past the end of the picture".into());
                }
                self.begin_pair_mb(addr, first);
                let top = addr.is_multiple_of(2);
                let skip = if self.hdr.slice_type == SliceType::I {
                    false
                } else if self.mbaff && !top && prev_skipped {
                    next_skip
                } else {
                    self.cabac_skip_flag(&mut c)
                };
                if self.mbaff && top {
                    if skip {
                        // A skipped top macroblock takes its pair's field flag
                        // from the bottom one, when that is coded.
                        self.pic.mbs[self.addr].kind = self.skip_kind();
                        let me = addr;
                        self.set_addr(addr + 1);
                        next_skip = self.cabac_skip_flag(&mut c);
                        self.set_addr(me);
                        if !next_skip {
                            let f = self.cabac_field_flag(&mut c);
                            self.set_field(f);
                        }
                    } else {
                        let f = self.cabac_field_flag(&mut c);
                        self.set_field(f);
                    }
                }
                prev_skipped = skip;
                if skip {
                    self.decode_skip()?;
                } else {
                    self.parse_mb_cabac(&mut c)?;
                    self.decode_mb()?;
                }
                self.pic.decoded += 1;
                addr += 1;
                if !(self.mbaff && top) && c.terminate() == 1 {
                    break;
                }
                if c.overrun() {
                    return Err("CABAC slice data overrun".into());
                }
            }
        } else {
            let mut run_left = 0usize;
            let mut have_run = false;
            loop {
                if self.hdr.slice_type != SliceType::I && !have_run {
                    run_left = r.ue() as usize;
                    if run_left > total {
                        return Err("mb_skip_run past the picture".into());
                    }
                    have_run = true;
                }
                if addr >= total {
                    if run_left > 0 {
                        return Err("skip run past the end of the picture".into());
                    }
                    return Err("slice runs past the end of the picture".into());
                }
                self.begin_pair_mb(addr, first);
                let top = addr.is_multiple_of(2);
                if run_left > 0 {
                    run_left -= 1;
                    if self.mbaff && top && run_left == 0 {
                        // The bottom macroblock is coded: the pair's flag
                        // comes before the skipped top one.
                        let f = r.flag();
                        self.set_field(f);
                    }
                    self.decode_skip()?;
                    self.pic.decoded += 1;
                    addr += 1;
                    if run_left == 0 && !r.more_rbsp_data() {
                        break;
                    }
                    continue;
                }
                have_run = false;
                if self.mbaff && top {
                    let f = r.flag();
                    self.set_field(f);
                }
                self.parse_mb_cavlc(&mut r)?;
                self.decode_mb()?;
                self.pic.decoded += 1;
                addr += 1;
                if r.overrun() {
                    return Err("CAVLC slice data overrun".into());
                }
                if !r.more_rbsp_data() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Reference list `l` as the current macroblock indexes it: an MBAFF
    /// field macroblock's is the field list of its parity.
    #[inline]
    pub fn rl(&self, l: usize) -> &'a [RefEntry] {
        if self.mb_field {
            &self.frefs[self.mb_y % 2][l]
        } else {
            &self.refs[l]
        }
    }

    /// `num_ref_idx_lX_active` for the current macroblock (doubled for an
    /// MBAFF field macroblock).
    #[inline]
    pub fn num_ref(&self, l: usize) -> u32 {
        self.hdr.num_ref_idx[l] << self.mb_field as u32
    }

    /// Field scans and CABAC field contexts: a field picture or an MBAFF
    /// field macroblock.
    #[inline]
    pub fn field_coded(&self) -> bool {
        self.hdr.field_pic || self.mb_field
    }

    fn skip_kind(&self) -> MbKind {
        if self.hdr.slice_type == SliceType::P {
            MbKind::PSkip
        } else {
            MbKind::BSkip
        }
    }

    /// [`Self::begin_mb`], with MBAFF's inferred field flag: a pair keeps
    /// the previous pair's flag, except that a new row of pairs takes the
    /// pair above's (ffmpeg `predict_field_decoding_flag`); the slice's first
    /// pair starts as frame.
    fn begin_pair_mb(&mut self, addr: usize, first: usize) {
        if self.mbaff && addr.is_multiple_of(2) {
            let w = self.pic.w_mbs;
            if addr == first {
                self.mb_field = false;
            } else if (addr / 2).is_multiple_of(w) {
                // The bottom macroblock of the pair above.
                let above = (2 * (addr / 2 / w) - 1) * w;
                let m = &self.pic.mbs[above];
                self.mb_field = m.slice == self.slice_num && m.field;
            }
        }
        self.begin_mb(addr);
    }

    // ------------------------------------------------------------------
    // Neighbours
    // ------------------------------------------------------------------

    /// Address of the macroblock at offset `(dx, dy)`, if it is available:
    /// inside the picture, already decoded, and in this slice. In an MBAFF
    /// frame, the macroblock covering luma location `(-1, 0)`, `(0, -1)`,
    /// `(16, -1)` or `(-1, -1)` (§6.4.11.1).
    #[inline]
    pub fn mb_nb(&self, dx: isize, dy: isize) -> Option<usize> {
        if self.mbaff {
            let xn = [-1, 0, 16][(dx + 1) as usize];
            return self.loc_nb(xn, dy.min(0), 16, 16).map(|(a, _, _)| a);
        }
        let x = self.mb_x as isize + dx;
        let y = self.mb_y as isize + dy;
        if x < 0 || y < 0 || x >= self.pic.w_mbs as isize || y >= self.pic.h_mbs as isize {
            return None;
        }
        let a = y as usize * self.pic.w_mbs + x as usize;
        if a == self.addr || self.pic.mbs[a].slice != self.slice_num {
            return None;
        }
        Some(a)
    }

    /// MBAFF neighbouring location (§6.4.12.2): the available macroblock
    /// covering location `(xn, yn)` of the current macroblock's view, in a
    /// component whose macroblocks are `maxw × maxh`, and the location within
    /// it. A location maps onto the picture row it shows (every other row in
    /// a field macroblock), and from there into whichever macroblock of the
    /// pair holds that row: this is Table 6-4, and ffmpeg's
    /// `left_block_options` / `top_xy` / `topleft_xy` adjustments.
    pub fn loc_nb(&self, xn: isize, yn: isize, maxw: isize, maxh: isize) -> Option<(usize, usize, usize)> {
        let w = self.pic.w_mbs as isize;
        let (px, pr, bot) = (self.mb_x as isize, (self.mb_y / 2) as isize, (self.mb_y % 2) as isize);
        let y = if self.mb_field { 2 * yn + bot } else { yn + maxh * bot };
        let (dxp, xw) = (xn.div_euclid(maxw), xn.rem_euclid(maxw));
        let (dyp, yp) = (y.div_euclid(2 * maxh), y.rem_euclid(2 * maxh));
        let (tx, tr) = (px + dxp, pr + dyp);
        if tx < 0 || tr < 0 || tx >= w || (dyp == 0 && dxp > 0) {
            return None;
        }
        let field = if dxp == 0 && dyp == 0 {
            self.mb_field
        } else {
            self.pic.mbs[(2 * tr * w + tx) as usize].field
        };
        let (mbi, yw) = if field { (yp & 1, yp >> 1) } else { (yp / maxh, yp % maxh) };
        let a = ((2 * tr + mbi) * w + tx) as usize;
        if a != self.addr && self.pic.mbs[a].slice != self.slice_num {
            return None;
        }
        Some((a, xw as usize, yw as usize))
    }

    /// The 4×4 block at `(x, y)` (4-sample units relative to the current
    /// macroblock, `x` in `-1..=4`, `y` in `-1..=3`): `(mb address, raster
    /// index)`. The current macroblock is returned too; callers that care
    /// whether its block is decoded yet check that themselves.
    #[inline]
    pub fn blk_nb(&self, x: isize, y: isize) -> Option<(usize, usize)> {
        self.blk_nb_row(x, y, false)
    }

    /// [`Self::blk_nb`], locating the block by its last sample row instead
    /// of its first: neighbour D of a partition is luma location
    /// `(x - 1, y - 1)`, the bottom row of the block above-left. Only an
    /// MBAFF left pair of the other kind tells the two apart.
    pub fn blk_nb_row(&self, x: isize, y: isize, last_row: bool) -> Option<(usize, usize)> {
        if (0..4).contains(&x) && (0..4).contains(&y) {
            return Some((self.addr, (y * 4 + x) as usize));
        }
        if self.mbaff {
            let xn = if x < 0 { -1 } else { 4 * x };
            let yn = if y < 0 { -1 } else { 4 * y + if last_row { 3 } else { 0 } };
            return self.loc_nb(xn, yn, 16, 16).map(|(a, xw, yw)| (a, (yw / 4) * 4 + xw / 4));
        }
        let (dx, bx) = if x < 0 {
            (-1, x + 4)
        } else if x >= 4 {
            (1, x - 4)
        } else {
            (0, x)
        };
        let (dy, by) = if y < 0 { (-1, y + 4) } else { (0, y) };
        let raster = (by * 4 + bx) as usize;
        if dx == 1 && dy == 0 {
            return None;
        }
        self.mb_nb(dx, dy).map(|a| (a, raster))
    }

    /// The chroma 4×4 block at `(x, y)` (`x` in `-1..2`, `y` in `-1..rows`):
    /// `(mb address, index into nnz_c)`.
    pub fn cblk_nb(&self, x: isize, y: isize) -> Option<(usize, usize)> {
        let rows = if self.c422 { 4 } else { 2 };
        if x >= 0 && y >= 0 {
            return Some((self.addr, (y * 2 + x) as usize));
        }
        if self.mbaff {
            let xn = if x < 0 { -1 } else { 4 * x };
            let yn = if y < 0 { -1 } else { 4 * y };
            return self.loc_nb(xn, yn, 8, 4 * rows).map(|(a, xw, yw)| (a, (yw / 4) * 2 + xw / 4));
        }
        let (dx, bx) = if x < 0 { (-1, x + 2) } else { (0, x) };
        let (dy, by) = if y < 0 { (-1, y + rows) } else { (0, y) };
        let a = self.mb_nb(dx, dy)?;
        Some((a, (by * 2 + bx) as usize))
    }

    /// Whether intra prediction may use the left samples of rows
    /// `y0..y0 + n` of the current macroblock's view (luma, or chroma). In an
    /// MBAFF frame those rows can come from both macroblocks of the left
    /// pair, which constrained intra prediction judges separately (ffmpeg
    /// `left_samples_available`).
    fn left_avail(&self, y0: usize, n: usize, chroma: bool) -> bool {
        if !self.mbaff {
            return self.intra_nb(-1, 0);
        }
        let (maxw, maxh) = if chroma { (8, if self.c422 { 16 } else { 8 }) } else { (16, 16) };
        (y0..y0 + n).all(|y| {
            self.loc_nb(-1, y as isize, maxw, maxh)
                .is_some_and(|(a, _, _)| !self.pps.constrained_intra_pred || self.pic.mbs[a].kind.is_intra())
        })
    }

    /// Whether the neighbouring macroblock may be used for intra prediction.
    #[inline]
    fn intra_nb(&self, dx: isize, dy: isize) -> bool {
        match self.mb_nb(dx, dy) {
            Some(a) => !self.pps.constrained_intra_pred || self.pic.mbs[a].kind.is_intra(),
            None => false,
        }
    }

    // ------------------------------------------------------------------
    // QP and scaling
    // ------------------------------------------------------------------

    fn update_qp(&mut self, delta: i32) -> Result<()> {
        let off = 6 * (self.bd_y as i32 - 8);
        if delta < -(26 + off / 2) || delta > 25 + off / 2 {
            return Err(format!("mb_qp_delta {delta} out of range"));
        }
        self.qp = ((self.qp + delta + 52 + 2 * off) % (52 + off)) - off;
        Ok(())
    }

    /// `QP'C` for component `c` (0 Cb, 1 Cr) at luma `QPY` `qpy`.
    fn qpc_prime(&self, qpy: i32, c: usize) -> i32 {
        let off_c = 6 * (self.bd_c as i32 - 8);
        let qpi = (qpy + self.pps.chroma_qp_offset[c]).clamp(-off_c, 51);
        chroma_qp(qpi) + off_c
    }

    /// `QPC` without the bit-depth offset — what deblocking averages.
    fn qpc(&self, qpy: i32, c: usize) -> i32 {
        self.qpc_prime(qpy, c) - 6 * (self.bd_c as i32 - 8)
    }

    // ------------------------------------------------------------------
    // Macroblock reconstruction
    // ------------------------------------------------------------------

    pub fn decode_skip(&mut self) -> Result<()> {
        let a = self.addr;
        self.pic.mbs[a].qp = self.qp as i8;
        self.pic.mbs[a].qpc = [self.qpc(self.qp, 0) as i8, self.qpc(self.qp, 1) as i8];
        self.prev_qp_nz = false;
        if self.hdr.slice_type == SliceType::P {
            self.pic.mbs[a].kind = MbKind::PSkip;
            self.p_skip_motion();
        } else {
            self.pic.mbs[a].kind = MbKind::BSkip;
            self.pic.mbs[a].b_direct16 = true;
            self.pic.mbs[a].direct8 = 0xF;
            self.direct_motion(0xF);
        }
        self.motion_compensate();
        Ok(())
    }

    pub fn decode_mb(&mut self) -> Result<()> {
        let a = self.addr;
        let kind = self.syn.kind;
        if kind == MbKind::IPcm {
            return self.decode_pcm();
        }
        let has_residual = self.syn.cbp != 0 || kind == MbKind::I16x16;
        if has_residual {
            self.update_qp(self.syn.qp_delta)?;
        }
        self.prev_qp_nz = has_residual && self.syn.qp_delta != 0;
        {
            let qp = self.qp;
            let qpc = [self.qpc(qp, 0) as i8, self.qpc(qp, 1) as i8];
            let m = &mut self.pic.mbs[a];
            m.kind = kind;
            m.qp = qp as i8;
            m.qpc = qpc;
            m.cbp = self.syn.cbp;
            m.t8x8 = self.syn.t8x8;
            m.chroma_mode = self.syn.chroma_mode;
        }
        self.set_nz_mask();
        let saved = if kind.is_intra() { self.patch_top_left() } else { None };
        if kind.is_intra() {
            self.intra_luma()?;
        } else {
            self.inter_motion()?;
            self.motion_compensate();
            self.inter_luma_residual();
        }
        if self.chroma && kind.is_intra() {
            self.intra_chroma_pred();
        }
        if let Some(saved) = saved {
            self.restore_top_left(saved);
        }
        if self.chroma {
            self.chroma_residual(kind.is_intra());
        }
        Ok(())
    }

    /// The top-left neighbour samples ffmpeg predicts from in a field
    /// picture with `disable_deblocking_filter_idc` 2: those after the row
    /// above's vertical-edge filtering (see [`deblock::row_after_vertical`]).
    /// Writes them into the picture and returns what they replaced.
    fn patch_top_left(&mut self) -> Option<[(usize, usize, u16); 3]> {
        let w = self.pic.w_mbs;
        if !self.hdr.field_pic
            || self.hdr.disable_deblocking != 2
            || self.mb_x == 0
            || self.mb_y == 0
            || self.pic.mbs[self.addr - w].slice != self.slice_num
        {
            return None;
        }
        let row = self.mb_y - 1;
        if self.tl_row.as_ref().is_none_or(|t| t.0 != row) {
            let cf = self.sps.chroma_format_idc;
            self.tl_row = Some((row, deblock::row_after_vertical(self.pic, row, self.bd_y, self.bd_c, cf)));
        }
        let filtered = &self.tl_row.as_ref().unwrap().1;
        let mut saved = [(usize::MAX, 0, 0); 3];
        let chh = if self.c422 { 16 } else { 8 };
        for c in 0..if self.chroma { 3 } else { 1 } {
            let (bw, bh) = if c == 0 { (16, 16) } else { (8, chh) };
            let fp = &filtered[c];
            let (x, h) = (self.mb_x * bw - 1, bh - 1);
            let v = fp.data[h * fp.width + x];
            let p = &mut self.pic.planes[c];
            let at = (self.mb_y * bh - 1) * p.width + x;
            saved[c] = (c, at, p.data[at]);
            p.data[at] = v;
        }
        Some(saved)
    }

    fn restore_top_left(&mut self, saved: [(usize, usize, u16); 3]) {
        for (c, at, v) in saved {
            if c != usize::MAX {
                self.pic.planes[c].data[at] = v;
            }
        }
    }

    fn set_nz_mask(&mut self) {
        let m = &mut self.pic.mbs[self.addr];
        let mut mask = 0u16;
        for k in 0..16 {
            if m.nnz[k] != 0 {
                mask |= 1 << k;
            }
        }
        if m.t8x8 {
            for b8 in 0..4 {
                let bits: u16 = 0x33 << ((b8 % 2) * 2 + (b8 / 2) * 8);
                if mask & bits != 0 {
                    mask |= bits;
                }
            }
        }
        m.nz_mask = mask;
    }

    fn decode_pcm(&mut self) -> Result<()> {
        let g = self.geo;
        let (x0, y0) = (self.mb_x * 16, self.vmb_y * 16);
        for y in 0..16 {
            for x in 0..16 {
                self.pic.planes[0].set(x0 + x, g.row(y0 + y), self.syn.pcm[y * 16 + x]);
            }
        }
        if self.chroma {
            let (ch, cy0) = if self.c422 { (16, y0) } else { (8, y0 / 2) };
            for c in 0..2 {
                for y in 0..ch {
                    for x in 0..8 {
                        let v = self.syn.pcm[256 + c * 8 * ch + y * 8 + x];
                        self.pic.planes[1 + c].set(x0 / 2 + x, g.row(cy0 + y), v);
                    }
                }
            }
        }
        let qpc = [self.qpc(0, 0) as i8, self.qpc(0, 1) as i8];
        let m = &mut self.pic.mbs[self.addr];
        m.kind = MbKind::IPcm;
        m.qp = 0;
        m.qpc = qpc;
        m.cbp = 0x2F;
        m.nnz = [16; 16];
        m.nnz_c = [[16; 8]; 2];
        m.cbf_dc = 7;
        m.nz_mask = 0xFFFF;
        self.prev_qp_nz = false;
        Ok(())
    }

    /// Resolve `Intra4x4PredMode` / `Intra8x8PredMode` for the block whose
    /// top-left 4×4 is `(bx, by)` (§8.3.1.1).
    fn intra_mode(&self, bx: usize, by: usize, prev: bool, rem: u8) -> u8 {
        let cur = &self.pic.mbs[self.addr];
        let nb = |x: isize, y: isize| -> Option<u8> {
            let (a, k) = self.blk_nb(x, y)?;
            let m = if a == self.addr { cur } else { &self.pic.mbs[a] };
            if a != self.addr && self.pps.constrained_intra_pred && !m.kind.is_intra() {
                return None;
            }
            Some(if a == self.addr || m.kind.is_nxn() {
                m.ipred[k]
            } else {
                2
            })
        };
        let a = nb(bx as isize - 1, by as isize);
        let b = nb(bx as isize, by as isize - 1);
        let pred = match (a, b) {
            (Some(a), Some(b)) => a.min(b),
            _ => 2,
        };
        if prev {
            pred
        } else if rem < pred {
            rem
        } else {
            rem + 1
        }
    }

    /// Gather intra neighbours of the `n × n` luma block at `(x0, y0)` in the
    /// picture, whose top-left 4×4 is `(bx, by)` in the macroblock.
    /// Returns `(top, left, at, al, atl)`.
    fn gather_nxn(&self, bx: usize, by: usize, n: usize) -> ([i32; 17], [i32; 9], bool, bool, bool) {
        let g = self.geo;
        let n4 = n / 4;
        let p = &self.pic.planes[0];
        let x0 = self.mb_x * 16 + bx * 4;
        let y0 = self.vmb_y * 16 + by * 4;
        let al = bx > 0 || self.left_avail(by * 4, n, false);
        let at = by > 0 || self.intra_nb(0, -1);
        let atl = match (bx > 0, by > 0) {
            (true, true) => true,
            (false, true) => self.left_avail(by * 4 - 1, 1, false),
            (true, false) => self.intra_nb(0, -1),
            (false, false) => self.intra_nb(-1, -1),
        };
        let atr = if by == 0 {
            if bx + n4 < 4 {
                self.intra_nb(0, -1)
            } else {
                self.intra_nb(1, -1)
            }
        } else if bx + n4 >= 4 {
            false
        } else if n == 4 {
            RASTER_TO_BLK[(by - 1) * 4 + bx + 1] < RASTER_TO_BLK[by * 4 + bx]
        } else {
            // 8×8: only the bottom-left block's top-right (block 1) is decoded.
            bx == 0 && by == 2
        };
        let mut top = [0i32; 17];
        let mut left = [0i32; 9];
        if atl {
            top[0] = p.at(x0 - 1, g.row(y0 - 1)) as i32;
            left[0] = top[0];
        }
        if at {
            for i in 0..n {
                top[1 + i] = p.at(x0 + i, g.row(y0 - 1)) as i32;
            }
            for i in n..2 * n {
                top[1 + i] = if atr {
                    p.at(x0 + i, g.row(y0 - 1)) as i32
                } else {
                    top[n]
                };
            }
        }
        if al {
            for i in 0..n {
                left[1 + i] = p.at(x0 - 1, g.row(y0 + i)) as i32;
            }
        }
        (top, left, at, al, atl)
    }

    fn intra_luma(&mut self) -> Result<()> {
        let g = self.geo;
        let kind = self.syn.kind;
        let bd = self.bd_y;
        let qp = self.qp + 6 * (bd as i32 - 8);
        match kind {
            MbKind::I4x4 => {
                for blk in 0..16 {
                    let (bx, by) = BLK_XY[blk];
                    let mode = self.intra_mode(bx, by, self.syn.prev_flag[blk], self.syn.rem[blk]);
                    self.pic.mbs[self.addr].ipred[by * 4 + bx] = mode;
                    let (top, left, at, al, _) = self.gather_nxn(bx, by, 4);
                    let mut out = [0i32; 16];
                    intra::pred_nxn(&mut out, 4, mode, &top, &left, at, al, bd);
                    let (x0, y0) = (self.mb_x * 16 + bx * 4, self.vmb_y * 16 + by * 4);
                    let coded = self.syn.coded & (1 << blk) != 0;
                    let coef = self.syn.coef[blk];
                    self.put_4x4(x0, y0, &out, coded.then_some(&coef), qp, 0, None);
                }
            }
            MbKind::I8x8 => {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
                    let mode =
                        self.intra_mode(bx, by, self.syn.prev_flag[b8], self.syn.rem[b8]);
                    for k in [0, 1, 4, 5] {
                        self.pic.mbs[self.addr].ipred[by * 4 + bx + k] = mode;
                    }
                    let (mut top, mut left, at, al, atl) = self.gather_nxn(bx, by, 8);
                    intra::filter_8x8_refs(&mut top, &mut left, at, al, atl);
                    let mut out = [0i32; 64];
                    intra::pred_nxn(&mut out, 8, mode, &top, &left, at, al, bd);
                    let (x0, y0) = (self.mb_x * 16 + bx * 4, self.vmb_y * 16 + by * 4);
                    let coded = self.syn.coded8 & (1 << b8) != 0;
                    let coef = self.syn.coef8[b8];
                    self.put_8x8(x0, y0, &out, coded.then_some(&coef), qp, 0);
                }
            }
            _ => {
                // Intra 16×16.
                let al = self.left_avail(0, 16, false);
                let at = self.intra_nb(0, -1);
                let atl = self.intra_nb(-1, -1);
                let mode = self.syn.i16_mode;
                if (mode == 0 && !at) || (mode == 1 && !al) || (mode == 3 && !(at && al && atl)) {
                    return Err("Intra 16x16 mode uses unavailable neighbours".into());
                }
                let (x0, y0) = (self.mb_x * 16, self.vmb_y * 16);
                let p = &self.pic.planes[0];
                let mut top = [0i32; 17];
                let mut left = [0i32; 17];
                if atl {
                    top[0] = p.at(x0 - 1, g.row(y0 - 1)) as i32;
                    left[0] = top[0];
                }
                if at {
                    for i in 0..16 {
                        top[1 + i] = p.at(x0 + i, g.row(y0 - 1)) as i32;
                    }
                }
                if al {
                    for i in 0..16 {
                        left[1 + i] = p.at(x0 - 1, g.row(y0 + i)) as i32;
                    }
                }
                let mut out = [0i32; 256];
                intra::pred_16x16(&mut out, mode, &top, &left, at, al, bd);
                // Luma DC: inverse Hadamard then scale.
                let dc = self.luma_dc(qp);
                for blk in 0..16 {
                    let (bx, by) = BLK_XY[blk];
                    let mut pb = [0i32; 16];
                    for y in 0..4 {
                        for x in 0..4 {
                            pb[y * 4 + x] = out[(by * 4 + y) * 16 + bx * 4 + x];
                        }
                    }
                    let coef = self.syn.coef[blk];
                    let d = dc[by * 4 + bx];
                    let coded = self.syn.coded & (1 << blk) != 0 || d != 0;
                    self.put_4x4(
                        x0 + bx * 4,
                        y0 + by * 4,
                        &pb,
                        coded.then_some(&coef),
                        qp,
                        0,
                        Some(d),
                    );
                }
            }
        }
        Ok(())
    }

    /// Inverse Hadamard and scaling of the Intra 16×16 DC levels (§8.5.10).
    fn luma_dc(&self, qp: i32) -> [i32; 16] {
        let c = &self.syn.dc_luma;
        let mut t = [0i32; 16];
        for i in 0..4 {
            let (a, b, cc, d) = (c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]);
            t[i * 4] = a + b + cc + d;
            t[i * 4 + 1] = a + b - cc - d;
            t[i * 4 + 2] = a - b - cc + d;
            t[i * 4 + 3] = a - b + cc - d;
        }
        let mut f = [0i32; 16];
        for j in 0..4 {
            let (a, b, cc, d) = (t[j], t[4 + j], t[8 + j], t[12 + j]);
            f[j] = a + b + cc + d;
            f[4 + j] = a + b - cc - d;
            f[8 + j] = a - b - cc + d;
            f[12 + j] = a - b + cc - d;
        }
        let ls = self.ls4[0][(qp % 6) as usize][0];
        let mut out = [0i32; 16];
        for k in 0..16 {
            out[k] = if qp >= 36 {
                (f[k] * ls) << (qp / 6 - 6)
            } else {
                (f[k] * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
            };
        }
        out
    }

    /// Add a 4×4 residual (levels `coef`, raster) over prediction `pred` and
    /// store at `(x0, y0)` of `plane`. `dc` replaces coefficient 0 already
    /// scaled (Intra 16×16 and chroma).
    #[allow(clippy::too_many_arguments)]
    fn put_4x4(
        &mut self,
        x0: usize,
        y0: usize,
        pred: &[i32; 16],
        coef: Option<&[i32; 16]>,
        qp: i32,
        plane: usize,
        dc: Option<i32>,
    ) {
        let g = self.geo;
        let bd = if plane == 0 { self.bd_y } else { self.bd_c };
        let max = (1i32 << bd) - 1;
        let p = &mut self.pic.planes[plane];
        let Some(coef) = coef else {
            for y in 0..4 {
                for x in 0..4 {
                    p.set(x0 + x, g.row(y0 + y), pred[y * 4 + x] as u16);
                }
            }
            return;
        };
        let intra = self.syn.kind.is_intra();
        let list = plane + if intra { 0 } else { 3 };
        let ls = &self.ls4[list][(qp % 6) as usize];
        let mut d = [0i32; 16];
        for k in 0..16 {
            if coef[k] != 0 {
                d[k] = if qp >= 24 {
                    (coef[k] * ls[k]) << (qp / 6 - 4)
                } else {
                    (coef[k] * ls[k] + (1 << (3 - qp / 6))) >> (4 - qp / 6)
                };
            }
        }
        if let Some(dc) = dc {
            d[0] = dc;
        }
        let r = idct4(&d);
        for y in 0..4 {
            for x in 0..4 {
                let v = (pred[y * 4 + x] + r[y * 4 + x]).clamp(0, max);
                p.set(x0 + x, g.row(y0 + y), v as u16);
            }
        }
    }

    fn put_8x8(&mut self, x0: usize, y0: usize, pred: &[i32; 64], coef: Option<&[i32; 64]>, qp: i32, plane: usize) {
        let g = self.geo;
        let max = (1i32 << self.bd_y) - 1;
        let p = &mut self.pic.planes[plane];
        let Some(coef) = coef else {
            for y in 0..8 {
                for x in 0..8 {
                    p.set(x0 + x, g.row(y0 + y), pred[y * 8 + x] as u16);
                }
            }
            return;
        };
        let intra = self.syn.kind.is_intra();
        let ls = &self.ls8[if intra { 0 } else { 1 }][(qp % 6) as usize];
        let mut d = [0i32; 64];
        for k in 0..64 {
            if coef[k] != 0 {
                d[k] = if qp >= 36 {
                    (coef[k] * ls[k]) << (qp / 6 - 6)
                } else {
                    (coef[k] * ls[k] + (1 << (5 - qp / 6))) >> (6 - qp / 6)
                };
            }
        }
        let r = idct8(&d);
        for y in 0..8 {
            for x in 0..8 {
                let v = (pred[y * 8 + x] + r[y * 8 + x]).clamp(0, max);
                p.set(x0 + x, g.row(y0 + y), v as u16);
            }
        }
    }

    fn inter_luma_residual(&mut self) {
        let g = self.geo;
        let qp = self.qp + 6 * (self.bd_y as i32 - 8);
        let (mx, my) = (self.mb_x * 16, self.vmb_y * 16);
        if self.syn.t8x8 {
            for b8 in 0..4 {
                if self.syn.coded8 & (1 << b8) == 0 {
                    continue;
                }
                let (x0, y0) = (mx + (b8 % 2) * 8, my + (b8 / 2) * 8);
                let mut pred = [0i32; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        pred[y * 8 + x] = self.pic.planes[0].at(x0 + x, g.row(y0 + y)) as i32;
                    }
                }
                let coef = self.syn.coef8[b8];
                self.put_8x8(x0, y0, &pred, Some(&coef), qp, 0);
            }
        } else {
            for blk in 0..16 {
                if self.syn.coded & (1 << blk) == 0 {
                    continue;
                }
                let (bx, by) = BLK_XY[blk];
                let (x0, y0) = (mx + bx * 4, my + by * 4);
                let mut pred = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        pred[y * 4 + x] = self.pic.planes[0].at(x0 + x, g.row(y0 + y)) as i32;
                    }
                }
                let coef = self.syn.coef[blk];
                self.put_4x4(x0, y0, &pred, Some(&coef), qp, 0, None);
            }
        }
    }

    fn intra_chroma_pred(&mut self) {
        let g = self.geo;
        let ch = if self.c422 { 16 } else { 8 };
        // Left availability per 4-row chroma block (MBAFF pairs of the other
        // kind; ffmpeg's ALZHEIMER_DC modes are this for 4:2:0).
        let mut al_rows = [false; 4];
        for (r, a) in al_rows.iter_mut().enumerate().take(ch / 4) {
            *a = self.left_avail(r * 4, 4, true);
        }
        let at = self.intra_nb(0, -1);
        let atl = self.intra_nb(-1, -1);
        let (x0, y0) = (self.mb_x * 8, self.vmb_y * ch);
        for c in 1..3 {
            let p = &self.pic.planes[c];
            let mut top = [0i32; 9];
            let mut left = [0i32; 17];
            if atl {
                top[0] = p.at(x0 - 1, g.row(y0 - 1)) as i32;
                left[0] = top[0];
            }
            if at {
                for i in 0..8 {
                    top[1 + i] = p.at(x0 + i, g.row(y0 - 1)) as i32;
                }
            }
            for i in 0..ch {
                if al_rows[i / 4] {
                    left[1 + i] = p.at(x0 - 1, g.row(y0 + i)) as i32;
                }
            }
            let mut out = [0i32; 128];
            intra::pred_chroma(&mut out, ch, self.syn.chroma_mode, &top, &left, at, &al_rows, self.bd_c);
            let p = &mut self.pic.planes[c];
            for y in 0..ch {
                for x in 0..8 {
                    p.set(x0 + x, g.row(y0 + y), out[y * 8 + x] as u16);
                }
            }
        }
    }

    fn chroma_residual(&mut self, intra: bool) {
        let g = self.geo;
        let cbp_c = self.syn.cbp >> 4;
        if cbp_c == 0 {
            return;
        }
        let ch = if self.c422 { 16 } else { 8 };
        let (x0, y0) = (self.mb_x * 8, self.vmb_y * ch);
        for c in 0..2 {
            let qp = self.qpc_prime(self.qp, c);
            let list = 1 + c + if intra { 0 } else { 3 };
            let cc = self.syn.dc_c[c];
            let mut dcs = [0i32; 8];
            if self.c422 {
                // §8.5.11.1-2 (4:2:2): c is 4×2 from the scan
                // c = [[c0, c2], [c1, c5], [c3, c6], [c4, c7]];
                // f = A4 · c · A2ᵀ, then QP'c + 3 dequantisation.
                let m = [[cc[0], cc[2]], [cc[1], cc[5]], [cc[3], cc[6]], [cc[4], cc[7]]];
                let t: Vec<[i32; 2]> = m.iter().map(|r| [r[0] + r[1], r[0] - r[1]]).collect();
                let qdc = qp + 3;
                let ls = self.ls4[list][(qdc % 6) as usize][0];
                for x in 0..2 {
                    let (z0, z1) = (t[0][x] + t[2][x], t[0][x] - t[2][x]);
                    let (z2, z3) = (t[1][x] - t[3][x], t[1][x] + t[3][x]);
                    let f = [z0 + z3, z1 + z2, z1 - z2, z0 - z3];
                    for (y, fv) in f.into_iter().enumerate() {
                        dcs[y * 2 + x] = if qdc >= 36 {
                            (fv * ls) << (qdc / 6 - 6)
                        } else {
                            (fv * ls + (1 << (5 - qdc / 6))) >> (6 - qdc / 6)
                        };
                    }
                }
            } else {
                let ls = self.ls4[list][(qp % 6) as usize][0];
                let f = [
                    cc[0] + cc[1] + cc[2] + cc[3],
                    cc[0] - cc[1] + cc[2] - cc[3],
                    cc[0] + cc[1] - cc[2] - cc[3],
                    cc[0] - cc[1] - cc[2] + cc[3],
                ];
                for blk in 0..4 {
                    dcs[blk] = ((f[blk] * ls) << (qp / 6)) >> 5;
                }
            }
            for blk in 0..if self.c422 { 8 } else { 4 } {
                let dc = dcs[blk];
                let coded = self.syn.coded_c[c] & (1 << blk) != 0;
                if dc == 0 && !coded {
                    continue;
                }
                let (bx, by) = (x0 + (blk % 2) * 4, y0 + (blk / 2) * 4);
                let mut pred = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        pred[y * 4 + x] = self.pic.planes[1 + c].at(bx + x, g.row(by + y)) as i32;
                    }
                }
                let coef = self.syn.ac_c[c][blk];
                self.put_4x4(bx, by, &pred, Some(&coef), qp, 1 + c, Some(dc));
            }
        }
    }

    // ------------------------------------------------------------------
    // Motion
    // ------------------------------------------------------------------

    /// Neighbour partition motion for list `l` at 4×4 position `(x, y)`
    /// relative to the macroblock. `done` marks the current macroblock's 4×4
    /// blocks whose motion is already known.
    fn nb_motion(&self, l: usize, x: isize, y: isize, done: u16) -> NbMotion {
        self.nb_motion_row(l, x, y, done, false)
    }

    /// [`Self::nb_motion`] for neighbour D (see [`Self::blk_nb_row`]).
    fn nb_motion_d(&self, l: usize, x: isize, y: isize, done: u16) -> NbMotion {
        self.nb_motion_row(l, x, y, done, true)
    }

    fn nb_motion_row(&self, l: usize, x: isize, y: isize, done: u16, last_row: bool) -> NbMotion {
        let (a, k) = self.blk_nb_row(x, y, last_row)?;
        let m = &self.pic.mbs[a];
        if a == self.addr {
            if done & (1 << k) == 0 {
                return None;
            }
        } else if m.kind.is_intra() {
            return Some((-1, [0, 0]));
        }
        let r = m.ref_idx[l][(k / 8) * 2 + (k % 4) / 2];
        if r < 0 {
            return Some((-1, [0, 0]));
        }
        let mv = m.mv[l][k];
        // MBAFF: a neighbour of the other kind counts in this one's units
        // (§8.4.1.3.1; ffmpeg MAP_F2F).
        if a != self.addr && self.mbaff && m.field != self.mb_field {
            return Some(if self.mb_field { (r * 2, [mv[0], mv[1] / 2]) } else { (r >> 1, [mv[0], mv[1] * 2]) });
        }
        Some((r, mv))
    }

    /// `mvpLX` (§8.4.1.3) for the partition at `(x, y)` of size `w × h` (4×4
    /// units). `dir` selects the 16×8 / 8×16 directional rule: 1/2 for the
    /// upper/lower 16×8, 3/4 for the left/right 8×16.
    #[allow(clippy::too_many_arguments)]
    fn mv_pred(&self, l: usize, r: i8, x: usize, y: usize, w: usize, dir: u8, done: u16) -> [i16; 2] {
        let (xi, yi) = (x as isize, y as isize);
        let a = self.nb_motion(l, xi - 1, yi, done);
        let b = self.nb_motion(l, xi, yi - 1, done);
        let c = self
            .nb_motion(l, xi + w as isize, yi - 1, done)
            .or_else(|| self.nb_motion_d(l, xi - 1, yi - 1, done));
        let un = (-1i8, [0i16, 0]);
        match dir {
            1 => {
                if let Some(b) = b.filter(|b| b.0 == r) {
                    return b.1;
                }
            }
            2 | 3 => {
                if let Some(a) = a.filter(|a| a.0 == r) {
                    return a.1;
                }
            }
            4 => {
                if let Some(c) = c.filter(|c| c.0 == r) {
                    return c.1;
                }
            }
            _ => {}
        }
        let (a, b, c) = match (a, b, c) {
            (Some(a), None, None) => (a, a, a),
            _ => (a.unwrap_or(un), b.unwrap_or(un), c.unwrap_or(un)),
        };
        let matches = (a.0 == r) as u8 + (b.0 == r) as u8 + (c.0 == r) as u8;
        if matches == 1 {
            if a.0 == r {
                return a.1;
            }
            if b.0 == r {
                return b.1;
            }
            return c.1;
        }
        let med = |p: i16, q: i16, s: i16| p.max(q).min(p.min(q).max(s));
        [med(a.1[0], b.1[0], c.1[0]), med(a.1[1], b.1[1], c.1[1])]
    }

    /// Store motion for list `l` over the 4×4 blocks of a partition.
    #[allow(clippy::too_many_arguments)]
    fn set_motion(&mut self, l: usize, x: usize, y: usize, w: usize, h: usize, r: i8, mv: [i16; 2]) {
        let id = if r >= 0 {
            self.rl(l).get(r as usize).map(|e| e.id as i32).unwrap_or(-1)
        } else {
            -1
        };
        let m = &mut self.pic.mbs[self.addr];
        for yy in y..y + h {
            for xx in x..x + w {
                m.mv[l][yy * 4 + xx] = if r >= 0 { mv } else { [0, 0] };
                let k8 = (yy / 2) * 2 + xx / 2;
                m.ref_idx[l][k8] = r;
                m.ref_id[l][k8] = id;
            }
        }
    }

    fn p_skip_motion(&mut self) {
        let a = self.nb_motion(0, -1, 0, 0);
        let b = self.nb_motion(0, 0, -1, 0);
        let zero = match (a, b) {
            (Some(a), Some(b)) => (a.0 == 0 && a.1 == [0, 0]) || (b.0 == 0 && b.1 == [0, 0]),
            _ => true,
        };
        let mv = if zero {
            [0, 0]
        } else {
            self.mv_pred(0, 0, 0, 0, 4, 0, 0)
        };
        self.set_motion(0, 0, 0, 4, 4, 0, mv);
        self.set_motion(1, 0, 0, 4, 4, -1, [0, 0]);
    }

    fn check_ref(&self, l: usize, r: i8) -> Result<()> {
        if r < 0 || r as usize >= self.rl(l).len() {
            return Err(format!("ref_idx_l{l} {r} past the list"));
        }
        Ok(())
    }

    fn inter_motion(&mut self) -> Result<()> {
        let is_b = self.syn.kind == MbKind::B;
        let t = self.syn.mb_type;
        if is_b && t == 0 {
            // B_Direct_16x16.
            let m = &mut self.pic.mbs[self.addr];
            m.b_direct16 = true;
            m.direct8 = 0xF;
            self.direct_motion(0xF);
            return Ok(());
        }
        let (shape, preds) = if is_b { b_mb_layout(t) } else { p_mb_layout(t) };
        let mut done = 0u16;
        if shape != Shape::S8x8 {
            for (i, &(x, y, w, h)) in shape.parts().iter().enumerate() {
                let dir = match (shape, i) {
                    (Shape::S16x8, 0) => 1,
                    (Shape::S16x8, _) => 2,
                    (Shape::S8x16, 0) => 3,
                    (Shape::S8x16, _) => 4,
                    _ => 0,
                };
                for l in 0..2 {
                    if preds[i] & (1 << l) != 0 {
                        let r = self.syn.ref_idx[l][i];
                        self.check_ref(l, r)?;
                        let p = self.mv_pred(l, r, x, y, w, dir, done);
                        let d = self.syn.mvd[l][y * 4 + x];
                        let mv = [
                            (p[0] as i32 + d[0]) as i16,
                            (p[1] as i32 + d[1]) as i16,
                        ];
                        self.set_motion(l, x, y, w, h, r, mv);
                    } else {
                        self.set_motion(l, x, y, w, h, -1, [0, 0]);
                    }
                }
                for yy in y..y + h {
                    for xx in x..x + w {
                        done |= 1 << (yy * 4 + xx);
                    }
                }
            }
            return Ok(());
        }
        // 8×8 sub-macroblocks, in order.
        let mut direct = 0u8;
        for s in 0..4 {
            if is_b && self.syn.sub[s] == 0 {
                direct |= 1 << s;
            }
        }
        self.pic.mbs[self.addr].direct8 = direct;
        for s in 0..4 {
            let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
            if direct & (1 << s) != 0 {
                self.direct_motion(1 << s);
            } else {
                let (pred, w, h) = if is_b {
                    b_sub_layout(self.syn.sub[s])
                } else {
                    p_sub_layout(self.syn.sub[s])
                };
                for l in 0..2 {
                    if pred & (1 << l) == 0 {
                        self.set_motion(l, sx, sy, 2, 2, -1, [0, 0]);
                    }
                }
                let mut sub_done = done;
                for py in (0..2).step_by(h) {
                    for px in (0..2).step_by(w) {
                        let (x, y) = (sx + px, sy + py);
                        for l in 0..2 {
                            if pred & (1 << l) == 0 {
                                continue;
                            }
                            let r = self.syn.ref_idx[l][s];
                            self.check_ref(l, r)?;
                            let p = self.mv_pred(l, r, x, y, w, 0, sub_done);
                            let d = self.syn.mvd[l][y * 4 + x];
                            let mv = [
                                (p[0] as i32 + d[0]) as i16,
                                (p[1] as i32 + d[1]) as i16,
                            ];
                            self.set_motion(l, x, y, w, h, r, mv);
                        }
                        for yy in y..y + h {
                            for xx in x..x + w {
                                sub_done |= 1 << (yy * 4 + xx);
                            }
                        }
                    }
                }
            }
            for yy in sy..sy + 2 {
                for xx in sx..sx + 2 {
                    done |= 1 << (yy * 4 + xx);
                }
            }
        }
        Ok(())
    }

    /// The colocated 4×4 block (§8.4.1.2.1) for raster block `k` of this
    /// macroblock: `(mvCol, refIdxCol, id of the picture it references,
    /// vertical scaling)`, or `None` when the colocated macroblock is intra.
    fn colocated(&self, k: usize) -> Option<([i16; 2], i8, i32, ColScale)> {
        let col = self.rl(1).first()?;
        let st = &self.dpb[col.dpb];
        let col_fields = st.field_mbs[0].is_some() || st.field_mbs[1].is_some();
        let w = self.pic.w_mbs;
        let (mx, my) = (self.mb_x, self.mb_y);
        let (x8, y8) = ((k % 4) / 2, (k / 4) / 2);
        // The corner block of the 8×8 the block is in.
        let corner = if self.sps.direct_8x8_inference { (y8 * 3) * 4 + x8 * 3 } else { k };
        // A frame macroblock over a field co-located picture or pair takes
        // the field closer in POC (bottom on a tie).
        let closer = || {
            let cur = self.pic.poc as i64;
            ((st.field_poc[0] as i64 - cur).abs() >= (st.field_poc[1] as i64 - cur).abs()) as usize
        };
        // (motion of the co-located MB, its 4×4 block, vertical scaling),
        // after ffmpeg h264_direct.c: a field macroblock of a frame pair
        // reads the pair's upper MB for its top 8×8 row and the lower one
        // (4×4 row 2) for the bottom; a frame macroblock of a field one
        // reads rows 2·(mb_y & 1) + y8 of the chosen field's macroblock.
        // (direct_8x8_inference is mandatory whenever the two mix.)
        let (m, k, scale): (&MbInfo, usize, ColScale) = if self.hdr.field_pic {
            if col_fields {
                (st.field_mbs[(col.structure == 2) as usize].as_ref()?.get(self.addr)?, corner, ColScale::Same)
            } else {
                // Field from a frame store: the MB at this field's rows; an
                // MBAFF field pair there is a field of the same parity.
                let par = self.hdr.bottom_field as usize;
                let m0 = st.mbs.get((2 * my + par) * w + mx)?;
                if m0.field {
                    (m0, corner, ColScale::Same)
                } else {
                    (st.mbs.get((2 * my + y8) * w + mx)?, (2 * y8) * 4 + x8 * 3, ColScale::FrameToField)
                }
            }
        } else if self.mb_field {
            // MBAFF field macroblock.
            let par = my % 2;
            if col_fields {
                (st.field_mbs[par].as_ref()?.get((my / 2) * w + mx)?, corner, ColScale::Same)
            } else {
                let m0 = st.mbs.get(self.addr)?;
                if m0.field {
                    (m0, corner, ColScale::Same)
                } else {
                    (st.mbs.get(((my & !1) + y8) * w + mx)?, (2 * y8) * 4 + x8 * 3, ColScale::FrameToField)
                }
            }
        } else {
            let row = 2 * (my & 1) + y8;
            if col_fields {
                (st.field_mbs[closer()].as_ref()?.get((my / 2) * w + mx)?, row * 4 + x8 * 3, ColScale::FieldToFrame)
            } else {
                let m0 = st.mbs.get(self.addr)?;
                if m0.field {
                    (st.mbs.get(((my & !1) + closer()) * w + mx)?, row * 4 + x8 * 3, ColScale::FieldToFrame)
                } else {
                    (m0, corner, ColScale::Same)
                }
            }
        };
        if m.kind.is_intra() || m.kind == MbKind::None {
            return None;
        }
        let k8 = (k / 8) * 2 + (k % 4) / 2;
        if m.ref_idx[0][k8] >= 0 {
            Some((m.mv[0][k], m.ref_idx[0][k8], m.ref_id[0][k8], scale))
        } else {
            Some((m.mv[1][k], m.ref_idx[1][k8], m.ref_id[1][k8], scale))
        }
    }

    /// Direct prediction (§8.4.1.2) for the 8×8 blocks in `mask`.
    fn direct_motion(&mut self, mask: u8) {
        if self.hdr.direct_spatial {
            self.direct_spatial(mask);
        } else {
            self.direct_temporal(mask);
        }
    }

    fn direct_spatial(&mut self, mask: u8) {
        // Reference indices and predictors from the whole macroblock's
        // neighbours.
        let mut refs = [-1i8; 2];
        let mut mvp = [[0i16; 2]; 2];
        for l in 0..2 {
            let a = self.nb_motion(l, -1, 0, 0);
            let b = self.nb_motion(l, 0, -1, 0);
            let c = self
                .nb_motion(l, 4, -1, 0)
                .or_else(|| self.nb_motion(l, -1, -1, 0));
            let min_pos = |x: i8, y: i8| if x >= 0 && y >= 0 { x.min(y) } else { x.max(y) };
            let ra = a.map_or(-1, |m| m.0);
            let rb = b.map_or(-1, |m| m.0);
            let rc = c.map_or(-1, |m| m.0);
            refs[l] = min_pos(ra, min_pos(rb, rc));
        }
        let zero_pred = refs[0] < 0 && refs[1] < 0;
        if zero_pred {
            refs = [0, 0];
        } else {
            for l in 0..2 {
                if refs[l] >= 0 {
                    mvp[l] = self.mv_pred(l, refs[l], 0, 0, 4, 0, 0);
                }
            }
        }
        let col_short = self.rl(1).first().is_some_and(|e| !e.long_term);
        for s in 0..4 {
            if mask & (1 << s) == 0 {
                continue;
            }
            let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
            for by in sy..sy + 2 {
                for bx in sx..sx + 2 {
                    let k = by * 4 + bx;
                    let col_zero = col_short
                        && !zero_pred
                        && match self.colocated(k) {
                            Some((mv, r, _, _)) => {
                                r == 0 && (-1..=1).contains(&mv[0]) && (-1..=1).contains(&mv[1])
                            }
                            None => false,
                        };
                    for l in 0..2 {
                        let r = refs[l];
                        let mv = if r < 0 || zero_pred || (r == 0 && col_zero) {
                            [0, 0]
                        } else {
                            mvp[l]
                        };
                        self.set_motion(l, bx, by, 1, 1, r, mv);
                    }
                }
            }
        }
    }

    fn direct_temporal(&mut self, mask: u8) {
        let cur_poc = self.cur_poc();
        for s in 0..4 {
            if mask & (1 << s) == 0 {
                continue;
            }
            let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
            for by in sy..sy + 2 {
                for bx in sx..sx + 2 {
                    let k = by * 4 + bx;
                    let (mv_col, ref0) = match self.colocated(k) {
                        None => ([0i16, 0], 0usize),
                        Some((mv, _, id, scale)) => {
                            // Map the co-located reference to this picture's
                            // structure: a field takes the same-parity field
                            // of a frame reference, a frame the whole frame.
                            let id = if !self.field_coded() {
                                id | 3
                            } else if id & 3 == 3 {
                                let par = if self.mb_field { self.mb_y % 2 } else { self.hdr.bottom_field as usize };
                                (id & !3) + 1 + par as i32
                            } else {
                                id
                            };
                            let r = self.rl(0).iter().position(|e| e.id as i32 == id).unwrap_or(0);
                            let my = match scale {
                                ColScale::Same => mv[1],
                                ColScale::FrameToField => mv[1] / 2,
                                ColScale::FieldToFrame => mv[1] * 2,
                            };
                            ([mv[0], my], r)
                        }
                    };
                    let (mv0, mv1) = match (self.rl(0).get(ref0), self.rl(1).first()) {
                        (Some(p0), Some(p1)) => {
                            let td = (p1.poc - p0.poc).clamp(-128, 127);
                            if p0.long_term || td == 0 {
                                (mv_col, [0, 0])
                            } else {
                                let tb = (cur_poc - p0.poc).clamp(-128, 127);
                                let tx = (16384 + (td / 2).abs()) / td;
                                let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                                let m0 = [
                                    ((dsf * mv_col[0] as i32 + 128) >> 8) as i16,
                                    ((dsf * mv_col[1] as i32 + 128) >> 8) as i16,
                                ];
                                (m0, [m0[0] - mv_col[0], m0[1] - mv_col[1]])
                            }
                        }
                        _ => ([0, 0], [0, 0]),
                    };
                    self.set_motion(0, bx, by, 1, 1, ref0 as i8, mv0);
                    self.set_motion(1, bx, by, 1, 1, 0, mv1);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Motion compensation
    // ------------------------------------------------------------------

    /// Predict the whole macroblock from the motion stored in its
    /// [`MbInfo`], in the largest blocks whose motion is uniform.
    fn motion_compensate(&mut self) {
        let m = self.pic.mbs[self.addr];
        let same = |a: usize, b: usize| {
            (0..2).all(|l| {
                m.mv[l][a] == m.mv[l][b]
                    && m.ref_idx[l][(a / 8) * 2 + (a % 4) / 2] == m.ref_idx[l][(b / 8) * 2 + (b % 4) / 2]
            })
        };
        if (1..16).all(|k| same(0, k)) {
            self.mc_block(&m, 0, 0, 4, 4);
            return;
        }
        for s in 0..4 {
            let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
            let k0 = sy * 4 + sx;
            if [1, 4, 5].iter().all(|&d| same(k0, k0 + d)) {
                self.mc_block(&m, sx, sy, 2, 2);
            } else {
                for d in [0, 1, 4, 5] {
                    let k = k0 + d;
                    self.mc_block(&m, k % 4, k / 4, 1, 1);
                }
            }
        }
    }

    /// Inter prediction of the `w × h` (4×4 units) block at `(bx, by)`.
    fn mc_block(&mut self, m: &MbInfo, bx: usize, by: usize, w: usize, h: usize) {
        let k = by * 4 + bx;
        let k8 = (by / 2) * 2 + bx / 2;
        let r = [m.ref_idx[0][k8], m.ref_idx[1][k8]];
        let (pw, ph) = (w * 4, h * 4);
        let (x0, y0) = (self.mb_x * 16 + bx * 4, self.vmb_y * 16 + by * 4);
        let mut pred = self.mc_pred.take().unwrap_or_else(|| Box::new([[0; 256]; 2]));
        let mut used = [false; 2];
        for l in 0..2 {
            if r[l] < 0 {
                continue;
            }
            let Some(e) = self.rl(l).get(r[l] as usize) else {
                continue;
            };
            used[l] = true;
            let planes = e.planes.clone();
            let refp = &planes[0];
            let mv = m.mv[l][k];
            inter::luma(
                refp,
                x0 as isize + (mv[0] >> 2) as isize,
                y0 as isize + (mv[1] >> 2) as isize,
                (mv[0] & 3) as usize,
                (mv[1] & 3) as usize,
                pw,
                ph,
                &mut pred[l],
                self.bd_y,
                &mut self.mc_win,
                &mut self.luma_scratch,
            );
        }
        self.weight_and_store(0, x0, y0, pw, ph, &pred, used, r);
        if !self.chroma {
            self.mc_pred = Some(pred);
            return;
        }
        let c422 = self.c422;
        let (cw, ch) = (pw / 2, if c422 { ph } else { ph / 2 });
        let (cx, cy) = (x0 / 2, if c422 { y0 } else { y0 / 2 });
        for c in 1..3 {
            let cpred = &mut pred;
            for l in 0..2 {
                if !used[l] {
                    continue;
                }
                let e = &self.rl(l)[r[l] as usize];
                let planes = e.planes.clone();
                let refp = &planes[c];
                let mut mv = m.mv[l][k];
                // Field prediction from the other parity: the chroma
                // vector shifts by a quarter of a chroma sample (ffmpeg
                // mc_dir_part: my += 2 * (cur parity - ref parity)).
                if (self.hdr.field_pic || self.mb_field) && e.structure != 3 {
                    let cur_par = if self.mb_field { (self.mb_y % 2) as i16 } else { self.hdr.bottom_field as i16 };
                    let ref_par = (e.structure == 2) as i16;
                    mv[1] += 2 * (cur_par - ref_par);
                }
                inter::chroma(
                    refp,
                    cx as isize + (mv[0] >> 3) as isize,
                    // 4:2:2: vertical chroma is luma-sized, quarter-sample
                    // vectors become eighths (§8.4.2.2.2).
                    cy as isize + (mv[1] >> if c422 { 2 } else { 3 }) as isize,
                    (mv[0] & 7) as i32,
                    if c422 { ((mv[1] & 3) << 1) as i32 } else { (mv[1] & 7) as i32 },
                    cw,
                    ch,
                    &mut cpred[l],
                    &mut self.mc_win,
                );
            }
            self.weight_and_store(c, cx, cy, cw, ch, cpred, used, r);
        }
        self.mc_pred = Some(pred);
    }

    /// Weighted sample prediction (§8.4.2.3) and store.
    #[allow(clippy::too_many_arguments)]
    fn weight_and_store(
        &mut self,
        c: usize,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        pred: &[[i32; 256]; 2],
        used: [bool; 2],
        r: [i8; 2],
    ) {
        let bd = if c == 0 { self.bd_y } else { self.bd_c };
        let max = (1i32 << bd) - 1;
        let n = w * h;
        let explicit = self.hdr.weights.as_ref();
        let implicit = self.hdr.slice_type == SliceType::B && self.pps.weighted_bipred_idc == 2;
        let (p0, p1) = (&pred[0][..n], &pred[1][..n]);
        match (used[0], used[1]) {
            (true, true) => {
                let wts = if let Some(wt) = explicit {
                    let (w0, o0) = wt.weights[0][self.wp_idx(r[0])][c];
                    let (w1, o1) = wt.weights[1][self.wp_idx(r[1])][c];
                    let log_wd = if c == 0 { wt.luma_log2_denom } else { wt.chroma_log2_denom } as i32;
                    let sh = bd as i32 - 8;
                    Some((w0, w1, (o0 << sh), (o1 << sh), log_wd))
                } else if implicit {
                    let (w0, w1) = self.implicit_weights(r[0] as usize, r[1] as usize);
                    Some((w0, w1, 0, 0, 5))
                } else {
                    None
                };
                match wts {
                    Some((w0, w1, o0, o1, lw)) => {
                        let o = (o0 + o1 + 1) >> 1;
                        self.store2(c, x0, y0, w, h, p0, p1, |a, b| {
                            (((a * w0 + b * w1 + (1 << lw)) >> (lw + 1)) + o).clamp(0, max)
                        });
                    }
                    None => self.store2(c, x0, y0, w, h, p0, p1, |a, b| (a + b + 1) >> 1),
                }
            }
            (true, false) | (false, true) => {
                let l = if used[0] { 0 } else { 1 };
                let pl = &pred[l][..n];
                match explicit {
                    Some(wt) => {
                        let (wl, ol) = wt.weights[l][self.wp_idx(r[l])][c];
                        let lw = if c == 0 { wt.luma_log2_denom } else { wt.chroma_log2_denom } as i32;
                        let o = ol << (bd as i32 - 8);
                        if lw >= 1 {
                            self.store1(c, x0, y0, w, h, pl, |a| (((a * wl + (1 << (lw - 1))) >> lw) + o).clamp(0, max));
                        } else {
                            self.store1(c, x0, y0, w, h, pl, |a| (a * wl + o).clamp(0, max));
                        }
                    }
                    None => self.store1(c, x0, y0, w, h, pl, |a| a),
                }
            }
            _ => {
                // No usable reference (broken stream): leave mid-grey.
                let grey = [1 << (bd - 1); 256];
                self.store1(c, x0, y0, w, h, &grey[..n], |a| a);
            }
        }
    }

    /// Write `f(src)` for each sample of `src` (`w × h`, row-major) to the
    /// block at `(x0, y0)` of plane `c`, a row at a time (`geo` maps a
    /// field's rows into the frame).
    #[inline(always)]
    fn store1(&mut self, c: usize, x0: usize, y0: usize, w: usize, h: usize, src: &[i32], f: impl Fn(i32) -> i32) {
        let g = self.geo;
        let p = &mut self.pic.planes[c];
        let stride = p.width;
        for (y, s) in src.chunks_exact(w).take(h).enumerate() {
            for (d, &a) in p.data[g.row(y0 + y) * stride + x0..][..w].iter_mut().zip(s) {
                *d = f(a) as u16;
            }
        }
    }

    /// [`store1`](Self::store1) over two predictions.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn store2(&mut self, c: usize, x0: usize, y0: usize, w: usize, h: usize, s0: &[i32], s1: &[i32], f: impl Fn(i32, i32) -> i32) {
        let g = self.geo;
        let p = &mut self.pic.planes[c];
        let stride = p.width;
        for (y, (a, b)) in s0.chunks_exact(w).zip(s1.chunks_exact(w)).take(h).enumerate() {
            for ((d, &a), &b) in p.data[g.row(y0 + y) * stride + x0..][..w].iter_mut().zip(a).zip(b) {
                *d = f(a, b) as u16;
            }
        }
    }

    /// The current macroblock's POC: its field's in an MBAFF field
    /// macroblock.
    fn cur_poc(&self) -> i32 {
        if self.mb_field {
            self.pic.field_poc[self.mb_y % 2]
        } else {
            self.pic.poc
        }
    }

    /// Explicit weight table index of reference `r` (§8.4.2.3:
    /// `refIdxL0WP`, halved in an MBAFF field macroblock).
    #[inline]
    fn wp_idx(&self, r: i8) -> usize {
        (r >> self.mb_field as i8) as usize
    }

    /// Implicit bi-prediction weights `(w0, w1)` (§8.4.2.3.1).
    fn implicit_weights(&self, r0: usize, r1: usize) -> (i32, i32) {
        let (Some(p0), Some(p1)) = (self.rl(0).get(r0), self.rl(1).get(r1)) else {
            return (32, 32);
        };
        let td = (p1.poc - p0.poc).clamp(-128, 127);
        if td == 0 || p0.long_term || p1.long_term {
            return (32, 32);
        }
        let tb = (self.cur_poc() - p0.poc).clamp(-128, 127);
        let tx = (16384 + (td / 2).abs()) / td;
        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
        if !(-64..=128).contains(&(dsf >> 2)) {
            return (32, 32);
        }
        (64 - (dsf >> 2), dsf >> 2)
    }

    // ------------------------------------------------------------------
    // CABAC context initialisation
    // ------------------------------------------------------------------

    fn init_contexts(&mut self) {
        let table = if self.hdr.slice_type == SliceType::I {
            &super::cabac_init::INIT_I
        } else {
            super::cabac_init::INIT_PB[self.hdr.cabac_init_idc as usize]
        };
        self.ctx = table
            .iter()
            .map(|&[m, n]| Ctx::h264(m, n, self.hdr.qp))
            .collect();
    }
}

/// 4×4 inverse transform (§8.5.12.2), rows then columns, `(x + 32) >> 6`.
pub fn idct4(d: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let r = &d[i * 4..i * 4 + 4];
        let e0 = r[0] + r[2];
        let e1 = r[0] - r[2];
        let e2 = (r[1] >> 1) - r[3];
        let e3 = r[1] + (r[3] >> 1);
        t[i * 4] = e0 + e3;
        t[i * 4 + 1] = e1 + e2;
        t[i * 4 + 2] = e1 - e2;
        t[i * 4 + 3] = e0 - e3;
    }
    let mut out = [0i32; 16];
    for j in 0..4 {
        let (f0, f1, f2, f3) = (t[j], t[4 + j], t[8 + j], t[12 + j]);
        let g0 = f0 + f2;
        let g1 = f0 - f2;
        let g2 = (f1 >> 1) - f3;
        let g3 = f1 + (f3 >> 1);
        out[j] = (g0 + g3 + 32) >> 6;
        out[4 + j] = (g1 + g2 + 32) >> 6;
        out[8 + j] = (g1 - g2 + 32) >> 6;
        out[12 + j] = (g0 - g3 + 32) >> 6;
    }
    out
}

fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let a0 = d[0] + d[4];
    let a4 = d[0] - d[4];
    let a2 = (d[2] >> 1) - d[6];
    let a6 = d[2] + (d[6] >> 1);
    let b0 = a0 + a6;
    let b2 = a4 + a2;
    let b4 = a4 - a2;
    let b6 = a0 - a6;
    let a1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let a3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let a5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let a7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let b1 = a1 + (a7 >> 2);
    let b7 = a7 - (a1 >> 2);
    let b3 = a3 + (a5 >> 2);
    let b5 = (a3 >> 2) - a5;
    [b0 + b7, b2 + b5, b4 + b3, b6 + b1, b6 - b1, b4 - b3, b2 - b5, b0 - b7]
}

/// 8×8 inverse transform (§8.5.13.2).
pub fn idct8(d: &[i32; 64]) -> [i32; 64] {
    let mut t = [0i32; 64];
    for i in 0..8 {
        let row = idct8_1d(d[i * 8..i * 8 + 8].try_into().unwrap());
        t[i * 8..i * 8 + 8].copy_from_slice(&row);
    }
    let mut out = [0i32; 64];
    for j in 0..8 {
        let col = idct8_1d(std::array::from_fn(|i| t[i * 8 + j]));
        for i in 0..8 {
            out[i * 8 + j] = (col[i] + 32) >> 6;
        }
    }
    out
}

/// How the rows of a macroblock's sample view map onto picture rows: every
/// row, or (MBAFF field macroblock) every other row starting at `par`.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub step: usize,
    pub par: usize,
}

impl Geo {
    #[inline(always)]
    pub fn row(self, y: usize) -> usize {
        y * self.step + self.par
    }
}

/// Vertical motion scaling between the co-located picture's structure and
/// the current one (§8.4.1.2.3 `vertMvScale`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColScale {
    Same,
    FrameToField,
    FieldToFrame,
}
