//! VP9 tile and block decoding (ffmpeg `vp9.c` decode_tiles/decode_sb and
//! `vp9block.c`): mode info, coefficient tokens, and loop-filter masks.

use super::data::*;
use super::probs::{Counts, ProbContext};
use super::rac::Rac;
use super::*;

/// `VP9Filter`: per-superblock loop-filter levels and edge masks.
#[derive(Clone)]
pub struct LfMask {
    pub level: [u8; 64],
    /// `[0 = y, 1 = uv][0 = col, 1 = row][8 rows][0=16, 1=8, 2=4, 3=inner4]`
    pub mask: [[[[u8; 4]; 8]; 2]; 2],
}

/// `VP9Block`.
#[derive(Clone, Copy, Default)]
pub struct Block {
    pub seg_id: u8,
    pub intra: bool,
    pub comp: bool,
    pub r#ref: [u8; 2],
    pub mode: [u8; 4],
    pub uvmode: u8,
    pub skip: bool,
    pub filter: u8,
    pub mv: [[Mv; 2]; 4],
    pub bs: u8,
    pub tx: u8,
    pub uvtx: u8,
    pub bl: u8,
    pub bp: u8,
}

/// Everything decoding one frame's tiles needs (ffmpeg's `VP9Context` and
/// `VP9TileData` for a single thread).
pub struct FrameDec<'a> {
    pub h: &'a Header,
    pub p: &'a ProbContext,
    pub coef: &'a CoefProbs,
    pub counts: &'a mut Counts,
    pub cur: &'a mut Frame,
    pub refs: &'a [Option<Arc<Frame>>; 3],
    pub segmap_ref: Option<&'a Frame>,
    pub mvpair: Option<&'a Frame>,
    pub mvscale: [[u16; 2]; 3],
    pub mvstep: [[u8; 2]; 3],
    pub cols: usize,
    pub rows: usize,
    pub sb_cols: usize,
    pub sb_rows: usize,
    pub ss_h: usize,
    pub ss_v: usize,
    pub bpp: u32,
    // Above contexts.
    pub above_partition_ctx: Vec<u8>,
    pub above_mode_ctx: Vec<u8>,
    pub above_y_nnz_ctx: Vec<u8>,
    pub above_uv_nnz_ctx: [Vec<u8>; 2],
    pub above_skip_ctx: Vec<u8>,
    pub above_txfm_ctx: Vec<u8>,
    pub above_segpred_ctx: Vec<u8>,
    pub above_intra_ctx: Vec<u8>,
    pub above_comp_ctx: Vec<u8>,
    pub above_ref_ctx: Vec<u8>,
    pub above_filter_ctx: Vec<u8>,
    pub above_mv_ctx: Vec<[Mv; 2]>,
    // Left contexts.
    pub left_y_nnz_ctx: [u8; 16],
    pub left_mode_ctx: [u8; 16],
    pub left_mv_ctx: [[Mv; 2]; 16],
    pub left_uv_nnz_ctx: [[u8; 16]; 2],
    pub left_partition_ctx: [u8; 8],
    pub left_skip_ctx: [u8; 8],
    pub left_txfm_ctx: [u8; 8],
    pub left_segpred_ctx: [u8; 8],
    pub left_intra_ctx: [u8; 8],
    pub left_comp_ctx: [u8; 8],
    pub left_ref_ctx: [u8; 8],
    pub left_filter_ctx: [u8; 8],
    // Block state.
    pub b: Block,
    pub row: usize,
    pub row7: usize,
    pub col: usize,
    pub col7: usize,
    pub tile_col_start: usize,
    pub min_mv: (i32, i32),
    pub max_mv: (i32, i32),
    pub block: Vec<i32>,
    pub uvblock: [Vec<i32>; 2],
    pub eob: Vec<u16>,
    pub uveob: [Vec<u16>; 2],
    /// Loop-filter masks, one per superblock (raster).
    pub lflvl: Vec<LfMask>,
    pub rac: Option<Rac<'a>>,
}

const LEFT_CTX: [u8; 13] = [0x0, 0x8, 0x0, 0x8, 0xc, 0x8, 0xc, 0xe, 0xc, 0xe, 0xf, 0xe, 0xf];
const ABOVE_CTX: [u8; 13] = [0x0, 0x0, 0x8, 0x8, 0x8, 0xc, 0xc, 0xc, 0xe, 0xe, 0xe, 0xf, 0xf];
const MAX_TX_FOR_BL_BP: [u8; 13] = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0];
const INTER_MODE_CTX_LUT: [[u8; 14]; 14] = {
    let mut t = [[6u8; 14]; 14];
    let mut i = 0;
    while i < 14 {
        let mut j = 10;
        while j < 14 {
            t[i][j] = 5;
            j += 1;
        }
        i += 1;
    }
    let mut j = 0;
    while j < 10 {
        t[10][j] = 5;
        t[11][j] = 5;
        t[12][j] = 5;
        t[13][j] = 5;
        j += 1;
    }
    t[10][10] = 2;
    t[10][11] = 2;
    t[10][12] = 1;
    t[10][13] = 3;
    t[11][10] = 2;
    t[11][11] = 2;
    t[11][12] = 1;
    t[11][13] = 3;
    t[12][10] = 1;
    t[12][11] = 1;
    t[12][12] = 0;
    t[12][13] = 3;
    t[13][10] = 3;
    t[13][11] = 3;
    t[13][12] = 3;
    t[13][13] = 4;
    t
};

/// `band_counts`.
const BAND_COUNTS: [[usize; 6]; 4] = [[1, 2, 3, 4, 3, 16 - 13], [1, 2, 3, 4, 11, 64 - 21], [1, 2, 3, 4, 11, 256 - 21], [1, 2, 3, 4, 11, 1024 - 21]];

/// `ff_vp9_scans` / `ff_vp9_scans_nb`: `[tx (4 = lossless)][txtp]`.
pub fn scan(tx: usize, txtp: usize) -> (&'static [i16], &'static [[i16; 2]]) {
    match (tx, txtp) {
        (0, 1) => (&COL_SCAN_4X4, &COL_SCAN_4X4_NB),
        (0, 2) => (&ROW_SCAN_4X4, &ROW_SCAN_4X4_NB),
        (0, _) | (4, _) => (&DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NB),
        (1, 1) => (&COL_SCAN_8X8, &COL_SCAN_8X8_NB),
        (1, 2) => (&ROW_SCAN_8X8, &ROW_SCAN_8X8_NB),
        (1, _) => (&DEFAULT_SCAN_8X8, &DEFAULT_SCAN_8X8_NB),
        (2, 1) => (&COL_SCAN_16X16, &COL_SCAN_16X16_NB),
        (2, 2) => (&ROW_SCAN_16X16, &ROW_SCAN_16X16_NB),
        (2, _) => (&DEFAULT_SCAN_16X16, &DEFAULT_SCAN_16X16_NB),
        _ => (&DEFAULT_SCAN_32X32, &DEFAULT_SCAN_32X32_NB),
    }
}

fn set_tile_offset(idx: u32, log2_n: u32, n: usize) -> (usize, usize) {
    let sb_start = ((idx as usize) * n) >> log2_n;
    let sb_end = ((idx as usize + 1) * n) >> log2_n;
    (sb_start.min(n) << 3, sb_end.min(n) << 3)
}

#[inline]
fn fill(a: &mut [u8], off: usize, n: usize, v: u8) {
    a[off..off + n].fill(v);
}

impl<'a> FrameDec<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        h: &'a Header,
        p: &'a ProbContext,
        coef: &'a CoefProbs,
        counts: &'a mut Counts,
        cur: &'a mut Frame,
        refs: &'a [Option<Arc<Frame>>; 3],
        segmap_ref: Option<&'a Frame>,
        mvpair: Option<&'a Frame>,
        mvscale: [[u16; 2]; 3],
        mvstep: [[u8; 2]; 3],
    ) -> Self {
        let (w, hh) = (cur.width as usize, cur.height as usize);
        let sb_cols = (w + 63) >> 6;
        let sb_rows = (hh + 63) >> 6;
        let n8 = sb_cols * 8;
        let n16 = sb_cols * 16;
        let kf = h.keyframe || h.intraonly;
        FrameDec {
            h,
            p,
            coef,
            counts,
            ss_h: cur.ss_h as usize,
            ss_v: cur.ss_v as usize,
            bpp: cur.bpp as u32,
            cur,
            refs,
            segmap_ref,
            mvpair,
            mvscale,
            mvstep,
            cols: (w + 7) >> 3,
            rows: (hh + 7) >> 3,
            sb_cols,
            sb_rows,
            above_partition_ctx: vec![0; n8],
            above_mode_ctx: vec![if kf { DC_PRED as u8 } else { NEARESTMV }; n16],
            above_y_nnz_ctx: vec![0; n16],
            above_uv_nnz_ctx: [vec![0; n16], vec![0; n16]],
            above_skip_ctx: vec![0; n8],
            above_txfm_ctx: vec![0; n8],
            above_segpred_ctx: vec![0; n8],
            above_intra_ctx: vec![0; n8],
            above_comp_ctx: vec![0; n8],
            above_ref_ctx: vec![0; n8],
            above_filter_ctx: vec![0; n8],
            above_mv_ctx: vec![[Mv::default(); 2]; n16],
            left_y_nnz_ctx: [0; 16],
            left_mode_ctx: [0; 16],
            left_mv_ctx: [[Mv::default(); 2]; 16],
            left_uv_nnz_ctx: [[0; 16]; 2],
            left_partition_ctx: [0; 8],
            left_skip_ctx: [0; 8],
            left_txfm_ctx: [0; 8],
            left_segpred_ctx: [0; 8],
            left_intra_ctx: [0; 8],
            left_comp_ctx: [0; 8],
            left_ref_ctx: [0; 8],
            left_filter_ctx: [0; 8],
            b: Block::default(),
            row: 0,
            row7: 0,
            col: 0,
            col7: 0,
            tile_col_start: 0,
            min_mv: (0, 0),
            max_mv: (0, 0),
            block: vec![0; 64 * 64],
            uvblock: [vec![0; 64 * 64], vec![0; 64 * 64]],
            eob: vec![0; 256],
            uveob: [vec![0; 256], vec![0; 256]],
            lflvl: vec![LfMask { level: [0; 64], mask: [[[[0; 4]; 8]; 2]; 2] }; sb_cols * sb_rows],
            rac: None,
        }
    }

    #[inline]
    pub fn c(&mut self) -> &mut Rac<'a> {
        self.rac.as_mut().unwrap()
    }

    /// `decode_tiles`.
    pub fn decode_tiles(&mut self, mut data: &'a [u8]) -> Result<(), String> {
        let h = self.h;
        let tile_cols = h.tile_cols as usize;
        for tile_row in 0..h.tile_rows {
            let (tile_row_start, tile_row_end) = set_tile_offset(tile_row, h.log2_tile_rows, self.sb_rows);
            let mut racs: Vec<Rac<'a>> = Vec::with_capacity(tile_cols);
            for tile_col in 0..tile_cols {
                let tile_size = if tile_col == tile_cols - 1 && tile_row == h.tile_rows - 1 {
                    data.len()
                } else {
                    if data.len() < 4 {
                        return Err("VP9: truncated tile size".into());
                    }
                    let s = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    data = &data[4..];
                    s
                };
                if tile_size > data.len() {
                    return Err("VP9: tile size past the packet".into());
                }
                let mut rac = Rac::new(&data[..tile_size])?;
                if rac.get_prob(128) {
                    return Err("VP9: tile marker bit set".into());
                }
                racs.push(rac);
                data = &data[tile_size..];
            }
            let mut row = tile_row_start;
            while row < tile_row_end {
                for tile_col in 0..tile_cols {
                    let (tile_col_start, tile_col_end) = set_tile_offset(tile_col as u32, h.log2_tile_cols, self.sb_cols);
                    self.tile_col_start = tile_col_start;
                    self.left_partition_ctx = [0; 8];
                    self.left_skip_ctx = [0; 8];
                    if h.keyframe || h.intraonly {
                        self.left_mode_ctx = [DC_PRED as u8; 16];
                    } else {
                        self.left_mode_ctx[..8].fill(NEARESTMV);
                    }
                    self.left_y_nnz_ctx = [0; 16];
                    self.left_uv_nnz_ctx = [[0; 16]; 2];
                    self.left_segpred_ctx = [0; 8];
                    self.rac = Some(racs[tile_col].clone());
                    let mut col = tile_col_start;
                    while col < tile_col_end {
                        let sb = (row >> 3) * self.sb_cols + (col >> 3);
                        self.lflvl[sb].mask = [[[[0; 4]; 8]; 2]; 2];
                        if self.c().is_end() {
                            return Err("VP9: tile data overrun".into());
                        }
                        self.decode_sb(row, col, BL_64X64);
                        col += 8;
                    }
                    racs[tile_col] = self.rac.take().unwrap();
                }
                row += 8;
            }
        }
        Ok(())
    }

    fn decode_sb(&mut self, row: usize, col: usize, bl: u8) {
        let c = ((self.above_partition_ctx[col] >> (3 - bl)) & 1) as usize | ((((self.left_partition_ctx[row & 7] >> (3 - bl)) & 1) as usize) << 1);
        let p = if self.h.keyframe || self.h.intraonly { DEFAULT_KF_PARTITION_PROBS[bl as usize][c] } else { self.p.partition[bl as usize][c] };
        let hbs = 4usize >> bl;
        let bp;
        if bl == BL_8X8 {
            bp = self.c().get_tree(&PARTITION_TREE, &p);
            self.decode_block(row, col, bl, bp);
        } else if col + hbs < self.cols {
            if row + hbs < self.rows {
                bp = self.c().get_tree(&PARTITION_TREE, &p);
                match bp {
                    PARTITION_NONE => self.decode_block(row, col, bl, bp),
                    PARTITION_H => {
                        self.decode_block(row, col, bl, bp);
                        self.decode_block(row + hbs, col, bl, bp);
                    }
                    PARTITION_V => {
                        self.decode_block(row, col, bl, bp);
                        self.decode_block(row, col + hbs, bl, bp);
                    }
                    _ => {
                        self.decode_sb(row, col, bl + 1);
                        self.decode_sb(row, col + hbs, bl + 1);
                        self.decode_sb(row + hbs, col, bl + 1);
                        self.decode_sb(row + hbs, col + hbs, bl + 1);
                    }
                }
            } else if self.c().get_prob(p[1]) {
                bp = PARTITION_SPLIT;
                self.decode_sb(row, col, bl + 1);
                self.decode_sb(row, col + hbs, bl + 1);
            } else {
                bp = PARTITION_H;
                self.decode_block(row, col, bl, bp);
            }
        } else if row + hbs < self.rows {
            if self.c().get_prob(p[2]) {
                bp = PARTITION_SPLIT;
                self.decode_sb(row, col, bl + 1);
                self.decode_sb(row + hbs, col, bl + 1);
            } else {
                bp = PARTITION_V;
                self.decode_block(row, col, bl, bp);
            }
        } else {
            bp = PARTITION_SPLIT;
            self.decode_sb(row, col, bl + 1);
        }
        self.counts.partition[bl as usize][c][bp as usize] += 1;
    }

    /// `ff_vp9_decode_block`.
    fn decode_block(&mut self, row: usize, col: usize, bl: u8, bp: u8) {
        let bs = bl * 3 + bp;
        let w4 = BWH_TAB[1][bs as usize][0] as usize;
        let h4 = BWH_TAB[1][bs as usize][1] as usize;
        self.row = row;
        self.row7 = row & 7;
        self.col = col;
        self.col7 = col & 7;
        self.min_mv = (-(128 + col as i32 * 64), -(128 + row as i32 * 64));
        self.max_mv = (128 + (self.cols as i32 - col as i32 - w4 as i32) * 64, 128 + (self.rows as i32 - row as i32 - h4 as i32) * 64);
        self.b.bs = bs;
        self.b.bl = bl;
        self.b.bp = bp;
        self.decode_mode();
        let b = &mut self.b;
        b.uvtx = b.tx - (((self.ss_h != 0) && w4 * 2 == (1 << b.tx)) || ((self.ss_v != 0) && h4 * 2 == (1 << b.tx))) as u8;
        if !self.b.skip {
            let has_coeffs = self.decode_coeffs();
            if !has_coeffs && self.b.bs <= BS_8X8 && !self.b.intra {
                self.b.skip = true;
                fill(&mut self.above_skip_ctx, col, w4, 1);
                fill(&mut self.left_skip_ctx, self.row7, h4, 1);
            }
        } else {
            let row7 = self.row7;
            fill(&mut self.above_y_nnz_ctx, col * 2, w4 * 2, 0);
            fill(&mut self.left_y_nnz_ctx, row7 * 2, h4 * 2, 0);
            for pl in 0..2 {
                if self.ss_h != 0 {
                    fill(&mut self.above_uv_nnz_ctx[pl], col, w4, 0);
                } else {
                    fill(&mut self.above_uv_nnz_ctx[pl], col * 2, w4 * 2, 0);
                }
                if self.ss_v != 0 {
                    fill(&mut self.left_uv_nnz_ctx[pl], row7, h4, 0);
                } else {
                    fill(&mut self.left_uv_nnz_ctx[pl], row7 * 2, h4 * 2, 0);
                }
            }
        }
        if self.b.intra {
            self.intra_recon();
        } else {
            self.inter_recon();
        }
        // Loop-filter level and edges.
        let b = self.b;
        let lvl = self.h.segmentation.feat[b.seg_id as usize].lflvl[if b.intra { 0 } else { b.r#ref[0] as usize + 1 }][(b.mode[3] != ZEROMV) as usize];
        if self.h.filter_level != 0 && lvl > 0 {
            let x_end = (self.cols - col).min(w4);
            let y_end = (self.rows - row).min(h4);
            let skip_inter = !b.intra && b.skip;
            let (col7, row7) = (self.col7, self.row7);
            let sb = (row >> 3) * self.sb_cols + (col >> 3);
            let lf = &mut self.lflvl[sb];
            for y in 0..h4 {
                lf.level[(row7 + y) * 8 + col7..(row7 + y) * 8 + col7 + w4].fill(lvl);
            }
            mask_edges(&mut lf.mask[0], 0, 0, row7, col7, x_end, y_end, 0, 0, b.tx, skip_inter);
            if self.ss_h != 0 || self.ss_v != 0 {
                let col_end = if self.cols & 1 != 0 && col + w4 >= self.cols { self.cols & 7 } else { 0 };
                let row_end = if self.rows & 1 != 0 && row + h4 >= self.rows { self.rows & 7 } else { 0 };
                mask_edges(&mut lf.mask[1], self.ss_h, self.ss_v, row7, col7, x_end, y_end, col_end, row_end, b.uvtx, skip_inter);
            }
        }
    }

    /// `decode_mode`.
    fn decode_mode(&mut self) {
        let h = self.h;
        let (row, col, row7) = (self.row, self.col, self.row7);
        let bs = self.b.bs as usize;
        let max_tx = MAX_TX_FOR_BL_BP[bs];
        let bw4 = BWH_TAB[1][bs][0] as usize;
        let w4 = (self.cols - col).min(bw4);
        let bh4 = BWH_TAB[1][bs][1] as usize;
        let h4 = (self.rows - row).min(bh4);
        let have_a = row > 0;
        let have_l = col > self.tile_col_start;
        let seg = &h.segmentation;
        let kf = h.keyframe || h.intraonly;

        // Segment id.
        if !seg.enabled {
            self.b.seg_id = 0;
        } else if kf {
            self.b.seg_id = if !seg.update_map { 0 } else { self.c().get_tree(&SEGMENTATION_TREE, &seg.prob) };
        } else if !seg.update_map || (seg.temporal && {
            let c = (self.above_segpred_ctx[col] + self.left_segpred_ctx[row7]) as usize;
            let pp = seg.pred_prob[c];
            self.c().get_prob(pp)
        }) {
            if let (false, Some(rf)) = (h.errorres, self.segmap_ref) {
                let mut pred = 8u8;
                for y in 0..h4 {
                    let base = (y + row) * 8 * self.sb_cols + col;
                    for x in 0..w4 {
                        pred = pred.min(rf.segmap[base + x]);
                    }
                }
                self.b.seg_id = pred.min(7);
            } else {
                self.b.seg_id = 0;
            }
            fill(&mut self.above_segpred_ctx, col, w4, 1);
            fill(&mut self.left_segpred_ctx, row7, h4, 1);
        } else {
            self.b.seg_id = self.c().get_tree(&SEGMENTATION_TREE, &seg.prob);
            fill(&mut self.above_segpred_ctx, col, w4, 0);
            fill(&mut self.left_segpred_ctx, row7, h4, 0);
        }
        if seg.enabled && (seg.update_map || kf) {
            let stride = 8 * self.sb_cols;
            for y in 0..bh4 {
                let o = (row + y) * stride + col;
                if o + bw4 <= self.cur.segmap.len() {
                    self.cur.segmap[o..o + bw4].fill(self.b.seg_id);
                }
            }
        }
        let feat = seg.feat[self.b.seg_id as usize];

        // Skip.
        self.b.skip = seg.enabled && feat.skip_enabled;
        if !self.b.skip {
            let c = (self.left_skip_ctx[row7] + self.above_skip_ctx[col]) as usize;
            let p = self.p.skip[c];
            self.b.skip = self.c().get_prob(p);
            self.counts.skip[c][self.b.skip as usize] += 1;
        }

        // Intra / inter.
        if kf {
            self.b.intra = true;
        } else if seg.enabled && feat.ref_enabled {
            self.b.intra = feat.ref_val == 0;
        } else {
            let c = if have_a && have_l {
                let c = (self.above_intra_ctx[col] + self.left_intra_ctx[row7]) as usize;
                c + (c == 2) as usize
            } else if have_a {
                2 * self.above_intra_ctx[col] as usize
            } else if have_l {
                2 * self.left_intra_ctx[row7] as usize
            } else {
                0
            };
            let p = self.p.intra[c];
            let bit = self.c().get_prob(p);
            self.counts.intra[c][bit as usize] += 1;
            self.b.intra = !bit;
        }

        // Transform size.
        if (self.b.intra || !self.b.skip) && h.txfmmode == TX_SWITCHABLE {
            let c = if have_a {
                if have_l {
                    let a = if self.above_skip_ctx[col] != 0 { max_tx } else { self.above_txfm_ctx[col] };
                    let l = if self.left_skip_ctx[row7] != 0 { max_tx } else { self.left_txfm_ctx[row7] };
                    (a + l > max_tx) as usize
                } else if self.above_skip_ctx[col] != 0 {
                    1
                } else {
                    (self.above_txfm_ctx[col] * 2 > max_tx) as usize
                }
            } else if have_l {
                if self.left_skip_ctx[row7] != 0 {
                    1
                } else {
                    (self.left_txfm_ctx[row7] * 2 > max_tx) as usize
                }
            } else {
                1
            };
            let p = *self.p;
            self.b.tx = match max_tx {
                TX_32X32 => {
                    let mut t = self.c().get_prob(p.tx32p[c][0]) as u8;
                    if t != 0 {
                        t += self.c().get_prob(p.tx32p[c][1]) as u8;
                        if t == 2 {
                            t += self.c().get_prob(p.tx32p[c][2]) as u8;
                        }
                    }
                    self.counts.tx32p[c][t as usize] += 1;
                    t
                }
                TX_16X16 => {
                    let mut t = self.c().get_prob(p.tx16p[c][0]) as u8;
                    if t != 0 {
                        t += self.c().get_prob(p.tx16p[c][1]) as u8;
                    }
                    self.counts.tx16p[c][t as usize] += 1;
                    t
                }
                TX_8X8 => {
                    let t = self.c().get_prob(p.tx8p[c]) as u8;
                    self.counts.tx8p[c][t as usize] += 1;
                    t
                }
                _ => TX_4X4,
            };
        } else {
            self.b.tx = max_tx.min(h.txfmmode);
        }

        let mut vref = 0u8;
        let mut filter_id = 0u8;
        if kf {
            self.kf_modes();
        } else if self.b.intra {
            self.b.comp = false;
            let p = *self.p;
            if bs > BS_8X8 as usize {
                let m0 = self.c().get_tree(&INTRAMODE_TREE, &p.y_mode[0]);
                self.b.mode[0] = m0;
                self.counts.y_mode[0][m0 as usize] += 1;
                if bs != BS_8X4 as usize {
                    let m = self.c().get_tree(&INTRAMODE_TREE, &p.y_mode[0]);
                    self.b.mode[1] = m;
                    self.counts.y_mode[0][m as usize] += 1;
                } else {
                    self.b.mode[1] = m0;
                }
                if bs != BS_4X8 as usize {
                    let m2 = self.c().get_tree(&INTRAMODE_TREE, &p.y_mode[0]);
                    self.b.mode[2] = m2;
                    self.counts.y_mode[0][m2 as usize] += 1;
                    if bs != BS_8X4 as usize {
                        let m = self.c().get_tree(&INTRAMODE_TREE, &p.y_mode[0]);
                        self.b.mode[3] = m;
                        self.counts.y_mode[0][m as usize] += 1;
                    } else {
                        self.b.mode[3] = m2;
                    }
                } else {
                    self.b.mode[2] = self.b.mode[0];
                    self.b.mode[3] = self.b.mode[1];
                }
            } else {
                const SIZE_GROUP: [usize; 10] = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1];
                let sz = SIZE_GROUP[bs];
                let m = self.c().get_tree(&INTRAMODE_TREE, &p.y_mode[sz]);
                self.b.mode = [m; 4];
                self.counts.y_mode[sz][m as usize] += 1;
            }
            let m3 = self.b.mode[3] as usize;
            let uv = self.c().get_tree(&INTRAMODE_TREE, &p.uv_mode[m3]);
            self.b.uvmode = uv;
            self.counts.uv_mode[m3][uv as usize] += 1;
        } else {
            (vref, filter_id) = self.inter_modes(have_a, have_l);
        }

        // Contexts.
        let b = self.b;
        let kf = h.keyframe || h.intraonly;
        let switchable = h.filtermode == FilterMode::Switchable;
        let set = |dir_skip: &mut [u8], dir_txfm: &mut [u8], dir_part: &mut [u8], off: usize, n: usize, part: u8| {
            dir_skip[off..off + n].fill(b.skip as u8);
            dir_txfm[off..off + n].fill(b.tx);
            dir_part[off..off + n].fill(part);
        };
        let wn = BWH_TAB[1][bs][0] as usize;
        let hn = BWH_TAB[1][bs][1] as usize;
        set(&mut self.above_skip_ctx, &mut self.above_txfm_ctx, &mut self.above_partition_ctx, col, wn, ABOVE_CTX[bs]);
        set(&mut self.left_skip_ctx, &mut self.left_txfm_ctx, &mut self.left_partition_ctx, row7, hn, LEFT_CTX[bs]);
        if !kf {
            fill(&mut self.above_intra_ctx, col, wn, b.intra as u8);
            fill(&mut self.above_comp_ctx, col, wn, b.comp as u8);
            fill(&mut self.above_mode_ctx, col, wn, b.mode[3]);
            fill(&mut self.left_intra_ctx, row7, hn, b.intra as u8);
            fill(&mut self.left_comp_ctx, row7, hn, b.comp as u8);
            fill(&mut self.left_mode_ctx, row7, hn, b.mode[3]);
            if !b.intra {
                fill(&mut self.above_ref_ctx, col, wn, vref);
                fill(&mut self.left_ref_ctx, row7, hn, vref);
                if switchable {
                    fill(&mut self.above_filter_ctx, col, wn, filter_id);
                    fill(&mut self.left_filter_ctx, row7, hn, filter_id);
                }
            }
        }
        if !kf {
            if bs > BS_8X8 as usize {
                self.left_mv_ctx[row7 * 2] = b.mv[1];
                self.left_mv_ctx[row7 * 2 + 1] = b.mv[3];
                self.above_mv_ctx[col * 2] = b.mv[2];
                self.above_mv_ctx[col * 2 + 1] = b.mv[3];
            } else {
                for n in 0..w4 * 2 {
                    self.above_mv_ctx[col * 2 + n] = b.mv[3];
                }
                for n in 0..h4 * 2 {
                    self.left_mv_ctx[row7 * 2 + n] = b.mv[3];
                }
            }
        }
        // Motion for later MV prediction.
        for y in 0..h4 {
            let o = (row + y) * self.sb_cols * 8 + col;
            for x in 0..w4 {
                let m = &mut self.cur.mv[o + x];
                if b.intra {
                    m.r#ref = [-1, -1];
                } else if b.comp {
                    m.r#ref = [b.r#ref[0] as i8, b.r#ref[1] as i8];
                    m.mv = b.mv[3];
                } else {
                    m.r#ref = [b.r#ref[0] as i8, -1];
                    m.mv[0] = b.mv[3][0];
                }
            }
        }
    }

    /// Keyframe / intra-only luma and chroma modes (default probabilities,
    /// contexts from the above/left modes).
    fn kf_modes(&mut self) {
        let bs = self.b.bs;
        let (ai, li) = (self.col * 2, self.row7 << 1);
        self.b.comp = false;
        let probs = |a: u8, l: u8| &DEFAULT_KF_YMODE_PROBS[a as usize][l as usize];
        if bs > BS_8X8 {
            let (a0, l0) = (self.above_mode_ctx[ai], self.left_mode_ctx[li]);
            let m0 = self.c().get_tree(&INTRAMODE_TREE, probs(a0, l0));
            self.b.mode[0] = m0;
            self.above_mode_ctx[ai] = m0;
            if bs != BS_8X4 {
                let a1 = self.above_mode_ctx[ai + 1];
                let m1 = self.c().get_tree(&INTRAMODE_TREE, probs(a1, m0));
                self.b.mode[1] = m1;
                self.left_mode_ctx[li] = m1;
                self.above_mode_ctx[ai + 1] = m1;
            } else {
                self.b.mode[1] = m0;
                self.left_mode_ctx[li] = m0;
                self.above_mode_ctx[ai + 1] = m0;
            }
            if bs != BS_4X8 {
                let (a0, l1) = (self.above_mode_ctx[ai], self.left_mode_ctx[li + 1]);
                let m2 = self.c().get_tree(&INTRAMODE_TREE, probs(a0, l1));
                self.b.mode[2] = m2;
                self.above_mode_ctx[ai] = m2;
                if bs != BS_8X4 {
                    let a1 = self.above_mode_ctx[ai + 1];
                    let m3 = self.c().get_tree(&INTRAMODE_TREE, probs(a1, m2));
                    self.b.mode[3] = m3;
                    self.left_mode_ctx[li + 1] = m3;
                    self.above_mode_ctx[ai + 1] = m3;
                } else {
                    self.b.mode[3] = m2;
                    self.left_mode_ctx[li + 1] = m2;
                    self.above_mode_ctx[ai + 1] = m2;
                }
            } else {
                self.b.mode[2] = self.b.mode[0];
                self.b.mode[3] = self.b.mode[1];
                self.left_mode_ctx[li + 1] = self.b.mode[1];
                self.above_mode_ctx[ai + 1] = self.b.mode[1];
            }
        } else {
            let (a, l) = (self.above_mode_ctx[ai], self.left_mode_ctx[li]);
            let m = self.c().get_tree(&INTRAMODE_TREE, probs(a, l));
            self.b.mode = [m; 4];
            let wn = BWH_TAB[0][bs as usize][0] as usize;
            let hn = BWH_TAB[0][bs as usize][1] as usize;
            self.above_mode_ctx[ai..ai + wn].fill(m);
            self.left_mode_ctx[li..li + hn].fill(m);
        }
        let m3 = self.b.mode[3] as usize;
        self.b.uvmode = self.c().get_tree(&INTRAMODE_TREE, &DEFAULT_KF_UVMODE_PROBS[m3]);
    }

    /// Inter block: comp flag, references, modes, filter and motion
    /// vectors. Returns `(vref, filter_id)` for the contexts.
    fn inter_modes(&mut self, have_a: bool, have_l: bool) -> (u8, u8) {
        let h = self.h;
        let (col, row7) = (self.col, self.row7);
        let bs = self.b.bs;
        let feat = h.segmentation.feat[self.b.seg_id as usize];
        let p = *self.p;
        let (ai, li) = (self.above_intra_ctx[col], self.left_intra_ctx[row7]);
        let (ac, lc) = (self.above_comp_ctx[col], self.left_comp_ctx[row7]);
        let (ar, lr) = (self.above_ref_ctx[col], self.left_ref_ctx[row7]);
        let fix = h.fixcompref;
        let var = h.varcompref;
        if h.segmentation.enabled && feat.ref_enabled {
            self.b.comp = false;
            self.b.r#ref[0] = feat.ref_val - 1;
        } else {
            if h.comppredmode != CompPred::Switchable {
                self.b.comp = h.comppredmode == CompPred::Comp;
            } else {
                let c = if have_a {
                    if have_l {
                        if ac != 0 && lc != 0 {
                            4
                        } else if ac != 0 {
                            2 + (li != 0 || lr == fix) as usize
                        } else if lc != 0 {
                            2 + (ai != 0 || ar == fix) as usize
                        } else {
                            ((ai == 0 && ar == fix) ^ (li == 0 && lr == fix)) as usize
                        }
                    } else if ac != 0 {
                        3
                    } else {
                        (ai == 0 && ar == fix) as usize
                    }
                } else if have_l {
                    if lc != 0 {
                        3
                    } else {
                        (li == 0 && lr == fix) as usize
                    }
                } else {
                    1
                };
                let bit = self.c().get_prob(p.comp[c]);
                self.b.comp = bit;
                self.counts.comp[c][bit as usize] += 1;
            }
            if self.b.comp {
                let fix_idx = h.signbias[fix as usize] as usize;
                let var_idx = 1 - fix_idx;
                self.b.r#ref[fix_idx] = fix;
                let c = if have_a {
                    if have_l {
                        if ai != 0 {
                            if li != 0 {
                                2
                            } else {
                                1 + 2 * (lr != var[1]) as usize
                            }
                        } else if li != 0 {
                            1 + 2 * (ar != var[1]) as usize
                        } else {
                            let (refl, refa) = (lr, ar);
                            if refl == refa && refa == var[1] {
                                0
                            } else if lc == 0 && ac == 0 {
                                if (refa == fix && refl == var[0]) || (refl == fix && refa == var[0]) {
                                    4
                                } else if refa == refl {
                                    3
                                } else {
                                    1
                                }
                            } else if lc == 0 {
                                if refa == var[1] && refl != var[1] {
                                    1
                                } else if refl == var[1] && refa != var[1] {
                                    2
                                } else {
                                    4
                                }
                            } else if ac == 0 {
                                if refl == var[1] && refa != var[1] {
                                    1
                                } else if refa == var[1] && refl != var[1] {
                                    2
                                } else {
                                    4
                                }
                            } else if refl == refa {
                                4
                            } else {
                                2
                            }
                        }
                    } else if ai != 0 {
                        2
                    } else if ac != 0 {
                        4 * (ar != var[1]) as usize
                    } else {
                        3 * (ar != var[1]) as usize
                    }
                } else if have_l {
                    if li != 0 {
                        2
                    } else if lc != 0 {
                        4 * (lr != var[1]) as usize
                    } else {
                        3 * (lr != var[1]) as usize
                    }
                } else {
                    2
                };
                let bit = self.c().get_prob(p.comp_ref[c]);
                self.b.r#ref[var_idx] = var[bit as usize];
                self.counts.comp_ref[c][bit as usize] += 1;
            } else {
                let c = if have_a && ai == 0 {
                    if have_l && li == 0 {
                        if lc != 0 {
                            if ac != 0 {
                                1 + (fix == 0 || lr == 0 || ar == 0) as usize
                            } else {
                                3 * (ar == 0) as usize + (fix == 0 || lr == 0) as usize
                            }
                        } else if ac != 0 {
                            3 * (lr == 0) as usize + (fix == 0 || ar == 0) as usize
                        } else {
                            2 * (lr == 0) as usize + 2 * (ar == 0) as usize
                        }
                    } else if ai != 0 {
                        2
                    } else if ac != 0 {
                        1 + (fix == 0 || ar == 0) as usize
                    } else {
                        4 * (ar == 0) as usize
                    }
                } else if have_l && li == 0 {
                    if lc != 0 {
                        1 + (fix == 0 || lr == 0) as usize
                    } else {
                        4 * (lr == 0) as usize
                    }
                } else {
                    2
                };
                let bit = self.c().get_prob(p.single_ref[c][0]);
                self.counts.single_ref[c][0][bit as usize] += 1;
                if !bit {
                    self.b.r#ref[0] = 0;
                } else {
                    let c = if have_a {
                        if have_l {
                            if li != 0 {
                                if ai != 0 {
                                    2
                                } else if ac != 0 {
                                    1 + 2 * (fix == 1 || ar == 1) as usize
                                } else if ar == 0 {
                                    3
                                } else {
                                    4 * (ar == 1) as usize
                                }
                            } else if ai != 0 {
                                if li != 0 {
                                    2
                                } else if lc != 0 {
                                    1 + 2 * (fix == 1 || lr == 1) as usize
                                } else if lr == 0 {
                                    3
                                } else {
                                    4 * (lr == 1) as usize
                                }
                            } else if ac != 0 {
                                if lc != 0 {
                                    if lr == ar {
                                        3 * (fix == 1 || lr == 1) as usize
                                    } else {
                                        2
                                    }
                                } else if lr == 0 {
                                    1 + 2 * (fix == 1 || ar == 1) as usize
                                } else {
                                    3 * (lr == 1) as usize + (fix == 1 || ar == 1) as usize
                                }
                            } else if lc != 0 {
                                if ar == 0 {
                                    1 + 2 * (fix == 1 || lr == 1) as usize
                                } else {
                                    3 * (ar == 1) as usize + (fix == 1 || lr == 1) as usize
                                }
                            } else if ar == 0 {
                                if lr == 0 {
                                    3
                                } else {
                                    4 * (lr == 1) as usize
                                }
                            } else if lr == 0 {
                                4 * (ar == 1) as usize
                            } else {
                                2 * (lr == 1) as usize + 2 * (ar == 1) as usize
                            }
                        } else if ai != 0 || (ac == 0 && ar == 0) {
                            2
                        } else if ac != 0 {
                            3 * (fix == 1 || ar == 1) as usize
                        } else {
                            4 * (ar == 1) as usize
                        }
                    } else if have_l {
                        if li != 0 || (lc == 0 && lr == 0) {
                            2
                        } else if lc != 0 {
                            3 * (fix == 1 || lr == 1) as usize
                        } else {
                            4 * (lr == 1) as usize
                        }
                    } else {
                        2
                    };
                    let bit = self.c().get_prob(p.single_ref[c][1]);
                    self.counts.single_ref[c][1][bit as usize] += 1;
                    self.b.r#ref[0] = 1 + bit as u8;
                }
            }
        }

        if bs <= BS_8X8 {
            if h.segmentation.enabled && feat.skip_enabled {
                self.b.mode = [ZEROMV; 4];
            } else {
                const OFF: [usize; 10] = [3, 0, 0, 1, 0, 0, 0, 0, 0, 0];
                let o = OFF[bs as usize];
                let c = INTER_MODE_CTX_LUT[self.above_mode_ctx[col + o] as usize][self.left_mode_ctx[row7 + o] as usize] as usize;
                let m = self.c().get_tree(&INTER_MODE_TREE, &p.mv_mode[c]);
                self.b.mode = [m; 4];
                self.counts.mv_mode[c][(m - 10) as usize] += 1;
            }
        }

        let filter_id;
        if h.filtermode == FilterMode::Switchable {
            let c = if have_a && self.above_mode_ctx[col] >= NEARESTMV {
                if have_l && self.left_mode_ctx[row7] >= NEARESTMV {
                    if self.above_filter_ctx[col] == self.left_filter_ctx[row7] {
                        self.left_filter_ctx[row7] as usize
                    } else {
                        3
                    }
                } else {
                    self.above_filter_ctx[col] as usize
                }
            } else if have_l && self.left_mode_ctx[row7] >= NEARESTMV {
                self.left_filter_ctx[row7] as usize
            } else {
                3
            };
            filter_id = self.c().get_tree(&FILTER_TREE, &p.filter[c]);
            self.counts.filter[c][filter_id as usize] += 1;
            self.b.filter = FILTER_LUT[filter_id as usize];
        } else {
            filter_id = 0;
            self.b.filter = h.filtermode as u8;
        }

        if bs > BS_8X8 {
            let c = INTER_MODE_CTX_LUT[self.above_mode_ctx[col] as usize][self.left_mode_ctx[row7] as usize] as usize;
            let m0 = self.c().get_tree(&INTER_MODE_TREE, &p.mv_mode[c]);
            self.b.mode[0] = m0;
            self.counts.mv_mode[c][(m0 - 10) as usize] += 1;
            self.fill_mv(0, m0, 0);
            if bs != BS_8X4 {
                let m1 = self.c().get_tree(&INTER_MODE_TREE, &p.mv_mode[c]);
                self.b.mode[1] = m1;
                self.counts.mv_mode[c][(m1 - 10) as usize] += 1;
                self.fill_mv(1, m1, 1);
            } else {
                self.b.mode[1] = m0;
                self.b.mv[1] = self.b.mv[0];
            }
            if bs != BS_4X8 {
                let m2 = self.c().get_tree(&INTER_MODE_TREE, &p.mv_mode[c]);
                self.b.mode[2] = m2;
                self.counts.mv_mode[c][(m2 - 10) as usize] += 1;
                self.fill_mv(2, m2, 2);
                if bs != BS_8X4 {
                    let m3 = self.c().get_tree(&INTER_MODE_TREE, &p.mv_mode[c]);
                    self.b.mode[3] = m3;
                    self.counts.mv_mode[c][(m3 - 10) as usize] += 1;
                    self.fill_mv(3, m3, 3);
                } else {
                    self.b.mode[3] = m2;
                    self.b.mv[3] = self.b.mv[2];
                }
            } else {
                self.b.mode[2] = self.b.mode[0];
                self.b.mv[2] = self.b.mv[0];
                self.b.mode[3] = self.b.mode[1];
                self.b.mv[3] = self.b.mv[1];
            }
        } else {
            let m = self.b.mode[0];
            self.fill_mv(0, m, -1);
            let mv = self.b.mv[0];
            self.b.mv = [mv; 4];
        }
        let vref = self.b.r#ref[if self.b.comp { h.signbias[h.varcompref[0] as usize] as usize } else { 0 }];
        (vref, filter_id)
    }

    /// `decode_coeffs`: returns whether any block has coefficients.
    fn decode_coeffs(&mut self) -> bool {
        let h = self.h;
        let b = self.b;
        let (row, col) = (self.row, self.col);
        let bs = b.bs as usize;
        let mut w4 = (BWH_TAB[1][bs][0] as usize) << 1;
        let mut h4 = (BWH_TAB[1][bs][1] as usize) << 1;
        let mut end_x = (2 * (self.cols - col)).min(w4);
        let mut end_y = (2 * (self.rows - row)).min(h4);
        let qmul = h.segmentation.feat[b.seg_id as usize].qmul;
        let tx = 4 * h.lossless as usize + b.tx as usize;
        let inter = !b.intra as usize;
        let mut total = false;
        let eight = self.bpp == 8;
        let bpp = self.bpp;

        // Y.
        let step = 1usize << b.tx;
        let (a_off, l_off) = (col * 2, (row & 7) << 1);
        if step > 1 {
            merge(&mut self.left_y_nnz_ctx[l_off..], end_y, step);
            merge(&mut self.above_y_nnz_ctx[a_off..], end_x, step);
        }
        let mut n = 0usize;
        let mut y = 0;
        while y < end_y {
            let mut x = 0;
            while x < end_x {
                let mode = b.mode[if b.tx == TX_4X4 && b.bs > BS_8X8 { n } else { 0 }] as usize;
                let txtp = INTRA_TXFM_TYPE[mode] as usize;
                let (sc, nb) = scan(tx, txtp);
                let nnz = (self.above_y_nnz_ctx[a_off + x] + self.left_y_nnz_ctx[l_off + y]) as usize;
                let rac = self.rac.as_mut().unwrap();
                let ret = decode_coeffs_b(
                    rac,
                    &mut self.block[16 * n..16 * n + 16 * step * step],
                    16 * step * step,
                    b.tx == TX_32X32,
                    eight,
                    bpp,
                    &mut self.counts.coef[b.tx as usize][0][inter],
                    &mut self.counts.eob[b.tx as usize][0][inter],
                    &self.coef[b.tx as usize][0][inter],
                    nnz,
                    sc,
                    nb,
                    &BAND_COUNTS[b.tx as usize],
                    &qmul[0],
                );
                let nz = (ret != 0) as u8;
                self.above_y_nnz_ctx[a_off + x] = nz;
                self.left_y_nnz_ctx[l_off + y] = nz;
                total |= ret != 0;
                self.eob[n] = ret as u16;
                x += step;
                n += step * step;
            }
            y += step;
        }
        if step > 1 {
            splat(&mut self.above_y_nnz_ctx[a_off..], end_x, step, end_x == w4);
            splat(&mut self.left_y_nnz_ctx[l_off..], end_y, step, end_y == h4);
        }

        // U/V.
        let uvstep = 1usize << b.uvtx;
        w4 >>= self.ss_h;
        end_x >>= self.ss_h;
        h4 >>= self.ss_v;
        end_y >>= self.ss_v;
        let (sc, nb) = scan(b.uvtx as usize, DCT_DCT);
        for pl in 0..2 {
            let a_off = col << (1 - self.ss_h);
            let l_off = (row & 7) << (1 - self.ss_v);
            if uvstep > 1 {
                merge(&mut self.left_uv_nnz_ctx[pl][l_off..], end_y, uvstep);
                merge(&mut self.above_uv_nnz_ctx[pl][a_off..], end_x, uvstep);
            }
            let mut n = 0usize;
            let mut y = 0;
            while y < end_y {
                let mut x = 0;
                while x < end_x {
                    let nnz = (self.above_uv_nnz_ctx[pl][a_off + x] + self.left_uv_nnz_ctx[pl][l_off + y]) as usize;
                    let rac = self.rac.as_mut().unwrap();
                    let ret = decode_coeffs_b(
                        rac,
                        &mut self.uvblock[pl][16 * n..16 * n + 16 * uvstep * uvstep],
                        16 * uvstep * uvstep,
                        b.uvtx == TX_32X32,
                        eight,
                        bpp,
                        &mut self.counts.coef[b.uvtx as usize][1][inter],
                        &mut self.counts.eob[b.uvtx as usize][1][inter],
                        &self.coef[b.uvtx as usize][1][inter],
                        nnz,
                        sc,
                        nb,
                        &BAND_COUNTS[b.uvtx as usize],
                        &qmul[1],
                    );
                    let nz = (ret != 0) as u8;
                    self.above_uv_nnz_ctx[pl][a_off + x] = nz;
                    self.left_uv_nnz_ctx[pl][l_off + y] = nz;
                    total |= ret != 0;
                    self.uveob[pl][n] = ret as u16;
                    x += uvstep;
                    n += uvstep * uvstep;
                }
                y += uvstep;
            }
            if uvstep > 1 {
                splat(&mut self.above_uv_nnz_ctx[pl][a_off..], end_x, uvstep, end_x == w4);
                splat(&mut self.left_uv_nnz_ctx[pl][l_off..], end_y, uvstep, end_y == h4);
            }
        }
        total
    }
}

/// `MERGE`: an n-entry context run counts as non-zero if any entry is.
fn merge(la: &mut [u8], end: usize, step: usize) {
    let mut n = 0;
    while n < end {
        let any = la[n..n + step].iter().any(|&v| v != 0);
        la[n] = any as u8;
        n += step;
    }
}

/// `SPLAT`: spread each transform's context over its entries, only up to
/// the frame edge unless the block is whole.
fn splat(la: &mut [u8], end: usize, step: usize, whole: bool) {
    if step == 2 {
        // `for (n = 1; n < end; n += step) la[n] = la[n - 1]`: stops at the
        // frame edge even inside a transform.
        let mut n = 1;
        while n < end {
            la[n] = la[n - 1];
            n += 2;
        }
        return;
    }
    let mut n = 0;
    while n < end {
        let v = la[n];
        if whole {
            la[n + 1..n + step].fill(v);
        } else {
            let k = (end - n - 1).min(step - 1);
            la[n + 1..n + 1 + k].fill(v);
        }
        n += step;
    }
}

/// `decode_coeffs_b_generic`. Coefficients land in `coef` at raster
/// positions; 8-bit streams keep them as `int16_t` (wrapping), as ffmpeg.
#[allow(clippy::too_many_arguments)]
fn decode_coeffs_b(
    c: &mut Rac,
    coef: &mut [i32],
    n_coeffs: usize,
    is_tx32x32: bool,
    eight: bool,
    bpp: u32,
    cnt: &mut [[[u32; 3]; 6]; 6],
    eob: &mut [[[u32; 2]; 6]; 6],
    p: &[[[u8; 11]; 6]; 6],
    mut nnz: usize,
    scan: &[i16],
    nb: &[[i16; 2]],
    band_counts: &[usize; 6],
    qmul: &[i16; 2],
) -> usize {
    let mut i = 0usize;
    let mut band = 0usize;
    let mut band_left = band_counts[band];
    let mut tp = &p[0][nnz];
    let mut cache = [0u8; 1024];
    'outer: loop {
        let val = c.get_prob(tp[0]);
        eob[band][nnz][val as usize] += 1;
        if !val {
            break;
        }
        // skip_eob:
        loop {
            if !c.get_prob(tp[1]) {
                cnt[band][nnz][0] += 1;
                band_left -= 1;
                if band_left == 0 {
                    band += 1;
                    band_left = band_counts[band.min(5)];
                }
                cache[scan[i] as usize] = 0;
                nnz = (1 + cache[nb[i][0] as usize] as usize + cache[nb[i][1] as usize] as usize) >> 1;
                tp = &p[band.min(5)][nnz];
                i += 1;
                if i == n_coeffs {
                    break 'outer;
                }
                continue;
            }
            break;
        }
        let rc = scan[i] as usize;
        let val: i32;
        if !c.get_prob(tp[2]) {
            cnt[band][nnz][1] += 1;
            val = 1;
            cache[rc] = 1;
        } else {
            cnt[band][nnz][2] += 1;
            if !c.get_prob(tp[3]) {
                if !c.get_prob(tp[4]) {
                    cache[rc] = 2;
                    val = 2;
                } else {
                    val = 3 + c.get_prob(tp[5]) as i32;
                    cache[rc] = 3;
                }
            } else if !c.get_prob(tp[6]) {
                cache[rc] = 4;
                if !c.get_prob(tp[7]) {
                    val = c.get_prob(159) as i32 + 5;
                } else {
                    let mut v = ((c.get_prob(165) as i32) << 1) + 7;
                    v += c.get_prob(145) as i32;
                    val = v;
                }
            } else {
                cache[rc] = 5;
                if !c.get_prob(tp[8]) {
                    if !c.get_prob(tp[9]) {
                        let mut v = 11 + ((c.get_prob(173) as i32) << 2);
                        v += (c.get_prob(148) as i32) << 1;
                        v += c.get_prob(140) as i32;
                        val = v;
                    } else {
                        let mut v = 19 + ((c.get_prob(176) as i32) << 3);
                        v += (c.get_prob(155) as i32) << 2;
                        v += (c.get_prob(140) as i32) << 1;
                        v += c.get_prob(135) as i32;
                        val = v;
                    }
                } else if !c.get_prob(tp[10]) {
                    let mut v = ((c.get_prob(180) as i32) << 4) + 35;
                    v += (c.get_prob(157) as i32) << 3;
                    v += (c.get_prob(141) as i32) << 2;
                    v += (c.get_prob(134) as i32) << 1;
                    v += c.get_prob(130) as i32;
                    val = v;
                } else {
                    let mut v = 67i32;
                    if !eight {
                        if bpp == 12 {
                            v += (c.get_prob(255) as i32) << 17;
                            v += (c.get_prob(255) as i32) << 16;
                        }
                        v += (c.get_prob(255) as i32) << 15;
                        v += (c.get_prob(255) as i32) << 14;
                    }
                    for (shift, pr) in [(13, 254), (12, 254), (11, 254), (10, 252), (9, 249), (8, 243), (7, 230), (6, 196), (5, 177), (4, 153), (3, 140), (2, 133), (1, 130), (0, 129)] {
                        v += (c.get_prob(pr) as i32) << shift;
                    }
                    val = v;
                }
            }
        }
        band_left -= 1;
        if band_left == 0 {
            band += 1;
            band_left = band_counts[band.min(5)];
        }
        let q = qmul[(i != 0) as usize] as u32;
        let s = if c.get() { -val } else { val };
        let prod = (s as u32).wrapping_mul(q);
        let v = if is_tx32x32 { (prod as i32) / 2 } else { prod as i32 };
        coef[rc] = if eight { v as i16 as i32 } else { v };
        nnz = (1 + cache[nb[i][0] as usize] as usize + cache[nb[i][1] as usize] as usize) >> 1;
        tp = &p[band.min(5)][nnz];
        i += 1;
        if i >= n_coeffs {
            break;
        }
    }
    i
}

/// `mask_edges`.
#[allow(clippy::too_many_arguments)]
fn mask_edges(mask: &mut [[[u8; 4]; 8]; 2], ss_h: usize, ss_v: usize, row_and_7: usize, col_and_7: usize, mut w: usize, mut h: usize, col_end: usize, row_end: usize, tx: u8, skip_inter: bool) {
    const WIDE_FILTER_COL_MASK: [u32; 2] = [0x11, 0x01];
    const WIDE_FILTER_ROW_MASK: [usize; 2] = [0x03, 0x07];
    if tx == TX_4X4 && (ss_v | ss_h) != 0 {
        if h == ss_v {
            if row_and_7 & 1 != 0 {
                return;
            }
            if row_end == 0 {
                h += 1;
            }
        }
        if w == ss_h {
            if col_and_7 & 1 != 0 {
                return;
            }
            if col_end == 0 {
                w += 1;
            }
        }
    }
    let t = 1u32 << col_and_7;
    let m_col = (t << w) - t;
    if tx == TX_4X4 && !skip_inter {
        let m_row_8 = m_col & WIDE_FILTER_COL_MASK[ss_h];
        let m_row_4 = m_col - m_row_8;
        for y in row_and_7..h + row_and_7 {
            let col_mask_id = 2 - (y & WIDE_FILTER_ROW_MASK[ss_v] == 0) as usize;
            mask[0][y][1] |= m_row_8 as u8;
            mask[0][y][2] |= m_row_4 as u8;
            if (ss_h & ss_v) != 0 && (col_end & 1) != 0 && (y & 1) != 0 {
                mask[1][y][col_mask_id] |= ((t << (w - 1)) - t) as u8;
            } else {
                mask[1][y][col_mask_id] |= m_col as u8;
            }
            if ss_h == 0 {
                mask[0][y][3] |= m_col as u8;
            }
            if ss_v == 0 {
                if ss_h != 0 && (col_end & 1) != 0 {
                    mask[1][y][3] |= ((t << (w - 1)) - t) as u8;
                } else {
                    mask[1][y][3] |= m_col as u8;
                }
            }
        }
    } else if !skip_inter {
        let mask_id = (tx == TX_8X8) as usize;
        let l2 = tx as i32 + ss_h as i32 - 1;
        const MASKS: [u32; 4] = [0xff, 0x55, 0x11, 0x01];
        let m_row = m_col & MASKS[l2 as usize];
        if ss_h != 0 && tx > TX_8X8 && (w ^ (w - 1)) == 1 {
            let m_row_16 = ((t << (w - 1)) - t) & MASKS[l2 as usize];
            let m_row_8 = m_row - m_row_16;
            for y in row_and_7..h + row_and_7 {
                mask[0][y][0] |= m_row_16 as u8;
                mask[0][y][1] |= m_row_8 as u8;
            }
        } else {
            for y in row_and_7..h + row_and_7 {
                mask[0][y][mask_id] |= m_row as u8;
            }
        }
        let l2 = tx as usize + ss_v - 1;
        let step1d = 1usize << l2;
        if ss_v != 0 && tx > TX_8X8 && (h ^ (h - 1)) == 1 {
            let mut y = row_and_7;
            while y < h + row_and_7 - 1 {
                mask[1][y][0] |= m_col as u8;
                y += step1d;
            }
            if y - row_and_7 == h - 1 {
                mask[1][y][1] |= m_col as u8;
            }
        } else {
            let mut y = row_and_7;
            while y < h + row_and_7 {
                mask[1][y][mask_id] |= m_col as u8;
                y += step1d;
            }
        }
    } else if tx != TX_4X4 {
        let mask_id = ((tx == TX_8X8) || (h == ss_v)) as usize;
        mask[1][row_and_7][mask_id] |= m_col as u8;
        let mask_id = ((tx == TX_8X8) || (w == ss_h)) as usize;
        for y in row_and_7..h + row_and_7 {
            mask[0][y][mask_id] |= t as u8;
        }
    } else {
        let t8 = t & WIDE_FILTER_COL_MASK[ss_h];
        let t4 = t - t8;
        for y in row_and_7..h + row_and_7 {
            mask[0][y][2] |= t4 as u8;
            mask[0][y][1] |= t8 as u8;
        }
        mask[1][row_and_7][2 - (row_and_7 & WIDE_FILTER_ROW_MASK[ss_v] == 0) as usize] |= m_col as u8;
    }
}
