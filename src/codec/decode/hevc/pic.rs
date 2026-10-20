//! Per-picture state: CTB and tile layout, and the 4×4-granular metadata
//! that prediction, deblocking and SAO read.

use std::sync::Arc;

use super::params::{Pps, Sps};
use crate::codec::decode::picture::Plane;

/// Where CTBs sit in tile scan, and the z-scan order of every 4×4 block
/// (§6.5.1, §6.5.2). Depends on the SPS and PPS only.
pub struct Layout {
    pub ctb_w: usize,
    pub ctb_h: usize,
    pub log2_ctb: u32,
    pub rs2ts: Vec<u32>,
    pub ts2rs: Vec<u32>,
    /// Tile index per CTB, by raster address.
    pub tile: Vec<u16>,
    /// Column boundaries of tiles, in CTBs.
    pub col_bd: Vec<u32>,
    /// Z-scan order per 4×4 block, raster over the picture in 4×4 units.
    pub z: Vec<u32>,
    pub w4: usize,
    pub h4: usize,
}

impl Layout {
    pub fn new(sps: &Sps, pps: &Pps) -> Layout {
        let ctb_w = sps.width_ctbs() as usize;
        let ctb_h = sps.height_ctbs() as usize;
        let (col_bd, row_bd) = if pps.tiles {
            (
                pps.tile_cols.boundaries(ctb_w as u32),
                pps.tile_rows.boundaries(ctb_h as u32),
            )
        } else {
            (vec![0, ctb_w as u32], vec![0, ctb_h as u32])
        };
        let n = ctb_w * ctb_h;
        let mut rs2ts = vec![0u32; n];
        let mut tile = vec![0u16; n];
        for rs in 0..n {
            let tbx = (rs % ctb_w) as u32;
            let tby = (rs / ctb_w) as u32;
            let tx = (0..col_bd.len() - 1).rfind(|&i| tbx >= col_bd[i]).unwrap_or(0);
            let ty = (0..row_bd.len() - 1).rfind(|&j| tby >= row_bd[j]).unwrap_or(0);
            let mut v = 0u32;
            for i in 0..tx {
                v += (row_bd[ty + 1] - row_bd[ty]) * (col_bd[i + 1] - col_bd[i]);
            }
            for j in 0..ty {
                v += ctb_w as u32 * (row_bd[j + 1] - row_bd[j]);
            }
            v += (tby - row_bd[ty]) * (col_bd[tx + 1] - col_bd[tx]) + tbx - col_bd[tx];
            rs2ts[rs] = v;
            tile[rs] = (ty * (col_bd.len() - 1) + tx) as u16;
        }
        let mut ts2rs = vec![0u32; n];
        for (rs, &ts) in rs2ts.iter().enumerate() {
            ts2rs[ts as usize] = rs as u32;
        }
        let w4 = (sps.width as usize).div_ceil(4);
        let h4 = (sps.height as usize).div_ceil(4);
        let shift = sps.log2_ctb - 2;
        let mut z = vec![0u32; w4 * h4];
        for y in 0..h4 {
            for x in 0..w4 {
                let rs = (y >> shift) * ctb_w + (x >> shift);
                let mut v = rs2ts[rs] << (2 * shift);
                for i in 0..shift {
                    let m = 1usize << i;
                    if x & m != 0 {
                        v += (m * m) as u32;
                    }
                    if y & m != 0 {
                        v += (2 * m * m) as u32;
                    }
                }
                z[y * w4 + x] = v;
            }
        }
        Layout {
            ctb_w,
            ctb_h,
            log2_ctb: sps.log2_ctb,
            rs2ts,
            ts2rs,
            tile,
            col_bd,
            z,
            w4,
            h4,
        }
    }

    /// Whether the CTB at tile-scan address `ts` starts a tile.
    pub fn tile_start(&self, ts: usize) -> bool {
        ts == 0 || self.tile[self.ts2rs[ts] as usize] != self.tile[self.ts2rs[ts - 1] as usize]
    }

    /// The tile column start (in CTBs) for CTB column `x`.
    pub fn tile_col_start(&self, x: u32) -> u32 {
        *self.col_bd.iter().rfind(|&&b| b <= x).unwrap_or(&0)
    }
}

/// Motion of one 4×4 block. `ref_idx < 0` means the list is unused; both
/// unused means intra (or not yet decoded).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvField {
    pub mv: [[i16; 2]; 2],
    pub ref_idx: [i8; 2],
}

impl MvField {
    pub const INTRA: MvField = MvField {
        mv: [[0; 2]; 2],
        ref_idx: [-1, -1],
    };

    #[inline]
    pub fn pred(&self, l: usize) -> bool {
        self.ref_idx[l] >= 0
    }
}

/// Per-slice facts later stages need.
#[derive(Clone, Debug)]
pub struct SliceInfo {
    pub addr_rs: u32,
    pub deblocking_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub lf_across_slices: bool,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    /// `(POC, long-term)` of each reference, per list — a colocated
    /// picture's motion is read through these.
    pub refs: [Vec<(i32, bool)>; 2],
}

/// SAO parameters of one CTB, one component.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sao {
    /// 0 off, 1 band, 2 edge.
    pub kind: u8,
    pub band_or_class: u8,
    pub offsets: [i16; 4],
}

/// Flags per 4×4 block.
pub mod flag {
    pub const INTRA: u8 = 1;
    pub const SKIP: u8 = 2;
    /// PCM with `pcm_loop_filter_disabled`, or `cu_transquant_bypass`: loop
    /// filters leave these samples alone.
    pub const NO_FILTER: u8 = 4;
    pub const PCM: u8 = 8;
    /// Transform block with non-zero luma coefficients.
    pub const CBF: u8 = 16;
    /// Left / top edge of this block is a transform-block edge.
    pub const EDGE_V: u8 = 32;
    pub const EDGE_H: u8 = 64;
}

pub struct PicState {
    pub planes: [Plane; 3],
    pub width: usize,
    pub height: usize,
    pub layout: Arc<Layout>,
    pub poc: i32,
    pub flags: Vec<u8>,
    /// Extra edge flags: bit 0 vertical PU edge, bit 1 horizontal PU edge.
    pub pu_edge: Vec<u8>,
    pub depth: Vec<u8>,
    pub ipm: Vec<u8>,
    pub qp: Vec<i8>,
    pub mv: Vec<MvField>,
    /// Slice index (into `slices`) per CTB, `u16::MAX` until decoded.
    pub ctb_slice: Vec<u16>,
    pub slices: Vec<SliceInfo>,
    pub sao: Vec<[Sao; 3]>,
    pub decoded_ctbs: usize,
}

impl PicState {
    pub fn new(sps: &Sps, layout: Arc<Layout>) -> PicState {
        let (w, h) = (sps.width as usize, sps.height as usize);
        let n4 = layout.w4 * layout.h4;
        let nctb = layout.ctb_w * layout.ctb_h;
        PicState {
            planes: [
                Plane::new(w, h, 1 << (sps.bit_depth_y - 1)),
                Plane::new(w.div_ceil(2), h.div_ceil(2), 1 << (sps.bit_depth_c - 1)),
                Plane::new(w.div_ceil(2), h.div_ceil(2), 1 << (sps.bit_depth_c - 1)),
            ],
            width: w,
            height: h,
            layout,
            poc: 0,
            flags: vec![0; n4],
            pu_edge: vec![0; n4],
            depth: vec![0; n4],
            ipm: vec![1; n4],
            qp: vec![0; n4],
            mv: vec![MvField::INTRA; n4],
            ctb_slice: vec![u16::MAX; nctb],
            slices: Vec::new(),
            sao: vec![[Sao::default(); 3]; nctb],
            decoded_ctbs: 0,
        }
    }

    #[inline]
    pub fn idx4(&self, x: usize, y: usize) -> usize {
        (y >> 2) * self.layout.w4 + (x >> 2)
    }

    #[inline]
    pub fn ctb_rs(&self, x: usize, y: usize) -> usize {
        let s = self.layout.log2_ctb;
        (y >> s) * self.layout.ctb_w + (x >> s)
    }
}
