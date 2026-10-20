//! Per-macroblock state: what neighbours and later pictures read
//! ([`MbInfo`]), and the parsed syntax of the macroblock being decoded
//! ([`MbSyn`]).

use std::sync::Arc;

use crate::codec::decode::picture::Plane;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MbKind {
    #[default]
    None,
    I4x4,
    I8x8,
    I16x16,
    IPcm,
    P,
    B,
    PSkip,
    BSkip,
}

impl MbKind {
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::IPcm)
    }
    #[inline]
    pub fn is_skip(self) -> bool {
        matches!(self, MbKind::PSkip | MbKind::BSkip)
    }
    /// `I_NxN`, whose prediction modes neighbours inherit.
    #[inline]
    pub fn is_nxn(self) -> bool {
        matches!(self, MbKind::I4x4 | MbKind::I8x8)
    }
}

/// What a decoded macroblock leaves behind for its neighbours, the
/// deblocking filter, and pictures that use this one as a colocated
/// reference. 4×4 blocks are indexed in raster order within the macroblock.
#[derive(Clone, Copy, Debug)]
pub struct MbInfo {
    /// Slice number within the picture; `u32::MAX` until decoded.
    pub slice: u32,
    pub kind: MbKind,
    /// `B_Skip` or `B_Direct_16x16` — CABAC context selection asks.
    pub b_direct16: bool,
    /// `QPY` (0 for `I_PCM`, as deblocking wants it).
    pub qp: i8,
    /// `QPC` for Cb and Cr, without the bit-depth offset.
    pub qpc: [i8; 2],
    /// Luma bits 0-3, chroma in bits 4-5 (`CodedBlockPattern`).
    pub cbp: u8,
    pub t8x8: bool,
    pub chroma_mode: u8,
    /// Intra 4×4/8×8 prediction mode per 4×4 block (8×8 modes replicated).
    pub ipred: [u8; 16],
    /// `TotalCoeff` per luma 4×4 block (CAVLC `nC`; CABAC reads `!= 0`).
    pub nnz: [u8; 16],
    /// `TotalCoeff` of chroma AC blocks, `[Cb, Cr][raster 2×2]`.
    pub nnz_c: [[u8; 4]; 2],
    /// `coded_block_flag` of DC blocks: bit 0 luma (Intra 16×16), 1 Cb, 2 Cr.
    pub cbf_dc: u8,
    /// Which 4×4 blocks have coefficients, for deblocking (spread over 8×8
    /// when the transform is 8×8).
    pub nz_mask: u16,
    /// 8×8 blocks whose motion came from direct prediction.
    pub direct8: u8,
    pub ref_idx: [[i8; 4]; 2],
    /// Unique id of the referenced picture per 8×8 block, `-1` for none.
    pub ref_id: [[i32; 4]; 2],
    pub mv: [[[i16; 2]; 16]; 2],
    /// `|mvd|` per 4×4 block, saturated — only ever compared against 32.
    pub mvd: [[[u8; 2]; 16]; 2],
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo {
            slice: u32::MAX,
            kind: MbKind::None,
            b_direct16: false,
            qp: 0,
            qpc: [0; 2],
            cbp: 0,
            t8x8: false,
            chroma_mode: 0,
            ipred: [2; 16],
            nnz: [0; 16],
            nnz_c: [[0; 4]; 2],
            cbf_dc: 0,
            nz_mask: 0,
            direct8: 0,
            ref_idx: [[-1; 4]; 2],
            ref_id: [[-1; 4]; 2],
            mv: [[[0; 2]; 16]; 2],
            mvd: [[[0; 2]; 16]; 2],
        }
    }
}

/// The parsed syntax of one macroblock, before reconstruction.
pub struct MbSyn {
    pub kind: MbKind,
    /// Raw `mb_type` within its slice-type table (P 0-4, B 0-22, I 0-25).
    pub mb_type: u32,
    pub i16_mode: u8,
    pub prev_flag: [bool; 16],
    pub rem: [u8; 16],
    pub chroma_mode: u8,
    pub sub: [u8; 4],
    pub ref_idx: [[i8; 4]; 2],
    /// `[list][raster 4×4] = (x, y)`, filled over each partition's area.
    pub mvd: [[[i32; 2]; 16]; 2],
    pub cbp: u8,
    pub t8x8: bool,
    pub qp_delta: i32,
    pub pcm: Vec<u16>,
    /// Levels per 4×4 block by `blkIdx`, raster positions within the block.
    pub coef: [[i32; 16]; 16],
    /// 8×8 levels per 8×8 block, raster positions.
    pub coef8: [[i32; 64]; 4],
    /// Intra 16×16 DC levels, raster over the 4×4 grid of blocks.
    pub dc_luma: [i32; 16],
    pub dc_c: [[i32; 4]; 2],
    /// Chroma AC levels, `[Cb, Cr][blkIdx][raster]`.
    pub ac_c: [[[i32; 16]; 4]; 2],
    /// Luma 4×4 blocks (by `blkIdx`) with any level.
    pub coded: u16,
    /// 8×8 blocks with any level.
    pub coded8: u8,
    pub coded_c: [u8; 2],
}

impl MbSyn {
    pub fn new() -> Self {
        MbSyn {
            kind: MbKind::None,
            mb_type: 0,
            i16_mode: 0,
            prev_flag: [false; 16],
            rem: [0; 16],
            chroma_mode: 0,
            sub: [0; 4],
            ref_idx: [[0; 4]; 2],
            mvd: [[[0; 2]; 16]; 2],
            cbp: 0,
            t8x8: false,
            qp_delta: 0,
            pcm: vec![0; 384],
            coef: [[0; 16]; 16],
            coef8: [[0; 64]; 4],
            dc_luma: [0; 16],
            dc_c: [[0; 4]; 2],
            ac_c: [[[0; 16]; 4]; 2],
            coded: 0,
            coded8: 0,
            coded_c: [0; 2],
        }
    }

    /// Clear for the next macroblock. Residual arrays are cleared lazily:
    /// only the blocks the previous macroblock wrote.
    pub fn reset(&mut self) {
        for b in 0..16 {
            if self.coded & (1 << b) != 0 {
                self.coef[b] = [0; 16];
            }
        }
        for b in 0..4 {
            if self.coded8 & (1 << b) != 0 {
                self.coef8[b] = [0; 64];
            }
        }
        self.dc_luma = [0; 16];
        self.dc_c = [[0; 4]; 2];
        for c in 0..2 {
            if self.coded_c[c] != 0 {
                self.ac_c[c] = [[0; 16]; 4];
            }
        }
        self.coded = 0;
        self.coded8 = 0;
        self.coded_c = [0; 2];
        self.kind = MbKind::None;
        self.mb_type = 0;
        self.i16_mode = 0;
        self.prev_flag = [false; 16];
        self.rem = [0; 16];
        self.chroma_mode = 0;
        self.sub = [0; 4];
        self.ref_idx = [[0; 4]; 2];
        self.mvd = [[[0; 2]; 16]; 2];
        self.cbp = 0;
        self.t8x8 = false;
        self.qp_delta = 0;
    }
}

/// Partition shape of a non-8×8 inter macroblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    S16x16,
    S16x8,
    S8x16,
    S8x8,
}

impl Shape {
    /// Partitions as `(x, y, w, h)` in 4×4 units.
    pub fn parts(self) -> &'static [(usize, usize, usize, usize)] {
        match self {
            Shape::S16x16 => &[(0, 0, 4, 4)],
            Shape::S16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
            Shape::S8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
            Shape::S8x8 => &[(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
        }
    }
}

/// Prediction-list use: bit 0 L0, bit 1 L1.
pub type PredFlags = u8;

/// `(shape, [pred per partition])` for a P `mb_type` 0-4.
pub fn p_mb_layout(t: u32) -> (Shape, [PredFlags; 2]) {
    match t {
        0 => (Shape::S16x16, [1, 0]),
        1 => (Shape::S16x8, [1, 1]),
        2 => (Shape::S8x16, [1, 1]),
        _ => (Shape::S8x8, [1, 1]),
    }
}

/// `(shape, [pred per partition])` for a B `mb_type` 1-22.
pub fn b_mb_layout(t: u32) -> (Shape, [PredFlags; 2]) {
    const T: [(Shape, [u8; 2]); 21] = [
        (Shape::S16x16, [1, 0]),
        (Shape::S16x16, [2, 0]),
        (Shape::S16x16, [3, 0]),
        (Shape::S16x8, [1, 1]),
        (Shape::S8x16, [1, 1]),
        (Shape::S16x8, [2, 2]),
        (Shape::S8x16, [2, 2]),
        (Shape::S16x8, [1, 2]),
        (Shape::S8x16, [1, 2]),
        (Shape::S16x8, [2, 1]),
        (Shape::S8x16, [2, 1]),
        (Shape::S16x8, [1, 3]),
        (Shape::S8x16, [1, 3]),
        (Shape::S16x8, [2, 3]),
        (Shape::S8x16, [2, 3]),
        (Shape::S16x8, [3, 1]),
        (Shape::S8x16, [3, 1]),
        (Shape::S16x8, [3, 2]),
        (Shape::S8x16, [3, 2]),
        (Shape::S16x8, [3, 3]),
        (Shape::S8x16, [3, 3]),
    ];
    if (1..=21).contains(&t) {
        T[t as usize - 1]
    } else {
        (Shape::S8x8, [0, 0])
    }
}

/// `(pred, w, h)` in 4×4 units for a P `sub_mb_type`.
pub fn p_sub_layout(t: u8) -> (PredFlags, usize, usize) {
    match t {
        0 => (1, 2, 2),
        1 => (1, 2, 1),
        2 => (1, 1, 2),
        _ => (1, 1, 1),
    }
}

/// `(pred, w, h)` for a B `sub_mb_type`; pred 0 is `B_Direct_8x8`.
pub fn b_sub_layout(t: u8) -> (PredFlags, usize, usize) {
    match t {
        0 => (0, 2, 2),
        1 => (1, 2, 2),
        2 => (2, 2, 2),
        3 => (3, 2, 2),
        4 => (1, 2, 1),
        5 => (1, 1, 2),
        6 => (2, 2, 1),
        7 => (2, 1, 2),
        8 => (3, 2, 1),
        9 => (3, 1, 2),
        10 => (1, 1, 1),
        11 => (2, 1, 1),
        _ => (3, 1, 1),
    }
}

/// A reference picture held for inter prediction.
pub struct RefPic {
    pub id: u32,
    pub planes: Arc<[Plane; 3]>,
    /// Motion of every macroblock — for colocated (direct) prediction.
    pub mbs: Arc<Vec<MbInfo>>,
    pub poc: i32,
    pub frame_num: u32,
    pub frame_num_wrap: i32,
    pub long_term: bool,
    pub lt_idx: u32,
}

/// One entry of a reference picture list.
#[derive(Clone, Copy, Debug)]
pub struct RefEntry {
    /// Index into the decoder's reference store.
    pub dpb: usize,
    pub id: u32,
    pub poc: i32,
    pub long_term: bool,
}
