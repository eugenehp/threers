//! VP9 decoder: a port of ffmpeg 7.1's native `vp9` decoder (`vp9.c`,
//! `vp9block.c`, `vp9recon.c`, `vp9mvs.c`, `vp9prob.c`, `vp9lpf.c`,
//! `vp9dsp_template.c`), single-threaded path, every profile (8/10/12-bit;
//! 4:2:0, 4:2:2, 4:4:0, 4:4:4, RGB), reference scaling, show-existing and
//! intra-only frames, and superframes.
//!
//! VP9 decoding is integer and normative, and ffmpeg's decoder matches
//! libvpx; porting ffmpeg's code keeps its libvpx-compatibility quirks
//! (each marked where it appears). Planes are `u16` for every bit depth and
//! allocated superblock-aligned; the frame is cropped on output. The whole
//! frame is decoded before the loop filter runs superblock row by row, which
//! is what ffmpeg's per-row filtering plus its `intra_pred_data` backup of
//! unfiltered rows amounts to.

pub mod block;
pub mod data;
pub mod dsp;
pub mod lpf;
pub mod mvs;
pub mod probs;
pub mod rac;
pub mod recon;

use std::sync::Arc;

use self::probs::{Counts, ProbContext, SavedCtx};
use self::rac::Rac;
use crate::decode::bits::BitReader;
pub use crate::decode::vp9_packet::{packet_shows, split_superframe};
use crate::decode::picture::Plane;

// IntraPredMode (ffmpeg order).
/// Intra prediction mode: Vertical.
pub const VERT_PRED: usize = 0;
/// Intra prediction mode: Horizontal.
pub const HOR_PRED: usize = 1;
/// Intra prediction mode: DC.
pub const DC_PRED: usize = 2;
/// Intra prediction mode: Diagonal down-left.
pub const DIAG_DOWN_LEFT_PRED: usize = 3;
/// Intra prediction mode: Diagonal down-right.
pub const DIAG_DOWN_RIGHT_PRED: usize = 4;
/// Intra prediction mode: Vertical-right.
pub const VERT_RIGHT_PRED: usize = 5;
/// Intra prediction mode: Horizontal-down.
pub const HOR_DOWN_PRED: usize = 6;
/// Intra prediction mode: Vertical-left.
pub const VERT_LEFT_PRED: usize = 7;
/// Intra prediction mode: Horizontal-up.
pub const HOR_UP_PRED: usize = 8;
/// Intra prediction mode: True motion (VP8's TM).
pub const TM_VP8_PRED: usize = 9;
/// Intra prediction mode: DC from the left edge only.
pub const LEFT_DC_PRED: usize = 10;
/// Intra prediction mode: DC from the top edge only.
pub const TOP_DC_PRED: usize = 11;
/// Intra prediction mode: DC of 128 (no edges).
pub const DC_128_PRED: usize = 12;
/// Intra prediction mode: DC of 127 (no top edge).
pub const DC_127_PRED: usize = 13;
/// Intra prediction mode: DC of 129 (no left edge).
pub const DC_129_PRED: usize = 14;
// InterPredMode.
/// Inter prediction mode: Nearest MV.
pub const NEARESTMV: u8 = 10;
/// Inter prediction mode: Near MV.
pub const NEARMV: u8 = 11;
/// Inter prediction mode: Zero MV.
pub const ZEROMV: u8 = 12;
/// Inter prediction mode: New MV.
pub const NEWMV: u8 = 13;
// TxfmMode.
/// Transform mode: 4×4 only.
pub const TX_4X4: u8 = 0;
/// Transform mode: Up to 8×8.
pub const TX_8X8: u8 = 1;
/// Transform mode: Up to 16×16.
pub const TX_16X16: u8 = 2;
/// Transform mode: Up to 32×32.
pub const TX_32X32: u8 = 3;
/// Transform mode: Chosen per block.
pub const TX_SWITCHABLE: u8 = 4;
// TxfmType.
/// Transform type: DCT both ways.
pub const DCT_DCT: usize = 0;
// BlockSize.
/// Block size 64×64.
pub const BS_64X64: u8 = 0;
/// Block size 8×8.
pub const BS_8X8: u8 = 9;
/// Block size 8×4.
pub const BS_8X4: u8 = 10;
/// Block size 4×8.
pub const BS_4X8: u8 = 11;
/// Block size 4×4.
pub const BS_4X4: u8 = 12;
// BlockLevel.
/// Partition level 64×64.
pub const BL_64X64: u8 = 0;
/// Partition level 8×8.
pub const BL_8X8: u8 = 3;
// BlockPartition.
/// Partition: None.
pub const PARTITION_NONE: u8 = 0;
/// Partition: Horizontal.
pub const PARTITION_H: u8 = 1;
/// Partition: Vertical.
pub const PARTITION_V: u8 = 2;
/// Partition: Split into four.
pub const PARTITION_SPLIT: u8 = 3;

/// `FilterMode` (block filter values 0-3; 4 = switchable in the header).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// 8-tap smooth.
    Smooth = 0,
    /// 8-tap regular.
    #[default]
    Regular = 1,
    /// 8-tap sharp.
    Sharp = 2,
    /// Bilinear.
    Bilinear = 3,
    /// Chosen per block.
    Switchable = 4,
}

impl FilterMode {
    fn from_bits(v: u32) -> Self {
        [FilterMode::Smooth, FilterMode::Regular, FilterMode::Sharp, FilterMode::Bilinear][v as usize & 3]
    }
}

/// `CompPredMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompPred {
    /// Single reference only.
    #[default]
    Single,
    /// Compound prediction only.
    Comp,
    /// Chosen per block.
    Switchable,
}

/// A motion vector in eighth samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    /// Horizontal.
    pub x: i16,
    /// Vertical.
    pub y: i16,
}

impl Mv {
    #[inline]
    fn packed(self) -> u32 {
        (self.x as u16 as u32) | ((self.y as u16 as u32) << 16)
    }
}

/// A block's motion as later frames read it (two lists).
#[derive(Clone, Copy, Debug, Default)]
pub struct MvRefPair {
    /// Vector per list.
    pub mv: [Mv; 2],
    /// Reference per list, `-1` for none.
    pub r#ref: [i8; 2],
}

/// One segment's features and what they resolve to.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegFeat {
    /// Alternate quantizer enabled.
    pub q_enabled: bool,
    /// Alternate loop filter level enabled.
    pub lf_enabled: bool,
    /// Fixed reference enabled.
    pub ref_enabled: bool,
    /// Skip enabled.
    pub skip_enabled: bool,
    /// The fixed reference.
    pub ref_val: u8,
    /// The quantizer value (delta or absolute).
    pub q_val: i16,
    /// The loop filter value (delta or absolute).
    pub lf_val: i8,
    /// Dequantizers: `[luma, chroma][dc, ac]`.
    pub qmul: [[i16; 2]; 2],
    /// Loop filter level per reference `[ref][mode]` (intra, last, golden, altref).
    pub lflvl: [[u8; 2]; 4],
}

/// `segmentation_params()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Segmentation {
    /// `segmentation_enabled`.
    pub enabled: bool,
    /// `segmentation_temporal_update`.
    pub temporal: bool,
    /// `segmentation_abs_or_delta_update`.
    pub absolute_vals: bool,
    /// `segmentation_update_map`.
    pub update_map: bool,
    /// `segmentation_tree_probs`.
    pub prob: [u8; 7],
    /// `segmentation_pred_prob`.
    pub pred_prob: [u8; 3],
    /// Per segment.
    pub feat: [SegFeat; 8],
}

/// `VP9BitstreamHeader`. It persists across frames: fields a frame does not
/// code keep the previous frame's values, as in ffmpeg.
#[derive(Clone, Debug, Default)]
pub struct Header {
    /// `profile`.
    pub profile: u8,
    /// Bits per sample.
    pub bpp: u8,
    /// `frame_type == KEY_FRAME`.
    pub keyframe: bool,
    /// `!show_frame`.
    pub invisible: bool,
    /// `error_resilient_mode`.
    pub errorres: bool,
    /// `intra_only`.
    pub intraonly: bool,
    /// `reset_frame_context`.
    pub resetctx: u8,
    /// `refresh_frame_flags`.
    pub refreshrefmask: u8,
    /// `allow_high_precision_mv`.
    pub highprecisionmvs: bool,
    /// `interpolation_filter`.
    pub filtermode: FilterMode,
    /// Compound prediction is possible (sign biases differ).
    pub allowcompinter: bool,
    /// `refresh_frame_context`.
    pub refreshctx: bool,
    /// `frame_parallel_decoding_mode`.
    pub parallelmode: bool,
    /// `frame_context_idx`.
    pub framectxid: u8,
    /// The previous frame's MVs are usable.
    pub use_last_frame_mvs: bool,
    /// `ref_frame_idx` for last, golden, altref.
    pub refidx: [u8; 3],
    /// `ref_frame_sign_bias` for the same.
    pub signbias: [bool; 3],
    /// The fixed reference of compound prediction.
    pub fixcompref: u8,
    /// Its variable references.
    pub varcompref: [u8; 2],
    /// `loop_filter_level`.
    pub filter_level: u8,
    /// `loop_filter_sharpness`.
    pub sharpness: i8,
    /// `loop_filter_delta_enabled`.
    pub lf_delta_enabled: bool,
    /// `loop_filter_delta_update`.
    pub lf_delta_updated: bool,
    /// `loop_filter_mode_deltas`.
    pub lf_delta_mode: [i8; 2],
    /// `loop_filter_ref_deltas`.
    pub lf_delta_ref: [i8; 4],
    /// `base_q_idx`.
    pub yac_qi: u8,
    /// `delta_q_y_dc`.
    pub ydc_qdelta: i8,
    /// `delta_q_uv_dc`.
    pub uvdc_qdelta: i8,
    /// `delta_q_uv_ac`.
    pub uvac_qdelta: i8,
    /// Lossless: every quantizer 0.
    pub lossless: bool,
    /// `segmentation_params()`.
    pub segmentation: Segmentation,
    /// `tx_mode`.
    pub txfmmode: u8,
    /// `reference_select`.
    pub comppredmode: CompPred,
    /// `tile_cols_log2`.
    pub log2_tile_cols: u32,
    /// `tile_rows_log2`.
    pub log2_tile_rows: u32,
    /// Tile columns.
    pub tile_cols: u32,
    /// Tile rows.
    pub tile_rows: u32,
}

/// A decoded frame (a reference, or the output).
pub struct Frame {
    /// Superblock-aligned planes; the visible picture is `width × height`.
    pub planes: [Plane; 3],
    /// Visible width.
    pub width: u32,
    /// Visible height.
    pub height: u32,
    /// Horizontal chroma subsampling (log2).
    pub ss_h: u8,
    /// Vertical chroma subsampling (log2).
    pub ss_v: u8,
    /// Bits per sample.
    pub bpp: u8,
    /// VP9 `color_space` (0 unknown, 1 BT.601, 2 BT.709, 3 SMPTE-170,
    /// 4 SMPTE-240, 5 BT.2020, 6 reserved, 7 sRGB) and full range.
    pub color_space: u8,
    /// Full-range samples.
    pub full_range: bool,
    /// Segment id per 8×8 block, row stride `8 * sb_cols`.
    pub segmap: Vec<u8>,
    /// Motion per 8×8 block, as later frames read it.
    pub mv: Vec<MvRefPair>,
    /// Width in superblocks.
    pub sb_cols: usize,
}

impl Frame {
    /// As a [`DecodedFrame`](crate::decode::DecodedFrame): planes
    /// cropped, VP9's `color_space` as the matrix number ffmpeg reports
    /// (`AVColorSpace`; 0 for RGB, whose planes are G, B, R). 4:4:0 has no
    /// [`ChromaFormat`](crate::decode::ChromaFormat) and is refused.
    pub fn to_decoded(&self) -> Result<crate::decode::DecodedFrame, String> {
        use crate::decode::{ChromaFormat, ColorInfo, DecodedFrame};
        let chroma = match (self.ss_h, self.ss_v) {
            (1, 1) => ChromaFormat::Yuv420,
            (1, 0) => ChromaFormat::Yuv422,
            (0, 0) => ChromaFormat::Yuv444,
            _ => ChromaFormat::Yuv440,
        };
        const SPC: [u32; 8] = [2, 5, 1, 6, 7, 9, 2, 0];
        let (y, _, _) = self.cropped(0);
        let (u, _, _) = self.cropped(1);
        let (v, _, _) = self.cropped(2);
        Ok(DecodedFrame {
            width: self.width,
            height: self.height,
            bit_depth: self.bpp,
            chroma,
            y,
            u,
            v,
            color: ColorInfo::from_vui(Some(SPC[self.color_space as usize & 7]), self.full_range, self.height),
        })
    }

    /// The visible luma/chroma planes, tightly packed.
    pub fn cropped(&self, p: usize) -> (Vec<u16>, usize, usize) {
        let (w, h) = if p == 0 {
            (self.width as usize, self.height as usize)
        } else {
            (
                (self.width as usize + self.ss_h as usize) >> self.ss_h,
                (self.height as usize + self.ss_v as usize) >> self.ss_v,
            )
        };
        let pl = &self.planes[p];
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend_from_slice(&pl.data[y * pl.width..y * pl.width + w]);
        }
        (out, w, h)
    }
}

/// A hardware back end for [`Vp9Decoder`]: the decoder parses the
/// uncompressed header and keeps the eight reference slots; the accelerator
/// parses the compressed header, adapts probabilities and reconstructs.
pub trait Vp9Accel {
    /// Decode one frame.
    fn frame(&mut self, f: &AccelFrame<'_>) -> Result<(), String>;
    /// The samples of a frame it decoded.
    fn read(&mut self, frame: &Frame) -> Result<crate::decode::DecodedFrame, String>;
}

/// Everything a [`Vp9Accel`] is told about one frame.
pub struct AccelFrame<'a> {
    /// The frame header.
    pub h: &'a Header,
    /// The whole frame: uncompressed header, compressed header, tiles.
    pub data: &'a [u8],
    /// Bytes of the uncompressed header.
    pub uncompressed_size: usize,
    /// Bytes of the compressed header.
    pub compressed_size: usize,
    /// The frame being decoded (sizes and format; no samples).
    pub cur: &'a Arc<Frame>,
    /// The reference slots as the frame finds them.
    pub refs: &'a [Option<Arc<Frame>>; 8],
}

/// The decoder.
pub struct Vp9Decoder {
    h: Header,
    refs: [Option<Arc<Frame>>; 8],
    /// The previous decoded frame (ffmpeg `CUR_FRAME` before allocation),
    /// and the frames whose segmentation map / motion the next frame reads.
    last: Option<Arc<Frame>>,
    segmap_ref: Option<Arc<Frame>>,
    prob_ctx: [SavedCtx; 4],
    last_keyframe: bool,
    lim_lut: [u8; 64],
    mblim_lut: [u8; 64],
    ss_h: u8,
    ss_v: u8,
    bpp_index: usize,
    color_space: u8,
    full_range: bool,
    mvscale: [[u16; 2]; 3],
    mvstep: [[u8; 2]; 3],
    accel: Option<Box<dyn Vp9Accel>>,
    /// Header sizes of the frame being decoded, for the accelerator.
    header_sizes: (usize, usize),
}

impl Default for Vp9Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// `inv_recenter_nonneg`.
fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

const INV_MAP_TABLE: [u8; 255] = [
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1, 2,
    3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
    148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230,
    231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250,
    251, 252, 253, 253,
];

/// `update_prob`: a differential forward probability update.
fn update_prob(c: &mut Rac, p: u8) -> u8 {
    let mut d = if !c.get() {
        c.get_uint(4) as usize
    } else if !c.get() {
        c.get_uint(4) as usize + 16
    } else if !c.get() {
        c.get_uint(5) as usize + 32
    } else {
        let mut d = c.get_uint(7) as usize;
        if d >= 65 {
            d = (d << 1) - 65 + c.get() as usize;
        }
        d + 64
    };
    d = d.min(254);
    let p = p as i32;
    let v = if p <= 128 {
        1 + inv_recenter_nonneg(INV_MAP_TABLE[d] as i32, p - 1)
    } else {
        255 - inv_recenter_nonneg(INV_MAP_TABLE[d] as i32, 255 - p)
    };
    v as u8
}

fn sbits_inv(r: &mut BitReader, n: u32) -> i32 {
    let v = r.u(n) as i32;
    if r.flag() {
        -v
    } else {
        v
    }
}

/// `av_clip_uintp2(v, n)`.
fn clip_u(v: i32, n: u32) -> i32 {
    v.clamp(0, (1 << n) - 1)
}


impl Vp9Decoder {
    /// A decoder waiting for a keyframe.
    pub fn new() -> Self {
        Vp9Decoder {
            h: Header { sharpness: -1, ..Default::default() },
            refs: Default::default(),
            last: None,
            segmap_ref: None,
            prob_ctx: [SavedCtx::default_ctx(), SavedCtx::default_ctx(), SavedCtx::default_ctx(), SavedCtx::default_ctx()],
            last_keyframe: false,
            lim_lut: [0; 64],
            mblim_lut: [0; 64],
            ss_h: 1,
            ss_v: 1,
            bpp_index: 0,
            color_space: 0,
            full_range: false,
            mvscale: [[0; 2]; 3],
            mvstep: [[0; 2]; 3],
            accel: None,
            header_sizes: (0, 0),
        }
    }

    /// Decode one packet (a frame or superframe); returns the frames to show.
    /// Hand reconstruction to `accel` (see [`Vp9Accel`]).
    pub fn set_accel(&mut self, accel: Box<dyn Vp9Accel>) {
        self.accel = Some(accel);
    }

    /// The samples of `frame`, from the accelerator that decoded it.
    pub fn accel_read(&mut self, frame: &Frame) -> Result<crate::decode::DecodedFrame, String> {
        self.accel.as_mut().ok_or("no accelerator")?.read(frame)
    }

    /// Decode one packet (a frame or a superframe) and return the frames it shows.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<Arc<Frame>>, String> {
        let mut out = Vec::new();
        for f in split_superframe(packet) {
            if let Some(frame) = self.decode_frame(f)? {
                out.push(frame);
            }
        }
        Ok(out)
    }

    fn read_colorspace_details(&mut self, r: &mut BitReader) -> Result<(), String> {
        let bits = if self.h.profile <= 1 { 0 } else { 1 + r.flag() as usize };
        self.bpp_index = bits;
        self.h.bpp = 8 + bits as u8 * 2;
        self.color_space = r.u(3) as u8;
        if self.color_space == 7 {
            // RGB (profile 1/3 only).
            self.ss_h = 0;
            self.ss_v = 0;
            self.full_range = true;
            if self.h.profile & 1 != 0 {
                if r.flag() {
                    return Err("VP9: reserved bit set in RGB".into());
                }
            } else {
                return Err(format!("VP9: RGB not supported in profile {}", self.h.profile));
            }
        } else {
            self.full_range = r.flag();
            if self.h.profile & 1 != 0 {
                self.ss_h = r.flag() as u8;
                self.ss_v = r.flag() as u8;
                if self.ss_h == 1 && self.ss_v == 1 {
                    return Err(format!("VP9: 4:2:0 not supported in profile {}", self.h.profile));
                } else if r.flag() {
                    return Err("VP9: color details reserved bit set".into());
                }
            } else {
                self.ss_h = 1;
                self.ss_v = 1;
            }
        }
        Ok(())
    }

    /// `decode_frame_header`: `Ok(None)` after the compressed header (then
    /// `(offset of tile data)`), `Ok(Some(ref))` for show-existing.
    fn decode_frame_header(&mut self, data: &[u8], probs: &mut ProbContext, coef: &mut CoefProbs, w_out: &mut (u32, u32)) -> Result<Result<usize, u8>, String> {
        let mut r = BitReader::new(data);
        if r.u(2) != 2 {
            return Err("VP9: invalid frame marker".into());
        }
        let mut profile = r.u(1) as u8;
        profile |= (r.u(1) as u8) << 1;
        if profile == 3 {
            profile += r.u(1) as u8;
        }
        if profile > 3 {
            return Err(format!("VP9: profile {profile} is not supported"));
        }
        self.h.profile = profile;
        if r.flag() {
            return Ok(Err(r.u(3) as u8));
        }
        self.last_keyframe = self.h.keyframe;
        self.h.keyframe = !r.flag();
        let last_invisible = self.h.invisible;
        self.h.invisible = !r.flag();
        self.h.errorres = r.flag();
        self.h.use_last_frame_mvs = !self.h.errorres && !last_invisible;
        let (w, h);
        if self.h.keyframe {
            if r.u(24) != 0x49_8342 {
                return Err("VP9: invalid sync code".into());
            }
            self.read_colorspace_details(&mut r)?;
            self.h.refreshrefmask = 0xff;
            w = r.u(16) + 1;
            h = r.u(16) + 1;
            if r.flag() {
                r.skip(32);
            }
        } else {
            self.h.intraonly = if self.h.invisible { r.flag() } else { false };
            self.h.resetctx = if self.h.errorres { 0 } else { r.u(2) as u8 };
            if self.h.intraonly {
                if r.u(24) != 0x49_8342 {
                    return Err("VP9: invalid sync code".into());
                }
                if profile >= 1 {
                    self.read_colorspace_details(&mut r)?;
                } else {
                    self.ss_h = 1;
                    self.ss_v = 1;
                    self.h.bpp = 8;
                    self.bpp_index = 0;
                    self.color_space = 1;
                    self.full_range = false;
                }
                self.h.refreshrefmask = r.u(8) as u8;
                w = r.u(16) + 1;
                h = r.u(16) + 1;
                if r.flag() {
                    r.skip(32);
                }
            } else {
                self.h.refreshrefmask = r.u(8) as u8;
                for i in 0..3 {
                    self.h.refidx[i] = r.u(3) as u8;
                    self.h.signbias[i] = r.flag() && !self.h.errorres;
                }
                if self.h.refidx.iter().any(|&i| self.refs[i as usize].is_none()) {
                    return Err("VP9: not all references are available".into());
                }
                let rs = |i: usize| {
                    let f = self.refs[self.h.refidx[i] as usize].as_ref().unwrap();
                    (f.width, f.height)
                };
                if r.flag() {
                    (w, h) = rs(0);
                } else if r.flag() {
                    (w, h) = rs(1);
                } else if r.flag() {
                    (w, h) = rs(2);
                } else {
                    w = r.u(16) + 1;
                    h = r.u(16) + 1;
                }
                self.h.use_last_frame_mvs &= self.last.as_ref().is_some_and(|f| f.width == w && f.height == h);
                if r.flag() {
                    r.skip(32);
                }
                self.h.highprecisionmvs = r.flag();
                self.h.filtermode = if r.flag() { FilterMode::Switchable } else { FilterMode::from_bits(r.u(2)) };
                let sb = self.h.signbias;
                self.h.allowcompinter = sb[0] != sb[1] || sb[0] != sb[2];
                if self.h.allowcompinter {
                    if sb[0] == sb[1] {
                        self.h.fixcompref = 2;
                        self.h.varcompref = [0, 1];
                    } else if sb[0] == sb[2] {
                        self.h.fixcompref = 1;
                        self.h.varcompref = [0, 2];
                    } else {
                        self.h.fixcompref = 0;
                        self.h.varcompref = [1, 2];
                    }
                }
            }
        }
        self.h.refreshctx = if self.h.errorres { false } else { r.flag() };
        self.h.parallelmode = if self.h.errorres { true } else { r.flag() };
        let c = r.u(2) as usize;
        self.h.framectxid = c as u8;
        if self.h.keyframe || self.h.intraonly {
            // BUG: libvpx ignores this field in keyframes.
            self.h.framectxid = 0;
        }
        if self.h.keyframe || self.h.errorres || self.h.intraonly {
            self.h.lf_delta_ref = [1, 0, -1, -1];
            self.h.lf_delta_mode = [0, 0];
            self.h.segmentation.feat = [SegFeat::default(); 8];
        }
        self.h.filter_level = r.u(6) as u8;
        let sharp = r.u(3) as i8;
        if self.h.sharpness != sharp {
            for i in 1..=63 {
                let mut limit = i as i32;
                if sharp > 0 {
                    limit >>= (sharp as i32 + 3) >> 2;
                    limit = limit.min(9 - sharp as i32);
                }
                limit = limit.max(1);
                self.lim_lut[i] = limit as u8;
                self.mblim_lut[i] = (2 * (i as i32 + 2) + limit) as u8;
            }
        }
        self.h.sharpness = sharp;
        self.h.lf_delta_enabled = r.flag();
        if self.h.lf_delta_enabled {
            self.h.lf_delta_updated = r.flag();
            if self.h.lf_delta_updated {
                for i in 0..4 {
                    if r.flag() {
                        self.h.lf_delta_ref[i] = sbits_inv(&mut r, 6) as i8;
                    }
                }
                for i in 0..2 {
                    if r.flag() {
                        self.h.lf_delta_mode[i] = sbits_inv(&mut r, 6) as i8;
                    }
                }
            }
        }
        self.h.yac_qi = r.u(8) as u8;
        self.h.ydc_qdelta = if r.flag() { sbits_inv(&mut r, 4) as i8 } else { 0 };
        self.h.uvdc_qdelta = if r.flag() { sbits_inv(&mut r, 4) as i8 } else { 0 };
        self.h.uvac_qdelta = if r.flag() { sbits_inv(&mut r, 4) as i8 } else { 0 };
        self.h.lossless = self.h.yac_qi == 0 && self.h.ydc_qdelta == 0 && self.h.uvdc_qdelta == 0 && self.h.uvac_qdelta == 0;
        let seg = &mut self.h.segmentation;
        seg.enabled = r.flag();
        if seg.enabled {
            seg.update_map = r.flag();
            if seg.update_map {
                for i in 0..7 {
                    seg.prob[i] = if r.flag() { r.u(8) as u8 } else { 255 };
                }
                seg.temporal = r.flag();
                if seg.temporal {
                    for i in 0..3 {
                        seg.pred_prob[i] = if r.flag() { r.u(8) as u8 } else { 255 };
                    }
                }
            }
            if r.flag() {
                seg.absolute_vals = r.flag();
                for i in 0..8 {
                    let f = &mut seg.feat[i];
                    f.q_enabled = r.flag();
                    if f.q_enabled {
                        f.q_val = sbits_inv(&mut r, 8) as i16;
                    }
                    f.lf_enabled = r.flag();
                    if f.lf_enabled {
                        f.lf_val = sbits_inv(&mut r, 6) as i8;
                    }
                    f.ref_enabled = r.flag();
                    if f.ref_enabled {
                        f.ref_val = r.u(2) as u8;
                    }
                    f.skip_enabled = r.flag();
                }
            }
        } else {
            seg.temporal = false;
            seg.update_map = false;
        }
        // qmul / lflvl per segment.
        let nseg = if self.h.segmentation.enabled { 8 } else { 1 };
        for i in 0..nseg {
            let h = &mut self.h;
            let f = h.segmentation.feat[i];
            let mut qyac = if h.segmentation.enabled && f.q_enabled {
                if h.segmentation.absolute_vals {
                    clip_u(f.q_val as i32, 8)
                } else {
                    clip_u(h.yac_qi as i32 + f.q_val as i32, 8)
                }
            } else {
                h.yac_qi as i32
            };
            let qydc = clip_u(qyac + h.ydc_qdelta as i32, 8) as usize;
            let quvdc = clip_u(qyac + h.uvdc_qdelta as i32, 8) as usize;
            let quvac = clip_u(qyac + h.uvac_qdelta as i32, 8) as usize;
            qyac = clip_u(qyac, 8);
            let b = self.bpp_index;
            let fm = &mut h.segmentation.feat[i];
            fm.qmul[0][0] = data::DC_QLOOKUP[b][qydc];
            fm.qmul[0][1] = data::AC_QLOOKUP[b][qyac as usize];
            fm.qmul[1][0] = data::DC_QLOOKUP[b][quvdc];
            fm.qmul[1][1] = data::AC_QLOOKUP[b][quvac];
            let sh = (h.filter_level >= 32) as i32;
            let lflvl = if h.segmentation.enabled && f.lf_enabled {
                if h.segmentation.absolute_vals {
                    clip_u(f.lf_val as i32, 6)
                } else {
                    clip_u(h.filter_level as i32 + f.lf_val as i32, 6)
                }
            } else {
                h.filter_level as i32
            };
            let fm = &mut h.segmentation.feat[i];
            if h.lf_delta_enabled {
                let v = clip_u(lflvl + (h.lf_delta_ref[0] as i32) * (1 << sh), 6) as u8;
                fm.lflvl[0] = [v, v];
                for j in 1..4 {
                    fm.lflvl[j][0] = clip_u(lflvl + ((h.lf_delta_ref[j] as i32 + h.lf_delta_mode[0] as i32) * (1 << sh)), 6) as u8;
                    fm.lflvl[j][1] = clip_u(lflvl + ((h.lf_delta_ref[j] as i32 + h.lf_delta_mode[1] as i32) * (1 << sh)), 6) as u8;
                }
            } else {
                fm.lflvl = [[lflvl as u8; 2]; 4];
            }
        }
        // Tiling.
        let sb_cols = (w + 63) >> 6;
        self.h.log2_tile_cols = 0;
        while sb_cols > (64 << self.h.log2_tile_cols) {
            self.h.log2_tile_cols += 1;
        }
        let mut max = 0;
        while (sb_cols >> max) >= 4 {
            max += 1;
        }
        let max = (max - 1).max(0) as u32;
        while max > self.h.log2_tile_cols {
            if r.flag() {
                self.h.log2_tile_cols += 1;
            } else {
                break;
            }
        }
        // decode012
        self.h.log2_tile_rows = if !r.flag() { 0 } else { 1 + r.flag() as u32 };
        self.h.tile_rows = 1 << self.h.log2_tile_rows;
        self.h.tile_cols = 1 << self.h.log2_tile_cols;
        // Reference scaling.
        if !self.h.keyframe && !self.h.intraonly {
            let mut valid = 0;
            for i in 0..3 {
                let rf = self.refs[self.h.refidx[i] as usize].as_ref().unwrap();
                let (refw, refh) = (rf.width, rf.height);
                if rf.bpp != self.h.bpp || rf.ss_h != self.ss_h || rf.ss_v != self.ss_v {
                    return Err("VP9: reference pixel format differs from the frame's".into());
                } else if refw == w && refh == h {
                    self.mvscale[i] = [0, 0];
                } else {
                    if w * 2 < refw || h * 2 < refh || w > 16 * refw || h > 16 * refh {
                        self.mvscale[i] = [0xFFFF, 0xFFFF];
                        continue;
                    }
                    self.mvscale[i] = [((refw << 14) / w) as u16, ((refh << 14) / h) as u16];
                    self.mvstep[i] = [((16 * self.mvscale[i][0] as u32) >> 14) as u8, ((16 * self.mvscale[i][1] as u32) >> 14) as u8];
                }
                valid += 1;
            }
            if valid == 0 {
                return Err("VP9: no valid reference frame".into());
            }
        }
        if self.h.keyframe || self.h.errorres || (self.h.intraonly && self.h.resetctx == 3) {
            for p in self.prob_ctx.iter_mut() {
                *p = SavedCtx::default_ctx();
            }
        } else if self.h.intraonly && self.h.resetctx == 2 {
            self.prob_ctx[c] = SavedCtx::default_ctx();
        }
        let size2 = r.u(16) as usize;
        r.byte_align();
        let off = r.pos() / 8;
        if size2 > data.len().saturating_sub(off) {
            return Err("VP9: invalid compressed header size".into());
        }
        *w_out = (w, h);
        if self.accel.is_some() {
            // The compressed header is the accelerator's to parse.
            self.header_sizes = (off, size2);
            return Ok(Ok(off + size2));
        }
        let mut rc = Rac::new(&data[off..off + size2])?;
        if rc.get_prob(128) {
            return Err("VP9: marker bit set".into());
        }
        let ctx = self.prob_ctx[c].clone();
        *probs = ctx.p;
        // Transform mode.
        if self.h.lossless {
            self.h.txfmmode = TX_4X4;
        } else {
            self.h.txfmmode = rc.get_uint(2) as u8;
            if self.h.txfmmode == 3 {
                self.h.txfmmode += rc.get() as u8;
            }
            if self.h.txfmmode == TX_SWITCHABLE {
                for i in 0..2 {
                    if rc.get_prob(252) {
                        probs.tx8p[i] = update_prob(&mut rc, probs.tx8p[i]);
                    }
                }
                for i in 0..2 {
                    for j in 0..2 {
                        if rc.get_prob(252) {
                            probs.tx16p[i][j] = update_prob(&mut rc, probs.tx16p[i][j]);
                        }
                    }
                }
                for i in 0..2 {
                    for j in 0..3 {
                        if rc.get_prob(252) {
                            probs.tx32p[i][j] = update_prob(&mut rc, probs.tx32p[i][j]);
                        }
                    }
                }
            }
        }
        // Coefficients.
        for i in 0..4 {
            let rf = &ctx.coef[i];
            if rc.get() {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..6 {
                            for m in 0..6 {
                                if m >= 3 && l == 0 {
                                    break;
                                }
                                let p = &mut coef[i][j][k][l][m];
                                let rr = &rf[j][k][l][m];
                                for n in 0..3 {
                                    p[n] = if rc.get_prob(252) { update_prob(&mut rc, rr[n]) } else { rr[n] };
                                }
                                let p2 = p[2] as usize;
                                p[3..11].copy_from_slice(&data::MODEL_PARETO8[p2]);
                            }
                        }
                    }
                }
            } else {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..6 {
                            for m in 0..6 {
                                // (ffmpeg: `m > 3`, so the DC band copies a
                                // fourth, unused context.)
                                if m > 3 && l == 0 {
                                    break;
                                }
                                let p = &mut coef[i][j][k][l][m];
                                p[..3].copy_from_slice(&rf[j][k][l][m]);
                                let p2 = p[2] as usize;
                                p[3..11].copy_from_slice(&data::MODEL_PARETO8[p2]);
                            }
                        }
                    }
                }
            }
            if self.h.txfmmode as usize == i {
                break;
            }
        }
        // Modes.
        for i in 0..3 {
            if rc.get_prob(252) {
                probs.skip[i] = update_prob(&mut rc, probs.skip[i]);
            }
        }
        if !self.h.keyframe && !self.h.intraonly {
            for i in 0..7 {
                for j in 0..3 {
                    if rc.get_prob(252) {
                        probs.mv_mode[i][j] = update_prob(&mut rc, probs.mv_mode[i][j]);
                    }
                }
            }
            if self.h.filtermode == FilterMode::Switchable {
                for i in 0..4 {
                    for j in 0..2 {
                        if rc.get_prob(252) {
                            probs.filter[i][j] = update_prob(&mut rc, probs.filter[i][j]);
                        }
                    }
                }
            }
            for i in 0..4 {
                if rc.get_prob(252) {
                    probs.intra[i] = update_prob(&mut rc, probs.intra[i]);
                }
            }
            if self.h.allowcompinter {
                let mut m = rc.get() as u8;
                if m != 0 {
                    m += rc.get() as u8;
                }
                self.h.comppredmode = [CompPred::Single, CompPred::Comp, CompPred::Switchable][m as usize];
                if self.h.comppredmode == CompPred::Switchable {
                    for i in 0..5 {
                        if rc.get_prob(252) {
                            probs.comp[i] = update_prob(&mut rc, probs.comp[i]);
                        }
                    }
                }
            } else {
                self.h.comppredmode = CompPred::Single;
            }
            if self.h.comppredmode != CompPred::Comp {
                for i in 0..5 {
                    for j in 0..2 {
                        if rc.get_prob(252) {
                            probs.single_ref[i][j] = update_prob(&mut rc, probs.single_ref[i][j]);
                        }
                    }
                }
            }
            if self.h.comppredmode != CompPred::Single {
                for i in 0..5 {
                    if rc.get_prob(252) {
                        probs.comp_ref[i] = update_prob(&mut rc, probs.comp_ref[i]);
                    }
                }
            }
            for i in 0..4 {
                for j in 0..9 {
                    if rc.get_prob(252) {
                        probs.y_mode[i][j] = update_prob(&mut rc, probs.y_mode[i][j]);
                    }
                }
            }
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..3 {
                        if rc.get_prob(252) {
                            probs.partition[3 - i][j][k] = update_prob(&mut rc, probs.partition[3 - i][j][k]);
                        }
                    }
                }
            }
            // mv fields don't use the update_prob subexp model.
            let mvp = |rc: &mut Rac, p: &mut u8| {
                if rc.get_prob(252) {
                    *p = ((rc.get_uint(7) << 1) | 1) as u8;
                }
            };
            for i in 0..3 {
                mvp(&mut rc, &mut probs.mv_joint[i]);
            }
            for i in 0..2 {
                let m = &mut probs.mv_comp[i];
                mvp(&mut rc, &mut m.sign);
                for j in 0..10 {
                    mvp(&mut rc, &mut m.classes[j]);
                }
                mvp(&mut rc, &mut m.class0);
                for j in 0..10 {
                    mvp(&mut rc, &mut m.bits[j]);
                }
            }
            for i in 0..2 {
                let m = &mut probs.mv_comp[i];
                for j in 0..2 {
                    for k in 0..3 {
                        mvp(&mut rc, &mut m.class0_fp[j][k]);
                    }
                }
                for j in 0..3 {
                    mvp(&mut rc, &mut m.fp[j]);
                }
            }
            if self.h.highprecisionmvs {
                for i in 0..2 {
                    mvp(&mut rc, &mut probs.mv_comp[i].class0_hp);
                    mvp(&mut rc, &mut probs.mv_comp[i].hp);
                }
            }
        }
        Ok(Ok(off + size2))
    }

    fn decode_frame(&mut self, data: &[u8]) -> Result<Option<Arc<Frame>>, String> {
        let retain_segmap_ref = self.segmap_ref.is_some() && (!self.h.segmentation.enabled || !self.h.segmentation.update_map);
        let mut probs = data::DEFAULT_PROBS;
        let mut coef: Box<CoefProbs> = Box::new([[[[[[0u8; 11]; 6]; 6]; 2]; 2]; 4]);
        let mut wh = (0, 0);
        let off = match self.decode_frame_header(data, &mut probs, &mut coef, &mut wh)? {
            Err(r) => {
                let f = self.refs[r as usize].clone().ok_or_else(|| format!("VP9: requested reference {r} not available"))?;
                return Ok(Some(f));
            }
            Ok(off) => off,
        };
        let (w, h) = wh;
        let h_ = self.h.clone();
        // Reference frames for segmentation-map prediction and last-frame MVs.
        let src = if !h_.keyframe && !h_.intraonly && !h_.errorres { self.last.clone() } else { None };
        if !retain_segmap_ref || h_.keyframe || h_.intraonly {
            self.segmap_ref = src.clone();
        }
        let mvpair = src;
        let sb_cols = ((w + 63) >> 6) as usize;
        let sb_rows = ((h + 63) >> 6) as usize;
        if mvpair.as_ref().is_none_or(|f| f.width != w || f.height != h) {
            self.segmap_ref = None;
        }
        let (ss_h, ss_v) = (self.ss_h, self.ss_v);
        let lw = sb_cols * 64;
        let lh = sb_rows * 64;
        if let Some(accel) = self.accel.as_mut() {
            let sized = |w: usize, h: usize| Plane { data: Vec::new(), width: w, height: h };
            let cur = Arc::new(Frame {
                planes: [sized(lw, lh), sized(lw >> ss_h, lh >> ss_v), sized(lw >> ss_h, lh >> ss_v)],
                width: w,
                height: h,
                ss_h,
                ss_v,
                bpp: h_.bpp,
                color_space: self.color_space,
                full_range: self.full_range,
                segmap: Vec::new(),
                mv: Vec::new(),
                sb_cols,
            });
            let (uncompressed_size, compressed_size) = self.header_sizes;
            accel.frame(&AccelFrame { h: &h_, data, uncompressed_size, compressed_size, cur: &cur, refs: &self.refs })?;
            for i in 0..8 {
                if h_.refreshrefmask & (1 << i) != 0 {
                    self.refs[i] = Some(cur.clone());
                }
            }
            self.last = Some(cur.clone());
            return Ok(if h_.invisible { None } else { Some(cur) });
        }
        let fill = 0u16;
        let mut cur = Frame {
            planes: [
                Plane::new(lw, lh, fill),
                Plane::new(lw >> ss_h, lh >> ss_v, fill),
                Plane::new(lw >> ss_h, lh >> ss_v, fill),
            ],
            width: w,
            height: h,
            ss_h,
            ss_v,
            bpp: h_.bpp,
            color_space: self.color_space,
            full_range: self.full_range,
            segmap: vec![0; 64 * sb_cols * sb_rows],
            mv: vec![MvRefPair::default(); 64 * sb_cols * sb_rows],
            sb_cols,
        };
        let refs3: [Option<Arc<Frame>>; 3] = if h_.keyframe || h_.intraonly {
            [None, None, None]
        } else {
            [0, 1, 2].map(|i| self.refs[h_.refidx[i] as usize].clone())
        };
        let mut counts = Counts::new();
        if h_.refreshctx && h_.parallelmode {
            let ctx = &mut self.prob_ctx[h_.framectxid as usize];
            for i in 0..4 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..6 {
                            for m in 0..6 {
                                ctx.coef[i][j][k][l][m].copy_from_slice(&coef[i][j][k][l][m][..3]);
                            }
                        }
                    }
                }
                if h_.txfmmode as usize == i {
                    break;
                }
            }
            ctx.p = probs;
        }
        {
            let mut fd = block::FrameDec::new(
                &h_,
                &probs,
                &coef,
                &mut counts,
                &mut cur,
                &refs3,
                self.segmap_ref.as_deref(),
                mvpair.as_deref(),
                self.mvscale,
                self.mvstep,
            );
            fd.decode_tiles(&data[off..])?;
            if h_.filter_level != 0 {
                lpf::loop_filter_frame(&mut fd, &self.lim_lut, &self.mblim_lut);
            }
        }
        if h_.refreshctx && !h_.parallelmode {
            let fc = h_.framectxid as usize;
            probs::adapt_probs(&mut self.prob_ctx[fc], &probs, &counts, &h_, self.last_keyframe);
        }
        let cur = Arc::new(cur);
        for i in 0..8 {
            if h_.refreshrefmask & (1 << i) != 0 {
                self.refs[i] = Some(cur.clone());
            }
        }
        self.last = Some(cur.clone());
        Ok(if h_.invisible { None } else { Some(cur) })
    }
}

/// Coefficient probabilities: `[tx][plane][inter][band][ctx]`, 3 coded and
/// 8 Pareto-model entries.
pub type CoefProbs = [[[[[[u8; 11]; 6]; 6]; 2]; 2]; 4];
