//! HEVC parameter sets and the slice segment header (H.265 §7.3.2, §7.3.6).

use super::tables::{DEFAULT_SCALING, SCAN4, SCAN8X8};
use crate::codec::decode::bits::BitReader;

pub type Result<T> = std::result::Result<T, String>;

fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

/// Scaling factors per `sizeId` (4, 8, 16, 32) and `matrixId` (`cIdx` +
/// 3 for inter), expanded to raster order at full block size.
#[derive(Clone, Debug)]
pub struct ScalingFactors {
    /// `[sizeId][matrixId]`, `n * n` entries in raster order (`y * n + x`).
    pub m: [[Vec<u8>; 6]; 4],
}

/// Coded scaling lists, before expansion: per `sizeId`/`matrixId` the up to
/// 64 coefficients in diagonal order, and the DC for 16×16 and 32×32.
#[derive(Clone, Debug)]
struct ScalingLists {
    lists: [[[u8; 64]; 6]; 4],
    dc: [[u8; 6]; 4],
}

impl ScalingLists {
    fn default_lists() -> Self {
        let mut s = ScalingLists {
            lists: [[[16; 64]; 6]; 4],
            dc: [[16; 6]; 4],
        };
        for size in 1..4 {
            for m in 0..6 {
                s.lists[size][m] = DEFAULT_SCALING[(m >= 3) as usize];
            }
        }
        s
    }

    fn parse(r: &mut BitReader) -> Result<Self> {
        let mut s = Self::default_lists();
        for size in 0..4 {
            let step = if size == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                if !r.flag() {
                    let delta = r.ue() as usize;
                    if delta != 0 {
                        let d = delta * step;
                        if d > m {
                            return Err("scaling_list_pred_matrix_id_delta out of range".into());
                        }
                        let rm = m - d;
                        s.lists[size][m] = s.lists[size][rm];
                        s.dc[size][m] = s.dc[size][rm];
                    } else {
                        // Default list.
                        s.lists[size][m] = if size == 0 {
                            [16; 64]
                        } else {
                            DEFAULT_SCALING[(m >= 3) as usize]
                        };
                        s.dc[size][m] = 16;
                    }
                } else {
                    let mut next = 8i32;
                    let n = (1usize << (4 + (size << 1))).min(64);
                    if size > 1 {
                        let dc = r.se().saturating_add(8);
                        if !(1..=255).contains(&dc) {
                            return Err("scaling_list_dc_coef out of range".into());
                        }
                        next = dc;
                        s.dc[size][m] = dc as u8;
                    }
                    for i in 0..n {
                        let delta = r.se();
                        // (next + delta + 256) % 256 for the legal −128..=127,
                        // without overflowing on garbage.
                        next = (next as i64 + delta as i64).rem_euclid(256) as i32;
                        s.lists[size][m][i] = next as u8;
                    }
                    if size <= 1 {
                        s.dc[size][m] = s.lists[size][m][0];
                    }
                }
                m += step;
            }
        }
        // 32×32 chroma lists (4:4:4 only) copy from 16×16 — keep them sane.
        for m in [1, 2, 4, 5] {
            s.lists[3][m] = s.lists[2][m];
            s.dc[3][m] = s.dc[2][m];
        }
        Ok(s)
    }

    fn expand(&self) -> ScalingFactors {
        let mut out: [[Vec<u8>; 6]; 4] = Default::default();
        for size in 0..4 {
            let n = 4usize << size;
            for m in 0..6 {
                let mut f = vec![0u8; n * n];
                if size == 0 {
                    for (i, &(x, y)) in SCAN4[0].iter().enumerate() {
                        f[y as usize * 4 + x as usize] = self.lists[0][m][i];
                    }
                } else {
                    let rep = n / 8;
                    for (i, &(x, y)) in SCAN8X8.iter().enumerate() {
                        for j in 0..rep {
                            for k in 0..rep {
                                let (xx, yy) = (x as usize * rep + k, y as usize * rep + j);
                                f[yy * n + xx] = self.lists[size][m][i];
                            }
                        }
                    }
                    if size >= 2 {
                        f[0] = self.dc[size][m];
                    }
                }
                out[size][m] = f;
            }
        }
        ScalingFactors { m: out }
    }
}

/// A short-term reference picture set (§7.4.8).
#[derive(Clone, Debug, Default)]
pub struct StRps {
    /// Negative deltas, closest first, with `used_by_curr_pic`.
    pub neg: Vec<(i32, bool)>,
    /// Positive deltas, closest first.
    pub pos: Vec<(i32, bool)>,
}

impl StRps {
    fn num_delta_pocs(&self) -> usize {
        self.neg.len() + self.pos.len()
    }

    /// Parse `st_ref_pic_set(idx)`. `sets` are those already parsed from
    /// the SPS. `in_slice` marks the set in a slice header (whose `idx` is
    /// `num_short_term_ref_pic_sets`) — the only one that codes
    /// `delta_idx_minus1`. While the SPS itself is being parsed `idx` also
    /// equals the number parsed so far, so the index cannot tell them apart.
    fn parse(r: &mut BitReader, idx: usize, sets: &[StRps], in_slice: bool) -> Result<StRps> {
        let inter = idx != 0 && r.flag();
        if inter {
            let delta_idx = if in_slice { r.ue() as usize + 1 } else { 1 };
            if delta_idx > idx {
                return Err("delta_idx_minus1 out of range".into());
            }
            let rref = &sets[idx - delta_idx];
            let sign = r.flag();
            let abs = r.ue() as i32 + 1;
            let delta_rps = if sign { -abs } else { abs };
            let n = rref.num_delta_pocs();
            let mut used = vec![false; n + 1];
            let mut use_delta = vec![true; n + 1];
            for j in 0..=n {
                used[j] = r.flag();
                if !used[j] {
                    use_delta[j] = r.flag();
                }
            }
            // §7.4.8, equations 7-61 and 7-62. Entries 0..neg are the
            // reference set's negative pictures, then its positive ones,
            // then the reference picture itself.
            let nn = rref.neg.len();
            let mut out = StRps::default();
            for j in (0..rref.pos.len()).rev() {
                let d = rref.pos[j].0 + delta_rps;
                if d < 0 && use_delta[nn + j] {
                    out.neg.push((d, used[nn + j]));
                }
            }
            if delta_rps < 0 && use_delta[n] {
                out.neg.push((delta_rps, used[n]));
            }
            for j in 0..nn {
                let d = rref.neg[j].0 + delta_rps;
                if d < 0 && use_delta[j] {
                    out.neg.push((d, used[j]));
                }
            }
            for j in (0..nn).rev() {
                let d = rref.neg[j].0 + delta_rps;
                if d > 0 && use_delta[j] {
                    out.pos.push((d, used[j]));
                }
            }
            if delta_rps > 0 && use_delta[n] {
                out.pos.push((delta_rps, used[n]));
            }
            for j in 0..rref.pos.len() {
                let d = rref.pos[j].0 + delta_rps;
                if d > 0 && use_delta[nn + j] {
                    out.pos.push((d, used[nn + j]));
                }
            }
            Ok(out)
        } else {
            let nneg = r.ue() as usize;
            let npos = r.ue() as usize;
            if nneg > 16 || npos > 16 {
                return Err("too many pictures in a reference picture set".into());
            }
            let mut out = StRps::default();
            let mut poc = 0;
            for _ in 0..nneg {
                poc -= r.ue() as i32 + 1;
                out.neg.push((poc, r.flag()));
            }
            poc = 0;
            for _ in 0..npos {
                poc += r.ue() as i32 + 1;
                out.pos.push((poc, r.flag()));
            }
            Ok(out)
        }
    }
}

fn profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> u32 {
    r.skip(2 + 1);
    let profile_idc = r.u(5);
    r.skip(32 + 4 + 43 + 1 + 8);
    let mut prof = [false; 8];
    let mut lvl = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        prof[i] = r.flag();
        lvl[i] = r.flag();
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.skip(2);
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if prof[i] {
            r.skip(88);
        }
        if lvl[i] {
            r.skip(8);
        }
    }
    profile_idc
}

fn sub_layer_hrd(r: &mut BitReader, cpb_cnt: u32, sub_pic: bool) {
    for _ in 0..=cpb_cnt {
        r.ue();
        r.ue();
        if sub_pic {
            r.ue();
            r.ue();
        }
        r.skip(1);
    }
}

fn hrd_parameters(r: &mut BitReader, common: bool, max_sub_layers_minus1: u32) {
    let mut nal = false;
    let mut vcl = false;
    let mut sub_pic = false;
    if common {
        nal = r.flag();
        vcl = r.flag();
        if nal || vcl {
            sub_pic = r.flag();
            if sub_pic {
                r.skip(8 + 5 + 1 + 5);
            }
            r.skip(4 + 4);
            if sub_pic {
                r.skip(4);
            }
            r.skip(5 + 5 + 5);
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let general = r.flag();
        let within_cvs = if general { true } else { r.flag() };
        let mut low_delay = false;
        if within_cvs {
            r.ue();
        } else {
            low_delay = r.flag();
        }
        let cpb_cnt = if !low_delay { r.ue() } else { 0 };
        if nal {
            sub_layer_hrd(r, cpb_cnt, sub_pic);
        }
        if vcl {
            sub_layer_hrd(r, cpb_cnt, sub_pic);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub chroma_format_idc: u32,
    pub width: u32,
    pub height: u32,
    /// Luma samples cropped: left, right, top, bottom.
    pub crop: [u32; 4],
    pub bit_depth_y: u32,
    pub bit_depth_c: u32,
    pub log2_max_poc_lsb: u32,
    pub max_dec_pic_buffering: u32,
    /// `sps_max_num_reorder_pics` of the highest sub-layer.
    pub max_num_reorder: u32,
    /// `sps_max_latency_increase_plus1` of the highest sub-layer.
    pub max_latency_increase_plus1: u32,
    pub log2_min_cb: u32,
    pub log2_ctb: u32,
    pub log2_min_tb: u32,
    pub log2_max_tb: u32,
    pub max_th_depth_inter: u32,
    pub max_th_depth_intra: u32,
    pub scaling: Option<ScalingFactors>,
    pub amp: bool,
    pub sao: bool,
    pub pcm: bool,
    pub pcm_bit_depth_y: u32,
    pub pcm_bit_depth_c: u32,
    pub log2_min_pcm: u32,
    pub log2_max_pcm: u32,
    pub pcm_loop_filter_disabled: bool,
    pub st_rps: Vec<StRps>,
    pub long_term_present: bool,
    pub lt_ref_pics: Vec<(u32, bool)>,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
    /// The VUI's colour description, when it has one.
    pub colour_primaries: Option<u32>,
    pub transfer_characteristics: Option<u32>,
    pub matrix_coefficients: Option<u32>,
    pub full_range: bool,
}

impl Sps {
    pub fn parse(rbsp: &[u8]) -> Result<Sps> {
        let mut r = BitReader::new(rbsp);
        r.skip(4);
        let max_sub_layers_minus1 = r.u(3);
        r.skip(1);
        let profile_idc = profile_tier_level(&mut r, max_sub_layers_minus1);
        let id = r.ue();
        if id > 15 {
            return Err("SPS id out of range".into());
        }
        let chroma_format_idc = r.ue();
        if chroma_format_idc == 3 && r.flag() {
            return Err("separate colour planes are not supported".into());
        }
        if chroma_format_idc != 1 {
            return Err(format!(
                "HEVC chroma_format_idc {chroma_format_idc} is not supported (4:2:0 only)"
            ));
        }
        let width = r.ue();
        let height = r.ue();
        let mut crop = [0u32; 4];
        if r.flag() {
            crop = [r.ue(), r.ue(), r.ue(), r.ue()].map(|c| c.saturating_mul(2));
        }
        // Bounds before anything is sized from these (§7.4.3.2.1): garbage
        // reads as all-ones Exp-Golomb codes.
        if width == 0 || height == 0 || width > 1 << 14 || height > 1 << 14 {
            return Err("SPS picture size out of range".into());
        }
        if crop[0].saturating_add(crop[1]) >= width || crop[2].saturating_add(crop[3]) >= height {
            return Err("SPS conformance window exceeds the picture".into());
        }
        let bit_depth_y = r.ue().saturating_add(8);
        let bit_depth_c = r.ue().saturating_add(8);
        if bit_depth_y > 12 || bit_depth_c > 12 {
            return Err("bit depths above 12 are not supported".into());
        }
        let log2_max_poc_lsb = r.ue().saturating_add(4);
        if log2_max_poc_lsb > 16 {
            return Err("log2_max_pic_order_cnt_lsb out of range".into());
        }
        let ordering_all = r.flag();
        let mut max_dec_pic_buffering = 1;
        let mut max_num_reorder = 0;
        let mut max_latency_increase_plus1 = 0;
        let start = if ordering_all { 0 } else { max_sub_layers_minus1 };
        for _ in start..=max_sub_layers_minus1 {
            max_dec_pic_buffering = r.ue().saturating_add(1);
            if max_dec_pic_buffering > 16 {
                return Err("sps_max_dec_pic_buffering out of range".into());
            }
            max_num_reorder = r.ue();
            max_latency_increase_plus1 = r.ue();
        }
        let log2_min_cb = r.ue().saturating_add(3);
        let log2_ctb = log2_min_cb.saturating_add(r.ue());
        let log2_min_tb = r.ue().saturating_add(2);
        let log2_max_tb = log2_min_tb.saturating_add(r.ue());
        if !(4..=6).contains(&log2_ctb) || log2_max_tb > 5 || log2_min_tb >= log2_min_cb || log2_max_tb > log2_ctb {
            return Err("coding/transform block sizes out of range".into());
        }
        let max_th_depth_inter = r.ue();
        let max_th_depth_intra = r.ue();
        let mut scaling = None;
        if r.flag() {
            let lists = if r.flag() {
                ScalingLists::parse(&mut r)?
            } else {
                ScalingLists::default_lists()
            };
            scaling = Some(lists.expand());
        }
        let amp = r.flag();
        let sao = r.flag();
        let pcm = r.flag();
        let (mut pcm_bit_depth_y, mut pcm_bit_depth_c, mut log2_min_pcm, mut log2_max_pcm) = (8, 8, 0, 0);
        let mut pcm_loop_filter_disabled = false;
        if pcm {
            pcm_bit_depth_y = r.u(4) + 1;
            pcm_bit_depth_c = r.u(4) + 1;
            log2_min_pcm = r.ue().saturating_add(3);
            log2_max_pcm = log2_min_pcm.saturating_add(r.ue());
            pcm_loop_filter_disabled = r.flag();
        }
        let n = r.ue() as usize;
        if n > 64 {
            return Err("num_short_term_ref_pic_sets out of range".into());
        }
        let mut st_rps = Vec::with_capacity(n);
        for i in 0..n {
            let s = StRps::parse(&mut r, i, &st_rps, false)?;
            st_rps.push(s);
        }
        let long_term_present = r.flag();
        let mut lt_ref_pics = Vec::new();
        if long_term_present {
            let n = r.ue();
            if n > 32 {
                return Err("num_long_term_ref_pics_sps out of range".into());
            }
            for _ in 0..n {
                lt_ref_pics.push((r.u(log2_max_poc_lsb), r.flag()));
            }
        }
        let temporal_mvp = r.flag();
        let strong_intra_smoothing = r.flag();
        let (mut colour_primaries, mut transfer_characteristics) = (None, None);
        let mut matrix_coefficients = None;
        let mut full_range = false;
        if r.flag() {
            // vui_parameters()
            if r.flag() && r.u(8) == 255 {
                r.skip(32);
            }
            if r.flag() {
                r.skip(1);
            }
            if r.flag() {
                r.skip(3);
                full_range = r.flag();
                if r.flag() {
                    colour_primaries = Some(r.u(8));
                    transfer_characteristics = Some(r.u(8));
                    matrix_coefficients = Some(r.u(8));
                }
            }
            if r.flag() {
                r.ue();
                r.ue();
            }
            r.skip(3);
            if r.flag() {
                r.ue();
                r.ue();
                r.ue();
                r.ue();
            }
            if r.flag() {
                r.skip(64);
                if r.flag() {
                    r.ue();
                }
                if r.flag() {
                    hrd_parameters(&mut r, true, max_sub_layers_minus1);
                }
            }
            if r.flag() {
                r.skip(3);
                r.ue();
                r.ue();
                r.ue();
                r.ue();
                r.ue();
            }
        }
        if r.flag() {
            let range = r.flag();
            let _others = r.u(7);
            if range && r.u(9) != 0 {
                return Err("HEVC range extension tools are not supported".into());
            }
        }
        if r.overrun() {
            return Err("SPS runs past the NAL".into());
        }
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err("picture size out of range".into());
        }
        let _ = profile_idc;
        Ok(Sps {
            id,
            profile_idc,
            chroma_format_idc,
            width,
            height,
            crop,
            bit_depth_y,
            bit_depth_c,
            log2_max_poc_lsb,
            max_dec_pic_buffering,
            max_num_reorder,
            max_latency_increase_plus1,
            log2_min_cb,
            log2_ctb,
            log2_min_tb,
            log2_max_tb,
            max_th_depth_inter,
            max_th_depth_intra,
            scaling,
            amp,
            sao,
            pcm,
            pcm_bit_depth_y,
            pcm_bit_depth_c,
            log2_min_pcm,
            log2_max_pcm,
            pcm_loop_filter_disabled,
            st_rps,
            long_term_present,
            lt_ref_pics,
            temporal_mvp,
            strong_intra_smoothing,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        })
    }

    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb
    }
    pub fn width_ctbs(&self) -> u32 {
        self.width.div_ceil(self.ctb_size())
    }
    pub fn height_ctbs(&self) -> u32 {
        self.height.div_ceil(self.ctb_size())
    }
}

#[derive(Clone, Debug)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub dependent_slices: bool,
    pub output_flag_present: bool,
    pub extra_slice_header_bits: u32,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_default: [u32; 2],
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip: bool,
    pub cu_qp_delta: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles: bool,
    pub entropy_sync: bool,
    /// Explicit tile column widths / row heights in CTBs, or `None` for
    /// uniform spacing with the given count.
    pub tile_cols: TileSpacing,
    pub tile_rows: TileSpacing,
    pub loop_filter_across_tiles: bool,
    pub loop_filter_across_slices: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub scaling: Option<ScalingFactors>,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u32,
    pub slice_header_extension_present: bool,
}

#[derive(Clone, Debug)]
pub enum TileSpacing {
    Uniform(u32),
    Explicit(Vec<u32>),
}

impl TileSpacing {
    /// Boundaries in CTBs: `n + 1` entries from 0 to `total`.
    pub fn boundaries(&self, total: u32) -> Vec<u32> {
        let mut b = vec![0];
        match self {
            TileSpacing::Uniform(n) => {
                for i in 0..*n {
                    b.push(((i + 1) * total) / n);
                }
            }
            TileSpacing::Explicit(sizes) => {
                let mut acc = 0;
                for &s in sizes {
                    acc += s;
                    b.push(acc.min(total));
                }
                b.push(total);
            }
        }
        b
    }
}

impl Pps {
    pub fn parse(rbsp: &[u8]) -> Result<Pps> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue();
        let sps_id = r.ue();
        if id > 63 || sps_id > 15 {
            return Err("PPS id out of range".into());
        }
        let dependent_slices = r.flag();
        let output_flag_present = r.flag();
        let extra_slice_header_bits = r.u(3);
        let sign_data_hiding = r.flag();
        let cabac_init_present = r.flag();
        let num_ref_idx_default = [r.ue().saturating_add(1), r.ue().saturating_add(1)];
        if num_ref_idx_default[0] > 15 || num_ref_idx_default[1] > 15 {
            return Err("num_ref_idx_default out of range".into());
        }
        let init_qp = r.se().saturating_add(26);
        let constrained_intra_pred = r.flag();
        let transform_skip = r.flag();
        let cu_qp_delta = r.flag();
        let diff_cu_qp_delta_depth = if cu_qp_delta { r.ue() } else { 0 };
        let cb_qp_offset = r.se();
        let cr_qp_offset = r.se();
        let slice_chroma_qp_offsets_present = r.flag();
        let weighted_pred = r.flag();
        let weighted_bipred = r.flag();
        let transquant_bypass = r.flag();
        let tiles = r.flag();
        let entropy_sync = r.flag();
        let mut tile_cols = TileSpacing::Uniform(1);
        let mut tile_rows = TileSpacing::Uniform(1);
        let mut loop_filter_across_tiles = true;
        if tiles {
            let ncols = r.ue().saturating_add(1);
            let nrows = r.ue().saturating_add(1);
            // Level 6.2's limits are 20 × 22; past those it is not a stream.
            if ncols > 20 || nrows > 22 {
                return Err("tile counts out of range".into());
            }
            if ncols > 64 || nrows > 64 {
                return Err("too many tiles".into());
            }
            if r.flag() {
                tile_cols = TileSpacing::Uniform(ncols);
                tile_rows = TileSpacing::Uniform(nrows);
            } else {
                tile_cols = TileSpacing::Explicit((1..ncols).map(|_| r.ue().saturating_add(1)).collect());
                tile_rows = TileSpacing::Explicit((1..nrows).map(|_| r.ue().saturating_add(1)).collect());
            }
            loop_filter_across_tiles = r.flag();
        }
        let loop_filter_across_slices = r.flag();
        let mut deblocking_override_enabled = false;
        let mut deblocking_disabled = false;
        let mut beta_offset = 0;
        let mut tc_offset = 0;
        if r.flag() {
            deblocking_override_enabled = r.flag();
            deblocking_disabled = r.flag();
            if !deblocking_disabled {
                beta_offset = r.se().saturating_mul(2);
                tc_offset = r.se().saturating_mul(2);
            }
        }
        let mut scaling = None;
        if r.flag() {
            scaling = Some(ScalingLists::parse(&mut r)?.expand());
        }
        let lists_modification_present = r.flag();
        let log2_parallel_merge_level = r.ue().saturating_add(2);
        let slice_header_extension_present = r.flag();
        if r.flag() {
            let range = r.flag();
            let _others = r.u(7);
            if range {
                if transform_skip
                    && r.ue() != 0 {
                        return Err("transform skip above 4x4 (range extension) is not supported".into());
                    }
                if r.flag() {
                    return Err("cross-component prediction is not supported".into());
                }
                if r.flag() {
                    return Err("chroma QP offset lists are not supported".into());
                }
                let log2_sao_luma = r.ue();
                let log2_sao_chroma = r.ue();
                if log2_sao_luma != 0 || log2_sao_chroma != 0 {
                    return Err("SAO offset scaling is not supported".into());
                }
            }
        }
        Ok(Pps {
            id,
            sps_id,
            dependent_slices,
            output_flag_present,
            extra_slice_header_bits,
            sign_data_hiding,
            cabac_init_present,
            num_ref_idx_default,
            init_qp,
            constrained_intra_pred,
            transform_skip,
            cu_qp_delta,
            diff_cu_qp_delta_depth,
            cb_qp_offset,
            cr_qp_offset,
            slice_chroma_qp_offsets_present,
            weighted_pred,
            weighted_bipred,
            transquant_bypass,
            tiles,
            entropy_sync,
            tile_cols,
            tile_rows,
            loop_filter_across_tiles,
            loop_filter_across_slices,
            deblocking_override_enabled,
            deblocking_disabled,
            beta_offset,
            tc_offset,
            scaling,
            lists_modification_present,
            log2_parallel_merge_level,
            slice_header_extension_present,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    B,
    P,
    I,
}

/// Explicit weighted prediction: `[list][ref] = [(weight, offset); Y, Cb, Cr]`
/// with offsets already scaled to the bit depth.
#[derive(Clone, Debug, Default)]
pub struct PredWeights {
    pub luma_log2_denom: u32,
    pub chroma_log2_denom: u32,
    pub w: [Vec<[(i32, i32); 3]>; 2],
}

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub first_slice_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u32,
    pub dependent: bool,
    pub segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output: bool,
    pub poc_lsb: u32,
    pub st_rps: StRps,
    /// Long-term entries: `(poc value, msb present, used by current)`.
    pub lt: Vec<(i32, bool, bool)>,
    pub temporal_mvp: bool,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub num_ref_idx: [u32; 2],
    pub list_entry: [Option<Vec<u32>>; 2],
    pub mvd_l1_zero: bool,
    pub cabac_init_flag: bool,
    pub collocated_from_l0: bool,
    pub collocated_ref_idx: u32,
    pub weights: Option<PredWeights>,
    pub max_merge_cand: u32,
    pub qp_delta: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub deblocking_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub loop_filter_across_slices: bool,
    /// Byte offset of `slice_segment_data()` in the RBSP.
    pub data_offset: usize,
}

impl SliceHeader {
    /// Parse a slice segment header. A dependent segment copies everything
    /// but its address from `prev`, the last independent one.
    pub fn parse(
        rbsp: &[u8],
        nal_type: u32,
        spss: &[Option<Sps>],
        ppss: &[Option<Pps>],
        prev: Option<&SliceHeader>,
    ) -> Result<SliceHeader> {
        let mut r = BitReader::new(rbsp);
        let first = r.flag();
        let mut no_output_of_prior_pics = false;
        if (16..=23).contains(&nal_type) {
            no_output_of_prior_pics = r.flag();
        }
        let pps_id = r.ue();
        let pps = ppss
            .get(pps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| format!("slice names missing PPS {pps_id}"))?;
        let sps = spss[pps.sps_id as usize]
            .as_ref()
            .ok_or("slice's PPS names a missing SPS")?;
        let mut dependent = false;
        let mut segment_address = 0;
        if !first {
            if pps.dependent_slices {
                dependent = r.flag();
            }
            let ctbs = sps.width_ctbs() * sps.height_ctbs();
            segment_address = r.u(ceil_log2(ctbs));
            if segment_address >= ctbs {
                return Err("slice_segment_address out of range".into());
            }
        }
        let mut h = if dependent {
            let mut h = prev
                .ok_or("dependent slice segment without a preceding slice")?
                .clone();
            h.first_slice_in_pic = first;
            h.dependent = true;
            h.segment_address = segment_address;
            h
        } else {
            let mut h = SliceHeader {
                first_slice_in_pic: first,
                no_output_of_prior_pics,
                pps_id,
                dependent: false,
                segment_address,
                slice_type: SliceType::I,
                pic_output: true,
                poc_lsb: 0,
                st_rps: StRps::default(),
                lt: Vec::new(),
                temporal_mvp: false,
                sao_luma: false,
                sao_chroma: false,
                num_ref_idx: [0, 0],
                list_entry: [None, None],
                mvd_l1_zero: false,
                cabac_init_flag: false,
                collocated_from_l0: true,
                collocated_ref_idx: 0,
                weights: None,
                max_merge_cand: 5,
                qp_delta: 0,
                cb_qp_offset: 0,
                cr_qp_offset: 0,
                deblocking_disabled: pps.deblocking_disabled,
                beta_offset: pps.beta_offset,
                tc_offset: pps.tc_offset,
                loop_filter_across_slices: pps.loop_filter_across_slices,
                data_offset: 0,
            };
            r.skip(pps.extra_slice_header_bits);
            h.slice_type = match r.ue() {
                0 => SliceType::B,
                1 => SliceType::P,
                2 => SliceType::I,
                t => return Err(format!("slice_type {t} out of range")),
            };
            if pps.output_flag_present {
                h.pic_output = r.flag();
            }
            let idr = nal_type == 19 || nal_type == 20;
            let mut num_pic_total_curr = 0u32;
            if !idr {
                h.poc_lsb = r.u(sps.log2_max_poc_lsb);
                if !r.flag() {
                    h.st_rps = StRps::parse(&mut r, sps.st_rps.len(), &sps.st_rps, true)?;
                } else {
                    if sps.st_rps.is_empty() {
                        return Err("slice refers to an SPS RPS but there are none".into());
                    }
                    let idx = if sps.st_rps.len() > 1 {
                        r.u(ceil_log2(sps.st_rps.len() as u32)) as usize
                    } else {
                        0
                    };
                    h.st_rps = sps
                        .st_rps
                        .get(idx)
                        .ok_or("short_term_ref_pic_set_idx out of range")?
                        .clone();
                }
                if sps.long_term_present {
                    let n_sps = if !sps.lt_ref_pics.is_empty() { r.ue() } else { 0 };
                    let n_pics = r.ue();
                    if n_sps as usize > sps.lt_ref_pics.len() || n_sps + n_pics > 32 {
                        return Err("too many long-term pictures".into());
                    }
                    let max_lsb = 1i32 << sps.log2_max_poc_lsb;
                    let mut msb_cycle = 0i32;
                    for i in 0..(n_sps + n_pics) {
                        let (lsb, used) = if i < n_sps {
                            let idx = if sps.lt_ref_pics.len() > 1 {
                                r.u(ceil_log2(sps.lt_ref_pics.len() as u32)) as usize
                            } else {
                                0
                            };
                            let e = sps.lt_ref_pics.get(idx).ok_or("lt_idx_sps out of range")?;
                            (e.0 as i32, e.1)
                        } else {
                            (r.u(sps.log2_max_poc_lsb) as i32, r.flag())
                        };
                        let msb_present = r.flag();
                        let mut poc = lsb;
                        if msb_present {
                            let delta = r.ue() as i32;
                            msb_cycle = if i == 0 || i == n_sps { delta } else { delta + msb_cycle };
                            // The rest of the MSB comes from the current
                            // POC, which the caller adds: §7.4.7.1 (7-52).
                            poc = lsb - msb_cycle * max_lsb;
                        }
                        h.lt.push((poc, msb_present, used));
                    }
                }
                if sps.temporal_mvp {
                    h.temporal_mvp = r.flag();
                }
                num_pic_total_curr = h.st_rps.neg.iter().filter(|e| e.1).count() as u32
                    + h.st_rps.pos.iter().filter(|e| e.1).count() as u32
                    + h.lt.iter().filter(|e| e.2).count() as u32;
            }
            if sps.sao {
                h.sao_luma = r.flag();
                h.sao_chroma = r.flag();
            }
            if h.slice_type != SliceType::I {
                h.num_ref_idx = pps.num_ref_idx_default;
                if r.flag() {
                    h.num_ref_idx[0] = r.ue().saturating_add(1);
                    if h.slice_type == SliceType::B {
                        h.num_ref_idx[1] = r.ue().saturating_add(1);
                    }
                }
                if h.slice_type != SliceType::B {
                    h.num_ref_idx[1] = 0;
                }
                if h.num_ref_idx[0] > 15 || h.num_ref_idx[1] > 15 {
                    return Err("num_ref_idx_active out of range".into());
                }
                if pps.lists_modification_present && num_pic_total_curr > 1 {
                    let bits = ceil_log2(num_pic_total_curr);
                    for l in 0..(1 + (h.slice_type == SliceType::B) as usize) {
                        if r.flag() {
                            h.list_entry[l] =
                                Some((0..h.num_ref_idx[l]).map(|_| r.u(bits)).collect());
                        }
                    }
                }
                if h.slice_type == SliceType::B {
                    h.mvd_l1_zero = r.flag();
                }
                if pps.cabac_init_present {
                    h.cabac_init_flag = r.flag();
                }
                if h.temporal_mvp {
                    if h.slice_type == SliceType::B {
                        h.collocated_from_l0 = r.flag();
                    }
                    let l = if h.collocated_from_l0 { 0 } else { 1 };
                    if h.num_ref_idx[l] > 1 {
                        h.collocated_ref_idx = r.ue();
                        if h.collocated_ref_idx >= h.num_ref_idx[l] {
                            return Err("collocated_ref_idx out of range".into());
                        }
                    }
                }
                if (pps.weighted_pred && h.slice_type == SliceType::P)
                    || (pps.weighted_bipred && h.slice_type == SliceType::B)
                {
                    h.weights = Some(pred_weight_table(&mut r, sps, &h)?);
                }
                h.max_merge_cand = 5 - r.ue();
                if !(1..=5).contains(&h.max_merge_cand) {
                    return Err("five_minus_max_num_merge_cand out of range".into());
                }
            }
            h.qp_delta = r.se();
            if pps.slice_chroma_qp_offsets_present {
                h.cb_qp_offset = r.se();
                h.cr_qp_offset = r.se();
            }
            let mut override_flag = false;
            if pps.deblocking_override_enabled {
                override_flag = r.flag();
            }
            if override_flag {
                h.deblocking_disabled = r.flag();
                if !h.deblocking_disabled {
                    h.beta_offset = r.se().saturating_mul(2);
                    h.tc_offset = r.se().saturating_mul(2);
                }
            }
            if pps.loop_filter_across_slices
                && (h.sao_luma || h.sao_chroma || !h.deblocking_disabled)
            {
                h.loop_filter_across_slices = r.flag();
            }
            h
        };
        if pps.tiles || pps.entropy_sync {
            let n = r.ue();
            if n > 0 {
                let len = r.ue().saturating_add(1);
                if len > 32 {
                    return Err("offset_len_minus1 out of range".into());
                }
                for _ in 0..n {
                    r.skip(len);
                }
            }
        }
        if pps.slice_header_extension_present {
            let len = r.ue();
            r.skip(8 * len);
        }
        // byte_alignment(): a one bit, then zeros.
        r.skip(1);
        r.byte_align();
        if r.overrun() {
            return Err("slice header runs past the NAL".into());
        }
        h.data_offset = r.pos() / 8;
        Ok(h)
    }
}

fn pred_weight_table(r: &mut BitReader, sps: &Sps, h: &SliceHeader) -> Result<PredWeights> {
    let mut w = PredWeights {
        luma_log2_denom: r.ue(),
        ..Default::default()
    };
    if w.luma_log2_denom > 7 {
        return Err("luma_log2_weight_denom out of range".into());
    }
    let cd = (w.luma_log2_denom as i32).saturating_add(r.se());
    if !(0..=7).contains(&cd) {
        return Err("chroma weight denominator out of range".into());
    }
    w.chroma_log2_denom = cd as u32;
    let sy = sps.bit_depth_y as i32 - 8;
    let sc = sps.bit_depth_c as i32 - 8;
    for l in 0..(1 + (h.slice_type == SliceType::B) as usize) {
        let n = h.num_ref_idx[l] as usize;
        let luma: Vec<bool> = (0..n).map(|_| r.flag()).collect();
        let chroma: Vec<bool> = (0..n).map(|_| r.flag()).collect();
        for i in 0..n {
            let mut e = [
                (1 << w.luma_log2_denom, 0),
                (1 << w.chroma_log2_denom, 0),
                (1 << w.chroma_log2_denom, 0),
            ];
            if luma[i] {
                e[0].0 += r.se();
                e[0].1 = r.se() << sy;
            }
            if chroma[i] {
                for ec in e.iter_mut().skip(1).take(2) {
                    let dw = r.se();
                    let doff = r.se();
                    let wc = (1 << w.chroma_log2_denom) + dw;
                    let half = 128;
                    let off = (half - ((half * wc) >> w.chroma_log2_denom) + doff).clamp(-half, half - 1);
                    *ec = (wc, off << sc);
                }
            }
            w.w[l].push(e);
        }
    }
    Ok(w)
}
