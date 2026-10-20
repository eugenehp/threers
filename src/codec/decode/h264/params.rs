//! H.264 sequence and picture parameter sets and the slice header (§7.3.2,
//! §7.3.3).

use super::tables::{DEFAULT_4X4, DEFAULT_8X8, ZIGZAG4, ZIGZAG8};
use crate::codec::decode::bits::BitReader;

pub type Result<T> = std::result::Result<T, String>;

/// Scaling matrices in raster order: six 4×4 (Y/Cb/Cr intra, Y/Cb/Cr inter)
/// and six 8×8 (same order; only the luma two exist outside 4:4:4).
#[derive(Clone, Debug, PartialEq)]
pub struct ScalingLists {
    pub m4: [[u8; 16]; 6],
    pub m8: [[u8; 64]; 6],
}

impl ScalingLists {
    pub fn flat() -> Self {
        ScalingLists {
            m4: [[16; 16]; 6],
            m8: [[16; 64]; 6],
        }
    }
}

/// Parse one `scaling_list()` (§7.3.2.1.1.1). Returns `None` when
/// `useDefaultScalingMatrixFlag` is set; otherwise the list in raster order.
fn scaling_list<const N: usize>(r: &mut BitReader, zigzag: &[usize; N]) -> Option<[u8; N]> {
    let mut out = [0u8; N];
    let mut last = 8i32;
    let mut next = 8i32;
    for j in 0..N {
        if next != 0 {
            let delta = r.se();
            // Same as (last + delta + 256) % 256 for the legal −128..=127,
            // without overflowing on garbage.
            next = (last as i64 + delta as i64).rem_euclid(256) as i32;
            if j == 0 && next == 0 {
                return None;
            }
        }
        let v = if next == 0 { last } else { next };
        out[zigzag[j]] = v as u8;
        last = v;
    }
    Some(out)
}

/// Read a full set of scaling lists, resolving each absent one by fall-back
/// rule A (`fallback == None`, an SPS or a PPS over an SPS without lists) or
/// B (`fallback == Some(sps lists)`) — Table 7-2.
fn read_scaling_lists(
    r: &mut BitReader,
    count: usize,
    fallback: Option<&ScalingLists>,
) -> ScalingLists {
    let mut s = ScalingLists::flat();
    for i in 0..count {
        let present = r.flag();
        if i < 6 {
            let parsed = if present {
                Some(scaling_list(r, &ZIGZAG4).unwrap_or(DEFAULT_4X4[i / 3]))
            } else {
                None
            };
            s.m4[i] = match parsed {
                Some(l) => l,
                None if i == 0 || i == 3 => match fallback {
                    Some(f) => f.m4[i],
                    None => DEFAULT_4X4[i / 3],
                },
                None => s.m4[i - 1],
            };
        } else {
            let k = i - 6;
            let parsed = if present {
                Some(scaling_list(r, &ZIGZAG8).unwrap_or(DEFAULT_8X8[k % 2]))
            } else {
                None
            };
            s.m8[k] = match parsed {
                Some(l) => l,
                None if k < 2 => match fallback {
                    Some(f) => f.m8[k],
                    None => DEFAULT_8X8[k],
                },
                None => s.m8[k - 2],
            };
        }
    }
    // Lists the syntax never reached keep the fall-back rule's answer.
    for i in count..12 {
        if i >= 6 {
            let k = i - 6;
            s.m8[k] = match fallback {
                Some(f) => f.m8[k],
                None if k < 2 => DEFAULT_8X8[k],
                None => s.m8[k - 2],
            };
        }
    }
    s
}

#[derive(Clone, Debug)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub chroma_format_idc: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub transform_bypass: bool,
    pub scaling: Option<ScalingLists>,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u32,
    pub gaps_allowed: bool,
    pub width_mbs: u32,
    pub height_mbs: u32,
    pub frame_mbs_only: bool,
    pub mbaff: bool,
    pub direct_8x8_inference: bool,
    /// Luma samples cropped off left, right, top, bottom.
    pub crop: [u32; 4],
    /// The VUI's colour description, when it has one.
    pub colour_primaries: Option<u32>,
    pub transfer_characteristics: Option<u32>,
    pub matrix_coefficients: Option<u32>,
    pub full_range: bool,
}

impl Sps {
    pub fn parse(rbsp: &[u8]) -> Result<Sps> {
        let mut r = BitReader::new(rbsp);
        let profile_idc = r.u(8);
        r.skip(16); // constraint flags, reserved, level_idc
        let id = r.ue();
        if id > 31 {
            return Err(format!("SPS id {id} out of range"));
        }
        let mut chroma_format_idc = 1;
        let mut bit_depth_luma = 8;
        let mut bit_depth_chroma = 8;
        let mut transform_bypass = false;
        let mut scaling = None;
        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma_format_idc = r.ue();
            if chroma_format_idc == 3 && r.flag() {
                return Err("separate colour planes are not supported".into());
            }
            bit_depth_luma = r.ue().saturating_add(8);
            bit_depth_chroma = r.ue().saturating_add(8);
            transform_bypass = r.flag();
            if r.flag() {
                let n = if chroma_format_idc != 3 { 8 } else { 12 };
                scaling = Some(read_scaling_lists(&mut r, n, None));
            }
        }
        if chroma_format_idc > 1 {
            return Err(format!(
                "H.264 chroma_format_idc {chroma_format_idc} is not supported (4:2:0 and 4:0:0 only)"
            ));
        }
        if bit_depth_luma > 14 || bit_depth_chroma > 14 {
            return Err("bit depth out of range".into());
        }
        // Range-checked against §7.4.2.1.1 before the arithmetic: a truncated
        // or garbage SPS reads all-ones Exp-Golomb codes, which would overflow.
        let log2_max_frame_num = r.ue().saturating_add(4);
        if log2_max_frame_num > 16 {
            return Err("log2_max_frame_num out of range".into());
        }
        let poc_type = r.ue();
        let mut log2_max_poc_lsb = 0;
        let mut delta_pic_order_always_zero = false;
        let mut offset_for_non_ref_pic = 0;
        let mut offset_for_top_to_bottom_field = 0;
        let mut offset_for_ref_frame = Vec::new();
        match poc_type {
            0 => {
                log2_max_poc_lsb = r.ue().saturating_add(4);
                if log2_max_poc_lsb > 16 {
                    return Err("log2_max_pic_order_cnt_lsb out of range".into());
                }
            }
            1 => {
                delta_pic_order_always_zero = r.flag();
                offset_for_non_ref_pic = r.se();
                offset_for_top_to_bottom_field = r.se();
                let n = r.ue();
                if n > 255 {
                    return Err("too many POC cycle offsets".into());
                }
                for _ in 0..n {
                    offset_for_ref_frame.push(r.se());
                }
            }
            2 => {}
            _ => return Err(format!("pic_order_cnt_type {poc_type}")),
        }
        let max_num_ref_frames = r.ue();
        let gaps_allowed = r.flag();
        let width_mbs = r.ue().saturating_add(1);
        let height_map_units = r.ue().saturating_add(1);
        if width_mbs > 1 << 12 || height_map_units > 1 << 12 {
            return Err("SPS picture size out of range".into());
        }
        let frame_mbs_only = r.flag();
        let mut mbaff = false;
        if !frame_mbs_only {
            mbaff = r.flag();
        }
        let height_mbs = height_map_units * if frame_mbs_only { 1 } else { 2 };
        let direct_8x8_inference = r.flag();
        let mut crop = [0u32; 4];
        if r.flag() {
            let (cx, cy) = match chroma_format_idc {
                0 => (1, 2 - frame_mbs_only as u32),
                _ => (2, 2 * (2 - frame_mbs_only as u32)),
            };
            crop = [r.ue(), r.ue(), r.ue(), r.ue()];
            crop = [
                crop[0].saturating_mul(cx),
                crop[1].saturating_mul(cx),
                crop[2].saturating_mul(cy),
                crop[3].saturating_mul(cy),
            ];
            // The decoder subtracts these from the coded size.
            if crop[0].saturating_add(crop[1]) >= width_mbs * 16 || crop[2].saturating_add(crop[3]) >= height_mbs * 16 {
                return Err("SPS cropping exceeds the picture".into());
            }
        }
        let (mut colour_primaries, mut transfer_characteristics) = (None, None);
        let mut matrix_coefficients = None;
        let mut full_range = false;
        if r.flag() {
            // vui_parameters(): only as far as the colour description.
            if r.flag() {
                let idc = r.u(8);
                if idc == 255 {
                    r.skip(32);
                }
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
        }
        if width_mbs > 1024 || height_mbs > 1024 {
            return Err("picture size out of range".into());
        }
        if crop[0] + crop[1] >= width_mbs * 16 || crop[2] + crop[3] >= height_mbs * 16 {
            return Err("cropping removes the whole picture".into());
        }
        Ok(Sps {
            id,
            profile_idc,
            chroma_format_idc,
            bit_depth_luma,
            bit_depth_chroma,
            transform_bypass,
            scaling,
            log2_max_frame_num,
            poc_type,
            log2_max_poc_lsb,
            delta_pic_order_always_zero,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offset_for_ref_frame,
            max_num_ref_frames,
            gaps_allowed,
            width_mbs,
            height_mbs,
            frame_mbs_only,
            mbaff,
            direct_8x8_inference,
            crop,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub cabac: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_ref_idx_default: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub chroma_qp_offset: [i32; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    /// Present lists from the PPS, or `None` to use the SPS's.
    pub scaling: Option<ScalingLists>,
}

impl Pps {
    /// Parse against the SPS it names (needed for the scaling-list count and
    /// fall-back).
    pub fn parse(rbsp: &[u8], spss: &[Option<Sps>]) -> Result<Pps> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue();
        let sps_id = r.ue();
        if id > 255 || sps_id > 31 {
            return Err("PPS id out of range".into());
        }
        let sps = spss[sps_id as usize]
            .as_ref()
            .ok_or_else(|| format!("PPS {id} names missing SPS {sps_id}"))?;
        let cabac = r.flag();
        let bottom_field_pic_order_in_frame_present = r.flag();
        let num_slice_groups = r.ue().saturating_add(1);
        if num_slice_groups > 1 {
            return Err("slice groups (FMO) are not supported".into());
        }
        let num_ref_idx_default = [r.ue().saturating_add(1), r.ue().saturating_add(1)];
        if num_ref_idx_default[0] > 32 || num_ref_idx_default[1] > 32 {
            return Err("num_ref_idx out of range".into());
        }
        let weighted_pred = r.flag();
        let weighted_bipred_idc = r.u(2);
        let pic_init_qp = r.se().saturating_add(26);
        let _pic_init_qs = r.se().saturating_add(26);
        let cb_off = r.se();
        let deblocking_filter_control_present = r.flag();
        let constrained_intra_pred = r.flag();
        let redundant_pic_cnt_present = r.flag();
        let mut transform_8x8_mode = false;
        let mut scaling = None;
        let mut cr_off = cb_off;
        if r.more_rbsp_data() {
            transform_8x8_mode = r.flag();
            if r.flag() {
                let n = 6 + if sps.chroma_format_idc != 3 { 2 } else { 6 } * transform_8x8_mode as usize;
                scaling = Some(read_scaling_lists(&mut r, n, sps.scaling.as_ref()));
            }
            cr_off = r.se();
        }
        Ok(Pps {
            id,
            sps_id,
            cabac,
            bottom_field_pic_order_in_frame_present,
            num_ref_idx_default,
            weighted_pred,
            weighted_bipred_idc,
            pic_init_qp,
            chroma_qp_offset: [cb_off, cr_off],
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            transform_8x8_mode,
            scaling,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
}

/// One `ref_pic_list_modification` operation.
#[derive(Clone, Copy, Debug)]
pub struct ListMod {
    pub idc: u32,
    pub value: u32,
}

/// One `memory_management_control_operation`.
#[derive(Clone, Copy, Debug)]
pub struct Mmco {
    pub op: u32,
    pub diff_pic_nums_minus1: u32,
    pub long_term_pic_num: u32,
    pub long_term_frame_idx: u32,
    pub max_long_term_frame_idx_plus1: u32,
}

/// Explicit weighted prediction for one list: `[ref][Y, Cb, Cr]` of
/// `(weight, offset)`.
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    pub luma_log2_denom: u32,
    pub chroma_log2_denom: u32,
    pub weights: [Vec<[(i32, i32); 3]>; 2],
    /// Whether any explicit flag was set for `[list][ref]` — when none was,
    /// the default (unweighted) formula applies.
    pub explicit: [Vec<bool>; 2],
}

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub first_mb: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub idr: bool,
    pub nal_ref_idc: u32,
    pub poc_lsb: u32,
    pub delta_poc_bottom: i32,
    pub delta_poc: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub direct_spatial: bool,
    pub num_ref_idx: [u32; 2],
    pub list_mods: [Vec<ListMod>; 2],
    pub weights: Option<WeightTable>,
    pub long_term_reference: bool,
    pub mmcos: Option<Vec<Mmco>>,
    pub cabac_init_idc: u32,
    pub qp: i32,
    pub disable_deblocking: u32,
    pub alpha_offset: i32,
    pub beta_offset: i32,
    /// Bit offset of `slice_data()` in the RBSP.
    pub data_bit_offset: usize,
}

impl SliceHeader {
    pub fn parse(
        rbsp: &[u8],
        nal_type: u32,
        nal_ref_idc: u32,
        spss: &[Option<Sps>],
        ppss: &[Option<Pps>],
    ) -> Result<SliceHeader> {
        let mut r = BitReader::new(rbsp);
        let first_mb = r.ue();
        let slice_type = match r.ue() % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            t => return Err(format!("SP/SI slices (type {t}) are not supported")),
        };
        let pps_id = r.ue();
        let pps = ppss
            .get(pps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| format!("slice names missing PPS {pps_id}"))?;
        let sps = spss[pps.sps_id as usize]
            .as_ref()
            .ok_or("slice's PPS names a missing SPS")?;
        let frame_num = r.u(sps.log2_max_frame_num);
        if !sps.frame_mbs_only {
            if r.flag() {
                return Err("interlaced (field) pictures are not supported".into());
            }
            if sps.mbaff {
                return Err("MBAFF interlaced pictures are not supported".into());
            }
        }
        let idr = nal_type == 5;
        if idr {
            r.ue();
        }
        let mut poc_lsb = 0;
        let mut delta_poc_bottom = 0;
        let mut delta_poc = [0; 2];
        if sps.poc_type == 0 {
            poc_lsb = r.u(sps.log2_max_poc_lsb);
            if pps.bottom_field_pic_order_in_frame_present {
                delta_poc_bottom = r.se();
            }
        }
        if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
            delta_poc[0] = r.se();
            if pps.bottom_field_pic_order_in_frame_present {
                delta_poc[1] = r.se();
            }
        }
        let mut redundant_pic_cnt = 0;
        if pps.redundant_pic_cnt_present {
            redundant_pic_cnt = r.ue();
        }
        let mut direct_spatial = false;
        if slice_type == SliceType::B {
            direct_spatial = r.flag();
        }
        let mut num_ref_idx = [0, 0];
        if slice_type != SliceType::I {
            num_ref_idx = pps.num_ref_idx_default;
            if r.flag() {
                num_ref_idx[0] = r.ue().saturating_add(1);
                if slice_type == SliceType::B {
                    num_ref_idx[1] = r.ue().saturating_add(1);
                }
            }
            if slice_type != SliceType::B {
                num_ref_idx[1] = 0;
            }
            if num_ref_idx[0] > 32 || num_ref_idx[1] > 32 {
                return Err("num_ref_idx_active out of range".into());
            }
        }
        let mut list_mods: [Vec<ListMod>; 2] = [Vec::new(), Vec::new()];
        for list in 0..2 {
            if ((list == 0 && slice_type != SliceType::I) || (list == 1 && slice_type == SliceType::B))
                && r.flag() {
                    loop {
                        let idc = r.ue();
                        if idc == 3 {
                            break;
                        }
                        if idc > 3 || list_mods[list].len() > 64 {
                            return Err("bad ref_pic_list_modification".into());
                        }
                        let value = r.ue();
                        list_mods[list].push(ListMod { idc, value });
                    }
                }
        }
        let mut weights = None;
        if (pps.weighted_pred && slice_type == SliceType::P)
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B)
        {
            let mut w = WeightTable {
                luma_log2_denom: r.ue(),
                ..Default::default()
            };
            if sps.chroma_format_idc != 0 {
                w.chroma_log2_denom = r.ue();
            }
            if w.luma_log2_denom > 7 || w.chroma_log2_denom > 7 {
                return Err("weight denominator out of range".into());
            }
            let lists = if slice_type == SliceType::B { 2 } else { 1 };
            for list in 0..lists {
                for _ in 0..num_ref_idx[list] {
                    let mut e = [
                        (1 << w.luma_log2_denom, 0),
                        (1 << w.chroma_log2_denom, 0),
                        (1 << w.chroma_log2_denom, 0),
                    ];
                    let mut any = false;
                    if r.flag() {
                        e[0] = (r.se(), r.se());
                        any = true;
                    }
                    if sps.chroma_format_idc != 0 && r.flag() {
                        e[1] = (r.se(), r.se());
                        e[2] = (r.se(), r.se());
                        any = true;
                    }
                    w.weights[list].push(e);
                    w.explicit[list].push(any);
                }
            }
            weights = Some(w);
        }
        let mut long_term_reference = false;
        let mut mmcos = None;
        if nal_ref_idc != 0 {
            if idr {
                r.skip(1); // no_output_of_prior_pics_flag
                long_term_reference = r.flag();
            } else if r.flag() {
                let mut ops = Vec::new();
                loop {
                    let op = r.ue();
                    if op == 0 {
                        break;
                    }
                    if op > 6 || ops.len() > 64 {
                        return Err("bad memory_management_control_operation".into());
                    }
                    let mut m = Mmco {
                        op,
                        diff_pic_nums_minus1: 0,
                        long_term_pic_num: 0,
                        long_term_frame_idx: 0,
                        max_long_term_frame_idx_plus1: 0,
                    };
                    if op == 1 || op == 3 {
                        m.diff_pic_nums_minus1 = r.ue();
                    }
                    if op == 2 {
                        m.long_term_pic_num = r.ue();
                    }
                    if op == 3 || op == 6 {
                        m.long_term_frame_idx = r.ue();
                    }
                    if op == 4 {
                        m.max_long_term_frame_idx_plus1 = r.ue();
                    }
                    ops.push(m);
                }
                mmcos = Some(ops);
            }
        }
        let mut cabac_init_idc = 0;
        if pps.cabac && slice_type != SliceType::I {
            cabac_init_idc = r.ue();
            if cabac_init_idc > 2 {
                return Err("cabac_init_idc out of range".into());
            }
        }
        let qp = pps.pic_init_qp.saturating_add(r.se());
        let qp_min = -6 * (sps.bit_depth_luma as i32 - 8);
        if qp < qp_min || qp > 51 {
            return Err(format!("slice QP {qp} out of range"));
        }
        let mut disable_deblocking = 0;
        let mut alpha_offset = 0;
        let mut beta_offset = 0;
        if pps.deblocking_filter_control_present {
            disable_deblocking = r.ue();
            if disable_deblocking > 2 {
                return Err("disable_deblocking_filter_idc out of range".into());
            }
            if disable_deblocking != 1 {
                alpha_offset = r.se().saturating_mul(2);
                beta_offset = r.se().saturating_mul(2);
            }
        }
        if r.overrun() {
            return Err("slice header runs past the NAL".into());
        }
        Ok(SliceHeader {
            first_mb,
            slice_type,
            pps_id,
            frame_num,
            idr,
            nal_ref_idc,
            poc_lsb,
            delta_poc_bottom,
            delta_poc,
            redundant_pic_cnt,
            direct_spatial,
            num_ref_idx,
            list_mods,
            weights,
            long_term_reference,
            mmcos,
            cabac_init_idc,
            qp,
            disable_deblocking,
            alpha_offset,
            beta_offset,
            data_bit_offset: r.pos(),
        })
    }
}
