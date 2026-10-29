//! AV1 enums, generated from dav1d 1.5.1 `src/levels.h` by
//! `tribev2-ref/scripts/av1gen/gen_levels.py`. Do not edit by hand.
#![allow(dead_code)]

// enum ObuMetaType
pub const OBU_META_HDR_CLL: u8 = 1;
pub const OBU_META_HDR_MDCV: u8 = 2;
pub const OBU_META_SCALABILITY: u8 = 3;
pub const OBU_META_ITUT_T35: u8 = 4;
pub const OBU_META_TIMECODE: u8 = 5;

// enum TxfmSize
pub const TX_4X4: usize = 0;
pub const TX_8X8: usize = 1;
pub const TX_16X16: usize = 2;
pub const TX_32X32: usize = 3;
pub const TX_64X64: usize = 4;
pub const N_TX_SIZES: usize = 5;

// enum BlockLevel
pub const BL_128X128: usize = 0;
pub const BL_64X64: usize = 1;
pub const BL_32X32: usize = 2;
pub const BL_16X16: usize = 3;
pub const BL_8X8: usize = 4;
pub const N_BL_LEVELS: usize = 5;

// enum RectTxfmSize
pub const RTX_4X8: usize = 5;
pub const RTX_8X4: usize = 6;
pub const RTX_8X16: usize = 7;
pub const RTX_16X8: usize = 8;
pub const RTX_16X32: usize = 9;
pub const RTX_32X16: usize = 10;
pub const RTX_32X64: usize = 11;
pub const RTX_64X32: usize = 12;
pub const RTX_4X16: usize = 13;
pub const RTX_16X4: usize = 14;
pub const RTX_8X32: usize = 15;
pub const RTX_32X8: usize = 16;
pub const RTX_16X64: usize = 17;
pub const RTX_64X16: usize = 18;
pub const N_RECT_TX_SIZES: usize = 19;

// enum TxfmType
pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;
pub const FLIPADST_DCT: u8 = 4;
pub const DCT_FLIPADST: u8 = 5;
pub const FLIPADST_FLIPADST: u8 = 6;
pub const ADST_FLIPADST: u8 = 7;
pub const FLIPADST_ADST: u8 = 8;
pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;
pub const N_TX_TYPES: usize = 16;
pub const WHT_WHT: u8 = 16;
pub const N_TX_TYPES_PLUS_LL: usize = 17;

// enum TxClass
pub const TX_CLASS_2D: u8 = 0;
pub const TX_CLASS_H: u8 = 1;
pub const TX_CLASS_V: u8 = 2;

// enum IntraPredMode
pub const DC_PRED: usize = 0;
pub const VERT_PRED: usize = 1;
pub const HOR_PRED: usize = 2;
pub const DIAG_DOWN_LEFT_PRED: usize = 3;
pub const DIAG_DOWN_RIGHT_PRED: usize = 4;
pub const VERT_RIGHT_PRED: usize = 5;
pub const HOR_DOWN_PRED: usize = 6;
pub const HOR_UP_PRED: usize = 7;
pub const VERT_LEFT_PRED: usize = 8;
pub const SMOOTH_PRED: usize = 9;
pub const SMOOTH_V_PRED: usize = 10;
pub const SMOOTH_H_PRED: usize = 11;
pub const PAETH_PRED: usize = 12;
pub const N_INTRA_PRED_MODES: usize = 13;
pub const CFL_PRED: usize = 13;
pub const N_UV_INTRA_PRED_MODES: usize = 14;
pub const N_IMPL_INTRA_PRED_MODES: usize = 14;
pub const LEFT_DC_PRED: usize = 3;
pub const TOP_DC_PRED: usize = 4;
pub const DC_128_PRED: usize = 5;
pub const Z1_PRED: usize = 6;
pub const Z2_PRED: usize = 7;
pub const Z3_PRED: usize = 8;
pub const FILTER_PRED: usize = 13;

// enum InterIntraPredMode
pub const II_DC_PRED: u8 = 0;
pub const II_VERT_PRED: u8 = 1;
pub const II_HOR_PRED: u8 = 2;
pub const II_SMOOTH_PRED: u8 = 3;
pub const N_INTER_INTRA_PRED_MODES: usize = 4;

// enum BlockPartition
pub const PARTITION_NONE: usize = 0;
pub const PARTITION_H: usize = 1;
pub const PARTITION_V: usize = 2;
pub const PARTITION_SPLIT: usize = 3;
pub const PARTITION_T_TOP_SPLIT: usize = 4;
pub const PARTITION_T_BOTTOM_SPLIT: usize = 5;
pub const PARTITION_T_LEFT_SPLIT: usize = 6;
pub const PARTITION_T_RIGHT_SPLIT: usize = 7;
pub const PARTITION_H4: usize = 8;
pub const PARTITION_V4: usize = 9;
pub const N_PARTITIONS: usize = 10;
pub const N_SUB8X8_PARTITIONS: usize = 4;

// enum BlockSize
pub const BS_128X128: usize = 0;
pub const BS_128X64: usize = 1;
pub const BS_64X128: usize = 2;
pub const BS_64X64: usize = 3;
pub const BS_64X32: usize = 4;
pub const BS_64X16: usize = 5;
pub const BS_32X64: usize = 6;
pub const BS_32X32: usize = 7;
pub const BS_32X16: usize = 8;
pub const BS_32X8: usize = 9;
pub const BS_16X64: usize = 10;
pub const BS_16X32: usize = 11;
pub const BS_16X16: usize = 12;
pub const BS_16X8: usize = 13;
pub const BS_16X4: usize = 14;
pub const BS_8X32: usize = 15;
pub const BS_8X16: usize = 16;
pub const BS_8X8: usize = 17;
pub const BS_8X4: usize = 18;
pub const BS_4X16: usize = 19;
pub const BS_4X8: usize = 20;
pub const BS_4X4: usize = 21;
pub const N_BS_SIZES: usize = 22;

// enum Filter2d
pub const FILTER_2D_8TAP_REGULAR: usize = 0;
pub const FILTER_2D_8TAP_REGULAR_SMOOTH: usize = 1;
pub const FILTER_2D_8TAP_REGULAR_SHARP: usize = 2;
pub const FILTER_2D_8TAP_SHARP_REGULAR: usize = 3;
pub const FILTER_2D_8TAP_SHARP_SMOOTH: usize = 4;
pub const FILTER_2D_8TAP_SHARP: usize = 5;
pub const FILTER_2D_8TAP_SMOOTH_REGULAR: usize = 6;
pub const FILTER_2D_8TAP_SMOOTH: usize = 7;
pub const FILTER_2D_8TAP_SMOOTH_SHARP: usize = 8;
pub const FILTER_2D_BILINEAR: usize = 9;
pub const N_2D_FILTERS: usize = 10;

// enum MVJoint
pub const MV_JOINT_ZERO: usize = 0;
pub const MV_JOINT_H: usize = 1;
pub const MV_JOINT_V: usize = 2;
pub const MV_JOINT_HV: usize = 3;
pub const N_MV_JOINTS: usize = 4;

// enum InterPredMode
pub const NEARESTMV: u8 = 0;
pub const NEARMV: u8 = 1;
pub const GLOBALMV: u8 = 2;
pub const NEWMV: u8 = 3;
pub const N_INTER_PRED_MODES: usize = 4;

// enum DRL_PROXIMITY
pub const NEAREST_DRL: u8 = 0;
pub const NEARER_DRL: u8 = 1;
pub const NEAR_DRL: u8 = 2;
pub const NEARISH_DRL: u8 = 3;

// enum CompInterPredMode
pub const NEARESTMV_NEARESTMV: u8 = 0;
pub const NEARMV_NEARMV: u8 = 1;
pub const NEARESTMV_NEWMV: u8 = 2;
pub const NEWMV_NEARESTMV: u8 = 3;
pub const NEARMV_NEWMV: u8 = 4;
pub const NEWMV_NEARMV: u8 = 5;
pub const GLOBALMV_GLOBALMV: u8 = 6;
pub const NEWMV_NEWMV: u8 = 7;
pub const N_COMP_INTER_PRED_MODES: usize = 8;

// enum CompInterType
pub const COMP_INTER_NONE: u8 = 0;
pub const COMP_INTER_WEIGHTED_AVG: u8 = 1;
pub const COMP_INTER_AVG: u8 = 2;
pub const COMP_INTER_SEG: u8 = 3;
pub const COMP_INTER_WEDGE: u8 = 4;

// enum InterIntraType
pub const INTER_INTRA_NONE: u8 = 0;
pub const INTER_INTRA_BLEND: u8 = 1;
pub const INTER_INTRA_WEDGE: u8 = 2;

// enum MotionMode
pub const MM_TRANSLATION: u8 = 0;
pub const MM_OBMC: u8 = 1;
pub const MM_WARP: u8 = 2;

pub const N_SWITCHABLE_FILTERS: u8 = 3;
pub const N_FILTERS: u8 = 4;
pub const FILTER_8TAP_REGULAR: u8 = 0;
pub const FILTER_8TAP_SMOOTH: u8 = 1;
pub const FILTER_8TAP_SHARP: u8 = 2;
pub const FILTER_BILINEAR: u8 = 3;
pub const FILTER_SWITCHABLE: u8 = 4;
