//! VP9 motion-vector entropy coding (libvpx `vp9_encodemv` / `read_mv`).
//!
//! MVs are in 1/8-pel units. With `allow_high_precision_mv = 0` the decoder
//! forces the high-precision bit to 1, so only even MV values round-trip.

use crate::vp9::bool_coder::BoolEncoder;

/// `default_nmv_context.joints[MV_JOINTS - 1]`.
pub const DEFAULT_MV_JOINTS: [u8; 3] = [32, 64, 96];

/// One component of `default_nmv_context`.
#[derive(Clone, Copy)]
pub struct NmvComponent {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: u8,
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
}

/// Vertical then horizontal — `default_nmv_context.comps`.
pub const DEFAULT_NMV_COMPS: [NmvComponent; 2] = [
    NmvComponent {
        sign: 128,
        classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
        class0: 216,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
    },
    NmvComponent {
        sign: 128,
        classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
        class0: 208,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
    },
];

pub const MV_JOINT_ZERO: i8 = 0;
pub const MV_JOINT_HNZVZ: i8 = 1;
pub const MV_JOINT_HZVNZ: i8 = 2;
pub const MV_JOINT_HNZVNZ: i8 = 3;

/// `vp9_mv_joint_tree` (`-MV_JOINT_ZERO` is 0).
pub const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

/// `vp9_mv_class_tree`.
pub const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];

/// `vp9_mv_class0_tree`.
pub const MV_CLASS0_TREE: [i8; 2] = [0, -1];

/// `vp9_mv_fp_tree`.
pub const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

const CLASS0_BITS: i32 = 1;
const CLASS0_SIZE: i32 = 1 << CLASS0_BITS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    pub fn joint(self) -> i8 {
        match (self.row == 0, self.col == 0) {
            (true, true) => MV_JOINT_ZERO,
            (true, false) => MV_JOINT_HNZVZ,
            (false, true) => MV_JOINT_HZVNZ,
            (false, false) => MV_JOINT_HNZVNZ,
        }
    }
}

fn mv_class_base(c: i32) -> i32 {
    if c != 0 {
        CLASS0_SIZE << (c + 2)
    } else {
        0
    }
}

fn get_mv_class(z: i32) -> (i32, i32) {
    let c = if z >= CLASS0_SIZE * 4096 {
        10
    } else {
        log2_floor((z >> 3) as u32) as i32
    };
    (c, z - mv_class_base(c))
}

fn log2_floor(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        31 - n.leading_zeros()
    }
}

fn use_mv_hp(ref_mv: Mv) -> bool {
    ref_mv.row.unsigned_abs() < 64 && ref_mv.col.unsigned_abs() < 64
}

/// Encode `mv` as a delta from `ref_mv` (`vp9_encode_mv`).
pub fn write_mv(e: &mut BoolEncoder, mv: Mv, ref_mv: Mv, allow_hp: bool) {
    let diff = Mv {
        row: mv.row - ref_mv.row,
        col: mv.col - ref_mv.col,
    };
    let joint = diff.joint();
    write_tree(e, &MV_JOINT_TREE, &DEFAULT_MV_JOINTS, joint);

    let use_hp = allow_hp && use_mv_hp(ref_mv);
    if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
        write_mv_component(e, diff.row, &DEFAULT_NMV_COMPS[0], use_hp);
    }
    if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
        write_mv_component(e, diff.col, &DEFAULT_NMV_COMPS[1], use_hp);
    }
}

fn write_mv_component(e: &mut BoolEncoder, comp: i16, mvcomp: &NmvComponent, use_hp: bool) {
    assert_ne!(comp, 0);
    let sign = comp < 0;
    let mag = i32::from(comp.unsigned_abs());
    let z = mag - 1;
    let (mv_class, offset) = get_mv_class(z);
    let d = offset >> 3;
    let fr = (offset >> 1) & 3;
    let hp = offset & 1;

    e.put_bool(mvcomp.sign, sign);
    write_tree(e, &MV_CLASS_TREE, &mvcomp.classes, mv_class as i8);

    if mv_class == 0 {
        e.put_bool(mvcomp.class0, d != 0);
        write_tree(e, &MV_FP_TREE, &mvcomp.class0_fp[d as usize], fr as i8);
    } else {
        // `n = mv_class + CLASS0_BITS - 1`
        let nbits = mv_class + CLASS0_BITS - 1;
        for i in 0..nbits {
            e.put_bool(mvcomp.bits[i as usize], ((d >> i) & 1) != 0);
        }
        write_tree(e, &MV_FP_TREE, &mvcomp.fp, fr as i8);
    }

    if use_hp {
        // Default hp probs from nmvc; unused while allow_hp=false.
        e.put_bool(128, hp != 0);
    }
}

fn write_tree(e: &mut BoolEncoder, tree: &[i8], probs: &[u8], token: i8) {
    let path = tree_path(tree, 0, token).expect("token not in tree");
    for (node, bit) in path {
        e.put_bool(probs[node >> 1], bit);
    }
}

fn tree_path(tree: &[i8], node: usize, token: i8) -> Option<Vec<(usize, bool)>> {
    for bit in 0..2 {
        let child = tree[node + bit];
        if child <= 0 {
            if -child == token {
                return Some(vec![(node, bit == 1)]);
            }
        } else if let Some(mut rest) = tree_path(tree, child as usize, token) {
            rest.insert(0, (node, bit == 1));
            return Some(rest);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class0_one_and_two_pel() {
        let (c, o) = get_mv_class(7); // mag 8
        assert_eq!((c, o >> 3, (o >> 1) & 3, o & 1), (0, 0, 3, 1));
        let (c, o) = get_mv_class(15); // mag 16
        assert_eq!((c, o >> 3, (o >> 1) & 3, o & 1), (0, 1, 3, 1));
    }

    #[test]
    fn joint_types() {
        assert_eq!(Mv { row: 0, col: 0 }.joint(), MV_JOINT_ZERO);
        assert_eq!(Mv { row: 0, col: 16 }.joint(), MV_JOINT_HNZVZ);
        assert_eq!(Mv { row: 16, col: 0 }.joint(), MV_JOINT_HZVNZ);
        assert_eq!(Mv { row: 16, col: 16 }.joint(), MV_JOINT_HNZVNZ);
    }

    #[test]
    fn encode_zero_diff_finishes() {
        let mut e = BoolEncoder::new();
        write_mv(
            &mut e,
            Mv { row: 16, col: 0 },
            Mv { row: 16, col: 0 },
            false,
        );
        assert!(!e.finish().is_empty());
    }
}
