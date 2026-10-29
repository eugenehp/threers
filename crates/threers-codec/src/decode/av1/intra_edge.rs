//! Edge-availability tree for the partition walk (dav1d `src/intra_edge.c`).

use std::sync::OnceLock;

use super::levels::*;

pub const EDGE_I444_TOP_HAS_RIGHT: u8 = 1 << 0;
pub const EDGE_I422_TOP_HAS_RIGHT: u8 = 1 << 1;
pub const EDGE_I420_TOP_HAS_RIGHT: u8 = 1 << 2;
pub const EDGE_I444_LEFT_HAS_BOTTOM: u8 = 1 << 3;
pub const EDGE_I422_LEFT_HAS_BOTTOM: u8 = 1 << 4;
pub const EDGE_I420_LEFT_HAS_BOTTOM: u8 = 1 << 5;
pub const EDGE_ALL_TOP_HAS_RIGHT: u8 = EDGE_I444_TOP_HAS_RIGHT | EDGE_I422_TOP_HAS_RIGHT | EDGE_I420_TOP_HAS_RIGHT;
pub const EDGE_ALL_LEFT_HAS_BOTTOM: u8 = EDGE_I444_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM | EDGE_I420_LEFT_HAS_BOTTOM;
pub const EDGE_ALL_TR_AND_BL: u8 = EDGE_ALL_TOP_HAS_RIGHT | EDGE_ALL_LEFT_HAS_BOTTOM;

#[derive(Clone, Copy, Default)]
pub struct EdgeNode {
    pub o: u8,
    pub h: [u8; 2],
    pub v: [u8; 2],
    /// Branch nodes (bl < BL_8X8).
    pub h4: u8,
    pub v4: u8,
    pub split: [usize; 4],
    /// Tip nodes (BL_8X8).
    pub tip_split: [u8; 3],
}

pub struct EdgeTree {
    pub nodes: Vec<EdgeNode>,
    /// Roots for BL_128X128 and BL_64X64 superblocks.
    pub root: [usize; 2],
}

fn init_edges(node: &mut EdgeNode, bl: usize, edge_flags: u8) {
    node.o = edge_flags;
    node.h[0] = edge_flags | EDGE_ALL_LEFT_HAS_BOTTOM;
    node.v[0] = edge_flags | EDGE_ALL_TOP_HAS_RIGHT;
    if bl == BL_8X8 {
        node.h[1] = edge_flags & (EDGE_ALL_LEFT_HAS_BOTTOM | EDGE_I420_TOP_HAS_RIGHT);
        node.v[1] = edge_flags & (EDGE_ALL_TOP_HAS_RIGHT | EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
        node.tip_split[0] = (edge_flags & EDGE_ALL_TOP_HAS_RIGHT) | EDGE_I422_LEFT_HAS_BOTTOM;
        node.tip_split[1] = edge_flags | EDGE_I444_TOP_HAS_RIGHT;
        node.tip_split[2] = edge_flags & (EDGE_I420_TOP_HAS_RIGHT | EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
    } else {
        node.h[1] = edge_flags & EDGE_ALL_LEFT_HAS_BOTTOM;
        node.v[1] = edge_flags & EDGE_ALL_TOP_HAS_RIGHT;
        node.h4 = EDGE_ALL_LEFT_HAS_BOTTOM;
        node.v4 = EDGE_ALL_TOP_HAS_RIGHT;
        if bl == BL_16X16 {
            node.h4 |= edge_flags & EDGE_I420_TOP_HAS_RIGHT;
            node.v4 |= edge_flags & (EDGE_I420_LEFT_HAS_BOTTOM | EDGE_I422_LEFT_HAS_BOTTOM);
        }
    }
}

fn init_mode_node(nodes: &mut Vec<EdgeNode>, bl: usize, top_has_right: bool, left_has_bottom: bool) -> usize {
    let idx = nodes.len();
    nodes.push(EdgeNode::default());
    let mut n0 = EdgeNode::default();
    init_edges(
        &mut n0,
        bl,
        (if top_has_right { EDGE_ALL_TOP_HAS_RIGHT } else { 0 }) | (if left_has_bottom { EDGE_ALL_LEFT_HAS_BOTTOM } else { 0 }),
    );
    for n in 0..4 {
        let thr = !(n == 3 || (n == 1 && !top_has_right));
        let lhb = n == 0 || (n == 2 && left_has_bottom);
        n0.split[n] = if bl == BL_16X16 {
            let t = nodes.len();
            let mut tip = EdgeNode::default();
            init_edges(
                &mut tip,
                bl + 1,
                (if thr { EDGE_ALL_TOP_HAS_RIGHT } else { 0 }) | (if lhb { EDGE_ALL_LEFT_HAS_BOTTOM } else { 0 }),
            );
            nodes.push(tip);
            t
        } else {
            init_mode_node(nodes, bl + 1, thr, lhb)
        };
    }
    nodes[idx] = n0;
    idx
}

pub fn tree() -> &'static EdgeTree {
    static T: OnceLock<EdgeTree> = OnceLock::new();
    T.get_or_init(|| {
        let mut nodes = Vec::new();
        let r128 = init_mode_node(&mut nodes, BL_128X128, true, false);
        let r64 = init_mode_node(&mut nodes, BL_64X64, true, false);
        EdgeTree { nodes, root: [r128, r64] }
    })
}
