use crate::math::{Box3, Matrix4};
use std::cell::Cell;

use super::mesh_bvh::MeshBvh;

// A correct dual-BVH descent visits each (node_a, node_b) pair at most once and
// only ever descends, so any single recursion path is bounded by the total node
// count. On some degenerate curved-CSG inputs the upstream algorithm can recurse
// without progress: two branches keep passing an unshrinking box back and forth,
// so the *tree* of recursive calls explodes exponentially even though each single
// path stays under the depth cap (so it neither overflows the stack nor returns —
// it just spins for hours). We guard both failure modes:
//   * DEPTH / MAX_DEPTH bounds any one recursion path (stack-overflow guard);
//   * VISITS / VISIT_CAP bounds the *total* number of `traverse` calls. A correct
//     descent makes at most ~2·na·nb node-pair visits, so a generous multiple of
//     that can only be exceeded by the pathological re-traversal — when it is, we
//     bail the whole cast with the pairs found so far. bvhcast only feeds the
//     float CSG fallback (the exact kernel handles the well-formed cases), so an
//     early, slightly-incomplete result is strictly better than never returning,
//     and it holds identically on wasm (no thread to time out).
thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static MAX_DEPTH: Cell<u32> = const { Cell::new(0) };
    static VISITS: Cell<u64> = const { Cell::new(0) };
    static VISIT_CAP: Cell<u64> = const { Cell::new(0) };
}

/// Decrements the recursion counter when a `traverse` frame unwinds (including
/// the early return at the depth cap).
struct DepthGuard;
impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Dual-BVH traversal matching upstream three-mesh-bvh `cast/bvhcast.js`.
/// Returns BVH-layout triangle index pairs `(ia, ib)` for geometry A and B.
pub fn bvhcast(a: &MeshBvh, b: &MeshBvh, matrix_to_local: &Matrix4) -> Vec<(usize, usize)> {
    if a.node_count() == 0 || b.node_count() == 0 {
        return Vec::new();
    }
    // A monotone descent path can't be longer than the sum of node counts.
    MAX_DEPTH.with(|m| m.set((a.node_count() + b.node_count()) as u32 + 1024));
    DEPTH.with(|d| d.set(0));
    // Total-work budget: ~2·na·nb is the correct-traversal ceiling; the extra ×4
    // and the 1M floor give ample headroom for small meshes and the reversed
    // double-descent, while still capping the exponential runaway far short of a
    // hang.
    let (na, nb) = (a.node_count() as u64, b.node_count() as u64);
    let cap = na.saturating_mul(nb).saturating_mul(4).saturating_add(1 << 20);
    VISIT_CAP.with(|c| c.set(cap));
    VISITS.with(|v| v.set(0));
    let mat_b_to_a = *matrix_to_local;
    let mat_a_to_b = matrix_to_local.invert();
    let curr_box = a.node_bounds(0).apply_matrix4(&mat_a_to_b);
    let mut pairs = Vec::new();
    traverse(
        a, b, 0, 0, mat_b_to_a, mat_a_to_b, &mut pairs, curr_box, false,
    );
    pairs
}

fn traverse(
    a: &MeshBvh,
    b: &MeshBvh,
    node_a: u32,
    node_b: u32,
    mat_2_to_1: Matrix4,
    mat_1_to_2: Matrix4,
    pairs: &mut Vec<(usize, usize)>,
    curr_box: Box3,
    reversed: bool,
) {
    DEPTH.with(|d| d.set(d.get() + 1));
    let _guard = DepthGuard;
    if DEPTH.with(|d| d.get()) > MAX_DEPTH.with(|m| m.get()) {
        return; // degenerate non-terminating descent — bail out of this branch
    }
    // Total-work budget: once tripped, every subsequent frame returns in O(1), so
    // the whole traversal unwinds promptly instead of exploring exponentially.
    let visits = VISITS.with(|v| {
        let n = v.get() + 1;
        v.set(n);
        n
    });
    if visits > VISIT_CAP.with(|c| c.get()) {
        return;
    }

    let (s1, s2, n1, n2) = if reversed {
        (b, a, node_b, node_a)
    } else {
        (a, b, node_a, node_b)
    };

    let leaf1 = s1.is_leaf(n1);
    let leaf2 = s2.is_leaf(n2);

    if leaf1 && leaf2 {
        let (offset_a, count_a) = if reversed {
            s2.leaf_range(n2)
        } else {
            s1.leaf_range(n1)
        };
        let (offset_b, count_b) = if reversed {
            s1.leaf_range(n1)
        } else {
            s2.leaf_range(n2)
        };
        collect_leaf_pairs(offset_a, count_a, offset_b, count_b, pairs);
        return;
    }

    if leaf2 {
        let new_box = s2.node_bounds(n2).apply_matrix4(&mat_2_to_1);
        let cl = s1.child_left(n1);
        let cr = s1.child_right(n1);
        if new_box.intersects_box(&s1.node_bounds(cl)) {
            traverse(
                a,
                b,
                if reversed { node_a } else { cl },
                if reversed { cl } else { node_b },
                mat_1_to_2,
                mat_2_to_1,
                pairs,
                new_box,
                !reversed,
            );
        }
        if new_box.intersects_box(&s1.node_bounds(cr)) {
            traverse(
                a,
                b,
                if reversed { node_a } else { cr },
                if reversed { cr } else { node_b },
                mat_1_to_2,
                mat_2_to_1,
                pairs,
                new_box,
                !reversed,
            );
        }
        return;
    }

    let cl2 = s2.child_left(n2);
    let cr2 = s2.child_right(n2);
    let left_box2 = s2.node_bounds(cl2);
    let right_box2 = s2.node_bounds(cr2);
    let left_hit = curr_box.intersects_box(&left_box2);
    let right_hit = curr_box.intersects_box(&right_box2);

    if left_hit && right_hit {
        traverse(
            a, b, node_a, cl2, mat_2_to_1, mat_1_to_2, pairs, curr_box, reversed,
        );
        traverse(
            a, b, node_a, cr2, mat_2_to_1, mat_1_to_2, pairs, curr_box, reversed,
        );
    } else if left_hit {
        if leaf1 {
            traverse(
                a, b, node_a, cl2, mat_2_to_1, mat_1_to_2, pairs, curr_box, reversed,
            );
        } else {
            let new_box = left_box2.apply_matrix4(&mat_2_to_1);
            let cl1 = s1.child_left(n1);
            let cr1 = s1.child_right(n1);
            if new_box.intersects_box(&s1.node_bounds(cl1)) {
                traverse(
                    a,
                    b,
                    if reversed { cl2 } else { cl1 },
                    if reversed { cl1 } else { cl2 },
                    mat_1_to_2,
                    mat_2_to_1,
                    pairs,
                    new_box,
                    !reversed,
                );
            }
            if new_box.intersects_box(&s1.node_bounds(cr1)) {
                traverse(
                    a,
                    b,
                    if reversed { cl2 } else { cr1 },
                    if reversed { cr1 } else { cl2 },
                    mat_1_to_2,
                    mat_2_to_1,
                    pairs,
                    new_box,
                    !reversed,
                );
            }
        }
    } else if right_hit {
        if leaf1 {
            traverse(
                a, b, node_a, cr2, mat_2_to_1, mat_1_to_2, pairs, curr_box, reversed,
            );
        } else {
            let new_box = right_box2.apply_matrix4(&mat_2_to_1);
            let cl1 = s1.child_left(n1);
            let cr1 = s1.child_right(n1);
            if new_box.intersects_box(&s1.node_bounds(cl1)) {
                traverse(
                    a,
                    b,
                    if reversed { cr2 } else { cl1 },
                    if reversed { cl1 } else { cr2 },
                    mat_1_to_2,
                    mat_2_to_1,
                    pairs,
                    new_box,
                    !reversed,
                );
            }
            if new_box.intersects_box(&s1.node_bounds(cr1)) {
                traverse(
                    a,
                    b,
                    if reversed { cr2 } else { cr1 },
                    if reversed { cr1 } else { cr2 },
                    mat_1_to_2,
                    mat_2_to_1,
                    pairs,
                    new_box,
                    !reversed,
                );
            }
        }
    }
}

/// Upstream MeshBVH leaf iteration: outer B, inner A.
fn collect_leaf_pairs(
    offset_a: usize,
    count_a: usize,
    offset_b: usize,
    count_b: usize,
    pairs: &mut Vec<(usize, usize)>,
) {
    for ib in offset_b..offset_b + count_b {
        for ia in offset_a..offset_a + count_a {
            pairs.push((ia, ib));
        }
    }
}
