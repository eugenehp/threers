//! Compact median-split AABB BVH over a triangle soup — accelerates the two
//! O(N·M) hot paths: broad-phase candidate pairs (`overlaps`) and the ray-parity
//! inside test (`ray_leaves`). Built once per mesh and reused across all queries.

use super::V3;

struct Node {
    lo: V3,
    hi: V3,
    // Leaf iff `count > 0`: triangles `order[start..start+count]`.
    // Internal otherwise: children are node `start` (left) and `right`.
    start: u32,
    count: u32,
    right: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<usize>,
}

fn tri_box(t: &[V3; 3]) -> (V3, V3) {
    let mut lo = t[0];
    let mut hi = t[0];
    for v in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    (lo, hi)
}

impl Bvh {
    pub fn build(tris: &[[V3; 3]]) -> Bvh {
        let boxes: Vec<(V3, V3)> = tris.iter().map(tri_box).collect();
        let cent: Vec<V3> = boxes
            .iter()
            .map(|(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5])
            .collect();
        let mut order: Vec<usize> = (0..tris.len()).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !tris.is_empty() {
            build_node(&mut nodes, &mut order, &boxes, &cent, 0, tris.len());
        }
        Bvh { nodes, order }
    }

    /// Triangle indices whose AABB overlaps `qbox`.
    pub fn overlaps(&self, qbox: (V3, V3), out: &mut Vec<usize>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !box_overlap(&(n.lo, n.hi), &qbox) {
                continue;
            }
            if n.count > 0 {
                out.extend_from_slice(&self.order[n.start as usize..(n.start + n.count) as usize]);
            } else {
                stack.push(n.start);
                stack.push(n.right);
            }
        }
    }

    /// Triangle indices whose AABB the segment `p → p+dir` (t∈[0,1]) crosses.
    pub fn ray_leaves(&self, p: V3, dir: V3, out: &mut Vec<usize>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let inv = [
            if dir[0] != 0.0 { 1.0 / dir[0] } else { f64::INFINITY },
            if dir[1] != 0.0 { 1.0 / dir[1] } else { f64::INFINITY },
            if dir[2] != 0.0 { 1.0 / dir[2] } else { f64::INFINITY },
        ];
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !seg_box(p, dir, inv, n.lo, n.hi) {
                continue;
            }
            if n.count > 0 {
                out.extend_from_slice(&self.order[n.start as usize..(n.start + n.count) as usize]);
            } else {
                stack.push(n.start);
                stack.push(n.right);
            }
        }
    }
}

fn build_node(
    nodes: &mut Vec<Node>,
    order: &mut [usize],
    boxes: &[(V3, V3)],
    cent: &[V3],
    start: usize,
    end: usize,
) -> u32 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in &order[start..end] {
        for k in 0..3 {
            lo[k] = lo[k].min(boxes[i].0[k]);
            hi[k] = hi[k].max(boxes[i].1[k]);
        }
    }
    let idx = nodes.len() as u32;
    let count = end - start;
    if count <= 3 {
        nodes.push(Node { lo, hi, start: start as u32, count: count as u32, right: 0 });
        return idx;
    }
    // Split on the widest centroid axis at the median.
    let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap()).unwrap();
    let mid = start + count / 2;
    order[start..end].select_nth_unstable_by(count / 2, |&a, &b| {
        cent[a][axis].partial_cmp(&cent[b][axis]).unwrap()
    });
    nodes.push(Node { lo, hi, start: 0, count: 0, right: 0 });
    let left = build_node(nodes, order, boxes, cent, start, mid);
    let right = build_node(nodes, order, boxes, cent, mid, end);
    nodes[idx as usize].start = left;
    nodes[idx as usize].right = right;
    idx
}

fn box_overlap(a: &(V3, V3), b: &(V3, V3)) -> bool {
    for k in 0..3 {
        if a.0[k] > b.1[k] || b.0[k] > a.1[k] {
            return false;
        }
    }
    true
}

/// Does segment `p + t·dir`, `t ∈ [0,1]`, intersect the box `[lo,hi]`? (slab test)
fn seg_box(p: V3, dir: V3, inv: [f64; 3], lo: V3, hi: V3) -> bool {
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for k in 0..3 {
        if dir[k] == 0.0 {
            if p[k] < lo[k] || p[k] > hi[k] {
                return false;
            }
        } else {
            let mut ta = (lo[k] - p[k]) * inv[k];
            let mut tb = (hi[k] - p[k]) * inv[k];
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlaps_and_ray_match_bruteforce() {
        // Random-ish triangles; BVH queries must match brute force.
        let mut tris = Vec::new();
        let mut s = 12345u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f64) / (1u64 << 24) as f64 * 10.0 - 5.0
        };
        for _ in 0..200 {
            let o = [r(), r(), r()];
            tris.push([o, [o[0] + 0.5, o[1], o[2]], [o[0], o[1] + 0.5, o[2]]]);
        }
        let bvh = Bvh::build(&tris);
        let boxes: Vec<(V3, V3)> = tris.iter().map(tri_box).collect();

        // Broad-phase contract: candidates are a superset (no misses) — callers
        // re-test exactly. So every truly-overlapping triangle must be returned.
        let qbox = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mut got = Vec::new();
        bvh.overlaps(qbox, &mut got);
        let got_set: std::collections::HashSet<usize> = got.into_iter().collect();
        for i in 0..tris.len() {
            if box_overlap(&boxes[i], &qbox) {
                assert!(got_set.contains(&i), "overlaps missed triangle {i}");
            }
        }

        let (p, dir) = ([0.0, 0.0, 0.0], [3.0, 2.0, 1.0]);
        let inv = [1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0];
        let mut rl = Vec::new();
        bvh.ray_leaves(p, dir, &mut rl);
        let ray_set: std::collections::HashSet<usize> = rl.into_iter().collect();
        for i in 0..tris.len() {
            if seg_box(p, dir, inv, boxes[i].0, boxes[i].1) {
                assert!(ray_set.contains(&i), "ray_leaves missed triangle {i}");
            }
        }
    }
}
