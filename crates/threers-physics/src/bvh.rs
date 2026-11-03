//! Dynamic bounding-volume hierarchy.
//!
//! One structure serving three jobs: broad-phase pair finding, scene queries,
//! and the adjacency that island detection walks. Before this existed, every
//! raycast in the world was a loop over every body — fine at a hundred bodies,
//! ruinous at ten thousand.
//!
//! # Why a tree rather than a sorted sweep
//!
//! Sweep-and-prune is excellent at finding overlapping pairs and useless at
//! answering "what does this ray hit". A tree does both, and it does them
//! *incrementally*: bodies that did not move cost nothing.
//!
//! # Fat bounds
//!
//! Every leaf is stored with a margin around its true bounds. A body that
//! shifts slightly stays inside its fattened box, so nothing has to be
//! reinserted — which is the whole reason this is cheaper than rebuilding. The
//! margin trades a few false-positive pairs (which the narrow phase discards
//! anyway) for far fewer tree edits.

use crate::math::Aabb;
use threers::math::{Ray, Vector3};

/// Handle to an entry in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProxyId(u32);

impl ProxyId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

const NIL: u32 = u32::MAX;

/// How much slack to leave around a leaf's true bounds.
const MARGIN: f32 = 0.1;

/// How far ahead of a moving body to extend its fat bounds, as a multiple of
/// the motion. Predicting where it is going keeps a body travelling in a
/// straight line from being reinserted every single step.
const PREDICTION: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
struct Node {
    /// Fattened bounds. For an internal node, the union of its children.
    bounds: Aabb,
    parent: u32,
    left: u32,
    right: u32,
    /// Payload for a leaf — a body slot index. `NIL` on internal nodes.
    user: u32,
    /// Distance from this node to its deepest leaf. `0` on a leaf.
    height: i32,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.left == NIL
    }
}

/// A dynamic AABB tree.
#[derive(Debug, Clone, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    root: u32,
    free: Vec<u32>,
    /// Leaves whose bounds changed since the last pair query.
    moved: Vec<u32>,
    leaf_count: usize,
}

impl Bvh {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: NIL,
            free: Vec::new(),
            moved: Vec::new(),
            leaf_count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.leaf_count
    }

    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.moved.clear();
        self.root = NIL;
        self.leaf_count = 0;
    }

    /// Payload of a leaf.
    pub fn user_data(&self, proxy: ProxyId) -> Option<u32> {
        self.nodes
            .get(proxy.index())
            .filter(|n| n.is_leaf())
            .map(|n| n.user)
    }

    /// Tight-ish bounds of a leaf (the fattened box actually stored).
    pub fn bounds(&self, proxy: ProxyId) -> Option<Aabb> {
        self.nodes.get(proxy.index()).map(|n| n.bounds)
    }

    /// Insert `bounds` carrying `user`, returning its handle.
    pub fn insert(&mut self, bounds: &Aabb, user: u32) -> ProxyId {
        let node = self.allocate(fatten(bounds, Vector3::ZERO), user);
        self.insert_leaf(node);
        self.leaf_count += 1;
        self.moved.push(node);
        ProxyId(node)
    }

    pub fn remove(&mut self, proxy: ProxyId) {
        let index = proxy.0;
        if index as usize >= self.nodes.len() || !self.nodes[index as usize].is_leaf() {
            return;
        }
        self.remove_leaf(index);
        self.moved.retain(|&m| m != index);
        self.free.push(index);
        self.leaf_count -= 1;
    }

    /// Update a leaf's bounds.
    ///
    /// Returns `true` if the tree had to be re-edited. `displacement` is where
    /// the body is heading, used to bias the new fat box forward.
    pub fn update(&mut self, proxy: ProxyId, bounds: &Aabb, displacement: Vector3) -> bool {
        let index = proxy.0;
        if index as usize >= self.nodes.len() || !self.nodes[index as usize].is_leaf() {
            return false;
        }
        // Still inside the slack we left last time: nothing to do. This is the
        // case that makes the structure cheap, and most bodies hit it.
        if contains(&self.nodes[index as usize].bounds, bounds) {
            return false;
        }

        let user = self.nodes[index as usize].user;
        self.remove_leaf(index);
        self.nodes[index as usize].bounds = fatten(bounds, displacement);
        self.nodes[index as usize].user = user;
        self.insert_leaf(index);
        if !self.moved.contains(&index) {
            self.moved.push(index);
        }
        true
    }

    /// Mark a leaf as needing pair re-evaluation even though its bounds are
    /// unchanged — used when a body wakes up.
    pub fn touch(&mut self, proxy: ProxyId) {
        if !self.moved.contains(&proxy.0) {
            self.moved.push(proxy.0);
        }
    }

    /// Visit the payload of every leaf whose bounds overlap `query`.
    pub fn query_aabb(&self, query: &Aabb, mut visit: impl FnMut(u32, ProxyId)) {
        if self.root == NIL {
            return;
        }
        // An explicit stack: a degenerate tree can get deep, and this is on the
        // hot path for every query in the engine.
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if !node.bounds.intersects_box(query) {
                continue;
            }
            if node.is_leaf() {
                visit(node.user, ProxyId(index));
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// Visit every leaf a ray could hit, nearest-first is *not* guaranteed —
    /// callers that need the closest hit still compare distances.
    pub fn query_ray(&self, ray: &Ray, max_toi: f32, mut visit: impl FnMut(u32, ProxyId)) {
        if self.root == NIL {
            return;
        }
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            // A ray starting inside the box counts as a hit; `intersect_box`
            // reports the exit distance in that case, which must not be
            // compared against `max_toi`.
            let hit = if node.bounds.contains_point(ray.origin) {
                true
            } else {
                matches!(ray.intersect_box(&node.bounds), Some(t) if t <= max_toi)
            };
            if !hit {
                continue;
            }
            if node.is_leaf() {
                visit(node.user, ProxyId(index));
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// Collect overlapping pairs of payloads, as `(lo, hi)` with `lo < hi`.
    ///
    /// Only leaves that moved are queried, and each pair is reported once. The
    /// output is sorted, so it is identical run to run.
    pub fn find_pairs(&mut self, out: &mut Vec<(u32, u32)>, mut accept: impl FnMut(u32, u32) -> bool) {
        out.clear();
        if self.root == NIL {
            self.moved.clear();
            return;
        }

        let moved = std::mem::take(&mut self.moved);
        for &leaf in &moved {
            if leaf as usize >= self.nodes.len() || !self.nodes[leaf as usize].is_leaf() {
                continue;
            }
            let node = self.nodes[leaf as usize];
            let bounds = node.bounds;

            let mut stack = vec![self.root];
            while let Some(index) = stack.pop() {
                let other = self.nodes[index as usize];
                if !other.bounds.intersects_box(&bounds) {
                    continue;
                }
                if other.is_leaf() {
                    if index == leaf {
                        continue;
                    }
                    let (lo, hi) = if node.user < other.user {
                        (node.user, other.user)
                    } else {
                        (other.user, node.user)
                    };
                    if accept(lo, hi) {
                        out.push((lo, hi));
                    }
                } else {
                    stack.push(other.left);
                    stack.push(other.right);
                }
            }
        }
        self.moved = moved;
        self.moved.clear();

        // Two moved leaves overlapping each other produce the pair twice.
        out.sort_unstable();
        out.dedup();
    }

    /// Total surface area of every internal node — the standard measure of tree
    /// quality. Lower is better; it is what the insertion heuristic minimises.
    pub fn quality(&self) -> f32 {
        if self.root == NIL {
            return 0.0;
        }
        let mut total = 0.0;
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if node.is_leaf() {
                continue;
            }
            total += surface_area(&node.bounds);
            stack.push(node.left);
            stack.push(node.right);
        }
        total
    }

    /// Depth of the tree. Useful for asserting the balancing works.
    pub fn height(&self) -> i32 {
        if self.root == NIL {
            -1
        } else {
            self.nodes[self.root as usize].height
        }
    }

    // ---- internals --------------------------------------------------------

    fn allocate(&mut self, bounds: Aabb, user: u32) -> u32 {
        let node = Node {
            bounds,
            parent: NIL,
            left: NIL,
            right: NIL,
            user,
            height: 0,
        };
        if let Some(index) = self.free.pop() {
            self.nodes[index as usize] = node;
            index
        } else {
            self.nodes.push(node);
            self.nodes.len() as u32 - 1
        }
    }

    /// Choose where to put a new leaf, then rebalance the path back to the root.
    fn insert_leaf(&mut self, leaf: u32) {
        if self.root == NIL {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NIL;
            return;
        }

        // Descend toward whichever child costs less to grow — the surface-area
        // heuristic. Picking the nearer box instead produces trees that are
        // fine on paper and much worse to query.
        let leaf_bounds = self.nodes[leaf as usize].bounds;
        let mut index = self.root;
        while !self.nodes[index as usize].is_leaf() {
            let node = self.nodes[index as usize];
            let area = surface_area(&node.bounds);
            let combined = surface_area(&node.bounds.union(&leaf_bounds));

            // Cost of making this node the new sibling.
            let cost_here = 2.0 * combined;
            // Cost of pushing the leaf further down, whichever way.
            let inherited = 2.0 * (combined - area);

            let descend_cost = |child: u32| -> f32 {
                let child_node = self.nodes[child as usize];
                let merged = surface_area(&child_node.bounds.union(&leaf_bounds));
                if child_node.is_leaf() {
                    merged + inherited
                } else {
                    merged - surface_area(&child_node.bounds) + inherited
                }
            };
            let left_cost = descend_cost(node.left);
            let right_cost = descend_cost(node.right);

            if cost_here < left_cost && cost_here < right_cost {
                break;
            }
            index = if left_cost < right_cost {
                node.left
            } else {
                node.right
            };
        }

        // Splice a new internal node in above the sibling.
        let sibling = index;
        let old_parent = self.nodes[sibling as usize].parent;
        let new_parent = self.allocate(
            self.nodes[sibling as usize].bounds.union(&leaf_bounds),
            NIL,
        );
        self.nodes[new_parent as usize].parent = old_parent;
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;
        self.nodes[new_parent as usize].left = sibling;
        self.nodes[new_parent as usize].right = leaf;
        self.nodes[sibling as usize].parent = new_parent;
        self.nodes[leaf as usize].parent = new_parent;

        if old_parent == NIL {
            self.root = new_parent;
        } else if self.nodes[old_parent as usize].left == sibling {
            self.nodes[old_parent as usize].left = new_parent;
        } else {
            self.nodes[old_parent as usize].right = new_parent;
        }

        self.refit_from(new_parent);
    }

    fn remove_leaf(&mut self, leaf: u32) {
        if self.root == leaf {
            self.root = NIL;
            self.nodes[leaf as usize].parent = NIL;
            return;
        }

        let parent = self.nodes[leaf as usize].parent;
        let grandparent = self.nodes[parent as usize].parent;
        let sibling = if self.nodes[parent as usize].left == leaf {
            self.nodes[parent as usize].right
        } else {
            self.nodes[parent as usize].left
        };

        // The parent existed only to hold two children; with one gone, the
        // sibling takes its place.
        if grandparent == NIL {
            self.root = sibling;
            self.nodes[sibling as usize].parent = NIL;
        } else {
            if self.nodes[grandparent as usize].left == parent {
                self.nodes[grandparent as usize].left = sibling;
            } else {
                self.nodes[grandparent as usize].right = sibling;
            }
            self.nodes[sibling as usize].parent = grandparent;
            self.refit_from(grandparent);
        }
        self.free.push(parent);
        self.nodes[leaf as usize].parent = NIL;
    }

    /// Walk to the root, re-uniting bounds and rebalancing as we go.
    fn refit_from(&mut self, start: u32) {
        let mut index = start;
        while index != NIL {
            index = self.balance(index);
            let node = self.nodes[index as usize];
            let (left, right) = (node.left, node.right);
            if left != NIL && right != NIL {
                self.nodes[index as usize].bounds = self.nodes[left as usize]
                    .bounds
                    .union(&self.nodes[right as usize].bounds);
                self.nodes[index as usize].height = 1 + self.nodes[left as usize]
                    .height
                    .max(self.nodes[right as usize].height);
            }
            index = self.nodes[index as usize].parent;
        }
    }

    /// AVL-style rotation, so a run of insertions in sorted order cannot
    /// degenerate the tree into a linked list.
    fn balance(&mut self, a: u32) -> u32 {
        let node = self.nodes[a as usize];
        if node.is_leaf() || node.height < 2 {
            return a;
        }
        let (b, c) = (node.left, node.right);
        let imbalance = self.nodes[c as usize].height - self.nodes[b as usize].height;

        if imbalance > 1 {
            return self.rotate(a, c, b);
        }
        if imbalance < -1 {
            return self.rotate(a, b, c);
        }
        a
    }

    /// Pivot `heavy` up above `a`, pushing `light` down.
    fn rotate(&mut self, a: u32, heavy: u32, light: u32) -> u32 {
        let (f, g) = (self.nodes[heavy as usize].left, self.nodes[heavy as usize].right);

        self.nodes[heavy as usize].left = a;
        self.nodes[heavy as usize].parent = self.nodes[a as usize].parent;
        self.nodes[a as usize].parent = heavy;

        // Re-point the grandparent, or the root.
        let grandparent = self.nodes[heavy as usize].parent;
        if grandparent == NIL {
            self.root = heavy;
        } else if self.nodes[grandparent as usize].left == a {
            self.nodes[grandparent as usize].left = heavy;
        } else {
            self.nodes[grandparent as usize].right = heavy;
        }

        // Keep whichever of the heavy node's children is taller on top.
        let (keep, drop) = if self.nodes[f as usize].height > self.nodes[g as usize].height {
            (f, g)
        } else {
            (g, f)
        };
        self.nodes[heavy as usize].right = keep;

        // `drop` becomes a child of `a`, in the slot `heavy` vacated.
        if self.nodes[a as usize].left == heavy {
            self.nodes[a as usize].left = drop;
        } else {
            self.nodes[a as usize].right = drop;
        }
        self.nodes[drop as usize].parent = a;

        self.nodes[a as usize].bounds = self.nodes[light as usize]
            .bounds
            .union(&self.nodes[drop as usize].bounds);
        self.nodes[heavy as usize].bounds = self.nodes[a as usize]
            .bounds
            .union(&self.nodes[keep as usize].bounds);
        self.nodes[a as usize].height = 1 + self.nodes[light as usize]
            .height
            .max(self.nodes[drop as usize].height);
        self.nodes[heavy as usize].height = 1 + self.nodes[a as usize]
            .height
            .max(self.nodes[keep as usize].height);

        heavy
    }
}

fn fatten(bounds: &Aabb, displacement: Vector3) -> Aabb {
    let margin = Vector3::new(MARGIN, MARGIN, MARGIN);
    let mut min = bounds.min - margin;
    let mut max = bounds.max + margin;

    // Extend along the direction of travel only, so a body moving right does
    // not pay for slack on its left.
    let predicted = displacement * PREDICTION;
    for (lo, hi, d) in [
        (&mut min.x, &mut max.x, predicted.x),
        (&mut min.y, &mut max.y, predicted.y),
        (&mut min.z, &mut max.z, predicted.z),
    ] {
        if d < 0.0 {
            *lo += d;
        } else {
            *hi += d;
        }
    }
    Aabb::new(min, max)
}

fn contains(outer: &Aabb, inner: &Aabb) -> bool {
    outer.min.x <= inner.min.x
        && outer.min.y <= inner.min.y
        && outer.min.z <= inner.min.z
        && outer.max.x >= inner.max.x
        && outer.max.y >= inner.max.y
        && outer.max.z >= inner.max.z
}

fn surface_area(b: &Aabb) -> f32 {
    let d = b.size();
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(cx: f32, cy: f32, cz: f32, h: f32) -> Aabb {
        Aabb::new(
            Vector3::new(cx - h, cy - h, cz - h),
            Vector3::new(cx + h, cy + h, cz + h),
        )
    }

    /// Every parent's bounds must contain both children, heights must be
    /// consistent, and parent links must agree. Checked after every mutation in
    /// the tests below, because a tree that is subtly wrong still answers most
    /// queries correctly.
    fn assert_consistent(tree: &Bvh) {
        if tree.root == NIL {
            return;
        }
        let mut stack = vec![tree.root];
        let mut leaves = 0;
        while let Some(index) = stack.pop() {
            let node = tree.nodes[index as usize];
            if node.is_leaf() {
                leaves += 1;
                assert_eq!(node.height, 0, "leaf {index} has height {}", node.height);
                continue;
            }
            let (l, r) = (node.left, node.right);
            assert_eq!(tree.nodes[l as usize].parent, index, "left child of {index}");
            assert_eq!(tree.nodes[r as usize].parent, index, "right child of {index}");
            assert!(
                contains(&node.bounds, &tree.nodes[l as usize].bounds),
                "node {index} does not contain its left child"
            );
            assert!(
                contains(&node.bounds, &tree.nodes[r as usize].bounds),
                "node {index} does not contain its right child"
            );
            let want = 1 + tree.nodes[l as usize].height.max(tree.nodes[r as usize].height);
            assert_eq!(node.height, want, "height of {index}");
            stack.push(l);
            stack.push(r);
        }
        assert_eq!(leaves, tree.leaf_count, "leaf count drifted");
        assert_eq!(tree.nodes[tree.root as usize].parent, NIL, "root has a parent");
    }

    #[test]
    fn insert_and_remove_keep_the_tree_consistent() {
        let mut tree = Bvh::new();
        let mut proxies = Vec::new();
        for i in 0..64 {
            let p = tree.insert(&aabb(i as f32 * 2.0, 0.0, 0.0, 0.4), i);
            proxies.push(p);
            assert_consistent(&tree);
        }
        assert_eq!(tree.len(), 64);

        // Remove every other one.
        for (i, p) in proxies.iter().enumerate() {
            if i % 2 == 0 {
                tree.remove(*p);
                assert_consistent(&tree);
            }
        }
        assert_eq!(tree.len(), 32);

        for (i, p) in proxies.iter().enumerate() {
            if i % 2 == 1 {
                tree.remove(*p);
            }
        }
        assert_consistent(&tree);
        assert!(tree.is_empty());
    }

    #[test]
    fn sorted_insertion_does_not_degenerate_the_tree() {
        // The pathological case for an unbalanced tree: everything inserted in
        // increasing order along one axis.
        let mut tree = Bvh::new();
        for i in 0..1024 {
            tree.insert(&aabb(i as f32, 0.0, 0.0, 0.4), i);
        }
        assert_consistent(&tree);
        // A linked list would be 1024 deep; a balanced tree is about 10.
        assert!(
            tree.height() < 30,
            "tree degenerated to height {}",
            tree.height()
        );
    }

    #[test]
    fn aabb_queries_find_exactly_the_overlapping_leaves() {
        let mut tree = Bvh::new();
        for i in 0..50 {
            tree.insert(&aabb(i as f32 * 2.0, 0.0, 0.0, 0.5), i);
        }

        let mut found = Vec::new();
        tree.query_aabb(&aabb(10.0, 0.0, 0.0, 1.0), |user, _| found.push(user));
        found.sort_unstable();

        // Bodies sit at x = 0, 2, 4 ... with half-extent 0.5, fattened by the
        // margin. The query spans 9..11, so it must find the one at 10 and may
        // find its neighbours through the margin — but nothing far away.
        assert!(found.contains(&5), "missed the body at x = 10");
        for user in found {
            let x = user as f32 * 2.0;
            assert!((x - 10.0).abs() <= 2.5, "found a body at x = {x}");
        }
    }

    #[test]
    fn a_query_against_an_empty_tree_is_harmless() {
        let tree = Bvh::new();
        let mut hits = 0;
        tree.query_aabb(&aabb(0.0, 0.0, 0.0, 1.0), |_, _| hits += 1);
        tree.query_ray(
            &Ray::new(Vector3::ZERO, Vector3::UP),
            100.0,
            |_, _| hits += 1,
        );
        assert_eq!(hits, 0);
        assert_eq!(tree.height(), -1);
    }

    #[test]
    fn ray_queries_hit_what_is_on_the_line_and_skip_what_is_not() {
        let mut tree = Bvh::new();
        // A column of boxes going up.
        for i in 0..20 {
            tree.insert(&aabb(0.0, i as f32 * 2.0, 0.0, 0.5), i);
        }
        // And a row going sideways, which the ray must not touch.
        for i in 0..20 {
            tree.insert(&aabb(20.0 + i as f32 * 2.0, 0.0, 0.0, 0.5), 100 + i);
        }

        let mut hits = Vec::new();
        tree.query_ray(
            &Ray::new(Vector3::new(0.0, -5.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            100.0,
            |user, _| hits.push(user),
        );
        assert!(hits.len() >= 20, "the ray should cross the whole column");
        assert!(
            hits.iter().all(|u| *u < 100),
            "the ray hit the sideways row"
        );
    }

    #[test]
    fn a_ray_starting_inside_a_box_still_reports_it() {
        let mut tree = Bvh::new();
        tree.insert(&aabb(0.0, 0.0, 0.0, 5.0), 7);
        let mut hits = Vec::new();
        tree.query_ray(
            &Ray::new(Vector3::ZERO, Vector3::UP),
            100.0,
            |user, _| hits.push(user),
        );
        assert_eq!(hits, vec![7]);
    }

    #[test]
    fn max_toi_prunes_distant_boxes() {
        let mut tree = Bvh::new();
        tree.insert(&aabb(0.0, 5.0, 0.0, 0.5), 1);
        tree.insert(&aabb(0.0, 500.0, 0.0, 0.5), 2);

        let mut hits = Vec::new();
        tree.query_ray(
            &Ray::new(Vector3::ZERO, Vector3::UP),
            10.0,
            |user, _| hits.push(user),
        );
        assert_eq!(hits, vec![1], "the far box should have been pruned");
    }

    #[test]
    fn a_small_move_inside_the_margin_does_not_re_edit_the_tree() {
        let mut tree = Bvh::new();
        let p = tree.insert(&aabb(0.0, 0.0, 0.0, 1.0), 0);
        // Well within the fattening margin.
        assert!(!tree.update(p, &aabb(0.01, 0.0, 0.0, 1.0), Vector3::ZERO));
        // Beyond it.
        assert!(tree.update(p, &aabb(5.0, 0.0, 0.0, 1.0), Vector3::ZERO));
        assert_consistent(&tree);
    }

    #[test]
    fn pairs_match_a_brute_force_sweep() {
        // The property that matters: whatever the tree reports must equal what
        // an all-pairs test reports.
        let boxes: Vec<Aabb> = (0..120)
            .map(|i| {
                let f = i as f32;
                aabb(
                    (f * 0.7).sin() * 12.0,
                    (f * 1.3).cos() * 12.0,
                    (f * 2.1).sin() * 12.0,
                    0.9,
                )
            })
            .collect();

        let mut tree = Bvh::new();
        let proxies: Vec<ProxyId> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| tree.insert(b, i as u32))
            .collect();
        assert_consistent(&tree);

        let mut pairs = Vec::new();
        tree.find_pairs(&mut pairs, |_, _| true);

        // Brute force over the *fattened* bounds, since that is what the tree
        // stores and therefore what it can legitimately report.
        let mut expected = Vec::new();
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let a = tree.bounds(proxies[i]).unwrap();
                let b = tree.bounds(proxies[j]).unwrap();
                if a.intersects_box(&b) {
                    expected.push((i as u32, j as u32));
                }
            }
        }
        expected.sort_unstable();
        assert_eq!(pairs, expected);
    }

    #[test]
    fn only_moved_leaves_are_re_paired() {
        let mut tree = Bvh::new();
        let a = tree.insert(&aabb(0.0, 0.0, 0.0, 1.0), 0);
        tree.insert(&aabb(1.5, 0.0, 0.0, 1.0), 1);

        let mut pairs = Vec::new();
        tree.find_pairs(&mut pairs, |_, _| true);
        assert_eq!(pairs, vec![(0, 1)], "the initial insert should pair them");

        // Nothing moved: no work, no pairs.
        tree.find_pairs(&mut pairs, |_, _| true);
        assert!(pairs.is_empty(), "an idle world should report no new pairs");

        // Move one far enough to leave its margin, and the pair returns.
        tree.update(a, &aabb(1.4, 0.0, 0.0, 1.0), Vector3::ZERO);
        tree.find_pairs(&mut pairs, |_, _| true);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn the_accept_predicate_filters_pairs() {
        let mut tree = Bvh::new();
        for i in 0..8 {
            tree.insert(&aabb(i as f32 * 0.5, 0.0, 0.0, 1.0), i);
        }
        let mut pairs = Vec::new();
        // Reject everything.
        tree.find_pairs(&mut pairs, |_, _| false);
        assert!(pairs.is_empty());
    }

    #[test]
    fn the_tree_stays_reasonable_under_heavy_churn() {
        let mut tree = Bvh::new();
        let mut live: Vec<(ProxyId, u32)> = Vec::new();
        let mut next = 0u32;

        for step in 0..500 {
            // Add two, remove one, move a few — the shape of a real scene.
            for _ in 0..2 {
                let f = next as f32;
                let p = tree.insert(&aabb((f * 0.37).sin() * 20.0, (f * 0.11).cos() * 20.0, 0.0, 0.5), next);
                live.push((p, next));
                next += 1;
            }
            if live.len() > 4 && step % 2 == 0 {
                let (p, _) = live.remove(step % live.len());
                tree.remove(p);
            }
            for (i, (p, _)) in live.iter().enumerate().take(8) {
                let f = (step + i) as f32;
                tree.update(*p, &aabb((f * 0.23).sin() * 20.0, (f * 0.19).cos() * 20.0, 0.0, 0.5), Vector3::ZERO);
            }
        }
        assert_consistent(&tree);
        // Balanced enough that queries stay logarithmic.
        let ideal = (live.len() as f32).log2().ceil() as i32;
        assert!(
            tree.height() <= ideal * 3 + 4,
            "height {} for {} leaves (ideal about {ideal})",
            tree.height(),
            live.len()
        );
    }
}
