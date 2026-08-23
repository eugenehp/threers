//! The acceleration structure the path tracer traverses: one bounding-volume
//! hierarchy over every triangle in the scene, in world space.
//!
//! This is deliberately *not* [`crate::mesh_bvh::MeshBvh`]. That one is built
//! per `BufferGeometry` in local space, which is what a picking raycast wants —
//! it asks about one object at a time. A path tracer asks "what does this ray
//! hit anywhere in the scene", millions of times, and a two-level walk over N
//! per-object trees with a matrix inverse at each entry costs more than one
//! flat tree over pre-transformed triangles. The trade is memory (vertices are
//! denormalised, 36 bytes a triangle) and a rebuild whenever anything moves.
//!
//! The split is chosen by a binned surface-area heuristic over all three axes,
//! which is the standard quality/build-time compromise: it costs one pass over
//! the primitives per axis instead of the sort a full sweep needs, and lands
//! within a few percent of the sweep's traversal cost.

use crate::math::Vector3;

/// Number of buckets the SAH sweep is approximated with, per axis.
const BINS: usize = 12;
/// Triangles at or below this count become a leaf without further testing.
const MAX_LEAF_TRIS: usize = 4;
/// Relative cost of a node traversal against one ray-triangle test.
const TRAVERSAL_COST: f32 = 1.0;
/// Depth beyond which a node is forced to a leaf, so a pathological input
/// cannot overflow the build stack.
const MAX_DEPTH: u32 = 64;

/// A closest-hit record. `u`/`v` are the barycentric weights of the second and
/// third vertices, so the first weighs `1 - u - v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtHit {
    pub t: f32,
    pub triangle: u32,
    pub u: f32,
    pub v: f32,
}

/// 32 bytes, one cache line to two: bounds interleaved with the child links so
/// a slab test and the descent read the same fetch.
#[derive(Debug, Clone, Copy)]
struct Node {
    min: [f32; 3],
    /// Leaf: first index into `order`. Internal: index of the left child, whose
    /// sibling is always the next node.
    left_or_first: u32,
    max: [f32; 3],
    /// 0 marks an internal node.
    count: u32,
}

impl Node {
    const EMPTY: Node = Node {
        min: [f32::INFINITY; 3],
        left_or_first: 0,
        max: [f32::NEG_INFINITY; 3],
        count: 0,
    };
}

/// A node exposed for packing into a device buffer. Leaves have `count > 0`
/// and `left_or_first` indexing [`RtBvh::order`]; internal nodes have
/// `count == 0` and `left_or_first` indexing the node array.
#[derive(Debug, Clone, Copy)]
pub struct BvhNodeView {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub left_or_first: u32,
    pub count: u32,
}

/// A world-space BVH over a triangle soup.
#[derive(Debug, Clone, Default)]
pub struct RtBvh {
    nodes: Vec<Node>,
    /// Triangle indices, permuted so every leaf owns a contiguous run.
    order: Vec<u32>,
}

/// Axis-aligned bounds in the plain-array form the builder works in.
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    const EMPTY: Aabb = Aabb {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    fn grow_point(&mut self, p: Vector3) {
        let p = [p.x, p.y, p.z];
        for ((lo, hi), v) in self.min.iter_mut().zip(self.max.iter_mut()).zip(p) {
            *lo = lo.min(v);
            *hi = hi.max(v);
        }
    }

    fn grow(&mut self, other: &Aabb) {
        for ((lo, hi), (a, b)) in self
            .min
            .iter_mut()
            .zip(self.max.iter_mut())
            .zip(other.min.iter().zip(other.max.iter()))
        {
            *lo = lo.min(*a);
            *hi = hi.max(*b);
        }
    }

    /// Surface area, or 0 for an empty box. The SAH is a ratio of these, so an
    /// empty child must contribute nothing rather than a negative number.
    fn area(&self) -> f32 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx < 0.0 || dy < 0.0 || dz < 0.0 {
            return 0.0;
        }
        2.0 * (dx * dy + dy * dz + dz * dx)
    }
}

impl RtBvh {
    /// Build over `tris`, each already in world space.
    pub fn build(tris: &[[Vector3; 3]]) -> Self {
        let n = tris.len();
        let mut bvh = RtBvh {
            // 2n-1 is the exact node count of a binary tree over n leaves of one
            // primitive each; every leaf here holds at least one, so this is an
            // upper bound and the vector never reallocates mid-build.
            nodes: Vec::with_capacity((2 * n).max(1)),
            order: (0..n as u32).collect(),
        };
        bvh.nodes.push(Node::EMPTY);
        if n == 0 {
            return bvh;
        }

        // Per-triangle bounds and centroids, computed once — the binning pass
        // reads them O(depth) times each.
        let mut bounds = Vec::with_capacity(n);
        let mut centroids = Vec::with_capacity(n);
        for t in tris {
            let mut b = Aabb::EMPTY;
            b.grow_point(t[0]);
            b.grow_point(t[1]);
            b.grow_point(t[2]);
            centroids.push([
                0.5 * (b.min[0] + b.max[0]),
                0.5 * (b.min[1] + b.max[1]),
                0.5 * (b.min[2] + b.max[2]),
            ]);
            bounds.push(b);
        }

        bvh.subdivide(0, 0, n, &bounds, &centroids, 0);
        bvh
    }

    /// Number of nodes in the tree (1 for an empty scene).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangles indexed.
    pub fn triangle_count(&self) -> usize {
        self.order.len()
    }

    /// One node, in the flat form a GPU buffer wants. The tree is already an
    /// array of 32-byte records, so packing it for a device is a copy rather
    /// than a rebuild.
    pub fn node(&self, i: usize) -> BvhNodeView {
        let n = self.nodes[i];
        BvhNodeView {
            min: n.min,
            max: n.max,
            left_or_first: n.left_or_first,
            count: n.count,
        }
    }

    /// The triangle permutation leaves index into.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Root bounds. Degenerate (min > max) when the tree is empty.
    pub fn bounds(&self) -> (Vector3, Vector3) {
        let root = self.nodes[0];
        (
            Vector3::new(root.min[0], root.min[1], root.min[2]),
            Vector3::new(root.max[0], root.max[1], root.max[2]),
        )
    }

    fn subdivide(
        &mut self,
        node_idx: usize,
        start: usize,
        count: usize,
        bounds: &[Aabb],
        centroids: &[[f32; 3]],
        depth: u32,
    ) {
        let mut node_bounds = Aabb::EMPTY;
        let mut centroid_bounds = Aabb::EMPTY;
        for &tri in &self.order[start..start + count] {
            node_bounds.grow(&bounds[tri as usize]);
            centroid_bounds.grow_point(Vector3::new(
                centroids[tri as usize][0],
                centroids[tri as usize][1],
                centroids[tri as usize][2],
            ));
        }
        self.nodes[node_idx].min = node_bounds.min;
        self.nodes[node_idx].max = node_bounds.max;

        let make_leaf = |bvh: &mut Self| {
            bvh.nodes[node_idx].left_or_first = start as u32;
            bvh.nodes[node_idx].count = count as u32;
        };

        if count <= MAX_LEAF_TRIS || depth >= MAX_DEPTH {
            make_leaf(self);
            return;
        }

        let Some(split) = self.find_split(
            start,
            count,
            &node_bounds,
            &centroid_bounds,
            bounds,
            centroids,
        ) else {
            make_leaf(self);
            return;
        };

        // Partition `order[start..start+count]` in place around the plane.
        let mid = {
            let axis = split.axis;
            let scale = split.scale;
            let lo = centroid_bounds.min[axis];
            let mut i = start;
            let mut j = start + count;
            while i < j {
                let tri = self.order[i] as usize;
                let bin = (((centroids[tri][axis] - lo) * scale) as usize).min(BINS - 1);
                if bin < split.bin {
                    i += 1;
                } else {
                    j -= 1;
                    self.order.swap(i, j);
                }
            }
            i
        };

        // Binning can still put everything on one side when many centroids
        // coincide; splitting down the middle keeps the tree finite.
        let left_count = mid - start;
        if left_count == 0 || left_count == count {
            make_leaf(self);
            return;
        }

        let left = self.nodes.len() as u32;
        self.nodes.push(Node::EMPTY);
        self.nodes.push(Node::EMPTY);
        self.nodes[node_idx].left_or_first = left;
        self.nodes[node_idx].count = 0;

        self.subdivide(
            left as usize,
            start,
            left_count,
            bounds,
            centroids,
            depth + 1,
        );
        self.subdivide(
            left as usize + 1,
            mid,
            count - left_count,
            bounds,
            centroids,
            depth + 1,
        );
    }

    /// The best binned-SAH split, or `None` when leaving the node whole is
    /// cheaper than any of them.
    fn find_split(
        &self,
        start: usize,
        count: usize,
        node_bounds: &Aabb,
        centroid_bounds: &Aabb,
        bounds: &[Aabb],
        centroids: &[[f32; 3]],
    ) -> Option<Split> {
        let parent_area = node_bounds.area();
        // Cost of leaving it as a leaf, in ray-triangle tests.
        let leaf_cost = count as f32;
        let mut best: Option<Split> = None;
        let mut best_cost = leaf_cost;

        // `axis` indexes the three components of each centroid, not the
        // centroid array itself.
        #[allow(clippy::needless_range_loop)]
        for axis in 0..3 {
            let extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
            if extent <= 1e-12 {
                continue;
            }
            let scale = BINS as f32 / extent;
            let lo = centroid_bounds.min[axis];

            let mut bin_bounds = [Aabb::EMPTY; BINS];
            let mut bin_counts = [0u32; BINS];
            for &tri in &self.order[start..start + count] {
                let tri = tri as usize;
                let bin = (((centroids[tri][axis] - lo) * scale) as usize).min(BINS - 1);
                bin_counts[bin] += 1;
                bin_bounds[bin].grow(&bounds[tri]);
            }

            // Prefix sweep from the left, suffix from the right, so each of the
            // BINS-1 candidate planes is evaluated in O(1).
            let mut left_area = [0.0f32; BINS - 1];
            let mut left_count = [0u32; BINS - 1];
            let mut acc = Aabb::EMPTY;
            let mut acc_n = 0u32;
            for i in 0..BINS - 1 {
                acc.grow(&bin_bounds[i]);
                acc_n += bin_counts[i];
                left_area[i] = acc.area();
                left_count[i] = acc_n;
            }
            let mut acc = Aabb::EMPTY;
            let mut acc_n = 0u32;
            for i in (0..BINS - 1).rev() {
                acc.grow(&bin_bounds[i + 1]);
                acc_n += bin_counts[i + 1];
                let right_area = acc.area();
                if left_count[i] == 0 || acc_n == 0 {
                    continue;
                }
                let cost = TRAVERSAL_COST
                    + (left_area[i] * left_count[i] as f32 + right_area * acc_n as f32)
                        / parent_area.max(1e-12);
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(Split {
                        axis,
                        bin: i + 1,
                        scale,
                    });
                }
            }
        }
        best
    }

    /// Closest hit in `[t_min, t_max)`, or `None`.
    pub fn intersect(
        &self,
        tris: &[[Vector3; 3]],
        origin: Vector3,
        dir: Vector3,
        t_min: f32,
        t_max: f32,
    ) -> Option<RtHit> {
        if self.order.is_empty() {
            return None;
        }
        let inv = inv_dir(dir);
        let mut best: Option<RtHit> = None;
        let mut t_far = t_max;

        // Explicit stack: recursion here would be a call per node on the
        // hottest path in the renderer.
        let mut stack = [0u32; MAX_DEPTH as usize * 2 + 8];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = self.nodes[node as usize];
            if n.count > 0 {
                let first = n.left_or_first as usize;
                for &tri in &self.order[first..first + n.count as usize] {
                    let t = tris[tri as usize];
                    if let Some((t_hit, u, v)) = intersect_triangle(t, origin, dir, t_min, t_far) {
                        t_far = t_hit;
                        best = Some(RtHit {
                            t: t_hit,
                            triangle: tri,
                            u,
                            v,
                        });
                    }
                }
            } else {
                let left = n.left_or_first;
                let right = left + 1;
                let d_left = slab(&self.nodes[left as usize], origin, inv, t_min, t_far);
                let d_right = slab(&self.nodes[right as usize], origin, inv, t_min, t_far);
                // Descend into the nearer child first: whatever it hits shrinks
                // `t_far`, and the far child is then often rejected outright.
                let (near, far, d_far) = if d_left <= d_right {
                    (left, right, d_right)
                } else {
                    (right, left, d_left)
                };
                if d_left.min(d_right) < f32::INFINITY {
                    if d_far < f32::INFINITY && sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node = near;
                    continue;
                }
            }
            if sp == 0 {
                break;
            }
            sp -= 1;
            node = stack[sp];
        }
        best
    }

    /// Whether *anything* blocks the segment. Returns on the first hit rather
    /// than the nearest one, which is roughly twice as fast for shadow rays.
    pub fn occluded(
        &self,
        tris: &[[Vector3; 3]],
        origin: Vector3,
        dir: Vector3,
        t_min: f32,
        t_max: f32,
    ) -> bool {
        if self.order.is_empty() {
            return false;
        }
        let inv = inv_dir(dir);
        let mut stack = [0u32; MAX_DEPTH as usize * 2 + 8];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = self.nodes[node as usize];
            if n.count > 0 {
                let first = n.left_or_first as usize;
                for &tri in &self.order[first..first + n.count as usize] {
                    if intersect_triangle(tris[tri as usize], origin, dir, t_min, t_max).is_some() {
                        return true;
                    }
                }
            } else {
                let left = n.left_or_first;
                let right = left + 1;
                let d_left = slab(&self.nodes[left as usize], origin, inv, t_min, t_max);
                let d_right = slab(&self.nodes[right as usize], origin, inv, t_min, t_max);
                let (near, far, d_far) = if d_left <= d_right {
                    (left, right, d_right)
                } else {
                    (right, left, d_left)
                };
                if d_left.min(d_right) < f32::INFINITY {
                    if d_far < f32::INFINITY && sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node = near;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node = stack[sp];
        }
    }

    /// Every hit along the segment, nearest first. Used by the transparent
    /// shadow walk, which must attenuate through each layer rather than stop.
    pub fn intersect_all(
        &self,
        tris: &[[Vector3; 3]],
        origin: Vector3,
        dir: Vector3,
        t_min: f32,
        t_max: f32,
        out: &mut Vec<RtHit>,
    ) {
        out.clear();
        if self.order.is_empty() {
            return;
        }
        let inv = inv_dir(dir);
        let mut stack = [0u32; MAX_DEPTH as usize * 2 + 8];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = self.nodes[node as usize];
            if n.count > 0 {
                let first = n.left_or_first as usize;
                for &tri in &self.order[first..first + n.count as usize] {
                    if let Some((t, u, v)) =
                        intersect_triangle(tris[tri as usize], origin, dir, t_min, t_max)
                    {
                        out.push(RtHit {
                            t,
                            triangle: tri,
                            u,
                            v,
                        });
                    }
                }
            } else {
                let left = n.left_or_first;
                let right = left + 1;
                let d_left = slab(&self.nodes[left as usize], origin, inv, t_min, t_max);
                let d_right = slab(&self.nodes[right as usize], origin, inv, t_min, t_max);
                if d_left < f32::INFINITY && sp < stack.len() {
                    stack[sp] = left;
                    sp += 1;
                }
                if d_right < f32::INFINITY && sp < stack.len() {
                    stack[sp] = right;
                    sp += 1;
                }
            }
            if sp == 0 {
                break;
            }
            sp -= 1;
            node = stack[sp];
        }
        out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[derive(Debug, Clone, Copy)]
struct Split {
    axis: usize,
    /// First bin that goes to the right child.
    bin: usize,
    scale: f32,
}

/// Reciprocal direction for the slab test. A zero component yields ±∞, which is
/// exactly what the slab test wants: the ray is parallel to that pair of planes
/// and the interval it contributes is the whole line, so long as the origin
/// lies between them — and IEEE comparison against ±∞ gets that right without
/// a branch. `0 * ∞` is the one case that does not (it is NaN), so components
/// are floored at the smallest normal rather than left at exactly zero.
fn inv_dir(dir: Vector3) -> [f32; 3] {
    let safe = |v: f32| {
        if v.abs() < 1e-30 {
            if v < 0.0 {
                -1e-30
            } else {
                1e-30
            }
        } else {
            v
        }
    };
    [1.0 / safe(dir.x), 1.0 / safe(dir.y), 1.0 / safe(dir.z)]
}

/// Ray/AABB entry distance, or `INFINITY` when the ray misses the box within
/// `[t_min, t_max]`.
#[inline]
fn slab(node: &Node, origin: Vector3, inv: [f32; 3], t_min: f32, t_max: f32) -> f32 {
    let o = [origin.x, origin.y, origin.z];
    let mut tmin = t_min;
    let mut tmax = t_max;
    for a in 0..3 {
        let t0 = (node.min[a] - o[a]) * inv[a];
        let t1 = (node.max[a] - o[a]) * inv[a];
        let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        if lo > tmin {
            tmin = lo;
        }
        if hi < tmax {
            tmax = hi;
        }
        if tmin > tmax {
            return f32::INFINITY;
        }
    }
    tmin
}

/// Möller–Trumbore, two-sided. Returns `(t, u, v)` where `u` and `v` are the
/// barycentric weights of `tri[1]` and `tri[2]`.
#[inline]
pub fn intersect_triangle(
    tri: [Vector3; 3],
    origin: Vector3,
    dir: Vector3,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32)> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    // Reject only exact edge-on hits: culling by sign here would drop the back
    // faces the tracer needs for transmission and for interior shading.
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tv = origin - tri[0];
    let u = tv.dot(pv) * inv_det;
    if !(-1e-7..=1.0 + 1e-7).contains(&u) {
        return None;
    }
    let qv = tv.cross(e1);
    let v = dir.dot(qv) * inv_det;
    if v < -1e-7 || u + v > 1.0 + 1e-7 {
        return None;
    }
    let t = e2.dot(qv) * inv_det;
    if t < t_min || t >= t_max {
        return None;
    }
    Some((t, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytrace::sampler::{uniform_sphere, Rng};

    fn quad(z: f32) -> Vec<[Vector3; 3]> {
        vec![
            [
                Vector3::new(-1.0, -1.0, z),
                Vector3::new(1.0, -1.0, z),
                Vector3::new(1.0, 1.0, z),
            ],
            [
                Vector3::new(-1.0, -1.0, z),
                Vector3::new(1.0, 1.0, z),
                Vector3::new(-1.0, 1.0, z),
            ],
        ]
    }

    /// A pseudo-random cloud of small triangles, for differential testing.
    fn soup(n: usize) -> Vec<[Vector3; 3]> {
        let mut rng = Rng::new(1, 2);
        (0..n)
            .map(|_| {
                let c = Vector3::new(
                    rng.next_f32() * 10.0 - 5.0,
                    rng.next_f32() * 10.0 - 5.0,
                    rng.next_f32() * 10.0 - 5.0,
                );
                let s = 0.05 + rng.next_f32() * 0.4;
                [
                    c + uniform_sphere(rng.next_f32(), rng.next_f32()) * s,
                    c + uniform_sphere(rng.next_f32(), rng.next_f32()) * s,
                    c + uniform_sphere(rng.next_f32(), rng.next_f32()) * s,
                ]
            })
            .collect()
    }

    fn brute_force(
        tris: &[[Vector3; 3]],
        o: Vector3,
        d: Vector3,
        t_min: f32,
        t_max: f32,
    ) -> Option<RtHit> {
        let mut best: Option<RtHit> = None;
        let mut far = t_max;
        for (i, t) in tris.iter().enumerate() {
            if let Some((t_hit, u, v)) = intersect_triangle(*t, o, d, t_min, far) {
                far = t_hit;
                best = Some(RtHit {
                    t: t_hit,
                    triangle: i as u32,
                    u,
                    v,
                });
            }
        }
        best
    }

    #[test]
    fn empty_scene_never_hits() {
        let bvh = RtBvh::build(&[]);
        assert!(bvh
            .intersect(&[], Vector3::ZERO, Vector3::new(0.0, 0.0, 1.0), 0.0, 1e30)
            .is_none());
        assert!(!bvh.occluded(&[], Vector3::ZERO, Vector3::new(0.0, 0.0, 1.0), 0.0, 1e30));
    }

    #[test]
    fn hits_a_quad_head_on() {
        let tris = quad(0.0);
        let bvh = RtBvh::build(&tris);
        let hit = bvh
            .intersect(
                &tris,
                Vector3::new(0.1, 0.1, -3.0),
                Vector3::new(0.0, 0.0, 1.0),
                1e-4,
                1e30,
            )
            .expect("expected a hit");
        assert!((hit.t - 3.0).abs() < 1e-4, "t = {}", hit.t);
    }

    #[test]
    fn hits_from_behind_too() {
        let tris = quad(0.0);
        let bvh = RtBvh::build(&tris);
        assert!(bvh
            .intersect(
                &tris,
                Vector3::new(0.0, 0.0, 3.0),
                Vector3::new(0.0, 0.0, -1.0),
                1e-4,
                1e30
            )
            .is_some());
    }

    #[test]
    fn barycentrics_reconstruct_the_hit_point() {
        let tris = quad(0.0);
        let bvh = RtBvh::build(&tris);
        let o = Vector3::new(0.37, -0.21, -2.0);
        let d = Vector3::new(0.0, 0.0, 1.0);
        let hit = bvh.intersect(&tris, o, d, 1e-4, 1e30).unwrap();
        let t = tris[hit.triangle as usize];
        let p = t[0] * (1.0 - hit.u - hit.v) + t[1] * hit.u + t[2] * hit.v;
        assert!((p - (o + d * hit.t)).length() < 1e-4);
    }

    #[test]
    fn matches_brute_force_over_a_soup() {
        let tris = soup(2000);
        let bvh = RtBvh::build(&tris);
        let mut rng = Rng::new(11, 12);
        for _ in 0..3000 {
            let o = uniform_sphere(rng.next_f32(), rng.next_f32()) * 12.0;
            let d = (uniform_sphere(rng.next_f32(), rng.next_f32()) * 3.0 - o).normalize();
            let a = bvh.intersect(&tris, o, d, 1e-4, 1e30);
            let b = brute_force(&tris, o, d, 1e-4, 1e30);
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => assert!(
                    (a.t - b.t).abs() < 1e-3,
                    "bvh t={} brute t={} (tri {} vs {})",
                    a.t,
                    b.t,
                    a.triangle,
                    b.triangle
                ),
                (a, b) => panic!("disagreement: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn occlusion_agrees_with_closest_hit() {
        let tris = soup(800);
        let bvh = RtBvh::build(&tris);
        let mut rng = Rng::new(21, 22);
        for _ in 0..2000 {
            let o = uniform_sphere(rng.next_f32(), rng.next_f32()) * 8.0;
            let d = (uniform_sphere(rng.next_f32(), rng.next_f32()) * 2.0 - o).normalize();
            let t_max = 20.0;
            let closest = bvh.intersect(&tris, o, d, 1e-4, t_max).is_some();
            assert_eq!(closest, bvh.occluded(&tris, o, d, 1e-4, t_max));
        }
    }

    #[test]
    fn intersect_all_returns_every_layer_in_order() {
        let mut tris = Vec::new();
        for i in 0..5 {
            tris.extend(quad(i as f32));
        }
        let bvh = RtBvh::build(&tris);
        let mut hits = Vec::new();
        // Off the diagonal the two triangles share, or the ray legitimately
        // hits both of them at the same t.
        bvh.intersect_all(
            &tris,
            Vector3::new(0.3, -0.2, -1.0),
            Vector3::new(0.0, 0.0, 1.0),
            1e-4,
            1e30,
            &mut hits,
        );
        assert_eq!(hits.len(), 5, "one hit per plane, got {hits:?}");
        for w in hits.windows(2) {
            assert!(w[0].t <= w[1].t, "not sorted: {hits:?}");
        }
    }

    #[test]
    fn t_range_is_respected() {
        let tris = quad(0.0);
        let bvh = RtBvh::build(&tris);
        let o = Vector3::new(0.0, 0.0, -3.0);
        let d = Vector3::new(0.0, 0.0, 1.0);
        assert!(bvh.intersect(&tris, o, d, 1e-4, 2.9).is_none());
        assert!(bvh.intersect(&tris, o, d, 3.1, 1e30).is_none());
        assert!(bvh.intersect(&tris, o, d, 1e-4, 3.1).is_some());
    }

    #[test]
    fn axis_aligned_rays_do_not_produce_nan() {
        // A ray whose direction has two zero components stresses the 0 * inf
        // case in the slab test.
        let tris = quad(0.0);
        let bvh = RtBvh::build(&tris);
        for d in [
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ] {
            let hit = bvh.intersect(&tris, Vector3::new(0.0, 0.0, -5.0), d, 1e-4, 1e30);
            if let Some(h) = hit {
                assert!(h.t.is_finite());
            }
        }
    }

    #[test]
    fn tree_covers_every_triangle_exactly_once() {
        let tris = soup(500);
        let bvh = RtBvh::build(&tris);
        let mut seen = vec![0u32; tris.len()];
        for &i in &bvh.order {
            seen[i as usize] += 1;
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "triangle permutation is not one-to-one"
        );
        assert_eq!(bvh.triangle_count(), tris.len());
    }
}
