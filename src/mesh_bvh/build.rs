use crate::core::BufferGeometry;
use crate::math::{Box3, Vector3};

use super::node::BvhNode;

/// Build strategy constants (mirror three-mesh-bvh).
pub const CENTER: u32 = 0;
pub const AVERAGE: u32 = 1;
pub const SAH: u32 = 2;

/// Options for BVH construction.
#[derive(Debug, Clone, Copy)]
pub struct BuildOptions {
    pub strategy: u32,
    pub max_depth: u32,
    pub max_leaf_tris: u32,
    /// First triangle index in geometry (0 = start).
    pub offset: u32,
    /// Triangle count from offset (0 = remainder).
    pub count: u32,
    /// When true, preserve original geometry index order (three-mesh-bvh `indirect`).
    pub indirect: bool,
    /// Split at the median centroid when the chosen plane separates nothing,
    /// instead of giving up and making the whole subtree one leaf.
    ///
    /// Off by default, because three-mesh-bvh makes such a range a leaf, and
    /// the CSG evaluator reads the tree's shape rather than just querying it:
    /// `bvhcast` meets triangle pairs in tree order, and a face is cut in that
    /// order, so a different tree cuts the same region into different pieces.
    /// The default builds three-mesh-bvh 0.7.6's tree exactly.
    ///
    /// Turn it on for queries — raycasts, closest-point — where nothing reads
    /// the shape and a collapsed subtree is a linear scan. See
    /// `tests/mesh_bvh_raycast.rs`.
    pub split_degenerate: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            strategy: CENTER,
            max_depth: 40,
            max_leaf_tris: 10,
            offset: 0,
            count: 0,
            indirect: false,
            split_degenerate: false,
        }
    }
}

pub struct BuildResult {
    pub nodes: Vec<BvhNode>,
    pub triangle_order: Vec<usize>,
    pub positions: Vec<f32>,
    pub triangle_indices: Vec<(u32, u32, u32)>,
    pub root_bounds: Box3,
}

/// Extract triangle list and build a BVH.
pub fn build_bvh(geometry: &BufferGeometry, options: BuildOptions) -> Option<BuildResult> {
    let pos_attr = geometry.get_attribute("position")?;
    if pos_attr.item_size != 3 {
        return None;
    }
    let positions = pos_attr.array.clone();
    let vert_count = positions.len() / 3;

    let triangle_indices: Vec<(u32, u32, u32)> = if let Some(idx) = &geometry.index {
        (0..idx.len() / 3)
            .map(|t| (idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]))
            .collect()
    } else {
        (0..vert_count / 3)
            .map(|t| (t as u32 * 3, t as u32 * 3 + 1, t as u32 * 3 + 2))
            .collect()
    };

    if triangle_indices.is_empty() {
        return None;
    }

    let start_tri = options.offset as usize;
    if start_tri >= triangle_indices.len() {
        return None;
    }
    let tri_count = if options.count == 0 {
        triangle_indices.len() - start_tri
    } else {
        options.count as usize
    };
    if tri_count == 0 || start_tri + tri_count > triangle_indices.len() {
        return None;
    }

    let mut triangle_order: Vec<usize> = (start_tri..start_tri + tri_count).collect();
    // three-mesh-bvh's `computeTriangleBounds`, in the order the build
    // permutes alongside `triangle_order`.
    let mut tri_bounds: Vec<[f32; 6]> = triangle_order
        .iter()
        .map(|&t| triangle_bounds(&positions, triangle_indices[t]))
        .collect();
    let (root, root_centroids) = range_bounds(&tri_bounds, 0, tri_count);
    let root_bounds = to_box(&root);

    let mut nodes = Vec::new();
    let mut build = Build {
        tri_bounds: &mut tri_bounds,
        order: &mut triangle_order,
        options,
        nodes: &mut nodes,
    };
    build.split(0, tri_count, root, root_centroids, 0);

    Some(BuildResult {
        nodes,
        triangle_order,
        positions,
        triangle_indices,
        root_bounds,
    })
}

/// three-mesh-bvh 0.7.6's `buildTree`: the same splits, the same Hoare
/// partition and the same f32 bounds, so the tree — and so the order `bvhcast`
/// meets triangle pairs in, which decides how three-bvh-csg cuts a face — is
/// three-mesh-bvh's.
struct Build<'a> {
    /// Per triangle, `[cx, hx, cy, hy, cz, hz]`: centre and padded half-extent.
    tri_bounds: &'a mut [[f32; 6]],
    order: &'a mut [usize],
    options: BuildOptions,
    nodes: &'a mut Vec<BvhNode>,
}

impl Build<'_> {
    /// `splitNode`: a leaf, or two children split where `getOptimalSplit`
    /// says. `bounds` and `centroids` are the range's node and centroid bounds.
    fn split(
        &mut self,
        start: usize,
        count: usize,
        bounds: [f32; 6],
        centroids: [f32; 6],
        depth: u32,
    ) -> u32 {
        let node_index = self.nodes.len() as u32;
        let leaf = |nodes: &mut Vec<BvhNode>| {
            nodes.push(BvhNode::leaf(to_box(&bounds), start as u32, count as u32));
            node_index
        };
        if count <= self.options.max_leaf_tris as usize || depth >= self.options.max_depth {
            return leaf(self.nodes);
        }
        let Some((axis, pos)) = optimal_split(
            &bounds,
            &centroids,
            &self.tri_bounds[start..start + count],
            self.options.strategy,
        ) else {
            return leaf(self.nodes);
        };
        let mut split_at = partition(self.tri_bounds, self.order, start, count, axis, pos);
        if self.options.split_degenerate && (split_at == start || split_at == start + count) {
            // The plane put every triangle on one side; three-mesh-bvh makes the
            // whole range a leaf, which a query then scans linearly. Splitting
            // at the median centroid always separates the set, unless every
            // centroid is identical — and then the list position is as good.
            let mid = count / 2;
            let mut pairs: Vec<([f32; 6], usize)> = (start..start + count)
                .map(|i| (self.tri_bounds[i], self.order[i]))
                .collect();
            pairs.select_nth_unstable_by(mid, |a, b| {
                a.0[axis * 2]
                    .partial_cmp(&b.0[axis * 2])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (k, (tb, t)) in pairs.into_iter().enumerate() {
                self.tri_bounds[start + k] = tb;
                self.order[start + k] = t;
            }
            split_at = start + mid;
        }
        if split_at == start || split_at == start + count {
            return leaf(self.nodes);
        }
        self.nodes.push(BvhNode::internal(to_box(&bounds), 0, 0));
        let left_count = split_at - start;
        let (lb, lc) = range_bounds(self.tri_bounds, start, left_count);
        let left = self.split(start, left_count, lb, lc, depth + 1);
        let (rb, rc) = range_bounds(self.tri_bounds, split_at, count - left_count);
        let right = self.split(split_at, count - left_count, rb, rc, depth + 1);
        self.nodes[node_index as usize] = BvhNode::split(to_box(&bounds), left, right, axis as u8);
        node_index
    }
}

/// `computeTriangleBounds` for one triangle: per axis the centre
/// `min + half` and the half-extent padded by float32 epsilon scaled to the
/// coordinates, `half + (|min| + half)·2⁻²⁴`, computed in f64 and stored f32.
fn triangle_bounds(positions: &[f32], tri: (u32, u32, u32)) -> [f32; 6] {
    const FLOAT32_EPSILON: f64 = 1.0 / 16_777_216.0;
    let mut out = [0.0f32; 6];
    for el in 0..3 {
        let a = positions[tri.0 as usize * 3 + el] as f64;
        let b = positions[tri.1 as usize * 3 + el] as f64;
        let c = positions[tri.2 as usize * 3 + el] as f64;
        let min = a.min(b).min(c);
        let max = a.max(b).max(c);
        let half = (max - min) / 2.0;
        out[el * 2] = (min + half) as f32;
        out[el * 2 + 1] = (half + (min.abs() + half) * FLOAT32_EPSILON) as f32;
    }
    out
}

/// `getBounds`: the union of a range's triangle bounds and the bounds of
/// their centres, both `[minx, miny, minz, maxx, maxy, maxz]` in f32.
fn range_bounds(tri_bounds: &[[f32; 6]], start: usize, count: usize) -> ([f32; 6], [f32; 6]) {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut c = b;
    for t in &tri_bounds[start..start + count] {
        for d in 0..3 {
            let (center, half) = (t[d * 2] as f64, t[d * 2 + 1] as f64);
            let (lo, hi) = (center - half, center + half);
            if lo < b[d] {
                b[d] = lo;
            }
            if hi > b[d + 3] {
                b[d + 3] = hi;
            }
            if center < c[d] {
                c[d] = center;
            }
            if center > c[d + 3] {
                c[d + 3] = center;
            }
        }
    }
    (b.map(|v| v as f32), c.map(|v| v as f32))
}

fn to_box(b: &[f32; 6]) -> Box3 {
    Box3::new(
        Vector3::new(b[0], b[1], b[2]),
        Vector3::new(b[3], b[4], b[5]),
    )
}

/// `getLongestEdgeIndex`: the first axis of greatest extent.
fn longest_edge(b: &[f32; 6]) -> Option<usize> {
    let mut axis = None;
    let mut dist = f64::NEG_INFINITY;
    for i in 0..3 {
        let d = b[i + 3] as f64 - b[i] as f64;
        if d > dist {
            dist = d;
            axis = Some(i);
        }
    }
    axis
}

/// `computeSurfaceArea`.
fn surface_area(b: &[f32; 6]) -> f64 {
    let d0 = b[3] as f64 - b[0] as f64;
    let d1 = b[4] as f64 - b[1] as f64;
    let d2 = b[5] as f64 - b[2] as f64;
    2.0 * (d0 * d1 + d1 * d2 + d2 * d0)
}

/// `expandByTriangleBounds`, into f32 bounds.
fn expand(t: &[f32; 6], bounds: &mut [f32; 6]) {
    for d in 0..3 {
        let (center, half) = (t[d * 2] as f64, t[d * 2 + 1] as f64);
        let (lo, hi) = (center - half, center + half);
        if lo < bounds[d] as f64 {
            bounds[d] = lo as f32;
        }
        if hi > bounds[d + 3] as f64 {
            bounds[d + 3] = hi as f32;
        }
    }
}

const EMPTY: [f32; 6] = [
    f32::INFINITY,
    f32::INFINITY,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
];

fn union(a: &[f32; 6], b: &[f32; 6]) -> [f32; 6] {
    let mut out = [0.0; 6];
    for d in 0..3 {
        out[d] = if a[d] < b[d] { a[d] } else { b[d] };
        out[d + 3] = if a[d + 3] > b[d + 3] {
            a[d + 3]
        } else {
            b[d + 3]
        };
    }
    out
}

const SAH_BIN_COUNT: usize = 32;
const TRIANGLE_INTERSECT_COST: f64 = 1.25;
const TRAVERSAL_COST: f64 = 1.0;

/// `getOptimalSplit`: the axis and position to split a range at, or `None`
/// to make it a leaf (only SAH declines, when no split beats not splitting).
fn optimal_split(
    bounds: &[f32; 6],
    centroids: &[f32; 6],
    tris: &[[f32; 6]],
    strategy: u32,
) -> Option<(usize, f64)> {
    match strategy {
        CENTER => {
            let axis = longest_edge(centroids)?;
            Some((
                axis,
                (centroids[axis] as f64 + centroids[axis + 3] as f64) / 2.0,
            ))
        }
        AVERAGE => {
            let axis = longest_edge(bounds)?;
            let sum: f64 = tris.iter().map(|t| t[axis * 2] as f64).sum();
            Some((axis, sum / tris.len() as f64))
        }
        _ => sah_split(bounds, centroids, tris),
    }
}

fn sah_split(bounds: &[f32; 6], centroids: &[f32; 6], tris: &[[f32; 6]]) -> Option<(usize, f64)> {
    let count = tris.len();
    let root_area = surface_area(bounds);
    let mut best_cost = TRIANGLE_INTERSECT_COST * count as f64;
    let mut best = None;
    let mut consider =
        |axis: usize, pos: f64, left: usize, left_bounds: &[f32; 6], right_bounds: &[f32; 6]| {
            let right = count - left;
            let lp = if left != 0 {
                surface_area(left_bounds) / root_area
            } else {
                0.0
            };
            let rp = if right != 0 {
                surface_area(right_bounds) / root_area
            } else {
                0.0
            };
            let cost =
                TRAVERSAL_COST + TRIANGLE_INTERSECT_COST * (lp * left as f64 + rp * right as f64);
            if cost < best_cost {
                best_cost = cost;
                best = Some((axis, pos));
            }
        };
    for a in 0..3 {
        let axis_left = centroids[a] as f64;
        let axis_length = centroids[a + 3] as f64 - axis_left;
        let bin_width = axis_length / SAH_BIN_COUNT as f64;
        if count < SAH_BIN_COUNT / 4 {
            // Few triangles: every centre is a candidate, sorted (stably),
            // repeats dropped.
            let mut candidates: Vec<f64> = tris.iter().map(|t| t[a * 2] as f64).collect();
            candidates.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            candidates.dedup();
            for &candidate in &candidates {
                let (mut lb, mut rb, mut left) = (EMPTY, EMPTY, 0usize);
                for t in tris {
                    if t[a * 2] as f64 >= candidate {
                        expand(t, &mut rb);
                    } else {
                        expand(t, &mut lb);
                        left += 1;
                    }
                }
                consider(a, candidate, left, &lb, &rb);
            }
        } else {
            let mut bin_counts = [0usize; SAH_BIN_COUNT];
            let mut bin_bounds = [EMPTY; SAH_BIN_COUNT];
            for t in tris {
                let relative = t[a * 2] as f64 - axis_left;
                // JavaScript's `~~`: truncation, with NaN and ±∞ as 0.
                let q = relative / bin_width;
                let mut bin = if q.is_finite() { q.trunc() as i64 } else { 0 };
                if bin >= SAH_BIN_COUNT as i64 {
                    bin = SAH_BIN_COUNT as i64 - 1;
                }
                let bin = bin.max(0) as usize;
                bin_counts[bin] += 1;
                expand(t, &mut bin_bounds[bin]);
            }
            let mut right_cache = [EMPTY; SAH_BIN_COUNT];
            right_cache[SAH_BIN_COUNT - 1] = bin_bounds[SAH_BIN_COUNT - 1];
            for i in (0..SAH_BIN_COUNT - 1).rev() {
                right_cache[i] = union(&bin_bounds[i], &right_cache[i + 1]);
            }
            let (mut left_bounds, mut left) = (EMPTY, 0usize);
            for i in 0..SAH_BIN_COUNT - 1 {
                if bin_counts[i] != 0 {
                    left_bounds = if left == 0 {
                        bin_bounds[i]
                    } else {
                        union(&bin_bounds[i], &left_bounds)
                    };
                }
                left += bin_counts[i];
                let candidate = axis_left + bin_width + i as f64 * bin_width;
                consider(a, candidate, left, &left_bounds, &right_cache[i + 1]);
            }
        }
    }
    best
}

/// `partition` / `partition_indirect`: Hoare partitioning of a range by
/// triangle centre on `axis`; a centre on the plane goes right. Returns the
/// first index on the right.
fn partition(
    tri_bounds: &mut [[f32; 6]],
    order: &mut [usize],
    start: usize,
    count: usize,
    axis: usize,
    pos: f64,
) -> usize {
    let mut left = start as isize;
    let mut right = (start + count) as isize - 1;
    let c = axis * 2;
    loop {
        while left <= right && (tri_bounds[left as usize][c] as f64) < pos {
            left += 1;
        }
        while left <= right && (tri_bounds[right as usize][c] as f64) >= pos {
            right -= 1;
        }
        if left < right {
            tri_bounds.swap(left as usize, right as usize);
            order.swap(left as usize, right as usize);
            left += 1;
            right -= 1;
        } else {
            return left as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, BufferGeometry};

    fn unit_triangle_geometry() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g
    }

    #[test]
    fn builds_with_all_strategies() {
        let geom = unit_triangle_geometry();
        for strategy in [CENTER, AVERAGE, SAH] {
            let opts = BuildOptions {
                strategy,
                ..Default::default()
            };
            assert!(build_bvh(&geom, opts).is_some(), "strategy {strategy}");
        }
    }
}
