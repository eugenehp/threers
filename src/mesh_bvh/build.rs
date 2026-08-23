use crate::core::BufferGeometry;
use crate::math::{Box3, Vector3};

use super::node::{bounds_from_triangles, BvhNode};

/// Build strategy constants (mirror three-mesh-bvh).
pub const CENTER: u32 = 0;
pub const AVERAGE: u32 = 1;
pub const SAH: u32 = 2;

const SAH_BINS: usize = 32;

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
    let root_bounds =
        bounds_from_triangles(&positions, &triangle_indices, &triangle_order, 0, tri_count);

    let mut nodes = Vec::new();
    build_recursive(
        &positions,
        &triangle_indices,
        &mut triangle_order,
        0,
        tri_count,
        0,
        options,
        &mut nodes,
    );

    Some(BuildResult {
        nodes,
        triangle_order,
        positions,
        triangle_indices,
        root_bounds,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_recursive(
    positions: &[f32],
    triangle_indices: &[(u32, u32, u32)],
    triangle_order: &mut [usize],
    start: usize,
    count: usize,
    depth: u32,
    options: BuildOptions,
    nodes: &mut Vec<BvhNode>,
) -> u32 {
    let bounds = bounds_from_triangles(positions, triangle_indices, triangle_order, start, count);
    let node_index = nodes.len() as u32;

    if count <= options.max_leaf_tris as usize || depth >= options.max_depth {
        nodes.push(BvhNode::leaf(bounds, start as u32, count as u32));
        return node_index;
    }

    nodes.push(BvhNode::internal(bounds, 0, 0));

    let axis = longest_axis(&bounds);
    let split = compute_split(
        positions,
        triangle_indices,
        triangle_order,
        start,
        count,
        axis,
        &bounds,
        options.strategy,
    );

    let mut i = start;
    let mut j = start + count;
    while i < j {
        let tri_idx = triangle_order[i];
        let tri = triangle_indices[tri_idx];
        let cx = centroid_axis(positions, tri, axis);
        if cx < split {
            i += 1;
        } else {
            j -= 1;
            triangle_order.swap(i, j);
        }
    }

    if i == start || i == start + count {
        nodes[node_index as usize] = BvhNode::leaf(bounds, start as u32, count as u32);
        return node_index;
    }

    let left_count = i - start;
    let right_count = count - left_count;

    let left = build_recursive(
        positions,
        triangle_indices,
        triangle_order,
        start,
        left_count,
        depth + 1,
        options,
        nodes,
    );
    let right = build_recursive(
        positions,
        triangle_indices,
        triangle_order,
        start + left_count,
        right_count,
        depth + 1,
        options,
        nodes,
    );

    nodes[node_index as usize] = BvhNode::internal(bounds, left, right);
    node_index
}

#[allow(clippy::too_many_arguments)]
fn compute_split(
    positions: &[f32],
    triangle_indices: &[(u32, u32, u32)],
    triangle_order: &[usize],
    start: usize,
    count: usize,
    axis: usize,
    bounds: &Box3,
    strategy: u32,
) -> f32 {
    match strategy {
        AVERAGE => {
            let mut sum = 0.0f32;
            for i in start..start + count {
                let tri = triangle_indices[triangle_order[i]];
                sum += centroid_axis(positions, tri, axis);
            }
            sum / count as f32
        }
        SAH => compute_sah_split(
            positions,
            triangle_indices,
            triangle_order,
            start,
            count,
            axis,
            bounds,
        ),
        _ => (bounds.min.axis(axis) + bounds.max.axis(axis)) * 0.5,
    }
}

fn compute_sah_split(
    positions: &[f32],
    triangle_indices: &[(u32, u32, u32)],
    triangle_order: &[usize],
    start: usize,
    count: usize,
    axis: usize,
    bounds: &Box3,
) -> f32 {
    let mut best_cost = f32::INFINITY;
    let mut best_split = (bounds.min.axis(axis) + bounds.max.axis(axis)) * 0.5;
    let min_a = bounds.min.axis(axis);
    let max_a = bounds.max.axis(axis);
    if (max_a - min_a).abs() < 1e-8 {
        return best_split;
    }

    for bin in 1..SAH_BINS {
        let t = bin as f32 / SAH_BINS as f32;
        let split = min_a * (1.0 - t) + max_a * t;
        let (left_bounds, right_bounds, left_count, right_count) = partition_bounds(
            positions,
            triangle_indices,
            triangle_order,
            start,
            count,
            axis,
            split,
        );
        if left_count == 0 || right_count == 0 {
            continue;
        }
        let cost = surface_area(&left_bounds) * left_count as f32
            + surface_area(&right_bounds) * right_count as f32;
        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }
    best_split
}

fn partition_bounds(
    positions: &[f32],
    triangle_indices: &[(u32, u32, u32)],
    triangle_order: &[usize],
    start: usize,
    count: usize,
    axis: usize,
    split: f32,
) -> (Box3, Box3, usize, usize) {
    let mut left = Box3::empty();
    let mut right = Box3::empty();
    let mut left_count = 0usize;
    let mut right_count = 0usize;
    // `i` walks a slice of the shared permutation, which is written back below.
    #[allow(clippy::needless_range_loop)]
    for i in start..start + count {
        let tri_idx = triangle_order[i];
        let tri = triangle_indices[tri_idx];
        let b = triangle_bounds(positions, tri);
        let cx = centroid_axis(positions, tri, axis);
        if cx < split {
            left = left.union(&b);
            left_count += 1;
        } else {
            right = right.union(&b);
            right_count += 1;
        }
    }
    (left, right, left_count, right_count)
}

fn triangle_bounds(positions: &[f32], tri: (u32, u32, u32)) -> Box3 {
    let mut b = Box3::empty();
    for vi in [tri.0, tri.1, tri.2] {
        let base = vi as usize * 3;
        b.expand_by_point(Vector3::new(
            positions[base],
            positions[base + 1],
            positions[base + 2],
        ));
    }
    b
}

fn surface_area(b: &Box3) -> f32 {
    if b.is_empty() {
        return 0.0;
    }
    let s = b.size();
    2.0 * (s.x * s.y + s.x * s.z + s.y * s.z)
}

fn longest_axis(bounds: &Box3) -> usize {
    let size = bounds.max - bounds.min;
    if size.x >= size.y && size.x >= size.z {
        0
    } else if size.y >= size.z {
        1
    } else {
        2
    }
}

fn centroid_axis(positions: &[f32], tri: (u32, u32, u32), axis: usize) -> f32 {
    let mut sum = 0.0f32;
    for vi in [tri.0, tri.1, tri.2] {
        sum += positions[vi as usize * 3 + axis];
    }
    sum / 3.0
}

trait AxisIndex {
    fn axis(&self, i: usize) -> f32;
}

impl AxisIndex for Vector3 {
    fn axis(&self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
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
