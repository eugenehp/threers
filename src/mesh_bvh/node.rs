use crate::math::{Box3, Vector3};

/// Flat BVH node: axis-aligned bounds plus child links or leaf triangle range.
#[derive(Debug, Clone, Copy)]
pub struct BvhNode {
    pub bounds: Box3,
    /// Internal node: index of left child. Leaf: triangle offset in `triangle_order`.
    pub left_or_offset: u32,
    /// Internal node: index of right child. Leaf: triangle count.
    pub right_or_count: u32,
    pub is_leaf: bool,
}

impl BvhNode {
    pub fn leaf(bounds: Box3, offset: u32, count: u32) -> Self {
        Self {
            bounds,
            left_or_offset: offset,
            right_or_count: count,
            is_leaf: true,
        }
    }

    pub fn internal(bounds: Box3, left: u32, right: u32) -> Self {
        Self {
            bounds,
            left_or_offset: left,
            right_or_count: right,
            is_leaf: false,
        }
    }

    /// Export as 8 f32 values for JS shapecast: min(3), max(3), meta0, meta1.
    /// meta0: leaf => -(offset+1), internal => left child index
    /// meta1: leaf => count, internal => right child index
    pub fn to_buffer(self, out: &mut [f32], index: usize) {
        let base = index * 8;
        out[base] = self.bounds.min.x;
        out[base + 1] = self.bounds.min.y;
        out[base + 2] = self.bounds.min.z;
        out[base + 3] = self.bounds.max.x;
        out[base + 4] = self.bounds.max.y;
        out[base + 5] = self.bounds.max.z;
        if self.is_leaf {
            out[base + 6] = -(self.left_or_offset as f32 + 1.0);
            out[base + 7] = self.right_or_count as f32;
        } else {
            out[base + 6] = self.left_or_offset as f32;
            out[base + 7] = self.right_or_count as f32;
        }
    }
}

/// Compute the AABB enclosing a set of triangle centroids / vertices.
pub fn bounds_from_triangles(
    positions: &[f32],
    indices: &[(u32, u32, u32)],
    tri_order: &[usize],
    start: usize,
    count: usize,
) -> Box3 {
    let mut b = Box3::empty();
    for i in start..start + count {
        let tri = indices[tri_order[i]];
        for &vi in &[tri.0, tri.1, tri.2] {
            let base = vi as usize * 3;
            b.expand_by_point(Vector3::new(
                positions[base],
                positions[base + 1],
                positions[base + 2],
            ));
        }
    }
    if b.is_empty() {
        Box3::new(Vector3::ZERO, Vector3::ZERO)
    } else {
        b
    }
}
