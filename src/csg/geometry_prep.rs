//! Geometry helpers for CSG brushes (indexing, soup expansion, reference bins).

use crate::core::{BufferAttribute, BufferGeometry};

pub fn tri_count(geometry: &BufferGeometry) -> usize {
    if let Some(idx) = &geometry.index {
        idx.len() / 3
    } else {
        geometry
            .get_attribute("position")
            .map(|a| a.count() / 3)
            .unwrap_or(0)
    }
}

pub fn index_at(geometry: &BufferGeometry, tri: usize, corner: usize) -> usize {
    let i = tri * 3 + corner;
    if let Some(idx) = &geometry.index {
        idx[i] as usize
    } else {
        i
    }
}

/// Expand indexed geometry to triangle soup (three.js `toNonIndexed`).
pub fn ensure_non_indexed(geometry: &mut BufferGeometry) {
    geometry.expand_index();
}

pub fn ensure_index(geometry: &mut BufferGeometry) {
    if geometry.index.is_some() {
        return;
    }
    let count = geometry
        .get_attribute("position")
        .map(|a| a.count())
        .unwrap_or(0);
    let index: Vec<u32> = (0..count as u32).collect();
    geometry.set_index(index);
}

/// Each triangle's group, as three-bvh-csg's `Brush.prepareGeometry` builds
/// `groupIndices`: group `i` covers triangles `start/3 .. (start + count)/3`,
/// later groups overwriting earlier ones, and a triangle in none is group 0.
pub fn build_group_indices(geometry: &BufferGeometry) -> Vec<u16> {
    let n = tri_count(geometry);
    let mut out = vec![0u16; n];
    for (i, g) in geometry.groups.iter().enumerate() {
        let first = (g.start / 3) as usize;
        let end = ((g.start as u64 + g.count as u64) / 3).min(n as u64) as usize;
        for t in out.iter_mut().take(end).skip(first) {
            *t = i as u16;
        }
    }
    out
}

pub fn read_position(attr: &BufferAttribute, index: usize) -> crate::math::Vector3 {
    let i = index * attr.item_size;
    crate::math::Vector3::new(attr.array[i], attr.array[i + 1], attr.array[i + 2])
}
