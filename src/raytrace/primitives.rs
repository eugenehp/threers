//! Convert non-mesh drawables into triangle soup the path tracer can hit.

use std::sync::Arc;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::materials::Material;
use crate::math::{Matrix4, Vector3};

/// A camera-facing quad in world space, as two triangles.
pub fn sprite_quads(world: &Matrix4, size: f32) -> Arc<BufferGeometry> {
    let hs = size.max(1e-4) * 0.5;
    let mut positions = vec![
        -hs, -hs, 0.0, //
        hs, -hs, 0.0, //
        hs, hs, 0.0, //
        -hs, hs, 0.0,
    ];
    for chunk in positions.chunks_exact_mut(3) {
        let p = Vector3::new(chunk[0], chunk[1], chunk[2]).apply_matrix4(world);
        chunk[0] = p.x;
        chunk[1] = p.y;
        chunk[2] = p.z;
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    geom.set_index(vec![0, 1, 2, 0, 2, 3]);
    Arc::new(geom)
}

/// Expand line segments into thin world-space quads (view-independent ribbon).
pub fn line_segment_quads(
    geometry: &BufferGeometry,
    world: &Matrix4,
    width: f32,
) -> Option<Arc<BufferGeometry>> {
    let pos = geometry.get_attribute("position")?;
    if pos.item_size < 3 || pos.count() < 2 {
        return None;
    }
    let half = width.max(1e-4) * 0.5;
    let mut out_pos = Vec::new();
    let mut out_idx = Vec::new();
    let read = |i: usize| -> Vector3 {
        let b = i * pos.item_size;
        Vector3::new(pos.array[b], pos.array[b + 1], pos.array[b + 2]).apply_matrix4(world)
    };
    let n_verts = pos.count();
    let is_segments = n_verts % 2 == 0;
    let pairs = if is_segments {
        (0..n_verts).step_by(2).collect::<Vec<_>>()
    } else {
        (0..n_verts.saturating_sub(1)).collect::<Vec<_>>()
    };
    for &i in &pairs {
        // Segments pair (0,1), (2,3), …; a strip pairs (0,1), (1,2), …. Which
        // it is decides `pairs` above, not the step from `i`, which is one
        // either way.
        let j = i + 1;
        if j >= n_verts {
            break;
        }
        let a = read(i);
        let b = read(j);
        let dir = b - a;
        if dir.length_sq() < 1e-12 {
            continue;
        }
        let t = dir.normalize();
        let side = if t.cross(Vector3::UP).length_sq() > 1e-6 {
            t.cross(Vector3::UP).normalize()
        } else {
            t.cross(Vector3::RIGHT).normalize()
        };
        let base = (out_pos.len() / 3) as u32;
        for corner in [a - side * half, a + side * half, b + side * half, b - side * half] {
            out_pos.extend_from_slice(&[corner.x, corner.y, corner.z]);
        }
        out_idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    if out_idx.is_empty() {
        return None;
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(out_pos, 3));
    geom.set_index(out_idx);
    Some(Arc::new(geom))
}

/// Expand each point into a small octahedron (8 triangles).
pub fn point_spheres(
    geometry: &BufferGeometry,
    world: &Matrix4,
    radius: f32,
) -> Option<Arc<BufferGeometry>> {
    let pos = geometry.get_attribute("position")?;
    if pos.item_size < 3 || pos.count() == 0 {
        return None;
    }
    let r = radius.max(1e-4);
    let mut out_pos = Vec::new();
    let mut out_idx = Vec::new();
    for i in 0..pos.count() {
        let b = i * pos.item_size;
        let c = Vector3::new(pos.array[b], pos.array[b + 1], pos.array[b + 2]).apply_matrix4(world);
        let base = (out_pos.len() / 3) as u32;
        let axes = [
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(-r, 0.0, 0.0),
            Vector3::new(0.0, r, 0.0),
            Vector3::new(0.0, -r, 0.0),
            Vector3::new(0.0, 0.0, r),
            Vector3::new(0.0, 0.0, -r),
        ];
        for a in &axes {
            let p = c + *a;
            out_pos.extend_from_slice(&[p.x, p.y, p.z]);
        }
        let v = |k: u32| base + k;
        for tri in [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ] {
            out_idx.extend_from_slice(&[v(tri[0]), v(tri[1]), v(tri[2])]);
        }
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(out_pos, 3));
    geom.set_index(out_idx);
    Some(Arc::new(geom))
}

/// Default line width in world units when the material does not specify one.
pub fn line_width_for(_material: &Material) -> f32 {
    0.02
}

/// Default point radius in world units.
pub fn point_radius_for(_material: &Material) -> f32 {
    0.05
}
