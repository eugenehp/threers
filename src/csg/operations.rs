use std::collections::BTreeSet;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{Matrix3, Triangle};

use super::attribute_data::{AttrSet, TypedAttributeData};
use super::brush::CsgBrush;
use super::constants::*;
use super::geometry_prep::{index_at, read_position, tri_count};
use super::hit_side::{
    get_hit_side_js, get_hit_side_with_coplanar_check_js, get_operation_action, matrix_b_into_a,
};
use super::intersection_map::IntersectionMap;
use super::js_topology::{
    interp_bary_js, js_tri_from_indices, matrix_a_to_b_brushes, read_position_js, JsMatrix3,
    JsMatrix4, JsVec3, SplitBary,
};
use super::triangle_splitter::TriangleSplitter;
use super::triangle_utils::is_tri_degenerate;

const FLOATING_COPLANAR_EPSILON: f32 = 1e-14;

pub fn collect_intersecting_triangles(
    a: &CsgBrush,
    b: &CsgBrush,
) -> (IntersectionMap, IntersectionMap) {
    let mut a_map = IntersectionMap::new();
    let mut b_map = IntersectionMap::new();
    let matrix = matrix_b_into_a(&a.matrix_world, &b.matrix_world);
    let a_bvh = a.bounds_tree();
    let b_bvh = b.bounds_tree();
    let pairs = a_bvh.bvhcast(&b_bvh, &matrix);
    let a_pos = a.geometry.get_attribute("position").unwrap();
    let b_pos = b.geometry.get_attribute("position").unwrap();

    for (ia, ib) in pairs {
        let va = resolve_tri(a, ia);
        let vb_local = resolve_tri(b, ib);
        let vb = Triangle::new(
            vb_local.a.apply_matrix4(&matrix),
            vb_local.b.apply_matrix4(&matrix),
            vb_local.c.apply_matrix4(&matrix),
        );
        if is_tri_degenerate(&va, 1e-14) || is_tri_degenerate(&vb, 1e-14) {
            continue;
        }
        let mut intersected = va.intersects_triangle_ext(&vb, true);
        if !intersected {
            let pa = va.plane();
            let pb = vb.plane();
            if pa.normal.dot(pb.normal) == 1.0
                && (pa.constant - pb.constant).abs() < FLOATING_COPLANAR_EPSILON
            {
                intersected = true;
            }
        }
        if intersected {
            let va_idx = a_bvh.resolve_triangle_index(ia).unwrap_or(ia);
            let vb_idx = b_bvh.resolve_triangle_index(ib).unwrap_or(ib);
            a_map.add(va_idx, vb_idx);
            b_map.add(vb_idx, va_idx);
        }
        let _ = (a_pos, b_pos);
    }
    (a_map, b_map)
}

fn resolve_tri(brush: &CsgBrush, order_index: usize) -> Triangle {
    brush.bounds_tree().triangle_at_order_index(order_index)
}

pub fn perform_operation(
    a: &mut CsgBrush,
    b: &mut CsgBrush,
    operations: &[u8],
    splitter: &mut TriangleSplitter,
    attribute_data: &mut [TypedAttributeData],
    use_groups: bool,
) {
    a.prepare_geometry();
    b.prepare_geometry();
    let (a_inter, b_inter) = collect_intersecting_triangles(a, b);
    let group_offset_a = if use_groups { 0 } else { -1 };
    perform_split_triangle_operations(
        a,
        b,
        &a_inter,
        operations,
        false,
        splitter,
        attribute_data,
        group_offset_a,
    );
    perform_whole_triangle_operations(
        a,
        b,
        &a_inter,
        operations,
        false,
        attribute_data,
        group_offset_a,
    );

    let non_hollow = operations
        .iter()
        .any(|&op| op != HOLLOW_INTERSECTION && op != HOLLOW_SUBTRACTION);
    if non_hollow {
        let group_offset_b = if use_groups { 1 } else { -1 };
        perform_split_triangle_operations(
            b,
            a,
            &b_inter,
            operations,
            true,
            splitter,
            attribute_data,
            group_offset_b,
        );
        perform_whole_triangle_operations(
            b,
            a,
            &b_inter,
            operations,
            true,
            attribute_data,
            group_offset_b,
        );
    }
}

fn perform_split_triangle_operations(
    a: &CsgBrush,
    b: &CsgBrush,
    intersection_map: &IntersectionMap,
    operations: &[u8],
    invert: bool,
    splitter: &mut TriangleSplitter,
    attribute_data: &mut [TypedAttributeData],
    group_offset: i32,
) {
    let inverted_geometry = a.matrix_world.determinant() < 0.0;
    let matrix = matrix_a_to_b_brushes(a, b);
    let mut normal_matrix = JsMatrix3::from_matrix3(&Matrix3::normal_matrix(&a.matrix_world));
    if inverted_geometry {
        normal_matrix = normal_matrix.multiply_scalar(-1.0);
    }
    let world_matrix = a.js_matrix_world();
    let b_bvh = b.bounds_tree();
    let pos = a.geometry.get_attribute("position").unwrap();

    for &ia in &intersection_map.ids {
        let group_index = if group_offset == -1 {
            0usize
        } else {
            a.group_indices[ia] as usize + group_offset as usize
        };
        let i0 = index_at(&a.geometry, ia, 0);
        let i1 = index_at(&a.geometry, ia, 1);
        let i2 = index_at(&a.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        splitter.reset();
        splitter.initialize(tri_a);
        if let Some(neighbors) = intersection_map.intersection_set.get(&ia) {
            let b_pos = b.geometry.get_attribute("position").unwrap();
            for &ib in neighbors {
                let j0 = index_at(&b.geometry, ib, 0);
                let j1 = index_at(&b.geometry, ib, 1);
                let j2 = index_at(&b.geometry, ib, 2);
                let tri_b = js_tri_from_indices(b_pos, j0, j1, j2, None);
                splitter.split_by_triangle(tri_b);
            }
        }
        for ti in 0..splitter.triangle_count() {
            let clipped = splitter.clipped_triangle(ti);
            let hit_side = if splitter.coplanar_triangle_used {
                get_hit_side_with_coplanar_check_js(clipped, &b_bvh)
            } else {
                get_hit_side_js(clipped, &b_bvh)
            };
            let bary = TriangleSplitter::barycoords_in_tri(tri_a, clipped);
            for (oi, op) in operations.iter().enumerate() {
                let action = get_operation_action(*op, hit_side, invert);
                if action == SKIP_TRI {
                    continue;
                }
                let invert_tri = action == INVERT_TRI;
                append_attribute_from_triangle(
                    ia,
                    bary,
                    &a.geometry,
                    &world_matrix,
                    &normal_matrix,
                    attribute_data[oi].get_group_attr_set(group_index),
                    inverted_geometry != invert_tri,
                );
            }
        }
    }
}

fn perform_whole_triangle_operations(
    a: &CsgBrush,
    b: &CsgBrush,
    split_set: &IntersectionMap,
    operations: &[u8],
    invert: bool,
    attribute_data: &mut [TypedAttributeData],
    group_offset: i32,
) {
    let inverted_geometry = a.matrix_world.determinant() < 0.0;
    let matrix = matrix_a_to_b_brushes(a, b);
    let mut normal_matrix = JsMatrix3::from_matrix3(&Matrix3::normal_matrix(&a.matrix_world));
    if inverted_geometry {
        normal_matrix = normal_matrix.multiply_scalar(-1.0);
    }
    let world_matrix = a.js_matrix_world();
    let b_bvh = b.bounds_tree();
    let half_edges = a.half_edges.as_ref().expect("half edges");
    let count = tri_count(&a.geometry);
    let pos = a.geometry.get_attribute("position").unwrap();
    let mut traverse_set: BTreeSet<usize> = (0..count)
        .filter(|i| !split_set.intersection_set.contains_key(i))
        .collect();

    while let Some(id) = traverse_set.pop_first() {
        let mut stack = vec![id];

        let i0 = index_at(&a.geometry, id, 0);
        let i1 = index_at(&a.geometry, id, 1);
        let i2 = index_at(&a.geometry, id, 2);
        let tri = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        let hit_side = get_hit_side_js(tri, &b_bvh);
        let mut actions = Vec::new();
        let mut op_indices = Vec::new();
        for (oi, op) in operations.iter().enumerate() {
            let action = get_operation_action(*op, hit_side, invert);
            if action != SKIP_TRI {
                actions.push(action);
                op_indices.push(oi);
            }
        }

        while let Some(curr) = stack.pop() {
            for e in 0..3 {
                let sid = half_edges.sibling_triangle_index(curr, e);
                if sid >= 0 && traverse_set.remove(&(sid as usize)) {
                    stack.push(sid as usize);
                }
            }
            if actions.is_empty() {
                continue;
            }
            let j0 = index_at(&a.geometry, curr, 0);
            let j1 = index_at(&a.geometry, curr, 1);
            let j2 = index_at(&a.geometry, curr, 2);
            let tri_local = Triangle::new(
                read_position(pos, j0),
                read_position(pos, j1),
                read_position(pos, j2),
            );
            if is_tri_degenerate(&tri_local, 1e-14) {
                continue;
            }
            let group_index = if group_offset == -1 {
                0usize
            } else {
                a.group_indices[curr] as usize + group_offset as usize
            };
            for (&action, &oi) in actions.iter().zip(op_indices.iter()) {
                let invert_tri = action == INVERT_TRI;
                append_attributes_from_indices(
                    j0,
                    j1,
                    j2,
                    &a.geometry,
                    &world_matrix,
                    &normal_matrix,
                    attribute_data[oi].get_group_attr_set(group_index),
                    invert_tri != inverted_geometry,
                );
            }
        }
    }
}

#[cfg(test)]
fn tri_key_from_split_bary(
    geometry: &BufferGeometry,
    tri_index: usize,
    matrix_world: &JsMatrix4,
    bary: SplitBary,
) -> super::topology::TriKey {
    use super::topology::TriKey;
    use super::triangle_utils::hash_coord;
    let pos = geometry.get_attribute("position").unwrap();
    let i0 = index_at(geometry, tri_index, 0);
    let i1 = index_at(geometry, tri_index, 1);
    let i2 = index_at(geometry, tri_index, 2);
    let sa = read_position_js(pos, i0).apply_matrix4(matrix_world);
    let sb = read_position_js(pos, i1).apply_matrix4(matrix_world);
    let sc = read_position_js(pos, i2).apply_matrix4(matrix_world);
    let a = interp_bary_js(sa, sb, sc, bary.0);
    let b = interp_bary_js(sa, sb, sc, bary.1);
    let c = interp_bary_js(sa, sb, sc, bary.2);
    TriKey((
        hash_coord(a.x),
        hash_coord(a.y),
        hash_coord(a.z),
        hash_coord(b.x),
        hash_coord(b.y),
        hash_coord(b.z),
        hash_coord(c.x),
        hash_coord(c.y),
        hash_coord(c.z),
    ))
}

/// All clipped shell triangles (before hit-side), as world-space tri keys.
#[cfg(test)]
pub(crate) fn shell_split_clipped_tri_keys(
    a: &mut CsgBrush,
    b: &mut CsgBrush,
) -> Vec<super::topology::TriKey> {
    a.prepare_geometry();
    b.prepare_geometry();
    let (a_inter, _) = collect_intersecting_triangles(a, b);
    let mut splitter = TriangleSplitter::new();
    let matrix = matrix_a_to_b_brushes(a, b);
    let world_matrix = a.js_matrix_world();
    let pos = a.geometry.get_attribute("position").unwrap();
    let mut keys = Vec::new();
    for &ia in &a_inter.ids {
        let i0 = index_at(&a.geometry, ia, 0);
        let i1 = index_at(&a.geometry, ia, 1);
        let i2 = index_at(&a.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        splitter.reset();
        splitter.initialize(tri_a);
        if let Some(neighbors) = a_inter.intersection_set.get(&ia) {
            let b_pos = b.geometry.get_attribute("position").unwrap();
            for &ib in neighbors {
                let j0 = index_at(&b.geometry, ib, 0);
                let j1 = index_at(&b.geometry, ib, 1);
                let j2 = index_at(&b.geometry, ib, 2);
                let tri_b = js_tri_from_indices(b_pos, j0, j1, j2, None);
                splitter.split_by_triangle(tri_b);
            }
        }
        for (_, bary) in splitter.clipped_with_bary(tri_a) {
            keys.push(tri_key_from_split_bary(
                &a.geometry,
                ia,
                &world_matrix,
                bary,
            ));
        }
    }
    keys
}

/// Shell-side split clipped triangle count (before hit-side culling).
#[cfg(test)]
pub(crate) fn shell_split_clipped_count(a: &mut CsgBrush, b: &mut CsgBrush) -> usize {
    a.prepare_geometry();
    b.prepare_geometry();
    let (a_inter, _) = collect_intersecting_triangles(a, b);
    let mut splitter = TriangleSplitter::new();
    let matrix = matrix_a_to_b_brushes(a, b);
    let pos = a.geometry.get_attribute("position").unwrap();
    let mut total = 0usize;
    for &ia in &a_inter.ids {
        let i0 = index_at(&a.geometry, ia, 0);
        let i1 = index_at(&a.geometry, ia, 1);
        let i2 = index_at(&a.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        splitter.reset();
        splitter.initialize(tri_a);
        if let Some(neighbors) = a_inter.intersection_set.get(&ia) {
            let b_pos = b.geometry.get_attribute("position").unwrap();
            for &ib in neighbors {
                let j0 = index_at(&b.geometry, ib, 0);
                let j1 = index_at(&b.geometry, ib, 1);
                let j2 = index_at(&b.geometry, ib, 2);
                let tri_b = js_tri_from_indices(b_pos, j0, j1, j2, None);
                splitter.split_by_triangle(tri_b);
            }
        }
        total += splitter.triangle_count();
    }
    total
}

/// Shell-side kept keys after hit-side filter (same keying as JS `export-shell-split-tris.mjs`).
#[cfg(test)]
pub(crate) fn shell_split_tri_keys(
    a: &mut CsgBrush,
    b: &mut CsgBrush,
) -> Vec<super::topology::TriKey> {
    a.prepare_geometry();
    b.prepare_geometry();
    let (a_inter, _) = collect_intersecting_triangles(a, b);
    let mut splitter = TriangleSplitter::new();
    let matrix = matrix_a_to_b_brushes(a, b);
    let world_matrix = a.js_matrix_world();
    let pos = a.geometry.get_attribute("position").unwrap();
    let b_bvh = b.bounds_tree();
    let mut keys = Vec::new();
    for &ia in &a_inter.ids {
        let i0 = index_at(&a.geometry, ia, 0);
        let i1 = index_at(&a.geometry, ia, 1);
        let i2 = index_at(&a.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        splitter.reset();
        splitter.initialize(tri_a);
        if let Some(neighbors) = a_inter.intersection_set.get(&ia) {
            let b_pos = b.geometry.get_attribute("position").unwrap();
            for &ib in neighbors {
                let j0 = index_at(&b.geometry, ib, 0);
                let j1 = index_at(&b.geometry, ib, 1);
                let j2 = index_at(&b.geometry, ib, 2);
                let tri_b = js_tri_from_indices(b_pos, j0, j1, j2, None);
                splitter.split_by_triangle(tri_b);
            }
        }
        for ti in 0..splitter.triangle_count() {
            let clipped = splitter.clipped_triangle(ti);
            let hit_side = if splitter.coplanar_triangle_used {
                get_hit_side_with_coplanar_check_js(clipped, &b_bvh)
            } else {
                get_hit_side_js(clipped, &b_bvh)
            };
            if get_operation_action(ADDITION, hit_side, false) == SKIP_TRI {
                continue;
            }
            let bary = TriangleSplitter::barycoords_in_tri(tri_a, clipped);
            keys.push(tri_key_from_split_bary(
                &a.geometry,
                ia,
                &world_matrix,
                bary,
            ));
        }
    }
    keys
}

fn append_attribute_from_triangle(
    tri_index: usize,
    bary: SplitBary,
    geometry: &BufferGeometry,
    matrix_world: &JsMatrix4,
    normal_matrix: &JsMatrix3,
    attr_set: &mut AttrSet,
    invert: bool,
) {
    let i0 = index_at(geometry, tri_index, 0);
    let i1 = index_at(geometry, tri_index, 1);
    let i2 = index_at(geometry, tri_index, 2);
    for (key, arr) in attr_set.iter_mut() {
        let attr = geometry.get_attribute(key).expect("attr");
        if key == "position" {
            let v0 = read_position_js(attr, i0).apply_matrix4(matrix_world);
            let v1 = read_position_js(attr, i1).apply_matrix4(matrix_world);
            let v2 = read_position_js(attr, i2).apply_matrix4(matrix_world);
            push_bary_js(v0, v1, v2, bary, arr, invert, false);
        } else if key == "normal" {
            let mut v0 = read_position_js(attr, i0).apply_normal_matrix(normal_matrix);
            let mut v1 = read_position_js(attr, i1).apply_normal_matrix(normal_matrix);
            let mut v2 = read_position_js(attr, i2).apply_normal_matrix(normal_matrix);
            if invert {
                v0 = v0.multiply_scalar(-1.0);
                v1 = v1.multiply_scalar(-1.0);
                v2 = v2.multiply_scalar(-1.0);
            }
            push_bary_js(v0, v1, v2, bary, arr, invert, true);
        } else if let Some(attr) = geometry.get_attribute(key) {
            let a = read_vec(attr, i0);
            let b = read_vec(attr, i1);
            let c = read_vec(attr, i2);
            push_bary_vec_f64(&a, &b, &c, bary, arr.item_size, arr, invert);
        }
    }
}

fn append_attributes_from_indices(
    i0: usize,
    i1: usize,
    i2: usize,
    geometry: &BufferGeometry,
    matrix_world: &JsMatrix4,
    normal_matrix: &JsMatrix3,
    attr_set: &mut AttrSet,
    invert: bool,
) {
    append_attribute_from_index(i0, geometry, matrix_world, normal_matrix, attr_set, invert);
    if invert {
        append_attribute_from_index(i2, geometry, matrix_world, normal_matrix, attr_set, invert);
        append_attribute_from_index(i1, geometry, matrix_world, normal_matrix, attr_set, invert);
    } else {
        append_attribute_from_index(i1, geometry, matrix_world, normal_matrix, attr_set, invert);
        append_attribute_from_index(i2, geometry, matrix_world, normal_matrix, attr_set, invert);
    }
}

fn append_attribute_from_index(
    index: usize,
    geometry: &BufferGeometry,
    matrix_world: &JsMatrix4,
    normal_matrix: &JsMatrix3,
    attr_set: &mut AttrSet,
    invert: bool,
) {
    for (key, arr) in attr_set.iter_mut() {
        let attr = geometry.get_attribute(key).expect("attr");
        if key == "position" {
            let p = read_position_js(attr, index).apply_matrix4(matrix_world);
            arr.push_values(&[p.x as f32, p.y as f32, p.z as f32]);
        } else if key == "normal" {
            let mut n = read_position_js(attr, index).apply_normal_matrix(normal_matrix);
            if invert {
                n = n.multiply_scalar(-1.0);
            }
            arr.push_values(&[n.x as f32, n.y as f32, n.z as f32]);
        } else {
            let v = read_vec(attr, index);
            arr.push_values(&v);
        }
    }
}

fn read_vec(attr: &BufferAttribute, index: usize) -> Vec<f32> {
    let i = index * attr.item_size;
    attr.array[i..i + attr.item_size].to_vec()
}

fn push_bary_js(
    v0: JsVec3,
    v1: JsVec3,
    v2: JsVec3,
    bary: SplitBary,
    arr: &mut super::attribute_data::TypedArray,
    invert: bool,
    normalize: bool,
) {
    let mut a = interp_bary_js(v0, v1, v2, bary.0);
    let mut b = interp_bary_js(v0, v1, v2, bary.1);
    let mut c = interp_bary_js(v0, v1, v2, bary.2);
    if normalize {
        a = a.normalize();
        b = b.normalize();
        c = c.normalize();
    }
    if invert {
        arr.push_values(&[a.x as f32, a.y as f32, a.z as f32]);
        arr.push_values(&[c.x as f32, c.y as f32, c.z as f32]);
        arr.push_values(&[b.x as f32, b.y as f32, b.z as f32]);
    } else {
        arr.push_values(&[a.x as f32, a.y as f32, a.z as f32]);
        arr.push_values(&[b.x as f32, b.y as f32, b.z as f32]);
        arr.push_values(&[c.x as f32, c.y as f32, c.z as f32]);
    }
}

fn push_bary_vec_f64(
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    bary: SplitBary,
    item_size: usize,
    arr: &mut super::attribute_data::TypedArray,
    invert: bool,
) {
    let mut out0 = vec![0.0; item_size];
    let mut out1 = vec![0.0; item_size];
    let mut out2 = vec![0.0; item_size];
    for i in 0..item_size {
        let a0 = v0[i] as f64;
        let a1 = v1[i] as f64;
        let a2 = v2[i] as f64;
        out0[i] = (a0 * bary.0[0] + a1 * bary.0[1] + a2 * bary.0[2]) as f32;
        out1[i] = (a0 * bary.1[0] + a1 * bary.1[1] + a2 * bary.1[2]) as f32;
        out2[i] = (a0 * bary.2[0] + a1 * bary.2[1] + a2 * bary.2[2]) as f32;
    }
    if invert {
        arr.push_values(&out0);
        arr.push_values(&out2);
        arr.push_values(&out1);
    } else {
        arr.push_values(&out0);
        arr.push_values(&out1);
        arr.push_values(&out2);
    }
}

pub fn assign_buffer_data(geometry: &mut BufferGeometry, data: &TypedAttributeData) {
    let vert_count = data.get_total_length("position") / 3;
    let mut positions = vec![0.0f32; vert_count * 3];
    let mut offset = 0usize;
    for gi in 0..data.group_count {
        if let Some(arr) = data.group_attributes[gi].get("position") {
            let len = arr.data.len();
            positions[offset..offset + len].copy_from_slice(&arr.data);
            offset += len;
        }
    }
    geometry.set_attribute("position", BufferAttribute::new(positions, 3));
    if let Some(normals_src) = data.group_attributes[0].get("normal") {
        if !normals_src.data.is_empty() {
            let mut normals = vec![0.0f32; vert_count * 3];
            let mut no = 0usize;
            for gi in 0..data.group_count {
                if let Some(arr) = data.group_attributes[gi].get("normal") {
                    let len = arr.data.len();
                    normals[no..no + len].copy_from_slice(&arr.data);
                    no += len;
                }
            }
            geometry.set_attribute("normal", BufferAttribute::new(normals, 3));
        }
    }
    geometry.index = None;
    geometry.dispose_bounds_tree();
}
