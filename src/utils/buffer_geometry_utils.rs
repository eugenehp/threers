use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

/// Compute per-vertex normals by averaging incident face normals. Mirrors
/// three.js's `BufferGeometry.computeVertexNormals()`.
pub fn compute_vertex_normals(geom: &mut BufferGeometry) {
    let pos = match geom.get_attribute("position") {
        Some(p) => p.array.clone(),
        None => return,
    };
    let vert_count = pos.len() / 3;
    let mut normals = vec![0.0f32; vert_count * 3];

    let add = |normals: &mut Vec<f32>, i: usize, n: Vector3| {
        normals[i * 3] += n.x;
        normals[i * 3 + 1] += n.y;
        normals[i * 3 + 2] += n.z;
    };

    let read = |i: usize| -> Vector3 { Vector3::new(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]) };

    if let Some(idx) = geom.index.clone() {
        for tri in idx.chunks_exact(3) {
            let a = read(tri[0] as usize);
            let b = read(tri[1] as usize);
            let c = read(tri[2] as usize);
            let face_n = (b - a).cross(c - a);
            add(&mut normals, tri[0] as usize, face_n);
            add(&mut normals, tri[1] as usize, face_n);
            add(&mut normals, tri[2] as usize, face_n);
        }
    } else {
        for i in 0..vert_count / 3 {
            let a = read(i * 3);
            let b = read(i * 3 + 1);
            let c = read(i * 3 + 2);
            let face_n = (b - a).cross(c - a);
            add(&mut normals, i * 3, face_n);
            add(&mut normals, i * 3 + 1, face_n);
            add(&mut normals, i * 3 + 2, face_n);
        }
    }

    for n in normals.chunks_exact_mut(3) {
        let v = Vector3::new(n[0], n[1], n[2]).normalize();
        n[0] = v.x;
        n[1] = v.y;
        n[2] = v.z;
    }
    geom.set_attribute("normal", BufferAttribute::new(normals, 3));
}

/// Per-vertex tangent computation (MikkTSpace-flavoured but simpler). Output
/// is `vec4` with w = bitangent sign. Caller must ensure `uv` and `normal`
/// attributes are populated first.
pub fn compute_tangents(geom: &mut BufferGeometry) {
    let pos = match geom.get_attribute("position") {
        Some(a) => a.array.clone(),
        None => return,
    };
    let uvs = match geom.get_attribute("uv") {
        Some(a) => a.array.clone(),
        None => return,
    };
    let nor = match geom.get_attribute("normal") {
        Some(a) => a.array.clone(),
        None => return,
    };
    let count = pos.len() / 3;

    let mut tan1 = vec![Vector3::ZERO; count];
    let mut tan2 = vec![Vector3::ZERO; count];

    let mut process_tri = |a: usize, b: usize, c: usize| {
        let v0 = Vector3::new(pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2]);
        let v1 = Vector3::new(pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]);
        let v2 = Vector3::new(pos[c * 3], pos[c * 3 + 1], pos[c * 3 + 2]);
        let (u0, w0) = (uvs[a * 2], uvs[a * 2 + 1]);
        let (u1, w1) = (uvs[b * 2], uvs[b * 2 + 1]);
        let (u2, w2) = (uvs[c * 2], uvs[c * 2 + 1]);
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let du1 = u1 - u0;
        let dw1 = w1 - w0;
        let du2 = u2 - u0;
        let dw2 = w2 - w0;
        let denom = du1 * dw2 - du2 * dw1;
        if denom == 0.0 {
            return;
        }
        let r = 1.0 / denom;
        let sdir = Vector3::new(
            (dw2 * e1.x - dw1 * e2.x) * r,
            (dw2 * e1.y - dw1 * e2.y) * r,
            (dw2 * e1.z - dw1 * e2.z) * r,
        );
        let tdir = Vector3::new(
            (du1 * e2.x - du2 * e1.x) * r,
            (du1 * e2.y - du2 * e1.y) * r,
            (du1 * e2.z - du2 * e1.z) * r,
        );
        tan1[a] = tan1[a] + sdir;
        tan1[b] = tan1[b] + sdir;
        tan1[c] = tan1[c] + sdir;
        tan2[a] = tan2[a] + tdir;
        tan2[b] = tan2[b] + tdir;
        tan2[c] = tan2[c] + tdir;
    };

    if let Some(idx) = geom.index.clone() {
        for tri in idx.chunks_exact(3) {
            process_tri(tri[0] as usize, tri[1] as usize, tri[2] as usize);
        }
    } else {
        for i in 0..count / 3 {
            process_tri(i * 3, i * 3 + 1, i * 3 + 2);
        }
    }

    let mut tangents = Vec::with_capacity(count * 4);
    for i in 0..count {
        let n = Vector3::new(nor[i * 3], nor[i * 3 + 1], nor[i * 3 + 2]);
        let t = tan1[i];
        let proj = (t - n * n.dot(t)).normalize();
        let w = if n.cross(t).dot(tan2[i]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        tangents.extend_from_slice(&[proj.x, proj.y, proj.z, w]);
    }
    geom.set_attribute("tangent", BufferAttribute::new(tangents, 4));
}

/// Merge multiple geometries into one. All sources must share the same set of
/// attribute names and item sizes. Index buffers are concatenated and offset.
/// Geometries with no index get a synthetic one. Mirrors three.js's
/// `BufferGeometryUtils.mergeGeometries`. Polygon groupings are carried
/// across; see [`BufferGeometry::polygons`].
pub fn merge_geometries(geoms: &[BufferGeometry]) -> Option<BufferGeometry> {
    if geoms.is_empty() {
        return None;
    }
    let mut out = BufferGeometry::new();
    let attr_names: Vec<String> = geoms[0].attributes.keys().cloned().collect();
    let mut merged: std::collections::HashMap<String, (usize, Vec<f32>)> =
        std::collections::HashMap::new();
    for name in &attr_names {
        let item_size = geoms[0].get_attribute(name)?.item_size;
        merged.insert(name.clone(), (item_size, Vec::new()));
    }
    let mut indices: Vec<u32> = Vec::new();
    let mut vert_offset = 0u32;
    for g in geoms {
        for name in &attr_names {
            let attr = g.get_attribute(name)?;
            let (_, vec) = merged.get_mut(name)?;
            vec.extend_from_slice(&attr.array);
        }
        let vc = g
            .get_attribute("position")
            .map(|a| a.count() as u32)
            .unwrap_or(0);
        if let Some(idx) = &g.index {
            indices.extend(idx.iter().map(|i| i + vert_offset));
        } else {
            indices.extend((0..vc).map(|i| i + vert_offset));
        }
        vert_offset += vc;
    }
    for (name, (item_size, data)) in merged {
        out.set_attribute(name, BufferAttribute::new(data, item_size));
    }
    out.set_index(indices);
    // Concatenation keeps every source's triangles in order, so polygon
    // groupings concatenate too. A source without one contributes a polygon
    // per triangle, which is what it already was.
    if geoms.iter().any(|g| g.polygon_counts().is_some()) {
        let polygons = geoms
            .iter()
            .flat_map(|g| g.polygon_ranges().map(|r| r.len() as u32))
            .collect();
        out.set_polygons(polygons);
    }
    Some(out)
}

/// Translate every vertex so the geometry's bounding-box centroid is at origin.
pub fn center(geom: &mut BufferGeometry) {
    let bb = geom.compute_bounding_box();
    let c = bb.center();
    if let Some(attr) = geom.attributes.get_mut("position") {
        for v in attr.array.chunks_exact_mut(3) {
            v[0] -= c.x;
            v[1] -= c.y;
            v[2] -= c.z;
        }
    }
    geom.bounding_box = None;
    geom.bounding_sphere = None;
}

/// Uniform scale.
pub fn scale(geom: &mut BufferGeometry, sx: f32, sy: f32, sz: f32) {
    if let Some(attr) = geom.attributes.get_mut("position") {
        for v in attr.array.chunks_exact_mut(3) {
            v[0] *= sx;
            v[1] *= sy;
            v[2] *= sz;
        }
    }
    geom.bounding_box = None;
    geom.bounding_sphere = None;
}
