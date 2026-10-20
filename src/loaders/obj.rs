use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry};

/// Minimal Wavefront OBJ parser. Reads `v` (position), `vn` (normal),
/// `vt` (uv), and `f` (face) records. Negative indices are accepted (offset
/// from end). Quad and polygon faces are triangulated by fan from vertex 0,
/// and which triangles were one face is kept in
/// [`BufferGeometry::polygons`].
///
/// Face-corners that reference the same `v/vt/vn` index triple are **welded**
/// into a single output vertex (indexed), so adjacent faces share vertices.
/// This is what makes `compute_vertex_normals` produce *smooth* shading on
/// normal-less meshes — an unwelded triangle soup can only be flat-shaded.
pub struct ObjLoader;

impl ObjLoader {
    pub fn parse(src: &str) -> BufferGeometry {
        Self::parse_with_materials(src).0
    }

    /// [`parse`](Self::parse) plus the `usemtl` names, one per material run,
    /// in order — three.js `OBJLoader`'s material bookkeeping. When the file
    /// uses more than one material run the geometry gets one group per run
    /// (`add_group(start, count, i)`, counted in index entries), and the names
    /// are the material array those groups index; with one run (or none) there
    /// are no groups. As in three.js, a `usemtl` that no face follows is
    /// dropped, faces before the first `usemtl` belong to the first run, and a
    /// name used twice is two runs (two array entries).
    pub fn parse_with_materials(src: &str) -> (BufferGeometry, Vec<String>) {
        // (name, start, count) in index entries; count -1 = still open.
        let mut runs: Vec<(String, i64, i64)> = Vec::new();
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut uvs: Vec<f32> = Vec::new();

        // Welded per-vertex output, keyed by the (v, vt, vn) index triple.
        let mut out_pos: Vec<f32> = Vec::new();
        let mut out_nrm: Vec<f32> = Vec::new();
        let mut out_uv: Vec<f32> = Vec::new();
        let mut out_idx: Vec<u32> = Vec::new();
        // Triangles per face, in face order.
        let mut out_polygons: Vec<u32> = Vec::new();
        let mut vertex_map: HashMap<(i32, i32, i32), u32> = HashMap::new();

        for raw_line in src.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(tag) = parts.next() else {
                continue;
            };
            match tag {
                "usemtl" => {
                    let name = parts.collect::<Vec<_>>().join(" ");
                    let at = out_idx.len() as i64;
                    let mut start = 0;
                    if let Some(last) = runs.last_mut() {
                        if last.2 < 0 {
                            last.2 = at - last.1;
                        }
                        start = last.1 + last.2;
                        if last.2 <= 0 {
                            runs.pop();
                        }
                    }
                    runs.push((name, start, -1));
                }
                "v" => {
                    let xs: Vec<f32> = parts.filter_map(|s| s.parse().ok()).collect();
                    if xs.len() >= 3 {
                        positions.extend_from_slice(&xs[..3]);
                    }
                }
                "vn" => {
                    let xs: Vec<f32> = parts.filter_map(|s| s.parse().ok()).collect();
                    if xs.len() >= 3 {
                        normals.extend_from_slice(&xs[..3]);
                    }
                }
                "vt" => {
                    let xs: Vec<f32> = parts.filter_map(|s| s.parse().ok()).collect();
                    if xs.len() >= 2 {
                        uvs.extend_from_slice(&xs[..2]);
                    }
                }
                "f" => {
                    // Each token is `v[/vt[/vn]]` (1-based, possibly negative).
                    let tokens: Vec<&str> = parts.collect();
                    let mut verts: Vec<(i32, i32, i32)> = Vec::with_capacity(tokens.len());
                    for t in tokens {
                        let pieces: Vec<&str> = t.split('/').collect();
                        let pi: i32 = pieces.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                        let ti: i32 = pieces
                            .get(1)
                            .filter(|s| !s.is_empty())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                        let ni: i32 = pieces.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                        verts.push((pi, ti, ni));
                    }
                    // Fan-triangulate, welding shared vertices.
                    if verts.len() >= 3 {
                        out_polygons.push(verts.len() as u32 - 2);
                    }
                    for i in 1..verts.len().saturating_sub(1) {
                        for v in [verts[0], verts[i], verts[i + 1]] {
                            let index = *vertex_map.entry(v).or_insert_with(|| {
                                let new_index = (out_pos.len() / 3) as u32;
                                push_vertex(
                                    &positions,
                                    &uvs,
                                    &normals,
                                    v,
                                    &mut out_pos,
                                    &mut out_uv,
                                    &mut out_nrm,
                                );
                                new_index
                            });
                            out_idx.push(index);
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(last) = runs.last_mut() {
            if last.2 < 0 {
                last.2 = out_idx.len() as i64 - last.1;
            }
        }
        if runs.len() > 1 {
            runs.retain(|r| r.2 > 0);
        }
        let mut g = BufferGeometry::new();
        if !out_pos.is_empty() {
            g.set_attribute("position", BufferAttribute::new(out_pos, 3));
        }
        // Only expose normals the file actually provided (`out_nrm` is always
        // padded with `[0,0,1]` defaults, so guard on the source `vn` records).
        // Downstream can then detect the absence and `compute_vertex_normals`,
        // rather than clobbering an artist's smooth normals.
        if !normals.is_empty() {
            g.set_attribute("normal", BufferAttribute::new(out_nrm, 3));
        }
        if !out_uv.is_empty() {
            g.set_attribute("uv", BufferAttribute::new(out_uv, 2));
        }
        if !out_idx.is_empty() {
            g.set_index(out_idx);
            g.set_polygons(out_polygons);
        }
        if runs.len() > 1 {
            for (i, r) in runs.iter().enumerate() {
                g.add_group(r.1 as u32, r.2 as u32, i as u32);
            }
        }
        (g, runs.into_iter().map(|r| r.0).collect())
    }

    /// The whole file as one mesh, as three.js `OBJLoader` builds it without
    /// a material library: a `MeshPhongMaterial` per `usemtl` run — a material
    /// array over the geometry's groups when there is more than one run.
    pub fn parse_mesh(src: &str) -> crate::core::Mesh {
        use crate::materials::{Material, PhongMaterial};
        use std::sync::Arc;
        let (g, names) = Self::parse_with_materials(src);
        let phong = || Arc::new(Material::Phong(PhongMaterial::default()));
        if names.len() > 1 {
            crate::core::Mesh::with_materials(Arc::new(g), names.iter().map(|_| phong()).collect())
        } else {
            crate::core::Mesh::from_arc(Arc::new(g), phong())
        }
    }
}

fn push_vertex(
    positions: &[f32],
    uvs: &[f32],
    normals: &[f32],
    (pi, ti, ni): (i32, i32, i32),
    out_pos: &mut Vec<f32>,
    out_uv: &mut Vec<f32>,
    out_nrm: &mut Vec<f32>,
) {
    let resolve = |idx: i32, len: usize| -> Option<usize> {
        if idx == 0 {
            return None;
        }
        if idx > 0 {
            Some((idx as usize).saturating_sub(1).min(len.saturating_sub(1)))
        } else {
            let from_end = (-idx) as usize;
            len.checked_sub(from_end)
        }
    };
    if let Some(i) = resolve(pi, positions.len() / 3) {
        out_pos.extend_from_slice(&positions[i * 3..i * 3 + 3]);
    } else {
        out_pos.extend_from_slice(&[0.0, 0.0, 0.0]);
    }
    if let Some(i) = resolve(ti, uvs.len() / 2) {
        out_uv.extend_from_slice(&uvs[i * 2..i * 2 + 2]);
    } else {
        out_uv.extend_from_slice(&[0.0, 0.0]);
    }
    if let Some(i) = resolve(ni, normals.len() / 3) {
        out_nrm.extend_from_slice(&normals[i * 3..i * 3 + 3]);
    } else {
        out_nrm.extend_from_slice(&[0.0, 0.0, 1.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// three.js r165 `OBJLoader` on this file gives groups
    /// `[{0,12,0},{12,3,1},{15,3,2}]` and materials `["red","green","red"]`
    /// (`MeshPhongMaterial`s): the leading face joins the first run, `blue` has
    /// no faces and is dropped, and `red` used twice is two entries.
    #[test]
    fn usemtl_runs_become_groups_as_three() {
        let src = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 2 1 0\n\
f 1 2 3\nusemtl red\nf 1 3 4\nf 2 5 6 3\nusemtl blue\nusemtl green\n\
f 1 2 4\nusemtl red\nf 5 6 3\n";
        let (g, names) = ObjLoader::parse_with_materials(src);
        assert_eq!(names, ["red", "green", "red"]);
        let groups: Vec<(u32, u32, u32)> =
            g.groups.iter().map(|g| (g.start, g.count, g.material_index)).collect();
        assert_eq!(groups, [(0, 12, 0), (12, 3, 1), (15, 3, 2)]);
        let mesh = ObjLoader::parse_mesh(src);
        assert_eq!(mesh.materials.len(), 3);
        assert!(matches!(&*mesh.materials[0], crate::materials::Material::Phong(_)));
        // One material: no groups, no array.
        let (g1, n1) = ObjLoader::parse_with_materials("v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl a\nf 1 2 3\n");
        assert!(g1.groups.is_empty());
        assert_eq!(n1, ["a"]);
        assert!(!ObjLoader::parse_mesh("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").is_multi_material());
    }

    #[test]
    fn a_quad_is_one_polygon_of_two_triangles() {
        let g = ObjLoader::parse("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
        assert_eq!(g.triangle_count(), 2);
        assert_eq!(g.polygon_counts(), Some(&[2u32][..]));
        assert_eq!(g.polygon_loop(0..2), Some(vec![0, 1, 2, 3]));
        assert_eq!(g.polygon_edges().len(), 4, "the diagonal is not an edge");
    }

    #[test]
    fn mixed_faces_keep_their_corner_counts() {
        let src = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0.5 0
v 2.5 1.5 0
v 1.8 2.2 0
v 1 2 0
f 1 2 3 4
f 2 5 3
f 3 5 6 7 8
";
        let g = ObjLoader::parse(src);
        assert_eq!(g.polygon_counts(), Some(&[2u32, 1, 3][..]));
        let corners: Vec<usize> = g
            .polygon_ranges()
            .map(|r| g.polygon_loop(r).unwrap().len())
            .collect();
        assert_eq!(corners, [4, 3, 5]);
        // 4 + 3 + 5 sides, two of them shared: 2–3 and 3–5.
        assert_eq!(g.polygon_edges().len(), 10);
    }
}
