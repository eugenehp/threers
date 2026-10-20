//! Stanford PLY — three.js r165 `PLYExporter.parse`, ASCII and binary.

use super::{
    apply_matrix4, attr3, count, linear_to_srgb, normal_matrix, transform_normal, ExportKind,
    ExportObject, NumberFormat,
};

/// three.js's `PLYExporter` options, plus the threers extensions.
#[derive(Debug, Clone, Default)]
pub struct PlyOptions {
    pub binary: bool,
    /// Binary byte order. three.js defaults to big endian, and so does this.
    pub little_endian: bool,
    /// `excludeAttributes`: any of `"index"`, `"normal"`, `"uv"`, `"color"`.
    pub exclude_attributes: Vec<String>,
    /// threers extension: one face per polygon the geometry remembers.
    pub polygons: bool,
    pub numbers: NumberFormat,
    /// A `comment` line after `format`, when set.
    pub comment: Option<String>,
}

/// The objects' meshes and point clouds as one PLY, or `None` where three.js
/// returns `null`: faces asked for over a vertex count that is not a whole
/// number of triangles.
pub fn write_ply(objects: &[ExportObject], options: &PlyOptions) -> Option<Vec<u8>> {
    let excluded = |name: &str| options.exclude_attributes.iter().any(|a| a == name);
    let mut include_indices = true;
    let mut include_normals = false;
    let mut include_colors = false;
    let mut include_uvs = false;

    let used: Vec<&ExportObject> = objects
        .iter()
        .filter(|o| matches!(o.kind, ExportKind::Mesh | ExportKind::Points))
        .filter(|o| o.geometry.get_attribute("position").is_some())
        .collect();

    // Per object, the faces it writes: polygon loops or triangles.
    let mut vertex_count = 0usize;
    // In triangles, as three.js counts it, to detect a fractional count.
    let mut tri_index_count = 0usize;
    let mut face_lists: Vec<Vec<Vec<u32>>> = Vec::with_capacity(used.len());
    for o in &used {
        let g = &*o.geometry;
        let n = count(g, "position").unwrap_or(0);
        vertex_count += n;
        include_normals |= g.get_attribute("normal").is_some();
        include_colors |= g.get_attribute("color").is_some();
        let mut faces = Vec::new();
        if o.kind == ExportKind::Points {
            include_indices = false;
        } else {
            include_uvs |= g.get_attribute("uv").is_some();
            tri_index_count += g.index.as_ref().map(|i| i.len()).unwrap_or(n);
            if options.polygons && g.polygon_counts().is_some() {
                faces = g.faces().collect();
            } else if let Some(index) = &g.index {
                faces = index.chunks_exact(3).map(|t| t.to_vec()).collect();
            } else {
                faces = (0..n as u32 / 3)
                    .map(|t| vec![t * 3, t * 3 + 1, t * 3 + 2])
                    .collect();
            }
        }
        face_lists.push(faces);
    }
    include_indices &= !excluded("index");
    include_normals &= !excluded("normal");
    include_colors &= !excluded("color");
    include_uvs &= !excluded("uv");
    if include_indices && !tri_index_count.is_multiple_of(3) {
        return None;
    }
    let face_count: usize = face_lists.iter().map(Vec::len).sum();
    let widest = face_lists
        .iter()
        .flatten()
        .map(Vec::len)
        .max()
        .unwrap_or(3);
    // three.js only writes triangles; a polygon past 255 corners needs a
    // wider count than `uchar`.
    let count_type = if widest > 255 { "int" } else { "uchar" };

    let format = if options.binary {
        if options.little_endian {
            "binary_little_endian"
        } else {
            "binary_big_endian"
        }
    } else {
        "ascii"
    };
    let mut header = format!("ply\nformat {format} 1.0\n");
    if let Some(c) = &options.comment {
        header.push_str(&format!("comment {c}\n"));
    }
    header.push_str(&format!(
        "element vertex {vertex_count}\nproperty float x\nproperty float y\nproperty float z\n"
    ));
    if include_normals {
        header.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    if include_uvs {
        header.push_str("property float s\nproperty float t\n");
    }
    if include_colors {
        header.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    if include_indices {
        header.push_str(&format!(
            "element face {face_count}\nproperty list {count_type} int vertex_index\n"
        ));
    }
    header.push_str("end_header\n");

    // `Math.floor(LinearToSRGB(c) * 255)` into a byte, as `setUint8` stores it
    // (modulo 256; NaN is 0).
    let colour_byte = |c: f64| -> u8 {
        let v = (linear_to_srgb(c) * 255.0).floor();
        if v.is_finite() {
            (v as i64).rem_euclid(256) as u8
        } else {
            0
        }
    };

    if options.binary {
        let mut out = header.into_bytes();
        let le = options.little_endian;
        let f32b = |out: &mut Vec<u8>, v: f64| {
            let v = v as f32;
            out.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
        };
        let u32b = |out: &mut Vec<u8>, v: u32| {
            out.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
        };
        let mut faces_out = Vec::new();
        let mut written = 0u32;
        for (o, faces) in used.iter().zip(&face_lists) {
            let g = &*o.geometry;
            let n = count(g, "position").unwrap_or(0);
            let nm = normal_matrix(&o.matrix_world);
            for i in 0..n {
                for c in apply_matrix4(&o.matrix_world, attr3(g, "position", i)) {
                    f32b(&mut out, c);
                }
                if include_normals {
                    let v = if g.get_attribute("normal").is_some() {
                        transform_normal(&nm, attr3(g, "normal", i))
                    } else {
                        [0.0; 3]
                    };
                    for c in v {
                        f32b(&mut out, c);
                    }
                }
                if include_uvs {
                    let v = if g.get_attribute("uv").is_some() {
                        attr3(g, "uv", i)
                    } else {
                        [0.0; 3]
                    };
                    f32b(&mut out, v[0]);
                    f32b(&mut out, v[1]);
                }
                if include_colors {
                    if g.get_attribute("color").is_some() {
                        for c in attr3(g, "color", i) {
                            out.push(colour_byte(c));
                        }
                    } else {
                        out.extend_from_slice(&[255, 255, 255]);
                    }
                }
            }
            if include_indices {
                for face in faces {
                    if count_type == "int" {
                        u32b(&mut faces_out, face.len() as u32);
                    } else {
                        faces_out.push(face.len() as u8);
                    }
                    for v in face {
                        u32b(&mut faces_out, v + written);
                    }
                }
            }
            written += n as u32;
        }
        out.extend_from_slice(&faces_out);
        Some(out)
    } else {
        let mut vertex_list = String::new();
        let mut face_list = String::new();
        let num = |out: &mut String, v: f64| options.numbers.write(out, v);
        let mut written = 0usize;
        for (o, faces) in used.iter().zip(&face_lists) {
            let g = &*o.geometry;
            let n = count(g, "position").unwrap_or(0);
            let nm = normal_matrix(&o.matrix_world);
            for i in 0..n {
                let v = apply_matrix4(&o.matrix_world, attr3(g, "position", i));
                num(&mut vertex_list, v[0]);
                for c in &v[1..] {
                    vertex_list.push(' ');
                    num(&mut vertex_list, *c);
                }
                if include_normals {
                    if g.get_attribute("normal").is_some() {
                        for c in transform_normal(&nm, attr3(g, "normal", i)) {
                            vertex_list.push(' ');
                            num(&mut vertex_list, c);
                        }
                    } else {
                        vertex_list.push_str(" 0 0 0");
                    }
                }
                if include_uvs {
                    if g.get_attribute("uv").is_some() {
                        let uv = attr3(g, "uv", i);
                        for c in &uv[..2] {
                            vertex_list.push(' ');
                            num(&mut vertex_list, *c);
                        }
                    } else {
                        vertex_list.push_str(" 0 0");
                    }
                }
                if include_colors {
                    if g.get_attribute("color").is_some() {
                        for c in attr3(g, "color", i) {
                            let v = (linear_to_srgb(c) * 255.0).floor();
                            vertex_list.push(' ');
                            vertex_list.push_str(&super::js_number(v));
                        }
                    } else {
                        vertex_list.push_str(" 255 255 255");
                    }
                }
                vertex_list.push('\n');
            }
            if include_indices {
                for face in faces {
                    face_list.push_str(&face.len().to_string());
                    for v in face {
                        face_list.push(' ');
                        face_list.push_str(&(*v as usize + written).to_string());
                    }
                    face_list.push('\n');
                }
            }
            written += n;
        }
        let mut out = header;
        out.push_str(&vertex_list);
        if include_indices {
            out.push_str(&face_list);
        }
        out.push('\n');
        Some(out.into_bytes())
    }
}
