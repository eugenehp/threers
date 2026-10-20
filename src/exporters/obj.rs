//! Wavefront OBJ — three.js r165 `OBJExporter.parse`.

use super::{
    apply_matrix4, attr3, count, linear_to_srgb, normal_matrix, transform_normal, ExportKind,
    ExportObject, NumberFormat,
};

/// What [`write_obj`] writes beyond three.js's defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjOptions {
    /// threers extension: one `f` per polygon the geometry remembers
    /// ([`crate::BufferGeometry::faces`]) rather than one per triangle.
    pub polygons: bool,
    pub numbers: NumberFormat,
}

/// The objects as OBJ text, in order: `o` per object, `usemtl` when the
/// material is named, world-space `v`, `vt`, world-space normalised `vn`, and
/// `f` (meshes), `l` (lines) or `p` (points), indices running on across
/// objects.
pub fn write_obj(objects: &[ExportObject], options: &ObjOptions) -> String {
    let mut out = String::new();
    let mut index_vertex = 0usize;
    let mut index_uv = 0usize;
    let mut index_normal = 0usize;
    let num = |out: &mut String, v: f64| options.numbers.write(out, v);

    for o in objects {
        let g = &*o.geometry;
        let vertices = count(g, "position");
        out.push_str("o ");
        out.push_str(&o.name);
        out.push('\n');
        match o.kind {
            ExportKind::Mesh => {
                if !o.material_name.is_empty() {
                    out.push_str("usemtl ");
                    out.push_str(&o.material_name);
                    out.push('\n');
                }
                let mut nb_vertex = 0;
                if let Some(n) = vertices {
                    for i in 0..n {
                        let v = apply_matrix4(&o.matrix_world, attr3(g, "position", i));
                        out.push('v');
                        for c in v {
                            out.push(' ');
                            num(&mut out, c);
                        }
                        out.push('\n');
                        nb_vertex += 1;
                    }
                }
                let uvs = count(g, "uv");
                let mut nb_uv = 0;
                if let Some(n) = uvs {
                    for i in 0..n {
                        let uv = attr3(g, "uv", i);
                        out.push_str("vt ");
                        num(&mut out, uv[0]);
                        out.push(' ');
                        num(&mut out, uv[1]);
                        out.push('\n');
                        nb_uv += 1;
                    }
                }
                let normals = count(g, "normal");
                let mut nb_normal = 0;
                if let Some(n) = normals {
                    let nm = normal_matrix(&o.matrix_world);
                    for i in 0..n {
                        let v = transform_normal(&nm, attr3(g, "normal", i));
                        out.push_str("vn");
                        for c in v {
                            out.push(' ');
                            num(&mut out, c);
                        }
                        out.push('\n');
                        nb_normal += 1;
                    }
                }
                let corner = |out: &mut String, j: usize| {
                    out.push_str(&(index_vertex + j).to_string());
                    if normals.is_some() || uvs.is_some() {
                        out.push('/');
                        if uvs.is_some() {
                            out.push_str(&(index_uv + j).to_string());
                        }
                        if normals.is_some() {
                            out.push('/');
                            out.push_str(&(index_normal + j).to_string());
                        }
                    }
                };
                let polygon_faces = options
                    .polygons
                    .then(|| g.polygon_counts().map(|_| g.faces().collect::<Vec<_>>()))
                    .flatten();
                if let Some(faces) = polygon_faces {
                    for face in faces {
                        out.push('f');
                        for v in face {
                            out.push(' ');
                            corner(&mut out, v as usize + 1);
                        }
                        out.push('\n');
                    }
                } else if let Some(index) = &g.index {
                    for tri in index.chunks(3) {
                        out.push('f');
                        for m in 0..3 {
                            out.push(' ');
                            // Past a short last triangle three.js reads
                            // `undefined + 1`, which is NaN.
                            match tri.get(m) {
                                Some(v) => corner(&mut out, *v as usize + 1),
                                None => out.push_str("NaN"),
                            }
                        }
                        out.push('\n');
                    }
                } else if let Some(n) = vertices {
                    let mut i = 0;
                    while i < n {
                        out.push('f');
                        for m in 0..3 {
                            out.push(' ');
                            corner(&mut out, i + m + 1);
                        }
                        out.push('\n');
                        i += 3;
                    }
                }
                index_vertex += nb_vertex;
                index_uv += nb_uv;
                index_normal += nb_normal;
            }
            ExportKind::Line | ExportKind::LineSegments | ExportKind::LineLoop => {
                let n = vertices.unwrap_or(0);
                for i in 0..n {
                    let v = apply_matrix4(&o.matrix_world, attr3(g, "position", i));
                    out.push('v');
                    for c in v {
                        out.push(' ');
                        num(&mut out, c);
                    }
                    out.push('\n');
                }
                if o.kind == ExportKind::Line {
                    out.push_str("l ");
                    for j in 1..=n {
                        out.push_str(&(index_vertex + j).to_string());
                        out.push(' ');
                    }
                    out.push('\n');
                }
                if o.kind == ExportKind::LineSegments {
                    let mut j = 1;
                    while j < n {
                        out.push_str(&format!("l {} {}\n", index_vertex + j, index_vertex + j + 1));
                        j += 2;
                    }
                }
                index_vertex += n;
            }
            ExportKind::Points => {
                let colors = g.get_attribute("color").is_some();
                if let Some(n) = vertices {
                    for i in 0..n {
                        let v = apply_matrix4(&o.matrix_world, attr3(g, "position", i));
                        out.push('v');
                        for c in v {
                            out.push(' ');
                            num(&mut out, c);
                        }
                        if colors {
                            for c in attr3(g, "color", i) {
                                out.push(' ');
                                num(&mut out, linear_to_srgb(c));
                            }
                        }
                        out.push('\n');
                    }
                    out.push_str("p ");
                    for j in 1..=n {
                        out.push_str(&(index_vertex + j).to_string());
                        out.push(' ');
                    }
                    out.push('\n');
                    index_vertex += n;
                }
            }
        }
    }
    out
}
